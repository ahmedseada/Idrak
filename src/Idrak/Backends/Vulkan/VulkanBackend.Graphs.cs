// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using static Idrak.Backends.Vulkan.VulkanDriver;

namespace Idrak.Backends.Vulkan;

// Recorded graphs (ComputeGraph): between BeginCapture and EndCapture the commands the recording thread queues
// (dispatches with their pushed descriptors or descriptor sets, push constants, fills, copies, and the barriers between
// dependent commands) go into secondary command buffers instead of the queue's batches, and nothing runs. Replaying
// executes those command buffers from the current batch (one vkCmdExecuteCommands), so the host records nothing per
// replay. A recording longer than a batch (the measured commands per batch) continues in a further command buffer; a
// replay executes them all in order.
//
// A recorded step must stay valid when replayed later: what changes between replays lives in device memory (decoding
// positions, the sampler's step), and the storages it uses stay where they are. Storages the recording frees belong to the
// graph (they are not reused by other work while the graph exists, and go back to the pool when it is destroyed);
// allocations during the recording reuse those first, in recording order like the commands. The graph starts with a
// barrier (after whatever ran before it) and ends with one (before whatever follows, the host included), so the queue's
// barrier bookkeeping needs nothing per command of the graph.
//
// While recording, the backend's queue lock is held, so other threads' work waits instead of landing in the graph. An
// upload during the recording happens at once (as on CUDA): into mapped memory by the host, else through the staging
// buffer on the queue, outside the graph. Reading device memory during the recording fails (nothing recorded has run),
// which also turns a host fallback into a failed recording: ComputeGraph then re-runs the step instead.
internal sealed unsafe partial class VulkanBackend
{
    // The recording in progress (null when none).
    private Capture? _capture;

    // Graphs recorded and not yet destroyed.
    private int _liveGraphs;

    /// <summary>Graphs replayed (for tests and diagnostics).</summary>
    internal long Replays;

    /// <summary>Recorded graphs not yet destroyed (for tests and diagnostics).</summary>
    internal int LiveGraphs => Volatile.Read(ref _liveGraphs);

    /// <summary>Whether a graph is being recorded on this device now (for tests and diagnostics).</summary>
    internal bool Capturing => _capture is not null;

    private sealed class Capture(Batch batch, Batch normal, int thread)
    {
        // The command buffer being recorded and the descriptor pools of the graph, in the shape of a batch so dispatches
        // record into it as into any batch.
        public readonly Batch Batch = batch;
        public readonly List<IntPtr> Commands = [batch.Commands];
        public readonly HashSet<VulkanBlock> Blocks = [];
        public readonly Dictionary<int, Stack<VulkanBlock>> Free = [];
        public readonly int Thread = thread;
        public Batch Normal = normal;
        public int Total;
    }

    // A recorded graph: its command buffers, in order, the descriptor pools their sets come from, the blocks it uses
    // and the commands it holds.
    private sealed class Graph(IntPtr[] commands, List<DescriptorPool> pools, VulkanBlock[] blocks, int count)
    {
        public readonly IntPtr[] Commands = commands;
        public readonly List<DescriptorPool> Pools = pools;
        public readonly VulkanBlock[] Blocks = blocks;
        public readonly int Count = count;
    }

    // IDRAK_VULKAN_GRAPHS=0: no recorded graphs; callers run their steps directly (diagnostics).
    public override bool SupportsGraphs => Environment.GetEnvironmentVariable("IDRAK_VULKAN_GRAPHS") is not ("0" or "false");

    public override void BeginCapture()
    {
        _gate.Enter();                                                     // held until EndCapture or AbortCapture
        try
        {
            if (_capture is not null)
            {
                throw new InvalidOperationException("A graph is already being recorded on this device.");
            }

            Submit();                                                      // what was queued before runs on its own
            var batch = new Batch { Recording = true, Number = _recording };
            batch.Commands = BeginSecondary();
            lock (_pool)
            {
                _capture = new Capture(batch, _batch, Environment.CurrentManagedThreadId);
            }

            _batch = batch;
            GraphBarrier(batch.Commands, toHost: false);                   // after whatever ran before the replay
            _boundPipeline = 0;
        }
        catch
        {
            _gate.Exit();
            throw;
        }
    }

    public override (IntPtr Executable, IntPtr Graph, List<Storage> Owned) EndCapture()
    {
        if (_capture is null || !_gate.IsHeldByCurrentThread)
        {
            throw new InvalidOperationException("No graph is being recorded on this thread.");
        }

        try
        {
            var capture = _capture;
            var commands = capture.Batch.Commands;
            GraphBarrier(commands, toHost: true);                          // before whatever follows the replay
            Check(vkEndCommandBuffer(commands), nameof(vkEndCommandBuffer));
            capture.Total += capture.Batch.Count;
            var graph = new Graph([.. capture.Commands], capture.Batch.Pools, [.. capture.Blocks], capture.Total);
            var owned = StopCapture(capture);
            Interlocked.Increment(ref _liveGraphs);
            return (GCHandle.ToIntPtr(GCHandle.Alloc(graph)), IntPtr.Zero, owned);
        }
        catch
        {
            if (_capture is { } failed)
            {
                foreach (var storage in StopCapture(failed))
                {
                    storage.Release();
                }

                FreeGraph(failed.Commands, failed.Batch.Pools);
            }

            throw;
        }
        finally
        {
            _gate.Exit();
        }
    }

    public override List<Storage> AbortCapture()
    {
        if (_capture is not { } capture || !_gate.IsHeldByCurrentThread)
        {
            return [];
        }

        try
        {
            vkEndCommandBuffer(capture.Batch.Commands);                    // the result does not matter: it is freed
            var owned = StopCapture(capture);
            FreeGraph(capture.Commands, capture.Batch.Pools);
            return owned;
        }
        finally
        {
            _gate.Exit();
        }
    }

    public override void ReplayGraph(IntPtr executable)
    {
        var graph = GraphOf(executable);
        lock (_gate)
        {
            if (_capture is not null)
            {
                throw new InvalidOperationException("A graph cannot be replayed while another is being recorded.");
            }

            var commands = Record([], 0);                                  // begins the batch when needed
            fixed (IntPtr* recorded = graph.Commands)
            {
                vkCmdExecuteCommands(commands, (uint)graph.Commands.Length, recorded);
            }

            // The graph ends with a barrier, so nothing queued next depends on a command before it without one; the
            // host waits for this batch before touching any block the graph uses. Its command buffers leave no state.
            _span++;
            foreach (var block in graph.Blocks)
            {
                block.LastUse = _recording;
            }

            _boundPipeline = 0;
            Replays++;
            _batch.Count += graph.Count;
            if (_batch.Count >= _maxBatchCommands)
            {
                Submit();
            }
        }
    }

    public override void DestroyGraph(IntPtr executable, IntPtr graph)
    {
        var handle = GCHandle.FromIntPtr(executable);
        var recorded = (Graph)handle.Target!;
        lock (_gate)
        {
            SubmitAndWait();                                               // no batch that executes it is pending
            FreeGraph(recorded.Commands, recorded.Pools);
        }

        handle.Free();
        Interlocked.Decrement(ref _liveGraphs);
    }

    /// <summary>The commands and descriptor sets of a recorded graph (for tests and diagnostics).</summary>
    internal (int CommandBuffers, int Commands, int Blocks) DescribeGraph(IntPtr executable)
    {
        var graph = GraphOf(executable);
        return (graph.Commands.Length, graph.Count, graph.Blocks.Length);
    }

    private static Graph GraphOf(IntPtr executable) =>
        GCHandle.FromIntPtr(executable).Target as Graph ?? throw new ArgumentException("Not a graph recorded on a Vulkan device.", nameof(executable));

    // A secondary command buffer, begun for a recording that may be executed again while an earlier execution is pending.
    private IntPtr BeginSecondary()
    {
        var allocate = new VkCommandBufferAllocateInfo
        {
            SType = StructureCommandBufferAllocateInfo,
            CommandPool = _commandPool,
            Level = CommandBufferLevelSecondary,
            CommandBufferCount = 1,
        };
        IntPtr commands;
        Check(vkAllocateCommandBuffers(_device, &allocate, &commands), nameof(vkAllocateCommandBuffers));
        var inheritance = new VkCommandBufferInheritanceInfo { SType = StructureCommandBufferInheritanceInfo };
        var begin = new VkCommandBufferBeginInfo { SType = StructureCommandBufferBeginInfo, Flags = CommandBufferSimultaneousUse, InheritanceInfo = &inheritance };
        int result = vkBeginCommandBuffer(commands, &begin);
        if (result != Success)
        {
            vkFreeCommandBuffers(_device, _commandPool, 1, &commands);
            Check(result, nameof(vkBeginCommandBuffer));
        }

        return commands;
    }

    // A barrier between everything before it and everything after it (and the host, when `toHost`).
    private void GraphBarrier(IntPtr commands, bool toHost)
    {
        var barrier = new VkMemoryBarrier
        {
            SType = StructureMemoryBarrier,
            SourceAccessMask = AccessShaderWrite | AccessTransferWrite,
            DestinationAccessMask = AccessShaderRead | AccessShaderWrite | AccessTransferRead | AccessTransferWrite | (toHost ? AccessHostRead | AccessHostWrite : 0),
        };
        vkCmdPipelineBarrier(commands, StageComputeShader | StageTransfer, StageComputeShader | StageTransfer | (toHost ? StageHost : 0), 0, 1, &barrier, 0, null, 0, null);
        _span++;
        Barriers++;
    }

    // A recording grown to a batch's length continues in a new command buffer (called by Submit for the recording's
    // batch; other calls, such as a wait for a block before an upload, leave the recording as it is).
    private void ContinueCapture(Capture capture)
    {
        var batch = capture.Batch;
        if (batch.Count < _maxBatchCommands)
        {
            return;
        }

        Check(vkEndCommandBuffer(batch.Commands), nameof(vkEndCommandBuffer));
        capture.Total += batch.Count;
        batch.Count = 0;
        batch.Commands = BeginSecondary();
        capture.Commands.Add(batch.Commands);
        _boundPipeline = 0;
    }

    // The blocks a command of the recording uses become the graph's (called by Record for the recording's batch).
    private static void CaptureBlocks(Capture capture, ReadOnlySpan<VulkanBlock> blocks)
    {
        foreach (var block in blocks)
        {
            capture.Blocks.Add(block);
        }
    }

    // Ends the recording: the queue goes back to its own batch (begun with a barrier: nothing recorded ran, so commands
    // queued next may depend on commands queued before the recording) and the blocks the recording freed become
    // storages the graph owns (counted in use again; releasing them returns them to the pool).
    private List<Storage> StopCapture(Capture capture)
    {
        var owned = new List<Storage>();
        lock (_pool)
        {
            _capture = null;
            foreach (var (length, bucket) in capture.Free)
            {
                foreach (var block in bucket)
                {
                    _memory.Reused(BlockBytes(length));
                    owned.Add(new VulkanStorage(this, block, length));
                }
            }
        }

        _batch = capture.Normal;
        GraphBarrier(Record([], 0), toHost: false);
        _boundPipeline = 0;
        return owned;
    }

    // Frees a graph's command buffers and descriptor pools (nothing pending may execute them).
    private void FreeGraph(IReadOnlyList<IntPtr> commands, List<DescriptorPool> pools)
    {
        var buffers = commands.ToArray();
        fixed (IntPtr* b = buffers)
        {
            vkFreeCommandBuffers(_device, _commandPool, (uint)buffers.Length, b);
        }

        foreach (var pool in pools)
        {
            vkDestroyDescriptorPool(_device, pool.Handle, null);
        }

        pools.Clear();
    }

    // A block freed during this thread's recording, of `length` floats, for an allocation of the recording; null when
    // none (or no recording). Under the pool lock.
    private VulkanBlock? TakeCaptured(int length) =>
        _capture is { } capture && capture.Thread == Environment.CurrentManagedThreadId && capture.Free.TryGetValue(length, out var bucket)
        && bucket.Count > 0 ? bucket.Pop() : null;

    // Whether a block freed now goes to the recording's blocks (freed by the recording thread while it records). Under
    // the pool lock.
    private Stack<VulkanBlock>? CaptureFreeFor(int length)
    {
        if (_capture is not { } capture || capture.Thread != Environment.CurrentManagedThreadId)
        {
            return null;
        }

        if (!capture.Free.TryGetValue(length, out var bucket))
        {
            capture.Free[length] = bucket = new Stack<VulkanBlock>();
        }

        return bucket;
    }

    // Runs `copy` on the queue's own batch while a graph is recorded (an upload through the staging buffer happens at
    // once, outside the graph), with a barrier on each side of the switch: the span bookkeeping is shared by both.
    private void OutsideCapture(Capture capture, Action copy)
    {
        var recording = _batch;
        _batch = capture.Normal;
        try
        {
            GraphBarrier(Record([], 0), toHost: false);
            copy();
        }
        finally
        {
            capture.Normal = _batch;
            _batch = recording;
            GraphBarrier(recording.Commands, toHost: false);
            _boundPipeline = 0;
        }
    }
}
