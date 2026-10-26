// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Gpu.Abstractions;
using static Idrak.Gpu.Vulkan.VulkanDriver;

namespace Idrak.Gpu.Vulkan;

// The queue: commands (dispatches, copies, fills) are recorded into the current batch's command buffer; a batch is
// submitted when the host needs its results, before it waits, or when it grows long. Each batch has a fence, its command
// buffer and (without push descriptors) its descriptor pools, recycled when the fence signals. Batches are numbered: a
// block remembers the last batch that used it, and the host reads or writes a block only after that batch has finished.
//
// Barriers only where commands depend on each other: the commands recorded since the last barrier form a span (numbered,
// across batches); a block remembers the span in which a command last wrote it and last read it. A command gets a barrier
// first when it reads or writes a block written in the current span (read after write, write after write) or writes a
// block read in it (write after read); otherwise it may run alongside the commands before it.
internal sealed unsafe partial class VulkanBackend : IVulkanKernelDispatcher
{
    // Guards the queue, the batches and the pipelines (the pool has its own lock: Return runs on any thread).
    private readonly Lock _gate = new();

    private ulong _commandPool;
    private ulong _pipelineCache;
    private Batch _batch = null!;
    private readonly Queue<Batch> _inFlight = new();
    private readonly Stack<Batch> _free = new();

    // The number of the batch being recorded, and of the last batch known to have finished.
    private ulong _recording = 1;
    private ulong _completed;

    // Commands per batch before it is submitted on its own, and batches submitted but not waited for, at most: measured
    // when the backend starts (VulkanBackend.Tuning.cs; the measurement sets its own while it runs).
    private int _maxBatchCommands = 1;

    // IDRAK_VULKAN_BARRIERS=all: a barrier before every command, not only where a storage is shared (diagnostics: when
    // results change with it, a dependency was missed).
    private static readonly bool s_barrierEveryCommand = Environment.GetEnvironmentVariable("IDRAK_VULKAN_BARRIERS") == "all";

    // IDRAK_VULKAN_BARRIER_KERNELS=a,b: a barrier before every command of these kernels ("copy" and "fill" for copies
    // and fills); IDRAK_VULKAN_KERNEL_LOG=FILE: the names of the commands recorded, once each (diagnostics: which
    // command needs a dependency the bookkeeping missed).
    private static readonly HashSet<string>? s_barrierKernels = Environment.GetEnvironmentVariable("IDRAK_VULKAN_BARRIER_KERNELS") is { Length: > 0 } names
        ? new HashSet<string>(names.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries), StringComparer.Ordinal) : null;

    private static readonly string? s_kernelLog = Environment.GetEnvironmentVariable("IDRAK_VULKAN_KERNEL_LOG") is { Length: > 0 } log ? log : null;

    private static readonly HashSet<string> s_logged = new(StringComparer.Ordinal);

    // IDRAK_VULKAN_CHECK_NAN=1 (diagnostics, slow): after every kernel, wait for it and read back the storages it writes;
    // a NaN or +∞ there is reported on the error output with the kernel, its width and reductions, the storage, the
    // index and the push constants, the first CheckNanReports of them (the first names the kernel that made it: every
    // earlier kernel's outputs were finite). -∞ is left out (masks hold it); a storage of packed values (16-bit, 8-bit or
    // 4-bit words) may read as NaN, and its kernel names it.
    internal static bool CheckNan { get; set; } = Environment.GetEnvironmentVariable("IDRAK_VULKAN_CHECK_NAN") is "1" or "true";

    private const int CheckNanReports = 12;

    private static int s_nanReports;

    // Whether this device said the check is on (once, so a run with nothing to report shows it ran).
    private int _nanCheckShown;

    private void CheckWrittenFinite(VulkanKernel kernel, uint groupsX, uint groupsY, uint groupsZ, ReadOnlySpan<Storage> storages, ReadOnlySpan<byte> push)
    {
        if (Interlocked.Exchange(ref _nanCheckShown, 1) == 0)
        {
            Console.Error.WriteLine($"IDRAK_VULKAN_CHECK_NAN: {Name}: checking what every kernel writes (no recorded graphs)");
        }

        if (Capturing || Volatile.Read(ref s_nanReports) >= CheckNanReports)
        {
            return;
        }

        for (int i = 0; i < storages.Length; i++)
        {
            if ((kernel.Writes & (1UL << i)) == 0 || storages[i].Length <= 0)
            {
                continue;
            }

            var values = new float[storages[i].Length];
            Download(storages[i], values);
            int at = Array.FindIndex(values, v => float.IsNaN(v) || float.IsPositiveInfinity(v));
            if (at < 0)
            {
                continue;
            }

            int count = values.Count(v => float.IsNaN(v) || float.IsPositiveInfinity(v));
            var words = new int[push.Length / 4];
            for (int w = 0; w < words.Length; w++)
            {
                words[w] = BitConverter.ToInt32(push.Slice(4 * w, 4));
            }

            if (Interlocked.Increment(ref s_nanReports) <= CheckNanReports)
            {
                Console.Error.WriteLine(string.Create(System.Globalization.CultureInfo.InvariantCulture,
                    $"IDRAK_VULKAN_CHECK_NAN: {Name}: kernel '{kernel.Name}' (local size {kernel.LocalSizeX}, {(kernel.UsesSubgroups ? "subgroups" : "no subgroups")}, groups {groupsX} x {groupsY} x {groupsZ}) wrote {values[at]} at index {at} of storage {i} ({count} of {values.Length} values not finite); push constants as ints: [{string.Join(", ", words)}]"));
            }
        }
    }

    private static void LogCommand(string name)
    {
        if (s_kernelLog is not null && s_logged.Add(name))
        {
            File.AppendAllText(s_kernelLog, name + Environment.NewLine);
        }
    }
    private int _maxInFlight = 1;

    // A descriptor pool holds as many sets as a batch has commands, each as wide as a kernel may bind (the device's
    // maxPerStageDescriptorStorageBuffers, at most VulkanKernel.MaxBindings), so one pool always serves a whole batch.
    private DescriptorPool BatchPool()
    {
        uint sets = (uint)Math.Max(_maxBatchCommands, 1);
        uint wide = Math.Max(Math.Min(_physical.Properties.MaxPerStageDescriptorStorageBuffers, (uint)VulkanKernel.MaxBindings), 1);
        return NewPool(sets, (uint)Math.Min((ulong)sets * wide, uint.MaxValue));
    }

    // The span of commands since the last barrier (see the top of the file).
    private ulong _span = 1;

    // The pipeline bound in the command buffer being recorded (0: none yet).
    private ulong _boundPipeline;

    // The blocks of the command being recorded (reused: dispatches allocate nothing).
    private readonly VulkanBlock[] _commandBlocks = new VulkanBlock[VulkanKernel.MaxBindings];

    // The descriptors of the command being recorded, reused under _gate like _commandBlocks: a kernel binds up to
    // MaxBindings storages (a plug-in decides how many), too many for the stack on every dispatch.
    private readonly VkDescriptorBufferInfo[] _descriptorBuffers = new VkDescriptorBufferInfo[VulkanKernel.MaxBindings];
    private readonly VkWriteDescriptorSet[] _descriptorWrites = new VkWriteDescriptorSet[VulkanKernel.MaxBindings];

    private readonly Dictionary<VulkanKernel, Pipeline> _pipelines = [];
    private readonly Dictionary<(int Bindings, bool Pushed), ulong> _setLayouts = [];
    private readonly Dictionary<(int Bindings, int PushBytes, bool Pushed), ulong> _pipelineLayouts = [];

    /// <summary>Kernels dispatched (for tests and diagnostics).</summary>
    internal long Dispatches;

    /// <summary>Batches submitted to the queue (for tests and diagnostics).</summary>
    internal long Submissions;

    /// <summary>Barriers recorded between commands (for tests and diagnostics).</summary>
    internal long Barriers;

    /// <summary>When set, counts the dispatches by kernel name (for tests and diagnostics).</summary>
    internal System.Collections.Concurrent.ConcurrentDictionary<string, long>? DispatchesByKernel;

    private sealed class Batch
    {
        public IntPtr Commands;
        public ulong Fence;
        public readonly List<DescriptorPool> Pools = [];
        public int Pool;                                                       // the pool sets come from now
        public ulong Number;
        public bool Recording;
        public int Count;
        public string Last = "";                                               // the last command recorded by name (a lost device names it)
    }

    // What a batch's last command is called when it replayed a recorded graph.
    private const string ReplayedGraph = "a recorded graph";

    private sealed class DescriptorPool(ulong handle, uint sets, uint descriptors)
    {
        public readonly ulong Handle = handle;
        public readonly uint SetCapacity = sets;
        public readonly uint Capacity = descriptors;
        public uint Sets = sets;
        public uint Descriptors = descriptors;
    }

    private sealed record Pipeline(ulong Handle, ulong Layout, VulkanBackend Owner);

    // A plug-in's kernel (VulkanKernel.Dispatch): every storage bound whole.
    void IVulkanKernelDispatcher.Dispatch(VulkanKernel kernel, uint groupsX, uint groupsY, uint groupsZ, ReadOnlySpan<Storage> storages,
        ReadOnlySpan<byte> pushConstants) => Dispatch(kernel, groupsX, groupsY, groupsZ, storages, pushConstants);

    /// <summary>
    /// Queues <paramref name="kernel"/> over groupsX × groupsY × groupsZ workgroups: binding i of descriptor set 0 is
    /// <paramref name="storages"/>[i]; <paramref name="pushConstants"/> fills its push-constant block (exactly
    /// <see cref="VulkanKernel.PushConstantBytes"/> bytes). Dispatches run in the order queued wherever they share a
    /// storage one of them writes (<see cref="VulkanKernel.Writes"/>), and may overlap where they do not; reading a storage
    /// (<see cref="Backend.Download"/>) waits for the dispatches that use it. With <paramref name="windows"/>, binding i
    /// is the window of storage i starting at byte windows[i] (a multiple of minStorageBufferOffsetAlignment) and at most
    /// <see cref="MaxStorageBytes"/> long (VulkanBackend.LargeStorage.cs); without, every storage is bound whole.
    /// </summary>
    public void Dispatch(VulkanKernel kernel, uint groupsX, uint groupsY, uint groupsZ, ReadOnlySpan<Storage> storages, ReadOnlySpan<byte> pushConstants,
        ReadOnlySpan<long> windows = default)
    {
        ArgumentNullException.ThrowIfNull(kernel);
        if (storages.Length != kernel.Bindings)
        {
            throw new ArgumentException($"Vulkan kernel '{kernel.Name}' binds {kernel.Bindings} storages; {storages.Length} given.", nameof(storages));
        }

        if (pushConstants.Length != kernel.PushConstantBytes)
        {
            throw new ArgumentException($"Vulkan kernel '{kernel.Name}' takes {kernel.PushConstantBytes} bytes of push constants; {pushConstants.Length} given.", nameof(pushConstants));
        }

        ref readonly var p = ref _physical.Properties;
        if (groupsX > p.MaxComputeWorkGroupCountX || groupsY > p.MaxComputeWorkGroupCountY || groupsZ > p.MaxComputeWorkGroupCountZ)
        {
            throw new ArgumentOutOfRangeException(nameof(groupsX),
                $"Vulkan kernel '{kernel.Name}': {groupsX} × {groupsY} × {groupsZ} workgroups exceed {Name}'s {p.MaxComputeWorkGroupCountX} × {p.MaxComputeWorkGroupCountY} × {p.MaxComputeWorkGroupCountZ}.");
        }

        if (groupsX == 0 || groupsY == 0 || groupsZ == 0)
        {
            return;
        }

        int bindings = kernel.Bindings;
        lock (_gate)
        {
            var pipeline = PipelineOf(kernel);
            var blocks = _commandBlocks.AsSpan(0, bindings);
            fixed (VkDescriptorBufferInfo* buffers = _descriptorBuffers)
            fixed (VkWriteDescriptorSet* writes = _descriptorWrites)
            try
            {
                for (int i = 0; i < bindings; i++)
                {
                    blocks[i] = BlockOf(storages[i]);
                    long bytes = BlockBytes(storages[i].Length), offset = 0;
                    if (i < windows.Length)
                    {
                        offset = windows[i];
                        bytes = WindowBytes(bytes, offset);
                    }

                    if (bytes > MaxStorageBytes || bytes <= 0)
                    {
                        throw new ArgumentException(
                            $"Vulkan kernel '{kernel.Name}': storage {i} holds {bytes:N0} bytes, more than {Name} binds ({MaxStorageBytes:N0}); use the host fallback for it.", nameof(storages));
                    }

                    buffers[i] = new VkDescriptorBufferInfo { Buffer = blocks[i].Buffer, Offset = (ulong)offset, Range = (ulong)bytes };
                    writes[i] = new VkWriteDescriptorSet
                    {
                        SType = StructureWriteDescriptorSet,
                        DestinationBinding = (uint)i,
                        DescriptorCount = 1,
                        DescriptorType = DescriptorStorageBuffer,
                        BufferInfo = &buffers[i],
                    };
                }

                var commands = Record(blocks, kernel.Writes, kernel.Name);
                if (_boundPipeline != pipeline.Handle)
                {
                    vkCmdBindPipeline(commands, PipelineBindPointCompute, pipeline.Handle);
                    _boundPipeline = pipeline.Handle;
                }

                if (bindings > 0)
                {
                    if (_usePush)
                    {
                        _pushDescriptorSet(commands, PipelineBindPointCompute, pipeline.Layout, 0, (uint)bindings, writes);
                    }
                    else
                    {
                        ulong set = AllocateSet(SetLayout(bindings), (uint)bindings);
                        for (int i = 0; i < bindings; i++)
                        {
                            writes[i].DestinationSet = set;
                        }

                        vkUpdateDescriptorSets(_device, (uint)bindings, writes, 0, null);
                        vkCmdBindDescriptorSets(commands, PipelineBindPointCompute, pipeline.Layout, 0, 1, &set, 0, null);
                    }
                }

                if (pushConstants.Length > 0)
                {
                    fixed (byte* values = pushConstants)
                    {
                        vkCmdPushConstants(commands, pipeline.Layout, ShaderStageCompute, 0, (uint)pushConstants.Length, values);
                    }
                }

                vkCmdDispatch(commands, groupsX, groupsY, groupsZ);
            }
            finally
            {
                blocks.Clear();                                                // no blocks kept alive past the call
            }

            Dispatches++;
            DispatchesByKernel?.AddOrUpdate(kernel.Name, 1, static (_, n) => n + 1);
            Recorded();
            if (CheckNan)
            {
                CheckWrittenFinite(kernel, groupsX, groupsY, groupsZ, storages, pushConstants);
            }
        }
    }

    private void StartQueue()
    {
        var poolInfo = new VkCommandPoolCreateInfo
        {
            SType = StructureCommandPoolCreateInfo,
            Flags = CommandPoolResetCommandBuffer,
            QueueFamilyIndex = _physical.QueueFamily,
        };
        Check(vkCreateCommandPool(_device, &poolInfo, null, out _commandPool), nameof(vkCreateCommandPool));
        var cacheInfo = new VkPipelineCacheCreateInfo { SType = StructurePipelineCacheCreateInfo };
        Check(vkCreatePipelineCache(_device, &cacheInfo, null, out _pipelineCache), nameof(vkCreatePipelineCache));
        _batch = NewBatch();
    }

    private void StopQueue()
    {
        foreach (var batch in _free.Append(_batch))
        {
            vkDestroyFence(_device, batch.Fence, null);
            foreach (var pool in batch.Pools)
            {
                vkDestroyDescriptorPool(_device, pool.Handle, null);
            }
        }

        _free.Clear();
        vkDestroyCommandPool(_device, _commandPool, null);                    // frees the command buffers
        foreach (var pipeline in _pipelines.Values)
        {
            vkDestroyPipeline(_device, pipeline.Handle, null);
        }

        foreach (var layout in _pipelineLayouts.Values)
        {
            vkDestroyPipelineLayout(_device, layout, null);
        }

        foreach (var layout in _setLayouts.Values)
        {
            vkDestroyDescriptorSetLayout(_device, layout, null);
        }

        vkDestroyPipelineCache(_device, _pipelineCache, null);
        _pipelines.Clear();
        _pipelineLayouts.Clear();
        _setLayouts.Clear();
    }

    private Batch NewBatch()
    {
        var batch = new Batch();
        var allocate = new VkCommandBufferAllocateInfo
        {
            SType = StructureCommandBufferAllocateInfo,
            CommandPool = _commandPool,
            Level = CommandBufferLevelPrimary,
            CommandBufferCount = 1,
        };
        IntPtr commands;
        Check(vkAllocateCommandBuffers(_device, &allocate, &commands), nameof(vkAllocateCommandBuffers));
        batch.Commands = commands;
        var fenceInfo = new VkFenceCreateInfo { SType = StructureFenceCreateInfo };
        Check(vkCreateFence(_device, &fenceInfo, null, out batch.Fence), nameof(vkCreateFence));
        return batch;
    }

    // Makes, up front, the batches the queue cycles through (the one recorded, those in flight, the one retired as the
    // next is submitted), each with a descriptor pool when dispatches allocate sets: recording dispatches then
    // allocates nothing once the device runs.
    private void PrepareBatches()
    {
        lock (_gate)
        {
            _free.EnsureCapacity(_maxInFlight + 2);
            _inFlight.EnsureCapacity(_maxInFlight + 2);
            for (int i = _free.Count + _inFlight.Count + 1; i < _maxInFlight + 2; i++)
            {
                _free.Push(NewBatch());
            }

            if (!_usePush)
            {
                foreach (var batch in _free.Append(_batch))
                {
                    if (batch.Pools.Count == 0)
                    {
                        batch.Pools.Add(BatchPool());
                    }
                }
            }
        }
    }

    // The current batch's command buffer, begun if needed, for a command that uses `blocks` (bit i of `writes` set when
    // it writes blocks[i]): a barrier first when it depends on a command of the current span (the barrier then waits
    // for every earlier command, in this batch or an earlier one, and makes its writes visible); the blocks are marked.
    private IntPtr Record(ReadOnlySpan<VulkanBlock> blocks, ulong writes, string name = "", bool forceBarrier = false)
    {
        var batch = _batch;
        if (!batch.Recording)
        {
            var begin = new VkCommandBufferBeginInfo { SType = StructureCommandBufferBeginInfo, Flags = CommandBufferOneTimeSubmit };
            Check(vkBeginCommandBuffer(batch.Commands, &begin), nameof(vkBeginCommandBuffer));
            batch.Recording = true;
            batch.Number = _recording;
            batch.Last = "";
            _boundPipeline = 0;
        }

        bool hazard = forceBarrier;
        for (int i = 0; i < blocks.Length; i++)
        {
            var block = blocks[i];
            hazard |= block.WrittenIn == _span || ((writes >> i & 1) != 0 && block.ReadIn == _span);
        }

        hazard |= s_barrierEveryCommand && blocks.Length > 0;
        hazard |= s_barrierKernels is not null && blocks.Length > 0 && s_barrierKernels.Contains(name);
        if (name.Length > 0)
        {
            LogCommand(name);
            batch.Last = name;
        }

        if (hazard)
        {
            var barrier = new VkMemoryBarrier
            {
                SType = StructureMemoryBarrier,
                SourceAccessMask = AccessShaderWrite | AccessTransferWrite,
                DestinationAccessMask = AccessShaderRead | AccessShaderWrite | AccessTransferRead | AccessTransferWrite,
            };
            vkCmdPipelineBarrier(batch.Commands, StageComputeShader | StageTransfer, StageComputeShader | StageTransfer, 0, 1, &barrier, 0, null, 0, null);
            _span++;
            Barriers++;
        }

        for (int i = 0; i < blocks.Length; i++)
        {
            var block = blocks[i];
            if ((writes >> i & 1) != 0)
            {
                block.WrittenIn = _span;
            }
            else
            {
                block.ReadIn = _span;
            }

            block.LastUse = _recording;
        }

        if (_capture is { } capture && ReferenceEquals(batch, capture.Batch))
        {
            CaptureBlocks(capture, blocks);                                    // a graph is recorded (VulkanBackend.Graphs.cs)
        }

        return batch.Commands;
    }

    // After a command: a long batch goes to the device now, so it starts working while more is recorded.
    private void Recorded()
    {
        if (++_batch.Count >= _maxBatchCommands)
        {
            Submit();
        }
    }

    // Copies bytes between blocks in queue order.
    private void RecordCopy(VulkanBlock source, long sourceOffset, VulkanBlock destination, long destinationOffset, long bytes)
    {
        _commandBlocks[0] = source;
        _commandBlocks[1] = destination;
        var commands = Record(_commandBlocks.AsSpan(0, 2), writes: 0b10, "copy");
        _commandBlocks.AsSpan(0, 2).Clear();
        var region = new VkBufferCopy { SourceOffset = (ulong)sourceOffset, DestinationOffset = (ulong)destinationOffset, Size = (ulong)bytes };
        vkCmdCopyBuffer(commands, source.Buffer, destination.Buffer, 1, &region);
        Recorded();
    }

    // Zeros a block in queue order.
    private void RecordFill(VulkanBlock block)
    {
        // Always after a barrier when queued work may still use the block. The span bookkeeping already orders a fill
        // after the commands it knows of, but on one vendor's driver training that zeros recycled
        // blocks went wrong unless a barrier preceded the fill (synchronization validation finds no hazard in the same
        // commands on Mesa's software driver). Fills are rare (a zeroed allocation of a busy block), so it costs little.
        var commands = Record(new ReadOnlySpan<VulkanBlock>(in block), writes: 1, "fill", forceBarrier: block.LastUse > _completed);
        vkCmdFillBuffer(commands, block.Buffer, 0, WholeSize, 0);
        Recorded();
    }

    // Submits the current batch (when it has commands), ending it with a barrier that makes its writes visible to the
    // host once its fence signals.
    private void Submit()
    {
        var batch = _batch;
        if (_capture is { } capture && ReferenceEquals(batch, capture.Batch))
        {
            ContinueCapture(capture);                                          // a recorded graph is not submitted
            return;
        }

        if (!batch.Recording)
        {
            return;
        }

        var barrier = new VkMemoryBarrier
        {
            SType = StructureMemoryBarrier,
            SourceAccessMask = AccessShaderWrite | AccessTransferWrite,
            DestinationAccessMask = AccessHostRead | AccessHostWrite,
        };
        vkCmdPipelineBarrier(batch.Commands, StageComputeShader | StageTransfer, StageHost, 0, 1, &barrier, 0, null, 0, null);
        Check(vkEndCommandBuffer(batch.Commands), nameof(vkEndCommandBuffer));
        IntPtr commands = batch.Commands;
        var submit = new VkSubmitInfo { SType = StructureSubmitInfo, CommandBufferCount = 1, CommandBuffers = &commands };
        CheckWork(vkQueueSubmit(_queue, 1, &submit, batch.Fence), nameof(vkQueueSubmit), batch);
        _inFlight.Enqueue(batch);
        Submissions++;
        _recording++;
        _batch = _free.Count > 0 ? _free.Pop() : NewBatch();
        if (_inFlight.Count > _maxInFlight)
        {
            Retire(_inFlight.Dequeue());
        }
    }

    // Waits for the submitted batches numbered up to `number`.
    private void WaitUntil(ulong number)
    {
        while (_inFlight.Count > 0 && _inFlight.Peek().Number <= number)
        {
            Retire(_inFlight.Dequeue());
        }
    }

    // Submits everything recorded and waits for all of it.
    private void SubmitAndWait()
    {
        Submit();
        WaitUntil(ulong.MaxValue);
    }

    // Waits until the work using `block` has finished, so the host may read or write it.
    private void WaitFor(VulkanBlock block)
    {
        if (block.LastUse <= _completed)
        {
            return;
        }

        if (block.LastUse == _recording)
        {
            Submit();
        }

        WaitUntil(block.LastUse);
    }

    // Waits for a submitted batch, then makes it ready to record again.
    private void Retire(Batch batch)
    {
        ulong fence = batch.Fence;
        CheckWork(vkWaitForFences(_device, 1, &fence, 1, ulong.MaxValue), nameof(vkWaitForFences), batch);
        Check(vkResetFences(_device, 1, &fence), nameof(vkResetFences));
        foreach (var pool in batch.Pools)
        {
            Check(vkResetDescriptorPool(_device, pool.Handle, 0), nameof(vkResetDescriptorPool));
            (pool.Sets, pool.Descriptors) = (pool.SetCapacity, pool.Capacity);
        }

        Check(vkResetCommandBuffer(batch.Commands, 0), nameof(vkResetCommandBuffer));
        (batch.Pool, batch.Recording, batch.Count) = (0, false, 0);
        _completed = Math.Max(_completed, batch.Number);
        _free.Push(batch);
    }

    // The result of submitting or waiting for `batch`: a lost device names the batch's work (its last command) and the
    // ways around it (VulkanDriver.DeviceLostMessage); any other failure as Check reports it.
    private void CheckWork(int result, string call, Batch batch)
    {
        if (result == ErrorDeviceLost)
        {
            throw new VulkanException(DeviceLostMessage(call, Name, batch.Count, batch.Last, RetryOnHost));
        }

        Check(result, call);
    }

    // A descriptor set for this batch, from its pools (pools are counted here, so allocation never fails for lack of room).
    private ulong AllocateSet(ulong layout, uint descriptors)
    {
        var batch = _batch;
        while (true)
        {
            if (batch.Pool == batch.Pools.Count)
            {
                batch.Pools.Add(BatchPool());
            }

            var pool = batch.Pools[batch.Pool];
            if (pool.Sets == 0 || pool.Descriptors < descriptors)
            {
                batch.Pool++;
                continue;
            }

            var info = new VkDescriptorSetAllocateInfo
            {
                SType = StructureDescriptorSetAllocateInfo,
                DescriptorPool = pool.Handle,
                DescriptorSetCount = 1,
                SetLayouts = &layout,
            };
            ulong set;
            Check(vkAllocateDescriptorSets(_device, &info, &set), nameof(vkAllocateDescriptorSets));
            pool.Sets--;
            pool.Descriptors -= descriptors;
            return set;
        }
    }

    private DescriptorPool NewPool(uint sets, uint descriptors)
    {
        var size = new VkDescriptorPoolSize { Type = DescriptorStorageBuffer, DescriptorCount = descriptors };
        var info = new VkDescriptorPoolCreateInfo
        {
            SType = StructureDescriptorPoolCreateInfo,
            MaxSets = sets,
            PoolSizeCount = 1,
            PoolSizes = &size,
        };
        Check(vkCreateDescriptorPool(_device, &info, null, out ulong pool), nameof(vkCreateDescriptorPool));
        return new DescriptorPool(pool, sets, descriptors);
    }

    // The descriptor set layout of `bindings` storage buffers (bindings 0 to bindings - 1), made once per descriptor mode.
    private ulong SetLayout(int bindings)
    {
        if (_setLayouts.TryGetValue((bindings, _usePush), out ulong layout))
        {
            return layout;
        }

        var entries = new VkDescriptorSetLayoutBinding[Math.Max(bindings, 1)];
        for (int i = 0; i < bindings; i++)
        {
            entries[i] = new VkDescriptorSetLayoutBinding
            {
                Binding = (uint)i,
                DescriptorType = DescriptorStorageBuffer,
                DescriptorCount = 1,
                StageFlags = ShaderStageCompute,
            };
        }

        fixed (VkDescriptorSetLayoutBinding* e = entries)
        {
            var info = new VkDescriptorSetLayoutCreateInfo
            {
                SType = StructureDescriptorSetLayoutCreateInfo,
                Flags = _usePush ? DescriptorSetLayoutPushDescriptor : 0,
                BindingCount = (uint)bindings,
                Bindings = e,
            };
            Check(vkCreateDescriptorSetLayout(_device, &info, null, out layout), nameof(vkCreateDescriptorSetLayout));
        }

        _setLayouts[(bindings, _usePush)] = layout;
        return layout;
    }

    // The pipeline of `kernel` on this device, built on its first dispatch.
    private Pipeline PipelineOf(VulkanKernel kernel)
    {
        if (kernel.LastPipeline is Pipeline last && ReferenceEquals(last.Owner, this))
        {
            return last;
        }

        if (_pipelines.TryGetValue(kernel, out var pipeline))
        {
            kernel.LastPipeline = pipeline;
            return pipeline;
        }

        ThrowIfRefused(kernel);                                                // asked once per process (VulkanBackend.Refusals.cs)
        ref readonly var p = ref _physical.Properties;
        if (kernel.Bindings > p.MaxPerStageDescriptorStorageBuffers)
        {
            throw new VulkanException($"Vulkan kernel '{kernel.Name}' binds {kernel.Bindings} storages; {Name} allows {p.MaxPerStageDescriptorStorageBuffers}.");
        }

        if (kernel.PushConstantBytes > p.MaxPushConstantsSize)
        {
            throw new VulkanException($"Vulkan kernel '{kernel.Name}' takes {kernel.PushConstantBytes} bytes of push constants; {Name} allows {p.MaxPushConstantsSize}.");
        }

        var key = (kernel.Bindings, kernel.PushConstantBytes, _usePush);
        if (!_pipelineLayouts.TryGetValue(key, out ulong layout))
        {
            ulong setLayout = SetLayout(kernel.Bindings);
            var range = new VkPushConstantRange { StageFlags = ShaderStageCompute, Offset = 0, Size = (uint)kernel.PushConstantBytes };
            var info = new VkPipelineLayoutCreateInfo
            {
                SType = StructurePipelineLayoutCreateInfo,
                SetLayoutCount = 1,
                SetLayouts = &setLayout,
                PushConstantRangeCount = kernel.PushConstantBytes > 0 ? 1u : 0u,
                PushConstantRanges = &range,
            };
            Check(vkCreatePipelineLayout(_device, &info, null, out layout), nameof(vkCreatePipelineLayout));
            _pipelineLayouts[key] = layout;
        }

        ulong module;
        uint[] words = Limits.InfNanPreserve ? SpirvModule.WithInfNanPreserved(kernel.Spirv) : kernel.Spirv;
        fixed (uint* code = words)
        {
            var info = new VkShaderModuleCreateInfo { SType = StructureShaderModuleCreateInfo, CodeSize = (nuint)words.Length * sizeof(uint), Code = code };
            int result = vkCreateShaderModule(_device, &info, null, out module);
            if (result != Success)
            {
                throw Refuse(kernel, $"vkCreateShaderModule: {VulkanDriver.Describe(result)}");
            }
        }

        try
        {
            ulong handle = BuildPipeline(kernel, module, layout);                // the ladder: VulkanBackend.Refusals.cs
            pipeline = new Pipeline(handle, layout, this);
            _pipelines[kernel] = pipeline;
            kernel.LastPipeline = pipeline;
            return pipeline;
        }
        finally
        {
            vkDestroyShaderModule(_device, module, null);
        }
    }
}
