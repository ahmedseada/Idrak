// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Diagnostics;

namespace Idrak.Abstraction;

/// <summary>
/// A recorded sequence of device work that can be replayed with a single launch. On CUDA this is a CUDA Graph:
/// the ~hundreds of small kernel launches of, say, one transformer decoding step become one
/// <c>cuGraphLaunch</c>, removing nearly all CPU launch overhead. On the CPU (or if recording fails) replaying
/// simply runs the step again, so code using graphs works everywhere.
/// </summary>
/// <remarks>
/// A recorded step must be replayable as-is: kernel arguments (sizes, offsets, scalars) are frozen at recording
/// time, so anything that changes between replays must live in device memory (e.g. a position counter updated by
/// the step itself). The step must not copy data from the host or read results back while recording, and the
/// tensors it uses (weights, caches, input and output buffers) must stay alive and on the same device.
/// </remarks>
public sealed class ComputeGraph : IDisposable
{
    private readonly Device _device;
    private readonly IGraphCaptureBackend? _graphs;                   // the device's recording, when the step was recorded
    private readonly Action? _fallback;
    private readonly IntPtr _executable;
    private readonly IntPtr _graph;
    private readonly List<Tensor> _tensors;
    private readonly List<Storage> _owned;
    private bool _disposed;

    private ComputeGraph(Device device, Action? fallback, IntPtr executable, IntPtr graph, List<Tensor> tensors, List<Storage> owned, string? failure)
    {
        _device = device;
        _graphs = executable != IntPtr.Zero ? (IGraphCaptureBackend)device.Backend : null;
        _fallback = fallback;
        _executable = executable;
        _graph = graph;
        _tensors = tensors;
        _owned = owned;
        FailureReason = failure;
    }

    [ThreadStatic]
    private static int t_capturing;

    // Open RunStepsDirectly scopes (any thread).
    private static int s_direct;

    /// <summary>
    /// Until the returned scope is disposed, graphs recorded on any thread replay by running their steps, as on a device
    /// without graphs: every kernel launches on its own, so a profiler (the "kernels" probe, <c>GpuProfiler</c>) times
    /// each one, decoding steps included. Slower; for profiling and diagnostics. Graphs recorded before keep replaying.
    /// </summary>
    public static IDisposable RunStepsDirectly()
    {
        Interlocked.Increment(ref s_direct);
        return new DirectScope();
    }

    private sealed class DirectScope : IDisposable
    {
        private int _ended;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _ended, 1) == 0)
            {
                Interlocked.Decrement(ref s_direct);
            }
        }
    }

    /// <summary>Whether work on this thread is being recorded into a graph (its buffers must outlive the recording).</summary>
    public static bool IsCapturing => t_capturing > 0;

    /// <summary>True when replays use a real device graph; false when they re-run the step (CPU, or recording failed).</summary>
    public bool IsRecorded => _executable != IntPtr.Zero;

    /// <summary>Why recording was not possible, when <see cref="IsRecorded"/> is false on a device that supports graphs.</summary>
    public string? FailureReason { get; }

    /// <summary>
    /// Records <paramref name="step"/> on <paramref name="device"/>. The step is executed once as part of recording
    /// only on devices without graph support; on CUDA, recording does not run it (call <see cref="Replay"/>).
    /// </summary>
    public static ComputeGraph Capture(Device device, Action step) => Capture(device, step, "a step");

    /// <summary>
    /// Records <paramref name="step"/> on <paramref name="device"/>, as <see cref="Capture(Device, Action)"/>;
    /// <paramref name="work"/> names it ("a decoding step") where a failed recording is reported
    /// (<see cref="GraphRecordingFailed"/>, with <see cref="FailureReason"/>).
    /// </summary>
    public static ComputeGraph Capture(Device device, Action step, string work)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(work);
        var backend = device.Backend;
        if (!backend.SupportsGraphs || Volatile.Read(ref s_direct) > 0)
        {
            return new ComputeGraph(device, step, IntPtr.Zero, IntPtr.Zero, [], [], null);
        }

        var graphs = (IGraphCaptureBackend)backend;                       // SupportsGraphs: the device records
        long started = Telemetry.StartStage();
        var scope = new TensorScope();
        graphs.BeginCapture();
        try
        {
            using (Autograd.NoGrad())
            {
                t_capturing++;
                try
                {
                    step();
                }
                finally
                {
                    t_capturing--;
                }
            }

            var (executable, graph, owned) = graphs.EndCapture();
            var kept = scope.Detach();
            if (started != 0)
            {
                long bytes = 0;
                owned.ForEach(s => bytes += 4L * s.Length);
                kept.ForEach(t => bytes += 4L * t.Size);
                Telemetry.InferenceStage(InferenceStageKind.GraphCapture, device, started, 1, bytes);
            }

            return new ComputeGraph(device, null, executable, graph, kept, owned, null);
        }
        catch (Exception ex)
        {
            foreach (var storage in graphs.AbortCapture())
            {
                storage.Release();
            }

            scope.Dispose();
            Telemetry.GraphRecordingFailed(new GraphRecordingFailed(backend.Kind, work, ex.Message));
            return new ComputeGraph(device, step, IntPtr.Zero, IntPtr.Zero, [], [], ex.Message);
        }
    }

    /// <summary>Runs the recorded work again (asynchronously on the GPU).</summary>
    public void Replay()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_graphs is { } graphs)
        {
            graphs.ReplayGraph(_executable);
            return;
        }

        using var scope = new TensorScope();
        using (Autograd.NoGrad())
        {
            _fallback!();
        }
    }

    /// <summary>Destroys the graph and returns its memory to the device pool.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        if (_graphs is { } graphs)
        {
            _device.Backend.Synchronize();
            graphs.DestroyGraph(_executable, _graph);
        }

        _tensors.ForEach(t => t.Dispose());
        _owned.ForEach(s => s.Release());
    }
}
