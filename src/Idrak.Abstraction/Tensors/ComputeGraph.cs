// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;

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
    private readonly Action? _fallback;
    private readonly IntPtr _executable;
    private readonly IntPtr _graph;
    private readonly List<Tensor> _tensors;
    private readonly List<Storage> _owned;
    private bool _disposed;

    private ComputeGraph(Device device, Action? fallback, IntPtr executable, IntPtr graph, List<Tensor> tensors, List<Storage> owned, string? failure)
    {
        _device = device;
        _fallback = fallback;
        _executable = executable;
        _graph = graph;
        _tensors = tensors;
        _owned = owned;
        FailureReason = failure;
    }

    [ThreadStatic]
    private static int t_capturing;

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
    public static ComputeGraph Capture(Device device, Action step)
    {
        var backend = device.Backend;
        if (!backend.SupportsGraphs)
        {
            return new ComputeGraph(device, step, IntPtr.Zero, IntPtr.Zero, [], [], null);
        }

        var scope = new TensorScope();
        backend.BeginCapture();
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

            var (executable, graph, owned) = backend.EndCapture();
            return new ComputeGraph(device, null, executable, graph, scope.Detach(), owned, null);
        }
        catch (Exception ex)
        {
            foreach (var storage in backend.AbortCapture())
            {
                storage.Release();
            }

            scope.Dispose();
            return new ComputeGraph(device, step, IntPtr.Zero, IntPtr.Zero, [], [], ex.Message);
        }
    }

    /// <summary>Runs the recorded work again (asynchronously on the GPU).</summary>
    public void Replay()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_executable != IntPtr.Zero)
        {
            _device.Backend.ReplayGraph(_executable);
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
        if (_executable != IntPtr.Zero)
        {
            _device.Backend.Synchronize();
            _device.Backend.DestroyGraph(_executable, _graph);
        }

        _tensors.ForEach(t => t.Dispose());
        _owned.ForEach(s => s.Release());
    }
}
