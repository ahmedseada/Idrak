// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction;
using Idrak.Abstraction.Devices;
using Idrak.Gpu.Cuda;

// Device and scope state (plan 36, G-1 2a): a TensorScope disposed out of order stays live; CUDA's per-thread launch
// settings never leak past a refused launch; its tensor-core modules load once whichever threads reach them first.
// The CUDA tests run on a CUDA device only and report SKIP elsewhere.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] DeviceStateGroup =
    [
        ("tensor scope: disposing an outer scope first throws, the inner stays current, a later tensor lands in the live scope, and disposing in order afterwards disposes every tensor", d => { CpuOnly(d); TensorScopeOutOfOrder(); }),
        ("cuda launch: a launch with the wrong argument count throws naming the kernel, leaves no shared bytes or cooperative flag to the next launch, and the next launch runs", CudaLaunchRefusedLeavesNothing),
        ("tensor cores: two threads' first use loads the tensor-core modules once, both get the same answer and both launch", TensorCoresLoadOnce),
    ];

    private static void TensorScopeOutOfOrder()
    {
        var outer = new TensorScope();
        var a = Tensor.Zeros([2]);
        var inner = new TensorScope();
        var b = Tensor.Zeros([2]);
        Check(outer.Owns(a) && inner.Owns(b), "each tensor is tracked by the scope current when it is made");

        bool threw = false;
        try
        {
            outer.Dispose();
        }
        catch (InvalidOperationException e) when (e.Message.Contains("reverse order", StringComparison.Ordinal))
        {
            threw = true;
        }

        Check(threw, "disposing the outer scope while the inner one is current throws, naming the order");
        Check(!a.IsDisposed, "the refused dispose disposed none of the outer scope's tensors");
        var c = Tensor.Zeros([2]);
        Check(inner.Owns(c) && !outer.Owns(c), "after the refused dispose the inner scope is still current: a new tensor lands in it");

        inner.Dispose();
        Check(b.IsDisposed && c.IsDisposed && !a.IsDisposed, "disposing the inner scope disposes its tensors only");
        var d = Tensor.Zeros([2]);
        Check(outer.Owns(d), "the outer scope is current again and live (the refused dispose did not mark it dead): a later tensor lands in it");

        outer.Dispose();
        Check(a.IsDisposed && d.IsDisposed, "disposing the outer scope in order afterwards disposes its tensors, the one made before and the one made after the refusal");
        var later = Tensor.Zeros([2]);
        Check(!outer.Owns(later), "after the outer scope is disposed the enclosing scope is current: a new tensor is not tracked by the disposed one");
        outer.Dispose();                                                   // a second dispose is a no-op
        later.Dispose();
    }

    private static CudaBackend CudaOnly(Device device)
    {
        if (device.Backend is not CudaBackend cuda)
        {
            Skip($"needs a CUDA device ({device} is not one; run with IDRAK_DEVICES=cuda)");
            throw new InvalidOperationException("unreachable");
        }

        return cuda;
    }

    private static void CudaLaunchRefusedLeavesNothing(Device device)
    {
        var cuda = CudaOnly(device);
        var y = cuda.Allocate(4, zeroed: true);
        try
        {
            string? refusal = null;
            try
            {
                cuda.LaunchFill(4096, cooperative: true, 1UL);                // fill_f32 declares 3 parameters
            }
            catch (InvalidOperationException e)
            {
                refusal = e.Message;
            }

            Check(refusal is not null && refusal.Contains("fill_f32", StringComparison.Ordinal) && refusal.Contains("1 arguments", StringComparison.Ordinal),
                $"a launch with the wrong argument count throws naming the kernel and the count: {refusal ?? "it launched"}");
            Check(CudaBackend.PendingLaunchSettings == (0u, false),
                $"the refused launch took its settings: none left for the next launch on this thread, not {CudaBackend.PendingLaunchSettings}");

            cuda.FillKernel(y, 4, 2.5f);
            var values = new float[4];
            cuda.Download(y, values);
            Check(values.All(v => v == 2.5f), $"the next launch runs with its own settings: [{string.Join(", ", values)}]");
        }
        finally
        {
            y.Release();
        }
    }

    private static void TensorCoresLoadOnce(Device device)
    {
        _ = CudaOnly(device);
        // A backend of its own, so its tensor cores are untried whatever ran before (the device's shared one may have loaded them).
        var fresh = (CudaBackend)device.CreateSeparateBackend(new SeparateBackendOptions("tensor cores: first use", null));
        using var start = new Barrier(2);
        var reasons = new string?[2];
        var filled = new float[2][];
        var failures = new Exception?[2];
        var threads = Enumerable.Range(0, 2).Select(i => new Thread(() =>
        {
            try
            {
                start.SignalAndWait();
                reasons[i] = ((IMatrixUnitBackend)fresh).TensorCoresUnavailable();
                var y = fresh.Allocate(8, zeroed: false);
                try
                {
                    fresh.FillKernel(y, 8, i + 1);
                    var values = new float[8];
                    fresh.Download(y, values);
                    filled[i] = values;
                }
                finally
                {
                    y.Release();
                }
            }
            catch (Exception e)
            {
                failures[i] = e;
            }
        })).ToArray();
        foreach (var thread in threads)
        {
            thread.Start();
        }

        foreach (var thread in threads)
        {
            thread.Join();
        }

        Check(failures.All(f => f is null), $"both threads ran: {string.Join(" | ", failures.Where(f => f is not null).Select(f => f!.Message))}");
        Check(fresh.TensorCoreLoads == 1, $"the tensor-core modules loaded once for two threads' first use, not {fresh.TensorCoreLoads} times");
        Check(reasons[0] == reasons[1], $"both threads got the same answer (no thread saw the load half done): '{reasons[0]}' and '{reasons[1]}'");
        Check(filled[0]!.All(v => v == 1f) && filled[1]!.All(v => v == 2f), "both threads launched after the load");
        Check(((IMatrixUnitBackend)fresh).TensorCoresUnavailable() == reasons[0] && fresh.TensorCoreLoads == 1, "a later call reads the result without loading again");
    }
}
