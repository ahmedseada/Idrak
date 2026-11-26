// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction;
using Idrak.Abstraction.Devices;
using Idrak.Gpu.Cuda;

// Device and scope state (plan 36, G-1 2a): a TensorScope disposed out of order stays live; CUDA's per-thread launch
// settings never leak past a refused launch. The CUDA test runs on a CUDA device only and reports SKIP elsewhere.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] DeviceStateGroup =
    [
        ("tensor scope: disposing an outer scope first throws, the inner stays current, a later tensor lands in the live scope, and disposing in order afterwards disposes every tensor", d => { CpuOnly(d); TensorScopeOutOfOrder(); }),
        ("cuda launch: a launch with the wrong argument count throws naming the kernel, leaves no shared bytes or cooperative flag to the next launch, and the next launch runs", CudaLaunchRefusedLeavesNothing),
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
}
