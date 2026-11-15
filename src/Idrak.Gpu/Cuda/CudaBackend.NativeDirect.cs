// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Runtime.CompilerServices;

namespace Idrak.Gpu.Cuda;

// The experiment's direct native lane (widened inside the product): the native bfloat16 kernel in the device's measured
// shape, staging x from float32 and W decoded from its description as each block of k lands in shared memory
// (PtxKernels.NativeMma with a NativeSource), so no packing pass and no full copy of W run; its k split measured per
// product shape (Tune: unsplit unless a split is clearly faster), once.
internal sealed unsafe partial class CudaBackend
{
    private readonly ConcurrentDictionary<string, IntPtr?> _nativeDirect = new(StringComparer.Ordinal);

    // A number for each source, for the measured splits' keys.
    private static readonly ConcurrentDictionary<string, int> DirectIds = new(StringComparer.Ordinal);

    // The direct kernel of the source in the device's native shape (NativeKernel), loaded on first use; null where the
    // device has no native kernel or its driver refuses this one (TypedUnavailableReason says why).
    private IntPtr? NativeDirect(PtxKernels.NativeSource source)
    {
        if (NativeKernel() is not { } native)
        {
            return null;
        }

        return _nativeDirect.TryGetValue(source.Code, out var known) ? known : LoadNativeDirect(source, native.Config);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private IntPtr? LoadNativeDirect(PtxKernels.NativeSource source, PtxKernels.NativeMmaConfig config)
    {
        lock (_typedLock)
        {
            if (_nativeDirect.TryGetValue(source.Code, out var known))
            {
                return known;
            }

            string ptx = PtxKernels.NativeMma(PtxKernels.NativeKind.BF16, config, out string entry, source);
            int shared = PtxKernels.NativeMmaShared(config), dynamic = shared > 48 * 1024 ? shared : 0;
            IntPtr? function = LoadDynamicModule(entry, [entry], ptx, _ => dynamic, out string? reason) is { } kernels ? kernels[entry] : null;
            if (function is null)
            {
                TypedUnavailableReason = reason;
            }

            _nativeDirect[source.Code] = function;
            return function;
        }
    }

    // Whether the direct lane takes the product: a bfloat16 run, prompt-sized, a source the kernel decodes, k whole units
    // (8 values, a block), row scales only on 8-bit integers, and the kernel loads here.
    private bool NativeDirectLane(ElementType type, Storage? scales, int m, int k) =>
        m > PtxKernels.TypedGemvRows[^1] && BFloat16Products && k % 8 == 0 && PtxKernels.NativeSource.Of(type) is { } source && k % source.Unit == 0
        && (scales is null || source.Typed is { Kind: 'i' }) && NativeDirect(source) is not null;

    // y [m, n] = x · Wᵀ by the direct lane, its k split measured per shape.
    private bool RunNativeDirect(Storage x, Storage w, Storage? scales, Storage y, int m, int n, int k)
    {
        if (PtxKernels.NativeSource.Of(w.ElementType) is not { } source || NativeDirect(source) is not { } function || NativeKernel() is not { } native)
        {
            return false;
        }

        var config = native.Config;
        int kp = PtxKernels.NativePadded(k);
        int[] splits = NativeSplits(config, kp);
        var key = new TuneKey(TuneOp.ProductRoute, 9300 + DirectIds.GetOrAdd(source.Code, static _ => DirectIds.Count), TuneSizes.Class(m), n, kp, splits.Length);
        if (!TryTuned(key, splits, 1, out int split))
        {
            split = MeasureDirect(key, splits, function, config, x, w, scales, y, m, n, k, kp);
        }

        LaunchDirect(function, config, x, w, scales, y, m, n, k, kp, split);
        return true;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private int MeasureDirect(TuneKey key, int[] splits, IntPtr function, PtxKernels.NativeMmaConfig config, Storage x, Storage w, Storage? scales, Storage y, int m, int n, int k, int kp) =>
        Tune(key, splits, 1, s => LaunchDirect(function, config, x, w, scales, y, m, n, k, kp, s));

    // The direct kernel over `splits` ranges of k's blocks (grid z): with one, C is y; with more, the parts then their sum.
    private void LaunchDirect(IntPtr function, PtxKernels.NativeMmaConfig config, Storage x, Storage w, Storage? scales, Storage y, int m, int n, int k, int kp, int splits)
    {
        int shared = PtxKernels.NativeMmaShared(config);
        int blockValues = config.RowBytes * 8 / 16, blocks = kp / blockValues, per = (blocks + splits - 1) / splits;
        var target = splits == 1 ? y : _splitPartials.Get(this, splits * m * n);
        t_sharedBytes = shared > 48 * 1024 ? (uint)shared : 0;
        Launch(function, (uint)((n + config.Bn - 1) / config.Bn), (uint)((m + 127) / 128), (uint)splits, PtxKernels.NativeMmaThreads, 1,
            P(x), P(w), P(target), U(m), U(n), U(kp), U(kp), U(per), U(k), U((int)w.ElementType.ByteCount(k)), scales is null ? 0UL : P(scales));
        if (splits > 1)
        {
            var sum = Typed(DecodeOf(ElementType.Float32)!)!.Sum;
            Launch(sum, (uint)((m * n + 255) / 256), 1, 256, 1, P(target), P(y), U(m * n), U(splits));
        }
    }

    /// <summary>Tests: the direct native lane run directly (false where it does not take the product).</summary>
    internal bool StoredNativeDirect(ElementType type, Storage x, Storage weight, Storage? scales, Storage y, int m, int n, int k) =>
        NativeDirectLane(type, scales, m, k) && RunNativeDirect(x, weight, scales, y, m, n, k);
}
