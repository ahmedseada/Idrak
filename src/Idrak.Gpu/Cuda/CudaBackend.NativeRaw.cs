// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Gpu.Cuda;

// The experiment's raw int8 lane (no packing pass over the weights): the native bfloat16 kernel in the device's measured
// shape reading int8 weights' codes as they are stored (PtxKernels.NativeMma with raw: copied into shared memory a byte a
// value, widened in registers as each fragment loads, the row scales applied to C), x packed once in the permuted order
// its fragments read (once for a group's products sharing x: SharedPermuted); its k split measured per product shape, once. A candidate of the measured lanes (StoredMeasured),
// taken only where it measures faster than the packed lane, which widens every weight into bfloat16 rows on each call.
internal sealed unsafe partial class CudaBackend
{
    private IntPtr? _nativeRaw, _packPermuted;
    private bool _nativeRawTried;

    // The raw kernel in the device's native shape and the permuted x pack, loaded on first use; null where either does not
    // load (TypedUnavailableReason says why) or the device has no native kernel.
    private (IntPtr Kernel, IntPtr PackX, PtxKernels.NativeMmaConfig Config)? NativeRaw()
    {
        if (NativeKernel() is not { } native)
        {
            return null;
        }

        if (!Volatile.Read(ref _nativeRawTried))
        {
            LoadNativeRaw(native.Config);
        }

        return _nativeRaw is { } kernel && _packPermuted is { } pack ? (kernel, pack, native.Config) : null;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private void LoadNativeRaw(PtxKernels.NativeMmaConfig config)
    {
        lock (_typedLock)
        {
            if (_nativeRawTried)
            {
                return;
            }

            try
            {
                var kind = PtxKernels.NativeKind.BF16;
                string ptx = PtxKernels.NativeMma(kind, config, out string entry, raw: true);
                int shared = PtxKernels.NativeMmaShared(config, raw: true), dynamic = shared > 48 * 1024 ? shared : 0;
                _nativeRaw = LoadDynamicModule(entry, [entry], ptx, _ => dynamic, out string? reason) is { } kernels ? kernels[entry] : null;
                string pack = PtxKernels.NativePackXPermutedName(kind);
                _packPermuted = LoadDynamicModule(pack, [pack], PtxKernels.NativePackXPermuted(kind), static _ => 0, out string? packReason) is { } packs ? packs[pack] : null;
                if (_nativeRaw is null || _packPermuted is null)
                {
                    TypedUnavailableReason = reason ?? packReason;
                }
            }
            finally
            {
                Volatile.Write(ref _nativeRawTried, true);
            }
        }
    }

    // Whether the raw lane takes the product: a bfloat16 run, prompt-sized, signed 8-bit codes (a scale a row or none), k a
    // multiple of 16 (each row's codes start on a 16-byte copy).
    private bool NativeRawLane(TypedDecode d, int m, int k) =>
        m > PtxKernels.TypedGemvRows[^1] && BFloat16Products && d is { Kind: 'i', Bits: 8, Signed: true } && k % 16 == 0 && NativeRaw() is not null;

    // y [m, n] = x · Wᵀ by the raw lane, its k split measured per shape.
    private bool RunNativeRaw(TypedDecode d, Storage x, Storage w, Storage? scales, Storage y, int m, int n, int k)
    {
        if (!NativeRawLane(d, m, k) || NativeRaw() is not var (kernel, packX, config))
        {
            return false;
        }

        int kp = PtxKernels.NativePadded(k);
        var shared = SharedPermuted(x, m, k, packX);                 // a group's products share one pack (SharedInput)
        var px = shared ?? Allocate(m * kp / 2, zeroed: false);
        try
        {
            if (shared is null)
            {
                LaunchPack(packX, m, kp, P(x), P(px), U(m), U(k), U(kp));
            }

            int[] splits = NativeSplits(config, kp);
            var key = new TuneKey(TuneOp.ProductRoute, 9400, TuneSizes.Class(m), n, kp, splits.Length);
            if (!TryTuned(key, splits, 1, out int split))
            {
                split = MeasureRaw(key, splits, kernel, config, px, w, scales, y, m, n, k, kp);
            }

            LaunchRaw(kernel, config, px, w, scales, y, m, n, k, kp, split);
            return true;
        }
        finally
        {
            if (shared is null)
            {
                px.Release();
            }
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private int MeasureRaw(TuneKey key, int[] splits, IntPtr kernel, PtxKernels.NativeMmaConfig config, Storage px, Storage w, Storage? scales, Storage y, int m, int n, int k, int kp) =>
        Tune(key, splits, 1, s => LaunchRaw(kernel, config, px, w, scales, y, m, n, k, kp, s));

    // The raw kernel over `splits` ranges of k's blocks (grid z): with one, C is y; with more, the parts then their sum.
    private void LaunchRaw(IntPtr kernel, PtxKernels.NativeMmaConfig config, Storage px, Storage w, Storage? scales, Storage y, int m, int n, int k, int kp, int splits)
    {
        int shared = PtxKernels.NativeMmaShared(config, raw: true);
        int blockValues = config.RowBytes * 8 / 16, blocks = kp / blockValues, per = (blocks + splits - 1) / splits;
        var target = splits == 1 ? y : _splitPartials.Get(this, splits * m * n);
        t_sharedBytes = shared > 48 * 1024 ? (uint)shared : 0;
        Launch(kernel, (uint)((n + config.Bn - 1) / config.Bn), (uint)((m + 127) / 128), (uint)splits, PtxKernels.NativeMmaThreads, 1,
            P(px), P(w), P(target), U(m), U(n), U(kp), U(kp), U(per), U(k), U(k), scales is null ? 0UL : P(scales));
        if (splits > 1)
        {
            var sum = Typed(DecodeOf(ElementType.Float32)!)!.Sum;
            Launch(sum, (uint)((m * n + 255) / 256), 1, 256, 1, P(target), P(y), U(m * n), U(splits));
        }
    }

    /// <summary>Tests: the raw int8 lane run directly (false where it does not take the product).</summary>
    internal bool StoredNativeRaw(ElementType type, Storage x, Storage weight, Storage? scales, Storage y, int m, int n, int k) =>
        DecodeOf(type) is { } d && RunNativeRaw(d, x, weight, scales, y, m, n, k);
}
