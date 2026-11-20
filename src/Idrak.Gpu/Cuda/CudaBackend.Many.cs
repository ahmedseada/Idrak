// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Cuda;

// The experiment's grouped few-row products of stored weights (StoredMatMulManyKernel, the routed counterpart of today's
// PackedMatMulMany): 2 or 3 weights of one type sharing x (queries, keys and values; gate and up) in one launch of the
// description's few-row kernel, its _many variant (the column over all the weights' rows remapped to its own product's
// at the start, then the same body), so each output is exactly as its own launch computes it.
internal sealed unsafe partial class CudaBackend
{
    public override bool StoredMatMulManyKernel(Storage x, int m, int k, ReadOnlySpan<(Storage Values, Storage? Scales, Storage Output, int Columns)> products) =>
        m > PtxKernels.TypedGemvRows[^1] ? SharedInput(x, m, k, products) : Grouped(x, m, k, products, null, 0);

    public override bool StoredMatMulManyAvailable(int m) => ProductRoutes.Current != ProductRoute.Name && (m <= PtxKernels.TypedGemvRows[^1] || BFloat16Products);

    // x packed once for the products sharing it (StoredMatMulManyKernel above the few-row sizes), while they run: each
    // product takes its own measured lane, and the native lanes read these packs (RunNative and SuperNative the kind's rows,
    // RunNativeRaw the permuted bfloat16 rows its fragments read) instead of packing x again. Each pack is made when a
    // product's lane first asks for it, so a group whose lanes read x as it is packs nothing.
    private SharedX? _sharedX;

    private sealed class SharedX(Storage x, int m, int k, PtxKernels.NativeKind kind, IntPtr pack)
    {
        public Storage X { get; } = x;

        public int M { get; } = m;

        public int K { get; } = k;

        public PtxKernels.NativeKind Kind { get; } = kind;

        // The kind's packing kernel of float32 x.
        public IntPtr Pack { get; } = pack;

        public Storage? Packed { get; set; }

        public Storage? Permuted { get; set; }

        public bool Holds(Storage x, int m, int k) => ReferenceEquals(X, x) && M == m && K == k;

        public void Release()
        {
            Packed?.Release();
            Permuted?.Release();
        }
    }

    private bool SharedInput(Storage x, int m, int k, ReadOnlySpan<(Storage Values, Storage? Scales, Storage Output, int Columns)> products)
    {
        // x packed in the kind the group's weights are multiplied in (NativeKind.For: one kind for all, else no shared pack).
        var kinds = new HashSet<PtxKernels.NativeKind>();
        foreach (var product in products)
        {
            kinds.Add(DecodeOf(product.Values.ElementType) is { } typed ? PtxKernels.NativeKind.For(typed) : PtxKernels.NativeKind.BF16);
        }

        var kind = kinds.Count == 1 ? kinds.First() : null;
        if (kind is null || ProductRoutes.Current == ProductRoute.Name || !BFloat16Products || products.Length is < 2 or > 3 || k % 8 != 0 || NativeKernel(kind) is null
            || NativePack(DecodeOf(ElementType.Float32)!, kind) is not { } pack)
        {
            return false;
        }

        var shared = new SharedX(x, m, k, kind, pack.X);
        var list = products.ToArray();                                     // prompt-sized: one small copy per call
        bool EachOwn()
        {
            foreach (var (values, scales, output, columns) in list)
            {
                bool done = values.ElementType.Super is not null || values.ElementType.Grid is not null ? StoredSuper(x, values, output, m, columns, k) : StoredMeasured(x, values, scales, output, m, columns, k);
                if (!done)
                {
                    return false;                                    // the caller runs them one by one
                }
            }

            return true;
        }

        try
        {
            _sharedX = shared;
            return TensorRowsGroup(x, m, k, list, EachOwn);          // one launch over the group, where it measures faster
        }
        finally
        {
            _sharedX = null;
            shared.Release();
        }
    }

    // The shared pack of x where it is this product's input in its kind (SharedInput), packed on the first ask; else null.
    private Storage? SharedPack(Storage x, int m, int k, PtxKernels.NativeKind kind)
    {
        if (_sharedX is not { } shared || shared.Kind != kind || !shared.Holds(x, m, k))
        {
            return null;
        }

        if (shared.Packed is null)
        {
            int kp = PtxKernels.NativePadded(k);
            var px = Allocate(m * kp / 2, zeroed: false);
            LaunchPack(shared.Pack, m, kp, P(x), P(px), U(m), U(k), U(kp));
            shared.Packed = px;
        }

        return shared.Packed;
    }

    // The shared permuted bfloat16 pack of x for the raw int8 lane (RunNativeRaw) where x is the group's, packed on the first
    // ask by `packPermuted`; else null.
    private Storage? SharedPermuted(Storage x, int m, int k, IntPtr packPermuted)
    {
        if (_sharedX is not { } shared || !shared.Holds(x, m, k))
        {
            return null;
        }

        if (shared.Permuted is null)
        {
            int kp = PtxKernels.NativePadded(k);
            var px = Allocate(m * kp / 2, zeroed: false);
            LaunchPack(packPermuted, m, kp, P(x), P(px), U(m), U(k), U(kp));
            shared.Permuted = px;
        }

        return shared.Permuted;
    }

    // The fused epilogues are candidates, not defaults: each shape measures, once and on the real operands,
    // the fused pass against the products followed by the pass of its own (GatedActivation, AddRmsNormAffine), and the
    // fused one runs only where it is clearly faster (Tune's fallback is the composition). A number for each type, for the keys.
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, int> EpilogueIds = new(StringComparer.Ordinal);

    private static TuneKey EpilogueKey(int kind, int detail, ElementType type, int m, int n, int k) =>
        new(TuneOp.ProductRoute, (kind == 0 ? 4000 : 5000) + EpilogueIds.GetOrAdd(type.Name, static _ => EpilogueIds.Count), m, n, k, detail);

    public override bool? StoredEpilogueFused(int kind, int detail, ElementType type, int m, int n, int k)
    {
        if (ProductRoutes.Current == ProductRoute.Name || m <= 0 || m > PtxKernels.TypedGemvRows[^1])
        {
            return false;
        }

        return TryTuned(EpilogueKey(kind, detail, type, m, n, k), [0, 1], 0, out int lane) ? lane == 1 : null;
    }

    public override bool StoredMatMulGatedPairKernel(int activation, Storage x, int m, int k, ReadOnlySpan<(Storage Values, Storage? Scales, Storage Output, int Columns)> products,
        Storage hidden)
    {
        if (products.Length != 2 || products[0].Columns != products[1].Columns || activation is < 0 or > 2)
        {
            return false;
        }

        var array = products.ToArray();
        int n = array[0].Columns;
        var key = EpilogueKey(0, activation, array[0].Values.ElementType, m, n, k);
        if (!TryTuned(key, [0, 1], 0, out int lane))
        {
            lane = Tune(key, [0, 1], 0, c => GatedLane(c, activation, x, m, k, array, hidden));
        }

        return GatedLane(lane, activation, x, m, k, array, hidden);
    }

    /// <summary>Tests: the fused gate/up pass run directly (not measured against the composition).</summary>
    internal bool StoredGatedFused(int activation, Storage x, int m, int k, (Storage Values, Storage? Scales, Storage Output, int Columns)[] products, Storage hidden) =>
        GatedLane(1, activation, x, m, k, products, hidden);

    /// <summary>Tests: the fused residual and norm pass run directly (not measured against the composition).</summary>
    internal bool StoredAddNormFused(Storage x, Storage weight, Storage? scales, Storage y, int m, int n, int k, Storage residual, Storage sum, Storage gain,
        Storage normalized, float eps, float offset) =>
        AddNormLane(1, x, weight, scales, y, m, n, k, residual, sum, gain, normalized, eps, offset);

    // 1 the fused pass; 0 the grouped products, then act(gate) · up as a pass of its own.
    private bool GatedLane(int lane, int activation, Storage x, int m, int k, (Storage Values, Storage? Scales, Storage Output, int Columns)[] products, Storage hidden)
    {
        if (lane == 1)
        {
            return Grouped(x, m, k, products, hidden, activation);
        }

        if (!Grouped(x, m, k, products, null, 0))
        {
            return false;
        }

        GatedActivation(products[0].Output, products[1].Output, hidden, m * products[0].Columns, activation);
        return true;
    }

    // The fused epilogues' count of finished blocks (one word, zero between launches: the last block resets it).
    private Storage? _epilogueCount;

    private ulong EpilogueCount() => P(_epilogueCount ??= Allocate(1, zeroed: true));

    public override bool StoredMatMulAddRmsNormKernel(Storage x, Storage weight, Storage? scales, Storage y, int m, int n, int k, Storage residual, Storage sum, Storage gain,
        Storage normalized, float eps, float offset)
    {
        if (ProductRoutes.Current == ProductRoute.Name || m <= 0 || m > PtxKernels.TypedGemvRows[^1] || k <= 0 || n <= 0)
        {
            return false;
        }

        var key = EpilogueKey(1, scales is null ? 0 : 1, weight.ElementType, m, n, k);
        if (!TryTuned(key, [0, 1], 0, out int lane))
        {
            lane = Tune(key, [0, 1], 0, c => AddNormLane(c, x, weight, scales, y, m, n, k, residual, sum, gain, normalized, eps, offset));
        }

        return AddNormLane(lane, x, weight, scales, y, m, n, k, residual, sum, gain, normalized, eps, offset);
    }

    // 1 the fused pass; 0 the product (the routed few-row lane), then residual + y and its norm as a pass of its own.
    private bool AddNormLane(int lane, Storage x, Storage weight, Storage? scales, Storage y, int m, int n, int k, Storage residual, Storage sum, Storage gain,
        Storage normalized, float eps, float offset)
    {
        if (lane == 0)
        {
            bool product = weight.ElementType.Super is not null || weight.ElementType.Grid is not null ? StoredSuper(x, weight, y, m, n, k) : StoredMeasured(x, weight, scales, y, m, n, k);
            if (!product)
            {
                return false;
            }

            AddRmsNormAffine(residual, y, sum, gain, normalized, m, n, eps, offset);
            return true;
        }

        return FusedAddNorm(x, weight, scales, y, m, n, k, residual, sum, gain, normalized, eps, offset);
    }

    private bool FusedAddNorm(Storage x, Storage weight, Storage? scales, Storage y, int m, int n, int k, Storage residual, Storage sum, Storage gain,
        Storage normalized, float eps, float offset)
    {

        var type = weight.ElementType;
        if (type.Super is not null || type.Grid is not null)
        {
            if (scales is not null || Decoded(type) is not var (f, blockValues) || k % blockValues != 0)
            {
                return false;
            }

            Launch(f.GemvAddNorm[TypedRowVariant(m)], (uint)((n + 7) / 8), 1, 256, 1, P(x), P(weight), P(y), U(m), U(n), U(k),
                P(residual), P(sum), P(gain), P(normalized), F(eps), F(offset), EpilogueCount());
            return true;
        }

        if (DecodeOf(type) is not { } d || (d.PerValue ? (long)k * d.Bits % 32 != 0 : k % PtxKernels.TypedTileDepth != 0 || scales is not null) || Typed(d) is not { } functions)
        {
            return false;
        }

        bool wide = d.PerValue && (long)k * d.Bits / 8 % 16 == 0;
        Launch((wide ? functions.GemvWideAddNorm : functions.GemvAddNorm)[TypedRowVariant(m)], (uint)((n + PtxKernels.TypedWarpColumns - 1) / PtxKernels.TypedWarpColumns), 1,
            (uint)(PtxKernels.TypedWarpColumns * 32), 1, P(x), P(weight), scales is null ? 0UL : P(scales), P(y), U(m), U(n), U(k),
            P(residual), P(sum), P(gain), P(normalized), F(eps), F(offset), EpilogueCount());
        return true;
    }

    // 2 or 3 products in one launch; with `hidden`, the pair's act(gate) · up written by the last block too.
    private bool Grouped(Storage x, int m, int k, ReadOnlySpan<(Storage Values, Storage? Scales, Storage Output, int Columns)> products, Storage? hidden, int activation)
    {
        if (ProductRoutes.Current == ProductRoute.Name || products.Length is < 2 or > 3 || m <= 0 || m > PtxKernels.TypedGemvRows[^1] || k <= 0)
        {
            return false;
        }

        var type = products[0].Values.ElementType;
        bool scaled = products[0].Scales is not null;
        int total = 0;
        Span<ulong> w = stackalloc ulong[3], s = stackalloc ulong[3], y = stackalloc ulong[3], n = stackalloc ulong[3];
        for (int i = 0; i < products.Length; i++)
        {
            var p = products[i];
            if (p.Values.ElementType != type || p.Scales is not null != scaled || p.Columns <= 0)
            {
                return false;
            }

            (w[i], s[i], y[i], n[i]) = (P(p.Values), p.Scales is null ? 0UL : P(p.Scales), P(p.Output), U(p.Columns));
            total += p.Columns;
        }

        if (type.Super is not null || type.Grid is not null)
        {
            if (scaled || Decoded(type) is not var (f, blockValues) || k % blockValues != 0)
            {
                return false;
            }

            if (hidden is not null)
            {
                Launch(f.GemvGated[TypedRowVariant(m)], (uint)((total + 7) / 8), 1, 256, 1,
                    P(x), w[0], y[0], U(m), n[0], U(k), w[1], y[1], n[1], w[2], y[2], n[2], P(hidden), EpilogueCount(), U(activation));
                return true;
            }

            Launch(f.GemvMany[TypedRowVariant(m)], (uint)((total + 7) / 8), 1, 256, 1,
                P(x), w[0], y[0], U(m), n[0], U(k), w[1], y[1], n[1], w[2], y[2], n[2]);
            return true;
        }

        if (DecodeOf(type) is not { } d || (d.PerValue ? (long)k * d.Bits % 32 != 0 : k % PtxKernels.TypedTileDepth != 0 || scaled) || Typed(d) is not { } functions)
        {
            return false;
        }

        bool wide = d.PerValue && (long)k * d.Bits / 8 % 16 == 0;
        uint grid = (uint)((total + PtxKernels.TypedWarpColumns - 1) / PtxKernels.TypedWarpColumns), threads = (uint)(PtxKernels.TypedWarpColumns * 32);
        if (hidden is not null)
        {
            Launch((wide ? functions.GemvWideGated : functions.GemvGated)[TypedRowVariant(m)], grid, 1, threads, 1,
                P(x), w[0], s[0], y[0], U(m), n[0], U(k), w[1], s[1], y[1], n[1], w[2], s[2], y[2], n[2], P(hidden), EpilogueCount(), U(activation));
            return true;
        }

        Launch((wide ? functions.GemvOutputsWideMany : functions.GemvOutputsMany)[TypedRowVariant(m)], grid, 1, threads, 1,
            P(x), w[0], s[0], y[0], U(m), n[0], U(k), w[1], s[1], y[1], n[1], w[2], s[2], y[2], n[2]);
        return true;
    }
}
