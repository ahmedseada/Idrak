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
        m > PtxKernels.TypedGemvRows[^1] ? SharedInput(x, m, k, products) : FewMany(x, m, k, products);

    // Whether the measured route's products of `type` take the written kernels only where the setting does (ProductRoutes.Lanes):
    // false with the described lane alone where it reads the type.
    private static bool WrittenTaken(ElementType type) =>
        ProductRoutes.Current != ProductRoute.Measured || ProductRoutes.Lanes != RouteLanes.Described || DescribedOf(type) is null;

    // A few rows' group under the measured route: of the group's one launches (the written few-row kernels' Grouped: 1; the
    // described rows product's by each warp split: DescribedFusedLane + split; each output as its own launch by that lane
    // computes it) and the split launches (each product by its own measured lane: 0), the fastest measured once a shape on
    // the real operands, cold as the products are (Tune; the reference is the grouped launch of the first product's own
    // lane, where it has one). Each product's own choice alone split q6_k's queries, keys and values and its gate and up
    // apart, slower than one launch (305 tokens a second against 334 with the described lane alone). Other routes, the
    // written one.
    private bool FewMany(Storage x, int m, int k, ReadOnlySpan<(Storage Values, Storage? Scales, Storage Output, int Columns)> products)
    {
        var type = products[0].Values.ElementType;
        if (ProductRoutes.Current != ProductRoute.Measured)
        {
            return Grouped(x, m, k, products, null, 0);
        }

        Span<int> candidates = stackalloc int[8];
        int count = GroupCandidates(products, m, k, candidates);
        var key = GroupKey(type, m, k, products);
        int reference = GroupReference(x, m, k, products, candidates[..count]);
        if (!TryTuned(key, candidates[..count], reference, out int lane))
        {
            var array = products.ToArray();                                // measured once a shape
            lane = Tune(key, candidates[..count], reference, c => GroupLane(c, x, m, k, array), cold: true);
        }

        if (_groupsSaid.TryAdd(key, 0))
        {
            SayStored(type, m, products[0].Columns, k, string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{lane switch
            {
                0 => "each product by its own lane",
                1 => "written kernels' one launch",
                _ => $"described lane's one launch ({1 << (lane - DescribedFusedLane)} warps a group of 8 rows)",
            }} (of {count} lanes)"), string.Create(System.Globalization.CultureInfo.InvariantCulture,
                $"a group of {products.Length} sharing x ({string.Join(", ", products.ToArray().Select(p => p.Columns))} outputs)"));
        }

        return GroupLane(lane, x, m, k, products);
    }

    // The group shapes this process has recorded (ProductRoutes.Chose).
    private readonly System.Collections.Concurrent.ConcurrentDictionary<TuneKey, byte> _groupsSaid = new();

    // A number for each type, for the groups' keys.
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, int> FewGroupIds = new(StringComparer.Ordinal);

    private static TuneKey GroupKey(ElementType type, int m, int k, ReadOnlySpan<(Storage Values, Storage? Scales, Storage Output, int Columns)> products) =>
        new(TuneOp.ProductRoute, 7000 + FewGroupIds.GetOrAdd(type.Name, static _ => FewGroupIds.Count), m, k, products[0].Columns, products.Length > 1 ? products[1].Columns : 0,
            products.Length > 2 ? products[2].Columns : 0, 1 + (int)ProductRoutes.Lanes);

    // The group's candidates: the split launches (0) first, then each one launch the setting takes that reads the group
    // (all of one type, all with a scale an output row or none).
    private int GroupCandidates(ReadOnlySpan<(Storage Values, Storage? Scales, Storage Output, int Columns)> products, int m, int k, Span<int> into)
    {
        int count = 0;
        into[count++] = 0;
        var type = products[0].Values.ElementType;
        bool scaled = products[0].Scales is not null;
        if (products.Length is < 2 or > 3 || m is <= 0 or > 8 || k <= 0)
        {
            return count;
        }

        foreach (var (values, scales, _, columns) in products)
        {
            if (values.ElementType != type || scales is not null != scaled || columns <= 0)
            {
                return count;
            }
        }

        if (WrittenTaken(type) && WrittenFuses(type, scaled, k))
        {
            into[count++] = 1;
        }

        if (DescribedTaken(type) && DescribedOf(type) is { } plan && DescribedRow(plan, type, k, out _, out int align) && (long)m * k <= int.MaxValue / 2
            && Described(plan, align, k % 2 == 0) is not null)
        {
            for (int split = 3; split >= 0; split--)
            {
                into[count++] = DescribedFusedLane + split;
            }
        }

        return count;
    }

    // The group's reference: the one launch of the first product's own lane (RoutedLane) where the group has it, else the
    // split launches.
    private int GroupReference(Storage x, int m, int k, ReadOnlySpan<(Storage Values, Storage? Scales, Storage Output, int Columns)> products, ReadOnlySpan<int> candidates)
    {
        if (candidates.Length == 1)
        {
            return 0;
        }

        var first = products[0];
        int own = RoutedLane(x, first.Values, first.Scales, first.Output, m, first.Columns, k) ?? -1;
        int grouped = own is 0 or SuperFewLane ? 1 : own is >= DescribedRowsLane and < DescribedRowsLane + 4 ? DescribedFusedLane + own - DescribedRowsLane : 0;
        return candidates.Contains(grouped) ? grouped : 0;
    }

    // One lane of a group: 0 each product by its own measured lane, 1 the written kernels' one launch, DescribedFusedLane +
    // split the described rows product's one launch.
    private bool GroupLane(int lane, Storage x, int m, int k, ReadOnlySpan<(Storage Values, Storage? Scales, Storage Output, int Columns)> products) => lane switch
    {
        1 => Grouped(x, m, k, products, null, 0),
        >= DescribedFusedLane => RunDescribedMany(x, m, k, products, lane - DescribedFusedLane),
        _ => EachRouted(x, m, k, products),
    };

    /// <summary>Tests: the lane a few rows' group settled on (GroupLane's: 0 each product by its own lane), or null where none is measured yet.</summary>
    internal int? StoredGroupChoice(Storage x, int m, int k, (Storage Values, Storage? Scales, Storage Output, int Columns)[] products)
    {
        Span<int> candidates = stackalloc int[8];
        int count = GroupCandidates(products, m, k, candidates);
        int reference = GroupReference(x, m, k, products, candidates[..count]);
        return TryTuned(GroupKey(products[0].Values.ElementType, m, k, products), candidates[..count], reference, out int lane) ? lane : null;
    }

    // Each product of a group by its own measured lane (StoredRouted); false where one is not taken so (the caller runs them all).
    private bool EachRouted(Storage x, int m, int k, ReadOnlySpan<(Storage Values, Storage? Scales, Storage Output, int Columns)> products)
    {
        foreach (var (values, scales, output, columns) in products)
        {
            if (!StoredRouted(x, values, scales, output, m, columns, k))
            {
                return false;
            }
        }

        return true;
    }

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
        bool measured = ProductRoutes.Current == ProductRoute.Measured;
        bool EachOwn()
        {
            foreach (var (values, scales, output, columns) in list)
            {
                // Under the measured route each by its own measured lane (StoredRouted: none widening), else as before.
                bool done = measured && StoredRouted(x, values, scales, output, m, columns, k)
                    || (values.ElementType.Super is not null || values.ElementType.Grid is not null ? StoredSuper(x, values, output, m, columns, k) : StoredMeasured(x, values, scales, output, m, columns, k));
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
            // The tensor rows' one launch over the group (written; its outputs the bits of each product's tensor rows): under the
            // measured route only where every product takes the tensor rows alone (RoutedLane), else each its own lane; it
            // runs where it measures faster.
            return !measured || EachTakes(x, m, k, list, 6) ? TensorRowsGroup(x, m, k, list, EachOwn) : EachOwn();
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

    // The fused epilogues are candidates, not defaults: each shape measures, once and on the real operands, the products
    // followed by the pass of its own (GatedActivation, AddRmsNormAffine: 0) against each fused pass the route takes: the
    // written kernels' (1, where the setting takes them and they read the type) and the described lane's by each warp split
    // of its rows product (DescribedFusedLane + split, where the setting takes the described lane and it reads the type), and
    // a fused one runs only where it is clearly faster (Tune's fallback is the composition), timed cold as the few-row products are
    // (a decoding step reads each weight from memory; warm timings of these 10 us passes chose other splits run to run). A
    // number for each type, for the keys.
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, int> EpilogueIds = new(StringComparer.Ordinal);

    // The described lane's fused passes: DescribedFusedLane + split (log2 of the warps a group of 8 rows takes, 0 - 3).
    private const int DescribedFusedLane = 10;

    private static TuneKey EpilogueKey(int kind, int detail, ElementType type, int m, int n, int k) =>
        new(TuneOp.ProductRoute, (kind == 0 ? 4000 : 5000) + EpilogueIds.GetOrAdd(type.Name, static _ => EpilogueIds.Count), m, n, k, detail, 1 + (int)ProductRoutes.Lanes);

    // Whether the measured route's products of `type` take the described lane (ProductRoutes.Lanes) and it reads the type.
    private static bool DescribedTaken(ElementType type) =>
        ProductRoutes.Current == ProductRoute.Measured && ProductRoutes.Lanes != RouteLanes.Written && DescribedOf(type) is not null;

    // The epilogue's candidates at the shape (kind 0 the gate/up pair, 1 the residual and norm), the composition first.
    private int EpilogueCandidates(int kind, ElementType type, bool scaled, int m, int k, Span<int> into)
    {
        int count = 0;
        into[count++] = 0;
        if (WrittenTaken(type) && WrittenFuses(type, scaled, k))
        {
            into[count++] = 1;
        }

        if (DescribedTaken(type) && DescribedFuses(kind, type, m, k))
        {
            for (int split = 3; split >= 0; split--)
            {
                into[count++] = DescribedFusedLane + split;
            }
        }

        return count;
    }

    // Whether the written few-row kernels have the type's fused variants (Grouped's and FusedAddNorm's).
    private bool WrittenFuses(ElementType type, bool scaled, int k) => type.Super is not null || type.Grid is not null
        ? !scaled && Decoded(type) is var (_, values) && k % values == 0
        : DecodeOf(type) is { } d && (d.PerValue ? (long)k * d.Bits % 32 == 0 : k % PtxKernels.TypedTileDepth == 0 && !scaled) && Typed(d) is not null;

    // Whether the described lane's module has the type's fused variant for m rows (its shared memory within the device's).
    private bool DescribedFuses(int kind, ElementType type, int m, int k) => m is > 0 and <= 8 && DescribedOf(type) is { } plan
        && DescribedRow(plan, type, k, out _, out int align) && (long)m * k <= int.MaxValue / 2 && Described(plan, align, k % 2 == 0) is { } functions
        && (kind == 0 ? functions.RowsGated : functions.RowsAddNorm)[DescribedFewVariant(m)] != 0;

    public override bool? StoredEpilogueFused(int kind, int detail, ElementType type, int m, int n, int k)
    {
        if (ProductRoutes.Current == ProductRoute.Name || m <= 0 || m > PtxKernels.TypedGemvRows[^1])
        {
            return false;
        }

        Span<int> candidates = stackalloc int[8];
        int count = EpilogueCandidates(kind, type, kind == 1 && detail != 0, m, k, candidates);
        if (count == 1)
        {
            return false;
        }

        return TryTuned(EpilogueKey(kind, detail, type, m, n, k), candidates[..count], 0, out int lane) ? lane != 0 : null;
    }

    public override bool StoredMatMulGatedPairKernel(int activation, Storage x, int m, int k, ReadOnlySpan<(Storage Values, Storage? Scales, Storage Output, int Columns)> products,
        Storage hidden)
    {
        if (products.Length != 2 || products[0].Columns != products[1].Columns || activation is < 0 or > 2 || ProductRoutes.Current == ProductRoute.Name
            || m <= 0 || m > PtxKernels.TypedGemvRows[^1])
        {
            return false;
        }

        var array = products.ToArray();
        int n = array[0].Columns;
        Span<int> candidates = stackalloc int[8];
        int count = EpilogueCandidates(0, array[0].Values.ElementType, array[0].Scales is not null, m, k, candidates);
        if (count == 1)
        {
            return false;                                                  // no fused pass: the caller composes
        }

        var key = EpilogueKey(0, activation, array[0].Values.ElementType, m, n, k);
        if (!TryTuned(key, candidates[..count], 0, out int lane))
        {
            lane = Tune(key, candidates[..count], 0, c => GatedLane(c, activation, x, m, k, array, hidden), cold: true);
        }

        SayEpilogue(key, 0, array[0].Values.ElementType, m, n, k, lane, count);
        return GatedLane(lane, activation, x, m, k, array, hidden);
    }

    // The epilogue shapes this process has recorded (ProductRoutes.Chose), measured here or read from the tuning cache.
    private readonly System.Collections.Concurrent.ConcurrentDictionary<TuneKey, byte> _epiloguesSaid = new();

    private void SayEpilogue(TuneKey key, int kind, ElementType type, int m, int n, int k, int lane, int count)
    {
        if (_epiloguesSaid.TryAdd(key, 0))
        {
            string what = lane switch
            {
                0 => "the products, then the pass of its own",
                1 => "written kernels' fused pass",
                _ => $"described lane's fused pass ({1 << (lane - DescribedFusedLane)} warps a group of 8 rows)",
            };
            SayStored(type, m, n, k, string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{what} (of {count} lanes)"),
                kind == 0 ? "gate/up pair, act(gate) · up" : "residual + y and its RMS norm");
        }
    }

    /// <summary>
    /// Tests: a fused gate/up pass run directly by <paramref name="lane"/> (1 the written kernels', DescribedFusedLane +
    /// split the described lane's: StoredEpilogueCandidates), not measured against the composition.
    /// </summary>
    internal bool StoredGatedFused(int activation, Storage x, int m, int k, (Storage Values, Storage? Scales, Storage Output, int Columns)[] products, Storage hidden,
        int lane = 1) =>
        lane != 0 && GatedLane(lane, activation, x, m, k, products, hidden);

    /// <summary>Tests: a fused residual and norm pass run directly by <paramref name="lane"/> (as StoredGatedFused's), not measured against the composition.</summary>
    internal bool StoredAddNormFused(Storage x, Storage weight, Storage? scales, Storage y, int m, int n, int k, Storage residual, Storage sum, Storage gain,
        Storage normalized, float eps, float offset, int lane = 1) =>
        lane != 0 && AddNormLane(lane, x, weight, scales, y, m, n, k, residual, sum, gain, normalized, eps, offset);

    /// <summary>Tests: the epilogue's lanes at the shape (kind 0 the gate/up pair, 1 the residual and norm), the composition (0) first.</summary>
    internal int[] StoredEpilogueCandidates(int kind, ElementType type, bool scaled, int m, int k)
    {
        Span<int> candidates = stackalloc int[8];
        return candidates[..EpilogueCandidates(kind, type, scaled, m, k, candidates)].ToArray();
    }

    /// <summary>Tests: the epilogue lane the shape settled on (detail: the activation, or whether the weight has scales), or null where none is measured yet.</summary>
    internal int? StoredEpilogueChoice(int kind, int detail, ElementType type, bool scaled, int m, int n, int k)
    {
        var candidates = StoredEpilogueCandidates(kind, type, scaled, m, k);
        return TryTuned(EpilogueKey(kind, detail, type, m, n, k), candidates, 0, out int lane) ? lane : null;
    }

    /// <summary>
    /// Tests: the product lane a fused lane's own products take (the written few-row kernel: 0, or the super-block one; the
    /// described rows product of the fused lane's split), for StoredRoutedLane.
    /// </summary>
    internal static int StoredEpilogueProductLane(ElementType type, int fusedLane) => fusedLane >= DescribedFusedLane ? DescribedRowsLane + fusedLane - DescribedFusedLane
        : type.Super is not null || type.Grid is not null ? SuperFewLane : 0;

    /// <summary>Tests: a stored weight's product by one of the measured route's lanes (StoredRouted's candidates), not measured.</summary>
    internal bool StoredRoutedLane(int lane, Storage x, Storage weight, Storage? scales, Storage y, int m, int n, int k) =>
        RunRouted(lane, DescribedOf(weight.ElementType), x, weight, scales, y, m, n, k);

    // 0 the grouped products, then act(gate) · up as a pass of its own; 1 the written kernels' fused pass; DescribedFusedLane
    // + split the described lane's.
    private bool GatedLane(int lane, int activation, Storage x, int m, int k, (Storage Values, Storage? Scales, Storage Output, int Columns)[] products, Storage hidden)
    {
        if (lane == 1)
        {
            return Grouped(x, m, k, products, hidden, activation);
        }

        if (lane >= DescribedFusedLane)
        {
            return RunDescribedGated(activation, x, m, k, products, hidden, lane - DescribedFusedLane);
        }

        if (!FewMany(x, m, k, products))
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

        Span<int> candidates = stackalloc int[8];
        int count = EpilogueCandidates(1, weight.ElementType, scales is not null, m, k, candidates);
        if (count == 1)
        {
            return false;                                                  // no fused pass: the caller composes
        }

        var key = EpilogueKey(1, scales is null ? 0 : 1, weight.ElementType, m, n, k);
        if (!TryTuned(key, candidates[..count], 0, out int lane))
        {
            lane = Tune(key, candidates[..count], 0, c => AddNormLane(c, x, weight, scales, y, m, n, k, residual, sum, gain, normalized, eps, offset), cold: true);
        }

        SayEpilogue(key, 1, weight.ElementType, m, n, k, lane, count);
        return AddNormLane(lane, x, weight, scales, y, m, n, k, residual, sum, gain, normalized, eps, offset);
    }

    // 0 the product (the routed few-row lane), then residual + y and its norm as a pass of its own; 1 the written kernels'
    // fused pass; DescribedFusedLane + split the described lane's.
    private bool AddNormLane(int lane, Storage x, Storage weight, Storage? scales, Storage y, int m, int n, int k, Storage residual, Storage sum, Storage gain,
        Storage normalized, float eps, float offset)
    {
        if (lane >= DescribedFusedLane)
        {
            return RunDescribedAddNorm(x, weight, scales, m, n, k, residual, sum, gain, normalized, eps, offset, lane - DescribedFusedLane);
        }

        if (lane == 0)
        {
            bool product = ProductRoutes.Current == ProductRoute.Measured && StoredRouted(x, weight, scales, y, m, n, k)
                || (weight.ElementType.Super is not null || weight.ElementType.Grid is not null ? StoredSuper(x, weight, y, m, n, k) : StoredMeasured(x, weight, scales, y, m, n, k));
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
