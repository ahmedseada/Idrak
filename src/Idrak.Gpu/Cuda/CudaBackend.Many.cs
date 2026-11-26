// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Cuda;

// The grouped few-row products of stored weights (StoredMatMulManyKernel): 2 or 3 weights of one description sharing x
// (queries, keys and values; gate and up) in one launch of the described lane's rows product, its _many variant (the
// column over all the weights' rows remapped to its own product's at the start, then the same body), so each output is
// exactly as its own launch computes it; and the fused epilogues of the described lane (the gate/up pair's activation, a
// product's residual and norm), each a candidate measured against the composition.
internal sealed unsafe partial class CudaBackend
{
    public override bool StoredMatMulManyKernel(Storage x, int m, int k, ReadOnlySpan<(Storage Values, Storage? Scales, Storage Output, int Columns)> products) =>
        m <= PtxKernels.TypedGemvRows[^1] && FewMany(x, m, k, products);

    // A few rows' group: of the group's one launches (the described rows product's by each warp split: DescribedFusedLane +
    // split; each output as its own launch by that lane computes it) and the split launches (each product by its own
    // measured lane: 0), the fastest measured once a shape on the real operands, cold as the products are (Tune; the
    // reference is the grouped launch of the first product's own lane, where it has one).
    private bool FewMany(Storage x, int m, int k, ReadOnlySpan<(Storage Values, Storage? Scales, Storage Output, int Columns)> products)
    {
        if (products.Length == 0 || products[0].Values.Description is not { } description)
        {
            return false;
        }

        Span<int> candidates = stackalloc int[8];
        int count = GroupCandidates(products, m, k, candidates);
        var key = GroupKey(description, m, k, products);
        int reference = GroupReference(x, m, k, products, candidates[..count]);
        if (!TryTuned(key, candidates[..count], reference, out int lane))
        {
            var array = products.ToArray();                                // measured once a shape
            lane = Tune(key, candidates[..count], reference, c => GroupLane(c, x, m, k, array), cold: true);
        }

        if (_groupsSaid.TryAdd(key, 0))
        {
            SayStored(description, m, products[0].Columns, k, string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{lane switch
            {
                0 => "each product by its own lane",
                _ => $"described lane's one launch ({1 << (lane - DescribedFusedLane)} warps a group of 8 rows)",
            }} (of {count} lanes)"), string.Create(System.Globalization.CultureInfo.InvariantCulture,
                $"a group of {products.Length} sharing x ({string.Join(", ", products.ToArray().Select(p => p.Columns))} outputs)"));
        }

        return GroupLane(lane, x, m, k, products);
    }

    // The group shapes this process has recorded (ProductRoutes.Chose).
    private readonly System.Collections.Concurrent.ConcurrentDictionary<TuneKey, byte> _groupsSaid = new();

    // A number for each description, for the groups' keys.
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<TypeDescription, int> FewGroupIds = new();

    private static TuneKey GroupKey(TypeDescription description, int m, int k, ReadOnlySpan<(Storage Values, Storage? Scales, Storage Output, int Columns)> products) =>
        new(TuneOp.ProductRoute, 7000 + FewGroupIds.GetOrAdd(description, static _ => FewGroupIds.Count), m, k, products[0].Columns, products.Length > 1 ? products[1].Columns : 0,
            products.Length > 2 ? products[2].Columns : 0, 1);

    // The group's candidates: the split launches (0) first, then the described lane's one launches where it reads the group
    // (all of one description, all with a scale an output row or none).
    private int GroupCandidates(ReadOnlySpan<(Storage Values, Storage? Scales, Storage Output, int Columns)> products, int m, int k, Span<int> into)
    {
        int count = 0;
        into[count++] = 0;
        var description = products[0].Values.Description;
        bool scaled = products[0].Scales is not null;
        if (description is null || products.Length is < 2 or > 3 || m is <= 0 or > 8 || k <= 0)
        {
            return count;
        }

        foreach (var (values, scales, _, columns) in products)
        {
            if (!Equals(values.Description, description) || scales is not null != scaled || columns <= 0)
            {
                return count;
            }
        }

        if (DescribedOf(description) is { } plan && DescribedRow(plan, description, k, out _, out int align) && (long)m * k <= int.MaxValue / 2
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
        int grouped = own is >= DescribedRowsLane and < DescribedRowsLane + 4 ? DescribedFusedLane + own - DescribedRowsLane : 0;
        return candidates.Contains(grouped) ? grouped : 0;
    }

    // One lane of a group: 0 each product by its own measured lane, DescribedFusedLane + split the described rows product's
    // one launch.
    private bool GroupLane(int lane, Storage x, int m, int k, ReadOnlySpan<(Storage Values, Storage? Scales, Storage Output, int Columns)> products) => lane switch
    {
        >= DescribedFusedLane => RunDescribedMany(x, m, k, products, lane - DescribedFusedLane),
        _ => EachRouted(x, m, k, products),
    };

    /// <summary>Tests: the lane a few rows' group settled on (GroupLane's: 0 each product by its own lane), or null where none is measured yet.</summary>
    internal int? StoredGroupChoice(Storage x, int m, int k, (Storage Values, Storage? Scales, Storage Output, int Columns)[] products)
    {
        Span<int> candidates = stackalloc int[8];
        int count = GroupCandidates(products, m, k, candidates);
        int reference = GroupReference(x, m, k, products, candidates[..count]);
        return products[0].Values.Description is { } description && TryTuned(GroupKey(description, m, k, products), candidates[..count], reference, out int lane) ? lane : null;
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

    public override bool StoredMatMulManyAvailable(int m) => m <= PtxKernels.TypedGemvRows[^1];

    // The fused epilogues are candidates, not defaults: each shape measures, once and on the real operands, the products
    // followed by the pass of its own (GatedActivation, AddRmsNormAffine: 0) against the described lane's fused pass by each
    // warp split of its rows product (DescribedFusedLane + split, where it reads the description), and a fused one runs
    // only where it is clearly faster (Tune's fallback is the composition), timed cold as the few-row products are. A number
    // for each description, for the keys.
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<TypeDescription, int> EpilogueIds = new();

    // The described lane's fused passes: DescribedFusedLane + split (log2 of the warps a group of 8 rows takes, 0 - 3).
    private const int DescribedFusedLane = 10;

    private static TuneKey EpilogueKey(int kind, int detail, TypeDescription description, int m, int n, int k) =>
        new(TuneOp.ProductRoute, (kind == 0 ? 4000 : 5000) + EpilogueIds.GetOrAdd(description, static _ => EpilogueIds.Count), m, n, k, detail, 1);

    // The epilogue's candidates at the shape (kind 0 the gate/up pair, 1 the residual and norm), the composition first.
    private int EpilogueCandidates(int kind, TypeDescription? description, int m, int k, Span<int> into)
    {
        int count = 0;
        into[count++] = 0;
        if (description is not null && DescribedFuses(kind, description, m, k))
        {
            for (int split = 3; split >= 0; split--)
            {
                into[count++] = DescribedFusedLane + split;
            }
        }

        return count;
    }

    // Whether the described lane's module has the description's fused variant for m rows (its shared memory within the device's).
    private bool DescribedFuses(int kind, TypeDescription description, int m, int k) => m is > 0 and <= 8 && DescribedOf(description) is { } plan
        && DescribedRow(plan, description, k, out _, out int align) && (long)m * k <= int.MaxValue / 2 && Described(plan, align, k % 2 == 0) is { } functions
        && (kind == 0 ? functions.RowsGated : functions.RowsAddNorm)[DescribedFewVariant(m)] != 0;

    public override bool? StoredEpilogueFused(int kind, int detail, TypeDescription description, int m, int n, int k)
    {
        if (m <= 0 || m > PtxKernels.TypedGemvRows[^1])
        {
            return false;
        }

        Span<int> candidates = stackalloc int[8];
        int count = EpilogueCandidates(kind, description, m, k, candidates);
        if (count == 1)
        {
            return false;
        }

        return TryTuned(EpilogueKey(kind, detail, description, m, n, k), candidates[..count], 0, out int lane) ? lane != 0 : null;
    }

    public override bool StoredMatMulGatedPairKernel(int activation, Storage x, int m, int k, ReadOnlySpan<(Storage Values, Storage? Scales, Storage Output, int Columns)> products,
        Storage hidden)
    {
        if (products.Length != 2 || products[0].Columns != products[1].Columns || activation is < 0 or > 2 || m <= 0 || m > PtxKernels.TypedGemvRows[^1]
            || products[0].Values.Description is not { } description)
        {
            return false;
        }

        var array = products.ToArray();
        int n = array[0].Columns;
        Span<int> candidates = stackalloc int[8];
        int count = EpilogueCandidates(0, description, m, k, candidates);
        if (count == 1)
        {
            return false;                                                  // no fused pass: the caller composes
        }

        var key = EpilogueKey(0, activation, description, m, n, k);
        if (!TryTuned(key, candidates[..count], 0, out int lane))
        {
            lane = Tune(key, candidates[..count], 0, c => GatedLane(c, activation, x, m, k, array, hidden), cold: true);
        }

        SayEpilogue(key, 0, description, m, n, k, lane, count);
        return GatedLane(lane, activation, x, m, k, array, hidden);
    }

    // The epilogue shapes this process has recorded (ProductRoutes.Chose), measured here or read from the tuning cache.
    private readonly System.Collections.Concurrent.ConcurrentDictionary<TuneKey, byte> _epiloguesSaid = new();

    private void SayEpilogue(TuneKey key, int kind, TypeDescription description, int m, int n, int k, int lane, int count)
    {
        if (_epiloguesSaid.TryAdd(key, 0))
        {
            string what = lane == 0 ? "the products, then the pass of its own" : $"described lane's fused pass ({1 << (lane - DescribedFusedLane)} warps a group of 8 rows)";
            SayStored(description, m, n, k, string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{what} (of {count} lanes)"),
                kind == 0 ? "gate/up pair, act(gate) · up" : "residual + y and its RMS norm");
        }
    }

    /// <summary>
    /// Tests: a fused gate/up pass run directly by <paramref name="lane"/> (DescribedFusedLane + split the described lane's:
    /// StoredEpilogueCandidates), not measured against the composition.
    /// </summary>
    internal bool StoredGatedFused(int activation, Storage x, int m, int k, (Storage Values, Storage? Scales, Storage Output, int Columns)[] products, Storage hidden,
        int lane = DescribedFusedLane) =>
        lane != 0 && GatedLane(lane, activation, x, m, k, products, hidden);

    /// <summary>Tests: a fused residual and norm pass run directly by <paramref name="lane"/> (as StoredGatedFused's), not measured against the composition.</summary>
    internal bool StoredAddNormFused(Storage x, Storage weight, Storage? scales, Storage y, int m, int n, int k, Storage residual, Storage sum, Storage gain,
        Storage normalized, float eps, float offset, int lane = DescribedFusedLane) =>
        lane != 0 && AddNormLane(lane, x, weight, scales, y, m, n, k, residual, sum, gain, normalized, eps, offset);

    /// <summary>Tests: the epilogue's lanes at the shape (kind 0 the gate/up pair, 1 the residual and norm), the composition (0) first.</summary>
    internal int[] StoredEpilogueCandidates(int kind, TypeDescription description, int m, int k)
    {
        Span<int> candidates = stackalloc int[8];
        return candidates[..EpilogueCandidates(kind, description, m, k, candidates)].ToArray();
    }

    /// <summary>Tests: the epilogue lane the shape settled on (detail: the activation, or whether the weight has scales), or null where none is measured yet.</summary>
    internal int? StoredEpilogueChoice(int kind, int detail, TypeDescription description, int m, int n, int k)
    {
        var candidates = StoredEpilogueCandidates(kind, description, m, k);
        return TryTuned(EpilogueKey(kind, detail, description, m, n, k), candidates, 0, out int lane) ? lane : null;
    }

    /// <summary>Tests: the product lane a fused lane's own products take (the described rows product of the fused lane's split), for StoredRoutedLane.</summary>
    internal static int StoredEpilogueProductLane(int fusedLane) => DescribedRowsLane + fusedLane - DescribedFusedLane;

    /// <summary>Tests: a stored weight's product by one of the described lane's variants (StoredRouted's candidates), not measured.</summary>
    internal bool StoredRoutedLane(int lane, Storage x, Storage weight, Storage? scales, Storage y, int m, int n, int k) =>
        RunRouted(lane, DescribedOf(weight.Description), x, weight, scales, y, m, n, k);

    // 0 the grouped products, then act(gate) · up as a pass of its own; DescribedFusedLane + split the described lane's.
    private bool GatedLane(int lane, int activation, Storage x, int m, int k, (Storage Values, Storage? Scales, Storage Output, int Columns)[] products, Storage hidden)
    {
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
        if (m <= 0 || m > PtxKernels.TypedGemvRows[^1] || k <= 0 || n <= 0 || weight.Description is not { } description)
        {
            return false;
        }

        Span<int> candidates = stackalloc int[8];
        int count = EpilogueCandidates(1, description, m, k, candidates);
        if (count == 1)
        {
            return false;                                                  // no fused pass: the caller composes
        }

        var key = EpilogueKey(1, scales is null ? 0 : 1, description, m, n, k);
        if (!TryTuned(key, candidates[..count], 0, out int lane))
        {
            lane = Tune(key, candidates[..count], 0, c => AddNormLane(c, x, weight, scales, y, m, n, k, residual, sum, gain, normalized, eps, offset), cold: true);
        }

        SayEpilogue(key, 1, description, m, n, k, lane, count);
        return AddNormLane(lane, x, weight, scales, y, m, n, k, residual, sum, gain, normalized, eps, offset);
    }

    // 0 the product (the routed few-row lane), then residual + y and its norm as a pass of its own; DescribedFusedLane +
    // split the described lane's.
    private bool AddNormLane(int lane, Storage x, Storage weight, Storage? scales, Storage y, int m, int n, int k, Storage residual, Storage sum, Storage gain,
        Storage normalized, float eps, float offset)
    {
        if (lane >= DescribedFusedLane)
        {
            return RunDescribedAddNorm(x, weight, scales, m, n, k, residual, sum, gain, normalized, eps, offset, lane - DescribedFusedLane);
        }

        if (!StoredRouted(x, weight, scales, y, m, n, k))
        {
            return false;
        }

        AddRmsNormAffine(residual, y, sum, gain, normalized, m, n, eps, offset);
        return true;
    }
}
