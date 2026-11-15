// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Globalization;

namespace Idrak.Gpu.Cuda;

// The experiment's tensor-row lane (6; the super-block and grid formats' 3): today's tensor-core product
// (PtxKernels.TensorCore.cs, register-staged 128 × 128 tiles, x read as float32 and rounded in the kernel, k split in the
// kernel with the splits' ordered sum) over the routes' weights a row an output: read as they are stored where the kernel
// reads them (bfloat16 pairs, …_nt_bf16w_f32; int8 codes with a scale an output, …_nt_int8w_f32), else widened into
// bfloat16 rows for the call (a typed description's pack kernel, a super-block or grid description's widening) and read
// as bfloat16 rows: no packing pass over x, no separate pass adding k splits. Float16 keeps its own kind (exact there,
// rounded here). A candidate of the measured lanes (StoredMeasured, StoredSuper), taken only where it measures faster; a
// group of products sharing x also measures one launch over all of them (…_multi_f32: their column tiles side by side, as
// today's grouped products run) against each product's own lane.
internal sealed unsafe partial class CudaBackend
{
    // [bfloat16 ? 0 : 1, multi ? 1 : 0]
    private static readonly string[] TensorRowsKernels = ["gemm_tc_nt_bf16w_f32", "gemm_tc_nt_bf16w_multi_f32", "gemm_tc_nt_int8w_f32", "gemm_tc_nt_int8w_multi_f32"];

    // How a type's weights reach the tensor rows: the kernels (one product, a group), whether the rows are int8 codes with a
    // scale an output, the bytes of a block of 32 where they are scaled blocks read as stored (0: none), and the widening
    // into bfloat16 rows of kp (0: read as stored; Super: a super-block or grid description's, else a typed description's
    // pack kernel). Variant: what the k splits are kept under; Label: the profile's name.
    private readonly record struct RowsSource(IntPtr Kernel, IntPtr Multi, bool Int8, int BlockBytes, IntPtr Widen, bool Super, int Variant, string Label);

    // The source of `type`'s weights (with `scales`, k values a row) for a prompt of m rows; null where the lane does not
    // take them: a few rows, no bfloat16 products, k under one tile, float16, or no kernel loaded.
    private RowsSource? TensorRowsSource(ElementType type, Storage? scales, int m, int k)
    {
        if (m <= PtxKernels.TypedGemvRows[^1] || !BFloat16Products || k < 32 || TensorKernel(TensorRowsKernels[0]) is not { } bf16
            || TensorKernel(TensorRowsKernels[1]) is not { } bf16Multi)
        {
            return null;
        }

        if (type.Super is not null || type.Grid is not null)
        {
            return scales is null && k % 2 == 0 && Decoded(type) is var (f, values) && k % values == 0
                ? new RowsSource(bf16, bf16Multi, false, 0, f.WidenBf16, true, 42, "bf16w widened") : null;
        }

        if (DecodeOf(type) is not { } d || PtxKernels.NativeKind.F16.Holds(d))
        {
            return null;
        }

        if (PtxKernels.NativeKind.BF16.Holds(d))
        {
            return scales is null && k % 2 == 0 ? new RowsSource(bf16, bf16Multi, false, 0, 0, false, 40, "bf16w") : null;
        }

        if (d is { Kind: 'i', Bits: 8, Signed: true } && scales is not null)
        {
            return k % 4 == 0 && TensorKernel(TensorRowsKernels[2]) is { } int8 && TensorKernel(TensorRowsKernels[3]) is { } int8Multi
                ? new RowsSource(int8, int8Multi, true, 0, 0, false, 41, "int8w") : null;
        }

        // Scaled blocks of 32 the kernel decodes as its tiles load (one block a row of a k tile): read as stored.
        if (d.Block is { } block && scales is null && k % 32 == 0 && PtxKernels.BlockRows(block) && BlockRowsKernels(d) is var (one, many))
        {
            return new RowsSource(one, many, false, block.Bytes, 0, false, 100 + GroupIds.GetOrAdd(type.Name, static _ => GroupIds.Count), d.Code);
        }

        // Any other description the native lane widens (its conditions): widened rows of kp, then read as bfloat16.
        return MatrixUnitLane(d, m, k) && (scales is null || d.PerValue) && k % 8 == 0 && NativePack(d, PtxKernels.NativeKind.BF16) is { } pack
            ? new RowsSource(bf16, bf16Multi, false, 0, pack.W, false, 42, "bf16w widened") : null;
    }

    // Each block description's tensor-row kernels (one product, a group), loaded on first use: null where the driver
    // refuses the module (TypedUnavailableReason says why).
    private readonly ConcurrentDictionary<TypedDecode, (IntPtr One, IntPtr Many)?> _blockRows = new();

    private (IntPtr One, IntPtr Many)? BlockRowsKernels(TypedDecode d)
    {
        if (_blockRows.TryGetValue(d, out var known))
        {
            return known;
        }

        lock (_typedLock)
        {
            if (_blockRows.TryGetValue(d, out known))
            {
                return known;
            }

            var (names, source) = PtxKernels.TensorRowsBlockModule(d);
            string one = string.Create(CultureInfo.InvariantCulture, $"gemm_tc_nt_{d.Code}_f32"), many = string.Create(CultureInfo.InvariantCulture, $"gemm_tc_nt_{d.Code}_multi_f32");
            var kernels = LoadDynamicModule("tensor_rows_" + d.Code, names, source, static _ => 0, out string? reason);
            (IntPtr, IntPtr)? pair = kernels is not null && kernels.TryGetValue(one, out var f1) && kernels.TryGetValue(many, out var f2) ? (f1, f2) : null;
            if (pair is null)
            {
                TypedUnavailableReason = reason ?? $"the tensor rows of {d.Code} did not load";
            }

            _blockRows[d] = pair;
            return pair;
        }
    }

    // The rows the kernel reads for one weight: the weight itself, or its bfloat16 rows widened into a block of kp the caller
    // releases. Words: a row's 32-bit words.
    private (Storage Rows, Storage? Scales, int Words, bool Owned) RowsOf(in RowsSource source, Storage w, Storage? scales, int n, int k)
    {
        if (source.Widen == 0)
        {
            // Blocks: ldb is a row's bytes (k / 32 blocks); int8 quads and bfloat16 pairs: its 32-bit words.
            return (w, scales, source.BlockBytes > 0 ? k / 32 * source.BlockBytes : source.Int8 ? k / 4 : k / 2, false);
        }

        int kp = PtxKernels.NativePadded(k);
        var rows = Allocate(n * kp / 2, zeroed: false);
        if (source.Super)
        {
            Launch(source.Widen, (uint)(((long)n * kp / 8 + 255) / 256), 1, 256, 1, P(w), P(rows), U(n), U(k), U(kp));
        }
        else
        {
            LaunchPack(source.Widen, n, kp, P(w), scales is null ? 0UL : P(scales), P(rows), U(n), U(k), U(kp));
        }

        return (rows, null, kp / 2, true);
    }

    // y [m, n] = x · Wᵀ by the lane, its k splits measured per shape (TuneOp.PackedSplits, variants past today's).
    private bool RunTensorRows(Storage x, Storage w, Storage? scales, Storage y, int m, int n, int k)
    {
        if (TensorRowsSource(w.ElementType, scales, m, k) is not { } source)
        {
            return false;
        }

        var (rows, rowScales, words, owned) = RowsOf(source, w, scales, n, k);
        try
        {
            int rowTiles = (m + PtxKernels.TensorTile - 1) / PtxKernels.TensorTile, columnTiles = (n + PtxKernels.TensorTile - 1) / PtxKernels.TensorTile;
            void Run(int splits)
            {
                if (_profile is not null)
                {
                    _profileLabel = string.Create(CultureInfo.InvariantCulture, $"gemm_tc_nt_{source.Label} {m}x{n}x{k}");
                    _profileFlops = 2.0 * m * n * k;
                }

                LaunchTensor(source.Kernel, columnTiles, rowTiles, PtxKernels.TensorTile, splits, (uint)splits, P(x), P(rows), P(y), U(m), U(n), U(k), F(0f),
                    0UL, 0UL, 0UL, 0UL, U(k), U(words), U(n), rowScales is null ? 0UL : P(rowScales));
            }

            Run(PackedSplits is int forced ? ForcedSplits(forced, k, rowTiles * columnTiles)
                : PromptSplitsTuned(new TuneKey(TuneOp.PackedSplits, source.Variant, m, n, k), rowTiles * columnTiles, k, Run));
            return true;
        }
        finally
        {
            if (owned)
            {
                rows.Release();
            }
        }
    }

    // A group's products (2-3 of one type sharing x, every width a whole 128-column tile but the last's) in one launch of the
    // lane's multi kernel; false where it cannot take them.
    private bool RunTensorRowsMany(Storage x, int m, int k, (Storage Values, Storage? Scales, Storage Output, int Columns)[] products)
    {
        if (products.Length is < 2 or > 3 || TensorRowsSource(products[0].Values.ElementType, products[0].Scales, m, k) is not { } source)
        {
            return false;
        }

        int columnTiles = 0;
        for (int i = 0; i < products.Length; i++)
        {
            var product = products[i];
            if (product.Columns == 0 || i < products.Length - 1 && product.Columns % PtxKernels.TensorTile != 0
                || product.Values.ElementType != products[0].Values.ElementType || (product.Scales is null) != (products[0].Scales is null))
            {
                return false;
            }

            columnTiles += (product.Columns + PtxKernels.TensorTile - 1) / PtxKernels.TensorTile;
        }

        var rows = new (Storage Rows, Storage? Scales, int Words, bool Owned)[products.Length];
        try
        {
            for (int i = 0; i < products.Length; i++)
            {
                rows[i] = RowsOf(source, products[i].Values, products[i].Scales, products[i].Columns, k);
            }

            int rowTiles = (m + PtxKernels.TensorTile - 1) / PtxKernels.TensorTile;
            static ulong Scales(Storage? scales) => scales is null ? 0UL : P(scales);
            void Run(int splits)
            {
                if (_profile is not null)
                {
                    _profileLabel = string.Create(CultureInfo.InvariantCulture,
                        $"gemm_tc_nt_{source.Label} multi {m}x{string.Join('+', products.Select(p => p.Columns))}x{k}");
                    _profileFlops = 0;
                    foreach (var product in products)
                    {
                        _profileFlops += 2.0 * m * product.Columns * k;
                    }
                }

                var p0 = products[0];
                var p1 = products[1];
                var p2 = products.Length == 3 ? products[2] : products[1];
                var r2 = products.Length == 3 ? rows[2] : rows[1];
                int n2 = products.Length == 3 ? products[2].Columns : 0;
                LaunchTensor(source.Multi, columnTiles, rowTiles, PtxKernels.TensorTile, splits, (uint)splits,
                    P(x), P(rows[0].Rows), P(p0.Output), U(m), U(p0.Columns), U(k), F(0f), 0UL, 0UL, 0UL, 0UL, U(k), U(rows[0].Words), U(p0.Columns),
                    Scales(rows[0].Scales), P(rows[1].Rows), P(p1.Output), Scales(rows[1].Scales), U(p1.Columns),
                    P(r2.Rows), P(p2.Output), Scales(r2.Scales), U(n2));
            }

            var key = new TuneKey(TuneOp.PackedMultiSplits, source.Variant, m, k, products[0].Columns, products[1].Columns, products.Length == 3 ? products[2].Columns : 0);
            Run(PackedSplits is int forced ? ForcedSplits(forced, k, rowTiles * columnTiles) : PromptSplitsTuned(key, rowTiles * columnTiles, k, Run));
            return true;
        }
        finally
        {
            foreach (var (stored, _, _, owned) in rows)
            {
                if (owned)
                {
                    stored.Release();
                }
            }
        }
    }

    // A number for each type, for the group choice's keys.
    private static readonly ConcurrentDictionary<string, int> GroupIds = new(StringComparer.Ordinal);

    // Whether a group sharing x runs as one launch of the lane (1) or each product by its own measured lane (0): measured
    // once per group shape on the real operands, both ways after a first run of each (which settles their own choices).
    private bool TensorRowsGroup(Storage x, int m, int k, ReadOnlySpan<(Storage Values, Storage? Scales, Storage Output, int Columns)> products,
        Func<bool> eachOwn)
    {
        if (products.Length is < 2 or > 3 || TensorRowsSource(products[0].Values.ElementType, products[0].Scales, m, k) is null)
        {
            return eachOwn();
        }

        var array = products.ToArray();                                     // prompt-sized: one small copy per call
        var type = array[0].Values.ElementType;
        var key = new TuneKey(TuneOp.ProductRoute, 9700 + GroupIds.GetOrAdd(type.Name, static _ => GroupIds.Count), TuneSizes.Class(m), k, array[0].Columns,
            array[1].Columns, array.Length == 3 ? array[2].Columns : 0);
        if (!TryTuned(key, [0, 1], 0, out int lane))
        {
            if (!RunTensorRowsMany(x, m, k, array))
            {
                return eachOwn();
            }

            if (!eachOwn())
            {
                return false;
            }

            bool own = true;
            lane = Tune(key, [0, 1], 0, c => own &= c == 1 ? RunTensorRowsMany(x, m, k, array) : eachOwn());
            if (!own)
            {
                lane = 0;
            }

            int columns = 0;
            foreach (var product in array)
            {
                columns += product.Columns;
            }

            string how = lane == 1 ? "in one tensor-row launch" : "each product by its own lane";
            SayStored(type, m, columns, k, string.Create(CultureInfo.InvariantCulture, $"a group of {array.Length}, {how}"));
        }

        return lane == 1 ? RunTensorRowsMany(x, m, k, array) || eachOwn() : eachOwn();
    }

    /// <summary>Tests: the tensor-row lane run directly on one product (false where it does not take it).</summary>
    internal bool StoredTensorRows(Storage x, Storage weight, Storage? scales, Storage y, int m, int n, int k) =>
        RunTensorRows(x, weight, scales, y, m, n, k);

    /// <summary>Tests: the tensor-row lane's one launch over a group sharing x (false where it does not take them).</summary>
    internal bool StoredTensorRowsMany(Storage x, int m, int k, (Storage Values, Storage? Scales, Storage Output, int Columns)[] products) =>
        RunTensorRowsMany(x, m, k, products);
}
