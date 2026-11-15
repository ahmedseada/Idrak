// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Globalization;

namespace Idrak.Gpu.Cuda;

// The experiment's tensor-row lane (6; the super-block and grid formats' 3): today's tensor-core product
// (PtxKernels.TensorCore.cs, register-staged tiles of 128 or 64 rows by 128 columns, x read as float32 and rounded in the
// kernel, k split in the kernel with the splits' ordered sum) over the routes' weights a row an output: read as they are
// stored where the kernel reads them (bfloat16 and float16 pairs, …_nt_bf16w_f32 and …_nt_f16w_f32, 16 bytes a load; int8
// codes with a scale an output, …_nt_int8w_f32; scaled blocks of 32 decoded as each tile loads, …_nt_{code}_f32), else
// widened into bfloat16 rows for the call (a typed description's pack kernel, a super-block or grid description's
// widening) and read as bfloat16 rows: no packing pass over x, no separate pass adding k splits. A candidate of the
// measured lanes (StoredMeasured, StoredSuper), taken only where it measures faster; the row tile and the k splits are
// measured per shape; a group of products sharing x also measures one launch over all of them (…_multi_f32: their column
// tiles side by side, as today's grouped products run) against each product's own lane.
internal sealed unsafe partial class CudaBackend
{
    // The row tiles a launch is measured with.
    private static readonly int[] RowsTiles = [PtxKernels.TensorTile, 64];

    // A source's kernels: one product and a group, each with 128- and 64-row tiles.
    private readonly record struct RowsKernels(IntPtr One, IntPtr One64, IntPtr Many, IntPtr Many64)
    {
        public IntPtr For(bool multi, int tile) => multi ? tile == 64 ? Many64 : Many : tile == 64 ? One64 : One;
    }

    // How a type's weights reach the tensor rows: the kernels, whether the rows are int8 codes with a scale an output, the
    // bytes of a block of 32 where they are scaled blocks read as stored (0: none), and the widening into bfloat16 rows of
    // kp (0: read as stored; Super: a super-block or grid description's, else a typed description's pack kernel). Variant:
    // what the measured choices are kept under; Label: the profile's name.
    private readonly record struct RowsSource(RowsKernels Kernels, bool Int8, int BlockBytes, IntPtr Widen, bool Super, int Variant, string Label, int BlockValues = 32);

    // The built-in kernels of a 16-bit or int8 source ("bf16w", "f16w", "int8w"), or null where one did not load.
    private RowsKernels? BuiltInRows(string format) =>
        TensorKernel($"gemm_tc_nt_{format}_f32") is { } one && TensorKernel($"gemm_tc_nt_{format}_m64_f32") is { } one64
        && TensorKernel($"gemm_tc_nt_{format}_multi_f32") is { } many && TensorKernel($"gemm_tc_nt_{format}_multi_m64_f32") is { } many64
            ? new RowsKernels(one, one64, many, many64) : null;

    // The source of `type`'s weights (with `scales`, k values a row) for a prompt of m rows; null where the lane does not
    // take them: a few rows, no bfloat16 products, k under one tile or not whole loads, or no kernel loaded. `widen`: a
    // super-block or grid description's widened rows instead of its values decoded as the tiles load (two lanes, measured).
    private RowsSource? TensorRowsSource(ElementType type, Storage? scales, int m, int k, bool widen = false)
    {
        if (m <= PtxKernels.TypedGemvRows[^1] || !BFloat16Products || k < 32 || BuiltInRows("bf16w") is not { } bf16)
        {
            return null;
        }

        if (type.Super is not null || type.Grid is not null)
        {
            if (scales is not null || k % 8 != 0 || Decoded(type) is not var (f, values) || k % values != 0)
            {
                return null;
            }

            int id = 300 + GroupIds.GetOrAdd(type.Name, static _ => GroupIds.Count);
            return !widen && DecodedRowsKernels(type) is { } decoded
                ? new RowsSource(decoded, false, type.Super?.Bytes ?? type.Grid!.Bytes, 0, false, id, type.Name, values)
                : widen ? new RowsSource(bf16, false, 0, f.WidenBf16, true, id + 1000, "bf16w widened") : null;
        }

        if (DecodeOf(type) is not { } d)
        {
            return null;
        }

        // 16-bit floats as stored: bfloat16 in bfloat16 products, float16 in float16 ones (the native float16 kind's
        // arithmetic: x rounded to float16, the weights exact).
        if (PtxKernels.NativeKind.BF16.Holds(d))
        {
            return scales is null && k % 8 == 0 ? new RowsSource(bf16, false, 0, 0, false, 40, "bf16w") : null;
        }

        if (PtxKernels.NativeKind.F16.Holds(d))
        {
            return scales is null && k % 8 == 0 && BuiltInRows("f16w") is { } f16 ? new RowsSource(f16, false, 0, 0, false, 43, "f16w") : null;
        }

        if (d is { Kind: 'i', Bits: 8, Signed: true } && scales is not null)
        {
            return k % 16 == 0 && BuiltInRows("int8w") is { } int8 ? new RowsSource(int8, true, 0, 0, false, 41, "int8w") : null;
        }

        // Scaled blocks of 32 the kernel decodes as its tiles load (one block a row of a k tile): read as stored.
        if (d.Block is { } block && scales is null && k % 32 == 0 && PtxKernels.BlockRows(block) && BlockRowsKernels(d) is { } blocks)
        {
            return new RowsSource(blocks, false, block.Bytes, 0, false, 100 + GroupIds.GetOrAdd(type.Name, static _ => GroupIds.Count), d.Code);
        }

        // Any other description the native lane widens (its conditions): widened rows of kp, then read as bfloat16.
        return MatrixUnitLane(d, m, k) && (scales is null || d.PerValue) && k % 8 == 0 && NativePack(d, PtxKernels.NativeKind.BF16) is { } pack
            ? new RowsSource(bf16, false, 0, pack.W, false, 42, "bf16w widened") : null;
    }

    // Each block description's tensor-row kernels, loaded on first use: null where the driver refuses the module
    // (TypedUnavailableReason says why).
    private readonly ConcurrentDictionary<TypedDecode, RowsKernels?> _blockRows = new();

    private RowsKernels? BlockRowsKernels(TypedDecode d)
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
            var kernels = LoadDynamicModule("tensor_rows_" + d.Code, names, source, static _ => 0, out string? reason);
            IntPtr Find(string suffix) =>
                kernels is not null && kernels.TryGetValue(string.Create(CultureInfo.InvariantCulture, $"gemm_tc_nt_{d.Code}{suffix}_f32"), out var f) ? f : 0;
            RowsKernels? found = null;
            if (kernels is not null && Find("") is not 0 and var one && Find("_m64") is not 0 and var one64 && Find("_multi") is not 0 and var many
                && Find("_multi_m64") is not 0 and var many64)
            {
                found = new RowsKernels(one, one64, many, many64);
            }
            else
            {
                TypedUnavailableReason = reason ?? $"the tensor rows of {d.Code} did not load";
            }

            _blockRows[d] = found;
            return found;
        }
    }

    // Each super-block or grid description's tensor-row kernels (its values decoded as the tiles load), loaded on first use:
    // null where the driver refuses the module (TypedUnavailableReason says why).
    private readonly ConcurrentDictionary<string, RowsKernels?> _decodedRows = new(StringComparer.Ordinal);

    private RowsKernels? DecodedRowsKernels(ElementType type)
    {
        if (_decodedRows.TryGetValue(type.Name, out var known))
        {
            return known;
        }

        lock (_typedLock)
        {
            if (_decodedRows.TryGetValue(type.Name, out known))
            {
                return known;
            }

            var (code, tables, decode) = type.Super is { } s
                ? (PtxKernels.SuperCode(s), PtxKernels.SuperPiecesDeclaration(s), PtxKernels.SuperDecode(s))
                : (PtxKernels.GridCode(type.Grid!), PtxKernels.GridTables(type.Grid!), PtxKernels.GridDecode(type.Grid!));
            var (names, source) = PtxKernels.TensorRowsDecodedModule(code, tables, decode);
            var kernels = LoadDynamicModule("tensor_rows_" + code, names, source, static _ => 0, out string? reason);
            IntPtr Find(string suffix) =>
                kernels is not null && kernels.TryGetValue(string.Create(CultureInfo.InvariantCulture, $"gemm_tc_nt_{code}{suffix}_f32"), out var f) ? f : 0;
            RowsKernels? found = null;
            if (kernels is not null && Find("") is not 0 and var one && Find("_m64") is not 0 and var one64 && Find("_multi") is not 0 and var many
                && Find("_multi_m64") is not 0 and var many64)
            {
                found = new RowsKernels(one, one64, many, many64);
            }
            else
            {
                TypedUnavailableReason = reason ?? $"the tensor rows of {type.Name} did not load";
            }

            _decodedRows[type.Name] = found;
            return found;
        }
    }

    // The rows the kernel reads for one weight: the weight itself, or its bfloat16 rows widened into a block of kp the caller
    // releases. Words: ldb (a row's 32-bit words; for blocks, its bytes).
    private (Storage Rows, Storage? Scales, int Words, bool Owned) RowsOf(in RowsSource source, Storage w, Storage? scales, int n, int k)
    {
        if (source.Widen == 0)
        {
            return (w, scales, source.BlockBytes > 0 ? k / source.BlockValues * source.BlockBytes : source.Int8 ? k / 4 : k / 2, false);
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

    // The row tile of a launch at this shape (128 or 64 rows: few output tiles fill the device unsplit with 64), measured
    // once (cold for a gathering source: Gathers), each tile with its own k splits; `run(tile)` launches with it.
    private int RowsTile(in RowsSource source, TuneKey key, Action<int> run) =>
        RowsTileOverride ?? (TryTuned(key, RowsTiles, PtxKernels.TensorTile, out int tile) ? tile : Tune(key, RowsTiles, PtxKernels.TensorTile, run, cold: Gathers(source)));

    // Whether a source's kernels gather scattered bytes per 8 values (a super-block or grid decoder): its choices are timed
    // cold (MeasureSuper says why); every other source reads contiguous rows and is timed warm (MeasureStoredRoute).
    private static bool Gathers(in RowsSource source) => source.BlockValues > 32;

    // y [m, n] = x · Wᵀ by the lane (`widen`: a super-block or grid description's widened rows, TensorRowsSource).
    private bool RunTensorRows(Storage x, Storage w, Storage? scales, Storage y, int m, int n, int k, bool widen = false)
    {
        if (TensorRowsSource(w.ElementType, scales, m, k, widen) is not { } source)
        {
            return false;
        }

        var (rows, rowScales, words, owned) = RowsOf(source, w, scales, n, k);
        try
        {
            int columnTiles = (n + PtxKernels.TensorTile - 1) / PtxKernels.TensorTile;
            void Run(int tile)
            {
                int rowTiles = (m + tile - 1) / tile;
                void RunSplits(int splits)
                {
                    if (_profile is not null)
                    {
                        _profileLabel = string.Create(CultureInfo.InvariantCulture, $"gemm_tc_nt_{source.Label}{(tile == 64 ? " m64" : "")} {m}x{n}x{k}");
                        _profileFlops = 2.0 * m * n * k;
                    }

                    LaunchTensor(source.Kernels.For(false, tile), columnTiles, rowTiles, tile, splits, (uint)splits, P(x), P(rows), P(y), U(m), U(n), U(k), F(0f),
                        0UL, 0UL, 0UL, 0UL, U(k), U(words), U(n), rowScales is null ? 0UL : P(rowScales));
                }

                RunSplits(PackedSplits is int forced ? ForcedSplits(forced, k, rowTiles * columnTiles)
                    : PromptSplitsTuned(new TuneKey(TuneOp.PackedSplits, source.Variant + (tile == 64 ? 500 : 0), m, n, k), rowTiles * columnTiles, k, RunSplits));
            }

            Run(RowsTile(source, new TuneKey(TuneOp.PackedTile, 200 + source.Variant, m, n, k), Run));
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

            static ulong Scales(Storage? scales) => scales is null ? 0UL : P(scales);
            int n0 = products[0].Columns, n1 = products[1].Columns, n2 = products.Length == 3 ? products[2].Columns : 0;
            void Run(int tile)
            {
                int rowTiles = (m + tile - 1) / tile;
                void RunSplits(int splits)
                {
                    if (_profile is not null)
                    {
                        _profileLabel = string.Create(CultureInfo.InvariantCulture,
                            $"gemm_tc_nt_{source.Label} multi{(tile == 64 ? " m64" : "")} {m}x{string.Join('+', products.Select(p => p.Columns))}x{k}");
                        _profileFlops = 0;
                        foreach (var product in products)
                        {
                            _profileFlops += 2.0 * m * product.Columns * k;
                        }
                    }

                    var r2 = products.Length == 3 ? rows[2] : rows[1];
                    var p2 = products.Length == 3 ? products[2] : products[1];
                    LaunchTensor(source.Kernels.For(true, tile), columnTiles, rowTiles, tile, splits, (uint)splits,
                        P(x), P(rows[0].Rows), P(products[0].Output), U(m), U(n0), U(k), F(0f), 0UL, 0UL, 0UL, 0UL, U(k), U(rows[0].Words), U(n0),
                        Scales(rows[0].Scales), P(rows[1].Rows), P(products[1].Output), Scales(rows[1].Scales), U(n1),
                        P(r2.Rows), P(p2.Output), Scales(r2.Scales), U(n2));
                }

                var key = new TuneKey(TuneOp.PackedMultiSplits, source.Variant + (tile == 64 ? 500 : 0), m, k, n0, n1, n2);
                RunSplits(PackedSplits is int forced ? ForcedSplits(forced, k, rowTiles * columnTiles) : PromptSplitsTuned(key, rowTiles * columnTiles, k, RunSplits));
            }

            Run(RowsTile(source, new TuneKey(TuneOp.PackedTile, 1200 + source.Variant, m, k, n0, n1, n2), Run));
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

    // A number for each type, for the measured choices' keys.
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
            var source = TensorRowsSource(type, array[0].Scales, m, k)!.Value;
            lane = Tune(key, [0, 1], 0, c => own &= c == 1 ? RunTensorRowsMany(x, m, k, array) : eachOwn(), cold: Gathers(source));
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

    /// <summary>Tests: the tensor-row lane run directly on one product (false where it does not take it); `widen`: as RunTensorRows.</summary>
    internal bool StoredTensorRows(Storage x, Storage weight, Storage? scales, Storage y, int m, int n, int k, bool widen = false) =>
        RunTensorRows(x, weight, scales, y, m, n, k, widen);

    /// <summary>Tests: the row tile the tensor rows launch with (128 or 64) instead of the measured one.</summary>
    internal static int? RowsTileOverride { get; set; }

    /// <summary>Tests: the tensor-row lane's one launch over a group sharing x (false where it does not take them).</summary>
    internal bool StoredTensorRowsMany(Storage x, int m, int k, (Storage Values, Storage? Scales, Storage Output, int Columns)[] products) =>
        RunTensorRowsMany(x, m, k, products);
}
