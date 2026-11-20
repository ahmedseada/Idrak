// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Idrak.Abstraction.Devices.Cpu;

// The experiment's CPU lanes for weights stored in the file's layout (ProductRoutes): y = x · W for W
// stored as n rows of k values of any type, its description choosing how a row widens (IRowWiden: a float's bits, an
// integer's, a scaled block's codes; any other type through its own conversion), with a scale per row or none. Two lanes,
// each written once for every type (the JIT compiles one copy per widening): the row lane, each output's row widened once
// and dotted with every row of x (decoding-sized products), and the panel lane, a tile of rows widened into a [columns, k]
// panel and multiplied by the float32 product reading it transposed (prompt-sized products). The generic route takes the
// row lane up to 8 rows and the panel lane above; the measured route the faster of the two at the shape, measured once
// (Faster). A 4-bit super-block (NibbleBlocks: ggml's Q4_K by its description) takes, under the measured route, only its
// two lanes that never widen it: its codes decoded in registers inside the dot products, or int8 dot products over the
// stored codes with x rounded to int8 a block of 32 values; the faster at the shape.
internal sealed partial class CpuBackend
{
    /// <summary>How the values of a stored row widen to float32: the type's description decides which.</summary>
    private interface IRowWiden
    {
        /// <summary>Values [first, first + destination.Length) of a row whose bytes start at <paramref name="row"/>'s first byte (whole blocks for a block format).</summary>
        static abstract void Widen(ReadOnlySpan<byte> row, int first, Span<float> destination, in StoredLane lane);
    }

    // A float of 8 exponent bits and 7 mantissa bits (bfloat16): each value is float32's high half.
    private struct HighHalfWiden : IRowWiden
    {
        public static void Widen(ReadOnlySpan<byte> row, int first, Span<float> destination, in StoredLane lane) =>
            HalfFloats.WidenBFloat16(MemoryMarshal.Cast<byte, ushort>(row).Slice(first, destination.Length), destination);
    }

    // float32 as stored.
    private struct SameWiden : IRowWiden
    {
        public static void Widen(ReadOnlySpan<byte> row, int first, Span<float> destination, in StoredLane lane) =>
            MemoryMarshal.Cast<byte, float>(row).Slice(first, destination.Length).CopyTo(destination);
    }

    // An 8-bit signed integer: widened exactly (vectors, then a scalar tail).
    private struct SignedByteWiden : IRowWiden
    {
        public static void Widen(ReadOnlySpan<byte> row, int first, Span<float> destination, in StoredLane lane) =>
            ElementConversions.Int8ToFloat32(row.Slice(first, destination.Length), destination);
    }

    // A scaled block of 8-bit codes, or 4-bit codes in ggml's order, and a float16 scale (ScaledBlock): each block's codes
    // widened and times its scale.
    private struct ByteBlockWiden : IRowWiden
    {
        public static void Widen(ReadOnlySpan<byte> row, int first, Span<float> destination, in StoredLane lane)
        {
            var block = lane.Block!;
            int values = block.Values;
            for (int b = 0; b < destination.Length / values; b++)
            {
                var bytes = row.Slice((first / values + b) * block.Bytes, block.Bytes);
                float scale = (float)BinaryPrimitives.ReadHalfLittleEndian(bytes.Slice(block.ScaleOffset, 2));
                var into = destination.Slice(b * values, values);
                if (block.CodeBits == 4)
                {
                    // ggml's order: byte i holds value i (low half) and value i + values / 2 (high half).
                    var nibbles = bytes.Slice(block.CodeOffset, values / 2);
                    var table = block.CodeTable;
                    int i = 0;
                    if (Vector128.IsHardwareAccelerated)
                    {
                        var grid = table is null ? default : TableVector(table);
                        for (; i + 16 <= values / 2; i += 16)
                        {
                            if (table is null)
                            {
                                NibblesToFloats(nibbles.Slice(i, 16), into.Slice(i, 16), into.Slice(i + values / 2, 16), block.SignedCodes);
                            }
                            else
                            {
                                NibblesThroughTable(nibbles.Slice(i, 16), into.Slice(i, 16), into.Slice(i + values / 2, 16), grid);
                            }
                        }
                    }

                    for (; i < values / 2; i++)
                    {
                        int low = nibbles[i] & 15, high = nibbles[i] >> 4;
                        into[i] = table is not null ? table[low] : block.SignedCodes ? (low ^ 8) - 8 : low;
                        into[i + values / 2] = table is not null ? table[high] : block.SignedCodes ? (high ^ 8) - 8 : high;
                    }

                    ScaleBy(into, block.CodeBias, scale);
                    continue;
                }

                var codes = bytes.Slice(block.CodeOffset, values);
                if (block.SignedCodes)
                {
                    ElementConversions.Int8ToFloat32(codes, into);
                }
                else
                {
                    for (int i = 0; i < values; i++)
                    {
                        into[i] = codes[i];
                    }
                }

                ScaleBy(into, block.CodeBias, scale);
            }
        }

        // 16 bytes of 4-bit codes: their low halves into `low`, their high halves into `high`, as floats (two's complement
        // over 4 bits when `signed`).
        private static void NibblesToFloats(ReadOnlySpan<byte> bytes, Span<float> low, Span<float> high, bool signed)
        {
            var v = Vector128.Create(bytes);
            var mask = Vector128.Create((byte)15);
            var lo = v & mask;
            var hi = Vector128.ShiftRightLogical(v.AsUInt16(), 4).AsByte() & mask;
            Store(lo, low, signed);
            Store(hi, high, signed);

            static void Store(Vector128<byte> codes, Span<float> into, bool signed)
            {
                var (s0, s1) = Vector128.Widen(codes);
                var (i0, i1) = Vector128.Widen(s0);
                var (i2, i3) = Vector128.Widen(s1);
                Four(i0.AsInt32(), into, 0, signed);
                Four(i1.AsInt32(), into, 4, signed);
                Four(i2.AsInt32(), into, 8, signed);
                Four(i3.AsInt32(), into, 12, signed);
            }

            static void Four(Vector128<int> ints, Span<float> into, int at, bool signed)
            {
                if (signed)
                {
                    var eight = Vector128.Create(8);
                    ints = (ints ^ eight) - eight;
                }

                Vector128.ConvertToSingle(ints).CopyTo(into.Slice(at, 4));
            }
        }

        // A 16-entry code table as int8 lanes (its entries fit int8: WideningOf checks).
        internal static Vector128<sbyte> TableVector(IReadOnlyList<int> table)
        {
            Span<sbyte> entries = stackalloc sbyte[16];
            for (int i = 0; i < 16; i++)
            {
                entries[i] = (sbyte)table[i];
            }

            return Vector128.Create(entries);
        }

        // 16 bytes of 4-bit codes through a 16-entry table: their low halves' entries into `low`, their high halves' into `high`.
        private static void NibblesThroughTable(ReadOnlySpan<byte> bytes, Span<float> low, Span<float> high, Vector128<sbyte> table)
        {
            var v = Vector128.Create(bytes);
            var mask = Vector128.Create((byte)15);
            Store(Vector128.Shuffle(table, (v & mask).AsSByte()), low);
            Store(Vector128.Shuffle(table, (Vector128.ShiftRightLogical(v.AsUInt16(), 4).AsByte() & mask).AsSByte()), high);

            static void Store(Vector128<sbyte> entries, Span<float> into)
            {
                var (s0, s1) = Vector128.Widen(entries);
                var (i0, i1) = Vector128.Widen(s0);
                var (i2, i3) = Vector128.Widen(s1);
                Vector128.ConvertToSingle(i0).CopyTo(into);
                Vector128.ConvertToSingle(i1).CopyTo(into[4..]);
                Vector128.ConvertToSingle(i2).CopyTo(into[8..]);
                Vector128.ConvertToSingle(i3).CopyTo(into[12..]);
            }
        }

        // into[i] = (into[i] + bias) · scale.
        private static void ScaleBy(Span<float> into, int bias, float scale)
        {
            int i = 0;
            if (Vector.IsHardwareAccelerated)
            {
                var s = new Vector<float>(scale);
                var add = new Vector<float>(bias);
                var v = MemoryMarshal.Cast<float, Vector<float>>(into);
                for (int j = 0; j < v.Length; j++)
                {
                    v[j] = bias == 0 ? v[j] * s : (v[j] + add) * s;
                }

                i = v.Length * Vector<float>.Count;
            }

            for (; i < into.Length; i++)
            {
                into[i] = (into[i] + bias) * scale;
            }
        }
    }

    // A super-block format (ElementType.Super: the k-quants): its whole blocks widened by the description alone.
    private struct SuperWiden : IRowWiden
    {
        public static void Widen(ReadOnlySpan<byte> row, int first, Span<float> destination, in StoredLane lane)
        {
            var super = lane.Type.Super!;
            lane.Type.Super!.Decode(row.Slice(first / super.Values * super.Bytes, destination.Length / super.Values * super.Bytes), destination);
        }
    }

    // A grid-codebook format (ElementType.Grid: IQ2_XXS, IQ3_XXS): its whole blocks widened by the description alone.
    private struct GridWiden : IRowWiden
    {
        public static void Widen(ReadOnlySpan<byte> row, int first, Span<float> destination, in StoredLane lane)
        {
            var grid = lane.Type.Grid!;
            grid.Decode(row.Slice(first / grid.Values * grid.Bytes, destination.Length / grid.Values * grid.Bytes), destination);
        }
    }

    // Any other type: its own conversion (ElementType.Decode) over the values asked.
    private struct DecodeWiden : IRowWiden
    {
        public static void Widen(ReadOnlySpan<byte> row, int first, Span<float> destination, in StoredLane lane)
        {
            var type = lane.Type;
            long start = type.ByteCount(first), taken = type.ByteCount(destination.Length);
            type.Decode(row.Slice((int)start, (int)taken), destination);
        }
    }

    // A stored weight's words, its type and description, its row scales (or null), k and the bytes of a row.
    private sealed record StoredLane(ElementType Type, ScaledBlock? Block, float[] Words, int Length, float[]? RowScales, int K, long RowBytes);

    private enum Widening : byte
    {
        Decode,
        HighHalf,
        Same,
        SignedByte,
        ByteBlock,
        Super,
        Grid,
    }

    // The widening a type's description gives (the description, never the name).
    private static Widening WideningOf(ElementType type)
    {
        if (type.Super is not null)
        {
            return Widening.Super;
        }

        if (type.Grid is not null)
        {
            return Widening.Grid;
        }

        if (type.Scaled is { Scale: { Exponent: 5, Mantissa: 10 } } block && (block.CodeBits == 8 && block.CodeTable is null
                || block.CodeBits == 4 && block.NibblesSplit && (block.CodeTable is null || block.CodeTable is { Count: 16 } grid && grid.All(e => e is >= -128 and <= 127)))
            && block.Values % 8 == 0 && block.CodeOffset + block.Values * block.CodeBits / 8 <= block.Bytes)
        {
            return Widening.ByteBlock;
        }

        if (type.IsBlock)
        {
            return Widening.Decode;
        }

        return type.Float switch
        {
            { Exponent: 8, Mantissa: 7 } when type.BlockBits == 16 => Widening.HighHalf,
            { Exponent: 8, Mantissa: 23 } when type.BlockBits == 32 => Widening.Same,
            null when type.IsInteger && type.IsSigned && type.BlockBits == 8 => Widening.SignedByte,
            _ => Widening.Decode,
        };
    }

    /// <summary>
    /// The routed product of a stored weight (ProductRoutes generic or measured): a scale per row or none, no zero points,
    /// in the CPU's own memory. False when the weight is not such a one (the stored product then runs as today).
    /// </summary>
    private bool StoredRouted(Storage x, Storage weight, Storage? scales, Storage? zeros, Storage y, int m, int n, int k, StoredScales layout)
    {
        bool perRow = !layout.HasScales || layout is { BlockRows: 1, ScaleColumns: 1 } && layout.BlockColumns >= k;
        if (zeros is not null || !perRow || weight is MappedStorage || m == 0 || n == 0 || k == 0)
        {
            return false;
        }

        var type = weight.ElementType;
        var lane = new StoredLane(type, type.Scaled, D(weight), weight.Length, scales is null ? null : D(scales), k, type.ByteCount(k));
        float[] xv = D(x), yv = D(y);
        var widening = WideningOf(type);
        // A 4-bit super-block under the measured route: only the nibble lanes, which never widen the weight (NibbleBlocks).
        var nibbles = ProductRoutes.Current == ProductRoute.Measured && type.Super is { } super ? NibbleBlocks.Of(super) : null;
        int chosen = ProductRoutes.Current == ProductRoute.Generic
            ? m <= Int8DirectRows ? RowLaneId : PanelLaneId
            : StoredLanes.TryGetValue((type.Name, LaneKey(type.Name, m, n, k).Rows, n, k), out int known) ? known
            : nibbles is not null ? MeasureNibbleLanes(nibbles, lane, xv, yv, m, n, k)
            : MeasureStoredLanes(widening, lane, x, y, m, n, k);
        if (chosen is DecodedLaneId or IntegerLaneId)
        {
            NibbleProduct(chosen == IntegerLaneId, nibbles ?? NibbleBlocks.Of(type.Super!)!, lane, xv, yv, m, n, k);
            s_nibblesMultiplied = true;
            return true;
        }

        RunLane(chosen, widening, lane, x, y, m, n, k);
        return true;
    }

    // The stored products' lanes: each output's row widened once (decoding-sized), a panel of rows widened and multiplied
    // transposed, the whole weight widened once and the float32 product (today's prompt path: ExpandedLane), the tiles; and
    // a 4-bit super-block's two lanes over its stored codes (NibbleBlocks), decoded in registers or by int8 dot products.
    private const int RowLaneId = 0, PanelLaneId = 1, ExpandedLaneId = 2, TilesLaneId = 3, DecodedLaneId = 4, IntegerLaneId = 5;

    private static readonly string[] LaneNames = ["row lane", "panel lane", "today's path (widened once, float32 product)",
        "tiles (panels widened in the product)", "decoded lane (codes decoded in registers, float32 dot products)",
        "integer lane (x rounded to int8 a block of 32, int8 dot products over the stored codes)"];

    // Each shape's measured lane.
    private static readonly ConcurrentDictionary<(string Type, int Rows, int N, int K), int> StoredLanes = new();

    // The shape's fastest lane, measured once on the real operands (Faster, a lane against the best so far); without
    // measuring (CpuTuning.Autotune off), the row lane up to 8 rows and today's path above.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private int MeasureStoredLanes(Widening widening, StoredLane lane, Storage x, Storage y, int m, int n, int k)
    {
        var key = (lane.Type.Name, LaneKey(lane.Type.Name, m, n, k).Rows, n, k);
        int best = m <= Int8DirectRows ? RowLaneId : ExpandedLaneId;
        if (CpuTuning.Autotune)
        {
            // Today's path (the weights widened whole, then the float32 product) only above the few-row sizes: with few rows it
            // widens every row as the row lane does and copies them too, and its copy of the weights (k · n float32: an output
            // head's is hundreds of MiB) would be the run's memory peak.
            int[] candidates = [RowLaneId, PanelLaneId, .. m > Int8DirectRows ? new[] { ExpandedLaneId } : [],
                .. CpuMatMul.TakesPanels(m, n, k) ? new[] { TilesLaneId } : []];
            var times = new double[candidates.Length];
            best = candidates[0];
            for (int c = 1; c < candidates.Length; c++)
            {
                int challenger = candidates[c], holder = best;
                var (th, tc) = Faster.Time(() => RunLane(holder, widening, lane, x, y, m, n, k), () => RunLane(challenger, widening, lane, x, y, m, n, k));
                times[Array.IndexOf(candidates, holder)] = th;
                times[c] = tc;
                if (tc < th)
                {
                    best = challenger;
                }
            }

            ProductRoutes.Chose("cpu", string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{lane.Type.Name} {m}x{k}->{n}"),
                string.Create(System.Globalization.CultureInfo.InvariantCulture,
                    $"{LaneNames[best]} ({string.Join(", ", candidates.Select((l, i) => $"{LaneNames[l].Split(' ')[0]} {times[i]:0.###} ms"))})"));
        }

        StoredLanes[key] = best;
        return best;
    }

    private void RunLane(int id, Widening widening, StoredLane lane, Storage x, Storage y, int m, int n, int k)
    {
        switch (id)
        {
            case RowLaneId or PanelLaneId:
                RunStoredLane(id == RowLaneId, widening, lane, D(x), D(y), m, n, k);
                break;
            case ExpandedLaneId:
                RunExpanded(widening, lane, x, y, m, n, k);
                break;
            case TilesLaneId when CpuMatMul.TakesPanels(m, n, k):
                RunTiles(widening, lane, D(x), D(y), m, n, k);
                break;
            default:
                RunStoredLane(false, widening, lane, D(x), D(y), m, n, k);
                break;
        }
    }

    // The tiles lane: the float32 product over panels widened from the stored rows inside each tile (MultiplyPanelsFrom).
    private static void RunTiles(Widening widening, StoredLane lane, float[] x, float[] y, int m, int n, int k)
    {
        int nr = CpuMatMul.PanelColumns;
        switch (widening)
        {
            case Widening.HighHalf:
                CpuMatMul.MultiplyPanelsFrom(x, new RowPanels<HighHalfWiden>(lane, k, nr), y, m, n, k);
                break;
            case Widening.Same:
                CpuMatMul.MultiplyPanelsFrom(x, new RowPanels<SameWiden>(lane, k, nr), y, m, n, k);
                break;
            case Widening.SignedByte:
                CpuMatMul.MultiplyPanelsFrom(x, new RowPanels<SignedByteWiden>(lane, k, nr), y, m, n, k);
                break;
            case Widening.ByteBlock:
                CpuMatMul.MultiplyPanelsFrom(x, new RowPanels<ByteBlockWiden>(lane, k, nr), y, m, n, k);
                break;
            case Widening.Super:
                CpuMatMul.MultiplyPanelsFrom(x, new RowPanels<SuperWiden>(lane, k, nr), y, m, n, k);
                break;
            case Widening.Grid:
                CpuMatMul.MultiplyPanelsFrom(x, new RowPanels<GridWiden>(lane, k, nr), y, m, n, k);
                break;
            default:
                CpuMatMul.MultiplyPanelsFrom(x, new RowPanels<DecodeWiden>(lane, k, nr), y, m, n, k);
                break;
        }
    }

    // Panel q of a stored weight: its nr rows widened (each times its scale) into `rows`, then laid out [k][nr] in `buffer`.
    private readonly struct RowPanels<TW>(StoredLane lane, int k, int nr) : CpuMatMul.IPanelSource where TW : struct, IRowWiden
    {
        public bool Fills => true;

        public ref float Panel(int panel, float[] buffer, float[] rows)
        {
            var bytes = MemoryMarshal.AsBytes(lane.Words.AsSpan(0, lane.Length));
            int j0 = panel * nr;
            for (int c = 0; c < nr; c++)
            {
                var into = rows.AsSpan(c * k, k);
                TW.Widen(bytes.Slice((int)((j0 + c) * lane.RowBytes), (int)lane.RowBytes), 0, into, lane);
                if (lane.RowScales is { } s)
                {
                    float scale = s[j0 + c];
                    for (int i = 0; i < k; i++)
                    {
                        into[i] *= scale;
                    }
                }
            }

            for (int p = 0; p < k; p++)
            {
                for (int c = 0; c < nr; c++)
                {
                    buffer[p * nr + c] = rows[c * k + p];
                }
            }

            return ref buffer[0];
        }
    }

    private void RunExpanded(Widening widening, StoredLane lane, Storage x, Storage y, int m, int n, int k)
    {
        switch (widening)
        {
            case Widening.HighHalf:
                ExpandedLane<HighHalfWiden>(lane, x, y, m, n, k);
                break;
            case Widening.Same:
                ExpandedLane<SameWiden>(lane, x, y, m, n, k);
                break;
            case Widening.SignedByte:
                ExpandedLane<SignedByteWiden>(lane, x, y, m, n, k);
                break;
            case Widening.ByteBlock:
                ExpandedLane<ByteBlockWiden>(lane, x, y, m, n, k);
                break;
            case Widening.Super:
                ExpandedLane<SuperWiden>(lane, x, y, m, n, k);
                break;
            case Widening.Grid:
                ExpandedLane<GridWiden>(lane, x, y, m, n, k);
                break;
            default:
                ExpandedLane<DecodeWiden>(lane, x, y, m, n, k);
                break;
        }
    }

    // Today's prompt path over the stored layout: the whole weight widened once, then the CPU's float32 product (its tuned
    // tiles), as today's bfloat16 and int8 products take more rows than the few-row limit. Where the tiled kernel takes the
    // shape (CpuMatMul.TakesPanels) the rows are widened straight into its B panels (nr rows at a time into a scratch in
    // the cache, then laid out [k][nr]), and the product reads them as they are (MultiplyPanels: the same tiles, kernels and
    // sums, no [k, n] copy, no transpose, no panel copies); otherwise into [k, n] (rows written transposed) and MatMul.
    private void ExpandedLane<TW>(StoredLane lane, Storage x, Storage y, int m, int n, int k) where TW : struct, IRowWiden
    {
        // The widened weights in a buffer from the shared pool (sized by powers of two, so the shapes a model alternates
        // between reuse their buffers; the backend's own pool keeps exact sizes and frees the others on a miss).
        float[] wv = ArrayPool<float>.Shared.Rent(k * n);
        var w = new CpuStorage(this, wv, k * n);
        try
        {
            bool panels = CpuMatMul.TakesPanels(m, n, k);
            int rows = panels ? CpuMatMul.PanelColumns : 16;
            Parallel.For(0, (n + rows - 1) / rows, ComputeResources.ParallelOptions, t =>
            {
                var scratch = ArrayPool<float>.Shared.Rent(rows * k);
                try
                {
                    var bytes = MemoryMarshal.AsBytes(lane.Words.AsSpan(0, lane.Length));
                    int j0 = t * rows, count = Math.Min(rows, n - j0);
                    for (int c = 0; c < count; c++)
                    {
                        var into = scratch.AsSpan(c * k, k);
                        TW.Widen(bytes.Slice((int)((j0 + c) * lane.RowBytes), (int)lane.RowBytes), 0, into, lane);
                        if (lane.RowScales is { } s)
                        {
                            float scale = s[j0 + c];
                            int i = 0;
                            if (Vector.IsHardwareAccelerated)
                            {
                                var v = MemoryMarshal.Cast<float, Vector<float>>(into);
                                for (int q = 0; q < v.Length; q++)
                                {
                                    v[q] *= scale;
                                }

                                i = v.Length * Vector<float>.Count;
                            }

                            for (; i < k; i++)
                            {
                                into[i] *= scale;
                            }
                        }
                    }

                    if (panels)
                    {
                        // Panel t: [k][rows], written in order.
                        long at = (long)t * k * rows;
                        for (int p = 0; p < k; p++, at += rows)
                        {
                            for (int c = 0; c < rows; c++)
                            {
                                wv[at + c] = scratch[c * k + p];
                            }
                        }
                    }
                    else
                    {
                        for (int p = 0; p < k; p++)
                        {
                            long at = (long)p * n + j0;
                            for (int c = 0; c < count; c++)
                            {
                                wv[at + c] = scratch[c * k + p];
                            }
                        }
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(scratch);
                }
            });
            if (panels)
            {
                CpuMatMul.MultiplyPanels(D(x), wv, D(y), m, n, k);
            }
            else
            {
                MatMul(x, w, y, m, n, k, false, false, 0f);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(wv);
        }
    }

    private static void RunStoredLane(bool rows, Widening widening, StoredLane lane, float[] x, float[] y, int m, int n, int k)
    {
        switch (widening)
        {
            case Widening.HighHalf:
                RunStoredLane<HighHalfWiden>(rows, lane, x, y, m, n, k);
                break;
            case Widening.Same:
                RunStoredLane<SameWiden>(rows, lane, x, y, m, n, k);
                break;
            case Widening.SignedByte:
                RunStoredLane<SignedByteWiden>(rows, lane, x, y, m, n, k);
                break;
            case Widening.ByteBlock:
                RunStoredLane<ByteBlockWiden>(rows, lane, x, y, m, n, k);
                break;
            case Widening.Super:
                RunStoredLane<SuperWiden>(rows, lane, x, y, m, n, k);
                break;
            case Widening.Grid:
                RunStoredLane<GridWiden>(rows, lane, x, y, m, n, k);
                break;
            default:
                RunStoredLane<DecodeWiden>(rows, lane, x, y, m, n, k);
                break;
        }
    }

    private static void RunStoredLane<TW>(bool rows, StoredLane lane, float[] x, float[] y, int m, int n, int k) where TW : struct, IRowWiden
    {
        if (rows)
        {
            int threads = Math.Max(1, ComputeResources.MaxCpuThreads), chunk = Math.Max(1, n / (threads * 4));
            Parallel.For(0, (n + chunk - 1) / chunk, ComputeResources.ParallelOptions, c => RowLane<TW>(lane, x, y, m, n, k, c * chunk, Math.Min(chunk, n - c * chunk)));
            return;
        }

        PanelLane<TW>(lane, x, y, m, n, k);
    }

    // Outputs [j0, j0 + count): each row widened once, then dotted with every row of x, times the row's scale.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void RowLane<TW>(StoredLane lane, float[] x, float[] y, int m, int n, int k, int j0, int count) where TW : struct, IRowWiden
    {
        var bytes = MemoryMarshal.AsBytes(lane.Words.AsSpan(0, lane.Length));
        float[] row = ArrayPool<float>.Shared.Rent(k);
        try
        {
            var values = row.AsSpan(0, k);
            for (int j = j0; j < j0 + count; j++)
            {
                TW.Widen(bytes.Slice(checked((int)(j * lane.RowBytes)), (int)lane.RowBytes), 0, values, lane);
                float scale = lane.RowScales is { } s ? s[j] : 1f;
                for (int i = 0; i < m; i++)
                {
                    float dot = RowDot(x.AsSpan(i * k, k), values);
                    y[i * n + j] = lane.RowScales is null ? dot : dot * scale;
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(row);
        }
    }

    // Σ a[i] · b[i]: four vector accumulators, added in a fixed order, then a scalar tail.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float RowDot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int i = 0;
        float sum = 0f;
        if (Vector.IsHardwareAccelerated && a.Length >= 4 * Vector<float>.Count)
        {
            var va = MemoryMarshal.Cast<float, Vector<float>>(a);
            var vb = MemoryMarshal.Cast<float, Vector<float>>(b);
            Vector<float> s0 = default, s1 = default, s2 = default, s3 = default;
            int v = 0;
            for (; v + 4 <= va.Length; v += 4)
            {
                s0 = Vector.FusedMultiplyAdd(va[v], vb[v], s0);
                s1 = Vector.FusedMultiplyAdd(va[v + 1], vb[v + 1], s1);
                s2 = Vector.FusedMultiplyAdd(va[v + 2], vb[v + 2], s2);
                s3 = Vector.FusedMultiplyAdd(va[v + 3], vb[v + 3], s3);
            }

            for (; v < va.Length; v++)
            {
                s0 = Vector.FusedMultiplyAdd(va[v], vb[v], s0);
            }

            sum = Vector.Sum((s0 + s1) + (s2 + s3));
            i = va.Length * Vector<float>.Count;
        }

        for (; i < a.Length; i++)
        {
            sum = MathF.FusedMultiplyAdd(a[i], b[i], sum);
        }

        return sum;
    }

    // A tile of rows at a time widened (rows in parallel) into the [columns, k] panel, times each row's scale, and the
    // float32 product reading it transposed (every thread); tiles of 256 rows (k of 3072: 3 MB of panel).
    private static void PanelLane<TW>(StoredLane lane, float[] x, float[] y, int m, int n, int k) where TW : struct, IRowWiden
    {
        int tile = Math.Min(n, 256);
        float[] panel = ArrayPool<float>.Shared.Rent(tile * k);
        float[] c = ArrayPool<float>.Shared.Rent(m * tile);
        try
        {
            for (int j0 = 0; j0 < n; j0 += tile)
            {
                int columns = Math.Min(tile, n - j0), start = j0;
                Parallel.For(0, columns, ComputeResources.ParallelOptions, j => PanelRow<TW>(lane, panel, start + j, j, k));
                CpuMatMul.Multiply(x, 0, panel, 0, c, 0, m, columns, k, false, true, 0f);
                for (int i = 0; i < m; i++)
                {
                    c.AsSpan(i * columns, columns).CopyTo(y.AsSpan(i * n + j0, columns));
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(panel);
            ArrayPool<float>.Shared.Return(c);
        }
    }

    private static void PanelRow<TW>(StoredLane lane, float[] panel, int row, int at, int k) where TW : struct, IRowWiden
    {
        var bytes = MemoryMarshal.AsBytes(lane.Words.AsSpan(0, lane.Length));
        var values = panel.AsSpan(at * k, k);
        TW.Widen(bytes.Slice(checked((int)(row * lane.RowBytes)), (int)lane.RowBytes), 0, values, lane);
        if (lane.RowScales is { } scales)
        {
            float s = scales[row];
            for (int i = 0; i < values.Length; i++)
            {
                values[i] *= s;
            }
        }
    }

    // ------------------------------------------------------------------ 4-bit super-blocks read where they are stored

    // A 4-bit super-block weight's two lanes (NibbleBlocks), neither of which writes the weight's values anywhere, whole or a
    // panel: the decoded lane decodes each block's codes in registers as its dot products read them (the values
    // SuperBlock.Decode gives, bit for bit) and multiplies them with x in float32; the integer lane rounds x to int8 a block
    // of 32 values at a time (one scale a block) and takes the CPU's int8 dot products (AVX-VNNI, else AVX2's multiply-adds)
    // of those with the stored codes, the blocks' float scales applied a block at a time. The measured route keeps the
    // faster at each shape, timed on the real operands.

    // Whether the integer lane runs here: the CPU has int8 dot products (x86's AVX2, with AVX-VNNI's where reported).
    private static bool IntegerDots => Avx2.IsSupported && Fma.IsSupported;

    // The shape's faster nibble lane, measured once on the real operands (Faster); without measuring (CpuTuning.Autotune
    // off) the integer lane where the CPU has int8 dot products, else the decoded lane.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static int MeasureNibbleLanes(NibbleBlocks blocks, StoredLane lane, float[] x, float[] y, int m, int n, int k)
    {
        var key = (lane.Type.Name, LaneKey(lane.Type.Name, m, n, k).Rows, n, k);
        int best = IntegerDots ? IntegerLaneId : DecodedLaneId;
        if (CpuTuning.Autotune && IntegerDots)
        {
            var (decoded, integer) = Faster.Time(() => NibbleProduct(false, blocks, lane, x, y, m, n, k), () => NibbleProduct(true, blocks, lane, x, y, m, n, k));
            best = integer < decoded ? IntegerLaneId : DecodedLaneId;
            ProductRoutes.Chose("cpu", string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{lane.Type.Name} {m}x{k}->{n}"),
                string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{LaneNames[best]} (decoded {decoded:0.###} ms, integer {integer:0.###} ms)"));
        }

        StoredLanes[key] = best;
        return best;
    }

    /// <summary>
    /// y [m, n] = x [m, k] · Wᵀ for W a 4-bit super-block weight of n rows stored as <paramref name="words"/> (its bytes,
    /// <paramref name="length"/> words), by the nibble lane asked (<paramref name="integer"/>: x rounded to int8 a block of
    /// 32 values; else decoded in registers). False where the type's description is not one the lanes read, or the
    /// integer lane is asked of a CPU without int8 dot products. For tests: each lane apart from the measured choice.
    /// </summary>
    internal static bool NibbleLane(bool integer, ElementType type, float[] words, int length, float[] x, float[] y, int m, int n, int k)
    {
        if (type.Super is not { } super || NibbleBlocks.Of(super) is not { } blocks || integer && !IntegerDots || k % super.Values != 0)
        {
            return false;
        }

        NibbleProduct(integer, blocks, new StoredLane(type, null, words, length, null, k, type.ByteCount(k)), x, y, m, n, k);
        return true;
    }

    // The product by one nibble lane: the integer lane's x rounded first (RoundRows), then the outputs' columns in chunks
    // spread over the threads (NibbleColumns), each chunk every row of x.
    private static void NibbleProduct(bool integer, NibbleBlocks blocks, StoredLane lane, float[] x, float[] y, int m, int n, int k)
    {
        int superBlocks = k / NibbleBlocks.Values;
        sbyte[]? codes = null;
        float[]? scales = null;
        try
        {
            if (integer)
            {
                codes = ArrayPool<sbyte>.Shared.Rent(m * k);
                scales = ArrayPool<float>.Shared.Rent(m * superBlocks * 16);
                RoundRows(x, codes, scales, m, k);
            }

            int threads = Math.Max(1, ComputeResources.MaxCpuThreads);
            int chunk = Math.Max(1, (n + 4 * threads - 1) / (4 * threads)), tasks = (n + chunk - 1) / chunk;
            var body = new NibbleRange(integer, blocks, lane, x, codes, scales, y, m, n, k);
            CpuRanges.Run(tasks, chunk, n, threads, ref body);                         // `body` is this frame's, on the stack
        }
        finally
        {
            if (codes is not null)
            {
                ArrayPool<sbyte>.Shared.Return(codes);
            }

            if (scales is not null)
            {
                ArrayPool<float>.Shared.Return(scales);
            }
        }
    }

    // Products of 4-bit super-block weights sharing x, under the measured route (the queries, keys and values; the gate and
    // up): in one pass where all are one such type, any rows; offered once such a weight has been multiplied here, so the
    // groups of other types' weights are not asked (their outputs allocated) only to run one by one.
    public override bool StoredMatMulManyAvailable(int m) => ProductRoutes.Current == ProductRoute.Measured && s_nibblesMultiplied;

    // Whether a 4-bit super-block weight has been multiplied by its nibble lanes in this process.
    private static volatile bool s_nibblesMultiplied;

    // 2 or 3 such products in one pass (NibbleMany): x rounded once where an integer lane runs, the weights' columns spread over
    // the threads as one range, each column by its own product's measured lane (measured here first where it has not been),
    // so each output is its own product's, bit for bit. False for any other weights: the callers then run them one by one.
    public override bool StoredMatMulManyKernel(Storage x, int m, int k, ReadOnlySpan<(Storage Values, Storage? Scales, Storage Output, int Columns)> products)
    {
        var type = products.Length > 0 ? products[0].Values.ElementType : null;
        if (ProductRoutes.Current != ProductRoute.Measured || products.Length is < 2 or > 3 || m <= 0 || k <= 0
            || type?.Super is not { } super || NibbleBlocks.Of(super) is not { } blocks || k % NibbleBlocks.Values != 0)
        {
            return false;
        }

        foreach (var (values, scales, output, n) in products)
        {
            if (values is not CpuStorage || !ReferenceEquals(values.ElementType, type) || n <= 0 || values.ElementCount < (long)n * k
                || scales is not null and not CpuStorage || scales is not null && scales.Length < n || output is not CpuStorage)
            {
                return false;
            }
        }

        float[] xv = D(x);
        var parts = new ThreeParts();
        for (int j = 0; j < products.Length; j++)
        {
            var (values, scales, output, n) = products[j];
            var lane = new StoredLane(type, null, D(values), values.Length, scales is null ? null : D(scales), k, type.ByteCount(k));
            float[] yv = D(output);
            int chosen = StoredLanes.TryGetValue((type.Name, LaneKey(type.Name, m, n, k).Rows, n, k), out int known) ? known : MeasureNibbleLanes(blocks, lane, xv, yv, m, n, k);
            if (chosen is not (DecodedLaneId or IntegerLaneId))
            {
                return false;
            }

            parts[j] = new NibblePart(lane, yv, n, chosen == IntegerLaneId);
        }

        NibbleMany(blocks, ((ReadOnlySpan<NibblePart>)parts)[..products.Length], xv, m, k);
        s_nibblesMultiplied = true;
        return true;
    }

    // One product of a grouped pass: its weight, output, columns and lane.
    private readonly record struct NibblePart(StoredLane Lane, float[] Y, int N, bool Integer);

    // Up to three parts, on the stack.
    [InlineArray(3)]
    private struct ThreeParts
    {
        private NibblePart _part;
    }

    // The products of `parts` in one range of columns over the threads (NibbleProduct's, their columns one after another).
    private static void NibbleMany(NibbleBlocks blocks, ReadOnlySpan<NibblePart> parts, float[] x, int m, int k)
    {
        int superBlocks = k / NibbleBlocks.Values, total = 0;
        bool integer = false;
        foreach (var part in parts)
        {
            total += part.N;
            integer |= part.Integer;
        }

        sbyte[]? codes = null;
        float[]? scales = null;
        try
        {
            if (integer)
            {
                codes = ArrayPool<sbyte>.Shared.Rent(m * k);
                scales = ArrayPool<float>.Shared.Rent(m * superBlocks * 16);
                RoundRows(x, codes, scales, m, k);
            }

            int threads = Math.Max(1, ComputeResources.MaxCpuThreads);
            int chunk = Math.Max(1, (total + 4 * threads - 1) / (4 * threads)), tasks = (total + chunk - 1) / chunk;
            var body = new NibbleManyRange(blocks, x, codes, scales, m, k, parts[0], parts[1], parts.Length > 2 ? parts[2] : default);
            CpuRanges.Run(tasks, chunk, total, threads, ref body);                     // `body` is this frame's, on the stack
        }
        finally
        {
            if (codes is not null)
            {
                ArrayPool<sbyte>.Shared.Return(codes);
            }

            if (scales is not null)
            {
                ArrayPool<float>.Shared.Return(scales);
            }
        }
    }

    // A grouped pass's chunks of columns (CpuRanges): the parts' columns one after another, a chunk split where it crosses
    // from one part to the next.
    private readonly struct NibbleManyRange(NibbleBlocks blocks, float[] x, sbyte[]? xq, float[]? xs, int m, int k, NibblePart p0, NibblePart p1, NibblePart p2) : IRangeBody
    {
        public void Run(int first, int last)
        {
            Part(first, last, 0, p0);
            Part(first, last, p0.N, p1);
            if (p2.Lane is not null)
            {
                Part(first, last, p0.N + p1.N, p2);
            }
        }

        private void Part(int first, int last, int start, NibblePart part)
        {
            int a = Math.Max(first, start), b = Math.Min(last, start + part.N);
            if (a < b)
            {
                NibbleColumns(part.Integer, blocks, part.Lane, x, xq, xs, part.Y, m, part.N, k, a - start, b - start);
            }
        }
    }

    // The nibble lanes' chunks of columns as a range body (CpuRanges: all helpers queued at once, no allocation a call).
    private readonly struct NibbleRange(bool integer, NibbleBlocks blocks, StoredLane lane, float[] x, sbyte[]? xq, float[]? xs, float[] y, int m, int n, int k) : IRangeBody
    {
        public void Run(int first, int last) => NibbleColumns(integer, blocks, lane, x, xq, xs, y, m, n, k, first, last);
    }

    // Columns [j0, j1) of y for every row of x. One row of x: each stored row read once, in order, each block's scales decoded
    // as its codes are dotted (NibbleBlocks.Block). More: the columns' scales decoded a group of rows at a time (a table of
    // at most 32 KB, NibbleBlocks.Scales), then four rows of x at a time against each of the group's weight rows (the four
    // rows of x stay in the cache across the group, the group's stored rows across the rows of x), the rows left over one at
    // a time.
    private static void NibbleColumns(bool integer, NibbleBlocks blocks, StoredLane lane, float[] x, sbyte[]? xq, float[]? xs, float[] y, int m, int n, int k, int j0, int j1)
    {
        int superBlocks = k / NibbleBlocks.Values, rowTable = superBlocks * 16;
        if (m == 1 && blocks.InLanes)
        {
            ref byte first = ref MemoryMarshal.GetReference(MemoryMarshal.AsBytes(lane.Words.AsSpan(0, lane.Length)));
            for (int c = j0; c < j1; c++)
            {
                ref byte row = ref Unsafe.Add(ref first, (nint)(c * lane.RowBytes));
                float sum = integer
                    ? IntegerOne(ref row, blocks.CodeOffset, blocks.Bytes, superBlocks, ref Unsafe.NullRef<float>(), blocks, ref MemoryMarshal.GetArrayDataReference(xq!), ref MemoryMarshal.GetArrayDataReference(xs!))
                    : DecodedOne(ref row, blocks.CodeOffset, blocks.Bytes, superBlocks, ref Unsafe.NullRef<float>(), blocks, ref MemoryMarshal.GetArrayDataReference(x));
                y[c] = lane.RowScales is { } rs ? sum * rs[c] : sum;
            }

            return;
        }

        int group = Math.Clamp(8192 / rowTable, 1, j1 - j0);
        float[] table = ArrayPool<float>.Shared.Rent(group * rowTable);
        try
        {
            var bytes = MemoryMarshal.AsBytes(lane.Words.AsSpan(0, lane.Length));
            ref byte stored = ref MemoryMarshal.GetReference(bytes);
            ref float scaleTable = ref MemoryMarshal.GetArrayDataReference(table);
            ref float xf = ref MemoryMarshal.GetArrayDataReference(x);
            ref sbyte xc = ref xq is null ? ref Unsafe.NullRef<sbyte>() : ref MemoryMarshal.GetArrayDataReference(xq);
            ref float xsc = ref xs is null ? ref Unsafe.NullRef<float>() : ref MemoryMarshal.GetArrayDataReference(xs);
            int bytesPerBlock = blocks.Bytes, codeOffset = blocks.CodeOffset;
            for (int c0 = j0; c0 < j1; c0 += group)
            {
                int count = Math.Min(group, j1 - c0);
                for (int c = 0; c < count; c++)
                {
                    blocks.Scales(bytes.Slice(checked((int)((c0 + c) * lane.RowBytes)), (int)lane.RowBytes), superBlocks, table.AsSpan(c * rowTable, rowTable));
                }

                int i = 0;
                for (; i + 4 <= m; i += 4)
                {
                    for (int c = 0; c < count; c++)
                    {
                        ref byte codes = ref Unsafe.Add(ref stored, (nint)((c0 + c) * lane.RowBytes) + codeOffset);
                        ref float rowScales = ref Unsafe.Add(ref scaleTable, c * rowTable);
                        float s0, s1, s2, s3;
                        if (integer)
                        {
                            IntegerFour(ref codes, bytesPerBlock, superBlocks, ref rowScales, ref Unsafe.Add(ref xc, (nint)i * k), ref Unsafe.Add(ref xsc, (nint)i * rowTable), k, out s0, out s1, out s2, out s3);
                        }
                        else
                        {
                            DecodedFour(ref codes, bytesPerBlock, superBlocks, ref rowScales, ref Unsafe.Add(ref xf, (nint)i * k), k, out s0, out s1, out s2, out s3);
                        }

                        float scale = lane.RowScales is { } rs ? rs[c0 + c] : 1f;
                        int at = i * n + c0 + c;
                        y[at] = lane.RowScales is null ? s0 : s0 * scale;
                        y[at + n] = lane.RowScales is null ? s1 : s1 * scale;
                        y[at + 2 * n] = lane.RowScales is null ? s2 : s2 * scale;
                        y[at + 3 * n] = lane.RowScales is null ? s3 : s3 * scale;
                    }
                }

                for (; i < m; i++)
                {
                    for (int c = 0; c < count; c++)
                    {
                        ref byte row = ref Unsafe.Add(ref stored, (nint)((c0 + c) * lane.RowBytes));
                        ref float rowScales = ref Unsafe.Add(ref scaleTable, c * rowTable);
                        float sum = integer
                            ? IntegerOne(ref row, codeOffset, bytesPerBlock, superBlocks, ref rowScales, null, ref Unsafe.Add(ref xc, (nint)i * k), ref Unsafe.Add(ref xsc, (nint)i * rowTable))
                            : DecodedOne(ref row, codeOffset, bytesPerBlock, superBlocks, ref rowScales, null, ref Unsafe.Add(ref xf, (nint)i * k));
                        y[i * n + c0 + c] = lane.RowScales is { } rs ? sum * rs[c0 + c] : sum;
                    }
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(table);
        }
    }

    // ---- the integer lane

    // Each step's lane l (bytes 4l..4l + 3) of a super-block's codes, as the integer lane's kernels read them, from x's codes
    // in value order, in 4-byte words: step u = 2t + s pairs with the stored codes' word t of each group (s = 0) or word
    // t + 4 (s = 1), low halves in lanes 0-3 (sub-blocks 0, 2, 4, 6) and high halves in lanes 4-7 (sub-blocks 1, 3, 5, 7):
    // word 16g + 8h + 4(s xor h) + t of the values, for g = l mod 4 and h = l / 4 (IntegerFour's steps).
    private static readonly int[] StepWords = [.. Enumerable.Range(0, 64).Select(p =>
    {
        int u = p / 8, l = p % 8, t = u / 2, s = u % 2, g = l % 4, h = l / 4;
        return 16 * g + 8 * h + 4 * (s ^ h) + t;
    })];

    // x rounded to int8 for the integer lane, row by row (RoundRow), over the threads where x is large.
    private static void RoundRows(float[] x, sbyte[] codes, float[] scales, int m, int k)
    {
        int table = k / NibbleBlocks.Values * 16;
        if (m > 1 && (long)m * k >= 1 << 16)
        {
            Parallel.For(0, m, ComputeResources.ParallelOptions, i => RoundRow(x.AsSpan(i * k, k), codes.AsSpan(i * k, k), scales.AsSpan(i * table, table)));
            return;
        }

        for (int i = 0; i < m; i++)
        {
            RoundRow(x.AsSpan(i * k, k), codes.AsSpan(i * k, k), scales.AsSpan(i * table, table));
        }
    }

    // A row of x rounded to int8 a block of 32 values at a time: the block's scale amax / 127 (0 where every value is 0), its
    // codes value · 127 / amax rounded to even, laid out a super-block at a time in the kernels' step order (StepWords); per
    // super-block 16 floats: its blocks' scales at 0-7 and each scale times its codes' sum at 8-15, in lane order
    // (sub-blocks 0, 2, 4, 6, 1, 3, 5, 7). The integer lane's product is then the product of x so rounded.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void RoundRow(ReadOnlySpan<float> x, Span<sbyte> codes, Span<float> scales)
    {
        Span<int> natural = stackalloc int[64];
        ref float row = ref MemoryMarshal.GetReference(x);
        for (int b = 0; b < x.Length / NibbleBlocks.Values; b++)
        {
            var into = scales.Slice(b * 16, 16);
            for (int s = 0; s < 8; s++)
            {
                ref float v = ref Unsafe.Add(ref row, b * NibbleBlocks.Values + s * 32);
                var v0 = Vector256.LoadUnsafe(ref v);
                var v1 = Vector256.LoadUnsafe(ref v, 8);
                var v2 = Vector256.LoadUnsafe(ref v, 16);
                var v3 = Vector256.LoadUnsafe(ref v, 24);
                var most = Vector256.Max(Vector256.Max(Vector256.Abs(v0), Vector256.Abs(v1)), Vector256.Max(Vector256.Abs(v2), Vector256.Abs(v3)));
                var half = Vector128.Max(most.GetLower(), most.GetUpper());
                half = Vector128.Max(half, Vector128.Shuffle(half, Vector128.Create(2, 3, 0, 1)));
                half = Vector128.Max(half, Vector128.Shuffle(half, Vector128.Create(1, 0, 3, 2)));
                float amax = half.ToScalar(), scale = amax / 127f;
                var inverse = Vector256.Create(amax > 0f ? 127f / amax : 0f);
                var q0 = Vector256.ConvertToInt32(Vector256.Round(v0 * inverse));
                var q1 = Vector256.ConvertToInt32(Vector256.Round(v1 * inverse));
                var q2 = Vector256.ConvertToInt32(Vector256.Round(v2 * inverse));
                var q3 = Vector256.ConvertToInt32(Vector256.Round(v3 * inverse));
                int sum = Vector256.Sum((q0 + q1) + (q2 + q3));
                Vector256.Narrow(Vector256.Narrow(q0, q1), Vector256.Narrow(q2, q3)).AsInt32().CopyTo(natural.Slice(s * 8, 8));
                int at = (s & 1) * 4 + (s >> 1);
                into[at] = scale;
                into[8 + at] = scale * sum;
            }

            var words = MemoryMarshal.Cast<sbyte, int>(codes.Slice(b * NibbleBlocks.Values, NibbleBlocks.Values));
            for (int p = 0; p < 64; p++)
            {
                words[p] = natural[StepWords[p]];
            }
        }
    }

    // Lanes 0-3 of `words` shifted by nothing and lanes 4-7 by 4, each byte's low four bits: from a group-word vector
    // [word t of groups 0-3 | word t + 4 of groups 0-3], the low halves of word t (sub-blocks 0, 2, 4, 6) and the high
    // halves of word t + 4 (sub-blocks 1, 3, 5, 7).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> Halves(Vector256<uint> words, Vector256<uint> shifts, Vector256<byte> low) =>
        Avx2.ShiftRightLogicalVariable(words, shifts).AsByte() & low;

    // sum + the dot products of 4 codes (0-15) by 4 of x's int8 codes, in each 32-bit lane (x's 32 bytes from `at`).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> Dot(Vector256<int> sum, Vector256<byte> codes, ref sbyte x, nint at)
    {
        var values = Vector256.LoadUnsafe(ref x, (nuint)at);
        return AvxVnni.IsSupported
            ? AvxVnni.MultiplyWideningAndAdd(sum, codes, values)
            : sum + Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(codes, values), Vector256.Create((short)1));
    }

    // One super-block's sums for a row of x, its 8 lanes the 8 sub-blocks (lane order), as floats: Σ (code · scale_b) times
    // x's scale plus −dmin · minimum_b times x's scale · Σ x codes, added to `sum` (scales: the row's 16 of the block).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Scaled(Vector256<float> sum, Vector256<int> dots, Vector256<float> scales, Vector256<float> minimums, ref float x)
    {
        sum = Fma.MultiplyAdd(Avx.ConvertToVector256Single(dots) * Vector256.LoadUnsafe(ref x), scales, sum);
        return Fma.MultiplyAdd(minimums, Vector256.LoadUnsafe(ref x, 8), sum);
    }

    // Four rows of x (codes from `x`, k a row; their scales from `xs`, 16 a super-block) by one stored row (its codes from
    // `codes`, a super-block every `blockBytes`; its scales from `table`, 16 a super-block): a super-block's four groups of
    // 32 code bytes transposed by words, so each step's lane l holds four codes of sub-block (0, 2, 4, 6, 1, 3, 5, 7)[l] and
    // the int8 dot products of a whole super-block land in one register a row, lane per sub-block; then the float scales.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void IntegerFour(ref byte codes, int blockBytes, int blocks, ref float table, ref sbyte x, ref float xs, int k,
        out float y0, out float y1, out float y2, out float y3)
    {
        var low = Vector256.Create((byte)15);
        var shifts = Vector256.Create(0u, 0u, 0u, 0u, 4u, 4u, 4u, 4u);
        nint r1 = k, r2 = 2 * (nint)k, r3 = 3 * (nint)k, s1 = blocks * 16, s2 = 2 * s1, s3 = 3 * s1;
        Vector256<float> f0 = default, f1 = default, f2 = default, f3 = default;
        for (int b = 0; b < blocks; b++)
        {
            ref byte g = ref Unsafe.Add(ref codes, (nint)b * blockBytes);
            var g0 = Vector256.LoadUnsafe(ref g).AsUInt32();
            var g1 = Vector256.LoadUnsafe(ref g, 32).AsUInt32();
            var g2 = Vector256.LoadUnsafe(ref g, 64).AsUInt32();
            var g3 = Vector256.LoadUnsafe(ref g, 96).AsUInt32();
            var p01 = Avx2.UnpackLow(g0, g1).AsUInt64();
            var p23 = Avx2.UnpackLow(g2, g3).AsUInt64();
            var h01 = Avx2.UnpackHigh(g0, g1).AsUInt64();
            var h23 = Avx2.UnpackHigh(g2, g3).AsUInt64();
            ref sbyte xb = ref Unsafe.Add(ref x, (nint)b * NibbleBlocks.Values);
            Vector256<int> i0 = default, i1 = default, i2 = default, i3 = default;

            // Word t of each group, t = 0..3: [t | t + 4] of groups 0-3, then its halves swapped.
            var q = Avx2.UnpackLow(p01, p23).AsUInt32();
            var w = Halves(q, shifts, low);
            var v = Halves(Avx2.Permute2x128(q, q, 1), shifts, low);
            i0 = Dot(i0, w, ref xb, 0);
            i0 = Dot(i0, v, ref xb, 32);
            i1 = Dot(i1, w, ref xb, r1);
            i1 = Dot(i1, v, ref xb, r1 + 32);
            i2 = Dot(i2, w, ref xb, r2);
            i2 = Dot(i2, v, ref xb, r2 + 32);
            i3 = Dot(i3, w, ref xb, r3);
            i3 = Dot(i3, v, ref xb, r3 + 32);
            q = Avx2.UnpackHigh(p01, p23).AsUInt32();
            w = Halves(q, shifts, low);
            v = Halves(Avx2.Permute2x128(q, q, 1), shifts, low);
            i0 = Dot(i0, w, ref xb, 64);
            i0 = Dot(i0, v, ref xb, 96);
            i1 = Dot(i1, w, ref xb, r1 + 64);
            i1 = Dot(i1, v, ref xb, r1 + 96);
            i2 = Dot(i2, w, ref xb, r2 + 64);
            i2 = Dot(i2, v, ref xb, r2 + 96);
            i3 = Dot(i3, w, ref xb, r3 + 64);
            i3 = Dot(i3, v, ref xb, r3 + 96);
            q = Avx2.UnpackLow(h01, h23).AsUInt32();
            w = Halves(q, shifts, low);
            v = Halves(Avx2.Permute2x128(q, q, 1), shifts, low);
            i0 = Dot(i0, w, ref xb, 128);
            i0 = Dot(i0, v, ref xb, 160);
            i1 = Dot(i1, w, ref xb, r1 + 128);
            i1 = Dot(i1, v, ref xb, r1 + 160);
            i2 = Dot(i2, w, ref xb, r2 + 128);
            i2 = Dot(i2, v, ref xb, r2 + 160);
            i3 = Dot(i3, w, ref xb, r3 + 128);
            i3 = Dot(i3, v, ref xb, r3 + 160);
            q = Avx2.UnpackHigh(h01, h23).AsUInt32();
            w = Halves(q, shifts, low);
            v = Halves(Avx2.Permute2x128(q, q, 1), shifts, low);
            i0 = Dot(i0, w, ref xb, 192);
            i0 = Dot(i0, v, ref xb, 224);
            i1 = Dot(i1, w, ref xb, r1 + 192);
            i1 = Dot(i1, v, ref xb, r1 + 224);
            i2 = Dot(i2, w, ref xb, r2 + 192);
            i2 = Dot(i2, v, ref xb, r2 + 224);
            i3 = Dot(i3, w, ref xb, r3 + 192);
            i3 = Dot(i3, v, ref xb, r3 + 224);

            var scales = Vector256.LoadUnsafe(ref table, (nuint)(b * 16));
            var minimums = Vector256.LoadUnsafe(ref table, (nuint)(b * 16 + 8));
            ref float sb = ref Unsafe.Add(ref xs, b * 16);
            f0 = Scaled(f0, i0, scales, minimums, ref sb);
            f1 = Scaled(f1, i1, scales, minimums, ref Unsafe.Add(ref sb, s1));
            f2 = Scaled(f2, i2, scales, minimums, ref Unsafe.Add(ref sb, s2));
            f3 = Scaled(f3, i3, scales, minimums, ref Unsafe.Add(ref sb, s3));
        }

        y0 = Vector256.Sum(f0);
        y1 = Vector256.Sum(f1);
        y2 = Vector256.Sum(f2);
        y3 = Vector256.Sum(f3);
    }

    // IntegerFour for one row of x, the stored row's blocks from `row` (codes `codeOffset` into each); each block's scales from
    // `table`, or, with `decode`, decoded from the block as it is read (NibbleBlocks.Block: no table).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float IntegerOne(ref byte row, int codeOffset, int blockBytes, int blocks, ref float table, NibbleBlocks? decode, ref sbyte x, ref float xs)
    {
        var low = Vector256.Create((byte)15);
        var shifts = Vector256.Create(0u, 0u, 0u, 0u, 4u, 4u, 4u, 4u);
        Span<float> own = stackalloc float[16];
        ref float decoded = ref MemoryMarshal.GetReference(own);
        Vector256<float> f = default;
        for (int b = 0; b < blocks; b++)
        {
            ref byte block = ref Unsafe.Add(ref row, (nint)b * blockBytes);
            scoped ref float factors = ref Unsafe.Add(ref table, b * 16);
            if (decode is not null)
            {
                decode.Block(ref block, ref decoded);
                factors = ref decoded;
            }

            ref byte g = ref Unsafe.Add(ref block, codeOffset);
            var g0 = Vector256.LoadUnsafe(ref g).AsUInt32();
            var g1 = Vector256.LoadUnsafe(ref g, 32).AsUInt32();
            var g2 = Vector256.LoadUnsafe(ref g, 64).AsUInt32();
            var g3 = Vector256.LoadUnsafe(ref g, 96).AsUInt32();
            var p01 = Avx2.UnpackLow(g0, g1).AsUInt64();
            var p23 = Avx2.UnpackLow(g2, g3).AsUInt64();
            var h01 = Avx2.UnpackHigh(g0, g1).AsUInt64();
            var h23 = Avx2.UnpackHigh(g2, g3).AsUInt64();
            ref sbyte xb = ref Unsafe.Add(ref x, (nint)b * NibbleBlocks.Values);
            var q = Avx2.UnpackLow(p01, p23).AsUInt32();
            var i = Dot(Dot(default, Halves(q, shifts, low), ref xb, 0), Halves(Avx2.Permute2x128(q, q, 1), shifts, low), ref xb, 32);
            q = Avx2.UnpackHigh(p01, p23).AsUInt32();
            i = Dot(Dot(i, Halves(q, shifts, low), ref xb, 64), Halves(Avx2.Permute2x128(q, q, 1), shifts, low), ref xb, 96);
            q = Avx2.UnpackLow(h01, h23).AsUInt32();
            i = Dot(Dot(i, Halves(q, shifts, low), ref xb, 128), Halves(Avx2.Permute2x128(q, q, 1), shifts, low), ref xb, 160);
            q = Avx2.UnpackHigh(h01, h23).AsUInt32();
            i = Dot(Dot(i, Halves(q, shifts, low), ref xb, 192), Halves(Avx2.Permute2x128(q, q, 1), shifts, low), ref xb, 224);
            f = Scaled(f, i, Vector256.LoadUnsafe(ref factors), Vector256.LoadUnsafe(ref factors, 8), ref Unsafe.Add(ref xs, b * 16));
        }

        return Vector256.Sum(f);
    }

    // ---- the decoded lane

    // 8 code bytes from `at` as 32-bit integers (zero-extended).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> Codes8(ref byte at)
    {
        var bytes = Vector128.CreateScalarUnsafe(Unsafe.ReadUnaligned<ulong>(ref at)).AsByte();
        if (Avx2.IsSupported)
        {
            return Avx2.ConvertToVector256Int32(bytes);
        }

        var shorts = Vector128.WidenLower(bytes);
        return Vector256.Create(Vector128.WidenLower(shorts), Vector128.WidenUpper(shorts)).AsInt32();
    }

    // Four rows of x (float32, k a row, from `x`) by one stored row (its codes from `codes`, a super-block every
    // `blockBytes`; its scales from `table`, 16 a super-block): 8 codes at a time decoded in registers to the values
    // SuperBlock.Decode gives (code · scale − minimum: a product, then a sum, as it rounds them), each multiplied with the
    // four rows; the low and high halves of a group (sub-blocks 2g and 2g + 1) in their own sums, so each sum's FMAs wait on
    // each other half as often.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void DecodedFour(ref byte codes, int blockBytes, int blocks, ref float table, ref float x, int k,
        out float y0, out float y1, out float y2, out float y3)
    {
        var mask = Vector256.Create(15);
        nint r1 = k, r2 = 2 * (nint)k, r3 = 3 * (nint)k;
        Vector256<float> l0 = default, l1 = default, l2 = default, l3 = default, h0 = default, h1 = default, h2 = default, h3 = default;
        for (int b = 0; b < blocks; b++)
        {
            ref byte block = ref Unsafe.Add(ref codes, (nint)b * blockBytes);
            ref float xb = ref Unsafe.Add(ref x, (nint)b * NibbleBlocks.Values);
            ref float scales = ref Unsafe.Add(ref table, b * 16);
            for (int g = 0; g < 4; g++)
            {
                // Sub-block 2g's scale and minimum in lane g of the table, 2g + 1's in lane 4 + g.
                var dl = Vector256.Create(Unsafe.Add(ref scales, g));
                var dh = Vector256.Create(Unsafe.Add(ref scales, 4 + g));
                var ml = Vector256.Create(Unsafe.Add(ref scales, 8 + g));
                var mh = Vector256.Create(Unsafe.Add(ref scales, 12 + g));
                ref byte group = ref Unsafe.Add(ref block, 32 * g);
                ref float xg = ref Unsafe.Add(ref xb, 64 * g);
                for (int p = 0; p < 32; p += 8)
                {
                    var c = Codes8(ref Unsafe.Add(ref group, p));
                    var wl = Vector256.ConvertToSingle(c & mask) * dl + ml;
                    var wh = Vector256.ConvertToSingle(Vector256.ShiftRightLogical(c, 4)) * dh + mh;
                    ref float xp = ref Unsafe.Add(ref xg, p);
                    l0 = Vector256.FusedMultiplyAdd(wl, Vector256.LoadUnsafe(ref xp), l0);
                    h0 = Vector256.FusedMultiplyAdd(wh, Vector256.LoadUnsafe(ref xp, 32), h0);
                    l1 = Vector256.FusedMultiplyAdd(wl, Vector256.LoadUnsafe(ref Unsafe.Add(ref xp, r1)), l1);
                    h1 = Vector256.FusedMultiplyAdd(wh, Vector256.LoadUnsafe(ref Unsafe.Add(ref xp, r1), 32), h1);
                    l2 = Vector256.FusedMultiplyAdd(wl, Vector256.LoadUnsafe(ref Unsafe.Add(ref xp, r2)), l2);
                    h2 = Vector256.FusedMultiplyAdd(wh, Vector256.LoadUnsafe(ref Unsafe.Add(ref xp, r2), 32), h2);
                    l3 = Vector256.FusedMultiplyAdd(wl, Vector256.LoadUnsafe(ref Unsafe.Add(ref xp, r3)), l3);
                    h3 = Vector256.FusedMultiplyAdd(wh, Vector256.LoadUnsafe(ref Unsafe.Add(ref xp, r3), 32), h3);
                }
            }
        }

        y0 = Vector256.Sum(l0 + h0);
        y1 = Vector256.Sum(l1 + h1);
        y2 = Vector256.Sum(l2 + h2);
        y3 = Vector256.Sum(l3 + h3);
    }

    // DecodedFour for one row of x, the blocks and their scales as IntegerOne takes them.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float DecodedOne(ref byte row, int codeOffset, int blockBytes, int blocks, ref float table, NibbleBlocks? decode, ref float x)
    {
        var mask = Vector256.Create(15);
        Span<float> own = stackalloc float[16];
        ref float decoded = ref MemoryMarshal.GetReference(own);
        Vector256<float> l = default, h = default;
        for (int b = 0; b < blocks; b++)
        {
            ref byte start = ref Unsafe.Add(ref row, (nint)b * blockBytes);
            scoped ref float scales = ref Unsafe.Add(ref table, b * 16);
            if (decode is not null)
            {
                decode.Block(ref start, ref decoded);
                scales = ref decoded;
            }

            ref byte block = ref Unsafe.Add(ref start, codeOffset);
            ref float xb = ref Unsafe.Add(ref x, (nint)b * NibbleBlocks.Values);
            for (int g = 0; g < 4; g++)
            {
                var dl = Vector256.Create(Unsafe.Add(ref scales, g));
                var dh = Vector256.Create(Unsafe.Add(ref scales, 4 + g));
                var ml = Vector256.Create(Unsafe.Add(ref scales, 8 + g));
                var mh = Vector256.Create(Unsafe.Add(ref scales, 12 + g));
                ref byte group = ref Unsafe.Add(ref block, 32 * g);
                ref float xg = ref Unsafe.Add(ref xb, 64 * g);
                for (int p = 0; p < 32; p += 8)
                {
                    var c = Codes8(ref Unsafe.Add(ref group, p));
                    l = Vector256.FusedMultiplyAdd(Vector256.ConvertToSingle(c & mask) * dl + ml, Vector256.LoadUnsafe(ref Unsafe.Add(ref xg, p)), l);
                    h = Vector256.FusedMultiplyAdd(Vector256.ConvertToSingle(Vector256.ShiftRightLogical(c, 4)) * dh + mh, Vector256.LoadUnsafe(ref Unsafe.Add(ref xg, p + 32)), h);
                }
            }
        }

        return Vector256.Sum(l + h);
    }

    /// <summary>
    /// A super-block format the nibble lanes read where it is stored, by its description alone (never its name): 256 values
    /// a block, one plane of 4-bit codes in groups of 64 (a group's 32 bytes hold its values 0-31 in their low halves and
    /// 32-63 in their high halves), no code bias, sub-blocks of 32 values each with its scale and minimum from pieces, the
    /// super-scales float16 or bfloat16 (ggml's Q4_K). Null for any other description, or where 256-bit vectors do not run.
    /// </summary>
    private sealed class NibbleBlocks
    {
        /// <summary>Values a block holds.</summary>
        public const int Values = 256;

        private static readonly ConditionalWeakTable<SuperBlock, object> Plans = new();

        private readonly SuperBlock super;

        // Fields 0-7 the sub-blocks' scales, 8-15 their minimums: field f's pieces at [start[f], start[f + 1]), each its byte,
        // shift, mask and place in the value; bits[f] the field's width.
        private readonly int[] start, offset, shift, mask, place, bits;

        private NibbleBlocks(SuperBlock super, IReadOnlyList<IReadOnlyList<ScalePiece>> fields)
        {
            this.super = super;
            CodeOffset = super.Planes[0].Offset;
            start = new int[fields.Count + 1];
            bits = new int[fields.Count];
            var (o, s, k, p) = (new List<int>(), new List<int>(), new List<int>(), new List<int>());
            for (int f = 0; f < fields.Count; f++)
            {
                int at = 0;
                foreach (var piece in fields[f])
                {
                    o.Add(piece.Offset);
                    s.Add(piece.Shift);
                    k.Add((1 << piece.Bits) - 1);
                    p.Add(at);
                    at += piece.Bits;
                }

                bits[f] = at;
                start[f + 1] = o.Count;
            }

            (offset, shift, mask, place) = ([.. o], [.. s], [.. k], [.. p]);
            pieces = Avx2.IsSupported ? PiecesInLanes() : null;
        }

        // The fields' pieces as vector controls (PiecesInLanes), or null: the scalar Read.
        private sealed record LanePieces(int Window, Vector256<byte>[] Bytes, Vector256<uint>[] Shifts, Vector256<uint>[] Masks, Vector256<uint>[] Places, Vector256<uint> Widths);

        private readonly LanePieces? pieces;

        /// <summary>Whether a block's scales decode as vectors (<see cref="Block"/>).</summary>
        public bool InLanes => pieces is not null;

        // The fields' pieces as vector controls, a vector each piece index, lane l the field of sub-block (0, 2, 4, 6, 1, 3,
        // 5, 7)[l] (the scales, then the minimums): byte shuffles that put each piece's byte, from the 16 bytes at `window`,
        // in its lane's low byte, then its shift, mask and place. Null where the pieces' bytes span more than 16.
        private LanePieces? PiecesInLanes()
        {
            int window = offset.Length == 0 ? 0 : offset.Min(), most = Enumerable.Range(0, bits.Length).Max(f => start[f + 1] - start[f]);
            if (offset.Length > 0 && (offset.Max() - window >= 16 || window + 16 > super.Bytes))
            {
                return null;
            }

            int fields = bits.Length;
            var (bytes, shifts, masks, places) = (new Vector256<byte>[2 * most], new Vector256<uint>[2 * most], new Vector256<uint>[2 * most], new Vector256<uint>[2 * most]);
            Span<byte> control = stackalloc byte[32];
            Span<uint> s = stackalloc uint[8], m = stackalloc uint[8], p = stackalloc uint[8], widths = stackalloc uint[8];
            for (int half = 0; half < 2; half++)
            {
                for (int q = 0; q < most; q++)
                {
                    control.Fill(0x80);
                    for (int l = 0; l < 8; l++)
                    {
                        int field = half * 8 + (l < 4 ? 2 * l : 2 * (l - 4) + 1), piece = start[Math.Min(field, fields - 1)] + q;
                        bool has = field < fields && piece < start[field + 1];
                        control[4 * l] = has ? (byte)(offset[piece] - window) : (byte)0x80;
                        (s[l], m[l], p[l]) = has ? ((uint)shift[piece], (uint)mask[piece], (uint)place[piece]) : (0u, 0u, 0u);
                        widths[l] = half == 0 && field < fields ? (uint)(32 - bits[field]) : 0u;
                    }

                    bytes[half * most + q] = Vector256.Create<byte>(control);
                    (shifts[half * most + q], masks[half * most + q], places[half * most + q]) = (Vector256.Create<uint>(s), Vector256.Create<uint>(m), Vector256.Create<uint>(p));
                }
            }

            return new LanePieces(window, bytes, shifts, masks, places, Vector256.Create<uint>(widths));
        }

        /// <summary>
        /// One block's 16 factors (<see cref="Scales"/>' layout) into <paramref name="into"/>, from the block at
        /// <paramref name="block"/>: its fields 8 at a time (FieldsInLanes), the same integers as Read, so the same floats.
        /// Only where <see cref="InLanes"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Block(ref byte block, ref float into)
        {
            var lanes = pieces!;
            float d = Super(ref block, super.ScaleOffset), dmin = super.MinOffset >= 0 ? Super(ref block, super.MinOffset) : 0f;
            var at = Vector128.LoadUnsafe(ref Unsafe.Add(ref block, lanes.Window));
            var both = Vector256.Create(at, at);
            var scales = FieldsInLanes(both, 0, lanes.Bytes, lanes.Shifts, lanes.Masks, lanes.Places);
            if (super.SubScalesSigned)
            {
                scales = Avx2.ShiftRightArithmeticVariable(Avx2.ShiftLeftLogicalVariable(scales.AsUInt32(), lanes.Widths).AsInt32(), lanes.Widths);
            }

            (Vector256.ConvertToSingle(scales + Vector256.Create(super.SubScaleBias)) * Vector256.Create(d)).StoreUnsafe(ref into);
            var lows = super.SubMins is not null
                ? -(Vector256.ConvertToSingle(FieldsInLanes(both, 1, lanes.Bytes, lanes.Shifts, lanes.Masks, lanes.Places)) * Vector256.Create(dmin))
                : Vector256.Create(-0f);
            lows.StoreUnsafe(ref into, 8);
        }

        // The super-scale at byte `at` of a block: SuperBlock.FloatAt's float (float16 or bfloat16 bits), read in place.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float Super(ref byte block, int at)
        {
            ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(MemoryMarshal.CreateReadOnlySpan(ref Unsafe.Add(ref block, at), 2));
            return super.Scale.Exponent == 8 ? BitConverter.Int32BitsToSingle(bits << 16) : (float)BitConverter.UInt16BitsToHalf(bits);
        }

        // Fields [8 · half, 8 · half + 8) of a block, lane l sub-block (0, 2, 4, 6, 1, 3, 5, 7)[l]'s (Read's values).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<int> FieldsInLanes(Vector256<byte> window, int half, Vector256<byte>[] bytes, Vector256<uint>[] shifts, Vector256<uint>[] masks, Vector256<uint>[] places)
        {
            int most = bytes.Length / 2;
            Vector256<uint> value = default;
            for (int q = half * most; q < half * most + most; q++)
            {
                var piece = Avx2.Shuffle(window, bytes[q]).AsUInt32();
                value |= Avx2.ShiftLeftLogicalVariable(Avx2.ShiftRightLogicalVariable(piece, shifts[q]) & masks[q], places[q]);
            }

            return value.AsInt32();
        }

        /// <summary>Bytes a block takes.</summary>
        public int Bytes => super.Bytes;

        /// <summary>The first code byte in a block.</summary>
        public int CodeOffset { get; }

        /// <summary>The plan for <paramref name="super"/>, or null where the lanes do not read it.</summary>
        public static NibbleBlocks? Of(SuperBlock super) => Plans.GetValue(super, static s => (object?)Make(s) ?? s) as NibbleBlocks;

        private static NibbleBlocks? Make(SuperBlock s)
        {
            bool reads = Vector256.IsHardwareAccelerated && BitConverter.IsLittleEndian
                && s is { Values: Values, SubValues: 32, CodeBias: 0, Planes: [{ Bits: 4, Group: 64 } plane], SubScales.Count: 8 }
                && plane.Offset >= 0 && plane.Offset + 128 <= s.Bytes
                && s.Scale is { Exponent: 5, Mantissa: 10 } or { Exponent: 8, Mantissa: 7 }
                && s.SubMins is null or { Count: 8 };
            return reads ? new NibbleBlocks(s, [.. s.SubScales, .. s.SubMins ?? []]) : null;
        }

        /// <summary>
        /// The scales of a stored row's <paramref name="blocks"/> blocks into <paramref name="table"/>, 16 floats a block:
        /// its sub-blocks' d · (scale + bias) at 0-7 and −(dmin · minimum) at 8-15 (−0 without minimums), in lane order
        /// (sub-blocks 0, 2, 4, 6, 1, 3, 5, 7): the factors <see cref="SuperBlock.Decode"/> takes, so code · scale + the
        /// second is its value, bit for bit.
        /// </summary>
        public void Scales(ReadOnlySpan<byte> row, int blocks, Span<float> table)
        {
            bool minimums = super.SubMins is not null;
            if (pieces is not null && row.Length >= blocks * super.Bytes && table.Length >= blocks * 16)
            {
                ref byte first = ref MemoryMarshal.GetReference(row);
                ref float into = ref MemoryMarshal.GetReference(table);
                for (int b = 0; b < blocks; b++)
                {
                    Block(ref Unsafe.Add(ref first, b * super.Bytes), ref Unsafe.Add(ref into, b * 16));
                }

                return;
            }

            for (int b = 0; b < blocks; b++)
            {
                var block = row.Slice(b * super.Bytes, super.Bytes);
                float d = super.FloatAt(block, super.ScaleOffset), dmin = super.MinOffset >= 0 ? super.FloatAt(block, super.MinOffset) : 0f;
                var into = table.Slice(b * 16, 16);
                for (int j = 0; j < 8; j++)
                {
                    int at = (j & 1) * 4 + (j >> 1);
                    into[at] = d * (Read(block, j, super.SubScalesSigned) + super.SubScaleBias);
                    into[8 + at] = minimums ? -(dmin * Read(block, 8 + j, false)) : -0f;
                }
            }
        }

        // Field `field` of a block from its pieces (SuperBlock.Assemble's value).
        private int Read(ReadOnlySpan<byte> block, int field, bool signed)
        {
            int value = 0;
            for (int p = start[field]; p < start[field + 1]; p++)
            {
                value |= (block[offset[p]] >> shift[p] & mask[p]) << place[p];
            }

            int width = bits[field];
            return signed ? value << (32 - width) >> (32 - width) : value;
        }
    }
}
