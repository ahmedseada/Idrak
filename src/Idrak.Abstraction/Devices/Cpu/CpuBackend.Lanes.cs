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
// (Faster). These formats are read where they are stored and never widened under the measured route (IRowFormat, the
// shared part below): a 4-bit super-block (NibbleFormat: ggml's Q4_K by its description), a 32-value block of 4-bit
// codes with one float16 scale or of E2M1 values sharing an E8M0 scale (ScaledFormat: ggml's q4_0, OCP's MXFP4), a
// 6-bit super-block (SixFormat: ggml's Q6_K), 64-bit integer codes with a scale an output row (LongFormat: int64) and a
// 32-value block of 2-bit codes with one float16 scale (TwoBitFormat: int2), each with its codes decoded in registers
// inside the dot products or int8 dot products over the stored codes with x rounded to int8 a block of 32 values; bfloat16
// (BFloat16Format), its values made float32 in registers; and float64 (DoubleFormat), its values multiplied as they are
// in float64 FMAs; the faster lane at each shape. Under the measured route the described lane (DescribedFormat, at the
// end) reads any type from its description alone (ElementType.Description) and is the first candidate; the formats above
// are candidates beside it, kept at a shape where they measure faster (ProductRoutes.Lanes: all, described, written).
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

    // Stored row j of a weight's bytes (a weight past 2 GB too: StoredSpan).
    private static ReadOnlySpan<byte> StoredRow(StoredLane lane, int j) => StoredSpan(lane.Words, lane.Length, j * lane.RowBytes, (int)lane.RowBytes);

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

        if (type.Scaled is { Scale: { Exponent: 5, Mantissa: 10 }, CodeType: null, ScaleType: null } block && (block.CodeBits == 8 && block.CodeTable is null
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
        // A weight the described lane reads (any type with a description it takes) or a written format reads (FormatOf: q4_k,
        // q4_0, mxfp4, q6_k, bfloat16, int64, float64, int2) under the measured route: only those lanes, which never widen
        // it, the shape's lane among them measured once (Chosen; ProductRoutes.Lanes says which are candidates).
        if (ProductRoutes.Current == ProductRoute.Measured && CandidatesOf(type, k, new RoutedUse(lane, D(x), D(y), m, n, k)))
        {
            s_formatsMultiplied = true;
            return true;
        }

        var widening = WideningOf(type);
        int chosen = ProductRoutes.Current == ProductRoute.Generic
            ? m <= Int8DirectRows ? RowLaneId : PanelLaneId
            : StoredLanes.TryGetValue((type.Name, LaneKey(type.Name, m, n, k).Rows, n, k), out int known) ? known
            : MeasureStoredLanes(widening, lane, x, y, m, n, k);
        RunLane(chosen, widening, lane, x, y, m, n, k);
        return true;
    }

    // The stored products' lanes: each output's row widened once (decoding-sized), a panel of rows widened and multiplied
    // transposed, the whole weight widened once and the float32 product (today's prompt path: ExpandedLane), the tiles; and
    // a 4-bit or 6-bit weight's two lanes over its stored codes (NibbleFormat, ScaledFormat, SixFormat), decoded in registers
    // or by int8 dot products; a bfloat16 weight's two lanes over its stored values, made float32 in registers in even and odd pairs or by
    // a shift (BFloat16Format); the packed bfloat16 product's two kernels reading its values in registers (the tiled and the
    // streaming, CpuBackend.Typed.cs).
    private const int RowLaneId = 0, PanelLaneId = 1, ExpandedLaneId = 2, TilesLaneId = 3, DecodedLaneId = 4, IntegerLaneId = 5,
        PairedLaneId = 6, ShiftedLaneId = 7, TiledLaneId = 8, StreamingLaneId = 9, MixedLaneId = 10, DescribedLaneId = 11;

    private static readonly string[] LaneNames = ["row lane", "panel lane", "today's path (widened once, float32 product)",
        "tiles (panels widened in the product)", "decoded lane (stored values decoded in registers inside the dot products)",
        "integer lane (x rounded to int8 a block of 32, int8 dot products over the stored codes)",
        "paired lane (bfloat16 made float32 in registers in even and odd pairs, by a shift and a mask, x reordered to match)",
        "shifted lane (bfloat16 made float32 in registers by a widening load and a shift)",
        "tiled lane (bfloat16 read in place in registers, a panel of rows' sums kept in registers)",
        "streaming lane (bfloat16 read in place in registers, each weight row added to every row's sums)",
        "mixed lane (x widened to float64 in registers as it is read, float64 FMAs over the stored values)",
        "described lane (any type read from its description alone, its values decoded in registers inside the dot products)"];

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
            int j0 = panel * nr;
            for (int c = 0; c < nr; c++)
            {
                var into = rows.AsSpan(c * k, k);
                TW.Widen(StoredRow(lane, j0 + c), 0, into, lane);
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
                    int j0 = t * rows, count = Math.Min(rows, n - j0);
                    for (int c = 0; c < count; c++)
                    {
                        var into = scratch.AsSpan(c * k, k);
                        TW.Widen(StoredRow(lane, j0 + c), 0, into, lane);
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
        float[] row = ArrayPool<float>.Shared.Rent(k);
        try
        {
            var values = row.AsSpan(0, k);
            for (int j = j0; j < j0 + count; j++)
            {
                TW.Widen(StoredRow(lane, j), 0, values, lane);
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
        var values = panel.AsSpan(at * k, k);
        TW.Widen(StoredRow(lane, row), 0, values, lane);
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

    // q4_k as a format read where it is stored (IRowFormat): a 4-bit super-block (NibbleBlocks) by its description; the
    // decoded lane timed against the integer lane (the decoded alone where the CPU has no int8 dot products); x rounded to
    // int8 for the integer lane (RoundRows); a group's scales decoded into a table for its kernels (NibbleKernels).
    private readonly unsafe struct NibbleFormat : IRowFormat<NibbleFormat, NibbleKernels>
    {
        private readonly NibbleBlocks blocks;

        private NibbleFormat(NibbleBlocks blocks) => this.blocks = blocks;

        public static bool Reads(ElementType type, int k, out NibbleFormat format)
        {
            format = type.Super is { } super && k % NibbleBlocks.Values == 0 && NibbleBlocks.Of(super) is { } plan ? new NibbleFormat(plan) : default;
            return format.blocks is not null;
        }

        public int First => DecodedLaneId;

        public int Second => IntegerLaneId;

        public bool BothRun => IntegerDots;

        public int Unmeasured => IntegerLaneId;

        public int GroupBytes => NibbleGroupBytes;

        // Columns over every row of x, a quarter of a thread's share: each stored row's scales decoded once a product.
        public (int Columns, int Rows) Tile(int total, int m, int threads, nint rowBytes) => ((total + 4 * threads - 1) / (4 * threads), m);

        public int Span(int k) => k;

        public LaneInput Prepare(int lane, float[] x, int m, int k)
        {
            if (lane != IntegerLaneId)
            {
                return new LaneInput(x, false, null, null);
            }

            var codes = ArrayPool<sbyte>.Shared.Rent(m * k);
            var scales = ArrayPool<float>.Shared.Rent(m * (k / NibbleBlocks.Values) * 16);
            RoundRows(x, codes, scales, m, k);
            return new LaneInput(x, false, codes, scales);
        }

        // A group's scales' table; a row read alone takes one only where its scales do not decode in vector lanes.
        public (int Floats, int Bytes) Scratch(int k, int group) =>
            (group > 0 || !blocks.InLanes ? Math.Max(group, 1) * (k / NibbleBlocks.Values) * 16 : 0, 0);

        public NibbleKernels Kernels(int lane, int k, LanePointers x, float* floats, byte* bytes) => new(lane == IntegerLaneId, blocks, k, x, floats);
    }

    // Stored bytes a group of q4_k rows takes with more than one row of x (RowGroups): the rows whose scales' table fills
    // 32 KB (16 floats a super-block of 144 bytes).
    private const int NibbleGroupBytes = 73728;

    // q4_k's lanes as RowGroups' kernels: a group's scales decoded once into `table` (NibbleBlocks.Scales, 16 floats a
    // super-block); a row read alone decodes each block's scales as its codes are read (NibbleBlocks.Block, no table) where
    // they decode in vector lanes, else into the table first.
    private readonly unsafe struct NibbleKernels : IRowKernels
    {
        private readonly bool integer;
        private readonly NibbleBlocks blocks;
        private readonly int k, superBlocks, rowTable, codeOffset, blockBytes;
        private readonly float* x, xs, table;
        private readonly sbyte* xq;

        public NibbleKernels(bool integer, NibbleBlocks blocks, int k, LanePointers input, float* table)
        {
            this.integer = integer;
            this.blocks = blocks;
            this.k = k;
            this.table = table;
            superBlocks = k / NibbleBlocks.Values;
            rowTable = superBlocks * 16;
            codeOffset = blocks.CodeOffset;
            blockBytes = blocks.Bytes;
            x = input.Values;
            xq = input.Codes;
            xs = input.Scales;
        }

        public void Group(byte* stored, nint rowBytes, int c0, int count)
        {
            for (int c = 0; c < count; c++)
            {
                blocks.Scales(new ReadOnlySpan<byte>(stored + (c0 + c) * rowBytes, (int)rowBytes), superBlocks, new Span<float>(table + c * rowTable, rowTable));
            }
        }

        // All of k a pass (Span: k).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Four(byte* row, int c, int i, int p0, int p1, out float y0, out float y1, out float y2, out float y3)
        {
            if (integer)
            {
                IntegerFour(ref *(row + codeOffset), blockBytes, superBlocks, ref *(table + c * rowTable), ref *(xq + (nint)i * k), ref *(xs + (nint)i * rowTable), k,
                    out y0, out y1, out y2, out y3);
            }
            else
            {
                DecodedFour(ref *(row + codeOffset), blockBytes, superBlocks, ref *(table + c * rowTable), ref *(x + (nint)i * k), k, out y0, out y1, out y2, out y3);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float One(byte* row, int c, int i) => integer
            ? IntegerOne(ref *row, codeOffset, blockBytes, superBlocks, ref *(table + c * rowTable), null, ref *(xq + (nint)i * k), ref *(xs + (nint)i * rowTable))
            : DecodedOne(ref *row, codeOffset, blockBytes, superBlocks, ref *(table + c * rowTable), null, ref *(x + (nint)i * k));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Single(byte* row, int i)
        {
            if (!blocks.InLanes)
            {
                blocks.Scales(new ReadOnlySpan<byte>(row, superBlocks * blockBytes), superBlocks, new Span<float>(table, rowTable));
                return One(row, 0, i);
            }

            return integer
                ? IntegerOne(ref *row, codeOffset, blockBytes, superBlocks, ref Unsafe.NullRef<float>(), blocks, ref *(xq + (nint)i * k), ref *(xs + (nint)i * rowTable))
                : DecodedOne(ref *row, codeOffset, blockBytes, superBlocks, ref Unsafe.NullRef<float>(), blocks, ref *(x + (nint)i * k));
        }
    }

    // Products of weights in one format read where it is stored (FormatOf: q4_k, q4_0, mxfp4, q6_k, bfloat16, int64, float64, int2) sharing x, under the
    // measured route (the queries, keys and values; the gate and up): in one pass, any rows; offered once such a weight has
    // been multiplied here, so the groups of other types' weights are not asked (their outputs allocated) only to run one
    // by one.
    public override bool StoredMatMulManyAvailable(int m) => ProductRoutes.Current == ProductRoute.Measured && s_formatsMultiplied;

    // Whether a weight in a format read where it is stored has been multiplied in this process.
    private static volatile bool s_formatsMultiplied;

    // 2 or 3 such products in one pass (FormatGrouped): each by its own lane for the shape (measured first where it has not
    // been), x prepared once for each lane they take, the weights' columns spread over the threads as one range, so each
    // output is its own product's, bit for bit. False for any other weights: the callers then run them one by one.
    public override bool StoredMatMulManyKernel(Storage x, int m, int k, ReadOnlySpan<(Storage Values, Storage? Scales, Storage Output, int Columns)> products)
    {
        var type = products.Length > 0 ? products[0].Values.ElementType : null;
        if (ProductRoutes.Current != ProductRoute.Measured || products.Length is < 2 or > 3 || m <= 0 || k <= 0 || type is null)
        {
            return false;
        }

        var group = new ThreeProducts();
        for (int j = 0; j < products.Length; j++)
        {
            var (values, scales, output, n) = products[j];
            if (values is not CpuStorage || !ReferenceEquals(values.ElementType, type) || n <= 0 || values.ElementCount < (long)n * k
                || scales is not null and not CpuStorage || scales is not null && scales.Length < n || output is not CpuStorage)
            {
                return false;
            }

            group[j] = products[j];
        }

        if (!CandidatesOf(type, k, new GroupedUse(type, D(x), m, k, group, products.Length)))
        {
            return false;
        }

        s_formatsMultiplied = true;
        return true;
    }

    // One product of a pass over stored rows: its weight, output, columns and lane.
    private readonly record struct RowPart(StoredLane Lane, float[] Y, int N, int LaneId);

    // Up to three parts, on the stack.
    [InlineArray(3)]
    private struct ThreeParts
    {
        private RowPart _part;
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

    // ------------------------------------------------------------------ 32-value blocks of 4-bit codes read where they are stored

    // A scaled block of 32 4-bit codes in ggml's split order with one float16 scale (ScaledNibbles: q4_0 by its description)
    // takes, under the measured route, two lanes neither of which writes the weight's values anywhere, whole or a panel: the
    // decoded lane decodes each block's codes in registers as its dot products read them ((code + bias) · scale: the values
    // ByteBlockWiden gives, bit for bit) and multiplies them with x in float32; the integer lane rounds x to int8 a block of
    // 32 values (one scale a block) and takes the CPU's int8 dot products of those with the stored codes, eight blocks'
    // code bytes transposed by words in registers so each block's sum lands in its own lane, the code bias added through
    // x's code sums and the float scales applied eight blocks at a time. The measured route keeps the faster at each shape.
    // A block whose codes are values of a 4-bit element type with a scale of an 8-bit one (OCP's MXFP4: fp4 E2M1 values
    // sharing an E8M0 scale, by its description) takes the same lanes: each code's value, a whole number in the codes' unit
    // (E2M1's: halves), read by a byte shuffle (the decoded lane's signed, the integer lane's less the least, which comes
    // back through x's code sums), its scale the scale type's value of its byte times the unit, gathered from a table.

    // q4_0 and MXFP4 as a format read where it is stored (IRowFormat): a 32-value block of 4-bit codes (ScaledNibbles) by its
    // description; the decoded lane timed against the integer lane; x rounded to int8 for the integer lane
    // (RoundScaledRows); a group's scales' table and its rows' short last groups as scratch for its kernels (ScaledKernels:
    // `bytes` holds the group of a row read alone, then a group a row of the group).
    private readonly unsafe struct ScaledFormat : IRowFormat<ScaledFormat, ScaledKernels>
    {
        private readonly ScaledNibbles plan;

        private ScaledFormat(ScaledNibbles plan) => this.plan = plan;

        public static bool Reads(ElementType type, int k, out ScaledFormat format)
        {
            format = type.Scaled is { } block && k % ScaledNibbles.Values == 0 && ScaledNibbles.Of(block) is { } found ? new ScaledFormat(found) : default;
            return format.plan is not null;
        }

        public int First => DecodedLaneId;

        public int Second => IntegerLaneId;

        public bool BothRun => true;

        public int Unmeasured => IntegerLaneId;

        public int GroupBytes => ScaledGroupBytes;

        public (int Columns, int Rows) Tile(int total, int m, int threads, nint rowBytes) => GroupTile(total, rowBytes, ScaledGroupBytes);

        public int Span(int k) => k;

        public LaneInput Prepare(int lane, float[] x, int m, int k)
        {
            if (lane != IntegerLaneId)
            {
                return new LaneInput(x, false, null, null);
            }

            int groups = ScaledNibbles.Groups(k);
            var codes = ArrayPool<sbyte>.Shared.Rent(m * groups * ScaledNibbles.GroupCodes);
            var scales = ArrayPool<float>.Shared.Rent(m * groups * 16);
            RoundScaledRows(x, codes, scales, m, k, plan.Bias);
            return new LaneInput(x, false, codes, scales);
        }

        // A group's scales' table (8 floats a group of 8 blocks) and, for the integer lane, its steps (64 floats' room a group
        // of 8 blocks: ScaledSteps), a row of the group after another; its rows' short last groups as bytes.
        public (int Floats, int Bytes) Scratch(int k, int group) =>
            (group * ScaledNibbles.Groups(k) * (ScaledNibbles.Group + ScaledStepFloats), (group + 1) * ScaledNibbles.Group * plan.Bytes);

        public ScaledKernels Kernels(int lane, int k, LanePointers x, float* floats, byte* bytes) =>
            new(lane == IntegerLaneId, plan, k, x.Values, x.Codes, x.Scales, floats, bytes + ScaledNibbles.Group * plan.Bytes, bytes);
    }

    // Stored bytes a group of q4_0 rows takes with more than one row of x (RowGroups).
    private const int ScaledGroupBytes = 16384;

    // Floats of scratch a group of 8 blocks' integer-lane steps take (ScaledSteps: 8 steps of 32 bytes).
    private const int ScaledStepFloats = 64;

    // The lanes over 32-value nibble blocks as RowGroups' kernels (the integer lane's or the decoded lane's). A group of
    // stored rows: their blocks' scales decoded once into `table` (8 floats a group of 8 blocks), for the integer lane each
    // row's codes made into its steps once after its scales (ScaledSteps: transposed by words, the codes split and, mapped,
    // looked up), which every four rows of x then read; and a row's last blocks short of a group of 8 copied as stored
    // (their bytes, not their values) into a group padded with zero blocks (`tails`, one a row of the group; `tail` for a
    // row read alone), so the kernels read whole groups.
    private readonly unsafe struct ScaledKernels : IRowKernels
    {
        private readonly bool integer;
        private readonly ScaledNibbles plan;
        private readonly int k, blocks, groups, full, groupBytes, rowTable, tailBytes;
        private readonly nint xStride, sStride;
        private readonly float* x, xs, table;
        private readonly sbyte* xq;
        private readonly byte* tails, tail;

        public ScaledKernels(bool integer, ScaledNibbles plan, int k, float* x, sbyte* xq, float* xs, float* table, byte* tails, byte* tail)
        {
            this.integer = integer;
            this.plan = plan;
            this.k = k;
            this.x = x;
            this.xq = xq;
            this.xs = xs;
            this.table = table;
            this.tails = tails;
            this.tail = tail;
            blocks = k / ScaledNibbles.Values;
            groups = ScaledNibbles.Groups(k);
            full = blocks / ScaledNibbles.Group;
            groupBytes = ScaledNibbles.Group * plan.Bytes;
            // A row's scales, then (the integer lane) its steps.
            rowTable = groups * (ScaledNibbles.Group + (integer ? ScaledStepFloats : 0));
            tailBytes = (blocks - full * ScaledNibbles.Group) * plan.Bytes;
            xStride = (nint)groups * ScaledNibbles.GroupCodes;
            sStride = (nint)groups * 16;
        }

        // Compiled optimized from its first call, so the shape's lane is measured on the code it then runs.
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Group(byte* stored, nint rowBytes, int c0, int count)
        {
            nint scaleOffset = plan.ScaleOffset, codeOffset = plan.CodeOffset;
            for (int c = 0; c < count; c++)
            {
                byte* row = stored + (c0 + c) * rowBytes, own = tails + c * groupBytes;
                if (tailBytes > 0)
                {
                    new ReadOnlySpan<byte>(row + full * groupBytes, tailBytes).CopyTo(new Span<byte>(own, tailBytes));
                }

                float* scales = table + c * rowTable;
                for (int g = 0; g < groups; g++)
                {
                    byte* group = g < full ? row + g * groupBytes : own;
                    plan.Scales(group + scaleOffset).Store(scales + g * ScaledNibbles.Group);
                    if (integer)
                    {
                        ScaledSteps(group + codeOffset, plan, (byte*)(scales + groups * ScaledNibbles.Group + g * ScaledStepFloats));
                    }
                }
            }
        }

        // All of k a pass (Span: k).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Four(byte* row, int c, int i, int p0, int p1, out float y0, out float y1, out float y2, out float y3)
        {
            if (integer)
            {
                float* scales = table + c * rowTable;
                ScaledStepsFour(scales + groups * ScaledNibbles.Group, scales, groups, xq + i * xStride, xs + i * sStride, xStride, sStride, out y0, out y1, out y2, out y3);
            }
            else
            {
                ScaledDecodedFour(row, tails + c * groupBytes, blocks, plan, table + c * rowTable, x + (nint)i * k, k, out y0, out y1, out y2, out y3);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float One(byte* row, int c, int i) => integer
            ? ScaledIntegerOne(row, tails + c * groupBytes, groups, full, plan, xq + i * xStride, xs + i * sStride)
            : ScaledDecodedOne(row, tails + c * groupBytes, blocks, plan, x + (nint)i * k);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Single(byte* row, int i)
        {
            if (tailBytes > 0)
            {
                new ReadOnlySpan<byte>(row + full * groupBytes, tailBytes).CopyTo(new Span<byte>(tail, tailBytes));
            }

            return integer
                ? ScaledIntegerOne(row, tail, groups, full, plan, xq + i * xStride, xs + i * sStride)
                : ScaledDecodedOne(row, tail, blocks, plan, x + (nint)i * k);
        }
    }

    // ------------------------------------------------------------------ weights read where they are stored: the shared part

    // The CPU multiplies weights of these formats where their rows are stored, writing nothing of a weight anywhere: q4_k's
    // super-blocks (NibbleFormat), q4_0's and MXFP4's 32-value blocks (ScaledFormat), q6_k's super-blocks (SixFormat), bfloat16
    // (BFloat16Format), int64's codes (LongFormat), float64 (DoubleFormat) and int2's 32-value blocks (TwoBitFormat). A format brings what
    // it alone knows (IRowFormat: its description, its two lanes, what each makes of x first, the rows a group takes, the
    // scratch and the kernels over a tile, IRowKernels); the rest is theirs together: the list of formats (FormatOf), the
    // routed product and its lane measured per shape (RowProduct, ILaneChoice, Chosen), the grouped pass (FormatGrouped),
    // the tiles over the threads (RowTiles) and the loop over a tile (RowGroups). The packed bfloat16 product (a tied head)
    // takes the same measured choice of lanes (ILaneChoice, CpuBackend.Typed.cs).

    /// <summary>
    /// A product with candidate lanes (up to three), timed on the real operands once per shape and the fastest kept for the
    /// shape (Chosen): a weight's product over its stored rows by the described lane and its written format's lanes
    /// (LaneCandidates), the packed bfloat16 product.
    /// </summary>
    private interface ILaneChoice
    {
        /// <summary>The name the shape's choice is kept under (with the rows' class, n and k).</summary>
        string Name { get; }

        /// <summary>The candidates that run on this CPU (1 to 3).</summary>
        int Count { get; }

        /// <summary>Candidate <paramref name="index"/>'s lane, the first the one the others are timed against.</summary>
        int Candidate(int index);

        /// <summary>The lane without measuring (CpuTuning.Autotune off).</summary>
        int Unmeasured { get; }

        /// <summary>
        /// Whether the lanes are timed briefly (TimeBrief: fixed rounds, medians, the second kept only when 3% faster) rather
        /// than until their times settle (Faster: the faster kept).
        /// </summary>
        bool Brief { get; }

        /// <summary>The product by <paramref name="lane"/>: all of y written.</summary>
        void Run(int lane);
    }

    /// <summary>
    /// A weight format the CPU multiplies where its rows (k values each, a stored row's bytes apart) are stored: read by its
    /// description alone; two lanes, the faster measured per shape; what each lane makes of x first; the stored rows a
    /// group of a tile takes, the values of k a pass takes and the scratch its kernels take; its kernels over a tile.
    /// </summary>
    private unsafe interface IRowFormat<TFormat, TKernels> where TFormat : struct, IRowFormat<TFormat, TKernels> where TKernels : struct, IRowKernels
    {
        /// <summary>The format of <paramref name="type"/>'s rows of <paramref name="k"/> values; false where its description is not one the format reads (or the CPU cannot run it).</summary>
        static abstract bool Reads(ElementType type, int k, out TFormat format);

        /// <summary>The lane timed first.</summary>
        int First { get; }

        /// <summary>The lane timed against it.</summary>
        int Second { get; }

        /// <summary>Whether the second lane runs on this CPU (else the first, unmeasured).</summary>
        bool BothRun { get; }

        /// <summary>The lane without measuring.</summary>
        int Unmeasured { get; }

        /// <summary>Stored bytes a group of a tile takes with more than one row of x (RowGroups).</summary>
        int GroupBytes { get; }

        /// <summary>
        /// The columns and the rows of x a tile takes with more than one row of x (RowTiles), of <paramref name="total"/>
        /// columns and <paramref name="m"/> rows over <paramref name="threads"/> threads, the stored rows
        /// <paramref name="rowBytes"/> apart.
        /// </summary>
        (int Columns, int Rows) Tile(int total, int m, int threads, nint rowBytes);

        /// <summary>Values of k four rows of x pass over a group at a time (k or more: all of k in one pass).</summary>
        int Span(int k);

        /// <summary>x as <paramref name="lane"/> reads it: x itself, or what the lane makes of it first (pooled; LaneInput.Release).</summary>
        LaneInput Prepare(int lane, float[] x, int m, int k);

        /// <summary>The floats and bytes of scratch a tile's kernels take, for a group of <paramref name="group"/> stored rows (0: rows read alone).</summary>
        (int Floats, int Bytes) Scratch(int k, int group);

        /// <summary>The kernels of <paramref name="lane"/> for a tile: x and the scratch (its bytes cleared) pinned.</summary>
        TKernels Kernels(int lane, int k, LanePointers x, float* floats, byte* bytes);
    }

    /// <summary>A lane's kernels over a weight's rows read where they are stored (RowGroups').</summary>
    private unsafe interface IRowKernels
    {
        /// <summary>Before stored rows [c0, c0 + count) are multiplied by more than one row of x.</summary>
        void Group(byte* stored, nint rowBytes, int c0, int count);

        /// <summary>Rows i..i + 3 of x by the group's stored row c (at <paramref name="row"/>), over values [p0, p1) of k.</summary>
        void Four(byte* row, int c, int i, int p0, int p1, out float y0, out float y1, out float y2, out float y3);

        /// <summary>Row i of x by the group's stored row c.</summary>
        float One(byte* row, int c, int i);

        /// <summary>Row i of x by a stored row read alone (one row of x: each stored row read once, no group).</summary>
        float Single(byte* row, int i);
    }

    /// <summary>What to do with the format of a weight once FormatOf finds it, generic over the format (its code compiled for it).</summary>
    private interface IFormatUse
    {
        void Use<TF, TK>(TF format) where TF : struct, IRowFormat<TF, TK> where TK : struct, IRowKernels;
    }

    /// <summary>x as a lane reads it: x itself, or what the lane made of it first (int8 codes and their scales; values reordered).</summary>
    private readonly record struct LaneInput(float[] Values, bool Owned, sbyte[]? Codes, float[]? Scales)
    {
        /// <summary>The pooled buffers back to the pool.</summary>
        public void Release()
        {
            if (Owned)
            {
                ArrayPool<float>.Shared.Return(Values);
            }

            if (Codes is not null)
            {
                ArrayPool<sbyte>.Shared.Return(Codes);
            }

            if (Scales is not null)
            {
                ArrayPool<float>.Shared.Return(Scales);
            }
        }
    }

    /// <summary>A lane's x, pinned for a tile (LaneInput's arrays).</summary>
    private readonly unsafe struct LanePointers
    {
        public LanePointers(float* values, sbyte* codes, float* scales)
        {
            Values = values;
            Codes = codes;
            Scales = scales;
        }

        public float* Values { get; }

        public sbyte* Codes { get; }

        public float* Scales { get; }
    }

    // The formats read where they are stored, in one place: the one that reads `type`'s rows of k values, given to `use`;
    // false for any other type.
    private static bool FormatOf<TU>(ElementType type, int k, ref TU use) where TU : struct, IFormatUse
    {
        if (NibbleFormat.Reads(type, k, out var q4k))
        {
            use.Use<NibbleFormat, NibbleKernels>(q4k);
            return true;
        }

        if (ScaledFormat.Reads(type, k, out var q40))
        {
            use.Use<ScaledFormat, ScaledKernels>(q40);
            return true;
        }

        if (SixFormat.Reads(type, k, out var q6k))
        {
            use.Use<SixFormat, SixKernels>(q6k);
            return true;
        }

        if (BFloat16Format.Reads(type, k, out var bf16))
        {
            use.Use<BFloat16Format, BFloat16Kernels>(bf16);
            return true;
        }

        if (LongFormat.Reads(type, k, out var int64))
        {
            use.Use<LongFormat, LongKernels>(int64);
            return true;
        }

        if (DoubleFormat.Reads(type, k, out var float64))
        {
            use.Use<DoubleFormat, DoubleKernels>(float64);
            return true;
        }

        if (TwoBitFormat.Reads(type, k, out var int2))
        {
            use.Use<TwoBitFormat, TwoBitKernels>(int2);
            return true;
        }

        return false;
    }

    // The shape's lane of `product`: kept from its measuring, or measured now (Measure).
    private static int Chosen<TC>(in TC product, int m, int n, int k) where TC : struct, ILaneChoice =>
        StoredLanes.TryGetValue((product.Name, LaneKey(product.Name, m, n, k).Rows, n, k), out int known) ? known : Measure(product, m, n, k);

    // The fastest of the product's candidates, timed once on the real operands, each against the fastest so far (each run
    // writes all of y, which the caller writes again: Faster until the times settle, or TimeBrief's fixed rounds with a
    // challenger kept only when 3% faster) and kept for the shape; a lone candidate unmeasured (recorded as the only one);
    // without measuring (CpuTuning.Autotune off), the product's unmeasured lane.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static int Measure<TC>(TC product, int m, int n, int k) where TC : struct, ILaneChoice
    {
        int count = Math.Clamp(product.Count, 1, 3), best = count > 1 ? product.Unmeasured : product.Candidate(0);
        if (count == 1)
        {
            ProductRoutes.Chose("cpu", string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{product.Name} {m}x{k}->{n}"), $"{LaneNames[best]} (the only candidate)");
        }
        else if (CpuTuning.Autotune)
        {
            var times = new double[count];
            int at = 0;
            best = product.Candidate(0);
            for (int c = 1; c < count; c++)
            {
                int holder = best, challenger = product.Candidate(c);
                var (a, b) = product.Brief
                    ? TimeBrief(() => product.Run(holder), () => product.Run(challenger))
                    : Faster.Time(() => product.Run(holder), () => product.Run(challenger));
                times[at] = a;
                times[c] = b;
                if (b < (product.Brief ? 0.97 * a : a))
                {
                    (best, at) = (challenger, c);
                }
            }

            ProductRoutes.Chose("cpu", string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{product.Name} {m}x{k}->{n}"),
                string.Create(System.Globalization.CultureInfo.InvariantCulture,
                    $"{LaneNames[best]} ({string.Join(", ", Enumerable.Range(0, count).Select(i => $"{LaneNames[product.Candidate(i)].Split(' ')[0]} {times[i]:0.###} ms"))})"));
        }

        StoredLanes[(product.Name, LaneKey(product.Name, m, n, k).Rows, n, k)] = best;
        return best;
    }

    // A weight's candidates as a choice of lanes (Chosen): the described lane where it reads the weight's type and the
    // setting takes it (first: the others are timed against it), then the written format's lanes where one reads the type
    // and the setting takes them; its product by any of them.
    private readonly struct LaneCandidates<TF, TK>(TF written, bool hasWritten, DescribedFormat described, bool hasDescribed,
        StoredLane lane, float[] x, float[] y, int m, int n, int k) : ILaneChoice
        where TF : struct, IRowFormat<TF, TK> where TK : struct, IRowKernels
    {
        public string Name => lane.Type.Name + ProductRoutes.Lanes switch
        {
            RouteLanes.Described => " (described lane)",
            RouteLanes.Written => " (written kernels)",
            _ => "",
        };

        public int Count => (hasDescribed ? 1 : 0) + (hasWritten ? written.BothRun ? 2 : 1 : 0);

        public int Candidate(int index)
        {
            if (hasDescribed)
            {
                if (index == 0)
                {
                    return DescribedLaneId;
                }

                index--;
            }

            return index == 0 ? written.First : written.Second;
        }

        public int Unmeasured => hasDescribed ? DescribedLaneId : written.BothRun ? written.Unmeasured : written.First;

        public bool Brief => false;

        public void Run(int laneId)
        {
            if (laneId == DescribedLaneId)
            {
                FormatProduct<DescribedFormat, DescribedKernels>(described, laneId, lane, x, y, m, n, k);
                return;
            }

            FormatProduct<TF, TK>(written, laneId, lane, x, y, m, n, k);
        }
    }

    /// <summary>What to do with a weight's candidates once CandidatesOf finds them, generic over the written format (its code compiled for it).</summary>
    private interface ICandidateUse
    {
        void Use<TF, TK>(TF written, bool hasWritten, DescribedFormat described, bool hasDescribed) where TF : struct, IRowFormat<TF, TK> where TK : struct, IRowKernels;
    }

    // A written format found by FormatOf, handed on with the described lane's candidacy.
    private readonly struct WithDescribed<TU>(TU use, DescribedFormat described, bool hasDescribed) : IFormatUse where TU : struct, ICandidateUse
    {
        public void Use<TF, TK>(TF format) where TF : struct, IRowFormat<TF, TK> where TK : struct, IRowKernels =>
            use.Use<TF, TK>(format, true, described, hasDescribed);
    }

    // Under the measured route, the candidates of `type`'s rows of k values given to `use` (ProductRoutes.Lanes): the
    // described lane where it reads the type (but with "written"), a written format where one reads the type (but with
    // "described" where the described lane reads it); false where neither does.
    private static bool CandidatesOf<TU>(ElementType type, int k, TU use) where TU : struct, ICandidateUse
    {
        var lanes = ProductRoutes.Lanes;
        bool hasDescribed = DescribedFormat.Reads(type, k, out var described) && lanes != RouteLanes.Written;
        if (!(hasDescribed && lanes == RouteLanes.Described))
        {
            var written = new WithDescribed<TU>(use, described, hasDescribed);
            if (FormatOf(type, k, ref written))
            {
                return true;
            }
        }

        if (!hasDescribed)
        {
            return false;
        }

        use.Use<DescribedFormat, DescribedKernels>(described, false, described, true);
        return true;
    }

    // The routed product of a weight (StoredRouted): the shape's lane among its candidates (Chosen), then the product by it.
    private readonly struct RoutedUse(StoredLane lane, float[] x, float[] y, int m, int n, int k) : ICandidateUse
    {
        public void Use<TF, TK>(TF written, bool hasWritten, DescribedFormat described, bool hasDescribed) where TF : struct, IRowFormat<TF, TK> where TK : struct, IRowKernels
        {
            var candidates = new LaneCandidates<TF, TK>(written, hasWritten, described, hasDescribed, lane, x, y, m, n, k);
            candidates.Run(Chosen(candidates, m, n, k));
        }
    }

    // Up to three products of a grouped pass, on the stack.
    [InlineArray(3)]
    private struct ThreeProducts
    {
        private (Storage Values, Storage? Scales, Storage Output, int Columns) _product;
    }

    // The grouped pass of weights of one type (FormatGrouped).
    private readonly struct GroupedUse(ElementType type, float[] x, int m, int k, ThreeProducts products, int count) : ICandidateUse
    {
        public void Use<TF, TK>(TF written, bool hasWritten, DescribedFormat described, bool hasDescribed) where TF : struct, IRowFormat<TF, TK> where TK : struct, IRowKernels =>
            FormatGrouped<TF, TK>(written, hasWritten, described, hasDescribed, type, x, m, k, products, count);
    }

    // One lane of a weight in its format, apart from the measured choice (StoredRowLane, for tests).
    private struct LaneUse(bool second, StoredLane lane, float[] x, float[] y, int m, int n, int k) : IFormatUse
    {
        public bool Ran;

        public void Use<TF, TK>(TF format) where TF : struct, IRowFormat<TF, TK> where TK : struct, IRowKernels
        {
            if (second && !format.BothRun)
            {
                return;
            }

            FormatProduct<TF, TK>(format, second ? format.Second : format.First, lane, x, y, m, n, k);
            Ran = true;
        }
    }

    /// <summary>
    /// y [m, n] = x [m, k] · Wᵀ for W a weight in a format read where it is stored (q4_k, q4_0, mxfp4, q6_k, bfloat16, int64, float64, int2:
    /// FormatOf) of n rows stored as <paramref name="words"/> (its bytes, <paramref name="length"/> words), each output times
    /// its row's scale where <paramref name="scales"/> are given, by the format's first lane or, with
    /// <paramref name="second"/>, its second (q4_k, q4_0, mxfp4, q6_k, int64 and int2: decoded, integer; bfloat16: shifted, paired; float64: decoded, mixed). False
    /// where no format reads the type, or the second lane does not run on this CPU. For tests: each lane apart from the
    /// measured choice.
    /// </summary>
    internal static bool StoredRowLane(bool second, ElementType type, float[] words, int length, float[] x, float[] y, int m, int n, int k, float[]? scales = null)
    {
        var use = new LaneUse(second, new StoredLane(type, type.Scaled, words, length, scales, k, type.ByteCount(k)), x, y, m, n, k);
        return FormatOf(type, k, ref use) && use.Ran;
    }

    /// <summary>
    /// y [m, n] = x [m, k] · Wᵀ by the described lane alone (any type read from its description, ElementType.Description;
    /// none of the written formats' kernels), W's n rows stored as <paramref name="words"/>, each output times its row's
    /// scale where <paramref name="scales"/> are given. False where the described lane does not read the type here. For
    /// tests, apart from the measured choice.
    /// </summary>
    internal static bool DescribedRowLane(ElementType type, float[] words, int length, float[] x, float[] y, int m, int n, int k, float[]? scales = null)
    {
        if (!DescribedFormat.Reads(type, k, out var described))
        {
            return false;
        }

        FormatProduct<DescribedFormat, DescribedKernels>(described, DescribedLaneId, new StoredLane(type, type.Scaled, words, length, scales, k, type.ByteCount(k)), x, y, m, n, k);
        return true;
    }

    /// <summary>The arithmetic the described lane takes for <paramref name="type"/>'s rows of <paramref name="k"/> values ("integer", "float32", "float64"), or null where it does not read the type here. For tests.</summary>
    internal static string? DescribedArithmetic(ElementType type, int k) => DescribedFormat.Reads(type, k, out var described) ? described.Arithmetic : null;

    // The product by one lane (FormatMany with one part).
    private static void FormatProduct<TF, TK>(TF format, int laneId, StoredLane lane, float[] x, float[] y, int m, int n, int k)
        where TF : struct, IRowFormat<TF, TK> where TK : struct, IRowKernels
    {
        var parts = new ThreeParts();
        parts[0] = new RowPart(lane, y, n, laneId);
        FormatMany<TF, TK>(format, ((ReadOnlySpan<RowPart>)parts)[..1], x, m, k);
    }

    // 2 or 3 weights of one type sharing x: each by its own lane for the shape among its candidates (measured here first
    // where it has not been), then one pass for those the described lane takes and one for those a written format's lanes
    // take (FormatMany), so each output is its own product's, bit for bit.
    private static void FormatGrouped<TF, TK>(TF written, bool hasWritten, DescribedFormat described, bool hasDescribed, ElementType type, float[] x, int m, int k,
        ThreeProducts products, int count)
        where TF : struct, IRowFormat<TF, TK> where TK : struct, IRowKernels
    {
        ThreeParts byDescribed = new(), byWritten = new();
        int a = 0, b = 0;
        for (int j = 0; j < count; j++)
        {
            var (values, scales, output, n) = products[j];
            var lane = new StoredLane(type, type.Scaled, D(values), values.Length, scales is null ? null : D(scales), k, type.ByteCount(k));
            float[] y = D(output);
            int chosen = Chosen(new LaneCandidates<TF, TK>(written, hasWritten, described, hasDescribed, lane, x, y, m, n, k), m, n, k);
            if (chosen == DescribedLaneId)
            {
                byDescribed[a++] = new RowPart(lane, y, n, chosen);
            }
            else
            {
                byWritten[b++] = new RowPart(lane, y, n, chosen);
            }
        }

        if (a > 0)
        {
            FormatMany<DescribedFormat, DescribedKernels>(described, ((ReadOnlySpan<RowPart>)byDescribed)[..a], x, m, k);
        }

        if (b > 0)
        {
            FormatMany<TF, TK>(written, ((ReadOnlySpan<RowPart>)byWritten)[..b], x, m, k);
        }
    }

    // The products of `parts` (1 to 3 weights of one format sharing x): x prepared once for each lane they take, then their
    // tiles over the threads (RowTiles, FormatColumns).
    private static void FormatMany<TF, TK>(TF format, ReadOnlySpan<RowPart> parts, float[] x, int m, int k)
        where TF : struct, IRowFormat<TF, TK> where TK : struct, IRowKernels
    {
        bool first = false, second = false;
        foreach (var part in parts)
        {
            first |= part.LaneId == format.First;
            second |= part.LaneId != format.First;
        }

        LaneInput a = default, b = default;
        try
        {
            if (first)
            {
                a = format.Prepare(format.First, x, m, k);
            }

            if (second)
            {
                b = format.Prepare(format.Second, x, m, k);
            }

            RowTiles<TF, TK>(format, parts, a, b, m, k);
        }
        finally
        {
            a.Release();
            b.Release();
        }
    }

    // Stored bytes a tile takes with one row of x: columns of about this many bytes (small enough that the last tiles keep
    // every thread busy to the end). With more rows a tile is a group of stored rows of a format's own bytes
    // (IRowFormat.GroupBytes: the rows a tile's loop keeps in the cache while it passes the rows of x over them).
    private const int RowTileBytes = 16384;

    // Rows of x a tile of a group of stored rows takes (GroupTile: the tiles split x's rows as well as the columns).
    private const int RowTileRows = 32;

    // A tile of a group of stored rows (`groupBytes`) by RowTileRows rows of x: q4_0's and bfloat16's with more than one row.
    private static (int Columns, int Rows) GroupTile(int total, nint rowBytes, int groupBytes) => (RowGroup(rowBytes, total, groupBytes), RowTileRows);

    // The most stored rows a group of a tile's loop takes (its four rows' partial sums on the stack).
    private const int MostGroupRows = 64;

    // The stored rows a group of a tile's loop takes: those within `groupBytes`, at least one, at most MostGroupRows.
    private static int RowGroup(nint rowBytes, int columns, int groupBytes) =>
        (int)Math.Clamp(groupBytes / Math.Max(1, rowBytes), 1, Math.Clamp(columns, 1, MostGroupRows));

    // The tiles of `parts` in one range over the threads (CpuRanges, a tile a chunk): columns and rows of x, each part's
    // tiles after the other's; each output its own product's, whichever thread runs it. One row of x: columns in whole cache
    // lines of y (16 floats), so tiles on other threads do not write into the same lines as each output is written. More
    // rows: the format's tile (IRowFormat.Tile).
    private static void RowTiles<TF, TK>(TF format, ReadOnlySpan<RowPart> parts, LaneInput first, LaneInput second, int m, int k)
        where TF : struct, IRowFormat<TF, TK> where TK : struct, IRowKernels
    {
        int total = 0;
        foreach (var part in parts)
        {
            total += part.N;
        }

        int threads = Math.Max(1, ComputeResources.MaxCpuThreads);
        long rowBytes = Math.Max(1, parts[0].Lane.RowBytes);
        var (wanted, rows) = m == 1
            ? ((Math.Max(RowTileBytes / rowBytes, (total + 64L * threads - 1) / (64L * threads)) + 15) / 16 * 16, 1)
            : format.Tile(total, m, threads, (nint)rowBytes);
        int columns = (int)Math.Clamp(wanted, 1, total);
        rows = Math.Clamp(rows, 1, m);
        int rowBlocks = (m + rows - 1) / rows;
        int t1 = (parts[0].N + columns - 1) / columns * rowBlocks;
        int t2 = parts.Length > 1 ? t1 + (parts[1].N + columns - 1) / columns * rowBlocks : t1;
        int tiles = parts.Length > 2 ? t2 + (parts[2].N + columns - 1) / columns * rowBlocks : t2;
        var body = new RowTileRange<TF, TK>(format, first, second, m, k, columns, rows, rowBlocks, t1, t2,
            parts[0], parts.Length > 1 ? parts[1] : default, parts.Length > 2 ? parts[2] : default);
        CpuRanges.Run(tiles, 1, tiles, threads, ref body);                             // `body` is this frame's, on the stack
    }

    // RowTiles' range: tile t of part p (its tiles from t_p) is its column block t / rowBlocks and its block of rows of x
    // t mod rowBlocks, by the part's lane and that lane's x.
    private readonly struct RowTileRange<TF, TK>(TF format, LaneInput first, LaneInput second, int m, int k, int columns, int rows, int rowBlocks, int t1, int t2,
        RowPart p0, RowPart p1, RowPart p2) : IRangeBody
        where TF : struct, IRowFormat<TF, TK> where TK : struct, IRowKernels
    {
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Run(int from, int to)
        {
            for (int t = from; t < to; t++)
            {
                var (part, local) = t < t1 ? (p0, t) : t < t2 ? (p1, t - t1) : (p2, t - t2);
                int j0 = local / rowBlocks * columns, i0 = local % rowBlocks * rows;
                FormatColumns<TF, TK>(format, part, part.LaneId == format.First ? first : second, k, j0, Math.Min(part.N, j0 + columns), i0, Math.Min(m, i0 + rows));
            }
        }
    }

    // Columns [j0, j1) of a part's y for rows [i0, i1) of x by the part's lane (RowGroups with the format's kernels): the
    // scratch the kernels take (its bytes cleared), pinned with the operands for the tile.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void FormatColumns<TF, TK>(TF format, in RowPart part, LaneInput input, int k, int j0, int j1, int i0, int i1)
        where TF : struct, IRowFormat<TF, TK> where TK : struct, IRowKernels
    {
        var lane = part.Lane;
        nint rowBytes = (nint)lane.RowBytes;
        var (floats, bytes) = format.Scratch(k, i1 - i0 > 1 ? RowGroup(rowBytes, j1 - j0, format.GroupBytes) : 0);
        float[]? floatArray = floats > 0 ? ArrayPool<float>.Shared.Rent(floats) : null;
        byte[]? byteArray = bytes > 0 ? ArrayPool<byte>.Shared.Rent(bytes) : null;
        try
        {
            fixed (float* words = lane.Words)
            fixed (float* xv = input.Values)
            fixed (sbyte* xc = input.Codes)
            fixed (float* xs = input.Scales)
            fixed (float* scratch = floatArray)
            fixed (byte* scratchBytes = byteArray)
            {
                if (scratchBytes is not null)
                {
                    new Span<byte>(scratchBytes, bytes).Clear();
                }

                var kernels = format.Kernels(part.LaneId, k, new LanePointers(xv, xc, xs), scratch, scratchBytes);
                RowGroups(ref kernels, (byte*)words, rowBytes, k, format.Span(k), format.GroupBytes, lane.RowScales, part.Y, part.N, j0, j1, i0, i1);
            }
        }
        finally
        {
            if (floatArray is not null)
            {
                ArrayPool<float>.Shared.Return(floatArray);
            }

            if (byteArray is not null)
            {
                ArrayPool<byte>.Shared.Return(byteArray);
            }
        }
    }

    // Columns [j0, j1) of y (n a row) for rows [i0, i1) of x (k a row) by `kernels` over the stored rows at `stored`. One row
    // of x: each stored row read once, in order. More: a group of stored rows (RowGroup, `groupBytes`) at a time, set up once
    // (IRowKernels.Group), then four rows of x at a time against each of the group's rows, the rows left over one at a time.
    // The four rows of x pass over the group's rows `span` values of k at a time (all of k where span ≥ k), so that much of
    // them stays in the cache across the group, and the group's stored rows across the rows of x; a span's partial sums are
    // added in order. Each output times its row's scale where the weight has them. Compiled optimized from its first call,
    // as FormatColumns and RowTileRange.Run are: a format's kernels inline into it only when it is optimized, and the
    // first passes of a run would otherwise call them through unoptimized code.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void RowGroups<TK>(ref TK kernels, byte* stored, nint rowBytes, int k, int span, int groupBytes, float[]? scales, float[] y, int n,
        int j0, int j1, int i0, int i1)
        where TK : struct, IRowKernels
    {
        if (i1 - i0 == 1)
        {
            for (int c = j0; c < j1; c++)
            {
                float sum = kernels.Single(stored + c * rowBytes, i0);
                y[i0 * n + c] = scales is null ? sum : sum * scales[c];
            }

            return;
        }

        int group = RowGroup(rowBytes, j1 - j0, groupBytes);
        float* sums = stackalloc float[4 * MostGroupRows];
        for (int c0 = j0; c0 < j1; c0 += group)
        {
            int count = Math.Min(group, j1 - c0);
            kernels.Group(stored, rowBytes, c0, count);
            int i = i0;
            for (; i + 4 <= i1; i += 4)
            {
                if (span >= k)
                {
                    // One pass over k: each output written as its four rows are done.
                    for (int c = 0; c < count; c++)
                    {
                        kernels.Four(stored + (c0 + c) * rowBytes, c, i, 0, k, out float s0, out float s1, out float s2, out float s3);
                        float scale = scales is null ? 1f : scales[c0 + c];
                        int at = i * n + c0 + c;
                        y[at] = scales is null ? s0 : s0 * scale;
                        y[at + n] = scales is null ? s1 : s1 * scale;
                        y[at + 2 * n] = scales is null ? s2 : s2 * scale;
                        y[at + 3 * n] = scales is null ? s3 : s3 * scale;
                    }

                    continue;
                }

                // Passes over k a span at a time: each span's partial sums added in order, the outputs written at the end.
                new Span<float>(sums, 4 * count).Clear();
                for (int p0 = 0; p0 < k; p0 += span)
                {
                    int p1 = Math.Min(k, p0 + span);
                    for (int c = 0; c < count; c++)
                    {
                        kernels.Four(stored + (c0 + c) * rowBytes, c, i, p0, p1, out float s0, out float s1, out float s2, out float s3);
                        sums[4 * c] += s0;
                        sums[4 * c + 1] += s1;
                        sums[4 * c + 2] += s2;
                        sums[4 * c + 3] += s3;
                    }
                }

                for (int c = 0; c < count; c++)
                {
                    float scale = scales is null ? 1f : scales[c0 + c];
                    int at = i * n + c0 + c;
                    y[at] = scales is null ? sums[4 * c] : sums[4 * c] * scale;
                    y[at + n] = scales is null ? sums[4 * c + 1] : sums[4 * c + 1] * scale;
                    y[at + 2 * n] = scales is null ? sums[4 * c + 2] : sums[4 * c + 2] * scale;
                    y[at + 3 * n] = scales is null ? sums[4 * c + 3] : sums[4 * c + 3] * scale;
                }
            }

            for (; i < i1; i++)
            {
                for (int c = 0; c < count; c++)
                {
                    float sum = kernels.One(stored + (c0 + c) * rowBytes, c, i);
                    y[i * n + c0 + c] = scales is null ? sum : sum * scales[c0 + c];
                }
            }
        }
    }

    // x rounded to int8 for the integer lane over 32-value blocks, row by row (RoundScaledRow), over the threads where x is large.
    private static void RoundScaledRows(float[] x, sbyte[] codes, float[] scales, int m, int k, int bias)
    {
        int groups = ScaledNibbles.Groups(k), rowCodes = groups * ScaledNibbles.GroupCodes, rowScales = groups * 16;
        if (m > 1 && (long)m * k >= 1 << 16)
        {
            Parallel.For(0, m, ComputeResources.ParallelOptions, i => RoundScaledRow(x.AsSpan(i * k, k), codes.AsSpan(i * rowCodes, rowCodes), scales.AsSpan(i * rowScales, rowScales), bias));
            return;
        }

        for (int i = 0; i < m; i++)
        {
            RoundScaledRow(x.AsSpan(i * k, k), codes.AsSpan(i * rowCodes, rowCodes), scales.AsSpan(i * rowScales, rowScales), bias);
        }
    }

    // A row of x rounded to int8 a block of 32 values at a time: the block's scale amax / 127 (0 where every value is 0), its
    // codes value · 127 / amax rounded to even; laid out a group of 8 blocks at a time in the integer kernels' step order
    // (step s, lane l: block l's codes of values 4s..4s+3, 32 bytes a step); per group 16 floats: the blocks' scales at 0-7
    // and, as int32 bits at 8-15, the code bias times each block's code sum. A last group short of 8 blocks is padded with
    // zero codes, scales and sums. The integer lane's product is then the product of x so rounded.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void RoundScaledRow(ReadOnlySpan<float> x, Span<sbyte> codes, Span<float> scales, int bias)
    {
        Span<int> natural = stackalloc int[64];
        ref float row = ref MemoryMarshal.GetReference(x);
        int blocks = x.Length / ScaledNibbles.Values;
        for (int g = 0; g * ScaledNibbles.Group < blocks; g++)
        {
            int count = Math.Min(ScaledNibbles.Group, blocks - g * ScaledNibbles.Group);
            var into = scales.Slice(g * 16, 16);
            var sums = MemoryMarshal.Cast<float, int>(into[8..]);
            if (count < ScaledNibbles.Group)
            {
                natural.Clear();
                into.Clear();
            }

            for (int l = 0; l < count; l++)
            {
                ref float v = ref Unsafe.Add(ref row, (g * ScaledNibbles.Group + l) * ScaledNibbles.Values);
                var v0 = Vector256.LoadUnsafe(ref v);
                var v1 = Vector256.LoadUnsafe(ref v, 8);
                var v2 = Vector256.LoadUnsafe(ref v, 16);
                var v3 = Vector256.LoadUnsafe(ref v, 24);
                var most = Vector256.Max(Vector256.Max(Vector256.Abs(v0), Vector256.Abs(v1)), Vector256.Max(Vector256.Abs(v2), Vector256.Abs(v3)));
                var half = Vector128.Max(most.GetLower(), most.GetUpper());
                half = Vector128.Max(half, Vector128.Shuffle(half, Vector128.Create(2, 3, 0, 1)));
                half = Vector128.Max(half, Vector128.Shuffle(half, Vector128.Create(1, 0, 3, 2)));
                float amax = half.ToScalar();
                var inverse = Vector256.Create(amax > 0f ? 127f / amax : 0f);
                var q0 = Vector256.ConvertToInt32(Vector256.Round(v0 * inverse));
                var q1 = Vector256.ConvertToInt32(Vector256.Round(v1 * inverse));
                var q2 = Vector256.ConvertToInt32(Vector256.Round(v2 * inverse));
                var q3 = Vector256.ConvertToInt32(Vector256.Round(v3 * inverse));
                Vector256.Narrow(Vector256.Narrow(q0, q1), Vector256.Narrow(q2, q3)).AsInt32().CopyTo(natural.Slice(l * 8, 8));
                into[l] = amax / 127f;
                sums[l] = bias * Vector256.Sum((q0 + q1) + (q2 + q3));
            }

            var words = MemoryMarshal.Cast<sbyte, int>(codes.Slice(g * ScaledNibbles.GroupCodes, ScaledNibbles.GroupCodes));
            for (int s = 0; s < 8; s++)
            {
                for (int l = 0; l < ScaledNibbles.Group; l++)
                {
                    words[s * 8 + l] = natural[l * 8 + s];
                }
            }
        }
    }

    // Words 0-3 of the code bytes of 8 blocks (block l's at `codes` + l · `bytes`) transposed: t_w's lane l is word w of block
    // l's codes (its bytes 4w..4w + 3: values 4w..4w + 3 in their low halves, 16 + 4w.. in their high halves).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void BlockWords(byte* codes, nint bytes, out Vector256<uint> t0, out Vector256<uint> t1, out Vector256<uint> t2, out Vector256<uint> t3)
    {
        var r0 = Vector256.Create(Vector128.Load(codes), Vector128.Load(codes + 4 * bytes)).AsUInt32();
        var r1 = Vector256.Create(Vector128.Load(codes + bytes), Vector128.Load(codes + 5 * bytes)).AsUInt32();
        var r2 = Vector256.Create(Vector128.Load(codes + 2 * bytes), Vector128.Load(codes + 6 * bytes)).AsUInt32();
        var r3 = Vector256.Create(Vector128.Load(codes + 3 * bytes), Vector128.Load(codes + 7 * bytes)).AsUInt32();
        var a = Avx2.UnpackLow(r0, r1).AsUInt64();
        var b = Avx2.UnpackHigh(r0, r1).AsUInt64();
        var c = Avx2.UnpackLow(r2, r3).AsUInt64();
        var d = Avx2.UnpackHigh(r2, r3).AsUInt64();
        t0 = Avx2.UnpackLow(a, c).AsUInt32();
        t1 = Avx2.UnpackHigh(a, c).AsUInt32();
        t2 = Avx2.UnpackLow(b, d).AsUInt32();
        t3 = Avx2.UnpackHigh(b, d).AsUInt32();
    }

    // The float16 scales of 8 blocks (block l's at `at` + offsets[l]) as float32, exactly (subnormals, infinities and NaNs too).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<float> HalfScales(byte* at, Vector256<int> offsets) => HalfLanes(Avx2.GatherVector256((int*)at, offsets, 1));

    // The float16 in each lane's low 16 bits as float32, exactly (subnormals, infinities and NaNs too).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> HalfLanes(Vector256<int> bits)
    {
        var magnitude = bits & Vector256.Create(0x7FFF);
        var value = (Vector256.ShiftLeft(magnitude, 13).AsSingle() * Vector256.Create(0x77800000).AsSingle()).AsInt32();   // · 2^112
        var special = Vector256.GreaterThan(magnitude, Vector256.Create(0x7BFF)) & Vector256.Create(0x7F800000);
        return (value | special | Vector256.ShiftLeft(bits & Vector256.Create(0x8000), 16)).AsSingle();
    }

    // sum + the dot products of 4 codes (0-15) by 4 of x's int8 codes (32 bytes at `x`), in each 32-bit lane.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<int> DotAt(Vector256<int> sum, Vector256<byte> codes, sbyte* x)
    {
        var values = Vector256.Load(x);
        return AvxVnni.IsSupported
            ? AvxVnni.MultiplyWideningAndAdd(sum, codes, values)
            : sum + Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(codes, values), Vector256.Create((short)1));
    }

    // A group of 8 blocks' codes (block l's at `codes` + l · the block's bytes) made into the integer lane's 8 steps of 32
    // bytes at `into`: transposed by words (BlockWords), each word's low halves then high halves (step s: lane l holds four
    // codes of block l), a mapped plan's codes looked up as their values less the least (MXFP4: E2M1's in halves, plus 12).
    // Compiled optimized from its first call: its caller (ScaledKernels.Group) runs unoptimized code for its first calls,
    // where nothing inlines into it (measured: the integer lane 9 times slower while it was, and the decoded lane kept).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void ScaledSteps(byte* codes, ScaledNibbles plan, byte* into)
    {
        var low = Vector256.Create((byte)15);
        var unsigned = plan.UnsignedCodes;
        bool mapped = plan.Mapped;
        BlockWords(codes, plan.Bytes, out var t0, out var t1, out var t2, out var t3);
        Nibbles(t0.AsByte() & low, mapped, unsigned).Store(into);
        Nibbles(t1.AsByte() & low, mapped, unsigned).Store(into + 32);
        Nibbles(t2.AsByte() & low, mapped, unsigned).Store(into + 64);
        Nibbles(t3.AsByte() & low, mapped, unsigned).Store(into + 96);
        Nibbles(Vector256.ShiftRightLogical(t0, 4).AsByte() & low, mapped, unsigned).Store(into + 128);
        Nibbles(Vector256.ShiftRightLogical(t1, 4).AsByte() & low, mapped, unsigned).Store(into + 160);
        Nibbles(Vector256.ShiftRightLogical(t2, 4).AsByte() & low, mapped, unsigned).Store(into + 192);
        Nibbles(Vector256.ShiftRightLogical(t3, 4).AsByte() & low, mapped, unsigned).Store(into + 224);
    }

    // Four rows of x (codes from `x`, a row every `xStride`; scales and code sums from `xs`, a row every `sStride`) by one
    // stored row made into steps once (ScaledSteps: `groups` groups of 8 steps at `steps`, ScaledStepFloats a group; their
    // scales from `table`, 8 a group): each step dotted with each row's step of x, a block's sum (bias times x's code sum
    // to start) in its lane; then the blocks' scales times x's, 8 blocks at a time.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void ScaledStepsFour(float* steps, float* table, int groups, sbyte* x, float* xs, nint xStride, nint sStride,
        out float y0, out float y1, out float y2, out float y3)
    {
        Vector256<float> f0 = default, f1 = default, f2 = default, f3 = default;
        for (int g = 0; g < groups; g++)
        {
            byte* at = (byte*)(steps + g * ScaledStepFloats);
            var s0 = Vector256.Load(at);
            var s1 = Vector256.Load(at + 32);
            var s2 = Vector256.Load(at + 64);
            var s3 = Vector256.Load(at + 96);
            var s4 = Vector256.Load(at + 128);
            var s5 = Vector256.Load(at + 160);
            var s6 = Vector256.Load(at + 192);
            var s7 = Vector256.Load(at + 224);
            var dw = Vector256.Load(table + g * ScaledNibbles.Group);
            sbyte* xg = x + g * ScaledNibbles.GroupCodes;
            float* sg = xs + g * 16;

            var i0 = Vector256.Load((int*)(sg + 8));
            i0 = DotAt(i0, s0, xg);
            i0 = DotAt(i0, s1, xg + 32);
            i0 = DotAt(i0, s2, xg + 64);
            i0 = DotAt(i0, s3, xg + 96);
            i0 = DotAt(i0, s4, xg + 128);
            i0 = DotAt(i0, s5, xg + 160);
            i0 = DotAt(i0, s6, xg + 192);
            i0 = DotAt(i0, s7, xg + 224);
            f0 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i0), dw * Vector256.Load(sg), f0);

            xg += xStride;
            sg += sStride;
            var i1 = Vector256.Load((int*)(sg + 8));
            i1 = DotAt(i1, s0, xg);
            i1 = DotAt(i1, s1, xg + 32);
            i1 = DotAt(i1, s2, xg + 64);
            i1 = DotAt(i1, s3, xg + 96);
            i1 = DotAt(i1, s4, xg + 128);
            i1 = DotAt(i1, s5, xg + 160);
            i1 = DotAt(i1, s6, xg + 192);
            i1 = DotAt(i1, s7, xg + 224);
            f1 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i1), dw * Vector256.Load(sg), f1);

            xg += xStride;
            sg += sStride;
            var i2 = Vector256.Load((int*)(sg + 8));
            i2 = DotAt(i2, s0, xg);
            i2 = DotAt(i2, s1, xg + 32);
            i2 = DotAt(i2, s2, xg + 64);
            i2 = DotAt(i2, s3, xg + 96);
            i2 = DotAt(i2, s4, xg + 128);
            i2 = DotAt(i2, s5, xg + 160);
            i2 = DotAt(i2, s6, xg + 192);
            i2 = DotAt(i2, s7, xg + 224);
            f2 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i2), dw * Vector256.Load(sg), f2);

            xg += xStride;
            sg += sStride;
            var i3 = Vector256.Load((int*)(sg + 8));
            i3 = DotAt(i3, s0, xg);
            i3 = DotAt(i3, s1, xg + 32);
            i3 = DotAt(i3, s2, xg + 64);
            i3 = DotAt(i3, s3, xg + 96);
            i3 = DotAt(i3, s4, xg + 128);
            i3 = DotAt(i3, s5, xg + 160);
            i3 = DotAt(i3, s6, xg + 192);
            i3 = DotAt(i3, s7, xg + 224);
            f3 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i3), dw * Vector256.Load(sg), f3);
        }

        y0 = Vector256.Sum(f0);
        y1 = Vector256.Sum(f1);
        y2 = Vector256.Sum(f2);
        y3 = Vector256.Sum(f3);
    }

    // ScaledIntegerFour for one row of x.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float ScaledIntegerOne(byte* row, byte* tail, int groups, int full, ScaledNibbles plan, sbyte* x, float* xs)
    {
        var low = Vector256.Create((byte)15);
        var unsigned = plan.UnsignedCodes;
        bool mapped = plan.Mapped;
        nint bytes = plan.Bytes, codeOffset = plan.CodeOffset, scaleOffset = plan.ScaleOffset;
        Vector256<float> f = default;
        for (int g = 0; g < groups; g++)
        {
            byte* blocks = g < full ? row + g * ScaledNibbles.Group * bytes : tail;
            BlockWords(blocks + codeOffset, bytes, out var t0, out var t1, out var t2, out var t3);
            sbyte* xg = x + g * ScaledNibbles.GroupCodes;
            float* sg = xs + g * 16;
            var i = Vector256.Load((int*)(sg + 8));
            i = DotAt(i, Nibbles(t0.AsByte() & low, mapped, unsigned), xg);
            i = DotAt(i, Nibbles(t1.AsByte() & low, mapped, unsigned), xg + 32);
            i = DotAt(i, Nibbles(t2.AsByte() & low, mapped, unsigned), xg + 64);
            i = DotAt(i, Nibbles(t3.AsByte() & low, mapped, unsigned), xg + 96);
            i = DotAt(i, Nibbles(Vector256.ShiftRightLogical(t0, 4).AsByte() & low, mapped, unsigned), xg + 128);
            i = DotAt(i, Nibbles(Vector256.ShiftRightLogical(t1, 4).AsByte() & low, mapped, unsigned), xg + 160);
            i = DotAt(i, Nibbles(Vector256.ShiftRightLogical(t2, 4).AsByte() & low, mapped, unsigned), xg + 192);
            i = DotAt(i, Nibbles(Vector256.ShiftRightLogical(t3, 4).AsByte() & low, mapped, unsigned), xg + 224);
            f = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i), plan.Scales(blocks + scaleOffset) * Vector256.Load(sg), f);
        }

        return Vector256.Sum(f);
    }

    // A step's codes (0-15) as the integer lane reads them: as they are, or a mapped plan's unsigned values by a byte shuffle.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> Nibbles(Vector256<byte> codes, bool mapped, Vector256<byte> unsigned) => mapped ? Avx2.Shuffle(unsigned, codes) : codes;

    // A block's 32 codes (16 bytes at `codes`, ggml's split order) as four float vectors of their values before the scale,
    // in value order: the codes themselves (q4_0's, widened as bytes: low halves, then high), or a mapped plan's values by
    // a byte shuffle (MXFP4: E2M1's in halves), sign-extended.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void BlockCodes(byte* codes, ScaledNibbles plan, out Vector256<float> w0, out Vector256<float> w1, out Vector256<float> w2, out Vector256<float> w3)
    {
        if (plan.Mapped)
        {
            var v = Vector128.Load(codes);
            var nibble = Vector128.Create((byte)15);
            var lo = Ssse3.Shuffle(plan.CodeValues, (v & nibble).AsSByte());
            var hi = Ssse3.Shuffle(plan.CodeValues, (Vector128.ShiftRightLogical(v.AsUInt16(), 4).AsByte() & nibble).AsSByte());
            w0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(lo));
            w1 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(lo, 8)));
            w2 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(hi));
            w3 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(Sse2.ShiftRightLogical128BitLane(hi, 8)));
            return;
        }

        var mask = Vector256.Create(15);
        var ca = Avx2.ConvertToVector256Int32(codes);
        var cb = Avx2.ConvertToVector256Int32(codes + 8);
        w0 = Avx.ConvertToVector256Single(ca & mask);
        w1 = Avx.ConvertToVector256Single(cb & mask);
        w2 = Avx.ConvertToVector256Single(Vector256.ShiftRightLogical(ca, 4));
        w3 = Avx.ConvertToVector256Single(Vector256.ShiftRightLogical(cb, 4));
    }

    // Four rows of x (float32, k a row, from `x`) by one stored row of `blocks` blocks (groups of 8 from `row`, the last
    // short group from `tail`; their scales from `table`, 8 a group): each block's 32 codes decoded in registers to
    // (code + bias) · scale, exactly the values ByteBlockWiden gives (code · scale + bias · scale in one FMA: both exact),
    // each multiplied with the four rows; the first and second 8 of each half in their own sums.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void ScaledDecodedFour(byte* row, byte* tail, int blocks, ScaledNibbles plan, float* table, float* x, int k,
        out float y0, out float y1, out float y2, out float y3)
    {
        var bias = Vector256.Create((float)plan.ValueBias);
        nint bytes = plan.Bytes, codeOffset = plan.CodeOffset, r1 = k, r2 = 2 * (nint)k, r3 = 3 * (nint)k;
        int full = blocks / ScaledNibbles.Group;
        Vector256<float> a0 = default, a1 = default, a2 = default, a3 = default, h0 = default, h1 = default, h2 = default, h3 = default;
        for (int g = 0; g * ScaledNibbles.Group < blocks; g++)
        {
            byte* group = g < full ? row + g * ScaledNibbles.Group * bytes : tail;
            var dw = Vector256.Load(table + g * ScaledNibbles.Group);
            var bw = dw * bias;
            int count = Math.Min(ScaledNibbles.Group, blocks - g * ScaledNibbles.Group);
            for (int l = 0; l < count; l++)
            {
                var lane = Vector256.Create(l);
                var d = Avx2.PermuteVar8x32(dw, lane);
                var b = Avx2.PermuteVar8x32(bw, lane);
                BlockCodes(group + l * bytes + codeOffset, plan, out var c0, out var c1, out var c2, out var c3);
                var w0 = Fma.MultiplyAdd(c0, d, b);
                var w1 = Fma.MultiplyAdd(c1, d, b);
                var w2 = Fma.MultiplyAdd(c2, d, b);
                var w3 = Fma.MultiplyAdd(c3, d, b);
                float* xb = x + (g * ScaledNibbles.Group + l) * ScaledNibbles.Values;
                a0 = Fma.MultiplyAdd(w0, Vector256.Load(xb), a0);
                h0 = Fma.MultiplyAdd(w1, Vector256.Load(xb + 8), h0);
                a0 = Fma.MultiplyAdd(w2, Vector256.Load(xb + 16), a0);
                h0 = Fma.MultiplyAdd(w3, Vector256.Load(xb + 24), h0);
                a1 = Fma.MultiplyAdd(w0, Vector256.Load(xb + r1), a1);
                h1 = Fma.MultiplyAdd(w1, Vector256.Load(xb + r1 + 8), h1);
                a1 = Fma.MultiplyAdd(w2, Vector256.Load(xb + r1 + 16), a1);
                h1 = Fma.MultiplyAdd(w3, Vector256.Load(xb + r1 + 24), h1);
                a2 = Fma.MultiplyAdd(w0, Vector256.Load(xb + r2), a2);
                h2 = Fma.MultiplyAdd(w1, Vector256.Load(xb + r2 + 8), h2);
                a2 = Fma.MultiplyAdd(w2, Vector256.Load(xb + r2 + 16), a2);
                h2 = Fma.MultiplyAdd(w3, Vector256.Load(xb + r2 + 24), h2);
                a3 = Fma.MultiplyAdd(w0, Vector256.Load(xb + r3), a3);
                h3 = Fma.MultiplyAdd(w1, Vector256.Load(xb + r3 + 8), h3);
                a3 = Fma.MultiplyAdd(w2, Vector256.Load(xb + r3 + 16), a3);
                h3 = Fma.MultiplyAdd(w3, Vector256.Load(xb + r3 + 24), h3);
            }
        }

        y0 = Vector256.Sum(a0 + h0);
        y1 = Vector256.Sum(a1 + h1);
        y2 = Vector256.Sum(a2 + h2);
        y3 = Vector256.Sum(a3 + h3);
    }

    // ScaledDecodedFour for one row of x, each of a block's four vectors in its own sum.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float ScaledDecodedOne(byte* row, byte* tail, int blocks, ScaledNibbles plan, float* x)
    {
        var bias = Vector256.Create((float)plan.ValueBias);
        nint bytes = plan.Bytes, codeOffset = plan.CodeOffset, scaleOffset = plan.ScaleOffset;
        int full = blocks / ScaledNibbles.Group;
        Vector256<float> a0 = default, a1 = default, a2 = default, a3 = default;
        for (int g = 0; g * ScaledNibbles.Group < blocks; g++)
        {
            byte* group = g < full ? row + g * ScaledNibbles.Group * bytes : tail;
            var dw = plan.Scales(group + scaleOffset);
            var bw = dw * bias;
            int count = Math.Min(ScaledNibbles.Group, blocks - g * ScaledNibbles.Group);
            for (int l = 0; l < count; l++)
            {
                var lane = Vector256.Create(l);
                var d = Avx2.PermuteVar8x32(dw, lane);
                var b = Avx2.PermuteVar8x32(bw, lane);
                BlockCodes(group + l * bytes + codeOffset, plan, out var c0, out var c1, out var c2, out var c3);
                float* xb = x + (g * ScaledNibbles.Group + l) * ScaledNibbles.Values;
                a0 = Fma.MultiplyAdd(Fma.MultiplyAdd(c0, d, b), Vector256.Load(xb), a0);
                a1 = Fma.MultiplyAdd(Fma.MultiplyAdd(c1, d, b), Vector256.Load(xb + 8), a1);
                a2 = Fma.MultiplyAdd(Fma.MultiplyAdd(c2, d, b), Vector256.Load(xb + 16), a2);
                a3 = Fma.MultiplyAdd(Fma.MultiplyAdd(c3, d, b), Vector256.Load(xb + 24), a3);
            }
        }

        return Vector256.Sum((a0 + a1) + (a2 + a3));
    }

    /// <summary>
    /// A scaled-block format the 32-value nibble lanes read where it is stored, by its description alone (never its name):
    /// 32 values a block, 4-bit codes in ggml's split order (byte i holds value i in its low half and value i + 16 in its
    /// high half), and either unsigned codes with a bias, no code table and one float16 scale a block (ggml's q4_0), or
    /// codes each a value of a 4-bit element type with one scale a block of an 8-bit element type (OCP's MXFP4: fp4 E2M1
    /// codes, an E8M0 scale), the code type's 16 values whole numbers within int8 times one power of two (E2M1's: halves).
    /// Null for any other description, or where AVX2 and FMA do not run.
    /// </summary>
    private sealed unsafe class ScaledNibbles
    {
        /// <summary>Values a block holds.</summary>
        public const int Values = 32;

        /// <summary>Blocks the kernels take at a time (a lane each).</summary>
        public const int Group = 8;

        /// <summary>x's int8 codes a group takes in the integer lane (8 steps of 32).</summary>
        public const int GroupCodes = Group * Values;

        private static readonly ConditionalWeakTable<ScaledBlock, object> Plans = new();

        // Mapped codes: the scale type's 256 values times the codes' unit, pinned (the kernels gather them); else null.
        private readonly float[]? scaleValues;
        private readonly float* scaleTable;

        private ScaledNibbles(ScaledBlock block, sbyte[]? codes, float unit)
        {
            Bytes = block.Bytes;
            CodeOffset = block.CodeOffset;
            ScaleOffset = block.ScaleOffset;
            Offsets = Vector256.Create(0, Bytes, 2 * Bytes, 3 * Bytes, 4 * Bytes, 5 * Bytes, 6 * Bytes, 7 * Bytes);
            if (codes is null)
            {
                Bias = ValueBias = block.CodeBias;
                return;
            }

            // The integer lane's codes the values less the least (unsigned), the least back through x's code sums.
            int least = codes.Min();
            Bias = least;
            Mapped = true;
            Span<byte> unsigned = stackalloc byte[32];
            for (int c = 0; c < 32; c++)
            {
                unsigned[c] = (byte)(codes[c % 16] - least);
            }

            UnsignedCodes = Vector256.Create<byte>(unsigned);
            CodeValues = Vector128.Create<sbyte>(codes);
            scaleValues = GC.AllocateArray<float>(256, pinned: true);
            scaleTable = (float*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(scaleValues));
            Span<byte> one = stackalloc byte[1];
            for (int e = 0; e < 256; e++)
            {
                one[0] = (byte)e;
                block.ScaleType!.Decode(one, scaleValues.AsSpan(e, 1));
                scaleValues[e] *= unit;
            }
        }

        /// <summary>Bytes a block takes.</summary>
        public int Bytes { get; }

        /// <summary>The first code byte in a block.</summary>
        public int CodeOffset { get; }

        /// <summary>The scale's first byte in a block.</summary>
        public int ScaleOffset { get; }

        /// <summary>Added to each of the integer lane's codes before the scale (through x's code sums).</summary>
        public int Bias { get; }

        /// <summary>Added to each code's value before the scale in the decoded lane (q4_0's bias; 0 for mapped codes).</summary>
        public int ValueBias { get; }

        /// <summary>Whether each code is a value of a code type (MXFP4's E2M1), read through a byte shuffle, its scale a byte of a scale type.</summary>
        public bool Mapped { get; }

        /// <summary>Mapped codes: each code's value in the codes' unit less the least, both 128-bit halves (the integer lane's unsigned codes).</summary>
        public Vector256<byte> UnsignedCodes { get; }

        /// <summary>Mapped codes: each code's value in the codes' unit (the decoded lane's).</summary>
        public Vector128<sbyte> CodeValues { get; }

        /// <summary>Block l's first byte from a group's, in lane l.</summary>
        public Vector256<int> Offsets { get; }

        /// <summary>
        /// The scales of 8 blocks (block l's from <paramref name="at"/> + Offsets[l]) as float32: float16 exactly (HalfScales),
        /// or a mapped plan's scale byte's value of the scale type times the codes' unit (gathered from its table).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector256<float> Scales(byte* at) => Mapped
            ? Avx2.GatherVector256(scaleTable, Avx2.GatherVector256((int*)at, Offsets, 1) & Vector256.Create(0xFF), 4)
            : HalfScales(at, Offsets);

        /// <summary>Groups of 8 blocks a row of <paramref name="k"/> values takes (the last may be short).</summary>
        public static int Groups(int k) => (k / Values + Group - 1) / Group;

        /// <summary>The plan for <paramref name="block"/>, or null where the lanes do not read it.</summary>
        public static ScaledNibbles? Of(ScaledBlock block) => Plans.GetValue(block, static b => (object?)Make(b) ?? b) as ScaledNibbles;

        private static ScaledNibbles? Make(ScaledBlock b)
        {
            bool runs = Avx2.IsSupported && Fma.IsSupported && BitConverter.IsLittleEndian
                && b is { Values: Values, CodeBits: 4, NibblesSplit: true, CodeTable: null }
                && b.CodeOffset >= 0 && b.CodeOffset + Values / 2 <= b.Bytes && b.ScaleOffset >= 0 && b.ScaleOffset + 4 <= b.Bytes;
            if (!runs)
            {
                return null;
            }

            if (b is { CodeType: null, ScaleType: null, SignedCodes: false, Scale: { Exponent: 5, Mantissa: 10 }, CodeBias: >= -127 and <= 127 })
            {
                return new ScaledNibbles(b, null, 1f);
            }

            // Codes of a 4-bit type and a scale of an 8-bit type, no bias: the code type's values as int8 integers times 2^-s.
            if (b is not { CodeType: { BlockBits: 4, ToFloat32: not null } codeType, ScaleType: { BlockBits: 8, ToFloat32: not null }, CodeBias: 0 })
            {
                return null;
            }

            Span<float> values = stackalloc float[16];
            Span<byte> code = stackalloc byte[1];
            for (int c = 0; c < 16; c++)
            {
                code[0] = (byte)c;
                codeType.Decode(code, values.Slice(c, 1));
            }

            for (int s = 0; s <= 6; s++)
            {
                var codes = new sbyte[16];
                bool whole = true;
                for (int c = 0; c < 16 && whole; c++)
                {
                    float scaled = values[c] * (1 << s);
                    whole = float.IsFinite(scaled) && scaled == MathF.Round(scaled) && scaled is >= -127f and <= 127f;
                    codes[c] = whole ? (sbyte)scaled : (sbyte)0;
                }

                // The integer lane's unsigned codes stay within what AVX2's multiply-adds take without saturating.
                if (whole && codes.Max() - codes.Min() <= 127)
                {
                    return new ScaledNibbles(b, codes, 1f / (1 << s));
                }
            }

            return null;
        }
    }

    // ------------------------------------------------------------------ bfloat16 rows read where they are stored

    // A bfloat16 weight (by its description: a 16-bit float of 8 exponent and 7 mantissa bits, float32's high half) takes,
    // under the measured route, two lanes over its stored rows (RowGroups' kernels), neither of which writes the weight's
    // values anywhere: each value made float32 in registers as the dot products read it (its bits in a float's high half:
    // exactly its value), then float32 FMAs with x. The paired lane makes two vectors of each 32-byte load, one of its even
    // values (each 32-bit pair shifted up by 16 bits) and one of its odd values (each pair's low half masked off), x
    // reordered once a call to match; the shifted lane widens 8 values at a time by a widening load and a shift, x as it
    // is. The measured route keeps the faster at each shape.

    // Whether a type is one the bfloat16 lanes read (by its description), and the CPU runs them.
    private static bool ReadsBFloat16Rows(ElementType type) =>
        WideningOf(type) == Widening.HighHalf && Avx2.IsSupported && Fma.IsSupported && BitConverter.IsLittleEndian;

    // bfloat16 as a format read where it is stored (IRowFormat): by its description (ReadsBFloat16Rows); the shifted lane
    // timed against the paired lane; x reordered for the paired lane (PairRows); a group of 64 KB of stored rows, passes
    // over k of BFloat16Span values; no scratch (BFloat16Kernels).
    private readonly unsafe struct BFloat16Format : IRowFormat<BFloat16Format, BFloat16Kernels>
    {
        public static bool Reads(ElementType type, int k, out BFloat16Format format)
        {
            format = default;
            return ReadsBFloat16Rows(type);
        }

        public int First => ShiftedLaneId;

        public int Second => PairedLaneId;

        public bool BothRun => true;

        public int Unmeasured => ShiftedLaneId;

        public int GroupBytes => BFloat16GroupBytes;

        public (int Columns, int Rows) Tile(int total, int m, int threads, nint rowBytes) => GroupTile(total, rowBytes, BFloat16GroupBytes);

        public int Span(int k) => BFloat16Span;

        public LaneInput Prepare(int lane, float[] x, int m, int k)
        {
            if (lane != PairedLaneId)
            {
                return new LaneInput(x, false, null, null);
            }

            var reordered = ArrayPool<float>.Shared.Rent(m * k);
            PairRows(x, reordered, m, k);
            return new LaneInput(reordered, true, null, null);
        }

        public (int Floats, int Bytes) Scratch(int k, int group) => (0, 0);

        public BFloat16Kernels Kernels(int lane, int k, LanePointers x, float* floats, byte* bytes) => new(lane == PairedLaneId, x.Values, k);
    }

    // Stored bytes a group of bfloat16 rows takes with more than one row of x (RowGroups): float32 rows of x take 4 bytes a
    // value against the rows' 2, so the group is larger than a 4-bit weight's for the rows of x to be read as few times.
    private const int BFloat16GroupBytes = 65536;

    // Values of k four rows of x pass over a group's bfloat16 rows at a time (RowGroups): 32 KB of x, which stays in a core's
    // L1 across the group.
    private const int BFloat16Span = 2048;

    // The bfloat16 lanes as RowGroups' kernels: nothing to set up for a group of rows.
    private readonly unsafe struct BFloat16Kernels(bool paired, float* x, int k) : IRowKernels
    {
        public void Group(byte* stored, nint rowBytes, int c0, int count)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Four(byte* row, int c, int i, int p0, int p1, out float y0, out float y1, out float y2, out float y3)
        {
            if (paired)
            {
                PairedFour((ushort*)row + p0, x + (nint)i * k + p0, k, p1 - p0, out y0, out y1, out y2, out y3);
            }
            else
            {
                ShiftedFour((ushort*)row + p0, x + (nint)i * k + p0, k, p1 - p0, out y0, out y1, out y2, out y3);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float One(byte* row, int c, int i) => Single(row, i);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Single(byte* row, int i) => paired
            ? PairedOne((ushort*)row, x + (nint)i * k, k)
            : ShiftedOne((ushort*)row, x + (nint)i * k, k);
    }

    // x for the paired lane, row by row (PairRow), over the threads where x is large.
    private static void PairRows(float[] x, float[] into, int m, int k)
    {
        if (m > 1 && (long)m * k >= 1 << 16)
        {
            Parallel.For(0, m, ComputeResources.ParallelOptions, i => PairRow(x.AsSpan(i * k, k), into.AsSpan(i * k, k)));
            return;
        }

        for (int i = 0; i < m; i++)
        {
            PairRow(x.AsSpan(i * k, k), into.AsSpan(i * k, k));
        }
    }

    // A row of x for the paired lane: each whole group of 16 values in the order the lane makes of 16 stored values (the even
    // ones 0, 2, .. 14, then the odd ones 1, 3, .. 15), the rest as they are.
    private static void PairRow(ReadOnlySpan<float> x, Span<float> into)
    {
        ref float from = ref MemoryMarshal.GetReference(x);
        ref float to = ref MemoryMarshal.GetReference(into);
        int p = 0;
        for (; p + 16 <= x.Length; p += 16)
        {
            var a = Vector256.LoadUnsafe(ref from, (nuint)p);
            var b = Vector256.LoadUnsafe(ref from, (nuint)p + 8);
            Avx2.Permute4x64(Avx.Shuffle(a, b, 0x88).AsDouble(), 0xD8).AsSingle().StoreUnsafe(ref to, (nuint)p);
            Avx2.Permute4x64(Avx.Shuffle(a, b, 0xDD).AsDouble(), 0xD8).AsSingle().StoreUnsafe(ref to, (nuint)p + 8);
        }

        x[p..].CopyTo(into[p..]);
    }

    // Values the one-row bfloat16 kernels ask the cache for ahead of the line they read (1 KB: past a row's end, into the next
    // row of the tile, where the hardware's prefetchers stop at a page's edge). A prefetch never faults, past the weight too.
    private const int BFloat16Ahead = 512;

    // A stored bfloat16's value: its bits in a float32's high half.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float BFloat16Value(ushort bits) => BitConverter.Int32BitsToSingle(bits << 16);

    // Four rows of x (reordered by PairRow, `stride` apart, from `x`) by `count` values of a stored bfloat16 row (at
    // `row`; `count` a multiple of 16 but at a row's end): each 16 values loaded at once and made two float32 vectors, the
    // even values by a shift of each 32-bit pair and the odd ones by a mask, each multiplied with the four rows' matching 8;
    // the halves in their own sums, the last count mod 16 one at a time.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void PairedFour(ushort* row, float* x, int stride, int count, out float y0, out float y1, out float y2, out float y3)
    {
        var high = Vector256.Create(0xFFFF0000u);
        nint r1 = stride, r2 = 2 * (nint)stride, r3 = 3 * (nint)stride;
        Vector256<float> a0 = default, a1 = default, a2 = default, a3 = default, b0 = default, b1 = default, b2 = default, b3 = default;
        int p = 0;
        for (; p + 16 <= count; p += 16)
        {
            var v = Vector256.Load((uint*)(row + p));
            var lo = Vector256.ShiftLeft(v, 16).AsSingle();
            var hi = (v & high).AsSingle();
            float* xp = x + p;
            a0 = Fma.MultiplyAdd(lo, Vector256.Load(xp), a0);
            b0 = Fma.MultiplyAdd(hi, Vector256.Load(xp + 8), b0);
            a1 = Fma.MultiplyAdd(lo, Vector256.Load(xp + r1), a1);
            b1 = Fma.MultiplyAdd(hi, Vector256.Load(xp + r1 + 8), b1);
            a2 = Fma.MultiplyAdd(lo, Vector256.Load(xp + r2), a2);
            b2 = Fma.MultiplyAdd(hi, Vector256.Load(xp + r2 + 8), b2);
            a3 = Fma.MultiplyAdd(lo, Vector256.Load(xp + r3), a3);
            b3 = Fma.MultiplyAdd(hi, Vector256.Load(xp + r3 + 8), b3);
        }

        float t0 = 0f, t1 = 0f, t2 = 0f, t3 = 0f;
        for (; p < count; p++)
        {
            float w = BFloat16Value(row[p]);
            t0 = MathF.FusedMultiplyAdd(w, x[p], t0);
            t1 = MathF.FusedMultiplyAdd(w, x[r1 + p], t1);
            t2 = MathF.FusedMultiplyAdd(w, x[r2 + p], t2);
            t3 = MathF.FusedMultiplyAdd(w, x[r3 + p], t3);
        }

        y0 = Vector256.Sum(a0 + b0) + t0;
        y1 = Vector256.Sum(a1 + b1) + t1;
        y2 = Vector256.Sum(a2 + b2) + t2;
        y3 = Vector256.Sum(a3 + b3) + t3;
    }

    // PairedFour for one row of x, 32 values a step in four sums.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float PairedOne(ushort* row, float* x, int k)
    {
        var high = Vector256.Create(0xFFFF0000u);
        Vector256<float> a = default, b = default, c = default, d = default;
        int p = 0;
        for (; p + 32 <= k; p += 32)
        {
            Sse.Prefetch0(row + p + BFloat16Ahead);
            var v = Vector256.Load((uint*)(row + p));
            var u = Vector256.Load((uint*)(row + p + 16));
            a = Fma.MultiplyAdd(Vector256.ShiftLeft(v, 16).AsSingle(), Vector256.Load(x + p), a);
            b = Fma.MultiplyAdd((v & high).AsSingle(), Vector256.Load(x + p + 8), b);
            c = Fma.MultiplyAdd(Vector256.ShiftLeft(u, 16).AsSingle(), Vector256.Load(x + p + 16), c);
            d = Fma.MultiplyAdd((u & high).AsSingle(), Vector256.Load(x + p + 24), d);
        }

        for (; p + 16 <= k; p += 16)
        {
            var v = Vector256.Load((uint*)(row + p));
            a = Fma.MultiplyAdd(Vector256.ShiftLeft(v, 16).AsSingle(), Vector256.Load(x + p), a);
            b = Fma.MultiplyAdd((v & high).AsSingle(), Vector256.Load(x + p + 8), b);
        }

        float t = 0f;
        for (; p < k; p++)
        {
            t = MathF.FusedMultiplyAdd(BFloat16Value(row[p]), x[p], t);
        }

        return Vector256.Sum((a + b) + (c + d)) + t;
    }

    // Four rows of x (as they are, `stride` apart, from `x`) by `count` values of a stored bfloat16 row (at `row`): 8 values
    // at a time widened by a widening load and shifted to a float32's high half, each multiplied with the four rows'
    // matching 8; two sums a row, the last count mod 16 one at a time.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void ShiftedFour(ushort* row, float* x, int stride, int count, out float y0, out float y1, out float y2, out float y3)
    {
        nint r1 = stride, r2 = 2 * (nint)stride, r3 = 3 * (nint)stride;
        Vector256<float> a0 = default, a1 = default, a2 = default, a3 = default, b0 = default, b1 = default, b2 = default, b3 = default;
        int p = 0;
        for (; p + 16 <= count; p += 16)
        {
            var lo = Avx2.ShiftLeftLogical(Avx2.ConvertToVector256Int32(row + p), 16).AsSingle();
            var hi = Avx2.ShiftLeftLogical(Avx2.ConvertToVector256Int32(row + p + 8), 16).AsSingle();
            float* xp = x + p;
            a0 = Fma.MultiplyAdd(lo, Vector256.Load(xp), a0);
            b0 = Fma.MultiplyAdd(hi, Vector256.Load(xp + 8), b0);
            a1 = Fma.MultiplyAdd(lo, Vector256.Load(xp + r1), a1);
            b1 = Fma.MultiplyAdd(hi, Vector256.Load(xp + r1 + 8), b1);
            a2 = Fma.MultiplyAdd(lo, Vector256.Load(xp + r2), a2);
            b2 = Fma.MultiplyAdd(hi, Vector256.Load(xp + r2 + 8), b2);
            a3 = Fma.MultiplyAdd(lo, Vector256.Load(xp + r3), a3);
            b3 = Fma.MultiplyAdd(hi, Vector256.Load(xp + r3 + 8), b3);
        }

        float t0 = 0f, t1 = 0f, t2 = 0f, t3 = 0f;
        for (; p < count; p++)
        {
            float w = BFloat16Value(row[p]);
            t0 = MathF.FusedMultiplyAdd(w, x[p], t0);
            t1 = MathF.FusedMultiplyAdd(w, x[r1 + p], t1);
            t2 = MathF.FusedMultiplyAdd(w, x[r2 + p], t2);
            t3 = MathF.FusedMultiplyAdd(w, x[r3 + p], t3);
        }

        y0 = Vector256.Sum(a0 + b0) + t0;
        y1 = Vector256.Sum(a1 + b1) + t1;
        y2 = Vector256.Sum(a2 + b2) + t2;
        y3 = Vector256.Sum(a3 + b3) + t3;
    }

    // ShiftedFour for one row of x, 32 values a step in four sums.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float ShiftedOne(ushort* row, float* x, int k)
    {
        Vector256<float> a = default, b = default, c = default, d = default;
        int p = 0;
        for (; p + 32 <= k; p += 32)
        {
            Sse.Prefetch0(row + p + BFloat16Ahead);
            a = Fma.MultiplyAdd(Avx2.ShiftLeftLogical(Avx2.ConvertToVector256Int32(row + p), 16).AsSingle(), Vector256.Load(x + p), a);
            b = Fma.MultiplyAdd(Avx2.ShiftLeftLogical(Avx2.ConvertToVector256Int32(row + p + 8), 16).AsSingle(), Vector256.Load(x + p + 8), b);
            c = Fma.MultiplyAdd(Avx2.ShiftLeftLogical(Avx2.ConvertToVector256Int32(row + p + 16), 16).AsSingle(), Vector256.Load(x + p + 16), c);
            d = Fma.MultiplyAdd(Avx2.ShiftLeftLogical(Avx2.ConvertToVector256Int32(row + p + 24), 16).AsSingle(), Vector256.Load(x + p + 24), d);
        }

        for (; p + 8 <= k; p += 8)
        {
            a = Fma.MultiplyAdd(Avx2.ShiftLeftLogical(Avx2.ConvertToVector256Int32(row + p), 16).AsSingle(), Vector256.Load(x + p), a);
        }

        float t = 0f;
        for (; p < k; p++)
        {
            t = MathF.FusedMultiplyAdd(BFloat16Value(row[p]), x[p], t);
        }

        return Vector256.Sum((a + b) + (c + d)) + t;
    }

    // ------------------------------------------------------------------ 64-bit integer codes read where they are stored

    // A weight of 64-bit two's complement codes (by its description: a signed integer type of 64 bits a value: int64's
    // weights, codes of its full range with one scale an output row, StoredPackedFormat.RowScaled) takes, under the measured
    // route, two lanes, neither of which writes the weight's values anywhere nor narrows them: the decoded lane makes each
    // code float64 in registers (one rounding of 64 bits to 53) and multiplies it with x widened to float64 once a call, its
    // sums in float64; the integer lane rounds x to int8 a block of 32 values (one scale a block) and takes the CPU's int8
    // dot products of x's codes with each byte of the stored codes, four values' byte b in lane b of a step (the code's
    // eight byte planes in eight lanes), the top byte's sign through a bias of 128 and x's code sums, each plane's sums
    // times 2^(8b) and the block's scale. Each output times its row's scale (RowGroups). The measured route keeps the faster.

    // int64 codes as a format read where they are stored (IRowFormat): by their description; the decoded lane timed against
    // the integer lane (the decoded alone where the CPU has no 8-bit dot products of unsigned bytes, AVX-VNNI, or k is not
    // whole blocks of 32); x widened to float64 for the decoded lane, rounded to int8 for the integer lane (RoundLongRows).
    private readonly unsafe struct LongFormat : IRowFormat<LongFormat, LongKernels>
    {
        private readonly bool reads, blocks;

        private LongFormat(bool blocks)
        {
            reads = true;
            this.blocks = blocks;
        }

        public static bool Reads(ElementType type, int k, out LongFormat format)
        {
            format = type is { IsInteger: true, IsSigned: true, BlockSize: 1, BlockBits: 64 } && Avx2.IsSupported && Fma.IsSupported && BitConverter.IsLittleEndian
                ? new LongFormat(k % LongBlock == 0) : default;
            return format.reads;
        }

        public int First => DecodedLaneId;

        public int Second => IntegerLaneId;

        public bool BothRun => blocks && AvxVnni.IsSupported;

        public int Unmeasured => DecodedLaneId;

        public int GroupBytes => LongGroupBytes;

        // Columns over up to LongTileRows rows of x, an eighth of a thread's share: each stored row (8 bytes a value) read from
        // memory once a product of up to that many rows (32-row tiles read it once a block of rows: measured about 65 GB/s
        // of weights on 128-row prompts, memory-bound; tiles of every row of a 512-row window stream x from L3 for every
        // group of stored rows).
        public (int Columns, int Rows) Tile(int total, int m, int threads, nint rowBytes) => ((total + 8 * threads - 1) / (8 * threads), Math.Min(m, LongTileRows));

        public int Span(int k) => k;

        public LaneInput Prepare(int lane, float[] x, int m, int k)
        {
            if (lane != IntegerLaneId)
            {
                var wide = ArrayPool<float>.Shared.Rent(2 * m * k);
                WidenRows(x, wide, m, k);
                return new LaneInput(wide, true, null, null);
            }

            var codes = ArrayPool<sbyte>.Shared.Rent(m * k);
            var scales = ArrayPool<float>.Shared.Rent(m * (k / LongBlock) * 2);
            RoundLongRows(x, codes, scales, m, k);
            return new LaneInput(x, false, codes, scales);
        }

        // The integer lane's steps of a group's stored rows (each row's bytes, transposed): as floats, which the shared
        // driver does not clear (every byte is written before it is read).
        public (int Floats, int Bytes) Scratch(int k, int group) => (group * 2 * k, 0);

        public LongKernels Kernels(int lane, int k, LanePointers x, float* floats, byte* bytes) => new(lane == IntegerLaneId, k, x, (byte*)floats);
    }

    // Values of x a block of the integer lane's rounding takes (one scale each).
    private const int LongBlock = 32;

    // Rows of x an int64 tile takes at most (LongFormat.Tile): x's int8 codes and block tables for that many rows of k values
    // (about 1.25 bytes a value) stay in a core's L2 while a group of stored rows passes under them.
    private const int LongTileRows = 128;

    // Stored bytes a group of int64 rows takes with more than one row of x (RowGroups), all of k a pass. (256 KB groups
    // passed over in spans of 512 values of k measured no faster on prompts: 173 tokens a second against 176.)
    private const int LongGroupBytes = 65536;

    // The int64 lanes as RowGroups' kernels. The integer lane: a group's stored rows made into steps once (LongStep: each
    // row's 8-byte codes transposed into byte planes, in `steps`), which every four rows of x then read; a row read alone
    // makes its steps as it reads them. x as float64 (the decoded lane: its values, two floats' room each) or as int8 codes
    // with 2 floats a block (the integer lane: RoundLongRow).
    private readonly unsafe struct LongKernels : IRowKernels
    {
        private readonly bool integer;
        private readonly int k;
        private readonly nint table;
        private readonly LanePointers input;
        private readonly byte* steps;

        public LongKernels(bool integer, int k, LanePointers input, byte* steps)
        {
            this.integer = integer;
            this.k = k;
            this.input = input;
            this.steps = steps;
            table = (nint)(k / LongBlock) * 2;
        }

        public void Group(byte* stored, nint rowBytes, int c0, int count)
        {
            if (integer)
            {
                for (int c = 0; c < count; c++)
                {
                    LongSteps((long*)(stored + (c0 + c) * rowBytes), steps + c * rowBytes, k);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Four(byte* row, int c, int i, int p0, int p1, out float y0, out float y1, out float y2, out float y3)
        {
            if (integer)
            {
                LongIntegerFour(steps + ((nint)c * k + p0) * sizeof(long), input.Codes + (nint)i * k + p0, input.Scales + i * table + p0 / LongBlock * 2, k, table, p1 - p0,
                    out y0, out y1, out y2, out y3);
            }
            else
            {
                WideFour<LongLoad>(row + (nint)p0 * sizeof(long), (double*)input.Values + (nint)i * k + p0, k, p1 - p0, out y0, out y1, out y2, out y3);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float One(byte* row, int c, int i) => Single(row, i);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Single(byte* row, int i) => integer
            ? LongIntegerOne((long*)row, input.Codes + (nint)i * k, input.Scales + i * table, k)
            : (float)WideSum<LongLoad>(row, (double*)input.Values + (nint)i * k, k);
    }

    // x as float64 for the decoded lane (each value exactly), row by row, over the threads where x is large; `into` holds
    // the doubles' bytes (two floats' room a value).
    private static unsafe void WidenRows(float[] x, float[] into, int m, int k)
    {
        static void Row(ReadOnlySpan<float> from, Span<double> to)
        {
            int p = 0;
            for (; p + 4 <= from.Length; p += 4)
            {
                Avx.ConvertToVector256Double(Vector128.Create(from.Slice(p, 4))).CopyTo(to.Slice(p, 4));
            }

            for (; p < from.Length; p++)
            {
                to[p] = from[p];
            }
        }

        var doubles = MemoryMarshal.Cast<float, double>(into.AsSpan(0, 2 * m * k));
        if (m > 1 && (long)m * k >= 1 << 16)
        {
            fixed (float* wide = into)
            {
                nint at = (nint)wide;
                Parallel.For(0, m, ComputeResources.ParallelOptions, i => Row(x.AsSpan(i * k, k), new Span<double>((double*)at + (nint)i * k, k)));
            }

            return;
        }

        for (int i = 0; i < m; i++)
        {
            Row(x.AsSpan(i * k, k), doubles.Slice(i * k, k));
        }
    }

    // x rounded to int8 for the int64 integer lane, row by row (RoundLongRow), over the threads where x is large.
    private static void RoundLongRows(float[] x, sbyte[] codes, float[] scales, int m, int k)
    {
        int table = k / LongBlock * 2;
        if (m > 1 && (long)m * k >= 1 << 16)
        {
            Parallel.For(0, m, ComputeResources.ParallelOptions, i => RoundLongRow(x.AsSpan(i * k, k), codes.AsSpan(i * k, k), scales.AsSpan(i * table, table)));
            return;
        }

        for (int i = 0; i < m; i++)
        {
            RoundLongRow(x.AsSpan(i * k, k), codes.AsSpan(i * k, k), scales.AsSpan(i * table, table));
        }
    }

    // A row of x rounded to int8 a block of 32 values at a time: the block's scale amax / 127 (0 where every value is 0), its
    // codes value · 127 / amax rounded to even, in value order; per block 2 floats: the scale, and as int32 bits −128 times
    // the codes' sum (the top byte's bias, which the integer lane's byte plane 7 starts from). The integer lane's product is
    // then the product of x so rounded.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void RoundLongRow(ReadOnlySpan<float> x, Span<sbyte> codes, Span<float> scales)
    {
        ref float row = ref MemoryMarshal.GetReference(x);
        var sums = MemoryMarshal.Cast<float, int>(scales);
        for (int b = 0; b < x.Length / LongBlock; b++)
        {
            ref float v = ref Unsafe.Add(ref row, b * LongBlock);
            var v0 = Vector256.LoadUnsafe(ref v);
            var v1 = Vector256.LoadUnsafe(ref v, 8);
            var v2 = Vector256.LoadUnsafe(ref v, 16);
            var v3 = Vector256.LoadUnsafe(ref v, 24);
            var most = Vector256.Max(Vector256.Max(Vector256.Abs(v0), Vector256.Abs(v1)), Vector256.Max(Vector256.Abs(v2), Vector256.Abs(v3)));
            var top = Vector128.Max(most.GetLower(), most.GetUpper());
            top = Vector128.Max(top, Vector128.Shuffle(top, Vector128.Create(2, 3, 0, 1)));
            top = Vector128.Max(top, Vector128.Shuffle(top, Vector128.Create(1, 0, 3, 2)));
            float amax = top.ToScalar();
            var inverse = Vector256.Create(amax > 0f ? 127f / amax : 0f);
            var q0 = Vector256.ConvertToInt32(Vector256.Round(v0 * inverse));
            var q1 = Vector256.ConvertToInt32(Vector256.Round(v1 * inverse));
            var q2 = Vector256.ConvertToInt32(Vector256.Round(v2 * inverse));
            var q3 = Vector256.ConvertToInt32(Vector256.Round(v3 * inverse));
            Vector256.Narrow(Vector256.Narrow(q0, q1), Vector256.Narrow(q2, q3)).CopyTo(codes.Slice(b * LongBlock, LongBlock));
            scales[2 * b] = amax / 127f;
            sums[2 * b + 1] = -128 * Vector256.Sum((q0 + q1) + (q2 + q3));
        }
    }

    // Each byte plane's weight 2^(8l), in lane l (the integer lane's sums, a lane a plane, times these at the end).
    private static Vector256<float> LongPlanes => Vector256.Create(1f, 256f, 65536f, 16777216f, 4294967296f, 1099511627776f, 281474976710656f, 72057594037927936f);

    // The sums a block of the integer lane starts from: 0 in lanes 0-6, the top byte's bias (x's int32 bits at `xs` + 1) in 7.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<int> LongStart(float* xs) => Avx2.Blend(Vector256<int>.Zero, Vector256.Create(*(int*)(xs + 1)), 0x80);

    // Four int64 codes (at `at`) as float64: exact up to 2^53, else rounded once (the high 16 bits and the low 48 each made
    // a float64 exactly through the bits of 3 · 2^67 and 2^52, then added).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<double> LongsToDoubles(long* at)
    {
        var v = Vector256.Load(at);
        var high = Avx2.Blend(Avx2.ShiftRightArithmetic(v.AsInt32(), 16).AsInt16(), Vector256<short>.Zero, 0x33).AsInt64() + Vector256.Create(442721857769029238784.0).AsInt64();
        var low = Avx2.Blend(v.AsInt16(), Vector256.Create(4503599627370496.0).AsInt16(), 0x88).AsDouble();
        return (high.AsDouble() - Vector256.Create(442726361368656609280.0)) + low;
    }

    /// <summary>How a stored 64-bit value is made float64 in registers (the decoded lanes of int64's and float64's weights).</summary>
    private unsafe interface IWideLoad
    {
        /// <summary>The four values from <paramref name="at"/> as float64.</summary>
        static abstract Vector256<double> Four(byte* at);

        /// <summary>The value at <paramref name="at"/> as float64.</summary>
        static abstract double One(byte* at);
    }

    // int64 codes: LongsToDoubles (one rounding of 64 bits to 53).
    private unsafe struct LongLoad : IWideLoad
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<double> Four(byte* at) => LongsToDoubles((long*)at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double One(byte* at) => *(long*)at;
    }

    // float64 values: as stored.
    private unsafe struct DoubleLoad : IWideLoad
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector256<double> Four(byte* at) => Vector256.Load((double*)at);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double One(byte* at) => *(double*)at;
    }

    // Four rows of x (float64, `stride` apart, from `x`) by `count` 64-bit values of a stored row (at `row`): 8 values at a
    // time made float64 in registers (TL), each multiplied with the four rows' matching values in float64 FMAs; two sums a
    // row, the last count mod 8 one at a time; each output its float64 sum rounded to float32 once.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void WideFour<TL>(byte* row, double* x, int stride, int count, out float y0, out float y1, out float y2, out float y3) where TL : struct, IWideLoad
    {
        nint r1 = stride, r2 = 2 * (nint)stride, r3 = 3 * (nint)stride;
        Vector256<double> a0 = default, a1 = default, a2 = default, a3 = default, b0 = default, b1 = default, b2 = default, b3 = default;
        int p = 0;
        for (; p + 8 <= count; p += 8)
        {
            var w0 = TL.Four(row + 8 * p);
            var w1 = TL.Four(row + 8 * p + 32);
            double* xp = x + p;
            a0 = Fma.MultiplyAdd(w0, Vector256.Load(xp), a0);
            b0 = Fma.MultiplyAdd(w1, Vector256.Load(xp + 4), b0);
            a1 = Fma.MultiplyAdd(w0, Vector256.Load(xp + r1), a1);
            b1 = Fma.MultiplyAdd(w1, Vector256.Load(xp + r1 + 4), b1);
            a2 = Fma.MultiplyAdd(w0, Vector256.Load(xp + r2), a2);
            b2 = Fma.MultiplyAdd(w1, Vector256.Load(xp + r2 + 4), b2);
            a3 = Fma.MultiplyAdd(w0, Vector256.Load(xp + r3), a3);
            b3 = Fma.MultiplyAdd(w1, Vector256.Load(xp + r3 + 4), b3);
        }

        double t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        for (; p < count; p++)
        {
            double w = TL.One(row + 8 * p);
            t0 = Math.FusedMultiplyAdd(w, x[p], t0);
            t1 = Math.FusedMultiplyAdd(w, x[r1 + p], t1);
            t2 = Math.FusedMultiplyAdd(w, x[r2 + p], t2);
            t3 = Math.FusedMultiplyAdd(w, x[r3 + p], t3);
        }

        y0 = (float)(Vector256.Sum(a0 + b0) + t0);
        y1 = (float)(Vector256.Sum(a1 + b1) + t1);
        y2 = (float)(Vector256.Sum(a2 + b2) + t2);
        y3 = (float)(Vector256.Sum(a3 + b3) + t3);
    }

    // WideFour for one row of x, 16 values a step in four sums. (Asking for the bytes 1 KB ahead measured no faster on
    // int64: generation streams the weights at the memory's rate, 82 GB/s, as the head's product alone does.)
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe double WideSum<TL>(byte* row, double* x, int k) where TL : struct, IWideLoad
    {
        Vector256<double> a = default, b = default, c = default, d = default;
        int p = 0;
        for (; p + 16 <= k; p += 16)
        {
            byte* at = row + 8 * p;
            a = Fma.MultiplyAdd(TL.Four(at), Vector256.Load(x + p), a);
            b = Fma.MultiplyAdd(TL.Four(at + 32), Vector256.Load(x + p + 4), b);
            c = Fma.MultiplyAdd(TL.Four(at + 64), Vector256.Load(x + p + 8), c);
            d = Fma.MultiplyAdd(TL.Four(at + 96), Vector256.Load(x + p + 12), d);
        }

        for (; p + 4 <= k; p += 4)
        {
            a = Fma.MultiplyAdd(TL.Four(row + 8 * p), Vector256.Load(x + p), a);
        }

        double t = 0;
        for (; p < k; p++)
        {
            t = Math.FusedMultiplyAdd(TL.One(row + 8 * p), x[p], t);
        }

        return Vector256.Sum((a + b) + (c + d)) + t;
    }

    // WideFour with x as float32 (`stride` apart, from `x`), each 4 values widened to float64 in registers as they are read:
    // the same FMAs in the same order, so the same sums.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void WideMixedFour<TL>(byte* row, float* x, int stride, int count, out float y0, out float y1, out float y2, out float y3) where TL : struct, IWideLoad
    {
        nint r1 = stride, r2 = 2 * (nint)stride, r3 = 3 * (nint)stride;
        Vector256<double> a0 = default, a1 = default, a2 = default, a3 = default, b0 = default, b1 = default, b2 = default, b3 = default;
        int p = 0;
        for (; p + 8 <= count; p += 8)
        {
            var w0 = TL.Four(row + 8 * p);
            var w1 = TL.Four(row + 8 * p + 32);
            float* xp = x + p;
            a0 = Fma.MultiplyAdd(w0, Avx.ConvertToVector256Double(Sse.LoadVector128(xp)), a0);
            b0 = Fma.MultiplyAdd(w1, Avx.ConvertToVector256Double(Sse.LoadVector128(xp + 4)), b0);
            a1 = Fma.MultiplyAdd(w0, Avx.ConvertToVector256Double(Sse.LoadVector128(xp + r1)), a1);
            b1 = Fma.MultiplyAdd(w1, Avx.ConvertToVector256Double(Sse.LoadVector128(xp + r1 + 4)), b1);
            a2 = Fma.MultiplyAdd(w0, Avx.ConvertToVector256Double(Sse.LoadVector128(xp + r2)), a2);
            b2 = Fma.MultiplyAdd(w1, Avx.ConvertToVector256Double(Sse.LoadVector128(xp + r2 + 4)), b2);
            a3 = Fma.MultiplyAdd(w0, Avx.ConvertToVector256Double(Sse.LoadVector128(xp + r3)), a3);
            b3 = Fma.MultiplyAdd(w1, Avx.ConvertToVector256Double(Sse.LoadVector128(xp + r3 + 4)), b3);
        }

        double t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        for (; p < count; p++)
        {
            double w = TL.One(row + 8 * p);
            t0 = Math.FusedMultiplyAdd(w, x[p], t0);
            t1 = Math.FusedMultiplyAdd(w, x[r1 + p], t1);
            t2 = Math.FusedMultiplyAdd(w, x[r2 + p], t2);
            t3 = Math.FusedMultiplyAdd(w, x[r3 + p], t3);
        }

        y0 = (float)(Vector256.Sum(a0 + b0) + t0);
        y1 = (float)(Vector256.Sum(a1 + b1) + t1);
        y2 = (float)(Vector256.Sum(a2 + b2) + t2);
        y3 = (float)(Vector256.Sum(a3 + b3) + t3);
    }

    // WideMixedFour for one row of x, 16 values a step in four sums (WideSum's, x widened in registers).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe double WideMixedSum<TL>(byte* row, float* x, int k) where TL : struct, IWideLoad
    {
        Vector256<double> a = default, b = default, c = default, d = default;
        int p = 0;
        for (; p + 16 <= k; p += 16)
        {
            byte* at = row + 8 * p;
            a = Fma.MultiplyAdd(TL.Four(at), Avx.ConvertToVector256Double(Sse.LoadVector128(x + p)), a);
            b = Fma.MultiplyAdd(TL.Four(at + 32), Avx.ConvertToVector256Double(Sse.LoadVector128(x + p + 4)), b);
            c = Fma.MultiplyAdd(TL.Four(at + 64), Avx.ConvertToVector256Double(Sse.LoadVector128(x + p + 8)), c);
            d = Fma.MultiplyAdd(TL.Four(at + 96), Avx.ConvertToVector256Double(Sse.LoadVector128(x + p + 12)), d);
        }

        for (; p + 4 <= k; p += 4)
        {
            a = Fma.MultiplyAdd(TL.Four(row + 8 * p), Avx.ConvertToVector256Double(Sse.LoadVector128(x + p)), a);
        }

        double t = 0;
        for (; p < k; p++)
        {
            t = Math.FusedMultiplyAdd(TL.One(row + 8 * p), x[p], t);
        }

        return Vector256.Sum((a + b) + (c + d)) + t;
    }

    // Four int64 codes (at `at`, 32 bytes) as one step of the integer lane: lane b holds the four codes' byte b (the dwords'
    // low halves to lanes 0-3, their high halves to 4-7, then each 128-bit half's 4 × 4 bytes transposed), the top bytes
    // offset by 128 (unsigned: the code's sign comes back through x's code sums).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<byte> LongStep(long* at, Vector256<int> halves, Vector256<byte> transpose, Vector256<byte> bias) =>
        Avx2.Shuffle(Avx2.PermuteVar8x32(Vector256.Load((int*)at), halves).AsByte(), transpose) ^ bias;

    // sum + x's codes [4s, 4s + 4) (one dword, at `x`) by a step's bytes, in each lane (AVX-VNNI: unsigned bytes by signed).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<int> LongDot(Vector256<int> sum, Vector256<byte> step, sbyte* x) =>
        AvxVnni.MultiplyWideningAndAdd(sum, step, Vector256.Create(*(int*)x).AsSByte());

    // A block's sums for a row of x (its 8 steps against x's codes from `x`, its sums starting from the top byte's bias:
    // LongStart), times x's block scale (`xs`), added to f (each lane's plane weight applied at the end: LongPlanes).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<float> LongBlockSum(Vector256<float> f, Vector256<byte> s0, Vector256<byte> s1, Vector256<byte> s2, Vector256<byte> s3,
        Vector256<byte> s4, Vector256<byte> s5, Vector256<byte> s6, Vector256<byte> s7, sbyte* x, float* xs)
    {
        var i = LongStart(xs);
        i = LongDot(i, s0, x);
        i = LongDot(i, s1, x + 4);
        i = LongDot(i, s2, x + 8);
        i = LongDot(i, s3, x + 12);
        i = LongDot(i, s4, x + 16);
        i = LongDot(i, s5, x + 20);
        i = LongDot(i, s6, x + 24);
        i = LongDot(i, s7, x + 28);
        return Fma.MultiplyAdd(Avx.ConvertToVector256Single(i), Vector256.Create(*xs), f);
    }

    // A stored row's k int64 codes (at `row`) as the integer lane's steps (LongStep: 32 bytes a step of 4 codes) into `into`.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void LongSteps(long* row, byte* into, int k)
    {
        var halves = Vector256.Create(0, 2, 4, 6, 1, 3, 5, 7);
        var transpose = Vector256.Create((byte)0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
        var bias = Vector256.Create(0, 0, 0, 0, 0, 0, 0, unchecked((int)0x80808080)).AsByte();
        for (int p = 0; p + 4 <= k; p += 4)
        {
            LongStep(row + p, halves, transpose, bias).Store(into + p * sizeof(long));
        }
    }

    // Four rows of x (int8 codes from `x`, `stride` apart; their 2 floats a block from `xs`, `table` apart) by `count` int64
    // codes of a stored row made into steps (at `steps`, 32 bytes a step of 4 codes: LongSteps; whole blocks of 32): each
    // step loaded once and dotted with each row's 4 codes; a block's sums (a lane a byte plane, from the top byte's bias)
    // times x's block scale; each lane times its plane's weight and the lanes added at the end. (Two chains of dot
    // products a row measured no faster: 169 tokens a second on prompts against 176.)
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void LongIntegerFour(byte* steps, sbyte* x, float* xs, int stride, nint table, int count, out float y0, out float y1, out float y2, out float y3)
    {
        nint r1 = stride, r2 = 2 * (nint)stride, r3 = 3 * (nint)stride, t2 = 2 * table, t3 = 3 * table;
        Vector256<float> f0 = default, f1 = default, f2 = default, f3 = default;
        for (int b = 0; b < count / LongBlock; b++)
        {
            byte* w = steps + b * LongBlock * sizeof(long);
            sbyte* xb = x + b * LongBlock;
            float* sb = xs + b * 2;
            var i0 = LongStart(sb);
            var i1 = LongStart(sb + table);
            var i2 = LongStart(sb + t2);
            var i3 = LongStart(sb + t3);
            for (int s = 0; s < 8; s++)
            {
                var step = Vector256.Load(w + 32 * s);
                i0 = LongDot(i0, step, xb + 4 * s);
                i1 = LongDot(i1, step, xb + r1 + 4 * s);
                i2 = LongDot(i2, step, xb + r2 + 4 * s);
                i3 = LongDot(i3, step, xb + r3 + 4 * s);
            }

            f0 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i0), Vector256.Create(*sb), f0);
            f1 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i1), Vector256.Create(*(sb + table)), f1);
            f2 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i2), Vector256.Create(*(sb + t2)), f2);
            f3 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i3), Vector256.Create(*(sb + t3)), f3);
        }

        var planes = LongPlanes;
        y0 = Vector256.Sum(f0 * planes);
        y1 = Vector256.Sum(f1 * planes);
        y2 = Vector256.Sum(f2 * planes);
        y3 = Vector256.Sum(f3 * planes);
    }

    // LongIntegerFour for one row of x.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float LongIntegerOne(long* row, sbyte* x, float* xs, int k)
    {
        var halves = Vector256.Create(0, 2, 4, 6, 1, 3, 5, 7);
        var transpose = Vector256.Create((byte)0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
        var bias = Vector256.Create(0, 0, 0, 0, 0, 0, 0, unchecked((int)0x80808080)).AsByte();
        Vector256<float> f = default;
        for (int b = 0; b < k / LongBlock; b++)
        {
            long* w = row + b * LongBlock;
            f = LongBlockSum(f, LongStep(w, halves, transpose, bias), LongStep(w + 4, halves, transpose, bias), LongStep(w + 8, halves, transpose, bias),
                LongStep(w + 12, halves, transpose, bias), LongStep(w + 16, halves, transpose, bias), LongStep(w + 20, halves, transpose, bias),
                LongStep(w + 24, halves, transpose, bias), LongStep(w + 28, halves, transpose, bias), x + b * LongBlock, xs + b * 2);
        }

        return Vector256.Sum(f * LongPlanes);
    }

    // ------------------------------------------------------------------ float64 rows read where they are stored

    // A float64 weight (by its description: a float of 11 exponent and 52 mantissa bits, 64 bits a value) takes, under the
    // measured route, two lanes over its stored rows, neither of which writes the weight's values anywhere nor narrows them:
    // both multiply the stored values, read in registers as they are, with x in float64 FMAs, each output's float64 sum
    // rounded to float32 once; the decoded lane reads x widened to float64 once a call (WidenRows), the mixed lane widens
    // x's float32 values in registers as it reads them (the same FMAs in the same order: the same sums). No integer lane: the
    // values are not integer codes. The measured route keeps the faster at each shape.

    // float64 as a format read where it is stored (IRowFormat): by its description; the decoded lane timed against the mixed
    // lane; x widened to float64 for the decoded lane; tiles of up to DoubleTileRows rows of x over a group of stored rows,
    // passes over k of DoubleSpan values; no scratch (DoubleKernels).
    private readonly unsafe struct DoubleFormat : IRowFormat<DoubleFormat, DoubleKernels>
    {
        public static bool Reads(ElementType type, int k, out DoubleFormat format)
        {
            format = default;
            return type is { BlockSize: 1, BlockBits: 64, Float: { Exponent: 11, Mantissa: 52 } } && Avx2.IsSupported && Fma.IsSupported && BitConverter.IsLittleEndian;
        }

        public int First => DecodedLaneId;

        public int Second => MixedLaneId;

        public bool BothRun => true;

        public int Unmeasured => DecodedLaneId;

        public int GroupBytes => DoubleGroupBytes;

        public (int Columns, int Rows) Tile(int total, int m, int threads, nint rowBytes) => ((total + 8 * threads - 1) / (8 * threads), Math.Min(m, DoubleTileRows));

        public int Span(int k) => DoubleSpan;

        public LaneInput Prepare(int lane, float[] x, int m, int k)
        {
            if (lane != DecodedLaneId)
            {
                return new LaneInput(x, false, null, null);
            }

            var wide = ArrayPool<float>.Shared.Rent(2 * m * k);
            WidenRows(x, wide, m, k);
            return new LaneInput(wide, true, null, null);
        }

        public (int Floats, int Bytes) Scratch(int k, int group) => (0, 0);

        public DoubleKernels Kernels(int lane, int k, LanePointers x, float* floats, byte* bytes) => new(lane == MixedLaneId, x.Values, k);
    }

    // Stored bytes a group of float64 rows takes with more than one row of x (RowGroups; at most the tile's columns): the
    // tile's x passes once a group, so the larger the group the less of x streams a stored row (128 KB groups held two rows
    // of 6,144 values, under which 3-6 MB of x streamed from L3).
    private const int DoubleGroupBytes = 1 << 20;

    // Values of k four rows of x pass over a group's float64 rows at a time (RowGroups): x's four rows as float64 (16 KB) stay
    // in L1 across the group, an efficiency core's too (32 KB).
    private const int DoubleSpan = 512;

    // Rows of x a float64 tile takes at most (DoubleFormat.Tile): each stored row read from memory once a product of up to that
    // many rows.
    private const int DoubleTileRows = 128;

    // The float64 lanes as RowGroups' kernels: nothing to set up for a group of rows; x as float64 (the decoded lane: two
    // floats' room a value) or as it is (the mixed lane).
    private readonly unsafe struct DoubleKernels(bool mixed, float* x, int k) : IRowKernels
    {
        public void Group(byte* stored, nint rowBytes, int c0, int count)
        {
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Four(byte* row, int c, int i, int p0, int p1, out float y0, out float y1, out float y2, out float y3)
        {
            if (mixed)
            {
                WideMixedFour<DoubleLoad>(row + (nint)p0 * sizeof(double), x + (nint)i * k + p0, k, p1 - p0, out y0, out y1, out y2, out y3);
            }
            else
            {
                WideFour<DoubleLoad>(row + (nint)p0 * sizeof(double), (double*)x + (nint)i * k + p0, k, p1 - p0, out y0, out y1, out y2, out y3);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float One(byte* row, int c, int i) => Single(row, i);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Single(byte* row, int i) => mixed
            ? (float)WideMixedSum<DoubleLoad>(row, x + (nint)i * k, k)
            : (float)WideSum<DoubleLoad>(row, (double*)x + (nint)i * k, k);
    }

    // ------------------------------------------------------------------ 32-value blocks of 2-bit codes read where they are stored

    // A scaled block of 32 2-bit codes in value order (four a byte, the first in the lowest bits) with one float16 scale
    // (TwoBitBlocks: int2's weights by their description, two's complement codes −2..1, or unsigned ones with a bias) takes,
    // under the measured route, two lanes neither of which writes the weight's values anywhere: the decoded lane makes each
    // block's values 8 at a time in registers through a table of the block's four values (the codes' values times its
    // scale: the values the type's conversion gives, bit for bit) and multiplies them with x in float32; the integer lane
    // rounds x to int8 a block of 32 values and takes the CPU's int8 dot products of x's codes with the stored codes, eight
    // blocks' code words gathered into eight lanes and each word's four bit pairs a step, the codes' offset added through
    // x's code sums and the float scales applied eight blocks at a time. The measured route keeps the faster at each shape.

    // int2 as a format read where it is stored (IRowFormat): a 32-value block of 2-bit codes (TwoBitBlocks) by its
    // description; the decoded lane timed against the integer lane; x rounded to int8 for the integer lane
    // (RoundTwoBitRows); no scratch (TwoBitKernels: each group of blocks' scales and codes gathered as it is read).
    private readonly unsafe struct TwoBitFormat : IRowFormat<TwoBitFormat, TwoBitKernels>
    {
        private readonly TwoBitBlocks plan;

        private TwoBitFormat(TwoBitBlocks plan) => this.plan = plan;

        public static bool Reads(ElementType type, int k, out TwoBitFormat format)
        {
            format = type.Scaled is { } block && k % TwoBitBlocks.Values == 0 && TwoBitBlocks.Of(block) is { } found ? new TwoBitFormat(found) : default;
            return format.plan is not null;
        }

        public int First => DecodedLaneId;

        public int Second => IntegerLaneId;

        public bool BothRun => true;

        public int Unmeasured => IntegerLaneId;

        public int GroupBytes => TwoBitGroupBytes;

        // A group of stored rows by 32 rows of x (GroupTile). (Columns over up to 128 rows of x, a group's steps read by all
        // of them: no faster, 683 tokens a second on prompts either way.)
        public (int Columns, int Rows) Tile(int total, int m, int threads, nint rowBytes) => GroupTile(total, rowBytes, TwoBitGroupBytes);

        public int Span(int k) => k;

        public LaneInput Prepare(int lane, float[] x, int m, int k)
        {
            if (lane != IntegerLaneId)
            {
                return new LaneInput(x, false, null, null);
            }

            int groups = TwoBitBlocks.Groups(k);
            var codes = ArrayPool<sbyte>.Shared.Rent(m * groups * TwoBitBlocks.GroupCodes);
            var scales = ArrayPool<float>.Shared.Rent(m * groups * 16);
            RoundTwoBitRows(x, codes, scales, m, k, plan.Offset);
            return new LaneInput(x, false, codes, scales);
        }

        // The integer lane's steps and scales of a group's stored rows (TwoBitSteps: 72 floats' room a group of 8 blocks), as
        // floats, which the shared driver does not clear (every one is written before it is read).
        public (int Floats, int Bytes) Scratch(int k, int group) => (group * TwoBitBlocks.Groups(k) * TwoBitStepFloats, 0);

        public TwoBitKernels Kernels(int lane, int k, LanePointers x, float* floats, byte* bytes) => new(lane == IntegerLaneId, plan, k, x, floats);
    }

    // Stored bytes a group of int2 rows takes with more than one row of x (RowGroups).
    private const int TwoBitGroupBytes = 16384;

    // Floats of scratch a group of 8 blocks' steps and scales take (TwoBitSteps): 8 steps of 32 bytes, then 8 scales.
    private const int TwoBitStepFloats = 72;

    // The int2 lanes as RowGroups' kernels. The integer lane: a group's stored rows made into steps once (TwoBitSteps: each
    // group of 8 blocks' codes gathered and split into its 8 steps, its scales gathered), which every four rows of x then
    // read; a row read alone gathers each group as it reads it. The decoded lane: nothing to set up.
    private readonly unsafe struct TwoBitKernels : IRowKernels
    {
        private readonly bool integer;
        private readonly TwoBitBlocks plan;
        private readonly int k, blocks, groups;
        private readonly nint xStride, sStride;
        private readonly float* x, xs, steps;
        private readonly sbyte* xq;

        public TwoBitKernels(bool integer, TwoBitBlocks plan, int k, LanePointers input, float* steps)
        {
            this.integer = integer;
            this.plan = plan;
            this.k = k;
            this.steps = steps;
            blocks = k / TwoBitBlocks.Values;
            groups = TwoBitBlocks.Groups(k);
            xStride = (nint)groups * TwoBitBlocks.GroupCodes;
            sStride = (nint)groups * 16;
            x = input.Values;
            xq = input.Codes;
            xs = input.Scales;
        }

        public void Group(byte* stored, nint rowBytes, int c0, int count)
        {
            if (integer)
            {
                for (int c = 0; c < count; c++)
                {
                    TwoBitSteps(stored + (c0 + c) * rowBytes, plan, blocks, steps + (nint)c * groups * TwoBitStepFloats);
                }
            }
        }

        // All of k a pass (Span: k).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Four(byte* row, int c, int i, int p0, int p1, out float y0, out float y1, out float y2, out float y3)
        {
            if (integer)
            {
                TwoBitIntegerFour(steps + (nint)c * groups * TwoBitStepFloats, groups, xq + i * xStride, xs + i * sStride, xStride, sStride, out y0, out y1, out y2, out y3);
            }
            else
            {
                TwoBitDecodedFour(row, plan, blocks, x + (nint)i * k, k, out y0, out y1, out y2, out y3);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float One(byte* row, int c, int i) => Single(row, i);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Single(byte* row, int i) => integer
            ? TwoBitIntegerOne(row, plan, blocks, xq + i * xStride, xs + i * sStride)
            : TwoBitDecodedOne(row, plan, blocks, x + (nint)i * k);
    }

    // x rounded to int8 for the 2-bit integer lane, row by row (RoundTwoBitRow), over the threads where x is large.
    private static void RoundTwoBitRows(float[] x, sbyte[] codes, float[] scales, int m, int k, int offset)
    {
        int groups = TwoBitBlocks.Groups(k), rowCodes = groups * TwoBitBlocks.GroupCodes, rowScales = groups * 16;
        if (m > 1 && (long)m * k >= 1 << 16)
        {
            Parallel.For(0, m, ComputeResources.ParallelOptions, i => RoundTwoBitRow(x.AsSpan(i * k, k), codes.AsSpan(i * rowCodes, rowCodes), scales.AsSpan(i * rowScales, rowScales), offset));
            return;
        }

        for (int i = 0; i < m; i++)
        {
            RoundTwoBitRow(x.AsSpan(i * k, k), codes.AsSpan(i * rowCodes, rowCodes), scales.AsSpan(i * rowScales, rowScales), offset);
        }
    }

    // A row of x rounded to int8 a block of 32 values at a time (the block's scale amax / 127, 0 where every value is 0; its
    // codes value · 127 / amax rounded to even), laid out a group of 8 blocks at a time in the integer kernels' step order
    // (step 4h + p, lane l: block l's values 16h + 4j + p for j = 0..3, 32 bytes a step); per group 16 floats: the blocks'
    // scales at 0-7 and, as int32 bits at 8-15, the codes' offset times each block's code sum. A last group short of 8
    // blocks is padded with zero codes, scales and sums. The integer lane's product is then the product of x so rounded.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void RoundTwoBitRow(ReadOnlySpan<float> x, Span<sbyte> codes, Span<float> scales, int offset)
    {
        Span<sbyte> natural = stackalloc sbyte[TwoBitBlocks.Values];
        ref float row = ref MemoryMarshal.GetReference(x);
        int blocks = x.Length / TwoBitBlocks.Values;
        for (int g = 0; g * TwoBitBlocks.Group < blocks; g++)
        {
            int count = Math.Min(TwoBitBlocks.Group, blocks - g * TwoBitBlocks.Group);
            var into = scales.Slice(g * 16, 16);
            var sums = MemoryMarshal.Cast<float, int>(into[8..]);
            var group = codes.Slice(g * TwoBitBlocks.GroupCodes, TwoBitBlocks.GroupCodes);
            if (count < TwoBitBlocks.Group)
            {
                into.Clear();
                group.Clear();
            }

            for (int l = 0; l < count; l++)
            {
                ref float v = ref Unsafe.Add(ref row, (g * TwoBitBlocks.Group + l) * TwoBitBlocks.Values);
                var v0 = Vector256.LoadUnsafe(ref v);
                var v1 = Vector256.LoadUnsafe(ref v, 8);
                var v2 = Vector256.LoadUnsafe(ref v, 16);
                var v3 = Vector256.LoadUnsafe(ref v, 24);
                var most = Vector256.Max(Vector256.Max(Vector256.Abs(v0), Vector256.Abs(v1)), Vector256.Max(Vector256.Abs(v2), Vector256.Abs(v3)));
                var top = Vector128.Max(most.GetLower(), most.GetUpper());
                top = Vector128.Max(top, Vector128.Shuffle(top, Vector128.Create(2, 3, 0, 1)));
                top = Vector128.Max(top, Vector128.Shuffle(top, Vector128.Create(1, 0, 3, 2)));
                float amax = top.ToScalar();
                var inverse = Vector256.Create(amax > 0f ? 127f / amax : 0f);
                var q0 = Vector256.ConvertToInt32(Vector256.Round(v0 * inverse));
                var q1 = Vector256.ConvertToInt32(Vector256.Round(v1 * inverse));
                var q2 = Vector256.ConvertToInt32(Vector256.Round(v2 * inverse));
                var q3 = Vector256.ConvertToInt32(Vector256.Round(v3 * inverse));
                Vector256.Narrow(Vector256.Narrow(q0, q1), Vector256.Narrow(q2, q3)).CopyTo(natural);
                into[l] = amax / 127f;
                sums[l] = offset * Vector256.Sum((q0 + q1) + (q2 + q3));
                for (int h = 0; h < 2; h++)
                {
                    for (int p = 0; p < 4; p++)
                    {
                        int step = (4 * h + p) * 32 + 4 * l;
                        for (int j = 0; j < 4; j++)
                        {
                            group[step + j] = natural[16 * h + 4 * j + p];
                        }
                    }
                }
            }
        }
    }

    // The lanes of a group of `count` blocks (all for a whole group of 8): lane l set where l < count.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> GroupMask(int count) => Vector256.LessThan(Vector256.Create(0, 1, 2, 3, 4, 5, 6, 7), Vector256.Create(count));

    // The dword at `at` + offsets[l] in lane l, for the lanes of `mask` (0 elsewhere, nothing read there).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<int> GroupWords(byte* at, Vector256<int> offsets, Vector256<int> mask) =>
        Avx2.GatherMaskVector256(Vector256<int>.Zero, (int*)at, offsets, mask, 1);

    // f + a group's sums for a row of x: its 8 steps dotted with x's (32 bytes a step from `x`), each block's sum in its lane
    // from x's code sum times the codes' offset (`xs` + 8), then times the blocks' scales `dw` and x's (`xs`).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<float> TwoBitGroup(Vector256<float> f, Vector256<byte> s0, Vector256<byte> s1, Vector256<byte> s2, Vector256<byte> s3,
        Vector256<byte> s4, Vector256<byte> s5, Vector256<byte> s6, Vector256<byte> s7, Vector256<float> dw, sbyte* x, float* xs)
    {
        var i = Vector256.Load((int*)(xs + 8));
        i = DotAt(i, s0, x);
        i = DotAt(i, s1, x + 32);
        i = DotAt(i, s2, x + 64);
        i = DotAt(i, s3, x + 96);
        i = DotAt(i, s4, x + 128);
        i = DotAt(i, s5, x + 160);
        i = DotAt(i, s6, x + 192);
        i = DotAt(i, s7, x + 224);
        return Fma.MultiplyAdd(Avx.ConvertToVector256Single(i), dw * Vector256.Load(xs), f);
    }

    // A stored row of `blocks` 2-bit blocks made into the integer lane's steps: each group of 8 blocks' two code words
    // gathered into lane l per block (each code's high bit flipped where they are two's complement: code + 2), each word's
    // four bit pairs a step (32 bytes), the group's 8 steps then its 8 scales (gathered, float16) into `into`,
    // TwoBitStepFloats a group; a last group short of 8 blocks masked (its missing lanes 0).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void TwoBitSteps(byte* row, TwoBitBlocks plan, int blocks, float* into)
    {
        var low = Vector256.Create(0x03030303u);
        var flip = Vector256.Create(plan.Flip);
        var offsets = plan.Offsets;
        nint bytes = plan.Bytes;
        for (int g = 0; g * TwoBitBlocks.Group < blocks; g++)
        {
            var mask = GroupMask(blocks - g * TwoBitBlocks.Group);
            byte* group = row + g * TwoBitBlocks.Group * bytes;
            var c0 = GroupWords(group + plan.CodeOffset, offsets, mask).AsUInt32() ^ flip;
            var c1 = GroupWords(group + plan.CodeOffset + 4, offsets, mask).AsUInt32() ^ flip;
            byte* at = (byte*)(into + g * TwoBitStepFloats);
            (c0 & low).Store((uint*)at);
            (Vector256.ShiftRightLogical(c0, 2) & low).Store((uint*)(at + 32));
            (Vector256.ShiftRightLogical(c0, 4) & low).Store((uint*)(at + 64));
            (Vector256.ShiftRightLogical(c0, 6) & low).Store((uint*)(at + 96));
            (c1 & low).Store((uint*)(at + 128));
            (Vector256.ShiftRightLogical(c1, 2) & low).Store((uint*)(at + 160));
            (Vector256.ShiftRightLogical(c1, 4) & low).Store((uint*)(at + 192));
            (Vector256.ShiftRightLogical(c1, 6) & low).Store((uint*)(at + 224));
            HalfLanes(GroupWords(group + plan.ScaleOffset, offsets, mask)).Store((float*)(at + 256));
        }
    }

    // Four rows of x (codes from `x`, a row every `xStride`; scales and code sums from `xs`, a row every `sStride`) by one
    // stored row made into steps (TwoBitSteps, `groups` groups of 8 blocks from `steps`): each step loaded once and dotted
    // with each row's; a block's sum (its codes' offset times x's code sum to start) in its lane; then the blocks' scales
    // times x's, 8 blocks at a time.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void TwoBitIntegerFour(float* steps, int groups, sbyte* x, float* xs, nint xStride, nint sStride,
        out float y0, out float y1, out float y2, out float y3)
    {
        Vector256<float> f0 = default, f1 = default, f2 = default, f3 = default;
        for (int g = 0; g < groups; g++)
        {
            byte* at = (byte*)(steps + g * TwoBitStepFloats);
            sbyte* xg = x + g * TwoBitBlocks.GroupCodes;
            float* sg = xs + g * 16;
            var i0 = Vector256.Load((int*)(sg + 8));
            var i1 = Vector256.Load((int*)(sg + sStride + 8));
            var i2 = Vector256.Load((int*)(sg + 2 * sStride + 8));
            var i3 = Vector256.Load((int*)(sg + 3 * sStride + 8));
            for (int s = 0; s < 8; s++)
            {
                var step = Vector256.Load(at + 32 * s);
                i0 = DotAt(i0, step, xg + 32 * s);
                i1 = DotAt(i1, step, xg + xStride + 32 * s);
                i2 = DotAt(i2, step, xg + 2 * xStride + 32 * s);
                i3 = DotAt(i3, step, xg + 3 * xStride + 32 * s);
            }

            var dw = Vector256.Load((float*)(at + 256));
            f0 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i0), dw * Vector256.Load(sg), f0);
            f1 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i1), dw * Vector256.Load(sg + sStride), f1);
            f2 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i2), dw * Vector256.Load(sg + 2 * sStride), f2);
            f3 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i3), dw * Vector256.Load(sg + 3 * sStride), f3);
        }

        y0 = Vector256.Sum(f0);
        y1 = Vector256.Sum(f1);
        y2 = Vector256.Sum(f2);
        y3 = Vector256.Sum(f3);
    }

    // TwoBitIntegerFour for one row of x.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float TwoBitIntegerOne(byte* row, TwoBitBlocks plan, int blocks, sbyte* x, float* xs)
    {
        var low = Vector256.Create(0x03030303u);
        var flip = Vector256.Create(plan.Flip);
        var offsets = plan.Offsets;
        nint bytes = plan.Bytes;
        Vector256<float> f = default;
        for (int g = 0; g * TwoBitBlocks.Group < blocks; g++)
        {
            var mask = GroupMask(blocks - g * TwoBitBlocks.Group);
            byte* group = row + g * TwoBitBlocks.Group * bytes;
            var c0 = GroupWords(group + plan.CodeOffset, offsets, mask).AsUInt32() ^ flip;
            var c1 = GroupWords(group + plan.CodeOffset + 4, offsets, mask).AsUInt32() ^ flip;
            f = TwoBitGroup(f, (c0 & low).AsByte(), (Vector256.ShiftRightLogical(c0, 2) & low).AsByte(), (Vector256.ShiftRightLogical(c0, 4) & low).AsByte(),
                (Vector256.ShiftRightLogical(c0, 6) & low).AsByte(), (c1 & low).AsByte(), (Vector256.ShiftRightLogical(c1, 2) & low).AsByte(),
                (Vector256.ShiftRightLogical(c1, 4) & low).AsByte(), (Vector256.ShiftRightLogical(c1, 6) & low).AsByte(),
                HalfLanes(GroupWords(group + plan.ScaleOffset, offsets, mask)), x + g * TwoBitBlocks.GroupCodes, xs + g * 16);
        }

        return Vector256.Sum(f);
    }

    // Four rows of x (float32, k a row, from `x`) by one stored row of `blocks` 2-bit blocks: each group's 8 scales gathered
    // (float16), then each block's codes 8 at a time made float32 in registers through the block's table of its four values
    // times its scale (a 32-bit word of 16 codes shifted by each value's 2-bit place, the table lane picked by the low 3
    // bits: the table repeats every 4), each multiplied with the four rows; alternate vectors in their own sums.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void TwoBitDecodedFour(byte* row, TwoBitBlocks plan, int blocks, float* x, int k, out float y0, out float y1, out float y2, out float y3)
    {
        var first = Vector256.Create(0u, 2, 4, 6, 8, 10, 12, 14);
        var second = Vector256.Create(16u, 18, 20, 22, 24, 26, 28, 30);
        var values = plan.Table;
        var offsets = plan.Offsets;
        nint bytes = plan.Bytes, r1 = k, r2 = 2 * (nint)k, r3 = 3 * (nint)k;
        Vector256<float> a0 = default, a1 = default, a2 = default, a3 = default, b0 = default, b1 = default, b2 = default, b3 = default;
        for (int g = 0; g * TwoBitBlocks.Group < blocks; g++)
        {
            int count = Math.Min(TwoBitBlocks.Group, blocks - g * TwoBitBlocks.Group);
            byte* group = row + g * TwoBitBlocks.Group * bytes;
            var dw = HalfLanes(GroupWords(group + plan.ScaleOffset, offsets, GroupMask(count)));
            for (int l = 0; l < count; l++)
            {
                var table = values * Avx2.PermuteVar8x32(dw, Vector256.Create(l));
                byte* codes = group + l * bytes + plan.CodeOffset;
                var lo = Vector256.Create(*(uint*)codes);
                var hi = Vector256.Create(*(uint*)(codes + 4));
                var w0 = Avx2.PermuteVar8x32(table, Avx2.ShiftRightLogicalVariable(lo, first).AsInt32());
                var w1 = Avx2.PermuteVar8x32(table, Avx2.ShiftRightLogicalVariable(lo, second).AsInt32());
                var w2 = Avx2.PermuteVar8x32(table, Avx2.ShiftRightLogicalVariable(hi, first).AsInt32());
                var w3 = Avx2.PermuteVar8x32(table, Avx2.ShiftRightLogicalVariable(hi, second).AsInt32());
                float* xb = x + (g * TwoBitBlocks.Group + l) * TwoBitBlocks.Values;
                a0 = Fma.MultiplyAdd(w0, Vector256.Load(xb), a0);
                b0 = Fma.MultiplyAdd(w1, Vector256.Load(xb + 8), b0);
                a0 = Fma.MultiplyAdd(w2, Vector256.Load(xb + 16), a0);
                b0 = Fma.MultiplyAdd(w3, Vector256.Load(xb + 24), b0);
                a1 = Fma.MultiplyAdd(w0, Vector256.Load(xb + r1), a1);
                b1 = Fma.MultiplyAdd(w1, Vector256.Load(xb + r1 + 8), b1);
                a1 = Fma.MultiplyAdd(w2, Vector256.Load(xb + r1 + 16), a1);
                b1 = Fma.MultiplyAdd(w3, Vector256.Load(xb + r1 + 24), b1);
                a2 = Fma.MultiplyAdd(w0, Vector256.Load(xb + r2), a2);
                b2 = Fma.MultiplyAdd(w1, Vector256.Load(xb + r2 + 8), b2);
                a2 = Fma.MultiplyAdd(w2, Vector256.Load(xb + r2 + 16), a2);
                b2 = Fma.MultiplyAdd(w3, Vector256.Load(xb + r2 + 24), b2);
                a3 = Fma.MultiplyAdd(w0, Vector256.Load(xb + r3), a3);
                b3 = Fma.MultiplyAdd(w1, Vector256.Load(xb + r3 + 8), b3);
                a3 = Fma.MultiplyAdd(w2, Vector256.Load(xb + r3 + 16), a3);
                b3 = Fma.MultiplyAdd(w3, Vector256.Load(xb + r3 + 24), b3);
            }
        }

        y0 = Vector256.Sum(a0 + b0);
        y1 = Vector256.Sum(a1 + b1);
        y2 = Vector256.Sum(a2 + b2);
        y3 = Vector256.Sum(a3 + b3);
    }

    // TwoBitDecodedFour for one row of x, each of a block's four vectors in its own sum.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float TwoBitDecodedOne(byte* row, TwoBitBlocks plan, int blocks, float* x)
    {
        var first = Vector256.Create(0u, 2, 4, 6, 8, 10, 12, 14);
        var second = Vector256.Create(16u, 18, 20, 22, 24, 26, 28, 30);
        var values = plan.Table;
        var offsets = plan.Offsets;
        nint bytes = plan.Bytes;
        Vector256<float> a0 = default, a1 = default, a2 = default, a3 = default;
        for (int g = 0; g * TwoBitBlocks.Group < blocks; g++)
        {
            int count = Math.Min(TwoBitBlocks.Group, blocks - g * TwoBitBlocks.Group);
            byte* group = row + g * TwoBitBlocks.Group * bytes;
            var dw = HalfLanes(GroupWords(group + plan.ScaleOffset, offsets, GroupMask(count)));
            for (int l = 0; l < count; l++)
            {
                var table = values * Avx2.PermuteVar8x32(dw, Vector256.Create(l));
                byte* codes = group + l * bytes + plan.CodeOffset;
                var lo = Vector256.Create(*(uint*)codes);
                var hi = Vector256.Create(*(uint*)(codes + 4));
                float* xb = x + (g * TwoBitBlocks.Group + l) * TwoBitBlocks.Values;
                a0 = Fma.MultiplyAdd(Avx2.PermuteVar8x32(table, Avx2.ShiftRightLogicalVariable(lo, first).AsInt32()), Vector256.Load(xb), a0);
                a1 = Fma.MultiplyAdd(Avx2.PermuteVar8x32(table, Avx2.ShiftRightLogicalVariable(lo, second).AsInt32()), Vector256.Load(xb + 8), a1);
                a2 = Fma.MultiplyAdd(Avx2.PermuteVar8x32(table, Avx2.ShiftRightLogicalVariable(hi, first).AsInt32()), Vector256.Load(xb + 16), a2);
                a3 = Fma.MultiplyAdd(Avx2.PermuteVar8x32(table, Avx2.ShiftRightLogicalVariable(hi, second).AsInt32()), Vector256.Load(xb + 24), a3);
            }
        }

        return Vector256.Sum((a0 + a1) + (a2 + a3));
    }

    /// <summary>
    /// A scaled-block format the 2-bit lanes read where it is stored, by its description alone (never its name): 32 values a
    /// block, 2-bit codes in value order (four a byte, the first in the lowest bits), two's complement or unsigned with a
    /// bias, no code table, one float16 scale a block (int2). Null for any other description, or where AVX2 and FMA do not
    /// run.
    /// </summary>
    private sealed class TwoBitBlocks
    {
        /// <summary>Values a block holds.</summary>
        public const int Values = 32;

        /// <summary>Blocks the kernels take at a time (a lane each).</summary>
        public const int Group = 8;

        /// <summary>x's int8 codes a group takes in the integer lane (8 steps of 32).</summary>
        public const int GroupCodes = Group * Values;

        private static readonly ConditionalWeakTable<ScaledBlock, object> Plans = new();

        private TwoBitBlocks(ScaledBlock b)
        {
            Bytes = b.Bytes;
            CodeOffset = b.CodeOffset;
            ScaleOffset = b.ScaleOffset;
            Flip = b.SignedCodes ? 0xAAAAAAAAu : 0u;
            Offset = b.SignedCodes ? b.CodeBias - 2 : b.CodeBias;
            Offsets = Vector256.Create(0, Bytes, 2 * Bytes, 3 * Bytes, 4 * Bytes, 5 * Bytes, 6 * Bytes, 7 * Bytes);
            Span<float> table = stackalloc float[8];
            for (int u = 0; u < 8; u++)
            {
                int code = u & 3;
                table[u] = (b.SignedCodes ? (code ^ 2) - 2 : code) + b.CodeBias;
            }

            Table = Vector256.Create<float>(table);
        }

        /// <summary>Bytes a block takes.</summary>
        public int Bytes { get; }

        /// <summary>The first code byte in a block.</summary>
        public int CodeOffset { get; }

        /// <summary>The scale's first byte in a block.</summary>
        public int ScaleOffset { get; }

        /// <summary>XORed into a word of codes for the integer lane: each code's high bit flipped where they are two's complement (code + 2), else nothing.</summary>
        public uint Flip { get; }

        /// <summary>Added to a code so flipped to give its value before the scale.</summary>
        public int Offset { get; }

        /// <summary>Block l's first byte from a group's, in lane l.</summary>
        public Vector256<int> Offsets { get; }

        /// <summary>The value (before the scale) of the code in a lane index's low two bits, lanes 4-7 repeating 0-3.</summary>
        public Vector256<float> Table { get; }

        /// <summary>Groups of 8 blocks a row of <paramref name="k"/> values takes (the last may be short).</summary>
        public static int Groups(int k) => (k / Values + Group - 1) / Group;

        /// <summary>The plan for <paramref name="block"/>, or null where the lanes do not read it.</summary>
        public static TwoBitBlocks? Of(ScaledBlock block) => Plans.GetValue(block, static b => (object?)Make(b) ?? b) as TwoBitBlocks;

        private static TwoBitBlocks? Make(ScaledBlock b)
        {
            bool reads = Avx2.IsSupported && Fma.IsSupported && BitConverter.IsLittleEndian
                && b is { Values: Values, CodeBits: 2, NibblesSplit: false, CodeTable: null, Scale: { Exponent: 5, Mantissa: 10 }, CodeBias: >= -64 and <= 64 }
                && b.CodeOffset >= 0 && b.CodeOffset + 8 <= b.Bytes && b.ScaleOffset >= 0 && b.ScaleOffset + 4 <= b.Bytes;
            return reads ? new TwoBitBlocks(b) : null;
        }
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

    // ------------------------------------------------------------------ 6-bit super-blocks read where they are stored

    // A super-block of 6-bit codes (SixBlocks: q6_k by its description) takes, under the measured route, two lanes neither
    // of which writes the weight's values anywhere, whole or a panel: the decoded lane joins each value's two code planes in
    // registers as its dot products read them ((code + bias) · scale: the values SuperBlock.Decode gives, bit for bit) and
    // multiplies them with x in float32; the integer lane rounds x to int8 a block of 32 values (one scale a block) and
    // takes the CPU's int8 dot products of those with the joined codes, a half block's four 32-byte code vectors transposed
    // by words so each sub-block's sum lands in its own lane, the code bias added through x's code sums and the float scales
    // applied a half block at a time. The measured route keeps the faster at each shape.

    // q6_k as a format read where it is stored (IRowFormat): a 6-bit super-block (SixBlocks) by its description; the decoded
    // lane timed against the integer lane (the decoded alone where the CPU has no int8 dot products); x rounded to int8 for
    // the integer lane (RoundSixRows); a group's scales decoded into a table for its kernels (SixKernels).
    private readonly unsafe struct SixFormat : IRowFormat<SixFormat, SixKernels>
    {
        private readonly SixBlocks blocks;

        private SixFormat(SixBlocks blocks) => this.blocks = blocks;

        public static bool Reads(ElementType type, int k, out SixFormat format)
        {
            format = type.Super is { } super && k % SixBlocks.Values == 0 && SixBlocks.Of(super) is { } plan ? new SixFormat(plan) : default;
            return format.blocks is not null;
        }

        public int First => DecodedLaneId;

        public int Second => IntegerLaneId;

        public bool BothRun => IntegerDots;

        public int Unmeasured => IntegerLaneId;

        public int GroupBytes => SixGroupBytes;

        // Columns over every row of x, an eighth of a thread's share (measured 2-3% faster on prompts than a quarter, and than
        // a twelfth or a sixteenth): each stored row's scales decoded once a product.
        public (int Columns, int Rows) Tile(int total, int m, int threads, nint rowBytes) => ((total + 8 * threads - 1) / (8 * threads), m);

        public int Span(int k) => k;

        public LaneInput Prepare(int lane, float[] x, int m, int k)
        {
            if (lane != IntegerLaneId)
            {
                return new LaneInput(x, false, null, null);
            }

            var codes = ArrayPool<sbyte>.Shared.Rent(m * k);
            var scales = ArrayPool<float>.Shared.Rent(m * (k / SixBlocks.Half) * 16);
            RoundSixRows(x, codes, scales, m, k, blocks.CodeBias);
            return new LaneInput(x, false, codes, scales);
        }

        // A group's scales' table; a row read alone decodes each block's scales as it reads the block (no table).
        public (int Floats, int Bytes) Scratch(int k, int group) => (group * (k / SixBlocks.Values) * 16, 0);

        public SixKernels Kernels(int lane, int k, LanePointers x, float* floats, byte* bytes) => new(lane == IntegerLaneId, blocks, k, x, floats);
    }

    // Stored bytes a group of q6_k rows takes with more than one row of x (RowGroups): the rows whose scales' table fills
    // 32 KB (16 floats a super-block of 210 bytes).
    private const int SixGroupBytes = 107520;

    // Bytes a row read alone asks the cache for ahead of the block it reads (SixAhead: measured 1 KB best of 512 B to 2 KB,
    // into L1 as fast as into L2 and faster than past the caches). A prefetch never faults, past the weight too.
    private const int SixAheadBytes = 1024;

    // q6_k's lanes as RowGroups' kernels: a group's scales decoded once into `table` (SixBlocks.Scales, 16 floats a
    // super-block); a row read alone decodes each block's scales as its codes are read (no table).
    private readonly unsafe struct SixKernels : IRowKernels
    {
        private readonly bool integer;
        private readonly SixBlocks blocks;
        private readonly int k, superBlocks, rowTable, xsRow;
        private readonly float* x, xs, table;
        private readonly sbyte* xq;

        public SixKernels(bool integer, SixBlocks blocks, int k, LanePointers input, float* table)
        {
            this.integer = integer;
            this.blocks = blocks;
            this.k = k;
            this.table = table;
            superBlocks = k / SixBlocks.Values;
            rowTable = superBlocks * 16;
            xsRow = k / SixBlocks.Half * 16;
            x = input.Values;
            xq = input.Codes;
            xs = input.Scales;
        }

        public void Group(byte* stored, nint rowBytes, int c0, int count)
        {
            for (int c = 0; c < count; c++)
            {
                byte* row = stored + (c0 + c) * rowBytes;
                for (int b = 0; b < superBlocks; b++)
                {
                    blocks.Scales(row + b * blocks.Bytes, table + c * rowTable + b * 16);
                }
            }
        }

        // All of k a pass (Span: k).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Four(byte* row, int c, int i, int p0, int p1, out float y0, out float y1, out float y2, out float y3)
        {
            if (integer)
            {
                SixIntegerFour(row, blocks, superBlocks, table + c * rowTable, xq + (nint)i * k, xs + (nint)i * xsRow, k, xsRow, out y0, out y1, out y2, out y3);
            }
            else
            {
                SixDecodedFour(row, blocks, superBlocks, table + c * rowTable, x + (nint)i * k, k, out y0, out y1, out y2, out y3);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float One(byte* row, int c, int i) => integer
            ? SixIntegerOne(row, blocks, superBlocks, table + c * rowTable, xq + (nint)i * k, xs + (nint)i * xsRow)
            : SixDecodedOne(row, blocks, superBlocks, table + c * rowTable, x + (nint)i * k);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Single(byte* row, int i) => integer
            ? SixIntegerOne(row, blocks, superBlocks, null, xq + (nint)i * k, xs + (nint)i * xsRow)
            : SixDecodedOne(row, blocks, superBlocks, null, x + (nint)i * k);
    }

    // x rounded to int8 for the 6-bit integer lane, row by row (RoundSixRow), over the threads where x is large.
    private static void RoundSixRows(float[] x, sbyte[] codes, float[] scales, int m, int k, int bias)
    {
        int table = k / SixBlocks.Half * 16;
        if (m > 1 && (long)m * k >= 1 << 16)
        {
            Parallel.For(0, m, ComputeResources.ParallelOptions, i => RoundSixRow(x.AsSpan(i * k, k), codes.AsSpan(i * k, k), scales.AsSpan(i * table, table), bias));
            return;
        }

        for (int i = 0; i < m; i++)
        {
            RoundSixRow(x.AsSpan(i * k, k), codes.AsSpan(i * k, k), scales.AsSpan(i * table, table), bias);
        }
    }

    // A row of x rounded to int8 a block of 32 values at a time: the block's scale amax / 127 (0 where every value is 0), its
    // codes value · 127 / amax rounded to even; laid out a half super-block (128 values) at a time in the kernels' step order
    // (step s, lane l: the half's values 32 (l mod 4) + 16 (l / 4) + 4s .. + 3, 32 bytes a step); per half 16 floats: at l
    // lane l's block scale, at 8 + l (as int32 bits) the code bias times the codes' sum over lane l's 16 values. The
    // integer lane's product is then the product of x so rounded.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void RoundSixRow(ReadOnlySpan<float> x, Span<sbyte> codes, Span<float> scales, int bias)
    {
        Span<int> natural = stackalloc int[8];
        ref float row = ref MemoryMarshal.GetReference(x);
        var words = MemoryMarshal.Cast<sbyte, int>(codes);
        for (int half = 0; half < x.Length / SixBlocks.Half; half++)
        {
            var into = scales.Slice(half * 16, 16);
            var sums = MemoryMarshal.Cast<float, int>(into[8..]);
            for (int g = 0; g < 4; g++)
            {
                ref float v = ref Unsafe.Add(ref row, half * SixBlocks.Half + g * 32);
                var v0 = Vector256.LoadUnsafe(ref v);
                var v1 = Vector256.LoadUnsafe(ref v, 8);
                var v2 = Vector256.LoadUnsafe(ref v, 16);
                var v3 = Vector256.LoadUnsafe(ref v, 24);
                var most = Vector256.Max(Vector256.Max(Vector256.Abs(v0), Vector256.Abs(v1)), Vector256.Max(Vector256.Abs(v2), Vector256.Abs(v3)));
                var top = Vector128.Max(most.GetLower(), most.GetUpper());
                top = Vector128.Max(top, Vector128.Shuffle(top, Vector128.Create(2, 3, 0, 1)));
                top = Vector128.Max(top, Vector128.Shuffle(top, Vector128.Create(1, 0, 3, 2)));
                float amax = top.ToScalar();
                var inverse = Vector256.Create(amax > 0f ? 127f / amax : 0f);
                var q0 = Vector256.ConvertToInt32(Vector256.Round(v0 * inverse));
                var q1 = Vector256.ConvertToInt32(Vector256.Round(v1 * inverse));
                var q2 = Vector256.ConvertToInt32(Vector256.Round(v2 * inverse));
                var q3 = Vector256.ConvertToInt32(Vector256.Round(v3 * inverse));
                Vector256.Narrow(Vector256.Narrow(q0, q1), Vector256.Narrow(q2, q3)).AsInt32().CopyTo(natural);

                // Block g's values 0-15 are lane g's, 16-31 lane g + 4's; its word w goes to step w mod 4.
                into[g] = into[g + 4] = amax / 127f;
                sums[g] = bias * Vector256.Sum(q0 + q1);
                sums[g + 4] = bias * Vector256.Sum(q2 + q3);
                for (int w = 0; w < 8; w++)
                {
                    words[half * 32 + 8 * (w % 4) + g + 4 * (w / 4)] = natural[w];
                }
            }
        }
    }

    // A half block's codes (its 64 low-plane bytes from `low`, 32 high-plane bytes from `high`) joined to 6-bit codes as four
    // 32-byte vectors of the half's values 0-31, 32-63, 64-95 and 96-127 in order, then transposed by words in each 128-bit
    // half: step s's lane l holds four codes of sub-block (0, 2, 4, 6, 1, 3, 5, 7)[l] (values 32 (l mod 4) + 16 (l / 4) + 4s
    // .. + 3), as RoundSixRow lays out x.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void SixSteps(byte* low, byte* high, out Vector256<byte> s0, out Vector256<byte> s1, out Vector256<byte> s2, out Vector256<byte> s3)
    {
        var nibble = Vector256.Create((byte)15);
        var top = Vector256.Create((byte)48);
        var a = Vector256.Load(low);
        var b = Vector256.Load(low + 32);
        var h = Vector256.Load(high).AsUInt16();
        var c0 = (a & nibble) | (Vector256.ShiftLeft(h, 4).AsByte() & top);
        var c1 = (b & nibble) | (Vector256.ShiftLeft(h, 2).AsByte() & top);
        var c2 = (Vector256.ShiftRightLogical(a.AsUInt16(), 4).AsByte() & nibble) | (h.AsByte() & top);
        var c3 = (Vector256.ShiftRightLogical(b.AsUInt16(), 4).AsByte() & nibble) | (Vector256.ShiftRightLogical(h, 2).AsByte() & top);
        var u0 = Avx2.UnpackLow(c0.AsUInt32(), c1.AsUInt32()).AsUInt64();
        var u1 = Avx2.UnpackHigh(c0.AsUInt32(), c1.AsUInt32()).AsUInt64();
        var u2 = Avx2.UnpackLow(c2.AsUInt32(), c3.AsUInt32()).AsUInt64();
        var u3 = Avx2.UnpackHigh(c2.AsUInt32(), c3.AsUInt32()).AsUInt64();
        s0 = Avx2.UnpackLow(u0, u2).AsByte();
        s1 = Avx2.UnpackHigh(u0, u2).AsByte();
        s2 = Avx2.UnpackLow(u1, u3).AsByte();
        s3 = Avx2.UnpackHigh(u1, u3).AsByte();
    }

    // f + a half block's sums for a row of x: its four steps of codes dotted with x's (32 bytes a step from `x`), each
    // sub-block's sum in its lane from the code bias times x's code sum (`xs` + 8), then times the sub-blocks' scales `w` and
    // x's block scales (`xs`).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<float> SixHalf(Vector256<float> f, Vector256<byte> t0, Vector256<byte> t1, Vector256<byte> t2, Vector256<byte> t3, Vector256<float> w,
        sbyte* x, float* xs)
    {
        var i = Vector256.Load((int*)(xs + 8));
        i = DotAt(i, t0, x);
        i = DotAt(i, t1, x + 32);
        i = DotAt(i, t2, x + 64);
        i = DotAt(i, t3, x + 96);
        return Fma.MultiplyAdd(Avx.ConvertToVector256Single(i), w * Vector256.Load(xs), f);
    }

    // Four rows of x (int8 codes from `x`, k a row; their scales and code sums from `xs`, `xsRow` a row) by one stored row of
    // `blocks` super-blocks (from `row`; their scales from `table`, 16 a block in lane order): each half block's codes joined
    // and transposed once (SixSteps), then dotted with each row's.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void SixIntegerFour(byte* row, SixBlocks plan, int blocks, float* table, sbyte* x, float* xs, int k, int xsRow,
        out float y0, out float y1, out float y2, out float y3)
    {
        nint bytes = plan.Bytes, lowOffset = plan.LowOffset, highOffset = plan.HighOffset, r1 = k, r2 = 2 * (nint)k, r3 = 3 * (nint)k;
        nint s1 = xsRow, s2 = 2 * (nint)xsRow, s3 = 3 * (nint)xsRow;
        Vector256<float> f0 = default, f1 = default, f2 = default, f3 = default;
        for (int b = 0; b < blocks; b++)
        {
            byte* block = row + b * bytes;
            for (int h = 0; h < 2; h++)
            {
                SixSteps(block + lowOffset + 64 * h, block + highOffset + 32 * h, out var t0, out var t1, out var t2, out var t3);
                var w = Vector256.Load(table + b * 16 + 8 * h);
                sbyte* xh = x + b * SixBlocks.Values + SixBlocks.Half * h;
                float* sh = xs + (2 * b + h) * 16;
                f0 = SixHalf(f0, t0, t1, t2, t3, w, xh, sh);
                f1 = SixHalf(f1, t0, t1, t2, t3, w, xh + r1, sh + s1);
                f2 = SixHalf(f2, t0, t1, t2, t3, w, xh + r2, sh + s2);
                f3 = SixHalf(f3, t0, t1, t2, t3, w, xh + r3, sh + s3);
            }
        }

        y0 = Vector256.Sum(f0);
        y1 = Vector256.Sum(f1);
        y2 = Vector256.Sum(f2);
        y3 = Vector256.Sum(f3);
    }

    // The 256 bytes SixAheadBytes past a block asked into the cache (a row read alone streams from memory: measured 7% more
    // tokens a second decoding).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void SixAhead(byte* block)
    {
        byte* ahead = block + SixAheadBytes;
        Sse.Prefetch0(ahead);
        Sse.Prefetch0(ahead + 64);
        Sse.Prefetch0(ahead + 128);
        Sse.Prefetch0(ahead + 192);
    }

    // SixIntegerFour for one row of x; each block's scales from `table`, or, where it is null (a row read alone), decoded
    // from the block as it is read (SixBlocks.Scales: no table), the bytes ahead asked into the cache (SixAhead).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float SixIntegerOne(byte* row, SixBlocks plan, int blocks, float* table, sbyte* x, float* xs)
    {
        nint bytes = plan.Bytes, lowOffset = plan.LowOffset, highOffset = plan.HighOffset;
        float* own = stackalloc float[16];
        Vector256<float> f = default;
        for (int b = 0; b < blocks; b++)
        {
            byte* block = row + b * bytes;
            float* scales = table + b * 16;
            if (table is null)
            {
                SixAhead(block);
                plan.Scales(block, own);
                scales = own;
            }

            for (int h = 0; h < 2; h++)
            {
                SixSteps(block + lowOffset + 64 * h, block + highOffset + 32 * h, out var t0, out var t1, out var t2, out var t3);
                f = SixHalf(f, t0, t1, t2, t3, Vector256.Load(scales + 8 * h), x + b * SixBlocks.Values + SixBlocks.Half * h, xs + (2 * b + h) * 16);
            }
        }

        return Vector256.Sum(f);
    }

    // ---- the decoded lane

    // 8 values of each of a half block's four runs of 32 (values 32t + p .. + 7 for run t; the low plane's bytes at `low` + p
    // and + 32 + p, the high plane's at `high` + p) joined in registers to code + bias, as floats (exact).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void SixRuns(byte* low, byte* high, int p, Vector256<int> bias, out Vector256<float> w0, out Vector256<float> w1, out Vector256<float> w2, out Vector256<float> w3)
    {
        var nibble = Vector256.Create(15);
        var top = Vector256.Create(48);
        var a = Avx2.ConvertToVector256Int32(low + p);
        var b = Avx2.ConvertToVector256Int32(low + 32 + p);
        var h = Avx2.ConvertToVector256Int32(high + p);
        w0 = Avx.ConvertToVector256Single(((a & nibble) | (Vector256.ShiftLeft(h, 4) & top)) + bias);
        w1 = Avx.ConvertToVector256Single(((b & nibble) | (Vector256.ShiftLeft(h, 2) & top)) + bias);
        w2 = Avx.ConvertToVector256Single((Vector256.ShiftRightLogical(a, 4) | (h & top)) + bias);
        w3 = Avx.ConvertToVector256Single((Vector256.ShiftRightLogical(b, 4) | (Vector256.ShiftRightLogical(h, 2) & top)) + bias);
    }

    // Four rows of x (float32, k a row, from `x`) by one stored row of `blocks` super-blocks (from `row`; their scales from
    // `table`, 16 a block in lane order): 8 values of each of a half block's four runs at a time (SixRuns), each times its
    // sub-block's scale (2t + p / 16, in lane 4 (p / 16) + t of the half: SuperBlock.Decode's values, the code and bias
    // summed as integers, then one product) and multiplied with the four rows; the even and odd runs in their own sums.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void SixDecodedFour(byte* row, SixBlocks plan, int blocks, float* table, float* x, int k,
        out float y0, out float y1, out float y2, out float y3)
    {
        var bias = Vector256.Create(plan.CodeBias);
        nint bytes = plan.Bytes, lowOffset = plan.LowOffset, highOffset = plan.HighOffset, r1 = k, r2 = 2 * (nint)k, r3 = 3 * (nint)k;
        Vector256<float> a0 = default, a1 = default, a2 = default, a3 = default, b0 = default, b1 = default, b2 = default, b3 = default;
        for (int b = 0; b < blocks; b++)
        {
            byte* block = row + b * bytes;
            for (int h = 0; h < 2; h++)
            {
                byte* low = block + lowOffset + 64 * h, high = block + highOffset + 32 * h;
                float* scales = table + b * 16 + 8 * h, xh = x + b * SixBlocks.Values + SixBlocks.Half * h;
                for (int p = 0; p < 32; p += 8)
                {
                    SixRuns(low, high, p, bias, out var w0, out var w1, out var w2, out var w3);
                    float* s = scales + 4 * (p >> 4);
                    w0 *= Vector256.Create(s[0]);
                    w1 *= Vector256.Create(s[1]);
                    w2 *= Vector256.Create(s[2]);
                    w3 *= Vector256.Create(s[3]);
                    float* xp = xh + p;
                    a0 = Fma.MultiplyAdd(w0, Vector256.Load(xp), a0);
                    b0 = Fma.MultiplyAdd(w1, Vector256.Load(xp + 32), b0);
                    a0 = Fma.MultiplyAdd(w2, Vector256.Load(xp + 64), a0);
                    b0 = Fma.MultiplyAdd(w3, Vector256.Load(xp + 96), b0);
                    a1 = Fma.MultiplyAdd(w0, Vector256.Load(xp + r1), a1);
                    b1 = Fma.MultiplyAdd(w1, Vector256.Load(xp + r1 + 32), b1);
                    a1 = Fma.MultiplyAdd(w2, Vector256.Load(xp + r1 + 64), a1);
                    b1 = Fma.MultiplyAdd(w3, Vector256.Load(xp + r1 + 96), b1);
                    a2 = Fma.MultiplyAdd(w0, Vector256.Load(xp + r2), a2);
                    b2 = Fma.MultiplyAdd(w1, Vector256.Load(xp + r2 + 32), b2);
                    a2 = Fma.MultiplyAdd(w2, Vector256.Load(xp + r2 + 64), a2);
                    b2 = Fma.MultiplyAdd(w3, Vector256.Load(xp + r2 + 96), b2);
                    a3 = Fma.MultiplyAdd(w0, Vector256.Load(xp + r3), a3);
                    b3 = Fma.MultiplyAdd(w1, Vector256.Load(xp + r3 + 32), b3);
                    a3 = Fma.MultiplyAdd(w2, Vector256.Load(xp + r3 + 64), a3);
                    b3 = Fma.MultiplyAdd(w3, Vector256.Load(xp + r3 + 96), b3);
                }
            }
        }

        y0 = Vector256.Sum(a0 + b0);
        y1 = Vector256.Sum(a1 + b1);
        y2 = Vector256.Sum(a2 + b2);
        y3 = Vector256.Sum(a3 + b3);
    }

    // SixDecodedFour for one row of x, each run in its own sum; the scales as SixIntegerOne takes them.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float SixDecodedOne(byte* row, SixBlocks plan, int blocks, float* table, float* x)
    {
        var bias = Vector256.Create(plan.CodeBias);
        nint bytes = plan.Bytes, lowOffset = plan.LowOffset, highOffset = plan.HighOffset;
        float* own = stackalloc float[16];
        Vector256<float> a0 = default, a1 = default, a2 = default, a3 = default;
        for (int b = 0; b < blocks; b++)
        {
            byte* block = row + b * bytes;
            float* scales = table + b * 16;
            if (table is null)
            {
                SixAhead(block);
                plan.Scales(block, own);
                scales = own;
            }

            for (int h = 0; h < 2; h++)
            {
                byte* low = block + lowOffset + 64 * h, high = block + highOffset + 32 * h;
                float* sh = scales + 8 * h, xh = x + b * SixBlocks.Values + SixBlocks.Half * h;
                for (int p = 0; p < 32; p += 8)
                {
                    SixRuns(low, high, p, bias, out var w0, out var w1, out var w2, out var w3);
                    float* s = sh + 4 * (p >> 4);
                    float* xp = xh + p;
                    a0 = Fma.MultiplyAdd(w0 * Vector256.Create(s[0]), Vector256.Load(xp), a0);
                    a1 = Fma.MultiplyAdd(w1 * Vector256.Create(s[1]), Vector256.Load(xp + 32), a1);
                    a2 = Fma.MultiplyAdd(w2 * Vector256.Create(s[2]), Vector256.Load(xp + 64), a2);
                    a3 = Fma.MultiplyAdd(w3 * Vector256.Create(s[3]), Vector256.Load(xp + 96), a3);
                }
            }
        }

        return Vector256.Sum((a0 + a1) + (a2 + a3));
    }

    /// <summary>
    /// A super-block format the 6-bit lanes read where it is stored, by its description alone (never its name): 256 values a
    /// block in two groups of 128, each value's code from a plane of 4 bits and one of 2 bits (a group's 64 low-plane bytes
    /// hold its values 0-63 in their low halves and 64-127 in their high halves; its 32 high-plane bytes values 0-31, 32-63,
    /// 64-95 and 96-127 in their bit pairs from the lowest), a code bias, sub-blocks of 16 values each with an 8-bit scale in
    /// 16 consecutive bytes, no minimums, the super-scale float16 or bfloat16 (ggml's Q6_K). Null for any other description,
    /// or where AVX2 and FMA do not run.
    /// </summary>
    private sealed unsafe class SixBlocks
    {
        /// <summary>Values a block holds.</summary>
        public const int Values = 256;

        /// <summary>Values a group of the code planes holds (the kernels' half block).</summary>
        public const int Half = 128;

        private static readonly ConditionalWeakTable<SuperBlock, object> Plans = new();

        private readonly int scaleBytes, superScale, subScaleBias;
        private readonly bool signed, bfloat16;

        private SixBlocks(SuperBlock s)
        {
            Bytes = s.Bytes;
            LowOffset = s.Planes[0].Offset;
            HighOffset = s.Planes[1].Offset;
            CodeBias = s.CodeBias;
            scaleBytes = s.SubScales[0][0].Offset;
            superScale = s.ScaleOffset;
            subScaleBias = s.SubScaleBias;
            signed = s.SubScalesSigned;
            bfloat16 = s.Scale.Exponent == 8;
        }

        /// <summary>Bytes a block takes.</summary>
        public int Bytes { get; }

        /// <summary>The low (4-bit) plane's first byte in a block.</summary>
        public int LowOffset { get; }

        /// <summary>The high (2-bit) plane's first byte in a block.</summary>
        public int HighOffset { get; }

        /// <summary>Added to each code before the scale.</summary>
        public int CodeBias { get; }

        /// <summary>The plan for <paramref name="super"/>, or null where the lanes do not read it.</summary>
        public static SixBlocks? Of(SuperBlock super) => Plans.GetValue(super, static s => (object?)Make(s) ?? s) as SixBlocks;

        private static SixBlocks? Make(SuperBlock s)
        {
            bool reads = Avx2.IsSupported && Fma.IsSupported && BitConverter.IsLittleEndian
                && s is { Values: Values, SubValues: 16, SubMins: null, CodeBias: >= -128 and <= 127, SubScaleBias: >= -65536 and <= 65536, SubScales.Count: 16,
                    Planes: [{ Bits: 4, Group: Half } low, { Bits: 2, Group: Half } high] }
                && s.Scale is { Exponent: 5, Mantissa: 10 } or { Exponent: 8, Mantissa: 7 }
                && low.Offset >= 0 && low.Offset + 128 <= s.Bytes && high.Offset >= 0 && high.Offset + 64 <= s.Bytes
                && s.ScaleOffset >= 0 && s.ScaleOffset + 2 <= s.Bytes
                && s.SubScales[0] is [{ Shift: 0, Bits: 8 } first] && first.Offset >= 0 && first.Offset + 16 <= s.Bytes
                && Enumerable.Range(0, 16).All(j => s.SubScales[j] is [{ Shift: 0, Bits: 8 } piece] && piece.Offset == first.Offset + j);
            return reads ? new SixBlocks(s) : null;
        }

        /// <summary>
        /// One block's 16 sub-block scales d · (scale_j + bias) (the factors <see cref="SuperBlock.Decode"/> takes, bit for
        /// bit) into <paramref name="into"/>, in lane order: half h's at 8h, lane l the scale of sub-block 8h + (0, 2, 4, 6,
        /// 1, 3, 5, 7)[l].
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Scales(byte* block, float* into)
        {
            var bytes = Ssse3.Shuffle(Vector128.Load(block + scaleBytes), Vector128.Create((byte)0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15));
            var upper = Sse2.ShiftRightLogical128BitLane(bytes, 8);
            var low = signed ? Avx2.ConvertToVector256Int32(bytes.AsSByte()) : Avx2.ConvertToVector256Int32(bytes);
            var high = signed ? Avx2.ConvertToVector256Int32(upper.AsSByte()) : Avx2.ConvertToVector256Int32(upper);
            ushort bits = *(ushort*)(block + superScale);
            var d = Vector256.Create(bfloat16 ? BitConverter.Int32BitsToSingle(bits << 16) : (float)BitConverter.UInt16BitsToHalf(bits));
            var bias = Vector256.Create(subScaleBias);
            (d * Avx.ConvertToVector256Single(low + bias)).Store(into);
            (d * Avx.ConvertToVector256Single(high + bias)).Store(into + 8);
        }
    }

    // ------------------------------------------------------------------ the described lane: any type from its description

    // One lane for every type (DescribedFormat), reading nothing but the type's description (ElementType.Description: a
    // block's values and bytes; each value's code from its bit planes, an integer, a float of its bits or a table's entry; a
    // code offset; the scales and minimums covering it, each a number of its own type), compiled once a description into a
    // plan (DescribedPlan) of what the description implies, never the type's name and no code of a type's own. The plan
    // takes the arithmetic the values allow, and the lane decodes them in registers inside the dot products, writing no
    // weight's value anywhere, whole or a panel:
    // - integer: codes of at most 8 bits whose numbers are whole multiples of one unit spanning at most 255 units (127 without
    //   AVX-VNNI), or whole numbers of 2, 4 or 8 bytes (each byte a plane of its own, AVX-VNNI): x rounded to int8 a block of
    //   32 values, each code made an unsigned byte in registers (its plane's bits shifted and masked, then its number less
    //   the least: as it is, its sign bit flipped or by a byte shuffle), 8 chunks' codes transposed into the int8 dot
    //   products' lanes (or, a chunk's codes 16 bytes or fewer, gathered straight into them: GatheredSteps), the least added
    //   back through x's code sums; the scales and minimums of the 8 chunks gathered and read by their number types at once
    //   (an integer, a table of an 8-bit type's numbers, a 16-bit float by its bits); where every span's codes load alike,
    //   the load and the map are the kernels' type parameters (UniformOne, UniformRowSteps); a wide integer's byte planes
    //   are made by its plane count's own step (IStepLoad.Plane);
    // - float32: values a float32 holds (16-bit floats by their bits, 16-bit integers, float32, codes of up to 8 bits
    //   through a table of their numbers times their scales), float32 FMAs with x;
    // - float64: 64-bit floats as stored, 32- and 64-bit integers made float64 in registers, FMAs with x in float64 (no
    //   narrowing of a 64-bit value to float32).
    // The written formats (FormatOf) stay as candidates: the measured route times them against the described lane at each
    // shape (RouteLanes.All) and keeps the faster; RouteLanes.Described takes the described lane alone.

    // How the described lane computes, by what the type's values allow (DescribedPlan.Make).
    private enum DescribedMode : byte
    {
        Integer,
        Single,
        Double,
    }

    // How the integer lane makes a code an unsigned byte: as it is, its sign bit flipped, through a 16-entry byte shuffle, or
    // through a 256-entry table a byte at a time.
    private enum CodeMap : byte
    {
        Same,
        Flip,
        Shuffle,
        Table,
    }

    // How a block's scale or minimum reads (DescribedPlan.Factor): an integer (unsigned, two's complement), through a table
    // of its numbers (8 bits or fewer), a 16-bit float by its bits, a float32 as it is, or one lane at a time by its type.
    private enum FactorRead : byte
    {
        Unsigned,
        Signed,
        Table,
        Half,
        Single,
        Scalar,
    }

    // How the float lanes make a stored value: a 16-bit float by its bits, a 16-bit integer, a float32 as it is, codes of up
    // to 8 bits through a table of their numbers (float32); a float64 as it is, a 64-bit or 32-bit integer (float64).
    private enum WideRead : byte
    {
        Half,
        Short,
        Single,
        Codes,
        Double,
        Long,
        Int,
    }

    /// <summary>
    /// What the described lane makes of a type's description (<see cref="TypeDescription"/>), once a description: the
    /// arithmetic its values allow; the layout's period (whole blocks, spans of 32 values and groups of 8 chunks); for codes
    /// of at most 8 bits, each span's runs of bytes a plane's fields fill (one load, a shift and a mask a plane) and the
    /// order they give the span's values; how a code becomes the integer lane's byte; each value of the period's place in
    /// x's layout; for each scale and minimum, the 8 lanes' bytes, shifts and masks at each group of the period and how its
    /// number reads. Null where the CPU cannot run it (AVX2 and FMA) or the description is one the lane cannot read in
    /// registers (codes wider than 8 bits that are not one plane of whole bytes a value).
    /// </summary>
    private sealed unsafe class DescribedPlan
    {
        private static readonly ConditionalWeakTable<TypeDescription, object> Plans = new();

        // The pinned arrays the kernels read through pointers, kept alive with the plan.
        private readonly List<Array> held = [];

        private DescribedPlan(TypeDescription d)
        {
            D = d;
            Values = d.Values;
            Bytes = d.Bytes;
        }

        public TypeDescription D { get; }

        /// <summary>Values a block holds.</summary>
        public int Values { get; }

        /// <summary>Bytes a block takes.</summary>
        public int Bytes { get; }

        public DescribedMode Mode { get; private set; }

        /// <summary>The float lanes' reading of a value.</summary>
        public WideRead Wide { get; private set; }

        /// <summary>Values and bytes of the layout's period: whole blocks, spans of 32 values and groups.</summary>
        public int Period { get; private set; }

        public int PeriodBytes { get; private set; }

        /// <summary>Byte planes of a code in the integer lane: 1, or a wide integer's bytes (2, 4, 8).</summary>
        public int P { get; private set; } = 1;

        /// <summary>Values a chunk takes (one scale and one lane of the dot products).</summary>
        public int C { get; private set; } = 32;

        /// <summary>Chunks a group takes (8 / P).</summary>
        public int Cg { get; private set; } = 8;

        /// <summary>The integer lane's steps a group takes (C / 4).</summary>
        public int T { get; private set; } = 8;

        /// <summary>Values a group takes (Cg · C).</summary>
        public int GroupValues { get; private set; } = 256;

        /// <summary>Groups a period takes.</summary>
        public int GroupsPerPeriod { get; private set; }

        /// <summary>The integer lane's codes are its numbers over this.</summary>
        public float Unit { get; private set; } = 1f;

        /// <summary>Whether the type has minimums (a float added a chunk: x's sums times them).</summary>
        public bool Mins { get; private set; }

        /// <summary>The floats of x's lane data a group takes in the integer lane: scales, code-sum starts, then (minimums) sums.</summary>
        public int LaneFloats { get; private set; }

        /// <summary>A wide integer's top byte plane's lanes (all bits set), where its code-sum start goes.</summary>
        public Vector256<int> TopLanes { get; private set; }

        /// <summary>One scale of one piece and no bias or minimums, read as a 16-bit float by its bits (1) or through its table (2); else 0.</summary>
        private int OneScale => oneScale;

        private int oneScale;

        // OneScale: a 16-bit float scale's reading as vectors, the unit folded in; a table scale's numbers times the unit.
        private HalfVectors oneHalf;
        private float* oneTable;

        /// <summary>Bytes a group's made steps take in the integer lane: its steps, its scales, then (minimums) its minimums.</summary>
        public int GroupStepBytes { get; private set; }

        /// <summary>Each lane's weight in the final sum (2^(8p) for byte plane p), or null: every lane 1.</summary>
        public Vector256<float>? LaneWeights { get; private set; }

        /// <summary>Code planes (codes of at most 8 bits).</summary>
        public int Planes { get; private set; }

        public CodeMap Map { get; private set; }

        public Vector256<byte> Flip { get; private set; }

        public Vector256<byte> Shuffle { get; private set; }

        /// <summary>The bytes the top byte plane of a signed wide integer is flipped by (0x80 in its lanes).</summary>
        public Vector256<byte> TopFlip { get; private set; }

        /// <summary>
        /// Where every span's codes are one plane loaded the same way (1, 2, 4 or 8 runs) and mapped by a flip or a 16-entry
        /// shuffle or not at all: the runs (the kernels then take that load and that map as type parameters); else 0.
        /// </summary>
        public int UniformRuns { get; private set; }

        /// <summary>Each span's run's first byte (UniformRuns).</summary>
        public int* RunOffsets => runOffset;

        /// <summary>Each span's run's shifts (UniformRuns).</summary>
        public Vector256<ulong>* RunShifts => runShift;

        /// <summary>The one plane's mask (UniformRuns).</summary>
        public Vector256<byte> PlaneMask0 { get; private set; }

        /// <summary>The map's vector: the flip, or the shuffle's table (UniformRuns).</summary>
        public Vector256<byte> MapVector { get; private set; }

        /// <summary>
        /// Whether a group's steps are gathered (UniformRuns of 2 or more, chunks of 32: each chunk's run is 16 bytes or fewer,
        /// so dword d of the 8 chunks' runs is one gather, and each run's shift of it a step: step q · (n / 4) + d), rather than
        /// loaded a chunk at a time and transposed.
        /// </summary>
        public bool Gathered { get; private set; }

        /// <summary>Gathered with flipped codes: each byte's fields' top bits, XORed into the gathered words (the steps then mapped as they are).</summary>
        public Vector256<uint> WordFlip { get; private set; }

        /// <summary>Gathered: per group of the period, the 8 chunks' runs' first bytes, a lane each.</summary>
        public Vector256<int>* GatherOffsets { get; private set; }

        /// <summary>Gathered: per group of the period and run, the 8 chunks' shifts of that run, a lane each.</summary>
        public Vector256<uint>* GatherShifts { get; private set; }

        /// <summary>
        /// Per span of the period, the two byte shuffles that put a block of 32 of x's codes (value order) in the span's order
        /// (from its low 16 values, from its high 16: OrderShuffles[2r], [2r + 1]), or null where every span is in value order.
        /// </summary>
        public Vector256<byte>* OrderShuffles { get; private set; }

        // The shuffles of OrderShuffles, once Order is known.
        private void Shuffles()
        {
            if (Identity)
            {
                return;
            }

            int spans = Period / 32;
            OrderShuffles = Pin<Vector256<byte>>(2 * spans);
            Span<byte> low = stackalloc byte[32], high = stackalloc byte[32];
            for (int r = 0; r < spans; r++)
            {
                for (int t = 0; t < 32; t++)
                {
                    int from = Order[32 * r + t] - 32 * r;
                    low[t] = from < 16 ? (byte)from : (byte)0x80;
                    high[t] = from >= 16 ? (byte)(from - 16) : (byte)0x80;
                }

                OrderShuffles[2 * r] = Vector256.Create<byte>(low);
                OrderShuffles[2 * r + 1] = Vector256.Create<byte>(high);
            }
        }

        /// <summary>Each byte plane's least, added through x's code sums (plane 0: the code offset; the top: −128 where signed).</summary>
        public int* PlaneLeast { get; private set; }

        /// <summary>Each value of the period's byte in x's integer layout.</summary>
        public int* XPlace { get; private set; }

        /// <summary>Each value of the period's chunk.</summary>
        public int* ChunkOf { get; private set; }

        /// <summary>Position t of span r's value (32r + t: the order its codes are made in).</summary>
        public int* Order { get; private set; }

        /// <summary>Whether every span's order is value order (the float lanes read x as it is).</summary>
        public bool Identity { get; private set; }

        /// <summary>The float lanes' table of each code's number (2^bits floats: the number plus the code offset).</summary>
        public float* CodeValues { get; private set; }

        /// <summary>The 16-bit float reading: its sign shift, magnitude mask, mantissa shift, the power of two it is times, and where its specials start (0: none).</summary>
        public (int SignShift, int Magnitude, int Shift, float Times, int Special) Half16 { get; private set; }

        /// <summary>Added to a wide value's number (the code offset).</summary>
        public double Offset { get; private set; }

        /// <summary>Whether a wide integer is signed.</summary>
        public bool Signed { get; private set; }

        // Codes of at most 8 bits: per span of the period and plane, the run's first byte, how many runs (1, 2, 4, 8; 0: read
        // one value at a time) and their shifts (per 64-bit lane, or per 32-bit lane for 8 runs); per plane its mask and low bit.
        private int* runOffset;
        private byte* runKind;
        private Vector256<ulong>* runShift;
        private Vector256<byte>* planeMask;
        private int* planeLow;
        private byte* byteTable;

        // Scales then minimums: count, scales' count, pieces at most; per factor its reading, bits, bias, pieces and table;
        // per group of the period, factor and piece, the lanes' offsets, shifts, masks and lefts; per group, factor and lane
        // the block and index (one lane at a time).
        private int factors, scaleFactors, pieces;
        private FactorRead[] reads = [];
        private int[] factorBits = [], factorPieces = [];
        private float[] factorBias = [];
        private float*[] factorTables = null!;
        private (int SignShift, int Magnitude, int Shift, float Times, int Special)[] factorHalves = [];
        private Vector256<int>* fOffset;
        private Vector256<int>* fShift;
        private Vector256<int>* fMask;
        private Vector256<int>* fLeft;
        private int[] scalarBlock = [], scalarIndex = [];

        /// <summary>The plan of <paramref name="d"/>, or null where the lane does not read it here.</summary>
        public static DescribedPlan? Of(TypeDescription d) => Plans.GetValue(d, static x => (object?)Make(x) ?? x) as DescribedPlan;

        /// <summary>Groups a row of <paramref name="k"/> values takes (the last may be short).</summary>
        public int Groups(int k) => (k + GroupValues - 1) / GroupValues;

        // A pinned array of n T's, held by the plan.
        private T* Pin<T>(int n) where T : unmanaged
        {
            var array = GC.AllocateArray<T>(Math.Max(1, n), pinned: true);
            held.Add(array);
            return (T*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(array));
        }

        private static DescribedPlan? Make(TypeDescription d)
        {
            if (!(Avx2.IsSupported && Fma.IsSupported && BitConverter.IsLittleEndian && d.Valid))
            {
                return null;
            }

            var plan = new DescribedPlan(d);
            if (!(plan.Wide64() || plan.Narrow()))
            {
                return null;
            }

            plan.Finish();
            return plan;
        }

        // A code of one plane of 2, 4 or 8 whole bytes a value, no scales: the integer lane over its byte planes where the CPU
        // has AVX-VNNI and the code is an integer, else a float lane.
        private bool Wide64()
        {
            var d = D;
            if (d is not { Values: 1, Planes: [{ Offset: 0, Group: 1, Bits: 16 or 32 or 64 } plane], Scales: null or [], Mins: null or [] } || d.Bytes != plane.Bits / 8)
            {
                return false;
            }

            var code = d.Code;
            int bytes = plane.Bits / 8;
            Offset = d.CodeOffset;
            Signed = code.Signed;
            Period = 32;
            PeriodBytes = 32 * bytes;
            if (code.Kind == NumberKind.Integer && AvxVnni.IsSupported && d.CodeOffset == 0)
            {
                Mode = DescribedMode.Integer;
                P = bytes;
                C = 32;
                Cg = 8 / P;
                T = 8;
                GroupValues = 32 * Cg;
                Period = GroupValues;
                PeriodBytes = GroupValues * bytes;
                PlaneLeast = Pin<int>(P);
                PlaneLeast[P - 1] = code.Signed ? -128 : 0;
                Span<float> weights = stackalloc float[8];
                Span<byte> flip = stackalloc byte[32];
                Span<int> top = stackalloc int[8];
                for (int l = 0; l < 8; l++)
                {
                    weights[l] = MathF.ScaleB(1f, 8 * (l % P));
                    top[l] = l % P == P - 1 ? -1 : 0;
                    if (code.Signed && l % P == P - 1)
                    {
                        flip[4 * l] = flip[4 * l + 1] = flip[4 * l + 2] = flip[4 * l + 3] = 0x80;
                    }
                }

                LaneWeights = Vector256.Create<float>(weights);
                TopFlip = Vector256.Create<byte>(flip);
                TopLanes = Vector256.Create<int>(top);
                Identity = true;
                Places();
                return true;
            }

            switch (code.Kind, plane.Bits)
            {
                case (NumberKind.Float, 16) when HalfOf(code) is { } half:
                    Mode = DescribedMode.Single;
                    Wide = WideRead.Half;
                    Half16 = half;
                    return true;
                case (NumberKind.Integer, 16):
                    Mode = DescribedMode.Single;
                    Wide = WideRead.Short;
                    return true;
                case (NumberKind.Float, 32) when code is { Exponent: 8, Mantissa: 23, ExponentBias: 127, Signed: true, Specials: FloatSpecials.Ieee, Subnormals: true }:
                    Mode = DescribedMode.Single;
                    Wide = WideRead.Single;
                    return true;
                case (NumberKind.Float, 64) when code is { Exponent: 11, Mantissa: 52, ExponentBias: 1023, Signed: true, Specials: FloatSpecials.Ieee, Subnormals: true }:
                    Mode = DescribedMode.Double;
                    Wide = WideRead.Double;
                    return true;
                case (NumberKind.Integer, 64) when code.Signed:
                    Mode = DescribedMode.Double;
                    Wide = WideRead.Long;
                    return true;
                case (NumberKind.Integer, 32) when code.Signed:
                    Mode = DescribedMode.Double;
                    Wide = WideRead.Int;
                    return true;
                default:
                    return false;
            }
        }

        // A 16-bit float read by its bits (a float32's exponent range or less, subnormals, IEEE's or no specials): the sign's
        // shift to bit 31, the magnitude's mask, the shift putting its exponent at float32's, the power of two that rebiases it
        // (1: none), and the magnitude its specials start at (0: none: they come through as they are, or there are none).
        private static (int SignShift, int Magnitude, int Shift, float Times, int Special)? HalfOf(NumberType t)
        {
            if (t is not { Kind: NumberKind.Float, Bits: 16, Subnormals: true, Specials: FloatSpecials.Ieee or FloatSpecials.None } || t.Exponent > 8
                || t.ExponentBias is <= 0 or >= 254 || t.Exponent == 8 && t.ExponentBias != 127)
            {
                return null;
            }

            int magnitude = t.Signed ? 0x7FFF : 0xFFFF;
            int special = t.Exponent < 8 && t.Specials == FloatSpecials.Ieee ? ((1 << t.Exponent) - 1) << t.Mantissa : 0;
            return (t.Signed ? 16 : 0, magnitude, 23 - t.Mantissa, MathF.ScaleB(1f, 127 - t.ExponentBias), special);
        }

        // Codes of at most 8 bits, from planes of 1, 2, 4 or 8 bits: each span's runs and order, the chunk (one scale a lane)
        // the factors allow, the integer lane where the codes' numbers are whole multiples of a unit within a byte's span, else
        // the float32 lane through a table of the codes' numbers.
        private bool Narrow()
        {
            var d = D;
            if (d.Code.Bits > 8 || d.Planes.Any(p => p.Bits is not (1 or 2 or 4 or 8)))
            {
                return false;
            }

            var all = (d.Scales ?? []).Concat(d.Mins ?? []).ToList();
            int chunk = all.All(f => f.Values % 32 == 0) ? 32 : all.All(f => f.Values % 16 == 0) ? 16 : all.All(f => f.Values % 8 == 0) ? 8 : 0;
            if (chunk == 0)
            {
                return false;
            }

            // The codes' numbers, and the unit that makes them whole within a byte's span.
            int count = 1 << d.Code.Bits;
            var numbers = new double[count];
            for (int c = 0; c < count; c++)
            {
                numbers[c] = d.Code.Value((ulong)c) + d.CodeOffset;
            }

            int unitShift = -1, least = 0, span = AvxVnni.IsSupported ? 255 : 127;
            for (int s = 0; s <= 7 && unitShift < 0 && chunk >= 16 && numbers.All(double.IsFinite); s++)
            {
                var scaled = numbers.Select(n => Math.ScaleB(n, s)).ToArray();
                if (scaled.All(v => v == Math.Round(v)) && scaled.Max() - scaled.Min() <= span)
                {
                    unitShift = s;
                    least = (int)scaled.Min();
                }
            }

            Mode = unitShift >= 0 ? DescribedMode.Integer : DescribedMode.Single;
            Wide = WideRead.Codes;
            C = chunk;
            Cg = 8;
            T = chunk / 4;
            GroupValues = 8 * chunk;
            Period = Lcm(Lcm(d.Values, 32), GroupValues);
            PeriodBytes = Period / d.Values * d.Bytes;
            Planes = d.Planes.Count;
            if (!Spans() || !Places() || !Factors(all, d.Scales?.Count ?? 0))
            {
                return false;
            }

            Mins = d.Mins is { Count: > 0 };
            CodeValues = Pin<float>(count);
            for (int c = 0; c < count; c++)
            {
                CodeValues[c] = (float)numbers[c];
            }

            if (Mode == DescribedMode.Integer)
            {
                Unit = MathF.ScaleB(1f, -unitShift);
                PlaneLeast = Pin<int>(1);
                PlaneLeast[0] = least;
                var unsigned = new byte[count];
                for (int c = 0; c < count; c++)
                {
                    unsigned[c] = (byte)(Math.ScaleB(numbers[c], unitShift) - least);
                }

                bool same = Enumerable.Range(0, count).All(c => unsigned[c] == c);
                bool flip = Enumerable.Range(0, count).All(c => unsigned[c] == (c ^ count / 2));
                if (same)
                {
                    Map = CodeMap.Same;
                }
                else if (flip)
                {
                    Map = CodeMap.Flip;
                    Flip = Vector256.Create((byte)(count / 2));
                }
                else if (count <= 16)
                {
                    Map = CodeMap.Shuffle;
                    Span<byte> table = stackalloc byte[32];
                    for (int i = 0; i < 32; i++)
                    {
                        table[i] = unsigned[i % 16 % count];
                    }

                    Shuffle = Vector256.Create<byte>(table);
                }
                else
                {
                    Map = CodeMap.Table;
                    byteTable = Pin<byte>(256);
                    for (int i = 0; i < 256; i++)
                    {
                        byteTable[i] = unsigned[i % count];
                    }
                }

                int spans = Period / 32;
                if (Planes == 1 && Map != CodeMap.Table && Enumerable.Range(0, spans).All(r => runKind[r] == runKind[0]) && runKind[0] != 0)
                {
                    UniformRuns = runKind[0];
                }

                if (UniformRuns >= 2 && C == 32)
                {
                    // Chunk l of group g is span 8g + l: its run's first byte, and each run q's shift (a 64-bit lane's, or
                    // a 32-bit lane's for 8 runs).
                    int runs = UniformRuns, groups = Period / GroupValues;
                    GatherOffsets = Pin<Vector256<int>>(groups);
                    GatherShifts = Pin<Vector256<uint>>(groups * runs);
                    Span<int> lanes = stackalloc int[8];
                    for (int g = 0; g < groups; g++)
                    {
                        for (int l = 0; l < 8; l++)
                        {
                            lanes[l] = runOffset[8 * g + l];
                        }

                        GatherOffsets[g] = Vector256.Create<int>(lanes);
                        for (int q = 0; q < runs; q++)
                        {
                            for (int l = 0; l < 8; l++)
                            {
                                var shift = runShift[8 * g + l];
                                lanes[l] = runs == 8 ? (int)shift.AsUInt32().GetElement(q) : (int)shift.GetElement(4 * q / runs);
                            }

                            GatherShifts[g * runs + q] = Vector256.Create<int>(lanes).AsUInt32();
                        }
                    }

                    Gathered = true;
                    if (Map == CodeMap.Flip)
                    {
                        // Each field's top bit flipped in the gathered words at once (every run's shift of the flip),
                        // rather than in each step.
                        int fieldTops = 0;
                        var first = runShift[0];
                        for (int q = 0; q < runs; q++)
                        {
                            fieldTops |= (count / 2) << (runs == 8 ? (int)first.AsUInt32().GetElement(q) : (int)first.GetElement(4 * q / runs));
                        }

                        WordFlip = Vector256.Create((uint)(fieldTops * 0x01010101));
                        // Every span's runs at the same shifts, or the steps are loaded and transposed (their map then a step at a time).
                        Gathered = Enumerable.Range(0, spans).All(r => runShift[r].Equals(first));
                    }
                }
            }

            Ready();
            return true;
        }

        // The one-scale read's vectors and table, once the unit is known (OneScale).
        private void Ready()
        {
            if (oneScale == 1)
            {
                oneHalf = new HalfVectors(factorHalves[0], Unit);
            }
            else if (oneScale == 2)
            {
                oneTable = Pin<float>(256);
                for (int c = 0; c < 256; c++)
                {
                    oneTable[c] = factorTables[0][c] * Unit;
                }
            }
        }

        // What the kernels read of the plan on every row, stored once it is made.
        private void Finish()
        {
            GroupsPerPeriod = Period / GroupValues;
            LaneFloats = P > 1 ? 2 * Cg : Mins ? 24 : 16;
            GroupStepBytes = P > 1 ? T * 32 : T * 32 + (Mins ? 64 : 32);
            PlaneMask0 = planeMask == null ? default : planeMask[0];
            MapVector = Map == CodeMap.Flip ? Flip : Shuffle;
        }

        private static int Lcm(int a, int b) => a / Gcd(a, b) * b;

        private static int Gcd(int a, int b) => b == 0 ? a : Gcd(b, a % b);

        // Each span of the period: per plane the bytes its 32 values' fields lie in, as runs of n bytes at one shift each
        // (n · runs = 32, every run over the same bytes): one load, a shift a run and a mask make the span's codes, in the
        // order the runs give; a span whose planes do not fall so (or give other orders) is read one value at a time, in value
        // order.
        private bool Spans()
        {
            var d = D;
            int spans = Period / 32;
            runOffset = Pin<int>(spans * Planes);
            runKind = Pin<byte>(spans * Planes);
            runShift = Pin<Vector256<ulong>>(spans * Planes);
            planeMask = Pin<Vector256<byte>>(Planes);
            planeLow = Pin<int>(Planes);
            Order = Pin<int>(Period);
            int low = 0;
            for (int p = 0; p < Planes; p++)
            {
                planeMask[p] = Vector256.Create((byte)((1 << d.Planes[p].Bits) - 1));
                planeLow[p] = low;
                low += d.Planes[p].Bits;
            }

            Span<int> at = stackalloc int[32];
            Span<int> shift = stackalloc int[32];
            Span<int> mine = stackalloc int[32];
            Span<int> order = stackalloc int[32];
            Span<int> shifts = stackalloc int[8];
            for (int r = 0; r < spans; r++)
            {
                bool vector = true;
                for (int p = 0; p < Planes && vector; p++)
                {
                    var plane = d.Planes[p];
                    int first = int.MaxValue, runs = 0;
                    for (int t = 0; t < 32; t++)
                    {
                        int v = 32 * r + t;
                        var (b, s) = plane.Place(v % d.Values);
                        at[t] = v / d.Values * d.Bytes + b;
                        shift[t] = s;
                        first = Math.Min(first, at[t]);
                        if (shifts[..runs].IndexOf(s) < 0)
                        {
                            if (runs == 8)
                            {
                                vector = false;
                                break;
                            }

                            shifts[runs++] = s;
                        }
                    }

                    if (!vector || runs is not (1 or 2 or 4 or 8))
                    {
                        vector = false;
                        break;
                    }

                    shifts[..runs].Sort();
                    int n = 32 / runs;
                    mine.Fill(-1);
                    for (int t = 0; t < 32 && vector; t++)
                    {
                        int q = shifts[..runs].IndexOf(shift[t]), b = at[t] - first;
                        vector = b < n && mine[q * n + b] < 0;
                        if (vector)
                        {
                            mine[q * n + b] = 32 * r + t;
                        }
                    }

                    if (!vector || p > 0 && !mine.SequenceEqual(order))
                    {
                        vector = false;
                        break;
                    }

                    mine.CopyTo(order);
                    int i = r * Planes + p;
                    runOffset[i] = first;
                    runKind[i] = (byte)runs;
                    // 8 runs: a shift a 32-bit lane; fewer: a shift a 64-bit lane, run q's in lanes [4q / runs, 4(q + 1) / runs).
                    runShift[i] = runs == 8
                        ? Vector256.Create(shifts[0], shifts[1], shifts[2], shifts[3], shifts[4], shifts[5], shifts[6], shifts[7]).AsUInt64()
                        : Vector256.Create((ulong)shifts[0], (ulong)shifts[runs / 4], (ulong)shifts[2 * runs / 4], (ulong)shifts[3 * runs / 4]);
                }

                for (int t = 0; t < 32; t++)
                {
                    Order[32 * r + t] = vector ? order[t] : 32 * r + t;
                }

                if (!vector)
                {
                    for (int p = 0; p < Planes; p++)
                    {
                        runKind[r * Planes + p] = 0;
                    }
                }
            }

            Identity = Enumerable.Range(0, Period).All(v => Order[v] == v);
            Shuffles();
            return true;
        }

        // Each value of the period's place in x's integer layout (group g's bytes from g · GroupValues: step s, lane l = j · P
        // + p, 4 bytes of chunk j's positions 4s..4s + 3 in its span's order) and its chunk; false where a chunk's values do
        // not share every scale (the order mixes them).
        private bool Places()
        {
            XPlace = Pin<int>(Period);
            ChunkOf = Pin<int>(Period);
            if (Order == null)
            {
                Order = Pin<int>(Period);
                for (int v = 0; v < Period; v++)
                {
                    Order[v] = v;
                }
            }

            for (int position = 0; position < Period; position++)
            {
                int v = Order[position], c = position / C, inChunk = position % C;
                int g = c / Cg, j = c % Cg;
                XPlace[v] = g * GroupValues + inChunk / 4 * 4 * Cg + 4 * j + inChunk % 4;
                ChunkOf[v] = c;
            }

            foreach (var f in (D.Scales ?? []).Concat(D.Mins ?? []))
            {
                for (int c = 0; c < Period / C; c++)
                {
                    int v0 = Order[c * C];
                    for (int t = 1; t < C; t++)
                    {
                        int v = Order[c * C + t];
                        if (v / D.Values != v0 / D.Values || v % D.Values / f.Values != v0 % D.Values / f.Values)
                        {
                            return false;
                        }
                    }
                }
            }

            return true;
        }

        // Each scale and minimum: how it reads, and at each group of the period the 8 lanes' dwords (from the period's first
        // byte, kept within the period), the shift and mask of its bits (each piece's), and where a piece's bits go.
        private bool Factors(List<BlockScale> all, int scales)
        {
            factors = all.Count;
            scaleFactors = scales;
            pieces = Math.Max(1, all.Count == 0 ? 1 : all.Max(f => f.Pieces is { } p ? p.Max(x => x.Count) : 1));
            int groups = Period / GroupValues;
            reads = new FactorRead[factors];
            factorBits = new int[factors];
            factorPieces = new int[factors];
            factorBias = new float[factors];
            factorTables = new float*[factors];
            factorHalves = new (int, int, int, float, int)[factors];
            fOffset = Pin<Vector256<int>>(groups * factors * pieces);
            fShift = Pin<Vector256<int>>(groups * factors * pieces);
            fMask = Pin<Vector256<int>>(groups * factors * pieces);
            fLeft = Pin<Vector256<int>>(groups * factors * pieces);
            scalarBlock = new int[groups * factors * 8];
            scalarIndex = new int[groups * factors * 8];
            Span<int> offsets = stackalloc int[8], shifts = stackalloc int[8], masks = stackalloc int[8], lefts = stackalloc int[8];
            for (int f = 0; f < factors; f++)
            {
                var factor = all[f];
                var type = factor.Type;
                factorBits[f] = factor.Pieces is { } given ? given[0].Sum(x => x.Bits) : type.Bits;
                factorPieces[f] = factor.Pieces is { } listed ? listed.Max(x => x.Count) : 1;
                factorBias[f] = (float)factor.Bias;
                if (type.Kind == NumberKind.Integer && factorBits[f] <= 24)
                {
                    reads[f] = type.Signed ? FactorRead.Signed : FactorRead.Unsigned;
                }
                else if (factorBits[f] <= 8)
                {
                    reads[f] = FactorRead.Table;
                    var table = Pin<float>(256);
                    for (int c = 0; c < 256; c++)
                    {
                        table[c] = (float)(type.Value((ulong)c) + factor.Bias);
                    }

                    factorTables[f] = table;
                }
                else if (HalfOf(type) is { } half && factor.Pieces is null)
                {
                    reads[f] = FactorRead.Half;
                    factorHalves[f] = half;
                }
                else if (type is { Kind: NumberKind.Float, Bits: 32, Exponent: 8, Mantissa: 23, ExponentBias: 127, Signed: true, Specials: FloatSpecials.Ieee } && factor.Pieces is null)
                {
                    reads[f] = FactorRead.Single;
                }
                else
                {
                    reads[f] = FactorRead.Scalar;
                }

                for (int g = 0; g < groups; g++)
                {
                    for (int q = 0; q < pieces; q++)
                    {
                        for (int l = 0; l < 8; l++)
                        {
                            int c = g * Cg + l / P, v0 = Order[c * C], block = v0 / D.Values, index = v0 % D.Values / factor.Values;
                            int start = block * D.Bytes, bit, bits, left = 0;
                            scalarBlock[(g * factors + f) * 8 + l] = start;
                            scalarIndex[(g * factors + f) * 8 + l] = index;
                            if (factor.Pieces is { } p)
                            {
                                var own = p[index];
                                if (q >= own.Count)
                                {
                                    (offsets[l], shifts[l], masks[l], lefts[l]) = (start, 0, 0, 0);
                                    continue;
                                }

                                for (int e = 0; e < q; e++)
                                {
                                    left += own[e].Bits;
                                }

                                bit = 8 * own[q].Offset + own[q].Shift;
                                bits = own[q].Bits;
                            }
                            else if (q > 0)
                            {
                                (offsets[l], shifts[l], masks[l], lefts[l]) = (start, 0, 0, 0);
                                continue;
                            }
                            else
                            {
                                bit = 8 * factor.Offset + index * type.Bits;
                                bits = type.Bits;
                            }

                            int dword = start + bit / 8, s = bit % 8;
                            int kept = Math.Max(0, Math.Min(dword, PeriodBytes - 4));
                            s += 8 * (dword - kept);
                            if (s + bits > 32 && reads[f] != FactorRead.Scalar)
                            {
                                reads[f] = FactorRead.Scalar;
                            }

                            (offsets[l], shifts[l], masks[l], lefts[l]) = (kept, Math.Min(s, 31), bits >= 32 ? -1 : (1 << bits) - 1, left);
                        }

                        int i = (g * factors + f) * pieces + q;
                        fOffset[i] = Vector256.Create<int>(offsets);
                        fShift[i] = Vector256.Create<int>(shifts);
                        fMask[i] = Vector256.Create<int>(masks);
                        fLeft[i] = Vector256.Create<int>(lefts);
                    }
                }
            }

            if (factors == 1 && scaleFactors == 1 && factorPieces[0] == 1 && factorBias[0] == 0f)
            {
                oneScale = reads[0] switch
                {
                    FactorRead.Half => 1,
                    FactorRead.Table => 2,
                    _ => 0,
                };
            }

            return PeriodBytes >= 4 || factors == 0;
        }

        /// <summary>
        /// The codes of span <paramref name="span"/> of the period at <paramref name="period"/> (its first byte), 32 bytes in the
        /// span's order: each plane's run loaded once, shifted (a run each) and masked, the planes joined (the first the lowest
        /// bits); a span not in runs one value at a time.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<byte> Raw(byte* period, int span)
        {
            Vector256<byte> code = default;
            int at = span * Planes;
            for (int p = 0; p < Planes; p++, at++)
            {
                byte* from = period + runOffset[at];
                Vector256<byte> v;
                switch (runKind[at])
                {
                    case 1:
                        v = Avx2.ShiftRightLogicalVariable(Vector256.Load(from).AsUInt64(), runShift[at]).AsByte();
                        break;
                    case 2:
                        v = Avx2.ShiftRightLogicalVariable(Avx2.BroadcastVector128ToVector256(from).AsUInt64(), runShift[at]).AsByte();
                        break;
                    case 4:
                        v = Avx2.ShiftRightLogicalVariable(Vector256.Create(*(ulong*)from), runShift[at]).AsByte();
                        break;
                    case 8:
                        v = Avx2.ShiftRightLogicalVariable(Vector256.Create(*(uint*)from), runShift[at].AsUInt32()).AsByte();
                        break;
                    default:
                        return OneByOne(period, span);
                }

                code |= Vector256.ShiftLeft((v & planeMask[p]).AsUInt16(), planeLow[p]).AsByte();
            }

            return code;
        }

        // A span's codes one value at a time, in value order, by the description (a layout the runs do not take).
        [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
        private Vector256<byte> OneByOne(byte* period, int span)
        {
            Span<byte> codes = stackalloc byte[32];
            for (int t = 0; t < 32; t++)
            {
                int v = 32 * span + t;
                codes[t] = (byte)D.CodeBits(new ReadOnlySpan<byte>(period + v / Values * Bytes, Bytes), v % Values);
            }

            return Vector256.Create<byte>(codes);
        }

        /// <summary>The integer lane's bytes of span <paramref name="span"/>'s codes: each its number in units less the least.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<byte> Codes(byte* period, int span)
        {
            var code = Raw(period, span);
            return Map switch
            {
                CodeMap.Flip => code ^ Flip,
                CodeMap.Shuffle => Avx2.Shuffle(Shuffle, code),
                CodeMap.Table => Through(code),
                _ => code,
            };
        }

        // Each byte through the 256-entry table (codes of 5 to 8 bits whose numbers are not their bits).
        [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
        private Vector256<byte> Through(Vector256<byte> code)
        {
            Span<byte> bytes = stackalloc byte[32];
            code.CopyTo(bytes);
            for (int i = 0; i < 32; i++)
            {
                bytes[i] = byteTable[bytes[i]];
            }

            return Vector256.Create<byte>(bytes);
        }

        /// <summary>
        /// Group <paramref name="gp"/> of the period at <paramref name="period"/> as the integer lane's steps (codes of at most 8
        /// bits: a wide integer's are made by IStepLoad.Plane): lane l of step s holds 4 bytes of chunk l, its positions
        /// 4s..4s + 3 in its span's order. T = 4: s4..s7 are not written.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Steps(byte* period, int gp, out Vector256<byte> s0, out Vector256<byte> s1, out Vector256<byte> s2, out Vector256<byte> s3,
            out Vector256<byte> s4, out Vector256<byte> s5, out Vector256<byte> s6, out Vector256<byte> s7)
        {
            int span = gp * (GroupValues / 32);
            if (C == 32)
            {
                Transpose8(Codes(period, span), Codes(period, span + 1), Codes(period, span + 2), Codes(period, span + 3), Codes(period, span + 4),
                    Codes(period, span + 5), Codes(period, span + 6), Codes(period, span + 7), out s0, out s1, out s2, out s3, out s4, out s5, out s6, out s7);
                return;
            }

            Transpose4(Codes(period, span), Codes(period, span + 1), Codes(period, span + 2), Codes(period, span + 3), out s0, out s1, out s2, out s3);
            s4 = s5 = s6 = s7 = default;
        }

        /// <summary>
        /// The scales (each lane's product of the scales covering its chunk, times the unit) and minimums (their product, less
        /// where they are subtracted) of group <paramref name="gp"/> of the period at <paramref name="period"/>, read for the 8
        /// lanes at once; lanes from <paramref name="valid"/> on 0.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Factor(byte* period, int gp, int valid, out Vector256<float> scale, out Vector256<float> min)
        {
            if (OneScale != 0)
            {
                // One scale: its field gathered and read with the unit folded in (a 16-bit float's constants as vectors made
                // once a plan, or a table of its numbers times the unit), no loop over factors.
                var bits = Avx2.ShiftRightLogicalVariable(Avx2.GatherVector256((int*)period, fOffset[gp], 1).AsUInt32(), fShift[gp].AsUInt32()).AsInt32();
                var one = OneScale == 1 ? oneHalf.Of(bits) : Avx2.GatherVector256(oneTable, bits & fMask[gp], 4);
                scale = valid < 8 ? one & GroupMask(valid).AsSingle() : one;
                min = default;
                return;
            }

            var s = Vector256.Create(1f);
            var product = s;
            for (int f = 0; f < factors; f++)
            {
                var v = Read(period, gp, f);
                if (f < scaleFactors)
                {
                    s = f == 0 ? v : s * v;
                }
                else
                {
                    product = f == scaleFactors ? v : product * v;
                }
            }

            if (Unit != 1f)
            {
                s *= Vector256.Create(Unit);
            }

            var mn = !Mins ? Vector256<float>.Zero : D.MinsAdded ? product : -product;
            if (valid < 8)
            {
                var keep = GroupMask(valid).AsSingle();
                s &= keep;
                mn &= keep;
            }

            scale = s;
            min = mn;
        }

        // Factor f's numbers for group gp's 8 lanes: its bits gathered (a dword a lane, each piece's shifted, masked and put
        // in place), then read by its type.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private Vector256<float> Read(byte* period, int gp, int f)
        {
            int at = (gp * factors + f) * pieces;
            var bits = Avx2.ShiftRightLogicalVariable(Avx2.GatherVector256((int*)period, fOffset[at], 1).AsUInt32(), fShift[at].AsUInt32()).AsInt32() & fMask[at];
            for (int q = 1; q < factorPieces[f]; q++)
            {
                var word = Avx2.GatherVector256((int*)period, fOffset[at + q], 1);
                bits |= Avx2.ShiftLeftLogicalVariable((Avx2.ShiftRightLogicalVariable(word.AsUInt32(), fShift[at + q].AsUInt32()).AsInt32() & fMask[at + q]).AsUInt32(),
                    fLeft[at + q].AsUInt32()).AsInt32();
            }

            float bias = factorBias[f];
            switch (reads[f])
            {
                case FactorRead.Unsigned:
                    return Avx.ConvertToVector256Single(bits) + Vector256.Create(bias);
                case FactorRead.Signed:
                    int up = 32 - factorBits[f];
                    return Avx.ConvertToVector256Single(Vector256.ShiftRightArithmetic(Vector256.ShiftLeft(bits, up), up)) + Vector256.Create(bias);
                case FactorRead.Table:
                    return Avx2.GatherVector256(factorTables[f], bits, 4);
                case FactorRead.Half:
                    return HalfBits(bits, factorHalves[f]) + Vector256.Create(bias);
                case FactorRead.Single:
                    return bits.AsSingle() + Vector256.Create(bias);
                default:
                    return ReadOneByOne(period, gp, f);
            }
        }

        // Factor f's numbers one lane at a time, by its type (a layout or a number the gathers do not take).
        [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
        private Vector256<float> ReadOneByOne(byte* period, int gp, int f)
        {
            var factor = f < scaleFactors ? D.Scales![f] : D.Mins![f - scaleFactors];
            Span<float> values = stackalloc float[8];
            for (int l = 0; l < 8; l++)
            {
                int i = (gp * factors + f) * 8 + l;
                values[l] = (float)factor.Value(new ReadOnlySpan<byte>(period + scalarBlock[i], Bytes), scalarIndex[i]);
            }

            return Vector256.Create<float>(values);
        }

        /// <summary>16-bit floats (each lane's low 16 bits) as float32 by their bits (HalfOf): exactly, specials too.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<float> HalfBits(Vector256<int> bits, (int SignShift, int Magnitude, int Shift, float Times, int Special) h)
        {
            var magnitude = bits & Vector256.Create(h.Magnitude);
            var value = Vector256.ShiftLeft(magnitude, h.Shift).AsSingle();
            if (h.Times != 1f)
            {
                value *= Vector256.Create(h.Times);
            }

            var result = value.AsInt32();
            if (h.Special != 0)
            {
                result |= Vector256.GreaterThanOrEqual(magnitude, Vector256.Create(h.Special)) & Vector256.Create(0x7F800000);
            }

            if (h.SignShift != 0)
            {
                result |= Vector256.ShiftLeft(bits & Vector256.Create(0x8000), h.SignShift);
            }

            return result.AsSingle();
        }
    }

    // Steps of the described lane's integer arithmetic a group of stored rows makes at most with more than one row of x (its
    // rows' steps, scales and minimums stay in a core's L2 while four rows of x pass over them; int2's prompt measured 634
    // tokens a second with 32 KB, 660 with 64 KB, 682 with 128 KB, 673 with 256 KB).
    private const int DescribedStepBytes = 131072;

    // Rows of x a tile takes at most where x is int8 codes or the stored values are 4 or more bytes each (each stored row
    // read from memory, and made into steps, once a prompt of that many rows); the float32 lane's 16-bit values take a group
    // of stored rows by RowTileRows rows.
    private const int DescribedTileRows = 128;

    // Stored bytes a group of a heavy type's rows (4 or more bytes a value) takes: the float64 lane's (its tile's x passes once
    // a group, 4 rows of x in L1 a span of k), the integer lane's over byte planes.
    private const int DescribedDoubleGroupBytes = 1 << 20, DescribedPlaneGroupBytes = 65536;

    // Stored bytes a group of 16-bit (or float32) rows takes in the float32 lane, and values of k four rows of x pass over it
    // at a time (32 KB of x stays in L1); the float64 lane's pass (four rows of x as float64, 16 KB).
    private const int DescribedSingleGroupBytes = 65536, DescribedSingleSpan = 2048, DescribedDoubleSpan = 512;

    /// <summary>
    /// The described lane as a format read where it is stored (IRowFormat): one lane (DescribedLaneId) over any type with a
    /// description the plan reads (DescribedPlan), x rounded to int8 in the plan's layout (integer), as it is or in the
    /// spans' order (float32), or widened to float64 (float64). Its tiles are columns over up to DescribedTileRows rows of
    /// x, but the float32 lane's over 16-bit values (a group of stored rows by RowTileRows rows).
    /// </summary>
    private readonly unsafe struct DescribedFormat : IRowFormat<DescribedFormat, DescribedKernels>
    {
        private readonly DescribedPlan plan;
        private readonly int k;

        private DescribedFormat(DescribedPlan plan, int k)
        {
            this.plan = plan;
            this.k = k;
        }

        public static bool Reads(ElementType type, int k, out DescribedFormat format)
        {
            format = type.Description is { } d && k > 0 && k % d.Values == 0 && type.ByteCount(k) == (long)k / d.Values * d.Bytes && DescribedPlan.Of(d) is { } plan
                ? new DescribedFormat(plan, k) : default;
            return format.plan is not null;
        }

        public int First => DescribedLaneId;

        public int Second => DescribedLaneId;

        public bool BothRun => false;

        public int Unmeasured => DescribedLaneId;

        /// <summary>The arithmetic the plan took: "integer", "float32" or "float64".</summary>
        public string Arithmetic => plan.Mode switch
        {
            DescribedMode.Integer => "integer",
            DescribedMode.Single => "float32",
            _ => "float64",
        };

        private bool Heavy => plan.Bytes >= 4 * plan.Values;

        private long RowBytes => (long)k / plan.Values * plan.Bytes;

        public int GroupBytes => plan.Mode switch
        {
            DescribedMode.Integer when plan.P > 1 => DescribedPlaneGroupBytes,
            DescribedMode.Integer => (int)Math.Min(int.MaxValue, RowBytes * Math.Max(1, DescribedStepBytes / Math.Max(1, plan.Groups(k) * plan.GroupStepBytes))),
            DescribedMode.Double => DescribedDoubleGroupBytes,
            _ => plan.Wide == WideRead.Codes ? ScaledGroupBytes : DescribedSingleGroupBytes,
        };

        // Integer arithmetic and codes: columns over up to DescribedTileRows rows of x, a quarter of a thread's share, so each
        // stored row is read (and made into steps) once a prompt of that many rows. (Groups of stored rows by 32 rows of x read
        // each row four times a 128-row prompt: q4_k's prompt 611 tokens a second under the full measured choice against 634
        // by its written kernels, though the lane measured faster at each shape on one weight held in the cache.)
        public (int Columns, int Rows) Tile(int total, int m, int threads, nint rowBytes) => Heavy
            ? ((total + 8 * threads - 1) / (8 * threads), Math.Min(m, DescribedTileRows))
            : plan.Mode == DescribedMode.Integer || plan.Wide == WideRead.Codes
                ? ((total + 4 * threads - 1) / (4 * threads), Math.Min(m, DescribedTileRows))
                : GroupTile(total, rowBytes, GroupBytes);

        public int Span(int k) => plan.Mode == DescribedMode.Double ? DescribedDoubleSpan : plan.Mode == DescribedMode.Single && plan.Wide != WideRead.Codes ? DescribedSingleSpan : k;

        public LaneInput Prepare(int lane, float[] x, int m, int k)
        {
            switch (plan.Mode)
            {
                case DescribedMode.Integer:
                    int groups = plan.Groups(k);
                    var codes = ArrayPool<sbyte>.Shared.Rent(m * groups * plan.GroupValues);
                    var lanes = ArrayPool<float>.Shared.Rent(m * groups * plan.LaneFloats);
                    RoundDescribedRows(plan, x, codes, lanes, m, k);
                    return new LaneInput(x, false, codes, lanes);
                case DescribedMode.Double:
                    var wide = ArrayPool<float>.Shared.Rent(2 * m * k);
                    WidenRows(x, wide, m, k);
                    return new LaneInput(wide, true, null, null);
                default:
                    if (plan.Wide != WideRead.Codes)
                    {
                        return new LaneInput(x, false, null, null);
                    }

                    int padded = plan.Groups(k) * plan.GroupValues;
                    var ordered = ArrayPool<float>.Shared.Rent(m * padded);
                    OrderDescribedRows(plan, x, ordered, m, k, padded);
                    return new LaneInput(ordered, true, null, null);
            }
        }

        // The integer lane's made steps of a group's rows, as floats (every byte written before it is read); a row's last
        // period where it is short of one, copied into zeros (its bytes cleared).
        public (int Floats, int Bytes) Scratch(int k, int group) => plan.Mode switch
        {
            DescribedMode.Integer => (group * plan.Groups(k) * plan.GroupStepBytes / sizeof(float), plan.PeriodBytes),
            DescribedMode.Single when plan.Wide == WideRead.Codes => (0, plan.PeriodBytes),
            _ => (0, 0),
        };

        public DescribedKernels Kernels(int lane, int k, LanePointers x, float* floats, byte* bytes) => new(plan, k, x, (byte*)floats, bytes);
    }

    // The described lane as RowGroups' kernels. Integer: a group's stored rows made into steps, scales and minimums once
    // (DescribedRowSteps), which every four rows of x then read; a row read alone makes them in registers as it reads them.
    // Float32 and float64: each value made in registers as the dot products read it, nothing to set up.
    private readonly unsafe struct DescribedKernels : IRowKernels
    {
        private readonly DescribedPlan plan;
        private readonly int k, groups, full, tailBytes, rowSteps, chunks;
        private readonly nint xRow, lRow;
        private readonly float* x, lanes;
        private readonly sbyte* xq;
        private readonly byte* steps, tail;

        public DescribedKernels(DescribedPlan plan, int k, LanePointers input, byte* steps, byte* tail)
        {
            this.plan = plan;
            this.k = k;
            this.steps = steps;
            this.tail = tail;
            x = input.Values;
            xq = input.Codes;
            lanes = input.Scales;
            groups = plan.Groups(k);
            full = k / plan.Period;
            tailBytes = (int)((long)k / plan.Values * plan.Bytes - (long)full * plan.PeriodBytes);
            rowSteps = groups * plan.GroupStepBytes;
            chunks = (k + plan.C - 1) / plan.C;
            // x's row: the integer layout's whole groups, or (float32 codes) the spans' order in whole groups.
            xRow = (nint)groups * plan.GroupValues;
            lRow = (nint)groups * plan.LaneFloats;
        }

        // A row's last period copied (its bytes, as stored) into the zeros of `tail`, where the row is short of whole periods.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private void Tail(byte* row)
        {
            if (tailBytes > 0)
            {
                new ReadOnlySpan<byte>(row + (nint)full * plan.PeriodBytes, tailBytes).CopyTo(new Span<byte>(tail, tailBytes));
            }
        }

        // Compiled optimized from its first call, so the shape's lane is measured on the code it then runs.
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Group(byte* stored, nint rowBytes, int c0, int count)
        {
            if (plan.Mode != DescribedMode.Integer)
            {
                return;
            }

            for (int c = 0; c < count; c++)
            {
                byte* row = stored + (c0 + c) * rowBytes;
                Tail(row);
                RowStepsOf(plan, row, tail, full, groups, chunks, steps + (nint)c * rowSteps);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public void Four(byte* row, int c, int i, int p0, int p1, out float y0, out float y1, out float y2, out float y3)
        {
            switch (plan.Mode)
            {
                case DescribedMode.Integer:
                    DescribedIntegerFour(plan, steps + (nint)c * rowSteps, groups, xq + i * xRow, lanes + i * lRow, xRow, lRow, out y0, out y1, out y2, out y3);
                    return;
                case DescribedMode.Double:
                    DescribedDoubleFour(plan, row + (nint)p0 * plan.Bytes, (double*)x + (nint)i * k + p0, k, p1 - p0, out y0, out y1, out y2, out y3);
                    return;
                default:
                    if (plan.Wide == WideRead.Codes)
                    {
                        Tail(row);
                        DescribedCodesFour(plan, row, tail, full, groups, chunks, x + i * xRow, xRow, out y0, out y1, out y2, out y3);
                        return;
                    }

                    DescribedSingleFour(plan, row + (nint)p0 * plan.Bytes, x + (nint)i * k + p0, k, p1 - p0, out y0, out y1, out y2, out y3);
                    return;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public float One(byte* row, int c, int i) => plan.Mode == DescribedMode.Integer
            ? DescribedIntegerSteps(plan, steps + (nint)c * rowSteps, groups, xq + i * xRow, lanes + i * lRow)
            : Single(row, i);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public float Single(byte* row, int i)
        {
            switch (plan.Mode)
            {
                case DescribedMode.Integer:
                    Tail(row);
                    return DescribedIntegerOne(plan, row, tail, full, groups, chunks, xq + i * xRow, lanes + i * lRow);
                case DescribedMode.Double:
                    return DescribedDoubleOne(plan, row, (double*)x + (nint)i * k, k);
                default:
                    if (plan.Wide == WideRead.Codes)
                    {
                        Tail(row);
                        return DescribedCodesOne(plan, row, tail, full, groups, chunks, x + i * xRow);
                    }

                    return DescribedSingleOne(plan, row, x + (nint)i * k, k);
            }
        }
    }

    // x rounded to int8 for the described lane's integer arithmetic, row by row (RoundDescribedRow), over the threads where
    // x is large.
    private static void RoundDescribedRows(DescribedPlan plan, float[] x, sbyte[] codes, float[] lanes, int m, int k)
    {
        int groups = plan.Groups(k), xRow = groups * plan.GroupValues, lRow = groups * plan.LaneFloats;
        if (m > 1 && (long)m * k >= 1 << 16)
        {
            Parallel.For(0, m, ComputeResources.ParallelOptions, i => RoundDescribedRow(plan, x.AsSpan(i * k, k), codes.AsSpan(i * xRow, xRow), lanes.AsSpan(i * lRow, lRow)));
            return;
        }

        for (int i = 0; i < m; i++)
        {
            RoundDescribedRow(plan, x.AsSpan(i * k, k), codes.AsSpan(i * xRow, xRow), lanes.AsSpan(i * lRow, lRow));
        }
    }

    // A row of x rounded to int8 a block of 32 values at a time (the block's scale amax / 127, 0 where every value is 0; its
    // codes value · 127 / amax rounded to even), laid out as the plan's steps are (XPlace): each block put in its span's
    // order by two byte shuffles (OrderShuffles; none where spans are in value order), a group's blocks transposed as the
    // stored codes are (one chunk a lane of each step; a wide integer's chunks 4 bytes each of a step's 4 · Cg); per group
    // LaneFloats floats: each lane's block scale, as int32 bits its plane's least times its chunk's code sum, and (minimums)
    // its block scale times its chunk's code sum (in plane 0's lanes). Blocks past k are zero codes, scales and sums. The
    // integer lane's product is then the product of x so rounded.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void RoundDescribedRow(DescribedPlan plan, ReadOnlySpan<float> x, Span<sbyte> codes, Span<float> lanes)
    {
        int k = x.Length, groups = plan.Groups(k), spansPerPeriod = plan.Period / 32, spans = plan.GroupValues / 32, p = plan.P, laneFloats = plan.LaneFloats;
        bool half = plan.C == 16, mins = plan.Mins;
        int* least = plan.PlaneLeast;
        Vector256<byte>* order = plan.OrderShuffles;
        Span<float> block = stackalloc float[32];
        Span<float> scales = stackalloc float[8];
        Span<int> sums = stackalloc int[16];
        Vector256<byte>* made = stackalloc Vector256<byte>[8];
        var flip = Vector256.Create((byte)0x80);
        ref sbyte into = ref MemoryMarshal.GetReference(codes);
        for (int g = 0; g < groups; g++)
        {
            for (int q = 0; q < spans; q++)
            {
                int from = g * plan.GroupValues + 32 * q, count = Math.Clamp(k - from, 0, 32);
                if (count == 0)
                {
                    made[q] = default;
                    scales[q] = 0f;
                    sums[2 * q] = sums[2 * q + 1] = 0;
                    continue;
                }

                scoped ReadOnlySpan<float> values = x.Slice(from, count);
                if (count < 32)
                {
                    block.Clear();
                    values.CopyTo(block);
                    values = block;
                }

                ref float v = ref MemoryMarshal.GetReference(values);
                var v0 = Vector256.LoadUnsafe(ref v);
                var v1 = Vector256.LoadUnsafe(ref v, 8);
                var v2 = Vector256.LoadUnsafe(ref v, 16);
                var v3 = Vector256.LoadUnsafe(ref v, 24);
                var most = Vector256.Max(Vector256.Max(Vector256.Abs(v0), Vector256.Abs(v1)), Vector256.Max(Vector256.Abs(v2), Vector256.Abs(v3)));
                var top = Vector128.Max(most.GetLower(), most.GetUpper());
                top = Vector128.Max(top, Vector128.Shuffle(top, Vector128.Create(2, 3, 0, 1)));
                top = Vector128.Max(top, Vector128.Shuffle(top, Vector128.Create(1, 0, 3, 2)));
                float amax = top.ToScalar();
                var inverse = Vector256.Create(amax > 0f ? 127f / amax : 0f);
                var rounded = Vector256.Narrow(
                    Vector256.Narrow(Vector256.ConvertToInt32(Vector256.Round(v0 * inverse)), Vector256.ConvertToInt32(Vector256.Round(v1 * inverse))),
                    Vector256.Narrow(Vector256.ConvertToInt32(Vector256.Round(v2 * inverse)), Vector256.ConvertToInt32(Vector256.Round(v3 * inverse)))).AsByte();
                if (order != null)
                {
                    int r = from / 32 % spansPerPeriod;
                    rounded = Avx2.Shuffle(Avx2.Permute4x64(rounded.AsUInt64(), 0x44).AsByte(), order[2 * r])
                        | Avx2.Shuffle(Avx2.Permute4x64(rounded.AsUInt64(), 0xEE).AsByte(), order[2 * r + 1]);
                }

                made[q] = rounded;
                scales[q] = amax / 127f;
                var eights = Avx2.SumAbsoluteDifferences(rounded ^ flip, Vector256<byte>.Zero).AsUInt64();   // (code + 128) a byte, 8 bytes a lane
                sums[2 * q] = (int)(eights.GetElement(0) + eights.GetElement(1)) - 16 * 128;
                sums[2 * q + 1] = (int)(eights.GetElement(2) + eights.GetElement(3)) - 16 * 128;
            }

            byte* to = (byte*)Unsafe.AsPointer(ref Unsafe.Add(ref into, g * plan.GroupValues));
            switch (p)
            {
                case 1 when !half:
                    Transpose8(made[0], made[1], made[2], made[3], made[4], made[5], made[6], made[7], out var s0, out var s1, out var s2, out var s3, out var s4, out var s5, out var s6, out var s7);
                    s0.Store(to);
                    s1.Store(to + 32);
                    s2.Store(to + 64);
                    s3.Store(to + 96);
                    s4.Store(to + 128);
                    s5.Store(to + 160);
                    s6.Store(to + 192);
                    s7.Store(to + 224);
                    break;
                case 1:
                    Transpose4(made[0], made[1], made[2], made[3], out var h0, out var h1, out var h2, out var h3);
                    h0.Store(to);
                    h1.Store(to + 32);
                    h2.Store(to + 64);
                    h3.Store(to + 96);
                    break;
                case 8:
                    made[0].Store(to);
                    break;
                case 4:
                    var lo = Avx2.UnpackLow(made[0].AsUInt32(), made[1].AsUInt32());
                    var hi = Avx2.UnpackHigh(made[0].AsUInt32(), made[1].AsUInt32());
                    Avx2.Permute2x128(lo, hi, 0x20).AsByte().Store(to);
                    Avx2.Permute2x128(lo, hi, 0x31).AsByte().Store(to + 32);
                    break;
                default:
                    var t0 = Avx2.UnpackLow(made[0].AsUInt32(), made[1].AsUInt32());
                    var t1 = Avx2.UnpackHigh(made[0].AsUInt32(), made[1].AsUInt32());
                    var t2 = Avx2.UnpackLow(made[2].AsUInt32(), made[3].AsUInt32());
                    var t3 = Avx2.UnpackHigh(made[2].AsUInt32(), made[3].AsUInt32());
                    var u0 = Avx2.UnpackLow(t0.AsUInt64(), t2.AsUInt64());
                    var u1 = Avx2.UnpackHigh(t0.AsUInt64(), t2.AsUInt64());
                    var u2 = Avx2.UnpackLow(t1.AsUInt64(), t3.AsUInt64());
                    var u3 = Avx2.UnpackHigh(t1.AsUInt64(), t3.AsUInt64());
                    Avx2.Permute2x128(u0, u1, 0x20).AsByte().Store(to);
                    Avx2.Permute2x128(u2, u3, 0x20).AsByte().Store(to + 32);
                    Avx2.Permute2x128(u0, u1, 0x31).AsByte().Store(to + 64);
                    Avx2.Permute2x128(u2, u3, 0x31).AsByte().Store(to + 96);
                    break;
            }

            var own = lanes.Slice(g * laneFloats, laneFloats);
            if (p > 1)
            {
                // A wide integer's chunks: each its block scale, then its top plane's least times its code sum (spread to the
                // chunk's lanes as x's codes are: IStepLoad).
                var tops = MemoryMarshal.Cast<float, int>(own[plan.Cg..]);
                for (int j = 0; j < plan.Cg; j++)
                {
                    own[j] = scales[j];
                    tops[j] = least[p - 1] * (sums[2 * j] + sums[2 * j + 1]);
                }

                continue;
            }

            var starts = MemoryMarshal.Cast<float, int>(own.Slice(8, 8));
            for (int l = 0; l < 8; l++)
            {
                int j = l / p, plane = l % p;
                float xs = half ? scales[j / 2] : scales[j];
                int sum = half ? sums[j] : sums[2 * j] + sums[2 * j + 1];
                own[l] = xs;
                starts[l] = least[plane] * sum;
                if (mins)
                {
                    own[16 + l] = plane == 0 ? xs * sum : 0f;
                }
            }
        }
    }

    // x in the spans' order for the float32 lane over codes (OrderDescribedRow), row by row, each `padded` (whole groups).
    private static void OrderDescribedRows(DescribedPlan plan, float[] x, float[] into, int m, int k, int padded)
    {
        for (int i = 0; i < m; i++)
        {
            OrderDescribedRow(plan, x.AsSpan(i * k, k), into.AsSpan(i * padded, padded));
        }
    }

    // A row of x in the order each span's codes are made (DescribedPlan.Order: position t of a period takes the value at
    // Order[t]), whole groups: the positions of values past the row 0.
    private static unsafe void OrderDescribedRow(DescribedPlan plan, ReadOnlySpan<float> x, Span<float> into)
    {
        int period = plan.Period;
        int* order = plan.Order;
        for (int at = 0; at < into.Length; at++)
        {
            int inPeriod = at % period, v = at - inPeriod + order[inPeriod];
            into[at] = v < x.Length ? x[v] : 0f;
        }
    }

    // A stored row's integer steps, scales and minimums (DescribedPlan.Steps and Factor), a group after another, into `into`
    // (GroupStepBytes a group); its last period from `tail` where it is short.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void DescribedRowSteps(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, byte* into)
    {
        int gpp = plan.GroupsPerPeriod, t = plan.T, stepBytes = plan.GroupStepBytes;
        for (int g = 0; g < groups; g++)
        {
            int period = g / gpp, gp = g - period * gpp;
            byte* at = period < full ? row + (nint)period * plan.PeriodBytes : tail;
            byte* to = into + (nint)g * stepBytes;
            plan.Steps(at, gp, out var s0, out var s1, out var s2, out var s3, out var s4, out var s5, out var s6, out var s7);
            s0.Store(to);
            s1.Store(to + 32);
            s2.Store(to + 64);
            s3.Store(to + 96);
            if (t == 8)
            {
                s4.Store(to + 128);
                s5.Store(to + 160);
                s6.Store(to + 192);
                s7.Store(to + 224);
            }

            plan.Factor(at, gp, Math.Min(plan.Cg, chunks - g * plan.Cg) * plan.P, out var scale, out var min);
            scale.Store((float*)(to + 32 * t));
            if (plan.Mins)
            {
                min.Store((float*)(to + 32 * t + 32));
            }
        }
    }

    /// <summary>
    /// How the integer lane loads a step of x's codes (P = 1, 8 chunks' 4 bytes; P = 2, 4 chunks' each in 2 lanes; P = 4, 2
    /// chunks' each in 4; P = 8, one chunk's in all 8), and, for a wide integer (P > 1), makes step s of a group of its
    /// stored codes (chunk j's from `at` + 32 · P · j): lane j · P + p byte p of chunk j's codes 4s..4s + 3.
    /// </summary>
    private unsafe interface IStepLoad
    {
        static abstract Vector256<sbyte> Load(sbyte* at);

        static abstract Vector256<byte> Plane(byte* at, int s);
    }

    private unsafe struct EightChunks : IStepLoad
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<sbyte> Load(sbyte* at) => Vector256.Load(at);

        public static Vector256<byte> Plane(byte* at, int s) => throw new NotSupportedException("Codes of at most 8 bits have one byte plane.");
    }

    private unsafe struct FourChunks : IStepLoad
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<sbyte> Load(sbyte* at) => Avx2.PermuteVar8x32(Avx2.BroadcastVector128ToVector256((int*)at), Vector256.Create(0, 0, 1, 1, 2, 2, 3, 3)).AsSByte();

        // 16-bit codes: 4 chunks' 4 codes (8 bytes each, 64 bytes apart), each 128-bit half's two chunks transposed.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<byte> Plane(byte* at, int s) => Avx2.Shuffle(Vector256.Create(*(ulong*)(at + 8 * s), *(ulong*)(at + 64 + 8 * s), *(ulong*)(at + 128 + 8 * s),
            *(ulong*)(at + 192 + 8 * s)).AsByte(), Vector256.Create((byte)0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15, 0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15));
    }

    private unsafe struct TwoChunks : IStepLoad
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<sbyte> Load(sbyte* at) => Avx2.PermuteVar8x32(Vector256.CreateScalarUnsafe(*(long*)at).AsInt32(), Vector256.Create(0, 0, 0, 0, 1, 1, 1, 1)).AsSByte();

        // 32-bit codes: 2 chunks' 4 codes (16 bytes each, 128 bytes apart), each 128-bit half's 4 × 4 bytes transposed.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<byte> Plane(byte* at, int s) => Avx2.Shuffle(Vector256.Create(Vector128.Load(at + 16 * s), Vector128.Load(at + 128 + 16 * s)),
            Vector256.Create((byte)0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15));
    }

    private unsafe struct OneChunk : IStepLoad
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<sbyte> Load(sbyte* at) => Vector256.Create(*(int*)at).AsSByte();

        // 64-bit codes: one chunk's 4 codes (32 bytes), their low dwords then their high ones, each half's 4 × 4 bytes transposed.
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<byte> Plane(byte* at, int s) => Avx2.Shuffle(Avx2.PermuteVar8x32(Vector256.Load((int*)(at + 32 * s)), Vector256.Create(0, 2, 4, 6, 1, 3, 5, 7)).AsByte(),
            Vector256.Create((byte)0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15));
    }

    // A wide integer's (P > 1) stored row made into the integer lane's steps (8 a group, 32 bytes each, no scales: x's block
    // scales and the planes' weights apply), its last period from `tail`; TX makes each step.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void WideRowSteps<TX>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, byte* into) where TX : struct, IStepLoad
    {
        int periodBytes = plan.PeriodBytes;
        var flip = plan.TopFlip;
        for (int g = 0; g < groups; g++)
        {
            byte* at = g < full ? row + (nint)g * periodBytes : tail;
            byte* to = into + (nint)g * 256;
            (TX.Plane(at, 0) ^ flip).Store(to);
            (TX.Plane(at, 1) ^ flip).Store(to + 32);
            (TX.Plane(at, 2) ^ flip).Store(to + 64);
            (TX.Plane(at, 3) ^ flip).Store(to + 96);
            (TX.Plane(at, 4) ^ flip).Store(to + 128);
            (TX.Plane(at, 5) ^ flip).Store(to + 160);
            (TX.Plane(at, 6) ^ flip).Store(to + 192);
            (TX.Plane(at, 7) ^ flip).Store(to + 224);
        }
    }

    // f + a wide integer's group's sums for a row of x: its 8 steps dotted with x's (TX), each lane from x's start (the top
    // plane's least times the chunk's code sum), then times x's block scales (both spread to the chunks' lanes by TX).
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe Vector256<float> WideRow<TX>(Vector256<float> f, Vector256<byte> s0, Vector256<byte> s1, Vector256<byte> s2, Vector256<byte> s3,
        Vector256<byte> s4, Vector256<byte> s5, Vector256<byte> s6, Vector256<byte> s7, sbyte* x, int xStep, float* lanes, int cg, Vector256<int> top)
        where TX : struct, IStepLoad
    {
        var i = TX.Load((sbyte*)(lanes + cg)).AsInt32() & top;
        i = StepDot(i, s0, TX.Load(x));
        i = StepDot(i, s1, TX.Load(x + xStep));
        i = StepDot(i, s2, TX.Load(x + 2 * xStep));
        i = StepDot(i, s3, TX.Load(x + 3 * xStep));
        i = StepDot(i, s4, TX.Load(x + 4 * xStep));
        i = StepDot(i, s5, TX.Load(x + 5 * xStep));
        i = StepDot(i, s6, TX.Load(x + 6 * xStep));
        i = StepDot(i, s7, TX.Load(x + 7 * xStep));
        return Fma.MultiplyAdd(Avx.ConvertToVector256Single(i), TX.Load((sbyte*)lanes).AsSingle(), f);
    }

    // A wide integer's one row of x by one stored row read alone: each group's 8 steps made in registers (TX.Plane) as they
    // are dotted with x.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float WideMadeOne<TX>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, sbyte* x, float* lanes) where TX : struct, IStepLoad
    {
        int periodBytes = plan.PeriodBytes, groupValues = plan.GroupValues, laneFloats = plan.LaneFloats, cg = plan.Cg, xStep = 4 * cg;
        var flip = plan.TopFlip;
        var top = plan.TopLanes;
        Vector256<float> f = default;
        for (int g = 0; g < groups; g++)
        {
            byte* at = g < full ? row + (nint)g * periodBytes : tail;
            f = WideRow<TX>(f, TX.Plane(at, 0) ^ flip, TX.Plane(at, 1) ^ flip, TX.Plane(at, 2) ^ flip, TX.Plane(at, 3) ^ flip, TX.Plane(at, 4) ^ flip,
                TX.Plane(at, 5) ^ flip, TX.Plane(at, 6) ^ flip, TX.Plane(at, 7) ^ flip, x + (nint)g * groupValues, xStep, lanes + (nint)g * laneFloats, cg, top);
        }

        return Vector256.Sum(f * plan.LaneWeights!.Value);
    }

    // A wide integer's one row of x by a stored row made into steps (WideRowSteps).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float WideStepsOne<TX>(DescribedPlan plan, byte* steps, int groups, sbyte* x, float* lanes) where TX : struct, IStepLoad
    {
        int groupValues = plan.GroupValues, laneFloats = plan.LaneFloats, cg = plan.Cg, xStep = 4 * cg;
        var top = plan.TopLanes;
        Vector256<float> f = default;
        for (int g = 0; g < groups; g++)
        {
            byte* at = steps + (nint)g * 256;
            f = WideRow<TX>(f, Vector256.Load(at), Vector256.Load(at + 32), Vector256.Load(at + 64), Vector256.Load(at + 96), Vector256.Load(at + 128),
                Vector256.Load(at + 160), Vector256.Load(at + 192), Vector256.Load(at + 224), x + (nint)g * groupValues, xStep, lanes + (nint)g * laneFloats, cg, top);
        }

        return Vector256.Sum(f * plan.LaneWeights!.Value);
    }

    // sum + the dot products of a step's unsigned bytes with x's signed codes, 4 a 32-bit lane (AVX-VNNI; else AVX2's
    // multiply-adds, which the plan takes only where the bytes span at most 127).
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static Vector256<int> StepDot(Vector256<int> sum, Vector256<byte> step, Vector256<sbyte> x) => AvxVnni.IsSupported
        ? AvxVnni.MultiplyWideningAndAdd(sum, step, x)
        : sum + Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(step, x), Vector256.Create((short)1));

    // The described integer lane, four rows of x by one stored row made into steps (DescribedRowSteps), dispatched on the
    // step's x load (the plan's byte planes).
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe void DescribedIntegerFour(DescribedPlan plan, byte* steps, int groups, sbyte* x, float* lanes, nint xRow, nint lRow,
        out float y0, out float y1, out float y2, out float y3)
    {
        switch (plan.P)
        {
            case 1:
                ChunkStepsFour(plan, steps, groups, x, lanes, xRow, lRow, out y0, out y1, out y2, out y3);
                break;
            case 2:
                StepsFour<FourChunks>(plan, steps, groups, x, lanes, xRow, lRow, out y0, out y1, out y2, out y3);
                break;
            case 4:
                StepsFour<TwoChunks>(plan, steps, groups, x, lanes, xRow, lRow, out y0, out y1, out y2, out y3);
                break;
            default:
                StepsFour<OneChunk>(plan, steps, groups, x, lanes, xRow, lRow, out y0, out y1, out y2, out y3);
                break;
        }
    }

    // Four rows of x (codes from `x`, a row every `xRow`; lane data from `lanes`, a row every `lRow`) by one stored row's steps
    // of a wide integer (P > 1, `groups` groups of GroupStepBytes, no scales): each step loaded once and dotted with each
    // row's step of x, a lane's sum from x's start (the top plane's least times its chunk's code sum); then times x's block
    // scales (spread to the chunks' lanes as x's codes are), and the lanes weighted by their planes at the end.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void StepsFour<TX>(DescribedPlan plan, byte* steps, int groups, sbyte* x, float* lanes, nint xRow, nint lRow,
        out float y0, out float y1, out float y2, out float y3) where TX : struct, IStepLoad
    {
        int groupValues = plan.GroupValues, laneFloats = plan.LaneFloats, cg = plan.Cg, xStep = 4 * cg;
        var top = plan.TopLanes;
        Vector256<float> f0 = default, f1 = default, f2 = default, f3 = default;
        for (int g = 0; g < groups; g++)
        {
            byte* at = steps + (nint)g * 256;
            var s0 = Vector256.Load(at);
            var s1 = Vector256.Load(at + 32);
            var s2 = Vector256.Load(at + 64);
            var s3 = Vector256.Load(at + 96);
            var s4 = Vector256.Load(at + 128);
            var s5 = Vector256.Load(at + 160);
            var s6 = Vector256.Load(at + 192);
            var s7 = Vector256.Load(at + 224);
            sbyte* xg = x + (nint)g * groupValues;
            float* lg = lanes + (nint)g * laneFloats;
            f0 = WideRow<TX>(f0, s0, s1, s2, s3, s4, s5, s6, s7, xg, xStep, lg, cg, top);
            f1 = WideRow<TX>(f1, s0, s1, s2, s3, s4, s5, s6, s7, xg + xRow, xStep, lg + lRow, cg, top);
            f2 = WideRow<TX>(f2, s0, s1, s2, s3, s4, s5, s6, s7, xg + 2 * xRow, xStep, lg + 2 * lRow, cg, top);
            f3 = WideRow<TX>(f3, s0, s1, s2, s3, s4, s5, s6, s7, xg + 3 * xRow, xStep, lg + 3 * lRow, cg, top);
        }

        var w = plan.LaneWeights!.Value;
        y0 = Vector256.Sum(f0 * w);
        y1 = Vector256.Sum(f1 * w);
        y2 = Vector256.Sum(f2 * w);
        y3 = Vector256.Sum(f3 * w);
    }

    // StepsFour for codes of at most 8 bits (P = 1: a chunk a lane, x's steps 32 bytes): a group's steps (8, or 4 for chunks
    // of 16) loaded once and held while each of the four rows of x is dotted with them in turn.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void ChunkStepsFour(DescribedPlan plan, byte* steps, int groups, sbyte* x, float* lanes, nint xRow, nint lRow,
        out float y0, out float y1, out float y2, out float y3)
    {
        int t = plan.T, stepBytes = plan.GroupStepBytes, groupValues = plan.GroupValues, laneFloats = plan.LaneFloats;
        bool mins = plan.Mins, eight = t == 8;
        Vector256<float> f0 = default, f1 = default, f2 = default, f3 = default;
        for (int g = 0; g < groups; g++)
        {
            byte* at = steps + (nint)g * stepBytes;
            var s0 = Vector256.Load(at);
            var s1 = Vector256.Load(at + 32);
            var s2 = Vector256.Load(at + 64);
            var s3 = Vector256.Load(at + 96);
            Vector256<byte> s4 = default, s5 = default, s6 = default, s7 = default;
            if (eight)
            {
                s4 = Vector256.Load(at + 128);
                s5 = Vector256.Load(at + 160);
                s6 = Vector256.Load(at + 192);
                s7 = Vector256.Load(at + 224);
            }

            var scale = Vector256.Load((float*)(at + 32 * t));
            sbyte* xg = x + (nint)g * groupValues;
            float* lg = lanes + (nint)g * laneFloats;
            f0 = ChunkRow(f0, eight, s0, s1, s2, s3, s4, s5, s6, s7, scale, xg, lg);
            f1 = ChunkRow(f1, eight, s0, s1, s2, s3, s4, s5, s6, s7, scale, xg + xRow, lg + lRow);
            f2 = ChunkRow(f2, eight, s0, s1, s2, s3, s4, s5, s6, s7, scale, xg + 2 * xRow, lg + 2 * lRow);
            f3 = ChunkRow(f3, eight, s0, s1, s2, s3, s4, s5, s6, s7, scale, xg + 3 * xRow, lg + 3 * lRow);
            if (mins)
            {
                var min = Vector256.Load((float*)(at + 32 * t + 32));
                f0 = Fma.MultiplyAdd(min, Vector256.Load(lg + 16), f0);
                f1 = Fma.MultiplyAdd(min, Vector256.Load(lg + lRow + 16), f1);
                f2 = Fma.MultiplyAdd(min, Vector256.Load(lg + 2 * lRow + 16), f2);
                f3 = Fma.MultiplyAdd(min, Vector256.Load(lg + 3 * lRow + 16), f3);
            }
        }

        y0 = Vector256.Sum(f0);
        y1 = Vector256.Sum(f1);
        y2 = Vector256.Sum(f2);
        y3 = Vector256.Sum(f3);
    }

    // f + a group's sums for a row of x: its steps dotted with x's (32 bytes a step from `x`), each lane's sum from x's start
    // (`lanes` + 8), then times the lanes' scales and x's (`lanes`).
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe Vector256<float> ChunkRow(Vector256<float> f, bool eight, Vector256<byte> s0, Vector256<byte> s1, Vector256<byte> s2, Vector256<byte> s3,
        Vector256<byte> s4, Vector256<byte> s5, Vector256<byte> s6, Vector256<byte> s7, Vector256<float> scale, sbyte* x, float* lanes)
    {
        var i = Vector256.Load((int*)(lanes + 8));
        i = StepDot(i, s0, Vector256.Load(x));
        i = StepDot(i, s1, Vector256.Load(x + 32));
        i = StepDot(i, s2, Vector256.Load(x + 64));
        i = StepDot(i, s3, Vector256.Load(x + 96));
        if (eight)
        {
            i = StepDot(i, s4, Vector256.Load(x + 128));
            i = StepDot(i, s5, Vector256.Load(x + 160));
            i = StepDot(i, s6, Vector256.Load(x + 192));
            i = StepDot(i, s7, Vector256.Load(x + 224));
        }

        return Fma.MultiplyAdd(Avx.ConvertToVector256Single(i), scale * Vector256.Load(lanes), f);
    }

    // StepsFour for one row of x (the rows of a group left over after its fours).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float DescribedIntegerSteps(DescribedPlan plan, byte* steps, int groups, sbyte* x, float* lanes) => plan.P switch
    {
        1 => StepsOne<EightChunks>(plan, steps, groups, x, lanes),
        2 => WideStepsOne<FourChunks>(plan, steps, groups, x, lanes),
        4 => WideStepsOne<TwoChunks>(plan, steps, groups, x, lanes),
        _ => WideStepsOne<OneChunk>(plan, steps, groups, x, lanes),
    };

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float StepsOne<TX>(DescribedPlan plan, byte* steps, int groups, sbyte* x, float* lanes) where TX : struct, IStepLoad
    {
        int t = plan.T, stepBytes = plan.GroupStepBytes, groupValues = plan.GroupValues, laneFloats = plan.LaneFloats, xStep = 4 * plan.Cg;
        bool mins = plan.Mins;
        Vector256<float> f = default;
        for (int g = 0; g < groups; g++)
        {
            byte* at = steps + (nint)g * stepBytes;
            sbyte* xg = x + (nint)g * groupValues;
            float* lg = lanes + (nint)g * laneFloats;
            var xs = Vector256.Load(lg);
            var i = Vector256.Load((int*)(lg + 8));
            for (int s = 0; s < t; s++)
            {
                i = StepDot(i, Vector256.Load(at + 32 * s), TX.Load(xg + s * xStep));
            }

            f = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i), Vector256.Load((float*)(at + 32 * t)) * xs, f);
            if (mins)
            {
                f = Fma.MultiplyAdd(Vector256.Load((float*)(at + 32 * t + 32)), Vector256.Load(lg + 16), f);
            }
        }

        return Vector256.Sum(plan.LaneWeights is { } w ? f * w : f);
    }

    // The described integer lane for one row of x read alone (one row of x, or a tile's one row): each group's steps,
    // scales and minimums made in registers from the stored row (its last period from `tail`) as they are dotted with x;
    // a plan whose spans are one plane loaded the same way takes its load and map as type parameters (UniformOne).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float DescribedIntegerOne(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, sbyte* x, float* lanes)
    {
        if (plan.P == 1 && plan.UniformRuns != 0)
        {
            return plan.UniformRuns switch
            {
                1 => UniformOneBy<OneRun>(plan, row, tail, full, groups, chunks, x, lanes),
                2 => UniformOneBy<TwoRuns>(plan, row, tail, full, groups, chunks, x, lanes),
                4 => UniformOneBy<FourRuns>(plan, row, tail, full, groups, chunks, x, lanes),
                _ => UniformOneBy<EightRuns>(plan, row, tail, full, groups, chunks, x, lanes),
            };
        }

        return plan.P switch
        {
            1 => MadeOne<EightChunks>(plan, row, tail, full, groups, chunks, x, lanes),
            2 => WideMadeOne<FourChunks>(plan, row, tail, full, groups, x, lanes),
            4 => WideMadeOne<TwoChunks>(plan, row, tail, full, groups, x, lanes),
            _ => WideMadeOne<OneChunk>(plan, row, tail, full, groups, x, lanes),
        };
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float UniformOneBy<TR>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, sbyte* x, float* lanes) where TR : struct, IRunLoad =>
        plan.Map switch
        {
            CodeMap.Same => UniformOne<TR, SameCodes>(plan, row, tail, full, groups, chunks, x, lanes),
            CodeMap.Flip => UniformOne<TR, FlippedCodes>(plan, row, tail, full, groups, chunks, x, lanes),
            _ => UniformOne<TR, ShuffledCodes>(plan, row, tail, full, groups, chunks, x, lanes),
        };

    // A stored row's integer steps (DescribedRowSteps), by the plan's uniform load and map as type parameters where it has them.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void RowStepsOf(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, byte* into)
    {
        switch (plan.P)
        {
            case 2:
                WideRowSteps<FourChunks>(plan, row, tail, full, groups, into);
                return;
            case 4:
                WideRowSteps<TwoChunks>(plan, row, tail, full, groups, into);
                return;
            case 8:
                WideRowSteps<OneChunk>(plan, row, tail, full, groups, into);
                return;
        }

        if (plan.UniformRuns == 0)
        {
            DescribedRowSteps(plan, row, tail, full, groups, chunks, into);
            return;
        }

        switch (plan.UniformRuns)
        {
            case 1:
                RowStepsBy<OneRun>(plan, row, tail, full, groups, chunks, into);
                break;
            case 2:
                RowStepsBy<TwoRuns>(plan, row, tail, full, groups, chunks, into);
                break;
            case 4:
                RowStepsBy<FourRuns>(plan, row, tail, full, groups, chunks, into);
                break;
            default:
                RowStepsBy<EightRuns>(plan, row, tail, full, groups, chunks, into);
                break;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void RowStepsBy<TR>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, byte* into) where TR : struct, IRunLoad
    {
        switch (plan.Map)
        {
            case CodeMap.Same:
                UniformRowSteps<TR, SameCodes>(plan, row, tail, full, groups, chunks, into);
                break;
            case CodeMap.Flip:
                UniformRowSteps<TR, FlippedCodes>(plan, row, tail, full, groups, chunks, into);
                break;
            default:
                UniformRowSteps<TR, ShuffledCodes>(plan, row, tail, full, groups, chunks, into);
                break;
        }
    }

    /// <summary>How the described lane loads a span's run of its one code plane and shifts it (every span the same: DescribedPlan.UniformRuns).</summary>
    private unsafe interface IRunLoad
    {
        static abstract Vector256<byte> Load(byte* at, Vector256<ulong> shift);
    }

    // One run: 32 bytes at one shift.
    private unsafe struct OneRun : IRunLoad
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<byte> Load(byte* at, Vector256<ulong> shift) => Avx2.ShiftRightLogicalVariable(Vector256.Load(at).AsUInt64(), shift).AsByte();
    }

    // Two runs: 16 bytes at two shifts (a 128-bit half each).
    private unsafe struct TwoRuns : IRunLoad
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<byte> Load(byte* at, Vector256<ulong> shift) => Avx2.ShiftRightLogicalVariable(Avx2.BroadcastVector128ToVector256(at).AsUInt64(), shift).AsByte();
    }

    // Four runs: 8 bytes at four shifts (a 64-bit lane each).
    private unsafe struct FourRuns : IRunLoad
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<byte> Load(byte* at, Vector256<ulong> shift) => Avx2.ShiftRightLogicalVariable(Vector256.Create(*(ulong*)at), shift).AsByte();
    }

    // Eight runs: 4 bytes at eight shifts (a 32-bit lane each).
    private unsafe struct EightRuns : IRunLoad
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<byte> Load(byte* at, Vector256<ulong> shift) => Avx2.ShiftRightLogicalVariable(Vector256.Create(*(uint*)at), shift.AsUInt32()).AsByte();
    }

    /// <summary>How the integer lane makes a code its byte (DescribedPlan.Map): as it is, flipped, or by a 16-entry byte shuffle.</summary>
    private interface ICodeMap
    {
        static abstract Vector256<byte> Map(Vector256<byte> code, Vector256<byte> table);
    }

    private struct SameCodes : ICodeMap
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<byte> Map(Vector256<byte> code, Vector256<byte> table) => code;
    }

    private struct FlippedCodes : ICodeMap
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<byte> Map(Vector256<byte> code, Vector256<byte> table) => code ^ table;
    }

    private struct ShuffledCodes : ICodeMap
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static Vector256<byte> Map(Vector256<byte> code, Vector256<byte> table) => Avx2.Shuffle(table, code);
    }

    // Span s's codes (of the period at `at`) as the integer lane's bytes, by TR and TM.
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe Vector256<byte> SpanCodes<TR, TM>(byte* at, int* offsets, Vector256<ulong>* shifts, int s, Vector256<byte> mask, Vector256<byte> table)
        where TR : struct, IRunLoad where TM : struct, ICodeMap => TM.Map(TR.Load(at + offsets[s], shifts[s]) & mask, table);

    // Group gp of the period at `at` as the integer lane's steps (DescribedPlan.Steps, a plan with uniform runs): 8 spans
    // transposed (chunks of 32), or 4 (chunks of 16: s4..s7 not written).
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe void UniformSteps<TR, TM>(byte* at, int gp, int* offsets, Vector256<ulong>* shifts, Vector256<byte> mask, Vector256<byte> table, bool c32,
        out Vector256<byte> s0, out Vector256<byte> s1, out Vector256<byte> s2, out Vector256<byte> s3, out Vector256<byte> s4, out Vector256<byte> s5,
        out Vector256<byte> s6, out Vector256<byte> s7) where TR : struct, IRunLoad where TM : struct, ICodeMap
    {
        if (c32)
        {
            int s = 8 * gp;
            Transpose8(SpanCodes<TR, TM>(at, offsets, shifts, s, mask, table), SpanCodes<TR, TM>(at, offsets, shifts, s + 1, mask, table),
                SpanCodes<TR, TM>(at, offsets, shifts, s + 2, mask, table), SpanCodes<TR, TM>(at, offsets, shifts, s + 3, mask, table),
                SpanCodes<TR, TM>(at, offsets, shifts, s + 4, mask, table), SpanCodes<TR, TM>(at, offsets, shifts, s + 5, mask, table),
                SpanCodes<TR, TM>(at, offsets, shifts, s + 6, mask, table), SpanCodes<TR, TM>(at, offsets, shifts, s + 7, mask, table),
                out s0, out s1, out s2, out s3, out s4, out s5, out s6, out s7);
            return;
        }

        int r = 4 * gp;
        Transpose4(SpanCodes<TR, TM>(at, offsets, shifts, r, mask, table), SpanCodes<TR, TM>(at, offsets, shifts, r + 1, mask, table),
            SpanCodes<TR, TM>(at, offsets, shifts, r + 2, mask, table), SpanCodes<TR, TM>(at, offsets, shifts, r + 3, mask, table), out s0, out s1, out s2, out s3);
        s4 = s5 = s6 = s7 = default;
    }

    // Group gp's steps gathered (DescribedPlan.Gathered): dword d of the 8 chunks' runs (each its own first byte, a lane
    // each) in one gather, each run q's shifts of it masked and mapped: step q · D + d, D = 8 / runs (the positions 4s..4s + 3
    // of each chunk in its span's order, as the transposed loads give them).
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe void GatheredSteps<TM>(byte* at, Vector256<int> offsets, int* spanOffsets, Vector256<uint>* shifts, int runs, Vector256<uint> flip, Vector256<byte> mask, Vector256<byte> table,
        out Vector256<byte> s0, out Vector256<byte> s1, out Vector256<byte> s2, out Vector256<byte> s3, out Vector256<byte> s4, out Vector256<byte> s5,
        out Vector256<byte> s6, out Vector256<byte> s7) where TM : struct, ICodeMap
    {
        switch (runs)
        {
            case 8:
                var g = Avx2.GatherVector256((int*)at, offsets, 1).AsUInt32() ^ flip;
                s0 = GatheredStep<TM>(g, shifts[0], mask, table);
                s1 = GatheredStep<TM>(g, shifts[1], mask, table);
                s2 = GatheredStep<TM>(g, shifts[2], mask, table);
                s3 = GatheredStep<TM>(g, shifts[3], mask, table);
                s4 = GatheredStep<TM>(g, shifts[4], mask, table);
                s5 = GatheredStep<TM>(g, shifts[5], mask, table);
                s6 = GatheredStep<TM>(g, shifts[6], mask, table);
                s7 = GatheredStep<TM>(g, shifts[7], mask, table);
                return;
            case 4:
                var a0 = Avx2.GatherVector256((int*)at, offsets, 1).AsUInt32() ^ flip;
                var a1 = Avx2.GatherVector256((int*)(at + 4), offsets, 1).AsUInt32() ^ flip;
                s0 = GatheredStep<TM>(a0, shifts[0], mask, table);
                s1 = GatheredStep<TM>(a1, shifts[0], mask, table);
                s2 = GatheredStep<TM>(a0, shifts[1], mask, table);
                s3 = GatheredStep<TM>(a1, shifts[1], mask, table);
                s4 = GatheredStep<TM>(a0, shifts[2], mask, table);
                s5 = GatheredStep<TM>(a1, shifts[2], mask, table);
                s6 = GatheredStep<TM>(a0, shifts[3], mask, table);
                s7 = GatheredStep<TM>(a1, shifts[3], mask, table);
                return;
            default:
                // Two runs of 16 bytes: the 8 chunks' runs loaded (chunk l and l + 4 a vector) and their dwords transposed by
                // unpacks (8 loads and 8 unpacks: fewer than 4 gathers), dword d of every chunk in b_d.
                var r0 = Vector256.Create(Vector128.Load(at + spanOffsets[0]), Vector128.Load(at + spanOffsets[4])).AsUInt32();
                var r1 = Vector256.Create(Vector128.Load(at + spanOffsets[1]), Vector128.Load(at + spanOffsets[5])).AsUInt32();
                var r2 = Vector256.Create(Vector128.Load(at + spanOffsets[2]), Vector128.Load(at + spanOffsets[6])).AsUInt32();
                var r3 = Vector256.Create(Vector128.Load(at + spanOffsets[3]), Vector128.Load(at + spanOffsets[7])).AsUInt32();
                var w0 = Avx2.UnpackLow(r0, r1).AsUInt64();
                var w1 = Avx2.UnpackHigh(r0, r1).AsUInt64();
                var w2 = Avx2.UnpackLow(r2, r3).AsUInt64();
                var w3 = Avx2.UnpackHigh(r2, r3).AsUInt64();
                var b0 = Avx2.UnpackLow(w0, w2).AsUInt32() ^ flip;
                var b1 = Avx2.UnpackHigh(w0, w2).AsUInt32() ^ flip;
                var b2 = Avx2.UnpackLow(w1, w3).AsUInt32() ^ flip;
                var b3 = Avx2.UnpackHigh(w1, w3).AsUInt32() ^ flip;
                s0 = GatheredStep<TM>(b0, shifts[0], mask, table);
                s1 = GatheredStep<TM>(b1, shifts[0], mask, table);
                s2 = GatheredStep<TM>(b2, shifts[0], mask, table);
                s3 = GatheredStep<TM>(b3, shifts[0], mask, table);
                s4 = GatheredStep<TM>(b0, shifts[1], mask, table);
                s5 = GatheredStep<TM>(b1, shifts[1], mask, table);
                s6 = GatheredStep<TM>(b2, shifts[1], mask, table);
                s7 = GatheredStep<TM>(b3, shifts[1], mask, table);
                return;
        }
    }

    // A gathered dword vector at one run's shifts, masked and mapped: a step.
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static Vector256<byte> GatheredStep<TM>(Vector256<uint> words, Vector256<uint> shift, Vector256<byte> mask, Vector256<byte> table) where TM : struct, ICodeMap =>
        TM.Map(Avx2.ShiftRightLogicalVariable(words, shift).AsByte() & mask, table);

    // MadeOne for a plan with uniform runs (P = 1), its load and map as type parameters.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float UniformOne<TR, TM>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, sbyte* x, float* lanes)
        where TR : struct, IRunLoad where TM : struct, ICodeMap
    {
        int gpp = plan.GroupsPerPeriod, groupValues = plan.GroupValues, laneFloats = plan.LaneFloats, periodBytes = plan.PeriodBytes;
        bool mins = plan.Mins, c32 = plan.C == 32;
        int* offsets = plan.RunOffsets;
        Vector256<ulong>* shifts = plan.RunShifts;
        var mask = plan.PlaneMask0;
        var table = plan.MapVector;
        bool gathered = plan.Gathered;
        int runs = plan.UniformRuns;
        Vector256<int>* gatherOffsets = plan.GatherOffsets;
        Vector256<uint>* gatherShifts = plan.GatherShifts;
        var wordFlip = plan.WordFlip;
        Vector256<float> f = default;
        for (int g = 0; g < groups; g++)
        {
            int period = gpp == 1 ? g : g / gpp, gp = g - period * gpp;
            byte* at = period < full ? row + (nint)period * periodBytes : tail;
            Vector256<byte> s0, s1, s2, s3, s4, s5, s6, s7;
            if (gathered)
            {
                if (typeof(TM) == typeof(FlippedCodes))
                {
                    GatheredSteps<SameCodes>(at, gatherOffsets[gp], offsets + 8 * gp, gatherShifts + gp * runs, runs, wordFlip, mask, table, out s0, out s1, out s2, out s3, out s4, out s5, out s6, out s7);
                }
                else
                {
                    GatheredSteps<TM>(at, gatherOffsets[gp], offsets + 8 * gp, gatherShifts + gp * runs, runs, default, mask, table, out s0, out s1, out s2, out s3, out s4, out s5, out s6, out s7);
                }
            }
            else
            {
                UniformSteps<TR, TM>(at, gp, offsets, shifts, mask, table, c32, out s0, out s1, out s2, out s3, out s4, out s5, out s6, out s7);
            }

            plan.Factor(at, gp, Math.Min(8, chunks - g * 8), out var scale, out var min);
            sbyte* xg = x + (nint)g * groupValues;
            float* lg = lanes + (nint)g * laneFloats;
            var i = Vector256.Load((int*)(lg + 8));
            i = StepDot(i, s0, Vector256.Load(xg));
            i = StepDot(i, s1, Vector256.Load(xg + 32));
            i = StepDot(i, s2, Vector256.Load(xg + 64));
            i = StepDot(i, s3, Vector256.Load(xg + 96));
            if (c32)
            {
                i = StepDot(i, s4, Vector256.Load(xg + 128));
                i = StepDot(i, s5, Vector256.Load(xg + 160));
                i = StepDot(i, s6, Vector256.Load(xg + 192));
                i = StepDot(i, s7, Vector256.Load(xg + 224));
            }

            f = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i), scale * Vector256.Load(lg), f);
            if (mins)
            {
                f = Fma.MultiplyAdd(min, Vector256.Load(lg + 16), f);
            }
        }

        return Vector256.Sum(f);
    }

    // DescribedRowSteps for a plan with uniform runs (P = 1), its load and map as type parameters.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void UniformRowSteps<TR, TM>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, byte* into)
        where TR : struct, IRunLoad where TM : struct, ICodeMap
    {
        int gpp = plan.GroupsPerPeriod, t = plan.T, stepBytes = plan.GroupStepBytes, periodBytes = plan.PeriodBytes;
        bool mins = plan.Mins, c32 = t == 8;
        int* offsets = plan.RunOffsets;
        Vector256<ulong>* shifts = plan.RunShifts;
        var mask = plan.PlaneMask0;
        var table = plan.MapVector;
        bool gathered = plan.Gathered;
        int runs = plan.UniformRuns;
        Vector256<int>* gatherOffsets = plan.GatherOffsets;
        Vector256<uint>* gatherShifts = plan.GatherShifts;
        var wordFlip = plan.WordFlip;
        for (int g = 0; g < groups; g++)
        {
            int period = gpp == 1 ? g : g / gpp, gp = g - period * gpp;
            byte* at = period < full ? row + (nint)period * periodBytes : tail;
            byte* to = into + (nint)g * stepBytes;
            Vector256<byte> s0, s1, s2, s3, s4, s5, s6, s7;
            if (gathered)
            {
                if (typeof(TM) == typeof(FlippedCodes))
                {
                    GatheredSteps<SameCodes>(at, gatherOffsets[gp], offsets + 8 * gp, gatherShifts + gp * runs, runs, wordFlip, mask, table, out s0, out s1, out s2, out s3, out s4, out s5, out s6, out s7);
                }
                else
                {
                    GatheredSteps<TM>(at, gatherOffsets[gp], offsets + 8 * gp, gatherShifts + gp * runs, runs, default, mask, table, out s0, out s1, out s2, out s3, out s4, out s5, out s6, out s7);
                }
            }
            else
            {
                UniformSteps<TR, TM>(at, gp, offsets, shifts, mask, table, c32, out s0, out s1, out s2, out s3, out s4, out s5, out s6, out s7);
            }

            s0.Store(to);
            s1.Store(to + 32);
            s2.Store(to + 64);
            s3.Store(to + 96);
            if (c32)
            {
                s4.Store(to + 128);
                s5.Store(to + 160);
                s6.Store(to + 192);
                s7.Store(to + 224);
            }

            plan.Factor(at, gp, Math.Min(8, chunks - g * 8), out var scale, out var min);
            scale.Store((float*)(to + 32 * t));
            if (mins)
            {
                min.Store((float*)(to + 32 * t + 32));
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float MadeOne<TX>(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, sbyte* x, float* lanes) where TX : struct, IStepLoad
    {
        int gpp = plan.GroupsPerPeriod, t = plan.T, groupValues = plan.GroupValues, laneFloats = plan.LaneFloats, xStep = 4 * plan.Cg, cg = plan.Cg, p = plan.P;
        bool mins = plan.Mins;
        Vector256<float> f = default;
        for (int g = 0; g < groups; g++)
        {
            int period = gpp == 1 ? g : g / gpp, gp = g - period * gpp;
            byte* at = period < full ? row + (nint)period * plan.PeriodBytes : tail;
            plan.Steps(at, gp, out var s0, out var s1, out var s2, out var s3, out var s4, out var s5, out var s6, out var s7);
            plan.Factor(at, gp, Math.Min(cg, chunks - g * cg) * p, out var scale, out var min);
            sbyte* xg = x + (nint)g * groupValues;
            float* lg = lanes + (nint)g * laneFloats;
            var xs = Vector256.Load(lg);
            var i = Vector256.Load((int*)(lg + 8));
            i = StepDot(i, s0, TX.Load(xg));
            i = StepDot(i, s1, TX.Load(xg + xStep));
            i = StepDot(i, s2, TX.Load(xg + 2 * xStep));
            i = StepDot(i, s3, TX.Load(xg + 3 * xStep));
            if (t == 8)
            {
                i = StepDot(i, s4, TX.Load(xg + 4 * xStep));
                i = StepDot(i, s5, TX.Load(xg + 5 * xStep));
                i = StepDot(i, s6, TX.Load(xg + 6 * xStep));
                i = StepDot(i, s7, TX.Load(xg + 7 * xStep));
            }

            f = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i), scale * xs, f);
            if (mins)
            {
                f = Fma.MultiplyAdd(min, Vector256.Load(lg + 16), f);
            }
        }

        return Vector256.Sum(plan.LaneWeights is { } w ? f * w : f);
    }

    /// <summary>How the float32 lane makes 8 stored values of 2 or 4 bytes each float32 in registers (and one, for a row's tail).</summary>
    private unsafe interface ISingleRead
    {
        Vector256<float> Eight(byte* at);

        float One(byte* at);
    }

    // A 16-bit float with float32's exponent bits and bias (bfloat16's description): float32's high half, by a shift.
    private readonly unsafe struct HighHalfRead : ISingleRead
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<float> Eight(byte* at) => Avx2.ShiftLeftLogical(Avx2.ConvertToVector256Int32((ushort*)at), 16).AsSingle();

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public float One(byte* at) => BitConverter.Int32BitsToSingle(*(ushort*)at << 16);
    }

    // Any other 16-bit float by its bits (DescribedPlan.HalfBits).
    private readonly unsafe struct HalfRead((int SignShift, int Magnitude, int Shift, float Times, int Special) h) : ISingleRead
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<float> Eight(byte* at) => DescribedPlan.HalfBits(Avx2.ConvertToVector256Int32((ushort*)at), h);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public float One(byte* at) => DescribedPlan.HalfBits(Vector256.Create((int)*(ushort*)at), h).ToScalar();
    }

    // A 16-bit integer (two's complement where signed).
    private readonly unsafe struct ShortRead(bool signed) : ISingleRead
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<float> Eight(byte* at) =>
            Avx.ConvertToVector256Single(signed ? Avx2.ConvertToVector256Int32((short*)at) : Avx2.ConvertToVector256Int32((ushort*)at));

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public float One(byte* at) => signed ? *(short*)at : *(ushort*)at;
    }

    // float32 as stored.
    private readonly unsafe struct SingleRead : ISingleRead
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<float> Eight(byte* at) => Vector256.Load((float*)at);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public float One(byte* at) => *(float*)at;
    }

    // A read plus the code offset (a description with one; one without takes the read alone).
    private readonly unsafe struct SingleOffset<TR>(TR read, float offset) : ISingleRead where TR : struct, ISingleRead
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<float> Eight(byte* at) => read.Eight(at) + Vector256.Create(offset);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public float One(byte* at) => read.One(at) + offset;
    }

    /// <summary>How the float64 lane makes 4 stored values float64 in registers (and one, for a row's tail).</summary>
    private unsafe interface IDoubleRead
    {
        Vector256<double> Four(byte* at);

        double One(byte* at);
    }

    // float64 as stored.
    private readonly unsafe struct DoubleRead : IDoubleRead
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<double> Four(byte* at) => Vector256.Load((double*)at);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public double One(byte* at) => *(double*)at;
    }

    // A 64-bit two's complement integer, rounded once to float64 (LongsToDoubles).
    private readonly unsafe struct LongRead : IDoubleRead
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<double> Four(byte* at) => LongsToDoubles((long*)at);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public double One(byte* at) => *(long*)at;
    }

    // A 32-bit two's complement integer, exactly.
    private readonly unsafe struct IntRead : IDoubleRead
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<double> Four(byte* at) => Avx.ConvertToVector256Double(Vector128.Load((int*)at));

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public double One(byte* at) => *(int*)at;
    }

    // A read plus the code offset.
    private readonly unsafe struct DoubleOffset<TR>(TR read, double offset) : IDoubleRead where TR : struct, IDoubleRead
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<double> Four(byte* at) => read.Four(at) + Vector256.Create(offset);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public double One(byte* at) => read.One(at) + offset;
    }

    // The described float64 lane: four rows of x (float64, `stride` apart) by `count` stored values, dispatched on how a value
    // reads (and whether the description adds a code offset).
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe void DescribedDoubleFour(DescribedPlan plan, byte* row, double* x, int stride, int count, out float y0, out float y1, out float y2, out float y3)
    {
        switch (plan.Wide)
        {
            case WideRead.Long:
                DoubleFourBy(new LongRead(), 8, plan.Offset, row, x, stride, count, out y0, out y1, out y2, out y3);
                break;
            case WideRead.Int:
                DoubleFourBy(new IntRead(), 4, plan.Offset, row, x, stride, count, out y0, out y1, out y2, out y3);
                break;
            default:
                DoubleFourBy(new DoubleRead(), 8, plan.Offset, row, x, stride, count, out y0, out y1, out y2, out y3);
                break;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe void DoubleFourBy<TR>(TR read, int bytes, double offset, byte* row, double* x, int stride, int count, out float y0, out float y1, out float y2, out float y3)
        where TR : struct, IDoubleRead
    {
        if (offset == 0)
        {
            DoubleFour(read, bytes, row, x, stride, count, out y0, out y1, out y2, out y3);
            return;
        }

        DoubleFour(new DoubleOffset<TR>(read, offset), bytes, row, x, stride, count, out y0, out y1, out y2, out y3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe float DescribedDoubleOne(DescribedPlan plan, byte* row, double* x, int k) => plan.Wide switch
    {
        WideRead.Long => (float)DoubleOneBy(new LongRead(), 8, plan.Offset, row, x, k),
        WideRead.Int => (float)DoubleOneBy(new IntRead(), 4, plan.Offset, row, x, k),
        _ => (float)DoubleOneBy(new DoubleRead(), 8, plan.Offset, row, x, k),
    };

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe double DoubleOneBy<TR>(TR read, int bytes, double offset, byte* row, double* x, int k) where TR : struct, IDoubleRead =>
        offset == 0 ? DoubleOne(read, bytes, row, x, k) : DoubleOne(new DoubleOffset<TR>(read, offset), bytes, row, x, k);

    // Four rows of x (float64, `stride` apart: widened once a call) by `count` values of `bytes` bytes each: 8 values a step
    // made float64 in registers (TR), each multiplied with the four rows' matching values in float64 FMAs; two sums a row,
    // the last count mod 8 one at a time; each output its float64 sum rounded to float32 once. (x read as float32 and
    // widened in registers instead measured slower: 171 tokens a second on Qwen3-1.7B's prompts against 177.)
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void DoubleFour<TR>(TR read, int bytes, byte* row, double* x, int stride, int count, out float y0, out float y1, out float y2, out float y3)
        where TR : struct, IDoubleRead
    {
        nint r1 = stride, r2 = 2 * (nint)stride, r3 = 3 * (nint)stride;
        Vector256<double> a0 = default, a1 = default, a2 = default, a3 = default, b0 = default, b1 = default, b2 = default, b3 = default;
        int p = 0;
        for (; p + 8 <= count; p += 8)
        {
            var w0 = read.Four(row + (nint)p * bytes);
            var w1 = read.Four(row + (nint)(p + 4) * bytes);
            double* xp = x + p;
            a0 = Fma.MultiplyAdd(w0, Vector256.Load(xp), a0);
            b0 = Fma.MultiplyAdd(w1, Vector256.Load(xp + 4), b0);
            a1 = Fma.MultiplyAdd(w0, Vector256.Load(xp + r1), a1);
            b1 = Fma.MultiplyAdd(w1, Vector256.Load(xp + r1 + 4), b1);
            a2 = Fma.MultiplyAdd(w0, Vector256.Load(xp + r2), a2);
            b2 = Fma.MultiplyAdd(w1, Vector256.Load(xp + r2 + 4), b2);
            a3 = Fma.MultiplyAdd(w0, Vector256.Load(xp + r3), a3);
            b3 = Fma.MultiplyAdd(w1, Vector256.Load(xp + r3 + 4), b3);
        }

        double t0 = 0, t1 = 0, t2 = 0, t3 = 0;
        for (; p < count; p++)
        {
            double w = read.One(row + (nint)p * bytes);
            t0 = Math.FusedMultiplyAdd(w, x[p], t0);
            t1 = Math.FusedMultiplyAdd(w, x[r1 + p], t1);
            t2 = Math.FusedMultiplyAdd(w, x[r2 + p], t2);
            t3 = Math.FusedMultiplyAdd(w, x[r3 + p], t3);
        }

        y0 = (float)(Vector256.Sum(a0 + b0) + t0);
        y1 = (float)(Vector256.Sum(a1 + b1) + t1);
        y2 = (float)(Vector256.Sum(a2 + b2) + t2);
        y3 = (float)(Vector256.Sum(a3 + b3) + t3);
    }

    // DoubleFour for one row of x, 16 values a step in four sums.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe double DoubleOne<TR>(TR read, int bytes, byte* row, double* x, int k) where TR : struct, IDoubleRead
    {
        Vector256<double> a = default, b = default, c = default, d = default;
        int p = 0;
        for (; p + 16 <= k; p += 16)
        {
            byte* at = row + (nint)p * bytes;
            a = Fma.MultiplyAdd(read.Four(at), Vector256.Load(x + p), a);
            b = Fma.MultiplyAdd(read.Four(at + 4 * bytes), Vector256.Load(x + p + 4), b);
            c = Fma.MultiplyAdd(read.Four(at + 8 * bytes), Vector256.Load(x + p + 8), c);
            d = Fma.MultiplyAdd(read.Four(at + 12 * bytes), Vector256.Load(x + p + 12), d);
        }

        for (; p + 4 <= k; p += 4)
        {
            a = Fma.MultiplyAdd(read.Four(row + (nint)p * bytes), Vector256.Load(x + p), a);
        }

        double t = 0;
        for (; p < k; p++)
        {
            t = Math.FusedMultiplyAdd(read.One(row + (nint)p * bytes), x[p], t);
        }

        return Vector256.Sum((a + b) + (c + d)) + t;
    }

    // The described float32 lane over values of 2 or 4 bytes: four rows of x by `count` values of a stored row, dispatched on
    // how a value reads (and whether the description adds a code offset).
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe void DescribedSingleFour(DescribedPlan plan, byte* row, float* x, int stride, int count, out float y0, out float y1, out float y2, out float y3)
    {
        float offset = (float)plan.D.CodeOffset;
        switch (plan.Wide)
        {
            case WideRead.Half when plan.Half16 is { Shift: 16, Times: 1f, Special: 0, SignShift: 16 }:
                SingleFourBy(new HighHalfRead(), 2, offset, row, x, stride, count, out y0, out y1, out y2, out y3);
                break;
            case WideRead.Half:
                SingleFourBy(new HalfRead(plan.Half16), 2, offset, row, x, stride, count, out y0, out y1, out y2, out y3);
                break;
            case WideRead.Short:
                SingleFourBy(new ShortRead(plan.D.Code.Signed), 2, offset, row, x, stride, count, out y0, out y1, out y2, out y3);
                break;
            default:
                SingleFourBy(new SingleRead(), 4, offset, row, x, stride, count, out y0, out y1, out y2, out y3);
                break;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe void SingleFourBy<TR>(TR read, int bytes, float offset, byte* row, float* x, int stride, int count, out float y0, out float y1, out float y2, out float y3)
        where TR : struct, ISingleRead
    {
        if (offset == 0f)
        {
            SingleFour(read, bytes, row, x, stride, count, out y0, out y1, out y2, out y3);
            return;
        }

        SingleFour(new SingleOffset<TR>(read, offset), bytes, row, x, stride, count, out y0, out y1, out y2, out y3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe float DescribedSingleOne(DescribedPlan plan, byte* row, float* x, int k)
    {
        float offset = (float)plan.D.CodeOffset;
        return plan.Wide switch
        {
            WideRead.Half when plan.Half16 is { Shift: 16, Times: 1f, Special: 0, SignShift: 16 } => SingleOneBy(new HighHalfRead(), 2, offset, row, x, k),
            WideRead.Half => SingleOneBy(new HalfRead(plan.Half16), 2, offset, row, x, k),
            WideRead.Short => SingleOneBy(new ShortRead(plan.D.Code.Signed), 2, offset, row, x, k),
            _ => SingleOneBy(new SingleRead(), 4, offset, row, x, k),
        };
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static unsafe float SingleOneBy<TR>(TR read, int bytes, float offset, byte* row, float* x, int k) where TR : struct, ISingleRead =>
        offset == 0f ? SingleOne(read, bytes, row, x, k) : SingleOne(new SingleOffset<TR>(read, offset), bytes, row, x, k);

    // Four rows of x (`stride` apart) by `count` values of `bytes` bytes each: 16 values a step made float32 in registers
    // (TR), each multiplied with the four rows' matching values; two sums a row, the last count mod 16 one at a time.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void SingleFour<TR>(TR read, int bytes, byte* row, float* x, int stride, int count, out float y0, out float y1, out float y2, out float y3)
        where TR : struct, ISingleRead
    {
        nint r1 = stride, r2 = 2 * (nint)stride, r3 = 3 * (nint)stride;
        Vector256<float> a0 = default, a1 = default, a2 = default, a3 = default, b0 = default, b1 = default, b2 = default, b3 = default;
        int p = 0;
        for (; p + 16 <= count; p += 16)
        {
            var lo = read.Eight(row + (nint)p * bytes);
            var hi = read.Eight(row + (nint)(p + 8) * bytes);
            float* xp = x + p;
            a0 = Fma.MultiplyAdd(lo, Vector256.Load(xp), a0);
            b0 = Fma.MultiplyAdd(hi, Vector256.Load(xp + 8), b0);
            a1 = Fma.MultiplyAdd(lo, Vector256.Load(xp + r1), a1);
            b1 = Fma.MultiplyAdd(hi, Vector256.Load(xp + r1 + 8), b1);
            a2 = Fma.MultiplyAdd(lo, Vector256.Load(xp + r2), a2);
            b2 = Fma.MultiplyAdd(hi, Vector256.Load(xp + r2 + 8), b2);
            a3 = Fma.MultiplyAdd(lo, Vector256.Load(xp + r3), a3);
            b3 = Fma.MultiplyAdd(hi, Vector256.Load(xp + r3 + 8), b3);
        }

        float t0 = 0f, t1 = 0f, t2 = 0f, t3 = 0f;
        for (; p < count; p++)
        {
            float w = read.One(row + (nint)p * bytes);
            t0 = MathF.FusedMultiplyAdd(w, x[p], t0);
            t1 = MathF.FusedMultiplyAdd(w, x[r1 + p], t1);
            t2 = MathF.FusedMultiplyAdd(w, x[r2 + p], t2);
            t3 = MathF.FusedMultiplyAdd(w, x[r3 + p], t3);
        }

        y0 = Vector256.Sum(a0 + b0) + t0;
        y1 = Vector256.Sum(a1 + b1) + t1;
        y2 = Vector256.Sum(a2 + b2) + t2;
        y3 = Vector256.Sum(a3 + b3) + t3;
    }

    // SingleFour for one row of x, 32 values a step in four sums.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float SingleOne<TR>(TR read, int bytes, byte* row, float* x, int k) where TR : struct, ISingleRead
    {
        Vector256<float> a = default, b = default, c = default, d = default;
        int p = 0;
        for (; p + 32 <= k; p += 32)
        {
            byte* at = row + (nint)p * bytes;
            a = Fma.MultiplyAdd(read.Eight(at), Vector256.Load(x + p), a);
            b = Fma.MultiplyAdd(read.Eight(at + 8 * bytes), Vector256.Load(x + p + 8), b);
            c = Fma.MultiplyAdd(read.Eight(at + 16 * bytes), Vector256.Load(x + p + 16), c);
            d = Fma.MultiplyAdd(read.Eight(at + 24 * bytes), Vector256.Load(x + p + 24), d);
        }

        for (; p + 8 <= k; p += 8)
        {
            a = Fma.MultiplyAdd(read.Eight(row + (nint)p * bytes), Vector256.Load(x + p), a);
        }

        float t = 0f;
        for (; p < k; p++)
        {
            t = MathF.FusedMultiplyAdd(read.One(row + (nint)p * bytes), x[p], t);
        }

        return Vector256.Sum((a + b) + (c + d)) + t;
    }

    // The described float32 lane over codes of up to 8 bits (no integer arithmetic for their numbers): four rows of x (in
    // the spans' order) by one stored row, each group's scales and minimums read at once (DescribedPlan.Factor), each span's
    // codes made (DescribedPlan.Raw) and 8 at a time looked up in the table of their numbers, times their chunk's scale less
    // its minimum, each multiplied with the four rows; the row's last period from `tail`.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void DescribedCodesFour(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, float* x, nint stride,
        out float y0, out float y1, out float y2, out float y3)
    {
        int gpp = plan.GroupsPerPeriod, spans = plan.GroupValues / 32, c = plan.C;
        float* table = plan.CodeValues;
        Vector256<float> a0 = default, a1 = default, a2 = default, a3 = default;
        for (int g = 0; g < groups; g++)
        {
            int period = g / gpp, gp = g - period * gpp;
            byte* at = period < full ? row + (nint)period * plan.PeriodBytes : tail;
            plan.Factor(at, gp, Math.Min(8, chunks - g * 8), out var scale, out var min);
            for (int s = 0; s < spans; s++)
            {
                var codes = plan.Raw(at, gp * spans + s);
                for (int e = 0; e < 4; e++)
                {
                    var lane = Vector256.Create((32 * s + 8 * e) / c);
                    var w = Fma.MultiplyAdd(Avx2.GatherVector256(table, Avx2.ConvertToVector256Int32(EightAt(codes, e)), 4),
                        Avx2.PermuteVar8x32(scale, lane), Avx2.PermuteVar8x32(min, lane));
                    float* xp = x + (nint)g * plan.GroupValues + 32 * s + 8 * e;
                    a0 = Fma.MultiplyAdd(w, Vector256.Load(xp), a0);
                    a1 = Fma.MultiplyAdd(w, Vector256.Load(xp + stride), a1);
                    a2 = Fma.MultiplyAdd(w, Vector256.Load(xp + 2 * stride), a2);
                    a3 = Fma.MultiplyAdd(w, Vector256.Load(xp + 3 * stride), a3);
                }
            }
        }

        y0 = Vector256.Sum(a0);
        y1 = Vector256.Sum(a1);
        y2 = Vector256.Sum(a2);
        y3 = Vector256.Sum(a3);
    }

    // DescribedCodesFour for one row of x.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe float DescribedCodesOne(DescribedPlan plan, byte* row, byte* tail, int full, int groups, int chunks, float* x)
    {
        int gpp = plan.GroupsPerPeriod, spans = plan.GroupValues / 32, c = plan.C;
        float* table = plan.CodeValues;
        Vector256<float> a = default, b = default;
        for (int g = 0; g < groups; g++)
        {
            int period = g / gpp, gp = g - period * gpp;
            byte* at = period < full ? row + (nint)period * plan.PeriodBytes : tail;
            plan.Factor(at, gp, Math.Min(8, chunks - g * 8), out var scale, out var min);
            for (int s = 0; s < spans; s++)
            {
                var codes = plan.Raw(at, gp * spans + s);
                float* xp = x + (nint)g * plan.GroupValues + 32 * s;
                for (int e = 0; e < 4; e += 2)
                {
                    var lane = Vector256.Create((32 * s + 8 * e) / c);
                    var next = Vector256.Create((32 * s + 8 * e + 8) / c);
                    a = Fma.MultiplyAdd(Fma.MultiplyAdd(Avx2.GatherVector256(table, Avx2.ConvertToVector256Int32(EightAt(codes, e)), 4),
                        Avx2.PermuteVar8x32(scale, lane), Avx2.PermuteVar8x32(min, lane)), Vector256.Load(xp + 8 * e), a);
                    b = Fma.MultiplyAdd(Fma.MultiplyAdd(Avx2.GatherVector256(table, Avx2.ConvertToVector256Int32(EightAt(codes, e + 1)), 4),
                        Avx2.PermuteVar8x32(scale, next), Avx2.PermuteVar8x32(min, next)), Vector256.Load(xp + 8 * e + 8), b);
                }
            }
        }

        return Vector256.Sum(a + b);
    }

    // Bytes 8e..8e + 7 of a span's codes, in the low 8 bytes.
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static Vector128<byte> EightAt(Vector256<byte> codes, int e) => e switch
    {
        0 => codes.GetLower(),
        1 => Sse2.ShiftRightLogical128BitLane(codes.GetLower(), 8),
        2 => codes.GetUpper(),
        _ => Sse2.ShiftRightLogical128BitLane(codes.GetUpper(), 8),
    };

    /// <summary>
    /// A 16-bit float's reading by its bits (DescribedPlan.HalfOf) as vectors made once, a factor (a power of two: the
    /// integer lane's unit) folded in: the magnitude masked and shifted to float32's place, times the power of two that
    /// rebiases it and the factor, its specials' exponent set (none: never), its sign moved to bit 31 (none: masked off).
    /// </summary>
    private readonly struct HalfVectors((int SignShift, int Magnitude, int Shift, float Times, int Special) h, float factor)
    {
        private readonly Vector256<int> magnitude = Vector256.Create(h.Magnitude), special = Vector256.Create(h.Special == 0 ? int.MaxValue : h.Special),
            sign = Vector256.Create(h.SignShift == 0 ? 0 : 0x8000);
        private readonly Vector256<float> times = Vector256.Create(h.Times * factor);
        private readonly int shift = h.Shift;

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public Vector256<float> Of(Vector256<int> bits)
        {
            var m = bits & magnitude;
            var value = (Vector256.ShiftLeft(m, shift).AsSingle() * times).AsInt32();
            value |= Vector256.GreaterThanOrEqual(m, special) & Vector256.Create(0x7F800000);
            return (value | Vector256.ShiftLeft(bits & sign, 16)).AsSingle();
        }
    }

    // 8 chunks' 32 bytes (a chunk a vector) as the integer lane's 8 steps: step s holds dword s of each chunk, in lane order.
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static void Transpose8(Vector256<byte> r0, Vector256<byte> r1, Vector256<byte> r2, Vector256<byte> r3, Vector256<byte> r4, Vector256<byte> r5,
        Vector256<byte> r6, Vector256<byte> r7, out Vector256<byte> s0, out Vector256<byte> s1, out Vector256<byte> s2, out Vector256<byte> s3,
        out Vector256<byte> s4, out Vector256<byte> s5, out Vector256<byte> s6, out Vector256<byte> s7)
    {
        var t0 = Avx2.UnpackLow(r0.AsUInt32(), r1.AsUInt32());
        var t1 = Avx2.UnpackHigh(r0.AsUInt32(), r1.AsUInt32());
        var t2 = Avx2.UnpackLow(r2.AsUInt32(), r3.AsUInt32());
        var t3 = Avx2.UnpackHigh(r2.AsUInt32(), r3.AsUInt32());
        var t4 = Avx2.UnpackLow(r4.AsUInt32(), r5.AsUInt32());
        var t5 = Avx2.UnpackHigh(r4.AsUInt32(), r5.AsUInt32());
        var t6 = Avx2.UnpackLow(r6.AsUInt32(), r7.AsUInt32());
        var t7 = Avx2.UnpackHigh(r6.AsUInt32(), r7.AsUInt32());
        var u0 = Avx2.UnpackLow(t0.AsUInt64(), t2.AsUInt64());
        var u1 = Avx2.UnpackHigh(t0.AsUInt64(), t2.AsUInt64());
        var u2 = Avx2.UnpackLow(t1.AsUInt64(), t3.AsUInt64());
        var u3 = Avx2.UnpackHigh(t1.AsUInt64(), t3.AsUInt64());
        var u4 = Avx2.UnpackLow(t4.AsUInt64(), t6.AsUInt64());
        var u5 = Avx2.UnpackHigh(t4.AsUInt64(), t6.AsUInt64());
        var u6 = Avx2.UnpackLow(t5.AsUInt64(), t7.AsUInt64());
        var u7 = Avx2.UnpackHigh(t5.AsUInt64(), t7.AsUInt64());
        s0 = Avx2.Permute2x128(u0, u4, 0x20).AsByte();
        s4 = Avx2.Permute2x128(u0, u4, 0x31).AsByte();
        s1 = Avx2.Permute2x128(u1, u5, 0x20).AsByte();
        s5 = Avx2.Permute2x128(u1, u5, 0x31).AsByte();
        s2 = Avx2.Permute2x128(u2, u6, 0x20).AsByte();
        s6 = Avx2.Permute2x128(u2, u6, 0x31).AsByte();
        s3 = Avx2.Permute2x128(u3, u7, 0x20).AsByte();
        s7 = Avx2.Permute2x128(u3, u7, 0x31).AsByte();
    }

    // 4 spans of two 16-value chunks each (chunk 2q in span q's low half, 2q + 1 in its high half) as the integer lane's 4
    // steps: step s holds dword s of each of the 8 chunks, in lane order.
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static void Transpose4(Vector256<byte> r0, Vector256<byte> r1, Vector256<byte> r2, Vector256<byte> r3,
        out Vector256<byte> s0, out Vector256<byte> s1, out Vector256<byte> s2, out Vector256<byte> s3)
    {
        var pairs = Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7);
        var w0 = Avx2.PermuteVar8x32(r0.AsInt32(), pairs).AsUInt64();
        var w1 = Avx2.PermuteVar8x32(r1.AsInt32(), pairs).AsUInt64();
        var w2 = Avx2.PermuteVar8x32(r2.AsInt32(), pairs).AsUInt64();
        var w3 = Avx2.PermuteVar8x32(r3.AsInt32(), pairs).AsUInt64();
        var a = Avx2.UnpackLow(w0, w1);
        var b = Avx2.UnpackHigh(w0, w1);
        var c = Avx2.UnpackLow(w2, w3);
        var d = Avx2.UnpackHigh(w2, w3);
        s0 = Avx2.Permute2x128(a, c, 0x20).AsByte();
        s2 = Avx2.Permute2x128(a, c, 0x31).AsByte();
        s1 = Avx2.Permute2x128(b, d, 0x20).AsByte();
        s3 = Avx2.Permute2x128(b, d, 0x31).AsByte();
    }
}
