// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Idrak.Abstraction.Devices.Cpu;

// The experiment's CPU lanes for weights stored in the file's layout (ProductRoutes): y = x · W for W
// stored as n rows of k values of any type, its description choosing how a row widens (IRowWiden: a float's bits, an
// integer's, a scaled block's codes; any other type through its own conversion), with a scale per row or none. Two lanes,
// each written once for every type (the JIT compiles one copy per widening): the row lane, each output's row widened once
// and dotted with every row of x (decoding-sized products), and the panel lane, a tile of rows widened into a [columns, k]
// panel and multiplied by the float32 product reading it transposed (prompt-sized products). The generic route takes the
// row lane up to 8 rows and the panel lane above; the measured route the faster of the two at the shape, measured once
// (Faster).
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
        int chosen = ProductRoutes.Current == ProductRoute.Generic
            ? m <= Int8DirectRows ? RowLaneId : PanelLaneId
            : StoredLanes.TryGetValue((type.Name, LaneKey(type.Name, m, n, k).Rows, n, k), out int known) ? known
            : MeasureStoredLanes(widening, lane, x, y, m, n, k);
        RunLane(chosen, widening, lane, x, y, m, n, k);
        return true;
    }

    // The stored products' lanes: each output's row widened once (decoding-sized), a panel of rows widened and multiplied
    // transposed, the whole weight widened once and the float32 product (today's prompt path: ExpandedLane), the tiles.
    private const int RowLaneId = 0, PanelLaneId = 1, ExpandedLaneId = 2, TilesLaneId = 3;

    private static readonly string[] LaneNames = ["row lane", "panel lane", "today's path (widened once, float32 product)",
        "tiles (panels widened in the product)"];

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
}
