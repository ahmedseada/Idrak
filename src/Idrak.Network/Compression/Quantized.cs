// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.InteropServices;
using Idrak.Networking.Abstractions;
using Idrak.Networking.Training;

namespace Idrak.Networking.Compression;

/// <summary>
/// "int8" in <see cref="GradientCompressions"/> (plan 37, N0): every gradient value travels as one byte, a whole number
/// from -127 to 127 times its piece's scale; with error feedback (<see cref="ClusterOptions.ErrorFeedback"/>) each rank
/// adds the rounding error it did not send to its next step's values, so nothing drifts.
/// </summary>
internal sealed class Int8Compression : IGradientCompression
{
    public string Name => "int8";

    public string Summary => "every gradient value travels as one byte, an int8 times its piece's scale; with error feedback (IDRAK_CLUSTER_ERROR_FEEDBACK) each rank's rounding error goes with its next step";

    public IGradientCompressor Create(ICollectives collectives, IReadOnlyList<Tensor> parameters, ClusterOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return new QuantizedCompressor(Name, QuantizedFormat.Int8, collectives, parameters, options.ErrorFeedback);
    }
}

/// <summary>
/// "fp8" in <see cref="GradientCompressions"/> (plan 37, N0): every gradient value travels as one byte, an 8-bit float
/// (E4M3: 4 exponent bits, 3 mantissa bits, no infinities, largest 448) times its piece's scale; with error feedback each
/// rank adds the rounding error it did not send to its next step's values. Against int8, small values keep more of their
/// size and large ones less (3 bits of mantissa at every size, against int8's even steps).
/// </summary>
internal sealed class Float8Compression : IGradientCompression
{
    public string Name => "fp8";

    public string Summary => "every gradient value travels as one byte, an fp8 (E4M3) times its piece's scale; with error feedback (IDRAK_CLUSTER_ERROR_FEEDBACK) each rank's rounding error goes with its next step";

    public IGradientCompressor Create(ICollectives collectives, IReadOnlyList<Tensor> parameters, ClusterOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return new QuantizedCompressor(Name, QuantizedFormat.Float8E4M3, collectives, parameters, options.ErrorFeedback);
    }
}

/// <summary>How <see cref="QuantizedCompressor"/> writes a value in its byte.</summary>
internal enum QuantizedFormat
{
    /// <summary>A whole number from -127 to 127 (two's complement).</summary>
    Int8,

    /// <summary>An 8-bit float, E4M3 (<see cref="Float8E4M3"/>).</summary>
    Float8E4M3,
}

/// <summary>
/// A rank's int8 or fp8 compressor (plan 37, N0 and N0b). This rank's values are weighted and its error-feedback residual
/// added; each value is then rounded to the nearest code of its scale (largest magnitude / 127 for int8, / 448 for fp8;
/// ties to even), one byte a value. Every sum is made in rank order (rule 86: an order fixed by the ranks), and every rank
/// ends with the same decoded values, bit for bit.
/// <para>
/// Two ways, the one that sends fewer bytes taken bucket by bucket (<see cref="Exchanges"/>, the same on every rank).
/// All-gathered: each rank's scale for the whole bucket and its codes go to every rank, and every rank sums them all;
/// a rank sends (W - 1) · (n + 4) bytes for a bucket of n values, one rounding. Exchanged, where the collectives exchange
/// blocks (<see cref="IAllToAllHostValueCollectives"/>, Idrak.Network's own and a simulated rank over them): the bucket is
/// cut in one piece per rank (as a ring cuts it), each piece with its own scale; each rank sends each owner its piece's
/// codes (an all-to-all), the owner decodes them and sums them in rank order, rounds the sum to codes again, and the
/// reduced pieces' codes are all-gathered and decoded by every rank: about 2 (W - 1) / W · n bytes, two roundings. Against
/// a float32 ring all-reduce's 2 (W - 1) / W · 4n, either is a quarter on 2 ranks, where all-gathering costs no more and
/// rounds once, and the exchange keeps the quarter on any number of ranks beyond.
/// </para>
/// <para>
/// Error feedback (<see cref="ClusterOptions.ErrorFeedback"/>): what a rank's codes did not carry (its values less the
/// decoded codes) and, for the pieces it owns, what the reduced sum's codes did not carry are its residual, added to its
/// next step's values, so the sum of what is sent over the steps differs from the sum of the gradients by one step's
/// rounding only. The residual is one float per trained value, allocated once per parameter on first use, as top-k's. A
/// bucket with a NaN or an infinity travels as a NaN scale, so the step sees it, and the residuals it touched are dropped.
/// </para>
/// </summary>
internal sealed class QuantizedCompressor(string name, QuantizedFormat format, ICollectives collectives, IReadOnlyList<Tensor> parameters, bool feedback) : IGradientCompressor
{
    private readonly HostValues _values = new(collectives);
    private readonly float[]?[] _residual = new float[]?[parameters.Count];
    private readonly float _largestCode = format == QuantizedFormat.Int8 ? 127f : Float8E4M3.MaxValue;

    public string Name => name;

    /// <summary>The floats a block of <paramref name="values"/> codes takes: its scale, then its codes four to a float.</summary>
    public static int Slots(int values) => 1 + ((values + 3) / 4);

    /// <summary>Where piece <paramref name="piece"/> of a bucket of <paramref name="values"/> values starts, of <paramref name="world"/> pieces (as a ring cuts it).</summary>
    public static int PieceStart(int values, int world, int piece) => (piece * (values / world)) + Math.Min(piece, values % world);

    /// <summary>The values of piece <paramref name="piece"/> of a bucket of <paramref name="values"/> values, of <paramref name="world"/> pieces.</summary>
    public static int PieceSize(int values, int world, int piece) => (values / world) + (piece < values % world ? 1 : 0);

    /// <summary>
    /// Whether a bucket of <paramref name="values"/> values goes in two stages (an all-to-all, then an all-gather of the
    /// owners' sums) rather than all-gathered: where the collectives exchange blocks and that sends fewer bytes (two blocks of
    /// a piece's codes to each other rank, against one of the whole bucket's). Never on 2 ranks, where all-gathering costs no
    /// more and rounds once; from 3 ranks on for a bucket of more than a few values.
    /// </summary>
    public bool Exchanges(int values) =>
        collectives.WorldSize > 1 && _values.CanAllToAll && 2 * Slots(PieceSize(values, collectives.WorldSize, 0)) < Slots(values);

    public void Reduce(Span<float> values, float weight, ReadOnlySpan<GradientPiece> pieces)
    {
        if (values.IsEmpty)
        {
            return;
        }

        if (Exchanges(values.Length))
        {
            ReduceExchanged(values, weight, pieces);
        }
        else
        {
            ReduceGathered(values, weight, pieces);
        }
    }

    public void Forget(ReadOnlySpan<GradientPiece> pieces)
    {
        foreach (var piece in pieces)
        {
            _residual[piece.Parameter]?.AsSpan(piece.Offset, piece.Count).Clear();
        }
    }

    public void Dispose() => _values.Dispose();

    // Every rank's scale for the whole bucket and its codes all-gathered; every rank sums them all in rank order.
    private void ReduceGathered(Span<float> values, float weight, ReadOnlySpan<GradientPiece> pieces)
    {
        int n = values.Length, world = collectives.WorldSize, rank = collectives.Rank, per = Slots(n), bytes = 4 * (per - 1);
        var all = ArrayPool<float>.Shared.Rent(checked(per * world));
        var codes = ArrayPool<byte>.Shared.Rent(bytes);
        try
        {
            var mine = all.AsSpan(rank * per, per);
            mine.Clear();                                            // a rank with no rows sends a zero scale and zero codes
            if (weight != 0f)
            {
                var own = codes.AsSpan(0, bytes);
                own[n..].Clear();                                    // the last float's unused bytes travel as zeros
                Prepare(values, weight, pieces);
                mine[0] = Quantize(values, own[..n]);
                Settle(values, mine[0], pieces);
                Pack(own, mine[1..]);
            }

            _values.AllGather(all.AsSpan(0, per * world), per);
            values.Clear();
            for (int r = 0; r < world; r++)                          // rank order: the same sum on every rank
            {
                AddBlock(values, all.AsSpan(r * per, per), codes.AsSpan(0, bytes));
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(all);
            ArrayPool<byte>.Shared.Return(codes);
        }
    }

    // Two stages: each owner's piece from every rank (all-to-all), summed by the owner in rank order and rounded again;
    // the owners' codes all-gathered and decoded by every rank.
    private void ReduceExchanged(Span<float> values, float weight, ReadOnlySpan<GradientPiece> pieces)
    {
        int n = values.Length, world = collectives.WorldSize, rank = collectives.Rank;
        int largest = PieceSize(n, world, 0), per = Slots(largest), bytes = 4 * (per - 1);
        var send = ArrayPool<float>.Shared.Rent(checked(per * world));
        var receive = ArrayPool<float>.Shared.Rent(per * world);
        var codes = ArrayPool<byte>.Shared.Rent(bytes);
        var sum = ArrayPool<float>.Shared.Rent(Math.Max(1, largest));
        try
        {
            var outgoing = send.AsSpan(0, per * world);
            outgoing.Clear();                                        // a rank with no rows sends zero scales and zero codes
            if (weight != 0f)
            {
                Prepare(values, weight, pieces);
                bool finite = true;
                for (int p = 0; p < world; p++)
                {
                    var block = outgoing.Slice(p * per, per);
                    var own = codes.AsSpan(0, bytes);
                    own.Clear();
                    int size = PieceSize(n, world, p);
                    block[0] = Quantize(values.Slice(PieceStart(n, world, p), size), own[..size]);
                    finite &= !float.IsNaN(block[0]);
                    Pack(own, block[1..]);
                }

                if (!finite)
                {
                    for (int p = 0; p < world; p++)
                    {
                        outgoing[p * per] = float.NaN;               // one NaN makes the whole bucket NaN, as gathered
                    }
                }

                Settle(values, finite ? 0f : float.NaN, pieces);
            }

            var incoming = receive.AsSpan(0, per * world);
            _values.AllToAll(outgoing, incoming, per);

            // This rank's piece: every rank's codes of it, summed in rank order, then rounded to codes of its own scale.
            int start = PieceStart(n, world, rank), mine = PieceSize(n, world, rank);
            var reduced = sum.AsSpan(0, mine);
            reduced.Clear();
            for (int r = 0; r < world; r++)
            {
                AddBlock(reduced, incoming.Slice(r * per, per), codes.AsSpan(0, bytes));
            }

            var gathered = outgoing;                                 // the sent blocks are done with: the gather's buffer
            var block2 = gathered.Slice(rank * per, per);
            var own2 = codes.AsSpan(0, bytes);
            own2.Clear();
            block2[0] = Quantize(reduced, own2[..mine]);
            Pack(own2, block2[1..]);
            if (feedback && !float.IsNaN(block2[0]))
            {
                KeepAlso(reduced, start, pieces);                    // the owner keeps what its sum's codes did not carry
            }

            _values.AllGather(gathered, per);
            values.Clear();
            for (int p = 0; p < world; p++)
            {
                int size = PieceSize(n, world, p);
                AddBlock(values.Slice(PieceStart(n, world, p), size), gathered.Slice(p * per, per), codes.AsSpan(0, bytes));
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(send);
            ArrayPool<float>.Shared.Return(receive);
            ArrayPool<byte>.Shared.Return(codes);
            ArrayPool<float>.Shared.Return(sum);
        }
    }

    // values += a block's codes times its scale (one scale, then the codes four to a float); nothing for a zero scale.
    private void AddBlock(Span<float> values, Span<float> block, Span<byte> scratch)
    {
        float scale = block[0];
        if (scale == 0f)
        {
            return;                                                  // every value of it is zero
        }

        var received = Unpack(block[1..], scratch)[..values.Length];
        if (format == QuantizedFormat.Int8)
        {
            AddInt8(values, received, scale);
        }
        else
        {
            AddFloat8(values, received, scale);
        }
    }

    // This rank's values weighted, with its residual added.
    private void Prepare(Span<float> values, float weight, ReadOnlySpan<GradientPiece> pieces)
    {
        DenseCompressor.Scale(values, weight);
        if (feedback)
        {
            foreach (var piece in pieces)
            {
                if (_residual[piece.Parameter] is { } kept)
                {
                    Add(values.Slice(piece.At, piece.Count), kept.AsSpan(piece.Offset, piece.Count));
                }
            }
        }
    }

    // After the codes: a NaN scale drops the residual (a step that sees a NaN or infinity carries nothing on); else what
    // the codes did not carry, left in `values`, becomes the residual.
    private void Settle(ReadOnlySpan<float> values, float scale, ReadOnlySpan<GradientPiece> pieces)
    {
        if (float.IsNaN(scale))
        {
            Forget(pieces);
        }
        else
        {
            Keep(values, pieces);
        }
    }

    /// <summary>
    /// <paramref name="values"/> as codes of one scale, which it returns: NaN (codes zero) when a value is NaN or infinite, 0
    /// (codes zero, the values kept whole) when all are zero or too small to scale; else each value is left as what its code
    /// does not carry.
    /// </summary>
    private float Quantize(Span<float> values, Span<byte> codes)
    {
        float largest = LargestMagnitude(values);
        if (!float.IsFinite(largest))
        {
            codes.Clear();
            return float.NaN;
        }

        float inverse = _largestCode / largest;
        if (largest == 0f || !float.IsFinite(inverse))
        {
            codes.Clear();
            return 0f;
        }

        float scale = largest / _largestCode;
        if (format == QuantizedFormat.Int8)
        {
            EncodeInt8(values, codes, inverse, scale);
        }
        else
        {
            EncodeFloat8(values, codes, inverse, scale);
        }

        return scale;
    }

    // The values not sent become the residual (error feedback on).
    private void Keep(ReadOnlySpan<float> values, ReadOnlySpan<GradientPiece> pieces)
    {
        if (!feedback)
        {
            return;
        }

        foreach (var piece in pieces)
        {
            var residual = _residual[piece.Parameter] ??= new float[parameters[piece.Parameter].Size];
            values.Slice(piece.At, piece.Count).CopyTo(residual.AsSpan(piece.Offset, piece.Count));
        }
    }

    // The owned piece's second rounding (values at `start` of the bucket) added to the residual of the parameters it covers.
    private void KeepAlso(ReadOnlySpan<float> rest, int start, ReadOnlySpan<GradientPiece> pieces)
    {
        foreach (var piece in pieces)
        {
            int from = Math.Max(piece.At, start), to = Math.Min(piece.At + piece.Count, start + rest.Length);
            if (from >= to)
            {
                continue;
            }

            var residual = _residual[piece.Parameter] ??= new float[parameters[piece.Parameter].Size];
            Add(residual.AsSpan(piece.Offset + (from - piece.At), to - from), rest.Slice(from - start, to - from));
        }
    }

    /// <summary>
    /// <paramref name="values"/> as int8 codes: round(clamp(v · inverse, ±127)), ties to even; each value is left as what
    /// the code does not carry, v - code · scale (the receivers' decoding, bit for bit). A vector loop and a scalar tail,
    /// alike to the bit.
    /// </summary>
    internal static void EncodeInt8(Span<float> values, Span<byte> codes, float inverse, float scale)
    {
        var signed = MemoryMarshal.Cast<byte, sbyte>(codes);
        int i = 0, k = Vector<int>.Count;
        if (Vector.IsHardwareAccelerated && values.Length >= 4 * k)
        {
            var f = new Vector<float>(inverse);
            var s = new Vector<float>(scale);
            var hi = new Vector<float>(127f);
            var lo = new Vector<float>(-127f);
            for (; i <= values.Length - (4 * k); i += 4 * k)
            {
                var v0 = new Vector<float>(values[i..]);
                var v1 = new Vector<float>(values[(i + k)..]);
                var v2 = new Vector<float>(values[(i + (2 * k))..]);
                var v3 = new Vector<float>(values[(i + (3 * k))..]);
                var r0 = Vector.Round(Vector.Clamp(v0 * f, lo, hi));
                var r1 = Vector.Round(Vector.Clamp(v1 * f, lo, hi));
                var r2 = Vector.Round(Vector.Clamp(v2 * f, lo, hi));
                var r3 = Vector.Round(Vector.Clamp(v3 * f, lo, hi));
                Vector.Narrow(Vector.Narrow(Vector.ConvertToInt32(r0), Vector.ConvertToInt32(r1)), Vector.Narrow(Vector.ConvertToInt32(r2), Vector.ConvertToInt32(r3)))
                    .CopyTo(signed[i..]);
                (v0 - (r0 * s)).CopyTo(values[i..]);
                (v1 - (r1 * s)).CopyTo(values[(i + k)..]);
                (v2 - (r2 * s)).CopyTo(values[(i + (2 * k))..]);
                (v3 - (r3 * s)).CopyTo(values[(i + (3 * k))..]);
            }
        }

        for (; i < values.Length; i++)
        {
            float r = MathF.Round(Math.Clamp(values[i] * inverse, -127f, 127f));
            signed[i] = (sbyte)r;
            values[i] -= r * scale;
        }
    }

    /// <summary>
    /// <paramref name="values"/> as fp8 E4M3 codes of v · inverse (at most 448 in magnitude by the scale; nearest, ties to
    /// even); each value is left as v - decoded · scale.
    /// </summary>
    internal static void EncodeFloat8(Span<float> values, Span<byte> codes, float inverse, float scale)
    {
        for (int i = 0; i < values.Length; i++)
        {
            byte code = Float8E4M3.Encode(values[i] * inverse);
            codes[i] = code;
            values[i] -= Float8E4M3.Decode(code) * scale;
        }
    }

    /// <summary>values += code · scale for each int8 code: a vector loop (codes widened to floats) and a scalar tail, alike to the bit.</summary>
    internal static void AddInt8(Span<float> values, ReadOnlySpan<byte> codes, float scale)
    {
        var signed = MemoryMarshal.Cast<byte, sbyte>(codes);
        int i = 0, k = Vector<int>.Count;
        if (Vector.IsHardwareAccelerated && values.Length >= Vector<sbyte>.Count)
        {
            var s = new Vector<float>(scale);
            for (; i <= values.Length - Vector<sbyte>.Count; i += Vector<sbyte>.Count)
            {
                Vector.Widen(new Vector<sbyte>(signed[i..]), out Vector<short> low, out Vector<short> high);
                Vector.Widen(low, out Vector<int> a, out Vector<int> b);
                Vector.Widen(high, out Vector<int> c, out Vector<int> d);
                AddScaled(values[i..], a, s);
                AddScaled(values[(i + k)..], b, s);
                AddScaled(values[(i + (2 * k))..], c, s);
                AddScaled(values[(i + (3 * k))..], d, s);
            }
        }

        for (; i < values.Length; i++)
        {
            values[i] += signed[i] * scale;
        }
    }

    /// <summary>values += decoded · scale for each fp8 E4M3 code.</summary>
    internal static void AddFloat8(Span<float> values, ReadOnlySpan<byte> codes, float scale)
    {
        for (int i = 0; i < values.Length; i++)
        {
            values[i] += Float8E4M3.Decode(codes[i]) * scale;
        }
    }

    /// <summary>The largest magnitude of <paramref name="values"/>, or NaN when one is NaN or infinite.</summary>
    internal static float LargestMagnitude(ReadOnlySpan<float> values)
    {
        int i = 0;
        float largest = 0f, check = 0f;
        if (Vector.IsHardwareAccelerated && values.Length >= Vector<float>.Count)
        {
            var most = Vector<float>.Zero;
            var finite = Vector<float>.Zero;                         // v - v: 0 for a finite value, NaN otherwise
            for (; i <= values.Length - Vector<float>.Count; i += Vector<float>.Count)
            {
                var v = new Vector<float>(values[i..]);
                most = Vector.Max(most, Vector.Abs(v));
                finite += v - v;
            }

            for (int j = 0; j < Vector<float>.Count; j++)
            {
                largest = Math.Max(largest, most[j]);
                check += finite[j];
            }
        }

        for (; i < values.Length; i++)
        {
            largest = Math.Max(largest, MathF.Abs(values[i]));
            check += values[i] - values[i];
        }

        return check == 0f ? largest : float.NaN;
    }

    private static void AddScaled(Span<float> values, Vector<int> codes, Vector<float> scale) =>
        (new Vector<float>(values) + (Vector.ConvertToSingle(codes) * scale)).CopyTo(values);

    private static void Add(Span<float> values, ReadOnlySpan<float> other)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            for (; i <= values.Length - Vector<float>.Count; i += Vector<float>.Count)
            {
                (new Vector<float>(values[i..]) + new Vector<float>(other[i..])).CopyTo(values[i..]);
            }
        }

        for (; i < values.Length; i++)
        {
            values[i] += other[i];
        }
    }

    // The codes into floats, four to a float, the first code in the lowest byte of the float's little-endian bits, so
    // the bytes on the wire are the codes in order on any machine (the float32 wire sends a float's bits unchanged).
    private static void Pack(ReadOnlySpan<byte> codes, Span<float> slots)
    {
        if (BitConverter.IsLittleEndian)
        {
            codes.CopyTo(MemoryMarshal.AsBytes(slots));
            return;
        }

        for (int j = 0; j < slots.Length; j++)
        {
            slots[j] = BitConverter.UInt32BitsToSingle(BinaryPrimitives.ReadUInt32LittleEndian(codes[(4 * j)..]));
        }
    }

    private static ReadOnlySpan<byte> Unpack(Span<float> slots, Span<byte> scratch)
    {
        if (BitConverter.IsLittleEndian)
        {
            return MemoryMarshal.AsBytes(slots);
        }

        for (int j = 0; j < slots.Length; j++)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(scratch[(4 * j)..], BitConverter.SingleToUInt32Bits(slots[j]));
        }

        return scratch;
    }
}

/// <summary>
/// The 8-bit float E4M3 (OCP 8-bit floating point, "E4M3FN", PyTorch's <c>float8_e4m3fn</c>): a sign bit, 4 exponent bits
/// (bias 7) and 3 mantissa bits; no infinities; S.1111.111 is NaN; the largest finite value is 448 (S.1111.110), the
/// smallest normal 2^-6, the smallest subnormal 2^-9. <see cref="Encode"/> rounds to the nearest code, ties to even, and
/// saturates a finite value or an infinity beyond 448 to ±448 (the "satfinite" conversion); NaN encodes as NaN.
/// </summary>
internal static class Float8E4M3
{
    /// <summary>The largest finite value.</summary>
    public const float MaxValue = 448f;

    private const uint MinNormalBits = 0x3C800000u;                  // 2^-6
    private const uint MaxValueBits = 0x43E00000u;                   // 448

    private static readonly float[] Values = Table();

    /// <summary>The value of <paramref name="code"/> (a NaN code as a quiet NaN of its sign).</summary>
    public static float Decode(byte code) => Values[code];

    /// <summary>The code nearest <paramref name="value"/> (ties to even), ±448 for larger magnitudes, NaN for NaN.</summary>
    public static byte Encode(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        uint sign = (bits >> 24) & 0x80u;
        uint magnitude = bits & 0x7FFFFFFFu;
        if (magnitude > 0x7F800000u)
        {
            return (byte)(sign | 0x7Fu);                             // NaN
        }

        if (magnitude >= MaxValueBits)
        {
            return (byte)(sign | 0x7Eu);                             // 448 and beyond, infinity too: ±448
        }

        if (magnitude < MinNormalBits)
        {
            // A subnormal: m · 2^-9 with m from 0 to 8 (8 is 2^-6, the smallest normal's code). Scaling by 2^9 is exact.
            uint m = (uint)MathF.Round(BitConverter.UInt32BitsToSingle(magnitude) * 512f);
            return (byte)(sign | m);
        }

        // A normal: the exponent rebiased (127 to 7), the mantissa's top 3 bits, rounded on the other 20, ties to even;
        // a carry out of the mantissa steps the exponent, as the code's bits run on.
        uint code = (((magnitude >> 23) - 120u) << 3) | ((magnitude >> 20) & 7u);
        uint rest = magnitude & 0xFFFFFu;
        if (rest > 0x80000u || (rest == 0x80000u && (code & 1u) != 0))
        {
            code++;
        }

        return (byte)(sign | code);
    }

    private static float[] Table()
    {
        var table = new float[256];
        for (int code = 0; code < 256; code++)
        {
            int exponent = (code >> 3) & 15, mantissa = code & 7;
            float magnitude = exponent == 15 && mantissa == 7 ? float.NaN
                : exponent == 0 ? mantissa * (1f / 512f)                                      // m · 2^-9
                : BitConverter.UInt32BitsToSingle((uint)(exponent + 120) << 23 | (uint)mantissa << 20);
            table[code] = (code & 0x80) != 0 ? -magnitude : magnitude;
        }

        return table;
    }
}
