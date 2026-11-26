// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;

namespace Idrak.Abstraction;

/// <summary>
/// A bit plane of a block's codes (<see cref="TypeDescription.Planes"/>): fields of <paramref name="Bits"/> bits from byte
/// <paramref name="Offset"/>, in groups of <paramref name="Group"/> values. A group takes Group · Bits / 8 bytes; value w of
/// a group lies in byte w mod (Group · Bits / 8) of it, at bit Bits · ⌊w / (Group · Bits / 8)⌋ (ggml's split order: Q4_K's
/// codes are a plane of 4 bits in groups of 64, Q6_K's high bits a plane of 2 bits in groups of 128).
/// </summary>
/// <param name="Offset">The plane's first byte in the block.</param>
/// <param name="Bits">Bits a field takes (1, 2, 4 or 8; up to 64 in a stream or listed; a digit's number's in a plane of digits).</param>
/// <param name="Group">Values a group holds (a power of two).</param>
/// <param name="Stride">
/// Bytes from a group's first byte to the next group's, where groups do not follow one another (other fields between them,
/// as IQ2_XXS's grid indices lie between its signs and scales); 0: the group's own bytes.
/// </param>
/// <param name="Stream">
/// Whether the fields are a continuous stream of bits from the plane's first byte (field v at bit v · Bits, little-endian),
/// each field read at its bit position, across byte boundaries where it falls so (codes of 3, 5, 6 or 7 bits packed tight);
/// Group and Stride then say nothing.
/// </param>
/// <param name="Places">
/// Each field's place in the block, listed (field v at Places[v]: its byte and its lowest bit there, the field running into
/// the next bytes where its bits pass the byte's; a plane of digits: its byte and its digit), where no rule of Offset, Group,
/// Stride or the stream places them (ggml's IQ1_S high index bits, 3 a field in 16-bit words of 4; IQ2_XS's 9-bit indices,
/// one a 16-bit word); null: by the rule, Offset and the rest saying where.
/// </param>
/// <param name="Radix">
/// 0: each field is Bits bits. 2 or more: each field is a digit of base Radix of its byte read as a fraction, most
/// significant first: digit i of byte b is ⌊Radix · ((b · Radix^i) mod 256) / 256⌋, its byte and i from
/// <paramref name="Places"/> (ggml's TQ1_0: five base-3 digits a byte, b = ⌈n · 256 / 243⌉ of the digits' number n); Bits
/// the bits a digit's number takes (Radix − 1 fits them).
/// </param>
public sealed record CodePlane(int Offset, int Bits, int Group, int Stride = 0, bool Stream = false, IReadOnlyList<FieldPlace>? Places = null, int Radix = 0)
{
    /// <summary>Digits of <see cref="Radix"/> a byte holds as a fraction (the largest n with Radix^n at most 256; 0 for a plane of bits).</summary>
    public int Digits
    {
        get
        {
            int n = 0;
            for (int power = Radix; Radix >= 2 && power <= 256; power *= Radix)
            {
                n++;
            }

            return n;
        }
    }

    /// <summary>
    /// Digit <paramref name="digit"/> (0 the most significant) of <paramref name="value"/> read as a fraction in base
    /// <see cref="Radix"/>: ⌊Radix · ((value · Radix^digit) mod 256) / 256⌋.
    /// </summary>
    public int Digit(byte value, int digit)
    {
        int power = 1;
        for (int i = 0; i < digit; i++)
        {
            power *= Radix;
        }

        return (value * power & 0xFF) * Radix >> 8;
    }

    /// <summary>
    /// The byte of value <paramref name="value"/>'s field in the block and the field's lowest bit in it (a stream's or a
    /// listed field may run into the next bytes; a plane of digits: the digit).
    /// </summary>
    public (int Byte, int Shift) Place(int value)
    {
        if (Places is { } places)
        {
            var place = places[value];
            return (place.Byte, place.Shift);
        }

        if (Stream)
        {
            long bit = 8L * Offset + (long)value * Bits;
            return ((int)(bit >> 3), (int)(bit & 7));
        }

        int per = Group * Bits / 8, w = value % Group;
        return (Offset + value / Group * (Stride > 0 ? Stride : per) + w % per, Bits * (w / per));
    }
}

/// <summary>
/// Where a listed field of a code plane lies (<see cref="CodePlane.Places"/>): byte <paramref name="Byte"/> of the block,
/// from its bit <paramref name="Shift"/> (a plane of digits: its digit <paramref name="Shift"/>, 0 the most significant).
/// </summary>
/// <param name="Byte">The field's byte in the block.</param>
/// <param name="Shift">The field's lowest bit in the byte (0 to 7), or a digit's place.</param>
public readonly record struct FieldPlace(int Byte, int Shift);

/// <summary>
/// A piece of a block's scale or minimum (<see cref="BlockScale.Pieces"/>): <paramref name="Bits"/> bits at bit
/// <paramref name="Shift"/> of byte <paramref name="Offset"/> of the block; a value's pieces are listed from its low bits up.
/// </summary>
/// <param name="Offset">The byte in the block.</param>
/// <param name="Shift">The piece's lowest bit in the byte.</param>
/// <param name="Bits">Bits the piece takes.</param>
public readonly record struct ScalePiece(int Offset, int Shift, int Bits)
{
    /// <summary>Bits [<paramref name="first"/>, + <paramref name="count"/>) of the bytes from byte <paramref name="at"/>, as pieces a byte each (its low bits first).</summary>
    /// <param name="at">The first byte.</param>
    /// <param name="first">The first bit, counted from <paramref name="at"/>'s lowest.</param>
    /// <param name="count">The bits.</param>
    public static ScalePiece[] Run(int at, int first, int count)
    {
        var pieces = new List<ScalePiece>();
        while (count > 0)
        {
            int shift = first % 8, take = Math.Min(count, 8 - shift);
            pieces.Add(new ScalePiece(at + first / 8, shift, take));
            first += take;
            count -= take;
        }

        return [.. pieces];
    }
}

/// <summary>What a field of bits is as a number (<see cref="NumberType"/>).</summary>
public enum NumberKind
{
    /// <summary>A whole number: its bits, two's complement when signed.</summary>
    Integer,

    /// <summary>A float: a sign bit when signed (the highest), then the exponent's bits, then the mantissa's.</summary>
    Float,

    /// <summary>A code standing for its entry of a table.</summary>
    Table,
}

/// <summary>Which codes of a float are not finite values (<see cref="NumberType"/>).</summary>
public enum FloatSpecials
{
    /// <summary>IEEE 754's: the largest exponent is an infinity (mantissa 0) or a NaN (any other mantissa).</summary>
    Ieee,

    /// <summary>Only the code with every exponent and mantissa bit set is a NaN; no infinity (fp8 E4M3fn, E8M0).</summary>
    AllOnesNaN,

    /// <summary>None: every code is a finite value (fp4 E2M1).</summary>
    None,
}

/// <summary>
/// How a field of <paramref name="Bits"/> bits reads as a number: a whole number (two's complement when
/// <paramref name="Signed"/>); a float of a sign bit (when <paramref name="Signed"/>, the highest), <paramref name="Exponent"/>
/// exponent bits biased by <paramref name="ExponentBias"/> and <paramref name="Mantissa"/> mantissa bits, the exponent 0 a
/// subnormal (or, without <paramref name="Subnormals"/>, the implicit 1 kept: E8M0's 2^-127), its codes past the finite
/// ones by <paramref name="Specials"/>; or a code standing for its entry of <paramref name="Table"/> (2^Bits entries).
/// A value's code and a block's scales are each one of these (<see cref="TypeDescription"/>).
/// </summary>
/// <param name="Bits">Bits the field takes (1 to 64).</param>
/// <param name="Kind">Integer, float or a table's entry.</param>
/// <param name="Signed">Whether it has a sign (two's complement for an integer, a sign bit for a float).</param>
/// <param name="Exponent">A float's exponent bits.</param>
/// <param name="Mantissa">A float's mantissa bits.</param>
/// <param name="ExponentBias">Subtracted from a float's exponent.</param>
/// <param name="Subnormals">Whether a float's exponent 0 is a subnormal (else 2^(−bias) · (1 + mantissa)).</param>
/// <param name="Specials">A float's codes that are not finite.</param>
/// <param name="Table">Each code's value (2^Bits entries), for a table's codes; a codebook's (<paramref name="Width"/> more than 1) 2^Bits · Width, code c's value i at c · Width + i.</param>
/// <param name="Width">
/// Values one code stands for: 1, a number; more, a codebook (a table's codes only), a code indexing a vector of that many
/// values, one index giving several values (ggml's IQ2_XXS grid: an 8-bit index, 8 magnitudes).
/// </param>
public sealed record NumberType(int Bits, NumberKind Kind, bool Signed, int Exponent = 0, int Mantissa = 0, int ExponentBias = 0, bool Subnormals = true,
    FloatSpecials Specials = FloatSpecials.Ieee, IReadOnlyList<double>? Table = null, int Width = 1)
{
    /// <summary>A whole number of <paramref name="bits"/> bits (two's complement when <paramref name="signed"/>).</summary>
    public static NumberType Integer(int bits, bool signed) => new(bits, NumberKind.Integer, signed);

    /// <summary>A float of a sign bit (when <paramref name="signed"/>), <paramref name="exponent"/> and <paramref name="mantissa"/> bits, the bias 2^(exponent − 1) − 1 unless given.</summary>
    public static NumberType Floating(int exponent, int mantissa, bool signed = true, int? bias = null, bool subnormals = true, FloatSpecials specials = FloatSpecials.Ieee) =>
        new((signed ? 1 : 0) + exponent + mantissa, NumberKind.Float, signed, exponent, mantissa, bias ?? (1 << (exponent - 1)) - 1, subnormals, specials);

    /// <summary>Codes of <paramref name="bits"/> bits, each standing for its entry of <paramref name="table"/>.</summary>
    public static NumberType Entries(int bits, IReadOnlyList<double> table) => new(bits, NumberKind.Table, false, Table: table);

    /// <summary>
    /// A codebook: codes of <paramref name="bits"/> bits, each standing for a vector of <paramref name="width"/> values,
    /// code c's value i <paramref name="table"/>[c · width + i].
    /// </summary>
    public static NumberType Codebook(int bits, int width, IReadOnlyList<double> table) => new(bits, NumberKind.Table, false, Table: table, Width: width);

    /// <summary>Whether the description holds together: the bits add up, a table has an entry a code (Width of them a codebook's).</summary>
    public bool Valid => Bits is >= 1 and <= 64 && Width >= 1 && (Width == 1 || Kind == NumberKind.Table) && Kind switch
    {
        NumberKind.Integer => true,
        NumberKind.Float => Exponent >= 1 && Mantissa >= 0 && (Signed ? 1 : 0) + Exponent + Mantissa == Bits && Exponent <= 15,
        _ => Bits <= 16 && Table is { } t && t.Count == (1 << Bits) * Width,
    };

    /// <summary>Value <paramref name="within"/> (below Width) of the vector the field's <paramref name="bits"/> stand for: a codebook's entry's.</summary>
    public double Value(ulong bits, int within) => Width == 1 ? Value(bits) : Table![(int)(bits & ((1UL << Bits) - 1)) * Width + within];

    /// <summary>The number the field's <paramref name="bits"/> (its lowest Bits bits) stand for (a codebook's: its entry's first value).</summary>
    public double Value(ulong bits)
    {
        bits = Bits == 64 ? bits : bits & ((1UL << Bits) - 1);
        switch (Kind)
        {
            case NumberKind.Integer:
                return Signed && Bits < 64 ? (long)(bits << (64 - Bits)) >> (64 - Bits) : Signed ? (long)bits : (double)bits;
            case NumberKind.Table:
                return Table![(int)bits * Width];
        }

        bool negative = Signed && (bits >> (Bits - 1) & 1) != 0;
        ulong magnitude = Signed ? bits & ((1UL << (Bits - 1)) - 1) : bits;
        long e = (long)(magnitude >> Mantissa), top = (1L << Exponent) - 1;
        ulong m = magnitude & ((1UL << Mantissa) - 1);
        double value;
        if (Specials == FloatSpecials.Ieee && e == top)
        {
            value = m == 0 ? double.PositiveInfinity : double.NaN;
        }
        else if (Specials == FloatSpecials.AllOnesNaN && magnitude == (Signed ? (1UL << (Bits - 1)) - 1 : Bits == 64 ? ulong.MaxValue : (1UL << Bits) - 1))
        {
            value = double.NaN;
        }
        else if (e == 0 && Subnormals)
        {
            value = Math.ScaleB(m, 1 - ExponentBias - Mantissa);
        }
        else
        {
            value = Math.ScaleB((double)((1UL << Mantissa) | m), (int)e - ExponentBias - Mantissa);
        }

        return negative ? -value : value;
    }
}

/// <summary>
/// A block's scale or minimum (<see cref="TypeDescription"/>): a number of <paramref name="Type"/> for each
/// <paramref name="Values"/> values of the block; scale i at bits [8 · <paramref name="Offset"/> + i · Type.Bits, + Type.Bits)
/// of the block (little-endian), or, with <paramref name="Pieces"/>, assembled from Pieces[i] (its low bits first); plus
/// <paramref name="Bias"/>.
/// </summary>
/// <param name="Type">The number each is.</param>
/// <param name="Values">Values one covers (a block's, or a sub-block's).</param>
/// <param name="Offset">The first one's first byte in the block.</param>
/// <param name="Pieces">Each one's bits as pieces, or null: one after another from <paramref name="Offset"/>.</param>
/// <param name="Bias">Added to each one's number.</param>
public sealed record BlockScale(NumberType Type, int Values, int Offset, IReadOnlyList<IReadOnlyList<ScalePiece>>? Pieces = null, double Bias = 0)
{
    /// <summary>The bits of scale <paramref name="index"/> of <paramref name="block"/>.</summary>
    public ulong Bits(ReadOnlySpan<byte> block, int index)
    {
        if (Pieces is { } pieces)
        {
            ulong value = 0;
            int at = 0;
            var own = pieces[index];
            for (int i = 0; i < own.Count; i++)
            {
                var piece = own[i];
                value |= (ulong)(block[piece.Offset] >> piece.Shift & ((1 << piece.Bits) - 1)) << at;
                at += piece.Bits;
            }

            return value;
        }

        return TypeDescription.Field(block, 8L * Offset + (long)index * Type.Bits, Type.Bits);
    }

    /// <summary>The number of scale <paramref name="index"/> of <paramref name="block"/>, its bias added.</summary>
    public double Value(ReadOnlySpan<byte> block, int index) => Type.Value(Bits(block, index)) + Bias;

    /// <summary>
    /// The number scale <paramref name="index"/> of <paramref name="block"/> gives the value <paramref name="within"/> of the
    /// values it covers (a codebook's: its entry's value of that place; a number's: the same for each), its bias added.
    /// </summary>
    public double Value(ReadOnlySpan<byte> block, int index, int within) => Type.Value(Bits(block, index), within % Type.Width) + Bias;
}

/// <summary>
/// How a type's values read from their bits, whatever the type: blocks of <paramref name="Values"/> values in
/// <paramref name="Bytes"/> bytes; value v's code is the fields of <paramref name="Planes"/> at its place in the block (the
/// first plane the lowest bits), read as a number of <paramref name="Code"/>'s kind; value v = (that number +
/// <paramref name="CodeOffset"/>) · the product of the <paramref name="Scales"/> covering v, less the product of the
/// <paramref name="Mins"/> covering v (added where <paramref name="MinsAdded"/>). A per-element type is a block of one value
/// (or of the values a byte holds). A codebook code (<see cref="NumberType.Width"/> more than 1) gives Width values from one
/// index: the planes place each entry's index (entry e = v / Width), and value v takes its entry's value v mod Width; a
/// codebook scale (its Values its Width: a vector of factors, as IQ2_XXS's sign masks) gives each value it covers its own
/// factor. A scale level shared over many blocks (a scale whose Values pass a block's, as a second-level scale of block
/// scales) lives in a group of <paramref name="Group"/> blocks: the group's blocks one after another, then such levels, the
/// group <paramref name="GroupBytes"/> bytes; each level's scale i of a group (its values i · Values of the group on) at its
/// own position, bits [8 · Offset + i · its bits) from the group's first byte (or its pieces', from there). A kernel that
/// reads this alone multiplies any type described so, with no code of its own (the described lanes).
/// </summary>
/// <param name="Values">Values a block holds.</param>
/// <param name="Bytes">Bytes a block takes.</param>
/// <param name="Code">What a value's code is as a number: its bits, integer or float with its bits, or a table.</param>
/// <param name="Planes">Where a code's bits lie in a block, the lowest first (one plane for most types).</param>
/// <param name="CodeOffset">Added to a code's number before the scales.</param>
/// <param name="Scales">Multiplied with a value's number, or null: none.</param>
/// <param name="Mins">Their product taken from (or added to) a value, or null: none.</param>
/// <param name="MinsAdded">Whether the minimums' product is added rather than subtracted.</param>
/// <param name="Group">Blocks a group holds: more than 1 where a scale level is shared over many blocks (stored after them).</param>
/// <param name="GroupBytes">Bytes a group takes, its blocks and its shared levels (0: Group · Bytes).</param>
public sealed record TypeDescription(int Values, int Bytes, NumberType Code, IReadOnlyList<CodePlane> Planes, double CodeOffset = 0,
    IReadOnlyList<BlockScale>? Scales = null, IReadOnlyList<BlockScale>? Mins = null, bool MinsAdded = false, int Group = 1, int GroupBytes = 0)
{
    /// <summary>Values a group of blocks holds (Group · Values): the unit a row is whole of.</summary>
    public int UnitValues => Group * Values;

    /// <summary>Bytes a group takes (GroupBytes, or Group · Bytes).</summary>
    public int UnitBytes => GroupBytes > 0 ? GroupBytes : Group * Bytes;

    /// <summary>Whether <paramref name="scale"/> is a level shared over many blocks (its values pass a block's: stored at the group's own position).</summary>
    public bool Shared(BlockScale scale) => scale.Values > Values;

    /// <summary>Value <paramref name="v"/>'s block's first byte in its group (v within the group).</summary>
    public int BlockStart(int v) => v % UnitValues / Values * Bytes;

    /// <summary>The byte <paramref name="scale"/>'s Offset counts from for value <paramref name="v"/> of a group: its block's first, or the group's (a shared level).</summary>
    public int ScaleBase(BlockScale scale, int v) => Shared(scale) ? 0 : BlockStart(v);

    /// <summary>The index of <paramref name="scale"/>'s number covering value <paramref name="v"/> of a group (in its block, or in the group for a shared level).</summary>
    public int ScaleIndex(BlockScale scale, int v) => (Shared(scale) ? v % UnitValues : v % Values) / scale.Values;

    /// <summary>
    /// Whether the description holds together: the planes' bits make the code's, each plane's fields (an entry's, a
    /// codebook's) within a block, the scales cover whole values (a codebook scale exactly its entry's values).
    /// </summary>
    public bool Valid => Values >= 1 && Bytes >= 1 && Group >= 1 && UnitBytes >= Group * Bytes && Code.Valid && Values % Code.Width == 0 && Planes.Count >= 1
        && Planes.Sum(p => p.Bits) == Code.Bits
        && Planes.All(PlaneHolds)
        && (Scales ?? []).Concat(Mins ?? []).All(s => s.Type.Valid && s.Values >= 1 && (s.Type.Width == 1 || s.Values == s.Type.Width) && (Shared(s)
            ? s.Values % Values == 0 && UnitValues % s.Values == 0 && s.Type.Width == 1
                && (s.Pieces is { } shared ? shared.Count == UnitValues / s.Values : 8L * s.Offset + (long)UnitValues / s.Values * s.Type.Bits <= 8L * UnitBytes)
            : Values % s.Values == 0 && (s.Pieces is { } pieces ? pieces.Count == Values / s.Values : 8L * s.Offset + (long)Values / s.Values * s.Type.Bits <= 8L * Bytes)));

    // Whether plane p's fields hold together: fields of 1, 2, 4 or 8 bits in groups (or whole bytes a value), a stream's or
    // listed fields of up to 64 bits, or listed digits of a radix (each a digit its byte holds, its number within the bits);
    // a listed plane a field an entry; each field within a block.
    private bool PlaneHolds(CodePlane p)
    {
        int fields = Values / Code.Width;
        if (p.Bits < 1 || p.Group < 1 || p.Places is { } listed && listed.Count != fields)
        {
            return false;
        }

        if (p.Radix != 0)
        {
            return p.Radix >= 2 && p.Places is { } digits && !p.Stream && p.Bits <= 8 && p.Radix - 1 < 1 << p.Bits
                && digits.All(f => f.Byte >= 0 && f.Byte < Bytes && f.Shift >= 0 && f.Shift < p.Digits);
        }

        bool shape = p.Places is { } places ? p.Bits <= 64 && places.All(f => f.Byte >= 0 && f.Shift is >= 0 and <= 7)
            : p.Stream ? p.Bits <= 64
            : (p.Bits is 1 or 2 or 4 or 8 || p.Bits % 8 == 0 && p.Group == 1) && p.Group * p.Bits % 8 == 0 && (p.Stride == 0 || p.Stride >= p.Group * p.Bits / 8);
        return shape && Enumerable.Range(0, fields).All(v => p.Place(v).Byte + (p.Place(v).Shift + p.Bits + 7) / 8 <= Bytes);
    }

    /// <summary>Whether a value's code or factors come from a codebook (a code or a scale of more than one value an entry).</summary>
    public bool HasCodebook => Code.Width > 1 || (Scales ?? []).Concat(Mins ?? []).Any(s => s.Type.Width > 1);

    /// <summary>Bits [<paramref name="bit"/>, + <paramref name="bits"/>) of <paramref name="bytes"/>, little-endian (at most 64).</summary>
    public static ulong Field(ReadOnlySpan<byte> bytes, long bit, int bits)
    {
        ulong value = 0;
        for (int i = 0; i < bits; i++)
        {
            long at = bit + i;
            value |= (ulong)(bytes[(int)(at >> 3)] >> (int)(at & 7) & 1) << i;
        }

        return value;
    }

    /// <summary>
    /// Value <paramref name="v"/>'s code bits in <paramref name="block"/> (a group's bytes where Group passes 1, v within the
    /// group), its planes' fields joined (the first the lowest bits; a plane of digits gives its digit): a codebook's, its
    /// entry's index.
    /// </summary>
    public ulong CodeBits(ReadOnlySpan<byte> block, int v)
    {
        block = block[BlockStart(v)..];
        v %= Values;
        ulong code = 0;
        int low = 0, entry = v / Code.Width;
        for (int p = 0; p < Planes.Count; p++)
        {
            var plane = Planes[p];
            var (at, shift) = plane.Place(entry);
            code |= (plane.Radix != 0 ? (ulong)plane.Digit(block[at], shift) : Field(block, 8L * at + shift, plane.Bits)) << low;
            low += plane.Bits;
        }

        return code;
    }

    /// <summary>
    /// Value <paramref name="v"/> of <paramref name="block"/> (a group's bytes where Group passes 1, v within the group), by
    /// the description alone (in float64; the reference a kernel's values are checked against).
    /// </summary>
    public double ValueOf(ReadOnlySpan<byte> block, int v)
    {
        v %= UnitValues;
        double value = Code.Value(CodeBits(block, v), v % Code.Width) + CodeOffset;
        foreach (var scale in Scales ?? [])
        {
            value *= scale.Value(block[ScaleBase(scale, v)..], ScaleIndex(scale, v), v % scale.Values);
        }

        if (Mins is { Count: > 0 } mins)
        {
            double min = 1;
            foreach (var m in mins)
            {
                min *= m.Value(block[ScaleBase(m, v)..], ScaleIndex(m, v), v % m.Values);
            }

            value = MinsAdded ? value + min : value - min;
        }

        return value;
    }

    /// <summary>The values of whole groups of blocks of <paramref name="source"/> into <paramref name="destination"/>, by the description alone.</summary>
    public void Decode(ReadOnlySpan<byte> source, Span<float> destination)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = (float)ValueOf(source.Slice(i / UnitValues * UnitBytes, UnitBytes), i % UnitValues);
        }
    }
}

/// <summary>
/// The writer of any described type (<see cref="TypeDescription"/>): float32 values stored by the type's description alone,
/// with no code of the type's own. Per group of blocks it
/// searches, by measured squared error, for every number the description stores:
/// 1. each unit (the values the finest scale or minimum covers) its own continuous scale, and minimum where there are
///    minimums, with its codes: candidate scales putting the unit's extremes at the codes' extremes and at half steps of the
///    codes' numbers inside and past them, each candidate's codes the nearest and their least-squares scale (and minimum),
///    the best refined by quantizing again with it;
/// 2. the levels of scales from the units' scales: the finest level's numbers and their parent's continuous value by the
///    same search over the finest level's numbers (weighted by how much a unit's error moves with its scale), level by level
///    up to the coarsest, which is rounded to its number type; then each finer level's number the nearest its target over
///    the levels stored above it; the minimums' own levels alike from the units' minimums (a minimum stored in a scale's own
///    bits taking that scale's number);
/// 3. each value's code given the stored levels: its number type's nearest number, or a codebook's nearest entry (a vector
///    factor's entry, as a sign mask, chosen first from the targets' signs, then again given the entries);
/// 4. each level's numbers searched again, the finest first: each chunk's least-squares number given the codes, rounded, and
///    its neighbours, the codes chosen again for each and the error measured, the best kept.
/// Each number type's numbers are read from the type (a table, an integer, a float of any bits up to 16, a bias added; a per-element code of more bits, up to 64, rounded by arithmetic), each
/// field written where the description places it (planes of bits or of digits, listed places, pieces, levels shared over a
/// group of blocks); codes sharing a field (IQ1_S's delta sign, one a sub-block) are chosen together for the field's number
/// that leaves them the least error. A type of no scale, minimum or factor (a per-element type) is rounded as its number
/// type defines: to the nearest number, a tie to the even code, a zero's sign kept, a float past its range its infinity,
/// its NaN or its largest as its specials say; a code of ±0 keeps its target's sign inside blocks too, so every code of a
/// finite number is one some value writes. The same bytes on every CPU: scalar arithmetic in a fixed order.
/// </summary>
internal sealed class DescribedEncoder
{
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<TypeDescription, object> Made = new();

    // A unit's candidate scales put its value of largest magnitude at the codes' extreme moved by these half steps of the
    // extreme's step (outward positive; past the extreme clips the largest values, which the error may favour); with a
    // minimum, also its range over the codes' range moved by these half steps. A codebook's entries cost a nearest-entry
    // search each, and its search tries fewer (BookSteps, BookNeighbours, BookPasses; measured on a real model's weights:
    // the same error within 0.2% at twice the speed).
    private static readonly float[] ScaleSteps = [-2f, -1f, 0f, 1f, 2f, 4f, 6f, 8f];
    private static readonly float[] BookSteps = [-1f, 0f, 1f, 2f];
    private static readonly float[] RangeSteps = [-2f, -1f, 0f, 1f, 2f, 4f];

    // Times a search quantizes again with its best scale; the levels' numbers searched again by neighbours this far either
    // side (a level of at most SmallGrid numbers), in this many passes (a codebook's fewer, as BookSteps).
    private const int Refinements = 4, SmallGrid = 256, Neighbours = 4, Passes = 2, BookNeighbours = 2, BookPasses = 1;

    // The most numbers a field shared by several codes may take (its codes searched for each).
    private const int SharedLimit = 16;

    private readonly TypeDescription d;
    private readonly int n, size, unit, units, slots, codeWidth, neighbours, passes;
    private readonly Grid? codeGrid;
    private readonly Grid[]? sharedGrids;
    private readonly Book? codeBook;
    private readonly FactorLevel? factor;
    private readonly Level[] scales, mins, free, all;
    private readonly bool minsAdded, negativeScale, direct, wholeBytes;
    private readonly WideNumber? wide;
    private readonly int minSign;
    private readonly Fields codes;
    private readonly DigitPlane[] digitPlanes;
    private readonly int[]? leader;
    private readonly ulong sharedMask;
    private readonly ulong[] sharedValues;
    private readonly double top, bottom, topStep, bottomStep;

    private DescribedEncoder(TypeDescription d)
    {
        this.d = d;
        n = d.UnitValues;
        size = d.UnitBytes;
        codeWidth = d.Code.Width;
        slots = n / codeWidth;

        // The code's planes: each code slot's field in each, the lowest bits first: its bits at their place in its block, or a
        // digit of a byte in a plane of digits (CodePlane.Radix). A field several slots share (one bit that entries of a
        // sub-block hold in common) groups those slots, whose codes must agree there: leader[s] is the first slot of s's group.
        int planes = d.Planes.Count;
        var lows = new int[planes];
        for (int p = 1; p < planes; p++)
        {
            lows[p] = lows[p - 1] + d.Planes[p - 1].Bits;
        }

        var start = new int[slots + 1];
        var bits = new List<int>();
        var widths = new List<int>();
        var pieceLows = new List<int>();
        var keys = new long[planes, slots];
        var digitSlots = new List<(int Byte, int Digit)>[planes];
        for (int s = 0; s < slots; s++)
        {
            int v = s * codeWidth, block = d.BlockStart(v), entry = v % d.Values / codeWidth;
            for (int p = 0; p < planes; p++)
            {
                var plane = d.Planes[p];
                var (at, shift) = plane.Place(entry);
                if (plane.Radix == 0)
                {
                    bits.Add(8 * (block + at) + shift);
                    widths.Add(plane.Bits);
                    pieceLows.Add(lows[p]);
                    keys[p, s] = 8L * (block + at) + shift;
                }
                else
                {
                    (digitSlots[p] ??= []).Add((block + at, shift));
                    keys[p, s] = 64L * (block + at) + shift;
                }
            }

            start[s + 1] = bits.Count;
        }

        codes = new Fields(start, [.. bits], [.. widths], [.. pieceLows]);
        digitPlanes =[.. Enumerable.Range(0, planes).Where(p => d.Planes[p].Radix != 0)
            .Select(p => DigitPlane.Of(d.Planes[p], lows[p], digitSlots[p], size) ?? throw Unwritten(d, $"its plane of base-{d.Planes[p].Radix} digits has numbers of digits no byte reads as"))];

        int[]? partition = null;
        for (int p = 0; p < planes; p++)
        {
            var first = new Dictionary<long, int>();
            var lead = new int[slots];
            bool shared = false;
            for (int s = 0; s < slots; s++)
            {
                if (first.TryGetValue(keys[p, s], out int l))
                {
                    lead[s] = l;
                    shared = true;
                }
                else
                {
                    first[keys[p, s]] = s;
                    lead[s] = s;
                }
            }

            if (shared)
            {
                if (partition is not null && !partition.AsSpan().SequenceEqual(lead))
                {
                    throw Unwritten(d, "its planes share fields among different groups of codes");
                }

                partition = lead;
                sharedMask |= (d.Planes[p].Bits >= 64 ? ulong.MaxValue : (1UL << d.Planes[p].Bits) - 1) << lows[p];
            }
        }

        leader = partition;
        int sharedBits = BitOperations.PopCount(sharedMask);
        if (1 << Math.Min(sharedBits, 30) > SharedLimit)
        {
            throw Unwritten(d, $"a field its codes share takes {sharedBits} bits (at most {BitOperations.Log2(SharedLimit)} are searched)");
        }

        sharedValues = [.. Enumerable.Range(0, 1 << sharedBits).Select(k => Scatter((ulong)k, sharedMask))];

        // The codes' numbers: those the planes can hold (a plane of digits holds digits below its radix).
        Func<ulong, bool>? held = digitPlanes.Length == 0 ? null : code => digitPlanes.All(dp => dp.Holds(code));

        // A per-element code wider than 16 bits (no scale, minimum, factor or shared field: binary32, binary64, an integer of
        // 32 or 64 bits, a float of any bits up to 64): each value rounded by arithmetic (WideNumber), as a table of its
        // numbers would round it.
        if (codeWidth == 1 && d.Code.Bits > 16 && (d.Scales?.Count ?? 0) == 0 && (d.Mins?.Count ?? 0) == 0 && sharedMask == 0 && digitPlanes.Length == 0
            && WideNumber.Of(d.Code) is { } number)
        {
            wide = number;
            direct = true;
            unit = n;
            units = 1;
            scales = mins = free = all = [];
            wholeBytes = WholeBytes(d);
            if (!codes.Claim(new bool[8 * size], repeats: true))
            {
                throw Unwritten(d, "two of its fields share bits");
            }

            return;
        }
        if (codeWidth == 1)
        {
            codeGrid = Grid.Of(d.Code, d.CodeOffset, held) ?? throw Unwritten(d, "its codes' numbers are not a table, an integer or a float of up to 16 bits");
            if (sharedMask != 0)
            {
                sharedGrids = [.. sharedValues.Select(value => Grid.Of(d.Code, d.CodeOffset, code => (held?.Invoke(code) ?? true) && (code & sharedMask) == value)
                    ?? throw Unwritten(d, "a number of its shared field holds no code"))];
            }
        }
        else
        {
            codeBook = Book.Of(d.Code, d.CodeOffset, held) ?? throw Unwritten(d, "its codebook is not a table of up to 16 bits");
        }

        var scaleLevels = new List<Level>();
        foreach (var s in d.Scales ?? [])
        {
            if (s.Type.Width > 1)
            {
                if (factor is not null || s.Values != s.Type.Width)
                {
                    throw Unwritten(d, "it has more than one level of vector factors");
                }

                factor = new FactorLevel(Book.Of(s.Type, s.Bias) ?? throw Unwritten(d, "a vector factor is not a table of up to 16 bits"), s.Values, FieldsOf(d, s, n / s.Values));
                continue;
            }

            scaleLevels.Add(LevelOf(d, s, scale: true));
        }

        var minLevels = new List<Level>();
        var freeLevels = new List<Level>();
        foreach (var m in d.Mins ?? [])
        {
            if (m.Type.Width > 1)
            {
                throw Unwritten(d, "a minimum is a vector of factors");
            }

            if (scaleLevels.Find(s => SameField(s.Source, m)) is { } tied)
            {
                tied.Min = true;                                            // stored in a scale's own bits: that scale's number
                minLevels.Add(tied);
                continue;
            }

            var level = LevelOf(d, m, scale: false);
            minLevels.Add(level);
            freeLevels.Add(level);
        }

        scales = [.. scaleLevels.OrderByDescending(l => l.Values)];
        mins = [.. minLevels.OrderByDescending(l => l.Values)];
        free = [.. freeLevels.OrderByDescending(l => l.Values)];
        all = [.. scales.Concat(free).OrderBy(l => l.Values)];
        for (int i = 0; i < all.Length; i++)
        {
            all[i].Index = i;
        }

        if (!Nested(scales) || !Nested(free))
        {
            throw Unwritten(d, "its levels do not nest (each level's values a whole number of the next finer level's)");
        }

        unit = Math.Min(scales.Length > 0 ? scales[^1].Values : n, mins.Length > 0 ? mins[^1].Values : n);
        if (unit % codeWidth != 0 || factor is { } f && unit % f.Values != 0 || n % unit != 0 || all.Any(l => l.Values % unit != 0))
        {
            throw Unwritten(d, "its codes, factors and levels do not share whole units");
        }

        units = n / unit;
        neighbours = codeBook is null ? Neighbours : BookNeighbours;
        passes = codeBook is null ? Passes : BookPasses;
        minsAdded = d.MinsAdded;
        negativeScale = scales.Length > 0 && scales[^1].Grid.HasNegative;
        if (mins.Length == 0 || mins.Any(l => l.Values == mins[^1].Values && l.Grid.HasNegative && l.Grid.HasPositive))
        {
            minSign = 0;                                                    // a unit's minimum of either sign
        }
        else if (mins.All(l => !(l.Grid.HasNegative && l.Grid.HasPositive)))
        {
            int sign = mins.Aggregate(1, (s, l) => l.Grid.HasNegative ? -s : s);
            minSign = minsAdded ? sign : -sign;                             // the one sign the levels allow
        }
        else
        {
            minSign = 2;                                                    // a sign shared by the group's units
        }

        // A group of codes sharing a field lies within a unit (its codes chosen together).
        if (leader is { } leaders && Enumerable.Range(0, slots).Any(s => leaders[s] * codeWidth / unit != s * codeWidth / unit))
        {
            throw Unwritten(d, "codes sharing a field lie in different units of its scales");
        }

        // Values of no scale, minimum or factor, each its own code (a per-element type) are rounded as a float format
        // defines (Grid.Round).
        direct = scales.Length == 0 && mins.Length == 0 && factor is null && codeGrid is not null && sharedMask == 0 && digitPlanes.Length == 0;
        wholeBytes = direct && WholeBytes(d);

        // No two fields share a bit (a minimum stored in a scale's bits is that scale's level; codes sharing a field, theirs;
        // a plane of digits takes its bytes whole).
        var owned = new bool[8 * size];
        if (!codes.Claim(owned, repeats: true) || digitPlanes.Any(p => !p.Claim(owned)) || all.Any(l => !l.Fields.Claim(owned))
            || factor is { } owner && !owner.Fields.Claim(owned))
        {
            throw Unwritten(d, "two of its fields share bits");
        }

        // The codes' numbers (times a vector factor's): their extremes and the steps beside them.
        double[] numbers = codeGrid?.Values ?? codeBook!.Levels;
        if (factor is { } vector)
        {
            numbers = [.. vector.Book.Levels.SelectMany(p => numbers.Select(q => p * q)).Distinct().Order()];
        }

        top = numbers[^1];
        bottom = numbers[0];
        topStep = numbers.Length > 1 ? numbers[^1] - numbers[^2] : 1;
        bottomStep = numbers.Length > 1 ? numbers[1] - numbers[0] : 1;
    }

    /// <summary>The writer of <paramref name="d"/>, or null where its description is one it does not write (said by <see cref="Why"/>).</summary>
    public static DescribedEncoder? Of(TypeDescription? d) => d is null ? null : Made.GetValue(d, static x => Make(x)) as DescribedEncoder;

    /// <summary>Why <paramref name="d"/> has no writer, or null where it has one.</summary>
    public static string? Why(TypeDescription d) => Made.GetValue(d, static x => Make(x)) as string;

    private static object Make(TypeDescription d)
    {
        if (!d.Valid)
        {
            return "its description does not hold together";
        }

        try
        {
            return new DescribedEncoder(d);
        }
        catch (NotSupportedException e)
        {
            return e.Message;
        }
    }

    private static NotSupportedException Unwritten(TypeDescription d, string why) => new($"A type of {d.Values}-value blocks in {d.Bytes} bytes is not written from its description: {why}.");

    // Whether a minimum is stored in a scale's own bits as the same number.
    private static bool SameField(BlockScale a, BlockScale b) => a.Values == b.Values && a.Offset == b.Offset && a.Bias == b.Bias && SameType(a.Type, b.Type)
        && (a.Pieces is null ? b.Pieces is null : b.Pieces is not null && a.Pieces.Count == b.Pieces.Count && a.Pieces.Zip(b.Pieces).All(p => p.First.SequenceEqual(p.Second)));

    private static bool SameType(NumberType a, NumberType b) => a with { Table = null } == b with { Table = null }
        && (a.Table is null ? b.Table is null : b.Table is not null && a.Table.SequenceEqual(b.Table));

    private static bool Nested(Level[] levels)
    {
        for (int i = 1; i < levels.Length; i++)
        {
            if (levels[i - 1].Values % levels[i].Values != 0)
            {
                return false;
            }
        }

        return true;
    }

    // The bits of k placed at the set bits of mask, the lowest first.
    private static ulong Scatter(ulong k, ulong mask)
    {
        ulong placed = 0;
        for (int b = 0, i = 0; b < 64; b++)
        {
            if ((mask >> b & 1) != 0)
            {
                placed |= (k >> i++ & 1) << b;
            }
        }

        return placed;
    }

    private Level LevelOf(TypeDescription d, BlockScale s, bool scale)
    {
        var grid = Grid.Of(s.Type, s.Bias) ?? throw Unwritten(d, "a scale or minimum is not a table, an integer or a float of up to 16 bits");
        return new Level(s, grid, s.Values, n / s.Values, FieldsOf(d, s, n / s.Values)) { Scale = scale, Min = !scale };
    }

    // Each chunk's bits of a scale or minimum in a group (its pieces', or its run of bits from its offset).
    private static Fields FieldsOf(TypeDescription d, BlockScale s, int chunks)
    {
        var start = new int[chunks + 1];
        var bits = new List<int>();
        var widths = new List<int>();
        for (int j = 0; j < chunks; j++)
        {
            int v = j * s.Values, at = d.ScaleBase(s, v), index = d.ScaleIndex(s, v);
            if (s.Pieces is { } pieces)
            {
                foreach (var piece in pieces[index])
                {
                    bits.Add(8 * (at + piece.Offset) + piece.Shift);
                    widths.Add(piece.Bits);
                }
            }
            else
            {
                bits.Add(8 * (at + s.Offset) + index * s.Type.Bits);
                widths.Add(s.Type.Bits);
            }

            start[j + 1] = bits.Count;
        }

        return new Fields(start, [.. bits], [.. widths]);
    }

    /// <summary>Stores whole groups of <paramref name="source"/> into <paramref name="destination"/>.</summary>
    public void Encode(ReadOnlySpan<float> source, Span<byte> destination) => Encode(source, destination, []);

    /// <summary>
    /// <see cref="Encode(ReadOnlySpan{float}, Span{byte})"/>, and where <paramref name="chosen"/> is not empty, the value it
    /// chose for each (its stored numbers' scale times its code's number plus its minimum, in float32).
    /// </summary>
    internal void Encode(ReadOnlySpan<float> source, Span<byte> destination, Span<float> chosen)
    {
        if (source.Length % n != 0)
        {
            throw new ArgumentException($"The type stores whole groups of {n} values; {source.Length} is not.", nameof(source));
        }

        if (wholeBytes)
        {
            // One value a block, its code whole bytes of its own: each value's code written little-endian where it lies.
            for (int i = 0; i < source.Length; i++)
            {
                ulong code = Round(source[i]);
                Put(destination.Slice(i * size, size), code);
                if (!chosen.IsEmpty)
                {
                    double number = d.Code.Value(code);
                    chosen[i] = (float)(d.CodeOffset == 0 ? number : number + d.CodeOffset);
                }
            }

            return;
        }

        var w = Work.For(this);
        for (int g = 0; g < source.Length / n; g++)
        {
            if (direct)
            {
                // Each value its code, rounded as the code's float format defines (or the nearest number of an integer or a table).
                for (int v = 0; v < n; v++)
                {
                    w.CodeBits[v] = Round(source[g * n + v]);
                    if (!chosen.IsEmpty)
                    {
                        double number = d.Code.Value(w.CodeBits[v]);
                        chosen[g * n + v] = (float)(d.CodeOffset == 0 ? number : number + d.CodeOffset);
                    }
                }

                WriteCodes(w, destination.Slice(g * size, size));
                continue;
            }

            EncodeGroup(source.Slice(g * n, n), destination.Slice(g * size, size), w);
            if (!chosen.IsEmpty)
            {
                for (int u = 0; u < units; u++)
                {
                    UnitFactors(w, u, out float a, out float c);
                    for (int v = u * unit; v < (u + 1) * unit; v++)
                    {
                        chosen[g * n + v] = a * w.G[v] + c;
                    }
                }
            }
        }
    }

    /// <summary>Whether float64 values are written from the float64 value itself (a type of no scale, minimum or factor: each value its own code).</summary>
    public bool EncodesDoubles => direct;

    /// <summary>
    /// <paramref name="source"/> (float64) stored into <paramref name="destination"/>, each value rounded from the float64
    /// value itself (<see cref="EncodesDoubles"/> types only); a last group of fewer values filled with zeros.
    /// </summary>
    public void Encode(ReadOnlySpan<double> source, Span<byte> destination)
    {
        if (!direct)
        {
            throw new InvalidOperationException("A type of scales, minimums or factors is written from float32 values.");
        }

        if (wholeBytes)
        {
            for (int i = 0; i < source.Length; i++)
            {
                Put(destination.Slice(i * size, size), Round(source[i]));
            }

            return;
        }

        var w = Work.For(this);
        for (int g = 0; g * n < source.Length; g++)
        {
            for (int v = 0; v < n; v++)
            {
                int i = g * n + v;
                w.CodeBits[v] = Round(i < source.Length ? source[i] : 0);
            }

            WriteCodes(w, destination.Slice(g * size, size));
        }
    }

    // A per-element value's code: rounded by its number type's table, or by arithmetic for a code wider than 16 bits.
    private ulong Round(double x) => wide is { } number ? number.Round(d.CodeOffset == 0 ? x : x - d.CodeOffset) : codeGrid!.Round(x);

    // A code's bytes, little-endian.
    private static void Put(Span<byte> at, ulong code)
    {
        for (int b = 0; b < at.Length; b++, code >>= 8)
        {
            at[b] = (byte)code;
        }
    }

    // One value a block, its code one plane of whole bytes from the block's first (written straight, a value at a time).
    private static bool WholeBytes(TypeDescription d) => BitConverter.IsLittleEndian && d is { Values: 1, Group: 1 } && d.UnitBytes == d.Bytes
        && d.Planes is [{ Offset: 0, Group: 1, Stride: 0, Stream: false, Places: null, Radix: 0 } plane] && plane.Bits == 8 * d.Bytes;

    private void EncodeGroup(ReadOnlySpan<float> source, Span<byte> group, Work w)
    {
        for (int v = 0; v < n; v++)
        {
            w.X[v] = float.IsFinite(source[v]) ? source[v] : 0f;
        }

        // 1. Each unit's own scale and minimum (a sign shared by the group's units: the units' majority, the rest again).
        for (int u = 0; u < units; u++)
        {
            SolveUnit(w, u, minSign == 2 ? 0 : minSign);
        }

        if (minSign == 2)
        {
            double sum = 0;
            for (int u = 0; u < units; u++)
            {
                sum += w.C[u];
            }

            int sign = sum >= 0 ? 1 : -1;
            for (int u = 0; u < units; u++)
            {
                if (w.C[u] * sign < 0)
                {
                    SolveUnit(w, u, sign);
                }
            }
        }

        // 2. The levels from the units' scales, then the minimums' own levels over what the scales' levels give them.
        if (scales.Length > 0)
        {
            Factor(w, scales, w.A, w.WA);
        }

        if (free.Length > 0)
        {
            for (int u = 0; u < units; u++)
            {
                double tied = 1;
                foreach (var level in mins)
                {
                    if (level.Scale)
                    {
                        tied *= level.Grid.Values[w.Ranks[level.Index][u * unit / level.Values]];
                    }
                }

                double target = minsAdded ? w.C[u] : -w.C[u];
                w.T[u] = tied != 0 ? (float)(target / tied) : 0f;
                w.Wt[u] = (float)(w.WC[u] * tied * tied);
            }

            Factor(w, free, w.T, w.Wt);
        }

        // 3. The codes given the stored levels.
        for (int u = 0; u < units; u++)
        {
            UnitFactors(w, u, out float a, out float c);
            w.Error[u] = QuantizeUnit(w, u, a, c, w.G, w.Codes, w.Factors);
        }

        // 4. Each level's numbers searched again by measured error, the finest first.
        for (int pass = 0; pass < passes; pass++)
        {
            foreach (var level in all)
            {
                for (int j = 0; j < level.Chunks; j++)
                {
                    Search(w, level, j);
                }
            }
        }

        Write(w, group);
    }

    // Unit u's own scale a and minimum c (the value a · g + c) and codes: Solve over its values.
    private void SolveUnit(Work w, int u, int sign)
    {
        var y = w.X.AsSpan(u * unit, unit);
        var g = w.TryG.AsSpan(0, unit);
        var best = w.BestG.AsSpan(0, unit);
        w.Quantizer.Unit = u;
        if (scales.Length == 0)
        {
            // No scale: the codes' numbers are the values, less a minimum where there is one.
            float c = 0f;
            for (int r = 0; r < Refinements; r++)
            {
                w.Quantizer.Run(y, 1f, c, g);
                if (mins.Length == 0)
                {
                    break;
                }

                double s = 0;
                for (int i = 0; i < unit; i++)
                {
                    s += y[i] - g[i];
                }

                c = (float)(s / unit);
                if (sign != 0 && c * sign < 0)
                {
                    c = 0f;
                }
            }

            w.A[u] = 1f;
            w.C[u] = c;
            w.WA[u] = unit;
            w.WC[u] = unit;
            return;
        }

        Solve(w.Quantizer, y, w.Ones.AsSpan(0, unit), negativeScale, mins.Length > 0, sign, g, best, out float a, out float m);
        double sgg = 0;
        for (int i = 0; i < unit; i++)
        {
            sgg += (double)best[i] * best[i];
        }

        w.A[u] = a;
        w.C[u] = m;
        w.WA[u] = (float)sgg;
        w.WC[u] = unit;
    }

    // The numbers of `levels` (coarse to fine, each nested in the one before) whose products come closest to the units'
    // targets t (weights wt): the finest level's chunks' targets the weighted means of their units'; bottom up, each parent's
    // continuous number and its children's numbers by Solve over the children's targets and the child level's numbers; the
    // coarsest rounded to its own; top down, each finer level's number nearest its target over the stored levels above it.
    private void Factor(Work w, Level[] levels, ReadOnlySpan<float> t, ReadOnlySpan<float> wt)
    {
        int last = levels.Length - 1;
        var leaf = levels[last];
        int per = leaf.Values / unit;
        var leafTargets = w.LevelTargets[leaf.Index];
        var leafWeights = w.LevelWeights[leaf.Index];
        for (int j = 0; j < leaf.Chunks; j++)
        {
            double sw = 0, swt = 0, st = 0;
            for (int u = j * per; u < (j + 1) * per; u++)
            {
                sw += wt[u];
                swt += (double)wt[u] * t[u];
                st += t[u];
            }

            leafTargets[j] = (float)(sw > 0 ? swt / sw : st / per);
            leafWeights[j] = (float)sw;
        }

        for (int i = last; i >= 1; i--)
        {
            var child = levels[i];
            var parent = levels[i - 1];
            int k = parent.Values / child.Values;
            bool negative = false;
            for (int l = 0; l < i; l++)
            {
                negative |= levels[l].Grid.HasNegative;
            }

            var childTargets = w.LevelTargets[child.Index];
            var childWeights = w.LevelWeights[child.Index];
            var ranks = w.Ranks[child.Index];
            for (int p = 0; p < parent.Chunks; p++)
            {
                var ys = childTargets.AsSpan(p * k, k);
                var ws = childWeights.AsSpan(p * k, k);
                var best = w.BestG.AsSpan(0, k);
                Solve(child.Quantizer, ys, ws, negative, false, 0, w.TryG.AsSpan(0, k), best, out float lambda, out _);
                double weight = 0;
                for (int c = 0; c < k; c++)
                {
                    weight += (double)ws[c] * best[c] * best[c];
                    ranks[p * k + c] = child.Grid.Nearest(best[c]);
                }

                w.LevelTargets[parent.Index][p] = lambda;
                w.LevelWeights[parent.Index][p] = (float)weight;
            }
        }

        var coarsest = levels[0];
        for (int p = 0; p < coarsest.Chunks; p++)
        {
            w.Ranks[coarsest.Index][p] = coarsest.Grid.Nearest(w.LevelTargets[coarsest.Index][p]);
        }

        for (int i = 1; i <= last; i++)
        {
            var level = levels[i];
            var targets = w.LevelTargets[level.Index];
            var ranks = w.Ranks[level.Index];
            for (int j = 0; j < level.Chunks; j++)
            {
                double above = 1;
                for (int l = 0; l < i; l++)
                {
                    above *= levels[l].Grid.Values[w.Ranks[levels[l].Index][j * level.Values / levels[l].Values]];
                }

                if (above != 0)
                {
                    ranks[j] = level.Grid.Nearest(targets[j] / above);
                }
            }
        }
    }

    // The scale a (and, `affine`, the added c) and codes making a · g + c closest to y under weights w, g the numbers q gives
    // for its targets: the targets of candidate scales putting y's largest at the codes' largest number and y's smallest at
    // their smallest (and, `negative`, the other way round), moved by half steps (ScaleSteps); with `affine`, y's range over
    // the codes' range so moved (RangeSteps); each candidate's codes the nearest, their least-squares a and c (Fit), the best
    // refined by quantizing again with it. Returns the weighted squared error; `best` holds the best codes' numbers.
    private static double Solve(Quantizer q, ReadOnlySpan<float> y, ReadOnlySpan<float> w, bool negative, bool affine, int sign,
        Span<float> g, Span<float> best, out float a, out float c)
    {
        float ymax = float.MinValue, ymin = float.MaxValue;
        double sxx = 0;
        for (int i = 0; i < y.Length; i++)
        {
            ymax = MathF.Max(ymax, y[i]);
            ymin = MathF.Min(ymin, y[i]);
            sxx += (double)w[i] * y[i] * y[i];
        }

        var s = new Candidates { Q = q, Y = y, W = w, G = g, Best = best, Negative = negative, Affine = affine, Sign = sign, Sxx = sxx, Error = double.PositiveInfinity };
        s.Try(0f, 0f);                                                       // every target 0
        foreach (float k in q.Steps)
        {
            if (ymax > 0 && q.Top > 0)
            {
                s.Try((float)((q.Top + k * q.TopStep / 2) / ymax), 0f);
            }

            if (ymin < 0 && q.Bottom < 0)
            {
                s.Try((float)((q.Bottom - k * q.BottomStep / 2) / ymin), 0f);
            }

            if (negative && ymax > 0 && q.Bottom < 0)
            {
                s.Try((float)((q.Bottom - k * q.BottomStep / 2) / ymax), 0f);
            }

            if (negative && ymin < 0 && q.Top > 0)
            {
                s.Try((float)((q.Top + k * q.TopStep / 2) / ymin), 0f);
            }
        }

        if (affine && ymax > ymin && q.Top > q.Bottom)
        {
            foreach (float k in RangeSteps)
            {
                double t = (q.Top - q.Bottom + k * (q.TopStep + q.BottomStep) / 4) / (ymax - ymin);
                if (t > 0)
                {
                    s.Try((float)t, (float)(ymin - q.Bottom / t));
                }
            }
        }

        for (int r = 0; r < Refinements && s.A != 0f; r++)
        {
            double before = s.Error;
            s.Try(1f / s.A, s.C);
            if (!(s.Error < before))
            {
                break;
            }
        }

        a = s.A;
        c = s.C;
        return s.Error;
    }

    // A candidate scale's codes and their fit, the best kept (Solve).
    private ref struct Candidates
    {
        public Quantizer Q;
        public ReadOnlySpan<float> Y, W;
        public Span<float> G, Best;
        public bool Negative, Affine;
        public int Sign;
        public double Sxx, Error;
        public float A, C;

        // The targets (y − c0) · t, their codes, the codes' least-squares a and c.
        public void Try(float t, float c0)
        {
            if (!float.IsFinite(t) || !float.IsFinite(c0))
            {
                return;
            }

            Q.Run(Y, t, c0, G);
            double e = Fit(Y, W, G, Negative, Affine, Sign, Sxx, out float a, out float c);
            if (e < Error)
            {
                Error = e;
                A = a;
                C = c;
                G.CopyTo(Best);
            }
        }
    }

    // The least-squares a (and, `affine`, c) of y over the codes' numbers g under weights w (c of `sign` or 0; a not negative
    // unless `negative`), and the weighted squared error of a · g + c.
    private static double Fit(ReadOnlySpan<float> y, ReadOnlySpan<float> w, ReadOnlySpan<float> g, bool negative, bool affine, int sign, double sxx, out float a, out float c)
    {
        double sw = 0, sg = 0, sgg = 0, sy = 0, syg = 0;
        for (int i = 0; i < y.Length; i++)
        {
            double wi = w[i], gi = g[i], yi = y[i];
            sw += wi;
            sg += wi * gi;
            sgg += wi * gi * gi;
            sy += wi * yi;
            syg += wi * yi * gi;
        }

        double aa = sgg > 0 ? syg / sgg : 0, cc = 0;
        if (affine)
        {
            double det = sw * sgg - sg * sg;
            if (det > 1e-9 * sw * sgg)
            {
                aa = (sw * syg - sy * sg) / det;
                cc = (sy - aa * sg) / sw;
            }

            if (sign != 0 && cc * sign < 0)
            {
                cc = 0;
                aa = sgg > 0 ? syg / sgg : 0;
            }
        }

        if (!negative && aa < 0)
        {
            aa = 0;
            cc = affine && sw > 0 && sy * sign >= 0 ? sy / sw : 0;
        }

        a = (float)aa;
        c = (float)cc;
        return sxx - 2 * aa * syg - 2 * cc * sy + aa * aa * sgg + 2 * aa * cc * sg + cc * cc * sw;
    }

    // Unit u's stored scale (the product of its scale levels' numbers) and minimum (∓ the product of its minimum levels').
    private void UnitFactors(Work w, int u, out float a, out float c)
    {
        float scale = 1f, min = 1f;
        foreach (var level in scales)
        {
            scale *= level.Grid.Floats[w.Ranks[level.Index][u * unit / level.Values]];
        }

        foreach (var level in mins)
        {
            min *= level.Grid.Floats[w.Ranks[level.Index][u * unit / level.Values]];
        }

        a = scale;
        c = mins.Length == 0 ? 0f : minsAdded ? min : -min;
    }

    // Unit u's codes for scale a and minimum c, each nearest its target (x − c) / a: numbers into g, codes and factors into
    // their slots (the unit's part of each); the squared error.
    private double QuantizeUnit(Work w, int u, float a, float c, Span<float> g, Span<int> codeSlots, Span<int> factorSlots)
    {
        var y = w.X.AsSpan(u * unit, unit);
        var numbers = g.Slice(u * unit, unit);
        w.Quantizer.Unit = u;
        w.Quantizer.Run(y, a != 0f ? 1f / a : 0f, c, numbers);
        w.Quantizer.Codes.AsSpan(0, unit / codeWidth).CopyTo(codeSlots.Slice(u * unit / codeWidth));
        if (factor is { } f)
        {
            w.Quantizer.Factors.AsSpan(0, unit / f.Values).CopyTo(factorSlots.Slice(u * unit / f.Values));
        }

        double error = 0;
        for (int i = 0; i < unit; i++)
        {
            float r = y[i] - (a * numbers[i] + c);
            error += (double)r * r;
        }

        return error;
    }

    // Chunk j of a level searched again: its least-squares number given the codes (each value λ · α + β, λ its number),
    // rounded, and (a grid of few numbers) its neighbours, the codes chosen again for each and the error measured.
    private void Search(Work w, Level level, int j)
    {
        int per = level.Values / unit, u0 = j * per;
        var ranks = w.Ranks[level.Index];
        int r0 = ranks[j];
        double current = 0, saa = 0, sab = 0;
        double sign = minsAdded ? 1 : -1;
        for (int u = u0; u < u0 + per; u++)
        {
            current += w.Error[u];
            double scaleOthers = 1, minOthers = 1, scale = 1, min = 1;
            foreach (var other in scales)
            {
                double value = other.Grid.Values[w.Ranks[other.Index][u * unit / other.Values]];
                scale *= value;
                scaleOthers *= other == level ? 1 : value;
            }

            foreach (var other in mins)
            {
                double value = other.Grid.Values[w.Ranks[other.Index][u * unit / other.Values]];
                min *= value;
                minOthers *= other == level ? 1 : value;
            }

            double c = mins.Length == 0 ? 0 : sign * min;
            for (int v = u * unit; v < (u + 1) * unit; v++)
            {
                double g = w.G[v];
                double alpha = (level.Scale ? scaleOthers * g : 0) + (level.Min ? sign * minOthers : 0);
                double beta = (level.Scale ? 0 : scale * g) + (level.Min ? 0 : c);
                saa += alpha * alpha;
                sab += (w.X[v] - beta) * alpha;
            }
        }

        Span<int> candidates = stackalloc int[1 + 2 * Neighbours];
        int count = Add(candidates, 0, saa > 0 ? level.Grid.Nearest(sab / saa) : r0, r0, level.Grid.Count);
        if (level.Grid.Count <= SmallGrid)
        {
            for (int step = 1; step <= neighbours; step++)
            {
                count = Add(candidates, count, r0 - step, r0, level.Grid.Count);
                count = Add(candidates, count, r0 + step, r0, level.Grid.Count);
            }
        }

        double best = current;
        int bestRank = r0;
        int first = u0 * unit, values = per * unit, slotFirst = first / codeWidth, slotCount = values / codeWidth;
        int factorWidth = factor?.Values ?? 1, factorFirst = factor is null ? 0 : first / factorWidth, factorCount = factor is null ? 0 : values / factorWidth;
        for (int i = 0; i < count; i++)
        {
            ranks[j] = candidates[i];
            double e = 0;
            for (int u = u0; u < u0 + per && e < best; u++)
            {
                UnitFactors(w, u, out float a, out float c);
                w.TryError[u] = QuantizeUnit(w, u, a, c, w.TryG, w.TryCodes, w.TryFactors);
                e += w.TryError[u];
            }

            if (e < best)
            {
                best = e;
                bestRank = candidates[i];
                w.TryG.AsSpan(first, values).CopyTo(w.BestG.AsSpan(first, values));
                w.TryCodes.AsSpan(slotFirst, slotCount).CopyTo(w.BestCodes.AsSpan(slotFirst, slotCount));
                w.TryFactors.AsSpan(factorFirst, factorCount).CopyTo(w.BestFactors.AsSpan(factorFirst, factorCount));
                w.TryError.AsSpan(u0, per).CopyTo(w.BestError.AsSpan(u0, per));
            }
        }

        ranks[j] = bestRank;
        if (bestRank != r0)
        {
            w.BestG.AsSpan(first, values).CopyTo(w.G.AsSpan(first, values));
            w.BestCodes.AsSpan(slotFirst, slotCount).CopyTo(w.Codes.AsSpan(slotFirst, slotCount));
            w.BestFactors.AsSpan(factorFirst, factorCount).CopyTo(w.Factors.AsSpan(factorFirst, factorCount));
            w.BestError.AsSpan(u0, per).CopyTo(w.Error.AsSpan(u0, per));
        }
    }

    // A rank added to a search's candidates: within the grid, not the current one, not listed already.
    private static int Add(Span<int> list, int count, int rank, int current, int ranks)
    {
        if (rank >= 0 && rank < ranks && rank != current && !list[..count].Contains(rank))
        {
            list[count++] = rank;
        }

        return count;
    }

    // The group's bytes: each code slot's planes, each level's chunks, each factor's chunks.
    private void Write(Work w, Span<byte> group)
    {
        for (int s = 0; s < slots; s++)
        {
            w.CodeBits[s] = codeGrid is { } grid ? grid.Codes[w.Codes[s]] : (ulong)w.Codes[s];
        }

        WriteCodes(w, group);
        foreach (var level in all)
        {
            var ranks = w.Ranks[level.Index];
            for (int j = 0; j < level.Chunks; j++)
            {
                level.Fields.Write(group, j, level.Grid.Codes[ranks[j]]);
            }
        }

        if (factor is { } f)
        {
            for (int c = 0; c < n / f.Values; c++)
            {
                f.Fields.Write(group, c, (ulong)w.Factors[c]);
            }
        }
    }

    // The group cleared, then each code slot's bits (CodeBits) into its planes: the planes of bits field by field, each plane
    // of digits its bytes whole.
    private void WriteCodes(Work w, Span<byte> group)
    {
        group.Clear();
        for (int s = 0; s < slots; s++)
        {
            codes.Write(group, s, w.CodeBits[s]);
        }

        foreach (var plane in digitPlanes)
        {
            plane.Write(group, w.CodeBits, w.Digits);
        }
    }

    // Fields of bits in a group: field f's pieces at [start[f], start[f + 1]), each its first bit and width, a field's number's
    // lowest bits the first piece's (or, with `lows`, each piece the number's bits from lows[piece]: a code's planes).
    private sealed class Fields(int[] start, int[] bits, int[] widths, int[]? lows = null)
    {
        public void Write(Span<byte> group, int field, ulong value)
        {
            for (int p = start[field]; p < start[field + 1]; p++)
            {
                int b = bits[p];
                ulong piece = lows is null ? value : value >> lows[p];
                for (int i = 0; i < widths[p]; i++, b++, piece >>= 1)
                {
                    if ((piece & 1) != 0)
                    {
                        group[b >> 3] |= (byte)(1 << (b & 7));
                    }
                }

                if (lows is null)
                {
                    value = widths[p] >= 64 ? 0 : value >> widths[p];
                }
            }
        }

        // Marks the fields' bits as owned; false where a bit is owned already or past the group (with `repeats`, a piece
        // listed again whole is the same field: codes sharing it).
        public bool Claim(bool[] owned, bool repeats = false)
        {
            var seen = new HashSet<(int, int)>();
            for (int p = 0; p < bits.Length; p++)
            {
                if (repeats && !seen.Add((bits[p], widths[p])))
                {
                    continue;
                }

                for (int b = bits[p]; b < bits[p] + widths[p]; b++)
                {
                    if (b < 0 || b >= owned.Length || owned[b])
                    {
                        return false;
                    }

                    owned[b] = true;
                }
            }

            return true;
        }
    }

    // A plane of digits (CodePlane.Radix): each code slot's digit of a byte; a byte written from its digits as the byte that
    // reads as them (each number of digits' first such byte, found once from the plane's own reading of every byte).
    private sealed class DigitPlane
    {
        private readonly int radix, digits, low, size;
        private readonly ulong mask;
        private readonly int[] slotByte, slotDigit, touched;
        private readonly byte[] byteOf;

        private DigitPlane(int radix, int digits, int low, ulong mask, int size, int[] slotByte, int[] slotDigit, byte[] byteOf)
        {
            (this.radix, this.digits, this.low, this.mask, this.size, this.slotByte, this.slotDigit, this.byteOf) = (radix, digits, low, mask, size, slotByte, slotDigit, byteOf);
            touched = [.. slotByte.Distinct().Order()];
        }

        // The plane at `low` of the code, its slots' (byte, digit), in groups of `size` bytes; null where a number of digits
        // has no byte that reads as it.
        public static DigitPlane? Of(CodePlane plane, int low, List<(int Byte, int Digit)> slots, int size)
        {
            int radix = plane.Radix, digits = plane.Digits, numbers = 1;
            for (int i = 0; i < digits; i++)
            {
                numbers *= radix;
            }

            var byteOf = new int[numbers];
            byteOf.AsSpan().Fill(-1);
            for (int b = 0; b < 256; b++)
            {
                int number = 0;
                for (int i = 0; i < digits; i++)
                {
                    number = number * radix + plane.Digit((byte)b, i);
                }

                if (byteOf[number] < 0)
                {
                    byteOf[number] = b;
                }
            }

            return byteOf.Any(b => b < 0) ? null
                : new DigitPlane(radix, digits, low, (1UL << plane.Bits) - 1, size, [.. slots.Select(s => s.Byte)], [.. slots.Select(s => s.Digit)], [.. byteOf.Select(b => (byte)b)]);
        }

        // Whether a code's digit in this plane is one the radix holds.
        public bool Holds(ulong code) => (code >> low & mask) < (ulong)radix;

        // Each touched byte from its slots' digits (a digit no slot gives: 0).
        public void Write(Span<byte> group, ReadOnlySpan<ulong> codes, Span<int> scratch)
        {
            var of = scratch[..(size * digits)];
            of.Clear();
            for (int s = 0; s < slotByte.Length; s++)
            {
                of[slotByte[s] * digits + slotDigit[s]] = (int)(codes[s] >> low & mask);
            }

            foreach (int b in touched)
            {
                int number = 0;
                for (int i = 0; i < digits; i++)
                {
                    number = number * radix + of[b * digits + i];
                }

                group[b] = byteOf[number];
            }
        }

        // Its bytes, whole.
        public bool Claim(bool[] owned)
        {
            foreach (int b in touched)
            {
                for (int i = 8 * b; i < 8 * b + 8; i++)
                {
                    if (i >= owned.Length || owned[i])
                    {
                        return false;
                    }

                    owned[i] = true;
                }
            }

            return true;
        }
    }

    // A level of scales or minimums (a minimum stored in a scale's bits is that scale's level, both Scale and Min).
    private sealed class Level(BlockScale source, Grid grid, int values, int chunks, Fields fields)
    {
        public BlockScale Source { get; } = source;

        public Grid Grid { get; } = grid;

        public GridQuantizer Quantizer { get; } = new(grid);

        public int Values { get; } = values;

        public int Chunks { get; } = chunks;

        public Fields Fields { get; } = fields;

        public int Index { get; set; }

        public bool Scale { get; set; }

        public bool Min { get; set; }
    }

    // A level of vector factors (a codebook scale: a sign mask a group of values, say).
    private sealed class FactorLevel(Book book, int values, Fields fields)
    {
        public Book Book { get; } = book;

        public int Values { get; } = values;

        public Fields Fields { get; } = fields;
    }

    /// <summary>
    /// The distinct finite numbers of a number type of up to 16 bits (the codes a filter holds), its bias added, ascending,
    /// each with a code giving it; a signed float's −0 and +0 apart. Nearest rounds to the nearest number, a tie to the even
    /// code (its lowest bit clear: a float's even mantissa, an integer's even number), a zero keeping its sign;
    /// <see cref="Round"/> also rounds a float past its largest as its specials define (an infinity, a NaN, or the largest).
    /// </summary>
    internal sealed class Grid
    {
        private static readonly System.Collections.Concurrent.ConcurrentDictionary<(NumberType Type, double Bias), Grid?> Made = new();

        private readonly NumberType type;
        private readonly int[] ranks;
        private readonly double[] mids;
        private readonly bool uniform, half;
        private readonly double first, step, inverse, overflow;
        private readonly int negativeZero = -1, positiveZero = -1;

        private Grid(NumberType type, double bias, List<(double Value, ulong Code)> numbers)
        {
            this.type = type;
            static bool Negative(double v) => v == 0 && double.IsNegative(v);
            numbers.Sort((p, q) => p.Value != q.Value ? p.Value.CompareTo(q.Value) : Negative(p.Value) != Negative(q.Value) ? (Negative(p.Value) ? -1 : 1) : p.Code.CompareTo(q.Code));
            var values = new List<double>();
            var codes = new List<ulong>();
            foreach (var (value, code) in numbers)
            {
                if (values.Count == 0 || value != values[^1] || Negative(value) != Negative(values[^1]))
                {
                    values.Add(value);
                    codes.Add(code);
                }
            }

            Values = [.. values];
            Floats = [.. values.Select(v => (float)v)];
            Codes = [.. codes];
            mids = [.. Values.Zip(Values.Skip(1), (p, q) => (p + q) / 2)];
            ranks = new int[1 << type.Bits];
            ranks.AsSpan().Fill(-1);
            for (int r = 0; r < Values.Length; r++)
            {
                if (Values[r] == 0 && Negative(Values[r]))
                {
                    negativeZero = r;
                }
                else if (Values[r] == 0)
                {
                    positiveZero = r;
                }
            }

            foreach (var (value, code) in numbers)
            {
                ranks[code] = value == 0 && negativeZero >= 0 ? (Negative(value) ? negativeZero : positiveZero) : Array.BinarySearch(Values, value);
            }

            if (negativeZero < 0 || positiveZero < 0)
            {
                negativeZero = positiveZero = -1;                           // one zero: no sign to keep
            }

            first = Values[0];
            step = Values.Length > 1 ? Values[1] - Values[0] : 1;
            inverse = 1 / step;
            uniform = Values.Length > 1 && Values.Select((v, i) => v == first + i * step).All(x => x);
            half = bias == 0 && type is { Kind: NumberKind.Float, Bits: 16, Exponent: 5, Mantissa: 10, ExponentBias: 15, Signed: true, Subnormals: true, Specials: FloatSpecials.Ieee };

            // A float's overflow: past its largest number by half the step the next exponent would take (the largest plus
            // the largest's own step, halved).
            if (type.Kind == NumberKind.Float && bias == 0 && Values[^1] > 0)
            {
                double largest = Values[^1];
                overflow = largest + Math.ScaleB(1, Math.ILogB(largest) - type.Mantissa) / 2;
            }
        }

        public double[] Values { get; }

        public float[] Floats { get; }

        public ulong[] Codes { get; }

        public int Count => Values.Length;

        public bool HasNegative => Values[0] < 0;

        public bool HasPositive => Values[^1] > 0;

        /// <summary>The grid of <paramref name="type"/> plus <paramref name="bias"/> (made once), or null (not one number of up to 16 bits, or no finite number).</summary>
        public static Grid? Of(NumberType type, double bias) => type.Width == 1 && type.Bits <= 16 && type.Valid ? Made.GetOrAdd((type, bias), static key => Make(key.Type, key.Bias, null)) : null;

        /// <summary>The grid of <paramref name="type"/> plus <paramref name="bias"/> over the codes <paramref name="holds"/> takes (all where null).</summary>
        public static Grid? Of(NumberType type, double bias, Func<ulong, bool>? holds) => holds is null ? Of(type, bias)
            : type.Width == 1 && type.Bits <= 16 && type.Valid ? Make(type, bias, holds) : null;

        private static Grid? Make(NumberType type, double bias, Func<ulong, bool>? holds)
        {
            var numbers = new List<(double, ulong)>();
            for (ulong code = 0; code < 1UL << type.Bits; code++)
            {
                double value = bias == 0 ? type.Value(code) : type.Value(code) + bias;    // −0 kept (−0 + 0 is +0)
                if (double.IsFinite(value) && (holds?.Invoke(code) ?? true))
                {
                    numbers.Add((value, code));
                }
            }

            return numbers.Count == 0 ? null : new Grid(type, bias, numbers);
        }

        /// <summary>The rank of a code (−1: not a finite number of the grid).</summary>
        public int RankOf(ulong code) => code < (ulong)ranks.Length ? ranks[code] : -1;

        /// <summary>The rank of the number nearest <paramref name="x"/> (NaN as 0), a tie to the even code, a zero keeping its sign.</summary>
        public int Nearest(double x)
        {
            if (double.IsNaN(x))
            {
                x = 0;
            }

            if (half)
            {
                var h = (Half)x;                                            // to nearest, ties to even
                return Half.IsInfinity(h) ? (x > 0 ? Count - 1 : 0) : ranks[BitConverter.HalfToUInt16Bits(h)];
            }

            if (x == 0 && negativeZero >= 0)
            {
                return double.IsNegative(x) ? negativeZero : positiveZero;
            }

            if (uniform)
            {
                return Uniform(x);
            }

            int lo = 0, hi = Count - 1;
            while (lo < hi)
            {
                int m = (lo + hi) >> 1;
                if (x > mids[m])
                {
                    lo = m + 1;
                }
                else if (x < mids[m])
                {
                    hi = m;
                }
                else
                {
                    return Even(m, m + 1);
                }
            }

            return lo;
        }

        // Evenly spaced numbers: the nearest by the step, a tie to the even code.
        private int Uniform(double x)
        {
            double q = (x - first) * inverse;
            int last = Count - 1;
            if (!(q > 0))
            {
                return 0;
            }

            if (q >= last)
            {
                return last;
            }

            double floor = Math.Floor(q), fraction = q - floor;
            int r = (int)floor;
            return fraction > 0.5 ? r + 1 : fraction < 0.5 ? r : Even(r, r + 1);
        }

        private int Even(int a, int b) => (Codes[a] & 1) == 0 ? a : b;

        /// <summary>
        /// The code of <paramref name="x"/> as the number type defines rounding to it: <see cref="Nearest"/>, and a float past
        /// its largest number by half its next step or more (a tie to the even code) its infinity of that sign (IEEE's
        /// specials), its NaN of that sign (a NaN code alone: no infinity) or its largest (no specials); NaN its quiet NaN of
        /// that sign (a format with no NaN: its zero). A table's or an integer's: the nearest, the largest past its range.
        /// </summary>
        public ulong Round(double x)
        {
            ulong sign = type.Signed && type.Kind == NumberKind.Float && double.IsNegative(x) ? 1UL << (type.Bits - 1) : 0;
            if (double.IsNaN(x))
            {
                return Special(sign, nan: true) ?? Codes[Nearest(0)];
            }

            if (overflow > 0 && Math.Abs(x) >= overflow && (type.Signed || x > 0))
            {
                int extreme = x > 0 ? Count - 1 : 0;
                if (Math.Abs(x) > overflow || (Codes[extreme] & 1) != 0)
                {
                    return Special(sign, nan: false) ?? Codes[extreme];
                }
            }

            return Codes[Nearest(x)];
        }

        // A float's code past its numbers: IEEE's infinity (all exponent bits) or quiet NaN (and the mantissa's top bit), the
        // all-ones NaN of a format with no infinity; null for a format with neither.
        private ulong? Special(ulong sign, bool nan)
        {
            if (type.Kind != NumberKind.Float)
            {
                return null;
            }

            ulong exponent = ((1UL << type.Exponent) - 1) << type.Mantissa;
            return type.Specials switch
            {
                FloatSpecials.Ieee => sign | exponent | (nan && type.Mantissa > 0 ? 1UL << (type.Mantissa - 1) : 0),
                FloatSpecials.AllOnesNaN => sign | exponent | ((1UL << type.Mantissa) - 1),
                _ => null,
            };
        }

        /// <summary>
        /// Each target (y − c0) · t's nearest number into <paramref name="numbers"/> and its rank into <paramref name="ranks"/>
        /// (none where empty): <see cref="Nearest"/> a value, evenly spaced numbers by their step in one loop.
        /// </summary>
        public void Quantize(ReadOnlySpan<float> y, float t, float c0, Span<float> numbers, Span<int> ranks)
        {
            bool keep = !ranks.IsEmpty;
            if (uniform)
            {
                for (int i = 0; i < y.Length; i++)
                {
                    int r = Uniform((y[i] - c0) * t);
                    numbers[i] = Floats[r];
                    if (keep)
                    {
                        ranks[i] = r;
                    }
                }

                return;
            }

            for (int i = 0; i < y.Length; i++)
            {
                int r = Nearest((y[i] - c0) * t);
                numbers[i] = Floats[r];
                if (keep)
                {
                    ranks[i] = r;
                }
            }
        }
    }

    /// <summary>
    /// A codebook of up to 16-bit codes (its offset added to each entry's numbers): each code's vector of Width numbers, and
    /// the entry nearest a target vector under weights. Its entries are searched in parts by their indices' top bits, as many
    /// bits as make every part's patterns few (each value one of the numbers the part's entries hold: IQ1's 12-bit index, its
    /// top bit the delta's sign, is two parts of three numbers a value); a part of few patterns lists each pattern's nearest
    /// entries once, and a target's nearest there is the best of its own pattern's list (each target value rounded to the
    /// nearest number); any other part is measured entry by entry. Entries the planes cannot hold are never chosen, and a
    /// constraint (index bits that must take given values: a field codes share) searches the entries that agree.
    /// </summary>
    internal sealed class Book
    {
        private static readonly System.Collections.Concurrent.ConcurrentDictionary<(NumberType Type, double Offset), Book?> Made = new();

        // Entries listed a pattern, the most patterns a part lists, the most index bits the parts split by.
        private const int Listed = 16, PatternLimit = 1 << 16, MostSplit = 4;

        private readonly bool[]? held;
        private readonly Part[] parts;
        private readonly int bits, split;

        private Book(NumberType type, double offset, Func<ulong, bool>? holds)
        {
            bits = type.Bits;
            Entries = 1 << bits;
            Width = type.Width;
            Table = [.. type.Table!.Take(Entries * Width).Select(v => (float)(v + offset))];
            held = holds is null ? null : [.. Enumerable.Range(0, Entries).Select(e => holds((ulong)e))];
            Levels = [.. Enumerable.Range(0, Entries).Where(Holds).SelectMany(e => Table.Skip(e * Width).Take(Width)).Select(v => (double)v).Distinct().Order()];
            parts = [new Part(this, 0, Entries)];
            for (int s = 1; s <= Math.Min(MostSplit, bits) && !parts.All(p => p.Listable); s++)
            {
                var made = Enumerable.Range(0, 1 << s).Select(k => new Part(this, k << (bits - s), 1 << (bits - s))).ToArray();
                if (made.All(p => p.Listable))
                {
                    (parts, split) = (made, s);
                }
            }
        }

        public int Entries { get; }

        public int Width { get; }

        /// <summary>Entry e's number i at e · Width + i.</summary>
        public float[] Table { get; }

        /// <summary>The distinct numbers the entries it may choose hold, ascending.</summary>
        public double[] Levels { get; }

        /// <summary>The codebook of <paramref name="type"/> plus <paramref name="offset"/> (over the codes <paramref name="holds"/> takes, all where null), or null (not a table of up to 16 bits).</summary>
        public static Book? Of(NumberType type, double offset, Func<ulong, bool>? holds = null) => type is { Kind: NumberKind.Table, Bits: <= 16, Table: not null } && type.Valid
            ? holds is null ? Made.GetOrAdd((type, offset), static key => new Book(key.Type, key.Offset, null)) : new Book(type, offset, holds)
            : null;

        private bool Holds(int e) => held is null || held[e];

        /// <summary>The entry nearest <paramref name="target"/> (Width values) under <paramref name="weight"/>: the least Σ w (t − e)².</summary>
        public int Nearest(ReadOnlySpan<float> target, ReadOnlySpan<float> weight) => Nearest(target, weight, 0, 0, out _);

        /// <summary>The nearest entry whose index bits under <paramref name="mask"/> are <paramref name="value"/>'s, and its distance.</summary>
        public int Nearest(ReadOnlySpan<float> target, ReadOnlySpan<float> weight, ulong mask, ulong value, out float distance)
        {
            int best = -1;
            distance = float.PositiveInfinity;
            ulong top = split == 0 ? 0 : ((1UL << split) - 1) << (bits - split), within = mask & ~top;
            foreach (var part in parts)
            {
                if ((((ulong)part.First ^ value) & mask & top) == 0)
                {
                    part.Search(this, target, weight, within, value, ref best, ref distance);
                }
            }

            if (best < 0)
            {
                Measure(target, weight, 0, Entries, mask, value, ref best, ref distance);
            }

            return Math.Max(best, 0);
        }

        // Entries first..first + count that it holds and that agree with the constraint, the nearest kept in (best, distance).
        private void Measure(ReadOnlySpan<float> target, ReadOnlySpan<float> weight, int first, int count, ulong mask, ulong value, ref int best, ref float distance)
        {
            for (int e = first; e < first + count; e++)
            {
                if (Holds(e) && (((ulong)e ^ value) & mask) == 0)
                {
                    Consider(target, weight, e, ref best, ref distance);
                }
            }
        }

        // Entry e against the best so far (its distance summed until it passes it).
        private void Consider(ReadOnlySpan<float> target, ReadOnlySpan<float> weight, int e, ref int best, ref float distance)
        {
            var entry = Table.AsSpan(e * Width, Width);
            float sum = 0f;
            for (int i = 0; i < Width && sum < distance; i++)
            {
                float r = target[i] - entry[i];
                sum += weight[i] * r * r;
            }

            if (sum < distance)
            {
                distance = sum;
                best = e;
            }
        }

        // Entries First..First + Count of the codebook: the numbers its held entries hold, and its patterns' lists where few.
        private sealed class Part
        {
            private readonly double[] levels;
            private readonly float[] mids;
            private readonly int listed, heldCount;
            private readonly long patterns;
            private int[]? lists;

            public Part(Book book, int first, int count)
            {
                First = first;
                Count = count;
                var entries = Enumerable.Range(first, count).Where(book.Holds).ToArray();
                heldCount = entries.Length;
                levels = [.. entries.SelectMany(e => book.Table.Skip(e * book.Width).Take(book.Width)).Select(v => (double)v).Distinct().Order()];
                mids = [.. levels.Zip(levels.Skip(1), (p, q) => (float)((p + q) / 2))];
                listed = Math.Min(heldCount, Listed);
                double many = Math.Pow(Math.Max(1, levels.Length), book.Width);
                patterns = heldCount > listed && many <= PatternLimit ? (long)many : 0;
            }

            public int First { get; }

            public int Count { get; }

            // Whether its search is short: few entries, or few patterns to list.
            public bool Listable => heldCount <= Listed || patterns > 0;

            public void Search(Book book, ReadOnlySpan<float> target, ReadOnlySpan<float> weight, ulong mask, ulong value, ref int best, ref float distance)
            {
                if (patterns == 0)
                {
                    book.Measure(target, weight, First, Count, mask, value, ref best, ref distance);
                    return;
                }

                var list = Volatile.Read(ref lists) ?? List(book);
                int pattern = 0, scale = 1;
                for (int i = 0; i < book.Width; i++)
                {
                    pattern += Level(target[i]) * scale;
                    scale *= levels.Length;
                }

                int before = best;
                for (int c = pattern * listed; c < (pattern + 1) * listed; c++)
                {
                    int e = list[c];
                    if ((((ulong)e ^ value) & mask) == 0)
                    {
                        book.Consider(target, weight, e, ref best, ref distance);
                    }
                }

                if (mask != 0 && best == before)
                {
                    book.Measure(target, weight, First, Count, mask, value, ref best, ref distance);   // none of the list agrees
                }
            }

            // The index of the number nearest x.
            private int Level(float x)
            {
                int lo = 0, hi = levels.Length - 1;
                while (lo < hi)
                {
                    int m = (lo + hi) >> 1;
                    if (x > mids[m])
                    {
                        lo = m + 1;
                    }
                    else
                    {
                        hi = m;
                    }
                }

                return lo;
            }

            // Each pattern's nearest held entries (by squared distance, then index), made once.
            private int[] List(Book book)
            {
                var made = new int[patterns * listed];
                int width = book.Width;
                Parallel.For(0, patterns, ComputeResources.ParallelOptions, () => (new float[width], new float[listed], new int[listed]), (p, _, scratch) =>
                {
                    var (point, distances, indices) = scratch;
                    long rest = p;
                    for (int i = 0; i < width; i++, rest /= levels.Length)
                    {
                        point[i] = (float)levels[rest % levels.Length];
                    }

                    int kept = 0;
                    for (int e = First; e < First + Count; e++)
                    {
                        if (!book.Holds(e))
                        {
                            continue;
                        }

                        float distance = 0f;
                        for (int i = 0; i < width; i++)
                        {
                            float r = point[i] - book.Table[e * width + i];
                            distance += r * r;
                        }

                        if (kept == listed && distance >= distances[kept - 1])
                        {
                            continue;
                        }

                        int at = kept == listed ? kept - 1 : kept++;
                        while (at > 0 && distances[at - 1] > distance)
                        {
                            distances[at] = distances[at - 1];
                            indices[at] = indices[at - 1];
                            at--;
                        }

                        distances[at] = distance;
                        indices[at] = e;
                    }

                    indices.AsSpan(0, listed).CopyTo(made.AsSpan((int)(p * listed), listed));
                    return scratch;
                }, static _ => { });
                Volatile.Write(ref lists, made);
                return made;
            }
        }
    }

    // What a search quantizes with: targets to the nearest codes' numbers, and the numbers' extremes and steps beside them.
    private abstract class Quantizer
    {
        public double Top { get; protected init; }

        public double Bottom { get; protected init; }

        public double TopStep { get; protected init; }

        public double BottomStep { get; protected init; }

        // The candidates' half steps from the extremes (Solve).
        public float[] Steps { get; protected init; } = ScaleSteps;

        // Each target (y − c0) · t to its nearest codes' number in g.
        public abstract void Run(ReadOnlySpan<float> y, float t, float c0, Span<float> g);
    }

    // A level's numbers (the factoring's children).
    private sealed class GridQuantizer : Quantizer
    {
        private readonly Grid grid;

        public GridQuantizer(Grid grid)
        {
            this.grid = grid;
            var v = grid.Values;
            Top = v[^1];
            Bottom = v[0];
            TopStep = v.Length > 1 ? v[^1] - v[^2] : 1;
            BottomStep = v.Length > 1 ? v[1] - v[0] : 1;
        }

        public override void Run(ReadOnlySpan<float> y, float t, float c0, Span<float> g) => grid.Quantize(y, t, c0, g, []);
    }

    // A unit's codes (a thread's own: it keeps the codes it chose): a grid's nearest numbers, or a codebook's nearest entries
    // (with vector factors: each factor's entry nearest the targets' signs, each code's entry nearest the targets over the
    // factors, then each factor's again given the entries and the entries again, kept where nearer).
    private sealed class CodeQuantizer : Quantizer
    {
        private readonly DescribedEncoder e;
        private readonly int[] savedCodes, savedFactors;
        private readonly float[] targets, saved, target, weight;

        public CodeQuantizer(DescribedEncoder encoder)
        {
            e = encoder;
            Top = e.top;
            Bottom = e.bottom;
            TopStep = e.topStep;
            BottomStep = e.bottomStep;
            Steps = e.codeBook is null ? ScaleSteps : BookSteps;
            Codes = new int[e.unit];
            Factors = new int[e.unit];
            savedCodes = new int[e.unit];
            savedFactors = new int[e.unit];
            targets = new float[e.unit];
            saved = new float[e.unit];
            int width = Math.Max(e.codeWidth, e.factor?.Values ?? 1);
            target = new float[width];
            weight = new float[width];
        }

        // The last run's codes (a value's rank, or an entry's index, a slot each) and factors (a chunk each).
        public int[] Codes { get; }

        public int[] Factors { get; }

        // The unit a run quantizes (its codes' slots from Unit · its slots).
        public int Unit { get; set; }

        public override void Run(ReadOnlySpan<float> y, float t, float c0, Span<float> g)
        {
            if (e.codeGrid is { } grid)
            {
                if (e.sharedGrids is not { } subs)
                {
                    grid.Quantize(y, t, c0, g, Codes);
                    return;
                }

                // Codes sharing a field: each group's codes by the number of the field that leaves them the least error.
                var leaders = e.leader!;
                int s0 = Unit * y.Length;
                for (int i = 0; i < y.Length; i++)
                {
                    if (leaders[s0 + i] != s0 + i)
                    {
                        continue;
                    }

                    int bestK = 0;
                    double bestError = double.PositiveInfinity;
                    for (int k = 0; k < subs.Length; k++)
                    {
                        double error = 0;
                        for (int j = i; j < y.Length; j++)
                        {
                            if (leaders[s0 + j] == s0 + i)
                            {
                                float zj = (y[j] - c0) * t, r = zj - subs[k].Floats[subs[k].Nearest(zj)];
                                error += (double)r * r;
                            }
                        }

                        if (error < bestError)
                        {
                            (bestError, bestK) = (error, k);
                        }
                    }

                    var sub = subs[bestK];
                    for (int j = i; j < y.Length; j++)
                    {
                        if (leaders[s0 + j] == s0 + i)
                        {
                            int r = sub.Nearest((y[j] - c0) * t);
                            Codes[j] = grid.RankOf(sub.Codes[r]);
                            g[j] = sub.Floats[r];
                        }
                    }
                }

                return;
            }

            var z = targets.AsSpan(0, y.Length);
            for (int i = 0; i < y.Length; i++)
            {
                z[i] = (y[i] - c0) * t;
            }

            var book = e.codeBook!;
            if (e.factor is not { } f)
            {
                Entries(book, null, z, g);
                return;
            }

            var factors = f.Book;
            int fw = factors.Width, width = book.Width;
            for (int c = 0; c < z.Length / fw; c++)
            {
                for (int i = 0; i < fw; i++)
                {
                    float zi = z[c * fw + i];
                    target[i] = zi < 0 ? -1f : 1f;
                    weight[i] = zi * zi;
                }

                Factors[c] = factors.Nearest(target.AsSpan(0, fw), weight.AsSpan(0, fw));
            }

            double first = Entries(book, factors, z, g);
            bool changed = false;
            for (int c = 0; c < z.Length / fw; c++)
            {
                for (int i = 0; i < fw; i++)
                {
                    int v = c * fw + i;
                    float number = book.Table[Codes[v / width] * width + v % width];
                    target[i] = number != 0f ? z[v] / number : 0f;
                    weight[i] = number * number;
                }

                int k = factors.Nearest(target.AsSpan(0, fw), weight.AsSpan(0, fw));
                if (k != Factors[c])
                {
                    if (!changed)
                    {
                        Codes.AsSpan(0, z.Length / width).CopyTo(savedCodes);
                        Factors.AsSpan(0, z.Length / fw).CopyTo(savedFactors);
                        g.CopyTo(saved);
                        changed = true;
                    }

                    Factors[c] = k;
                }
            }

            if (changed && !(Entries(book, factors, z, g) < first))
            {
                savedCodes.AsSpan(0, z.Length / width).CopyTo(Codes);
                savedFactors.AsSpan(0, z.Length / fw).CopyTo(Factors);
                saved.AsSpan(0, z.Length).CopyTo(g);
            }
        }

        // Each entry's code nearest its targets over the factors (none: 1; codes sharing a field: each group's by the number of
        // the field that leaves them the least error); numbers into g; the squared distance to z.
        private double Entries(Book book, Book? factors, ReadOnlySpan<float> z, Span<float> g)
        {
            int width = book.Width, fw = factors?.Width ?? 1, count = z.Length / width;
            if (e.leader is not { } leaders)
            {
                for (int s = 0; s < count; s++)
                {
                    Codes[s] = Entry(book, factors, z, s, 0, 0, out _);
                }
            }
            else
            {
                int s0 = Unit * count;
                for (int s = 0; s < count; s++)
                {
                    if (leaders[s0 + s] != s0 + s)
                    {
                        continue;
                    }

                    ulong bestValue = 0;
                    double bestError = double.PositiveInfinity;
                    foreach (ulong value in e.sharedValues)
                    {
                        double error = 0;
                        for (int j = s; j < count; j++)
                        {
                            if (leaders[s0 + j] == s0 + s)
                            {
                                Entry(book, factors, z, j, e.sharedMask, value, out float distance);
                                error += distance;
                            }
                        }

                        if (error < bestError)
                        {
                            (bestError, bestValue) = (error, value);
                        }
                    }

                    for (int j = s; j < count; j++)
                    {
                        if (leaders[s0 + j] == s0 + s)
                        {
                            Codes[j] = Entry(book, factors, z, j, e.sharedMask, bestValue, out _);
                        }
                    }
                }
            }

            double total = 0;
            for (int s = 0; s < count; s++)
            {
                int k = Codes[s];
                for (int i = 0; i < width; i++)
                {
                    int v = s * width + i;
                    float phi = factors is null ? 1f : factors.Table[Factors[v / fw] * fw + v % fw];
                    g[v] = phi * book.Table[k * width + i];
                    float r = z[v] - g[v];
                    total += (double)r * r;
                }
            }

            return total;
        }

        // Slot s's entry nearest its targets over the factors, among the entries whose index bits under mask are value's.
        private int Entry(Book book, Book? factors, ReadOnlySpan<float> z, int s, ulong mask, ulong value, out float distance)
        {
            int width = book.Width, fw = factors?.Width ?? 1;
            for (int i = 0; i < width; i++)
            {
                int v = s * width + i;
                float phi = factors is null ? 1f : factors.Table[Factors[v / fw] * fw + v % fw];
                target[i] = phi != 0f ? z[v] / phi : 0f;
                weight[i] = phi * phi;
            }

            return book.Nearest(target.AsSpan(0, width), weight.AsSpan(0, width), mask, value, out distance);
        }
    }

    // A thread's working arrays for one encoder.
    private sealed class Work
    {
        [ThreadStatic]
        private static Work? last;

        private Work(DescribedEncoder e)
        {
            Owner = e;
            int n = e.n;
            X = new float[n];
            G = new float[n];
            TryG = new float[n];
            BestG = new float[n];
            Ones = new float[n];
            Ones.AsSpan().Fill(1f);
            Codes = new int[n];
            TryCodes = new int[n];
            BestCodes = new int[n];
            Factors = new int[n];
            TryFactors = new int[n];
            BestFactors = new int[n];
            A = new float[e.units];
            C = new float[e.units];
            WA = new float[e.units];
            WC = new float[e.units];
            T = new float[e.units];
            Wt = new float[e.units];
            Error = new double[e.units];
            TryError = new double[e.units];
            BestError = new double[e.units];
            Ranks = [.. e.all.Select(l => new int[l.Chunks])];
            LevelTargets = [.. e.all.Select(l => new float[l.Chunks])];
            LevelWeights = [.. e.all.Select(l => new float[l.Chunks])];
            Quantizer = new CodeQuantizer(e);
            CodeBits = new ulong[n];
            Digits = new int[8 * e.size];
        }

        public DescribedEncoder Owner { get; }

        // Each code slot's bits as written, and a plane of digits' digits a byte.
        public ulong[] CodeBits { get; }

        public int[] Digits { get; }

        public float[] X { get; }

        public float[] G { get; }

        public float[] TryG { get; }

        public float[] BestG { get; }

        public float[] Ones { get; }

        public int[] Codes { get; }

        public int[] TryCodes { get; }

        public int[] BestCodes { get; }

        public int[] Factors { get; }

        public int[] TryFactors { get; }

        public int[] BestFactors { get; }

        public float[] A { get; }

        public float[] C { get; }

        public float[] WA { get; }

        public float[] WC { get; }

        public float[] T { get; }

        public float[] Wt { get; }

        public double[] Error { get; }

        public double[] TryError { get; }

        public double[] BestError { get; }

        public int[][] Ranks { get; }

        public float[][] LevelTargets { get; }

        public float[][] LevelWeights { get; }

        public CodeQuantizer Quantizer { get; }

        public static Work For(DescribedEncoder e) => last is { } w && w.Owner == e ? w : last = new Work(e);
    }
}
