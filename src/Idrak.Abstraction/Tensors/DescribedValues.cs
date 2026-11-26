// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Idrak.Abstraction;

/// <summary>
/// The default reading of any described type (<see cref="IElementType.Read(ReadOnlySpan{byte}, Span{float})"/>): each
/// value exactly as <see cref="TypeDescription.ValueOf"/> reads it (the same float64 operations in the same order, so the
/// same bits), made once a description into a plan of where each value's fields lie, a table of its codes' numbers (codes
/// of up to 16 bits) and each scale's chunks; a machine format (<see cref="NativeKinds"/>) of 32 or 64 bits read as the
/// machine reads it (its value, a NaN's payload kept).
/// </summary>
internal sealed class DescribedReader
{
    private static readonly ConditionalWeakTable<TypeDescription, DescribedReader> Made = new();

    private readonly TypeDescription d;
    private readonly int n, size, planes, width;
    private readonly NativeKind native;
    private readonly int[] placeByte, placeShift;
    private readonly int[] planeBits, planeLow, planeRadix;
    private readonly CodePlane[] planeOf;
    private readonly double[]? codeNumbers;
    private readonly float[]? plainFloats;
    private readonly Factor[] scales, mins;
    private readonly bool minsAdded, oneByte, twoBytes;

    private DescribedReader(TypeDescription d)
    {
        this.d = d;
        n = d.UnitValues;
        size = d.UnitBytes;
        width = d.Code.Width;
        native = BitConverter.IsLittleEndian ? NativeKinds.Of(d) : NativeKind.None;
        planes = d.Planes.Count;
        planeOf = [.. d.Planes];
        planeBits = [.. d.Planes.Select(p => p.Bits)];
        planeRadix = [.. d.Planes.Select(p => p.Radix)];
        planeLow = new int[planes];
        for (int p = 1; p < planes; p++)
        {
            planeLow[p] = planeLow[p - 1] + planeBits[p - 1];
        }

        // Each value's field in each plane: its byte from the group's first and its lowest bit there (a digit's place).
        placeByte = new int[n * planes];
        placeShift = new int[n * planes];
        for (int v = 0; v < n; v++)
        {
            int block = d.BlockStart(v), entry = v % d.Values / width;
            for (int p = 0; p < planes; p++)
            {
                var (at, shift) = d.Planes[p].Place(entry);
                placeByte[v * planes + p] = block + at;
                placeShift[v * planes + p] = shift;
            }
        }

        // The codes' numbers, the code offset added (as ValueOf adds it), for codes of up to 16 bits.
        if (d.Code.Bits <= 16)
        {
            int codes = 1 << d.Code.Bits;
            codeNumbers = new double[codes * width];
            for (int c = 0; c < codes; c++)
            {
                for (int i = 0; i < width; i++)
                {
                    codeNumbers[c * width + i] = d.Code.Value((ulong)c, i) + d.CodeOffset;
                }
            }
        }

        scales = [.. (d.Scales ?? []).Select(s => new Factor(d, s))];
        mins = [.. (d.Mins ?? []).Select(s => new Factor(d, s))];
        minsAdded = d.MinsAdded;
        if (codeNumbers is not null && scales.Length == 0 && mins.Length == 0)
        {
            plainFloats = [.. codeNumbers.Select(v => (float)v)];
        }

        // One value a block, its code a byte or a 16-bit word of its own: read straight through the table.
        bool alone = d is { Values: 1, Group: 1 } && planes == 1 && planeOf[0] is { Offset: 0, Places: null, Radix: 0, Stream: false, Group: 1 } && scales.Length == 0 && mins.Length == 0
            && width == 1 && BitConverter.IsLittleEndian;
        oneByte = alone && planeBits[0] == 8 && size == 1;
        twoBytes = alone && planeBits[0] == 16 && size == 2;
    }

    /// <summary>The reader of <paramref name="d"/>, made once.</summary>
    public static DescribedReader Of(TypeDescription d)
    {
        ArgumentNullException.ThrowIfNull(d);
        return Made.GetValue(d, static x => x.Valid ? new DescribedReader(x) : throw new ArgumentException("The description does not hold together; its values are not read.", nameof(d)));
    }

    /// <summary>The first <c>values.Length</c> values of <paramref name="stored"/> as float32 (each value's float64 rounded once).</summary>
    public void Read(ReadOnlySpan<byte> stored, Span<float> values)
    {
        Need(stored, values.Length);
        switch (native)
        {
            case NativeKind.Float32:
                MemoryMarshal.Cast<byte, float>(stored[..(values.Length * 4)]).CopyTo(values);
                return;
            case NativeKind.Float64:
                var doubles = MemoryMarshal.Cast<byte, double>(stored[..(values.Length * 8)]);
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = (float)doubles[i];
                }

                return;
            case NativeKind.Int32:
                var ints = MemoryMarshal.Cast<byte, int>(stored[..(values.Length * 4)]);
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = (float)(double)ints[i];
                }

                return;
            case NativeKind.UInt32:
                var uints = MemoryMarshal.Cast<byte, uint>(stored[..(values.Length * 4)]);
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = (float)(double)uints[i];
                }

                return;
            case NativeKind.Int64:
                var longs = MemoryMarshal.Cast<byte, long>(stored[..(values.Length * 8)]);
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = (float)(double)longs[i];
                }

                return;
            case NativeKind.UInt64:
                var ulongs = MemoryMarshal.Cast<byte, ulong>(stored[..(values.Length * 8)]);
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = (float)(double)ulongs[i];
                }

                return;
        }

        if (plainFloats is { } table && (oneByte || twoBytes))
        {
            if (oneByte)
            {
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = table[stored[i]];
                }
            }
            else
            {
                var words = MemoryMarshal.Cast<byte, ushort>(stored[..(values.Length * 2)]);
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = table[words[i]];
                }
            }

            return;
        }

        Units(stored, values.Length, values, default);
    }

    /// <summary>The first <c>values.Length</c> values of <paramref name="stored"/> as float64, as the description reads them.</summary>
    public void Read(ReadOnlySpan<byte> stored, Span<double> values)
    {
        Need(stored, values.Length);
        switch (native)
        {
            case NativeKind.Float32:
                var floats = MemoryMarshal.Cast<byte, float>(stored[..(values.Length * 4)]);
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = floats[i];
                }

                return;
            case NativeKind.Float64:
                MemoryMarshal.Cast<byte, double>(stored[..(values.Length * 8)]).CopyTo(values);
                return;
            case NativeKind.Int32:
                var ints = MemoryMarshal.Cast<byte, int>(stored[..(values.Length * 4)]);
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = ints[i];
                }

                return;
            case NativeKind.Int64:
                var longs = MemoryMarshal.Cast<byte, long>(stored[..(values.Length * 8)]);
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = longs[i];
                }

                return;
        }

        if (codeNumbers is { } table && plainFloats is not null && (oneByte || twoBytes))
        {
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = table[oneByte ? stored[i] : BinaryPrimitives.ReadUInt16LittleEndian(stored[(2 * i)..])];
            }

            return;
        }

        Units(stored, values.Length, default, values);
    }

    // Whether the bytes hold the groups the values take.
    private void Need(ReadOnlySpan<byte> stored, int count)
    {
        long bytes = (count + (long)n - 1) / n * size;
        if (stored.Length < bytes)
        {
            throw new ArgumentException($"{count} values of {n}-value groups in {size} bytes take {bytes} bytes; {stored.Length} are given.", nameof(stored));
        }
    }

    // The values group by group (floats or doubles, whichever is given), each as ValueOf reads it.
    private void Units(ReadOnlySpan<byte> stored, int count, Span<float> floats, Span<double> doubles)
    {
        int most = 0;
        foreach (var f in scales.Concat(mins))
        {
            most = Math.Max(most, f.Chunks);
        }

        double[] chunk = ArrayPool<double>.Shared.Rent(Math.Max(1, most * (scales.Length + mins.Length)));
        try
        {
            for (int g = 0; g * n < count; g++)
            {
                var unit = stored.Slice(g * size, size);
                for (int f = 0; f < scales.Length + mins.Length; f++)
                {
                    var factor = f < scales.Length ? scales[f] : mins[f - scales.Length];
                    factor.Numbers(unit, chunk.AsSpan(f * most, factor.Chunks));
                }

                int values = Math.Min(n, count - g * n);
                for (int v = 0; v < values; v++)
                {
                    ulong code = 0;
                    for (int p = 0; p < planes; p++)
                    {
                        int at = placeByte[v * planes + p], shift = placeShift[v * planes + p];
                        code |= (planeRadix[p] != 0 ? (ulong)planeOf[p].Digit(unit[at], shift) : Field(unit, at, shift, planeBits[p])) << planeLow[p];
                    }

                    int within = v % width;
                    double value = codeNumbers is { } table ? table[(int)code * width + within] : d.Code.Value(code, within) + d.CodeOffset;
                    for (int f = 0; f < scales.Length; f++)
                    {
                        value *= scales[f].Number(unit, chunk, f * most, v);
                    }

                    if (mins.Length > 0)
                    {
                        double min = 1;
                        for (int f = 0; f < mins.Length; f++)
                        {
                            min *= mins[f].Number(unit, chunk, (scales.Length + f) * most, v);
                        }

                        value = minsAdded ? value + min : value - min;
                    }

                    if (doubles.IsEmpty)
                    {
                        floats[g * n + v] = (float)value;
                    }
                    else
                    {
                        doubles[g * n + v] = value;
                    }
                }
            }
        }
        finally
        {
            ArrayPool<double>.Shared.Return(chunk);
        }
    }

    // Bits [8 · at + shift, + bits) of the group, little-endian (TypeDescription.Field's bits, a word at a time).
    private static ulong Field(ReadOnlySpan<byte> unit, int at, int shift, int bits)
    {
        if (shift + bits <= 64 && at + 8 <= unit.Length)
        {
            ulong word = BinaryPrimitives.ReadUInt64LittleEndian(unit[at..]) >> shift;
            return bits == 64 ? word : word & ((1UL << bits) - 1);
        }

        return TypeDescription.Field(unit, 8L * at + shift, bits);
    }

    // A scale or minimum: each chunk of the group (the values one number covers) its number, read once a group (a vector
    // factor's numbers read a value at a time).
    private sealed class Factor
    {
        private readonly BlockScale s;
        private readonly int[] chunkBase, chunkIndex;

        public Factor(TypeDescription d, BlockScale s)
        {
            this.s = s;
            Chunks = d.UnitValues / s.Values;
            chunkBase = new int[Chunks];
            chunkIndex = new int[Chunks];
            for (int j = 0; j < Chunks; j++)
            {
                int v = j * s.Values;
                chunkBase[j] = d.ScaleBase(s, v);
                chunkIndex[j] = d.ScaleIndex(s, v);
            }
        }

        public int Chunks { get; }

        // Each chunk's number (a vector factor's: its bits, read again by Number).
        public void Numbers(ReadOnlySpan<byte> unit, Span<double> into)
        {
            for (int j = 0; j < Chunks; j++)
            {
                ulong bits = s.Bits(unit[chunkBase[j]..], chunkIndex[j]);
                into[j] = s.Type.Width == 1 ? s.Type.Value(bits, 0) + s.Bias : BitConverter.UInt64BitsToDouble(bits);
            }
        }

        // The number value v of the group takes (BlockScale.Value's: the type's number of its bits, plus the bias).
        public double Number(ReadOnlySpan<byte> unit, double[] chunk, int first, int v)
        {
            int j = v / s.Values;
            return s.Type.Width == 1 ? chunk[first + j] : s.Type.Value(BitConverter.DoubleToUInt64Bits(chunk[first + j]), v % s.Values % s.Type.Width) + s.Bias;
        }
    }
}

/// <summary>
/// The default writing of any described type (<see cref="IElementType.Write(ReadOnlySpan{float}, Span{byte})"/>): whole
/// groups by the described encoder, a last group of fewer values filled with zeros first; float64 values of a per-element
/// type rounded from the float64 value itself.
/// </summary>
internal static class DescribedWriting
{
    public static void Write(TypeDescription d, ReadOnlySpan<float> values, Span<byte> stored)
    {
        var encoder = Encoder(d, values.Length, stored.Length);
        int n = d.UnitValues, size = d.UnitBytes, whole = values.Length / n * n;
        encoder.Encode(values[..whole], stored[..(whole / n * size)]);
        if (whole < values.Length)
        {
            Span<float> last = n <= 256 ? stackalloc float[n] : new float[n];
            last.Clear();
            values[whole..].CopyTo(last);
            encoder.Encode(last, stored.Slice(whole / n * size, size));
        }
    }

    public static void Write(TypeDescription d, ReadOnlySpan<double> values, Span<byte> stored)
    {
        var encoder = Encoder(d, values.Length, stored.Length);
        if (encoder.EncodesDoubles)
        {
            encoder.Encode(values, stored);
            return;
        }

        float[] floats = ArrayPool<float>.Shared.Rent(Math.Max(1, values.Length));
        try
        {
            for (int i = 0; i < values.Length; i++)
            {
                floats[i] = (float)values[i];
            }

            Write(d, floats.AsSpan(0, values.Length), stored);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(floats);
        }
    }

    private static DescribedEncoder Encoder(TypeDescription d, int count, int bytes)
    {
        ArgumentNullException.ThrowIfNull(d);
        var encoder = DescribedEncoder.Of(d) ?? throw new NotSupportedException(DescribedEncoder.Why(d));
        long need = (count + (long)d.UnitValues - 1) / d.UnitValues * d.UnitBytes;
        return bytes >= need ? encoder : throw new ArgumentException($"{count} values take {need} bytes; {bytes} are given.", nameof(bytes));
    }
}

/// <summary>
/// The rounding of a value to a number type of any width up to 64 bits, by arithmetic (no table of its codes): an integer
/// to the nearest, a tie to the even number, past its range its largest or smallest; a float to the nearest, a tie to the
/// even code, a zero keeping its sign, past its largest by half the next step or more (a tie to the even code) its
/// infinity, its NaN or its largest as its specials say, a NaN its quiet NaN of that sign. The rounding the described
/// encoder's table of a narrower number type gives (DescribedEncoder's grid), for any width.
/// </summary>
internal sealed class WideNumber
{
    private readonly NumberType t;
    private readonly ulong signBit, magnitudeOnes, largestCode, exponentField;
    private readonly double largest, overflow;
    private readonly bool binary32, binary64;

    private WideNumber(NumberType t)
    {
        this.t = t;
        signBit = t.Signed ? 1UL << (t.Bits - 1) : 0;
        magnitudeOnes = (t.Signed ? 1UL << (t.Bits - 1) : t.Bits == 64 ? 0 : 1UL << t.Bits) - 1;
        if (t.Kind == NumberKind.Float)
        {
            exponentField = ((1UL << t.Exponent) - 1) << t.Mantissa;
            largestCode = t.Specials switch
            {
                FloatSpecials.Ieee => exponentField - 1,
                FloatSpecials.AllOnesNaN => magnitudeOnes - 1,
                _ => magnitudeOnes,
            };
            largest = t.Value(largestCode);
            overflow = largest > 0 ? largest + Math.ScaleB(1, Math.ILogB(largest) - t.Mantissa) / 2 : 0;
            binary32 = t is { Bits: 32, Exponent: 8, Mantissa: 23, ExponentBias: 127, Signed: true, Subnormals: true, Specials: FloatSpecials.Ieee };
            binary64 = t is { Bits: 64, Exponent: 11, Mantissa: 52, ExponentBias: 1023, Signed: true, Subnormals: true, Specials: FloatSpecials.Ieee };
        }
    }

    /// <summary>The rounding to <paramref name="t"/>, or null where it is a table or a codebook.</summary>
    public static WideNumber? Of(NumberType t) => t.Valid && t.Width == 1 && t.Kind is NumberKind.Integer or NumberKind.Float ? new WideNumber(t) : null;

    /// <summary>The code of the number nearest <paramref name="x"/>.</summary>
    public ulong Round(double x) => t.Kind == NumberKind.Integer ? Integer(x) : Float(x);

    private ulong Integer(double x)
    {
        if (double.IsNaN(x))
        {
            return 0;
        }

        double r = Math.Round(x, MidpointRounding.ToEven);
        if (t.Signed)
        {
            double top = Math.ScaleB(1, t.Bits - 1);
            long value = r >= top ? (long)(magnitudeOnes) : r <= -top ? -(long)(magnitudeOnes) - 1 : (long)r;
            return t.Bits == 64 ? (ulong)value : (ulong)value & ((1UL << t.Bits) - 1);
        }

        return r <= 0 ? 0 : r >= Math.ScaleB(1, t.Bits) ? magnitudeOnes : (ulong)r;
    }

    private ulong Float(double x)
    {
        if (binary32)
        {
            return double.IsNaN(x) ? (double.IsNegative(x) ? 0xFFC00000UL : 0x7FC00000UL) : BitConverter.SingleToUInt32Bits((float)x);
        }

        if (binary64)
        {
            return double.IsNaN(x) ? (double.IsNegative(x) ? 0xFFF8000000000000UL : 0x7FF8000000000000UL) : BitConverter.DoubleToUInt64Bits(x);
        }

        int m = t.Mantissa;
        ulong sign = t.Signed && double.IsNegative(x) ? signBit : 0;
        if (double.IsNaN(x))
        {
            return Special(sign, nan: true) ?? Zero(0);
        }

        if (!t.Signed && (x < 0 || x == 0))
        {
            return 0;                                                       // the smallest: zero, or the least normal
        }

        double a = Math.Abs(x);
        if (largest > 0 && a >= overflow && (t.Signed || x > 0) && (a > overflow || (largestCode & 1) != 0))
        {
            return Special(sign, nan: false) ?? sign | largestCode;
        }

        if (a == 0)
        {
            return Zero(sign);
        }

        int e = Math.ILogB(a), least = t.Subnormals ? 1 - t.ExponentBias : -t.ExponentBias;
        ulong code;
        if (e < least)
        {
            code = t.Subnormals ? (ulong)Math.Round(Math.ScaleB(a, m - least), MidpointRounding.ToEven) : 0;
        }
        else
        {
            double s = Math.ScaleB(a, m - e);
            double r = m == 0 && s == 1.5 ? ((e + t.ExponentBias & 1) == 0 ? 1 : 2) : Math.Round(s, MidpointRounding.ToEven);
            if (r >= Math.ScaleB(1, m + 1))
            {
                e++;
                r = Math.ScaleB(1, m);
            }

            code = (ulong)(e + t.ExponentBias) << m | ((ulong)r - (1UL << m));
        }

        return sign | Math.Min(code, largestCode);
    }

    // A zero of a float with subnormals (its sign kept); without them the nearest to zero, as the grid ties it (the code of
    // the lower rank: the negative least where signed).
    private ulong Zero(ulong sign) => t.Subnormals ? sign : signBit;

    private ulong? Special(ulong sign, bool nan) => t.Specials switch
    {
        FloatSpecials.Ieee => sign | exponentField | (nan && t.Mantissa > 0 ? 1UL << (t.Mantissa - 1) : 0),
        FloatSpecials.AllOnesNaN => sign | magnitudeOnes,
        _ => null,
    };
}
