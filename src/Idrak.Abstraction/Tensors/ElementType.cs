// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Numerics;
using System.Runtime.InteropServices;
using Idrak.Abstraction.Formats;

namespace Idrak.Abstraction;

/// <summary>Widens stored values to float32: <paramref name="source"/> holds the bytes of <paramref name="destination"/>.Length values (whole blocks for a block format).</summary>
/// <param name="source">The stored bytes.</param>
/// <param name="destination">Where the values go.</param>
public delegate void ElementDecoder(ReadOnlySpan<byte> source, Span<float> destination);

/// <summary>Stores float32 values in a type: <paramref name="destination"/> receives the bytes of <paramref name="source"/>.Length values (whole blocks for a block format).</summary>
/// <param name="source">The values.</param>
/// <param name="destination">Where the stored bytes go.</param>
public delegate void ElementEncoder(ReadOnlySpan<float> source, Span<byte> destination);

/// <summary>
/// The scales of a block format (plan 37, T0): each block's <paramref name="PerBlock"/> scales of type
/// <paramref name="Type"/> (float16, float32, a 6-bit code, an 8-bit exponent), whether it has zero points (a minimum or
/// an offset per block or sub-block), and how they sit in the block, for people.
/// </summary>
/// <param name="Type">The scales' own type.</param>
/// <param name="PerBlock">Scales per block.</param>
/// <param name="ZeroPoints">Whether a block also has zero points (minimums).</param>
/// <param name="Layout">How the scales sit in a block, or null.</param>
public sealed record BlockScales(string Type, int PerBlock, bool ZeroPoints, string? Layout = null);

/// <summary>
/// An element type (plan 37, T0): how a tensor's values are stored, as data a registry holds (<see cref="ElementTypes"/>),
/// never a C# type parameter. A per-element type (float32, bfloat16, int32, int4, ...) has a block of one value; a block
/// format (ggml's Q4_K, IQ2_XXS, MXFP4, a block-scaled FP8) stores <see cref="BlockSize"/> values in
/// <see cref="BlockBits"/> bits with its <see cref="Scales"/>, so its bits per value may be fractional. Each entry carries
/// its conversions to and from float32 (<see cref="ToFloat32"/>, <see cref="FromFloat32"/>; on the CPU) and names the
/// operations that read it natively (<see cref="Kernels"/>); any other operation reads it through a conversion the
/// dispatcher makes and counts (<see cref="ElementTypes.Conversions"/>). A new format is registered, never coded into
/// <see cref="Tensor"/>.
/// </summary>
public sealed class ElementType : IKind
{
    /// <summary>32-bit IEEE float: what every operation computes in.</summary>
    public static readonly ElementType Float32 = new()
    {
        Name = "float32", Summary = "32-bit IEEE float, what every operation computes in", BlockBits = 32, IsFloat = true, IsSigned = true, Kernels = ["*"],
        ToFloat32 = ElementConversions.Float32ToFloat32, FromFloat32 = ElementConversions.Float32FromFloat32,
    };

    /// <summary>bfloat16: float32's high 16 bits (8 exponent bits, 7 mantissa bits).</summary>
    public static readonly ElementType BFloat16 = new()
    {
        Name = "bfloat16", Summary = "bfloat16: float32's high 16 bits (rounded to nearest, ties to even)", BlockBits = 16, IsFloat = true, IsSigned = true,
        Kernels = ["BFloat16MatMul", "GatherBFloat16", "GatherBFloat16Columns", "PackBFloat16"],
        ToFloat32 = ElementConversions.BFloat16ToFloat32, FromFloat32 = ElementConversions.BFloat16FromFloat32,
    };

    /// <summary>IEEE half (5 exponent bits, 10 mantissa bits).</summary>
    public static readonly ElementType Float16 = new()
    {
        Name = "float16", Summary = "16-bit IEEE half (rounded to nearest, ties to even)", BlockBits = 16, IsFloat = true, IsSigned = true, Kernels = [],
        ToFloat32 = ElementConversions.Float16ToFloat32, FromFloat32 = ElementConversions.Float16FromFloat32,
    };

    /// <summary>fp8 E4M3 (the "fn" form: no infinity, ±448 at most, one NaN code a sign).</summary>
    public static readonly ElementType Float8E4M3 = new()
    {
        Name = "float8_e4m3", Summary = "8-bit float E4M3fn: 4 exponent bits, 3 mantissa bits, no infinity, ±448 at most (an overflow gives NaN)", BlockBits = 8,
        IsFloat = true, IsSigned = true, Kernels = [],
        ToFloat32 = ElementConversions.E4M3ToFloat32, FromFloat32 = ElementConversions.E4M3FromFloat32,
    };

    /// <summary>fp8 E5M2 (a half's high byte: infinities and NaNs as a half's).</summary>
    public static readonly ElementType Float8E5M2 = new()
    {
        Name = "float8_e5m2", Summary = "8-bit float E5M2: a half's high byte, 5 exponent bits, 2 mantissa bits, with infinities", BlockBits = 8,
        IsFloat = true, IsSigned = true, Kernels = [],
        ToFloat32 = ElementConversions.E5M2ToFloat32, FromFloat32 = ElementConversions.E5M2FromFloat32,
    };

    /// <summary>64-bit signed integer (indices past 2³¹, read through int32 kernels by a counted conversion).</summary>
    public static readonly ElementType Int64 = new()
    {
        Name = "int64", Summary = "64-bit signed integer (indices; read by the index kernels through an int32 conversion)", BlockBits = 64, IsSigned = true, Kernels = [],
        ToFloat32 = ElementConversions.Int64ToFloat32, FromFloat32 = ElementConversions.Int64FromFloat32,
    };

    /// <summary>32-bit signed integer: indices, token ids and offsets, exact past 2²⁴.</summary>
    public static readonly ElementType Int32 = new()
    {
        Name = "int32", Summary = "32-bit signed integer: indices, token ids and offsets, exact past 2^24", BlockBits = 32, IsSigned = true,
        Kernels = ["Gather", "GatherBFloat16", "GatherBFloat16Columns", "OneHot", "ScatterAdd", "LogSoftmaxTopK", "SampleRows", "PenalizeRows", "HistoryPush",
            "RouteTopK", "ExpertGatedPair", "ExpertMatMul", "ConvertElements"],
        ToFloat32 = ElementConversions.Int32ToFloat32, FromFloat32 = ElementConversions.Int32FromFloat32,
    };

    /// <summary>8-bit signed integer.</summary>
    public static readonly ElementType Int8 = new()
    {
        Name = "int8", Summary = "8-bit signed integer", BlockBits = 8, IsSigned = true, Kernels = [],
        ToFloat32 = ElementConversions.Int8ToFloat32, FromFloat32 = ElementConversions.Int8FromFloat32,
    };

    /// <summary>8-bit unsigned integer.</summary>
    public static readonly ElementType UInt8 = new()
    {
        Name = "uint8", Summary = "8-bit unsigned integer", BlockBits = 8, Kernels = [],
        ToFloat32 = ElementConversions.UInt8ToFloat32, FromFloat32 = ElementConversions.UInt8FromFloat32,
    };

    /// <summary>4-bit signed integer, two a byte (the first in the low nibble).</summary>
    public static readonly ElementType Int4 = new()
    {
        Name = "int4", Summary = "4-bit signed integer in [-8, 7], two a byte, the first value in the low nibble", BlockBits = 4, IsSigned = true, Kernels = [],
        ToFloat32 = ElementConversions.Int4ToFloat32, FromFloat32 = ElementConversions.Int4FromFloat32,
    };

    /// <summary>2-bit signed integer, four a byte (the first in the low bits).</summary>
    public static readonly ElementType Int2 = new()
    {
        Name = "int2", Summary = "2-bit signed integer in [-2, 1], four a byte, the first value in the lowest bits", BlockBits = 2, IsSigned = true, Kernels = [],
        ToFloat32 = ElementConversions.Int2ToFloat32, FromFloat32 = ElementConversions.Int2FromFloat32,
    };

    /// <summary>A truth value, one byte each (0 or 1, as PyTorch's bool).</summary>
    public static readonly ElementType Bool = new()
    {
        Name = "bool", Summary = "a truth value, one byte each (0 false, 1 true; any value not 0 reads true)", BlockBits = 8, Kernels = [],
        ToFloat32 = ElementConversions.BoolToFloat32, FromFloat32 = ElementConversions.BoolFromFloat32,
    };

    /// <inheritdoc />
    public required string Name { get; init; }

    /// <inheritdoc />
    public required string Summary { get; init; }

    /// <summary>Values a block holds: 1 for a per-element type.</summary>
    public int BlockSize
    {
        get;
        init => field = value >= 1 ? value : throw new ArgumentOutOfRangeException(nameof(BlockSize), value, "A block holds at least one value.");
    } = 1;

    /// <summary>The bits a block takes, its scales and zero points included.</summary>
    public required int BlockBits
    {
        get;
        init => field = value >= 1 ? value : throw new ArgumentOutOfRangeException(nameof(BlockBits), value, "A block takes at least one bit.");
    }

    /// <summary>The bits a value takes on average: <see cref="BlockBits"/> / <see cref="BlockSize"/> (fractional for most block formats).</summary>
    public double BitsPerValue => (double)BlockBits / BlockSize;

    /// <summary>Whether values are floating point (false for integers, truth values and integer codes with scales).</summary>
    public bool IsFloat { get; init; }

    /// <summary>Whether values may be negative.</summary>
    public bool IsSigned { get; init; }

    /// <summary>Whether it is a block format: more than one value a block, or scales.</summary>
    public bool IsBlock => BlockSize > 1 || Scales is not null;

    /// <summary>A block format's scales and zero points, or null (a per-element type, or a format that does not describe them).</summary>
    public BlockScales? Scales { get; init; }

    /// <summary>
    /// The operations that read or write the type natively, by name (<c>Operation.Name</c>; "*" for every one): any other
    /// reads it through a counted conversion.
    /// </summary>
    public IReadOnlyList<string> Kernels { get; init; } = [];

    /// <summary>Its widening to float32 on the CPU, or null when it has none.</summary>
    public ElementDecoder? ToFloat32 { get; init; }

    /// <summary>Its storing from float32 on the CPU, or null when it has none (a format read only, such as most of ggml's).</summary>
    public ElementEncoder? FromFloat32 { get; init; }

    /// <summary>The bytes <paramref name="values"/> values take: whole blocks, a last partial byte rounded up.</summary>
    /// <exception cref="ArgumentException">A block format given a count that is not whole blocks.</exception>
    public long ByteCount(long values)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(values);
        if (BlockSize > 1 && values % BlockSize != 0)
        {
            throw new ArgumentException($"{Name} stores whole blocks of {BlockSize} values; {values} is not.", nameof(values));
        }

        return checked((values / BlockSize * BlockBits + 7) / 8);
    }

    /// <summary>The 4-byte words a storage of <paramref name="values"/> values of this type takes (what <c>Backend.Allocate</c> gives).</summary>
    public long StorageWords(long values) => (ByteCount(values) + 3) / 4;

    /// <summary>Widens <paramref name="destination"/>.Length stored values from <paramref name="source"/> to float32.</summary>
    /// <exception cref="NotSupportedException">The type has no conversion to float32.</exception>
    public void Decode(ReadOnlySpan<byte> source, Span<float> destination)
    {
        var decode = ToFloat32 ?? throw new NotSupportedException($"The element type {Name} has no conversion to float32.");
        if (source.Length < ByteCount(destination.Length))
        {
            throw new ArgumentException($"{destination.Length} {Name} values take {ByteCount(destination.Length)} bytes; {source.Length} given.", nameof(source));
        }

        decode(source, destination);
    }

    /// <summary>Stores <paramref name="source"/> in this type into <paramref name="destination"/>.</summary>
    /// <exception cref="NotSupportedException">The type has no conversion from float32.</exception>
    public void Encode(ReadOnlySpan<float> source, Span<byte> destination)
    {
        var encode = FromFloat32 ?? throw new NotSupportedException($"The element type {Name} has no conversion from float32.");
        if (destination.Length < ByteCount(source.Length))
        {
            throw new ArgumentException($"{source.Length} {Name} values take {ByteCount(source.Length)} bytes; {destination.Length} given.", nameof(destination));
        }

        encode(source, destination);
    }

    /// <summary>"int32 (32 bits a value)", "q4_k (4.5 bits a value, blocks of 256)".</summary>
    public string Describe() => BlockSize > 1
        ? string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{Name} ({BitsPerValue:0.###} bits a value, blocks of {BlockSize})")
        : string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{Name} ({BitsPerValue:0.###} bits a value)");

    /// <inheritdoc />
    public override string ToString() => Name;
}

/// <summary>Which element type a run uses and how it was chosen (<see cref="ElementTypes.Choose"/>).</summary>
/// <param name="Type">The type.</param>
/// <param name="Picked">Whether "auto" picked it (else the user selected it).</param>
/// <param name="Source">Where the choice came from when the user made it ("--dtype", "IDRAK_DEFAULT_DTYPE", an option's name), else null.</param>
public sealed record ElementTypeChoice(ElementType Type, bool Picked, string? Source)
{
    /// <summary>"picked by auto", or "selected by the user (IDRAK_DEFAULT_DTYPE)".</summary>
    public string How => Picked ? "picked by auto" : $"selected by the user ({Source})";

    /// <summary>"element type: float32 (picked by auto)": the line a load's report shows.</summary>
    public override string ToString() => $"element type: {Type.Name} ({How})";
}

/// <summary>
/// The element types, by name (ignoring case): the per-element types float32, bfloat16, float16, float8_e4m3,
/// float8_e5m2, int64, int32, int8, uint8, int4, int2 and bool, which Idrak.Abstraction registers, and the block formats
/// the packages register (Idrak adds ggml's, from its GGUF types). A name nobody registered is refused with "not
/// registered". The default type of a run comes from <see cref="EnvironmentVariable"/> (IDRAK_DEFAULT_DTYPE) or a run's
/// option (<c>--dtype</c>): <see cref="Auto"/> (the default) picks per run, and until plan 37's T2 it picks float32
/// everywhere, so every result stays as it was. The conversions the dispatcher makes for an operation whose kernel takes
/// another type are counted here (<see cref="Conversions"/>) and published to telemetry
/// (<see cref="Diagnostics.ElementsConverted"/>).
/// </summary>
public static class ElementTypes
{
    /// <summary>The variable naming the default element type: "IDRAK_DEFAULT_DTYPE".</summary>
    public const string EnvironmentVariable = "IDRAK_DEFAULT_DTYPE";

    /// <summary>The choice that picks the type per run (the default): float32 until plan 37's T2.</summary>
    public const string Auto = "auto";

    private static readonly KindTable<ElementType> Table = new(nameof(ElementTypes), "Element type", "add it with ElementTypes.Register",
        comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a type decides how stored bytes read: another type cannot stand in for it");

    private static long s_conversions, s_convertedElements;
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, long> s_byOperation = new(StringComparer.Ordinal);

    static ElementTypes()
    {
        foreach (var type in (ReadOnlySpan<ElementType>)[ElementType.Float32, ElementType.BFloat16, ElementType.Float16, ElementType.Float8E4M3, ElementType.Float8E5M2,
                     ElementType.Int64, ElementType.Int32, ElementType.Int8, ElementType.UInt8, ElementType.Int4, ElementType.Int2, ElementType.Bool])
        {
            Table.RegisterDefault(type);
        }

        LibraryDefaults.Ensure(typeof(ElementTypes));                     // the packages' block formats (Idrak: ggml's)
    }

    /// <summary>Registers <paramref name="type"/> under its name (over the library's of that name, which stays behind it).</summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ElementType type)
    {
        ArgumentNullException.ThrowIfNull(type);
        ArgumentException.ThrowIfNullOrWhiteSpace(type.Name);
        Table.Register(type, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's type <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyList<string> Names => Table.Names;

    /// <summary>The registered types, in the order of <see cref="Names"/>.</summary>
    public static IReadOnlyList<ElementType> All => Table.Kinds;

    /// <summary>The type registered as <paramref name="name"/>, or null.</summary>
    public static ElementType? Find(ReadOnlySpan<char> name) => Table.Find(name);

    /// <summary>The type registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotRegisteredException">None is: the message names it and the registered ones.</exception>
    public static ElementType Get(ReadOnlySpan<char> name) => Table.Get(name);

    /// <summary>The library's type <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static ElementType? Default(string name) => Table.Default(name);

    /// <summary>Who registered the type <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered type as "name: summary", one per line.</summary>
    public static string Describe() => Table.Describe();

    /// <summary>The values <see cref="Choose"/> accepts until plan 37's T2: auto and float32.</summary>
    public static IReadOnlyList<string> Accepted { get; } = [Auto, ElementType.Float32.Name];

    /// <summary>
    /// The default type of a run: <paramref name="asked"/> (a run's option, named <paramref name="source"/> in the choice
    /// and in a refusal) when given, else <see cref="EnvironmentVariable"/>, else <see cref="Auto"/>. Auto picks float32
    /// until plan 37's T2, so every result stays as it was.
    /// </summary>
    /// <exception cref="ArgumentException">A value other than those <see cref="Accepted"/> lists (the message names them).</exception>
    public static ElementTypeChoice Choose(string? asked = null, string source = "--dtype")
    {
        string? value = asked, from = source;
        if (string.IsNullOrWhiteSpace(value))
        {
            value = Environment.GetEnvironmentVariable(EnvironmentVariable);
            from = EnvironmentVariable;
        }

        if (string.IsNullOrWhiteSpace(value) || value.Trim().Equals(Auto, StringComparison.OrdinalIgnoreCase))
        {
            return new ElementTypeChoice(ElementType.Float32, Picked: true, Source: null);
        }

        if (value.Trim().Equals(ElementType.Float32.Name, StringComparison.OrdinalIgnoreCase))
        {
            return new ElementTypeChoice(ElementType.Float32, Picked: false, Source: from);
        }

        throw new ArgumentException($"{from} '{value}': the default element type is one of {string.Join(", ", Accepted)} (auto picks per run; the other types come with plan 37's T2).");
    }

    /// <summary>
    /// Publishes <paramref name="choice"/> for a run on <paramref name="device"/> to telemetry
    /// (<see cref="Diagnostics.ElementTypeChosen"/>, at <see cref="Diagnostics.TelemetryLevel.Devices"/>): a load says which type
    /// it computes in and how it was chosen.
    /// </summary>
    public static void Report(ElementTypeChoice choice, Device device)
    {
        ArgumentNullException.ThrowIfNull(choice);
        ArgumentNullException.ThrowIfNull(device);
        if (Diagnostics.Telemetry.IsEnabled(Diagnostics.TelemetryLevel.Devices))
        {
            Diagnostics.Telemetry.ElementTypeChosen(new Diagnostics.ElementTypeChosen(device.ToString(), choice.Type.Name, choice.How));
        }
    }

    /// <summary>Conversions the dispatcher made so far for an operation whose kernel takes another type (<see cref="ConvertedElements"/> elements).</summary>
    public static long Conversions => Interlocked.Read(ref s_conversions);

    /// <summary>The elements those conversions moved.</summary>
    public static long ConvertedElements => Interlocked.Read(ref s_convertedElements);

    /// <summary>The conversions so far by operation ("Gather", "RouteTopK", ...).</summary>
    public static IReadOnlyDictionary<string, long> ConversionsByOperation => new Dictionary<string, long>(s_byOperation, StringComparer.Ordinal);

    // Counts one conversion for `operation` and publishes it while someone listens at TelemetryLevel.Devices.
    internal static void Converted(string device, string operation, ElementType from, ElementType to, long elements)
    {
        Interlocked.Increment(ref s_conversions);
        Interlocked.Add(ref s_convertedElements, elements);
        s_byOperation.AddOrUpdate(operation, 1, static (_, n) => n + 1);
        if (Diagnostics.Telemetry.IsEnabled(Diagnostics.TelemetryLevel.Devices))
        {
            Diagnostics.Telemetry.ElementsConverted(new Diagnostics.ElementsConverted(device, operation, from.Name, to.Name, elements));
        }
    }
}

// The built-in types' conversions on the CPU: whole vectors where the type is a plain number (a scalar tail for the rest),
// value by value where it is packed or coded.
internal static class ElementConversions
{
    public static void Float32ToFloat32(ReadOnlySpan<byte> source, Span<float> destination) =>
        MemoryMarshal.Cast<byte, float>(source)[..destination.Length].CopyTo(destination);

    public static void Float32FromFloat32(ReadOnlySpan<float> source, Span<byte> destination) =>
        MemoryMarshal.AsBytes(source).CopyTo(destination);

    public static void BFloat16ToFloat32(ReadOnlySpan<byte> source, Span<float> destination) =>
        HalfFloats.WidenBFloat16(MemoryMarshal.Cast<byte, ushort>(source)[..destination.Length], destination);

    public static void BFloat16FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        var halves = MemoryMarshal.Cast<byte, ushort>(destination);
        for (int i = 0; i < source.Length; i++)
        {
            halves[i] = Devices.BFloat16Bits.Round(source[i]);
        }
    }

    public static void Float16ToFloat32(ReadOnlySpan<byte> source, Span<float> destination) =>
        HalfFloats.WidenFloat16(MemoryMarshal.Cast<byte, ushort>(source)[..destination.Length], destination);

    public static void Float16FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        var halves = MemoryMarshal.Cast<byte, ushort>(destination);
        for (int i = 0; i < source.Length; i++)
        {
            halves[i] = BitConverter.HalfToUInt16Bits((Half)source[i]);   // round to nearest, ties to even
        }
    }

    // Every fp8 code's value (a NaN code as the quiet NaN of its sign), as PyTorch and ml_dtypes widen them.
    private static readonly float[] E4M3 = Float8Table(e4m3: true), E5M2 = Float8Table(e4m3: false);

    private static float[] Float8Table(bool e4m3)
    {
        var table = new float[256];
        for (int code = 0; code < 256; code++)
        {
            uint sign = (uint)(code >> 7) << 31;
            float value;
            if (e4m3)
            {
                int e = (code >> 3) & 15, m = code & 7;
                value = e == 15 && m == 7 ? float.NaN
                    : e == 0 ? m * (1f / 512f)
                    : BitConverter.UInt32BitsToSingle((uint)(e - 7 + 127) << 23 | (uint)m << 20);
            }
            else
            {
                value = (float)BitConverter.UInt16BitsToHalf((ushort)((code & 0x7F) << 8));
            }

            table[code] = float.IsNaN(value) ? BitConverter.UInt32BitsToSingle(sign | 0x7FC00000u)
                : BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(value) | sign);
        }

        return table;
    }

    public static void E4M3ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = E4M3[source[i]];
        }
    }

    public static void E5M2ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = E5M2[source[i]];
        }
    }

    // E4M3fn: codes 0..126 finite (0x7E = 448), 0x7F NaN; past 448 + half a step (464, ties to the even 448) is NaN.
    public static void E4M3FromFloat32(ReadOnlySpan<float> source, Span<byte> destination) => Float8From(source, destination, E4M3, 0x7E, 0x7F);

    // E5M2: codes 0..123 finite (0x7B = 57344), 0x7C infinity; past 57344 + half a step (61440, ties to the even 0x7C) is infinity.
    public static void E5M2FromFloat32(ReadOnlySpan<float> source, Span<byte> destination) => Float8From(source, destination, E5M2, 0x7B, 0x7C);

    // Round to the nearest code (ties to the even code), the codes' values ascending from 0 to `max`; past the largest the
    // code `overflow` (NaN or infinity), NaN to the NaN code (0x7F for both).
    private static void Float8From(ReadOnlySpan<float> source, Span<byte> destination, float[] table, int max, int overflow)
    {
        float top = table[max], next = max == 0x7E ? 480f : 65536f;
        for (int i = 0; i < source.Length; i++)
        {
            float x = source[i];
            byte sign = (byte)(BitConverter.SingleToUInt32Bits(x) >> 24 & 0x80);
            float a = MathF.Abs(x);
            int code;
            if (float.IsNaN(x))
            {
                code = 0x7F;
            }
            else if (a > top)
            {
                float mid = (top + next) / 2f;
                code = a < mid || a == mid && (max & 1) == 0 ? max : overflow;
            }
            else
            {
                int lo = 0, hi = max;                                       // the largest code whose value is at most a
                while (lo < hi)
                {
                    int m = (lo + hi + 1) >> 1;
                    if (table[m] <= a)
                    {
                        lo = m;
                    }
                    else
                    {
                        hi = m - 1;
                    }
                }

                code = lo;
                if (lo < max)
                {
                    float below = a - table[lo], above = table[lo + 1] - a;
                    if (above < below || above == below && (lo & 1) == 1)
                    {
                        code = lo + 1;
                    }
                }
            }

            destination[i] = (byte)(code | sign);
        }
    }

    public static void Int64ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        var values = MemoryMarshal.Cast<byte, long>(source);
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = values[i];
        }
    }

    public static void Int64FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        var values = MemoryMarshal.Cast<byte, long>(destination);
        for (int i = 0; i < source.Length; i++)
        {
            values[i] = float.IsNaN(source[i]) ? 0 : (long)Math.Clamp(MathF.Round(source[i]), long.MinValue, long.MaxValue);
        }
    }

    public static void Int32ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        var values = MemoryMarshal.Cast<byte, int>(source)[..destination.Length];
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            for (; i <= values.Length - Vector<int>.Count; i += Vector<int>.Count)
            {
                Vector.ConvertToSingle(new Vector<int>(values[i..])).CopyTo(destination[i..]);
            }
        }

        for (; i < values.Length; i++)
        {
            destination[i] = values[i];
        }
    }

    // Rounded to nearest (ties to even) and saturated to the type's range; NaN gives 0.
    public static void Int32FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        var values = MemoryMarshal.Cast<byte, int>(destination);
        for (int i = 0; i < source.Length; i++)
        {
            float x = MathF.Round(source[i]);
            values[i] = float.IsNaN(x) ? 0 : x >= 2147483648f ? int.MaxValue : x <= -2147483648f ? int.MinValue : (int)x;
        }
    }

    public static void Int8ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        var values = MemoryMarshal.Cast<byte, sbyte>(source);
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = values[i];
        }
    }

    public static void Int8FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        for (int i = 0; i < source.Length; i++)
        {
            destination[i] = (byte)(sbyte)Saturate(source[i], -128, 127);
        }
    }

    public static void UInt8ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = source[i];
        }
    }

    public static void UInt8FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        for (int i = 0; i < source.Length; i++)
        {
            destination[i] = (byte)Saturate(source[i], 0, 255);
        }
    }

    public static void Int4ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            int nibble = source[i >> 1] >> ((i & 1) * 4) & 0xF;
            destination[i] = (nibble ^ 8) - 8;                                  // sign-extend 4 bits
        }
    }

    public static void Int4FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        destination[..(int)((source.Length + 1L) / 2)].Clear();
        for (int i = 0; i < source.Length; i++)
        {
            destination[i >> 1] |= (byte)((Saturate(source[i], -8, 7) & 0xF) << ((i & 1) * 4));
        }
    }

    public static void Int2ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            int bits = source[i >> 2] >> ((i & 3) * 2) & 0x3;
            destination[i] = (bits ^ 2) - 2;                                    // sign-extend 2 bits
        }
    }

    public static void Int2FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        destination[..(int)((source.Length + 3L) / 4)].Clear();
        for (int i = 0; i < source.Length; i++)
        {
            destination[i >> 2] |= (byte)((Saturate(source[i], -2, 1) & 0x3) << ((i & 3) * 2));
        }
    }

    public static void BoolToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = source[i] != 0 ? 1f : 0f;
        }
    }

    public static void BoolFromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        for (int i = 0; i < source.Length; i++)
        {
            destination[i] = source[i] != 0f && !float.IsNaN(source[i]) ? (byte)1 : (byte)0;
        }
    }

    // Rounded to nearest (ties to even), clamped to [low, high]; NaN gives 0.
    private static int Saturate(float x, int low, int high) => float.IsNaN(x) ? 0 : (int)Math.Clamp(MathF.Round(x), low, high);

    /// <summary>
    /// Converts <paramref name="count"/> elements of <paramref name="from"/> in <paramref name="source"/> to
    /// <paramref name="to"/> in <paramref name="destination"/>. Integers to and from float32 and between int32 and int64
    /// are exact or refused (an index that is not a whole number, or past what the other type holds exactly: float32 holds
    /// integers up to 2²⁴); any other pair goes through float32 with the types' own conversions.
    /// </summary>
    /// <exception cref="ArgumentException">A value the destination type cannot hold exactly (integers), naming it.</exception>
    public static void Convert(ElementType from, ReadOnlySpan<byte> source, ElementType to, Span<byte> destination, int count)
    {
        if (ReferenceEquals(from, to))
        {
            int bytes = (int)from.ByteCount(count);
            source[..bytes].CopyTo(destination);
            return;
        }

        if (ReferenceEquals(to, ElementType.Int32) && ReferenceEquals(from, ElementType.Float32))
        {
            FloatsToInt32(MemoryMarshal.Cast<byte, float>(source)[..count], MemoryMarshal.Cast<byte, int>(destination)[..count]);
            return;
        }

        if (ReferenceEquals(to, ElementType.Float32) && ReferenceEquals(from, ElementType.Int32))
        {
            Int32ToFloats(MemoryMarshal.Cast<byte, int>(source)[..count], MemoryMarshal.Cast<byte, float>(destination)[..count]);
            return;
        }

        if (ReferenceEquals(to, ElementType.Int32) && ReferenceEquals(from, ElementType.Int64))
        {
            var longs = MemoryMarshal.Cast<byte, long>(source);
            var ints = MemoryMarshal.Cast<byte, int>(destination);
            for (int i = 0; i < count; i++)
            {
                long v = longs[i];
                ints[i] = v is >= int.MinValue and <= int.MaxValue ? (int)v : throw new ArgumentException($"Index {v} (element {i}) is past what int32 holds: the index kernels read int32.");
            }

            return;
        }

        if (ReferenceEquals(to, ElementType.Int64) && ReferenceEquals(from, ElementType.Int32))
        {
            var ints = MemoryMarshal.Cast<byte, int>(source);
            var longs = MemoryMarshal.Cast<byte, long>(destination);
            for (int i = 0; i < count; i++)
            {
                longs[i] = ints[i];
            }

            return;
        }

        if (ReferenceEquals(to, ElementType.Float32) && ReferenceEquals(from, ElementType.Int64))
        {
            var longs = MemoryMarshal.Cast<byte, long>(source);
            var floats = MemoryMarshal.Cast<byte, float>(destination);
            for (int i = 0; i < count; i++)
            {
                long v = longs[i];
                floats[i] = Math.Abs(v) <= 1L << 24 ? v : throw PastFloat(v, i);
            }

            return;
        }

        if (ReferenceEquals(to, ElementType.Int64) && ReferenceEquals(from, ElementType.Float32))
        {
            var floats = MemoryMarshal.Cast<byte, float>(source);
            var longs = MemoryMarshal.Cast<byte, long>(destination);
            for (int i = 0; i < count; i++)
            {
                float v = floats[i];
                longs[i] = MathF.Truncate(v) == v && MathF.Abs(v) < 9.2e18f ? (long)v : throw NotWhole(v, i);
            }

            return;
        }

        // Any other pair through float32, a chunk at a time (whole blocks).
        int chunk = Math.Max(from.BlockSize, to.BlockSize) * Math.Max(1, 65536 / Math.Max(from.BlockSize, to.BlockSize));
        float[] values = ArrayPool<float>.Shared.Rent(Math.Min(count, chunk));
        try
        {
            for (int at = 0; at < count; at += chunk)
            {
                int n = Math.Min(chunk, count - at);
                var part = values.AsSpan(0, n);
                from.Decode(source[(int)from.ByteCount(at)..], part);
                to.Encode(part, destination[(int)to.ByteCount(at)..]);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(values);
        }
    }

    // float32 indices to int32: whole numbers only (every float32 whole number in int32's range is exact).
    private static void FloatsToInt32(ReadOnlySpan<float> source, Span<int> destination)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            for (; i <= source.Length - Vector<float>.Count; i += Vector<float>.Count)
            {
                var v = new Vector<float>(source[i..]);
                var whole = Vector.ConvertToSingle(Vector.ConvertToInt32(v));
                if (!Vector.EqualsAll(whole, v))
                {
                    break;                                                  // a fraction, NaN or a value out of range: checked one by one
                }

                Vector.ConvertToInt32(v).CopyTo(destination[i..]);
            }
        }

        for (; i < source.Length; i++)
        {
            float v = source[i];
            destination[i] = MathF.Truncate(v) == v && v >= -2147483648f && v < 2147483648f ? (int)v : throw NotWhole(v, i);
        }
    }

    // int32 indices to float32: exact only up to 2²⁴.
    private static void Int32ToFloats(ReadOnlySpan<int> source, Span<float> destination)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            var limit = new Vector<int>(1 << 24);
            for (; i <= source.Length - Vector<int>.Count; i += Vector<int>.Count)
            {
                var v = new Vector<int>(source[i..]);
                if (Vector.GreaterThanAny(Vector.Abs(v), limit) || Vector.LessThanAny(v, -limit))
                {
                    break;                                                  // past 2^24 (or int.MinValue): checked one by one
                }

                Vector.ConvertToSingle(v).CopyTo(destination[i..]);
            }
        }

        for (; i < source.Length; i++)
        {
            int v = source[i];
            destination[i] = v is >= -(1 << 24) and <= 1 << 24 ? v : throw PastFloat(v, i);
        }
    }

    private static ArgumentException NotWhole(float value, int at) =>
        new($"Index {value.ToString(System.Globalization.CultureInfo.InvariantCulture)} (element {at}) is not a whole number an integer index type holds.");

    private static ArgumentException PastFloat(long value, int at) =>
        new($"Index {value} (element {at}) is past 2^24: a kernel that reads indices as float32 cannot read it exactly (that device's integer index kernel is not built yet; run it on a device whose index type is int32, such as the CPU).");
}
