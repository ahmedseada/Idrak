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
        Name = "int64", Summary = "64-bit signed integer (indices; read by the index kernels through an int32 conversion)", BlockBits = 64, IsSigned = true, IsInteger = true, Kernels = [],
        ToFloat32 = ElementConversions.Int64ToFloat32, FromFloat32 = ElementConversions.Int64FromFloat32,
    };

    /// <summary>32-bit signed integer: indices, token ids and offsets, exact past 2²⁴.</summary>
    public static readonly ElementType Int32 = new()
    {
        Name = "int32", Summary = "32-bit signed integer: indices, token ids and offsets, exact past 2^24", BlockBits = 32, IsSigned = true, IsInteger = true,
        Kernels = ["Gather", "GatherBFloat16", "GatherBFloat16Columns", "OneHot", "ScatterAdd", "LogSoftmaxTopK", "SampleRows", "PenalizeRows", "HistoryPush",
            "RouteTopK", "ExpertGatedPair", "ExpertMatMul", "ConvertElements"],
        ToFloat32 = ElementConversions.Int32ToFloat32, FromFloat32 = ElementConversions.Int32FromFloat32,
    };

    /// <summary>8-bit signed integer.</summary>
    public static readonly ElementType Int8 = new()
    {
        Name = "int8", Summary = "8-bit signed integer", BlockBits = 8, IsSigned = true, IsInteger = true, Kernels = [],
        ToFloat32 = ElementConversions.Int8ToFloat32, FromFloat32 = ElementConversions.Int8FromFloat32,
    };

    /// <summary>8-bit unsigned integer.</summary>
    public static readonly ElementType UInt8 = new()
    {
        Name = "uint8", Summary = "8-bit unsigned integer", BlockBits = 8, IsInteger = true, Kernels = [],
        ToFloat32 = ElementConversions.UInt8ToFloat32, FromFloat32 = ElementConversions.UInt8FromFloat32,
    };

    /// <summary>4-bit signed integer, two a byte (the first in the low nibble).</summary>
    public static readonly ElementType Int4 = new()
    {
        Name = "int4", Summary = "4-bit signed integer in [-8, 7], two a byte, the first value in the low nibble", BlockBits = 4, IsSigned = true, IsInteger = true, Kernels = [],
        ToFloat32 = ElementConversions.Int4ToFloat32, FromFloat32 = ElementConversions.Int4FromFloat32,
    };

    /// <summary>2-bit signed integer, four a byte (the first in the low bits).</summary>
    public static readonly ElementType Int2 = new()
    {
        Name = "int2", Summary = "2-bit signed integer in [-2, 1], four a byte, the first value in the lowest bits", BlockBits = 2, IsSigned = true, IsInteger = true, Kernels = [],
        ToFloat32 = ElementConversions.Int2ToFloat32, FromFloat32 = ElementConversions.Int2FromFloat32,
    };

    /// <summary>A truth value, one byte each (0 or 1, as PyTorch's bool).</summary>
    public static readonly ElementType Bool = new()
    {
        Name = "bool", Summary = "a truth value, one byte each (0 false, 1 true; any value not 0 reads true)", BlockBits = 8, Kernels = [],
        ToFloat32 = ElementConversions.BoolToFloat32, FromFloat32 = ElementConversions.BoolFromFloat32,
    };

    /// <summary>16-bit signed integer (plan 37, T2): audio's 16-bit samples kept as stored; storage and conversions only.</summary>
    public static readonly ElementType Int16 = new()
    {
        Name = "int16", Summary = "16-bit signed integer (audio's 16-bit samples kept as stored; storage and conversions only)", BlockBits = 16, IsSigned = true, IsInteger = true, Kernels = [],
        ToFloat32 = ElementConversions.Int16ToFloat32, FromFloat32 = ElementConversions.Int16FromFloat32,
    };

    /// <summary>16-bit unsigned integer (plan 37, T2): token ids of vocabularies under 65,536 in datasets; storage and conversions only.</summary>
    public static readonly ElementType UInt16 = new()
    {
        Name = "uint16", Summary = "16-bit unsigned integer (token ids of vocabularies under 65,536, half of int32's memory; storage and conversions only)", BlockBits = 16, IsInteger = true, Kernels = [],
        ToFloat32 = ElementConversions.UInt16ToFloat32, FromFloat32 = ElementConversions.UInt16FromFloat32,
    };

    /// <summary>64-bit IEEE float (plan 37, T2): values kept for checks against a reference; storage and conversions only.</summary>
    public static readonly ElementType Float64 = new()
    {
        Name = "float64", Summary = "64-bit IEEE float (values kept for checks against a reference; storage and conversions only, narrowed to float32 rounded to nearest)",
        BlockBits = 64, IsFloat = true, IsSigned = true, Kernels = [],
        ToFloat32 = ElementConversions.Float64ToFloat32, FromFloat32 = ElementConversions.Float64FromFloat32,
    };

    /// <summary>32-bit unsigned integer (plan 37, T3): ids and indices past int32's range in half of int64's memory.</summary>
    public static readonly ElementType UInt32 = new()
    {
        Name = "uint32", Summary = "32-bit unsigned integer (ids and indices up to 4,294,967,295)", BlockBits = 32, IsInteger = true, Kernels = [],
        ToFloat32 = ElementConversions.UInt32ToFloat32, FromFloat32 = ElementConversions.UInt32FromFloat32,
    };

    /// <summary>64-bit unsigned integer (plan 37, T3).</summary>
    public static readonly ElementType UInt64 = new()
    {
        Name = "uint64", Summary = "64-bit unsigned integer", BlockBits = 64, IsInteger = true, Kernels = [],
        ToFloat32 = ElementConversions.UInt64ToFloat32, FromFloat32 = ElementConversions.UInt64FromFloat32,
    };

    /// <summary>2-bit unsigned integer, four a byte (the first in the low bits) (plan 37, T3).</summary>
    public static readonly ElementType UInt2 = new()
    {
        Name = "uint2", Summary = "2-bit unsigned integer in [0, 3], four a byte, the first value in the lowest bits", BlockBits = 2, IsInteger = true, Kernels = [],
        ToFloat32 = ElementConversions.UInt2ToFloat32, FromFloat32 = ElementConversions.UInt2FromFloat32,
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

    /// <summary>
    /// Whether values are whole numbers of <see cref="BlockBits"/> bits, two's complement when <see cref="IsSigned"/> (int64
    /// down to int2, uint64 down to uint2): ids and indices may be stored in one (<see cref="ElementTypes.ChooseIds"/>).
    /// </summary>
    public bool IsInteger { get; init; }

    /// <summary>The smallest value an integer type holds (0 for an unsigned one).</summary>
    /// <exception cref="InvalidOperationException">Not an integer type.</exception>
    public Int128 MinValue => IsInteger ? IsSigned ? -((Int128)1 << (BlockBits - 1)) : 0 : throw NotInteger();

    /// <summary>The largest value an integer type holds.</summary>
    /// <exception cref="InvalidOperationException">Not an integer type.</exception>
    public Int128 MaxValue => IsInteger ? ((Int128)1 << (IsSigned ? BlockBits - 1 : BlockBits)) - 1 : throw NotInteger();

    /// <summary>Whether an integer type holds every value in [<paramref name="smallest"/>, <paramref name="largest"/>]; false for any other type.</summary>
    public bool Holds(long smallest, long largest) => IsInteger && smallest >= MinValue && largest <= MaxValue;

    private InvalidOperationException NotInteger() => new($"{Name} is not an integer type: it has no range of whole numbers.");

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

/// <summary>Which element type a run computes in on a device and how it was chosen (<see cref="ElementTypes.Choose"/>).</summary>
/// <param name="Type">The type.</param>
/// <param name="Picked">Whether "auto" picked it (else the user selected it).</param>
/// <param name="Source">Where the choice came from when the user made it ("--dtype", "IDRAK_DEFAULT_DTYPE", an option's name), else null.</param>
public sealed record ElementTypeChoice(ElementType Type, bool Picked, string? Source)
{
    /// <summary>The device the run computes on ("cuda:0"), or null when the choice names none.</summary>
    public string? Device { get; init; }

    /// <summary>
    /// What the device reported (plan 37, T2): the types it computes natively and why, "bfloat16, float8_e4m3, float32
    /// (compute capability 8.9: ...)"; null when the choice names no device.
    /// </summary>
    public string? Native { get; init; }

    /// <summary>Whether the device computes <see cref="Type"/> natively (else its operations compute in float32 over it).</summary>
    public bool IsNative { get; init; } = true;

    /// <summary>
    /// What the type is chosen for, as the report names it: "element type" (the values a run computes in,
    /// <see cref="ElementTypes.Choose"/>), "id type" (ids and indices, <see cref="ElementTypes.ChooseIds"/>) or "sum type
    /// for MatMul" (an operation's sums, <see cref="SumTypes.Choose"/>).
    /// </summary>
    public string Role { get; init; } = "element type";

    /// <summary>
    /// What auto measured or found that the pick follows from ("ids from 0 to 151,935 fit uint32", the sum types' times
    /// and errors), said in place of the device's report; null for a pick from what the device reports.
    /// </summary>
    public string? Measured { get; init; }

    /// <summary>
    /// "picked by auto: cuda:0 computes bfloat16, float32 natively (...)", or "selected by the user (IDRAK_DEFAULT_DTYPE)"
    /// (with "; cpu computes it in float32" where the device has no arithmetic of the type).
    /// </summary>
    public string How => Picked
        ? Measured is not null ? $"picked by auto: {Measured}" : Native is null ? "picked by auto" : $"picked by auto: {Device} computes {Native}"
        : IsNative ? $"selected by the user ({Source})" : $"selected by the user ({Source}); {Device} computes it in float32 (it computes {Native})";

    /// <summary>"element type: bfloat16 on cuda:0 (picked by auto: ...)": the line a load's report shows (its <see cref="Role"/> first).</summary>
    public override string ToString() => Device is null ? $"{Role}: {Type.Name} ({How})" : $"{Role}: {Type.Name} on {Device} ({How})";
}

/// <summary>
/// The element types, by name (ignoring case): the per-element types float32, bfloat16, float16, float8_e4m3,
/// float8_e5m2, int64, int32, int8, uint8, int4, int2, bool, int16, uint16, float64, uint32, uint64 and uint2, which Idrak.Abstraction
/// registers, and the block formats the packages register (Idrak adds ggml's, from its GGUF types). The short names
/// bf16, f16, fp16, half, f32, fp32, f64, fp64, double and fp8 find their types too. A name nobody registered is refused
/// with "not registered". The type a run computes in on a device comes from a run's option (<c>--dtype</c>), else
/// <see cref="EnvironmentVariable"/> (IDRAK_DEFAULT_DTYPE), else <see cref="Auto"/> (the default, plan 37 T2), which
/// picks what the device computes natively as its backend reports it (<see cref="Devices.BackendCapabilities.NativeTypes"/>,
/// measured at run time): bfloat16 where native, float16 where only it is, float32 otherwise; float32 selected keeps every
/// result as before T2. The conversions the dispatcher makes for an operation whose kernel takes another type are counted
/// here (<see cref="Conversions"/>) and published to telemetry (<see cref="Diagnostics.ElementsConverted"/>).
/// </summary>
public static class ElementTypes
{
    /// <summary>The variable naming the default element type: "IDRAK_DEFAULT_DTYPE".</summary>
    public const string EnvironmentVariable = "IDRAK_DEFAULT_DTYPE";

    /// <summary>The choice that picks the type per device (the default): what the device computes natively, best first.</summary>
    public const string Auto = "auto";

    private static readonly KindTable<ElementType> Table = new(nameof(ElementTypes), "Element type", "add it with ElementTypes.Register",
        comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a type decides how stored bytes read: another type cannot stand in for it");

    // The short names people write for the built-ins ("bf16"), resolved before the registry (rule 15: no string made).
    private static readonly Dictionary<string, string> ShortNames = new(StringComparer.OrdinalIgnoreCase)
    {
        ["f32"] = "float32", ["fp32"] = "float32", ["bf16"] = "bfloat16", ["f16"] = "float16", ["fp16"] = "float16", ["half"] = "float16",
        ["f64"] = "float64", ["fp64"] = "float64", ["double"] = "float64", ["fp8"] = "float8_e4m3", ["e4m3"] = "float8_e4m3", ["e5m2"] = "float8_e5m2",
    };

    private static readonly Dictionary<string, string>.AlternateLookup<ReadOnlySpan<char>> ShortLookup = ShortNames.GetAlternateLookup<ReadOnlySpan<char>>();

    private static long s_conversions, s_convertedElements;
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, long> s_byOperation = new(StringComparer.Ordinal);

    static ElementTypes()
    {
        foreach (var type in (ReadOnlySpan<ElementType>)[ElementType.Float32, ElementType.BFloat16, ElementType.Float16, ElementType.Float8E4M3, ElementType.Float8E5M2,
                     ElementType.Int64, ElementType.Int32, ElementType.Int8, ElementType.UInt8, ElementType.Int4, ElementType.Int2, ElementType.Bool,
                     ElementType.Int16, ElementType.UInt16, ElementType.Float64, ElementType.UInt32, ElementType.UInt64, ElementType.UInt2])
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

    /// <summary>The type registered as <paramref name="name"/> (or a built-in's short name: bf16, f16, fp32, ...), or null.</summary>
    public static ElementType? Find(ReadOnlySpan<char> name) =>
        Table.Find(name) ?? (ShortLookup.TryGetValue(name.Trim(), out var full) ? Table.Find(full) : null);

    /// <summary>The type registered as <paramref name="name"/> (or a built-in's short name: bf16, f16, fp32, ...).</summary>
    /// <exception cref="NotRegisteredException">None is: the message names it and the registered ones.</exception>
    public static ElementType Get(ReadOnlySpan<char> name) => Find(name) ?? Table.Get(name);

    /// <summary>The library's type <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static ElementType? Default(string name) => Table.Default(name);

    /// <summary>Who registered the type <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered type as "name: summary", one per line.</summary>
    public static string Describe() => Table.Describe();

    /// <summary>The types a run may compute in (plan 37, T2), best first as <see cref="Auto"/> prefers them: bfloat16, float16, float32.</summary>
    public static IReadOnlyList<ElementType> ComputeTypes { get; } = [ElementType.BFloat16, ElementType.Float16, ElementType.Float32];

    /// <summary>The values <see cref="Choose"/> accepts: auto and the <see cref="ComputeTypes"/> (with their short names, bf16, f16, fp32, ...).</summary>
    public static IReadOnlyList<string> Accepted { get; } = [Auto, .. ComputeTypes.Select(t => t.Name)];

    /// <summary>
    /// The type <see cref="Auto"/> picks on a device whose backend reports <paramref name="capabilities"/>: the first of
    /// <see cref="ComputeTypes"/> it computes natively (<see cref="Devices.BackendCapabilities.NativeTypes"/>), float32 when
    /// it computes none of the others.
    /// </summary>
    public static ElementType Pick(Devices.BackendCapabilities capabilities)
    {
        ArgumentNullException.ThrowIfNull(capabilities);
        foreach (var type in ComputeTypes)
        {
            if (IsNative(capabilities, type))
            {
                return type;
            }
        }

        return ElementType.Float32;
    }

    // Whether the device computes `type` natively (float32 always: every backend's kernels compute it).
    private static bool IsNative(Devices.BackendCapabilities capabilities, ElementType type)
    {
        if (ReferenceEquals(type, ElementType.Float32))
        {
            return true;
        }

        foreach (var native in capabilities.NativeTypes)
        {
            if (ReferenceEquals(native, type))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// The type a run computes in on <paramref name="device"/> (the CPU when null): <paramref name="asked"/> (a run's
    /// option, named <paramref name="source"/> in the choice and in a refusal) when given, else
    /// <see cref="EnvironmentVariable"/>, else <see cref="Auto"/>, which picks what the device computes natively
    /// (<see cref="Pick"/>, from what its backend measured at run time). The choice says how it was made and what the
    /// device reported; a type the device has no arithmetic for is stored in it and computed in float32 (said so).
    /// </summary>
    /// <exception cref="ArgumentException">A value other than those <see cref="Accepted"/> lists (the message names them).</exception>
    public static ElementTypeChoice Choose(string? asked = null, string source = "--dtype", Device? device = null)
    {
        device ??= Device.Cpu;
        var capabilities = device.Backend.Capabilities;
        string native = $"{string.Join(", ", capabilities.NativeTypes.Select(t => t.Name))} natively ({capabilities.NativeReport})";
        string? value = asked, from = source;
        if (string.IsNullOrWhiteSpace(value))
        {
            value = Environment.GetEnvironmentVariable(EnvironmentVariable);
            from = EnvironmentVariable;
        }

        if (string.IsNullOrWhiteSpace(value) || value.Trim().Equals(Auto, StringComparison.OrdinalIgnoreCase))
        {
            return new ElementTypeChoice(Pick(capabilities), Picked: true, Source: null) { Device = device.ToString(), Native = native };
        }

        var type = Find(value.AsSpan().Trim());
        if (type is null || !ComputeTypes.Contains(type))
        {
            throw new ArgumentException($"{from} '{value}': the element type a run computes in is one of {string.Join(", ", Accepted)} (bf16, f16 and fp32 for short; auto picks what the device computes natively).");
        }

        return new ElementTypeChoice(type, Picked: false, Source: from) { Device = device.ToString(), Native = native, IsNative = IsNative(capabilities, type) };
    }

    /// <summary>The variable naming the type ids and indices are stored in: "IDRAK_ID_DTYPE".</summary>
    public const string IdEnvironmentVariable = "IDRAK_ID_DTYPE";

    /// <summary>
    /// The types ids and indices may be stored in, as auto tries them: every registered integer type of whole bytes
    /// (<see cref="ElementType.IsInteger"/>: uint8, int8, uint16, int16, uint32, int32, uint64, int64 and any an app
    /// registers), fewest bits first, unsigned before signed.
    /// </summary>
    public static IReadOnlyList<ElementType> IdTypes =>
        [.. All.Where(t => t.IsInteger && t.BlockSize == 1 && t.BlockBits % 8 == 0).OrderBy(t => t.BlockBits).ThenBy(t => t.IsSigned)];

    /// <summary>
    /// The type ids or indices in [<paramref name="smallest"/>, <paramref name="largest"/>] are stored in on
    /// <paramref name="device"/> (the CPU when null), from the data: <paramref name="asked"/> (a run's option, named
    /// <paramref name="source"/>) when given, else <see cref="IdEnvironmentVariable"/>, else <see cref="Auto"/>, which picks
    /// the smallest of <see cref="IdTypes"/> that holds them (a vocabulary of 50,000 tokens: uint16; an expert index:
    /// uint8). A kernel that reads another index type (<see cref="Devices.BackendCapabilities.IndexType"/>) gets them
    /// through the dispatcher's counted conversion.
    /// </summary>
    /// <exception cref="ArgumentException">A value that is no integer type, or a type that does not hold the data (the message names both).</exception>
    public static ElementTypeChoice ChooseIds(long largest, long smallest = 0, string? asked = null, string source = "--id-dtype", Device? device = null)
    {
        if (smallest > largest)
        {
            throw new ArgumentException($"Ids from {smallest} to {largest}: the smallest is past the largest.", nameof(smallest));
        }

        device ??= Device.Cpu;
        var capabilities = device.Backend.Capabilities;
        string range = string.Create(System.Globalization.CultureInfo.InvariantCulture, $"ids from {smallest:N0} to {largest:N0}");
        string native = $"its index kernels read {capabilities.IndexType.Name}";
        string? value = asked, from = source;
        if (string.IsNullOrWhiteSpace(value))
        {
            value = Environment.GetEnvironmentVariable(IdEnvironmentVariable);
            from = IdEnvironmentVariable;
        }

        if (string.IsNullOrWhiteSpace(value) || value.Trim().Equals(Auto, StringComparison.OrdinalIgnoreCase))
        {
            var fits = IdTypes.FirstOrDefault(t => t.Holds(smallest, largest)) ?? throw new ArgumentException($"No registered integer type holds {range}.");
            return new ElementTypeChoice(fits, Picked: true, Source: null) { Device = device.ToString(), Native = native, Role = "id type", Measured = $"{range} fit {fits.Name}" };
        }

        var type = Find(value.AsSpan().Trim());
        if (type is null || !IdTypes.Contains(type))
        {
            throw new ArgumentException($"{from} '{value}': ids are stored in one of {Auto}, {string.Join(", ", IdTypes.Select(t => t.Name))} (auto picks the smallest that holds them).");
        }

        if (!type.Holds(smallest, largest))
        {
            throw new ArgumentException($"{from} '{value}': {type.Name} holds [{type.MinValue}, {type.MaxValue}], not the data's {range}.");
        }

        return new ElementTypeChoice(type, Picked: false, Source: from) { Device = device.ToString(), Native = native, Role = "id type" };
    }

    /// <summary>
    /// The registered type of the C# type <typeparamref name="T"/>: float32 for <see cref="float"/>, float64 for
    /// <see cref="double"/>, float16 for <see cref="Half"/>, int8 to int64 and uint8 to uint64 for the integers, bool for
    /// <see cref="bool"/>; null for a C# type no built-in stores (bfloat16, fp8, int4 and the block formats have none:
    /// their bytes are read with <c>Tensor.ToBytes</c>). Tested on <c>typeof(T)</c>, which the JIT folds (rule 89: no box).
    /// </summary>
    public static ElementType? Of<T>()
        where T : unmanaged =>
        typeof(T) == typeof(float) ? ElementType.Float32
        : typeof(T) == typeof(double) ? ElementType.Float64
        : typeof(T) == typeof(Half) ? ElementType.Float16
        : typeof(T) == typeof(sbyte) ? ElementType.Int8
        : typeof(T) == typeof(byte) ? ElementType.UInt8
        : typeof(T) == typeof(short) ? ElementType.Int16
        : typeof(T) == typeof(ushort) ? ElementType.UInt16
        : typeof(T) == typeof(int) ? ElementType.Int32
        : typeof(T) == typeof(uint) ? ElementType.UInt32
        : typeof(T) == typeof(long) ? ElementType.Int64
        : typeof(T) == typeof(ulong) ? ElementType.UInt64
        : typeof(T) == typeof(bool) ? ElementType.Bool
        : null;

    /// <summary>
    /// Publishes <paramref name="choice"/> for a run on <paramref name="device"/> to telemetry
    /// (<see cref="Diagnostics.ElementTypeChosen"/>, at <see cref="Diagnostics.TelemetryLevel.Devices"/>): a load says which type
    /// it computes in on each device, how it was chosen and what the device reported.
    /// </summary>
    public static void Report(ElementTypeChoice choice, Device device)
    {
        ArgumentNullException.ThrowIfNull(choice);
        ArgumentNullException.ThrowIfNull(device);
        if (Diagnostics.Telemetry.IsEnabled(Diagnostics.TelemetryLevel.Devices))
        {
            Diagnostics.Telemetry.ElementTypeChosen(new Diagnostics.ElementTypeChosen(device.ToString(), choice.Type.Name, choice.How, choice.Native ?? "", choice.Role));
        }
    }

    /// <summary>Conversions the dispatcher made so far for an operation whose kernel takes another type (<see cref="ConvertedElements"/> elements).</summary>
    public static long Conversions => Interlocked.Read(ref s_conversions);

    /// <summary>The elements those conversions moved.</summary>
    public static long ConvertedElements => Interlocked.Read(ref s_convertedElements);

    /// <summary>The conversions so far by operation ("Gather", "RouteTopK", ...).</summary>
    public static IReadOnlyDictionary<string, long> ConversionsByOperation => new Dictionary<string, long>(s_byOperation, StringComparer.Ordinal);

    /// <summary>
    /// Counts one conversion <paramref name="backend"/> made for <paramref name="operation"/> because its kernel takes
    /// another type than its operand's (plan 37, T0; the brief: never widened silently): <paramref name="elements"/>
    /// elements of <paramref name="from"/> converted to <paramref name="to"/>, in <see cref="Conversions"/>,
    /// <see cref="ConversionsByOperation"/> and the telemetry (<see cref="Diagnostics.ElementsConverted"/>), as the
    /// dispatcher counts its own. A backend whose kernel for a type is missing and that runs the operation through another
    /// type (bfloat16 values computed as float32) counts each such conversion here.
    /// </summary>
    public static void CountConversion(Devices.Backend backend, Operations.Operation operation, ElementType from, ElementType to, long elements)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(operation);
        ArgumentNullException.ThrowIfNull(from);
        ArgumentNullException.ThrowIfNull(to);
        ArgumentOutOfRangeException.ThrowIfNegative(elements);
        Converted(backend.Kind, operation.Name, from, to, elements);
    }

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

    // Widened a vector at a time (the halves of each 16-bit vector to int32, then to float32), a scalar tail for the rest.
    public static void Int16ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        var values = MemoryMarshal.Cast<byte, short>(source)[..destination.Length];
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            for (; i <= values.Length - Vector<short>.Count; i += Vector<short>.Count)
            {
                Vector.Widen(new Vector<short>(values[i..]), out var low, out var high);
                Vector.ConvertToSingle(low).CopyTo(destination[i..]);
                Vector.ConvertToSingle(high).CopyTo(destination[(i + Vector<int>.Count)..]);
            }
        }

        for (; i < values.Length; i++)
        {
            destination[i] = values[i];
        }
    }

    public static void Int16FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        var values = MemoryMarshal.Cast<byte, short>(destination);
        for (int i = 0; i < source.Length; i++)
        {
            values[i] = (short)Saturate(source[i], short.MinValue, short.MaxValue);
        }
    }

    public static void UInt16ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        var values = MemoryMarshal.Cast<byte, ushort>(source)[..destination.Length];
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            for (; i <= values.Length - Vector<ushort>.Count; i += Vector<ushort>.Count)
            {
                Vector.Widen(new Vector<ushort>(values[i..]), out var low, out var high);
                Vector.ConvertToSingle(Vector.AsVectorInt32(low)).CopyTo(destination[i..]);     // at most 65,535: exact as int32
                Vector.ConvertToSingle(Vector.AsVectorInt32(high)).CopyTo(destination[(i + Vector<int>.Count)..]);
            }
        }

        for (; i < values.Length; i++)
        {
            destination[i] = values[i];
        }
    }

    public static void UInt16FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        var values = MemoryMarshal.Cast<byte, ushort>(destination);
        for (int i = 0; i < source.Length; i++)
        {
            values[i] = (ushort)Saturate(source[i], ushort.MinValue, ushort.MaxValue);
        }
    }

    // Rounded to the nearest float32 (exact up to 2^24, as int32's).
    public static void UInt32ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        var values = MemoryMarshal.Cast<byte, uint>(source);
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = values[i];
        }
    }

    // Rounded to nearest (ties to even) and saturated to [0, 2^32 - 1]; NaN gives 0.
    public static void UInt32FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        var values = MemoryMarshal.Cast<byte, uint>(destination);
        for (int i = 0; i < source.Length; i++)
        {
            float x = MathF.Round(source[i]);
            values[i] = float.IsNaN(x) || x <= 0f ? 0u : x >= 4294967296f ? uint.MaxValue : (uint)x;
        }
    }

    public static void UInt64ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        var values = MemoryMarshal.Cast<byte, ulong>(source);
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = values[i];
        }
    }

    // Rounded to nearest (ties to even) and saturated to [0, 2^64 - 1]; NaN gives 0.
    public static void UInt64FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        var values = MemoryMarshal.Cast<byte, ulong>(destination);
        for (int i = 0; i < source.Length; i++)
        {
            float x = MathF.Round(source[i]);
            values[i] = float.IsNaN(x) || x <= 0f ? 0ul : x >= 18446744073709551616f ? ulong.MaxValue : (ulong)x;
        }
    }

    public static void UInt2ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = source[i >> 2] >> ((i & 3) * 2) & 0x3;
        }
    }

    public static void UInt2FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        destination[..(int)((source.Length + 3L) / 4)].Clear();
        for (int i = 0; i < source.Length; i++)
        {
            destination[i >> 2] |= (byte)((Saturate(source[i], 0, 3) & 0x3) << ((i & 3) * 2));
        }
    }

    // Narrowed to the nearest float32 (ties to even; past float32's range an infinity, as a cast does).
    public static void Float64ToFloat32(ReadOnlySpan<byte> source, Span<float> destination)
    {
        var values = MemoryMarshal.Cast<byte, double>(source);
        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = (float)values[i];
        }
    }

    // Exact: every float32 is a float64.
    public static void Float64FromFloat32(ReadOnlySpan<float> source, Span<byte> destination)
    {
        var values = MemoryMarshal.Cast<byte, double>(destination);
        for (int i = 0; i < source.Length; i++)
        {
            values[i] = source[i];
        }
    }

    // Rounded to nearest (ties to even), clamped to [low, high]; NaN gives 0.
    private static int Saturate(float x, int low, int high) => float.IsNaN(x) ? 0 : (int)Math.Clamp(MathF.Round(x), low, high);

    /// <summary>
    /// Converts <paramref name="count"/> elements of <paramref name="from"/> in <paramref name="source"/> to
    /// <paramref name="to"/> in <paramref name="destination"/>. Indices between float32, int32 and int64 are exact or
    /// refused (an index that is not a whole number, or past what the other type holds exactly: float32 holds integers up
    /// to 2²⁴); between any two integer types of whole bytes (uint8, uint16, int16, uint32, ...) exact, or refused for a
    /// value past the destination's range; any other pair goes through float32 with the types' own conversions.
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

        // Between two integer types of whole bytes (ids and indices: uint8, uint16, int32, ...): exact, or refused for a
        // value the destination does not hold.
        if (from.IsInteger && to.IsInteger && from.BlockBits % 8 == 0 && to.BlockBits % 8 == 0)
        {
            for (int i = 0; i < count; i++)
            {
                var v = ReadInteger(from, source, i);
                if (v < to.MinValue || v > to.MaxValue)
                {
                    throw new ArgumentException($"Index {v} (element {i}) is past what {to.Name} holds ([{to.MinValue}, {to.MaxValue}]).");
                }

                WriteInteger(to, destination, i, v);
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

    // Element i of an integer type of whole bytes, little-endian.
    private static Int128 ReadInteger(ElementType type, ReadOnlySpan<byte> source, int i) => (type.BlockBits, type.IsSigned) switch
    {
        (8, true) => (sbyte)source[i],
        (8, false) => source[i],
        (16, true) => System.Buffers.Binary.BinaryPrimitives.ReadInt16LittleEndian(source[(2 * i)..]),
        (16, false) => System.Buffers.Binary.BinaryPrimitives.ReadUInt16LittleEndian(source[(2 * i)..]),
        (32, true) => System.Buffers.Binary.BinaryPrimitives.ReadInt32LittleEndian(source[(4 * i)..]),
        (32, false) => System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(source[(4 * i)..]),
        (64, true) => System.Buffers.Binary.BinaryPrimitives.ReadInt64LittleEndian(source[(8 * i)..]),
        (64, false) => System.Buffers.Binary.BinaryPrimitives.ReadUInt64LittleEndian(source[(8 * i)..]),
        _ => throw new NotSupportedException($"{type.Name}: an integer of {type.BlockBits} bits is not read as whole bytes."),
    };

    // Writes v (held by the type) as element i, little-endian.
    private static void WriteInteger(ElementType type, Span<byte> destination, int i, Int128 v)
    {
        switch (type.BlockBits)
        {
            case 8:
                destination[i] = (byte)v;
                break;
            case 16:
                System.Buffers.Binary.BinaryPrimitives.WriteUInt16LittleEndian(destination[(2 * i)..], (ushort)v);
                break;
            case 32:
                System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(destination[(4 * i)..], (uint)v);
                break;
            case 64:
                System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(destination[(8 * i)..], (ulong)v);
                break;
            default:
                throw new NotSupportedException($"{type.Name}: an integer of {type.BlockBits} bits is not written as whole bytes.");
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
