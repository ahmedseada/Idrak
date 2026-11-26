// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction;

/// <summary>
/// The number kinds a device may compute directly (<see cref="IElementType.Native"/>): a per-element value of one of the
/// machine's own formats, whose stored bits a device's instructions take as they are. <see cref="None"/> for every other
/// type, which a device reads through its description (a block's codes and scales, a format of its own bits).
/// </summary>
public enum NativeKind
{
    /// <summary>No machine format: the values are read through the type's description.</summary>
    None,

    /// <summary>IEEE 754 binary32.</summary>
    Float32,

    /// <summary>IEEE 754 binary64.</summary>
    Float64,

    /// <summary>IEEE 754 binary16.</summary>
    Float16,

    /// <summary>bfloat16: binary32's sign and exponent, 7 mantissa bits.</summary>
    BFloat16,

    /// <summary>OCP fp8 E4M3 (no infinity, the all-ones codes NaN).</summary>
    Float8E4M3,

    /// <summary>OCP fp8 E5M2 (IEEE's specials).</summary>
    Float8E5M2,

    /// <summary>A signed 8-bit integer.</summary>
    Int8,

    /// <summary>An unsigned 8-bit integer.</summary>
    UInt8,

    /// <summary>A signed 16-bit integer.</summary>
    Int16,

    /// <summary>An unsigned 16-bit integer.</summary>
    UInt16,

    /// <summary>A signed 32-bit integer.</summary>
    Int32,

    /// <summary>An unsigned 32-bit integer.</summary>
    UInt32,

    /// <summary>A signed 64-bit integer.</summary>
    Int64,

    /// <summary>An unsigned 64-bit integer.</summary>
    UInt64,
}

/// <summary>
/// A type of stored values, presenting itself: its description (how its values read from their bits,
/// <see cref="TypeDescription"/>), a label, its native kind, its byte sizes, and how its values are read and written.
/// Idrak uses a type through this interface alone (a type referenced directly, or handed over), never by its name: whatever the type is, its description says what its
/// bits are, and every member below has a default written once from the description, safe and fast (the description's
/// own decoding for reading, the described encoder for writing). A type implements
/// <see cref="Description"/> and <see cref="Name"/>; it implements another member only where its trace and measurement
/// found it necessary (a special instruction, a speed-up), and such a member gives the same bits as the default.
/// </summary>
public interface IElementType
{
    /// <summary>How the type's values read from their bits: its blocks, codes, scales and their number types.</summary>
    TypeDescription Description { get; }

    /// <summary>A label for people ("bfloat16", "q4_k"): shown, never used to choose anything.</summary>
    string Name { get; }

    /// <summary>
    /// Whether the type's values are one of a machine's own formats (<see cref="NativeKind"/>), which a device may compute
    /// directly: by default read from the description alone (<see cref="NativeKinds.Of"/>), so a type described as IEEE's
    /// binary32 is native float32 whatever its name.
    /// </summary>
    NativeKind Native => NativeKinds.Of(Description);

    /// <summary>The bytes <paramref name="values"/> values take: whole groups of blocks (<see cref="TypeDescription.UnitValues"/>, <see cref="TypeDescription.UnitBytes"/>).</summary>
    /// <param name="values">The values.</param>
    long ByteCount(long values)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(values);
        var d = Description;
        return (values + d.UnitValues - 1) / d.UnitValues * d.UnitBytes;
    }

    /// <summary>
    /// The first <c>values.Length</c> values stored in <paramref name="stored"/> (whole groups of blocks from its first
    /// byte; the last may hold fewer values), as float32: each value as the description reads it (in float64), rounded
    /// once to float32. A 64-bit value is narrowed only here, where float32 is asked for; read it as float64 with
    /// <see cref="Read(ReadOnlySpan{byte}, Span{double})"/>.
    /// </summary>
    /// <param name="stored">The stored bytes (at least <c>ByteCount(values.Length)</c>).</param>
    /// <param name="values">Where the values go.</param>
    void Read(ReadOnlySpan<byte> stored, Span<float> values) => DescribedReader.Of(Description).Read(stored, values);

    /// <summary>The first <c>values.Length</c> values stored in <paramref name="stored"/> as the description reads them, in float64 (no value of 64 bits or fewer narrowed).</summary>
    /// <param name="stored">The stored bytes (at least <c>ByteCount(values.Length)</c>).</param>
    /// <param name="values">Where the values go.</param>
    void Read(ReadOnlySpan<byte> stored, Span<double> values) => DescribedReader.Of(Description).Read(stored, values);

    /// <summary>
    /// <paramref name="values"/> stored as this type into <paramref name="stored"/> (<c>ByteCount(values.Length)</c>
    /// bytes): by the described encoder, which writes any type from its description (a per-element type rounded as its
    /// number type defines; a block type's scales and codes searched by measured error); a last group of fewer values is
    /// filled with zeros.
    /// </summary>
    /// <param name="values">The values.</param>
    /// <param name="stored">Where the stored bytes go.</param>
    /// <exception cref="NotSupportedException">The description is one the described encoder does not write (<see cref="Unwritable"/> says why).</exception>
    void Write(ReadOnlySpan<float> values, Span<byte> stored) => DescribedWriting.Write(Description, values, stored);

    /// <summary>
    /// <paramref name="values"/> (float64) stored as this type: a per-element type rounded from the float64 value itself
    /// (no 64-bit value narrowed first); a block type's values searched in float32, as <see cref="Write(ReadOnlySpan{float}, Span{byte})"/>.
    /// </summary>
    /// <param name="values">The values.</param>
    /// <param name="stored">Where the stored bytes go.</param>
    /// <exception cref="NotSupportedException">The description is one the described encoder does not write.</exception>
    void Write(ReadOnlySpan<double> values, Span<byte> stored) => DescribedWriting.Write(Description, values, stored);

    /// <summary>Why values cannot be written as this type, or null where they can.</summary>
    string? Unwritable => DescribedEncoder.Why(Description);

    /// <summary>
    /// The type values are written as row by row when they are rows of <paramref name="length"/> values (a weight's output
    /// rows of its input features, a table's rows), the rows one after another: by default this type itself, whose
    /// description holds for a row of any length. A type whose rows carry a number of their own gives the type of such a row,
    /// its description that row's: int8 and int16 hold no fraction, so a row of them takes a float32 scale of its own
    /// (<see cref="Types.RowScale"/>: a block of the row's scale and its codes, read and written by the defaults).
    /// </summary>
    /// <param name="length">Values a row holds.</param>
    IElementType ForRow(int length) => this;
}

/// <summary>
/// The types a package hands over (its own, defined each in its own class): an immutable list, nothing added to it, no
/// lookup in it. An app or a loader gives Idrak the type objects it means; a user's text naming a type is matched against
/// the types' own names only where the text comes in (a tool's switch), by walking such a list.
/// </summary>
public interface ITypeProvider
{
    /// <summary>The types handed over.</summary>
    IReadOnlyList<IElementType> Types { get; }
}
/// <summary>What a description says of a native kind (<see cref="IElementType.Native"/>'s default).</summary>
public static class NativeKinds
{
    /// <summary>
    /// The machine format <paramref name="description"/> is, from its bits alone: one value a block, its code one plane of
    /// whole bytes from the block's first, no offset, scale or minimum, and the code an integer of 8, 16, 32 or 64 bits or
    /// a float of binary32's, binary64's, binary16's, bfloat16's, E4M3's or E5M2's bits; else <see cref="NativeKind.None"/>.
    /// </summary>
    /// <param name="description">The description.</param>
    public static NativeKind Of(TypeDescription description)
    {
        ArgumentNullException.ThrowIfNull(description);
        if (description is not { Values: 1, Group: 1, CodeOffset: 0, Scales: null or [], Mins: null or [], Sums: null or [], Planes: [{ Offset: 0, Group: 1, Stride: 0, Stream: false, Places: null, Radix: 0 } plane] }
            || plane.Bits != description.Code.Bits || description.Bytes * 8 != plane.Bits || description.UnitBytes != description.Bytes)
        {
            return NativeKind.None;
        }

        var c = description.Code;
        return (c.Kind, c.Bits, c.Signed) switch
        {
            (NumberKind.Integer, 8, true) => NativeKind.Int8,
            (NumberKind.Integer, 8, false) => NativeKind.UInt8,
            (NumberKind.Integer, 16, true) => NativeKind.Int16,
            (NumberKind.Integer, 16, false) => NativeKind.UInt16,
            (NumberKind.Integer, 32, true) => NativeKind.Int32,
            (NumberKind.Integer, 32, false) => NativeKind.UInt32,
            (NumberKind.Integer, 64, true) => NativeKind.Int64,
            (NumberKind.Integer, 64, false) => NativeKind.UInt64,
            (NumberKind.Float, _, true) when c.Subnormals => (c.Exponent, c.Mantissa, c.ExponentBias, c.Specials) switch
            {
                (8, 23, 127, FloatSpecials.Ieee) => NativeKind.Float32,
                (11, 52, 1023, FloatSpecials.Ieee) => NativeKind.Float64,
                (5, 10, 15, FloatSpecials.Ieee) => NativeKind.Float16,
                (8, 7, 127, FloatSpecials.Ieee) => NativeKind.BFloat16,
                (4, 3, 7, FloatSpecials.AllOnesNaN) => NativeKind.Float8E4M3,
                (5, 2, 15, FloatSpecials.Ieee) => NativeKind.Float8E5M2,
                _ => NativeKind.None,
            },
            _ => NativeKind.None,
        };
    }
}
