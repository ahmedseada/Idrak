// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Types;

/// <summary>A signed 16-bit integer (two's complement, little-endian).</summary>
public sealed class I16 : IElementType
{
    private I16()
    {
    }

    /// <summary>The type.</summary>
    public static IElementType Type { get; } = new I16();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(1, 2, NumberType.Integer(16, signed: true), [new CodePlane(0, 16, 1)]);

    /// <inheritdoc />
    public string Name => "int16";

    /// <summary>
    /// A row of int16 values written with a float32 scale of its own (int16 holds no fraction: a weight's row is its codes
    /// times the row's scale, <see cref="RowScale"/>): a type of its own for this row, made where it is asked (a loader asks
    /// once a weight, and the weight's tensor carries it); two rows of one length describe themselves alike (their
    /// descriptions are equal by value), whatever object each is.
    /// </summary>
    /// <param name="length">Values a row holds.</param>
    public IElementType ForRow(int length) => new RowScale(this, length);

    /// <inheritdoc />
    public override string ToString() => Name;
}
