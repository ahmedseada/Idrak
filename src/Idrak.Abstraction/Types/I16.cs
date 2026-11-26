// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Types;

/// <summary>A signed 16-bit integer (two's complement, little-endian).</summary>
public sealed class I16 : IElementType
{
    private readonly RowScales _rows;

    private I16()
    {
        _rows = new RowScales(this);
    }

    /// <summary>The type.</summary>
    public static IElementType Type { get; } = new I16();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(1, 2, NumberType.Integer(16, signed: true), [new CodePlane(0, 16, 1)]);

    /// <inheritdoc />
    public string Name => "int16";

    /// <summary>
    /// A row of int16 values written with a float32 scale of its own (int16 holds no fraction: a weight's row is its codes
    /// times the row's scale, <see cref="RowScale"/>), each row length's type made once.
    /// </summary>
    /// <param name="length">Values a row holds.</param>
    public IElementType ForRow(int length) => _rows.Of(length);

    /// <inheritdoc />
    public override string ToString() => Name;
}
