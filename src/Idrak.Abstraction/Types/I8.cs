// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Types;

/// <summary>A signed 8-bit integer (two's complement).</summary>
public sealed class I8 : IElementType
{
    private readonly RowScales _rows;

    private I8()
    {
        _rows = new RowScales(this);
    }

    /// <summary>The type.</summary>
    public static IElementType Type { get; } = new I8();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(1, 1, NumberType.Integer(8, signed: true), [new CodePlane(0, 8, 1)]);

    /// <inheritdoc />
    public string Name => "int8";

    /// <summary>
    /// A row of int8 values written with a float32 scale of its own (int8 holds no fraction: a weight's row is its codes
    /// times the row's scale, <see cref="RowScale"/>), each row length's type made once.
    /// </summary>
    /// <param name="length">Values a row holds.</param>
    public IElementType ForRow(int length) => _rows.Of(length);

    /// <inheritdoc />
    public override string ToString() => Name;
}
