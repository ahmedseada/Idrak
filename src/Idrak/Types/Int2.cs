// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Types;

/// <summary>2-bit signed codes in blocks of 32, each block a float16 scale (its first 2 bytes), then its 8 bytes of codes.</summary>
public sealed class Int2 : IElementType
{
    private Int2()
    {
    }

    /// <summary>The type.</summary>
    public static IElementType Type { get; } = new Int2();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(32, 10, NumberType.Integer(2, signed: true), [new CodePlane(2, 2, 4)], Scales: [new BlockScale(NumberType.Floating(5, 10), 32, 0)]);

    /// <inheritdoc />
    public string Name => "int2";

    /// <inheritdoc />
    public override string ToString() => Name;
}
