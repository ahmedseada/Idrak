// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Types;

/// <summary>An unsigned 8-bit integer.</summary>
public sealed class U8 : IElementType
{
    private U8()
    {
    }

    /// <summary>The type.</summary>
    public static U8 Type { get; } = new();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(1, 1, NumberType.Integer(8, signed: false), [new CodePlane(0, 8, 1)]);

    /// <inheritdoc />
    public string Name => "uint8";

    /// <inheritdoc />
    public override string ToString() => Name;
}
