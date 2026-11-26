// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Types;

/// <summary>An unsigned 16-bit integer (little-endian).</summary>
public sealed class U16 : IElementType
{
    private U16()
    {
    }

    /// <summary>The type.</summary>
    public static U16 Type { get; } = new();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(1, 2, NumberType.Integer(16, signed: false), [new CodePlane(0, 16, 1)]);

    /// <inheritdoc />
    public string Name => "uint16";

    /// <inheritdoc />
    public override string ToString() => Name;
}
