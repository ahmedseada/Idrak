// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Types;

/// <summary>An unsigned 32-bit integer (little-endian).</summary>
public sealed class U32 : IElementType
{
    private U32()
    {
    }

    /// <summary>The type.</summary>
    public static U32 Type { get; } = new();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(1, 4, NumberType.Integer(32, signed: false), [new CodePlane(0, 32, 1)]);

    /// <inheritdoc />
    public string Name => "uint32";

    /// <inheritdoc />
    public override string ToString() => Name;
}
