// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Types;

/// <summary>A signed 32-bit integer (two's complement, little-endian): the type of indices, exact over its whole range.</summary>
public sealed class I32 : IElementType
{
    private I32()
    {
    }

    /// <summary>The type.</summary>
    public static IElementType Type { get; } = new I32();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(1, 4, NumberType.Integer(32, signed: true), [new CodePlane(0, 32, 1)]);

    /// <inheritdoc />
    public string Name => "int32";

    /// <inheritdoc />
    public override string ToString() => Name;
}
