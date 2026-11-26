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

    /// <inheritdoc />
    public override string ToString() => Name;
}
