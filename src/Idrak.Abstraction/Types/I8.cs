// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Types;

/// <summary>A signed 8-bit integer (two's complement).</summary>
public sealed class I8 : IElementType
{
    private I8()
    {
    }

    /// <summary>The type.</summary>
    public static I8 Type { get; } = new();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(1, 1, NumberType.Integer(8, signed: true), [new CodePlane(0, 8, 1)]);

    /// <inheritdoc />
    public string Name => "int8";

    /// <inheritdoc />
    public override string ToString() => Name;
}
