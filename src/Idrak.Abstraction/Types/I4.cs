// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Types;

/// <summary>4-bit signed integers (−8 to 7), two a byte, the first in the low half.</summary>
public sealed class I4 : IElementType
{
    private I4()
    {
    }

    /// <summary>The type.</summary>
    public static I4 Type { get; } = new();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(2, 1, NumberType.Integer(4, signed: true), [new CodePlane(0, 4, 2)]);

    /// <inheritdoc />
    public string Name => "int4";

    /// <inheritdoc />
    public override string ToString() => Name;
}
