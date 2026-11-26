// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Types;

/// <summary>2-bit unsigned integers (0 to 3), four a byte, the first in the lowest bits.</summary>
public sealed class U2 : IElementType
{
    private U2()
    {
    }

    /// <summary>The type.</summary>
    public static U2 Type { get; } = new();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(4, 1, NumberType.Integer(2, signed: false), [new CodePlane(0, 2, 4)]);

    /// <inheritdoc />
    public string Name => "uint2";

    /// <inheritdoc />
    public override string ToString() => Name;
}
