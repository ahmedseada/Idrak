// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Types;

/// <summary>OCP fp4 E2M1: ±0, 0.5, 1, 1.5, 2, 3, 4 and 6, no infinity or NaN; two a byte, the first in the low half.</summary>
public sealed class F4E2M1 : IElementType
{
    private F4E2M1()
    {
    }

    /// <summary>The type.</summary>
    public static IElementType Type { get; } = new F4E2M1();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(2, 1, NumberType.Floating(2, 1, bias: 1, specials: FloatSpecials.None), [new CodePlane(0, 4, 2)]);

    /// <inheritdoc />
    public string Name => "float4_e2m1";

    /// <inheritdoc />
    public override string ToString() => Name;
}
