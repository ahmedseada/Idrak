// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Types;

/// <summary>OCP fp8 E4M3 (E4M3fn): 4 exponent bits, 3 mantissa bits, no infinity, ±448 at most, the all-ones codes NaN.</summary>
public sealed class F8E4M3 : IElementType
{
    private F8E4M3()
    {
    }

    /// <summary>The type.</summary>
    public static F8E4M3 Type { get; } = new();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(1, 1, NumberType.Floating(4, 3, specials: FloatSpecials.AllOnesNaN), [new CodePlane(0, 8, 1)]);

    /// <inheritdoc />
    public string Name => "float8_e4m3";

    /// <inheritdoc />
    public override string ToString() => Name;
}
