// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Types;

/// <summary>OCP's E8M0 scale: 2^(e − 127), no sign and no mantissa, 0xFF NaN.</summary>
public sealed class F8E8M0 : IElementType
{
    private F8E8M0()
    {
    }

    /// <summary>The type.</summary>
    public static IElementType Type { get; } = new F8E8M0();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(1, 1, NumberType.Floating(8, 0, signed: false, bias: 127, subnormals: false, specials: FloatSpecials.AllOnesNaN), [new CodePlane(0, 8, 1)]);

    /// <inheritdoc />
    public string Name => "float8_e8m0";

    /// <inheritdoc />
    public override string ToString() => Name;
}
