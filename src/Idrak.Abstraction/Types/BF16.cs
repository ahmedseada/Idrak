// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Types;

/// <summary>bfloat16: binary32's sign and 8 exponent bits, 7 mantissa bits.</summary>
public sealed class BF16 : IElementType
{
    private BF16()
    {
    }

    /// <summary>The type.</summary>
    public static BF16 Type { get; } = new();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(1, 2, NumberType.Floating(8, 7), [new CodePlane(0, 16, 1)]);

    /// <inheritdoc />
    public string Name => "bfloat16";

    /// <inheritdoc />
    public override string ToString() => Name;
}
