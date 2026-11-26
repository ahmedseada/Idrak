// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Types;

/// <summary>ggml's Q8_K (an activation type of its k-quant dot products): 256 signed 8-bit codes times a float32 scale d a block; its 16 sums of 16 codes (bytes 260-291) are not part of the values. The described encoder does not write a float32 scale level yet.</summary>
public sealed class Q8_K : IElementType
{
    private Q8_K()
    {
    }

    /// <summary>The type.</summary>
    public static IElementType Type { get; } = new Q8_K();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(256, 292, NumberType.Integer(8, signed: true), [new CodePlane(4, 8, 1)], 0, [new BlockScale(NumberType.Floating(8, 23), 256, 0)]);

    /// <inheritdoc />
    public string Name => "q8_k";

    /// <inheritdoc />
    public override string ToString() => Name;
}
