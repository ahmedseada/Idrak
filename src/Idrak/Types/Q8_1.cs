// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Types;

/// <summary>
/// ggml's Q8_1 (an activation type of its dot products): 32 signed 8-bit codes times a float16 scale d a block; its sum
/// s (bytes 2-3, what ggml's dot products read) is the sum of the block's values, d · Σq, as float16: described as a sum,
/// not part of the values, written from the values stored.
/// </summary>
public sealed class Q8_1 : IElementType
{
    private Q8_1()
    {
    }

    /// <summary>The type.</summary>
    public static IElementType Type { get; } = new Q8_1();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(32, 36, NumberType.Integer(8, signed: true), [new CodePlane(4, 8, 1)], 0, [new BlockScale(NumberType.Floating(5, 10), 32, 0)],
        Sums: [new BlockSum(NumberType.Floating(5, 10), 32, 2, OfValues: true)]);

    /// <inheritdoc />
    public string Name => "q8_1";

    /// <inheritdoc />
    public override string ToString() => Name;
}
