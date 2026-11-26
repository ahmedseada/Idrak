// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Types;

/// <summary>ggml's Q8_1 (an activation type of its dot products): 32 signed 8-bit codes times a float16 scale d a block; its sum s = d · Σq (bytes 2-3, what ggml's dot products read) is not part of the values: read as nothing, written as zero.</summary>
public sealed class Q8_1 : IElementType
{
    private Q8_1()
    {
    }

    /// <summary>The type.</summary>
    public static IElementType Type { get; } = new Q8_1();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(32, 36, NumberType.Integer(8, signed: true), [new CodePlane(4, 8, 1)], 0, [new BlockScale(NumberType.Floating(5, 10), 32, 0)]);

    /// <inheritdoc />
    public string Name => "q8_1";

    /// <inheritdoc />
    public override string ToString() => Name;
}
