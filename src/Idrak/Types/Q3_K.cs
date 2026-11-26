// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Types;

/// <summary>ggml's Q3_K: 3-bit codes less 4 in super-blocks of 256 (two planes, the low two bits and the high bit), value v of sub-block j (of 16) d · (scale_j − 32) · q, scale_j 6 bits, d float16.</summary>
public sealed class Q3_K : IElementType
{
    private Q3_K()
    {
    }

    /// <summary>The type.</summary>
    public static IElementType Type { get; } = new Q3_K();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(256, 110, NumberType.Integer(3, false), [new CodePlane(32, 2, 128), new CodePlane(0, 1, 256)], -4, [new BlockScale(NumberType.Floating(5, 10), 256, 108), new BlockScale(NumberType.Integer(6, false), 16, 0, [.. Enumerable.Range(0, 16).Select(Scale)], -32)]);

    /// <inheritdoc />
    public string Name => "q3_k";

    // The twelve bytes from byte 96: sixteen 6-bit scales, the low four bits in the halves of bytes 96-103, the high two
    // in the bit pairs of bytes 104-107.
    private static ScalePiece[] Scale(int j) => (j / 4) switch
    {
        0 => [new(96 + j, 0, 4), new(104 + j, 0, 2)],
        1 => [new(96 + j, 0, 4), new(104 + j - 4, 2, 2)],
        2 => [new(96 + j - 8, 4, 4), new(104 + j - 8, 4, 2)],
        _ => [new(96 + j - 8, 4, 4), new(104 + j - 12, 6, 2)],
    };

    /// <inheritdoc />
    public override string ToString() => Name;
}
