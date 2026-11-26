// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Types;

/// <summary>ggml's Q5_K: 5-bit codes in super-blocks of 256 (a plane of 4 bits and one of the fifth bits), value v of sub-block j (of 32) d · scale_j · q − dmin · min_j, scale_j and min_j 6 bits, d and dmin float16.</summary>
public sealed class Q5_K : IElementType
{
    private Q5_K()
    {
    }

    /// <summary>The type.</summary>
    public static IElementType Type { get; } = new Q5_K();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(256, 176, NumberType.Integer(5, false), [new CodePlane(48, 4, 64), new CodePlane(16, 1, 256)], 0, [new BlockScale(NumberType.Floating(5, 10), 256, 0), new BlockScale(NumberType.Integer(6, false), 32, 0, [.. Enumerable.Range(0, 8).Select(Scale)])], [new BlockScale(NumberType.Floating(5, 10), 256, 2), new BlockScale(NumberType.Integer(6, false), 32, 0, [.. Enumerable.Range(0, 8).Select(Min)])]);

    /// <inheritdoc />
    public string Name => "q5_k";

    // The twelve bytes from byte 4: sub-blocks 0-3 a 6-bit scale and minimum each; 4-7 their low four bits in bytes 12-15
    // and their high two in the top bits of bytes 4-11.
    private static ScalePiece[] Scale(int j) => j < 4 ? [new(4 + j, 0, 6)] : [new(8 + j, 0, 4), new(j, 6, 2)];

    private static ScalePiece[] Min(int j) => j < 4 ? [new(8 + j, 0, 6)] : [new(8 + j, 4, 4), new(4 + j, 6, 2)];

    /// <inheritdoc />
    public override string ToString() => Name;
}
