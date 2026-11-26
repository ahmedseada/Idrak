// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Types;

/// <summary>ggml's Q2_K: 2-bit codes in super-blocks of 256, value v of sub-block j (of 16) d · scale_j · q − dmin · min_j, scale_j and min_j 4 bits, d and dmin float16.</summary>
public sealed class Q2_K : IElementType
{
    private Q2_K()
    {
    }

    /// <summary>The type.</summary>
    public static Q2_K Type { get; } = new();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(256, 84, NumberType.Integer(2, false), [new CodePlane(16, 2, 128)], 0, [new BlockScale(NumberType.Floating(5, 10), 256, 80), new BlockScale(NumberType.Integer(4, false), 16, 0, [.. Enumerable.Range(0, 16).Select(j => new[] { new ScalePiece(j, 0, 4) })])], [new BlockScale(NumberType.Floating(5, 10), 256, 82), new BlockScale(NumberType.Integer(4, false), 16, 0, [.. Enumerable.Range(0, 16).Select(j => new[] { new ScalePiece(j, 4, 4) })])]);

    /// <inheritdoc />
    public string Name => "q2_k";

    /// <inheritdoc />
    public override string ToString() => Name;
}
