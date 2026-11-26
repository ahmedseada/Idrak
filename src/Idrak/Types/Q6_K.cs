// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Types;

/// <summary>ggml's Q6_K: 6-bit codes less 32 in super-blocks of 256 (a plane of 4 bits and one of 2), value v of sub-block j (of 16) d · scale_j · q, scale_j a signed byte, d float16.</summary>
public sealed class Q6_K : IElementType
{
    private Q6_K()
    {
    }

    /// <summary>The type.</summary>
    public static Q6_K Type { get; } = new();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(256, 210, NumberType.Integer(6, false), [new CodePlane(0, 4, 128), new CodePlane(128, 2, 128)], -32, [new BlockScale(NumberType.Floating(5, 10), 256, 208), new BlockScale(NumberType.Integer(8, true), 16, 0, [.. Enumerable.Range(0, 16).Select(j => new[] { new ScalePiece(192 + j, 0, 8) })])]);

    /// <inheritdoc />
    public string Name => "q6_k";

    /// <inheritdoc />
    public override string ToString() => Name;
}
