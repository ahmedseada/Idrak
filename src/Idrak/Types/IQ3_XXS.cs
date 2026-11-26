// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models;

namespace Idrak.Types;

/// <summary>ggml's IQ3_XXS: a codebook of its grid's magnitudes (4 a byte index, times the sub-scales' half), each sub-block's 4-bit scale plus 1/2, each group of 8 values' signs a 7-bit index of ggml's sign masks, a float16 super-scale.</summary>
public sealed class IQ3_XXS : IElementType
{
    private IQ3_XXS()
    {
    }

    /// <summary>The type.</summary>
    public static IQ3_XXS Type { get; } = new();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(256, 98, GgufGrids.Codebook(8, 4, GgufGrids.IQ3XXS, 0.5), [new CodePlane(2, 8, 1)], 0, [new BlockScale(NumberType.Floating(5, 10), 256, 0), new BlockScale(NumberType.Integer(4, false), 32, 0, [.. Enumerable.Range(0, 8).Select(j => ScalePiece.Run(66 + 4 * j, 28, 4))], 0.5), new BlockScale(GgufGrids.KSignVectors, 8, 0, [.. Enumerable.Range(0, 32).Select(s => ScalePiece.Run(66 + 4 * (s / 4), 7 * (s % 4), 7))])]);

    /// <inheritdoc />
    public string Name => "iq3_xxs";

    /// <inheritdoc />
    public override string ToString() => Name;
}
