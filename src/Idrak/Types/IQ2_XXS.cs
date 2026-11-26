// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models;

namespace Idrak.Types;

/// <summary>ggml's IQ2_XXS: a codebook of its grid's magnitudes (8 a byte index, times the sub-scales' quarter), each sub-block's 4-bit scale plus 1/2, each group of 8 values' signs a 7-bit index of ggml's sign masks, a float16 super-scale.</summary>
public sealed class IQ2_XXS : IElementType
{
    private IQ2_XXS()
    {
    }

    /// <summary>The type.</summary>
    public static IQ2_XXS Type { get; } = new();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(256, 66, GgufGrids.Codebook(8, 8, GgufGrids.IQ2XXS, 0.25), [new CodePlane(2, 8, 4, 8)], 0, [new BlockScale(NumberType.Floating(5, 10), 256, 0), new BlockScale(NumberType.Integer(4, false), 32, 0, [.. Enumerable.Range(0, 8).Select(j => ScalePiece.Run(2 + 8 * j + 4, 28, 4))], 0.5), new BlockScale(GgufGrids.KSignVectors, 8, 0, [.. Enumerable.Range(0, 32).Select(s => ScalePiece.Run(2 + 8 * (s / 4) + 4, 7 * (s % 4), 7))])]);

    /// <inheritdoc />
    public string Name => "iq2_xxs";

    /// <inheritdoc />
    public override string ToString() => Name;
}
