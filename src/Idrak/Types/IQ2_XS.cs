// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models;

namespace Idrak.Types;

/// <summary>ggml's IQ2_XS: a codebook of its grid's magnitudes by a 9-bit index (a 16-bit word with a 7-bit index of ggml's sign masks), each sub-block of 16 a 4-bit scale plus 1/2, a float16 super-scale.</summary>
public sealed class IQ2_XS : IElementType
{
    private IQ2_XS()
    {
    }

    /// <summary>The type.</summary>
    public static IElementType Type { get; } = new IQ2_XS();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(256, 74, GgufGrids.Codebook(9, 8, GgufGrids.IQ2XS, 0.25), [new CodePlane(2, 9, 1, Places: [.. Enumerable.Range(0, 32).Select(e => new FieldPlace(2 + 2 * e, 0))])], Scales: [new BlockScale(NumberType.Floating(5, 10), 256, 0), new BlockScale(NumberType.Integer(4, false), 16, 66, Bias: 0.5), new BlockScale(GgufGrids.KSignVectors, 8, 0, [.. Enumerable.Range(0, 32).Select(e => new[] { new ScalePiece(3 + 2 * e, 1, 7) })])]);

    /// <inheritdoc />
    public string Name => "iq2_xs";

    /// <inheritdoc />
    public override string ToString() => Name;
}
