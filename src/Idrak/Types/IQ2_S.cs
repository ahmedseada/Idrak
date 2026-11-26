// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models;

namespace Idrak.Types;

/// <summary>ggml's IQ2_S: a codebook of its grid's magnitudes by a 10-bit index (a byte and 2 high bits in a plane of their own), each sub-block of 16 a 4-bit scale plus 1/2, each group of 8 values' signs a byte, a float16 super-scale.</summary>
public sealed class IQ2_S : IElementType
{
    private IQ2_S()
    {
    }

    /// <summary>The type.</summary>
    public static IElementType Type { get; } = new IQ2_S();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(256, 82, GgufGrids.Codebook(10, 8, GgufGrids.IQ2S, 0.25), [new CodePlane(2, 8, 1), new CodePlane(66, 2, 4)], Scales: [new BlockScale(NumberType.Floating(5, 10), 256, 0), new BlockScale(NumberType.Integer(4, false), 16, 74, Bias: 0.5), new BlockScale(GgufGrids.SignBits, 8, 34)]);

    /// <inheritdoc />
    public string Name => "iq2_s";

    /// <inheritdoc />
    public override string ToString() => Name;
}
