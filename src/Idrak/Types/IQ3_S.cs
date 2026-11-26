// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models;

namespace Idrak.Types;

/// <summary>ggml's IQ3_S: a codebook of its grid's magnitudes by a 9-bit index (a byte and a high bit in a plane of its own), each sub-block of 32 an odd scale 1 + 2s, each group of 8 values' signs a byte, a float16 super-scale.</summary>
public sealed class IQ3_S : IElementType
{
    private IQ3_S()
    {
    }

    /// <summary>The type.</summary>
    public static IQ3_S Type { get; } = new();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(256, 110, GgufGrids.Codebook(9, 4, GgufGrids.IQ3S, 1), [new CodePlane(2, 8, 1), new CodePlane(66, 1, 8)], Scales: [new BlockScale(NumberType.Floating(5, 10), 256, 0), new BlockScale(GgufGrids.Odd(4), 32, 106), new BlockScale(GgufGrids.SignBits, 8, 74)]);

    /// <inheritdoc />
    public string Name => "iq3_s";

    /// <inheritdoc />
    public override string ToString() => Name;
}
