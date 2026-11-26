// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models;

namespace Idrak.Types;

/// <summary>ggml's IQ1_S: a codebook of its grid's values plus the delta (±1/8, its sign the index's top bit) by a 12-bit index (a byte, 3 high bits listed in 16-bit words of 4, the delta's sign a bit a sub-block), each sub-block's odd scale 1 + 2s, a float16 super-scale.</summary>
public sealed class IQ1_S : IElementType
{
    private IQ1_S()
    {
    }

    /// <summary>The type.</summary>
    public static IQ1_S Type { get; } = new();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(256, 50, GgufGrids.Iq1Codebook, [new CodePlane(2, 8, 1), new CodePlane(0, 3, 1, Places: [.. Enumerable.Range(0, 32).Select(i => new FieldPlace(34 + 2 * (i / 4) + 3 * (i % 4) / 8, 3 * (i % 4) % 8))]), new CodePlane(0, 1, 1, Places: [.. Enumerable.Range(0, 32).Select(i => new FieldPlace(35 + 2 * (i / 4), 7))])], Scales: [new BlockScale(NumberType.Floating(5, 10), 256, 0), new BlockScale(GgufGrids.Odd(3), 32, 0, [.. Enumerable.Range(0, 8).Select(ib => new[] { new ScalePiece(35 + 2 * ib, 4, 3) })])]);

    /// <inheritdoc />
    public string Name => "iq1_s";

    /// <inheritdoc />
    public override string ToString() => Name;
}
