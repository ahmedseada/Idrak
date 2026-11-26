// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models;

namespace Idrak.Types;

/// <summary>ggml's IQ1_M: IQ1_S's codebook by a byte and a nibble an entry, each sub-block of 16 an odd scale 1 + 2s, the float16 super-scale the top nibbles of its 4 scale words.</summary>
public sealed class IQ1_M : IElementType
{
    private IQ1_M()
    {
    }

    /// <summary>The type.</summary>
    public static IElementType Type { get; } = new IQ1_M();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(256, 56, GgufGrids.Iq1Codebook, [new CodePlane(0, 8, 1), new CodePlane(32, 4, 2)], Scales: [new BlockScale(NumberType.Floating(5, 10), 256, 0, [[new(49, 4, 4), new(51, 4, 4), new(53, 4, 4), new(55, 4, 4)]]), new BlockScale(GgufGrids.Odd(3), 16, 0, [.. Enumerable.Range(0, 16).Select(m => ScalePiece.Run(48 + 2 * (m / 4), 3 * (m % 4), 3))])]);

    /// <inheritdoc />
    public string Name => "iq1_m";

    /// <inheritdoc />
    public override string ToString() => Name;
}
