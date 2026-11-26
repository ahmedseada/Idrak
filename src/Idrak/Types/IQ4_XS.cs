// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models;

namespace Idrak.Types;

/// <summary>ggml's IQ4_XS: IQ4_NL's grid entry by a 4-bit code, times d (float16) and each sub-block's 6-bit scale less 32 (its low 4 bits a nibble from byte 4, its high 2 from the 16-bit word at 2).</summary>
public sealed class IQ4_XS : IElementType
{
    private IQ4_XS()
    {
    }

    /// <summary>The type.</summary>
    public static IQ4_XS Type { get; } = new();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(256, 136, GgufGrids.Iq4Numbers, [new CodePlane(8, 4, 32)], Scales: [new BlockScale(NumberType.Floating(5, 10), 256, 0), new BlockScale(NumberType.Integer(6, false), 32, 0, [.. Enumerable.Range(0, 8).Select(ib => new[] { new ScalePiece(4 + ib / 2, 4 * (ib % 2), 4), new ScalePiece(2 + ib / 4, 2 * (ib % 4), 2) })], -32)]);

    /// <inheritdoc />
    public string Name => "iq4_xs";

    /// <inheritdoc />
    public override string ToString() => Name;
}
