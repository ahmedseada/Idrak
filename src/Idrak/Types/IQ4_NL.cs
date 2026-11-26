// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models;

namespace Idrak.Types;

/// <summary>ggml's IQ4_NL: 32 4-bit codes of its non-linear grid (kvalues_iq4nl) times one float16 scale a block.</summary>
public sealed class IQ4_NL : IElementType
{
    private IQ4_NL()
    {
    }

    /// <summary>The type.</summary>
    public static IElementType Type { get; } = new IQ4_NL();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(32, 18, GgufGrids.Iq4Numbers, [new CodePlane(2, 4, 32)], 0, [new BlockScale(NumberType.Floating(5, 10), 32, 0)]);

    /// <inheritdoc />
    public string Name => "iq4_nl";

    /// <inheritdoc />
    public override string ToString() => Name;
}
