// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Types;

/// <summary>ggml's Q8_0: 32 signed 8-bit codes times one float16 scale a block.</summary>
public sealed class Q8_0 : IElementType
{
    private Q8_0()
    {
    }

    /// <summary>The type.</summary>
    public static IElementType Type { get; } = new Q8_0();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(32, 34, NumberType.Integer(8, signed: true), [new CodePlane(2, 8, 1)], 0, [new BlockScale(NumberType.Floating(5, 10), 32, 0)]);

    /// <inheritdoc />
    public string Name => "q8_0";

    /// <inheritdoc />
    public override string ToString() => Name;
}
