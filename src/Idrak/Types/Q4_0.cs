// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Types;

/// <summary>ggml's Q4_0: 32 4-bit codes less 8 times one float16 scale a block, value i + 16 in byte i's high half.</summary>
public sealed class Q4_0 : IElementType
{
    private Q4_0()
    {
    }

    /// <summary>The type.</summary>
    public static IElementType Type { get; } = new Q4_0();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(32, 18, NumberType.Integer(4, signed: false), [new CodePlane(2, 4, 32)], -8, [new BlockScale(NumberType.Floating(5, 10), 32, 0)]);

    /// <inheritdoc />
    public string Name => "q4_0";

    /// <inheritdoc />
    public override string ToString() => Name;
}
