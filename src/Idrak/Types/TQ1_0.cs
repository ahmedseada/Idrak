// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Types;

/// <summary>ggml's TQ1_0: ternary digits less 1, five a byte read as a base-3 fraction (value 32p + m digit p of byte m, 160 + 16p + m digit p of byte 32 + m, 240 + 4p + m digit p of byte 48 + m), a float16 scale after them.</summary>
public sealed class TQ1_0 : IElementType
{
    private TQ1_0()
    {
    }

    /// <summary>The type.</summary>
    public static IElementType Type { get; } = new TQ1_0();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(256, 54, NumberType.Integer(2, false), [new CodePlane(0, 2, 1, Places: [.. Enumerable.Range(0, 256).Select(Digit)], Radix: 3)], -1, [new BlockScale(NumberType.Floating(5, 10), 256, 52)]);

    /// <inheritdoc />
    public string Name => "tq1_0";

    // Value v's byte and digit.
    private static FieldPlace Digit(int v) => v < 160 ? new(v % 32, v / 32) : v < 240 ? new(32 + (v - 160) % 16, (v - 160) / 16) : new(48 + (v - 240) % 4, (v - 240) / 4);

    /// <inheritdoc />
    public override string ToString() => Name;
}
