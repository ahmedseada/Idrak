// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Types;

/// <summary>ggml's TQ2_0: ternary 2-bit codes less 1 (−1, 0, 1), value 128c + 32s + m in byte 32c + m at bit 2s, a float16 scale after them.</summary>
public sealed class TQ2_0 : IElementType
{
    private TQ2_0()
    {
    }

    /// <summary>The type.</summary>
    public static IElementType Type { get; } = new TQ2_0();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(256, 66, NumberType.Integer(2, false), [new CodePlane(0, 2, 128)], -1, [new BlockScale(NumberType.Floating(5, 10), 256, 64)]);

    /// <inheritdoc />
    public string Name => "tq2_0";

    /// <inheritdoc />
    public override string ToString() => Name;
}
