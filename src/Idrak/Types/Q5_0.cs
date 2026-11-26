// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Types;

/// <summary>ggml's Q5_0: 5-bit codes less 16 times a float16 scale, 32 a block, their fifth bits a stream of 1-bit fields.</summary>
public sealed class Q5_0 : IElementType
{
    private Q5_0()
    {
    }

    /// <summary>The type.</summary>
    public static Q5_0 Type { get; } = new();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(32, 22, NumberType.Integer(5, false), [new CodePlane(6, 4, 32), new CodePlane(2, 1, 1, Stream: true)], -16, [new BlockScale(NumberType.Floating(5, 10), 32, 0)]);

    /// <inheritdoc />
    public string Name => "q5_0";

    /// <inheritdoc />
    public override string ToString() => Name;
}
