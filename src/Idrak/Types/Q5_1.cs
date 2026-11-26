// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Types;

/// <summary>ggml's Q5_1: d · q + m, d and m float16, 32 5-bit codes a block, their fifth bits a stream of 1-bit fields.</summary>
public sealed class Q5_1 : IElementType
{
    private Q5_1()
    {
    }

    /// <summary>The type.</summary>
    public static IElementType Type { get; } = new Q5_1();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(32, 24, NumberType.Integer(5, false), [new CodePlane(8, 4, 32), new CodePlane(4, 1, 1, Stream: true)], Scales: [new BlockScale(NumberType.Floating(5, 10), 32, 0)], Mins: [new BlockScale(NumberType.Floating(5, 10), 32, 2)], MinsAdded: true);

    /// <inheritdoc />
    public string Name => "q5_1";

    /// <inheritdoc />
    public override string ToString() => Name;
}
