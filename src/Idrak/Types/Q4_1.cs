// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Types;

/// <summary>ggml's Q4_1: d · q + m, d and m float16, 32 4-bit codes a block.</summary>
public sealed class Q4_1 : IElementType
{
    private Q4_1()
    {
    }

    /// <summary>The type.</summary>
    public static IElementType Type { get; } = new Q4_1();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(32, 20, NumberType.Integer(4, false), [new CodePlane(4, 4, 32)], Scales: [new BlockScale(NumberType.Floating(5, 10), 32, 0)], Mins: [new BlockScale(NumberType.Floating(5, 10), 32, 2)], MinsAdded: true);

    /// <inheritdoc />
    public string Name => "q4_1";

    /// <inheritdoc />
    public override string ToString() => Name;
}
