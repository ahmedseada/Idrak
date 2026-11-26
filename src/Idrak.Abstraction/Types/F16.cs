// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Types;

/// <summary>IEEE 754 binary16.</summary>
public sealed class F16 : IElementType
{
    private F16()
    {
    }

    /// <summary>The type.</summary>
    public static IElementType Type { get; } = new F16();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(1, 2, NumberType.Floating(5, 10), [new CodePlane(0, 16, 1)]);

    /// <inheritdoc />
    public string Name => "float16";

    /// <inheritdoc />
    public override string ToString() => Name;
}
