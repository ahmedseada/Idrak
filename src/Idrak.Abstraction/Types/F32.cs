// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Types;

/// <summary>IEEE 754 binary32: the compute type of activations and a device's own storage (one value a word).</summary>
public sealed class F32 : IElementType
{
    private F32()
    {
    }

    /// <summary>The type.</summary>
    public static F32 Type { get; } = new();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(1, 4, NumberType.Floating(8, 23), [new CodePlane(0, 32, 1)]);

    /// <inheritdoc />
    public string Name => "float32";

    /// <inheritdoc />
    public override string ToString() => Name;
}
