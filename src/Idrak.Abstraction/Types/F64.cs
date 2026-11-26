// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Types;

/// <summary>IEEE 754 binary64, read and written without narrowing.</summary>
public sealed class F64 : IElementType
{
    private F64()
    {
    }

    /// <summary>The type.</summary>
    public static F64 Type { get; } = new();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(1, 8, NumberType.Floating(11, 52), [new CodePlane(0, 64, 1)]);

    /// <inheritdoc />
    public string Name => "float64";

    /// <inheritdoc />
    public override string ToString() => Name;
}
