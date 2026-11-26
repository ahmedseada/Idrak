// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Types;

/// <summary>OCP fp8 E5M2: a binary16's high byte, IEEE's specials.</summary>
public sealed class F8E5M2 : IElementType
{
    private F8E5M2()
    {
    }

    /// <summary>The type.</summary>
    public static IElementType Type { get; } = new F8E5M2();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(1, 1, NumberType.Floating(5, 2), [new CodePlane(0, 8, 1)]);

    /// <inheritdoc />
    public string Name => "float8_e5m2";

    /// <inheritdoc />
    public override string ToString() => Name;
}
