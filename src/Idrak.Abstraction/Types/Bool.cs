// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Types;

/// <summary>A truth value a byte: its lowest bit, 0 or 1 (a value written as the nearer of the two, a tie to 0).</summary>
public sealed class Bool : IElementType
{
    private Bool()
    {
    }

    /// <summary>The type.</summary>
    public static Bool Type { get; } = new();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(1, 1, NumberType.Integer(1, signed: false), [new CodePlane(0, 1, 1, Places: [new FieldPlace(0, 0)])]);

    /// <inheritdoc />
    public string Name => "bool";

    /// <inheritdoc />
    public override string ToString() => Name;
}
