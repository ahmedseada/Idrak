// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Types;

/// <summary>An unsigned 64-bit integer (little-endian), read and written without narrowing.</summary>
public sealed class U64 : IElementType
{
    private U64()
    {
    }

    /// <summary>The type.</summary>
    public static U64 Type { get; } = new();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(1, 8, NumberType.Integer(64, signed: false), [new CodePlane(0, 64, 1)]);

    /// <inheritdoc />
    public string Name => "uint64";

    /// <inheritdoc />
    public override string ToString() => Name;
}
