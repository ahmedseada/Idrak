// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Types;

/// <summary>A signed 64-bit integer (two's complement, little-endian), read and written without narrowing.</summary>
public sealed class I64 : IElementType
{
    private I64()
    {
    }

    /// <summary>The type.</summary>
    public static IElementType Type { get; } = new I64();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(1, 8, NumberType.Integer(64, signed: true), [new CodePlane(0, 64, 1)]);

    /// <inheritdoc />
    public string Name => "int64";

    /// <inheritdoc />
    public override string ToString() => Name;
}
