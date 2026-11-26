// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Types;

/// <summary>OCP's MXFP4 as GGUF lays it out: an E8M0 scale byte a block of 32, then E2M1 codes, value i in byte i's low half and i + 16 in its high half.</summary>
public sealed class MXFP4 : IElementType
{
    private MXFP4()
    {
    }

    /// <summary>The type.</summary>
    public static MXFP4 Type { get; } = new();

    /// <inheritdoc />
    public TypeDescription Description { get; } = new(32, 17, NumberType.Floating(2, 1, bias: 1, specials: FloatSpecials.None), [new CodePlane(1, 4, 32)], Scales: [new BlockScale(NumberType.Floating(8, 0, signed: false, bias: 127, subnormals: false, specials: FloatSpecials.AllOnesNaN), 32, 0)]);

    /// <inheritdoc />
    public string Name => "mxfp4";

    /// <inheritdoc />
    public override string ToString() => Name;
}
