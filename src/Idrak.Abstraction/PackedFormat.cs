// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction;

/// <summary>The packed formats a <c>Idrak.Layers.Linear</c> layer's weights can be held in (see <c>Idrak.Abstraction.Generation.PackedWeight</c>).</summary>
public enum PackedFormat
{
    /// <summary>A signed byte per weight with one scale per column (<c>Idrak.Abstraction.Generation.Int8Weight</c>).</summary>
    Int8,

    /// <summary>A signed nibble per weight with one scale per group of 32 rows of a column (<c>Idrak.Abstraction.Generation.Int4Weight</c>).</summary>
    Int4,

    /// <summary>bfloat16 values (<c>Idrak.Abstraction.Generation.BFloat16Weight</c>).</summary>
    BFloat16,
}
