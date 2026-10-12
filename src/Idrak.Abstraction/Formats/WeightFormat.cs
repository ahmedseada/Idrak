// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Formats;

/// <summary>How <c>Idrak.ModuleFiles.Save</c> stores floating-point values in a weights file.</summary>
public enum WeightFormat
{
    /// <summary>32-bit floats: exact (4 bytes per value).</summary>
    Float32,

    /// <summary>IEEE half precision (2 bytes per value): about 3 significant digits, range ±65504.</summary>
    Float16,

    /// <summary>bfloat16 (2 bytes per value): the range of float32 with about 2–3 significant digits.</summary>
    BFloat16,
}
