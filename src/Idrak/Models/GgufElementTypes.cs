// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

// ggml's block formats as element types (plan 37, T0): each block type of the GGUF types (GgufTypes) registered in
// ElementTypes under its lower-case name (q4_0, q4_k, iq2_xxs, mxfp4, ...), its bits a value from its block, its scales
// where its layout has a plain description, and its widening to float32 through the GGUF type's own dequantizer (read at
// each call, so an app's type registered over the library's is the one used). Per-element GGUF types (F32, F16, BF16)
// are the built-in types already.
internal static class GgufElementTypes
{
    // The scales of the formats whose layout is a plain per-block (or per-sub-block) scale, as ggml lays them out.
    private static readonly Dictionary<string, BlockScales> Scales = new(StringComparer.OrdinalIgnoreCase)
    {
        ["Q4_0"] = new("float16", 1, false, "one scale d, then 32 4-bit values"),
        ["Q4_1"] = new("float16", 1, true, "a scale d and a minimum m, then 32 4-bit values"),
        ["Q5_0"] = new("float16", 1, false, "one scale d, the 32 fifth bits, then 32 4-bit values"),
        ["Q5_1"] = new("float16", 1, true, "a scale d and a minimum m, the 32 fifth bits, then 32 4-bit values"),
        ["Q8_0"] = new("float16", 1, false, "one scale d, then 32 8-bit values"),
        ["Q2_K"] = new("4-bit codes of float16 super-scales", 16, true, "16 sub-blocks of 16, each a 4-bit scale and minimum, scaled by d and dmin"),
        ["Q3_K"] = new("6-bit codes of a float16 super-scale", 16, false, "16 sub-blocks of 16, each a 6-bit scale, scaled by d"),
        ["Q4_K"] = new("6-bit codes of float16 super-scales", 8, true, "8 sub-blocks of 32, each a 6-bit scale and minimum, scaled by d and dmin"),
        ["Q5_K"] = new("6-bit codes of float16 super-scales", 8, true, "8 sub-blocks of 32, each a 6-bit scale and minimum, scaled by d and dmin"),
        ["Q6_K"] = new("8-bit codes of a float16 super-scale", 16, false, "16 sub-blocks of 16, each an 8-bit scale, scaled by d"),
        ["MXFP4"] = new("8-bit exponent (E8M0)", 1, false, "one shared power of two, then 32 FP4 (E2M1) values"),
    };

    public static void RegisterAll()
    {
        foreach (int id in GgufTypes.Ids)
        {
            if (GgufTypes.Find(id) is not { BlockValues: > 1 } type)
            {
                continue;                                                   // F32, F16, BF16: the built-in types
            }

            int typeId = id;
            ElementTypes.Register(new ElementType
            {
                Name = type.Name.ToLowerInvariant(),
                Summary = string.Create(CultureInfo.InvariantCulture, $"ggml {type.Name}: blocks of {type.BlockValues} values in {type.BlockBytes} bytes (GGUF type {id})"),
                BlockSize = type.BlockValues,
                BlockBits = type.BlockBytes * 8,
                IsFloat = type.Name.StartsWith("MXFP", StringComparison.OrdinalIgnoreCase),
                IsSigned = true,
                Scales = Scales.GetValueOrDefault(type.Name),
                ToFloat32 = (source, destination) => GgufTypes.Get(typeId).Dequantize(source, destination),
            });
        }
    }
}
