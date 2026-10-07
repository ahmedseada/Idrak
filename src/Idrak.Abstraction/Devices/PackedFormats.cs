// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.


namespace Idrak.Abstraction.Devices;

/// <summary>How each <see cref="PackedFormat"/> lays out its words, for the backends' packed kernels.</summary>
public static class PackedFormats
{
    /// <summary>Weights per 32-bit word: four int8, eight int4, two bfloat16.</summary>
    public static int ValuesPerWord(this PackedFormat format) => format switch
    {
        PackedFormat.Int8 => 4,
        PackedFormat.Int4 => 8,
        _ => 2,
    };

    /// <summary>Row multiple a split of k must start on (int4's scales cover groups of 64 rows in a split).</summary>
    public static int SplitAlignment(this PackedFormat format) => format == PackedFormat.Int4 ? 64 : 1;

    /// <summary>The format's part of kernel and profile names: "int8", "int4", "bf16".</summary>
    public static string KernelName(this PackedFormat format) => format switch
    {
        PackedFormat.Int8 => "int8",
        PackedFormat.Int4 => "int4",
        _ => "bf16",
    };
}
