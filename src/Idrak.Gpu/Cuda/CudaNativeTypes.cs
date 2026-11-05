// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Cuda;

/// <summary>
/// The element types the CUDA backend's kernels compute natively on a device (plan 37, T2), from the compute capability
/// the device reports, as the tensor-core loader gates its modules: bfloat16 tensor cores from 8.0, fp8 (e4m3) from 8.9.
/// No float16 product is built (the tensor-core kernels take bfloat16), so float16 is not listed. float32 always.
/// </summary>
internal static class CudaNativeTypes
{
    /// <summary>The types a device of compute capability <paramref name="major"/>.<paramref name="minor"/> computes natively here, best first.</summary>
    public static IReadOnlyList<ElementType> Of(int major, int minor)
    {
        var types = new List<ElementType>(3);
        if (major >= 8)
        {
            types.Add(ElementType.BFloat16);
        }

        if (major * 10 + minor >= 89)
        {
            types.Add(ElementType.Float8E4M3);
        }

        types.Add(ElementType.Float32);
        return types;
    }
}
