// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Cuda;

/// <summary>
/// The element types the CUDA backend's kernels compute natively on a device (plan 37, T2): bfloat16 and float16 where
/// the compute capability the device reports reaches the PTX target their tensor-core kernels declare
/// (<see cref="PtxKernels.TensorCoreTarget"/>), fp8 (e4m3) where it reaches the fp8 kernels' target
/// (<see cref="PtxKernels.Float8Target"/>); float32 always. The targets are the kernels' own, never a list of cards.
/// </summary>
internal static class CudaNativeTypes
{
    /// <summary>The types the device computes natively here, best first: bfloat16 and float16 (one module, <paramref name="tensorCores"/>) and fp8 where it reaches their kernels' targets.</summary>
    public static IReadOnlyList<ElementType> Of(bool tensorCores, bool float8)
    {
        var types = new List<ElementType>(4);
        if (tensorCores)
        {
            types.Add(ElementType.BFloat16);
            types.Add(ElementType.Float16);
        }

        if (float8)
        {
            types.Add(ElementType.Float8E4M3);
        }

        types.Add(ElementType.Float32);
        return types;
    }

    /// <summary>What the device reported and what follows from it, for people: its compute capability against each kernel family's declared target.</summary>
    public static string Report(int major, int minor, bool tensorCores, bool float8) =>
        $"compute capability {major}.{minor}: the bfloat16 and float16 tensor-core kernels (PTX {PtxKernels.TensorCoreTarget}) {(tensorCores ? "load" : "need more")}, "
        + $"the fp8 ones (PTX {PtxKernels.Float8Target}) {(float8 ? "load" : "need more")}";
}
