// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Operations;

namespace Idrak.Gpu;

// Products and attention a run asks for in a reduced precision (MixedPrecision: bfloat16, float16, fp8) where the device
// has no kernel of that type for the call run in float32: never in another reduced type in its place (plan 37: no fixed
// fallback type; an fp8 run never takes the bfloat16 kernels), and never silently: each such call counts its operands as
// converted from the asked type to float32 (ElementTypes.CountConversion: ElementTypes.Conversions, by operation, and the
// ElementsConverted telemetry). A shape the reduced kernels do not serve in any precision (too few rows, say) is not
// counted: it runs float32 in every mode. Where both kernels exist and a measurement picked float32 (Vulkan's mixed
// products), nothing is missing and nothing is counted.
internal static class ReducedPrecision
{
    // The type the thread's MixedPrecision asks the products to multiply in, or null in float32 mode (fp8: e4m3, the
    // format of the fp8 products).
    internal static ElementType? Asked => MixedPrecision.Current switch
    {
        MatMulPrecision.BFloat16 => ElementType.BFloat16,
        MatMulPrecision.Float16 => ElementType.Float16,
        MatMulPrecision.Float8 => ElementType.Float8E4M3,
        _ => null,
    };

    // Counts one call of `operation` on `backend` that runs in float32 for want of a kernel in the asked type: its
    // `elements` operand elements (nothing in float32 mode).
    internal static void InFloat32(Backend backend, Operation operation, long elements)
    {
        if (Asked is { } asked)
        {
            ElementTypes.CountConversion(backend, operation, asked, ElementType.Float32, elements);
        }
    }
}
