// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu;

// The element-type conversions the GPU backends run on the device (ConvertElements), one kernel name per pair on CUDA,
// Vulkan and HIP alike: float32 to and from int32 (the index conversions), bfloat16 and float16, each bit for bit the
// CPU's (ElementConversions), NaNs included. Any other pair goes through the host's conversion (Backend.ConvertElements'
// fallback). The value types each backend reports for ConvertElements are these.
internal static class DeviceConversions
{
    /// <summary>The types the device conversions read and write (BackendCapabilities.ValueTypes of ConvertElements).</summary>
    internal static IReadOnlyList<ElementType> Types { get; } = [ElementType.Float32, ElementType.Int32, ElementType.BFloat16, ElementType.Float16];

    /// <summary>The kernel converting <paramref name="from"/> to <paramref name="to"/> on the device, or null for a pair the host converts.</summary>
    internal static string? Kernel(ElementType from, ElementType to) =>
        ReferenceEquals(from, ElementType.Float32) ? (ReferenceEquals(to, ElementType.Int32) ? "convert_f32_i32"
            : ReferenceEquals(to, ElementType.BFloat16) ? "convert_f32_bf16" : ReferenceEquals(to, ElementType.Float16) ? "convert_f32_f16" : null)
        : ReferenceEquals(to, ElementType.Float32) ? (ReferenceEquals(from, ElementType.Int32) ? "convert_i32_f32"
            : ReferenceEquals(from, ElementType.BFloat16) ? "convert_bf16_f32" : ReferenceEquals(from, ElementType.Float16) ? "convert_f16_f32" : null)
        : null;

    /// <summary>Whether <paramref name="kernel"/> narrows: a work-item per output word of two 16-bit values (else one per element).</summary>
    internal static bool Narrows(string kernel) => kernel is "convert_f32_bf16" or "convert_f32_f16";
}
