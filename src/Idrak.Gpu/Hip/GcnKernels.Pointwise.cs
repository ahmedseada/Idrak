// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Hip;

// HIP's element-wise kernels in GCN (plan 35, G3a).
internal static partial class GcnKernels
{
    /// <summary>The element-wise kernels, in the order they join <see cref="All"/>.</summary>
    internal static IReadOnlyList<GcnKernelSource> Pointwise { get; } = [];
}
