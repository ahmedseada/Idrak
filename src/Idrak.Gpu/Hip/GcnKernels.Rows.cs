// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Hip;

// HIP's row kernels in GCN: norms, softmax and their kin (plan 35, G3b).
internal static partial class GcnKernels
{
    /// <summary>The row kernels, in the order they join <see cref="All"/>.</summary>
    internal static IReadOnlyList<GcnKernelSource> Rows { get; } = [];
}
