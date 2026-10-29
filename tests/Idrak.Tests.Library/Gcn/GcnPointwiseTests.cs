// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// HIP's pointwise kernels in GCN (GcnKernels.Pointwise), each run by the GCN interpreter and proven against the CPU (plan 35, G3a).
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] GcnPointwiseGroup =
    [
    ];
}
