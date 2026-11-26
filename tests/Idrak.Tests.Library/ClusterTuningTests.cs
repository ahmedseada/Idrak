// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Abstraction.Cluster;
using Idrak.Abstraction.Operations;
using Idrak.Cluster;
using Idrak.Gpu.Vulkan;

// Phase 3: the ranks of a cluster run rank 0's measured kernel choices (ClusterOptions.ShareTuning, IDRAK_CLUSTER_SHARE_TUNING):
// in one process each rank's backend follows rank 0's before the ranks' threads start; between processes rank 0 sends
// what it knows as the ranks join. A rank on another device or driver keeps its own and says so. Every cluster here is
// an in-process one.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] ClusterTuningGroup =
    [
    ];
}
