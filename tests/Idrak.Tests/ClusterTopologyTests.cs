// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Cluster;

// Plan 23, phase C3: two-level topologies of nodes of a few ranks each.
internal abstract partial class DesktopTests : Tests
{
    // A topology of nodes of `sizes` ranks, every rank on `device`, joined by `inter` between nodes.
    private static ClusterTopology NodesOf(IReadOnlyList<int> sizes, string device, string inter = "ethernet-10g") =>
        ClusterTopology.Create([.. sizes.Select((n, i) => new ClusterNodeSpec($"node{i}", null, [.. Enumerable.Repeat(new ClusterDeviceSpec(device, null), n)]))],
            null, sizes.Count > 1 ? inter : null);
}
