// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Networking.Collectives;

internal sealed class NaiveProvider : ICollectivesProvider
{
    public string Name => "naive";

    public string Summary => "every rank to rank 0, combined in rank order and sent back: the reference the others agree with";

    public ICollectives Create(ITransport transport, Device device, ClusterContext context) => new NaiveCollectives(transport, device, context);
}

internal sealed class RingProvider : ICollectivesProvider
{
    public string Name => "ring";

    public string Summary => "reduce-scatter then all-gather around the ranks: each sends and receives 2(N-1)/N of the values";

    public ICollectives Create(ITransport transport, Device device, ClusterContext context) => new RingCollectives(transport, device, context);
}

internal sealed class HierarchicalProvider : ICollectivesProvider
{
    public string Name => "hierarchical";

    public string Summary => "inside each node over its link, then a ring across the nodes on the shards (one rank per node), then inside again: the slow links carry only the shards";

    public ICollectives Create(ITransport transport, Device device, ClusterContext context) => new HierarchicalCollectives(transport, device, context);
}
