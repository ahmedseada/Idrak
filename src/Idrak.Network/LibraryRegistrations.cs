// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Networking.Collectives;
using Idrak.Networking.Diagnostics;
using Idrak.Networking.Selection;
using Idrak.Networking.Training;

namespace Idrak.Networking;

/// <summary>
/// What this assembly adds to the registries of Idrak.Abstraction (plan 23): the "naive", "ring" and "hierarchical" collectives, the
/// "fastest" transport selector, the "all-reduce" and "local-sgd" gradient synchronisations and the "network" profiler probe
/// (the gradient compressions are its own registry, <see cref="Abstractions.GradientCompressions"/>). <c>LibraryDefaults.Ensure</c> calls
/// <see cref="RegisterFor"/> once per registry, before that registry is first used, when the application ships it.
/// </summary>
internal static class LibraryRegistrations
{
    private static readonly Dictionary<Type, Action> ByRegistry = new()
    {
        [typeof(Abstraction.Cluster.Collectives)] = () =>
        {
            Abstraction.Cluster.Collectives.Register(new NaiveProvider());
            Abstraction.Cluster.Collectives.Register(new RingProvider());
            Abstraction.Cluster.Collectives.Register(new HierarchicalProvider());
        },
        [typeof(TransportSelectors)] = () => TransportSelectors.Register(new FastestSelector()),
        [typeof(GradientSyncs)] = () =>
        {
            GradientSyncs.Register(new AllReduceSyncProvider());
            GradientSyncs.Register(new LocalSgdProvider());
        },
        [typeof(ProfileProbes)] = () => ProfileProbes.Register(new NetworkProbe()),
    };

    private static void RegisterFor(Type registry)
    {
        if (ByRegistry.TryGetValue(registry, out var register))
        {
            register();
        }
    }
}
