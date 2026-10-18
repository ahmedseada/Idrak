// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Networking.Collectives;
using Idrak.Networking.Diagnostics;

namespace Idrak.Networking;

/// <summary>
/// What this assembly adds to the registries of Idrak.Abstraction (plan 23): the "naive" and "ring" collectives, the
/// "fastest" transport selector and the "network" profiler probe. <c>LibraryDefaults.Ensure</c> calls
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
        },
        [typeof(TransportSelectors)] = () => TransportSelectors.Register(new FastestSelector()),
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
