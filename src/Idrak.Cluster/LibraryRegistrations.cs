// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Cluster.Diagnostics;

namespace Idrak.Cluster;

/// <summary>
/// What this assembly adds to the registries of Idrak.Abstraction (plan 23): the "sim" devices, the "sim" transport, the
/// built-in link profiles, the "none" failure model and the "cluster" profiler probe. <c>LibraryDefaults.Ensure</c> calls
/// <see cref="RegisterFor"/> once per registry, before that registry is first used, when the application ships it.
/// </summary>
internal static class LibraryRegistrations
{
    private static readonly Dictionary<Type, Action> ByRegistry = new()
    {
        [typeof(DeviceProviders)] = () => DeviceProviders.Register(new SimulatedDevices.Provider()),
        [typeof(Transports)] = () => Transports.Register(new SimTransportProvider()),
        [typeof(LinkProfiles)] = () =>
        {
            foreach (var profile in BuiltInLinks.All())
            {
                LinkProfiles.Register(profile);
            }
        },
        [typeof(FailureModels)] = () => FailureModels.Register(new NoFailures()),
        [typeof(ProfileProbes)] = () => ProfileProbes.Register(new ClusterProbe()),
    };

    private static void RegisterFor(Type registry)
    {
        if (ByRegistry.TryGetValue(registry, out var register))
        {
            register();
        }
    }
}
