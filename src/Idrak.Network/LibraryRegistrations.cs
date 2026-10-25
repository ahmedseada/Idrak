// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Networking.Collectives;
using Idrak.Networking.Diagnostics;
using Idrak.Networking.Selection;
using Idrak.Networking.Transports;
using Idrak.Networking.Training;

[assembly: LibraryRegistrations(typeof(Idrak.Networking.LibraryRegistrations))]

namespace Idrak.Networking;

/// <summary>
/// What this assembly adds to the registries of Idrak.Abstraction (plan 23): the "naive", "ring" and "hierarchical" collectives, the
/// "fastest" transport selector, the "all-reduce", "local-sgd" and "sharded" gradient synchronisations, the "gpipe" pipeline
/// schedule, the "all-reduce" stop agreement, the "control-file" run steering, the "network" profiler probe and the "cluster" setting advisor
/// (the gradient compressions are its own registry, <see cref="Abstractions.GradientCompressions"/>). <c>LibraryDefaults.Ensure</c> calls
/// <see cref="RegisterFor"/> once per registry, before that registry is first used, when the application ships it.
/// </summary>
internal sealed class LibraryRegistrations : ILibraryRegistrations
{
    private static readonly Dictionary<Type, Action> ByRegistry = new()
    {
        [typeof(Abstraction.Cluster.Collectives)] = () =>
        {
            Abstraction.Cluster.Collectives.Register(new NaiveProvider());
            Abstraction.Cluster.Collectives.Register(new RingProvider());
            Abstraction.Cluster.Collectives.Register(new HierarchicalProvider());
        },
        [typeof(Abstraction.Cluster.Transports)] = () =>
        {
            Abstraction.Cluster.Transports.Register(new SharedMemoryProvider());
            Abstraction.Cluster.Transports.Register(new PciHostProvider());
            Abstraction.Cluster.Transports.Register(new NvlinkProvider());
            Abstraction.Cluster.Transports.Register(new NvlinkProvider("p2p"));
            Abstraction.Cluster.Transports.Register(new DirectLinkProvider());
            Abstraction.Cluster.Transports.Register(new TcpProvider());
        },
        [typeof(TransportSelectors)] = () => TransportSelectors.Register(new FastestSelector()),
        [typeof(GradientSyncs)] = () =>
        {
            GradientSyncs.Register(new AllReduceSyncProvider());
            GradientSyncs.Register(new LocalSgdProvider());
            GradientSyncs.Register(new ShardedSyncProvider());
        },
        [typeof(PipelineSchedules)] = () => PipelineSchedules.Register(new GPipeSchedule()),
        [typeof(StopAgreements)] = () => StopAgreements.Register(new AllReduceStopAgreementProvider()),   // Plan 27, R7: the ranks agree on a stop
        [typeof(RunSteerings)] = () => RunSteerings.Register(new ControlFileSteeringProvider()),          // Plan 27, R11: a trainer's boundaries steered by the control file
        [typeof(ProfileProbes)] = () => ProfileProbes.Register(new NetworkProbe()),
        [typeof(SettingAdvisors)] = () => SettingAdvisors.Register(new ClusterSettingAdvisor()),   // decision 4: how a run sends less
    };

    public void RegisterFor(Type registry)
    {
        if (ByRegistry.TryGetValue(registry, out var register))
        {
            register();
        }
    }
}
