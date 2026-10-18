// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Cluster;

/// <summary>
/// Replays a simulated run (plan 23, phase C3): a fault's telemetry record and the event log name its seed and step, and a
/// rerun of the same work with that seed through that step draws the same faults and charges the same virtual clock, bit
/// for bit, because every draw comes from (seed, step) and every charge from the message's identity, never from thread
/// timing. Every rank sits out the steps after the one asked (<see cref="SimulationOptions.ReplayStep"/>).
/// </summary>
public static class ClusterReplay
{
    /// <summary>
    /// The faults the failure model <paramref name="failures"/> (a <see cref="FailureModels"/> name) draws on
    /// <paramref name="topology"/> from <paramref name="seed"/> for steps 0 to <paramref name="throughStep"/>, without
    /// running anything: the schedule a run of the seed injects.
    /// </summary>
    /// <exception cref="NotSupportedException">The model is not registered.</exception>
    public static IReadOnlyList<Fault> Schedule(IClusterTopology topology, string failures, int seed, long throughStep)
    {
        ArgumentNullException.ThrowIfNull(topology);
        ArgumentOutOfRangeException.ThrowIfNegative(throughStep);
        return FaultInjector.Schedule(FailureModels.Get(failures), topology, seed, throughStep);
    }

    /// <summary>
    /// Reruns <paramref name="work"/> on <paramref name="topology"/> with <paramref name="seed"/> through step
    /// <paramref name="step"/> (<see cref="InProcessCluster.Run(IClusterTopology, ClusterOptions, Action{ClusterRank})"/> with the seed and <see cref="SimulationOptions.ReplayStep"/>
    /// set): the result's <see cref="ClusterRunResult.Log"/> holds the faults drawn and each rank's clock at every step up to
    /// it, equal to the original run's.
    /// </summary>
    public static ClusterRunResult Run(IClusterTopology topology, ClusterOptions? options, int seed, long step, Action<ClusterRank> work)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(step);
        options ??= new ClusterOptions();
        return InProcessCluster.Run(topology, options with { Simulation = options.Simulation with { Seed = seed, ReplayStep = step } }, work);
    }

    /// <summary>
    /// Replays a run steered while it went (plan 23, phase C9: faults injected from the dashboard): as the run above, with the live faults
    /// <paramref name="steered"/>'s event log recorded injected again at their steps, so the replay's faults, clocks and
    /// training equal the steered run's through <paramref name="step"/>, bit for bit. (A pause changes no result, so it needs
    /// no replaying; a stop ends the run where a replay's step would.)
    /// </summary>
    public static ClusterRunResult Run(IClusterTopology topology, ClusterOptions? options, int seed, long step, Action<ClusterRank> work, EventLog steered)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(step);
        ArgumentNullException.ThrowIfNull(steered);
        options ??= new ClusterOptions();
        return InProcessCluster.Run(topology, options with { Simulation = options.Simulation with { Seed = seed, ReplayStep = step } }, work,
            new ClusterRunControl(steered.LiveFaults.Where(f => f.Step <= step)));
    }
}
