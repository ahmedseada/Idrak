// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;

namespace Idrak.Cluster;

/// <summary>One rank of a simulated cluster as its work sees it.</summary>
/// <param name="Rank">Its rank.</param>
/// <param name="Device">Its device, carved from the physical one its topology entry names, with its cap.</param>
/// <param name="Transport">Its end of the "sim" transport.</param>
/// <param name="Collectives">Its collectives over that transport.</param>
/// <param name="Context">The topology, its rank and the options.</param>
public sealed record ClusterRank(int Rank, Device Device, ITransport Transport, ICollectives Collectives, ClusterContext Context);

/// <summary>What a simulated run gave: each rank's failure, if any, and its time on the virtual clock, and the wall time.</summary>
/// <param name="Errors">Each rank's exception, or null when it finished.</param>
/// <param name="Simulated">Each rank's network time on the virtual clock.</param>
/// <param name="Wall">The time the run took here.</param>
public sealed record ClusterRunResult(IReadOnlyList<Exception?> Errors, IReadOnlyList<TimeSpan> Simulated, TimeSpan Wall)
{
    /// <summary>Whether every rank finished.</summary>
    public bool Succeeded => Errors.All(e => e is null);

    /// <summary>Throws when a rank failed, naming every rank that did.</summary>
    /// <exception cref="AggregateException">The ranks' exceptions.</exception>
    public void ThrowIfFailed()
    {
        if (!Succeeded)
        {
            throw new AggregateException("Ranks failed: " + string.Join("; ", Errors.Select((e, r) => e is null ? null : $"rank {r}: {e.Message}").Where(m => m is not null)),
                Errors.Where(e => e is not null).Select(e => e!));
        }
    }
}

/// <summary>
/// Runs a cluster inside this process (plan 23, the in-process mode): every rank a thread of its own on a device carved
/// from the physical device its topology entry names (<see cref="SimulatedDevices.Carve"/>, the entry's memory as its cap),
/// joined by the "sim" transport with the algorithm <see cref="ClusterOptions.Collective"/> names ("ring" when null).
/// Deterministic for a seed: the virtual clock counts network time only.
/// </summary>
public static class InProcessCluster
{
    /// <summary>Runs <paramref name="work"/> on every rank of <paramref name="topology"/> and waits for them all.</summary>
    /// <exception cref="NotSupportedException">The collective algorithm or the "sim" transport is not registered.</exception>
    public static ClusterRunResult Run(IClusterTopology topology, ClusterOptions? options, Action<ClusterRank> work)
    {
        ArgumentNullException.ThrowIfNull(topology);
        ArgumentNullException.ThrowIfNull(work);
        options = (options ?? new ClusterOptions()) with
        {
            Coordinator = "sim://" + Guid.NewGuid().ToString("N", CultureInfo.InvariantCulture),   // this run's own rendezvous
            Simulation = (options?.Simulation ?? new SimulationOptions()) with { Mode = SimulationMode.InProcess },
        };
        var transports = Transports.Get("sim");
        var collectives = Collectives.Get(options.Collective ?? "ring");
        int world = topology.WorldSize;
        var devices = new Device[world];
        for (int r = 0; r < world; r++)
        {
            var (node, local) = topology.Place(r);
            var spec = topology.Nodes[node].Devices[local];
            devices[r] = SimulatedDevices.Carve(Device.Parse(spec.Device), spec.MemoryBytes, r);
        }

        var errors = new Exception?[world];
        var simulated = new TimeSpan[world];
        var threads = new Thread[world];
        long start = Stopwatch.GetTimestamp();
        for (int r = 0; r < world; r++)
        {
            int rank = r;
            threads[r] = new Thread(() =>
            {
                var context = new ClusterContext(topology, rank, options);
                ITransport? transport = null;
                ICollectives? group = null;
                try
                {
                    transport = transports.Connect(context);
                    group = collectives.Create(transport, devices[rank], context);
                    State(rank, RankState.Joined, null);
                    work(new ClusterRank(rank, devices[rank], transport, group, context));
                    simulated[rank] = transport.SimulatedTime;
                    State(rank, RankState.Finished, null);
                }
                catch (Exception e)
                {
                    errors[rank] = e;
                    simulated[rank] = transport?.SimulatedTime ?? TimeSpan.Zero;
                    State(rank, RankState.Dropped, e.Message);
                }
                finally
                {
                    group?.Dispose();
                    transport?.Dispose();
                }
            })
            { IsBackground = true, Name = $"sim rank {rank}" };
        }

        foreach (var thread in threads)
        {
            thread.Start();
        }

        foreach (var thread in threads)
        {
            thread.Join();
        }

        return new ClusterRunResult(errors, simulated, Stopwatch.GetElapsedTime(start));
    }

    private static void State(int rank, RankState state, string? detail)
    {
        if (Telemetry.IsEnabled(TelemetryLevel.Network))
        {
            Telemetry.RankState(new RankStateChanged(rank, state, -1, detail));
        }
    }
}
