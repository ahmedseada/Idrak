// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Cluster.Diagnostics;

/// <summary>
/// The "cluster" profiler probe (plan 23): over a run of a simulated cluster, the network time the virtual clock charged,
/// per rank across its collectives (the slowest rank's is the step's predicted network time) and per link. It reads the
/// network telemetry of the "sim" transport, so it records nothing for a workload that runs no simulated cluster.
/// </summary>
internal sealed class ClusterProbe : IProfileProbe
{
    public string Name => "cluster";

    public string Summary => "a simulated cluster's network time on the virtual clock: per rank over its collectives, and per link";

    public bool Intrusive => false;

    public bool Supports(Device device) => true;

    public Action Begin(ProfileRun run)
    {
        var hook = new Hook();
        var subscription = Telemetry.Subscribe(hook);
        return () =>
        {
            subscription.Dispose();
            hook.Record(run);
        };
    }

    private sealed class Hook : ITelemetryHook
    {
        private readonly Lock _gate = new();
        private readonly SortedDictionary<(int From, int To), (long Count, long Bytes, double Seconds)> _links = [];
        private readonly SortedDictionary<int, (long Count, double Seconds)> _ranks = [];

        public TelemetryLevel Levels => TelemetryLevel.Network;

        public void OnTransfer(in TransferCompleted e)
        {
            if (e.Transport != "sim")
            {
                return;
            }

            lock (_gate)
            {
                var (count, bytes, seconds) = _links.GetValueOrDefault((e.From, e.To));
                _links[(e.From, e.To)] = (count + 1, bytes + e.Bytes, seconds + e.Simulated.TotalSeconds);
            }
        }

        public void OnCollective(in CollectiveCompleted e)
        {
            if (e.Simulated <= TimeSpan.Zero)
            {
                return;
            }

            lock (_gate)
            {
                var (count, seconds) = _ranks.GetValueOrDefault(e.Rank);
                _ranks[e.Rank] = (count + 1, seconds + e.Simulated.TotalSeconds);
            }
        }

        public void Record(ProfileRun run)
        {
            lock (_gate)
            {
                if (_links.Count == 0 && _ranks.Count == 0)
                {
                    return;
                }

                if (_ranks.Count > 0)
                {
                    var slowest = _ranks.MaxBy(r => r.Value.Seconds);
                    run.Record("simulated network", slowest.Value.Seconds * 1e3, "ms");
                    run.Record("simulated slowest rank", slowest.Key, "rank");
                }

                run.Record("simulated transfers", _links.Values.Sum(l => l.Count), "transfers");
                var inv = CultureInfo.InvariantCulture;
                run.Add(new ProfileTable("simulated links", ["from", "to", "transfers", "bytes", "simulated ms"],
                    [.. _links.OrderByDescending(l => l.Value.Seconds).Select(l => (IReadOnlyList<string>)
                        [l.Key.From.ToString(inv), l.Key.To.ToString(inv), l.Value.Count.ToString(inv), l.Value.Bytes.ToString(inv), (l.Value.Seconds * 1e3).ToString("F3", inv)])]));
            }
        }
    }
}
