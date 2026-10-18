// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Cluster.Diagnostics;

/// <summary>
/// The "cluster" profiler probe (plan 23): over a run of a simulated cluster, the network time the virtual clock charged,
/// per rank across its collectives (the slowest rank's is the step's predicted network time) and per link, and with the
/// training steps the run made (rank 0's gradient synchronisations or pipeline steps) the step as measured here beside
/// the step predicted for the topology's links: the measured step with rank 0's network wall time replaced by the
/// slowest rank's time on the virtual clock. It reads the network telemetry of the "sim" transport, so it records
/// nothing for a workload that runs no simulated cluster.
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
        private readonly SortedDictionary<int, (long Count, double Seconds, double Wall)> _ranks = [];
        private long _steps;

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
                var (count, seconds, wall) = _ranks.GetValueOrDefault(e.Rank);
                _ranks[e.Rank] = (count + 1, seconds + e.Simulated.TotalSeconds, wall + e.Wall.TotalSeconds);
            }
        }

        public void OnGradientSync(in GradientsSynced e)
        {
            if (e.Rank == 0)
            {
                Interlocked.Increment(ref _steps);
            }
        }

        public void OnPipelineStep(in PipelineStepped e)
        {
            if (e.Rank == 0)
            {
                Interlocked.Increment(ref _steps);
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
                    if (_steps > 0 && run.Value("wall") is { } wall)
                    {
                        double here = _ranks.TryGetValue(0, out var first) ? first.Wall * 1e3 : 0;
                        run.Record("steps", _steps, "steps");
                        run.Record("measured step", wall / _steps, "ms");
                        run.Record("predicted step", Math.Max(0, wall - here + (slowest.Value.Seconds * 1e3)) / _steps, "ms");
                    }
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
