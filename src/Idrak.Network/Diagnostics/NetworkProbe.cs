// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Networking.Diagnostics;

/// <summary>
/// The "network" profiler probe (plan 23): over a run, the collectives each rank made (their wall time, the bytes and the
/// time blocked on peers, and the rank waited on most), the transfers per transport and pair, each rank's gradient
/// synchronisations (the buckets of a step, how many went during the backward pass, the network and exposed time) and
/// each pipeline stage's steps (its wall, blocked time and bubble), so a profile shows network time next to kernel time.
/// Records nothing for a workload that sends nothing.
/// </summary>
internal sealed class NetworkProbe : IProfileProbe
{
    public string Name => "network";

    public string Summary => "collectives and transfers: wall time, bytes, time blocked on peers, the pairs that cost most";

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
        private readonly SortedDictionary<(string Transport, int From, int To), (long Count, long Bytes, double Seconds)> _pairs = [];
        private readonly SortedDictionary<(string Collective, string Operation), (long Count, long Bytes, double Seconds, double Blocked)> _collectives = [];
        private readonly Dictionary<int, int> _slowest = [];
        private readonly SortedDictionary<int, (long Steps, long Buckets, long Overlapped, long BucketBytes, double Network, double Exposed, double Backward)> _syncs = [];
        private readonly SortedDictionary<(int Rank, int Stage), (long Steps, double Wall, double Blocked, double Bubble)> _stages = [];

        public TelemetryLevel Levels => TelemetryLevel.Network;

        public void OnTransfer(in TransferCompleted e)
        {
            lock (_gate)
            {
                var (count, bytes, seconds) = _pairs.GetValueOrDefault((e.Transport, e.From, e.To));
                _pairs[(e.Transport, e.From, e.To)] = (count + 1, bytes + e.Bytes, seconds + e.Wall.TotalSeconds);
            }
        }

        public void OnCollective(in CollectiveCompleted e)
        {
            lock (_gate)
            {
                var (count, bytes, seconds, blocked) = _collectives.GetValueOrDefault((e.Collective, e.Operation));
                _collectives[(e.Collective, e.Operation)] = (count + 1, bytes + e.BytesSent, seconds + e.Wall.TotalSeconds, blocked + e.Blocked.TotalSeconds);
                if (e.SlowestRank != e.Rank)
                {
                    _slowest[e.SlowestRank] = _slowest.GetValueOrDefault(e.SlowestRank) + 1;
                }
            }
        }

        public void OnGradientSync(in GradientsSynced e)
        {
            lock (_gate)
            {
                var (steps, buckets, overlapped, bytes, network, exposed, backward) = _syncs.GetValueOrDefault(e.Rank);
                _syncs[e.Rank] = (steps + 1, buckets + e.Buckets, overlapped + e.Overlapped, Math.Max(bytes, e.BucketBytes), network + e.Network.TotalSeconds,
                    exposed + e.Exposed.TotalSeconds, backward + e.Backward.TotalSeconds);
            }
        }

        public void OnPipelineStep(in PipelineStepped e)
        {
            lock (_gate)
            {
                var (steps, wall, blocked, bubble) = _stages.GetValueOrDefault((e.Rank, e.Stage));
                _stages[(e.Rank, e.Stage)] = (steps + 1, wall + e.Wall.TotalSeconds, blocked + e.Blocked.TotalSeconds, bubble + e.Bubble);
            }
        }

        public void Record(ProfileRun run)
        {
            lock (_gate)
            {
                if (_pairs.Count == 0 && _collectives.Count == 0 && _syncs.Count == 0 && _stages.Count == 0)
                {
                    return;
                }

                var inv = CultureInfo.InvariantCulture;
                run.Record("collectives", _collectives.Values.Sum(c => c.Count), "calls");
                run.Record("network wall", _collectives.Values.Sum(c => c.Seconds) * 1e3, "ms");
                run.Record("network blocked", _collectives.Values.Sum(c => c.Blocked) * 1e3, "ms");
                run.Record("network sent", _collectives.Values.Sum(c => c.Bytes) / 1e6, "MB");
                if (_slowest.Count > 0)
                {
                    run.Record("most waited-on rank", _slowest.MaxBy(s => s.Value).Key, "rank");
                }

                run.Add(new ProfileTable("collectives", ["collective", "operation", "calls", "bytes sent", "wall ms", "blocked ms"],
                    [.. _collectives.Select(c => (IReadOnlyList<string>)[c.Key.Collective, c.Key.Operation, c.Value.Count.ToString(inv), c.Value.Bytes.ToString(inv),
                        (c.Value.Seconds * 1e3).ToString("F3", inv), (c.Value.Blocked * 1e3).ToString("F3", inv)])]));
                run.Add(new ProfileTable("transfers", ["transport/pair", "transfers", "bytes", "wall ms"],
                    [.. _pairs.OrderByDescending(p => p.Value.Seconds).Select(p => (IReadOnlyList<string>)[$"{p.Key.Transport}/{p.Key.From}->{p.Key.To}",
                        p.Value.Count.ToString(inv), p.Value.Bytes.ToString(inv), (p.Value.Seconds * 1e3).ToString("F3", inv)])]));
                if (_syncs.Count > 0)
                {
                    // A step's buckets: how many, how many went while the backward pass ran, and what the step waited after it.
                    var first = _syncs.First().Value;
                    run.Record("gradient steps", first.Steps, "steps");
                    run.Record("buckets per step", first.Buckets / (double)Math.Max(1, first.Steps), "buckets");
                    run.Record("exposed network per step", first.Exposed / Math.Max(1, first.Steps) * 1e3, "ms");
                    run.Add(new ProfileTable("gradient buckets", ["rank", "steps", "buckets/step", "during backward", "bucket bytes", "network ms/step", "exposed ms/step"],
                        [.. _syncs.Select(p => (IReadOnlyList<string>)[p.Key.ToString(inv), p.Value.Steps.ToString(inv),
                            (p.Value.Buckets / (double)Math.Max(1, p.Value.Steps)).ToString("F1", inv), (p.Value.Overlapped / (double)Math.Max(1, p.Value.Buckets)).ToString("P0", inv),
                            p.Value.BucketBytes.ToString(inv), (p.Value.Network / Math.Max(1, p.Value.Steps) * 1e3).ToString("F3", inv),
                            (p.Value.Exposed / Math.Max(1, p.Value.Steps) * 1e3).ToString("F3", inv)])]));
                }

                if (_stages.Count > 0)
                {
                    run.Add(new ProfileTable("pipeline stages", ["rank", "stage", "steps", "wall ms/step", "blocked ms/step", "bubble"],
                        [.. _stages.Select(p => (IReadOnlyList<string>)[p.Key.Rank.ToString(inv), p.Key.Stage.ToString(inv), p.Value.Steps.ToString(inv),
                            (p.Value.Wall / Math.Max(1, p.Value.Steps) * 1e3).ToString("F3", inv), (p.Value.Blocked / Math.Max(1, p.Value.Steps) * 1e3).ToString("F3", inv),
                            (p.Value.Bubble / Math.Max(1, p.Value.Steps)).ToString("P0", inv)])]));
                }
            }
        }
    }
}
