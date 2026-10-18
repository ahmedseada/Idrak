// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Networking.Diagnostics;

/// <summary>
/// The "network" profiler probe (plan 23): over a run, the collectives each rank made (their wall time, the bytes and the
/// time blocked on peers, and the rank waited on most) and the transfers per transport and pair, so a profile shows
/// network time next to kernel time. Records nothing for a workload that sends nothing.
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

        public void Record(ProfileRun run)
        {
            lock (_gate)
            {
                if (_pairs.Count == 0 && _collectives.Count == 0)
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
            }
        }
    }
}
