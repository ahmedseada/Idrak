// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Idrak.Cluster;

/// <summary>One measured transfer a calibration fits: the link it crossed, its bytes and the seconds it took.</summary>
/// <param name="Link">The link profile's name, as the topology names the pair's link.</param>
/// <param name="From">The sending rank.</param>
/// <param name="To">The receiving rank.</param>
/// <param name="Bytes">Its bytes.</param>
/// <param name="Seconds">The seconds it took (latency included).</param>
public readonly record struct CalibrationSample(string Link, int From, int To, long Bytes, double Seconds);

/// <summary>
/// A network adapter shared by a machine's ranks, as a bench of concurrent pairs shows it: the pairs between two nodes sent
/// at once moved less than that many times one pair alone.
/// </summary>
/// <param name="Link">The link between the two nodes.</param>
/// <param name="Pairs">The pairs that sent at once.</param>
/// <param name="Alone">One pair's bytes a second alone (the median of the bench's single pairs on that link).</param>
/// <param name="Together">Every pair's bytes a second together.</param>
/// <param name="Shared">Whether the pairs shared one adapter (together below <see cref="Pairs"/> times alone, beyond the tolerance).</param>
public sealed record AdapterShare(string Link, int Pairs, double Alone, double Together, bool Shared);

/// <summary>What a calibration fitted for one link, against the profile in use before it.</summary>
/// <param name="Link">The link profile's name.</param>
/// <param name="BandwidthBefore">The bandwidth before (0: not known).</param>
/// <param name="LatencyBefore">The latency before.</param>
/// <param name="Fitted">The fitted profile, as kept.</param>
public sealed record LinkCalibrationResult(string Link, double BandwidthBefore, TimeSpan LatencyBefore, CalibratedLink Fitted);

/// <summary>
/// Calibrates link profiles against measured transfers (plan 23, phase C8; <c>idrak cluster calibrate</c>): the transfers
/// come from a run's telemetry log (its <c>transfer</c> records over real transports) or a bench's JSON (<c>idrak cluster
/// bench -j</c>: per pair, half the round trip of an empty message and the time of a message with its answer), each mapped
/// to its link through the topology; per link, bandwidth and latency are fitted by least squares over
/// <c>seconds = latency + bytes / bandwidth</c> with the residual, and a bench's concurrent pairs show whether a machine's
/// ranks share one network adapter. <see cref="Keep"/> writes the results into the tuning cache
/// (<see cref="ClusterCache"/>), where <see cref="LinkProfiles.Resolve"/> finds them, and publishes each as
/// <see cref="ProfileCalibrated"/>.
/// </summary>
public static class LinkCalibration
{
    /// <summary>
    /// The least-squares fit of <c>seconds = latency + bytes / bandwidth</c> over <paramref name="samples"/>: the bandwidth
    /// (bytes a second), the latency (0 or more) and the root-mean-square residual (seconds). With one message size only the
    /// latency is 0 and the bandwidth the mean rate.
    /// </summary>
    /// <exception cref="ArgumentException">No sample, or none that moved bytes.</exception>
    public static (double Bandwidth, TimeSpan Latency, double Residual) Fit(IReadOnlyList<(long Bytes, double Seconds)> samples)
    {
        ArgumentNullException.ThrowIfNull(samples);
        if (samples.Count == 0 || samples.All(s => s.Bytes <= 0))
        {
            throw new ArgumentException("A fit needs transfers that moved bytes.");
        }

        double n = samples.Count, meanX = samples.Average(s => (double)s.Bytes), meanY = samples.Average(s => s.Seconds);
        double sxx = samples.Sum(s => (s.Bytes - meanX) * (s.Bytes - meanX)), sxy = samples.Sum(s => (s.Bytes - meanX) * (s.Seconds - meanY));
        double slope, intercept;
        if (sxx > 0 && sxy > 0)
        {
            slope = sxy / sxx;
            intercept = meanY - (slope * meanX);
            if (intercept < 0)
            {
                // A latency below zero is noise: the line through the origin instead.
                slope = samples.Sum(s => s.Bytes * s.Seconds) / samples.Sum(s => (double)s.Bytes * s.Bytes);
                intercept = 0;
            }
        }
        else
        {
            var moved = samples.Where(s => s.Bytes > 0).ToList();
            slope = moved.Sum(s => s.Seconds) / moved.Sum(s => (double)s.Bytes);
            intercept = 0;
        }

        double residual = Math.Sqrt(samples.Sum(s => Math.Pow(s.Seconds - (intercept + (slope * s.Bytes)), 2)) / n);
        return (slope > 0 ? 1 / slope : 0, TimeSpan.FromSeconds(intercept), residual);
    }

    /// <summary>
    /// The transfers of a bench's JSON (<c>idrak cluster bench -j</c>: its <c>rows</c>): per pair, an empty message taking the
    /// warm latency, and a message of its bytes taking the round trip measured less one latency (the empty answer's way
    /// back), each on the link <paramref name="topology"/> gives the pair.
    /// </summary>
    /// <exception cref="InvalidDataException">The JSON has no bench rows, or a row names a rank the topology does not have.</exception>
    public static IReadOnlyList<CalibrationSample> FromBench(JsonObject bench, IClusterTopology topology)
    {
        ArgumentNullException.ThrowIfNull(bench);
        ArgumentNullException.ThrowIfNull(topology);
        if (bench["rows"] is not JsonArray rows)
        {
            throw new InvalidDataException("not a bench's JSON (no \"rows\": write one with idrak cluster bench -j).");
        }

        var samples = new List<CalibrationSample>();
        foreach (var row in rows.OfType<JsonObject>())
        {
            int a = Int(row, "rank_a"), b = Int(row, "rank_b");
            Check(topology, a, b);
            string link = topology.Link(a, b).Name;
            double latency = Number(row, "warm_latency_seconds");
            long bytes = (long)Number(row, "bytes");
            double bandwidth = Number(row, "warm_bytes_per_second");
            samples.Add(new CalibrationSample(link, a, b, 0, latency));
            if (bytes > 0 && bandwidth > 0)
            {
                samples.Add(new CalibrationSample(link, a, b, bytes, Math.Max(bytes / bandwidth - latency, 1e-9)));
            }
        }

        return samples;
    }

    /// <summary>
    /// The transfers of a run's telemetry log (JSON Lines, a run folder's <c>log.jsonl</c>): its <c>transfer</c> records over
    /// real transports (a simulated transfer took the virtual clock's time, which a calibration does not correct), each its
    /// wall time on the link <paramref name="topology"/> gives the pair. Lines that are not JSON are skipped.
    /// </summary>
    public static IReadOnlyList<CalibrationSample> FromLog(string path, IClusterTopology topology)
    {
        ArgumentNullException.ThrowIfNull(path);
        ArgumentNullException.ThrowIfNull(topology);
        var samples = new List<CalibrationSample>();
        foreach (string line in File.ReadLines(path))
        {
            if (!line.Contains("\"transfer\"", StringComparison.Ordinal))
            {
                continue;
            }

            JsonObject? record;
            try
            {
                record = JsonNode.Parse(line) as JsonObject;
            }
            catch (JsonException)
            {
                continue;
            }

            if (record is null || (string?)record["event"] != "transfer" || (string?)record["transport"] is null or "sim")
            {
                continue;
            }

            int from = Int(record, "from"), to = Int(record, "to");
            if ((uint)from >= (uint)topology.WorldSize || (uint)to >= (uint)topology.WorldSize || from == to)
            {
                continue;
            }

            samples.Add(new CalibrationSample(topology.Link(from, to).Name, from, to, (long)Number(record, "bytes"), Number(record, "wall_ms") / 1e3));
        }

        return samples;
    }

    /// <summary>
    /// The shared adapters a bench's concurrent rows show (<c>idrak cluster bench --concurrent</c>: its <c>concurrent</c>
    /// rows, every pair between two nodes sending at once): a node pair whose pairs together moved less than
    /// (1 - <paramref name="tolerance"/>) times their count times one pair alone shares one adapter.
    /// </summary>
    public static IReadOnlyList<AdapterShare> AdaptersFromBench(JsonObject bench, IClusterTopology topology, double tolerance = 0.25)
    {
        ArgumentNullException.ThrowIfNull(bench);
        ArgumentNullException.ThrowIfNull(topology);
        ArgumentOutOfRangeException.ThrowIfNegative(tolerance);
        if (bench["concurrent"] is not JsonArray concurrent || bench["rows"] is not JsonArray rows)
        {
            return [];
        }

        var shares = new List<AdapterShare>();
        foreach (var row in concurrent.OfType<JsonObject>())
        {
            int a = Int(row, "rank_a"), b = Int(row, "rank_b"), pairs = Int(row, "pairs");
            Check(topology, a, b);
            int nodeA = topology.Place(a).Node, nodeB = topology.Place(b).Node;
            string link = topology.Link(a, b).Name;
            var alone = rows.OfType<JsonObject>()
                .Where(r => Pair(topology, Int(r, "rank_a"), Int(r, "rank_b"), nodeA, nodeB))
                .Select(r => Number(r, "warm_bytes_per_second")).Order().ToList();
            if (alone.Count == 0 || pairs < 2)
            {
                continue;
            }

            double one = alone[alone.Count / 2], together = Number(row, "warm_bytes_per_second");
            shares.Add(new AdapterShare(link, pairs, one, together, together < (1 - tolerance) * pairs * one));
        }

        return shares;
    }

    /// <summary>
    /// Fits every link of <paramref name="samples"/> with at least two transfers (and moved bytes), against the profile
    /// <see cref="LinkProfiles.Resolve"/> gives it now; a link of <paramref name="adapters"/> that shares one keeps the
    /// adapter's rate. Nothing is kept: <see cref="Keep"/> does that.
    /// </summary>
    public static IReadOnlyList<LinkCalibrationResult> Calibrate(IEnumerable<CalibrationSample> samples, string source, IEnumerable<AdapterShare>? adapters = null)
    {
        ArgumentNullException.ThrowIfNull(samples);
        var shared = (adapters ?? []).Where(a => a.Shared).GroupBy(a => a.Link, StringComparer.Ordinal).ToDictionary(g => g.Key, g => g.Min(a => a.Together), StringComparer.Ordinal);
        var results = new List<LinkCalibrationResult>();
        var now = DateTimeOffset.Now;
        foreach (var link in samples.GroupBy(s => s.Link, StringComparer.Ordinal).OrderBy(g => g.Key, StringComparer.Ordinal))
        {
            var points = link.Select(s => (s.Bytes, s.Seconds)).ToList();
            if (points.Count < 2 || points.All(p => p.Bytes <= 0))
            {
                continue;
            }

            var (bandwidth, latency, residual) = Fit(points);
            ILinkProfile? before;
            try
            {
                before = LinkProfiles.Resolve(link.Key);
            }
            catch (NotSupportedException)
            {
                before = null;
            }

            var fitted = new CalibratedLink(link.Key, bandwidth, latency, before?.Jitter ?? TimeSpan.Zero, residual, points.Count,
                shared.GetValueOrDefault(link.Key), now, source);
            results.Add(new LinkCalibrationResult(link.Key, before?.BandwidthBytesPerSecond ?? 0, before?.Latency ?? TimeSpan.Zero, fitted));
        }

        return results;
    }

    /// <summary>
    /// Keeps each result in the tuning cache (<see cref="ClusterCache.Keep(CalibratedLink, string)"/>, under
    /// <paramref name="file"/> or the cache's folder) and publishes it as <see cref="ProfileCalibrated"/>; returns the files
    /// written (none when the cache is off).
    /// </summary>
    public static IReadOnlyList<string> Keep(IEnumerable<LinkCalibrationResult> results, string? file = null)
    {
        ArgumentNullException.ThrowIfNull(results);
        var files = new List<string>();
        foreach (var result in results)
        {
            if (ClusterCache.Keep(result.Fitted, file) is { } written && !files.Contains(written))
            {
                files.Add(written);
            }

            if (Telemetry.IsEnabled(TelemetryLevel.Network))
            {
                Telemetry.Calibrated(new ProfileCalibrated(result.Link, result.BandwidthBefore, result.Fitted.BandwidthBytesPerSecond, result.LatencyBefore,
                    result.Fitted.Latency, result.Fitted.Residual, result.Fitted.Samples));
            }
        }

        return files;
    }

    private static bool Pair(IClusterTopology topology, int a, int b, int nodeA, int nodeB)
    {
        if ((uint)a >= (uint)topology.WorldSize || (uint)b >= (uint)topology.WorldSize)
        {
            return false;
        }

        int x = topology.Place(a).Node, y = topology.Place(b).Node;
        return (x == nodeA && y == nodeB) || (x == nodeB && y == nodeA);
    }

    private static void Check(IClusterTopology topology, int a, int b)
    {
        if ((uint)a >= (uint)topology.WorldSize || (uint)b >= (uint)topology.WorldSize || a == b)
        {
            throw new InvalidDataException($"the bench names ranks {a} and {b}; the topology has ranks 0 to {topology.WorldSize - 1} (the bench's topology, or give --topology).");
        }
    }

    private static int Int(JsonObject row, string key) => row[key] is JsonValue v && v.TryGetValue(out int i) ? i
        : row[key] is JsonValue d && d.TryGetValue(out double x) && x == Math.Floor(x) ? (int)x
        : throw new InvalidDataException($"a row has no whole number \"{key}\".");

    private static double Number(JsonObject row, string key) => row[key] is JsonValue v && v.TryGetValue(out double d) && double.IsFinite(d) ? d
        : row[key] is JsonValue s && s.TryGetValue(out string? text) && double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double parsed) ? parsed
        : throw new InvalidDataException($"a row has no number \"{key}\".");
}
