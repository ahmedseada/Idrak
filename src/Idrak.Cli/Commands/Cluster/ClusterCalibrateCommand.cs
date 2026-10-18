// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Cluster;
using Idrak.Cli.Shared;
using Idrak.Cluster;

namespace Idrak.Cli.Commands.Cluster;

/// <summary>
/// <c>idrak cluster calibrate --from RUN|bench.json</c> (plan 23, phase C8): fits each link's bandwidth and latency to the
/// transfers a run's log or a bench's JSON measured (<see cref="LinkCalibration"/>), and keeps them in the tuning cache's
/// cluster folder, one file per link with the residual, where every later topology that names the link finds them;
/// <c>--dry-run</c> prints the deltas only.
/// </summary>
internal sealed class ClusterCalibrateCommand : Command
{
    public override string Name => "cluster calibrate";

    public override string Summary => "Fits each link's bandwidth and latency to a run's or bench's transfers and keeps the profiles";

    public override string Usage => """
        --from RUN|BENCH.json [--topology FILE] [--dry-run] [--shared-tolerance F]

        Reads measured transfers: a bench's JSON (idrak cluster bench -j writes it: per pair, half the round trip of an
        empty message and a message's time; its topology is the one the bench names) or a run's telemetry log (a run
        folder's log.jsonl, its transfers over real transports; give the run's --topology). Each transfer is placed on
        the link the topology gives its pair, and each link's bandwidth and latency are fitted by least squares over
        seconds = latency + bytes / bandwidth, with the residual. A bench of concurrent pairs (cluster bench
        --concurrent) also shows whether a machine's ranks share one network adapter: when the pairs between two
        machines together move less than their count times one pair alone, the link keeps the adapter's rate and the
        simulator gives each pair its share. The fitted profiles are written to the tuning cache's cluster folder, one
        file per link (IDRAK_CLUSTER_PROFILES_FILE puts them in one file), and every topology naming the link uses them
        from then on: idrak cluster inspect shows them as calibrated, idrak tuning show lists them. --dry-run prints the
        deltas without writing.

        Options:
              --from PATH          a bench's JSON, a run folder, or a telemetry log (.jsonl)
              --topology FILE      the topology the transfers ran on (a bench's JSON names its own)
              --dry-run            print the deltas only
              --shared-tolerance F how far below their count times one pair alone the concurrent pairs may move and
                                   still have an adapter each (0.25)

        Examples:
          idrak cluster bench --topology lan.json --concurrent -j > bench.json
          idrak cluster calibrate --from bench.json
          idrak cluster calibrate --from runs/lan-run --topology lan.json --dry-run
        """;

    public override IReadOnlyCollection<string> ValueOptions => ["--from", "--topology", "--shared-tolerance"];

    public override IReadOnlyCollection<string> Flags => ["--dry-run"];

    public override int Run(CommandContext context)
    {
        string from = context.Option("--from") ?? throw new UsageException("Give the measured transfers with --from: a bench's JSON (idrak cluster bench -j) or a run folder.");
        double tolerance = context.Option("--shared-tolerance") is { } text
            ? double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double t) && t is >= 0 and < 1 ? t : throw new UsageException($"--shared-tolerance is a share from 0 to 1, not '{text}'.")
            : 0.25;
        string? log = Directory.Exists(from) ? Path.Combine(from, "log.jsonl") : File.Exists(from) ? from : null;
        if (log is null || !File.Exists(log))
        {
            context.Error($"{from} is neither a bench's JSON nor a run folder with a log.jsonl.");
            return ExitCodes.Failed;
        }

        JsonObject? bench = null;
        if (!log.EndsWith(".jsonl", StringComparison.OrdinalIgnoreCase))
        {
            try
            {
                bench = JsonNode.Parse(File.ReadAllText(log)) as JsonObject;
            }
            catch (JsonException)
            {
                bench = null;                                              // a log of JSON lines named otherwise
            }
        }

        string? topologyPath = context.Option("--topology") ?? (string?)bench?["topology"] ?? Environment.GetEnvironmentVariable("IDRAK_CLUSTER_TOPOLOGY");
        if (topologyPath is null)
        {
            throw new UsageException($"Give the topology {from} ran on with --topology FILE: its transfers are placed on the links it names.");
        }

        IReadOnlyList<CalibrationSample> samples;
        IReadOnlyList<AdapterShare> adapters = [];
        try
        {
            var topology = ClusterTopology.Load(topologyPath);
            if (bench is not null)
            {
                samples = LinkCalibration.FromBench(bench, topology);
                adapters = LinkCalibration.AdaptersFromBench(bench, topology, tolerance);
            }
            else
            {
                samples = LinkCalibration.FromLog(log, topology);
            }
        }
        catch (Exception e) when (e is InvalidDataException or NotSupportedException or FileNotFoundException or DirectoryNotFoundException)
        {
            context.Error(e.Message);
            return ExitCodes.Failed;
        }

        var results = LinkCalibration.Calibrate(samples, Path.GetFullPath(log), adapters);
        var inv = CultureInfo.InvariantCulture;
        context.Write($"{samples.Count} measured transfer(s) from {from} on {samples.Select(s => s.Link).Distinct().Count()} link(s) of {topologyPath}");
        if (results.Count == 0)
        {
            context.Error("No link has two measured transfers that moved bytes: nothing to fit (a bench measures each pair; a run's log keeps its "
                          + "transfers over real transports only).");
            return ExitCodes.Failed;
        }

        context.Table(["link", "bandwidth before", "after", "latency before", "after", "residual", "transfers", "shared adapter"],
            [.. results.Select(r => (IReadOnlyList<string>)
            [
                r.Link, Rate(r.BandwidthBefore), Rate(r.Fitted.BandwidthBytesPerSecond), Micro(r.LatencyBefore.TotalSeconds), Micro(r.Fitted.Latency.TotalSeconds),
                Micro(r.Fitted.Residual), r.Fitted.Samples.ToString(inv), r.Fitted.AdapterBytesPerSecond > 0 ? Rate(r.Fitted.AdapterBytesPerSecond) : "no",
            ])]);
        foreach (var share in adapters)
        {
            context.Write($"{share.Link}: {share.Pairs} pairs at once moved {Rate(share.Together)} together, one alone {Rate(share.Alone)}: "
                          + (share.Shared ? "one adapter shared by the machine's ranks" : "an adapter each"));
        }

        bool dryRun = Terminal.DryRun(context);
        IReadOnlyList<string> files = [];
        if (!dryRun)
        {
            files = LinkCalibration.Keep(results);
            context.Write(files.Count == 0
                ? "the tuning cache is off (IDRAK_TUNING_CACHE=0 or an empty IDRAK_CLUSTER_PROFILES_FILE): nothing was kept"
                : $"kept in {string.Join(", ", files)}; topologies naming these links use them (idrak cluster inspect, idrak tuning show)");
        }
        else
        {
            context.Write("--dry-run: nothing was kept");
        }

        context.WriteJson(new JsonObject
        {
            ["from"] = Path.GetFullPath(from),
            ["topology"] = Path.GetFullPath(topologyPath),
            ["transfers"] = samples.Count,
            ["written"] = !dryRun && files.Count > 0,
            ["files"] = new JsonArray([.. files.Select(f => (JsonNode)f)]),
            ["links"] = new JsonArray([.. results.Select(r => (JsonNode)new JsonObject
            {
                ["link"] = r.Link,
                ["bandwidth_before"] = r.BandwidthBefore,
                ["bandwidth"] = r.Fitted.BandwidthBytesPerSecond,
                ["latency_before_seconds"] = r.LatencyBefore.TotalSeconds,
                ["latency_seconds"] = r.Fitted.Latency.TotalSeconds,
                ["residual_seconds"] = r.Fitted.Residual,
                ["samples"] = r.Fitted.Samples,
                ["adapter_bytes_per_second"] = r.Fitted.AdapterBytesPerSecond,
            })]),
        });
        return ExitCodes.Ok;
    }

    private static string Rate(double bytesPerSecond) => bytesPerSecond > 0
        ? string.Create(CultureInfo.InvariantCulture, $"{bytesPerSecond / (1 << 20):F1} MiB/s")
        : "unknown";

    private static string Micro(double seconds) => string.Create(CultureInfo.InvariantCulture, $"{seconds * 1e6:F1} us");
}
