// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Abstraction.Cluster;
using Idrak.Cli;
using Idrak.Cluster;
using Idrak.Datasets;
using Idrak.Layers;
using Idrak.Training;

// Plan 23, phase C8c (decision 4): the "cluster" setting advisor. A data-parallel run's gradient synchronisation measures
// its first warm steps (the network's share of a step, a gradient slice reduced over a float32 and a bfloat16 wire) while
// someone listens for suggestions, and rank 0 asks the advisors once; the "cluster" advisor suggests a bf16 wire, gradient
// accumulation, local SGD, top-k, the bucket size and the hierarchical collective by those figures, never applying one.
// The conditions (ClusterSettingAdvisor): nothing at or under IDRAK_CLUSTER_SUGGEST_SHARE; hierarchical where ring runs
// over several nodes of several ranks; bf16 where the buckets travel uncompressed and the slice's error is within
// IDRAK_CLUSTER_SUGGEST_BF16_TOLERANCE; accumulation and local SGD by the factor A = ceil(E'(1-T)/(T·C)) (E' the exposed
// network time, halved where bf16 is suggested, C the step's compute, T the share) where E'/(C+E') is still above T, and
// top-k with the share T·C/((1-T)·E·W) beside them; a bucket size where one pinned bucket holds every gradient.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ClusterAdvisorGroup =
    [
        ("cluster: advisor on a slow link (ethernet-1g, a model whose all-reduce is most of the step) suggests --wire bf16 once within the default tolerance (bf16's rounding step), with the network's share and the measured bf16 error in its reason, and accumulation, local SGD and top-k where bf16 alone leaves the share above the threshold", ClusterAdvisorSlowLink),
        ("cluster: advisor on a fast link (a same-device link declared at 1 PB/s) suggests nothing: the network's share is under IDRAK_CLUSTER_SUGGEST_SHARE", ClusterAdvisorFastLink),
        ("cluster: advisor a bf16 error above the tolerance stops the bf16 suggestion; the others are made on the whole network time", ClusterAdvisorBf16Refused),
        ("cluster: advisor ring over two nodes of two ranks suggests the hierarchical collective; one bucket pinned for every gradient gets a bucket size that lets the network go during the backward pass", ClusterAdvisorHierarchicalAndBuckets),
        ("cluster: advisor the run's losses and weights are the same bit for bit with the advisor on and off; nothing is measured without a listener when it is unset or off, and on it adds its two slice collectives once", ClusterAdvisorChangesNothing),
        ("cluster: advisor cli cluster sim shows each suggestion once above the progress line, the run's log keeps them and idrak suggest RUN lists them", ClusterAdvisorCli),
    ];

    // The settings every advisor test trains with: equal splits (no speed measured, so reruns split alike) and ring.
    private static ClusterOptions AdvisorOptions() => DpOptions() with { Collective = "ring" };

    // Records what was suggested (every advisor's) and, through an app advisor of its own, what the cluster measured.
    private sealed class AdviceSeen : ITelemetryHook, ISettingAdvisor
    {
        public List<SettingSuggestion> Suggestions { get; } = [];

        public List<ClusterStepMeasure> Measures { get; } = [];

        public TelemetryLevel Levels => TelemetryLevel.Suggestions;

        public string Name => "cluster-test-recorder";

        public string Summary => "records the cluster's measurements (a test's)";

        public void OnSuggestion(in SettingSuggestion e)
        {
            lock (Suggestions)
            {
                Suggestions.Add(e);
            }
        }

        public IEnumerable<SettingSuggestion> Suggest(AdviceSituation situation)
        {
            if (situation.Subject is ClusterStepMeasure m)
            {
                lock (Measures)
                {
                    Measures.Add(m);
                }
            }

            return [];
        }

        public List<SettingSuggestion> Cluster => [.. Suggestions.Where(s => s.Source == "cluster")];
    }

    // Each rank's weights and epoch losses after training the Spirals classifier (three layers of 256: 520 KiB of gradients)
    // in batches of 32 a rank, so the all-reduce on a slow link is most of a step.
    private static (float[] Weights, float[] Losses) AdvisorTrain(ClusterRank rank, ClusterOptions options, int epochs = 2)
    {
        var data = Spirals(200, 3, 1);
        using var model = OverlapModel(rank.Device, width: 256, layers: 3);
        using var optimizer = new Sgd(model.Parameters(), 0.05f);
        var trainer = new Trainer(model, optimizer, Losses.CrossEntropy) { Cluster = rank.Collectives, ClusterOptions = options };
        var history = trainer.Fit(new DataLoader(data, 32, shuffle: true, device: rank.Device, seed: 3), epochs);
        return ([.. model.Parameters().SelectMany(p => p.ToArray())], [.. history.Epochs.Select(e => (float)e.Loss)]);
    }

    // A run with the suggestions listened to (or not), the measurements recorded through an app advisor.
    private static (AdviceSeen Seen, (float[] Weights, float[] Losses)[] Ranks) AdvisorRun(IClusterTopology topology, ClusterOptions options, bool listen = true)
    {
        var seen = new AdviceSeen();
        SettingAdvisors.Register(seen);
        try
        {
            using var subscription = listen ? Telemetry.Subscribe(seen) : null;
            var ranks = DpRun(topology, options, rank => AdvisorTrain(rank, options));
            return (seen, ranks);
        }
        finally
        {
            SettingAdvisors.Unregister(seen.Name);
        }
    }

    private static double ClusterSettingAdvisorTolerance => Idrak.Networking.Training.ClusterSettingAdvisor.DefaultBFloat16Tolerance;

    private static string AdvicePct(double share) => (share * 100).ToString("F0", CultureInfo.InvariantCulture) + "%";

    private static string Describe(ClusterStepMeasure m) => string.Create(CultureInfo.InvariantCulture,
        $"share {m.Share:P1} (exposed {m.Exposed.TotalMilliseconds:F3} ms of a {m.Step.TotalMilliseconds:F3} ms step, network {m.Network.TotalMilliseconds:F3} ms, backward {m.Backward.TotalMilliseconds:F3} ms), "
        + $"{m.GradientBytes / 1024.0:F0} KiB in {m.Buckets} bucket(s) of {m.BucketBytes / 1024.0:F0} KiB, {m.Steps} warm steps, {m.WorldSize} ranks on {m.Nodes} node(s), "
        + $"bf16 error {(m.BFloat16Error is { } e ? e.ToString("G3", CultureInfo.InvariantCulture) : "-")} on {m.SliceValues} values");

    private static void ClusterAdvisorSlowLink(Device device)
    {
        var options = AdvisorOptions();                                   // the default tolerance: bfloat16's rounding step, 2^-7
        var (seen, _) = AdvisorRun(NodesOf([1, 1], device.ToString(), "ethernet-1g"), options);
        Check(seen.Measures.Count == 1, $"rank 0 asked the advisors once: {seen.Measures.Count} measurement(s)");
        var m = seen.Measures[0];
        Console.WriteLine($"    slow link: {Describe(m)}");
        foreach (var s in seen.Cluster)
        {
            Console.WriteLine($"    suggested {s.Setting} = {s.Value}: {s.Reason}");
        }

        Check(m.Simulated && m.Steps == options.SuggestSteps && m.Share > options.SuggestShare && m.BFloat16Error is > 0,
            $"the first warm steps measured on the virtual clock, the network above the threshold, bf16's error measured: {Describe(m)}");
        var cluster = seen.Cluster;
        Check(cluster.Select(s => (s.Setting, s.Value)).Distinct().Count() == cluster.Count && cluster.All(s => s.Device.Length > 0),
            $"each suggestion once: {string.Join(", ", cluster.Select(s => $"{s.Setting}={s.Value}"))}");
        var wire = cluster.Where(s => s.Setting == "ClusterOptions.Wire").ToList();
        string error = m.BFloat16Error!.Value.ToString("G3", CultureInfo.InvariantCulture);
        Check(wire.Count == 1 && wire[0].Value == "bf16" && wire[0].Reason.Contains(AdvicePct(m.Share), StringComparison.Ordinal) && wire[0].Reason.Contains(error, StringComparison.Ordinal)
              && wire[0].Reason.Contains("--wire bf16", StringComparison.Ordinal),
            $"--wire bf16 suggested once with the share {AdvicePct(m.Share)} and the error {error} in its reason: {string.Join(" | ", wire.Select(s => s.Reason))}");

        // With bf16's half the bytes the share may still be above the threshold: then accumulation, local SGD and top-k.
        double exposed = m.Exposed.TotalSeconds / 2, compute = m.Compute.TotalSeconds, threshold = options.SuggestShare;
        bool still = compute > 0 && exposed / (compute + exposed) > threshold;
        int factor = still ? Math.Max(2, (int)Math.Ceiling(exposed * (1 - threshold) / (threshold * compute))) : 0;
        var accumulate = cluster.FirstOrDefault(s => s.Setting == "GradientAccumulation");
        var localSgd = cluster.FirstOrDefault(s => s.Setting == "ClusterOptions.LocalSgdEvery");
        var topk = cluster.FirstOrDefault(s => s.Setting == "ClusterOptions.TopK");
        string f = factor.ToString(CultureInfo.InvariantCulture);
        Check(still ? accumulate.Value == f && localSgd.Value == f && topk.Value is { Length: > 0 } && accumulate.Reason.Contains("with bf16", StringComparison.Ordinal)
                    : accumulate.Setting is null && localSgd.Setting is null && topk.Setting is null,
            $"accumulation and local SGD by {f} and top-k {(still ? "suggested" : "not suggested")} (with bf16 the share is {AdvicePct(exposed / (compute + exposed))}): "
            + $"{accumulate.Value}, {localSgd.Value}, {topk.Value}");
        Check(cluster.All(s => s.Setting is not ("ClusterOptions.Collective" or "ClusterOptions.BucketBytes")),
            "no hierarchical suggestion for nodes of one rank, no bucket size where it was measured");
    }

    private static void ClusterAdvisorFastLink(Device device)
    {
        // A link fast by declaration, not by measurement: same-device is measured on the device under test, and two
        // devices carved from one card on some backends (Vulkan: one VkDevice each) copy through the host, which is
        // slow beside a tiny model's step there. Card-agnostic: the test fixes the link, the advisor measures the rest.
        var options = AdvisorOptions();
        LinkProfiles.Register(new LinkProfile(ClusterTopology.SameDevice, 1e15, TimeSpan.Zero));
        try
        {
            var (seen, _) = AdvisorRun(DpTopology(device, 2), options);
            Check(seen.Measures.Count == 1, $"the fast link was measured too: {seen.Measures.Count}");
            var m = seen.Measures[0];
            Console.WriteLine($"    a 1 PB/s link: {Describe(m)}");
            Check(m.Share <= options.SuggestShare && seen.Cluster.Count == 0,
                $"the network's share {AdvicePct(m.Share)} is under {AdvicePct(options.SuggestShare)}: nothing suggested ({string.Join(", ", seen.Cluster.Select(s => s.Setting))})");
        }
        finally
        {
            LinkProfiles.Unregister(ClusterTopology.SameDevice);
        }
    }

    private static void ClusterAdvisorBf16Refused(Device device)
    {
        var options = AdvisorOptions() with { SuggestBFloat16Tolerance = 1e-9 };
        var (seen, _) = AdvisorRun(NodesOf([1, 1], device.ToString(), "ethernet-1g"), options);
        var m = seen.Measures.Single();
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    the default tolerance ({ClusterSettingAdvisorTolerance:G3}) {(m.BFloat16Error <= ClusterSettingAdvisorTolerance ? "accepts" : "refuses")} the measured {m.BFloat16Error:G3}"));
        Console.WriteLine($"    bf16 tolerance 1e-9: {Describe(m)}; suggested {string.Join(", ", seen.Cluster.Select(s => $"{s.Setting}={s.Value}"))}");
        double exposed = m.Exposed.TotalSeconds, compute = m.Compute.TotalSeconds, threshold = options.SuggestShare;
        int factor = Math.Max(2, (int)Math.Ceiling(exposed * (1 - threshold) / (threshold * compute)));
        var accumulate = seen.Cluster.FirstOrDefault(s => s.Setting == "GradientAccumulation");
        Check(m.BFloat16Error > 1e-9 && seen.Cluster.All(s => s.Setting != "ClusterOptions.Wire"), $"no bf16 suggestion with an error {m.BFloat16Error:G3} above the tolerance");
        Check(accumulate.Value == factor.ToString(CultureInfo.InvariantCulture) && !accumulate.Reason.Contains("with bf16", StringComparison.Ordinal),
            $"accumulation by {factor} from the whole exposed time: {accumulate.Value} ({accumulate.Reason})");
    }

    private static void ClusterAdvisorHierarchicalAndBuckets(Device device)
    {
        var options = AdvisorOptions();
        var (seen, _) = AdvisorRun(NodesOf([2, 2], device.ToString(), "ethernet-1g"), options);
        var m = seen.Measures.Single();
        Console.WriteLine($"    ring over 2 nodes of 2: {Describe(m)}");
        var hierarchical = seen.Cluster.Where(s => s.Setting == "ClusterOptions.Collective").ToList();
        Check(m.Nodes == 2 && m.WorldSize == 4 && hierarchical is [{ Value: "hierarchical" }] && hierarchical[0].Reason.Contains("2 nodes", StringComparison.Ordinal),
            $"hierarchical suggested for ring over 2 nodes: {string.Join(" | ", hierarchical.Select(s => s.Reason))}");

        var pinned = AdvisorOptions() with { BucketBytes = 64L << 20 };
        var (buckets, _) = AdvisorRun(NodesOf([1, 1], device.ToString(), "ethernet-1g"), pinned);
        var b = buckets.Measures.Single();
        Console.WriteLine($"    one pinned bucket: {Describe(b)}");
        var size = buckets.Cluster.Where(s => s.Setting == "ClusterOptions.BucketBytes").ToList();
        long bytes = size.Count == 1 ? ClusterTopology.ParseMemory(size[0].Value) : 0;
        Check(b.Buckets == 1 && b.Backward > TimeSpan.Zero && size.Count == 1 && bytes > 0 && bytes < b.GradientBytes,
            $"a bucket size under the gradients' {b.GradientBytes} B suggested for one pinned bucket: {string.Join(" | ", size.Select(s => $"{s.Value}: {s.Reason}"))}");
        Console.WriteLine($"    suggested bucket size {size[0].Value}: {size[0].Reason}");

        var flat = AdvisorRun(NodesOf([2, 2], device.ToString(), "ethernet-1g"), AdvisorOptions() with { Collective = "hierarchical" }).Seen;
        Check(flat.Cluster.All(s => s.Setting != "ClusterOptions.Collective"), "hierarchical set: not suggested again");
    }

    private static void ClusterAdvisorChangesNothing(Device device)
    {
        var topology = NodesOf([1, 1], device.ToString(), "ethernet-1g");
        var (on, onRanks) = AdvisorRun(topology, AdvisorOptions());
        var (off, offRanks) = AdvisorRun(topology, AdvisorOptions() with { Suggest = false });
        Check(on.Measures.Count == 1 && on.Cluster.Count > 0 && off.Measures.Count == 0 && off.Suggestions.Count == 0, "measured with a listener, nothing with IDRAK_CLUSTER_SUGGEST=0");
        for (int r = 0; r < onRanks.Length; r++)
        {
            Check(onRanks[r].Weights.SequenceEqual(offRanks[r].Weights) && onRanks[r].Losses.SequenceEqual(offRanks[r].Losses),
                $"rank {r}: the weights and losses bit for bit with the advisor on and off");
        }

        // The network's collectives counted (a Network listener alone does not turn the advisor on).
        int Collectives(ClusterOptions options)
        {
            var metrics = new MetricsRecorder(TelemetryLevel.Network);
            using (Telemetry.Subscribe(metrics))
            {
                DpRun(topology, options, rank => AdvisorTrain(rank, options));
            }

            return metrics.Collectives.Count(c => c.Rank == 0);
        }

        int unset = Collectives(AdvisorOptions()), disabled = Collectives(AdvisorOptions() with { Suggest = false }), forced = Collectives(AdvisorOptions() with { Suggest = true });
        Console.WriteLine($"    rank 0's collectives: {unset} unset without a listener, {disabled} off, {forced} on");
        Check(unset == disabled && forced == unset + 2, $"nothing measured without a listener ({unset} collectives, {disabled} off); on, the slice over both wires once ({forced})");
    }

    private static void ClusterAdvisorCli(Device device)
    {
        if (!FirstRun(nameof(ClusterAdvisorCli)))
        {
            return;
        }

        _ = device;
        string folder = TempFolder("cluster-advisor-cli");
        try
        {
            string topology = Path.Combine(folder, "slow.json");
            File.WriteAllText(topology, NodesOf([1, 1], "cpu", "ethernet-1g").ToJson());
            var random = new Random(5);
            var csv = new StringBuilder("x1,x2,y\n");
            for (int i = 0; i < 96; i++)
            {
                double a = random.NextDouble() * 4 - 2, b = random.NextDouble() * 4 - 2;
                csv.Append(CultureInfo.InvariantCulture, $"{a:F4},{b:F4},{3 * a - 2 * b + 1:F4}\n");
            }

            File.WriteAllText(Path.Combine(folder, "reg.csv"), csv.ToString());
            File.WriteAllText(Path.Combine(folder, "reg.json"), Network.Input(2).Linear(512).ReLU().Linear(512).ReLU().Linear(1).Named("reg").ToJson().ToJsonString());
            string run = Path.Combine(folder, "run");
            var (code, text, error) = ClusterCli("cluster", "sim", "--topology", topology, "--collective", "ring", "--", "train", Path.Combine(folder, "reg.json"),
                "--data", Path.Combine(folder, "reg.csv"), "--epochs", "3", "--batch", "16", "--run", run, "-o", Path.Combine(folder, "reg.ikm"));
            var shown = text.Split('\n').Where(l => l.StartsWith("suggested: ", StringComparison.Ordinal)).Select(l => l.TrimEnd('\r')).ToList();
            Console.WriteLine($"    cluster sim shows {shown.Count} suggestion(s):\n      " + string.Join("\n      ", shown));
            Check(code == 0 && shown.Count > 0 && shown.All(l => l.Contains("(cluster on ", StringComparison.Ordinal)) && shown.Distinct().Count() == shown.Count,
                $"cluster sim shows the cluster advisor's suggestions, each once: {code} {error}\n{text}");

            var listed = ClusterCli("suggest", run);
            var lines = listed.Output.Split('\n').Select(l => l.TrimEnd('\r')).Where(l => l.StartsWith("suggested: ", StringComparison.Ordinal)).ToList();
            Check(listed.Code == 0 && lines.SequenceEqual(shown), $"idrak suggest RUN lists what the run showed: {listed.Code} {listed.Error}\n{listed.Output}");
            var json = JsonNode.Parse(ClusterCli("suggest", run, "-j").Output)!;
            Check(json["suggestions"]!.AsArray().Count == shown.Count && json["suggestions"]!.AsArray().All(s => (string?)s!["source"] == "cluster"),
                $"idrak suggest RUN -j: {json.ToJsonString()}");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }
}
