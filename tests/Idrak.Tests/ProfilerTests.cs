// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;
using Idrak;

// The profiler contract (plan 17, H3): a workload of one's own run cold and warm with the probes around it, the intrusive
// probes in runs of their own, the report summed up by the chosen statistic and written by any registered writer.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ProfilerGroup =
    [
        ("profiler: a workload's stages and values, the cold run apart, the statistic over the warm runs; cpu, heap and memory around each timed run; the intrusive probes in runs of their own (operations and host fallbacks counted, kernels only where the device times them); an unknown probe refused naming the registered ones", ProfilerRuns),
        ("profiler: the table, markdown, csv and json writers; an app's probe and writer registered, used and removed; the criteria from IDRAK_PROFILE_*", ProfilerWritersAndCriteria),
    ];

    // A small product and softmax on the device, a stage of it timed, a value of the workload's own recorded.
    private static void ProfiledProduct(ProfileRun run)
    {
        using var scope = new TensorScope();
        var a = Tensor.From([.. Enumerable.Range(0, 64 * 64).Select(i => (i % 13) / 13f)], [64, 64], run.Device);
        var product = run.Stage("product", () => a.MatMul(a).Softmax());
        run.Record("items", product.Shape[0], "rows");
    }

    private static void ProfilerRuns(Device device)
    {
        var report = Profiler.Run("product", device, ProfiledProduct, new ProfileOptions { WarmRuns = 3 });
        int intrusive = report.Probes.Count(p => ProfileProbes.Get(p).Intrusive);
        Check(report.Runs.Count == 1 + 3 + intrusive && report.Runs[0].Kind == ProfileRunKind.Cold && report.Runs.Skip(1).Take(3).All(r => r.Kind == ProfileRunKind.Warm)
              && report.Runs.Skip(4).All(r => r.Kind == ProfileRunKind.Probe && r.Probe is not null),
            $"a cold run, 3 warm and one for each intrusive probe: {string.Join(", ", report.Runs.Select(r => $"{r.Kind}{r.Probe}"))}");
        Check(report["product"] is { Unit: "ms", Count: 3, Cold: > 0, Value: > 0, Source: null } stage && stage.Min <= stage.Value && stage.Value <= stage.Max,
            $"the stage over the warm runs: {report["product"]}");
        Check(report["items"] is { Value: 64, Unit: "rows" } && report["wall"] is { Count: 3 }, "the workload's own value and the wall time");
        foreach (string name in new[] { "cpu", "cores busy", "allocated", "gc0", "gc pause", "device in use", "device peak", "process peak" })
        {
            Check(report[name] is { Count: 3, Source: not null }, $"{name} measured around each warm run: {report[name]}");
        }

        Check(report.Runs.Where(r => r.Kind != ProfileRunKind.Probe).All(r => r.Value("operations") is null), "a trace never runs in a timed run");
        Check(report["operations"] is { Value: > 0, Count: 1, Source: "operations" } && report["operation kinds"] is { Value: >= 2 },
            $"the operations' pass counts the calls: {report["operations"]}, {report["operation kinds"]}");
        var pass = report.Runs.Single(r => r.Probe == "operations");
        Check(pass.Tables.Any(t => t.Name == "operations" && t.Rows.Any(r => r[0] == "Softmax")), "the operations by count name Softmax");
        bool times = device.Backend.Capabilities.Profiling;
        Check(times ? report["kernel time"] is { Value: > 0 } && report.Probes.Contains("kernels") : report.Skipped.Contains("kernels") && report["kernel time"] is null,
            times ? "the device times its kernels" : $"kernels skipped on {device}: {string.Join(", ", report.Skipped)}");

        // A device without kernels of its own: every operation a host fallback, counted by operation.
        if (device.Type == DeviceType.Cpu)
        {
            var minimal = Profiler.Run("minimal", Device.Get("minimal"), ProfiledProduct, new ProfileOptions { Cold = false, WarmRuns = 1, Probes = ["operations"] });
            Check(minimal.Runs.Count == 2 && minimal["host fallbacks"] is { Value: > 0 }
                  && minimal.Runs[1].Tables.Any(t => t.Name == "host fallbacks" && t.Rows.Any(r => r[0] == "Softmax")),
                $"host fallbacks on the minimal device: {minimal["host fallbacks"]}");
        }

        string refused = "";
        try
        {
            _ = Profiler.Run("x", device, _ => { }, new ProfileOptions { Probes = ["nope"] });
        }
        catch (NotSupportedException e)
        {
            refused = e.Message;
        }

        Check(refused.Contains("'nope' is not registered", StringComparison.Ordinal) && refused.Contains("cpu", StringComparison.Ordinal), $"an unknown probe: {refused}");
        Check(Profiler.Run("none", device, _ => { }, new ProfileOptions { Probes = [], WarmRuns = 1 }).Summary.Select(s => s.Name).SequenceEqual(["wall"]), "no probes: the wall time alone");

        // Graphs recorded while steps run directly (the kernels probe's pass) replay by running their step; after, as the device does.
        int ran = 0;
        using (ComputeGraph.RunStepsDirectly())
        using (var graph = ComputeGraph.Capture(device, () => ran++))
        {
            graph.Replay();
            graph.Replay();
            Check(!graph.IsRecorded && ran == 2, $"while steps run directly: recorded {graph.IsRecorded}, the step ran {ran} times in 2 replays");
        }

        using var filled = Tensor.Zeros([256], device);
        using (var after = ComputeGraph.Capture(device, () => device.Backend.Fill(filled.Storage, 256, 1f)))
        {
            Check(after.IsRecorded == device.Backend.SupportsGraphs, $"after the scope a graph is recorded where the device records them: {after.IsRecorded}");
        }
    }

    // A probe of the app's: a counter of runs.
    private sealed class CountingProbe : IProfileProbe
    {
        public int Begun;

        public string Name => "runs counted";

        public string Summary => "counts the runs (test)";

        public bool Intrusive => false;

        public bool Supports(Device device) => true;

        public Action Begin(ProfileRun run)
        {
            int n = ++Begun;
            return () => run.Record("run number", n);
        }
    }

    private sealed class NamesWriter : IProfileWriter
    {
        public string Name => "names";

        public string Summary => "the summary's names (test)";

        public void Write(ProfileReport report, TextWriter output) => output.Write(string.Join(";", report.Summary.Select(s => s.Name)));
    }

    private static void ProfilerWritersAndCriteria(Device device)
    {
        var options = new ProfileOptions { WarmRuns = 2, Probes = ["cpu", "heap"] };
        var report = Profiler.Run("write, me", device, ProfiledProduct, options);
        string Written(string format)
        {
            var text = new StringWriter();
            report.Write(text, format);
            return text.ToString();
        }

        string table = Written("table");
        Check(table.Contains("Profile \"write, me\"", StringComparison.Ordinal) && table.Contains("median", StringComparison.Ordinal)
              && table.Split('\n').Any(l => l.StartsWith("product", StringComparison.Ordinal) && l.Contains("ms", StringComparison.Ordinal)), $"table:\n{table}");
        string markdown = Written("markdown");
        Check(markdown.Contains("| value | unit | cold | median | min | max | from |", StringComparison.Ordinal), $"markdown:\n{markdown}");
        var csv = Written("csv").Split('\n', StringSplitOptions.RemoveEmptyEntries);
        Check(csv[0].Trim() == "report,device,run,kind,probe,source,name,value,unit" && csv.Skip(1).All(l => l.StartsWith("\"write, me\",", StringComparison.Ordinal))
              && csv.Count(l => l.Contains(",product,", StringComparison.Ordinal)) == 3, $"csv: {csv.Length} lines, {csv[0]} / {csv.ElementAtOrDefault(1)}");
        string json = Written("json");
        using (var document = JsonDocument.Parse(json))
        {
            var root = document.RootElement;
            Check(json.Trim().Split('\n').Length == 1 && root.GetProperty("report").GetString() == "write, me" && root.GetProperty("runs").GetArrayLength() == 3
                  && root.GetProperty("summary").EnumerateArray().Any(s => s.GetProperty("name").GetString() == "product" && s.GetProperty("count").GetInt32() == 2),
                "json: one line, the runs and the summary");
        }

        // An app's probe and writer, by name, then gone.
        var probe = new CountingProbe();
        ProfileProbes.Register(probe);
        ProfileWriters.Register(new NamesWriter());
        try
        {
            var counted = Profiler.Run("counted", device, _ => { }, new ProfileOptions { WarmRuns = 2, Probes = ["Runs Counted"] });
            Check(probe.Begun == 3 && counted["run number"] is { Cold: 1, Min: 2, Max: 3 }, $"the app's probe around every timed run: {counted["run number"]}");
            var names = new StringWriter();
            counted.Write(names, "NAMES");
            Check(names.ToString() == "wall;run number", $"the app's writer: {names}");
            Check(ProfileProbes.Origin("runs counted") != Overrides.Library && ProfileWriters.Names.Contains("names"), "registered by the app");
        }
        finally
        {
            ProfileProbes.Unregister("runs counted");
            ProfileWriters.Unregister("names");
        }

        Check(ProfileProbes.Find("runs counted") is null && ProfileWriters.Find("names") is null, "unregistered");
        Check(ProfileProbes.Names.SequenceEqual(["cpu", "heap", "memory", "operations", "kernels"]) && ProfileWriters.Names.SequenceEqual(["table", "markdown", "csv", "json"]),
            $"the library's: {string.Join(", ", ProfileProbes.Names)}; {string.Join(", ", ProfileWriters.Names)}");

        // The criteria from the environment.
        var environment = new Dictionary<string, string>
        {
            ["IDRAK_PROFILE_COLD"] = "0", ["IDRAK_PROFILE_WARM"] = "5", ["IDRAK_PROFILE_PROBES"] = "cpu, memory", ["IDRAK_PROFILE_STATISTIC"] = "Min",
            ["IDRAK_PROFILE_FORMAT"] = "json", ["IDRAK_PROFILE_OUT"] = "profile.jsonl",
        };
        var chosen = ProfileOptions.From(environment.GetValueOrDefault, new ProfileOptions { WarmRuns = 2 });
        Check(chosen is { Cold: false, WarmRuns: 5, Statistic: ProfileStatistic.Min, Format: "json", Output: "profile.jsonl" } && chosen.Probes!.SequenceEqual(["cpu", "memory"]),
            $"from the environment: {chosen}");
        Check(ProfileOptions.From(n => n == "IDRAK_PROFILE_PROBES" ? "all" : null, new ProfileOptions { Probes = ["cpu"] }).Probes is null
              && ProfileOptions.From(_ => null, null) == new ProfileOptions(), "all probes; nothing set keeps the defaults");
        string bad = "";
        try
        {
            _ = ProfileOptions.From(n => n == "IDRAK_PROFILE_WARM" ? "zero" : null, null);
        }
        catch (ArgumentException e)
        {
            bad = e.Message;
        }

        Check(bad.Contains("IDRAK_PROFILE_WARM=zero", StringComparison.Ordinal), $"a bad value named: {bad}");
    }
}
