// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json;
using Idrak;

// The profiler contract (plan 17, H3): a workload of one's own run cold and warm with the probes around it, the intrusive
// probes in runs of their own, the report summed up by the chosen statistic and written by any registered writer.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ProfilerGroup =
    [
        ("profiler: a workload's stages and values, the cold run apart, the statistic over the warm runs; cpu, heap, jit and memory around each timed run; the intrusive probes in runs of their own (operations and host fallbacks counted, kernels only where the device times them); an unknown probe refused naming the registered ones", ProfilerRuns),
        ("profiler: the table, markdown, csv and json writers; an app's probe and writer registered, used and removed; the criteria from IDRAK_PROFILE_*", ProfilerWritersAndCriteria),
        ("profiler: a watched run (Profiler.Watch) is one run with the probes that do not change it, the intrusive ones only when named (with a note), ended once from any thread, written with its notes; the jit probe records the compiling; the tuning, fallbacks, graphs and inference probes record from telemetry", ProfilerWatch),
    ];

    // One run of work watched as it runs: the watched probes around it, the kernel choices, host fallbacks and inference
    // stages the telemetry reported in it, the report written with its notes; ended once, from another thread too.
    private static void ProfilerWatch(Device device)
    {
        var watch = Profiler.Watch("a command", device);
        ProfiledProduct(watch.Run);
        Telemetry.Tuning(new TuningCompleted(device.Backend.Kind, "test product", 4, TimeSpan.FromMilliseconds(30), false));
        Telemetry.Tuning(new TuningCompleted(device.Backend.Kind, "test product", 2, TimeSpan.FromMilliseconds(10), true));
        Telemetry.Tuning(new TuningCompleted(device.Backend.Kind, "test convolution", 3, TimeSpan.FromMilliseconds(5), false));
        Telemetry.InferenceStage(InferenceStageKind.Prefill, device, Telemetry.StartStage(), 24, 0, 2, 3, 4);
        Telemetry.InferenceStage(InferenceStageKind.DecodeChunk, device, Telemetry.StartStage(), 8, 0, 2);
        Telemetry.InferenceStage(InferenceStageKind.DecodeChunk, device, Telemetry.StartStage(), 8, 0, 2);
        Telemetry.GraphRecordingFailed(new GraphRecordingFailed(device.Backend.Kind, "a training step", "Reading device memory while a graph is recorded"));
        Telemetry.GraphRecordingFailed(new GraphRecordingFailed(device.Backend.Kind, "a training step", "Reading device memory while a graph is recorded"));
        Telemetry.GraphRecordingFailed(new GraphRecordingFailed(device.Backend.Kind, "a decoding step", "out of device memory"));
        ProfileReport? other = null;
        var thread = new Thread(() => other = watch.End());                     // as Ctrl+C ends it
        thread.Start();
        thread.Join();
        var report = watch.End();
        Check(ReferenceEquals(report, other) && watch.Ended && report.Watched && report.Runs is [{ Kind: ProfileRunKind.Once, Index: 0 }],
            $"one run, ended once: {string.Join(", ", report.Runs.Select(r => r.Kind))}");
        Check(report.Probes.All(p => ProfileProbes.Get(p).Watched) && !report.Probes.Contains("bandwidth") && !report.Probes.Contains("operations")
              && report.Probes.Contains("tuning") && report.Probes.Contains("fallbacks") && report.Probes.Contains("graphs") && report.Probes.Contains("inference"),
            $"the watched probes: {string.Join(", ", report.Probes)}");
        Check(report.Notes.Any(n => n.Contains("operations", StringComparison.Ordinal) && n.Contains("IDRAK_PROFILE_PROBES", StringComparison.Ordinal)),
            $"the probes left out are named: {string.Join(" | ", report.Notes)}");
        Check(report["wall"] is { Count: 1, Value: > 0, Cold: null } && report["product"] is { Value: > 0 } && report["items"]?.Value == 64
              && report["cpu"] is { Source: "cpu" } && report["allocated"] is { Source: "heap" } && report["device peak"] is { Source: "memory" }
              && report["jit time"] is { Source: "jit", Unit: "ms", Value: >= 0 } && report["jit methods"] is { Source: "jit", Value: >= 0 } && report["jit IL"] is { Source: "jit", Unit: "KB" },
            $"the run's own values and the probes': {string.Join("; ", report.Summary.Select(s => $"{s.Name} {s.Value}"))}");
        Check(report["tunings"]?.Value == 3 && report["tuning time"] is { Value: 35, Unit: "ms" } && report["tuned operations"]?.Value == 2
              && report.Runs[0].Tables.Single(t => t.Name == "tuning") is { Rows: [[_, "test product", "2", "1", "6", "40", _], [_, "test convolution", "1", "0", "3", "5", _]] },
            $"the kernel choices by operation, nested ones within the outer time: {report["tunings"]}, {report["tuning time"]}");
        Check(report["prefill events"]?.Value == 1 && report["prefill count"]?.Value == 24 && report["prefill pieces"]?.Value == 3
              && report["decode chunk events"]?.Value == 2 && report["decode chunk count"]?.Value == 16 && report["decode rate"] is { Unit: "tok/s" }
              && report.Runs[0].Tables.Single(t => t.Name == "inference stages").Rows.Single(r => r[0] == "prefill") is [_, "1", "24", "2", "3", "4", ..],
            $"the inference stages, without waiting for the device: {string.Join("; ", report.Summary.Where(s => s.Source == "inference").Select(s => $"{s.Name} {s.Value}"))}");
        Check(report["fallback calls"] is { Unit: "calls" } && report["Prefill time"] is null, "the fallbacks recorded; the intrusive stages probe not run");
        Check(report["graph recordings failed"]?.Value == 3
              && report.Runs[0].Tables.Single(t => t.Name == "graph recordings failed") is { Rows: [[_, "a training step", "2", "Reading device memory while a graph is recorded"], [_, "a decoding step", "1", "out of device memory"]] },
            $"the failed graph recordings by work and reason, most first: {report["graph recordings failed"]}");

        var text = new StringWriter();
        report.Write(text);
        string written = text.ToString();
        Check(written.Contains("one run watched", StringComparison.Ordinal) && written.Contains("  not measured in a watched run", StringComparison.Ordinal)
              && written.Contains("measured", StringComparison.Ordinal) && !written.Contains("median", StringComparison.Ordinal) && written.Contains("tuning (2 rows)", StringComparison.Ordinal),
            $"the table of a watched run: {written}");
        var jsonText = new StringWriter();
        report.Write(jsonText, ProfileWriters.Json);
        var json = JsonDocument.Parse(jsonText.ToString()).RootElement;
        Check(json.GetProperty("options").GetProperty("watched").GetBoolean() && json.GetProperty("notes").GetArrayLength() == report.Notes.Count
              && json.GetProperty("runs")[0].GetProperty("kind").GetString() == "once", "the JSON of a watched run");

        // An intrusive probe named measures the run itself, with a note; nothing is measured after the end.
        using (var named = Profiler.Watch("named", device, new ProfileOptions { Probes = ["operations", "heap"] }))
        {
            ProfiledProduct(named.Run);
            named.Note("ended with exit code 0");
            var namedReport = named.End();
            Check(namedReport.Probes.SequenceEqual(["operations", "heap"]) && namedReport["operations"] is { Value: > 0 }
                  && namedReport.Notes.Any(n => n.Contains("include their cost", StringComparison.Ordinal)) && namedReport.Notes[^1] == "ended with exit code 0",
                $"a named intrusive probe in the watched run: {string.Join(" | ", namedReport.Notes)}");
        }

        // The host fallbacks of a device without kernels of its own, by operation, counted while the probe listens.
        if (device.Type == DeviceType.Cpu)
        {
            var minimal = Device.Get("minimal");
            long before = Idrak.Abstraction.Operations.Kernels.HostCalls(minimal.Backend);
            var fallbacks = Profiler.Watch("minimal", minimal, new ProfileOptions { Probes = ["fallbacks"] });
            ProfiledProduct(fallbacks.Run);
            var minimalReport = fallbacks.End();
            var table = minimalReport.Runs[0].Tables.Single(t => t.Name == "host fallbacks");
            Check(minimalReport["fallback calls"]?.Value == Idrak.Abstraction.Operations.Kernels.HostCalls(minimal.Backend) - before && minimalReport["fallback calls"]?.Value > 0
                  && table.Rows.Any(r => r[0] == "minimal" && r[1] == "Softmax" && long.Parse(r[2], CultureInfo.InvariantCulture) >= 1)
                  && table.Rows.Where(r => r[0] == "minimal").Sum(r => long.Parse(r[2], CultureInfo.InvariantCulture)) == minimalReport["fallback calls"]?.Value
                  && table.Columns[^1] == "why" && table.Rows.Where(r => r[0] == "minimal").All(r => r[3] == "no kernel of its own"),
                $"the host fallbacks by operation: {minimalReport["fallback calls"]}, {string.Join("; ", table.Rows.Select(r => string.Join(' ', r)))}");
        }
    }

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
        Check(report["device bandwidth"] is { Unit: "GB/s", Count: 3, Value: > 0, Source: "bandwidth" } bandwidth && bandwidth.Min == bandwidth.Max
              && Profiler.Run("again", device, _ => { }, new ProfileOptions { WarmRuns = 1, Probes = ["bandwidth"] })["device bandwidth"]?.Value == bandwidth.Value,
            $"the device's bandwidth beside every run, measured once per device: {report["device bandwidth"]}");

        // The stages probe: a workload's InferenceStage events summed in a run of its own, with the encoder's peak.
        var staged = Profiler.Run("stages", device, _ =>
        {
            Telemetry.InferenceStage(InferenceStageKind.CacheAllocation, device, Telemetry.Start(TelemetryLevel.Inference), 64, 1 << 20);
            Telemetry.InferenceStage(InferenceStageKind.CacheAllocation, device, Telemetry.Start(TelemetryLevel.Inference), 64, 1 << 20);
            using var held = Tensor.Zeros([1 << 18], device);
            Telemetry.InferenceStage(InferenceStageKind.Encoder, device, Telemetry.Start(TelemetryLevel.Inference), 1, 4096);
        }, new ProfileOptions { WarmRuns = 1, Probes = ["stages"] });
        Check(staged.Runs.Count == 3 && staged["CacheAllocation size"] is { Value: 2, Unit: "MB", Count: 1 } && staged["CacheAllocation count"]?.Value == 128
              && staged["Encoder count"]?.Value == 1 && staged["encoder peak above start"] is { Value: >= 0 } peakAbove && (device.Type != DeviceType.Cpu || peakAbove.Value >= 1),
            $"the stages in a run of their own: {string.Join("; ", staged.Summary.Select(s => $"{s.Name} {s.Value}"))}");
        foreach (string name in new[] { "cpu", "cores busy", "allocated", "gc0", "gc pause", "jit time", "jit methods", "jit IL", "device in use", "device peak", "process peak" })
        {
            Check(report[name] is { Count: 3, Source: not null }, $"{name} measured around each warm run: {report[name]}");
        }

        Check(report.Runs.Where(r => r.Kind != ProfileRunKind.Probe).All(r => r.Value("operations") is null), "a trace never runs in a timed run");
        Check(report["operations"] is { Value: > 0, Count: 1, Source: "operations" } && report["operation kinds"] is { Value: >= 2 },
            $"the operations' pass counts the calls: {report["operations"]}, {report["operation kinds"]}");
        var pass = report.Runs.Single(r => r.Probe == "operations");
        Check(pass.Tables.Any(t => t.Name == "operations" && t.Rows.Any(r => r[0] == "Softmax")), "the operations by count name Softmax");
        bool times = device.Backend is IKernelProfiler;
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
        Check(ProfileProbes.Names.SequenceEqual(["bandwidth", "cpu", "heap", "jit", "memory", "tuning", "fallbacks", "graphs", "inference", "operations", "kernels", "stages", "timings", "cluster", "network"]) && ProfileWriters.Names.SequenceEqual(["table", "markdown", "csv", "json"]),
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
        var all = ProfileOptions.From(n => n == "IDRAK_PROFILE_PROBES" ? "all" : null, new ProfileOptions { Probes = ["cpu"] }).Probes;
        Check(all is not null && all.SequenceEqual(ProfileProbes.Names) && ProfileOptions.From(_ => null, null) == new ProfileOptions(),
            $"all probes by name (a watched run takes the intrusive ones too), got {(all is null ? "null" : string.Join(", ", all))}; nothing set keeps the defaults");
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
