// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Idrak.Benches;

/// <summary>How large a bench's shapes are (IDRAK_BENCH_SIZE).</summary>
internal enum BenchSize
{
    Small,
    Medium,
    Large,
}

/// <summary>The run's settings, read once from the environment.</summary>
internal sealed record Settings(IReadOnlyList<Device> Devices, BenchSize Size, int Repeats, int Warmup, int Calls, double Seconds, string? Output, string? Model, string? Clips)
{
    public static Settings FromEnvironment()
    {
        List<Device> devices = Environment.GetEnvironmentVariable("IDRAK_DEVICES") is { Length: > 0 } chosen
            ? [.. chosen.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(Device.Parse)]
            : [.. Device.Available.Where(d => d.Type != DeviceType.Cpu)];
        if (devices.Count == 0)
        {
            devices = [Device.Cpu];
        }

        var size = Environment.GetEnvironmentVariable("IDRAK_BENCH_SIZE")?.Trim().ToLowerInvariant() switch
        {
            "small" => BenchSize.Small,
            "medium" => BenchSize.Medium,
            "large" => BenchSize.Large,
            null or "" => devices.All(d => d.Type == DeviceType.Cpu) ? BenchSize.Small : BenchSize.Medium,
            var other => throw new ArgumentException($"IDRAK_BENCH_SIZE '{other}': small, medium or large."),
        };
        return new Settings(devices, size, Int("IDRAK_BENCH_REPEATS", 5), Int("IDRAK_BENCH_WARMUP", 30), Math.Max(1, Int("IDRAK_BENCH_CALLS", 8)), Double("IDRAK_BENCH_SECONDS", 60),
            Text("IDRAK_BENCH_OUT"), Text("IDRAK_BENCH_MODEL"), Text("IDRAK_BENCH_CLIPS"));
    }

    public string Describe() => string.Create(CultureInfo.InvariantCulture,
        $"Idrak benches: {string.Join(", ", Devices)}; size {Size.ToString().ToLowerInvariant()}, {Warmup} warm-up calls then {Repeats} timed rounds of {Calls} calls, stress sections up to {Seconds:0} s{(Output is null ? "" : $", rows to {Output}")}");

    private static int Int(string name, int fallback) => Environment.GetEnvironmentVariable(name) is { Length: > 0 } s ? int.Parse(s, CultureInfo.InvariantCulture) : fallback;

    private static double Double(string name, double fallback) => Environment.GetEnvironmentVariable(name) is { Length: > 0 } s ? double.Parse(s, CultureInfo.InvariantCulture) : fallback;

    private static string? Text(string name) => Environment.GetEnvironmentVariable(name) is { Length: > 0 } s ? s : null;
}

/// <summary>One measured value: a section, a case, a device, a metric and its unit.</summary>
internal sealed record Row(string Section, string Case, string Device, string Metric, double Value, string Unit);

/// <summary>The run's rows, printed as they come and written as JSON for <c>compare</c>.</summary>
internal sealed class Report
{
    public List<Row> Rows { get; } = [];

    public void Add(string section, string @case, Device device, string metric, double value, string unit) =>
        Rows.Add(new Row(section, @case, device.ToString(), metric, value, unit));

    public void Write(string path, Settings settings)
    {
        var rows = new JsonArray();
        foreach (var row in Rows)
        {
            rows.Add(new JsonObject
            {
                ["section"] = row.Section, ["case"] = row.Case, ["device"] = row.Device, ["metric"] = row.Metric,
                ["value"] = double.IsFinite(row.Value) ? row.Value : null, ["unit"] = row.Unit,
            });
        }

        var root = new JsonObject
        {
            ["written"] = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture),
            ["size"] = settings.Size.ToString().ToLowerInvariant(),
            ["repeats"] = settings.Repeats,
            ["warmup"] = settings.Warmup,
            ["devices"] = new JsonArray([.. settings.Devices.Select(d => (JsonNode)$"{d} ({d.Name})")]),
            ["rows"] = rows,
        };
        File.WriteAllText(path, root.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
    }

    /// <summary>Prints every row the two files share with its change (lower is better for times and memory, higher for rates).</summary>
    public static int Compare(string beforePath, string afterPath)
    {
        var before = Read(beforePath);
        var after = Read(afterPath);
        Console.WriteLine($"{"section",-14} {"case",-44} {"metric",-16} {"before",14} {"after",14} {"change",9}");
        int shared = 0;
        foreach (var (key, b) in before)
        {
            if (!after.TryGetValue(key, out var a))
            {
                continue;
            }

            shared++;
            double change = b.Value == 0 ? double.NaN : (a.Value - b.Value) / Math.Abs(b.Value) * 100;
            bool higherIsBetter = a.Unit is "tok/s" or "GB/s" or "TFLOPS" or "x real time" or "steps/s";
            string mark = double.IsNaN(change) || Math.Abs(change) < 2 ? " " : (change < 0) == !higherIsBetter ? "+" : "-";
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"{a.Section,-14} {Trim(a.Case, 44),-44} {a.Metric,-16} {b.Value,14:#,0.###} {a.Value,14:#,0.###} {change,8:+0.0;-0.0;0.0}% {mark} {a.Unit}"));
        }

        Console.WriteLine($"{shared} rows compared ({before.Count} before, {after.Count} after); + better, - worse (beyond 2%)");
        return 0;
    }

    private static string Trim(string s, int width) => s.Length <= width ? s : s[..(width - 1)] + "…";

    private static Dictionary<string, Row> Read(string path)
    {
        var root = JsonNode.Parse(File.ReadAllText(path))!.AsObject();
        var rows = new Dictionary<string, Row>(StringComparer.Ordinal);
        foreach (var node in root["rows"]!.AsArray())
        {
            var o = node!.AsObject();
            var row = new Row((string)o["section"]!, (string)o["case"]!, (string)o["device"]!, (string)o["metric"]!, (double?)o["value"] ?? double.NaN, (string)o["unit"]!);
            rows[$"{row.Section}|{row.Case}|{row.Device}|{row.Metric}"] = row;
        }

        return rows;
    }
}

/// <summary>Timing and counting helpers shared by the sections.</summary>
internal static class Bench
{
    /// <summary>
    /// Runs <paramref name="action"/> once cold (compilation, first kernels, first allocations), then <paramref name="repeats"/>
    /// <paramref name="warmup"/> untimed calls, then the timed rounds, each in its own tensor scope with the device
    /// synchronized after it: the cold time and the warm median (rule 76: cold and warm apart).
    /// </summary>
    public static (double ColdMs, double WarmMs) Time(Device device, Action action, int repeats, int warmup = 0)
    {
        var backend = device.Backend;
        var watch = Stopwatch.StartNew();
        using (new TensorScope())
        {
            action();
            backend.Synchronize();
        }

        double cold = watch.Elapsed.TotalMilliseconds;
        for (int i = 0; i < warmup; i++)
        {
            // The tiered JIT optimizes a method after about thirty calls: untimed calls until then, so the warm rounds
            // measure the optimized code (rule 61), not the first tier's.
            using var scope = new TensorScope();
            action();
            backend.Synchronize();
        }

        var times = new double[Math.Max(1, repeats)];
        for (int i = 0; i < times.Length; i++)
        {
            watch.Restart();
            using (new TensorScope())
            {
                action();
                backend.Synchronize();
            }

            times[i] = watch.Elapsed.TotalMilliseconds;
        }

        Array.Sort(times);
        return (cold, times[times.Length / 2]);
    }

    /// <summary>Counts the operations (launches) <paramref name="action"/> runs, forward and backward, by name.</summary>
    public static Census Count(Action action)
    {
        var census = new Census();
        using (Telemetry.Subscribe(census))
        using (new TensorScope())
        {
            action();
        }

        return census;
    }

    /// <summary>The most bytes live at once while <paramref name="action"/> runs, beyond what was live before it.</summary>
    public static long PeakBytes(Device device, Action action)
    {
        long before = ComputeResources.GetMemoryUsage(device).InUse;
        ComputeResources.ResetPeakMemoryUsage(device);
        using (new TensorScope())
        {
            action();
            device.Backend.Synchronize();
        }

        return Math.Max(0, ComputeResources.GetMemoryUsage(device).Peak - before);
    }

    public static string Mib(long bytes) => string.Create(CultureInfo.InvariantCulture, $"{bytes / 1048576.0:F0} MiB");

    public static float[] Values(Random random, int count, float scale = 1f)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = (random.NextSingle() * 2 - 1) * scale;
        }

        return values;
    }

    public static Tensor RandomTensor(Random random, ReadOnlySpan<int> shape, Device device, float scale = 1f, bool requiresGrad = false)
    {
        int count = 1;
        foreach (int d in shape)
        {
            count *= d;
        }

        return Tensor.From(Values(random, count, scale), shape, device, requiresGrad);
    }

    public static string Line(string name, params ReadOnlySpan<(string Label, string Value)> columns)
    {
        var sb = new System.Text.StringBuilder();
        sb.Append(CultureInfo.InvariantCulture, $"   {name,-46}");
        foreach (var (label, value) in columns)
        {
            sb.Append(CultureInfo.InvariantCulture, $" {value,12} {label}");
        }

        return sb.ToString();
    }

    public static string F(double value, string format = "0.###") => value.ToString(format, CultureInfo.InvariantCulture);
}

/// <summary>Operations by name (forward and the backward steps apart), with their counts and output elements.</summary>
internal sealed class Census : ITelemetryHook
{
    private readonly Dictionary<string, (long Count, long Elements, TimeSpan Time)> _byOperation = new(StringComparer.Ordinal);

    public TelemetryLevel Levels => TelemetryLevel.Operations;

    public long Count { get; private set; }

    public long Elements { get; private set; }

    public IEnumerable<(string Operation, long Count, long Elements, TimeSpan Time)> ByCount =>
        _byOperation.OrderByDescending(p => p.Value.Count).Select(p => (p.Key, p.Value.Count, p.Value.Elements, p.Value.Time));

    public void OnOperation(in OperationCompleted e)
    {
        string key = (e.Backward ? "bwd " : "") + e.Operation;
        var t = _byOperation.GetValueOrDefault(key);
        _byOperation[key] = (t.Count + 1, t.Elements + e.Elements, t.Time + e.Duration);
        Count++;
        Elements += e.Elements;
    }
}

/// <summary>The sections by name.</summary>
internal static class Sections
{
    public static readonly string[] All = ["ops", "matmul", "train-decoder", "infer-decoder", "train-speech", "memory"];

    public static void Run(string section, Device device, Settings settings, Report report)
    {
        switch (section.ToLowerInvariant())
        {
            case "ops":
                OpsBench.Run(device, settings, report);
                break;
            case "matmul":
                OpsBench.RunProducts(device, settings, report);
                break;
            case "train-decoder":
                DecoderBench.RunTraining(device, settings, report);
                break;
            case "infer-decoder":
                DecoderBench.RunDecoding(device, settings, report);
                break;
            case "train-speech":
                SpeechBench.Run(device, settings, report);
                break;
            case "memory":
                MemoryBench.Run(device, settings, report);
                break;
            default:
                throw new ArgumentException($"unknown section {section}");
        }
    }
}
