// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Text;

namespace Idrak.Cli.Commands.Measure;

/// <summary>
/// <c>idrak bench [MODEL]</c> (<c>idrak b</c>): with a model, prompt and generation speed in tokens per second, the
/// matching GFLOP/s and the memory used on the device; without one, the kernel benchmarks (<see cref="KernelBench"/>).
/// <c>--devices</c> runs the same benchmark on several devices and <c>--matrix</c> across weight and KV formats, in
/// one table. <c>--save NAME</c> keeps the results under the cache folder and <c>--compare NAME</c> shows the change.
/// The kernel benchmarks report each measurement's first call (cold) apart from its warm value, the median of the
/// timed calls after a warm-up that settles (<see cref="BenchTiming"/>; <c>--warmup</c>, <c>--settle</c> and their
/// variables).
/// </summary>
internal sealed class BenchCommand : Command
{
    private static readonly string[] MatrixWeights = ["float32", "int8", "int4", "bf16"];
    private static readonly string[] MatrixKv = ["float32", "int8", "bfloat16"];

    public override string Name => "bench";

    public override IReadOnlyCollection<string> Aliases => ["b"];

    public override string Summary => "Speed of a model (tokens per second, GFLOP/s, memory) or of the kernels on a device";

    public override string Usage =>
        "[MODEL] [options]\n\n" +
        "With MODEL (a Hugging Face id, a folder, a GGUF file or an alias): prompt and generation tokens per second,\n" +
        "GFLOP/s (2 x parameters per token) and the memory used on the device. Without one: the kernel benchmarks,\n" +
        "each with its first call (Cold: what a first token pays) apart from its warm value: the median of the timed\n" +
        "calls after a warm-up that runs rounds until the last few agree and the JIT has nothing left to compile.\n\n" +
        "Options:\n" +
        "  -w, --weights FORMAT  int8, int4, bf16 or a registered packed format (the model's weights; the gemv kernels)\n" +
        "      --head-weights F  the output head's format beside -w (bf16: the checkpoint's own scores, less memory, slower)\n" +
        "  -k, --kv FORMAT       the key/value cache format (float32, int8, bfloat16 or a registered one)\n" +
        "      --context N       the model's context length\n" +
        "      --adapter DIR     merge a LoRA or DoRA adapter first\n" +
        "      --prompt-tokens N prompt length (default 512, at most the context)\n" +
        "      --tokens N        tokens to generate (default 128)\n" +
        "      --kernels LIST    without a model: matmul,gemv,attention (default all)\n" +
        "      --small           without a model: small shapes (slow devices, quick checks)\n" +
        "  -n, --repeat N        runs per measurement (default 3 with a model, 10 for kernels); the median is kept\n" +
        "      --warmup TIME     without a model: the longest warm-up of one measurement (default 2s; 0: none\n" +
        "                        beyond the cold call; IDRAK_BENCH_WARMUP)\n" +
        "      --warmup-calls N  without a model: the most warm-up calls of one measurement (default 100000;\n" +
        "                        IDRAK_BENCH_WARMUP_CALLS)\n" +
        "      --settle PCT      without a model: the warm-up stops once its last rounds' medians agree within PCT\n" +
        "                        percent and the last round compiled no method (default 5; IDRAK_BENCH_SETTLE)\n" +
        "      --settle-rounds N without a model: how many rounds in a row must agree (default 3; IDRAK_BENCH_SETTLE_ROUNDS)\n" +
        "      --warmup-round T  without a model: the shortest warm-up round, at least one call (default 100ms;\n" +
        "                        IDRAK_BENCH_WARMUP_ROUND)\n" +
        "      --devices LIST    the same benchmark on several devices in one table: 'all' (every listed device)\n" +
        "                        or names, e.g. cpu,vulkan:0 (instead of --device)\n" +
        "      --matrix          with a model: every weight format x KV format in one table (float32, int8, int4, bf16\n" +
        "                        x float32, int8, bfloat16; or the comma-separated lists given to -w and -k)\n" +
        "      --save NAME       keep the results as NAME under the cache folder (bench/NAME.json)\n" +
        "      --compare NAME    show the change against the results saved as NAME\n\n" +
        "Examples:\n" +
        "  idrak bench org/model -d vulkan:0 -w int8\n" +
        "  idrak b mymodel --save laptop && idrak b mymodel --compare laptop\n" +
        "  idrak bench --kernels matmul,gemv -d cuda:0 -j\n" +
        "  idrak bench --devices cpu --warmup 5s --settle 2 -n 30\n" +
        "  idrak b mymodel --devices all\n" +
        "  idrak b mymodel --matrix -w int8,int4 -d vulkan:0";

    public override IReadOnlyCollection<string> ValueOptions =>
        [.. ModelChoices.ValueOptions, "--prompt-tokens", "--tokens", "--kernels", "--repeat", "--save", "--compare", "--devices", .. WarmupOptions];

    // The kernel benchmarks' warm-up options (BenchSettle).
    private static readonly string[] WarmupOptions = ["--warmup", "--warmup-calls", "--settle", "--settle-rounds", "--warmup-round"];

    public override IReadOnlyCollection<string> Flags => ["--small", "--matrix"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } =
        new Dictionary<string, string>(ModelChoices.ShortForms) { ["-n"] = "--repeat" };

    /// <summary>One benchmark run: a device, and with a model its weight and KV formats (null: as stored, float32).</summary>
    private sealed record BenchRun(Device Device, string? Weights, string? Kv)
    {
        public string Label(bool devices, bool matrix) => string.Join(" ", new[]
        {
            devices ? Device.ToString() : null,
            matrix ? $"w {Weights ?? "float32"}" : null,
            matrix ? $"kv {Kv ?? "float32"}" : null,
        }.OfType<string>());
    }

    public override int Run(CommandContext context)
    {
        if (context.Positional.Count > 1)
        {
            throw new UsageException($"bench takes at most one model, not {context.Positional.Count} arguments.");
        }

        string? model = context.Positional.Count == 1 ? context.Positional[0] : null;
        if (model is not null && (context.Option("--kernels") is not null || context.Flag("--small")))
        {
            throw new UsageException("--kernels and --small choose the kernel benchmarks, which run without a model.");
        }

        if (model is not null && WarmupOptions.FirstOrDefault(o => context.Option(o) is not null) is { } warmup)
        {
            throw new UsageException($"{warmup} sets the kernel benchmarks' warm-up, which run without a model (a model's runs follow one warm-up generation).");
        }

        if (model is null && context.Flag("--matrix"))
        {
            throw new UsageException("--matrix compares a model's weight and KV formats; name a model (the kernel benchmarks already time every packed format).");
        }

        if (model is null && (context.Options("--adapter").Count > 0 || context.Option("--context") is not null || context.Option("--kv") is not null))
        {
            throw new UsageException("--kv, --context and --adapter apply to a model; name one, or leave them out for the kernel benchmarks.");
        }

        if (context.Option("--devices") is not null && context.Option("--device") is not null)
        {
            throw new UsageException("Choose --device (one device) or --devices (several), not both.");
        }

        string? compareName = context.Option("--compare");
        JsonObject? saved = compareName is null ? null : LoadSaved(context, compareName);
        var runs = Runs(context, model);
        bool manyDevices = runs.Select(r => r.Device).Distinct().Count() > 1, matrix = context.Flag("--matrix");
        var document = new JsonObject
        {
            ["kind"] = model is null ? "kernels" : "model",
            ["model"] = model,
            ["date"] = DateTimeOffset.UtcNow.ToString("o", CultureInfo.InvariantCulture),
        };
        if (model is null)
        {
            document["timing"] = BenchTiming.Method;
        }
        if (runs.Count == 1)
        {
            document["device"] = runs[0].Device.ToString();
            document["device_name"] = runs[0].Device.Name;
        }

        var all = new List<(string Label, BenchResult Result)>();
        var runsJson = new JsonArray();
        int failed = 0;
        foreach (var run in runs)
        {
            string label = run.Label(manyDevices, matrix);
            var runDocument = new JsonObject { ["device"] = run.Device.ToString() };
            try
            {
                runDocument["device_name"] = run.Device.Name;
                var results = model is null ? BenchKernels(context, run, runDocument) : BenchModel(context, model, run, runDocument);
                all.AddRange(results.Select(r => (label, r)));
                runDocument["results"] = new JsonArray([.. results.Select(r => (JsonNode)ToJson(r, label))]);
            }
            catch (Exception e) when (runs.Count > 1 && e is not (UsageException or OutOfMemoryException))
            {
                // One device or format that cannot run does not stop the others.
                failed++;
                runDocument["error"] = e.Message;
                context.Error($"{(label.Length > 0 ? label : run.Device.ToString())}: {e.Message}");
            }

            if (runs.Count > 1)
            {
                runDocument["label"] = label;
                runsJson.Add(item: runDocument);
            }
            else
            {
                foreach (var (key, value) in runDocument.ToList())
                {
                    runDocument.Remove(key);
                    document[key] = value;
                }
            }
        }

        if (runs.Count > 1)
        {
            document["runs"] = runsJson;
            document["results"] = new JsonArray([.. all.Select(p => (JsonNode)ToJson(p.Result, p.Label))]);
            WritePivot(context, runs.Select(r => r.Label(manyDevices, matrix)).ToList(), all);
        }
        else
        {
            context.Write("");
            if (all.Any(p => p.Result.Cold is not null))
            {
                context.Table(["Measurement", "Value", "Unit", "Cold", "Detail"],
                    all.Select(p => (IReadOnlyList<string>)[p.Result.Name, Format(p.Result.Value), p.Result.Unit, p.Result.Cold is { } cold ? Format(cold) : "", p.Result.Note ?? ""]));
            }
            else
            {
                context.Table(["Measurement", "Value", "Unit", "Detail"],
                    all.Select(p => (IReadOnlyList<string>)[p.Result.Name, Format(p.Result.Value), p.Result.Unit, p.Result.Note ?? ""]));
            }
        }

        if (saved is not null)
        {
            document["compare"] = Compare(context, compareName!, saved, all, (string?)document["timing"]);
        }

        if (context.Option("--save") is { } saveName)
        {
            string path = SavedPath(context, saveName);
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, document.ToJsonString(CommandContext.JsonOutput));
            document["saved"] = path;
            context.Write($"\nSaved as '{saveName}' ({path}); compare later with --compare {saveName}.");
        }

        context.WriteJson(document);
        return failed == runs.Count ? ExitCodes.Failed : ExitCodes.Ok;
    }

    // The devices (--devices or the context's device) times, with --matrix, the weight and KV formats.
    private static List<BenchRun> Runs(CommandContext context, string? model)
    {
        List<Device> devices = context.Option("--devices") is not { } list ? [context.Device]
            : list.Trim().Equals("all", StringComparison.OrdinalIgnoreCase) ? [.. Device.Available]
            : [.. list.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(ParseDevice).Distinct()];
        if (devices.Count == 0)
        {
            throw new UsageException("--devices needs 'all' or device names, e.g. cpu,vulkan:0.");
        }

        if (model is null || !context.Flag("--matrix"))
        {
            return [.. devices.Select(d => new BenchRun(d, null, null))];
        }

        static List<string> Formats(string? given, string[] fallback) =>
            given is null ? [.. fallback] : [.. given.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)];
        var weights = Formats(context.Option("--weights"), MatrixWeights);
        var kvs = Formats(context.Option("--kv"), MatrixKv);
        return [.. from d in devices from w in weights from k in kvs select new BenchRun(d, w, k)];
    }

    private static Device ParseDevice(string name)
    {
        try
        {
            return Device.Parse(name);
        }
        catch (Exception e) when (e is ArgumentException or InvalidOperationException or NotSupportedException)
        {
            throw new UsageException($"--devices: {e.Message} ('idrak devices' lists them).");
        }
    }

    // Loads the model on the run's device, then times the prompt and the generation (median of the runs) and reads the device's memory.
    private static List<BenchResult> BenchModel(CommandContext context, string name, BenchRun run, JsonObject document)
    {
        var choice = ModelChoices.Choose(context, name);
        if (context.Flag("--matrix"))
        {
            choice = choice with { Weights = run.Weights is "float32" or "f32" ? null : run.Weights, Kv = run.Kv };
        }

        int repeats = Positive(context, "--repeat", 3);
        int tokens = Positive(context, "--tokens", 128);
        var device = run.Device;
        long before = ComputeResources.GetMemoryUsage(device).InUse;
        using var model = ModelChoices.Load(context, choice, device);
        long weights = Math.Max(0, ComputeResources.GetMemoryUsage(device).InUse - before);
        var tokenizer = model.Tokenizer ?? throw new InvalidOperationException($"{choice.Model} has no tokenizer (tokenizer.json), so no prompt can be made for it.");
        int promptTokens = Math.Min(Positive(context, "--prompt-tokens", 512), Math.Max(1, model.MaxPositions - tokens - 1));
        if (promptTokens + tokens >= model.MaxPositions)
        {
            tokens = Math.Max(1, model.MaxPositions - promptTokens - 1);
        }

        string prompt = Prompt(tokenizer, promptTokens);
        string kv = choice.Kv ?? "float32";
        var generator = model.CreateGenerator(KeyValueLayouts.Get(kv), promptTokens + tokens + 1);
        generator.KeepCache = false;                                    // every run reads its whole prompt again
        var options = new GenerationOptions
        {
            TopK = 1, Temperature = 1f, RepeatPenalty = 1f, Seed = 0, NumPredict = tokens, NumCtx = promptTokens + tokens + 1,
        };

        context.Write(string.Create(CultureInfo.InvariantCulture, $"{choice.Model} on {device} ({device.Name}): {model.Spec.ParameterCount / 1e6:F1}M parameters, weights {choice.Weights ?? "as stored"}, KV cache {kv}"));
        context.Write($"  prompt {promptTokens} tokens, {tokens} generated, {repeats} run(s) after a warm-up");
        generator.Generate(prompt, options with { NumPredict = Math.Min(4, tokens) });     // warm-up: kernels built, choices measured
        var prompts = new List<double>();
        var generated = new List<double>();
        int measuredPrompt = 0, measuredGenerated = 0;
        using var progress = new ProgressLine(context, $"runs on {device}", repeats);
        for (int i = 0; i < repeats; i++)
        {
            var stats = generator.Generate(prompt, options).Stats;
            progress.Report(i + 1);
            measuredPrompt = stats.PromptTokens;
            measuredGenerated = stats.GeneratedTokens;
            prompts.Add(stats.PromptDuration.TotalSeconds > 0 ? stats.PromptTokens / stats.PromptDuration.TotalSeconds : 0);
            generated.Add(stats.TokensPerSecond);
            progress.Erase();
            context.Detail(string.Create(CultureInfo.InvariantCulture, $"  run {i + 1}: prompt {prompts[^1]:F1} tokens/s, generation {generated[^1]:F1} tokens/s"));
        }

        progress.Clear();

        var memory = ComputeResources.GetMemoryUsage(device);
        double promptRate = Median(prompts), generationRate = Median(generated);
        double flopsPerToken = 2.0 * model.Spec.ActiveParameterCount;                  // with experts, only those a token goes through
        document["settings"] = new JsonObject
        {
            ["weights"] = choice.Weights,
            ["kv"] = kv,
            ["prompt_tokens"] = measuredPrompt,
            ["generated_tokens"] = measuredGenerated,
            ["repeat"] = repeats,
            ["parameters"] = model.Spec.ParameterCount,
        };
        document["memory"] = new JsonObject
        {
            ["weights_bytes"] = weights,
            ["in_use_bytes"] = memory.InUse,
            ["cached_bytes"] = memory.Cached,
            ["limit_bytes"] = memory.Limit,
        };
        return
        [
            new BenchResult("prompt", promptRate, "tokens/s", true, $"{measuredPrompt} tokens"),
            new BenchResult("generation", generationRate, "tokens/s", true, $"{measuredGenerated} tokens"),
            new BenchResult("prompt GFLOP/s", promptRate * flopsPerToken / 1e9, "GFLOP/s", true, "2 x parameters per token"),
            new BenchResult("generation GFLOP/s", generationRate * flopsPerToken / 1e9, "GFLOP/s", true, "2 x parameters per token"),
            new BenchResult("weights memory", weights / 1048576.0, "MiB", false, "on the device after loading"),
            new BenchResult("memory in use", memory.InUse / 1048576.0, "MiB", false,
                memory.Limit is { } limit ? string.Create(CultureInfo.InvariantCulture, $"of {limit / 1048576.0:F0} MiB") : "after the runs"),
        ];
    }

    private static List<BenchResult> BenchKernels(CommandContext context, BenchRun run, JsonObject document)
    {
        var kernels = (context.Option("--kernels") ?? string.Join(',', KernelBench.Names))
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(k => k.ToLowerInvariant()).ToList();
        string? unknown = kernels.FirstOrDefault(k => !KernelBench.Names.Contains(k));
        if (kernels.Count == 0 || unknown is not null)
        {
            throw new UsageException($"--kernels takes {string.Join(", ", KernelBench.Names)} (comma-separated){(unknown is null ? "" : $", not '{unknown}'")}.");
        }

        int repeats = Positive(context, "--repeat", 10);
        var settle = Settle(context);
        bool small = context.Flag("--small");
        IReadOnlyList<string> formats = context.Option("--weights") is { } given
            ? [.. given.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(f => f.ToLowerInvariant() is "bf16" ? "bfloat16" : f.ToLowerInvariant())]
            : ["int8", "int4", "bfloat16"];
        var device = run.Device;
        context.Write(Messages.T("kernels on {0} ({1}): {2}; the median of {3} timed calls each", device, device.Name, string.Join(", ", kernels), repeats)
            + (small ? Messages.T(", small shapes") : ""));
        context.Write(Messages.T("  warm-up: until {0} rounds of at least {1} agree within {2:0.##}% and nothing is compiled, at most {3} or {4} calls; Cold: the first call",
            settle.Rounds, Span(settle.Round), settle.Tolerance * 100, Span(settle.Budget), settle.MaxCalls));
        List<BenchResult> results;
        using (var progress = new ProgressLine(context, "kernels"))
        {
            results = KernelBench.Run(device, kernels, repeats, small, formats, settle, step =>
            {
                progress.Erase();
                context.Detail($"  {step}");
                progress.Label = step;
                progress.Advance();
            }, result =>
            {
                progress.Erase();
                // The JIT's share of the first call: .NET compiling the call's path, apart from the device's work.
                string jit = result.ColdJitMs is { } jitMs
                    ? Messages.T(" ({0} ms of it compiling {1} methods)", Format(jitMs), result.ColdJitMethods ?? 0)
                    : "";
                context.Detail(result.Settled is true
                    ? Messages.T("    {0} {1} warm, {2} cold{3}; settled after {4} warm-up calls", Format(result.Value), result.Unit, Format(result.Cold ?? 0), jit,
                        result.WarmupCalls ?? 0)
                    : Messages.T("    {0} {1} warm, {2} cold{3}; not settled after {4} warm-up calls (the budget ran out)", Format(result.Value), result.Unit,
                        Format(result.Cold ?? 0), jit, result.WarmupCalls ?? 0));
            });
            progress.Clear();
        }

        int unsettled = results.Count(r => r.Settled is false);
        if (unsettled > 0)
        {
            context.Write(Messages.T("  {0} of {1} measurements did not settle within the warm-up budget (a busy machine?); raise --warmup or --settle", unsettled, results.Count));
        }

        document["settings"] = new JsonObject
        {
            ["kernels"] = new JsonArray([.. kernels.Select(k => (JsonNode)k)]),
            ["repeat"] = repeats,
            ["statistic"] = BenchTiming.Statistic,
            ["warmup"] = new JsonObject
            {
                ["settle_percent"] = settle.Tolerance * 100,
                ["rounds"] = settle.Rounds,
                ["round_ms"] = settle.Round.TotalMilliseconds,
                ["budget_ms"] = settle.Budget.TotalMilliseconds,
                ["max_calls"] = settle.MaxCalls,
            },
            ["small"] = small,
            ["packed_formats"] = new JsonArray([.. formats.Select(f => (JsonNode)f)]),
        };
        return results;
    }

    // The kernel benchmarks' warm-up: each option, else its environment variable, else the default.
    private static BenchSettle Settle(CommandContext context)
    {
        var fallback = BenchSettle.Default;
        (string Text, string From)? Given(string option, string variable) =>
            context.Option(option) is { } text ? (text, option)
            : Environment.GetEnvironmentVariable(variable) is { Length: > 0 } value && value.Trim().Length > 0 ? (value.Trim(), variable) : null;
        double Number(string option, string variable, double least, double otherwise) => Given(option, variable) is not { } given ? otherwise
            : double.TryParse(given.Text, NumberStyles.Float, CultureInfo.InvariantCulture, out double value) && value >= least && double.IsFinite(value) ? value
            : throw new UsageException(string.Create(CultureInfo.InvariantCulture, $"{given.From} needs a number of at least {least}, not '{given.Text}'."));
        int Whole(string option, string variable, int least, int otherwise) => Given(option, variable) is not { } given ? otherwise
            : int.TryParse(given.Text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) && value >= least ? value
            : throw new UsageException($"{given.From} needs a whole number of at least {least}, not '{given.Text}'.");
        TimeSpan Time(string option, string variable, TimeSpan otherwise) =>
            Given(option, variable) is { } given ? CommandContext.ParseDuration(given.Text, given.From) : otherwise;

        double percent = Number("--settle", BenchSettle.SettleVariable, 0, fallback.Tolerance * 100);
        if (percent <= 0)
        {
            throw new UsageException($"--settle ({BenchSettle.SettleVariable}) needs a percentage above 0, not {percent.ToString(CultureInfo.InvariantCulture)}.");
        }

        return new BenchSettle(percent / 100,
            Whole("--settle-rounds", BenchSettle.RoundsVariable, 1, fallback.Rounds),
            Time("--warmup-round", BenchSettle.RoundVariable, fallback.Round),
            Time("--warmup", BenchSettle.BudgetVariable, fallback.Budget),
            Whole("--warmup-calls", BenchSettle.CallsVariable, 0, fallback.MaxCalls));
    }

    // A short duration as typed back to --warmup: "100ms", "2s", "1.5s".
    private static string Span(TimeSpan time) => time.TotalSeconds >= 1
        ? string.Create(CultureInfo.InvariantCulture, $"{time.TotalSeconds:0.##}s")
        : string.Create(CultureInfo.InvariantCulture, $"{time.TotalMilliseconds:0.##}ms");

    // Several runs in one table: one row per measurement and a column per run (kernels on a few devices), or one row
    // per run and a column per measurement (a model's few measurements over many devices and formats).
    private static void WritePivot(CommandContext context, List<string> labels, List<(string Label, BenchResult Result)> all)
    {
        var names = all.Select(p => (p.Result.Name, p.Result.Unit)).Distinct().ToList();
        var byKey = all.GroupBy(p => (p.Label, p.Result.Name)).ToDictionary(g => g.Key, g => g.First().Result);
        bool cold = all.Any(p => p.Result.Cold is not null);                 // kernels: each run's cold value beside its warm one
        IEnumerable<string> Cells(string label, string name) => !byKey.TryGetValue((label, name), out var r) ? cold ? ["-", "-"] : ["-"]
            : cold ? [Format(r.Value), r.Cold is { } c ? Format(c) : "-"] : [Format(r.Value)];
        context.Write("");
        if (names.Count >= labels.Count)
        {
            context.Table(["Measurement", "Unit", .. labels.SelectMany(l => cold ? new[] { l, $"{l} cold" } : [l])],
                names.Select(n => (IReadOnlyList<string>)[n.Name, n.Unit, .. labels.SelectMany(l => Cells(l, n.Name))]));
        }
        else
        {
            context.Table(["Run", .. names.SelectMany(n => cold ? new[] { $"{n.Name} ({n.Unit})", $"{n.Name} cold ({n.Unit})" } : [$"{n.Name} ({n.Unit})"])],
                labels.Select(l => (IReadOnlyList<string>)[l, .. names.SelectMany(n => Cells(l, n.Name))]));
        }
    }

    // The change of every result that the saved run also has (matched by run label and name), as a table and as JSON.
    // Kernel results saved by the same timing method compare warm with warm and cold with cold; results saved before
    // the method was recorded (the mean after one warm-up call) still compare, with a note that the method changed.
    private static JsonObject Compare(CommandContext context, string name, JsonObject saved, List<(string Label, BenchResult Result)> results, string? timing)
    {
        var before = (saved["results"] as JsonArray ?? []).OfType<JsonObject>()
            .Where(r => r["name"] is not null && r["value"] is not null)
            .GroupBy(r => Key((string?)r["run"] ?? "", (string)r["name"]!))
            .ToDictionary(g => g.Key, g => (Value: (double)g.First()["value"]!, Cold: g.First()["cold"] is JsonValue c && c.TryGetValue(out double cold) ? cold : (double?)null));
        string? savedTiming = (string?)saved["timing"] ?? ((string?)saved["kind"] == "kernels" ? "mean-after-one-warmup" : null);
        bool sameTiming = savedTiming == timing;
        var rows = new JsonArray();
        var table = new List<IReadOnlyList<string>>();
        static string Change(double? change, bool? better) => change is null ? ""
            : string.Create(CultureInfo.InvariantCulture, $"{change.Value:+0.0%;-0.0%;0.0%}{(better is null ? "" : better.Value ? " better" : " worse")}");
        bool anyCold = false;
        foreach (var (label, r) in results)
        {
            if (!before.TryGetValue(Key(label, r.Name), out var old))
            {
                continue;
            }

            double? change = old.Value != 0 ? (r.Value - old.Value) / old.Value : null;
            bool? better = change is null || Math.Abs(change.Value) < 0.03 ? null : (change > 0) == r.HigherIsBetter;
            double? coldChange = r.Cold is { } cold && old.Cold is { } savedCold && savedCold != 0 ? (cold - savedCold) / savedCold : null;
            bool? coldBetter = coldChange is null || Math.Abs(coldChange.Value) < 0.03 ? null : (coldChange > 0) == r.HigherIsBetter;
            anyCold |= coldChange is not null;
            string shown = label.Length > 0 ? $"{label}: {r.Name}" : r.Name;
            rows.Add(item: new JsonObject
            {
                ["name"] = r.Name,
                ["run"] = label.Length > 0 ? label : null,
                ["value"] = r.Value,
                ["saved"] = old.Value,
                ["unit"] = r.Unit,
                ["change"] = change,
                ["better"] = better,
                ["cold"] = r.Cold,
                ["saved_cold"] = old.Cold,
                ["cold_change"] = coldChange,
            });
            table.Add([shown, Format(r.Value), Format(old.Value), r.Unit, Change(change, better), Change(coldChange, coldBetter)]);
        }

        context.Write($"\nAgainst '{name}' ({saved["device"] ?? "several devices"}, {saved["date"]}):");
        if (!sameTiming && timing is not null)
        {
            context.Write(Messages.T("  '{0}' was timed another way ({1}, now {2}): a change includes the method's", name, savedTiming ?? "unknown", timing));
        }

        if (table.Count == 0)
        {
            context.Write("  no measurement in common (another model, device or set of kernels?)");
        }
        else
        {
            context.Table(anyCold ? ["Measurement", "Now", "Saved", "Unit", "Change", "Cold change"] : ["Measurement", "Now", "Saved", "Unit", "Change"],
                anyCold ? table : table.Select(t => (IReadOnlyList<string>)[.. t.Take(5)]));
        }

        return new JsonObject
        {
            ["name"] = name, ["date"] = saved["date"]?.DeepClone(), ["timing"] = savedTiming, ["same_timing"] = sameTiming, ["results"] = rows,
        };
    }

    private static string Key(string label, string name) => label + "\n" + name;

    private static JsonObject LoadSaved(CommandContext context, string name)
    {
        string path = SavedPath(context, name);
        if (!File.Exists(path))
        {
            string folder = Path.GetDirectoryName(path)!;
            var known = Directory.Exists(folder) ? Directory.GetFiles(folder, "*.json").Select(Path.GetFileNameWithoutExtension).Order().ToList() : [];
            throw new UsageException($"No saved results named '{name}' ({path}). " + (known.Count == 0
                ? "Save some first with --save NAME."
                : $"Saved: {string.Join(", ", known)}."));
        }

        return JsonNode.Parse(File.ReadAllText(path)) as JsonObject ?? throw new InvalidDataException($"{path} is not a JSON object.");
    }

    private static string SavedPath(CommandContext context, string name)
    {
        if (name.Length == 0 || name.IndexOfAny([.. Path.GetInvalidFileNameChars(), '/', '\\']) >= 0 || name is "." or "..")
        {
            throw new UsageException($"'{name}' cannot name saved results: use letters, digits, '-' or '_'.");
        }

        return Path.Combine(context.CacheFolder, "bench", name + ".json");
    }

    // A prompt of about `count` tokens: ordinary words repeated, cut to that many tokens.
    private static string Prompt(ITokenizer tokenizer, int count)
    {
        const string Words = "The quick brown fox jumps over the lazy dog while the river runs past the old mill and the town wakes up. ";
        var text = new System.Text.StringBuilder(Words);
        while (tokenizer.Encode(text.ToString()).Count < count)
        {
            text.Append(Words);
        }

        var ids = tokenizer.Encode(text.ToString());
        return tokenizer.Decode(ids.Take(count));
    }

    private static int Positive(CommandContext context, string option, int fallback)
    {
        int value = context.IntOption(option, fallback);
        return value > 0 ? value : throw new UsageException($"{option} needs a number above 0, not {value}.");
    }

    private static double Median(List<double> values)
    {
        var sorted = values.Order().ToList();
        return sorted.Count == 0 ? 0 : sorted.Count % 2 == 1 ? sorted[sorted.Count / 2] : (sorted[sorted.Count / 2 - 1] + sorted[sorted.Count / 2]) / 2;
    }

    private static JsonObject ToJson(BenchResult r, string label) => new()
    {
        ["name"] = r.Name,
        ["run"] = label.Length > 0 ? label : null,
        ["value"] = r.Value,
        ["unit"] = r.Unit,
        ["higher_is_better"] = r.HigherIsBetter,
        ["detail"] = r.Note,
        ["cold"] = r.Cold,
        ["cold_jit_ms"] = r.ColdJitMs,
        ["cold_jit_methods"] = r.ColdJitMethods,
        ["warmup_calls"] = r.WarmupCalls,
        ["settled"] = r.Settled,
    };

    private static string Format(double value) => value.ToString(Math.Abs(value) >= 100 ? "F0" : Math.Abs(value) >= 10 ? "F1" : "F2", CultureInfo.InvariantCulture);
}
