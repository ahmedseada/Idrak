// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Generation;
using Idrak.Models;
using Idrak.Text;
using BenchSettle = global::Idrak.Cli.Commands.Measure.BenchSettle;
using BenchTiming = global::Idrak.Cli.Commands.Measure.BenchTiming;

// The idrak tool's Measure and Retrieval groups (bench, eval, perplexity, profile, check, tuning show, rag ...), run
// in-process through CommandLine.Run on the tiny fixture model and texts made here; speeds are not checked, only the
// output's shape.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] CliMeasureGroup =
    [
        ("cli measure: every command has help with an example and its environment; short forms are its own", d => { if (d == Device.Cpu) CliMeasureHelp(); }),
        ("cli measure: bench without a model runs the kernel benchmarks (JSON shape, cold and warm apart, the cold call's compiling, --save and --compare)", CliBenchKernels),
        ("cli measure: bench warm-up settles on a fake clock: the first call apart with the JIT's share of it, rounds until they agree and nothing compiles, the median", d => { if (d == Device.Cpu) CliBenchSettle(); }),
        ("cli measure: bench MODEL gives tokens per second, GFLOP/s and memory; --save and --compare", CliBenchModel),
        ("cli measure: bench --matrix and --devices give one run per format and device", CliBenchMatrix),
        ("cli measure: bench usage errors exit with 2", d => { if (d == Device.Cpu) CliBenchErrors(); }),
        ("cli measure: perplexity of a text, float32 and int8 weights", CliPerplexity),
        ("cli measure: profile times every layer, operation and kernel of one decoding step", CliProfile),
        ("cli measure: eval scores answers on conversations and writes them", CliEval),
        ("cli measure: check against a reference passes, and a changed reference fails", CliCheck),
        ("cli measure: tuning show reads the CPU, CUDA and Vulkan caches; --reset clears one", d => { if (d == Device.Cpu) CliTuningShow(); }),
    ];

    private static readonly (string Name, Action<Device> Run)[] CliRetrievalGroup =
    [
        ("cli retrieval: rag index, search and eval with keywords", d => { if (d == Device.Cpu) CliRagKeywords(); }),
        ("cli retrieval: rag index with a model (hybrid), search and ask", CliRagHybrid),
    ];

    private static string MeasureModel => TestData("gguf/tiny-llama-hf");

    // Runs idrak with a cache and config of its own; returns the exit code and both outputs.
    private static (int Code, string Output, string Error) MeasureIdrak(string cache, params string[] args)
    {
        var output = new StringWriter();
        var error = new StringWriter();
        int code = global::Idrak.Cli.CommandLine.Run([.. args, "--cache", cache, "--config", Path.Combine(cache, "no-config.json")], output, error);
        return (code, output.ToString(), error.ToString());
    }

    private static JsonObject IdrakJson(string cache, params string[] args)
    {
        var (code, output, error) = MeasureIdrak(cache, [.. args, "-j"]);
        Check(code == 0, $"idrak {string.Join(' ', args)} exited with {code}: {error}{output}");
        return JsonNode.Parse(output) as JsonObject ?? throw new Exception($"not one JSON object: {output}");
    }

    private static string MeasureTemp()
    {
        string folder = Path.Combine(Path.GetTempPath(), "idrak-cli-measure-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(folder);
        return folder;
    }

    private static void CliMeasureHelp()
    {
        string cache = MeasureTemp();
        try
        {
            var group = global::Idrak.Cli.Commands.MeasureCommands.All.Concat(global::Idrak.Cli.Commands.RetrievalCommands.All).ToList();
            Check(group.Count == 10, $"10 commands, got {group.Count}");
            foreach (var command in group)
            {
                var (code, output, _) = MeasureIdrak(cache, ["help", .. command.Name.Split(' ')]);
                Check(code == 0 && output.Contains("Examples:") && output.Contains("Environment (idrak help env for all):") && output.Contains($"idrak {command.Name}"), $"help {command.Name}: {output}");
                foreach (var (shortForm, longForm) in command.ShortForms)
                {
                    Check(!global::Idrak.Cli.CommandContext.CommonShortForms.ContainsKey(shortForm), $"{command.Name}: {shortForm} is a common short form");
                    Check(command.ValueOptions.Contains(longForm) || command.Flags.Contains(longForm), $"{command.Name}: {shortForm} names unknown {longForm}");
                }
            }

            Check(MeasureIdrak(cache, "b", "--help").Output.Contains("idrak bench"), "the alias b selects bench");
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }

    private static void CliBenchKernels(Device device)
    {
        string cache = MeasureTemp();
        try
        {
            var json = IdrakJson(cache, "bench", "--small", "-n", "1", "-d", device.ToString());
            Check((string?)json["kind"] == "kernels" && (string?)json["device"] == device.ToString(), json.ToJsonString());
            Check((string?)json["timing"] == BenchTiming.Method && (string?)json["settings"]!["statistic"] == "median"
                  && (double)json["settings"]!["warmup"]!["budget_ms"]! == 2000 && (int)json["settings"]!["warmup"]!["rounds"]! == 3,
                $"the timing method and the default warm-up are recorded: {json["settings"]?.ToJsonString()}");
            var results = json["results"]!.AsArray();
            Check(results.Count == 4 + 6 + 6, $"4 matmul, 6 gemv and 6 attention results, got {results.Count}");
            foreach (var r in results)
            {
                Check(r!["name"] is not null && (double)r["value"]! > 0 && r["unit"] is not null, $"result {r.ToJsonString()}");
                Check((double?)r["cold"] > 0 && (int?)r["warmup_calls"] >= 0 && (int?)r["warmup_calls"] <= BenchSettle.Default.MaxCalls && r["settled"] is not null,
                    $"each kernel result has its first call (cold) apart, its warm-up calls and whether it settled: {r.ToJsonString()}");
                Check((double?)r["cold_jit_ms"] >= 0 && (long?)r["cold_jit_methods"] >= 0,
                    $"each kernel result has the JIT's share of its first call (ms and methods): {r.ToJsonString()}");
            }

            var gemv = IdrakJson(cache, "b", "--small", "-n", "1", "--kernels", "gemv", "-w", "int8", "--warmup", "0", "-d", device.ToString());
            Check(gemv["results"]!.AsArray().Count == 2 && gemv["results"]!.AsArray().All(r => ((string)r!["name"]!).StartsWith("gemv int8")), gemv.ToJsonString());
            Check(gemv["results"]!.AsArray().All(r => (int)r!["warmup_calls"]! == 0 && (bool)r["settled"]! == false), $"--warmup 0: no warm-up beyond the cold call: {gemv.ToJsonString()}");

            JsonObject? capped = null;
            WithEnvironment(new Dictionary<string, string?> { [BenchSettle.CallsVariable] = "3" },
                () => capped = IdrakJson(cache, "bench", "--small", "-n", "1", "--kernels", "matmul", "-d", device.ToString()));
            Check((int)capped!["settings"]!["warmup"]!["max_calls"]! == 3 && capped["results"]!.AsArray().All(r => (int)r!["warmup_calls"]! <= 3),
                $"{BenchSettle.CallsVariable}=3 caps the warm-up: {capped.ToJsonString()}");

            string[] matmul = ["bench", "--small", "-n", "1", "--kernels", "matmul", "--warmup", "100ms", "-d", device.ToString()];
            var (code, text, _) = MeasureIdrak(cache, [.. matmul, "-v"]);
            Check(code == 0 && text.Contains("matmul 128x128x128 float32") && text.Contains("GFLOP/s") && text.Contains("Cold") && text.Contains("median"), text);
            Check(System.Text.RegularExpressions.Regex.IsMatch(text, @" warm, [0-9.]+ cold \([0-9.]+ ms of it compiling \d+ methods\); (not )?settled after \d+ warm-up calls"),
                $"-v: each measurement's line names the JIT's share of its first call: {text}");

            // --save keeps the method; --compare against it compares warm with warm and cold with cold, and against
            // results saved before the method was recorded (no "timing", no cold values) it says the method changed.
            IdrakJson(cache, [.. matmul, "--save", "kernels"]);
            var compared = IdrakJson(cache, [.. matmul, "--compare", "kernels"]);
            var rows = compared["compare"]!["results"]!.AsArray();
            Check(rows.Count == 4 && (bool)compared["compare"]!["same_timing"]! && rows.All(r => (double?)r!["saved_cold"] > 0 && r["cold_change"] is not null),
                $"compared with the same method, cold values included: {compared["compare"]!.ToJsonString()}");
            string savedPath = Path.Combine(cache, "bench", "kernels.json");
            var old = JsonNode.Parse(File.ReadAllText(savedPath))!.AsObject();
            old.Remove("timing");
            foreach (var r in old["results"]!.AsArray())
            {
                r!.AsObject().Remove("cold");
                r.AsObject().Remove("cold_jit_ms");
                r.AsObject().Remove("cold_jit_methods");
            }

            File.WriteAllText(Path.Combine(cache, "bench", "old.json"), old.ToJsonString());
            var (oldCode, oldText, _) = MeasureIdrak(cache, [.. matmul, "--compare", "old"]);
            Check(oldCode == 0 && oldText.Contains("was timed another way") && oldText.Contains("mean-after-one-warmup") && oldText.Contains("Against 'old'"),
                $"results saved before the timing method still compare, with a note: {oldText}");
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }

    // The kernel benchmarks' timing on a fake clock (µs ticks) and a fake JIT count: the first call is the cold one;
    // warm-up rounds of at least 1 ms run while the JIT keeps compiling and until three rounds' medians agree; the
    // timed calls' median ignores an outlier a mean would not.
    private static void CliBenchSettle()
    {
        long now = 0, compiled = 0;
        int call = 0, drains = 0, betweens = 0;
        void Run()
        {
            int i = call++;
            now += i == 0 ? 5000 : i <= 40 ? 300 : i % 7 == 0 ? 1000 : 100;    // cold, unoptimized, then optimized with an outlier every 7th call
            if (i < 45)
            {
                compiled++;                                                  // the JIT is still compiling
            }
        }

        var settle = new BenchSettle(0.05, 3, TimeSpan.FromMilliseconds(1), TimeSpan.FromSeconds(1), 10_000);
        var time = BenchTiming.Measure(Run, () => drains++, 9, settle, () => betweens++, () => now, 1e6, () => compiled, () => TimeSpan.FromMilliseconds(compiled * 0.5));
        Check(time.ColdMicros == 5000, $"the first call apart: {time.ColdMicros} µs, expected 5000");
        Check(time.ColdJitMethods == 1 && time.ColdJitMilliseconds == 0.5,
            $"the JIT's share of the first call (one method, 0.5 ms): {time.ColdJitMethods} methods, {time.ColdJitMilliseconds} ms");
        Check(time.Settled && time.WarmupCalls == 63, $"settled once three rounds agree and the JIT is quiet (call 63): settled {time.Settled}, {time.WarmupCalls} calls");
        Check(time.Micros == 100, $"the median of the timed calls (one 1000 µs outlier among 9): {time.Micros} µs, expected 100");
        Check(drains == 2 * (1 + 63 + 9) && betweens == 1 + 63 + 9, $"every call drained before and after, with the reset before it: {drains} drains, {betweens} resets");

        // A JIT that never goes quiet: the call limit, then the time budget, stop the warm-up unsettled.
        now = 0;
        call = 1;
        int busy = 0;
        var limited = BenchTiming.Measure(Run, () => { }, 3, settle with { MaxCalls = 50 }, null, () => now, 1e6, () => busy++);
        Check(!limited.Settled && limited.WarmupCalls == 50, $"stopped at the call limit: settled {limited.Settled}, {limited.WarmupCalls} calls");
        call = 1;
        long start = now;
        var budgeted = BenchTiming.Measure(Run, () => { }, 3, settle with { Budget = TimeSpan.FromMilliseconds(100) }, null, () => now, 1e6, () => busy++);
        Check(!budgeted.Settled && now - start is >= 100_000 and < 106_000, $"stopped at the budget: settled {budgeted.Settled}, {now - start} µs");
        var none = BenchTiming.Measure(Run, () => { }, 3, settle with { Budget = TimeSpan.Zero }, null, () => now, 1e6, () => compiled);
        Check(none.WarmupCalls == 0 && !none.Settled, $"a zero budget: no warm-up beyond the cold call ({none.WarmupCalls})");

        Check(BenchTiming.Agree([300, 100, 104, 103], 3, 0.05) && !BenchTiming.Agree([100, 106], 2, 0.05) && !BenchTiming.Agree([100, 100], 3, 0.05),
            "rounds agree when the last ones lie within the tolerance of the smallest, and only when there are enough");
        Check(BenchTiming.Median([5, 1, 3]) == 3 && BenchTiming.Median([4, 1, 3, 2]) == 2.5 && BenchTiming.Median([]) == 0, "median");
    }

    private static void CliBenchModel(Device device)
    {
        string cache = MeasureTemp();
        try
        {
            string[] args = ["bench", MeasureModel, "--prompt-tokens", "16", "--tokens", "8", "-n", "1", "-d", device.ToString()];
            var json = IdrakJson(cache, [.. args, "--save", "first"]);
            Check((string?)json["kind"] == "model", json.ToJsonString());
            var names = json["results"]!.AsArray().Select(r => (string)r!["name"]!).ToList();
            Check(names.SequenceEqual(["prompt", "generation", "prompt GFLOP/s", "generation GFLOP/s", "weights memory", "memory in use"]), string.Join(", ", names));
            Check(json["results"]!.AsArray().Take(4).All(r => (double)r!["value"]! > 0), "speeds above 0");
            Check((long)json["memory"]!["weights_bytes"]! > 0 && (int)json["settings"]!["generated_tokens"]! > 0, json.ToJsonString());
            Check(File.Exists(Path.Combine(cache, "bench", "first.json")), "saved under the cache folder");

            var compared = IdrakJson(cache, [.. args, "--compare", "first"]);
            Check(compared["compare"]!["results"]!.AsArray().Count == 6 && (string?)compared["compare"]!["name"] == "first", compared.ToJsonString());
            var (code, text, _) = MeasureIdrak(cache, [.. args, "--compare", "first"]);
            Check(code == 0 && text.Contains("generation") && text.Contains("Against 'first'"), text);
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }

    private static void CliBenchMatrix(Device device)
    {
        string cache = MeasureTemp();
        try
        {
            string devices = device == Device.Cpu ? "cpu" : $"cpu,{device}";
            var json = IdrakJson(cache, "bench", MeasureModel, "--prompt-tokens", "8", "--tokens", "4", "-n", "1", "--matrix", "-w", "float32,int8", "-k", "float32,int8",
                "--devices", devices);
            int expected = 4 * devices.Split(',').Length;
            var runs = json["runs"]!.AsArray();
            Check(runs.Count == expected && runs.All(r => r!["error"] is null && r["results"]!.AsArray().Count == 6), json.ToJsonString());
            Check(json["results"]!.AsArray().Count == expected * 6 && json["results"]!.AsArray().All(r => r!["run"] is not null), "every result names its run");
            Check(runs.Select(r => (string)r!["label"]!).Distinct().Count() == expected, "labels differ");

            var (code, text, _) = MeasureIdrak(cache, "bench", "--small", "-n", "1", "--kernels", "matmul", "--devices", devices);
            Check(code == 0 && text.Contains("Measurement") && text.Contains("matmul 256x256x256 float32"), text);
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }

    private static void CliBenchErrors()
    {
        string cache = MeasureTemp();
        try
        {
            Check(MeasureIdrak(cache, "bench", "--kernels", "nope").Code == 2, "unknown kernel");
            Check(MeasureIdrak(cache, "bench", "--matrix").Code == 2, "--matrix without a model");
            Check(MeasureIdrak(cache, "bench", "--compare", "missing").Error.Contains("No saved results"), "missing saved results");
            Check(MeasureIdrak(cache, "bench", MeasureModel, "--small").Code == 2, "--small with a model");
            Check(MeasureIdrak(cache, "bench", "-d", "cpu", "--devices", "cpu").Code == 2, "--device with --devices");
            Check(MeasureIdrak(cache, "bench", "--save", "../x", "--small", "--kernels", "matmul").Code == 2, "a path as a name");
            Check(MeasureIdrak(cache, "bench", "--settle", "0").Code == 2, "--settle 0");
            Check(MeasureIdrak(cache, "bench", "--warmup", "soon").Code == 2, "--warmup without a duration");
            Check(MeasureIdrak(cache, "bench", "--settle-rounds", "0").Code == 2, "--settle-rounds 0");
            Check(MeasureIdrak(cache, "bench", MeasureModel, "--warmup", "1s").Code == 2, "--warmup with a model");
            (int Code, string Output, string Error) bad = (0, "", "");
            WithEnvironment(new Dictionary<string, string?> { [BenchSettle.SettleVariable] = "lots" }, () => bad = MeasureIdrak(cache, "bench", "--kernels", "matmul"));
            Check(bad.Code == 2 && bad.Error.Contains(BenchSettle.SettleVariable), $"a bad {BenchSettle.SettleVariable} is a usage error naming it: {bad.Code} {bad.Error}");
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }

    private static void CliPerplexity(Device device)
    {
        string cache = MeasureTemp();
        try
        {
            string text = Path.Combine(cache, "text.txt");
            File.WriteAllText(text, string.Concat(Enumerable.Repeat("The town by the river has a mill. People walk to the market in the morning. ", 20)));
            var json = IdrakJson(cache, "perplexity", MeasureModel, text, "--window", "64", "-d", device.ToString());
            double perplexity = (double)json["perplexity"]!;
            Check(double.IsFinite(perplexity) && perplexity > 1 && (int)json["windows"]! > 1 && (int)json["scored_tokens"]! == (int)json["tokens"]! - (int)json["windows"]!, json.ToJsonString());
            var int8 = IdrakJson(cache, "perplexity", MeasureModel, text, "--window", "64", "-w", "int8", "-d", device.ToString());
            Check(Math.Abs((double)int8["perplexity"]! - perplexity) < 0.1 * perplexity, $"int8 {int8["perplexity"]} near float32 {perplexity}");
            Check(MeasureIdrak(cache, "perplexity", MeasureModel, text, "--window", "100000").Code == 2, "a window past the context");
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }

    private static void CliProfile(Device device)
    {
        string cache = MeasureTemp();
        try
        {
            var json = IdrakJson(cache, "profile", MeasureModel, "--prompt-tokens", "8", "-d", device.ToString());
            var layers = json["layers"]!.AsArray();
            Check(layers.Count == 5 && layers.Select(l => (string)l!["name"]!).SequenceEqual(["embed", "layers.0", "layers.1", "norm", "head"]), json["layers"]!.ToJsonString());
            Check(json["operations"]!.AsArray().Count > 0 && json["inner_layers"]!.AsArray().Count > 0 && (double)json["step_ms"]! > 0, json.ToJsonString());
            var (code, text, _) = MeasureIdrak(cache, "profile", MeasureModel, "--prompt-tokens", "8", "--top", "3", "-d", device.ToString());
            Check(code == 0 && text.Contains("Layers of the network") && text.Contains("Operations"), text);
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }

    private static void CliEval(Device device)
    {
        string cache = MeasureTemp();
        try
        {
            string set = Path.Combine(cache, "set.jsonl");
            File.WriteAllLines(set,
            [
                """{"messages": [{"role": "user", "content": "Say a word."}, {"role": "assistant", "content": "word"}]}""",
                """{"messages": [{"role": "user", "content": "What is 2 + 2?"}, {"role": "assistant", "content": "#### 4"}]}""",
                """{"messages": [{"role": "user", "content": "No answer here."}]}""",
            ]);
            string answers = Path.Combine(cache, "answers.jsonl");
            var json = IdrakJson(cache, "eval", MeasureModel, set, "--max-tokens", "4", "--metric", "f1", "-o", answers, "-d", device.ToString());
            Check((int)json["conversations"]! == 3 && (int)json["scored"]! == 2 && (string?)json["metric"] == "f1", json.ToJsonString());
            Check(json["answers"]!.AsArray().All(a => a!["reference"] is not null && (double)a["score"]! is >= 0 and <= 1), json.ToJsonString());
            Check(File.ReadAllLines(answers).Length == 2, "two answers written");
            Check(MeasureIdrak(cache, "eval", MeasureModel, set, "--metric", "bleu").Code == 2, "unknown metric");
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }

    private static void CliCheck(Device device)
    {
        string cache = MeasureTemp();
        try
        {
            // A reference made from the model itself on this device: what transformers would write, in its layout.
            var precision = MixedPrecision.Default;
            MixedPrecision.Default = MatMulPrecision.Float32;
            JsonObject reference;
            try
            {
                using var model = PretrainedModel.Load(MeasureModel, new PretrainedOptions { Device = device });
                var tokenizer = model.Tokenizer!;
                string text = "hello world, the town by the river";
                var ids = tokenizer.Encode(text);
                var messages = new List<ChatMessage> { new("user", "hi there") };
                float[] logits;
                using (var input = Tensor.From([.. ids.Select(i => (float)i)], [1, ids.Count], device))
                using (var predicted = model.Network.Predict(input))
                {
                    var all = predicted.ToArray();
                    logits = all[^model.Spec.Vocabulary..];
                }

                var generator = model.CreateGenerator(global::Idrak.Abstraction.Generation.KeyValueFormat.Float32, model.MaxPositions);
                var (generated, _, _) = generator.Generate(text, new GenerationOptions { TopK = 1, Temperature = 1f, RepeatPenalty = 1f, NumPredict = 6, NumCtx = model.MaxPositions, Seed = 0 });
                reference = new JsonObject
                {
                    ["texts"] = new JsonArray(new JsonObject { ["text"] = text, ["ids"] = new JsonArray([.. ids.Select(i => (JsonNode)i)]) }),
                    ["tools"] = new JsonArray(),
                    ["chats"] = new JsonArray(new JsonObject
                    {
                        ["messages"] = new JsonArray(new JsonObject { ["role"] = "user", ["content"] = "hi there" }),
                        ["tools"] = new JsonArray(),
                        ["add_generation_prompt"] = true,
                        ["rendered"] = model.JinjaTemplate!.Render(messages, [], null, true),
                    }),
                    ["runs"] = new JsonArray(new JsonObject
                    {
                        ["prompt"] = text,
                        ["ids"] = new JsonArray([.. ids.Select(i => (JsonNode)i)]),
                        ["logits"] = new JsonArray([.. logits.Select(v => (JsonNode)v)]),
                        ["generated"] = new JsonArray([.. Enumerable.Range(0, 6).Select(i => (JsonNode)0)]),
                        ["generated_text"] = generated,
                    }),
                };
            }
            finally
            {
                MixedPrecision.Default = precision;
            }

            string path = Path.Combine(cache, "reference.json");
            File.WriteAllText(path, reference.ToJsonString());
            var json = IdrakJson(cache, "check", MeasureModel, "--reference", path, "-d", device.ToString());
            Check((bool)json["ok"]! && (int)json["failures"]! == 0 && json["runs"]!.AsArray().Count == 1, json.ToJsonString());

            reference["chats"]![0]!["rendered"] = "something else";
            reference["texts"]![0]!["ids"]![0] = 99999;
            File.WriteAllText(path, reference.ToJsonString());
            var (code, output, _) = MeasureIdrak(cache, "check", MeasureModel, "--reference", path, "-d", device.ToString());
            Check(code == 1 && output.Contains("DIFF") && output.Contains("2 check(s) differ"), output);
            Check(MeasureIdrak(cache, "check", MeasureModel).Code == 2, "--reference is required");
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }

    private static void CliTuningShow()
    {
        if (Environment.GetEnvironmentVariable("IDRAK_TUNING_CACHE") is not null || Environment.GetEnvironmentVariable("IDRAK_CPU_TUNING_FILE") is not null
            || Environment.GetEnvironmentVariable("IDRAK_VULKAN_TUNING_CACHE") is not null)
        {
            return;                                                          // the caches are elsewhere; this test writes its own
        }

        string cache = MeasureTemp();
        try
        {
            Directory.CreateDirectory(Path.Combine(cache, "cpu"));
            File.WriteAllText(Path.Combine(cache, "cpu", "tuning.tsv"), "v2|Test CPU|AVX2|4/4|L1 32768|.NET 10|4 threads\t65536\t131072\n");
            Directory.CreateDirectory(Path.Combine(cache, "tuning", "cuda"));
            File.WriteAllText(Path.Combine(cache, "tuning", "cuda", "gpu.txt"),
                "idrak-tuning 5\ndevice: Test GPU; compute 9.0\ndriver: CUDA 13000\nlibrary: 0.1\nGemvSplits 0 1 2048 1024 0 0 0 = 16\nMatMul 1 2 3 4 5 6 7 = 2\n");
            Directory.CreateDirectory(Path.Combine(cache, "vulkan"));
            File.WriteAllText(Path.Combine(cache, "vulkan", "tuning.tsv"),
                "dev/drv/1/ac/push-descriptors\t1\tTest Vulkan\nkernels/dev/drv/1/ac/w64/subgroups/v1/Gemv 0 1 768 256 0 0 0\t3\tTest Vulkan\n");
            var json = IdrakJson(cache, "tuning", "show");
            var caches = json["caches"]!.AsArray().ToDictionary(c => (string)c!["backend"]!, c => c!["choices"]!.AsArray());
            Check(caches["cpu"].Count == 2 && caches["cuda"].Count == 2 && caches["vulkan"].Count == 2, json.ToJsonString());
            Check(caches["cuda"].Any(c => (string?)c!["choice"] == "GemvSplits 0 1 2048 1024 0 0 0" && (string?)c["value"] == "16" && ((string)c["device"]!).Contains("Test GPU")), caches["cuda"].ToJsonString());
            Check(caches["vulkan"].Any(c => (string?)c!["choice"] == "kernel w64/subgroups/v1/Gemv 0 1 768 256 0 0 0" && (string?)c["value"] == "3"), caches["vulkan"].ToJsonString());
            var vulkanOnly = IdrakJson(cache, "tuning", "show", "--backend", "vulkan");
            Check(vulkanOnly["caches"]!.AsArray().Count == 1, "--backend filters");

            var reset = IdrakJson(cache, "tuning", "show", "--reset", "--backend", "cpu");
            Check(!File.Exists(Path.Combine(cache, "cpu", "tuning.tsv")) && File.Exists(Path.Combine(cache, "vulkan", "tuning.tsv"))
                && reset["caches"]![0]!["deleted"]!.AsArray().Count == 1, reset.ToJsonString());
            Check(MeasureIdrak(cache, "tuning", "show", "--backend", "tpu").Code == 2, "unknown backend");
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }

    // Three small documents about towns, written in a folder of the test.
    private static string CliDocuments(string cache)
    {
        string docs = Path.Combine(cache, "docs");
        Directory.CreateDirectory(Path.Combine(docs, "towns"));
        foreach (var town in Towns)
        {
            File.WriteAllText(Path.Combine(docs, "towns", town.Id + ".md"), $"# {town.Id}\n\n{town.Text}\n");
        }

        File.WriteAllText(Path.Combine(docs, "skip.bin"), "not text");
        return docs;
    }

    private static void CliRagKeywords()
    {
        string cache = MeasureTemp();
        try
        {
            string docs = CliDocuments(cache), index = Path.Combine(cache, "towns.idx");
            var built = IdrakJson(cache, "rag", "index", docs, "-o", index, "--chunk", "8", "--overlap", "2");
            Check((int)built["documents"]! == 3 && (int)built["chunks"]! > 3 && !(bool)built["vectors"]!, built.ToJsonString());
            Check(MeasureIdrak(cache, "rag", "index", docs, "-o", index).Code == 2, "an existing index needs --force");

            var found = IdrakJson(cache, "rag", "search", "harbour fishing port", "--index", index, "--top", "2");
            Check((string?)found["hits"]![0]!["document"] == "towns/corin.md" && found["hits"]!.AsArray().Count == 2, found.ToJsonString());
            var (code, text, _) = MeasureIdrak(cache, "rag", "search", "hill", "--index", index);
            Check(code == 0 && text.Contains("towns/belle.md"), text);

            string questions = Path.Combine(cache, "questions.jsonl");
            File.WriteAllLines(questions,
            [
                """{"question": "Which town has a busy harbour?", "document": "corin.md"}""",
                """{"question": "Where are summers hot and dry?", "passage": "belle are HOT and dry"}""",
            ]);
            var quality = IdrakJson(cache, "rag", "eval", "--index", index, "--questions", questions, "--top", "3");
            Check((int)quality["questions"]! == 2 && (double)quality["hit_rate"]! == 1 && (double)quality["mrr"]! > 0.5, quality.ToJsonString());
            Check(MeasureIdrak(cache, "rag", "search", "x", "--index", Path.Combine(cache, "none.idx")).Code == 2, "a missing index");
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }

    private static void CliRagHybrid(Device device)
    {
        string cache = MeasureTemp();
        try
        {
            string docs = CliDocuments(cache), index = Path.Combine(cache, "towns.idx");
            var built = IdrakJson(cache, "rag", "index", docs, "-o", index, "--chunk", "12", "--overlap", "2", "-m", MeasureModel, "-d", device.ToString());
            Check((bool)built["vectors"]! && (int)built["dimensions"]! == 64, built.ToJsonString());
            var found = IdrakJson(cache, "rag", "search", "harbour of Corin", "--index", index, "--top", "3", "-d", device.ToString());
            Check((string?)found["search"] == "hybrid" && found["hits"]!.AsArray().Any(h => (string?)h!["document"] == "towns/corin.md")
                && found["hits"]!.AsArray().Any(h => h!["vector_rank"] is not null), found.ToJsonString());

            var answer = IdrakJson(cache, "rag", "ask", "Which town is a port?", "--index", index, "-m", MeasureModel, "--top", "2", "--max-tokens", "4", "-d", device.ToString());
            Check(answer["answer"] is not null && answer["passages"]!.AsArray().Count == 2 && answer["cited"] is JsonArray, answer.ToJsonString());
            Check(MeasureIdrak(cache, "rag", "ask", "q", "--index", index).Code == 2, "--model is required");
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }
}
