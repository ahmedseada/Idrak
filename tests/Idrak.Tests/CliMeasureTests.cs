// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Generation;
using Idrak.LanguageModels;

// The idrak tool's Measure and Retrieval groups (bench, eval, perplexity, profile, check, tuning show, rag ...), run
// in-process through CommandLine.Run on the tiny fixture model and texts made here; speeds are not checked, only the
// output's shape.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] CliMeasureGroup =
    [
        ("cli measure: every command has help with an example and its environment; short forms are its own", d => { if (d == Device.Cpu) CliMeasureHelp(); }),
        ("cli measure: bench without a model runs the kernel benchmarks (JSON shape)", CliBenchKernels),
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
    private static (int Code, string Output, string Error) RunIdrak(string cache, params string[] args)
    {
        var output = new StringWriter();
        var error = new StringWriter();
        int code = global::Idrak.Cli.CommandLine.Run([.. args, "--cache", cache, "--config", Path.Combine(cache, "no-config.json")], output, error);
        return (code, output.ToString(), error.ToString());
    }

    private static JsonObject IdrakJson(string cache, params string[] args)
    {
        var (code, output, error) = RunIdrak(cache, [.. args, "-j"]);
        Check(code == 0, $"idrak {string.Join(' ', args)} exited with {code}: {error}{output}");
        return JsonNode.Parse(output) as JsonObject ?? throw new Exception($"not one JSON object: {output}");
    }

    private static string CliTemp()
    {
        string folder = Path.Combine(Path.GetTempPath(), "idrak-cli-measure-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(folder);
        return folder;
    }

    private static void CliMeasureHelp()
    {
        string cache = CliTemp();
        try
        {
            var group = global::Idrak.Cli.Commands.MeasureCommands.All.Concat(global::Idrak.Cli.Commands.RetrievalCommands.All).ToList();
            Check(group.Count == 10, $"10 commands, got {group.Count}");
            foreach (var command in group)
            {
                var (code, output, _) = RunIdrak(cache, ["help", .. command.Name.Split(' ')]);
                Check(code == 0 && output.Contains("Examples:") && output.Contains("Environment:") && output.Contains($"idrak {command.Name}"), $"help {command.Name}: {output}");
                foreach (var (shortForm, longForm) in command.ShortForms)
                {
                    Check(!global::Idrak.Cli.CommandContext.CommonShortForms.ContainsKey(shortForm), $"{command.Name}: {shortForm} is a common short form");
                    Check(command.ValueOptions.Contains(longForm) || command.Flags.Contains(longForm), $"{command.Name}: {shortForm} names unknown {longForm}");
                }
            }

            Check(RunIdrak(cache, "b", "--help").Output.Contains("idrak bench"), "the alias b selects bench");
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }

    private static void CliBenchKernels(Device device)
    {
        string cache = CliTemp();
        try
        {
            var json = IdrakJson(cache, "bench", "--small", "-n", "1", "-d", device.ToString());
            Check((string?)json["kind"] == "kernels" && (string?)json["device"] == device.ToString(), json.ToJsonString());
            var results = json["results"]!.AsArray();
            Check(results.Count == 4 + 6 + 6, $"4 matmul, 6 gemv and 6 attention results, got {results.Count}");
            foreach (var r in results)
            {
                Check(r!["name"] is not null && (double)r["value"]! > 0 && r["unit"] is not null, $"result {r.ToJsonString()}");
            }

            var gemv = IdrakJson(cache, "b", "--small", "-n", "1", "--kernels", "gemv", "-w", "int8", "-d", device.ToString());
            Check(gemv["results"]!.AsArray().Count == 2 && gemv["results"]!.AsArray().All(r => ((string)r!["name"]!).StartsWith("gemv int8")), gemv.ToJsonString());
            var (code, text, _) = RunIdrak(cache, "bench", "--small", "-n", "1", "--kernels", "matmul", "-d", device.ToString());
            Check(code == 0 && text.Contains("matmul 128x128x128 float32") && text.Contains("GFLOP/s"), text);
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }

    private static void CliBenchModel(Device device)
    {
        string cache = CliTemp();
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
            var (code, text, _) = RunIdrak(cache, [.. args, "--compare", "first"]);
            Check(code == 0 && text.Contains("generation") && text.Contains("Against 'first'"), text);
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }

    private static void CliBenchMatrix(Device device)
    {
        string cache = CliTemp();
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

            var (code, text, _) = RunIdrak(cache, "bench", "--small", "-n", "1", "--kernels", "matmul", "--devices", devices);
            Check(code == 0 && text.Contains("measurement") && text.Contains("matmul 256x256x256 float32"), text);
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }

    private static void CliBenchErrors()
    {
        string cache = CliTemp();
        try
        {
            Check(RunIdrak(cache, "bench", "--kernels", "nope").Code == 2, "unknown kernel");
            Check(RunIdrak(cache, "bench", "--matrix").Code == 2, "--matrix without a model");
            Check(RunIdrak(cache, "bench", "--compare", "missing").Error.Contains("No saved results"), "missing saved results");
            Check(RunIdrak(cache, "bench", MeasureModel, "--small").Code == 2, "--small with a model");
            Check(RunIdrak(cache, "bench", "-d", "cpu", "--devices", "cpu").Code == 2, "--device with --devices");
            Check(RunIdrak(cache, "bench", "--save", "../x", "--small", "--kernels", "matmul").Code == 2, "a path as a name");
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }

    private static void CliPerplexity(Device device)
    {
        string cache = CliTemp();
        try
        {
            string text = Path.Combine(cache, "text.txt");
            File.WriteAllText(text, string.Concat(Enumerable.Repeat("The town by the river has a mill. People walk to the market in the morning. ", 20)));
            var json = IdrakJson(cache, "perplexity", MeasureModel, text, "--window", "64", "-d", device.ToString());
            double perplexity = (double)json["perplexity"]!;
            Check(double.IsFinite(perplexity) && perplexity > 1 && (int)json["windows"]! > 1 && (int)json["scored_tokens"]! == (int)json["tokens"]! - (int)json["windows"]!, json.ToJsonString());
            var int8 = IdrakJson(cache, "perplexity", MeasureModel, text, "--window", "64", "-w", "int8", "-d", device.ToString());
            Check(Math.Abs((double)int8["perplexity"]! - perplexity) < 0.1 * perplexity, $"int8 {int8["perplexity"]} near float32 {perplexity}");
            Check(RunIdrak(cache, "perplexity", MeasureModel, text, "--window", "100000").Code == 2, "a window past the context");
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }

    private static void CliProfile(Device device)
    {
        string cache = CliTemp();
        try
        {
            var json = IdrakJson(cache, "profile", MeasureModel, "--prompt-tokens", "8", "-d", device.ToString());
            var layers = json["layers"]!.AsArray();
            Check(layers.Count == 5 && layers.Select(l => (string)l!["name"]!).SequenceEqual(["embed", "layers.0", "layers.1", "norm", "head"]), json["layers"]!.ToJsonString());
            Check(json["operations"]!.AsArray().Count > 0 && json["inner_layers"]!.AsArray().Count > 0 && (double)json["step_ms"]! > 0, json.ToJsonString());
            var (code, text, _) = RunIdrak(cache, "profile", MeasureModel, "--prompt-tokens", "8", "--top", "3", "-d", device.ToString());
            Check(code == 0 && text.Contains("Layers of the network") && text.Contains("Operations"), text);
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }

    private static void CliEval(Device device)
    {
        string cache = CliTemp();
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
            Check(RunIdrak(cache, "eval", MeasureModel, set, "--metric", "bleu").Code == 2, "unknown metric");
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }

    private static void CliCheck(Device device)
    {
        string cache = CliTemp();
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

                var generator = model.CreateGenerator(global::Idrak.Layers.KeyValueFormat.Float32, model.MaxPositions);
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
                        ["rendered"] = model.ChatTemplate!.Render(messages, [], null, true),
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
            var (code, output, _) = RunIdrak(cache, "check", MeasureModel, "--reference", path, "-d", device.ToString());
            Check(code == 1 && output.Contains("DIFF") && output.Contains("2 check(s) differ"), output);
            Check(RunIdrak(cache, "check", MeasureModel).Code == 2, "--reference is required");
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

        string cache = CliTemp();
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
            Check(RunIdrak(cache, "tuning", "show", "--backend", "tpu").Code == 2, "unknown backend");
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
        string cache = CliTemp();
        try
        {
            string docs = CliDocuments(cache), index = Path.Combine(cache, "towns.idx");
            var built = IdrakJson(cache, "rag", "index", docs, "-o", index, "--chunk", "8", "--overlap", "2");
            Check((int)built["documents"]! == 3 && (int)built["chunks"]! > 3 && !(bool)built["vectors"]!, built.ToJsonString());
            Check(RunIdrak(cache, "rag", "index", docs, "-o", index).Code == 2, "an existing index needs --force");

            var found = IdrakJson(cache, "rag", "search", "harbour fishing port", "--index", index, "--top", "2");
            Check((string?)found["hits"]![0]!["document"] == "towns/corin.md" && found["hits"]!.AsArray().Count == 2, found.ToJsonString());
            var (code, text, _) = RunIdrak(cache, "rag", "search", "hill", "--index", index);
            Check(code == 0 && text.Contains("towns/belle.md"), text);

            string questions = Path.Combine(cache, "questions.jsonl");
            File.WriteAllLines(questions,
            [
                """{"question": "Which town has a busy harbour?", "document": "corin.md"}""",
                """{"question": "Where are summers hot and dry?", "passage": "belle are HOT and dry"}""",
            ]);
            var quality = IdrakJson(cache, "rag", "eval", "--index", index, "--questions", questions, "--top", "3");
            Check((int)quality["questions"]! == 2 && (double)quality["hit_rate"]! == 1 && (double)quality["mrr"]! > 0.5, quality.ToJsonString());
            Check(RunIdrak(cache, "rag", "search", "x", "--index", Path.Combine(cache, "none.idx")).Code == 2, "a missing index");
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }

    private static void CliRagHybrid(Device device)
    {
        string cache = CliTemp();
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
            Check(RunIdrak(cache, "rag", "ask", "q", "--index", index).Code == 2, "--model is required");
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }
}
