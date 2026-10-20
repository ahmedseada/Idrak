// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Generation;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Text;
using Idrak.Text.Abstractions;
using Idrak.Text.Plugins.Gemma;

// Plan 19: language models made from scratch (phase A1: a model from a spec, saved as a folder; A3: the pretraining loop).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] PretrainGroup =
    [
        ("pretrain: every Llama-style family writes a spec as config.json and reads the same spec back; a spec a family cannot say, and a family that reads checkpoints only, are refused", PretrainConfigRoundTrip),
        ("pretrain: a model made from a spec is a folder of the family's checkpoint (the tensors' names and shapes of transformers' own), seeded, loading and chatting like any model; the created template takes images", PretrainCreate),
        ("pretrain: making a model refuses an unknown architecture, a tokenizer larger than the vocabulary, a missing token and a folder that holds files", PretrainCreateRefusals),
        ("pretrain: a loaded model saved as a checkpoint writes transformers' tensors back value for value, LayerNorm and learned positions included, and saves over its own folder", PretrainSaveCheckpoint),
        ("pretrain: every parameter trains: the first step's loss and gradients, three SGD steps' losses and weights, and three AdamW steps' are transformers'", PretrainReferenceSteps),
        ("pretrain: a run stopped after its second step and resumed from the checkpoint gives the uninterrupted run's losses bit for bit (the optimizer's moments, the data order and the rate restored)", PretrainResume),
        ("pretrain: a stage trains only its parts, sources are drawn by their weights, records train their answers, telemetry sees every step and the loss of the tiny model falls", PretrainStagesAndSources),
        ("pretrain: options read back from JSON, and runs with no stage, no source, an unknown part, optimizer or schedule, or a packed model are refused", PretrainRefusals),
    ];

    // A token file of the reference's first four 32-token windows (129 ids), so every step trains on exactly those.
    private static string ReferenceWindows(string folder)
    {
        string path = Path.Combine(folder, "windows.bin");
        var bytes = File.ReadAllBytes(PretrainData("tokens.bin"));
        File.WriteAllBytes(path, bytes[..(129 * 2)]);
        return path;
    }

    private static PretrainingOptions TextStage(string tokens, int steps, string optimizer, float rate) => new()
    {
        Stages = [new PretrainingStage("text") { Sources = [new PretrainingSource(PretrainingSources.Tokens, [tokens])], Steps = steps, LearningRate = rate, Schedule = "constant", WarmupFraction = 0 }],
        MaxLength = 32, BatchTokens = 128, Optimizer = optimizer, WeightDecay = optimizer == "sgd" ? 0f : 0.1f, Beta1 = 0.9f, Beta2 = 0.95f, Momentum = 0f, MaxGradientNorm = 0f,
        CudaGraphs = false, KeepCheckpoints = 0,
    };

    private static void PretrainReferenceSteps(Device device)
    {
        string folder = TempFolder("pretrain-reference");
        try
        {
            string windows = ReferenceWindows(folder);
            var step = JsonNode.Parse(File.ReadAllText(PretrainData("step/step.json")))!.AsObject();
            var initial = ReadTensors(Path.Combine(TinyLlama, "model.safetensors"));

            // One SGD step: the loss, and the gradients as the weights moved (w0 - w1) / rate.
            using (var model = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = device }))
            {
                var result = Pretrainer.Train(model, TextStage(windows, 1, "sgd", 0.2f), Path.Combine(folder, "one"));
                float expected = (float)step["sgd"]!["losses"]![0]!;
                Check(Math.Abs(result.Losses[0] - expected) < 1e-4f, $"step 1's loss {result.Losses[0]:F6}, transformers' {expected:F6}");
                var after = ReadTensors(Path.Combine(result.LastCheckpoint!, "model.safetensors"));
                var grads = ReadTensors(PretrainData("step/grads-step1.safetensors"));
                float largest = grads.Values.Max(g => g.Values.Max(MathF.Abs));
                foreach (var (name, (_, g)) in grads)
                {
                    var moved = initial[name].Values.Zip(after[name].Values, (a, b) => (a - b) / 0.2f).ToArray();
                    AssertClose(g, moved, 1e-4f * largest, $"the gradient of {name}");
                }
            }

            foreach (var (kind, rate, weightsFile, tolerance) in new[] { ("sgd", 0.2f, "step/weights-step3.safetensors", 5e-5f), ("adamw", 1e-3f, "step/adamw-folded/weights-step3.safetensors", 1e-5f) })
            {
                using var model = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = device });
                var result = Pretrainer.Train(model, TextStage(windows, 3, kind, rate), Path.Combine(folder, kind));
                var expected = step[kind]!["losses"]!.AsArray().Select(l => (float)l!).ToArray();
                AssertClose(expected, [.. result.Losses], 1e-4f, $"{kind}: three steps' losses (torch's)");
                // AdamW's weights against the reference computed with Idrak's formula (the bias corrections folded into the
                // step size, so epsilon sits elsewhere than in torch's: see the fixture's README).
                var after = ReadTensors(Path.Combine(result.LastCheckpoint!, "model.safetensors"));
                foreach (var (name, (_, w)) in ReadTensors(PretrainData(weightsFile)))
                {
                    AssertClose(w, after[name].Values, tolerance, $"{kind}: {name} after three steps");
                }
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void PretrainResume(Device device)
    {
        string folder = TempFolder("pretrain-resume");
        try
        {
            var options = TextStage(PretrainData("tokens.bin"), 6, "adamw", 3e-3f) with { BatchTokens = 64, SaveEvery = 2, Seed = 5 };
            options = options with { Stages = [options.Stages[0] with { Schedule = "cosine", WarmupFraction = 0.2f }] };
            IReadOnlyList<float> whole;
            using (var model = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = device }))
            {
                whole = Pretrainer.Train(model, options, Path.Combine(folder, "whole")).Losses;
            }

            string run = Path.Combine(folder, "stopped");
            using (var model = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = device }))
            {
                using var stop = new CancellationTokenSource();
                var progress = new SynchronousProgress<PretrainingProgress>(p =>
                {
                    if (p.Step == 2)
                    {
                        stop.Cancel();
                    }
                });
                Throws<OperationCanceledException>(() => Pretrainer.Train(model, options, run, null, progress, stop.Token), "the run stops when cancelled");
            }

            string latest = Pretrainer.LatestCheckpoint(run)!;
            Check(latest.EndsWith("checkpoint-2", StringComparison.Ordinal), $"the latest checkpoint is the second step's: {latest}");
            var checkpoint = Pretrainer.ReadCheckpoint(latest);
            Check(checkpoint.Step == 2 && checkpoint.StageStep == 2 && checkpoint.Losses.Count == 2 && checkpoint.Sources[0].Drawn == 4, $"the checkpoint says where the run was: step {checkpoint.Step}, drawn {checkpoint.Sources[0].Drawn}");
            using (var model = PretrainedModel.Load(latest, new PretrainedOptions { Device = device }))
            {
                var resumed = Pretrainer.Train(model, checkpoint.Options, run, checkpoint);
                Check(resumed.Steps == 6 && resumed.Losses.Count == 6, $"{resumed.Steps} steps, {resumed.Losses.Count} losses");
                Check(resumed.Losses.SequenceEqual(whole), $"the resumed losses [{string.Join(", ", resumed.Losses.Select(l => l.ToString("R", System.Globalization.CultureInfo.InvariantCulture)))}] "
                    + $"against [{string.Join(", ", whole.Select(l => l.ToString("R", System.Globalization.CultureInfo.InvariantCulture)))}]");
            }

            Check(Directory.GetDirectories(run, "checkpoint-*").Length == 3, "a checkpoint every second step");
            using var last = PretrainedModel.Load(Pretrainer.LatestCheckpoint(run)!, new PretrainedOptions { Device = device });
            Check(last.Tokenizer is not null && last.ChatTemplate is not null, "a checkpoint is a model folder");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private sealed class StepCounter : ITelemetryHook
    {
        public int Batches;
        public int Started;

        public TelemetryLevel Levels => TelemetryLevel.Training | TelemetryLevel.Batches;

        public void OnTrainingStarted(in TrainingStarted e) => Started++;

        public void OnBatchCompleted(in BatchCompleted e) => Batches++;
    }

    private static void PretrainStagesAndSources(Device device)
    {
        string folder = TempFolder("pretrain-stages");
        try
        {
            string windows = ReferenceWindows(folder);
            var initial = ReadTensors(Path.Combine(TinyLlama, "model.safetensors"));
            using (var model = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = device }))
            {
                var options = TextStage(windows, 1, "sgd", 0.2f);
                options = options with { Stages = [options.Stages[0] with { Trained = ["embeddings"] }] };
                var after = ReadTensors(Path.Combine(Pretrainer.Train(model, options, Path.Combine(folder, "embeddings")).LastCheckpoint!, "model.safetensors"));
                Check(!after["model.embed_tokens.weight"].Values.SequenceEqual(initial["model.embed_tokens.weight"].Values), "the embeddings train");
                Check(after.Where(t => t.Key != "model.embed_tokens.weight").All(t => t.Value.Values.SequenceEqual(initial[t.Key].Values)), "nothing else does");
            }

            using (var model = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = device }))
            {
                var options = TextStage(windows, 1, "sgd", 0.2f);
                options = options with { Stages = [options.Stages[0] with { Trained = ["decoder"] }] };
                var after = ReadTensors(Path.Combine(Pretrainer.Train(model, options, Path.Combine(folder, "decoder")).LastCheckpoint!, "model.safetensors"));
                Check(after["model.embed_tokens.weight"].Values.SequenceEqual(initial["model.embed_tokens.weight"].Values)
                    && !after["model.layers.0.self_attn.q_proj.weight"].Values.SequenceEqual(initial["model.layers.0.self_attn.q_proj.weight"].Values)
                    && !after["model.norm.weight"].Values.SequenceEqual(initial["model.norm.weight"].Values), "the decoder trains, the embeddings stay");
            }

            // Two sources weighted 3 to 1: drawn 3 to 1.
            using (var model = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = device }))
            {
                string tokens = PretrainData("tokens.bin");
                var options = TextStage(tokens, 10, "adamw", 1e-3f) with { SaveEvery = 5 };
                options = options with { Stages = [options.Stages[0] with
                {
                    Steps = 10, Sources = [new PretrainingSource(PretrainingSources.Tokens, [tokens]) { Weight = 3 }, new PretrainingSource(PretrainingSources.Tokens, [tokens])],
                }] };
                string run = Path.Combine(folder, "weights");
                using var stop = new CancellationTokenSource();
                Throws<OperationCanceledException>(() => Pretrainer.Train(model, options, run, null,
                    new SynchronousProgress<PretrainingProgress>(p => { if (p.Step == 5) { stop.Cancel(); } }), stop.Token), "stopped mid-stage");
                var drawn = Pretrainer.ReadCheckpoint(Pretrainer.LatestCheckpoint(run)!).Sources;
                Check(drawn[0].Drawn == 15 && drawn[1].Drawn == 5, $"drawn {drawn[0].Drawn} and {drawn[1].Drawn} of 20 by weights 3 and 1");
            }

            // Records: conversations through the chat template, answers trained.
            string records = Path.Combine(folder, "records.jsonl");
            File.WriteAllLines(records,
            [
                """{"messages": [{"role": "user", "content": "read the scan"}, {"role": "assistant", "content": "the court decision 12/05/1446"}]}""",
                """{"messages": [{"role": "user", "content": "اقرأ"}, {"role": "assistant", "content": "وزارة العدل المحكمة"}]}""",
                """{"conversations": [{"from": "human", "value": "write"}, {"from": "gpt", "value": "a page with lines of text"}]}""",
            ]);
            using (var model = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = device }))
            {
                var options = TextStage(windows, 3, "adamw", 1e-3f) with { BatchTokens = 128 };
                options = options with { Stages = [options.Stages[0] with { Sources = [new PretrainingSource(PretrainingSources.Records, [records]) { Format = "messages" }] }] };
                File.WriteAllLines(records, File.ReadAllLines(records).Take(2));
                var result = Pretrainer.Train(model, options, Path.Combine(folder, "records"));
                Check(result.Losses.Count == 3 && result.Losses.All(float.IsFinite), $"records train: {string.Join(", ", result.Losses)}");
            }

            // Telemetry, and the loss of 200 steps on the whole token file.
            var counter = new StepCounter();
            using (Telemetry.Subscribe(counter))
            using (var model = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = device }))
            {
                var options = TextStage(PretrainData("tokens.bin"), 200, "adamw", 3e-3f) with { BatchTokens = 128, Seed = 1 };
                options = options with { Stages = [options.Stages[0] with { Schedule = "cosine", WarmupFraction = 0.05f }] };
                var result = Pretrainer.Train(model, options, Path.Combine(folder, "learn"));
                Check(counter.Started == 1 && counter.Batches == 200, $"telemetry: {counter.Started} start, {counter.Batches} steps");
                float first = result.Losses[0], end = result.Losses.TakeLast(10).Average();
                Check(end < first / 2, $"the loss falls from {first:F3} to {end:F3}");
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void PretrainRefusals(Device device)
    {
        if (!FirstRun(nameof(PretrainRefusals)))
        {
            return;
        }

        string folder = TempFolder("pretrain-refusals-run");
        try
        {
            var good = TextStage(PretrainData("tokens.bin"), 2, "adamw", 1e-3f);
            var back = PretrainingOptions.FromJson(good.ToJson());
            Check(JsonNode.DeepEquals(back.ToJson(), good.ToJson()), "options read back from JSON");
            var withUnknown = good.ToJson();
            withUnknown["epochs"] = 3;
            Throws<InvalidDataException>(() => PretrainingOptions.FromJson(withUnknown), "an unknown option");

            using var model = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = Device.Cpu });
            Throws<ArgumentException>(() => Pretrainer.Train(model, good with { Stages = [] }, folder), "no stages");
            Throws<ArgumentException>(() => Pretrainer.Train(model, good with { Stages = [good.Stages[0] with { Sources = [] }] }, folder), "no sources");
            Throws<ArgumentException>(() => Pretrainer.Train(model, good with { Stages = [good.Stages[0] with { Trained = ["wings"] }] }, folder), "an unknown part");
            Throws<ArgumentException>(() => Pretrainer.Train(model, good with { Stages = [good.Stages[0] with { Trained = ["tower"] }] }, folder), "a vision part of a text model");
            Throws<ArgumentException>(() => Pretrainer.Train(model, good with { Optimizer = "lion" }, folder), "an unknown optimizer");
            Throws<ArgumentException>(() => Pretrainer.Train(model, good with { Stages = [good.Stages[0] with { Schedule = "steps" }] }, folder), "an unknown schedule");
            Throws<NotSupportedException>(() => Pretrainer.Train(model, good with { Stages = [good.Stages[0] with { Sources = [new PretrainingSource("parquet", ["x"])] }] }, folder), "an unregistered source");
            Throws<FileNotFoundException>(() => Pretrainer.Train(model, good with { Stages = [good.Stages[0] with { Sources = [new PretrainingSource("tokens", [PretrainData("missing.bin")])] }] }, folder), "a missing file");
            using var packed = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = Device.Cpu, Int8 = true });
            Throws<InvalidOperationException>(() => Pretrainer.Train(packed, good, folder), "int8 weights");
            Check(PretrainingSources.Names.Contains("tokens") && PretrainingSources.Names.Contains("records") && PretrainingSources.Origin("tokens") == Idrak.Abstraction.Overrides.Library, "the library's source kinds");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static string TinyLlama => PretrainData("tiny-llama");

    private static DecoderSpec TinyLlamaSpec()
    {
        var config = JsonNode.Parse(File.ReadAllText(Path.Combine(TinyLlama, "config.json")))!.AsObject();
        return PretrainedArchitectures.Get("LlamaForCausalLM").Spec(config, []);
    }

    private static ModelFiles TinyFiles() => new()
    {
        Tokenizer = BpeTokenizer.Load(PretrainData("tokenizer")),
        BosToken = "<s>",
        PadToken = "<pad>",
        StopTokens = ["</s>"],
    };

    // A safetensors file's tensors: name → (shape, values).
    private static Dictionary<string, (int[] Shape, float[] Values)> ReadTensors(string path)
    {
        using var opened = SafeTensorsReader.Open(path);
        ITensorStore reader = opened;
        return reader.Names.ToDictionary(n => n, n => (reader.ShapeOf(n), reader.Read(n)), StringComparer.Ordinal);
    }

    private static void PretrainConfigRoundTrip(Device device)
    {
        if (!FirstRun(nameof(PretrainConfigRoundTrip)))
        {
            return;
        }

        GemmaPlugin.Register();                      // GemmaForCausalLM and Gemma3ForCausalLM are Idrak.Text.Plugins' family
        var llama = TinyLlamaSpec();
        var specs = new (string Architecture, DecoderSpec Spec)[]
        {
            ("LlamaForCausalLM", llama),
            ("LlamaForCausalLM", llama with { TieEmbeddings = false, QkvBias = true, OutputBias = true, Rope = new RopeSettings(500000f, 4, Scaling: new RopeScaling("llama3", new JsonObject
            {
                ["factor"] = 8.0, ["low_freq_factor"] = 1.0, ["high_freq_factor"] = 4.0, ["original_max_position_embeddings"] = 32, ["max_position_embeddings"] = 64,
            })) }),
            ("MistralForCausalLM", llama with { SlidingWindow = 16 }),
            ("Qwen2ForCausalLM", llama with { QkvBias = true, SlidingWindow = 16, SlidingWindowLayers = [false, true] }),
            ("Qwen3ForCausalLM", llama with { QkNorm = true }),
            ("GemmaForCausalLM", llama with { NormOffset = 1f, EmbeddingScale = MathF.Sqrt(llama.Dim), TieEmbeddings = true, Activation = FeedForwardActivation.Gelu }),
        };
        foreach (var (name, spec) in specs)
        {
            var family = PretrainedArchitectures.Get(name);
            Check(family.Config is not null, $"{name} writes configurations");
            var config = family.Config!(spec);
            var back = family.Spec(config, []);
            Check(JsonNode.DeepEquals(back.ToJson(), spec.ToJson()), $"{name}: the spec reads back the same\n{spec.ToJson()}\n{back.ToJson()}");
        }

        foreach (var (name, spec, what) in new (string, DecoderSpec, string)[]
        {
            ("LlamaForCausalLM", llama with { Norm = DecoderNorm.Layer }, "layer normalization"),
            ("LlamaForCausalLM", llama with { QkNorm = true }, "q/k norms in Llama"),
            ("LlamaForCausalLM", llama with { QkvBias = true, OutputBias = false }, "biases Llama's one key cannot say"),
            ("Qwen2ForCausalLM", llama, "Qwen2 without its q/k/v biases"),
        })
        {
            Throws<NotSupportedException>(() => PretrainedArchitectures.Get(name).Config!(spec), what);
        }

        Check(PretrainedArchitectures.Get("Gemma3ForCausalLM").Config is null && PretrainedArchitectures.Get("MixtralForCausalLM").Config is null,
            "families read from checkpoints only write no configuration");
        var template = JsonNode.Parse(File.ReadAllText(Path.Combine(TinyLlama, "tokenizer_config.json")))!["chat_template"]!.GetValue<string>();
        Check(template == ModelFiles.ChatMLWithParts, "the created models' template is the reference's");
    }

    private static void PretrainCreate(Device device)
    {
        string folder = TempFolder("pretrain-create");
        string again = TempFolder("pretrain-create-again");
        try
        {
            var spec = TinyLlamaSpec();
            using (var model = PretrainedModel.Create(folder, spec, "LlamaForCausalLM", TinyFiles(), new CreateOptions { Device = device, Seed = 7 }))
            {
                var expected = ReadTensors(Path.Combine(TinyLlama, "model.safetensors"));
                var created = ReadTensors(Path.Combine(folder, "model.safetensors"));
                Check(created.Keys.Order(StringComparer.Ordinal).SequenceEqual(expected.Keys.Order(StringComparer.Ordinal)),
                    $"tensor names as transformers': {string.Join(", ", created.Keys.Except(expected.Keys))} | {string.Join(", ", expected.Keys.Except(created.Keys))}");
                Check(created.All(t => t.Value.Shape.SequenceEqual(expected[t.Key].Shape)), "tensor shapes as transformers'");
                var embedding = created["model.embed_tokens.weight"].Values;
                double std = Math.Sqrt(embedding.Select(v => (double)v * v).Average());
                Check(Math.Abs(std - 0.02) < 0.002, $"weights drawn at the asked deviation ({std:F4})");
                Check(created["model.norm.weight"].Values.All(v => v == 1f), "norm gains start at one");

                var config = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "config.json")))!.AsObject();
                Check((string?)config["architectures"]?[0] == "LlamaForCausalLM" && config["idrak_spec"] is JsonObject && (string?)config["torch_dtype"] == "float32", "config.json names the family and keeps the spec");
                Check(JsonNode.DeepEquals(model.Spec.ToJson(), spec.ToJson()), "the loaded model has the spec");
                Check(model.Tokenizer!.Encode("رقم 12 page").SequenceEqual(TinyFiles().Tokenizer.Encode("رقم 12 page")), "the tokenizer is the one given");
                Check(model.ChatTemplate is JinjaChatTemplate jinja && jinja.PartKinds.Contains(ChatParts.Image), "the created template takes image parts");
                var chat = model.CreateChat(KeyValueFormat.Float32, 64);
                var answer = chat.Chat(new ChatRequest([new ChatMessage("user", "read the scan")], Options: new GenerationOptions { Temperature = 0, NumPredict = 4 }));
                Check(answer.Done, "a created model chats (nonsense, as expected)");
            }

            using (var same = PretrainedModel.Create(again, TinyLlamaSpec(), "LlamaForCausalLM", TinyFiles(), new CreateOptions { Device = device, Seed = 7 }))
            {
                Check(File.ReadAllBytes(Path.Combine(again, "model.safetensors")).AsSpan().SequenceEqual(File.ReadAllBytes(Path.Combine(folder, "model.safetensors"))),
                    "the same seed draws the same weights");
            }

            using (var other = PretrainedModel.Create(again, TinyLlamaSpec(), "LlamaForCausalLM", TinyFiles(), new CreateOptions { Device = device, Seed = 8, Overwrite = true }))
            {
                Check(!File.ReadAllBytes(Path.Combine(again, "model.safetensors")).AsSpan().SequenceEqual(File.ReadAllBytes(Path.Combine(folder, "model.safetensors"))),
                    "another seed draws other weights");
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
            Directory.Delete(again, recursive: true);
        }
    }

    private static void PretrainCreateRefusals(Device device)
    {
        if (!FirstRun(nameof(PretrainCreateRefusals)))
        {
            return;
        }

        GemmaPlugin.Register();                      // Gemma3ForCausalLM below is refused for reading checkpoints only, not for being unknown
        string folder = TempFolder("pretrain-refusals");
        try
        {
            var spec = TinyLlamaSpec();
            Throws<NotSupportedException>(() => PretrainedModel.Create(folder, spec, "NoSuchForCausalLM", TinyFiles()), "an unknown architecture");
            Throws<NotSupportedException>(() => PretrainedModel.Create(folder, spec, "Gemma3ForCausalLM", TinyFiles()), "a family that reads checkpoints only");
            Throws<ArgumentException>(() => PretrainedModel.Create(folder, spec with { Vocabulary = 100 }, "LlamaForCausalLM", TinyFiles()), "a tokenizer larger than the vocabulary");
            Throws<ArgumentException>(() => PretrainedModel.Create(folder, spec, "LlamaForCausalLM", TinyFiles() with { EosToken = "<eos>" }), "a missing token");
            File.WriteAllText(Path.Combine(folder, "keep.txt"), "x");
            Throws<IOException>(() => PretrainedModel.Create(folder, spec, "LlamaForCausalLM", TinyFiles()), "a folder that holds files");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void PretrainSaveCheckpoint(Device device)
    {
        string folder = TempFolder("pretrain-save");
        try
        {
            using (var model = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = device }))
            {
                model.SaveHuggingFace(folder, SafeTensorType.F32);
            }

            var expected = ReadTensors(Path.Combine(TinyLlama, "model.safetensors"));
            var saved = ReadTensors(Path.Combine(folder, "model.safetensors"));
            Check(saved.Count == expected.Count && saved.All(t => expected.TryGetValue(t.Key, out var e) && e.Shape.SequenceEqual(t.Value.Shape) && e.Values.AsSpan().SequenceEqual(t.Value.Values)),
                "every tensor written back as read");
            Check(File.Exists(Path.Combine(folder, "tokenizer.json")) && File.Exists(Path.Combine(folder, "tokenizer_config.json")), "the tokenizer files are copied");

            // Saved over its own folder: the file is read to the end before it is replaced.
            using (var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
            {
                model.SaveHuggingFace(folder, SafeTensorType.F32);
            }

            Check(ReadTensors(Path.Combine(folder, "model.safetensors")).All(t => expected[t.Key].Values.AsSpan().SequenceEqual(t.Value.Values)), "saved over its own folder");

            // A GPT-style spec (layer normalization, learned positions) is written and read back by a family of the test's own.
            var gpt = new DecoderSpec
            {
                Vocabulary = 64, Dim = 16, Layers = 1, Heads = 2, KvHeads = 2, HeadDim = 8, FfDim = 32, MaxPositions = 16,
                Norm = DecoderNorm.Layer, Gated = false, Activation = FeedForwardActivation.Gelu, Rope = null, LearnedPositions = true,
                QkvBias = true, OutputBias = true, FeedForwardBias = true, TieEmbeddings = true,
            };
            const string Name = "TestGptForCausalLM";
            PretrainedArchitectures.Register(Name, new PretrainedArchitecture
            {
                Spec = (config, _) => DecoderSpec.FromJson(config["idrak_spec"]!.AsObject()),
                TensorName = name => name,
                Config = spec => new JsonObject { ["model_type"] = "test-gpt" },
            });
            try
            {
                string gptFolder = Path.Combine(folder, "gpt");
                var files = new ModelFiles { Tokenizer = BpeTokenizer.Load(PretrainData("tokenizer")) };
                Throws<ArgumentException>(() => PretrainedModel.Create(gptFolder, gpt, Name, files), "a 512-token tokenizer for a 64-token vocabulary");
                using var created = PretrainedModel.Create(gptFolder, gpt with { Vocabulary = 512 }, Name, files, new CreateOptions { Device = device, Seed = 3 });
                var tensors = ReadTensors(Path.Combine(gptFolder, "model.safetensors"));
                Check(tensors.ContainsKey("pos.weight") && tensors.ContainsKey("layers.0.attn_norm.bias") && tensors.ContainsKey("norm.bias"), $"learned positions and layer norms written: {string.Join(", ", tensors.Keys.Order())}");
                string resaved = Path.Combine(folder, "gpt-saved");
                created.SaveHuggingFace(resaved, SafeTensorType.F32);
                var back = ReadTensors(Path.Combine(resaved, "model.safetensors"));
                Check(back.All(t => tensors[t.Key].Values.AsSpan().SequenceEqual(t.Value.Values)), "and read back value for value");
            }
            finally
            {
                PretrainedArchitectures.Unregister(Name);
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }
}
