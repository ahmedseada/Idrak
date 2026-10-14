// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Generation;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Nlp;

// Plan 19: language models made from scratch (phase A1: a model from a spec, saved as a folder; A3: the pretraining loop).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] PretrainGroup =
    [
        ("pretrain: every Llama-style family writes a spec as config.json and reads the same spec back; a spec a family cannot say, and a family that reads checkpoints only, are refused", PretrainConfigRoundTrip),
        ("pretrain: a model made from a spec is a folder of the family's checkpoint (the tensors' names and shapes of transformers' own), seeded, loading and chatting like any model; the created template takes images", PretrainCreate),
        ("pretrain: making a model refuses an unknown architecture, a tokenizer larger than the vocabulary, a missing token and a folder that holds files", PretrainCreateRefusals),
        ("pretrain: a loaded model saved as a checkpoint writes transformers' tensors back value for value, LayerNorm and learned positions included, and saves over its own folder", PretrainSaveCheckpoint),
    ];

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
