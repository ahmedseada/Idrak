// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Generation;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Text;
using Idrak.Text.Abstractions;
using Idrak.Text.Plugins.Gemma;
using Idrak.Text.Plugins.Llama;
using Idrak.Text.Plugins.Mistral;
using Idrak.Text.Plugins.Qwen;

// Plan 19: language models made from scratch (phase A1: a model from a spec, saved as a folder; A3: the pretraining loop).
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] PretrainGroup =
    [
        ("pretrain: every Llama-style family writes a spec as config.json and reads the same spec back; a spec a family cannot say, and a family that reads checkpoints only, are refused", PretrainConfigRoundTrip),
    ];

    internal sealed class StepCounter : ITelemetryHook
    {
        public int Batches;
        public int Started;

        public TelemetryLevel Levels => TelemetryLevel.Training | TelemetryLevel.Batches;

        public void OnTrainingStarted(in TrainingStarted e) => Started++;

        public void OnBatchCompleted(in BatchCompleted e) => Batches++;
    }

    internal static string TinyLlama => PretrainData("tiny-llama");

    internal static DecoderSpec TinyLlamaSpec()
    {
        var config = JsonNode.Parse(File.ReadAllText(Path.Combine(TinyLlama, "config.json")))!.AsObject();
        return LlamaArchitectures.Llama().Spec(config, []);
    }

    // A safetensors file's tensors: name → (shape, values).
    internal static Dictionary<string, (int[] Shape, float[] Values)> ReadTensors(string path)
    {
        using var opened = SafeTensorsReader.Open(path);
        ITensorStore reader = opened;
        return reader.Names.ToDictionary(n => n, n => (reader.ShapeOf(n), reader.Read(n)), StringComparer.Ordinal);
    }

    internal static void PretrainConfigRoundTrip(Device device)
    {
        if (!FirstRun(nameof(PretrainConfigRoundTrip)))
        {
            return;
        }

        var llama = TinyLlamaSpec();
        var specs = new (string Architecture, IPretrainedArchitecture Family, DecoderSpec Spec)[]
        {
            ("LlamaForCausalLM", LlamaArchitectures.Llama(), llama),
            ("LlamaForCausalLM", LlamaArchitectures.Llama(), llama with { TieEmbeddings = false, QkvBias = true, OutputBias = true, Rope = new RopeSettings(500000f, 4) }),
            ("MistralForCausalLM", MistralArchitectures.Mistral(), llama with { SlidingWindow = 16 }),
            ("Qwen2ForCausalLM", QwenArchitectures.Qwen2(), llama with { QkvBias = true, SlidingWindow = 16, SlidingWindowLayers = [false, true] }),
            ("Qwen3ForCausalLM", QwenArchitectures.Qwen3(), llama with { QkNorm = true }),
            ("GemmaForCausalLM", GemmaArchitectures.Gemma(), llama with { NormOffset = 1f, EmbeddingScale = MathF.Sqrt(llama.Dim), TieEmbeddings = true, Activation = FeedForwardActivation.Gelu }),
        };
        foreach (var (name, family, spec) in specs)
        {
            Check(family is IConfigWritingArchitecture, $"{name} writes configurations");
            var config = ((IConfigWritingArchitecture)family).Config(spec);
            var back = family.Spec(config, []);
            Check(JsonNode.DeepEquals(back.ToJson(), spec.ToJson()), $"{name}: the spec reads back the same\n{spec.ToJson()}\n{back.ToJson()}");
        }

        foreach (var (family, spec, what) in new (IPretrainedArchitecture, DecoderSpec, string)[]
        {
            (LlamaArchitectures.Llama(), llama with { Norm = DecoderNorm.Layer }, "layer normalization"),
            (LlamaArchitectures.Llama(), llama with { QkNorm = true }, "q/k norms in Llama"),
            (LlamaArchitectures.Llama(), llama with { QkvBias = true, OutputBias = false }, "biases Llama's one key cannot say"),
            (QwenArchitectures.Qwen2(), llama, "Qwen2 without its q/k/v biases"),
        })
        {
            Throws<NotSupportedException>(() => ((IConfigWritingArchitecture)family).Config(spec), what);
        }

        Check(GemmaArchitectures.Gemma3() is not IConfigWritingArchitecture && MistralArchitectures.Mixtral() is not IConfigWritingArchitecture,
            "families read from checkpoints only write no configuration");
        var template = JsonNode.Parse(File.ReadAllText(Path.Combine(TinyLlama, "tokenizer_config.json")))!["chat_template"]!.GetValue<string>();
        Check(template == ModelFiles.ChatMLWithParts, "the created models' template is the reference's");
    }
}
