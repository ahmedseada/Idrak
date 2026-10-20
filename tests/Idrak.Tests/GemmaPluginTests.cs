// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Abstraction.Plugins;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Text.Plugins.Gemma;
using Idrak.Vision.Plugins.Gemma;

// Plan 26, P6: the Gemma family is Idrak.Text.Plugins' (Gemma, Gemma 2, Gemma 3 and the text decoder of Gemma 3 with
// images) and its vision part Idrak.Vision.Plugins'; the library keeps none of it. The family's outputs are checked by
// the groups that always checked them (pretrained:, decoder:, vision contracts, ...), which register the plug-ins.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] GemmaGroup =
    [
        ("plugins: gemma: the library registers no Gemma architecture or GGUF mapping; without Idrak.Text.Plugins' family a Gemma checkpoint and a gemma2 GGUF file are refused as not registered, naming the package; Register adds the four architectures and the three GGUF mappings (gemma, gemma2, gemma3) from the plug-in, once, Unregister takes them back, and the Gemma 3 vision plug-in registers its text decoder only when missing and takes it back only then", GemmaPluginRegistration),
    ];

    private static void GemmaPluginRegistration(Device device)
    {
        bool text = GemmaPlugin.IsRegistered, vision = Gemma3VisionPlugin.IsRegistered;
        try
        {
            Gemma3VisionPlugin.Unregister();
            GemmaPlugin.Unregister();
            foreach (string name in GemmaPlugin.Architectures)
            {
                Check(!PretrainedArchitectures.Names.Contains(name) && PretrainedArchitectures.Default(name) is null && PretrainedArchitectures.Origin(name) is null,
                    $"{name}: the library keeps no Gemma architecture");
                string message = Failure<NotSupportedException>(() => PretrainedArchitectures.Get(name)).Message;
                Check(message.Contains($"No architecture '{name}' is registered", StringComparison.Ordinal) && message.Contains("Idrak.Text.Plugins", StringComparison.Ordinal)
                      && message.Contains("RegisterAll()", StringComparison.Ordinal), $"{name}: the message names the package: {message}");
            }

            Check(GemmaPlugin.GgufNames.All(n => GgufArchitectures.Find(n) is null && GgufArchitectures.Default(n) is null), "the library keeps no Gemma GGUF mapping");

            // A library user without the package: the checkpoint is refused before a weight is read, naming the package.
            string refused = Failure<NotSupportedException>(() => PretrainedModel.Load(TestData("vlm/tiny-gemma3-v5"), new PretrainedOptions { Device = device }).Dispose()).Message;
            Check(refused.Contains("No architecture 'Gemma3ForConditionalGeneration' is registered", StringComparison.Ordinal)
                  && refused.Contains("Idrak.Text.Plugins", StringComparison.Ordinal), $"a Gemma 3 checkpoint without the package: {refused}");
            string cache = Path.Combine(Path.GetTempPath(), "ns-gguf-" + Guid.NewGuid().ToString("N"));
            string prepared = Failure<NotSupportedException>(() => GgufModel.Prepare(TestData("gguf/tiny-gemma2.gguf"), cache)).Message;
            Check(prepared.Contains("GGUF architecture 'gemma2' is not registered", StringComparison.Ordinal) && prepared.Contains("Idrak.Text.Plugins", StringComparison.Ordinal),
                $"a gemma2 GGUF file without the package: {prepared}");

            GemmaPlugin.Register();
            GemmaPlugin.Register();
            string assembly = typeof(GemmaPlugin).Assembly.GetName().Name!;
            Check(GemmaPlugin.IsRegistered && ReferenceEquals(IdrakPlugins.Find("gemma"), GemmaPlugin.Plugin)
                  && GemmaPlugin.Architectures.All(n => PretrainedArchitectures.Origin(n) == assembly && PretrainedArchitectures.Default(n) is null)
                  && PretrainedArchitectures.Names.Count(GemmaPlugin.Architectures.Contains) == 4
                  && GemmaPlugin.GgufNames.All(n => GgufArchitectures.Origin(n) == assembly && GgufArchitectures.Default(n) is null) && GgufArchitectures.Names.Count(GemmaPlugin.GgufNames.Contains) == 3
                  && GemmaPlugin.GgufNames.Select(n => GgufArchitectures.Get(n).HuggingFace).SequenceEqual([GemmaPlugin.GemmaForCausalLM, GemmaPlugin.Gemma2ForCausalLM, GemmaPlugin.Gemma3ForCausalLM]),
                "Register (twice) adds the four architectures and the three GGUF mappings once, from the plug-in's assembly, over no library default");
            Check(GemmaGguf.TensorName("blk.1.post_ffw_norm.weight") == "model.layers.1.post_feedforward_layernorm.weight"
                  && GemmaGguf.TensorName("blk.0.ffn_norm.weight") == "model.layers.0.pre_feedforward_layernorm.weight"
                  && GemmaGguf.TensorName("blk.2.post_attention_norm.weight") == "model.layers.2.post_attention_layernorm.weight"
                  && GemmaGguf.TensorName("blk.0.attn_q.weight") == GgufArchitecture.CommonTensorName("blk.0.attn_q.weight")
                  && GemmaGguf.StoredOffset("model.layers.0.self_attn.q_norm.weight") == 1f && GemmaGguf.StoredOffset("model.norm.weight") == 1f
                  && GemmaGguf.StoredOffset("model.layers.0.mlp.up_proj.weight") == 0f && GgufArchitectures.Get("gemma").TensorName is null,
                "llama.cpp's Gemma 2 and 3 norm names, the rest the common ones (Gemma's too); the 1 added to every norm gain, to nothing else");
            Check(PretrainedArchitectures.Get(GemmaPlugin.GemmaForCausalLM).Config is not null && PretrainedArchitectures.Get(GemmaPlugin.Gemma3ForCausalLM).Config is null
                  && PretrainedArchitectures.Get(GemmaPlugin.Gemma3ForConditionalGeneration).ForCheckpoint is not null,
                "Gemma writes its configuration; Gemma 3 reads checkpoints only; Gemma 3 with images fits each checkpoint's naming");
            Check(GemmaArchitectures.TensorName("layers.3.post_mlp_norm.weight") == "model.layers.3.post_feedforward_layernorm.weight"
                  && GemmaArchitectures.TensorName("layers.0.mlp_norm.weight") == "model.layers.0.pre_feedforward_layernorm.weight"
                  && GemmaArchitectures.TensorName("layers.1.attn.q.weight") == PretrainedFamilies.CommonTensorName("layers.1.attn.q.weight"),
                "the four norms under Gemma's names, the rest under the Llama names");
            GemmaPlugin.Unregister();
            Check(!GemmaPlugin.IsRegistered && IdrakPlugins.Find("gemma") is null && !PretrainedArchitectures.Names.Any(GemmaPlugin.Architectures.Contains)
                  && !GgufArchitectures.Names.Any(GemmaPlugin.GgufNames.Contains), "Unregister takes the four and the GGUF mappings back");

            // The vision family needs its text decoder: registered with it when missing, and taken back with it only then.
            Gemma3VisionPlugin.Register();
            Check(GemmaPlugin.IsRegistered && VisionFamilies.Find(Gemma3VisionFamily.Architecture) is Gemma3VisionFamily, "the vision plug-in registers the text decoder it needs");
            Gemma3VisionPlugin.Unregister();
            Check(!GemmaPlugin.IsRegistered && VisionFamilies.Find(Gemma3VisionFamily.Architecture) is null
                  && !PretrainedArchitectures.Names.Contains(GemmaPlugin.Gemma3ForConditionalGeneration), "and takes it back, having added it");
            GemmaPlugin.Register();
            Gemma3VisionPlugin.Register();
            Gemma3VisionPlugin.Unregister();
            Check(GemmaPlugin.IsRegistered && VisionFamilies.Find(Gemma3VisionFamily.Architecture) is null, "a text family registered before the vision one stays");
        }
        finally
        {
            Gemma3VisionPlugin.Unregister();
            GemmaPlugin.Unregister();
            if (text)
            {
                GemmaPlugin.Register();
            }

            if (vision)
            {
                Gemma3VisionPlugin.Register();
            }
        }
    }
}
