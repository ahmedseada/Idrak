// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Text;

// A model's own recommended sampling (generation_config.json, SamplingSettings): read at load, carried by its generators and
// chat models, and laid under explicit values (explicit first, the model's next, the options' own last).
internal abstract partial class Tests
{
    internal static void RecommendedSamplingIsRead(Device device)
    {
        if (device.Type != DeviceType.Cpu || !FirstRun(nameof(RecommendedSamplingIsRead)))
        {
            return;                                                             // files and options: one device shows it
        }

        string folder = TempFolder("recommended-sampling");
        PretrainedModel.Create(folder, TinyLlamaSpec(), "LlamaForCausalLM", TinyFiles(), new CreateOptions { Device = device, Seed = 3 }).Dispose();
        using (var plain = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
        {
            IChatModel chat = plain.CreateChat();
            Check(plain.RecommendedSampling is null && plain.CreateGenerator().RecommendedSampling is null && chat.RecommendedSampling is null,
                "a generation_config.json with token ids only recommends nothing: the model, its generator and its chat model say null");
        }

        // Qwen2.5-7B-Instruct's file, as Hugging Face ships it.
        string path = Path.Combine(folder, SamplingSettings.FileName);
        var config = JsonNode.Parse(File.ReadAllText(path))!.AsObject();
        foreach (var (key, value) in new (string, JsonNode)[] { ("do_sample", true), ("temperature", 0.7), ("top_p", 0.8), ("top_k", 20), ("repetition_penalty", 1.05) })
        {
            config[key] = value;
        }

        File.WriteAllText(path, config.ToJsonString());
        var expected = new SamplingSettings { Temperature = 0.7f, TopP = 0.8f, TopK = 20, RepeatPenalty = 1.05f, Greedy = false };
        using (var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
        {
            var read = model.RecommendedSampling;
            Check(read is not null && read with { Source = null } == expected && read.Source == path, $"the model's recommendation is read: {read}");
            IChatModel chat = model.CreateChat();
            ITextModel text = model.CreateGenerator();
            Check(chat.RecommendedSampling == read && text.RecommendedSampling == read && chat.WithTools(ToolRegistry.Create().Build(), 1).RecommendedSampling == read,
                "the generator, the chat model and a tool-running wrapper carry it");
        }

        // No explicit value: the model's values over the library's defaults; min_p (not recommended) stays the default's.
        var resolved = SamplingSettings.Resolve(null, null, expected);
        Check(resolved.Options is { Temperature: 0.7f, TopK: 20, TopP: 0.8f, MinP: 0f, RepeatPenalty: 1.05f, NumCtx: 2048 }
              && resolved.FromModel.Select(s => s.Name).SequenceEqual(["temperature", "top_k", "top_p", "repeat_penalty"])
              && resolved.Settings.Single(s => s.Name == "min_p").Source == SamplingSettings.Default,
            $"the recommendation under no explicit value: {resolved.Describe()}");

        // Explicit values win, value by value; the baseline's other options stay.
        var baseline = new GenerationOptions { NumCtx = 512, NumPredict = 9, Temperature = 0.6f, TopK = 30, TopP = 0.95f, RepeatPenalty = 1f };
        var mixed = SamplingSettings.Resolve(baseline, new SamplingSettings { Temperature = 0.3f, TopK = 5 }, expected);
        Check(mixed.Options is { Temperature: 0.3f, TopK: 5, TopP: 0.8f, RepeatPenalty: 1.05f, NumCtx: 512, NumPredict: 9 }
              && mixed.Settings.Single(s => s.Name == "temperature").Source == SamplingSettings.Given, $"explicit values win: {mixed.Describe()}");
        Check(SamplingSettings.Resolve(baseline, null, null).Options == baseline, "no recommendation and nothing explicit: the baseline as it is");

        // Greedy: the model's do_sample false is top-k 1, unless the app says how to choose tokens; the app's greedy always wins.
        var greedyModel = new SamplingSettings { Greedy = true, Temperature = 0.7f };
        Check(SamplingSettings.Resolve(null, null, greedyModel).Options is { TopK: 1, Temperature: 1f, TopP: 1f, MinP: 0f }, "do_sample false: greedy");
        Check(SamplingSettings.Resolve(null, new SamplingSettings { Temperature = 0.5f }, greedyModel).Options is { TopK: 40, Temperature: 0.5f },
            "an explicit temperature turns off the model's greedy decoding");
        Check(SamplingSettings.Resolve(null, new SamplingSettings { Temperature = 0f }, expected).Options is { TopK: 1, Temperature: 1f }
              && expected.ApplyTo(new GenerationOptions { NumPredict = 4 }) is { Temperature: 0.7f, TopK: 20, NumPredict: 4 },
            "an explicit temperature 0 is greedy over the model's sampling; ApplyTo lays the recommendation over given options");

        // Values out of range are left out; a file with none recommends nothing.
        var odd = SamplingSettings.FromJson(new JsonObject { ["top_p"] = 1.5, ["temperature"] = -1, ["top_k"] = 2.5, ["min_p"] = 0.05, ["repetition_penalty"] = "1.1" });
        Check(odd == new SamplingSettings { MinP = 0.05f, RepeatPenalty = 1.1f } && SamplingSettings.FromJson(new JsonObject { ["eos_token_id"] = 2 }) is null,
            $"out-of-range values left out, numbers in strings read, no sampling key: null ({odd})");
    }

    // The weights "auto" choice (PretrainedOptions.AutoWeights): the device's free memory faked by AutoWeightsBudget.
    internal static void AutoWeightsChooseByMemory(Device device)
    {
        if (device.Type != DeviceType.Cpu || !FirstRun(nameof(AutoWeightsChooseByMemory)))
        {
            return;                                                             // a choice made on the host: one device shows it
        }

        // A 7B-shaped spec (Qwen2.5-7B's sizes; never built): bfloat16 does not fit 14.5 GB with 4096 tokens' caches, int8 does.
        var big = TinyLlamaSpec() with { Dim = 3584, Layers = 28, Heads = 28, KvHeads = 4, HeadDim = 128, FfDim = 18944, Vocabulary = 152064, TieEmbeddings = false, MaxPositions = 32768 };
        Check(AutoWeights.WeightBytes(big, "float32") == big.ParameterCount * 4 && AutoWeights.WeightBytes(big, "bfloat16") == big.ParameterCount * 2
              && AutoWeights.WeightBytes(big, "int8") < AutoWeights.WeightBytes(big, "bfloat16") && AutoWeights.WeightBytes(big, "int4") < AutoWeights.WeightBytes(big, "int8"),
            $"weights' bytes by format ({big.ParameterCount} parameters)");
        var chosen = AutoWeights.Choose(big, 4096, 14_500_000_000, offload: false);
        Check(chosen is { Format: "int8", Fits: true, Offload: false } && chosen.ToString().StartsWith("int8 (auto): ", StringComparison.Ordinal),
            $"14.5 GB free: int8 ({chosen})");
        Check(AutoWeights.Choose(big, 4096, null, offload: false).Format == "float32", "a device that reports no free memory: the first format");
        string refusal = "";
        try
        {
            AutoWeights.Choose(big, 4096, 3_000_000_000, offload: false, model: "big");
        }
        catch (InsufficientMemoryException e)
        {
            refusal = e.Message;
        }

        Check(refusal.StartsWith("big does not fit the device: int4", StringComparison.Ordinal) && refusal.Contains("offload", StringComparison.Ordinal)
              && refusal.Contains("not even int4's weights fit", StringComparison.Ordinal), $"nothing fits: refused before loading, naming offload ({refusal})");
        Check(AutoWeights.Choose(big, 4096, 3_000_000_000, offload: true) is { Format: "int4", Offload: true, Fits: false }, "nothing fits with offload on: int4, offloaded");
        var tight = AutoWeights.Choose(big, 4096, AutoWeights.WeightBytes(big, "int4") * 12 / 10, offload: true);
        Check(tight.Format == "int4", $"int4 fits a 1.2x budget of its weights ({tight})");

        // The tiny model: each format's own need as the budget; the load takes the first that fits, and its logits are those
        // of the same format named (rule 71: the choice changes nothing but the format).
        string folder = TempFolder("auto-weights");
        var spec = TinyLlamaSpec();
        PretrainedModel.Create(folder, spec, "LlamaForCausalLM", TinyFiles(), new CreateOptions { Device = device, Seed = 5 }).Dispose();
        using var ids = Tensor.From([1f, 5f, 9f, 3f], [1, 4], device);
        float[] Logits(PretrainedModel model)
        {
            using (Autograd.NoGrad())
            {
                return model.Network.Forward(ids).ToArray();
            }
        }

        int context = Math.Min(AutoWeights.DefaultContext, spec.MaxPositions);
        long Need(string format)
        {
            long weights = AutoWeights.WeightBytes(spec, format);
            return weights + AutoWeights.WorkingBytes(spec, context, weights, AutoWeights.DefaultHeadroom);
        }

        foreach (string format in AutoWeights.Formats)
        {
            long budget = Need(format);
            string expected = AutoWeights.Formats.First(f => Need(f) <= budget);
            // A float32 run: its formats start at float32 (a 16-bit run's leave it out, plan 37 T2: "element types:" tests it).
            using var auto = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, AutoWeights = true, AutoWeightsBudget = budget, PackedCache = false, ElementType = "float32" });
            using var named = PretrainedModel.Load(folder, new PretrainedOptions
            {
                Device = device, Int8 = expected == "int8", Int4 = expected == "int4", BFloat16 = expected == "bfloat16", PackedCache = false,
            });
            Check(auto.Weights is { } w && w.Format == expected && w.AvailableBytes == budget && w.Context == context && auto.Notes.Any(n => n.StartsWith($"weights {expected} (auto)", StringComparison.Ordinal)),
                $"a budget of {format}'s need: {expected} ({auto.Weights})");
            Check(Logits(auto).Select(BitConverter.SingleToInt32Bits).SequenceEqual(Logits(named).Select(BitConverter.SingleToInt32Bits)),
                $"{expected} chosen: the logits of {expected} named, bit for bit");
        }

        string tooSmall = "";
        try
        {
            using var _ = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, AutoWeights = true, AutoWeightsBudget = 1000 });
        }
        catch (InsufficientMemoryException e)
        {
            tooSmall = e.Message;
        }

        Check(tooSmall.Contains("does not fit the device", StringComparison.Ordinal), $"a budget nothing fits: the load is refused ({tooSmall})");
        using (var explicitFormat = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, AutoWeights = true, AutoWeightsBudget = 1000, Int8 = true, PackedCache = false }))
        {
            Check(explicitFormat.Weights is null && explicitFormat.Network.Descendants().OfType<Idrak.Layers.Linear>().All(l => l.PackedWeight?.Name == "int8"),
                "a named format wins over the choice (no budget check)");
        }

        using (var off = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, PackedCache = false }))
        {
            Check(off.Weights is null && off.Network.Descendants().OfType<Idrak.Layers.Linear>().All(l => l.PackedWeight is null), "the choice off: float32, as before");
        }

        // A bfloat16 checkpoint: float32 is not tried (the same values in twice the memory).
        string halves = TempFolder("auto-weights-bf16");
        PretrainedModel.Create(halves, spec, "LlamaForCausalLM", TinyFiles(), new CreateOptions { Device = device, Seed = 5, Type = Idrak.Models.SafeTensorType.BF16 }).Dispose();
        using (var bf16 = PretrainedModel.Load(halves, new PretrainedOptions { Device = device, AutoWeights = true, AutoWeightsBudget = long.MaxValue / 4 }))
        {
            Check(bf16.Weights?.Format == "bfloat16", $"a bfloat16 checkpoint with room to spare: bfloat16 ({bf16.Weights})");
        }
    }
}
