// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
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
}
