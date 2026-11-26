// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Generation;
using Idrak.Models;
using Idrak.Text;
using Idrak.Vision.Plugins;

// The vision-language families loaded by their own names: Idrak.Vision.Plugins' list (the text families with the
// vision-language ones) walked where a checkpoint comes in, its text decoder and its vision part each the family stating
// the checkpoint's name.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] VisionFamilyLoadGroup =
    [
        ("vision families: the tiny Qwen2-VL, Qwen2.5-VL, Gemma 3 (three namings), LLaVA and IdrakVision checkpoints load by their own names through the families handed over: the text decoder and the vision part each the family stating the checkpoint's name, the vision encoder built from the checkpoint as wide as the decoder, the decoder generating with the folder's own chat template; given the text families alone, a vision checkpoint is refused naming its name", VisionFamiliesLoad),
    ];

    internal static void VisionFamiliesLoad(Device device)
    {
        CpuOnly(device);
        var families = BuiltInVisionFamilies.All;
        var options = new Idrak.Abstraction.Generation.GenerationOptions { TopK = 1, NumPredict = 4, Seed = 1 };
        foreach (string fixture in new[] { "vlm-qwen/tiny-qwen2-vl", "vlm-qwen/tiny-qwen2.5-vl", "vlm/tiny-gemma3", "vlm/tiny-gemma3-pre452", "vlm/tiny-gemma3-v5", "vlm-llava/tiny-llava", "vlm-scratch/tiny-idrak-vision" })
        {
            string folder = TestData(fixture);
            string name = (string)JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "config.json")))!["architectures"]![0]!;
            using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, Families = families, ChatTemplateReader = JinjaChatTemplate.Reader });
            Check(model.Architecture.Names.Contains(name) && model.Vision is { } vision && families.VisionFamilies.Any(f => f.Names.Contains(name) && f.Name == vision.Family),
                $"{fixture}: the {name} text decoder and vision part found by the checkpoint's own name ({model.Vision?.Family ?? "no vision part"})");
            using (var encoder = model.CreateVisionEncoder())
            {
                Check(encoder.Width == model.Spec.Dim, $"{fixture}: the vision encoder's features are the decoder's width ({encoder.Width}, {model.Spec.Dim})");
            }

            Check(model.ChatTemplate is not null, $"{fixture}: the folder's own chat template read");
            _ = new TextGenerator(model.Network, model.Tokenizer!, 64).Generate("hello", options);
            Console.WriteLine($"  {fixture}: a {name} by its own names, its vision part {model.Vision!.Family}, generating");
        }

        string refused = "";
        try
        {
            using var _ = PretrainedModel.Load(TestData("vlm/tiny-gemma3"), new PretrainedOptions { Device = device, Families = Idrak.Text.Plugins.BuiltInFamilies.All });
        }
        catch (NotSupportedException e)
        {
            refused = e.Message;
        }

        Check(refused.Contains("'Gemma3ForConditionalGeneration'", StringComparison.Ordinal), $"the text families alone refuse a vision checkpoint naming its name: {refused}");
    }
}
