// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models;

// Plan 19, phases B1 and B2: the vision-language model made from scratch (IdrakVision): its fixtures, as the pretraining
// and distillation tests read them.
internal abstract partial class Tests
{
    internal static string ScratchData(string relative) => TestData($"vlm-scratch/{relative}");

    internal static PretrainedModel LoadScratch(Device device) => PretrainedModel.Load(ScratchData("tiny-idrak-vision"), new PretrainedOptions { Device = device, Families = Idrak.Vision.Plugins.BuiltInVisionFamilies.All });

    internal static JsonObject ScratchDescription()
    {
        var config = JsonNode.Parse(File.ReadAllText(ScratchData("tiny-idrak-vision/config.json")))!.AsObject();
        return new JsonObject
        {
            ["vision_config"] = config["vision_config"]!.DeepClone(),
            ["projector_config"] = config["projector_config"]!.DeepClone(),
            ["image_attention"] = "image-blocks",
        };
    }
}
