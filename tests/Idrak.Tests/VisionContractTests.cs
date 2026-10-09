// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Gemma3Vision;
using Idrak.Models;
using Idrak.Models.Abstractions;

// Plan 11, "Vision contracts (families are applications)": the library registers no vision family. Gemma 3's is a
// registration from outside it (samples/Idrak.Gemma3Vision), registered by the tests that use it; without it a Gemma 3
// vision checkpoint is refused, naming the registry.
internal static partial class Tests
{
    /// <summary>Registers the Gemma 3 vision family (the sample plug-in's entry point), as an app or <c>-P</c> does.</summary>
    private static void RegisterGemma3Vision() => Gemma3VisionPlugin.Register();

    /// <summary>
    /// Removes the Gemma 3 vision family and its attention rule, and checks that the tiny Gemma 3 vision checkpoint then
    /// fails to load with the "not registered" error naming the registry (no fallback to any family).
    /// </summary>
    private static void Gemma3VisionNotRegistered(Device device)
    {
        VisionFamilies.Unregister(Gemma3VisionFamily.Architecture);
        ImageAttentionRules.Unregister(Gemma3VisionPlugin.ImageBlocks);
        Check(VisionFamilies.Find(Gemma3VisionFamily.Architecture) is null && VisionFamilies.Default(Gemma3VisionFamily.Architecture) is null
              && ImageAttentionRules.Find(Gemma3VisionPlugin.ImageBlocks) is null && ImageAttentionRules.Names.SequenceEqual([ImageAttentionRules.Causal]),
            "the library registers no vision family and no image attention rule but the causal one");
        try
        {
            using var _ = PretrainedModel.Load(TestData("vlm/tiny-gemma3-v5"), new PretrainedOptions { Device = device });
            Check(false, "a vision checkpoint whose family is not registered is refused");
        }
        catch (NotSupportedException e)
        {
            Check(e.Message.Contains("Vision family 'Gemma3ForConditionalGeneration' is not registered", StringComparison.Ordinal)
                  && e.Message.Contains("VisionFamilies.Register", StringComparison.Ordinal), $"the error names the registry: {e.Message}");
        }

        Check(VisionFamilies.For("Gemma3ForCausalLM", new System.Text.Json.Nodes.JsonObject()) is null, "a text checkpoint needs no vision family");
    }
}
