// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models;
using Idrak.Models.Abstractions;

namespace Idrak.Vision.Plugins.Qwen;

/// <summary>
/// The text decoder of a Qwen-VL checkpoint, registered under the family's architecture name
/// (<see cref="PretrainedArchitectures"/>, by <see cref="QwenVisionPlugin"/>): Qwen2's decoder (q, k and v biases) whose rotary embedding has three axes
/// (<see cref="RopeSettings.Sections"/> from <c>mrope_section</c>: time, height, width pairs), read from
/// <c>text_config</c> and its <c>rope_parameters</c> (transformers 5) or the top-level keys with
/// <c>rope_scaling {"type": "mrope"}</c> and <c>rope_theta</c> (the hub's 4.x checkpoints), the tensors under
/// <c>model.layers.*</c> (the hub's naming) or <c>model.language_model.layers.*</c>. A text-only prompt reads as Qwen2
/// does (its tokens take the same position on every axis).
/// </summary>
public static class QwenTextDecoder
{
    private const string Nested = "model.language_model.";

    /// <summary>
    /// The decoder as <see cref="PretrainedArchitectures"/> takes it (registered under each variant's architecture name by
    /// <see cref="QwenVisionPlugin"/>): the hub's tensor naming, or transformers 5's nested one when the checkpoint has it.
    /// </summary>
    public static PretrainedArchitecture Architecture() => For(nested: false);

    // The architecture in one tensor naming; the checkpoint's names choose the other when it has them.
    private static PretrainedArchitecture For(bool nested) => new()
    {
        Spec = Spec,
        TensorName = name => PretrainedFamilies.CommonTensorName(name) switch
        {
            string stored when nested && stored.StartsWith("model.", StringComparison.Ordinal) => Nested + stored["model.".Length..],
            var stored => stored,
        },
        ForCheckpoint = names => For(nested: names.Contains(Nested + "embed_tokens.weight")),
    };

    /// <summary>The decoder's spec from a Qwen-VL config.json (either layout).</summary>
    /// <exception cref="InvalidDataException">The configuration gives no <c>mrope_section</c>.</exception>
    public static DecoderSpec Spec(JsonObject config, List<string> notes)
    {
        ArgumentNullException.ThrowIfNull(config);
        var text = config["text_config"] is JsonObject nested ? (JsonObject)nested.DeepClone() : (JsonObject)config.DeepClone();
        foreach (string key in (string[])["tie_word_embeddings", "vocab_size"])
        {
            if (!text.ContainsKey(key) && config[key] is { } value)
            {
                text[key] = value.DeepClone();
            }
        }

        var rope = text["rope_parameters"] as JsonObject ?? text["rope_scaling"] as JsonObject ?? config["rope_scaling"] as JsonObject;
        if (rope?["mrope_section"] is not JsonArray sections)
        {
            throw new InvalidDataException("config.json gives no mrope_section (rope_parameters or rope_scaling): Qwen-VL's decoder rotates by three axes.");
        }

        string type = (string?)rope["rope_type"] ?? (string?)rope["type"] ?? "default";
        if (type is not ("default" or "mrope"))
        {
            throw new NotSupportedException($"Qwen-VL's rotary embedding of type '{type}' is not supported (default with mrope_section).");
        }

        text["rope_theta"] = (float?)rope["rope_theta"] ?? (float?)text["rope_theta"] ?? 1_000_000f;
        text.Remove("rope_parameters");
        text.Remove("rope_scaling");
        var spec = PretrainedFamilies.CommonSpec(text, notes) with { QkvBias = true };
        return spec with { Rope = spec.Rope! with { Sections = new RotarySections([.. sections.Select(s => (int)s!)]) } };
    }
}
