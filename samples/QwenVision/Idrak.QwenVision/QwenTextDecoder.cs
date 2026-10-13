// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models;
using Idrak.Models.Abstractions;

namespace Idrak.QwenVision;

/// <summary>
/// The text decoder of a Qwen-VL checkpoint, registered under the family's architecture name
/// (<see cref="PretrainedArchitectures"/>): a Qwen2-style decoder (q/k/v biases) whose rotary positions have three
/// sections (<c>rope_scaling.mrope_section</c>: time, height, width), read from <c>text_config</c> (Qwen2.5-VL in
/// transformers 5, and Qwen2-VL there) or the top-level keys (transformers 4.x), with the tensor names of both layouts
/// (<c>model.layers.*</c>; <c>model.language_model.layers.*</c>). One registration per family: the library holds no
/// Qwen-VL decoder, as it holds no Gemma 3 one after plan 16, step 4. The sections need the library's multi-axis rotary
/// positions (plan 16, step 2); until then the spec fails loudly rather than reading the positions as one axis.
/// </summary>
public static class QwenTextDecoder
{
    /// <summary>Registers the variant's text decoder under its architecture name.</summary>
    public static void Register(QwenVlVariant variant) => PretrainedArchitectures.Register(QwenVisionFamily.ArchitectureOf(variant), For(variant));

    /// <summary>The architecture of one variant's text decoder.</summary>
    public static PretrainedArchitecture For(QwenVlVariant variant) => new()
    {
        Spec = (config, notes) => Spec(config, notes, variant),
        TensorName = TensorName,
    };

    // The decoder's spec: Qwen2's common keys with biases, plus the rotary sections (step 2 gives DecoderSpec a place for them).
    private static DecoderSpec Spec(JsonObject config, List<string> notes, QwenVlVariant variant)
    {
        ArgumentNullException.ThrowIfNull(config);
        throw QwenVisionPlugin.NotBuilt($"the text decoder of {QwenVisionFamily.ArchitectureOf(variant)} (multi-axis rotary positions come first)", 2);
    }

    // The checkpoint's name of a decoder tensor in Idrak's naming, or null for the vision part. Both layouts: the
    // language model at the root (transformers 4.x) or under model.language_model (4.52 and later, 5).
    private static string? TensorName(string name)
    {
        const string nested = "model.language_model.";
        if (name.StartsWith("visual.", StringComparison.Ordinal) || name.StartsWith("model.visual.", StringComparison.Ordinal))
        {
            return null;
        }

        return PretrainedFamilies.LlamaTensorName(name.StartsWith(nested, StringComparison.Ordinal) ? "model." + name[nested.Length..] : name);
    }
}
