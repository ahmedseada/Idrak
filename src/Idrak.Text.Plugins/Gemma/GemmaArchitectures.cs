// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models;
using Idrak.Models.Abstractions;

namespace Idrak.Text.Plugins.Gemma;

/// <summary>
/// The Gemma families as <see cref="IPretrainedArchitecture"/>s (what <see cref="GemmaPlugin"/> registers): Gemma
/// (<see cref="Gemma"/>, the Llama naming with (1 + w) gains, scaled and tied embeddings, written from a spec too), Gemma 2
/// and Gemma 3 text (<see cref="Gemma2"/>, <see cref="Gemma3"/>, on the common keys read by
/// <see cref="PretrainedFamilies.CommonSpec"/>), and the text decoder of Gemma 3 with images (<see cref="Gemma3VisionLanguage()"/>,
/// read from text_config in whichever of its namings the checkpoint uses; its vision part is a <see cref="VisionFamilies"/>
/// registration of Idrak.Vision.Plugins). Each call returns a new architecture; register them with
/// <see cref="GemmaPlugin.Register"/>, or under names of your own with <see cref="PretrainedArchitectures.Register"/>.
/// </summary>
public static class GemmaArchitectures
{
    /// <summary>Gemma (<c>GemmaForCausalLM</c>): the Llama naming, gains stored as a difference from 1, embeddings scaled by √dim and tied; writes its configuration from a spec ("gemma").</summary>
    public static IPretrainedArchitecture Gemma() => PretrainedFamilies.CommonArchitecture((config, spec, _) => spec with
    {
        NormOffset = 1f,                                                  // Gemma scales by (1 + weight)
        EmbeddingScale = MathF.Sqrt(spec.Dim),
        TieEmbeddings = true,
    }, "gemma");

    /// <summary>Gemma 2 (<c>Gemma2ForCausalLM</c>): read from checkpoints only.</summary>
    public static IPretrainedArchitecture Gemma2() => PretrainedFamilies.Architecture((config, notes) => GemmaSpec(config, PretrainedFamilies.CommonSpec(config, notes), version: 2), TensorName);

    /// <summary>Gemma 3 text (<c>Gemma3ForCausalLM</c>): read from checkpoints only, transformers 5's rope_parameters too.</summary>
    public static IPretrainedArchitecture Gemma3() => PretrainedFamilies.Architecture((config, notes) => Gemma3Spec(Gemma3Keys(config), notes), TensorName);

    /// <summary>
    /// The text decoder of Gemma 3 with images (<c>Gemma3ForConditionalGeneration</c>): Gemma 3's, read from text_config,
    /// never soft-capping its logits, in whichever of its two namings the checkpoint uses
    /// (<see cref="ICheckpointFittedArchitecture.ForCheckpoint"/>).
    /// </summary>
    public static IPretrainedArchitecture Gemma3VisionLanguage() => Gemma3VisionLanguage(Gemma3Layouts[0]);

    // Gemma 2 and Gemma 3 (text) on top of the common keys: gains stored as a difference from 1, embeddings scaled by √dim
    // and tied, normalizations after attention and the feed-forward block too, scores scaled by query_pre_attn_scalar^-1/2,
    // windowed layers alternating (every other layer for Gemma 2, five in six for Gemma 3, unless layer_types says), and
    // soft-capping (Gemma 2 by default). Gemma 3 also normalizes queries and keys and rotates the windowed layers with a
    // local base (rope_local_base_freq), unscaled. Defaults are transformers'.
    private static DecoderSpec GemmaSpec(JsonObject c, DecoderSpec spec, int version)
    {
        float? Optional(string key, float? fallback) => c.ContainsKey(key) ? (float?)c[key] : fallback;
        if (version == 3 && (bool?)c["use_bidirectional_attention"] == true)
        {
            throw new NotSupportedException("Gemma 3 with bidirectional attention (an embedding model) is not supported.");
        }

        int window = (int?)c["sliding_window"] ?? 4096;
        int pattern = version == 2 ? 2 : (int?)c["sliding_window_pattern"] ?? 6;
        bool[] windowed = c["layer_types"] is JsonArray types
            ? [.. types.Select(type => (string?)type == "sliding_attention")]
            : [.. Enumerable.Range(0, spec.Layers).Select(i => (i + 1) % pattern != 0)];
        var rope = spec.Rope!;
        if (version == 3 && c["rope_theta"] is null)
        {
            rope = rope with { Theta = 1_000_000f };
        }

        return spec with
        {
            NormOffset = 1f,
            EmbeddingScale = MathF.Sqrt(spec.Dim),
            TieEmbeddings = (bool?)c["tie_word_embeddings"] ?? true,
            PostNorms = true,
            QkNorm = version == 3,
            AttentionScale = 1f / MathF.Sqrt(Optional("query_pre_attn_scalar", null) ?? 256f),
            AttentionSoftcap = Optional("attn_logit_softcapping", version == 2 ? 50f : null),
            LogitSoftcap = Optional("final_logit_softcapping", version == 2 ? 30f : null),
            SlidingWindow = windowed.Contains(true) ? window : null,
            SlidingWindowLayers = windowed.Contains(true) ? windowed : null,
            Rope = rope,
            SlidingWindowRope = version == 3 ? new RopeSettings((float?)c["rope_local_base_freq"] ?? 10000f, rope.RotaryDim) : null,
        };
    }

    private static DecoderSpec Gemma3Spec(JsonObject c, List<string> notes) => GemmaSpec(c, PretrainedFamilies.CommonSpec(c, notes), version: 3);

    // transformers 5 writes the RoPE settings of Gemma 3 as rope_parameters, one entry per layer kind ({"full_attention":
    // {"rope_type": "linear", "factor": 8, "rope_theta": 1e6}, "sliding_attention": {"rope_type": "default", "rope_theta":
    // 1e4}}) or one for all, instead of rope_theta, rope_scaling and rope_local_base_freq. Read into those keys (a copy).
    private static JsonObject Gemma3Keys(JsonObject c)
    {
        if (c["rope_parameters"] is not JsonObject parameters)
        {
            return c;
        }

        c = (JsonObject)c.DeepClone();
        bool perKind = parameters.ContainsKey("full_attention") || parameters.ContainsKey("sliding_attention");
        if ((perKind ? parameters["full_attention"] : parameters) is JsonObject full)
        {
            if (full["rope_theta"] is { } theta)
            {
                c["rope_theta"] = theta.DeepClone();
            }

            var scaling = (JsonObject)full.DeepClone();
            scaling.Remove("rope_theta");
            c["rope_scaling"] = ((string?)full["rope_type"] ?? "default") == "default" ? null : scaling;
        }

        if (perKind && parameters["sliding_attention"] is JsonObject sliding)
        {
            if (((string?)sliding["rope_type"] ?? "default") != "default")
            {
                throw new NotSupportedException($"RoPE scaling '{sliding["rope_type"]}' on the sliding-window layers is not supported (Gemma 3 scales the global layers only).");
            }

            if (sliding["rope_theta"] is { } local)
            {
                c["rope_local_base_freq"] = local.DeepClone();
            }
        }

        return c;
    }

    // The namings of a Gemma3ForConditionalGeneration checkpoint's text decoder (plan 11): where its model and its lm_head
    // (never saved: tied) are. transformers 4.52 to 4.57 rename the modules in memory but save under the old names. The
    // vision part's namings are its vision family's (VisionFamilies), which Idrak.Vision.Plugins registers.
    private sealed record Gemma3Layout(string Name, string Text, string Head);

    private static readonly Gemma3Layout[] Gemma3Layouts =
    [
        new("save_pretrained (language_model.model.*)", "language_model.model.", "language_model.lm_head."),
        new("state dict of transformers 4.52 to 4.57 (model.language_model.*)", "model.language_model.", "lm_head."),
    ];

    // The text decoder of Gemma 3 with images: Gemma 3's, read from text_config, never soft-capping its logits
    // (Gemma3ForConditionalGeneration does not, whatever final_logit_softcapping says), in whichever naming the checkpoint
    // uses. The vision part is read by a registered vision family of the same name (Idrak.Vision.Plugins').
    private static IPretrainedArchitecture Gemma3VisionLanguage(Gemma3Layout layout) => new Gemma3VisionLanguageDecoder(layout);

    private sealed class Gemma3VisionLanguageDecoder(Gemma3Layout layout) : PretrainedArchitecture, ICheckpointFittedArchitecture
    {
        public override DecoderSpec Spec(JsonObject config, List<string> notes)
        {
            var text = Gemma3TextConfig(config);
            var spec = Gemma3Spec(text, notes);
            // Text embeddings are scaled by √width held in the weights' type (Gemma3TextScaledWordEmbedding): √2560 rounds in bfloat16.
            return spec with { LogitSoftcap = null, EmbeddingScale = InType(MathF.Sqrt(spec.Dim), (string?)text["dtype"] ?? (string?)text["torch_dtype"]) };
        }

        public override string? TensorName(string name) => GemmaArchitectures.TensorName(name) switch
        {
            null => null,
            var stored when stored.StartsWith("model.", StringComparison.Ordinal) => layout.Text + stored["model.".Length..],
            var stored => layout.Head + stored["lm_head.".Length..],
        };

        public IPretrainedArchitecture ForCheckpoint(IReadOnlySet<string> names)
        {
            var texts = Gemma3Layouts.Where(l => names.Contains(l.Text + "embed_tokens.weight")).ToList();
            if (texts.Count == 0)
            {
                throw new InvalidDataException("This Gemma3ForConditionalGeneration checkpoint has no text decoder under a known name ("
                    + string.Join(", ", Gemma3Layouts.Select(l => l.Text + "embed_tokens.weight").Distinct()) + ").");
            }

            return Gemma3VisionLanguage(texts[0]);
        }
    }

    // text_config read as a Gemma3ForCausalLM configuration: the keys transformers keeps at the top level (tied embeddings,
    // the weights' type) when text_config lacks them, then Gemma3TextConfig's defaults for absent keys (the original
    // release's text_config names only its width, layers, MLP, window and RoPE scaling).
    private static JsonObject Gemma3TextConfig(JsonObject config)
    {
        var text = config["text_config"] is JsonObject given ? (JsonObject)given.DeepClone() : new JsonObject();
        foreach (string key in (string[])["tie_word_embeddings", "dtype", "torch_dtype"])
        {
            if (!text.ContainsKey(key) && config[key] is { } value)
            {
                text[key] = value.DeepClone();
            }
        }

        foreach (var (key, value) in JsonNode.Parse(Gemma3TextDefaults)!.AsObject())      // parsed, so numbers read as any type, as config.json's do
        {
            if (!text.ContainsKey(key))
            {
                text[key] = value!.DeepClone();
            }
        }

        return Gemma3Keys(text);
    }

    // transformers' Gemma3TextConfig defaults (4.57 and 5.x) for the keys the common reader needs.
    private const string Gemma3TextDefaults = """
        {"vocab_size": 262208, "hidden_size": 2304, "intermediate_size": 9216, "num_hidden_layers": 26, "num_attention_heads": 8,
         "num_key_value_heads": 4, "head_dim": 256, "hidden_activation": "gelu_pytorch_tanh", "max_position_embeddings": 131072,
         "rms_norm_eps": 1e-6, "query_pre_attn_scalar": 256, "sliding_window": 4096, "rope_local_base_freq": 10000.0}
        """;

    // A value as the torch type named holds it ("bfloat16" rounds to nearest even, "float16"; float32 otherwise).
    private static float InType(float value, string? type)
    {
        if (type == "bfloat16")
        {
            uint bits = BitConverter.SingleToUInt32Bits(value);
            return BitConverter.UInt32BitsToSingle((bits + 0x7FFFu + ((bits >> 16) & 1u)) & 0xFFFF0000u);
        }

        return type == "float16" ? (float)(Half)value : value;
    }

    /// <summary>
    /// Idrak weight names → Gemma 2 / Gemma 3 checkpoint names: the common names (<see cref="PretrainedFamilies.CommonTensorName"/>),
    /// with four normalizations per layer (input_layernorm, post_attention_layernorm after attention,
    /// pre_feedforward_layernorm, post_feedforward_layernorm).
    /// </summary>
    public static string? TensorName(string name)
    {
        var parts = name.Split('.');
        if (parts.Length == 4 && parts[0] == "layers" && parts[3] == "weight")
        {
            string? norm = parts[2] switch
            {
                "attn_norm" => "input_layernorm",
                "post_attn_norm" => "post_attention_layernorm",
                "mlp_norm" => "pre_feedforward_layernorm",
                "post_mlp_norm" => "post_feedforward_layernorm",
                _ => null,
            };
            if (norm is not null)
            {
                return $"model.layers.{parts[1]}.{norm}.weight";
            }
        }

        return PretrainedFamilies.CommonTensorName(name);
    }
}
