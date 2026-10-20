// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models;
using Idrak.Models.Abstractions;

namespace Idrak.Text.Plugins.Gemma;

/// <summary>
/// How llama.cpp's Gemma files are read (what <see cref="GemmaPlugin"/> registers in <see cref="GgufArchitectures"/>):
/// "gemma", "gemma2" and "gemma3" as <c>GemmaForCausalLM</c>, <c>Gemma2ForCausalLM</c> and <c>Gemma3ForCausalLM</c>,
/// read by <see cref="GemmaArchitectures"/>. llama.cpp's converter (convert_hf_to_gguf.py) writes every norm gain with
/// Gemma's 1 added (its graph multiplies by the stored gain), so the 1 is taken back off as the gains are read
/// (<see cref="StoredOffset"/>); Gemma 2 and Gemma 3 keep their four norms per layer under llama.cpp's names
/// (<see cref="TensorName"/>). What the file's keys say becomes the configuration's (<see cref="GgufArchitecture.Configure"/>):
/// GELU (tanh) in the feed-forward block; the sliding window, its layer pattern, the soft-capping and the windowed layers'
/// RoPE base, each when the file has it, else what llama.cpp assumes without it (a window of 4096 every other layer and
/// soft-capping at 50 and 30 for Gemma 2, five windowed layers in six with a base of 10,000 for Gemma 3, none without a
/// window); and the query scalar, which the files do not store (<see cref="ImpliedQueryScalar"/> unless one is given).
/// The tokenizer is the file's SentencePiece one (tokenizer.ggml.model "llama"), read as Idrak reads that kind.
/// </summary>
public static class GemmaGguf
{
    /// <summary>llama.cpp's "gemma": <c>GemmaForCausalLM</c>, the common tensor names, gains stored with the 1 added, GELU.</summary>
    public static GgufArchitecture Gemma() => new()
    {
        HuggingFace = GemmaPlugin.GemmaForCausalLM,
        StoredOffset = StoredOffset,
        Configure = (file, config, notes) => Configure(file, config, notes, version: 1, ImpliedQueryScalar),
    };

    /// <summary>
    /// llama.cpp's "gemma2": <c>Gemma2ForCausalLM</c> with its four norms per layer (<see cref="TensorName"/>), gains stored
    /// with the 1 added, GELU, soft-capping and the sliding window from the file when it has them.
    /// </summary>
    /// <param name="queryScalar">
    /// The query scalar (query_pre_attn_scalar: scores are scaled by its inverse square root) of a file, given the file
    /// and the configuration read so far; null takes <see cref="ImpliedQueryScalar"/>.
    /// </param>
    public static GgufArchitecture Gemma2(Func<GgufFile, JsonObject, float>? queryScalar = null) => new()
    {
        HuggingFace = GemmaPlugin.Gemma2ForCausalLM,
        TensorName = TensorName,
        StoredOffset = StoredOffset,
        Configure = (file, config, notes) => Configure(file, config, notes, version: 2, queryScalar ?? ImpliedQueryScalar),
    };

    /// <summary>
    /// llama.cpp's "gemma3": <c>Gemma3ForCausalLM</c> with its four norms per layer and its query and key norms, gains
    /// stored with the 1 added, GELU, the sliding window with its pattern and the windowed layers' RoPE base from the file
    /// when it has them (the global layers' base and scaling are the file's usual RoPE keys).
    /// </summary>
    /// <param name="queryScalar">As for <see cref="Gemma2"/>.</param>
    public static GgufArchitecture Gemma3(Func<GgufFile, JsonObject, float>? queryScalar = null) => new()
    {
        HuggingFace = GemmaPlugin.Gemma3ForCausalLM,
        TensorName = TensorName,
        StoredOffset = StoredOffset,
        Configure = (file, config, notes) => Configure(file, config, notes, version: 3, queryScalar ?? ImpliedQueryScalar),
    };

    /// <summary>
    /// llama.cpp's names of Gemma 2 and Gemma 3 tensors → the Hugging Face ones: the norm after attention
    /// (blk.N.post_attention_norm → post_attention_layernorm), the one before the feed-forward block (blk.N.ffn_norm →
    /// pre_feedforward_layernorm) and the one after it (blk.N.post_ffw_norm → post_feedforward_layernorm); the rest as
    /// <see cref="GgufArchitecture.CommonTensorName"/> names them.
    /// </summary>
    public static string? TensorName(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        if (name.Split('.') is ["blk", var layer, var part, "weight"])
        {
            string? norm = part switch
            {
                "post_attention_norm" => "post_attention_layernorm",
                "ffn_norm" => "pre_feedforward_layernorm",
                "post_ffw_norm" => "post_feedforward_layernorm",
                _ => null,
            };
            if (norm is not null)
            {
                return $"model.layers.{layer}.{norm}.weight";
            }
        }

        return GgufArchitecture.CommonTensorName(name);
    }

    /// <summary>
    /// What llama.cpp's converter added to a Gemma tensor, by Hugging Face name: 1 to every norm gain (a name ending in
    /// "norm.weight": the layers' norms, the query and key norms, the final norm), as Gemma's norms scale by (1 + gain);
    /// 0 to the rest.
    /// </summary>
    public static float StoredOffset(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        return name.EndsWith("norm.weight", StringComparison.Ordinal) ? 1f : 0f;
    }

    /// <summary>
    /// The query scalar llama.cpp uses for a Gemma 2 or Gemma 3 file (its files do not store one): the width per head
    /// (hidden_size / num_attention_heads) for the layer counts llama.cpp reads as the family's widest size (46 layers for
    /// gemma2, 62 for gemma3), the head width (head_dim) otherwise, as the models were trained.
    /// </summary>
    public static float ImpliedQueryScalar(GgufFile file, JsonObject config)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(config);
        string arch = (string?)config["model_type"] ?? "";
        int layers = (int?)config["num_hidden_layers"] ?? 0;
        bool widest = (arch, layers) is ("gemma2", 46) or ("gemma3", 62);
        int heads = (int?)config["num_attention_heads"] ?? 0;
        return widest && heads > 0 ? (int)config["hidden_size"]! / heads : (int)config["head_dim"]!;
    }

    // The family's keys in the configuration, from the file's (each only when the file has it; the family's defaults
    // in GemmaArchitectures are llama.cpp's for a file without it, except where set here).
    private static void Configure(GgufFile file, JsonObject config, List<string> notes, int version, Func<GgufFile, JsonObject, float> queryScalar)
    {
        string arch = (string?)config["model_type"] ?? "";
        bool Has(string key) => file.Metadata.ContainsKey($"{arch}.{key}");
        config["hidden_act"] = "gelu_pytorch_tanh";                // llama.cpp's Gemma feed-forward: GELU, tanh approximation
        if (version == 1)
        {
            return;
        }

        config["query_pre_attn_scalar"] = queryScalar(file, config);
        if (version == 2 && Has("attn_logit_softcapping"))
        {
            config["attn_logit_softcapping"] = file.Get($"{arch}.attn_logit_softcapping", 0.0);
        }

        if (Has("final_logit_softcapping"))
        {
            config["final_logit_softcapping"] = file.Get($"{arch}.final_logit_softcapping", 0.0);
        }

        int layers = (int?)config["num_hidden_layers"] ?? 0;
        int window = file.Get($"{arch}.attention.sliding_window", 0);
        if (version == 3 && window <= 0)
        {
            // Gemma 3 without a window: every layer global, as llama.cpp reads such a file.
            config["layer_types"] = new JsonArray([.. Enumerable.Range(0, layers).Select(_ => (JsonNode?)"full_attention")]);
            return;
        }

        if (window > 0)
        {
            config["sliding_window"] = window;
        }

        // The windowed layers: an array of flags, or every n-th layer global (llama.cpp's pattern, the last of each n).
        if (file.Metadata.TryGetValue($"{arch}.attention.sliding_window_pattern", out var pattern))
        {
            bool[]? windowed = pattern switch
            {
                bool[] flags when flags.Length >= layers => flags[..layers],
                bool[] => throw new InvalidDataException($"{file.Path}: {arch}.attention.sliding_window_pattern has fewer flags than the {layers} layers."),
                _ when file.Get($"{arch}.attention.sliding_window_pattern", -1) is var n and >= 0 =>
                    [.. Enumerable.Range(0, layers).Select(i => n == 0 || i % n < n - 1)],
                _ => throw new InvalidDataException($"{file.Path}: {arch}.attention.sliding_window_pattern is neither flags nor a layer count."),
            };
            config["layer_types"] = new JsonArray([.. windowed.Select(w => (JsonNode?)(w ? "sliding_attention" : "full_attention"))]);
        }

        if (Has("rope.freq_base_swa"))
        {
            double local = file.Get($"{arch}.rope.freq_base_swa", 10000.0);
            if (version == 3)
            {
                config["rope_local_base_freq"] = local;
            }
            else if (local != file.Get($"{arch}.rope.freq_base", 10000.0))
            {
                notes.Add($"{arch}.rope.freq_base_swa ({local}) is not read: Gemma 2's windowed layers use the global RoPE base.");
            }
        }
    }
}
