// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Layers;

namespace Idrak.Models;

/// <summary>
/// The model families this assembly reads (registered in <see cref="PretrainedArchitectures"/>, Idrak.Abstraction, when
/// the registry is first used), and the pieces they are made of, for families of your own:
/// <see cref="LlamaStyle"/> for those sharing the Llama naming, the specs read from the common configuration keys
/// (<see cref="CommonSpec"/>, <see cref="ExpertSpec"/>) and the checkpoint names of each family.
/// </summary>
public static class PretrainedFamilies
{
    // Llama, Mistral, Qwen2, Qwen3, Gemma, Gemma 2 and Gemma 3 (text), and the mixture-of-experts families Mixtral,
    // Qwen2-MoE and Qwen3-MoE.
    internal static IEnumerable<(string Name, PretrainedArchitecture Architecture)> BuiltIns() =>
    [
        ("LlamaForCausalLM", LlamaStyle((config, spec, _) => spec)),
        ("MistralForCausalLM", LlamaStyle((config, spec, _) => spec)),
        ("Qwen2ForCausalLM", LlamaStyle((config, spec, _) => spec with { QkvBias = true })),
        ("Qwen3ForCausalLM", LlamaStyle((config, spec, _) => spec with { QkNorm = true })),
        ("GemmaForCausalLM", LlamaStyle((config, spec, _) => spec with
        {
            NormOffset = 1f,                                                  // Gemma scales by (1 + weight)
            EmbeddingScale = MathF.Sqrt(spec.Dim),
            TieEmbeddings = true,
        })),
        ("Gemma2ForCausalLM", new() { Spec = (config, notes) => GemmaSpec(config, CommonSpec(config, notes), version: 2), TensorName = GemmaTensorName }),
        ("Gemma3ForCausalLM", new() { Spec = (config, notes) => GemmaSpec(config, CommonSpec(config, notes), version: 3), TensorName = GemmaTensorName }),
        ("MixtralForCausalLM", new() { Spec = (config, notes) => ExpertSpec(config, notes, normalizeTopK: true), TensorName = MixtralTensorName }),
        ("Qwen2MoeForCausalLM", new() { Spec = (config, notes) => ExpertSpec(config, notes, normalizeTopK: false) with { QkvBias = true }, TensorName = QwenMoeTensorName }),
        ("Qwen3MoeForCausalLM", new() { Spec = (config, notes) => ExpertSpec(config, notes, normalizeTopK: false) with { QkNorm = true }, TensorName = QwenMoeTensorName }),
    ];

    /// <summary>
    /// An architecture with the Llama configuration keys and weight names (model.embed_tokens, model.layers.i.self_attn.q_proj,
    /// input_layernorm, post_attention_layernorm, mlp.gate_proj/up_proj/down_proj, model.norm, lm_head), which many
    /// families share. <paramref name="adjust"/> changes the spec read from the common keys (for example biases or q/k norm).
    /// </summary>
    public static PretrainedArchitecture LlamaStyle(Func<JsonObject, DecoderSpec, List<string>, DecoderSpec> adjust) => new()
    {
        Spec = (config, notes) => adjust(config, CommonSpec(config, notes), notes),
        TensorName = LlamaTensorName,
    };

    /// <summary>
    /// The spec from the configuration keys most families share (hidden_size, num_attention_heads, rope_theta, …). A
    /// configuration with experts is refused: families with experts read it with <see cref="ExpertSpec"/>.
    /// </summary>
    public static DecoderSpec CommonSpec(JsonObject c, List<string> notes)
    {
        if (c["num_local_experts"] is not null || (int?)c["num_experts"] is > 0)
        {
            throw new NotSupportedException("This configuration has experts, which this family does not read; mixture-of-experts families "
                + "(MixtralForCausalLM, Qwen2MoeForCausalLM, Qwen3MoeForCausalLM, or one registered with PretrainedFamilies.ExpertSpec) do.");
        }

        return Common(c, notes);
    }

    /// <summary>
    /// The spec of a mixture-of-experts model: the common keys (<see cref="CommonSpec"/>) and the experts' keys:
    /// num_local_experts (Mixtral) or num_experts (Qwen), num_experts_per_tok, moe_intermediate_size (each expert's
    /// hidden size; intermediate_size when absent), shared_expert_intermediate_size (a gated shared expert), norm_topk_prob
    /// (<paramref name="normalizeTopK"/> when absent), and which layers have experts (decoder_sparse_step, mlp_only_layers).
    /// </summary>
    public static DecoderSpec ExpertSpec(JsonObject c, List<string> notes, bool normalizeTopK)
    {
        var spec = Common(c, notes);
        int experts = (int?)c["num_local_experts"] ?? (int?)c["num_experts"] ?? throw new InvalidDataException("config.json has no 'num_local_experts' or 'num_experts'.");
        int perToken = (int?)c["num_experts_per_tok"] ?? throw new InvalidDataException("config.json has no 'num_experts_per_tok'.");
        if (experts < 1 || perToken < 1 || perToken > experts)
        {
            throw new InvalidDataException($"config.json routes each token to {perToken} of {experts} experts.");
        }

        // Qwen: layer i has experts unless listed in mlp_only_layers, and when (i + 1) is a multiple of decoder_sparse_step.
        int step = Math.Max(1, (int?)c["decoder_sparse_step"] ?? 1);
        var dense = c["mlp_only_layers"] is JsonArray only ? only.Select(l => (int)l!).ToHashSet() : [];
        bool[] layers = [.. Enumerable.Range(0, spec.Layers).Select(i => !dense.Contains(i) && (i + 1) % step == 0)];
        if ((double?)c["router_jitter_noise"] is > 0)
        {
            notes.Add("router_jitter_noise (noise on the router's scores while training) is not applied.");
        }

        return spec with
        {
            Experts = experts,
            ExpertsPerToken = perToken,
            ExpertFfDim = (int?)c["moe_intermediate_size"] ?? 0,
            SharedExpertFfDim = (int?)c["shared_expert_intermediate_size"] ?? 0,
            NormalizeTopK = (bool?)c["norm_topk_prob"] ?? normalizeTopK,
            ExpertLayers = layers.All(l => l) ? null : layers,
        };
    }

    // The common keys, experts or not.
    private static DecoderSpec Common(JsonObject c, List<string> notes)
    {
        int Int(string key) => (int?)c[key] ?? throw new InvalidDataException($"config.json has no '{key}'.");

        int dim = Int("hidden_size"), heads = Int("num_attention_heads");
        int headDim = (int?)c["head_dim"] ?? dim / heads;
        int maxPositions = (int?)c["max_position_embeddings"] ?? 4096;
        float theta = (float?)c["rope_theta"] ?? 10000f;
        int? rotary = c["partial_rotary_factor"] is { } factor ? (int)(headDim * (float)factor) : null;
        RopeScaling? scaling = null;
        if (c["rope_scaling"] is JsonObject s)
        {
            string type = (string?)s["rope_type"] ?? (string?)s["type"] ?? "default";
            if (type != "default")
            {
                if (!RopeScalings.Contains(type))
                {
                    throw new NotSupportedException($"RoPE scaling '{type}' is not registered ({string.Join(", ", RopeScalings.Names)}); add it with RopeScalings.Register, "
                        + "or remove rope_scaling to use the model within its original context.");
                }

                // The parameters as the configuration gives them, plus the model's context length, which several methods
                // fall back on (yarn without original_max_position_embeddings; dynamic always).
                var parameters = (JsonObject)s.DeepClone();
                parameters.Remove("type");
                parameters.Remove("rope_type");
                parameters["max_position_embeddings"] ??= maxPositions;
                scaling = new RopeScaling(type, parameters);
            }
        }

        // Sliding-window attention, as transformers reads it: the window (none when use_sliding_window is false, as Qwen2
        // and Qwen3 have it), in the layers layer_types marks "sliding_attention", else from max_window_layers on (Qwen2,
        // Qwen3), else in every layer (Mistral).
        int layers = Int("num_hidden_layers");
        int? window = (bool?)c["use_sliding_window"] ?? true ? (int?)c["sliding_window"] : null;
        bool[]? windowed = null;
        if (window is not null && c["layer_types"] is JsonArray types)
        {
            windowed = [.. types.Select(type => (string?)type == "sliding_attention")];
        }
        else if (window is not null && (int?)c["max_window_layers"] is int fullLayers)
        {
            windowed = [.. Enumerable.Range(0, layers).Select(i => i >= fullLayers)];
        }

        if (windowed is not null && !windowed.Contains(true))
        {
            (window, windowed) = (null, null);
        }

        string activation = (string?)c["hidden_act"] ?? (string?)c["hidden_activation"] ?? "silu";
        if (activation == "gelu")
        {
            notes.Add("Exact GELU is computed with the tanh approximation (differences below 0.001).");
        }

        bool attentionBias = (bool?)c["attention_bias"] ?? false;
        return new DecoderSpec
        {
            Vocabulary = Int("vocab_size"), Dim = dim, Layers = layers, Heads = heads,
            KvHeads = (int?)c["num_key_value_heads"] ?? heads, HeadDim = headDim, FfDim = Int("intermediate_size"), MaxPositions = maxPositions,
            NormEpsilon = (float?)c["rms_norm_eps"] ?? 1e-6f,
            Activation = activation switch
            {
                "silu" or "swish" => FeedForwardActivation.Silu,
                "gelu" or "gelu_new" or "gelu_pytorch_tanh" or "gelu_fast" => FeedForwardActivation.Gelu,
                "relu" => FeedForwardActivation.Relu,
                _ => throw new NotSupportedException($"Activation '{activation}' is not supported."),
            },
            Rope = new RopeSettings(theta, rotary, Interleaved: false, scaling),
            QkvBias = attentionBias, OutputBias = attentionBias, FeedForwardBias = (bool?)c["mlp_bias"] ?? false,
            TieEmbeddings = (bool?)c["tie_word_embeddings"] ?? false,
            SlidingWindow = window, SlidingWindowLayers = windowed,
        };
    }

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

    /// <summary>
    /// Idrak weight names → Gemma 2 / Gemma 3 checkpoint names: the Llama names, with four normalizations per layer
    /// (input_layernorm, post_attention_layernorm after attention, pre_feedforward_layernorm, post_feedforward_layernorm).
    /// </summary>
    public static string? GemmaTensorName(string name)
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

        return LlamaTensorName(name);
    }

    /// <summary>
    /// Idrak weight names → Mixtral checkpoint names: the Llama names, with each layer's experts under block_sparse_moe
    /// (the router "gate", and experts.j.w1 / w3 / w2 for each expert's gate, up and down projections).
    /// </summary>
    public static string? MixtralTensorName(string name)
    {
        var parts = name.Split('.');
        if (parts.Length >= 5 && parts[0] == "layers" && parts[2] == "mlp")
        {
            string layer = $"model.layers.{parts[1]}.block_sparse_moe";
            if (parts is [_, _, _, "router", var kind])
            {
                return $"{layer}.gate.{kind}";
            }

            if (parts is [_, _, _, "experts", var expert, var projection, var kind2])
            {
                string? stored = projection switch { "gate" => "w1", "up" => "w3", "down" => "w2", _ => null };
                return stored is null ? null : $"{layer}.experts.{expert}.{stored}.{kind2}";
            }
        }

        return LlamaTensorName(name);
    }

    /// <summary>
    /// Idrak weight names → Qwen2-MoE / Qwen3-MoE checkpoint names: the Llama names, with the router as mlp.gate, the
    /// experts as mlp.experts.j.gate_proj / up_proj / down_proj, the shared expert as mlp.shared_expert.*_proj and its
    /// gate as mlp.shared_expert_gate.
    /// </summary>
    public static string? QwenMoeTensorName(string name)
    {
        var parts = name.Split('.');
        if (parts.Length >= 5 && parts[0] == "layers" && parts[2] == "mlp")
        {
            string layer = $"model.layers.{parts[1]}.mlp";
            switch (parts)
            {
                case [_, _, _, "router", var kind]:
                    return $"{layer}.gate.{kind}";
                case [_, _, _, "shared_gate", var kind]:
                    return $"{layer}.shared_expert_gate.{kind}";
                case [_, _, _, "shared", var projection, var kind]:
                    return $"{layer}.shared_expert.{projection}_proj.{kind}";
                case [_, _, _, "experts", var expert, var projection, var kind]:
                    return $"{layer}.experts.{expert}.{projection}_proj.{kind}";
            }
        }

        return LlamaTensorName(name);
    }

    /// <summary>Idrak weight names → Llama-style checkpoint names.</summary>
    public static string? LlamaTensorName(string name)
    {
        if (name == "embed.weight")
        {
            return "model.embed_tokens.weight";
        }

        if (name == "norm.weight")
        {
            return "model.norm.weight";
        }

        if (name.StartsWith("head.", StringComparison.Ordinal))
        {
            return "lm_head." + name[5..];
        }

        // layers.{i}.{part}
        var parts = name.Split('.');
        if (parts.Length < 4 || parts[0] != "layers")
        {
            return null;
        }

        string layer = $"model.layers.{parts[1]}", rest = string.Join('.', parts[2..]);
        return rest switch
        {
            "attn_norm.weight" => $"{layer}.input_layernorm.weight",
            "mlp_norm.weight" => $"{layer}.post_attention_layernorm.weight",
            "attn.q_norm.weight" => $"{layer}.self_attn.q_norm.weight",
            "attn.k_norm.weight" => $"{layer}.self_attn.k_norm.weight",
            _ when rest.StartsWith("attn.", StringComparison.Ordinal) => $"{layer}.self_attn.{parts[3]}_proj.{parts[^1]}",
            _ when rest.StartsWith("mlp.", StringComparison.Ordinal) => $"{layer}.mlp.{parts[3]}_proj.{parts[^1]}",
            _ => null,
        };
    }
}

/// <summary>Reads a model's weights from a safetensors checkpoint through an architecture's name mapping.</summary>
internal sealed class CheckpointWeights(ITensorStore reader, PretrainedArchitecture architecture) : IWeightSource
{
    public HashSet<string> Used { get; } = [];

    /// <summary>A PEFT adapter whose updates are added to the weights as they are read (null for none).</summary>
    public AdapterMerge? Adapter { get; init; }

    public float[]? Read(string name, IReadOnlyList<int> shape)
    {
        if (architecture.TensorName(name) is not { } stored || !reader.Contains(stored))
        {
            return null;
        }

        Used.Add(stored);
        var storedShape = reader.ShapeOf(stored);
        bool transposed = architecture.Transposed(name);
        int[] expected = transposed ? [.. shape.Reverse()] : [.. shape];
        if (!storedShape.SequenceEqual(expected))
        {
            throw new InvalidDataException($"'{stored}' is [{string.Join(", ", storedShape)}]; the model expects [{string.Join(", ", expected)}] for {name}.");
        }

        // Transposed while reading: one full array per tensor instead of the stored order plus its transpose.
        var values = transposed ? reader.ReadTransposed(stored) : reader.Read(stored);

        if (transposed && Adapter is not null && stored.EndsWith(".weight", StringComparison.Ordinal))
        {
            Adapter.AddTo(stored[..^".weight".Length], values, shape[0], shape[1]);
        }

        return values;
    }
}
