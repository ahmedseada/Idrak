// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak.Layers;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// The model families this assembly reads (registered in <see cref="PretrainedArchitectures"/>, Idrak.Abstraction, when
/// the registry is first used), and the pieces they are made of, for families of your own:
/// <see cref="LlamaStyle(Func{JsonObject, DecoderSpec, List{string}, DecoderSpec}, string)"/> for those sharing the Llama naming, the specs read from the common configuration keys
/// (<see cref="CommonSpec"/>, <see cref="ExpertSpec"/>) and the checkpoint names of each family.
/// </summary>
public static class PretrainedFamilies
{
    // Llama, Mistral, Qwen2, Qwen3, Gemma, Gemma 2 and Gemma 3 (text, and the vision-language model's text decoder: its
    // vision part is a VisionFamilies registration the library does not make), and the mixture-of-experts families Mixtral, Qwen2-MoE and Qwen3-MoE.
    internal static IEnumerable<(string Name, PretrainedArchitecture Architecture)> BuiltIns() =>
    [
        ("LlamaForCausalLM", LlamaStyle((config, spec, _) => spec, "llama")),
        ("MistralForCausalLM", LlamaStyle((config, spec, _) => spec, "mistral")),
        ("Qwen2ForCausalLM", LlamaStyle((config, spec, _) => spec with { QkvBias = true }, "qwen2")),
        ("Qwen3ForCausalLM", LlamaStyle((config, spec, _) => spec with { QkNorm = true }, "qwen3")),
        ("GemmaForCausalLM", LlamaStyle((config, spec, _) => spec with
        {
            NormOffset = 1f,                                                  // Gemma scales by (1 + weight)
            EmbeddingScale = MathF.Sqrt(spec.Dim),
            TieEmbeddings = true,
        }, "gemma")),
        ("Gemma2ForCausalLM", new() { Spec = (config, notes) => GemmaSpec(config, CommonSpec(config, notes), version: 2), TensorName = GemmaTensorName }),
        ("Gemma3ForCausalLM", new() { Spec = (config, notes) => Gemma3Spec(Gemma3Keys(config), notes), TensorName = GemmaTensorName }),
        ("Gemma3ForConditionalGeneration", Gemma3VisionLanguage(Gemma3Layouts[0])),
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
    /// <see cref="LlamaStyle(Func{JsonObject, DecoderSpec, List{string}, DecoderSpec})"/> that can also write a spec as
    /// config.json (<see cref="PretrainedArchitecture.Config"/>, through <see cref="CommonConfig"/> with
    /// <paramref name="modelType"/>), so models of the family can be made from a spec.
    /// </summary>
    public static PretrainedArchitecture LlamaStyle(Func<JsonObject, DecoderSpec, List<string>, DecoderSpec> adjust, string modelType)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelType);
        var architecture = LlamaStyle(adjust);
        return new PretrainedArchitecture
        {
            Spec = architecture.Spec,
            TensorName = architecture.TensorName,
            Config = spec => CheckedConfig(spec, CommonConfig(spec, modelType), architecture.Spec, modelType),
        };
    }

    /// <summary>
    /// A spec as the common configuration keys (the inverse of <see cref="CommonSpec"/>): hidden_size,
    /// num_hidden_layers, num_attention_heads, num_key_value_heads, head_dim, intermediate_size, vocab_size,
    /// max_position_embeddings, rms_norm_eps, rope_theta (and partial_rotary_factor, rope_scaling), hidden_act,
    /// attention_bias, mlp_bias, tie_word_embeddings and the sliding window (sliding_window, layer_types), with
    /// <paramref name="modelType"/> as "model_type". What the keys cannot say (a norm other than RMS, learned positions,
    /// experts, …) is the caller's to check (a family's <see cref="PretrainedArchitecture.Config"/> reads the result back).
    /// </summary>
    public static JsonObject CommonConfig(DecoderSpec spec, string modelType)
    {
        ArgumentNullException.ThrowIfNull(spec);
        var c = new JsonObject
        {
            ["model_type"] = modelType,
            ["vocab_size"] = spec.Vocabulary,
            ["hidden_size"] = spec.Dim,
            ["num_hidden_layers"] = spec.Layers,
            ["num_attention_heads"] = spec.Heads,
            ["num_key_value_heads"] = spec.KvHeads,
            ["head_dim"] = spec.HeadDim,
            ["intermediate_size"] = spec.FfDim,
            ["max_position_embeddings"] = spec.MaxPositions,
            ["rms_norm_eps"] = spec.NormEpsilon,
            ["hidden_act"] = spec.Activation switch
            {
                FeedForwardActivation.Silu => "silu",
                FeedForwardActivation.Gelu => "gelu_pytorch_tanh",
                FeedForwardActivation.Relu => "relu",
                _ => throw new NotSupportedException($"Activation {spec.Activation} has no configuration name."),
            },
            ["attention_bias"] = spec.OutputBias,
            ["mlp_bias"] = spec.FeedForwardBias,
            ["tie_word_embeddings"] = spec.TieEmbeddings,
        };
        if (spec.Rope is { } rope)
        {
            c["rope_theta"] = rope.Theta;
            if (rope.RotaryDim is int rotary)
            {
                c["partial_rotary_factor"] = (double)rotary / spec.HeadDim;
            }

            if (rope.Scaling is { } scaling)
            {
                var parameters = scaling.Parameters;
                if ((int?)parameters["max_position_embeddings"] == spec.MaxPositions)
                {
                    parameters.Remove("max_position_embeddings");     // the reader puts the context length back
                }

                parameters["rope_type"] = scaling.Type;
                c["rope_scaling"] = parameters;
            }
        }

        if (spec.SlidingWindow is int window)
        {
            c["sliding_window"] = window;
            if (spec.SlidingWindowLayers is { } layers)
            {
                var types = new JsonArray();
                foreach (bool windowed in layers)
                {
                    types.Add((JsonNode)JsonValue.Create(windowed ? "sliding_attention" : "full_attention"));
                }

                c["layer_types"] = types;
            }
        }

        // As a config.json read from disk is: numbers that read as any numeric type (a float written here reads as a double there).
        return JsonNode.Parse(c.ToJsonString())!.AsObject();
    }

    // A family's configuration of a spec, read back by the family and refused when it does not say the same spec (the
    // dropout, a training setting no configuration key holds, aside).
    private static JsonObject CheckedConfig(DecoderSpec spec, JsonObject config, Func<JsonObject, List<string>, DecoderSpec> read, string family)
    {
        DecoderSpec back;
        try
        {
            back = read(config, []);
        }
        catch (Exception e) when (e is NotSupportedException or InvalidDataException)
        {
            throw new NotSupportedException($"The {family} configuration cannot hold this spec: {e.Message}", e);
        }

        var expected = (spec with { Dropout = 0f }).ToJson();
        var actual = (back with { Dropout = 0f }).ToJson();
        if (expected["rope"]?["scaling"]?["parameters"] is JsonObject parameters)
        {
            parameters["max_position_embeddings"] ??= spec.MaxPositions;   // what the reader adds, and the scalings fall back on
        }

        if (!JsonNode.DeepEquals(expected, actual))
        {
            var differ = expected.Select(p => p.Key).Union(actual.Select(p => p.Key)).Where(k => !JsonNode.DeepEquals(expected[k], actual[k]));
            throw new NotSupportedException($"The {family} configuration cannot hold this spec: "
                + string.Join("; ", differ.Select(k => $"{k} would read back as {actual[k]?.ToJsonString() ?? "nothing"}, not {expected[k]?.ToJsonString() ?? "nothing"}")) + ".");
        }

        return config;
    }

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

    private static DecoderSpec Gemma3Spec(JsonObject c, List<string> notes) => GemmaSpec(c, CommonSpec(c, notes), version: 3);

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
    // vision part's namings are its vision family's (VisionFamilies), which the library does not register.
    private sealed record Gemma3Layout(string Name, string Text, string Head);

    private static readonly Gemma3Layout[] Gemma3Layouts =
    [
        new("save_pretrained (language_model.model.*)", "language_model.model.", "language_model.lm_head."),
        new("state dict of transformers 4.52 to 4.57 (model.language_model.*)", "model.language_model.", "lm_head."),
    ];

    // The text decoder of Gemma 3 with images: Gemma 3's, read from text_config, never soft-capping its logits
    // (Gemma3ForConditionalGeneration does not, whatever final_logit_softcapping says), in whichever naming the checkpoint
    // uses. The vision part is read by a registered vision family of the same name (none in the library).
    private static PretrainedArchitecture Gemma3VisionLanguage(Gemma3Layout layout) => new()
    {
        Spec = (config, notes) =>
        {
            var text = Gemma3TextConfig(config);
            var spec = Gemma3Spec(text, notes);
            // Text embeddings are scaled by √width held in the weights' type (Gemma3TextScaledWordEmbedding): √2560 rounds in bfloat16.
            return spec with { LogitSoftcap = null, EmbeddingScale = InType(MathF.Sqrt(spec.Dim), (string?)text["dtype"] ?? (string?)text["torch_dtype"]) };
        },
        TensorName = name => GemmaTensorName(name) switch
        {
            null => null,
            var stored when stored.StartsWith("model.", StringComparison.Ordinal) => layout.Text + stored["model.".Length..],
            var stored => layout.Head + stored["lm_head.".Length..],
        },
        ForCheckpoint = names =>
        {
            var texts = Gemma3Layouts.Where(l => names.Contains(l.Text + "embed_tokens.weight")).ToList();
            if (texts.Count == 0)
            {
                throw new InvalidDataException("This Gemma3ForConditionalGeneration checkpoint has no text decoder under a known name ("
                    + string.Join(", ", Gemma3Layouts.Select(l => l.Text + "embed_tokens.weight").Distinct()) + ").");
            }

            return Gemma3VisionLanguage(texts[0]);
        },
    };

    // text_config read as a Gemma3ForCausalLM configuration: the keys transformers keeps at the top level (tied embeddings,
    // the weights' type) when text_config lacks them, then Gemma3TextConfig's defaults for absent keys (the original
    // gemma-3-4b-it's text_config names only its width, layers, MLP, window and RoPE scaling).
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

/// <summary>
/// Reads a model's weights from a safetensors checkpoint through an architecture's name mapping; bfloat16 weights of
/// bfloat16 tensors as their stored bits (<see cref="ReadBFloat16"/>). Keeps the time spent reading and merging, for
/// <see cref="PretrainedModel.LoadTimes"/>.
/// </summary>
internal sealed class CheckpointWeights(ITensorStore reader, PretrainedArchitecture architecture) : IWeightSource
{
    private long _readTicks, _mergeTicks, _uploadTicks;

    public HashSet<string> Used { get; } = [];

    /// <summary>A PEFT adapter whose updates are added to the weights as they are read (null for none).</summary>
    public AdapterMerge? Adapter { get; init; }

    /// <summary>The time spent in the store's reads so far: its file reads and the conversion to the layer's layout.</summary>
    public TimeSpan ReadTime => Stopwatch.GetElapsedTime(0, _readTicks);

    /// <summary>The time spent adding the adapter's updates to the weights so far.</summary>
    public TimeSpan MergeTime => Stopwatch.GetElapsedTime(0, _mergeTicks);

    /// <summary>The time spent putting bfloat16 weights read as their bits on the device so far (memory and upload).</summary>
    public TimeSpan UploadTime => Stopwatch.GetElapsedTime(0, _uploadTicks);

    public float[]? Read(string name, IReadOnlyList<int> shape)
    {
        if (Stored(name, shape) is not { } stored)
        {
            return null;
        }

        // Transposed while reading: one full array per tensor instead of the stored order plus its transpose.
        bool transposed = architecture.Transposed(name);
        long start = Stopwatch.GetTimestamp();
        var values = transposed ? reader.ReadTransposed(stored) : reader.Read(stored);
        long read = Stopwatch.GetTimestamp();
        _readTicks += read - start;
        if (transposed && Adapter is not null && stored.EndsWith(".weight", StringComparison.Ordinal))
        {
            Adapter.AddTo(stored[..^".weight".Length], values, shape[0], shape[1]);
            _mergeTicks += Stopwatch.GetTimestamp() - read;
        }

        return values;
    }

    public BFloat16Weight? ReadBFloat16(string name, IReadOnlyList<int> shape, Device device, bool transposed = false)
    {
        if (Stored(name, shape) is not { } stored)
        {
            return null;
        }

        // A bfloat16 tensor's bits are the weights' bits (FromValues rounds each float32 copy back to itself), so they go
        // to the device as 16-bit values, transposed as such where the layout needs it. Not when the adapter updates the
        // tensor: its values change, so they are merged in float32 and rounded, as Read and the interface's default do.
        bool storedTransposed = architecture.Transposed(name);
        bool merged = storedTransposed && Adapter is not null && stored.EndsWith(".weight", StringComparison.Ordinal)
            && Adapter.Updates(stored[..^".weight".Length]);
        if (!merged && shape.Count == 2 && reader.FormatOf(stored) == WeightFormat.BFloat16)
        {
            int rows = transposed ? shape[1] : shape[0], columns = transposed ? shape[0] : shape[1];
            long begun = Stopwatch.GetTimestamp(), reading = 0;
            var weight = BFloat16Weight.FromBits(rows, columns, device, (Span<ushort> bits, int stride) =>
            {
                long start = Stopwatch.GetTimestamp();
                bool copied = reader.ReadBFloat16(stored, storedTransposed != transposed, bits, stride);
                reading = Stopwatch.GetTimestamp() - start;
                return copied;
            });
            _readTicks += reading;
            _uploadTicks += Stopwatch.GetTimestamp() - begun - reading;     // the weights' device memory and upload
            if (weight is not null)
            {
                return weight;
            }
        }

        var values = Read(name, shape)!;
        return transposed
            ? BFloat16Weight.FromValues(HostParallel.Transpose(values, shape[0], shape[1]), shape[1], shape[0], device)
            : BFloat16Weight.FromValues(values, shape[0], shape[1], device);
    }

    // The stored name of the weight `name` [shape] (marked used), or null when the checkpoint does not have it.
    private string? Stored(string name, IReadOnlyList<int> shape)
    {
        if (architecture.TensorName(name) is not { } stored || !reader.Contains(stored))
        {
            return null;
        }

        Used.Add(stored);
        var storedShape = reader.ShapeOf(stored);
        int[] expected = architecture.Transposed(name) ? [.. shape.Reverse()] : [.. shape];
        if (!storedShape.SequenceEqual(expected))
        {
            throw new InvalidDataException($"'{stored}' is [{string.Join(", ", storedShape)}]; the model expects [{string.Join(", ", expected)}] for {name}.");
        }

        return stored;
    }
}
