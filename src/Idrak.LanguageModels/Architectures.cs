// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Layers;

namespace Idrak.LanguageModels;

/// <summary>
/// How to read one family of pretrained models: its configuration (a Hugging Face <c>config.json</c>) becomes a
/// <see cref="DecoderSpec"/>, and each of Idrak's weight names (see <see cref="DecoderSpec"/>) is found in the
/// checkpoint. A family that does not fit <see cref="DecoderSpec"/> builds its own network with <see cref="Build"/>.
/// Register new families with <see cref="PretrainedArchitectures.Register"/>.
/// </summary>
public sealed class PretrainedArchitecture
{
    /// <summary>
    /// The model described by a configuration; append anything approximated to the notes. With <see cref="Build"/>, the
    /// spec still gives the model's sizes (vocabulary, width, layers, heads, context length) to the tokenizer, generation
    /// and tools, and is not built.
    /// </summary>
    public required Func<JsonObject, List<string>, DecoderSpec> Spec { get; init; }

    /// <summary>
    /// Builds the network itself instead of <see cref="DecoderSpec.Build"/>, or null (the default) to build the spec. The
    /// network takes token ids [batch, time] and returns logits [batch, time, vocabulary]; to work with the KV cache its
    /// attention layers implement <see cref="ICachedModule"/>, and to be fine-tuned it ends with its output head (a
    /// <see cref="Linear"/>).
    /// </summary>
    public Func<PretrainedBuildContext, Sequential>? Build { get; init; }

    /// <summary>The checkpoint's name for one of Idrak's weight names (null when the checkpoint does not store it).</summary>
    public required Func<string, string?> TensorName { get; init; }

    /// <summary>
    /// Whether the checkpoint stores this weight as [out, in] (the PyTorch Linear layout), so it is transposed to
    /// Idrak's [in, out]. By default: every projection (attention, feed-forward, head) is transposed.
    /// </summary>
    public Func<string, bool> Transposed { get; init; } = name =>
        name.Contains(".attn.", StringComparison.Ordinal) && !name.Contains("_norm", StringComparison.Ordinal) && name.EndsWith(".weight", StringComparison.Ordinal)
        || name.Contains(".mlp.", StringComparison.Ordinal) && name.EndsWith(".weight", StringComparison.Ordinal)
        || name == "head.weight";
}

/// <summary>What <see cref="PretrainedArchitecture.Build"/> receives.</summary>
/// <param name="Config">The model's config.json.</param>
/// <param name="Spec">The spec <see cref="PretrainedArchitecture.Spec"/> read from it.</param>
/// <param name="Weights">
/// The checkpoint's tensors by Idrak's names, through <see cref="PretrainedArchitecture.TensorName"/> and
/// <see cref="PretrainedArchitecture.Transposed"/> (with a merged adapter's updates added), as <see cref="DecoderSpec.Build"/> reads them.
/// </param>
/// <param name="Checkpoint">The checkpoint's tensors by their stored names, as stored.</param>
/// <param name="Options">Device, packed weight format and context length chosen by the caller.</param>
/// <param name="Notes">Append anything approximated (shown in <see cref="PretrainedModel.Notes"/>).</param>
public sealed record PretrainedBuildContext(JsonObject Config, DecoderSpec Spec, IWeightSource Weights, ITensorStore Checkpoint,
    DecoderBuildOptions Options, List<string> Notes);

/// <summary>
/// The model families <see cref="PretrainedModel.Load"/> knows, by the architecture name in <c>config.json</c>
/// ("architectures": [...]). Llama, Mistral, Qwen2, Qwen3, Gemma, Gemma 2 and Gemma 3 (text) are registered; add others
/// with <see cref="Register"/>, usually with <see cref="LlamaStyle"/> when they share the Llama naming.
/// </summary>
public static class PretrainedArchitectures
{
    private static readonly Dictionary<string, PretrainedArchitecture> Registry = new(StringComparer.Ordinal)
    {
        ["LlamaForCausalLM"] = LlamaStyle((config, spec, _) => spec),
        ["MistralForCausalLM"] = LlamaStyle((config, spec, _) => spec),
        ["Qwen2ForCausalLM"] = LlamaStyle((config, spec, _) => spec with { QkvBias = true }),
        ["Qwen3ForCausalLM"] = LlamaStyle((config, spec, _) => spec with { QkNorm = true }),
        ["GemmaForCausalLM"] = LlamaStyle((config, spec, _) => spec with
        {
            NormOffset = 1f,                                                  // Gemma scales by (1 + weight)
            EmbeddingScale = MathF.Sqrt(spec.Dim),
            TieEmbeddings = true,
        }),
        ["Gemma2ForCausalLM"] = new() { Spec = (config, notes) => GemmaSpec(config, CommonSpec(config, notes), version: 2), TensorName = GemmaTensorName },
        ["Gemma3ForCausalLM"] = new() { Spec = (config, notes) => GemmaSpec(config, CommonSpec(config, notes), version: 3), TensorName = GemmaTensorName },
    };

    /// <summary>Registers (or replaces) how to read the architecture <paramref name="name"/>.</summary>
    public static void Register(string name, PretrainedArchitecture architecture)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(architecture);
        lock (Registry)
        {
            Registry[name] = architecture;
        }
    }

    /// <summary>Removes the architecture <paramref name="name"/>; returns whether it was registered.</summary>
    public static bool Unregister(string name)
    {
        lock (Registry)
        {
            return Registry.Remove(name);
        }
    }

    /// <summary>The registered architecture names.</summary>
    public static IReadOnlyCollection<string> Names
    {
        get
        {
            lock (Registry)
            {
                return [.. Registry.Keys];
            }
        }
    }

    /// <summary>The architecture registered as <paramref name="name"/>.</summary>
    public static PretrainedArchitecture Get(string name)
    {
        lock (Registry)
        {
            return Registry.TryGetValue(name, out var architecture) ? architecture
                : throw new NotSupportedException($"No architecture '{name}' is registered ({string.Join(", ", Registry.Keys)}); add it with PretrainedArchitectures.Register.");
        }
    }

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

    /// <summary>The spec from the configuration keys most families share (hidden_size, num_attention_heads, rope_theta, …).</summary>
    public static DecoderSpec CommonSpec(JsonObject c, List<string> notes)
    {
        int Int(string key) => (int?)c[key] ?? throw new InvalidDataException($"config.json has no '{key}'.");
        if (c["num_local_experts"] is not null || c["num_experts"] is not null)
        {
            throw new NotSupportedException("Mixture-of-experts models are not supported.");
        }

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

/// <summary>
/// Folds a PEFT LoRA adapter (adapter_config.json, adapter_model.safetensors) into weights while they are read, so a
/// fine-tuned model is quantized with its update included and runs as fast as the base model.
/// </summary>
internal sealed class AdapterMerge : IDisposable
{
    private readonly SafeTensorsReader _reader;
    private readonly float _scale;

    public AdapterMerge(string folder)
    {
        var config = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "adapter_config.json")))!.AsObject();
        int rank = (int?)config["r"] ?? throw new InvalidDataException("adapter_config.json has no r.");
        float alpha = (float?)config["lora_alpha"] ?? rank;
        _scale = (bool?)config["use_rslora"] == true ? alpha / MathF.Sqrt(rank) : alpha / rank;
        _reader = SafeTensorsReader.Open(Path.Combine(folder, "adapter_model.safetensors"));
    }

    /// <summary>How many weights received an update.</summary>
    public int Merged { get; private set; }

    /// <summary>
    /// Adds scale · (B·A)ᵀ to <paramref name="weight"/> ([inputs, outputs], Idrak's layout) when the adapter has
    /// lora_A [r, inputs] and lora_B [outputs, r] for the checkpoint module <paramref name="module"/>.
    /// </summary>
    public void AddTo(string module, float[] weight, int inputs, int outputs)
    {
        string prefix = "base_model.model." + module;
        string a = $"{prefix}.lora_A.weight", b = $"{prefix}.lora_B.weight";
        if (!_reader.Contains(a))
        {
            (a, b) = ($"{prefix}.lora_A.default.weight", $"{prefix}.lora_B.default.weight");
            if (!_reader.Contains(a))
            {
                return;
            }
        }

        var down = _reader.Read(a);                                           // [r, inputs]
        int rank = down.Length / inputs;
        var up = Idrak.HostParallel.Transpose(_reader.Read(b), outputs, rank); // [r, outputs]
        float scale = _scale;
        Parallel.For(0, inputs, i =>
        {
            var row = weight.AsSpan(i * outputs, outputs);
            for (int k = 0; k < rank; k++)
            {
                float factor = down[k * inputs + i] * scale;
                if (factor == 0)
                {
                    continue;
                }

                AddScaled(row, up.AsSpan(k * outputs, outputs), factor);
            }
        });
        Merged++;
    }

    // row += source · factor
    private static void AddScaled(Span<float> row, ReadOnlySpan<float> source, float factor)
    {
        int j = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated)
        {
            var rows = System.Runtime.InteropServices.MemoryMarshal.Cast<float, System.Numerics.Vector<float>>(row);
            var sources = System.Runtime.InteropServices.MemoryMarshal.Cast<float, System.Numerics.Vector<float>>(source);
            var f = new System.Numerics.Vector<float>(factor);
            for (int v = 0; v < rows.Length; v++)
            {
                rows[v] += sources[v] * f;
            }

            j = rows.Length * System.Numerics.Vector<float>.Count;
        }

        for (; j < row.Length; j++)
        {
            row[j] += source[j] * factor;
        }
    }

    public void Dispose() => _reader.Dispose();
}
