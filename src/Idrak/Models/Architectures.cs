// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak.Layers;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// The pieces model families are made of, for the families of plug-ins and apps (the library registers none in
/// <see cref="PretrainedArchitectures"/>; Idrak.Text.Plugins and Idrak.Vision.Plugins do):
/// <see cref="CommonArchitecture(Func{JsonObject, DecoderSpec, List{string}, DecoderSpec}, string)"/> for those sharing
/// the common configuration keys and checkpoint names, the specs read from the common configuration keys
/// (<see cref="CommonSpec"/>, <see cref="ExpertSpec"/>), the configuration of a spec (<see cref="CommonConfig"/>) and the
/// common checkpoint names (<see cref="CommonTensorName"/>).
/// </summary>
public static class PretrainedFamilies
{
    /// <summary>
    /// An architecture with the common configuration keys (<see cref="CommonSpec"/>) and weight names
    /// (<see cref="CommonTensorName"/>: model.embed_tokens, model.layers.i.self_attn.q_proj, input_layernorm,
    /// post_attention_layernorm, mlp.gate_proj/up_proj/down_proj, model.norm, lm_head), which many families share.
    /// <paramref name="adjust"/> changes the spec read from the common keys (for example biases or q/k norm).
    /// </summary>
    public static IPretrainedArchitecture CommonArchitecture(Func<JsonObject, DecoderSpec, List<string>, DecoderSpec> adjust)
    {
        ArgumentNullException.ThrowIfNull(adjust);
        return new CommonDecoder(adjust);
    }

    /// <summary>
    /// <see cref="CommonArchitecture(Func{JsonObject, DecoderSpec, List{string}, DecoderSpec})"/> that can also write a
    /// spec as config.json (<see cref="IConfigWritingArchitecture.Config"/>, through <see cref="CommonConfig"/> with
    /// <paramref name="modelType"/>), so models of the family can be made from a spec.
    /// </summary>
    public static IConfigWritingArchitecture CommonArchitecture(Func<JsonObject, DecoderSpec, List<string>, DecoderSpec> adjust, string modelType)
    {
        ArgumentNullException.ThrowIfNull(adjust);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelType);
        return new CommonDecoderWithConfig(adjust, modelType);
    }

    /// <summary>
    /// An architecture from its spec reader and its checkpoint names, with the usual layout
    /// (<see cref="PretrainedArchitecture.TransposedProjection"/>): for a family that only reads its configuration its own way
    /// or names its tensors its own way. A family with more to say implements <see cref="IPretrainedArchitecture"/> and the
    /// capabilities it has.
    /// </summary>
    public static IPretrainedArchitecture Architecture(Func<JsonObject, List<string>, DecoderSpec> spec, Func<string, string?> tensorName)
    {
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(tensorName);
        return new ReadDecoder(spec, tensorName);
    }

    /// <summary>The former name of <see cref="CommonArchitecture(Func{JsonObject, DecoderSpec, List{string}, DecoderSpec})"/>, kept for one release.</summary>
    [Obsolete("Renamed CommonArchitecture: it reads the common decoder keys and checkpoint names, whichever family uses them.")]
    public static IPretrainedArchitecture LlamaStyle(Func<JsonObject, DecoderSpec, List<string>, DecoderSpec> adjust) => CommonArchitecture(adjust);

    /// <summary>The former name of <see cref="CommonArchitecture(Func{JsonObject, DecoderSpec, List{string}, DecoderSpec}, string)"/>, kept for one release.</summary>
    [Obsolete("Renamed CommonArchitecture: it reads the common decoder keys and checkpoint names, whichever family uses them.")]
    public static IConfigWritingArchitecture LlamaStyle(Func<JsonObject, DecoderSpec, List<string>, DecoderSpec> adjust, string modelType) =>
        CommonArchitecture(adjust, modelType);

    // The common keys and names, the spec adjusted by the family.
    private class CommonDecoder(Func<JsonObject, DecoderSpec, List<string>, DecoderSpec> adjust) : PretrainedArchitecture
    {
        public override DecoderSpec Spec(JsonObject config, List<string> notes) => adjust(config, CommonSpec(config, notes), notes);

        public override string? TensorName(string name) => CommonTensorName(name);
    }

    // The common architecture that writes a spec back as the common keys, checked by reading them again.
    private sealed class CommonDecoderWithConfig(Func<JsonObject, DecoderSpec, List<string>, DecoderSpec> adjust, string modelType)
        : CommonDecoder(adjust), IConfigWritingArchitecture
    {
        public JsonObject Config(DecoderSpec spec) => CheckedConfig(spec, CommonConfig(spec, modelType), Spec, modelType);
    }

    // A family's own spec reader and names.
    private sealed class ReadDecoder(Func<JsonObject, List<string>, DecoderSpec> spec, Func<string, string?> tensorName) : PretrainedArchitecture
    {
        public override DecoderSpec Spec(JsonObject config, List<string> notes) => spec(config, notes);

        public override string? TensorName(string name) => tensorName(name);
    }

    /// <summary>
    /// A spec as the common configuration keys (the inverse of <see cref="CommonSpec"/>): hidden_size,
    /// num_hidden_layers, num_attention_heads, num_key_value_heads, head_dim, intermediate_size, vocab_size,
    /// max_position_embeddings, rms_norm_eps, rope_theta (and partial_rotary_factor, rope_scaling), hidden_act,
    /// attention_bias, mlp_bias, tie_word_embeddings and the sliding window (sliding_window, layer_types), with
    /// <paramref name="modelType"/> as "model_type". What the keys cannot say (a norm other than RMS, learned positions,
    /// experts, …) is the caller's to check (a family's <see cref="IConfigWritingArchitecture.Config"/> reads the result back).
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
                + "(Idrak.Text.Plugins' MixtralForCausalLM, Qwen2MoeForCausalLM, Qwen3MoeForCausalLM, or one registered with PretrainedFamilies.ExpertSpec) do.");
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
                    throw new NotRegisteredException(nameof(RopeScalings), type, RopeScalings.Names,
                        $"RoPE scaling '{type}' is not registered ({string.Join(", ", RopeScalings.Names)}); add it with RopeScalings.Register, "
                        + "reference Idrak.Text.Plugins (a family's own scaling) and call TextPlugins.RegisterAll(), or remove rope_scaling to use the model within its original context.");
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

    /// <summary>
    /// Idrak weight names → the common checkpoint names (model.embed_tokens, model.layers.i.self_attn.q_proj, …, model.norm,
    /// lm_head); null for a weight they do not name.
    /// </summary>
    public static string? CommonTensorName(string name)
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

    /// <summary>The former name of <see cref="CommonTensorName"/>, kept for one release.</summary>
    [Obsolete("Renamed CommonTensorName: the checkpoint names are the common ones, whichever family uses them.")]
    public static string? LlamaTensorName(string name) => CommonTensorName(name);
}

/// <summary>
/// Reads a model's weights from a safetensors checkpoint through an architecture's name mapping; bfloat16 weights of
/// bfloat16 tensors as their stored bits (<see cref="ReadBFloat16"/>). Keeps the time spent reading and merging, for
/// <see cref="PretrainedModel.LoadTimes"/>.
/// </summary>
internal sealed class CheckpointWeights(ITensorStore reader, IPretrainedArchitecture architecture) : IWeightSource
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

    /// <summary>
    /// A deferred build (plan 23, C8b): each weight is checked against the checkpoint's header and answered with a placeholder
    /// (<see cref="DeferredTensors.Placeholder"/>), nothing read; the loader gives each tensor its source afterwards.
    /// </summary>
    public bool Deferred { get; init; }

    public float[]? Read(string name, IReadOnlyList<int> shape)
    {
        if (Stored(name, shape) is not { } stored)
        {
            return null;
        }

        if (Deferred)
        {
            return DeferredTensors.Placeholder(shape.Aggregate(1, (a, b) => checked(a * b)));
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

    // A tensor read as stored (no transposition, no adapter adding to it): its rows are the store's rows.
    private bool AsStored(string name, string stored) =>
        !Deferred && !architecture.Transposed(name) && (Adapter is null || !stored.EndsWith(".weight", StringComparison.Ordinal) || !Adapter.Updates(stored[..^".weight".Length]));

    public bool ReadRows(string name, IReadOnlyList<int> shape, int first, int count, Span<float> destination)
    {
        if (Stored(name, shape) is not { } stored || !AsStored(name, stored))
        {
            return false;
        }

        long start = Stopwatch.GetTimestamp();
        bool read = reader.ReadRows(stored, first, count, destination);
        _readTicks += Stopwatch.GetTimestamp() - start;
        return read;
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
        if (!merged && shape.Count == 2 && reader is IBFloat16TensorStore bits16 && reader.FormatOf(stored) == WeightFormat.BFloat16)
        {
            int rows = transposed ? shape[1] : shape[0], columns = transposed ? shape[0] : shape[1];
            long begun = Stopwatch.GetTimestamp(), reading = 0;
            var weight = BFloat16Weight.FromBits(rows, columns, device, (Span<ushort> bits, int stride) =>
            {
                long start = Stopwatch.GetTimestamp();
                bool copied = bits16.ReadBFloat16(stored, storedTransposed != transposed, bits, stride);
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

        // A float32 or float16 matrix as stored (an embedding table, or its transpose for a tied head): rounded a block of
        // rows at a time, never held as float32 whole or transposed.
        if (shape.Count == 2 && AsStored(name, stored)
            && BFloat16Weight.FromRows(shape[0], shape[1], device, (first, count, destination) => ReadRows(name, shape, first, count, destination), transposed) is { } rounded)
        {
            return rounded;
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
