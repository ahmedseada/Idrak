using System.Text.Json.Nodes;
using Idrak.Layers;

namespace Idrak.LanguageModels;

/// <summary>
/// How to read one family of pretrained models: its configuration (a Hugging Face <c>config.json</c>) becomes a
/// <see cref="DecoderSpec"/>, and each of Idrak's weight names (see <see cref="DecoderSpec"/>) is found in the
/// checkpoint. Register new families with <see cref="PretrainedArchitectures.Register"/>.
/// </summary>
public sealed class PretrainedArchitecture
{
    /// <summary>The model described by a configuration; append anything approximated to the notes.</summary>
    public required Func<JsonObject, List<string>, DecoderSpec> Spec { get; init; }

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

/// <summary>
/// The model families <see cref="PretrainedModel.Load"/> knows, by the architecture name in <c>config.json</c>
/// ("architectures": [...]). Llama, Mistral, Qwen2, Qwen3 and Gemma are registered; add others with
/// <see cref="Register"/>, usually with <see cref="LlamaStyle"/> when they share the Llama naming.
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
    };

    /// <summary>Registers (or replaces) how to read the architecture <paramref name="name"/>.</summary>
    public static void Register(string name, PretrainedArchitecture architecture)
    {
        lock (Registry)
        {
            Registry[name] = architecture;
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
            scaling = type switch
            {
                "default" => null,
                "linear" => new RopeScaling("linear", (float)s["factor"]!),
                "llama3" => new RopeScaling("llama3", (float)s["factor"]!, (float?)s["low_freq_factor"] ?? 1f, (float?)s["high_freq_factor"] ?? 4f,
                    (int?)s["original_max_position_embeddings"] ?? 8192),
                _ => throw new NotSupportedException($"RoPE scaling '{type}' is not supported (linear, llama3); remove rope_scaling to use the model within its original context."),
            };
        }

        if (c["sliding_window"] is { } window && (int?)window is int w && w < maxPositions && ((bool?)c["use_sliding_window"] ?? true))
        {
            notes.Add($"Sliding-window attention ({w} positions) is not implemented: outputs are identical up to {w} positions, attention is full beyond.");
        }

        string activation = (string?)c["hidden_act"] ?? (string?)c["hidden_activation"] ?? "silu";
        if (activation == "gelu")
        {
            notes.Add("Exact GELU is computed with the tanh approximation (differences below 0.001).");
        }

        bool attentionBias = (bool?)c["attention_bias"] ?? false;
        return new DecoderSpec
        {
            Vocabulary = Int("vocab_size"), Dim = dim, Layers = Int("num_hidden_layers"), Heads = heads,
            KvHeads = (int?)c["num_key_value_heads"] ?? heads, HeadDim = headDim, FfDim = Int("intermediate_size"), MaxPositions = maxPositions,
            NormEpsilon = (float?)c["rms_norm_eps"] ?? 1e-6f,
            Activation = activation switch
            {
                "silu" or "swish" => FeedForwardActivation.Silu,
                "gelu" or "gelu_new" or "gelu_pytorch_tanh" => FeedForwardActivation.Gelu,
                "relu" => FeedForwardActivation.Relu,
                _ => throw new NotSupportedException($"Activation '{activation}' is not supported."),
            },
            Rope = new RopeSettings(theta, rotary, Interleaved: false, scaling),
            QkvBias = attentionBias, OutputBias = attentionBias, FeedForwardBias = (bool?)c["mlp_bias"] ?? false,
            TieEmbeddings = (bool?)c["tie_word_embeddings"] ?? false,
        };
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
        var values = reader.Read(stored);
        bool transposed = architecture.Transposed(name);
        int[] expected = transposed ? [.. shape.Reverse()] : [.. shape];
        if (!storedShape.SequenceEqual(expected))
        {
            throw new InvalidDataException($"'{stored}' is [{string.Join(", ", storedShape)}]; the model expects [{string.Join(", ", expected)}] for {name}.");
        }

        if (transposed)
        {
            values = Idrak.HostParallel.Transpose(values, storedShape[0], storedShape[1]);
        }

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
