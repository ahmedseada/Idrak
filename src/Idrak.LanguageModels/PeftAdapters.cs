// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.LanguageModels;

/// <summary>
/// A PEFT adapter folder's adapter_config.json, checked: only LoRA adapters (peft_type LORA) are read, and options that
/// change what the tensors mean and that Idrak does not apply are refused with an error naming them, instead of the
/// adapter being loaded as if it were plain LoRA.
/// </summary>
internal sealed record PeftAdapterConfig(int Rank, float Alpha, float Scale, bool UseDora)
{
    public static PeftAdapterConfig Read(string folder)
    {
        string path = Path.Combine(folder, "adapter_config.json");
        var config = JsonNode.Parse(File.ReadAllText(path)) as JsonObject ?? throw new InvalidDataException($"{path} is not a JSON object.");
        if ((string?)config["peft_type"] is { } type && !type.Equals("LORA", StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException($"{path}: peft_type \"{type}\" is not supported; Idrak loads LoRA adapters (peft_type LORA, with use_dora for DoRA).");
        }

        int rank = (int?)config["r"] ?? throw new InvalidDataException($"{path} has no r.");
        if (rank <= 0)
        {
            throw new InvalidDataException($"{path}: r is {rank}.");
        }

        var unsupported = new List<string>();
        if (config["rank_pattern"] is JsonObject { Count: > 0 })
        {
            unsupported.Add("rank_pattern (ranks per module)");
        }

        if (config["alpha_pattern"] is JsonObject { Count: > 0 })
        {
            unsupported.Add("alpha_pattern (alphas per module)");
        }

        if ((string?)config["bias"] is { } bias && bias != "none")
        {
            unsupported.Add($"bias \"{bias}\" (trained biases)");
        }

        if ((bool?)config["lora_bias"] == true)
        {
            unsupported.Add("lora_bias");
        }

        if (config["modules_to_save"] is JsonArray { Count: > 0 })
        {
            unsupported.Add("modules_to_save (fully trained modules)");
        }

        if (config["layer_replication"] is JsonArray { Count: > 0 })
        {
            unsupported.Add("layer_replication");
        }

        if (config["trainable_token_indices"] is JsonNode)
        {
            unsupported.Add("trainable_token_indices");
        }

        if ((bool?)config["use_qalora"] == true)
        {
            unsupported.Add("use_qalora");
        }

        if (unsupported.Count > 0)
        {
            throw new NotSupportedException($"{path}: {string.Join(", ", unsupported)} {(unsupported.Count == 1 ? "is" : "are")} not supported; "
                                            + "Idrak loads LoRA and DoRA adapters with one rank and alpha for every module.");
        }

        float alpha = (float?)config["lora_alpha"] ?? rank;
        bool rslora = (bool?)config["use_rslora"] == true;
        return new PeftAdapterConfig(rank, alpha, rslora ? alpha / MathF.Sqrt(rank) : alpha / rank, (bool?)config["use_dora"] == true);
    }

    /// <summary>
    /// The adapter tensors of the checkpoint module <paramref name="module"/> (its weight name without ".weight"): lora_A,
    /// lora_B and, for DoRA, the magnitude vector, under the names peft writes (with or without the adapter name
    /// "default"); null when the file has none for it.
    /// </summary>
    public (string A, string B, string? Magnitude)? Names(SafeTensorsReader reader, string module)
    {
        string prefix = "base_model.model." + module;
        string? Find(params string[] names) => names.FirstOrDefault(reader.Contains);
        if (Find($"{prefix}.lora_A.weight", $"{prefix}.lora_A.default.weight") is not { } a)
        {
            return null;
        }

        string b = Find($"{prefix}.lora_B.weight", $"{prefix}.lora_B.default.weight")
            ?? throw new InvalidDataException($"The adapter has {a} but no lora_B for {module}.");
        string? magnitude = Find($"{prefix}.lora_magnitude_vector", $"{prefix}.lora_magnitude_vector.weight", $"{prefix}.lora_magnitude_vector.default",
            $"{prefix}.lora_magnitude_vector.default.weight");
        if (magnitude is not null && !UseDora)
        {
            throw new InvalidDataException($"The adapter has a DoRA magnitude vector for {module} but its adapter_config.json does not set use_dora.");
        }

        if (magnitude is null && UseDora)
        {
            throw new InvalidDataException($"The adapter sets use_dora but has no lora_magnitude_vector for {module}.");
        }

        return (a, b, magnitude);
    }

    /// <summary>Throws when the file holds tensors that no layer took (adapters for modules the model has not, or does not adapt).</summary>
    public static void CheckAllUsed(SafeTensorsReader reader, ICollection<string> used, string folder)
    {
        var unused = reader.Tensors.Keys.Where(k => !used.Contains(k)).Order(StringComparer.Ordinal).ToList();
        if (unused.Count > 0)
        {
            throw new InvalidDataException($"The adapter in {folder} has {unused.Count} tensors that match no adaptable layer of the model "
                                           + $"(for example {string.Join(", ", unused.Take(3))}): an adapter for another architecture, or for layers "
                                           + "Idrak does not adapt (embeddings, tied output heads).");
        }
    }
}

/// <summary>
/// Folds a PEFT LoRA adapter (adapter_config.json, adapter_model.safetensors) into weights while they are read, so a
/// fine-tuned model is quantized with its update included and runs as fast as the base model.
/// </summary>
internal sealed class AdapterMerge : IDisposable
{
    private readonly SafeTensorsReader _reader;
    private readonly PeftAdapterConfig _config;
    private readonly HashSet<string> _used = [];
    private readonly string _folder;

    public AdapterMerge(string folder)
    {
        _config = PeftAdapterConfig.Read(folder);
        _folder = folder;
        _reader = SafeTensorsReader.Open(Path.Combine(folder, "adapter_model.safetensors"));
    }

    /// <summary>How many weights received an update.</summary>
    public int Merged { get; private set; }

    /// <summary>Throws when some of the adapter's tensors matched no weight that was read.</summary>
    public void CheckAllUsed() => PeftAdapterConfig.CheckAllUsed(_reader, _used, _folder);

    /// <summary>
    /// Adds scale · (B·A)ᵀ to <paramref name="weight"/> ([inputs, outputs], Idrak's layout) when the adapter has
    /// lora_A [r, inputs] and lora_B [outputs, r] for the checkpoint module <paramref name="module"/>; for DoRA, then scales
    /// each output column to its magnitude: W' = m ⊙ (W + s·A·B) / ‖W + s·A·B‖ (see <see cref="DoraAdapter"/>).
    /// </summary>
    public void AddTo(string module, float[] weight, int inputs, int outputs)
    {
        if (_config.Names(_reader, module) is not var (a, b, magnitude))
        {
            return;
        }

        _used.Add(a);
        _used.Add(b);
        if (magnitude is not null)
        {
            _used.Add(magnitude);
        }

        var down = _reader.Read(a);                                           // [r, inputs]
        int rank = down.Length / inputs;
        var up = Idrak.Abstraction.Devices.HostParallel.Transpose(_reader.Read(b), outputs, rank); // [r, outputs]
        float scale = _config.Scale;
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
        if (magnitude is not null)
        {
            var m = _reader.Read(magnitude);
            var norms = new double[outputs];
            for (int i = 0; i < inputs; i++)
            {
                for (int o = 0; o < outputs; o++)
                {
                    double v = weight[i * outputs + o];
                    norms[o] += v * v;
                }
            }

            var factors = new float[outputs];
            for (int o = 0; o < outputs; o++)
            {
                factors[o] = (float)(m[o] / Math.Sqrt(norms[o]));
            }

            Parallel.For(0, inputs, i =>
            {
                var row = weight.AsSpan(i * outputs, outputs);
                for (int o = 0; o < outputs; o++)
                {
                    row[o] *= factors[o];
                }
            });
        }

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
