// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models;
using Idrak.Models.Abstractions;

namespace Idrak.Text.Plugins.Mistral;

/// <summary>
/// How the Mistral family is read: Mistral (<see cref="Mistral()"/>, the common configuration keys and checkpoint names),
/// Mixtral (<see cref="Mixtral()"/>, its experts under block_sparse_moe, <see cref="MixtralTensorName"/>), and the split
/// pattern of the tekken GGUF pre-tokenizer (<see cref="TekkenPattern"/>, llama.cpp's llama-vocab.cpp). Their GGUF files
/// are llama.cpp's "llama" (<see cref="Llama.LlamaPlugin"/>'s mapping, Mixtral when the file has experts).
/// </summary>
public static class MistralArchitectures
{
    /// <summary>llama.cpp's split pattern for the tekken pre-tokenizer (llama-vocab.cpp).</summary>
    public const string TekkenPattern = @"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    /// <summary>MistralForCausalLM: the common keys and names as they are, written back as config.json with model_type "mistral".</summary>
    public static IPretrainedArchitecture Mistral() => PretrainedFamilies.CommonArchitecture((config, spec, _) => spec, "mistral");

    /// <summary>MixtralForCausalLM: the experts' keys (<see cref="PretrainedFamilies.ExpertSpec"/>, the chosen experts' weights renormalized unless norm_topk_prob says otherwise) and <see cref="MixtralTensorName"/>.</summary>
    public static IPretrainedArchitecture Mixtral() => PretrainedFamilies.Architecture((config, notes) => PretrainedFamilies.ExpertSpec(config, notes, normalizeTopK: true), MixtralTensorName);

    /// <summary>
    /// Idrak weight names → Mixtral checkpoint names: the common names (<see cref="PretrainedFamilies.CommonTensorName"/>),
    /// with each layer's experts under block_sparse_moe (the router "gate", and experts.j.w1 / w3 / w2 for each expert's
    /// gate, up and down projections).
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

        return PretrainedFamilies.CommonTensorName(name);
    }
}
