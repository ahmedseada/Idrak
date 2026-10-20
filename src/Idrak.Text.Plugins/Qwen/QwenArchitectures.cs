// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models;
using Idrak.Models.Abstractions;

namespace Idrak.Text.Plugins.Qwen;

/// <summary>
/// How the Qwen family is read: Qwen2 (<see cref="Qwen2()"/>, query, key and value biases) and Qwen3
/// (<see cref="Qwen3()"/>, q/k norm) on the common configuration keys and checkpoint names, their mixtures of experts
/// (<see cref="Qwen2Moe()"/>, <see cref="Qwen3Moe()"/>, <see cref="MoeTensorName"/>), their GGUF mappings
/// (<see cref="Gguf"/>) and the split pattern of their GGUF pre-tokenizers (<see cref="SplitPattern"/>, llama.cpp's
/// llama-vocab.cpp).
/// </summary>
public static class QwenArchitectures
{
    /// <summary>llama.cpp's split pattern for the qwen2 family (llama-vocab.cpp), the names in <see cref="QwenPlugin.PreTokenizers"/>.</summary>
    public const string SplitPattern = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    /// <summary>Qwen2ForCausalLM: the common keys with query, key and value biases, written back as config.json with model_type "qwen2".</summary>
    public static PretrainedArchitecture Qwen2() => PretrainedFamilies.CommonArchitecture((config, spec, _) => spec with { QkvBias = true }, "qwen2");

    /// <summary>Qwen3ForCausalLM: the common keys with q/k norm, written back as config.json with model_type "qwen3".</summary>
    public static PretrainedArchitecture Qwen3() => PretrainedFamilies.CommonArchitecture((config, spec, _) => spec with { QkNorm = true }, "qwen3");

    /// <summary>Qwen2MoeForCausalLM: the experts' keys (<see cref="PretrainedFamilies.ExpertSpec"/>, not renormalized unless norm_topk_prob says so) with query, key and value biases.</summary>
    public static PretrainedArchitecture Qwen2Moe() => new() { Spec = (config, notes) => PretrainedFamilies.ExpertSpec(config, notes, normalizeTopK: false) with { QkvBias = true }, TensorName = MoeTensorName };

    /// <summary>Qwen3MoeForCausalLM: the experts' keys (<see cref="PretrainedFamilies.ExpertSpec"/>, not renormalized unless norm_topk_prob says so) with q/k norm.</summary>
    public static PretrainedArchitecture Qwen3Moe() => new() { Spec = (config, notes) => PretrainedFamilies.ExpertSpec(config, notes, normalizeTopK: false) with { QkNorm = true }, TensorName = MoeTensorName };

    /// <summary>
    /// llama.cpp's architecture names and their mappings: "qwen2", "qwen3" (no interleaved query and key rows), "qwen2moe"
    /// (experts' weights not renormalized unless the file says so) and "qwen3moe" (renormalized unless the file says not).
    /// </summary>
    public static IReadOnlyList<(string Name, GgufArchitecture Architecture)> Gguf() =>
    [
        ("qwen2", new() { HuggingFace = QwenPlugin.Qwen2ForCausalLM, InterleavedQueryKeys = false }),
        ("qwen3", new() { HuggingFace = QwenPlugin.Qwen3ForCausalLM, InterleavedQueryKeys = false }),
        ("qwen2moe", new() { HuggingFace = QwenPlugin.Qwen2MoeForCausalLM, InterleavedQueryKeys = false, NormalizeTopK = false }),
        ("qwen3moe", new() { HuggingFace = QwenPlugin.Qwen3MoeForCausalLM, InterleavedQueryKeys = false, NormalizeTopK = true }),
    ];

    /// <summary>
    /// Idrak weight names → Qwen2-MoE / Qwen3-MoE checkpoint names: the common names
    /// (<see cref="PretrainedFamilies.CommonTensorName"/>), with the router as mlp.gate, the experts as
    /// mlp.experts.j.gate_proj / up_proj / down_proj, the shared expert as mlp.shared_expert.*_proj and its gate as
    /// mlp.shared_expert_gate.
    /// </summary>
    public static string? MoeTensorName(string name)
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

        return PretrainedFamilies.CommonTensorName(name);
    }
}
