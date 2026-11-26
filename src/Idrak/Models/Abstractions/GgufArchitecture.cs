// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Models.Abstractions;

/// <summary>
/// How llama.cpp stores one model family (general.architecture in the file): the Hugging Face architecture it corresponds
/// to, and whether it interleaved the query and key rows of each head for its rotary layout. The file's settings are read
/// from the usual keys under the family's name ({name}.embedding_length, {name}.block_count, …), with the family's own
/// keys added by <see cref="Configure"/>, and its tensors by llama.cpp's names (token_embd, blk.N.attn_q, …) through
/// <see cref="TensorName"/>, less what llama.cpp added to their values (<see cref="StoredOffset"/>).
/// <see cref="GgufArchitecture"/> is a base with the common names.
/// </summary>
public interface IGgufArchitecture
{
    /// <summary>The Hugging Face architecture (config.json's "architectures").</summary>
    string HuggingFace { get; }

    /// <summary>Whether llama.cpp interleaved each head's query and key rows (Llama does; they are put back on load).</summary>
    bool InterleavedQueryKeys { get; }

    /// <summary>
    /// The Hugging Face architecture of this family's files with experts ({name}.expert_count above 0), when it is not
    /// <see cref="HuggingFace"/> (llama files with experts are Mixtral); null: <see cref="HuggingFace"/>.
    /// </summary>
    string? WithExperts { get; }

    /// <summary>
    /// Whether the family renormalizes the chosen experts' weights when the file does not say ({name}.expert_weights_norm),
    /// as llama.cpp decides per family; null leaves it to the Hugging Face family's default.
    /// </summary>
    bool? NormalizeTopK { get; }

    /// <summary>The config.json key a file's expert count ({name}.expert_count) is written under, as the Hugging Face family with experts names it.</summary>
    string ExpertCountKey { get; }

    /// <summary>
    /// Adds the family's own keys to the config.json written for a file (the file, the config with the common keys, the
    /// notes of what was assumed), for what the family stores its own way. It runs before the file's generic RoPE scaling
    /// keys ({name}.rope.scaling.*) are read, which are left out when it sets "rope_scaling" itself.
    /// </summary>
    void Configure(GgufFile file, System.Text.Json.Nodes.JsonObject config, List<string> notes);

    /// <summary>
    /// The Hugging Face name of a tensor of this family's files, from its llama.cpp name ("blk.3.ffn_norm.weight"), or null
    /// for a tensor not read by name (experts, read through the family's own names, or one not read at all).
    /// </summary>
    string? TensorName(string name);

    /// <summary>
    /// What llama.cpp's converter added to every value of a tensor when it wrote this family's files, by the tensor's
    /// Hugging Face name (0 for nothing): subtracted as the tensor is read, so the weights are the checkpoint's (a family
    /// whose norm gains are stored as a difference from 1 gets them back so, while the file holds 1 + gain).
    /// </summary>
    float StoredOffset(string name);

    /// <summary>
    /// The names a GGUF file calls this family by, as the family states them (llama.cpp's <c>general.architecture</c>:
    /// "qwen3"): what a file's own name is compared with where it comes in (<see cref="IFamilyProvider.GgufArchitectures"/>).
    /// </summary>
    IReadOnlyList<string> Names => [];
}

/// <summary>
/// A family's GGUF mapping (<see cref="IGgufArchitecture"/>) with the common tensor names (<see cref="CommonTensorName"/>),
/// nothing added to any tensor and no keys of its own, unless a family deriving from it says otherwise.
/// </summary>
public class GgufArchitecture : IGgufArchitecture
{
    /// <inheritdoc />
    public required string HuggingFace { get; init; }

    /// <inheritdoc />
    public IReadOnlyList<string> Names { get; init; } = [];

    /// <inheritdoc />
    public bool InterleavedQueryKeys { get; init; }

    /// <inheritdoc />
    public string? WithExperts { get; init; }

    /// <inheritdoc />
    public bool? NormalizeTopK { get; init; }

    /// <summary>The config.json key a file's expert count is written under: "num_experts" unless set (a family may call it "num_local_experts").</summary>
    public string ExpertCountKey { get; init; } = "num_experts";

    /// <summary>Adds the family's own keys to a file's config.json; none by default.</summary>
    public virtual void Configure(GgufFile file, System.Text.Json.Nodes.JsonObject config, List<string> notes)
    {
    }

    /// <summary>The Hugging Face name of a tensor of this family's files: <see cref="CommonTensorName"/> by default.</summary>
    public virtual string? TensorName(string name) => CommonTensorName(name);

    /// <summary>What llama.cpp added to a tensor's values: nothing by default.</summary>
    public virtual float StoredOffset(string name) => 0f;

    /// <summary>
    /// llama.cpp's names of the common decoder's tensors → the Hugging Face ones: token_embd, output_norm and output →
    /// model.embed_tokens, model.norm and lm_head; blk.N.attn_norm, ffn_norm, attn_q, attn_k, attn_v, attn_output,
    /// attn_q_norm, attn_k_norm, ffn_gate, ffn_up, ffn_down → model.layers.N.input_layernorm, post_attention_layernorm,
    /// self_attn.q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, mlp.gate_proj, up_proj, down_proj (each .weight or .bias as
    /// stored); null for any other name.
    /// </summary>
    public static string? CommonTensorName(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        switch (name)
        {
            case "token_embd.weight": return "model.embed_tokens.weight";
            case "output_norm.weight": return "model.norm.weight";
            case "output.weight": return "lm_head.weight";
        }

        var parts = name.Split('.');
        if (parts is not ["blk", var layer, var part, var kind])
        {
            return null;
        }

        string? module = part switch
        {
            "attn_norm" => "input_layernorm",
            "ffn_norm" => "post_attention_layernorm",
            "attn_q" => "self_attn.q_proj",
            "attn_k" => "self_attn.k_proj",
            "attn_v" => "self_attn.v_proj",
            "attn_output" => "self_attn.o_proj",
            "attn_q_norm" => "self_attn.q_norm",
            "attn_k_norm" => "self_attn.k_norm",
            "ffn_gate" => "mlp.gate_proj",
            "ffn_up" => "mlp.up_proj",
            "ffn_down" => "mlp.down_proj",
            _ => null,
        };
        return module is null ? null : $"model.layers.{layer}.{module}.{kind}";
    }
}
