// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Models.Abstractions;

/// <summary>
/// How llama.cpp stores one model family (general.architecture in the file): the Hugging Face architecture it corresponds
/// to, and whether it interleaved the query and key rows of each head for its rotary layout. The file's settings are read
/// from the usual keys under the family's name ({name}.embedding_length, {name}.block_count, …), with the family's own
/// keys added by <see cref="Configure"/>, and its tensors by llama.cpp's names (token_embd, blk.N.attn_q, …) through
/// <see cref="TensorName"/> (<see cref="CommonTensorName"/> unless set), less what llama.cpp added to their values
/// (<see cref="StoredOffset"/>). Register new families with <see cref="GgufArchitectures.Register"/>.
/// </summary>
public sealed class GgufArchitecture
{
    /// <summary>The Hugging Face architecture (config.json's "architectures"), registered in <see cref="PretrainedArchitectures"/>.</summary>
    public required string HuggingFace { get; init; }

    /// <summary>Whether llama.cpp interleaved each head's query and key rows (Llama does; they are put back on load).</summary>
    public bool InterleavedQueryKeys { get; init; }

    /// <summary>
    /// The Hugging Face architecture of this family's files with experts ({name}.expert_count above 0), when it is not
    /// <see cref="HuggingFace"/> (llama files with experts are Mixtral); null: <see cref="HuggingFace"/>.
    /// </summary>
    public string? WithExperts { get; init; }

    /// <summary>
    /// Whether the family renormalizes the chosen experts' weights when the file does not say ({name}.expert_weights_norm),
    /// as llama.cpp decides per family; null leaves it to the Hugging Face family's default.
    /// </summary>
    public bool? NormalizeTopK { get; init; }

    /// <summary>
    /// The config.json key a file's expert count ({name}.expert_count) is written under, as the Hugging Face family with
    /// experts names it: "num_experts" unless set (a family may call it "num_local_experts").
    /// </summary>
    public string ExpertCountKey { get; init; } = "num_experts";

    /// <summary>
    /// Adds the family's own keys to the config.json written for a file (the file, the config with the common keys, the
    /// notes of what was assumed), for what the family stores its own way; null (the default) for none. It runs before
    /// the file's generic RoPE scaling keys ({name}.rope.scaling.*) are read, which are left out when it sets
    /// "rope_scaling" itself.
    /// </summary>
    public Action<GgufFile, System.Text.Json.Nodes.JsonObject, List<string>>? Configure { get; init; }

    /// <summary>
    /// The Hugging Face name of a tensor of this family's files, from its llama.cpp name ("blk.3.ffn_norm.weight"), or null
    /// for a tensor not read by name (experts, read through the family's own names, or one not read at all). Null (the
    /// default): <see cref="CommonTensorName"/>. A family whose names differ maps its own and returns
    /// <see cref="CommonTensorName"/> for the rest.
    /// </summary>
    public Func<string, string?>? TensorName { get; init; }

    /// <summary>
    /// What llama.cpp's converter added to every value of a tensor when it wrote this family's files, by the tensor's
    /// Hugging Face name (0 for nothing): subtracted as the tensor is read, so the weights are the checkpoint's (a family
    /// whose norm gains are stored as a difference from 1 gets them back so, while the file holds 1 + gain). Null (the
    /// default): nothing added to any tensor.
    /// </summary>
    public Func<string, float>? StoredOffset { get; init; }

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

/// <summary>
/// The model families GGUF files are read as, by GGUF architecture name. The library registers none: the families are
/// plug-ins' (Idrak.Text.Plugins registers llama, qwen2, qwen3, qwen2moe, qwen3moe, gemma, gemma2 and gemma3); add others
/// with <see cref="Register"/>.
/// </summary>
public static class GgufArchitectures
{
    private static readonly SlotTable<string, GgufArchitecture> Registry = new(nameof(GgufArchitectures), comparer: StringComparer.Ordinal,
        unguarded: "an architecture is a description read once, not a call");

    /// <summary>Registers how to read the GGUF architecture <paramref name="name"/>, until <see cref="Unregister"/>.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(string name, GgufArchitecture architecture)
    {
        ArgumentNullException.ThrowIfNull(architecture);
        Registry.Register(name, architecture, System.Reflection.Assembly.GetCallingAssembly(), architecture.HuggingFace);
    }

    /// <summary>Removes the architecture <paramref name="name"/> an app or plug-in registered; false when none did.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered architecture names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The architecture registered as <paramref name="name"/>.</summary>
    public static GgufArchitecture Get(string name) => Find(name)
        ?? throw new NotSupportedException($"No GGUF architecture '{name}' is registered ({string.Join(", ", Names)}); add it with GgufArchitectures.Register, or reference Idrak.Text.Plugins and call TextPlugins.RegisterAll().");

    /// <summary>The architecture registered as <paramref name="name"/>, or null when there is none.</summary>
    public static GgufArchitecture? Find(string name) => Registry.Find(name);

    /// <summary>The library's architecture <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static GgufArchitecture? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the architecture <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);
}
