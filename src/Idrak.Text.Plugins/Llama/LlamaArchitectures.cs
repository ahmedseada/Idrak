// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models;
using Idrak.Models.Abstractions;

namespace Idrak.Text.Plugins.Llama;

/// <summary>
/// How the Llama family is read: its Hugging Face architecture (<see cref="Llama()"/>, the common configuration keys and
/// checkpoint names), its GGUF mapping (<see cref="Gguf()"/>: llama.cpp's "llama", which Mistral's files use too, with
/// its experts read as Mixtral, and Llama 3's RoPE scaling stored as rope_freqs.weight), the split pattern of its
/// GGUF pre-tokenizers (<see cref="SplitPattern"/>, llama.cpp's llama-vocab.cpp), and Llama 3's chat template for a GGUF
/// file without one (<see cref="ChatTemplateFallback"/>).
/// </summary>
public static class LlamaArchitectures
{
    /// <summary>llama.cpp's split pattern for the llama3 family (llama-vocab.cpp), the names in <see cref="LlamaPlugin.PreTokenizers"/>.</summary>
    public const string SplitPattern = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    /// <summary>
    /// Llama 3's chat turns (<c>&lt;|start_header_id|&gt;role&lt;|end_header_id|&gt;</c> … <c>&lt;|eot_id|&gt;</c>, after the
    /// begin token), the template a GGUF file without one gets when its vocabulary holds <c>&lt;|start_header_id|&gt;</c>
    /// (<see cref="ChatTemplateFallback"/>, registered in <see cref="GgufChatTemplates"/> as "llama3").
    /// </summary>
    public const string ChatTemplateText = "{{- bos_token }}{%- for message in messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}{%- endfor %}"
                                           + "{%- if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{%- endif %}";

    /// <summary>
    /// The fallback chat template of a GGUF file without one (<see cref="GgufChatTemplates"/>): <see cref="ChatTemplateText"/>,
    /// for a file whose vocabulary holds <c>&lt;|start_header_id|&gt;</c>.
    /// </summary>
    public static GgufChatTemplate ChatTemplateFallback() => new() { Matches = (_, tokens) => tokens.Contains("<|start_header_id|>"), Template = ChatTemplateText };

    /// <summary>LlamaForCausalLM: the common keys and names as they are, written back as config.json with model_type "llama".</summary>
    public static IPretrainedArchitecture Llama() => PretrainedFamilies.CommonArchitecture((config, spec, _) => spec, "llama");

    /// <summary>
    /// llama.cpp's "llama": LlamaForCausalLM (Mistral's files too), MixtralForCausalLM when the file has experts (counted
    /// as num_local_experts), each head's query and key rows interleaved, and Llama 3's RoPE scaling recovered from the
    /// per-frequency divisors it stores (rope_freqs.weight).
    /// </summary>
    public static IGgufArchitecture Gguf() => new LlamaGguf
    {
        HuggingFace = LlamaPlugin.LlamaForCausalLM,
        InterleavedQueryKeys = true,
        WithExperts = "MixtralForCausalLM",
        ExpertCountKey = "num_local_experts",
    };

    // llama.cpp's "llama": Llama 3's RoPE scaling recovered from rope_freqs.weight when the file has it.
    private sealed class LlamaGguf : GgufArchitecture
    {
        public override void Configure(GgufFile file, JsonObject config, List<string> notes)
        {
            if (file.Tensors.ContainsKey("rope_freqs.weight"))
            {
                config["rope_scaling"] = Llama3Scaling(file, config, notes);
            }
        }
    }

    // Llama 3.1+ stores its RoPE scaling as per-frequency divisors (rope_freqs.weight): recover the parameters and check them.
    private static JsonObject Llama3Scaling(GgufFile file, JsonObject config, List<string> notes)
    {
        var divisors = file.Read("rope_freqs.weight");
        double theta = (double)config["rope_theta"]!;
        int headDim = (int)config["head_dim"]!;
        double factor = divisors.Max();
        foreach (int original in new[] { 8192, 4096, 16384, 32768 })
        {
            double worst = 0;
            for (int i = 0; i < divisors.Length; i++)
            {
                double wavelength = 2 * Math.PI * Math.Pow(theta, 2.0 * i / headDim);
                double low = original / 1.0, high = original / 4.0;
                double expected = wavelength < high ? 1 : wavelength > low ? factor
                    : 1 / ((1 - (original / wavelength - 1) / 3) / factor + (original / wavelength - 1) / 3);
                worst = Math.Max(worst, Math.Abs(expected - divisors[i]) / expected);
            }

            if (worst < 1e-3)
            {
                return new JsonObject { ["rope_type"] = "llama3", ["factor"] = factor, ["low_freq_factor"] = 1.0, ["high_freq_factor"] = 4.0,
                    ["original_max_position_embeddings"] = original };
            }
        }

        notes.Add("rope_freqs.weight does not match Llama 3's scaling with the usual parameters; they were assumed (factor from the file, original 8192).");
        return new JsonObject { ["rope_type"] = "llama3", ["factor"] = factor, ["low_freq_factor"] = 1.0, ["high_freq_factor"] = 4.0,
            ["original_max_position_embeddings"] = 8192 };
    }
}
