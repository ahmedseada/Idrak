// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models.Abstractions;

namespace Idrak.Text.Plugins.Llama;

/// <summary>
/// The Llama family's names: Hugging Face's <c>LlamaForCausalLM</c> (config.json's "architectures", read by
/// <see cref="LlamaArchitectures.Llama"/>) and llama.cpp's "llama" (<see cref="LlamaArchitectures.Gguf"/>; Mistral's GGUF
/// files too). Its GGUF pre-tokenizers' split pattern (<see cref="LlamaArchitectures.SplitPattern"/>), Llama 3's RoPE
/// scaling (<see cref="LlamaRopeScaling"/>) and Llama 3's chat template for GGUF files without one
/// (<see cref="LlamaArchitectures.ChatTemplateFallback"/>) are its parts. Registering them is not on the new design yet.
/// </summary>
public static class LlamaPlugin
{
    /// <summary>Llama: "LlamaForCausalLM".</summary>
    public const string LlamaForCausalLM = "LlamaForCausalLM";

    /// <summary>llama.cpp's architecture name: "llama".</summary>
    public const string GgufName = "llama";
}
