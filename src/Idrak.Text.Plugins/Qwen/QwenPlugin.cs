// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models.Abstractions;

namespace Idrak.Text.Plugins.Qwen;

/// <summary>
/// The Qwen family's names and parts: Hugging Face's <c>Qwen2ForCausalLM</c>, <c>Qwen3ForCausalLM</c> and their mixtures
/// of experts <c>Qwen2MoeForCausalLM</c> and <c>Qwen3MoeForCausalLM</c> (config.json's "architectures", read by
/// <see cref="QwenArchitectures"/>), llama.cpp's "qwen2", "qwen3", "qwen2moe" and "qwen3moe"
/// (<see cref="QwenArchitectures.Gguf"/>), its GGUF pre-tokenizers' split pattern (<see cref="QwenArchitectures.SplitPattern"/>),
/// the XML tool-call format "qwen3-coder", and Qwen's fill-in-the-middle format "qwen" (<see cref="FillInMiddle"/>).
/// Registering them is not on the new design yet. The vision-language Qwen models are Idrak.Vision.Plugins'.
/// </summary>
public static class QwenPlugin
{
    /// <summary>Qwen2: "Qwen2ForCausalLM".</summary>
    public const string Qwen2ForCausalLM = "Qwen2ForCausalLM";

    /// <summary>Qwen3: "Qwen3ForCausalLM".</summary>
    public const string Qwen3ForCausalLM = "Qwen3ForCausalLM";

    /// <summary>Qwen2's mixture of experts: "Qwen2MoeForCausalLM".</summary>
    public const string Qwen2MoeForCausalLM = "Qwen2MoeForCausalLM";

    /// <summary>Qwen3's mixture of experts: "Qwen3MoeForCausalLM".</summary>
    public const string Qwen3MoeForCausalLM = "Qwen3MoeForCausalLM";

    /// <summary>The XML tool-call format with one element per parameter in <see cref="ToolCallFormats"/>: "qwen3-coder".</summary>
    public const string ToolCallFormat = "qwen3-coder";

    /// <summary>Qwen's fill-in-the-middle format in <see cref="FillInMiddleFormats"/>: "qwen".</summary>
    public const string FillInMiddleFormat = "qwen";

    /// <summary>
    /// Qwen's fill-in-the-middle format (Qwen2.5-Coder, Qwen3-Coder; the tokens are in every Qwen2.5 and Qwen3 tokenizer):
    /// <c>&lt;|fim_prefix|&gt;prefix&lt;|fim_suffix|&gt;suffix&lt;|fim_middle|&gt;</c>, the repository level of the
    /// Qwen2.5-Coder report before it (<c>&lt;|repo_name|&gt;name\n</c>, <c>&lt;|file_sep|&gt;path\ntext\n</c> per file, then
    /// <c>&lt;|file_sep|&gt;path\n</c> of the file completed), ending on the stop ids Qwen2.5-Coder's README gives
    /// (<c>&lt;|fim_prefix|&gt;</c>, <c>&lt;|fim_suffix|&gt;</c>, <c>&lt;|fim_pad|&gt;</c>, <c>&lt;|repo_name|&gt;</c>,
    /// <c>&lt;|file_sep|&gt;</c>, <c>&lt;|endoftext|&gt;</c>, <c>&lt;|im_end|&gt;</c>).
    /// </summary>
    public static FillInMiddleLayout FillInMiddle { get; } = FillInMiddleLayout.PrefixSuffixMiddle(FillInMiddleFormat,
        "Qwen's <|fim_prefix|> prefix <|fim_suffix|> suffix <|fim_middle|>, with <|repo_name|> and <|file_sep|> files before it",
        new FillInMiddleTokens("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")
        {
            Repository = "<|repo_name|>",
            FileSeparator = "<|file_sep|>",
            Stop = ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>", "<|endoftext|>", "<|im_end|>"],
        });
}

// The family's tool-call format.
internal sealed class Qwen3CoderToolCallFormat : IToolCallFormat
{
    public string Name => QwenPlugin.ToolCallFormat;

    public bool Detect(ToolCallProbe probe) => probe.Call is { } call ? call.Contains("<function=" + ToolCallProbe.FunctionName + ">", StringComparison.Ordinal)
        : probe.Shows("<function=") && probe.Shows("<parameter=");

    public IToolCallParser Create(ToolCallContext context) => new Qwen3CoderToolCallParser(context);
}
