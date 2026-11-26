// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models.Abstractions;

namespace Idrak.Text.Plugins.Gemma;

/// <summary>
/// The Gemma family's names and parts: Gemma, Gemma 2 and Gemma 3 text models (Hugging Face's <c>GemmaForCausalLM</c>,
/// <c>Gemma2ForCausalLM</c>, <c>Gemma3ForCausalLM</c>, config.json's "architectures") and the text decoder of Gemma 3 with
/// images (<c>Gemma3ForConditionalGeneration</c>), read by <see cref="GemmaArchitectures"/>; llama.cpp's "gemma",
/// "gemma2" and "gemma3", read by <see cref="GemmaGguf"/>; and CodeGemma's fill-in-the-middle format "gemma"
/// (<see cref="FillInMiddle"/>). Registering them is not on the new design yet.
/// </summary>
public static class GemmaPlugin
{
    /// <summary>Gemma: "GemmaForCausalLM".</summary>
    public const string GemmaForCausalLM = "GemmaForCausalLM";

    /// <summary>Gemma 2: "Gemma2ForCausalLM".</summary>
    public const string Gemma2ForCausalLM = "Gemma2ForCausalLM";

    /// <summary>Gemma 3 text: "Gemma3ForCausalLM".</summary>
    public const string Gemma3ForCausalLM = "Gemma3ForCausalLM";

    /// <summary>Gemma 3 with images (its text decoder here): "Gemma3ForConditionalGeneration".</summary>
    public const string Gemma3ForConditionalGeneration = "Gemma3ForConditionalGeneration";

    /// <summary>CodeGemma's fill-in-the-middle format in <see cref="FillInMiddleFormats"/>: "gemma".</summary>
    public const string FillInMiddleFormat = "gemma";

    /// <summary>
    /// CodeGemma's fill-in-the-middle format (Hugging Face's CodeGemma post):
    /// <c>&lt;bos&gt;&lt;|fim_prefix|&gt;prefix&lt;|fim_suffix|&gt;suffix&lt;|fim_middle|&gt;</c> (its tokenizer adds the
    /// <c>&lt;bos&gt;</c>; Idrak's adds none, so the format writes it), other files each after a
    /// <c>&lt;|file_separator|&gt;</c> and its path, ending on the post's terminators (the three FIM tokens,
    /// <c>&lt;|file_separator|&gt;</c> and <c>&lt;eos&gt;</c>).
    /// </summary>
    public static FillInMiddleLayout FillInMiddle { get; } = FillInMiddleLayout.PrefixSuffixMiddle(FillInMiddleFormat,
        "CodeGemma's <|fim_prefix|> prefix <|fim_suffix|> suffix <|fim_middle|>, with <|file_separator|> files before it",
        new FillInMiddleTokens("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")
        {
            Begin = "<bos>",
            FileSeparator = "<|file_separator|>",
            Stop = ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|file_separator|>", "<eos>"],
        });
}
