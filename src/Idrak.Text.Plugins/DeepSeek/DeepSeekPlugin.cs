// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models.Abstractions;
using Idrak.Text.Plugins.Qwen;

namespace Idrak.Text.Plugins.DeepSeek;

/// <summary>
/// The DeepSeek family's own parts: its special-token tool-call format "deepseek", the GGUF pre-tokenizer
/// "deepseek-r1-qwen" (its models distilled into Qwen split as Qwen2 does, <see cref="QwenArchitectures.SplitPattern"/>),
/// and DeepSeek-Coder's fill-in-the-middle format "deepseek" (<see cref="FillInMiddle"/>). Registering them is not on the
/// new design yet. Its models distilled into Llama or Qwen load through those families.
/// </summary>
public static class DeepSeekPlugin
{
    /// <summary>The GGUF pre-tokenizer name: "deepseek-r1-qwen".</summary>
    public const string PreTokenizer = "deepseek-r1-qwen";

    /// <summary>DeepSeek's tool-call format in <see cref="ToolCallFormats"/>: "deepseek".</summary>
    public const string ToolCallFormat = "deepseek";

    /// <summary>DeepSeek-Coder's fill-in-the-middle format in <see cref="FillInMiddleFormats"/>: "deepseek".</summary>
    public const string FillInMiddleFormat = "deepseek";

    /// <summary>
    /// DeepSeek-Coder's fill-in-the-middle format (its README's "code insertion"):
    /// <c>&lt;｜begin▁of▁sentence｜&gt;&lt;｜fim▁begin｜&gt;prefix&lt;｜fim▁hole｜&gt;suffix&lt;｜fim▁end｜&gt;</c> (its tokenizer adds the
    /// beginning of sentence; Idrak's adds none, so the format writes it), the repository's files at the start of the prefix
    /// as its README's repository-level completion writes them (<c>#path\ntext\n</c>), ending on
    /// <c>&lt;｜end▁of▁sentence｜&gt;</c> or the instruct models' <c>&lt;|EOT|&gt;</c>.
    /// </summary>
    public static FillInMiddleLayout FillInMiddle { get; } = FillInMiddleLayout.PrefixSuffixMiddle(FillInMiddleFormat,
        "DeepSeek-Coder's <｜fim▁begin｜> prefix <｜fim▁hole｜> suffix <｜fim▁end｜>, with #path files at the start of the prefix",
        new FillInMiddleTokens("<｜fim▁begin｜>", "<｜fim▁hole｜>", "<｜fim▁end｜>")
        {
            Begin = "<｜begin▁of▁sentence｜>",
            SnippetMarker = "#",
            Stop = ["<｜end▁of▁sentence｜>", "<|EOT|>"],
        });
}

// The family's tool-call format.
internal sealed class DeepSeekToolCallFormat : IToolCallFormat
{
    public string Name => DeepSeekPlugin.ToolCallFormat;

    public bool Detect(ToolCallProbe probe) => probe.Shows(DeepSeekToolCallParser.CallBegin);

    public IToolCallParser Create(ToolCallContext context) => new DeepSeekToolCallParser(context);
}
