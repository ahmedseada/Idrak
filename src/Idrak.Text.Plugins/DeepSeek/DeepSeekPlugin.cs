// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;
using Idrak.Models.Abstractions;
using Idrak.Text.Plugins.Qwen;

namespace Idrak.Text.Plugins.DeepSeek;

/// <summary>
/// The DeepSeek family's own parts: its special-token tool-call format "deepseek" in <see cref="ToolCallFormats"/>, and
/// the GGUF pre-tokenizer "deepseek-r1-qwen" in <see cref="GgufPreTokenizers"/> (its models distilled into Qwen split as
/// Qwen2 does, <see cref="QwenArchitectures.SplitPattern"/>), and DeepSeek-Coder's fill-in-the-middle format "deepseek"
/// in <see cref="FillInMiddleFormats"/> (<see cref="FillInMiddle"/>). Its models distilled into Llama or Qwen load through
/// those families.
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

    /// <summary>The plug-in, as <see cref="IdrakPlugins"/> records it ("deepseek"; listed in <see cref="TextPlugins.All"/>).</summary>
    public static IIdrakPlugin Plugin { get; } = new DeepSeekRegistration();

    /// <summary>Whether the family is registered.</summary>
    public static bool IsRegistered => Plugin.IsRegistered;

    /// <summary>Registers the family's parts (through <see cref="IdrakPlugins"/>); once: calling it again changes nothing.</summary>
    public static void Register() => IdrakPlugins.Register(Plugin);

    /// <summary>Unregisters the family's parts, leaving the registries as <see cref="Register"/> found them.</summary>
    public static void Unregister() => IdrakPlugins.Unregister(Plugin);

    // The pre-tokenizer, the tool-call format and the fill-in-the-middle format.
    private sealed class DeepSeekRegistration() : IdrakPlugin("deepseek")
    {
        protected override void RegisterParts()
        {
            Part(() => GgufPreTokenizers.Register(PreTokenizer, QwenArchitectures.SplitPattern), () => GgufPreTokenizers.Unregister(PreTokenizer));
            Part(() => ToolCallFormats.Register(new DeepSeekToolCallFormat()),
                () => ToolCallFormats.Unregister(ToolCallFormat));
            Part(() => FillInMiddleFormats.Register(FillInMiddle), () => FillInMiddleFormats.Unregister(FillInMiddleFormat));
        }
    }
}

// The family's tool-call format, registered in ToolCallFormats under its name.
internal sealed class DeepSeekToolCallFormat : IToolCallFormat
{
    public string Name => DeepSeekPlugin.ToolCallFormat;

    public string Summary => "DeepSeek's tool calls between its call markers";

    public IReadOnlySet<string> Keys => OptionKeys.None;

    public bool Detect(ToolCallProbe probe) => probe.Shows(DeepSeekToolCallParser.CallBegin);

    public IToolCallParser Create(ToolCallContext context) => new DeepSeekToolCallParser(context);
}
