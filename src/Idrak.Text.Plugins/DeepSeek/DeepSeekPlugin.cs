// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;
using Idrak.Models.Abstractions;
using Idrak.Text.Plugins.Qwen;

namespace Idrak.Text.Plugins.DeepSeek;

/// <summary>
/// The DeepSeek family's own parts: its special-token tool-call format "deepseek" in <see cref="ToolCallFormats"/>, and
/// the GGUF pre-tokenizer "deepseek-r1-qwen" in <see cref="GgufPreTokenizers"/> (its models distilled into Qwen split as
/// Qwen2 does, <see cref="QwenArchitectures.SplitPattern"/>). Its models distilled into Llama or Qwen load through
/// those families.
/// </summary>
public static class DeepSeekPlugin
{
    /// <summary>The GGUF pre-tokenizer name: "deepseek-r1-qwen".</summary>
    public const string PreTokenizer = "deepseek-r1-qwen";

    /// <summary>DeepSeek's tool-call format in <see cref="ToolCallFormats"/>: "deepseek".</summary>
    public const string ToolCallFormat = "deepseek";

    /// <summary>The plug-in, as <see cref="IdrakPlugins"/> records it ("deepseek"; listed in <see cref="TextPlugins.All"/>).</summary>
    public static IIdrakPlugin Plugin { get; } = new DeepSeekRegistration();

    /// <summary>Whether the family is registered.</summary>
    public static bool IsRegistered => Plugin.IsRegistered;

    /// <summary>Registers the family's parts (through <see cref="IdrakPlugins"/>); once: calling it again changes nothing.</summary>
    public static void Register() => IdrakPlugins.Register(Plugin);

    /// <summary>Unregisters the family's parts, leaving the registries as <see cref="Register"/> found them.</summary>
    public static void Unregister() => IdrakPlugins.Unregister(Plugin);

    // The pre-tokenizer and the tool-call format.
    private sealed class DeepSeekRegistration() : IdrakPlugin("deepseek")
    {
        protected override void RegisterParts()
        {
            Part(() => GgufPreTokenizers.Register(PreTokenizer, QwenArchitectures.SplitPattern), () => GgufPreTokenizers.Unregister(PreTokenizer));
            Part(() => ToolCallFormats.Register(ToolCallFormat, p => p.Shows(DeepSeekToolCallParser.CallBegin), c => new DeepSeekToolCallParser(c)),
                () => ToolCallFormats.Unregister(ToolCallFormat));
        }
    }
}
