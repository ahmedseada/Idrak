// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;
using Idrak.Models.Abstractions;

namespace Idrak.Text.Plugins.Qwen;

/// <summary>
/// The Qwen family: Hugging Face's <c>Qwen2ForCausalLM</c>, <c>Qwen3ForCausalLM</c> and their mixtures of experts
/// <c>Qwen2MoeForCausalLM</c> and <c>Qwen3MoeForCausalLM</c> (config.json's "architectures") in
/// <see cref="PretrainedArchitectures"/> (<see cref="QwenArchitectures"/>), llama.cpp's "qwen2", "qwen3", "qwen2moe" and
/// "qwen3moe" in <see cref="GgufArchitectures"/>, its GGUF pre-tokenizer names in <see cref="GgufPreTokenizers"/>
/// (<see cref="PreTokenizers"/>), and the XML tool-call format "qwen3-coder" in <see cref="ToolCallFormats"/>. The
/// vision-language Qwen models are Idrak.Vision.Plugins'.
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

    /// <summary>The architecture names the plug-in registers in <see cref="PretrainedArchitectures"/>, in order.</summary>
    public static IReadOnlyList<string> Architectures { get; } = [Qwen2ForCausalLM, Qwen3ForCausalLM, Qwen2MoeForCausalLM, Qwen3MoeForCausalLM];

    /// <summary>llama.cpp's pre-tokenizer names that split as Qwen2 does (<see cref="QwenArchitectures.SplitPattern"/>): "qwen2", "qwen35", "megrez", "hunyuan".</summary>
    public static IReadOnlyList<string> PreTokenizers { get; } = ["qwen2", "qwen35", "megrez", "hunyuan"];

    /// <summary>The plug-in, as <see cref="IdrakPlugins"/> records it ("qwen"; listed in <see cref="TextPlugins.All"/>).</summary>
    public static IIdrakPlugin Plugin { get; } = new QwenRegistration();

    /// <summary>Whether the family is registered.</summary>
    public static bool IsRegistered => Plugin.IsRegistered;

    /// <summary>Registers the family (through <see cref="IdrakPlugins"/>); once: calling it again changes nothing.</summary>
    public static void Register() => IdrakPlugins.Register(Plugin);

    /// <summary>Unregisters the family, leaving the registries as <see cref="Register"/> found them.</summary>
    public static void Unregister() => IdrakPlugins.Unregister(Plugin);

    // The four architectures, the GGUF mappings, each pre-tokenizer name and the tool-call format.
    private sealed class QwenRegistration() : IdrakPlugin("qwen")
    {
        protected override void RegisterParts()
        {
            Part(() => PretrainedArchitectures.Register(Qwen2ForCausalLM, QwenArchitectures.Qwen2()), () => PretrainedArchitectures.Unregister(Qwen2ForCausalLM));
            Part(() => PretrainedArchitectures.Register(Qwen3ForCausalLM, QwenArchitectures.Qwen3()), () => PretrainedArchitectures.Unregister(Qwen3ForCausalLM));
            Part(() => PretrainedArchitectures.Register(Qwen2MoeForCausalLM, QwenArchitectures.Qwen2Moe()), () => PretrainedArchitectures.Unregister(Qwen2MoeForCausalLM));
            Part(() => PretrainedArchitectures.Register(Qwen3MoeForCausalLM, QwenArchitectures.Qwen3Moe()), () => PretrainedArchitectures.Unregister(Qwen3MoeForCausalLM));
            foreach (var (name, architecture) in QwenArchitectures.Gguf())
            {
                Part(() => GgufArchitectures.Register(name, architecture), () => GgufArchitectures.Unregister(name));
            }

            foreach (string name in PreTokenizers)
            {
                Part(() => GgufPreTokenizers.Register(name, QwenArchitectures.SplitPattern), () => GgufPreTokenizers.Unregister(name));
            }

            Part(() => ToolCallFormats.Register(ToolCallFormat, p => p.Call is { } call ? call.Contains("<function=" + ToolCallProbe.FunctionName + ">", StringComparison.Ordinal)
                : p.Shows("<function=") && p.Shows("<parameter="), c => new Qwen3CoderToolCallParser(c)), () => ToolCallFormats.Unregister(ToolCallFormat));
        }
    }
}
