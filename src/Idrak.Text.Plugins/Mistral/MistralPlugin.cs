// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;
using Idrak.Models.Abstractions;
using Idrak.Text.Plugins.Llama;

namespace Idrak.Text.Plugins.Mistral;

/// <summary>
/// The Mistral family: Hugging Face's <c>MistralForCausalLM</c> and <c>MixtralForCausalLM</c> (config.json's
/// "architectures") in <see cref="PretrainedArchitectures"/> (<see cref="MistralArchitectures"/>), the tekken GGUF
/// pre-tokenizer in <see cref="GgufPreTokenizers"/>, and Mistral's tool-call format "mistral" (<c>[TOOL_CALLS]</c>) in
/// <see cref="ToolCallFormats"/>. Its GGUF files are llama.cpp's "llama", so it registers <see cref="LlamaPlugin"/> when
/// that is missing (and takes it back only then).
/// </summary>
public static class MistralPlugin
{
    /// <summary>Mistral: "MistralForCausalLM".</summary>
    public const string MistralForCausalLM = "MistralForCausalLM";

    /// <summary>Mixtral, Mistral's mixture of experts: "MixtralForCausalLM".</summary>
    public const string MixtralForCausalLM = "MixtralForCausalLM";

    /// <summary>The tekken GGUF pre-tokenizer: "tekken".</summary>
    public const string PreTokenizer = "tekken";

    /// <summary>Mistral's tool-call format in <see cref="ToolCallFormats"/>: "mistral".</summary>
    public const string ToolCallFormat = "mistral";

    /// <summary>The architecture names the plug-in registers in <see cref="PretrainedArchitectures"/>, in order.</summary>
    public static IReadOnlyList<string> Architectures { get; } = [MistralForCausalLM, MixtralForCausalLM];

    /// <summary>The plug-in, as <see cref="IdrakPlugins"/> records it ("mistral"; listed in <see cref="TextPlugins.All"/>).</summary>
    public static IIdrakPlugin Plugin { get; } = new MistralRegistration();

    /// <summary>Whether the family is registered.</summary>
    public static bool IsRegistered => Plugin.IsRegistered;

    /// <summary>Registers the family (through <see cref="IdrakPlugins"/>); once: calling it again changes nothing.</summary>
    public static void Register() => IdrakPlugins.Register(Plugin);

    /// <summary>Unregisters the family, leaving the registries as <see cref="Register"/> found them.</summary>
    public static void Unregister() => IdrakPlugins.Unregister(Plugin);

    // Llama's GGUF mapping when missing, the two architectures, the pre-tokenizer and the tool-call format.
    private sealed class MistralRegistration() : IdrakPlugin("mistral")
    {
        protected override void RegisterParts()
        {
            Requires(LlamaPlugin.Plugin);
            Part(() => PretrainedArchitectures.Register(MistralForCausalLM, MistralArchitectures.Mistral()), () => PretrainedArchitectures.Unregister(MistralForCausalLM));
            Part(() => PretrainedArchitectures.Register(MixtralForCausalLM, MistralArchitectures.Mixtral()), () => PretrainedArchitectures.Unregister(MixtralForCausalLM));
            Part(() => GgufPreTokenizers.Register(PreTokenizer, MistralArchitectures.TekkenPattern), () => GgufPreTokenizers.Unregister(PreTokenizer));
            Part(() => ToolCallFormats.Register(ToolCallFormat, p => p.Shows(MistralToolCallParser.Marker), c => new MistralToolCallParser(c)),
                () => ToolCallFormats.Unregister(ToolCallFormat));
        }
    }
}
