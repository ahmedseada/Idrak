// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;
using Idrak.Models.Abstractions;

namespace Idrak.Text.Plugins.Llama;

/// <summary>
/// The Llama family: Hugging Face's <c>LlamaForCausalLM</c> (config.json's "architectures") in
/// <see cref="PretrainedArchitectures"/>, llama.cpp's "llama" in <see cref="GgufArchitectures"/> (Mistral's GGUF files
/// too; with experts they are Mixtral, <see cref="Mistral.MistralPlugin"/>'s), its GGUF pre-tokenizer names in
/// <see cref="GgufPreTokenizers"/> (<see cref="PreTokenizers"/>, which also become the fallback for a name nobody
/// registered when none is set), Llama 3's RoPE scaling "llama3" in <see cref="RopeScalings"/>
/// (<see cref="LlamaRopeScaling"/>), and Llama 3's chat template for GGUF files without one, "llama3" in
/// <see cref="GgufChatTemplates"/> (<see cref="LlamaArchitectures.ChatTemplateFallback"/>), so Idrak's <c>PretrainedModel.Load</c>
/// and <c>GgufModel</c> read them.
/// </summary>
public static class LlamaPlugin
{
    /// <summary>Llama: "LlamaForCausalLM".</summary>
    public const string LlamaForCausalLM = "LlamaForCausalLM";

    /// <summary>llama.cpp's architecture name: "llama".</summary>
    public const string GgufName = "llama";

    /// <summary>The architecture names the plug-in registers in <see cref="PretrainedArchitectures"/>.</summary>
    public static IReadOnlyList<string> Architectures { get; } = [LlamaForCausalLM];

    /// <summary>
    /// llama.cpp's pre-tokenizer names that split as Llama 3 does (<see cref="LlamaArchitectures.SplitPattern"/>): "llama3",
    /// "llama-bpe", "llama-v3", "smaug-bpe", "falcon3", "pixtral".
    /// </summary>
    public static IReadOnlyList<string> PreTokenizers { get; } = ["llama3", "llama-bpe", "llama-v3", "smaug-bpe", "falcon3", "pixtral"];

    /// <summary>The name of Llama 3's chat template in <see cref="GgufChatTemplates"/>: "llama3".</summary>
    public const string ChatTemplateName = "llama3";

    /// <summary>The plug-in, as <see cref="IdrakPlugins"/> records it ("llama"; listed in <see cref="TextPlugins.All"/>).</summary>
    public static IIdrakPlugin Plugin { get; } = new LlamaRegistration();

    /// <summary>Whether the family is registered.</summary>
    public static bool IsRegistered => Plugin.IsRegistered;

    /// <summary>Registers the family (through <see cref="IdrakPlugins"/>); once: calling it again changes nothing.</summary>
    public static void Register() => IdrakPlugins.Register(Plugin);

    /// <summary>Unregisters the family, leaving the registries as <see cref="Register"/> found them.</summary>
    public static void Unregister() => IdrakPlugins.Unregister(Plugin);

    // The architecture, the GGUF mapping, each pre-tokenizer name, the fallback when none is set, the RoPE scaling, and the
    // GGUF fallback chat template.
    private sealed class LlamaRegistration() : IdrakPlugin("llama")
    {
        private bool _fallback;                                      // whether this plug-in set GgufPreTokenizers.Fallback

        protected override void RegisterParts()
        {
            Part(() => PretrainedArchitectures.Register(LlamaForCausalLM, LlamaArchitectures.Llama()), () => PretrainedArchitectures.Unregister(LlamaForCausalLM));
            Part(() => GgufArchitectures.Register(GgufName, LlamaArchitectures.Gguf()), () => GgufArchitectures.Unregister(GgufName));
            foreach (string name in PreTokenizers)
            {
                Part(() => GgufPreTokenizers.Register(name, LlamaArchitectures.SplitPattern), () => GgufPreTokenizers.Unregister(name));
            }

            Part(() =>
            {
                if (GgufPreTokenizers.Fallback is null)
                {
                    GgufPreTokenizers.Fallback = PreTokenizers[0];
                    _fallback = true;
                }
            }, () =>
            {
                if (_fallback && GgufPreTokenizers.Fallback == PreTokenizers[0])
                {
                    GgufPreTokenizers.Fallback = null;
                }

                _fallback = false;
            });
            Part(() => RopeScalings.Register(LlamaRopeScaling.Kind), () => RopeScalings.Unregister(LlamaRopeScaling.Name));
            Part(() => GgufChatTemplates.Register(ChatTemplateName, LlamaArchitectures.ChatTemplateFallback()), () => GgufChatTemplates.Unregister(ChatTemplateName));
        }
    }
}
