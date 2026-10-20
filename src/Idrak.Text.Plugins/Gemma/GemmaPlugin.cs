// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;
using Idrak.Models.Abstractions;

namespace Idrak.Text.Plugins.Gemma;

/// <summary>
/// The Gemma family: Gemma, Gemma 2 and Gemma 3 text models (Hugging Face's <c>GemmaForCausalLM</c>,
/// <c>Gemma2ForCausalLM</c>, <c>Gemma3ForCausalLM</c>, config.json's "architectures") and the text decoder of Gemma 3 with
/// images (<c>Gemma3ForConditionalGeneration</c>, in each of its checkpoints' namings), registered in
/// <see cref="PretrainedArchitectures"/> (<see cref="GemmaArchitectures"/>), so Idrak's <c>PretrainedModel.Load</c> reads
/// them. The vision part of Gemma 3 with images is Idrak.Vision.Plugins' (its Gemma 3 vision plug-in registers this one
/// when it is missing); without it, such a checkpoint is refused naming the vision family.
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

    /// <summary>The architecture names the plug-in registers in <see cref="PretrainedArchitectures"/>, in order.</summary>
    public static IReadOnlyList<string> Architectures { get; } = [GemmaForCausalLM, Gemma2ForCausalLM, Gemma3ForCausalLM, Gemma3ForConditionalGeneration];

    /// <summary>The plug-in, as <see cref="IdrakPlugins"/> records it ("gemma"; listed in <see cref="TextPlugins.All"/>).</summary>
    public static IIdrakPlugin Plugin { get; } = new GemmaRegistration();

    /// <summary>Whether the family is registered.</summary>
    public static bool IsRegistered => Plugin.IsRegistered;

    /// <summary>Registers the family under each of its architecture names (through <see cref="IdrakPlugins"/>); once: calling it again changes nothing.</summary>
    public static void Register() => IdrakPlugins.Register(Plugin);

    /// <summary>Unregisters the family, leaving <see cref="PretrainedArchitectures"/> as <see cref="Register"/> found it.</summary>
    public static void Unregister() => IdrakPlugins.Unregister(Plugin);

    // The family under each architecture name, one part each.
    private sealed class GemmaRegistration() : IdrakPlugin("gemma")
    {
        protected override void RegisterParts()
        {
            Part(() => PretrainedArchitectures.Register(GemmaForCausalLM, GemmaArchitectures.Gemma()), () => PretrainedArchitectures.Unregister(GemmaForCausalLM));
            Part(() => PretrainedArchitectures.Register(Gemma2ForCausalLM, GemmaArchitectures.Gemma2()), () => PretrainedArchitectures.Unregister(Gemma2ForCausalLM));
            Part(() => PretrainedArchitectures.Register(Gemma3ForCausalLM, GemmaArchitectures.Gemma3()), () => PretrainedArchitectures.Unregister(Gemma3ForCausalLM));
            Part(() => PretrainedArchitectures.Register(Gemma3ForConditionalGeneration, GemmaArchitectures.Gemma3VisionLanguage()),
                () => PretrainedArchitectures.Unregister(Gemma3ForConditionalGeneration));
        }
    }
}
