// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;
using Idrak.Models;

namespace Idrak.Audio.Plugins.Vits;

/// <summary>
/// The VITS text-to-speech family (transformers' <c>VitsModel</c>, the architecture of the MMS-TTS checkpoints): the
/// family in <see cref="SpeechSynthesisFamilies"/> under <see cref="VitsSpeechFamily.Architecture"/>, uroman's
/// romanisation of the Arabic script in <see cref="TextRomanizers"/> as <see cref="UromanArabic"/> (the checkpoints that
/// set <c>is_uroman</c> read romanised text), and, through <see cref="EspeakNg.EspeakNgPlugin"/>, the espeak-ng phonemizer
/// its <c>phonemize</c> checkpoints read (registered when missing, and taken back with this family only then).
/// </summary>
public static class VitsPlugin
{
    /// <summary>The name uroman's Arabic romanizer is registered under in <see cref="TextRomanizers"/>: "uroman-arabic".</summary>
    public const string UromanArabic = "uroman-arabic";

    /// <summary>The plug-in, as <see cref="IdrakPlugins"/> records it ("vits"; listed in <see cref="AudioPlugins.All"/>).</summary>
    public static IIdrakPlugin Plugin { get; } = new VitsRegistration();

    /// <summary>Whether the family is registered.</summary>
    public static bool IsRegistered => Plugin.IsRegistered;

    /// <summary>Registers the family, its romanizer and the phonemizer it needs (through <see cref="IdrakPlugins"/>); once: calling it again changes nothing.</summary>
    public static void Register() => IdrakPlugins.Register(Plugin);

    /// <summary>Unregisters the family, leaving the registries as <see cref="Register"/> found them.</summary>
    public static void Unregister() => IdrakPlugins.Unregister(Plugin);

    // The phonemizer it needs, its romanizer, then the family.
    private sealed class VitsRegistration() : IdrakPlugin("vits")
    {
        protected override void RegisterParts()
        {
            Requires(EspeakNg.EspeakNgPlugin.Plugin);
            Part(() => TextRomanizers.Register(new UromanArabicRomanizer()), () => TextRomanizers.Unregister(UromanArabic));
            Part(() => SpeechSynthesisFamilies.Register(new VitsSpeechFamily()), () => SpeechSynthesisFamilies.Unregister(VitsSpeechFamily.Architecture));
        }
    }
}

/// <summary>
/// The family: reads a VITS checkpoint folder (config.json, vocab.json, the safetensors) into a <see cref="VitsSynthesizer"/>;
/// with <see cref="SpeechSynthesisLoadOptions.Training"/>, one the adversarial trainer fine-tunes (plan 21, S9 route C).
/// </summary>
public sealed class VitsSpeechFamily : ISpeechSynthesisFamily
{
    /// <summary>The architecture name the checkpoints give, "VitsModel".</summary>
    public const string Architecture = "VitsModel";

    /// <inheritdoc />
    public string Name => Architecture;

    /// <inheritdoc />
    public ISpeechSynthesizer Read(SpeechSynthesisCheckpoint checkpoint, SpeechSynthesisLoadOptions options)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        ArgumentNullException.ThrowIfNull(options);
        var config = VitsConfig.FromJson(checkpoint.Config);
        var tokenizer = VitsTokenizer.Read(checkpoint.Folder ?? throw new InvalidDataException("A VITS checkpoint is read from a folder (vocab.json lives beside config.json)."));
        if (tokenizer.Size > config.VocabSize)
        {
            throw new InvalidDataException($"vocab.json has ids up to {tokenizer.Size - 1}, config.json's vocab_size is {config.VocabSize}.");
        }

        if (options.Training && config.Speakers > 1)
        {
            throw new NotSupportedException($"Training fine-tunes a single-speaker checkpoint (route C: one person's voice); this one has {config.Speakers} speakers.");
        }

        var weights = new StoredLayers(checkpoint.Tensors, options.Device ?? Device.Default, options.Training);
        return new VitsSynthesizer(weights, config, tokenizer, checkpoint.Notes, checkpoint.Folder, checkpoint.Open);
    }
}
