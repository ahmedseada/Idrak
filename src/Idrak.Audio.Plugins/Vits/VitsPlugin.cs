// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Plugins;
using Idrak.Models;

namespace Idrak.Audio.Plugins.Vits;

/// <summary>
/// The VITS text-to-speech family (transformers' <c>VitsModel</c>, the architecture of the MMS-TTS checkpoints): the
/// family in <see cref="SpeechSynthesisFamilies"/> under <see cref="VitsSpeechFamily.Architecture"/>, uroman's
/// romanisation of the Arabic script in <see cref="TextRomanizers"/> as <see cref="UromanArabic"/> (the checkpoints that
/// set <c>is_uroman</c> read romanised text), and, through <see cref="EspeakNg.EspeakNgPlugin"/>, the espeak-ng phonemizer
/// its <c>phonemize</c> checkpoints read (registered when missing, and taken back with this family only then).
/// <see cref="Family"/> is the family and its romanizer without the phonemizer, for a host that may start no outside
/// program (an app): its <c>phonemize</c> checkpoints then say what they need
/// (<see cref="SpeechSynthesisModels.TextNeeds"/>) instead of running espeak-ng.
/// </summary>
public static class VitsPlugin
{
    /// <summary>The name uroman's Arabic romanizer is registered under in <see cref="TextRomanizers"/>: "uroman-arabic".</summary>
    public const string UromanArabic = "uroman-arabic";

    /// <summary>The plug-in, as <see cref="IdrakPlugins"/> records it ("vits"; listed in <see cref="AudioPlugins.All"/>): <see cref="Family"/> and the espeak-ng phonemizer.</summary>
    public static IIdrakPlugin Plugin { get; } = new VitsRegistration();

    /// <summary>
    /// The family and its romanizer alone, without the espeak-ng phonemizer ("vits-family"): what <see cref="Plugin"/>
    /// builds on, registered by itself where no outside program may run.
    /// </summary>
    public static IIdrakPlugin Family { get; } = new VitsFamilyRegistration();

    /// <summary>Whether the family is registered (by <see cref="Register"/> or <see cref="RegisterFamily"/>).</summary>
    public static bool IsRegistered => Family.IsRegistered;

    /// <summary>Registers the family, its romanizer and the phonemizer it needs (through <see cref="IdrakPlugins"/>); once: calling it again changes nothing.</summary>
    public static void Register() => IdrakPlugins.Register(Plugin);

    /// <summary>Unregisters what <see cref="Register"/> registered, leaving the registries as it found them (a <see cref="RegisterFamily"/> before it stays).</summary>
    public static void Unregister() => IdrakPlugins.Unregister(Plugin);

    /// <summary>
    /// Registers the family and its romanizer without the espeak-ng phonemizer (<see cref="Family"/>, through
    /// <see cref="IdrakPlugins"/>); once. A <c>phonemize</c> checkpoint then needs a phonemizer registered another way
    /// (<see cref="TextPhonemizers.Register"/>), and says so (<see cref="SpeechTextNeeds.Missing"/>).
    /// </summary>
    public static void RegisterFamily() => IdrakPlugins.Register(Family);

    /// <summary>Unregisters what <see cref="RegisterFamily"/> registered, unless <see cref="Register"/> holds it.</summary>
    public static void UnregisterFamily()
    {
        if (!Plugin.IsRegistered)
        {
            IdrakPlugins.Unregister(Family);
        }
    }

    // The phonemizer it needs, then the family with its romanizer.
    private sealed class VitsRegistration() : IdrakPlugin("vits")
    {
        protected override void RegisterParts()
        {
            Requires(EspeakNg.EspeakNgPlugin.Plugin);
            Requires(Family);
        }
    }

    // Its romanizer, then the family.
    private sealed class VitsFamilyRegistration() : IdrakPlugin("vits-family")
    {
        protected override void RegisterParts()
        {
            Part(() => TextRomanizers.Register(new UromanArabicRomanizer()), () => TextRomanizers.Unregister(UromanArabic));
            Part(() => SpeechSynthesisFamilies.Register(new VitsSpeechFamily()), () => SpeechSynthesisFamilies.Unregister(VitsSpeechFamily.Architecture));
        }
    }
}

/// <summary>
/// The family: reads a VITS checkpoint folder (config.json, vocab.json, the safetensors) into a <see cref="VitsSynthesizer"/>;
/// with <see cref="SpeechSynthesisLoadOptions.Training"/>, one the adversarial trainer fine-tunes (plan 21, S9 route C).
/// What a checkpoint's text needs (phonemes, romanised text) is its tokenizer_config.json's (<see cref="TextNeeds"/>).
/// </summary>
public sealed class VitsSpeechFamily : ITrainableSpeechSynthesisFamily, ITextFrontEndSynthesisFamily
{
    /// <summary>The architecture name the checkpoints give, "VitsModel".</summary>
    public const string Architecture = "VitsModel";

    /// <inheritdoc />
    public string Name => Architecture;

    /// <inheritdoc />
    /// <remarks>
    /// The tokenizer's flags (<see cref="VitsTokenizer.Read"/>): <c>phonemize</c> with its <c>language</c> (phonemes from the
    /// most recently registered phonemizer), <c>is_uroman</c> (romanised text).
    /// </remarks>
    public SpeechTextNeeds TextNeeds(JsonObject config, string folder)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentException.ThrowIfNullOrEmpty(folder);
        var tokenizer = VitsTokenizer.Read(folder);
        return new SpeechTextNeeds
        {
            Phonemes = tokenizer.Phonemize,
            PhonemeLanguage = tokenizer.Phonemize ? tokenizer.PhonemeLanguage : null,
            Phonemizer = tokenizer.Phonemize ? tokenizer.Phonemizer : null,
            Romanized = tokenizer.IsUroman,
        };
    }

    /// <inheritdoc />
    public ISpeechSynthesizer Read(SpeechSynthesisCheckpoint checkpoint, SpeechSynthesisLoadOptions options) => ReadVits(checkpoint, options);

    /// <inheritdoc />
    public IAdversarialSpeechSynthesizer ReadForTraining(SpeechSynthesisCheckpoint checkpoint, SpeechSynthesisLoadOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return ReadVits(checkpoint, options with { Training = true });
    }

    private static VitsSynthesizer ReadVits(SpeechSynthesisCheckpoint checkpoint, SpeechSynthesisLoadOptions options)
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
