// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models;

namespace Idrak.Audio.Plugins.Vits;

/// <summary>
/// The VITS text-to-speech family (transformers' <c>VitsModel</c>, the architecture of the MMS-TTS checkpoints): the
/// family <see cref="VitsSpeechFamily"/> under <see cref="VitsSpeechFamily.Architecture"/>, uroman's romanisation of the
/// Arabic script as <see cref="UromanArabic"/> (the checkpoints that set <c>is_uroman</c> read romanised text), and the
/// espeak-ng phonemizer its <c>phonemize</c> checkpoints read (<see cref="EspeakNg.EspeakNgPlugin"/>).
/// </summary>
public static class VitsPlugin
{
    /// <summary>uroman's Arabic romanizer's name: "uroman-arabic".</summary>
    public const string UromanArabic = "uroman-arabic";
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
    /// The tokenizer's flags (<see cref="VitsTokenizer.Read"/>): <c>phonemize</c> with its <c>language</c> (phonemes from a
    /// phonemizer), <c>is_uroman</c> (romanised text).
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
