// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models;

namespace Idrak.VitsSpeech;

/// <summary>
/// Registers the VITS text-to-speech family (transformers' <c>VitsModel</c>, the architecture of the MMS-TTS
/// checkpoints) in <see cref="SpeechSynthesisFamilies"/> under <see cref="VitsSpeechFamily.Architecture"/>.
/// </summary>
public static class VitsSpeechPlugin
{
    /// <summary>Registers the family.</summary>
    public static void Register() => SpeechSynthesisFamilies.Register(new VitsSpeechFamily());

    /// <summary>The entry the command line calls for <c>-P Idrak.VitsSpeech.dll</c>.</summary>
    public static void RegisterIdrakPlugin() => Register();

    /// <summary>Removes the registration <see cref="Register"/> made.</summary>
    public static void Unregister() => SpeechSynthesisFamilies.Unregister(VitsSpeechFamily.Architecture);
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
