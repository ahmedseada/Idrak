// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

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

/// <summary>The family: reads a VITS checkpoint folder (config.json, vocab.json, the safetensors) into a <see cref="VitsSynthesizer"/>.</summary>
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

        var weights = new VitsWeights(checkpoint.Tensors, options.Device ?? Device.Default);
        return new VitsSynthesizer(weights, config, tokenizer, checkpoint.Notes);
    }
}
