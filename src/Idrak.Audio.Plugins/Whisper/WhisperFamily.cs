// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models;

namespace Idrak.Audio.Plugins.Whisper;

/// <summary>
/// The Whisper-style family: reads a checkpoint saved by transformers (<c>WhisperForConditionalGeneration</c>) into a
/// <see cref="WhisperRecognizer"/>. The folder holds config.json, the weights, tokenizer.json, preprocessor_config.json
/// (the features) and, when saved, generation_config.json (the token ids; the tokenizer's added tokens otherwise).
/// </summary>
public sealed class WhisperFamily : ISpeechFamily
{
    /// <summary>The architecture name the family is registered under.</summary>
    public const string Architecture = "WhisperForConditionalGeneration";

    /// <inheritdoc />
    public string Name => Architecture;

    /// <inheritdoc />
    public ISpeechRecognizer Read(SpeechCheckpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        string folder = checkpoint.Folder ?? throw new InvalidDataException($"{checkpoint.Path}: a Whisper checkpoint is a folder (tokenizer.json and preprocessor_config.json beside the weights).");
        string tokenizerFile = Path.Combine(folder, "tokenizer.json");
        if (!File.Exists(tokenizerFile))
        {
            throw new InvalidDataException($"{folder} has no tokenizer.json.");
        }

        var generation = ReadJson(Path.Combine(folder, "generation_config.json"));
        var config = WhisperConfig.Read(checkpoint.Config, generation, WhisperConfig.AddedTokens(tokenizerFile));
        var tokenizer = PretrainedTokenizers.Load(tokenizerFile);
        var preprocessor = ReadJson(Path.Combine(folder, "preprocessor_config.json"));
        var defaults = new AudioFeatureOptions { Mels = config.Mels };
        var featureOptions = preprocessor is null ? defaults : AudioFeatureOptions.FromConfig(preprocessor, defaults);
        if (preprocessor is null)
        {
            checkpoint.Notes.Add("no preprocessor_config.json: the features take the log-mel defaults with the model's mel count");
        }

        if (featureOptions.Mels != config.Mels)
        {
            throw new InvalidDataException($"preprocessor_config.json gives {featureOptions.Mels} mels, config.json {config.Mels}.");
        }

        var features = AudioFeatureExtractors.Create("log-mel", featureOptions);
        var device = checkpoint.Options.Device ?? Device.Default;
        var layout = new WhisperLayout();                                     // where each tensor went, for Save
        var encoder = WhisperEncoder.Read(checkpoint.Tensors, config, checkpoint.Options.Weights, device, "model.encoder.", layout);
        try
        {
            var decoder = WhisperDecoder.Read(checkpoint.Tensors, config, checkpoint.Options.Weights, device, "model.decoder.", layout);
            var source = new WhisperSource(folder, checkpoint.Config, generation, layout);
            return new WhisperRecognizer(config, tokenizer, features, encoder, decoder, device, checkpoint.Options.Cache, [.. checkpoint.Notes], source,
                checkpoint.Options.Language, checkpoint.Options.Task);
        }
        catch
        {
            encoder.Dispose();
            throw;
        }
    }

    private static JsonObject? ReadJson(string file) =>
        File.Exists(file) ? JsonNode.Parse(File.ReadAllText(file)) as JsonObject ?? throw new InvalidDataException($"{file} is not a JSON object.") : null;
}
