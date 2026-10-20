// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models;

namespace Idrak.Audio.Plugins.IdrakAudio;

/// <summary>
/// The IdrakAudio synthesis family (registered in <see cref="SpeechSynthesisFamilies"/> under
/// <see cref="IdrakAudioFamily.Architecture"/>): it makes a "tts" model from a spec (<see cref="Create"/>, through
/// <c>SpeechSynthesisModels.Create</c>) and reads one back from its folder (<see cref="Read"/>, through
/// <c>SpeechSynthesisModels.Load</c>) as an <see cref="IdrakAudioTtsModel"/>. A recognizer task is refused, naming the
/// recognizers' loader.
/// </summary>
public sealed class IdrakAudioSynthesisFamily : ISpeechSynthesisFamily
{
    /// <inheritdoc />
    public string Name => IdrakAudioFamily.Architecture;

    /// <inheritdoc />
    public ISpeechSynthesizer Read(SpeechSynthesisCheckpoint checkpoint, SpeechSynthesisLoadOptions options)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        ArgumentNullException.ThrowIfNull(options);
        var config = IdrakAudioConfig.FromJson(checkpoint.Config);
        Task(config);
        string tokenizerFile = Path.Combine(checkpoint.Folder ?? throw new InvalidDataException("An IdrakAudio model is read from its folder (tokenizer.json lives beside config.json)."), "tokenizer.json");
        if (!File.Exists(tokenizerFile))
        {
            throw new InvalidDataException($"An IdrakAudio model folder holds its tokenizer.json; {tokenizerFile} was not found.");
        }

        var tokenizer = PretrainedTokenizers.Load(tokenizerFile);
        var device = options.Device ?? Device.Default;
        var weights = IdrakAudioWeights.Read(checkpoint.Tensors, device);
        IdrakAudioTtsModel? model = null;
        try
        {
            model = new IdrakAudioTtsModel(config, weights, tokenizer, device);
            if (weights.Unread() is { Count: > 0 } unread)
            {
                throw new InvalidDataException($"The IdrakAudio checkpoint holds tensors its config does not use ({string.Join(", ", unread.Take(5))}"
                    + $"{(unread.Count > 5 ? ", …" : "")}): its config.json and weights do not match.");
            }

            return model;
        }
        catch
        {
            if (model is not null)
            {
                model.Dispose();
            }
            else
            {
                foreach (var entry in weights.Entries)
                {
                    entry.Tensor.Dispose();
                }
            }

            throw;
        }
    }

    /// <inheritdoc />
    /// <remarks>
    /// The spec holds config.json's keys (<see cref="IdrakAudioConfig"/>; a missing key takes its default) with "task":
    /// "tts"; the classes are the tokenizer's tokens (label i is token i). The weights are drawn on the CPU from
    /// <paramref name="seed"/>.
    /// </remarks>
    public CreatedSpeechModel Create(JsonObject spec, ITokenizer tokenizer, int seed)
    {
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(tokenizer);
        if (tokenizer is not IPretrainedTokenizer)
        {
            throw new NotSupportedException($"An IdrakAudio model keeps its tokenizer as tokenizer.json; give a pretrained tokenizer (one PretrainedTokenizers.Load reads or a TokenizerTrainers trainer learns), not a {tokenizer.GetType().Name}.");
        }

        var config = IdrakAudioConfig.FromJson(spec, tokenizer.VocabularySize);
        Task(config);
        if (config.Classes != tokenizer.VocabularySize)
        {
            throw new ArgumentException($"The spec gives {config.Classes} classes; a tts model's classes are the tokenizer's {tokenizer.VocabularySize} tokens.", nameof(spec));
        }

        var weights = IdrakAudioWeights.Draw(seed, config.InitScale, Device.Cpu);
        using var network = new IdrakAudioTtsNetwork(config, weights);
        var tensors = new List<(string, int[], float[])>(weights.Entries.Count);
        foreach (var (name, tensor, stored, transposed) in weights.Entries)
        {
            tensors.Add((name, stored, IdrakAudioWeights.Stored(tensor, transposed)));
        }

        return new CreatedSpeechModel(config.ToJson(), tensors, new Dictionary<string, string>());
    }

    private static void Task(IdrakAudioConfig config)
    {
        if (!config.Synthesis)
        {
            throw new NotSupportedException($"IdrakAudio task '{config.Task}' is a speech recognizer, not a synthesizer: read it with SpeechModels.Load (SpeechSynthesisModels reads \"{IdrakAudioFamily.TtsTask}\" models).");
        }
    }
}
