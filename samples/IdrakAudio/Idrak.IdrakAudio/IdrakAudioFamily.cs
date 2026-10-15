// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models;

namespace Idrak.IdrakAudio;

/// <summary>
/// The IdrakAudio speech family (registered under <see cref="Architecture"/>): it makes a model from a spec
/// (<see cref="Create"/>, through <c>SpeechModels.Create</c>) and reads one back from its folder (<see cref="Read"/>, through
/// <c>SpeechModels.Load</c>). The task is the config's "task": "ctc" (<see cref="IdrakAudioCtcModel"/>) and "seq2seq"
/// (<see cref="IdrakAudioSeq2SeqModel"/>) are built; another is refused naming them.
/// </summary>
public sealed class IdrakAudioFamily : ISpeechFamily
{
    /// <summary>The architecture name the family is registered under and its config.json gives.</summary>
    public const string Architecture = "IdrakAudio";

    /// <summary>The CTC recognizer task.</summary>
    public const string CtcTask = "ctc";

    /// <summary>The encoder-decoder recognizer task.</summary>
    public const string Seq2SeqTask = "seq2seq";

    /// <inheritdoc />
    public string Name => Architecture;

    /// <inheritdoc />
    public ISpeechRecognizer Read(SpeechCheckpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        var config = IdrakAudioConfig.FromJson(checkpoint.Config);
        Task(config);
        string tokenizerFile = Path.Combine(checkpoint.Folder ?? Path.GetDirectoryName(Path.GetFullPath(checkpoint.Path))!, "tokenizer.json");
        if (!File.Exists(tokenizerFile))
        {
            throw new InvalidDataException($"An IdrakAudio model folder holds its tokenizer.json; {tokenizerFile} was not found.");
        }

        var tokenizer = BpeTokenizer.Load(tokenizerFile);
        var device = checkpoint.Options.Device ?? Device.Default;
        var weights = IdrakAudioWeights.Read(checkpoint.Tensors, device);
        ISpeechRecognizer? model = null;
        try
        {
            model = config.Sequence ? new IdrakAudioSeq2SeqModel(config, weights, tokenizer, device) : new IdrakAudioCtcModel(config, weights, tokenizer, device);
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
    /// The spec holds config.json's keys (<see cref="IdrakAudioConfig"/>; a missing key takes its default); the classes are
    /// the tokenizer's tokens and the blank ("ctc"), or the tokens and the start and end token, one class past them
    /// ("seq2seq"). The weights are drawn on the CPU from <paramref name="seed"/>.
    /// </remarks>
    public CreatedSpeechModel Create(JsonObject spec, ITokenizer tokenizer, int seed)
    {
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(tokenizer);
        if (tokenizer is not BpeTokenizer)
        {
            throw new NotSupportedException($"An IdrakAudio model keeps its tokenizer as tokenizer.json; give a BpeTokenizer (TokenizerTrainers \"bpe\"), not a {tokenizer.GetType().Name}.");
        }

        var config = IdrakAudioConfig.FromJson(spec, tokenizer.VocabularySize);
        Task(config);
        if (config.Classes != tokenizer.VocabularySize + 1)
        {
            throw new ArgumentException($"The spec gives {config.Classes} classes; the tokenizer's {tokenizer.VocabularySize} tokens and the {(config.Sequence ? "start and end token" : "blank")} make {tokenizer.VocabularySize + 1}.", nameof(spec));
        }

        var weights = IdrakAudioWeights.Draw(seed, config.InitScale, Device.Cpu);
        using var encoder = new IdrakAudioEncoder(config, weights);
        using var decoder = config.Sequence ? IdrakAudioDecoder.Build(config, weights).Decoder : null;
        var tensors = new List<(string, int[], float[])>(weights.Entries.Count);
        foreach (var (name, tensor, stored, transposed) in weights.Entries)
        {
            tensors.Add((name, stored, IdrakAudioWeights.Stored(tensor, transposed)));
        }

        return new CreatedSpeechModel(config.ToJson(), tensors, new Dictionary<string, string>());
    }

    private static void Task(IdrakAudioConfig config)
    {
        if (config.Task is not (CtcTask or Seq2SeqTask))
        {
            throw new NotSupportedException($"IdrakAudio task '{config.Task}' is not built yet: this plug-in makes and reads \"{CtcTask}\" and \"{Seq2SeqTask}\" models (plan 21, S6).");
        }
    }
}
