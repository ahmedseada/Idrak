// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json;
using Idrak.Audio;
using Idrak.Models;
using Idrak.Abstraction.Generation;

namespace Idrak.Audio.Plugins.IdrakAudio;

/// <summary>
/// An IdrakAudio "ctc" model: the encoder and head (<see cref="IdrakAudioEncoder"/>), its tokenizer and its features, as a
/// speech recognizer (sound in, text out through the library's <see cref="CtcDecoders"/>: greedy, or "beam" when
/// <see cref="TranscriptionOptions.Beams"/> is above 1) and as a model <c>SpeechTrainer</c> trains and saves.
/// </summary>
public sealed class IdrakAudioCtcModel : ICtcSpeechModel
{
    private readonly IReadOnlyList<(string Name, Tensor Tensor, int[] Stored, bool Transposed)> _layout;
    private readonly IdrakAudioTranscriber _transcriber;

    internal IdrakAudioCtcModel(IdrakAudioConfig config, IdrakAudioWeights weights, IPretrainedTokenizer tokenizer, Device device)
    {
        Config = config;
        Device = device;
        Tokenizer = tokenizer;
        Encoder = new IdrakAudioEncoder(config, weights);
        _layout = [.. weights.Entries];
        Features = AudioFeatureExtractors.Create(config.FeatureExtractor, config.Features);
        _transcriber = new IdrakAudioTranscriber(config, Features, device, Read, labels => Text(labels));
        if (tokenizer.VocabularySize + 1 != config.Classes)
        {
            Encoder.Dispose();
            throw new InvalidDataException($"The IdrakAudio head has {config.Classes} classes; the tokenizer's {tokenizer.VocabularySize} tokens and the blank make {tokenizer.VocabularySize + 1}.");
        }
    }

    /// <summary>The settings.</summary>
    public IdrakAudioConfig Config { get; }

    /// <summary>The encoder and head.</summary>
    public IdrakAudioEncoder Encoder { get; }

    /// <summary>The tokenizer: token i is class i below the blank, i + 1 from it on.</summary>
    public IPretrainedTokenizer Tokenizer { get; }

    /// <inheritdoc />
    public string Architecture => IdrakAudioFamily.Architecture;

    /// <inheritdoc />
    public Module Network => Encoder;

    /// <inheritdoc />
    public IAudioFeatureExtractor Features { get; }

    /// <inheritdoc />
    public int Classes => Config.Classes;

    /// <inheritdoc />
    public int Blank => Config.BlankId;

    /// <inheritdoc />
    public Device Device { get; }

    /// <inheritdoc />
    public IReadOnlySet<string> Languages { get; } = new HashSet<string>();

    /// <inheritdoc />
    public int StepsFor(int frames) => Config.StepsFor(frames);

    /// <inheritdoc />
    public int[] Labels(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var ids = Tokenizer.Encode(text);
        var labels = new int[ids.Count];
        for (int i = 0; i < labels.Length; i++)
        {
            labels[i] = ids[i] < Blank ? ids[i] : ids[i] + 1;
        }

        return labels;
    }

    /// <inheritdoc />
    public string Text(ReadOnlySpan<int> labels)
    {
        var ids = new int[labels.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = labels[i] < Blank ? labels[i] : labels[i] - 1;
        }

        return Tokenizer.Decode(ids);
    }

    /// <inheritdoc />
    public Tensor Logits(Tensor features, IReadOnlyList<int> frames, SpeechForwardOptions? options = null) =>
        Encoder.Run(features, frames, options?.Checkpointing ?? false);

    /// <inheritdoc />
    public void Save(string folder)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(folder);
        Directory.CreateDirectory(folder);
        File.WriteAllText(Path.Combine(folder, "config.json"), Config.ToJson().ToJsonString(new JsonSerializerOptions { WriteIndented = true }) + "\n");
        SafeTensorsWriter.Write(Path.Combine(folder, "model.safetensors"), Export(), SafeTensorType.F32);
        Tokenizer.Save(Path.Combine(folder, "tokenizer.json"));
    }

    /// <summary>Each parameter (the encoder's, in the order they were made) with its checkpoint name and whether it is stored transposed.</summary>
    internal IEnumerable<(string Name, Tensor Parameter, bool Transposed)> NamedParameters()
    {
        var parameters = Encoder.Parameters().ToList();
        if (parameters.Count != _layout.Count)
        {
            throw new InvalidOperationException($"The encoder holds {parameters.Count} parameters; {_layout.Count} were read.");
        }

        for (int i = 0; i < parameters.Count; i++)
        {
            yield return (_layout[i].Name, parameters[i], _layout[i].Transposed);
        }
    }

    // The parameters as they are now, under their checkpoint names and layouts.
    internal IEnumerable<(string Name, int[] Shape, float[] Values)> Export() =>
        NamedParameters().Select((p, i) => (p.Name, _layout[i].Stored, IdrakAudioWeights.Stored(p.Parameter, p.Transposed)));

    /// <inheritdoc />
    public Transcription Transcribe(AudioData audio, TranscriptionOptions? options = null, IProgress<TranscriptionProgress>? progress = null, CancellationToken cancellationToken = default) =>
        _transcriber.Transcribe(audio, options, progress, cancellationToken);

    /// <inheritdoc />
    public IAsyncEnumerable<TranscriptSegment> TranscribeStream(IAsyncEnumerable<AudioData> chunks, TranscriptionOptions? options = null, CancellationToken cancellationToken = default) =>
        _transcriber.TranscribeStream(chunks, options, cancellationToken);

    // One stretch's features read: the encoder, a log-softmax and the CTC decoder (greedy, or "beam" above one beam).
    private (int[] Labels, float Score) Read(Tensor features, TranscriptionOptions options, ref TimeSpan encoder, ref TimeSpan decoder)
    {
        var clock = Stopwatch.StartNew();
        int frames = features.Shape[1];
        var logProbs = Encoder.Run(features, null, checkpointing: false).LogSoftmax();
        int steps = StepsFor(frames);
        var read = logProbs.ToArray();
        encoder += clock.Elapsed;
        clock.Restart();
        var decodeOptions = new CtcDecodeOptions { Blank = Blank, BeamWidth = Math.Max(1, options.Beams) };
        var best = CtcDecoders.Decode(read.AsMemory(0, steps * Classes), steps, Classes, options.Beams > 1 ? CtcDecoders.Beam : CtcDecoders.Greedy, decodeOptions)[0];
        decoder += clock.Elapsed;
        return (best.Labels, best.LogProbability / Math.Max(1, steps));
    }

    /// <inheritdoc />
    public void Dispose() => Encoder.Dispose();

    /// <inheritdoc />
    public override string ToString() => $"IdrakAudio ctc ({Config.Dim} wide, {Config.Layers} blocks, {Config.Classes} classes) on {Device}";
}
