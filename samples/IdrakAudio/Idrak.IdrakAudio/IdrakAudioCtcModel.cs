// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Idrak.Audio;
using Idrak.Models;
using Idrak.Abstraction.Generation;

namespace Idrak.IdrakAudio;

/// <summary>
/// An IdrakAudio "ctc" model: the encoder and head (<see cref="IdrakAudioEncoder"/>), its tokenizer and its features, as a
/// speech recognizer (sound in, text out through the library's <see cref="CtcDecoders"/>: greedy, or "beam" when
/// <see cref="TranscriptionOptions.Beams"/> is above 1) and as a model <c>SpeechTrainer</c> trains and saves.
/// </summary>
public sealed class IdrakAudioCtcModel : ICtcSpeechModel
{
    private readonly IReadOnlyList<(string Name, Tensor Tensor, int[] Stored, bool Transposed)> _layout;

    internal IdrakAudioCtcModel(IdrakAudioConfig config, IdrakAudioWeights weights, BpeTokenizer tokenizer, Device device)
    {
        Config = config;
        Device = device;
        Tokenizer = tokenizer;
        Encoder = new IdrakAudioEncoder(config, weights);
        _layout = [.. weights.Entries];
        Features = AudioFeatureExtractors.Create(config.FeatureExtractor, config.Features);
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
    public BpeTokenizer Tokenizer { get; }

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
    public Transcription Transcribe(AudioData audio, TranscriptionOptions? options = null, IProgress<TranscriptionProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        options ??= new TranscriptionOptions();
        var total = Stopwatch.StartNew();
        var clock = Stopwatch.StartNew();
        var sound = Prepare(audio);
        var decode = clock.Elapsed;
        int chunk = ChunkSamples(options);
        int count = (sound.Length + chunk - 1) / chunk;
        var segments = new List<TranscriptSegment>(count);
        TimeSpan features = default, encoder = default, decoder = default;
        for (int c = 0; c < count; c++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int start = c * chunk, length = Math.Min(chunk, sound.Length - start);
            segments.Add(Piece(sound.Channel(0).Slice(start, length), start, options, ref features, ref encoder, ref decoder));
            progress?.Report(new TranscriptionProgress(c + 1, count, "decoder", total.Elapsed));
        }

        string text = string.Join(" ", segments.Select(s => s.Text).Where(t => t.Length > 0));
        var stats = new TranscriptionStats(audio.Duration, decode, features, encoder, decoder, Device.Backend.GetMemoryUsage().Peak, count, 1);
        return new Transcription(text, null, segments, stats);
    }

    /// <inheritdoc />
    public async IAsyncEnumerable<TranscriptSegment> TranscribeStream(IAsyncEnumerable<AudioData> chunks, TranscriptionOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(chunks);
        options ??= new TranscriptionOptions();
        int size = ChunkSamples(options);
        var pending = new List<float>(size);
        long offset = 0;
        TimeSpan features = default, encoder = default, decoder = default;
        await foreach (var piece in chunks.WithCancellation(cancellationToken))
        {
            pending.AddRange(Prepare(piece).Channel(0));
            while (pending.Count >= size)
            {
                var samples = pending.GetRange(0, size).ToArray();
                pending.RemoveRange(0, size);
                yield return Piece(samples, offset, options, ref features, ref encoder, ref decoder);
                offset += size;
            }
        }

        if (pending.Count > 0)
        {
            yield return Piece(pending.ToArray(), offset, options, ref features, ref encoder, ref decoder);
        }
    }

    private int ChunkSamples(TranscriptionOptions options) =>
        Math.Max(1, (int)Math.Round((options.ChunkSeconds > 0 ? options.ChunkSeconds : Config.MaxSeconds) * Features.Options.SampleRate));

    // One channel at the features' rate.
    private AudioData Prepare(AudioData audio)
    {
        if (audio.Channels > 1)
        {
            var mono = new float[audio.Length];
            audio.ToMono(mono);
            audio = new AudioData(mono, 1, audio.SampleRate);
        }

        return Resampler.Resample(audio, Features.Options.SampleRate);
    }

    // One stretch of sound transcribed: its features (a stretch shorter than a window padded with silence to one), the
    // encoder, a log-softmax and the CTC decoder; the segment timed from `start` samples.
    private TranscriptSegment Piece(ReadOnlySpan<float> samples, long start, TranscriptionOptions options, ref TimeSpan features, ref TimeSpan encoder, ref TimeSpan decoder)
    {
        int rate = Features.Options.SampleRate;
        var padded = new float[Math.Max(samples.Length, Features.Options.WindowSize)];
        samples.CopyTo(padded);
        var clock = Stopwatch.StartNew();
        using var scope = new TensorScope();
        using var _ = Autograd.NoGrad();
        var values = Features.Extract(new AudioData(padded, 1, rate), Device);
        int frames = values.Shape[0];
        features += clock.Elapsed;
        clock.Restart();
        var logProbs = Encoder.Run(values.Reshape(1, frames, Config.Mels), null, checkpointing: false).LogSoftmax();
        int steps = StepsFor(frames);
        var read = logProbs.ToArray();
        encoder += clock.Elapsed;
        clock.Restart();
        var decodeOptions = new CtcDecodeOptions { Blank = Blank, BeamWidth = Math.Max(1, options.Beams) };
        var best = CtcDecoders.Decode(read.AsMemory(0, steps * Classes), steps, Classes, options.Beams > 1 ? CtcDecoders.Beam : CtcDecoders.Greedy, decodeOptions)[0];
        string text = Text(best.Labels);
        decoder += clock.Elapsed;
        return new TranscriptSegment(start / (double)rate, (start + samples.Length) / (double)rate, text, best.LogProbability / Math.Max(1, steps), best.Labels);
    }

    /// <inheritdoc />
    public void Dispose() => Encoder.Dispose();

    /// <inheritdoc />
    public override string ToString() => $"IdrakAudio ctc ({Config.Dim} wide, {Config.Layers} blocks, {Config.Classes} classes) on {Device}";
}
