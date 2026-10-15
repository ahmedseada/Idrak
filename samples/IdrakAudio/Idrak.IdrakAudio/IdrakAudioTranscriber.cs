// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Runtime.CompilerServices;
using Idrak.Audio;

namespace Idrak.IdrakAudio;

/// <summary>
/// How an IdrakAudio model transcribes, whatever its task: the sound mixed to one channel at the features' rate, cut in
/// stretches of the options' chunk length (the config's <see cref="IdrakAudioConfig.MaxSeconds"/> when they give none), each
/// stretch's features (one shorter than a window padded with silence to one) read by the task (<see cref="ReadPiece"/>:
/// its labels and their mean log-probability) and timed from its first sample.
/// </summary>
internal sealed class IdrakAudioTranscriber(IdrakAudioConfig config, IAudioFeatureExtractor features, Device device, IdrakAudioTranscriber.ReadPiece read, Func<int[], string> text)
{
    /// <summary>One stretch's features [1, frames, mels] read to labels; the encoder's and the decoder's time added to <paramref name="encoder"/> and <paramref name="decoder"/>.</summary>
    public delegate (int[] Labels, float Score) ReadPiece(Tensor features, TranscriptionOptions options, ref TimeSpan encoder, ref TimeSpan decoder);

    public Transcription Transcribe(AudioData audio, TranscriptionOptions? options, IProgress<TranscriptionProgress>? progress, CancellationToken cancellationToken)
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
        TimeSpan featureTime = default, encoder = default, decoder = default;
        for (int c = 0; c < count; c++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int start = c * chunk, length = Math.Min(chunk, sound.Length - start);
            segments.Add(Piece(sound.Channel(0).Slice(start, length), start, options, ref featureTime, ref encoder, ref decoder));
            progress?.Report(new TranscriptionProgress(c + 1, count, "decoder", total.Elapsed));
        }

        string joined = string.Join(" ", segments.Select(s => s.Text).Where(t => t.Length > 0));
        var stats = new TranscriptionStats(audio.Duration, decode, featureTime, encoder, decoder, device.Backend.GetMemoryUsage().Peak, count, 1);
        return new Transcription(joined, null, segments, stats);
    }

    public async IAsyncEnumerable<TranscriptSegment> TranscribeStream(IAsyncEnumerable<AudioData> chunks, TranscriptionOptions? options,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(chunks);
        options ??= new TranscriptionOptions();
        int size = ChunkSamples(options);
        var pending = new List<float>(size);
        long offset = 0;
        TimeSpan featureTime = default, encoder = default, decoder = default;
        await foreach (var piece in chunks.WithCancellation(cancellationToken))
        {
            pending.AddRange(Prepare(piece).Channel(0));
            while (pending.Count >= size)
            {
                var samples = pending.GetRange(0, size).ToArray();
                pending.RemoveRange(0, size);
                yield return Piece(samples, offset, options, ref featureTime, ref encoder, ref decoder);
                offset += size;
            }
        }

        if (pending.Count > 0)
        {
            yield return Piece(pending.ToArray(), offset, options, ref featureTime, ref encoder, ref decoder);
        }
    }

    private int ChunkSamples(TranscriptionOptions options) =>
        Math.Max(1, (int)Math.Round((options.ChunkSeconds > 0 ? options.ChunkSeconds : config.MaxSeconds) * features.Options.SampleRate));

    // One channel at the features' rate.
    private AudioData Prepare(AudioData audio)
    {
        if (audio.Channels > 1)
        {
            var mono = new float[audio.Length];
            audio.ToMono(mono);
            audio = new AudioData(mono, 1, audio.SampleRate);
        }

        return Resampler.Resample(audio, features.Options.SampleRate);
    }

    // One stretch of sound transcribed and timed from `start` samples.
    private TranscriptSegment Piece(ReadOnlySpan<float> samples, long start, TranscriptionOptions options, ref TimeSpan featureTime, ref TimeSpan encoder, ref TimeSpan decoder)
    {
        int rate = features.Options.SampleRate;
        var padded = new float[Math.Max(samples.Length, features.Options.WindowSize)];
        samples.CopyTo(padded);
        var clock = Stopwatch.StartNew();
        using var scope = new TensorScope();
        using var _ = Autograd.NoGrad();
        var values = features.Extract(new AudioData(padded, 1, rate), device);
        int frames = values.Shape[0];
        featureTime += clock.Elapsed;
        var (labels, score) = read(values.Reshape(1, frames, config.Mels), options, ref encoder, ref decoder);
        return new TranscriptSegment(start / (double)rate, (start + samples.Length) / (double)rate, text(labels), score, labels);
    }
}
