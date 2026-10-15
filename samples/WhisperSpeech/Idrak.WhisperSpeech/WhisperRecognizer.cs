// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using Idrak.Audio;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;

namespace Idrak.WhisperSpeech;

/// <summary>
/// A Whisper-style speech recognizer: sound to log-mel features (the library's "log-mel" extractor with the checkpoint's
/// preprocessor_config.json), the encoder once per segment, its output projected once into every cross-attention layer's
/// keys and values, then greedy or beam decoding (<see cref="BeamSearch"/>) from the prompt
/// <c>&lt;|startoftranscript|&gt;&lt;|lang|&gt;&lt;|task|&gt;[&lt;|notimestamps|&gt;]</c> to the end token, with the
/// checkpoint's suppressed tokens and, with timestamps, the family's timestamp rules. Every setting comes from the
/// checkpoint's files or <see cref="TranscriptionOptions"/>.
/// </summary>
public sealed class WhisperRecognizer : ISpeechRecognizer
{
    private readonly CrossAttention[] _crosses;
    private bool _disposed;

    internal WhisperRecognizer(WhisperConfig config, BpeTokenizer tokenizer, IAudioFeatureExtractor features, WhisperEncoder encoder, Sequential decoder, Device device,
        KeyValueFormat cache, IReadOnlyList<string> notes)
    {
        Config = config;
        Tokenizer = tokenizer;
        Features = features;
        Encoder = encoder;
        Decoder = decoder;
        Device = device;
        CacheFormat = cache;
        Notes = notes;
        Languages = config.Languages.Keys.ToHashSet(StringComparer.Ordinal);
        _crosses = [.. decoder.Descendants().OfType<CrossAttention>()];
        encoder.Eval();
        decoder.Eval();
    }

    /// <summary>The checkpoint's settings.</summary>
    public WhisperConfig Config { get; }

    /// <summary>The checkpoint's tokenizer.</summary>
    public BpeTokenizer Tokenizer { get; }

    /// <summary>The feature extractor (preprocessor_config.json).</summary>
    public IAudioFeatureExtractor Features { get; }

    /// <summary>The audio encoder.</summary>
    public WhisperEncoder Encoder { get; }

    /// <summary>The text decoder (embedding, positions, blocks with cross-attention, norm, head).</summary>
    public Sequential Decoder { get; }

    /// <inheritdoc />
    public Device Device { get; }

    /// <summary>The decoder's key/value cache format.</summary>
    public KeyValueFormat CacheFormat { get; }

    /// <summary>What was approximated while loading.</summary>
    public IReadOnlyList<string> Notes { get; }

    /// <inheritdoc />
    public IReadOnlySet<string> Languages { get; }

    /// <summary>
    /// The encoder's output for <paramref name="features"/> ([mels, frames] or [batch, mels, frames], the frames of one
    /// segment): [batch, frames / 2, dim]. The caller owns the result.
    /// </summary>
    public Tensor Encode(Tensor features)
    {
        ArgumentNullException.ThrowIfNull(features);
        ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Inference);
        using var noGrad = Autograd.NoGrad();
        using var scope = new TensorScope();
        var input = features.Device == Device ? features : features.To(Device);
        var encoded = Encoder.Forward(input.Rank == 2 ? input.Reshape(1, input.Shape[0], input.Shape[1]) : input);
        Telemetry.InferenceStage(InferenceStageKind.Encoder, Device, start, encoded.Shape[0], 4L * encoded.Size);
        return TensorScope.Untrack(encoded);
    }

    /// <summary>
    /// The decoder's prompt: &lt;|startoftranscript|&gt;, the language's token, the task's and, without timestamps,
    /// &lt;|notimestamps|&gt;; before them, when <paramref name="previous"/> is given and the checkpoint has a prompt
    /// token, that text's tokens after it (the last half of the positions at most).
    /// </summary>
    /// <exception cref="NotSupportedException">The language or the task has no token in this checkpoint.</exception>
    public IReadOnlyList<int> Prompt(string? language, string task = "transcribe", bool timestamps = false, string? previous = null)
    {
        var prompt = new List<int>();
        if (previous is { Length: > 0 } && Config.PrevToken is { } prev)
        {
            var ids = Tokenizer.Encode(" " + previous.Trim());
            int keep = Math.Max(0, Config.MaxTargetPositions / 2 - 1);
            prompt.Add(prev);
            prompt.AddRange(ids.Count > keep ? ids.Skip(ids.Count - keep) : ids);
        }

        prompt.Add(Config.StartToken);
        if (language is not null)
        {
            prompt.Add(Config.Languages.TryGetValue(language, out int lang) ? lang
                : throw new NotSupportedException($"This checkpoint has no token for the language '{language}' (it knows {string.Join(", ", Config.Languages.Keys.Order())})."));
        }

        if (Config.Tasks.Count > 0 || Config.Languages.Count > 0)
        {
            prompt.Add(Config.Tasks.TryGetValue(task, out int t) ? t
                : throw new NotSupportedException($"This checkpoint has no token for the task '{task}' (it knows {string.Join(", ", Config.Tasks.Keys.Order())})."));
        }

        if (!timestamps)
        {
            prompt.Add(Config.NoTimestampsToken);
        }

        return prompt;
    }

    /// <summary>The logits of the position after <paramref name="prompt"/> over <paramref name="encoded"/> ([1, positions, dim]): [vocabulary].</summary>
    public float[] Logits(Tensor encoded, IReadOnlyList<int> prompt)
    {
        ArgumentNullException.ThrowIfNull(encoded);
        ArgumentNullException.ThrowIfNull(prompt);
        ThrowIfDisposed();
        using var noGrad = Autograd.NoGrad();
        using var context = new DecodingContext(Device, 1, Config.MaxTargetPositions, CacheFormat) { LastPositionOnly = true };
        foreach (var cross in _crosses)
        {
            cross.SetMemory(encoded, context);
        }

        using var scope = new TensorScope();
        using var ids = Tensor.From([.. prompt.Select(p => (float)p)], [1, prompt.Count], Device);
        return Decoder.ForwardCached(ids, context).ToArray();
    }

    /// <summary>
    /// The language the model hears in <paramref name="encoded"/>: one decoder step after &lt;|startoftranscript|&gt;, the
    /// most likely of the language tokens; null when the checkpoint has none.
    /// </summary>
    public string? DetectLanguage(Tensor encoded)
    {
        if (Config.Languages.Count == 0)
        {
            return null;
        }

        var logits = Logits(encoded, [Config.StartToken]);
        string? best = null;
        float bestLogit = float.NegativeInfinity;
        foreach (var (code, id) in Config.Languages)
        {
            if ((uint)id < (uint)logits.Length && logits[id] > bestLogit)
            {
                (best, bestLogit) = (code, logits[id]);
            }
        }

        return best;
    }

    /// <summary>
    /// Decodes one segment from <paramref name="prompt"/> over <paramref name="encoded"/> ([1, positions, dim]): the
    /// hypotheses best first (greedy for one beam), each with the generated tokens.
    /// </summary>
    public IReadOnlyList<BeamHypothesis> Decode(Tensor encoded, IReadOnlyList<int> prompt, TranscriptionOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(encoded);
        ArgumentNullException.ThrowIfNull(prompt);
        ThrowIfDisposed();
        options ??= new TranscriptionOptions();
        if (options.Temperature > 0f)
        {
            throw new NotSupportedException("Sampling (Temperature above 0) is not in this family yet: set Temperature 0 for greedy or beam decoding.");
        }

        int beams = Math.Max(1, options.Beams);
        int room = Config.MaxTargetPositions - prompt.Count;
        if (room <= 0)
        {
            throw new ArgumentException($"A prompt of {prompt.Count} tokens leaves no room in {Config.MaxTargetPositions} positions.", nameof(prompt));
        }

        var beamOptions = new BeamOptions
        {
            Beams = beams,
            LengthPenalty = options.LengthPenalty,
            MaxTokens = options.MaxTokens > 0 ? Math.Min(options.MaxTokens, room) : room,
            MinTokens = options.MinTokens,
            EndToken = Config.EndToken,
            SuppressTokens = options.SuppressTokens ? Config.SuppressTokens : [],
            BeginSuppressTokens = options.SuppressTokens ? Config.BeginSuppressTokens : [],
        };
        bool timestamps = !prompt.Contains(Config.NoTimestampsToken);
        LogitsFilter? filter = timestamps && options.TimestampRules ? TimestampRules : null;
        using var noGrad = Autograd.NoGrad();
        using var context = new DecodingContext(Device, beams, Config.MaxTargetPositions, CacheFormat) { LastPositionOnly = true, Cancellation = cancellationToken };
        foreach (var cross in _crosses)
        {
            cross.SetMemory(encoded, context);                               // the encoder's output projected once per segment
        }

        return BeamSearch.Decode((ids, ctx) => Decoder.ForwardCached(ids, ctx), context, prompt, beamOptions, filter, cancellationToken);
    }

    // transformers' WhisperTimeStampLogitsProcessor: a timestamp opens and closes each phrase (after a lone timestamp
    // comes text, after a pair a timestamp or the end), timestamps never go back, the first one lies within the
    // allowed start, and when the timestamps together are likelier than any text token a timestamp is chosen.
    private void TimestampRules(int step, ReadOnlySpan<int> generated, Span<float> logits)
    {
        int begin = Config.TimestampBegin, eos = Config.EndToken;
        if (begin >= logits.Length)
        {
            return;
        }

        logits[Config.NoTimestampsToken] = float.NegativeInfinity;
        if (generated.Length > 0)
        {
            bool lastTimestamp = generated[^1] >= begin;
            bool penultimateTimestamp = generated.Length < 2 || generated[^2] >= begin;
            if (lastTimestamp)
            {
                if (penultimateTimestamp)
                {
                    logits[begin..].Fill(float.NegativeInfinity);           // a phrase's start: text next
                }
                else if ((uint)eos < (uint)logits.Length)
                {
                    logits[..eos].Fill(float.NegativeInfinity);             // a phrase's end: a timestamp or the end next
                }
            }

            int last = -1;
            foreach (int token in generated)
            {
                if (token >= begin)
                {
                    last = token;
                }
            }

            if (last >= 0)
            {
                int from = lastTimestamp && !penultimateTimestamp ? last : last + 1;
                if (from > begin)
                {
                    logits[begin..Math.Min(from, logits.Length)].Fill(float.NegativeInfinity);   // never back in time
                }
            }
        }

        if (step == 0)
        {
            logits[..begin].Fill(float.NegativeInfinity);                    // the first token is a timestamp
            if (Config.MaxInitialTimestampIndex is { } initial && begin + initial + 1 < logits.Length)
            {
                logits[(begin + initial + 1)..].Fill(float.NegativeInfinity);
            }
        }

        // The timestamps' probability against the best text token's: log-sum-exp over the timestamps, in double.
        float max = float.NegativeInfinity;
        foreach (float x in logits)
        {
            max = Math.Max(max, x);
        }

        if (float.IsNegativeInfinity(max))
        {
            return;
        }

        double sum = 0, timestampSum = 0;
        float bestText = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
        {
            float x = logits[i];
            if (float.IsNegativeInfinity(x))
            {
                continue;
            }

            double e = Math.Exp(x - max);
            sum += e;
            if (i >= begin)
            {
                timestampSum += e;
            }
            else
            {
                bestText = Math.Max(bestText, x);
            }
        }

        if (timestampSum > 0 && (float.IsNegativeInfinity(bestText) || Math.Log(timestampSum) > bestText - max))
        {
            logits[..begin].Fill(float.NegativeInfinity);
        }
    }

    /// <inheritdoc />
    public Transcription Transcribe(AudioData audio, TranscriptionOptions? options = null, IProgress<TranscriptionProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        ThrowIfDisposed();
        options ??= new TranscriptionOptions();
        var total = Stopwatch.StartNew();
        ComputeResources.ResetPeakMemoryUsage(Device);
        int rate = Features.Options.SampleRate;
        var sound = Mono(Resampler.Resample(audio, rate));
        var decodeTime = total.Elapsed;

        double segmentSeconds = Features.Options.PadSeconds ?? sound.Duration.TotalSeconds;
        double chunkSeconds = options.ChunkSeconds > 0 ? Math.Min(options.ChunkSeconds, segmentSeconds) : segmentSeconds;
        double stepSeconds = options.ChunkSeconds > 0 ? Math.Max(chunkSeconds - Math.Max(0, options.OverlapSeconds), chunkSeconds / 2) : chunkSeconds;
        int chunkSamples = Math.Max(1, (int)Math.Round(chunkSeconds * rate)), stepSamples = Math.Max(1, (int)Math.Round(stepSeconds * rate));
        int count = sound.Length == 0 ? 1 : 1 + Math.Max(0, (sound.Length - chunkSamples + stepSamples - 1) / stepSamples);
        var segments = new List<TranscriptSegment>();
        TimeSpan features = TimeSpan.Zero, encoder = TimeSpan.Zero, decoder = TimeSpan.Zero;
        string? language = options.Language, carried = options.Prompt;
        for (int i = 0; i < count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int from = i * stepSamples;
            int length = Math.Min(chunkSamples, sound.Length - from);
            if (length <= 0)
            {
                break;
            }

            double offset = (double)from / rate;
            progress?.Report(new TranscriptionProgress(i, count, "features", total.Elapsed));
            var start = total.Elapsed;
            using var noGrad = Autograd.NoGrad();
            using var scope = new TensorScope();
            var extracted = Features.Extract(sound.Slice(from, length), Device);     // [frames, mels]
            var input = extracted.Transpose();                                         // [mels, frames]
            features += total.Elapsed - start;
            progress?.Report(new TranscriptionProgress(i, count, "encoder", total.Elapsed));
            start = total.Elapsed;
            using var encoded = Encode(input);
            encoder += total.Elapsed - start;
            progress?.Report(new TranscriptionProgress(i, count, "decoder", total.Elapsed));
            start = total.Elapsed;
            language ??= DetectLanguage(encoded);
            var prompt = Prompt(language, options.Task, options.Timestamps, options.CarryText ? carried : options.Prompt);
            var best = Decode(encoded, prompt, options, cancellationToken).FirstOrDefault();
            decoder += total.Elapsed - start;
            if (best is null)
            {
                continue;
            }

            var found = Segments(best, offset, (double)length / rate, options.Timestamps);
            segments.AddRange(found);
            if (found.Count > 0)
            {
                carried = string.Join(" ", found.Select(s => s.Text)).Trim();
            }
        }

        progress?.Report(new TranscriptionProgress(count, count, "done", total.Elapsed));
        string text = string.Join(" ", segments.Select(s => s.Text.Trim()).Where(t => t.Length > 0));
        long peak = ComputeResources.GetMemoryUsage(Device).Peak;
        return new Transcription(text, language, segments, new TranscriptionStats(sound.Duration, decodeTime, features, encoder, decoder, peak, count, 1));
    }

    // The segments of a hypothesis: with timestamps asked for, timestamp tokens open and close phrases (times from the
    // segment's offset); without them, or without any timestamp token, one segment over the whole chunk (a timestamp token
    // the model gives anyway is not a time then). Special tokens are not part of the text.
    private List<TranscriptSegment> Segments(BeamHypothesis best, double offset, double seconds, bool timestamps)
    {
        var tokens = best.Tokens;
        float average = tokens.Count == 0 ? 0f : best.LogProb / tokens.Count;
        var result = new List<TranscriptSegment>();
        var text = new List<int>();
        var piece = new List<int>();
        double? start = null;
        int begin = Config.TimestampBegin;
        foreach (int token in tokens)
        {
            if (timestamps && token >= begin)
            {
                double time = offset + (token - begin) * Config.TimePrecision;
                if (start is { } s && text.Count > 0)
                {
                    result.Add(new TranscriptSegment(s, time, Decode(text), average, [.. piece]));
                    text.Clear();
                    piece.Clear();
                }

                start = time;
                piece.Add(token);
            }
            else
            {
                piece.Add(token);
                if (!Tokenizer.IsSpecial(token) && token < Config.EndToken)
                {
                    text.Add(token);
                }
            }
        }

        if (text.Count > 0)
        {
            result.Add(new TranscriptSegment(start ?? offset, offset + seconds, Decode(text), average, [.. piece]));
        }

        return result;
    }

    private string Decode(List<int> ids) => Tokenizer.Decode(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(ids)).Trim();

    private static AudioData Mono(AudioData audio)
    {
        if (audio.Channels == 1)
        {
            return audio;
        }

        var samples = GC.AllocateUninitializedArray<float>(audio.Length);
        audio.ToMono(samples);
        return new AudioData(samples, 1, audio.SampleRate);
    }

    /// <inheritdoc />
    public async IAsyncEnumerable<TranscriptSegment> TranscribeStream(IAsyncEnumerable<AudioData> chunks, TranscriptionOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(chunks);
        ThrowIfDisposed();
        options ??= new TranscriptionOptions();
        int rate = Features.Options.SampleRate;
        double chunkSeconds = options.ChunkSeconds > 0 ? options.ChunkSeconds : Features.Options.PadSeconds ?? 30;
        int chunkSamples = Math.Max(1, (int)Math.Round(chunkSeconds * rate));
        var pending = new List<float>();
        double offset = 0;
        string? carried = options.Prompt;
        var each = options with { ChunkSeconds = 0 };
        await foreach (var chunk in chunks.WithCancellation(cancellationToken).ConfigureAwait(false))
        {
            var mono = Mono(Resampler.Resample(chunk, rate));
            pending.AddRange(mono.Channel(0));
            while (pending.Count >= chunkSamples)
            {
                foreach (var segment in await Next(chunkSamples).ConfigureAwait(false))
                {
                    yield return segment;
                }
            }
        }

        if (pending.Count > 0)
        {
            foreach (var segment in await Next(pending.Count).ConfigureAwait(false))
            {
                yield return segment;
            }
        }

        async Task<List<TranscriptSegment>> Next(int samples)
        {
            var piece = new AudioData([.. pending.Take(samples)], 1, rate);
            pending.RemoveRange(0, samples);
            double at = offset;
            offset += (double)samples / rate;
            var request = each with { Prompt = options.CarryText ? carried : options.Prompt };
            var result = await Task.Run(() => Transcribe(piece, request, null, cancellationToken), CancellationToken.None).ConfigureAwait(false);
            if (result.Text.Length > 0)
            {
                carried = result.Text;
            }

            return [.. result.Segments.Select(s => s with { Start = s.Start + at, End = s.End + at })];
        }
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        Encoder.Dispose();
        Decoder.Dispose();
    }
}
