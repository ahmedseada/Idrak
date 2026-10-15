// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using Idrak.Nlp.Abstractions;

namespace Idrak.Audio;

/// <summary>
/// The audio parts of a conversation as soft tokens (<see cref="ISoftPartSource"/> for <see cref="ChatParts.Audio"/>),
/// over a family's audio part (<see cref="PretrainedAudio"/>) and its encoder: each <see cref="ChatAudio"/> is decoded
/// by the registered codecs, its blocks of soft tokens asked of the encoder once per sound (kept by the sound's hash),
/// and its features encoded once per sound, options and encoder, kept in a feature cache (<see cref="IFeatureCache"/>,
/// keyed by the sound's hash and the extractor's options) so a second turn or epoch encodes nothing (rule 75). Give it
/// to <c>ChatGenerator.Parts</c> or <c>ChatTranscriptEncoder.Parts</c>. When the family's audio part trains
/// (<see cref="IAudioTuningPart"/>) it is a trainable source too (<see cref="ITrainableSoftParts"/>): the training step
/// reads its features with gradients, and while a part trains no features are cached (they change every step).
/// </summary>
public sealed class ChatAudioParts : ITrainableSoftParts
{
    private readonly ConcurrentDictionary<string, IReadOnlyList<ImageTokenLayout>> _blocks = new(StringComparer.Ordinal);
    private readonly Lock _family = new();                       // the family's Blocks and Encode, one call at a time
    private readonly bool _ownsCache, _ownsEncoder;
    private readonly string _options;

    /// <summary>
    /// A source over <paramref name="audio"/> with <paramref name="encoder"/> (the family's, on the model's device).
    /// </summary>
    /// <param name="audio">The model's audio part.</param>
    /// <param name="encoder">Its encoder (built with <see cref="PretrainedAudio.CreateEncoder"/> when null, and disposed with this).</param>
    /// <param name="cache">Where features are kept (not disposed by this); a new "memory" one when null (disposed with this).</param>
    /// <param name="checkpoint">A text that changes when the checkpoint's weights do, for the cache's keys; the model folder's name when null.</param>
    public ChatAudioParts(PretrainedAudio audio, IAudioLanguageEncoder? encoder = null, IFeatureCache? cache = null, string? checkpoint = null)
    {
        Audio = audio ?? throw new ArgumentNullException(nameof(audio));
        _ownsEncoder = encoder is null;
        Encoder = encoder ?? audio.CreateEncoder();
        _ownsCache = cache is null;
        Cache = cache ?? FeatureCaches.Create(FeatureCaches.Memory);
        Checkpoint = checkpoint ?? audio.Family;
        _options = audio.Features.Options.ToJson().ToJsonString();
    }

    // A source over a model's part (the "audio" model part's Open): its encoder built with `options`, and the cache
    // owned when `ownsCache` (both disposed with this).
    internal ChatAudioParts(PretrainedAudio audio, AudioEncoderOptions options, IFeatureCache? cache, bool ownsCache)
    {
        Audio = audio ?? throw new ArgumentNullException(nameof(audio));
        _ownsEncoder = true;
        Encoder = audio.CreateEncoder(options);
        _ownsCache = cache is null || ownsCache;
        Cache = cache ?? FeatureCaches.Create(FeatureCaches.Memory);
        Checkpoint = audio.Family;
        _options = audio.Features.Options.ToJson().ToJsonString();
    }

    /// <summary>The model's audio part.</summary>
    public PretrainedAudio Audio { get; }

    /// <summary>The family's encoder.</summary>
    public IAudioLanguageEncoder Encoder { get; }

    /// <summary>Where features are kept between calls.</summary>
    public IFeatureCache Cache { get; }

    /// <summary>What identifies the checkpoint in the cache's keys.</summary>
    public string Checkpoint { get; }

    /// <summary>Sounds whose features the cache gave back without encoding.</summary>
    public long CacheHits => Interlocked.Read(ref _hits);

    /// <summary>Sounds the encoder encoded.</summary>
    public long Encoded => Interlocked.Read(ref _encoded);

    private long _hits, _encoded;

    /// <inheritdoc />
    public string Kind => ChatParts.Audio;

    /// <inheritdoc />
    public IReadOnlyList<string> TrainableParts => (Audio as IAudioTuningPart)?.TrainableParts ?? [];

    /// <inheritdoc />
    public IReadOnlyList<Tensor> Parameters { get; private set; } = [];

    /// <inheritdoc />
    public IReadOnlyCollection<string> Trained { get; private set; } = [];

    /// <inheritdoc />
    public IReadOnlyList<Tensor> Train(IReadOnlyCollection<string> parts)
    {
        ArgumentNullException.ThrowIfNull(parts);
        if (Audio is not IAudioTuningPart tuning)
        {
            return parts.Count == 0 ? [] : throw new ArgumentException($"Audio family '{Audio.Family}' trains no parts (its audio part is not an IAudioTuningPart); asked for {string.Join(", ", parts)}.", nameof(parts));
        }

        lock (_family)
        {
            Parameters = tuning.Train(Encoder, parts);
            Trained = [.. parts];
        }

        return Parameters;
    }

    /// <inheritdoc />
    public void SetTraining(bool training)
    {
        if (Audio is IAudioTuningPart tuning)
        {
            tuning.SetTraining(Encoder, training);
        }
    }

    /// <inheritdoc />
    public IReadOnlyList<ImageFeatures> EncodeTraining(IReadOnlyList<ChatPart> parts, bool checkpointing)
    {
        ArgumentNullException.ThrowIfNull(parts);
        if (Audio is not IAudioTuningPart tuning)
        {
            return Encode(parts);
        }

        var decoded = new List<AudioData>(parts.Count);
        foreach (var part in parts)
        {
            decoded.Add(Read(Of(part)));
        }

        long start = Telemetry.Start(TelemetryLevel.Inference);
        IReadOnlyList<ImageFeatures> encoded;
        lock (_family)
        {
            encoded = tuning.EncodeTraining(Encoder, decoded, checkpointing);
        }

        Interlocked.Add(ref _encoded, parts.Count);
        Telemetry.InferenceStage(InferenceStageKind.Encoder, Encoder.Device, start, parts.Count, 0);
        return encoded;
    }

    /// <inheritdoc />
    public IReadOnlyList<(string Name, int[] Shape, float[] Values)> Export(IReadOnlyCollection<string> parts)
    {
        ArgumentNullException.ThrowIfNull(parts);
        if (parts.Count == 0)
        {
            return [];
        }

        var tuning = Audio as IAudioTuningPart ?? throw new InvalidOperationException($"Audio family '{Audio.Family}' trains no parts.");
        lock (_family)
        {
            return tuning.Export(Encoder, parts);
        }
    }

    /// <inheritdoc />
    public int Width => Encoder.Width;

    /// <inheritdoc />
    public IImagePromptFormat Format => Audio.PromptFormat;

    /// <inheritdoc />
    public IImageAttentionRule Attention => Audio.Attention;

    /// <summary>The cache key of <paramref name="audio"/>'s features under this encoder and the extractor's options.</summary>
    public FeatureCacheKey Key(ChatAudio audio)
    {
        ArgumentNullException.ThrowIfNull(audio);
        return new FeatureCacheKey(audio.Hash, "", _options, Audio.Family, Checkpoint, "float32", FeatureCacheKey.FeaturesStage);
    }

    /// <summary>The sound decoded (<see cref="ChatAudio.Decode"/>), with the decode stage in the telemetry.</summary>
    public static AudioData Read(ChatAudio audio)
    {
        ArgumentNullException.ThrowIfNull(audio);
        long start = Telemetry.Start(TelemetryLevel.Inference);
        var decoded = audio.Decode();
        Telemetry.InferenceStage(InferenceStageKind.AudioDecode, null, start, (long)decoded.Channels * decoded.Length, audio.Data.Length);
        return decoded;
    }

    /// <inheritdoc />
    public IReadOnlyList<ImageTokenLayout> Blocks(ChatPart part)
    {
        var audio = Of(part);
        return _blocks.GetOrAdd(audio.Hash, _ =>
        {
            var decoded = Read(audio);
            lock (_family)
            {
                return Encoder.Blocks(decoded);
            }
        });
    }

    /// <inheritdoc />
    public IReadOnlyList<ImageFeatures> Encode(IReadOnlyList<ChatPart> parts)
    {
        ArgumentNullException.ThrowIfNull(parts);
        var audios = new ChatAudio[parts.Count];
        for (int i = 0; i < parts.Count; i++)
        {
            audios[i] = Of(parts[i]);
        }

        // Cache hits first (no decoding), then every miss decoded and encoded in one call, in order.
        var features = new IReadOnlyList<ImageFeatures>?[parts.Count];
        var misses = new List<int>();
        for (int i = 0; i < audios.Length; i++)
        {
            var blocks = _blocks.TryGetValue(audios[i].Hash, out var known) ? known : null;
            if (Parameters.Count == 0 && blocks is [var single] && Cache.TryGet(Key(audios[i]), Encoder.Device, out var cached))
            {
                if (cached.Rank == 2 && cached.Shape[0] == single.Tokens && cached.Shape[1] == Encoder.Width)
                {
                    features[i] = [new ImageFeatures(cached, single)];
                    Interlocked.Increment(ref _hits);
                    continue;
                }

                cached.Dispose();                                          // another shape: not this encoder's; encoded again
            }

            misses.Add(i);
        }

        if (misses.Count > 0)
        {
            var decoded = new List<AudioData>(misses.Count);
            foreach (int i in misses)
            {
                decoded.Add(Read(audios[i]));
            }

            long start = Telemetry.Start(TelemetryLevel.Inference);
            IReadOnlyList<ImageFeatures> encoded;
            lock (_family)
            {
                encoded = Encoder.Encode(decoded);
            }

            try
            {
                int at = 0;
                for (int m = 0; m < misses.Count; m++)
                {
                    int i = misses[m];
                    var blocks = _blocks.GetOrAdd(audios[i].Hash, _ => { lock (_family) { return Encoder.Blocks(decoded[m]); } });
                    if (at + blocks.Count > encoded.Count)
                    {
                        throw new InvalidOperationException($"The audio encoder gave {encoded.Count} blocks' features for {misses.Count} sounds of more blocks.");
                    }

                    var own = new ImageFeatures[blocks.Count];
                    for (int b = 0; b < blocks.Count; b++)
                    {
                        own[b] = encoded[at + b];
                        if (!own[b].Layout.Equals(blocks[b]) || own[b].Features.Shape[^1] != Encoder.Width)
                        {
                            throw new InvalidOperationException($"Sound {i}, block {b}: the encoder gave {Tensor.FormatShape(own[b].Features.Shape)} ({own[b].Layout}); its layout says {blocks[b]} of width {Encoder.Width}.");
                        }
                    }

                    if (blocks.Count == 1 && Parameters.Count == 0)
                    {
                        Cache.Put(Key(audios[i]), own[0].Features);           // one block: cached as it is (several: encoded per call)
                    }

                    features[i] = own;
                    at += blocks.Count;
                    Interlocked.Increment(ref _encoded);
                }

                if (at != encoded.Count)
                {
                    throw new InvalidOperationException($"The audio encoder gave {encoded.Count} blocks' features for {at} blocks.");
                }
            }
            catch
            {
                foreach (var f in encoded)
                {
                    f.Dispose();
                }

                foreach (var f in features)
                {
                    if (f is not null)
                    {
                        foreach (var one in f)
                        {
                            one.Dispose();
                        }
                    }
                }

                throw;
            }

            if (start != 0)
            {
                long bytes = 0;
                foreach (var f in encoded)
                {
                    bytes += 4L * f.Features.Size;
                }

                Telemetry.InferenceStage(InferenceStageKind.Encoder, Encoder.Device, start, misses.Count, bytes);
            }
        }

        var all = new List<ImageFeatures>();
        foreach (var f in features)
        {
            all.AddRange(f!);
        }

        return all;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_ownsEncoder)
        {
            Encoder.Dispose();
        }

        if (_ownsCache)
        {
            Cache.Dispose();
        }
    }

    private static ChatAudio Of(ChatPart part) =>
        part as ChatAudio ?? throw new ArgumentException($"An audio source reads ChatAudio parts, not {(part is null ? "null" : $"'{part.Kind}' ({part.GetType().Name})")}.", nameof(part));
}
