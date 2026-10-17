// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Runtime.CompilerServices;
using Idrak.Abstraction.Diagnostics;
using Idrak.Audio;
using Idrak.Audio.Layers;
using Idrak.Models;

namespace Idrak.VitsSpeech;

/// <summary>
/// A loaded VITS model (transformers' <c>VitsModel</c>, as MMS-TTS ships it): text to ids, ids through the text
/// encoder, durations (the stochastic predictor in reverse, or the deterministic one), the prior expanded to frames by
/// a gather of repeated rows, the flow in reverse, the HiFi-GAN decoder; one device pass per piece of text. The
/// rate, noise scales and seed come from <see cref="SpeechSynthesisOptions"/> (null scales: the checkpoint's).
/// </summary>
public sealed class VitsSynthesizer : IAdversarialSpeechSynthesizer
{
    private readonly VitsTextEncoder _encoder;
    private readonly IVitsDurationPredictor _durations;
    private readonly ResidualCouplingBlock _flow;
    private readonly HifiGanGenerator _decoder;
    private readonly PosteriorEncoder? _posterior;                               // training: the spectrogram's latents
    private readonly VitsTraining? _training;
    private readonly Tensor? _speakers;                                          // [speakers, E]
    private readonly string[] _voices;
    private bool _disposed;

    internal VitsSynthesizer(StoredLayers weights, VitsConfig config, VitsTokenizer tokenizer, List<string> notes, string? folder = null, Func<ITensorStore>? open = null)
    {
        Config = config;
        Tokenizer = tokenizer;
        Device = weights.Device;
        Notes = notes;
        Folder = folder;
        Open = open;
        _encoder = new VitsTextEncoder(weights, config);
        _durations = config.StochasticDurations
            ? new VitsStochasticDurationPredictor(weights, config)
            : new VitsDeterministicDurationPredictor(weights, config);
        _flow = ResidualCouplingBlock.FromStored(weights, "flow", config.PriorFlows, config.FlowSize, config.HiddenSize, config.PriorWaveNetLayers, config.WaveNetKernelSize,
            config.WaveNetDilationRate, config.SpeakerEmbeddingSize);
        _decoder = HifiGanGenerator.FromStored(weights, "decoder", config.FlowSize, config.UpsampleInitialChannel, config.UpsampleRates, config.UpsampleKernelSizes,
            config.ResblockKernelSizes, config.ResblockDilationSizes, config.LeakyReluSlope, config.FinalLeakyReluSlope, config.SpeakerEmbeddingSize);
        if (config.SpeakerEmbeddingSize > 0 && config.Speakers > 1)
        {
            _speakers = weights.Tensor("embed_speaker.weight", config.Speakers, config.SpeakerEmbeddingSize);
            _voices = SpeakerNames(folder, config.Speakers);
        }
        else
        {
            _voices = [];
        }

        Vocoder = new DecoderVocoder(this);
        if (weights.Trainable)
        {
            _posterior = PosteriorEncoder.FromStored(weights, "posterior_encoder", config.SpectrogramBins, config.HiddenSize, config.FlowSize, config.PosteriorWaveNetLayers,
                config.WaveNetKernelSize, config.WaveNetDilationRate, config.SpeakerEmbeddingSize);
            _training = new VitsTraining(this, weights);
        }
    }

    /// <summary>The folder the model was read from (its files are copied when it is saved), or null.</summary>
    public string? Folder { get; }

    internal Func<ITensorStore>? Open { get; }

    internal VitsTextEncoder Encoder => _encoder;

    internal IVitsDurationPredictor StochasticDurations => _durations;

    internal ResidualCouplingBlock Flow => _flow;

    internal HifiGanGenerator Decoder => _decoder;

    internal PosteriorEncoder Posterior => _posterior ?? throw NotTrainable();

    /// <summary>Whether the model was loaded to be trained (<see cref="SpeechSynthesisLoadOptions.Training"/>).</summary>
    public bool Trainable => _training is not null;

    internal VitsTraining Training => _training ?? throw NotTrainable();

    /// <inheritdoc />
    public string Architecture => VitsSpeechFamily.Architecture;

    /// <inheritdoc />
    public Module Generator => Training.Generator;

    /// <inheritdoc />
    public IReadOnlyList<string> Parts => Training.Generator.Parts;

    /// <inheritdoc />
    public void Freeze(IReadOnlyCollection<string> parts) => Training.Generator.Freeze(parts);

    /// <inheritdoc />
    IAudioFeatureExtractor IAdversarialSpeechSynthesizer.Features => Training.Features;

    /// <inheritdoc />
    public int FeatureWidth => Training.Features.Width;

    /// <inheritdoc />
    public string Refused => "have fewer frames than their tokens or than the decoder's segment";

    /// <inheritdoc />
    public IReadOnlyList<string> TrainingFiles => VitsTraining.TrainingFiles;

    /// <inheritdoc />
    public int[] Labels(string text) => Tokenizer.Encode(text);

    /// <inheritdoc />
    public bool Accepts(int frames, int[] labels, AdversarialSynthesisOptions options) => Training.Accepts(frames, labels, options);

    /// <inheritdoc />
    public Module Discriminators(AdversarialSynthesisOptions options, int seed) => Training.Discriminators(options, seed).Module;

    /// <inheritdoc />
    public IAdversarialSynthesisStep Begin(AdversarialSynthesisBatch batch, AdversarialSynthesisOptions options, bool keep) => Training.Begin(batch, options, keep);

    /// <inheritdoc />
    public float MelError(AdversarialSynthesisBatch batch, AdversarialSynthesisOptions options) => Training.MelError(batch, options);

    /// <inheritdoc />
    public void Save(string folder) => Training.Save(folder);

    private static InvalidOperationException NotTrainable() =>
        new("This VITS model was loaded to speak: load it with SpeechSynthesisLoadOptions.Training (idrak voice train does) to train it.");

    /// <summary>The checkpoint's configuration.</summary>
    public VitsConfig Config { get; }

    /// <summary>The checkpoint's tokenizer (its flags can be changed).</summary>
    public VitsTokenizer Tokenizer { get; }

    /// <summary>What was approximated while loading (nothing, for a complete checkpoint).</summary>
    public IReadOnlyList<string> Notes { get; }

    /// <inheritdoc />
    public Device Device { get; }

    /// <inheritdoc />
    public int SampleRate => Config.SamplingRate;

    /// <summary>
    /// The checkpoint's recorded speakers (route A of plan 21's S9), by name: those of <see cref="SpeakersFile"/> beside
    /// config.json (VCTK's p225, ... as an app writes them; the hub's config.json holds only their count), else their
    /// indices. <see cref="SpeechSynthesisOptions.Voice"/> takes a name (any case) or an index; without a voice the
    /// first speaker speaks (the contract's default). Empty for a single-speaker checkpoint.
    /// </summary>
    public IReadOnlyList<string> Voices => _voices;

    /// <summary>
    /// The speakers' names beside config.json: a JSON array in index order (<c>["p225", "p226"]</c>) or an object of
    /// names and indices (<c>{"p225": 0}</c>); speakers it leaves out keep their indices.
    /// </summary>
    public const string SpeakersFile = "speakers.json";

    // The speakers' names: speakers.json's, else the indices.
    private static string[] SpeakerNames(string? folder, int count)
    {
        var names = Enumerable.Range(0, count).Select(i => i.ToString(CultureInfo.InvariantCulture)).ToArray();
        string? path = folder is null ? null : Path.Combine(folder, SpeakersFile);
        if (path is null || !File.Exists(path))
        {
            return names;
        }

        var json = System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(path));
        IEnumerable<(string? Name, int Index)> given = json switch
        {
            System.Text.Json.Nodes.JsonArray list => list.Select((n, i) => ((string?)n, i)),
            System.Text.Json.Nodes.JsonObject map => map.Select(p => ((string?)p.Key, p.Value is System.Text.Json.Nodes.JsonValue v && v.TryGetValue(out int i) ? i : -1)),
            _ => throw new InvalidDataException($"{path} is a list of the speakers' names in index order, or an object of names and indices."),
        };
        foreach (var (name, index) in given)
        {
            if (string.IsNullOrWhiteSpace(name) || (uint)index >= (uint)count)
            {
                throw new InvalidDataException($"{path}: '{name}' at {index} is not a name of one of the checkpoint's {count} speakers (0 to {count - 1}).");
            }

            names[index] = name;
        }

        var repeated = names.GroupBy(n => n, StringComparer.OrdinalIgnoreCase).FirstOrDefault(g => g.Count() > 1);
        return repeated is null ? names : throw new InvalidDataException($"{path} names two speakers '{repeated.Key}'.");
    }

    /// <summary>
    /// The seconds of sound the decoder makes in one piece (each piece with its context, so the samples are a whole
    /// pass's): the device's memory follows the piece, not the text's length. 0: the whole utterance in one pass.
    /// <see cref="SpeechSynthesisOptions.DecoderSeconds"/> sets it for one call.
    /// </summary>
    public double DecoderChunkSeconds { get; set; } = 10;

    /// <summary>The decoder's piece in latent frames for <see cref="DecoderChunkSeconds"/> (0: one pass).</summary>
    internal int DecoderChunkFrames => ChunkFrames(DecoderChunkSeconds);

    // The decoder's piece in latent frames for pieces of this many seconds (0: one pass).
    private int ChunkFrames(double seconds) => seconds <= 0 ? 0 : Math.Max(1, (int)(seconds * Config.SamplingRate / Config.HopLength));

    /// <summary>The HiFi-GAN decoder as a vocoder over the flow's latents [frames, flow_size] (what a model trained to produce them can use).</summary>
    public IVocoder Vocoder { get; }

    /// <inheritdoc />
    public (AudioData Audio, SynthesisStats Stats) Speak(string text, SpeechSynthesisOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        ObjectDisposedException.ThrowIf(_disposed, this);
        options ??= new SpeechSynthesisOptions();
        Device.Backend.ResetPeakMemoryUsage();
        long started = Stopwatch.GetTimestamp();
        var pieces = SplitText(text, options.MaxCharacters);
        var ids = new List<int[]>(pieces.Count);
        foreach (var piece in pieces)
        {
            var encoded = Tokenizer.Encode(piece);
            if (encoded.Length > 0)
            {
                ids.Add(encoded);
            }
        }

        var textTime = Stopwatch.GetElapsedTime(started);
        var samples = new List<float[]>(ids.Count);
        TimeSpan model = TimeSpan.Zero, vocoder = TimeSpan.Zero;
        long total = 0;
        for (int i = 0; i < ids.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var (piece, _, modelTime, vocoderTime) = SynthesizeCore(ids[i], options, options.Seed + i);
            samples.Add(piece);
            total += piece.Length;
            model += modelTime;
            vocoder += vocoderTime;
        }

        var joined = new float[total];
        int at = 0;
        foreach (var piece in samples)
        {
            piece.CopyTo(joined, at);
            at += piece.Length;
        }

        var audio = new AudioData(joined, 1, SampleRate);
        if (options.SampleRate is { } rate && rate != SampleRate)
        {
            audio = Resampler.Resample(audio, rate);
        }

        long peak = Device.Backend.GetMemoryUsage().Peak;
        double seconds = (double)total / SampleRate;
        var elapsed = Stopwatch.GetElapsedTime(started);
        return (audio, new SynthesisStats(textTime, model, vocoder, peak, seconds > 0 ? elapsed.TotalSeconds / seconds : 0));
    }

    /// <inheritdoc />
    public async IAsyncEnumerable<AudioData> SpeakStream(string text, SpeechSynthesisOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        ObjectDisposedException.ThrowIf(_disposed, this);
        options ??= new SpeechSynthesisOptions();
        var pieces = SplitText(text, options.MaxCharacters);
        int index = 0;
        foreach (var piece in pieces)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var ids = Tokenizer.Encode(piece);
            if (ids.Length == 0)
            {
                continue;
            }

            int seed = options.Seed + index++;
            var (samples, _, _, _) = await Task.Run(() => SynthesizeCore(ids, options, seed), cancellationToken).ConfigureAwait(false);
            var audio = new AudioData(samples, 1, SampleRate);
            yield return options.SampleRate is { } rate && rate != SampleRate ? Resampler.Resample(audio, rate) : audio;
        }
    }

    /// <summary>Speaks token ids directly (no text): the samples at <see cref="SampleRate"/> and each token's frames.</summary>
    public (AudioData Audio, int[] Durations) Synthesize(ReadOnlySpan<int> ids, SpeechSynthesisOptions? options = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        options ??= new SpeechSynthesisOptions();
        var (samples, durations, _, _) = SynthesizeCore(ids.ToArray(), options, options.Seed);
        return (new AudioData(samples, 1, SampleRate), durations);
    }

    /// <summary>The pieces <paramref name="text"/> is spoken in (<see cref="SpeechSynthesisText.Pieces"/>).</summary>
    public static IReadOnlyList<string> SplitText(string text, int maxCharacters) => SpeechSynthesisText.Pieces(text, maxCharacters);

    // One piece: the samples, the durations, and the model's and the decoder's times.
    private (float[] Samples, int[] Durations, TimeSpan Model, TimeSpan Vocoder) SynthesizeCore(int[] ids, SpeechSynthesisOptions options, int seed)
    {
        if (ids.Length == 0)
        {
            throw new ArgumentException("There are no ids to speak.", nameof(ids));
        }

        if (options.Rate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options), $"SpeechSynthesisOptions.Rate is positive, got {options.Rate}.");
        }

        float noiseScale = options.NoiseScale ?? Config.NoiseScale;
        float durationNoise = options.DurationNoiseScale ?? Config.NoiseScaleDuration;
        float lengthScale = 1f / (options.Rate * Config.SpeakingRate);
        var random = new Random(seed);
        long started = Stopwatch.GetTimestamp();
        long telemetryStart = Telemetry.Start(TelemetryLevel.Inference);
        using var noGrad = Autograd.NoGrad();
        using var scope = new TensorScope();
        var speaker = Speaker(options.Voice);
        var (hidden, means, logVariances) = _encoder.Forward(ids);               // [T, C], [T, F], [T, F]
        var logDurations = _durations.LogDurations(hidden.Transpose(), speaker, durationNoise, random);
        var durations = new int[ids.Length];
        int frames = 0;
        for (int i = 0; i < durations.Length; i++)
        {
            durations[i] = (int)Math.Ceiling(Math.Exp(logDurations[i]) * lengthScale);
            frames += durations[i];
        }

        if (frames < 1)
        {
            durations[0] = 1;                                                    // torch clamps the total to at least one frame
            frames = 1;
        }

        // The prior expanded to frames: a gather of each token's row repeated for its duration.
        var rows = new int[frames];
        for (int i = 0, at = 0; i < durations.Length; i++)
        {
            for (int k = 0; k < durations[i]; k++)
            {
                rows[at++] = i;
            }
        }

        var indices = VitsOps.Indices(rows, Device);
        var prior = means.EmbeddingLookup(indices);                              // [frames, F]
        if (noiseScale != 0)
        {
            var noise = new float[frames * Config.FlowSize];
            for (int i = 0; i < noise.Length; i++)
            {
                noise[i] = (float)random.NextGaussian();
            }

            prior = prior + Tensor.From(noise, [frames, Config.FlowSize], Device) * logVariances.EmbeddingLookup(indices).Exp() * noiseScale;
        }

        var latents = _flow.Reverse(prior.Transpose(), speaker);                 // [F, frames]
        Telemetry.InferenceStage(InferenceStageKind.Encoder, Device, telemetryStart, ids.Length, (long)frames * Config.FlowSize * sizeof(float));
        var modelTime = Stopwatch.GetElapsedTime(started);
        long decoderStart = Stopwatch.GetTimestamp();
        var samples = _decoder.Samples(latents, speaker, ChunkFrames(options.DecoderSeconds ?? DecoderChunkSeconds));   // [1, frames · hop]
        return (samples, durations, modelTime, Stopwatch.GetElapsedTime(decoderStart));
    }

    // The speaker's embedding as a column [E, 1] (the first speaker's when no voice is asked for), or null for a
    // single-speaker model.
    private Tensor? Speaker(string? voice)
    {
        if (_speakers is null)
        {
            if (voice is not null)
            {
                throw new ArgumentException($"This model has one voice; '{voice}' was asked for.", nameof(voice));
            }

            return null;
        }

        return _speakers.EmbeddingLookup(VitsOps.Indices([voice is null ? 0 : SpeakerIndex(voice)], Device)).Transpose();
    }

    /// <summary>The index of speaker <paramref name="voice"/>: one of <see cref="Voices"/> (any case), or an index.</summary>
    /// <exception cref="ArgumentException">Not a speaker of this model; the message lists them.</exception>
    public int SpeakerIndex(string voice)
    {
        ArgumentException.ThrowIfNullOrEmpty(voice);
        int named = Array.FindIndex(_voices, n => n.Equals(voice, StringComparison.OrdinalIgnoreCase));
        if (named >= 0)
        {
            return named;
        }

        if (int.TryParse(voice, NumberStyles.Integer, CultureInfo.InvariantCulture, out int index) && (uint)index < (uint)_voices.Length)
        {
            return index;
        }

        throw new ArgumentException(_voices.Length == 0
            ? $"This model has one voice; '{voice}' was asked for."
            : $"Voice '{voice}' is not one of this model's {_voices.Length} speakers ({Listed(_voices)}, or 0 to {_voices.Length - 1}).", nameof(voice));
    }

    private static string Listed(string[] names) => names.Length <= 12 ? string.Join(", ", names) : $"{string.Join(", ", names.Take(12))}, …";

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _encoder.Dispose();
        _durations.Dispose();
        _flow.Dispose();
        _decoder.Dispose();
        _posterior?.Dispose();
        _training?.Dispose();
        _speakers?.Dispose();
    }

    // The decoder as a vocoder: latents [frames, flow_size] (frames as rows) to samples.
    private sealed class DecoderVocoder(VitsSynthesizer owner) : IVocoder
    {
        public string Name => "hifigan";

        public int SampleRate => owner.SampleRate;

        public int Mels => owner.Config.FlowSize;

        public AudioData Synthesize(Tensor logMel, Device device)
        {
            ArgumentNullException.ThrowIfNull(logMel);
            if (logMel.Rank != 2 || logMel.Shape[1] != Mels)
            {
                throw new ArgumentException($"The HiFi-GAN decoder takes [frames, {Mels}], got {Tensor.FormatShape(logMel.Shape)}.", nameof(logMel));
            }

            using var noGrad = Autograd.NoGrad();
            using var scope = new TensorScope();
            var samples = owner._decoder.Samples(logMel.To(owner.Device).Transpose(), null, owner.DecoderChunkFrames);
            return new AudioData(samples, 1, SampleRate);
        }
    }
}
