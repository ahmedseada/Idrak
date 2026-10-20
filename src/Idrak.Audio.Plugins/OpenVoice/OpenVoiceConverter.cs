// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Diagnostics;
using Idrak.Audio.Layers;
using Idrak.Models;

namespace Idrak.Audio.Plugins.OpenVoice;

/// <summary>
/// OpenVoice's tone colour converter (<c>SynthesizerTrn.voice_conversion</c>): the source's linear spectrogram through
/// the posterior encoder to latents (noise of <see cref="VoiceConversionOptions.Tau"/> around their means), the flow
/// forward with the source's embedding to the prior, back with the target's, and the HiFi-GAN decoder. A v2 checkpoint
/// (<see cref="OpenVoiceConfig.ZeroG"/>) gives the posterior encoder and the decoder a zero embedding. Its reference
/// encoder gives the embeddings (<see cref="Embed"/>: the mean of the clips', long ones in windows of <see cref="EmbedSeconds"/>). OpenVoice's watermark is not added.
/// </summary>
public sealed class OpenVoiceConverter : IVoiceConverter
{
    private readonly OpenVoiceReferenceEncoder _reference;
    private readonly PosteriorEncoder _posterior;
    private readonly ResidualCouplingBlock _flow;
    private readonly HifiGanGenerator _decoder;
    private readonly AudioFeatureOptions _spectrum;
    private bool _disposed;

    internal OpenVoiceConverter(StoredLayers weights, OpenVoiceConfig config, string? folder, List<string> notes)
    {
        Config = config;
        Folder = folder;
        Notes = notes;
        Device = weights.Device;
        _reference = new OpenVoiceReferenceEncoder(weights, config);
        _posterior = PosteriorEncoder.FromStored(weights, "posterior_encoder", config.Bins, config.HiddenChannels, config.InterChannels, config.PosteriorLayers,
            config.WaveNetKernelSize, 1, config.GinChannels);
        _flow = Flow(weights, config);
        _decoder = HifiGanGenerator.FromStored(weights, "decoder", config.InterChannels, config.UpsampleInitialChannel, config.UpsampleRates, config.UpsampleKernelSizes,
            config.ResblockKernelSizes, config.ResblockDilationSizes, config.LeakyReluSlope, config.FinalLeakyReluSlope, config.GinChannels);
        _spectrum = new AudioFeatureOptions
        {
            SampleRate = config.SamplingRate,
            WindowSize = config.FilterLength,
            HopSize = config.HopLength,
            Window = "hann",
            Center = false,
        };
    }

    /// <summary>The flow of <paramref name="config"/> (<see cref="ResidualCouplingBlock"/>: its couplings' WaveNets conditioned on the embedding, dilation 1) from <paramref name="weights"/>.</summary>
    internal static ResidualCouplingBlock Flow(StoredLayers weights, OpenVoiceConfig config) =>
        ResidualCouplingBlock.FromStored(weights, "flow", config.Flows, config.InterChannels, config.HiddenChannels, config.FlowLayers, config.WaveNetKernelSize, 1, config.GinChannels);

    /// <summary>The checkpoint's configuration.</summary>
    public OpenVoiceConfig Config { get; }

    /// <summary>What was approximated while loading.</summary>
    public IReadOnlyList<string> Notes { get; }

    /// <inheritdoc />
    public string Name => OpenVoiceFamily.FamilyName;

    /// <inheritdoc />
    public int SampleRate => Config.SamplingRate;

    /// <inheritdoc />
    public int EmbeddingSize => Config.GinChannels;

    /// <inheritdoc />
    public Device Device { get; }

    /// <inheritdoc />
    public string? Folder { get; }

    /// <summary>
    /// OpenVoice's <c>spectrogram_torch</c> of one channel at <see cref="SampleRate"/>: reflect-padded by (n_fft − hop) / 2
    /// at both ends, frames of n_fft every hop (not centred), a periodic Hann window, √(re² + im² + 1e-6): [frames, bins].
    /// </summary>
    public Tensor Spectrogram(ReadOnlySpan<float> samples)
    {
        int pad = (Config.FilterLength - Config.HopLength) / 2;
        if (samples.Length <= pad)
        {
            throw new ArgumentException($"{samples.Length} samples are too few for a spectrogram ({pad + 1} at least, {(double)(pad + 1) / SampleRate:0.###} s).", nameof(samples));
        }

        var padded = new float[samples.Length + 2 * pad];
        samples.CopyTo(padded.AsSpan(pad));
        for (int i = 1; i <= pad; i++)
        {
            padded[pad - i] = samples[i];                                        // reflect: the edge sample is not repeated
            padded[pad + samples.Length - 1 + i] = samples[samples.Length - 1 - i];
        }

        using var power = Stft.Power(padded, _spectrum, Device);
        using var floored = power + 1e-6f;
        return floored.Sqrt();
    }

    /// <summary>
    /// Clips (and speech embedded as a source) longer than one and a half times this are embedded in equal windows of
    /// about this many seconds and the windows' embeddings averaged, as OpenVoice's se_extractor splits its references
    /// into 10 s segments: the reference encoder's memory stays bounded whatever the length. 0: each clip whole.
    /// </summary>
    public double EmbedSeconds { get; set; } = 10;

    /// <summary>The seconds of sound the decoder makes in one piece, with its context (the same samples as one pass); 0: each conversion piece in one pass. <see cref="VoiceConversionOptions.DecoderSeconds"/> sets it for one call.</summary>
    public double DecoderChunkSeconds { get; set; } = 10;

    /// <inheritdoc />
    public float[] Embed(IReadOnlyList<AudioData> clips, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(clips);
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (clips.Count == 0)
        {
            throw new ArgumentException("There are no clips to embed.", nameof(clips));
        }

        var sum = new double[EmbeddingSize];
        int windows = 0;
        foreach (var clip in clips)
        {
            var samples = Mono(clip);
            int window = EmbedSeconds > 0 ? (int)Math.Min(int.MaxValue, EmbedSeconds * SampleRate) : int.MaxValue;
            int count = samples.Length <= window + window / 2 ? 1 : (int)Math.Round((double)samples.Length / window);
            for (int k = 0; k < count; k++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int from = (int)((long)samples.Length * k / count), to = (int)((long)samples.Length * (k + 1) / count);
                long started = Telemetry.Start(TelemetryLevel.Inference);
                using var noGrad = Autograd.NoGrad();
                using var scope = new TensorScope();
                var embedding = _reference.Forward(Spectrogram(samples.AsSpan(from, to - from))).ToArray();
                Telemetry.InferenceStage(InferenceStageKind.Encoder, Device, started, to - from, (long)(to - from) * sizeof(float));
                for (int i = 0; i < sum.Length; i++)
                {
                    sum[i] += embedding[i];
                }

                windows++;
            }
        }

        var mean = new float[EmbeddingSize];
        for (int i = 0; i < mean.Length; i++)
        {
            mean[i] = (float)(sum[i] / windows);
        }

        return mean;
    }

    /// <inheritdoc />
    public AudioData Convert(AudioData speech, ReadOnlySpan<float> target, VoiceConversionOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(speech);
        ObjectDisposedException.ThrowIf(_disposed, this);
        options ??= new VoiceConversionOptions();
        if (target.Length != EmbeddingSize)
        {
            throw new ArgumentException($"The target embedding holds {target.Length} values; this converter's are {EmbeddingSize} wide.", nameof(target));
        }

        if (options.Source is { } given && given.Length != EmbeddingSize)
        {
            throw new ArgumentException($"VoiceConversionOptions.Source holds {given.Length} values; this converter's embeddings are {EmbeddingSize} wide.", nameof(options));
        }

        ArgumentOutOfRangeException.ThrowIfNegative(options.Tau, nameof(options));
        var samples = Mono(speech);
        var source = options.Source ?? Embed([new AudioData(samples, 1, SampleRate)], cancellationToken);
        var targetValues = target.ToArray();
        var random = new Random(options.Seed);
        var output = new float[samples.Length / Config.HopLength * Config.HopLength];
        int written = 0;
        foreach (var (from, count) in Pieces(samples, options.MaxSeconds))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var piece = ConvertPiece(samples.AsSpan(from, count), source, targetValues, options.Tau, random, options.DecoderSeconds ?? DecoderChunkSeconds);
            piece.CopyTo(output.AsSpan(written));
            written += piece.Length;
        }

        return new AudioData(written == output.Length ? output : output[..written], 1, SampleRate);
    }

    /// <summary>Disposes the layers.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _reference.Dispose();
        _posterior.Dispose();
        _flow.Dispose();
        _decoder.Dispose();
    }

    // One piece: [samples / hop · hop] samples of it in the target's voice.
    private float[] ConvertPiece(ReadOnlySpan<float> samples, float[] source, float[] target, float tau, Random random, double decoderSeconds)
    {
        long started = Telemetry.Start(TelemetryLevel.Inference);
        using var noGrad = Autograd.NoGrad();
        using var scope = new TensorScope();
        var spectrogram = Spectrogram(samples).Transpose();                     // [bins, T]
        var sourceColumn = Tensor.From(source, [EmbeddingSize, 1], Device);
        var targetColumn = Tensor.From(target, [EmbeddingSize, 1], Device);
        var zero = Config.ZeroG ? Tensor.Zeros([EmbeddingSize, 1], Device) : null;
        var latents = _posterior.Forward(spectrogram, zero ?? sourceColumn, tau, random);   // [F, T]
        var prior = _flow.Forward(latents, sourceColumn);
        var converted = _flow.Reverse(prior, targetColumn);
        Telemetry.InferenceStage(InferenceStageKind.Encoder, Device, started, spectrogram.Shape[1], (long)latents.Shape[0] * latents.Shape[1] * sizeof(float));
        return _decoder.Samples(converted, zero ?? targetColumn, decoderSeconds <= 0 ? 0 : Math.Max(1, (int)(decoderSeconds * SampleRate / Config.HopLength)));
    }

    // The speech mixed to one channel at the converter's rate.
    private float[] Mono(AudioData audio)
    {
        if (audio.SampleRate != SampleRate)
        {
            audio = Resampler.Resample(audio, SampleRate);
        }

        if (audio.Channels == 1)
        {
            return audio.Samples;
        }

        var mono = GC.AllocateUninitializedArray<float>(audio.Length);
        audio.ToMono(mono);
        return mono;
    }

    // The pieces (start, length) speech is converted in: whole hops, each at most maxSeconds, cut at the quietest hop
    // of the last fifth before the limit (a pause, most likely), so the joins fall where little is heard; the last piece
    // takes the rest (its frames, as one pass's: the samples over the hop, rounded down).
    private IEnumerable<(int From, int Count)> Pieces(float[] samples, double maxSeconds)
    {
        int hop = Config.HopLength, pad = (Config.FilterLength - hop) / 2;
        int shortest = (Math.Max(2 * hop, pad + 1) + hop - 1) / hop * hop;      // what a spectrogram's padding needs
        int limit = maxSeconds <= 0 ? int.MaxValue : Math.Max(Math.Max(hop * 8, 2 * shortest), (int)Math.Min(int.MaxValue, maxSeconds * SampleRate) / hop * hop);
        int from = 0;
        while (samples.Length - from >= hop)
        {
            int count = samples.Length - from;
            if (count > limit)
            {
                int best = limit, searchFrom = Math.Max(shortest, limit - limit / 5 / hop * hop);
                double quietest = double.MaxValue;
                for (int end = limit; end >= searchFrom; end -= hop)
                {
                    double energy = 0;
                    foreach (float s in samples.AsSpan(from + end - hop, hop))
                    {
                        energy += (double)s * s;
                    }

                    if (energy < quietest)
                    {
                        quietest = energy;
                        best = end;
                    }
                }

                count = samples.Length - from - best < shortest ? samples.Length - from : best;   // never leave a tail too short to convert
            }

            yield return (from, count);
            from += count;
        }
    }
}
