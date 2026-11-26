// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;

namespace Idrak.Audio.Abstractions;

/// <summary>
/// How sound becomes a model's features: the frame (a short-time Fourier transform of windowed frames), the mel
/// filterbank and the log, as transformers' Whisper feature extractor computes them by default. Every value is a
/// setting; <see cref="FromConfig"/> reads a family's <c>preprocessor_config.json</c>.
/// </summary>
public sealed record AudioFeatureOptions
{
    /// <summary>The rate the features are computed at (the family's); sound at another rate is resampled first.</summary>
    public int SampleRate { get; init; } = 16000;

    /// <summary>The frame's samples (n_fft).</summary>
    public int WindowSize { get; init; } = 400;

    /// <summary>Samples between frames.</summary>
    public int HopSize { get; init; } = 160;

    /// <summary>The window: "hann" (periodic), "hamming" (periodic) or "rectangular".</summary>
    public string Window { get; init; } = "hann";

    /// <summary>Whether frames are centred on their samples: the signal reflect-padded by half a window at both ends.</summary>
    public bool Center { get; init; } = true;

    /// <summary>Mel bands.</summary>
    public int Mels { get; init; } = 80;

    /// <summary>The lowest frequency of the filterbank, Hz.</summary>
    public float MinFrequency { get; init; }

    /// <summary>The highest frequency of the filterbank, Hz; half the rate when null.</summary>
    public float? MaxFrequency { get; init; }

    /// <summary>The mel scale: "slaney" or "htk".</summary>
    public string MelScale { get; init; } = "slaney";

    /// <summary>The filterbank's normalisation: "slaney" (each band's area 1) or "none".</summary>
    public string MelNorm { get; init; } = "slaney";

    /// <summary>The least power before the log (values below are raised to it).</summary>
    public float LogFloor { get; init; } = 1e-10f;

    /// <summary>Log values more than this below the segment's largest are raised to that line; null for none.</summary>
    public float? DynamicRange { get; init; } = 8f;

    /// <summary>The affine normalisation after the log, (x + Offset) * Scale; null for none.</summary>
    public (float Offset, float Scale)? Normalize { get; init; } = (4f, 0.25f);

    /// <summary>Pad (with silence) or cut each segment to this length before the frames; null for the sound's own length.</summary>
    public double? PadSeconds { get; init; } = 30;

    /// <summary>
    /// Which computation: "auto" (the host on the CPU, the composed device operations elsewhere), "host" or "composed";
    /// <c>IDRAK_STFT_PATH</c> overrides. "kernel" names a fused device kernel (none on the new design yet).
    /// </summary>
    public string Path { get; init; } = "auto";

    /// <summary>The frequency bins of a frame: WindowSize / 2 + 1.</summary>
    public int Bins => WindowSize / 2 + 1;

    /// <summary>The highest frequency in effect: <see cref="MaxFrequency"/>, or half the rate.</summary>
    public float TopFrequency => MaxFrequency ?? SampleRate / 2f;

    /// <summary>
    /// Options from transformers' <c>preprocessor_config.json</c> (<c>sampling_rate</c>, <c>n_fft</c>, <c>hop_length</c>,
    /// <c>feature_size</c> (the mels), <c>chunk_length</c> (the seconds padded to)), the other keys as in
    /// <paramref name="defaults"/> (or the defaults here). A key missing keeps the default.
    /// </summary>
    public static AudioFeatureOptions FromConfig(JsonObject config, AudioFeatureOptions? defaults = null)
    {
        ArgumentNullException.ThrowIfNull(config);
        var d = defaults ?? new AudioFeatureOptions();
        int Int(string key, int fallback) => config[key] is { } v && v.GetValueKind() == System.Text.Json.JsonValueKind.Number ? (int)v : fallback;
        return d with
        {
            SampleRate = Int("sampling_rate", d.SampleRate),
            WindowSize = Int("n_fft", d.WindowSize),
            HopSize = Int("hop_length", d.HopSize),
            Mels = Int("feature_size", d.Mels),
            PadSeconds = config["chunk_length"] is { } chunk && chunk.GetValueKind() == System.Text.Json.JsonValueKind.Number ? (double)chunk : d.PadSeconds,
        };
    }

    /// <summary>The options as JSON (for a run's record).</summary>
    public JsonObject ToJson() => new()
    {
        ["sample_rate"] = SampleRate, ["window_size"] = WindowSize, ["hop_size"] = HopSize, ["window"] = Window, ["center"] = Center,
        ["mels"] = Mels, ["min_frequency"] = MinFrequency, ["max_frequency"] = MaxFrequency, ["mel_scale"] = MelScale, ["mel_norm"] = MelNorm,
        ["log_floor"] = LogFloor, ["dynamic_range"] = DynamicRange, ["normalize"] = Normalize is { } n ? new JsonArray(n.Offset, n.Scale) : null,
        ["pad_seconds"] = PadSeconds, ["path"] = Path,
    };

    internal void Check()
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(SampleRate, nameof(SampleRate));
        ArgumentOutOfRangeException.ThrowIfLessThan(WindowSize, 2, nameof(WindowSize));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(HopSize, nameof(HopSize));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(Mels, nameof(Mels));
        if (MinFrequency < 0 || TopFrequency <= MinFrequency || TopFrequency > SampleRate / 2f)
        {
            throw new ArgumentException($"The filterbank spans {MinFrequency.ToString(CultureInfo.InvariantCulture)} to {TopFrequency.ToString(CultureInfo.InvariantCulture)} Hz at {SampleRate} Hz.");
        }

        if (MelScale is not ("slaney" or "htk") || MelNorm is not ("slaney" or "none") || Window is not ("hann" or "hamming" or "rectangular"))
        {
            throw new ArgumentException($"MelScale is slaney or htk ({MelScale}), MelNorm slaney or none ({MelNorm}), Window hann, hamming or rectangular ({Window}).");
        }

        if (Path is not ("auto" or "host" or "composed" or "kernel"))
        {
            throw new ArgumentException($"Path is auto, host, composed or kernel, not '{Path}'.");
        }
    }
}

/// <summary>Turns sound into a model's features.</summary>
public interface IAudioFeatureExtractor
{
    /// <summary>The extractor's name, e.g. "log-mel".</summary>
    string Name { get; }

    /// <summary>Its settings.</summary>
    AudioFeatureOptions Options { get; }

    /// <summary>The frames a sound of <paramref name="samples"/> samples (at <see cref="AudioFeatureOptions.SampleRate"/>) gives, padding included.</summary>
    int FramesFor(long samples);

    /// <summary>The features of <paramref name="audio"/> (mixed to one channel, resampled when needed) on <paramref name="device"/>: [frames, mels].</summary>
    Tensor Extract(AudioData audio, Device device);
}

