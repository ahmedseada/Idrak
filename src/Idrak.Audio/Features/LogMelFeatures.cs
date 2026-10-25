// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Audio;

/// <summary>
/// Log-mel features, as transformers' Whisper feature extractor computes them: the sound mixed to one channel and
/// resampled to the options' rate, padded or cut to <see cref="AudioFeatureOptions.PadSeconds"/>, its power spectra
/// (<see cref="Stft"/>) through the mel filterbank, log10 with a floor, the segment's dynamic range clipped, then an
/// affine normalisation; [frames, mels], the last centred frame dropped as Whisper drops it. The host path and the
/// composed device path give the same values (within float rounding); <see cref="AudioFeatureOptions.Path"/> chooses.
/// </summary>
public sealed class LogMelFeatures : IAudioFeatureExtractor
{
    private static readonly string? PathOverride = Environment.GetEnvironmentVariable("IDRAK_STFT_PATH") is { Length: > 0 } p ? p : null;

    /// <summary>The extractor with <paramref name="options"/> (Whisper's defaults when null).</summary>
    public LogMelFeatures(AudioFeatureOptions? options = null)
    {
        Options = options ?? new AudioFeatureOptions();
        Options.Check();
    }

    /// <inheritdoc />
    public string Name => "log-mel";

    /// <inheritdoc />
    public AudioFeatureOptions Options { get; }

    /// <summary>The samples a segment holds after padding or cutting (the sound's own when <see cref="AudioFeatureOptions.PadSeconds"/> is null).</summary>
    public long SegmentSamples(long samples) => Options.PadSeconds is { } seconds ? (long)Math.Round(seconds * Options.SampleRate) : samples;

    /// <inheritdoc />
    public int FramesFor(long samples)
    {
        int frames = Stft.FramesFor(SegmentSamples(samples), Options);
        return Options.Center ? Math.Max(0, frames - 1) : frames;             // the last centred frame is dropped
    }

    /// <summary>Which computation <paramref name="device"/> gets: "host" or "composed" (<see cref="AudioFeatureOptions.Path"/>, <c>IDRAK_STFT_PATH</c>).</summary>
    public string PathFor(Device device)
    {
        ArgumentNullException.ThrowIfNull(device);
        string path = PathOverride ?? Options.Path;
        return path switch
        {
            "auto" => device.Type == DeviceType.Cpu ? "host" : "composed",
            "host" or "composed" => path,
            "kernel" => throw new NotSupportedException("No fused STFT kernel is registered yet (AudioFeatureOptions.Path \"kernel\"); use auto, host or composed."),
            _ => throw new ArgumentException($"STFT path '{path}' is not auto, host, composed or kernel."),
        };
    }

    /// <inheritdoc />
    public Tensor Extract(AudioData audio, Device device)
    {
        ArgumentNullException.ThrowIfNull(audio);
        ArgumentNullException.ThrowIfNull(device);
        long start = Telemetry.StartStage();
        var segment = Segment(audio);
        int frames = FramesFor(audio.Length);
        if (frames == 0)
        {
            throw new ArgumentException($"{audio.Length} samples give no frame of {Options.WindowSize}.", nameof(audio));
        }

        var result = PathFor(device) == "host" ? OnHost(segment, frames, device) : Composed(segment, frames, device);
        Telemetry.InferenceStage(InferenceStageKind.AudioFeatures, device, start, frames, (long)frames * Options.Mels * sizeof(float));
        return result;
    }

    /// <summary>The features of several sounds, [batch, frames, mels] (each padded to the same length: <see cref="AudioFeatureOptions.PadSeconds"/> must be set).</summary>
    public Tensor Extract(IReadOnlyList<AudioData> audios, Device device)
    {
        ArgumentNullException.ThrowIfNull(audios);
        if (Options.PadSeconds is null)
        {
            throw new InvalidOperationException("A batch of features needs one length: set AudioFeatureOptions.PadSeconds.");
        }

        var items = new Tensor[audios.Count];
        try
        {
            for (int i = 0; i < audios.Count; i++)
            {
                items[i] = Extract(audios[i], device);
            }

            return Tensor.Stack(items);
        }
        finally
        {
            foreach (var item in items)
            {
                item?.Dispose();
            }
        }
    }

    /// <summary>The sound as one channel at the options' rate, padded or cut to the segment length.</summary>
    public float[] Segment(AudioData audio)
    {
        ArgumentNullException.ThrowIfNull(audio);
        var mono = audio.Channels == 1 ? audio : AudioCodecsMono(audio);
        var resampled = Resampler.Resample(mono, Options.SampleRate);
        long length = SegmentSamples(resampled.Length);
        if (length > int.MaxValue)
        {
            throw new ArgumentException($"A segment of {length} samples is too long.", nameof(audio));
        }

        var segment = new float[length];
        resampled.Pad(0, segment);
        return segment;
    }

    private static AudioData AudioCodecsMono(AudioData audio)
    {
        var mono = GC.AllocateUninitializedArray<float>(audio.Length);
        audio.ToMono(mono);
        return new AudioData(mono, 1, audio.SampleRate);
    }

    private Tensor OnHost(float[] segment, int frames, Device device)
    {
        int bins = Options.Bins, mels = Options.Mels;
        int allFrames = Stft.FramesFor(segment.Length, Options);
        var power = GC.AllocateUninitializedArray<float>(allFrames * bins);
        Stft.PowerOnHost(segment, Options, power);
        var filterbank = MelFilterbank.Build(Options);
        var features = GC.AllocateUninitializedArray<float>(frames * mels);
        float floor = Options.LogFloor, scale = 1f / MathF.Log(10f);
        HostParallel.For(frames, 32, (first, last) =>
        {
            for (int f = first; f < last; f++)
            {
                var row = power.AsSpan(f * bins, bins);
                var output = features.AsSpan(f * mels, mels);
                output.Clear();
                for (int b = 0; b < bins; b++)
                {
                    float p = row[b];
                    if (p == 0f)
                    {
                        continue;
                    }

                    var weights = filterbank.AsSpan(b * mels, mels);
                    for (int m = 0; m < mels; m++)
                    {
                        output[m] += p * weights[m];                             // bins in order: the host's fixed sum
                    }
                }

                for (int m = 0; m < mels; m++)
                {
                    output[m] = MathF.Log(Math.Max(output[m], floor)) * scale;
                }
            }
        });

        if (Options.DynamicRange is { } range)
        {
            float max = float.NegativeInfinity;
            foreach (float v in features)
            {
                max = Math.Max(max, v);
            }

            float line = max - range;
            for (int i = 0; i < features.Length; i++)
            {
                features[i] = Math.Max(features[i], line);
            }
        }

        if (Options.Normalize is { } n)
        {
            for (int i = 0; i < features.Length; i++)
            {
                features[i] = (features[i] + n.Offset) * n.Scale;
            }
        }

        var result = Tensor.Empty([frames, mels], device, zeroed: false);
        result.Load(features);
        return result;
    }

    private Tensor Composed(float[] segment, int frames, Device device)
    {
        int mels = Options.Mels;
        using var power = Stft.Power(segment, Options, device);                  // [frames + 1, bins] when centred
        using var kept = power.Shape[0] == frames ? null : power.Narrow(0, 0, frames);
        using var mel = (kept ?? power).MatMul(MelFilterbank.Tensor(Options, device));   // [frames, mels]
        using var floored = mel.Clamp(Options.LogFloor, float.PositiveInfinity);
        using var logged = floored.Log();
        Tensor current = logged * (1f / MathF.Log(10f));
        if (Options.DynamicRange is { } range)
        {
            // The segment's largest value: one max pooling to one value, read once (one value per segment).
            using var plane = current.Reshape(1, 1, frames, mels);
            using var pooled = plane.MaxPool(new ConvGeometry(1, 1, frames, mels, frames, mels, frames, mels, 0, 0));
            float max = pooled.Item();
            using var before = current;
            current = before.Maximum(max - range);
        }

        if (Options.Normalize is { } n)
        {
            using var before = current;
            using var shifted = before + n.Offset;
            current = shifted * n.Scale;
        }

        return current;
    }
}
