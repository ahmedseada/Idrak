// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Audio;

// "highpass" (live): a Butterworth high-pass of `order` (default 2: the RBJ cookbook's, Q = 1/√2; higher orders as
// cascaded sections, FilterDesign.Butterworth) run over each channel in order, in double, its memory kept between blocks
// (BiquadStage); the DC offset and the rumble below the cutoff go, the voice above it is untouched in level. The filter
// starts settled on the first sample's level (no step at the start). A cutoff of 0 leaves the sound as it is. Order 2
// gives the samples of the high-pass before plan 31 bit for bit.
internal sealed class HighPassProcessor : ILiveAudioProcessor
{
    public const double DefaultHertz = 60;
    public const int DefaultOrder = 2;

    public string Name => AudioProcessors.HighPass;

    public string Label => "High-pass filter";

    public string Summary => $"a Butterworth high-pass at the value, Hz (default {DefaultHertz}; 0 leaves the sound), of `order` (default {DefaultOrder}): the DC offset and the rumble go";

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        new("", "Cutoff", AudioProcessorInput.Number)
        {
            Min = 0, Step = 10, Unit = "Hz", Optional = true, Default = "60", Help = "Below it the sound is turned down, 6 dB an octave per order (12 at order 2)",
        },
        FilterDesign.Order,
    ];

    public IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels)
    {
        double cutoff = step.Number(DefaultHertz);
        FilterDesign.ThrowIfNotBelowHalf(Name, "cutoff", cutoff, sampleRate);
        Biquad[] sections = cutoff <= 0 ? [] : FilterDesign.Butterworth(high: true, step.Integer("order", DefaultOrder), cutoff, sampleRate);
        return new BiquadStage(sections, sampleRate, channels);
    }
}

// "trim" (whole recording): the silence at both ends cut. A frame more than `silence` dB under the loudest is silence;
// the speech runs from the first frame that is not to the last, the value's seconds kept on both sides. A silent
// recording stays as it is.
internal sealed class TrimProcessor : IRecordingAudioProcessor
{
    public const double DefaultPad = 0.1, DefaultSilence = 40, DefaultHop = 0.01;

    public string Name => AudioProcessors.Trim;

    public string Label => "Trim silence";

    public string Summary => $"the silence at both ends cut, the value's seconds kept on both sides (default {DefaultPad}); silence is a frame `silence` dB under the loudest";

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        new("", "Pad", AudioProcessorInput.Number) { Min = 0, Step = 0.05, Unit = "s", Optional = true, Default = "0.1", Help = "The silence kept on both sides of the speech" },
        CleaningParameters.Silence,
        CleaningParameters.Hop,
    ];

    public AudioData Apply(AudioData audio, AudioProcessorStep step)
    {
        ArgumentNullException.ThrowIfNull(audio);
        if (audio.Length == 0)
        {
            return audio;
        }

        var (decibels, loudest, hop) = CleaningLevels.Frames(audio, step.Number("hop", DefaultHop));
        float threshold = loudest - (float)step.Number("silence", DefaultSilence);
        int first = Array.FindIndex(decibels, d => d >= threshold), last = Array.FindLastIndex(decibels, d => d >= threshold);
        if (first < 0)
        {
            return audio;
        }

        int pad = (int)Math.Round(step.Number(DefaultPad) * audio.SampleRate);
        int start = Math.Max(0, first * hop - pad), end = Math.Min(audio.Length, (last + 1) * hop + pad);
        return start == 0 && end == audio.Length ? audio : audio.Slice(start, end - start);
    }
}

// "normalize" (whole recording): the speech level (the RMS over the frames that are not silence) brought to the value,
// dBFS, by one gain over every channel, lowered when the peak would pass `peak`. A silent recording stays as it is.
internal sealed class NormalizeProcessor : IRecordingAudioProcessor
{
    public const double DefaultTarget = -20, DefaultPeak = 0.95;

    public string Name => AudioProcessors.Normalize;

    public string Label => "Normalize level";

    public string Summary => $"the speech level (the RMS of the frames that are not silence) brought to the value, dBFS (default {DefaultTarget}), the peak kept under `peak`";

    public string? LiveAlternative => "agc";

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        new("", "Target", AudioProcessorInput.Number) { Max = 0, Step = 1, Unit = "dBFS", Optional = true, Default = "-20", Help = "The speech level aimed at" },
        CleaningParameters.Silence,
        CleaningParameters.Hop,
        new("peak", "Peak ceiling", AudioProcessorInput.Number)
        {
            Min = 0, MinExclusive = true, Max = 1, Step = 0.01, Optional = true, Default = "0.95", Help = "The highest sample allowed after the gain (the gain is lowered to keep under it)",
        },
    ];

    public AudioData Apply(AudioData audio, AudioProcessorStep step)
    {
        ArgumentNullException.ThrowIfNull(audio);
        if (audio.Length == 0)
        {
            return audio;
        }

        var (decibels, loudest, _) = CleaningLevels.Frames(audio, step.Number("hop", TrimProcessor.DefaultHop));
        float threshold = loudest - (float)step.Number("silence", TrimProcessor.DefaultSilence);
        double sum = 0;
        int count = 0;
        foreach (float d in decibels)
        {
            if (d >= threshold && d > -200f)
            {
                sum += Math.Pow(10, d / 10);
                count++;
            }
        }

        if (count == 0 || !(sum > 0))
        {
            return audio;
        }

        double level = 10 * Math.Log10(sum / count), ceiling = step.Number("peak", DefaultPeak);
        double gain = Math.Pow(10, (step.Number(DefaultTarget) - level) / 20);
        var samples = audio.Samples;
        float peak = 0;
        foreach (float x in samples)
        {
            peak = Math.Max(peak, Math.Abs(x));
        }

        if (peak * gain > ceiling)
        {
            gain = ceiling / peak;
        }

        if (Math.Abs(gain - 1) < 1e-6)
        {
            return audio;
        }

        var output = GC.AllocateUninitializedArray<float>(samples.Length);
        CleaningLevels.Scale(samples, (float)gain, output);
        return new AudioData(output, audio.Channels, audio.SampleRate);
    }
}

// "denoise" (whole recording): spectral gating over each channel. The spectrum (a Hann window of the largest power of
// two of samples under `window` seconds, a quarter of it the hop); the noise floor is each frequency's mean magnitude
// over the quietest `quantile` share of the frames (the pauses: what is left when nobody speaks); a frame's gain at a
// frequency is (magnitude - value · floor) / magnitude, no lower than `floor` dB and no higher than 1, smoothed over the
// neighbouring frame on each side (no musical noise); the gained spectrum goes back through the inverse STFT to the
// recording's exact length.
internal sealed class DenoiseProcessor : IRecordingAudioProcessor
{
    public const double DefaultStrength = 1.5, DefaultQuantile = 0.1, DefaultFloor = -25, DefaultWindow = 0.032;

    public string Name => AudioProcessors.Denoise;

    public string Label => "Denoise";

    public string Summary => $"spectral gating: the noise floor read from the quietest frames (the pauses) taken off value times (default {DefaultStrength}), no frequency turned down past `floor` dB";

    public string? LiveAlternative => RepairProcessors.DenoiseLive;

    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
    [
        new("", "Strength", AudioProcessorInput.Number)
        {
            Min = 0, Step = 0.1, Optional = true, Default = "1.5", Help = "How many times the noise floor is taken off (1: the floor itself; more removes more noise and thins the voice)",
        },
        new("quantile", "Noise share", AudioProcessorInput.Number)
        {
            Min = 0, Max = 1, Step = 0.05, Optional = true, Default = "0.1", Help = "The share of the frames, the quietest, the noise floor is read from",
        },
        new("floor", "Floor", AudioProcessorInput.Number) { Max = 0, Step = 1, Unit = "dB", Optional = true, Default = "-25", Help = "The most a frequency is turned down" },
        new("window", "Window", AudioProcessorInput.Number)
        {
            Min = 0, MinExclusive = true, Step = 0.008, Unit = "s", Optional = true, Default = "0.032",
            Help = "The window the spectrum is taken over (rounded down to a power of two of samples; the hop a quarter of it)",
        },
    ];

    public AudioData Apply(AudioData audio, AudioProcessorStep step)
    {
        ArgumentNullException.ThrowIfNull(audio);
        int size = 16;
        double window = step.Number("window", DefaultWindow);
        while (size * 2 <= window * audio.SampleRate)
        {
            size *= 2;
        }

        if (audio.Length < 2 * size)
        {
            return audio;
        }

        var stft = new AudioFeatureOptions { SampleRate = audio.SampleRate, WindowSize = size, HopSize = size / 4, Window = "hann", Center = true, PadSeconds = null };
        int length = audio.Length, frames = Stft.FramesFor(length, stft), bins = stft.Bins;
        float floorGain = (float)Math.Pow(10, step.Number("floor", DefaultFloor) / 20);
        float strength = (float)step.Number(DefaultStrength);
        var real = GC.AllocateUninitializedArray<float>(frames * bins);
        var imaginary = GC.AllocateUninitializedArray<float>(frames * bins);
        var magnitude = GC.AllocateUninitializedArray<float>(frames * bins);
        var gain = GC.AllocateUninitializedArray<float>(frames * bins);
        var energy = GC.AllocateUninitializedArray<float>(frames);
        var order = GC.AllocateUninitializedArray<int>(frames);
        var floor = new float[bins];
        var output = GC.AllocateUninitializedArray<float>(audio.Samples.Length);
        int quietest = Math.Clamp((int)Math.Round(step.Number("quantile", DefaultQuantile) * frames), 1, frames);
        for (int c = 0; c < audio.Channels; c++)
        {
            Stft.ComplexOnHost(audio.Channel(c), stft, real, imaginary);
            for (int f = 0; f < frames; f++)
            {
                float sum = 0;
                for (int b = 0; b < bins; b++)
                {
                    int i = f * bins + b;
                    magnitude[i] = MathF.Sqrt(real[i] * real[i] + imaginary[i] * imaginary[i]);
                    sum += magnitude[i] * magnitude[i];
                }

                energy[f] = sum;
                order[f] = f;
            }

            Array.Sort(energy, order);                                            // the quietest frames first (energy is reordered with them)
            Array.Clear(floor);
            for (int q = 0; q < quietest; q++)
            {
                var m = magnitude.AsSpan(order[q] * bins, bins);
                for (int b = 0; b < bins; b++)
                {
                    floor[b] += m[b] / quietest;
                }
            }

            for (int f = 0; f < frames; f++)
            {
                var m = magnitude.AsSpan(f * bins, bins);
                var g = gain.AsSpan(f * bins, bins);
                for (int b = 0; b < bins; b++)
                {
                    g[b] = m[b] > 0 ? Math.Clamp((m[b] - strength * floor[b]) / m[b], floorGain, 1f) : floorGain;
                }
            }

            for (int f = 0; f < frames; f++)
            {
                // Smoothed over the neighbouring frame on each side only: across frequencies a smoothed gain would wear
                // down the edges of every harmonic.
                int f0 = Math.Max(0, f - 1), f1 = Math.Min(frames - 1, f + 1);
                for (int b = 0; b < bins; b++)
                {
                    float sum = 0;
                    for (int ff = f0; ff <= f1; ff++)
                    {
                        sum += gain[ff * bins + b];
                    }

                    float smoothed = Math.Clamp(sum / (f1 - f0 + 1), floorGain, 1f);
                    real[f * bins + b] *= smoothed;
                    imaginary[f * bins + b] *= smoothed;
                }
            }

            var samples = Stft.Inverse(real, imaginary, frames, stft, length);
            samples.AsSpan(0, length).CopyTo(output.AsSpan(c * length, length));
        }

        return new AudioData(output, audio.Channels, audio.SampleRate);
    }
}

// The options "trim" and "normalize" share.
internal static class CleaningParameters
{
    public static readonly AudioProcessorParameter Silence = new("silence", "Silence below the loudest", AudioProcessorInput.Number)
    {
        Min = 0, MinExclusive = true, Step = 1, Unit = "dB", Optional = true, Default = "40", Help = "A frame this far under the loudest frame is silence",
    };

    public static readonly AudioProcessorParameter Hop = new("hop", "Frame", AudioProcessorInput.Number)
    {
        Min = 0, MinExclusive = true, Step = 0.005, Unit = "s", Optional = true, Default = "0.01", Help = "The frames the level is measured over",
    };
}
