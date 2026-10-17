// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;

namespace Idrak.Audio;

// The level of a recording frame by frame: each hop's mean square of the channels' mean, in dB (silence -200), and the
// loudest's; "trim" and "normalize" read it (one pass over the samples, in order).
internal static class CleaningLevels
{
    public static (float[] Decibels, float Loudest, int Hop) Frames(AudioData audio, double hopSeconds)
    {
        int hop = Math.Max(1, (int)Math.Round(hopSeconds * audio.SampleRate));
        int length = audio.Length, channels = audio.Channels, frames = (length + hop - 1) / hop;
        var samples = audio.Samples;
        var decibels = new float[frames];
        float loudest = -200f;
        for (int f = 0; f < frames; f++)
        {
            int start = f * hop, end = Math.Min(length, start + hop);
            double sum = 0;
            if (channels == 1)
            {
                foreach (float x in samples.AsSpan(start, end - start))
                {
                    sum += (double)x * x;
                }
            }
            else
            {
                for (int i = start; i < end; i++)
                {
                    double mix = 0;
                    for (int c = 0; c < channels; c++)
                    {
                        mix += samples[c * length + i];
                    }

                    mix /= channels;
                    sum += mix * mix;
                }
            }

            double meanSquare = sum / (end - start);
            decibels[f] = meanSquare > 0 ? (float)(10 * Math.Log10(meanSquare)) : -200f;
            loudest = Math.Max(loudest, decibels[f]);
        }

        return (decibels, loudest, hop);
    }

    // y = g · x, the vector width at a time and a scalar tail.
    public static void Scale(ReadOnlySpan<float> x, float gain, Span<float> y)
    {
        int i = 0, width = Vector<float>.Count;
        var g = new Vector<float>(gain);
        for (; i + width <= x.Length; i += width)
        {
            (new Vector<float>(x.Slice(i, width)) * g).CopyTo(y.Slice(i, width));
        }

        for (; i < x.Length; i++)
        {
            y[i] = x[i] * gain;
        }
    }
}

// "highpass": a second-order Butterworth high-pass (the RBJ cookbook's, Q = 1/√2) run over each channel in order, in
// double; the DC offset and the rumble below the cutoff go, the voice above it is untouched in level.
internal sealed class HighPassCleaner : IAudioCleaner
{
    public string Name => AudioCleaners.HighPass;

    public AudioData Clean(AudioData audio, AudioCleaningOptions options)
    {
        double cutoff = options.HighPassHertz;
        if (cutoff <= 0 || audio.Length == 0)
        {
            return audio;
        }

        if (cutoff >= audio.SampleRate / 2.0)
        {
            throw new ArgumentException($"highpass: the cutoff {cutoff} Hz is not below half the rate ({audio.SampleRate} Hz).", nameof(options));
        }

        double w0 = 2 * Math.PI * cutoff / audio.SampleRate, cos = Math.Cos(w0), alpha = Math.Sin(w0) / Math.Sqrt(2);
        double a0 = 1 + alpha;
        double b0 = (1 + cos) / 2 / a0, b1 = -(1 + cos) / a0, b2 = b0, a1 = -2 * cos / a0, a2 = (1 - alpha) / a0;
        int length = audio.Length;
        var samples = audio.Samples;
        var output = GC.AllocateUninitializedArray<float>(samples.Length);
        for (int c = 0; c < audio.Channels; c++)
        {
            var x = samples.AsSpan(c * length, length);
            var y = output.AsSpan(c * length, length);
            double x1 = x[0], x2 = x[0], y1 = 0, y2 = 0;                        // settled on the first sample's level: no step at the start
            for (int i = 0; i < length; i++)
            {
                double xi = x[i];
                double yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
                y[i] = (float)yi;
                (x2, x1, y2, y1) = (x1, xi, y1, yi);
            }
        }

        return new AudioData(output, audio.Channels, audio.SampleRate);
    }
}

// "trim": the silence at both ends cut. A frame more than SilenceDecibels under the loudest is silence; the speech runs
// from the first frame that is not to the last, TrimPadSeconds kept on both sides. A silent recording stays as it is.
internal sealed class TrimCleaner : IAudioCleaner
{
    public string Name => AudioCleaners.Trim;

    public AudioData Clean(AudioData audio, AudioCleaningOptions options)
    {
        if (audio.Length == 0)
        {
            return audio;
        }

        var (decibels, loudest, hop) = CleaningLevels.Frames(audio, options.HopSeconds);
        float threshold = loudest - (float)options.SilenceDecibels;
        int first = Array.FindIndex(decibels, d => d >= threshold), last = Array.FindLastIndex(decibels, d => d >= threshold);
        if (first < 0)
        {
            return audio;
        }

        int pad = (int)Math.Round(options.TrimPadSeconds * audio.SampleRate);
        int start = Math.Max(0, first * hop - pad), end = Math.Min(audio.Length, (last + 1) * hop + pad);
        return start == 0 && end == audio.Length ? audio : audio.Slice(start, end - start);
    }
}

// "normalize": the speech level (the RMS over the frames that are not silence) brought to TargetDecibels by one gain over
// every channel, lowered when the peak would pass PeakCeiling. A silent recording stays as it is.
internal sealed class NormalizeCleaner : IAudioCleaner
{
    public string Name => AudioCleaners.Normalize;

    public AudioData Clean(AudioData audio, AudioCleaningOptions options)
    {
        if (audio.Length == 0)
        {
            return audio;
        }

        var (decibels, loudest, _) = CleaningLevels.Frames(audio, options.HopSeconds);
        float threshold = loudest - (float)options.SilenceDecibels;
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

        double level = 10 * Math.Log10(sum / count);
        double gain = Math.Pow(10, (options.TargetDecibels - level) / 20);
        var samples = audio.Samples;
        float peak = 0;
        foreach (float x in samples)
        {
            peak = Math.Max(peak, Math.Abs(x));
        }

        if (peak * gain > options.PeakCeiling)
        {
            gain = options.PeakCeiling / peak;
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

// "denoise": spectral gating over each channel. The spectrum (a Hann window of the largest power of two of samples
// under DenoiseWindowSeconds, a quarter of it the hop); the noise floor is each frequency's mean magnitude over the
// quietest NoiseQuantile share of the frames (the pauses: what is left when nobody speaks); a frame's gain at a frequency
// is (magnitude - DenoiseStrength · floor) / magnitude, no lower than DenoiseFloorDecibels and no higher than 1, smoothed
// over the neighbouring frame on each side (no musical noise); the gained spectrum goes back through the inverse STFT to
// the recording's exact length.
internal sealed class DenoiseCleaner : IAudioCleaner
{
    public string Name => AudioCleaners.Denoise;

    public AudioData Clean(AudioData audio, AudioCleaningOptions options)
    {
        int size = 16;
        while (size * 2 <= options.DenoiseWindowSeconds * audio.SampleRate)
        {
            size *= 2;
        }

        if (audio.Length < 2 * size)
        {
            return audio;
        }

        var stft = new AudioFeatureOptions { SampleRate = audio.SampleRate, WindowSize = size, HopSize = size / 4, Window = "hann", Center = true, PadSeconds = null };
        int length = audio.Length, frames = Stft.FramesFor(length, stft), bins = stft.Bins;
        float floorGain = (float)Math.Pow(10, options.DenoiseFloorDecibels / 20);
        float strength = (float)options.DenoiseStrength;
        var real = GC.AllocateUninitializedArray<float>(frames * bins);
        var imaginary = GC.AllocateUninitializedArray<float>(frames * bins);
        var magnitude = GC.AllocateUninitializedArray<float>(frames * bins);
        var gain = GC.AllocateUninitializedArray<float>(frames * bins);
        var energy = GC.AllocateUninitializedArray<float>(frames);
        var order = GC.AllocateUninitializedArray<int>(frames);
        var floor = new float[bins];
        var output = GC.AllocateUninitializedArray<float>(audio.Samples.Length);
        int quietest = Math.Clamp((int)Math.Round(options.NoiseQuantile * frames), 1, frames);
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
