// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
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

// The cleaning stages (AudioCleaners) run the processors of their names (AudioProcessors), each step made from the
// stage's AudioCleaningOptions; the numbers are written round-trip ("R"), so the processor reads the very doubles.
internal static class CleaningSteps
{
    public static AudioData Run(string name, AudioData audio, string? value, params ReadOnlySpan<(string Key, double Value)> options)
    {
        var pairs = new KeyValuePair<string, string>[options.Length];
        for (int i = 0; i < options.Length; i++)
        {
            pairs[i] = KeyValuePair.Create(options[i].Key, Text(options[i].Value));
        }

        return AudioProcessors.Get(name).Apply(audio, new AudioProcessorStep(name, value, pairs));
    }

    public static string Text(double value) => value.ToString("R", CultureInfo.InvariantCulture);
}

// "highpass": the processor "highpass" at HighPassHertz; 0 leaves the sound as it is.
internal sealed class HighPassCleaner : IAudioCleaner
{
    public string Name => AudioCleaners.HighPass;

    public AudioData Clean(AudioData audio, AudioCleaningOptions options) => options.HighPassHertz <= 0 || audio.Length == 0
        ? audio
        : CleaningSteps.Run(AudioProcessors.HighPass, audio, CleaningSteps.Text(options.HighPassHertz));
}

// "trim": the processor "trim" with TrimPadSeconds, SilenceDecibels and HopSeconds.
internal sealed class TrimCleaner : IAudioCleaner
{
    public string Name => AudioCleaners.Trim;

    public AudioData Clean(AudioData audio, AudioCleaningOptions options) => CleaningSteps.Run(AudioProcessors.Trim, audio, CleaningSteps.Text(options.TrimPadSeconds),
        ("silence", options.SilenceDecibels), ("hop", options.HopSeconds));
}

// "normalize": the processor "normalize" with TargetDecibels, SilenceDecibels, HopSeconds and PeakCeiling.
internal sealed class NormalizeCleaner : IAudioCleaner
{
    public string Name => AudioCleaners.Normalize;

    public AudioData Clean(AudioData audio, AudioCleaningOptions options) => CleaningSteps.Run(AudioProcessors.Normalize, audio, CleaningSteps.Text(options.TargetDecibels),
        ("silence", options.SilenceDecibels), ("hop", options.HopSeconds), ("peak", options.PeakCeiling));
}

// "denoise": the processor "denoise" with DenoiseStrength, NoiseQuantile, DenoiseFloorDecibels and DenoiseWindowSeconds.
internal sealed class DenoiseCleaner : IAudioCleaner
{
    public string Name => AudioCleaners.Denoise;

    public AudioData Clean(AudioData audio, AudioCleaningOptions options) => CleaningSteps.Run(AudioProcessors.Denoise, audio, CleaningSteps.Text(options.DenoiseStrength),
        ("quantile", options.NoiseQuantile), ("floor", options.DenoiseFloorDecibels), ("window", options.DenoiseWindowSeconds));
}
