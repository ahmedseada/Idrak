// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using Idrak.Audio;
using Idrak.Audio.Abstractions;

// Plan 31, P0: the four cleaning stages as processors, the same samples as before bit for bit; the helpers the audio
// processing tests share.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] AudioProcessingGroup =
    [
        ("audio processing: the cleaners run through their processors give the samples of the cleaners before plan 31 bit for bit (each stage, mono and stereo, a tone in bursts under noise with a DC offset)", AudioProcessingCleanersUnchanged),
    ];

    internal static void AudioProcessingCleanersUnchanged(Device device)
    {
        _ = device;
        var options = new AudioCleaningOptions { HighPassHertz = 75, TargetDecibels = -17.5, DenoiseStrength = 1.75, NoiseQuantile = 0.15, TrimPadSeconds = 0.07 };
        foreach (var settings in new[] { new AudioCleaningOptions(), options })
        {
            foreach (int channels in new[] { 1, 2 })
            {
                var sound = ProcessingTone(16000, 1.5, channels, noise: 0.02f, dc: 0.05f, bursts: true);
                foreach (var (name, cleaner) in new (string, IAudioCleaner)[] { ("highpass", new HighPassCleaner()), ("denoise", new DenoiseCleaner()), ("trim", new TrimCleaner()), ("normalize", new NormalizeCleaner()) })
                {
                    var now = cleaner.Clean(sound, settings);
                    var then = CleanersBefore31.Clean(name, sound, settings);
                    Check(now.Length == then.Length && now.Channels == then.Channels && ProcessingSame(now.Samples, then.Samples),
                        $"{name}, {channels} channel(s): {now.Length} samples, the cleaner's before plan 31 {then.Length}, the same bits");
                }
            }
        }
    }

    // A tone of 440 Hz at half scale, `channels` copies (the second a little quieter), optionally in bursts (0.2 s on,
    // 0.2 s off), with white noise and a DC offset.
    internal static AudioData ProcessingTone(int rate, double seconds, int channels, float noise = 0, float dc = 0, bool bursts = false)
    {
        int length = (int)Math.Round(seconds * rate);
        var random = new Random(31);
        var samples = new float[channels * length];
        for (int c = 0; c < channels; c++)
        {
            for (int i = 0; i < length; i++)
            {
                bool on = !bursts || (i / (rate / 5)) % 2 == 0;
                double tone = on ? 0.5 * Math.Sin(2 * Math.PI * 440 * i / rate) / (1 + c) : 0;
                samples[c * length + i] = (float)(tone + noise * (2 * random.NextDouble() - 1) + dc);
            }
        }

        return new AudioData(samples, channels, rate);
    }

    // The sound in blocks of `size` (copies: a block is the stage's).
    internal static IEnumerable<AudioData> ProcessingBlocks(AudioData sound, int size)
    {
        for (int at = 0; at < sound.Length; at += size)
        {
            yield return sound.Slice(at, Math.Min(size, sound.Length - at));
        }
    }

    internal static AudioData ProcessingJoin(IReadOnlyList<AudioData> pieces, int channels, int rate)
    {
        int length = pieces.Sum(p => p.Length);
        var samples = new float[channels * length];
        for (int c = 0; c < channels; c++)
        {
            int at = c * length;
            foreach (var piece in pieces)
            {
                piece.Channel(c).CopyTo(samples.AsSpan(at, piece.Length));
                at += piece.Length;
            }
        }

        return new AudioData(samples, channels, rate);
    }

    // The sound with its first `latency` samples (a stage's shift) left out: aligned with what Apply gives.
    internal static AudioData ProcessingAligned(AudioData sound, int latency) => latency == 0 ? sound : sound.Slice(latency, sound.Length - latency);

    // Planar samples [channels, n] with each channel's first `skip` left out.
    internal static float[] ProcessingAligned(float[] planar, int channels, int skip)
    {
        int length = planar.Length / channels, kept = length - skip;
        var aligned = new float[channels * kept];
        for (int c = 0; c < channels; c++)
        {
            planar.AsSpan(c * length + skip, kept).CopyTo(aligned.AsSpan(c * kept));
        }

        return aligned;
    }

    // Bit for bit: the floats' bits, not their values (so -0 and 0 differ and NaN equals itself).
    internal static bool ProcessingSame(float[] a, float[] b) => MemoryMarshal.Cast<float, int>(a.AsSpan()).SequenceEqual(MemoryMarshal.Cast<float, int>(b.AsSpan()));

    // The cleaning stages as they were before plan 31 (src/Idrak.Audio/AudioCleaning.cs at d957cd00), kept verbatim as the
    // reference the processors must match bit for bit.
    internal static class CleanersBefore31
    {
        public static AudioData Clean(string name, AudioData audio, AudioCleaningOptions options) => name switch
        {
            "highpass" => HighPass(audio, options),
            "trim" => Trim(audio, options),
            "normalize" => Normalize(audio, options),
            "denoise" => Denoise(audio, options),
            _ => throw new ArgumentException(name),
        };

        private static (float[] Decibels, float Loudest, int Hop) Frames(AudioData audio, double hopSeconds)
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

        private static void Scale(ReadOnlySpan<float> x, float gain, Span<float> y)
        {
            int i = 0, width = System.Numerics.Vector<float>.Count;
            var g = new System.Numerics.Vector<float>(gain);
            for (; i + width <= x.Length; i += width)
            {
                (new System.Numerics.Vector<float>(x.Slice(i, width)) * g).CopyTo(y.Slice(i, width));
            }

            for (; i < x.Length; i++)
            {
                y[i] = x[i] * gain;
            }
        }

        private static AudioData HighPass(AudioData audio, AudioCleaningOptions options)
        {
            double cutoff = options.HighPassHertz;
            if (cutoff <= 0 || audio.Length == 0)
            {
                return audio;
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
                double x1 = x[0], x2 = x[0], y1 = 0, y2 = 0;
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

        private static AudioData Trim(AudioData audio, AudioCleaningOptions options)
        {
            if (audio.Length == 0)
            {
                return audio;
            }

            var (decibels, loudest, hop) = Frames(audio, options.HopSeconds);
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

        private static AudioData Normalize(AudioData audio, AudioCleaningOptions options)
        {
            if (audio.Length == 0)
            {
                return audio;
            }

            var (decibels, loudest, _) = Frames(audio, options.HopSeconds);
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
            Scale(samples, (float)gain, output);
            return new AudioData(output, audio.Channels, audio.SampleRate);
        }

        private static AudioData Denoise(AudioData audio, AudioCleaningOptions options)
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
            var real = new float[frames * bins];
            var imaginary = new float[frames * bins];
            var magnitude = new float[frames * bins];
            var gain = new float[frames * bins];
            var energy = new float[frames];
            var order = new int[frames];
            var floor = new float[bins];
            var output = new float[audio.Samples.Length];
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

                Array.Sort(energy, order);
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
}
