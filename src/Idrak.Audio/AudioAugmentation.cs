// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;

namespace Idrak.Audio;

// "speed": the sound taken as if recorded at rate · factor and resampled back to its rate, so it lasts 1 / factor as long
// (the pitch moves with it), Kaldi's speed perturbation; a factor of 1 leaves it as it is.
internal sealed class SpeedAugmentation : IAudioAugmentation
{
    public string Name => AudioAugmentations.Speed;

    public AudioAugmentationStage Stage => AudioAugmentationStage.Sound;

    public AudioData Augment(AudioData audio, Random random, AudioAugmentationOptions options)
    {
        double factor = options.SpeedFactors[random.Next(options.SpeedFactors.Count)];
        int rate = (int)Math.Round(audio.SampleRate * factor);
        if (rate == audio.SampleRate)
        {
            return audio;
        }

        return Resampler.Resample(new AudioData(audio.Samples, audio.Channels, rate), audio.SampleRate);
    }
}

// "noise": white noise, or a stretch of a noise file, added at a signal-to-noise ratio drawn between the limits (the
// signal's power its mean square; a silent signal is left as it is).
internal sealed class NoiseAugmentation : IAudioAugmentation
{
    // Noise files decoded once per (path, rate), one channel.
    private static readonly ConcurrentDictionary<(string Path, int Rate), float[]> Decoded = new();

    public string Name => AudioAugmentations.Noise;

    public AudioAugmentationStage Stage => AudioAugmentationStage.Sound;

    public AudioData Augment(AudioData audio, Random random, AudioAugmentationOptions options)
    {
        var samples = audio.Samples;
        int length = audio.Length * audio.Channels;
        double power = 0;
        for (int i = 0; i < length; i++)
        {
            power += samples[i] * (double)samples[i];
        }

        power /= Math.Max(1, length);
        double snr = options.MinSnrDb + random.NextDouble() * (options.MaxSnrDb - options.MinSnrDb);
        if (power <= 0 || length == 0)
        {
            return audio;
        }

        var output = new float[length];
        if (options.NoiseFiles.Count == 0)
        {
            double scale = Math.Sqrt(power / Math.Pow(10, snr / 10));
            for (int i = 0; i < length; i += 2)
            {
                // Box-Muller: two normal draws a pair of uniforms.
                double u = 1 - random.NextDouble(), v = random.NextDouble();
                double radius = Math.Sqrt(-2 * Math.Log(u));
                output[i] = (float)(samples[i] + scale * radius * Math.Cos(2 * Math.PI * v));
                if (i + 1 < length)
                {
                    output[i + 1] = (float)(samples[i + 1] + scale * radius * Math.Sin(2 * Math.PI * v));
                }
            }

            return new AudioData(output, audio.Channels, audio.SampleRate);
        }

        string path = options.NoiseFiles[random.Next(options.NoiseFiles.Count)];
        var noise = Decoded.GetOrAdd((Path.GetFullPath(path), audio.SampleRate), key =>
        {
            var decoded = Resampler.Resample(AudioCodecs.Decode(key.Path), key.Rate);
            var mono = new float[decoded.Length];
            decoded.ToMono(mono);
            return mono;
        });
        if (noise.Length == 0)
        {
            return audio;
        }

        int offset = random.Next(noise.Length);
        double noisePower = 0;
        for (int i = 0; i < length; i++)
        {
            float n = noise[(offset + i) % noise.Length];
            noisePower += n * (double)n;
        }

        noisePower /= length;
        float gain = noisePower > 0 ? (float)Math.Sqrt(power / Math.Pow(10, snr / 10) / noisePower) : 0f;
        for (int i = 0; i < length; i++)
        {
            output[i] = samples[i] + gain * noise[(offset + i) % noise.Length];
        }

        return new AudioData(output, audio.Channels, audio.SampleRate);
    }
}

// "spec-mask": SpecAugment's masks: TimeMasks stretches of at most min(TimeMaskFrames, TimeMaskFraction · frames) frames and
// FrequencyMasks bands of at most FrequencyMaskBands mels, each width drawn uniformly from 0 to its limit, filled with the
// utterance's mean feature (so a mask reads as an average frame, not as silence).
internal sealed class SpecMaskAugmentation : IAudioAugmentation
{
    public string Name => AudioAugmentations.SpecMask;

    public AudioAugmentationStage Stage => AudioAugmentationStage.Features;

    public void Augment(Span<float> features, int frames, int mels, Random random, AudioAugmentationOptions options)
    {
        if (frames == 0 || mels == 0)
        {
            return;
        }

        double sum = 0;
        foreach (float f in features[..(frames * mels)])
        {
            sum += f;
        }

        float mean = (float)(sum / (frames * (double)mels));
        int widest = Math.Min(options.TimeMaskFrames, (int)(options.TimeMaskFraction * frames));
        for (int m = 0; m < options.TimeMasks; m++)
        {
            int width = random.Next(widest + 1);
            int start = random.Next(frames - width + 1);
            features.Slice(start * mels, width * mels).Fill(mean);
        }

        int bands = Math.Min(options.FrequencyMaskBands, mels);
        for (int m = 0; m < options.FrequencyMasks; m++)
        {
            int width = random.Next(bands + 1);
            int start = random.Next(mels - width + 1);
            if (width == 0)
            {
                continue;
            }

            for (int t = 0; t < frames; t++)
            {
                features.Slice(t * mels + start, width).Fill(mean);
            }
        }
    }
}
