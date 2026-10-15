// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;

namespace Idrak.Audio;

// "silence": the stretches between pauses. Each frame of HopSeconds gets its energy (the mean square of its samples,
// averaged over the channels) in one pass over each channel's samples in order; a frame more than SilenceDecibels below
// the loudest frame is silent; a run of silent frames of at least MinSilenceSeconds (or at either end) separates pieces,
// shorter pauses stay inside one. Each piece keeps PadSeconds of silence on both sides (at most half the pause to its
// neighbour); one longer than MaxSeconds is cut again at its quietest frame in the second half of the limit; one shorter
// than MinSeconds is left out.
internal sealed class SilenceSplitter : IAudioSplitter
{
    public string Name => AudioSplitters.Silence;

    public IReadOnlyList<AudioPiece> Split(AudioData audio, AudioSplitOptions options)
    {
        ArgumentNullException.ThrowIfNull(audio);
        ArgumentNullException.ThrowIfNull(options);
        options.Check();
        int length = audio.Length;
        if (length == 0)
        {
            return [];
        }

        int hop = Math.Max(1, (int)Math.Round(options.HopSeconds * audio.SampleRate));
        int frames = (length + hop - 1) / hop;
        double[] energy = ArrayPool<double>.Shared.Rent(frames);
        try
        {
            Energies(audio, hop, energy.AsSpan(0, frames));
            double loudest = 0;
            for (int f = 0; f < frames; f++)
            {
                loudest = Math.Max(loudest, energy[f]);
            }

            if (loudest <= 0)
            {
                return [];                                                       // all silence
            }

            double floor = loudest * Math.Pow(10, -options.SilenceDecibels / 10.0);
            int minSilence = Math.Max(1, (int)Math.Ceiling(options.MinSilenceSeconds * audio.SampleRate / hop));
            // The voiced stretches (frame ranges), joined across pauses shorter than the minimum.
            var stretches = new List<(int First, int Last)>();
            int f0 = -1, lastVoiced = -1;
            for (int f = 0; f < frames; f++)
            {
                if (energy[f] <= floor)
                {
                    continue;
                }

                if (f0 < 0)
                {
                    f0 = f;
                }
                else if (f - lastVoiced - 1 >= minSilence)
                {
                    stretches.Add((f0, lastVoiced));
                    f0 = f;
                }

                lastVoiced = f;
            }

            if (f0 >= 0)
            {
                stretches.Add((f0, lastVoiced));
            }

            int pad = (int)Math.Round(options.PadSeconds * audio.SampleRate);
            int min = (int)Math.Round(options.MinSeconds * audio.SampleRate), max = (int)Math.Round(options.MaxSeconds * audio.SampleRate);
            var pieces = new List<AudioPiece>(stretches.Count);
            for (int i = 0; i < stretches.Count; i++)
            {
                int start = stretches[i].First * hop, end = Math.Min(length, (stretches[i].Last + 1) * hop);
                int before = i == 0 ? start : (start - Math.Min(length, (stretches[i - 1].Last + 1) * hop)) / 2;
                int after = i == stretches.Count - 1 ? length - end : (stretches[i + 1].First * hop - end) / 2;
                start -= Math.Min(pad, before);
                end += Math.Min(pad, after);
                if (max > 0)
                {
                    while (end - start > max)
                    {
                        int cut = Quietest(energy, frames, hop, start + max / 2, start + max);
                        cut = cut > start ? cut : start + max;                  // a limit of a frame or less: cut at the limit
                        Keep(pieces, start, cut, min);
                        start = cut;
                    }
                }

                Keep(pieces, start, end, min);
            }

            return pieces;
        }
        finally
        {
            ArrayPool<double>.Shared.Return(energy);
        }
    }

    private static void Keep(List<AudioPiece> pieces, int start, int end, int min)
    {
        if (end > start && end - start >= min)
        {
            pieces.Add(new AudioPiece(start, end - start));
        }
    }

    // The frame boundary of the quietest whole frame starting in [from, to): where a long piece is cut again.
    private static int Quietest(double[] energy, int frames, int hop, int from, int to)
    {
        int first = Math.Min(frames - 1, (from + hop - 1) / hop), last = Math.Min(frames - 1, Math.Max(first, to / hop - 1));
        int best = first;
        for (int f = first + 1; f <= last; f++)
        {
            if (energy[f] < energy[best])
            {
                best = f;
            }
        }

        return Math.Clamp(best * hop, from, to);
    }

    // The mean square of each frame, averaged over the channels: each channel's samples walked once, in order.
    private static void Energies(AudioData audio, int hop, Span<double> energy)
    {
        energy.Clear();
        int length = audio.Length;
        for (int c = 0; c < audio.Channels; c++)
        {
            var samples = audio.Channel(c);
            for (int f = 0, at = 0; at < length; f++, at += hop)
            {
                var frame = samples.Slice(at, Math.Min(hop, length - at));
                double sum = 0;
                foreach (float s in frame)
                {
                    sum += (double)s * s;
                }

                energy[f] += sum / frame.Length;
            }
        }

        if (audio.Channels > 1)
        {
            for (int f = 0; f < energy.Length; f++)
            {
                energy[f] /= audio.Channels;
            }
        }
    }
}

// "seconds": pieces of MaxSeconds one after another; the last one, shorter, is kept when it holds at least MinSeconds.
internal sealed class SecondsSplitter : IAudioSplitter
{
    public string Name => AudioSplitters.Seconds;

    public IReadOnlyList<AudioPiece> Split(AudioData audio, AudioSplitOptions options)
    {
        ArgumentNullException.ThrowIfNull(audio);
        ArgumentNullException.ThrowIfNull(options);
        options.Check();
        if (!(options.MaxSeconds > 0))
        {
            throw new ArgumentException("The \"seconds\" splitter needs AudioSplitOptions.MaxSeconds (the length of a piece).", nameof(options));
        }

        int piece = Math.Max(1, (int)Math.Round(options.MaxSeconds * audio.SampleRate)), min = (int)Math.Round(options.MinSeconds * audio.SampleRate);
        var pieces = new List<AudioPiece>(audio.Length / piece + 1);
        for (int start = 0; start < audio.Length; start += piece)
        {
            int count = Math.Min(piece, audio.Length - start);
            if (count >= Math.Max(1, min))
            {
                pieces.Add(new AudioPiece(start, count));
            }
        }

        return pieces;
    }
}
