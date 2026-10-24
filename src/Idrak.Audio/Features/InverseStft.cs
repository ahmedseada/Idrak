// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;

namespace Idrak.Audio;

/// <summary>
/// The complex transform and its inverse on the host (plan 21, S5: what Griffin-Lim iterates). <see cref="Inverse"/>
/// is <c>torch.istft</c>: each frame's one-sided spectrum back to its windowed samples, the frames overlap-added at
/// their hops, the window-square envelope divided out, the centre padding trimmed. Frames run in parallel over blocks;
/// the overlap-add runs in parallel over stretches of the output, each sample's sums through the frames in order, so
/// the sums are in a fixed order (rule 86) and nothing the size of the spectra is copied.
/// </summary>
public static partial class Stft
{
    /// <summary>
    /// The complex spectra of <paramref name="samples"/> on the host: the real parts into <paramref name="real"/>, the
    /// imaginary into <paramref name="imaginary"/>, each [frames, bins] (torch's sign: X[k] = Σ x[n] e^(-2πi n k / N)).
    /// </summary>
    public static unsafe void ComplexOnHost(ReadOnlySpan<float> samples, AudioFeatureOptions options, Span<float> real, Span<float> imaginary)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Check();
        int frames = FramesFor(samples.Length, options), bins = options.Bins, size = options.WindowSize, hop = options.HopSize;
        if (real.Length < (long)frames * bins || imaginary.Length < (long)frames * bins)
        {
            throw new ArgumentException($"{Math.Min(real.Length, imaginary.Length)} values hold fewer than {frames} x {bins}.", nameof(real));
        }

        var padded = Padded(samples, options);
        var window = Window(options.Window, size);
        var fft = Fft.For(size);
        fixed (float* realOut = real, imaginaryOut = imaginary)
        {
            nint outReal = (nint)realOut, outImaginary = (nint)imaginaryOut;    // written in place: no copy of the spectra (a span cannot cross into the body)
            HostParallel.For(frames, 64, (first, last) =>
            {
                var scratch = ArrayPool<double>.Shared.Rent(fft.ScratchLength + 2 * bins);   // once per block of frames
                try
                {
                    var work = scratch.AsSpan(0, fft.ScratchLength);
                    var re = scratch.AsSpan(fft.ScratchLength, bins);
                    var im = scratch.AsSpan(fft.ScratchLength + bins, bins);
                    for (int f = first; f < last; f++)
                    {
                        fft.Forward(padded.AsSpan(f * hop, size), window, re, im, work);
                        var rowRe = new Span<float>((float*)outReal + (long)f * bins, bins);
                        var rowIm = new Span<float>((float*)outImaginary + (long)f * bins, bins);
                        for (int k = 0; k < bins; k++)
                        {
                            rowRe[k] = (float)re[k];
                            rowIm[k] = (float)im[k];
                        }
                    }
                }
                finally
                {
                    ArrayPool<double>.Shared.Return(scratch);
                }
            });
        }
    }

    /// <summary>The samples a spectrogram of <paramref name="frames"/> frames gives back (the centre padding removed when centred).</summary>
    public static int SamplesFor(int frames, AudioFeatureOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(frames);
        int full = options.WindowSize + options.HopSize * (frames - 1);
        return options.Center ? full - 2 * (options.WindowSize / 2) : full;
    }

    /// <summary>
    /// The signal whose transform is <paramref name="real"/> + i <paramref name="imaginary"/> ([frames, bins] each), as
    /// <c>torch.istft</c> reconstructs it: the windowed frames overlap-added and divided by the window-square envelope,
    /// the centre padding trimmed, the result cut or zero-padded to <paramref name="length"/> when given (otherwise
    /// <see cref="SamplesFor"/>).
    /// </summary>
    /// <remarks>
    /// A sample at either end that no frame weights above zero (without centring, the first sample under a periodic
    /// window's zero) has nothing to recover and comes back as 0; torch refuses such a transform outright.
    /// </remarks>
    /// <exception cref="ArgumentException">The window-square envelope is zero between covered samples (the hop leaves a gap the window does not cover), or everywhere.</exception>
    public static unsafe float[] Inverse(ReadOnlySpan<float> real, ReadOnlySpan<float> imaginary, int frames, AudioFeatureOptions options, int? length = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Check();
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(frames);
        int bins = options.Bins, size = options.WindowSize, hop = options.HopSize;
        if (real.Length < (long)frames * bins || imaginary.Length < (long)frames * bins)
        {
            throw new ArgumentException($"{Math.Min(real.Length, imaginary.Length)} values hold fewer than {frames} x {bins}.", nameof(real));
        }

        var window = Window(options.Window, size);
        var fft = Fft.For(size);
        int full = size + hop * (frames - 1);
        int start = options.Center ? size / 2 : 0;
        int kept = options.Center ? full - 2 * (size / 2) : full;
        int target = length ?? kept;
        int copied = Math.Min(kept, target);
        const double Uncovered = 1e-11;
        int firstCovered = 0, lastCovered = copied - 1;
        while (firstCovered < copied && Envelope(window, hop, frames, start + firstCovered) <= Uncovered)
        {
            firstCovered++;
        }

        while (lastCovered >= firstCovered && Envelope(window, hop, frames, start + lastCovered) <= Uncovered)
        {
            lastCovered--;
        }

        if (copied > 0 && firstCovered == copied)
        {
            throw new ArgumentException($"The window ({options.Window}, {size}) with hop {hop} covers none of the {copied} samples; the transform cannot be inverted.", nameof(options));
        }

        var samples = new float[target];                                        // the uncovered ends stay 0
        int covered = lastCovered - firstCovered + 1, gap = int.MaxValue;
        if (covered > 0)
        {
            fixed (float* realIn = real, imaginaryIn = imaginary)
            {
                nint re = (nint)realIn, im = (nint)imaginaryIn;                 // read in place: no copy of the spectra
                // The overlap-add over stretches of the output, each on its own core: a stretch takes the frames that
                // reach it in their order (a frame reaching two stretches is transformed by both), so every sample's
                // sums add the frames in the same order as one pass over them would: the same bits (rule 86).
                HostParallel.For(covered, 8 * size, (first, last) =>
                {
                    int from = start + firstCovered + first, to = start + firstCovered + last;   // [from, to) of the full signal
                    int span = to - from;
                    var doubles = ArrayPool<double>.Shared.Rent(fft.ScratchLength + size + 2 * span);
                    try
                    {
                        var work = doubles.AsSpan(0, fft.ScratchLength);
                        var x = doubles.AsSpan(fft.ScratchLength, size);
                        var sum = doubles.AsSpan(fft.ScratchLength + size, span);
                        var envelope = doubles.AsSpan(fft.ScratchLength + size + span, span);
                        sum.Clear();
                        envelope.Clear();
                        int f0 = Math.Max(0, (from - size + hop) / hop), f1 = Math.Min(frames - 1, (to - 1) / hop);
                        for (int f = f0; f <= f1; f++)
                        {
                            int at = f * hop, n0 = Math.Max(0, from - at), n1 = Math.Min(size, to - at);
                            if (n0 >= n1)
                            {
                                continue;
                            }

                            // The one-sided spectrum back to its samples (irfft: the parts at 0 Hz and N / 2 taken as real).
                            fft.Inverse(new ReadOnlySpan<float>((float*)re + (long)f * bins, bins), new ReadOnlySpan<float>((float*)im + (long)f * bins, bins), x, work);
                            for (int n = n0; n < n1; n++)
                            {
                                sum[at + n - from] += (float)(x[n] * window[n]);         // the windowed sample as a float, as the frames were kept
                                envelope[at + n - from] += (double)window[n] * window[n];
                            }
                        }

                        for (int m = 0; m < span; m++)
                        {
                            double e = envelope[m];
                            if (e <= Uncovered)
                            {
                                InterlockedMin(ref gap, from - start + m);
                                continue;
                            }

                            samples[from - start + m] = (float)(sum[m] / e);
                        }
                    }
                    finally
                    {
                        ArrayPool<double>.Shared.Return(doubles);
                    }
                });
            }
        }

        if (gap != int.MaxValue)
        {
            throw new ArgumentException($"The window ({options.Window}, {size}) with hop {hop} leaves sample {gap} uncovered (envelope {Envelope(window, hop, frames, start + gap)}) "
                + "between covered samples; the transform cannot be inverted there.", nameof(options));
        }

        return samples;
    }

    // The window-square envelope at sample m of the full signal: the frames that reach it, in their order.
    private static double Envelope(float[] window, int hop, int frames, int m)
    {
        double e = 0;
        for (int f = Math.Max(0, (m - window.Length + hop) / hop); f <= Math.Min(frames - 1, m / hop); f++)
        {
            int n = m - f * hop;
            if (n >= 0 && n < window.Length)
            {
                e += (double)window[n] * window[n];
            }
        }

        return e;
    }

    private static void InterlockedMin(ref int target, int value)
    {
        int seen = Volatile.Read(ref target);
        while (value < seen)
        {
            int was = Interlocked.CompareExchange(ref target, value, seen);
            if (was == seen)
            {
                return;
            }

            seen = was;
        }
    }
}
