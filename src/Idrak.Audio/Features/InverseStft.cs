// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;

namespace Idrak.Audio;

/// <summary>
/// The complex transform and its inverse on the host (plan 21, S5: what Griffin-Lim iterates). <see cref="Inverse"/>
/// is <c>torch.istft</c>: each frame's one-sided spectrum back to its windowed samples, the frames overlap-added at
/// their hops, the window-square envelope divided out, the centre padding trimmed. Frames run in parallel over blocks;
/// the overlap-add goes through the frames in order, so the sums are in a fixed order (rule 86).
/// </summary>
public static partial class Stft
{
    /// <summary>
    /// The complex spectra of <paramref name="samples"/> on the host: the real parts into <paramref name="real"/>, the
    /// imaginary into <paramref name="imaginary"/>, each [frames, bins] (torch's sign: X[k] = Σ x[n] e^(-2πi n k / N)).
    /// </summary>
    public static void ComplexOnHost(ReadOnlySpan<float> samples, AudioFeatureOptions options, Span<float> real, Span<float> imaginary)
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
        var plan = Plans.GetOrAdd(size, FftPlan.Create);
        var outReal = GC.AllocateUninitializedArray<float>(frames * bins);
        var outImaginary = GC.AllocateUninitializedArray<float>(frames * bins);
        HostParallel.For(frames, 64, (first, last) =>
        {
            var scratch = ArrayPool<double>.Shared.Rent(4 * size);
            try
            {
                var re = scratch.AsSpan(0, size);
                var im = scratch.AsSpan(size, size);
                var outRe = scratch.AsSpan(2 * size, size);
                var outIm = scratch.AsSpan(3 * size, size);
                for (int f = first; f < last; f++)
                {
                    int at = f * hop;
                    for (int n = 0; n < size; n++)
                    {
                        re[n] = window[n] * padded[at + n];
                        im[n] = 0;
                    }

                    plan.Transform(re, im, outRe, outIm);
                    var rowRe = outReal.AsSpan(f * bins, bins);
                    var rowIm = outImaginary.AsSpan(f * bins, bins);
                    for (int k = 0; k < bins; k++)
                    {
                        rowRe[k] = (float)outRe[k];
                        rowIm[k] = (float)outIm[k];
                    }
                }
            }
            finally
            {
                ArrayPool<double>.Shared.Return(scratch);
            }
        });
        outReal.AsSpan().CopyTo(real);
        outImaginary.AsSpan().CopyTo(imaginary);
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
    public static float[] Inverse(ReadOnlySpan<float> real, ReadOnlySpan<float> imaginary, int frames, AudioFeatureOptions options, int? length = null)
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
        var plan = Plans.GetOrAdd(size, FftPlan.Create);
        var realCopy = real[..(frames * bins)].ToArray();                        // the spans cannot cross into the parallel body
        var imaginaryCopy = imaginary[..(frames * bins)].ToArray();
        var windowed = GC.AllocateUninitializedArray<float>(frames * size);     // each frame's windowed samples, [frames, size]
        HostParallel.For(frames, 64, (first, last) =>
        {
            var scratch = ArrayPool<double>.Shared.Rent(4 * size);
            try
            {
                var re = scratch.AsSpan(0, size);
                var im = scratch.AsSpan(size, size);
                var outRe = scratch.AsSpan(2 * size, size);
                var outIm = scratch.AsSpan(3 * size, size);
                for (int f = first; f < last; f++)
                {
                    // The full conjugate-symmetric spectrum, conjugated: its forward transform, conjugated and scaled by 1 / N, is the inverse.
                    var rowRe = realCopy.AsSpan(f * bins, bins);
                    var rowIm = imaginaryCopy.AsSpan(f * bins, bins);
                    for (int k = 0; k < bins; k++)
                    {
                        re[k] = rowRe[k];
                        im[k] = -rowIm[k];
                    }

                    for (int k = bins; k < size; k++)
                    {
                        re[k] = rowRe[size - k];
                        im[k] = rowIm[size - k];
                    }

                    plan.Transform(re, im, outRe, outIm);
                    var frame = windowed.AsSpan(f * size, size);
                    for (int n = 0; n < size; n++)
                    {
                        frame[n] = (float)(outRe[n] / size * window[n]);
                    }
                }
            }
            finally
            {
                ArrayPool<double>.Shared.Return(scratch);
            }
        });

        int full = size + hop * (frames - 1);
        var sum = new double[full];
        var envelope = new double[full];
        for (int f = 0; f < frames; f++)
        {
            int at = f * hop;
            var frame = windowed.AsSpan(f * size, size);
            for (int n = 0; n < size; n++)
            {
                sum[at + n] += frame[n];
                envelope[at + n] += (double)window[n] * window[n];
            }
        }

        int start = options.Center ? size / 2 : 0;
        int kept = options.Center ? full - 2 * (size / 2) : full;
        int target = length ?? kept;
        int copied = Math.Min(kept, target);
        const double Uncovered = 1e-11;
        int firstCovered = 0, lastCovered = copied - 1;
        while (firstCovered < copied && envelope[start + firstCovered] <= Uncovered)
        {
            firstCovered++;
        }

        while (lastCovered >= firstCovered && envelope[start + lastCovered] <= Uncovered)
        {
            lastCovered--;
        }

        if (copied > 0 && firstCovered == copied)
        {
            throw new ArgumentException($"The window ({options.Window}, {size}) with hop {hop} covers none of the {copied} samples; the transform cannot be inverted.", nameof(options));
        }

        var samples = new float[target];
        for (int n = firstCovered; n <= lastCovered; n++)                        // the uncovered ends stay 0
        {
            double e = envelope[start + n];
            if (e <= Uncovered)
            {
                throw new ArgumentException($"The window ({options.Window}, {size}) with hop {hop} leaves sample {n} uncovered (envelope {e}) between covered samples; the transform cannot be inverted there.", nameof(options));
            }

            samples[n] = (float)(sum[start + n] / e);
        }

        return samples;
    }
}
