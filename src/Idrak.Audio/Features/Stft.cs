// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Collections.Concurrent;

namespace Idrak.Audio;

/// <summary>
/// The short-time Fourier transform of sound as power or magnitude spectra, [frames, bins] (bins = window / 2 + 1), as
/// <c>torch.stft</c> computes it (a periodic window, centred frames with reflect padding, one-sided). Two computations
/// give the same values: the host's (a mixed-radix FFT, frames in parallel) and the composed device path (the frames
/// unfolded by <c>Im2Col</c>, one matrix product with the windowed cosine and sine basis, the halves squared and added:
/// every step an operation each backend has, so it runs on every device with no kernel of its own).
/// </summary>
public static partial class Stft
{
    private static readonly ConcurrentDictionary<(Device Device, int Window, string Name), Tensor> Bases = new();
    private static readonly ConcurrentDictionary<int, FftPlan> Plans = new();

    /// <summary>The frames a signal of <paramref name="samples"/> samples gives (reflect padding when centred).</summary>
    public static int FramesFor(long samples, AudioFeatureOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        long padded = options.Center ? samples + options.WindowSize / 2 * 2 : samples;
        return padded < options.WindowSize ? 0 : (int)((padded - options.WindowSize) / options.HopSize + 1);
    }

    /// <summary>The window's values (periodic: the symmetric window of n + 1 points without its last).</summary>
    public static float[] Window(string name, int size)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(size, 2);
        var w = new float[size];
        for (int n = 0; n < size; n++)
        {
            double c = Math.Cos(2 * Math.PI * n / size);
            w[n] = (float)(name switch
            {
                "hann" => 0.5 - 0.5 * c,
                "hamming" => 0.54 - 0.46 * c,
                "rectangular" => 1.0,
                _ => throw new ArgumentException($"Window '{name}' is not known (hann, hamming, rectangular).", nameof(name)),
            });
        }

        return w;
    }

    /// <summary>The signal with its reflect padding (half a window at both ends) when centred, else itself.</summary>
    public static float[] Padded(ReadOnlySpan<float> samples, AudioFeatureOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (!options.Center)
        {
            return samples.ToArray();
        }

        int pad = options.WindowSize / 2;
        if (samples.Length <= pad)
        {
            throw new ArgumentException($"{samples.Length} samples are too few to reflect-pad by {pad}.", nameof(samples));
        }

        var padded = new float[samples.Length + 2 * pad];
        samples.CopyTo(padded.AsSpan(pad));
        for (int i = 1; i <= pad; i++)
        {
            padded[pad - i] = samples[i];                                        // the reflection leaves the edge sample out
            padded[pad + samples.Length - 1 + i] = samples[samples.Length - 1 - i];
        }

        return padded;
    }

    /// <summary>The power spectra of <paramref name="samples"/> on the host: [frames, bins] into <paramref name="power"/>.</summary>
    public static void PowerOnHost(ReadOnlySpan<float> samples, AudioFeatureOptions options, Span<float> power)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Check();
        int frames = FramesFor(samples.Length, options), bins = options.Bins, size = options.WindowSize, hop = options.HopSize;
        if (power.Length < (long)frames * bins)
        {
            throw new ArgumentException($"{power.Length} values hold fewer than {frames} x {bins}.", nameof(power));
        }

        var padded = Padded(samples, options);
        var window = Window(options.Window, size);
        var plan = Plans.GetOrAdd(size, FftPlan.Create);
        var target = GC.AllocateUninitializedArray<float>(frames * bins);
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
                        re[n] = window[n] * padded[at + n];                      // the window applied as the frame is copied (rule 73)
                        im[n] = 0;
                    }

                    plan.Transform(re, im, outRe, outIm);
                    var row = target.AsSpan(f * bins, bins);
                    for (int k = 0; k < bins; k++)
                    {
                        row[k] = (float)(outRe[k] * outRe[k] + outIm[k] * outIm[k]);
                    }
                }
            }
            finally
            {
                ArrayPool<double>.Shared.Return(scratch);
            }
        });
        target.AsSpan().CopyTo(power);
    }

    /// <summary>The power spectra of <paramref name="samples"/> on <paramref name="device"/> by the composed path: [frames, bins].</summary>
    public static Tensor Power(ReadOnlySpan<float> samples, AudioFeatureOptions options, Device device)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(device);
        options.Check();
        int size = options.WindowSize, bins = options.Bins;
        var padded = Padded(samples, options);
        int frames = FramesFor(samples.Length, options);
        if (frames == 0)
        {
            throw new ArgumentException($"{samples.Length} samples give no frame of {size}.", nameof(samples));
        }

        using var signal = Tensor.Empty([1, 1, 1, padded.Length], device, zeroed: false);
        signal.Load(padded);
        using var cols = signal.Im2Col(new ConvGeometry(1, 1, 1, padded.Length, 1, size, 1, options.HopSize, 0, 0));   // [frames, size]
        using var spectrum = cols.MatMul(Basis(options, device));                                                      // [frames, 2 · bins]: cos | -sin
        using var real = spectrum.Narrow(1, 0, bins);
        using var imaginary = spectrum.Narrow(1, bins, bins);
        using var realSquared = real.Square();
        using var imaginarySquared = imaginary.Square();
        return realSquared + imaginarySquared;
    }

    /// <summary>The magnitude spectra (the square root of <see cref="Power"/>).</summary>
    public static Tensor Magnitude(ReadOnlySpan<float> samples, AudioFeatureOptions options, Device device)
    {
        using var power = Power(samples, options, device);
        return power.Sqrt();
    }

    // The windowed DFT basis [size, 2 · bins]: column k is w[n] cos(2π n k / size), column bins + k is -w[n] sin(2π n k /
    // size) (torch's sign), built in double and stored as floats, once per device and window (not the caller's to dispose).
    internal static Tensor Basis(AudioFeatureOptions options, Device device)
    {
        return Bases.GetOrAdd((device, options.WindowSize, options.Window), key =>
        {
            int size = key.Window, bins = size / 2 + 1;
            var window = Window(key.Name, size);
            var values = new float[size * 2 * bins];
            for (int n = 0; n < size; n++)
            {
                for (int k = 0; k < bins; k++)
                {
                    double angle = 2 * Math.PI * n * k / size;
                    values[n * 2 * bins + k] = (float)(window[n] * Math.Cos(angle));
                    values[n * 2 * bins + bins + k] = (float)(-window[n] * Math.Sin(angle));
                }
            }

            var t = Tensor.Empty([size, 2 * bins], key.Device, zeroed: false, track: false);
            t.Load(values);
            return t;
        });
    }

    // A complex FFT of one size, mixed radix (2, 3, 5; any other prime factor by a direct sum), decimation in time,
    // butterflies in a fixed order: the same bits every run.
    private sealed class FftPlan
    {
        private readonly int _size;
        private readonly int[] _factors;
        private readonly double[] _cos, _sin;                                    // e^(-2πi j / size), j < size

        private FftPlan(int size, int[] factors)
        {
            _size = size;
            _factors = factors;
            _cos = new double[size];
            _sin = new double[size];
            for (int j = 0; j < size; j++)
            {
                _cos[j] = Math.Cos(2 * Math.PI * j / size);
                _sin[j] = -Math.Sin(2 * Math.PI * j / size);
            }
        }

        public static FftPlan Create(int size)
        {
            var factors = new List<int>();
            int rest = size;
            foreach (int radix in new[] { 4, 2, 3, 5 })
            {
                while (rest % radix == 0)
                {
                    factors.Add(radix);
                    rest /= radix;
                }
            }

            for (int p = 7; rest > 1; p += 2)
            {
                while (rest % p == 0)
                {
                    factors.Add(p);
                    rest /= p;
                }
            }

            return new FftPlan(size, [.. factors]);
        }

        /// <summary>X = FFT(x): the outputs into <paramref name="outRe"/> and <paramref name="outIm"/>.</summary>
        public void Transform(ReadOnlySpan<double> re, ReadOnlySpan<double> im, Span<double> outRe, Span<double> outIm)
        {
            var scratch = ArrayPool<double>.Shared.Rent(2 * _size);
            try
            {
                Recurse(re, im, 0, 1, outRe, outIm, _size, 0, scratch.AsSpan(0, _size), scratch.AsSpan(_size, _size));
            }
            finally
            {
                ArrayPool<double>.Shared.Return(scratch);
            }
        }

        // The n-point transform of x[offset + k·stride] into y[0 .. n): the r sub-transforms of n / r points, then the
        // r-point butterflies combining them with the twiddles of the full size (W_size^(j · size / n)).
        private void Recurse(ReadOnlySpan<double> re, ReadOnlySpan<double> im, int offset, int stride, Span<double> yRe, Span<double> yIm, int n, int depth,
            Span<double> tRe, Span<double> tIm)
        {
            if (n == 1)
            {
                yRe[0] = re[offset];
                yIm[0] = im[offset];
                return;
            }

            int r = _factors[depth], m = n / r;
            for (int q = 0; q < r; q++)
            {
                Recurse(re, im, offset + q * stride, stride * r, yRe.Slice(q * m, m), yIm.Slice(q * m, m), m, depth + 1, tRe, tIm);
            }

            // y[q · m + k] holds sub-transform q; X[k + s · m] = Σ_q W_n^(q k) W_r^(q s) y[q · m + k].
            int step = _size / n;                                                // W_n^j = W_size^(j · step)
            for (int k = 0; k < m; k++)
            {
                for (int q = 0; q < r; q++)
                {
                    int tw = (q * k * step) % _size;
                    double a = yRe[q * m + k], b = yIm[q * m + k];
                    tRe[q] = a * _cos[tw] - b * _sin[tw];
                    tIm[q] = a * _sin[tw] + b * _cos[tw];
                }

                for (int s = 0; s < r; s++)
                {
                    double sumRe = 0, sumIm = 0;
                    for (int q = 0; q < r; q++)
                    {
                        int tw = (q * s * m * step) % _size;                     // W_r^(q s) = W_size^(q s · size / r)
                        sumRe += tRe[q] * _cos[tw] - tIm[q] * _sin[tw];
                        sumIm += tRe[q] * _sin[tw] + tIm[q] * _cos[tw];
                    }

                    tRe[r + s] = sumRe;
                    tIm[r + s] = sumIm;
                }

                for (int s = 0; s < r; s++)
                {
                    yRe[k + s * m] = tRe[r + s];
                    yIm[k + s * m] = tIm[r + s];
                }
            }
        }
    }
}
