// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Idrak.Audio;

// The library's one FFT, in double (the host STFT and its inverse, denoise, spectral-gate and the live spectral stages):
// N real samples to the N / 2 + 1 bins X[k] = Σ x[n] e^(-2πi n k / N) (numpy's rfft, torch's sign) and back (irfft:
// x[n] = (1 / N) Σ over the full conjugate-symmetric spectrum, the imaginary parts at 0 Hz and N / 2 left out). An even N
// runs as the complex FFT of N / 2 points z[t] = x[2t] + i x[2t + 1], split into the bins after (half the work of the
// real input taken as complex); an odd N as the complex FFT of N points. The complex FFT is a Stockham autosort
// (iterative, natural order in and out, no digit reversal) of radix 4, 2, 3, 5 and any other prime factor by a direct
// sum: each stage reads one half of the caller's scratch and writes the other, along the inputs in memory order, with
// its twiddles worked out when the plan is made (the loops only add to indices). One plan per size, shared and
// read-only; the scratch is the caller's (ScratchLength doubles), so a transform allocates nothing. Every butterfly adds
// in a fixed order and no product is fused into an add: the same bits every run.
internal sealed class Fft
{
    private static readonly ConcurrentDictionary<int, Fft> Plans = new();

    // cos and sin of 2π/3 and 2π/5, 4π/5 (the radix-3 and radix-5 butterflies), as literals so they are the same on every machine.
    private const double Sin3 = 0.866025403784438646763723170753;
    private const double Cos5A = 0.309016994374947424102293417183, Cos5B = -0.809016994374947424102293417183;
    private const double Sin5A = 0.951056516295153572116439333379, Sin5B = 0.587785252292473129168705954639;

    private readonly int _size, _points;                                        // N and the complex points M (N / 2 when N is even, else N)
    private readonly bool _packed;                                              // N even: two samples a complex point
    private readonly Stage[] _stages;
    private readonly double[] _splitRe, _splitIm;                               // e^(-2πi k / N), k ≤ M (N even)
    private readonly double _scale;                                             // 1 / N

    private Fft(int size)
    {
        _size = size;
        _packed = size % 2 == 0;
        _points = _packed ? size / 2 : size;
        _scale = 1.0 / size;
        var stages = new List<Stage>();
        int before = 1, rest = _points, largest = 0;
        foreach (int radix in Factors(_points))
        {
            int length = before * radix;
            var re = new double[before * (radix - 1)];
            var im = new double[re.Length];
            for (int j = 0; j < before; j++)
            {
                for (int q = 1; q < radix; q++)
                {
                    double angle = 2 * Math.PI * (q * j) / length;               // W_L^(q j), q j < L
                    re[j * (radix - 1) + q - 1] = Math.Cos(angle);
                    im[j * (radix - 1) + q - 1] = -Math.Sin(angle);
                }
            }

            double[] rootRe = [], rootIm = [];
            if (radix > 5)
            {
                largest = Math.Max(largest, radix);
                rootRe = new double[radix];
                rootIm = new double[radix];
                for (int e = 0; e < radix; e++)
                {
                    rootRe[e] = Math.Cos(2 * Math.PI * e / radix);
                    rootIm[e] = -Math.Sin(2 * Math.PI * e / radix);
                }
            }

            rest /= radix;
            stages.Add(new Stage(radix, before, rest, re, im, rootRe, rootIm));
            before = length;
        }

        _stages = [.. stages];
        ScratchLength = 4 * _points + 2 * largest;
        _splitRe = [];
        _splitIm = [];
        if (_packed)
        {
            _splitRe = new double[_points + 1];
            _splitIm = new double[_points + 1];
            for (int k = 0; k <= _points; k++)
            {
                _splitRe[k] = Math.Cos(2 * Math.PI * k / size);
                _splitIm[k] = -Math.Sin(2 * Math.PI * k / size);
            }
        }
    }

    /// <summary>The samples a transform takes.</summary>
    public int Size => _size;

    /// <summary>The bins it gives (N / 2 + 1).</summary>
    public int Bins => _size / 2 + 1;

    /// <summary>The doubles of scratch a transform needs.</summary>
    public int ScratchLength { get; }

    /// <summary>The plan for <paramref name="size"/> samples, made once and shared.</summary>
    public static Fft For(int size)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(size);
        return Plans.GetOrAdd(size, static n => new Fft(n));
    }

    /// <summary>X = rfft(x): <see cref="Size"/> samples to <see cref="Bins"/> real and imaginary parts (those at 0 Hz and N / 2 exactly 0).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Forward(ReadOnlySpan<double> x, Span<double> real, Span<double> imaginary, Span<double> scratch)
    {
        CheckForward(x.Length, real.Length, imaginary.Length, scratch.Length);
        int m = _points;
        var re = scratch[..m];
        var im = scratch.Slice(m, m);
        if (_packed)
        {
            for (int t = 0; t < m; t++)
            {
                re[t] = x[2 * t];
                im[t] = x[2 * t + 1];
            }
        }
        else
        {
            x[..m].CopyTo(re);
            im.Clear();
        }

        Unpack(scratch.Slice(Transform(scratch), 2 * _points), real, imaginary);
    }

    /// <summary>X = rfft(w · x): the frame windowed as it is read (each product in float, as the frame and the window are).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Forward(ReadOnlySpan<float> frame, ReadOnlySpan<float> window, Span<double> real, Span<double> imaginary, Span<double> scratch)
    {
        CheckForward(Math.Min(frame.Length, window.Length), real.Length, imaginary.Length, scratch.Length);
        int m = _points;
        var re = scratch[..m];
        var im = scratch.Slice(m, m);
        if (_packed)
        {
            for (int t = 0; t < m; t++)
            {
                re[t] = window[2 * t] * frame[2 * t];
                im[t] = window[2 * t + 1] * frame[2 * t + 1];
            }
        }
        else
        {
            for (int t = 0; t < m; t++)
            {
                re[t] = window[t] * frame[t];
            }

            im.Clear();
        }

        Unpack(scratch.Slice(Transform(scratch), 2 * _points), real, imaginary);
    }

    /// <summary>x = irfft(X): <see cref="Bins"/> real and imaginary parts to <see cref="Size"/> samples (scaled by 1 / N).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Inverse(ReadOnlySpan<double> real, ReadOnlySpan<double> imaginary, Span<double> x, Span<double> scratch) => InverseOf(real, imaginary, x, scratch);

    /// <summary>x = irfft(X) of float parts (each widened to double as it is read).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Inverse(ReadOnlySpan<float> real, ReadOnlySpan<float> imaginary, Span<double> x, Span<double> scratch) => InverseOf(real, imaginary, x, scratch);

    // The complex points' factors: fours first, then a two, threes, fives and the other primes in increasing order.
    private static List<int> Factors(int points)
    {
        var factors = new List<int>();
        int rest = points;
        foreach (int radix in (ReadOnlySpan<int>)[4, 2, 3, 5])
        {
            while (rest % radix == 0)
            {
                factors.Add(radix);
                rest /= radix;
            }
        }

        for (int p = 7; rest > 1; p += 2)
        {
            if ((long)p * p > rest)
            {
                p = rest;                                                       // what is left is prime
            }

            while (rest % p == 0)
            {
                factors.Add(p);
                rest /= p;
            }
        }

        return factors;
    }

    private void CheckForward(int samples, int real, int imaginary, int scratch)
    {
        if (samples < _size || real < Bins || imaginary < Bins || scratch < ScratchLength)
        {
            throw new ArgumentException($"An FFT of {_size} takes {_size} samples, {Bins} bins and {ScratchLength} doubles of scratch, not {samples}, {Math.Min(real, imaginary)} and {scratch}.");
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void InverseOf<T>(ReadOnlySpan<T> real, ReadOnlySpan<T> imaginary, Span<double> x, Span<double> scratch)
        where T : unmanaged, INumberBase<T>
    {
        if (real.Length < Bins || imaginary.Length < Bins || x.Length < _size || scratch.Length < ScratchLength)
        {
            throw new ArgumentException($"An inverse FFT of {_size} takes {Bins} bins, {_size} samples and {ScratchLength} doubles of scratch, not {Math.Min(real.Length, imaginary.Length)}, {x.Length} and {scratch.Length}.");
        }

        int m = _points;
        var re = scratch[..m];
        var im = scratch.Slice(m, m);
        if (_packed)
        {
            // Z[k] = E[k] + i O[k] W^-k (E and O the even and odd samples' spectra, each twice over here; the 1 / 2 is
            // in the scale), conjugated: the forward transform of the conjugate, conjugated, is the inverse.
            for (int k = 0; k < m; k++)
            {
                int b = m - k;
                double xr = double.CreateTruncating(real[k]), xi = k == 0 ? 0 : double.CreateTruncating(imaginary[k]);
                double cr = double.CreateTruncating(real[b]), ci = b == m ? 0 : -double.CreateTruncating(imaginary[b]);  // conj(X[M - k])
                double er = xr + cr, ei = xi + ci, dr = xr - cr, di = xi - ci;
                double wr = _splitRe[k], wi = -_splitIm[k];                     // W^-k = e^(+2πi k / N)
                double oddRe = dr * wr - di * wi, oddIm = dr * wi + di * wr;
                re[k] = er - oddIm;
                im[k] = -(ei + oddRe);
            }

            var z = scratch.Slice(Transform(scratch), 2 * m);
            double scale = _scale;
            for (int t = 0; t < m; t++)
            {
                x[2 * t] = z[t] * scale;
                x[2 * t + 1] = -z[m + t] * scale;
            }
        }
        else
        {
            // The full conjugate-symmetric spectrum, conjugated (the imaginary part at 0 Hz left out).
            int bins = Bins;
            re[0] = double.CreateTruncating(real[0]);
            im[0] = 0;
            for (int k = 1; k < bins; k++)
            {
                re[k] = double.CreateTruncating(real[k]);
                im[k] = -double.CreateTruncating(imaginary[k]);
            }

            for (int k = bins; k < m; k++)
            {
                re[k] = double.CreateTruncating(real[m - k]);
                im[k] = double.CreateTruncating(imaginary[m - k]);
            }

            var z = scratch.Slice(Transform(scratch), m);
            double scale = _scale;
            for (int n = 0; n < m; n++)
            {
                x[n] = z[n] * scale;
            }
        }
    }

    // The bins from the complex transform z (its M real parts, then its M imaginary). An even N's bin k
    // is E[k] + W^k O[k], E = (Z[k] + conj Z[M - k]) / 2 the even samples' spectrum and O = (Z[k] - conj Z[M - k]) / 2i
    // the odd ones'; 0 Hz and N / 2 are Z[0]'s parts' sum and difference.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Unpack(ReadOnlySpan<double> z, Span<double> real, Span<double> imaginary)
    {
        int m = _points;
        var zr = z[..m];
        var zi = z.Slice(m, m);
        if (!_packed)
        {
            int bins = Bins;
            zr[..bins].CopyTo(real);
            zi[..bins].CopyTo(imaginary);
            imaginary[0] = 0;
            return;
        }

        real[0] = zr[0] + zi[0];
        imaginary[0] = 0;
        real[m] = zr[0] - zi[0];
        imaginary[m] = 0;
        for (int k = 1; k < m; k++)
        {
            int b = m - k;
            double er = (zr[k] + zr[b]) * 0.5, ei = (zi[k] - zi[b]) * 0.5;
            double oddRe = (zi[k] + zi[b]) * 0.5, oddIm = (zr[b] - zr[k]) * 0.5;
            double wr = _splitRe[k], wi = _splitIm[k];
            real[k] = er + (wr * oddRe - wi * oddIm);
            imaginary[k] = ei + (wr * oddIm + wi * oddRe);
        }
    }

    // The complex transform of the M points in scratch[0, 2M) (real parts, then imaginary), stage by stage between the
    // two halves of scratch[0, 4M); the offset of the half holding the result.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private int Transform(Span<double> scratch)
    {
        int m = _points, from = 0, to = 2 * m;
        ref double origin = ref MemoryMarshal.GetReference(scratch);
        var temp = scratch[(4 * m)..];
        foreach (var stage in _stages)
        {
            ref double ar = ref Unsafe.Add(ref origin, from);
            ref double br = ref Unsafe.Add(ref origin, to);
            switch (stage.Radix)
            {
                case 4: Radix4(stage, ref ar, ref Unsafe.Add(ref ar, m), ref br, ref Unsafe.Add(ref br, m), m); break;
                case 2: Radix2(stage, ref ar, ref Unsafe.Add(ref ar, m), ref br, ref Unsafe.Add(ref br, m), m); break;
                case 3: Radix3(stage, ref ar, ref Unsafe.Add(ref ar, m), ref br, ref Unsafe.Add(ref br, m), m); break;
                case 5: Radix5(stage, ref ar, ref Unsafe.Add(ref ar, m), ref br, ref Unsafe.Add(ref br, m), m); break;
                default: RadixAny(stage, ref ar, ref Unsafe.Add(ref ar, m), ref br, ref Unsafe.Add(ref br, m), m, temp); break;
            }

            (from, to) = (to, from);
        }

        return from;
    }

    // Each stage: L = before · p, r = M / L. Input point k + r q + r p j (sub-transform k + r q of length before, its bin
    // j) times W_L^(q j) goes into the p-point DFT whose output s is bin j + before · s of sub-transform k of length L, at
    // k + r j + (M / p) s. Rows k run along memory on both sides.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Radix2(Stage stage, ref double ar, ref double ai, ref double br, ref double bi, int m)
    {
        int before = stage.Before, r = stage.Rest, half = m >> 1;
        ref double wr = ref MemoryMarshal.GetArrayDataReference(stage.Re);
        ref double wi = ref MemoryMarshal.GetArrayDataReference(stage.Im);
        for (int j = 0; j < before; j++)
        {
            double w1r = Unsafe.Add(ref wr, j), w1i = Unsafe.Add(ref wi, j);
            int i = 2 * r * j, o = r * j;
            for (int k = 0; k < r; k++, i++, o++)
            {
                double x0r = Unsafe.Add(ref ar, i), x0i = Unsafe.Add(ref ai, i);
                double yr = Unsafe.Add(ref ar, i + r), yi = Unsafe.Add(ref ai, i + r);
                double x1r = yr * w1r - yi * w1i, x1i = yr * w1i + yi * w1r;
                Unsafe.Add(ref br, o) = x0r + x1r;
                Unsafe.Add(ref bi, o) = x0i + x1i;
                Unsafe.Add(ref br, o + half) = x0r - x1r;
                Unsafe.Add(ref bi, o + half) = x0i - x1i;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Radix4(Stage stage, ref double ar, ref double ai, ref double br, ref double bi, int m)
    {
        int before = stage.Before, r = stage.Rest, quarter = m >> 2;
        ref double wr = ref MemoryMarshal.GetArrayDataReference(stage.Re);
        ref double wi = ref MemoryMarshal.GetArrayDataReference(stage.Im);
        for (int j = 0; j < before; j++)
        {
            double w1r = Unsafe.Add(ref wr, 3 * j), w1i = Unsafe.Add(ref wi, 3 * j);
            double w2r = Unsafe.Add(ref wr, 3 * j + 1), w2i = Unsafe.Add(ref wi, 3 * j + 1);
            double w3r = Unsafe.Add(ref wr, 3 * j + 2), w3i = Unsafe.Add(ref wi, 3 * j + 2);
            int i = 4 * r * j, o = r * j;
            for (int k = 0; k < r; k++, i++, o++)
            {
                double x0r = Unsafe.Add(ref ar, i), x0i = Unsafe.Add(ref ai, i);
                double yr = Unsafe.Add(ref ar, i + r), yi = Unsafe.Add(ref ai, i + r);
                double x1r = yr * w1r - yi * w1i, x1i = yr * w1i + yi * w1r;
                yr = Unsafe.Add(ref ar, i + 2 * r);
                yi = Unsafe.Add(ref ai, i + 2 * r);
                double x2r = yr * w2r - yi * w2i, x2i = yr * w2i + yi * w2r;
                yr = Unsafe.Add(ref ar, i + 3 * r);
                yi = Unsafe.Add(ref ai, i + 3 * r);
                double x3r = yr * w3r - yi * w3i, x3i = yr * w3i + yi * w3r;
                double t0r = x0r + x2r, t0i = x0i + x2i, t1r = x0r - x2r, t1i = x0i - x2i;
                double t2r = x1r + x3r, t2i = x1i + x3i, t3r = x1r - x3r, t3i = x1i - x3i;
                Unsafe.Add(ref br, o) = t0r + t2r;                              // y0 = t0 + t2
                Unsafe.Add(ref bi, o) = t0i + t2i;
                Unsafe.Add(ref br, o + quarter) = t1r + t3i;                    // y1 = t1 - i t3
                Unsafe.Add(ref bi, o + quarter) = t1i - t3r;
                Unsafe.Add(ref br, o + 2 * quarter) = t0r - t2r;                // y2 = t0 - t2
                Unsafe.Add(ref bi, o + 2 * quarter) = t0i - t2i;
                Unsafe.Add(ref br, o + 3 * quarter) = t1r - t3i;                // y3 = t1 + i t3
                Unsafe.Add(ref bi, o + 3 * quarter) = t1i + t3r;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Radix3(Stage stage, ref double ar, ref double ai, ref double br, ref double bi, int m)
    {
        int before = stage.Before, r = stage.Rest, third = m / 3;
        ref double wr = ref MemoryMarshal.GetArrayDataReference(stage.Re);
        ref double wi = ref MemoryMarshal.GetArrayDataReference(stage.Im);
        for (int j = 0; j < before; j++)
        {
            double w1r = Unsafe.Add(ref wr, 2 * j), w1i = Unsafe.Add(ref wi, 2 * j);
            double w2r = Unsafe.Add(ref wr, 2 * j + 1), w2i = Unsafe.Add(ref wi, 2 * j + 1);
            int i = 3 * r * j, o = r * j;
            for (int k = 0; k < r; k++, i++, o++)
            {
                double x0r = Unsafe.Add(ref ar, i), x0i = Unsafe.Add(ref ai, i);
                double yr = Unsafe.Add(ref ar, i + r), yi = Unsafe.Add(ref ai, i + r);
                double x1r = yr * w1r - yi * w1i, x1i = yr * w1i + yi * w1r;
                yr = Unsafe.Add(ref ar, i + 2 * r);
                yi = Unsafe.Add(ref ai, i + 2 * r);
                double x2r = yr * w2r - yi * w2i, x2i = yr * w2i + yi * w2r;
                double t1r = x1r + x2r, t1i = x1i + x2i, t2r = x1r - x2r, t2i = x1i - x2i;
                double mr = x0r - 0.5 * t1r, mi = x0i - 0.5 * t1i;              // x0 + cos(2π/3) t1
                double nr = Sin3 * t2i, ni = -(Sin3 * t2r);                     // -i sin(2π/3) t2
                Unsafe.Add(ref br, o) = x0r + t1r;
                Unsafe.Add(ref bi, o) = x0i + t1i;
                Unsafe.Add(ref br, o + third) = mr + nr;
                Unsafe.Add(ref bi, o + third) = mi + ni;
                Unsafe.Add(ref br, o + 2 * third) = mr - nr;
                Unsafe.Add(ref bi, o + 2 * third) = mi - ni;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Radix5(Stage stage, ref double ar, ref double ai, ref double br, ref double bi, int m)
    {
        int before = stage.Before, r = stage.Rest, fifth = m / 5;
        ref double wr = ref MemoryMarshal.GetArrayDataReference(stage.Re);
        ref double wi = ref MemoryMarshal.GetArrayDataReference(stage.Im);
        for (int j = 0; j < before; j++)
        {
            int at = 4 * j;
            double w1r = Unsafe.Add(ref wr, at), w1i = Unsafe.Add(ref wi, at);
            double w2r = Unsafe.Add(ref wr, at + 1), w2i = Unsafe.Add(ref wi, at + 1);
            double w3r = Unsafe.Add(ref wr, at + 2), w3i = Unsafe.Add(ref wi, at + 2);
            double w4r = Unsafe.Add(ref wr, at + 3), w4i = Unsafe.Add(ref wi, at + 3);
            int i = 5 * r * j, o = r * j;
            for (int k = 0; k < r; k++, i++, o++)
            {
                double x0r = Unsafe.Add(ref ar, i), x0i = Unsafe.Add(ref ai, i);
                double yr = Unsafe.Add(ref ar, i + r), yi = Unsafe.Add(ref ai, i + r);
                double x1r = yr * w1r - yi * w1i, x1i = yr * w1i + yi * w1r;
                yr = Unsafe.Add(ref ar, i + 2 * r);
                yi = Unsafe.Add(ref ai, i + 2 * r);
                double x2r = yr * w2r - yi * w2i, x2i = yr * w2i + yi * w2r;
                yr = Unsafe.Add(ref ar, i + 3 * r);
                yi = Unsafe.Add(ref ai, i + 3 * r);
                double x3r = yr * w3r - yi * w3i, x3i = yr * w3i + yi * w3r;
                yr = Unsafe.Add(ref ar, i + 4 * r);
                yi = Unsafe.Add(ref ai, i + 4 * r);
                double x4r = yr * w4r - yi * w4i, x4i = yr * w4i + yi * w4r;
                double a1r = x1r + x4r, a1i = x1i + x4i, b1r = x1r - x4r, b1i = x1i - x4i;
                double a2r = x2r + x3r, a2i = x2i + x3i, b2r = x2r - x3r, b2i = x2i - x3i;
                double p1r = x0r + Cos5A * a1r + Cos5B * a2r, p1i = x0i + Cos5A * a1i + Cos5B * a2i;   // y1 and y4's common part
                double p2r = x0r + Cos5B * a1r + Cos5A * a2r, p2i = x0i + Cos5B * a1i + Cos5A * a2i;   // y2 and y3's
                double s1r = Sin5A * b1r + Sin5B * b2r, s1i = Sin5A * b1i + Sin5B * b2i;               // y1 = p1 - i s1, y4 = p1 + i s1
                double s2r = Sin5B * b1r - Sin5A * b2r, s2i = Sin5B * b1i - Sin5A * b2i;               // y2 = p2 - i s2, y3 = p2 + i s2
                Unsafe.Add(ref br, o) = x0r + a1r + a2r;
                Unsafe.Add(ref bi, o) = x0i + a1i + a2i;
                Unsafe.Add(ref br, o + fifth) = p1r + s1i;
                Unsafe.Add(ref bi, o + fifth) = p1i - s1r;
                Unsafe.Add(ref br, o + 2 * fifth) = p2r + s2i;
                Unsafe.Add(ref bi, o + 2 * fifth) = p2i - s2r;
                Unsafe.Add(ref br, o + 3 * fifth) = p2r - s2i;
                Unsafe.Add(ref bi, o + 3 * fifth) = p2i + s2r;
                Unsafe.Add(ref br, o + 4 * fifth) = p1r - s1i;
                Unsafe.Add(ref bi, o + 4 * fifth) = p1i + s1r;
            }
        }
    }

    // Any other prime p: the twiddled inputs into temp, then each output the direct sum over them in order of q.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void RadixAny(Stage stage, ref double ar, ref double ai, ref double br, ref double bi, int m, Span<double> temp)
    {
        int p = stage.Radix, before = stage.Before, r = stage.Rest, part = m / p;
        var tr = temp[..p];
        var ti = temp.Slice(p, p);
        var rootRe = stage.RootRe;
        var rootIm = stage.RootIm;
        var wr = stage.Re;
        var wi = stage.Im;
        for (int j = 0; j < before; j++)
        {
            int at = (p - 1) * j;
            int i = p * r * j, o = r * j;
            for (int k = 0; k < r; k++, i++, o++)
            {
                tr[0] = Unsafe.Add(ref ar, i);
                ti[0] = Unsafe.Add(ref ai, i);
                for (int q = 1; q < p; q++)
                {
                    double yr = Unsafe.Add(ref ar, i + q * r), yi = Unsafe.Add(ref ai, i + q * r);
                    double w1r = wr[at + q - 1], w1i = wi[at + q - 1];
                    tr[q] = yr * w1r - yi * w1i;
                    ti[q] = yr * w1i + yi * w1r;
                }

                for (int s = 0; s < p; s++)
                {
                    double sr = tr[0], si = ti[0];
                    int e = 0;
                    for (int q = 1; q < p; q++)
                    {
                        e += s;
                        if (e >= p)
                        {
                            e -= p;                                             // e = q s mod p
                        }

                        sr += tr[q] * rootRe[e] - ti[q] * rootIm[e];
                        si += tr[q] * rootIm[e] + ti[q] * rootRe[e];
                    }

                    Unsafe.Add(ref br, o + s * part) = sr;
                    Unsafe.Add(ref bi, o + s * part) = si;
                }
            }
        }
    }

    // One stage of the complex FFT: its radix p, the length before it (L / p), the rows r = M / L, W_L^(q j) at
    // [j (p - 1) + q - 1], and for a radix past 5 the p roots W_p^e.
    private sealed class Stage(int radix, int before, int rest, double[] re, double[] im, double[] rootRe, double[] rootIm)
    {
        public int Radix { get; } = radix;

        public int Before { get; } = before;

        public int Rest { get; } = rest;

        public double[] Re { get; } = re;

        public double[] Im { get; } = im;

        public double[] RootRe { get; } = rootRe;

        public double[] RootIm { get; } = rootIm;
    }
}
