// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.
// Ported from xiph's RNNoise v0.1.1 (src/kiss_fft.c, src/_kiss_fft_guts.h: Copyright (c) 2003-2004 Mark Borgerding,
// 2005-2007 Xiph.Org Foundation; BSD-3-Clause, its text in README.md in this folder).

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Idrak.Audio.Plugins.RnNoise;

/// <summary>A complex value of KISS FFT (its <c>kiss_fft_cpx</c>): real and imaginary parts in float.</summary>
internal struct RnNoiseComplex
{
    public float R;
    public float I;
}

/// <summary>
/// RNNoise's FFT: KISS FFT in float (the radix 2, 3, 4 and 5 butterflies, the factors reversed so the radix 4 comes
/// last), forward with the 1/n scale as RNNoise builds it. Every product and sum is the C code's, in its order and in
/// float, so the transform gives its values. Immutable once made (one per size, shared).
/// </summary>
internal sealed class RnNoiseFft
{
    private const int MaxFactors = 8;
    private readonly RnNoiseComplex[] _twiddles;
    private readonly short[] _bitrev;
    private readonly short[] _factors = new short[2 * MaxFactors];
    private readonly float _scale;

    /// <summary>The transform of <paramref name="size"/> points (sizes whose prime factors are 2, 3 and 5).</summary>
    public RnNoiseFft(int size)
    {
        Size = size;
        _scale = 1f / size;
        _twiddles = new RnNoiseComplex[size];
        for (int i = 0; i < size; i++)
        {
            const double Pi = 3.14159265358979323846264338327;
            double phase = (-2 * Pi / size) * i;
            _twiddles[i] = new RnNoiseComplex { R = (float)Math.Cos(phase), I = (float)Math.Sin(phase) };
        }

        if (!Factor(size, _factors))
        {
            throw new ArgumentException($"KISS FFT takes sizes of factors 2, 3 and 5, not {size}.", nameof(size));
        }

        _bitrev = new short[size];
        BitReverse(0, 0, 1, 1, 0);
    }

    /// <summary>The number of points.</summary>
    public int Size { get; }

    /// <summary>The forward transform of <paramref name="input"/> into <paramref name="output"/> (another buffer), scaled by 1/n.</summary>
    public void Forward(ReadOnlySpan<RnNoiseComplex> input, Span<RnNoiseComplex> output)
    {
        if (input.Length < Size || output.Length < Size)
        {
            throw new ArgumentException($"A {Size}-point transform takes {Size} values in and out, not {input.Length} and {output.Length}.");
        }

        var bitrev = _bitrev.AsSpan(0, Size);
        for (int i = 0; i < bitrev.Length; i++)
        {
            var x = input[i];
            ref var o = ref output[bitrev[i]];
            o.R = _scale * x.R;
            o.I = _scale * x.I;
        }

        Transform(output);
    }

    // kf_factor: powers of 4, then 2, then the odd primes; the order reversed so the radix 4 is last.
    private static bool Factor(int n, short[] factors)
    {
        int p = 4, stages = 0, whole = n;
        do
        {
            while (n % p != 0)
            {
                p = p switch { 4 => 2, 2 => 3, _ => p + 2 };
                if (p > 32000 || (long)p * p > n)
                {
                    p = n;
                }
            }

            n /= p;
            if (p > 5)
            {
                return false;
            }

            factors[2 * stages] = (short)p;
            if (p == 2 && stages > 1)
            {
                factors[2 * stages] = 4;
                factors[2] = 2;
            }

            stages++;
        }
        while (n > 1);

        n = whole;
        for (int i = 0; i < stages / 2; i++)
        {
            (factors[2 * i], factors[2 * (stages - i - 1)]) = (factors[2 * (stages - i - 1)], factors[2 * i]);
        }

        for (int i = 0; i < stages; i++)
        {
            n /= factors[2 * i];
            factors[2 * i + 1] = (short)n;
        }

        return true;
    }

    // compute_bitrev_table, with offsets into _bitrev and _factors for its pointers.
    private void BitReverse(int fout, int f, int fstride, int inStride, int factor)
    {
        int p = _factors[factor], m = _factors[factor + 1];
        if (m == 1)
        {
            for (int j = 0; j < p; j++)
            {
                _bitrev[f] = (short)(fout + j);
                f += fstride * inStride;
            }
        }
        else
        {
            for (int j = 0; j < p; j++)
            {
                BitReverse(fout, f, fstride * p, inStride, factor + 2);
                f += fstride * inStride;
                fout += m;
            }
        }
    }

    // rnn_fft_impl: the butterflies from the last stage to the first. The butterflies index the buffer through refs: every
    // index follows from the factors and stays under Size, and the buffer holds Size values (checked by Forward; rule 51).
    private void Transform(Span<RnNoiseComplex> fout)
    {
        Span<int> fstride = stackalloc int[MaxFactors + 1];
        fstride[0] = 1;
        int stages = 0, m;
        do
        {
            int p = _factors[2 * stages];
            m = _factors[2 * stages + 1];
            fstride[stages + 1] = fstride[stages] * p;
            stages++;
        }
        while (m != 1);

        ref var f = ref MemoryMarshal.GetReference(fout);
        ref var tw = ref MemoryMarshal.GetArrayDataReference(_twiddles);
        m = _factors[2 * stages - 1];
        for (int i = stages - 1; i >= 0; i--)
        {
            int m2 = i != 0 ? _factors[2 * i - 1] : 1;
            switch (_factors[2 * i])
            {
                case 2:
                    Butterfly2(ref f, m, fstride[i]);
                    break;
                case 4:
                    Butterfly4(ref f, ref tw, fstride[i], m, fstride[i], m2);
                    break;
                case 3:
                    Butterfly3(ref f, ref tw, fstride[i], m, fstride[i], m2);
                    break;
                case 5:
                    Butterfly5(ref f, ref tw, fstride[i], m, fstride[i], m2);
                    break;
            }

            m = m2;
        }
    }

    private static void Butterfly2(ref RnNoiseComplex fout, int m, int n)
    {
        if (m == 1)
        {
            for (int i = 0; i < n; i++)
            {
                ref var f0 = ref Unsafe.Add(ref fout, 2 * i);
                ref var f1 = ref Unsafe.Add(ref f0, 1);
                var t = f1;
                f1.R = f0.R - t.R;
                f1.I = f0.I - t.I;
                f0.R += t.R;
                f0.I += t.I;
            }

            return;
        }

        const float Tw = 0.7071067812f;
        for (int i = 0; i < n; i++)
        {
            ref var f = ref Unsafe.Add(ref fout, 8 * i);
            for (int k = 0; k < 4; k++)
            {
                ref var a = ref Unsafe.Add(ref f, k);
                ref var b = ref Unsafe.Add(ref f, k + 4);
                RnNoiseComplex t;
                switch (k)
                {
                    case 0:
                        t = b;
                        break;
                    case 1:
                        t.R = (b.R + b.I) * Tw;
                        t.I = (b.I - b.R) * Tw;
                        break;
                    case 2:
                        t.R = b.I;
                        t.I = -b.R;
                        break;
                    default:
                        t.R = (b.I - b.R) * Tw;
                        t.I = -(b.I + b.R) * Tw;
                        break;
                }

                b.R = a.R - t.R;
                b.I = a.I - t.I;
                a.R += t.R;
                a.I += t.I;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Butterfly4(ref RnNoiseComplex fout, ref RnNoiseComplex tw, int fstride, int m, int n, int mm)
    {
        if (m == 1)
        {
            // All the twiddles are 1.
            for (int i = 0; i < n; i++)
            {
                ref var f0 = ref Unsafe.Add(ref fout, 4 * i);
                ref var f1 = ref Unsafe.Add(ref f0, 1);
                ref var f2 = ref Unsafe.Add(ref f0, 2);
                ref var f3 = ref Unsafe.Add(ref f0, 3);
                RnNoiseComplex s0, s1;
                s0.R = f0.R - f2.R;
                s0.I = f0.I - f2.I;
                f0.R += f2.R;
                f0.I += f2.I;
                s1.R = f1.R + f3.R;
                s1.I = f1.I + f3.I;
                f2.R = f0.R - s1.R;
                f2.I = f0.I - s1.I;
                f0.R += s1.R;
                f0.I += s1.I;
                s1.R = f1.R - f3.R;
                s1.I = f1.I - f3.I;
                f1.R = s0.R + s1.I;
                f1.I = s0.I - s1.R;
                f3.R = s0.R - s1.I;
                f3.I = s0.I + s1.R;
            }

            return;
        }

        int m2 = 2 * m, m3 = 3 * m;
        for (int i = 0; i < n; i++)
        {
            ref var f = ref Unsafe.Add(ref fout, i * mm);
            for (int j = 0; j < m; j++)
            {
                ref var a = ref Unsafe.Add(ref f, j);
                ref var b = ref Unsafe.Add(ref a, m);
                ref var c = ref Unsafe.Add(ref a, m2);
                ref var d = ref Unsafe.Add(ref a, m3);
                RnNoiseComplex s0 = Mul(b, Unsafe.Add(ref tw, j * fstride)), s1 = Mul(c, Unsafe.Add(ref tw, 2 * j * fstride)), s2 = Mul(d, Unsafe.Add(ref tw, 3 * j * fstride)), s3, s4, s5;
                s5.R = a.R - s1.R;
                s5.I = a.I - s1.I;
                a.R += s1.R;
                a.I += s1.I;
                s3.R = s0.R + s2.R;
                s3.I = s0.I + s2.I;
                s4.R = s0.R - s2.R;
                s4.I = s0.I - s2.I;
                c.R = a.R - s3.R;
                c.I = a.I - s3.I;
                a.R += s3.R;
                a.I += s3.I;
                b.R = s5.R + s4.I;
                b.I = s5.I - s4.R;
                d.R = s5.R - s4.I;
                d.I = s5.I + s4.R;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Butterfly3(ref RnNoiseComplex fout, ref RnNoiseComplex tw, int fstride, int m, int n, int mm)
    {
        int m2 = 2 * m;
        float epi3 = Unsafe.Add(ref tw, fstride * m).I;
        for (int i = 0; i < n; i++)
        {
            ref var f = ref Unsafe.Add(ref fout, i * mm);
            for (int k = 0; k < m; k++)
            {
                ref var a = ref Unsafe.Add(ref f, k);
                ref var b = ref Unsafe.Add(ref a, m);
                ref var c = ref Unsafe.Add(ref a, m2);
                RnNoiseComplex s1 = Mul(b, Unsafe.Add(ref tw, k * fstride)), s2 = Mul(c, Unsafe.Add(ref tw, 2 * k * fstride)), s0, s3;
                s3.R = s1.R + s2.R;
                s3.I = s1.I + s2.I;
                s0.R = s1.R - s2.R;
                s0.I = s1.I - s2.I;
                b.R = a.R - s3.R * .5f;
                b.I = a.I - s3.I * .5f;
                s0.R *= epi3;
                s0.I *= epi3;
                a.R += s3.R;
                a.I += s3.I;
                c.R = b.R + s0.I;
                c.I = b.I - s0.R;
                b.R -= s0.I;
                b.I += s0.R;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Butterfly5(ref RnNoiseComplex fout, ref RnNoiseComplex tw, int fstride, int m, int n, int mm)
    {
        RnNoiseComplex ya = Unsafe.Add(ref tw, fstride * m), yb = Unsafe.Add(ref tw, fstride * 2 * m);
        for (int i = 0; i < n; i++)
        {
            ref var f = ref Unsafe.Add(ref fout, i * mm);
            for (int u = 0; u < m; u++)
            {
                ref var f0 = ref Unsafe.Add(ref f, u);
                ref var f1 = ref Unsafe.Add(ref f0, m);
                ref var f2 = ref Unsafe.Add(ref f0, 2 * m);
                ref var f3 = ref Unsafe.Add(ref f0, 3 * m);
                ref var f4 = ref Unsafe.Add(ref f0, 4 * m);
                RnNoiseComplex s0 = f0, s1 = Mul(f1, Unsafe.Add(ref tw, u * fstride)), s2 = Mul(f2, Unsafe.Add(ref tw, 2 * u * fstride)),
                    s3 = Mul(f3, Unsafe.Add(ref tw, 3 * u * fstride)), s4 = Mul(f4, Unsafe.Add(ref tw, 4 * u * fstride));
                RnNoiseComplex s5, s6, s7, s8, s9, s10, s11, s12;
                s7.R = s1.R + s4.R;
                s7.I = s1.I + s4.I;
                s10.R = s1.R - s4.R;
                s10.I = s1.I - s4.I;
                s8.R = s2.R + s3.R;
                s8.I = s2.I + s3.I;
                s9.R = s2.R - s3.R;
                s9.I = s2.I - s3.I;
                f0.R += s7.R + s8.R;
                f0.I += s7.I + s8.I;
                s5.R = s0.R + (s7.R * ya.R + s8.R * yb.R);
                s5.I = s0.I + (s7.I * ya.R + s8.I * yb.R);
                s6.R = s10.I * ya.I + s9.I * yb.I;
                s6.I = -(s10.R * ya.I + s9.R * yb.I);
                f1.R = s5.R - s6.R;
                f1.I = s5.I - s6.I;
                f4.R = s5.R + s6.R;
                f4.I = s5.I + s6.I;
                s11.R = s0.R + (s7.R * yb.R + s8.R * ya.R);
                s11.I = s0.I + (s7.I * yb.R + s8.I * ya.R);
                s12.R = s9.I * ya.I - s10.I * yb.I;
                s12.I = s10.R * yb.I - s9.R * ya.I;
                f2.R = s11.R + s12.R;
                f2.I = s11.I + s12.I;
                f3.R = s11.R - s12.R;
                f3.I = s11.I - s12.I;
            }
        }
    }

    // C_MUL: (a.r b.r - a.i b.i, a.r b.i + a.i b.r), each product rounded to float before the sum.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static RnNoiseComplex Mul(RnNoiseComplex a, RnNoiseComplex b) => new() { R = a.R * b.R - a.I * b.I, I = a.R * b.I + a.I * b.R };
}
