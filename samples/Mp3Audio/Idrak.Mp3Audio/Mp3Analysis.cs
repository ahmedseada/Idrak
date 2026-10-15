// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;
using System.Runtime.CompilerServices;
using static Idrak.Mp3Audio.Mp3EncoderTables;

namespace Idrak.Mp3Audio;

/// <summary>
/// The encoder's hybrid filterbank (ISO/IEC 11172-3 Annex C): the polyphase analysis of each 32 samples into 32
/// subband samples, then per granule the long-block MDCT of every subband's 36 samples (the previous granule's 18 and
/// its own) and the aliasing butterflies, into 576 frequency lines. Every vector loop runs across output lanes (subbands
/// or window taps), each lane summed in the scalar loop's order, so the values are the same for every vector width and
/// without SIMD. One instance holds one worker's state: the previous granule's subband samples per channel.
/// </summary>
internal sealed class Mp3Analysis
{
    private const int Rows = 36, Bands = 32;

    private readonly float[][] _z;                                    // per channel [36][32]: the previous granule's rows, then this one's
    private readonly float[] _block = new float[512];                  // a block of input padded with silence past the ends
    private readonly float[] _yrev = new float[64];
    private readonly float[] _y = new float[32];
    private readonly float[] _line = new float[Bands];

    public Mp3Analysis(int channels)
    {
        _z = new float[channels][];
        for (int c = 0; c < channels; c++)
        {
            _z[c] = new float[Rows * Bands];
        }
    }

    /// <summary>Makes granule <paramref name="granule"/> - 1 of channel <paramref name="channel"/> the previous one (silence before the start).</summary>
    public void Prime(int channel, ReadOnlySpan<float> samples, long granule)
    {
        var z = _z[channel];
        if (granule <= 0)
        {
            z.AsSpan(0, 18 * Bands).Clear();
            return;
        }

        Subbands(samples, granule - 1, z.AsSpan(0, 18 * Bands));
    }

    /// <summary>
    /// The 576 frequency lines of granule <paramref name="granule"/> of a channel into <paramref name="xr"/>; the
    /// granule before it must have been the last one given (or <see cref="Prime"/>d).
    /// </summary>
    public void Granule(int channel, ReadOnlySpan<float> samples, long granule, Span<float> xr)
    {
        var z = _z[channel];
        Subbands(samples, granule, z.AsSpan(18 * Bands, 18 * Bands));
        Mdct(z, xr);
        z.AsSpan(18 * Bands, 18 * Bands).CopyTo(z);
    }

    // The 18 rows of 32 subband samples of a granule (11172-3 C.1.3), the odd subbands' odd samples negated (the
    // frequency inversion the MDCT expects).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Subbands(ReadOnlySpan<float> samples, long granule, Span<float> rows)
    {
        for (int tau = 0; tau < 18; tau++)
        {
            long start = 32 * (18 * granule + tau) - 480;              // the oldest of the 512 samples, the newest is start + 511
            ReadOnlySpan<float> block;
            if (start >= 0 && start + 512 <= samples.Length)
            {
                block = samples.Slice((int)start, 512);
            }
            else
            {
                var padded = _block.AsSpan();
                padded.Clear();
                long from = Math.Max(0, start), to = Math.Min(samples.Length, start + 512);
                if (to > from)
                {
                    samples[(int)from..(int)to].CopyTo(padded[(int)(from - start)..]);
                }

                block = padded;
            }

            var row = rows.Slice(tau * Bands, Bands);
            Polyphase(block, row);
            if ((tau & 1) != 0)
            {
                for (int i = 1; i < Bands; i += 2)
                {
                    row[i] = -row[i];
                }
            }
        }
    }

    // One block: windowing into Y (Y[63 - k] = sum of W[64 j + k] a[64 j + k]), the symmetric fold of Y to 32 values, and
    // the 32 x 32 matrixing.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Polyphase(ReadOnlySpan<float> a, Span<float> s)
    {
        ReadOnlySpan<float> w = AnalysisWindow, m = AnalysisMatrix;
        Span<float> yrev = _yrev, y = _y;
        int k = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int n = Vector<float>.Count;
            for (; k <= 64 - n; k += n)
            {
                var acc = new Vector<float>(w[k..]) * new Vector<float>(a[k..]);
                for (int j = 1; j < 8; j++)
                {
                    acc += new Vector<float>(w[(64 * j + k)..]) * new Vector<float>(a[(64 * j + k)..]);
                }

                acc.CopyTo(yrev[k..]);
            }
        }

        for (; k < 64; k++)
        {
            float acc = w[k] * a[k];
            for (int j = 1; j < 8; j++)
            {
                acc += w[64 * j + k] * a[64 * j + k];
            }

            yrev[k] = acc;
        }

        y[0] = yrev[47];
        for (int d = 1; d <= 16; d++)
        {
            y[d] = yrev[47 - d] + yrev[47 + d];
        }

        for (int d = 17; d < 32; d++)
        {
            y[d] = yrev[47 - d] - yrev[d - 17];
        }

        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int n = Vector<float>.Count;
            for (; i <= Bands - n; i += n)
            {
                var acc = new Vector<float>(m[i..]) * y[0];
                for (int d = 1; d < 32; d++)
                {
                    acc += new Vector<float>(m[(d * 32 + i)..]) * y[d];
                }

                acc.CopyTo(s[i..]);
            }
        }

        for (; i < Bands; i++)
        {
            float acc = m[i] * y[0];
            for (int d = 1; d < 32; d++)
            {
                acc += m[d * 32 + i] * y[d];
            }

            s[i] = acc;
        }
    }

    // The long-block MDCT of every subband (36 rows of z: X[k][sb] = sum over n of T[n][k] z[n][sb]) into xr[18 sb + k],
    // then the aliasing butterflies between neighbouring subbands (the inverse of the decoder's).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Mdct(float[] z, Span<float> xr)
    {
        ReadOnlySpan<float> t = MdctMatrix;
        Span<float> line = _line;
        for (int k = 0; k < 18; k++)
        {
            int sb = 0;
            if (Vector.IsHardwareAccelerated)
            {
                int n = Vector<float>.Count;
                for (; sb <= Bands - n; sb += n)
                {
                    var acc = new Vector<float>(z.AsSpan(sb)) * t[k];
                    for (int r = 1; r < Rows; r++)
                    {
                        acc += new Vector<float>(z.AsSpan(r * Bands + sb)) * t[r * 18 + k];
                    }

                    acc.CopyTo(line[sb..]);
                }
            }

            for (; sb < Bands; sb++)
            {
                float acc = z[sb] * t[k];
                for (int r = 1; r < Rows; r++)
                {
                    acc += z[r * Bands + sb] * t[r * 18 + k];
                }

                line[sb] = acc;
            }

            for (sb = 0; sb < Bands; sb++)
            {
                xr[sb * 18 + k] = line[sb];
            }
        }

        ReadOnlySpan<float> cs = AliasCs, ca = AliasCa;
        for (int sb = 1; sb < Bands; sb++)
        {
            for (int i = 0; i < 8; i++)
            {
                float bu = xr[18 * sb - 1 - i], bd = xr[18 * sb + i];
                xr[18 * sb - 1 - i] = bu * cs[i] + bd * ca[i];
                xr[18 * sb + i] = bd * cs[i] - bu * ca[i];
            }
        }
    }

    /// <summary>
    /// The masking model of one granule of one channel: the energy of each of the 22 scalefactor bands, the noise it
    /// may hold (the energies spread across the bands with the offset, or the threshold of hearing where that is
    /// higher), and the perceptual entropy (the bits a line needs to keep its noise there, summed).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float Masking(ReadOnlySpan<float> xr, ReadOnlySpan<int> bandStart, ReadOnlySpan<float> spread, ReadOnlySpan<float> ath, Span<float> energy, Span<float> xmin)
    {
        Energies(xr, bandStart, energy);
        for (int b = 0; b < 22; b++)
        {
            var row = spread.Slice(b * 22, 22);
            float threshold = 0;
            for (int j = 0; j < 22; j++)
            {
                threshold += row[j] * energy[j];
            }

            xmin[b] = Math.Max(threshold, ath[b]);
        }

        return Entropy(energy, xmin, bandStart);
    }

    /// <summary>The energy of each of the 22 scalefactor bands; returns their sum.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float Energies(ReadOnlySpan<float> xr, ReadOnlySpan<int> bandStart, Span<float> energy)
    {
        float total = 0;
        for (int b = 0; b < 22; b++)
        {
            float e = 0;
            for (int i = bandStart[b]; i < bandStart[b + 1]; i++)
            {
                e += xr[i] * xr[i];
            }

            energy[b] = e;
            total += e;
        }

        return total;
    }

    /// <summary>The perceptual entropy of bands of <paramref name="energy"/> held to <paramref name="xmin"/>: sum of width x log2(energy / xmin) / 2 where above.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float Entropy(ReadOnlySpan<float> energy, ReadOnlySpan<float> xmin, ReadOnlySpan<int> bandStart)
    {
        float pe = 0;
        for (int b = 0; b < 22; b++)
        {
            if (energy[b] > xmin[b])
            {
                pe += (bandStart[b + 1] - bandStart[b]) * 0.5f * Log2(energy[b] / xmin[b]);
            }
        }

        return pe;
    }

    /// <summary>log2 of a positive float from its bits and a cubic on the mantissa (within 0.01): the same on every machine.</summary>
    public static float Log2(float x)
    {
        int bits = BitConverter.SingleToInt32Bits(x);
        int exponent = ((bits >> 23) & 255) - 127;
        float m = BitConverter.Int32BitsToSingle((bits & 0x7FFFFF) | 0x3F800000) - 1f;
        return exponent + m * (1.44254494f + m * (-0.71814525f + m * 0.27561033f));
    }
}
