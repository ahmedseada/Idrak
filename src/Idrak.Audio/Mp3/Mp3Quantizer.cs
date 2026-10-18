// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;
using System.Runtime.CompilerServices;
using static Idrak.Audio.Mp3EncoderTables;
using static Idrak.Audio.Mp3Huffman;

namespace Idrak.Audio;

/// <summary>One granule of one channel as its side information describes it.</summary>
internal struct Mp3Granule
{
    public int Part23Length, BigValues, GlobalGain, ScalefacCompress, Table0, Table1, Table2, Region0Count, Region1Count, ScalefacScale, Count1Table;
}

/// <summary>
/// Quantizes one granule of one channel within a bit budget and writes its main data (scalefactors and Huffman codes),
/// as 11172-3 Annex C describes the loops: the inner (rate) loop finds the smallest global gain whose Huffman codes fit;
/// the outer (distortion) loop amplifies, through the scalefactors, the bands whose noise passes the masking threshold
/// and runs the rate loop again, keeping the pass with the least noise above the thresholds. Then the region split and
/// tables that code it in the fewest bits, and the gain lowered while that fits. Long blocks, no scfsi, no preflag. One
/// instance is one worker's scratch.
/// </summary>
internal sealed class Mp3Quantizer
{
    private const int Lines = 576, Bands = 21, MaxValue = 8206;
    private const float Rounding = 0.4054f;                               // 11172-3 C.1.5.4.4.1's nint(x^0.75 - 0.0946)

    private static readonly float[] Pow43 = BuildPow43();

    private readonly int[] _start;
    private readonly bool _lsf;
    private readonly Mp3Quality _quality;
    private readonly float[] _abs = new float[Lines], _xr34 = new float[Lines], _amp = new float[Lines];
    private readonly int[] _ix = new int[Lines];
    private readonly int[] _sf = new int[Bands], _bestSf = new int[Bands];
    private readonly float[] _noise = new float[Bands];
    private readonly int[] _prefix = new int[Held * 23], _escapes = new int[23], _bandMax = new int[22];
    private readonly int[] _memoCost = new int[23 * 23], _memoTable = new int[23 * 23];
    private int _lines, _limit, _evaluated;
    private float _maxAmp;
    private Layout _layout;

    private struct Layout
    {
        public int BigEnd, Count1End, Count1Table, S0, S1, T0, T1, T2, Bits;
    }

    /// <param name="bandStart">The 23 starts of the long scalefactor bands (the last 576).</param>
    /// <param name="lsf">MPEG-2's scalefactors (scalefac_compress of 9 bits) instead of MPEG-1's.</param>
    /// <param name="quality">The loops' effort.</param>
    public Mp3Quantizer(int[] bandStart, bool lsf, Mp3Quality quality)
    {
        _start = bandStart;
        _lsf = lsf;
        _quality = quality;
    }

    /// <summary>
    /// Codes <paramref name="xr"/> (576 lines) in at most <paramref name="target"/> bits (scalefactors and Huffman codes)
    /// into <paramref name="bits"/>; <paramref name="xmin"/> is the noise each of the 21 bands may hold. Returns the bits.
    /// </summary>
    public int Encode(ReadOnlySpan<float> xr, ReadOnlySpan<float> xmin, int target, Span<byte> bits, out Mp3Granule side)
    {
        side = new Mp3Granule { GlobalGain = 210 };
        float max34 = Prepare(xr);
        if (max34 == 0 || target <= 0)
        {
            return 0;                                                    // silence, or no bits: every line 0
        }

        Array.Clear(_sf);
        int passes = _quality switch { Mp3Quality.Fast => 1, Mp3Quality.Standard => 8, _ => 24 };
        int patience = _quality == Mp3Quality.High ? 4 : 2, stale = 0;   // passes in a row that did not lower the noise above the thresholds
        int bestGain = -1, bestLimit = _lines, gain = -1;
        float bestOver = float.MaxValue;
        int bestCount = int.MaxValue;
        for (int pass = 0; pass < passes; pass++)
        {
            int part2 = Part2Bits(_sf, out _);
            int budget = target - part2;
            if (part2 < 0 || budget < 0)
            {
                break;
            }

            Amplify();
            _limit = _lines;
            gain = Search(budget, gain);
            if (gain < 0)
            {
                if (pass > 0)
                {
                    break;
                }

                gain = Truncate(budget);                                 // not even the coarsest gain fits: drop the top lines
            }

            if (_quality == Mp3Quality.Fast)
            {
                bestGain = gain;
                bestLimit = _limit;
                break;
            }

            float over = Distortion(gain, xmin, out int overCount);
            if (bestGain < 0 || over < bestOver || (over == bestOver && overCount < bestCount))
            {
                (bestGain, bestLimit, bestOver, bestCount) = (gain, _limit, over, overCount);
                _sf.CopyTo(_bestSf, 0);
                stale = 0;
            }
            else if (++stale >= patience)
            {
                break;
            }

            if (overCount == 0 || _limit < _lines)
            {
                break;
            }

            bool all = true, past = false;
            for (int b = 0; b < Bands; b++)
            {
                if (_noise[b] > xmin[b])
                {
                    past |= ++_sf[b] > (b < 11 ? 15 : 7);
                }

                all &= _sf[b] > 0;
            }

            if (all || past)
            {
                break;
            }
        }

        if (_quality != Mp3Quality.Fast)
        {
            _bestSf.CopyTo(_sf, 0);
            Amplify();
            int budget = target - Part2Bits(_sf, out _);
            Quantize(bestGain, bestLimit);
            Count(bestLimit);
            OptimalSplit();
            gain = bestGain;
            while (bestLimit == _lines && gain > MinGain())
            {
                Quantize(gain - 1, _lines);                              // a finer gain, if the best split makes room for it
                Count(_lines);
                OptimalSplit();
                if (_layout.Bits > budget)
                {
                    Quantize(gain, _lines);
                    Count(_lines);
                    OptimalSplit();
                    break;
                }

                gain--;
            }

            bestGain = gain;
        }
        else if (_evaluated != bestGain)
        {
            Quantize(bestGain, bestLimit);
            Count(bestLimit);
        }

        return Write(xr, bestGain, bits, ref side);
    }

    // |xr| and |xr|^(3/4) (two square roots: the same on every machine), the last line that is not 0; the largest |xr|^(3/4).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private float Prepare(ReadOnlySpan<float> xr)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            for (; i <= Lines - Vector<float>.Count; i += Vector<float>.Count)
            {
                var a = Vector.Abs(new Vector<float>(xr[i..]));
                a.CopyTo(_abs.AsSpan(i));
                Vector.SquareRoot(a * Vector.SquareRoot(a)).CopyTo(_xr34.AsSpan(i));
            }
        }

        for (; i < Lines; i++)
        {
            float a = MathF.Abs(xr[i]);
            _abs[i] = a;
            _xr34[i] = MathF.Sqrt(a * MathF.Sqrt(a));
        }

        float max = 0;
        _lines = 0;
        for (i = 0; i < Lines; i++)
        {
            if (_xr34[i] > 0)
            {
                _lines = i + 1;
                max = Math.Max(max, _xr34[i]);
            }
        }

        Array.Clear(_ix);
        _evaluated = -1;
        return max;
    }

    // The lines amplified by their band's scalefactor (2^(3/8 sf) in the 3/4 power domain), and the largest.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Amplify()
    {
        float max = 0;
        for (int b = 0; b < 22; b++)
        {
            int from = _start[b], to = Math.Min(_start[b + 1], _lines);
            float factor = b < Bands ? Step(6 * _sf[b]) : 1f;
            for (int i = from; i < to; i++)
            {
                float v = _xr34[i] * factor;
                _amp[i] = v;
                max = Math.Max(max, v);
            }
        }

        _maxAmp = max;
        _evaluated = -1;
    }

    // 2^(e / 16), exactly the table's value scaled by a power of two.
    private static float Step(int e) => MathF.ScaleB(Pow2Sixteenths[e & 15], e >> 4);

    // The smallest global gain at which no line quantizes past 8206 (the largest value the escape tables hold).
    private int MinGain()
    {
        const float limit = MaxValue + 0.5f;                             // below 8206.5946, so x + 0.4054 never rounds up to 8207
        int gain = Math.Clamp(210 - (int)(16f / 3 * Mp3Analysis.Log2(limit / _maxAmp)), 0, 255);
        while (gain < 255 && _maxAmp * Step(-3 * (gain - 210)) >= limit)
        {
            gain++;
        }

        while (gain > 0 && _maxAmp * Step(-3 * (gain - 1 - 210)) < limit)
        {
            gain--;
        }

        return gain;
    }

    // The rate loop: the smallest gain whose codes fit `budget`, bracketed from `hint` (the last pass's gain, -1 for
    // none) outwards, then halved; -1 when not even 255 fits. Leaves _ix and _layout at the gain returned.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private int Search(int budget, int hint)
    {
        int lo = MinGain(), hi;
        if (hint < lo)
        {
            hi = 255;                                                    // no pass before: halve the whole range (255 checked last)
        }
        else if (Fits(hint = Math.Min(hint, 255), budget))
        {
            hi = hint;
            for (int step = 1; hi > lo; step *= 2)
            {
                int probe = Math.Max(lo, hi - step);
                if (!Fits(probe, budget))
                {
                    lo = probe + 1;
                    break;
                }

                hi = probe;
            }
        }
        else
        {
            if (hint == 255)
            {
                return -1;
            }

            lo = hint + 1;
            for (int step = 1; ; step *= 2)
            {
                int at = Math.Min(255, hint + step);
                if (Fits(at, budget))
                {
                    hi = at;
                    break;
                }

                if (at == 255)
                {
                    return -1;
                }

                lo = at + 1;
            }
        }

        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (Fits(mid, budget))
            {
                hi = mid;
            }
            else
            {
                lo = mid + 1;
            }
        }

        if (_evaluated != hi && !Fits(hi, budget))
        {
            return -1;                                                   // only 255 is left unchecked: not even it fits
        }

        return hi;
    }

    private bool Fits(int gain, int budget)
    {
        Quantize(gain, _limit);
        Count(_limit);
        return _layout.Bits <= budget;
    }

    // Not even gain 255 fits (a budget of a few bits): the top lines dropped until the rest does.
    private int Truncate(int budget)
    {
        Quantize(255, _lines);
        int lo = 0, hi = _lines;                                         // lo fits (no lines), hi does not
        while (hi - lo > 1)
        {
            int mid = (lo + hi) >> 1;
            Count(mid);
            if (_layout.Bits <= budget)
            {
                lo = mid;
            }
            else
            {
                hi = mid;
            }
        }

        _limit = lo;
        Quantize(255, lo);
        Count(lo);
        return 255;
    }

    // ix = (int)(amp x 2^(-3/16 (gain - 210)) + 0.4054) for the lines below `limit`, 0 above.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Quantize(int gain, int limit)
    {
        float step = Step(-3 * (gain - 210));
        Span<int> ix = _ix;
        ReadOnlySpan<float> amp = _amp;
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            var s = new Vector<float>(step);
            var r = new Vector<float>(Rounding);
            for (; i <= limit - Vector<float>.Count; i += Vector<float>.Count)
            {
                Vector.ConvertToInt32(new Vector<float>(amp[i..]) * s + r).CopyTo(ix[i..]);
            }
        }

        for (; i < limit; i++)
        {
            ix[i] = (int)(amp[i] * step + Rounding);
        }

        ix[limit.._lines].Clear();
        _evaluated = gain;
    }

    // The noise of each band against the lines it was coded from, and the sum of log2(noise / xmin) of the bands past
    // their threshold (with their count).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private float Distortion(int gain, ReadOnlySpan<float> xmin, out int count)
    {
        float over = 0;
        count = 0;
        var pow43 = Pow43;
        var abs = _abs;
        var ix = _ix;
        for (int b = 0; b < Bands; b++)
        {
            float dq = Step(4 * (gain - 210 - 2 * _sf[b]));
            float noise = 0;
            for (int i = _start[b], end = _start[b + 1]; i < end; i++)
            {
                float d = abs[i] - pow43[ix[i]] * dq;
                noise += d * d;
            }

            _noise[b] = noise;
            if (noise > xmin[b])
            {
                count++;
                over += Mp3Analysis.Log2(noise / xmin[b]);
            }
        }

        return over;
    }

    // The scalefactors' bits, and the scalefac_compress that holds them (-1 when none does).
    private int Part2Bits(ReadOnlySpan<int> sf, out int compress)
    {
        compress = 0;
        if (!_lsf)
        {
            int need1 = BitLength(Max(sf[..11])), need2 = BitLength(Max(sf[11..]));
            int best = -1;
            for (int c = 0; c < 16; c++)
            {
                int s1 = Mp3Tables.ScfcDecode[c] >> 2, s2 = Mp3Tables.ScfcDecode[c] & 3;
                if (s1 >= need1 && s2 >= need2 && (best < 0 || 11 * s1 + 10 * s2 < best))
                {
                    best = 11 * s1 + 10 * s2;
                    compress = c;
                }
            }

            return best;
        }

        int n0 = BitLength(Max(sf[..6])), n1 = BitLength(Max(sf[6..11])), n2 = BitLength(Max(sf[11..16])), n3 = BitLength(Max(sf[16..21]));
        if (n0 > 4 || n1 > 4 || n2 > 3 || n3 > 3)
        {
            return -1;
        }

        compress = ((n0 * 5 + n1) << 4) | (n2 << 2) | n3;                 // 13818-3 2.4.3.2, scalefac_compress below 400
        return 6 * n0 + 5 * n1 + 5 * n2 + 5 * n3;
    }

    // The largest of non-negative values (0 for none).
    private static int Max(ReadOnlySpan<int> values)
    {
        int i = 0, max = 0;
        if (Vector.IsHardwareAccelerated && values.Length >= Vector<int>.Count)
        {
            var m = Vector<int>.Zero;
            for (; i <= values.Length - Vector<int>.Count; i += Vector<int>.Count)
            {
                m = Vector.Max(m, new Vector<int>(values[i..]));
            }

            for (int k = 0; k < Vector<int>.Count; k++)
            {
                max = Math.Max(max, m[k]);
            }
        }

        for (; i < values.Length; i++)
        {
            max = Math.Max(max, values[i]);
        }

        return max;
    }

    private static int BitLength(int value) => 32 - BitOperations.LeadingZeroCount((uint)value);

    // The bits of _ix below `limit` with the default region split: the zero tail, the count1 quads (all 0 or 1), the
    // big values in three regions split near a third and two thirds at band edges, each with its cheapest table.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Count(int limit)
    {
        var ix = _ix.AsSpan(0, Lines);
        int last = ix[..limit].LastIndexOfAnyExcept(0);
        int count1End = last < 0 ? 0 : (last + 2) & ~1, bigEnd = count1End;
        while (bigEnd >= 4 && (ix[bigEnd - 1] | ix[bigEnd - 2] | ix[bigEnd - 3] | ix[bigEnd - 4]) <= 1)
        {
            bigEnd -= 4;
        }

        int a = 0, b = 0;
        ReadOnlySpan<byte> count1 = Count1Lengths;
        for (int q = bigEnd; q < count1End; q += 4)
        {
            int quad = (ix[q] << 3) | (ix[q + 1] << 2) | (ix[q + 2] << 1) | ix[q + 3];
            int signs = ix[q] + ix[q + 1] + ix[q + 2] + ix[q + 3];
            a += count1[quad] + signs;
            b += 4 + signs;
        }

        var layout = new Layout { BigEnd = bigEnd, Count1End = count1End, Count1Table = a <= b ? 0 : 1, S0 = 1, S1 = 2 };
        int bits = Math.Min(a, b);
        if (bigEnd > 0)
        {
            int s0 = 1;
            while (s0 < 16 && _start[s0 + 1] <= bigEnd / 3)
            {
                s0++;
            }

            int s1 = s0 + 1;
            while (s1 < Math.Min(s0 + 8, 22) && _start[s1 + 1] <= 2 * bigEnd / 3)
            {
                s1++;
            }

            int r0 = Math.Min(_start[s0], bigEnd), r1 = Math.Min(_start[s1], bigEnd);
            bool wide = _quality == Mp3Quality.High;
            bits += RegionBits(0, r0, wide, out layout.T0) + RegionBits(r0, r1, wide, out layout.T1) + RegionBits(r1, bigEnd, wide, out layout.T2);
            layout.S0 = s0;
            layout.S1 = s1;
        }

        layout.Bits = bits;
        _layout = layout;
    }

    // The bits of the pairs in [from, to) with the cheapest table of those that hold their largest value (and of the next
    // group up when `wide`), and that table's number.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private int RegionBits(int from, int to, bool wide, out int table)
    {
        var ix = _ix;
        int max = Max(ix.AsSpan(from, to - from));
        table = 0;
        if (max == 0)
        {
            return 0;
        }

        Span<int> tables = stackalloc int[6];
        int n = Candidates(max, tables);
        if (wide && max < 16)
        {
            n += Candidates(max <= 1 ? 2 : max <= 2 ? 3 : max <= 3 ? 4 : max <= 5 ? 6 : max <= 7 ? 8 : 16, tables[n..]);
        }

        Span<int> sums = stackalloc int[6], bases = stackalloc int[6];
        for (int c = 0; c < n; c++)
        {
            sums[c] = 0;
            bases[c] = HeldIndex[tables[c]] << 8;
        }

        var pairBits = PairBits;
        int escapes = 0;
        for (int i = from; i < to; i += 2)
        {
            int x = ix[i], y = ix[i + 1];
            escapes += (x >= 15 ? 1 : 0) + (y >= 15 ? 1 : 0);
            int at = (Math.Min(x, 15) << 4) | Math.Min(y, 15);
            for (int c = 0; c < n; c++)
            {
                sums[c] += pairBits[bases[c] + at];
            }
        }

        int best = int.MaxValue;
        for (int c = 0; c < n; c++)
        {
            int bits = sums[c] + escapes * Linbits[tables[c]];
            if (bits < best)
            {
                best = bits;
                table = tables[c];
            }
        }

        return best;
    }

    // The tables of the group that holds `max` (11172-3 2.4.2.7's table_select): 1; 2, 3; 5, 6; 7-9; 10-12; 13, 15; and
    // past 15 the escape tables of each group with the fewest linbits that hold it.
    private static int Candidates(int max, Span<int> tables)
    {
        switch (max)
        {
            case 1:
                tables[0] = 1;
                return 1;
            case 2:
                (tables[0], tables[1]) = (2, 3);
                return 2;
            case 3:
                (tables[0], tables[1]) = (5, 6);
                return 2;
            case <= 5:
                (tables[0], tables[1], tables[2]) = (7, 8, 9);
                return 3;
            case <= 7:
                (tables[0], tables[1], tables[2]) = (10, 11, 12);
                return 3;
            case <= 15:
                (tables[0], tables[1]) = (13, 15);
                return 2;
            default:
                tables[0] = EscapeTable(16, max);
                tables[1] = EscapeTable(24, max);
                return 2;
        }
    }

    private static int EscapeTable(int first, int max)
    {
        int need = max <= 15 ? 1 : BitLength(max - 15);
        int t = first;
        while (Linbits[t] < need)
        {
            t++;
        }

        return t;
    }

    // The region split (region0 and region1 counts) and the tables that code the big values in the fewest bits, over
    // every split: the bits of each band per table summed once, each region the cheapest table that holds its values.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void OptimalSplit()
    {
        ref var layout = ref _layout;
        int bigEnd = layout.BigEnd;
        if (bigEnd == 0)
        {
            return;
        }

        int count1Bits = layout.Bits - BigBits(layout);
        int bands = 0;
        while (_start[bands] < bigEnd)
        {
            bands++;
        }

        var ix = _ix;
        var pairBits = PairBits;
        Span<int> sums = stackalloc int[Held];
        Array.Clear(_prefix, 0, Held);
        _escapes[0] = 0;
        for (int b = 0; b < bands; b++)
        {
            sums.Clear();
            int max = 0, escapes = 0;
            for (int i = _start[b], end = Math.Min(_start[b + 1], bigEnd); i < end; i += 2)
            {
                int x = ix[i], y = ix[i + 1];
                max = Math.Max(max, Math.Max(x, y));
                escapes += (x >= 15 ? 1 : 0) + (y >= 15 ? 1 : 0);
                int at = (Math.Min(x, 15) << 4) | Math.Min(y, 15);
                for (int h = 0; h < Held; h++)
                {
                    sums[h] += pairBits[(h << 8) + at];
                }
            }

            _bandMax[b] = max;
            _escapes[b + 1] = _escapes[b] + escapes;
            for (int h = 0; h < Held; h++)
            {
                _prefix[(b + 1) * Held + h] = _prefix[b * Held + h] + sums[h];
            }
        }

        _memoCost.AsSpan().Fill(-1);
        int best = int.MaxValue;
        for (int s0 = 1; s0 <= Math.Min(16, bands); s0++)
        {
            int c0 = Range(0, s0, out int t0);
            for (int s1 = s0 + 1; s1 <= Math.Min(s0 + 8, 22); s1++)
            {
                int e1 = Math.Min(s1, bands);
                int bits = c0 + Range(s0, e1, out int t1) + Range(e1, bands, out int t2);
                if (bits < best)
                {
                    best = bits;
                    (layout.S0, layout.S1, layout.T0, layout.T1, layout.T2) = (s0, s1, t0, t1, t2);
                }

                if (s1 >= bands)
                {
                    break;
                }
            }
        }

        layout.Bits = best + count1Bits;
    }

    // The big values' bits of a layout (its regions with its tables).
    private int BigBits(in Layout layout)
    {
        int r0 = Math.Min(_start[layout.S0], layout.BigEnd), r1 = Math.Min(_start[layout.S1], layout.BigEnd);
        return TableBits(0, r0, layout.T0) + TableBits(r0, r1, layout.T1) + TableBits(r1, layout.BigEnd, layout.T2);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private int TableBits(int from, int to, int table)
    {
        if (table == 0)
        {
            return 0;
        }

        int bits = 0, h = HeldIndex[table] << 8, lin = Linbits[table];
        var ix = _ix;
        var pairBits = PairBits;
        for (int i = from; i < to; i += 2)
        {
            int x = ix[i], y = ix[i + 1];
            bits += pairBits[h + ((Math.Min(x, 15) << 4) | Math.Min(y, 15))] + (x >= 15 ? lin : 0) + (y >= 15 ? lin : 0);
        }

        return bits;
    }

    // Bands [from, to) as one region: the cheapest table that holds their largest value (memoized per call).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private int Range(int from, int to, out int table)
    {
        table = 0;
        if (from >= to)
        {
            return 0;
        }

        int key = from * 23 + to;
        if (_memoCost[key] >= 0)
        {
            table = _memoTable[key];
            return _memoCost[key];
        }

        int max = 0;
        for (int b = from; b < to; b++)
        {
            max = Math.Max(max, _bandMax[b]);
        }

        int best = 0;
        if (max > 0)
        {
            best = int.MaxValue;
            int escapes = _escapes[to] - _escapes[from];
            for (int h = 0; h < Held; h++)
            {
                int t = HuffmanTables[h];
                bool escape = h >= Escape16;
                if (!escape && XLen[h] <= max)
                {
                    continue;
                }

                if (escape)
                {
                    t = EscapeTable(t, max);
                }

                int bits = _prefix[to * Held + h] - _prefix[from * Held + h] + (escape ? escapes * Linbits[t] : 0);
                if (bits < best)
                {
                    best = bits;
                    table = t;
                }
            }
        }

        _memoCost[key] = best;
        _memoTable[key] = table;
        return best;
    }

    // The granule's main data: its scalefactors (part 2), then the big values region by region and the count1 quads
    // (part 3), each value's sign after its code (and its linbits); fills the side information.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private int Write(ReadOnlySpan<float> xr, int gain, Span<byte> bits, ref Mp3Granule side)
    {
        var layout = _layout;
        int part2 = Part2Bits(_sf, out int compress);
        var writer = new Mp3BitWriter(bits);
        if (!_lsf)
        {
            int s1 = Mp3Tables.ScfcDecode[compress] >> 2, s2 = Mp3Tables.ScfcDecode[compress] & 3;
            for (int b = 0; b < Bands; b++)
            {
                writer.Write((uint)_sf[b], b < 11 ? s1 : s2);
            }
        }
        else
        {
            int n0 = (compress >> 4) / 5, n1 = (compress >> 4) % 5, n2 = (compress >> 2) & 3, n3 = compress & 3;
            for (int b = 0; b < Bands; b++)
            {
                writer.Write((uint)_sf[b], b < 6 ? n0 : b < 11 ? n1 : b < 16 ? n2 : n3);
            }
        }

        var ix = _ix;
        int r0 = Math.Min(_start[layout.S0], layout.BigEnd), r1 = Math.Min(_start[layout.S1], layout.BigEnd);
        WriteRegion(ref writer, xr, 0, r0, layout.T0);
        WriteRegion(ref writer, xr, r0, r1, layout.T1);
        WriteRegion(ref writer, xr, r1, layout.BigEnd, layout.T2);
        for (int q = layout.BigEnd; q < layout.Count1End; q += 4)
        {
            int quad = (ix[q] << 3) | (ix[q + 1] << 2) | (ix[q + 2] << 1) | ix[q + 3];
            if (layout.Count1Table == 0)
            {
                writer.Write(Count1Codes[quad], Count1Lengths[quad]);
            }
            else
            {
                writer.Write((uint)(15 - quad), 4);
            }

            for (int i = q; i < q + 4; i++)
            {
                if (ix[i] != 0)
                {
                    writer.Write(xr[i] < 0 ? 1u : 0u, 1);
                }
            }
        }

        writer.Flush();
        if (writer.Bits != part2 + layout.Bits || writer.Bits > 4095)
        {
            throw new InvalidOperationException($"the granule wrote {writer.Bits} bits, counted {part2} + {layout.Bits}");
        }

        side = new Mp3Granule
        {
            Part23Length = writer.Bits,
            BigValues = layout.BigEnd / 2,
            GlobalGain = gain,
            ScalefacCompress = compress,
            Table0 = layout.T0,
            Table1 = layout.T1,
            Table2 = layout.T2,
            Region0Count = layout.S0 - 1,
            Region1Count = layout.S1 - layout.S0 - 1,
            ScalefacScale = 0,                                           // scalefactor steps of 2^(1/2): scalefac_scale 1 is not used
            Count1Table = layout.Count1Table,
        };
        return writer.Bits;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void WriteRegion(ref Mp3BitWriter writer, ReadOnlySpan<float> xr, int from, int to, int table)
    {
        if (table == 0)
        {
            return;
        }

        int h = HeldIndex[table] << 8, lin = Linbits[table];
        for (int i = from; i < to; i += 2)
        {
            int x = _ix[i], y = _ix[i + 1];
            int at = h + ((Math.Min(x, 15) << 4) | Math.Min(y, 15));
            writer.Write(Codes[at], CodeLengths[at]);
            WriteValue(ref writer, x, xr[i] < 0, lin);
            WriteValue(ref writer, y, xr[i + 1] < 0, lin);
        }
    }

    private static void WriteValue(ref Mp3BitWriter writer, int value, bool negative, int lin)
    {
        if (lin > 0 && value >= 15)
        {
            writer.Write((uint)(value - 15), lin);
        }

        if (value != 0)
        {
            writer.Write(negative ? 1u : 0u, 1);
        }
    }

    // i^(4/3) for i up to 8206 as i * cbrt(i), the cube root by Newton's steps from the last one (only +, * and /: the
    // same values on every machine).
    private static float[] BuildPow43()
    {
        var table = new float[MaxValue + 1];
        double root = 1;
        for (int i = 1; i <= MaxValue; i++)
        {
            for (int step = 0; step < 8; step++)
            {
                root = (2 * root + i / (root * root)) / 3;
            }

            table[i] = (float)(i * root);
        }

        return table;
    }
}
