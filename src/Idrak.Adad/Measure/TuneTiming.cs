// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Adad.Measure;

/// <summary>The class of a size a measured choice is kept for (copied from Idrak's TuneSizes).</summary>
internal static class SizeClass
{
    /// <summary>Sizes up to this are their own class.</summary>
    public const int Exact = 32;

    /// <summary>The class of <paramref name="size"/>: the size itself up to <see cref="Exact"/>, else the next of 1, 1.25, 1.5 or 1.75 times a power of two.</summary>
    public static int Of(int size)
    {
        if (size <= Exact)
        {
            return size;
        }

        long power = 1L << (63 - System.Numerics.BitOperations.LeadingZeroCount((ulong)size)); // the largest power of two ≤ size
        long quarter = power / 4;
        long rounded = (size + quarter - 1) / quarter * quarter;                               // up to the next quarter of the octave
        return (int)Math.Min(rounded, int.MaxValue);
    }
}

/// <summary>How candidates are timed and one is chosen (copied from Idrak's TuneTiming, as the proof measured).</summary>
internal static class TuneTiming
{
    /// <summary>
    /// Rounds of paired timings per candidate; the median of the pair ratios is kept, so up to three outliers (either
    /// way) cannot move the choice. Even, so each candidate runs as often before its reference as after it.
    /// </summary>
    internal const int Rounds = 8;

    /// <summary>A candidate replaces the default's choice only when its median is at least this much faster (3%).</summary>
    internal const float Margin = 0.97f;

    /// <summary>GPU time (ms) every candidate runs, in turns, before timing, at the least (Warm)…</summary>
    internal const float WarmMinMs = 25f;

    /// <summary>… and at the most, when its speed has not settled before.</summary>
    internal const float WarmMaxMs = 250f;

    /// <summary>
    /// The index of the candidate to keep. Each candidate is timed in pairs with the reference (the default's choice,
    /// <paramref name="fallback"/>, when it is a candidate, else the first), the two timings back to back, over
    /// <paramref name="rounds"/> rounds; the reference runs first in even rounds and second in odd ones, and the
    /// candidates take their turns forwards, then backwards. Each pair gives the ratio of the candidate's timing to the
    /// reference's, and the medians of these ratios are compared. The default stays unless the fastest beats it by the
    /// margin, and a candidate that beats it is timed against it again in new pairs and kept only when it wins by the margin
    /// a second time. <paramref name="time"/>(index) returns one timing of that candidate (ms per run);
    /// <paramref name="medians"/>, when given, receives each candidate's median ratio, <paramref name="confirmation"/>[0]
    /// the second median (NaN when nothing needed confirming).
    /// </summary>
    internal static int Choose(ReadOnlySpan<int> candidates, int fallback, Func<int, float> time, int rounds = Rounds, float margin = Margin,
        float[]? medians = null, bool confirm = true, float[]? confirmation = null)
    {
        int count = candidates.Length;
        int formula = candidates.IndexOf(fallback);
        int reference = Math.Max(formula, 0);
        var ratios = new float[count * rounds];
        for (int round = 0; round < rounds; round++)
        {
            for (int i = 0; i < count; i++)
            {
                int c = round % 2 == 0 ? i : count - 1 - i;
                if (c == reference)
                {
                    continue;
                }

                float t, r;
                if (round % 2 == 0)
                {
                    r = time(reference);
                    t = time(c);
                }
                else
                {
                    t = time(c);
                    r = time(reference);
                }

                ratios[c * rounds + round] = t / Math.Max(r, 1e-9f);
            }
        }

        medians ??= new float[count];
        for (int c = 0; c < count; c++)
        {
            medians[c] = c == reference ? 1f : Median(ratios.AsSpan(c * rounds, rounds));
        }

        int fastest = 0;
        for (int c = 1; c < count; c++)
        {
            if (medians[c] < medians[fastest])
            {
                fastest = c;
            }
        }

        if (confirmation is { Length: > 0 })
        {
            confirmation[0] = float.NaN;
        }

        if (formula < 0)
        {
            return fastest;
        }

        if (medians[fastest] >= medians[formula] * margin)
        {
            return formula;
        }

        if (!confirm)
        {
            return fastest;
        }

        var again = new float[rounds];
        for (int round = 0; round < rounds; round++)
        {
            float t, r;
            if (round % 2 == 0)
            {
                r = time(reference);
                t = time(fastest);
            }
            else
            {
                t = time(fastest);
                r = time(reference);
            }

            again[round] = t / Math.Max(r, 1e-9f);
        }

        float second = Median(again);
        if (confirmation is { Length: > 0 })
        {
            confirmation[0] = second;
        }

        return second < margin ? fastest : formula;
    }

    /// <summary>
    /// Runs <paramref name="pass"/> (every candidate once; returns its GPU time in ms) until the GPU's speed has settled:
    /// at least <paramref name="minMs"/> of GPU time, then until two consecutive windows of about
    /// <paramref name="windowMs"/> each take the same time per pass within <paramref name="tolerance"/>, or
    /// <paramref name="maxMs"/> in all. Returns the GPU time spent.
    /// </summary>
    internal static float Warm(Func<float> pass, float minMs = WarmMinMs, float maxMs = WarmMaxMs, float windowMs = 5f, float tolerance = 0.03f)
    {
        float total = 0f, window = 0f, previous = float.NaN;
        int passes = 0;
        for (int i = 0; i < 100_000; i++)
        {
            float t = Math.Max(0f, pass());
            total += t;
            window += t;
            passes++;
            if (total >= maxMs)
            {
                break;
            }

            if (window >= windowMs)
            {
                float perPass = window / passes;
                if (total >= minMs && !float.IsNaN(previous) && MathF.Abs(perPass - previous) <= tolerance * previous)
                {
                    break;
                }

                previous = perPass;
                (window, passes) = (0f, 0);
            }
        }

        return total;
    }

    /// <summary>The median of <paramref name="values"/> (sorted in place; the mean of the middle two for an even count).</summary>
    internal static float Median(Span<float> values)
    {
        values.Sort();
        int n = values.Length;
        return n == 0 ? float.NaN : n % 2 == 1 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
    }

    /// <summary>Runs per timing: back-to-back runs adding up to about 0.2 ms, at most 16; cold timings at most 4.</summary>
    internal static int TimingRepeats(float onceMs, bool cold) =>
        Math.Clamp((int)MathF.Ceiling(0.2f / Math.Max(onceMs, 1e-3f)), 1, cold ? 4 : 16);

    /// <summary>Floats read to evict the L2 cache before a cold timing: twice the size the device reports; none when it reports none.</summary>
    internal static long L2FlushFloats(int l2Bytes) => l2Bytes <= 0 ? 0 : 2L * l2Bytes / sizeof(float);
}
