// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using Idrak.Abstraction.Devices.Cpu;

namespace Idrak.Abstraction.Generation;

// The weights the CPU sampler draws a row from (Backend.SampleRowsKernel's rule: temperature, top-k, min-p, top-p, and the
// non-finite scores), shared with TokenSampler.Probabilities so a host-side check (speculative decoding) reads the very
// distribution a step samples from: the same operations in the same order. e^x and ln x are the CPU's own (CpuMath), the
// same bits on every machine.
internal static class SamplingWeights
{
    // Fills e[0..z.Length) with the row's unnormalized weights and returns their sum (0: nothing to sample). `top` holds at
    // least max(1, min(topK, z.Length)) floats; the candidate buffers are rented on first use (at z.Length) and stay the
    // caller's to return.
    public static float Compute(ReadOnlySpan<float> z, Span<float> e, float temperature, int topK, float topP, float minP, float[] top,
        ref float[]? candidateScores, ref float[]? candidateWeights)
    {
        float invT = 1f / MathF.Max(temperature, 1e-3f);
        float max = float.NegativeInfinity;
        foreach (float v in z)
        {
            float sc = v * invT;
            if (sc > max)                                                // NaN is never a candidate
            {
                max = sc;
            }
        }

        // Non-finite scores (Backend.SampleRowsKernel): any +∞ makes the row a uniform draw among its +∞ tokens; no
        // finite score and no +∞ leaves nothing to sample (ids gets 0, the statistics -1).
        bool infinite = max == float.PositiveInfinity, empty = max == float.NegativeInfinity;

        // Top-k: the k-th largest distinct scaled score is the cut-off (ties at the cut-off are kept); fewer than k
        // distinct scores keep everything. One pass keeping the k largest distinct scores, largest first.
        float threshold = float.NegativeInfinity;
        if (topK > 0 && topK < z.Length && !infinite && !empty)
        {
            int count = 0;
            foreach (float v in z)
            {
                float sc = v * invT;
                if (float.IsNaN(sc) || count == topK && !(sc > top[count - 1]))
                {
                    continue;
                }

                int at = count;
                while (at > 0 && top[at - 1] < sc)
                {
                    at--;
                }

                if (at > 0 && top[at - 1] == sc)
                {
                    continue;                                            // already kept
                }

                int last = Math.Min(count, topK - 1);
                for (int i = last; i > at; i--)
                {
                    top[i] = top[i - 1];
                }

                top[at] = sc;
                count = Math.Min(count + 1, topK);
            }

            threshold = count == topK ? top[topK - 1] : float.NegativeInfinity;
        }

        // Min-p: keep tokens at least minP times as likely as the best: s >= max + ln(minP).
        if (minP > 0f && !infinite && !empty)
        {
            threshold = MathF.Max(threshold, max + CpuMath.Log(minP));
        }

        // Top-p (nucleus): the highest cut-off whose kept mass is still >= topP of the total, by bisection. Only scores at
        // or above the bisection's lower bound can count, so their weights are computed once and summed in token order.
        if (topP > 0f && topP < 1f && !infinite && !empty)
        {
            float total = 0f, floor = MathF.Max(max - 40f, threshold);
            candidateScores ??= ArrayPool<float>.Shared.Rent(z.Length);
            candidateWeights ??= ArrayPool<float>.Shared.Rent(z.Length);
            int candidates = 0;
            foreach (float v in z)
            {
                float sc = v * invT;
                if (sc >= threshold)
                {
                    float w = CpuMath.Exp(sc - max);
                    total += w;
                    if (sc >= floor)
                    {
                        candidateScores[candidates] = sc;
                        candidateWeights[candidates++] = w;
                    }
                }
            }

            float goal = total * topP, lo = floor, hi = max;
            for (int it = 0; it < 24; it++)
            {
                float mid = (lo + hi) * 0.5f, mass = 0f;
                for (int c = 0; c < candidates; c++)
                {
                    mass += candidateScores[c] >= mid ? candidateWeights[c] : 0f;
                }

                if (mass >= goal)
                {
                    // Every later midpoint is at least this one, so scores below it never count again: drop them (in
                    // order, so the sums add the same values in the same order; the dropped ones only added zeros).
                    lo = mid;
                    int kept = 0;
                    for (int c = 0; c < candidates; c++)
                    {
                        if (candidateScores[c] >= mid)
                        {
                            candidateScores[kept] = candidateScores[c];
                            candidateWeights[kept++] = candidateWeights[c];
                        }
                    }

                    candidates = kept;
                }
                else
                {
                    hi = mid;
                }
            }

            threshold = lo;
        }

        float sum = 0f;
        for (int j = 0; j < z.Length; j++)
        {
            float sc = z[j] * invT;
            e[j] = infinite ? (sc == float.PositiveInfinity ? 1f : 0f) : sc >= threshold && sc > float.NegativeInfinity ? CpuMath.Exp(sc - max) : 0f;
            sum += e[j];
        }

        return sum;
    }
}
