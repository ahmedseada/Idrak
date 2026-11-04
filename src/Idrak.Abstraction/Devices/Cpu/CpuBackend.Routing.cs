// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;

namespace Idrak.Abstraction.Devices.Cpu;

// The mixture-of-experts operations (Backend.Routing.cs) on the CPU: the routing as the layer's host routing computes it
// (MixtureOfExperts.Route, the same order of every comparison and sum), and each expert's products over its rows as its own
// layers compute them: CpuMatMul.Multiply over exactly those rows (one FMA chain over k from zero for every element, so
// the rows a product covers never change a value), the bias added after it, and the gated activation over just those rows
// (its vector loop and scalar tail fall where they fall for the expert's own tensor).
internal sealed partial class CpuBackend
{
    public override void RouteTopKKernel(Storage probabilities, Storage weights, Storage offsets, Storage tokenOfRow, Storage slotRow, int tokens, int experts, int k,
        bool normalize)
    {
        float[] pv = D(probabilities), wv = D(weights), ov = D(offsets), tv = D(tokenOfRow), sv = D(slotRow);
        int slots = tokens * k;
        int[] choices = ArrayPool<int>.Shared.Rent(Math.Max(1, slots));
        int[] cursor = ArrayPool<int>.Shared.Rent(experts);
        // k comes from the model: on the stack only while small (rule 45).
        const int StackExperts = 64;
        int[]? rented = k <= StackExperts ? null : ArrayPool<int>.Shared.Rent(k);
        Span<int> best = rented is null ? stackalloc int[StackExperts] : rented;
        best = best[..k];
        try
        {
            var counts = cursor.AsSpan(0, experts);
            counts.Clear();
            for (int i = 0; i < tokens; i++)
            {
                var row = pv.AsSpan(i * experts, experts);
                int count = 0;
                for (int e = 0; e < experts; e++)
                {
                    // Insert e after every chosen expert at least as probable (earlier indices win ties).
                    int at = count;
                    while (at > 0 && row[e] > row[best[at - 1]])
                    {
                        at--;
                    }

                    if (at >= k)
                    {
                        continue;
                    }

                    for (int j = Math.Min(count, k - 1); j > at; j--)
                    {
                        best[j] = best[j - 1];
                    }

                    best[at] = e;
                    count = Math.Min(count + 1, k);
                }

                best.Sort();
                float sum = 0f;
                foreach (int e in best)
                {
                    sum += row[e];
                }

                for (int s = 0; s < k; s++)
                {
                    int e = best[s];
                    choices[i * k + s] = e;
                    wv[i * k + s] = normalize ? row[e] / sum : row[e];
                    counts[e]++;
                }
            }

            // Rows expert by expert, each expert's in slot order.
            int start = 0;
            for (int e = 0; e < experts; e++)
            {
                int c = counts[e];
                ov[e] = start;
                counts[e] = start;                                            // now the expert's next row
                start += c;
            }

            ov[experts] = slots;
            for (int slot = 0; slot < slots; slot++)
            {
                int row = counts[choices[slot]]++;
                tv[row] = slot / k;
                sv[slot] = row;
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(choices);
            ArrayPool<int>.Shared.Return(cursor);
            if (rented is not null)
            {
                ArrayPool<int>.Shared.Return(rented);
            }
        }
    }

    public override void ExpertGatedPairKernel(Storage x, Storage offsets, ReadOnlySpan<(Storage Gate, Storage? GateBias, Storage Up, Storage? UpBias)> experts, Storage hidden,
        int rows, int k, int n, int activation)
    {
        float[] xv = D(x), ov = D(offsets), hv = D(hidden);
        for (int e = 0; e < experts.Length; e++)
        {
            var (first, count) = ExpertRows(ov, e, rows);
            if (count == 0)
            {
                continue;
            }

            var (gate, gateBias, up, upBias) = experts[e];
            int size = count * n;
            float[] g = ArrayPool<float>.Shared.Rent(size), u = ArrayPool<float>.Shared.Rent(size);
            try
            {
                Project(xv, first * k, D(gate), gateBias, g, 0, count, n, k);
                Project(xv, first * k, D(up), upBias, u, 0, count, n, k);
                int at = first * n;
                // As GatedActivationKernel over the expert's own tensor: chunks of whole vectors, the tail at its end.
                For(size, (long)size * 8, (lo, hi) =>
                {
                    CpuMath.Gated(g.AsSpan(lo, hi - lo), u.AsSpan(lo, hi - lo), hv.AsSpan(at + lo, hi - lo), activation);
                }, wholeVectors: true);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(g);
                ArrayPool<float>.Shared.Return(u);
            }
        }
    }

    public override void ExpertMatMulKernel(Storage x, Storage offsets, ReadOnlySpan<(Storage Weight, Storage? Bias)> experts, Storage y, int rows, int k, int n)
    {
        float[] xv = D(x), ov = D(offsets), yv = D(y);
        for (int e = 0; e < experts.Length; e++)
        {
            var (first, count) = ExpertRows(ov, e, rows);
            if (count > 0)
            {
                var (weight, bias) = experts[e];
                Project(xv, first * k, D(weight), bias, yv, first * n, count, n, k);
            }
        }
    }

    // The first row and the row count of expert e, from the offsets RouteTopK wrote (checked against the rows given).
    private static (int First, int Count) ExpertRows(float[] offsets, int e, int rows)
    {
        int first = (int)offsets[e], end = (int)offsets[e + 1];
        if (first < 0 || end < first || end > rows)
        {
            throw new ArgumentException($"Expert {e}'s rows {first} to {end} are not within the {rows} rows given (offsets as RouteTopK writes them).");
        }

        return (first, end - first);
    }

    // c[rows] = a[rows]·w + bias, as a layer's product computes it: the product (beta 0: c is only written), then the bias
    // added to each row.
    private static void Project(float[] a, int aOffset, float[] w, Storage? bias, float[] c, int cOffset, int rows, int n, int k)
    {
        CpuMatMul.Multiply(a, aOffset, w, 0, c, cOffset, rows, n, k, false, false, 0f);
        if (bias is not null)
        {
            var b = D(bias).AsSpan(0, n);
            for (int r = 0; r < rows; r++)
            {
                var row = c.AsSpan(cOffset + r * n, n);
                AddLoop.Apply(row, b, row);
            }
        }
    }
}
