// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;

namespace Idrak.Abstraction.Devices.Cpu;

// The mixture-of-experts operations (Backend.Routing.cs): the routing (each token's k most probable experts, ties to the
// lowest index, in increasing expert order; the weights' sum in that order), and each expert's products over its rows,
// the float32 product over exactly those rows (each output fused in k's order from zero, so the rows a product covers
// never change a value), the bias added after it, and the gated activation over those rows (element-wise: the same
// bits by any split). Indices and offsets are read and written as their storages' types hold them (IndicesOf).
internal sealed unsafe partial class CpuBackend
{
    public override void RouteTopKKernel(Storage probabilities, Storage weights, Storage offsets, Storage tokenOfRow, Storage slotRow, int tokens, int experts, int k,
        bool normalize)
    {
        int slots = tokens * k;
        int[] choices = ArrayPool<int>.Shared.Rent(Math.Max(1, slots));
        int[] next = ArrayPool<int>.Shared.Rent(Math.Max(1, experts));
        int[] best = ArrayPool<int>.Shared.Rent(Math.Max(1, k));
        try
        {
            fixed (float* pp = V(probabilities), pw = V(weights), po = D(offsets), pt = D(tokenOfRow), ps = D(slotRow))
            {
                var counts = next.AsSpan(0, experts);
                counts.Clear();
                var chosen = best.AsSpan(0, k);
                for (int i = 0; i < tokens; i++)
                {
                    float* row = pp + (long)i * experts;
                    int count = 0;
                    for (int e = 0; e < experts; e++)
                    {
                        // e goes after every chosen expert at least as probable (an earlier index wins a tie).
                        int at = count;
                        while (at > 0 && row[e] > row[chosen[at - 1]])
                        {
                            at--;
                        }

                        if (at >= k)
                        {
                            continue;
                        }

                        for (int j = Math.Min(count, k - 1); j > at; j--)
                        {
                            chosen[j] = chosen[j - 1];
                        }

                        chosen[at] = e;
                        count = Math.Min(count + 1, k);
                    }

                    chosen.Sort();
                    float sum = 0f;
                    for (int s = 0; s < k; s++)
                    {
                        sum += row[chosen[s]];
                    }

                    for (int s = 0; s < k; s++)
                    {
                        int e = chosen[s];
                        choices[i * k + s] = e;
                        pw[(long)i * k + s] = normalize ? row[e] / sum : row[e];
                        counts[e]++;
                    }
                }

                // The rows expert by expert, each expert's in slot order.
                var offsetsOut = IndicesOf(offsets, po);
                var tokensOut = IndicesOf(tokenOfRow, pt);
                var slotsOut = IndicesOf(slotRow, ps);
                int start = 0;
                for (int e = 0; e < experts; e++)
                {
                    int c = counts[e];
                    offsetsOut.Set(e, start);
                    counts[e] = start;                                             // now the expert's next row
                    start += c;
                }

                offsetsOut.Set(experts, slots);
                for (int slot = 0; slot < slots; slot++)
                {
                    int row = counts[choices[slot]]++;
                    tokensOut.Set(row, slot / k);
                    slotsOut.Set(slot, row);
                }
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(choices);
            ArrayPool<int>.Shared.Return(next);
            ArrayPool<int>.Shared.Return(best);
        }
    }

    public override void ExpertGatedPairKernel(Storage x, Storage offsets, ReadOnlySpan<(Storage Gate, Storage? GateBias, Storage Up, Storage? UpBias)> experts, Storage hidden,
        int rows, int k, int n, int activation)
    {
        var gated = ArrayPool<float>.Shared.Rent(Math.Max(1, rows * n));
        try
        {
            fixed (float* px = V(x), po = D(offsets), ph = V(hidden), pg = gated)
            {
                var bounds = IndicesOf(offsets, po);
                for (int e = 0; e < experts.Length; e++)
                {
                    var (first, count) = ExpertRows(bounds, e, rows);
                    if (count == 0)
                    {
                        continue;
                    }

                    var (gate, gateBias, up, upBias) = experts[e];
                    float* xRows = px + (long)first * k, into = ph + (long)first * n;
                    Project(xRows, gate, gateBias, pg, count, n, k);
                    Project(xRows, up, upBias, into, count, n, k);
                    Gated(pg, into, into, (long)count * n, activation);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(gated);
        }
    }

    public override void ExpertMatMulKernel(Storage x, Storage offsets, ReadOnlySpan<(Storage Weight, Storage? Bias)> experts, Storage y, int rows, int k, int n)
    {
        fixed (float* px = V(x), po = D(offsets), py = V(y))
        {
            var bounds = IndicesOf(offsets, po);
            for (int e = 0; e < experts.Length; e++)
            {
                var (first, count) = ExpertRows(bounds, e, rows);
                if (count > 0)
                {
                    var (weight, bias) = experts[e];
                    Project(px + (long)first * k, weight, bias, py + (long)first * n, count, n, k);
                }
            }
        }
    }

    // The first row and the row count of expert e, from the offsets RouteTopK wrote (checked against the rows given).
    private static (int First, int Count) ExpertRows(Indices offsets, int e, int rows)
    {
        long first = offsets[e], end = offsets[e + 1];
        if (first < 0 || end < first || end > rows)
        {
            throw new ArgumentException($"Expert {e}'s rows {first} to {end} are not within the {rows} rows given (offsets as RouteTopK writes them).");
        }

        return ((int)first, (int)(end - first));
    }

    // c [rows, n] = a [rows, k] · w [k, n], then the bias added to each row, as a layer's product and bias.
    private static void Project(float* a, Storage w, Storage? bias, float* c, int rows, int n, int k)
    {
        fixed (float* pw = V(w), pb = bias is null ? null : V(bias))
        {
            Product(a, pw, null, c, 1, rows, n, k, false, false, 0f);
            if (pb != null)
            {
                Rows("AddRowVector", new RowVectorRows(c, pb, c, n, rows, gelu: false), rows, n, 1);
            }
        }
    }
}
