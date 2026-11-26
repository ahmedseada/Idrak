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
            fixed (float* pp = D(probabilities), pw = V(weights), po = D(offsets), pt = D(tokenOfRow), ps = D(slotRow))
            {
                var counts = next.AsSpan(0, experts);
                counts.Clear();
                var chosen = best.AsSpan(0, k);
                var tp = new TypedRanges(probabilities, 0, "RouteTopK");          // a typed token's row read through its type
                for (int i = 0; i < tokens; i++)
                {
                    float* row = tp.At(pp, (long)i * experts, experts) + (long)i * experts;
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
            fixed (float* px = D(x), po = D(offsets), ph = V(hidden), pg = gated)
            {
                var bounds = IndicesOf(offsets, po);
                int step = ExpertStep(x, rows, k);
                var tx = new TypedInput(x, px, 0, (long)step * k, "ExpertGatedPair");
                for (int e = 0; e < experts.Length; e++)
                {
                    var (first, count) = ExpertRows(bounds, e, rows);
                    var (gate, gateBias, up, upBias) = experts[e];
                    for (int r = first; r < first + count; r += step)
                    {
                        // A typed x's rows read through its type a chunk of rows at a time (each output row its own sum).
                        int taken = Math.Min(step, first + count - r);
                        float* xRows = tx.Range((long)r * k, (long)taken * k) + (long)r * k, into = ph + (long)r * n;
                        Project(xRows, gate, gateBias, pg, taken, n, k);
                        Project(xRows, up, upBias, into, taken, n, k);
                        Gated(pg, into, into, (long)taken * n, activation);
                    }
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
        fixed (float* px = D(x), po = D(offsets), py = V(y))
        {
            var bounds = IndicesOf(offsets, po);
            int step = ExpertStep(x, rows, k);
            var tx = new TypedInput(x, px, 0, (long)step * k, "ExpertMatMul");
            for (int e = 0; e < experts.Length; e++)
            {
                var (first, count) = ExpertRows(bounds, e, rows);
                var (weight, bias) = experts[e];
                for (int r = first; r < first + count; r += step)
                {
                    int taken = Math.Min(step, first + count - r);
                    Project(tx.Range((long)r * k, (long)taken * k) + (long)r * k, weight, bias, py + (long)r * n, taken, n, k);
                }
            }
        }
    }

    // The rows of x an expert's product takes at a time: all where x is float32's own, else a chunk's worth (whole tiles).
    private static int ExpertStep(Storage x, int rows, int k) =>
        NativeFloats(x) ? Math.Max(1, rows) : Math.Max(TileRows, ChunkValues / Math.Max(1, k) / TileRows * TileRows);

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

    // c [rows, n] = a [rows, k] · w [k, n], then the bias added to each row, as a layer's product and bias: a typed w read
    // through its type as the product packs it, a typed bias once, its own size.
    private static void Project(float* a, Storage w, Storage? bias, float* c, int rows, int n, int k)
    {
        fixed (float* pw = D(w), wb = bias is null ? null : D(bias))
        {
            Product(a, pw, null, c, 1, rows, n, k, false, false, 0f, bType: NativeFloats(w) ? null : TypedB(w, "the experts' product"));
            if (bias is not null)
            {
                var tb = new TypedInput(bias, wb, 3, n, "the experts' product");
                Rows("AddRowVector", new RowVectorRows(c, tb.Range(0, n), c, n, rows, gelu: false), rows, n, 1);
            }
        }
    }
}
