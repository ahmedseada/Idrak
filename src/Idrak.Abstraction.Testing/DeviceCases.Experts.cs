// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction.Testing;

// Mixture of experts: the routing (RouteTopK) and the experts' products over rows grouped by expert (ExpertGatedPair,
// ExpertMatMul), against plain loops.
public static partial class DeviceCases
{
    private static void Experts(DeviceCaseContext c)
    {
        var b = c.Backend;
        foreach (var (tokens, experts, k, normalize) in new[]
                 {
                     (1, 1, 1, true), (5, 4, 1, false), (6, 4, 4, true), (c.Size(2, 40), c.Size(2, 9), 2, true), (c.Size(2, 40), 7, 3, false),
                 })
        {
            // Probabilities rounded to tenths: many ties, which go to the lowest index.
            float[] pv = [.. c.Values(tokens * experts).Select(v => MathF.Round(MathF.Abs(v), 1) + 0.05f)];
            int slots = tokens * k;
            Storage weights = c.Zeros(slots), offsets = c.Zeros(experts + 1), tokenOfRow = c.Zeros(slots), slotRow = c.Zeros(slots);
            b.RouteTopK(c.Storage(pv), weights, offsets, tokenOfRow, slotRow, tokens, experts, k, normalize);

            // The expected routing: a stable sort by decreasing probability, the first k listed in increasing order.
            var choices = new int[slots];
            var expectedWeights = new float[slots];
            for (int i = 0; i < tokens; i++)
            {
                var chosen = Enumerable.Range(0, experts).OrderByDescending(e => pv[i * experts + e]).Take(k).Order().ToArray();
                float sum = 0f;
                foreach (int e in chosen)
                {
                    sum += pv[i * experts + e];
                }

                for (int s = 0; s < k; s++)
                {
                    choices[i * k + s] = chosen[s];
                    expectedWeights[i * k + s] = normalize ? pv[i * experts + chosen[s]] / sum : pv[i * experts + chosen[s]];
                }
            }

            var expectedOffsets = new float[experts + 1];
            var expectedTokens = new float[slots];
            var expectedRows = new float[slots];
            int row = 0;
            for (int e = 0; e < experts; e++)
            {
                expectedOffsets[e] = row;
                for (int slot = 0; slot < slots; slot++)
                {
                    if (choices[slot] == e)
                    {
                        expectedTokens[row] = slot / k;
                        expectedRows[slot] = row++;
                    }
                }
            }

            expectedOffsets[experts] = slots;
            string shape = $"{tokens} tokens, {experts} experts, k {k}{(normalize ? ", normalized" : "")}";
            c.ExpectClose(expectedWeights, Read(weights), 1e-6f, $"routing weights ({shape})");
            c.ExpectClose(expectedOffsets, Read(offsets), 0f, $"routing offsets ({shape})");
            c.ExpectClose(expectedTokens, Read(tokenOfRow), 0f, $"token of each row ({shape})");
            c.ExpectClose(expectedRows, Read(slotRow), 0f, $"row of each slot ({shape})");

            // The experts' products over those rows: odd widths, biases on some experts only, every activation kind.
            int dim = c.Size(1, 13), hidden = c.Size(1, 11);
            float[] xv = c.Values(slots * dim);
            var gates = new float[experts][];
            var ups = new float[experts][];
            var downs = new float[experts][];
            var gateBiases = new float[]?[experts];
            var upBiases = new float[]?[experts];
            var downBiases = new float[]?[experts];
            var pairs = new (Storage Gate, Storage? GateBias, Storage Up, Storage? UpBias)[experts];
            var products = new (Storage Weight, Storage? Bias)[experts];
            for (int e = 0; e < experts; e++)
            {
                (gates[e], ups[e], downs[e]) = (c.Values(dim * hidden), c.Values(dim * hidden), c.Values(hidden * dim));
                (gateBiases[e], upBiases[e], downBiases[e]) = e % 2 == 1 ? (c.Values(hidden), c.Values(hidden), c.Values(dim)) : (null, null, null);
                pairs[e] = (c.Storage(gates[e]), gateBiases[e] is { } gb ? c.Storage(gb) : null, c.Storage(ups[e]), upBiases[e] is { } ub ? c.Storage(ub) : null);
                products[e] = (c.Storage(downs[e]), downBiases[e] is { } db ? c.Storage(db) : null);
            }

            // The expert of each row (rows expert by expert).
            var expertOfRow = new int[slots];
            for (int slot = 0; slot < slots; slot++)
            {
                expertOfRow[(int)expectedRows[slot]] = choices[slot];
            }

            var x = c.Storage(xv);
            Storage h = null!;                                                    // the last kind's, read by the down products
            float[] hv = [];
            foreach (int kind in new[] { 0, 1, 2, 3 })
            {
                h = c.Zeros(slots * hidden);
                b.ExpertGatedPair(x, offsets, pairs, h, slots, dim, hidden, kind);
                var expectedHidden = new float[slots * hidden];
                for (int r = 0; r < slots; r++)
                {
                    int e = expertOfRow[r];
                    for (int j = 0; j < hidden; j++)
                    {
                        float g = gateBiases[e]?[j] ?? 0f, u = upBiases[e]?[j] ?? 0f;
                        for (int p = 0; p < dim; p++)
                        {
                            g += xv[r * dim + p] * gates[e][p * hidden + j];
                            u += xv[r * dim + p] * ups[e][p * hidden + j];
                        }

                        expectedHidden[r * hidden + j] = GatedReference(g, u, kind);
                    }
                }

                hv = Read(h);
                c.ExpectClose(expectedHidden, hv, 2e-4f, $"experts' gated pair, kind {kind} ({shape})");
            }

            var y = c.Zeros(slots * dim);
            b.ExpertMatMul(h, offsets, products, y, slots, hidden, dim);
            var expectedY = new float[slots * dim];
            for (int r = 0; r < slots; r++)
            {
                int e = expertOfRow[r];
                for (int j = 0; j < dim; j++)
                {
                    float sum = downBiases[e]?[j] ?? 0f;
                    for (int p = 0; p < hidden; p++)
                    {
                        sum += hv[r * hidden + p] * downs[e][p * dim + j];
                    }

                    expectedY[r * dim + j] = sum;
                }
            }

            c.ExpectClose(expectedY, Read(y), 2e-4f, $"experts' products ({shape})");
        }
    }

    // act(gate) · up for the gated activation kinds (0 SiLU, 1 GELU tanh, 2 ReLU, 3 tanh · sigmoid).
    private static float GatedReference(float gate, float up, int kind) => kind switch
    {
        0 => gate / (1f + MathF.Exp(-gate)) * up,
        1 => 0.5f * gate * (1f + MathF.Tanh(0.7978845608f * (gate + 0.044715f * gate * gate * gate))) * up,
        3 => MathF.Tanh(gate) / (1f + MathF.Exp(-up)),
        _ => MathF.Max(gate, 0f) * up,
    };
}
