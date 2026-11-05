// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// The mixture-of-experts operations (plan 37, R0; Backend.Routing.cs) as kernels: the routing on the device, as the CPU's
// RouteTopK computes it (the same comparisons and the same sum, each weight's division correctly rounded in integer
// arithmetic, so the weights are the host routing's bit for bit), and every expert's gated pair and product over the rows
// its offsets give, read where they are: nothing goes back to the host, and a recorded step replays them. Each output
// element of an expert's product is one fused multiply-add chain over k from zero, in order (rule 86): the same sum
// whatever rows the expert has and however the work is spread.
internal static partial class VulkanKernels
{
    /// <summary>Rows of an expert's product a workgroup takes at a time (its invocations over the columns, each row in turn).</summary>
    public const int ExpertRows = 4;

    /// <summary>The most experts per token the device's routing takes (a token's list in the invocation's own memory); more take the host fallback.</summary>
    public const int MaxRouteTopK = 64;

    private static IEnumerable<(string Name, Func<SpirvKernel> Build)> RoutingKernels()
    {
        // Each token's k most probable experts (ties to the lowest index), listed in increasing expert order into choices
        // [tokens · k] (int32), and their weights: the probabilities, or each divided by the sum of the chosen ones added
        // in increasing expert order. One invocation per token, as the CPU's RouteTopK walks a row; the token's list is
        // kept in the invocation's own memory (at most MaxRouteTopK) and written out once.
        yield return ("route_choose", () =>
        {
            var k = new KernelBuilder("route_choose", Block);
            var (probabilities, weights, choices) = (k.Buffer("probabilities"), k.Buffer("weights"), k.Buffer("choices"));
            var (tokens, experts, top, normalize) = (k.PushInt("tokens"), k.PushInt("experts"), k.PushInt("k"), k.PushInt("normalize"));
            var best = k.LocalArray(MaxRouteTopK);
            Grid(k, tokens, i =>
            {
                var row = i * experts;
                var first = i * top;
                // The chosen expert at place `at` of the list, the place clamped into it (read only where it counts).
                Val Chosen(Val at) => best[k.Clamp(at, k.Int(0), top - 1)].AsInt();
                var count = k.Local(0);
                k.For(k.Int(0), experts, e =>
                {
                    // Insert e after every chosen expert at least as probable (earlier indices win ties): walk back from
                    // the list's end while e is more probable than the one before.
                    var p = probabilities[row + e];
                    var at = k.Local(count.V);
                    var walking = k.Local(1);
                    k.For(k.Int(0), count.V, _ => k.If(walking.V.Ne(0), () =>
                        k.If(p > probabilities[row + Chosen(at.V - 1)], () => at.V = at.V - 1, () => walking.V = k.Int(0))));
                    k.If(at.V < top, () =>
                    {
                        var last = k.Min(count.V, top - 1);
                        k.For(k.Int(0), last - at.V, t =>
                        {
                            var j = last - t;
                            best[j] = best[j - 1];
                        });
                        best[at.V] = e.AsFloat();
                        count.V = k.Min(count.V + 1, top);
                    });
                });

                // The k choices in increasing expert order (insertion sort; they are distinct).
                k.For(k.Int(1), top, a =>
                {
                    var value = Chosen(a);
                    var b = k.Local(a);
                    var moving = k.Local(1);
                    k.For(k.Int(0), a, _ => k.If(moving.V.Ne(0), () =>
                        k.If(Chosen(b.V - 1) > value, () =>
                        {
                            best[b.V] = best[b.V - 1];
                            b.V = b.V - 1;
                        }, () => moving.V = k.Int(0))));
                    best[b.V] = value.AsFloat();
                });

                var sum = k.Local(0f);
                k.For(k.Int(0), top, s => sum.V = k.AddRounded(sum.V, probabilities[row + Chosen(s)]));
                k.For(k.Int(0), top, s =>
                {
                    var p = probabilities[row + Chosen(s)];
                    var weight = k.Local(p);
                    k.If(normalize.Ne(0), () => weight.V = DivideRounded(k, p, sum.V));
                    weights[first + s] = weight.V;
                    choices[first + s] = best[s];
                });
            });
            return k.Build();
        });

        // From the choices, in three dispatches whose every buffer is only read or only written (no barrier): each
        // expert's slot count; then, by one invocation, offsets[e] (expert e's first row when the slots are listed expert by
        // expert, each expert's in slot order; offsets[experts] = slots); then each expert's rows: tokenOfRow[row] and
        // slotRow[slot] (int32). One invocation per expert walks the slots in order.
        yield return ("route_counts", () =>
        {
            var k = new KernelBuilder("route_counts", Block);
            var (choices, counts) = (k.Buffer("choices"), k.Buffer("counts"));
            var (slots, experts) = (k.PushInt("slots"), k.PushInt("experts"));
            Grid(k, experts, e =>
            {
                var count = k.Local(0);
                k.For(k.Int(0), slots, s => k.If(choices.Int(s).Eq(e), () => count.V = count.V + 1));
                counts[e] = count.V;
            });
            return k.Build();
        });

        yield return ("route_offsets", () =>
        {
            var k = new KernelBuilder("route_offsets", Block);
            var (counts, offsets) = (k.Buffer("counts"), k.Buffer("offsets"));
            var (slots, experts) = (k.PushInt("slots"), k.PushInt("experts"));
            Grid(k, k.Int(1), _ =>
            {
                var start = k.Local(0);
                k.For(k.Int(0), experts, e =>
                {
                    offsets[e] = start.V;
                    start.V = start.V + counts.Int(e);
                });
                offsets[experts] = slots;
            });
            return k.Build();
        });

        yield return ("route_place", () =>
        {
            var k = new KernelBuilder("route_place", Block);
            var (choices, offsets, tokenOfRow, slotRow) = (k.Buffer("choices"), k.Buffer("offsets"), k.Buffer("tokenOfRow"), k.Buffer("slotRow"));
            var (slots, experts, top) = (k.PushInt("slots"), k.PushInt("experts"), k.PushInt("k"));
            Grid(k, experts, e =>
            {
                var cursor = k.Local(offsets.Int(e));
                k.For(k.Int(0), slots, s => k.If(choices.Int(s).Eq(e), () =>
                {
                    tokenOfRow[cursor.V] = s / top;
                    slotRow[s] = cursor.V;
                    cursor.V = cursor.V + 1;
                }));
            });
            return k.Build();
        });

        // Each expert's workgroups for its gated pair and product, from the offsets where they are (plan 37, R0d): counts
        // [3 · experts] (int32) holds expert e's dispatch at 3e (x = its row blocks of ExpertRows, at most `most`; y = z =
        // 1), read by vkCmdDispatchIndirect, so an expert without rows dispatches no workgroup.
        yield return ("expert_sizes", () =>
        {
            var k = new KernelBuilder("expert_sizes", Block);
            var (offsets, counts) = (k.Buffer("offsets"), k.Buffer("counts"));
            var (experts, most) = (k.PushInt("experts"), k.PushInt("most"));
            Grid(k, experts, e =>
            {
                var rows = offsets.Int(e + 1) - offsets.Int(e);
                counts[e * 3] = k.Min((rows + (ExpertRows - 1)) / ExpertRows, most);
                counts[e * 3 + 1] = k.Int(1);
                counts[e * 3 + 2] = k.Int(1);
            });
            return k.Build();
        });

        // One expert's gate and up projections over its rows (offsets[expert] to offsets[expert + 1] of x [rows, k]) and
        // their gated activation into those rows of hidden [rows, n]: gate and up [k, n], biases [n] where flagged. The
        // workgroups take blocks of ExpertRows rows from the expert's first, their invocations the columns.
        yield return ("expert_gated_pair", () =>
        {
            var k = new KernelBuilder("expert_gated_pair", Block);
            var (x, offsets, gate, gateBias, up, upBias, hidden) = (k.Buffer("x"), k.Buffer("offsets"), k.Buffer("gate"), k.Buffer("gateBias"), k.Buffer("up"),
                k.Buffer("upBias"), k.Buffer("hidden"));
            var (expert, depth, n, activation) = (k.PushInt("expert"), k.PushInt("k"), k.PushInt("n"), k.PushInt("activation"));
            var (gateBiased, upBiased) = (k.PushInt("gateBiased"), k.PushInt("upBiased"));
            ExpertRowBlocks(k, offsets, expert, n, (r, j) =>
            {
                var (g, u) = (k.Local(0f), k.Local(0f));
                k.For(k.Int(0), depth, p =>
                {
                    var input = x[r * depth + p];
                    g.V = k.Fma(input, gate[p * n + j], g.V);
                    u.V = k.Fma(input, up[p * n + j], u.V);
                });
                k.If(gateBiased.Ne(0), () => g.V = k.AddRounded(g.V, gateBias[j]));
                k.If(upBiased.Ne(0), () => u.V = k.AddRounded(u.V, upBias[j]));
                var (value, _) = Activation(k, g.V, activation, slope: false);
                var (upValue, _) = UpSide(k, u.V, activation, slope: false);
                hidden[r * n + j] = value.V * upValue.V;
            });
            return k.Build();
        });

        // One expert's product over its rows: those rows of y [rows, n] = x·weight + bias, weight [k, n], bias [n] where flagged.
        yield return ("expert_matmul", () =>
        {
            var k = new KernelBuilder("expert_matmul", Block);
            var (x, offsets, weight, bias, y) = (k.Buffer("x"), k.Buffer("offsets"), k.Buffer("weight"), k.Buffer("bias"), k.Buffer("y"));
            var (expert, depth, n, biased) = (k.PushInt("expert"), k.PushInt("k"), k.PushInt("n"), k.PushInt("biased"));
            ExpertRowBlocks(k, offsets, expert, n, (r, j) =>
            {
                var sum = k.Local(0f);
                k.For(k.Int(0), depth, p => sum.V = k.Fma(x[r * depth + p], weight[p * n + j], sum.V));
                k.If(biased.Ne(0), () => sum.V = k.AddRounded(sum.V, bias[j]));
                y[r * n + j] = sum.V;
            });
            return k.Build();
        });
    }

    // body(row, column) over expert `expert`'s rows (offsets[expert] to offsets[expert + 1], read on the device) and the n
    // columns: workgroup g takes the expert's row blocks g, g + groups, … of ExpertRows rows, its invocations the columns.
    private static void ExpertRowBlocks(KernelBuilder k, Buf offsets, Val expert, Val n, Action<Val, Val> body)
    {
        var first = offsets.Int(expert);
        var end = offsets.Int(expert + 1);
        var blocks = (end - first + (ExpertRows - 1)) / ExpertRows;
        k.For(k.GroupX, blocks, b =>
        {
            var start = first + b * ExpertRows;
            var stop = k.Min(start + ExpertRows, end);
            k.For(k.LocalX, n, j => k.For(start, stop, r => body(r, j)), k.Int(Block));
        }, k.GroupsX);
    }

    // a / b correctly rounded (to nearest, ties to even) in integer arithmetic on their bits, whatever the device's
    // division: for a ≥ 0 (zero, subnormal or normal) and a normal b > 0 with a ≤ b (a routing weight's share), as the
    // CPU's division gives it. The mantissas' long division yields the 24 bits of the result, a guard bit and whether a
    // remainder is left; a result below the normal range is shifted into a subnormal before it rounds.
    private static Val DivideRounded(KernelBuilder k, Val a, Val b)
    {
        var result = k.Local(0);
        var (abits, bbits) = (a.AsInt(), b.AsInt());
        k.If(abits.Ne(0), () =>
        {
            var ea = k.Local((abits >> 23) & 0xFF);
            var ma = k.Local(abits & 0x7FFFFF);
            k.If(ea.V.Eq(0), () =>
            {
                ea.V = k.Int(1);                                           // a subnormal: normalize its mantissa
                k.While(() => (ma.V & 0x800000).Eq(0), () =>
                {
                    ma.V = ma.V << 1;
                    ea.V = ea.V - 1;
                });
            }, () => ma.V = ma.V | 0x800000);
            var mb = (bbits & 0x7FFFFF) | 0x800000;
            var e = k.Local(ea.V - ((bbits >> 23) & 0xFF) + 127);
            k.If(ma.V < mb, () =>
            {
                ma.V = ma.V << 1;
                e.V = e.V - 1;
            });

            // 25 quotient bits (the leading 1, 23 more, a guard bit); the remainder stays below 2 · mb < 2^25.
            var q = k.Local(0);
            var r = k.Local(ma.V);
            k.For(k.Int(0), k.Int(25), _ =>
            {
                q.V = q.V << 1;
                k.If(r.V >= mb, () =>
                {
                    r.V = r.V - mb;
                    q.V = q.V | 1;
                });
                r.V = r.V << 1;
            });

            var sticky = r.V.Ne(0);
            k.If(e.V >= 1, () =>
            {
                var mantissa = k.Local(q.V >> 1);
                k.If((q.V & 1).Ne(0) & (sticky | (mantissa.V & 1).Ne(0)), () => mantissa.V = mantissa.V + 1);
                // A carry past 24 bits moves into the exponent through the addition itself.
                result.V = (e.V << 23) + mantissa.V - 0x800000;
            }, () =>
            {
                var shift = k.Min(k.Int(1) - e.V, k.Int(26));                // the bits a subnormal result drops
                var mantissa = k.Local(q.V.ShiftRight(shift + 1));
                var guard = (q.V.ShiftRight(shift) & 1).Ne(0);
                var below = (q.V & ((k.Int(1).ShiftLeft(shift)) - 1)).Ne(0);
                k.If(guard & (sticky | below | (mantissa.V & 1).Ne(0)), () => mantissa.V = mantissa.V + 1);
                result.V = mantissa.V;
            });
        });
        return result.V.AsFloat();
    }
}
