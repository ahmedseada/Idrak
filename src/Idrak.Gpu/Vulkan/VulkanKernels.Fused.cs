// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// Fused decoding kernels (Backend.FusedKernels): the attention layer's norms, rotary positions and head layout (or cache
// writes) in one row pass.
internal static partial class VulkanKernels
{
    /// <summary>Largest head size of <c>norm_rope_heads</c> (its row is staged in workgroup memory).</summary>
    public const int NormRopeMaxCols = AttentionMaxDim;

    private static IEnumerable<(string, Func<SpirvKernel>)> FusedKernels()
    {
        // The attention layer's projections in the layouts attention reads (Backend.NormRopeHeads): rows1 query rows of q,
        // then rows2 key rows of k and rows2 value rows of v (row = (b · steps + s) · heads + h, heads = kvHeads for keys
        // and values). A workgroup per row: the row is staged in workgroup memory, query and key rows RMS-normalized with
        // their gain when flags bit 0 (as rms_norm_rope: x · inv · (gain + offset)) and turned by the rotary tables (pairs
        // p < half, as rope; half 0: none); values copied. Queries go to yq [batch, heads, steps, cols]; keys and values to
        // yk / yv [batch, kvHeads, capacity, stride] at row offset + s (offset = position[0] when flags bit 2, else 0).
        //
        // Workgroup memory: row[j] is written by invocation j % width while loading (the reduction's barriers follow), then
        // each element by one invocation while normalizing and turning (pairs (i, j) by the invocation of p, the rest by
        // the invocation of their index), read for the stores after a barrier; a barrier ends the row.
        yield return ("norm_rope_heads", () =>
        {
            var k = new KernelBuilder("norm_rope_heads", Block);
            var (q, key, value) = (k.Buffer("q"), k.Buffer("k"), k.Buffer("v"));
            var (gainQ, gainK) = (k.Buffer("gainQ"), k.Buffer("gainK"));
            var (cos, sin, positions, position) = (k.Buffer("cos"), k.Buffer("sin"), k.Buffer("positions"), k.Buffer("position"));
            var (yq, yk, yv) = (k.Buffer("yq"), k.Buffer("yk"), k.Buffer("yv"));
            var (rows1, rows2, heads, kvHeads, steps) = (k.PushInt("rows1"), k.PushInt("rows2"), k.PushInt("heads"), k.PushInt("kvHeads"), k.PushInt("steps"));
            var (cols, half, interleaved, flags) = (k.PushInt("cols"), k.PushInt("half"), k.PushInt("interleaved"), k.PushInt("flags"));
            var (capacity, stride) = (k.PushInt("capacity"), k.PushInt("stride"));
            var (epsQ, offsetQ, epsK, offsetK) = (k.PushFloat("epsQ"), k.PushFloat("offsetQ"), k.PushFloat("epsK"), k.PushFloat("offsetK"));
            var row = k.Shared("row", NormRopeMaxCols);
            var scratch = k.Shared("scratch", Block);
            var lane = k.LocalX;
            var (normed, cached) = ((flags & 1).Ne(0), (flags & 4).Ne(0));
            var pairs = interleaved.Ne(0);
            var start = k.Local(k.Int(0));
            k.If(cached, () => start.V = position[k.Int(0)].ToInt());
            EachRow(k, rows1 + 2 * rows2, at =>
            {
                var segment = k.Select(at < rows1, k.Int(0), k.Select(at < rows1 + rows2, k.Int(1), k.Int(2)));
                var r = at - k.Select(segment.Eq(0), k.Int(0), k.Select(segment.Eq(1), rows1, rows1 + rows2));
                var perStep = k.Select(segment.Eq(0), heads, kvHeads);
                var (h, t) = (r % perStep, r / perStep);
                var (s, b) = (t % steps, t / steps);
                var squares = k.Local(0f);
                k.For(lane, cols, Block, j =>
                {
                    var x = k.Local(ScalarKind.Float);
                    k.If(segment.Eq(0), () => x.V = q[r * cols + j], () => k.If(segment.Eq(1), () => x.V = key[r * cols + j], () => x.V = value[r * cols + j]));
                    row[j] = x.V;
                    squares.V = k.Fma(x.V, x.V, squares.V);
                });
                var total = k.ReduceSum(scratch, squares.V);                  // its barriers also order the row's stores
                var norm = normed & (segment < 2);
                var query = segment.Eq(0);
                var scale = k.Select(norm, 1f / k.Sqrt(total / cols.ToFloat() + k.Select(query, epsQ, epsK)), k.Float(1f));
                var shift = k.Select(query, offsetQ, offsetK);
                var turned = k.Select(segment < 2, half, k.Int(0));

                // (gain[j] + offset), or 1 without a norm.
                Val Gain(Val j)
                {
                    var g = k.Local(1f);
                    k.If(norm, () => k.If(query, () => g.V = gainQ[j] + shift, () => g.V = gainK[j] + shift));
                    return g.V;
                }

                k.For(lane, turned, Block, p =>
                {
                    var position = positions[s].ToInt();
                    var i = k.Select(pairs, 2 * p, p);
                    var j = k.Select(pairs, 2 * p + 1, p + half);
                    var a = row[i] * scale * Gain(i);
                    var c = row[j] * scale * Gain(j);
                    var co = cos[position * half + p];
                    var si = sin[position * half + p];
                    row[i] = k.Fma(a, co, -(c * si));
                    row[j] = k.Fma(c, co, a * si);
                });
                k.For(lane, cols - 2 * turned, Block, e =>
                {
                    var j = 2 * turned + e;
                    row[j] = row[j] * scale * Gain(j);
                });
                k.Barrier();
                k.If(query, () =>
                {
                    var first = ((b * heads + h) * steps + s) * cols;
                    k.For(lane, cols, Block, j => yq[first + j] = row[j]);
                }, () =>
                {
                    var slot = (b * kvHeads + h) * capacity + start.V + s;
                    var isKey = segment.Eq(1);
                    var first = slot * stride;
                    k.For(lane, cols, Block, j => k.If(isKey, () => yk[first + j] = row[j], () => yv[first + j] = row[j]));
                });
                k.Barrier();                                                  // the row is read before the next one is staged
            });
            return k.Build();
        });
    }
}
