// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Backends.Vulkan;

// Shape kernels (strided copies, permutations, sums and broadcasts along an axis, one-hot rows, embedding gradients),
// dropout and the gradient-clipping factor.
internal static partial class VulkanKernels
{
    /// <summary>Dimensions <c>permute</c> takes: shapes are padded at the front with 1 (stride 0) to this rank.</summary>
    public const int PermuteRank = 6;

    private static IEnumerable<(string, Func<SpirvKernel>)> ShapeKernels()
    {
        // dst[dstOffset + r·dstStride + c] (+)= src[srcOffset + r·srcStride + c] for r < rows, c < cols.
        yield return ("copy_2d", () =>
        {
            var k = new KernelBuilder("copy_2d", Block);
            var (src, dst) = (k.Buffer("src"), k.Buffer("dst"));
            var (srcOffset, srcStride, dstOffset, dstStride) = (k.PushInt("srcOffset"), k.PushInt("srcStride"), k.PushInt("dstOffset"), k.PushInt("dstStride"));
            var (rows, cols, accumulate) = (k.PushInt("rows"), k.PushInt("cols"), k.PushInt("accumulate"));
            Grid(k, rows * cols, i =>
            {
                var (r, c) = (i / cols, i % cols);
                var to = dstOffset + r * dstStride + c;
                var v = src[srcOffset + r * srcStride + c];
                k.If(accumulate.Ne(0), () => dst[to] = dst[to] + v, () => dst[to] = v);
            });
            return k.Build();
        });

        // y[flat output index] (+)= x[Σ_d coordinate_d · inStride_d] over the output shape (6 dimensions, padded in front
        // with size 1 and stride 0).
        yield return ("permute", () =>
        {
            var k = new KernelBuilder("permute", Block);
            var (x, y) = (k.Buffer("x"), k.Buffer("y"));
            var shape = new Val[PermuteRank];
            var strides = new Val[PermuteRank];
            for (int d = 0; d < PermuteRank; d++)
            {
                shape[d] = k.PushInt($"shape{d}");
            }

            for (int d = 0; d < PermuteRank; d++)
            {
                strides[d] = k.PushInt($"stride{d}");
            }

            var accumulate = k.PushInt("accumulate");
            var total = shape[0];
            for (int d = 1; d < PermuteRank; d++)
            {
                total = total * shape[d];
            }

            Grid(k, total, i =>
            {
                var rest = i;
                var offset = k.Int(0);
                for (int d = PermuteRank - 1; d >= 0; d--)
                {
                    offset = offset + rest % shape[d] * strides[d];
                    rest = rest / shape[d];
                }

                var v = x[offset];
                k.If(accumulate.Ne(0), () => y[i] = y[i] + v, () => y[i] = v);
            });
            return k.Build();
        });

        // y[o, i] (+)= scale · Σ_d x[o, d, i] for a [outer, dim, inner] view: one invocation per output.
        yield return ("sum_axis", () =>
        {
            var k = new KernelBuilder("sum_axis", Block);
            var (x, y) = (k.Buffer("x"), k.Buffer("y"));
            var (outer, dim, inner, scale, accumulate) = (k.PushInt("outer"), k.PushInt("dim"), k.PushInt("inner"), k.PushFloat("scale"), k.PushInt("accumulate"));
            Grid(k, outer * inner, idx =>
            {
                var (o, i) = (idx / inner, idx % inner);
                var acc = k.Local(0f);
                k.For(k.Int(0), dim, 1, d => acc.V = acc.V + x[(o * dim + d) * inner + i]);
                var v = scale * acc.V;
                k.If(accumulate.Ne(0), () => y[idx] = y[idx] + v, () => y[idx] = v);
            });
            return k.Build();
        });

        // dx[o, d, i] += scale · dy[o, i].
        yield return ("broadcast_axis", () =>
        {
            var k = new KernelBuilder("broadcast_axis", Block);
            var (dy, dx) = (k.Buffer("dy"), k.Buffer("dx"));
            var (outer, dim, inner, scale) = (k.PushInt("outer"), k.PushInt("dim"), k.PushInt("inner"), k.PushFloat("scale"));
            Grid(k, outer * dim * inner, idx =>
            {
                var (o, i) = (idx / (dim * inner), idx % inner);
                dx[idx] = dx[idx] + scale * dy[o * inner + i];
            });
            return k.Build();
        });

        // y[i, :] = one-hot of indices[i] over classes columns (indices clamped into range; the CPU throws).
        yield return ("one_hot", () =>
        {
            var k = new KernelBuilder("one_hot", Block);
            var (indices, y) = (k.Buffer("indices"), k.Buffer("y"));
            var (count, classes) = (k.PushInt("count"), k.PushInt("classes"));
            Grid(k, count * classes, idx =>
            {
                var (i, j) = (idx / classes, idx % classes);
                var hot = k.Clamp(indices[i].ToInt(), k.Int(0), classes - 1);
                y[idx] = k.Select(j.Eq(hot), k.Float(1f), k.Float(0f));
            });
            return k.Build();
        });

        // dtable[indices[i], :] += dy[i, :]: one invocation per column, walking the indices in order (repeated indices
        // add up without atomics). TODO(tuning): parallel over the indices for narrow tables.
        yield return ("scatter_add", () =>
        {
            var k = new KernelBuilder("scatter_add", Block);
            var (dy, indices, dtable) = (k.Buffer("dy"), k.Buffer("indices"), k.Buffer("dtable"));
            var (count, dim, vocabulary) = (k.PushInt("count"), k.PushInt("dim"), k.PushInt("vocabulary"));
            Grid(k, dim, d => k.For(k.Int(0), count, 1, i =>
            {
                var at = k.Clamp(indices[i].ToInt(), k.Int(0), vocabulary - 1) * dim + d;
                dtable[at] = dtable[at] + dy[i * dim + d];
            }));
            return k.Build();
        });

        // Inverted dropout with the shared counter-based mask (DropoutMask.Keep): y = keep(i) ? x / (1 - p) : 0.
        yield return ("dropout", () => Dropout("dropout", accumulate: false));

        // dx += keep(i) ? dy / (1 - p) : 0, regenerating the same mask from the seed.
        yield return ("dropout_backward", () => Dropout("dropout_backward", accumulate: true));

        // factor[0] = min(1, maxNorm / sqrt(sumSquares[0])) (1 when the sum is 0). One invocation.
        yield return ("clip_factor", () =>
        {
            var k = new KernelBuilder("clip_factor", Block);
            var (sumSquares, factor) = (k.Buffer("sumSquares"), k.Buffer("factor"));
            var maxNorm = k.PushFloat("maxNorm");
            k.If(k.GlobalX.Eq(0), () =>
            {
                var sum = sumSquares[k.Int(0)];
                factor[k.Int(0)] = k.Select(sum > 0f, k.Min(k.Float(1f), maxNorm / k.Sqrt(sum)), k.Float(1f));
            });
            return k.Build();
        });
    }

    private static SpirvKernel Dropout(string name, bool accumulate)
    {
        var k = new KernelBuilder(name, Block);
        var (x, y) = (k.Buffer(accumulate ? "dy" : "x"), k.Buffer(accumulate ? "dx" : "y"));
        var (n, p, seed) = (k.PushInt("n"), k.PushFloat("p"), k.PushUInt("seed"));
        var scale = 1f / (1f - p);
        Grid(k, n, i =>
        {
            // MurmurHash3 finalizer of (index · golden ratio) ^ seed; the top 24 bits as a uniform number in [0, 1).
            var h = i.ToUInt() * k.UInt(0x9E3779B9u) ^ seed;
            h = h ^ h >> 16;
            h = h * k.UInt(0x85EBCA6Bu);
            h = h ^ h >> 13;
            h = h * k.UInt(0xC2B2AE35u);
            h = h ^ h >> 16;
            var u = (h >> 8).ToFloat() * (1f / 16777216f);
            var v = k.Select(u >= p, x[i] * scale, k.Float(0f));
            y[i] = accumulate ? y[i] + v : v;
        });
        return k.Build();
    }
}
