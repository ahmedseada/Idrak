// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Backends.Cuda;

// Host-side launches for the kernels in PtxKernels.Advanced.cs.
internal sealed unsafe partial class CudaBackend
{
    private readonly Dictionary<string, IntPtr> _kernels;

    private IntPtr K(string name) => _kernels[name];

    private static ulong Pn(Storage? s) => s is null ? 0UL : P(s);

    public override void Softmax(Storage x, Storage y, int rows, int cols, bool log) =>
        LaunchRows(K("softmax_f32"), rows, P(x), P(y), U(cols), U(log ? 1 : 0), U(rows));

    public override void SoftmaxBackward(Storage y, Storage dy, Storage dx, int rows, int cols, bool log) =>
        LaunchRows(K("softmax_bwd_f32"), rows, P(y), P(dy), P(dx), U(cols), U(log ? 1 : 0), U(rows));

    public override void ArgMax(Storage x, Storage y, int rows, int cols) =>
        LaunchRows(K("argmax_f32"), rows, P(x), P(y), U(cols), U(rows));

    public override void ClassMatch(Storage predictions, Storage targets, Storage y, int rows, int cols, float threshold) =>
        LaunchRows(K("class_match_f32"), rows, P(predictions), P(targets), P(y), U(cols), F(threshold), U(rows));

    public override void NormStats(Storage x, Storage mean, Storage variance, Storage invStd, int outer, int groups, int inner, float eps)
    {
        if (groups > 0)
        {
            Launch(K("norm_stats_f32"), (uint)groups, 1, (uint)_shapes.BlockSize, 1,
                P(x), P(mean), P(variance), P(invStd), U(outer), U(groups), U(inner), F(eps));
        }
    }

    public override void NormApply(Storage x, Storage mean, Storage invStd, Storage y, int outer, int groups, int inner)
    {
        int n = outer * groups * inner;
        Launch1D(K("norm_apply_f32"), n, P(x), P(mean), P(invStd), P(y), U(groups), U(inner), U(n));
    }

    public override void NormBackward(Storage dxhat, Storage xhat, Storage sum1, Storage sum2, Storage invStd, Storage dx, int outer, int groups, int inner)
    {
        int n = outer * groups * inner;
        Launch1D(K("norm_bwd_f32"), n, P(dxhat), P(xhat), P(sum1), P(sum2), P(invStd), P(dx), U(groups), U(inner), F(outer * inner), U(n));
    }

    public override void GroupScaleShift(Storage x, Storage? scale, Storage? shift, Storage y, int n, int groups, int inner, bool accumulate)
    {
        int flags = (scale is null ? 0 : 1) | (shift is null ? 0 : 2) | (accumulate ? 4 : 0);
        Launch1D(K("group_scale_shift_f32"), n, P(x), Pn(scale), Pn(shift), P(y), U(groups), U(inner), U(flags), U(n));
    }

    public override void GroupReduce(Storage a, Storage? b, Storage sumA, Storage? sumAB, int outer, int groups, int inner)
    {
        if (groups > 0)
        {
            bool hasB = b is not null && sumAB is not null;
            Launch(K("group_reduce_f32"), (uint)groups, 1, (uint)_shapes.BlockSize, 1,
                P(a), hasB ? P(b!) : P(a), P(sumA), hasB ? P(sumAB!) : P(sumA), U(outer), U(groups), U(inner), U(hasB ? 1 : 0));
        }
    }

    public override void InvSqrt(Storage x, Storage y, int n, float eps) =>
        Launch1D(K("inv_sqrt_f32"), n, P(x), P(y), F(eps), U(n));

    public override void GatherBFloat16(Storage packed, Storage indices, Storage y, int count, int dim, int vocabulary)
    {
        int n = count * dim;
        Launch1D(K("gather_bf16_f32"), n, P(packed), P(indices), P(y), U(dim), U(vocabulary - 1), U((dim + 1) / 2), U(n));
    }

    public override void Gather(Storage table, Storage indices, Storage y, int count, int dim, int vocabulary)
    {
        int n = count * dim;
        Launch1D(K("gather_f32"), n, P(table), P(indices), P(y), U(dim), U(vocabulary - 1), U(n));
    }

    public override void OneHot(Storage indices, Storage y, int count, int classes)
    {
        int n = count * classes;
        Launch1D(K("one_hot_f32"), n, P(indices), P(y), U(classes), U(classes - 1), U(n));
    }

    public override void ScatterAdd(Storage dy, Storage indices, Storage dtable, int count, int dim, int vocabulary)
    {
        int n = count * dim;
        Launch1D(K("scatter_add_f32"), n, P(dy), P(indices), P(dtable), U(dim), U(vocabulary - 1), U(n));
    }

    private void LaunchPatches(string kernel, Storage a, Storage b, in ConvGeometry g)
    {
        int n = g.Positions * g.PatchSize;
        Launch1D(K(kernel), n, P(a), P(b),
            U(g.C), U(g.H), U(g.W), U(g.KW), U(g.SH), U(g.SW), U(g.PH), U(g.PW),
            U(g.OW), U(g.PatchSize), U(g.KH * g.KW), U(g.OH * g.OW), U(n));
    }

    public override void Im2Col(Storage x, Storage cols, in ConvGeometry g) => LaunchPatches("im2col_f32", x, cols, g);

    public override void Col2Im(Storage dcols, Storage dx, in ConvGeometry g) => LaunchPatches("col2im_f32", dcols, dx, g);

    public override void MaxPool(Storage x, Storage y, Storage argmax, in ConvGeometry g)
    {
        int n = g.N * g.C * g.OH * g.OW;
        Launch1D(K("maxpool_f32"), n, P(x), P(y), P(argmax),
            U(g.H), U(g.W), U(g.KH), U(g.KW), U(g.SH), U(g.SW), U(g.PH), U(g.PW), U(g.OW), U(g.OH * g.OW), U(n));
    }

    public override void MaxPoolBackward(Storage dy, Storage argmax, Storage dx, int count) =>
        Launch1D(K("maxpool_bwd_f32"), count, P(dy), P(argmax), P(dx), U(count));

    public override void Permute(Storage x, Storage y, ReadOnlySpan<int> outShape, ReadOnlySpan<int> inStrides, bool accumulate)
    {
        const int R = PtxKernels.MaxPermuteRank;
        if (outShape.Length > R)
        {
            throw new NotSupportedException($"Permute supports at most {R} dimensions on CUDA.");
        }

        // Right-align the dimensions; unused leading slots have size 1 and stride 0.
        Span<ulong> args = stackalloc ulong[2 + 2 * R + 2];
        int pad = R - outShape.Length, n = 1;
        args[0] = P(x);
        args[1] = P(y);
        for (int d = 0; d < R; d++)
        {
            int size = d < pad ? 1 : outShape[d - pad];
            args[2 + d] = (uint)size;
            args[2 + R + d] = d < pad ? 0UL : (uint)inStrides[d - pad];
            n *= size;
        }

        args[2 + 2 * R] = accumulate ? 1UL : 0UL;
        args[3 + 2 * R] = (uint)n;
        Launch1D(K("permute_f32"), n, args);
    }

    public override void Copy2D(Storage src, int srcOffset, int srcStride, Storage dst, int dstOffset, int dstStride, int rows, int cols, bool accumulate)
    {
        int n = rows * cols;
        Launch1D(K("copy2d_f32"), n, P(src), P(dst), U(srcOffset), U(srcStride), U(dstOffset), U(dstStride), U(cols), U(accumulate ? 1 : 0), U(n));
    }

    public override void SumAxis(Storage x, Storage y, int outer, int dim, int inner, float scale, bool accumulate)
    {
        int n = outer * inner;
        Launch1D(K("sum_axis_f32"), n, P(x), P(y), U(dim), U(inner), F(scale), U(accumulate ? 1 : 0), U(n));
    }

    public override void BroadcastAxis(Storage dy, Storage dx, int outer, int dim, int inner, float scale)
    {
        int n = outer * dim * inner;
        Launch1D(K("broadcast_axis_f32"), n, P(dy), P(dx), U(dim * inner), U(inner), F(scale), U(n));
    }
}
