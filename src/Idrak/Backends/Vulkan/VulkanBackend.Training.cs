// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Backends.Vulkan;

// Training operations as generated kernels (VulkanKernels.Training.cs): the layer-norm gradient, grouped normalization
// and its reductions, column sums, the product with a bias, AdamW over many tensors, 8-bit Adam and the gradient of
// tiled causal attention. Reductions run in two passes, partial sums per split and then the splits in order, so a shape
// gives the same bits run after run; the split counts come from the width, or are measured per shape (VulkanTuneOp.Splits).
internal sealed partial class VulkanBackend
{
    // Reductions: fewest rows of a column per split, and fewest elements per invocation of a workgroup's split of a group.
    private const int SplitMinRows = 16, SplitMinPerLane = 4;

    // Variants of the reduction splits measured (the key's Variant): columns or groups, by first-pass mode.
    private const int SplitColumns = 0, SplitGroups = 4;

    // ------------------------------------------------------------------ layer norm

    public override void LayerNormBackward(Storage x, Storage gamma, Storage dy, Storage stats, Storage? dx, Storage? dgamma, Storage? dbeta, int rows, int cols)
    {
        if (!Fit(x, gamma, dy, stats, dx ?? x, dgamma ?? x, dbeta ?? x) || !ReductionFits(rows, cols))
        {
            base.LayerNormBackward(x, gamma, dy, stats, dx, dgamma, dbeta, rows, cols);
            return;
        }

        if (rows <= 0 || cols <= 0)
        {
            return;
        }

        if (dx is not null)
        {
            Span<byte> b = stackalloc byte[8];
            Rows("layer_norm_backward", rows, cols, [x, gamma, dy, stats, dx], new Push(b).I(rows).I(cols).Bytes);
        }

        if (dgamma is not null || dbeta is not null)
        {
            int flags = (dgamma is null ? 0 : 1) | (dbeta is null ? 0 : 2);
            ColumnSums(2, x, dy, stats, rows, cols, cols, 0, dgamma ?? dbeta!, dbeta ?? dgamma!, flags);
        }
    }

    // ------------------------------------------------------------------ grouped normalization

    public override void NormStats(Storage x, Storage mean, Storage variance, Storage invStd, int outer, int groups, int inner, float eps)
    {
        if (!Fit(x, mean, variance, invStd) || !ReductionFits(outer, groups * (long)inner))
        {
            base.NormStats(x, mean, variance, invStd, outer, groups, inner, eps);
            return;
        }

        GroupSums(0, x, x, outer, groups, inner, mean, variance, invStd, 0, eps);
    }

    public override void GroupReduce(Storage a, Storage? b, Storage sumA, Storage? sumAB, int outer, int groups, int inner)
    {
        if (!Fit(a, b ?? a, sumA, sumAB ?? sumA) || !ReductionFits(outer, groups * (long)inner))
        {
            base.GroupReduce(a, b, sumA, sumAB, outer, groups, inner);
            return;
        }

        GroupSums(1, a, b ?? a, outer, groups, inner, sumA, sumAB ?? sumA, sumA, sumAB is null ? 1 : 3, 0f);
    }

    public override void NormApply(Storage x, Storage mean, Storage invStd, Storage y, int outer, int groups, int inner)
    {
        long n = (long)outer * groups * inner;
        if (!Fit(x, mean, invStd, y) || n > int.MaxValue)
        {
            base.NormApply(x, mean, invStd, y, outer, groups, inner);
            return;
        }

        Span<byte> b = stackalloc byte[12];
        Grid("norm_apply", n, [x, mean, invStd, y], new Push(b).I((int)n).I(groups).I(inner).Bytes);
    }

    public override void NormBackward(Storage dxhat, Storage xhat, Storage sum1, Storage sum2, Storage invStd, Storage dx, int outer, int groups, int inner)
    {
        long n = (long)outer * groups * inner;
        if (!Fit(dxhat, xhat, sum1, sum2, invStd, dx) || n > int.MaxValue)
        {
            base.NormBackward(dxhat, xhat, sum1, sum2, invStd, dx, outer, groups, inner);
            return;
        }

        Span<byte> b = stackalloc byte[16];
        Grid("norm_backward", n, [dxhat, xhat, sum1, sum2, invStd, dx], new Push(b).I((int)n).I(groups).I(inner).F(outer * inner).Bytes);
    }

    public override void GroupScaleShift(Storage x, Storage? scale, Storage? shift, Storage y, int n, int groups, int inner, bool accumulate)
    {
        if (!Fit(x, scale ?? x, shift ?? x, y))
        {
            base.GroupScaleShift(x, scale, shift, y, n, groups, inner, accumulate);
            return;
        }

        int flags = (scale is null ? 0 : 1) | (shift is null ? 0 : 2) | (accumulate ? 4 : 0);
        Span<byte> b = stackalloc byte[16];
        Grid("group_scale_shift", n, [x, scale ?? x, shift ?? x, y], new Push(b).I(n).I(Math.Max(groups, 1)).I(Math.Max(inner, 1)).I(flags).Bytes);
    }

    // ------------------------------------------------------------------ column sums and products with a bias

    public override void SumColumns(Storage x, long offset, int ld, Storage y, int rows, int cols)
    {
        if (!Fit(x, y) || offset > int.MaxValue || !ReductionFits(rows, cols))
        {
            base.SumColumns(x, offset, ld, y, rows, cols);
            return;
        }

        ColumnSums(1, x, x, x, rows, cols, ld, (int)offset, y, y, 1);
    }

    // c = a·b + bias: the bias rows written first, then the product (the measured float32 kernel) added to them (beta 1).
    public override bool MatMulBias(Storage a, Storage b, Storage bias, Storage c, int m, int n, int k)
    {
        if (!Fit(a, b, bias, c) || (long)m * n > int.MaxValue)
        {
            return false;
        }

        if (m <= 0 || n <= 0)
        {
            return true;
        }

        Span<byte> p = stackalloc byte[8];
        Grid("broadcast_rows", (long)m * n, [bias, c], new Push(p).I(m * n).I(n).Bytes);
        if (k > 0)
        {
            BatchedMatMul(a, b, c, 1, m, n, k, false, false, 1f);
        }

        return true;
    }

    // ------------------------------------------------------------------ the two-pass reductions

    // Whether a reduction over `rows` rows of `cols` columns has partials that fit one storage at one split.
    private bool ReductionFits(long rows, long cols) =>
        rows * cols <= int.MaxValue && cols * VulkanKernels.PartialStride <= int.MaxValue && BlockBytes((int)Math.Min(int.MaxValue, cols * VulkanKernels.PartialStride)) <= MaxStorageBytes;

    // Most splits of a reduction with `units` partials per split (columns or groups) over `length` elements each, with at
    // least `minimum` elements per split, and the partials within one storage.
    private int MostSplits(int units, long length, long minimum)
    {
        long most = Math.Clamp(length / Math.Max(1, minimum), 1, Math.Min(Limits.MaxGroupsX, 1 << 16));
        while (most > 1 && ((long)units * most * VulkanKernels.PartialStride > int.MaxValue
               || BlockBytes((int)Math.Min(int.MaxValue, (long)units * most * VulkanKernels.PartialStride)) > MaxStorageBytes))
        {
            most /= 2;
        }

        return PowersOfTwo((int)most)[^1];
    }

    // The largest power of two at most `value` within [1, most].
    private static int PowerOfTwoWithin(long value, int most) => PowersOfTwo((int)Math.Clamp(value, 1, most))[^1];

    // Column sums of a [rows, cols] block (element (r, j) at offset + r·ld + j) with `mode` (see group_partials_columns),
    // the first sums added to out0 (flags bit 0) and the second to out1 (bit 1). Splits: as many as give the dispatch a
    // width of workgroups (width² invocations), with at least SplitMinRows rows each; measured per shape.
    private void ColumnSums(int mode, Storage a, Storage b, Storage stats, int rows, int cols, int ld, int offset, Storage out0, Storage out1, int flags)
    {
        if (rows <= 0 || cols <= 0)
        {
            return;
        }

        int most = MostSplits(cols, rows, SplitMinRows);
        int formula = PowerOfTwoWithin((long)Width * Width / cols, most);
        int splits = formula;
        if (most > 1)
        {
            var key = new VulkanTuneKey(VulkanTuneOp.Splits, SplitColumns + mode, rows, cols, ld);
            if (!TryTuned(key, out splits) || splits < 1 || splits > most || (splits & (splits - 1)) != 0)
            {
                splits = Autotune && !t_timing ? TuneColumnSums(key, most, formula, mode, a, b, stats, rows, cols, ld, offset, out0, out1, flags) : formula;
            }
        }

        RunColumnSums(splits, mode, a, b, stats, rows, cols, ld, offset, out0, out1, flags);
    }

    private int TuneColumnSums(VulkanTuneKey key, int most, int formula, int mode, Storage a, Storage b, Storage stats, int rows, int cols, int ld, int offset,
        Storage out0, Storage out1, int flags) =>
        TuneWithScratch(key, PowersOfTwo(most), formula, [a, b, stats, out0, out1], 0b11000,
            (s, bound) => RunColumnSums(s, mode, bound[0], bound[1], bound[2], rows, cols, ld, offset, bound[3], bound[4], flags));

    private void RunColumnSums(int splits, int mode, Storage a, Storage b, Storage stats, int rows, int cols, int ld, int offset, Storage out0, Storage out1, int flags)
    {
        var part = Allocate(splits * cols * VulkanKernels.PartialStride, zeroed: false);
        try
        {
            Span<byte> p = stackalloc byte[24];
            Grid("group_partials_columns", (long)splits * cols, [a, b, stats, part], new Push(p).I(rows).I(cols).I(ld).I(offset).I(splits).I(mode).Bytes);
            Span<byte> f = stackalloc byte[32];
            Grid("group_finish", cols, [part, out0, out1, out1], new Push(f).I(cols).I(splits).I(cols).I(1).I(1).I(1).I(flags).F(0f).Bytes);
        }
        finally
        {
            part.Release();                                                    // reused in queue order
        }
    }

    // Per-group reductions of the [outer, groups, inner] view: the moments into mean, variance and 1 / std (mode 0), or
    // Σ a and Σ a·b added to out0 and out1 by flags (mode 1). Runs of at least a width (inner ≥ width) go through a
    // workgroup per group and split; shorter runs through the column sums of [outer, groups·inner], each group then
    // adding its inner columns. Splits: as many as make a width of workgroups (or width² invocations), measured per shape.
    private void GroupSums(int mode, Storage a, Storage b, int outer, int groups, int inner, Storage out0, Storage out1, Storage out2, int flags, float eps)
    {
        if (groups <= 0)
        {
            return;
        }

        bool columns = inner < Width;
        long length = (long)outer * inner;
        int units = columns ? groups * inner : groups;
        int most = columns ? MostSplits(units, outer, SplitMinRows) : MostSplits(units, length, (long)Width * SplitMinPerLane);
        int formula = columns ? PowerOfTwoWithin((long)Width * Width / Math.Max(1, units), most) : PowerOfTwoWithin((Width + groups - 1) / groups, most);
        int splits = formula;
        if (most > 1 && units > 0)
        {
            var key = new VulkanTuneKey(VulkanTuneOp.Splits, (columns ? SplitColumns : SplitGroups) + mode, outer, groups, inner);
            if (!TryTuned(key, out splits) || splits < 1 || splits > most || (splits & (splits - 1)) != 0)
            {
                splits = Autotune && !t_timing ? TuneGroupSums(key, most, formula, columns, mode, a, b, outer, groups, inner, out0, out1, out2, flags, eps) : formula;
            }
        }

        RunGroupSums(splits, columns, mode, a, b, outer, groups, inner, out0, out1, out2, flags, eps);
    }

    private int TuneGroupSums(VulkanTuneKey key, int most, int formula, bool columns, int mode, Storage a, Storage b, int outer, int groups, int inner,
        Storage out0, Storage out1, Storage out2, int flags, float eps)
    {
        return TuneWithScratch(key, PowersOfTwo(most), formula, [a, b, out0, out1, out2], 0b11100,
            (s, bound) => RunGroupSums(s, columns, mode, bound[0], bound[1], outer, groups, inner, bound[2], bound[3], bound[4], flags, eps));
    }

    private void RunGroupSums(int splits, bool columns, int mode, Storage a, Storage b, int outer, int groups, int inner,
        Storage out0, Storage out1, Storage out2, int flags, float eps)
    {
        int units = columns ? groups * inner : groups;
        var part = Allocate(Math.Max(1, splits * units * VulkanKernels.PartialStride), zeroed: false);
        try
        {
            if (columns)
            {
                int cols = groups * inner;
                Span<byte> p = stackalloc byte[24];
                Grid("group_partials_columns", (long)splits * cols, [a, b, a, part], new Push(p).I(outer).I(cols).I(cols).I(0).I(splits).I(mode).Bytes);
            }
            else
            {
                Span<byte> p = stackalloc byte[20];
                Run("group_partials_rows", RowGroups((long)groups * splits), 1, 1, [a, b, part], new Push(p).I(outer).I(groups).I(inner).I(splits).I(mode).Bytes);
            }

            // Partial p = s·strideS + g·strideG + i: columns (s·cols + g·inner + i, i < inner), groups (g·splits + s).
            Span<byte> f = stackalloc byte[32];
            var push = columns ? new Push(f).I(groups).I(splits).I(groups * inner).I(inner).I(inner) : new Push(f).I(groups).I(splits).I(1).I(splits).I(1);
            Grid("group_finish", groups, [part, out0, out1, out2], push.I(mode == 0 ? 0 : 1).I(flags).F(eps).Bytes);
        }
        finally
        {
            part.Release();                                                    // reused in queue order
        }
    }

    // ------------------------------------------------------------------ optimizers

    public override void AdamStep8Bit(Storage p, Storage g, Storage m, Storage v, Storage absMax, Storage map, int n, float lr, float beta1, float beta2, float eps,
        float gradientScale, float decay)
    {
        if (!Fit(p, g, m, v, absMax, map))
        {
            base.AdamStep8Bit(p, g, m, v, absMax, map, n, lr, beta1, beta2, eps, gradientScale, decay);
            return;
        }

        int blocks = (n + Optimizers.AdamW8Bit.BlockSize - 1) / Optimizers.AdamW8Bit.BlockSize;
        if (blocks <= 0)
        {
            return;
        }

        Span<byte> b = stackalloc byte[32];
        Run("adam_8bit", RowGroups(blocks), 1, 1, [p, g, m, v, absMax, map],
            new Push(b).I(n).F(lr).F(beta1).F(beta2).F(eps).F(gradientScale).F(decay).I(blocks).Bytes);
    }

    // What FusedAdamW keeps between steps: the sizes it was laid out for, each tensor's first partial and partial count
    // of the norm, the partials and the clipping factor.
    private sealed class FusedAdamState(VulkanBackend backend) : IDisposable
    {
        public int[] Sizes = [];
        public int[] Offsets = [];
        public int[] Groups = [];
        public int Total;
        public Storage? Partials, Factor;
        public bool Disposed;

        public VulkanBackend Backend => backend;

        public void Dispose()
        {
            Partials?.Release();
            Factor?.Release();
            (Partials, Factor, Disposed) = (null, null, true);
        }
    }

    // The global norm in two passes: each tensor's sum of squares over up to a width of workgroups (each taking at least a
    // width of elements per invocation) into its partials, then one workgroup adds every partial in order and writes the
    // clipping factor; then one AdamW pass per tensor reads the factor (and zeroes the gradients when asked).
    public override bool FusedAdamW(ReadOnlySpan<(Storage P, Storage G, Storage M, Storage V, int N)> tensors, ref IDisposable? cache, float maxNorm,
        float lr, float decay, float beta1, float beta2, float eps, bool zeroGradients)
    {
        foreach (var (p, g, m, v, _) in tensors)
        {
            if (!Fit(p, g, m, v))
            {
                return base.FusedAdamW(tensors, ref cache, maxNorm, lr, decay, beta1, beta2, eps, zeroGradients);
            }
        }

        if (tensors.Length == 0)
        {
            return true;
        }

        if (cache is not FusedAdamState state || state.Disposed || state.Backend != this)
        {
            cache?.Dispose();
            cache = state = new FusedAdamState(this);
        }

        bool resized = state.Sizes.Length != tensors.Length;
        for (int t = 0; t < tensors.Length && !resized; t++)
        {
            resized = state.Sizes[t] != tensors[t].N;
        }

        if (resized)
        {
            state.Sizes = new int[tensors.Length];
            state.Offsets = new int[tensors.Length];
            state.Groups = new int[tensors.Length];
            long total = 0;
            long perGroup = (long)Width * Width;
            for (int t = 0; t < tensors.Length; t++)
            {
                int n = tensors[t].N;
                state.Sizes[t] = n;
                state.Offsets[t] = (int)total;
                state.Groups[t] = n <= 0 ? 0 : (int)Math.Clamp((n + perGroup - 1) / perGroup, 1, Math.Min(Width, Limits.MaxGroupsX));
                total += state.Groups[t];
            }

            state.Total = (int)total;
            state.Partials?.Release();
            state.Partials = Allocate(Math.Max(1, state.Total), zeroed: true);
        }

        state.Factor ??= Allocate(1, zeroed: false);
        Span<byte> b = stackalloc byte[28];
        if (maxNorm > 0f)
        {
            for (int t = 0; t < tensors.Length; t++)
            {
                if (state.Groups[t] > 0)
                {
                    Run("sum_squares_partials", (uint)state.Groups[t], 1, 1, [tensors[t].G, state.Partials!], new Push(b).I(tensors[t].N).I(state.Offsets[t]).Bytes);
                }
            }

            Run("clip_factor_partials", 1, 1, 1, [state.Partials!, state.Factor], new Push(b).I(state.Total).F(maxNorm).Bytes);
        }
        else
        {
            Fill(state.Factor, 1, 1f);
        }

        foreach (var (p, g, m, v, n) in tensors)
        {
            Grid("fused_adamw", n, [p, g, m, v, state.Factor], new Push(b).I(n).F(lr).F(decay).F(beta1).F(beta2).F(eps).B(zeroGradients).Bytes);
        }

        return true;
    }

    // ------------------------------------------------------------------ attention

    // Two passes (VulkanKernels.Training.cs): a workgroup per query row adds its dq and writes delta = dOutput · output
    // to a temporary; then a workgroup per key position adds its dk and dv. Every sum runs in a fixed order.
    public override void AttentionTiledBackward(Storage q, Storage keys, Storage values, Storage output, Storage logSumExp, Storage dOutput,
        Storage dq, Storage dkeys, Storage dvalues, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        long rows = (long)heads * rowsPerHead;
        if (dim > VulkanKernels.AttentionMaxDim || rows > int.MaxValue || !Fit(q, keys, values, output, logSumExp, dOutput, dq, dkeys, dvalues))
        {
            base.AttentionTiledBackward(q, keys, values, output, logSumExp, dOutput, dq, dkeys, dvalues, heads, rowsPerHead, steps, capacity, dim, scale, variant);
            return;
        }

        if (rows <= 0 || dim <= 0 || steps <= 0 || capacity <= 0)
        {
            return;
        }

        var delta = Allocate((int)rows, zeroed: false);
        try
        {
            Span<byte> b = stackalloc byte[32];
            var push = new Push(b).I(heads).I(rowsPerHead).I(steps).I(capacity).I(dim).F(scale).I(variant.Window).F(variant.Softcap).Bytes;
            Run("attention_backward_dq",
 RowGroups(rows), 1, 1, [q, keys, values, output, logSumExp, dOutput, dq, delta], push);
            Run("attention_backward_dkv", RowGroups((long)heads * Math.Min(capacity, steps)), 1, 1, [q, keys, values, logSumExp, dOutput, delta, dkeys, dvalues], push);
        }
        finally
        {
            delta.Release();                                                   // reused in queue order
        }
    }
}
