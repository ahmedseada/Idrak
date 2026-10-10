// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// Convolutions, average pooling and image resampling as generated kernels (VulkanKernels.Convolution.cs).
//
// A convolution or one of its gradients runs one of: the composed path (patches unfolded by im2col and the measured float32
// product, or the cooperative-matrix one under MixedPrecision: Backend.Convolution.cs), an implicit product that reads the
// operands where they lie (register-blocked or one output per invocation, at a candidate width; the weight gradient also
// at 1, 4, 16 or 64 splits of its sum), or the depthwise kernels (one input channel per group). Which is fastest is
// measured per shape the first time it runs (VulkanTuneOp.Convolution, keyed by the shape, the pass, the precision, the
// activation and the groups) and kept in the tuning cache; while nothing can be measured (IDRAK_AUTOTUNE=0, a graph being
// recorded) the formula's choice runs: the depthwise kernels where they apply, else the implicit product, blocked when
// its blocks would be at least half full, with the splits of ConvolutionShapes.FormulaSplits. Neither the implicit
// products nor the depthwise kernels keep anything beyond their output (and the weight gradient's partial sums).
internal sealed partial class VulkanBackend
{
    private const int ConvComposed = 0, ConvTile = 1, ConvBlocked = 2, ConvDepthwise = 3;

    // Tests and benchmarks: forces one path (ConvComposed, ConvTile, ConvBlocked or ConvDepthwise; at the device's width,
    // the formula's splits) where it can run.
    internal static int? ConvolutionPath { get; set; }

    public override void ConvolutionKernel(Storage x, Storage weight, Storage? bias, Storage y, in ConvGeometry g, int filters, int groups, ConvActivation activation)
    {
        if (!ConvFits(in g, filters, groups) || !Fit(x, weight, bias ?? weight, y))
        {
            base.ConvolutionKernel(x, weight, bias, y, in g, filters, groups, activation);
            return;
        }

        int flags = (bias is null ? 0 : 1) | (int)activation << 1;
        ReadOnlySpan<Storage> storages = [x, weight, bias ?? weight, y];
        Span<int> buffer = stackalloc int[ConvCandidatesMost];
        var candidates = ConvCandidates(ConvolutionShapes.Forward, in g, filters, groups, buffer);
        int choice = ConvChoice(ConvolutionShapes.Forward, in g, filters, groups, activation, candidates, out var key, out bool measure);
        if (measure)
        {
            choice = MeasureConvolution(key, candidates.ToArray(), choice, storages, g, filters, groups, flags, activation);
        }

        RunConvolution(choice, storages, g, filters, groups, flags, activation);
    }

    public override void ConvolutionBackwardInputKernel(Storage dy, Storage weight, Storage dx, in ConvGeometry g, int filters, int groups)
    {
        if (!ConvFits(in g, filters, groups) || !Fit(dy, weight, dx))
        {
            base.ConvolutionBackwardInputKernel(dy, weight, dx, in g, filters, groups);
            return;
        }

        ReadOnlySpan<Storage> storages = [dy, weight, dx];
        Span<int> buffer = stackalloc int[ConvCandidatesMost];
        var candidates = ConvCandidates(ConvolutionShapes.Input, in g, filters, groups, buffer);
        int choice = ConvChoice(ConvolutionShapes.Input, in g, filters, groups, ConvActivation.None, candidates, out var key, out bool measure);
        if (measure)
        {
            choice = MeasureConvolutionInput(key, candidates.ToArray(), choice, storages, g, filters, groups);
        }

        RunConvolutionInput(choice, storages, g, filters, groups);
    }

    public override void ConvolutionBackwardWeightKernel(Storage x, Storage dy, Storage dweight, in ConvGeometry g, int filters, int groups)
    {
        if (!ConvFits(in g, filters, groups) || !Fit(x, dy, dweight))
        {
            base.ConvolutionBackwardWeightKernel(x, dy, dweight, in g, filters, groups);
            return;
        }

        ReadOnlySpan<Storage> storages = [x, dy, dweight];
        Span<int> buffer = stackalloc int[ConvCandidatesMost];
        var candidates = ConvCandidates(ConvolutionShapes.Weight, in g, filters, groups, buffer);
        int choice = ConvChoice(ConvolutionShapes.Weight, in g, filters, groups, ConvActivation.None, candidates, out var key, out bool measure);
        if (measure)
        {
            choice = MeasureConvolutionWeight(key, candidates.ToArray(), choice, storages, g, filters, groups);
        }

        RunConvolutionWeight(choice, storages, g, filters, groups);
    }

    // Measuring each pass, kept apart from the operations so their usual calls make no closure.
    private int MeasureConvolution(VulkanTuneKey key, int[] candidates, int fallback, ReadOnlySpan<Storage> storages, ConvGeometry g, int filters, int groups,
        int flags, ConvActivation activation) =>
        ConvMeasure(key, candidates, fallback, storages.ToArray(), 1UL << 3, (c, s) => RunConvolution(c, s, g, filters, groups, flags, activation));

    private int MeasureConvolutionInput(VulkanTuneKey key, int[] candidates, int fallback, ReadOnlySpan<Storage> storages, ConvGeometry g, int filters,
        int groups) =>
        ConvMeasure(key, candidates, fallback, storages.ToArray(), 1UL << 2, (c, s) => RunConvolutionInput(c, s, g, filters, groups));

    private int MeasureConvolutionWeight(VulkanTuneKey key, int[] candidates, int fallback, ReadOnlySpan<Storage> storages, ConvGeometry g, int filters,
        int groups) =>
        ConvMeasure(key, candidates, fallback, storages.ToArray(), 1UL << 2, (c, s) => RunConvolutionWeight(c, s, g, filters, groups));

    // Whether the kernels index every operand within an int (the storages' lengths are ints; the sums of positions too).
    private static bool ConvFits(in ConvGeometry g, int filters, int groups) =>
        groups > 0 && filters > 0 && g.C % groups == 0 && filters % groups == 0 && g.OH > 0 && g.OW > 0
        && (long)g.N * g.C * g.H * g.W <= int.MaxValue && (long)g.N * filters * g.OH * g.OW <= int.MaxValue && (long)g.N * g.OH * g.OW <= int.MaxValue / 2
        && (long)filters * (g.PatchSize / groups) * 64 <= int.MaxValue;

    // The rows, columns and batch entries of a pass's product.
    private static (int Rows, int Columns, int Batch) ConvProduct(int pass, in ConvGeometry g, int filters, int groups) => pass switch
    {
        ConvolutionShapes.Forward => (filters / groups, g.OH * g.OW, g.N * groups),
        ConvolutionShapes.Input => (g.C / groups, g.H * g.W, g.N * groups),
        _ => (filters / groups, g.PatchSize / groups, groups),
    };

    // The path of one pass for this shape among `candidates` (ConvCandidates): stored, forced, or the formula's (see the
    // file's comment). `measure` is set when the shape should be measured now (ConvMeasure, under `key`; the formula's
    // choice is returned then).
    private int ConvChoice(int pass, in ConvGeometry g, int filters, int groups, ConvActivation activation, ReadOnlySpan<int> candidates, out VulkanTuneKey key,
        out bool measure)
    {
        key = default;
        measure = false;
        int fallback = ConvFormula(pass, in g, filters, groups, candidates);
        if (ConvolutionPath is int forced)
        {
            foreach (int c in candidates)
            {
                if ((c & 15) == forced && (WidthOf(c) == Width || forced == ConvComposed))
                {
                    return c;
                }
            }

            return fallback;
        }

        if (!ConvolutionShapes.Key(in g, filters, groups, out var shape))
        {
            return fallback;
        }

        key = new VulkanTuneKey(VulkanTuneOp.Convolution, ConvolutionShapes.Variant(pass, activation, groups), shape.A, shape.B, shape.C, shape.D, shape.E, shape.F);
        if (TryTuned(key, out int known) && candidates.IndexOf(known) >= 0)
        {
            return known;
        }

        measure = CanTune && candidates.Length > 1;
        return fallback;
    }

    // Measures a pass's candidates on scratch copies of what it writes (bit i of `writes`: storage i); the formula's
    // choice where memory runs short.
    private int ConvMeasure(VulkanTuneKey key, int[] candidates, int fallback, Storage[] storages, ulong writes, Action<int, Storage[]> run)
    {
        int chosen = fallback;
        var lengths = new int[storages.Length];
        for (int i = 0; i < storages.Length; i++)
        {
            lengths[i] = (writes & (1UL << i)) != 0 ? storages[i].Length : 0;
        }

        try
        {
            WithScratch(lengths, scratch =>
            {
                var bound = new Storage[storages.Length];
                for (int i = 0; i < storages.Length; i++)
                {
                    bound[i] = (writes & (1UL << i)) != 0 ? scratch[i] : storages[i];
                }

                // The composed path once before the timing, so the products it runs measure their own choices first.
                if (Array.IndexOf(candidates, WithWidth(Width, ConvComposed)) >= 0)
                {
                    run(WithWidth(Width, ConvComposed), bound);
                }

                chosen = Tune(key, candidates, fallback, c => run(c, bound));
            });
        }
        catch (ResourceLimitExceededException)
        {
            return fallback;
        }

        return chosen;
    }

    // Candidates of a pass at most: the composed path, the depthwise kernels, and the implicit product tiled and blocked
    // at each split count (ConvolutionShapes.SplitChoices) and each candidate width (at most three: CandidateWidths).
    private const int ConvCandidatesMost = 2 + 2 * 4 * 3;

    // The candidates of a pass, written to `into` (ConvCandidatesMost long): the composed path where its patches fit the
    // device (a binding, and half the memory it reports free), the implicit product tiled and blocked at each candidate
    // width where the workgroup counts take it (the weight gradient at each split count), the depthwise kernels where
    // every group reads one channel.
    private ReadOnlySpan<int> ConvCandidates(int pass, in ConvGeometry g, int filters, int groups, Span<int> into)
    {
        int count = 0;
        long patches = (long)g.Positions * g.PatchSize;
        long free = AvailableMemory() ?? long.MaxValue;
        if (patches <= int.MaxValue && BlockBytes((int)Math.Min(patches, int.MaxValue)) <= MaxStorageBytes && 4 * patches <= free / 2)
        {
            into[count++] = WithWidth(Width, ConvComposed);
        }

        var (rows, columns, _) = ConvProduct(pass, in g, filters, groups);
        long positions = (long)g.N * g.OH * g.OW;
        ReadOnlySpan<int> splits = pass == ConvolutionShapes.Weight ? ConvolutionShapes.SplitChoices : [1];
        ReadOnlySpan<int> variants = [ConvTile, ConvBlocked];
        foreach (int width in CandidateWidths)
        {
            foreach (int variant in variants)
            {
                int edge = VulkanKernels.MatSide(width) * (variant == ConvBlocked ? VulkanKernels.MatPer : 1);
                if (!MatFits(edge, rows, columns))
                {
                    continue;
                }

                foreach (int s in splits)
                {
                    if (!ConvolutionShapes.SplitTried(s, positions))
                    {
                        continue;
                    }

                    if (s == 1 || (long)s * filters * (g.PatchSize / groups) <= int.MaxValue && BlockBytes(s * filters * (g.PatchSize / groups)) <= MaxStorageBytes)
                    {
                        into[count++] = WithWidth(width, variant | System.Numerics.BitOperations.Log2((uint)s) << 4);
                    }
                }
            }
        }

        if (ConvolutionShapes.Depthwise(in g, groups))
        {
            into[count++] = WithWidth(Width, ConvDepthwise);
        }

        return into[..count];
    }

    // The formula's choice among the candidates (see the file's comment).
    private int ConvFormula(int pass, in ConvGeometry g, int filters, int groups, ReadOnlySpan<int> candidates)
    {
        if (ConvFind(candidates, ConvDepthwise) is int depthwise and >= 0)
        {
            return depthwise;
        }

        var (rows, columns, _) = ConvProduct(pass, in g, filters, groups);
        int edge = VulkanKernels.MatPer * VulkanKernels.MatSide(Width);
        int splitBits = pass == ConvolutionShapes.Weight ? System.Numerics.BitOperations.Log2((uint)ConvolutionShapes.FormulaSplits((long)g.N * g.OH * g.OW)) << 4 : 0;
        int variant = rows >= edge / 2 && columns >= edge / 2 ? ConvBlocked : ConvTile;
        ReadOnlySpan<int> order = [variant | splitBits, variant, ConvTile | splitBits, ConvTile, ConvComposed];
        foreach (int v in order)
        {
            if (ConvFind(candidates, v) is int found and >= 0)
            {
                return found;
            }
        }

        return candidates.Length > 0 ? candidates[0] : WithWidth(Width, ConvComposed);
    }

    // `variant` at the device's width when it is among the candidates, else -1.
    private int ConvFind(ReadOnlySpan<int> candidates, int variant) =>
        candidates.IndexOf(WithWidth(Width, variant)) >= 0 ? WithWidth(Width, variant) : -1;

    // The 72 bytes of push constants of every convolution kernel.
    private static ReadOnlySpan<byte> ConvPush(Span<byte> bytes, in ConvGeometry g, int filters, int groups, int flags, int splits) =>
        new Push(bytes).I(g.N).I(g.C).I(g.H).I(g.W).I(g.KH).I(g.KW).I(g.SH).I(g.SW).I(g.PH).I(g.PW).I(g.OH).I(g.OW).I(g.DH).I(g.DW)
            .I(filters).I(groups).I(flags).I(splits).Bytes;

    // Dispatches an implicit product (`kernel` blocked, `tile` its tiled variant): blocks of the tile along x (columns)
    // and y (rows), the batch entries along z (the kernels loop over those past the device's count).
    private void RunConvProduct(string kernel, string tile, int choice, int rows, int columns, int batch, ReadOnlySpan<Storage> storages, ReadOnlySpan<byte> push)
    {
        int width = WidthOf(choice), variant = choice & 15;
        int edge = VulkanKernels.MatSide(width) * (variant == ConvBlocked ? VulkanKernels.MatPer : 1);
        RunAt(variant == ConvBlocked ? kernel : tile, width, (uint)((columns + edge - 1) / edge), (uint)((rows + edge - 1) / edge),
            (uint)Math.Clamp(batch, 1, Limits.MaxGroupsZ), storages, push);
    }

    // storages: x, weight, bias (the weight when none), y.
    private void RunConvolution(int choice, ReadOnlySpan<Storage> s, ConvGeometry g, int filters, int groups, int flags, ConvActivation activation)
    {
        switch (choice & 15)
        {
            case ConvComposed:
                base.ConvolutionKernel(s[0], s[1], (flags & 1) != 0 ? s[2] : null, s[3], in g, filters, groups, activation);
                return;
            case ConvDepthwise:
            {
                Span<byte> b = stackalloc byte[72];
                Grid("conv_depthwise", (long)g.N * filters * g.OH * g.OW, s, ConvPush(b, g, filters, groups, flags, 1));
                return;
            }

            default:
            {
                Span<byte> b = stackalloc byte[72];
                var (rows, columns, batch) = ConvProduct(ConvolutionShapes.Forward, in g, filters, groups);
                RunConvProduct("conv_forward", "conv_forward_tile", choice, rows, columns, batch, s, ConvPush(b, g, filters, groups, flags, 1));
                return;
            }
        }
    }

    // storages: dy, weight, dx.
    private void RunConvolutionInput(int choice, ReadOnlySpan<Storage> s, ConvGeometry g, int filters, int groups)
    {
        Span<byte> b = stackalloc byte[72];
        switch (choice & 15)
        {
            case ConvComposed:
                base.ConvolutionBackwardInputKernel(s[0], s[1], s[2], in g, filters, groups);
                return;
            case ConvDepthwise:
                Grid("conv_depthwise_backward_input", (long)g.N * g.C * g.H * g.W, s, ConvPush(b, g, filters, groups, 0, 1));
                return;
            default:
                var (rows, columns, batch) = ConvProduct(ConvolutionShapes.Input, in g, filters, groups);
                RunConvProduct("conv_backward_input", "conv_backward_input_tile", choice, rows, columns, batch, s, ConvPush(b, g, filters, groups, 0, 1));
                return;
        }
    }

    // storages: x, dy, dweight. Several splits write their partial sums to a temporary [splits, filters, patch] that
    // conv_split_reduce adds to dweight in split order.
    private void RunConvolutionWeight(int choice, ReadOnlySpan<Storage> s, ConvGeometry g, int filters, int groups)
    {
        Span<byte> b = stackalloc byte[72];
        switch (choice & 15)
        {
            case ConvComposed:
                base.ConvolutionBackwardWeightKernel(s[0], s[1], s[2], in g, filters, groups);
                return;
            case ConvDepthwise:
                Run("conv_depthwise_backward_weight", RowGroups((long)filters * g.KH * g.KW), 1, 1, s, ConvPush(b, g, filters, groups, 0, 1));
                return;
        }

        int splits = 1 << ((choice >> 4) & 7);
        var (rows, columns, _) = ConvProduct(ConvolutionShapes.Weight, in g, filters, groups);
        int count = filters * (g.PatchSize / groups);
        if (splits == 1)
        {
            RunConvProduct("conv_backward_weight", "conv_backward_weight_tile", choice, rows, columns, groups, [s[0], s[1], s[2], s[2]], ConvPush(b, g, filters, groups, 0, 1));
            return;
        }

        var part = Allocate(splits * count, zeroed: false);
        try
        {
            RunConvProduct("conv_backward_weight", "conv_backward_weight_tile", choice, rows, columns, groups * splits, [s[0], s[1], s[2], part], ConvPush(b, g, filters, groups, 0, splits));
            Span<byte> r = stackalloc byte[8];
            Grid("conv_split_reduce", count, [part, s[2]], new Push(r).I(count).I(splits).Bytes);
        }
        finally
        {
            part.Release();                                                    // reused in queue order
        }
    }

    // ------------------------------------------------------------------ average pooling

    public override void AvgPoolKernel(Storage x, Storage y, in ConvGeometry g, bool countIncludePad, int padBottom, int padRight)
    {
        if ((long)g.N * g.C * g.H * g.W > int.MaxValue || !Fit(x, y))
        {
            base.AvgPoolKernel(x, y, in g, countIncludePad, padBottom, padRight);
            return;
        }

        Span<byte> b = stackalloc byte[68];
        Grid("avg_pool", (long)g.N * g.C * g.OH * g.OW, [x, y], PushPool(b, g, countIncludePad, padBottom, padRight));
    }

    public override void AvgPoolBackwardKernel(Storage dy, Storage dx, in ConvGeometry g, bool countIncludePad, int padBottom, int padRight)
    {
        if ((long)g.N * g.C * g.H * g.W > int.MaxValue || !Fit(dy, dx))
        {
            base.AvgPoolBackwardKernel(dy, dx, in g, countIncludePad, padBottom, padRight);
            return;
        }

        Span<byte> b = stackalloc byte[68];
        Grid("avg_pool_backward", (long)g.N * g.C * g.H * g.W, [dy, dx], PushPool(b, g, countIncludePad, padBottom, padRight));
    }

    private static ReadOnlySpan<byte> PushPool(Span<byte> bytes, in ConvGeometry g, bool countIncludePad, int padBottom, int padRight) =>
        new Push(bytes).I(g.N).I(g.C).I(g.H).I(g.W).I(g.KH).I(g.KW).I(g.SH).I(g.SW).I(g.PH).I(g.PW).I(g.OH).I(g.OW).I(1).I(1).B(countIncludePad)
            .I(padBottom).I(padRight).Bytes;

    // ------------------------------------------------------------------ resampling

    public override void ResizeNormalizeKernel(Storage x, Storage coefficients, Storage values, Storage y, int planes, int channels, int height, int width,
        int outHeight, int outWidth, int xTaps, int yTaps, bool bytes)
    {
        if ((long)planes * height * width > int.MaxValue || (long)planes * outHeight * outWidth > int.MaxValue || !Fit(x, coefficients, values, y))
        {
            base.ResizeNormalizeKernel(x, coefficients, values, y, planes, channels, height, width, outHeight, outWidth, xTaps, yTaps, bytes);
            return;
        }

        Span<byte> b = stackalloc byte[36];
        Grid("resize_normalize", (long)planes * outHeight * outWidth, [x, coefficients, values, y],
            new Push(b).I(planes).I(Math.Max(channels, 1)).I(height).I(width).I(outHeight).I(outWidth).I(xTaps).I(yTaps).B(bytes).Bytes);
    }

    // ------------------------------------------------------------------ interpolation and adaptive pooling

    public override void Interpolate2dKernel(Storage x, Storage y, int planes, int height, int width, int outHeight, int outWidth, InterpolationMode mode,
        bool alignCorners, float scaleHeight, float scaleWidth)
    {
        if ((long)planes * outHeight * outWidth > int.MaxValue || (long)planes * height * width > int.MaxValue || !Fit(x, y))
        {
            base.Interpolate2dKernel(x, y, planes, height, width, outHeight, outWidth, mode, alignCorners, scaleHeight, scaleWidth);
            return;
        }

        Span<byte> b = stackalloc byte[36];
        Grid("interpolate", (long)planes * outHeight * outWidth, [x, y], PushInterpolation(b, planes, height, width, outHeight, outWidth, mode, alignCorners, scaleHeight, scaleWidth));
    }

    public override void Interpolate2dBackwardKernel(Storage dy, Storage dx, int planes, int height, int width, int outHeight, int outWidth, InterpolationMode mode,
        bool alignCorners, float scaleHeight, float scaleWidth)
    {
        if ((long)planes * outHeight * outWidth > int.MaxValue || (long)planes * height * width > int.MaxValue || !Fit(dy, dx))
        {
            base.Interpolate2dBackwardKernel(dy, dx, planes, height, width, outHeight, outWidth, mode, alignCorners, scaleHeight, scaleWidth);
            return;
        }

        Span<byte> b = stackalloc byte[36];
        Grid("interpolate_backward", (long)planes * height * width, [dy, dx],
            PushInterpolation(b, planes, height, width, outHeight, outWidth, mode, alignCorners, scaleHeight, scaleWidth));
    }

    private static ReadOnlySpan<byte> PushInterpolation(Span<byte> bytes, int planes, int height, int width, int outHeight, int outWidth, InterpolationMode mode,
        bool alignCorners, float scaleHeight, float scaleWidth) =>
        new Push(bytes).I(planes).I(height).I(width).I(outHeight).I(outWidth).I(mode == InterpolationMode.Nearest ? 0 : 1).B(alignCorners).F(scaleHeight).F(scaleWidth).Bytes;

    // Whether the adaptive windows' index arithmetic ((o + 1) · size and (i + 1) · outSize) and the element counts stay
    // within an int.
    private static bool AdaptiveFits(int planes, int height, int width, int outHeight, int outWidth) =>
        (long)planes * height * width <= int.MaxValue && (long)planes * outHeight * outWidth <= int.MaxValue
        && (long)(outHeight + 1) * (height + 1) <= int.MaxValue && (long)(outWidth + 1) * (width + 1) <= int.MaxValue;

    public override void AdaptiveAvgPoolKernel(Storage x, Storage y, int planes, int height, int width, int outHeight, int outWidth)
    {
        if (!AdaptiveFits(planes, height, width, outHeight, outWidth) || !Fit(x, y))
        {
            base.AdaptiveAvgPoolKernel(x, y, planes, height, width, outHeight, outWidth);
            return;
        }

        Span<byte> b = stackalloc byte[20];
        Grid("adaptive_avg_pool", (long)planes * outHeight * outWidth, [x, y], new Push(b).I(planes).I(height).I(width).I(outHeight).I(outWidth).Bytes);
    }

    public override void AdaptiveAvgPoolBackwardKernel(Storage dy, Storage dx, int planes, int height, int width, int outHeight, int outWidth)
    {
        if (!AdaptiveFits(planes, height, width, outHeight, outWidth) || !Fit(dy, dx))
        {
            base.AdaptiveAvgPoolBackwardKernel(dy, dx, planes, height, width, outHeight, outWidth);
            return;
        }

        Span<byte> b = stackalloc byte[20];
        Grid("adaptive_avg_pool_backward", (long)planes * height * width, [dy, dx], new Push(b).I(planes).I(height).I(width).I(outHeight).I(outWidth).Bytes);
    }

    public override void AdaptiveMaxPoolKernel(Storage x, Storage y, Storage argmax, int planes, int height, int width, int outHeight, int outWidth)
    {
        if (!AdaptiveFits(planes, height, width, outHeight, outWidth) || !Fit(x, y, argmax))
        {
            base.AdaptiveMaxPoolKernel(x, y, argmax, planes, height, width, outHeight, outWidth);
            return;
        }

        Span<byte> b = stackalloc byte[20];
        Grid("adaptive_max_pool", (long)planes * outHeight * outWidth, [x, y, argmax], new Push(b).I(planes).I(height).I(width).I(outHeight).I(outWidth).Bytes);
    }

    public override void AdaptiveMaxPoolBackwardKernel(Storage dy, Storage argmax, Storage dx, int planes, int height, int width, int outHeight, int outWidth)
    {
        if (!AdaptiveFits(planes, height, width, outHeight, outWidth) || !Fit(dy, argmax, dx))
        {
            base.AdaptiveMaxPoolBackwardKernel(dy, argmax, dx, planes, height, width, outHeight, outWidth);
            return;
        }

        Span<byte> b = stackalloc byte[20];
        Grid("adaptive_max_pool_backward", (long)planes * height * width, [dy, argmax, dx], new Push(b).I(planes).I(height).I(width).I(outHeight).I(outWidth).Bytes);
    }
}
