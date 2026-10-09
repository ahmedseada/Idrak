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

        var geometry = g;
        int flags = (bias is null ? 0 : 1) | (int)activation << 1;
        Storage[] storages = [x, weight, bias ?? weight, y];
        int choice = ConvChoice(ConvolutionShapes.Forward, in g, filters, groups, activation, storages, 1UL << 3,
            (c, s) => RunConvolution(c, s, geometry, filters, groups, flags, activation));
        RunConvolution(choice, storages, g, filters, groups, flags, activation);
    }

    public override void ConvolutionBackwardInputKernel(Storage dy, Storage weight, Storage dx, in ConvGeometry g, int filters, int groups)
    {
        if (!ConvFits(in g, filters, groups) || !Fit(dy, weight, dx))
        {
            base.ConvolutionBackwardInputKernel(dy, weight, dx, in g, filters, groups);
            return;
        }

        var geometry = g;
        Storage[] storages = [dy, weight, dx];
        int choice = ConvChoice(ConvolutionShapes.Input, in g, filters, groups, ConvActivation.None, storages, 1UL << 2,
            (c, s) => RunConvolutionInput(c, s, geometry, filters, groups));
        RunConvolutionInput(choice, storages, g, filters, groups);
    }

    public override void ConvolutionBackwardWeightKernel(Storage x, Storage dy, Storage dweight, in ConvGeometry g, int filters, int groups)
    {
        if (!ConvFits(in g, filters, groups) || !Fit(x, dy, dweight))
        {
            base.ConvolutionBackwardWeightKernel(x, dy, dweight, in g, filters, groups);
            return;
        }

        var geometry = g;
        Storage[] storages = [x, dy, dweight];
        int choice = ConvChoice(ConvolutionShapes.Weight, in g, filters, groups, ConvActivation.None, storages, 1UL << 2,
            (c, s) => RunConvolutionWeight(c, s, geometry, filters, groups));
        RunConvolutionWeight(choice, storages, g, filters, groups);
    }

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

    // The path of one pass for this shape: measured, stored, forced, or the formula's (see the file's comment).
    private int ConvChoice(int pass, in ConvGeometry g, int filters, int groups, ConvActivation activation, Storage[] storages, ulong writes, Action<int, Storage[]> run)
    {
        var candidates = ConvCandidates(pass, in g, filters, groups);
        int fallback = ConvFormula(pass, in g, filters, groups, candidates);
        if (ConvolutionPath is int forced)
        {
            int at = Array.FindIndex(candidates, c => (c & 15) == forced && (WidthOf(c) == Width || forced == ConvComposed));
            return at >= 0 ? candidates[at] : fallback;
        }

        if (!ConvolutionShapes.Key(in g, filters, groups, out var shape))
        {
            return fallback;
        }

        var key = new VulkanTuneKey(VulkanTuneOp.Convolution, ConvolutionShapes.Variant(pass, activation, groups), shape.A, shape.B, shape.C, shape.D, shape.E, shape.F);
        if (TryTuned(key, out int known) && Array.IndexOf(candidates, known) >= 0)
        {
            return known;
        }

        if (!CanTune || candidates.Length <= 1)
        {
            return fallback;
        }

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

    // The candidates of a pass: the composed path where its patches fit the device (a binding, and half the memory it
    // reports free), the implicit product tiled and blocked at each candidate width where the workgroup counts take it
    // (the weight gradient at each split count), the depthwise kernels where every group reads one channel.
    private int[] ConvCandidates(int pass, in ConvGeometry g, int filters, int groups)
    {
        var candidates = new List<int>();
        long patches = (long)g.Positions * g.PatchSize;
        long free = AvailableMemory() ?? long.MaxValue;
        if (patches <= int.MaxValue && BlockBytes((int)Math.Min(patches, int.MaxValue)) <= MaxStorageBytes && 4 * patches <= free / 2)
        {
            candidates.Add(WithWidth(Width, ConvComposed));
        }

        var (rows, columns, _) = ConvProduct(pass, in g, filters, groups);
        int[] splits = pass == ConvolutionShapes.Weight ? ConvolutionShapes.SplitCounts((long)g.N * g.OH * g.OW) : [1];
        foreach (int width in CandidateWidths)
        {
            foreach (int variant in new[] { ConvTile, ConvBlocked })
            {
                int edge = VulkanKernels.MatSide(width) * (variant == ConvBlocked ? VulkanKernels.MatPer : 1);
                if (!MatFits(edge, rows, columns))
                {
                    continue;
                }

                foreach (int s in splits)
                {
                    if (s == 1 || (long)s * filters * (g.PatchSize / groups) <= int.MaxValue && BlockBytes(s * filters * (g.PatchSize / groups)) <= MaxStorageBytes)
                    {
                        candidates.Add(WithWidth(width, variant | System.Numerics.BitOperations.Log2((uint)s) << 4));
                    }
                }
            }
        }

        if (ConvolutionShapes.Depthwise(in g, groups))
        {
            candidates.Add(WithWidth(Width, ConvDepthwise));
        }

        return [.. candidates];
    }

    // The formula's choice among the candidates (see the file's comment).
    private int ConvFormula(int pass, in ConvGeometry g, int filters, int groups, int[] candidates)
    {
        int Find(int variant) => Array.IndexOf(candidates, WithWidth(Width, variant)) >= 0 ? WithWidth(Width, variant) : -1;
        if (Find(ConvDepthwise) is int depthwise and >= 0)
        {
            return depthwise;
        }

        var (rows, columns, _) = ConvProduct(pass, in g, filters, groups);
        int edge = VulkanKernels.MatPer * VulkanKernels.MatSide(Width);
        int splitBits = pass == ConvolutionShapes.Weight ? System.Numerics.BitOperations.Log2((uint)ConvolutionShapes.FormulaSplits((long)g.N * g.OH * g.OW)) << 4 : 0;
        int variant = rows >= edge / 2 && columns >= edge / 2 ? ConvBlocked : ConvTile;
        foreach (int v in new[] { variant | splitBits, variant, ConvTile | splitBits, ConvTile, ConvComposed })
        {
            if (Find(v) is int found and >= 0)
            {
                return found;
            }
        }

        return candidates.Length > 0 ? candidates[0] : WithWidth(Width, ConvComposed);
    }

    // The 72 bytes of push constants of every convolution kernel.
    private static ReadOnlySpan<byte> ConvPush(Span<byte> bytes, in ConvGeometry g, int filters, int groups, int flags, int splits) =>
        new Push(bytes).I(g.N).I(g.C).I(g.H).I(g.W).I(g.KH).I(g.KW).I(g.SH).I(g.SW).I(g.PH).I(g.PW).I(g.OH).I(g.OW).I(g.DH).I(g.DW)
            .I(filters).I(groups).I(flags).I(splits).Bytes;

    // Dispatches an implicit product: blocks of the tile along x (columns) and y (rows), the batch entries along z (the
    // kernels loop over those past the device's count).
    private void RunConvProduct(string kernel, int choice, int rows, int columns, int batch, ReadOnlySpan<Storage> storages, ReadOnlySpan<byte> push)
    {
        int width = WidthOf(choice), variant = choice & 15;
        int edge = VulkanKernels.MatSide(width) * (variant == ConvBlocked ? VulkanKernels.MatPer : 1);
        RunAt(variant == ConvBlocked ? kernel : kernel + "_tile", width, (uint)((columns + edge - 1) / edge), (uint)((rows + edge - 1) / edge),
            (uint)Math.Clamp(batch, 1, Limits.MaxGroupsZ), storages, push);
    }

    // storages: x, weight, bias (the weight when none), y.
    private void RunConvolution(int choice, Storage[] s, ConvGeometry g, int filters, int groups, int flags, ConvActivation activation)
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
                RunConvProduct("conv_forward", choice, rows, columns, batch, s, ConvPush(b, g, filters, groups, flags, 1));
                return;
            }
        }
    }

    // storages: dy, weight, dx.
    private void RunConvolutionInput(int choice, Storage[] s, ConvGeometry g, int filters, int groups)
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
                RunConvProduct("conv_backward_input", choice, rows, columns, batch, s, ConvPush(b, g, filters, groups, 0, 1));
                return;
        }
    }

    // storages: x, dy, dweight. Several splits write their partial sums to a temporary [splits, filters, patch] that
    // conv_split_reduce adds to dweight in split order.
    private void RunConvolutionWeight(int choice, Storage[] s, ConvGeometry g, int filters, int groups)
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
            RunConvProduct("conv_backward_weight", choice, rows, columns, groups, [s[0], s[1], s[2], s[2]], ConvPush(b, g, filters, groups, 0, 1));
            return;
        }

        var part = Allocate(splits * count, zeroed: false);
        try
        {
            RunConvProduct("conv_backward_weight", choice, rows, columns, groups * splits, [s[0], s[1], s[2], part], ConvPush(b, g, filters, groups, 0, splits));
            Span<byte> r = stackalloc byte[8];
            Grid("conv_split_reduce", count, [part, s[2]], new Push(r).I(count).I(splits).Bytes);
        }
        finally
        {
            part.Release();                                                    // reused in queue order
        }
    }

    // ------------------------------------------------------------------ average pooling

    public override void AvgPoolKernel(Storage x, Storage y, in ConvGeometry g, bool countIncludePad)
    {
        if ((long)g.N * g.C * g.H * g.W > int.MaxValue || !Fit(x, y))
        {
            base.AvgPoolKernel(x, y, in g, countIncludePad);
            return;
        }

        Span<byte> b = stackalloc byte[60];
        Grid("avg_pool", (long)g.N * g.C * g.OH * g.OW, [x, y], PushPool(b, g, countIncludePad));
    }

    public override void AvgPoolBackwardKernel(Storage dy, Storage dx, in ConvGeometry g, bool countIncludePad)
    {
        if ((long)g.N * g.C * g.H * g.W > int.MaxValue || !Fit(dy, dx))
        {
            base.AvgPoolBackwardKernel(dy, dx, in g, countIncludePad);
            return;
        }

        Span<byte> b = stackalloc byte[60];
        Grid("avg_pool_backward", (long)g.N * g.C * g.H * g.W, [dy, dx], PushPool(b, g, countIncludePad));
    }

    private static ReadOnlySpan<byte> PushPool(Span<byte> bytes, in ConvGeometry g, bool countIncludePad) =>
        new Push(bytes).I(g.N).I(g.C).I(g.H).I(g.W).I(g.KH).I(g.KW).I(g.SH).I(g.SW).I(g.PH).I(g.PW).I(g.OH).I(g.OW).I(1).I(1).B(countIncludePad).Bytes;

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
}
