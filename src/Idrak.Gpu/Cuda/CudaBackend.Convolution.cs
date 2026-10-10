// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Cuda;

// Convolutions, average pooling and image resampling (PtxKernels.Convolution.cs).
//
// A convolution or one of its gradients runs one of: the composed path (patches unfolded by im2col and the measured
// products, on bfloat16 tensor cores under MixedPrecision where the products go there: Backend.Convolution.cs), an
// implicit product reading the operands where they lie (64 × 64 tiles with 4 × 4 outputs a thread, or 16 × 16 tiles;
// the weight gradient also at 1, 4, 16, 64, 256 or 1,024 splits of its sum over positions: ConvWeightSplits), or the
// depthwise kernels (one input channel per group). Which is fastest is measured per shape on first use
// (TuneOp.Convolution, keyed by the shape, the pass, the precision, the activation and the groups; CudaBackend.Tuning.cs)
// and kept in the tuning cache. While nothing can be measured (a graph being recorded, the profiler, IDRAK_AUTOTUNE=0)
// the formula's choice runs: the depthwise kernels where they apply, else the implicit product, on 64 × 64 tiles when
// they would be at least half full, with the most splits of ConvWeightSplits that leave each at least 4,096 positions.
// The implicit products and the depthwise kernels keep nothing beyond their output (and the weight gradient's partial
// sums).
internal sealed unsafe partial class CudaBackend
{
    private const int ConvComposed = 0, ConvTile = 1, ConvBlocked = 2, ConvDepthwise = 3;

    // Tests and benchmarks: forces one path (ConvComposed, ConvTile, ConvBlocked or ConvDepthwise; the formula's splits)
    // where it can run.
    internal static int? ConvolutionPath { get; set; }

    // Tests: with ConvolutionPath, the weight gradient's split count where it is a candidate (else the path's first).
    internal static int? ConvolutionSplits { get; set; }

    // The split counts past ConvolutionShapes.SplitChoices tried on CUDA, for sums over many positions with few tiles.
    private static ReadOnlySpan<int> WideSplits => [256, 1024];

    public override void ConvolutionKernel(Storage x, Storage weight, Storage? bias, Storage y, in ConvGeometry g, int filters, int groups, ConvActivation activation)
    {
        if (!ConvFits(in g, filters, groups))
        {
            base.ConvolutionKernel(x, weight, bias, y, in g, filters, groups, activation);
            return;
        }

        var geometry = g;
        int flags = (bias is null ? 0 : 1) | (int)activation << 1;
        int choice = ConvChoice(ConvolutionShapes.Forward, in g, filters, groups, activation, y.Length,
            (c, output) => RunConvolution(c, x, weight, bias, output, geometry, filters, groups, flags, activation), writesOutput: true, y);
        RunConvolution(choice, x, weight, bias, y, g, filters, groups, flags, activation);
    }

    public override void ConvolutionBackwardInputKernel(Storage dy, Storage weight, Storage dx, in ConvGeometry g, int filters, int groups)
    {
        if (!ConvFits(in g, filters, groups))
        {
            base.ConvolutionBackwardInputKernel(dy, weight, dx, in g, filters, groups);
            return;
        }

        var geometry = g;
        int choice = ConvChoice(ConvolutionShapes.Input, in g, filters, groups, ConvActivation.None, dx.Length,
            (c, output) => RunConvolutionInput(c, dy, weight, output, geometry, filters, groups), writesOutput: false, dx);
        RunConvolutionInput(choice, dy, weight, dx, g, filters, groups);
    }

    public override void ConvolutionBackwardWeightKernel(Storage x, Storage dy, Storage dweight, in ConvGeometry g, int filters, int groups)
    {
        if (!ConvFits(in g, filters, groups))
        {
            base.ConvolutionBackwardWeightKernel(x, dy, dweight, in g, filters, groups);
            return;
        }

        var geometry = g;
        int choice = ConvChoice(ConvolutionShapes.Weight, in g, filters, groups, ConvActivation.None, dweight.Length,
            (c, output) => RunConvolutionWeight(c, x, dy, output, geometry, filters, groups), writesOutput: false, dweight);
        RunConvolutionWeight(choice, x, dy, dweight, g, filters, groups);
    }

    // Whether the kernels index every operand within 32 bits (the storages' lengths are ints; the sums of positions too).
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

    // The path of one pass for this shape: measured (the candidates write `output`, or scratch where the pass adds to its
    // output), stored, forced, or the formula's (see the file's comment).
    private int ConvChoice(int pass, in ConvGeometry g, int filters, int groups, ConvActivation activation, int outputLength, Action<int, Storage> run, bool writesOutput,
        Storage output)
    {
        var candidates = ConvCandidates(pass, in g, filters, groups);
        int fallback = ConvFormula(pass, in g, filters, groups, candidates);
        if (ConvolutionPath is int forced)
        {
            int at = Array.FindIndex(candidates, c => (c & 15) == forced && (ConvolutionSplits is not int s || 1 << ((c >> 4) & 15) == s));
            at = at >= 0 ? at : Array.FindIndex(candidates, c => (c & 15) == forced);
            return at >= 0 ? candidates[at] : fallback;
        }

        if (!ConvolutionShapes.Key(in g, filters, groups, out var shape) || candidates.Length <= 1)
        {
            return fallback;
        }

        var key = new TuneKey(TuneOp.Convolution, ConvolutionShapes.Variant(pass, activation, groups), shape.A, shape.B, shape.C, shape.D, shape.E, shape.F);
        if (TunedKnown(key) || !CanMeasure)
        {
            return Tune(key, candidates, fallback, _ => { });                 // the known (or stored) choice, else the formula
        }

        if (writesOutput)
        {
            return Tune(key, candidates, fallback, c => run(c, output));
        }

        var scratch = Allocate(outputLength, zeroed: true);
        try
        {
            return Tune(key, candidates, fallback, c => run(c, scratch));
        }
        finally
        {
            scratch.Release();
        }
    }

    // The candidates of a pass: the composed path where its patches fit (half the memory the device reports free), the
    // implicit product on both tiles (the weight gradient at each split count whose partial sums fit), the depthwise
    // kernels where every group reads one channel.
    private int[] ConvCandidates(int pass, in ConvGeometry g, int filters, int groups)
    {
        var candidates = new List<int>();
        long patches = (long)g.Positions * g.PatchSize;
        if (patches <= int.MaxValue && 4 * patches <= (AvailableMemory() ?? long.MaxValue) / 2)
        {
            candidates.Add(ConvComposed);
        }

        int[] splits = pass == ConvolutionShapes.Weight ? ConvWeightSplits(in g, filters, groups) : [1];
        var (rows, _, _) = ConvProduct(pass, in g, filters, groups);
        foreach (int variant in new[] { ConvTile, ConvBlocked })
        {
            int edge = variant == ConvBlocked ? PtxKernels.ConvBlockedEdge : PtxKernels.ConvTileEdge;
            if ((rows + edge - 1) / edge > _limits.MaxGridY)
            {
                continue;
            }

            foreach (int s in splits)
            {
                candidates.Add(variant | System.Numerics.BitOperations.Log2((uint)s) << 4);
            }
        }

        if (ConvolutionShapes.Depthwise(in g, groups))
        {
            candidates.Add(ConvDepthwise);
        }

        return [.. candidates];
    }

    // The formula's choice among the candidates (see the file's comment).
    private static int ConvFormula(int pass, in ConvGeometry g, int filters, int groups, int[] candidates)
    {
        if (Array.IndexOf(candidates, ConvDepthwise) >= 0)
        {
            return ConvDepthwise;
        }

        var (rows, columns, _) = ConvProduct(pass, in g, filters, groups);
        int edge = PtxKernels.ConvBlockedEdge;
        int splitBits = pass == ConvolutionShapes.Weight ? System.Numerics.BitOperations.Log2((uint)ConvWeightFormulaSplits(in g, filters, groups)) << 4 : 0;
        int variant = rows >= edge / 2 && columns >= edge / 2 ? ConvBlocked : ConvTile;
        foreach (int v in new[] { variant | splitBits, variant, ConvTile | splitBits, ConvTile, ConvComposed })
        {
            if (Array.IndexOf(candidates, v) >= 0)
            {
                return v;
            }
        }

        return candidates.Length > 0 ? candidates[0] : ConvComposed;
    }

    // The weight gradient's split counts: ConvolutionShapes.SplitCounts, and WideSplits where each split keeps at least 256
    // positions and the partial sums take no more memory than the input (sizes alone decide: rules 77, 81). A sum over a
    // million positions into a few tiles (a first layer: one input channel, 32 filters) otherwise runs on 64 blocks.
    private static int[] ConvWeightSplits(in ConvGeometry g, int filters, int groups)
    {
        long positions = (long)g.N * g.OH * g.OW, count = (long)filters * (g.PatchSize / groups), input = (long)g.N * g.C * g.H * g.W;
        var splits = new List<int>(ConvolutionShapes.SplitCounts(positions));
        foreach (int s in WideSplits)
        {
            if (positions / s >= 256 && s * count <= input)
            {
                splits.Add(s);
            }
        }

        return [.. splits];
    }

    // The splits while nothing is measured: the most of ConvWeightSplits that leave each split at least 4,096 positions
    // (ConvolutionShapes.FormulaSplits over the wider counts).
    private static int ConvWeightFormulaSplits(in ConvGeometry g, int filters, int groups)
    {
        long positions = (long)g.N * g.OH * g.OW;
        int most = 1;
        foreach (int s in ConvWeightSplits(in g, filters, groups))
        {
            if (s == 1 || positions / s >= 4096)
            {
                most = Math.Max(most, s);
            }
        }

        return most;
    }

    // The 18 geometry arguments every convolution kernel takes after its pointers.
    private static void ConvArguments(Span<ulong> args, in ConvGeometry g, int filters, int groups, int flags, int splits)
    {
        int[] values = [g.N, g.C, g.H, g.W, g.KH, g.KW, g.SH, g.SW, g.PH, g.PW, g.OH, g.OW, g.DH, g.DW, filters, groups, flags, splits];
        for (int i = 0; i < values.Length; i++)
        {
            args[i] = (uint)values[i];
        }
    }

    // Launches an implicit product: blocks of its tile along x (columns) and y (rows), the batch entries along z (the
    // kernels loop over those past the grid's z limit).
    private void LaunchConvProduct(string kernel, int choice, int rows, int columns, int batch, ReadOnlySpan<ulong> args)
    {
        bool blocked = (choice & 15) == ConvBlocked;
        int edge = blocked ? PtxKernels.ConvBlockedEdge : PtxKernels.ConvTileEdge;
        LaunchGrid(K(blocked ? kernel + "_f32" : kernel + "_tile_f32"), (uint)((columns + edge - 1) / edge), (uint)((rows + edge - 1) / edge),
            (uint)Math.Clamp(batch, 1, _limits.MaxGridZ), PtxKernels.ConvThreads, 1, 1, args);
    }

    private void RunConvolution(int choice, Storage x, Storage weight, Storage? bias, Storage y, ConvGeometry g, int filters, int groups, int flags, ConvActivation activation)
    {
        if ((choice & 15) == ConvComposed)
        {
            base.ConvolutionKernel(x, weight, bias, y, in g, filters, groups, activation);
            return;
        }

        // x, weight, bias (the weight when there is none), y, the geometry, and the element count of the depthwise kernel.
        Span<ulong> args = stackalloc ulong[23];
        args[0] = P(x);
        args[1] = P(weight);
        args[2] = bias is null ? P(weight) : P(bias);
        args[3] = P(y);
        ConvArguments(args[4..], g, filters, groups, flags, 1);
        if ((choice & 15) == ConvDepthwise)
        {
            int n = g.N * filters * g.OH * g.OW;
            args[22] = (uint)n;
            Launch1D(K("conv_dw_f32"), n, args);
            return;
        }

        var (rows, columns, batch) = ConvProduct(ConvolutionShapes.Forward, in g, filters, groups);
        LaunchConvProduct("conv_fwd", choice, rows, columns, batch, args[..22]);
    }

    private void RunConvolutionInput(int choice, Storage dy, Storage weight, Storage dx, ConvGeometry g, int filters, int groups)
    {
        if ((choice & 15) == ConvComposed)
        {
            base.ConvolutionBackwardInputKernel(dy, weight, dx, in g, filters, groups);
            return;
        }

        Span<ulong> args = stackalloc ulong[22];
        args[0] = P(dy);
        args[1] = P(weight);
        args[2] = P(dx);
        ConvArguments(args[3..], g, filters, groups, 0, 1);
        if ((choice & 15) == ConvDepthwise)
        {
            int n = g.N * g.C * g.H * g.W;
            args[21] = (uint)n;
            Launch1D(K("conv_dw_bwd_input_f32"), n, args);
            return;
        }

        var (rows, columns, batch) = ConvProduct(ConvolutionShapes.Input, in g, filters, groups);
        LaunchConvProduct("conv_bwd_input", choice, rows, columns, batch, args[..21]);
    }

    // Several splits write their partial sums to a temporary [splits, filters, patch] that conv_split_reduce_f32 adds to
    // dweight in split order.
    private void RunConvolutionWeight(int choice, Storage x, Storage dy, Storage dweight, ConvGeometry g, int filters, int groups)
    {
        int variant = choice & 15;
        if (variant == ConvComposed)
        {
            base.ConvolutionBackwardWeightKernel(x, dy, dweight, in g, filters, groups);
            return;
        }

        Span<ulong> args = stackalloc ulong[22];
        args[0] = P(x);
        args[1] = P(dy);
        args[2] = P(dweight);
        if (variant == ConvDepthwise)
        {
            ConvArguments(args[3..], g, filters, groups, 0, 1);
            int weights = filters * g.KH * g.KW;
            if (weights > 0)
            {
                LaunchGrid(K("conv_dw_bwd_weight_f32"), (uint)weights, 1, 1, (uint)_shapes.BlockSize, 1, 1, args[..21]);
            }

            return;
        }

        int splits = 1 << ((choice >> 4) & 15);
        var (rows, columns, _) = ConvProduct(ConvolutionShapes.Weight, in g, filters, groups);
        int count = filters * (g.PatchSize / groups);
        if (splits == 1)
        {
            args[3] = P(dweight);
            ConvArguments(args[4..], g, filters, groups, 0, 1);
            LaunchConvProduct("conv_bwd_weight", choice, rows, columns, groups, args);
            return;
        }

        var part = Allocate(splits * count, zeroed: false);
        try
        {
            args[3] = P(part);
            ConvArguments(args[4..], g, filters, groups, 0, splits);
            LaunchConvProduct("conv_bwd_weight", choice, rows, columns, groups * splits, args);
            Launch1D(K("conv_split_reduce_f32"), count, P(part), P(dweight), U(count), U(splits), U(count));
        }
        finally
        {
            part.Release();                                                    // reused in stream order
        }
    }

    // ------------------------------------------------------------------ average pooling and resampling

    public override void AvgPoolKernel(Storage x, Storage y, in ConvGeometry g, bool countIncludePad, int padBottom, int padRight)
    {
        int n = g.N * g.C * g.OH * g.OW;
        Launch1D(K("avgpool_f32"), n, P(x), P(y), U(g.H), U(g.W), U(g.KH), U(g.KW), U(g.SH), U(g.SW), U(g.PH), U(g.PW), U(g.OH), U(g.OW),
            U(countIncludePad ? 1 : 0), U(padBottom), U(padRight), U(n));
    }

    public override void AvgPoolBackwardKernel(Storage dy, Storage dx, in ConvGeometry g, bool countIncludePad, int padBottom, int padRight)
    {
        int n = g.N * g.C * g.H * g.W;
        Launch1D(K("avgpool_bwd_f32"), n, P(dy), P(dx), U(g.H), U(g.W), U(g.KH), U(g.KW), U(g.SH), U(g.SW), U(g.PH), U(g.PW), U(g.OH), U(g.OW),
            U(countIncludePad ? 1 : 0), U(padBottom), U(padRight), U(n));
    }

    public override void ResizeNormalizeKernel(Storage x, Storage coefficients, Storage values, Storage y, int planes, int channels, int height, int width,
        int outHeight, int outWidth, int xTaps, int yTaps, bool bytes)
    {
        if ((long)planes * height * width > int.MaxValue || (long)planes * outHeight * outWidth > int.MaxValue)
        {
            base.ResizeNormalizeKernel(x, coefficients, values, y, planes, channels, height, width, outHeight, outWidth, xTaps, yTaps, bytes);
            return;
        }

        int n = planes * outHeight * outWidth;
        Launch1D(K("resize_normalize_f32"), n, P(x), P(coefficients), P(values), P(y), U(planes), U(Math.Max(channels, 1)), U(height), U(width), U(outHeight),
            U(outWidth), U(xTaps), U(yTaps), U(bytes ? 1 : 0), U(n));
    }

    // ------------------------------------------------------------------ interpolation and adaptive pooling (PtxKernels.Resampling.cs)

    public override void Interpolate2dKernel(Storage x, Storage y, int planes, int height, int width, int outHeight, int outWidth, InterpolationMode mode,
        bool alignCorners, float scaleHeight, float scaleWidth)
    {
        int n = planes * outHeight * outWidth;
        Launch1D(K("interpolate_f32"), n, P(x), P(y), U(planes), U(height), U(width), U(outHeight), U(outWidth), U(mode == InterpolationMode.Nearest ? 0 : 1),
            U(alignCorners ? 1 : 0), F(scaleHeight), F(scaleWidth), U(n));
    }

    public override void Interpolate2dBackwardKernel(Storage dy, Storage dx, int planes, int height, int width, int outHeight, int outWidth, InterpolationMode mode,
        bool alignCorners, float scaleHeight, float scaleWidth)
    {
        int n = planes * height * width;
        Launch1D(K("interpolate_bwd_f32"), n, P(dy), P(dx), U(planes), U(height), U(width), U(outHeight), U(outWidth), U(mode == InterpolationMode.Nearest ? 0 : 1),
            U(alignCorners ? 1 : 0), F(scaleHeight), F(scaleWidth), U(n));
    }

    // The adaptive windows' index arithmetic ((o + 1) · size, (i + 1) · outSize) stays within 32 bits.
    private static bool AdaptiveFits(int height, int width, int outHeight, int outWidth) =>
        (long)(outHeight + 1) * (height + 1) <= int.MaxValue && (long)(outWidth + 1) * (width + 1) <= int.MaxValue;

    public override void AdaptiveAvgPoolKernel(Storage x, Storage y, int planes, int height, int width, int outHeight, int outWidth)
    {
        if (!AdaptiveFits(height, width, outHeight, outWidth))
        {
            base.AdaptiveAvgPoolKernel(x, y, planes, height, width, outHeight, outWidth);
            return;
        }

        int n = planes * outHeight * outWidth;
        Launch1D(K("adaptive_avgpool_f32"), n, P(x), P(y), U(planes), U(height), U(width), U(outHeight), U(outWidth), U(n));
    }

    public override void AdaptiveAvgPoolBackwardKernel(Storage dy, Storage dx, int planes, int height, int width, int outHeight, int outWidth)
    {
        if (!AdaptiveFits(height, width, outHeight, outWidth))
        {
            base.AdaptiveAvgPoolBackwardKernel(dy, dx, planes, height, width, outHeight, outWidth);
            return;
        }

        int n = planes * height * width;
        Launch1D(K("adaptive_avgpool_bwd_f32"), n, P(dy), P(dx), U(planes), U(height), U(width), U(outHeight), U(outWidth), U(n));
    }

    public override void AdaptiveMaxPoolKernel(Storage x, Storage y, Storage argmax, int planes, int height, int width, int outHeight, int outWidth)
    {
        if (!AdaptiveFits(height, width, outHeight, outWidth))
        {
            base.AdaptiveMaxPoolKernel(x, y, argmax, planes, height, width, outHeight, outWidth);
            return;
        }

        int n = planes * outHeight * outWidth;
        Launch1D(K("adaptive_maxpool_f32"), n, P(x), P(y), P(argmax), U(planes), U(height), U(width), U(outHeight), U(outWidth), U(n));
    }

    public override void AdaptiveMaxPoolBackwardKernel(Storage dy, Storage argmax, Storage dx, int planes, int height, int width, int outHeight, int outWidth)
    {
        if (!AdaptiveFits(height, width, outHeight, outWidth))
        {
            base.AdaptiveMaxPoolBackwardKernel(dy, argmax, dx, planes, height, width, outHeight, outWidth);
            return;
        }

        int n = planes * height * width;
        Launch1D(K("adaptive_maxpool_bwd_f32"), n, P(dy), P(argmax), P(dx), U(planes), U(height), U(width), U(outHeight), U(outWidth), U(n));
    }
}
