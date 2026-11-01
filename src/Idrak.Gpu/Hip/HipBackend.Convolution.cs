// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Abstraction.Devices;
using Idrak.Gpu.Cuda;

namespace Idrak.Gpu.Hip;

// HIP's convolution, pooling and resampling operations on its own GCN kernels (GcnKernels.Convolution.cs), with CUDA's
// arguments and launch rules: the unfolded patches and their adjoint, maximum, average and adaptive pooling with their
// gradients (every gradient a gather in output order: no atomics, the same bits every run), interpolation and its
// gradient, the resampling with normalization, and the depthwise convolution with both gradients.
//
// A convolution or one of its gradients runs the depthwise kernels where every group reads one channel (CUDA's formula
// takes them first), else the composed path (Backend.Convolution.cs: the patches unfolded by im2col_f32, the products of
// GcnKernels.Products.cs, the permutations and the bias and activation on HIP's own kernels), CUDA's composed candidate,
// or one of CUDA's implicit products (16 × 16 tiles, G6c; 64 × 64 blocked tiles, HIP parity), which read the operands where
// they lie and keep nothing beyond their output (and the weight gradient's partial sums). Which is fastest is measured
// per shape on first use, as CUDA's (TuneOp.Convolution, the weight gradient at each split count), and kept in HIP's
// tuning table (G7a); while nothing can be measured CUDA's formula chooses: the composed path for a product of fewer rows
// than a quarter of the small tile, else the 64 × 64 tiles when they would be at least half full, with the most splits
// that leave each at least 4,096 positions. CUDA's atomic count-only maxpool_bwd_f32 is chained and walked instead
// (HipBackend.Training.cs).
//
// Every per-output kernel walks its outputs grid-stride, launched as HIP's element-wise operations (PointwiseBlocks:
// rule 81); conv_dw_bwd_weight_f32 takes a block a weight, within the device's grid (rule 84). Each launch is labelled for
// --profile with its kernel, its shape and its floating-point operations (ConvolutionCost); the host fallback takes over
// when the kernels are unavailable, a count reaches 2^30 (the kernels' 32-bit byte offsets) or a size the kernels divide
// by is not positive.
internal sealed unsafe partial class HipBackend
{
    /// <summary>The cost of a convolution, pooling or resampling kernel from the table of GcnKernels.Convolution.cs; else the CTC's and the recurrent ones'.</summary>
    internal static (double Flops, double Bytes) ConvolutionCost(string kernel) => kernel switch
    {
        "im2col_f32" => (0, 8),
        "col2im_f32" or "maxpool_f32" or "avgpool_f32" or "adaptive_avgpool_f32" or "adaptive_maxpool_f32" => (1, 4),
        "maxpool_bwd_gather_f32" or "adaptive_maxpool_bwd_f32" => (1, 16),
        "avgpool_bwd_f32" => (2, 4),
        "adaptive_avgpool_bwd_f32" => (3, 12),
        "interpolate_f32" => (7, 20),
        "interpolate_bwd_f32" => (12, 24),
        "resize_normalize_f32" => (2, 8),
        "conv_dw_f32" or "conv_dw_bwd_input_f32" or "conv_dw_bwd_weight_f32" => (2, 8),
        "conv_fwd_tile_f32" or "conv_bwd_input_tile_f32" or "conv_bwd_weight_tile_f32" => (2, 0.5),
        "conv_fwd_f32" or "conv_bwd_input_f32" or "conv_bwd_weight_f32" => (2, 0.125),
        "conv_split_reduce_f32" => (1, 4),
        _ => CtcCost(kernel),
    };

    // Whether every count fits the kernels' 32-bit byte offsets (below 2^30 elements) and every divisor is positive.
    private static bool KernelSizes(ReadOnlySpan<long> counts, ReadOnlySpan<int> divisors)
    {
        foreach (long count in counts)
        {
            if (count < 0 || count >= GcnKernels.PointwiseMostElements)
            {
                return false;
            }
        }

        foreach (int divisor in divisors)
        {
            if (divisor <= 0)
            {
                return false;
            }
        }

        return true;
    }

    // Launches a per-output kernel over n outputs (grid-stride, as an element-wise kernel), labelled for the profile with
    // its shape and its floating-point operations (its cost a unit, `units` units).
    private void LaunchOutputs(GcnFunction function, long n, string shape, double units, params ReadOnlySpan<ulong> values)
    {
        if (n <= 0)
        {
            return;
        }

        if (Profiling)
        {
            LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"{function.Name} {shape}"), Cost(function.Name).Flops * units);
        }

        LaunchKernel(function, PointwiseBlocks(n), 1, values);
    }

    // A geometry's shape for the profile: channels, image, window, stride, dilation, images.
    private static string ConvolutionShape(in ConvGeometry g) =>
        string.Create(CultureInfo.InvariantCulture, $"{g.C}x{g.H}x{g.W} k{g.KH}x{g.KW} s{g.SH}x{g.SW} d{g.DH}x{g.DW} n{g.N}");

    // The image's elements and the output's, as longs.
    private static long InputCount(in ConvGeometry g) => (long)g.N * g.C * g.H * g.W;

    private static long OutputCount(in ConvGeometry g, int channels) => (long)g.N * channels * g.OH * g.OW;

    // ---- Unfolded patches ----------------------------------------------------------------------------------------------

    public override void Im2ColKernel(Storage x, Storage cols, in ConvGeometry g)
    {
        long n = (long)g.Positions * g.PatchSize;
        if (!KernelSizes([n, InputCount(g)], [g.PatchSize, g.KW, g.OW, g.OH, g.KH]) || Gcn("im2col_f32") is not { } f)
        {
            base.Im2ColKernel(x, cols, in g);
            return;
        }

        LaunchOutputs(f, n, ConvolutionShape(g), n, P(x), P(cols), U(g.C), U(g.H), U(g.W), U(g.KW), U(g.SH), U(g.SW), U(g.PH), U(g.PW), U(g.OW), U(g.PatchSize),
            U(g.KH * g.KW), U(g.OH * g.OW), U(g.DH), U(g.DW), U((int)n));
    }

    // A gather over dx's elements (each adds the column entries that read it, in a fixed order: rule 86).
    public override void Col2ImKernel(Storage dcols, Storage dx, in ConvGeometry g)
    {
        long n = InputCount(g);
        if (!KernelSizes([n, (long)g.Positions * g.PatchSize], [g.W, g.H, g.C, g.SH, g.SW]) || Gcn("col2im_f32") is not { } f)
        {
            base.Col2ImKernel(dcols, dx, in g);
            return;
        }

        LaunchOutputs(f, n, ConvolutionShape(g), n * g.KH * g.KW, P(dcols), P(dx), U(g.C), U(g.H), U(g.W), U(g.KH), U(g.KW), U(g.SH), U(g.SW), U(g.PH), U(g.PW),
            U(g.OH), U(g.OW), U(g.PatchSize), U(g.DH), U(g.DW), U((int)n));
    }

    // ---- Pooling -------------------------------------------------------------------------------------------------------

    public override void MaxPoolKernel(Storage x, Storage y, Storage argmax, in ConvGeometry g)
    {
        long n = OutputCount(g, g.C);
        if (!KernelSizes([n, InputCount(g)], [g.OW, g.OH]) || Gcn("maxpool_f32") is not { } f)
        {
            base.MaxPoolKernel(x, y, argmax, in g);
            return;
        }

        LaunchOutputs(f, n, ConvolutionShape(g), n * g.KH * g.KW, P(x), P(y), P(argmax), U(g.H), U(g.W), U(g.KH), U(g.KW), U(g.SH), U(g.SW), U(g.PH), U(g.PW), U(g.OW),
            U(g.OH * g.OW), U((int)n));
    }

    // A gather over dx's elements (each adds the gradients of the windows whose maximum it is, in output order: rule 86).
    public override void MaxPoolBackwardKernel(Storage dy, Storage argmax, Storage dx, in ConvGeometry g)
    {
        long n = InputCount(g);
        if (!KernelSizes([n, OutputCount(g, g.C)], [g.W, g.H, g.SH, g.SW]) || Gcn("maxpool_bwd_gather_f32") is not { } f)
        {
            base.MaxPoolBackwardKernel(dy, argmax, dx, in g);
            return;
        }

        LaunchOutputs(f, n, ConvolutionShape(g), n, P(dy), P(argmax), P(dx), U(g.H), U(g.W), U(g.KH), U(g.KW), U(g.SH), U(g.SW), U(g.PH), U(g.PW), U(g.OH), U(g.OW),
            U((int)n));
    }

    public override void AvgPoolKernel(Storage x, Storage y, in ConvGeometry g, bool countIncludePad, int padBottom, int padRight)
    {
        long n = OutputCount(g, g.C);
        if (!KernelSizes([n, InputCount(g)], [g.OW, g.OH]) || Gcn("avgpool_f32") is not { } f)
        {
            base.AvgPoolKernel(x, y, in g, countIncludePad, padBottom, padRight);
            return;
        }

        LaunchOutputs(f, n, ConvolutionShape(g), n * g.KH * g.KW, P(x), P(y), U(g.H), U(g.W), U(g.KH), U(g.KW), U(g.SH), U(g.SW), U(g.PH), U(g.PW), U(g.OH), U(g.OW),
            U(countIncludePad ? 1 : 0), U(padBottom), U(padRight), U((int)n));
    }

    public override void AvgPoolBackwardKernel(Storage dy, Storage dx, in ConvGeometry g, bool countIncludePad, int padBottom, int padRight)
    {
        long n = InputCount(g);
        if (!KernelSizes([n, OutputCount(g, g.C)], [g.W, g.H, g.SH, g.SW]) || Gcn("avgpool_bwd_f32") is not { } f)
        {
            base.AvgPoolBackwardKernel(dy, dx, in g, countIncludePad, padBottom, padRight);
            return;
        }

        LaunchOutputs(f, n, ConvolutionShape(g), n * g.KH * g.KW, P(dy), P(dx), U(g.H), U(g.W), U(g.KH), U(g.KW), U(g.SH), U(g.SW), U(g.PH), U(g.PW), U(g.OH), U(g.OW),
            U(countIncludePad ? 1 : 0), U(padBottom), U(padRight), U((int)n));
    }

    // ---- Resampling ----------------------------------------------------------------------------------------------------

    public override void ResizeNormalizeKernel(Storage x, Storage coefficients, Storage values, Storage y, int planes, int channels, int height, int width,
        int outHeight, int outWidth, int xTaps, int yTaps, bool bytes)
    {
        long n = (long)planes * outHeight * outWidth;
        long taps = (long)outWidth * (xTaps + 2) + (long)outHeight * (yTaps + 2);
        if (!KernelSizes([n, (long)planes * height * width, taps], [outWidth, outHeight, height, width]) || xTaps < 0 || yTaps < 0
            || Gcn("resize_normalize_f32") is not { } f)
        {
            base.ResizeNormalizeKernel(x, coefficients, values, y, planes, channels, height, width, outHeight, outWidth, xTaps, yTaps, bytes);
            return;
        }

        LaunchOutputs(f, n, string.Create(CultureInfo.InvariantCulture, $"{planes}x{height}x{width}->{outHeight}x{outWidth}"), n * Math.Max(1, xTaps + yTaps),
            P(x), P(coefficients), P(values), P(y), U(planes), U(Math.Max(channels, 1)), U(height), U(width), U(outHeight), U(outWidth), U(xTaps), U(yTaps),
            U(bytes ? 1 : 0), U((int)n));
    }

    public override void Interpolate2dKernel(Storage x, Storage y, int planes, int height, int width, int outHeight, int outWidth, InterpolationMode mode,
        bool alignCorners, float scaleHeight, float scaleWidth)
    {
        long n = (long)planes * outHeight * outWidth;
        if (!KernelSizes([n, (long)planes * height * width], [outWidth, outHeight, height, width]) || Gcn("interpolate_f32") is not { } f)
        {
            base.Interpolate2dKernel(x, y, planes, height, width, outHeight, outWidth, mode, alignCorners, scaleHeight, scaleWidth);
            return;
        }

        LaunchOutputs(f, n, string.Create(CultureInfo.InvariantCulture, $"{planes}x{height}x{width}->{outHeight}x{outWidth}"), n, P(x), P(y), U(planes), U(height),
            U(width), U(outHeight), U(outWidth), U(mode == InterpolationMode.Nearest ? 0 : 1), U(alignCorners ? 1 : 0), F(scaleHeight), F(scaleWidth), U((int)n));
    }

    public override void Interpolate2dBackwardKernel(Storage dy, Storage dx, int planes, int height, int width, int outHeight, int outWidth, InterpolationMode mode,
        bool alignCorners, float scaleHeight, float scaleWidth)
    {
        long n = (long)planes * height * width;
        if (!KernelSizes([n, (long)planes * outHeight * outWidth], [outWidth, outHeight, height, width]) || Gcn("interpolate_bwd_f32") is not { } f)
        {
            base.Interpolate2dBackwardKernel(dy, dx, planes, height, width, outHeight, outWidth, mode, alignCorners, scaleHeight, scaleWidth);
            return;
        }

        LaunchOutputs(f, n, string.Create(CultureInfo.InvariantCulture, $"{planes}x{height}x{width}<-{outHeight}x{outWidth}"), n, P(dy), P(dx), U(planes), U(height),
            U(width), U(outHeight), U(outWidth), U(mode == InterpolationMode.Nearest ? 0 : 1), U(alignCorners ? 1 : 0), F(scaleHeight), F(scaleWidth), U((int)n));
    }

    // The adaptive windows' index arithmetic ((o + 1) · size, (i + 1) · outSize) stays within 32 bits (CUDA's).
    private static bool AdaptiveSizes(int planes, int height, int width, int outHeight, int outWidth) =>
        (long)(outHeight + 1) * (height + 1) <= int.MaxValue && (long)(outWidth + 1) * (width + 1) <= int.MaxValue
        && KernelSizes([(long)planes * height * width, (long)planes * outHeight * outWidth], [height, width, outHeight, outWidth]);

    private static string AdaptiveShape(int planes, int height, int width, int outHeight, int outWidth) =>
        string.Create(CultureInfo.InvariantCulture, $"{planes}x{height}x{width}->{outHeight}x{outWidth}");

    public override void AdaptiveAvgPoolKernel(Storage x, Storage y, int planes, int height, int width, int outHeight, int outWidth)
    {
        if (!AdaptiveSizes(planes, height, width, outHeight, outWidth) || Gcn("adaptive_avgpool_f32") is not { } f)
        {
            base.AdaptiveAvgPoolKernel(x, y, planes, height, width, outHeight, outWidth);
            return;
        }

        int n = planes * outHeight * outWidth;
        LaunchOutputs(f, n, AdaptiveShape(planes, height, width, outHeight, outWidth), (long)planes * height * width, P(x), P(y), U(planes), U(height), U(width),
            U(outHeight), U(outWidth), U(n));
    }

    public override void AdaptiveAvgPoolBackwardKernel(Storage dy, Storage dx, int planes, int height, int width, int outHeight, int outWidth)
    {
        if (!AdaptiveSizes(planes, height, width, outHeight, outWidth) || Gcn("adaptive_avgpool_bwd_f32") is not { } f)
        {
            base.AdaptiveAvgPoolBackwardKernel(dy, dx, planes, height, width, outHeight, outWidth);
            return;
        }

        int n = planes * height * width;
        LaunchOutputs(f, n, AdaptiveShape(planes, height, width, outHeight, outWidth), n, P(dy), P(dx), U(planes), U(height), U(width), U(outHeight), U(outWidth), U(n));
    }

    public override void AdaptiveMaxPoolKernel(Storage x, Storage y, Storage argmax, int planes, int height, int width, int outHeight, int outWidth)
    {
        if (!AdaptiveSizes(planes, height, width, outHeight, outWidth) || Gcn("adaptive_maxpool_f32") is not { } f)
        {
            base.AdaptiveMaxPoolKernel(x, y, argmax, planes, height, width, outHeight, outWidth);
            return;
        }

        int n = planes * outHeight * outWidth;
        LaunchOutputs(f, n, AdaptiveShape(planes, height, width, outHeight, outWidth), (long)planes * height * width, P(x), P(y), P(argmax), U(planes), U(height),
            U(width), U(outHeight), U(outWidth), U(n));
    }

    // A gather over dx's elements (each adds the gradients of the windows whose maximum it is, in window order: rule 86).
    public override void AdaptiveMaxPoolBackwardKernel(Storage dy, Storage argmax, Storage dx, int planes, int height, int width, int outHeight, int outWidth)
    {
        if (!AdaptiveSizes(planes, height, width, outHeight, outWidth) || Gcn("adaptive_maxpool_bwd_f32") is not { } f)
        {
            base.AdaptiveMaxPoolBackwardKernel(dy, argmax, dx, planes, height, width, outHeight, outWidth);
            return;
        }

        int n = planes * height * width;
        LaunchOutputs(f, n, AdaptiveShape(planes, height, width, outHeight, outWidth), n, P(dy), P(argmax), P(dx), U(planes), U(height), U(width), U(outHeight),
            U(outWidth), U(n));
    }

    // ---- Convolution ---------------------------------------------------------------------------------------------------

    // The 18 geometry values every depthwise kernel takes after its pointers (CUDA's order), then n where it takes one.
    private static ulong[] ConvolutionValues(in ConvGeometry g, int filters, int groups, int flags, int n, int pointers, ReadOnlySpan<ulong> buffers)
    {
        var values = new ulong[pointers + 18 + (n >= 0 ? 1 : 0)];
        buffers.CopyTo(values);
        int[] geometry = [g.N, g.C, g.H, g.W, g.KH, g.KW, g.SH, g.SW, g.PH, g.PW, g.OH, g.OW, g.DH, g.DW, filters, groups, flags, 1];
        for (int i = 0; i < geometry.Length; i++)
        {
            values[pointers + i] = (uint)geometry[i];
        }

        if (n >= 0)
        {
            values[^1] = (uint)n;
        }

        return values;
    }

    // Whether the depthwise kernels take this convolution: one channel a group (CUDA's ConvolutionShapes.Depthwise), whole
    // filters a channel, every count within the kernels' byte offsets.
    private static bool DepthwiseRuns(in ConvGeometry g, int filters, int groups) =>
        ConvolutionShapes.Depthwise(in g, groups) && filters % groups == 0 && filters > 0
        && KernelSizes([InputCount(g), OutputCount(g, filters), (long)filters * g.KH * g.KW, (long)g.N * g.OH * g.OW], [g.OW, g.OH, g.W, g.H, g.C, g.SH, g.SW, g.KH, g.KW]);

    // The paths of a convolution pass that is not depthwise (CUDA's encoding: the variant in the low four bits, the weight
    // gradient's log2 splits above them).
    private const int ConvComposed = 0, ConvTile = 1, ConvBlocked = 2;

    /// <summary>
    /// Tests and benchmarks: forces the composed path (0), the 16 × 16 implicit product (1) or the 64 × 64 one (2) where it
    /// can run; null: the measured choice (CUDA's formula while nothing can be measured).
    /// </summary>
    internal static int? ConvolutionPath { get; set; }

    /// <summary>Tests: with <see cref="ConvolutionPath"/>, the weight gradient's split count where it is a candidate (else the path's first).</summary>
    internal static int? ConvolutionSplits { get; set; }

    // The rows, columns and batch entries of a pass's implicit product (CUDA's ConvProduct).
    private static (int Rows, int Columns, int Batch) ImplicitProduct(int pass, in ConvGeometry g, int filters, int groups) => pass switch
    {
        ConvolutionShapes.Forward => (filters / groups, g.OH * g.OW, g.N * groups),
        ConvolutionShapes.Input => (g.C / groups, g.H * g.W, g.N * groups),
        _ => (filters / groups, g.PatchSize / groups, groups),
    };

    // Whether the implicit products can index this pass: whole groups, every count within the kernels' byte offsets.
    private static bool ImplicitFits(in ConvGeometry g, int filters, int groups) =>
        groups > 0 && filters > 0 && g.C % groups == 0 && filters % groups == 0 && g.OH > 0 && g.OW > 0
        && KernelSizes([InputCount(g), OutputCount(g, filters), (long)filters * (g.PatchSize / groups), (long)g.N * g.OH * g.OW * 2,
            (long)filters * (g.PatchSize / groups) * 64], [g.OW, g.OH, g.W, g.H, g.KH, g.KW, g.SH, g.SW]);

    // The kernel of an implicit variant for a pass.
    private static string ImplicitName(int pass, int variant) => (pass switch
    {
        ConvolutionShapes.Forward => "conv_fwd",
        ConvolutionShapes.Input => "conv_bwd_input",
        _ => "conv_bwd_weight",
    }) + (variant == ConvBlocked ? "_f32" : "_tile_f32");

    // The weight gradient's split counts (CUDA's ConvWeightSplits: ConvolutionShapes.SplitCounts, and 256 and 1,024 where
    // each split keeps at least 256 positions and the partial sums take no more memory than the input), within the kernels'
    // byte offsets.
    private static int[] ConvWeightSplits(in ConvGeometry g, int filters, int groups)
    {
        long positions = (long)g.N * g.OH * g.OW, count = (long)filters * (g.PatchSize / groups), input = InputCount(g);
        var splits = new List<int>(ConvolutionShapes.SplitCounts(positions));
        foreach (int s in (ReadOnlySpan<int>)[256, 1024])
        {
            if (positions / s >= 256 && s * count <= input)
            {
                splits.Add(s);
            }
        }

        splits.RemoveAll(s => s > 1 && s * count >= GcnKernels.PointwiseMostElements);
        return [.. splits];
    }

    // CUDA's candidates of a pass: the composed path where its patches fit (half the memory the device reports free), the
    // implicit products on both tiles where their grid fits the device's (the weight gradient at each split count).
    private int[] ConvCandidates(int pass, in ConvGeometry g, int filters, int groups)
    {
        var candidates = new List<int>();
        long patches = (long)g.Positions * g.PatchSize;
        if (patches <= int.MaxValue && 4 * patches <= (((IMemoryReport)this).AvailableMemory() ?? long.MaxValue) / 2)
        {
            candidates.Add(ConvComposed);
        }

        if (!ImplicitFits(in g, filters, groups) || DeviceLimits.MaxGridDimZ <= 0)
        {
            return [.. candidates];
        }

        var (rows, columns, _) = ImplicitProduct(pass, in g, filters, groups);
        int[] splits = pass == ConvolutionShapes.Weight && PointwiseKernel("conv_split_reduce_f32", (long)filters * (g.PatchSize / groups)) is not null
            ? ConvWeightSplits(in g, filters, groups) : [1];
        foreach (int variant in (ReadOnlySpan<int>)[ConvTile, ConvBlocked])
        {
            int edge = variant == ConvBlocked ? GcnKernels.ConvBlockedEdge : GcnKernels.ConvTileEdge;
            if (rows <= 0 || columns <= 0 || (rows + edge - 1) / edge > DeviceLimits.MaxGridDimY || (columns + edge - 1) / edge > DeviceLimits.MaxGridDimX
                || Gcn(ImplicitName(pass, variant)) is null)
            {
                continue;
            }

            foreach (int s in splits)
            {
                candidates.Add(variant | System.Numerics.BitOperations.Log2((uint)s) << 4);
            }
        }

        return [.. candidates];
    }

    // CUDA's formula among the candidates: the composed path for a product of fewer rows than a quarter of the small tile
    // (most of every tile's work-items idle), else the 64 × 64 tiles when they would be at least half full, with the most
    // splits that leave each at least 4,096 positions.
    private static int ConvFormula(int pass, in ConvGeometry g, int filters, int groups, int[] candidates)
    {
        var (rows, columns, _) = ImplicitProduct(pass, in g, filters, groups);
        if (rows * 4 < GcnKernels.ConvTileEdge && Array.IndexOf(candidates, ConvComposed) >= 0)
        {
            return ConvComposed;
        }

        int most = 1;
        long positions = (long)g.N * g.OH * g.OW;
        foreach (int s in pass == ConvolutionShapes.Weight ? ConvWeightSplits(in g, filters, groups) : [1])
        {
            if (s == 1 || positions / s >= 4096)
            {
                most = Math.Max(most, s);
            }
        }

        int splitBits = pass == ConvolutionShapes.Weight ? System.Numerics.BitOperations.Log2((uint)most) << 4 : 0;
        int variant = rows >= GcnKernels.ConvBlockedEdge / 2 && columns >= GcnKernels.ConvBlockedEdge / 2 ? ConvBlocked : ConvTile;
        foreach (int v in (ReadOnlySpan<int>)[variant | splitBits, variant, ConvTile | splitBits, ConvTile, ConvComposed])
        {
            if (Array.IndexOf(candidates, v) >= 0)
            {
                return v;
            }
        }

        return candidates.Length > 0 ? candidates[0] : ConvComposed;
    }

    // The path of one pass for this shape: forced, measured once and kept in HIP's tuning table (the candidates write
    // `output`, or zeroed scratch where the pass adds to its output), or the formula's.
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
            return candidates.Length == 1 ? candidates[0] : fallback;
        }

        var key = new TuneKey(TuneOp.Convolution, ConvolutionShapes.Variant(pass, activation, groups), shape.A, shape.B, shape.C, shape.D, shape.E, shape.F);
        if (!Autotune || Tuning.TryKnown(key, candidates, out _) || !CanMeasure)
        {
            return Tune(key, candidates, fallback, _ => { });                    // the known choice, else the formula
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

    // Launches an implicit product: tiles along x (columns) and y (rows), the batch entries along z (the kernel loops over
    // those past the device's z), labelled for --profile with its shape and its operations.
    private void LaunchImplicit(int variant, int pass, in ConvGeometry g, int filters, int groups, int batch, params ReadOnlySpan<ulong> values)
    {
        var f = Gcn(ImplicitName(pass, variant)) ?? throw new InvalidOperationException($"{ImplicitName(pass, variant)} is not loaded.");
        var (rows, columns, _) = ImplicitProduct(pass, in g, filters, groups);
        int edge = variant == ConvBlocked ? GcnKernels.ConvBlockedEdge : GcnKernels.ConvTileEdge;
        if (Profiling)
        {
            long terms = pass switch
            {
                ConvolutionShapes.Forward => g.PatchSize / groups,
                ConvolutionShapes.Input => (long)(filters / groups) * g.KH * g.KW,
                _ => (long)g.N * g.OH * g.OW,
            };
            LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"{f.Name} {ConvolutionShape(g)}"),
                ConvolutionCost(f.Name).Flops * rows * (double)columns * terms * (pass == ConvolutionShapes.Weight ? groups : batch));
        }

        LaunchProduct(f, (columns + edge - 1) / edge, (rows + edge - 1) / edge, Math.Clamp(batch, 1, DeviceLimits.MaxGridDimZ), values);
    }

    // The 18 geometry values after `buffers`, with flags and splits.
    private static ulong[] ImplicitValues(in ConvGeometry g, int filters, int groups, int flags, int splits, ReadOnlySpan<ulong> buffers)
    {
        var values = ConvolutionValues(g, filters, groups, flags, -1, buffers.Length, buffers);
        values[^1] = (uint)splits;
        return values;
    }

    public override void ConvolutionKernel(Storage x, Storage weight, Storage? bias, Storage y, in ConvGeometry g, int filters, int groups, ConvActivation activation)
    {
        if (!DepthwiseRuns(in g, filters, groups) || Gcn("conv_dw_f32") is not { } f)
        {
            var geometry = g;
            int choice = ConvChoice(ConvolutionShapes.Forward, in g, filters, groups, activation, y.Length,
                (c, output) => RunConvolution(c, x, weight, bias, output, geometry, filters, groups, activation), writesOutput: true, y);
            RunConvolution(choice, x, weight, bias, y, g, filters, groups, activation);
            return;
        }

        int flags = (bias is null ? 0 : 1) | (int)activation << 1;
        long n = OutputCount(g, filters);
        LaunchOutputs(f, n, ConvolutionShape(g), n * g.KH * g.KW, ConvolutionValues(g, filters, groups, flags, (int)n, 4, [P(x), P(weight), bias is null ? P(weight) : P(bias), P(y)]));
    }

    private void RunConvolution(int choice, Storage x, Storage weight, Storage? bias, Storage y, ConvGeometry g, int filters, int groups, ConvActivation activation)
    {
        if ((choice & 15) == ConvComposed)
        {
            base.ConvolutionKernel(x, weight, bias, y, in g, filters, groups, activation);       // the composed path, on HIP's kernels
            return;
        }

        int flags = (bias is null ? 0 : 1) | (int)activation << 1;
        LaunchImplicit(choice & 15, ConvolutionShapes.Forward, in g, filters, groups, g.N * groups,
            ImplicitValues(g, filters, groups, flags, 1, [P(x), P(weight), bias is null ? P(weight) : P(bias), P(y)]));
    }

    public override void ConvolutionBackwardInputKernel(Storage dy, Storage weight, Storage dx, in ConvGeometry g, int filters, int groups)
    {
        if (!DepthwiseRuns(in g, filters, groups) || Gcn("conv_dw_bwd_input_f32") is not { } f)
        {
            var geometry = g;
            int choice = ConvChoice(ConvolutionShapes.Input, in g, filters, groups, ConvActivation.None, dx.Length,
                (c, output) => RunConvolutionInput(c, dy, weight, output, geometry, filters, groups), writesOutput: false, dx);
            RunConvolutionInput(choice, dy, weight, dx, g, filters, groups);
            return;
        }

        long n = InputCount(g);
        LaunchOutputs(f, n, ConvolutionShape(g), n * (filters / g.C) * g.KH * g.KW, ConvolutionValues(g, filters, groups, 0, (int)n, 3, [P(dy), P(weight), P(dx)]));
    }

    private void RunConvolutionInput(int choice, Storage dy, Storage weight, Storage dx, ConvGeometry g, int filters, int groups)
    {
        if ((choice & 15) == ConvComposed)
        {
            base.ConvolutionBackwardInputKernel(dy, weight, dx, in g, filters, groups);
            return;
        }

        LaunchImplicit(choice & 15, ConvolutionShapes.Input, in g, filters, groups, g.N * groups, ImplicitValues(g, filters, groups, 0, 1, [P(dy), P(weight), P(dx)]));
    }

    public override void ConvolutionBackwardWeightKernel(Storage x, Storage dy, Storage dweight, in ConvGeometry g, int filters, int groups)
    {
        if (!DepthwiseRuns(in g, filters, groups) || Gcn("conv_dw_bwd_weight_f32") is not { } f)
        {
            var geometry = g;
            int choice = ConvChoice(ConvolutionShapes.Weight, in g, filters, groups, ConvActivation.None, dweight.Length,
                (c, output) => RunConvolutionWeight(c, x, dy, output, geometry, filters, groups), writesOutput: false, dweight);
            RunConvolutionWeight(choice, x, dy, dweight, g, filters, groups);
            return;
        }

        int weights = filters * g.KH * g.KW;
        if (Profiling)
        {
            LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"{f.Name} {ConvolutionShape(g)}"), Cost(f.Name).Flops * weights * ((double)g.N * g.OH * g.OW));
        }

        LaunchKernel(f, GridFor(weights), 1, ConvolutionValues(g, filters, groups, 0, -1, 3, [P(x), P(dy), P(dweight)]));
    }

    // The weight gradient on an implicit product: one split adds to dweight; several write their partial sums to a
    // temporary [splits, filters, patch] that conv_split_reduce_f32 adds to dweight in split order (CUDA's).
    private void RunConvolutionWeight(int choice, Storage x, Storage dy, Storage dweight, ConvGeometry g, int filters, int groups)
    {
        int variant = choice & 15;
        if (variant == ConvComposed)
        {
            base.ConvolutionBackwardWeightKernel(x, dy, dweight, in g, filters, groups);
            return;
        }

        int splits = 1 << ((choice >> 4) & 15), count = filters * (g.PatchSize / groups);
        if (splits == 1)
        {
            LaunchImplicit(variant, ConvolutionShapes.Weight, in g, filters, groups, groups, ImplicitValues(g, filters, groups, 0, 1, [P(x), P(dy), P(dweight), P(dweight)]));
            return;
        }

        var part = Allocate(splits * count, zeroed: false);
        try
        {
            LaunchImplicit(variant, ConvolutionShapes.Weight, in g, filters, groups, groups * splits, ImplicitValues(g, filters, groups, 0, splits, [P(x), P(dy), P(dweight), P(part)]));
            var reduce = PointwiseKernel("conv_split_reduce_f32", count)!;
            if (Profiling)
            {
                LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"{reduce.Name} {count}x{splits}"), (double)count * splits);
            }

            LaunchKernel(reduce, PointwiseBlocks(count), 1, P(part), P(dweight), U(count), U(splits), U(count));
        }
        finally
        {
            part.Release();                                                    // reused in stream order
        }
    }
}
