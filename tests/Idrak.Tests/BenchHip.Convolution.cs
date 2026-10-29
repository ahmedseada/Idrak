// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Gpu.Hip;

// --bench-hip convolution (plan 35, G5c): HIP's convolution, pooling, resampling, CTC and recurrent operations on their
// GCN kernels, each timed cold (its first call) apart from warm (BenchOperation), with GB/s and GFLOP/s from the tables of
// GcnKernels.Convolution.cs, .Ctc.cs and .Recurrent.cs (HipBackend.Cost), the section ending with the host fallbacks it
// took (none expected). Sizes follow the large buffer (from the memory the device reports), never a card's: a batch of
// 64-channel images whose activations take about a quarter of it, a CTC batch of 64 lines of 200 steps, recurrent steps
// of 64 rows of 512 units.
internal abstract partial class DesktopTests
{
    private static void BenchHipConvolution(HipBench b)
    {
        if (!BenchKernelSet(b))
        {
            return;
        }

        var hip = b.Hip;
        long before = Idrak.Abstraction.Operations.Kernels.HostCalls(hip);
        const int Channels = 64, Side = 56;
        int images = (int)Math.Clamp(b.LargeElements / (4L * Channels * Side * Side), 1, 64);
        var g = new ConvGeometry(images, Channels, Side, Side, 3, 3, 1, 1, 1, 1);
        var pool = new ConvGeometry(images, Channels, Side, Side, 2, 2, 2, 2, 0, 0);
        int input = images * Channels * Side * Side, pooled = images * Channels * pool.OH * pool.OW, cols = g.Positions * g.PatchSize;
        string shape = string.Create(CultureInfo.InvariantCulture, $"{images} x {Channels} x {Side} x {Side}");
        using (var on = new BenchStorages(hip, new Random(11)))
        {
            Storage x = on.Random(input), y = on.Random(input), dx = on.Random(input), py = on.Random(pooled), argmax = on.Random(pooled);
            if ((long)cols < Idrak.Gpu.Hip.GcnKernels.PointwiseMostElements)
            {
                Storage patches = on.Random(cols);
                BenchOperation(b, $"unfolded 3x3 patches, {shape}", cols, HipBackend.Cost("im2col_f32"), () => hip.Im2Col(x, patches, g));
                BenchOperation(b, $"folded 3x3 patches, {shape}", (long)input * 9, HipBackend.Cost("col2im_f32"), () => hip.Col2Im(patches, dx, g));
                Storage w = on.Random(Channels * Channels * 9, -0.05f, 0.05f), bias = on.Random(Channels);
                BenchOperation(b, $"3x3 convolution (composed: patches and products), {shape}", (long)input * Channels * 9, (2, 0),
                    () => hip.Convolution(x, w, bias, y, g, Channels, 1, ConvActivation.Relu));
            }

            Storage dw = on.Random(Channels * 9, -0.1f, 0.1f), dwBias = on.Random(Channels), dwGrad = on.Random(Channels * 9);
            BenchOperation(b, $"3x3 depthwise convolution, {shape}", (long)input * 9, HipBackend.Cost("conv_dw_f32"),
                () => hip.Convolution(x, dw, dwBias, y, g, Channels, Channels, ConvActivation.Silu));
            BenchOperation(b, $"3x3 depthwise input gradient, {shape}", (long)input * 9, HipBackend.Cost("conv_dw_bwd_input_f32"),
                () => hip.ConvolutionBackwardInput(y, dw, dx, g, Channels, Channels));
            BenchOperation(b, $"3x3 depthwise weight gradient, {shape}", (long)input * 9, HipBackend.Cost("conv_dw_bwd_weight_f32"),
                () => hip.ConvolutionBackwardWeight(x, y, dwGrad, g, Channels, Channels));
            BenchOperation(b, $"2x2 max pooling, {shape}", (long)pooled * 4, HipBackend.Cost("maxpool_f32"), () => hip.MaxPool(x, py, argmax, pool));
            BenchOperation(b, $"2x2 max pooling gradient, {shape}", input, HipBackend.Cost("maxpool_bwd_gather_f32"), () => hip.MaxPoolBackward(py, argmax, dx, pool));
            BenchOperation(b, $"2x2 average pooling, {shape}", (long)pooled * 4, HipBackend.Cost("avgpool_f32"), () => hip.AvgPool(x, py, pool, true, 0, 0));
            BenchOperation(b, $"2x2 average pooling gradient, {shape}", (long)input * 4, HipBackend.Cost("avgpool_bwd_f32"), () => hip.AvgPoolBackward(py, dx, pool, true, 0, 0));
            int planes = images * Channels;
            Storage small = on.Random(planes * 7 * 7);
            BenchOperation(b, $"adaptive average pooling to 7x7, {shape}", input, HipBackend.Cost("adaptive_avgpool_f32"), () => hip.AdaptiveAvgPool(x, small, planes, Side, Side, 7, 7));
            BenchOperation(b, $"adaptive average pooling gradient, {shape}", input, HipBackend.Cost("adaptive_avgpool_bwd_f32"),
                () => hip.AdaptiveAvgPoolBackward(small, dx, planes, Side, Side, 7, 7));
            Storage half = on.Random(planes * (Side / 2) * (Side / 2));
            BenchOperation(b, $"bilinear interpolation to twice the size, {planes} planes of {Side / 2} x {Side / 2}", input, HipBackend.Cost("interpolate_f32"),
                () => hip.Interpolate2d(half, y, planes, Side / 2, Side / 2, Side, Side, InterpolationMode.Bilinear, false, 0.5f, 0.5f));
            BenchOperation(b, $"bilinear interpolation gradient, {planes} planes of {Side / 2} x {Side / 2}", (long)planes * (Side / 2) * (Side / 2), HipBackend.Cost("interpolate_bwd_f32"),
                () => hip.Interpolate2dBackward(y, half, planes, Side / 2, Side / 2, Side, Side, InterpolationMode.Bilinear, false, 0.5f, 0.5f));

            // CTC over a batch of lines (the rows in LDS), and the recurrent steps of a layer of 512 units.
            const int Lines = 64, Steps = 200, Classes = 96, Labels = 40;
            var random = new Random(12);
            int[] inputLengths = [.. Enumerable.Repeat(Steps, Lines)], targetLengths = [.. Enumerable.Repeat(Labels, Lines)];
            int[] targetOffsets = [.. Enumerable.Range(0, Lines).Select(n => n * Labels)];
            Storage logProbs = on.Random(Steps * Lines * Classes, -6f, -2f), losses = on.Random(Lines), lossGrads = on.Random(Lines, 0.5f, 1f), dLogProbs = on.Random(Steps * Lines * Classes);
            Storage targets = on.Of([.. Enumerable.Range(0, Lines * Labels).Select(_ => (float)random.Next(1, Classes))]);
            long ctcUnits = (long)Lines * Steps * (2 * Labels + 1);
            BenchOperation(b, $"CTC loss, {Lines} lines of {Steps} steps, {Labels} labels", ctcUnits, HipBackend.Cost("ctc_loss_f32"),
                () => hip.CtcLoss(logProbs, targets, losses, inputLengths, targetLengths, targetOffsets, Steps, Lines, Classes, 0, false, true));
            BenchOperation(b, $"CTC gradient, {Lines} lines of {Steps} steps, {Labels} labels", ctcUnits, HipBackend.Cost("ctc_loss_bwd_f32"),
                () => hip.CtcLossBackward(logProbs, targets, lossGrads, dLogProbs, inputLengths, targetLengths, targetOffsets, Steps, Lines, Classes, 0, false, true));
            const int Rows = 64, Units = 512, Time = 4;
            Storage projected = on.Random(Rows * Time * 4 * Units), weights = on.Random(Units * 4 * Units, -0.05f, 0.05f), cell = on.Random(Rows * Units);
            Storage output = on.Random(Rows * Time * Units), gates = on.Random(Rows * Time * 4 * Units, 0.05f, 0.95f), cells = on.Random(Rows * Time * Units);
            Storage dCell = on.Random(Rows * Units), dGates = on.Random(Rows * Time * 4 * Units);
            BenchOperation(b, $"LSTM step, {Rows} rows of {Units} units (the product inside)", (long)Rows * Units, HipBackend.RecurrentCost("lstm_step_f32", Units),
                () => hip.LstmStep(projected, weights, cell, output, gates, cells, 1, 0, Time, Rows, Units));
            BenchOperation(b, $"LSTM step gradient, {Rows} rows of {Units} units (the product inside)", (long)Rows * Units, HipBackend.RecurrentCost("lstm_step_bwd_f32", Units),
                () => hip.LstmStepBackward(gates, cells, output, weights, dCell, dGates, 1, 2, 0, Time, Rows, Units));
        }

        long host = Idrak.Abstraction.Operations.Kernels.HostCalls(hip) - before;
        b.Row("  (host fallbacks during the section)", host == 0 ? "none: every operation ran on its kernel" : $"{host} (SHOULD BE NONE)");
    }
}
