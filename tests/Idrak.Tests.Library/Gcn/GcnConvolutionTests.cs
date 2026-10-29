// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using Idrak.Abstraction.Devices;
using Idrak.Gpu.Abstractions;
using Idrak.Gpu.Hip;
using Idrak.Tests.Gcn;

// HIP's convolution, pooling, resampling, CTC and recurrent kernels in GCN (GcnKernels.Convolution, .Ctc, .Recurrent) and
// the packed gated activation's gradient, each run by the GCN interpreter and proven against the CPU (plan 35, G5c). Every
// kernel is assembled for gfx90c and gfx906 and run over small shapes that are multiples of nothing (strides, padding
// above and below, dilation, 1-D images, upsampling and downsampling, odd hidden sizes, a hidden size past one block of
// units), compared to the CPU backend within the device-against-CPU tolerance (OnBoth's, 1e-3 of max(1, |expected|)) or
// bit for bit where the arithmetic is the CPU's; run again on another grid and another wave order to give the same bits
// (every sum in a fixed order, no atomics: the gradients gather); and the interpreter reports no missing wait, race or
// access out of range (a report fails the run). Each test prints its largest errors.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] GcnConvolutionGroup =
    [
        ("gcn kernels: convolution: every convolution, pooling, resampling, CTC and recurrent kernel assembles for gfx90c and gfx906 with its block of 256 declared, within its register budget and LDS, its printed text reading back to its bytes, with its profile cost; GcnKernels.All lists each once", GcnConvolutionAssembled),
        ("gcn kernels: convolution: im2col_f32 and col2im_f32 equal the CPU's patches and their fold bit for bit (strides, padding, dilation, a 1-D image), the same bits on two grids and wave orders", GcnConvolutionPatches),
        ("gcn kernels: convolution: maxpool_f32 (values and arg-max), maxpool_bwd_gather_f32, avgpool_f32 and avgpool_bwd_f32 (padding counted or not, padding below and right) equal the CPU's bit for bit, the same bits on two grids and wave orders", GcnConvolutionPooling),
        ("gcn kernels: convolution: the adaptive average and maximum pooling and their gradients equal the CPU's bit for bit (downsampling, upsampling, the same size), the same bits on two grids and wave orders", GcnConvolutionAdaptive),
        ("gcn kernels: convolution: interpolate_f32 (nearest with its exact cases, bilinear with and without aligned corners) equals the CPU's bit for bit, interpolate_bwd_f32 within tolerance, the same bits on two grids and wave orders", GcnConvolutionInterpolate),
        ("gcn kernels: convolution: resize_normalize_f32 equals the CPU's resampling and normalization bit for bit (floats and Pillow's 8-bit passes, either pass skipped), the same bits on two grids and wave orders", GcnConvolutionResize),
        ("gcn kernels: convolution: conv_dw_f32 (a bias, every activation), conv_dw_bwd_input_f32 and conv_dw_bwd_weight_f32 equal the CPU's depthwise convolution and its gradients within tolerance (strides, padding, dilation, channel multipliers), the same bits on two grids and wave orders", GcnConvolutionDepthwise),
        ("gcn kernels: convolution: ctc_loss_f32 and ctc_loss_bwd_f32 (rows in LDS) and their _global forms (rows in scratch) equal the CPU's loss and gradient within tolerance (repeated labels, an empty target, a target too long for its input, zeroinf, batch-first or not), the same bits on two grids and wave orders", GcnConvolutionCtc),
        ("gcn kernels: convolution: lstm_cell_f32, gru_cell_f32 and their gradients equal the CPU's within tolerance (with and without the saved gates, a previous step, dOutput, the candidate's bias), the same bits on two grids and wave orders", GcnConvolutionCells),
        ("gcn kernels: convolution: lstm_step_f32, gru_step_f32 and their gradients equal the CPU's product and cell within tolerance (a hidden size past one block of units, with and without a previous or later step), the same bits on two wave orders", GcnConvolutionSteps),
        ("gcn kernels: convolution: gated_act_bwd_bf16_f32 equals the CPU's packed gated activation gradient within tolerance (every kind, added and written, an odd count), the same bits on two grids and wave orders", GcnConvolutionPackedGated),
    ];

    private static ulong Cu(int value) => (uint)value;

    private static ulong Cf(float value) => BitConverter.SingleToUInt32Bits(value);

    private static IEnumerable<GcnKernelSource> G5cKernels => [.. GcnKernels.Convolution, .. GcnKernels.Ctc, .. GcnKernels.Recurrent];

    internal static void GcnConvolutionAssembled(Device device)
    {
        _ = device;
        var names = G5cKernels.Select(k => k.Name).ToArray();
        Check(GcnKernels.Convolution.Count == 16 && GcnKernels.Ctc.Count == 4 && GcnKernels.Recurrent.Count == 8 && names.Distinct().Count() == names.Length,
            $"16 convolution, 4 CTC and 8 recurrent kernels, each once ({string.Join(", ", names)})");
        foreach (string name in names)
        {
            Check(GcnKernels.All.Count(k => k.Name == name) == 1, $"GcnKernels.All lists {name} once");
            Check(Encoding.UTF8.GetString(G5cKernels.Single(k => k.Name == name).Text).StartsWith($".kernel {name}\n", StringComparison.Ordinal), $"{name}'s text starts with its .kernel line");
            Check(HipBackend.Cost(name).Bytes > 0, $"{name} has its profile cost (HipBackend.ConvolutionCost, CtcCost, RecurrentCost)");
            foreach (var isa in TrainingIsas)
            {
                var (code, _) = TrainingKernel(name, isa);
                Check(code is { RequiredWorkgroupSize: (GcnKernels.RowThreads, 1, 1), MaxFlatWorkgroupSize: GcnKernels.RowThreads, WaveSize: 64 },
                    $"{name} ({isa}): declares the block of {GcnKernels.RowThreads} (required {code.RequiredWorkgroupSize}, max flat {code.MaxFlatWorkgroupSize})");
                Check(code.LdsBytes <= 65536, $"{name} ({isa}): {code.LdsBytes} bytes of LDS");
                if (isa == TrainingIsas[0])
                {
                    Console.WriteLine($"    {name}: {code.Code.Length} bytes of code, {code.Vgprs} VGPRs, {code.Sgprs} SGPRs, {code.LdsBytes} bytes of LDS");
                }
            }
        }

        Check(HipBackend.Cost("gated_act_bwd_bf16_f32").Bytes > 0 && GcnKernels.All.Count(k => k.Name == "gated_act_bwd_bf16_f32") == 1,
            "gated_act_bwd_bf16_f32 is listed once with its cost");
    }

    // A per-output kernel (one work-item an output, grid-stride) over n outputs on both processors: one block, then enough
    // with idle ones.
    private static void G5cCase(string what, string kernel, float[][] inputs, int[] outputs, Action<Backend, Storage[]> cpu, Func<ulong[], ulong[]> args, long n,
        Dictionary<string, float> worst, bool exact = false, string? label = null)
    {
        foreach (var isa in TrainingIsas)
        {
            int blocks = (int)Math.Max(1, (n + 255) / 256);
            TrainingCase($"{what} ({isa})", inputs, outputs, cpu, (m, a, grid, seed) => TrainingDispatch(kernel, isa, m, (grid, 1), seed, args(a)), (1, blocks + 2), worst,
                label ?? kernel, exact);
        }
    }

    private static string Describe(in ConvGeometry g) =>
        $"{g.N}x{g.C}x{g.H}x{g.W}, window {g.KH}x{g.KW}, stride {g.SH}x{g.SW}, padding {g.PH}x{g.PW} (below {g.PadBottom}, right {g.PadRight}), dilation {g.DH}x{g.DW} -> {g.OH}x{g.OW}";

    private static ConvGeometry[] PatchGeometries =>
    [
        new(2, 3, 7, 6, 3, 2, 2, 1, 1, 0) { DW = 2 },
        new(1, 2, 5, 9, 2, 3, 1, 2, 0, 1) { DH = 2 },
        new(2, 3, 1, 11, 1, 4, 1, 1, 0, 2),
    ];

    private static ConvGeometry[] PoolGeometries =>
    [
        new(2, 3, 7, 6, 3, 2, 2, 1, 1, 0),
        new(1, 2, 8, 9, 2, 3, 2, 2, 0, 1) { PadBottom = 1, PadRight = 2 },
        new(2, 1, 1, 10, 1, 3, 1, 2, 0, 1),
    ];

    internal static void GcnConvolutionPatches(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(61);
        foreach (var geometry in PatchGeometries)
        {
            var g = geometry;
            int input = g.N * g.C * g.H * g.W, cols = g.Positions * g.PatchSize;
            G5cCase($"im2col_f32, {Describe(g)}", "im2col_f32", [TrainingRandom(random, input, -2, 2), new float[cols]], [1], (b, s) => b.Im2Col(s[0], s[1], g),
                a => [a[0], a[1], Cu(g.C), Cu(g.H), Cu(g.W), Cu(g.KW), Cu(g.SH), Cu(g.SW), Cu(g.PH), Cu(g.PW), Cu(g.OW), Cu(g.PatchSize), Cu(g.KH * g.KW), Cu(g.OH * g.OW),
                    Cu(g.DH), Cu(g.DW), Cu(cols)], cols, worst, exact: true);
            G5cCase($"col2im_f32, {Describe(g)}", "col2im_f32", [TrainingRandom(random, cols, -1, 1), TrainingRandom(random, input, -1, 1)], [1],
                (b, s) => b.Col2Im(s[0], s[1], g),
                a => [a[0], a[1], Cu(g.C), Cu(g.H), Cu(g.W), Cu(g.KH), Cu(g.KW), Cu(g.SH), Cu(g.SW), Cu(g.PH), Cu(g.PW), Cu(g.OH), Cu(g.OW), Cu(g.PatchSize), Cu(g.DH),
                    Cu(g.DW), Cu(input)], input, worst, exact: true);
        }

        TrainingReport(worst);
    }

    internal static void GcnConvolutionPooling(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(62);
        foreach (var geometry in PoolGeometries)
        {
            var g = geometry;
            int input = g.N * g.C * g.H * g.W, outputs = g.N * g.C * g.OH * g.OW;
            float[] x = TrainingRandom(random, input, -2, 2);
            for (int i = 0; i + 3 < input; i += 7)
            {
                x[i + 3] = x[i];                                                   // ties: the first in row order wins
            }

            G5cCase($"maxpool_f32, {Describe(g)}", "maxpool_f32", [x, new float[outputs], new float[outputs]], [1, 2], (b, s) => b.MaxPool(s[0], s[1], s[2], g),
                a => [a[0], a[1], a[2], Cu(g.H), Cu(g.W), Cu(g.KH), Cu(g.KW), Cu(g.SH), Cu(g.SW), Cu(g.PH), Cu(g.PW), Cu(g.OW), Cu(g.OH * g.OW), Cu(outputs)], outputs, worst,
                exact: true);
            float[] argmax = RowsOnCpu([x, new float[outputs], new float[outputs]], (b, s) => b.MaxPool(s[0], s[1], s[2], g))[2];
            G5cCase($"maxpool_bwd_gather_f32, {Describe(g)}", "maxpool_bwd_gather_f32", [TrainingRandom(random, outputs, -1, 1), argmax, TrainingRandom(random, input, -1, 1)],
                [2], (b, s) => b.MaxPoolBackward(s[0], s[1], s[2], g),
                a => [a[0], a[1], a[2], Cu(g.H), Cu(g.W), Cu(g.KH), Cu(g.KW), Cu(g.SH), Cu(g.SW), Cu(g.PH), Cu(g.PW), Cu(g.OH), Cu(g.OW), Cu(input)], input, worst, exact: true);
            foreach (bool countPad in new[] { true, false })
            {
                ulong[] Pool(ulong[] a, int n) =>
                    [a[0], a[1], Cu(g.H), Cu(g.W), Cu(g.KH), Cu(g.KW), Cu(g.SH), Cu(g.SW), Cu(g.PH), Cu(g.PW), Cu(g.OH), Cu(g.OW), Cu(countPad ? 1 : 0), Cu(g.PadBottom),
                        Cu(g.PadRight), Cu(n)];
                G5cCase($"avgpool_f32, {Describe(g)}, padding counted {countPad}", "avgpool_f32", [x, new float[outputs]], [1],
                    (b, s) => b.AvgPool(s[0], s[1], g, countPad, g.PadBottom, g.PadRight), a => Pool(a, outputs), outputs, worst, exact: true);
                G5cCase($"avgpool_bwd_f32, {Describe(g)}, padding counted {countPad}", "avgpool_bwd_f32", [TrainingRandom(random, outputs, -1, 1), TrainingRandom(random, input, -1, 1)],
                    [1], (b, s) => b.AvgPoolBackward(s[0], s[1], g, countPad, g.PadBottom, g.PadRight), a => Pool(a, input), input, worst, exact: true);
            }
        }

        TrainingReport(worst);
    }

    internal static void GcnConvolutionAdaptive(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(63);
        foreach (var (planes, height, width, outHeight, outWidth) in new[] { (3, 7, 5, 3, 4), (2, 4, 4, 6, 5), (2, 5, 9, 5, 9), (1, 1, 13, 1, 4) })
        {
            int input = planes * height * width, outputs = planes * outHeight * outWidth;
            string shape = $"{planes} planes of {height}x{width} -> {outHeight}x{outWidth}";
            ulong[] Args(ulong[] a, int count, int pointers) => [.. a.Take(pointers), Cu(planes), Cu(height), Cu(width), Cu(outHeight), Cu(outWidth), Cu(count)];
            float[] x = TrainingRandom(random, input, -2, 2);
            G5cCase($"adaptive_avgpool_f32, {shape}", "adaptive_avgpool_f32", [x, new float[outputs]], [1], (b, s) => b.AdaptiveAvgPool(s[0], s[1], planes, height, width, outHeight, outWidth),
                a => Args(a, outputs, 2), outputs, worst, exact: true);
            G5cCase($"adaptive_avgpool_bwd_f32, {shape}", "adaptive_avgpool_bwd_f32", [TrainingRandom(random, outputs, -1, 1), TrainingRandom(random, input, -1, 1)], [1],
                (b, s) => b.AdaptiveAvgPoolBackward(s[0], s[1], planes, height, width, outHeight, outWidth), a => Args(a, input, 2), input, worst, exact: true);
            G5cCase($"adaptive_maxpool_f32, {shape}", "adaptive_maxpool_f32", [x, new float[outputs], new float[outputs]], [1, 2],
                (b, s) => b.AdaptiveMaxPool(s[0], s[1], s[2], planes, height, width, outHeight, outWidth), a => Args(a, outputs, 3), outputs, worst, exact: true);
            float[] argmax = RowsOnCpu([x, new float[outputs], new float[outputs]], (b, s) => b.AdaptiveMaxPool(s[0], s[1], s[2], planes, height, width, outHeight, outWidth))[2];
            G5cCase($"adaptive_maxpool_bwd_f32, {shape}", "adaptive_maxpool_bwd_f32", [TrainingRandom(random, outputs, -1, 1), argmax, TrainingRandom(random, input, -1, 1)], [2],
                (b, s) => b.AdaptiveMaxPoolBackward(s[0], s[1], s[2], planes, height, width, outHeight, outWidth), a => Args(a, input, 3), input, worst, exact: true);
        }

        TrainingReport(worst);
    }

    internal static void GcnConvolutionInterpolate(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(64);
        foreach (var (planes, height, width, outHeight, outWidth) in new[] { (2, 5, 7, 9, 4), (3, 4, 6, 8, 12), (2, 6, 6, 6, 6), (1, 1, 9, 1, 13) })
        {
            int input = planes * height * width, outputs = planes * outHeight * outWidth;
            foreach (var (mode, align) in new[] { (InterpolationMode.Nearest, false), (InterpolationMode.Bilinear, false), (InterpolationMode.Bilinear, true) })
            {
                float scaleHeight = align ? (outHeight > 1 ? (height - 1f) / (outHeight - 1f) : 0f) : (float)height / outHeight;
                float scaleWidth = align ? (outWidth > 1 ? (width - 1f) / (outWidth - 1f) : 0f) : (float)width / outWidth;
                string shape = $"{planes} planes of {height}x{width} -> {outHeight}x{outWidth}, {mode}{(align ? ", aligned corners" : "")}";
                ulong[] Args(ulong[] a, int count) =>
                    [a[0], a[1], Cu(planes), Cu(height), Cu(width), Cu(outHeight), Cu(outWidth), Cu(mode == InterpolationMode.Nearest ? 0 : 1), Cu(align ? 1 : 0), Cf(scaleHeight),
                        Cf(scaleWidth), Cu(count)];
                G5cCase($"interpolate_f32, {shape}", "interpolate_f32", [TrainingRandom(random, input, -2, 2), new float[outputs]], [1],
                    (b, s) => b.Interpolate2d(s[0], s[1], planes, height, width, outHeight, outWidth, mode, align, scaleHeight, scaleWidth), a => Args(a, outputs), outputs, worst,
                    exact: true);
                G5cCase($"interpolate_bwd_f32, {shape}", "interpolate_bwd_f32", [TrainingRandom(random, outputs, -1, 1), TrainingRandom(random, input, -1, 1)], [1],
                    (b, s) => b.Interpolate2dBackward(s[0], s[1], planes, height, width, outHeight, outWidth, mode, align, scaleHeight, scaleWidth), a => Args(a, input), input,
                    worst);
            }
        }

        TrainingReport(worst);
    }

    // The coefficients of a resampling (Backend.ResizeNormalize): per output column, then per output row, the first input
    // position and the tap count (int bits) and `taps` weights (floats, or int bits with 22 fractional bits for bytes).
    private static float[] ResizeCoefficients(Random random, int size, int outSize, int taps, bool bytes)
    {
        var values = new float[outSize * (taps + 2)];
        for (int o = 0; o < outSize; o++)
        {
            int count = 1 + random.Next(taps), first = random.Next(size - count + 1), at = o * (taps + 2);
            values[at] = BitConverter.Int32BitsToSingle(first);
            values[at + 1] = BitConverter.Int32BitsToSingle(count);
            for (int t = 0; t < taps; t++)
            {
                values[at + 2 + t] = bytes ? BitConverter.Int32BitsToSingle((1 << 22) / count + random.Next(-20000, 20000)) : random.NextSingle() / count;
            }
        }

        return values;
    }

    internal static void GcnConvolutionResize(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(65);
        const int Planes = 4, Channels = 2, Height = 5, Width = 7;
        foreach (bool bytes in new[] { false, true })
        {
            foreach (var (outHeight, outWidth, xTaps, yTaps) in new[] { (3, 4, 3, 2), (6, 9, 2, 3), (5, 4, 3, 0), (3, 7, 0, 2) })
            {
                float[] coefficients = [.. (xTaps > 0 ? ResizeCoefficients(random, Width, outWidth, xTaps, bytes) : []),
                    .. (yTaps > 0 ? ResizeCoefficients(random, Height, outHeight, yTaps, bytes) : [])];
                int pixels = Planes * Height * Width, outputs = Planes * outHeight * outWidth;
                float[] x = bytes
                    ? [.. Enumerable.Range(0, (pixels + 3) / 4).Select(_ => BitConverter.Int32BitsToSingle(random.Next(int.MinValue, int.MaxValue) & 0x3f7f7f7f))]
                    : TrainingRandom(random, pixels, 0, 1);
                float[] values = bytes ? TrainingRandom(random, 256 * Channels, -1, 1) : [.. Enumerable.Range(0, 2 * Channels).Select(i => i % 2 == 0 ? 2f + i : -0.5f)];
                string shape = $"{Planes} planes of {Height}x{Width} -> {outHeight}x{outWidth}, {xTaps} and {yTaps} taps, {(bytes ? "bytes" : "floats")}";
                G5cCase($"resize_normalize_f32, {shape}", "resize_normalize_f32", [x, coefficients.Length == 0 ? [0f] : coefficients, values, new float[outputs]], [3],
                    (b, s) => b.ResizeNormalize(s[0], s[1], s[2], s[3], Planes, Channels, Height, Width, outHeight, outWidth, xTaps, yTaps, bytes),
                    a => [a[0], a[1], a[2], a[3], Planes, Channels, Height, Width, Cu(outHeight), Cu(outWidth), Cu(xTaps), Cu(yTaps), Cu(bytes ? 1 : 0), Cu(outputs)], outputs, worst,
                    exact: true);
            }
        }

        TrainingReport(worst);
    }

    // The 18 geometry values of the depthwise kernels (CUDA's order), then n where the kernel takes it.
    private static ulong[] DepthwiseArgs(ulong[] pointers, in ConvGeometry g, int filters, int flags, int n) =>
        [.. pointers, Cu(g.N), Cu(g.C), Cu(g.H), Cu(g.W), Cu(g.KH), Cu(g.KW), Cu(g.SH), Cu(g.SW), Cu(g.PH), Cu(g.PW), Cu(g.OH), Cu(g.OW), Cu(g.DH), Cu(g.DW), Cu(filters),
            Cu(g.C), Cu(flags), 1, .. (n >= 0 ? (ulong[])[Cu(n)] : [])];

    internal static void GcnConvolutionDepthwise(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(66);
        foreach (var (geometry, multiplier) in new[]
        {
            (new ConvGeometry(2, 3, 7, 6, 3, 3, 1, 2, 1, 1), 1), (new ConvGeometry(1, 2, 9, 8, 3, 2, 2, 1, 1, 0) { DH = 2 }, 2), (new ConvGeometry(2, 4, 1, 12, 1, 5, 1, 2, 0, 2), 1),
        })
        {
            var g = geometry;
            int filters = g.C * multiplier, input = g.N * g.C * g.H * g.W, outputs = g.N * filters * g.OH * g.OW, weights = filters * g.KH * g.KW;
            float[] x = TrainingRandom(random, input, -1, 1), w = TrainingRandom(random, weights, -1, 1), bias = TrainingRandom(random, filters, -0.5f, 0.5f);
            foreach (var activation in Enum.GetValues<ConvActivation>())
            {
                bool hasBias = activation != ConvActivation.Relu;
                int flags = (hasBias ? 1 : 0) | (int)activation << 1;
                G5cCase($"conv_dw_f32, {Describe(g)}, {filters} filters, {activation}{(hasBias ? ", a bias" : "")}", "conv_dw_f32", [x, w, bias, new float[outputs]], [3],
                    (b, s) => b.Convolution(s[0], s[1], hasBias ? s[2] : null, s[3], g, filters, g.C, activation), a => DepthwiseArgs([a[0], a[1], hasBias ? a[2] : a[1], a[3]], g, filters, flags, outputs),
                    outputs, worst);
            }

            G5cCase($"conv_dw_bwd_input_f32, {Describe(g)}, {filters} filters", "conv_dw_bwd_input_f32", [TrainingRandom(random, outputs, -1, 1), w, TrainingRandom(random, input, -1, 1)],
                [2], (b, s) => b.ConvolutionBackwardInput(s[0], s[1], s[2], g, filters, g.C), a => DepthwiseArgs([a[0], a[1], a[2]], g, filters, 0, input), input, worst);
            foreach (var isa in TrainingIsas)
            {
                TrainingCase($"conv_dw_bwd_weight_f32 ({isa}), {Describe(g)}, {filters} filters", [x, TrainingRandom(random, outputs, -1, 1), TrainingRandom(random, weights, -1, 1)], [2],
                    (b, s) => b.ConvolutionBackwardWeight(s[0], s[1], s[2], g, filters, g.C),
                    (m, a, grid, seed) => TrainingDispatch("conv_dw_bwd_weight_f32", isa, m, (grid, 1), seed, DepthwiseArgs([a[0], a[1], a[2]], g, filters, 0, -1)),
                    (2, weights), worst, "conv_dw_bwd_weight_f32");
            }
        }

        TrainingReport(worst);
    }

    // Log-probabilities over the classes (log-softmax of random logits) for [steps, batch] or [batch, steps] rows.
    private static float[] CtcLogProbs(Random random, int rows, int classes)
    {
        var values = new float[rows * classes];
        for (int r = 0; r < rows; r++)
        {
            var logits = Enumerable.Range(0, classes).Select(_ => random.NextDouble() * 4 - 2).ToArray();
            double top = logits.Max(), sum = logits.Sum(l => Math.Exp(l - top));
            for (int c = 0; c < classes; c++)
            {
                values[r * classes + c] = (float)(logits[c] - top - Math.Log(sum));
            }
        }

        return values;
    }

    internal static void GcnConvolutionCtc(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(67);
        const int Steps = 9, Batch = 4, Classes = 5;
        int[][] labels = [[1, 1, 2], [], [3, 3, 3, 3], [4, 2, 1, 2, 4]];
        int[] inputLengths = [9, 5, 6, 9], targetLengths = [.. labels.Select(l => l.Length)];
        int[] targetOffsets = [.. labels.Select((_, i) => labels.Take(i).Sum(l => l.Length))];
        float[] targets = [.. labels.SelectMany(l => l.Select(v => (float)v)), 0f];
        float[] meta = [.. Enumerable.Range(0, Batch).SelectMany(n => new[] { inputLengths[n], targetLengths[n], targetOffsets[n] }).Select(BitConverter.Int32BitsToSingle)];
        int states = 2 * targetLengths.Max() + 1;
        foreach (int blank in new[] { 0, 4 })
        {
            int[][] shifted = [.. labels.Select(l => l.Select(v => blank == 0 ? v : v - 1).ToArray())];
            float[] shiftedTargets = [.. shifted.SelectMany(l => l.Select(v => (float)v)), 0f];
            foreach (bool batchFirst in new[] { false, true })
            {
                foreach (bool zeroInfinity in new[] { true, false })
                {
                    float[] logProbs = CtcLogProbs(random, Steps * Batch, Classes);
                    string shape = $"{Batch} sequences of up to {Steps} steps, {Classes} classes, blank {blank}{(batchFirst ? ", batch first" : "")}{(zeroInfinity ? ", zeroinf" : "")}";
                    foreach (bool shared in new[] { true, false })
                    {
                        string forward = shared ? "ctc_loss_f32" : "ctc_loss_global_f32", backward = shared ? "ctc_loss_bwd_f32" : "ctc_loss_bwd_global_f32";
                        foreach (var isa in TrainingIsas)
                        {
                            TrainingCase($"{forward} ({isa}), {shape}", [logProbs, shiftedTargets, meta, new float[Batch], new float[Batch * 2 * states]], [3],
                                (b, s) => b.CtcLoss(s[0], s[1], s[3], inputLengths, targetLengths, targetOffsets, Steps, Batch, Classes, blank, batchFirst, zeroInfinity),
                                (m, a, grid, seed) => TrainingDispatch(forward, isa, m, (grid, 1), seed, a[0], a[1], a[2], a[3], a[4], Steps, Batch, Classes, Cu(blank),
                                    Cu(batchFirst ? 1 : 0), Cu(zeroInfinity ? 1 : 0), Cu(states)),
                                (1, Batch), worst, forward);
                            float[] lossGrads = [0.5f, 1f, zeroInfinity ? 2f : 0f, -0.75f];                // an infinite loss's gradient only with zeroinf
                            TrainingCase($"{backward} ({isa}), {shape}",
                                [logProbs, shiftedTargets, meta, lossGrads, TrainingRandom(random, Steps * Batch * Classes, -0.1f, 0.1f), new float[Batch * Steps * (states + 1)],
                                    new float[Batch * 4 * states]], [4],
                                (b, s) => b.CtcLossBackward(s[0], s[1], s[3], s[4], inputLengths, targetLengths, targetOffsets, Steps, Batch, Classes, blank, batchFirst, zeroInfinity),
                                (m, a, grid, seed) => TrainingDispatch(backward, isa, m, (grid, 1), seed, a[0], a[1], a[2], a[3], a[4], a[5], a[6], Steps, Batch, Classes, Cu(blank),
                                    Cu(batchFirst ? 1 : 0), Cu(zeroInfinity ? 1 : 0), Cu(states)),
                                (1, Batch), worst, backward);
                        }
                    }
                }
            }
        }

        TrainingReport(worst);
    }

    internal static void GcnConvolutionCells(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(68);
        const int Batch = 3, Steps = 4, Step = 2, H = 37;
        int n = Batch * H;
        foreach (bool keep in new[] { true, false })
        {
            int flags = keep ? 3 : 0;
            float[] Fresh(int count) => TrainingRandom(random, count, -1, 1);
            float[] Gate(int count) => TrainingRandom(random, count, 0.05f, 0.95f);
            foreach (var isa in TrainingIsas)
            {
                TrainingCase($"lstm_cell_f32 ({isa}), {Batch} x {H}{(keep ? ", gates and cells kept" : "")}",
                    [Fresh(Batch * Steps * 4 * H), Fresh(Batch * 4 * H), Fresh(n), Fresh(n), Fresh(Batch * Steps * H), Fresh(Batch * Steps * 4 * H), Fresh(Batch * Steps * H)],
                    keep ? [2, 3, 4, 5, 6] : [2, 3, 4],
                    (b, s) => b.LstmCell(s[0], s[1], s[2], s[3], s[4], keep ? s[5] : null, keep ? s[6] : null, Step, Steps, Batch, H),
                    (m, a, grid, seed) => TrainingDispatch("lstm_cell_f32", isa, m, (grid, 1), seed, a[0], a[1], a[2], a[3], a[4], keep ? a[5] : a[4], keep ? a[6] : a[4], Step,
                        Steps, H, Cu(flags), Cu(n)),
                    (1, 3), worst, "lstm_cell_f32");
                TrainingCase($"gru_cell_f32 ({isa}), {Batch} x {H}{(keep ? ", a bias, gates kept" : "")}",
                    [Fresh(Batch * Steps * 3 * H), Fresh(Batch * 3 * H), Fresh(H), Fresh(n), Fresh(Batch * Steps * H), Fresh(Batch * Steps * 4 * H)],
                    keep ? [3, 4, 5] : [3, 4],
                    (b, s) => b.GruCell(s[0], s[1], keep ? s[2] : null, s[3], s[4], keep ? s[5] : null, Step, Steps, Batch, H),
                    (m, a, grid, seed) => TrainingDispatch("gru_cell_f32", isa, m, (grid, 1), seed, a[0], a[1], keep ? a[2] : a[1], a[3], a[4], keep ? a[5] : a[4], Step, Steps, H,
                        Cu(flags), Cu(n)),
                    (1, 3), worst, "gru_cell_f32");
                foreach (int previous in new[] { 1, -1 })
                {
                    int backwardFlags = (keep ? 1 : 0) | (previous >= 0 ? 2 : 0);
                    TrainingCase($"lstm_cell_bwd_f32 ({isa}), {Batch} x {H}{(keep ? ", dOutput" : "")}{(previous >= 0 ? ", a previous step" : "")}",
                        [Gate(Batch * Steps * 4 * H), Fresh(Batch * Steps * H), Fresh(Batch * Steps * H), Fresh(n), Fresh(n), Fresh(Batch * Steps * 4 * H), Fresh(Batch * 4 * H)],
                        [4, 5, 6],
                        (b, s) => b.LstmCellBackward(s[0], s[1], keep ? s[2] : null, s[3], s[4], s[5], s[6], Step, previous, Steps, Batch, H),
                        (m, a, grid, seed) => TrainingDispatch("lstm_cell_bwd_f32", isa, m, (grid, 1), seed, a[0], a[1], keep ? a[2] : a[1], a[3], a[4], a[5], a[6], Step,
                            Cu(Math.Max(previous, 0)), Steps, H, Cu(backwardFlags), Cu(n)),
                        (1, 3), worst, "lstm_cell_bwd_f32");
                    TrainingCase($"gru_cell_bwd_f32 ({isa}), {Batch} x {H}{(keep ? ", dOutput" : "")}{(previous >= 0 ? ", a previous step" : "")}",
                        [Gate(Batch * Steps * 4 * H), Fresh(Batch * Steps * H), Fresh(Batch * Steps * H), Fresh(n), Fresh(Batch * Steps * 3 * H), Fresh(Batch * Steps * 3 * H),
                            Fresh(Batch * 3 * H)],
                        [3, 4, 5, 6],
                        (b, s) => b.GruCellBackward(s[0], s[1], keep ? s[2] : null, s[3], s[4], s[5], s[6], Step, previous, Steps, Batch, H),
                        (m, a, grid, seed) => TrainingDispatch("gru_cell_bwd_f32", isa, m, (grid, 1), seed, a[0], a[1], keep ? a[2] : a[1], a[3], a[4], a[5], a[6], Step,
                            Cu(Math.Max(previous, 0)), Steps, H, Cu(backwardFlags), Cu(n)),
                        (1, 3), worst, "gru_cell_bwd_f32");
                }
            }
        }

        TrainingReport(worst);
    }

    // Rows `row` of each sequence of a [batch, steps, width] sequence into a [batch, width] storage of the CPU (Copy2D).
    private static Storage StepRows(Backend b, Storage sequence, int row, int steps, int batch, int width)
    {
        var rows = b.Allocate(batch * width, zeroed: true);
        b.Copy2D(sequence, row * width, steps * width, rows, 0, width, batch, width, false);
        return rows;
    }

    internal static void GcnConvolutionSteps(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(69);
        const int Batch = 2, Steps = 3, Step = 1, H = 70;
        int blocks = Batch * ((H + GcnKernels.StepUnits - 1) / GcnKernels.StepUnits);
        float[] Fresh(int count) => TrainingRandom(random, count, -1, 1);
        float[] Small(int count) => TrainingRandom(random, count, -0.2f, 0.2f);
        float[] Gate(int count) => TrainingRandom(random, count, 0.05f, 0.95f);
        foreach (int previous in new[] { 0, -1 })
        {
            string which = previous >= 0 ? ", a previous step" : ", the first step";
            foreach (var isa in TrainingIsas)
            {
                // Forward: the product of the previous output row with U [H, G·H], then the cell (the CPU composes them).
                int lstmFlags = 3 | (previous >= 0 ? 4 : 0);
                TrainingCase($"lstm_step_f32 ({isa}), {Batch} x {H}{which}",
                    [Fresh(Batch * Steps * 4 * H), Small(H * 4 * H), Fresh(Batch * H), Fresh(Batch * Steps * H), Fresh(Batch * Steps * 4 * H), Fresh(Batch * Steps * H)], [2, 3, 4, 5],
                    (b, s) =>
                    {
                        var recurrent = b.Allocate(Batch * 4 * H, zeroed: true);
                        var hidden = b.Allocate(Batch * H, zeroed: true);
                        if (previous >= 0)
                        {
                            var rows = StepRows(b, s[3], previous, Steps, Batch, H);
                            b.MatMul(rows, s[1], recurrent, Batch, 4 * H, H, false, false, 0f);
                            rows.Release();
                        }

                        b.LstmCell(s[0], recurrent, s[2], hidden, s[3], s[4], s[5], Step, Steps, Batch, H);
                        recurrent.Release();
                        hidden.Release();
                    },
                    (m, a, _, seed) => TrainingDispatch("lstm_step_f32", isa, m, (blocks, 1), seed, a[0], a[1], a[2], a[3], a[4], a[5], Step, Cu(Math.Max(previous, 0)), Steps, H,
                        Cu(lstmFlags)),
                    (blocks, blocks), worst, "lstm_step_f32");
                int gruFlags = 3 | (previous >= 0 ? 4 : 0);
                TrainingCase($"gru_step_f32 ({isa}), {Batch} x {H}{which}",
                    [Fresh(Batch * Steps * 3 * H), Small(H * 3 * H), Fresh(H), Fresh(Batch * Steps * H), Fresh(Batch * Steps * 4 * H)], [3, 4],
                    (b, s) =>
                    {
                        var recurrent = b.Allocate(Batch * 3 * H, zeroed: true);
                        var hidden = previous >= 0 ? StepRows(b, s[3], previous, Steps, Batch, H) : b.Allocate(Batch * H, zeroed: true);
                        if (previous >= 0)
                        {
                            b.MatMul(hidden, s[1], recurrent, Batch, 3 * H, H, false, false, 0f);
                        }

                        b.GruCell(s[0], recurrent, s[2], hidden, s[3], s[4], Step, Steps, Batch, H);
                        recurrent.Release();
                        hidden.Release();
                    },
                    (m, a, _, seed) => TrainingDispatch("gru_step_f32", isa, m, (blocks, 1), seed, a[0], a[1], a[2], a[3], a[4], Step, Cu(Math.Max(previous, 0)), Steps, H,
                        Cu(gruFlags)),
                    (blocks, blocks), worst, "gru_step_f32");

                // Gradients: the product of the later step's gate gradient with Uᵀ [G·H, H], then the cell's gradient.
                const int Next = 2;
                int next = previous >= 0 ? Next : -1, backwardFlags = 1 | (previous >= 0 ? 2 : 0) | (next >= 0 ? 4 : 0);
                TrainingCase($"lstm_step_bwd_f32 ({isa}), {Batch} x {H}{which}",
                    [Gate(Batch * Steps * 4 * H), Fresh(Batch * Steps * H), Fresh(Batch * Steps * H), Small(4 * H * H), Fresh(Batch * H), Fresh(Batch * Steps * 4 * H)], [4, 5],
                    (b, s) =>
                    {
                        var dHidden = b.Allocate(Batch * H, zeroed: true);
                        var dStep = b.Allocate(Batch * 4 * H, zeroed: true);
                        if (next >= 0)
                        {
                            var rows = StepRows(b, s[5], next, Steps, Batch, 4 * H);
                            b.MatMul(rows, s[3], dHidden, Batch, H, 4 * H, false, false, 0f);
                            rows.Release();
                        }

                        b.LstmCellBackward(s[0], s[1], s[2], dHidden, s[4], s[5], dStep, Step, previous, Steps, Batch, H);
                        dHidden.Release();
                        dStep.Release();
                    },
                    (m, a, _, seed) => TrainingDispatch("lstm_step_bwd_f32", isa, m, (blocks, 1), seed, a[0], a[1], a[2], a[3], a[4], a[5], Step, Cu(Math.Max(next, 0)),
                        Cu(Math.Max(previous, 0)), Steps, H, Cu(backwardFlags)),
                    (blocks, blocks), worst, "lstm_step_bwd_f32");
                TrainingCase($"gru_step_bwd_f32 ({isa}), {Batch} x {H}{which}",
                    [Gate(Batch * Steps * 4 * H), Fresh(Batch * Steps * H), Fresh(Batch * Steps * H), Small(3 * H * H), Fresh(Batch * H), Fresh(Batch * Steps * 3 * H),
                        Fresh(Batch * Steps * 3 * H)], [4, 5, 6],
                    (b, s) =>
                    {
                        var dStep = b.Allocate(Batch * 3 * H, zeroed: true);
                        if (next >= 0)
                        {
                            var rows = StepRows(b, s[6], next, Steps, Batch, 3 * H);
                            b.MatMul(rows, s[3], s[4], Batch, H, 3 * H, false, false, 1f);
                            rows.Release();
                        }

                        b.GruCellBackward(s[0], s[1], s[2], s[4], s[5], s[6], dStep, Step, previous, Steps, Batch, H);
                        dStep.Release();
                    },
                    (m, a, _, seed) => TrainingDispatch("gru_step_bwd_f32", isa, m, (blocks, 1), seed, a[0], a[1], a[2], a[3], a[4], a[5], a[6], Step, Cu(Math.Max(next, 0)),
                        Cu(Math.Max(previous, 0)), Steps, H, Cu(backwardFlags)),
                    (blocks, blocks), worst, "gru_step_bwd_f32");
            }
        }

        TrainingReport(worst);
    }

    internal static void GcnConvolutionPackedGated(Device device)
    {
        _ = device;
        var worst = new Dictionary<string, float>();
        var random = new Random(70);
        foreach (int kind in new[] { 0, 1, 2, 3 })
        {
            foreach (int flags in new[] { 3, 15, 1, 2 })
            {
                foreach (int n in new[] { 1, 255, 1003 })
                {
                    int words = (n + 1) / 2;
                    float[] Packed() => [.. Enumerable.Range(0, words).Select(_ =>
                        BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(random.NextSingle() * 6 - 3) >> 16 | BitConverter.SingleToUInt32Bits(random.NextSingle() * 6 - 3) & 0xffff0000u))];
                    G5cCase($"gated_act_bwd_bf16_f32, kind {kind}, flags {flags}, {n} elements", "gated_act_bwd_bf16_f32",
                        [Packed(), Packed(), TrainingRandom(random, n, -1, 1), TrainingRandom(random, n, -1, 1), TrainingRandom(random, n, -1, 1)], [3, 4],
                        (b, s) => b.GatedActivationBackwardPacked(s[0], s[1], s[2], s[3], s[4], n, kind, flags),
                        a => [a[0], a[1], a[2], a[3], a[4], Cu(kind), Cu(flags), Cu(n), Cu(words)], words, worst);
                }
            }
        }

        TrainingReport(worst);
    }
}
