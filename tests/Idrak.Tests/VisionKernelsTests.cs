// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Abstraction.Devices.Cpu;
using Idrak.Data;
using Idrak.Gpu.Cuda;
using Idrak.Gpu.Vulkan;
using Idrak.Layers;

// Plan 13, step 1: convolution, pooling, resampling and CTC as device operations. On the CPU: the inference fusion of a
// convolution with its batch norm and activation equals the layers run one by one, and ImagePreprocessor's device
// operation gives the host path's pixels bit for bit. On a GPU (CUDA, Vulkan): every convolution path (composed, the
// implicit products on both tiles and split counts, depthwise) and every new kernel matches the CPU with no host fallback.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] VisionKernelsGroup =
    [
        ("vision kernels: fused inference (Conv2d, BatchNorm in evaluation mode, an activation) in a Sequential equals the layers one by one; a training BatchNorm is not fused", FusedConvolutionMatches),
        ("vision kernels: ImagePreprocessor's device operation gives the host path's pixels bit for bit (bilinear, bicubic, lanczos, box; shrink, enlarge, one axis; colour and grey)", DeviceResizeMatchesHost),
        ("vision kernels: ConvTranspose2d through the convolution's input gradient matches the products and fold, forward and gradients", TransposedConvolutionMatches),
        ("vision kernels: every convolution path on the device (composed, implicit tiled and blocked, splits, depthwise; groups, dilation, asymmetric padding) matches the CPU, forward and both gradients, with no host fallback", DeviceConvolutionPaths),
        ("vision kernels: average pooling (ceil mode), interpolation, adaptive pooling, resize-normalize and CTC run as device kernels and match the CPU with no host fallback", DeviceImageKernels),
    ];

    private static void FusedConvolutionMatches(Device device)
    {
        var random = new Random(31);
        foreach (Module? activation in new Module?[] { new ReLU(), new Sigmoid(), new Tanh(), new GELU(), null })
        {
            foreach (bool withNorm in new[] { true, false })
            {
                if (!withNorm && activation is null)
                {
                    continue;
                }

                var conv = new Conv2d(3, 8, (3, 3), padding: (1, 1), device: device, random: random);
                var norm = new BatchNorm(8, device: device);
                using (Autograd.NoGrad())
                {
                    norm.Gamma.CopyFrom(RandomArray(random, 8));
                    norm.Beta.CopyFrom(RandomArray(random, 8));
                    norm.RunningMean.CopyFrom(RandomArray(random, 8));
                    norm.RunningVariance.CopyFrom([.. RandomArray(random, 8).Select(v => MathF.Abs(v) + 0.5f)]);
                }

                var layers = new List<Module> { conv };
                if (withNorm)
                {
                    layers.Add(norm);
                }

                if (activation is not null)
                {
                    layers.Add(activation);
                }

                var model = new Sequential(layers);
                model.Eval();
                using var x = Tensor.From(RandomArray(random, 2 * 3 * 9 * 7), [2, 3, 9, 7], device);
                float[] fused, oneByOne;
                using (Autograd.NoGrad())
                {
                    using var y = model.Forward(x);
                    fused = y.ToArray();
                    var z = x;
                    foreach (var layer in layers)
                    {
                        z = layer.Forward(z);
                    }

                    oneByOne = z.ToArray();
                }

                AssertClose(oneByOne, fused, 1e-4f, $"fused {(withNorm ? "conv + batch norm" : "conv")} + {activation?.ToString() ?? "nothing"}");
                Check(Conv2d.FusedSteps(layers, layers.Count) is { Length: 1 }, "the layers fuse into one step");
            }
        }

        // A batch norm in training mode is not folded.
        var training = new List<Module> { new Conv2d(2, 2, 3, device: device), new BatchNorm(2, device: device), new ReLU() };
        training[1].Train();
        Check(Conv2d.FusedSteps(training, 3) is null, "a training batch norm stays a layer of its own");
    }

    private static void DeviceResizeMatchesHost(Device device)
    {
        var random = new Random(7);
        foreach (var (h, w, oh, ow, channels) in new[] { (37, 53, 16, 24, 3), (9, 7, 20, 15, 3), (30, 40, 30, 17, 1), (25, 18, 11, 18, 3) })
        {
            var pixels = Enumerable.Range(0, channels * h * w).Select(_ => random.Next(256) / 255f).ToArray();
            var image = new ImageData(pixels, channels, h, w);
            foreach (var filter in new[] { ImageResampling.Bilinear, ImageResampling.Bicubic, ImageResampling.Lanczos, ImageResampling.Box })
            {
                var pre = new ImagePreprocessor { Height = oh, Width = ow, Resampling = filter, Mean = [0.48f, 0.45f, 0.40f], Std = [0.27f, 0.26f, 0.28f] };
                var host = pre.Pixels(image);
                using var onDevice = pre.ProcessOnDevice(image, device);
                var actual = onDevice.ToArray();
                Check(host.Length == actual.Length, $"{filter} {h}x{w} to {oh}x{ow}: {actual.Length} values, expected {host.Length}");
                for (int i = 0; i < host.Length; i++)
                {
                    Check(BitConverter.SingleToInt32Bits(host[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                        $"{filter} {h}x{w} to {oh}x{ow} ({channels} channels): value {i} is {actual[i]}, the host path's {host[i]}");
                }
            }
        }
    }

    private static void TransposedConvolutionMatches(Device device)
    {
        var random = new Random(11);
        foreach (var (groups, stride, padding, outputPadding) in new[] { (1, 2, 1, 1), (2, 1, 0, 0), (4, 2, 1, 0) })
        {
            var layer = new ConvTranspose2d(4, 8, (3, 3), (stride, stride), (padding, padding), (outputPadding, outputPadding), groups: groups, device: device, random: random);
            using var x = Tensor.From(RandomArray(random, 2 * 4 * 5 * 6), [2, 4, 5, 6], device, requiresGrad: true);
            var y = layer.Forward(x);
            var g = layer.Geometry(2, 5, 6);

            // The reference: the products and fold the layer ran before (rows · weight per group, then Col2Im).
            int n = 2, h = 5, w = 6, perGroup = 4 / groups, patch = layer.Weight.Shape[1];
            using (Autograd.NoGrad())
            {
                var rows = x.Permute(0, 2, 3, 1).Reshape(n * h * w, 4);
                var grouped = rows.Reshape(n * h * w, groups, perGroup).Permute(1, 0, 2);
                var columns = grouped.MatMul(layer.Weight.Reshape(groups, perGroup, patch)).Permute(1, 0, 2).Reshape(n * h * w, groups * patch);
                var expected = columns.Col2Im(g);
                if (layer.Bias is { } bias)
                {
                    expected = expected.GroupAffine(null, bias, 8, g.H * g.W);
                }

                AssertClose(expected.ToArray(), y.ToArray(), 1e-4f, $"transposed convolution, groups {groups}, stride {stride}");
            }

            GradCheck(device, x.Shape.ToArray(), input => layer.Forward(input).Square().Sum(), scale: 0.5f + groups / 8f);
        }
    }

    // Each GPU backend's forced convolution path (null: measured).
    private static void ForceConvolutionPath(int? path)
    {
        CudaBackend.ConvolutionPath = path;
        VulkanBackend.ConvolutionPath = path;
    }

    // Runs `op` on the CPU and the device on copies of the same inputs; every storage must agree (relative tolerance), and the
    // device must take no host fallback.
    private static void OnBoth(Device device, string what, float[][] inputs, Action<Backend, Storage[]> op, float tolerance = 1e-3f)
    {
        var cpu = CpuBackend.Instance;
        var backend = device.Backend;
        var host = inputs.Select(d => { var s = cpu.Allocate(Math.Max(1, d.Length), false); cpu.Upload(d, s); return s; }).ToArray();
        var gpu = inputs.Select(d => { var s = backend.Allocate(Math.Max(1, d.Length), false); backend.Upload(d, s); return s; }).ToArray();
        try
        {
            op(cpu, host);
            using (var trace = Idrak.Abstraction.Operations.Kernels.Trace(backend))
            {
                op(backend, gpu);
                backend.Synchronize();
                // Named by operation, so a failure says which one left the device.
                Check(trace.HostCalls == 0, $"{device}: {what} took the host fallback ({string.Join(", ", trace.HostCallsByOperation.Select(p => $"{p.Key} ×{p.Value}"))})");
            }
            for (int i = 0; i < inputs.Length; i++)
            {
                var (expected, actual) = (new float[inputs[i].Length], new float[inputs[i].Length]);
                cpu.Download(host[i], expected);
                backend.Download(gpu[i], actual);
                AssertClose(expected, actual, tolerance, $"{device}: {what}, storage {i}");
            }
        }
        finally
        {
            foreach (var s in host.Concat(gpu))
            {
                s.Release();
            }
        }
    }

    private static void DeviceConvolutionPaths(Device device)
    {
        if (device.Type is not (DeviceType.Cuda or DeviceType.Vulkan))
        {
            return;
        }

        var random = new Random(5);
        float[] R(int n) => RandomArray(random, n);
        (ConvGeometry G, int Filters, int Groups)[] cases =
        [
            (new(2, 3, 9, 7, 3, 3, 1, 1, 1, 1), 5, 1),
            (new(2, 16, 14, 13, 3, 3, 1, 1, 1, 1), 70, 1),                       // past one 64 x 64 tile, a sum past 16
            (new(1, 4, 11, 9, 3, 2, 2, 1, 1, 0) { DH = 2 }, 6, 2),               // groups, rectangular, strided, dilated
            (new(2, 6, 10, 10, 3, 3, 1, 1, 1, 1), 6, 6),                         // depthwise
            (new(2, 4, 9, 8, 5, 3, 2, 2, 2, 1) { DW = 2 }, 8, 4),                // depthwise, two filters a channel, dilated
            (new(1, 4, 7, 9, 3, 3, 2, 2, 0, 1) { PadBottom = 2, PadRight = 0 }, 4, 1),   // more padding below than above
            (new(3, 8, 6, 5, 1, 1, 1, 1, 0, 0), 12, 1),                          // pointwise
        ];
        try
        {
            foreach (int? path in new int?[] { 0, 1, 2, 3, null })
            {
                ForceConvolutionPath(path);
                foreach (var (g, filters, groups) in cases)
                {
                    if (path == 3 && !(groups > 1 && groups == g.C))
                    {
                        continue;
                    }

                    int input = g.N * g.C * g.H * g.W, output = g.N * filters * g.OH * g.OW, weights = filters * (g.PatchSize / groups);
                    string label = $"path {path?.ToString() ?? "measured"}, {g}, {filters} filters, {groups} groups";
                    OnBoth(device, "forward " + label, [R(input), R(weights), R(filters), new float[output]],
                        (b, s) => b.Convolution(s[0], s[1], s[2], s[3], g, filters, groups, ConvActivation.Relu));
                    OnBoth(device, "forward without bias " + label, [R(input), R(weights), new float[output]],
                        (b, s) => b.Convolution(s[0], s[1], null, s[2], g, filters, groups, ConvActivation.Silu));
                    OnBoth(device, "input gradient " + label, [R(output), R(weights), R(input)],
                        (b, s) => b.ConvolutionBackwardInput(s[0], s[1], s[2], g, filters, groups));
                    OnBoth(device, "weight gradient " + label, [R(input), R(output), R(weights)],
                        (b, s) => b.ConvolutionBackwardWeight(s[0], s[1], s[2], g, filters, groups));
                }
            }

            // A weight gradient over many positions (the split paths), measured.
            var large = new ConvGeometry(8, 8, 40, 40, 3, 3, 1, 1, 1, 1);
            OnBoth(device, "weight gradient over 12,800 positions", [R(8 * 8 * 1600), R(8 * 16 * 1600), R(16 * 72)],
                (b, s) => b.ConvolutionBackwardWeight(s[0], s[1], s[2], large, 16, 1), tolerance: 2e-3f);
        }
        finally
        {
            ForceConvolutionPath(null);
        }
    }

    private static void DeviceImageKernels(Device device)
    {
        if (device.Type is not (DeviceType.Cuda or DeviceType.Vulkan))
        {
            return;
        }

        var random = new Random(9);
        float[] R(int n) => RandomArray(random, n);

        // Average pooling, a ceil-mode geometry (the last window's extra padding not counted) and asymmetric padding.
        foreach (var (g, bottom, right) in new[]
        {
            (new ConvGeometry(2, 3, 9, 8, 3, 3, 2, 2, 1, 1), 1, 1),
            (new ConvGeometry(2, 2, 8, 7, 3, 3, 2, 2, 1, 0) { PadBottom = 2, PadRight = 2 }, 1, 1),
            (new ConvGeometry(1, 3, 6, 9, 2, 3, 2, 2, 0, 1) { PadBottom = 1, PadRight = 2 }, 1, 2),
        })
        {
            foreach (bool countPad in new[] { true, false })
            {
                int input = g.N * g.C * g.H * g.W, count = g.N * g.C * g.OH * g.OW;
                OnBoth(device, $"average pooling {g} ({countPad})", [R(input), new float[count]], (b, s) => b.AvgPool(s[0], s[1], g, countPad, bottom, right), 1e-5f);
                OnBoth(device, $"average pooling gradient {g} ({countPad})", [R(count), R(input)], (b, s) => b.AvgPoolBackward(s[0], s[1], g, countPad, bottom, right), 1e-5f);
            }
        }

        // Interpolation, both modes, corners aligned or not, up, down and odd ratios.
        foreach (var (h, w, oh, ow) in new[] { (5, 7, 10, 14), (12, 9, 5, 4), (6, 6, 13, 7), (4, 5, 1, 1) })
        {
            foreach (var mode in new[] { InterpolationMode.Nearest, InterpolationMode.Bilinear })
            {
                foreach (bool align in mode == InterpolationMode.Bilinear ? new[] { false, true } : new[] { false })
                {
                    float sh = align ? (oh > 1 ? (h - 1) / (float)(oh - 1) : 0f) : h / (float)oh, sw = align ? (ow > 1 ? (w - 1) / (float)(ow - 1) : 0f) : w / (float)ow;
                    string label = $"{mode} {h}x{w} to {oh}x{ow}{(align ? ", corners aligned" : "")}";
                    OnBoth(device, "interpolation " + label, [R(3 * h * w), new float[3 * oh * ow]], (b, s) => b.Interpolate2d(s[0], s[1], 3, h, w, oh, ow, mode, align, sh, sw), 1e-5f);
                    OnBoth(device, "interpolation gradient " + label, [R(3 * oh * ow), R(3 * h * w)],
                        (b, s) => b.Interpolate2dBackward(s[0], s[1], 3, h, w, oh, ow, mode, align, sh, sw), 1e-5f);
                }
            }
        }

        // Adaptive pooling, windows that overlap and that do not.
        foreach (var (h, w, oh, ow) in new[] { (7, 9, 3, 4), (8, 8, 4, 4), (5, 6, 7, 2) })
        {
            OnBoth(device, $"adaptive average pooling {h}x{w} to {oh}x{ow}", [R(2 * h * w), new float[2 * oh * ow]], (b, s) => b.AdaptiveAvgPool(s[0], s[1], 2, h, w, oh, ow), 1e-5f);
            OnBoth(device, $"adaptive average pooling gradient {h}x{w} to {oh}x{ow}", [R(2 * oh * ow), R(2 * h * w)],
                (b, s) => b.AdaptiveAvgPoolBackward(s[0], s[1], 2, h, w, oh, ow), 1e-5f);
            OnBoth(device, $"adaptive max pooling and its gradient {h}x{w} to {oh}x{ow}", [R(2 * h * w), new float[2 * oh * ow], new float[2 * oh * ow], R(2 * oh * ow), R(2 * h * w)],
                (b, s) =>
                {
                    b.AdaptiveMaxPool(s[0], s[1], s[2], 2, h, w, oh, ow);
                    b.AdaptiveMaxPoolBackward(s[3], s[2], s[4], 2, h, w, oh, ow);
                }, 0f);
        }

        // Resize-normalize: floats (bilinear coefficients) and Pillow's 8-bit passes.
        var (across, xTaps) = Tensor.BilinearTaps(23, 11, antialias: true);
        var (down, yTaps) = Tensor.BilinearTaps(17, 30, antialias: false);
        float[] coefficients = [.. across, .. down];
        OnBoth(device, "resize-normalize, floats", [R(6 * 17 * 23), coefficients, R(6), new float[6 * 30 * 11]],
            (b, s) => b.ResizeNormalize(s[0], s[1], s[2], s[3], 6, 3, 17, 23, 30, 11, xTaps, yTaps, bytes: false), 1e-5f);

        // CTC: a short label sequence (rows in workgroup or shared memory) and a long one (rows in the scratch buffer), both layouts.
        foreach (int labels in new[] { 5, 400 })
        {
            foreach (bool batchFirst in new[] { false, true })
            {
                int steps = labels * 2 + 10, batch = 3, classes = 7;
                var logits = R(steps * batch * classes);
                var logProbs = new float[logits.Length];
                for (int r = 0; r < steps * batch; r++)
                {
                    float max = logits.Skip(r * classes).Take(classes).Max();
                    double sum = logits.Skip(r * classes).Take(classes).Sum(v => Math.Exp(v - max));
                    for (int c = 0; c < classes; c++)
                    {
                        logProbs[r * classes + c] = (float)(logits[r * classes + c] - max - Math.Log(sum));
                    }
                }

                int[] lengths = [labels, labels / 2, 0], inputs = [steps, steps - 3, steps / 2], offsets = [0, labels, labels + labels / 2];
                var targets = Enumerable.Range(0, lengths.Sum()).Select(_ => (float)(1 + random.Next(classes - 1))).ToArray();
                string label = $"CTC, {labels} labels{(batchFirst ? ", batch first" : "")}";
                OnBoth(device, label, [logProbs, targets, new float[batch]],
                    (b, s) => b.CtcLoss(s[0], s[1], s[2], inputs, lengths, offsets, steps, batch, classes, 0, batchFirst, zeroInfinity: false), 1e-4f);
                OnBoth(device, label + " gradient", [logProbs, targets, [1f, 0.5f, 2f], R(logProbs.Length)],
                    (b, s) => b.CtcLossBackward(s[0], s[1], s[2], s[3], inputs, lengths, offsets, steps, batch, classes, 0, batchFirst, zeroInfinity: false), 1e-3f);
            }
        }

        // ImagePreprocessor on the device: the host path's pixels, bit for bit.
        DeviceResizeMatchesHost(device);
    }
}
