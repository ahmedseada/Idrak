// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Gpu.Vulkan;

// Convolution's choices (VulkanTuneOp.Convolution) give the same bits (docs/optimization.md rule 86): the forward pass, the
// input gradient and the weight gradient each run every path the backend can choose (the composed forward and input
// gradient, the implicit products tiled and register-blocked, the weight gradient's unfolded products, the weight gradient
// at every split count, the depthwise kernels), forced through
// VulkanBackend.ConvolutionPath and ConvolutionSplits, at every width with and without subgroups (VulkanOrderOver).
// IDRAK_ORDER_CASES=<text> keeps only the runs whose case or choice contains it (a bench of a few choices with
// IDRAK_ORDER_TIMES=1).
internal static partial class Tests
{
    private static void VulkanOrderConvolution(Device device) => VulkanOrderOver(device, "convolution", backend =>
    {
        var random = new Random(67);
        var runs = new List<OrderRun>();
        // (geometry, filters, groups): 2-D and 1-D (one row), strides, padding (asymmetric too), dilation, groups,
        // depthwise with one and two filters a channel, 20,000 positions, positions past what one split of the
        // register-blocked weight gradient walks (80,000 > 2^9 leaves of 128: its splits are raised to fit its stack), and
        // 32 and 16 filters a group (the input gradient's windows a multiple of every step, or of some: ConvSegments).
        (ConvGeometry G, int Filters, int Groups)[] cases =
        [
            (new(2, 3, 9, 7, 3, 3, 1, 1, 1, 1), 5, 1),
            (new(2, 4, 11, 9, 3, 2, 2, 1, 1, 0) { DH = 2 }, 6, 2),
            (new(1, 4, 7, 9, 3, 3, 2, 2, 0, 1) { PadBottom = 2, PadRight = 0 }, 4, 1),
            (new(2, 4, 1, 300, 1, 5, 1, 2, 0, 2) { DW = 2 }, 6, 1),
            (new(2, 6, 10, 10, 3, 3, 1, 1, 1, 1), 6, 6),
            (new(3, 4, 1, 500, 1, 7, 1, 1, 0, 3) { DW = 2 }, 8, 4),
            (new(2, 3, 100, 100, 3, 3, 1, 1, 1, 1), 4, 1),
            (new(2, 4, 100, 100, 3, 3, 1, 1, 1, 1), 4, 4),
            (new(2, 1, 1, 40000, 1, 3, 1, 1, 0, 1), 2, 1),
            (new(1, 4, 9, 9, 3, 3, 1, 1, 1, 1), 32, 1),
            (new(2, 4, 6, 7, 3, 3, 2, 2, 1, 1), 32, 2),
        ];
        const int Composed = 0, Tile = 1, Blocked = 2, Depthwise = 3, UnfoldedTile = 4, UnfoldedBlocked = 5;
        foreach (var (g, filters, groups) in cases)
        {
            int input = g.N * g.C * g.H * g.W, output = g.N * filters * g.OH * g.OW, weights = filters * (g.PatchSize / groups);
            var x = OrderValues(random, input);
            var w = OrderValues(random, weights);
            var bias = OrderValues(random, filters);
            var dy = OrderValues(random, output);
            var dx0 = OrderValues(random, input);
            var dw0 = OrderValues(random, weights);
            bool depthwise = groups > 1 && groups == g.C;
            string shape = $"{g.N}×{g.C}×{g.H}×{g.W}, {g.KH}×{g.KW} window, stride {g.SH}×{g.SW}, padding {g.PH},{g.PW},{g.PadBottom},{g.PadRight}, " +
                $"dilation {g.DH}×{g.DW}, {filters} filters, {groups} groups ({(long)g.N * g.OH * g.OW} positions)";
            int[] paths = depthwise ? [Composed, Tile, Blocked, Depthwise] : [Composed, Tile, Blocked];
            string Name(int path) => path switch
            {
                Composed => "composed", Tile => "implicit tiled", Blocked => "implicit register-blocked", UnfoldedTile => "unfolded tiled",
                UnfoldedBlocked => "unfolded register-blocked", _ => "depthwise",
            };

            // Runs `body` under a forced path (and the weight gradient's splits), clearing them after.
            static void Forced(int path, int? splits, Action body)
            {
                (VulkanBackend.ConvolutionPath, VulkanBackend.ConvolutionSplits) = (path, splits);
                try
                {
                    body();
                }
                finally
                {
                    (VulkanBackend.ConvolutionPath, VulkanBackend.ConvolutionSplits) = (null, null);
                }
            }

            foreach (var (activation, withBias) in new[] { (ConvActivation.Gelu, true), (ConvActivation.None, false) })
            {
                foreach (int path in paths)
                {
                    runs.Add(new OrderRun($"forward of {shape}, {(withBias ? "bias" : "no bias")}, {activation}", Name(path), b =>
                    {
                        var s = new OrderStorages(b);
                        var (xs, ws, bs, y) = (s.Up(x), s.Up(w), s.Up(bias), s.Out(output));
                        return new OrderCase(() => Forced(path, null, () => b.ConvolutionKernel(xs, ws, withBias ? bs : null, y, in g, filters, groups, activation)),
                            () => [s.Down(y)], s.Dispose);
                    }));
                }
            }

            foreach (int path in paths)
            {
                runs.Add(new OrderRun($"input gradient of {shape}", Name(path), b =>
                {
                    var s = new OrderStorages(b);
                    var (dys, ws, dx) = (s.Up(dy), s.Up(w), s.Up(dx0));
                    return new OrderCase(() =>
                    {
                        b.Upload(dx0, dx);                                 // accumulates: the same dx every run
                        Forced(path, null, () => b.ConvolutionBackwardInputKernel(dys, ws, dx, in g, filters, groups));
                    }, () => [s.Down(dx)], s.Dispose);
                }));
            }

            // The weight gradient at every split count of its positions (the backend fits each to the tree and the stack),
            // gathered and unfolded. The composed weight gradient sums its positions in turn; in float32 the unfolded
            // products stand in for it (VulkanBackend.ConvCandidates).
            foreach (int path in paths.Where(p => p != Composed).Concat([UnfoldedTile, UnfoldedBlocked]))
            {
                foreach (int? splits in path == Depthwise ? new int?[] { null } : OrderSplits.Append(256).Select(s => (int?)s))
                {
                    runs.Add(new OrderRun($"weight gradient of {shape}", splits is int n ? $"{Name(path)}, {n} splits" : Name(path), b =>
                    {
                        var s = new OrderStorages(b);
                        var (xs, dys, dw) = (s.Up(x), s.Up(dy), s.Up(dw0));
                        return new OrderCase(() =>
                        {
                            b.Upload(dw0, dw);                             // accumulates: the same dweight every run
                            Forced(path, splits, () => b.ConvolutionBackwardWeightKernel(xs, dys, dw, in g, filters, groups));
                        }, () => [s.Down(dw)], s.Dispose);
                    }));
                }
            }
        }

        return Environment.GetEnvironmentVariable("IDRAK_ORDER_CASES") is { Length: > 0 } only
            ? [.. runs.Where(r => $"{r.Case}: {r.Choice}".Contains(only, StringComparison.OrdinalIgnoreCase))]
            : runs;
    });
}
