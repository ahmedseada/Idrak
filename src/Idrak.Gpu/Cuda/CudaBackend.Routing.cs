// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Cuda;

// The mixture-of-experts operations (plan 37, R0c; PtxKernels.Routing.cs) on the device, as Vulkan's and HIP's:
// - the routing in four launches: each token's choices and weights, each expert's count, the offsets, then the rows
//   grouped by expert;
// - the experts' gated pairs and products in one launch per PtxKernels.ExpertsPerLaunch experts (plan 37, R0d). Each
//   launch walks its experts on the device, each expert's rows read from the offsets where they are.
// An empty expert therefore costs a few reads, not a launch. The grid is the device's resident threads at most (its
// multiprocessors × the threads each holds, as reported), fewer for few outputs; the host never learns the counts, and
// each expert's outputs go grid-stride.
// Nothing is read back, so a mixture-of-experts layer routes here (Backend.RoutesExperts) and a decoding step through it
// is captured in a CUDA graph, the experts' addresses captured with the launches' parameters. Shapes past the kernels'
// 32-bit indices take the host fallback.
internal sealed unsafe partial class CudaBackend
{
    // Each slot's chosen expert (and, while route_choose_f32 runs, each token's list) and each expert's slot count between
    // the routing's launches (kept in stream order, as the split sums' partials are).
    private readonly KeptScratch _routeChoices = new(), _routeCounts = new();

    public override void RouteTopKKernel(Storage probabilities, Storage weights, Storage offsets, Storage tokenOfRow, Storage slotRow, int tokens, int experts, int k,
        bool normalize)
    {
        long slots = (long)tokens * k;
        if (tokens < 0 || experts <= 0 || k <= 0 || k > experts || (long)tokens * experts > int.MaxValue || slots > int.MaxValue)
        {
            base.RouteTopKKernel(probabilities, weights, offsets, tokenOfRow, slotRow, tokens, experts, k, normalize);
            return;
        }

        var choices = _routeChoices.Get(this, (int)Math.Max(1, slots));
        var counts = _routeCounts.Get(this, experts);
        Launch1D(K("route_choose_f32"), tokens, P(probabilities), P(weights), P(choices), U(experts), U(k), U(normalize ? 1 : 0), U(tokens));
        Launch1D(K("route_counts_f32"), experts, P(choices), P(counts), U((int)slots), U(experts));
        Launch1D(K("route_offsets_f32"), 1, P(counts), P(offsets), U((int)slots), U(experts), U(1));
        Launch1D(K("route_place_f32"), experts, P(choices), P(offsets), P(tokenOfRow), P(slotRow), U((int)slots), U(k), U(experts));
    }

    public override void ExpertGatedPairKernel(Storage x, Storage offsets, ReadOnlySpan<(Storage Gate, Storage? GateBias, Storage Up, Storage? UpBias)> experts, Storage hidden,
        int rows, int k, int n, int activation)
    {
        long outputs = (long)rows * n;
        if (outputs <= 0)
        {
            return;                                                                              // no rows: nothing to write
        }

        if (k < 0 || activation is < 0 or > 3 || !ExpertFits(outputs, rows, k, n))
        {
            base.ExpertGatedPairKernel(x, offsets, experts, hidden, rows, k, n, activation);
            return;
        }

        // x, offsets, hidden; first, experts, k, cols, kind; then each expert's gate, up, gate bias and up bias (the slots
        // past the launch's experts 0).
        var f = K("expert_gated_f32");
        Span<ulong> values = stackalloc ulong[8 + 4 * PtxKernels.ExpertsPerLaunch];
        for (int first = 0; first < experts.Length; first += PtxKernels.ExpertsPerLaunch)
        {
            int count = Math.Min(PtxKernels.ExpertsPerLaunch, experts.Length - first);
            values.Clear();
            (values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]) = (P(x), P(offsets), P(hidden), U(first), U(count), U(k), U(n), U(activation));
            for (int j = 0; j < count; j++)
            {
                var (gate, gateBias, up, upBias) = experts[first + j];
                (values[8 + 4 * j], values[9 + 4 * j], values[10 + 4 * j], values[11 + 4 * j]) =
                    (P(gate), P(up), gateBias is null ? 0UL : P(gateBias), upBias is null ? 0UL : P(upBias));
            }

            Launch(f, ExpertBlocks(outputs), 1, (uint)_shapes.BlockSize, 1, values);
        }
    }

    public override void ExpertMatMulKernel(Storage x, Storage offsets, ReadOnlySpan<(Storage Weight, Storage? Bias)> experts, Storage y, int rows, int k, int n)
    {
        long outputs = (long)rows * n;
        if (outputs <= 0)
        {
            return;
        }

        if (k < 0 || !ExpertFits(outputs, rows, k, n))
        {
            base.ExpertMatMulKernel(x, offsets, experts, y, rows, k, n);
            return;
        }

        // x, offsets, y; first, experts, k, cols; then each expert's weight and bias (the slots past the launch's experts 0).
        var f = K("expert_matmul_f32");
        Span<ulong> values = stackalloc ulong[7 + 2 * PtxKernels.ExpertsPerLaunch];
        for (int first = 0; first < experts.Length; first += PtxKernels.ExpertsPerLaunch)
        {
            int count = Math.Min(PtxKernels.ExpertsPerLaunch, experts.Length - first);
            values.Clear();
            (values[0], values[1], values[2], values[3], values[4], values[5], values[6]) = (P(x), P(offsets), P(y), U(first), U(count), U(k), U(n));
            for (int j = 0; j < count; j++)
            {
                var (weight, bias) = experts[first + j];
                (values[7 + 2 * j], values[8 + 2 * j]) = (P(weight), bias is null ? 0UL : P(bias));
            }

            Launch(f, ExpertBlocks(outputs), 1, (uint)_shapes.BlockSize, 1, values);
        }
    }

    // An expert launch's blocks: one an output's block when one expert holds every row, at most the blocks the device
    // holds resident at once (each thread then walks its outputs grid-stride).
    private uint ExpertBlocks(long outputs)
    {
        long resident = Math.Max(1L, (long)_limits.Multiprocessors * _limits.MaxThreadsPerMultiprocessor / _shapes.BlockSize);
        return (uint)Math.Clamp((outputs + _shapes.BlockSize - 1) / _shapes.BlockSize, 1, Math.Min(resident, int.MaxValue));
    }

    // Whether the expert kernels' 32-bit element indices hold every buffer: the outputs, x's and a weight's elements.
    private static bool ExpertFits(long outputs, int rows, int k, int n) =>
        outputs <= int.MaxValue && (long)rows * k <= int.MaxValue && (long)k * n <= int.MaxValue;
}
