// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Cuda;

// The mixture-of-experts operations (plan 37, R0c; PtxKernels.Routing.cs) on the device, as Vulkan's and HIP's:
// - the routing in four launches: each token's choices and weights, each expert's count, the offsets, then the rows
//   grouped by expert;
// - each expert's gated pair and product in one launch per expert, its rows read from the offsets where they are.
// Nothing is read back, so a mixture-of-experts layer routes here (Backend.RoutesExperts) and a decoding step through it
// is captured in a CUDA graph. An expert's launch covers every row, because the host never learns the counts; the threads
// past its own rows leave at once. Shapes past the kernels' 32-bit indices take the host fallback.
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

        var f = K("expert_gated_f32");
        for (int e = 0; e < experts.Length; e++)
        {
            var (gate, gateBias, up, upBias) = experts[e];
            Launch1D(f, (int)outputs, P(x), P(offsets), P(gate), gateBias is null ? 0UL : P(gateBias), P(up), upBias is null ? 0UL : P(upBias), P(hidden),
                U(e), U(k), U(n), U(activation), U((int)outputs));
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

        var f = K("expert_matmul_f32");
        for (int e = 0; e < experts.Length; e++)
        {
            var (weight, bias) = experts[e];
            Launch1D(f, (int)outputs, P(x), P(offsets), P(weight), bias is null ? 0UL : P(bias), P(y), U(e), U(k), U(n), U((int)outputs));
        }
    }

    // Whether the expert kernels' 32-bit element indices hold every buffer: the outputs, x's and a weight's elements.
    private static bool ExpertFits(long outputs, int rows, int k, int n) =>
        outputs <= int.MaxValue && (long)rows * k <= int.MaxValue && (long)k * n <= int.MaxValue;
}
