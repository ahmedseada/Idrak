// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Hip;

// The mixture-of-experts operations (plan 37, R0b; GcnKernels.Routing.cs) on the device, as Vulkan's (VulkanBackend.Routing.cs):
// - the routing in four launches: each token's choices and weights, each expert's count, the offsets, then the rows
//   grouped by expert;
// - the experts' gated pairs and products in one launch per GcnKernels.ExpertsPerLaunch experts (plan 37, R0d). Each
//   launch walks its experts on the device, each expert's rows read from the offsets where they are.
// HIP has no launch whose size the device writes, so an expert without rows costs a few reads in a launch shared with
// the others, not a launch of its own. Its grid is an element-wise launch's for every row: the host never learns the
// counts, and each expert's outputs go grid-stride.
// Nothing is read back, so a mixture-of-experts layer routes here (Backend.RoutesExperts) and a decoding step through it
// is recorded. The host fallback takes over when the kernels are unavailable or a buffer holds 2^30 floats or more (the
// kernels' 32-bit byte offsets).
internal sealed unsafe partial class HipBackend
{
    // Each slot's chosen expert (and, while route_choose runs, each token's list) and each expert's slot count between the
    // routing's launches (kept in queue order, as the split sums' partials are).
    private readonly KeptScratch _routeChoices = new(), _routeCounts = new();

    /// <summary>The cost of a routing kernel per unit (the table at the top of GcnKernels.Routing.cs); (0, 0) for another.</summary>
    private static (double Flops, double Bytes) RoutingCost(string kernel) => kernel switch
    {
        "route_choose_f32" => (2, 16),                                                           // per token × expert
        "route_counts_f32" or "route_place_f32" => (0, 4),                                       // per expert × slot
        "route_offsets_f32" => (0, 8),                                                           // per expert
        "expert_gated_silu_f32" or "expert_gated_gelu_f32" or "expert_gated_relu_f32" or "expert_gated_tanh_f32" => (4, 16), // per output × k
        "expert_matmul_f32" => (2, 8),
        _ => (0, 0),
    };

    public override void RouteTopKKernel(Storage probabilities, Storage weights, Storage offsets, Storage tokenOfRow, Storage slotRow, int tokens, int experts, int k,
        bool normalize)
    {
        long slots = (long)tokens * k;
        if (tokens < 0 || experts <= 0 || k <= 0 || k > experts || !ShapeFits((long)tokens * experts, slots, experts + 1L)
            || Gcn("route_choose_f32") is not { } choose || Gcn("route_counts_f32") is not { } count || Gcn("route_offsets_f32") is not { } offset
            || Gcn("route_place_f32") is not { } place)
        {
            base.RouteTopKKernel(probabilities, weights, offsets, tokenOfRow, slotRow, tokens, experts, k, normalize);
            return;
        }

        var choices = _routeChoices.Get(this, (int)Math.Max(1, slots));
        var counts = _routeCounts.Get(this, experts);
        double routed = (double)slots * experts;
        LaunchShape(choose, tokens, Cost(choose.Name).Flops * tokens * (double)experts, P(probabilities), P(weights), P(choices), U(experts), U(k), U(normalize ? 1 : 0),
            U(tokens));
        LaunchShape(count, experts, Cost(count.Name).Flops * routed, P(choices), P(counts), U((int)slots), U(experts));
        LaunchShape(offset, 1, 0, P(counts), P(offsets), U((int)slots), U(experts), U(1));
        LaunchShape(place, experts, Cost(place.Name).Flops * routed, P(choices), P(offsets), P(tokenOfRow), P(slotRow), U((int)slots), U(k), U(experts));
    }

    public override void ExpertGatedPairKernel(Storage x, Storage offsets, ReadOnlySpan<(Storage Gate, Storage? GateBias, Storage Up, Storage? UpBias)> experts, Storage hidden,
        int rows, int k, int n, int activation)
    {
        long outputs = (long)rows * n;
        if (outputs <= 0)
        {
            return;                                                                              // no rows: nothing to write
        }

        if (k < 0 || activation is < 0 or > 3 || !ShapeFits(outputs, (long)rows * k, (long)k * n)
            || Gcn(GcnKernels.ExpertGatedKinds[activation]) is not { } f)
        {
            base.ExpertGatedPairKernel(x, offsets, experts, hidden, rows, k, n, activation);
            return;
        }

        // x, offsets, hidden; first, experts, k, n; then each expert's four addresses (the slots past the launch's experts 0).
        Span<ulong> values = stackalloc ulong[7 + 4 * GcnKernels.ExpertsPerLaunch];
        for (int first = 0; first < experts.Length; first += GcnKernels.ExpertsPerLaunch)
        {
            int count = Math.Min(GcnKernels.ExpertsPerLaunch, experts.Length - first);
            values.Clear();
            (values[0], values[1], values[2], values[3], values[4], values[5], values[6]) = (P(x), P(offsets), P(hidden), U(first), U(count), U(k), U(n));
            for (int j = 0; j < count; j++)
            {
                var (gate, gateBias, up, upBias) = experts[first + j];
                (values[7 + 4 * j], values[8 + 4 * j], values[9 + 4 * j], values[10 + 4 * j]) =
                    (P(gate), gateBias is null ? 0UL : P(gateBias), P(up), upBias is null ? 0UL : P(upBias));
            }

            LaunchShape(f, (int)outputs, Cost(f.Name).Flops * outputs * k * count / experts.Length, values);
        }
    }

    public override void ExpertMatMulKernel(Storage x, Storage offsets, ReadOnlySpan<(Storage Weight, Storage? Bias)> experts, Storage y, int rows, int k, int n)
    {
        long outputs = (long)rows * n;
        if (outputs <= 0)
        {
            return;
        }

        if (k < 0 || !ShapeFits(outputs, (long)rows * k, (long)k * n) || Gcn("expert_matmul_f32") is not { } f)
        {
            base.ExpertMatMulKernel(x, offsets, experts, y, rows, k, n);
            return;
        }

        // x, offsets, y; first, experts, k, n; then each expert's weight and bias (the slots past the launch's experts 0).
        Span<ulong> values = stackalloc ulong[7 + 2 * GcnKernels.ExpertsPerLaunch];
        for (int first = 0; first < experts.Length; first += GcnKernels.ExpertsPerLaunch)
        {
            int count = Math.Min(GcnKernels.ExpertsPerLaunch, experts.Length - first);
            values.Clear();
            (values[0], values[1], values[2], values[3], values[4], values[5], values[6]) = (P(x), P(offsets), P(y), U(first), U(count), U(k), U(n));
            for (int j = 0; j < count; j++)
            {
                var (weight, bias) = experts[first + j];
                (values[7 + 2 * j], values[8 + 2 * j]) = (P(weight), bias is null ? 0UL : P(bias));
            }

            LaunchShape(f, (int)outputs, Cost(f.Name).Flops * outputs * k * count / experts.Length, values);
        }
    }
}
