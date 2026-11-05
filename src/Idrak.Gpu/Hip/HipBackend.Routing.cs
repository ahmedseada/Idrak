// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Hip;

// The mixture-of-experts operations (plan 37, R0b; GcnKernels.Routing.cs) on the device, as Vulkan's (VulkanBackend.Routing.cs):
// the routing in four launches (each token's choices and weights, each expert's count, the offsets, then the rows grouped
// by expert), and each expert's gated pair and product in one launch per expert, its rows read from the offsets where
// they are. Nothing is read back, so a mixture-of-experts layer routes here (Backend.RoutesExperts) and a decoding step
// through it is recorded. An expert's launch covers every row (the host never learns the counts): the work-items past its
// own rows leave at once. The host fallback takes over when the kernels are unavailable or a buffer holds 2^30 floats or
// more (the kernels' 32-bit byte offsets).
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

        double flops = Cost(f.Name).Flops * outputs * k;
        for (int e = 0; e < experts.Length; e++)
        {
            var (gate, gateBias, up, upBias) = experts[e];
            LaunchShape(f, (int)outputs, flops, P(x), P(offsets), P(gate), gateBias is null ? 0UL : P(gateBias), P(up), upBias is null ? 0UL : P(upBias), P(hidden),
                U(e), U(k), U(n));
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

        double flops = Cost(f.Name).Flops * outputs * k;
        for (int e = 0; e < experts.Length; e++)
        {
            var (weight, bias) = experts[e];
            LaunchShape(f, (int)outputs, flops, P(x), P(offsets), P(weight), bias is null ? 0UL : P(bias), P(y), U(e), U(k), U(n));
        }
    }
}
