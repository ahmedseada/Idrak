// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// The mixture-of-experts operations (plan 37, R0; VulkanKernels.Routing.cs) on the device:
// - the routing in four dispatches: each token's choices and weights, each expert's count, the offsets, then the rows
//   grouped by expert;
// - each expert's gated pair and product in one indirect dispatch per expert (R0d). Its rows are read from the offsets
//   where they are, and its workgroups come from expert_sizes, run first on the same offsets. So an expert without rows
//   dispatches no workgroup, and one with few rows only the blocks it has.
// The sizes are made by each operation from its own offsets, not kept from the routing: the host routing (option B) gives
// the experts offsets it uploaded, and two operations share no state. Nothing is read back, so a mixture-of-experts layer
// routes here (Backend.RoutesExperts) and a decoding step through it is recorded. A storage past one binding takes the
// host fallback, as every operation here does.
internal sealed partial class VulkanBackend
{
    // Each slot's chosen expert and each expert's slot count between the routing's dispatches (kept in queue order, as the
    // split sums' partials are).
    private readonly KeptScratch _routeChoices = new(), _routeCounts = new();

    // Each expert's dispatch (three words) between expert_sizes and the experts' indirect dispatches, kept in queue order.
    private readonly KeptScratch _expertDispatches = new();

    public override void RouteTopKKernel(Storage probabilities, Storage weights, Storage offsets, Storage tokenOfRow, Storage slotRow, int tokens, int experts, int k,
        bool normalize)
    {
        int slots = tokens * k;
        var choices = _routeChoices.Get(this, Math.Max(1, slots));
        var counts = _routeCounts.Get(this, Math.Max(1, experts));
        if (k > VulkanKernels.MaxRouteTopK || !Fit(probabilities, weights, offsets, tokenOfRow, slotRow, choices, counts))
        {
            base.RouteTopKKernel(probabilities, weights, offsets, tokenOfRow, slotRow, tokens, experts, k, normalize);
            return;
        }

        Span<byte> choose = stackalloc byte[16];
        Grid("route_choose", tokens, [probabilities, weights, choices], new Push(choose).I(tokens).I(experts).I(k).B(normalize).Bytes);
        Span<byte> push = stackalloc byte[12];
        Grid("route_counts", experts, [choices, counts], new Push(push).I(slots).I(experts).Bytes);
        Run("route_offsets", 1, 1, 1, [counts, offsets], new Push(push).I(slots).I(experts).Bytes);
        Grid("route_place", experts, [choices, offsets, tokenOfRow, slotRow], new Push(push).I(slots).I(experts).I(k).Bytes);
    }

    public override void ExpertGatedPairKernel(Storage x, Storage offsets, ReadOnlySpan<(Storage Gate, Storage? GateBias, Storage Up, Storage? UpBias)> experts, Storage hidden,
        int rows, int k, int n, int activation)
    {
        bool fit = Fit(x, offsets, hidden);
        foreach (var (gate, gateBias, up, upBias) in experts)
        {
            fit = fit && Fit(gate, up) && (gateBias is null || Fit(gateBias)) && (upBias is null || Fit(upBias));
        }

        if (!fit)
        {
            base.ExpertGatedPairKernel(x, offsets, experts, hidden, rows, k, n, activation);
            return;
        }

        var dispatches = ExpertSizes(offsets, experts.Length);
        var kernel = Kernel("expert_gated_pair");
        Span<byte> push = stackalloc byte[24];
        for (int e = 0; e < experts.Length; e++)
        {
            var (gate, gateBias, up, upBias) = experts[e];
            // An absent bias binds x in its place, never read (its flag is off).
            DispatchIndirect(kernel, dispatches, 12L * e, [x, offsets, gate, gateBias ?? x, up, upBias ?? x, hidden],
                new Push(push).I(e).I(k).I(n).I(activation).B(gateBias is not null).B(upBias is not null).Bytes);
        }
    }

    public override void ExpertMatMulKernel(Storage x, Storage offsets, ReadOnlySpan<(Storage Weight, Storage? Bias)> experts, Storage y, int rows, int k, int n)
    {
        bool fit = Fit(x, offsets, y);
        foreach (var (weight, bias) in experts)
        {
            fit = fit && Fit(weight) && (bias is null || Fit(bias));
        }

        if (!fit)
        {
            base.ExpertMatMulKernel(x, offsets, experts, y, rows, k, n);
            return;
        }

        var dispatches = ExpertSizes(offsets, experts.Length);
        var kernel = Kernel("expert_matmul");
        Span<byte> push = stackalloc byte[16];
        for (int e = 0; e < experts.Length; e++)
        {
            var (weight, bias) = experts[e];
            DispatchIndirect(kernel, dispatches, 12L * e, [x, offsets, weight, bias ?? x, y], new Push(push).I(e).I(k).I(n).B(bias is not null).Bytes);
        }
    }

    // Each expert's dispatch from the offsets (expert_sizes: its row blocks, at most what the device takes; the kernels walk
    // the blocks past that), in a kept storage the experts' indirect dispatches read.
    private Storage ExpertSizes(Storage offsets, int experts)
    {
        var dispatches = _expertDispatches.Get(this, 3 * Math.Max(1, experts));
        Span<byte> push = stackalloc byte[8];
        Grid("expert_sizes", experts, [offsets, dispatches], new Push(push).I(experts).I((int)Math.Min(Limits.MaxGroupsX, (uint)int.MaxValue)).Bytes);
        return dispatches;
    }
}
