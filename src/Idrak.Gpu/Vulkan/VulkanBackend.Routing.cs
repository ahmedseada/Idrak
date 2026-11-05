// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// The mixture-of-experts operations (plan 37, R0; VulkanKernels.Routing.cs) on the device: the routing in four dispatches
// (each token's choices and weights, each expert's count, the offsets, then the rows grouped by expert), and each
// expert's gated pair and product in one
// dispatch per expert, its rows read from the offsets where they are. Nothing is read back, so a mixture-of-experts layer
// routes here (Backend.RoutesExperts) and a decoding step through it is recorded. A storage past one binding takes the
// host fallback, as every operation here does.
internal sealed partial class VulkanBackend
{
    // Each slot's chosen expert and each expert's slot count between the routing's dispatches (kept in queue order, as the
    // split sums' partials are).
    private readonly KeptScratch _routeChoices = new(), _routeCounts = new();

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

        uint groups = ExpertGroups(rows);
        Span<byte> push = stackalloc byte[24];
        for (int e = 0; e < experts.Length; e++)
        {
            var (gate, gateBias, up, upBias) = experts[e];
            // An absent bias binds x in its place, never read (its flag is off).
            Run("expert_gated_pair", groups, 1, 1, [x, offsets, gate, gateBias ?? x, up, upBias ?? x, hidden],
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

        uint groups = ExpertGroups(rows);
        Span<byte> push = stackalloc byte[16];
        for (int e = 0; e < experts.Length; e++)
        {
            var (weight, bias) = experts[e];
            Run("expert_matmul", groups, 1, 1, [x, offsets, weight, bias ?? x, y], new Push(push).I(e).I(k).I(n).B(bias is not null).Bytes);
        }
    }

    // Workgroups of an expert's dispatch: enough for one block of ExpertRows rows each when one expert holds every row
    // (the host never learns the counts), at most what the device takes; a workgroup past its expert's rows leaves at once.
    private uint ExpertGroups(int rows) => (uint)Math.Clamp(((long)rows + VulkanKernels.ExpertRows - 1) / VulkanKernels.ExpertRows, 1, Limits.MaxGroupsX);
}
