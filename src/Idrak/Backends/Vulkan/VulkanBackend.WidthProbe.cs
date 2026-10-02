// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Backends.Vulkan;

// The kernels' workgroup width, measured. The formula (VulkanKernels.WidthFor: eight subgroups) is the most a device's
// limits allow, not what runs fastest: on an Adreno GPU (subgroups of 64-128, so width 1024) the float32 product ran
// 1024³ in 798 ms at width 1024 against 25 ms at 256, and the tuning, which keeps the formula's choice when measuring
// every candidate would take too long, kept it; a later, larger dispatch then ran long enough for the driver to reset the
// device (VK_ERROR_DEVICE_LOST). So at start the product's kernels run a small shape at each width from a subgroup's
// worth (at least MinWidth) up to the formula's, and the fastest width is kept (the formula's unless another is at least
// WidthMargin faster), stored per device and driver like the runtime policy.
internal sealed partial class VulkanBackend
{
    /// <summary>A width must be at least this much faster (10%) than the formula's to replace it.</summary>
    private const double WidthMargin = 0.9;

    /// <summary>How the width was chosen: "formula" (one candidate), "measured", "cached" or "override".</summary>
    public string WidthChoice { get; private set; } = "formula";

    /// <summary>Tests: false keeps the formula's width without measuring (IDRAK_VULKAN_WIDTH_PROBE=0 for a process).</summary>
    internal static bool WidthProbe { get; set; } = Environment.GetEnvironmentVariable("IDRAK_VULKAN_WIDTH_PROBE") is not ("0" or "false");

    /// <summary>Tests: the widest width to measure up to instead of the formula's (within the device's limits), so a device
    /// whose formula leaves one candidate still measures several.</summary>
    internal static int? ProbeFormulaOverride { get; set; }

    private void ChooseWidth()
    {
        if (DeviceLimits.WidthOverride is not null)
        {
            WidthChoice = "override";
            return;
        }

        int formula = Limits.Width;
        if (ProbeFormulaOverride is int wider && wider > formula && (wider & (wider - 1)) == 0 && wider <= VulkanKernels.MaxWidth
            && wider <= Limits.MaxInvocations && wider <= Limits.MaxSizeX && VulkanKernels.SharedBytesBound(wider) <= Limits.SharedBytes)
        {
            SetWidth(wider);
            formula = wider;
        }

        var candidates = new List<int>();
        for (int w = Math.Max(VulkanKernels.MinWidth, Limits.SubgroupSize); w <= formula; w *= 2)
        {
            candidates.Add(w);
        }

        if (!WidthProbe || candidates.Count <= 1 || KernelsOff)
        {
            return;
        }

        string key = $"width/{formula}";
        var cached = CachedChoices();
        if (cached.TryGetValue(key, out string? text) && int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int stored)
            && candidates.Contains(stored))
        {
            SetWidth(stored);
            WidthChoice = "cached";
            return;
        }

        int chosen = MeasureWidth(candidates, formula);
        CacheChoices(new Dictionary<string, string> { [key] = chosen.ToString(CultureInfo.InvariantCulture) });
        SetWidth(chosen);
        WidthChoice = "measured";
    }

    private void SetWidth(int width)
    {
        if (width != Limits.Width)
        {
            _limits = Limits with { Width = width };
            _candidateWidths = null;
        }
    }

    // Seconds of the float32 product's three kernels (one invocation per output, tiled, register-blocked) on a 256³ shape
    // at each width, the median of three runs each; the width with the least total.
    private int MeasureWidth(List<int> candidates, int formula)
    {
        const int Size = 256;
        var a = Allocate(Size * Size, zeroed: true);
        var b = Allocate(Size * Size, zeroed: true);
        var c = Allocate(Size * Size, zeroed: true);
        bool timing = t_timing;
        t_timing = true;
        try
        {
            Span<byte> bytes = stackalloc byte[28];
            var push = new Push(bytes).I(1).I(Size).I(Size).I(Size).B(false).B(false).F(0f).Bytes.ToArray();
            var totals = new double[candidates.Count];
            for (int i = 0; i < candidates.Count; i++)
            {
                int width = candidates[i];
                foreach (int variant in new[] { MatSmall, MatTiled, MatBlocked })
                {
                    int edge = VulkanKernels.MatSide(width) * (variant == MatBlocked ? VulkanKernels.MatPer : 1);
                    if (variant != MatSmall && !MatFits(edge, Size, Size))
                    {
                        continue;
                    }

                    int choice = WithWidth(width, variant);
                    RunMatMul(choice, a, b, c, 1, Size, Size, push);   // builds the pipeline
                    var runs = new double[3];
                    for (int r = 0; r < runs.Length; r++)
                    {
                        runs[r] = TimeRuns(choice, 1, ch => RunMatMul(ch, a, b, c, 1, Size, Size, push));
                    }

                    Array.Sort(runs);
                    totals[i] += runs[1];
                }
            }

            int best = 0;
            for (int i = 1; i < totals.Length; i++)
            {
                if (totals[i] < totals[best])
                {
                    best = i;
                }
            }

            int formulaAt = candidates.IndexOf(formula);
            return formulaAt >= 0 && totals[best] >= totals[formulaAt] * WidthMargin ? formula : candidates[best];
        }
        finally
        {
            t_timing = timing;
            a.Release();
            b.Release();
            c.Release();
        }
    }
}
