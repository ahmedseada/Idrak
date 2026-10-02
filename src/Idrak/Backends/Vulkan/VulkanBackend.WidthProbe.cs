// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Backends.Vulkan;

// The kernels' workgroup width, measured. The formula (VulkanKernels.WidthFor: eight subgroups) is the most a device's
// limits allow, not what runs fastest: on an Adreno GPU (subgroups of 64-128, so width 1024) the float32 product ran
// 1024³ in 798 ms at width 1024 against 25 ms at 256, and the tuning, which keeps the formula's choice when measuring
// every candidate would take too long, kept it; a later, larger dispatch then ran long enough for the driver to reset the
// device (VK_ERROR_DEVICE_LOST). So at start a small product and one-row decoding work run at each width from a
// subgroup's worth (at least MinWidth) up to the formula's, and the fastest width is kept (the formula's unless another is
// at least WidthMargin faster), stored per device and driver like the runtime policy. Timing the product alone chose 512
// on that Adreno, where chat ran 13 tokens/s against 15.8 at 256.
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

        string key = $"width3/{formula}";
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
            _kernels.Clear();                                              // kernels by name are built at the width
            lock (_tuned)
            {
                _tuned.Clear();                                            // stored choices are per width: read again
                _tuningLoaded = false;
            }
        }
    }

    // Times each candidate width on work of both kinds a model runs: prompt-shaped products (a 256³ float32 product, an
    // int8 product of 64 rows 1024 -> 3072) and decoding-shaped work of one row (int8 products 1024 -> 3072 and 3072 -> 1024, an RMS norm of 1024,
    // a softmax of 32768), eight back to back as in a decoding step. Each part is compared with its fastest width and the
    // geometric mean of those ratios ranks the widths, so neither kind outweighs the other by its length. The formula's
    // width is kept unless another is clearly faster (WidthMargin).
    private int MeasureWidth(List<int> candidates, int formula)
    {
        const int Size = 256, Dim = 1024, Wide = 3072, Classes = 32768, Prompt = 64;
        int[] lengths = [Size * Size, Size * Size, Size * Size, Wide, Wide * Dim / 4, Wide, Dim, Dim, Classes, Prompt * Dim, Prompt * Wide];
        var s = new Storage[lengths.Length];
        int original = Limits.Width;
        bool timing = t_timing;
        t_timing = true;
        try
        {
            for (int i = 0; i < s.Length; i++)
            {
                s[i] = Allocate(lengths[i], zeroed: true);
            }

            var (a, b, c, x, q, scales, y, gain, logits) = (s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8]);
            var (rows, wide) = (s[9], s[10]);

            // The operations as a model calls them (each picks its kernel at the width as it would in use), so a kernel
            // that does not fit a width is not timed as if it took no time.
            var parts = new List<Action>
            {
                () => BatchedMatMul(a, b, c, 1, Size, Size, Size, false, false, 0f),
                () => Int8MatMul(rows, q, scales, wide, Prompt, Wide, Dim),
            };

            void Eight(Action step)
            {
                for (int r = 0; r < 8; r++)
                {
                    step();
                }
            }

            parts.Add(() => Eight(() => Int8MatMul(gain, q, scales, x, 1, Wide, Dim)));
            parts.Add(() => Eight(() => Int8MatMul(x, q, scales, gain, 1, Dim, Wide)));
            parts.Add(() => Eight(() => RmsNormAffine(gain, y, gain, 1, Dim, 1e-6f, 0f)));
            parts.Add(() => Eight(() => Softmax(logits, logits, 1, Classes, false)));

            var times = new double[candidates.Count, parts.Count];
            for (int i = 0; i < candidates.Count; i++)
            {
                SetWidth(candidates[i]);
                for (int p = 0; p < parts.Count; p++)
                {
                    var part = parts[p];
                    part();                                                // builds the pipelines
                    var runs = new double[3];
                    for (int r = 0; r < runs.Length; r++)
                    {
                        runs[r] = TimeRuns(0, 1, _ => part());
                    }

                    Array.Sort(runs);
                    times[i, p] = Math.Max(runs[1], 1e-6);
                }
            }

            var scores = new double[candidates.Count];
            for (int p = 0; p < parts.Count; p++)
            {
                double best = double.MaxValue;
                for (int i = 0; i < candidates.Count; i++)
                {
                    best = Math.Min(best, times[i, p]);
                }

                for (int i = 0; i < candidates.Count; i++)
                {
                    scores[i] += Math.Log(times[i, p] / best) / parts.Count;
                }
            }

            int chosen = 0;
            for (int i = 1; i < scores.Length; i++)
            {
                if (scores[i] < scores[chosen])
                {
                    chosen = i;
                }
            }

            int formulaAt = candidates.IndexOf(formula);
            return formulaAt >= 0 && Math.Exp(scores[chosen]) >= Math.Exp(scores[formulaAt]) * WidthMargin ? formula : candidates[chosen];
        }
        finally
        {
            t_timing = timing;
            SetWidth(original);
            foreach (var storage in s)
            {
                storage?.Release();
            }
        }
    }
}
