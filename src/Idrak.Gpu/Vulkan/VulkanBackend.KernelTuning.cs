// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;

namespace Idrak.Gpu.Vulkan;

// Choices no reported limit decides (how many ways to split k or the cached positions, which kernel variant of a
// product, a row kernel's narrow or wide form) are measured on the device in use: the first time a shape needs one,
// every candidate runs on the real inputs (writing scratch copies of the outputs), back to back and in turns over five
// rounds, and the candidate with the lowest median time is kept. A formula from the device's limits is the default: it
// is used while measuring is impossible (IDRAK_AUTOTUNE=0, or one measurement inside another), and a candidate
// replaces it only when clearly faster (TuneMargin, 3%), so timing noise does not move the choice. Choices are kept for
// the process and in the runtime's cache file (TuningCacheFile, per device and driver UUID and driver version, and per
// kernel width and reductions), so later processes skip the measuring.

/// <summary>Which choice a measurement is for.</summary>
internal enum VulkanTuneOp : byte
{
    /// <summary>A packed product: words per row of k and splits of k (candidate = splits · 8 + word variant).</summary>
    Gemv,

    /// <summary>Decoding attention: splits of the cached positions.</summary>
    Attention,

    /// <summary>A float32 product: the small, tiled or register-blocked kernel, or the cooperative-matrix one.</summary>
    MatMul,

    /// <summary>A row kernel: one invocation per row (1) or a workgroup per row (0).</summary>
    Rows,

    /// <summary>A prompt-sized packed product: the tiled packed kernel (at a width), the few-rows kernels, expanding the weights or the cooperative-matrix kernel.</summary>
    PackedPrompt,

    /// <summary>Attention over many query rows: the tiled kernel (at a width) or the decoding kernel.</summary>
    TiledAttention,

    /// <summary>A two-pass reduction (column sums, group statistics): the splits of each column or group.</summary>
    Splits,

    /// <summary>A fused packed product (several sharing an input, the gate/up pair, the gated down projection): as <see cref="Gemv"/>.</summary>
    FusedGemv,

    /// <summary>A kernel with subgroup operations: the subgroup size its pipeline requires (0: the device's default).</summary>
    SubgroupSize,

    /// <summary>A float32 product in reduced precision (MixedPrecision): the float32 choice (0) or the single-pass cooperative-matrix kernel (1).</summary>
    MixedMatMul,

    /// <summary>A prompt-sized packed product in reduced precision (MixedPrecision): the float32 choice (0) or the single-pass cooperative-matrix kernel (1).</summary>
    MixedPackedPrompt,

    /// <summary>Attention either way (Backend.PrefersComposedAttention): AttentionSpans (0) or the composed scores (1).</summary>
    AttentionPath,

    /// <summary>Attention over key ranges (attention_spans): the workgroup width.</summary>
    SpanAttention,
}

/// <summary>What a measured choice is for: the operation, its variant (kernel, format) and its shape.</summary>
internal readonly record struct VulkanTuneKey(VulkanTuneOp Op, int Variant, int A, int B, int C, int D = 0, int E = 0, int F = 0);

internal sealed unsafe partial class VulkanBackend
{
    /// <summary>Measure device-dependent choices on the device (default); off (IDRAK_AUTOTUNE=0): the formulas only.</summary>
    internal static bool Autotune = Environment.GetEnvironmentVariable("IDRAK_AUTOTUNE") is not ("0" or "false");

    /// <summary>
    /// Longest a shape's measurement may take, in milliseconds, estimated from one run of the formula's choice (a policy
    /// for how long the first use of a shape may stall, not a property of any device): beyond it the formula is kept.
    /// </summary>
    private const double TuneBudget = 500;

    // Bumped when kernels change what their variants mean, so stored choices of an older version are not used.
    private const int TuningVersion = 1;

    private readonly Dictionary<VulkanTuneKey, int> _tuned = [];
    private bool _tuningLoaded;

    // While candidates are timed, choices they need themselves use what is already known (or the formula), so one
    // measurement never starts another inside its timed region.
    [ThreadStatic]
    private static bool t_timing;

    private long _measurements;

    // Whether candidates can be timed now: measuring is on, not inside another measurement, and no graph is being recorded
    // (nothing recorded runs; the step before the recording measures its shapes).
    private bool CanTune => Autotune && !t_timing && _capture is null;

    /// <summary>Choices measured by this backend (tests: stored choices are not measured again).</summary>
    internal long Measurements => Interlocked.Read(ref _measurements);

    /// <summary>Choices known on this device (for tests and diagnostics).</summary>
    internal int TunedCount
    {
        get
        {
            lock (_tuned)
            {
                return _tuned.Count;
            }
        }
    }

    /// <summary>Forgets the choices known in this process (tests); the stored file is left as is.</summary>
    internal void ForgetTuning()
    {
        lock (_tuned)
        {
            _tuned.Clear();
            _tuningLoaded = true;
        }
    }

    // The choice for `key` if known (measured in this process or stored by an earlier one).
    private bool TryTuned(VulkanTuneKey key, out int value)
    {
        lock (_tuned)
        {
            if (!_tuningLoaded)
            {
                _tuningLoaded = true;
                LoadTuning();
            }

            return _tuned.TryGetValue(key, out value);
        }
    }

    // The fastest of `candidates` for `key`, measured now (callers look up known choices first, and measure again when a
    // stored one no longer fits); `fallback` (the formula's choice) while it cannot be measured. `run(candidate)` runs the
    // operation with that choice writing only scratch storage.
    private int Tune(VulkanTuneKey key, ReadOnlySpan<int> candidates, int fallback, Action<int> run)
    {
        if (candidates.Length <= 1)
        {
            return candidates.Length == 1 ? candidates[0] : fallback;
        }

        if (!CanTune)
        {
            return fallback;
        }

        // The formula's choice first, once untimed (builds its pipelines) and once timed: when measuring every candidate
        // would take longer than TuneBudget (a slow device, a large shape), the formula's choice is kept (and stored).
        Interlocked.Increment(ref _measurements);
        t_timing = true;
        var list = candidates.ToArray();
        double[] medians;
        try
        {
            int formulaAt = Array.IndexOf(list, fallback);
            if (formulaAt >= 0)
            {
                run(fallback);
                double first = TimeRuns(fallback, 1, run);
                if (first * list.Length * 8 > TuneBudget)
                {
                    lock (_tuned)
                    {
                        _tuned[key] = fallback;
                    }

                    SaveTuning(key, fallback);
                    return fallback;
                }
            }

            // A first run of each candidate outside the timing (builds its pipelines and settles the choices it needs),
            // then one timed run each; candidates over 1.5 times the fastest are dropped.
            var once = new double[list.Length];
            for (int c = 0; c < list.Length; c++)
            {
                if (c != formulaAt)
                {
                    run(list[c]);
                }

                once[c] = TimeRuns(list[c], 1, run);
            }

            double best = once.Min();
            var kept = Enumerable.Range(0, list.Length).Where(c => once[c] <= 1.5 * best || c == formulaAt).ToArray();

            // Each remaining candidate is timed as it runs in practice, several dispatches back to back (as many as take
            // about 0.5 ms, at most 32); candidates take turns, five rounds, so a device still raising its clocks slows
            // every candidate alike; the median of the five is kept.
            var repeats = kept.Select(c => Math.Clamp((int)Math.Ceiling(0.5 / Math.Max(once[c], 1e-3)), 1, 32)).ToArray();
            var times = kept.Select(_ => new double[5]).ToArray();
            for (int round = 0; round < 5; round++)
            {
                for (int i = 0; i < kept.Length; i++)
                {
                    times[i][round] = TimeRuns(list[kept[i]], repeats[i], run) / repeats[i];
                }
            }

            medians = new double[list.Length];
            Array.Fill(medians, double.MaxValue);
            for (int i = 0; i < kept.Length; i++)
            {
                medians[kept[i]] = times[i].Order().ElementAt(2);
            }
        }
        finally
        {
            t_timing = false;
        }

        int fastest = 0;
        for (int c = 1; c < medians.Length; c++)
        {
            if (medians[c] < medians[fastest])
            {
                fastest = c;
            }
        }

        int chosen = list[fastest];
        int formula = Array.IndexOf(list, fallback);
        if (formula >= 0 && medians[fastest] >= medians[formula] * TuneMargin)
        {
            chosen = fallback;                                             // within the noise of the formula's choice
        }

        lock (_tuned)
        {
            _tuned[key] = chosen;
        }

        SaveTuning(key, chosen);
        return chosen;
    }

    // Milliseconds of `count` back-to-back runs of one candidate: the queue drained before and after, timed on the host.
    private double TimeRuns(int candidate, int count, Action<int> run)
    {
        Synchronize();
        var watch = Stopwatch.StartNew();
        for (int i = 0; i < count; i++)
        {
            run(candidate);
        }

        Synchronize();
        return watch.Elapsed.TotalMilliseconds;
    }

    // Candidate counts 1, 2, 4, … up to `max`.
    private static int[] PowersOfTwo(int max)
    {
        var values = new List<int>();
        for (int s = 1; s <= Math.Max(1, max); s *= 2)
        {
            values.Add(s);
        }

        return [.. values];
    }

    // Storages of `lengths` floats for candidates to write (none where a length is 0), released after `body`.
    private void WithScratch(ReadOnlySpan<int> lengths, Action<Storage[]> body)
    {
        var scratch = new Storage[lengths.Length];
        try
        {
            for (int i = 0; i < scratch.Length; i++)
            {
                scratch[i] = lengths[i] > 0 ? Allocate(lengths[i], zeroed: true) : null!;
            }

            body(scratch);
        }
        finally
        {
            foreach (var s in scratch)
            {
                s?.Release();
            }
        }
    }

    // ------------------------------------------------------------------ the stored choices

    // The prefix of this device's kernel choices in the cache file: apart from the runtime's (which start with the
    // device key), per device and driver, kernel width, reductions, the cooperative-matrix shape where the products use
    // one (their candidates join the measurement, so choices stored without them are measured again) and version of the
    // kernels' variants.
    private string KernelTuningPrefix() =>
        $"kernels/{TuningKey()}/w{Width}/{(Limits.SubgroupArithmetic ? "subgroups" : "plain")}/{(MatrixShape is { } shape ? $"coop{shape}/" : "")}v{TuningVersion}/";

    // Reads the stored kernel choices ("prefix op variant a b c d e f", tab, value, tab, device name). Unreadable files
    // are ignored. Called with _tuned locked.
    private void LoadTuning()
    {
        try
        {
            if (TuningCacheFile is not { } path || !File.Exists(path))
            {
                return;
            }

            string prefix = KernelTuningPrefix();
            foreach (string line in File.ReadLines(path))
            {
                string[] fields = line.Split('\t');
                if (fields.Length < 2 || !fields[0].StartsWith(prefix, StringComparison.Ordinal))
                {
                    continue;
                }

                string[] parts = fields[0][prefix.Length..].Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length != 8 || !Enum.TryParse(parts[0], out VulkanTuneOp op) || !int.TryParse(fields[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int value))
                {
                    continue;
                }

                var numbers = new int[7];
                bool ok = true;
                for (int i = 0; i < 7; i++)
                {
                    ok &= int.TryParse(parts[i + 1], NumberStyles.Integer, CultureInfo.InvariantCulture, out numbers[i]);
                }

                if (ok)
                {
                    _tuned[new VulkanTuneKey(op, numbers[0], numbers[1], numbers[2], numbers[3], numbers[4], numbers[5], numbers[6])] = value;
                }
            }
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            // Measured again.
        }
    }

    // Adds one choice to the cache file (a line appended; the runtime's rewrite of its own lines keeps it).
    private void SaveTuning(VulkanTuneKey key, int value)
    {
        try
        {
            if (TuningCacheFile is not { } path)
            {
                return;
            }

            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
            File.AppendAllText(path, string.Create(CultureInfo.InvariantCulture,
                $"{KernelTuningPrefix()}{key.Op} {key.Variant} {key.A} {key.B} {key.C} {key.D} {key.E} {key.F}\t{value}\t{_physical.DeviceName}\n"));
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            // Kept in memory only.
        }
    }
}
