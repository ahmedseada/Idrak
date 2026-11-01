// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Gpu.Cuda;
using static Idrak.Gpu.Hip.HipRuntime;
using Weights = Idrak.Gpu.Hip.GcnKernels.PackedWeights;

namespace Idrak.Gpu.Hip;

// HIP's measured choices (plan 35, G7a), as CUDA's (CudaBackend.Tuning.cs, TuningCache.cs, CudaBackend.SharedTuning.cs):
// a choice that depends on the device is measured on the device in use the first time a shape needs it, never set from one
// card's benchmarks. Measured now: the packed few-row products' rows of k a step (G6a: 4 or 8) and k splits (1, 2, 4, …
// up to min(64, k / 64), and the formula's), one choice per format, kernel variant, rows, n and k (TuneOp.HipPackedProduct).
// Every candidate runs on the real inputs (it writes y, which the chosen one then writes again), is timed between two
// HIP events by CUDA's comparison (TuneTiming: a warm-up until the clocks settle, then pairs with the default's choice,
// medians, a 3% margin and a second confirmation), and the default (depth 4, the formula's splits) stays unless a
// candidate is clearly faster. The settings still decide: IDRAK_HIP_GEMV_DEPTH or IDRAK_HIP_GEMV_SPLITS, when given,
// leave only that value among the candidates (both given: nothing is measured); IDRAK_AUTOTUNE=0 keeps the defaults.
// Nothing is measured while a profile times the launches or the runtime lacks events. Choices are kept per device (its
// name, target, compute units, memory), runtime and driver, and library build with its GCN kernels, in the cache folder's
// tuning/hip/ (TuningCache's format; IDRAK_TUNING_CACHE=0 keeps none), and shared between a cluster's ranks
// (HipMeasuredChoices). Not measured yet: the finish form (IDRAK_HIP_FINISH), attention's splits and rows, the
// element-wise grids and the add-and-norm product (its tail is not safe to run twice); their settings and formulas decide.
internal sealed unsafe partial class HipBackend
{
    /// <summary>Measure device-dependent choices (default); off (IDRAK_AUTOTUNE=0): the defaults and formulas only.</summary>
    internal static bool Autotune { get; set; } = Environment.GetEnvironmentVariable("IDRAK_AUTOTUNE") is not ("0" or "false");

    // This backend's choices (null until first needed), or a cluster leader's it follows (HipMeasuredChoices).
    private HipTuningTable? _tuning;
    private HipBackend? _leader;
    private (IntPtr Start, IntPtr End) _tuneEvents;

    // While candidates are timed, the product they run takes the choice it is given (one measurement never starts another).
    [ThreadStatic]
    private static bool t_timing;

    /// <summary>The choices table of this backend: its own (read from its cache file when first needed) or its leader's.</summary>
    internal HipTuningTable Tuning
    {
        get
        {
            var table = Volatile.Read(ref _tuning);
            if (table is not null)
            {
                return table;
            }

            var made = new HipTuningTable(TuningIdentity, Autotune ? TuningCache.Folder(kind: "hip") : null);
            return Interlocked.CompareExchange(ref _tuning, made, null) ?? made;
        }
    }

    /// <summary>What this device's cache file belongs to: the device, the runtime and driver, the library build with its GCN kernels.</summary>
    internal TuningIdentity TuningIdentity => new(
        string.Create(CultureInfo.InvariantCulture, $"{Name}; {DeviceLimits.Architecture}; {DeviceLimits.ComputeUnits} CUs; {DeviceLimits.TotalMemory >> 20} MiB"),
        string.Create(CultureInfo.InvariantCulture, $"HIP runtime {FormatVersion(RuntimeVersion)}; driver {DriverVersion}"),
        TuningCache.LibraryBuild(HipKernelCache.KeyOf(GcnKernels.All, [], DeviceLimits.Architecture, "", "")));

    /// <summary>What this backend's measured choices apply to: the device's UUID (else its PCI address) and the cache file's identity.</summary>
    internal string ChoicesIdentity
    {
        get
        {
            var identity = TuningIdentity;
            string id = DeviceLimits.Uuid is { } uuid ? uuid.ToString("N") : DeviceLimits.PciAddress;
            return $"hip/{id}/{identity.Device}/{identity.Driver}/{identity.Library}";
        }
    }

    /// <summary>The backend whose choices this one runs (a cluster rank following rank 0), or null.</summary>
    internal HipBackend? Leader => _leader;

    /// <summary>Whether candidates can be timed now: autotuning on, not inside a measurement, no profile, the runtime's events there.</summary>
    private bool CanMeasure => Autotune && !t_timing && _profile is null && hipEventCreate != null && hipEventRecord != null && hipEventSynchronize != null
                               && hipEventElapsedTime != null;

    /// <summary>
    /// The packed few-row product's choice (rows of k a step, k splits) for its shape: a forced setting's, a kept or
    /// measured one, or the default (depth 4 or the forced one, the formula's splits or the forced count). Only the plain
    /// products measure: the fused forms take the default depth and the formula's splits, as before. `run(depth, splits)`
    /// runs the product with that choice into y (the chosen one then runs again).
    /// </summary>
    private (int Depth, int Splits) PackedChoice(Weights format, GcnKernels.PackedGemvVariant variant, int m, int n, int k, int columnBlocks, Action<int, int> run)
    {
        int formula = GemvSplitCount(DeviceLimits.ComputeUnits, columnBlocks, k);
        bool plain = variant == GcnKernels.PackedGemvVariant.Plain;
        int depth = plain ? ForcedGemvDepth ?? GcnKernels.PackedGemvDefaultDepth : GcnKernels.PackedGemvDefaultDepth;
        int splits = ForcedGemvSplits is int forced ? GemvSplitCount(DeviceLimits.ComputeUnits, columnBlocks, k, forced) : formula;
        if (!plain || t_timing || (ForcedGemvDepth is not null && ForcedGemvSplits is not null))
        {
            return (depth, splits);
        }

        // A depth whose kernel is not in the loaded set is never a candidate (it would time nothing and look fastest).
        int? depthOnly = ForcedGemvDepth ?? (Gcn(GcnKernels.PackedGemvKernel(format, m, 8)) is null ? GcnKernels.PackedGemvDefaultDepth : null);
        int[] candidates = PackedCandidates(depthOnly, ForcedGemvSplits is null ? null : splits, formula, k);
        int fallback = EncodePacked(depth, splits);
        var key = new TuneKey(TuneOp.HipPackedProduct, (int)format * 16 + (int)variant, m, n, k);
        int choice = Tune(key, candidates, fallback, c => run(c >> 8, c & 0xFF));
        return (choice >> 8, choice & 0xFF);
    }

    /// <summary>A packed product's choice as one number: <paramref name="depth"/> &lt;&lt; 8 | <paramref name="splits"/> (at most 255).</summary>
    internal static int EncodePacked(int depth, int splits) => depth << 8 | Math.Clamp(splits, 1, 255);

    /// <summary>
    /// The candidates of a packed product: each depth (<see cref="GcnKernels.PackedGemvDepths"/>, or the forced one) with
    /// each split count (1, 2, 4, … up to min(64, k / 64) and the formula's <paramref name="formula"/>, or the forced
    /// one), encoded by <see cref="EncodePacked"/>, in a stable order.
    /// </summary>
    internal static int[] PackedCandidates(int? forcedDepth, int? forcedSplits, int formula, int k)
    {
        var splitCounts = new List<int>();
        if (forcedSplits is int f)
        {
            splitCounts.Add(f);
        }
        else
        {
            for (int s = 1; s <= Math.Max(1, Math.Min(64, k / 64)); s *= 2)
            {
                splitCounts.Add(s);
            }

            if (!splitCounts.Contains(formula))
            {
                splitCounts.Add(formula);
                splitCounts.Sort();
            }
        }

        var candidates = new List<int>();
        foreach (int depth in forcedDepth is int d ? (int[])[d] : GcnKernels.PackedGemvDepths.ToArray())
        {
            foreach (int splits in splitCounts)
            {
                candidates.Add(EncodePacked(depth, splits));
            }
        }

        return [.. candidates];
    }

    // The fastest of `candidates` for `key`, measured once (or kept); `fallback` (the default) while it cannot be measured.
    private int Tune(TuneKey key, int[] candidates, int fallback, Action<int> run)
    {
        var table = Tuning;
        if (!Autotune || candidates.Length <= 1)
        {
            return candidates.Length == 1 ? candidates[0] : fallback;
        }

        if (table.TryKnown(key, candidates, out int known))
        {
            return known;
        }

        if (!CanMeasure)
        {
            return fallback;
        }

        lock (table.Measuring)
        {
            if (table.TryKnown(key, candidates, out known))
            {
                return known;                                              // another thread measured it meanwhile
            }

            int chosen = MeasureOnDevice(candidates, fallback, run);
            table.Record(key, chosen);
            return chosen;
        }
    }

    // Times the candidates between two HIP events on this device's stream (each first run outside the timing loads its
    // kernel), CUDA's comparison deciding.
    private int MeasureOnDevice(int[] candidates, int fallback, Action<int> run)
    {
        MakeCurrent();
        if (_tuneEvents.Start == IntPtr.Zero)
        {
            IntPtr start, end;
            Check(hipEventCreate(&start), nameof(hipEventCreate));
            Check(hipEventCreate(&end), nameof(hipEventCreate));
            _tuneEvents = (start, end);
        }

        t_timing = true;
        try
        {
            foreach (int candidate in candidates)
            {
                run(candidate);
            }

            return MeasureChoice(candidates, fallback, (candidate, count) =>
            {
                Check(hipEventRecord(_tuneEvents.Start, _stream), nameof(hipEventRecord));
                for (int i = 0; i < count; i++)
                {
                    run(candidate);
                }

                Check(hipEventRecord(_tuneEvents.End, _stream), nameof(hipEventRecord));
                Check(hipEventSynchronize(_tuneEvents.End), nameof(hipEventSynchronize));
                float ms;
                Check(hipEventElapsedTime(&ms, _tuneEvents.Start, _tuneEvents.End), nameof(hipEventElapsedTime));
                return ms;
            });
        }
        finally
        {
            t_timing = false;
        }
    }

    /// <summary>
    /// The choice among <paramref name="candidates"/>, <paramref name="time"/>(candidate, runs) giving the milliseconds of
    /// that many runs back to back: every candidate in turns until the timings settle (<see cref="TuneTiming.Warm"/>), the
    /// runs a timing takes from one run's time (about 0.2 ms, at most 16), then CUDA's paired comparison against
    /// <paramref name="fallback"/> (<see cref="TuneTiming.Choose"/>). Pure, so tests run it with a made-up timer.
    /// </summary>
    internal static int MeasureChoice(int[] candidates, int fallback, Func<int, int, float> time)
    {
        ArgumentNullException.ThrowIfNull(candidates);
        ArgumentNullException.ThrowIfNull(time);
        TuneTiming.Warm(() =>
        {
            float total = 0f;
            foreach (int candidate in candidates)
            {
                total += time(candidate, 1);
            }

            return total;
        });

        var repeats = new int[candidates.Length];
        for (int i = 0; i < candidates.Length; i++)
        {
            repeats[i] = CudaBackend.TimingRepeats(time(candidates[i], 1), cold: false);
        }

        return candidates[TuneTiming.Choose(candidates, fallback, i => time(candidates[i], repeats[i]) / repeats[i])];
    }

    // ---- Sharing between a cluster's ranks (HipMeasuredChoices) -----------------------------------------------------

    internal MeasuredChoices ExportChoices() => new(Kind, ChoicesIdentity, Name, Tuning.Export());

    internal MeasuredChoicesResult ImportChoices(MeasuredChoices choices)
    {
        if (Refusal(choices.Kind, choices.Identity, choices.Device) is { } refused)
        {
            return refused;
        }

        LeaveChoices();
        int taken = Tuning.Import(choices.Entries);
        return new MeasuredChoicesResult(true, taken, $"took rank 0's {taken} choices ({choices.Device})");
    }

    internal MeasuredChoicesResult FollowChoices(HipBackend leader)
    {
        if (ReferenceEquals(leader, this))
        {
            return new MeasuredChoicesResult(true, Tuning.Count, "rank 0's own backend");
        }

        if (Refusal(leader.Kind, leader.ChoicesIdentity, leader.Name) is { } refused)
        {
            return refused;
        }

        var shared = leader.Tuning;
        Volatile.Write(ref _tuning, shared);
        _leader = leader;
        return new MeasuredChoicesResult(true, shared.Count, $"shares rank 0's choices ({leader.Name})");
    }

    internal void LeaveChoices()
    {
        _leader = null;
        Volatile.Write(ref _tuning, null);                                     // its own again, read from its cache file when next needed
    }

    private MeasuredChoicesResult? Refusal(string kind, string identity, string device)
    {
        if (!string.Equals(kind, Kind, StringComparison.Ordinal))
        {
            return new MeasuredChoicesResult(false, 0, $"kept its own: rank 0 runs on another kind of device ({kind}: {device}) than {Name}");
        }

        return identity == ChoicesIdentity ? null
            : new MeasuredChoicesResult(false, 0, $"kept its own: rank 0's choices were measured on another GPU, driver or build ({device}: {identity}) than {Name}");
    }
}

/// <summary>
/// One HIP device's measured choices: those measured here, those its cache file keeps (read once, when first needed) and
/// those taken from another process's rank 0 (never written to the file). Thread-safe; a cluster's followers in one process
/// share their leader's table and its measuring lock.
/// </summary>
internal sealed class HipTuningTable
{
    private readonly Lock _gate = new();
    private readonly Dictionary<TuneKey, int> _tuned = [];
    private readonly Dictionary<TuneKey, int> _imported = [];
    private readonly string? _path;
    private Dictionary<TuneKey, int>? _persisted;

    /// <summary>A table for <paramref name="identity"/>'s device kept in <paramref name="folder"/> (null: kept in this process only).</summary>
    internal HipTuningTable(TuningIdentity identity, string? folder)
    {
        Identity = identity;
        _path = folder is null ? null : Path.Combine(folder, TuningCache.FileName(identity.Device));
    }

    /// <summary>What the choices apply to.</summary>
    internal TuningIdentity Identity { get; }

    /// <summary>The cache file, or null when none is kept.</summary>
    internal string? File => _path;

    /// <summary>Held while candidates are timed: one measurement a device at a time (followers share it).</summary>
    internal Lock Measuring { get; } = new();

    /// <summary>Choices known now: measured or taken here.</summary>
    internal int Count
    {
        get
        {
            lock (_gate)
            {
                return _tuned.Count;
            }
        }
    }

    /// <summary>The choice known for <paramref name="key"/> (measured, taken or kept in the file) when it is one of <paramref name="candidates"/>.</summary>
    internal bool TryKnown(TuneKey key, ReadOnlySpan<int> candidates, out int value)
    {
        lock (_gate)
        {
            if (_tuned.TryGetValue(key, out value))
            {
                return candidates.Contains(value);
            }

            _persisted ??= _path is null ? [] : TuningCache.Load(_path, Identity);
            if (_persisted.TryGetValue(key, out value) && candidates.Contains(value))
            {
                _tuned[key] = value;
                return true;
            }
        }

        value = 0;
        return false;
    }

    /// <summary>Keeps a choice measured here and writes the file (the choices measured here, merged with what others stored).</summary>
    internal void Record(TuneKey key, int value)
    {
        Dictionary<TuneKey, int> own;
        lock (_gate)
        {
            _tuned[key] = value;
            _imported.Remove(key);
            own = new Dictionary<TuneKey, int>(_tuned);
            foreach (var (taken, choice) in _imported)
            {
                if (own.TryGetValue(taken, out int current) && current == choice)
                {
                    own.Remove(taken);                                          // rank 0's, not measured here
                }
            }
        }

        if (_path is not null)
        {
            TuningCache.Save(_path, Identity, own);
        }
    }

    /// <summary>Every choice known: "t key" for those measured or taken here, "p key" for those the file keeps (read again).</summary>
    internal Dictionary<string, string> Export()
    {
        var entries = new Dictionary<string, string>(StringComparer.Ordinal);
        lock (_gate)
        {
            if (_path is not null)
            {
                _persisted = TuningCache.Load(_path, Identity);
            }

            foreach (var (key, value) in _persisted ?? [])
            {
                entries["p " + Format(key)] = value.ToString(CultureInfo.InvariantCulture);
            }

            foreach (var (key, value) in _tuned)
            {
                entries["t " + Format(key)] = value.ToString(CultureInfo.InvariantCulture);
            }
        }

        return entries;
    }

    /// <summary>Takes another process's choices (<see cref="Export"/>'s entries) in place of these; they are never written to the file. Returns how many.</summary>
    internal int Import(IReadOnlyDictionary<string, string> entries)
    {
        lock (_gate)
        {
            _tuned.Clear();
            _imported.Clear();
            _persisted = [];
            foreach (var (name, text) in entries)
            {
                if (name.Length > 2 && name[1] == ' ' && (name[0] is 't' or 'p') && TryParse(name[2..], out var key)
                    && int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value))
                {
                    _tuned[key] = value;
                    _imported[key] = value;
                }
            }

            return _tuned.Count;
        }
    }

    private static string Format(TuneKey key) =>
        string.Create(CultureInfo.InvariantCulture, $"{key.Op} {key.Variant} {key.A} {key.B} {key.C} {key.D} {key.E} {key.F}");

    private static bool TryParse(string text, out TuneKey key)
    {
        key = default;
        string[] parts = text.Split(' ');
        if (parts.Length != 8 || !Enum.TryParse(parts[0], out TuneOp op) || !Enum.IsDefined(op))
        {
            return false;
        }

        Span<int> n = stackalloc int[7];
        for (int i = 0; i < 7; i++)
        {
            if (!int.TryParse(parts[i + 1], NumberStyles.Integer, CultureInfo.InvariantCulture, out n[i]))
            {
                return false;
            }
        }

        key = new TuneKey(op, n[0], n[1], n[2], n[3], n[4], n[5], n[6]);
        return true;
    }
}

/// <summary>The HIP backend's measured choices between the ranks of a cluster (HipBackend.Tuning.cs).</summary>
internal sealed class HipMeasuredChoices : IMeasuredChoicesProvider
{
    public string Name => "hip";

    public string Summary => "the HIP kernels' measured choices (the packed few-row products' rows of k a step and k splits), per GPU, driver and library build";

    public bool Handles(Backend backend) => backend is HipBackend;

    public MeasuredChoices Export(Backend backend) => Of(backend).ExportChoices();

    public MeasuredChoicesResult Import(Backend backend, MeasuredChoices choices)
    {
        ArgumentNullException.ThrowIfNull(choices);
        return Of(backend).ImportChoices(choices);
    }

    public MeasuredChoicesResult Follow(Backend follower, Backend leader)
    {
        var mine = Of(follower);
        return leader is HipBackend hip ? mine.FollowChoices(hip)
            : new MeasuredChoicesResult(false, 0, $"kept its own: rank 0 runs on another kind of device ({leader.Kind}: {leader.Name}) than {mine.Name}");
    }

    public void Leave(Backend follower) => Of(follower).LeaveChoices();

    private static HipBackend Of(Backend backend) => backend as HipBackend
        ?? throw new ArgumentException($"{backend?.Name ?? "null"} is not a HIP backend.", nameof(backend));
}
