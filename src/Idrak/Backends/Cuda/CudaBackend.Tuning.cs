// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using static Idrak.Backends.Cuda.CudaDriver;

namespace Idrak.Backends.Cuda;

// Choices that depend on the card (how many ways to split k, which of two kernels, which tile) are measured on the card
// in use, not set from one card's benchmarks: the first time a shape needs one, every candidate runs on the real inputs
// and is timed with events, and the fastest is kept. Candidates write either the caller's output (which the chosen
// kernel then writes again) or scratch memory (outputs that are added to). The formula from the device's own counts
// (SMs, tiles) is the default: it is used while measuring is impossible (a graph is being recorded, the profiler runs,
// no device memory for scratch) and with IDRAK_AUTOTUNE=0, and a candidate replaces it only when its median time is
// clearly faster (TuneTiming), so timing noise does not move the choice. Measured choices are kept per GPU, driver and
// library build in the user's cache folder (TuningCache; IDRAK_TUNING_CACHE=0 turns that off), so the next start reads
// them instead of measuring again.
/// <summary>Which choice a measurement is for.</summary>
internal enum TuneOp : byte
{
    GemvSplits,
    GemvMultiSplits,
    Int8FewRows,
    PackedSplits,
    PackedTile,
    PackedMultiSplits,
    PackedLowRankSplits,
    GatedActivation,
    TensorSplits,
    FloatTile,
    DecodeSplits,
}

/// <summary>
/// What a measured choice is for: the operation, its kernel variant and its shape. Numbers only (no strings), so a lookup
/// on the per-token path allocates nothing and hashes a few integers.
/// </summary>
internal readonly record struct TuneKey(TuneOp Op, int Variant, int A, int B, int C, int D = 0, int E = 0, int F = 0);

internal sealed unsafe partial class CudaBackend
{
    /// <summary>Measure card-dependent choices on the device (default); off (IDRAK_AUTOTUNE=0): the formulas only.</summary>
    internal static bool Autotune = Environment.GetEnvironmentVariable("IDRAK_AUTOTUNE") is not ("0" or "false");

    /// <summary>Keep measured choices in the user's cache folder and read them at the next start (IDRAK_TUNING_CACHE=0: off).</summary>
    internal static bool PersistTuning = TuningCache.Folder() is not null;

    private readonly Dictionary<TuneKey, int> _tuned = [];

    // Choices read from the cache file (null until first needed); each is used only when it is still one of the
    // candidates the shape offers, so a file from a build with other candidates cannot pick an invalid one.
    private Dictionary<TuneKey, int>? _persisted;
    private (string Path, TuningIdentity Identity)? _tuningFile;
    private (IntPtr Start, IntPtr End) _tuneEvents;

    // While candidates are timed, choices they need themselves use what is already known (or the formula), so one
    // measurement never starts another inside its timed region.
    [ThreadStatic]
    private static bool t_timing;

    /// <summary>Choices measured on this device so far (for tests and diagnostics).</summary>
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

    /// <summary>Forgets the measured choices and those read from the cache file; the file is read again when next needed (tests).</summary>
    internal void ForgetTuning()
    {
        lock (_tuned)
        {
            _tuned.Clear();
            _persisted = null;
            _tuningFile = null;
        }
    }

    /// <summary>Choices timed on the device in this process (not read from the cache file; tests and diagnostics).</summary>
    internal int MeasuredCount { get; private set; }

    /// <summary>What the cache file of this GPU belongs to (tests and diagnostics).</summary>
    internal TuningIdentity TuningIdentity => new(
        $"{_deviceName}; compute {_computeMajor}.{_computeMinor}; {_multiprocessors} SMs; {_totalMemory >> 20} MiB",
        $"CUDA {_driverVersion}; {TuningCache.DriverRelease()}",
        TuningCache.LibraryBuild(PtxKernels.Source));

    /// <summary>The cache file of this GPU, or null when the cache is off (tests and diagnostics).</summary>
    internal string? TuningFile
    {
        get
        {
            lock (_tuned)
            {
                return TuningFileLocked()?.Path;
            }
        }
    }

    // The cache file and its identity, or null when the cache is off. Called under the _tuned lock.
    private (string Path, TuningIdentity Identity)? TuningFileLocked()
    {
        if (!PersistTuning || TuningCache.Folder() is not { } folder)
        {
            return null;
        }

        if (_tuningFile is not { } file)
        {
            var identity = TuningIdentity;
            file = (Path.Combine(folder, TuningCache.FileName(identity.Device)), identity);
            _tuningFile = file;
        }

        return file;
    }

    // A choice read from the cache file for `key`, when it is one of `candidates`. Called under the _tuned lock.
    private bool TryPersistedLocked(TuneKey key, ReadOnlySpan<int> candidates, out int value)
    {
        _persisted ??= TuningFileLocked() is { } file ? TuningCache.Load(file.Path, file.Identity) : [];
        return _persisted.TryGetValue(key, out value) && candidates.Contains(value);
    }

    // Writes the measured choices to the cache file (keeping those other processes wrote there).
    private void SaveTuning()
    {
        (string Path, TuningIdentity Identity)? file;
        Dictionary<TuneKey, int> snapshot;
        lock (_tuned)
        {
            file = TuningFileLocked();
            snapshot = new Dictionary<TuneKey, int>(_tuned);
        }

        if (file is { } f)
        {
            TuningCache.Save(f.Path, f.Identity, snapshot);
        }
    }

    // Whether `key` was measured already (callers skip preparing candidates then).
    private bool TunedKnown(TuneKey key)
    {
        lock (_tuned)
        {
            return _tuned.ContainsKey(key);
        }
    }

    // The fastest of `candidates` for `key`, measured once (or read from the cache file); `fallback` (the formula's
    // choice) while it cannot be measured. `run(candidate)` runs the operation with that choice; it must leave nothing the
    // caller relies on changed (it writes the output the caller writes again, or scratch memory).
    private int Tune(TuneKey key, ReadOnlySpan<int> candidates, int fallback, Action<int> run)
    {
        if (!Autotune)
        {
            return fallback;                                               // IDRAK_AUTOTUNE=0: the formulas only
        }

        lock (_tuned)
        {
            if (_tuned.TryGetValue(key, out int known))
            {
                return known;
            }

            if (candidates.Length > 1 && TryPersistedLocked(key, candidates, out int kept))
            {
                _tuned[key] = kept;
                return kept;
            }
        }

        if (candidates.Length <= 1)
        {
            return candidates.Length == 1 ? candidates[0] : fallback;
        }

        if (!CanMeasure)
        {
            return fallback;
        }

        MakeCurrent();
        using var use = UseStream();
        if (_tuneEvents.Start == IntPtr.Zero)
        {
            Check(cuEventCreate(out var start, 0), nameof(cuEventCreate));
            Check(cuEventCreate(out var end, 0), nameof(cuEventCreate));
            _tuneEvents = (start, end);
        }

        // A first run of each candidate outside the timing: loads its kernel and settles the choices it needs itself.
        foreach (int candidate in candidates)
        {
            run(candidate);
        }

        // Each candidate is timed as it runs in practice, several launches back to back (as many as take about 0.2 ms,
        // at most 16): a launch timed alone misses how splits overlap with the work around them. Candidates take turns
        // over TuneTiming.Rounds rounds and the medians are compared (TuneTiming.Choose), so neither a GPU still raising
        // its clocks (laptops) nor one lucky or unlucky timing decides.
        var list = candidates.ToArray();
        var repeats = new int[list.Length];
        int chosen;
        t_timing = true;
        try
        {
            for (int c = 0; c < list.Length; c++)
            {
                float once = TimeRuns(list[c], 1, run);
                repeats[c] = Math.Clamp((int)MathF.Ceiling(0.2f / Math.Max(once, 1e-3f)), 1, 16);
            }

            chosen = list[TuneTiming.Choose(list, fallback, c => TimeRuns(list[c], repeats[c], run) / repeats[c])];
        }
        finally
        {
            t_timing = false;
        }

        lock (_tuned)
        {
            _tuned[key] = chosen;
            MeasuredCount++;
        }

        SaveTuning();
        return chosen;
    }

    // Whether candidates can be timed now: not inside another measurement, not while the profiler runs or a graph is
    // being recorded.
    private bool CanMeasure => Autotune && !t_timing && _profile is null && _captureFree is null && Volatile.Read(ref _captureThread) == 0;

    // Milliseconds of `count` back-to-back runs of one candidate, measured with events on the work stream.
    private float TimeRuns(int candidate, int count, Action<int> run)
    {
        Check(cuEventRecord(_tuneEvents.Start, _stream), nameof(cuEventRecord));
        for (int i = 0; i < count; i++)
        {
            run(candidate);
        }

        Check(cuEventRecord(_tuneEvents.End, _stream), nameof(cuEventRecord));
        Check(cuEventSynchronize(_tuneEvents.End), nameof(cuEventSynchronize));
        Check(cuEventElapsedTime(out float ms, _tuneEvents.Start, _tuneEvents.End), nameof(cuEventElapsedTime));
        return ms;
    }

    // Split counts to try where any count works (chunks of k need no particular alignment): 1, 2, 3, 4, 6, 8, 12, ...
    // up to `max` (the measured best on prompt-sized products was often 3 or 6).
    private static int[] SplitCounts(int max)
    {
        var values = new List<int>();
        for (int s = 1; s <= Math.Max(1, max); s *= 2)
        {
            values.Add(s);
            if (s >= 2 && s + s / 2 <= max)
            {
                values.Add(s + s / 2);
            }
        }

        return [.. values];
    }

    // Split counts to try: 1, 2, 4, ... up to `max`.
    private static int[] PowersOfTwo(int max)
    {
        var values = new List<int>();
        for (int s = 1; s <= Math.Max(1, max); s *= 2)
        {
            values.Add(s);
        }

        return [.. values];
    }

    // Runs `body` with a device scratch block of `floats` (its address), then frees it; false when there is no room (the
    // caller then keeps the formula). Never falls back to system memory, where timings would mislead.
    private bool WithScratch(long floats, Action<ulong> body)
    {
        if (floats <= 0 || floats > int.MaxValue)
        {
            return false;
        }

        ulong block = TryDeviceBlock((int)floats, 0, useCache: true, out int capacity);
        if (block == 0)
        {
            return false;
        }

        try
        {
            body(block);
        }
        finally
        {
            lock (_pool)
            {
                ReturnDeviceBlockLocked(block, capacity);
            }
        }

        return true;
    }
}
