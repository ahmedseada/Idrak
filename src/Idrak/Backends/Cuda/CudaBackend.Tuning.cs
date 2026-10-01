using static Idrak.Backends.Cuda.CudaDriver;

namespace Idrak.Backends.Cuda;

// Choices that depend on the card (how many ways to split k, which of two kernels, which tile) are measured on the card
// in use, not set from one card's benchmarks: the first time a shape needs one, every candidate runs on the real inputs
// and is timed with events, and the fastest is kept for the rest of the process. Candidates write either the caller's
// output (which the chosen kernel then writes again) or scratch memory (outputs that are added to). The formula from the
// device's own counts (SMs, tiles) is the default: it is used while measuring is impossible (a graph is being recorded,
// the profiler runs, no device memory for scratch, IDRAK_AUTOTUNE=0), and a candidate replaces it only when it is
// clearly faster, so timing noise does not move the choice.
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

    /// <summary>A candidate replaces the formula's choice only when it is at least this much faster (3%).</summary>
    private const float TuneMargin = 0.97f;

    private readonly Dictionary<TuneKey, int> _tuned = [];
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

    /// <summary>Forgets the measured choices (tests).</summary>
    internal void ForgetTuning()
    {
        lock (_tuned)
        {
            _tuned.Clear();
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

    // The fastest of `candidates` for `key`, measured once; `fallback` (the formula's choice) while it cannot be measured.
    // `run(candidate)` runs the operation with that choice; it must leave nothing the caller relies on changed (it writes
    // the output the caller writes again, or scratch memory).
    private int Tune(TuneKey key, ReadOnlySpan<int> candidates, int fallback, Action<int> run)
    {
        lock (_tuned)
        {
            if (_tuned.TryGetValue(key, out int known))
            {
                return known;
            }
        }

        if (candidates.Length <= 1)
        {
            return candidates.Length == 1 ? candidates[0] : fallback;
        }

        if (!Autotune || t_timing || _profile is not null || _captureFree is not null || Volatile.Read(ref _captureThread) != 0)
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

        var times = new float[candidates.Length];
        t_timing = true;
        try
        {
            for (int c = 0; c < candidates.Length; c++)
            {
                float best = float.MaxValue;
                for (int repeat = 0; repeat < 3; repeat++)
                {
                    Check(cuEventRecord(_tuneEvents.Start, _stream), nameof(cuEventRecord));
                    run(candidates[c]);
                    Check(cuEventRecord(_tuneEvents.End, _stream), nameof(cuEventRecord));
                    Check(cuEventSynchronize(_tuneEvents.End), nameof(cuEventSynchronize));
                    Check(cuEventElapsedTime(out float ms, _tuneEvents.Start, _tuneEvents.End), nameof(cuEventElapsedTime));
                    best = Math.Min(best, ms);
                }

                times[c] = best;
            }
        }
        finally
        {
            t_timing = false;
        }

        int fastest = 0;
        for (int c = 1; c < times.Length; c++)
        {
            if (times[c] < times[fastest])
            {
                fastest = c;
            }
        }

        int chosen = candidates[fastest];
        int formula = candidates.IndexOf(fallback);
        if (formula >= 0 && times[fastest] >= times[formula] * TuneMargin)
        {
            chosen = fallback;                                             // within the noise of the formula's choice
        }

        lock (_tuned)
        {
            _tuned[key] = chosen;
        }

        return chosen;
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
