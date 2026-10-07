// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Abstraction.Diagnostics;

namespace Idrak.Abstraction;

/// <summary>What happens to the calls of a slot an app has overridden (see <see cref="SlotTable{TKey, TValue}"/>).</summary>
public enum SlotPolicy
{
    /// <summary>
    /// The default: the app's implementation answers; when it throws, the call is retried on the library default and
    /// the failure goes to telemetry (<see cref="TelemetryLevel.Overrides"/>).
    /// </summary>
    FallBack,

    /// <summary>The app's implementation answers and its failures reach the caller (tests and development).</summary>
    Throw,

    /// <summary>
    /// The library default answers; on a sample of calls (<see cref="Slot.ShadowRate"/>) the app's implementation runs
    /// too, and the two outputs, times and allocations are compared and reported to telemetry.
    /// </summary>
    Shadow,
}

/// <summary>
/// Builds what a slot hands out while an app has overridden it: <paramref name="app"/> guarded by the slot's policy,
/// with <paramref name="library"/> (the library default) to fall back to or compare with. A registry gives one to its
/// <see cref="SlotTable{TKey, TValue}"/>; it runs once per registration (not per call) and reads the policy from
/// <paramref name="slot"/> at each call, so a policy changed later applies at once.
/// </summary>
public delegate TValue SlotGuard<TValue>(Slot slot, TValue app, TValue library);

/// <summary>
/// One registry entry that an app has overridden: its address, the app's implementation, the policy, and the helpers a
/// <see cref="SlotGuard{TValue}"/> calls to apply the policy and report to telemetry.
/// </summary>
public sealed class Slot
{
    /// <summary>The default share of calls a <see cref="SlotPolicy.Shadow"/> slot compares: 1%.</summary>
    public const double DefaultShadowRate = 0.01;

    private sealed record Settings(SlotPolicy Policy, double ShadowRate);

    private volatile Settings _settings = new(SlotPolicy.FallBack, DefaultShadowRate);
    private long _fallBacks, _compared, _differed;

    internal Slot(string registry, string name)
    {
        Registry = registry;
        Name = name;
    }

    /// <summary>The registry, for example "RopeScalings".</summary>
    public string Registry { get; }

    /// <summary>The entry's name in the registry, for example "yarn".</summary>
    public string Name { get; }

    /// <summary>The app's implementation: its type, or the method of a delegate.</summary>
    public string Implementation { get; internal set; } = "";

    /// <summary>The assembly the app's implementation comes from.</summary>
    public string Origin { get; internal set; } = Overrides.Library;

    /// <summary>The policy (<see cref="SlotPolicy.FallBack"/> unless set).</summary>
    public SlotPolicy Policy => _settings.Policy;

    /// <summary>The share of calls a <see cref="SlotPolicy.Shadow"/> slot compares, from 0 to 1.</summary>
    public double ShadowRate => _settings.ShadowRate;

    /// <summary>Calls that fell back to the library default since the app registered.</summary>
    public long FallBacks => Interlocked.Read(ref _fallBacks);

    /// <summary>Calls compared in <see cref="SlotPolicy.Shadow"/>.</summary>
    public long Compared => Interlocked.Read(ref _compared);

    /// <summary>Compared calls whose outputs differed (or where the app's implementation threw).</summary>
    public long Differed => Interlocked.Read(ref _differed);

    internal void Set(SlotPolicy policy, double shadowRate)
    {
        if (!Enum.IsDefined(policy))
        {
            throw new ArgumentOutOfRangeException(nameof(policy));
        }

        if (!(shadowRate is >= 0 and <= 1))
        {
            throw new ArgumentOutOfRangeException(nameof(shadowRate), shadowRate, "The shadow rate is a share of calls, from 0 to 1.");
        }

        _settings = new(policy, shadowRate);
    }

    internal void Reset() => (_fallBacks, _compared, _differed) = (0, 0, 0);

    /// <summary>
    /// One call under the policy: <paramref name="app"/> answers (<see cref="SlotPolicy.Throw"/>), or answers and falls back
    /// to <paramref name="library"/> when it throws (<see cref="SlotPolicy.FallBack"/>), or <paramref name="library"/>
    /// answers and, on sampled calls, <paramref name="app"/> runs too and <paramref name="compare"/> says how they differ
    /// (<see cref="SlotPolicy.Shadow"/>; with no comparison only failures, times and allocations are reported).
    /// </summary>
    /// <param name="app">The app's implementation, called.</param>
    /// <param name="library">The library default, called.</param>
    /// <param name="compare">The difference between the library's output and the app's, or null when they agree (see <see cref="Comparisons"/>).</param>
    /// <param name="effects">
    /// The call changes something outside (writes files, fetches them): under <see cref="SlotPolicy.Shadow"/> only the
    /// library's runs, so nothing is done twice.
    /// </param>
    public TResult Call<TResult>(Func<TResult> app, Func<TResult> library, Func<TResult, TResult, string?>? compare = null, bool effects = false)
    {
        ArgumentNullException.ThrowIfNull(app);
        ArgumentNullException.ThrowIfNull(library);
        if (Policy != SlotPolicy.Shadow)
        {
            try
            {
                return app();
            }
            catch (Exception e) when (FallsBack(e))
            {
                return library();
            }
        }

        var run = effects ? null : Shadow();
        var answer = library();
        if (run is not null)
        {
            run.Answered();
            try
            {
                var other = app();
                run.Done(compare?.Invoke(answer, other));
            }
            catch (Exception e) when (e is not OperationCanceledException)
            {
                run.Failed(e);
            }
        }

        return answer;
    }

    /// <summary>
    /// For a guard that catches failures itself (<c>catch (Exception e) when (slot.FallsBack(e))</c>): under
    /// <see cref="SlotPolicy.FallBack"/>, reports <paramref name="error"/> to telemetry and returns true (the guard then
    /// calls the library default); otherwise returns false and the error goes on to the caller. A cancellation never
    /// falls back.
    /// </summary>
    public bool FallsBack(Exception error)
    {
        ArgumentNullException.ThrowIfNull(error);
        if (Policy != SlotPolicy.FallBack || error is OperationCanceledException)
        {
            return false;
        }

        Interlocked.Increment(ref _fallBacks);
        if (Telemetry.IsEnabled(TelemetryLevel.Overrides))
        {
            Telemetry.OverrideFellBack(new OverrideFellBack(Registry, Name, Implementation, Origin, error));
        }

        return true;
    }

    /// <summary>
    /// For a guard under <see cref="SlotPolicy.Shadow"/>: a run that times the library default from now, when this call
    /// is sampled; null otherwise (the policy is not Shadow, or the call is not in the sample). See <see cref="ShadowRun"/>.
    /// </summary>
    public ShadowRun? Shadow() => Samples() ? new ShadowRun(this) : null;

    /// <summary>
    /// Whether this call (or this whole run, for an object made once and used many times) is in the
    /// <see cref="SlotPolicy.Shadow"/> sample: false when the policy is not Shadow.
    /// </summary>
    public bool Samples()
    {
        var settings = _settings;
        return settings.Policy == SlotPolicy.Shadow && settings.ShadowRate > 0
            && (settings.ShadowRate >= 1 || Random.Shared.NextDouble() < settings.ShadowRate);
    }

    /// <summary>
    /// Reports a comparison a guard measured itself (a whole run of an object made once and used many times, such as a
    /// sampler through a generation): how the app's outputs differed (null: they agreed) or what it threw, and the time and
    /// managed allocations of each.
    /// </summary>
    public void ReportComparison(string? difference, TimeSpan libraryTime, TimeSpan overrideTime, long libraryBytes = 0, long overrideBytes = 0, Exception? error = null) =>
        Report(new OverrideCompared(Registry, Name, Implementation, Origin, difference is null && error is null,
            difference ?? (error is null ? null : $"the override threw {error.GetType().Name}: {error.Message}"), error,
            libraryTime, overrideTime, libraryBytes, overrideBytes));

    internal void Report(in OverrideCompared e)
    {
        Interlocked.Increment(ref _compared);
        if (!e.Agreed)
        {
            Interlocked.Increment(ref _differed);
        }

        if (Telemetry.IsEnabled(TelemetryLevel.Overrides))
        {
            Telemetry.OverrideCompared(in e);
        }
    }

    /// <inheritdoc />
    public override string ToString() => $"{Registry}/{Name}";
}

/// <summary>
/// One sampled call of a <see cref="SlotPolicy.Shadow"/> slot, from <see cref="Slot.Shadow"/>: run the library default,
/// call <see cref="Answered"/>, run the app's implementation, then call <see cref="Done"/> with the difference (or
/// <see cref="Failed"/> when it threw). The comparison, both times and both threads' managed allocations go to telemetry.
/// </summary>
public sealed class ShadowRun
{
    private readonly Slot _slot;
    private long _start = Stopwatch.GetTimestamp();
    private long _bytes = GC.GetAllocatedBytesForCurrentThread();
    private TimeSpan _libraryTime;
    private long _libraryBytes;
    private bool _answered;

    internal ShadowRun(Slot slot) => _slot = slot;

    /// <summary>The library default has answered: its time ends and the app's begins.</summary>
    public void Answered()
    {
        long now = Stopwatch.GetTimestamp(), bytes = GC.GetAllocatedBytesForCurrentThread();
        (_libraryTime, _libraryBytes, _answered) = (Stopwatch.GetElapsedTime(_start, now), bytes - _bytes, true);
        (_start, _bytes) = (Stopwatch.GetTimestamp(), GC.GetAllocatedBytesForCurrentThread());
    }

    /// <summary>The app's implementation answered; <paramref name="difference"/> says how its output differs from the library's (null: they agree).</summary>
    public void Done(string? difference) => Report(difference, null);

    /// <summary>The app's implementation threw <paramref name="error"/> (the library's answer stands).</summary>
    public void Failed(Exception error)
    {
        ArgumentNullException.ThrowIfNull(error);
        Report(null, error);
    }

    private void Report(string? difference, Exception? error)
    {
        if (!_answered)
        {
            Answered();
        }

        var appTime = Stopwatch.GetElapsedTime(_start);
        long appBytes = GC.GetAllocatedBytesForCurrentThread() - _bytes;
        _slot.ReportComparison(difference, _libraryTime, appTime, _libraryBytes, appBytes, error);
    }
}
