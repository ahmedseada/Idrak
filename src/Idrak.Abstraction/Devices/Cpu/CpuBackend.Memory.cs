// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;

namespace Idrak.Abstraction.Devices.Cpu;

/// <summary>
/// The CPU's memory: what the process may use in all and what it may still take, from the system where it reports
/// them, held to the runtime's heap limit where the runtime has one.
/// </summary>
/// <param name="Total">What the process may use in all (the CPU device's memory).</param>
/// <param name="Available">What it may still take now.</param>
/// <param name="System">What the system reports (empty where it reports nothing: the runtime's figures are used).</param>
/// <param name="RuntimeTotal">The runtime's figure (<see cref="GCMemoryInfo.TotalAvailableMemoryBytes"/>).</param>
/// <param name="RuntimeIsBudget">The runtime's figure is its current heap budget, which grows as the heap does (Mono), not a limit.</param>
internal readonly record struct CpuMemory(long Total, long Available, SystemMemoryReport System, long RuntimeTotal, bool RuntimeIsBudget)
{
    /// <summary>The runtime caps the heap (the CPU's tensors) below what the system reports: a GC heap limit, set or implied by a container.</summary>
    public bool RuntimeCaps => !RuntimeIsBudget && System.ProcessTotal is { } system && RuntimeTotal < system;
}

internal sealed partial class CpuBackend
{
    /// <summary>
    /// Whether the runtime is Mono (.NET for Android's and iOS's default), whose <see cref="GCMemoryInfo.TotalAvailableMemoryBytes"/>
    /// is its heap's current budget (a few hundred MiB at start, growing as the heap does), not the machine's memory or a
    /// limit; CoreCLR reports the machine's memory, or its heap limit where one is set or implied by a container's. Asked
    /// once, when the memory is first read (a fit check, a listing), not when the CPU's backend starts.
    /// </summary>
    internal static bool RuntimeIsMono => MonoRuntime.Is;

    private static class MonoRuntime
    {
        internal static readonly bool Is = Type.GetType("Mono.RuntimeStructs") is not null;
    }

    /// <summary>
    /// How long a reading of the system's memory is kept before it is read again (IDRAK_MEMORY_REFRESH_MS, default 250
    /// ms; 0 reads it on every call): fit checks and attention paths ask for it on every call, and the files cost tens of
    /// microseconds to read.
    /// </summary>
    internal static TimeSpan MemoryRefresh { get; set; } = TimeSpan.FromMilliseconds(
        Environment.GetEnvironmentVariable("IDRAK_MEMORY_REFRESH_MS") is { Length: > 0 } ms
        && int.TryParse(ms, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) && value >= 0 ? value : 250);

    private sealed record MemoryReading(CpuMemory Memory, long At);

    private static volatile MemoryReading? s_memory;

    /// <summary>The CPU's memory now (read again once <see cref="MemoryRefresh"/> has passed).</summary>
    internal static CpuMemory Memory
    {
        get
        {
            long now = Stopwatch.GetTimestamp();
            if (s_memory is { } last && Stopwatch.GetElapsedTime(last.At, now) < MemoryRefresh)
            {
                return last.Memory;
            }

            var info = GC.GetGCMemoryInfo();
            var memory = ChooseMemory(SystemMemory.Read(), info.TotalAvailableMemoryBytes, info.MemoryLoadBytes, RuntimeIsMono);
            s_memory = new MemoryReading(memory, now);
            return memory;
        }
    }

    /// <summary>
    /// The CPU's memory from what the system reports and the runtime's figures. Where the system reports nothing (Windows,
    /// macOS), the runtime's: its total, and its total less its load (as before). Where the runtime's figure is its heap
    /// budget (Mono), the system's alone. Otherwise (CoreCLR) the lower of each: a heap limit the runtime holds the
    /// process to (set, or implied by a container's limit) stays the limit, and what is available is also what the
    /// system has free now.
    /// </summary>
    internal static CpuMemory ChooseMemory(SystemMemoryReport system, long runtimeTotal, long runtimeLoad, bool runtimeIsBudget)
    {
        long runtimeAvailable = Math.Max(0, runtimeTotal - runtimeLoad);
        if (system.ProcessTotal is not { } total)
        {
            return new CpuMemory(runtimeTotal, runtimeAvailable, system, runtimeTotal, runtimeIsBudget);
        }

        long available = system.ProcessAvailable ?? total;
        return runtimeIsBudget
            ? new CpuMemory(total, Math.Min(available, total), system, runtimeTotal, true)
            : new CpuMemory(Math.Min(total, runtimeTotal), Math.Min(available, runtimeAvailable), system, runtimeTotal, false);
    }

    /// <summary>The memory facts for device listings: the CPU's, the system's (and its group limit) and the runtime's, labelled.</summary>
    internal static KeyValuePair<string, string>[] MemoryFacts(CpuMemory m)
    {
        var facts = new List<KeyValuePair<string, string>>(4)
        {
            new("memory", $"{SystemMemory.Format(m.Total)} the process may use, {SystemMemory.Format(m.Available)} of it free now"
                + (m.System.ProcessTotal is null ? " (as the runtime reports them: the system reports none here)"
                    : m.RuntimeCaps ? " (held to the runtime's heap limit)" : " (as the system reports them)")),
        };

        if (m.System.Total is { } total)
        {
            facts.Add(new("system memory", $"{SystemMemory.Format(total)}"
                + (m.System.Available is { } available ? $", {SystemMemory.Format(available)} available" : "") + " (/proc/meminfo)"));
        }

        if (m.System.Limit is { } limit)
        {
            facts.Add(new("memory limit", $"{SystemMemory.Format(limit)}"
                + (m.System.LimitUsage is { } used ? $", {SystemMemory.Format(used)} in use" : "") + $" ({m.System.LimitSource})"));
        }

        string runtime = RuntimeIsMono ? "Mono" : "CoreCLR";
        facts.Add(new("runtime memory", m.RuntimeIsBudget
            ? $"{SystemMemory.Format(m.RuntimeTotal)}: {runtime}'s heap budget now, not a limit (it grows as the heap does)"
            : m.RuntimeCaps
                ? $"{SystemMemory.Format(m.RuntimeTotal)}: {runtime}'s heap limit (set, or implied by the memory limit), which the CPU's tensors live within"
                : $"{SystemMemory.Format(m.RuntimeTotal)}: what {runtime} reports the heap may grow to"));
        return [.. facts];
    }
}
