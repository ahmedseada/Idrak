// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using static Idrak.Gpu.Hip.HipRuntime;

namespace Idrak.Gpu.Hip;

// Kernel timing for the library's GPU profiler (Idrak.Diagnostics.GpuProfiler, the profile's "kernels" probe, idrak
// profile), as CUDA's: while a profile runs, each GCN launch is timed on the device between two HIP events, created once
// and reused, and added up by the kernel's name, or by the label a launch was given with its floating-point operations
// (products and attention, so their rate shows). Outside a profile a launch is one call, as before.
internal sealed unsafe partial class HipBackend : IKernelProfiler
{
    // Kernel times per name while a profile runs (null otherwise).
    private Dictionary<string, (long Calls, long Ticks, double Flops)>? _profile;

    // The two events a timed launch is recorded between (zero until the first timed launch creates them), and the lock
    // that keeps one timed launch between them at a time.
    private (IntPtr Start, IntPtr End) _profileEvents;
    private readonly Lock _profileLock = new();

    // The label and floating-point operations of the next launch on this thread while profiling (later phases: products
    // as "kernel m x n x k" with 2·m·n·k, attention with its shape; rows and element-wise kernels as "kernel rows x cols" or
    // "kernel n" with their floating-point operations from GcnKernels.Rows.cs's and GcnKernels.Pointwise.cs's tables),
    // consumed by that launch.
    [ThreadStatic]
    private static string? t_profileLabel;

    [ThreadStatic]
    private static double t_profileFlops;

    /// <summary>Whether a profile is timing this device's launches.</summary>
    internal bool Profiling => _profile is not null;

    /// <summary>
    /// Why this device cannot time its launches (null when it can): the event functions its HIP runtime lacks. A profile
    /// then lists no kernels, and says why through the telemetry when it starts.
    /// </summary>
    internal string? ProfileUnavailableReason => ProfileUnavailable(Name, LoadedFrom, MissingEventFunctions());

    /// <summary>The reason <see cref="ProfileUnavailableReason"/> gives for <paramref name="missing"/>, or null when nothing is.</summary>
    internal static string? ProfileUnavailable(string device, string runtime, IReadOnlyList<string> missing) => missing.Count == 0 ? null
        : $"{device} cannot time its kernels: the HIP runtime{(runtime.Length > 0 ? $" from {runtime}" : "")} lacks {string.Join(", ", missing)}, "
          + "so a profile lists none of its launches (a newer graphics driver or ROCm runtime has them)";

    // The event functions a timed launch needs that the loaded runtime does not export.
    private static List<string> MissingEventFunctions()
    {
        var missing = new List<string>(4);
        if (hipEventCreate == null)
        {
            missing.Add(nameof(hipEventCreate));
        }

        if (hipEventRecord == null)
        {
            missing.Add(nameof(hipEventRecord));
        }

        if (hipEventSynchronize == null)
        {
            missing.Add(nameof(hipEventSynchronize));
        }

        if (hipEventElapsedTime == null)
        {
            missing.Add(nameof(hipEventElapsedTime));
        }

        return missing;
    }

    void IKernelProfiler.StartProfile()
    {
        (t_profileLabel, t_profileFlops) = (null, 0);
        if (ProfileUnavailableReason is { } why)
        {
            ReportProfileUnavailable(why);
        }

        _profile = [];
    }

    Dictionary<string, (long Calls, long Ticks, double Flops)> IKernelProfiler.StopProfile()
    {
        var profile = _profile ?? [];
        _profile = null;
        return profile;
    }

    /// <summary>
    /// Names the next launch on this thread <paramref name="label"/> (a kernel and its shape) with
    /// <paramref name="flops"/> floating-point operations, while a profile runs: its time is added under the label, and
    /// the profile reports the rate. Call it only when <see cref="Profiling"/>, right before the launch.
    /// </summary>
    internal static void LabelNextLaunch(string label, double flops) => (t_profileLabel, t_profileFlops) = (label, flops);

    /// <summary>
    /// Queues <paramref name="function"/> with its kernel arguments in <paramref name="extra"/> (hipModuleLaunchKernel's
    /// buffer form); while a profile runs and the runtime has events, waits for it and adds its device time (between two
    /// events, the host's launch cost excluded) to the profile. A launch the runtime refuses throws a
    /// <see cref="HipException"/> naming the kernel and its grid, which the telemetry reports.
    /// </summary>
    private void Launch(GcnFunction function, uint blocks, uint blocksY, uint blocksZ, uint blockSize, void** extra)
    {
        if (_profile is { } profile && hipEventCreate != null && hipEventRecord != null && hipEventSynchronize != null && hipEventElapsedTime != null)
        {
            LaunchTimed(profile, function, blocks, blocksY, blocksZ, blockSize, extra);
            return;
        }

        int result = hipModuleLaunchKernel(function.Handle, blocks, blocksY, blocksZ, blockSize, 1, 1, 0, _stream, null, extra);
        if (result != Success)
        {
            throw LaunchFailed(function, blocks, blocksY, blockSize, result);
        }
    }

    private void LaunchTimed(Dictionary<string, (long Calls, long Ticks, double Flops)> profile, GcnFunction function, uint blocks, uint blocksY, uint blocksZ, uint blockSize, void** extra)
    {
        string key = t_profileLabel ?? function.Name;
        double flops = t_profileFlops;
        (t_profileLabel, t_profileFlops) = (null, 0);
        float elapsed;
        lock (_profileLock)
        {
            if (_profileEvents.Start == IntPtr.Zero)
            {
                IntPtr start, end;
                Check(hipEventCreate(&start), nameof(hipEventCreate));
                Check(hipEventCreate(&end), nameof(hipEventCreate));
                _profileEvents = (start, end);
            }

            Check(hipEventRecord(_profileEvents.Start, _stream), nameof(hipEventRecord));
            int result = hipModuleLaunchKernel(function.Handle, blocks, blocksY, blocksZ, blockSize, 1, 1, 0, _stream, null, extra);
            if (result != Success)
            {
                throw LaunchFailed(function, blocks, blocksY, blockSize, result);
            }

            Check(hipEventRecord(_profileEvents.End, _stream), nameof(hipEventRecord));
            Check(hipEventSynchronize(_profileEvents.End), nameof(hipEventSynchronize));
            Check(hipEventElapsedTime(&elapsed, _profileEvents.Start, _profileEvents.End), nameof(hipEventElapsedTime));
        }

        long ticks = (long)(elapsed * (Stopwatch.Frequency / 1000.0));
        lock (profile)
        {
            var entry = profile.GetValueOrDefault(key);
            profile[key] = (entry.Calls + 1, entry.Ticks + ticks, entry.Flops + flops);
        }
    }
}
