// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction;

/// <summary>A snapshot of a device's memory accounting, in bytes.</summary>
/// <param name="InUse">Bytes held by live tensors.</param>
/// <param name="Cached">Bytes of freed blocks kept for reuse (returned to the system by <see cref="ComputeResources.ReleaseCachedMemory"/>).</param>
/// <param name="Limit">The configured cap, or null when unlimited.</param>
/// <param name="Offloaded">Bytes of this GPU's tensors kept in system memory (see <see cref="ComputeResources.OffloadToHostMemory"/>).</param>
/// <param name="Peak">The most bytes held by live tensors at once since the device started or <see cref="ComputeResources.ResetPeakMemoryUsage"/>.</param>
public readonly record struct MemoryUsage(long InUse, long Cached, long? Limit, long Offloaded = 0, long Peak = 0)
{
    /// <summary>InUse + Cached: what the library currently holds from the system.</summary>
    public long Reserved => InUse + Cached;

    /// <inheritdoc />
    public override string ToString() =>
        $"in use {Format(InUse)}, cached {Format(Cached)}" + (Limit is { } l ? $", limit {Format(l)}" : "")
        + (Offloaded > 0 ? $", offloaded to system memory {Format(Offloaded)}" : "");

    private static string Format(long bytes) => bytes switch
    {
        >= 1L << 30 => string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{bytes / (double)(1L << 30):F2} GiB"),
        >= 1L << 20 => string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{bytes / (double)(1L << 20):F1} MiB"),
        >= 1L << 10 => string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{bytes / (double)(1L << 10):F1} KiB"),
        _ => string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{bytes} B"),
    };
}

/// <summary>Thrown when an allocation would exceed a limit set on <see cref="ComputeResources"/>.</summary>
public sealed class ResourceLimitExceededException(string message) : Exception(message);

/// <summary>
/// Controls how much of the machine Idrak may use. By default there are no limits:
/// every CPU core is available and memory is allocated as the network needs it.
/// </summary>
/// <example>
/// <code>
/// ComputeResources.MaxCpuThreads = 4;                       // leave the other cores for the app
/// ComputeResources.GpuMemoryLimit = 2L * 1024 * 1024 * 1024; // at most 2 GiB per GPU
/// </code>
/// </example>
public static class ComputeResources
{
    private static ParallelOptions s_parallelOptions = new() { MaxDegreeOfParallelism = Environment.ProcessorCount };

    /// <summary>
    /// The maximum number of CPU threads used by CPU kernels, data loading and parsing.
    /// Defaults to <see cref="Environment.ProcessorCount"/>; 1 makes all CPU work single-threaded.
    /// </summary>
    public static int MaxCpuThreads
    {
        get => s_parallelOptions.MaxDegreeOfParallelism;
        set
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(value);
            s_parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = Math.Min(value, Environment.ProcessorCount) };
        }
    }

    /// <summary>Maximum bytes of tensor memory on the CPU (null = unlimited).</summary>
    public static long? CpuMemoryLimit { get; set; }

    /// <summary>Maximum bytes of tensor memory on each GPU (null = unlimited, up to what the card has).</summary>
    public static long? GpuMemoryLimit { get; set; }

    /// <summary>
    /// GPU memory left free for the display and other programs: a new GPU allocation that would leave less free is treated
    /// as not fitting, so the driver never has to page GPU memory out to system memory on its own (on Windows that makes
    /// long kernels, which the watchdog resets as a hung GPU). Null (the default): a sixteenth of each GPU's memory, at
    /// least 256 MiB (512 MiB on an 8 GB card, 1 GiB on a 16 GB one).
    /// </summary>
    public static long? GpuMemoryReserve { get; set; }

    /// <summary>
    /// When a GPU is full (or at <see cref="GpuMemoryLimit"/>), place new tensors in pinned system memory that the GPU
    /// reads and writes directly over PCIe, instead of failing: larger models and training runs work, slower for the data
    /// kept there. Off by default (then a full GPU raises <see cref="ResourceLimitExceededException"/>); the environment
    /// variable IDRAK_OFFLOAD=1 turns it on.
    /// </summary>
    public static bool OffloadToHostMemory { get; set; } = Environment.GetEnvironmentVariable("IDRAK_OFFLOAD") is "1" or "true";

    /// <summary>
    /// With <see cref="OffloadToHostMemory"/>: when the GPU fills up, first move the storages that are used least to
    /// system memory (optimizer state, then frozen weights such as a LoRA base model, largest first) to make room, and
    /// only then place new tensors there. On by default; off restores the older behaviour (whatever is allocated after
    /// the GPU is full goes to system memory).
    /// </summary>
    public static bool OffloadColdFirst { get; set; } = true;

    /// <summary>
    /// Before a layer computes (forward or backward), copy its offloaded weights to the GPU, the next layer's in the
    /// background while this one runs, instead of reading them over PCIe as the kernels go. On by default.
    /// </summary>
    public static bool PrefetchOffloadedWeights { get; set; } = true;

    /// <summary>
    /// Bring offloaded tensors back to the GPU when memory frees up (checked at each optimizer step and by
    /// <see cref="ReturnOffloaded"/>), hottest first, while the GPU keeps <see cref="OffloadReturnHeadroom"/> free.
    /// On by default.
    /// </summary>
    public static bool ReturnOffloadedTensors { get; set; } = true;

    /// <summary>
    /// Device memory kept free when offloaded tensors come back, so a step that needs a little more does not push them
    /// straight out again. Null (the default): an eighth of each GPU's memory, at least 256 MiB.
    /// </summary>
    public static long? OffloadReturnHeadroom { get; set; }

    /// <summary>
    /// Brings offloaded tensors back to the GPU (or every GPU when <paramref name="device"/> is null) where they fit now,
    /// e.g. after training, when the activations are gone: the GPU's cached blocks are released first, then tensors come
    /// back hottest first while the headroom (<see cref="OffloadReturnHeadroom"/>) stays free. Returns how many moved.
    /// </summary>
    public static int ReturnOffloaded(Device? device = null)
    {
        int moved = 0;
        foreach (var d in device is null ? Device.InUse() : [device])
        {
            if (d.Backend.Offload is { OffloadedCount: > 0 } offload)
            {
                d.Backend.ReleaseCachedMemory();
                moved += offload.Rebalance(makeRoom: false);
            }
        }

        return moved;
    }

    /// <summary>The options CPU loops run <see cref="Parallel"/> with: at most <see cref="MaxCpuThreads"/> threads.</summary>
    public static ParallelOptions ParallelOptions => s_parallelOptions;

    /// <summary>Whether CPU work should be split across threads at all.</summary>
    public static bool AllowParallel => s_parallelOptions.MaxDegreeOfParallelism > 1;

    /// <summary>Returns the memory accounting for <paramref name="device"/>.</summary>
    public static MemoryUsage GetMemoryUsage(Device device) => device.Backend.GetMemoryUsage();

    /// <summary>Starts the device's peak of bytes in use (<see cref="MemoryUsage.Peak"/>) again from what is in use now.</summary>
    public static void ResetPeakMemoryUsage(Device device) => device.Backend.ResetPeakMemoryUsage();

    /// <summary>Returns cached (unused) blocks to the system for one device, or every initialized device when null.</summary>
    public static void ReleaseCachedMemory(Device? device = null)
    {
        if (device is not null)
        {
            device.Backend.ReleaseCachedMemory();
            return;
        }

        foreach (var started in Device.InUse())
        {
            started.Backend.ReleaseCachedMemory();
        }
    }
}
