// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction;

/// <summary>A snapshot of a device's memory accounting, in bytes.</summary>
/// <param name="InUse">Bytes held by live tensors.</param>
/// <param name="Cached">Bytes of freed blocks kept for reuse (returned to the system by <see cref="ComputeResources.ReleaseCachedMemory"/>).</param>
/// <param name="Limit">The configured cap, or null when unlimited.</param>
/// <param name="Offloaded">Bytes of this GPU's tensors kept in system memory (see <see cref="ComputeResources.OffloadToHostMemory"/>).</param>
/// <param name="Peak">The most bytes held by live tensors at once since the device started or <see cref="ComputeResources.ResetPeakMemoryUsage"/>.</param>
/// <param name="Largest">The largest single block taken since then (what made a peak, when one block did).</param>
public readonly record struct MemoryUsage(long InUse, long Cached, long? Limit, long Offloaded = 0, long Peak = 0, long Largest = 0)
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
/// Controls how much of the machine Idrak may use, and where it keeps its caches (<see cref="CacheFolder"/>). By default
/// there are no limits: every CPU core is available and memory is allocated as the network needs it.
/// </summary>
/// <example>
/// <code>
/// ComputeResources.MaxCpuThreads = 4;                       // leave the other cores for the app
/// ComputeResources.GpuMemoryLimit = 2L * 1024 * 1024 * 1024; // at most 2 GiB per GPU
/// </code>
/// </example>
public static class ComputeResources
{
    private static ParallelOptions s_parallelOptions = new() { MaxDegreeOfParallelism = Math.Clamp(EnvironmentInt("IDRAK_CPU_THREADS") ?? Environment.ProcessorCount, 1, Environment.ProcessorCount) };

    /// <summary>
    /// The maximum number of CPU threads used by CPU kernels, data loading and parsing.
    /// Defaults to <see cref="Environment.ProcessorCount"/> (every core); 1 makes all CPU work single-threaded. The
    /// environment variable IDRAK_CPU_THREADS sets it at start.
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

    /// <summary>Maximum bytes of tensor memory on the CPU (null = unlimited). IDRAK_CPU_MEMORY_LIMIT_MB sets it in MiB.</summary>
    public static long? CpuMemoryLimit { get; set; } = EnvironmentMebibytes("IDRAK_CPU_MEMORY_LIMIT_MB");

    /// <summary>Maximum bytes of tensor memory on each GPU (null = unlimited, up to what the card has). IDRAK_GPU_MEMORY_LIMIT_MB sets it in MiB.</summary>
    public static long? GpuMemoryLimit { get; set; } = EnvironmentMebibytes("IDRAK_GPU_MEMORY_LIMIT_MB");

    /// <summary>
    /// The share of the memory a device reports available that one operation's working buffers may take at once (the
    /// patches an unfolded convolution makes of a batch, the composed attention's scores, a measured batch's items): the
    /// rest is the room for everything else the pass allocates. In (0, 1]; half by default. IDRAK_WORKSPACE_SHARE sets it.
    /// </summary>
    public static double WorkspaceShare
    {
        get => s_workspaceShare;
        set
        {
            if (!(value > 0) || value > 1)
            {
                throw new ArgumentOutOfRangeException(nameof(value), value, "The workspace share is in (0, 1].");
            }

            s_workspaceShare = value;
        }
    }

    /// <summary>
    /// How far a measured choice reaches, in octaves of the work that follows the data (a convolution's batch × height ×
    /// width, a product's rows × batch, attention's rows × key rows): a shape at a size class none measured takes the choice
    /// of the nearest measured class of the same shape within this distance instead of measuring its own. Training on
    /// utterances or images of varied lengths lands in a new class with nearly every batch, and each class measured costs
    /// every such shape of the model once (seconds a class on a large batch), while within an octave the fastest choice is
    /// nearly always the same. 1 by default; 0 measures every class. IDRAK_TUNE_NEIGHBOUR_OCTAVES sets it.
    /// </summary>
    public static double TuningNeighbourOctaves
    {
        get => s_tuningNeighbourOctaves;
        set
        {
            if (!(value >= 0))
            {
                throw new ArgumentOutOfRangeException(nameof(value), value, "The tuning neighbour distance is 0 or more octaves.");
            }

            s_tuningNeighbourOctaves = value;
        }
    }

    /// <summary>
    /// How long the first use of a shape may stall while its kernel choices are measured (a backend that bounds its
    /// measurement: Vulkan): each candidate runs once while this lasts, and the rounds that settle close calls run only
    /// when they fit in it too; past it the single runs decide (a candidate clearly faster than the formula's choice), and
    /// a formula's choice slower than half of it is kept without trying the others. 500 ms by default;
    /// IDRAK_TUNE_BUDGET_MS sets it in milliseconds.
    /// </summary>
    public static TimeSpan TuningBudget
    {
        get => s_tuningBudget;
        set
        {
            if (!(value > TimeSpan.Zero))
            {
                throw new ArgumentOutOfRangeException(nameof(value), value, "The tuning budget is a positive time.");
            }

            s_tuningBudget = value;
        }
    }

    /// <summary>
    /// The folder the library keeps what it downloads, prepares and measures in: downloads (<c>downloads/</c>), prepared
    /// <c>.gguf</c> models (<c>gguf/</c>), the CPU's, Vulkan's, CUDA's, HIP's and the cluster's measured choices and
    /// compiled kernels, the feature cache and the noise model. Set it at start-up, before the first device or download
    /// (an app sets its own storage here): every cache reads it when it first needs a path, so a value set then takes
    /// effect everywhere. Null (the default, and what setting null brings back): the folder <c>IDRAK_CACHE</c> names,
    /// else <c>.cache/idrak</c> in the user's profile folder; where the profile folder is empty or not absolute (as it can
    /// be inside a phone's app), <c>idrak</c> in the local application data folder, and where that is not absolute either,
    /// <c>idrak</c> in the temporary folder. A relative value is made absolute against the current folder when it is read,
    /// so the result is never relative. A cache's own variable still wins for that cache (<c>IDRAK_VULKAN_TUNING_CACHE</c>,
    /// <c>IDRAK_TUNING_CACHE</c>, <c>IDRAK_CPU_TUNING_FILE</c>, <c>IDRAK_HIP_KERNEL_CACHE</c>, <c>IDRAK_RNNOISE_MODEL</c>, …).
    /// </summary>
    /// <exception cref="ArgumentException">Set to an empty or blank folder.</exception>
    [System.Diagnostics.CodeAnalysis.AllowNull]
    public static string CacheFolder
    {
        get => Volatile.Read(ref s_cacheFolder) is { } set ? Path.GetFullPath(set) : CacheFolderFrom(Environment.GetEnvironmentVariable("IDRAK_CACHE"));
        set
        {
            if (value is not null)
            {
                ArgumentException.ThrowIfNullOrWhiteSpace(value);
            }

            Volatile.Write(ref s_cacheFolder, value);
        }
    }

    // CacheFolder as set (null: from IDRAK_CACHE or the default, read each time it is asked).
    private static string? s_cacheFolder;

    /// <summary>
    /// Whether a pretrained model's weights packed while it loads (int8, int4: quantized from the checkpoint's values) are
    /// kept on disk (in <see cref="PackedWeightCacheFolder"/>), so the next load of the same checkpoint with the same
    /// format reads them instead of reading and quantizing again: the same bits. On by default; IDRAK_PACKED_CACHE=0 (or off,
    /// false) turns it off at start; a load's own option wins (<c>PretrainedOptions.PackedCache</c>).
    /// </summary>
    public static bool PackedWeightCache { get; set; } = !SwitchedOff(Environment.GetEnvironmentVariable("IDRAK_PACKED_CACHE"));

    /// <summary>
    /// Where <see cref="PackedWeightCache"/> keeps packed weights: IDRAK_PACKED_CACHE_DIR when set, else "packed" in
    /// <see cref="CacheFolder"/>; setting it overrides both (null goes back to them). A load's own option wins
    /// (<c>PretrainedOptions.PackedCacheFolder</c>).
    /// </summary>
    /// <exception cref="ArgumentException">Set to an empty or blank folder.</exception>
    [System.Diagnostics.CodeAnalysis.AllowNull]
    public static string PackedWeightCacheFolder
    {
        get => Volatile.Read(ref s_packedCacheFolder) is { } set ? Path.GetFullPath(set)
            : Environment.GetEnvironmentVariable("IDRAK_PACKED_CACHE_DIR") is { Length: > 0 } variable ? Path.GetFullPath(variable)
            : Path.Combine(CacheFolder, "packed");
        set
        {
            if (value is not null)
            {
                ArgumentException.ThrowIfNullOrWhiteSpace(value);
            }

            Volatile.Write(ref s_packedCacheFolder, value);
        }
    }

    private static string? s_packedCacheFolder;

    // Whether a switch variable's value turns its setting off: 0, off, false or no (any case, spaces around).
    internal static bool SwitchedOff(string? value) => value.AsSpan().Trim() is var v
        && (v is "0" || v.Equals("off", StringComparison.OrdinalIgnoreCase) || v.Equals("false", StringComparison.OrdinalIgnoreCase) || v.Equals("no", StringComparison.OrdinalIgnoreCase));

    /// <summary>
    /// The cache folder when <see cref="CacheFolder"/> is not set: <paramref name="variable"/> (IDRAK_CACHE's value) made
    /// absolute when given, else under the first absolute folder of <paramref name="profile"/> (the user's profile),
    /// <paramref name="localData"/> (the local application data) and <paramref name="temporary"/> (the temporary folder),
    /// each the system's when null.
    /// </summary>
    internal static string CacheFolderFrom(string? variable, string? profile = null, string? localData = null, string? temporary = null)
    {
        if (!string.IsNullOrWhiteSpace(variable))
        {
            return Path.GetFullPath(variable);
        }

        profile ??= Environment.GetFolderPath(Environment.SpecialFolder.UserProfile, Environment.SpecialFolderOption.DoNotVerify);
        if (Path.IsPathFullyQualified(profile))
        {
            return Path.Combine(profile, ".cache", "idrak");
        }

        localData ??= Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData, Environment.SpecialFolderOption.DoNotVerify);
        return Path.IsPathFullyQualified(localData) ? Path.Combine(localData, "idrak") : Path.GetFullPath(Path.Combine(temporary ?? Path.GetTempPath(), "idrak"));
    }

    private static double s_tuningNeighbourOctaves = EnvironmentDouble("IDRAK_TUNE_NEIGHBOUR_OCTAVES") is { } octaves && octaves >= 0 ? octaves : 1;

    private static TimeSpan s_tuningBudget = TimeSpan.FromMilliseconds(EnvironmentDouble("IDRAK_TUNE_BUDGET_MS") is { } budget && budget > 0 ? budget : 500);

    private static double s_workspaceShare = EnvironmentDouble("IDRAK_WORKSPACE_SHARE") is { } share && share > 0 && share <= 1 ? share : 0.5;

    /// <summary>
    /// GPU memory left free for the display and other programs: a new GPU allocation that would leave less free is treated
    /// as not fitting, so the driver never has to page GPU memory out to system memory on its own (on Windows that makes
    /// long kernels, which the watchdog resets as a hung GPU). Null (the default): a sixteenth of each GPU's memory, at
    /// least 256 MiB (512 MiB on an 8 GB card, 1 GiB on a 16 GB one). IDRAK_GPU_MEMORY_RESERVE_MB sets it in MiB (0: none).
    /// </summary>
    public static long? GpuMemoryReserve { get; set; } = EnvironmentMebibytes("IDRAK_GPU_MEMORY_RESERVE_MB");

    /// <summary>
    /// When a GPU is full (or at <see cref="GpuMemoryLimit"/>), place new tensors in pinned system memory that the GPU
    /// reads and writes directly over PCIe, instead of failing: larger models and training runs work, slower for the data
    /// kept there. Off by default (then a full GPU raises <see cref="ResourceLimitExceededException"/>); the environment
    /// variable IDRAK_OFFLOAD=1 turns it on.
    /// </summary>
    public static bool OffloadToHostMemory { get; set; } = Environment.GetEnvironmentVariable("IDRAK_OFFLOAD") is "1" or "true";

    /// <summary>
    /// The size of the chunks a GPU places long-lived blocks in (<see cref="Backend.LongLivedAllocations"/>: weights and
    /// attention caches), so the driver's page rounding of each block (a block of a page and a little more takes two
    /// pages) is paid once a chunk. Null (the default): 32 of the device's allocation pages, as its driver reports them; 0:
    /// every block its own allocation. IDRAK_GPU_CHUNK_MB sets it in MiB.
    /// </summary>
    public static long? GpuChunkBytes { get; set; } = Environment.GetEnvironmentVariable("IDRAK_GPU_CHUNK_MB") is { Length: > 0 } chunk
        ? long.Parse(chunk, System.Globalization.CultureInfo.InvariantCulture) << 20 : null;

    /// <summary>
    /// The stack each GPU thread is given when a device opens: the driver reserves it for every thread the device can run
    /// at once. Null (the default): none beyond what a kernel needs (the driver raises it for any kernel that needs more,
    /// and the library's kernels need none); a value sets it. IDRAK_GPU_STACK_BYTES sets it.
    /// </summary>
    public static long? GpuStackBytes { get; set; } = Environment.GetEnvironmentVariable("IDRAK_GPU_STACK_BYTES") is { Length: > 0 } stack
        ? long.Parse(stack, System.Globalization.CultureInfo.InvariantCulture) : null;

    /// <summary>
    /// Where frozen lookup tables live on a GPU that can keep storages in system memory (see <see cref="LookupTablePlacement"/>):
    /// <see cref="LookupTablePlacement.Automatic"/> by default (on the GPU until it is full, then in system memory for
    /// good). IDRAK_LOOKUP_TABLES sets it: auto, host or device. A build's own setting (DecoderBuildOptions.LookupTables,
    /// PretrainedOptions.LookupTables) wins.
    /// </summary>
    public static LookupTablePlacement LookupTables { get; set; } = ParseLookupTables(Environment.GetEnvironmentVariable("IDRAK_LOOKUP_TABLES"));

    /// <summary>
    /// A placement by name: "auto" (or empty), "host" or "device", any case.
    /// </summary>
    /// <exception cref="ArgumentException">Another name.</exception>
    public static LookupTablePlacement ParseLookupTables(string? name) => name?.Trim().ToLowerInvariant() switch
    {
        null or "" or "auto" or "automatic" => LookupTablePlacement.Automatic,
        "host" or "system" => LookupTablePlacement.HostMemory,
        "device" or "gpu" => LookupTablePlacement.DeviceMemory,
        _ => throw new ArgumentException($"Lookup table placement '{name}': use auto, host or device.", nameof(name)),
    };

    /// <summary>
    /// Which vector paths the CPU's kernels take (<see cref="CpuVectorPaths"/>): <see cref="CpuVectorPaths.Automatic"/> by
    /// default (the platform's intrinsics where the runtime reports them, else portable 128-bit vectors where it
    /// accelerates them). IDRAK_CPU_VECTORS sets it at start: auto, native, portable or scalar. Every path gives the same
    /// results; the products read it on every call, the int4 kernels' lane shifts when the CPU's int4 kernels first run.
    /// </summary>
    public static CpuVectorPaths CpuVectors { get; set; } = ParseCpuVectors(Environment.GetEnvironmentVariable("IDRAK_CPU_VECTORS"));

    /// <summary>A vector path setting by name: "auto" (or empty), "native", "portable" or "scalar", any case.</summary>
    /// <exception cref="ArgumentException">Another name.</exception>
    public static CpuVectorPaths ParseCpuVectors(string? name) => name?.Trim().ToLowerInvariant() switch
    {
        null or "" or "auto" or "automatic" => CpuVectorPaths.Automatic,
        "native" or "intrinsics" => CpuVectorPaths.Native,
        "portable" => CpuVectorPaths.Portable,
        "scalar" or "none" => CpuVectorPaths.Scalar,
        _ => throw new ArgumentException($"CPU vector paths '{name}': use auto, native, portable or scalar.", nameof(name)),
    };

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
    /// straight out again. Null (the default): an eighth of each GPU's memory, at least 256 MiB. IDRAK_OFFLOAD_RETURN_HEADROOM_MB
    /// sets it in MiB.
    /// </summary>
    public static long? OffloadReturnHeadroom { get; set; } = EnvironmentMebibytes("IDRAK_OFFLOAD_RETURN_HEADROOM_MB");

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

    private static int? EnvironmentInt(string name) =>
        Environment.GetEnvironmentVariable(name) is { Length: > 0 } s && int.TryParse(s, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out int value) ? value : null;

    private static double? EnvironmentDouble(string name) =>
        Environment.GetEnvironmentVariable(name) is { Length: > 0 } s && double.TryParse(s, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out double value) ? value : null;

    private static long? EnvironmentMebibytes(string name) =>
        Environment.GetEnvironmentVariable(name) is { Length: > 0 } s && long.TryParse(s, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out long value) && value >= 0 ? value << 20 : null;

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
