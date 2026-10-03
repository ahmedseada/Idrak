// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Cli.Shared;

/// <summary>
/// The memory a device offers to the design rules. The CPU offers what the .NET runtime may use (or the CPU limit set
/// on <see cref="ComputeResources"/>); a GPU offers the limit set on <see cref="ComputeResources.GpuMemoryLimit"/>. The
/// library does not report a GPU's own memory size (a gap noted in the plan), so without a limit a GPU's memory is
/// unknown and the rules say so instead of guessing.
/// </summary>
internal static class DeviceMemory
{
    /// <summary>Bytes the device offers, or null when unknown.</summary>
    public static long? Total(Device device)
    {
        if (!device.IsGpu)
        {
            long available = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
            return ComputeResources.CpuMemoryLimit is { } limit ? Math.Min(limit, available) : available;
        }

        return ComputeResources.GpuMemoryLimit;
    }

    /// <summary>Bytes as text: "1.5 GB", "320 MB", "12 KB".</summary>
    public static string Format(double bytes) => bytes switch
    {
        >= 1e9 => $"{bytes / 1e9:0.#} GB",
        >= 1e6 => $"{bytes / 1e6:0.#} MB",
        >= 1e3 => $"{bytes / 1e3:0.#} KB",
        _ => $"{bytes:0} B",
    };

    /// <summary>A count as text: "45k", "1.2M", "3.4G", "812".</summary>
    public static string Count(double value) => value switch
    {
        >= 1e9 => $"{value / 1e9:0.#}G",
        >= 1e6 => $"{value / 1e6:0.#}M",
        >= 1e3 => $"{value / 1e3:0.#}k",
        _ => $"{value:0}",
    };
}
