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
}
