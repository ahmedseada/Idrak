// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;
using Idrak.Backends;
using Idrak.Backends.Cpu;
using Idrak.Backends.Cuda;
using Idrak.Backends.Hip;
using Idrak.Backends.Vulkan;

namespace Idrak.Diagnostics;

/// <summary>One kind of device beyond the CPU (CUDA, Vulkan, HIP, or a registered one) and how many were found.</summary>
/// <param name="Kind">The name in device names ("vulkan" in "vulkan:0").</param>
/// <param name="Display">The name in messages ("Vulkan").</param>
/// <param name="Count">Devices found (0 when the driver or runtime is missing).</param>
/// <param name="UnavailableReason">Why none were found, when none were; otherwise null.</param>
public sealed record BackendInfo(string Kind, string Display, int Count, string? UnavailableReason);

/// <summary>
/// What a device reports about itself, for device listings and diagnostics (read-only; nothing here changes what the
/// device does). Values a backend does not report are null.
/// </summary>
public sealed record DeviceInfo
{
    /// <summary>The device's name in options ("cpu", "vulkan:0").</summary>
    public required string Device { get; init; }

    /// <summary>The backend kind ("cpu", "cuda", "vulkan", "hip", ...).</summary>
    public required string Kind { get; init; }

    /// <summary>The backend's name in messages ("CPU", "Vulkan").</summary>
    public required string Backend { get; init; }

    /// <summary>The device's own name, with its driver where the backend adds it; "" when it could not start.</summary>
    public required string Name { get; init; }

    /// <summary>Whether <see cref="Idrak.Device.Available"/> (and so a plain test run) lists it; false: reached by name only.</summary>
    public bool Listed { get; init; }

    /// <summary>Whether it is <see cref="Idrak.Device.Default"/>.</summary>
    public bool IsDefault { get; init; }

    /// <summary>Why it is not listed or not chosen by default, when the backend says.</summary>
    public string? Note { get; init; }

    /// <summary>Memory the device's tensors live in, in bytes (the CPU: what the process may use).</summary>
    public long? MemoryBytes { get; init; }

    /// <summary>Compute units (multiprocessors, compute units; the CPU: logical processors); null when not reported.</summary>
    public int? ComputeUnits { get; init; }

    /// <summary>Lanes that run in lockstep (subgroup, warp or wavefront; the CPU: float lanes of a SIMD vector).</summary>
    public int? SubgroupSize { get; init; }

    /// <summary>Whether the device's products run on matrix units.</summary>
    public bool MatrixUnits { get; init; }

    /// <summary>Threads per workgroup (block) of the device's kernels, as chosen from its limits; null when none.</summary>
    public int? KernelWidth { get; init; }

    /// <summary>What the device reports itself to be: "discrete", "integrated", "virtual", "cpu" (a CPU or a software driver), or null.</summary>
    public string? HardwareKind { get; init; }

    /// <summary>The driver or runtime version, when the backend reports it separately from <see cref="Name"/>.</summary>
    public string? Driver { get; init; }

    /// <summary>Why the device's backend could not start, or null.</summary>
    public string? Error { get; init; }
}

/// <summary>
/// Every device and backend as the library sees them, for tools that list devices (the test runner's
/// <c>--list-devices</c>, <c>idrak devices</c> and <c>idrak doctor</c>).
/// </summary>
public static class DeviceListing
{
    /// <summary>The registered device kinds beyond the CPU, in registration order, with their device counts.</summary>
    public static IReadOnlyList<BackendInfo> Backends =>
        [.. DeviceProviders.All.Select(p => new BackendInfo(p.Kind, p.Display, p.Count, p.Count == 0 ? p.UnavailableReason : null))];

    /// <summary>
    /// The CPU and every device of every backend, listed or reached by name only (starting each one's backend; one that
    /// fails to start is returned with <see cref="DeviceInfo.Error"/> set).
    /// </summary>
    public static IReadOnlyList<DeviceInfo> All()
    {
        var devices = new List<DeviceInfo> { Describe(Device.Cpu) };
        foreach (var provider in DeviceProviders.All)
        {
            for (int i = 0; i < provider.Count; i++)
            {
                devices.Add(Describe(Device.Get(provider.Kind, i)));
            }
        }

        return devices;
    }

    /// <summary>What <paramref name="device"/> reports (starting its backend).</summary>
    public static DeviceInfo Describe(Device device)
    {
        ArgumentNullException.ThrowIfNull(device);
        var provider = device.Type == DeviceType.Cpu ? null : DeviceProviders.Find(device.Kind);
        bool isDefault = device == Device.Default;
        var info = new DeviceInfo
        {
            Device = device.ToString(),
            Kind = device.Kind,
            Backend = provider?.Display ?? "CPU",
            Name = "",
            Listed = provider?.Listed(device.Ordinal) ?? true,
            IsDefault = isDefault,
            Note = isDefault ? null : provider?.Note(device.Ordinal),
        };

        Backend backend;
        try
        {
            backend = device.Backend;
        }
        catch (Exception e)
        {
            return info with { Error = e.Message };
        }

        info = info with { Name = backend.Name, MatrixUnits = backend.Capabilities.MatrixUnits };
        return backend switch
        {
            CpuBackend => info with
            {
                MemoryBytes = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes,
                HardwareKind = "cpu",
                ComputeUnits = Environment.ProcessorCount,
                SubgroupSize = Vector<float>.Count,
            },
            CudaBackend cuda => info with
            {
                MemoryBytes = cuda.Limits.MemoryBytes,
                ComputeUnits = cuda.Limits.Multiprocessors,
                SubgroupSize = cuda.Limits.WarpSize,
                KernelWidth = cuda.Shapes.BlockSize,
                Driver = cuda.TuningIdentity.Driver,
            },
            VulkanBackend vulkan => info with
            {
                MemoryBytes = vulkan.StorageHeapBytes,
                SubgroupSize = (int)vulkan.Facts.SubgroupSize,
                KernelWidth = vulkan.Width,
                HardwareKind = vulkan.Facts.DeviceType switch { 1 => "integrated", 2 => "discrete", 3 => "virtual", 4 => "cpu", _ => null },
                Driver = vulkan.Driver,
            },
            HipBackend hip => info with
            {
                MemoryBytes = hip.DeviceLimits.TotalMemory,
                ComputeUnits = hip.DeviceLimits.ComputeUnits,
                SubgroupSize = hip.DeviceLimits.WarpSize,
                KernelWidth = HipKernels.BlockSizeFor(hip.DeviceLimits),
                HardwareKind = hip.DeviceLimits.Integrated ? "integrated" : "discrete",
            },
            _ => info,
        };
    }
}
