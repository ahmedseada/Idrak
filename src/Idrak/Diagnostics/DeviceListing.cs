// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

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

    /// <summary>Whether <see cref="Idrak.Abstraction.Device.Available"/> (and so a plain test run) lists it; false: reached by name only.</summary>
    public bool Listed { get; init; }

    /// <summary>Whether it is <see cref="Idrak.Abstraction.Device.Default"/>.</summary>
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

    /// <summary>More of what the device reports, by name, in the backend's order (<see cref="Idrak.Abstraction.Devices.BackendHardware.Facts"/>), or null.</summary>
    public IReadOnlyList<KeyValuePair<string, string>>? Facts { get; init; }

    /// <summary>Why the device's backend could not start, or null.</summary>
    public string? Error { get; init; }

    /// <summary>
    /// Why the device's own kernels are unavailable as a whole (<see cref="Idrak.Abstraction.Devices.IKernelAvailability"/>:
    /// a missing run-time compiler, a setting), so every operation runs through the host; null while they run.
    /// </summary>
    public string? KernelsUnavailable { get; init; }
}

/// <summary>
/// Every device and backend as the library sees them, for tools that list devices (the test runner's
/// <c>--list-devices</c>, <c>idrak devices</c> and <c>idrak doctor</c>).
/// </summary>
public static class DeviceListing
{
    /// <summary>The registered device kinds beyond the CPU, in registration order, with their device counts.</summary>
    public static IReadOnlyList<BackendInfo> Backends =>
        [.. DeviceProviders.All.Select(p => new BackendInfo(p.Name, p.Display, p.Count, p.Count == 0 ? p.UnavailableReason : null))];

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
                devices.Add(Describe(Device.Get(provider.Name, i)));
            }
        }

        return devices;
    }

    // The devices listed so far in this process (the first listing of each is its cold start).
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, bool> s_listed = new(StringComparer.Ordinal);

    /// <summary>
    /// What <paramref name="device"/> reports (starting its backend). Reported to telemetry (<see cref="DeviceListed"/>,
    /// <see cref="TelemetryLevel.Devices"/>): the time to start the backend and to read what it reports, the first listing
    /// of each device in the process (cold) apart from the later ones (warm).
    /// </summary>
    public static DeviceInfo Describe(Device device)
    {
        ArgumentNullException.ThrowIfNull(device);
        if (!Telemetry.IsEnabled(TelemetryLevel.Devices))
        {
            s_listed.TryAdd(device.ToString(), true);
            return DescribeTimed(device, out _, out _);
        }

        bool cold = s_listed.TryAdd(device.ToString(), true);
        var info = DescribeTimed(device, out var start, out var describe);
        Telemetry.DeviceListed(new DeviceListed(info.Device, start, describe, cold, info.Error));
        return info;
    }

    private static DeviceInfo DescribeTimed(Device device, out TimeSpan start, out TimeSpan describe)
    {
        long began = System.Diagnostics.Stopwatch.GetTimestamp();
        describe = TimeSpan.Zero;
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
            start = System.Diagnostics.Stopwatch.GetElapsedTime(began);
            return info with { Error = e.Message };
        }

        start = System.Diagnostics.Stopwatch.GetElapsedTime(began);
        long described = System.Diagnostics.Stopwatch.GetTimestamp();
        var hardware = backend.Hardware;
        var result = info with
        {
            Name = backend.Name,
            MatrixUnits = backend.MatrixUnits,
            MemoryBytes = hardware.MemoryBytes,
            ComputeUnits = hardware.ComputeUnits,
            SubgroupSize = hardware.SubgroupSize,
            KernelWidth = hardware.KernelWidth,
            HardwareKind = hardware.HardwareKind,
            Driver = hardware.Driver,
            Facts = hardware.Facts,
            KernelsUnavailable = (backend as Idrak.Abstraction.Devices.IKernelAvailability)?.KernelsUnavailable(),
        };
        describe = System.Diagnostics.Stopwatch.GetElapsedTime(described);
        return result;
    }
}
