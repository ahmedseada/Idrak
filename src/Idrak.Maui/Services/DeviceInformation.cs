// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Runtime.InteropServices;
using Idrak.Abstraction.Devices;
using Idrak.Diagnostics;
using Idrak.Maui.Abstractions;

namespace Idrak.Maui;

/// <summary>The devices as the library sees them in the app's process, over <see cref="DeviceListing"/> (<see cref="IDeviceInformation"/>).</summary>
public sealed class DeviceInformation : IDeviceInformation
{
    /// <inheritdoc />
    public DeviceReport Read(IReadOnlyList<string>? devices = null)
    {
        long start = Stopwatch.GetTimestamp();
        var infos = devices is null ? DeviceListing.All() : [.. devices.Select(d => DeviceListing.Describe(Device.Parse(d)))];
        var details = new List<DeviceDetails>(infos.Count);
        foreach (var info in infos)
        {
            details.Add(info.Error is null ? Details(info) : new DeviceDetails(info, null, 0));
        }

        using var process = Process.GetCurrentProcess();
        var facts = new ProcessFacts(RuntimeInformation.OSDescription, RuntimeInformation.ProcessArchitecture.ToString(), RuntimeInformation.FrameworkDescription,
            Environment.ProcessorCount, Device.Cpu.Backend.Hardware.MemoryBytes ?? GC.GetGCMemoryInfo().TotalAvailableMemoryBytes, process.WorkingSet64, GC.GetTotalMemory(forceFullCollection: false),
            ComputeResources.CacheFolder, ComputeResources.MaxCpuThreads);
        return new DeviceReport(details, DeviceListing.Backends, facts, Device.Default.ToString(), Stopwatch.GetElapsedTime(start));
    }

    private static DeviceDetails Details(DeviceInfo info)
    {
        var device = Device.Parse(info.Device);
        MemoryUsage? memory = null;
        int choices = 0;
        try
        {
            memory = ComputeResources.GetMemoryUsage(device);
            if (MeasuredChoiceProviders.All.FirstOrDefault(p => p.Handles(device.Backend)) is { } provider)
            {
                choices = provider.Export(device.Backend).Entries.Count;
            }
        }
        catch (Exception e) when (e is InvalidOperationException or IOException or UnauthorizedAccessException or NotSupportedException)
        {
            // A device that reports no memory or no cache: the listing keeps what it has.
        }

        return new DeviceDetails(info, memory, choices);
    }
}
