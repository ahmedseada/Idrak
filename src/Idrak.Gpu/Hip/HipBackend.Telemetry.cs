// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using static Idrak.Gpu.Hip.HipRuntime;

namespace Idrak.Gpu.Hip;

// HIP's failures and refusals in the library's telemetry (TelemetryLevel.Devices), as Vulkan's: what --verbose, the
// profile's probes and every hook see.
//   - A runtime call that fails throws a HipException, a DeviceException, which reports itself (DeviceFailed) unless the
//     calling code handles it (DeviceException.Handled: the device probe). A GCN code object the runtime refuses in every
//     format (LoadGcn) is one: its message names the target, each format with the runtime's error, and the runtime.
//   - A GCN launch the runtime refuses names the kernel and its grid (LaunchFailed).
//   - A profile on a runtime without the event functions times nothing and says why, once per device.
//   - The kernels' unavailability as a whole (KernelsUnavailableReason) goes with each operation's first host fallback
//     (HostFallback, through IKernelAvailability), as Vulkan's does.
//   - The kernel set's load (HipBackend.Kernels.cs) is reported once (KernelsLoaded): from the cache or assembled, its
//     time and size; kernels that do not assemble are a DeviceFailed of operation "kernels".
internal sealed partial class HipBackend
{
    // Whether this device has reported that it cannot time its kernels (once per device).
    private int _profileUnavailableReported;

    // The error a GCN launch the runtime refused throws: the kernel, its grid and the runtime's error. Created outside
    // any Handled scope it reports itself (DeviceFailed).
    private HipException LaunchFailed(GcnFunction function, uint blocks, uint blocksY, uint blockSize, int result) =>
        new(string.Create(CultureInfo.InvariantCulture,
            $"{Name} could not launch GCN kernel '{function.Name}' ({blocks}{(blocksY > 1 ? $" × {blocksY}" : "")} blocks of {blockSize}, code object format {function.Format} for {function.Target}): "
            + $"{nameof(hipModuleLaunchKernel)} failed with {Describe(result)}"));

    // Reports the kernel set's load (once per device: LoadKernels), while someone listens: from the cache or assembled,
    // its time and size (rules 76, 78).
    private void ReportKernelsLoaded(GcnKernelSet set)
    {
        if (Telemetry.IsEnabled(TelemetryLevel.Devices))
        {
            string assembled = set.FromCache ? "" : string.Create(CultureInfo.InvariantCulture, $", assembled in {set.Assembly.TotalMilliseconds:F1} ms");
            string detail = $"{set.Target}, code object format {set.Format}{assembled}{(set.Refused.Length > 0 ? $"; refused first: {set.Refused}" : "")}";
            Telemetry.KernelsLoaded(new KernelsLoaded($"hip:{Ordinal}", Kind, set.Functions.Count, set.FromCache, set.Duration, set.Bytes, detail));
        }
    }

    // Reports kernels that could not be made for this device (they do not assemble, or their code object does not
    // declare them as written: a fault of the library, never of the device), while someone listens.
    private void ReportKernelsFailed(string why)
    {
        if (Telemetry.IsEnabled(TelemetryLevel.Devices))
        {
            Telemetry.DeviceFailed(new DeviceFailed("hip", "kernels", why, RetriedOnHost: false, Hint: null));
        }
    }

    // Reports, once per device while someone listens, that a profile cannot time this device's kernels, and why.
    private void ReportProfileUnavailable(string why)
    {
        if (Telemetry.IsEnabled(TelemetryLevel.Devices) && Interlocked.Exchange(ref _profileUnavailableReported, 1) == 0)
        {
            Telemetry.DeviceFailed(new DeviceFailed("hip", "profile", why, RetriedOnHost: false, Hint: null));
        }
    }
}
