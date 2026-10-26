// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices;

/// <summary>
/// Thrown by a device whose driver refused to build one of its kernels (a Vulkan driver's vkCreateComputePipelines
/// failing, after the device's own retries: without a subgroup-size request, the kernel's other variants). The device
/// stays usable: on a device that builds its kernels when first used (<see cref="Backend.BuildsKernelsWhenUsed"/>) the
/// dispatcher runs the operation through its fallback instead (the host, its composed path, or the caller's own), and the
/// refusal is kept, so the driver is asked once per process. Only an operation with no fallback reaches the caller with it.
/// </summary>
public sealed class KernelUnavailableException : DeviceException
{
    /// <summary>Creates the error (reported to telemetry unless inside <see cref="DeviceException.Handled"/>).</summary>
    /// <param name="device">The kind of device ("vulkan").</param>
    /// <param name="kernel">The kernel refused, by name.</param>
    /// <param name="message">What was tried and what the driver answered.</param>
    public KernelUnavailableException(string device, string kernel, string message)
        : base(device, message)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(kernel);
        Kernel = kernel;
    }

    /// <summary>The kernel refused, by name.</summary>
    public string Kernel { get; }
}
