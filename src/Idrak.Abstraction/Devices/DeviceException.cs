// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices;

/// <summary>
/// Thrown when a device (a GPU driver or runtime call, a kernel) fails: the base of the CUDA, Vulkan and HIP errors. A
/// new one reports a <see cref="DeviceFailed"/> event to telemetry (<see cref="TelemetryLevel.Devices"/>), so every
/// device error is seen wherever it surfaces, including an asynchronous GPU error found at a later synchronize.
/// </summary>
/// <remarks>
/// A failing kernel's error reaches the caller. Code that expects a device error and handles it (a probe, a capability
/// check) opens <see cref="Handled"/> around it, so nothing is reported.
/// </remarks>
public class DeviceException : Exception
{
    // Open Handled scopes on this thread: errors raised inside are not reported here (the caller handles them).
    [ThreadStatic]
    private static int t_quiet;

    /// <summary>Creates the error and reports it to telemetry (unless inside <see cref="Handled"/>).</summary>
    /// <param name="device">The kind of device that failed ("cuda", "vulkan", "hip", …: <see cref="Backend.Kind"/>).</param>
    /// <param name="message">What failed.</param>
    /// <param name="inner">The error that caused it, or null.</param>
    public DeviceException(string device, string message, Exception? inner = null)
        : base(message, inner)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(device);
        Device = device;
        if (t_quiet == 0 && Telemetry.IsEnabled(TelemetryLevel.Devices))
        {
            Telemetry.DeviceFailed(new DeviceFailed(device, null, message, RetriedOnHost: false, Hint: null));
        }
    }

    /// <summary>The kind of device that failed ("cuda", "vulkan", "hip", …).</summary>
    public string Device { get; }

    /// <summary>
    /// Starts a scope on this thread in which device errors are expected and handled by the calling code (a probe, a
    /// capability check that falls back): they are not reported to telemetry. Dispose it on the same thread to end the
    /// scope (a <c>using</c> block around synchronous code).
    /// </summary>
    public static IDisposable Handled()
    {
        t_quiet++;
        return new Scope();
    }

    private sealed class Scope : IDisposable
    {
        private int _disposed;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0)
            {
                t_quiet--;
            }
        }
    }
}
