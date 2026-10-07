// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Operations;

namespace Idrak.Abstraction.Devices;

/// <summary>
/// Thrown when a device (a GPU driver or runtime call, a kernel) fails: the base of the CUDA, Vulkan and HIP errors. A
/// new one reports a <see cref="DeviceFailed"/> event to telemetry (<see cref="TelemetryLevel.Devices"/>), so every
/// device error is seen wherever it surfaces, including an asynchronous GPU error found at a later synchronize.
/// </summary>
/// <remarks>
/// A failing kernel is not retried on the CPU unless its device's <see cref="Backend.RetryOnHost"/> is on (plan 10,
/// decision 13): the error reaches the caller, and the event's <see cref="DeviceFailed.Hint"/> says how to turn the retry
/// on. Code that expects a device error and handles it (a probe, a capability check) opens <see cref="Handled"/> around
/// it, so nothing is reported.
/// </remarks>
public class DeviceException : Exception
{
    // Open Handled scopes and dispatcher retries on this thread: errors raised inside are not reported here (the caller
    // handles them, or the dispatcher reports them with the operation).
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
            Telemetry.DeviceFailed(new DeviceFailed(device, null, message, RetriedOnHost: false, Kernels.RetriesOnHost(device) ? null : Hint(device)));
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

    /// <summary>
    /// What a <see cref="DeviceFailed"/> event says when retry is off for <paramref name="device"/>: both ways to turn it
    /// on, and what it cannot cover.
    /// </summary>
    internal static string Hint(string device) =>
        $"To run an operation whose {device} kernel fails on the CPU instead, set IDRAK_RETRY_ON_HOST={device} (or 1 for every GPU device), "
        + "or set Backend.RetryOnHost = true on the device (device.Backend.RetryOnHost). Errors reported after an operation returned "
        + "(asynchronous GPU errors, found at a later synchronize or copy) can't be retried.";

    // The dispatcher's retry (Backend.NameRegistered): the device's kernel runs with reports off, so a failure is reported
    // once, by the dispatcher, with the operation.
    internal static void EnterRetry() => t_quiet++;

    internal static void LeaveRetry() => t_quiet--;

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
