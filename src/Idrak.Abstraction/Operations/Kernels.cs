// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction.Operations;

/// <summary>
/// The kernel table (plan 9): kernels registered for an operation on a kind of device ("cpu", "cuda", "vulkan", …), each
/// with an optional requirement on the device. A call to an operation runs the kernel registered last whose requirement
/// holds, else the device's own (<c>Backend.NameKernel</c>, whose default body is the host fallback). Each device resolves
/// its slots on its first call and again after a registration changes; a device with no kernel registered for it pays
/// one field read per call.
/// </summary>
internal static class Kernels
{
    private static readonly List<Registration> Registry = [];

    private static readonly List<WeakReference<Backend>> Devices = [];

    /// <summary>A fresh marker meaning "resolve again" (a new instance each time, so a resolution that raced a change is not kept).</summary>
    public static Delegate?[] Unresolved() => new Delegate?[0];

    /// <summary>Remembers a device so it is told when the registered kernels change.</summary>
    public static void Track(Backend backend)
    {
        lock (Devices)
        {
            Devices.RemoveAll(d => !d.TryGetTarget(out _));
            Devices.Add(new(backend));
        }
    }

    // Every device resolves its slots again on its next call.
    private static void Changed()
    {
        lock (Devices)
        {
            foreach (var d in Devices)
            {
                if (d.TryGetTarget(out var backend))
                {
                    backend.KernelsChanged();
                }
            }
        }
    }

    /// <summary>
    /// Registers <paramref name="kernel"/> for <paramref name="operation"/> on devices of <paramref name="deviceKind"/>
    /// (any case); it runs instead of the device's own kernel wherever <paramref name="requirement"/> holds (always when
    /// null). A later registration for the same operation and kind takes precedence. Returns a handle that removes it.
    /// </summary>
    /// <exception cref="ArgumentException">The kernel is not of the operation's <see cref="Operation.KernelType"/>.</exception>
    public static IDisposable Register(Operation operation, string deviceKind, Delegate kernel, Func<Backend, bool>? requirement = null)
    {
        ArgumentNullException.ThrowIfNull(operation);
        ArgumentException.ThrowIfNullOrWhiteSpace(deviceKind);
        ArgumentNullException.ThrowIfNull(kernel);
        if (kernel.GetType() != operation.KernelType)
        {
            throw new ArgumentException($"A kernel for {operation} is an {operation.KernelType.Name}, not a {kernel.GetType().Name}.", nameof(kernel));
        }

        var registration = new Registration(operation, deviceKind, kernel, requirement);
        lock (Registry)
        {
            Registry.Add(registration);
        }

        Changed();
        return registration;
    }

    /// <summary>The kernel each operation runs on <paramref name="backend"/> (null: the device's own), by <see cref="Operation.Index"/>.</summary>
    public static Delegate?[] Resolve(Backend backend)
    {
        var slots = new Delegate?[OperationIndex.Count];
        Registration[] registrations;
        lock (Registry)
        {
            registrations = [.. Registry];
        }

        foreach (var r in registrations)
        {
            if (r.DeviceKind.Equals(backend.Kind, StringComparison.OrdinalIgnoreCase) && (r.Requirement is null || r.Requirement(backend)))
            {
                slots[r.Operation.Index] = r.Kernel;   // in registration order: the last that applies wins
            }
        }

        return slots;
    }

    private static void Remove(Registration registration)
    {
        bool removed;
        lock (Registry)
        {
            removed = Registry.Remove(registration);
        }

        if (removed)
        {
            Changed();
        }
    }

    private sealed class Registration(Operation operation, string deviceKind, Delegate kernel, Func<Backend, bool>? requirement) : IDisposable
    {
        public Operation Operation { get; } = operation;

        public string DeviceKind { get; } = deviceKind;

        public Delegate Kernel { get; } = kernel;

        public Func<Backend, bool>? Requirement { get; } = requirement;

        public void Dispose() => Remove(this);
    }
}
