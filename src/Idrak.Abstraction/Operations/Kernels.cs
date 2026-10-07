// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction.Operations;

/// <summary>
/// The kernel table and dispatcher (plan 9): kernels registered for an operation on a kind of device ("cpu", "cuda",
/// "vulkan", …), each with an optional requirement on the device. A call to an operation (<c>Backend.Name(...)</c>) runs
/// the kernel registered last whose requirement holds, else the device's own (<c>Backend.NameKernel</c>, whose default
/// body is the host fallback, a composition of other operations, or none). <see cref="Chain"/> tells which one each
/// operation runs on a device, <see cref="Trace"/> counts the calls, and <see cref="HostCalls"/> the host fallbacks.
/// </summary>
/// <remarks>
/// Each device resolves its slots on its first call and again after a registration changes; a device with nothing
/// registered for it and no trace pays one field read per call.
/// </remarks>
public static class Kernels
{
    private static readonly List<Registration> Registry = [];

    private static readonly List<WeakReference<Backend>> Devices = [];

    // Per backend type, which operations it overrides (its own kernel), by index.
    private static readonly ConcurrentDictionary<Type, bool[]> Overrides = new();

    /// <summary>
    /// Registers <paramref name="kernel"/> for <paramref name="operation"/> on devices of <paramref name="deviceKind"/>
    /// (any case); it runs instead of the device's own kernel wherever <paramref name="requirement"/> holds (always when
    /// null). A later registration for the same operation and kind takes precedence. Returns a handle that removes it.
    /// </summary>
    /// <remarks>
    /// A kernel receives the device's backend first, so it can run the device's own kernel for the cases it does not
    /// handle (<c>backend.SoftmaxKernel(...)</c>). A requirement reads what the device reports
    /// (<see cref="Backend.Capabilities"/>), never a card's name.
    /// </remarks>
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

    /// <summary>
    /// Which kernel each operation runs on <paramref name="backend"/> now, in <see cref="Ops.All"/> order: a registered
    /// one, the device's own, the composed default, the host fallback, or none.
    /// </summary>
    public static IReadOnlyList<KernelChoice> Chain(Backend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        var registered = Resolve(backend);
        if (!Overrides.TryGetValue(backend.GetType(), out var own))
        {
            own = OwnKernels(backend.GetType());
            Overrides[backend.GetType()] = own;
        }

        var chain = new KernelChoice[Ops.All.Count];
        foreach (var operation in Ops.All)
        {
            var source = registered[operation.Index] is not null ? KernelSource.Registered
                : own[operation.Index] ? KernelSource.Device
                : operation.Fallback;
            chain[operation.Index] = new KernelChoice(operation, source);
        }

        return chain;
    }

    /// <summary>
    /// Starts counting the calls of each operation on <paramref name="backend"/> and its host fallbacks by operation,
    /// until the trace is disposed. Calls cost a little more while a trace is on (they leave the inlined fast path). One
    /// trace per device at a time.
    /// </summary>
    /// <exception cref="InvalidOperationException">The device is traced already.</exception>
    public static KernelTrace Trace(Backend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        var trace = new KernelTrace(backend);
        lock (Devices)
        {
            if (backend.Trace is not null)
            {
                throw new InvalidOperationException($"{backend.Name} is traced already; dispose that trace first.");
            }

            backend.Trace = trace;
        }

        return trace;
    }

    /// <summary>The operations <paramref name="backend"/> has run through the host fallback since it started.</summary>
    public static long HostCalls(Backend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        return backend.HostCalls;
    }

    /// <summary>
    /// Whether a failure on a device of <paramref name="kind"/> would be retried on the host (for the hint of a
    /// <see cref="DeviceFailed"/> event): <see cref="Backend.RetryOnHost"/> of a live device of that kind, or, before one
    /// starts, <c>IDRAK_RETRY_ON_HOST</c>.
    /// </summary>
    internal static bool RetriesOnHost(string kind)
    {
        bool any = false;
        lock (Devices)
        {
            foreach (var d in Devices)
            {
                if (d.TryGetTarget(out var backend) && backend.Kind.Equals(kind, StringComparison.OrdinalIgnoreCase))
                {
                    any = true;
                    if (backend.RetryOnHost)
                    {
                        return true;
                    }
                }
            }
        }

        return !any && Backend.RetryOnHostFrom(Environment.GetEnvironmentVariable("IDRAK_RETRY_ON_HOST"), kind);
    }

    /// <summary>A fresh marker meaning "resolve again" (a new instance each time, so a resolution that raced a change is not kept).</summary>
    internal static Delegate?[] Unresolved() => new Delegate?[0];

    /// <summary>Remembers a device so it is told when the registered kernels change.</summary>
    internal static void Track(Backend backend)
    {
        lock (Devices)
        {
            Devices.RemoveAll(d => !d.TryGetTarget(out _));
            Devices.Add(new(backend));
        }
    }

    /// <summary>Ends <paramref name="trace"/> on its device.</summary>
    internal static void EndTrace(KernelTrace trace)
    {
        lock (Devices)
        {
            if (ReferenceEquals(trace.Backend.Trace, trace))
            {
                trace.Backend.Trace = null;
            }
        }
    }

    /// <summary>The kernel each operation runs on <paramref name="backend"/> (null: the device's own), by <see cref="Operation.Index"/>.</summary>
    internal static Delegate?[] Resolve(Backend backend)
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

    // Which operations a backend type overrides: its NameKernel is declared below Backend.
    private static bool[] OwnKernels([DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicMethods)] Type type)
    {
        var own = new bool[Ops.All.Count];
        foreach (var method in type.GetMethods(BindingFlags.Public | BindingFlags.Instance))
        {
            if (method.DeclaringType != typeof(Backend) && method.IsVirtual && method.GetBaseDefinition().DeclaringType == typeof(Backend)
                && Find(method) is { } operation)
            {
                own[operation.Index] = true;
            }
        }

        return own;
    }

    // The operation whose kernel method this is (by name and parameter types), or null.
    private static Operation? Find(MethodInfo method)
    {
        string signature = string.Join(",", Array.ConvertAll(method.GetParameters(), p => p.ParameterType.Name));
        foreach (var operation in Ops.All)
        {
            if (operation.KernelMethod == method.Name && operation.Signature == signature)
            {
                return operation;
            }
        }

        return null;
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

/// <summary>Which kernel runs an operation on a device (<see cref="Kernels.Chain"/>).</summary>
/// <param name="Operation">The operation.</param>
/// <param name="Source">Where its kernel comes from.</param>
public readonly record struct KernelChoice(Operation Operation, KernelSource Source);

/// <summary>
/// The calls of each operation on one device and its host fallbacks, counted from <see cref="Kernels.Trace"/> until
/// disposed.
/// </summary>
public sealed class KernelTrace : IDisposable
{
    private readonly long[] _calls = new long[Ops.All.Count];
    private readonly ConcurrentDictionary<string, long> _hostCalls = new(StringComparer.Ordinal);

    internal KernelTrace(Backend backend) => Backend = backend;

    /// <summary>The device traced.</summary>
    public Backend Backend { get; }

    /// <summary>The calls of <paramref name="operation"/> through the dispatcher (<c>Backend.Name(...)</c>) so far.</summary>
    public long Calls(Operation operation) => Interlocked.Read(ref _calls[operation.Index]);

    /// <summary>The host fallbacks so far.</summary>
    public long HostCalls => _hostCalls.Values.Sum();

    /// <summary>
    /// The host fallbacks so far by operation name (<see cref="Operation.Name"/> without the overload number, or a
    /// copy's method name), a snapshot.
    /// </summary>
    public IReadOnlyDictionary<string, long> HostCallsByOperation => new Dictionary<string, long>(_hostCalls, StringComparer.Ordinal);

    /// <summary>Stops counting; the counts stay readable.</summary>
    public void Dispose() => Kernels.EndTrace(this);

    internal void Called(int operation) => Interlocked.Increment(ref _calls[operation]);

    internal void HostCalled(string operation) => _hostCalls.AddOrUpdate(operation, 1, static (_, n) => n + 1);
}
