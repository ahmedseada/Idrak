// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections;
using System.Collections.Concurrent;
using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction.Operations;

/// <summary>
/// Which kernel each operation runs on a device (plan 9): a call to an operation (<c>Backend.Name(...)</c>) runs the
/// device's own (<c>Backend.NameKernel</c>, whose default body is the host fallback, a composition of other operations,
/// or none). <see cref="Chain"/> tells which one each operation runs on a device, <see cref="Trace"/> counts the calls,
/// and <see cref="HostCalls"/> the host fallbacks.
/// </summary>
public static class Kernels
{
    // Starting and ending a device's trace.
    private static readonly Lock TraceGate = new();

    // Per backend type, which operations it has a kernel of its own for.
    private static readonly ConcurrentDictionary<Type, OwnKernelSet> Overrides = new();

    /// <summary>
    /// Which kernel each operation runs on <paramref name="backend"/> now, in <see cref="Operation.Index"/> order (the
    /// library's operations, <see cref="Ops.All"/>): the device's own, the composed default, the host fallback, or none.
    /// </summary>
    public static IReadOnlyList<KernelChoice> Chain(Backend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        var own = backend.OwnKernels;
        var chain = new KernelChoice[Ops.All.Count];
        foreach (var operation in Ops.All)
        {
            var source = own.Contains(operation) ? KernelSource.Device
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
        lock (TraceGate)
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
    /// The host fallbacks of <paramref name="backend"/> by operation, counted while a telemetry hook listened at
    /// <see cref="TelemetryLevel.Devices"/> (nothing is counted by operation while nobody listens): a copy.
    /// </summary>
    public static IReadOnlyDictionary<string, long> HostCallsByOperation(Backend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        return backend.HostCallsByOperation;
    }

    /// <summary>A fresh marker meaning "resolve again" (a new instance each time, so a resolution that raced a change is not kept).</summary>
    internal static Delegate?[] Unresolved() => new Delegate?[0];

    /// <summary>Ends <paramref name="trace"/> on its device.</summary>
    internal static void EndTrace(KernelTrace trace)
    {
        lock (TraceGate)
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
        return slots;
    }

    // Which operations the backend has a kernel of its own for (Backend.OwnKernels), built once per device type: the
    // capabilities a device implements and the kernels it overrides belong to its type.
    internal static OwnKernelSet OwnKernels(Backend backend) =>
        Overrides.GetOrAdd(backend.GetType(), static (_, b) => new OwnKernelSet(Backend.OwnKernelsOf(b)), backend);

}

/// <summary>The library's operations a device has a kernel of its own for (<see cref="Backend.OwnKernels"/>), by index.</summary>
internal sealed class OwnKernelSet(bool[] own) : IReadOnlySet<Operation>
{
    private readonly bool[] _own = own;
    private readonly int _count = own.Count(static o => o);

    public int Count => _count;

    // Whether the device has a kernel of its own for the operation at `index` (a library operation's).
    public bool Has(int index) => _own[index];

    public bool Contains(Operation item) => item is not null && _own[item.Index];

    public IEnumerator<Operation> GetEnumerator()
    {
        foreach (var operation in Ops.All)
        {
            if (_own[operation.Index])
            {
                yield return operation;
            }
        }
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    public bool IsProperSubsetOf(IEnumerable<Operation> other) => Others(other) is var set && set.Count > _count && IsSubsetOf(set);

    public bool IsProperSupersetOf(IEnumerable<Operation> other) => Others(other) is var set && set.Count < _count && IsSupersetOf(set);

    public bool IsSubsetOf(IEnumerable<Operation> other) => Others(other) is var set && this.All(set.Contains);

    public bool IsSupersetOf(IEnumerable<Operation> other) => other.All(Contains);

    public bool Overlaps(IEnumerable<Operation> other) => other.Any(Contains);

    public bool SetEquals(IEnumerable<Operation> other) => Others(other) is var set && set.Count == _count && set.All(Contains);

    private static HashSet<Operation> Others(IEnumerable<Operation> other) => other as HashSet<Operation> ?? [.. other];
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

    /// <summary>
    /// The calls of <paramref name="operation"/> through the dispatcher so far (<c>Backend.Name(...)</c>).
    /// </summary>
    public long Calls(Operation operation)
    {
        ArgumentNullException.ThrowIfNull(operation);
        return Interlocked.Read(ref _calls[operation.Index]);
    }

    /// <summary>The host fallbacks so far.</summary>
    public long HostCalls => _hostCalls.Values.Sum();

    /// <summary>
    /// The host fallbacks so far by operation name (<see cref="Operation.Name"/> without the overload number, or a
    /// copy's method name), a snapshot.
    /// </summary>
    public IReadOnlyDictionary<string, long> HostCallsByOperation => new Dictionary<string, long>(_hostCalls, StringComparer.Ordinal);

    /// <summary>Stops counting; the counts stay readable.</summary>
    public void Dispose() => Kernels.EndTrace(this);

    internal void Called(int operation)
    {
        Interlocked.Increment(ref _calls[operation]);
    }

    internal void HostCalled(string operation) => _hostCalls.AddOrUpdate(operation, 1, static (_, n) => n + 1);
}
