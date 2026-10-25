// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Idrak.Abstraction.Devices;

/// <summary>
/// Tensors whose values are not read yet (plan 23, C8b): a model built on the <see cref="Device"/> "deferred" holds no
/// memory for its weights; each tensor knows where its values come from (<see cref="SetSource"/>: a checkpoint's bytes)
/// and reads them when it moves to a real device (<see cref="Module.To(Device)"/>), one tensor at a time. So a pipeline stage or a
/// parameter shard reads only its own tensors from a checkpoint and the whole model is never built or held anywhere; any
/// family's builder works unchanged (the build runs on this device and every weight it reads is a placeholder,
/// <see cref="Placeholder"/>). Inside <see cref="StayDeferred"/> a move gives a tensor on the real device whose memory is
/// given back at once (<see cref="Backend.Evict"/>) and read on its first <see cref="Backend.Restore"/>, for a rank that
/// decides later which of them it holds.
/// </summary>
public static class DeferredTensors
{
    /// <summary>The device kind ("deferred").</summary>
    public const string Kind = "deferred";

    [ThreadStatic]
    private static Dictionary<int, float[]>? _placeholders;

    [ThreadStatic]
    private static int _staying;

    /// <summary>The device deferred tensors live on: no memory, values read from their source when they leave it.</summary>
    public static Device Device => Device.Get(Kind);

    /// <summary>Whether <paramref name="tensor"/> lives on the deferred device.</summary>
    public static bool IsDeferred(Tensor tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        return tensor.Storage is DeferredStorage;
    }

    /// <summary>The name its source was given (<see cref="SetSource"/>), or null.</summary>
    public static string? SourceOf(Tensor tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        return tensor.Storage is DeferredStorage { Source: not null } d ? d.SourceName : null;
    }

    /// <summary>
    /// Where <paramref name="tensor"/>'s values come from (it must be deferred): <paramref name="read"/> fills a span of
    /// its size, once each time the tensor leaves the deferred device; <paramref name="name"/> names it in messages.
    /// </summary>
    public static void SetSource(Tensor tensor, string name, Action<Span<float>> read)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        ArgumentNullException.ThrowIfNull(read);
        if (tensor.Storage is not DeferredStorage d)
        {
            throw new ArgumentException($"{name} is on {tensor.Device}, not the deferred device: its values are read already.", nameof(tensor));
        }

        d.SetSource(name, read);
    }

    /// <summary>
    /// A build on the deferred device on this thread: until disposed, <see cref="Placeholder"/> hands out the arrays a weight
    /// source returns instead of reading, and an upload of one keeps nothing (the tensor's source is set afterwards).
    /// </summary>
    public static IDisposable Build()
    {
        var outer = _placeholders;
        _placeholders = [];
        return new Restorer(() => _placeholders = outer);
    }

    /// <summary>
    /// An array of <paramref name="length"/> floats standing for values not read (inside <see cref="Build"/>): one per length
    /// on this thread, never written, so it costs address space and no memory.
    /// </summary>
    /// <exception cref="InvalidOperationException">Outside <see cref="Build"/>.</exception>
    public static float[] Placeholder(int length)
    {
        var placeholders = _placeholders ?? throw new InvalidOperationException("Placeholders are handed out inside DeferredTensors.Build only.");
        if (!placeholders.TryGetValue(length, out var array))
        {
            placeholders[length] = array = GC.AllocateUninitializedArray<float>(length);
        }

        return array;
    }

    /// <summary>
    /// Until disposed, a deferred tensor moved to a device on this thread (<see cref="Module.To(Device)"/>) becomes a tensor there
    /// whose memory is given back at once, its values read from the source on its first <see cref="Backend.Restore"/>.
    /// </summary>
    public static IDisposable StayDeferred()
    {
        _staying++;
        return new Restorer(() => _staying--);
    }

    // Whether `values` is this thread's placeholder of its length.
    internal static bool IsPlaceholder(ReadOnlySpan<float> values) =>
        !values.IsEmpty && _placeholders is { } placeholders && placeholders.TryGetValue(values.Length, out var array)
        && Unsafe.AreSame(ref MemoryMarshal.GetReference(values), ref MemoryMarshal.GetArrayDataReference(array));

    // A move of a deferred tensor inside StayDeferred: the evicted tensor on `device`, or null (move it as any tensor: its
    // values are read now).
    internal static Tensor? TryMove(Tensor tensor, Device device)
    {
        if (_staying == 0 || tensor.Storage is not DeferredStorage { Source: not null } deferred)
        {
            return null;
        }

        return Tensor.Evicted(tensor.Shape, device, tensor.RequiresGrad, storage =>
        {
            var values = ArrayPool<float>.Shared.Rent(Math.Max(1, deferred.Length));
            try
            {
                deferred.Read(values.AsSpan(0, deferred.Length));
                storage.Backend.Upload(values.AsSpan(0, deferred.Length), storage);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(values);
            }
        });
    }

    private sealed class Restorer(Action restore) : IDisposable
    {
        private Action? _restore = restore;

        public void Dispose() => Interlocked.Exchange(ref _restore, null)?.Invoke();
    }
}

/// <summary>A deferred tensor's storage: no memory, its values read from a source (or kept, for values computed in a build).</summary>
internal sealed class DeferredStorage(DeferredBackend backend, int length) : Storage(backend, length)
{
    private Action<Span<float>>? _source;

    public Action<Span<float>>? Source => _source;

    public string? SourceName { get; private set; }

    // Values the build computed (a rotary table) or an operation wrote: kept, since no source has them.
    public float[]? Kept;

    public void SetSource(string name, Action<Span<float>> read)
    {
        (_source, SourceName) = (read, name);
        ((DeferredBackend)Backend).Drop(this);                         // the source is authoritative
    }

    // The values into `target` (its length): the kept ones (written since), the source's, or zeros.
    public void Read(Span<float> target)
    {
        if (Kept is { } kept)
        {
            kept.AsSpan(0, target.Length).CopyTo(target);
        }
        else if (_source is { } source)
        {
            source(target);
        }
        else
        {
            target.Clear();
        }
    }
}

/// <summary>The deferred device: storages without memory (see <see cref="DeferredTensors"/>); kept values are its only bytes.</summary>
internal sealed class DeferredBackend : Backend
{
    public static readonly DeferredBackend Instance = new();
    private readonly MemoryAccountant _memory = new(() => null, "the deferred device");

    public override BackendCapabilities Capabilities { get; } = Cpu.CpuBackend.Instance.Capabilities;

    public override string Kind => DeferredTensors.Kind;

    public override string Name => "deferred (no memory: each tensor's values read from its source when it moves to a device)";

    public override Storage Allocate(int length, bool zeroed) => new DeferredStorage(this, length);

    public override void Return(Storage storage) => Drop((DeferredStorage)storage);

    internal void Drop(DeferredStorage storage)
    {
        if (Interlocked.Exchange(ref storage.Kept, null) is { } kept)
        {
            _memory.Returned((long)kept.Length * sizeof(float));
            _memory.Freed((long)kept.Length * sizeof(float));
        }
    }

    protected override void Detach(Storage storage) => Drop((DeferredStorage)storage);

    protected override void Attach(Storage storage, Storage fresh) => ((DeferredStorage)storage).Kept = Interlocked.Exchange(ref ((DeferredStorage)fresh).Kept, null);

    public override MemoryUsage GetMemoryUsage() => _memory.Usage;

    public override void ReleaseCachedMemory()
    {
    }

    public override void Upload(ReadOnlySpan<float> source, Storage destination)
    {
        var d = (DeferredStorage)destination;
        if (DeferredTensors.IsPlaceholder(source))
        {
            return;                                                    // a weight not read: its source is set after the build
        }

        source.CopyTo(Keep(d, read: source.Length < d.Length));       // written values are kept: they are what it holds now
    }

    public override void UploadRange(ReadOnlySpan<float> source, Storage destination, int offset) => source.CopyTo(Keep((DeferredStorage)destination, read: true).AsSpan(offset));

    public override void Download(Storage source, Span<float> destination) => DownloadRange(source, 0, destination);

    public override void DownloadRange(Storage source, int offset, Span<float> destination)
    {
        var d = (DeferredStorage)source;
        if (d.Kept is { } kept)
        {
            kept.AsSpan(offset, destination.Length).CopyTo(destination);
            return;
        }

        if (offset == 0 && destination.Length == d.Length)
        {
            d.Read(destination);
            return;
        }

        var values = ArrayPool<float>.Shared.Rent(Math.Max(1, d.Length));
        try
        {
            d.Read(values.AsSpan(0, d.Length));
            values.AsSpan(offset, destination.Length).CopyTo(destination);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(values);
        }
    }

    public override void Synchronize()
    {
    }

    // The storage's kept values (made from its source, when `read`, or zeros): an operation writes them.
    private float[] Keep(DeferredStorage d, bool read)
    {
        if (d.Kept is { } kept)
        {
            return kept;
        }

        var values = new float[Math.Max(1, d.Length)];
        if (read && d.Source is not null)
        {
            d.Read(values.AsSpan(0, d.Length));
        }

        _memory.Allocated((long)values.Length * sizeof(float));
        d.Kept = values;
        return values;
    }
}

/// <summary>The provider of the deferred device (by name only: never listed, never the default).</summary>
internal sealed class DeferredProvider : IDeviceProvider
{
    private volatile bool _started;

    public string Name => DeferredTensors.Kind;

    public string Summary => "a model's tensors before they are read (a pipeline stage's or a shard's load), each read from its source when it moves to a device";

    public DeviceType Type => DeviceType.Other;

    public int Count => 1;

    public Backend Create(int ordinal)
    {
        _started = true;
        return DeferredBackend.Instance;
    }

    public bool IsStarted(int ordinal) => _started;

    public bool Listed(int ordinal) => false;

    public string? Note(int ordinal) => "a model's tensors before they are read (a pipeline stage's or a shard's load): by name only";
}
