// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Idrak.Abstraction.Devices.Cpu;

internal sealed class CpuStorage(CpuBackend backend, float[] data, int length) : Storage(backend, length), IHostStorage
{
    public float[] Data = data;                                         // null while evicted (Backend.Evict)

    /// <inheritdoc />
    public Span<float> HostMemory => Data.AsSpan(0, Length);
}

/// <summary>
/// The CPU device: pooled managed arrays as its memory, and the described lane's products of stored weights
/// (CpuBackend.Lanes.cs), its only kernels.
/// </summary>
internal sealed partial class CpuBackend : Backend, IMemoryReport
{
    public static readonly CpuBackend Instance = new();

    /// <summary>Element-wise work (≈ element operations) from which kernels split across cores: measured, see <see cref="CpuTuning.ParallelElements"/>.</summary>
    internal static int ParallelThreshold => CpuTuning.ParallelElements;

    private readonly Dictionary<int, Stack<float[]>> _pool = [];

    // When each size of the pool was last taken or given back (a count of pool events): the least recently used sizes go
    // first when a miss frees cached blocks (ReleaseCachedFor).
    private readonly Dictionary<int, long> _poolUsed = [];
    private long _poolClock;
    private readonly MemoryAccountant _memory;
    private readonly string? _name;

    private CpuBackend()
    {
        _memory = new(() => ComputeResources.CpuMemoryLimit, "the CPU");
    }

    // A separate CPU backend (Device.CreateSeparateBackend): its own pool, held to its own cap (the CPU's limit when none).
    private CpuBackend(SeparateBackendOptions options)
    {
        long? cap = options.MemoryLimit;
        _memory = new(() => cap ?? ComputeResources.CpuMemoryLimit, options.Name);
        _name = options.Name;
    }

    internal static CpuBackend CreateSeparate(SeparateBackendOptions options) => new(options);

    // The CPU's own backend (Instance) lives for the process; a separate one (it has a name) is disposed by its caller.
    protected override bool LivesForProcess => _name is null;

    // A separate backend's pool goes: the arrays are the garbage collector's once nothing holds them.
    protected override void Dispose(bool disposing)
    {
        lock (_pool)
        {
            foreach (var (length, bucket) in _pool)
            {
                _memory.Freed((long)length * sizeof(float) * bucket.Count);
            }

            _pool.Clear();
            _poolUsed.Clear();
        }
    }

    // The CPU's limits: its described lane takes the rows of x it measures fastest at each shape; it has no attention kernel.
    public override BackendCapabilities Capabilities { get; } = new()
    {
        FewRows = CpuTuning.FewRows,
        DecodeAttentionHeadDim = 0,
        TiledAttentionHeadDim = 0,
    };

    public override string Kind => "cpu";

    public override string Name => _name ?? $"CPU ({Environment.ProcessorCount} threads, {System.Numerics.Vector<float>.Count}-wide SIMD)";

    public override BackendHardware Hardware
    {
        get
        {
            var memory = Memory;
            return new()
            {
                MemoryBytes = memory.Total,
                HardwareKind = "cpu",
                ComputeUnits = Environment.ProcessorCount,
                SubgroupSize = Vector<float>.Count,
                Facts = [.. HardwareFacts(CpuInfo.Current), .. MemoryFacts(memory)],
            };
        }
    }

    // What the processor reports (CpuInfo), for device listings.
    internal static KeyValuePair<string, string>[] HardwareFacts(CpuInfo cpu)
    {
        var facts = new List<KeyValuePair<string, string>>(7) { new("processor", cpu.Model.Length > 0 ? cpu.Model : "not reported") };
        if (cpu.CoreGroups.Length > 0)
        {
            facts.Add(new("core groups", CpuInfo.DescribeGroups(cpu.CoreGroups)));
        }

        facts.Add(new("threads", string.Create(CultureInfo.InvariantCulture, $"{cpu.LogicalProcessors} threads, {cpu.PhysicalCores} cores")));
        facts.Add(new("instruction sets", cpu.InstructionSets));
        facts.Add(new("vectors", string.Create(CultureInfo.InvariantCulture,
            $"{cpu.VectorFloats} floats ({cpu.WidestFloats} at the widest), {cpu.VectorRegisters} registers, fused multiply-add {(cpu.FusedMultiplyAdd ? "yes" : "no")}")));
        facts.Add(new("vector paths", CpuTuning.DescribeVectorPaths()));
        facts.Add(new("caches", string.Create(CultureInfo.InvariantCulture,
            $"L1d {cpu.L1 >> 10} KiB, L2 {cpu.L2 >> 10} KiB per core, L3 {(cpu.L3 > 0 ? $"{cpu.L3 >> 20} MiB" : "unknown")}, line {cpu.CacheLine} B{(cpu.CachesReported ? "" : " (not reported: fallbacks)")}")));
        return [.. facts];
    }

    public override Storage Allocate(int length, bool zeroed)
    {
        ThrowIfDisposed();
        long bytes = (long)length * sizeof(float);
        bool releaseCache = _memory.MustReleaseCacheFor(bytes); // throws when over the in-use limit
        float[]? data = null;
        lock (_pool)
        {
            if (_pool.TryGetValue(length, out var bucket) && bucket.Count > 0)
            {
                data = bucket.Pop();
                _poolUsed[length] = ++_poolClock;
                _memory.Reused(bytes);
            }
        }

        if (data is null)
        {
            // A miss while the pool holds at least as much in other sizes: as much of those goes first, the least recently
            // used sizes first (rule 77), so the process does not grow. Shapes that repeat (a training step, a prompt pass's
            // layers) keep hitting their sizes; shapes that change (lines of other widths, page after page) would otherwise
            // leave every earlier size cached and grow the process until the machine runs out. Emptying the whole pool on
            // every such miss (as before) also dropped the sizes still in use: a prompt pass interleaves a dozen sizes a
            // layer, so each layer allocated its activations anew (a CPU bfloat16 pass, about 1 GB of fresh arrays).
            if (releaseCache)
            {
                ReleaseCachedMemory();
            }
            else if (_memory.Usage.Cached >= bytes)
            {
                ReleaseCachedFor(bytes);
            }

            data = zeroed ? new float[length] : GC.AllocateUninitializedArray<float>(length);
            _memory.Allocated(bytes);
        }
        else if (zeroed)
        {
            Array.Clear(data);
        }

        return new CpuStorage(this, data, length);
    }

    protected override void Detach(Storage storage) => ((CpuStorage)storage).Data = null!;

    protected override void Attach(Storage storage, Storage fresh) => ((CpuStorage)storage).Data = ((CpuStorage)fresh).Data;

    public override void Return(Storage storage)
    {
        if (storage is MappedStorage mapped)
        {
            ReleaseMapping(mapped);                                        // a file's mapping (plan 37, W3): not pooled
            return;
        }

        var data = ((CpuStorage)storage).Data;
        lock (_pool)
        {
            if (IsDisposed)
            {
                return;                                                    // a tensor that outlived its backend: nothing to keep
            }

            if (!_pool.TryGetValue(data.Length, out var bucket))
            {
                _pool[data.Length] = bucket = new Stack<float[]>();
            }

            bucket.Push(data);
            _poolUsed[data.Length] = ++_poolClock;
            _memory.Returned((long)data.Length * sizeof(float));
        }
    }

    // Frees cached blocks, the sizes least recently taken or given back first, until at least `bytes` are freed (or the
    // pool is empty). A few sizes at most: each round finds the oldest by a scan, nothing allocated.
    private void ReleaseCachedFor(long bytes)
    {
        lock (_pool)
        {
            long freed = 0;
            while (freed < bytes)
            {
                int oldest = -1;
                long stamp = long.MaxValue;
                foreach (var (length, bucket) in _pool)
                {
                    if (bucket.Count > 0 && _poolUsed.GetValueOrDefault(length) < stamp)
                    {
                        (oldest, stamp) = (length, _poolUsed.GetValueOrDefault(length));
                    }
                }

                if (oldest < 0)
                {
                    return;
                }

                var blocks = _pool[oldest];
                long each = (long)oldest * sizeof(float);
                while (blocks.Count > 0 && freed < bytes)
                {
                    blocks.Pop();
                    _memory.Freed(each);
                    freed += each;
                }
            }
        }
    }

    public override MemoryUsage GetMemoryUsage() => _memory.Usage;

    public override void ResetPeakMemoryUsage() => _memory.ResetPeak();

    // System memory not in use (CpuBackend.Memory: the system's MemAvailable and its group limit where reported, held to
    // the runtime's heap limit where it has one; elsewhere the runtime's total less its load).
    long? IMemoryReport.AvailableMemory() => Memory.Available;

    // The driver's free memory: the CPU has no driver of its own (AvailableMemory is what the system reports).
    long? IMemoryReport.FreeMemory() => null;

    public override void ReleaseCachedMemory()
    {
        lock (_pool)
        {
            foreach (var (length, bucket) in _pool)
            {
                _memory.Freed((long)length * sizeof(float) * bucket.Count);
                bucket.Clear();
            }
        }
    }

    public override void Upload(ReadOnlySpan<float> source, Storage destination)
    {
        ThrowIfDisposed();
        if (destination is MappedStorage mapped)
        {
            WriteIntoMapping(source, mapped, 0);
            return;
        }

        source.CopyTo(D(destination));
    }

    public override void Download(Storage source, Span<float> destination)
    {
        ThrowIfDisposed();
        if (source is MappedStorage mapped)
        {
            CopyOutOfMapping(mapped, 0, destination, "downloaded (read whole, moved to another device)");
            return;
        }

        D(source).AsSpan(0, destination.Length).CopyTo(destination);
    }

    public override void DownloadRange(Storage source, int offset, Span<float> destination)
    {
        ThrowIfDisposed();
        if (source is MappedStorage mapped)
        {
            CopyOutOfMapping(mapped, offset, destination, "downloaded (read whole, moved to another device)");
            return;
        }

        D(source).AsSpan(offset, destination.Length).CopyTo(destination);
    }

    public override void UploadRange(ReadOnlySpan<float> source, Storage destination, int offset)
    {
        ThrowIfDisposed();
        if (destination is MappedStorage mapped)
        {
            WriteIntoMapping(source, mapped, offset);
            return;
        }

        source.CopyTo(D(destination).AsSpan(offset, source.Length));
    }

    public override void Copy(Storage x, Storage y, int n)
    {
        if (x is MappedStorage mapped)
        {
            CopyOutOfMapping(mapped, 0, D(y).AsSpan(0, n), "copied (cloned)");
            return;
        }

        D(x).AsSpan(0, n).CopyTo(D(y));
    }

    public override void Synchronize()
    {
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float[] D(Storage s) => ((CpuStorage)s).Data;
}
