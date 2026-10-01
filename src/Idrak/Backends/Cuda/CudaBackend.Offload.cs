// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using static Idrak.Backends.Cuda.CudaDriver;

namespace Idrak.Backends.Cuda;

// Offloading (ComputeResources.OffloadToHostMemory): CUDA's implementation of IMemoryOffload. An offloaded storage lives
// in pinned system memory mapped into the GPU's address space, so every kernel can still read and write it (over PCIe).
//
// - Moving: a storage moves by an asynchronous copy on the work stream and a swap of its address; the old block goes back
//   to its pool, where stream order keeps it from being reused before the copy has read it. Moving is refused while a
//   recorded graph exists (graphs hold raw addresses) or one is being recorded.
// - Cold first: allocations that found the GPU full and went to system memory are counted; at the next step boundary
//   (Rebalance, called by Optimizer.ZeroGrad) that many bytes of cold storages (optimizer state, then frozen weights,
//   largest first) move out instead, so the next step's activations fit on the GPU. Moving happens only at step
//   boundaries, where no operation holds a raw address.
// - Staging: before a layer computes, its offloaded weights get a GPU copy (Pointer on the copy, Home on the system
//   block) until the stage token is disposed; the next layer's copies start on a separate copy stream while this one
//   computes (Prefetch). Copy blocks come from the caching pool, and layers of one model share shapes, so after the
//   first pass staging allocates nothing.
// - Returning: when the GPU has room again (beyond the return headroom (ComputeResources.OffloadReturnHeadroom), not counting the cache, so a
//   step's own activations are never pushed out), offloaded storages come back, hottest first.
internal sealed unsafe partial class CudaBackend : IMemoryOffload
{
    public override IMemoryOffload? Offload => this;

    /// <summary>Recorded graphs not yet destroyed: they hold raw addresses, so nothing moves while any exists.</summary>
    private int _liveGraphs;

    // Under the pool lock: storages colder than Hot on the GPU (candidates to move out), storages in system memory (staged
    // or not), and the bytes of hot allocations that went to system memory since the last Rebalance.
    private readonly HashSet<CudaStorage> _cold = [];
    private readonly HashSet<CudaStorage> _away = [];
    private int _awayCount;
    private long _spilled;
    private long _peakSpill;                                               // the largest spill since the last explicit return

    // Background copies to the GPU (Prefetch): the copy block and the batch whose event marks the copy done, and the copies
    // dropped (their storage released, or not staged in time) whose blocks go back to the pool once the copy finished.
    private sealed class PrefetchBatch(IntPtr done)
    {
        public readonly IntPtr Done = done;
        public int Pending;
    }

    private readonly Dictionary<CudaStorage, (ulong Pointer, int Capacity, PrefetchBatch Batch)> _prefetched = [];
    private readonly List<(ulong Pointer, int Capacity, PrefetchBatch Batch)> _stalePrefetches = [];
    private PrefetchBatch? _latestBatch;
    private IntPtr _copyStream;
    private readonly Stack<IntPtr> _offloadEvents = new();

    /// <summary>Storages moved to system memory / back to the GPU, layers staged, and storages prefetched (for tests and diagnostics).</summary>
    internal long MovedToHost, MovedToDevice, StagedStorages, PrefetchedStorages;

    public int OffloadedCount => Volatile.Read(ref _awayCount);

    // Under the pool lock.
    private bool Movable => Volatile.Read(ref _liveGraphs) == 0 && _captureFree is null;

    private bool OnHostLocked(CudaStorage s) => s.Home != 0 || _hostBlocks.ContainsKey(s.Pointer);

    private void AddAwayLocked(CudaStorage s)
    {
        _away.Add(s);
        _awayCount = _away.Count;
    }

    private void RemoveAwayLocked(CudaStorage s)
    {
        _away.Remove(s);
        _awayCount = _away.Count;
    }

    private CudaStorage? Mine(Storage storage) => storage is CudaStorage s && ReferenceEquals(s.Backend, this) ? s : null;

    public bool IsOffloaded(Storage storage)
    {
        if (Mine(storage) is not { } s || OffloadedCount == 0)
        {
            return false;
        }

        lock (_pool)
        {
            return s.Pointer != 0 && OnHostLocked(s);
        }
    }

    public void SetPriority(Storage storage, OffloadPriority priority)
    {
        if (Mine(storage) is not { } s)
        {
            return;
        }

        lock (_pool)
        {
            s.OffloadPriority = priority;
            if (!s.Alive || s.Evicted || s.Pointer == 0)
            {
                return;
            }

            if (OnHostLocked(s))
            {
                // Cold data that went to system memory when allocated is where it belongs: nothing to make room for.
                if (priority > OffloadPriority.Hot)
                {
                    _spilled = Math.Max(0, _spilled - BlockBytes(s.Length));
                }
            }
            else if (priority > OffloadPriority.Hot)
            {
                _cold.Add(s);
            }
            else
            {
                _cold.Remove(s);
            }
        }
    }

    public bool MoveToHost(Storage storage, bool keep)
    {
        if (Mine(storage) is not { } s)
        {
            return false;
        }

        MakeCurrent();
        using var use = UseStream();
        lock (_pool)
        {
            if (s.Alive && !s.Evicted && s.Pointer != 0 && OnHostLocked(s))
            {
                s.KeepOnHost |= keep;
                return true;
            }
        }

        if (!MoveOut(s))
        {
            return false;
        }

        s.KeepOnHost = keep;
        return true;
    }

    public bool MoveToDevice(Storage storage)
    {
        if (Mine(storage) is not { } s)
        {
            return false;
        }

        MakeCurrent();
        using var use = UseStream();
        lock (_pool)
        {
            if (s.Alive && !s.Evicted && s.Pointer != 0 && !OnHostLocked(s))
            {
                return true;
            }
        }

        DropPrefetches(all: true);
        return MoveIn(s, keepFree: 0, useCache: true, released: null);
    }

    public int Rebalance(bool makeRoom = true)
    {
        lock (_pool)
        {
            if (_spilled == 0 && _awayCount == 0 && _prefetched.Count == 0 && _stalePrefetches.Count == 0)
            {
                return 0;                                                  // the usual case: nothing offloaded
            }
        }

        MakeCurrent();
        using var use = UseStream();
        DropPrefetches(all: true);
        long spilled;
        lock (_pool)
        {
            if (!Movable)
            {
                return 0;
            }

            (spilled, _spilled) = (_spilled, 0);
            _peakSpill = makeRoom ? Math.Max(_peakSpill, spilled) : 0;
        }

        int moved = 0;
        if (makeRoom && spilled > 0 && ComputeResources.OffloadColdFirst)
        {
            // The last step put `spilled` bytes of its own data in system memory: move that much cold data out instead.
            CudaStorage[] cold;
            lock (_pool)
            {
                cold = [.. _cold.OrderByDescending(c => c.OffloadPriority).ThenByDescending(c => c.Length)];
            }

            long freed = 0;
            foreach (var c in cold)
            {
                if (freed >= spilled)
                {
                    break;
                }

                if (MoveOut(c))
                {
                    freed += BlockBytes(c.Length);
                    moved++;
                }
            }

            if (moved > 0)
            {
                return moved;                                              // nothing comes back in the round that made room
            }
        }

        if (!ComputeResources.ReturnOffloadedTensors || OffloadedCount == 0)
        {
            return moved;
        }

        CudaStorage[] away;
        lock (_pool)
        {
            away = [.. _away.Where(a => !a.KeepOnHost && a.Home == 0).OrderBy(a => a.OffloadPriority).ThenByDescending(a => a.Length)];
        }

        // New device memory only (the cache holds the step's own activations), keeping the headroom free; cold data also
        // leaves room for the largest spill seen, so it does not come back only to be pushed out by the next step.
        var released = new List<(ulong Pointer, int Capacity)>();
        long peak;
        lock (_pool)
        {
            peak = _peakSpill;
        }

        foreach (var a in away)
        {
            long keepFree = ReturnHeadroom + (a.OffloadPriority > OffloadPriority.Hot ? peak : 0);
            if (MoveIn(a, keepFree, useCache: false, released))
            {
                moved++;
            }
        }

        if (released.Count > 0)
        {
            // Pinned memory is locked system RAM: give back the blocks that came home once their copies have run.
            Check(cuStreamSynchronize(_stream), nameof(cuStreamSynchronize));
            lock (_pool)
            {
                foreach (var (pointer, _) in released)
                {
                    Check(cuMemFreeHost(_hostBlocks[pointer]), nameof(cuMemFreeHost));
                    _hostBlocks.Remove(pointer);
                }
            }
        }

        return moved;
    }

    public void Prefetch(IReadOnlyList<Storage> storages)
    {
        if (OffloadedCount == 0 || !ComputeResources.PrefetchOffloadedWeights || storages.Count == 0)
        {
            return;
        }

        MakeCurrent();
        using var use = UseStream();
        lock (_pool)
        {
            if (_captureFree is not null)
            {
                return;
            }
        }

        DropPrefetches(all: false);
        PrefetchBatch? batch = null;
        foreach (var storage in storages)
        {
            if (Mine(storage) is not { } s)
            {
                continue;
            }

            IntPtr host;
            lock (_pool)
            {
                if (!s.Alive || s.Evicted || s.Home != 0 || _prefetched.ContainsKey(s) || !_hostBlocks.TryGetValue(s.Pointer, out host))
                {
                    continue;
                }
            }

            ulong block = TryDeviceBlock(s.Length, 0, useCache: true, out int capacity);
            if (block == 0)
            {
                break;                                                     // no room: the layer reads system memory directly
            }

            if (batch is null)
            {
                // The copy stream first waits for the work queued so far: pooled blocks may still be read by it.
                if (_copyStream == IntPtr.Zero)
                {
                    Check(cuStreamCreate(out _copyStream, StreamNonBlocking), nameof(cuStreamCreate));
                }

                IntPtr ready = TakeOffloadEvent();
                Check(cuEventRecord(ready, _stream), nameof(cuEventRecord));
                Check(cuStreamWaitEvent(_copyStream, ready, 0), nameof(cuStreamWaitEvent));
                _offloadEvents.Push(ready);                                // the wait above already took its state
                batch = new PrefetchBatch(TakeOffloadEvent());
            }

            Check(cuMemcpyHtoDAsync(block, (void*)host, (nuint)BlockBytes(s.Length), _copyStream), nameof(cuMemcpyHtoDAsync));
            lock (_pool)
            {
                _prefetched[s] = (block, capacity, batch);
            }

            batch.Pending++;
            PrefetchedStorages++;
        }

        if (batch is not null)
        {
            Check(cuEventRecord(batch.Done, _copyStream), nameof(cuEventRecord));
            _latestBatch = batch;
        }
    }

    public IDisposable? Stage(IReadOnlyList<Storage> storages)
    {
        if (OffloadedCount == 0 || storages.Count == 0)
        {
            return null;
        }

        MakeCurrent();
        using var use = UseStream();
        lock (_pool)
        {
            if (_captureFree is not null)
            {
                return null;
            }
        }

        List<CudaStorage>? staged = null;
        PrefetchBatch? waited = null;
        foreach (var storage in storages)
        {
            if (Mine(storage) is not { } s)
            {
                continue;
            }

            ulong block;
            int capacity;
            lock (_pool)
            {
                if (!s.Alive || s.Evicted || s.Pointer == 0 || s.Home != 0 || !_hostBlocks.ContainsKey(s.Pointer))
                {
                    continue;
                }

                if (_prefetched.Remove(s, out var prefetched))
                {
                    // Copied in the background: the work stream waits for the copy, the host does not.
                    if (!ReferenceEquals(waited, prefetched.Batch))
                    {
                        Check(cuStreamWaitEvent(_stream, prefetched.Batch.Done, 0), nameof(cuStreamWaitEvent));
                        waited = prefetched.Batch;
                    }

                    FinishPrefetchLocked(prefetched.Batch);
                    StageLocked(s, prefetched.Pointer, prefetched.Capacity);
                    (staged ??= []).Add(s);
                    continue;
                }
            }

            block = TryDeviceBlock(s.Length, 0, useCache: true, out capacity);
            if (block == 0)
            {
                continue;
            }

            lock (_pool)
            {
                if (!s.Alive || s.Evicted || s.Home != 0 || !_hostBlocks.TryGetValue(s.Pointer, out IntPtr host))
                {
                    ReturnDeviceBlockLocked(block, capacity);
                    continue;
                }

                Check(cuMemcpyHtoDAsync(block, (void*)host, (nuint)BlockBytes(s.Length), _stream), nameof(cuMemcpyHtoDAsync));
                StageLocked(s, block, capacity);
                (staged ??= []).Add(s);
            }
        }

        return staged is null ? null : new StageToken(this, staged);
    }

    // Kernels read the GPU copy from now on; Home keeps the system block.
    private void StageLocked(CudaStorage s, ulong block, int capacity)
    {
        (s.Home, s.HomeCapacity) = (s.Pointer, s.Capacity);
        (s.Pointer, s.Capacity) = (block, capacity);
        StagedStorages++;
    }

    private void Unstage(List<CudaStorage> staged)
    {
        MakeCurrent();
        using var use = UseStream();
        lock (_pool)
        {
            foreach (var s in staged)
            {
                if (s.Home == 0)
                {
                    continue;                                              // released while staged (Return gave both back)
                }

                // Queued kernels still read the copy: the pool hands it out again only in stream order.
                ReturnDeviceBlockLocked(s.Pointer, s.Capacity);
                (s.Pointer, s.Capacity) = (s.Home, s.HomeCapacity);
                s.Home = 0;
            }
        }
    }

    private sealed class StageToken(CudaBackend backend, List<CudaStorage> staged) : IDisposable
    {
        private int _disposed;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0)
            {
                backend.Unstage(staged);
            }
        }
    }

    // Moves a GPU storage to system memory (with the stream gate held); false when it cannot move now.
    private bool MoveOut(CudaStorage s)
    {
        lock (_pool)
        {
            if (!s.Alive || s.Evicted || s.Pointer == 0 || OnHostLocked(s) || !Movable)
            {
                return false;
            }
        }

        ulong host = TakeHostBlock(s.Length);
        lock (_pool)
        {
            if (!s.Alive || s.Evicted || s.Pointer == 0 || OnHostLocked(s) || !Movable)
            {
                PushHostBlockLocked(host, s.Length);
                return false;
            }

            Check(cuMemcpyDtoHAsync((void*)_hostBlocks[host], s.Pointer, (nuint)BlockBytes(s.Length), _stream), nameof(cuMemcpyDtoHAsync));
            var (old, oldCapacity) = (s.Pointer, s.Capacity);
            (s.Pointer, s.Capacity) = (host, s.Length);
            ReturnDeviceBlockLocked(old, oldCapacity);                     // reused only after the copy, in stream order
            _memory.Offloaded(BlockBytes(s.Length));
            _cold.Remove(s);
            AddAwayLocked(s);
            MovedToHost++;
            return true;
        }
    }

    // Moves an offloaded storage into GPU memory if it fits with keepFree bytes to spare; the system block goes to the host
    // pool, or into `released` when given (the caller frees those).
    private bool MoveIn(CudaStorage s, long keepFree, bool useCache, List<(ulong Pointer, int Capacity)>? released)
    {
        lock (_pool)
        {
            if (!s.Alive || s.Evicted || s.Home != 0 || !_hostBlocks.ContainsKey(s.Pointer) || !Movable)
            {
                return false;
            }
        }

        ulong block = TryDeviceBlock(s.Length, keepFree, useCache, out int capacity);
        if (block == 0)
        {
            return false;
        }

        lock (_pool)
        {
            if (!s.Alive || s.Evicted || s.Home != 0 || !_hostBlocks.TryGetValue(s.Pointer, out IntPtr host) || !Movable)
            {
                ReturnDeviceBlockLocked(block, capacity);
                return false;
            }

            Check(cuMemcpyHtoDAsync(block, (void*)host, (nuint)BlockBytes(s.Length), _stream), nameof(cuMemcpyHtoDAsync));
            var (old, oldCapacity) = (s.Pointer, s.Capacity);
            (s.Pointer, s.Capacity) = (block, capacity);
            s.KeepOnHost = false;
            _memory.Offloaded(-BlockBytes(oldCapacity));
            if (released is null)
            {
                PushHostBlockLocked(old, oldCapacity);
            }
            else
            {
                released.Add((old, oldCapacity));
            }

            RemoveAwayLocked(s);
            if (s.OffloadPriority > OffloadPriority.Hot)
            {
                _cold.Add(s);
            }

            MovedToDevice++;
            return true;
        }
    }

    // A GPU block for `length` floats: from the cache (when allowed), else new memory if the GPU keeps the reserve plus
    // keepFree free and the memory limit allows it; 0 when there is no room. Never falls back to system memory.
    private ulong TryDeviceBlock(int length, long keepFree, bool useCache, out int capacity)
    {
        capacity = length;
        long bytes = BlockBytes(length);
        var usage = _memory.Usage;
        if (usage.Limit is { } max && usage.InUse + bytes + keepFree > max)
        {
            return 0;
        }

        if (useCache)
        {
            lock (_pool)
            {
                ulong cached = TakeCached(length, out capacity);
                if (cached != 0)
                {
                    _memory.Reused(BlockBytes(capacity));
                    return cached;
                }
            }
        }

        capacity = length;
        if (cuMemGetInfo(out nuint free, out _) != 0 || (long)free - bytes < MemoryReserve + keepFree)
        {
            return 0;
        }

        if (cuMemAlloc(out ulong pointer, (nuint)bytes) != 0)
        {
            return 0;
        }

        _memory.Allocated(bytes);
        return pointer;
    }

    // A pinned system block for `length` floats (from the host pool, else new); counted by the caller.
    private ulong TakeHostBlock(int length)
    {
        lock (_pool)
        {
            if (_hostPool.TryGetValue(length, out var bucket) && bucket.Count > 0)
            {
                return bucket.Pop();
            }
        }

        Check(cuMemHostAlloc(out IntPtr host, (nuint)BlockBytes(length), HostAllocPortable | HostAllocDeviceMap), nameof(cuMemHostAlloc));
        Check(cuMemHostGetDevicePointer(out ulong pointer, host, 0), nameof(cuMemHostGetDevicePointer));
        lock (_pool)
        {
            _hostBlocks[pointer] = host;
        }

        return pointer;
    }

    private void PushHostBlockLocked(ulong pointer, int capacity)
    {
        if (!_hostPool.TryGetValue(capacity, out var bucket))
        {
            _hostPool[capacity] = bucket = new Stack<ulong>();
        }

        bucket.Push(pointer);
    }

    // A GPU block back to the cache (or, while this thread records a graph, to the graph's blocks). Under the pool lock.
    private void ReturnDeviceBlockLocked(ulong pointer, int capacity)
    {
        var target = _captureFree is not null && Environment.CurrentManagedThreadId == _captureThread ? _captureFree : _pool;
        if (!target.TryGetValue(capacity, out var bucket))
        {
            target[capacity] = bucket = new Stack<ulong>();
        }

        bucket.Push(pointer);
        if (ReferenceEquals(target, _pool))
        {
            _poolSizes.Add(capacity);
        }

        _memory.Returned(BlockBytes(capacity));
    }

    // Offload bookkeeping when a storage is released for good (from Return, under the pool lock; possibly on the finalizer
    // thread, so no driver calls): a staged storage gives back its GPU copy and points at its system block again.
    private void ForgetLocked(CudaStorage s)
    {
        _cold.Remove(s);
        if (_awayCount > 0 && _away.Contains(s))
        {
            RemoveAwayLocked(s);
            if (s.Home != 0)
            {
                ReturnDeviceBlockLocked(s.Pointer, s.Capacity);
                (s.Pointer, s.Capacity) = (s.Home, s.HomeCapacity);
                s.Home = 0;
            }
        }

        if (_prefetched.Count > 0 && _prefetched.Remove(s, out var prefetched))
        {
            _stalePrefetches.Add(prefetched);                              // its block returns once the copy has run
        }
    }

    // Prefetched copies not staged: all of them, or those of batches before the latest (with the stream gate held).
    // Their blocks return to the pool after the work stream waited for the copies.
    private void DropPrefetches(bool all)
    {
        lock (_pool)
        {
            if (_prefetched.Count > 0)
            {
                foreach (var (s, prefetched) in _prefetched.ToList())
                {
                    if (all || !ReferenceEquals(prefetched.Batch, _latestBatch))
                    {
                        _prefetched.Remove(s);
                        _stalePrefetches.Add(prefetched);
                    }
                }
            }

            PrefetchBatch? waited = null;
            foreach (var (pointer, capacity, batch) in _stalePrefetches)
            {
                if (!ReferenceEquals(waited, batch))
                {
                    Check(cuStreamWaitEvent(_stream, batch.Done, 0), nameof(cuStreamWaitEvent));
                    waited = batch;
                }

                FinishPrefetchLocked(batch);
                ReturnDeviceBlockLocked(pointer, capacity);
            }

            _stalePrefetches.Clear();
        }
    }

    private void FinishPrefetchLocked(PrefetchBatch batch)
    {
        if (--batch.Pending == 0)
        {
            _offloadEvents.Push(batch.Done);                               // a later record replaces its state
        }
    }

    private IntPtr TakeOffloadEvent()
    {
        lock (_pool)
        {
            if (_offloadEvents.Count > 0)
            {
                return _offloadEvents.Pop();
            }
        }

        return CreateEvent();
    }

    // Recorded graphs hold raw addresses: counted so nothing moves while one exists.
    private void GraphCreated() => Interlocked.Increment(ref _liveGraphs);

    private void GraphDestroyed() => Interlocked.Decrement(ref _liveGraphs);
}
