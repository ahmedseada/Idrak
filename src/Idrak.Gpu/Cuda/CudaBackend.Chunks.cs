// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using static Idrak.Gpu.Cuda.CudaDriver;

namespace Idrak.Gpu.Cuda;

// Long-lived blocks in shared chunks (Backend.LongLivedAllocations, ComputeResources.GpuChunkBytes). The driver rounds
// every allocation above half a page up to whole pages (measured: 1 MiB + 4 KiB takes 2 MiB, 10 MiB + 4 KiB 12 MiB),
// and a model's weights and its attention caches are hundreds of such blocks. Inside the scope, a block the driver would
// round (at least half a page, not a whole number of pages, at most an eighth of a chunk) is taken first-fit from a
// chunk's free ranges, else from a new chunk; the rounding is then paid once a chunk.
//
// - Freeing: a chunk's block goes back to the pool by its size like any other (Return); only when the pool's cache is
//   released (ReleaseCachedMemory) does it go back to its chunk's free ranges, merged with its neighbours, and a chunk
//   with nothing taken goes back to the driver.
// - Accounting: a new chunk counts as cached; a block taken from it moves its bytes to in use, as a cached block reused.
// - Never while a graph is recorded (the recording owns the blocks freed meanwhile), and never for a host block.
// - A full device: any block may take a chunk's free room (ChunkRoom) before the allocation fails.
internal sealed unsafe partial class CudaBackend
{
    private sealed class Chunk(ulong address, long bytes)
    {
        public readonly ulong Address = address;
        public readonly long Bytes = bytes;
        public readonly SortedDictionary<long, long> Free = new() { [0] = bytes };   // offset → bytes
        public long FreeBytes = bytes;
    }

    // Blocks start on this boundary inside a chunk (the driver's own allocations are at least 256-byte aligned).
    private const long ChunkAlignment = 512;

    // Under the pool lock: the chunks, and every block taken from one by its address (its chunk and its bytes).
    private readonly List<Chunk> _chunks = [];
    private readonly Dictionary<ulong, (Chunk Chunk, long Bytes)> _chunkBlocks = [];

    [ThreadStatic]
    private static int t_longLived;

    private long _page = -1;                                             // the driver's allocation page; 0 when it reports none

    /// <summary>Chunks taken from the driver and blocks placed in them (tests and diagnostics).</summary>
    internal long ChunksAllocated, ChunkedBlocks;

    /// <summary>The chunks held now (tests and diagnostics).</summary>
    internal int ChunkCount
    {
        get
        {
            lock (_pool)
            {
                return _chunks.Count;
            }
        }
    }

    public override IDisposable? LongLivedAllocations()
    {
        t_longLived++;
        return new LongLived();
    }

    private sealed class LongLived : IDisposable
    {
        private bool _done;

        public void Dispose()
        {
            if (!_done)
            {
                _done = true;
                t_longLived--;
            }
        }
    }

    // The driver's allocation page (cuMemGetAllocationGranularity), asked once; 0 when it does not report one (no chunks).
    private long Page
    {
        get
        {
            if (_page < 0)
            {
                var prop = new MemAllocationProp { Type = 1, LocationType = 1, LocationId = _deviceHandle };
                _page = cuMemGetAllocationGranularity(out nuint page, in prop, 0) == 0 ? (long)page : 0;
            }

            return _page;
        }
    }

    // The chunk size in use: the setting, else 32 pages; 0 when chunks are off.
    internal long ChunkSize => ComputeResources.GpuChunkBytes is { } set ? Math.Max(0, set) : 32 * Page;

    // Whether a new block of `bytes` goes into a chunk.
    private bool Chunked(long bytes)
    {
        if (t_longLived <= 0 || _captureFree is not null)
        {
            return false;
        }

        long page = Page, chunk = ChunkSize;
        return page > 0 && chunk > 0 && bytes >= page / 2 && bytes % page != 0 && bytes <= chunk / 8;
    }

    // A block of `bytes` from a chunk (a new one when none has room), with its capacity in floats; 0 when no chunk can
    // be made (the caller then allocates the block alone). With the work stream held, outside the pool lock.
    private ulong ChunkTake(long bytes, out int capacity)
    {
        long size = (bytes + ChunkAlignment - 1) / ChunkAlignment * ChunkAlignment;
        capacity = (int)(size / sizeof(float));
        lock (_pool)
        {
            foreach (var chunk in _chunks)
            {
                if (chunk.FreeBytes >= size && TakeFrom(chunk, size) is var found && found != 0)
                {
                    return found;
                }
            }
        }

        long chunkBytes = Math.Max(ChunkSize, (size + Page - 1) / Page * Page);
        if (cuMemGetInfo(out nuint free, out _) != 0 || (long)free - chunkBytes < MemoryReserve || cuMemAlloc(out ulong address, (nuint)chunkBytes) != 0)
        {
            return 0;
        }

        lock (_pool)
        {
            var made = new Chunk(address, chunkBytes);
            _chunks.Add(made);
            ChunksAllocated++;
            _memory.Allocated(chunkBytes);
            _memory.Returned(chunkBytes);                                  // free space in a chunk is cached memory
            return TakeFrom(made, size);
        }
    }

    // A block of `bytes` from the chunks' free room, for any block once the device has no room of its own (no chunk is
    // made for it); 0 when none holds it.
    private ulong ChunkRoom(long bytes, out int capacity)
    {
        long size = (bytes + ChunkAlignment - 1) / ChunkAlignment * ChunkAlignment;
        capacity = (int)Math.Min(int.MaxValue, size / sizeof(float));
        lock (_pool)
        {
            foreach (var chunk in _chunks)
            {
                if (chunk.FreeBytes >= size && TakeFrom(chunk, size) is var found && found != 0)
                {
                    return found;
                }
            }
        }

        return 0;
    }

    // The first free range of `chunk` that holds `size` bytes, taken from its start; 0 when none does. Under the pool lock.
    private ulong TakeFrom(Chunk chunk, long size)
    {
        foreach (var (offset, length) in chunk.Free)
        {
            if (length < size)
            {
                continue;
            }

            chunk.Free.Remove(offset);
            if (length > size)
            {
                chunk.Free[offset + size] = length - size;
            }

            chunk.FreeBytes -= size;
            ulong address = chunk.Address + (ulong)offset;
            _chunkBlocks[address] = (chunk, size);
            ChunkedBlocks++;
            return address;
        }

        return 0;
    }

    // Gives a cached block back to its chunk (merged with the free ranges beside it), and the chunk back to the driver
    // when nothing in it is taken; false when the block is not a chunk's. Under the pool lock, the work stream synchronized.
    private bool ChunkRelease(ulong pointer)
    {
        if (_chunkBlocks.Count == 0 || !_chunkBlocks.Remove(pointer, out var block))
        {
            return false;
        }

        var chunk = block.Chunk;
        long offset = (long)(pointer - chunk.Address), size = block.Bytes;
        foreach (var (start, length) in chunk.Free)
        {
            if (start + length == offset)
            {
                chunk.Free.Remove(start);
                (offset, size) = (start, length + size);                   // merged with the range before it
                break;
            }
        }

        if (chunk.Free.Remove(offset + size, out long after))
        {
            size += after;                                                 // and with the one after it
        }

        chunk.Free[offset] = size;
        chunk.FreeBytes += block.Bytes;
        if (chunk.FreeBytes == chunk.Bytes)
        {
            Check(cuMemFree(chunk.Address), nameof(cuMemFree));
            _chunks.Remove(chunk);
            _memory.Freed(chunk.Bytes);
        }

        return true;
    }
}
