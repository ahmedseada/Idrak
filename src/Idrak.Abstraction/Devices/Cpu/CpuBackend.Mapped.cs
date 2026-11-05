// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;

namespace Idrak.Abstraction.Devices.Cpu;

// A storage over a file's mapping (plan 37, W3): a weight's stored bytes read where the file's pages lie, so the CPU's own
// memory holds no copy of them (the system's page cache holds them once, shared with every reader of the file). Not
// pooled and not counted in the CPU's memory use; released, it closes its mapping. The stored-weight kernels read it in
// place (StoredRows); a download or a copy out of it is counted (MappedWeights), and a write into it makes the pages it
// touches private (the mapping is copy-on-write), counted too.
internal sealed class MappedStorage(CpuBackend backend, FileMapping mapping, int words) : Storage(backend, words)
{
    public FileMapping Mapping { get; } = mapping;
}

internal sealed partial class CpuBackend : IMappedStorageBackend
{
    public Storage MapStorage(FileMapping mapping, ElementType type, long count)
    {
        ArgumentNullException.ThrowIfNull(mapping);
        try
        {
            ArgumentNullException.ThrowIfNull(type);
            ThrowIfDisposed();
            long bytes = type.ByteCount(count), words = type.StorageWords(count);
            if (bytes != mapping.Length || words > int.MaxValue)
            {
                throw new ArgumentException($"{count} {type.Name} elements take {bytes} bytes; the mapping {mapping} holds {mapping.Length}.", nameof(mapping));
            }

            var storage = new MappedStorage(this, mapping, (int)words);
            storage.SetType(type, count);
            MappedWeights.CountMapped(bytes);
            return storage;
        }
        catch
        {
            mapping.Dispose();
            throw;
        }
    }

    // Words [offset, offset + destination.Length) of a mapped storage copied out of its mapping (the bytes past the
    // mapping's end, in its last word, read as zero), counted as a copy for `reason`.
    private static void CopyOutOfMapping(MappedStorage source, int offset, Span<float> destination, string reason)
    {
        var target = MemoryMarshal.AsBytes(destination);
        long start = (long)offset * sizeof(float);
        int available = (int)Math.Clamp(source.Mapping.Length - start, 0, target.Length);
        source.Mapping.Bytes(start, available).CopyTo(target);
        target[available..].Clear();
        MappedWeights.CountCopied(available, reason);
    }

    // Words written into a mapped storage from `offset`: the pages they touch become the process's own (copy-on-write),
    // counted as a copy; the file never changes.
    private static void WriteIntoMapping(ReadOnlySpan<float> source, MappedStorage destination, int offset)
    {
        var bytes = MemoryMarshal.AsBytes(source);
        long start = (long)offset * sizeof(float);
        int fits = (int)Math.Clamp(destination.Mapping.Length - start, 0, bytes.Length);
        bytes[..fits].CopyTo(destination.Mapping.WritableBytes(start, fits));
        MappedWeights.CountCopied(fits, "written in place (its pages made private)");
    }

    // Closes a mapped storage's mapping when it is released for good (it holds none of the CPU's memory).
    private static void ReleaseMapping(MappedStorage storage)
    {
        long bytes = storage.Mapping.Length;
        storage.Mapping.Dispose();
        MappedWeights.CountReleased(bytes);
    }
}
