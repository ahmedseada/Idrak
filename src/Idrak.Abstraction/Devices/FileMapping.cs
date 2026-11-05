// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Globalization;
using System.IO.MemoryMappedFiles;

namespace Idrak.Abstraction.Devices;

/// <summary>
/// A range of a file mapped into the process copy-on-write (plan 37, W3): the operating system's page cache holds the
/// file's pages once and the process reads them in place, so weights read from a checkpoint this way are not copied a
/// second time into memory. A write to the range (nothing in the library writes a mapped weight) makes a private copy of
/// the pages it touches and never reaches the file. Each file is mapped once per process (by its full path, size and time
/// of last change) and each range is a view of its own; the file is closed when its last range is disposed.
/// </summary>
public sealed unsafe class FileMapping : IDisposable
{
    // The files mapped now, by path, size and time of change, each with the ranges that read it.
    private static readonly Dictionary<string, SharedFile> s_files = new(StringComparer.Ordinal);

    private readonly SharedFile _file;
    private readonly MemoryMappedViewAccessor _view;
    private readonly byte* _pointer;
    private int _disposed;

    private FileMapping(SharedFile file, string path, long offset, long length)
    {
        (_file, Path, Offset, Length) = (file, path, offset, length);
        _view = file.Map.CreateViewAccessor(offset, length, MemoryMappedFileAccess.CopyOnWrite);
        try
        {
            byte* start = null;
            _view.SafeMemoryMappedViewHandle.AcquirePointer(ref start);
            _pointer = start + _view.PointerOffset;                         // the view starts on the system's granularity
        }
        catch
        {
            _view.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Maps <paramref name="length"/> bytes of <paramref name="path"/> from <paramref name="offset"/> copy-on-write.
    /// </summary>
    /// <exception cref="FileNotFoundException">The file does not exist.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The range is empty or reaches past the file's end.</exception>
    /// <exception cref="IOException">The system refused to map it (the message says why).</exception>
    public static FileMapping Map(string path, long offset, long length)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentOutOfRangeException.ThrowIfNegative(offset);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        string full = System.IO.Path.GetFullPath(path);
        var info = new FileInfo(full);
        if (!info.Exists)
        {
            throw new FileNotFoundException($"{full} does not exist.", full);
        }

        if (offset > info.Length - length)
        {
            throw new ArgumentOutOfRangeException(nameof(length), length, $"{length} bytes from {offset} reach past the end of {full} ({info.Length} bytes).");
        }

        string key = string.Create(CultureInfo.InvariantCulture, $"{full}|{info.Length}|{info.LastWriteTimeUtc.Ticks}");
        SharedFile file;
        lock (s_files)
        {
            if (s_files.TryGetValue(key, out var open))
            {
                file = open;
            }
            else
            {
                file = SharedFile.Open(full, key);
                s_files[key] = file;
            }

            file.References++;
        }

        try
        {
            return new FileMapping(file, full, offset, length);
        }
        catch
        {
            file.Release();
            throw;
        }
    }

    /// <summary>The files mapped now (each closed when its last range is disposed).</summary>
    public static int MappedFiles
    {
        get
        {
            lock (s_files)
            {
                return s_files.Count;
            }
        }
    }

    /// <summary>The file's full path.</summary>
    public string Path { get; }

    /// <summary>Where the range starts in the file, in bytes.</summary>
    public long Offset { get; }

    /// <summary>The range's bytes.</summary>
    public long Length { get; }

    /// <summary>
    /// <paramref name="count"/> bytes of the range from <paramref name="start"/>, read in place: valid until the mapping is
    /// disposed.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">The bytes are not within the range.</exception>
    public ReadOnlySpan<byte> Bytes(long start, int count) => new(At(start, count), count);

    // The same bytes, writable: a write makes the pages it touches private (copy-on-write), never reaching the file.
    internal Span<byte> WritableBytes(long start, int count) => new(At(start, count), count);

    private byte* At(long start, int count)
    {
        if (start < 0 || count < 0 || start > Length - count)
        {
            throw new ArgumentOutOfRangeException(nameof(count), $"{count} bytes from {start} are not within a mapped range of {Length}.");
        }

        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        return _pointer + start;
    }

    /// <summary>Unmaps the range; the file is closed when its last range is.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _view.SafeMemoryMappedViewHandle.ReleasePointer();
        _view.Dispose();
        _file.Release();
    }

    /// <inheritdoc />
    public override string ToString() => string.Create(CultureInfo.InvariantCulture, $"{Path} [{Offset}, +{Length})");

    // A file mapped once (copy-on-write, opened for reading and shared with other readers), with its ranges counted.
    private sealed class SharedFile(MemoryMappedFile map, string key)
    {
        public MemoryMappedFile Map => map;

        public int References;

        public static SharedFile Open(string path, string key)
        {
            // Read access is all a copy-on-write mapping needs; others may read the file (the checkpoint's readers) and
            // delete or replace it (the mapping keeps its pages).
            var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read | FileShare.Delete, 1, FileOptions.RandomAccess);
            try
            {
                return new SharedFile(MemoryMappedFile.CreateFromFile(stream, null, 0, MemoryMappedFileAccess.CopyOnWrite, HandleInheritability.None, leaveOpen: false), key);
            }
            catch
            {
                stream.Dispose();
                throw;
            }
        }

        public void Release()
        {
            lock (s_files)
            {
                if (--References > 0)
                {
                    return;
                }

                s_files.Remove(key);
            }

            map.Dispose();
        }
    }
}

/// <summary>
/// A device whose kernels read system memory in place (the CPU), so it can hold a storage over a file's mapping (plan 37,
/// W3): the storage's values are the mapped bytes, read where they lie, never copied into the device's memory nor counted
/// in its use (<see cref="Backend.GetMemoryUsage"/>). The kernels that read stored weights (<see cref="Backend.StoredMatMul"/>,
/// <see cref="Backend.ExpandStored"/>) read it in place; a copy out of it (a download, a move to another device, an
/// expansion to float32) is counted in <see cref="MappedWeights"/>.
/// </summary>
public interface IMappedStorageBackend
{
    /// <summary>
    /// A storage of <paramref name="count"/> elements of <paramref name="type"/> whose bytes are those of
    /// <paramref name="mapping"/> (exactly <see cref="ElementType.ByteCount"/> of them), read in place. The storage takes the
    /// mapping: it is disposed when the storage is released for good, or at once when this throws.
    /// </summary>
    /// <exception cref="ArgumentException">The mapping's bytes are not the elements' bytes.</exception>
    Storage MapStorage(FileMapping mapping, ElementType type, long count);
}

/// <summary>
/// What the file mappings of weights did (plan 37, W3), counted since the process started: the weights read in place from
/// a checkpoint's file mapping (each matrix once, when its storage is made) and those released, and every copy taken of
/// such a weight or instead of it: at load, where the file does not hold the weight as the product reads it (rows put back
/// in another order, a format the run does not compute in or keep), and after, where a layout must be materialized (a
/// download, a move to another device, an expansion to float32), each with why. Read two snapshots (<see cref="Read"/>)
/// around a run and take the difference (<see cref="MappedWeightCounts.Since"/>).
/// </summary>
public static class MappedWeights
{
    private static long s_mapped, s_mappedBytes, s_released, s_releasedBytes, s_copies, s_copiedBytes;
    private static readonly ConcurrentDictionary<string, long> s_copiedFor = new(StringComparer.Ordinal);

    /// <summary>The counts so far.</summary>
    public static MappedWeightCounts Read() => new(Interlocked.Read(ref s_mapped), Interlocked.Read(ref s_mappedBytes), Interlocked.Read(ref s_released),
        Interlocked.Read(ref s_releasedBytes), Interlocked.Read(ref s_copies), Interlocked.Read(ref s_copiedBytes), new Dictionary<string, long>(s_copiedFor, StringComparer.Ordinal));

    /// <summary>A weight of <paramref name="bytes"/> is now read in place from a file's mapping.</summary>
    public static void CountMapped(long bytes)
    {
        Interlocked.Increment(ref s_mapped);
        Interlocked.Add(ref s_mappedBytes, bytes);
    }

    /// <summary>A weight of <paramref name="bytes"/> read in place was released (its mapping closed).</summary>
    public static void CountReleased(long bytes)
    {
        Interlocked.Increment(ref s_released);
        Interlocked.Add(ref s_releasedBytes, bytes);
    }

    /// <summary><paramref name="bytes"/> of a weight were copied into memory out of a file's mapping or instead of it, for <paramref name="reason"/>.</summary>
    public static void CountCopied(long bytes, string reason)
    {
        ArgumentNullException.ThrowIfNull(reason);
        Interlocked.Increment(ref s_copies);
        Interlocked.Add(ref s_copiedBytes, bytes);
        s_copiedFor.AddOrUpdate(reason, bytes, (_, n) => n + bytes);
    }
}

/// <summary>A snapshot of <see cref="MappedWeights"/>, or the difference of two (<see cref="Since"/>).</summary>
/// <param name="Mapped">Weights read in place from a file's mapping.</param>
/// <param name="MappedBytes">Their bytes: what the page cache holds for them, not the process's own memory.</param>
/// <param name="Released">Weights read in place that were released.</param>
/// <param name="ReleasedBytes">Their bytes.</param>
/// <param name="Copies">Copies taken of a weight out of a mapping or instead of it.</param>
/// <param name="CopiedBytes">Their bytes, in the process's memory.</param>
/// <param name="CopiedFor">The copied bytes by why.</param>
public sealed record MappedWeightCounts(long Mapped, long MappedBytes, long Released, long ReleasedBytes, long Copies, long CopiedBytes,
    IReadOnlyDictionary<string, long> CopiedFor)
{
    /// <summary>The bytes read in place now: mapped and not released.</summary>
    public long LiveBytes => MappedBytes - ReleasedBytes;

    /// <summary>What was counted after <paramref name="before"/> (an earlier snapshot).</summary>
    public MappedWeightCounts Since(MappedWeightCounts before)
    {
        ArgumentNullException.ThrowIfNull(before);
        var copied = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (var (reason, bytes) in CopiedFor)
        {
            long delta = bytes - before.CopiedFor.GetValueOrDefault(reason);
            if (delta > 0)
            {
                copied[reason] = delta;
            }
        }

        return new MappedWeightCounts(Mapped - before.Mapped, MappedBytes - before.MappedBytes, Released - before.Released, ReleasedBytes - before.ReleasedBytes,
            Copies - before.Copies, CopiedBytes - before.CopiedBytes, copied);
    }

    /// <summary>"weights mapped: 14 read in place from the file (1.2 MB); 2 copies (0.1 MB: rows put back in order 0.1 MB)".</summary>
    public override string ToString()
    {
        string line = string.Create(CultureInfo.InvariantCulture, $"weights mapped: {Mapped} read in place from the file ({Megabytes(MappedBytes)}); {Copies} copies ({Megabytes(CopiedBytes)}");
        return CopiedFor.Count == 0 ? line + ")"
            : line + ": " + string.Join(", ", CopiedFor.OrderByDescending(p => p.Value).Select(p => $"{p.Key} {Megabytes(p.Value)}")) + ")";
    }

    private static string Megabytes(long bytes) => string.Create(CultureInfo.InvariantCulture, $"{bytes / 1e6:0.##} MB");
}
