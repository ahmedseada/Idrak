// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Numerics;
using Microsoft.Win32.SafeHandles;

namespace Idrak.Data.Pdf;

/// <summary>
/// Where a document's bytes are read from, at any offset and from any thread: a file through <see cref="RandomAccess"/>
/// (no position shared), bytes in memory (read in place, never copied), or any other seekable stream (reads under one
/// lock). The stream stays the caller's: nothing here disposes it.
/// </summary>
internal sealed class PdfSource
{
    private readonly SafeFileHandle? _handle;
    private readonly Stream? _stream;
    private readonly Lock _gate = new();

    private PdfSource(long length, SafeFileHandle? handle, ReadOnlyMemory<byte> memory, Stream? stream) =>
        (Length, _handle, Memory, _stream) = (length, handle, memory, stream);

    /// <summary>The file's length in bytes.</summary>
    public long Length { get; }

    /// <summary>The whole file when it is in memory (read in place), else empty.</summary>
    public ReadOnlyMemory<byte> Memory { get; }

    /// <summary>Whether the file is in memory, read in place.</summary>
    public bool InMemory => _handle is null && _stream is null;

    /// <summary>The source for <paramref name="stream"/>, read from its position 0.</summary>
    public static PdfSource For(Stream stream)
    {
        if (stream is FileStream file && file.CanSeek)
        {
            return new PdfSource(file.Length, file.SafeFileHandle, ReadOnlyMemory<byte>.Empty, null);
        }

        if (stream is MemoryStream memory && memory.TryGetBuffer(out var segment))
        {
            return new PdfSource(segment.Count, null, segment, null);
        }

        return new PdfSource(stream.Length, null, ReadOnlyMemory<byte>.Empty, stream);
    }

    /// <summary>Reads into <paramref name="destination"/> from <paramref name="offset"/>; fewer bytes only at the file's end.</summary>
    public int Read(long offset, Span<byte> destination)
    {
        if (offset >= Length || destination.IsEmpty)
        {
            return 0;
        }

        if (_handle is not null)
        {
            int total = 0;
            while (total < destination.Length && offset + total < Length)
            {
                int read = RandomAccess.Read(_handle, destination[total..], offset + total);
                if (read <= 0)
                {
                    break;
                }

                total += read;
            }

            return total;
        }

        if (_stream is not null)
        {
            lock (_gate)
            {
                _stream.Position = offset;
                return _stream.ReadAtLeast(destination, destination.Length, throwOnEndOfStream: false);
            }
        }

        var part = Memory.Span[(int)offset..];
        int count = Math.Min(part.Length, destination.Length);
        part[..count].CopyTo(destination);
        return count;
    }
}

/// <summary>
/// Some of a file's bytes: a slice of the file in memory, or a buffer rented from the pool and filled from the file,
/// returned on <see cref="Dispose"/>. Disposed once, by the one who read it (a <c>using</c> local; never copied).
/// </summary>
internal sealed class PdfWindow : IDisposable
{
    private byte[]? _rented;

    private PdfWindow(ReadOnlyMemory<byte> memory, byte[]? rented) => (Memory, _rented) = (memory, rented);

    public ReadOnlyMemory<byte> Memory { get; }

    public ReadOnlySpan<byte> Span => Memory.Span;

    /// <summary>Up to <paramref name="length"/> bytes from <paramref name="offset"/> (fewer at the file's end).</summary>
    public static PdfWindow Read(PdfSource source, long offset, long length)
    {
        long available = Math.Max(0, Math.Min(Math.Min(length, source.Length - offset), Array.MaxLength));
        if (available == 0)
        {
            return new PdfWindow(ReadOnlyMemory<byte>.Empty, null);
        }

        if (source.InMemory)
        {
            return new PdfWindow(source.Memory.Slice((int)Math.Min(offset, source.Length), (int)available), null);
        }

        byte[] rented = ArrayPool<byte>.Shared.Rent((int)available);
        try
        {
            int read = source.Read(offset, rented.AsSpan(0, (int)available));
            return new PdfWindow(rented.AsMemory(0, read), rented);
        }
        catch
        {
            ArrayPool<byte>.Shared.Return(rented);
            throw;
        }
    }

    /// <summary>The first <paramref name="length"/> bytes of <paramref name="rented"/>, a buffer rented from the pool that the window now owns (a stream's data decrypted).</summary>
    public static PdfWindow Own(byte[] rented, int length) => new(rented.AsMemory(0, length), rented);

    public void Dispose()
    {
        if (_rented is { } rented)
        {
            _rented = null;
            ArrayPool<byte>.Shared.Return(rented);
        }
    }
}

/// <summary>
/// A growing buffer for a decoded stream, never past a limit: writing past it throws the <see cref="PdfLimit"/> error the
/// owner words (a stream that decodes to more than <see cref="DocumentOptions.MaxStreamBytes"/>). It grows by doubling,
/// rented from the pool and returned on <see cref="Dispose"/>; its capacity never passes the limit, and the last step to
/// the limit, when the pool would round it up, is made at its size (so a stream bomb costs the limit, not twice it).
/// </summary>
internal sealed class PdfBuffer : IDisposable
{
    private const int ExactAbove = 1 << 20;

    private readonly long _limit;
    private readonly PdfLimitMessage _overLimit;
    private byte[] _array;
    private bool _pooled = true;
    private int _length;

    /// <param name="capacity">The first size to rent (grows by doubling).</param>
    /// <param name="limit">The most bytes it may hold.</param>
    /// <param name="overLimit">The message thrown when the limit is passed (a text, or one made only then).</param>
    public PdfBuffer(int capacity, long limit, PdfLimitMessage overLimit)
    {
        _limit = Math.Min(limit, Array.MaxLength);
        _overLimit = overLimit;
        _array = ArrayPool<byte>.Shared.Rent((int)Math.Min(Math.Max(capacity, 16), _limit));
    }

    public int Length => _length;

    /// <summary>Whether it holds as many bytes as the limit allows: one more passes it.</summary>
    public bool Full => _length >= _limit;

    public ReadOnlySpan<byte> Span => _array.AsSpan(0, _length);

    public ReadOnlyMemory<byte> Memory => _array.AsMemory(0, _length);

    /// <summary>The bytes written so far, to change in place (a predictor's rows).</summary>
    public Span<byte> Written => _array.AsSpan(0, _length);

    /// <summary>The error for passing the limit (for a writer that finds more to write when <see cref="Full"/>).</summary>
    public InvalidDataException OverLimit() => PdfLimit.Error(_overLimit.ToString());

    /// <summary>
    /// Room to write into: at least one byte and, when the limit allows, <paramref name="sizeHint"/>; never past the limit.
    /// </summary>
    /// <exception cref="InvalidDataException">It is <see cref="Full"/> (a <see cref="PdfLimit"/> error): the caller has more to write.</exception>
    public Span<byte> GetSpan(int sizeHint = 1)
    {
        long room = _limit - _length;
        if (room <= 0)
        {
            throw OverLimit();
        }

        int wanted = (int)Math.Min(Math.Max(sizeHint, 1), room);
        if (_array.Length - _length < wanted)
        {
            Grow(wanted);
        }

        return _array.AsSpan(_length, (int)Math.Min(_array.Length - _length, room));
    }

    /// <summary>Exactly <paramref name="count"/> bytes to write into, or the limit's error when they do not fit under it.</summary>
    public Span<byte> Reserve(int count)
    {
        if (_length + (long)count > _limit)
        {
            throw OverLimit();
        }

        if (_array.Length - _length < count)
        {
            Grow(count);
        }

        return _array.AsSpan(_length, count);
    }

    public void Advance(int count)
    {
        if (_length + (long)count > _limit)
        {
            throw OverLimit();
        }

        _length += count;
    }

    public void Write(ReadOnlySpan<byte> bytes)
    {
        bytes.CopyTo(Reserve(bytes.Length));
        _length += bytes.Length;
    }

    /// <summary>Keeps the first <paramref name="length"/> bytes (a predictor's rows are shorter than their input).</summary>
    public void Truncate(int length) => _length = Math.Min(_length, Math.Max(0, length));

    public void Clear() => _length = 0;

    public void Dispose()
    {
        byte[] array = _array;
        _array = [];
        _length = 0;
        if (_pooled && array.Length > 0)
        {
            ArrayPool<byte>.Shared.Return(array);
        }
    }

    private void Grow(int wanted)
    {
        long size = Math.Min(Math.Max((long)_array.Length * 2, (long)_length + wanted), _limit);
        bool exact = size == _limit && size > ExactAbove && !BitOperations.IsPow2(size);
        byte[] larger = exact ? GC.AllocateUninitializedArray<byte>((int)size) : ArrayPool<byte>.Shared.Rent((int)size);
        _array.AsSpan(0, _length).CopyTo(larger);
        if (_pooled)
        {
            ArrayPool<byte>.Shared.Return(_array);
        }

        (_array, _pooled) = (larger, !exact);
    }
}

/// <summary>
/// What a <see cref="PdfBuffer"/> says when its limit is passed: a text given whole, or a message a file makes only when
/// it is thrown (a page's content stream is decoded on every read of the page, and its limit is almost never passed).
/// </summary>
internal readonly struct PdfLimitMessage
{
    private readonly string? _text;
    private readonly PdfFile? _file;
    private readonly int _number;
    private readonly bool _page;

    private PdfLimitMessage(string? text, PdfFile? file, int number, bool page) => (_text, _file, _number, _page) = (text, file, number, page);

    public static implicit operator PdfLimitMessage(string text) => new(text, null, 0, false);

    /// <summary>Object <paramref name="number"/> of <paramref name="file"/>, a stream, decodes to more than the file's limit.</summary>
    public static PdfLimitMessage Stream(PdfFile file, int number) => new(null, file, number, false);

    /// <summary>Page <paramref name="page"/>'s (from 1) content streams of <paramref name="file"/> together pass the file's limit.</summary>
    public static PdfLimitMessage JoinedContent(PdfFile file, int page) => new(null, file, page, true);

    public override string ToString() =>
        _text ?? (_page
            ? $"{_file!.Name}: page {_number}: its content streams together pass the limit {nameof(DocumentOptions.MaxStreamBytes)} (DocumentOptions.MaxStreamBytes, or {DocumentOptions.MaxStreamBytesVariable})."
            : _file!.StreamOverLimit(_number));
}

/// <summary>
/// A stream's data after its filters: in the file's own memory when it had none, else in a pooled buffer returned on
/// <see cref="Dispose"/>. <see cref="ImageEncoding"/> names the image encoding it stopped before (its data handed over
/// encoded, for the image codecs); <see cref="Damaged"/> says a filter met damaged data and kept what it had decoded.
/// </summary>
internal sealed class PdfDecoded : IDisposable
{
    private PdfBuffer? _owned;
    private PdfWindow? _window;

    public PdfDecoded(ReadOnlyMemory<byte> memory, PdfBuffer? owned, PdfWindow? window, string? imageEncoding, bool damaged)
    {
        (Memory, _owned, _window, ImageEncoding, Damaged) = (memory, owned, window, imageEncoding, damaged);
    }

    public ReadOnlyMemory<byte> Memory { get; }

    public ReadOnlySpan<byte> Span => Memory.Span;

    public int Length => Memory.Length;

    public string? ImageEncoding { get; }

    public bool Damaged { get; }

    public void Dispose()
    {
        _owned?.Dispose();
        _owned = null;
        _window?.Dispose();
        _window = null;
    }
}
