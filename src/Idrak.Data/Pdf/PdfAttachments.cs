// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Data.Pdf;

/// <summary>
/// The files a PDF document carries (ISO 32000-2, 7.11.4 and 12.5.6.15): the catalog's <c>/Names /EmbeddedFiles</c> name
/// tree (kids and leaves, in its order), then each page's file-attachment annotations, a file listed once. Each is
/// given by its file specification: its name (<c>/UF</c>, else <c>/F</c>, else its key in the tree; the last part of a
/// path, so a name never leads out of a folder an app writes it to), its media type (the embedded stream's
/// <c>/Subtype</c>), its size (<c>/Params /Size</c>) and description (<c>/Desc</c>, else the annotation's
/// <c>/Contents</c>); its bytes are decoded (and decrypted) only when it is opened, within
/// <see cref="DocumentOptions.MaxStreamBytes"/>. A damaged entry, or a node of the tree met again, is passed over; a limit
/// or a cycle of references fails the listing.
/// </summary>
internal static class PdfAttachments
{
    /// <summary>The document's attachments, its <paramref name="pages"/> given as the page tree's leaves.</summary>
    public static IReadOnlyList<DocumentAttachment> Read(PdfFile file, IReadOnlyList<PdfValue> pages)
    {
        var found = new List<DocumentAttachment>();
        var listed = new HashSet<int>();   // the embedded file streams listed, by object number
        if (Damaged(() => file.Dictionary(file.Dictionary(file.Trailer["Root"u8])?["Names"u8] ?? PdfValue.Null)) is { } names)
        {
            Tree(file, names["EmbeddedFiles"u8], (key, specification) => Add(file, key, specification, null, found, listed));
        }

        foreach (var node in pages)
        {
            var annotations = Damaged(() => file.Resolve(file.Dictionary(node)?["Annots"u8] ?? PdfValue.Null).AsArray);
            foreach (var entry in annotations?.Items ?? [])
            {
                var annotation = Damaged(() => file.Dictionary(entry));
                if (annotation is not null && file.Resolve(annotation["Subtype"u8]).IsName("FileAttachment"u8))
                {
                    Add(file, null, annotation["FS"u8], annotation, found, listed);
                }
            }
        }

        return found;
    }

    // A name tree walked in order without recursion: each leaf's /Names pairs given to `leaf`; a node met again (a
    // cycle) is passed over, and nesting past MaxDepth fails as the other limits do.
    private static void Tree(PdfFile file, PdfValue root, Action<byte[]?, PdfValue> leaf)
    {
        var visited = new HashSet<int>();
        var stack = new Stack<(PdfValue Node, int Depth)>();
        stack.Push((root, 1));
        while (stack.Count > 0)
        {
            var (value, depth) = stack.Pop();
            if (value.IsReference && !visited.Add(value.AsReference.Number))
            {
                continue;
            }

            if (depth > file.MaxDepth)
            {
                throw PdfLimit.Error($"{file.Name}: the embedded files' name tree is nested deeper than {file.MaxDepth} levels, the limit {nameof(DocumentOptions.MaxDepth)} (DocumentOptions.MaxDepth).");
            }

            var node = Damaged(() => file.Dictionary(value));
            if (node is null)
            {
                continue;
            }

            if (Damaged(() => file.Resolve(node["Names"u8]).AsArray) is { } pairs)
            {
                for (int i = 0; i + 1 < pairs.Count; i += 2)
                {
                    var key = Damaged(() => file.Resolve(pairs[i]).AsString);
                    leaf(key?.Bytes, pairs[i + 1]);
                }
            }

            if (Damaged(() => file.Resolve(node["Kids"u8]).AsArray) is { } kids)
            {
                for (int i = kids.Count - 1; i >= 0; i--)
                {
                    stack.Push((kids[i], depth + 1));
                }
            }
        }
    }

    // One file specification's embedded file, unless it has none or it is listed already.
    private static void Add(PdfFile file, byte[]? key, PdfValue value, PdfDictionary? annotation, List<DocumentAttachment> found, HashSet<int> listed)
    {
        if (Damaged(() => file.Dictionary(value)) is not { } specification || Damaged(() => file.Dictionary(specification["EF"u8])) is not { } embedded)
        {
            return;   // a bare file name (a file outside the document, never fetched) or a damaged entry
        }

        // The file's own stream: /UF, else /F (or a platform's entry, from before PDF 1.7).
        var entry = Entry(embedded, "UF"u8) ?? Entry(embedded, "F"u8) ?? Entry(embedded, "Unix"u8) ?? Entry(embedded, "Mac"u8) ?? Entry(embedded, "DOS"u8);
        var stream = entry is { } reference ? Damaged(() => file.Resolve(reference).AsStream) : null;
        if (stream is null || !listed.Add(stream.Number))
        {
            return;
        }

        string name = FileName(Text(file, specification, "UF"u8) ?? Text(file, specification, "F"u8) ?? Text(file, specification, "Unix"u8) ?? Text(file, specification, "DOS"u8))
                      ?? FileName(key is null ? null : PdfText.Decode(key))
                      ?? string.Create(CultureInfo.InvariantCulture, $"attachment-{found.Count + 1}");
        long size = Damaged(() => file.Dictionary(stream.Dictionary["Params"u8])) is { } parameters && Damaged(() => file.Resolve(parameters["Size"u8])).TryInteger(out long bytes) && bytes >= 0 ? bytes : -1;
        string where = string.Create(CultureInfo.InvariantCulture, $"{file.Name}: the embedded file {name} (object {stream.Number})");
        found.Add(new DocumentAttachment(name, () => Open(file, stream, where))
        {
            MediaType = Damaged(() => file.Resolve(stream.Dictionary["Subtype"u8]).AsName?.ToString()),
            Size = size,
            Description = Text(file, specification, "Desc"u8) ?? (annotation is null ? null : Text(file, annotation, "Contents"u8)),
        });
    }

    private static PdfValue? Entry(PdfDictionary dictionary, ReadOnlySpan<byte> key) => dictionary.TryGet(key, out var value) && !value.IsNull ? value : null;

    // The embedded file's bytes, decoded (and decrypted) now: a stream over them, the caller's to dispose.
    private static Stream Open(PdfFile file, PdfStream stream, string where)
    {
        try
        {
            return new PdfDecodedStream(file.Decode(stream));
        }
        catch (InvalidDataException e) when (!e.Message.StartsWith(file.Name, StringComparison.Ordinal))
        {
            throw PdfLimit.Is(e) ? PdfLimit.Error($"{where}: {e.Message}") : new InvalidDataException($"{where}: {e.Message}", e);
        }
        catch (Exception e) when (e is not (InvalidDataException or NotSupportedException or OperationCanceledException or OutOfMemoryException or ObjectDisposedException))
        {
            throw new InvalidDataException($"{where}: damaged ({e.GetType().Name}: {e.Message}).", e);
        }
    }

    // A text entry of a dictionary, or null (a damaged one too).
    private static string? Text(PdfFile file, PdfDictionary dictionary, ReadOnlySpan<byte> key)
    {
        if (!dictionary.TryGet(key, out var value))
        {
            return null;
        }

        var text = Damaged(() => file.Resolve(value).AsString);
        return text is null ? null : PdfText.Decode(text.Bytes);
    }

    // A path's last part, without characters that are not a file name's; null when nothing is left.
    private static string? FileName(string? path)
    {
        if (path is null)
        {
            return null;
        }

        int cut = path.AsSpan().LastIndexOfAny('/', '\\', ':');
        var last = path.AsSpan(cut + 1).Trim();
        if (last.IsEmpty || last is "." or "..")
        {
            return null;
        }

        char[] chars = last.ToArray();
        for (int i = 0; i < chars.Length; i++)
        {
            chars[i] = char.IsControl(chars[i]) ? '_' : chars[i];
        }

        return new string(chars);
    }

    // A read that a damaged object answers with nothing (a limit or a cycle still fails).
    private static T? Damaged<T>(Func<T?> read)
    {
        try
        {
            return read();
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            return default;
        }
    }
}

/// <summary>
/// A read-only, seekable stream over a decoded stream's bytes, which it owns: its pooled buffer is returned when the
/// stream is disposed (an attachment's bytes, opened without a second copy).
/// </summary>
internal sealed class PdfDecodedStream(PdfDecoded decoded) : Stream
{
    private PdfDecoded? _decoded = decoded;
    private long _position;

    public override bool CanRead => _decoded is not null;

    public override bool CanSeek => _decoded is not null;

    public override bool CanWrite => false;

    public override long Length => Data.Length;

    public override long Position
    {
        get => _position;
        set
        {
            ArgumentOutOfRangeException.ThrowIfNegative(value);
            _position = value;
        }
    }

    private ReadOnlyMemory<byte> Data => (_decoded ?? throw new ObjectDisposedException(nameof(PdfDecodedStream))).Memory;

    public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));

    public override int Read(Span<byte> buffer)
    {
        var data = Data.Span;
        int count = (int)Math.Clamp(data.Length - _position, 0, buffer.Length);
        data.Slice((int)_position, count).CopyTo(buffer);
        _position += count;
        return count;
    }

    public override int ReadByte()
    {
        var data = Data.Span;
        return _position < data.Length ? data[(int)_position++] : -1;
    }

    public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.FromResult(Read(buffer.Span));
    }

    public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
        ReadAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();

    public override void CopyTo(Stream destination, int bufferSize)
    {
        var data = Data.Span;
        if (_position < data.Length)
        {
            destination.Write(data[(int)_position..]);
            _position = data.Length;
        }
    }

    public override long Seek(long offset, SeekOrigin origin)
    {
        long target = origin switch
        {
            SeekOrigin.Begin => offset,
            SeekOrigin.Current => _position + offset,
            SeekOrigin.End => Data.Length + offset,
            _ => throw new ArgumentOutOfRangeException(nameof(origin)),
        };
        ArgumentOutOfRangeException.ThrowIfNegative(target, nameof(offset));
        _position = target;
        return target;
    }

    public override void Flush()
    {
    }

    public override void SetLength(long value) => throw new NotSupportedException("An embedded file's stream is read-only.");

    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException("An embedded file's stream is read-only.");

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _decoded?.Dispose();
            _decoded = null;
        }

        base.Dispose(disposing);
    }
}
