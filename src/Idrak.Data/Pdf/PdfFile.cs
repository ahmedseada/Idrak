// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Globalization;

namespace Idrak.Data.Pdf;

/// <summary>
/// A PDF file's object layer, opened, not loaded: the header (found within the first KiB), the cross-reference (or its
/// repair), and objects read on demand from the source, each parsed once and kept in a cache bounded by
/// <see cref="DocumentOptions.CacheBytes"/>, object streams decoded once. References are resolved lazily; an object
/// needed to read itself, or a chain of references that comes back, is a cycle, reported. A cross-reference that points
/// at the wrong bytes is rebuilt by scanning, once, when an object is not where it says. Safe from several threads.
/// </summary>
internal sealed class PdfFile : IDisposable
{
    private const int FirstWindow = 4096;

    // The objects this thread is reading (to find an object that needs itself), across documents.
    [ThreadStatic]
    private static List<(PdfFile File, int Number)>? t_reading;

    private readonly PdfCache _cache;
    private readonly Lock _repairGate = new();
    private volatile PdfXref _xref;
    private PdfSecurity? _security;
    private bool _repairTried;
    private volatile bool _disposed;

    private PdfFile(string name, PdfSource source, DocumentOptions options)
    {
        Name = name;
        Source = source;
        MaxStreamBytes = options.MaxStreamBytes;   // each limit read once: the measured ones look at the free memory
        MaxObjects = options.MaxObjects;
        MaxDepth = options.MaxDepth;
        _cache = new PdfCache(options.CacheBytes);
        _xref = new PdfXref([], 0, PdfDictionary.Empty, [], 0, repaired: false);   // nothing listed while the cross-reference is read
    }

    /// <summary>The name messages give (the file's).</summary>
    public string Name { get; }

    public PdfSource Source { get; }

    public long MaxStreamBytes { get; }

    public long MaxObjects { get; }

    public int MaxDepth { get; }

    /// <summary>How many bytes come before <c>%PDF-</c>.</summary>
    public long HeaderOffset { get; private set; }

    /// <summary>The version the header states ("1.7").</summary>
    public string? HeaderVersion { get; private set; }

    public PdfXref Xref => _xref;

    public bool Repaired => _xref.Repaired;

    public PdfDictionary Trailer => _xref.Trailer;

    /// <summary>The security handler that decrypts the file's strings and streams (null when the file is not encrypted).</summary>
    public PdfSecurity? Security => _security;

    /// <summary>
    /// Decrypts every string and stream read from now on through <paramref name="security"/> (its key found at open),
    /// which is disposed with the file; what was read before (the encryption dictionary) is read again when asked.
    /// </summary>
    public void Unlock(PdfSecurity security)
    {
        _security = security;
        _cache.Clear();
    }

    /// <summary>The file <paramref name="stream"/> holds: its header and cross-reference read, nothing else.</summary>
    /// <exception cref="InvalidDataException">Not a PDF file, damaged beyond repair, or a limit passed (named).</exception>
    public static PdfFile Open(Stream stream, string name, DocumentOptions options)
    {
        var file = new PdfFile(name, PdfSource.For(stream), options);
        file.ReadHeader();
        file._xref = PdfXrefReader.Load(file);
        return file;
    }

    private void ReadHeader()
    {
        using var head = PdfWindow.Read(Source, 0, PdfFormat.HeaderWithin + 16);
        var span = head.Span;
        int at = span.IndexOf("%PDF-"u8);
        if (at < 0 || at >= PdfFormat.HeaderWithin)
        {
            throw new InvalidDataException($"{Name}: not a PDF file (no %PDF- header in its first KiB).");
        }

        HeaderOffset = at;
        var version = span[(at + 5)..];
        int length = 0;
        while (length < version.Length && length < 8 && (char.IsAsciiDigit((char)version[length]) || version[length] == (byte)'.'))
        {
            length++;
        }

        HeaderVersion = length >= 3 ? PdfText.Latin1(version[..length]) : null;
    }

    /// <summary>Object <paramref name="number"/> (null when the file has none, or freed it).</summary>
    /// <exception cref="InvalidDataException">The object is damaged, or needs itself to be read (a cycle).</exception>
    public PdfValue Get(int number)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        while (true)
        {
            var xref = _xref;
            var entry = xref[number];
            if (entry.Kind is PdfEntryKind.Missing or PdfEntryKind.Free)
            {
                return PdfValue.Null;
            }

            if (_cache.TryGet(number, out var cached, out _))
            {
                return cached;
            }

            bool retry;
            Enter(number);
            try
            {
                PdfValue value;
                int bytes;
                if (entry.Kind == PdfEntryKind.InUse)
                {
                    value = ReadIndirect(xref, number, entry, out retry, out bytes);
                }
                else
                {
                    (value, retry) = (ReadCompressed(number, entry, out bytes), false);
                }

                if (!retry)
                {
                    _cache.Add(number, value, null, bytes);
                    return value;
                }
            }
            catch (InvalidDataException e) when (PdfLimit.Is(e) && !e.Message.StartsWith(Name, StringComparison.Ordinal))
            {
                throw PdfLimit.Error($"{Name}: object {number}: {e.Message}.");
            }
            finally
            {
                Exit();
            }
        }
    }

    /// <summary><paramref name="value"/>, or the object its reference names (through a chain of references).</summary>
    /// <exception cref="InvalidDataException">The chain comes back to an object it passed (a cycle), or an object is damaged.</exception>
    public PdfValue Resolve(PdfValue value)
    {
        if (!value.IsReference)
        {
            return value;
        }

        int first = value.AsReference.Number;
        var resolved = Get(first);
        if (!resolved.IsReference)
        {
            return resolved;
        }

        var chain = new List<int> { first };
        while (resolved.IsReference)
        {
            int next = resolved.AsReference.Number;
            if (chain.Contains(next))
            {
                chain.Add(next);
                throw PdfLimit.Error($"{Name}: a cycle of references: object {string.Join(" refers to ", chain)}; the file is damaged or hostile.");
            }

            chain.Add(next);
            if (chain.Count > MaxDepth + 16)
            {
                throw PdfLimit.Error($"{Name}: a chain of {chain.Count} references, longer than the limit {nameof(DocumentOptions.MaxDepth)} allows (DocumentOptions.MaxDepth, {MaxDepth}).");
            }

            resolved = Get(next);
        }

        return resolved;
    }

    /// <summary>The dictionary <paramref name="value"/> is or names (a stream's dictionary too), or null.</summary>
    public PdfDictionary? Dictionary(PdfValue value) => Resolve(value).AsDictionary;

    /// <summary>The objects in use in the newest revision: number, generation and how they are stored.</summary>
    public IEnumerable<(int Number, int Generation, PdfEntryKind Kind)> LiveObjects()
    {
        var xref = _xref;
        for (int number = 1; number < xref.Count; number++)
        {
            var entry = xref.Entries[number];
            if (entry.Kind is PdfEntryKind.InUse or PdfEntryKind.Compressed)
            {
                yield return (number, entry.Kind == PdfEntryKind.InUse ? entry.Generation : 0, entry.Kind);
            }
        }
    }

    /// <summary>
    /// The data of <paramref name="stream"/> through its filters (every one but an image encoding, whose data is handed
    /// over as it is), in a buffer the caller disposes.
    /// </summary>
    /// <exception cref="InvalidDataException">
    /// It stores or decodes to more than <see cref="DocumentOptions.MaxStreamBytes"/> (a <see cref="PdfLimit"/> error), a
    /// filter is unknown, or its data is damaged from the start.
    /// </exception>
    public PdfDecoded Decode(PdfStream stream)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (stream.Length > MaxStreamBytes)
        {
            throw PdfLimit.Error(
                $"{Name}: object {stream.Number}: a stream of {stream.Length:N0} stored bytes, more than the limit {nameof(DocumentOptions.MaxStreamBytes)} ({MaxStreamBytes:N0}; DocumentOptions.MaxStreamBytes, or {DocumentOptions.MaxStreamBytesVariable}).");
        }

        try
        {
            if (stream.Dictionary.Contains("F"u8))
            {
                throw new InvalidDataException("its data is in an outside file, which a reader never fetches");
            }

            var filters = Filters(stream.Dictionary);
            var window = PdfWindow.Read(Source, stream.Offset, stream.Length);
            if (_security is { } security)
            {
                window = security.Decrypt(stream, window, ref filters);   // never longer than the stored bytes
            }

            return PdfFilters.Decode(window.Memory, window, filters, MaxStreamBytes, PdfLimitMessage.Stream(this, stream.Number));
        }
        catch (InvalidDataException e) when (!e.Message.StartsWith(Name, StringComparison.Ordinal))
        {
            string message = $"{Name}: object {stream.Number}: {e.Message}";
            throw PdfLimit.Is(e) ? PdfLimit.Error(message) : new InvalidDataException(message, e);
        }
    }

    /// <summary>What <see cref="Decode"/> throws when stream <paramref name="number"/> decodes past <see cref="MaxStreamBytes"/> (made only then).</summary>
    internal string StreamOverLimit(int number) => string.Create(CultureInfo.InvariantCulture,
        $"{Name}: object {number}: a stream decodes to more than {MaxStreamBytes:N0} bytes, the limit {nameof(DocumentOptions.MaxStreamBytes)} (DocumentOptions.MaxStreamBytes, or {DocumentOptions.MaxStreamBytesVariable}); raise it to read this file.");

    /// <summary>A stream's (or an inline image's) filters and their parameters, every reference resolved.</summary>
    internal PdfFilter[] Filters(PdfDictionary dictionary)
    {
        var filter = Resolve(dictionary["Filter"u8]);
        var parameters = Resolve(dictionary.TryGet("DecodeParms"u8, out var parms) ? parms : dictionary["DP"u8]);
        if (filter.IsNull)
        {
            return [];
        }

        if (filter.AsName is { } single)
        {
            return [new PdfFilter(single, Shallow(parameters.AsArray is { Count: > 0 } list ? Resolve(list[0]).AsDictionary : parameters.AsDictionary))];
        }

        var names = filter.AsArray ?? throw new InvalidDataException($"a /Filter that is {filter}");
        if (names.Count > MaxDepth)
        {
            throw PdfLimit.Error($"a chain of {names.Count} filters, more than the limit {nameof(DocumentOptions.MaxDepth)} ({MaxDepth}; DocumentOptions.MaxDepth)");
        }

        var filters = new PdfFilter[names.Count];
        var parameterList = parameters.AsArray;
        for (int i = 0; i < filters.Length; i++)
        {
            var name = Resolve(names[i]).AsName ?? throw new InvalidDataException($"a /Filter entry that is {names[i]}");
            var own = parameterList is not null && i < parameterList.Count ? Resolve(parameterList[i]).AsDictionary : i == 0 ? parameters.AsDictionary : null;
            filters[i] = new PdfFilter(name, Shallow(own));
        }

        return filters;
    }

    // A parameters dictionary with its values resolved (most hold numbers directly; a reference is read once here).
    private PdfDictionary? Shallow(PdfDictionary? dictionary)
    {
        if (dictionary is null)
        {
            return null;
        }

        bool any = false;
        for (int i = 0; i < dictionary.Count && !any; i++)
        {
            any = dictionary.ValueAt(i).IsReference;
        }

        if (!any)
        {
            return dictionary;
        }

        var keys = new PdfName[dictionary.Count];
        var values = new PdfValue[dictionary.Count];
        for (int i = 0; i < keys.Length; i++)
        {
            keys[i] = dictionary.KeyAt(i);
            values[i] = Resolve(dictionary.ValueAt(i));
        }

        return new PdfDictionary(keys, values);
    }

    // An object at its offset; another object there (or none) means the cross-reference is wrong: repaired once, then read again.
    private PdfValue ReadIndirect(PdfXref xref, int number, PdfEntry entry, out bool retry, out int bytes)
    {
        retry = false;
        bytes = 0;
        try
        {
            var value = ParseAt(entry.Offset + xref.Shift, header: true, StreamMode.Resolved, out int found, out _, out _, out bytes);
            if (found == number)
            {
                return _security is { } security ? security.DecryptStrings(value, number, entry.Generation) : value;
            }
        }
        catch (Exception e) when (PdfXrefReader.Repairable(e))
        {
            if (xref.Repaired || !RepairNow(xref))
            {
                throw e is InvalidDataException && e.Message.StartsWith(Name, StringComparison.Ordinal) ? e
                    : new InvalidDataException($"{Name}: object {number}: {e.Message}", e);
            }

            retry = true;
            return PdfValue.Null;
        }

        retry = !xref.Repaired && RepairNow(xref);
        return PdfValue.Null;
    }

    // An object inside an object stream.
    private PdfValue ReadCompressed(int number, PdfEntry entry, out int bytes)
    {
        var stream = ObjectStream((int)Math.Min(entry.Offset, int.MaxValue));
        return stream.Read(this, number, entry.Generation, 0, out bytes);
    }

    /// <summary>The cache key of the font read from object <paramref name="number"/> (apart from objects' and object streams' keys).</summary>
    public static long FontKey(int number) => long.MinValue + number;

    /// <summary>A part read from the file and kept under <paramref name="key"/> (a font), or null.</summary>
    public object? Part(long key) => _cache.TryGet(key, out _, out object? kept) ? kept : null;

    /// <summary>Keeps a part read from the file (a font) for the document's other pages, while the cache has room.</summary>
    public void KeepPart(long key, object part, long bytes) => _cache.Add(key, PdfValue.Null, part, bytes);

    /// <summary>The object stream <paramref name="number"/>, decoded once and kept while the cache has room.</summary>
    public PdfObjectStream ObjectStream(int number)
    {
        if (_cache.TryGet(-1L - number, out _, out object? kept) && kept is PdfObjectStream cached)
        {
            return cached;
        }

        var stream = Get(number).AsStream ?? throw new InvalidDataException($"{Name}: object {number}, named as an object stream, is not a stream.");
        var parsed = ParseObjectStream(stream);
        _cache.Add(-1L - number, PdfValue.Null, parsed, parsed.Data.Length + parsed.Numbers.Length * 8L + 64);
        return parsed;
    }

    /// <summary>The object numbers of the object stream at <paramref name="offset"/> (a repair lists them).</summary>
    public int[] ObjectStreamNumbersAt(long offset)
    {
        var value = ParseAt(offset, header: true, StreamMode.Direct, out _, out _, out _, out _);
        var stream = value.AsStream ?? throw new InvalidDataException($"{Name}: no object stream at offset {offset}.");
        return ParseObjectStream(stream).Numbers;
    }

    private PdfObjectStream ParseObjectStream(PdfStream stream)
    {
        var dictionary = stream.Dictionary;
        if (!Resolve(dictionary["N"u8]).TryInteger(out long count) || !Resolve(dictionary["First"u8]).TryInteger(out long first) || count < 0 || first < 0)
        {
            throw new InvalidDataException($"{Name}: object {stream.Number}: an object stream without its /N and /First.");
        }

        if (count > MaxObjects)
        {
            throw ObjectsLimit(count, $"the object stream {stream.Number}");
        }

        count = Math.Min(count, first / 4 + 1);   // each object takes four bytes of the header at least: a larger /N lies

        using var decoded = Decode(stream);
        if (first > decoded.Length)
        {
            throw new InvalidDataException($"{Name}: object {stream.Number}: an object stream whose /First {first} is past its {decoded.Length} bytes.");
        }

        byte[] data = decoded.Span.ToArray();
        var numbers = new int[count];
        var offsets = new int[count];
        var parser = new PdfParser(data.AsSpan(0, (int)first), MaxDepth, complete: true);
        for (int i = 0; i < count; i++)
        {
            if (!parser.TryUnsigned(out long number) || !parser.TryUnsigned(out long offset) || number > int.MaxValue || first + offset > data.Length)
            {
                // A header cut short: the objects it lists are kept, the rest are not there.
                Array.Fill(numbers, -1, i, numbers.Length - i);
                break;
            }

            numbers[i] = (int)number;
            offsets[i] = (int)(first + offset);
        }

        return new PdfObjectStream(stream.Number, data, numbers, offsets, dictionary["Extends"u8]);
    }

    /// <summary>The value at <paramref name="offset"/> with no object header (a trailer's dictionary).</summary>
    public PdfValue ReadValueAt(long offset) => ParseAt(offset, header: false, StreamMode.None, out _, out _, out _, out _);

    /// <summary>The object at <paramref name="offset"/> with its stream's dictionary alone (<paramref name="isStream"/> says it is one): what a repair classifies objects by.</summary>
    public PdfValue ReadHeaderAt(long offset, out bool isStream) => ParseAt(offset, header: true, StreamMode.Header, out _, out _, out isStream, out _);

    private enum StreamMode
    {
        None,       // a value alone
        Header,     // a stream's dictionary, its length not looked for
        Direct,     // a stream, its /Length used only when written directly (while the cross-reference is not read yet)
        Resolved,   // a stream, its /Length resolved when indirect
    }

    // Parses at `offset`, reading a window that grows (by four) while the object runs past it.
    private PdfValue ParseAt(long offset, bool header, StreamMode streams, out int number, out int generation, out bool isStream, out int bytes)
    {
        long available = Source.Length - offset;
        if (offset < 0 || available <= 0)
        {
            throw new InvalidDataException($"{Name}: the offset {offset} is outside the file ({Source.Length:N0} bytes).");
        }

        long window = Source.InMemory ? available : Math.Min(FirstWindow, available);
        while (true)
        {
            bool complete = window >= available;
            using var bytesRead = PdfWindow.Read(Source, offset, window);
            try
            {
                var parser = new PdfParser(bytesRead.Span, MaxDepth, complete);
                number = generation = -1;
                if (header && !parser.TryObjectHeader(out number, out generation))
                {
                    throw new InvalidDataException($"no object at offset {offset}");
                }

                var value = parser.ReadValue();
                isStream = false;
                if (streams != StreamMode.None && value.Kind == PdfKind.Dictionary && parser.TryKeyword("stream"u8))
                {
                    isStream = true;
                    if (streams != StreamMode.Header)
                    {
                        long dataStart = offset + parser.StreamStart();
                        value = PdfValue.Of(StreamAt(value.AsDictionary!, number, generation, dataStart, streams == StreamMode.Resolved));
                    }
                }

                bytes = parser.Position;
                return value;
            }
            catch (PdfTruncatedException) when (!complete)
            {
                if (window >= Array.MaxLength)
                {
                    throw new InvalidDataException($"{Name}: an object at offset {offset} longer than one buffer holds.");
                }

                window = Math.Min(Math.Min(window * 4, available), Array.MaxLength);
            }
        }
    }

    /// <summary>
    /// A stream whose data starts at <paramref name="dataStart"/>: its length from <c>/Length</c> (direct, or resolved
    /// when <paramref name="resolveLength"/>) when <c>endstream</c> follows it, else found by looking for <c>endstream</c>.
    /// </summary>
    public PdfStream StreamAt(PdfDictionary dictionary, int number, int generation, long dataStart, bool resolveLength)
    {
        long length = -1;
        var written = dictionary["Length"u8];
        if (written.TryInteger(out long direct))
        {
            length = direct;
        }
        else if (resolveLength && written.IsReference && written.AsReference.Number != number)
        {
            try
            {
                if (Resolve(written).TryInteger(out long indirect))
                {
                    length = indirect;
                }
            }
            catch (InvalidDataException)
            {
                // An indirect length that cannot be read: endstream says where the data ends.
            }
        }

        if (length >= 0 && dataStart + length <= Source.Length && EndstreamAt(dataStart + length))
        {
            return new PdfStream(dictionary, number, generation, dataStart, length, lengthRecovered: false);
        }

        return new PdfStream(dictionary, number, generation, dataStart, FindEndstream(dataStart), lengthRecovered: true);
    }

    // Whether "endstream" (or, from writers that leave it out, "endobj") follows `at` after an end of line.
    private bool EndstreamAt(long at)
    {
        using var window = PdfWindow.Read(Source, at, 32);
        var span = window.Span;
        int i = 0;
        while (i < span.Length && PdfParser.IsWhitespace(span[i]))
        {
            i++;
        }

        return span[i..].StartsWith("endstream"u8) || span[i..].StartsWith("endobj"u8);
    }

    // The length of a stream whose /Length is missing or wrong: up to the end of line before the next "endstream" (or
    // "endobj"), looked for no further than the stream limit allows (past it, the stream is too long anyway).
    private long FindEndstream(long dataStart)
    {
        long end = Find(dataStart, "endstream"u8);
        if (end < 0)
        {
            end = Find(dataStart, "endobj"u8);
        }

        if (end < 0)
        {
            return Math.Max(0, Math.Min(Source.Length, dataStart + MaxStreamBytes + 1) - dataStart);
        }

        Span<byte> before = stackalloc byte[2];
        int count = end - dataStart >= 2 ? Source.Read(end - 2, before) : 0;
        if (count == 2 && before[1] == (byte)'\n')
        {
            end -= before[0] == (byte)'\r' ? 2 : 1;
        }
        else if (count == 2 && before[1] == (byte)'\r')
        {
            end--;
        }

        return Math.Max(0, end - dataStart);
    }

    // Where `keyword` next starts from `from`, looked for in steps, no further than a stream may be long; -1 when not found.
    private long Find(long from, ReadOnlySpan<byte> keyword)
    {
        const int Step = 64 * 1024;
        long last = Math.Min(Source.Length, from + MaxStreamBytes + Step);
        for (long at = from; at < last; at += Step)
        {
            using var window = PdfWindow.Read(Source, at, Step + 16);
            int found = window.Span.IndexOf(keyword);
            if (found >= 0 && at + found < last)
            {
                return at + found;
            }
        }

        return -1;
    }

    /// <summary>The trailer's or a cross-reference stream's <c>/Size</c>, within <see cref="DocumentOptions.MaxObjects"/>.</summary>
    public long CheckSize(PdfDictionary dictionary, string what)
    {
        if (!dictionary["Size"u8].TryInteger(out long size) || size < 0)
        {
            return 0;
        }

        return size > MaxObjects ? throw ObjectsLimit(size, what) : size;
    }

    /// <summary>The error for more objects than <see cref="DocumentOptions.MaxObjects"/>.</summary>
    public InvalidDataException ObjectsLimit(long count, string what) =>
        PdfLimit.Error($"{Name}: {what} claims {count:N0} objects, more than the limit {nameof(DocumentOptions.MaxObjects)} ({MaxObjects:N0}; DocumentOptions.MaxObjects).");

    /// <summary>
    /// Rebuilds the cross-reference by scanning the file, once per document, unless another thread did since
    /// <paramref name="seen"/> was read (then true: read again); false when it cannot (repaired or tried already, a repair
    /// under way on this thread, or nothing found by scanning).
    /// </summary>
    /// <exception cref="InvalidDataException">The scan passed a limit (a <see cref="PdfLimit"/> error).</exception>
    public bool RepairNow(PdfXref seen)
    {
        if (_repairGate.IsHeldByCurrentThread)
        {
            return false;   // a repair is reading the file on this thread: it does not start another
        }

        lock (_repairGate)
        {
            if (_xref != seen)
            {
                return true;
            }

            if (seen.Repaired || _repairTried)
            {
                return false;
            }

            _repairTried = true;
            try
            {
                _xref = PdfXrefReader.Repair(this);
            }
            catch (Exception e) when (PdfXrefReader.Repairable(e))
            {
                return false;   // no better index found: the object's own error stands
            }

            _cache.Clear();
            return true;
        }
    }

    public void Dispose()
    {
        _disposed = true;
        _cache.Clear();
        _security?.Dispose();   // the file key is held only while the document is open
    }

    // Marks `number` as being read by this thread: reading it again before it is done is a cycle.
    private void Enter(int number)
    {
        var reading = t_reading ??= [];
        foreach (var (file, n) in reading)
        {
            if (file == this && n == number)
            {
                var chain = reading.Where(r => r.File == this).Select(r => r.Number.ToString(CultureInfo.InvariantCulture));
                throw PdfLimit.Error($"{Name}: a cycle of references: object {number} is needed to read itself (through objects {string.Join(", ", chain)}); the file is damaged or hostile.");
            }
        }

        if (reading.Count > MaxDepth + 16)
        {
            throw PdfLimit.Error($"{Name}: objects that need {reading.Count} others to be read, nested deeper than the limit {nameof(DocumentOptions.MaxDepth)} allows (DocumentOptions.MaxDepth, {MaxDepth}).");
        }

        reading.Add((this, number));
    }

    private static void Exit()
    {
        var reading = t_reading!;
        reading.RemoveAt(reading.Count - 1);
    }
}

/// <summary>An object stream (7.5.7) decoded once: its data, the objects' numbers and where each starts.</summary>
internal sealed class PdfObjectStream(int streamNumber, byte[] data, int[] numbers, int[] offsets, PdfValue extends)
{
    public int Number { get; } = streamNumber;

    public byte[] Data { get; } = data;

    /// <summary>The objects it holds, in order (-1 past a header cut short).</summary>
    public int[] Numbers { get; } = numbers;

    /// <summary>Object <paramref name="number"/>, at <paramref name="index"/> or wherever this stream (or one it extends) lists it; null when none does.</summary>
    public PdfValue Read(PdfFile file, int number, int index, int hops, out int bytes)
    {
        bytes = 0;
        int i = (uint)index < (uint)Numbers.Length && Numbers[index] == number ? index : Array.IndexOf(Numbers, number);
        if (i < 0)
        {
            // Not here: a stream this one extends may hold it (7.5.7, /Extends), followed no further than nesting allows.
            if (extends.IsReference && hops < file.MaxDepth && extends.AsReference.Number != Number)
            {
                return file.ObjectStream(extends.AsReference.Number).Read(file, number, -1, hops + 1, out bytes);
            }

            return PdfValue.Null;
        }

        var parser = new PdfParser(Data.AsSpan(offsets[i]), file.MaxDepth, complete: true);
        try
        {
            var value = parser.ReadValue();
            bytes = parser.Position * 2 + 64;
            return value;
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            throw new InvalidDataException($"{file.Name}: object {number} in the object stream {Number}: {e.Message}", e);
        }
    }
}

/// <summary>
/// The objects a document keeps parsed, and its decoded object streams, within a number of bytes (an estimate of each
/// one's size): when full, the first kept are dropped first. Safe from several threads.
/// </summary>
internal sealed class PdfCache(long limit)
{
    private readonly ConcurrentDictionary<long, (PdfValue Value, object? Extra, long Bytes)> _entries = new();
    private readonly ConcurrentQueue<long> _order = new();
    private long _bytes;

    public bool TryGet(long key, out PdfValue value, out object? extra)
    {
        if (_entries.TryGetValue(key, out var entry))
        {
            (value, extra) = (entry.Value, entry.Extra);
            return true;
        }

        (value, extra) = (PdfValue.Null, null);
        return false;
    }

    public void Add(long key, PdfValue value, object? extra, long bytes)
    {
        bytes = Math.Max(bytes, 64);
        if (bytes > limit || !_entries.TryAdd(key, (value, extra, bytes)))
        {
            return;
        }

        _order.Enqueue(key);
        if (Interlocked.Add(ref _bytes, bytes) > limit)
        {
            while (Volatile.Read(ref _bytes) > limit && _order.TryDequeue(out long oldest))
            {
                if (_entries.TryRemove(oldest, out var dropped))
                {
                    Interlocked.Add(ref _bytes, -dropped.Bytes);
                }
            }
        }
    }

    public void Clear()
    {
        _entries.Clear();
        _order.Clear();
        Interlocked.Exchange(ref _bytes, 0);
    }
}
