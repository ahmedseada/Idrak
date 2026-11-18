// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Data.Pdf;

/// <summary>What a cross-reference entry says of an object.</summary>
internal enum PdfEntryKind : byte
{
    /// <summary>Not listed.</summary>
    Missing,

    /// <summary>Freed (a newer revision deleted it).</summary>
    Free,

    /// <summary>In the file at an offset.</summary>
    InUse,

    /// <summary>Inside an object stream.</summary>
    Compressed,
}

/// <summary>
/// One object's entry: in use at <see cref="Offset"/> (from the file's start) with its generation, or compressed in the
/// object stream <see cref="Offset"/> at index <see cref="Generation"/>, or free.
/// </summary>
internal readonly record struct PdfEntry(PdfEntryKind Kind, long Offset, int Generation);

/// <summary>A cross-reference section as the file has it: its kind ("table", "stream", "objstm" for a stream listing compressed objects, "hybrid") and its offset as written.</summary>
internal sealed record PdfSection(string Kind, long Offset)
{
    /// <summary>The objects it lists in use, compressed (with their object stream) and free.</summary>
    public required IReadOnlyList<(int Number, PdfEntry Entry)> Entries { get; init; }
}

/// <summary>
/// A document's cross-reference: every object's entry (the newest revision's, an array by object number), the trailer
/// (every revision's keys, the newest winning), the sections read (newest first) and whether it was rebuilt by scanning.
/// Immutable: a repair makes a new one.
/// </summary>
internal sealed class PdfXref(PdfEntry[] entries, int count, PdfDictionary trailer, IReadOnlyList<PdfSection> sections, long shift, bool repaired)
{
    public PdfEntry[] Entries { get; } = entries;

    /// <summary>One past the highest object number listed.</summary>
    public int Count { get; } = count;

    public PdfDictionary Trailer { get; } = trailer;

    /// <summary>The sections read, newest first (empty when repaired).</summary>
    public IReadOnlyList<PdfSection> Sections { get; } = sections;

    /// <summary>What every offset of the file is moved by: the bytes before the header, when its offsets count from it.</summary>
    public long Shift { get; } = shift;

    public bool Repaired { get; } = repaired;

    public PdfEntry this[int number] => (uint)number < (uint)Count ? Entries[number] : default;
}

/// <summary>
/// Reads the cross-reference (ISO 32000-2, 7.5.4-7.5.8): from <c>startxref</c> through every <c>/Prev</c> (a loop is a
/// cycle, reported), classic tables and their trailers, cross-reference streams (<c>/W</c>, <c>/Index</c>) and hybrid
/// files (<c>/XRefStm</c>), the newest entry of each object winning. Offsets count from the file's start or, when that
/// finds nothing and bytes come before the header, from the header. A table that cannot be read is rebuilt by
/// <see cref="Repair"/>.
/// </summary>
internal static class PdfXrefReader
{
    private const int Tail = 4096;

    /// <summary>The cross-reference of <paramref name="file"/>, rebuilt by scanning when it cannot be read.</summary>
    /// <exception cref="InvalidDataException">A limit was passed, or the sections' <c>/Prev</c> entries form a cycle (a <see cref="PdfLimit"/> error).</exception>
    public static PdfXref Load(PdfFile file)
    {
        long? start = FindStartXref(file);
        if (start is not { } offset)
        {
            return Repair(file);
        }

        long shift;
        if (LooksLikeSection(file, offset))
        {
            shift = 0;
        }
        else if (file.HeaderOffset > 0 && LooksLikeSection(file, offset + file.HeaderOffset))
        {
            shift = file.HeaderOffset;
        }
        else
        {
            return Repair(file);
        }

        try
        {
            var xref = LoadChain(file, offset, shift);
            return xref.Trailer.Contains("Root"u8) ? xref : Repair(file);
        }
        catch (Exception e) when (Repairable(e))
        {
            return Repair(file);
        }
    }

    /// <summary>Whether <paramref name="e"/> is damage a repair may get around (not a limit, a cycle or a refusal).</summary>
    public static bool Repairable(Exception e) =>
        (e is InvalidDataException && !PdfLimit.Is(e)) || e is IndexOutOfRangeException or ArgumentException or OverflowException or FormatException or EndOfStreamException;

    // The last "startxref" in the file's tail and the offset after it.
    private static long? FindStartXref(PdfFile file)
    {
        foreach (long size in (ReadOnlySpan<long>)[Tail, 64 * 1024])
        {
            long from = Math.Max(0, file.Source.Length - size);
            using var window = PdfWindow.Read(file.Source, from, size);
            var span = window.Span;
            int at = span.LastIndexOf("startxref"u8);
            if (at >= 0)
            {
                var parser = new PdfParser(span[(at + 9)..], file.MaxDepth, complete: true);
                return parser.TryUnsigned(out long offset) ? offset : null;
            }

            if (from == 0)
            {
                break;
            }
        }

        return null;
    }

    // Whether a section starts at `offset`: "xref", or a cross-reference stream's object (its dictionary /Type /XRef).
    private static bool LooksLikeSection(PdfFile file, long offset)
    {
        if (offset < 0 || offset >= file.Source.Length)
        {
            return false;
        }

        using var window = PdfWindow.Read(file.Source, offset, 4096);
        var parser = new PdfParser(window.Span, file.MaxDepth, complete: true);
        if (parser.TryKeyword("xref"u8))
        {
            return true;
        }

        try
        {
            return parser.TryObjectHeader(out _, out _) && parser.ReadValue().AsDictionary is { } dictionary && dictionary["Type"u8].IsName("XRef"u8);
        }
        catch (InvalidDataException)
        {
            return false;   // not an object that ends within the window: not a section
        }
    }

    private static PdfXref LoadChain(PdfFile file, long first, long shift)
    {
        var map = new EntryMap(file);
        var sections = new List<PdfSection>();
        var trailers = new List<PdfDictionary>();
        var seen = new List<long>();
        long? offset = first;
        while (offset is { } at)
        {
            if (seen.Contains(at))
            {
                throw PdfLimit.Error($"{file.Name}: the cross-reference sections' /Prev entries form a cycle (offset {at} is reached twice, after {string.Join(", ", seen)}).");
            }

            seen.Add(at);
            if (seen.Count > file.MaxObjects)
            {
                throw file.ObjectsLimit(seen.Count, "cross-reference sections");
            }

            var section = ReadSection(file, at + shift, file.MaxObjects);
            var trailer = section.Trailer;
            List<(int Number, PdfEntry Entry)> entries = section.Entries;
            string kind = section.Kind;
            if (section.Kind == "table" && trailer["XRefStm"u8].TryInteger(out long hidden))
            {
                // A hybrid file: the hidden stream lists the objects a reader of tables does not see; read first, so the
                // table's free entries for them do not hide them.
                var stream = ReadSection(file, hidden + shift, file.MaxObjects);
                if (stream.Kind is not ("stream" or "objstm"))
                {
                    throw new InvalidDataException($"/XRefStm {hidden} is not a cross-reference stream");
                }

                entries = [.. stream.Entries, .. section.Entries];
                kind = "hybrid";
            }

            map.Merge(entries);
            trailers.Add(trailer);
            sections.Add(new PdfSection(kind, at) { Entries = entries });
            offset = trailer["Prev"u8].TryInteger(out long previous) && previous >= 0 ? previous : null;
        }

        return new PdfXref(map.Entries, map.Count, MergeTrailers(trailers), sections, shift, repaired: false);
    }

    // One section: "xref" and its subsections, then the trailer; or a cross-reference stream object.
    private static (string Kind, List<(int Number, PdfEntry Entry)> Entries, PdfDictionary Trailer) ReadSection(PdfFile file, long offset, long maxObjects)
    {
        long window = 16 * 1024;
        while (true)
        {
            long available = file.Source.Length - offset;
            if (offset < 0 || available <= 0)
            {
                throw new InvalidDataException($"a cross-reference offset {offset} outside the file");
            }

            bool complete = window >= available;
            using var bytes = PdfWindow.Read(file.Source, offset, Math.Min(window, available));
            try
            {
                var parser = new PdfParser(bytes.Span, file.MaxDepth, complete);
                if (parser.TryKeyword("xref"u8))
                {
                    return ReadTable(file, ref parser, maxObjects);
                }

                if (parser.TryObjectHeader(out int number, out int generation))
                {
                    var value = parser.ReadValue();
                    if (value.AsDictionary is not { } dictionary || value.Kind != PdfKind.Dictionary || !parser.TryKeyword("stream"u8))
                    {
                        throw new InvalidDataException($"object {number} at a cross-reference offset is not a stream");
                    }

                    long dataStart = offset + parser.StreamStart();
                    var stream = file.StreamAt(dictionary, number, generation, dataStart, resolveLength: false);
                    return ReadStream(file, stream, maxObjects);
                }

                throw new InvalidDataException($"no cross-reference at offset {offset}");
            }
            catch (PdfTruncatedException) when (!complete)
            {
                window = Math.Min(window * 4, Math.Min(available, Array.MaxLength));
            }
        }
    }

    // A classic table after "xref": subsections of "start count" and their entries "offset generation n|f".
    private static (string, List<(int, PdfEntry)>, PdfDictionary) ReadTable(PdfFile file, ref PdfParser parser, long maxObjects)
    {
        var entries = new List<(int, PdfEntry)>();
        while (!parser.TryKeyword("trailer"u8))
        {
            if (!parser.TryUnsigned(out long start) || !parser.TryUnsigned(out long count))
            {
                throw new InvalidDataException("a cross-reference table without its trailer");
            }

            if (start + count > maxObjects)
            {
                throw file.ObjectsLimit(start + count, "a cross-reference table");
            }

            for (long k = 0; k < count; k++)
            {
                if (!parser.TryUnsigned(out long offset) || !parser.TryUnsigned(out long generation))
                {
                    throw new InvalidDataException("a cross-reference entry that is not \"offset generation n|f\"");
                }

                parser.SkipWhitespace();
                var type = parser.Token();
                int number = (int)(start + k);
                if (number == 0)
                {
                    continue;   // the head of the free list
                }

                if (type.SequenceEqual("n"u8))
                {
                    entries.Add((number, offset > 0 ? new PdfEntry(PdfEntryKind.InUse, offset, (int)Math.Min(generation, int.MaxValue)) : new PdfEntry(PdfEntryKind.Free, 0, 0)));
                }
                else if (type.SequenceEqual("f"u8))
                {
                    entries.Add((number, new PdfEntry(PdfEntryKind.Free, 0, (int)Math.Min(generation, int.MaxValue))));
                }
                else
                {
                    throw new InvalidDataException($"a cross-reference entry of type '{PdfText.Latin1(type)}'");
                }
            }
        }

        var trailer = parser.ReadValue().AsDictionary ?? throw new InvalidDataException("a trailer that is not a dictionary");
        file.CheckSize(trailer, "the trailer");
        return ("table", entries, trailer);
    }

    // A cross-reference stream: /W field widths, /Index subsections (default [0 /Size]), its data decoded.
    private static (string, List<(int, PdfEntry)>, PdfDictionary) ReadStream(PdfFile file, PdfStream stream, long maxObjects)
    {
        var dictionary = stream.Dictionary;
        if (!dictionary["Type"u8].IsName("XRef"u8))
        {
            throw new InvalidDataException($"object {stream.Number} at a cross-reference offset is not /Type /XRef");
        }

        long size = file.CheckSize(dictionary, "the cross-reference stream");
        var widths = dictionary["W"u8].AsArray;
        if (widths is null || widths.Count < 3)
        {
            throw new InvalidDataException("a cross-reference stream without its /W");
        }

        Span<int> w = stackalloc int[3];
        for (int i = 0; i < 3; i++)
        {
            if (!widths[i].TryInteger(out long width) || width is < 0 or > 8)
            {
                throw new InvalidDataException($"a cross-reference stream's /W {widths[i]}");
            }

            w[i] = (int)width;
        }

        int entryWidth = w[0] + w[1] + w[2];
        if (entryWidth == 0)
        {
            throw new InvalidDataException("a cross-reference stream whose entries have no bytes");
        }

        var index = dictionary["Index"u8].AsArray;
        var ranges = new List<(long Start, long Count)>();
        if (index is null)
        {
            ranges.Add((0, size));
        }
        else
        {
            for (int i = 0; i + 1 < index.Count; i += 2)
            {
                if (!index[i].TryInteger(out long start) || !index[i + 1].TryInteger(out long count) || start < 0 || count < 0)
                {
                    throw new InvalidDataException("a cross-reference stream's /Index");
                }

                ranges.Add((start, count));
            }
        }

        foreach (var (start, count) in ranges)
        {
            if (start + count > maxObjects)
            {
                throw file.ObjectsLimit(start + count, "the cross-reference stream");
            }
        }

        var entries = new List<(int, PdfEntry)>();
        bool compressed = false;
        using (var decoded = file.Decode(stream))
        {
            var data = decoded.Span;
            int at = 0;
            foreach (var (start, count) in ranges)
            {
                for (long k = 0; k < count && at + entryWidth <= data.Length; k++, at += entryWidth)
                {
                    long type = w[0] == 0 ? 1 : Field(data.Slice(at, w[0]));
                    long second = Field(data.Slice(at + w[0], w[1]));
                    long third = Field(data.Slice(at + w[0] + w[1], w[2]));
                    int number = (int)(start + k);
                    if (number == 0 && type != 1)
                    {
                        continue;
                    }

                    switch (type)
                    {
                        case 0:
                            entries.Add((number, new PdfEntry(PdfEntryKind.Free, 0, (int)Math.Min(third, int.MaxValue))));
                            break;
                        case 1:
                            entries.Add((number, second > 0 ? new PdfEntry(PdfEntryKind.InUse, second, (int)Math.Min(third, int.MaxValue)) : new PdfEntry(PdfEntryKind.Free, 0, 0)));
                            break;
                        case 2:
                            if (second <= int.MaxValue && third <= int.MaxValue)
                            {
                                entries.Add((number, new PdfEntry(PdfEntryKind.Compressed, second, (int)third)));
                                compressed = true;
                            }

                            break;
                        default:
                            break;   // a type the standard keeps for later: the reference is to null
                    }
                }
            }
        }

        return (compressed ? "objstm" : "stream", entries, dictionary);
    }

    // A big-endian field of 0 to 8 bytes.
    private static long Field(ReadOnlySpan<byte> bytes)
    {
        long value = 0;
        foreach (byte b in bytes)
        {
            value = (value << 8) | b;
        }

        return value;
    }

    // Every revision's trailer keys, the newest (first) winning.
    private static PdfDictionary MergeTrailers(List<PdfDictionary> trailers)
    {
        if (trailers.Count == 1)
        {
            return trailers[0];
        }

        var keys = new List<PdfName>();
        var values = new List<PdfValue>();
        foreach (var trailer in trailers)
        {
            for (int i = 0; i < trailer.Count; i++)
            {
                var key = trailer.KeyAt(i);
                if (!keys.Exists(k => k.Bytes.SequenceEqual(key.Bytes)))
                {
                    keys.Add(key);
                    values.Add(trailer.ValueAt(i));
                }
            }
        }

        return new PdfDictionary([.. keys], [.. values]);
    }

    /// <summary>
    /// The cross-reference rebuilt by scanning the whole file, in order, for <c>n g obj</c> (a later definition of a
    /// number winning, as an update's does) and <c>trailer</c>; objects held in object streams are listed from those
    /// streams' headers, and the catalog is found by its type when no trailer names it.
    /// </summary>
    /// <exception cref="InvalidDataException">No document catalog is found, or more objects than the limit (a <see cref="PdfLimit"/> error).</exception>
    public static PdfXref Repair(PdfFile file)
    {
        const int Chunk = 1 << 20, Before = 32;
        var found = new Dictionary<int, (long Offset, int Generation)>();
        var trailerOffsets = new List<long>();
        long length = file.Source.Length;
        for (long region = 0; region < length; region += Chunk)
        {
            long from = Math.Max(0, region - Before);
            using var window = PdfWindow.Read(file.Source, from, region + Chunk + 16 - from);
            var span = window.Span;
            int lowest = (int)(region - from), highest = (int)Math.Min(span.Length, region + Chunk - from);
            for (int at = span[lowest..].IndexOf("obj"u8) + lowest; at >= lowest && at < highest; at = Next(span, at + 1, "obj"u8, highest))
            {
                if (ObjectAt(span, at, out int number, out int generation, out int start))
                {
                    found[number] = (from + start, generation);
                    if (found.Count > file.MaxObjects)
                    {
                        throw file.ObjectsLimit(found.Count, "the objects found repairing the file");
                    }
                }
            }

            for (int at = span[lowest..].IndexOf("trailer"u8) + lowest; at >= lowest && at < highest; at = Next(span, at + 1, "trailer"u8, highest))
            {
                bool before = at == 0 || PdfParser.IsWhitespace(span[at - 1]) || PdfParser.IsDelimiter(span[at - 1]);
                bool after = at + 7 >= span.Length || PdfParser.IsWhitespace(span[at + 7]) || PdfParser.IsDelimiter(span[at + 7]);
                if (before && after)
                {
                    trailerOffsets.Add(from + at + 7);
                }
            }
        }

        long count = found.Count == 0 ? 0 : found.Keys.Max() + 1L;
        if (count > file.MaxObjects || count > Array.MaxLength)
        {
            throw file.ObjectsLimit(count, "the objects found repairing the file");
        }

        var entries = new PdfEntry[count];
        foreach (var (number, (offset, generation)) in found)
        {
            entries[number] = new PdfEntry(PdfEntryKind.InUse, offset, generation);
        }

        // What each object is: object streams to list, cross-reference streams and catalogs to find the trailer by.
        var objectStreams = new List<(long Offset, int Number)>();
        var trailers = new List<(long Offset, PdfDictionary Trailer)>();
        var catalogs = new List<(long Offset, int Number, int Generation)>();
        foreach (var (number, (offset, generation)) in found)
        {
            PdfValue value;
            bool isStream;
            try
            {
                value = file.ReadHeaderAt(offset, out isStream);
            }
            catch (Exception e) when (Repairable(e))
            {
                continue;   // a damaged object: listed, read (and failed) when asked for
            }

            if (value.AsDictionary is not { } dictionary)
            {
                continue;
            }

            if (dictionary["Type"u8].IsName("ObjStm"u8) && isStream)
            {
                objectStreams.Add((offset, number));
            }
            else if (dictionary["Type"u8].IsName("XRef"u8) && dictionary.Contains("Root"u8))
            {
                trailers.Add((offset, dictionary));
            }
            else if (dictionary["Type"u8].IsName("Catalog"u8))
            {
                catalogs.Add((offset, number, generation));
            }
        }

        foreach (long offset in trailerOffsets)
        {
            try
            {
                if (file.ReadValueAt(offset).AsDictionary is { } written)
                {
                    trailers.Add((offset, written));
                }
            }
            catch (Exception e) when (Repairable(e))
            {
                // A damaged trailer adds nothing.
            }
        }

        // Objects in object streams, the newest stream first, where no object of that number stands on its own.
        objectStreams.Sort((a, b) => b.Offset.CompareTo(a.Offset));
        foreach (var (offset, stream) in objectStreams)
        {
            int[] numbers;
            try
            {
                numbers = file.ObjectStreamNumbersAt(offset);
            }
            catch (Exception e) when (Repairable(e))
            {
                continue;
            }

            for (int i = 0; i < numbers.Length; i++)
            {
                int number = numbers[i];
                if (number < 0)
                {
                    continue;
                }

                if (number >= entries.Length)
                {
                    if (number + 1L > file.MaxObjects || number + 1L > Array.MaxLength)
                    {
                        throw file.ObjectsLimit(number + 1L, "the objects found repairing the file");
                    }

                    Array.Resize(ref entries, number + 1);
                }

                if (entries[number].Kind == PdfEntryKind.Missing)
                {
                    entries[number] = new PdfEntry(PdfEntryKind.Compressed, stream, i);
                }
            }
        }

        trailers.Sort((a, b) => b.Offset.CompareTo(a.Offset));
        var trailer = trailers.Count == 0 ? PdfDictionary.Empty : MergeTrailers([.. trailers.Select(t => t.Trailer)]);

        if (!RootIsCatalog(file, trailer, entries))
        {
            catalogs.Sort((a, b) => b.Offset.CompareTo(a.Offset));
            if (catalogs.Count == 0)
            {
                throw new InvalidDataException($"{file.Name}: no document catalog found, by its cross-reference or by scanning the file (is it a PDF file?).");
            }

            var root = catalogs[0];
            trailer = WithRoot(trailer, PdfValue.Reference(root.Number, root.Generation));
        }

        return new PdfXref(entries, entries.Length, trailer, [], 0, repaired: true);
    }

    private static int Next(ReadOnlySpan<byte> span, int from, ReadOnlySpan<byte> what, int highest)
    {
        if (from >= highest)
        {
            return -1;
        }

        int at = span[from..].IndexOf(what);
        return at < 0 ? -1 : from + at;
    }

    // Whether "obj" at `at` ends "n g obj": a number, whitespace, a number, whitespace (or none), at a token's start.
    private static bool ObjectAt(ReadOnlySpan<byte> span, int at, out int number, out int generation, out int start)
    {
        number = generation = start = 0;
        if (at + 3 < span.Length && !PdfParser.IsWhitespace(span[at + 3]) && !PdfParser.IsDelimiter(span[at + 3]))
        {
            return false;
        }

        int i = at - 1;
        while (i >= 0 && PdfParser.IsWhitespace(span[i]))
        {
            i--;
        }

        int end = i + 1;
        while (i >= 0 && char.IsAsciiDigit((char)span[i]))
        {
            i--;
        }

        if (end - (i + 1) is < 1 or > 5 || i < 0 || !PdfParser.IsWhitespace(span[i]))
        {
            return false;
        }

        if (!int.TryParse(span[(i + 1)..end], NumberStyles.None, CultureInfo.InvariantCulture, out generation))
        {
            return false;
        }

        while (i >= 0 && PdfParser.IsWhitespace(span[i]))
        {
            i--;
        }

        end = i + 1;
        while (i >= 0 && char.IsAsciiDigit((char)span[i]))
        {
            i--;
        }

        if (end - (i + 1) is < 1 or > 10 || (i >= 0 && !PdfParser.IsWhitespace(span[i]) && !PdfParser.IsDelimiter(span[i])))
        {
            return false;
        }

        start = i + 1;
        return int.TryParse(span[start..end], NumberStyles.None, CultureInfo.InvariantCulture, out number);
    }

    private static bool RootIsCatalog(PdfFile file, PdfDictionary trailer, PdfEntry[] entries)
    {
        var root = trailer["Root"u8];
        if (!root.IsReference || (uint)root.AsReference.Number >= (uint)entries.Length || entries[root.AsReference.Number].Kind != PdfEntryKind.InUse)
        {
            return root.IsReference && (uint)root.AsReference.Number < (uint)entries.Length && entries[root.AsReference.Number].Kind == PdfEntryKind.Compressed;
        }

        try
        {
            var dictionary = file.ReadHeaderAt(entries[root.AsReference.Number].Offset, out _).AsDictionary;
            return dictionary is not null && (dictionary["Type"u8].IsName("Catalog"u8) || dictionary.Contains("Pages"u8));
        }
        catch (Exception e) when (Repairable(e))
        {
            return false;
        }
    }

    private static PdfDictionary WithRoot(PdfDictionary trailer, PdfValue root)
    {
        var keys = new List<PdfName> { PdfName.From("Root"u8) };
        var values = new List<PdfValue> { root };
        for (int i = 0; i < trailer.Count; i++)
        {
            if (!trailer.KeyAt(i).Is("Root"u8))
            {
                keys.Add(trailer.KeyAt(i));
                values.Add(trailer.ValueAt(i));
            }
        }

        return new PdfDictionary([.. keys], [.. values]);
    }

    // Entries merged section by section, newest first: an object keeps the first entry it is given.
    private sealed class EntryMap(PdfFile file)
    {
        public PdfEntry[] Entries { get; private set; } = new PdfEntry[64];

        public int Count { get; private set; }

        public void Merge(List<(int Number, PdfEntry Entry)> entries)
        {
            foreach (var (number, entry) in entries)
            {
                if (number < 0)
                {
                    continue;
                }

                if (number >= Entries.Length)
                {
                    if (number + 1L > file.MaxObjects)
                    {
                        throw file.ObjectsLimit(number + 1L, "the cross-reference");
                    }

                    var larger = new PdfEntry[(int)Math.Min(Math.Max(Entries.Length * 2L, number + 1L), file.MaxObjects)];
                    Entries.CopyTo(larger, 0);
                    Entries = larger;
                }

                if (Entries[number].Kind == PdfEntryKind.Missing)
                {
                    Entries[number] = entry;
                }

                Count = Math.Max(Count, number + 1);
            }
        }
    }
}
