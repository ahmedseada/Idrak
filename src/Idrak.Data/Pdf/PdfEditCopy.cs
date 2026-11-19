// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Globalization;
using System.Security.Cryptography;

namespace Idrak.Data.Pdf;

/// <summary>
/// Copies objects of the documents an edit or a merge takes pages from into the file being written. Each object is read
/// once (decrypted as its document reads it) and the objects it refers to are followed depth first and written before it,
/// so an object whose content, with the numbers its references got, is already written is not written again: images,
/// fonts and their programs that several files (or one) hold twice are written once, found by a SHA-256 of their content.
/// Objects are renumbered, or, for the document an incremental update appends to, referred to by their own numbers. A
/// reference to a page the result does not hold is written as null (or, while a builder still adds pages, numbered now
/// and written as null at the end if the page never comes), one to a page-tree node as the result's page tree. Streams are
/// copied as they are stored (decrypted, and encrypted again for the result when it is), and, with
/// <see cref="Recompress"/>, deflated when stored plain or deflated at a faster level than the options' (the zlib header
/// says which), kept so only when smaller. A chain of references deeper than a few dozen objects (a long list of
/// bookmarks) is numbered and written after, so the walk never runs deep. Not safe from several threads.
/// </summary>
internal sealed class PdfEditCopier : IDisposable
{
    // Past this depth an object is numbered now and written after (Drain).
    private const int Deepest = 48;

    // A source object's number while it is being written and has none yet; one that is null in the result.
    private const int Pending = 0, Null = -1;

    private readonly PdfFileWriter _file;
    private readonly int _level;
    private readonly Dictionary<PdfDocument, Source> _sources = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<ContentKey, int> _written = [];
    private readonly Queue<(Source Source, int Number)> _later = new();
    private readonly Stack<PdfSyntax> _free = new();
    private readonly IncrementalHash _hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
    private readonly Stack<PdfBuffer> _freeBuffers = new();

    /// <param name="file">Where the objects go.</param>
    /// <param name="level">The Flate level of the result (<see cref="DocumentWriteOptions.Compression"/>).</param>
    /// <param name="recompress">Whether streams stored plain or deflated faster are deflated at <paramref name="level"/> when that is smaller.</param>
    public PdfEditCopier(PdfFileWriter file, int level, bool recompress) => (_file, _level, Recompress) = (file, level, recompress);

    /// <summary>The result's page tree: what references to the sources' page-tree nodes become.</summary>
    public int Root { get; set; }

    /// <summary>Whether plain and quickly deflated streams are deflated again when it pays.</summary>
    public bool Recompress { get; }

    /// <summary>
    /// The source <paramref name="document"/> is copied from, made on its first use: <paramref name="identity"/> refers to
    /// its objects by their own numbers (the document an update appends to); <paramref name="reserve"/> numbers a page
    /// referred to before it is copied (a builder adding pages), else such a reference is null.
    /// </summary>
    public Source For(PdfDocument document, bool identity = false, bool reserve = false)
    {
        if (!_sources.TryGetValue(document, out var source))
        {
            var pages = new HashSet<int>();
            for (int i = 0; i < (identity ? 0 : document.PageCount); i++)   // referred to as it is: its pages need no listing
            {
                var node = document.PageObject(i).Node;
                if (node.IsReference)
                {
                    pages.Add(node.AsReference.Number);
                }
            }

            source = new Source(document, identity, reserve, pages);
            _sources.Add(document, source);
        }

        return source;
    }

    /// <summary>Makes <paramref name="number"/> the result's page for the source's page object <paramref name="page"/> (what references to it become).</summary>
    public static void MapPage(Source source, int page, int number) => source.PageNumbers[page] = number;

    /// <summary>
    /// The number the source's page object <paramref name="page"/> is written under: the one a reference to it reserved,
    /// else a new one (a page copied a second time gets a new one too).
    /// </summary>
    public int PageNumber(Source source, int page)
    {
        if (source.PageNumbers.TryGetValue(page, out int number) && source.PagesWritten.Add(page))
        {
            return number;
        }

        number = _file.Allocate();
        if (source.PagesWritten.Add(page))
        {
            source.PageNumbers[page] = number;
        }

        return number;
    }

    /// <summary>
    /// The result's reference for <paramref name="source"/>'s <paramref name="reference"/>: the object copied (with what it
    /// refers to) when it is not yet, or its own number for a source referred to as it is; a number below 0 is null.
    /// </summary>
    public (int Number, int Generation) Reference(Source source, PdfReference reference, int depth = 0)
    {
        if (source.Identity)
        {
            return (reference.Number, reference.Generation);
        }

        int number = reference.Number;
        if (source.Pages.Contains(number))
        {
            if (source.PageNumbers.TryGetValue(number, out int page))
            {
                return (page, 0);
            }

            if (!source.Reserve)
            {
                return (Null, 0);
            }

            page = _file.Allocate();
            source.PageNumbers[number] = page;
            return (page, 0);
        }

        if (source.Numbers.TryGetValue(number, out int known))
        {
            if (known != Pending)
            {
                return (known, 0);
            }

            // Being written, further up: numbered now (a cycle), and so never taken for another object's duplicate.
            known = _file.Allocate();
            source.Numbers[number] = known;
            return (known, 0);
        }

        if (depth >= Deepest)
        {
            int later = _file.Allocate();
            source.Numbers[number] = later;
            _later.Enqueue((source, number));
            return (later, 0);
        }

        return (Copy(source, number, Pending, depth), 0);
    }

    /// <summary>
    /// Writes <paramref name="value"/>, an object of <paramref name="source"/>, with every reference made the result's
    /// (copying what it names first): numbers as the shortest text that reads back as them, strings literal or hex
    /// (whichever is shorter), a space only where the syntax needs one.
    /// </summary>
    public void Value(PdfSyntax s, Source source, PdfValue value, int depth = 0)
    {
        switch (value.Kind)
        {
            case PdfKind.Null:
                s.Raw("null"u8);
                return;
            case PdfKind.Boolean:
                s.Raw(value.AsBoolean ? "true"u8 : "false"u8);
                return;
            case PdfKind.Integer:
                value.TryInteger(out long integer);
                s.Integer(integer);
                return;
            case PdfKind.Real:
                value.TryNumber(out double real);
                Real(s, real);
                return;
            case PdfKind.Reference:
                var (number, generation) = Reference(source, value.AsReference, depth);
                if (number < 0)
                {
                    s.Raw("null"u8);
                }
                else
                {
                    s.Integer(number).Space().Integer(generation).Raw(" R"u8);
                }

                return;
            case PdfKind.Name:
                s.Name(value.AsName!.Bytes);
                return;
            case PdfKind.String:
                String(s, value.AsString!.Bytes);
                return;
            case PdfKind.Array:
                s.Byte((byte)'[');
                bool regular = false;
                foreach (var item in value.AsArray!.Items)
                {
                    if (regular && StartsRegular(item))
                    {
                        s.Space();
                    }

                    Value(s, source, item, depth);
                    regular = EndsRegular(item);
                }

                s.Byte((byte)']');
                return;
            default:
                var dictionary = value.AsDictionary!;
                s.Raw("<<"u8);
                for (int i = 0; i < dictionary.Count; i++)
                {
                    Entry(s, source, dictionary.KeyAt(i).Bytes, dictionary.ValueAt(i), depth);
                }

                s.Raw(">>"u8);
                return;
        }
    }

    /// <summary>Writes one dictionary entry: the key, a space when the value needs one, the value (as <see cref="Value"/>).</summary>
    public void Entry(PdfSyntax s, Source source, ReadOnlySpan<byte> key, PdfValue value, int depth = 0)
    {
        s.Name(key);
        if (StartsRegular(value))
        {
            s.Space();
        }

        Value(s, source, value, depth);
    }

    /// <summary>
    /// Writes object <paramref name="number"/> as <paramref name="body"/> writes it: first into a buffer of its own, so
    /// that what it refers to is written before it; then where the file puts it (an object stream, or on its own, its
    /// strings encrypted: <paramref name="body"/> is then called again, every reference it names already numbered).
    /// </summary>
    public void WriteObject(int number, Action<PdfSyntax> body)
    {
        var scratch = Rent();
        try
        {
            body(scratch);
            var target = _file.BeginSmall(number);
            if (target.Encrypts)
            {
                body(target);
            }
            else
            {
                target.Raw(scratch.Span);
            }

            _file.EndSmall();
        }
        finally
        {
            Return(scratch);
        }
    }

    /// <summary>Writes the objects numbered past the deepest walk and not written yet.</summary>
    public void Drain()
    {
        while (_later.TryDequeue(out var item))
        {
            Copy(item.Source, item.Number, item.Source.Numbers[item.Number], 0);
        }
    }

    /// <summary>Drains, then writes a null object for each page a reference numbered and that never came (a builder's).</summary>
    public void Finish()
    {
        Drain();
        foreach (var source in _sources.Values)
        {
            foreach (var (page, number) in source.PageNumbers)
            {
                if (source.Reserve && !source.PagesWritten.Contains(page))
                {
                    _file.BeginSmall(number).Raw("null"u8);
                    _file.EndSmall();
                }
            }
        }
    }

    public void Dispose()
    {
        while (_free.TryPop(out var scratch))
        {
            scratch.Dispose();
        }

        while (_freeBuffers.TryPop(out var buffer))
        {
            buffer.Dispose();
        }
        _hash.Dispose();
    }

    // Copies object `number` of `source` (numbered `preset` already, or Pending): what it refers to first, then itself,
    // unless an object of the same content is written already. Its result number.
    private int Copy(Source source, int number, int preset, int depth)
    {
        source.Numbers[number] = preset;
        if (source.Replaced.TryGetValue(number, out var body))
        {
            // An object an edit writes anew (a form field filled, a widget's appearance changed): numbered first, so what
            // it refers to that comes back to it finds its number.
            int replaced = preset != Pending ? preset : _file.Allocate();
            source.Numbers[number] = replaced;
            WriteObject(replaced, body);
            return replaced;
        }

        if (source.Dropped.Contains(number))
        {
            return Settle(source, number, Null);                              // an object an edit removes (a flattened field): null where still named
        }

        PdfValue value;
        try
        {
            value = source.File.Get(number);
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            value = PdfValue.Null;                                            // a damaged object reads as null, as a reader takes it
        }

        var dictionary = value.AsDictionary;
        if (value.Kind == PdfKind.Dictionary && dictionary!["Type"u8].IsName("Pages"u8) && Root > 0)
        {
            return Settle(source, number, Root);                              // a node of the source's page tree: the result's
        }

        if (value.IsNull || (value.Kind == PdfKind.Dictionary && dictionary!["Type"u8].IsName("Page"u8)))
        {
            return Settle(source, number, Null);                              // nothing, or a page the source's tree does not hold
        }

        return value.AsStream is { } stream ? CopyStream(source, number, stream, depth) : CopyObject(source, number, value, depth);
    }

    // An object that is not copied: what it becomes (a null object, or one naming `target`, when it was numbered already).
    private int Settle(Source source, int number, int target)
    {
        int numbered = source.Numbers[number];
        if (numbered == Pending)
        {
            source.Numbers[number] = target;
            return target;
        }

        var s = _file.BeginSmall(numbered);
        if (target > 0)
        {
            s.Reference(target);
        }
        else
        {
            s.Raw("null"u8);
        }

        _file.EndSmall();
        return numbered;
    }

    private int CopyObject(Source source, int number, PdfValue value, int depth)
    {
        var scratch = Rent();
        try
        {
            Value(scratch, source, value, depth + 1);
            int numbered = source.Numbers[number];
            bool once = numbered == Pending && Dedupable(value);
            ContentKey key = default;
            if (once)
            {
                key = Key((byte)'O', scratch.Span, []);
                if (_written.TryGetValue(key, out int same))
                {
                    source.Numbers[number] = same;
                    return same;
                }
            }

            int result = numbered != Pending ? numbered : _file.Allocate();
            source.Numbers[number] = result;
            if (once)
            {
                _written[key] = result;
            }

            var target = _file.BeginSmall(result);
            if (target.Encrypts)
            {
                Value(target, source, value, depth + 1);
            }
            else
            {
                target.Raw(scratch.Span);
            }

            _file.EndSmall();
            return result;
        }
        finally
        {
            Return(scratch);
        }
    }

    private int CopyStream(Source source, int number, PdfStream stream, int depth)
    {
        var file = source.File;
        if (stream.Length > file.MaxStreamBytes)
        {
            throw PdfLimit.Error(
                $"{file.Name}: object {number}: a stream of {stream.Length:N0} stored bytes, more than the limit {nameof(DocumentOptions.MaxStreamBytes)} allows (DocumentOptions.MaxStreamBytes, or {DocumentOptions.MaxStreamBytesVariable}).");
        }

        var filters = file.Filters(stream.Dictionary);
        int declared = filters.Length;
        var window = PdfWindow.Read(file.Source, stream.Offset, stream.Length);
        var scratch = Rent();
        PdfBuffer? recodedData = null;                                       // its own: what this stream refers to is copied while it is held
        try
        {
            if (file.Security is { } security)
            {
                window = security.Decrypt(stream, window, ref filters);       // the data as the source's reader sees it, still encoded
            }

            bool refiltered = filters.Length != declared;                     // a crypt filter of its own taken off
            var data = window.Span;
            if (Recompress && !stream.Dictionary.Contains("F"u8) && Recoded(file, number, filters, window, recodedData = RentBuffer()))
            {
                data = recodedData!.Span;
                refiltered = true;
                filters = [new PdfFilter(PdfName.From("FlateDecode"u8), null)];
            }

            Entries(scratch, source, stream.Dictionary, refiltered ? filters : null, depth + 1);
            int numbered = source.Numbers[number];
            bool once = numbered == Pending;
            ContentKey key = default;
            if (once)
            {
                key = Key((byte)'S', scratch.Span, data);
                if (_written.TryGetValue(key, out int same))
                {
                    source.Numbers[number] = same;
                    return same;
                }
            }

            int result = numbered != Pending ? numbered : _file.Allocate();
            source.Numbers[number] = result;
            if (once)
            {
                _written[key] = result;
            }

            var target = _file.BeginStream(result);
            if (target.Encrypts)
            {
                Entries(target, source, stream.Dictionary, refiltered ? filters : null, depth + 1);
            }
            else
            {
                target.Raw(scratch.Span);
            }

            var type = stream.Dictionary["Type"u8];
            _file.EndStream(data, deflate: false, type.IsName("Metadata"u8) ? PdfCryptUse.Metadata : type.IsName("EmbeddedFile"u8) ? PdfCryptUse.EmbeddedFile : PdfCryptUse.Stream);
            return result;
        }
        finally
        {
            Return(scratch);
            window.Dispose();
            if (recodedData is not null)
            {
                _freeBuffers.Push(recodedData);
            }
        }
    }

    // A stream's dictionary entries but /Length (the file writes it); /Filter and /DecodeParms from `filters` when the data
    // changed form (deflated again, or a crypt filter of its own taken off).
    private void Entries(PdfSyntax s, Source source, PdfDictionary dictionary, PdfFilter[]? filters, int depth)
    {
        for (int i = 0; i < dictionary.Count; i++)
        {
            var key = dictionary.KeyAt(i);
            if (key.Is("Length"u8) || (filters is not null && (key.Is("Filter"u8) || key.Is("DecodeParms"u8))))
            {
                continue;
            }

            Entry(s, source, key.Bytes, dictionary.ValueAt(i), depth);
        }

        if (filters is not { Length: > 0 })
        {
            return;
        }

        s.Raw("/Filter["u8);
        foreach (var filter in filters)
        {
            s.Name(filter.Name.Bytes);
        }

        s.Byte((byte)']');
        if (filters.Any(f => f.Parameters is not null))
        {
            s.Raw("/DecodeParms["u8);
            for (int i = 0; i < filters.Length; i++)
            {
                if (i > 0)
                {
                    s.Space();
                }

                if (filters[i].Parameters is { } parameters)
                {
                    Value(s, source, PdfValue.Of(parameters), depth);
                }
                else
                {
                    s.Raw("null"u8);
                }
            }

            s.Byte((byte)']');
        }
    }

    // Whether the stream's data deflated at the options' level (into `output`) is smaller than as stored, when it is stored
    // plain or deflated (with no predictor) at a faster level than that; else it is copied as stored.
    private bool Recoded(PdfFile file, int number, PdfFilter[] filters, PdfWindow stored, PdfBuffer output)
    {
        var data = stored.Span;
        if (_level == 0 || data.IsEmpty)
        {
            return false;
        }

        if (filters.Length == 0)
        {
            PdfDeflate.Compress(data, _level, output);
        }
        else
        {
            bool plainFlate = filters.Length == 1 && filters[0].Name.Is("FlateDecode"u8)
                              && (filters[0].Parameters is null || !file.Resolve(filters[0].Parameters!["Predictor"u8]).TryInteger(out long predictor) || predictor <= 1);
            if (!plainFlate || data.Length < 2 || (data[0] & 0x0F) != 8 || ((data[0] << 8) | data[1]) % 31 != 0 || data[1] >> 6 >= LevelClass(_level))
            {
                return false;                                                 // not plain zlib, or deflated at this level or harder
            }

            using var decoded = PdfFilters.Decode(stored.Memory, null, filters, file.MaxStreamBytes,
                string.Create(CultureInfo.InvariantCulture, $"{file.Name}: object {number}: a stream decodes to more than the limit {nameof(DocumentOptions.MaxStreamBytes)} allows (DocumentOptions.MaxStreamBytes, or {DocumentOptions.MaxStreamBytesVariable})."));
            if (decoded.Damaged)
            {
                return false;                                                 // kept as it is: what it holds past the damage too
            }

            PdfDeflate.Compress(decoded.Span, _level, output);
        }

        return output.Length < data.Length;
    }

    // zlib's level class (its header's FLEVEL): 0 for level 1, 1 for 2-5, 2 for 6, 3 for 7-9.
    private static int LevelClass(int level) => level switch { <= 1 => 0, <= 5 => 1, 6 => 2, _ => 3 };

    // Whether an object is written once whatever holds it: data with no identity of its own (numbers, names, strings,
    // arrays, streams, fonts, their descriptors and encodings, graphics states). Pages, annotations, fields, bookmarks and
    // structure elements are each their own.
    private static bool Dedupable(PdfValue value)
    {
        if (value.Kind != PdfKind.Dictionary)
        {
            return true;
        }

        var type = value.AsDictionary!["Type"u8];
        return type.IsName("Font"u8) || type.IsName("FontDescriptor"u8) || type.IsName("ExtGState"u8) || type.IsName("Encoding"u8);
    }

    // The SHA-256 of an object's kind, its syntax as the result writes it and its data.
    private ContentKey Key(byte kind, ReadOnlySpan<byte> syntax, ReadOnlySpan<byte> data)
    {
        Span<byte> length = stackalloc byte[9];
        length[0] = kind;
        BinaryPrimitives.WriteInt64BigEndian(length[1..], syntax.Length);
        _hash.AppendData(length);
        _hash.AppendData(syntax);
        _hash.AppendData(data);
        Span<byte> digest = stackalloc byte[32];
        _hash.GetHashAndReset(digest);
        return new ContentKey(BinaryPrimitives.ReadUInt64LittleEndian(digest), BinaryPrimitives.ReadUInt64LittleEndian(digest[8..]),
            BinaryPrimitives.ReadUInt64LittleEndian(digest[16..]), BinaryPrimitives.ReadUInt64LittleEndian(digest[24..]));
    }

    // A number as the shortest text that reads back as the same double, never with an exponent.
    private static void Real(PdfSyntax s, double value)
    {
        if (!double.IsFinite(value) || value == 0)
        {
            s.Byte((byte)'0');
            return;
        }

        Span<byte> text = stackalloc byte[352];
        value.TryFormat(text, out int written, "R", CultureInfo.InvariantCulture);
        if (text[..written].IndexOfAny((byte)'E', (byte)'e') >= 0)
        {
            value.TryFormat(text, out written, "0.##########", CultureInfo.InvariantCulture);
            if (text[..written].SequenceEqual("-0"u8))
            {
                (text[0], written) = ((byte)'0', 1);
            }
        }

        s.Raw(text[..written]);
    }

    // A string as a literal, or in hex when its control bytes would make the literal longer.
    private static void String(PdfSyntax s, ReadOnlySpan<byte> bytes)
    {
        int control = 0;
        foreach (byte b in bytes)
        {
            if (b is < 0x20 or 0x7F)
            {
                control++;
            }
        }

        if (control * 3 > bytes.Length)
        {
            s.HexString(bytes);
        }
        else
        {
            s.Literal(bytes);
        }
    }

    // Whether a value's syntax starts or ends with a regular character (a space must separate two such tokens).
    private static bool StartsRegular(PdfValue value) => value.Kind is PdfKind.Null or PdfKind.Boolean or PdfKind.Integer or PdfKind.Real or PdfKind.Reference;

    private static bool EndsRegular(PdfValue value) => value.Kind is PdfKind.Null or PdfKind.Boolean or PdfKind.Integer or PdfKind.Real or PdfKind.Reference or PdfKind.Name;

    private PdfSyntax Rent()
    {
        if (_free.TryPop(out var scratch))
        {
            scratch.Clear();
            return scratch;
        }

        return new PdfSyntax(1024);
    }

    private void Return(PdfSyntax scratch) => _free.Push(scratch);

    private PdfBuffer RentBuffer() =>
        _freeBuffers.TryPop(out var buffer) ? buffer : new PdfBuffer(64 * 1024, Array.MaxLength, "a stream deflated again passes the largest array .NET holds");

    /// <summary>A document objects are copied from: how its references are made the result's, and what was copied.</summary>
    public sealed class Source(PdfDocument document, bool identity, bool reserve, HashSet<int> pages)
    {
        public PdfDocument Document { get; } = document;

        public PdfFile File { get; } = document.File;

        /// <summary>Whether its objects are referred to by their own numbers (the document an update appends to).</summary>
        public bool Identity { get; } = identity;

        /// <summary>Whether a page referred to before it is copied is numbered then (a builder still adding pages).</summary>
        public bool Reserve { get; } = reserve;

        /// <summary>Its page objects' numbers.</summary>
        public HashSet<int> Pages { get; } = pages;

        /// <summary>Each object copied: its number in the result (Pending while it is written and has none, -1 for null).</summary>
        public Dictionary<int, int> Numbers { get; } = [];

        /// <summary>Each page object's number in the result.</summary>
        public Dictionary<int, int> PageNumbers { get; } = [];

        /// <summary>The page objects written (a builder's).</summary>
        public HashSet<int> PagesWritten { get; } = [];

        /// <summary>Objects an edit writes anew instead of copying them (by number): each one's body, references made the result's through the copier.</summary>
        public Dictionary<int, Action<PdfSyntax>> Replaced { get; } = [];

        /// <summary>Objects an edit removes (by number): null where something still names them.</summary>
        public HashSet<int> Dropped { get; } = [];
    }

    // A SHA-256, as a key.
    private readonly record struct ContentKey(ulong A, ulong B, ulong C, ulong D);
}
