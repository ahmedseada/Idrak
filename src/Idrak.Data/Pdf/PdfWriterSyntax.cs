// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Text;

namespace Idrak.Data.Pdf;

/// <summary>
/// PDF syntax written into a pooled buffer that grows (<see cref="PdfBuffer"/>, returned on <see cref="Dispose"/>): numbers
/// invariant and without exponents (a float as the shortest text that reads back as it, a computed value to a fixed number
/// of decimals), names with <c>#xx</c> escapes, literal strings escaped, hex strings, text strings (ASCII as a literal,
/// anything else as UTF-16BE with its mark), references, and any value of the reader's object model
/// (<see cref="PdfValue"/>), so what is read can be written back. Nothing here allocates but the buffer's growth.
/// </summary>
internal sealed class PdfSyntax : IDisposable
{
    private static readonly string[] Fixed = [.. Enumerable.Range(0, 10).Select(d => "F" + d.ToString(CultureInfo.InvariantCulture))];

    private PdfBuffer _buffer;
    private PdfSecurity? _cipher;
    private int _cipherNumber, _cipherGeneration;

    public PdfSyntax(int capacity = 4096) =>
        _buffer = new PdfBuffer(capacity, Array.MaxLength, "a part of a PDF being written passes the largest array .NET holds");

    public int Length => _buffer.Length;

    /// <summary>Whether strings are written encrypted (see <see cref="Encrypting"/>).</summary>
    public bool Encrypts => _cipher is not null;

    /// <summary>
    /// Encrypts every string written from now on with <paramref name="security"/>'s key for object
    /// <paramref name="number"/> and <paramref name="generation"/> (an object of an encrypted file that stands on its own:
    /// its strings written as hex digits of their ciphertext), or, with null, writes them as they are (an object stream's
    /// objects, which the stream's own encryption covers).
    /// </summary>
    public void Encrypting(PdfSecurity? security, int number, int generation) => (_cipher, _cipherNumber, _cipherGeneration) = (security, number, generation);

    public ReadOnlySpan<byte> Span => _buffer.Span;

    /// <summary>The buffer itself (a stream deflated into it, a predictor's rows written in place).</summary>
    public PdfBuffer Buffer => _buffer;

    public void Clear() => _buffer.Clear();

    public void Dispose() => _buffer.Dispose();

    public PdfSyntax Raw(ReadOnlySpan<byte> bytes)
    {
        _buffer.Write(bytes);
        return this;
    }

    public PdfSyntax Byte(byte value)
    {
        _buffer.Reserve(1)[0] = value;
        _buffer.Advance(1);
        return this;
    }

    public PdfSyntax Space() => Byte((byte)' ');

    public PdfSyntax Line() => Byte((byte)'\n');

    public PdfSyntax Integer(long value)
    {
        var span = _buffer.GetSpan(24);
        value.TryFormat(span, out int written, default, CultureInfo.InvariantCulture);
        _buffer.Advance(written);
        return this;
    }

    /// <summary>A float as the shortest text that reads back as it (a value too small or too large for that without an exponent, to ten decimals).</summary>
    /// <exception cref="ArgumentException">The value is not finite.</exception>
    public PdfSyntax Number(float value)
    {
        if (!float.IsFinite(value))
        {
            throw new ArgumentException($"A PDF number is finite, not {value}.", nameof(value));
        }

        if (value == 0f)
        {
            return Byte((byte)'0');                                            // also -0
        }

        var span = _buffer.GetSpan(64);
        value.TryFormat(span, out int written, default, CultureInfo.InvariantCulture);
        if (span[..written].IndexOfAny((byte)'E', (byte)'e') >= 0)
        {
            // PDF numbers have no exponent: a tiny or huge value in plain digits.
            ((double)value).TryFormat(span, out written, "0.##########", CultureInfo.InvariantCulture);
            if (span[..written].SequenceEqual("-0"u8))
            {
                (span[0], written) = ((byte)'0', 1);
            }
        }

        _buffer.Advance(written);
        return this;
    }

    /// <summary>A computed value to <paramref name="decimals"/> decimals at most (trailing zeros left out).</summary>
    /// <exception cref="ArgumentException">The value is not finite.</exception>
    public PdfSyntax Number(double value, int decimals)
    {
        if (!double.IsFinite(value))
        {
            throw new ArgumentException($"A PDF number is finite, not {value}.", nameof(value));
        }

        var span = _buffer.GetSpan(352);
        value.TryFormat(span, out int written, Fixed[decimals], CultureInfo.InvariantCulture);
        written = Trimmed(span, written);
        _buffer.Advance(written);
        return this;
    }

    /// <summary>The value a reader parses from <see cref="Number(double, int)"/>'s text.</summary>
    public static double Written(double value, int decimals)
    {
        Span<byte> span = stackalloc byte[352];
        value.TryFormat(span, out int written, Fixed[decimals], CultureInfo.InvariantCulture);
        written = Trimmed(span, written);
        return double.Parse(span[..written], NumberStyles.Float, CultureInfo.InvariantCulture);
    }

    /// <summary>The value a reader parses from <see cref="Number(float)"/>'s text.</summary>
    public static double Written(float value)
    {
        if (value == 0f)
        {
            return 0;
        }

        Span<byte> span = stackalloc byte[64];
        value.TryFormat(span, out int written, default, CultureInfo.InvariantCulture);
        if (span[..written].IndexOfAny((byte)'E', (byte)'e') >= 0)
        {
            ((double)value).TryFormat(span, out written, "0.##########", CultureInfo.InvariantCulture);
        }

        return double.Parse(span[..written], NumberStyles.Float, CultureInfo.InvariantCulture);
    }

    // Trailing zeros (and a trailing point) dropped; "-0" written as "0".
    private static int Trimmed(Span<byte> span, int written)
    {
        if (span[..written].IndexOf((byte)'.') >= 0)
        {
            while (span[written - 1] == (byte)'0')
            {
                written--;
            }

            if (span[written - 1] == (byte)'.')
            {
                written--;
            }
        }

        if (written == 2 && span[0] == (byte)'-' && span[1] == (byte)'0')
        {
            (span[0], written) = ((byte)'0', 1);
        }

        return written;
    }

    /// <summary>A name: '/' and its bytes, each one outside '!' to '~', a delimiter or '#' written as <c>#xx</c>.</summary>
    public PdfSyntax Name(ReadOnlySpan<byte> name)
    {
        var span = _buffer.GetSpan(1 + 3 * name.Length);
        int at = 0;
        span[at++] = (byte)'/';
        foreach (byte b in name)
        {
            if (b is < 0x21 or > 0x7E or (byte)'#' or (byte)'(' or (byte)')' or (byte)'<' or (byte)'>' or (byte)'[' or (byte)']' or (byte)'{' or (byte)'}'
                or (byte)'/' or (byte)'%')
            {
                span[at++] = (byte)'#';
                span[at++] = Hex(b >> 4);
                span[at++] = Hex(b & 0xF);
            }
            else
            {
                span[at++] = b;
            }
        }

        _buffer.Advance(at);
        return this;
    }

    /// <summary>A name from text (its UTF-8 bytes, escaped).</summary>
    public PdfSyntax Name(string name)
    {
        int count = Encoding.UTF8.GetByteCount(name);
        byte[]? rented = count > 256 ? ArrayPool<byte>.Shared.Rent(count) : null;
        try
        {
            Span<byte> bytes = rented is not null ? rented : stackalloc byte[256];
            int written = Encoding.UTF8.GetBytes(name, bytes);
            return Name(bytes[..written]);
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<byte>.Shared.Return(rented);
            }
        }
    }

    /// <summary>A literal string: its bytes between parentheses, the parentheses and backslash escaped, line ends and other control bytes as escapes.</summary>
    public PdfSyntax Literal(ReadOnlySpan<byte> bytes)
    {
        if (_cipher is not null)
        {
            return Encrypted(bytes);
        }

        var span = _buffer.GetSpan(2 + 4 * bytes.Length);
        int at = 0;
        span[at++] = (byte)'(';
        foreach (byte b in bytes)
        {
            switch (b)
            {
                case (byte)'(' or (byte)')' or (byte)'\\':
                    span[at++] = (byte)'\\';
                    span[at++] = b;
                    break;
                case (byte)'\n':
                    span[at++] = (byte)'\\';
                    span[at++] = (byte)'n';
                    break;
                case (byte)'\r':
                    span[at++] = (byte)'\\';
                    span[at++] = (byte)'r';
                    break;
                case < 0x20 or 0x7F:
                    span[at++] = (byte)'\\';
                    span[at++] = (byte)('0' + (b >> 6));
                    span[at++] = (byte)('0' + ((b >> 3) & 7));
                    span[at++] = (byte)('0' + (b & 7));
                    break;
                default:
                    span[at++] = b;
                    break;
            }
        }

        span[at++] = (byte)')';
        _buffer.Advance(at);
        return this;
    }

    /// <summary>A hex string: two digits a byte between angle brackets.</summary>
    public PdfSyntax HexString(ReadOnlySpan<byte> bytes) => _cipher is not null ? Encrypted(bytes) : Hex(bytes);

    private PdfSyntax Hex(ReadOnlySpan<byte> bytes)
    {
        var span = _buffer.GetSpan(2 + 2 * bytes.Length);
        span[0] = (byte)'<';
        int at = 1;
        foreach (byte b in bytes)
        {
            span[at++] = Hex(b >> 4);
            span[at++] = Hex(b & 0xF);
        }

        span[at++] = (byte)'>';
        _buffer.Advance(at);
        return this;
    }

    /// <summary>Hex digits of <paramref name="value"/>'s two bytes, big-endian, with no brackets (a glyph id in a hex string).</summary>
    public PdfSyntax HexPair(int value)
    {
        var span = _buffer.Reserve(4);
        span[0] = Hex((value >> 12) & 0xF);
        span[1] = Hex((value >> 8) & 0xF);
        span[2] = Hex((value >> 4) & 0xF);
        span[3] = Hex(value & 0xF);
        _buffer.Advance(4);
        return this;
    }

    /// <summary>
    /// A text string (ISO 32000-2, 7.9.2.2): text of printable ASCII alone as a literal (those bytes are the same in
    /// PDFDocEncoding), any other as UTF-16BE after its byte order mark, in hex.
    /// </summary>
    public PdfSyntax Text(string text)
    {
        if (_cipher is not null)
        {
            return EncryptedText(text);
        }

        if (Ascii(text))
        {
            var span = _buffer.GetSpan(2 + 2 * text.Length);
            int at = 0;
            span[at++] = (byte)'(';
            foreach (char c in text)
            {
                if (c is '(' or ')' or '\\')
                {
                    span[at++] = (byte)'\\';
                }

                span[at++] = (byte)c;
            }

            span[at++] = (byte)')';
            _buffer.Advance(at);
            return this;
        }

        var hex = _buffer.GetSpan(2 + 4 + 4 * text.Length);
        int h = 0;
        hex[h++] = (byte)'<';
        hex[h++] = (byte)'F';
        hex[h++] = (byte)'E';
        hex[h++] = (byte)'F';
        hex[h++] = (byte)'F';
        foreach (char c in text)
        {
            hex[h++] = Hex((c >> 12) & 0xF);
            hex[h++] = Hex((c >> 8) & 0xF);
            hex[h++] = Hex((c >> 4) & 0xF);
            hex[h++] = Hex(c & 0xF);
        }

        hex[h++] = (byte)'>';
        _buffer.Advance(h);
        return this;
    }

    /// <summary>A reference: "N 0 R".</summary>
    public PdfSyntax Reference(int number) => Integer(number).Raw(" 0 R"u8);

    /// <summary>A value of the reader's object model, written back as syntax (a stream's dictionary alone: its data is the caller's).</summary>
    public PdfSyntax Value(PdfValue value)
    {
        switch (value.Kind)
        {
            case PdfKind.Null:
                return Raw("null"u8);
            case PdfKind.Boolean:
                return Raw(value.AsBoolean ? "true"u8 : "false"u8);
            case PdfKind.Integer:
                value.TryInteger(out long integer);
                return Integer(integer);
            case PdfKind.Real:
                value.TryNumber(out double real);
                return Number(real, 6);
            case PdfKind.Reference:
                var reference = value.AsReference;
                return Integer(reference.Number).Space().Integer(reference.Generation).Raw(" R"u8);
            case PdfKind.Name:
                return Name(value.AsName!.Bytes);
            case PdfKind.String:
                return Literal(value.AsString!.Bytes);
            case PdfKind.Array:
                Byte((byte)'[');
                var items = value.AsArray!.Items;
                for (int i = 0; i < items.Length; i++)
                {
                    if (i > 0)
                    {
                        Space();
                    }

                    Value(items[i]);
                }

                return Byte((byte)']');
            default:
                var dictionary = value.AsDictionary!;
                Raw("<<"u8);
                for (int i = 0; i < dictionary.Count; i++)
                {
                    Name(dictionary.KeyAt(i).Bytes).Space().Value(dictionary.ValueAt(i));
                    if (i + 1 < dictionary.Count)
                    {
                        Space();
                    }
                }

                return Raw(">>"u8);
        }
    }

    // A string encrypted for the object being written, as hex digits.
    private PdfSyntax Encrypted(ReadOnlySpan<byte> plain)
    {
        byte[] encrypted = ArrayPool<byte>.Shared.Rent(PdfSecurity.MostEncryptedLength(plain.Length));
        try
        {
            int length = _cipher!.EncryptString(plain, _cipherNumber, _cipherGeneration, encrypted);
            return Hex(encrypted.AsSpan(0, length));
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(encrypted);
        }
    }

    // A text string's bytes (ASCII, or UTF-16BE after its mark, as Text writes them) encrypted.
    private PdfSyntax EncryptedText(string text)
    {
        bool ascii = Ascii(text);
        int length = ascii ? text.Length : 2 + 2 * text.Length;
        byte[] bytes = ArrayPool<byte>.Shared.Rent(Math.Max(1, length));
        try
        {
            if (ascii)
            {
                for (int i = 0; i < text.Length; i++)
                {
                    bytes[i] = (byte)text[i];
                }
            }
            else
            {
                (bytes[0], bytes[1]) = (0xFE, 0xFF);
                Encoding.BigEndianUnicode.GetBytes(text, bytes.AsSpan(2));
            }

            return Encrypted(bytes.AsSpan(0, length));
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(bytes);
        }
    }

    private static bool Ascii(string text)
    {
        foreach (char c in text)
        {
            if (c is < ' ' or > '~')
            {
                return false;
            }
        }

        return true;
    }

    private static byte Hex(int digit) => (byte)(digit < 10 ? '0' + digit : 'A' + digit - 10);
}

/// <summary>A write-only stream into a <see cref="PdfBuffer"/> (what deflate writes its output to).</summary>
internal sealed class PdfBufferStream(PdfBuffer buffer) : Stream
{
    public override bool CanRead => false;

    public override bool CanSeek => false;

    public override bool CanWrite => true;

    public override long Length => buffer.Length;

    public override long Position
    {
        get => buffer.Length;
        set => throw new NotSupportedException("A buffer is written in order.");
    }

    public override void Write(byte[] array, int offset, int count) => buffer.Write(array.AsSpan(offset, count));

    public override void Write(ReadOnlySpan<byte> bytes) => buffer.Write(bytes);

    public override void Flush()
    {
    }

    public override int Read(byte[] array, int offset, int count) => throw new NotSupportedException("A buffer is written, not read.");

    public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException("A buffer is written in order.");

    public override void SetLength(long value) => throw new NotSupportedException("A buffer is written in order.");
}

/// <summary>Flate (zlib, through <see cref="ZLibStream"/>) at a level from 1 (fastest) to 9 (smallest).</summary>
internal static class PdfDeflate
{
    /// <summary><paramref name="data"/> deflated into <paramref name="output"/> (cleared first).</summary>
    public static void Compress(ReadOnlySpan<byte> data, int level, PdfBuffer output)
    {
        output.Clear();
        using var sink = new PdfBufferStream(output);
        using var zlib = new ZLibStream(sink, new ZLibCompressionOptions { CompressionLevel = level }, leaveOpen: true);
        zlib.Write(data);
    }

    /// <summary><paramref name="source"/> read to its end and deflated into <paramref name="output"/> (cleared first); the bytes read.</summary>
    public static long Compress(Stream source, int level, PdfBuffer output)
    {
        output.Clear();
        byte[] chunk = ArrayPool<byte>.Shared.Rent(64 * 1024);
        long total = 0;
        try
        {
            using var sink = new PdfBufferStream(output);
            using var zlib = level > 0 ? new ZLibStream(sink, new ZLibCompressionOptions { CompressionLevel = level }, leaveOpen: true) : null;
            int read;
            while ((read = source.Read(chunk, 0, chunk.Length)) > 0)
            {
                total += read;
                if (zlib is not null)
                {
                    zlib.Write(chunk, 0, read);
                }
                else
                {
                    output.Write(chunk.AsSpan(0, read));
                }
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(chunk);
        }

        return total;
    }
}

/// <summary>
/// The file being written: the header, then objects as they are given (each one's offset kept, eight bytes an object, for
/// the cross-reference), small objects packed into compressed object streams when the version takes them, and at the end
/// the cross-reference (a stream, or a classic table for versions before 1.5) and the trailer. Bytes go to the destination
/// in blocks as they are made (the destination need not seek); nothing else is kept.
/// </summary>
internal sealed partial class PdfFileWriter : IDisposable
{
    private const int Block = 64 * 1024, MaxPacked = 100;
    private const long NotWritten = long.MinValue;

    private readonly Stream _destination;
    private readonly int _level;
    private readonly PdfSyntax _pending = new(Block + 4096);
    private PdfSyntax? _packed, _packedData, _scratch, _crypt;
    private readonly List<(int Number, int Offset)> _packedObjects = [];
    private long[] _entries = new long[64];
    private int _count = 1;
    private long _written;

    // Objects written again with a generation other than 0 (an update's), what offsets are counted from (an update of a
    // file with bytes before its header), whether the small object begun was packed, and the object being written.
    private Dictionary<int, int>? _generations;
    private long _shift;
    private bool _smallPacked;
    private int _currentNumber, _currentGeneration, _xrefNumber;
    private bool _currentEncrypted;

    /// <param name="destination">Where the bytes go (written in order).</param>
    /// <param name="version">The header's version ("1.7").</param>
    /// <param name="level">The Flate level of the streams this writes (0: stored).</param>
    /// <param name="packing">Whether small objects go into object streams.</param>
    public PdfFileWriter(Stream destination, string version, int level, bool packing)
    {
        (_destination, _level) = (destination, level);
        _entries[0] = NotWritten;
        _pending.Raw("%PDF-"u8).Raw(Encoding.ASCII.GetBytes(version)).Raw([(byte)'\n', (byte)'%', 0xE2, 0xE3, 0xCF, 0xD3, (byte)'\n']);  // a comment of high bytes: a binary file
        _packed = packing ? new PdfSyntax(Block + 4096) : null;
    }

    /// <summary>Where the next byte goes, from the file's start.</summary>
    public long Position => _written + _pending.Length;

    /// <summary>The Flate level of the streams this writes (0: stored).</summary>
    public int Level => _level;

    /// <summary>
    /// The security handler every string and stream is encrypted by from now on (null: none), but for
    /// <see cref="PlainNumber"/>'s and the cross-reference stream's.
    /// </summary>
    public PdfSecurity? Security { get; set; }

    /// <summary>The object written without encryption (the encryption dictionary; 0 for none).</summary>
    public int PlainNumber { get; set; }

    /// <summary>The encryption dictionary the trailer names (0: none).</summary>
    public int EncryptNumber { get; set; }

    /// <summary>A new object's number (it must be written before the end).</summary>
    public int Allocate()
    {
        if (_count == _entries.Length)
        {
            Array.Resize(ref _entries, _entries.Length * 2);
        }

        _entries[_count] = NotWritten;
        return _count++;
    }

    /// <summary>
    /// Starts object <paramref name="number"/> in the file ("N G obj", its generation 0 unless an update writes it again
    /// with another); the caller writes its value, then <see cref="EndObject"/>. In an encrypted file its strings are
    /// encrypted as they are written.
    /// </summary>
    public PdfSyntax BeginObject(int number)
    {
        _entries[number] = Position - _shift;
        int generation = Generation(number);
        _currentEncrypted = Security is not null && number != PlainNumber && number != _xrefNumber;
        (_currentNumber, _currentGeneration) = (number, generation);
        if (_currentEncrypted)
        {
            _pending.Encrypting(Security, number, generation);
        }

        return _pending.Integer(number).Space().Integer(generation).Raw(" obj\n"u8);
    }

    /// <summary>The generation object <paramref name="number"/> is written with.</summary>
    public int Generation(int number) => _generations is not null && _generations.TryGetValue(number, out int generation) ? generation : 0;

    public void EndObject()
    {
        _pending.Encrypting(null, 0, 0);
        _currentEncrypted = false;
        _pending.Raw("\nendobj\n"u8);
        if (_pending.Length >= Block)
        {
            Flush();
        }
    }

    /// <summary>
    /// Starts a small object (no stream): packed into the current object stream when the file packs them, else written in
    /// the file. The caller writes its value, then <see cref="EndSmall"/>.
    /// </summary>
    public PdfSyntax BeginSmall(int number)
    {
        // An object stream holds objects of generation 0 alone, and never the encryption dictionary (7.5.7).
        _smallPacked = _packed is not null && number != PlainNumber && Generation(number) == 0;
        if (!_smallPacked)
        {
            return BeginObject(number);
        }

        _packedObjects.Add((number, _packed!.Length));
        return _packed;
    }

    public void EndSmall()
    {
        if (!_smallPacked)
        {
            EndObject();
            return;
        }

        _packed!.Line();
        if (_packedObjects.Count >= MaxPacked || _packed.Length >= Block)
        {
            FlushPacked();
        }
    }

    /// <summary>Starts a stream object's dictionary ("N 0 obj &lt;&lt;"); the caller writes its entries (no /Length), then <see cref="EndStream"/>.</summary>
    public PdfSyntax BeginStream(int number) => BeginObject(number).Raw("<<"u8);

    /// <summary>
    /// Ends a stream whose dictionary the caller wrote: deflated at the file's level when <paramref name="deflate"/> (its
    /// /Filter written here), else stored as given (the caller's own /Filter, or none); in an encrypted file encrypted by
    /// the crypt filter <paramref name="use"/> names; its /Length, then its data.
    /// </summary>
    public void EndStream(ReadOnlySpan<byte> data, bool deflate, PdfCryptUse use = PdfCryptUse.Stream)
    {
        if (deflate && _level > 0)
        {
            var scratch = _scratch ??= new PdfSyntax(Block);
            PdfDeflate.Compress(data, _level, scratch.Buffer);
            _pending.Raw("/Filter/FlateDecode"u8);
            WriteStreamData(Encrypted(scratch.Span, use));
            return;
        }

        WriteStreamData(Encrypted(data, use));
    }

    // A stream's data as the file stores it: encrypted for the object being written, when it is.
    private ReadOnlySpan<byte> Encrypted(ReadOnlySpan<byte> data, PdfCryptUse use)
    {
        if (!_currentEncrypted || Security is not { } security || security.StreamMethod(use) == PdfCryptMethod.Identity)
        {
            return data;
        }

        var crypt = _crypt ??= new PdfSyntax(Block);
        crypt.Clear();
        var room = crypt.Buffer.Reserve(PdfSecurity.MostEncryptedLength(data.Length));
        int length = security.EncryptStream(data, _currentNumber, _currentGeneration, use, room);
        crypt.Buffer.Advance(length);
        return crypt.Span;
    }

    private void WriteStreamData(ReadOnlySpan<byte> data)
    {
        _pending.Raw("/Length "u8).Integer(data.Length).Raw(">>\nstream\n"u8);
        if (data.Length < Block)
        {
            _pending.Raw(data);
        }
        else
        {
            Flush();
            _destination.Write(data);
            _written += data.Length;
        }

        _pending.Raw("\nendstream"u8);
        EndObject();
    }

    /// <summary>Writes the current object stream, if any objects wait in it.</summary>
    public void FlushPacked()
    {
        if (_packed is null || _packedObjects.Count == 0)
        {
            return;
        }

        int number = Allocate();
        var data = _packedData ??= new PdfSyntax(Block + 4096);
        data.Clear();
        foreach (var (n, offset) in _packedObjects)
        {
            data.Integer(n).Space().Integer(offset).Space();                   // the header: each object's number and offset
        }

        int first = data.Length;
        data.Raw(_packed.Span);
        for (int i = 0; i < _packedObjects.Count; i++)
        {
            _entries[_packedObjects[i].Number] = -((long)number * 65536 + i) - 1;
        }

        BeginStream(number).Raw("/Type/ObjStm/N "u8).Integer(_packedObjects.Count).Raw("/First "u8).Integer(first);
        EndStream(data.Span, deflate: true);
        _packed.Clear();
        _packedObjects.Clear();
    }

    /// <summary>Writes what is waiting to the destination.</summary>
    public void Flush()
    {
        if (_pending.Length > 0)
        {
            _destination.Write(_pending.Span);
            _written += _pending.Length;
            _pending.Clear();
        }
    }

    /// <summary>
    /// The cross-reference and the trailer (<c>/Root</c> <paramref name="root"/>, <c>/Info</c> <paramref name="info"/> when
    /// it is above 0, <c>/Encrypt</c> when <see cref="EncryptNumber"/> is, <c>/ID</c> when <paramref name="id"/> is not
    /// empty: <paramref name="id"/> and <paramref name="secondId"/>, or <paramref name="id"/> twice), <c>startxref</c> and
    /// <c>%%EOF</c>: a stream, or a classic table when <paramref name="classic"/>.
    /// </summary>
    /// <exception cref="InvalidOperationException">An object allocated was never written (a fault of the writer).</exception>
    public void Finish(int root, int info, ReadOnlySpan<byte> id, bool classic, ReadOnlySpan<byte> secondId = default)
    {
        FlushPacked();
        for (int n = 1; n < _count; n++)
        {
            if (_entries[n] == NotWritten)
            {
                throw new InvalidOperationException($"The PDF writer allocated object {n} and never wrote it.");
            }
        }

        if (classic)
        {
            long at = Position;
            _pending.Raw("xref\n0 "u8).Integer(_count).Raw("\n0000000000 65535 f\r\n"u8);
            Span<byte> entry = stackalloc byte[20];
            " 00000 n\r\n"u8.CopyTo(entry[10..]);
            for (int n = 1; n < _count; n++)
            {
                _entries[n].TryFormat(entry, out _, "D10", CultureInfo.InvariantCulture);
                _pending.Raw(entry);
                if (_pending.Length >= Block)
                {
                    Flush();
                }
            }

            _pending.Raw("trailer\n"u8);
            Trailer(_pending, root, info, id, secondId);
            _pending.Raw("\nstartxref\n"u8).Integer(at).Raw("\n%%EOF\n"u8);
            Flush();
            return;
        }

        int self = Allocate();
        long start = Position;
        _entries[self] = start;
        long largest = 0;
        for (int n = 1; n < _count; n++)
        {
            long e = _entries[n];
            largest = Math.Max(largest, e >= 0 ? e : (-e - 1) / 65536);
        }

        int width = 1;
        while (width < 8 && largest >> (8 * width) != 0)
        {
            width++;
        }

        int row = 1 + width + 2;
        using var rows = new PdfSyntax(row * _count);
        var data = rows.Buffer.Reserve(row * _count);
        data.Clear();
        data[row - 2] = 0xFF;                                                  // object 0: free, generation 65535
        data[row - 1] = 0xFF;
        Span<byte> field = stackalloc byte[8];
        for (int n = 1; n < _count; n++)
        {
            var r = data.Slice(n * row, row);
            long e = _entries[n];
            long second = e >= 0 ? e : (-e - 1) / 65536;
            int third = e >= 0 ? 0 : (int)((-e - 1) % 65536);
            r[0] = (byte)(e >= 0 ? 1 : 2);
            BinaryPrimitives.WriteInt64BigEndian(field, second);
            field[(8 - width)..].CopyTo(r[1..]);
            BinaryPrimitives.WriteUInt16BigEndian(r[(1 + width)..], (ushort)third);
        }

        rows.Buffer.Advance(row * _count);
        _xrefNumber = self;                                                    // a cross-reference stream is never encrypted
        var s = BeginStream(self);
        s.Raw("/Type/XRef/W[1 "u8).Integer(width).Raw(" 2]"u8);
        Trailer(s, root, info, id, secondId, inside: true);
        EndStream(rows.Span, deflate: true);
        _pending.Raw("startxref\n"u8).Integer(start).Raw("\n%%EOF\n"u8);
        Flush();
    }

    /// <summary>
    /// What an unfinished file ends with: the objects waiting written out and a trailer naming the catalog, but no
    /// cross-reference (a reader rebuilds it by scanning, finds the catalog by the trailer, and says the file was repaired).
    /// </summary>
    public void Unfinished(int root)
    {
        FlushPacked();
        _pending.Raw("trailer\n"u8);
        Trailer(_pending, root, 0, [], []);
        _pending.Line();
        Flush();
    }

    private void Trailer(PdfSyntax s, int root, int info, ReadOnlySpan<byte> id, ReadOnlySpan<byte> second, bool inside = false)
    {
        if (!inside)
        {
            s.Raw("<<"u8);
        }

        s.Raw("/Size "u8).Integer(_count).Raw("/Root "u8).Reference(root);
        if (info > 0)
        {
            s.Raw("/Info "u8).Reference(info);
        }

        if (EncryptNumber > 0)
        {
            s.Raw("/Encrypt "u8).Reference(EncryptNumber);
        }

        if (!id.IsEmpty)
        {
            s.Raw("/ID["u8).HexString(id).HexString(second.IsEmpty ? id : second).Byte((byte)']');
        }

        if (!inside)
        {
            s.Raw(">>"u8);
        }
    }

    public void Dispose()
    {
        _pending.Dispose();
        _packed?.Dispose();
        _packedData?.Dispose();
        _scratch?.Dispose();
        _crypt?.Dispose();
        _packed = _packedData = _scratch = _crypt = null;
    }
}
