// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Data.Pdf;

/// <summary>The kinds of a PDF object (ISO 32000-2, 7.3).</summary>
internal enum PdfKind : byte
{
    Null,
    Boolean,
    Integer,
    Real,
    String,
    Name,
    Array,
    Dictionary,
    Stream,
    Reference,
}

/// <summary>An indirect reference: an object's number and generation.</summary>
internal readonly record struct PdfReference(int Number, int Generation)
{
    public override string ToString() => string.Create(CultureInfo.InvariantCulture, $"{Number} {Generation} R");
}

/// <summary>
/// One PDF object, held without a box for the scalars: numbers, booleans and references live in the value itself, the
/// rest (strings, names, arrays, dictionaries, streams) in an immutable object of their own. The default value is null.
/// </summary>
internal readonly struct PdfValue
{
    private readonly object? _object;
    private readonly long _bits;

    private PdfValue(PdfKind kind, long bits, object? value) => (Kind, _bits, _object) = (kind, bits, value);

    public PdfKind Kind { get; }

    public static PdfValue Null => default;

    public static PdfValue Boolean(bool value) => new(PdfKind.Boolean, value ? 1 : 0, null);

    public static PdfValue Integer(long value) => new(PdfKind.Integer, value, null);

    public static PdfValue Real(double value) => new(PdfKind.Real, BitConverter.DoubleToInt64Bits(value), null);

    public static PdfValue Reference(int number, int generation) => new(PdfKind.Reference, (uint)number | ((long)(uint)generation << 32), null);

    public static PdfValue Of(PdfString value) => new(PdfKind.String, 0, value);

    public static PdfValue Of(PdfName value) => new(PdfKind.Name, 0, value);

    public static PdfValue Of(PdfArray value) => new(PdfKind.Array, 0, value);

    public static PdfValue Of(PdfDictionary value) => new(PdfKind.Dictionary, 0, value);

    public static PdfValue Of(PdfStream value) => new(PdfKind.Stream, 0, value);

    public bool IsNull => Kind == PdfKind.Null;

    public bool IsReference => Kind == PdfKind.Reference;

    /// <summary>The boolean (false for any other kind).</summary>
    public bool AsBoolean => Kind == PdfKind.Boolean && _bits != 0;

    /// <summary>The reference (only for <see cref="PdfKind.Reference"/>).</summary>
    public PdfReference AsReference => Kind == PdfKind.Reference ? new PdfReference((int)(uint)_bits, (int)(uint)(_bits >> 32)) : default;

    public PdfString? AsString => _object as PdfString;

    public PdfName? AsName => _object as PdfName;

    public PdfArray? AsArray => _object as PdfArray;

    public PdfStream? AsStream => _object as PdfStream;

    /// <summary>The dictionary, or a stream's dictionary.</summary>
    public PdfDictionary? AsDictionary => _object switch
    {
        PdfDictionary dictionary => dictionary,
        PdfStream stream => stream.Dictionary,
        _ => null,
    };

    /// <summary>An integer, or a real that is a whole number within range.</summary>
    public bool TryInteger(out long value)
    {
        if (Kind == PdfKind.Integer)
        {
            value = _bits;
            return true;
        }

        if (Kind == PdfKind.Real)
        {
            double real = BitConverter.Int64BitsToDouble(_bits);
            if (real == Math.Floor(real) && Math.Abs(real) < 9e15)
            {
                value = (long)real;
                return true;
            }
        }

        value = 0;
        return false;
    }

    /// <summary>A number (an integer or a real).</summary>
    public bool TryNumber(out double value)
    {
        value = Kind switch
        {
            PdfKind.Integer => _bits,
            PdfKind.Real => BitConverter.Int64BitsToDouble(_bits),
            _ => double.NaN,
        };
        return Kind is PdfKind.Integer or PdfKind.Real;
    }

    /// <summary>Whether this is the name <paramref name="name"/>.</summary>
    public bool IsName(ReadOnlySpan<byte> name) => _object is PdfName n && n.Is(name);

    /// <summary>The object as PDF syntax (for messages and for the information's other entries); long values are cut short.</summary>
    public override string ToString()
    {
        var text = new StringBuilder();
        Write(text, 0);
        return text.ToString();
    }

    internal void Write(StringBuilder text, int depth)
    {
        if (text.Length > 4096)
        {
            text.Append("...");
            return;
        }

        switch (Kind)
        {
            case PdfKind.Null:
                text.Append("null");
                break;
            case PdfKind.Boolean:
                text.Append(AsBoolean ? "true" : "false");
                break;
            case PdfKind.Integer:
                text.Append(_bits.ToString(CultureInfo.InvariantCulture));
                break;
            case PdfKind.Real:
                text.Append(BitConverter.Int64BitsToDouble(_bits).ToString("0.######", CultureInfo.InvariantCulture));
                break;
            case PdfKind.Reference:
                text.Append(AsReference.ToString());
                break;
            case PdfKind.Name:
                text.Append('/').Append(AsName!.ToString());
                break;
            case PdfKind.String:
                text.Append('(').Append(PdfText.Decode(AsString!.Bytes)).Append(')');
                break;
            case PdfKind.Array:
                text.Append('[');
                var items = AsArray!.Items;
                for (int i = 0; i < items.Length; i++)
                {
                    if (i > 0)
                    {
                        text.Append(' ');
                    }

                    if (depth > 8)
                    {
                        text.Append("...");
                        break;
                    }

                    items[i].Write(text, depth + 1);
                }

                text.Append(']');
                break;
            default:
                var dictionary = AsDictionary!;
                text.Append("<<");
                for (int i = 0; i < dictionary.Count; i++)
                {
                    if (depth > 8)
                    {
                        text.Append("...");
                        break;
                    }

                    text.Append('/').Append(dictionary.KeyAt(i).ToString()).Append(' ');
                    dictionary.ValueAt(i).Write(text, depth + 1);
                }

                text.Append(">>");
                if (Kind == PdfKind.Stream)
                {
                    text.Append(" stream");
                }

                break;
        }
    }
}

/// <summary>A name (its bytes after <c>#xx</c> escapes are decoded); the common ones are shared, made once.</summary>
internal sealed class PdfName
{
    // The names most files repeat in every object, made once and shared by every document (bucketed by length).
    private static readonly PdfName[][] Common = Bucket(
    [
        "Type", "Subtype", "Length", "Filter", "DecodeParms", "Kids", "Count", "Parent", "Page", "Pages", "MediaBox", "CropBox", "Resources",
        "Contents", "Rotate", "Font", "XObject", "ProcSet", "ExtGState", "ColorSpace", "Pattern", "Shading", "Properties", "Image", "Form", "BBox",
        "Matrix", "Width", "Height", "BitsPerComponent", "FlateDecode", "DCTDecode", "DeviceRGB", "DeviceGray", "DeviceCMYK", "Catalog", "Root",
        "Info", "Size", "Prev", "ID", "XRef", "ObjStm", "N", "First", "W", "Index", "Predictor", "Columns", "Colors", "BaseFont", "Encoding",
        "FirstChar", "LastChar", "Widths", "FontDescriptor", "Type1", "TrueType", "Type0", "Annots", "Rect", "Border", "Annot", "Link", "Dest",
        "URI", "S", "A", "P", "F", "D", "Metadata", "XML", "Ascent", "Descent", "CapHeight", "Flags", "FontBBox", "FontName", "ItalicAngle",
        "StemV", "FontFile2", "ToUnicode", "CIDSystemInfo", "Registry", "Ordering", "Supplement", "DescendantFonts", "WinAnsiEncoding",
        "Helvetica", "PDF", "Text", "ImageB", "ImageC", "Group", "Transparency", "CS", "Version", "Producer", "Creator", "Title", "Author",
    ]);

    private readonly byte[] _bytes;
    private string? _text;

    private PdfName(byte[] bytes) => _bytes = bytes;

    public ReadOnlySpan<byte> Bytes => _bytes;

    /// <summary>The name of <paramref name="bytes"/> (decoded): a shared one when it is common, else a new one.</summary>
    public static PdfName From(ReadOnlySpan<byte> bytes)
    {
        if (bytes.Length < Common.Length)
        {
            foreach (var name in Common[bytes.Length])
            {
                if (bytes.SequenceEqual(name._bytes))
                {
                    return name;
                }
            }
        }

        return new PdfName(bytes.ToArray());
    }

    public bool Is(ReadOnlySpan<byte> name) => name.SequenceEqual(_bytes);

    /// <summary>The name as text (UTF-8, as PDF 2.0 writes names; bytes that are not UTF-8 one character each).</summary>
    public override string ToString() => _text ??= PdfText.Utf8OrLatin1(_bytes);

    private static PdfName[][] Bucket(string[] names)
    {
        int longest = names.Max(n => n.Length);
        var buckets = new List<PdfName>[longest + 1];
        for (int i = 0; i < buckets.Length; i++)
        {
            buckets[i] = [];
        }

        foreach (string name in names.Distinct(StringComparer.Ordinal))
        {
            buckets[name.Length].Add(new PdfName(Encoding.ASCII.GetBytes(name)));
        }

        return [.. buckets.Select(b => b.ToArray())];
    }
}

/// <summary>A string: its bytes as the file holds them (escapes and hex digits decoded).</summary>
internal sealed class PdfString(byte[] bytes)
{
    public byte[] Bytes { get; } = bytes;
}

/// <summary>An array.</summary>
internal sealed class PdfArray(PdfValue[] items)
{
    public static PdfArray Empty { get; } = new([]);

    public PdfValue[] Items { get; } = items;

    public int Count => Items.Length;

    public PdfValue this[int index] => Items[index];
}

/// <summary>
/// A dictionary: keys and values in the order the file gives them (a key given twice keeps its last value). Lookups compare
/// the key's bytes (<c>"Type"u8</c>); a dictionary holds a few keys, so a scan beats hashing.
/// </summary>
internal sealed class PdfDictionary
{
    private readonly PdfName[] _keys;
    private readonly PdfValue[] _values;

    public PdfDictionary(PdfName[] keys, PdfValue[] values) => (_keys, _values) = (keys, values);

    public static PdfDictionary Empty { get; } = new([], []);

    public int Count => _keys.Length;

    public PdfName KeyAt(int index) => _keys[index];

    public PdfValue ValueAt(int index) => _values[index];

    /// <summary>The value of <paramref name="key"/> (unresolved), or null.</summary>
    public PdfValue this[ReadOnlySpan<byte> key] => TryGet(key, out var value) ? value : PdfValue.Null;

    public bool TryGet(ReadOnlySpan<byte> key, out PdfValue value)
    {
        int at = IndexOf(key);
        value = at >= 0 ? _values[at] : PdfValue.Null;
        return at >= 0;
    }

    /// <summary>Where <paramref name="key"/> is (its <see cref="KeyAt"/> and <see cref="ValueAt"/>), or -1.</summary>
    public int IndexOf(ReadOnlySpan<byte> key)
    {
        for (int i = 0; i < _keys.Length; i++)
        {
            if (_keys[i].Is(key))
            {
                return i;
            }
        }

        return -1;
    }

    public bool Contains(ReadOnlySpan<byte> key) => TryGet(key, out var value) && !value.IsNull;
}

/// <summary>
/// A stream: its dictionary and where its stored bytes are in the file (an offset from the file's start and a length
/// found from <c>/Length</c>, or from <c>endstream</c> when the length is wrong). Its data is read when it is decoded.
/// </summary>
internal sealed class PdfStream(PdfDictionary dictionary, int number, int generation, long offset, long length, bool lengthRecovered)
{
    public PdfDictionary Dictionary { get; } = dictionary;

    /// <summary>The number of the object the stream is.</summary>
    public int Number { get; } = number;

    public int Generation { get; } = generation;

    /// <summary>Where the stored bytes start, from the file's start.</summary>
    public long Offset { get; } = offset;

    /// <summary>How many bytes are stored.</summary>
    public long Length { get; } = length;

    /// <summary>Whether <c>/Length</c> was wrong and the length was found by <c>endstream</c>.</summary>
    public bool LengthRecovered { get; } = lengthRecovered;
}
