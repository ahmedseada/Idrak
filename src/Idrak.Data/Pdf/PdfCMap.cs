// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Data.Pdf;

/// <summary>
/// A CMap (ISO 32000-2, 9.7.5 and 9.10.3), as an embedded stream gives it: its code space ranges (how a string's bytes
/// split into codes of one to four bytes), its codes to CIDs (<c>cidchar</c>, <c>cidrange</c>, <c>notdefrange</c>) for a
/// composite font's encoding, its codes to text (<c>bfchar</c>, <c>bfrange</c> with a start or an array) for a ToUnicode
/// map, and its writing mode. Ranges are kept as ranges (a range over every code costs one entry); a CMap it uses
/// (<c>usecmap</c>) lends what this one does not define. The predefined Identity-H and Identity-V (two bytes, the code
/// is the CID) are built in; the other predefined CMaps, tied to one script each, are not (plan 28, decision 3).
/// Immutable once read, so safe from several threads.
/// </summary>
internal sealed class PdfCMap
{
    private readonly List<CodeRange> _codeSpace = [];
    private readonly Dictionary<long, string> _chars = [];
    private readonly List<UnicodeRange> _ranges = [];
    private readonly Dictionary<long, int> _cidChars = [];
    private readonly List<CidRange> _cidRanges = [];
    private readonly List<CidRange> _notdefRanges = [];
    private PdfCMap? _parent;
    private bool _identity;

    private PdfCMap()
    {
    }

    /// <summary>Whether text in it is written top to bottom (<c>/WMode 1</c>, or Identity-V).</summary>
    public bool Vertical { get; private set; }

    /// <summary>Whether it maps codes to text (a ToUnicode map).</summary>
    public bool HasUnicode => _chars.Count > 0 || _ranges.Count > 0 || _parent?.HasUnicode == true;

    /// <summary>About how many bytes it holds (for the document's cache).</summary>
    public long Bytes => 256 + 48L * (_chars.Count + _ranges.Count + _cidChars.Count + _cidRanges.Count) + (_parent?.Bytes ?? 0);

    /// <summary>Identity-H or Identity-V: two-byte codes, each its own CID.</summary>
    public static PdfCMap Identity(bool vertical)
    {
        var map = new PdfCMap { _identity = true, Vertical = vertical };
        map._codeSpace.Add(new CodeRange(2, 0x0000, 0xFFFF));
        return map;
    }

    /// <summary>The CMap <paramref name="data"/> holds, over <paramref name="parent"/> (a CMap it uses, or null).</summary>
    public static PdfCMap Parse(ReadOnlySpan<byte> data, PdfCMap? parent, bool vertical)
    {
        var map = new PdfCMap { _parent = parent, Vertical = vertical || parent?.Vertical == true };
        map.Read(data);
        return map;
    }

    /// <summary>
    /// The code at <paramref name="at"/> of <paramref name="text"/> by the code space ranges (the shortest that matches;
    /// bytes no range matches take the length of a range their first byte is in, else one byte); its length.
    /// </summary>
    public int NextCode(ReadOnlySpan<byte> text, int at, out uint code)
    {
        var space = CodeSpace;
        int rest = text.Length - at;
        for (int length = 1; length <= 4 && length <= rest; length++)
        {
            foreach (var range in space)
            {
                if (range.Length == length && range.Contains(text.Slice(at, length)))
                {
                    code = Value(text.Slice(at, length));
                    return length;
                }
            }
        }

        foreach (var range in space)
        {
            if (range.FirstByteContains(text[at]))
            {
                int length = Math.Min(range.Length, rest);
                code = Value(text.Slice(at, length));
                return length;
            }
        }

        code = text[at];
        return 1;
    }

    /// <summary>The code ranges of this CMap, or of the one it uses when it defines none.</summary>
    private List<CodeRange> CodeSpace => _codeSpace.Count > 0 || _parent is null ? _codeSpace : _parent.CodeSpace;

    /// <summary>Whether it has code space ranges (its own or its parent's).</summary>
    public bool HasCodeSpace => CodeSpace.Count > 0;

    /// <summary>The CID of a code: its mapping, the notdef range it is in, or 0.</summary>
    public int Cid(uint code, int length)
    {
        if (_identity)
        {
            return (int)code;
        }

        if (_cidChars.TryGetValue(Key(code, length), out int cid))
        {
            return cid;
        }

        if (Find(_cidRanges, code, length) is { } range)
        {
            return range.Cid + (int)(code - range.Low);
        }

        if (_parent is not null)
        {
            return _parent.Cid(code, length);
        }

        return Find(_notdefRanges, code, length) is { } notdef ? notdef.Cid : 0;
    }

    /// <summary>The text of a code written into <paramref name="destination"/>: the count written, or 0 when the map has none.</summary>
    public int Unicode(uint code, int length, Span<char> destination)
    {
        if (_chars.TryGetValue(Key(code, length), out string? text))
        {
            return Copy(text, destination);
        }

        int lo = 0, hi = _ranges.Count - 1, found = -1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >>> 1;
            var r = _ranges[mid];
            int c = r.Length != length ? r.Length.CompareTo(length) : r.Low.CompareTo(code);
            if (c <= 0)
            {
                found = mid;
                lo = mid + 1;
            }
            else
            {
                hi = mid - 1;
            }
        }

        for (int i = found; i >= 0 && _ranges[i].Length == length; i--)
        {
            var range = _ranges[i];
            if (code < range.Low || code > range.High)
            {
                if (code > range.High && i < found - 64)
                {
                    break;   // ranges rarely overlap: a few looked back at are enough
                }

                continue;
            }

            int offset = (int)(code - range.Low);
            if (range.Array is { } array)
            {
                return offset < array.Length ? Copy(array[offset], destination) : 0;
            }

            int count = Copy(range.Start!, destination);
            if (count > 0)
            {
                destination[count - 1] = (char)(destination[count - 1] + offset);
            }

            return count;
        }

        return _parent?.Unicode(code, length, destination) ?? 0;
    }

    private static int Copy(string text, Span<char> destination)
    {
        if (text.Length == 0 || (text.Length == 1 && text[0] == '\0'))
        {
            return 0;   // a code mapped to nothing (or to U+0000, as some writers fill unused codes): no text
        }

        int count = Math.Min(text.Length, destination.Length);
        text.AsSpan(0, count).CopyTo(destination);
        return count;
    }

    private static CidRange? Find(List<CidRange> ranges, uint code, int length)
    {
        foreach (var range in ranges)
        {
            if (range.Length == length && code >= range.Low && code <= range.High)
            {
                return range;
            }
        }

        return null;
    }

    private static long Key(uint code, int length) => (long)length << 32 | code;

    private static uint Value(ReadOnlySpan<byte> bytes)
    {
        uint value = 0;
        foreach (byte b in bytes)
        {
            value = value << 8 | b;
        }

        return value;
    }

    private static Block BlockOf(ReadOnlySpan<byte> word) =>
        word.SequenceEqual("codespacerange"u8) ? Block.CodeSpace
        : word.SequenceEqual("bfchar"u8) ? Block.BfChar
        : word.SequenceEqual("bfrange"u8) ? Block.BfRange
        : word.SequenceEqual("cidchar"u8) ? Block.CidChar
        : word.SequenceEqual("cidrange"u8) ? Block.CidRange
        : word.SequenceEqual("notdefchar"u8) ? Block.NotdefChar
        : word.SequenceEqual("notdefrange"u8) ? Block.NotdefRange
        : Block.None;

    // A token as the lexer gives it.
    private readonly record struct Tok(PdfToken Kind, int Start, int Length, double Number);

    private enum Block : byte
    {
        None,
        CodeSpace,
        BfChar,
        BfRange,
        CidChar,
        CidRange,
        NotdefChar,
        NotdefRange,
    }

    // The CMap's operators and the entries of each block, applied as they come (no list of every entry is kept).
    private void Read(ReadOnlySpan<byte> data)
    {
        var lexer = new PdfContentLexer(data);
        var block = Block.None;
        Span<Tok> entry = stackalloc Tok[3];
        int filled = 0;
        List<string>? array = null;
        Tok previous = default, beforePrevious = default;
        Span<byte> scratch = stackalloc byte[512];
        while (true)
        {
            var token = lexer.Next(out int start, out int length, out double number);
            if (token == PdfToken.End)
            {
                return;
            }

            if (token == PdfToken.Keyword)
            {
                var word = data.Slice(start, length);
                if (word.StartsWith("begin"u8))
                {
                    block = BlockOf(word[5..]);
                    filled = 0;
                }
                else if (word.StartsWith("end"u8))
                {
                    block = Block.None;
                    filled = 0;
                }
                else if (word.SequenceEqual("def"u8) && beforePrevious.Kind == PdfToken.Name && data.Slice(beforePrevious.Start, beforePrevious.Length).SequenceEqual("WMode"u8)
                         && previous.Kind == PdfToken.Number)
                {
                    Vertical = previous.Number == 1;
                }
                else if (word.SequenceEqual("usecmap"u8) && previous.Kind == PdfToken.Name && _parent is null)
                {
                    var used = data.Slice(previous.Start, previous.Length);
                    if (used.SequenceEqual("Identity-H"u8) || used.SequenceEqual("Identity-V"u8))
                    {
                        _parent = Identity(used.EndsWith("V"u8));
                    }
                }

                (beforePrevious, previous) = (previous, new Tok(token, start, length, number));
                continue;
            }

            (beforePrevious, previous) = (previous, new Tok(token, start, length, number));
            if (block == Block.None)
            {
                continue;
            }

            if (array is not null)
            {
                if (token == PdfToken.ArrayEnd)
                {
                    AddRangeArray(data, entry, array, scratch);
                    array = null;
                    filled = 0;
                }
                else if (token is PdfToken.HexString or PdfToken.LiteralString or PdfToken.Name)
                {
                    array.Add(DestinationText(data, new Tok(token, start, length, number), scratch));
                }

                continue;
            }

            if (token == PdfToken.ArrayStart && block == Block.BfRange && filled == 2)
            {
                array = [];
                continue;
            }

            if (token is not (PdfToken.HexString or PdfToken.LiteralString or PdfToken.Number or PdfToken.Name))
            {
                filled = 0;   // something an entry cannot hold: start over
                continue;
            }

            entry[filled++] = new Tok(token, start, length, number);
            int needed = block is Block.CodeSpace or Block.BfChar or Block.CidChar or Block.NotdefChar ? 2 : 3;
            if (filled == needed)
            {
                Apply(block, data, entry, scratch);
                filled = 0;
            }
        }
    }

    private void Apply(Block block, ReadOnlySpan<byte> data, ReadOnlySpan<Tok> entry, Span<byte> scratch)
    {
        if (!Code(data, entry[0], scratch, out uint low, out int length))
        {
            return;
        }

        switch (block)
        {
            case Block.CodeSpace:
                if (Code(data, entry[1], scratch, out uint high, out int highLength) && highLength == length)
                {
                    _codeSpace.Add(new CodeRange(length, low, high));
                }

                break;
            case Block.BfChar:
                _chars[Key(low, length)] = DestinationText(data, entry[1], scratch);
                break;
            case Block.CidChar or Block.NotdefChar:
                if (entry[1].Kind == PdfToken.Number && entry[1].Number is >= 0 and <= int.MaxValue)
                {
                    if (block == Block.CidChar)
                    {
                        _cidChars[Key(low, length)] = (int)entry[1].Number;
                    }
                    else
                    {
                        _notdefRanges.Add(new CidRange(length, low, low, (int)entry[1].Number));
                    }
                }

                break;
            default:
                if (!Code(data, entry[1], scratch, out uint top, out _) || top < low)
                {
                    return;
                }

                if (block == Block.BfRange)
                {
                    if (entry[2].Kind is PdfToken.HexString or PdfToken.LiteralString)
                    {
                        InsertRange(new UnicodeRange(length, low, top, DestinationText(data, entry[2], scratch), null));
                    }
                }
                else if (entry[2].Kind == PdfToken.Number && entry[2].Number is >= 0 and <= int.MaxValue)
                {
                    (block == Block.CidRange ? _cidRanges : _notdefRanges).Add(new CidRange(length, low, top, (int)entry[2].Number));
                }

                break;
        }
    }

    private void AddRangeArray(ReadOnlySpan<byte> data, ReadOnlySpan<Tok> entry, List<string> array, Span<byte> scratch)
    {
        if (Code(data, entry[0], scratch, out uint low, out int length) && Code(data, entry[1], scratch, out uint top, out _) && top >= low)
        {
            InsertRange(new UnicodeRange(length, low, Math.Min(top, low + (uint)Math.Max(0, array.Count - 1)), null, [.. array]));
        }
    }

    // Ranges kept sorted by code length, then first code (they come sorted in nearly every map: an append).
    private void InsertRange(UnicodeRange range)
    {
        int at = _ranges.Count;
        while (at > 0 && (_ranges[at - 1].Length > range.Length || (_ranges[at - 1].Length == range.Length && _ranges[at - 1].Low > range.Low)))
        {
            at--;
        }

        _ranges.Insert(at, range);
    }

    // A source code: a hex or literal string of one to four bytes.
    private static bool Code(ReadOnlySpan<byte> data, Tok token, Span<byte> scratch, out uint code, out int length)
    {
        code = 0;
        length = 0;
        if (token.Kind is not (PdfToken.HexString or PdfToken.LiteralString) || token.Length > scratch.Length)
        {
            return false;
        }

        var raw = data.Slice(token.Start, token.Length);
        length = token.Kind == PdfToken.HexString ? PdfContentLexer.DecodeHex(raw, scratch) : PdfContentLexer.DecodeLiteral(raw, scratch);
        if (length is < 1 or > 4)
        {
            return false;
        }

        code = Value(scratch[..length]);
        return true;
    }

    // A destination: UTF-16BE bytes (hex or literal), or a glyph name's text.
    private static string DestinationText(ReadOnlySpan<byte> data, Tok token, Span<byte> scratch)
    {
        var raw = data.Slice(token.Start, Math.Min(token.Length, token.Kind == PdfToken.HexString ? 2 * scratch.Length : scratch.Length));
        if (token.Kind == PdfToken.Name)
        {
            Span<char> chars = stackalloc char[PdfFontTables.MaxGlyphText];
            int named = PdfFontTables.GlyphText(raw, zapfDingbats: false, chars);
            return new string(chars[..named]);
        }

        int count = token.Kind == PdfToken.HexString ? PdfContentLexer.DecodeHex(raw, scratch) : PdfContentLexer.DecodeLiteral(raw, scratch);
        return count == 1 ? ((char)scratch[0]).ToString() : Encoding.BigEndianUnicode.GetString(scratch[..(count & ~1)]);
    }

    // A code space range: codes of `Length` bytes, each byte between the low and high bytes at its place.
    private readonly record struct CodeRange(int Length, uint Low, uint High)
    {
        public bool Contains(ReadOnlySpan<byte> code)
        {
            for (int i = 0; i < code.Length; i++)
            {
                int shift = 8 * (Length - 1 - i);
                byte b = code[i];
                if (b < (byte)(Low >> shift) || b > (byte)(High >> shift))
                {
                    return false;
                }
            }

            return true;
        }

        public bool FirstByteContains(byte b)
        {
            int shift = 8 * (Length - 1);
            return b >= (byte)(Low >> shift) && b <= (byte)(High >> shift);
        }
    }

    // Codes Low..High of one length to text: Start with its last character raised by the code's place, or Array's entry.
    private readonly record struct UnicodeRange(int Length, uint Low, uint High, string? Start, string[]? Array);

    private readonly record struct CidRange(int Length, uint Low, uint High, int Cid);
}
