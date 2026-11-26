// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Imaging;

/// <summary>
/// "cff": a bare CFF font program (what a PDF embeds as FontFile3 /Type1C or /CIDFontType0C), its first font read by
/// <see cref="CffOutlines"/>: outlines in 1,000 units an em (its FontMatrix applied), advances from its charstrings, and
/// its glyphs' names (the charset; a CID-keyed font's glyphs named "cid" and five digits) and built-in encoding given as
/// the "post" and "cmap" tables a TrueType font would have (<see cref="PostScriptFontTables"/>). It lists no file
/// extension: such programs come inside documents, read by their first bytes.
/// </summary>
public sealed class CffFontFormat : IFontFormat
{
    /// <summary>The format's name.</summary>
    public const string FormatName = "cff";

    /// <inheritdoc/>
    public string Name => FormatName;

    /// <summary>None: such programs come inside documents.</summary>
    public IReadOnlyCollection<string> Extensions { get; } = [];

    // A CFF header: version 1.x, a header size of 4 or more, an offset size of 1 to 4.
    /// <inheritdoc/>
    public bool Accepts(ReadOnlySpan<byte> header) => header.Length >= 4 && header[0] == 1 && header[2] >= 4 && header[3] is >= 1 and <= 4;

    /// <inheritdoc/>
    public IReadOnlyList<IFontFace> Read(ReadOnlyMemory<byte> file, string name) => [new CffFace(file, name)];
}

/// <summary>A bare CFF program's first font as a face (see <see cref="CffFontFormat"/>).</summary>
internal sealed class CffFace : IFontFace
{
    private const int UnitsPerEm = 1000;

    private readonly CffOutlines _cff;
    private readonly string[] _names;
    private readonly int[] _encoding;
    private readonly GlyphOutline?[] _outlines;
    private readonly int[] _advances;
    private readonly Lazy<byte[]> _post, _cmap;

    public CffFace(ReadOnlyMemory<byte> file, string name)
    {
        _cff = new CffOutlines(file, UnitsPerEm, 0, name);
        Family = _cff.FontName ?? name;
        int count = _cff.Count;
        var charset = _cff.Charset();
        _names = new string[count];
        for (int g = 0; g < count; g++)
        {
            _names[g] = g == 0 ? ".notdef" : _cff.CidKeyed ? string.Create(CultureInfo.InvariantCulture, $"cid{charset[g]:D5}") : _cff.String(charset[g]);
        }

        // The built-in encoding: a custom one's codes, else the predefined one's names (standard or Expert) through the charset.
        _encoding = _cff.CidKeyed ? new int[256]
            : _cff.Encoding(charset) ?? ByNames(_cff.ExpertEncoded ? PostScriptFontTables.ExpertEncoding : PostScriptFontTables.StandardEncoding, _names);
        _outlines = new GlyphOutline?[count];
        _advances = new int[count];
        Array.Fill(_advances, -1);
        double[] box = _cff.Top(5) is { Length: 4 } b ? b : [0, -200, 1000, 800];
        Metrics = new FontMetrics(UnitsPerEm, (int)Math.Round(box[3]), (int)Math.Round(box[1]), 0);
        _post = new Lazy<byte[]>(() => PostScriptFontTables.Post(_names));
        _cmap = new Lazy<byte[]>(() => PostScriptFontTables.Cmap(_encoding));
    }

    public string Family { get; }

    public string Style => "Regular";

    public string Format => CffFontFormat.FormatName;

    public FontMetrics Metrics { get; }

    public int GlyphCount => _names.Length;

    public int Glyph(int codePoint) => 0;

    public int Advance(int glyph)
    {
        if ((uint)glyph >= (uint)_advances.Length)
        {
            return 0;
        }

        int known = Volatile.Read(ref _advances[glyph]);
        if (known < 0)
        {
            known = (int)MathF.Round(_cff.Width(glyph));
            Volatile.Write(ref _advances[glyph], known);
        }

        return known;
    }

    public GlyphOutline Outline(int glyph)
    {
        if ((uint)glyph >= (uint)_outlines.Length)
        {
            return GlyphOutline.Empty;
        }

        var outline = Volatile.Read(ref _outlines[glyph]);
        if (outline is null)
        {
            outline = _cff.Outline(glyph);
            Volatile.Write(ref _outlines[glyph], outline);
        }

        return outline;
    }

    public ReadOnlyMemory<byte> Table(string tag) => tag switch
    {
        "post" => _post.Value,
        "cmap" => _cmap.Value,
        _ => ReadOnlyMemory<byte>.Empty,
    };

    /// <summary>An encoding by glyph names (a name a code) to glyphs, by the glyphs' own names.</summary>
    internal static int[] ByNames(string[] encoding, string[] names)
    {
        var index = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int g = names.Length - 1; g >= 1; g--)
        {
            index[names[g]] = g;
        }

        var glyphs = new int[256];
        for (int code = 0; code < 256 && code < encoding.Length; code++)
        {
            glyphs[code] = index.TryGetValue(encoding[code], out int g) ? g : 0;
        }

        return glyphs;
    }
}

/// <summary>
/// "type1": a Type 1 font program (Adobe Type 1 Font Format: the clear text, then the eexec-encrypted private part, as a
/// PDF embeds it in FontFile, or as a PFA or PFB file): its Subrs and CharStrings decrypted (lenIV), each charstring
/// interpreted into an outline (every path command, flex through othersubrs 0-2, hint replacement through othersubr 3,
/// seac's accented letters, div, setcurrentpoint), in 1,000 units an em (its FontMatrix applied); advances from hsbw or
/// sbw; the glyphs' names and the program's own encoding as the "post" and "cmap" tables
/// (<see cref="PostScriptFontTables"/>). No file extension is listed: programs come inside documents.
/// </summary>
public sealed class Type1FontFormat : IFontFormat
{
    /// <summary>The format's name.</summary>
    public const string FormatName = "type1";

    /// <inheritdoc/>
    public string Name => FormatName;

    /// <summary>None: such programs come inside documents.</summary>
    public IReadOnlyCollection<string> Extensions { get; } = [];

    /// <inheritdoc/>
    public bool Accepts(ReadOnlySpan<byte> header) =>
        header.StartsWith("%!PS-AdobeFont"u8) || header.StartsWith("%!FontType1"u8) || (header.Length >= 2 && header[0] == 0x80 && header[1] == 0x01);

    /// <inheritdoc/>
    public IReadOnlyList<IFontFace> Read(ReadOnlyMemory<byte> file, string name) => [new Type1Face(file.Span, name)];
}

/// <summary>A Type 1 program as a face (see <see cref="Type1FontFormat"/>).</summary>
internal sealed class Type1Face : IFontFace
{
    private const int UnitsPerEm = 1000, MaxDepth = 10, MaxStack = 64;

    private readonly string[] _names;
    private readonly byte[][] _charStrings;
    private readonly byte[][] _subrs;
    private readonly Dictionary<string, int> _byName = new(StringComparer.Ordinal);
    private readonly int[] _encoding;
    private readonly int _lenIV;
    private readonly double[] _matrix;
    private readonly GlyphOutline?[] _outlines;
    private readonly int[] _advances;
    private readonly Lazy<byte[]> _post, _cmap;

    public Type1Face(ReadOnlySpan<byte> program, string name)
    {
        var (clear, binary) = Parts(program);
        var text = Encoding.Latin1.GetString(clear);
        Family = Literal(text, "/FamilyName") ?? Name(text, "/FontName") ?? name;
        Style = Literal(text, "/Weight") is { Length: > 0 } weight && weight != "Medium" ? weight : "Regular";
        _matrix = Numbers(text, "/FontMatrix", 6) ?? [0.001, 0, 0, 0.001, 0, 0];
        double[] box = Numbers(text, "/FontBBox", 4) ?? [0, -200, 1000, 800];

        byte[] privatePart = Decrypt(binary, 55665, 4);
        _lenIV = PrivateNumber(privatePart, "/lenIV"u8) is int lenIV ? lenIV : 4;
        _subrs = Subrs(privatePart);
        var (names, strings) = CharStrings(privatePart);
        int notdef = names.IndexOf(".notdef");
        if (notdef > 0)
        {
            (names[0], names[notdef]) = (names[notdef], names[0]);
            (strings[0], strings[notdef]) = (strings[notdef], strings[0]);
        }
        else if (notdef < 0)
        {
            names.Insert(0, ".notdef");
            strings.Insert(0, []);
        }

        _names = [.. names];
        _charStrings = [.. strings];
        for (int g = _names.Length - 1; g >= 0; g--)
        {
            _byName[_names[g]] = g;
        }

        string[] encodingNames = Encoding1(text);
        _encoding = new int[256];
        for (int code = 0; code < 256; code++)
        {
            _encoding[code] = _byName.TryGetValue(encodingNames[code], out int g) && g > 0 ? g : 0;
        }

        _outlines = new GlyphOutline?[_names.Length];
        _advances = new int[_names.Length];
        Array.Fill(_advances, int.MinValue);
        double scale = _matrix[3] * UnitsPerEm;
        Metrics = new FontMetrics(UnitsPerEm, (int)Math.Round(box[3] * scale), (int)Math.Round(box[1] * scale), 0);
        _post = new Lazy<byte[]>(() => PostScriptFontTables.Post(_names));
        _cmap = new Lazy<byte[]>(() => PostScriptFontTables.Cmap(_encoding));
    }

    public string Family { get; }

    public string Style { get; }

    public string Format => Type1FontFormat.FormatName;

    public FontMetrics Metrics { get; }

    public int GlyphCount => _names.Length;

    public int Glyph(int codePoint) => 0;

    public int Advance(int glyph)
    {
        if ((uint)glyph >= (uint)_advances.Length)
        {
            return 0;
        }

        int known = Volatile.Read(ref _advances[glyph]);
        if (known == int.MinValue)
        {
            var interpreter = new Interpreter(this);
            interpreter.Run(glyph);
            known = (int)Math.Round(interpreter.Width * _matrix[0] * UnitsPerEm);
            Volatile.Write(ref _advances[glyph], known);
        }

        return known;
    }

    public GlyphOutline Outline(int glyph)
    {
        if ((uint)glyph >= (uint)_outlines.Length)
        {
            return GlyphOutline.Empty;
        }

        var outline = Volatile.Read(ref _outlines[glyph]);
        if (outline is null)
        {
            var interpreter = new Interpreter(this);
            interpreter.Run(glyph);
            outline = interpreter.Build();
            Volatile.Write(ref _outlines[glyph], outline);
        }

        return outline;
    }

    public ReadOnlyMemory<byte> Table(string tag) => tag switch
    {
        "post" => _post.Value,
        "cmap" => _cmap.Value,
        _ => ReadOnlyMemory<byte>.Empty,
    };

    // ---- the program -------------------------------------------------------------------------------------------------

    // The clear text and the encrypted part: a PFB's segments joined, or the text up to "eexec" and what follows it (hex
    // turned to bytes when it is hex).
    private static (byte[] Clear, byte[] Binary) Parts(ReadOnlySpan<byte> program)
    {
        if (program.Length >= 6 && program[0] == 0x80)
        {
            var clear = new List<byte>();
            var binary = new List<byte>();
            int at = 0;
            while (at + 6 <= program.Length && program[at] == 0x80 && program[at + 1] is 1 or 2)
            {
                int type = program[at + 1];
                long length = (uint)(program[at + 2] | program[at + 3] << 8 | program[at + 4] << 16 | program[at + 5] << 24);
                at += 6;
                int take = (int)Math.Min(length, program.Length - at);
                (type == 1 ? clear : binary).AddRange(program.Slice(at, take).ToArray());
                at += take;
            }

            return ([.. clear], binary.Count > 0 ? Hex([.. binary]) : []);
        }

        int eexec = program.IndexOf("eexec"u8);
        if (eexec < 0)
        {
            throw new InvalidDataException("A Type 1 program without eexec: no private part.");
        }

        int start = eexec + 5;
        while (start < program.Length && program[start] is (byte)' ' or (byte)'\r' or (byte)'\n' or (byte)'\t')
        {
            start++;
        }

        return (program[..eexec].ToArray(), Hex(program[start..].ToArray()));
    }

    // The encrypted part as bytes: hex digits (whitespace between them) when its first four bytes are hex, else as it is.
    private static byte[] Hex(byte[] data)
    {
        if (data.Length < 4 || !IsHex(data[0]) || !IsHex(data[1]) || !IsHex(data[2]) || !IsHex(data[3]))
        {
            return data;
        }

        var bytes = new byte[data.Length / 2 + 1];
        int n = 0, high = -1;
        foreach (byte b in data)
        {
            if (!IsHex(b))
            {
                if (b is (byte)' ' or (byte)'\r' or (byte)'\n' or (byte)'\t')
                {
                    continue;
                }

                break;
            }

            int v = b <= '9' ? b - '0' : (b | 0x20) - 'a' + 10;
            if (high < 0)
            {
                high = v;
            }
            else
            {
                bytes[n++] = (byte)(high << 4 | v);
                high = -1;
            }
        }

        return bytes.AsSpan(0, n).ToArray();
    }

    private static bool IsHex(byte b) => b is >= (byte)'0' and <= (byte)'9' or >= (byte)'a' and <= (byte)'f' or >= (byte)'A' and <= (byte)'F';

    // Type 1 decryption (r = 55665 for eexec, 4330 for charstrings), the first `skip` bytes dropped.
    private static byte[] Decrypt(ReadOnlySpan<byte> data, ushort key, int skip)
    {
        var plain = new byte[Math.Max(0, data.Length - skip)];
        ushort r = key;
        for (int i = 0; i < data.Length; i++)
        {
            byte c = data[i];
            byte p = (byte)(c ^ (r >> 8));
            r = (ushort)((c + r) * 52845 + 22719);
            if (i >= skip)
            {
                plain[i - skip] = p;
            }
        }

        return plain;
    }

    private byte[] CharString(byte[] encrypted) => _lenIV < 0 ? encrypted : Decrypt(encrypted, 4330, _lenIV);

    private static int? PrivateNumber(ReadOnlySpan<byte> data, ReadOnlySpan<byte> key)
    {
        int at = data.IndexOf(key);
        if (at < 0)
        {
            return null;
        }

        int i = at + key.Length;
        return Integer(data, ref i);
    }

    // An integer after whitespace at `i`, or null.
    private static int? Integer(ReadOnlySpan<byte> data, ref int i)
    {
        while (i < data.Length && data[i] is (byte)' ' or (byte)'\r' or (byte)'\n' or (byte)'\t')
        {
            i++;
        }

        int start = i;
        if (i < data.Length && data[i] == '-')
        {
            i++;
        }

        while (i < data.Length && data[i] is >= (byte)'0' and <= (byte)'9')
        {
            i++;
        }

        return i > start && int.TryParse(data[start..i], NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out int value) ? value : null;
    }

    // The next word (letters and the RD/ND/NP spellings) after whitespace.
    private static ReadOnlySpan<byte> Word(ReadOnlySpan<byte> data, ref int i)
    {
        while (i < data.Length && data[i] is (byte)' ' or (byte)'\r' or (byte)'\n' or (byte)'\t')
        {
            i++;
        }

        int start = i;
        while (i < data.Length && data[i] is not ((byte)' ' or (byte)'\r' or (byte)'\n' or (byte)'\t' or (byte)'/' or (byte)'[' or (byte)'{'))
        {
            i++;
        }

        return data[start..i];
    }

    // The Subrs: "dup i n RD <n bytes> NP" for each.
    private static byte[][] Subrs(ReadOnlySpan<byte> data)
    {
        int at = data.IndexOf("/Subrs"u8);
        if (at < 0)
        {
            return [];
        }

        int i = at + 6;
        int count = Math.Clamp(Integer(data, ref i) ?? 0, 0, 1 << 16);
        var subrs = new byte[count][];
        Array.Fill(subrs, []);
        for (int k = 0; k < count; k++)
        {
            int dup = data[i..].IndexOf("dup"u8);
            int charStrings = data[i..].IndexOf("/CharStrings"u8);
            if (dup < 0 || (charStrings >= 0 && charStrings < dup))
            {
                break;
            }

            i += dup + 3;
            int? index = Integer(data, ref i), length = Integer(data, ref i);
            Word(data, ref i);                                                       // RD or -|
            i++;                                                                     // the one space before the bytes
            if (index is not int s || length is not int n || n < 0 || i + n > data.Length)
            {
                break;
            }

            if ((uint)s < (uint)count)
            {
                subrs[s] = data.Slice(i, n).ToArray();
            }

            i += n;
        }

        return subrs;
    }

    // The CharStrings: "/name n RD <n bytes> ND" after "begin", up to "end".
    private static (List<string> Names, List<byte[]> Strings) CharStrings(ReadOnlySpan<byte> data)
    {
        var names = new List<string>();
        var strings = new List<byte[]>();
        int at = data.IndexOf("/CharStrings"u8);
        if (at < 0)
        {
            throw new InvalidDataException("A Type 1 program without CharStrings.");
        }

        int i = at + 12;
        int count = Math.Clamp(Integer(data, ref i) ?? 0, 0, 1 << 16);
        int begin = data[i..].IndexOf("begin"u8);
        if (begin < 0)
        {
            return (names, strings);
        }

        i += begin + 5;
        while (names.Count < ushort.MaxValue && i < data.Length)
        {
            while (i < data.Length && data[i] is (byte)' ' or (byte)'\r' or (byte)'\n' or (byte)'\t')
            {
                i++;
            }

            if (i >= data.Length || data[i] != '/')
            {
                break;                                                               // "end"
            }

            i++;
            var name = Word(data, ref i);
            int? length = Integer(data, ref i);
            Word(data, ref i);                                                       // RD or -|
            i++;
            if (length is not int n || n < 0 || i + n > data.Length)
            {
                break;
            }

            names.Add(Encoding.Latin1.GetString(name));
            strings.Add(data.Slice(i, n).ToArray());
            i += n;
            Word(data, ref i);                                                       // ND or |-
        }

        return (names, strings);
    }

    // The program's own encoding: StandardEncoding, or "dup CODE /NAME put" entries.
    private static string[] Encoding1(string text)
    {
        var names = new string[256];
        Array.Fill(names, ".notdef");
        int at = text.IndexOf("/Encoding", StringComparison.Ordinal);
        if (at < 0 || text.AsSpan(at + 9).TrimStart().StartsWith("StandardEncoding", StringComparison.Ordinal))
        {
            PostScriptFontTables.StandardEncoding.CopyTo(names, 0);
            return names;
        }

        int end = text.IndexOf(" def", at, StringComparison.Ordinal);
        int readOnly = text.IndexOf("readonly", at, StringComparison.Ordinal);
        if (readOnly >= 0 && (end < 0 || readOnly < end))
        {
            end = readOnly;
        }

        var part = text.AsSpan(at, (end < 0 ? text.Length : end) - at);
        int i = 0;
        while ((i = part.IndexOf("dup ", StringComparison.Ordinal)) >= 0)
        {
            part = part[(i + 4)..];
            int space = part.IndexOf(' ');
            if (space < 0 || !int.TryParse(part[..space], NumberStyles.None, CultureInfo.InvariantCulture, out int code) || code > 255)
            {
                continue;
            }

            var rest = part[(space + 1)..].TrimStart();
            if (rest.Length < 2 || rest[0] != '/')
            {
                continue;
            }

            var name = rest[1..];
            int stop = name.IndexOfAny(" \r\n\t/[]{}()<>");
            names[code] = (stop < 0 ? name : name[..stop]).ToString();
        }

        return names;
    }

    private static string? Literal(string text, string key)
    {
        int at = text.IndexOf(key, StringComparison.Ordinal);
        if (at < 0)
        {
            return null;
        }

        int open = text.IndexOf('(', at), close = open < 0 ? -1 : text.IndexOf(')', open);
        int def = text.IndexOf("def", at, StringComparison.Ordinal);
        return open >= 0 && close > open && (def < 0 || open < def) ? text[(open + 1)..close] : null;
    }

    private static string? Name(string text, string key)
    {
        int at = text.IndexOf(key, StringComparison.Ordinal);
        if (at < 0)
        {
            return null;
        }

        int slash = text.IndexOf('/', at + key.Length);
        if (slash < 0)
        {
            return null;
        }

        int end = slash + 1;
        while (end < text.Length && !char.IsWhiteSpace(text[end]) && text[end] is not ('/' or '[' or '{' or '('))
        {
            end++;
        }

        return end > slash + 1 ? text[(slash + 1)..end] : null;
    }

    private static double[]? Numbers(string text, string key, int count)
    {
        int at = text.IndexOf(key, StringComparison.Ordinal);
        if (at < 0)
        {
            return null;
        }

        int open = text.IndexOfAny(['[', '{'], at), close = open < 0 ? -1 : text.IndexOfAny([']', '}'], open);
        if (open < 0 || close < 0)
        {
            return null;
        }

        var parts = text[(open + 1)..close].Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length != count)
        {
            return null;
        }

        var values = new double[count];
        for (int k = 0; k < count; k++)
        {
            if (!double.TryParse(parts[k], NumberStyles.Float, CultureInfo.InvariantCulture, out values[k]))
            {
                return null;
            }
        }

        return values;
    }

    // ---- charstrings -------------------------------------------------------------------------------------------------

    // One glyph's Type 1 charstring run: its path (charstring units, then the FontMatrix into 1,000 units an em) and its width.
    private sealed class Interpreter(Type1Face face)
    {
        private readonly double[] _stack = new double[MaxStack];
        private readonly Stack<double> _results = new();
        private readonly List<OutlineVerb> _verbs = [];
        private readonly List<float> _points = [];
        private int _top;
        private double _x, _y, _sbx;
        private bool _open, _flex, _ended;
        private readonly List<(double X, double Y)> _flexPoints = [];

        public double Width { get; private set; }

        public void Run(int glyph)
        {
            try
            {
                Execute(face.CharString(face._charStrings[glyph]), 0);
            }
            catch (Exception e) when (e is IndexOutOfRangeException or InvalidOperationException or ArgumentException)
            {
                // A damaged charstring: what was drawn before the damage stays.
            }

            Close();
        }

        public GlyphOutline Build()
        {
            if (_verbs.Count == 0)
            {
                return GlyphOutline.Empty;
            }

            // The FontMatrix (charstring units to the em) times the units an em.
            var m = face._matrix;
            var points = _points.ToArray();
            for (int i = 0; i < points.Length; i += 2)
            {
                double x = points[i], y = points[i + 1];
                points[i] = (float)((m[0] * x + m[2] * y + m[4]) * UnitsPerEm);
                points[i + 1] = (float)((m[1] * x + m[3] * y + m[5]) * UnitsPerEm);
            }

            return new GlyphOutline([.. _verbs], points);
        }

        private void Execute(byte[] code, int depth)
        {
            if (depth > MaxDepth)
            {
                throw new InvalidOperationException("subroutines nested too deep");
            }

            int i = 0;
            while (i < code.Length && !_ended)
            {
                int v = code[i++];
                if (v >= 32)
                {
                    double number;
                    if (v <= 246)
                    {
                        number = v - 139;
                    }
                    else if (v <= 250)
                    {
                        number = (v - 247) * 256 + code[i++] + 108;
                    }
                    else if (v <= 254)
                    {
                        number = -(v - 251) * 256 - code[i++] - 108;
                    }
                    else
                    {
                        number = code[i] << 24 | code[i + 1] << 16 | code[i + 2] << 8 | code[i + 3];
                        i += 4;
                    }

                    Push(number);
                    continue;
                }

                if (v == 12)
                {
                    Escape(code[i++], depth);
                    continue;
                }

                switch (v)
                {
                    case 1 or 3:                                                     // hstem, vstem
                        _top = 0;
                        break;
                    case 4:                                                          // vmoveto
                        Move(0, Arg(0));
                        break;
                    case 5:                                                          // rlineto
                        Line(Arg(0), Arg(1));
                        break;
                    case 6:                                                          // hlineto
                        Line(Arg(0), 0);
                        break;
                    case 7:                                                          // vlineto
                        Line(0, Arg(0));
                        break;
                    case 8:                                                          // rrcurveto
                        Curve(Arg(0), Arg(1), Arg(2), Arg(3), Arg(4), Arg(5));
                        break;
                    case 9:                                                          // closepath
                        Close();
                        _top = 0;
                        break;
                    case 10:                                                         // callsubr
                    {
                        int s = (int)Pop();
                        if ((uint)s < (uint)face._subrs.Length)
                        {
                            Execute(face.CharString(face._subrs[s]), depth + 1);
                        }

                        break;
                    }

                    case 11:                                                         // return
                        return;
                    case 13:                                                         // hsbw
                        (_sbx, _x, _y, Width) = (Arg(0), Arg(0), 0, Arg(1));
                        _top = 0;
                        break;
                    case 14:                                                         // endchar
                        Close();
                        _ended = true;
                        break;
                    case 21:                                                         // rmoveto
                        Move(Arg(0), Arg(1));
                        break;
                    case 22:                                                         // hmoveto
                        Move(Arg(0), 0);
                        break;
                    case 30:                                                         // vhcurveto
                        Curve(0, Arg(0), Arg(1), Arg(2), Arg(3), 0);
                        break;
                    case 31:                                                         // hvcurveto
                        Curve(Arg(0), 0, Arg(1), Arg(2), 0, Arg(3));
                        break;
                    default:
                        _top = 0;
                        break;
                }
            }
        }

        private void Escape(int op, int depth)
        {
            switch (op)
            {
                case 6:                                                              // seac: asb adx ady bchar achar
                {
                    double asb = Arg(0), adx = Arg(1), ady = Arg(2);
                    int bchar = (int)Arg(3), achar = (int)Arg(4);
                    _top = 0;
                    Seac(asb, adx, ady, bchar, achar, depth);
                    _ended = true;
                    break;
                }

                case 7:                                                              // sbw
                    (_sbx, _x, _y, Width) = (Arg(0), Arg(0), Arg(1), Arg(2));
                    _top = 0;
                    break;
                case 12:                                                             // div
                {
                    double b = Pop(), a = Pop();
                    Push(b != 0 ? a / b : 0);
                    break;
                }

                case 16:                                                             // callothersubr
                {
                    int number = (int)Pop(), count = Math.Clamp((int)Pop(), 0, _top);
                    var args = new double[count];
                    for (int k = count - 1; k >= 0; k--)
                    {
                        args[k] = Pop();
                    }

                    OtherSubr(number, args);
                    break;
                }

                case 17:                                                             // pop
                    Push(_results.Count > 0 ? _results.Pop() : 0);
                    break;
                case 33:                                                             // setcurrentpoint
                    (_x, _y) = (Arg(0), Arg(1));
                    _top = 0;
                    break;
                default:                                                             // dotsection, vstem3, hstem3 and the rest
                    _top = 0;
                    break;
            }
        }

        // Flex (0 ends it, 1 starts it, 2 adds a point), hint replacement (3: its subroutine back), others their arguments back.
        private void OtherSubr(int number, double[] args)
        {
            switch (number)
            {
                case 1:
                    _flex = true;
                    _flexPoints.Clear();
                    break;
                case 2:
                    break;
                case 0 when args.Length >= 3:
                {
                    _flex = false;
                    // The reference point, then two curves' three points each.
                    var p = _flexPoints;
                    if (p.Count >= 7)
                    {
                        AddCurve(p[1], p[2], p[3]);
                        AddCurve(p[4], p[5], p[6]);
                        (_x, _y) = p[6];
                    }

                    _results.Push(args[2]);
                    _results.Push(args[1]);
                    break;
                }

                case 3 when args.Length >= 1:
                    _results.Push(args[0]);                                          // the hint subroutine, called next
                    break;
                default:
                    for (int k = 0; k < args.Length; k++)
                    {
                        _results.Push(args[k]);
                    }

                    break;
            }
        }

        // An accented letter: the base glyph, and the accent moved by (adx - asb + sbx, ady), both by the standard encoding.
        private void Seac(double asb, double adx, double ady, int bchar, int achar, int depth)
        {
            double sbx = _sbx;
            if ((uint)bchar < 256 && face._byName.TryGetValue(PostScriptFontTables.StandardEncoding[bchar], out int b))
            {
                var bse = new Interpreter(face);
                bse.Execute(face.CharString(face._charStrings[b]), depth + 1);
                bse.Close();
                AppendFrom(bse, 0, 0);
            }

            if ((uint)achar < 256 && face._byName.TryGetValue(PostScriptFontTables.StandardEncoding[achar], out int a))
            {
                var accent = new Interpreter(face);
                accent.Execute(face.CharString(face._charStrings[a]), depth + 1);
                accent.Close();
                AppendFrom(accent, adx - asb + sbx, ady);
            }
        }

        private void AppendFrom(Interpreter other, double dx, double dy)
        {
            Close();
            _verbs.AddRange(other._verbs);
            for (int i = 0; i < other._points.Count; i += 2)
            {
                _points.Add((float)(other._points[i] + dx));
                _points.Add((float)(other._points[i + 1] + dy));
            }
        }

        private void Push(double value)
        {
            if (_top < MaxStack)
            {
                _stack[_top++] = value;
            }
        }

        private double Pop() => _top > 0 ? _stack[--_top] : 0;

        // Operand k of the operator about to run, from the bottom of the stack.
        private double Arg(int k) => k < _top ? _stack[k] : 0;

        private void Move(double dx, double dy)
        {
            _top = 0;
            _x += dx;
            _y += dy;
            if (_flex)
            {
                _flexPoints.Add((_x, _y));
                return;
            }

            Close();
            _verbs.Add(OutlineVerb.Move);
            _points.Add((float)_x);
            _points.Add((float)_y);
            _open = true;
        }

        private void Line(double dx, double dy)
        {
            _top = 0;
            Open();
            _x += dx;
            _y += dy;
            _verbs.Add(OutlineVerb.Line);
            _points.Add((float)_x);
            _points.Add((float)_y);
        }

        private void Curve(double dx1, double dy1, double dx2, double dy2, double dx3, double dy3)
        {
            _top = 0;
            double x1 = _x + dx1, y1 = _y + dy1, x2 = x1 + dx2, y2 = y1 + dy2;
            AddCurve((x1, y1), (x2, y2), (x2 + dx3, y2 + dy3));
        }

        private void AddCurve((double X, double Y) c1, (double X, double Y) c2, (double X, double Y) end)
        {
            Open();
            _verbs.Add(OutlineVerb.Cubic);
            _points.AddRange([(float)c1.X, (float)c1.Y, (float)c2.X, (float)c2.Y, (float)end.X, (float)end.Y]);
            (_x, _y) = end;
        }

        // A path drawn with no move before it starts where the pen is.
        private void Open()
        {
            if (!_open)
            {
                _verbs.Add(OutlineVerb.Move);
                _points.Add((float)_x);
                _points.Add((float)_y);
                _open = true;
            }
        }

        private void Close()
        {
            if (_open)
            {
                _verbs.Add(OutlineVerb.Close);
                _open = false;
            }
        }
    }
}
