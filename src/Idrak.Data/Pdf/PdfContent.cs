// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Globalization;
using System.Runtime.CompilerServices;

namespace Idrak.Data.Pdf;

/// <summary>An affine map as PDF writes it, [a b c d e f]: (x, y) to (a x + c y + e, b x + d y + f).</summary>
internal readonly record struct PdfMatrix(double A, double B, double C, double D, double E, double F)
{
    public static PdfMatrix Identity { get; } = new(1, 0, 0, 1, 0, 0);

    /// <summary>This map followed by <paramref name="next"/>.</summary>
    public PdfMatrix Then(PdfMatrix next) => new(
        A * next.A + B * next.C, A * next.B + B * next.D,
        C * next.A + D * next.C, C * next.B + D * next.D,
        E * next.A + F * next.C + next.E, E * next.B + F * next.D + next.F);

    /// <summary>A move by (<paramref name="x"/>, <paramref name="y"/>) in this map's own space, then this map.</summary>
    public PdfMatrix Translated(double x, double y) => this with { E = x * A + y * C + E, F = x * B + y * D + F };
}

/// <summary>
/// A page's issues, counted by kind and detail in the order first met: nothing is made until the first comes, a few are
/// found by a scan of their list, and an index of where each is is made once there are more than a scan should pass.
/// </summary>
internal sealed class PdfIssues
{
    private const int ScanUpTo = 16;

    private List<(string Kind, string Detail, int Count)>? _issues;
    private Dictionary<(string Kind, string Detail), int>? _index;

    public void Add(string kind, string detail, int count = 1)
    {
        var issues = _issues ??= [];
        int at = IndexOf(issues, kind, detail);
        if (at >= 0)
        {
            var seen = issues[at];
            issues[at] = (seen.Kind, seen.Detail, seen.Count + count);
            return;
        }

        issues.Add((kind, detail, count));
        if (_index is not null)
        {
            _index[(kind, detail)] = issues.Count - 1;
        }
        else if (issues.Count > ScanUpTo)
        {
            _index = new Dictionary<(string Kind, string Detail), int>(issues.Count * 2);
            for (int i = 0; i < issues.Count; i++)
            {
                _index[(issues[i].Kind, issues[i].Detail)] = i;
            }
        }
    }

    private int IndexOf(List<(string Kind, string Detail, int Count)> issues, string kind, string detail)
    {
        if (_index is not null)
        {
            return _index.TryGetValue((kind, detail), out int at) ? at : -1;
        }

        for (int i = 0; i < issues.Count; i++)
        {
            if (string.Equals(issues[i].Kind, kind, StringComparison.Ordinal) && string.Equals(issues[i].Detail, detail, StringComparison.Ordinal))
            {
                return i;
            }
        }

        return -1;
    }

    /// <summary>The issues in the order first met; none: an empty list.</summary>
    public IReadOnlyList<DocumentIssue> ToList()
    {
        if (_issues is not { Count: > 0 } issues)
        {
            return [];
        }

        var list = new List<DocumentIssue>(issues.Count);
        foreach (var (kind, detail, count) in issues)
        {
            list.Add(new DocumentIssue(kind, detail, count));
        }

        return list;
    }
}

/// <summary>
/// The content-stream interpreter (ISO 32000-2, 8 and 9): the graphics state stack (<c>q Q cm</c>), text objects and
/// their state (<c>BT ET Tf Tm Td TD T* TL Tc Tw Tz Ts Tr</c>), showing (<c>Tj TJ ' "</c>), form XObjects drawn by
/// <c>Do</c> within <see cref="DocumentOptions.MaxDepth"/>, inline images (<c>BI ID EI</c>) passed over by their size or
/// their end, images (XObjects and inline) placed by the CTM when <see cref="ReadImages"/> asks for them, and marked
/// content whose <c>/ActualText</c> stands for its glyphs; other operators are passed over. Every
/// glyph is placed through the text matrix, the CTM and the page's crop box, rotation and unit, so each run's box is on
/// the page as shown. Runs, images and painted paths share one count, their place in the content. One reader reads one
/// page at a time, on one thread; the fonts it reads are kept by the document for its other pages. A document keeps a
/// reader from page to page (<see cref="Begin"/>, <see cref="End"/>): its working buffers (operands, glyphs, characters,
/// runs, the graphics state stack) come from the shared array pools for a page and go back at its end, and its font and
/// colour space caches are emptied for the next page; what a page is given (runs, paths, clips, images) is its own.
/// </summary>
internal sealed partial class PdfContentReader : IDisposable
{
    private const int MaxOperands = 1 << 16;

    // The operators, by up to three bytes.
    private const int OpSave = 'q', OpRestore = 'Q', OpQuote = '\'', OpDoubleQuote = '"', OpStroke = 'S', OpCloseStroke = 's', OpFill = 'f', OpFillOld = 'F',
        OpFillStroke = 'B', OpCloseFillStroke = 'b';
    private const int OpConcat = 'c' | 'm' << 8, OpBeginText = 'B' | 'T' << 8, OpEndText = 'E' | 'T' << 8, OpFont = 'T' | 'f' << 8, OpMove = 'T' | 'd' << 8,
        OpMoveLeading = 'T' | 'D' << 8, OpMatrix = 'T' | 'm' << 8, OpNextLine = 'T' | '*' << 8, OpLeading = 'T' | 'L' << 8, OpCharSpacing = 'T' | 'c' << 8,
        OpWordSpacing = 'T' | 'w' << 8, OpScale = 'T' | 'z' << 8, OpRise = 'T' | 's' << 8, OpMode = 'T' | 'r' << 8, OpShow = 'T' | 'j' << 8,
        OpShowArray = 'T' | 'J' << 8, OpDraw = 'D' | 'o' << 8, OpFillEvenOdd = 'f' | '*' << 8, OpFillStrokeEvenOdd = 'B' | '*' << 8,
        OpCloseFillStrokeEvenOdd = 'b' | '*' << 8, OpShade = 's' | 'h' << 8;
    private const int OpMarked = 'B' | 'M' << 8 | 'C' << 16, OpMarkedProperties = 'B' | 'D' << 8 | 'C' << 16, OpEndMarked = 'E' | 'M' << 8 | 'C' << 16;

    private readonly PdfFile _file;
    private int _page;
    private PdfIssues _issues = null!;                                     // set by Begin
    private bool _includeHidden;

    // The fonts the page has shown, by object number and by direct dictionary: emptied each page (a font's note is
    // counted on each page that shows it), the fonts themselves kept by the document.
    private readonly Dictionary<int, PdfFont> _fontsByObject = [];
    private readonly Dictionary<PdfDictionary, PdfFont> _fontsByDictionary = new(ReferenceEqualityComparer.Instance);
    private Glyph[] _glyphs = [];                                          // the page's glyphs, pooled (plain values: nothing to clear)
    private int _glyphCount;
    private PdfPooledList<Segment> _segments;                              // pooled lists: fields of their own, never readonly or copied
    private PdfPooledList<ReadRun> _runs;
    private PdfPooledList<State> _stack;
    private readonly HashSet<string> _missingFonts = [];
    private Operand[] _operands = [];                                      // pooled
    private int _operandCount;
    private byte[] _bytes = [];                                            // pooled
    private char[] _chars = [];                                            // pooled
    private int _charCount;
    private State _state;
    private int _skippedSaves;
    private PdfMatrix _tm = PdfMatrix.Identity, _tlm = PdfMatrix.Identity;
    private int _order;
    private int _marked;
    private MarkedSpan? _span;
    private bool _gapPending;
    private int _runStart = -1;
    private RunStyle _runStyle;
    private PdfPageImages? _images;

    /// <summary>A reader of page <paramref name="page"/> (from 1) of <paramref name="file"/>, read once and disposed.</summary>
    public PdfContentReader(PdfFile file, int page, PdfIssues issues, bool includeHidden)
    {
        _file = file;
        Begin(page, issues, includeHidden);
    }

    /// <summary>A reader a document keeps for its pages: each page is read between <see cref="Begin"/> and <see cref="End"/>.</summary>
    public PdfContentReader(PdfFile file) => _file = file;

    /// <summary>
    /// Starts page <paramref name="page"/> (from 1), its issues counted in <paramref name="issues"/>: whatever a page
    /// before left (one that failed part of the way) is let go, and the page's working buffers are rented.
    /// </summary>
    public void Begin(int page, PdfIssues issues, bool includeHidden)
    {
        End();
        (_page, _issues, _includeHidden) = (page, issues, includeHidden);
        _glyphs = ArrayPool<Glyph>.Shared.Rent(256);
        _operands = ArrayPool<Operand>.Shared.Rent(32);
        _bytes = ArrayPool<byte>.Shared.Rent(1024);
        _chars = ArrayPool<char>.Shared.Rent(4096);
    }

    /// <summary>
    /// Ends the page: its working buffers go back to the shared pools, its caches are emptied, and nothing it was given
    /// (runs, paths, clips, images) is held any more. Safe to call again.
    /// </summary>
    public void End()
    {
        Release(ref _glyphs);
        Release(ref _operands);
        Release(ref _bytes);
        Release(ref _chars);
        _segments.Return();
        _runs.Return();
        _stack.Return();
        _fontsByObject.Clear();
        _fontsByDictionary.Clear();
        _missingFonts.Clear();
        (_glyphCount, _charCount, _operandCount, _skippedSaves, _order, _marked, _runStart) = (0, 0, 0, 0, 0, 0, -1);
        _state = default;
        _tm = _tlm = PdfMatrix.Identity;
        _runStyle = default;
        _span = null;
        _gapPending = false;
        _images = null;
        _rewrite = null;
        EndVector();
        EndTags();
    }

    private static void Release<T>(ref T[] array)
    {
        if (array.Length > 0)
        {
            ArrayPool<T>.Shared.Return(array);   // plain values: nothing to clear
            array = [];
        }
    }

    /// <summary>The images read, in content order, when <see cref="ReadImages"/> was called before <see cref="Read"/> (else none).</summary>
    public IReadOnlyList<DocumentImage> Images => _images?.Images ?? [];

    /// <summary>Reads the page's images too (image XObjects at any depth of forms, and inline images), placed by the CTM.</summary>
    public void ReadImages() => _images ??= new PdfPageImages(_file, _page, _issues);

    /// <summary>
    /// Interprets the page's <paramref name="contents"/> (one stream or an array of them, read as one) with
    /// <paramref name="resources"/>; <paramref name="display"/> maps the page's user space onto the page as shown.
    /// A damaged stream is counted in the issues and read as far as it decodes; a limit passed fails the page.
    /// </summary>
    public void Read(PdfValue contents, PdfDictionary? resources, PdfMatrix display)
    {
        _state = new State { Ctm = display, Scale = 1 };
        StartVector(display);
        PdfValue resolved;
        try
        {
            resolved = _file.Resolve(contents);
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            _issues.Add(DocumentIssue.DamagedStream, $"page {_page}: its contents: {e.Message}");
            return;
        }

        // One stream (the most pages have) needs no list; an array of several is listed as it decodes.
        var parts = resolved.AsArray is { Count: > 1 } several ? new List<PdfDecoded>(several.Count) : null;
        PdfDecoded? one = null;
        PdfBuffer? joined = null;
        try
        {
            if (resolved.AsArray is { } array)
            {
                foreach (var part in array.Items)
                {
                    if (Decode(part) is { } decoded)
                    {
                        if (parts is null)
                        {
                            one = decoded;
                        }
                        else
                        {
                            parts.Add(decoded);
                        }
                    }
                }
            }
            else if (resolved.Kind == PdfKind.Stream)
            {
                one = Decode(resolved);
            }

            if (one is not null)
            {
                Run(one.Span, resources, 0);
            }
            else if (parts is { Count: 1 })
            {
                Run(parts[0].Span, resources, 0);
            }
            else if (parts is { Count: > 1 })
            {
                // Streams of one page are one content (an operator's operands may end one and its name start the next).
                long size = 0;
                foreach (var part in parts)
                {
                    size += part.Length + 1L;
                }

                joined = new PdfBuffer((int)Math.Min(size, int.MaxValue), _file.MaxStreamBytes, PdfLimitMessage.JoinedContent(_file, _page));
                foreach (var part in parts)
                {
                    joined.Write(part.Span);
                    joined.Write("\n"u8);
                }

                Run(joined.Span, resources, 0);
            }

            if (_span is not null)
            {
                EndSpan();   // a span the content never closes ends with it
            }
        }
        finally
        {
            joined?.Dispose();
            one?.Dispose();
            if (parts is not null)
            {
                foreach (var part in parts)
                {
                    part.Dispose();
                }
            }
        }
    }

    /// <summary>
    /// Interprets <paramref name="form"/>, a form XObject drawn on its own (an annotation's appearance), with
    /// <paramref name="placement"/> mapping its space onto the page as shown (its own <c>/Matrix</c> applied within, its
    /// <c>/BBox</c> a clip); its parts are read as a page's are, drawn in a group of <paramref name="opacity"/> when it is
    /// below 1 (an annotation's <c>/CA</c>).
    /// </summary>
    public void ReadForm(PdfStream form, PdfMatrix placement, float opacity = 1f)
    {
        _state = new State { Ctm = placement, Scale = 1 };
        StartVector(placement);
        if (_vector && opacity < 1f)
        {
            (_groups ??= []).Add(new DocumentGroup { Isolated = true, Alpha = Math.Clamp(opacity, 0f, 1f) });
            _state.Group = _groups.Count - 1;
        }

        DrawForm(form, "the appearance"u8, null, 0);
        if (_span is not null)
        {
            EndSpan();                                                         // a span the content never closes ends with it
        }
    }

    // One content stream decoded; a damaged one is counted (and read as far as it decodes), one that cannot be is skipped.
    private PdfDecoded? Decode(PdfValue part)
    {
        try
        {
            if (_file.Resolve(part).AsStream is not { } stream)
            {
                return null;
            }

            var decoded = _file.Decode(stream);
            if (decoded.Damaged)
            {
                _issues.Add(DocumentIssue.DamagedStream, $"page {_page}: object {stream.Number} is damaged part of the way");
            }

            return decoded;
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            _issues.Add(DocumentIssue.DamagedStream, $"page {_page}: {e.Message}");
            return null;
        }
    }

    public void Dispose() => End();

    // ---- the interpreter -----------------------------------------------------------------------------------------

    private void Run(ReadOnlySpan<byte> data, PdfDictionary? resources, int depth)
    {
        var lexer = new PdfContentLexer(data);
        _operandCount = 0;
        while (true)
        {
            var token = lexer.Next(out int start, out int length, out double number);
            if (token == PdfToken.End)
            {
                break;
            }

            if (token == PdfToken.Keyword)
            {
                var word = data.Slice(start, length);
                if (word.SequenceEqual("true"u8) || word.SequenceEqual("false"u8) || word.SequenceEqual("null"u8))
                {
                    Push(token, start, length, number);
                    continue;
                }

                if (_rewrite is not null && depth == 0)
                {
                    Rewrite(ref lexer, word, data, resources, start);          // a redaction: the operator copied or changed
                }
                else if (word.SequenceEqual("BI"u8))
                {
                    InlineImage(ref lexer, data, resources);
                }
                else
                {
                    Execute(Key(word), data, resources, depth);
                }

                _operandCount = 0;
                continue;
            }

            Push(token, start, length, number);
        }

        _operandCount = 0;
    }

    private void Push(PdfToken kind, int start, int length, double number)
    {
        if (_operandCount == MaxOperands)
        {
            _operandCount = 0;   // operands no operator takes: damaged content, dropped
        }

        if (_operandCount == _operands.Length)
        {
            var larger = ArrayPool<Operand>.Shared.Rent(Math.Max(32, _operands.Length * 2));
            _operands.AsSpan(0, _operandCount).CopyTo(larger);
            Release(ref _operands);
            _operands = larger;
        }

        _operands[_operandCount++] = new Operand(kind, start, length, number);
    }

    private static int Key(ReadOnlySpan<byte> word) => word.Length switch
    {
        1 => word[0],
        2 => word[0] | word[1] << 8,
        3 => word[0] | word[1] << 8 | word[2] << 16,
        _ => -1,
    };

    private void Execute(int op, ReadOnlySpan<byte> data, PdfDictionary? resources, int depth)
    {
        Span<double> n = stackalloc double[6];
        switch (op)
        {
            case OpSave:
                if (_stack.Count >= 16 * _file.MaxDepth)
                {
                    _skippedSaves++;   // nesting past what a page needs: kept as a count, so its Q is matched
                }
                else
                {
                    _stack.Add(_state);
                }

                break;
            case OpRestore:
                if (_skippedSaves > 0)
                {
                    _skippedSaves--;
                }
                else if (_stack.Count > 0)
                {
                    _state = _stack[^1];
                    _stack.Truncate(_stack.Count - 1);
                }

                break;
            case OpConcat:
                if (Numbers(6, n))
                {
                    _state.Ctm = new PdfMatrix(n[0], n[1], n[2], n[3], n[4], n[5]).Then(_state.Ctm);
                }

                break;
            case OpBeginText:
                _tm = _tlm = PdfMatrix.Identity;
                (_textClipFirst, _textClipCount) = (-1, 0);
                break;
            case OpEndText:
                EndTextClip();
                break;
            case OpFont:
                if (_operandCount >= 2 && _operands[_operandCount - 2].Kind == PdfToken.Name && _operands[_operandCount - 1].Kind == PdfToken.Number)
                {
                    var name = _operands[_operandCount - 2];
                    SelectFont(data.Slice(name.Start, name.Length), _operands[_operandCount - 1].Number, resources);
                }

                break;
            case OpMove:
                if (Numbers(2, n))
                {
                    MoveLine(n[0], n[1]);
                }

                break;
            case OpMoveLeading:
                if (Numbers(2, n))
                {
                    _state.Leading = -n[1];
                    MoveLine(n[0], n[1]);
                }

                break;
            case OpMatrix:
                if (Numbers(6, n))
                {
                    _tm = _tlm = new PdfMatrix(n[0], n[1], n[2], n[3], n[4], n[5]);
                }

                break;
            case OpNextLine:
                MoveLine(0, -_state.Leading);
                break;
            case OpLeading:
                if (Numbers(1, n))
                {
                    _state.Leading = n[0];
                }

                break;
            case OpCharSpacing:
                if (Numbers(1, n))
                {
                    _state.CharSpacing = n[0];
                }

                break;
            case OpWordSpacing:
                if (Numbers(1, n))
                {
                    _state.WordSpacing = n[0];
                }

                break;
            case OpScale:
                if (Numbers(1, n))
                {
                    _state.Scale = n[0] / 100;
                }

                break;
            case OpRise:
                if (Numbers(1, n))
                {
                    _state.Rise = n[0];
                }

                break;
            case OpMode:
                if (Numbers(1, n))
                {
                    _state.Mode = (int)Math.Clamp(n[0], 0, 7);
                }

                break;
            case OpShow:
                if (_operandCount >= 1 && IsString(_operands[_operandCount - 1]))
                {
                    ShowOne(data, _operands[_operandCount - 1]);
                }

                break;
            case OpQuote:
                MoveLine(0, -_state.Leading);
                if (_operandCount >= 1 && IsString(_operands[_operandCount - 1]))
                {
                    ShowOne(data, _operands[_operandCount - 1]);
                }

                break;
            case OpDoubleQuote:
                if (_operandCount >= 3 && _operands[_operandCount - 3].Kind == PdfToken.Number && _operands[_operandCount - 2].Kind == PdfToken.Number)
                {
                    _state.WordSpacing = _operands[_operandCount - 3].Number;
                    _state.CharSpacing = _operands[_operandCount - 2].Number;
                }

                MoveLine(0, -_state.Leading);
                if (_operandCount >= 1 && IsString(_operands[_operandCount - 1]))
                {
                    ShowOne(data, _operands[_operandCount - 1]);
                }

                break;
            case OpShowArray:
                ShowArray(data);
                break;
            case OpDraw:
                if (_operandCount >= 1 && _operands[_operandCount - 1].Kind == PdfToken.Name)
                {
                    var name = _operands[_operandCount - 1];
                    Draw(data.Slice(name.Start, name.Length), resources, depth);
                }

                break;
            case OpMarked:
                _marked++;
                Tagging(data, properties: false, resources);
                break;
            case OpMarkedProperties:
                _marked++;
                if (_span is null && _operandCount >= 2)
                {
                    BeginSpan(data, _operands[_operandCount - 1], resources);
                }

                Tagging(data, properties: true, resources);
                break;
            case OpEndMarked:
                if (_span is { } span && span.Depth == _marked)
                {
                    EndSpan();
                }

                Untag();
                _marked = Math.Max(0, _marked - 1);
                break;
            case OpStroke or OpCloseStroke or OpFill or OpFillOld or OpFillEvenOdd or OpFillStroke or OpFillStrokeEvenOdd or OpCloseFillStroke or OpCloseFillStrokeEvenOdd
                or OpShade:
                Paint(op, data, resources);   // a painted path or a shading takes its place in the content
                break;
            default:
                if (_vector)
                {
                    Vector(op, data, resources, depth);
                }

                break;
        }
    }

    private bool Numbers(int count, Span<double> values)
    {
        if (_operandCount < count)
        {
            return false;
        }

        for (int i = 0; i < count; i++)
        {
            var operand = _operands[_operandCount - count + i];
            if (operand.Kind != PdfToken.Number || !double.IsFinite(operand.Number))
            {
                return false;
            }

            values[i] = operand.Number;
        }

        return true;
    }

    private static bool IsString(Operand operand) => operand.Kind is PdfToken.LiteralString or PdfToken.HexString;

    // Td: a new line, (x, y) from the start of the current one.
    private void MoveLine(double x, double y) => _tm = _tlm = _tlm.Translated(x, y);

    // A string operand's bytes, decoded into the reader's buffer (valid until the next string is decoded).
    private ReadOnlySpan<byte> StringBytes(ReadOnlySpan<byte> data, Operand operand)
    {
        var raw = data.Slice(operand.Start, operand.Length);
        int needed = operand.Kind == PdfToken.HexString ? raw.Length / 2 + 1 : raw.Length;
        if (_bytes.Length < needed)
        {
            ArrayPool<byte>.Shared.Return(_bytes);
            _bytes = [];
            _bytes = ArrayPool<byte>.Shared.Rent(needed);
        }

        int count = operand.Kind == PdfToken.HexString ? PdfContentLexer.DecodeHex(raw, _bytes) : PdfContentLexer.DecodeLiteral(raw, _bytes);
        return _bytes.AsSpan(0, count);
    }

    // A name operand's bytes with #xx decoded (into a buffer of the caller's when it holds an escape).
    private static ReadOnlySpan<byte> NameBytes(ReadOnlySpan<byte> raw, Span<byte> buffer)
    {
        if (raw.IndexOf((byte)'#') < 0 || raw.Length > buffer.Length)
        {
            return raw;
        }

        return buffer[..PdfContentLexer.DecodeName(raw, buffer)];
    }

    // ---- fonts ---------------------------------------------------------------------------------------------------

    private void SelectFont(ReadOnlySpan<byte> rawName, double size, PdfDictionary? resources)
    {
        Span<byte> buffer = stackalloc byte[128];
        var name = NameBytes(rawName, buffer);
        _state.Size = double.IsFinite(size) ? size : 0;
        _state.Font = null;
        PdfValue entry;
        PdfName? key;
        try
        {
            entry = Lookup(resources, "Font"u8, name, out key);
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            _state.FontLabel = PdfText.Utf8OrLatin1(name);
            _issues.Add(DocumentIssue.DamagedStream, $"page {_page}: the font /{_state.FontLabel}: {e.Message}");
            return;
        }

        // The resources' own key names the font (its text made once, kept with the key); a name they lack is made here.
        _state.FontLabel = key?.ToString() ?? PdfText.Utf8OrLatin1(name);
        if (!entry.IsNull)
        {
            _state.Font = Font(entry, _state.FontLabel);
        }
    }

    private PdfValue Lookup(PdfDictionary? resources, ReadOnlySpan<byte> category, ReadOnlySpan<byte> name) => Lookup(resources, category, name, out _);

    // The resources' entry `name` of `category` (unresolved), with its key; null and none when there is none.
    private PdfValue Lookup(PdfDictionary? resources, ReadOnlySpan<byte> category, ReadOnlySpan<byte> name, out PdfName? key)
    {
        key = null;
        if (resources is null || _file.Dictionary(resources[category]) is not { } group)
        {
            return PdfValue.Null;
        }

        int at = group.IndexOf(name);
        if (at < 0)
        {
            return PdfValue.Null;
        }

        key = group.KeyAt(at);
        return group.ValueAt(at);
    }

    // A font resource, read once per page and kept by the document for its other pages.
    private PdfFont Font(PdfValue entry, string label)
    {
        int number = entry.IsReference ? entry.AsReference.Number : -1;
        if (number >= 0)
        {
            if (_fontsByObject.TryGetValue(number, out var known))
            {
                return known;
            }

            if (_file.Part(PdfFile.FontKey(number)) is PdfFont kept)
            {
                _fontsByObject[number] = kept;
                if (kept.Note is { } keptNote)
                {
                    _issues.Add(DocumentIssue.NotSupported, $"page {_page}: the font /{label}: {keptNote}");
                }

                return kept;
            }
        }

        PdfFont font;
        PdfDictionary? dictionary = null;
        try
        {
            dictionary = _file.Dictionary(entry);
            if (dictionary is not null && number < 0 && _fontsByDictionary.TryGetValue(dictionary, out var direct))
            {
                return direct;
            }

            font = dictionary is null ? PdfFonts.Fallback : PdfFonts.Read(_file, dictionary);
            if (dictionary is null)
            {
                _issues.Add(DocumentIssue.MissingFont, $"page {_page}: the font /{label} is not a font dictionary: its text is read as Helvetica");
            }
        }
        catch (Exception e) when (e is not (OutOfMemoryException or OperationCanceledException or InsufficientExecutionStackException) && !PdfLimit.Is(e))
        {
            _issues.Add(DocumentIssue.DamagedStream, $"page {_page}: the font /{label} cannot be read ({e.Message}): its text is read as Helvetica");
            font = PdfFonts.Fallback;
        }

        if (number >= 0)
        {
            _fontsByObject[number] = font;
            _file.KeepPart(PdfFile.FontKey(number), font, font.Bytes);
        }
        else if (dictionary is not null)
        {
            _fontsByDictionary[dictionary] = font;
        }

        if (font.Note is { } note)
        {
            _issues.Add(DocumentIssue.NotSupported, $"page {_page}: the font /{label}: {note}");
        }

        return font;
    }

    // The font text is shown in: the one set, or (when none is, or the page lacks it) Helvetica, counted.
    private PdfFont CurrentFont()
    {
        if (_state.Font is { } font)
        {
            return font;
        }

        string label = _state.FontLabel ?? "(none)";
        if (_missingFonts.Add(label))
        {
            _issues.Add(DocumentIssue.MissingFont, _state.FontLabel is null
                ? $"page {_page}: text shown before any font is set: read as Helvetica"
                : $"page {_page}: the font /{label} is not in the page's resources: its text is read as Helvetica");
        }

        return PdfFonts.Fallback;
    }

    // ---- showing text --------------------------------------------------------------------------------------------

    private void ShowOne(ReadOnlySpan<byte> data, Operand text)
    {
        BeginRun();
        ShowString(StringBytes(data, text));
        EndRun();
    }

    private void ShowArray(ReadOnlySpan<byte> data)
    {
        // The last array of the operands: strings shown, numbers moving the pen back (thousandths of the size).
        int end = _operandCount - 1;
        while (end >= 0 && _operands[end].Kind != PdfToken.ArrayEnd)
        {
            end--;
        }

        int start = end - 1;
        while (start >= 0 && _operands[start].Kind != PdfToken.ArrayStart)
        {
            start--;
        }

        if (start < 0)
        {
            return;
        }

        BeginRun();
        for (int i = start + 1; i < end; i++)
        {
            var item = _operands[i];
            if (item.Kind == PdfToken.Number && double.IsFinite(item.Number))
            {
                Adjust(item.Number);
            }
            else if (IsString(item))
            {
                ShowString(StringBytes(data, item));
            }
        }

        EndRun();
    }

    private void BeginRun()
    {
        var font = CurrentFont();
        var p = _tm.Then(_state.Ctm);
        double size = _state.Size, scale = _state.Scale;
        _segments.Add(new Segment(size * scale * p.A, size * scale * p.B, size * p.C, size * p.D, font.Ascent, font.Descent, font.Vertical, font));
        _runStart = _glyphCount;
        _runStyle = new RunStyle(font, _state.Mode, (float)(Math.Abs(size) * Math.Sqrt(p.C * p.C + p.D * p.D)), _segments.Count - 1, _vector ? Snapshot() : -1);
        _gapPending = false;
    }

    private void ShowString(ReadOnlySpan<byte> bytes)
    {
        var font = _runStyle.Font;
        var p = _tm.Then(_state.Ctm);
        double size = _state.Size, scale = _state.Scale, rise = _state.Rise;
        int segment = _segments.Count - 1;
        for (int i = 0; i < bytes.Length;)
        {
            int length = Math.Max(1, font.NextCode(bytes, i, out uint code));
            i += length;
            double width = font.Width(code, length);
            bool space = font.IsWordSpace(code, length);
            if (_chars.Length - _charCount < PdfFontTables.MaxGlyphText * 4)
            {
                GrowChars();
            }

            int count = font.Text(code, length, _chars.AsSpan(_charCount, PdfFontTables.MaxGlyphText * 4));
            bool unmapped = count == 0;
            if (unmapped)
            {
                _chars[_charCount] = '�';
                count = 1;
            }

            double tx, ty, height = 0;
            if (font.Vertical)
            {
                height = font.VerticalAdvance(code, length);
                (tx, ty) = (0, height * size + _state.CharSpacing + (space ? _state.WordSpacing : 0));
            }
            else
            {
                (tx, ty) = ((width * size + _state.CharSpacing + (space ? _state.WordSpacing : 0)) * scale, 0);
            }

            if (_glyphCount == _glyphs.Length)
            {
                var larger = ArrayPool<Glyph>.Shared.Rent(_glyphs.Length * 2);
                _glyphs.AsSpan(0, _glyphCount).CopyTo(larger);
                ArrayPool<Glyph>.Shared.Return(_glyphs);
                _glyphs = larger;
            }

            _glyphs[_glyphCount++] = new Glyph
            {
                Segment = segment, TextStart = _charCount, TextLength = count, X = p.E + rise * p.C, Y = p.F + rise * p.D, Width = width, Height = height,
                AdvanceX = tx * p.A + ty * p.C, AdvanceY = tx * p.B + ty * p.D, Code = code, CodeLength = length, Gap = _gapPending, Unmapped = unmapped,
            };
            _charCount += count;
            _gapPending = false;
            p = p.Translated(tx, ty);
            _tm = _tm.Translated(tx, ty);
        }
    }

    private void GrowChars()
    {
        char[] larger = ArrayPool<char>.Shared.Rent(Math.Max(_chars.Length * 2, _charCount + PdfFontTables.MaxGlyphText * 8));
        _chars.AsSpan(0, _charCount).CopyTo(larger);
        ArrayPool<char>.Shared.Return(_chars);
        _chars = larger;
    }

    // A TJ number: the pen moves back by it (thousandths of the size); a move is a gap a layout may read as a word break.
    private void Adjust(double amount)
    {
        double move = -amount / 1000 * _state.Size;
        _tm = _runStyle.Font.Vertical ? _tm.Translated(0, move) : _tm.Translated(move * _state.Scale, 0);
        _gapPending |= amount != 0;
    }

    private void EndRun()
    {
        int count = _glyphCount - _runStart;
        if (_span is { } span)
        {
            span.Style ??= _runStyle;
            return;   // the span's run is made when it ends
        }

        if (count > 0)
        {
            AddRun(_runStart, count, _runStyle, null);
        }
    }

    private void AddRun(int first, int count, RunStyle style, string? actualText)
    {
        int order = _order++;
        ClipRun(order, style.Mode);
        bool hidden = style.Mode is 3 or 7;
        if (hidden && !_includeHidden)
        {
            return;
        }

        _runs.Add(new ReadRun(first, count, order, style, actualText));
        _tags.Add(_tag);
    }

    // ---- marked content ------------------------------------------------------------------------------------------

    // A span whose properties (inline, or named in the resources' /Properties) give /ActualText: its glyphs make one run
    // with that text.
    private void BeginSpan(ReadOnlySpan<byte> data, Operand properties, PdfDictionary? resources)
    {
        string? text = null;
        try
        {
            if (properties.Kind == PdfToken.Dictionary)
            {
                var raw = data.Slice(properties.Start, properties.Length);
                if (raw.IndexOf("ActualText"u8) >= 0)
                {
                    var parser = new PdfParser(raw, _file.MaxDepth, complete: true);
                    text = ActualText(parser.ReadValue().AsDictionary);
                }
            }
            else if (properties.Kind == PdfToken.Name)
            {
                Span<byte> buffer = stackalloc byte[128];
                var name = NameBytes(data.Slice(properties.Start, properties.Length), buffer);
                text = ActualText(_file.Dictionary(Lookup(resources, "Properties"u8, name)));
            }
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            _issues.Add(DocumentIssue.DamagedStream, $"page {_page}: a marked-content span's properties: {e.Message}");
        }

        if (text is not null)
        {
            _span = new MarkedSpan(_marked, text, _glyphCount);
        }
    }

    private string? ActualText(PdfDictionary? properties) =>
        properties is not null && _file.Resolve(properties["ActualText"u8]).AsString is { } text ? PdfText.Decode(text.Bytes) : null;

    private void EndSpan()
    {
        var span = _span!;
        _span = null;
        int count = _glyphCount - span.FirstGlyph;
        if (count > 0 && span.Style is { } style)
        {
            AddRun(span.FirstGlyph, count, style, span.Text);
        }
    }

    // ---- XObjects and inline images ------------------------------------------------------------------------------

    private void Draw(ReadOnlySpan<byte> rawName, PdfDictionary? resources, int depth)
    {
        Span<byte> buffer = stackalloc byte[128];
        var name = NameBytes(rawName, buffer);
        PdfStream? stream;
        try
        {
            stream = _file.Resolve(Lookup(resources, "XObject"u8, name)).AsStream;
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            _issues.Add(DocumentIssue.DamagedStream, $"page {_page}: the XObject /{PdfText.Utf8OrLatin1(name)}: {e.Message}");
            return;
        }

        var subtype = stream is null ? null : _file.Resolve(stream.Dictionary["Subtype"u8]).AsName;
        if (subtype is null)
        {
            return;
        }

        if (subtype.Is("Image"u8))
        {
            int order = _order++;   // an image takes its place in the content
            _images?.Add(stream!, name, resources, _state.Ctm, order, ImageStyle());
            return;
        }

        if (!subtype.Is("Form"u8))
        {
            return;
        }

        DrawForm(stream!, name, resources, depth);
    }

    // A form XObject's content run under the current state: its /Matrix, its /BBox as a clip, its group, its resources
    // (else the caller's), within MaxDepth.
    private void DrawForm(PdfStream stream, ReadOnlySpan<byte> name, PdfDictionary? resources, int depth)
    {
        if (depth + 1 > _file.MaxDepth)
        {
            throw PdfLimit.Error(string.Create(CultureInfo.InvariantCulture,
                $"forms drawn within forms {depth + 1} levels deep, deeper than the limit {nameof(DocumentOptions.MaxDepth)} (DocumentOptions.MaxDepth, {_file.MaxDepth}); a form that draws itself never ends"));
        }

        if (!RuntimeHelpers.TryEnsureSufficientExecutionStack())
        {
            throw PdfLimit.Error(string.Create(CultureInfo.InvariantCulture,
                $"forms drawn within forms {depth + 1} levels deep, deeper than this thread's stack holds; lower the limit {nameof(DocumentOptions.MaxDepth)} (DocumentOptions.MaxDepth, {_file.MaxDepth} now)"));
        }

        PdfDecoded decoded;
        try
        {
            decoded = _file.Decode(stream);
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            _issues.Add(DocumentIssue.DamagedStream, $"page {_page}: the form /{PdfText.Utf8OrLatin1(name)}: {e.Message}");
            return;
        }

        using (decoded)
        {
            if (decoded.Damaged)
            {
                _issues.Add(DocumentIssue.DamagedStream, $"page {_page}: object {stream.Number} is damaged part of the way");
            }

            var formResources = _file.Dictionary(stream.Dictionary["Resources"u8]) ?? resources;
            var matrix = Matrix(stream.Dictionary["Matrix"u8]);
            var (state, saved, skipped, tm, tlm) = (_state, _stack.Count, _skippedSaves, _tm, _tlm);
            _stack.Add(_state);
            _state.Ctm = matrix.Then(_state.Ctm);
            EnterForm(stream.Dictionary);
            try
            {
                Run(decoded.Span, formResources, depth + 1);
            }
            finally
            {
                _stack.Truncate(saved);
                (_state, _skippedSaves, _tm, _tlm) = (state, skipped, tm, tlm);
            }
        }
    }

    private PdfMatrix Matrix(PdfValue value)
    {
        if (_file.Resolve(value).AsArray is not { Count: 6 } array)
        {
            return PdfMatrix.Identity;
        }

        Span<double> n = stackalloc double[6];
        for (int i = 0; i < 6; i++)
        {
            if (!_file.Resolve(array[i]).TryNumber(out n[i]) || !double.IsFinite(n[i]))
            {
                return PdfMatrix.Identity;
            }
        }

        return new PdfMatrix(n[0], n[1], n[2], n[3], n[4], n[5]);
    }

    // BI, its entries, ID, the data, EI: the data passed over by its size when it is unfiltered and the size is known,
    // else up to an "EI" between whitespace and followed by content-like bytes. The image takes its place in the content
    // (and, when images are read, its entries and data go to them).
    private void InlineImage(ref PdfContentLexer lexer, ReadOnlySpan<byte> data, PdfDictionary? resources)
    {
        long width = -1, height = -1, bits = -1, components = -1;
        bool filtered = false, mask = false;
        int entries = lexer.Position, entriesEnd;
        while (true)
        {
            var token = lexer.Next(out int start, out int length, out _);
            if (token == PdfToken.End)
            {
                return;
            }

            if (token == PdfToken.Keyword && data.Slice(start, length).SequenceEqual("ID"u8))
            {
                entriesEnd = start;
                break;
            }

            if (token != PdfToken.Name)
            {
                continue;
            }

            var key = data.Slice(start, length);
            var value = lexer.Next(out int valueStart, out int valueLength, out double number);
            var word = data.Slice(valueStart, valueLength);
            if (value == PdfToken.ArrayStart)
            {
                // An array value: a filter chain, or an indexed colour space ([/I base hival lookup]).
                bool first = true;
                while ((value = lexer.Next(out valueStart, out valueLength, out _)) is not (PdfToken.ArrayEnd or PdfToken.End))
                {
                    if (first && (key.SequenceEqual("CS"u8) || key.SequenceEqual("ColorSpace"u8)))
                    {
                        components = 1;   // an indexed space: one index a pixel
                    }

                    first = false;
                }

                if (key.SequenceEqual("F"u8) || key.SequenceEqual("Filter"u8))
                {
                    filtered = true;
                }

                continue;
            }

            if (key.SequenceEqual("W"u8) || key.SequenceEqual("Width"u8))
            {
                width = value == PdfToken.Number ? (long)number : -1;
            }
            else if (key.SequenceEqual("H"u8) || key.SequenceEqual("Height"u8))
            {
                height = value == PdfToken.Number ? (long)number : -1;
            }
            else if (key.SequenceEqual("BPC"u8) || key.SequenceEqual("BitsPerComponent"u8))
            {
                bits = value == PdfToken.Number ? (long)number : -1;
            }
            else if (key.SequenceEqual("F"u8) || key.SequenceEqual("Filter"u8))
            {
                filtered = true;
            }
            else if (key.SequenceEqual("IM"u8) || key.SequenceEqual("ImageMask"u8))
            {
                mask = value == PdfToken.Keyword && word.SequenceEqual("true"u8);
            }
            else if ((key.SequenceEqual("CS"u8) || key.SequenceEqual("ColorSpace"u8)) && value == PdfToken.Name)
            {
                components = word.SequenceEqual("G"u8) || word.SequenceEqual("DeviceGray"u8) || word.SequenceEqual("I"u8) || word.SequenceEqual("Indexed"u8) ? 1
                    : word.SequenceEqual("RGB"u8) || word.SequenceEqual("DeviceRGB"u8) ? 3
                    : word.SequenceEqual("CMYK"u8) || word.SequenceEqual("DeviceCMYK"u8) ? 4
                    : -1;
            }
        }

        int order = _order++;
        int dataStart = lexer.Position;
        if (dataStart < data.Length && PdfParser.IsWhitespace(data[dataStart]))
        {
            dataStart++;   // the one whitespace byte after ID
        }

        if (mask)
        {
            (components, bits) = (1, 1);
        }

        if (!filtered && width > 0 && height > 0 && bits > 0 && components > 0 && width < int.MaxValue && height < int.MaxValue)
        {
            long size = (width * components * bits + 7) / 8 * height;
            if (size >= 0 && dataStart + size <= data.Length && EndAt(data, (int)(dataStart + size)) is int after and >= 0)
            {
                lexer.Position = after;
                _images?.AddInline(data[entries..entriesEnd], data.Slice(dataStart, (int)size), resources, _state.Ctm, order, ImageStyle());
                return;
            }
        }

        int at = dataStart;
        while (true)
        {
            int found = data[at..].IndexOf("EI"u8);
            if (found < 0)
            {
                lexer.Position = data.Length;   // no end: the rest is the image's
                _images?.AddInline(data[entries..entriesEnd], data[Math.Min(dataStart, data.Length)..], resources, _state.Ctm, order, ImageStyle());
                return;
            }

            int p = at + found;
            if (p > dataStart && PdfParser.IsWhitespace(data[p - 1]) && (p + 2 == data.Length || PdfParser.IsWhitespace(data[p + 2]) || PdfParser.IsDelimiter(data[p + 2]))
                && LooksLikeContent(data[(p + 2)..]))
            {
                lexer.Position = p + 2;
                _images?.AddInline(data[entries..entriesEnd], data[dataStart..(p - 1)], resources, _state.Ctm, order, ImageStyle());   // the whitespace before EI is not the image's
                return;
            }

            at = p + 1;
        }
    }

    // Past "EI" when only whitespace lies between `at` and it, and it ends there; else -1.
    private static int EndAt(ReadOnlySpan<byte> data, int at)
    {
        while (at < data.Length && PdfParser.IsWhitespace(data[at]))
        {
            at++;
        }

        bool ends = at + 2 <= data.Length && data.Slice(at, 2).SequenceEqual("EI"u8)
                    && (at + 2 == data.Length || PdfParser.IsWhitespace(data[at + 2]) || PdfParser.IsDelimiter(data[at + 2]));
        return ends ? at + 2 : -1;
    }

    // What follows an inline image's end: text, not binary (the next bytes are printable or whitespace).
    private static bool LooksLikeContent(ReadOnlySpan<byte> rest)
    {
        foreach (byte b in rest[..Math.Min(rest.Length, 16)])
        {
            if (b > 0x7E || (b < 0x20 && !PdfParser.IsWhitespace(b)))
            {
                return false;
            }
        }

        return true;
    }

    // ---- what is read --------------------------------------------------------------------------------------------

    /// <summary>The page's runs as the contract gives them, with each run's glyphs and face when <paramref name="glyphs"/>.</summary>
    public DocumentTextRun[] TextRuns(bool glyphs, out int unmapped, out string? unmappedFonts)
    {
        var runs = new DocumentTextRun[_runs.Count];
        unmapped = 0;
        string? font = null;                                                   // the fonts of glyphs with no text: the first,
        List<string>? fonts = null;                                            // then a list once another comes
        for (int r = 0; r < runs.Length; r++)
        {
            var run = _runs[r];
            var first = _glyphs[run.First];
            var last = _glyphs[run.First + run.Count - 1];
            string text = run.ActualText ?? new string(_chars, first.TextStart, last.TextStart + last.TextLength - first.TextStart);
            var box = DisplayBox(run.First, run.Count);
            var segment = _segments[first.Segment];
            bool upright = !segment.Vertical && segment.A > 0 && Math.Abs(segment.B) <= 1e-6 * segment.A;
            if (run.ActualText is null)
            {
                int missing = 0;
                for (int g = run.First; g < run.First + run.Count; g++)
                {
                    missing += _glyphs[g].Unmapped ? 1 : 0;
                }

                if (missing > 0)
                {
                    unmapped += missing;
                    string name = run.Style.Font.Name ?? "a font with no name";
                    if (font is null)
                    {
                        font = name;
                    }
                    else if (fonts is not null ? !fonts.Contains(name) : !string.Equals(font, name, StringComparison.Ordinal))
                    {
                        (fonts ??= [font]).Add(name);
                    }
                }
            }

            var paint = run.Style.Paint >= 0 ? _paints[run.Style.Paint] : new RunPaint(DocumentColour.Black, DocumentColour.Black, 1f, -1, DocumentBlend.Normal, -1, -1);
            runs[r] = new DocumentTextRun(text, box, run.Style.Size)
            {
                Font = run.Style.Font.Name,
                Mode = run.Style.Mode switch { 1 or 5 => DocumentTextMode.Stroke, 2 or 6 => DocumentTextMode.FillStroke, 3 or 7 => DocumentTextMode.Hidden, _ => DocumentTextMode.Fill },
                Order = run.Order,
                Baseline = upright ? (float)first.Y : null,
                Logical = run.ActualText is not null,
                Glyphs = glyphs ? Glyphs(run) : default,
                Face = glyphs ? run.Style.Font.Face : null,
                Colour = paint.Fill, StrokeColour = paint.Stroke, StrokeWidth = paint.StrokeWidth, Clip = paint.Clip, Blend = paint.Blend, Group = paint.Group, SoftMask = paint.SoftMask,
                GlyphMatrix = glyphs ? new DocumentMatrix((float)segment.A, (float)segment.B, (float)segment.C, (float)segment.D, 0f, 0f) : null,
                Tag = _tags[r],
            };
        }

        // Each font once, in ordinal order (one font: its name as it is).
        fonts?.Sort(StringComparer.Ordinal);
        unmappedFonts = fonts is not null ? string.Join(", ", fonts) : font;
        return runs;
    }

    // Each glyph's id and origin: a vertical glyph's origin moved by its position vector (its /W2 entry's, else half its
    // width across and the font's vertical origin up) to where its outline is drawn from.
    private DocumentGlyph[] Glyphs(ReadRun run)
    {
        var glyphs = new DocumentGlyph[run.Count];
        for (int i = 0; i < glyphs.Length; i++)
        {
            var g = _glyphs[run.First + i];
            var s = _segments[g.Segment];
            double x = g.X, y = g.Y;
            if (s.Vertical)
            {
                var (vx, vy) = s.Font.Position(g.Code, g.CodeLength, g.Width);
                (x, y) = (x - (s.A * vx + s.C * vy), y - (s.B * vx + s.D * vy));
            }

            glyphs[i] = new DocumentGlyph(s.Font.GlyphId(g.Code, g.CodeLength), (float)x, (float)y, (float)Math.Sqrt(g.AdvanceX * g.AdvanceX + g.AdvanceY * g.AdvanceY));
        }

        return glyphs;
    }

    // The box around glyphs [first, first + count) on the page as shown.
    private DocumentBox DisplayBox(int first, int count) => Bounds(first, count, 1, 0);

    // The box around glyphs [first, first + count) in a frame turned by an angle whose cosine and sine are given (1, 0: the page).
    private DocumentBox Bounds(int first, int count, double cos, double sin)
    {
        double minX = double.PositiveInfinity, minY = double.PositiveInfinity, maxX = double.NegativeInfinity, maxY = double.NegativeInfinity;
        Span<double> corners = stackalloc double[8];
        for (int g = first; g < first + count; g++)
        {
            Corners(_glyphs[g], corners);
            for (int k = 0; k < 8; k += 2)
            {
                double x = corners[k] * cos + corners[k + 1] * sin, y = -corners[k] * sin + corners[k + 1] * cos;
                (minX, maxX, minY, maxY) = (Math.Min(minX, x), Math.Max(maxX, x), Math.Min(minY, y), Math.Max(maxY, y));
            }
        }

        return double.IsFinite(minX) && double.IsFinite(minY) && double.IsFinite(maxX) && double.IsFinite(maxY)
            ? new DocumentBox((float)minX, (float)minY, (float)(maxX - minX), (float)(maxY - minY))
            : DocumentBox.Empty;
    }

    // A glyph's four corners on the page: along the baseline its width, across it the font's descent to ascent (raised
    // by the rise); a vertical glyph its width across the column from its position vector's -vx (centred unless /W2
    // gives one) and its vertical advance down it.
    private void Corners(in Glyph g, Span<double> xy)
    {
        var s = _segments[g.Segment];
        double x0, x1, y0, y1;
        if (s.Vertical)
        {
            double vx = s.Font.Position(g.Code, g.CodeLength, g.Width).X;
            (x0, x1, y0, y1) = (-vx, g.Width - vx, g.Height, 0);
        }
        else
        {
            (x0, x1, y0, y1) = (0, g.Width, s.Descent, s.Ascent);
        }

        xy[0] = g.X + x0 * s.A + y0 * s.C;
        xy[1] = g.Y + x0 * s.B + y0 * s.D;
        xy[2] = g.X + x1 * s.A + y0 * s.C;
        xy[3] = g.Y + x1 * s.B + y0 * s.D;
        xy[4] = g.X + x1 * s.A + y1 * s.C;
        xy[5] = g.Y + x1 * s.B + y1 * s.D;
        xy[6] = g.X + x0 * s.A + y1 * s.C;
        xy[7] = g.Y + x0 * s.B + y1 * s.D;
    }

    /// <summary>
    /// The page's lines and text through the layout the options name. The layout is given the runs as pieces in the frame
    /// of the page's main text direction (turned text, a turned page and vertical columns read as lines): a run splits where
    /// a TJ offset moves the pen (a gap the layout's word-gap share judges) and around a glyph that stands for several
    /// characters (a piece in logical order, so a ligature keeps its characters' order in right-to-left text); an
    /// /ActualText span is one logical piece. A run that is one piece already (its text, box, baseline, size, order, tag
    /// and direction what its piece would have: upright text on a page read along its width) is given as itself, and a
    /// piece that is a whole run shares its text, so a page's text is made once. The lines' runs are given back as the
    /// page's runs.
    /// </summary>
    public DocumentText Arrange(DocumentTextRun[] runs, float width, float height, DocumentOptions options)
    {
        if (runs.Length == 0)
        {
            return DocumentText.Empty;
        }

        var (cos, sin) = MainDirection();
        var pieces = new List<DocumentTextRun>(runs.Length);
        var owners = new PdfPooledList<int>();                                 // each piece's run, pooled
        try
        {
            for (int r = 0; r < runs.Length; r++)
            {
                var run = _runs[r];
                if (run.ActualText is { } actual)
                {
                    AddPiece(pieces, ref owners, runs, r, actual, run.First, run.Count, cos, sin, logical: true);
                    continue;
                }

                int start = run.First, end = run.First + run.Count;
                for (int g = run.First; g < end; g++)
                {
                    var glyph = _glyphs[g];
                    bool several = SeveralCharacters(glyph);
                    if ((glyph.Gap || several) && g > start)
                    {
                        AddPiece(pieces, ref owners, runs, r, PieceText(runs, r, start, g), start, g - start, cos, sin, logical: false);
                        start = g;
                    }

                    if (several)
                    {
                        AddPiece(pieces, ref owners, runs, r, PieceText(runs, r, g, g + 1), g, 1, cos, sin, logical: true);
                        start = g + 1;
                    }
                }

                if (start < end)
                {
                    AddPiece(pieces, ref owners, runs, r, PieceText(runs, r, start, end), start, end - start, cos, sin, logical: false);
                }
            }

            bool turned = Math.Abs(sin) > Math.Abs(cos);
            var arranged = DocumentTextLayouts.Arrange(pieces, turned ? height : width, turned ? width : height, options);
            var lines = new List<DocumentLine>(arranged.Lines.Count);
            for (int l = 0; l < arranged.Lines.Count; l++)                      // by index: the layout's lists are interfaces (no enumerator made)
            {
                var line = arranged.Lines[l];
                var own = new List<int>(line.Runs.Count);
                for (int k = 0; k < line.Runs.Count; k++)
                {
                    int piece = line.Runs[k];
                    if ((uint)piece < (uint)owners.Count && !own.Contains(owners[piece]))
                    {
                        own.Add(owners[piece]);
                    }
                }

                var box = DocumentBox.Empty;
                foreach (int r in own)
                {
                    box = box.Union(runs[r].Box);
                }

                // A layout that gives each piece a line of its own (content order) gives each run one: its pieces' lines join.
                if (own.Count == 1 && lines.Count > 0 && lines[^1].Runs.Count == 1 && lines[^1].Runs[0] == own[0])
                {
                    var previous = lines[^1];
                    lines[^1] = previous with { Text = previous.Text + line.Text };
                    continue;
                }

                lines.Add(new DocumentLine(line.Text, box, line.RightToLeft, own));
            }

            // Lines joined back keep the layout's text, unless the layout gave its lines one a line (then so are the joined ones).
            bool joined = lines.Count != arranged.Lines.Count && arranged.Text == string.Join('\n', arranged.Lines.Select(l => l.Text));
            return new DocumentText(lines, joined ? string.Join('\n', lines.Select(l => l.Text)) : arranged.Text);
        }
        finally
        {
            owners.Return();
        }
    }

    private string Text(int first, int end)
    {
        var a = _glyphs[first];
        var b = _glyphs[end - 1];
        return new string(_chars, a.TextStart, b.TextStart + b.TextLength - a.TextStart);
    }

    // The text of glyphs [first, end) of run `run`: the run's own when they are all of it (the same characters), else made.
    private string PieceText(DocumentTextRun[] runs, int run, int first, int end) =>
        first == _runs[run].First && end == _runs[run].First + _runs[run].Count ? runs[run].Text : Text(first, end);

    private bool SeveralCharacters(in Glyph glyph) =>
        glyph.TextLength > 2 || (glyph.TextLength == 2 && !char.IsSurrogatePair(_chars[glyph.TextStart], _chars[glyph.TextStart + 1]));

    // A piece of run `run` for the layout: the run itself when every part of a piece is the run's already (the text the
    // same string, the box, baseline, direction, size, order and tag equal; the layouts read no other), else a piece.
    private void AddPiece(List<DocumentTextRun> pieces, ref PdfPooledList<int> owners, DocumentTextRun[] runs, int run, string text, int first, int count, double cos,
        double sin, bool logical)
    {
        var glyph = _glyphs[first];
        var box = Bounds(first, count, cos, sin);
        float baseline = (float)(-glyph.X * sin + glyph.Y * cos);
        var read = _runs[run];
        var tag = _tags[run];
        var whole = runs[run];
        bool same = ReferenceEquals(text, whole.Text) && whole.Box == box && whole.Baseline == baseline && whole.Logical == logical && whole.FontSize == read.Style.Size
                    && whole.Order == read.Order && whole.Tag == tag;
        pieces.Add(same ? whole : new DocumentTextRun(text, box, read.Style.Size) { Order = read.Order, Tag = tag, Logical = logical, Baseline = baseline });
        owners.Add(run);
    }

    // The direction most of the page's glyphs run in (to a tenth of a degree), as a cosine and sine: the frame lines are
    // found in. Each run's direction (in tenths of a degree) and index are sorted together in a buffer of the shared pool,
    // so a direction's runs sit together, its first run first; the direction of the most glyphs wins, the earlier of two
    // with as many.
    private (double Cos, double Sin) MainDirection()
    {
        int count = _runs.Count;
        long[] keys = ArrayPool<long>.Shared.Rent(count);
        try
        {
            for (int r = 0; r < count; r++)
            {
                keys[r] = (long)(int)Math.Round(Angle(r) * 1800 / Math.PI) << 32 | (uint)r;
            }

            var sorted = keys.AsSpan(0, count);
            sorted.Sort();
            int best = -1, bestGlyphs = 0;
            for (int i = 0; i < count;)
            {
                int key = (int)(sorted[i] >> 32), first = (int)(uint)sorted[i], glyphs = 0;
                for (; i < count && (int)(sorted[i] >> 32) == key; i++)
                {
                    glyphs += _runs[(int)(uint)sorted[i]].Count;
                }

                if (best < 0 || glyphs > bestGlyphs || (glyphs == bestGlyphs && first < best))
                {
                    (best, bestGlyphs) = (first, glyphs);
                }
            }

            double angle = Angle(best);
            return Math.Abs(angle) < 1e-9 ? (1, 0) : (Math.Cos(angle), Math.Sin(angle));
        }
        finally
        {
            ArrayPool<long>.Shared.Return(keys);
        }
    }

    // Run `run`'s direction: the angle of its first glyph's baseline (a vertical run's, down its column).
    private double Angle(int run)
    {
        var s = _segments[_glyphs[_runs[run].First].Segment];
        return s.Vertical ? Math.Atan2(-s.D, -s.C) : Math.Atan2(s.B, s.A);
    }

    /// <summary>A run read: its glyphs, its place in the content, how it is drawn, and the text a span gives it.</summary>
    internal readonly record struct ReadRun(int First, int Count, int Order, RunStyle Style, string? ActualText);

    /// <summary>How a run is drawn: its font, render mode, size on the page, first segment, and its paint (an index of the reader's, or -1).</summary>
    internal readonly record struct RunStyle(PdfFont Font, int Mode, float Size, int Segment, int Paint = -1);

    // One glyph shown: its text in the page's characters, its origin and advance on the page, its width and vertical
    // advance (text space units at size 1), its code, and whether a TJ offset came before it or it has no text.
    private struct Glyph
    {
        public int Segment;
        public int TextStart;
        public int TextLength;
        public double X;
        public double Y;
        public double Width;
        public double Height;
        public double AdvanceX;
        public double AdvanceY;
        public uint Code;
        public int CodeLength;
        public bool Gap;
        public bool Unmapped;
    }

    // What one showing operator's glyphs share: the map from text space at size 1 onto the page (no translation), the
    // font's ascent and descent, its direction and the font.
    private readonly record struct Segment(double A, double B, double C, double D, double Ascent, double Descent, bool Vertical, PdfFont Font);

    private readonly record struct Operand(PdfToken Kind, int Start, int Length, double Number);

    // The parts of the graphics state the text needs, and the vector content's (from D7).
    private struct State
    {
        public PdfColourSpace? FillSpace;
        public PdfColourSpace? StrokeSpace;
        public DocumentColour FillColour;
        public DocumentColour StrokeColour;
        public DocumentPattern? FillPattern;
        public DocumentPattern? StrokePattern;
        public double LineWidth;
        public int Cap;
        public int Join;
        public double MiterLimit;
        public double[]? Dash;
        public double DashPhase;
        public int Clip;
        public float FillAlpha;
        public float StrokeAlpha;
        public DocumentBlend Blend;
        public int SoftMask;
        public int Group;
        public PdfMatrix BaseCtm;
        public PdfMatrix Ctm;
        public PdfFont? Font;
        public string? FontLabel;
        public double Size;
        public double CharSpacing;
        public double WordSpacing;
        public double Scale;
        public double Leading;
        public double Rise;
        public int Mode;
    }

    // A marked-content span with /ActualText: where it started, its depth and text, and its first part's style.
    private sealed class MarkedSpan(int depth, string text, int firstGlyph)
    {
        public int Depth { get; } = depth;

        public string Text { get; } = text;

        public int FirstGlyph { get; } = firstGlyph;

        public RunStyle? Style { get; set; }
    }
}

/// <summary>
/// A list over an array of the shared pool, for a reader's working parts: the array is rented when the first item comes
/// (a larger one as it grows) and given back by <see cref="Return"/>, its references cleared so the pool holds none. Kept
/// in a field (or a local) of its own and passed by reference: a copy would share the array.
/// </summary>
internal struct PdfPooledList<T>
{
    private T[]? _items;
    private int _count;

    public readonly int Count => _count;

    /// <summary>Item <paramref name="index"/>, by reference.</summary>
    /// <exception cref="ArgumentOutOfRangeException">It is not below <see cref="Count"/>.</exception>
    public readonly ref T this[int index]
    {
        get
        {
            if ((uint)index >= (uint)_count)
            {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            return ref _items![index];
        }
    }

    public void Add(in T item)
    {
        var items = _items;
        if (items is null || _count == items.Length)
        {
            items = Grow();
        }

        items[_count++] = item;
    }

    /// <summary>Keeps the first <paramref name="count"/> items (0 or more).</summary>
    public void Truncate(int count)
    {
        if (count < _count)
        {
            if (RuntimeHelpers.IsReferenceOrContainsReferences<T>())
            {
                Array.Clear(_items!, count, _count - count);
            }

            _count = count;
        }
    }

    /// <summary>Empties the list and gives its array back to the pool.</summary>
    public void Return()
    {
        if (_items is { } items)
        {
            _items = null;
            Release(items, _count);
        }

        _count = 0;
    }

    private T[] Grow()
    {
        var larger = ArrayPool<T>.Shared.Rent(_items is null ? 16 : _items.Length * 2);
        if (_items is { } items)
        {
            items.AsSpan(0, _count).CopyTo(larger);
            Release(items, _count);
        }

        return _items = larger;
    }

    private static void Release(T[] items, int count)
    {
        if (RuntimeHelpers.IsReferenceOrContainsReferences<T>())
        {
            Array.Clear(items, 0, count);
        }

        ArrayPool<T>.Shared.Return(items);
    }
}
