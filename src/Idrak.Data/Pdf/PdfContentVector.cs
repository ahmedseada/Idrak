// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Data.Pdf;

/// <summary>
/// The content interpreter's vector part (ISO 32000-2, 8.4-8.7 and 11), read when <see cref="ReadVector"/> asks: path
/// construction (<c>m l c v y h re</c>) through the CTM into page points, painting (<c>S s f F f* B B* b b*</c>; <c>n</c>
/// ends a path unpainted), clipping (<c>W W*</c>, a form's /BBox, text in render modes 4 to 7), the line state
/// (<c>w J j M d</c>, and through <c>gs</c>), colours (<c>g G rg RG k K cs CS sc SC scn SCN</c> in every colour space, a
/// pattern's name), opacity, blend modes and soft masks (<c>gs</c>), shadings (<c>sh</c>), transparency groups (a form's
/// /Group), and tiling and shading patterns (each cell read once a page as a page of its own). Paths, runs, images and
/// shadings carry the clip, group and soft mask in force. A cubic that is a quadratic raised (its controls two thirds of
/// the way to one point, as writers write quadratics) is read back as that quadratic.
/// </summary>
internal sealed partial class PdfContentReader
{
    private const int OpMoveTo = 'm', OpLine = 'l', OpCurve = 'c', OpCurveV = 'v', OpCurveY = 'y', OpClosePath = 'h', OpEndPath = 'n', OpClip = 'W',
        OpWidth = 'w', OpCap = 'J', OpJoin = 'j', OpMiter = 'M', OpDash = 'd', OpGrey = 'g', OpGreyStroke = 'G', OpCmyk = 'k', OpCmykStroke = 'K';
    private const int OpRectangle = 'r' | 'e' << 8, OpClipEvenOdd = 'W' | '*' << 8, OpRgb = 'r' | 'g' << 8, OpRgbStroke = 'R' | 'G' << 8, OpSpace = 'c' | 's' << 8,
        OpSpaceStroke = 'C' | 'S' << 8, OpColour = 's' | 'c' << 8, OpColourStroke = 'S' | 'C' << 8, OpState = 'g' | 's' << 8;
    private const int OpColourN = 's' | 'c' << 8 | 'n' << 16, OpColourNStroke = 'S' | 'C' << 8 | 'N' << 16;

    private readonly PdfPathBuilder _path = new();
    private List<DocumentPath>? _paths;           // what the page is given: made with its first part, the page's own
    private List<DocumentPath>? _clips;
    private List<DocumentShading>? _shadings;
    private List<DocumentGroup>? _groups;
    private List<DocumentSoftMask>? _softMasks;
    private readonly List<RunPaint> _paints = [];
    private readonly Dictionary<string, PdfColourSpace> _spaces = new(StringComparer.Ordinal);
    private readonly Dictionary<string, DocumentPattern?> _patterns = new(StringComparer.Ordinal);
    private bool _vector;
    private int _clipRule;              // 0: none pending; 1: W; 2: W*
    private int _textClipFirst = -1, _textClipCount;
    private int _nesting;               // how deep this reader's content sits in patterns and soft masks
    private DocumentColour? _fixed;     // an uncoloured tiling's cell: the one colour every part is painted in

    /// <summary>Reads the page's vector content too (paths, clips, colours, shadings, patterns, groups, soft masks).</summary>
    public void ReadVector() => _vector = true;

    /// <summary>The painted paths, in content order.</summary>
    public IReadOnlyList<DocumentPath> Paths => (IReadOnlyList<DocumentPath>?)_paths ?? [];

    /// <summary>The clips the parts were drawn under.</summary>
    public IReadOnlyList<DocumentPath> Clips => (IReadOnlyList<DocumentPath>?)_clips ?? [];

    /// <summary>The shadings painted by <c>sh</c>, in content order.</summary>
    public IReadOnlyList<DocumentShading> Shadings => (IReadOnlyList<DocumentShading>?)_shadings ?? [];

    /// <summary>The transparency groups.</summary>
    public IReadOnlyList<DocumentGroup> Groups => (IReadOnlyList<DocumentGroup>?)_groups ?? [];

    /// <summary>The soft masks.</summary>
    public IReadOnlyList<DocumentSoftMask> SoftMasks => (IReadOnlyList<DocumentSoftMask>?)_softMasks ?? [];

    // The vector part at a page's end: what it made is the page's (the next page makes its own), the path, the paints and
    // the colour space and pattern caches start again.
    private void EndVector()
    {
        _path.Reset();
        _paths = _clips = null;
        _shadings = null;
        _groups = null;
        _softMasks = null;
        _paints.Clear();
        _spaces.Clear();
        _patterns.Clear();
        _vector = false;
        (_clipRule, _textClipFirst, _textClipCount, _nesting) = (0, -1, 0, 0);
        _fixed = null;
    }

    // The graphics state's vector part at the start of a content: black, a line of 1, no clip, opaque, normal.
    private void StartVector(PdfMatrix baseCtm)
    {
        _state.FillSpace = _state.StrokeSpace = null;
        _state.FillColour = _state.StrokeColour = DocumentColour.Black;
        (_state.FillPattern, _state.StrokePattern) = (null, null);
        (_state.LineWidth, _state.Cap, _state.Join, _state.MiterLimit, _state.Dash, _state.DashPhase) = (1, 0, 0, 10, null, 0);
        (_state.Clip, _state.FillAlpha, _state.StrokeAlpha, _state.Blend, _state.SoftMask, _state.Group) = (-1, 1f, 1f, DocumentBlend.Normal, -1, -1);
        _state.BaseCtm = baseCtm;
    }

    // ---- the operators -----------------------------------------------------------------------------------------------

    // A path, colour, line-state or graphics-state operator (only when vectors are read).
    private void Vector(int op, ReadOnlySpan<byte> data, PdfDictionary? resources, int depth)
    {
        Span<double> n = stackalloc double[6];
        switch (op)
        {
            case OpMoveTo when Numbers(2, n):
                _path.Move(Apply(n[0], n[1]));
                break;
            case OpLine when Numbers(2, n):
                _path.Line(Apply(n[0], n[1]));
                break;
            case OpCurve when Numbers(6, n):
                _path.Cubic(Apply(n[0], n[1]), Apply(n[2], n[3]), Apply(n[4], n[5]), quadratic: true);
                break;
            case OpCurveV when Numbers(4, n):
                _path.Cubic(_path.Current, Apply(n[0], n[1]), Apply(n[2], n[3]), quadratic: false);
                break;
            case OpCurveY when Numbers(4, n):
            {
                var end = Apply(n[2], n[3]);
                _path.Cubic(Apply(n[0], n[1]), end, end, quadratic: false);
                break;
            }

            case OpClosePath:
                _path.Close();
                break;
            case OpRectangle when Numbers(4, n):
                _path.Move(Apply(n[0], n[1]));
                _path.Line(Apply(n[0] + n[2], n[1]));
                _path.Line(Apply(n[0] + n[2], n[1] + n[3]));
                _path.Line(Apply(n[0], n[1] + n[3]));
                _path.Close();
                break;
            case OpEndPath:
                EndPath();
                break;
            case OpClip:
                _clipRule = 1;
                break;
            case OpClipEvenOdd:
                _clipRule = 2;
                break;
            case OpWidth when Numbers(1, n):
                _state.LineWidth = Math.Max(0, n[0]);
                break;
            case OpCap when Numbers(1, n):
                _state.Cap = (int)Math.Clamp(n[0], 0, 2);
                break;
            case OpJoin when Numbers(1, n):
                _state.Join = (int)Math.Clamp(n[0], 0, 2);
                break;
            case OpMiter when Numbers(1, n):
                _state.MiterLimit = Math.Max(1, n[0]);
                break;
            case OpDash:
                Dash();
                break;
            case OpGrey when Numbers(1, n):
                SetColour(stroke: false, PdfColourSpace.Grey, n[..1]);
                break;
            case OpGreyStroke when Numbers(1, n):
                SetColour(stroke: true, PdfColourSpace.Grey, n[..1]);
                break;
            case OpRgb when Numbers(3, n):
                SetColour(stroke: false, PdfColourSpace.Rgb, n[..3]);
                break;
            case OpRgbStroke when Numbers(3, n):
                SetColour(stroke: true, PdfColourSpace.Rgb, n[..3]);
                break;
            case OpCmyk when Numbers(4, n):
                SetColour(stroke: false, PdfColourSpace.Cmyk, n[..4]);
                break;
            case OpCmykStroke when Numbers(4, n):
                SetColour(stroke: true, PdfColourSpace.Cmyk, n[..4]);
                break;
            case OpSpace or OpSpaceStroke when _operandCount >= 1 && _operands[_operandCount - 1].Kind == PdfToken.Name:
                SetSpace(op == OpSpaceStroke, RawName(data, _operands[_operandCount - 1]), resources);
                break;
            case OpColour or OpColourN or OpColourStroke or OpColourNStroke:
                SetComponents(op is OpColourStroke or OpColourNStroke, data, resources, depth);
                break;
            case OpState when _operandCount >= 1 && _operands[_operandCount - 1].Kind == PdfToken.Name:
                SetState(RawName(data, _operands[_operandCount - 1]), resources, depth);
                break;
        }
    }

    private (double X, double Y) Apply(double x, double y) => PdfShadings.Apply(_state.Ctm, x, y);

    // A name operand's bytes (#xx decoded).
    private static byte[] RawName(ReadOnlySpan<byte> data, Operand operand)
    {
        var raw = data.Slice(operand.Start, operand.Length);
        if (raw.IndexOf((byte)'#') < 0)
        {
            return raw.ToArray();
        }

        var buffer = new byte[raw.Length];
        return buffer[..PdfContentLexer.DecodeName(raw, buffer)];
    }

    private static string Text(byte[] name) => PdfText.Utf8OrLatin1(name);

    // A name as a key of the page's caches (its bytes one to one).
    private static string Key(byte[] name) => System.Text.Encoding.Latin1.GetString(name);

    // d: [array] phase.
    private void Dash()
    {
        int end = _operandCount - 2;
        if (end < 0 || _operands[_operandCount - 1].Kind != PdfToken.Number || _operands[end].Kind != PdfToken.ArrayEnd)
        {
            return;
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

        var dash = new List<double>();
        for (int i = start + 1; i < end; i++)
        {
            if (_operands[i].Kind == PdfToken.Number && double.IsFinite(_operands[i].Number))
            {
                dash.Add(Math.Max(0, _operands[i].Number));
            }
        }

        _state.Dash = dash.Count == 0 || dash.All(d => d == 0) ? null : [.. dash];
        _state.DashPhase = double.IsFinite(_operands[_operandCount - 1].Number) ? _operands[_operandCount - 1].Number : 0;
    }

    // ---- painting ----------------------------------------------------------------------------------------------------

    // A painting operator: its place in the content (S s f F f* B B* b b* sh, as D3 counts them), and with vectors read,
    // the path painted (or the shading), then the clip it makes.
    private void Paint(int op, ReadOnlySpan<byte> data, PdfDictionary? resources)
    {
        int order = _order++;
        if (!_vector)
        {
            return;
        }

        if (op == OpShade)
        {
            if (_operandCount >= 1 && _operands[_operandCount - 1].Kind == PdfToken.Name)
            {
                Shade(RawName(data, _operands[_operandCount - 1]), resources, order);
            }

            return;
        }

        if (op is OpCloseStroke or OpCloseFillStroke or OpCloseFillStrokeEvenOdd)
        {
            _path.Close();
        }

        var fill = op switch
        {
            OpFill or OpFillOld or OpFillStroke or OpCloseFillStroke => DocumentFill.NonZero,
            OpFillEvenOdd or OpFillStrokeEvenOdd or OpCloseFillStrokeEvenOdd => DocumentFill.EvenOdd,
            _ => DocumentFill.None,
        };
        bool stroke = op is OpStroke or OpCloseStroke or OpFillStroke or OpFillStrokeEvenOdd or OpCloseFillStroke or OpCloseFillStrokeEvenOdd;
        if (!_path.IsEmpty)
        {
            var (pen, scale) = Pen(_state.Ctm);
            var (verbs, points) = _path.ToArrays();
            (_paths ??= []).Add(new DocumentPath(verbs, points)
            {
                Fill = fill,
                Stroke = stroke ? (float)(_state.LineWidth * scale) : null,
                StrokeMatrix = stroke ? pen : null,
                Cap = (DocumentLineCap)_state.Cap,
                Join = (DocumentLineJoin)_state.Join,
                MiterLimit = (float)_state.MiterLimit,
                Dash = stroke && _state.Dash is { } dash ? dash.Select(d => (float)(d * scale)).ToArray() : default,
                DashPhase = stroke ? (float)(_state.DashPhase * scale) : 0f,
                FillColour = fill != DocumentFill.None ? FillPaint() : default,
                StrokeColour = stroke ? StrokePaint() : default,
                FillPattern = fill != DocumentFill.None && _fixed is null ? _state.FillPattern : null,
                StrokePattern = stroke && _fixed is null ? _state.StrokePattern : null,
                Clip = _state.Clip, Order = order, Blend = _state.Blend, Group = _state.Group, SoftMask = _state.SoftMask,
            });
        }

        EndPath();
    }

    // A stroke's pen under the CTM (ISO 32000-2, 8.5.3.1: the line width is in user space): the root of the determinant's
    // size scales the width and the dashes into page points, and the CTM's linear part over that scale is the pen's shape
    // when it is not uniform (its columns not of one length at a right angle: stretched or skewed); null when it is (a
    // turn and one scale, mirrored or not: the pen stays round), within float rounding of a matrix written to a few places.
    private static (DocumentMatrix? Pen, double Scale) Pen(PdfMatrix m)
    {
        double scale = Math.Sqrt(Math.Abs(m.A * m.D - m.B * m.C));
        if (!(scale > 0) || !double.IsFinite(scale))
        {
            return (null, scale);                                      // the width as it always was: no pen to shape
        }

        double a = m.A / scale, b = m.B / scale, c = m.C / scale, d = m.D / scale;
        bool uniform = Math.Abs(a * a + b * b - 1) <= 1e-4 && Math.Abs(c * c + d * d - 1) <= 1e-4 && Math.Abs(a * c + b * d) <= 1e-4;
        return (uniform ? null : new DocumentMatrix((float)a, (float)b, (float)c, (float)d, 0f, 0f), scale);
    }

    private DocumentColour FillPaint() => (_fixed ?? _state.FillColour).WithAlpha(Math.Clamp(_state.FillAlpha, 0f, 1f));

    private DocumentColour StrokePaint() => (_fixed ?? _state.StrokeColour).WithAlpha(Math.Clamp(_state.StrokeAlpha, 0f, 1f));

    // The path's end: the clip it makes (W or W* before the painting operator), then a new path.
    private void EndPath()
    {
        if (_clipRule != 0 && !_path.IsEmpty)
        {
            var (verbs, points) = _path.ToArrays();
            (_clips ??= []).Add(new DocumentPath(verbs, points) { Fill = _clipRule == 2 ? DocumentFill.EvenOdd : DocumentFill.NonZero, Clip = _state.Clip });
            _state.Clip = _clips.Count - 1;
        }

        _clipRule = 0;
        _path.Clear();
    }

    // sh: the shading over the clip in force.
    private void Shade(byte[] name, PdfDictionary? resources, int order)
    {
        try
        {
            var value = Lookup(resources, "Shading"u8, name);
            if (value.IsNull)
            {
                _issues.Add(DocumentIssue.DamagedStream, $"page {_page}: the shading /{Text(name)} is not in the resources");
                return;
            }

            var shading = PdfShadings.Read(_file, value, resources, _state.Ctm);
            (_shadings ??= []).Add(new DocumentShading(shading.Kind, shading.Colour)
            {
                Coords = shading.Coords, DomainStart = shading.DomainStart, DomainEnd = shading.DomainEnd, ExtendStart = shading.ExtendStart, ExtendEnd = shading.ExtendEnd,
                Matrix = shading.Matrix, Triangles = shading.Triangles, Background = shading.Background,
                Order = order, Clip = _state.Clip, Alpha = _state.FillAlpha, Blend = _state.Blend, Group = _state.Group, SoftMask = _state.SoftMask,
            });
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            _issues.Add(DocumentIssue.DamagedStream, $"page {_page}: the shading /{Text(name)}: {e.Message}");
        }
    }

    // ---- colours -----------------------------------------------------------------------------------------------------

    private void SetColour(bool stroke, PdfColourSpace space, ReadOnlySpan<double> components)
    {
        var colour = PdfColours.ToColour(_file, space, components, 1f);
        if (stroke)
        {
            (_state.StrokeSpace, _state.StrokeColour, _state.StrokePattern) = (space, colour, null);
        }
        else
        {
            (_state.FillSpace, _state.FillColour, _state.FillPattern) = (space, colour, null);
        }
    }

    // cs / CS: the space by name (a device space, /Pattern, or the resources' /ColorSpace), its initial colour set.
    private void SetSpace(bool stroke, byte[] name, PdfDictionary? resources)
    {
        var space = Space(name, resources);
        if (space.Kind == PdfImageColour.Unknown && space.Name == "Pattern")
        {
            if (stroke)
            {
                (_state.StrokeSpace, _state.StrokePattern) = (space, null);
            }
            else
            {
                (_state.FillSpace, _state.FillPattern) = (space, null);
            }

            return;
        }

        SetColour(stroke, space, PdfColours.Initial(space));
    }

    // A colour space by name, read once a page: /Pattern (or [/Pattern base]) is kept as an unknown space named Pattern,
    // its base (the colour of an uncoloured tiling) its Base.
    private PdfColourSpace Space(byte[] name, PdfDictionary? resources)
    {
        string key = Key(name);
        if (_spaces.TryGetValue(key, out var known))
        {
            return known;
        }

        PdfColourSpace space;
        try
        {
            if (name.AsSpan().SequenceEqual("Pattern"u8))
            {
                space = new PdfColourSpace { Kind = PdfImageColour.Unknown, Name = "Pattern", Components = 0 };
            }
            else
            {
                var value = resources is not null && _file.Dictionary(resources["ColorSpace"u8]) is { } spaces && spaces.TryGet(name, out var named) ? named : PdfValue.Null;
                if (!value.IsNull && _file.Resolve(value).AsArray is { Count: >= 1 } array && _file.Resolve(array[0]).AsName is { } family && family.Is("Pattern"u8))
                {
                    var under = array.Count > 1 ? PdfColourSpace.Read(_file, array[1], resources) : null;
                    space = new PdfColourSpace { Kind = PdfImageColour.Unknown, Name = "Pattern", Components = under?.Components ?? 0, Base = under };
                }
                else
                {
                    space = PdfColourSpace.Read(_file, value.IsNull ? PdfValue.Of(PdfName.From(name)) : value, resources);
                }
            }
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            space = new PdfColourSpace { Kind = PdfImageColour.Unknown, Name = "unknown", Components = 1, Problem = e.Message };
        }

        if (space.Problem is { } problem)
        {
            _issues.Add(DocumentIssue.DamagedStream, $"page {_page}: the colour space /{Text(name)}: {problem}: painted black");
            space = PdfColourSpace.Grey;
        }

        _spaces[key] = space;
        return space;
    }

    // sc / scn / SC / SCN: the components in the space set (a pattern's name last in a Pattern space).
    private void SetComponents(bool stroke, ReadOnlySpan<byte> data, PdfDictionary? resources, int depth)
    {
        var space = (stroke ? _state.StrokeSpace : _state.FillSpace) ?? PdfColourSpace.Grey;
        Span<double> components = stackalloc double[32];
        int count = 0;
        byte[]? pattern = null;
        for (int i = 0; i < _operandCount; i++)
        {
            var operand = _operands[i];
            if (operand.Kind == PdfToken.Number && count < components.Length)
            {
                components[count++] = operand.Number;
            }
            else if (operand.Kind == PdfToken.Name && i == _operandCount - 1)
            {
                pattern = RawName(data, operand);
            }
        }

        if (space.Kind == PdfImageColour.Unknown && space.Name == "Pattern")
        {
            DocumentColour? under = space.Base is { } b && count > 0 ? PdfColours.ToColour(_file, b, components[..count], 1f) : null;
            var paint = pattern is null ? null : Pattern(pattern, under, resources, depth);
            if (stroke)
            {
                (_state.StrokePattern, _state.StrokeColour) = (paint, under ?? DocumentColour.Black);
            }
            else
            {
                (_state.FillPattern, _state.FillColour) = (paint, under ?? DocumentColour.Black);
            }

            return;
        }

        SetColour(stroke, space, components[..count]);
    }

    // ---- the graphics state and its soft masks -----------------------------------------------------------------------

    // gs: an ExtGState's line state, opacity, blend mode and soft mask.
    private void SetState(byte[] name, PdfDictionary? resources, int depth)
    {
        PdfDictionary? state;
        try
        {
            state = _file.Dictionary(Lookup(resources, "ExtGState"u8, name));
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            _issues.Add(DocumentIssue.DamagedStream, $"page {_page}: the graphics state /{Text(name)}: {e.Message}");
            return;
        }

        if (state is null)
        {
            return;
        }

        for (int i = 0; i < state.Count; i++)
        {
            var key = state.KeyAt(i);
            var value = _file.Resolve(state.ValueAt(i));
            if (key.Is("LW"u8) && value.TryNumber(out double lw))
            {
                _state.LineWidth = Math.Max(0, lw);
            }
            else if (key.Is("LC"u8) && value.TryInteger(out long lc))
            {
                _state.Cap = (int)Math.Clamp(lc, 0, 2);
            }
            else if (key.Is("LJ"u8) && value.TryInteger(out long lj))
            {
                _state.Join = (int)Math.Clamp(lj, 0, 2);
            }
            else if (key.Is("ML"u8) && value.TryNumber(out double ml))
            {
                _state.MiterLimit = Math.Max(1, ml);
            }
            else if (key.Is("D"u8) && value.AsArray is { Count: 2 } d && PdfFunction.Numbers(_file, d[0]) is { } pattern)
            {
                _state.Dash = pattern.Length == 0 || pattern.All(v => v <= 0) ? null : pattern;
                _state.DashPhase = _file.Resolve(d[1]).TryNumber(out double phase) ? phase : 0;
            }
            else if (key.Is("ca"u8) && value.TryNumber(out double ca))
            {
                _state.FillAlpha = (float)Math.Clamp(ca, 0, 1);
            }
            else if (key.Is("CA"u8) && value.TryNumber(out double strokeAlpha))
            {
                _state.StrokeAlpha = (float)Math.Clamp(strokeAlpha, 0, 1);
            }
            else if (key.Is("BM"u8))
            {
                _state.Blend = Blend(value);
            }
            else if (key.Is("SMask"u8))
            {
                _state.SoftMask = value.AsName is not null ? -1 : value.AsDictionary is { } mask ? SoftMask(mask, resources, depth) : _state.SoftMask;
            }
        }
    }

    // A blend mode by name (or the first known of an array of names); an unknown one is Normal.
    private DocumentBlend Blend(PdfValue value)
    {
        foreach (var item in value.AsArray?.Items ?? [value])
        {
            if (_file.Resolve(item).AsName is { } name && Enum.TryParse<DocumentBlend>(name.ToString(), ignoreCase: false, out var mode) && Enum.IsDefined(mode))
            {
                return mode;
            }

            if (_file.Resolve(item).AsName is { } compatible && compatible.Is("Compatible"u8))
            {
                return DocumentBlend.Normal;
            }
        }

        return DocumentBlend.Normal;
    }

    // An SMask dictionary: its group drawn as a page of its own where the CTM is now (luminosity on its backdrop, or alpha).
    private int SoftMask(PdfDictionary mask, PdfDictionary? resources, int depth)
    {
        try
        {
            if (_file.Resolve(mask["G"u8]).AsStream is not { } group)
            {
                return -1;
            }

            bool luminosity = _file.Resolve(mask["S"u8]).AsName is not { } kind || !kind.Is("Alpha"u8);
            var backdrop = DocumentColour.Black;
            if (PdfFunction.Numbers(_file, mask["BC"u8]) is { Length: > 0 } bc)
            {
                var space = _file.Dictionary(group.Dictionary["Group"u8]) is { } g && !g["CS"u8].IsNull ? PdfColourSpace.Read(_file, g["CS"u8], resources)
                    : bc.Length >= 4 ? PdfColourSpace.Cmyk : bc.Length == 3 ? PdfColourSpace.Rgb : PdfColourSpace.Grey;
                backdrop = space.Problem is null ? PdfColours.ToColour(_file, space, bc, 1f) : backdrop;
            }

            var ctm = PdfShadings.Matrix(_file, group.Dictionary["Matrix"u8]).Then(_state.Ctm);
            var content = SubPage(group, _file.Dictionary(group.Dictionary["Resources"u8]) ?? resources, ctm, ctm, depth, clipToBox: true, colour: null);
            (_softMasks ??= []).Add(new DocumentSoftMask(content, luminosity) { Backdrop = backdrop });
            return _softMasks.Count - 1;
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            _issues.Add(DocumentIssue.DamagedStream, $"page {_page}: a soft mask: {e.Message}");
            return -1;
        }
    }

    // ---- patterns ----------------------------------------------------------------------------------------------------

    // A pattern by name: a tiling's cell read once (an uncoloured one's for each colour it is painted in), or a shading.
    private DocumentPattern? Pattern(byte[] name, DocumentColour? under, PdfDictionary? resources, int depth)
    {
        string key = under is { } u ? $"{Key(name)}\u0001{u}" : Key(name);
        if (_patterns.TryGetValue(key, out var known))
        {
            return known;
        }

        DocumentPattern? pattern = null;
        try
        {
            var value = _file.Resolve(Lookup(resources, "Pattern"u8, name));
            var dictionary = value.AsStream?.Dictionary ?? value.AsDictionary;
            if (dictionary is null)
            {
                _issues.Add(DocumentIssue.DamagedStream, $"page {_page}: the pattern /{Text(name)} is not in the resources");
            }
            else
            {
                var matrix = PdfShadings.Matrix(_file, dictionary["Matrix"u8]).Then(_state.BaseCtm);
                long type = _file.Resolve(dictionary["PatternType"u8]).TryInteger(out long t) ? t : 0;
                if (type == 2)
                {
                    pattern = new DocumentPattern(PdfShadings.Read(_file, dictionary["Shading"u8], resources, matrix));
                }
                else if (type == 1 && value.AsStream is { } stream)
                {
                    double[] box = PdfFunction.Numbers(_file, dictionary["BBox"u8]) is { Length: 4 } b ? b : [0, 0, 1, 1];
                    double xStep = _file.Resolve(dictionary["XStep"u8]).TryNumber(out double x) && x != 0 ? x : box[2] - box[0];
                    double yStep = _file.Resolve(dictionary["YStep"u8]).TryNumber(out double y) && y != 0 ? y : box[3] - box[1];
                    bool uncoloured = _file.Resolve(dictionary["PaintType"u8]).TryInteger(out long paint) && paint == 2;
                    var cellResources = _file.Dictionary(dictionary["Resources"u8]) ?? resources;
                    var cell = SubPage(stream, cellResources, PdfMatrix.Identity, PdfMatrix.Identity, depth, clipToBox: false, colour: uncoloured ? under ?? DocumentColour.Black : null);
                    double left = Math.Min(box[0], box[2]), bottom = Math.Min(box[1], box[3]);
                    pattern = xStep != 0 && yStep != 0 && double.IsFinite(xStep) && double.IsFinite(yStep)
                        ? new DocumentPattern(cell, new DocumentBox((float)left, (float)bottom, (float)Math.Abs(box[2] - box[0]), (float)Math.Abs(box[3] - box[1])),
                            (float)xStep, (float)yStep, PdfShadings.Document(matrix))
                        : null;
                }
            }
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            _issues.Add(DocumentIssue.DamagedStream, $"page {_page}: the pattern /{Text(name)}: {e.Message}");
        }

        _patterns[key] = pattern;
        return pattern;
    }

    // A content of its own (a soft mask's group, a tiling's cell) read as a page of parts: `ctm` places it, `baseCtm` is
    // its patterns' space; a group's /BBox clips it when asked; `colour` paints every part of an uncoloured cell.
    private DocumentPage SubPage(PdfStream stream, PdfDictionary? resources, PdfMatrix ctm, PdfMatrix baseCtm, int depth, bool clipToBox, DocumentColour? colour)
    {
        if (_nesting + 1 > _file.MaxDepth || depth + 1 > _file.MaxDepth)
        {
            throw PdfLimit.Error(string.Create(CultureInfo.InvariantCulture,
                $"patterns and soft masks within each other {_nesting + 1} levels deep, deeper than the limit {nameof(DocumentOptions.MaxDepth)} (DocumentOptions.MaxDepth, {_file.MaxDepth})"));
        }

        using var reader = new PdfContentReader(_file, _page, _issues, _includeHidden) { _nesting = _nesting + 1, _fixed = colour };
        reader.ReadVector();
        reader.ReadImages();
        using var decoded = _file.Decode(stream);
        reader.ReadContent(decoded.Span, resources, ctm, baseCtm, clipToBox ? PdfFunction.Numbers(_file, stream.Dictionary["BBox"u8]) : null, depth + 1);
        var runs = reader.TextRuns(glyphs: true, out _, out _);
        return new DocumentPage(Math.Max(0, _page - 1), 0f, 0f)
        {
            Runs = runs, Images = reader.Images, Paths = reader.Paths, Clips = reader.Clips, Shadings = reader.Shadings, Groups = reader.Groups, SoftMasks = reader.SoftMasks,
        };
    }

    // A content run as a page's, from `ctm`, its parts clipped to `box` (a form's /BBox) when given.
    private void ReadContent(ReadOnlySpan<byte> content, PdfDictionary? resources, PdfMatrix ctm, PdfMatrix baseCtm, double[]? box, int depth)
    {
        _state = new State { Ctm = ctm, Scale = 1 };
        StartVector(baseCtm);
        if (box is { Length: 4 })
        {
            BoxClip(box);
        }

        Run(content, resources, depth);
        if (_span is not null)
        {
            EndSpan();
        }
    }

    // ---- forms -------------------------------------------------------------------------------------------------------

    // A form about to run (the CTM already through its /Matrix): its /BBox a clip under the one in force, its space the
    // patterns' own, and a transparency /Group a group of its own (the opacity, blend mode and soft mask in force are the
    // group's, and start again inside it).
    private void EnterForm(PdfDictionary form)
    {
        if (!_vector)
        {
            return;
        }

        _state.BaseCtm = _state.Ctm;
        if (PdfFunction.Numbers(_file, form["BBox"u8]) is { Length: 4 } box)
        {
            BoxClip(box);
        }

        if (_file.Dictionary(form["Group"u8]) is { } group && _file.Resolve(group["S"u8]).AsName is { } kind && kind.Is("Transparency"u8))
        {
            (_groups ??= []).Add(new DocumentGroup
            {
                Isolated = _file.Resolve(group["I"u8]).AsBoolean, Knockout = _file.Resolve(group["K"u8]).AsBoolean, Parent = _state.Group,
                Alpha = _state.FillAlpha, Blend = _state.Blend, SoftMask = _state.SoftMask,
            });
            _state.Group = _groups.Count - 1;
            (_state.FillAlpha, _state.StrokeAlpha, _state.Blend, _state.SoftMask) = (1f, 1f, DocumentBlend.Normal, -1);
        }
    }

    // A rectangle of user space (llx lly urx ury) through the CTM, a clip under the one in force.
    private void BoxClip(double[] box)
    {
        var (x0, y0) = Apply(box[0], box[1]);
        var (x1, y1) = Apply(box[2], box[1]);
        var (x2, y2) = Apply(box[2], box[3]);
        var (x3, y3) = Apply(box[0], box[3]);
        float[] points = [(float)x0, (float)y0, (float)x1, (float)y1, (float)x2, (float)y2, (float)x3, (float)y3];
        if (points.All(float.IsFinite))
        {
            (_clips ??= []).Add(new DocumentPath([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close], points) { Clip = _state.Clip });
            _state.Clip = _clips.Count - 1;
        }
    }

    // ---- text --------------------------------------------------------------------------------------------------------

    // A run's paint as it is shown: its colours (opacity included), its stroke's width on the page, its clip, group,
    // soft mask and blend mode.
    private int Snapshot()
    {
        double scale = Math.Sqrt(Math.Abs(_state.Ctm.A * _state.Ctm.D - _state.Ctm.B * _state.Ctm.C));
        _paints.Add(new RunPaint(FillPaint(), StrokePaint(), (float)(_state.LineWidth * scale), _state.Clip, _state.Blend, _state.Group, _state.SoftMask));
        return _paints.Count - 1;
    }

    // A run shown in a mode that clips (4 to 7): the text object's clip will take its glyphs.
    private void ClipRun(int order, int mode)
    {
        if (!_vector || mode < 4)
        {
            return;
        }

        if (_textClipFirst < 0)
        {
            _textClipFirst = order;
        }

        _textClipCount = order - _textClipFirst + 1;
    }

    // ET: the glyphs shown to clip in the text object become a clip under the one in force.
    private void EndTextClip()
    {
        if (_textClipFirst >= 0)
        {
            (_clips ??= []).Add(new DocumentPath([], []) { Fill = DocumentFill.NonZero, Clip = _state.Clip, ClipRun = _textClipFirst, ClipRunCount = _textClipCount });
            _state.Clip = _clips.Count - 1;
        }

        (_textClipFirst, _textClipCount) = (-1, 0);
    }

    /// <summary>How a run is painted: its colours, stroke width, clip, blend mode, group and soft mask.</summary>
    internal readonly record struct RunPaint(DocumentColour Fill, DocumentColour Stroke, float StrokeWidth, int Clip, DocumentBlend Blend, int Group, int SoftMask);

    // The style an image is placed with.
    private PdfPartStyle ImageStyle() => _vector
        ? new PdfPartStyle(_state.Clip, _state.FillAlpha, _fixed ?? _state.FillColour, _state.Blend, _state.Group, _state.SoftMask)
        : PdfPartStyle.None;
}

/// <summary>What an image is drawn with besides its place: its clip, opacity, a stencil's colour, blend mode, group and soft mask.</summary>
internal readonly record struct PdfPartStyle(int Clip, float Alpha, DocumentColour Colour, DocumentBlend Blend, int Group, int SoftMask)
{
    public static PdfPartStyle None { get; } = new(-1, 1f, DocumentColour.Black, DocumentBlend.Normal, -1, -1);
}

/// <summary>
/// A path being built, in page points: its verbs and points, the pen and the subpath's start. A cubic whose controls are
/// two thirds of the way from each end to one point is the quadratic of that point raised (how writers write quadratics),
/// and is kept as that quadratic when asked.
/// </summary>
internal sealed class PdfPathBuilder
{
    private readonly List<OutlineVerb> _verbs = [];
    private readonly List<float> _points = [];
    private (double X, double Y) _start;
    private bool _open;

    public (double X, double Y) Current { get; private set; }

    public bool IsEmpty => _verbs.Count == 0;

    public void Move((double X, double Y) p)
    {
        _verbs.Add(OutlineVerb.Move);
        Add(p);
        Current = _start = p;
        _open = true;
    }

    public void Line((double X, double Y) p)
    {
        if (!_open)
        {
            Move(Current);
        }

        _verbs.Add(OutlineVerb.Line);
        Add(p);
        Current = p;
    }

    public void Cubic((double X, double Y) c1, (double X, double Y) c2, (double X, double Y) end, bool quadratic)
    {
        if (!_open)
        {
            Move(Current);
        }

        var p0 = Current;
        if (quadratic)
        {
            double qx1 = p0.X + 1.5 * (c1.X - p0.X), qy1 = p0.Y + 1.5 * (c1.Y - p0.Y), qx2 = end.X + 1.5 * (c2.X - end.X), qy2 = end.Y + 1.5 * (c2.Y - end.Y);
            double size = Math.Max(1, Math.Max(Math.Max(Math.Abs(qx1), Math.Abs(qy1)), Math.Max(Math.Abs(end.X), Math.Abs(end.Y))));
            bool degenerate = Math.Abs(c1.X - p0.X) + Math.Abs(c1.Y - p0.Y) < 1e-9 && Math.Abs(c2.X - end.X) + Math.Abs(c2.Y - end.Y) < 1e-9;
            if (!degenerate && Math.Abs(qx1 - qx2) <= 1e-4 * size && Math.Abs(qy1 - qy2) <= 1e-4 * size)
            {
                _verbs.Add(OutlineVerb.Quadratic);
                Add(((qx1 + qx2) / 2, (qy1 + qy2) / 2));
                Add(end);
                Current = end;
                return;
            }
        }

        _verbs.Add(OutlineVerb.Cubic);
        Add(c1);
        Add(c2);
        Add(end);
        Current = end;
    }

    public void Close()
    {
        if (_open)
        {
            _verbs.Add(OutlineVerb.Close);
            Current = _start;
            _open = false;
        }
    }

    public void Clear()
    {
        _verbs.Clear();
        _points.Clear();
        _open = false;
    }

    /// <summary>Clears the path and puts the pen back at the origin, as a new builder has it (a reader's next page).</summary>
    public void Reset()
    {
        Clear();
        Current = _start = default;
    }

    public (OutlineVerb[] Verbs, float[] Points) ToArrays() => ([.. _verbs], [.. _points]);

    private void Add((double X, double Y) p)
    {
        _points.Add(double.IsFinite(p.X) ? (float)p.X : 0f);
        _points.Add(double.IsFinite(p.Y) ? (float)p.Y : 0f);
    }
}
