// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Data.Pdf;

/// <summary>
/// A Type 3 font as a face to draw with (ISO 32000-2, 9.6.4): each glyph (one a /CharProcs entry, glyph 0 none) is its
/// content stream's filled paths (<c>m l c v y h re</c> under <c>q Q cm</c>, painted by <c>f F f* B B* b b*</c>) in glyph
/// space through the font's /FontMatrix, in 1,000 units an em, so a run draws it as any other font's outline in the
/// run's colour (what <c>d1</c> glyphs are; a <c>d0</c> glyph's own colours and images are not kept, and strokes are left
/// out). Each glyph is read once, when first drawn; safe from several threads.
/// </summary>
internal sealed class PdfType3Face : IFontFace
{
    private const int UnitsPerEm = 1000;

    private readonly PdfFile _file;
    private readonly PdfStream?[] _procs;
    private readonly PdfMatrix _matrix;
    private readonly GlyphOutline?[] _outlines;

    private PdfType3Face(PdfFile file, string family, string[] names, PdfStream?[] procs, PdfMatrix matrix, FontMetrics metrics)
    {
        (_file, Family, _procs, _matrix, Metrics) = (file, family, procs, matrix, metrics);
        Names = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int g = 1; g < names.Length; g++)
        {
            Names.TryAdd(names[g], g);
        }

        _outlines = new GlyphOutline?[names.Length];
    }

    /// <summary>Each glyph's id by its name (its key in /CharProcs).</summary>
    public Dictionary<string, int> Names { get; }

    public string Family { get; }

    public string Style => "Regular";

    public string Format => "type3";

    public FontMetrics Metrics { get; }

    public int GlyphCount => _outlines.Length;

    /// <summary>The Type 3 font <paramref name="dictionary"/> as a face, or null when it has no /CharProcs.</summary>
    public static PdfType3Face? Read(PdfFile file, PdfDictionary dictionary, string? name)
    {
        if (file.Dictionary(dictionary["CharProcs"u8]) is not { } charProcs)
        {
            return null;
        }

        var names = new string[charProcs.Count + 1];
        var procs = new PdfStream?[charProcs.Count + 1];
        names[0] = ".notdef";
        for (int i = 0; i < charProcs.Count; i++)
        {
            names[i + 1] = charProcs.KeyAt(i).ToString();
            procs[i + 1] = file.Resolve(charProcs.ValueAt(i)).AsStream;
        }

        var matrix = PdfShadings.Matrix(file, dictionary["FontMatrix"u8]);
        if (matrix == PdfMatrix.Identity)
        {
            matrix = new PdfMatrix(0.001, 0, 0, 0.001, 0, 0);
        }

        // Glyph space through the font matrix, times the units an em.
        var toUnits = matrix.Then(new PdfMatrix(UnitsPerEm, 0, 0, UnitsPerEm, 0, 0));
        double[] box = PdfFunction.Numbers(file, dictionary["FontBBox"u8]) is { Length: 4 } b ? b : [0, -200, 1000, 800];
        var (_, top) = PdfShadings.Apply(toUnits, 0, box[3]);
        var (_, bottom) = PdfShadings.Apply(toUnits, 0, box[1]);
        return new PdfType3Face(file, name ?? "Type 3", names, procs, toUnits, new FontMetrics(UnitsPerEm, (int)Math.Round(top), (int)Math.Round(bottom), 0));
    }

    public int Glyph(int codePoint) => 0;

    public int Advance(int glyph) => 0;

    public GlyphOutline Outline(int glyph)
    {
        if ((uint)glyph >= (uint)_outlines.Length || _procs[glyph] is not { } proc)
        {
            return GlyphOutline.Empty;
        }

        var outline = Volatile.Read(ref _outlines[glyph]);
        if (outline is null)
        {
            try
            {
                using var decoded = _file.Decode(proc);
                outline = Paths(decoded.Span, _matrix);
            }
            catch (InvalidDataException e) when (!PdfLimit.Is(e))
            {
                outline = GlyphOutline.Empty;   // a damaged glyph draws nothing
            }

            Volatile.Write(ref _outlines[glyph], outline);
        }

        return outline;
    }

    public ReadOnlyMemory<byte> Table(string tag) => ReadOnlyMemory<byte>.Empty;

    // A glyph's filled paths through `matrix` (the glyph's CTM, set by cm, then the font's units).
    private static GlyphOutline Paths(ReadOnlySpan<byte> content, PdfMatrix units)
    {
        var lexer = new PdfContentLexer(content);
        var verbs = new List<OutlineVerb>();
        var points = new List<float>();
        var pathVerbs = new List<OutlineVerb>();
        var pathPoints = new List<float>();
        var stack = new Stack<PdfMatrix>();
        var ctm = PdfMatrix.Identity;
        Span<double> n = stackalloc double[6];
        int count = 0;
        double cx = 0, cy = 0;
        void Add(OutlineVerb verb, ReadOnlySpan<double> xy)
        {
            pathVerbs.Add(verb);
            for (int i = 0; i + 1 < xy.Length; i += 2)
            {
                var (x, y) = PdfShadings.Apply(ctm.Then(units), xy[i], xy[i + 1]);
                pathPoints.Add((float)x);
                pathPoints.Add((float)y);
            }
        }

        while (true)
        {
            var token = lexer.Next(out int start, out int length, out double number);
            if (token == PdfToken.End)
            {
                break;
            }

            if (token == PdfToken.Number)
            {
                if (count < n.Length)
                {
                    n[count] = number;
                }

                count++;
                continue;
            }

            if (token != PdfToken.Keyword)
            {
                count = 0;
                continue;
            }

            var op = content.Slice(start, length);
            switch (op.Length)
            {
                case 1 when op[0] == 'm' && count >= 2:
                    Add(OutlineVerb.Move, [n[0], n[1]]);
                    (cx, cy) = (n[0], n[1]);
                    break;
                case 1 when op[0] == 'l' && count >= 2:
                    Add(OutlineVerb.Line, [n[0], n[1]]);
                    (cx, cy) = (n[0], n[1]);
                    break;
                case 1 when op[0] == 'c' && count >= 6:
                    Add(OutlineVerb.Cubic, [n[0], n[1], n[2], n[3], n[4], n[5]]);
                    (cx, cy) = (n[4], n[5]);
                    break;
                case 1 when op[0] == 'v' && count >= 4:
                    Add(OutlineVerb.Cubic, [cx, cy, n[0], n[1], n[2], n[3]]);
                    (cx, cy) = (n[2], n[3]);
                    break;
                case 1 when op[0] == 'y' && count >= 4:
                    Add(OutlineVerb.Cubic, [n[0], n[1], n[2], n[3], n[2], n[3]]);
                    (cx, cy) = (n[2], n[3]);
                    break;
                case 1 when op[0] == 'h':
                    pathVerbs.Add(OutlineVerb.Close);
                    break;
                case 2 when op[0] == 'r' && op[1] == 'e' && count >= 4:
                    Add(OutlineVerb.Move, [n[0], n[1]]);
                    Add(OutlineVerb.Line, [n[0] + n[2], n[1]]);
                    Add(OutlineVerb.Line, [n[0] + n[2], n[1] + n[3]]);
                    Add(OutlineVerb.Line, [n[0], n[1] + n[3]]);
                    pathVerbs.Add(OutlineVerb.Close);
                    (cx, cy) = (n[0], n[1]);
                    break;
                case 2 when op[0] == 'c' && op[1] == 'm' && count >= 6:
                    ctm = new PdfMatrix(n[0], n[1], n[2], n[3], n[4], n[5]).Then(ctm);
                    break;
                case 1 when op[0] == 'q':
                    if (stack.Count < 64)
                    {
                        stack.Push(ctm);
                    }

                    break;
                case 1 when op[0] == 'Q':
                    if (stack.Count > 0)
                    {
                        ctm = stack.Pop();
                    }

                    break;
                case 1 when op[0] is (byte)'f' or (byte)'F' or (byte)'B' or (byte)'b':
                case 2 when op[1] == '*' && op[0] is (byte)'f' or (byte)'B' or (byte)'b':
                    verbs.AddRange(pathVerbs);
                    points.AddRange(pathPoints);
                    if (pathVerbs.Count > 0 && pathVerbs[^1] != OutlineVerb.Close)
                    {
                        verbs.Add(OutlineVerb.Close);
                    }

                    pathVerbs.Clear();
                    pathPoints.Clear();
                    break;
                case 1 when op[0] is (byte)'S' or (byte)'s' or (byte)'n':
                    pathVerbs.Clear();
                    pathPoints.Clear();
                    break;
            }

            count = 0;
        }

        return verbs.Count == 0 ? GlyphOutline.Empty : new GlyphOutline([.. verbs], [.. points]);
    }
}
