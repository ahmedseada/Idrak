// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Data.Pdf;

// Plan 28, phase D6: what a written PDF reads back as. Text (runs, glyph ids, places, boxes) comes from the library's
// reader (D3); images, vector paths, clips and attached files are read here through D1's objects, filters and content
// lexer, as phases D4 (images, attachments) and D7 (paths, clips) will read them: once they land, the reader's own parts
// are taken (a page that already has them keeps them), and this read-back can go.
internal abstract partial class Tests
{
    /// <summary>The PDF format with the parts D4 and D7 will read filled in from the file (see the file's comment).</summary>
    internal sealed class PdfWrittenReader : IDocumentFormat
    {
        private readonly PdfFormat _pdf = new();

        public string Name => _pdf.Name;

        public IReadOnlyCollection<string> Extensions => _pdf.Extensions;

        public IReadOnlyCollection<string> MediaTypes => _pdf.MediaTypes;

        public bool Accepts(ReadOnlySpan<byte> header) => _pdf.Accepts(header);

        public IDocument Open(Stream stream, string name, DocumentOptions options) => new PdfWrittenDocument(PdfDocument.Open(stream, name, options));
    }

    internal sealed class PdfWrittenDocument(PdfDocument inner) : IDocument
    {
        /// <summary>The file's object layer.</summary>
        public PdfFile Objects => inner.File;

        public string Format => inner.Format;

        public DocumentInfo Info => inner.Info;

        public int PageCount => inner.PageCount;

        public IReadOnlyList<DocumentField> Fields => inner.Fields;

        public IReadOnlyList<DocumentOutline> Outline => inner.Outline;

        public IReadOnlyList<DocumentSignature> Signatures => inner.Signatures;

        public string? PageLabel(int index) => inner.PageLabel(index);

        public DocumentDrawing DrawPage(int index, DocumentDrawOptions options) => inner.DrawPage(index, options);

        public IReadOnlyList<DocumentAttachment> Attachments => inner.Attachments.Count > 0 ? inner.Attachments : PdfReadAttachments(inner.File);

        public DocumentPage ReadPage(int index, DocumentParts parts)
        {
            var page = inner.ReadPage(index, parts);
            if ((parts & (DocumentParts.Images | DocumentParts.Paths)) == 0)
            {
                return page;
            }

            var (paths, clips, images) = PdfReadVector(inner.File, PdfPageDictionaries(inner.File)[index]);
            bool ownPaths = page.Paths.Count > 0 || page.Clips.Count > 0, ownImages = page.Images.Count > 0;
            return new DocumentPage(page.Index, page.Width, page.Height)
            {
                Rotation = page.Rotation, Runs = page.Runs, Lines = page.Lines, Text = page.Text, Issues = page.Issues, Annotations = page.Annotations,
                Images = ownImages || (parts & DocumentParts.Images) == 0 ? page.Images : images,
                Paths = ownPaths || (parts & DocumentParts.Paths) == 0 ? page.Paths : paths,
                Clips = ownPaths || (parts & DocumentParts.Paths) == 0 ? page.Clips : clips,
            };
        }

        public void Dispose() => inner.Dispose();
    }

    // The pages' dictionaries in order (the page tree walked).
    internal static List<PdfDictionary> PdfPageDictionaries(PdfFile file)
    {
        var pages = new List<PdfDictionary>();
        var catalog = file.Dictionary(file.Trailer["Root"u8])!;
        void Walk(PdfValue node, int depth)
        {
            var dictionary = file.Dictionary(node);
            if (dictionary is null || depth > 64)
            {
                return;
            }

            if (file.Resolve(dictionary["Kids"u8]).AsArray is { } kids && !dictionary["Type"u8].IsName("Page"u8))
            {
                foreach (var kid in kids.Items)
                {
                    Walk(kid, depth + 1);
                }

                return;
            }

            pages.Add(dictionary);
        }

        Walk(catalog["Pages"u8], 0);
        return pages;
    }

    // A page's content streams decoded and joined.
    internal static byte[] PdfPageContent(PdfFile file, PdfDictionary page)
    {
        var contents = file.Resolve(page["Contents"u8]);
        var parts = contents.AsArray?.Items ?? (contents.IsNull ? [] : [contents]);
        using var joined = new MemoryStream();
        foreach (var part in parts)
        {
            if (file.Resolve(part).AsStream is { } stream)
            {
                using var decoded = file.Decode(stream);
                joined.Write(decoded.Span);
                joined.WriteByte((byte)'\n');
            }
        }

        return joined.ToArray();
    }

    // The files a document carries: its EmbeddedFiles name tree, each file's name, media type, size and bytes.
    internal static List<DocumentAttachment> PdfReadAttachments(PdfFile file)
    {
        var found = new List<DocumentAttachment>();
        var catalog = file.Dictionary(file.Trailer["Root"u8]);
        var names = catalog is null ? null : file.Dictionary(catalog["Names"u8]);
        var tree = names is null ? null : file.Dictionary(names["EmbeddedFiles"u8]);
        void Walk(PdfDictionary? node, int depth)
        {
            if (node is null || depth > 32)
            {
                return;
            }

            if (file.Resolve(node["Names"u8]).AsArray is { } pairs)
            {
                for (int i = 0; i + 1 < pairs.Count; i += 2)
                {
                    var spec = file.Dictionary(pairs[i + 1]);
                    if (spec is null || file.Dictionary(spec["EF"u8]) is not { } ef || file.Resolve(ef["F"u8]).AsStream is not { } stream)
                    {
                        continue;
                    }

                    var label = file.Resolve(spec["UF"u8]).AsString ?? file.Resolve(spec["F"u8]).AsString ?? file.Resolve(pairs[i]).AsString;
                    string name = label is null ? "" : PdfText.Decode(label.Bytes);
                    byte[] bytes;
                    using (var decoded = file.Decode(stream))
                    {
                        bytes = decoded.Span.ToArray();
                    }

                    found.Add(new DocumentAttachment(name, () => new MemoryStream(bytes, writable: false))
                    {
                        MediaType = file.Resolve(stream.Dictionary["Subtype"u8]).AsName?.ToString(), Size = bytes.Length,
                    });
                }
            }

            if (file.Resolve(node["Kids"u8]).AsArray is { } kids)
            {
                foreach (var kid in kids.Items)
                {
                    Walk(file.Dictionary(kid), depth + 1);
                }
            }
        }

        Walk(tree, 0);
        return found;
    }

    // The graphics state the vector read-back keeps.
    private struct PdfReadState
    {
        public double A, B, C, D, E, F;
        public DocumentColour Fill, Stroke;
        public float FillAlpha, StrokeAlpha, Width, Miter, Phase;
        public DocumentLineCap Cap;
        public DocumentLineJoin Join;
        public float[] Dash;
        public int Clip;

        public readonly (double X, double Y) Map(double x, double y) => (A * x + C * y + E, B * x + D * y + F);
    }

    /// <summary>
    /// A page's painted paths, clips and images in page points from the top-left corner, read from its content: paths built
    /// by m l c v y h re (a cubic that is a quadratic raised read as the quadratic), painted by f F f* S B B* (b b* s close
    /// first), clips by W and W*, the line state (w J j M d), colours (g G rg RG k K) and opacity (gs: ca, CA), each in the
    /// current matrix (q Q cm); images drawn by Do. Paths, images and runs share one count of their places, as the text
    /// reader counts them.
    /// </summary>
    internal static (List<DocumentPath> Paths, List<DocumentPath> Clips, List<DocumentImage> Images) PdfReadVector(PdfFile file, PdfDictionary page)
    {
        var paths = new List<DocumentPath>();
        var clips = new List<DocumentPath>();
        var images = new List<DocumentImage>();
        var resources = file.Dictionary(page["Resources"u8]);
        double height = 0;
        if (file.Resolve(page["MediaBox"u8]).AsArray is { Count: 4 } media)
        {
            file.Resolve(media[1]).TryNumber(out double bottom);
            file.Resolve(media[3]).TryNumber(out double top);
            height = top - bottom;
        }

        // The display map (y turned down) is the starting matrix.
        var state = new PdfReadState
        {
            A = 1, D = -1, F = height, Fill = DocumentColour.Black, Stroke = DocumentColour.Black, FillAlpha = 1, StrokeAlpha = 1, Width = 1, Miter = 10, Dash = [], Clip = -1,
        };
        var stack = new Stack<PdfReadState>();
        var verbs = new List<OutlineVerb>();
        var points = new List<float>();
        var numbers = new List<double>();
        string? name = null;
        (double X, double Y) current = default, start = default;
        DocumentFill? pendingClip = null;
        int order = 0, marked = 0, spanAt = -1;
        bool spanText = false;
        byte[] content = PdfPageContent(file, page);
        var lexer = new PdfContentLexer(content);
        while (true)
        {
            var token = lexer.Next(out int at, out int length, out double number);
            if (token == PdfToken.End)
            {
                break;
            }

            if (token == PdfToken.Number)
            {
                numbers.Add(number);
                continue;
            }

            if (token == PdfToken.Name)
            {
                name = System.Text.Encoding.Latin1.GetString(content, at, length);
                continue;
            }

            if (token is PdfToken.ArrayStart or PdfToken.ArrayEnd or PdfToken.LiteralString or PdfToken.HexString or PdfToken.Dictionary)
            {
                if (token == PdfToken.Dictionary && content.AsSpan(at, length).IndexOf("ActualText"u8) >= 0)
                {
                    name = "\u0001ActualText";                                 // a span the text reader makes one run of
                }

                continue;
            }

            string op = System.Text.Encoding.ASCII.GetString(content, at, length);
            double N(int k) => numbers[numbers.Count - k];
            (double, double) Point(int k) => state.Map(N(k), N(k - 1));
            void Add(OutlineVerb verb, params ReadOnlySpan<(double X, double Y)> pairs)
            {
                verbs.Add(verb);
                foreach (var (x, y) in pairs)
                {
                    points.Add((float)x);
                    points.Add((float)y);
                }
            }

            switch (op)
            {
                case "q":
                    stack.Push(state with { Dash = state.Dash });
                    break;
                case "Q":
                    if (stack.Count > 0)
                    {
                        state = stack.Pop();
                    }

                    break;
                case "cm":
                    double a = N(6), b = N(5), c = N(4), d = N(3), e = N(2), f = N(1);
                    state = state with
                    {
                        A = a * state.A + b * state.C, B = a * state.B + b * state.D, C = c * state.A + d * state.C, D = c * state.B + d * state.D,
                        E = e * state.A + f * state.C + state.E, F = e * state.B + f * state.D + state.F,
                    };
                    break;
                case "w":
                    state.Width = (float)N(1);
                    break;
                case "J":
                    state.Cap = (DocumentLineCap)(int)N(1);
                    break;
                case "j":
                    state.Join = (DocumentLineJoin)(int)N(1);
                    break;
                case "M":
                    state.Miter = (float)N(1);
                    break;
                case "d":
                    state.Phase = (float)N(1);
                    state.Dash = [.. numbers.Take(numbers.Count - 1).Select(v => (float)v)];
                    break;
                case "g" or "G" or "rg" or "RG" or "k" or "K":
                    int count = op.ToLowerInvariant() switch { "g" => 1, "rg" => 3, _ => 4 };
                    var colour = count switch
                    {
                        1 => DocumentColour.Grey((float)N(1)),
                        3 => DocumentColour.Rgb((float)N(3), (float)N(2), (float)N(1)),
                        _ => DocumentColour.Cmyk((float)N(4), (float)N(3), (float)N(2), (float)N(1)),
                    };
                    if (char.IsUpper(op[0]))
                    {
                        state.Stroke = colour;
                    }
                    else
                    {
                        state.Fill = colour;
                    }

                    break;
                case "gs":
                    var gs = resources is null || name is null ? null : file.Dictionary(file.Dictionary(resources["ExtGState"u8])?[System.Text.Encoding.ASCII.GetBytes(name)] ?? PdfValue.Null);
                    if (gs is not null && file.Resolve(gs["ca"u8]).TryNumber(out double ca))
                    {
                        state.FillAlpha = (float)ca;
                    }

                    if (gs is not null && file.Resolve(gs["CA"u8]).TryNumber(out double strokeAlpha))
                    {
                        state.StrokeAlpha = (float)strokeAlpha;
                    }

                    break;
                case "m":
                    current = start = Point(2);
                    Add(OutlineVerb.Move, current);
                    break;
                case "l":
                    current = Point(2);
                    Add(OutlineVerb.Line, current);
                    break;
                case "c" or "v" or "y":
                    var p0 = current;
                    var (c1, c2, p3) = op switch
                    {
                        "c" => (Point(6), Point(4), Point(2)),
                        "v" => (current, Point(4), Point(2)),
                        _ => (Point(4), Point(2), Point(2)),
                    };
                    // A cubic whose controls are two thirds of the way to one point from each end is that quadratic.
                    var q1 = ((3 * c1.Item1 - p0.X) / 2, (3 * c1.Item2 - p0.Y) / 2);
                    var q2 = ((3 * c2.Item1 - p3.Item1) / 2, (3 * c2.Item2 - p3.Item2) / 2);
                    if (Math.Abs(q1.Item1 - q2.Item1) <= 1e-3 && Math.Abs(q1.Item2 - q2.Item2) <= 1e-3)
                    {
                        Add(OutlineVerb.Quadratic, ((q1.Item1 + q2.Item1) / 2, (q1.Item2 + q2.Item2) / 2), p3);
                    }
                    else
                    {
                        Add(OutlineVerb.Cubic, c1, c2, p3);
                    }

                    current = p3;
                    break;
                case "h":
                    Add(OutlineVerb.Close);
                    current = start;
                    break;
                case "re":
                    double rx = N(4), ry = N(3), rw = N(2), rh = N(1);
                    Add(OutlineVerb.Move, state.Map(rx, ry));
                    Add(OutlineVerb.Line, state.Map(rx + rw, ry));
                    Add(OutlineVerb.Line, state.Map(rx + rw, ry + rh));
                    Add(OutlineVerb.Line, state.Map(rx, ry + rh));
                    Add(OutlineVerb.Close);
                    current = start = state.Map(rx, ry);
                    break;
                case "W":
                    pendingClip = DocumentFill.NonZero;
                    break;
                case "W*":
                    pendingClip = DocumentFill.EvenOdd;
                    break;
                case "f" or "F" or "f*" or "S" or "s" or "B" or "B*" or "b" or "b*" or "n":
                    if (op is "s" or "b" or "b*")
                    {
                        Add(OutlineVerb.Close);
                    }

                    bool fills = op is "f" or "F" or "f*" or "B" or "B*" or "b" or "b*", strokes = op is "S" or "s" or "B" or "B*" or "b" or "b*";
                    if (op != "n")
                    {
                        double scale = Math.Sqrt(Math.Abs(state.A * state.D - state.B * state.C));
                        paths.Add(new DocumentPath([.. verbs], [.. points])
                        {
                            Fill = !fills ? DocumentFill.None : op.EndsWith('*') ? DocumentFill.EvenOdd : DocumentFill.NonZero,
                            Stroke = strokes ? (float)(state.Width * scale) : null, Cap = state.Cap, Join = state.Join, MiterLimit = state.Miter,
                            Dash = strokes ? state.Dash.Select(v => (float)(v * scale)).ToArray() : Array.Empty<float>(), DashPhase = strokes ? (float)(state.Phase * scale) : 0,
                            FillColour = state.Fill.WithAlpha(state.FillAlpha), StrokeColour = state.Stroke.WithAlpha(state.StrokeAlpha), Clip = state.Clip, Order = order++,
                        });
                    }

                    if (pendingClip is { } rule)
                    {
                        clips.Add(new DocumentPath([.. verbs], [.. points]) { Fill = rule, Clip = state.Clip });
                        state.Clip = clips.Count - 1;
                        pendingClip = null;
                    }

                    verbs.Clear();
                    points.Clear();
                    break;
                case "Do":
                    var xobject = resources is null || name is null ? null : file.Resolve(file.Dictionary(resources["XObject"u8])?[System.Text.Encoding.ASCII.GetBytes(name)] ?? PdfValue.Null).AsStream;
                    if (xobject is not null && file.Resolve(xobject.Dictionary["Subtype"u8]).IsName("Image"u8))
                    {
                        images.Add(PdfReadImage(file, xobject, state, order++));
                    }

                    break;
                case "BDC":
                    marked++;
                    if (spanAt < 0 && name == "\u0001ActualText")
                    {
                        (spanAt, spanText) = (marked, false);
                    }

                    break;
                case "BMC":
                    marked++;
                    break;
                case "EMC":
                    if (spanAt == marked)
                    {
                        order += spanText ? 1 : 0;
                        spanAt = -1;
                    }

                    marked = Math.Max(0, marked - 1);
                    break;
                case "Tj" or "TJ" or "'" or "\"":
                    if (spanAt >= 0)
                    {
                        spanText = true;
                    }
                    else
                    {
                        order++;
                    }

                    break;
                case "sh":
                    order++;
                    break;
            }

            numbers.Clear();
            name = null;
        }

        return (paths, clips, images);
    }

    // An image XObject as the page places it: its box through the current matrix, its pixels from its decoded stream
    // (Flate and its predictor through D1's filters; DCT through the jpeg codec).
    private static DocumentImage PdfReadImage(PdfFile file, PdfStream stream, PdfReadState state, int order)
    {
        var dictionary = stream.Dictionary;
        file.Resolve(dictionary["Width"u8]).TryInteger(out long width);
        file.Resolve(dictionary["Height"u8]).TryInteger(out long height);
        int channels = file.Resolve(dictionary["ColorSpace"u8]).IsName("DeviceRGB"u8) ? 3 : 1;
        byte[] stored;
        string? encoding;
        using (var decoded = file.Decode(stream))
        {
            stored = decoded.Span.ToArray();
            encoding = decoded.ImageEncoding;
        }

        var corners = new[] { state.Map(0, 0), state.Map(1, 0), state.Map(0, 1), state.Map(1, 1) };
        double left = corners.Min(c => c.X), top = corners.Min(c => c.Y), right = corners.Max(c => c.X), bottom = corners.Max(c => c.Y);
        int w = (int)width, h = (int)height;
        return new DocumentImage(encoding ?? DocumentImage.Raw, w, h, () => stored, () =>
        {
            if (encoding == DocumentImage.Jpeg)
            {
                return ImageCodecs.Get("jpeg").Decode(stored);
            }

            var planes = new byte[channels][];
            for (int c = 0; c < channels; c++)
            {
                planes[c] = new byte[w * h];
                for (int i = 0; i < w * h; i++)
                {
                    planes[c][i] = stored[i * channels + c];
                }
            }

            return new ImageData(planes, h, w, [.. Enumerable.Range(0, 256).Select(k => k / 255f)]);
        })
        {
            Box = new DocumentBox((float)left, (float)top, (float)(right - left), (float)(bottom - top)), Order = order, Clip = state.Clip,
        };
    }
}
