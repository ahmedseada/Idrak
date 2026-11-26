// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.PluginTests;

/// <summary>How a toy document format or writer misbehaves.</summary>
public enum ToyBehaviour
{
    /// <summary>As written.</summary>
    Plain,

    /// <summary>The format's pages, or the writer's documents, fail.</summary>
    Fails,

    /// <summary>The format's text, or the writer's bytes, differ from the plain toy's.</summary>
    Differs,
}

/// <summary>
/// A document format of one's own, written on Idrak's public API alone, the path an Office plug-in will take: a text file
/// of lines (a header, pages of images, paths, clips and glyph runs, the information, attached files, an end line naming
/// the page count). It reads what <see cref="ToyDocumentWriter"/> writes; damaged files fail with an
/// <see cref="InvalidDataException"/>, and the options' limits are kept (each image's bytes within
/// <see cref="DocumentOptions.MaxStreamBytes"/>, its lines within <see cref="DocumentOptions.MaxObjects"/>).
/// </summary>
/// <param name="name">Its name.</param>
/// <param name="behaviour">How it misbehaves.</param>
public sealed class ToyDocumentFormat(string name = ToyDocumentFormat.DefaultName, ToyBehaviour behaviour = ToyBehaviour.Plain) : IDocumentFormat
{
    /// <summary>The toy's name.</summary>
    public const string DefaultName = "outside-toy";

    /// <summary>The first line of every toy document.</summary>
    public const string Header = "IDRAK-TOY-DOCUMENT 1";

    /// <inheritdoc />
    public string Name { get; } = name;

    /// <inheritdoc />
    public IReadOnlyCollection<string> Extensions { get; } = [".toydoc"];

    /// <inheritdoc />
    public IReadOnlyCollection<string> MediaTypes { get; } = ["application/x-idrak-toy-document"];

    /// <inheritdoc />
    public bool Accepts(ReadOnlySpan<byte> header) => header.StartsWith("IDRAK-TOY-DOCUMENT 1\n"u8);

    /// <inheritdoc />
    public IDocument Open(Stream stream, string name, DocumentOptions options)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(options);
        using var copy = new MemoryStream();
        stream.Position = 0;
        stream.CopyTo(copy);
        return new ToyDocument(Name, copy.ToArray(), name, options, behaviour);
    }
}

/// <summary>A toy document open: its pages indexed at open, each parsed when read; safe to read from several threads.</summary>
internal sealed class ToyDocument : IDocument
{
    private readonly string[] _lines;
    private readonly List<(int Start, int End, float Width, float Height)> _pages = [];
    private readonly List<DocumentAttachment> _attachments = [];
    private readonly string _name;
    private readonly DocumentOptions _options;
    private readonly long _maxStreamBytes;
    private readonly ToyBehaviour _behaviour;

    public ToyDocument(string format, byte[] bytes, string name, DocumentOptions options, ToyBehaviour behaviour)
    {
        (Format, _name, _options, _behaviour) = (format, name, options, behaviour);
        _maxStreamBytes = options.MaxStreamBytes;
        try
        {
            _lines = Encoding.UTF8.GetString(bytes).Split('\n');
            if (_lines.Length == 0 || _lines[0] != ToyDocumentFormat.Header)
            {
                throw new InvalidDataException("not a toy document (no header line)");
            }

            if (_lines.Length > options.MaxObjects)
            {
                throw new InvalidDataException($"{_lines.Length} lines, more objects than DocumentOptions.{nameof(DocumentOptions.MaxObjects)} allows ({options.MaxObjects})");
            }

            int ended = -1;
            for (int i = 1; i < _lines.Length && ended < 0; i++)
            {
                string[] fields = _lines[i].Split(' ');
                switch (fields[0])
                {
                    case "page":
                        int end = Array.IndexOf(_lines, "end-page", i);
                        if (end < 0)
                        {
                            throw new InvalidDataException($"page {_pages.Count} has no end: the file is cut");
                        }

                        _pages.Add((i + 1, end, Number(fields[1]), Number(fields[2])));
                        i = end;
                        break;
                    case "info":
                        Info = new DocumentInfo { Title = Text(fields[1]), Author = Text(fields[2]), Subject = Text(fields[3]), Keywords = Text(fields[4]), Creator = Text(fields[5]) };
                        break;
                    case "attach":
                        byte[] content = Convert.FromBase64String(fields[3]);
                        _attachments.Add(new DocumentAttachment(Text(fields[1])!, () => new MemoryStream(content, writable: false)) { MediaType = Text(fields[2]), Size = content.Length });
                        break;
                    case "end":
                        if (int.Parse(fields[1], CultureInfo.InvariantCulture) != _pages.Count)
                        {
                            throw new InvalidDataException($"the end line counts {fields[1]} pages, the file holds {_pages.Count}");
                        }

                        ended = i;
                        break;
                    case "":
                        break;
                    default:
                        throw new InvalidDataException($"line {i + 1}: \"{fields[0]}\" is not a toy document's line");
                }
            }

            if (ended < 0)
            {
                throw new InvalidDataException("no end line: the file is cut");
            }
        }
        catch (Exception e) when (e is not InvalidDataException)
        {
            throw new InvalidDataException($"{name}: a damaged toy document ({e.GetType().Name}: {e.Message})", e);
        }
    }

    public string Format { get; }

    public DocumentInfo Info { get; } = DocumentInfo.Empty;

    public int PageCount => _pages.Count;

    public IReadOnlyList<DocumentAttachment> Attachments => _attachments;

    public IReadOnlyList<DocumentField> Fields => [];

    public IReadOnlyList<DocumentOutline> Outline => [];

    public IReadOnlyList<DocumentSignature> Signatures => [];

    public string? PageLabel(int index) => null;

    public DocumentDrawing DrawPage(int index, DocumentDrawOptions options) =>
        throw new NotSupportedException($"Page {index}: drawing through the page drawers by name (DocumentPageDrawers) is not on the new design yet.");

    public DocumentPage ReadPage(int index, DocumentParts parts)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, _pages.Count);
        if (_behaviour == ToyBehaviour.Fails)
        {
            throw new InvalidOperationException("The outside toy fails to read pages, on purpose.");
        }

        try
        {
            return Parse(index, parts);
        }
        catch (Exception e) when (e is not InvalidDataException)
        {
            throw new InvalidDataException($"{_name}: page {index} is damaged ({e.GetType().Name}: {e.Message})", e);
        }
    }

    public void Dispose()
    {
    }

    private DocumentPage Parse(int index, DocumentParts parts)
    {
        var (start, end, width, height) = _pages[index];
        var runs = new List<DocumentTextRun>();
        var images = new List<DocumentImage>();
        var paths = new List<DocumentPath>();
        var clips = new List<DocumentPath>();
        for (int i = start, order = 0; i < end; i++, order++)
        {
            string[] f = _lines[i].Split(' ');
            switch (f[0])
            {
                case "image":
                {
                    int pixelWidth = int.Parse(f[5], CultureInfo.InvariantCulture), pixelHeight = int.Parse(f[6], CultureInfo.InvariantCulture), channels = int.Parse(f[7], CultureInfo.InvariantCulture);
                    if ((long)f[8].Length * 3 / 4 > _maxStreamBytes)
                    {
                        throw new InvalidDataException($"an image of {f[8].Length * 3 / 4} bytes passes DocumentOptions.{nameof(DocumentOptions.MaxStreamBytes)} ({_maxStreamBytes})");
                    }

                    byte[] bytes = Convert.FromBase64String(f[8]);
                    if (channels is not (1 or 3) || pixelWidth <= 0 || pixelHeight <= 0 || bytes.Length != channels * pixelWidth * pixelHeight)
                    {
                        throw new InvalidDataException($"an image of {channels} x {pixelHeight} x {pixelWidth} holds {bytes.Length} bytes");
                    }

                    if ((parts & DocumentParts.Images) != 0)
                    {
                        images.Add(new DocumentImage(DocumentImage.Raw, pixelWidth, pixelHeight, () => bytes, () => Decode(bytes, channels, pixelHeight, pixelWidth))
                        {
                            Box = new DocumentBox(Number(f[1]), Number(f[2]), Number(f[3]), Number(f[4])), Order = order, Clip = int.Parse(f[9], CultureInfo.InvariantCulture),
                        });
                    }

                    break;
                }

                case "path":
                {
                    var path = ParsePath(f[11], f[12]);
                    if ((parts & DocumentParts.Paths) != 0)
                    {
                        paths.Add(new DocumentPath(path.Verbs, path.Points)
                        {
                            Fill = Fill(f[1]), Stroke = f[2] == "-" ? null : Number(f[2]), Cap = Enum.Parse<DocumentLineCap>(f[3]), Join = Enum.Parse<DocumentLineJoin>(f[4]),
                            MiterLimit = Number(f[5]), Dash = f[6] == "-" ? ReadOnlyMemory<float>.Empty : Numbers(f[6]), DashPhase = Number(f[7]),
                            FillColour = Colour(f[8]), StrokeColour = Colour(f[9]), Clip = int.Parse(f[10], CultureInfo.InvariantCulture), Order = order,
                        });
                    }

                    break;
                }

                case "clip":
                {
                    var path = ParsePath(f[3], f[4]);
                    if ((parts & DocumentParts.Paths) != 0)
                    {
                        clips.Add(new DocumentPath(path.Verbs, path.Points) { Fill = Fill(f[1]), Clip = int.Parse(f[2], CultureInfo.InvariantCulture) });
                    }

                    break;
                }

                case "text":
                {
                    var mode = Enum.Parse<DocumentTextMode>(f[7]);
                    string text = Text(f[11]) ?? "";
                    if (_behaviour == ToyBehaviour.Differs)
                    {
                        text = text.ToUpperInvariant();
                    }

                    DocumentGlyph[] glyphs = f[12] == "-" ? [] : f[12].Split(';').Select(g => g.Split(':')).Select(g => new DocumentGlyph(int.Parse(g[0], CultureInfo.InvariantCulture), Number(g[1]), Number(g[2]), Number(g[3]))).ToArray();
                    if ((parts & DocumentParts.Text) != 0 && (mode != DocumentTextMode.Hidden || _options.IncludeHidden))
                    {
                        runs.Add(new DocumentTextRun(text, new DocumentBox(Number(f[1]), Number(f[2]), Number(f[3]), Number(f[4])), Number(f[6]))
                        {
                            Baseline = Number(f[5]), Mode = mode, Colour = Colour(f[8]), Clip = int.Parse(f[9], CultureInfo.InvariantCulture), Font = Text(f[10]), Order = order,
                            Logical = true, Glyphs = (parts & DocumentParts.Glyphs) != 0 ? glyphs : ReadOnlyMemory<DocumentGlyph>.Empty,
                        });
                    }

                    break;
                }

                default:
                    throw new InvalidDataException($"line {i + 1}: \"{f[0]}\" is not a page's line");
            }
        }

        // The text layouts were picked by the options' name: not on the new design yet.
        var arranged = (parts & DocumentParts.Text) != 0
            ? throw new NotSupportedException($"The text layout '{_options.Layout}' by name is not on the new design yet.")
            : DocumentText.Empty;
        return new DocumentPage(index, width, height)
        {
            Runs = runs, Lines = arranged.Lines, Text = arranged.Text, Images = images, Paths = paths, Clips = clips,
        };
    }

    // Planar bytes, a level each: k stands for k / 255.
    private static ImageData Decode(byte[] bytes, int channels, int height, int width)
    {
        var planes = new byte[channels][];
        for (int c = 0; c < channels; c++)
        {
            planes[c] = bytes.AsSpan(c * height * width, height * width).ToArray();
        }

        var levels = new float[256];
        for (int k = 0; k < 256; k++)
        {
            levels[k] = k / 255f;
        }

        return new ImageData(planes, height, width, levels);
    }

    private static (OutlineVerb[] Verbs, float[] Points) ParsePath(string verbs, string points)
    {
        OutlineVerb[] steps = verbs == "-" ? [] : [.. verbs.Select(v => v switch
        {
            'M' => OutlineVerb.Move, 'L' => OutlineVerb.Line, 'Q' => OutlineVerb.Quadratic, 'C' => OutlineVerb.Cubic, 'Z' => OutlineVerb.Close,
            _ => throw new InvalidDataException($"'{v}' is not a path's step"),
        })];
        float[] values = points == "-" ? [] : Numbers(points);
        return (steps, values);
    }

    private static DocumentFill Fill(string text) => text switch { "N" => DocumentFill.NonZero, "E" => DocumentFill.EvenOdd, "-" => DocumentFill.None, _ => throw new InvalidDataException($"'{text}' is not a fill") };

    internal static DocumentColour Colour(string text)
    {
        string[] parts = text.Split(':');
        float[] c = Numbers(parts[1]);
        return parts[0] switch
        {
            "g" => DocumentColour.Grey(c[0]),
            "rgb" => DocumentColour.Rgb(c[0], c[1], c[2]),
            "cmyk" => DocumentColour.Cmyk(c[0], c[1], c[2], c[3]),
            _ => throw new InvalidDataException($"'{text}' is not a colour"),
        };
    }

    private static float Number(string text) => float.Parse(text, NumberStyles.Float, CultureInfo.InvariantCulture);

    private static float[] Numbers(string text) => [.. text.Split(',').Select(Number)];

    private static string? Text(string field) => field == "-" ? null : Encoding.UTF8.GetString(Convert.FromBase64String(field));
}

/// <summary>
/// The writer of <see cref="ToyDocumentFormat"/>'s files: each page written to the destination when the next starts (one
/// page in memory), the information, attached files and the end line at <see cref="IDocumentBuilder.Finish"/>; the same
/// input writes the same bytes. It writes no forms, bookmarks, labels, links, annotations, encryption or signatures, and
/// edits nothing (each a <see cref="NotSupportedException"/> saying so).
/// </summary>
/// <param name="name">Its name.</param>
/// <param name="behaviour">How it misbehaves.</param>
public sealed class ToyDocumentWriter(string name = ToyDocumentFormat.DefaultName, ToyBehaviour behaviour = ToyBehaviour.Plain) : IDocumentWriter
{
    /// <inheritdoc />
    public string Name { get; } = name;

    /// <inheritdoc />
    public IReadOnlyCollection<string> Extensions { get; } = [".toydoc"];

    /// <inheritdoc />
    public IReadOnlyCollection<string> MediaTypes { get; } = ["application/x-idrak-toy-document"];

    /// <inheritdoc />
    public IDocumentBuilder Create(Stream destination, DocumentWriteOptions options)
    {
        ArgumentNullException.ThrowIfNull(destination);
        if (behaviour == ToyBehaviour.Fails)
        {
            throw new InvalidOperationException("The outside toy fails to write documents, on purpose.");
        }

        return new ToyBuilder(destination, behaviour == ToyBehaviour.Differs);
    }

    /// <inheritdoc />
    public IDocumentEditor Edit(IDocument source, Stream destination, DocumentWriteOptions options) =>
        throw new NotSupportedException("The outside toy writer edits no documents.");

    private sealed class ToyBuilder : IDocumentBuilder
    {
        private readonly Stream _destination;
        private readonly List<(string Name, string MediaType, byte[] Bytes)> _attachments = [];
        private ToyPage? _page;
        private int _pages;
        private bool _finished;

        public ToyBuilder(Stream destination, bool differs)
        {
            _destination = destination;
            Write(ToyDocumentFormat.Header + "\n" + (differs ? "\n" : ""));
        }

        public DocumentInfo Info { get; set; } = DocumentInfo.Empty;

        public int PageCount => _pages;

        public IDocumentPageBuilder AddPage(float width, float height)
        {
            Flush();
            _page = new ToyPage(_pages++, width, height);
            return _page;
        }

        public IDocumentPageBuilder CopyPage(IDocument from, int index) => throw new NotSupportedException("The outside toy writer copies no pages.");

        public void Attach(string name, string mediaType, Stream content)
        {
            using var copy = new MemoryStream();
            content.CopyTo(copy);
            _attachments.Add((name, mediaType, copy.ToArray()));
        }

        public void AddField(DocumentField field) => throw new NotSupportedException("The outside toy writer writes no forms.");

        public void AddOutline(DocumentOutline outline) => throw new NotSupportedException("The outside toy writer writes no bookmarks.");

        public void SetPageLabels(IReadOnlyList<DocumentPageLabel> labels) => throw new NotSupportedException("The outside toy writer writes no page labels.");

        public void Encrypt(DocumentEncryption encryption) => throw new NotSupportedException("The outside toy writer encrypts nothing.");

        public void Sign(string field, SigningOptions options) => throw new NotSupportedException("The outside toy writer signs nothing.");

        public void Finish()
        {
            if (_finished)
            {
                return;
            }

            Flush();
            var end = new StringBuilder();
            if (Info != DocumentInfo.Empty)
            {
                end.Append("info ").Append(Field(Info.Title)).Append(' ').Append(Field(Info.Author)).Append(' ').Append(Field(Info.Subject)).Append(' ')
                    .Append(Field(Info.Keywords)).Append(' ').Append(Field(Info.Creator)).Append('\n');
            }

            foreach (var (name, mediaType, bytes) in _attachments)
            {
                end.Append("attach ").Append(Field(name)).Append(' ').Append(Field(mediaType)).Append(' ').Append(Convert.ToBase64String(bytes)).Append('\n');
            }

            end.Append("end ").Append(_pages.ToString(CultureInfo.InvariantCulture)).Append('\n');
            Write(end.ToString());
            _finished = true;
        }

        public void Dispose()
        {
        }

        // The page being built is written, and let go.
        private void Flush()
        {
            if (_page is { } page)
            {
                Write(page.Close());
                _page = null;
            }
        }

        private void Write(string text) => _destination.Write(Encoding.UTF8.GetBytes(text));
    }

    private sealed class ToyPage : IDocumentPageBuilder
    {
        private readonly Stack<int> _saved = new();
        private StringBuilder? _lines;
        private int _clip = -1, _clips;

        public ToyPage(int index, float width, float height)
        {
            (Index, Width, Height) = (index, width, height);
            _lines = new StringBuilder("page ").Append(N(width)).Append(' ').Append(N(height)).Append('\n');
        }

        public int Index { get; }

        public float Width { get; }

        public float Height { get; }

        public void Image(ImageData image, DocumentBox box, DocumentImageEncoding encoding)
        {
            ArgumentNullException.ThrowIfNull(image);
            if (image.Channels is not (1 or 3))
            {
                throw new ArgumentException($"The toy stores grey or RGB images, not {image.Channels} channels.", nameof(image));
            }

            float[] pixels = image.Pixels;
            byte[] bytes = new byte[pixels.Length];
            for (int i = 0; i < pixels.Length; i++)
            {
                bytes[i] = (byte)Math.Round(Math.Clamp(pixels[i], 0f, 1f) * 255f);
            }

            Lines.Append("image ").Append(N(box.X)).Append(' ').Append(N(box.Y)).Append(' ').Append(N(box.Width)).Append(' ').Append(N(box.Height)).Append(' ')
                .Append(image.Width.ToString(CultureInfo.InvariantCulture)).Append(' ').Append(image.Height.ToString(CultureInfo.InvariantCulture)).Append(' ')
                .Append(image.Channels.ToString(CultureInfo.InvariantCulture)).Append(' ').Append(Convert.ToBase64String(bytes)).Append(' ')
                .Append(_clip.ToString(CultureInfo.InvariantCulture)).Append('\n');
        }

        public void Path(DocumentPath path)
        {
            ArgumentNullException.ThrowIfNull(path);
            Lines.Append("path ").Append(FillName(path.Fill)).Append(' ').Append(path.Stroke is { } stroke ? N(stroke) : "-").Append(' ').Append(path.Cap).Append(' ').Append(path.Join).Append(' ')
                .Append(N(path.MiterLimit)).Append(' ').Append(path.Dash.IsEmpty ? "-" : string.Join(',', path.Dash.ToArray().Select(N))).Append(' ').Append(N(path.DashPhase)).Append(' ')
                .Append(ColourText(path.FillColour)).Append(' ').Append(ColourText(path.StrokeColour)).Append(' ').Append(_clip.ToString(CultureInfo.InvariantCulture)).Append(' ')
                .Append(Verbs(path)).Append(' ').Append(Points(path)).Append('\n');
        }

        public void Glyphs(IFontFace face, float size, ReadOnlySpan<DocumentGlyph> glyphs, string text, DocumentTextMode mode, DocumentColour colour)
        {
            ArgumentNullException.ThrowIfNull(face);
            ArgumentNullException.ThrowIfNull(text);
            float scale = size / face.Metrics.UnitsPerEm, ascender = face.Metrics.Ascender * scale, descender = face.Metrics.Descender * scale;
            float baseline = glyphs.Length > 0 ? glyphs[0].Y : 0f, left = glyphs.Length > 0 ? glyphs[0].X : 0f;
            float right = glyphs.Length > 0 ? glyphs[^1].X + glyphs[^1].Advance : 0f;
            var list = new StringBuilder();
            foreach (var g in glyphs)
            {
                list.Append(list.Length > 0 ? ";" : "").Append(g.Glyph.ToString(CultureInfo.InvariantCulture)).Append(':').Append(N(g.X)).Append(':').Append(N(g.Y)).Append(':').Append(N(g.Advance));
            }

            Lines.Append("text ").Append(N(left)).Append(' ').Append(N(baseline - ascender)).Append(' ').Append(N(right - left)).Append(' ').Append(N(ascender - descender)).Append(' ')
                .Append(N(baseline)).Append(' ').Append(N(size)).Append(' ').Append(mode).Append(' ').Append(ColourText(colour)).Append(' ').Append(_clip.ToString(CultureInfo.InvariantCulture)).Append(' ')
                .Append(Field(face.Family)).Append(' ').Append(Field(text)).Append(' ').Append(list.Length > 0 ? list.ToString() : "-").Append('\n');
        }

        public void Save() => _saved.Push(_clip);

        public void Restore() => _clip = _saved.Count > 0 ? _saved.Pop() : throw new InvalidOperationException("Restore without Save.");

        public void Clip(DocumentPath path)
        {
            ArgumentNullException.ThrowIfNull(path);
            Lines.Append("clip ").Append(FillName(path.Fill == DocumentFill.None ? DocumentFill.NonZero : path.Fill)).Append(' ').Append(_clip.ToString(CultureInfo.InvariantCulture)).Append(' ')
                .Append(Verbs(path)).Append(' ').Append(Points(path)).Append('\n');
            _clip = _clips++;
        }

        public void AddAnnotation(DocumentAnnotation annotation) => throw new NotSupportedException("The outside toy writer writes no annotations.");

        public void AddLink(DocumentLink link) => throw new NotSupportedException("The outside toy writer writes no links.");

        // The page's lines, ended; the page takes no more.
        internal string Close()
        {
            string text = Lines.Append("end-page\n").ToString();
            _lines = null;
            return text;
        }

        private StringBuilder Lines => _lines ?? throw new InvalidOperationException($"Page {Index} is written: add content before the next page starts.");

        private static string FillName(DocumentFill fill) => fill switch { DocumentFill.NonZero => "N", DocumentFill.EvenOdd => "E", _ => "-" };

        private static string Verbs(DocumentPath path) => path.Verbs.IsEmpty ? "-"
            : string.Concat(path.Verbs.ToArray().Select(v => v switch { OutlineVerb.Move => 'M', OutlineVerb.Line => 'L', OutlineVerb.Quadratic => 'Q', OutlineVerb.Cubic => 'C', _ => 'Z' }));

        private static string Points(DocumentPath path) => path.Points.IsEmpty ? "-" : string.Join(',', path.Points.ToArray().Select(N));
    }

    private static string ColourText(DocumentColour colour) => colour.Space switch
    {
        DocumentColourSpace.Grey => "g:" + N(colour[0]),
        DocumentColourSpace.Rgb => "rgb:" + N(colour[0]) + "," + N(colour[1]) + "," + N(colour[2]),
        _ => "cmyk:" + N(colour[0]) + "," + N(colour[1]) + "," + N(colour[2]) + "," + N(colour[3]),
    };

    private static string N(float value) => value.ToString("R", CultureInfo.InvariantCulture);

    private static string Field(string? text) => text is null ? "-" : Convert.ToBase64String(Encoding.UTF8.GetBytes(text));
}

