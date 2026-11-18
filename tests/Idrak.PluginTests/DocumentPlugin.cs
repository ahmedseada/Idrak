// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.PluginTests;

/// <summary>How a toy document format or writer misbehaves, to prove the registries' guards.</summary>
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
/// <param name="name">The name it registers under.</param>
/// <param name="behaviour">How it misbehaves (the guards' proof).</param>
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

    public DocumentDrawing DrawPage(int index, DocumentDrawOptions options) => DocumentPageDrawers.DrawPage(this, index, options);

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

        var arranged = (parts & DocumentParts.Text) != 0 ? DocumentTextLayouts.Arrange(runs, width, height, _options) : DocumentText.Empty;
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
/// <param name="name">The name it registers under.</param>
/// <param name="behaviour">How it misbehaves (the guards' proof).</param>
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

/// <summary>
/// The proofs of the document contract from outside (plan 28, D0): <see cref="ToyDocumentFormat"/> and
/// <see cref="ToyDocumentWriter"/> registered, used by name, extension and first bytes, checked by the conformance kit,
/// and put over a library format and writer of the same name to fall back and be compared.
/// </summary>
public static class DocumentPluginTests
{
    /// <summary>The name the proof's library default and app registration share.</summary>
    public const string ProofName = "outside-proof";

    private static readonly Lock Gate = new();
    private static bool s_defaults;

    /// <summary>A toy format and writer of one's own register, write and open files, pass the suites and unregister; an unknown name is refused naming the registered ones.</summary>
    public static void RegistersAndConforms(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                          // documents are read on the host: checked once
        }

        var format = new ToyDocumentFormat();
        var writer = new ToyDocumentWriter();
        string folder = Path.Combine(Path.GetTempPath(), "idrak-outside-documents", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(folder);
        DocumentFormats.Register(format);
        DocumentWriters.Register(writer);
        try
        {
            string origin = typeof(DocumentPluginTests).Assembly.GetName().Name!;
            Check(DocumentFormats.Names.Contains(ToyDocumentFormat.DefaultName) && DocumentFormats.Origin(ToyDocumentFormat.DefaultName) == origin
                  && DocumentFormats.Default(ToyDocumentFormat.DefaultName) is null && DocumentFormats.Extensions.Contains(".toydoc"),
                $"the format registered from outside: {string.Join(", ", DocumentFormats.Names)}, origin {DocumentFormats.Origin(ToyDocumentFormat.DefaultName)}");
            Check(DocumentWriters.Names.Contains(ToyDocumentFormat.DefaultName) && DocumentWriters.Origin(ToyDocumentFormat.DefaultName) == origin
                  && ReferenceEquals(DocumentWriters.For("a.TOYDOC"), writer), $"the writer registered from outside: {string.Join(", ", DocumentWriters.Names)}");

            // Written by extension to a file, then opened by its first bytes from the file, a stream and memory.
            string path = Path.Combine(folder, "letter.toydoc");
            using (var builder = DocumentWriters.Create(path))
            {
                builder.Info = new DocumentInfo { Title = "A letter", Author = "A sender" };
                var page = builder.AddPage(612f, 792f);
                page.Path(DocumentPath.Rectangle(new DocumentBox(72f, 72f, 200f, 100f)));
                page.Image(new ImageData([0f, 1f, 0.2f, 0.6f], 1, 2, 2), new DocumentBox(300f, 72f, 40f, 40f), DocumentImageEncoding.Flate);
                builder.Attach("notes.txt", "text/plain", new MemoryStream("Notes for the letter."u8.ToArray()));
                builder.Finish();
            }

            byte[] bytes = File.ReadAllBytes(path);
            Check(DocumentFormats.CanOpen(path) && ReferenceEquals(DocumentFormats.Detect(bytes), format) && DocumentFormats.Detect("%PDF-1.7\n"u8) is null,
                "found by extension and first bytes, and nothing else taken for it");
            using (var document = DocumentFormats.Open(path))
            {
                var page = document.ReadPage(0, DocumentParts.All);
                Check(document.Format == ToyDocumentFormat.DefaultName && document.PageCount == 1 && document.Info.Title == "A letter" && page.Paths.Count == 1
                      && page.Images.Count == 1 && page.Images[0].Decode().Pixels[2] == 0.2f && document.Attachments.Single().Size == 21,
                    $"opened from the file: {document.PageCount} pages, {document.Info.Title}");
            }

            using (var stream = File.OpenRead(path))
            using (var document = DocumentFormats.Open(stream, "letter"))
            {
                Check(document.PageCount == 1, "opened from a stream");
            }

            using (var document = DocumentFormats.Open(bytes, "letter"))
            {
                Check(document.PageCount == 1, "opened from memory");
            }

            try
            {
                DocumentFormats.Open("not a document of any format"u8.ToArray(), "plain");
                Check(false, "a file of no registered format opened");
            }
            catch (InvalidDataException e)
            {
                Check(e.Message.Contains(ToyDocumentFormat.DefaultName, StringComparison.Ordinal), $"refusal names the formats: {e.Message}");
            }

            // The conformance kit on the toy: files the writer made (one with text in a font, when the test data is here).
            var face = TestFace();
            var samples = new List<Abstraction.Testing.DocumentSample> { new(path, Pages: 1) };
            string text = Path.Combine(folder, "report.toydoc");
            using (var builder = DocumentWriters.Create(path: text))
            {
                var page = builder.AddPage(612f, 792f);
                if (face is not null)
                {
                    WriteLine(page, face, "A plain line of text.", 72f, 100f, DocumentTextMode.Fill);
                    WriteLine(page, face, "A line under the scan.", 72f, 130f, DocumentTextMode.Hidden);
                }

                builder.AddPage(300f, 200f).Path(new DocumentPath([OutlineVerb.Move, OutlineVerb.Cubic], [10f, 10f, 50f, 0f, 90f, 90f, 120f, 40f]) { Fill = DocumentFill.None, Stroke = 1f });
                builder.Finish();
            }

            samples.Add(new(text, Pages: 2, Texts: face is null ? null : ["A plain line of text.\nA line under the scan.", ""]));
            string cut = Path.Combine(folder, "cut.toydoc");
            File.WriteAllBytes(cut, bytes[..(bytes.Length / 2)]);
            samples.Add(new(cut, Error: "cut"));
            Abstraction.Testing.Conformance.CheckDocumentFormat(format, samples, options: new Abstraction.Testing.ContractCheckOptions { RandomCases = 16 }).ThrowIfFailed();
            Abstraction.Testing.Conformance.CheckDocumentWriter(writer, face: face).ThrowIfFailed();
        }
        finally
        {
            Check(DocumentFormats.Unregister(ToyDocumentFormat.DefaultName) && DocumentWriters.Unregister(ToyDocumentFormat.DefaultName)
                  && DocumentFormats.Find(ToyDocumentFormat.DefaultName) is null, "unregistered");
            Directory.Delete(folder, recursive: true);
        }

        try
        {
            DocumentFormats.Get(ToyDocumentFormat.DefaultName);
            Check(false, "an unregistered format was found");
        }
        catch (NotRegisteredException e)
        {
            Check(e.Registry == nameof(DocumentFormats) && e.Name == ToyDocumentFormat.DefaultName && e.Registered.SequenceEqual(DocumentFormats.Names)
                  && e.Message.Contains("DocumentFormats.Register", StringComparison.Ordinal), $"the refusal: {e.Message}");
        }

        try
        {
            DocumentWriters.Get(ToyDocumentFormat.DefaultName);
            Check(false, "an unregistered writer was found");
        }
        catch (NotRegisteredException e)
        {
            Check(e.Registry == nameof(DocumentWriters) && e.Registered.SequenceEqual(DocumentWriters.Names) && e.Message.Contains("DocumentWriters.Register", StringComparison.Ordinal),
                $"the refusal: {e.Message}");
        }
    }

    /// <summary>
    /// A toy format and writer over a library one of the same name (a stand-in for the library's, registered as its default
    /// for the process): the app's answers by default and its failure reaches the caller; under FallBack the library's
    /// answers the failing call; under Shadow the library's answers and the app's is compared; unregistered, the
    /// library's is back.
    /// </summary>
    public static void OverridesFallsBackAndShadows(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        lock (Gate)
        {
            if (!s_defaults)
            {
                // Library defaults stay for the process: the stand-in for a format the library ships.
                Overrides.AsLibraryDefaults(() =>
                {
                    DocumentFormats.Register(new ToyDocumentFormat(ProofName));
                    DocumentWriters.Register(new ToyDocumentWriter(ProofName));
                });
                s_defaults = true;
            }
        }

        Check(DocumentFormats.Origin(ProofName) == Overrides.Library && DocumentWriters.Origin(ProofName) == Overrides.Library, "the stand-ins are library defaults");
        byte[] library = Write(DocumentWriters.Get(ProofName));
        string expected;
        using (var document = DocumentFormats.Open(library, "proof"))
        {
            expected = document.ReadPage(0, DocumentParts.Text).Text;
        }

        try
        {
            // The format: fails on its pages.
            DocumentFormats.Register(new ToyDocumentFormat(ProofName, ToyBehaviour.Fails));
            Check(DocumentFormats.Origin(ProofName) == typeof(DocumentPluginTests).Assembly.GetName().Name && DocumentFormats.Default(ProofName) is not null, "the app's over the library's");
            using (var document = DocumentFormats.Open(library, "proof"))
            {
                try
                {
                    document.ReadPage(0, DocumentParts.Text);
                    Check(false, "the app's failure was not reported");
                }
                catch (InvalidOperationException)
                {
                    // Throw, the default policy: the app's failure reaches the caller.
                }

                DocumentFormats.SetPolicy(ProofName, SlotPolicy.FallBack);
                Check(document.ReadPage(0, DocumentParts.Text).Text == expected, "under FallBack the library's answers the failing call");
            }

            Check(Row(nameof(DocumentFormats)) is { Failures: >= 2, FallBacks: >= 1 }, $"the report: {Row(nameof(DocumentFormats))}");

            // The format: differs; compared under Shadow, the library's answering.
            DocumentFormats.Register(new ToyDocumentFormat(ProofName, ToyBehaviour.Differs));
            DocumentFormats.SetPolicy(ProofName, SlotPolicy.Shadow, shadowRate: 1);
            using (var document = DocumentFormats.Open(library, "proof"))
            {
                Check(document.ReadPage(0, DocumentParts.Text).Text == expected, "under Shadow the library's answers");
            }

            Check(Row(nameof(DocumentFormats)) is { Compared: >= 1, Differed: >= 1 }, $"the comparison: {Row(nameof(DocumentFormats))}");

            // The writer: fails to start a document, then writes other bytes.
            DocumentWriters.Register(new ToyDocumentWriter(ProofName, ToyBehaviour.Fails));
            DocumentWriters.SetPolicy(ProofName, SlotPolicy.FallBack);
            Check(Write(DocumentWriters.Get(ProofName)).AsSpan().SequenceEqual(library), "under FallBack the library's writes the document");
            DocumentWriters.Register(new ToyDocumentWriter(ProofName, ToyBehaviour.Differs));
            DocumentWriters.SetPolicy(ProofName, SlotPolicy.Shadow, shadowRate: 1);
            Check(Write(DocumentWriters.Get(ProofName)).AsSpan().SequenceEqual(library), "under Shadow the library's bytes are written");
            Check(Row(nameof(DocumentWriters)) is { Compared: >= 1, Differed: >= 1 }, $"the writers' comparison: {Row(nameof(DocumentWriters))}");
        }
        finally
        {
            DocumentFormats.Unregister(ProofName);
            DocumentWriters.Unregister(ProofName);
            DocumentFormats.SetPolicy(ProofName, SlotPolicy.Throw);
            DocumentWriters.SetPolicy(ProofName, SlotPolicy.Throw);
        }

        Check(DocumentFormats.Origin(ProofName) == Overrides.Library && DocumentWriters.Origin(ProofName) == Overrides.Library, "unregistered, the library's are back");
    }

    private static SlotOverride? Row(string registry) => Overrides.Report().FirstOrDefault(o => o.Registry == registry && o.Name == ProofName);

    // A one-page document of two rectangles and, where the test font is here, a line of text in it.
    private static byte[] Write(IDocumentWriter writer)
    {
        using var destination = new MemoryStream();
        using (var builder = writer.Create(destination, new DocumentWriteOptions()))
        {
            var page = builder.AddPage(400f, 300f);
            page.Path(DocumentPath.Rectangle(new DocumentBox(10f, 10f, 100f, 50f)));
            page.Path(DocumentPath.Rectangle(new DocumentBox(150f, 10f, 100f, 50f)));
            var face = TestFace();
            if (face is not null)
            {
                WriteLine(page, face, "A short report.", 20f, 120f, DocumentTextMode.Fill);
            }

            builder.Finish();
        }

        return destination.ToArray();
    }

    private static void WriteLine(IDocumentPageBuilder page, IFontFace face, string text, float x, float baseline, DocumentTextMode mode)
    {
        float size = 12f, scale = size / face.Metrics.UnitsPerEm;
        var glyphs = new List<DocumentGlyph>();
        foreach (var rune in text.EnumerateRunes())
        {
            int glyph = face.Glyph(rune.Value);
            glyphs.Add(new DocumentGlyph(glyph, x, baseline, face.Advance(glyph) * scale));
            x += face.Advance(glyph) * scale;
        }

        page.Glyphs(face, size, [.. glyphs], text, mode, DocumentColour.Black);
    }

    // The DejaVu Sans subset of tests/Idrak.Tests/data/fonts, found from the build output folder; null where it is not.
    private static IFontFace? TestFace()
    {
        for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory is not null; directory = directory.Parent)
        {
            string font = Path.Combine(directory.FullName, "tests", "Idrak.Tests", "data", "fonts", "dejavu-sans-subset.ttf");
            if (File.Exists(font))
            {
                return FontFormats.Read(font).Single();
            }
        }

        return null;
    }

    private static void Check(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }
}
