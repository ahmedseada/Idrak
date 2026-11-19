// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Data.Pdf;
using Idrak.Imaging;

// Plan 28, phase D6: PDF writing (the file, image pages, vector pages, text with fonts embedded, pages the library draws),
// every page kind read back through Idrak's own reader (and, for the parts D4 and D7 will read, through D1's objects and
// filters: DocumentPdfWriteReadBack.cs). Every file written goes to test-output/documents/written (NAME.pdf, and NAME.json
// with the text the reference readers must extract), for tools/documents/check_written.py.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] DocumentPdfWriteGroup =
    [
        ("documents pdf: write: the library's pdf writer and jpeg image encoder are registered (name, extensions, media type, the quality an option); DocumentWriterSuite passes on pdf with its defaults, with a classic table and stored streams (1.4, level 0), and without object streams", PdfWriteRegistered),
        ("documents pdf: write: the file: numbers invariant without exponents, names with #xx, strings escaped, the reader's objects written back as they read; object streams and a cross-reference stream, a classic table for 1.4, Flate at the option's level (0 stores), the trailer without a date or an ID unless given, the information in any script, a bad version refused", PdfWriteFile),
        ("documents pdf: write: image pages: grey, RGB and 8-bit planes stored exact (Flate, the PNG predictor where it pays, none on noise), JPEG equal to the codec's own round trip at the options' quality, Auto choosing every D6 sample's expectedAuto from measures equal to the sample's, the options' encoding over Auto, each image in its box", PdfWriteImages),
        ("documents pdf: write: vector pages: every verb (re, c, v and y, a quadratic), both fill rules, fill and stroke, caps, joins, miter limits, dashes with a phase, grey, RGB and CMYK with opacity, nested clips under Save and Restore read back exactly; a Restore without a Save, a path without a move and an empty dash refused", PdfWriteVector),
        ("documents pdf: write: text: each face embedded once as a Type0, Identity-H subset (ids kept, /W from the advances, a ToUnicode map), runs read back with their text, glyph ids, places and boxes; /ActualText where glyphs do not read as their text (a lam-alef ligature right to left, a character drawn by two glyphs); hidden and stroked runs; a CFF face, a mapped legacy face and whole embedding", PdfWriteText),
        ("documents pdf: write: a searchable scan (the page's image and its OCR text hidden over it) reads back both; the page-of-shapes helper writes a synthetic page (areas, dashed and dotted strokes, laid-out lines) whose paths and text read back", PdfWriteScanAndShapes),
        ("documents pdf: write: the same input writes the same bytes (on several threads too); a builder disposed without Finish leaves a file the reader repairs with its pages; attached files and the information read back; the parts later phases fill refused naming their phase; a page used after the next one started refused", PdfWriteDeterministicAndLimits),
        ("documents pdf: write: a 500-page image PDF written with one page of buffers (JPEG scans and Flate line art), timed cold and warm, the managed memory sampled per page and its growth bounded", PdfWriteManyPages),
        ("documents pdf: write: a sample document for a person to open (test-output/documents/written/sample.pdf): a vector page with text, a photograph, a searchable scan and a page of shapes", PdfWriteSample),
    ];

    private static readonly DocumentWriteOptions PdfWriteDefaults = new() { ImageEncoding = DocumentImageEncoding.Auto, EmbedFonts = DocumentFontEmbedding.Subset, JpegQuality = 90 };

    private static readonly OutlineVerb[] PdfBoxVerbs = [OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close];

    // A box's corners from its top-left, clockwise on the page (DocumentPath.Rectangle's points).
    private static float[] PdfBoxPoints(DocumentBox b) => [b.X, b.Y, b.Right, b.Y, b.Right, b.Bottom, b.X, b.Bottom];

    // The written files' folder (for check_written.py).
    private static string PdfWrittenFolder()
    {
        string folder = RepositoryPath("test-output", "documents", "written");
        Directory.CreateDirectory(folder);
        return folder;
    }

    // Saves a written file as NAME.pdf, and, when its pages' text is given, NAME.json for the checker.
    private static void PdfSaveWritten(string name, byte[] bytes, IReadOnlyList<(float Width, float Height, string Text)>? pages = null)
    {
        string folder = PdfWrittenFolder();
        File.WriteAllBytes(Path.Combine(folder, name + ".pdf"), bytes);
        string json = Path.Combine(folder, name + ".json");
        if (pages is null)
        {
            File.Delete(json);
            return;
        }

        var expected = new JsonObject
        {
            ["schema"] = 1, ["name"] = name, ["file"] = name + ".pdf",
            ["pages"] = new JsonArray([.. pages.Select((p, i) => (JsonNode)new JsonObject { ["index"] = i, ["width"] = p.Width, ["height"] = p.Height, ["text"] = p.Text })]),
        };
        File.WriteAllText(json, expected.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }) + "\n");
    }

    private static byte[] PdfWrite(Action<IDocumentBuilder> build, DocumentWriteOptions? options = null)
    {
        using var destination = new MemoryStream();
        using (var builder = DocumentWriters.Create(destination, "pdf", options ?? PdfWriteDefaults))
        {
            build(builder);
            builder.Finish();
        }

        return destination.ToArray();
    }

    private static PdfWrittenDocument PdfReadWritten(byte[] bytes, string name = "written") =>
        (PdfWrittenDocument)new PdfWrittenReader().Open(new MemoryStream(bytes, writable: false), name, new DocumentOptions());

    // A line shaped by its script's shaper, from (x, baseline) at size points: the glyphs left to right.
    private static DocumentGlyph[] PdfShaped(IFontFace face, string text, string script, bool rightToLeft, float x, float baseline, float size)
    {
        var shaped = TextShapers.Shape(face, text, new TextRun(0, text.Length, script, rightToLeft));
        float scale = size / face.Metrics.UnitsPerEm, pen = x;
        var glyphs = new DocumentGlyph[shaped.Length];
        for (int i = 0; i < shaped.Length; i++)
        {
            var g = shaped[i];
            glyphs[i] = new DocumentGlyph(g.Glyph, pen + g.XOffset * scale, baseline - g.YOffset * scale, g.XAdvance * scale);
            pen += g.XAdvance * scale;
        }

        return glyphs;
    }

    private static string PdfContentText(PdfWrittenDocument document, int page) =>
        Encoding.Latin1.GetString(PdfPageContent(document.Objects, PdfPageDictionaries(document.Objects)[page]));

    // Pixels the same to float rounding (two codecs' tables of a byte's value may differ in the last bit).
    private static bool PdfSamePixels(float[] a, float[] b) => a.Length == b.Length && a.Zip(b).All(v => MathF.Abs(v.First - v.Second) <= 1e-6f);

    private static bool PdfNear(DocumentBox a, DocumentBox b, float tolerance = 1e-3f) =>
        MathF.Abs(a.X - b.X) <= tolerance && MathF.Abs(a.Y - b.Y) <= tolerance && MathF.Abs(a.Width - b.Width) <= tolerance && MathF.Abs(a.Height - b.Height) <= tolerance;

    private static ImageData PdfNoise(int seed, int channels, int width, int height)
    {
        var random = new Random(seed);
        var pixels = new float[channels * width * height];
        for (int i = 0; i < pixels.Length; i++)
        {
            pixels[i] = random.Next(256) / 255f;
        }

        return new ImageData(pixels, channels, height, width);
    }

    // Flat colours and sharp edges (a diagram): few levels.
    private static ImageData PdfLineArt(int channels, int width, int height)
    {
        var pixels = new float[channels * width * height];
        int plane = width * height;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int band = (x / 37 + y / 23) % 4;
                for (int c = 0; c < channels; c++)
                {
                    pixels[c * plane + y * width + x] = band switch { 0 => 1f, 1 => 0f, 2 => (c + 1) * 60 / 255f, _ => 200 / 255f };
                }
            }
        }

        return new ImageData(pixels, channels, height, width);
    }

    private static void PdfWriteRegistered(Device device)
    {
        CpuOnly(device);
        var writer = DocumentWriters.Get("pdf");
        Check(writer.Name == "pdf" && writer.Extensions.Contains(".pdf") && writer.MediaTypes.Contains("application/pdf") && DocumentWriters.Origin("pdf") == Overrides.Library
              && DocumentWriters.For("Report.PDF").Name == "pdf" && DocumentWriters.Names.Contains("pdf"), $"the pdf writer: {string.Join(", ", DocumentWriters.Names)}");

        // The jpeg encoder: the JPEG encoder's bytes at the quality asked (Pillow's 75 when none is).
        var jpeg = ImageEncoders.Get("jpeg");
        var photo = PdfNoise(3, 3, 40, 24);
        Check(ImageEncoders.Origin("jpeg") == Overrides.Library && ImageEncoders.For("photo.JPG").Name == "jpeg" && ImageEncoders.For("photo.jpeg").Name == "jpeg"
              && jpeg.Encode(photo).AsSpan().SequenceEqual(JpegEncoder.Encode(photo, 75))
              && jpeg.Encode(photo, new ImageEncodeOptions { Quality = 40 }).AsSpan().SequenceEqual(JpegEncoder.Encode(photo, 40))
              && ImageEncoders.Get("png").Encode(photo, new ImageEncodeOptions { Quality = 40 }).AsSpan().SequenceEqual(ImageEncoders.Get("png").Encode(photo)),
            "the jpeg encoder writes JpegEncoder's bytes at the quality asked; png ignores it");
        ExpectRefusal<ArgumentOutOfRangeException>(() => _ = new ImageEncodeOptions { Quality = 0 }, "1 to 100", "a quality of 0");
        ExpectRefusal<ArgumentOutOfRangeException>(() => _ = new DocumentWriteOptions { SmoothSteps = 1.5f }, "0 to 1", "a share past 1");

        var face = SubsetFace();
        Conformance.CheckDocumentWriter(writer, new PdfWrittenReader(), face, PdfWriteDefaults, new ContractCheckOptions { RandomCases = 4 }).ThrowIfFailed();
        Conformance.CheckDocumentWriter(writer, new PdfWrittenReader(), face, PdfWriteDefaults with { Version = "1.4", Compression = 0 }, new ContractCheckOptions { RandomCases = 2 }).ThrowIfFailed();
        Conformance.CheckDocumentWriter(writer, new PdfWrittenReader(), face, PdfWriteDefaults with { ObjectStreams = false, Compression = 9 }, new ContractCheckOptions { RandomCases = 2 }).ThrowIfFailed();
    }

    private static void PdfWriteFile(Device device)
    {
        CpuOnly(device);

        // The syntax on its own: numbers, names, strings, text strings, and the reader's objects written back.
        using (var s = new PdfSyntax())
        {
            s.Number(123.45f).Space().Number(-0f).Space().Number(1e-7f).Space().Number(1e20f).Space().Number(600.0977, 3).Space().Number(-0.0001, 3).Space().Number(2.5, 6)
                .Space().Name("A name/with (delimiters)#é"u8.ToArray()).Space().Literal("a (b) \\ c\n\r\u0001"u8).Space().Text("plain").Space().Text("Grüße");
            string written = Encoding.Latin1.GetString(s.Span);
            Check(written == "123.45 0 0.0000001 100000002004088000000 600.098 0 2.5 /A#20name#2Fwith#20#28delimiters#29#23#C3#A9 (a \\(b\\) \\\\ c\\n\\r\\001) (plain) <FEFF0047007200FC00DF0065>",
                $"the syntax: {written}");
            Check(PdfSyntax.Written(600.0977, 3) == 600.098 && PdfSyntax.Written(123.45f) == 123.45 && (float)PdfSyntax.Written(1e20f) == 1e20f && (float)PdfSyntax.Written(1e-7f) == 1e-7f,
                "the values a reader reads (a float without digits for an exponent still reads back as itself)");

            byte[] source = "<< /Type /Example /Name#20With#28Odd#29 (str\\(ing\\) \\n) /Array [1 2.5 -3 true false null [/Nested <0A0B>] 7 0 R] /Dict << /K (v) >> >>"u8.ToArray();
            var value = new PdfParser(source, 16, complete: true).ReadValue();
            s.Clear();
            s.Value(value);
            var again = new PdfParser(s.Span.ToArray(), 16, complete: true).ReadValue();
            Check(value.ToString() == again.ToString(), $"the reader's objects written back: {value} became {again}");
        }

        var face = SubsetFace();
        void Build(IDocumentBuilder builder)
        {
            var page = builder.AddPage(300, 200);
            page.Path(new DocumentPath(PdfBoxVerbs, PdfBoxPoints(new DocumentBox(20, 20, 100, 50))));
            page.Glyphs(face, 12, PdfShaped(face, "Quarterly report", "Latn", false, 20, 120, 12), "Quarterly report", DocumentTextMode.Fill, DocumentColour.Black);
            page.Image(PdfLineArt(1, 64, 32), new DocumentBox(150, 20, 128, 64), DocumentImageEncoding.Flate);
        }

        byte[] modern = PdfWrite(Build);
        using (var document = PdfReadWritten(modern))
        {
            var file = document.Objects;
            string head = Encoding.Latin1.GetString(modern, 0, 15);
            Check(head.StartsWith("%PDF-1.7\n%", StringComparison.Ordinal) && modern[10] >= 0x80, $"the header: {head}");
            Check(file.Xref.Sections.Any(x => x.Kind is "stream" or "objstm") && file.LiveObjects().Any(o => o.Kind == PdfEntryKind.Compressed)
                  && !file.Trailer.Contains("ID"u8) && !file.Trailer.Contains("Info"u8) && !file.Repaired,
                $"a cross-reference stream, object streams, no ID or Info: {string.Join(", ", file.Xref.Sections.Select(x => x.Kind))}");
            Check(document.Info.Created is null && document.Info.Title is null && document.Info.Version == "1.7", "no information unless given");
            Check(document.ReadPage(0, DocumentParts.All).Text == "Quarterly report", "the page reads");
        }

        byte[] classic = PdfWrite(Build, PdfWriteDefaults with { Version = "1.4" });
        string classicText = Encoding.Latin1.GetString(classic);
        using (var document = PdfReadWritten(classic))
        {
            Check(classicText.StartsWith("%PDF-1.4", StringComparison.Ordinal) && classicText.Contains("\nxref\n0 ", StringComparison.Ordinal) && classicText.Contains("trailer\n<<", StringComparison.Ordinal)
                  && !classicText.Contains("/ObjStm", StringComparison.Ordinal) && !classicText.Contains("/XRef", StringComparison.Ordinal)
                  && document.Objects.Xref.Sections.All(x => x.Kind == "table") && !document.Objects.Repaired && document.ReadPage(0, DocumentParts.Text).Text == "Quarterly report",
                "a classic table for 1.4");
        }

        byte[] plainStreams = PdfWrite(Build, PdfWriteDefaults with { ObjectStreams = false });
        Check(!Encoding.Latin1.GetString(plainStreams).Contains("/ObjStm", StringComparison.Ordinal) && Encoding.Latin1.GetString(plainStreams).Contains("/XRef", StringComparison.Ordinal),
            "a cross-reference stream without object streams");
        byte[] stored = PdfWrite(Build, PdfWriteDefaults with { Compression = 0 }), smallest = PdfWrite(Build, PdfWriteDefaults with { Compression = 9 }),
            fastest = PdfWrite(Build, PdfWriteDefaults with { Compression = 1 });
        Check(!Encoding.Latin1.GetString(stored).Contains("/FlateDecode", StringComparison.Ordinal) && smallest.Length <= fastest.Length && fastest.Length < stored.Length,
            $"Flate at the option's level: stored {stored.Length:N0}, level 1 {fastest.Length:N0}, level 9 {smallest.Length:N0} bytes");
        foreach (byte[] bytes in new[] { stored, smallest, fastest, plainStreams })
        {
            using var document = PdfReadWritten(bytes);
            var page = document.ReadPage(0, DocumentParts.All);
            Check(page.Text == "Quarterly report" && page.Paths.Count == 1 && page.Images.Count == 1 && PdfSamePixels(page.Images[0].Decode().Pixels, PdfLineArt(1, 64, 32).Pixels),
                "every level reads the same page");
        }

        // The information in any script, other entries, an identifier when given; dates and an identifier made when not deterministic.
        var info = new DocumentInfo
        {
            Title = "Résumé of the quarter — 2026", Author = "Office team (north)", Subject = "Quarterly results", Keywords = "report, results", Creator = "Idrak tests",
            Producer = "Idrak", Created = new DateTimeOffset(2026, 11, 18, 9, 30, 0, TimeSpan.FromHours(2)), Entries = new Dictionary<string, string> { ["Review status"] = "Approved ✓" }.AsReadOnly(),
        };
        byte[] id = [.. Enumerable.Range(0, 16).Select(i => (byte)(i * 17))];
        byte[] described = PdfWrite(b =>
        {
            b.Info = info;
            Build(b);
        }, PdfWriteDefaults with { Id = id });
        using (var document = PdfReadWritten(described))
        {
            var read = document.Info;
            var trailerId = document.Objects.Resolve(document.Objects.Trailer["ID"u8]).AsArray;
            Check(read.Title == info.Title && read.Author == info.Author && read.Subject == info.Subject && read.Keywords == info.Keywords && read.Creator == info.Creator
                  && read.Producer == info.Producer && read.Created == info.Created && read.Modified is null && read.Entries.GetValueOrDefault("Review status") == "Approved ✓"
                  && trailerId is { Count: 2 } && trailerId[0].AsString!.Bytes.AsSpan().SequenceEqual(id),
                $"the information and the identifier: {read}");
        }

        byte[] dated = PdfWrite(Build, PdfWriteDefaults with { Deterministic = false });
        using (var document = PdfReadWritten(dated))
        {
            Check(document.Info.Created is not null && document.Info.Modified is not null && document.Objects.Resolve(document.Objects.Trailer["ID"u8]).AsArray is { Count: 2 },
                "not deterministic: dates and an identifier made");
        }

        foreach (string bad in new[] { "1.8", "3.0", "abc", "1.", "" })
        {
            ExpectRefusal<ArgumentException>(() => DocumentWriters.Create(new MemoryStream(), "pdf", PdfWriteDefaults with { Version = bad }).Dispose(), "DocumentWriteOptions.Version", $"version '{bad}'");
        }

        PdfSaveWritten("file-modern", modern, [(300, 200, "Quarterly report")]);
        PdfSaveWritten("file-classic", classic, [(300, 200, "Quarterly report")]);
        PdfSaveWritten("file-stored", stored, [(300, 200, "Quarterly report")]);
        PdfSaveWritten("file-information", described, [(300, 200, "Quarterly report")]);
    }

    private static void PdfWriteImages(Device device)
    {
        CpuOnly(device);
        var grey = PdfNoise(1, 1, 96, 21);
        var rgb = PdfNoise(2, 3, 40, 27);
        var lineArt = PdfLineArt(3, 120, 80);
        var planes = new ImageData([[.. Enumerable.Range(0, 30 * 20).Select(i => (byte)(i * 7))]], 20, 30, [.. Enumerable.Range(0, 256).Select(k => k * (1f / 255f))]);
        var placed = new[]
        {
            (grey, new DocumentBox(10, 10, 96, 21)), (rgb, new DocumentBox(150, 10, 38, 54)), (lineArt, new DocumentBox(10, 100, 240, 160)), (planes, new DocumentBox(300, 10, 60, 40)),
        };
        byte[] exact = PdfWrite(b =>
        {
            var page = b.AddPage(400, 300);
            foreach (var (image, box) in placed)
            {
                page.Image(image, box, DocumentImageEncoding.Flate);
            }
        });
        using (var document = PdfReadWritten(exact))
        {
            var page = document.ReadPage(0, DocumentParts.Images);
            Check(page.Images.Count == placed.Length, $"{page.Images.Count} images read");
            for (int i = 0; i < placed.Length; i++)
            {
                var (image, box) = placed[i];
                var read = page.Images[i];
                var pixels = read.Decode();
                float[] expected = image.TryGetPlanes(out var held, out _) ? [.. held[0].ToArray().Select(v => v / 255f)] : image.Pixels;
                Check(read.Encoding == DocumentImage.Raw && pixels.Channels == image.Channels && pixels.Width == image.Width && pixels.Height == image.Height
                      && PdfSamePixels(pixels.Pixels, expected), $"image {i}: its pixels exact");
                Check(PdfNear(read.Box, box), $"image {i}: box {read.Box}, written {box}");
            }

            // The predictor where it pays (line art), none where it does not (noise).
            var file = document.Objects;
            var xobjects = file.Dictionary(file.Dictionary(PdfPageDictionaries(file)[0]["Resources"u8])!["XObject"u8])!;
            bool Predicted(int k) => file.Resolve(xobjects[Encoding.ASCII.GetBytes($"Im{k}")]).AsStream!.Dictionary.Contains("DecodeParms"u8);
            Check(!Predicted(1) && !Predicted(2) && Predicted(3), "the PNG predictor on line art alone");
        }

        // JPEG: the image stream is the jpeg encoder's file at the options' quality; its pixels the codec's own round trip.
        var photo = PdfNoise(4, 3, 64, 48);
        byte[] jpegFile = PdfWrite(b => b.AddPage(200, 200).Image(photo, new DocumentBox(0, 0, 128, 96), DocumentImageEncoding.Jpeg), PdfWriteDefaults with { JpegQuality = 80 });
        using (var document = PdfReadWritten(jpegFile))
        {
            var read = document.ReadPage(0, DocumentParts.Images).Images.Single();
            byte[] expected = ImageEncoders.Get("jpeg").Encode(photo, new ImageEncodeOptions { Quality = 80 });
            Check(read.Encoding == DocumentImage.Jpeg && read.Bytes().Span.SequenceEqual(expected)
                  && read.Decode().Pixels.AsSpan().SequenceEqual(ImageCodecs.Get("jpeg").Decode(JpegEncoder.Encode(photo, 80)).Pixels), "JPEG: the encoder's bytes, the codec's pixels");
        }

        // Auto: every D6 input measured as the sample measured it, and stored as it expects.
        var (inputs, folder) = PdfWriteInputs();
        var images = new List<(string Name, ImageData Image, string Expected)>();
        foreach (var entry in inputs["images"]!.AsArray())
        {
            string name = (string)entry!["name"]!;
            var image = ImageCodecs.Get("png").Decode(File.ReadAllBytes(Path.Combine(folder, (string)entry["file"]!)));
            int w = image.Width, h = image.Height, c = image.Channels;
            byte[] levels = new byte[w * h * c];
            PdfWriterImages.Levels(image, levels);
            var measures = PdfWriterImages.Measure(levels, w, h, c, 0);
            Check(c == (int)entry["channels"]! && measures.Levels <= (int)entry["distinctColours"]! && (c == 3 || measures.Levels == (int)entry["distinctColours"]!)
                  && Math.Abs(Math.Round(measures.SmallSteps, 4) - (double)entry["smallSteps"]!) < 1e-9 && Math.Abs(Math.Round(measures.LargeSteps, 4) - (double)entry["largeSteps"]!) < 1e-9,
                $"{name}: measured {measures}, the sample's {entry["distinctColours"]} colours, {entry["smallSteps"]} small and {entry["largeSteps"]} large steps");
            images.Add((name, image, (string)entry["expectedAuto"]!));
        }

        Check(images.Count == 7, $"{images.Count} D6 inputs");
        byte[] auto = PdfWrite(b =>
        {
            foreach (var (_, image, _) in images)
            {
                b.AddPage(image.Width * 0.75f, image.Height * 0.75f).Image(image, new DocumentBox(0, 0, image.Width * 0.75f, image.Height * 0.75f), DocumentImageEncoding.Auto);
            }
        });
        using (var document = PdfReadWritten(auto))
        {
            for (int i = 0; i < images.Count; i++)
            {
                var (name, image, expected) = images[i];
                var read = document.ReadPage(i, DocumentParts.Images).Images.Single();
                string chosen = read.Encoding == DocumentImage.Jpeg ? "Jpeg" : "Flate";
                Check(chosen == expected, $"{name}: Auto chose {chosen}, the sample expects {expected}");
                Check(chosen == "Flate" ? PdfSamePixels(read.Decode().Pixels, image.Pixels)
                        : read.Decode().Pixels.AsSpan().SequenceEqual(ImageCodecs.Get("jpeg").Decode(JpegEncoder.Encode(image, 90)).Pixels),
                    $"{name}: stored exact, or the codec's round trip");
            }
        }

        // The options' encoding over Auto: a photograph stored exact, line art as JPEG.
        var photograph = images.First(i => i.Expected == "Jpeg").Image;
        byte[] forced = PdfWrite(b => b.AddPage(100, 100).Image(photograph, new DocumentBox(0, 0, 100, 100), DocumentImageEncoding.Auto),
            PdfWriteDefaults with { ImageEncoding = DocumentImageEncoding.Flate });
        byte[] forcedJpeg = PdfWrite(b => b.AddPage(100, 100).Image(lineArt, new DocumentBox(0, 0, 100, 100), DocumentImageEncoding.Auto),
            PdfWriteDefaults with { ImageEncoding = DocumentImageEncoding.Jpeg });
        using (var a = PdfReadWritten(forced))
        using (var j = PdfReadWritten(forcedJpeg))
        {
            Check(a.ReadPage(0, DocumentParts.Images).Images.Single().Encoding == DocumentImage.Raw && j.ReadPage(0, DocumentParts.Images).Images.Single().Encoding == DocumentImage.Jpeg,
                "the options' encoding wins over Auto");
        }

        ExpectRefusal<ArgumentException>(() => PdfWrite(b => b.AddPage(10, 10).Image(new ImageData(new float[2 * 4], 2, 2, 2), new DocumentBox(0, 0, 5, 5), DocumentImageEncoding.Flate)), "grey", "two channels");
        ExpectRefusal<ArgumentOutOfRangeException>(() => PdfWrite(b => b.AddPage(10, 10).Image(grey, new DocumentBox(0, 0, 0, 5), DocumentImageEncoding.Flate)), "above 0", "an empty box");
        PdfSaveWritten("images-exact", exact, [(400, 300, "")]);
        PdfSaveWritten("images-jpeg", jpegFile, [(200, 200, "")]);
        PdfSaveWritten("images-auto", auto, [.. images.Select(i => (i.Image.Width * 0.75f, i.Image.Height * 0.75f, ""))]);
    }

    // The D6 inputs' JSON and the samples' D6 folder.
    private static (JsonObject Inputs, string Folder) PdfWriteInputs()
    {
        string root = Environment.GetEnvironmentVariable("IDRAK_DOCUMENTS_SAMPLES") is { Length: > 0 } set ? set : RepositoryPath("test-output", "documents", "samples");
        string folder = Path.Combine(root, "d6"), path = Path.Combine(folder, "inputs.json");
        Check(File.Exists(path), $"The D6 samples are not in {folder}. Make them with: python -I tools/documents/make_fixtures.py -o test-output/documents/samples (see tools/documents/README.md), or set IDRAK_DOCUMENTS_SAMPLES to where they are.");
        return (JsonNode.Parse(File.ReadAllText(path))!.AsObject(), folder);
    }

    // The paths of a vector page, each with the clip it is drawn under.
    private static List<(DocumentPath Path, int Clip)> PdfVectorPage(IDocumentPageBuilder page)
    {
        var written = new List<(DocumentPath, int)>();
        void Draw(DocumentPath path, int clip)
        {
            page.Path(path);
            written.Add((path, clip));
        }

        Draw(new DocumentPath([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close], [20, 20, 120.5f, 30.25f, 60, 95])
        {
            Fill = DocumentFill.NonZero, FillColour = DocumentColour.Rgb(0.2f, 0.4f, 0.6f),
        }, -1);
        Draw(new DocumentPath(PdfBoxVerbs, PdfBoxPoints(new DocumentBox(140, 20, 80, 40))), -1);

        // A star: even-odd leaves its middle empty; filled and stroked.
        Draw(new DocumentPath([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close],
            [300, 20, 330, 110, 255, 55, 345, 55, 270, 110])
        {
            Fill = DocumentFill.EvenOdd, FillColour = DocumentColour.Cmyk(0.1f, 0.6f, 0f, 0.05f), Stroke = 1.5f, StrokeColour = DocumentColour.Grey(0.25f),
            Join = DocumentLineJoin.Miter, MiterLimit = 3.5f,
        }, -1);

        // Curves: a cubic, its v and y forms (a control at an end), a quadratic.
        Draw(new DocumentPath([OutlineVerb.Move, OutlineVerb.Cubic, OutlineVerb.Cubic, OutlineVerb.Cubic, OutlineVerb.Quadratic],
            [20, 150, 40, 120, 80, 180, 100, 150, 100, 150, 130, 120, 150, 150, 170, 180, 200, 140, 200, 140, 230, 120, 260, 150])
        {
            Fill = DocumentFill.None, Stroke = 2.5f, Cap = DocumentLineCap.Round, Join = DocumentLineJoin.Round, StrokeColour = DocumentColour.Rgb(0.8f, 0.1f, 0.1f).WithAlpha(0.5f),
        }, -1);
        Draw(new DocumentPath([OutlineVerb.Move, OutlineVerb.Line], [20, 200, 360, 200])
        {
            Fill = DocumentFill.None, Stroke = 1f, Cap = DocumentLineCap.Square, Dash = new float[] { 6, 2, 1, 2 }, DashPhase = 3f, StrokeColour = DocumentColour.Black,
        }, -1);

        // Clips nested under Save and Restore: an even-odd ring, then a box within it.
        page.Save();
        page.Clip(new DocumentPath([.. PdfBoxVerbs, .. PdfBoxVerbs], [.. PdfBoxPoints(new DocumentBox(20, 220, 180, 160)), .. PdfBoxPoints(new DocumentBox(60, 260, 100, 80))])
        {
            Fill = DocumentFill.EvenOdd,
        });
        Draw(new DocumentPath(PdfBoxVerbs, PdfBoxPoints(new DocumentBox(0, 210, 220, 180))) { FillColour = DocumentColour.Grey(0.6f).WithAlpha(0.75f) }, 0);
        page.Save();
        page.Clip(new DocumentPath(PdfBoxVerbs, PdfBoxPoints(new DocumentBox(100, 200, 200, 100))));
        Draw(new DocumentPath([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close], [40, 230, 260, 240, 150, 360])
        {
            Fill = DocumentFill.NonZero, FillColour = DocumentColour.Rgb(0.1f, 0.7f, 0.3f), Stroke = 0f,
        }, 1);
        page.Restore();
        Draw(new DocumentPath([OutlineVerb.Move, OutlineVerb.Line], [30, 230, 190, 370]) { Fill = DocumentFill.None, Stroke = 4f, Cap = DocumentLineCap.Butt }, 0);
        page.Restore();
        Draw(new DocumentPath(PdfBoxVerbs, PdfBoxPoints(new DocumentBox(250, 250, 100, 100))) { Stroke = 0.5f, StrokeColour = DocumentColour.Rgb(0, 0, 1) }, -1);
        return written;
    }

    // A path as read against the one written (points to a thousandth of a point, as the suite asks), or null when they agree.
    private static string? PdfPathDifference(DocumentPath written, DocumentPath read, int clip)
    {
        const float Tolerance = 1e-3f;
        if (!written.Verbs.Span.SequenceEqual(read.Verbs.Span))
        {
            return $"verbs {string.Join(" ", read.Verbs.ToArray())}, written {string.Join(" ", written.Verbs.ToArray())}";
        }

        var a = written.Points.Span;
        var b = read.Points.Span;
        for (int i = 0; i < a.Length; i++)
        {
            if (MathF.Abs(a[i] - b[i]) > Tolerance)
            {
                return $"point value {i}: {b[i]}, written {a[i]}";
            }
        }

        if (written.Fill != read.Fill || written.Stroke.HasValue != read.Stroke.HasValue || (written.Stroke is { } w && MathF.Abs(w - read.Stroke!.Value) > Tolerance))
        {
            return $"fill {read.Fill} stroke {read.Stroke}, written {written.Fill} {written.Stroke}";
        }

        if (written.Fill != DocumentFill.None && PdfColourDifference(written.FillColour, read.FillColour) is { } fill)
        {
            return "fill colour " + fill;
        }

        if (written.Stroke is not null)
        {
            if (written.Cap != read.Cap || written.Join != read.Join || MathF.Abs(written.MiterLimit - read.MiterLimit) > Tolerance
                || written.Dash.Length != read.Dash.Length || written.Dash.Span.ToArray().Zip(read.Dash.Span.ToArray()).Any(d => MathF.Abs(d.First - d.Second) > Tolerance)
                || MathF.Abs(written.DashPhase - read.DashPhase) > Tolerance)
            {
                return $"the line state {read.Cap} {read.Join} {read.MiterLimit} [{string.Join(" ", read.Dash.ToArray())}] {read.DashPhase}";
            }

            if (PdfColourDifference(written.StrokeColour, read.StrokeColour) is { } stroke)
            {
                return "stroke colour " + stroke;
            }
        }

        return read.Clip == clip ? null : $"clip {read.Clip}, written under {clip}";
    }

    private static string? PdfColourDifference(DocumentColour written, DocumentColour read)
    {
        if (written.Space != read.Space || MathF.Abs(written.Alpha - read.Alpha) > 1e-6f)
        {
            return $"{read}, written {written}";
        }

        for (int c = 0; c < written.Count; c++)
        {
            if (MathF.Abs(written[c] - read[c]) > 1e-6f)
            {
                return $"{read}, written {written}";
            }
        }

        return null;
    }

    private static void PdfWriteVector(Device device)
    {
        CpuOnly(device);
        List<(DocumentPath Path, int Clip)> written = [];
        byte[] bytes = PdfWrite(b => written = PdfVectorPage(b.AddPage(400, 400)));
        using (var document = PdfReadWritten(bytes))
        {
            var page = document.ReadPage(0, DocumentParts.All);
            Check(page.Paths.Count == written.Count, $"{page.Paths.Count} paths read, {written.Count} written");
            for (int i = 0; i < written.Count; i++)
            {
                string? difference = PdfPathDifference(written[i].Path, page.Paths[i], written[i].Clip);
                Check(difference is null, $"path {i}: {difference}");
                Check(page.Paths[i].Order == i, $"path {i} is at place {page.Paths[i].Order}");
            }

            Check(page.Clips.Count == 2 && page.Clips[0].Fill == DocumentFill.EvenOdd && page.Clips[0].Clip == -1 && page.Clips[1].Clip == 0 && page.Clips[1].Fill == DocumentFill.NonZero,
                $"the clips: {page.Clips.Count}, nested");
            string content = PdfContentText(document, 0);
            foreach (string op in new[] { " re\n", " c\n", " v\n", " y\n", "\nh\n", "\nf\n", "\nB*\n", "\nS\n", "\nB\n", " d\n", " gs\n", " k\n", " G\n", "W* n\n", "W n\n", "\nq\n", "\nQ\n" })
            {
                Check(content.Contains(op, StringComparison.Ordinal), $"the content has '{op.Trim()}'");
            }
        }

        ExpectRefusal<InvalidOperationException>(() => PdfWrite(b => b.AddPage(10, 10).Restore()), "nothing is saved", "a Restore without a Save");
        ExpectRefusal<ArgumentException>(() => PdfWrite(b => b.AddPage(10, 10).Path(new DocumentPath([OutlineVerb.Line], [1, 2]))), "starts with a move", "a path without a move");
        ExpectRefusal<ArgumentException>(() => PdfWrite(b => b.AddPage(10, 10).Path(new DocumentPath([OutlineVerb.Move, OutlineVerb.Line], [1, 2, 3, 4]) { Stroke = 1, Dash = new float[] { 0, 0 } })),
            "not all 0", "an empty dash");
        PdfSaveWritten("vector", bytes, [(400, 400, "")]);
    }

    private static void PdfWriteText(Device device)
    {
        CpuOnly(device);
        var face = SubsetFace();
        float upem = face.Metrics.UnitsPerEm;
        const float Size = 14f;
        string[] lines = ["Quarterly report", "Revenue grew by 12 percent.", "The team met every target."];
        string arabic = "تقرير الربع الأول لا";
        int e = face.Glyph('e'), grave = face.Glyph('`'), ligature = face.Glyph(0xFEFB);
        Check(e > 0 && grave > 0 && ligature > 0, "the test font's glyphs");
        var runs = new List<(DocumentGlyph[] Glyphs, string Text, DocumentTextMode Mode, bool Actual, float Size)>();
        byte[] bytes = PdfWrite(b =>
        {
            var page = b.AddPage(400, 300);
            float y = 40;
            foreach (string line in lines)
            {
                var glyphs = PdfShaped(face, line, "Latn", false, 30, y, Size);
                page.Glyphs(face, Size, glyphs, line, DocumentTextMode.Fill, DocumentColour.Grey(0.1f));
                runs.Add((glyphs, line, DocumentTextMode.Fill, false, Size));
                y += 20;
            }

            // The presentation form's own character first (the face maps it to its glyph), then the same glyph shaped from lam and alef.
            var form = new[] { new DocumentGlyph(ligature, 30, y, face.Advance(ligature) * Size / upem) };
            page.Glyphs(face, Size, form, "ﻻ", DocumentTextMode.Fill, DocumentColour.Black);
            runs.Add((form, "ﻻ", DocumentTextMode.Fill, false, Size));
            y += 20;
            var rtl = PdfShaped(face, arabic, "Arab", true, 30, y, Size);
            Check(rtl.Any(g => g.Glyph == ligature), "the shaper makes the lam-alef ligature");
            page.Glyphs(face, Size, rtl, arabic, DocumentTextMode.Fill, DocumentColour.Rgb(0.1f, 0.2f, 0.5f));
            runs.Add((rtl, arabic, DocumentTextMode.Fill, true, Size));
            y += 20;

            // A character drawn by two glyphs (a letter and its accent above it).
            var drawn = new[] { new DocumentGlyph(e, 30, y, face.Advance(e) * Size / upem), new DocumentGlyph(grave, 31.5f, y - 2, face.Advance(grave) * Size / upem) };
            page.Glyphs(face, Size, drawn, "è", DocumentTextMode.Stroke, DocumentColour.Rgb(0.5f, 0, 0));
            runs.Add((drawn, "è", DocumentTextMode.Stroke, true, Size));
            y += 20;
            var hidden = PdfShaped(face, "A hidden line over a scan.", "Latn", false, 30, y, 9);
            page.Glyphs(face, 9, hidden, "A hidden line over a scan.", DocumentTextMode.Hidden, DocumentColour.Black);
            runs.Add((hidden, "A hidden line over a scan.", DocumentTextMode.Hidden, false, 9));

            // A second page in the same face: one font for the document.
            var second = b.AddPage(300, 100);
            second.Glyphs(face, 12, PdfShaped(face, "Office move planned for June.", "Latn", false, 20, 50, 12), "Office move planned for June.", DocumentTextMode.FillStroke, DocumentColour.Black);
        });

        using (var document = PdfReadWritten(bytes))
        {
            var page = document.ReadPage(0, DocumentParts.All);
            Check(page.Runs.Count == runs.Count, $"{page.Runs.Count} runs read, {runs.Count} written: {string.Join(" | ", page.Runs.Select(r => r.Text))}");
            for (int i = 0; i < runs.Count; i++)
            {
                var (glyphs, text, mode, actual, size) = runs[i];
                var run = page.Runs[i];
                Check(run.Text == text && run.Mode == mode && run.Logical == actual && run.Order == i, $"run {i}: \"{run.Text}\" {run.Mode} logical {run.Logical}, written \"{text}\" {mode}");
                var read = run.Glyphs.Span;
                Check(read.Length == glyphs.Length, $"run {i}: {read.Length} glyphs, {glyphs.Length} written");
                for (int g = 0; g < glyphs.Length; g++)
                {
                    Check(read[g].Glyph == glyphs[g].Glyph && MathF.Abs(read[g].X - glyphs[g].X) <= 1e-3f && MathF.Abs(read[g].Y - glyphs[g].Y) <= 1e-3f,
                        $"run {i}, glyph {g}: {read[g]}, written {glyphs[g]}");
                }

                // Its box: from the first glyph's origin to the last one's advance, the face's descent to ascent at its size (runs on one baseline).
                if (glyphs.All(g => g.Y == glyphs[0].Y))
                {
                    float left = glyphs.Min(g => g.X), right = glyphs.Max(g => g.X + face.Advance(g.Glyph) * size / upem);
                    float top = glyphs[0].Y - face.Metrics.Ascender * size / upem, height = (face.Metrics.Ascender - face.Metrics.Descender) * size / upem;
                    Check(MathF.Abs(run.Box.X - left) < 0.01f && MathF.Abs(run.Box.Right - right) < 0.01f && MathF.Abs(run.Box.Y - top) < 0.01f && MathF.Abs(run.Box.Height - height) < 0.01f,
                        $"run {i}: box {run.Box}, expected ({left}, {top}) to {right}, {height} high");
                }

                Check(run.Face is not null && MathF.Abs(run.FontSize - size) < 1e-4f && run.Font!.Contains('+', StringComparison.Ordinal), $"run {i}: a subset font {run.Font} at {run.FontSize}");
            }

            Check(page.Lines.Any(l => l.Text == arabic && l.RightToLeft), $"the right-to-left line reads in logical order: {string.Join(" | ", page.Lines.Select(l => l.Text))}");
            string content = PdfContentText(document, 0);
            Check(content.Split("/ActualText").Length - 1 == 2 && content.Contains("\n3 Tr\n", StringComparison.Ordinal) && content.Contains("\n1 Tr\n", StringComparison.Ordinal),
                "two runs carry /ActualText (the ligature's and the accent's), the hidden one render mode 3, the stroked one 1");
            var second = document.ReadPage(1, DocumentParts.All);
            Check(second.Text == "Office move planned for June." && second.Runs[0].Mode == DocumentTextMode.FillStroke, "the second page");

            // One font for the document, its program a subset with the ids kept, its widths the advances.
            var file = document.Objects;
            var fonts = file.LiveObjects().Select(o => file.Get(o.Number).AsDictionary).Where(d => d is not null && d["Subtype"u8].IsName("Type0"u8)).ToList();
            Check(fonts.Count == 1, $"{fonts.Count} Type0 fonts for one face");
            var cidFont = file.Dictionary(file.Resolve(fonts[0]!["DescendantFonts"u8]).AsArray![0])!;
            var descriptor = file.Dictionary(cidFont["FontDescriptor"u8])!;
            byte[] programBytes;
            using (var program = file.Decode(file.Resolve(descriptor["FontFile2"u8]).AsStream!))
            {
                programBytes = program.Span.ToArray();
            }

            var embedded = FontFormats.Read(programBytes.AsMemory(), "embedded").Single();
            var used = runs.SelectMany(r => r.Glyphs).Select(g => g.Glyph).ToHashSet();
            int unused = Enumerable.Range(1, face.GlyphCount - 1).First(g => !used.Contains(g) && !face.Outline(g).IsEmpty);
            Check(embedded.GlyphCount == face.GlyphCount && embedded.Outline(unused).IsEmpty && used.All(g => embedded.Outline(g).Points.Span.SequenceEqual(face.Outline(g).Points.Span)),
                "the subset keeps the ids, the used glyphs' outlines, and empties the others");
            var widths = file.Resolve(cidFont["W"u8]).AsArray!;
            file.Resolve(widths[0]).TryInteger(out long firstGlyph);
            file.Resolve(file.Resolve(widths[1]).AsArray![0]).TryNumber(out double firstWidth);
            Check(Math.Abs(firstWidth - Math.Round(face.Advance((int)firstGlyph) * 1000.0 / upem, 3, MidpointRounding.AwayFromZero)) < 1e-6 && cidFont["CIDToGIDMap"u8].IsName("Identity"u8)
                  && fonts[0]!["Encoding"u8].IsName("Identity-H"u8), "/W from the advances, Identity-H, CIDToGIDMap Identity");
        }

        // A CFF face (its cubic outlines written as TrueType), a mapped legacy face, and a face embedded whole.
        var cff = FontFormats.Read(FontData("dejavu-sans-cff.otf")).Single();
        var legacyRaw = FontFormats.Get("opentype").Read(File.ReadAllBytes(FontData("arabic/FS_Hand_Style.ttf")), "FS_Hand_Style").Single();
        var legacy = FontMaps.Apply(legacyRaw);
        Check(FontMaps.MappedBy(legacy) is not null, "FS_Hand_Style is mapped by a library map");
        string legacyText = "سلام عليكم";
        var legacyGlyphs = PdfShaped(legacy, legacyText, "Arab", true, 20, 100, 24);
        byte[] others = PdfWrite(b =>
        {
            var page = b.AddPage(400, 200);
            page.Glyphs(cff, 16, PdfShaped(cff, "Newsletter for the spring", "Latn", false, 20, 40, 16), "Newsletter for the spring", DocumentTextMode.Fill, DocumentColour.Black);
            page.Glyphs(legacy, 24, legacyGlyphs, legacyText, DocumentTextMode.Fill, DocumentColour.Black);
        });
        byte[] whole = PdfWrite(b => b.AddPage(300, 100).Glyphs(face, 12, PdfShaped(face, lines[0], "Latn", false, 20, 50, 12), lines[0], DocumentTextMode.Fill, DocumentColour.Black),
            PdfWriteDefaults with { EmbedFonts = DocumentFontEmbedding.Whole });
        using (var document = PdfReadWritten(others))
        {
            var page = document.ReadPage(0, DocumentParts.All);
            Check(page.Runs.Count == 2 && page.Runs[0].Text == "Newsletter for the spring" && page.Runs[1].Text == legacyText
                  && page.Runs[1].Glyphs.Span.ToArray().Select(g => g.Glyph).SequenceEqual(legacyGlyphs.Select(g => g.Glyph)),
                $"the CFF and the mapped faces' runs: {string.Join(" | ", page.Runs.Select(r => r.Text))}");
        }

        using (var document = PdfReadWritten(whole))
        {
            var run = document.ReadPage(0, DocumentParts.All).Runs.Single();
            Check(run.Text == lines[0] && !run.Font!.Contains('+', StringComparison.Ordinal) && run.Face!.GlyphCount == face.GlyphCount
                  && Enumerable.Range(0, face.GlyphCount).All(g => run.Face!.Outline(g).Points.Span.SequenceEqual(face.Outline(g).Points.Span)),
                "embedded whole: every glyph's outline, no subset tag");
        }

        ExpectRefusal<ArgumentOutOfRangeException>(() => PdfWrite(b => b.AddPage(10, 10).Glyphs(face, 10, [new DocumentGlyph(face.GlyphCount, 0, 0, 1)], "x", DocumentTextMode.Fill, default)),
            "not one of the face's", "a glyph past the face");
        ExpectRefusal<ArgumentOutOfRangeException>(() => PdfWrite(b => b.AddPage(10, 10).Glyphs(face, 0, [new DocumentGlyph(1, 0, 0, 1)], "x", DocumentTextMode.Fill, default)),
            "above 0", "a size of 0");
        PdfSaveWritten("text", bytes);
        PdfSaveWritten("text-latin", PdfWrite(b =>
        {
            var page = b.AddPage(400, 120);
            for (int i = 0; i < lines.Length; i++)
            {
                page.Glyphs(face, Size, PdfShaped(face, lines[i], "Latn", false, 30, 40 + 20 * i, Size), lines[i], DocumentTextMode.Fill, DocumentColour.Black);
            }
        }), [(400, 120, string.Join("\n", lines))]);
        PdfSaveWritten("text-faces", others);
        PdfSaveWritten("text-whole", whole, [(300, 100, lines[0])]);
    }

    private static void PdfWriteScanAndShapes(Device device)
    {
        CpuOnly(device);
        var (inputs, folder) = PdfWriteInputs();
        var scan = ImageCodecs.Get("png").Decode(File.ReadAllBytes(Path.Combine(folder, "inputs.scan-grey.png")));
        string[] lines = [.. inputs["text"]!["lines"]!.AsArray().Select(l => (string)l!)];
        var face = SubsetFace();
        float width = scan.Width * 72f / 100, height = scan.Height * 72f / 100;
        byte[] searchable = PdfWrite(b =>
        {
            var page = b.AddPage(width, height);
            page.Image(scan, new DocumentBox(0, 0, width, height), DocumentImageEncoding.Auto);
            for (int i = 0; i < lines.Length; i++)
            {
                page.Glyphs(face, 11, PdfShaped(face, lines[i], "Latn", false, 60, 90 + 22 * i, 11), lines[i], DocumentTextMode.Hidden, DocumentColour.Black);
            }
        });
        using (var document = PdfReadWritten(searchable))
        {
            var page = document.ReadPage(0, DocumentParts.All);
            var image = page.Images.Single();
            Check(image.Encoding == DocumentImage.Jpeg && PdfNear(image.Box, new DocumentBox(0, 0, width, height)) && image.Order == 0
                  && image.Decode().Pixels.AsSpan().SequenceEqual(ImageCodecs.Get("jpeg").Decode(JpegEncoder.Encode(scan, 90)).Pixels),
                $"the scan's image: {image.Encoding} in {image.Box}");
            Check(page.Runs.Count == lines.Length && page.Runs.All(r => r.Hidden) && page.Runs.Select(r => r.Text).SequenceEqual(lines) && page.Text == string.Join("\n", lines),
                $"its hidden text: {page.Text}");
        }

        PdfSaveWritten("searchable-scan", searchable, [(width, height, string.Join("\n", lines))]);

        // The page-of-shapes helper: areas, strokes (dashed, dotted, closed), and laid-out lines, all as vector content.
        var plan = new ShapePlan(
            [
                new ShapePath([40, 40, 560, 40, 560, 300, 40, 300], closed: true, width: 1.5f),
                new ShapePath([40, 120, 560, 120], width: 0.5f, dash: [4, 2], tone: 0.6f),
                new ShapePath([300, 200]),
                new ShapePath([60, 260, 120, 220, 180, 260], width: 2f, dash: [0, 3]),
            ],
            [new ShapeArea([40, 40, 560, 40, 560, 80, 40, 80], 0.15f)]);
        var style = new DocumentShapeStyle { Scale = 0.75f, StrokeWidth = 2f };
        string text = "Project dashboard\nOpen tasks: 14";
        var block = TextLayout.Layout(text, new TextStyle(face, 24));
        byte[] shapes = PdfWrite(b =>
        {
            var page = b.AddPage(600 * 0.75f, 340 * 0.75f);
            DocumentShapes.Write(page, plan, style);
            DocumentShapes.Write(page, block, text, 60, 50, style);
        });
        var expected = DocumentShapes.Paths(plan, style);
        using (var document = PdfReadWritten(shapes))
        {
            var page = document.ReadPage(0, DocumentParts.All);
            Check(page.Paths.Count == expected.Count && expected.Select((p, i) => PdfPathDifference(p, page.Paths[i], -1)).All(d => d is null),
                $"the shapes' paths: {string.Join("; ", expected.Select((p, i) => i < page.Paths.Count ? PdfPathDifference(p, page.Paths[i], -1) : "missing"))}");
            Check(expected[0].Fill == DocumentFill.NonZero && expected[1].Stroke == 2f * 1.5f * 0.75f && expected[1].Verbs.Span[^1] == OutlineVerb.Close
                  && expected[2].Dash.Span.SequenceEqual([4 * 0.75f, 2 * 0.75f]) && expected[3].Verbs.Span.SequenceEqual([OutlineVerb.Move, OutlineVerb.Line]) && expected[4].Dash.Span[0] == 0,
                "an area filled, a closed path closed, a dot a line of no length, dashes in stroke widths");
            Check(page.Runs.Select(r => r.Text).SequenceEqual(block.Lines.Select(l => l.Text)) && page.Text == text, $"the laid-out lines: {page.Text}");
            var firstLine = block.Lines[0];
            var glyphs = page.Runs[0].Glyphs.ToArray();
            Check(glyphs.Length == firstLine.Glyphs.Count && Enumerable.Range(0, glyphs.Length).All(g => glyphs[g].Glyph == firstLine.Glyphs[g].Glyph
                      && MathF.Abs(glyphs[g].X - (60 + firstLine.Glyphs[g].X) * 0.75f) < 1e-3f && MathF.Abs(glyphs[g].Y - (50 + firstLine.Glyphs[g].Y) * 0.75f) < 1e-3f),
                "the glyphs where the layout placed them");
        }

        PdfSaveWritten("shapes", shapes, [(450, 255, text)]);
    }

    private static void PdfWriteDeterministicAndLimits(Device device)
    {
        CpuOnly(device);
        var face = SubsetFace();
        var notes = Encoding.UTF8.GetBytes("Notes for the report: the figures are final.\n");
        var table = Encoding.UTF8.GetBytes("region,sales\nnorth,12\nsouth,9\n");
        void Build(IDocumentBuilder b)
        {
            b.Info = new DocumentInfo { Title = "A technical report", Author = "The office team" };
            PdfVectorPage(b.AddPage(400, 400));
            b.AddPage(300, 300).Image(PdfNoise(9, 3, 50, 40), new DocumentBox(20, 20, 200, 160), DocumentImageEncoding.Jpeg);
            b.AddPage(300, 300).Image(PdfLineArt(1, 90, 60), new DocumentBox(20, 20, 180, 120), DocumentImageEncoding.Flate);
            b.AddPage(300, 100).Glyphs(face, 12, PdfShaped(face, "Thank you for your letter.", "Latn", false, 20, 50, 12), "Thank you for your letter.", DocumentTextMode.Fill, DocumentColour.Black);
            b.Attach("notes (final) é.txt", "text/plain", new MemoryStream(notes));
            b.Attach("data.csv", "text/csv", new MemoryStream(table));
        }

        byte[] first = PdfWrite(Build), second = PdfWrite(Build);
        var parallel = new byte[4][];
        Parallel.For(0, parallel.Length, i => parallel[i] = PdfWrite(Build));
        Check(first.AsSpan().SequenceEqual(second) && parallel.All(p => p.AsSpan().SequenceEqual(first)), "the same input, the same bytes (on several threads too)");
        using (var document = PdfReadWritten(first))
        {
            var attachments = document.Attachments;
            Check(attachments.Count == 2 && attachments[0].Name == "data.csv" && attachments[1].Name == "notes (final) é.txt" && attachments[0].MediaType == "text/csv"
                  && attachments[1].MediaType == "text/plain" && attachments[1].Size == notes.Length, $"the attached files, in the name tree's order: {string.Join(", ", attachments.Select(a => a.Name))}");
            foreach (var (attachment, bytes) in new[] { (attachments[0], table), (attachments[1], notes) })
            {
                using var read = attachment.Read();
                using var copy = new MemoryStream();
                read.CopyTo(copy);
                Check(copy.ToArray().AsSpan().SequenceEqual(bytes), $"{attachment.Name}: its bytes");
            }

            Check(document.PageCount == 4 && document.Info.Title == "A technical report", "four pages and the information");
        }

        // Disposed without Finish: the pages written so far and the page in hand; no cross-reference, which the reader rebuilds.
        using (var destination = new MemoryStream())
        {
            int paths;
            using (var builder = DocumentWriters.Create(destination, "pdf", PdfWriteDefaults))
            {
                paths = PdfVectorPage(builder.AddPage(400, 400)).Count;
                builder.AddPage(300, 300).Image(PdfLineArt(3, 40, 30), new DocumentBox(10, 10, 80, 60), DocumentImageEncoding.Flate);
                builder.AddPage(300, 100).Glyphs(face, 12, PdfShaped(face, "Unfinished", "Latn", false, 20, 50, 12), "Unfinished", DocumentTextMode.Fill, DocumentColour.Black);
            }

            byte[] unfinished = destination.ToArray();
            string tail = Encoding.Latin1.GetString(unfinished, Math.Max(0, unfinished.Length - 200), Math.Min(200, unfinished.Length));
            Check(!tail.Contains("%%EOF", StringComparison.Ordinal) && !tail.Contains("startxref", StringComparison.Ordinal) && tail.Contains("trailer\n<</Size ", StringComparison.Ordinal),
                $"no cross-reference written, a trailer naming the catalog: ...{tail[^Math.Min(60, tail.Length)..]}");
            using var document = PdfReadWritten(unfinished);
            Check(document.Info.Repaired && document.PageCount == 3, $"repaired: {document.Info.Repaired}, {document.PageCount} pages");
            var vector = document.ReadPage(0, DocumentParts.All);
            var image = document.ReadPage(1, DocumentParts.All).Images.Single();
            Check(vector.Paths.Count == paths && PdfSamePixels(image.Decode().Pixels, PdfLineArt(3, 40, 30).Pixels), "its finished pages read back");
            var text = document.ReadPage(2, DocumentParts.All);
            Check(text.Issues.Any(i => i.Kind == DocumentIssue.MissingFont), $"the page in hand is there; its font, written at Finish, is not (counted): {string.Join("; ", text.Issues)}");

            // Damaged by design: kept apart from the finished files the checker reads.
            string damaged = RepositoryPath("test-output", "documents", "written-unfinished");
            Directory.CreateDirectory(damaged);
            File.WriteAllBytes(Path.Combine(damaged, "unfinished.pdf"), unfinished);
        }

        // The parts later phases fill, and misuse.
        using (var builder = DocumentWriters.Create(new MemoryStream(), "pdf", PdfWriteDefaults))
        {
            var page = builder.AddPage(100, 100);
            ExpectRefusal<NotSupportedException>(() => builder.AddField(null!), "D9", "form fields");
            ExpectRefusal<NotSupportedException>(() => builder.AddOutline(null!), "D10", "bookmarks");
            ExpectRefusal<NotSupportedException>(() => builder.SetPageLabels([]), "D10", "page labels");
            ExpectRefusal<ArgumentNullException>(() => builder.Sign("Signature1", null!), "options", "signing without options");
            ExpectRefusal<NotSupportedException>(() => page.AddLink(null!), "D10", "links");
            ExpectRefusal<NotSupportedException>(() => page.AddAnnotation(null!), "D10", "annotations");
            builder.AddPage(100, 100);
            ExpectRefusal<InvalidOperationException>(() => page.Path(new DocumentPath(PdfBoxVerbs, PdfBoxPoints(new DocumentBox(0, 0, 5, 5)))), "was written", "a page used after the next one started");
            builder.Finish();
            ExpectRefusal<InvalidOperationException>(() => builder.AddPage(10, 10), "finished", "a page after Finish");
        }

        // A file by its path (the registry picks the writer by the extension).
        string path = Path.Combine(PdfWrittenFolder(), "created-by-path.pdf");
        using (var builder = DocumentWriters.Create(path, PdfWriteDefaults))
        {
            builder.AddPage(200, 100).Glyphs(face, 12, PdfShaped(face, "Created by its path", "Latn", false, 20, 50, 12), "Created by its path", DocumentTextMode.Fill, DocumentColour.Black);
            builder.Finish();
        }

        using (var document = DocumentFormats.Open(path))
        {
            Check(document.ReadPage(0, DocumentParts.Text).Text == "Created by its path", "a file written by its path");
        }

        File.WriteAllText(Path.ChangeExtension(path, ".json"), new JsonObject
        {
            ["schema"] = 1, ["name"] = "created-by-path", ["file"] = "created-by-path.pdf",
            ["pages"] = new JsonArray(new JsonObject { ["index"] = 0, ["width"] = 200, ["height"] = 100, ["text"] = "Created by its path" }),
        }.ToJsonString() + "\n");
        PdfSaveWritten("mixed", first);
    }

    private static void PdfWriteManyPages(Device device)
    {
        CpuOnly(device);
        const int Pages = 500;

        // A scan-like page (paper grain, printed bands) and a diagram, A4 at 100 dpi.
        var random = new Random(28);
        int width = 827, height = 1169;
        var scanPixels = new float[width * height];
        for (int y = 0; y < height; y++)
        {
            bool printed = y % 40 is >= 20 and < 32;
            for (int x = 0; x < width; x++)
            {
                float paper = (236 + random.Next(-12, 13)) / 255f;
                scanPixels[y * width + x] = printed && x % 9 < 6 && x is > 80 and < 740 ? 0.12f + random.Next(0, 8) / 255f : paper;
            }
        }

        var scan = new ImageData(scanPixels, 1, height, width);
        var diagram = PdfLineArt(1, width, height);
        var box = new DocumentBox(0, 0, 595.44f, 841.68f);

        // Each run: the time, the bytes written, the managed heap sampled after each page (garbage not yet collected
        // included) and after a full collection every 100 pages (what is live), its growth from page 20 to the end, and the
        // bytes allocated.
        (double Milliseconds, long Bytes, long Heap, long Live, long Growth, long Allocated) Run(ImageData image)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            long allocated = GC.GetTotalAllocatedBytes(precise: true), heap = 0, live = 0, settled = 0, collecting = 0;
            var counter = new PdfByteCounter();
            long start = Stopwatch.GetTimestamp();
            using (var builder = DocumentWriters.Create(counter, "pdf", PdfWriteDefaults))
            {
                for (int p = 0; p < Pages; p++)
                {
                    builder.AddPage(box.Width, box.Height).Image(image, box, DocumentImageEncoding.Auto);
                    heap = Math.Max(heap, GC.GetTotalMemory(forceFullCollection: false));
                    if (p == 20 || p % 100 == 99)
                    {
                        long before = Stopwatch.GetTimestamp();
                        long now = GC.GetTotalMemory(forceFullCollection: true);
                        collecting += Stopwatch.GetTimestamp() - before;               // the measuring's own collections are not the writer's time
                        live = Math.Max(live, now);
                        settled = p == 20 ? now : settled;
                    }
                }

                builder.Finish();
            }

            double milliseconds = Stopwatch.GetElapsedTime(start).TotalMilliseconds - Stopwatch.GetElapsedTime(0, collecting).TotalMilliseconds;
            long growth = GC.GetTotalMemory(forceFullCollection: true) - settled;
            return (milliseconds, counter.Length, heap, live, growth, GC.GetTotalAllocatedBytes(precise: true) - allocated);
        }

        foreach (var (name, image) in new[] { ("scans (Auto: JPEG)", scan), ("line art (Auto: Flate)", diagram) })
        {
            var cold = Run(image);
            var warm = Run(image);
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"    {Pages} pages of {name}, {width} x {height}: cold {cold.Milliseconds:F0} ms, warm {warm.Milliseconds:F0} ms ({warm.Milliseconds / Pages:F2} ms a page), {warm.Bytes / 1048576.0:F1} MiB written; live managed memory at most {warm.Live / 1048576.0:F1} MiB (a full collection every 100 pages), the heap with its garbage at most {warm.Heap / 1048576.0:F1} MiB (sampled each page), {warm.Growth / 1024.0:F0} KiB more after page 500 than after page 20, {warm.Allocated / Pages / 1024.0:F0} KiB allocated a page"));
            Check(cold.Bytes == warm.Bytes, "the same file twice");
            Check(warm.Growth < 4L << 20, $"{name}: the managed heap grew {warm.Growth:N0} bytes from page 20 to 500 (one page's buffers, not every page's)");
        }

        // The JPEG encoder on its own (what a JPEG page's time is mostly), warm.
        var encoder = ImageEncoders.Get("jpeg");
        var quality = new ImageEncodeOptions { Quality = PdfWriteDefaults.JpegQuality };
        encoder.Encode(scan, quality);
        long encodeAllocated = GC.GetTotalAllocatedBytes(precise: true), encodeStart = Stopwatch.GetTimestamp();
        for (int i = 0; i < 20; i++)
        {
            encoder.Encode(scan, quality);
        }

        double encodeMilliseconds = Stopwatch.GetElapsedTime(encodeStart).TotalMilliseconds / 20;
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    the jpeg encoder alone on the scan page: {encodeMilliseconds:F2} ms, {(GC.GetTotalAllocatedBytes(precise: true) - encodeAllocated) / 20 / 1024.0:F0} KiB allocated a call (warm, mean of 20)"));
    }

    // A stream that only counts what passes (the 500-page file is measured, not kept).
    private sealed class PdfByteCounter : Stream
    {
        private long _length;

        public override bool CanRead => false;

        public override bool CanSeek => false;

        public override bool CanWrite => true;

        public override long Length => _length;

        public override long Position
        {
            get => _length;
            set => throw new NotSupportedException();
        }

        public override void Write(byte[] buffer, int offset, int count) => _length += count;

        public override void Write(ReadOnlySpan<byte> buffer) => _length += buffer.Length;

        public override void Flush()
        {
        }

        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();

        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

        public override void SetLength(long value) => throw new NotSupportedException();
    }

    private static void PdfWriteSample(Device device)
    {
        CpuOnly(device);
        var face = SubsetFace();
        var (inputs, folder) = PdfWriteInputs();
        var scan = ImageCodecs.Get("png").Decode(File.ReadAllBytes(Path.Combine(folder, "inputs.scan-grey.png")));
        var photo = ImageCodecs.Get("png").Decode(File.ReadAllBytes(Path.Combine(folder, "inputs.photo-rgb.png")));
        string[] lines = [.. inputs["text"]!["lines"]!.AsArray().Select(l => (string)l!)];
        byte[] bytes = PdfWrite(b =>
        {
            b.Info = new DocumentInfo { Title = "Quarterly report", Author = "The office team", Subject = "A sample written by Idrak's PDF writer" };
            var page = b.AddPage(595, 842);
            PdfVectorPage(page);
            for (int i = 0; i < lines.Length; i++)
            {
                float size = i == 0 ? 20 : 12;
                page.Glyphs(face, size, PdfShaped(face, lines[i], "Latn", false, 60, 460 + 26 * i, size), lines[i], DocumentTextMode.Fill, DocumentColour.Grey(0.1f));
            }

            b.AddPage(photo.Width * 0.75f, photo.Height * 0.75f).Image(photo, new DocumentBox(0, 0, photo.Width * 0.75f, photo.Height * 0.75f), DocumentImageEncoding.Auto);
            float width = scan.Width * 0.72f, height = scan.Height * 0.72f;
            var scanned = b.AddPage(width, height);
            scanned.Image(scan, new DocumentBox(0, 0, width, height), DocumentImageEncoding.Auto);
            for (int i = 0; i < lines.Length; i++)
            {
                scanned.Glyphs(face, 11, PdfShaped(face, lines[i], "Latn", false, 60, 90 + 22 * i, 11), lines[i], DocumentTextMode.Hidden, DocumentColour.Black);
            }

            var style = new DocumentShapeStyle();
            var shapes = b.AddPage(450, 300);
            DocumentShapes.Write(shapes, new ShapePlan([new ShapePath([40, 40, 560, 40, 560, 360, 40, 360], closed: true), new ShapePath([40, 120, 560, 120], dash: [3, 2], tone: 0.5f)],
                [new ShapeArea([40, 40, 560, 40, 560, 120, 40, 120], 0.12f)]), style);
            DocumentShapes.Write(shapes, TextLayout.Layout("Project dashboard\nNext release: 2.3", new TextStyle(face, 28)), "Project dashboard\nNext release: 2.3", 60, 60, style);
        });

        using var document = PdfReadWritten(bytes);
        Check(document.PageCount == 4 && document.ReadPage(2, DocumentParts.Text).Text == string.Join("\n", lines), "the sample reads back");
        PdfSaveWritten("sample", bytes);
        Console.WriteLine($"    written: {Path.Combine(PdfWrittenFolder(), "sample.pdf")} ({bytes.Length:N0} bytes)");
    }
}
