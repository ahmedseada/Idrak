// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Idrak.Abstraction.Data;
using Idrak.Imaging;
using Idrak.Text;

// Plan 28's follow-up: idrak data synth writes PDFs. --format pdf: each drawn page (its look and all) an image page with
// the label's text hidden on the lines as drawn; --format pdf-vector: each page as drawn before any look, its text and
// shapes as vectors (the emblems' and stamps' words as text too). Every record names its document and the page's index in
// it, and every file is read back with the library's reader and drawn by its drawer.
internal abstract partial class DesktopTests
{
    private static readonly (string Name, Action<Device> Run)[] SyntheticPdfGroup =
    [
        ("synthetic data pdf: --format pdf on letter, plain and shapes pages and two seeds: a page each its pixels at the dpi in points, its image stored as Auto chose and decoding to the page's pixels (exact, or the JPEG codec's round trip), the label's text hidden over the lines as drawn (its glyphs read back in order where they were written, each run inside the box it was drawn in within a pixel, a page cut to its parts placed by the cut, the text where the ink is on a page with no look), the text the label's; every record's document and page open with the library's reader", SyntheticPdfScanned),
        ("synthetic data pdf: --format pdf-vector on letter, plain and shapes pages with emblems and stamps: no look; every glyph written (the label's and the ornaments' turned onto their paths) read back in order with its text, its fonts embedded as subsets; the label's text read back in reading order, the emblems' and stamps' words beside it; the page drawn by the library's drawer close to the page drawn before its look", SyntheticPdfVector),
        ("synthetic data pdf: several pages a document (pages per document): files named after their first page, pages in order, each record's page index; the same seed writes the same bytes twice, on one worker or many and into a zip; another seed other bytes", SyntheticPdfFiles),
        ("synthetic data pdf: memory does not grow with the page count: three hundred pages into one document, sampled per page after a full collection, scan-like and vector, with the time and allocation a page, cold and warm", SyntheticPdfMemory),
        ("synthetic data pdf: image output is unchanged: jpeg and png runs write the same images, records and manifests with the pages' vectors kept as without, their records name no pages", SyntheticPdfImagesUnchanged),
        ("synthetic data pdf: idrak data synth --format pdf and pdf-vector write documents with --writer, --pdf-pages, --pdf-images, --pdf-fonts, --dpi and --quality; a bad value of each, an unregistered writer, document options without --format and --pdf-images on vector pages are usage errors naming the option", SyntheticPdfCli),
    ];

    private const string PdfSynthParts = """
        {
          "headerStart": [["Northwind Trading Company", "Operations Department"], ["Bluebird Office Supplies", "Customer Service"]],
          "headerEnd": [["Ref. {digits:5}", "Page 1 of 1"]],
          "subjects": ["Subject: delivery of order {digits:5}"],
          "signatures": [["Alex Morgan", "Operations Manager"]],
          "footers": ["12 Harbour Road, Riverside"],
          "chances": {"headerStart": 1, "headerEnd": 1, "subject": 0.7, "signature": 1, "footer": 1}
        }
        """;

    // Generic texts: an ordinary letter, a technical report and a newsletter, documents between lines of ---.
    private static string PdfSynthInput(string folder)
    {
        string path = Path.Combine(folder, "letters.txt");
        File.WriteAllText(path, string.Join("\n---\n",
            "Dear Ms. Rivera,\n\nThank you for your order of 24 laptops and 12 monitors. The shipment left our warehouse on 14 March and should arrive within five working days.\n\nKind regards,\nDaniel Brooks",
            "Quarterly report: network upgrade\n\nThe office network moved to the new switches on 2 June. Average page load time fell from 1.8 seconds to 0.6 seconds.\n\nNext steps: replace the two remaining access points.",
            "Community newsletter\n\nThe garden club meets every Saturday at ten in the library hall. New members are welcome.\n\nThe book swap moves to the first Sunday of each month."), new UTF8Encoding(false));
        return path;
    }

    // Small pages (620 pixels at 75 to the inch: A4's width in points), the library's emblems and stamps on every letter,
    // the signature drawn by the ideal pen and the stamps in even ink (what a vector page writes), one real font.
    private static SyntheticPageOptions PdfSynthOptions(string folder, SyntheticPageOutput output, int seed = 4, int variants = 2, int workers = 2) => new()
    {
        Style = SyntheticPages.Style(new PageStyle { Fonts = [SubsetFace()] }, "width=620,height=877,size=14..18,emblem=1,stamp=1,signature-pen=ideal,stamp-ink=0"),
        Parts = SyntheticPageParts.FromJson(JsonNode.Parse(PdfSynthParts)!.AsObject(), folder),
        Variants = variants, Workers = workers, Seed = seed, Scans = ["office", "none"], Layouts = ["letter", "plain", "shapes"], Shapes = ["grid", "ruled", "framed"],
        Output = output, Dpi = 75,
    };

    // Each record of a written data set: its record and variant (from its manifest line), its document and page.
    private static List<(int Record, int Variant, string File, int Page, JsonNode Line, JsonNode About)> PdfSynthRecords(SyntheticPagesResult result)
    {
        var lines = File.ReadAllLines(result.Data).Select(l => JsonNode.Parse(l)!).ToList();
        var about = File.ReadAllLines(result.Manifest).Select(l => JsonNode.Parse(l)!).ToList();
        Check(lines.Count == about.Count && lines.Count == result.Pages, $"a record and a manifest line a page: {lines.Count}, {about.Count}, {result.Pages}");
        string folder = Path.GetDirectoryName(result.Data)!;
        return [.. lines.Select((l, i) => ((int)about[i]["record"]!, (int)about[i]["variant"]!, Path.Combine(folder, (string)l["images"]![0]!), (int)l["pages"]![0]!, l, about[i]))];
    }

    // The answer a plain text record's page was made from (the document's text, as the record reader gives it).
    private static string PdfSynthAnswer(string input, int record) => File.ReadAllText(input).Replace("\r", "", StringComparison.Ordinal).Split("\n---\n")[record].Trim();

    private static readonly DocumentOptions PdfSynthContent = new() { Layout = DocumentTextLayouts.Content };

    private static void SyntheticPdfScanned(Device device)
    {
        _ = device;
        string folder = SyntheticFolder("pdf-scan");
        try
        {
            string input = PdfSynthInput(folder);
            var problems = new List<string>();
            int pages = 0, cut = 0, jpeg = 0, exact = 0;
            foreach (int seed in (int[])[4, 11])
            {
                var options = PdfSynthOptions(folder, SyntheticPageOutput.ScannedDocuments, seed) with
                {
                    Style = PdfSynthOptions(folder, SyntheticPageOutput.ScannedDocuments).Style with { Crop = 0.4f },
                };
                var result = SyntheticPages.Write(input, Path.Combine(folder, $"s{seed}", "pages.jsonl"), options);
                foreach (var (record, variant, file, index, line, about) in PdfSynthRecords(result))
                {
                    pages++;
                    string where = $"seed {seed} record {record}-{variant} ({about["layout"]}, {about["profile"]})";
                    var made = SyntheticPages.Make(PdfSynthAnswer(input, record), options, record, variant);
                    var image = SyntheticPages.EightBit(made.Image);
                    cut += made.Drawn.Width != made.Vectors!.Width || made.Drawn.Height != made.Vectors.Height ? 1 : 0;
                    using var document = DocumentFormats.Open(file, PdfSynthContent);
                    var page = document.ReadPage(index, DocumentParts.All);
                    float scale = 72f / options.Dpi;
                    if (MathF.Abs(page.Width - image.Width * scale) > 1e-3f || MathF.Abs(page.Height - image.Height * scale) > 1e-3f)
                    {
                        problems.Add($"{where}: the page is {page.Width} x {page.Height} points for {image.Width} x {image.Height} pixels");
                    }

                    // The image: exact (Flate), or the JPEG codec's round trip of the page's pixels, compared as 8-bit levels
                    // (a drawn page's table of a byte's value and a codec's differ in the last bit: k / 255, k x (1 / 255)).
                    var stored = page.Images.Single();
                    jpeg += stored.Encoding == DocumentImage.Jpeg ? 1 : 0;
                    exact += stored.Encoding == DocumentImage.Jpeg ? 0 : 1;
                    var expected = stored.Encoding == DocumentImage.Jpeg ? ImageCodecs.Get("jpeg").Decode(JpegEncoder.Encode(image, 90)).Pixels : image.Pixels;
                    if (!PdfSynthLevels(stored.Decode().Pixels).AsSpan().SequenceEqual(PdfSynthLevels(expected)))
                    {
                        problems.Add($"{where}: the {stored.Encoding} image is not the page's pixels");
                    }

                    // The hidden text: the glyphs written over the lines (as the page wrote them), read back in order.
                    var capture = new PdfSynthCapture();
                    made.Vectors.WriteText(capture, scale, made.Drawn.X, made.Drawn.Y, DocumentTextMode.Hidden);
                    var hidden = page.Runs.Where(r => r.Hidden).ToList();
                    problems.AddRange(PdfSynthSameGlyphs(where, capture, hidden, 72f / options.Dpi));
                    string label = Regex.Replace(made.Text, @"\s+", "");
                    string read = Regex.Replace(string.Concat(hidden.Select(r => r.Text)), @"\s+", "");
                    if (read != label || made.Text != (string)line["conversations"]![1]!["value"]!)
                    {
                        problems.Add($"{where}: the hidden text reads '{read}', the label '{label}'");
                    }

                    // On a page with no look, the text stands where the ink is: each run's box holds the darkest pixels of its middle.
                    if ((string)about["profile"]! == "none")
                    {
                        foreach (var run in hidden.Where(r => r.Text.Trim().Length > 0))
                        {
                            if (PdfSynthInk(image, run.Box, scale) is { } paper)
                            {
                                problems.Add($"{where}: no ink under '{run.Text}' at {run.Box} ({paper:F2})");
                                break;
                            }
                        }
                    }
                }
            }

            Check(problems.Count == 0 && pages == 12 && cut > 0 && jpeg > 0, $"{pages} pages ({cut} cut, {jpeg} JPEG, {exact} exact):\n  " + string.Join("\n  ", problems.Take(20)));
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void SyntheticPdfVector(Device device)
    {
        _ = device;
        string folder = SyntheticFolder("pdf-vector");
        try
        {
            string input = PdfSynthInput(folder);
            var problems = new List<string>();
            var scores = new List<string>();
            int pages = 0, ornaments = 0;
            foreach (int seed in (int[])[4, 11])
            {
                var options = PdfSynthOptions(folder, SyntheticPageOutput.VectorDocuments, seed);
                var result = SyntheticPages.Write(input, Path.Combine(folder, $"v{seed}", "pages.jsonl"), options);
                foreach (var (record, variant, file, index, _, about) in PdfSynthRecords(result))
                {
                    pages++;
                    string where = $"seed {seed} record {record}-{variant} ({about["layout"]})";
                    if ((string)about["scan"]! != "" || (string)about["profile"]! != "none")
                    {
                        problems.Add($"{where}: a look on a vector page ({about["profile"]}: {about["scan"]})");
                    }

                    var made = SyntheticPages.Make(PdfSynthAnswer(input, record), options, record, variant);
                    using var document = DocumentFormats.Open(file, PdfSynthContent);
                    var page = document.ReadPage(index, DocumentParts.All);
                    float scale = 72f / options.Dpi;

                    // Every glyph written (the label's and the ornaments'), read back in order with its text.
                    var all = new PdfSynthCapture();
                    made.Vectors!.Write(all, scale, made.Drawn.X, made.Drawn.Y);
                    problems.AddRange(PdfSynthSameGlyphs(where, all, [.. page.Runs], 72f / options.Dpi, boxes: false));
                    string written = string.Concat(all.Runs.Select(r => r.Text)), read = string.Concat(page.Runs.Select(r => r.Text));
                    if (read != written)
                    {
                        problems.Add($"{where}: read '{read}', written '{written}'");
                    }

                    // The label's runs read back in reading order; the ornaments' words beside them.
                    var labelRuns = new PdfSynthCapture();
                    made.Vectors.WriteText(labelRuns, scale, made.Drawn.X, made.Drawn.Y, DocumentTextMode.Fill);
                    string label = Regex.Replace(string.Concat(labelRuns.Runs.Select(r => r.Text)), @"\s+", "");
                    if (label != Regex.Replace(made.Text, @"\s+", ""))
                    {
                        problems.Add($"{where}: the label's runs say '{label}', the label '{made.Text}'");
                    }

                    int turned = all.Runs.Count(r => r.Matrices is not null);
                    ornaments += turned;
                    if ((string)about["layout"]! == "letter" && ((string)about["page"]!).Contains("stamp stamp-", StringComparison.Ordinal) && turned == 0)
                    {
                        problems.Add($"{where}: a stamp with no words written");
                    }

                    if (page.Runs.Any(r => r.Font is not { } font || !Regex.IsMatch(font, "^[A-Z]{6}\\+")))
                    {
                        problems.Add($"{where}: a font not embedded as a subset: {string.Join(", ", page.Runs.Select(r => r.Font).Distinct())}");
                    }

                    // The page drawn by the library's drawer against the page drawn before its look.
                    var back = DocumentPageDrawers.DrawPage(document, index, new DocumentDrawOptions { Dpi = options.Dpi }).Image;
                    var (share, mean) = PdfDrawDifference(made.Image, back, 32);
                    scores.Add(string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{about["layout"]} {share:P2}"));
                    if (back.Width != made.Image.Width || back.Height != made.Image.Height || share > 0.02)
                    {
                        problems.Add($"{where}: drawn back {back.Width} x {back.Height} for {made.Image.Width} x {made.Image.Height}, {share:P2} of the pixels off by more than 32 levels (mean {mean:F2})");
                    }
                }
            }

            Console.WriteLine($"    pdf-vector pages drawn back (the share off by more than 32 levels): {string.Join(", ", scores)}");
            Check(problems.Count == 0 && pages == 12 && ornaments > 0, $"{pages} pages, {ornaments} turned glyph runs:\n  " + string.Join("\n  ", problems.Take(20)));
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void SyntheticPdfFiles(Device device)
    {
        _ = device;
        string folder = SyntheticFolder("pdf-files");
        try
        {
            string input = PdfSynthInput(folder);
            SyntheticPagesResult Run(string name, SyntheticPageOutput output, int seed, int workers, bool zip = false) =>
                SyntheticPages.Write(input, Path.Combine(folder, name, "pages.jsonl"), PdfSynthOptions(folder, output, seed, variants: 3, workers: workers) with { PagesPerDocument = 4, Zip = zip });

            foreach (var output in (SyntheticPageOutput[])[SyntheticPageOutput.ScannedDocuments, SyntheticPageOutput.VectorDocuments])
            {
                string kind = output == SyntheticPageOutput.ScannedDocuments ? "pdf" : "pdf-vector";
                var one = Run($"{kind}-one", output, 5, 1);
                var records = PdfSynthRecords(one);
                Check(records.Count == 9 && records.Select(r => Path.GetFileName(r.File)).SequenceEqual(["000000-00.pdf", "000000-00.pdf", "000000-00.pdf", "000000-00.pdf",
                          "000001-01.pdf", "000001-01.pdf", "000001-01.pdf", "000001-01.pdf", "000002-02.pdf"])
                      && records.Select(r => r.Page).SequenceEqual([0, 1, 2, 3, 0, 1, 2, 3, 0]) && records.All(r => (int)r.About["page_index"]! == r.Page),
                    $"{kind}: four pages a file, each named after its first page: {string.Join(", ", records.Select(r => $"{Path.GetFileName(r.File)}#{r.Page}"))}");
                foreach (var group in records.GroupBy(r => r.File))
                {
                    using var document = DocumentFormats.Open(group.Key);
                    Check(document.PageCount == group.Count(), $"{kind}: {Path.GetFileName(group.Key)} holds {document.PageCount} pages, {group.Count()} named");
                }

                var many = Run($"{kind}-many", output, 5, 4);
                var again = Run($"{kind}-again", output, 5, 3);
                var other = Run($"{kind}-other", output, 6, 2);
                string Hash(SyntheticPagesResult r) => string.Join(" ", Directory.GetFiles(r.Images).Order(StringComparer.Ordinal)
                    .Select(f => $"{Path.GetFileName(f)}:{Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(f)))}")
                    .Append(Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(r.Data)))).Append(Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(r.Manifest)))));
                Check(Hash(one) == Hash(many) && Hash(one) == Hash(again), $"{kind}: the same seed, the same bytes on one worker, four and three");
                Check(Hash(one) != Hash(other), $"{kind}: another seed, other bytes");

                var zipped = Run($"{kind}-zip", output, 5, 2, zip: true);
                using var archive = ZipFile.OpenRead(zipped.Images);
                Check(archive.Entries.Select(e => e.FullName).SequenceEqual(["pages/000000-00.pdf", "pages/000001-01.pdf", "pages/000002-02.pdf"]),
                    $"{kind}: the documents in the zip: {string.Join(", ", archive.Entries.Select(e => e.FullName))}");
                foreach (var entry in archive.Entries)
                {
                    using var stream = entry.Open();
                    using var copy = new MemoryStream();
                    stream.CopyTo(copy);
                    Check(copy.ToArray().AsSpan().SequenceEqual(File.ReadAllBytes(Path.Combine(folder, $"{kind}-one", entry.FullName))), $"{kind}: {entry.FullName} in the zip as in the folder");
                }
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void SyntheticPdfMemory(Device device)
    {
        _ = device;
        string folder = SyntheticFolder("pdf-memory");
        try
        {
            string input = PdfSynthInput(folder);
            foreach (var output in (SyntheticPageOutput[])[SyntheticPageOutput.ScannedDocuments, SyntheticPageOutput.VectorDocuments])
            {
                var options = PdfSynthOptions(folder, output, seed: 2) with { PagesPerDocument = 300 };
                var made = new List<SyntheticPage>();
                for (int i = 0; i < 9; i++)
                {
                    var page = SyntheticPages.Make(PdfSynthAnswer(input, i % 3), options, i % 3, i / 3);
                    made.Add(page with { Image = SyntheticPages.EightBit(page.Image) });
                }

                const int count = 300;
                var samples = new long[count];
                long allocated = 0, warmTicks = 0, coldTicks = 0;
                using (var documents = new SyntheticDocuments(options, folder, zip: null))
                {
                    for (int i = 0; i < count; i++)
                    {
                        long bytes = GC.GetTotalAllocatedBytes(precise: true), start = Stopwatch.GetTimestamp();
                        documents.Add(made[i % made.Count], $"memory-{output}.pdf");
                        long ticks = Stopwatch.GetTimestamp() - start;
                        if (i < made.Count)
                        {
                            coldTicks += ticks;
                        }
                        else
                        {
                            warmTicks += ticks;
                            allocated += GC.GetTotalAllocatedBytes(precise: true) - bytes;
                        }

                        samples[i] = GC.GetTotalMemory(forceFullCollection: true);
                    }
                }

                string file = Path.Combine(folder, $"memory-{output}.pdf");
                using (var document = DocumentFormats.Open(file))
                {
                    Check(document.PageCount == count, $"{output}: {document.PageCount} pages written");
                }

                // After the first pages (the fonts and the writer's buffers made), the memory holds steady.
                long early = samples.Skip(made.Count).Take(100).Max(), late = samples.Skip(count - 100).Max();
                int warm = count - made.Count;
                Console.WriteLine(string.Create(System.Globalization.CultureInfo.InvariantCulture,
                    $"    {output}: {new FileInfo(file).Length / 1024.0 / count:F1} KiB a page; cold {Stopwatch.GetElapsedTime(0, coldTicks).TotalMilliseconds / made.Count:F1} ms a page (the first {made.Count}), warm {Stopwatch.GetElapsedTime(0, warmTicks).TotalMilliseconds / warm:F2} ms and {allocated / 1024.0 / warm:F0} KiB allocated a page; managed memory after a full collection: page {made.Count} {samples[made.Count] / 1048576.0:F1} MiB, the most of pages {made.Count}-{made.Count + 99} {early / 1048576.0:F1} MiB, of the last 100 {late / 1048576.0:F1} MiB"));
                Check(late - early < 2 << 20, $"{output}: the memory grew from {early / 1048576.0:F1} MiB to {late / 1048576.0:F1} MiB over {count} pages");
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void SyntheticPdfImagesUnchanged(Device device)
    {
        _ = device;
        string folder = SyntheticFolder("pdf-images");
        try
        {
            string input = PdfSynthInput(folder);
            foreach (string format in (string[])["jpeg", "png"])
            {
                var options = PdfSynthOptions(folder, SyntheticPageOutput.Images, seed: 8) with { ImageFormat = format, Style = PdfSynthOptions(folder, SyntheticPageOutput.Images).Style with { Crop = 0.3f } };
                var plain = SyntheticPages.Write(input, Path.Combine(folder, $"{format}-plain", "pages.jsonl"), options);
                var kept = SyntheticPages.Write(input, Path.Combine(folder, $"{format}-kept", "pages.jsonl"), options with { Style = options.Style with { Vectors = true } });
                var names = Directory.GetFiles(plain.Images).Select(Path.GetFileName).Order(StringComparer.Ordinal).ToList();
                Check(names.Count == 6 && names.All(n => File.ReadAllBytes(Path.Combine(plain.Images, n!)).AsSpan().SequenceEqual(File.ReadAllBytes(Path.Combine(kept.Images, n!)))),
                    $"{format}: the same images with the vectors kept");
                Check(File.ReadAllText(plain.Data) == File.ReadAllText(kept.Data) && File.ReadAllText(plain.Manifest) == File.ReadAllText(kept.Manifest), $"{format}: the same records and manifest");
                Check(File.ReadAllLines(plain.Data).All(l => JsonNode.Parse(l)!["pages"] is null) && File.ReadAllLines(plain.Manifest).All(l => JsonNode.Parse(l)!["page_index"] is null),
                    $"{format}: an image's record names no page");
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void SyntheticPdfCli(Device device)
    {
        _ = device;
        string folder = SyntheticFolder("pdf-cli");
        try
        {
            string input = PdfSynthInput(folder);
            string fonts = Path.Combine(folder, "fonts");
            Directory.CreateDirectory(fonts);
            File.Copy(FontData("dejavu-sans-subset.ttf"), Path.Combine(fonts, "dejavu.ttf"));
            string[] common = ["--fonts", fonts, "--page", "width=620,height=877,size=14..18", "--seed", "3", "-n", "2", "-j"];
            var (code, output, error) = SynthCli(["data", "synth", input, "-o", Path.Combine(folder, "scan", "pages.jsonl"), "--format", "pdf", "--dpi", "75", "--pdf-images", "flate",
                "--quality", "80", .. common]);
            Check(code == 0, $"--format pdf: exit {code}: {error}");
            var json = JsonNode.Parse(output)!;
            var records = File.ReadAllLines((string)json["data"]!).Select(l => JsonNode.Parse(l)!).ToList();
            Check((string)json["format"]! == "pdf" && (int)json["pages"]! == 6 && records.Count == 6 && Directory.GetFiles((string)json["images"]!, "*.pdf").Length == 6
                  && records.All(r => ((string)r["images"]![0]!).EndsWith(".pdf", StringComparison.Ordinal) && (int)r["pages"]![0]! == 0),
                $"six scan-like PDFs: {output}");
            using (var document = DocumentFormats.Open(Path.Combine(folder, "scan", (string)records[0]["images"]![0]!)))
            {
                var page = document.ReadPage(0, DocumentParts.All);
                Check(MathF.Abs(page.Width - 620 * 72f / 75) < 1e-3f && page.Images.Single().Encoding != DocumentImage.Jpeg && page.Runs.Count > 0 && page.Runs.All(r => r.Hidden),
                    $"--dpi 75 and --pdf-images flate: {page.Width} points, {page.Images.Single().Encoding}");
            }

            (code, output, error) = SynthCli(["data", "synth", input, "-o", Path.Combine(folder, "vector", "pages.jsonl"), "--format", "pdf-vector", "--writer", "pdf",
                "--pdf-pages", "4", "--pdf-fonts", "whole", .. common]);
            Check(code == 0, $"--format pdf-vector: exit {code}: {error}");
            json = JsonNode.Parse(output)!;
            records = [.. File.ReadAllLines((string)json["data"]!).Select(l => JsonNode.Parse(l)!)];
            Check((string)json["format"]! == "pdf-vector" && Directory.GetFiles((string)json["images"]!, "*.pdf").Length == 2
                  && records.Select(r => (int)r["pages"]![0]!).SequenceEqual([0, 1, 2, 3, 0, 1]), $"four pages a PDF: {output}");
            using (var document = DocumentFormats.Open(Path.Combine(folder, "vector", (string)records[0]["images"]![0]!)))
            {
                var fontsUsed = document.ReadPage(0, DocumentParts.All).Runs.Select(r => r.Font).Distinct().ToList();
                Check(document.PageCount == 4 && fontsUsed.Count > 0 && fontsUsed.All(f => f is { } name && !Regex.IsMatch(name, "^[A-Z]{6}\\+")),
                    $"--pdf-fonts whole embeds the whole font (no subset tag): {string.Join(", ", fontsUsed)}");
            }

            foreach (var (args, names) in new (string[], string)[]
            {
                (["--format", "pdf", "--pdf-pages", "0"], "--pdf-pages"),
                (["--format", "pdf", "--dpi", "0"], "--dpi"),
                (["--format", "pdf", "--dpi", "many"], "--dpi"),
                (["--format", "pdf", "--pdf-images", "png"], "--pdf-images"),
                (["--format", "pdf", "--pdf-fonts", "all"], "--pdf-fonts"),
                (["--format", "pdf", "--quality", "0"], "--quality"),
                (["--format", "pdf", "--writer", "nope"], "nope"),
                (["--pdf-pages", "2"], "--pdf-pages"),
                (["--dpi", "300", "--writer", "pdf"], "--writer"),
                (["--format", "pdf-vector", "--pdf-images", "jpeg"], "--pdf-images"),
            })
            {
                var bad = SynthCli(["data", "synth", input, "-o", Path.Combine(folder, "bad", "pages.jsonl"), .. args, .. common]);
                Check(bad.Code == 2 && bad.Err.Contains(names, StringComparison.Ordinal), $"{string.Join(" ", args)}: exit {bad.Code}, {bad.Err.Trim()}");
            }

            Check(!File.Exists(Path.Combine(folder, "bad", "pages.jsonl")), "a refused run writes nothing");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // The glyphs read back (each read run's in order) against those written (each captured run's in order): the same ids,
    // in the same places; with `boxes`, each read run inside the box its written run was drawn in (along its baseline its
    // glyphs' advances, across it the face's descent to ascent) within `pixel` points.
    private static IEnumerable<string> PdfSynthSameGlyphs(string where, PdfSynthCapture written, IReadOnlyList<DocumentTextRun> read, float pixel, bool boxes = true)
    {
        var expected = written.Runs.SelectMany((r, i) => r.Glyphs.Select(g => (Run: i, Glyph: g))).ToList();
        var got = read.SelectMany(r => r.Glyphs.ToArray().Select(g => (Run: r, Glyph: g))).ToList();
        if (got.Count != expected.Count)
        {
            yield return $"{where}: {got.Count} glyphs read, {expected.Count} written";
            yield break;
        }

        for (int i = 0; i < got.Count; i++)
        {
            var (e, g) = (expected[i].Glyph, got[i].Glyph);
            if (e.Glyph != g.Glyph || MathF.Abs(e.X - g.X) > 2e-3f || MathF.Abs(e.Y - g.Y) > 2e-3f)
            {
                yield return $"{where}: glyph {i}: {g.Glyph} at ({g.X}, {g.Y}), written {e.Glyph} at ({e.X}, {e.Y})";
                yield break;
            }
        }

        if (!boxes)
        {
            yield break;
        }

        int at = 0;
        foreach (var run in read)
        {
            int count = run.Glyphs.Length;
            if (count == 0)
            {
                continue;
            }

            var source = written.Runs[expected[at].Run];
            at += count;
            float ascent = source.Face.Metrics.Ascender * source.Size / source.Face.Metrics.UnitsPerEm, descent = source.Face.Metrics.Descender * source.Size / source.Face.Metrics.UnitsPerEm;
            float x0 = source.Glyphs.Min(g => g.X), x1 = source.Glyphs.Max(g => g.X + g.Advance), y0 = source.Glyphs[0].Y - ascent, y1 = source.Glyphs[0].Y - descent;
            var box = run.Box;
            if (box.X < x0 - pixel || box.Right > x1 + pixel || box.Y < y0 - pixel || box.Bottom > y1 + pixel)
            {
                yield return $"{where}: '{run.Text}' read in {box}, drawn in ({x0}, {y0}) to ({x1}, {y1})";
                yield break;
            }
        }
    }

    // Values in [0, 1] as the 8-bit levels they stand for (rounded, as the encoders round them).
    private static byte[] PdfSynthLevels(float[] values) => [.. values.Select(v => (byte)Math.Clamp((int)MathF.Round(v * 255f), 0, 255))];

    // Null when the darkest pixel under the middle of `box` (points; the page's pixels at `scale` points a pixel) is ink,
    // else that darkest value.
    private static float? PdfSynthInk(ImageData image, DocumentBox box, float scale)
    {
        int x0 = Math.Clamp((int)(box.X / scale), 0, image.Width - 1), x1 = Math.Clamp((int)MathF.Ceiling(box.Right / scale), x0 + 1, image.Width);
        int y0 = Math.Clamp((int)((box.Y + box.Height * 0.25f) / scale), 0, image.Height - 1), y1 = Math.Clamp((int)MathF.Ceiling((box.Bottom - box.Height * 0.25f) / scale), y0 + 1, image.Height);
        var pixels = image.Pixels;
        float darkest = 1;
        for (int y = y0; y < y1; y++)
        {
            for (int x = x0; x < x1; x++)
            {
                darkest = MathF.Min(darkest, pixels[y * image.Width + x]);
            }
        }

        return darkest < 0.5f ? null : darkest;
    }

    // A page builder that keeps the text runs it is given (their faces, sizes, glyphs, matrices and text) and counts the rest.
    private sealed class PdfSynthCapture : IDocumentPageBuilder
    {
        public List<(IFontFace Face, float Size, DocumentGlyph[] Glyphs, DocumentMatrix[]? Matrices, string Text, DocumentTextMode Mode)> Runs { get; } = [];

        public int Paths { get; private set; }

        public int Index => 0;

        public float Width => 1000;

        public float Height => 1000;

        public void Image(ImageData image, DocumentBox box, DocumentImageEncoding encoding)
        {
        }

        public void Path(DocumentPath path) => Paths++;

        public void Glyphs(IFontFace face, float size, ReadOnlySpan<DocumentGlyph> glyphs, string text, DocumentTextMode mode, DocumentColour colour) =>
            Runs.Add((face, size, glyphs.ToArray(), null, text, mode));

        public void Glyphs(IFontFace face, ReadOnlySpan<DocumentGlyph> glyphs, ReadOnlySpan<DocumentMatrix> matrices, string text, DocumentTextMode mode, DocumentColour colour) =>
            Runs.Add((face, MathF.Sqrt(matrices[0].A * matrices[0].A + matrices[0].B * matrices[0].B), glyphs.ToArray(), matrices.ToArray(), text, mode));

        public void Save()
        {
        }

        public void Restore()
        {
        }

        public void Clip(DocumentPath path)
        {
        }

        public void AddAnnotation(DocumentAnnotation annotation)
        {
        }

        public void AddLink(DocumentLink link)
        {
        }
    }
}
