// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Data.Pdf;
using Idrak.Imaging;

// Plan 28, phase D11: editing in place and saving (incremental updates, a clean rewrite, page operations, merging and
// splitting, content over a page, saving encrypted), checked on the D11 samples tools/documents/make_fixtures.py writes
// into the ignored test-output/documents/samples (and D12's signed files, D2's encrypted ones, every readable sample for
// the rewrite). Every file written goes to test-output/documents/edited (NAME.pdf, and NAME.json with the text the
// reference readers must extract) or, encrypted with the user password reader-2026, to edited-encrypted, for
// tools/documents/check_written.py.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] DocumentPdfEditGroup =
    [
        ("documents pdf: edit: the D11 incremental samples (classic tables, cross-reference streams over object streams): each revision ends with %%EOF where its JSON says and its prefix opens, the sections are its revisions newest first, and the newest reads as its JSON's final (pages, rotations, title, text, the field's value, each page's annotations, the bookmark)", PdfEditSamples),
        ("documents pdf: edit: an incremental update (the information, a page turned, a stamp over a page; then pages added, moved and removed) keeps the source's bytes as its prefix, appends one section of the source's kind (a table after a table, a stream after a stream) whose /Prev is the source's last, reads as edited, reads the same as a rewrite of the same edits, and the same edits write the same bytes", PdfEditIncremental),
        ("documents pdf: edit: merge-a and merge-b merged by CopyPage and by InsertPages (incremental and rewrite) give merge-plan's pages, each with the font, image and graphics state its own /F1, /Im1 and /GS1 meant; the stamp over merge-a's pages named apart where its names clash (the page's text, image and state kept); split-source split into its parts with their text and rotations; an image and a font merged from two copies of a file written once", PdfEditMerge),
        ("documents pdf: edit: page-operations' cases (insert, remove, move, rotate, combined) give the pages, text and rotations its JSON says, saved incrementally and as a rewrite", PdfEditOperations),
        ("documents pdf: edit: a rewrite of every readable sample (only what the root reaches, renumbered, object streams, streams deflated again where it pays, duplicates once) is no larger than its source and reads the same: information, pages, sizes, rotations, text, runs and images (their data) and issues; twice the same bytes", PdfEditRewrite),
        ("documents pdf: edit: encryption: AES-256 (revision 6) written by the builder (object streams, a classic table, none packed) and by a rewrite opens with the user and the owner password and not without, /O /U /OE /UE /P and /Perms as the specification makes them, no string or stream plain; encrypted-aes256 updated incrementally keeps its encryption and permissions; its encryption removed by a rewrite opened with the owner password, refused with the user's; every D2 sample rewritten and updated with its own encryption reads the same", PdfEditEncryption),
        ("documents pdf: edit: each D12 signed sample updated incrementally (the information, a stamp) keeps every byte its signatures' byte ranges cover, its signature dictionaries read the same, and the stamp reads", PdfEditSigned),
        ("documents pdf: edit: a field the document lacks refused naming it; annotations and bookmarks misused refused (an annotation the page does not hold, a page out of range, a bookmark that is null); misuse refused (signing without options, pages out of range or named twice, a rotation not a multiple of 90, a second Save, encryption changed by an update or by the user password, a builder encrypted after its first page, a document another reader opened); a document opened by its path edited through the registry's wrapper", PdfEditRefusals),
        ("documents pdf: edit: timings and memory: an incremental one-field change on a large file (bytes written, cold and warm), a rewrite of it, and two files merged page by page (cold and warm, allocated bytes, the process's peak)", PdfEditTimings),
    ];

    // The user and owner passwords every encrypted file written here uses (encrypted-aes256's own).
    private const string PdfEditUser = "reader-2026", PdfEditOwner = "editor-2026";

    private static PdfSampleEntry PdfEditSample(string name) =>
        PdfSamples().FirstOrDefault(s => s.Name == name) ?? throw new InvalidOperationException($"The sample '{name}' is not in the samples: make them again (tools/documents/make_fixtures.py).");

    private static PdfDocument PdfEditOpen(byte[] bytes, string name, string? password = null) =>
        PdfDocument.Open(new MemoryStream(bytes, writable: false), name, new DocumentOptions { Password = password });

    // An edit of `source` saved as `mode` to bytes.
    private static byte[] PdfEditSave(IDocument source, Action<IDocumentEditor> edits, DocumentSaveMode mode, DocumentWriteOptions? options = null)
    {
        using var destination = new MemoryStream();
        using (var editor = new PdfWriter().Edit(source, destination, options ?? PdfWriteDefaults))
        {
            edits(editor);
            editor.Save(mode);
        }

        return destination.ToArray();
    }

    // Each page as read: its text, rotation and size.
    private static List<(string Text, int Rotation, float Width, float Height)> PdfEditPagesRead(IDocument document)
    {
        var pages = new List<(string, int, float, float)>();
        for (int i = 0; i < document.PageCount; i++)
        {
            var page = document.ReadPage(i, DocumentParts.Text);
            pages.Add((page.Text, page.Rotation, page.Width, page.Height));
        }

        return pages;
    }

    private static string PdfEditDescribe(List<(string Text, int Rotation, float Width, float Height)> pages) =>
        string.Join(" | ", pages.Select(p => string.Create(CultureInfo.InvariantCulture, $"{p.Text.Replace('\n', '/')} ({p.Rotation}, {p.Width}x{p.Height})")));

    // Saves a file written here for the checker: NAME.pdf and, when its pages' text is given, NAME.json.
    private static void PdfSaveEdited(string name, byte[] bytes, IReadOnlyList<(string Text, int Rotation, float Width, float Height)>? pages = null, bool encrypted = false)
    {
        string folder = RepositoryPath("test-output", "documents", encrypted ? "edited-encrypted" : "edited");
        Directory.CreateDirectory(folder);
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

    // A stamp over a page: a translucent band, a word at its foot, and (when asked) a small image: what the editor's
    // content names as /F1, /GS1 and /Im1 when nothing clashes.
    private static void PdfEditStamp(IDocumentPageBuilder page, IFontFace face, string word, bool image = false)
    {
        float size = 10, x = 12, baseline = page.Height - 10;
        page.Path(new DocumentPath(PdfBoxVerbs, PdfBoxPoints(new DocumentBox(6, page.Height - 24, page.Width - 12, 18)))
        {
            Fill = DocumentFill.NonZero, FillColour = DocumentColour.Rgb(0.9f, 0.85f, 0.2f).WithAlpha(0.4f),
        });
        page.Glyphs(face, size, PdfShaped(face, word, "Latn", false, x, baseline, size), word, DocumentTextMode.Fill, DocumentColour.Rgb(0.6f, 0.1f, 0.1f));
        if (image)
        {
            page.Image(PdfLineArt(3, 24, 12), new DocumentBox(page.Width - 40, page.Height - 22, 24, 12), DocumentImageEncoding.Flate);
        }
    }

    // A page's resources as the file has them: the dictionary of one category (/Font, /XObject, /ExtGState).
    private static PdfDictionary PdfEditResources(PdfDocument document, int page, string category)
    {
        var file = document.File;
        var resources = file.Dictionary(PdfEditPages.Resources(document, page)) ?? PdfDictionary.Empty;
        return file.Dictionary(resources[Encoding.ASCII.GetBytes(category)]) ?? PdfDictionary.Empty;
    }

    private static void PdfEditSamples(Device device)
    {
        CpuOnly(device);
        foreach (string name in new[] { "edit-incremental-table", "edit-incremental-stream" })
        {
            var sample = PdfEditSample(name);
            byte[] bytes = File.ReadAllBytes(sample.Path);
            var revisions = sample.Expected["revisions"]!.AsArray();
            var ends = sample.Expected["structure"]!["revisionEnds"]!.AsArray().Select(e => (int)e!).ToArray();
            Check(revisions.Count == ends.Length && ends[^1] <= bytes.Length, $"{name}: {revisions.Count} revisions and {ends.Length} ends");
            for (int k = 0; k < ends.Length; k++)
            {
                Check(Encoding.ASCII.GetString(bytes, 0, ends[k]).TrimEnd().EndsWith("%%EOF", StringComparison.Ordinal), $"{name}: revision {k + 1} ends with %%EOF at byte {ends[k]}");
                using var prefix = PdfEditOpen(bytes[..ends[k]], $"{name} to revision {k + 1}");
                Check(prefix.PageCount > 0 && !prefix.Info.Repaired && prefix.File.Xref.Sections.Count == k + 1, $"{name}: revision {k + 1} opens alone ({prefix.File.Xref.Sections.Count} sections)");
            }

            using var document = PdfEditOpen(bytes, name);
            var sections = document.File.Xref.Sections;
            Check(sections.Count == revisions.Count, $"{name}: {sections.Count} sections, {revisions.Count} revisions");
            for (int k = 0; k < sections.Count; k++)
            {
                var revision = revisions[revisions.Count - 1 - k]!;
                Check(sections[k].Kind == (string)revision["kind"]! && sections[k].Offset == (long)revision["offset"]!,
                    $"{name}: section {k} is a {sections[k].Kind} at {sections[k].Offset}, the revision a {revision["kind"]} at {revision["offset"]}");
            }

            PdfEditCheckFinal(name, document, sample.Expected["final"]!.AsObject());
        }
    }

    // What the newest revision reads, as the sample's `final` says.
    private static void PdfEditCheckFinal(string what, PdfDocument document, JsonObject final)
    {
        var pages = PdfEditPagesRead(document);
        var texts = final["pageText"]!.AsArray().Select(t => (string)t!).ToList();
        var rotations = final["rotations"]!.AsArray().Select(r => (int)r!).ToList();
        Check(document.PageCount == (int)final["pageCount"]! && pages.Select(p => p.Text).SequenceEqual(texts) && pages.Select(p => p.Rotation).SequenceEqual(rotations),
            $"{what}: the pages read {PdfEditDescribe(pages)}");
        Check(document.Info.Title == (string?)final["title"], $"{what}: the title is {document.Info.Title}, expected {final["title"]}");
        var file = document.File;
        var catalog = file.Dictionary(file.Trailer["Root"u8])!;
        var fields = new Dictionary<string, string>();
        if (file.Dictionary(catalog["AcroForm"u8]) is { } form && file.Resolve(form["Fields"u8]).AsArray is { } list)
        {
            foreach (var field in list.Items)
            {
                var dictionary = file.Dictionary(field)!;
                fields[PdfText.Decode(file.Resolve(dictionary["T"u8]).AsString!.Bytes)] = file.Resolve(dictionary["V"u8]).AsString is { } value ? PdfText.Decode(value.Bytes) : "";
            }
        }

        var expectedFields = final["fields"]!.AsObject().ToDictionary(f => f.Key, f => (string)f.Value!);
        Check(fields.Count == expectedFields.Count && fields.All(f => expectedFields.TryGetValue(f.Key, out var v) && v == f.Value), $"{what}: the fields {string.Join(", ", fields)}");
        if (final["annotations"] is JsonObject annotations)
        {
            foreach (var (page, kinds) in annotations)
            {
                var node = document.PageObject(int.Parse(page, CultureInfo.InvariantCulture)).Node;
                var read = file.Resolve(file.Dictionary(node)!["Annots"u8]).AsArray?.Items.Select(a => file.Dictionary(a)!["Subtype"u8].AsName!.ToString()).ToList() ?? [];
                Check(read.SequenceEqual(kinds!.AsArray().Select(k => (string)k!)), $"{what}: page {page}'s annotations are {string.Join(", ", read)}");
            }
        }

        if (final["outline"] is JsonArray outline)
        {
            var titles = new List<string>();
            var item = file.Dictionary(file.Dictionary(catalog["Outlines"u8])?["First"u8] ?? PdfValue.Null);
            while (item is not null && titles.Count < 100)
            {
                titles.Add(PdfText.Decode(file.Resolve(item["Title"u8]).AsString!.Bytes));
                item = file.Dictionary(item["Next"u8]);
            }

            Check(titles.SequenceEqual(outline.Select(o => (string)o!)), $"{what}: the bookmarks {string.Join(", ", titles)}");
        }
    }

    private static void PdfEditIncremental(Device device)
    {
        CpuOnly(device);
        var face = SubsetFace();
        foreach (string name in new[] { "edit-incremental-table", "edit-incremental-stream", "merge-a", "split-source" })
        {
            var sample = PdfEditSample(name);
            byte[] bytes = File.ReadAllBytes(sample.Path);
            using var source = PdfEditOpen(bytes, name);
            var before = PdfEditPagesRead(source);
            var sections = source.File.Xref.Sections;
            string expectedKind = sections[0].Kind is "table" or "hybrid" ? "table" : "stream";
            int stamped = before[0].Rotation == 0 ? 0 : before.FindLastIndex(p => p.Rotation == 0);   // over an upright page: one line more at its foot

            // A light edit (the page tree untouched), then one that changes the pages.
            void Light(IDocumentEditor e)
            {
                e.SetInfo(new DocumentInfo { Title = "Reviewed copy", Author = "The office team", Producer = "Idrak" });
                e.RotatePage(1, (before[1].Rotation + 90) % 360);
                PdfEditStamp(e.Overlay(stamped), face, "Reviewed");
            }

            void Heavy(IDocumentEditor e)
            {
                Light(e);
                var added = e.AddPage(300, 200);
                added.Glyphs(face, 14, PdfShaped(face, "Appendix", "Latn", false, 30, 60, 14), "Appendix", DocumentTextMode.Fill, DocumentColour.Black);
                e.MovePage(e.PageCount - 1, 1);
                e.RemovePages(e.PageCount - 1);
            }

            var expectedLight = new List<(string Text, int Rotation, float Width, float Height)>(before);
            expectedLight[stamped] = (before[stamped].Text + "\nReviewed", before[stamped].Rotation, before[stamped].Width, before[stamped].Height);
            expectedLight[1] = (expectedLight[1].Text, (before[1].Rotation + 90) % 360, before[1].Height, before[1].Width);
            var expectedHeavy = new List<(string Text, int Rotation, float Width, float Height)>(expectedLight);
            expectedHeavy.Insert(1, ("Appendix", 0, 300, 200));
            expectedHeavy.RemoveAt(expectedHeavy.Count - 1);

            foreach (var (label, edits, expected) in new (string, Action<IDocumentEditor>, List<(string, int, float, float)>)[] { ("light", Light, expectedLight), ("heavy", Heavy, expectedHeavy) })
            {
                string what = $"{name}, {label} edit";
                byte[] update = PdfEditSave(source, edits, DocumentSaveMode.Incremental);
                Check(update.Length > bytes.Length && update.AsSpan(0, bytes.Length).SequenceEqual(bytes), $"{what}: the source's {bytes.Length} bytes are the update's prefix");
                Check(PdfEditSave(source, edits, DocumentSaveMode.Incremental).AsSpan().SequenceEqual(update), $"{what}: the same edits, the same bytes");
                using var updated = PdfEditOpen(update, what);
                var read = PdfEditPagesRead(updated);
                Check(!updated.Info.Repaired && updated.File.Xref.Sections.Count == sections.Count + 1, $"{what}: {updated.File.Xref.Sections.Count} sections (repaired: {updated.Info.Repaired})");
                string kind = updated.File.Xref.Sections[0].Kind is "objstm" ? "stream" : updated.File.Xref.Sections[0].Kind;
                Check(kind == expectedKind && updated.File.Xref.Sections[1].Offset == sections[0].Offset,
                    $"{what}: the update's section is a {updated.File.Xref.Sections[0].Kind} pointing back to {updated.File.Xref.Sections[1].Offset} (the source's last: a {sections[0].Kind} at {sections[0].Offset})");
                Check(read.Count == expected.Count && read.Zip(expected).All(p => p.First.Text == p.Second.Item1 && p.First.Rotation == p.Second.Item2
                          && MathF.Abs(p.First.Width - p.Second.Item3) < 1e-3f && MathF.Abs(p.First.Height - p.Second.Item4) < 1e-3f),
                    $"{what}: read {PdfEditDescribe(read)}; expected {PdfEditDescribe(expected)}");
                Check(updated.Info.Title == "Reviewed copy" && updated.Info.Author == "The office team", $"{what}: the information {updated.Info.Title}, {updated.Info.Author}");

                byte[] rewrite = PdfEditSave(source, edits, DocumentSaveMode.Rewrite);
                using var rewritten = PdfEditOpen(rewrite, what + " as a rewrite");
                var again = PdfEditPagesRead(rewritten);
                Check(again.SequenceEqual(read) && rewritten.Info.Title == updated.Info.Title && rewritten.File.Xref.Sections.Count == 1, $"{what}: the rewrite reads {PdfEditDescribe(again)}");
                PdfSaveEdited($"incremental-{name}-{label}", update, read);
                PdfSaveEdited($"rewrite-{name}-{label}", rewrite, again);
            }
        }

        // The setting: DocumentWriteOptions.SaveMode (IDRAK_DOCUMENT_SAVE_MODE) is what an app passes when it does not choose.
        var options = PdfWriteDefaults with { SaveMode = DocumentSaveMode.Rewrite };
        using var merged = PdfEditOpen(File.ReadAllBytes(PdfEditSample("merge-a").Path), "merge-a");
        byte[] bySetting = PdfEditSave(merged, e => e.SetInfo(new DocumentInfo { Title = "By the setting" }), options.SaveMode, options);
        using var read2 = PdfEditOpen(bySetting, "by the setting");
        Check(read2.File.Xref.Sections.Count == 1 && read2.Info.Title == "By the setting", "the options' save mode chooses a rewrite");
    }

    private static void PdfEditMerge(Device device)
    {
        CpuOnly(device);
        // merge-plan holds no PDF (the samples' loader takes PDFs): its JSON beside merge-a's.
        var plan = JsonNode.Parse(File.ReadAllText(Path.Combine(Path.GetDirectoryName(PdfEditSample("merge-a").Path)!, "merge-plan.json")))!.AsObject();
        byte[] a = File.ReadAllBytes(PdfEditSample("merge-a").Path), b = File.ReadAllBytes(PdfEditSample("merge-b").Path);
        using var mergeA = PdfEditOpen(a, "merge-a");
        using var mergeB = PdfEditOpen(b, "merge-b");
        var result = plan["merge"]!["result"]!.AsArray();

        // Each merged page with its source's objects: the text, /F1's font, /Im1's data, /GS1's opacity.
        void CheckMerged(string what, byte[] bytes)
        {
            using var merged = PdfEditOpen(bytes, what);
            Check(merged.PageCount == result.Count, $"{what}: {merged.PageCount} pages");
            for (int i = 0; i < result.Count; i++)
            {
                var expected = result[i]!;
                var from = (string)expected["from"]! == "merge-a" ? mergeA : mergeB;
                int index = (int)expected["page"]!;
                var page = merged.ReadPage(i, DocumentParts.Text | DocumentParts.Images);
                Check(page.Text == (string)expected["text"]!, $"{what}: page {i} reads \"{page.Text}\"");
                string font = merged.File.Dictionary(PdfEditResources(merged, i, "Font")["F1"u8])!["BaseFont"u8].AsName!.ToString();
                Check(font.Split('+')[^1] == (string)expected["font"]!, $"{what}: page {i}'s /F1 is {font}");
                var own = from.ReadPage(index, DocumentParts.Images).Images.Single();
                Check(page.Images.Count == 1 && page.Images[0].Bytes().Span.SequenceEqual(own.Bytes().Span), $"{what}: page {i}'s /Im1 is its source's image");
                var state = merged.File.Dictionary(PdfEditResources(merged, i, "ExtGState")["GS1"u8])!;
                var sourceState = from.File.Dictionary(PdfEditResources(from, index, "ExtGState")["GS1"u8])!;
                Check(state["ca"u8].TryNumber(out double ca) && sourceState["ca"u8].TryNumber(out double sourceCa) && ca == sourceCa, $"{what}: page {i}'s /GS1 is its source's");
            }

            PdfSaveEdited(what.Replace(' ', '-').Replace(",", ""), bytes, PdfEditPagesRead(merged));
        }

        using (var destination = new MemoryStream())
        {
            using (var builder = new PdfWriter().Create(destination, PdfWriteDefaults))
            {
                foreach (var page in result)
                {
                    builder.CopyPage((string)page!["from"]! == "merge-a" ? mergeA : mergeB, (int)page["page"]!);
                }

                builder.Finish();
            }

            CheckMerged("merged by CopyPage", destination.ToArray());
        }

        foreach (var mode in new[] { DocumentSaveMode.Incremental, DocumentSaveMode.Rewrite })
        {
            CheckMerged($"merged by InsertPages {mode}", PdfEditSave(mergeA, e => e.InsertPages(2, mergeB, 0, 1), mode));
        }

        // The stamp over merge-a's pages: the stamp sample's word at its place and size, translucent, with an image: its
        // /F1, /GS1 and /Im1 clash with the page's (inherited from the page tree), so they are named apart.
        var face = SubsetFace();
        byte[] stampBytes = File.ReadAllBytes(PdfEditSample("stamp").Path);
        using var stampDocument = PdfEditOpen(stampBytes, "stamp");
        var run = stampDocument.ReadPage(0, DocumentParts.Text).Runs.Single();
        var stamp = plan["stamp"]!;
        void Stamp(IDocumentEditor e)
        {
            foreach (var index in stamp["pages"]!.AsArray().Select(p => (int)p!))
            {
                var over = e.Overlay(index);
                over.Path(new DocumentPath(PdfBoxVerbs, PdfBoxPoints(run.Box)) { Fill = DocumentFill.NonZero, FillColour = DocumentColour.Rgb(0.8f, 0.1f, 0.1f).WithAlpha(0.3f) });
                over.Glyphs(face, run.FontSize, PdfShaped(face, run.Text, "Latn", false, run.Box.X, run.Box.Bottom - run.FontSize * 0.2f, run.FontSize), run.Text, DocumentTextMode.Fill, DocumentColour.Rgb(0.8f, 0.1f, 0.1f));
                over.Image(PdfLineArt(3, 30, 20), new DocumentBox(10, 10, 30, 20), DocumentImageEncoding.Flate);
            }
        }

        foreach (var mode in new[] { DocumentSaveMode.Incremental, DocumentSaveMode.Rewrite })
        {
            byte[] stamped = PdfEditSave(mergeA, Stamp, mode);
            using var document = PdfEditOpen(stamped, $"stamped {mode}");
            var stampResult = stamp["result"]!.AsArray();
            for (int i = 0; i < stampResult.Count; i++)
            {
                var page = document.ReadPage(i, DocumentParts.Text | DocumentParts.Images);
                var words = stampResult[i]!["text"]!.AsArray().Select(w => (string)w!).ToList();
                Check(words.All(page.Text.Contains), $"stamped {mode}: page {i} reads \"{page.Text}\", expected {string.Join(", ", words)} in it");
                var fonts = PdfEditResources(document, i, "Font");
                var images = PdfEditResources(document, i, "XObject");
                var states = PdfEditResources(document, i, "ExtGState");
                Check(document.File.Dictionary(fonts["F1"u8])!["BaseFont"u8].IsName("Helvetica"u8) && fonts.Contains("Ov1F1"u8) && images.Contains("Im1"u8) && images.Contains("Ov1Im1"u8)
                      && document.File.Dictionary(states["GS1"u8])!["ca"u8].TryNumber(out double own) && own == 1 && document.File.Dictionary(states["Ov1GS1"u8])!["ca"u8].TryNumber(out double alpha) && Math.Abs(alpha - 0.3) < 1e-6,
                    $"stamped {mode}: page {i}'s names apart (fonts {fonts.Count}, images {images.Count}, states {states.Count})");
                var original = mergeA.ReadPage(i, DocumentParts.Images).Images.Single();
                Check(page.Images.Count == 2 && page.Images[0].Bytes().Span.SequenceEqual(original.Bytes().Span) && page.Images[1].PixelWidth == 30, $"stamped {mode}: page {i}'s images");
            }

            PdfSaveEdited($"stamped-{mode}", stamped, PdfEditPagesRead(document));
        }

        // Split: each part's pages copied into a file of their own (the builder), and the others removed (the editor).
        var split = PdfEditSample("split-source");
        using var whole = PdfEditOpen(File.ReadAllBytes(split.Path), "split-source");
        int part = 0;
        foreach (var expected in split.Expected["split"]!["parts"]!.AsArray())
        {
            var pages = expected!["pages"]!.AsArray().Select(p => (int)p!).ToArray();
            var texts = expected["text"]!.AsArray().Select(t => (string)t!).ToList();
            var rotations = expected["rotations"]!.AsArray().Select(r => (int)r!).ToList();
            using var destination = new MemoryStream();
            using (var builder = new PdfWriter().Create(destination, PdfWriteDefaults))
            {
                foreach (int page in pages)
                {
                    builder.CopyPage(whole, page);
                }

                builder.Finish();
            }

            byte[] removed = PdfEditSave(whole, e => e.RemovePages([.. Enumerable.Range(0, whole.PageCount).Except(pages)]), DocumentSaveMode.Rewrite);
            foreach (var (how, bytes) in new[] { ("copied", destination.ToArray()), ("removed", removed) })
            {
                using var document = PdfEditOpen(bytes, $"part {part + 1} {how}");
                var read = PdfEditPagesRead(document);
                Check(read.Select(p => p.Text).SequenceEqual(texts) && read.Select(p => p.Rotation).SequenceEqual(rotations), $"part {part + 1} {how}: {PdfEditDescribe(read)}");
                Check(bytes.Length < new FileInfo(split.Path).Length, $"part {part + 1} {how}: {bytes.Length} bytes, no more than the whole file's");
                PdfSaveEdited($"split-{part + 1}-{how}", bytes, read);
            }

            part++;
        }

        // Duplicates: merge-b copied from two copies of its file holds its image and its font program once.
        using var copy = PdfEditOpen(b, "merge-b, a second copy");
        using var twice = new MemoryStream();
        using (var builder = new PdfWriter().Create(twice, PdfWriteDefaults))
        {
            builder.CopyPage(mergeB, 0);
            builder.CopyPage(copy, 1);
            builder.CopyPage(mergeA, 0);
            builder.Finish();
        }

        using (var document = PdfEditOpen(twice.ToArray(), "merged twice"))
        {
            int images = 0, programs = 0;
            foreach (var (number, _, _) in document.File.LiveObjects())
            {
                var value = document.File.Get(number);
                images += value.AsStream?.Dictionary["Subtype"u8].IsName("Image"u8) == true ? 1 : 0;
                programs += value.AsDictionary?["FontFile2"u8].IsReference == true ? 1 : 0;
            }

            Check(images == 2 && programs == 1, $"merge-b's image and font written once, merge-a's image beside them: {images} images, {programs} embedded font programs");
            Check(PdfEditPagesRead(document).Select(p => p.Text).SequenceEqual(["Support report\nPage 1 of merge-b", "Response times\nPage 2 of merge-b", "Sales report\nPage 1 of merge-a"]),
                "the pages merged twice read as their sources");
        }
    }

    private static void PdfEditOperations(Device device)
    {
        CpuOnly(device);
        var sample = PdfEditSample("page-operations");
        byte[] bytes = File.ReadAllBytes(sample.Path);
        using var source = PdfEditOpen(bytes, "page-operations");
        var others = new Dictionary<string, PdfDocument>();
        try
        {
            foreach (var operation in sample.Expected["operations"]!.AsArray())
            {
                string name = (string)operation!["name"]!;
                var expected = operation["result"]!.AsArray();
                foreach (var mode in new[] { DocumentSaveMode.Incremental, DocumentSaveMode.Rewrite })
                {
                    byte[] edited = PdfEditSave(source, e =>
                    {
                        var rotations = Enumerable.Range(0, source.PageCount).Select(i => source.ReadPage(i, DocumentParts.None).Rotation).ToList();
                        foreach (var step in operation["steps"]!.AsArray())
                        {
                            switch ((string)step!["op"]!)
                            {
                                case "insert":
                                    string from = (string)step["from"]!;
                                    if (!others.TryGetValue(from, out var other))
                                    {
                                        others[from] = other = PdfEditOpen(File.ReadAllBytes(PdfEditSample(from).Path), from);
                                    }

                                    var pages = step["pages"]!.AsArray().Select(p => (int)p!).ToArray();
                                    e.InsertPages((int)step["at"]!, other, pages);
                                    rotations.InsertRange((int)step["at"]!, pages.Select(p => other.ReadPage(p, DocumentParts.None).Rotation));
                                    break;
                                case "remove":
                                    var removed = step["pages"]!.AsArray().Select(p => (int)p!).ToArray();
                                    e.RemovePages(removed);
                                    foreach (int page in removed.OrderDescending())
                                    {
                                        rotations.RemoveAt(page);
                                    }

                                    break;
                                case "move":
                                    e.MovePage((int)step["from"]!, (int)step["to"]!);
                                    int moved = rotations[(int)step["from"]!];
                                    rotations.RemoveAt((int)step["from"]!);
                                    rotations.Insert((int)step["to"]!, moved);
                                    break;
                                default:
                                    int at = (int)step["page"]!;
                                    rotations[at] = ((rotations[at] + (int)step["by"]!) % 360 + 360) % 360;
                                    e.RotatePage(at, rotations[at]);
                                    break;
                            }
                        }
                    }, mode);
                    using var document = PdfEditOpen(edited, $"{name} {mode}");
                    var read = PdfEditPagesRead(document);
                    Check(read.Count == expected.Count && read.Zip(expected).All(p => p.First.Text == (string)p.Second!["text"]! && p.First.Rotation == (int)p.Second["rotation"]!),
                        $"{name} {mode}: {PdfEditDescribe(read)}; expected {string.Join(" | ", expected.Select(r => $"{r!["text"]} ({r["rotation"]})"))}");
                    if (mode == DocumentSaveMode.Incremental)
                    {
                        Check(edited.AsSpan(0, bytes.Length).SequenceEqual(bytes), $"{name}: the source is the update's prefix");
                    }

                    PdfSaveEdited($"operations-{name}-{mode}", edited, read.All(p => p.Rotation == 0) ? read : null);
                }
            }
        }
        finally
        {
            foreach (var other in others.Values)
            {
                other.Dispose();
            }
        }

        // Content over a page turned 0, 90, 180 and 270 degrees reads upright where it was put on the page as shown.
        var face = SubsetFace();
        foreach (int rotation in new[] { 0, 90, 180, 270 })
        {
            byte[] stamped = PdfEditSave(source, e =>
            {
                e.RotatePage(0, rotation);
                var over = e.Overlay(0);
                over.Glyphs(face, 12, PdfShaped(face, "Checked", "Latn", false, 20, 30, 12), "Checked", DocumentTextMode.Fill, DocumentColour.Black);
            }, DocumentSaveMode.Incremental);
            using var document = PdfEditOpen(stamped, $"stamped at {rotation}");
            var page = document.ReadPage(0, DocumentParts.Text);
            var run = page.Runs.Single(r => r.Text == "Checked");
            Check(page.Rotation == rotation && MathF.Abs(run.Box.X - 20) < 0.01f && MathF.Abs(run.Baseline!.Value - 30) < 0.01f && run.Box.Width > run.Box.Height,
                $"the stamp over the page turned {rotation}: {run.Box} on its baseline {run.Baseline}");
        }
    }

    private static void PdfEditRewrite(Device device)
    {
        CpuOnly(device);
        int rewritten = 0;
        long before = 0, after = 0;
        List<string> larger = [], repaired = [];
        foreach (var sample in PdfSamples().Where(s => s.Readable))
        {
            byte[] bytes = File.ReadAllBytes(sample.Path);
            using var source = PdfEditOpen(bytes, sample.Name);
            byte[] clean;
            try
            {
                clean = PdfEditSave(source, _ => { }, DocumentSaveMode.Rewrite);
            }
            catch (Exception e) when (e is InvalidDataException or NotSupportedException or InvalidOperationException)
            {
                throw new InvalidOperationException($"{sample.Name}: the rewrite failed: {e.GetType().Name}: {e.Message}", e);
            }

            if (clean.Length > bytes.Length)
            {
                // A source the reader repaired gets the cross-reference it lacked: the one rewrite that may add.
                (source.Info.Repaired ? repaired : larger).Add(string.Create(CultureInfo.InvariantCulture, $"{sample.Name} {bytes.Length} -> {clean.Length}"));
            }

            using var back = PdfEditOpen(clean, sample.Name + " rewritten");
            PdfEditSameDocument(sample.Name, source, back);
            (before, after, rewritten) = (before + bytes.Length, after + clean.Length, rewritten + 1);
            if (rewritten % 10 == 1)
            {
                Check(PdfEditSave(source, _ => { }, DocumentSaveMode.Rewrite).AsSpan().SequenceEqual(clean), $"{sample.Name}: rewritten twice, the same bytes");
            }

            if (sample.Phases.Contains("D11") || sample.Phases.Contains("D6"))
            {
                PdfSaveEdited($"clean-{sample.Name}", clean);
            }
        }

        Check(rewritten >= 40 && larger.Count == 0, $"{rewritten} samples rewritten; larger than their source: {string.Join("; ", larger)}");
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    {rewritten} samples rewritten: {before:N0} bytes to {after:N0} ({100.0 * after / before:F1}%); repaired sources given their cross-reference: {(repaired.Count == 0 ? "none larger" : string.Join("; ", repaired))}"));
    }

    // Two documents read the same: information, pages, sizes, rotations, text, runs, images (their data) and issues.
    private static void PdfEditSameDocument(string what, IDocument expected, IDocument actual)
    {
        var a = expected.Info;
        var b = actual.Info;
        Check(a.Title == b.Title && a.Author == b.Author && a.Subject == b.Subject && a.Keywords == b.Keywords && a.Creator == b.Creator && a.Producer == b.Producer
              && a.Created == b.Created && a.Modified == b.Modified && a.Entries.Count == b.Entries.Count, $"{what}: the information reads the same");
        Check(expected.PageCount == actual.PageCount, $"{what}: {actual.PageCount} pages, {expected.PageCount} in the source");
        for (int i = 0; i < expected.PageCount; i++)
        {
            var p = expected.ReadPage(i, DocumentParts.Text | DocumentParts.Images);
            var q = actual.ReadPage(i, DocumentParts.Text | DocumentParts.Images);
            Check(p.Width == q.Width && p.Height == q.Height && p.Rotation == q.Rotation && p.Text == q.Text && p.Runs.Count == q.Runs.Count
                  && p.Runs.Zip(q.Runs).All(r => r.First.Text == r.Second.Text && r.First.Box == r.Second.Box && r.First.FontSize == r.Second.FontSize && r.First.Hidden == r.Second.Hidden),
                $"{what}: page {i} reads the same ({p.Width}x{p.Height} {p.Rotation} \"{p.Text}\" against {q.Width}x{q.Height} {q.Rotation} \"{q.Text}\")");
            Check(p.Issues.Select(x => (x.Kind, x.Count)).SequenceEqual(q.Issues.Select(x => (x.Kind, x.Count))), $"{what}: page {i}'s issues: {string.Join("; ", q.Issues)}");
            Check(p.Images.Count == q.Images.Count, $"{what}: page {i}: {q.Images.Count} images, {p.Images.Count} in the source");
            for (int k = 0; k < p.Images.Count; k++)
            {
                var x = p.Images[k];
                var y = q.Images[k];
                Check(x.Encoding == y.Encoding && x.PixelWidth == y.PixelWidth && x.PixelHeight == y.PixelHeight && x.Box == y.Box && x.Bytes().Span.SequenceEqual(y.Bytes().Span),
                    $"{what}: page {i}'s image {k} the same");
            }
        }
    }

    private static void PdfEditEncryption(Device device)
    {
        CpuOnly(device);
        var face = SubsetFace();
        var encryption = new DocumentEncryption { UserPassword = PdfEditUser, OwnerPassword = PdfEditOwner, Permissions = DocumentPermissions.Print | DocumentPermissions.Copy | DocumentPermissions.Accessibility };
        const string Secret = "Quarterly figures for the board";

        // The builder: text, an image, a file attached and the information, encrypted, in the three forms of the file.
        foreach (var (label, options) in new[] { ("modern", PdfWriteDefaults), ("classic", PdfWriteDefaults with { Version = "1.4" }), ("unpacked", PdfWriteDefaults with { ObjectStreams = false }) })
        {
            using var destination = new MemoryStream();
            using (var builder = new PdfWriter().Create(destination, options))
            {
                builder.Encrypt(encryption);
                builder.Info = new DocumentInfo { Title = Secret, Author = "The office team" };
                var page = builder.AddPage(400, 200);
                page.Glyphs(face, 14, PdfShaped(face, Secret, "Latn", false, 20, 40, 14), Secret, DocumentTextMode.Fill, DocumentColour.Black);
                page.Image(PdfLineArt(3, 40, 30), new DocumentBox(20, 60, 80, 60), DocumentImageEncoding.Flate);
                builder.Attach("notes.txt", "text/plain", new MemoryStream(Encoding.ASCII.GetBytes(Secret)));
                builder.Finish();
            }

            byte[] bytes = destination.ToArray();
            PdfEditCheckEncrypted($"built {label}", bytes, encryption, [Secret]);
            using var document = PdfEditOpen(bytes, label, PdfEditUser);
            using var attached = document.Attachments.Single().Read();
            using var copy = new MemoryStream();
            attached.CopyTo(copy);
            Check(Encoding.ASCII.GetString(copy.ToArray()) == Secret && document.Info.Title == Secret && document.File.HeaderVersion == (options.Version ?? "1.7"),
                $"built {label}: the attached file and the information read back");
            PdfSaveEdited($"built-{label}", bytes, PdfEditPagesRead(document), encrypted: true);
        }

        // encrypted-aes256: updated with its encryption kept (opened by the user), its encryption removed and replaced (opened by the owner).
        var sample = PdfEditSample("encrypted-aes256");
        byte[] source = File.ReadAllBytes(sample.Path);
        int permissions = (int)sample.Expected["encryption"]!["permissions"]!;
        using (var asUser = PdfEditOpen(source, "encrypted-aes256", PdfEditUser))
        {
            var pages = PdfEditPagesRead(asUser);
            byte[] update = PdfEditSave(asUser, e =>
            {
                e.SetInfo(new DocumentInfo { Title = "Salary review, reviewed" });
                PdfEditStamp(e.Overlay(1), face, "Reviewed");
            }, DocumentSaveMode.Incremental);
            Check(update.AsSpan(0, source.Length).SequenceEqual(source), "encrypted-aes256: the update keeps the source's bytes");
            var expected = new DocumentEncryption { UserPassword = PdfEditUser, OwnerPassword = PdfEditOwner, Permissions = PdfSecurity.ToPermissions(permissions) };
            PdfEditCheckEncrypted("encrypted-aes256 updated", update, expected, ["Reviewed", "Salary review, reviewed"], bits: permissions);
            using var updated = PdfEditOpen(update, "updated", PdfEditUser);
            var read = PdfEditPagesRead(updated);
            Check(read[0] == pages[0] && read[1].Text == pages[1].Text + "\nReviewed" && updated.Info.Title == "Salary review, reviewed", $"encrypted-aes256 updated: {PdfEditDescribe(read)}");
            PdfSaveEdited("encrypted-aes256-updated", update, read, encrypted: true);

            ExpectRefusal<InvalidOperationException>(() => PdfEditSave(asUser, e => e.Encrypt(null), DocumentSaveMode.Rewrite), "owner password", "encryption removed by the user");
            ExpectRefusal<InvalidOperationException>(() => PdfEditSave(asUser, e => e.Encrypt(encryption), DocumentSaveMode.Rewrite), "owner password", "encryption changed by the user");
            byte[] kept = PdfEditSave(asUser, _ => { }, DocumentSaveMode.Rewrite);
            PdfEditCheckEncrypted("encrypted-aes256 rewritten as it is", kept, expected, ["Salary review"], bits: permissions);
        }

        using (var asOwner = PdfEditOpen(source, "encrypted-aes256", PdfEditOwner))
        {
            // Unpacked and stored, so its strings show as written: plain.
            byte[] plain = PdfEditSave(asOwner, e => e.Encrypt(null), DocumentSaveMode.Rewrite, PdfWriteDefaults with { ObjectStreams = false, Compression = 0 });
            using (var opened = PdfEditOpen(plain, "decrypted"))
            {
                Check(!opened.Info.Encrypted && opened.File.Trailer["Encrypt"u8].IsNull && Encoding.Latin1.GetString(plain).Contains("Salary review", StringComparison.Ordinal),
                    "encrypted-aes256 rewritten without its encryption: open, its title plain");
                PdfEditSameDocument("decrypted", asOwner, opened);
                PdfSaveEdited("encrypted-aes256-decrypted", plain, PdfEditPagesRead(opened));
            }

            var other = new DocumentEncryption { UserPassword = "Grüße-2026", OwnerPassword = "another-owner", Permissions = DocumentPermissions.None };
            byte[] again = PdfEditSave(asOwner, e => e.Encrypt(other), DocumentSaveMode.Rewrite);
            PdfEditCheckEncrypted("encrypted-aes256 encrypted anew", again, other, ["Salary review"]);
            Check(PdfEditRefused(again, PdfEditUser), "encrypted anew: the old user password no longer opens it");
            ExpectRefusal<InvalidOperationException>(() => PdfEditSave(asOwner, e => e.Encrypt(other), DocumentSaveMode.Incremental), "rewrite", "encryption changed by an update");
        }

        // A plain file encrypted by a rewrite; a builder whose owner password is none (nobody can lift its permissions).
        using (var plainSource = PdfEditOpen(File.ReadAllBytes(PdfEditSample("merge-b").Path), "merge-b"))
        {
            byte[] encrypted = PdfEditSave(plainSource, e => e.Encrypt(encryption), DocumentSaveMode.Rewrite);
            PdfEditCheckEncrypted("merge-b encrypted by a rewrite", encrypted, encryption, ["Support report"]);
            using var opened = PdfEditOpen(encrypted, "merge-b encrypted", PdfEditUser);
            PdfEditSameDocument("merge-b encrypted", plainSource, opened);
            PdfSaveEdited("merge-b-encrypted", encrypted, PdfEditPagesRead(opened), encrypted: true);
        }

        // Every encrypted sample of D2 and D11, rewritten and updated with its own encryption, reads the same.
        int count = 0;
        foreach (var encrypted in PdfSamples().Where(s => s.Encrypted && !s.Hostile && s.Expected["openError"] is null && s.Expected["encryption"] is JsonObject))
        {
            string? password = (string?)encrypted.Expected["open"]?["password"];
            byte[] bytes = File.ReadAllBytes(encrypted.Path);
            PdfDocument opened;
            try
            {
                opened = PdfEditOpen(bytes, encrypted.Name, password);
            }
            catch (NotSupportedException)
            {
                continue;                                                      // a handler the reader names and refuses (D2 checks it)
            }

            using (opened)
            {
                foreach (var mode in new[] { DocumentSaveMode.Rewrite, DocumentSaveMode.Incremental })
                {
                    byte[] saved = PdfEditSave(opened, e => e.SetInfo(opened.Info with { Keywords = "edited" }), mode);
                    using var back = PdfEditOpen(saved, $"{encrypted.Name} {mode}", password);
                    Check(back.Info.Encrypted && back.Info.Keywords == "edited", $"{encrypted.Name} {mode}: encrypted, the keywords set");
                    for (int i = 0; i < opened.PageCount; i++)
                    {
                        Check(back.ReadPage(i, DocumentParts.Text).Text == opened.ReadPage(i, DocumentParts.Text).Text, $"{encrypted.Name} {mode}: page {i} reads the same");
                    }

                    Check(PdfEditRefused(saved, null) == PdfEditRefused(bytes, null), $"{encrypted.Name} {mode}: opens without a password as its source does");
                }
            }

            count++;
        }

        Check(count >= 20, $"{count} encrypted samples rewritten and updated");
    }

    // A file encrypted as `encryption` says: refused without a password and with a wrong one; open with each; its
    // encryption dictionary AES-256 revision 6 with /P and /Perms as made; `plainWords` nowhere in its bytes.
    private static void PdfEditCheckEncrypted(string what, byte[] bytes, DocumentEncryption encryption, string[] plainWords, int? bits = null)
    {
        Check(PdfEditRefused(bytes, null) == !string.IsNullOrEmpty(encryption.UserPassword) && PdfEditRefused(bytes, "not-the-password"), $"{what}: refused without its password and with a wrong one");
        string text = Encoding.Latin1.GetString(bytes);
        Check(!plainWords.Any(w => text.Contains(w, StringComparison.Ordinal)), $"{what}: none of {string.Join(", ", plainWords)} is in the file as written");
        foreach (var (password, owner) in new[] { (encryption.UserPassword, false), (encryption.OwnerPassword, true) })
        {
            if (password is null)
            {
                continue;
            }

            using var document = PdfEditOpen(bytes, what, password);
            var security = document.File.Security!;
            var dictionary = document.File.Dictionary(document.File.Trailer["Encrypt"u8])!;
            byte[] perms = document.File.Resolve(dictionary["Perms"u8]).AsString!.Bytes;
            Check(document.Info.Encrypted && security.Revision == 6 && security.Owner == owner && security.Strings == PdfCryptMethod.Aes256 && security.Streams == PdfCryptMethod.Aes256
                  && security.PermissionBits == (bits ?? PdfSecurity.ToBits(encryption.Permissions)) && security.Permissions == encryption.Permissions && security.CheckPerms(perms)
                  && document.File.Resolve(dictionary["O"u8]).AsString!.Bytes.Length == 48 && document.File.Resolve(dictionary["U"u8]).AsString!.Bytes.Length == 48
                  && document.File.Resolve(dictionary["OE"u8]).AsString!.Bytes.Length == 32 && document.File.Resolve(dictionary["UE"u8]).AsString!.Bytes.Length == 32,
                $"{what}: opened as {(owner ? "owner" : "user")}: revision {security.Revision}, /P {security.PermissionBits}, permissions {security.Permissions}");
            for (int i = 0; i < document.PageCount; i++)
            {
                document.ReadPage(i, DocumentParts.Text | DocumentParts.Images);
            }
        }
    }

    // Whether opening `bytes` with `password` is refused naming the option.
    private static bool PdfEditRefused(byte[] bytes, string? password)
    {
        try
        {
            using var document = PdfEditOpen(bytes, "refused", password);
            return false;
        }
        catch (InvalidDataException e) when (e.Message.Contains("DocumentOptions.Password", StringComparison.Ordinal))
        {
            return true;
        }
    }

    private static void PdfEditSigned(Device device)
    {
        CpuOnly(device);
        var face = SubsetFace();
        int signed = 0;
        foreach (var sample in PdfSamples().Where(s => s.Expected["signatures"] is JsonArray { Count: > 0 } && !s.Encrypted))
        {
            byte[] bytes = File.ReadAllBytes(sample.Path);
            using var source = PdfEditOpen(bytes, sample.Name);
            var before = PdfEditPagesRead(source);
            byte[] update = PdfEditSave(source, e =>
            {
                e.SetInfo(new DocumentInfo { Title = "Signed and stamped" });
                PdfEditStamp(e.Overlay(0), face, "Received");
            }, DocumentSaveMode.Incremental);
            foreach (var signature in sample.Expected["signatures"]!.AsArray())
            {
                var range = signature!["byteRange"]!.AsArray().Select(v => (long)v!).ToArray();
                for (int k = 0; k + 1 < range.Length; k += 2)
                {
                    Check(range[k] + range[k + 1] <= bytes.Length && update.AsSpan((int)range[k], (int)range[k + 1]).SequenceEqual(bytes.AsSpan((int)range[k], (int)range[k + 1])),
                        $"{sample.Name}: {signature["field"]}'s bytes {range[k]} to {range[k] + range[k + 1]} unchanged");
                }
            }

            using var updated = PdfEditOpen(update, sample.Name + " updated");
            var read = PdfEditPagesRead(updated);
            Check(read[0].Text == before[0].Text + "\nReceived" && updated.Info.Title == "Signed and stamped", $"{sample.Name} updated: {PdfEditDescribe(read)}");
            var signatures = Signatures(source.File);
            var kept = Signatures(updated.File);
            Check(signatures.Count == sample.Expected["signatures"]!.AsArray().Count && kept.SequenceEqual(signatures), $"{sample.Name}: {kept.Count} signature dictionaries read the same");
            PdfSaveEdited($"signed-{sample.Name}", update, read);
            signed++;
        }

        Check(signed >= 10, $"{signed} signed samples updated");

        // Each signature dictionary's byte range and the SHA-256 of its contents.
        static List<string> Signatures(PdfFile file)
        {
            var found = new List<string>();
            foreach (var (number, _, _) in file.LiveObjects())
            {
                if (file.Get(number).AsDictionary is { } dictionary && dictionary["ByteRange"u8].AsArray is { } range && dictionary["Contents"u8].AsString is { } contents)
                {
                    found.Add($"{string.Join(" ", range.Items.Select(v => v.ToString()))} {Convert.ToHexStringLower(SHA256.HashData(contents.Bytes))}");
                }
            }

            found.Sort(StringComparer.Ordinal);
            return found;
        }
    }

    private static void PdfEditRefusals(Device device)
    {
        CpuOnly(device);
        var sample = PdfEditSample("merge-a");
        byte[] bytes = File.ReadAllBytes(sample.Path);
        using var source = PdfEditOpen(bytes, "merge-a");
        using (var editor = new PdfWriter().Edit(source, new MemoryStream(), PdfWriteDefaults))
        {
            Check(ReferenceEquals(editor.Source, source) && editor.PageCount == 2, "the editor's source and pages");
            ExpectRefusal<ArgumentException>(() => editor.SetField("status", "Approved"), "no form field 'status'", "a field the document does not have");
            ExpectRefusal<ArgumentNullException>(() => editor.AddField(null!), "", "no field to add");
            ExpectRefusal<ArgumentException>(() => editor.Flatten("status"), "no form field 'status'", "a field to flatten the document does not have");
            ExpectRefusal<ArgumentException>(() => editor.RemoveAnnotation(new DocumentAnnotation(DocumentAnnotationKinds.Note, 0, default) { Id = "no-such-note" }), "no-such-note",
                "an annotation the page does not hold removed (D10 removes them)");
            ExpectRefusal<ArgumentOutOfRangeException>(() => editor.AddAnnotation(new DocumentAnnotation(DocumentAnnotationKinds.Note, 7, default)), "2 pages", "an annotation on a page out of range");
            ExpectRefusal<ArgumentNullException>(() => editor.AddOutline(null!), "outline", "a bookmark that is null");
            ExpectRefusal<ArgumentNullException>(() => editor.Sign("Signature1", null!), "options", "signing without options");
            ExpectRefusal<ArgumentOutOfRangeException>(() => editor.RotatePage(0, 45), "multiple of 90", "a rotation of 45");
            ExpectRefusal<ArgumentOutOfRangeException>(() => editor.RemovePages(5), "2 pages", "a page out of range");
            ExpectRefusal<ArgumentException>(() => editor.RemovePages(0, 0), "named twice", "a page named twice");
            ExpectRefusal<ArgumentOutOfRangeException>(() => editor.MovePage(0, 2), "2 pages", "a move past the end");
            ExpectRefusal<ArgumentOutOfRangeException>(() => editor.InsertPages(0, source, 2), "2 pages", "a page inserted that is not there");
            using var others = new PdfWrittenDocument(PdfEditOpen(bytes, "another reader's"));
            ExpectRefusal<NotSupportedException>(() => editor.InsertPages(0, others, 0), "another reader opened", "a page of another reader's document");
            ExpectRefusal<InvalidOperationException>(() => { editor.Encrypt(new DocumentEncryption()); editor.Save(DocumentSaveMode.Incremental); }, "rewrite", "encryption set by an update");
            ExpectRefusal<InvalidOperationException>(() => editor.Save(DocumentSaveMode.Rewrite), "saves once", "a second Save");
        }

        using (var other = new PdfWrittenDocument(PdfEditOpen(bytes, "another reader's")))
        {
            ExpectRefusal<NotSupportedException>(() => new PdfWriter().Edit(other, new MemoryStream(), PdfWriteDefaults), "another reader opened", "a document another reader opened");
        }

        using (var builder = new PdfWriter().Create(new MemoryStream(), PdfWriteDefaults))
        {
            builder.AddPage(100, 100);
            ExpectRefusal<InvalidOperationException>(() => builder.Encrypt(new DocumentEncryption()), "before the first page", "encrypted after the first page");
            ExpectRefusal<ArgumentOutOfRangeException>(() => builder.CopyPage(source, 2), "", "a page copied that is not there");
            builder.Finish();
        }
    }

    private static void PdfEditTimings(Device device)
    {
        CpuOnly(device);
        // The D13 large set, where make_fixtures.py --large writes it: beside the samples' folder.
        string samples = Environment.GetEnvironmentVariable("IDRAK_DOCUMENTS_SAMPLES") is { Length: > 0 } set ? set : RepositoryPath("test-output", "documents", "samples");
        string large = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(samples).TrimEnd(Path.DirectorySeparatorChar)) ?? samples, "large", "d13", "pages-3000.pdf");
        byte[] bytes;
        string name;
        if (File.Exists(large))
        {
            (bytes, name) = (File.ReadAllBytes(large), "pages-3000 (D13's large set)");
        }
        else
        {
            // Without the large set (make_fixtures.py --large): a file of 1,500 pages of text and a small image each.
            var face = SubsetFace();
            using var destination = new MemoryStream();
            using (var builder = new PdfWriter().Create(destination, PdfWriteDefaults))
            {
                var image = PdfLineArt(3, 60, 40);
                for (int i = 0; i < 1500; i++)
                {
                    var page = builder.AddPage(400, 300);
                    string line = string.Create(CultureInfo.InvariantCulture, $"Section {i + 1}: the figures for this part of the report.");
                    page.Glyphs(face, 12, PdfShaped(face, line, "Latn", false, 30, 40, 12), line, DocumentTextMode.Fill, DocumentColour.Black);
                    page.Image(image, new DocumentBox(30, 60, 120, 80), DocumentImageEncoding.Flate);
                }

                builder.Finish();
            }

            (bytes, name) = (destination.ToArray(), "1,500 pages written here (no large set)");
        }

        using var source = PdfEditOpen(bytes, name);
        using var copy = PdfEditOpen(bytes, name + ", a second copy");
        string otherPath = Path.Combine(Path.GetDirectoryName(large)!, "huge-image.pdf");
        string otherName = File.Exists(otherPath) ? "huge-image (D13's large set)" : "merge-b";
        using var other = PdfEditOpen(File.ReadAllBytes(File.Exists(otherPath) ? otherPath : PdfEditSample("merge-b").Path), otherName);
        int pages = source.PageCount;
        long peakBefore = Process.GetCurrentProcess().PeakWorkingSet64;

        // Each measure: the first call (cold), then the median of five (warm); bytes written and allocated.
        (double Cold, double Warm, long Bytes, long Allocated) Measure(Func<long> run, int warm = 5)
        {
            long start = Stopwatch.GetTimestamp();
            long written = run();
            double cold = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            var times = new List<double>();
            long allocated = 0;
            for (int i = 0; i < warm; i++)
            {
                GC.Collect();
                long before = GC.GetTotalAllocatedBytes(precise: true);
                start = Stopwatch.GetTimestamp();
                run();
                times.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds);
                allocated = GC.GetTotalAllocatedBytes(precise: true) - before;
            }

            times.Sort();
            return (cold, times[times.Count / 2], written, allocated);
        }

        // Written to a stream that only counts (nothing kept, nothing hashed).
        long Update(DocumentSaveMode mode)
        {
            using var counter = new PdfByteCounter();
            using (var editor = new PdfWriter().Edit(source, counter, PdfWriteDefaults))
            {
                editor.SetInfo(new DocumentInfo { Title = "A one-field change" });
                editor.Save(mode);
            }

            return counter.Length;
        }

        // Every page of `first`, then every page of `second`, copied into a new file.
        long Merge(PdfDocument first, PdfDocument second)
        {
            using var counter = new PdfByteCounter();
            using (var builder = new PdfWriter().Create(counter, PdfWriteDefaults))
            {
                foreach (var document in new[] { first, second })
                {
                    for (int i = 0; i < document.PageCount; i++)
                    {
                        builder.CopyPage(document, i);
                    }
                }

                builder.Finish();
            }

            return counter.Length;
        }

        // The update writes the source's bytes again (its prefix) and appends the change: what it adds is the measure.
        var update = Measure(() => Update(DocumentSaveMode.Incremental));
        var rewrite = Measure(() => Update(DocumentSaveMode.Rewrite), warm: 3);
        var merge = Measure(() => Merge(source, other), warm: 3);
        var twice = Measure(() => Merge(source, copy), warm: 3);
        long peak = Process.GetCurrentProcess().PeakWorkingSet64;
        Check(update.Bytes > bytes.Length && update.Bytes - bytes.Length < 4096, $"a one-field update appends {update.Bytes - bytes.Length} bytes");
        Check(rewrite.Bytes <= bytes.Length && merge.Bytes > 0 && twice.Bytes < bytes.Length * 1.1, $"the rewrite {rewrite.Bytes} bytes, the merges {merge.Bytes} and {twice.Bytes}");
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    {name}: {bytes.Length:N0} bytes, {pages} pages; a one-field incremental update: {update.Bytes - bytes.Length:N0} bytes appended ({update.Bytes:N0} written with the prefix), cold {update.Cold:F1} ms, warm {update.Warm:F1} ms, {update.Allocated / 1024.0:F0} KiB allocated"));
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    the rewrite: {rewrite.Bytes:N0} bytes, cold {rewrite.Cold:F0} ms, warm {rewrite.Warm:F0} ms ({rewrite.Warm * 1000 / pages:F0} µs a page), {rewrite.Allocated / 1048576.0:F1} MiB allocated"));
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    merged with {otherName} ({pages + other.PageCount} pages, {bytes.Length + other.File.Source.Length:N0} bytes in): {merge.Bytes:N0} bytes, cold {merge.Cold:F0} ms, warm {merge.Warm:F0} ms ({merge.Warm * 1000 / (pages + other.PageCount):F0} µs a page), {merge.Allocated / 1048576.0:F1} MiB allocated"));
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    merged with a copy of itself ({2 * pages} pages, every stream and font the same): {twice.Bytes:N0} bytes, cold {twice.Cold:F0} ms, warm {twice.Warm:F0} ms ({twice.Warm * 1000 / (2 * pages):F0} µs a page), {twice.Allocated / 1048576.0:F1} MiB allocated"));
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    the process's peak working set: {peakBefore / 1048576.0:F0} MiB before, {peak / 1048576.0:F0} MiB after; the managed heap now {GC.GetTotalMemory(forceFullCollection: true) / 1048576.0:F0} MiB"));
    }
}
