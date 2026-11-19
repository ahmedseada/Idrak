// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Data.Pdf;

// Plan 28, phase D10: annotations and navigation (every annotation kind read, drawn, created and removed; links,
// bookmarks, named destinations, page labels; a tagged file's order; redaction applied for real), checked on the D10
// samples tools/documents/make_fixtures.py writes into the ignored test-output/documents/samples. Every file written goes
// to test-output/documents/annotated (or, redacted, to test-output/documents/redacted), with NAME.json holding the text the
// reference readers must extract, for tools/documents/check_written.py.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] DocumentPdfAnnotationsGroup =
    [
        ("documents pdf: annotations: every annotation of the D10 samples (every kind with and without appearance streams, replies, popups, links) reads as its JSON says: id, kind, page, box, text, author, dates, colours, opacity, flags, points (quads, a line's ends, vertices), ink strokes, targets (web addresses, explicit and named destinations), popup, parent, thread, state, icon, border, line endings, appearance, attached file; replies under their note", PdfAnnotationsRead),
        ("documents pdf: annotations: navigation's bookmarks (titles, destinations by page and by name, colours, bold and italic, open and closed, nested), named destinations (the name tree's and the catalog's) and page labels (i ii 1 2 3 A-1 A-2 A Cover) read as its JSON says; a file without them reads none", PdfAnnotationsNavigation),
        ("documents pdf: annotations: a tagged page's text comes in its structure's order through the tagged layout (heading, left column, right column; the page number an artifact, left out), each run's tag its element's; lines stays the default, and a file without tags reads the same through both", PdfAnnotationsTagged),
        ("documents pdf: annotations: every kind created by the builder (an appearance made from its properties), links (a web address, a place), bookmarks and page labels, and annotations added and removed and bookmarks added by the editor (incremental and rewrite), read back as written; kept in test-output/documents/annotated for the reference tools", PdfAnnotationsWritten),
        ("documents pdf: annotations: redaction applied for real: redaction's own marks remove its words from the text, every decoded stream and the file's bytes, the path inside, the crossing path clipped outside the area, the image's pixels as its .redacted.png says, the marks themselves; the page draws as redaction-applied; boxes and text matches too; an incremental save refused", PdfAnnotationsRedaction),
        ("documents pdf: annotations: annotations draw (their appearances, or their looks from their properties; hidden ones and popups not) and the samples match pdfium's drawing within D7's tolerance", PdfAnnotationsDrawn),
        ("documents pdf: annotations: timings: every annotation of the D10 samples read, and a page redacted, cold and warm", PdfAnnotationsTimings),
    ];

    // The JSON's kinds as the library names them.
    private static readonly Dictionary<string, string> PdfAnnotationKindNames = new(StringComparer.Ordinal)
    {
        ["note"] = DocumentAnnotationKinds.Note, ["freeText"] = DocumentAnnotationKinds.FreeText, ["line"] = DocumentAnnotationKinds.Line,
        ["square"] = DocumentAnnotationKinds.Square, ["circle"] = DocumentAnnotationKinds.Circle, ["polygon"] = DocumentAnnotationKinds.Polygon,
        ["polyline"] = DocumentAnnotationKinds.Polyline, ["highlight"] = DocumentAnnotationKinds.Highlight, ["underline"] = DocumentAnnotationKinds.Underline,
        ["squiggly"] = DocumentAnnotationKinds.Squiggly, ["strikeOut"] = DocumentAnnotationKinds.StrikeOut, ["ink"] = DocumentAnnotationKinds.Ink,
        ["stamp"] = DocumentAnnotationKinds.Stamp, ["caret"] = DocumentAnnotationKinds.Caret, ["link"] = DocumentAnnotationKinds.Link,
        ["fileAttachment"] = DocumentAnnotationKinds.FileAttachment, ["popup"] = DocumentAnnotationKinds.Popup, ["redaction"] = DocumentAnnotationKinds.Redaction,
    };

    private static readonly Dictionary<string, DocumentDestinationFit> PdfFits = new(StringComparer.Ordinal)
    {
        ["XYZ"] = DocumentDestinationFit.Position, ["Fit"] = DocumentDestinationFit.Page, ["FitH"] = DocumentDestinationFit.Width, ["FitV"] = DocumentDestinationFit.Height,
        ["FitR"] = DocumentDestinationFit.Box, ["FitB"] = DocumentDestinationFit.Bounds, ["FitBH"] = DocumentDestinationFit.BoundsWidth, ["FitBV"] = DocumentDestinationFit.BoundsHeight,
    };

    private static bool PdfNear(float? a, double? b, double tolerance = 0.01) => a is null ? b is null : b is not null && Math.Abs(a.Value - b.Value) <= tolerance;

    private static float[] PdfFlat(JsonNode? node) => node switch
    {
        JsonArray array => [.. array.SelectMany(PdfFlat)],
        JsonValue value => [(float)value.GetValue<double>()],
        _ => [],
    };

    // A destination read against the JSON's target (explicit: its page, fit and numbers in user space; or a name and the
    // named destinations' pages). Pages here are upright and uncropped: points as shown are x and the height less y.
    private static string? PdfTargetDifference(DocumentDestination? got, JsonObject target, IDocument document, JsonObject? named)
    {
        if (target["name"] is JsonValue name)
        {
            string n = (string)name!;
            int page = named?[n]?["page"]?.GetValue<int>() ?? -1;
            return got is not null && got.Name == n && got.Page == page ? null : $"named {n} on page {page}, read {got?.Name} on page {got?.Page}";
        }

        int want = target["page"]!.GetValue<int>();
        string fit = (string)target["fit"]!;
        var values = target["values"]!.AsArray().Select(v => v is null ? (double?)null : v.GetValue<double>()).ToArray();
        float height = document.ReadPage(want, DocumentParts.None).Height;
        double? X(double? x) => x;
        double? Y(double? y) => y is null ? null : height - y;
        bool ok = got is not null && got.Page == want && got.Fit == PdfFits[fit];
        ok &= fit switch
        {
            "XYZ" => PdfNear(got!.X, X(values[0])) && PdfNear(got.Y, Y(values[1])) && PdfNear(got.Zoom, values[2] is > 0 ? values[2] : null),
            "FitH" or "FitBH" => PdfNear(got!.Y, Y(values[0])),
            "FitV" or "FitBV" => PdfNear(got!.X, X(values[0])),
            "FitR" => PdfNear(got!.X, values[0]) && PdfNear(got.Bottom, Y(values[1])) && PdfNear(got.Right, values[2]) && PdfNear(got.Y, Y(values[3])),
            _ => true,
        };
        return ok ? null : string.Create(CultureInfo.InvariantCulture,
            $"{fit} {target["values"]?.ToJsonString()} on page {want}, read {got?.Fit} on page {got?.Page} at ({got?.X}, {got?.Y}, {got?.Right}, {got?.Bottom}) zoom {got?.Zoom}");
    }

    private static string? PdfColourDifference(DocumentColour? got, JsonNode? expected, string what)
    {
        if (expected is null)
        {
            return got is null ? null : $"{what}: none expected, read {got}";
        }

        var rgb = expected["rgb"]!.AsArray().Select(v => v!.GetValue<int>()).ToArray();
        return got is { } c && Math.Abs(c.Red * 255 - rgb[0]) <= 1 && Math.Abs(c.Green * 255 - rgb[1]) <= 1 && Math.Abs(c.Blue * 255 - rgb[2]) <= 1
            ? null
            : $"{what}: [{string.Join(", ", rgb)}] expected, read {got}";
    }

    // Every annotation of a document, page by page.
    private static List<DocumentAnnotation> PdfAnnotationsOf(IDocument document, DocumentParts parts = DocumentParts.Annotations)
    {
        var all = new List<DocumentAnnotation>();
        for (int i = 0; i < document.PageCount; i++)
        {
            all.AddRange(document.ReadPage(i, parts).Annotations);
        }

        return all;
    }

    private static void PdfAnnotationsRead(Device device)
    {
        CpuOnly(device);
        var failures = new List<string>();
        int checkedCount = 0, expectedCount = 0;
        foreach (var sample in PdfSamples().Where(s => s.Phases.Contains("D10") && s.Expected["annotations"] is JsonArray))
        {
            expectedCount += sample.Expected["annotations"]!.AsArray().Count;
            using var document = DocumentFormats.Open(sample.Path);
            var read = PdfAnnotationsOf(document);
            var expected = sample.Expected["annotations"]!.AsArray();
            if (read.Count != expected.Count)
            {
                failures.Add($"{sample.Name}: {read.Count} annotations read, {expected.Count} in its JSON");
            }

            var named = sample.Expected["namedDestinations"]?.AsObject();
            foreach (var node in expected)
            {
                var e = node!.AsObject();
                string id = (string)e["id"]!;
                int page = e["page"]!.GetValue<int>();
                var a = read.FirstOrDefault(x => x.Page == page && x.Id == id);
                string where = $"{sample.Name}: {id}";
                if (a is null)
                {
                    failures.Add($"{where}: not read");
                    continue;
                }

                checkedCount++;
                var problems = new List<string>();
                void Expect(bool ok, string what)
                {
                    if (!ok)
                    {
                        problems.Add(what);
                    }
                }

                Expect(a.Kind == PdfAnnotationKindNames[(string)e["kind"]!], $"kind {a.Kind}");
                var box = PdfFlat(e["box"]);
                Expect(Math.Abs(a.Box.X - box[0]) < 0.01 && Math.Abs(a.Box.Y - box[1]) < 0.01 && Math.Abs(a.Box.Width - box[2]) < 0.01 && Math.Abs(a.Box.Height - box[3]) < 0.01,
                    $"box {a.Box}");
                Expect(a.Contents == (string?)e["contents"], $"contents '{a.Contents}'");
                Expect(a.Author == (string?)e["author"], $"author '{a.Author}'");
                Expect(a.Created == (e["created"] is JsonValue c ? (DateTimeOffset?)DateTimeOffset.Parse((string)c!, CultureInfo.InvariantCulture) : null), $"created {a.Created}");
                Expect(a.Modified == (e["modified"] is JsonValue m ? (DateTimeOffset?)DateTimeOffset.Parse((string)m!, CultureInfo.InvariantCulture) : null), $"modified {a.Modified}");
                if (PdfColourDifference(a.Colour, e["colour"], "colour") is { } colour)
                {
                    problems.Add(colour);
                }

                if (PdfColourDifference(a.InteriorColour, e["interiorColour"], "interior colour") is { } interior)
                {
                    problems.Add(interior);
                }

                Expect(Math.Abs(a.Opacity - e["opacity"]!.GetValue<double>()) < 1e-6, $"opacity {a.Opacity}");
                Expect((int)a.Flags == e["flags"]!.GetValue<int>(), $"flags {a.Flags}");
                var points = PdfFlat(e["points"]);
                var got = a.Points.ToArray();
                Expect(points.Length == got.Length && points.Zip(got).All(p => Math.Abs(p.First - p.Second) < 0.01), $"points [{string.Join(", ", got)}]");
                var ink = e["ink"]?.AsArray() ?? [];
                Expect(ink.Count == a.Ink.Count && ink.Select((stroke, k) => (stroke!, k)).All(s =>
                        s.Item1["verbs"]!.AsArray().Select(v => (string)v!).SequenceEqual(a.Ink[s.k].Verbs.ToArray().Select(v => v.ToString().ToLowerInvariant()))
                        && PdfFlat(s.Item1["points"]).Zip(a.Ink[s.k].Points.ToArray()).All(p => Math.Abs(p.First - p.Second) < 0.01)),
                    $"ink strokes {a.Ink.Count}");
                if (e["target"] is JsonObject target)
                {
                    if (target["uri"] is JsonValue uri)
                    {
                        Expect(a.Uri == (string)uri!, $"uri {a.Uri}");
                    }
                    else if (PdfTargetDifference(a.Destination, target, document, named) is { } difference)
                    {
                        problems.Add($"destination: {difference}");
                    }
                }
                else
                {
                    Expect(a.Uri is null && a.Destination is null, "a target where none is expected");
                }

                Expect(a.Popup == (string?)e["popup"], $"popup {a.Popup}");
                Expect(a.Parent == (string?)e["parent"], $"parent {a.Parent}");
                Expect(a.InReplyTo == (string?)e["inReplyTo"], $"in reply to {a.InReplyTo}");
                Expect(a.ReplyType == (string?)e["replyType"], $"reply type {a.ReplyType}");
                Expect(a.State == (string?)e["state"] && a.StateModel == (string?)e["stateModel"], $"state {a.State} / {a.StateModel}");
                Expect(a.Icon == (string?)e["icon"], $"icon {a.Icon}");
                Expect(e["borderWidth"] is JsonValue width ? PdfNear(a.BorderWidth, width.GetValue<double>()) : a.BorderWidth is null or 0f, $"border {a.BorderWidth}");
                Expect((e["lineEndings"]?.AsArray().Select(v => (string)v!) ?? []).SequenceEqual(a.LineEndings), $"line endings {string.Join(",", a.LineEndings)}");
                Expect((a.Appearance is not null) == e["hasAppearance"]!.GetValue<bool>(), $"appearance {(a.Appearance is null ? "none" : "read")}");
                if (e["attachment"] is JsonObject file)
                {
                    using var stream = a.Attachment?.Read();
                    string sha = stream is null ? "" : Convert.ToHexStringLower(SHA256.HashData(stream));
                    Expect(a.Attachment?.Name == (string)file["fileName"]! && a.Attachment.Size == file["size"]!.GetValue<long>() && sha == (string)file["sha256"]!,
                        $"attachment {a.Attachment?.Name} {a.Attachment?.Size} {sha}");
                }

                // A reply sits under the annotation it replies to.
                if (a.InReplyTo is { } parent && a.ReplyType == "reply")
                {
                    Expect(read.Any(x => x.Id == parent && x.Replies.Any(r => r.Id == id)), $"not among {parent}'s replies");
                }

                if (problems.Count > 0)
                {
                    failures.Add($"{where}: {string.Join("; ", problems)}");
                }
            }
        }

        Check(failures.Count == 0, $"{failures.Count} annotations read otherwise than their JSON says:\n      {string.Join("\n      ", failures)}");
        Check(checkedCount == expectedCount && checkedCount >= 40, $"{checkedCount} of {expectedCount} annotations checked: make the samples again (tools/documents/make_fixtures.py)");
        using var annotated = DocumentFormats.Open(PdfEditSample("annotations").Path);
        var note = PdfAnnotationsOf(annotated).Single(a => a.Id == "note-1");
        Check(note.Replies.Select(r => r.Id).SequenceEqual(["note-2", "note-3"]), $"note-1's thread: {string.Join(", ", note.Replies.Select(r => r.Id))}");
    }

    private static void PdfAnnotationsNavigation(Device device)
    {
        CpuOnly(device);
        var sample = PdfEditSample("navigation");
        using var document = DocumentFormats.Open(sample.Path);
        var named = sample.Expected["namedDestinations"]!.AsObject();
        var failures = new List<string>();
        void Outline(IReadOnlyList<DocumentOutline> got, JsonArray expected, string path)
        {
            if (got.Count != expected.Count)
            {
                failures.Add($"{path}: {got.Count} bookmarks, {expected.Count} expected");
                return;
            }

            for (int i = 0; i < got.Count; i++)
            {
                var e = expected[i]!.AsObject();
                var g = got[i];
                string where = $"{path}/{e["title"]}";
                if (g.Title != (string)e["title"]! || g.Open != e["open"]!.GetValue<bool>() || ((g.Style & DocumentOutlineStyle.Italic) != 0) != e["italic"]!.GetValue<bool>()
                    || ((g.Style & DocumentOutlineStyle.Bold) != 0) != e["bold"]!.GetValue<bool>())
                {
                    failures.Add($"{where}: read '{g.Title}', open {g.Open}, style {g.Style}");
                }

                if (PdfColourDifference(g.Colour, e["colour"], "colour") is { } colour)
                {
                    failures.Add($"{where}: {colour}");
                }

                if (e["target"] is JsonObject target)
                {
                    if (PdfTargetDifference(g.Destination, target, document, named) is { } difference)
                    {
                        failures.Add($"{where}: {difference}");
                    }
                }
                else if (g.Destination is not null || g.Uri is not null)
                {
                    failures.Add($"{where}: a target where none is expected");
                }

                Outline(g.Children, e["children"]!.AsArray(), where);
            }
        }

        Outline(document.Outline, sample.Expected["outline"]!.AsArray(), "outline");
        foreach (var (name, value) in named)
        {
            var destination = document.Destinations.FirstOrDefault(d => d.Name == name);
            var target = new JsonObject { ["page"] = value!["page"]!.GetValue<int>(), ["fit"] = (string)value["fit"]!, ["values"] = value["values"]!.DeepClone() };
            if (PdfTargetDifference(destination, target, document, null) is { } difference)
            {
                failures.Add($"named destination {name}: {difference}");
            }
        }

        Check(document.Destinations.Count == named.Count, $"{document.Destinations.Count} named destinations read, {named.Count} expected");
        var labels = sample.Expected["pageLabels"]!.AsArray().Select(l => (string)l!).ToList();
        var read = Enumerable.Range(0, document.PageCount).Select(document.PageLabel).ToList();
        Check(read.SequenceEqual(labels), $"page labels: {string.Join(" ", read)}; expected {string.Join(" ", labels)}");
        Check(failures.Count == 0, $"navigation:\n      {string.Join("\n      ", failures)}");

        // A file without navigation reads none.
        using var plain = DocumentFormats.Open(PdfEditSample("truetype-simple").Path);
        Check(plain.Outline.Count == 0 && plain.Destinations.Count == 0 && plain.PageLabel(0) is null, "a file without bookmarks, names or labels reads none");
    }

    private static void PdfAnnotationsTagged(Device device)
    {
        CpuOnly(device);
        var sample = PdfEditSample("tagged");
        var tagged = sample.Expected["tagged"]!.AsObject();
        using (var document = DocumentFormats.Open(sample.Path, new DocumentOptions { Layout = DocumentTextLayouts.Tagged }))
        {
            var page = document.ReadPage(0, DocumentParts.Text);
            Check(page.Text == (string)tagged["text"]!, $"the tagged text:\n{page.Text}\nexpected:\n{tagged["text"]}");
            var heading = page.Runs.Single(r => r.Text.Contains("Annual summary", StringComparison.Ordinal));
            var number = page.Runs.Single(r => r.Text.Contains("Page 1", StringComparison.Ordinal));
            Check(heading.Tag is { Type: "H1", IsArtifact: false } && number.Tag is { IsArtifact: true }, $"tags: the heading {heading.Tag}, the page number {number.Tag}");
            Check(page.Runs.Where(r => r.Text.StartsWith("The ", StringComparison.Ordinal)).All(r => r.Tag?.Type == "P"), "the columns' runs are tagged P");
        }

        using (var document = DocumentFormats.Open(sample.Path))
        {
            Check(new DocumentOptions().Layout == DocumentTextLayouts.Lines, "lines stays the default layout");
            var page = document.ReadPage(0, DocumentParts.Text);
            Check(page.Text.Contains("Page 1", StringComparison.Ordinal) && page.Text != (string)tagged["text"]!, "the lines layout reads by place, the artifact included");
        }

        Check(DocumentTextLayouts.Names.Contains(DocumentTextLayouts.Tagged) && DocumentTextLayouts.Origin(DocumentTextLayouts.Tagged) == Overrides.Library,
            $"tagged registered by the library: {DocumentTextLayouts.Describe()}");
        foreach (string name in new[] { "truetype-simple", "rtl", "mixed-direction" })
        {
            var plain = PdfEditSample(name);
            using var a = DocumentFormats.Open(plain.Path);
            using var b = DocumentFormats.Open(plain.Path, new DocumentOptions { Layout = DocumentTextLayouts.Tagged });
            for (int i = 0; i < a.PageCount; i++)
            {
                Check(a.ReadPage(i, DocumentParts.Text).Text == b.ReadPage(i, DocumentParts.Text).Text, $"{name}: page {i + 1} reads the same through tagged as through lines (no tags)");
            }
        }
    }

    // Annotations of every kind, links, bookmarks and labels, as the builder and the editor write them.
    private static List<DocumentAnnotation> PdfMadeAnnotations(int page, DocumentBox word, DocumentAttachment file)
    {
        var when = new DateTimeOffset(2026, 3, 2, 10, 15, 0, TimeSpan.Zero);
        var blue = DocumentColour.Rgb(0.1f, 0.3f, 0.8f);
        float[] Quad(DocumentBox b) => [b.X, b.Y, b.Right, b.Y, b.X, b.Bottom, b.Right, b.Bottom];
        DocumentAnnotation Made(string kind, string id, DocumentBox box) => new(kind, page, box)
        {
            Id = id, Contents = $"A {kind} written by the builder", Author = "Alex Morgan", Created = when, Modified = when.AddHours(1), Colour = blue,
        };

        return
        [
            Made(DocumentAnnotationKinds.Highlight, "made-highlight", word) with { Points = Quad(word), Colour = DocumentColour.Rgb(1f, 0.9f, 0.2f) },
            Made(DocumentAnnotationKinds.Underline, "made-underline", word) with { Points = Quad(word) },
            Made(DocumentAnnotationKinds.Squiggly, "made-squiggly", word) with { Points = Quad(word) },
            Made(DocumentAnnotationKinds.StrikeOut, "made-strike-out", word) with { Points = Quad(word) },
            Made(DocumentAnnotationKinds.Note, "made-note", new DocumentBox(520, 60, 20, 20)) with { Icon = "Comment", Popup = "made-popup" },
            Made(DocumentAnnotationKinds.Popup, "made-popup", new DocumentBox(380, 90, 180, 70)) with { Parent = "made-note", Open = true, Contents = null, Author = null, Colour = null },
            Made(DocumentAnnotationKinds.Note, "made-reply", new DocumentBox(520, 60, 20, 20)) with { InReplyTo = "made-note", ReplyType = "reply", Icon = "Comment" },
            Made(DocumentAnnotationKinds.FreeText, "made-free-text", new DocumentBox(60, 200, 200, 24)) with { Colour = null },
            Made(DocumentAnnotationKinds.Line, "made-line", new DocumentBox(296, 196, 130, 32)) with { Points = new float[] { 300, 224, 420, 200 }, BorderWidth = 2, LineEndings = ["None", "OpenArrow"] },
            Made(DocumentAnnotationKinds.Square, "made-square", new DocumentBox(60, 260, 100, 70)) with { BorderWidth = 2, InteriorColour = DocumentColour.Rgb(0.85f, 0.9f, 1f) },
            Made(DocumentAnnotationKinds.Circle, "made-circle", new DocumentBox(190, 260, 100, 70)) with { BorderWidth = 2 },
            Made(DocumentAnnotationKinds.Polygon, "made-polygon", new DocumentBox(328, 258, 94, 74)) with { Points = new float[] { 330, 270, 420, 280, 400, 330, 340, 320 }, BorderWidth = 1.5f },
            Made(DocumentAnnotationKinds.Polyline, "made-polyline", new DocumentBox(448, 258, 94, 74)) with { Points = new float[] { 450, 270, 540, 280, 520, 330, 460, 320 }, BorderWidth = 1.5f },
            Made(DocumentAnnotationKinds.Ink, "made-ink", new DocumentBox(76, 360, 88, 63)) with
            {
                BorderWidth = 2,
                Ink = [new DocumentPath([OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line], [80, 400, 100, 370, 120, 390]), new DocumentPath([OutlineVerb.Move, OutlineVerb.Line], [90, 415, 150, 415])],
            },
            Made(DocumentAnnotationKinds.Stamp, "made-stamp", new DocumentBox(220, 360, 160, 50)) with { Icon = "Approved" },
            Made(DocumentAnnotationKinds.Caret, "made-caret", new DocumentBox(205, 146, 10, 10)),
            Made(DocumentAnnotationKinds.FileAttachment, "made-attachment", new DocumentBox(420, 360, 16, 24)) with { Icon = "Paperclip", Attachment = file },
            Made(DocumentAnnotationKinds.Redaction, "made-redaction", new DocumentBox(60, 460, 120, 20)) with { InteriorColour = DocumentColour.Black, Colour = DocumentColour.Rgb(1, 0, 0) },
        ];
    }

    // Annotations written against those read back: kind, box, text, author, dates, colours, points, ink, thread, popup, attachment, look.
    private static void PdfMadeCheck(IReadOnlyList<DocumentAnnotation> written, IReadOnlyList<DocumentAnnotation> read, string what)
    {
        var failures = new List<string>();
        foreach (var w in written)
        {
            var r = read.FirstOrDefault(x => x.Id == w.Id && x.Page == w.Page);
            if (r is null)
            {
                failures.Add($"{w.Id}: not read");
                continue;
            }

            bool same = r.Kind == w.Kind && Math.Abs(r.Box.X - w.Box.X) < 0.01 && Math.Abs(r.Box.Y - w.Box.Y) < 0.01 && Math.Abs(r.Box.Width - w.Box.Width) < 0.01
                        && Math.Abs(r.Box.Height - w.Box.Height) < 0.01 && r.Contents == w.Contents && r.Author == w.Author && r.Created == w.Created && r.Modified == w.Modified
                        && r.Popup == w.Popup && r.Parent == w.Parent && r.InReplyTo == w.InReplyTo && r.Icon == w.Icon && r.Open == w.Open
                        && PdfNear(r.BorderWidth, w.BorderWidth) && r.LineEndings.SequenceEqual(w.LineEndings)
                        && r.Points.Length == w.Points.Length && r.Points.ToArray().Zip(w.Points.ToArray()).All(p => Math.Abs(p.First - p.Second) < 0.01)
                        && r.Ink.Count == w.Ink.Count && r.Ink.Zip(w.Ink).All(p => p.First.Points.ToArray().Zip(p.Second.Points.ToArray()).All(q => Math.Abs(q.First - q.Second) < 0.01))
                        && (r.Colour is null) == (w.Colour is null) && (r.InteriorColour is null) == (w.InteriorColour is null)
                        && (w.Attachment is null || (r.Attachment?.Name == w.Attachment.Name && r.Attachment.Size == w.Attachment.Size));
            bool look = w.Kind is DocumentAnnotationKinds.Popup or DocumentAnnotationKinds.FreeText ? r.Appearance is null : r.Appearance is not null;
            if (!same || !look)
            {
                failures.Add($"{w.Id}: read as {r.Kind} {r.Box} '{r.Contents}' by {r.Author}, points [{string.Join(", ", r.Points.ToArray())}], {r.Ink.Count} strokes, look {(r.Appearance is null ? "none" : "made")}");
            }
        }

        Check(failures.Count == 0, $"{what}: {failures.Count} annotations not read back as written:\n      {string.Join("\n      ", failures)}");
    }

    private static string PdfAnnotatedFolder() => Directory.CreateDirectory(RepositoryPath("test-output", "documents", "annotated")).FullName;

    private static void PdfAnnotationsWritten(Device device)
    {
        CpuOnly(device);
        var face = SubsetFace();
        byte[] csv = Encoding.ASCII.GetBytes("region,revenue\nNorth,48000\nSouth,38000\n");
        var file = new DocumentAttachment("figures.csv", () => new MemoryStream(csv, writable: false)) { MediaType = "text/csv", Size = csv.Length, Description = "Revenue by region" };
        string line = "Quarterly review: revenue rose in every region.";
        var glyphs = PdfShaped(face, line, "Latn", false, 60, 100, 12);
        // The word "revenue"'s box: its glyphs' origins and advances, the font's ascent and descent around the baseline.
        int at = line.IndexOf("revenue", StringComparison.Ordinal);
        var word = new DocumentBox(glyphs[at].X, 100 - 12 * 0.93f, glyphs[at + 6].X + glyphs[at + 6].Advance - glyphs[at].X, 12 * 1.17f);
        var made = PdfMadeAnnotations(0, word, file);
        var outline = new List<DocumentOutline>
        {
            new("Summary") { Destination = new DocumentDestination(0) { X = 60, Y = 80 }, Open = true, Colour = DocumentColour.Rgb(0, 0, 0.8f), Style = DocumentOutlineStyle.Bold,
                Children = [new DocumentOutline("Figures") { Destination = new DocumentDestination(1) { Fit = DocumentDestinationFit.Page } }, new DocumentOutline("Website") { Uri = "https://www.example.com/" }] },
            new("Appendix") { Destination = new DocumentDestination(1) { Fit = DocumentDestinationFit.Width, Y = 40 }, Style = DocumentOutlineStyle.Italic },
        };
        DocumentPageLabel[] labels = [new(0, DocumentPageNumbering.LowerRoman), new(1, DocumentPageNumbering.Decimal) { Prefix = "A-", FirstNumber = 3 }];
        byte[] bytes;
        using (var output = new MemoryStream())
        {
            using (var builder = DocumentWriters.Create(output, "pdf", PdfWriteDefaults))
            {
                var page = builder.AddPage(612, 792);
                page.Glyphs(face, 12, glyphs, line, DocumentTextMode.Fill, DocumentColour.Black);
                foreach (var annotation in made)
                {
                    page.AddAnnotation(annotation);
                }

                page.AddLink(new DocumentLink(new DocumentBox(60, 600, 200, 20), Uri: "https://www.example.com/report"));
                page.AddLink(new DocumentLink(new DocumentBox(320, 600, 200, 20), Destination: new DocumentDestination(1) { X = 40, Y = 30, Zoom = 1.5f }));
                var second = builder.AddPage(420, 300);
                second.Glyphs(face, 16, PdfShaped(face, "Figures", "Latn", false, 40, 50, 16), "Figures", DocumentTextMode.Fill, DocumentColour.Black);
                foreach (var item in outline)
                {
                    builder.AddOutline(item);
                }

                builder.SetPageLabels(labels);
                builder.Finish();
            }

            bytes = output.ToArray();
        }

        File.WriteAllBytes(Path.Combine(PdfAnnotatedFolder(), "created.pdf"), bytes);
        using (var document = DocumentFormats.Open(bytes, "created.pdf"))
        {
            File.WriteAllText(Path.Combine(PdfAnnotatedFolder(), "created.json"), PdfAnnotatedJson("created", document));
            var read = PdfAnnotationsOf(document, DocumentParts.Annotations | DocumentParts.Paths);
            PdfMadeCheck(made, read, "created by the builder");
            var links = read.Where(a => a.Kind == DocumentAnnotationKinds.Link).ToList();
            Check(links.Count == 2 && links[0].Uri == "https://www.example.com/report" && links[1].Destination is { Page: 1, Fit: DocumentDestinationFit.Position } d
                  && PdfNear(d.X, 40) && PdfNear(d.Y, 30) && PdfNear(d.Zoom, 1.5), $"links: {string.Join("; ", links.Select(l => $"{l.Uri} {l.Destination}"))}");
            Check(read.Single(a => a.Id == "made-note").Replies.Single().Id == "made-reply", "the reply under its note");
            Check(read.Single(a => a.Id == "made-highlight").Appearance!.Paths.Single().Blend == DocumentBlend.Multiply, "a highlight's look is multiplied over the text");
            var back = document.Outline;
            Check(back.Count == 2 && back[0] is { Title: "Summary", Open: true, Style: DocumentOutlineStyle.Bold, Destination.Page: 0 } && back[0].Children.Count == 2
                  && back[0].Children[0].Destination is { Page: 1, Fit: DocumentDestinationFit.Page } && back[0].Children[1].Uri == "https://www.example.com/"
                  && back[1] is { Title: "Appendix", Style: DocumentOutlineStyle.Italic, Destination: { Page: 1, Fit: DocumentDestinationFit.Width } a } && PdfNear(a.Y, 40)
                  && PdfNear(back[0].Destination!.X, 60) && PdfNear(back[0].Destination!.Y, 80) && PdfColourDifference(back[0].Colour, new JsonObject { ["rgb"] = new JsonArray(0, 0, 204) }, "") is null,
                $"bookmarks: {string.Join("; ", back.Select(b => $"{b.Title} {b.Destination} {b.Open} {b.Style}"))}");
            Check(document.PageLabel(0) == "i" && document.PageLabel(1) == "A-3", $"page labels {document.PageLabel(0)} {document.PageLabel(1)}");
        }

        // The editor: the annotations sample with two removed (a reply and a link) and two added, a bookmark added; both ways.
        var sample = PdfEditSample("annotations");
        foreach (var mode in new[] { DocumentSaveMode.Incremental, DocumentSaveMode.Rewrite })
        {
            using var source = DocumentFormats.Open(sample.Path);
            var before = PdfAnnotationsOf(source);
            var added = new List<DocumentAnnotation>
            {
                new(DocumentAnnotationKinds.Square, 0, new DocumentBox(400, 640, 120, 60)) { Id = "added-square", Colour = DocumentColour.Rgb(0.8f, 0.1f, 0.1f), BorderWidth = 3, Contents = "Added by the editor" },
                new(DocumentAnnotationKinds.Note, 0, new DocumentBox(560, 640, 20, 20)) { Id = "added-reply", InReplyTo = "note-1", ReplyType = "reply", Contents = "Seen", Icon = "Comment", Colour = DocumentColour.Rgb(0.2f, 0.6f, 1f) },
            };
            byte[] edited = PdfEditSave(source, editor =>
            {
                editor.RemoveAnnotation(before.Single(a => a.Id == "note-2"));
                editor.RemoveAnnotation(before.Single(a => a.Id == "link-1"));
                foreach (var annotation in added)
                {
                    editor.AddAnnotation(annotation);
                }

                editor.AddAnnotation(new DocumentAnnotation(DocumentAnnotationKinds.Link, 0, new DocumentBox(60, 700, 120, 20)) { Destination = new DocumentDestination(0) { Fit = DocumentDestinationFit.Page } });
                editor.AddOutline(new DocumentOutline("Review") { Destination = new DocumentDestination(0) { X = 60, Y = 92 } });
            }, mode);
            string name = mode == DocumentSaveMode.Incremental ? "annotations-edited-incremental" : "annotations-edited-rewrite";
            File.WriteAllBytes(Path.Combine(PdfAnnotatedFolder(), name + ".pdf"), edited);
            using var document = DocumentFormats.Open(edited, name);
            File.WriteAllText(Path.Combine(PdfAnnotatedFolder(), name + ".json"), PdfAnnotatedJson(name, document));
            var after = PdfAnnotationsOf(document);
            Check(!after.Any(a => a.Id is "note-2" or "link-1") && after.Count == before.Count - 2 + added.Count + 1, $"{name}: {after.Count} annotations ({string.Join(", ", after.Select(a => a.Id))})");
            PdfMadeCheck(added, after, name);
            Check(after.Single(a => a.Id == "note-1").Replies.Select(r => r.Id).SequenceEqual(["note-3", "added-reply"]), $"{name}: note-1's thread {string.Join(", ", after.Single(a => a.Id == "note-1").Replies.Select(r => r.Id))}");
            Check(document.Outline.Count == 1 && document.Outline[0] is { Title: "Review", Destination.Page: 0 }, $"{name}: the bookmark added");
            if (mode == DocumentSaveMode.Incremental)
            {
                byte[] original = File.ReadAllBytes(sample.Path);
                Check(edited.AsSpan(0, original.Length).SequenceEqual(original), $"{name}: the source's bytes are its prefix");
            }
        }
    }

    // A written file's text, page by page, for the checker.
    private static string PdfAnnotatedJson(string name, IDocument document)
    {
        var pages = new JsonArray();
        for (int i = 0; i < document.PageCount; i++)
        {
            var page = document.ReadPage(i, DocumentParts.Text);
            pages.Add(new JsonObject { ["index"] = i, ["width"] = page.Width, ["height"] = page.Height, ["text"] = page.Text });
        }

        return new JsonObject { ["schema"] = 1, ["name"] = name, ["file"] = name + ".pdf", ["pages"] = pages }.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }) + "\n";
    }

    private static string PdfRedactedFolder() => Directory.CreateDirectory(RepositoryPath("test-output", "documents", "redacted")).FullName;

    // Every decoded stream and every string of a file, for words that must be gone (as text and as the hex of their bytes).
    private static List<string> PdfLeftovers(byte[] file, IEnumerable<string> words)
    {
        using var document = PdfDocument.Open(new MemoryStream(file, writable: false), "leftovers", new DocumentOptions());
        var pdf = document.File;
        var haystacks = new List<(string Where, byte[] Bytes)> { ("the file's bytes", file) };
        foreach (var (number, _, _) in pdf.LiveObjects())
        {
            var value = pdf.Get(number);
            if (value.AsStream is { } stream)
            {
                try
                {
                    using var decoded = pdf.Decode(stream);
                    haystacks.Add(($"object {number}'s stream", decoded.Span.ToArray()));
                }
                catch (InvalidDataException)
                {
                    // An image encoding handed over as it is: its bytes are in the file's.
                }
            }

            haystacks.Add(($"object {number}", Encoding.Latin1.GetBytes(value.ToString())));
        }

        var found = new List<string>();
        foreach (string word in words)
        {
            byte[] text = Encoding.ASCII.GetBytes(word);
            byte[] hexUpper = Encoding.ASCII.GetBytes(Convert.ToHexString(text)), hexLower = Encoding.ASCII.GetBytes(Convert.ToHexStringLower(text));
            foreach (var (where, bytes) in haystacks)
            {
                if (bytes.AsSpan().IndexOf(text) >= 0 || bytes.AsSpan().IndexOf(hexUpper) >= 0 || bytes.AsSpan().IndexOf(hexLower) >= 0)
                {
                    found.Add($"'{word}' in {where}");
                }
            }
        }

        return found;
    }

    private static void PdfAnnotationsRedaction(Device device)
    {
        CpuOnly(device);
        var sample = PdfEditSample("redaction");
        var redaction = sample.Expected["redaction"]!.AsObject();
        var after = redaction["afterApplying"]!.AsObject();
        var removedWords = after["removedText"]!.AsArray().Select(w => (string)w!).ToList();
        byte[] source = File.ReadAllBytes(sample.Path);
        Check(PdfLeftovers(source, removedWords).Count > 0, "the sample holds the words before redaction (the search finds them)");

        // The sample's own marks, applied.
        DocumentRedactionResult result = null!;
        byte[] redacted;
        IReadOnlyList<DocumentPath> paths;
        using (var document = DocumentFormats.Open(sample.Path))
        {
            paths = document.ReadPage(0, DocumentParts.Paths).Paths;
            var areas = DocumentRedactions.Marked(document.ReadPage(0, DocumentParts.Annotations));
            Check(areas.Count == 3 && areas.All(a => a.Fill is { } f && f.Red == 0 && f.Green == 0 && f.Blue == 0), $"{areas.Count} areas marked, filled black");
            redacted = PdfEditSave(document, editor =>
            {
                result = editor.Redact(0, areas);
                ExpectRefusal<InvalidOperationException>(() => editor.Save(DocumentSaveMode.Incremental), "rewrite", "a redacted document saved incrementally");
            }, DocumentSaveMode.Rewrite);
        }

        var image = after["images"]![0]!.AsObject();
        int glyphs = removedWords.Sum(w => w.Length);
        Check(result is { Areas: 3, PathsRemoved: 1, PathsClipped: 1, Images: 1, Annotations: 3 } && result.Glyphs == glyphs && result.Characters == glyphs
              && result.Pixels == image["changedPixels"]!.GetValue<long>(),
            $"what was removed: {result}; expected {glyphs} glyphs, 1 path removed and 1 clipped, 1 image of {image["changedPixels"]} pixels, 3 annotations");
        File.WriteAllBytes(Path.Combine(PdfRedactedFolder(), "redaction.pdf"), redacted);
        var leftovers = PdfLeftovers(redacted, removedWords);
        Check(leftovers.Count == 0, $"removed words still in the written file: {string.Join("; ", leftovers)}");
        using (var document = DocumentFormats.Open(redacted, "redaction-redacted.pdf"))
        {
            File.WriteAllText(Path.Combine(PdfRedactedFolder(), "redaction.json"), PdfAnnotatedJson("redaction", document));
            var page = document.ReadPage(0, DocumentParts.All);
            var lines = page.Lines.Select(l => l.Text.Trim()).ToList();
            var expectedLines = after["lines"]!.AsArray().Select(l => ((string)l!).Trim()).ToList();
            Check(lines.SequenceEqual(expectedLines), $"the lines: {string.Join(" | ", lines)}; expected {string.Join(" | ", expectedLines)}");
            Check(page.Annotations.Count == 0, $"{page.Annotations.Count} annotations left (the marks applied are removed)");
            // The paths: the one inside gone, the crossing one under a clip that leaves out the area, the clear one as it was.
            DocumentColour ColourOf(int order) => paths.Single(p => p.Order == order).FillColour;
            bool Has(DocumentColour colour, out DocumentPath? found)
            {
                found = page.Paths.FirstOrDefault(p => Math.Abs(p.FillColour.Red - colour.Red) < 1e-3 && Math.Abs(p.FillColour.Green - colour.Green) < 1e-3 && Math.Abs(p.FillColour.Blue - colour.Blue) < 1e-3);
                return found is not null;
            }

            int removedOrder = after["paths"]!["removed"]![0]!.GetValue<int>(), clippedOrder = after["paths"]!["clipped"]![0]!.GetValue<int>(), keptOrder = after["paths"]!["kept"]![0]!.GetValue<int>();
            Check(!Has(ColourOf(removedOrder), out _), "the path inside the area is gone");
            bool hasClipped = Has(ColourOf(clippedOrder), out var clipped), hasKept = Has(ColourOf(keptOrder), out var kept);
            Check(hasClipped && clipped!.Clip >= 0 && hasKept && kept!.Clip < 0, $"the crossing path clipped ({clipped?.Clip}), the clear one as it was ({kept?.Clip})");
            var area = redaction["areas"]![2]!["box"]!.AsArray().Select(v => (float)v!.GetValue<double>()).ToArray();
            var clip = page.Clips[clipped!.Clip];
            Check(clip.Fill == DocumentFill.EvenOdd && PdfInside(clip, area[0] + area[2] / 2, area[1] + area[3] / 2) == false && PdfInside(clip, 5, 5) == true,
                "the crossing path's clip leaves out the area");
            // The image: its pixels as the sample's .redacted.png, exactly.
            var picture = page.Images.Single().Decode();
            byte[] interleaved = new byte[picture.Width * picture.Height * 3];
            var values = picture.Pixels;
            for (int p = 0; p < picture.Width * picture.Height; p++)
            {
                for (int c = 0; c < 3; c++)
                {
                    interleaved[3 * p + c] = (byte)Math.Round(values[c * picture.Width * picture.Height + p] * 255);
                }
            }

            Check(Convert.ToHexStringLower(SHA256.HashData(interleaved)) == (string)image["sha256"]!, "the image's pixels are the expected redacted ones");
            // Drawn as redaction-applied draws in pdfium.
            var reference = ImageCodecs.Get("png").Decode(File.ReadAllBytes(Path.Combine(Path.GetDirectoryName(sample.Path)!, "redaction-applied.p1.png")));
            var drawing = document.DrawPage(0, new DocumentDrawOptions { Dpi = 100, Fonts = PdfDrawFonts() });
            var (share, mean) = PdfDrawDifference(drawing.Image, reference, 32);
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"    redacted page against pdfium's redaction-applied: {share * 100:F2}% off by more than 32 levels (mean {mean:F2})"));
            Check(share <= 0.08, string.Create(CultureInfo.InvariantCulture, $"the redacted page draws as redaction-applied: {share * 100:F2}% off"));
        }

        // A box and a text match, through the library's finders.
        using (var document = DocumentFormats.Open(sample.Path))
        {
            var page = document.ReadPage(0, DocumentParts.Text | DocumentParts.Glyphs);
            var found = DocumentRedactions.Find(page, "48,000");
            Check(found.Count == 1, $"{found.Count} places of '48,000'");
            var box = new DocumentRedaction(new DocumentBox(25, 40, 100, 14)) { Fill = DocumentColour.White };   // the first line's first words
            byte[] bytes = PdfEditSave(document, editor => editor.Redact(0, [.. found, box]), DocumentSaveMode.Rewrite);
            File.WriteAllBytes(Path.Combine(PdfRedactedFolder(), "redaction-text-and-box.pdf"), bytes);
            using var back = DocumentFormats.Open(bytes, "redaction-text-and-box.pdf");
            File.WriteAllText(Path.Combine(PdfRedactedFolder(), "redaction-text-and-box.json"), PdfAnnotatedJson("redaction-text-and-box", back));
            string text = back.ReadPage(0, DocumentParts.Text).Text;
            Check(!text.Contains("48,000", StringComparison.Ordinal) && text.Contains("Robin Park", StringComparison.Ordinal) && !text.Contains("Account manager", StringComparison.Ordinal)
                  && PdfLeftovers(bytes, ["48,000"]).Count == 0, $"text and box redacted: {text}");
        }
    }

    // Whether a point is inside a closed path of lines by its fill rule (the even-odd count, or the nonzero winding).
    private static bool PdfInside(DocumentPath path, float x, float y)
    {
        var verbs = path.Verbs.Span;
        var p = path.Points.Span;
        int crossings = 0, winding = 0, at = 0;
        float sx = 0, sy = 0, cx = 0, cy = 0;
        void Edge(float x0, float y0, float x1, float y1)
        {
            if ((y0 <= y) != (y1 <= y) && x < x0 + (y - y0) / (y1 - y0) * (x1 - x0))
            {
                crossings++;
                winding += y1 > y0 ? 1 : -1;
            }
        }

        foreach (var verb in verbs)
        {
            if (verb == OutlineVerb.Move)
            {
                if (at > 0)
                {
                    Edge(cx, cy, sx, sy);
                }

                (sx, sy, cx, cy) = (p[at], p[at + 1], p[at], p[at + 1]);
                at += 2;
            }
            else if (verb == OutlineVerb.Line)
            {
                Edge(cx, cy, p[at], p[at + 1]);
                (cx, cy) = (p[at], p[at + 1]);
                at += 2;
            }
            else if (verb == OutlineVerb.Close)
            {
                Edge(cx, cy, sx, sy);
                (cx, cy) = (sx, sy);
            }
        }

        return path.Fill == DocumentFill.EvenOdd ? crossings % 2 == 1 : winding != 0;
    }

    private static void PdfAnnotationsDrawn(Device device)
    {
        CpuOnly(device);
        string fonts = PdfDrawFonts(), drawn = PdfDrawnFolder();
        var png = ImageCodecs.Get("png");
        var failures = new List<string>();
        foreach (string name in new[] { "annotations", "annotations-no-appearance", "navigation" })
        {
            var sample = PdfEditSample(name);
            using var document = DocumentFormats.Open(sample.Path);
            var reference = png.Decode(File.ReadAllBytes(Path.Combine(Path.GetDirectoryName(sample.Path)!, $"{name}.p1.png")));
            var options = new DocumentDrawOptions { Dpi = 100, Fonts = fonts, Parts = DocumentParts.Text | DocumentParts.Images | DocumentParts.Paths | DocumentParts.Annotations };
            var drawing = document.DrawPage(0, options);
            var plain = document.DrawPage(0, options with { Parts = DocumentParts.Text | DocumentParts.Images | DocumentParts.Paths });
            File.WriteAllBytes(Path.Combine(drawn, $"{name}.p1.png"), ImageEncoders.Get("png").Encode(drawing.Image));
            var (share, mean) = PdfDrawDifference(drawing.Image, reference, 32);
            var (without, _) = PdfDrawDifference(plain.Image, reference, 32);
            double tolerance = PdfDrawTolerance(sample, 0);
            string score = string.Create(CultureInfo.InvariantCulture,
                $"{name}: {share * 100:F2}% off by more than 32 levels (mean {mean:F2}) with annotations, {without * 100:F2}% without them, within {tolerance * 100:F0}%");
            Console.WriteLine($"    {score}");
            if (share > tolerance || (name != "navigation" && !(share < without)))
            {
                failures.Add(score);
            }

            if (drawing.Issues.Any(i => i.Kind == DocumentIssue.NotDrawn) && name == "annotations")
            {
                failures.Add($"{name}: {string.Join("; ", drawing.Issues.Select(i => i.Detail))}");
            }
        }

        Check(failures.Count == 0, $"annotations drawn past their tolerance:\n      {string.Join("\n      ", failures)}");

        // Hidden ones and popups are not drawn; one with neither an appearance nor a look is counted.
        var hidden = new DocumentAnnotation(DocumentAnnotationKinds.Square, 0, new DocumentBox(10, 10, 50, 50)) { Colour = DocumentColour.Black, BorderWidth = 4, Flags = DocumentAnnotationFlags.Hidden };
        var popup = new DocumentAnnotation(DocumentAnnotationKinds.Popup, 0, new DocumentBox(10, 10, 50, 50)) { Colour = DocumentColour.Black };
        var text = new DocumentAnnotation(DocumentAnnotationKinds.FreeText, 0, new DocumentBox(10, 10, 50, 50)) { Contents = "A note" };
        var page = new DocumentPage(0, 80, 80) { Annotations = [hidden, popup, text] };
        var result = DocumentPageDrawers.Draw(page, new DocumentDrawOptions { Dpi = 72, Parts = DocumentParts.All });
        Check(result.Image.Pixels.All(v => v > 0.99f) && result.Issues.Count(i => i.Kind == DocumentIssue.NotDrawn) == 1, "hidden and popup annotations not drawn; free text with no appearance counted");
        var square = DocumentPageDrawers.Draw(new DocumentPage(0, 80, 80) { Annotations = [hidden with { Flags = DocumentAnnotationFlags.Print }] }, new DocumentDrawOptions { Dpi = 72, Parts = DocumentParts.All });
        Check(square.Image.Pixels.Count(v => v < 0.5f) > 100, "a square with no appearance drawn from its properties");
    }

    private static void PdfAnnotationsTimings(Device device)
    {
        CpuOnly(device);
        var samples = PdfSamples().Where(s => s.Phases.Contains("D10") && s.Readable).Select(s => (s.Name, Bytes: File.ReadAllBytes(s.Path))).ToList();
        int ReadAll(DocumentParts parts)
        {
            int count = 0;
            foreach (var (name, bytes) in samples)
            {
                using var document = DocumentFormats.Open(bytes, name);
                count += PdfAnnotationsOf(document, parts).Count;
            }

            return count;
        }

        (TimeSpan Cold, TimeSpan Warm, long Bytes) Measure(Func<int> work)
        {
            long start = Stopwatch.GetTimestamp();
            work();
            var cold = Stopwatch.GetElapsedTime(start);
            var warm = new List<TimeSpan>();
            long allocated = 0;
            for (int i = 0; i < 5; i++)
            {
                long bytes = GC.GetAllocatedBytesForCurrentThread();
                start = Stopwatch.GetTimestamp();
                work();
                warm.Add(Stopwatch.GetElapsedTime(start));
                allocated = GC.GetAllocatedBytesForCurrentThread() - bytes;
            }

            return (cold, warm.Order().ElementAt(2), allocated);
        }

        int count = ReadAll(DocumentParts.Annotations);
        var annotations = Measure(() => ReadAll(DocumentParts.Annotations));
        var appearances = Measure(() => ReadAll(DocumentParts.All));
        var redaction = PdfEditSample("redaction");
        byte[] source = File.ReadAllBytes(redaction.Path);
        var redact = Measure(() =>
        {
            using var document = DocumentFormats.Open(source, "redaction.pdf");
            var areas = DocumentRedactions.Marked(document.ReadPage(0, DocumentParts.Annotations));
            return PdfEditSave(document, editor => editor.Redact(0, areas), DocumentSaveMode.Rewrite).Length;
        });
        foreach (var (what, m) in new[] { ($"{count} annotations of {samples.Count} D10 samples read (opened each pass)", annotations), ("the same with their appearances' parts", appearances), ("redaction's page redacted by its marks and saved", redact) })
        {
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"    {what}: cold {m.Cold.TotalMilliseconds:F1} ms, warm median {m.Warm.TotalMilliseconds:F2} ms, {m.Bytes / 1024.0:F0} KiB allocated a pass"));
        }

        Check(count > 50, $"{count} annotations read");
    }
}
