// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Data.Pdf;
using Idrak.Imaging;

// Plan 28, phase D9: forms. Fields read (every kind, inherited values, widgets, XFA data), filled (SetField: incremental
// and rewrite), added (the builder's and the editor's AddField), flattened (Flatten), each widget's appearance made, read
// back and drawn through D7's drawer. Completed (2026-11-19): turned and bordered widgets, rich text, fields held inside
// other objects, a /DA naming the face a text needs, XFA templates, and the reading's memory. Checked on the D9 samples tools/documents/make_fixtures.py writes into the ignored
// test-output/documents/samples; every file written goes to test-output/documents/forms (NAME.pdf; NAME.json with the
// pages' text for tools/documents/check_written.py; NAME.fields.json with each field's value, for a reference reader's
// form reading).
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] DocumentPdfFormsGroup =
    [
        ("documents pdf: forms: every D9 sample's fields read as its JSON says (name, kind, value and values, choices, flags, length, alignment, font and size, tooltip, caption; each widget's page, box, state and on state; values, flags and the default appearance inherited through the tree; an XFA form's data read as fields beside its AcroForm twins, its layout an issue of every page), each widget's appearance read as the page's annotation with the text its JSON says, and the fields read alike from several threads", PdfFormsRead),
        ("documents pdf: forms: forms-empty filled with forms-filled's values (incremental: the source its prefix; and a rewrite) reads back with those values and states, each widget's appearance showing its value as forms-filled's does, and draws through the coverage drawer, the widgets in place, within D7's tolerance of pdfium's drawing of the same values (forms-need-appearances, whose text and choice appearances pdfium makes from them); forms-filled drawn with its widgets within it of pdfium's; the same edits write the same bytes; values a field does not take refused", PdfFormsFill),
        ("documents pdf: forms: fields filled and added in Latin, Arabic (right to left, joined) and Cyrillic text are laid out through the text setter in a face given (embedded as a subset): their values and their appearances' text read back in logical order, the drawing with the widgets the same as the drawing of the form flattened; a text no font shows refused naming the options", PdfFormsScripts),
        ("documents pdf: forms: the builder adds every kind of field (text: one line, several lines, comb, password, a length, aligned; check box; radio group; list of several choices; drop-down list; push button; signature; a field under a dotted parent; one field on two pages) with their appearances, and they read back as added; a widget on a page written already, a name taken and a page never added refused; the editor adds a field to a document that has none", PdfFormsBuild),
        ("documents pdf: forms: flattening (every field, or some by name) draws each widget's appearance into its page (made when the form has none) and leaves no field or widget: forms-filled flattened draws as its flattened twin within D7's tolerance, a form asking the reader for its appearances flattened shows its values as page text, the inherited form's field on two pages flattened on both, the XFA form flattened drops its XFA", PdfFormsFlatten),
        ("documents pdf: forms: turned and bordered widgets (/MK /R, /BS): forms-styled filled (incremental and rewrite) reads back turned and bordered as its JSON says and draws within D9's tolerance of pdfium's own appearances for it, each turned box's text running its way (nearer pdfium's than pdfium's turned a half, a quarter turn's ink taller than wide); flattened it draws alike; the builder turns and borders fields as asked and refuses a turn of 45 degrees", PdfFormsStyled),
        ("documents pdf: forms: rich text: forms-rich's /RV read as styled runs over /DS; SetRichText writes /RV, its plain /V and the rich text flag, and an appearance in each run's font, size and colour (incremental and rewrite); a plain value set on a rich text field is its one run; the builder adds a rich text field; runs refused as they should be", PdfFormsRich),
        ("documents pdf: forms: fields held inside other objects (forms-held: object streams, a field written in its parent's /Kids and one in /Fields) filled in place (incremental: a cross-reference stream after a stream; and a rewrite): moved to objects of their own, their holders and widgets naming them; flattened too", PdfFormsHeld),
        ("documents pdf: forms: a text the field's /DA font lacks, set in a face: /DA names that face's font, which /DR lists under a name it did not hold (forms-faces: /F1 taken, /F1_1 given), drawn as pdfium draws the producer's appearance; a text the font has leaves /DA as it was", PdfFormsFaces),
        ("documents pdf: forms: XFA templates read (IDocument.FormTemplate): sections, fields with captions, kinds, choices, flags, bindings, values and the fields holding them as each XFA sample's JSON says; scripts listed as issues, never run; none for a form without one", PdfFormsTemplates),
        ("documents pdf: forms: memory: reading forms-filled's form (12 fields, 14 widgets) on a document kept open allocates about what it returns, warm (no closures, no record copied a node, each page's frame read once)", PdfFormsMemory),
        ("documents pdf: forms: timings: reading every field of a form, filling one field (incremental), flattening a page: cold and warm apart, with the bytes allocated", PdfFormsTimings),
    ];

    private static string PdfFormsFolder() => Directory.CreateDirectory(RepositoryPath("test-output", "documents", "forms")).FullName;

    // Saves a written form: NAME.pdf, NAME.json with its pages' text as the library reads them (for the checker), and
    // NAME.fields.json with each field's value.
    private static void PdfFormsSave(string name, byte[] bytes, bool text = true)
    {
        string folder = PdfFormsFolder();
        File.WriteAllBytes(Path.Combine(folder, name + ".pdf"), bytes);
        using var document = PdfEditOpen(bytes, name);
        var pages = new JsonArray();
        for (int i = 0; i < document.PageCount; i++)
        {
            var page = document.ReadPage(i, DocumentParts.Text);
            pages.Add(new JsonObject { ["index"] = i, ["width"] = page.Width, ["height"] = page.Height, ["text"] = page.Text });
        }

        string json = Path.Combine(folder, name + ".json");
        if (text)
        {
            File.WriteAllText(json, new JsonObject { ["schema"] = 1, ["name"] = name, ["file"] = name + ".pdf", ["pages"] = pages }.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }) + "\n");
        }
        else
        {
            File.Delete(json);
        }

        var fields = new JsonObject();
        foreach (var field in document.Fields)
        {
            fields[field.Name] = new JsonObject
            {
                ["kind"] = field.Kind.ToString(), ["value"] = field.Value, ["values"] = new JsonArray([.. field.Values.Select(v => (JsonNode?)v)]),
            };
        }

        File.WriteAllText(Path.Combine(folder, name + ".fields.json"), fields.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }) + "\n");
    }

    private static DocumentFieldKind PdfFormsKind(string kind) => kind switch
    {
        "text" => DocumentFieldKind.Text,
        "checkbox" => DocumentFieldKind.CheckBox,
        "radio" => DocumentFieldKind.Radio,
        "list" => DocumentFieldKind.List,
        "combo" => DocumentFieldKind.Combo,
        "pushbutton" => DocumentFieldKind.PushButton,
        _ => DocumentFieldKind.Signature,
    };

    // The flags a sample's JSON names, as the contract's (the kind's own bits, radio, push button and combo, and the ones
    // the contract does not carry left out).
    private static DocumentFieldFlags PdfFormsFlags(JsonNode? names, string kind)
    {
        var flags = DocumentFieldFlags.None;
        foreach (string name in (names as JsonArray ?? []).Select(n => (string)n!))
        {
            flags |= name switch
            {
                "readOnly" => DocumentFieldFlags.ReadOnly,
                "required" => DocumentFieldFlags.Required,
                "noExport" => DocumentFieldFlags.NoExport,
                "multiline" => DocumentFieldFlags.Multiline,
                "password" => DocumentFieldFlags.Password,
                "noToggleToOff" => DocumentFieldFlags.NoToggleToOff,
                "edit" => DocumentFieldFlags.Editable,
                "sort" => DocumentFieldFlags.Sort,
                "multiSelect" => DocumentFieldFlags.MultiSelect,
                "doNotSpellCheck" => DocumentFieldFlags.DoNotSpellCheck,
                "doNotScroll" => DocumentFieldFlags.DoNotScroll,
                "comb" => DocumentFieldFlags.Comb,
                "richText" => DocumentFieldFlags.RichText,
                "radiosInUnison" => kind == "text" ? DocumentFieldFlags.RichText : DocumentFieldFlags.RadiosInUnison,
                _ => DocumentFieldFlags.None,
            };
        }

        return flags;
    }

    // A field as read against its JSON: everything the contract carries.
    private static void PdfFormsCompare(string what, DocumentField field, JsonObject e)
    {
        string kind = (string)e["kind"]!;
        Check(field.Name == (string)e["name"]! && field.Kind == PdfFormsKind(kind), $"{what}: '{field.Name}' a {field.Kind}, expected '{e["name"]}' a {kind}");
        string at = $"{what}: '{field.Name}'";
        Check(field.Value == (string?)e["value"], $"{at}: the value '{field.Value}', expected '{e["value"]}'");
        List<string> values = e["values"] is JsonArray many ? [.. many.Select(v => (string)v!)] : e["value"] is JsonValue one ? [(string)one!] : [];
        Check(field.Values.SequenceEqual(values), $"{at}: the values [{string.Join(", ", field.Values)}], expected [{string.Join(", ", values)}]");
        var options = (e["options"] as JsonArray ?? []).Select(o => new DocumentFieldOption((string)o!["display"]!, (string)o["export"]!)).ToList();
        Check(field.Options.SequenceEqual(options), $"{at}: the options [{string.Join(", ", field.Options)}], expected [{string.Join(", ", options)}]");
        Check(field.Flags == PdfFormsFlags(e["flags"], kind), $"{at}: the flags {field.Flags}, expected {PdfFormsFlags(e["flags"], kind)} ({e["flags"]?.ToJsonString()})");
        Check(field.MaxLength == (int?)e["maxLength"], $"{at}: the most characters {field.MaxLength}, expected {e["maxLength"]}");
        var alignment = (string?)e["alignment"] switch { "centre" => DocumentFieldAlignment.Centre, "right" => DocumentFieldAlignment.Right, _ => DocumentFieldAlignment.Left };
        Check(field.Alignment == alignment, $"{at}: aligned {field.Alignment}, expected {alignment}");
        Check(field.Font == (string?)e["font"] && (e["fontSize"] is null || MathF.Abs(field.FontSize - (float)e["fontSize"]!) < 1e-4f),
            $"{at}: the font {field.Font} at {field.FontSize}, expected {e["font"]} at {e["fontSize"]}");
        Check(field.Tooltip == (string?)e["tooltip"] && field.DefaultValue == (string?)e["defaultValue"], $"{at}: the tooltip '{field.Tooltip}' and default '{field.DefaultValue}'");
        if (kind == "pushbutton")
        {
            Check(field.Caption == (string?)e["caption"], $"{at}: the caption '{field.Caption}', expected '{e["caption"]}'");
        }

        var widgets = e["widgets"]!.AsArray();
        Check(field.Widgets.Count == widgets.Count, $"{at}: {field.Widgets.Count} widgets, expected {widgets.Count}");
        for (int k = 0; k < widgets.Count; k++)
        {
            var w = widgets[k]!.AsObject();
            var read = field.Widgets[k];
            float[] box = [.. w["box"]!.AsArray().Select(v => (float)v!)];
            Check(read.Page == (int)w["page"]! && PdfNear(read.Box, new DocumentBox(box[0], box[1], box[2], box[3]), 0.01f),
                $"{at}: widget {k} on page {read.Page} at {read.Box}, expected page {w["page"]} at ({string.Join(", ", box)})");
            bool button = kind is "checkbox" or "radio";
            Check(read.State == (button ? (string?)w["state"] : null) && read.OnState == (button ? (string?)w["onState"] : null),
                $"{at}: widget {k} shows '{read.State}' (on '{read.OnState}'), expected '{w["state"]}' (on '{w["onState"]}')");
            Check(read.Rotation == ((int?)w["rotation"] ?? 0), $"{at}: widget {k} turned {read.Rotation}, expected {w["rotation"] ?? 0}");
        }

        var border = e["border"] as JsonObject;
        var style = (string?)border?["style"] switch
        {
            "dashed" => DocumentFieldBorderStyle.Dashed,
            "beveled" => DocumentFieldBorderStyle.Beveled,
            "inset" => DocumentFieldBorderStyle.Inset,
            "underline" => DocumentFieldBorderStyle.Underline,
            _ => DocumentFieldBorderStyle.Solid,
        };
        float[] dashes = [.. (border?["dashes"] as JsonArray ?? []).Select(d => (float)d!)];
        Check(field.BorderStyle == style && field.BorderWidth == (float?)border?["width"] && field.Dashes.SequenceEqual(dashes),
            $"{at}: its border {field.BorderStyle} {field.BorderWidth} [{string.Join(" ", field.Dashes)}], expected {border?.ToJsonString() ?? "none"}");
        var rich = e["richText"] as JsonArray ?? [];
        Check(PdfFormsStyles(field.RichText) == PdfFormsStyles(rich.Select(PdfFormsRun).ToList()) && field.RichText.Count == rich.Count,
            $"{at}: its rich text {PdfFormsDescribe(field.RichText)}, expected {PdfFormsDescribe([.. rich.Select(PdfFormsRun)])}");
    }

    // A run of rich text as a sample's JSON gives it.
    private static DocumentFieldRun PdfFormsRun(JsonNode? node)
    {
        var o = node!.AsObject();
        var colour = o["colour"] as JsonArray;
        return new DocumentFieldRun((string)o["text"]!)
        {
            Bold = (bool)o["bold"]!, Italic = (bool)o["italic"]!, Underline = (bool)o["underline"]!, StrikeOut = (bool)o["strikeOut"]!, Size = (float?)o["size"],
            Colour = colour is null ? null : DocumentColour.Rgb((float)colour[0]!, (float)colour[1]!, (float)colour[2]!), Family = (string?)o["family"],
        };
    }

    // Each character of rich text (line feeds aside) with its style, to compare runs however a text is cut into them.
    private static string PdfFormsStyles(IReadOnlyList<DocumentFieldRun> runs)
    {
        var text = new StringBuilder();
        foreach (var run in runs)
        {
            string colour = run.Colour is { } c ? string.Join(",", Enumerable.Range(0, c.Count).Select(i => c[i].ToString("F2", CultureInfo.InvariantCulture))) : "-";
            string size = run.Size is { } z ? z.ToString("F2", CultureInfo.InvariantCulture) : "-";
            foreach (char ch in run.Text.Where(ch => ch != '\n'))
            {
                text.Append(ch).Append('|').Append(run.Bold).Append(run.Italic).Append(run.Underline).Append(run.StrikeOut).Append('|').Append(size).Append('|').Append(colour)
                    .Append('|').Append(run.Family).Append('\n');
            }
        }

        return text.ToString();
    }

    private static string PdfFormsDescribe(IReadOnlyList<DocumentFieldRun> runs) =>
        "[" + string.Join(", ", runs.Select(r => $"\"{r.Text.Replace("\n", "/", StringComparison.Ordinal)}\"{(r.Bold ? " bold" : "")}{(r.Italic ? " italic" : "")}{(r.Underline ? " underlined" : "")}{(r.StrikeOut ? " struck" : "")} {r.Size} {r.Colour} {r.Family}")) + "]";

    // The text an appearance shows, whitespace dropped (a comb's cells read as words apart; lines as the layout joins them).
    private static string PdfFormsSquash(string? text) => text is null ? "" : string.Concat(text.Where(c => !char.IsWhiteSpace(c)));

    // Each widget annotation of the document's pages against the fields' JSON: one a widget, on its page, at its box, with
    // an appearance when the JSON says it has one, showing its text.
    private static void PdfFormsAppearances(string what, IDocument document, JsonArray fields, List<string> failures)
    {
        var expected = new List<(int Page, float[] Box, bool Has, string? Text, string Kind, string Name)>();
        foreach (var f in fields.Select(f => f!.AsObject()))
        {
            foreach (var w in f["widgets"]!.AsArray().Select(w => w!.AsObject()))
            {
                expected.Add(((int)w["page"]!, [.. w["box"]!.AsArray().Select(v => (float)v!)], (bool)w["hasAppearance"]!, (string?)w["appearanceText"], (string)f["kind"]!, (string)f["name"]!));
            }
        }

        int seen = 0;
        for (int p = 0; p < document.PageCount; p++)
        {
            var page = document.ReadPage(p, DocumentParts.Annotations);
            foreach (var annotation in page.Annotations)
            {
                seen++;
                var match = expected.FindIndex(x => x.Page == p && PdfNear(annotation.Box, new DocumentBox(x.Box[0], x.Box[1], x.Box[2], x.Box[3]), 0.01f));
                if (annotation.Kind != DocumentAnnotationKinds.Widget || match < 0)
                {
                    failures.Add($"{what}: page {p}: a {annotation.Kind} at {annotation.Box} no field's widget is at");
                    continue;
                }

                var x = expected[match];
                if (x.Has != annotation.Appearance is not null)
                {
                    failures.Add($"{what}: '{x.Name}' on page {p}: an appearance {annotation.Appearance is not null}, expected {x.Has}");
                }
                else if (x.Has && x.Text is { } text && x.Kind is "text" or "combo" or "pushbutton" && PdfFormsSquash(annotation.Appearance!.Text) != PdfFormsSquash(text))
                {
                    failures.Add($"{what}: '{x.Name}' on page {p} shows \"{annotation.Appearance.Text}\", expected \"{text}\"");
                }
            }
        }

        if (seen != expected.Count)
        {
            failures.Add($"{what}: {seen} widget annotations, expected {expected.Count}");
        }
    }

    // A page drawn with its widgets (or not) at `dpi`, the standard fonts from the test's folder.
    private static ImageData PdfFormsDraw(IDocument document, int page, double dpi, bool annotations = true) =>
        PdfDrawn(document, page, new DocumentDrawOptions
        {
            Dpi = (float)dpi, Fonts = PdfDrawFonts(),
            Parts = DocumentParts.Text | DocumentParts.Images | DocumentParts.Paths | (annotations ? DocumentParts.Annotations : DocumentParts.None),
        }).Image;

    private static void PdfFormsRead(Device device)
    {
        CpuOnly(device);
        var samples = PdfSamples().Where(s => s.Phases.Contains("D9") && s.Readable).ToList();
        Check(samples.Count >= 6, $"{samples.Count} D9 samples; the generator writes six (make them again: tools/documents/make_fixtures.py)");
        var failures = new List<string>();
        int fields = 0, widgets = 0;
        foreach (var sample in samples)
        {
            using var document = new PdfFormat().Open(new MemoryStream(File.ReadAllBytes(sample.Path)), sample.Path, new DocumentOptions());
            var expected = sample.Expected["fields"]!.AsArray();
            var read = document.Fields;
            var xfa = sample.Expected["form"]?["xfa"] as JsonObject;
            int extras = 0;
            if (xfa is not null)
            {
                // The XFA data's values: each held by its AcroForm twin (its name without the [0] indices), or a field of its own.
                var held = expected.Select(f => PdfForms.Unindexed((string)f!["name"]!)).ToHashSet();
                foreach (var (path, value) in xfa["data"]!.AsObject())
                {
                    var twin = read.FirstOrDefault(f => PdfForms.Unindexed(f.Name) == path);
                    Check(twin is not null && twin.Value == (string?)value, $"{sample.Name}: the XFA data '{path}' = '{value}' read as '{twin?.Name}' = '{twin?.Value}'");
                    if (!held.Contains(path))
                    {
                        extras++;
                        Check(twin!.Kind == DocumentFieldKind.Text && twin.Widgets.Count == 0, $"{sample.Name}: the XFA data '{path}' a text field of its own, with no widget");
                    }
                }

                var issues = document.ReadPage(0, DocumentParts.Text).Issues;
                Check(issues.Any(i => i.Kind == DocumentIssue.NotSupported && i.Detail.Contains("XFA", StringComparison.Ordinal)),
                    $"{sample.Name}: an XFA form's page says its layout is not drawn: {string.Join("; ", issues.Select(i => i.Detail))}");
            }
            else
            {
                Check(document.ReadPage(0, DocumentParts.Text).Issues.All(i => !i.Detail.Contains("XFA", StringComparison.Ordinal)), $"{sample.Name}: no XFA issue on a form without XFA");
            }

            Check(read.Count == expected.Count + extras, $"{sample.Name}: {read.Count} fields ({string.Join(", ", read.Select(f => f.Name))}), expected {expected.Count} and {extras} of the XFA data");
            for (int i = 0; i < expected.Count; i++)
            {
                PdfFormsCompare(sample.Name, read[i], expected[i]!.AsObject());
                fields++;
                widgets += read[i].Widgets.Count;
            }

            PdfFormsAppearances(sample.Name, document, expected, failures);

            // The same from several threads, each on a document of its own and on the one shared.
            var shared = PdfEditOpen(File.ReadAllBytes(sample.Path), sample.Name);
            string Describe(IReadOnlyList<DocumentField> f) => string.Join("|", f.Select(x => $"{x.Name}={x.Value}[{string.Join(",", x.Values)}]{x.Widgets.Count}"));
            string once = Describe(read);
            var described = new string[8];
            Parallel.For(0, described.Length, k => described[k] = Describe(k % 2 == 0 ? shared.Fields : PdfEditOpen(File.ReadAllBytes(sample.Path), sample.Name).Fields));
            Check(described.All(d => d == once), $"{sample.Name}: the fields read alike from several threads");
            shared.Dispose();
        }

        Check(failures.Count == 0, $"{failures.Count} appearances not as their JSON says:\n      {string.Join("\n      ", failures)}");
        Console.WriteLine($"    {samples.Count} samples, {fields} fields, {widgets} widgets read as their JSON says");
        Check(fields >= 40, $"{fields} fields compared; the D9 samples hold more");
    }

    // forms-filled's values set on a form of the same fields.
    private static void PdfFormsFillFrom(IDocumentEditor editor, JsonArray fields)
    {
        foreach (var f in fields.Select(f => f!.AsObject()))
        {
            string kind = (string)f["kind"]!, name = (string)f["name"]!;
            if (kind is "pushbutton" or "signature")
            {
                continue;
            }

            if (f["values"] is JsonArray many)
            {
                editor.SetField(name, [.. many.Select(v => (string)v!)]);
            }
            else if ((string?)f["value"] is { } value)
            {
                editor.SetField(name, value);
            }
        }
    }

    private static void PdfFormsFill(Device device)
    {
        CpuOnly(device);
        var empty = PdfEditSample("forms-empty");
        var filled = PdfEditSample("forms-filled");
        var fields = filled.Expected["fields"]!.AsArray();
        byte[] source = File.ReadAllBytes(empty.Path);
        double dpi = (double?)filled.Expected["reference"]?["dpi"] ?? 100;
        var reference = new PngCodec().Decode(File.ReadAllBytes(Path.Combine(Path.GetDirectoryName(filled.Path)!, "forms-filled.p1.png")));
        // The filled form against pdfium's drawing of the same values with appearances pdfium makes itself: this sample's
        // producer draws each text box's background inside its /Tx section and states no /MK colours, so a new appearance
        // (pdfium's, or any writer's by 12.7.4.3) shows the box without it, as forms-need-appearances does.
        var made = new PngCodec().Decode(File.ReadAllBytes(Path.Combine(Path.GetDirectoryName(filled.Path)!, "forms-need-appearances.p1.png")));
        var failures = new List<string>();

        // The library's drawing of forms-filled with its widgets against pdfium's: what the filled forms are held to.
        using (var original = PdfEditOpen(File.ReadAllBytes(filled.Path), "forms-filled"))
        {
            var (share, mean) = PdfDrawDifference(PdfFormsDraw(original, 0, dpi), reference, 32);
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"    forms-filled drawn with its widgets: {share * 100:F2}% off pdfium's by more than 32 levels (mean {mean:F2})"));
            Check(share <= 0.08, string.Create(CultureInfo.InvariantCulture, $"forms-filled drawn with its widgets: {share * 100:F2}% off pdfium's, past 8%"));
            var without = PdfFormsDraw(original, 0, dpi, annotations: false);
            Check(PdfDrawDifference(without, reference, 32).Share > share, "the widgets drawn change the page toward pdfium's (drawn without them it is further off)");
        }

        foreach (var mode in new[] { DocumentSaveMode.Incremental, DocumentSaveMode.Rewrite })
        {
            string label = mode == DocumentSaveMode.Incremental ? "incremental" : "rewrite";
            using var document = PdfEditOpen(source, "forms-empty");
            byte[] bytes = PdfEditSave(document, e => PdfFormsFillFrom(e, fields), mode);
            byte[] again = PdfEditSave(document, e => PdfFormsFillFrom(e, fields), mode);
            Check(bytes.AsSpan().SequenceEqual(again), $"forms-empty filled ({label}): the same edits write the same bytes");
            if (mode == DocumentSaveMode.Incremental)
            {
                Check(bytes.Length > source.Length && bytes.AsSpan(0, source.Length).SequenceEqual(source), "forms-empty filled incrementally keeps its bytes as the prefix");
            }

            using var back = PdfEditOpen(bytes, $"forms-empty filled ({label})");
            var read = back.Fields;
            Check(read.Count == fields.Count, $"forms-empty filled ({label}): {read.Count} fields");
            for (int i = 0; i < fields.Count; i++)
            {
                PdfFormsCompare($"forms-empty filled ({label})", read[i], fields[i]!.AsObject());
            }

            PdfFormsAppearances($"forms-empty filled ({label})", back, fields, failures);
            var (share, mean) = PdfDrawDifference(PdfFormsDraw(back, 0, dpi), made, 32);
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"    forms-empty filled ({label}) drawn with its widgets: {share * 100:F2}% off pdfium's drawing of the same values (forms-need-appearances) by more than 32 levels (mean {mean:F2})"));
            if (share > 0.08)
            {
                failures.Add(string.Create(CultureInfo.InvariantCulture, $"forms-empty filled ({label}) drawn: {share * 100:F2}% off pdfium's drawing of the same values, past 8%"));
            }

            PdfFormsSave($"forms-empty-filled-{label}", bytes);
        }

        // Values a field does not take.
        using (var document = PdfEditOpen(source, "forms-empty"))
        using (var editor = new PdfWriter().Edit(document, new MemoryStream(), PdfWriteDefaults))
        {
            ExpectRefusal<ArgumentException>(() => editor.SetField("postcode", "AB12345"), "at most 6", "a comb's seventh character");
            ExpectRefusal<ArgumentException>(() => editor.SetField("newsletter", "Maybe"), "'Yes'", "a check box's state it lacks");
            ExpectRefusal<ArgumentException>(() => editor.SetField("delivery", "Courier"), "'Express'", "a radio button it lacks");
            ExpectRefusal<ArgumentException>(() => editor.SetField("colour", "Red", "Blue"), "one choice", "two choices of a list of one");
            ExpectRefusal<ArgumentException>(() => editor.SetField("country", "Brazil"), "'Japan'", "a drop-down list's choice it lacks");
            ExpectRefusal<ArgumentException>(() => editor.SetField("reset", "x"), "push button", "a push button's value");
            ExpectRefusal<ArgumentException>(() => editor.SetField("signature", "x"), "D12", "a signature field's value");
            ExpectRefusal<ArgumentException>(() => editor.SetField("name", "a", "b"), "one value", "two values of a text box");
            editor.SetField("extras", "Gift wrap", "card");                    // a choice by its text or its export value
            editor.SetField("country", "");                                    // cleared
        }

        Check(failures.Count == 0, $"{failures.Count} differences:\n      {string.Join("\n      ", failures)}");
    }

    private static void PdfFormsScripts(Device device)
    {
        CpuOnly(device);
        var face = SubsetFace();
        var empty = PdfEditSample("forms-empty");
        byte[] source = File.ReadAllBytes(empty.Path);
        const string Arabic = "تقرير الربع الأول", Cyrillic = "Привет мир", Latin = "Quarterly report";
        var options = PdfWriteDefaults with { Forms = new DocumentFormOptions { Faces = [face] } };
        void Fill(IDocumentEditor e)
        {
            e.SetField("name", Arabic);
            e.SetField("comments", Cyrillic + "\n" + Latin);
            e.SetField("country", "Japan");
        }

        using var document = PdfEditOpen(source, "forms-empty");
        byte[] bytes = PdfEditSave(document, Fill, DocumentSaveMode.Incremental, options);
        using var back = PdfEditOpen(bytes, "forms in three scripts");
        var name = back.Fields.Single(f => f.Name == "name");
        var comments = back.Fields.Single(f => f.Name == "comments");
        Check(name.Value == Arabic && comments.Value == Cyrillic + "\n" + Latin, $"the values read back: '{name.Value}', '{comments.Value}'");
        var page = back.ReadPage(0, DocumentParts.Annotations | DocumentParts.Glyphs);
        var shown = page.Annotations.Select(a => a.Appearance?.Text ?? "").ToList();
        Check(shown.Contains(Arabic), $"the Arabic value's appearance reads back in logical order: {string.Join(" | ", shown)}");
        Check(shown.Contains(Cyrillic + "\n" + Latin), $"the two lines' appearance reads back: {string.Join(" | ", shown)}");
        Check(shown.Contains("Japan"), "the drop-down list in the field's own font (Helvetica has every character)");
        var faces = page.Annotations.Where(a => a.Appearance is { } shownBy && (shownBy.Text == Arabic || shownBy.Text.StartsWith(Cyrillic, StringComparison.Ordinal)))
            .SelectMany(a => a.Appearance!.Runs).ToList();
        Check(faces.Count >= 3 && faces.All(r => r.Face is not null && r.Font is { } f && f.Contains("DejaVu", StringComparison.Ordinal)),
            $"the texts no field font shows set in the face given, embedded: {string.Join(", ", faces.Select(r => $"{r.Text}: {r.Font}"))}");
        PdfFormsSave("forms-three-scripts", bytes);

        // The drawing with the widgets is the drawing of the same form flattened.
        byte[] flat = PdfEditSave(back, e => e.Flatten(), DocumentSaveMode.Rewrite, options);
        using var flattened = PdfEditOpen(flat, "forms in three scripts flattened");
        Check(flattened.Fields.Count == 0 && flattened.ReadPage(0, DocumentParts.Annotations).Annotations.Count == 0, "flattened: no field and no widget");
        string text = flattened.ReadPage(0, DocumentParts.Text).Text;
        Check(text.Contains(Arabic, StringComparison.Ordinal) && text.Contains(Cyrillic, StringComparison.Ordinal) && text.Contains(Latin, StringComparison.Ordinal),
            $"flattened: the values are the page's text, in logical order: {text.Replace('\n', '/')}");
        var withWidgets = PdfFormsDraw(back, 0, 100);
        var drawnFlat = PdfFormsDraw(flattened, 0, 100, annotations: false);
        var (share, mean) = PdfDrawDifference(withWidgets, drawnFlat, 32);
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"    three scripts: the form drawn with its widgets against it flattened: {share * 100:F3}% off by more than 32 levels (mean {mean:F3})"));
        Check(share <= 0.001, string.Create(CultureInfo.InvariantCulture, $"the form drawn with its widgets is drawn flattened alike: {share * 100:F3}% off"));
        int box = (int)(name.Widgets[0].Box.Y * 100 / 72), left = (int)(name.Widgets[0].Box.X * 100 / 72);
        Check(Enumerable.Range(left, 300).Any(x => withWidgets.Pixels[(box + 10) * withWidgets.Width + x] < 0.5f), "the Arabic value is drawn in its box");
        PdfFormsSave("forms-three-scripts-flattened", flat, text: false);

        // No font shows the text: refused naming the options.
        using (var plain = PdfEditOpen(source, "forms-empty"))
        using (var editor = new PdfWriter().Edit(plain, new MemoryStream(), PdfWriteDefaults with { Forms = new DocumentFormOptions { Fonts = null } }))
        {
            ExpectRefusal<InvalidOperationException>(() => editor.SetField("name", Arabic), "DocumentWriteOptions.Forms.Faces", "a text no font shows");
        }

        // The builder: a field in each script.
        byte[] built = PdfWrite(b =>
        {
            b.AddPage(300, 200);
            b.AddField(new DocumentField("greeting", DocumentFieldKind.Text) { Value = Cyrillic, Widgets = [new DocumentWidget(0, new DocumentBox(20, 20, 200, 24))] });
            b.AddField(new DocumentField("title", DocumentFieldKind.Text) { Value = Arabic, Alignment = DocumentFieldAlignment.Right, Widgets = [new DocumentWidget(0, new DocumentBox(20, 60, 200, 24))] });
            b.AddField(new DocumentField("subject", DocumentFieldKind.Text) { Value = Latin, Widgets = [new DocumentWidget(0, new DocumentBox(20, 100, 200, 24))] });
        }, options);
        using var builtBack = PdfEditOpen(built, "built in three scripts");
        Check(string.Join("|", builtBack.Fields.Select(f => f.Value)) == $"{Cyrillic}|{Arabic}|{Latin}", $"the built fields' values: {string.Join("|", builtBack.Fields.Select(f => f.Value))}");
        var builtShown = builtBack.ReadPage(0, DocumentParts.Annotations).Annotations.Select(a => a.Appearance?.Text).ToList();
        Check(builtShown.SequenceEqual([Cyrillic, Arabic, Latin]), $"the built fields' appearances: {string.Join(" | ", builtShown)}");
        PdfFormsSave("forms-built-scripts", built);
    }

    private static void PdfFormsBuild(Device device)
    {
        CpuOnly(device);
        var face = SubsetFace();
        var options = PdfWriteDefaults with { Forms = new DocumentFormOptions { Faces = [face] } };
        DocumentWidget W(int page, float x, float y, float w, float h, string? on = null) => new(page, new DocumentBox(x, y, w, h)) { OnState = on };
        var added = new List<DocumentField>
        {
            new("name", DocumentFieldKind.Text) { Value = "Sam Rivera", Flags = DocumentFieldFlags.Required, Tooltip = "Full name", FontSize = 11, Font = "Helvetica", Border = DocumentColour.Grey(0.5f), Widgets = [W(0, 150, 60, 250, 20)] },
            new("notes", DocumentFieldKind.Text) { Value = "The parcel arrives on Monday.\nPlease ring twice.", Flags = DocumentFieldFlags.Multiline, Widgets = [W(0, 150, 90, 250, 48)] },
            new("code", DocumentFieldKind.Text) { Value = "XY42Z", Flags = DocumentFieldFlags.Comb, MaxLength = 5, FontSize = 12, Widgets = [W(0, 150, 150, 100, 20)] },
            new("secret", DocumentFieldKind.Text) { Value = "7351", Flags = DocumentFieldFlags.Password, Widgets = [W(0, 150, 180, 80, 20)] },
            new("total", DocumentFieldKind.Text) { Value = "1,250.00", Alignment = DocumentFieldAlignment.Right, MaxLength = 12, DefaultValue = "0.00", Background = DocumentColour.Rgb(0.95f, 0.95f, 0.8f), Widgets = [W(0, 150, 210, 120, 20)] },
            new("updates", DocumentFieldKind.CheckBox) { Value = "Yes", Widgets = [W(0, 150, 240, 16, 16)] },
            new("terms", DocumentFieldKind.CheckBox) { Mark = DocumentFieldMark.Cross, Widgets = [W(0, 200, 240, 16, 16, "Agreed")] },
            new("size", DocumentFieldKind.Radio) { Value = "Medium", Flags = DocumentFieldFlags.NoToggleToOff, Options = [new("Small", "Small"), new("Medium", "Medium"), new("Large", "Large")],
                Widgets = [W(0, 150, 270, 16, 16), W(0, 200, 270, 16, 16), W(0, 250, 270, 16, 16)] },
            new("toppings", DocumentFieldKind.List) { Values = ["olives", "basil"], Flags = DocumentFieldFlags.MultiSelect, Options = [new("Olives", "olives"), new("Peppers", "peppers"), new("Basil", "basil")], Widgets = [W(0, 150, 300, 120, 50)] },
            new("city", DocumentFieldKind.Combo) { Value = "Lisbon", Options = [new("Lisbon", "Lisbon"), new("Oslo", "Oslo")], Widgets = [W(0, 150, 360, 120, 20)] },
            new("send", DocumentFieldKind.PushButton) { Caption = "Send", Background = DocumentColour.Rgb(0.2f, 0.4f, 0.7f), TextColour = DocumentColour.White, Widgets = [W(0, 150, 390, 80, 24)] },
            new("approval", DocumentFieldKind.Signature) { Border = DocumentColour.Grey(0.4f), Widgets = [W(0, 150, 430, 200, 50)] },
            new("address.street", DocumentFieldKind.Text) { Value = "4 Harbour Road", Widgets = [W(0, 150, 500, 200, 20)] },
            new("address.city", DocumentFieldKind.Text) { Value = "Portsmouth", Widgets = [W(0, 150, 530, 200, 20)] },
            new("contact", DocumentFieldKind.Text) { Value = "office@example.com", Widgets = [W(0, 150, 560, 200, 20), W(1, 150, 60, 200, 20)] },
        };
        byte[] bytes = PdfWrite(b =>
        {
            b.AddPage(450, 600);
            foreach (var field in added.Take(14))
            {
                b.AddField(field);
            }

            b.AddField(added[14]);                                             // a widget on the page being built and one on the next
            b.AddPage(450, 300);
            ExpectRefusal<ArgumentException>(() => b.AddField(new DocumentField("late", DocumentFieldKind.Text) { Widgets = [W(0, 10, 10, 50, 20)] }), "written already", "a widget on a page written already");
            ExpectRefusal<ArgumentException>(() => b.AddField(new DocumentField("contact", DocumentFieldKind.Text) { Widgets = [W(1, 10, 10, 50, 20)] }), "there already", "a name taken");
            ExpectRefusal<ArgumentException>(() => b.AddField(new DocumentField("address.street.number", DocumentFieldKind.Text) { Widgets = [W(1, 10, 10, 50, 20)] }), "with a value", "a field under a field with a value");
        }, options);

        using var back = PdfEditOpen(bytes, "a form built");
        var read = back.Fields;
        Check(read.Count == added.Count, $"{read.Count} fields read back ({string.Join(", ", read.Select(f => f.Name))}), {added.Count} added");
        foreach (var field in added)
        {
            var r = read.SingleOrDefault(f => f.Name == field.Name);
            string at = $"the built field '{field.Name}'";
            Check(r is not null && r.Kind == field.Kind, $"{at}: read as {r?.Kind}");
            IReadOnlyList<string> values = field.Values.Count > 0 ? field.Values : field.Value is { } v ? [v] : [];
            Check(r!.Values.SequenceEqual(values) && r.Value == (field.Kind == DocumentFieldKind.List && (field.Flags & DocumentFieldFlags.MultiSelect) != 0 ? null : values.FirstOrDefault()),
                $"{at}: values [{string.Join(", ", r.Values)}] ('{r.Value}'), added [{string.Join(", ", values)}]");
            Check(r.Flags == field.Flags && r.MaxLength == (field.Kind == DocumentFieldKind.Text ? field.MaxLength : null) && r.Alignment == field.Alignment && r.Tooltip == field.Tooltip
                  && r.DefaultValue == field.DefaultValue, $"{at}: flags {r.Flags}, length {r.MaxLength}, aligned {r.Alignment}, tooltip {r.Tooltip}, default {r.DefaultValue}");
            if (field.Kind is DocumentFieldKind.List or DocumentFieldKind.Combo)
            {
                Check(r.Options.SequenceEqual(field.Options), $"{at}: options [{string.Join(", ", r.Options)}]");
            }

            Check(r.Widgets.Count == field.Widgets.Count && r.Widgets.Zip(field.Widgets).All(p => p.First.Page == p.Second.Page && PdfNear(p.First.Box, p.Second.Box, 0.01f)),
                $"{at}: widgets {string.Join(", ", r.Widgets.Select(w => $"{w.Page} {w.Box}"))}");
            if (field.Kind == DocumentFieldKind.PushButton)
            {
                Check(r.Caption == field.Caption && r.Background == field.Background, $"{at}: caption '{r.Caption}', background {r.Background}");
            }

            if (field.Kind is DocumentFieldKind.CheckBox or DocumentFieldKind.Radio)
            {
                Check(r.Mark == field.Mark || (field.Kind == DocumentFieldKind.Radio && r.Mark == DocumentFieldMark.Check), $"{at}: its mark {r.Mark}");
                string? on = field.Value;
                Check(r.Widgets.All(w => w.OnState is not null && w.State == (w.OnState == on ? on : "Off")), $"{at}: states {string.Join(", ", r.Widgets.Select(w => $"{w.State}/{w.OnState}"))}");
            }
        }

        // Each widget's appearance shows the value (two pages: the field on both).
        var shown = new List<string>();
        for (int p = 0; p < back.PageCount; p++)
        {
            shown.AddRange(back.ReadPage(p, DocumentParts.Annotations).Annotations.Select(a => a.Appearance?.Text ?? ""));
        }

        foreach (string text in new[] { "Sam Rivera", "The parcel arrives on Monday.\nPlease ring twice.", "1,250.00", "****", "Lisbon", "Send", "4 Harbour Road", "Portsmouth" })
        {
            Check(shown.Contains(text), $"an appearance shows \"{text}\": {string.Join(" | ", shown)}");
        }

        Check(shown.Count(t => t == "office@example.com") == 2 && shown.Any(t => PdfFormsSquash(t) == "XY42Z") && shown.Any(t => t.Contains("Olives", StringComparison.Ordinal)),
            "the field on two pages shown on both, the comb's cells, the list's choices");
        var drawn = PdfFormsDraw(back, 0, 100);
        float ink = 0;
        foreach (var w in read.Single(f => f.Name == "updates").Widgets)
        {
            int x0 = (int)(w.Box.X * 100 / 72), y0 = (int)(w.Box.Y * 100 / 72), x1 = (int)(w.Box.Right * 100 / 72), y1 = (int)(w.Box.Bottom * 100 / 72);
            for (int y = y0; y < y1; y++)
            {
                for (int x = x0; x < x1; x++)
                {
                    ink += 1 - drawn.Pixels[y * drawn.Width + x];
                }
            }
        }

        Check(ink > 10, $"the check box's mark is drawn ({ink:F1} of ink)");
        PdfFormsSave("forms-built", bytes);

        // A value set over a field's own appearance keeps what it draws outside its /Tx section (12.7.4.3): here the
        // background the builder drew before it, once.
        byte[] retotalled = PdfEditSave(back, e => e.SetField("total", "99.00"), DocumentSaveMode.Incremental, options);
        using (var again = PdfEditOpen(retotalled, "a form built, its total set"))
        {
            static string Content(PdfDocument d)
            {
                var widget = d.Form.Find("total")!.Widgets[0];
                using var decoded = d.File.Decode(PdfForms.AppearanceStream(d.File, widget.Dictionary, null)!);
                return Encoding.Latin1.GetString(decoded.Span);
            }

            string before = Content(back), after = Content(again);
            int section = before.IndexOf("/Tx BMC", StringComparison.Ordinal);
            string frame = section > 0 ? before[..section] : "";
            Check(frame.Length > 0 && after.StartsWith(frame, StringComparison.Ordinal) && after.IndexOf(frame, 1, StringComparison.Ordinal) < 0,
                $"the field's background kept once before its new text section: \"{frame.Trim()}\" / \"{after[..Math.Min(after.Length, 120)].Trim()}\"");
            var total = again.Fields.Single(f => f.Name == "total");
            Check(total.Value == "99.00" && again.ReadPage(0, DocumentParts.Annotations).Annotations.Any(a => PdfNear(a.Box, total.Widgets[0].Box, 0.01f) && a.Appearance?.Text == "99.00"),
                "the new value read back and shown");
        }

        // A page never added: refused at Finish.
        using (var builder = new PdfWriter().Create(new MemoryStream(), options))
        {
            builder.AddPage(200, 200);
            builder.AddField(new DocumentField("later", DocumentFieldKind.Text) { Value = "x", Widgets = [W(3, 10, 10, 50, 20)] });
            ExpectRefusal<InvalidOperationException>(builder.Finish, "page 3", "a widget on a page never added");
        }

        // The editor adds a field to a document with no form.
        var merge = PdfEditSample("merge-a");
        using var source = PdfEditOpen(File.ReadAllBytes(merge.Path), "merge-a");
        foreach (var mode in new[] { DocumentSaveMode.Incremental, DocumentSaveMode.Rewrite })
        {
            byte[] edited = PdfEditSave(source, e =>
            {
                e.AddField(new DocumentField("reviewer", DocumentFieldKind.Text) { Value = "Jamie Chen", Widgets = [W(1, 40, 40, 200, 22)] });
                e.AddField(new DocumentField("review.done", DocumentFieldKind.CheckBox) { Value = "Yes", Widgets = [W(0, 40, 80, 16, 16)] });
            }, mode, options);
            using var editedBack = PdfEditOpen(edited, $"merge-a with a form ({mode})");
            var fields = editedBack.Fields;
            Check(fields.Count == 2 && fields[0].Name == "reviewer" && fields[0].Value == "Jamie Chen" && fields[0].Widgets[0].Page == 1 && fields[1].Name == "review.done"
                  && fields[1].Value == "Yes", $"merge-a with a form ({mode}): {string.Join(", ", fields.Select(f => $"{f.Name}={f.Value} p{f.Widgets.FirstOrDefault().Page}"))}");
            Check(editedBack.ReadPage(1, DocumentParts.Annotations).Annotations.Single().Appearance?.Text == "Jamie Chen", $"merge-a with a form ({mode}): the field's appearance");
            Check(PdfEditPagesRead(editedBack).Select(p => p.Text).SequenceEqual(PdfEditPagesRead(source).Select(p => p.Text)), $"merge-a with a form ({mode}): its pages' text kept");
            PdfFormsSave($"merge-a-with-form-{(mode == DocumentSaveMode.Incremental ? "incremental" : "rewrite")}", edited);
        }

        // Fields nested past the limit fail naming it.
        byte[] deep = PdfWrite(b =>
        {
            b.AddPage(100, 100);
            b.AddField(new DocumentField(string.Join(".", Enumerable.Range(0, 20).Select(k => "n" + k.ToString(CultureInfo.InvariantCulture))), DocumentFieldKind.Text) { Value = "deep", Widgets = [W(0, 5, 5, 50, 20)] });
        });
        using (var tight = PdfDocument.Open(new MemoryStream(deep, writable: false), "deep form", new DocumentOptions { MaxDepth = 8 }))
        {
            ExpectRefusal<InvalidDataException>(() => _ = tight.Fields, "MaxDepth", "fields nested past the limit");
        }

        using var loose = PdfEditOpen(deep, "deep form");
        Check(loose.Fields.Single().Value == "deep", "the deep field within the default limit");
    }

    private static void PdfFormsFlatten(Device device)
    {
        CpuOnly(device);
        var filled = PdfEditSample("forms-filled");
        var twin = PdfEditSample("forms-flattened");
        double dpi = (double?)filled.Expected["reference"]?["dpi"] ?? 100;
        var pdfiumTwin = new PngCodec().Decode(File.ReadAllBytes(Path.Combine(Path.GetDirectoryName(twin.Path)!, "forms-flattened.p1.png")));
        using var twinDocument = PdfEditOpen(File.ReadAllBytes(twin.Path), "forms-flattened");
        Check(twinDocument.Fields.Count == 0, "the flattened twin has no fields");
        var twinDrawn = PdfFormsDraw(twinDocument, 0, dpi, annotations: false);
        var failures = new List<string>();
        using var source = PdfEditOpen(File.ReadAllBytes(filled.Path), "forms-filled");
        foreach (var mode in new[] { DocumentSaveMode.Incremental, DocumentSaveMode.Rewrite })
        {
            string label = mode == DocumentSaveMode.Incremental ? "incremental" : "rewrite";
            byte[] bytes = PdfEditSave(source, e => e.Flatten(), mode);
            using var flat = PdfEditOpen(bytes, $"forms-filled flattened ({label})");
            Check(flat.Fields.Count == 0 && flat.ReadPage(0, DocumentParts.Annotations).Annotations.Count == 0, $"forms-filled flattened ({label}): no field, no widget");
            var drawn = PdfFormsDraw(flat, 0, dpi);
            var (share, mean) = PdfDrawDifference(drawn, twinDrawn, 32);
            var (pdfium, pdfiumMean) = PdfDrawDifference(drawn, pdfiumTwin, 32);
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"    forms-filled flattened ({label}): {share * 100:F3}% off the twin drawn alike (mean {mean:F3}), {pdfium * 100:F2}% off pdfium's twin (mean {pdfiumMean:F2})"));
            if (share > 0.02 || pdfium > 0.08)
            {
                failures.Add(string.Create(CultureInfo.InvariantCulture, $"forms-filled flattened ({label}): {share * 100:F2}% off its twin (2%), {pdfium * 100:F2}% off pdfium's (8%)"));
            }

            PdfFormsSave($"forms-filled-flattened-{label}", bytes, text: false);
        }

        // Some fields by name: the others kept; a parent's name takes its children.
        byte[] some = PdfEditSave(source, e => e.Flatten("name", "delivery"), DocumentSaveMode.Incremental);
        using (var partly = PdfEditOpen(some, "forms-filled, two fields flattened"))
        {
            var names = partly.Fields.Select(f => f.Name).ToList();
            Check(!names.Contains("name") && !names.Contains("delivery") && names.Count == filled.Expected["fields"]!.AsArray().Count - 2,
                $"two fields flattened, the others kept: {string.Join(", ", names)}");
            Check(partly.ReadPage(0, DocumentParts.Annotations).Annotations.Count == 14 - 4, "their four widgets left out of the page");
            Check(partly.ReadPage(0, DocumentParts.Text).Text.Contains("Alex Morgan", StringComparison.Ordinal), "the flattened field's value is page text");
        }

        // A form asking the reader for its appearances: made from the values, then drawn into the page.
        var need = PdfEditSample("forms-need-appearances");
        using (var needs = PdfEditOpen(File.ReadAllBytes(need.Path), "forms-need-appearances"))
        {
            byte[] bytes = PdfEditSave(needs, e => e.Flatten(), DocumentSaveMode.Rewrite);
            using var flat = PdfEditOpen(bytes, "forms-need-appearances flattened");
            string text = flat.ReadPage(0, DocumentParts.Text).Text;
            foreach (string value in new[] { "Alex Morgan", "Please deliver after 10 am.", "Call the front desk on arrival.", "****", "PO-2024-17", "Japan", "Reset form", "Green" })
            {
                Check(text.Contains(value, StringComparison.Ordinal), $"forms-need-appearances flattened shows \"{value}\": {text.Replace('\n', '/')}");
            }

            Check(PdfFormsSquash(text).Contains("AB1234", StringComparison.Ordinal), "the comb's characters, each in its cell");
            Check(flat.Fields.Count == 0, "forms-need-appearances flattened: no field");
            PdfFormsSave("forms-need-appearances-flattened", bytes, text: false);
        }

        // A field on two pages flattened on both; the XFA form flattened drops its XFA.
        var inherited = PdfEditSample("forms-inherited");
        using (var tree = PdfEditOpen(File.ReadAllBytes(inherited.Path), "forms-inherited"))
        {
            byte[] bytes = PdfEditSave(tree, e => e.Flatten("contact", "address"), DocumentSaveMode.Incremental);
            using var flat = PdfEditOpen(bytes, "forms-inherited flattened");
            Check(flat.Fields.Select(f => f.Name).SequenceEqual(["terms.agree"]), $"the address fields and contact flattened, terms kept: {string.Join(", ", flat.Fields.Select(f => f.Name))}");
            Check(flat.ReadPage(0, DocumentParts.Text).Text.Contains("office@example.com", StringComparison.Ordinal)
                  && flat.ReadPage(1, DocumentParts.Text).Text.Contains("office@example.com", StringComparison.Ordinal), "contact drawn into both its pages");
            Check(flat.ReadPage(0, DocumentParts.Text).Text.Contains("12 Garden Lane", StringComparison.Ordinal), "an inherited field drawn into its page");
            PdfFormsSave("forms-inherited-flattened", bytes, text: false);
        }

        var xfa = PdfEditSample("forms-xfa");
        using (var xfaForm = PdfEditOpen(File.ReadAllBytes(xfa.Path), "forms-xfa"))
        {
            byte[] bytes = PdfEditSave(xfaForm, e => e.Flatten(), DocumentSaveMode.Rewrite);
            using var flat = PdfEditOpen(bytes, "forms-xfa flattened");
            Check(flat.Fields.Count == 0 && !flat.HasXfa && flat.ReadPage(0, DocumentParts.Text).Issues.Count == 0, "the XFA form flattened: no field, no XFA, no issue");
            Check(flat.ReadPage(0, DocumentParts.Text).Text.Contains("Jordan Lee", StringComparison.Ordinal), "its twins' values drawn into the page");
            PdfFormsSave("forms-xfa-flattened", bytes, text: false);
        }

        Check(failures.Count == 0, $"{failures.Count} differences:\n      {string.Join("\n      ", failures)}");
    }

    // A drawing saved beside a written form (NAME.idrak.p1.png), for a reference renderer's drawing of the same file.
    private static void PdfFormsSaveDrawing(string name, ImageData image) =>
        File.WriteAllBytes(Path.Combine(PdfFormsFolder(), name + ".idrak.p1.png"), new PngEncoder().Encode(image));

    // The share of a box's pixels (two pixels in from its edges) off by more than 32 levels between two drawings after a
    // 3 x 3 mean, the second read turned a half within the box when `flip`.
    private static double PdfFormsRegion(ImageData ours, ImageData reference, DocumentBox box, double dpi, bool flip)
    {
        int w = Math.Min(ours.Width, reference.Width), h = Math.Min(ours.Height, reference.Height);
        float[] a = PdfBlur(ours, w, h), b = PdfBlur(reference, w, h);
        int x0 = Math.Max(0, (int)Math.Ceiling(box.X * dpi / 72) + 2), y0 = Math.Max(0, (int)Math.Ceiling(box.Y * dpi / 72) + 2);
        int x1 = Math.Min(w, (int)(box.Right * dpi / 72) - 2), y1 = Math.Min(h, (int)(box.Bottom * dpi / 72) - 2);
        long off = 0, all = 0;
        for (int y = y0; y < y1; y++)
        {
            for (int x = x0; x < x1; x++)
            {
                int rx = flip ? x0 + x1 - 1 - x : x, ry = flip ? y0 + y1 - 1 - y : y;
                float worst = 0;
                for (int c = 0; c < 3; c++)
                {
                    worst = MathF.Max(worst, MathF.Abs(a[3 * (y * w + x) + c] - b[3 * (ry * w + rx) + c]));
                }

                off += worst > 32 ? 1 : 0;
                all++;
            }
        }

        return (double)off / Math.Max(1, all);
    }

    // The bounds of the ink (pixels darker than half) inside a box, 4 pixels in from its edges (its border left out).
    private static (int Width, int Height) PdfFormsInk(ImageData image, DocumentBox box, double dpi)
    {
        int x0 = (int)(box.X * dpi / 72) + 4, y0 = (int)(box.Y * dpi / 72) + 4, x1 = (int)(box.Right * dpi / 72) - 4, y1 = (int)(box.Bottom * dpi / 72) - 4;
        int left = int.MaxValue, top = int.MaxValue, right = -1, bottom = -1;
        for (int y = Math.Max(0, y0); y < Math.Min(image.Height, y1); y++)
        {
            for (int x = Math.Max(0, x0); x < Math.Min(image.Width, x1); x++)
            {
                if (image.Pixels[y * image.Width + x] < 0.5f)
                {
                    (left, top, right, bottom) = (Math.Min(left, x), Math.Min(top, y), Math.Max(right, x), Math.Max(bottom, y));
                }
            }
        }

        return right < 0 ? (0, 0) : (right - left + 1, bottom - top + 1);
    }

    private static void PdfFormsStyled(Device device)
    {
        CpuOnly(device);
        var sample = PdfEditSample("forms-styled");
        var fields = sample.Expected["fields"]!.AsArray();
        double dpi = (double?)sample.Expected["reference"]?["dpi"] ?? 100;
        var reference = new PngCodec().Decode(File.ReadAllBytes(Path.Combine(Path.GetDirectoryName(sample.Path)!, "forms-styled.p1.png")));
        byte[] source = File.ReadAllBytes(sample.Path);
        var failures = new List<string>();
        byte[]? filled = null;
        foreach (var mode in new[] { DocumentSaveMode.Incremental, DocumentSaveMode.Rewrite })
        {
            string label = mode == DocumentSaveMode.Incremental ? "incremental" : "rewrite";
            using var document = PdfEditOpen(source, "forms-styled");
            byte[] bytes = PdfEditSave(document, e => PdfFormsFillFrom(e, fields), mode);
            filled ??= bytes;
            using var back = PdfEditOpen(bytes, $"forms-styled filled ({label})");
            var read = back.Fields;
            Check(read.Count == fields.Count, $"forms-styled filled ({label}): {read.Count} fields");
            for (int i = 0; i < fields.Count; i++)
            {
                PdfFormsCompare($"forms-styled filled ({label})", read[i], fields[i]!.AsObject());
            }

            // Each appearance shows its value, read through the matrix that turns it.
            var shown = back.ReadPage(0, DocumentParts.Annotations).Annotations.Select(a => PdfFormsSquash(a.Appearance?.Text)).ToList();
            foreach (var f in read)
            {
                Check(shown.Contains(PdfFormsSquash(f.Value)), $"forms-styled filled ({label}): '{f.Name}' shown: {string.Join(" | ", shown)}");
            }

            var drawn = PdfFormsDraw(back, 0, dpi);
            var (share, mean) = PdfDrawDifference(drawn, reference, 32);
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"    forms-styled filled ({label}) drawn with its widgets: {share * 100:F2}% off pdfium's drawing of its own appearances by more than 32 levels (mean {mean:F2})"));
            if (share > 0.08)
            {
                failures.Add(string.Create(CultureInfo.InvariantCulture, $"forms-styled filled ({label}): {share * 100:F2}% off pdfium's drawing, past 8%"));
            }

            foreach (var f in read.Where(f => f.Widgets[0].Rotation != 0))
            {
                var box = f.Widgets[0].Box;
                double same = PdfFormsRegion(drawn, reference, box, dpi, flip: false), turned = PdfFormsRegion(drawn, reference, box, dpi, flip: true);
                var (inkWidth, inkHeight) = PdfFormsInk(drawn, box, dpi);
                Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                    $"      '{f.Name}' turned {f.Widgets[0].Rotation}: {same * 100:F1}% of its box off pdfium's, {turned * 100:F1}% off pdfium's turned a half; its ink {inkWidth} x {inkHeight} pixels"));
                if (!(same < turned) || (f.Widgets[0].Rotation is 90 or 270 && !(inkHeight > 2 * inkWidth)))
                {
                    failures.Add($"forms-styled filled ({label}): '{f.Name}' not turned as pdfium turns it ({same:P1} against {turned:P1} turned; ink {inkWidth} x {inkHeight})");
                }
            }

            PdfFormsSave($"forms-styled-filled-{label}", bytes);
            PdfFormsSaveDrawing($"forms-styled-filled-{label}", drawn);
        }

        // Flattened, it draws as it did with its widgets.
        using (var withWidgets = PdfEditOpen(filled!, "forms-styled filled"))
        {
            byte[] flat = PdfEditSave(withWidgets, e => e.Flatten(), DocumentSaveMode.Rewrite);
            using var flattened = PdfEditOpen(flat, "forms-styled flattened");
            var (share, _) = PdfDrawDifference(PdfFormsDraw(withWidgets, 0, dpi), PdfFormsDraw(flattened, 0, dpi, annotations: false), 32);
            Check(flattened.Fields.Count == 0 && share <= 0.001, string.Create(CultureInfo.InvariantCulture, $"forms-styled flattened: {flattened.Fields.Count} fields, {share * 100:F3}% off its drawing with widgets"));
            PdfFormsSave("forms-styled-flattened", flat, text: false);
        }

        // The builder turns and borders fields as asked.
        byte[] built = PdfWrite(b =>
        {
            b.AddPage(300, 300);
            b.AddField(new DocumentField("side", DocumentFieldKind.Text)
            {
                Value = "Along the side", FontSize = 11, Border = DocumentColour.Grey(0.3f), BorderStyle = DocumentFieldBorderStyle.Dashed, BorderWidth = 1.5f, Dashes = [4f, 2f],
                Widgets = [new DocumentWidget(0, new DocumentBox(20, 40, 26, 200)) { Rotation = 90 }],
            });
            b.AddField(new DocumentField("raised", DocumentFieldKind.Text)
            {
                Value = "Raised", FontSize = 11, Border = DocumentColour.Grey(0.3f), Background = DocumentColour.Grey(0.85f), BorderStyle = DocumentFieldBorderStyle.Beveled,
                Widgets = [new DocumentWidget(0, new DocumentBox(80, 40, 160, 26))],
            });
            b.AddField(new DocumentField("flipped", DocumentFieldKind.Text)
            {
                Value = "Upside down", FontSize = 11, Border = DocumentColour.Black, BorderStyle = DocumentFieldBorderStyle.Underline,
                Widgets = [new DocumentWidget(0, new DocumentBox(80, 100, 160, 26)) { Rotation = 180 }],
            });
            ExpectRefusal<ArgumentException>(() => b.AddField(new DocumentField("askew", DocumentFieldKind.Text) { Widgets = [new DocumentWidget(0, new DocumentBox(10, 10, 50, 20)) { Rotation = 45 }] }),
                "90", "a widget turned 45 degrees");
        });
        using var builtBack = PdfEditOpen(built, "styled fields built");
        var side = builtBack.Fields.Single(f => f.Name == "side");
        var raised = builtBack.Fields.Single(f => f.Name == "raised");
        var flipped = builtBack.Fields.Single(f => f.Name == "flipped");
        Check(side.Widgets[0].Rotation == 90 && side.BorderStyle == DocumentFieldBorderStyle.Dashed && side.BorderWidth == 1.5f && side.Dashes.SequenceEqual([4f, 2f]),
            $"the built 'side': turned {side.Widgets[0].Rotation}, {side.BorderStyle} {side.BorderWidth} [{string.Join(" ", side.Dashes)}]");
        Check(raised.Widgets[0].Rotation == 0 && raised.BorderStyle == DocumentFieldBorderStyle.Beveled && raised.BorderWidth == 1f,
            $"the built 'raised': {raised.BorderStyle} {raised.BorderWidth}");
        Check(flipped.Widgets[0].Rotation == 180 && flipped.BorderStyle == DocumentFieldBorderStyle.Underline, $"the built 'flipped': turned {flipped.Widgets[0].Rotation}, {flipped.BorderStyle}");
        var builtShown = builtBack.ReadPage(0, DocumentParts.Annotations).Annotations.Select(a => a.Appearance?.Text).ToList();
        Check(builtShown.SequenceEqual(["Along the side", "Raised", "Upside down"]), $"the built fields' appearances: {string.Join(" | ", builtShown)}");
        var (width, height) = PdfFormsInk(PdfFormsDraw(builtBack, 0, 100), side.Widgets[0].Box, 100);
        Check(height > 2 * width, $"the built 'side' runs up its box: its ink {width} x {height} pixels");
        PdfFormsSave("forms-styled-built", built);
        Check(failures.Count == 0, $"{failures.Count} differences:\n      {string.Join("\n      ", failures)}");
    }

    private static void PdfFormsRich(Device device)
    {
        CpuOnly(device);
        var sample = PdfEditSample("forms-rich");
        byte[] source = File.ReadAllBytes(sample.Path);
        using var document = PdfEditOpen(source, "forms-rich");
        var message = document.Fields.Single(f => f.Name == "message");
        Check(message.RichText.Count == 6 && string.Concat(message.RichText.Select(r => r.Text)) == message.Value && (message.Flags & DocumentFieldFlags.RichText) != 0,
            $"forms-rich's message: {PdfFormsDescribe(message.RichText)} for '{message.Value}'");
        DocumentFieldRun[] runs =
        [
            new("Bold start") { Bold = true }, new(" then italic red") { Italic = true, Colour = DocumentColour.Rgb(0.8f, 0f, 0f) },
            new("\nTimes at 13 points") { Family = "Times", Size = 13 }, new(" underlined") { Underline = true }, new(" struck") { StrikeOut = true },
        ];
        string plain = string.Concat(runs.Select(r => r.Text));
        foreach (var mode in new[] { DocumentSaveMode.Incremental, DocumentSaveMode.Rewrite })
        {
            string label = mode == DocumentSaveMode.Incremental ? "incremental" : "rewrite";
            byte[] bytes = PdfEditSave(document, e =>
            {
                e.SetRichText("note", runs);
                e.SetRichText("title", [new DocumentFieldRun("Plain made rich") { Bold = true }]);
                e.SetField("message", "Replaced plainly");
            }, mode);
            using var back = PdfEditOpen(bytes, $"forms-rich filled ({label})");
            var note = back.Fields.Single(f => f.Name == "note");
            Check(note.Value == plain && (note.Flags & DocumentFieldFlags.RichText) != 0 && PdfFormsStyles(note.RichText) == PdfFormsStyles(runs),
                $"forms-rich filled ({label}): note '{note.Value}' {note.Flags} {PdfFormsDescribe(note.RichText)}");
            var title = back.Fields.Single(f => f.Name == "title");
            Check(title.Value == "Plain made rich" && (title.Flags & DocumentFieldFlags.RichText) != 0 && title.RichText is [{ Bold: true, Text: "Plain made rich" }],
                $"forms-rich filled ({label}): the plain field made rich: '{title.Value}' {title.Flags} {PdfFormsDescribe(title.RichText)}");
            var replaced = back.Fields.Single(f => f.Name == "message");
            Check(replaced.Value == "Replaced plainly" && string.Concat(replaced.RichText.Select(r => r.Text)) == "Replaced plainly" && replaced.RichText.All(r => !r.Bold && !r.Italic),
                $"forms-rich filled ({label}): a plain value set on a rich text field is its one run: {PdfFormsDescribe(replaced.RichText)}");

            // The appearance: each run in its own font, size and colour.
            var page = back.ReadPage(0, DocumentParts.Annotations);
            var appearance = page.Annotations.Single(a => PdfNear(a.Box, note.Widgets[0].Box, 0.01f)).Appearance!;
            Check(PdfFormsSquash(appearance.Text) == PdfFormsSquash(plain), $"forms-rich filled ({label}): the note shows \"{appearance.Text}\"");
            var shownRuns = appearance.Runs;
            string Runs() => string.Join(", ", shownRuns.Select(r => $"{r.Text}: {r.Font} {r.FontSize} {r.Colour}"));
            Check(shownRuns.Any(r => r.Text.Contains("Bold", StringComparison.Ordinal) && r.Font?.Contains("Bold", StringComparison.Ordinal) == true),
                $"forms-rich filled ({label}): the bold run in a bold font: {Runs()}");
            Check(shownRuns.Any(r => r.Text.Contains("italic", StringComparison.Ordinal) && r.Font?.Contains("Oblique", StringComparison.Ordinal) == true
                                     && r.Colour.Space == DocumentColourSpace.Rgb && MathF.Abs(r.Colour[0] - 0.8f) < 0.01f && r.Colour[1] < 0.01f),
                $"forms-rich filled ({label}): the italic run in an oblique font, red: {Runs()}");
            Check(shownRuns.Any(r => r.Text.Contains("Times", StringComparison.Ordinal) && r.Font?.Contains("Times", StringComparison.Ordinal) == true && MathF.Abs(r.FontSize - 13) < 0.01f),
                $"forms-rich filled ({label}): the Times run at 13 points: {Runs()}");
            PdfFormsSave($"forms-rich-filled-{label}", bytes);
            PdfFormsSaveDrawing($"forms-rich-filled-{label}", PdfFormsDraw(back, 0, 100));
        }

        // Runs refused as they should be.
        using (var editor = new PdfWriter().Edit(document, new MemoryStream(), PdfWriteDefaults))
        {
            ExpectRefusal<ArgumentException>(() => editor.SetRichText("note", [new DocumentFieldRun("x") { Size = 0 }]), "above 0", "a run of size 0");
            ExpectRefusal<ArgumentException>(() => editor.SetRichText("nothing", [new DocumentFieldRun("x")]), "no form field", "a field the form lacks");
        }

        // The builder adds a rich text field.
        byte[] built = PdfWrite(b =>
        {
            b.AddPage(300, 200);
            b.AddField(new DocumentField("memo", DocumentFieldKind.Text) { Flags = DocumentFieldFlags.Multiline, FontSize = 10, RichText = runs, Widgets = [new DocumentWidget(0, new DocumentBox(20, 20, 260, 80))] });
            ExpectRefusal<ArgumentException>(() => b.AddField(new DocumentField("tick", DocumentFieldKind.CheckBox) { RichText = [new DocumentFieldRun("x")], Widgets = [new DocumentWidget(0, new DocumentBox(20, 120, 16, 16))] }),
                "text field", "rich text on a check box");
        });
        using var builtBack = PdfEditOpen(built, "a rich text field built");
        var memo = builtBack.Fields.Single();
        Check(memo.Value == plain && memo.Flags == (DocumentFieldFlags.Multiline | DocumentFieldFlags.RichText) && PdfFormsStyles(memo.RichText) == PdfFormsStyles(runs),
            $"the built rich text field: '{memo.Value}' {memo.Flags} {PdfFormsDescribe(memo.RichText)}");
        Check(PdfFormsSquash(builtBack.ReadPage(0, DocumentParts.Annotations).Annotations.Single().Appearance?.Text) == PdfFormsSquash(plain), "the built rich text field's appearance shows its text");
        PdfFormsSave("forms-rich-built", built);
    }

    private static void PdfFormsHeld(Device device)
    {
        CpuOnly(device);
        var sample = PdfEditSample("forms-held");
        byte[] source = File.ReadAllBytes(sample.Path);
        (string Name, string Value)[] values = [("applicant.name", "Robin Ellis"), ("applicant.email", "robin.ellis@example.com"), ("remarks", "Arrives on Tuesday.\nCall ahead."), ("city", "Lisbon")];
        using var document = PdfEditOpen(source, "forms-held");
        Check(document.Form.Find("applicant.name")!.Number == 0 && document.Form.Find("remarks")!.Number == 0 && Encoding.Latin1.GetString(source).Contains("/ObjStm", StringComparison.Ordinal),
            "forms-held: applicant.name and remarks held inside other objects, the others in object streams");
        foreach (var mode in new[] { DocumentSaveMode.Incremental, DocumentSaveMode.Rewrite })
        {
            string label = mode == DocumentSaveMode.Incremental ? "incremental" : "rewrite";
            byte[] bytes = PdfEditSave(document, e =>
            {
                foreach (var (name, value) in values)
                {
                    e.SetField(name, value);
                }
            }, mode);
            if (mode == DocumentSaveMode.Incremental)
            {
                Check(bytes.AsSpan(0, source.Length).SequenceEqual(source) && Encoding.Latin1.GetString(bytes, source.Length, bytes.Length - source.Length).Contains("/XRef", StringComparison.Ordinal),
                    "forms-held filled incrementally: its bytes the prefix, the update's cross-reference a stream as the source's");
            }

            using var back = PdfEditOpen(bytes, $"forms-held filled ({label})");
            foreach (var (name, value) in values)
            {
                var field = back.Fields.Single(f => f.Name == name);
                var held = back.Form.Find(name)!;
                Check(field.Value == value && held.Number > 0, $"forms-held filled ({label}): '{name}' = '{field.Value}', object {held.Number}");
                foreach (var widget in held.Widgets.Where(w => w.Number != held.Number))
                {
                    Check(widget.Dictionary["Parent"u8] is { IsReference: true } parent && parent.AsReference.Number == held.Number,
                        $"forms-held filled ({label}): '{name}''s widget names it as its parent ({widget.Dictionary["Parent"u8]})");
                }
            }

            var annotations = back.ReadPage(0, DocumentParts.Annotations).Annotations;
            Check(annotations.Count == 4 && values.All(v => annotations.Any(a => PdfFormsSquash(a.Appearance?.Text) == PdfFormsSquash(v.Value))),
                $"forms-held filled ({label}): its 4 widgets show the values: {string.Join(" | ", annotations.Select(a => a.Appearance?.Text))}");
            PdfFormsSave($"forms-held-filled-{label}", bytes);
        }

        // The held fields flattened: out of their holders, drawn into the page.
        byte[] flat = PdfEditSave(document, e =>
        {
            e.SetField("remarks", "Flattened remarks");
            e.Flatten("remarks", "applicant.name");
        }, DocumentSaveMode.Incremental);
        using var flattened = PdfEditOpen(flat, "forms-held flattened");
        Check(flattened.Fields.Select(f => f.Name).SequenceEqual(["applicant.email", "city"]) && flattened.ReadPage(0, DocumentParts.Annotations).Annotations.Count == 2,
            $"forms-held, its held fields flattened: {string.Join(", ", flattened.Fields.Select(f => f.Name))}");
        Check(flattened.ReadPage(0, DocumentParts.Text).Text.Contains("Flattened remarks", StringComparison.Ordinal), "the flattened held field's value is page text");
        PdfFormsSave("forms-held-flattened", flat, text: false);
    }

    private static void PdfFormsFaces(Device device)
    {
        CpuOnly(device);
        var sample = PdfEditSample("forms-faces");
        byte[] source = File.ReadAllBytes(sample.Path);
        double dpi = (double?)sample.Expected["reference"]?["dpi"] ?? 100;
        var reference = new PngCodec().Decode(File.ReadAllBytes(Path.Combine(Path.GetDirectoryName(sample.Path)!, "forms-faces.p1.png")));
        var options = PdfWriteDefaults with { Forms = new DocumentFormOptions { Faces = [SubsetFace()] } };
        const string Greeting = "Добрый день";
        using var document = PdfEditOpen(source, "forms-faces");
        Check(document.Fields.Single(f => f.Name == "greeting").Font == "Helv", "forms-faces: the greeting's /DA names Helvetica");
        foreach (var mode in new[] { DocumentSaveMode.Incremental, DocumentSaveMode.Rewrite })
        {
            string label = mode == DocumentSaveMode.Incremental ? "incremental" : "rewrite";
            byte[] bytes = PdfEditSave(document, e =>
            {
                e.SetField("greeting", Greeting);
                e.SetField("note", "Plain Latin text");
            }, mode, options);
            using var back = PdfEditOpen(bytes, $"forms-faces filled ({label})");
            var greeting = back.Fields.Single(f => f.Name == "greeting");
            var note = back.Fields.Single(f => f.Name == "note");
            Check(greeting.Value == Greeting && greeting.Font == "F1_1" && greeting.FontSize == 12 && note.Font == "Helv" && note.Value == "Plain Latin text",
                $"forms-faces filled ({label}): the greeting's /DA names {greeting.Font} at {greeting.FontSize}, the note's {note.Font}");
            var fonts = back.File.Dictionary(back.Form.Resources!["Font"u8])!;
            var added = back.File.Dictionary(fonts["F1_1"u8]);
            Check(added is not null && added["Subtype"u8].IsName("Type0"u8) && PdfFonts.Read(back.File, added).Name?.Contains("DejaVu", StringComparison.Ordinal) == true
                  && fonts.Contains("F1"u8) && fonts.Contains("Helv"u8) && !ReferenceEquals(back.File.Dictionary(fonts["F1"u8]), added),
                $"forms-faces filled ({label}): /DR lists the face's font as /F1_1 beside its own /F1 and /Helv");
            var shown = back.ReadPage(0, DocumentParts.Annotations).Annotations.Select(a => a.Appearance?.Text).ToList();
            Check(shown.Contains(Greeting) && shown.Contains("Plain Latin text"), $"forms-faces filled ({label}): shown {string.Join(" | ", shown)}");
            // The note's box left out: its producer drew the box's background inside its /Tx section, which a value set
            // replaces (12.7.4.3), so pdfium's drawing of the empty note shows a background the filled one does not.
            var drawn = PdfFormsDraw(back, 0, dpi);
            var skip = new bool[drawn.Width * drawn.Height];
            var noteBox = note.Widgets[0].Box;
            for (int y = Math.Max(0, (int)(noteBox.Y * dpi / 72) - 2); y < Math.Min(drawn.Height, (int)(noteBox.Bottom * dpi / 72) + 3); y++)
            {
                for (int x = Math.Max(0, (int)(noteBox.X * dpi / 72) - 2); x < Math.Min(drawn.Width, (int)(noteBox.Right * dpi / 72) + 3); x++)
                {
                    skip[y * drawn.Width + x] = true;
                }
            }

            var (share, mean) = PdfDrawDifference(drawn, reference, 32, skip);
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"    forms-faces filled ({label}) drawn: {share * 100:F2}% off pdfium's drawing of the producer's appearance of the same greeting by more than 32 levels (mean {mean:F2}), the note's box left out"));
            Check(share <= 0.08, string.Create(CultureInfo.InvariantCulture, $"forms-faces filled ({label}): {share * 100:F2}% off pdfium's drawing, past 8%"));
            PdfFormsSave($"forms-faces-filled-{label}", bytes);
            PdfFormsSaveDrawing($"forms-faces-filled-{label}", drawn);
        }
    }

    private static void PdfFormsTemplates(Device device)
    {
        CpuOnly(device);
        int compared = 0, templates = 0;
        foreach (var sample in PdfSamples().Where(s => s.Phases.Contains("D9") && s.Readable))
        {
            using var document = new PdfFormat().Open(new MemoryStream(File.ReadAllBytes(sample.Path)), sample.Path, new DocumentOptions());
            var expected = sample.Expected["form"]?["xfa"]?["template"] as JsonObject;
            var template = document.FormTemplate;
            if (expected is null)
            {
                Check(template is null, $"{sample.Name}: no template, read as none");
                continue;
            }

            templates++;
            Check(template is not null, $"{sample.Name}: its XFA template read");
            var sections = expected["sections"]!.AsArray().Select(x => $"{(string?)x!["name"]}={(string?)x["path"]}<{(string?)x["parent"]}").ToList();
            Check(template!.Sections.Select(x => $"{x.Name}={x.Path}<{x.Parent}").SequenceEqual(sections),
                $"{sample.Name}: sections {string.Join(", ", template.Sections.Select(x => $"{x.Name}={x.Path}<{x.Parent}"))}, expected {string.Join(", ", sections)}");
            var fields = expected["fields"]!.AsArray();
            Check(template.Fields.Count == fields.Count, $"{sample.Name}: {template.Fields.Count} template fields ({string.Join(", ", template.Fields.Select(f => f.Path))}), expected {fields.Count}");
            for (int i = 0; i < Math.Min(fields.Count, template.Fields.Count); i++)
            {
                var e = fields[i]!.AsObject();
                var f = template.Fields[i];
                string kind = (string)e["kind"]!;
                var options = (e["options"] as JsonArray ?? []).Select(o => new DocumentFieldOption((string)o!["display"]!, (string)o["export"]!)).ToList();
                Check(f.Name == (string)e["name"]! && f.Path == (string)e["path"]! && f.Kind == PdfFormsKind(kind) && f.Section == (string?)e["section"] && f.Caption == (string?)e["caption"]
                      && f.Binding == (string?)e["binding"] && f.Value == (string?)e["value"] && f.Options.SequenceEqual(options) && f.Flags == PdfFormsFlags(e["flags"], kind)
                      && f.Field == (string?)e["field"],
                    $"{sample.Name}: template field {i}: {f.Name} {f.Path} {f.Kind} in {f.Section}, '{f.Caption}', bound to {f.Binding} = '{f.Value}', [{string.Join(", ", f.Options)}], {f.Flags}, held by {f.Field}; expected {e.ToJsonString()}");
                if (f.Field is { } held)
                {
                    var twin = document.Fields.SingleOrDefault(x => x.Name == held);
                    Check(twin is not null && twin.Value == f.Value, $"{sample.Name}: '{f.Path}' held by the field '{held}' with its value ('{twin?.Value}')");
                }

                compared++;
            }

            int scripts = (int)expected["scripts"]!;
            Check(template.Issues.Count == scripts && template.Issues.All(x => x.Kind == DocumentIssue.NotSupported && x.Detail.Contains("not run", StringComparison.Ordinal)),
                $"{sample.Name}: {template.Issues.Count} issues ({string.Join("; ", template.Issues.Select(x => x.Detail))}), expected its {scripts} scripts listed as not run");
        }

        Console.WriteLine($"    {templates} XFA templates, {compared} template fields read as their JSON says");
        Check(templates >= 2 && compared >= 16, $"{templates} templates and {compared} fields compared; the D9 samples hold more");
    }

    private static void PdfFormsMemory(Device device)
    {
        CpuOnly(device);
        var filled = PdfEditSample("forms-filled");
        using var document = PdfEditOpen(File.ReadAllBytes(filled.Path), "forms-filled");
        var form = PdfForms.Read(document);                                    // the objects read once: what is measured is the form's reading
        for (int i = 0; i < 200; i++)
        {
            form = PdfForms.Read(document);
        }

        long least = long.MaxValue;
        for (int r = 0; r < 15; r++)
        {
            long before = GC.GetAllocatedBytesForCurrentThread();
            form = PdfForms.Read(document);
            least = Math.Min(least, GC.GetAllocatedBytesForCurrentThread() - before);
        }

        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    reading forms-filled's form ({form.Fields.Count} fields, {form.Fields.Sum(f => f.Widgets.Count)} widgets), the document kept open, warm: {least / 1024.0:F1} KiB (40.4 KiB before the forms were completed)"));
        Check(form.Fields.Count == 12 && form.Fields.Sum(f => f.Widgets.Count) == 14, "forms-filled's form: 12 fields, 14 widgets");
        Check(least <= 24 * 1024, string.Create(CultureInfo.InvariantCulture, $"reading forms-filled's form allocates {least / 1024.0:F1} KiB, past 24 KiB (about what it returns)"));
    }

    private static void PdfFormsTimings(Device device)
    {
        CpuOnly(device);
        var filled = PdfEditSample("forms-filled");
        byte[] bytes = File.ReadAllBytes(filled.Path);
        (double Ms, long Bytes) Measure(Action action)
        {
            long allocated = GC.GetAllocatedBytesForCurrentThread();
            long start = Stopwatch.GetTimestamp();
            action();
            return (Stopwatch.GetElapsedTime(start).TotalMilliseconds, GC.GetAllocatedBytesForCurrentThread() - allocated);
        }

        (double Ms, long Bytes) Warm(Action action)
        {
            for (int i = 0; i < 30; i++)
            {
                action();
            }

            var runs = Enumerable.Range(0, 7).Select(_ => Measure(action)).OrderBy(r => r.Ms).ToList();
            return runs[runs.Count / 2];
        }

        void Read()
        {
            using var document = PdfEditOpen(bytes, "forms-filled");
            _ = document.Fields;
        }

        void Fill()
        {
            using var document = PdfEditOpen(bytes, "forms-filled");
            using var destination = new MemoryStream();
            using var editor = new PdfWriter().Edit(document, destination, PdfWriteDefaults);
            editor.SetField("name", "Robin Ellis");
            editor.Save(DocumentSaveMode.Incremental);
        }

        void Flatten()
        {
            using var document = PdfEditOpen(bytes, "forms-filled");
            using var destination = new MemoryStream();
            using var editor = new PdfWriter().Edit(document, destination, PdfWriteDefaults);
            editor.Flatten();
            editor.Save(DocumentSaveMode.Incremental);
        }

        foreach (var (what, action) in new (string, Action)[] { ("reading every field (14 widgets)", Read), ("filling one field, saved incrementally", Fill), ("flattening the page (12 fields), saved incrementally", Flatten) })
        {
            var cold = Measure(action);
            var warm = Warm(action);
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"    {what}: cold {cold.Ms:F2} ms ({cold.Bytes / 1024.0:F1} KiB), warm {warm.Ms:F3} ms ({warm.Bytes / 1024.0:F1} KiB)"));
        }
    }
}
