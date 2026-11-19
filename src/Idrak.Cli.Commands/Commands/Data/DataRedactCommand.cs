// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands.Data;

/// <summary>
/// <c>idrak data redact IN.pdf (--boxes FILE | --text TEXT | --marked) -o OUT.pdf</c> (plan 28, decision 17): what lies
/// under the areas removed for real (text, the parts of paths, image pixels) through the document editor's
/// <see cref="IDocumentEditor.Redact"/>, areas given by page and box, found where a text stands, or the file's own
/// redaction marks; the result written as a rewrite, so nothing removed survives in an earlier revision.
/// </summary>
internal sealed class DataRedactCommand : Command
{
    public override string Name => "data redact";

    public override string Summary => "Remove text, drawings and image pixels under areas of a PDF for real, saved as a clean rewrite";

    public override string Usage => """
        IN.pdf (--boxes FILE | --text TEXT | --marked) -o OUT.pdf [options]

        Reads IN.pdf, removes what lies under each area (the glyphs whose boxes overlap it leave the text, a drawing inside
        it is removed and one crossing it keeps only its parts outside, an image's pixels under it take the area's colour,
        the annotations over it are removed), fills each area, and writes OUT.pdf as one clean file (no earlier revision
        holds what was removed). Areas come from any of the three options, together.

        Options:
          -o, --out FILE      the redacted PDF (not the input)
              --boxes FILE    areas as JSON: [{"page": 1, "box": [x, y, width, height]}, ...] in points from the page's
                              top-left corner as shown (a page from 1; "fill": "#rrggbb" or "none" for one area's colour)
              --text TEXT     every place TEXT stands (again for more); matched along each line of the text the file
                              shows, so a word split across pieces, fonts or kerning is found too
              --ignore-case   match --text without case
              --marked        the file's own redaction marks (each mark's quads, filled in its own colour)
              --fill COLOUR   the areas' colour: black (the default), white, none (left unfilled) or #rrggbb
              --password PW   the password of an encrypted input (the output is written without encryption)

        Shows its progress (pages done, areas removed) and then a line per page with what was removed: glyphs and their
        characters, drawings removed and clipped, images and their pixels, annotations.

        Examples:
          idrak data redact report.pdf --text "Project Falcon" -o report-shared.pdf
          idrak data redact newsletter.pdf --boxes areas.json -o newsletter-public.pdf
          idrak data redact letter.pdf --marked -o letter-sent.pdf --json
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } = ["--out", "--boxes", "--text", "--fill", "--password"];

    public override IReadOnlyCollection<string> Flags { get; } = ["--marked", "--ignore-case"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-o"] = "--out" };

    public override int Run(CommandContext context)
    {
        string input = context.Argument(0, "IN.pdf (the PDF to redact)");
        string output = context.Option("--out") ?? throw new UsageException("Give -o, --out FILE for the redacted PDF.");
        var texts = context.Options("--text").Where(t => t.Length > 0).ToList();
        bool marked = context.Flag("--marked");
        string? boxesFile = context.Option("--boxes");
        if (texts.Count == 0 && !marked && boxesFile is null)
        {
            throw new UsageException("Give the areas: --boxes FILE, --text TEXT or --marked.");
        }

        if (!File.Exists(input))
        {
            throw new UsageException($"{input} is not there.");
        }

        if (Path.GetFullPath(input).Equals(Path.GetFullPath(output), StringComparison.OrdinalIgnoreCase))
        {
            throw new UsageException("Write the redacted PDF to another file (-o): the input is read while it is written.");
        }

        var fill = context.Option("--fill") is { } given ? Colour(given, "--fill") : DocumentColour.Black;
        var comparison = context.Flag("--ignore-case") ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        List<(int Page, DocumentRedaction Area)> boxes = boxesFile is null ? [] : Boxes(boxesFile, fill);
        var options = new DocumentOptions { Password = context.Option("--password") };
        var parts = (texts.Count > 0 ? DocumentParts.Text | DocumentParts.Glyphs : DocumentParts.None) | (marked ? DocumentParts.Annotations : DocumentParts.None);
        var results = new List<DocumentRedactionResult>();
        string temporary = output + ".partial";
        try
        {
            using var document = DocumentFormats.Open(input, options);
            if (boxes.Any(b => b.Page > document.PageCount))
            {
                throw new UsageException($"{boxesFile}: an area is on page {boxes.Max(b => b.Page)}, and {input} has {document.PageCount} pages.");
            }

            using (var destination = new FileStream(temporary, FileMode.Create, FileAccess.Write, FileShare.None))
            using (var editor = DocumentWriters.Get("pdf").Edit(document, destination, new DocumentWriteOptions()))
            {
                int areas = 0;
                using (var line = new ProgressLine(context, $"redacting {Path.GetFileName(input)}", document.PageCount, ProgressUnit.Items))
                {
                    line.Detail = () => string.Create(CultureInfo.InvariantCulture, $"{areas:N0} areas removed");
                    for (int page = 0; page < document.PageCount; page++)
                    {
                        var found = new List<DocumentRedaction>(boxes.Where(b => b.Page == page + 1).Select(b => b.Area));
                        if (parts != DocumentParts.None)
                        {
                            var read = document.ReadPage(page, parts);
                            foreach (string text in texts)
                            {
                                found.AddRange(DocumentRedactions.Find(read, text, comparison, options: options).Select(a => a with { Fill = fill }));
                            }

                            if (marked)
                            {
                                found.AddRange(context.Option("--fill") is null ? DocumentRedactions.Marked(read) : DocumentRedactions.Marked(read).Select(a => a with { Fill = fill }));
                            }
                        }

                        if (found.Count > 0)
                        {
                            results.Add(editor.Redact(page, found));
                            areas += found.Count;
                        }

                        line.Advance();
                    }

                    line.NextStage($"writing {Path.GetFileName(output)}", null, ProgressUnit.Items);
                    editor.Save(DocumentSaveMode.Rewrite);
                    line.Finish();
                }
            }

            File.Move(temporary, output, overwrite: true);
        }
        catch (Exception e) when (e is InvalidDataException or NotSupportedException or IOException or UnauthorizedAccessException or ArgumentException)
        {
            File.Delete(temporary);
            throw new UsageException($"{input}: {e.Message}");
        }
        catch
        {
            File.Delete(temporary);
            throw;
        }

        foreach (var r in results)
        {
            context.Write(string.Create(CultureInfo.InvariantCulture,
                $"page {r.Page + 1}: {r.Areas} area{(r.Areas == 1 ? "" : "s")}: {r.Glyphs:N0} glyphs ({r.Characters:N0} characters), {r.PathsRemoved:N0} drawings removed, {r.PathsClipped:N0} clipped, {r.Images:N0} images ({r.Pixels:N0} pixels), {r.Annotations:N0} annotations"));
        }

        context.Write(string.Create(CultureInfo.InvariantCulture,
            $"{results.Sum(r => r.Areas):N0} areas on {results.Count:N0} pages: {results.Sum(r => r.Glyphs):N0} glyphs, {results.Sum(r => r.PathsRemoved + r.PathsClipped):N0} drawings, {results.Sum(r => r.Pixels):N0} pixels and {results.Sum(r => r.Annotations):N0} annotations removed; written to {output}"));
        context.WriteJson(new JsonObject
        {
            ["input"] = input,
            ["output"] = output,
            ["pages"] = new JsonArray([.. results.Select(r => (JsonNode)new JsonObject
            {
                ["page"] = r.Page + 1, ["areas"] = r.Areas, ["glyphs"] = r.Glyphs, ["characters"] = r.Characters, ["pathsRemoved"] = r.PathsRemoved,
                ["pathsClipped"] = r.PathsClipped, ["images"] = r.Images, ["pixels"] = r.Pixels, ["annotations"] = r.Annotations,
            })]),
        });
        return ExitCodes.Ok;
    }

    // The areas of a boxes file: a JSON array of {"page": N, "box": [x, y, width, height], "fill"?}.
    private static List<(int Page, DocumentRedaction Area)> Boxes(string path, DocumentColour? fill)
    {
        JsonNode? root;
        try
        {
            root = JsonNode.Parse(File.ReadAllText(path));
        }
        catch (Exception e) when (e is IOException or JsonException or UnauthorizedAccessException)
        {
            throw new UsageException($"{path}: {e.Message}");
        }

        if (root is not JsonArray array)
        {
            throw new UsageException($"{path}: a JSON array of areas, each {{\"page\": 1, \"box\": [x, y, width, height]}}.");
        }

        var areas = new List<(int, DocumentRedaction)>();
        foreach (var item in array)
        {
            if (item is not JsonObject area || area["page"] is not JsonValue page || !page.TryGetValue(out int number) || number < 1
                || area["box"] is not JsonArray { Count: 4 } box || box.Any(v => v is not JsonValue n || !n.TryGetValue(out double _)))
            {
                throw new UsageException($"{path}: area {areas.Count + 1} is not {{\"page\": N (from 1), \"box\": [x, y, width, height]}}.");
            }

            var values = box.Select(v => (float)v!.GetValue<double>()).ToArray();
            if (!(values[2] > 0 && values[3] > 0) || values.Any(v => !float.IsFinite(v)))
            {
                throw new UsageException($"{path}: area {areas.Count + 1}'s width and height are above 0.");
            }

            var own = area["fill"] is JsonValue colour && colour.TryGetValue(out string? text) ? Colour(text, $"{path}: area {areas.Count + 1}'s fill") : fill;
            areas.Add((number, new DocumentRedaction(new DocumentBox(values[0], values[1], values[2], values[3])) { Fill = own }));
        }

        return areas;
    }

    // A colour by name (black, white, none) or #rrggbb; null for none.
    private static DocumentColour? Colour(string text, string what)
    {
        switch (text.Trim().ToLowerInvariant())
        {
            case "black":
                return DocumentColour.Black;
            case "white":
                return DocumentColour.White;
            case "none":
                return null;
        }

        var hex = text.Trim().TrimStart('#');
        if (hex.Length == 6 && int.TryParse(hex, NumberStyles.HexNumber, CultureInfo.InvariantCulture, out int rgb))
        {
            return DocumentColour.Rgb((rgb >> 16 & 255) / 255f, (rgb >> 8 & 255) / 255f, (rgb & 255) / 255f);
        }

        throw new UsageException($"{what}: '{text}' is not a colour (black, white, none or #rrggbb).");
    }
}
