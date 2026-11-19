// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Imaging;

/// <summary>
/// "shaped", the library's text setter (<see cref="DocumentTextSetters.Shaped"/>): each paragraph laid out by
/// <see cref="TextLayout"/> (the Unicode bidirectional algorithm, runs of one script and direction, each shaped by its
/// script's shaper with the first face that has its characters, lines broken at spaces), each line's glyphs given back as
/// runs of one face, standing for the characters of their clusters, placed from the line's start on its baseline. Nothing
/// here is for one language or script.
/// </summary>
internal sealed class ShapedTextSetter : IDocumentTextSetter
{
    public string Name => DocumentTextSetters.Shaped;

    public string Summary => "each paragraph through the bidirectional algorithm, its runs shaped by their scripts' shapers in the first face that has their characters, lines broken at spaces";

    public IReadOnlyList<DocumentSetLine> Set(string text, DocumentTextSetting setting)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(setting);
        if (setting.Faces is not { Count: > 0 } faces || faces.Any(f => f is null))
        {
            throw new ArgumentException("Text is set in at least one face (DocumentTextSetting.Faces), none of them null.", nameof(setting));
        }

        if (!(setting.Size > 0f) || !float.IsFinite(setting.Size))
        {
            throw new ArgumentException($"Text is set at a finite size above 0, not {setting.Size}.", nameof(setting));
        }

        var style = new TextStyle(faces[0], setting.Size) { Fallbacks = faces.Count > 1 ? [.. faces.Skip(1)] : [] };
        var lines = new List<DocumentSetLine>();
        foreach (string raw in text.Split('\n'))
        {
            string paragraph = raw.TrimEnd('\r');
            if (paragraph.Length == 0)
            {
                lines.Add(new DocumentSetLine("", 0f, false, []));
                continue;
            }

            var block = TextLayout.Layout(paragraph, style, setting.Width, TextAlignment.Left, setting.RightToLeft);
            foreach (var line in block.Lines)
            {
                lines.Add(Line(paragraph, line));
            }
        }

        return lines;
    }

    // A laid-out line as runs of one face at one scale, each glyph placed from the line's left edge on its baseline.
    private static DocumentSetLine Line(string paragraph, TextLine line)
    {
        if (line.Glyphs.Count == 0)
        {
            return new DocumentSetLine(line.Text, 0f, line.RightToLeft, []);
        }

        int end = Math.Min(paragraph.Length, line.Start + line.Length);
        while (end > line.Start && char.IsWhiteSpace(paragraph[end - 1]))
        {
            end--;                                                             // trailing spaces draw nothing
        }

        // The line's clusters in text order, each up to the next (the last to the line's end).
        var clusters = line.Glyphs.Select(g => g.Cluster).Distinct().Order().ToList();
        var runs = new List<DocumentSetRun>();
        float left = line.Left;
        int i = 0;
        while (i < line.Glyphs.Count)
        {
            var first = line.Glyphs[i];
            int j = i;
            while (j + 1 < line.Glyphs.Count && ReferenceEquals(line.Glyphs[j + 1].Font, first.Font) && line.Glyphs[j + 1].Scale == first.Scale)
            {
                j++;
            }

            runs.Add(Run(paragraph, line, i, j, end, clusters, left));
            i = j + 1;
        }

        return new DocumentSetLine(line.Text, line.Width, line.RightToLeft, runs);
    }

    // Glyphs [first, last] of a line, one face at one scale, as a run standing for the characters of their clusters.
    private static DocumentSetRun Run(string paragraph, TextLine line, int first, int last, int end, List<int> clusters, float left)
    {
        var face = line.Glyphs[first].Font;
        float scale = line.Glyphs[first].Scale;
        var glyphs = new DocumentGlyph[last - first + 1];
        var own = new SortedSet<int>();
        for (int k = first; k <= last; k++)
        {
            var g = line.Glyphs[k];
            glyphs[k - first] = new DocumentGlyph(g.Glyph, g.X - left, g.Y - line.Baseline, face.Advance(g.Glyph) * scale);
            own.Add(g.Cluster);
        }

        var characters = new StringBuilder();
        foreach (int start in own)
        {
            int next = clusters.BinarySearch(start) + 1;
            int stop = next < clusters.Count ? clusters[next] : end;
            if (stop > start && start < paragraph.Length)
            {
                characters.Append(paragraph, start, Math.Min(stop, paragraph.Length) - start);
            }
        }

        return new DocumentSetRun(face, face.Metrics.UnitsPerEm * scale, glyphs, characters.ToString());
    }
}
