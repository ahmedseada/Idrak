// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Data;

// "tagged": the order a tagged document's author gave. The runs of each structure element (DocumentTextRun.Tag's Element)
// make a block, the blocks in the structure's order (the least Order of their runs); runs that are not tagged make one
// block after them; artifacts are left out of the lines and the text. Each block is laid out by "lines" (its runs grouped
// into lines by their places, words split at gaps, the reading order), and the lines follow one another, the text joined by
// line feeds. A page with no tagged run is laid out by "lines" as it is: the layout reads any file, tagged or not.
internal sealed class TaggedLayout : IDocumentTextLayout
{
    private readonly LinesLayout _lines = new();

    public string Name => DocumentTextLayouts.Tagged;

    public string Summary => "the order a tagged document's author gave: each structure element's runs laid out as lines, the elements in the structure's order, artifacts left out";

    public DocumentText Arrange(IReadOnlyList<DocumentTextRun> runs, float width, float height, DocumentOptions options)
    {
        ArgumentNullException.ThrowIfNull(runs);
        ArgumentNullException.ThrowIfNull(options);
        bool tagged = false;
        foreach (var run in runs)
        {
            tagged |= run.Tag is { IsArtifact: false };
        }

        if (!tagged)
        {
            return _lines.Arrange(runs, width, height, options);
        }

        // The blocks: an element's runs (its key the element), then the runs no element holds (one block, last).
        var blocks = new Dictionary<int, (int First, List<int> Runs)>();
        for (int r = 0; r < runs.Count; r++)
        {
            var tag = runs[r].Tag;
            if (tag is { IsArtifact: true })
            {
                continue;
            }

            int key = tag is { } t ? t.Element : int.MaxValue;
            int order = tag is { } u ? u.Order : int.MaxValue;
            if (blocks.TryGetValue(key, out var block))
            {
                block.Runs.Add(r);
                blocks[key] = (Math.Min(block.First, order), block.Runs);
            }
            else
            {
                blocks[key] = (order, [r]);
            }
        }

        var lines = new List<DocumentLine>();
        foreach (var (_, (_, members)) in blocks.OrderBy(b => b.Key == int.MaxValue ? 1 : 0).ThenBy(b => b.Value.First).ThenBy(b => b.Key))
        {
            members.Sort((a, b) => (runs[a].Tag?.Order ?? int.MaxValue).CompareTo(runs[b].Tag?.Order ?? int.MaxValue) is var c && c != 0 ? c : runs[a].Order.CompareTo(runs[b].Order));
            var own = new DocumentTextRun[members.Count];
            for (int i = 0; i < own.Length; i++)
            {
                own[i] = runs[members[i]];
            }

            foreach (var line in _lines.Arrange(own, width, height, options).Lines)
            {
                var mapped = new int[line.Runs.Count];
                for (int i = 0; i < mapped.Length; i++)
                {
                    mapped[i] = members[line.Runs[i]];
                }

                lines.Add(line with { Runs = mapped });
            }
        }

        return new DocumentText(lines, string.Join('\n', lines.Select(l => l.Text)));
    }
}
