// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;

namespace Idrak.Abstraction.Data;

/// <summary>The direction a layout gives a page's lines.</summary>
public enum DocumentTextDirection : byte
{
    /// <summary>Each line's own, from its first strong character; a line with none takes the page's.</summary>
    Auto,

    /// <summary>Every line left to right.</summary>
    LeftToRight,

    /// <summary>Every line right to left.</summary>
    RightToLeft,
}

/// <summary>
/// Arranges a page's text runs into lines and text, from their places on the page and the reading order the options name;
/// register one with <see cref="DocumentTextLayouts.Register"/>. Formats call the layout <see cref="DocumentOptions.Layout"/>
/// names (<see cref="DocumentTextLayouts.Arrange"/>). Safe to use from several threads at once.
/// </summary>
public interface IDocumentTextLayout : IKind
{
    /// <summary>
    /// The lines and text of a page of <paramref name="width"/> x <paramref name="height"/> points holding
    /// <paramref name="runs"/> (in content order).
    /// </summary>
    /// <exception cref="NotRegisteredException">The reading order the options name is not registered.</exception>
    DocumentText Arrange(IReadOnlyList<DocumentTextRun> runs, float width, float height, DocumentOptions options);
}

/// <summary>
/// The document text layouts, by name (ignoring case). The library's, in Idrak.Abstraction: <see cref="Lines"/> (runs
/// grouped into lines by baseline and size, each line's direction from its first strong character by Unicode's rule
/// through the reading order, runs ordered by their places in that direction, words split where a gap passes a share of the
/// font size, then the reading order for logical order) and <see cref="Content"/> (the runs in content order, joined, no
/// layout). A layout of one's own (a page's tagged structure, columns) registers with <see cref="Register"/>; one under a
/// library name takes its place, which stays behind it (see <see cref="SetPolicy"/>).
/// </summary>
public static class DocumentTextLayouts
{
    /// <summary>"lines": the runs laid out into lines by their places on the page.</summary>
    public const string Lines = "lines";

    /// <summary>"content": the runs in content order, joined, no layout.</summary>
    public const string Content = "content";

    private static readonly KindTable<IDocumentTextLayout> Table = new(nameof(DocumentTextLayouts), "Document text layout",
        $"add it with DocumentTextLayouts.Register (DocumentOptions.Layout and {DocumentOptions.LayoutVariable} name the layout)",
        (slot, app, library) => new GuardedLayout(slot, app, library), StringComparer.OrdinalIgnoreCase);

    static DocumentTextLayouts()
    {
        Table.RegisterDefault(new LinesLayout());
        Table.RegisterDefault(new ContentLayout());
        LibraryDefaults.Ensure(typeof(DocumentTextLayouts));
    }

    /// <summary>Registers <paramref name="layout"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IDocumentTextLayout layout) => Table.Register(layout, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's layout <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>The layout registered as <paramref name="name"/>, or null.</summary>
    public static IDocumentTextLayout? Find(ReadOnlySpan<char> name) => Table.Find(name);

    /// <summary>The layout registered as <paramref name="name"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotRegisteredException">None is: the message names the registered ones.</exception>
    public static IDocumentTextLayout Get(ReadOnlySpan<char> name) => Table.Get(name);

    /// <summary>The library's layout <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IDocumentTextLayout? Default(string name) => Table.Default(name);

    /// <summary>Who registered the layout <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>
    /// What happens when the app's layout <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/> arranges with the library's), or whether it only runs beside the library's
    /// (<see cref="SlotPolicy.Shadow"/>: the library's text is used, the two are compared).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>Each registered layout as "name: summary", one per line.</summary>
    public static string Describe() => Table.Describe();

    /// <summary>The lines and text of a page's runs, by the layout <paramref name="options"/> names.</summary>
    /// <exception cref="NotRegisteredException">The layout or the reading order it needs is not registered.</exception>
    public static DocumentText Arrange(IReadOnlyList<DocumentTextRun> runs, float width, float height, DocumentOptions options)
    {
        ArgumentNullException.ThrowIfNull(runs);
        ArgumentNullException.ThrowIfNull(options);
        return Get(options.Layout).Arrange(runs, width, height, options);
    }

    /// <summary>The reading order <paramref name="options"/> names; refused naming the option when it is not registered.</summary>
    /// <exception cref="NotRegisteredException">It is not registered.</exception>
    public static IReadingOrder ReadingOrderFor(DocumentOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return ReadingOrders.Find(options.ReadingOrder)
            ?? throw new NotRegisteredException(nameof(ReadingOrders), options.ReadingOrder, ReadingOrders.Names,
                $"A document's text is put in logical order by the reading order '{options.ReadingOrder}' (DocumentOptions.ReadingOrder, {DocumentOptions.ReadingOrderVariable}), "
                + $"which is not registered (registered: {(ReadingOrders.Names.Count == 0 ? "none" : string.Join(", ", ReadingOrders.Names))}); the library's (bidi, runs) come with "
                + "Idrak.Imaging, or add one with ReadingOrders.Register.");
    }

    // An app's layout over the library's: each call falls back on its own, or is compared with the library's text.
    private sealed class GuardedLayout(Slot slot, IDocumentTextLayout app, IDocumentTextLayout library) : IDocumentTextLayout
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public DocumentText Arrange(IReadOnlyList<DocumentTextRun> runs, float width, float height, DocumentOptions options) =>
            slot.Call(() => app.Arrange(runs, width, height, options), () => library.Arrange(runs, width, height, options),
                (a, b) => Comparisons.Difference(a.Text, b.Text) ?? Comparisons.Exact(a.Lines.Count, b.Lines.Count));
    }
}

// "lines": runs grouped into lines by baseline and size; each line's direction from its first strong character (the
// reading order's RightToLeft over its runs in content order), a line with none taking the page's; its runs ordered by
// their places in that direction, a space where the gap between two passes the word gap; the text met that way (the
// columns, read in the line's direction) put in logical order by the reading order. Runs without a place (an empty box)
// are lines of their own, in content order, after the placed ones.
internal sealed class LinesLayout : IDocumentTextLayout
{
    public string Name => DocumentTextLayouts.Lines;

    public string Summary => "runs grouped into lines by baseline and size, ordered by their places in each line's direction (its first strong character), words split at gaps, then the reading order";

    public DocumentText Arrange(IReadOnlyList<DocumentTextRun> runs, float width, float height, DocumentOptions options)
    {
        ArgumentNullException.ThrowIfNull(runs);
        ArgumentNullException.ThrowIfNull(options);
        if (runs.Count == 0)
        {
            return DocumentText.Empty;
        }

        var order = DocumentTextLayouts.ReadingOrderFor(options);
        var groups = Group(runs, options.LineTolerance);
        bool? page = null;
        bool pageKnown = false;
        var lines = new List<DocumentLine>(groups.Count);
        var builder = new StringBuilder();
        foreach (var group in groups)
        {
            bool rightToLeft;
            if (options.Direction != DocumentTextDirection.Auto)
            {
                rightToLeft = options.Direction == DocumentTextDirection.RightToLeft;
            }
            else if (order.RightToLeft(ContentText(runs, group)) is { } own)
            {
                rightToLeft = own;
            }
            else
            {
                if (!pageKnown)
                {
                    (page, pageKnown) = (order.RightToLeft(ContentText(runs, null)), true);
                }

                rightToLeft = page ?? false;
            }

            lines.Add(Line(runs, group, rightToLeft, order, options.WordGap, builder));
        }

        return new DocumentText(lines, string.Join('\n', lines.Select(l => l.Text)));
    }

    // The runs' indices in lines, top to bottom: placed runs sorted by baseline (then x, then content order) and swept,
    // a run joining the line when its baseline is within the tolerance of the line's largest run's; runs without a place
    // each a line, in content order. Runs with no text are left out.
    private static List<int[]> Group(IReadOnlyList<DocumentTextRun> runs, float tolerance)
    {
        var placed = new List<int>(runs.Count);
        var unplaced = new List<int>();
        for (int i = 0; i < runs.Count; i++)
        {
            if (runs[i].Text.Length == 0)
            {
                continue;
            }

            (runs[i].Box.IsEmpty ? unplaced : placed).Add(i);
        }

        var keys = new float[runs.Count];
        foreach (int i in placed)
        {
            keys[i] = runs[i].Baseline ?? runs[i].Box.Bottom;
        }

        placed.Sort((a, b) =>
        {
            int c = keys[a].CompareTo(keys[b]);
            c = c != 0 ? c : runs[a].Box.X.CompareTo(runs[b].Box.X);
            c = c != 0 ? c : runs[a].Order.CompareTo(runs[b].Order);
            return c != 0 ? c : a.CompareTo(b);
        });

        var groups = new List<int[]>();
        var current = new List<int>();
        float lineBaseline = 0f, lineSize = 0f;
        foreach (int i in placed)
        {
            float size = MathF.Max(runs[i].FontSize, 0f);
            if (current.Count > 0 && MathF.Abs(keys[i] - lineBaseline) <= tolerance * MathF.Max(size, lineSize))
            {
                current.Add(i);
                if (size > lineSize)
                {
                    (lineBaseline, lineSize) = (keys[i], size);
                }

                continue;
            }

            if (current.Count > 0)
            {
                groups.Add([.. current]);
                current.Clear();
            }

            current.Add(i);
            (lineBaseline, lineSize) = (keys[i], size);
        }

        if (current.Count > 0)
        {
            groups.Add([.. current]);
        }

        unplaced.Sort((a, b) => runs[a].Order != runs[b].Order ? runs[a].Order.CompareTo(runs[b].Order) : a.CompareTo(b));
        foreach (int i in unplaced)
        {
            groups.Add([i]);
        }

        return groups;
    }

    // The text of a line's runs (or the page's, when null) in content order: what the direction is read from.
    private static string ContentText(IReadOnlyList<DocumentTextRun> runs, int[]? group)
    {
        int[] indices = group is null ? [.. Enumerable.Range(0, runs.Count)] : [.. group];
        Array.Sort(indices, (a, b) => runs[a].Order != runs[b].Order ? runs[a].Order.CompareTo(runs[b].Order) : a.CompareTo(b));
        return string.Concat(indices.Select(i => runs[i].Text));
    }

    private static DocumentLine Line(IReadOnlyList<DocumentTextRun> runs, int[] group, bool rightToLeft, IReadingOrder order, float wordGap, StringBuilder columns)
    {
        // By place in the line's direction: from the left edge, or from the right edge for a right-to-left line.
        Array.Sort(group, (a, b) =>
        {
            int c = rightToLeft ? runs[b].Box.Right.CompareTo(runs[a].Box.Right) : runs[a].Box.X.CompareTo(runs[b].Box.X);
            c = c != 0 ? c : runs[a].Order.CompareTo(runs[b].Order);
            return c != 0 ? c : a.CompareTo(b);
        });

        columns.Clear();
        var box = DocumentBox.Empty;
        for (int k = 0; k < group.Length; k++)
        {
            var run = runs[group[k]];
            if (k > 0)
            {
                var previous = runs[group[k - 1]];
                float gap = rightToLeft ? previous.Box.X - run.Box.Right : run.Box.X - previous.Box.Right;
                if (gap > wordGap * MathF.Max(previous.FontSize, run.FontSize) && columns.Length > 0 && !char.IsWhiteSpace(columns[^1]) && !char.IsWhiteSpace(run.Text[0]))
                {
                    columns.Append(' ');
                }
            }

            // The run's characters as the line's columns meet them in its direction.
            columns.Append(run.Logical ? order.ToColumns(run.Text, rightToLeft) : rightToLeft ? Reversed(run.Text) : run.Text);
            box = box.Union(run.Box);
        }

        return new DocumentLine(order.FromColumns(columns.ToString(), rightToLeft), box, rightToLeft, group);
    }

    // A text read from its end: its text elements (a base and its marks, a surrogate pair) in reverse order, each kept whole.
    internal static string Reversed(string text)
    {
        if (text.Length < 2)
        {
            return text;
        }

        return string.Create(text.Length, text, static (destination, source) =>
        {
            int at = destination.Length;
            var rest = source.AsSpan();
            while (rest.Length > 0)
            {
                int length = Math.Max(1, StringInfo.GetNextTextElementLength(rest));
                at -= length;
                rest[..length].CopyTo(destination[at..]);
                rest = rest[length..];
            }
        });
    }
}

// "content": the runs in content order, their text joined as it is; one line per run (left to right), no layout.
internal sealed class ContentLayout : IDocumentTextLayout
{
    public string Name => DocumentTextLayouts.Content;

    public string Summary => "the runs in content order, joined as they are, one line per run; no layout";

    public DocumentText Arrange(IReadOnlyList<DocumentTextRun> runs, float width, float height, DocumentOptions options)
    {
        ArgumentNullException.ThrowIfNull(runs);
        if (runs.Count == 0)
        {
            return DocumentText.Empty;
        }

        int[] indices = [.. Enumerable.Range(0, runs.Count)];
        Array.Sort(indices, (a, b) => runs[a].Order != runs[b].Order ? runs[a].Order.CompareTo(runs[b].Order) : a.CompareTo(b));
        var lines = new DocumentLine[indices.Length];
        for (int k = 0; k < indices.Length; k++)
        {
            var run = runs[indices[k]];
            lines[k] = new DocumentLine(run.Text, run.Box, false, [indices[k]]);
        }

        return new DocumentText(lines, string.Concat(lines.Select(l => l.Text)));
    }
}
