// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Data.Pdf;

/// <summary>
/// A PDF document's navigation and structure (ISO 32000-2, 12.3 and 14.7): the bookmarks (<c>/Outlines</c>: titles,
/// destinations or web addresses, open or closed, colours and styles, nested), destinations (a page and how it is shown:
/// <c>/XYZ</c>, <c>/Fit</c>, <c>/FitH</c>, <c>/FitV</c>, <c>/FitR</c>, <c>/FitB</c>, <c>/FitBH</c>, <c>/FitBV</c>, in the
/// page's points as shown) and named destinations (the <c>/Names /Dests</c> name tree and the catalog's older
/// <c>/Dests</c>), page labels (the <c>/PageLabels</c> number tree), and a tagged file's structure tree (each marked
/// content's element and place in the structure's order). Each is read once a document, on first need; a damaged entry is
/// passed over, a limit or a cycle fails as the other limits do. A link's or a bookmark's target is data: an action other
/// than going to a place or a web address (a script, a launch) is not taken.
/// </summary>
internal sealed partial class PdfDocument
{
    private readonly Lock _navigationGate = new();
    private IReadOnlyList<DocumentOutline>? _outline;
    private IReadOnlyList<DocumentPageLabel>? _labels;
    private IReadOnlyList<DocumentDestination>? _destinations;
    private Dictionary<string, PdfValue>? _named;
    private Dictionary<int, int>? _pageIndices;
    private Dictionary<(int Page, int Mcid), DocumentTag>? _structure;
    private bool _labelsRead, _structureRead;

    public IReadOnlyList<DocumentOutline> Outline
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            lock (_navigationGate)
            {
                return _outline ??= Navigation("bookmarks", () => Items(Catalog(_file)["Outlines"u8])) ?? [];
            }
        }
    }

    public IReadOnlyList<DocumentDestination> Destinations
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            lock (_navigationGate)
            {
                return _destinations ??= Navigation<IReadOnlyList<DocumentDestination>>("named destinations", () =>
                    [.. Named().OrderBy(n => n.Key, StringComparer.Ordinal).Select(n => Destination(n.Value, 0) is { } d ? d with { Name = n.Key } : null).OfType<DocumentDestination>()]) ?? [];
            }
        }
    }

    public string? PageLabel(int index)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, _pages.Length);
        ObjectDisposedException.ThrowIf(_disposed, this);
        lock (_navigationGate)
        {
            if (!_labelsRead)
            {
                _labels = Navigation("page labels", Labels);
                _labelsRead = true;
            }
        }

        return _labels is { Count: > 0 } labels ? DocumentPageLabel.Find(labels, index) : null;
    }

    /// <summary>
    /// The page's marked content's tags (by <c>/MCID</c>) in a tagged file, for the content reader; null when the file has
    /// no structure tree.
    /// </summary>
    internal PdfMarks? Marks(PdfValue page)
    {
        lock (_navigationGate)
        {
            if (!_structureRead)
            {
                _structure = Navigation("structure tree", Structure);
                _structureRead = true;
            }
        }

        if (_structure is not { } structure)
        {
            return null;
        }

        return new PdfMarks(structure, page.IsReference ? page.AsReference.Number : -1);
    }

    /// <summary>
    /// The destination <paramref name="value"/> gives: an explicit one (an array: a page and how it is shown), a name (a
    /// string or a name, looked up in the named destinations: the result keeps the name), or a dictionary holding one in
    /// <c>/D</c>; null when it gives none this document holds.
    /// </summary>
    internal DocumentDestination? Destination(PdfValue value, int depth)
    {
        if (depth > 4)
        {
            return null;                                                       // a name naming a name naming ...: none
        }

        var resolved = Safe(() => _file.Resolve(value));
        string? name = resolved.AsString is { } text ? PdfText.Decode(text.Bytes) : resolved.AsName?.ToString();
        if (name is not null)
        {
            Dictionary<string, PdfValue> named;
            lock (_navigationGate)
            {
                named = Named();
            }

            return named.TryGetValue(name, out var target) && Destination(target, depth + 1) is { } found ? found with { Name = name } : new DocumentDestination(-1) { Name = name };
        }

        if (resolved.AsDictionary is { } dictionary && resolved.AsArray is null)
        {
            return dictionary.TryGet("D"u8, out var inner) ? Destination(inner, depth + 1) : null;
        }

        return resolved.AsArray is { Count: >= 1 } array ? Explicit(array) : null;
    }

    /// <summary>What an action does that a reader may follow: a web address (<c>/URI</c>) or a place (<c>/GoTo</c>); any other action is not taken (both null).</summary>
    internal (string? Uri, DocumentDestination? Destination) Action(PdfValue value)
    {
        if (Safe(() => _file.Dictionary(value)) is not { } action)
        {
            return (null, null);
        }

        var kind = Safe(() => _file.Resolve(action["S"u8])).AsName;
        if (kind is not null && kind.Is("URI"u8) && Safe(() => _file.Resolve(action["URI"u8])).AsString is { } uri)
        {
            return (PdfText.Latin1(uri.Bytes), null);                         // a URI is 7-bit ASCII (12.6.4.8)
        }

        return kind is not null && kind.Is("GoTo"u8) ? (null, Destination(action["D"u8], 0)) : (null, null);
    }

    /// <summary>The index of the page <paramref name="page"/> refers to (an object of the page tree, or a page number), or -1.</summary>
    internal int PageIndex(PdfValue page)
    {
        if (page.IsReference)
        {
            lock (_navigationGate)
            {
                if (_pageIndices is null)
                {
                    _pageIndices = [];
                    for (int i = _pages.Length - 1; i >= 0; i--)
                    {
                        if (_pages[i].Node.IsReference)
                        {
                            _pageIndices[_pages[i].Node.AsReference.Number] = i;
                        }
                    }
                }

                return _pageIndices.TryGetValue(page.AsReference.Number, out int index) ? index : -1;
            }
        }

        return page.TryInteger(out long number) && number >= 0 && number < _pages.Length ? (int)number : -1;
    }

    // An explicit destination: the page, then the fit and its numbers, in the page's points as shown.
    private DocumentDestination? Explicit(PdfArray array)
    {
        int page = PageIndex(array[0]);
        if (page < 0)
        {
            return null;
        }

        var display = Frame(page).Display;
        double? At(int i) => i < array.Count && Safe(() => _file.Resolve(array[i])).TryNumber(out double n) && double.IsFinite(n) ? n : null;
        (float X, float Y) Shown(double x, double y) => ((float)(display.A * x + display.C * y + display.E), (float)(display.B * x + display.D * y + display.F));
        var fit = array.Count > 1 ? Safe(() => _file.Resolve(array[1])).AsName : null;
        var destination = new DocumentDestination(page);
        if (fit is null || fit.Is("XYZ"u8))
        {
            double? left = At(2), top = At(3), zoom = At(4);
            var (x, y) = Shown(left ?? 0, top ?? 0);
            return destination with { X = left is null ? null : x, Y = top is null ? null : y, Zoom = zoom is > 0 ? (float)zoom.Value : null };
        }

        if (fit.Is("Fit"u8) || fit.Is("FitB"u8))
        {
            return destination with { Fit = fit.Is("Fit"u8) ? DocumentDestinationFit.Page : DocumentDestinationFit.Bounds };
        }

        if (fit.Is("FitH"u8) || fit.Is("FitBH"u8))
        {
            double? top = At(2);
            return destination with { Fit = fit.Is("FitH"u8) ? DocumentDestinationFit.Width : DocumentDestinationFit.BoundsWidth, Y = top is null ? null : Shown(0, top.Value).Y };
        }

        if (fit.Is("FitV"u8) || fit.Is("FitBV"u8))
        {
            double? left = At(2);
            return destination with { Fit = fit.Is("FitV"u8) ? DocumentDestinationFit.Height : DocumentDestinationFit.BoundsHeight, X = left is null ? null : Shown(left.Value, 0).X };
        }

        if (fit.Is("FitR"u8) && At(2) is { } l && At(3) is { } b && At(4) is { } r && At(5) is { } t)
        {
            var (x0, y0) = Shown(l, t);
            var (x1, y1) = Shown(r, b);
            return destination with { Fit = DocumentDestinationFit.Box, X = Math.Min(x0, x1), Y = Math.Min(y0, y1), Right = Math.Max(x0, x1), Bottom = Math.Max(y0, y1) };
        }

        return destination;
    }

    // The named destinations: the name tree's (/Names /Dests), then the catalog's /Dests dictionary for names it lacks.
    private Dictionary<string, PdfValue> Named()
    {
        if (_named is { } known)
        {
            return known;
        }

        var named = new Dictionary<string, PdfValue>(StringComparer.Ordinal);
        try
        {
            var catalog = Catalog(_file);
            if (Safe(() => _file.Dictionary(catalog["Names"u8])) is { } names)
            {
                PdfAttachments.Tree(_file, names["Dests"u8], (key, value) =>
                {
                    if (key is not null)
                    {
                        named.TryAdd(PdfText.Decode(key), value);
                    }
                });
            }

            if (Safe(() => _file.Dictionary(catalog["Dests"u8])) is { } dests)
            {
                for (int i = 0; i < dests.Count; i++)
                {
                    named.TryAdd(dests.KeyAt(i).ToString(), dests.ValueAt(i));
                }
            }
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            // A damaged catalog or tree: the names read so far.
        }

        return _named = named;
    }

    // The bookmarks under an item (its /First, then each /Next), each with its own; a bookmark met again ends its list.
    private List<DocumentOutline> Items(PdfValue root)
    {
        var visited = new HashSet<int>();
        int count = 0;
        List<DocumentOutline> Level(PdfValue first, int depth)
        {
            var items = new List<DocumentOutline>();
            if (depth > _file.MaxDepth)
            {
                throw PdfLimit.Error($"{_file.Name}: bookmarks nested deeper than {_file.MaxDepth} levels, the limit {nameof(DocumentOptions.MaxDepth)} (DocumentOptions.MaxDepth).");
            }

            for (var at = first; !at.IsNull;)
            {
                if (at.IsReference && !visited.Add(at.AsReference.Number))
                {
                    break;                                                     // a bookmark met again: a cycle, its list ends
                }

                if (++count > _file.MaxObjects)
                {
                    throw _file.ObjectsLimit(count, "the bookmarks");
                }

                var item = Safe(() => _file.Dictionary(at));
                if (item is null)
                {
                    break;
                }

                var (uri, destination) = item.Contains("Dest"u8) ? (null, Destination(item["Dest"u8], 0)) : Action(item["A"u8]);
                var children = item.Contains("First"u8) ? Level(item["First"u8], depth + 1) : [];
                long open = Safe(() => _file.Resolve(item["Count"u8])).TryInteger(out long c) ? c : 0;
                long style = Safe(() => _file.Resolve(item["F"u8])).TryInteger(out long f) ? f : 0;
                items.Add(new DocumentOutline(Safe(() => _file.Resolve(item["Title"u8])).AsString is { } title ? PdfText.Decode(title.Bytes) : "")
                {
                    Destination = destination, Uri = uri, Open = children.Count > 0 && open > 0, Colour = Colour(item["C"u8]),
                    Style = (DocumentOutlineStyle)(style & 3), Children = children,
                });
                at = item["Next"u8];
            }

            return items;
        }

        return Safe(() => _file.Dictionary(root)) is { } outlines ? Level(outlines["First"u8], 1) : [];
    }

    // The page labels' ranges, from the number tree's pairs (a start page and its label dictionary), by start.
    private IReadOnlyList<DocumentPageLabel>? Labels()
    {
        var root = Catalog(_file)["PageLabels"u8];
        if (root.IsNull)
        {
            return null;
        }

        var ranges = new SortedDictionary<int, DocumentPageLabel>();
        var visited = new HashSet<int>();
        var stack = new Stack<(PdfValue Node, int Depth)>();
        stack.Push((root, 1));
        while (stack.Count > 0)
        {
            var (value, depth) = stack.Pop();
            if ((value.IsReference && !visited.Add(value.AsReference.Number)) || Safe(() => _file.Dictionary(value)) is not { } node)
            {
                continue;
            }

            if (depth > _file.MaxDepth)
            {
                throw PdfLimit.Error($"{_file.Name}: the page labels' number tree is nested deeper than {_file.MaxDepth} levels, the limit {nameof(DocumentOptions.MaxDepth)} (DocumentOptions.MaxDepth).");
            }

            if (Safe(() => _file.Resolve(node["Nums"u8])).AsArray is { } pairs)
            {
                for (int i = 0; i + 1 < pairs.Count; i += 2)
                {
                    if (Safe(() => _file.Resolve(pairs[i])).TryInteger(out long start) && start >= 0 && start < _pages.Length
                        && Safe(() => _file.Dictionary(pairs[i + 1])) is { } label)
                    {
                        var style = Safe(() => _file.Resolve(label["S"u8])).AsName;
                        var numbering = style is null ? DocumentPageNumbering.None
                            : style.Is("D"u8) ? DocumentPageNumbering.Decimal : style.Is("R"u8) ? DocumentPageNumbering.UpperRoman
                            : style.Is("r"u8) ? DocumentPageNumbering.LowerRoman : style.Is("A"u8) ? DocumentPageNumbering.UpperLetters
                            : style.Is("a"u8) ? DocumentPageNumbering.LowerLetters : DocumentPageNumbering.None;
                        ranges[(int)start] = new DocumentPageLabel((int)start, numbering)
                        {
                            Prefix = Safe(() => _file.Resolve(label["P"u8])).AsString is { } prefix ? PdfText.Decode(prefix.Bytes) : null,
                            FirstNumber = Safe(() => _file.Resolve(label["St"u8])).TryInteger(out long first) && first is >= 1 and <= int.MaxValue ? (int)first : 1,
                        };
                    }
                }
            }

            if (Safe(() => _file.Resolve(node["Kids"u8])).AsArray is { } kids)
            {
                for (int i = kids.Count - 1; i >= 0; i--)
                {
                    stack.Push((kids[i], depth + 1));
                }
            }
        }

        return [.. ranges.Values];
    }

    // The structure tree walked once in its order: each element numbered, each marked content (an /MCID under an element,
    // or a marked-content reference) given its element's type and number and its own place, keyed by its page's object;
    // object references and elements met again are passed over. Null when the file has no structure tree.
    private Dictionary<(int Page, int Mcid), DocumentTag>? Structure()
    {
        var root = Safe(() => _file.Dictionary(Catalog(_file)["StructTreeRoot"u8]));
        if (root is null)
        {
            return null;
        }

        var marks = new Dictionary<(int Page, int Mcid), DocumentTag>();
        var visited = new HashSet<int>();
        var stack = new Stack<(PdfValue Node, int Page, string Type, int Element, int Depth)>();
        Push(stack, root["K"u8], -1, "", -1, 1);
        int elements = 0, order = 0;
        while (stack.Count > 0)
        {
            var (value, page, type, element, depth) = stack.Pop();
            if (value.IsReference && !visited.Add(value.AsReference.Number))
            {
                continue;
            }

            if (depth > _file.MaxDepth)
            {
                throw PdfLimit.Error($"{_file.Name}: the structure tree is nested deeper than {_file.MaxDepth} levels, the limit {nameof(DocumentOptions.MaxDepth)} (DocumentOptions.MaxDepth).");
            }

            var resolved = Safe(() => _file.Resolve(value));
            if (resolved.TryInteger(out long mcid))
            {
                if (element >= 0 && mcid is >= 0 and <= int.MaxValue)
                {
                    marks.TryAdd((page, (int)mcid), new DocumentTag(type, element, order++));
                }

                continue;
            }

            if (resolved.AsDictionary is not { } node)
            {
                continue;
            }

            var kind = Safe(() => _file.Resolve(node["Type"u8])).AsName;
            int own = node["Pg"u8].IsReference ? node["Pg"u8].AsReference.Number : page;
            if (kind is not null && kind.Is("MCR"u8))
            {
                if (element >= 0 && Safe(() => _file.Resolve(node["MCID"u8])).TryInteger(out long reference) && reference is >= 0 and <= int.MaxValue)
                {
                    marks.TryAdd((own, (int)reference), new DocumentTag(type, element, order++));
                }

                continue;
            }

            if (kind is not null && kind.Is("OBJR"u8))
            {
                continue;
            }

            if (++elements > _file.MaxObjects)
            {
                throw _file.ObjectsLimit(elements, "the structure tree");
            }

            string name = Safe(() => _file.Resolve(node["S"u8])).AsName?.ToString() ?? "";
            Push(stack, node["K"u8], own, name, elements - 1, depth + 1);
        }

        return marks;

        void Push(Stack<(PdfValue, int, string, int, int)> stack, PdfValue kids, int page, string type, int element, int depth)
        {
            var resolved = Safe(() => _file.Resolve(kids));
            if (resolved.AsArray is { } array)
            {
                for (int i = array.Count - 1; i >= 0; i--)
                {
                    stack.Push((array[i], page, type, element, depth));
                }
            }
            else if (!kids.IsNull)
            {
                stack.Push((kids, page, type, element, depth));
            }
        }
    }

    // A colour array: grey, RGB or CMYK; none (empty) or anything else is null.
    internal DocumentColour? Colour(PdfValue value)
    {
        if (Safe(() => _file.Resolve(value)).AsArray is not { } array)
        {
            return null;
        }

        Span<float> c = stackalloc float[4];
        for (int i = 0; i < Math.Min(4, array.Count); i++)
        {
            c[i] = Safe(() => _file.Resolve(array[i])).TryNumber(out double n) && double.IsFinite(n) ? (float)Math.Clamp(n, 0, 1) : 0f;
        }

        return array.Count switch
        {
            1 => DocumentColour.Grey(c[0]),
            3 => DocumentColour.Rgb(c[0], c[1], c[2]),
            4 => DocumentColour.Cmyk(c[0], c[1], c[2], c[3]),
            _ => null,
        };
    }

    // A part of the navigation read, its errors worded with the file's name: a damaged one is none, a limit fails.
    private T? Navigation<T>(string what, Func<T?> read) where T : class
    {
        try
        {
            return read();
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            return null;
        }
        catch (InvalidDataException e) when (!e.Message.StartsWith(_file.Name, StringComparison.Ordinal))
        {
            throw PdfLimit.Error(string.Create(CultureInfo.InvariantCulture, $"{_file.Name}: its {what}: {e.Message}"));
        }
    }

    // A value read where damage is passed over: a damaged object reads as null; a limit or a cycle still fails.
    private T Safe<T>(Func<T> read)
    {
        try
        {
            return read();
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            return default!;
        }
    }
}
