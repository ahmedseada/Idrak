// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Xml;

namespace Idrak.Data.Pdf;

/// <summary>One widget of a form field: its object, its page and rectangle, the state it shows and the state that shows it on.</summary>
internal sealed class PdfFormWidget(PdfValue node, PdfDictionary dictionary)
{
    /// <summary>Its object (a reference, or the dictionary itself when it is held inside another object).</summary>
    public PdfValue Node { get; } = node;

    public PdfDictionary Dictionary { get; } = dictionary;

    /// <summary>Its page's index (-1 when no page holds it).</summary>
    public int Page { get; set; } = -1;

    /// <summary>Its rectangle in its page's user space (left, bottom, right, top).</summary>
    public (double Left, double Bottom, double Right, double Top) Rect { get; set; }

    /// <summary>Its box on its page as shown (top-left origin).</summary>
    public DocumentBox Box { get; set; }

    /// <summary>The appearance state it shows (<c>/AS</c>), or null.</summary>
    public string? State { get; set; }

    /// <summary>The state that shows it on (a check box's or radio button's appearance state other than Off), or null.</summary>
    public string? OnState { get; set; }

    /// <summary>Whether it is not shown (its flags hide it).</summary>
    public bool Hidden { get; set; }

    /// <summary>How far its content is turned counter-clockwise (<c>/MK /R</c>): 0, 90, 180 or 270.</summary>
    public int Rotation { get; set; }

    /// <summary>Its width and height in user space.</summary>
    public (float Width, float Height) Size => ((float)Math.Abs(Rect.Right - Rect.Left), (float)Math.Abs(Rect.Top - Rect.Bottom));

    /// <summary>Its object's number, or 0 when it is held inside another object.</summary>
    public int Number => Node.IsReference ? Node.AsReference.Number : 0;
}

/// <summary>A terminal field of a form: what it holds (inherited where it takes it from an ancestor), its widgets and the contract's view.</summary>
internal sealed class PdfFormField(string name, PdfValue node, PdfDictionary dictionary, PdfValue parent)
{
    /// <summary>Its full, dotted name.</summary>
    public string Name { get; } = name;

    /// <summary>Its object (a reference, or the dictionary itself).</summary>
    public PdfValue Node { get; } = node;

    public PdfDictionary Dictionary { get; } = dictionary;

    /// <summary>Its parent field's object (null for a field at the top).</summary>
    public PdfValue Parent { get; } = parent;

    public DocumentFieldKind Kind { get; set; }

    /// <summary>Its flags as written (<c>/Ff</c>, inherited).</summary>
    public long Flags { get; set; }

    /// <summary>Its default appearance (<c>/DA</c>, inherited, else the form's).</summary>
    public byte[]? Appearance { get; set; }

    /// <summary>Its choices (<c>/Opt</c>, inherited): each one's export value and the text shown.</summary>
    public List<DocumentFieldOption> Options { get; } = [];

    /// <summary>Its widgets, in tree order (the field itself first when it is its own widget).</summary>
    public List<PdfFormWidget> Widgets { get; } = [];

    /// <summary>The contract's view of it.</summary>
    public DocumentField Field { get; set; } = null!;

    /// <summary>Its object's number, or 0 when it is held inside another object.</summary>
    public int Number => Node.IsReference ? Node.AsReference.Number : 0;
}

/// <summary>A node of a form's field tree (a field, or a field holding others).</summary>
/// <param name="Node">Its object.</param>
/// <param name="Parent">Its parent's object (null at the top).</param>
/// <param name="Dictionary">Its dictionary.</param>
/// <param name="Name">Its full, dotted name.</param>
/// <param name="Terminal">Whether it is a field with a value (its kids, if any, its widgets) rather than one holding fields.</param>
internal readonly record struct PdfFormNode(PdfValue Node, PdfValue Parent, PdfDictionary Dictionary, string Name, bool Terminal);

/// <summary>A document's form: its AcroForm dictionary, its fields and their tree, its resources, and its XFA data.</summary>
internal sealed class PdfForm
{
    public static PdfForm None { get; } = new(PdfValue.Null, null);

    public PdfForm(PdfValue value, PdfDictionary? dictionary) => (Value, Dictionary) = (value, dictionary);

    /// <summary>The catalog's <c>/AcroForm</c> (a reference or the dictionary), null when there is no form.</summary>
    public PdfValue Value { get; }

    public PdfDictionary? Dictionary { get; }

    /// <summary>The form's default resources (<c>/DR</c>), or null.</summary>
    public PdfDictionary? Resources { get; set; }

    /// <summary>Whether a reader is asked to make the fields' appearances (<c>/NeedAppearances</c>).</summary>
    public bool NeedAppearances { get; set; }

    /// <summary>The terminal fields, in tree order.</summary>
    public List<PdfFormField> Fields { get; } = [];

    /// <summary>Every node of the field tree, parents before their children.</summary>
    public List<PdfFormNode> Nodes { get; } = [];

    /// <summary>The XFA form's data values by path, in document order (none without an XFA form).</summary>
    public List<(string Path, string Value)> XfaData { get; } = [];

    /// <summary>The fields as the contract gives them: the terminal fields', then XFA data values no field holds.</summary>
    public IReadOnlyList<DocumentField> Public { get; set; } = [];

    /// <summary>The terminal field named <paramref name="name"/>, or null.</summary>
    public PdfFormField? Find(string name) => Fields.Find(f => string.Equals(f.Name, name, StringComparison.Ordinal));
}

/// <summary>
/// Reads a PDF document's form (ISO 32000-2, 12.7): the AcroForm's field tree from <c>/Fields</c> (kids walked without
/// recursion, a node met twice passed over, nesting within <see cref="DocumentOptions.MaxDepth"/>), each terminal field
/// with what it inherits (<c>/FT /Ff /V /DV /DA /Q /MaxLen /Opt /RV /DS</c>), its kind, value or values (a text field's
/// rich text as runs, <see cref="PdfRichText"/>), choices, flags, default appearance (font, size, colour), tooltip,
/// appearance characteristics (<c>/MK</c>) and border (<c>/BS</c>), and its widgets (the field itself when it is its own,
/// else its kids without a name): each one's page (by <c>/P</c>, else the page whose <c>/Annots</c> holds it), box as
/// shown, turn, appearance state and on state. An XFA form's <c>datasets</c> packet is read as data (XML, no DTD, nothing
/// outside fetched): each leaf's value by its path; values no AcroForm field holds become text fields of their own; its
/// template is read on its own (<see cref="PdfFormTemplates"/>). Its layout is not drawn. Pages' widget annotations come with their appearances read as
/// content of the page, with every other annotation (<see cref="PdfAnnotations.Read"/>, through
/// <see cref="AppearanceStream"/> and <see cref="Placement"/>).
/// </summary>
internal static class PdfForms
{
    // The flags (bit positions from 1) a field's /Ff gives, by kind.
    private const long ReadOnly = 1, Required = 1 << 1, NoExport = 1 << 2, Multiline = 1 << 12, Password = 1 << 13, NoToggleToOff = 1 << 14, Radio = 1 << 15,
        PushButton = 1 << 16, Combo = 1 << 17, Edit = 1 << 18, Sort = 1 << 19, MultiSelect = 1 << 21, DoNotSpellCheck = 1 << 22, DoNotScroll = 1 << 23,
        Comb = 1 << 24, RichText = 1 << 25, RadiosInUnison = 1 << 25;

    /// <summary>A text field's flag (in <c>/Ff</c>) that marks its value as rich text.</summary>
    public const long RichTextFlag = RichText;

    // An annotation's flags that hide it: Hidden and NoView.
    private const long HiddenFlags = 2 | 32;

    /// <summary>Whether the file's catalog names an XFA form (<c>/AcroForm /XFA</c>); a catalog that cannot be read holds none.</summary>
    public static bool HasXfa(PdfFile file)
    {
        try
        {
            return file.Dictionary(file.Trailer["Root"u8]) is { } catalog && file.Dictionary(catalog["AcroForm"u8]) is { } acro && !file.Resolve(acro["XFA"u8]).IsNull;
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            return false;
        }
    }

    /// <summary>
    /// The document's form; <see cref="PdfForm.None"/> when its catalog names none. The tree is walked once with no
    /// closure, what a node passes on copied as a value, and each page's frame read once for every widget on it.
    /// </summary>
    public static PdfForm Read(PdfDocument document)
    {
        var file = document.File;
        var catalog = file.Dictionary(file.Trailer["Root"u8]);
        var value = catalog?["AcroForm"u8] ?? PdfValue.Null;
        var acro = SafeDictionary(file, value);
        if (acro is null)
        {
            return PdfForm.None;
        }

        var form = new PdfForm(value, acro)
        {
            Resources = SafeDictionary(file, acro["DR"u8]),
            NeedAppearances = file.Resolve(acro["NeedAppearances"u8]).AsBoolean,
        };
        var top = new Passed(new Inherited { Appearance = file.Resolve(acro["DA"u8]).AsString?.Bytes });
        var pages = new PageFinder(document);
        var fields = SafeArray(file, acro["Fields"u8]) ?? PdfArray.Empty;
        var visited = new HashSet<int>();
        var stack = new Stack<Pending>(Math.Max(4, fields.Count));
        for (int i = fields.Count - 1; i >= 0; i--)
        {
            stack.Push(new Pending(fields[i], PdfValue.Null, "", top, 1));
        }

        while (stack.Count > 0)
        {
            var (node, parent, prefix, inherited, depth) = stack.Pop();
            if (depth > file.MaxDepth)
            {
                throw PdfLimit.Error($"{file.Name}: form fields nested deeper than {file.MaxDepth} levels, the limit {nameof(DocumentOptions.MaxDepth)} (DocumentOptions.MaxDepth).");
            }

            if (node.IsReference && !visited.Add(node.AsReference.Number))
            {
                continue;                                                      // a node met again (a cycle, or listed twice): read once
            }

            if (SafeDictionary(file, node) is not { } dictionary)
            {
                continue;
            }

            string? partial = Text(file, dictionary["T"u8]);
            string name = partial is null ? prefix : prefix.Length == 0 ? partial : string.Concat(prefix, ".", partial);
            var passed = inherited.Value;
            passed.Take(file, dictionary);
            var kids = SafeArray(file, dictionary["Kids"u8]);
            bool holdsFields = false;
            if (kids is not null)
            {
                foreach (var kid in kids.Items)
                {
                    if (SafeDictionary(file, kid) is { } k && k.Contains("T"u8))
                    {
                        holdsFields = true;
                        break;
                    }
                }
            }

            form.Nodes.Add(new PdfFormNode(node, parent, dictionary, name, !holdsFields));
            if (holdsFields)
            {
                var shared = new Passed(passed);                               // one copy for all its kids
                for (int i = kids!.Count - 1; i >= 0; i--)
                {
                    if (SafeDictionary(file, kids[i]) is { } k && k.Contains("T"u8))
                    {
                        stack.Push(new Pending(kids[i], node, name, shared, depth + 1));
                    }
                }

                continue;
            }

            if (passed.Type is null)
            {
                continue;                                                      // no field type anywhere above: not a field
            }

            var field = new PdfFormField(name, node, dictionary, parent) { Flags = passed.Flags, Appearance = passed.Appearance };
            field.Kind = Kind(passed.Type, passed.Flags);
            if (IsWidget(dictionary))
            {
                field.Widgets.Add(Widget(file, node, dictionary, pages));
            }

            if (kids is not null)
            {
                foreach (var kid in kids.Items)
                {
                    if (kid.IsReference && !visited.Add(kid.AsReference.Number))
                    {
                        continue;
                    }

                    if (SafeDictionary(file, kid) is { } widget && !widget.Contains("T"u8))
                    {
                        field.Widgets.Add(Widget(file, kid, widget, pages));
                    }
                }
            }

            Options(file, passed.Options, field);
            form.Fields.Add(field);
            field.Field = Contract(file, field, in passed);
        }

        if (!file.Resolve(acro["XFA"u8]).IsNull)
        {
            Xfa(file, acro["XFA"u8], form.XfaData);
        }

        // The terminal fields', then the XFA data values no field holds (by its name without the [0]s).
        List<(string Path, string Value)>? extras = null;
        if (form.XfaData.Count > 0)
        {
            var held = new HashSet<string>(StringComparer.Ordinal);
            foreach (var f in form.Fields)
            {
                held.Add(Unindexed(f.Name));
            }

            foreach (var entry in form.XfaData)
            {
                if (!held.Contains(entry.Path))
                {
                    (extras ??= []).Add(entry);
                }
            }
        }

        var all = new DocumentField[form.Fields.Count + (extras?.Count ?? 0)];
        for (int i = 0; i < form.Fields.Count; i++)
        {
            all[i] = form.Fields[i].Field;
        }

        for (int i = 0; extras is not null && i < extras.Count; i++)
        {
            var (path, data) = extras[i];
            all[form.Fields.Count + i] = new DocumentField(path, DocumentFieldKind.Text) { Value = data, Values = [data] };
        }

        form.Public = all;
        return form;
    }

    /// <summary>
    /// The map an appearance stream is drawn under (ISO 32000-2, 12.5.5): the stream's box through its own matrix, the
    /// bounds of that mapped onto <paramref name="rect"/> (scaled and moved); the form's own matrix is applied within. Null
    /// when the box is empty.
    /// </summary>
    public static PdfMatrix? Placement(PdfFile file, PdfStream stream, (double Left, double Bottom, double Right, double Top) rect)
    {
        if (PdfFunction.Numbers(file, stream.Dictionary["BBox"u8]) is not { Length: 4 } box)
        {
            return null;
        }

        var matrix = Matrix(file, stream.Dictionary["Matrix"u8]);
        double x0 = double.MaxValue, y0 = double.MaxValue, x1 = double.MinValue, y1 = double.MinValue;
        foreach (var (x, y) in new[] { (box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3]) })
        {
            double px = matrix.A * x + matrix.C * y + matrix.E, py = matrix.B * x + matrix.D * y + matrix.F;
            (x0, y0, x1, y1) = (Math.Min(x0, px), Math.Min(y0, py), Math.Max(x1, px), Math.Max(y1, py));
        }

        if (!(x1 - x0 > 0) || !(y1 - y0 > 0) || !double.IsFinite(x1 - x0) || !double.IsFinite(y1 - y0))
        {
            return null;
        }

        double sx = (rect.Right - rect.Left) / (x1 - x0), sy = (rect.Top - rect.Bottom) / (y1 - y0);
        return new PdfMatrix(sx, 0, 0, sy, rect.Left - sx * x0, rect.Bottom - sy * y0);
    }

    /// <summary>A widget's appearance stream for <paramref name="state"/> (its <c>/AP /N</c>: the stream itself, or the state's in a dictionary of states; the only one when it has no state), or null.</summary>
    public static PdfStream? AppearanceStream(PdfFile file, PdfDictionary widget, string? state)
    {
        var normal = SafeResolve(file, SafeDictionary(file, widget["AP"u8])?["N"u8] ?? PdfValue.Null);
        if (normal.AsStream is { } stream)
        {
            return stream;
        }

        if (normal.AsDictionary is not { } states)
        {
            return null;
        }

        if (state is not null)
        {
            int length = Encoding.UTF8.GetByteCount(state);
            Span<byte> key = length <= 128 ? stackalloc byte[128] : new byte[length];
            Encoding.UTF8.GetBytes(state, key);
            return SafeResolve(file, states[key[..length]]).AsStream;
        }

        return states.Count == 1 ? SafeResolve(file, states.ValueAt(0)).AsStream : null;
    }

    /// <summary>A widget's appearance state (<c>/AS</c>), or null.</summary>
    public static string? State(PdfFile file, PdfDictionary widget) => file.Resolve(widget["AS"u8]).AsName?.ToString();

    /// <summary>Whether an annotation's flags hide it (Hidden, NoView).</summary>
    public static bool Hides(PdfFile file, PdfDictionary annotation) => file.Resolve(annotation["F"u8]).TryInteger(out long flags) && (flags & HiddenFlags) != 0;

    /// <summary>An annotation's <c>/Rect</c> (its corners in any order), or null when it is not one.</summary>
    public static (double Left, double Bottom, double Right, double Top)? Rect(PdfFile file, PdfDictionary annotation)
    {
        if (file.Resolve(annotation["Rect"u8]).AsArray is not { Count: 4 } array
            || !Finite(file, array[0], out double x0) || !Finite(file, array[1], out double y0) || !Finite(file, array[2], out double x1) || !Finite(file, array[3], out double y1))
        {
            return null;
        }

        return (Math.Min(x0, x1), Math.Min(y0, y1), Math.Max(x0, x1), Math.Max(y0, y1));
    }

    /// <summary>A rectangle of a page's user space as a box on the page as shown (its corners through the display map, their bounds).</summary>
    public static DocumentBox Box((double Left, double Bottom, double Right, double Top) rect, PdfMatrix display)
    {
        double x0 = double.MaxValue, y0 = double.MaxValue, x1 = double.MinValue, y1 = double.MinValue;
        void Corner(double x, double y)
        {
            double px = display.A * x + display.C * y + display.E, py = display.B * x + display.D * y + display.F;
            (x0, y0, x1, y1) = (Math.Min(x0, px), Math.Min(y0, py), Math.Max(x1, px), Math.Max(y1, py));
        }

        Corner(rect.Left, rect.Bottom);
        Corner(rect.Right, rect.Bottom);
        Corner(rect.Right, rect.Top);
        Corner(rect.Left, rect.Top);
        return new DocumentBox((float)x0, (float)y0, (float)(x1 - x0), (float)(y1 - y0));
    }

    /// <summary>A widget's turn (<c>/MK /R</c>, counter-clockwise): 0, 90, 180 or 270 (any other value read as 0).</summary>
    public static int Rotation(PdfFile file, PdfDictionary widget)
    {
        if (SafeDictionary(file, widget["MK"u8]) is not { } mk || !file.Resolve(mk["R"u8]).TryInteger(out long r) || r % 90 != 0)
        {
            return 0;
        }

        return (int)((r % 360 + 360) % 360);
    }

    /// <summary>A widget's border (<c>/BS</c>): its style, its width (null when it states none) and a dashed one's dashes (none when it states none).</summary>
    public static (DocumentFieldBorderStyle Style, float? Width, float[] Dashes) Border(PdfFile file, PdfDictionary widget)
    {
        if (SafeDictionary(file, widget["BS"u8]) is not { } bs)
        {
            return (DocumentFieldBorderStyle.Solid, null, []);
        }

        var style = file.Resolve(bs["S"u8]).AsName is { } s
            ? s.Is("D"u8) ? DocumentFieldBorderStyle.Dashed : s.Is("B"u8) ? DocumentFieldBorderStyle.Beveled : s.Is("I"u8) ? DocumentFieldBorderStyle.Inset
            : s.Is("U"u8) ? DocumentFieldBorderStyle.Underline : DocumentFieldBorderStyle.Solid
            : DocumentFieldBorderStyle.Solid;
        float? width = file.Resolve(bs["W"u8]).TryNumber(out double w) && w >= 0 && w < 1000 ? (float)w : null;
        float[] dashes = [];
        if (SafeArray(file, bs["D"u8]) is { Count: > 0 and <= 16 } d)
        {
            var read = new float[d.Count];
            bool any = false;
            for (int i = 0; i < read.Length; i++)
            {
                if (!Finite(file, d[i], out double v) || v < 0 || v > 1000)
                {
                    read = null!;
                    break;
                }

                read[i] = (float)v;
                any |= v > 0;
            }

            dashes = read is not null && any ? read : [];
        }

        return (style, width, dashes);
    }

    /// <summary>A default appearance string's font resource, size and colour (<c>/Helv 11 Tf 0 g</c>): the last of each it sets.</summary>
    public static (string? Font, float Size, DocumentColour Colour) ParseAppearance(ReadOnlySpan<byte> appearance)
    {
        string? font = null;
        float size = 0;
        var colour = DocumentColour.Black;
        var lexer = new PdfContentLexer(appearance);
        Span<double> numbers = stackalloc double[4];
        int count = 0;
        string? name = null;
        Span<byte> buffer = stackalloc byte[128];
        while (true)
        {
            var token = lexer.Next(out int start, out int length, out double number);
            if (token == PdfToken.End)
            {
                break;
            }

            switch (token)
            {
                case PdfToken.Number:
                    if (count == numbers.Length)
                    {
                        numbers[..^1].CopyTo(numbers);                         // DA carries few operands: keep the last four
                        count--;
                    }

                    numbers[count++] = number;
                    break;
                case PdfToken.Name:
                    var raw = appearance.Slice(start, length);
                    int decoded = raw.Length <= buffer.Length ? PdfContentLexer.DecodeName(raw, buffer) : 0;
                    name = decoded > 0 ? PdfText.Utf8OrLatin1(buffer[..decoded]) : null;
                    count = 0;
                    break;
                case PdfToken.Keyword:
                    var word = appearance.Slice(start, length);
                    if (word.SequenceEqual("Tf"u8) && count >= 1)
                    {
                        (font, size) = (name, (float)Math.Clamp(numbers[count - 1], 0, 10000));
                    }
                    else if (word.SequenceEqual("g"u8) && count >= 1)
                    {
                        colour = DocumentColour.Grey(Unit(numbers[count - 1]));
                    }
                    else if (word.SequenceEqual("rg"u8) && count >= 3)
                    {
                        colour = DocumentColour.Rgb(Unit(numbers[count - 3]), Unit(numbers[count - 2]), Unit(numbers[count - 1]));
                    }
                    else if (word.SequenceEqual("k"u8) && count >= 4)
                    {
                        colour = DocumentColour.Cmyk(Unit(numbers[0]), Unit(numbers[1]), Unit(numbers[2]), Unit(numbers[3]));
                    }

                    count = 0;
                    break;
                default:
                    count = 0;
                    break;
            }
        }

        return (font, size, colour);
    }

    /// <summary>A colour array of an appearance characteristic (<c>/MK /BG</c>, <c>/BC</c>): grey, RGB or CMYK by its count; null when empty or none.</summary>
    public static DocumentColour? Colour(PdfFile file, PdfValue value)
    {
        var n = PdfFunction.Numbers(file, value);
        return n?.Length switch
        {
            1 => DocumentColour.Grey(Unit(n[0])),
            3 => DocumentColour.Rgb(Unit(n[0]), Unit(n[1]), Unit(n[2])),
            4 => DocumentColour.Cmyk(Unit(n[0]), Unit(n[1]), Unit(n[2]), Unit(n[3])),
            _ => null,
        };
    }

    /// <summary>A text string (PDFDocEncoding or UTF-16 with its mark), or null when the value is not a string.</summary>
    public static string? Text(PdfFile file, PdfValue value) => file.Resolve(value).AsString is { } s ? PdfText.Decode(s.Bytes) : null;

    /// <summary>A name without its <c>[n]</c> parts at index 0 (an XFA form's AcroForm twins: <c>order[0].item[1].product[0]</c> to <c>order.item[1].product</c>).</summary>
    public static string Unindexed(string name) => name.Replace("[0]", "", StringComparison.Ordinal);

    // ---- the field tree --------------------------------------------------------------------------------------------

    // A node waiting to be read: its object, its parent's, its parent's name and what its ancestors pass on.
    private readonly record struct Pending(PdfValue Node, PdfValue Parent, string Prefix, Passed Inherited, int Depth);

    // What a node holding fields passes on, kept once for all its kids.
    private sealed class Passed(Inherited value)
    {
        public Inherited Value { get; } = value;
    }

    // What a field takes from its ancestors, copied as a value from node to node (no allocation): the field type's name as
    // the document holds it, the flags, the value and default, the default appearance, the alignment, the length, the
    // choices, and the rich text and its default style.
    private struct Inherited
    {
        public byte[]? Appearance;
        public PdfName? Type;
        public long Flags;
        public PdfValue Value, Default, Options, RichValue, RichStyle;
        public int Alignment;
        public int? MaxLength;

        /// <summary>Whether a field node sets the default appearance (else it is the form's, which only text and choice fields take).</summary>
        public bool OwnAppearance;

        // What `node` sets of it, over what its ancestors passed.
        public void Take(PdfFile file, PdfDictionary node)
        {
            if (file.Resolve(node["FT"u8]).AsName is { } type)
            {
                Type = type;
            }

            if (file.Resolve(node["Ff"u8]).TryInteger(out long flags))
            {
                Flags = flags;
            }

            if (node.TryGet("V"u8, out var value) && !value.IsNull)
            {
                Value = value;
            }

            if (node.TryGet("DV"u8, out var defaultValue) && !defaultValue.IsNull)
            {
                Default = defaultValue;
            }

            if (file.Resolve(node["DA"u8]).AsString is { } appearance)
            {
                (Appearance, OwnAppearance) = (appearance.Bytes, true);
            }

            if (file.Resolve(node["Q"u8]).TryInteger(out long q))
            {
                Alignment = (int)Math.Clamp(q, 0, 2);
            }

            if (file.Resolve(node["MaxLen"u8]).TryInteger(out long max) && max >= 0)
            {
                MaxLength = (int)Math.Min(max, int.MaxValue);
            }

            if (node.TryGet("Opt"u8, out var options) && !options.IsNull)
            {
                Options = options;
            }

            if (node.TryGet("RV"u8, out var rich) && !rich.IsNull)
            {
                RichValue = rich;
            }

            if (node.TryGet("DS"u8, out var style) && !style.IsNull)
            {
                RichStyle = style;
            }
        }
    }

    // Each widget's page: by its /P (the pages' object numbers, mapped on the first need), else the page whose /Annots
    // lists it (mapped on the first widget whose /P fails); and each page's frame read once for all its widgets.
    private sealed class PageFinder(PdfDocument document)
    {
        private Dictionary<int, int>? _numbers, _annotated;
        private Dictionary<int, PdfMatrix>? _displays;
        private int _lastPage = -1;
        private PdfMatrix _lastDisplay;

        public int Find(PdfDictionary widget, int number)
        {
            if (widget["P"u8] is { IsReference: true } p && (_numbers ??= PageNumbers(document)).TryGetValue(p.AsReference.Number, out int index))
            {
                return index;
            }

            return number > 0 && (_annotated ??= Annotated(document)).TryGetValue(number, out int found) ? found : -1;
        }

        public PdfMatrix Display(int page)
        {
            if (page == _lastPage)
            {
                return _lastDisplay;
            }

            if (_displays is null || !_displays.TryGetValue(page, out var display))
            {
                display = document.Frame(page).Display;
                if (_lastPage >= 0)
                {
                    (_displays ??= [])[_lastPage] = _lastDisplay;              // a second page: the first one kept too
                }
            }

            (_lastPage, _lastDisplay) = (page, display);
            return display;
        }
    }

    private static DocumentFieldKind Kind(PdfName type, long flags) =>
        type.Is("Tx"u8) ? DocumentFieldKind.Text
        : type.Is("Btn"u8) ? (flags & PushButton) != 0 ? DocumentFieldKind.PushButton : (flags & Radio) != 0 ? DocumentFieldKind.Radio : DocumentFieldKind.CheckBox
        : type.Is("Ch"u8) ? (flags & Combo) != 0 ? DocumentFieldKind.Combo : DocumentFieldKind.List
        : type.Is("Sig"u8) ? DocumentFieldKind.Signature
        : DocumentFieldKind.Text;

    // Whether a field's dictionary is also its widget (a field of one widget written as one object).
    private static bool IsWidget(PdfDictionary dictionary) => dictionary["Subtype"u8].IsName("Widget"u8) || dictionary.Contains("Rect"u8);

    private static PdfFormWidget Widget(PdfFile file, PdfValue node, PdfDictionary dictionary, PageFinder pages)
    {
        var widget = new PdfFormWidget(node, dictionary)
        {
            State = State(file, dictionary), OnState = OnState(file, dictionary), Hidden = Hides(file, dictionary), Rotation = Rotation(file, dictionary),
        };
        widget.Page = pages.Find(dictionary, widget.Number);
        if (Rect(file, dictionary) is { } rect)
        {
            widget.Rect = rect;
            if (widget.Page >= 0)
            {
                widget.Box = Box(rect, pages.Display(widget.Page));
            }
        }

        return widget;
    }

    // The appearance state that shows a button on: a state of its normal (else down) appearances other than Off.
    private static string? OnState(PdfFile file, PdfDictionary widget)
    {
        if (SafeDictionary(file, widget["AP"u8]) is not { } ap)
        {
            return null;
        }

        return On(SafeResolve(file, ap["N"u8])) ?? On(SafeResolve(file, ap["D"u8]));

        // A dictionary of states (not a stream: a stream's dictionary holds no states): its first state other than Off.
        static string? On(PdfValue appearances)
        {
            if (appearances.Kind != PdfKind.Dictionary || appearances.AsDictionary is not { } states)
            {
                return null;
            }

            for (int i = 0; i < states.Count; i++)
            {
                if (!states.KeyAt(i).Is("Off"u8))
                {
                    return states.KeyAt(i).ToString();
                }
            }

            return null;
        }
    }

    // A choice field's /Opt (each item text, or [export text]); a button's /Opt gives its states' export values.
    private static void Options(PdfFile file, PdfValue value, PdfFormField field)
    {
        if (SafeArray(file, value) is not { } options)
        {
            return;
        }

        foreach (var item in options.Items)
        {
            var resolved = file.Resolve(item);
            if (resolved.AsArray is { Count: >= 2 } pair)
            {
                string export = Text(file, pair[0]) ?? "", shown = Text(file, pair[1]) ?? export;
                field.Options.Add(new DocumentFieldOption(shown, export));
            }
            else if (Text(file, resolved) is { } text)
            {
                field.Options.Add(new DocumentFieldOption(text, text));
            }
        }
    }

    // The contract's view of a terminal field.
    private static DocumentField Contract(PdfFile file, PdfFormField field, in Inherited passed)
    {
        // The form's default appearance is for fields of variable text (text and choice fields); others take their own.
        bool variable = field.Kind is DocumentFieldKind.Text or DocumentFieldKind.List or DocumentFieldKind.Combo;
        var (font, size, colour) = passed.Appearance is { } da && (passed.OwnAppearance || variable) ? ParseAppearance(da) : (null, 0f, DocumentColour.Black);
        var widgets = field.Widgets;
        var mk = widgets.Count > 0 ? SafeDictionary(file, widgets[0].Dictionary["MK"u8]) : null;
        string? caption = mk is null ? null : Text(file, mk["CA"u8]);
        string? value = null;
        IReadOnlyList<string> values = [];
        IReadOnlyList<DocumentFieldOption> options = field.Options;
        bool button = field.Kind is DocumentFieldKind.CheckBox or DocumentFieldKind.Radio;
        switch (field.Kind)
        {
            case DocumentFieldKind.Text:
                value = TextValue(file, passed.Value);
                values = value is null ? [] : [value];
                break;
            case DocumentFieldKind.CheckBox or DocumentFieldKind.Radio:
                string? state = file.Resolve(passed.Value).AsName?.ToString();
                for (int i = 0; state is null && i < widgets.Count; i++)
                {
                    state = widgets[i].State is { } s && s != "Off" ? s : null;
                }

                value = state is null or "Off" ? null : ButtonExport(state, field.Options);
                values = value is null ? [] : [value];
                if (field.Kind == DocumentFieldKind.Radio || field.Options.Count > 0)
                {
                    // Each on state once (in widget order), as its export value.
                    var on = new List<DocumentFieldOption>(widgets.Count);
                    for (int i = 0; i < widgets.Count; i++)
                    {
                        if (widgets[i].OnState is not { } s || Seen(widgets, i, s))
                        {
                            continue;
                        }

                        string export = ButtonExport(s, field.Options);
                        on.Add(new DocumentFieldOption(export, export));
                    }

                    options = on;
                }

                break;
            case DocumentFieldKind.List or DocumentFieldKind.Combo:
                var chosen = ChoiceValues(file, passed.Value);
                bool multi = (passed.Flags & MultiSelect) != 0;
                values = chosen;
                value = !multi && chosen.Count > 0 ? chosen[0] : null;
                break;
        }

        var shown = new DocumentWidget[widgets.Count];
        for (int i = 0; i < shown.Length; i++)
        {
            var w = widgets[i];
            shown[i] = new DocumentWidget(w.Page, w.Box, button ? w.State : null) { OnState = button ? w.OnState : null, Rotation = w.Rotation };
        }

        var (borderStyle, borderWidth, dashes) = widgets.Count > 0 ? Border(file, widgets[0].Dictionary) : (DocumentFieldBorderStyle.Solid, null, []);
        return new DocumentField(field.Name, field.Kind)
        {
            Value = value, Values = values, Options = options, Flags = Flags(field.Kind, passed.Flags), MaxLength = passed.MaxLength, Widgets = shown,
            Tooltip = Text(file, field.Dictionary["TU"u8]), DefaultValue = DefaultValue(file, passed.Default), Alignment = (DocumentFieldAlignment)passed.Alignment,
            Font = font, FontSize = size, TextColour = colour, Background = mk is null ? null : Colour(file, mk["BG"u8]), Border = mk is null ? null : Colour(file, mk["BC"u8]),
            Caption = field.Kind == DocumentFieldKind.PushButton ? caption : null, Mark = Mark(field.Kind, caption),
            BorderStyle = borderStyle, BorderWidth = borderWidth, Dashes = dashes,
            RichText = field.Kind == DocumentFieldKind.Text && !passed.RichValue.IsNull ? PdfRichText.Read(file, passed.RichValue, passed.RichStyle) : [],
        };

        // Whether an earlier widget has the same on state.
        static bool Seen(List<PdfFormWidget> widgets, int index, string state)
        {
            for (int k = 0; k < index; k++)
            {
                if (string.Equals(widgets[k].OnState, state, StringComparison.Ordinal))
                {
                    return true;
                }
            }

            return false;
        }
    }

    // A text value: a string, or a stream's text (a rich text value kept as a stream).
    private static string? TextValue(PdfFile file, PdfValue value)
    {
        var resolved = SafeResolve(file, value);
        if (resolved.AsString is { } s)
        {
            return PdfText.Decode(s.Bytes);
        }

        if (resolved.AsStream is { } stream)
        {
            using var decoded = file.Decode(stream);
            return PdfText.Decode(decoded.Span);
        }

        return null;
    }

    // A choice field's values: a string, or an array of strings.
    private static List<string> ChoiceValues(PdfFile file, PdfValue value)
    {
        var resolved = SafeResolve(file, value);
        var values = new List<string>();
        if (resolved.AsArray is { } array)
        {
            foreach (var item in array.Items)
            {
                if (Text(file, item) is { } text)
                {
                    values.Add(text);
                }
            }
        }
        else if (Text(file, resolved) is { } single)
        {
            values.Add(single);
        }

        return values;
    }

    private static string? DefaultValue(PdfFile file, PdfValue value)
    {
        var resolved = SafeResolve(file, value);
        return resolved.AsName?.ToString() ?? Text(file, resolved) ?? (resolved.AsArray is { Count: > 0 } array ? Text(file, array[0]) : null);
    }

    // A button state's export value: through /Opt when the state is an index into it (states named "0", "1", ...).
    private static string ButtonExport(string state, List<DocumentFieldOption> options) =>
        options.Count > 0 && int.TryParse(state, NumberStyles.None, CultureInfo.InvariantCulture, out int index) && index < options.Count ? options[index].Export : state;

    private static DocumentFieldFlags Flags(DocumentFieldKind kind, long ff)
    {
        var flags = DocumentFieldFlags.None;
        void Set(long bit, DocumentFieldFlags flag)
        {
            if ((ff & bit) != 0)
            {
                flags |= flag;
            }
        }

        Set(ReadOnly, DocumentFieldFlags.ReadOnly);
        Set(Required, DocumentFieldFlags.Required);
        Set(NoExport, DocumentFieldFlags.NoExport);
        switch (kind)
        {
            case DocumentFieldKind.Text:
                Set(Multiline, DocumentFieldFlags.Multiline);
                Set(Password, DocumentFieldFlags.Password);
                Set(DoNotSpellCheck, DocumentFieldFlags.DoNotSpellCheck);
                Set(DoNotScroll, DocumentFieldFlags.DoNotScroll);
                Set(Comb, DocumentFieldFlags.Comb);
                Set(RichText, DocumentFieldFlags.RichText);
                break;
            case DocumentFieldKind.CheckBox or DocumentFieldKind.Radio or DocumentFieldKind.PushButton:
                Set(NoToggleToOff, DocumentFieldFlags.NoToggleToOff);
                Set(RadiosInUnison, DocumentFieldFlags.RadiosInUnison);
                break;
            case DocumentFieldKind.List or DocumentFieldKind.Combo:
                Set(Edit, DocumentFieldFlags.Editable);
                Set(Sort, DocumentFieldFlags.Sort);
                Set(MultiSelect, DocumentFieldFlags.MultiSelect);
                Set(DoNotSpellCheck, DocumentFieldFlags.DoNotSpellCheck);
                break;
        }

        return flags;
    }

    /// <summary>The flags a field's /Ff holds for <paramref name="flags"/> of <paramref name="kind"/> (with the kind's own bits: radio, push button, combo).</summary>
    public static long FieldFlags(DocumentFieldKind kind, DocumentFieldFlags flags)
    {
        long ff = 0;
        void Set(DocumentFieldFlags flag, long bit)
        {
            if ((flags & flag) != 0)
            {
                ff |= bit;
            }
        }

        Set(DocumentFieldFlags.ReadOnly, ReadOnly);
        Set(DocumentFieldFlags.Required, Required);
        Set(DocumentFieldFlags.NoExport, NoExport);
        switch (kind)
        {
            case DocumentFieldKind.Text:
                Set(DocumentFieldFlags.Multiline, Multiline);
                Set(DocumentFieldFlags.Password, Password);
                Set(DocumentFieldFlags.DoNotSpellCheck, DoNotSpellCheck);
                Set(DocumentFieldFlags.DoNotScroll, DoNotScroll);
                Set(DocumentFieldFlags.Comb, Comb);
                Set(DocumentFieldFlags.RichText, RichText);
                break;
            case DocumentFieldKind.CheckBox:
                Set(DocumentFieldFlags.NoToggleToOff, NoToggleToOff);
                break;
            case DocumentFieldKind.Radio:
                ff |= Radio;
                Set(DocumentFieldFlags.NoToggleToOff, NoToggleToOff);
                Set(DocumentFieldFlags.RadiosInUnison, RadiosInUnison);
                break;
            case DocumentFieldKind.PushButton:
                ff |= PushButton;
                break;
            case DocumentFieldKind.Combo:
                ff |= Combo;
                Set(DocumentFieldFlags.Editable, Edit);
                Set(DocumentFieldFlags.Sort, Sort);
                Set(DocumentFieldFlags.DoNotSpellCheck, DoNotSpellCheck);
                break;
            case DocumentFieldKind.List:
                Set(DocumentFieldFlags.Sort, Sort);
                Set(DocumentFieldFlags.MultiSelect, MultiSelect);
                Set(DocumentFieldFlags.DoNotSpellCheck, DoNotSpellCheck);
                break;
        }

        return ff;
    }

    // A button's mark by its caption's ZapfDingbats character (/MK /CA): a check box's check mark and a radio button's
    // circle unless it names another.
    private static DocumentFieldMark Mark(DocumentFieldKind kind, string? caption) => caption switch
    {
        "4" => DocumentFieldMark.Check,
        "l" => DocumentFieldMark.Circle,
        "8" => DocumentFieldMark.Cross,
        "u" => DocumentFieldMark.Diamond,
        "n" => DocumentFieldMark.Square,
        "H" => DocumentFieldMark.Star,
        _ => kind == DocumentFieldKind.Radio ? DocumentFieldMark.Circle : DocumentFieldMark.Check,
    };

    /// <summary>The ZapfDingbats character a viewer draws <paramref name="mark"/> with (<c>/MK /CA</c>).</summary>
    public static string MarkCharacter(DocumentFieldMark mark) => mark switch
    {
        DocumentFieldMark.Circle => "l",
        DocumentFieldMark.Cross => "8",
        DocumentFieldMark.Diamond => "u",
        DocumentFieldMark.Square => "n",
        DocumentFieldMark.Star => "H",
        _ => "4",
    };

    // Each page's object number to its index.
    private static Dictionary<int, int> PageNumbers(PdfDocument document)
    {
        var pages = new Dictionary<int, int>();
        for (int i = 0; i < document.PageCount; i++)
        {
            int number = document.PageNumber(i);
            if (number > 0)
            {
                pages.TryAdd(number, i);
            }
        }

        return pages;
    }

    // Each annotation object the pages' /Annots list, to its page (the first page that lists it).
    private static Dictionary<int, int> Annotated(PdfDocument document)
    {
        var file = document.File;
        var found = new Dictionary<int, int>();
        for (int i = 0; i < document.PageCount; i++)
        {
            if (SafeDictionary(file, document.PageObject(i).Node) is not { } page || SafeArray(file, page["Annots"u8]) is not { } annots)
            {
                continue;
            }

            foreach (var item in annots.Items)
            {
                if (item.IsReference)
                {
                    found.TryAdd(item.AsReference.Number, i);
                }
            }
        }

        return found;
    }

    private static PdfMatrix Matrix(PdfFile file, PdfValue value) =>
        PdfFunction.Numbers(file, value) is { Length: 6 } m && m.All(double.IsFinite) ? new PdfMatrix(m[0], m[1], m[2], m[3], m[4], m[5]) : PdfMatrix.Identity;

    private static float Unit(double value) => (float)Math.Clamp(double.IsFinite(value) ? value : 0, 0, 1);

    /// <summary>A value resolved; null when it cannot be read: a part of the form that cannot be read is passed over (a limit passed still throws).</summary>
    public static PdfValue SafeResolve(PdfFile file, PdfValue value)
    {
        try
        {
            return file.Resolve(value);
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            return PdfValue.Null;
        }
    }

    /// <summary>A value's dictionary; null when it is none or cannot be read (a limit passed still throws).</summary>
    public static PdfDictionary? SafeDictionary(PdfFile file, PdfValue value) => SafeResolve(file, value).AsDictionary;

    /// <summary>A value's array; null when it is none or cannot be read (a limit passed still throws).</summary>
    public static PdfArray? SafeArray(PdfFile file, PdfValue value) => SafeResolve(file, value).AsArray;

    // A finite number.
    private static bool Finite(PdfFile file, PdfValue value, out double number) => file.Resolve(value).TryNumber(out number) && double.IsFinite(number);

    // ---- XFA ---------------------------------------------------------------------------------------------------------

    // The datasets packet's data values: /XFA an array of packet names and streams, or one stream of the whole XDP.
    private static void Xfa(PdfFile file, PdfValue value, List<(string Path, string Value)> data)
    {
        if (Packet(file, value, "datasets", "<xfa:datasets"u8, "</xfa:datasets>"u8) is not { } datasets)
        {
            return;
        }

        try
        {
            ReadData(datasets, file.MaxStreamBytes, file.MaxDepth, data);
        }
        catch (XmlException)
        {
            data.Clear();                                                      // damaged data: none read
        }
    }

    /// <summary>
    /// An XFA packet's bytes: from <c>/XFA</c>'s array of packet names and streams the stream named <paramref name="name"/>,
    /// or from one stream of the whole XDP the element from <paramref name="open"/> to <paramref name="close"/>; null when
    /// there is none.
    /// </summary>
    public static byte[]? Packet(PdfFile file, PdfValue value, string name, ReadOnlySpan<byte> open, ReadOnlySpan<byte> close)
    {
        var resolved = SafeResolve(file, value);
        if (resolved.AsArray is { } packets)
        {
            for (int i = 0; i + 1 < packets.Count; i++, i++)
            {
                if (Text(file, packets[i]) == name && SafeResolve(file, packets[i + 1]).AsStream is { } stream)
                {
                    using var decoded = file.Decode(stream);
                    return decoded.Span.ToArray();
                }
            }
        }
        else if (resolved.AsStream is { } whole)
        {
            using var decoded = file.Decode(whole);
            var span = decoded.Span;
            int start = Opening(span, open), end = span.IndexOf(close);
            if (start >= 0 && end > start)
            {
                return span[start..(end + close.Length)].ToArray();
            }
        }

        return null;
    }

    // Where an element opens: its name followed by a space, a '>' or a '/' (not a longer name sharing its start).
    private static int Opening(ReadOnlySpan<byte> xml, ReadOnlySpan<byte> open)
    {
        int from = 0;
        while (from < xml.Length && xml[from..].IndexOf(open) is >= 0 and var at)
        {
            int next = from + at + open.Length;
            if (next < xml.Length && xml[next] is (byte)' ' or (byte)'>' or (byte)'/' or (byte)'\t' or (byte)'\n' or (byte)'\r')
            {
                return from + at;
            }

            from = next;
        }

        return -1;
    }

    // The leaves under xfa:data, each by its elements' names joined by dots (a name that repeats among its siblings
    // indexed from its second, [1], as XFA's SOM names it), the leaf's text its value.
    private static void ReadData(byte[] xml, long maxBytes, int maxDepth, List<(string Path, string Value)> data)
    {
        var settings = new XmlReaderSettings
        {
            DtdProcessing = DtdProcessing.Prohibit, XmlResolver = null, IgnoreComments = true, IgnoreProcessingInstructions = true,
            MaxCharactersInDocument = Math.Max(1, maxBytes), MaxCharactersFromEntities = 1024,
        };
        using var stream = new MemoryStream(xml, writable: false);
        using var reader = XmlReader.Create(stream, settings);
        var path = new List<string>();
        var counts = new List<Dictionary<string, int>>();
        var hasChildren = new List<bool>();
        var text = new StringBuilder();
        int dataDepth = -1;
        while (reader.Read())
        {
            switch (reader.NodeType)
            {
                case XmlNodeType.Element:
                    if (dataDepth < 0)
                    {
                        if (reader.LocalName == "data" && !reader.IsEmptyElement)
                        {
                            dataDepth = reader.Depth;
                            counts.Add(new Dictionary<string, int>(StringComparer.Ordinal));   // the top elements' names
                        }

                        break;
                    }

                    if (reader.Depth - dataDepth > maxDepth)
                    {
                        throw PdfLimit.Error($"XFA data nested deeper than {maxDepth} levels, the limit {nameof(DocumentOptions.MaxDepth)} (DocumentOptions.MaxDepth).");
                    }

                    if (hasChildren.Count > 0)
                    {
                        hasChildren[^1] = true;
                    }

                    var siblings = counts[^1];
                    siblings.TryGetValue(reader.LocalName, out int seen);
                    siblings[reader.LocalName] = seen + 1;

                    path.Add(seen == 0 ? reader.LocalName : string.Create(CultureInfo.InvariantCulture, $"{reader.LocalName}[{seen}]"));
                    text.Clear();
                    if (reader.IsEmptyElement)
                    {
                        data.Add((string.Join('.', path), ""));
                        path.RemoveAt(path.Count - 1);
                        break;
                    }

                    counts.Add(new Dictionary<string, int>(StringComparer.Ordinal));
                    hasChildren.Add(false);
                    break;
                case XmlNodeType.Text or XmlNodeType.CDATA or XmlNodeType.SignificantWhitespace or XmlNodeType.Whitespace:
                    if (dataDepth >= 0 && path.Count > 0)
                    {
                        text.Append(reader.Value);
                    }

                    break;
                case XmlNodeType.EndElement:
                    if (dataDepth < 0)
                    {
                        break;
                    }

                    if (path.Count == 0)
                    {
                        return;                                                // xfa:data ends
                    }

                    if (!hasChildren[^1])
                    {
                        data.Add((string.Join('.', path), text.ToString()));
                    }

                    path.RemoveAt(path.Count - 1);
                    counts.RemoveAt(counts.Count - 1);
                    hasChildren.RemoveAt(hasChildren.Count - 1);
                    text.Clear();
                    break;
            }
        }
    }
}
