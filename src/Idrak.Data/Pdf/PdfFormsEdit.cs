// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Data.Pdf;

/// <summary>
/// The fonts a field's text may be shown in, chosen per text: the field's own simple font when it has a code for every
/// character; else faces that show it: <see cref="DocumentFormOptions.Faces"/> and the faces of the fonts folder
/// (<see cref="DocumentFormOptions.Fonts"/>, read once when first needed), and the field's embedded font program, those named
/// as the field's font first, the first that shows every character leading and the others behind it for what it lacks.
/// Nothing here names a script or a language: a font shows a text or it does not.
/// </summary>
internal sealed class PdfFieldFonts(DocumentFormOptions options)
{
    private IReadOnlyList<IFontFace>? _folder;

    // Reading a folder through the font formats is not on the new design yet.
    private static IReadOnlyList<IFontFace> ReadFolder(string folder) =>
        throw new NotSupportedException($"{folder}: reading fonts through the registered font formats is not on the new design yet.");

    /// <summary>The faces of the fonts folder (none when it is not set or not there).</summary>
    public IReadOnlyList<IFontFace> Folder => _folder ??= options.Fonts is { Length: > 0 } path && Directory.Exists(path) ? ReadFolder(path) : [];

    /// <summary>
    /// Faces that together show <paramref name="text"/>, the one that shows it alone (or the most of it) first, those named
    /// <paramref name="name"/> before the others; null when they cannot.
    /// </summary>
    public IReadOnlyList<IFontFace>? Faces(string text, string? name, IFontFace? own)
    {
        var all = new List<IFontFace>();
        string wanted = Plain(name);
        void Add(IFontFace face)
        {
            if (!all.Exists(f => ReferenceEquals(f, face)))
            {
                all.Add(face);
            }
        }

        if (wanted.Length > 0)
        {
            foreach (var face in options.Faces.Concat(Folder))
            {
                if (Named(face, wanted))
                {
                    Add(face);
                }
            }
        }

        if (own is not null)
        {
            Add(own);
        }

        foreach (var face in options.Faces.Concat(Folder))
        {
            Add(face);
        }

        if (all.Count == 0)
        {
            return null;
        }

        var missing = all.Select(f => Missing(f, text)).ToList();
        int alone = missing.IndexOf(0);
        if (alone >= 0)
        {
            var ordered = new List<IFontFace> { all[alone] };
            ordered.AddRange(all.Where((_, i) => i != alone));
            return ordered;
        }

        // None alone: together, each character in some face?
        foreach (var rune in text.EnumerateRunes())
        {
            if (Shows(rune.Value) && !all.Exists(f => f.Glyph(rune.Value) != 0))
            {
                return null;
            }
        }

        return [.. all.Select((f, i) => (f, i)).OrderBy(p => missing[p.i]).ThenBy(p => p.i).Select(p => p.f)];
    }

    /// <summary>The first visible character of <paramref name="text"/> that <paramref name="shown"/> says is not shown (for a message), or -1.</summary>
    public static int FirstMissing(string text, Func<int, bool> shown)
    {
        foreach (var rune in text.EnumerateRunes())
        {
            if (Shows(rune.Value) && !shown(rune.Value))
            {
                return rune.Value;
            }
        }

        return -1;
    }

    // How many of the text's visible characters the face lacks.
    private static int Missing(IFontFace face, string text)
    {
        int missing = 0;
        foreach (var rune in text.EnumerateRunes())
        {
            if (Shows(rune.Value) && face.Glyph(rune.Value) == 0)
            {
                missing++;
            }
        }

        return missing;
    }

    // Whether a character needs a glyph (spaces and controls do not).
    private static bool Shows(int codePoint) => !(codePoint <= 0x20 || codePoint is 0x7F or 0xA0 || (codePoint < 0x10000 && char.IsWhiteSpace((char)codePoint)) || Rune.GetUnicodeCategory(new Rune(codePoint)) is UnicodeCategory.Format or UnicodeCategory.Control);

    // A font's name as letters and digits, lower case, its subset tag dropped ("ABCDEF+DejaVuSans" to "dejavusans").
    private static string Plain(string? name)
    {
        if (string.IsNullOrEmpty(name))
        {
            return "";
        }

        if (name.Length > 7 && name[6] == '+')
        {
            name = name[7..];
        }

        var plain = new StringBuilder(name.Length);
        foreach (char c in name)
        {
            if (char.IsLetterOrDigit(c))
            {
                plain.Append(char.ToLowerInvariant(c));
            }
        }

        return plain.ToString();
    }

    private static bool Named(IFontFace face, string wanted)
    {
        string family = Plain(face.Family);
        return family.Length > 0 && (wanted == family || wanted == family + Plain(face.Style));
    }
}

/// <summary>
/// A form XObject made for a widget, numbered when the document is written (an appearance a field's new value, or an added
/// field, shows).
/// </summary>
internal sealed class PdfMadeAppearance(PdfAppearance appearance)
{
    public PdfAppearance Appearance { get; } = appearance;

    /// <summary>Its object number (0 until the document is written).</summary>
    public int Number { get; set; }
}

/// <summary>A widget as it now shows: a source appearance stream, or one made, and its state.</summary>
internal sealed record PdfShownAppearance(PdfValue Source, PdfMadeAppearance? Made, string? State);

/// <summary>
/// The form changes of a PDF editor (C23): fields filled (<see cref="SetField"/>, and as rich text <see cref="SetRichText"/>:
/// each widget's appearance made to show the value, turned as its <c>/MK /R</c> says and bordered as its <c>/BS</c> says;
/// a check box's or radio button's state switched, its appearances made when it has none for the state; a text no simple
/// font of the field's shows set in a face, which the field's <c>/DA</c> then names, added to the form's <c>/DR</c>),
/// fields added (<see cref="AddField"/>) and fields flattened (<see cref="Flatten"/>: each widget's appearance, as it shows
/// now or made when it has none, drawn over its page as a form XObject; the widgets left out of their pages' annotations,
/// the fields out of the tree). Written by the editor's save: an incremental update writes the changed objects again under
/// their numbers and the new ones after; a rewrite writes them where the copier reaches them
/// (<see cref="PdfEditCopier.Source.Replaced"/>). A field or widget held inside another object (a dictionary written in its
/// parent's <c>/Kids</c> or the form's <c>/Fields</c>) is moved to an object of its own when it changes: its holder written
/// again naming it, its widgets' <c>/Parent</c> naming it, a widget added to its page's annotations. A form's catalog entry
/// is written again when its tree changes (the XFA packets then dropped: the AcroForm is what is shown); the form's own
/// object, when it is one, when only its fields' places or its resources change. Not safe from several threads.
/// </summary>
internal sealed class PdfFormEditor
{
    private readonly PdfEditor _editor;
    private readonly PdfDocument _source;
    private readonly PdfForm _form;
    private readonly DocumentWriteOptions _options;
    private readonly PdfFieldFonts _fonts;
    private readonly Dictionary<int, ObjectEdit> _edits = [];
    private readonly List<PdfMadeAppearance> _made = [];
    private readonly Dictionary<PdfFormWidget, PdfShownAppearance> _shown = [];
    private readonly Dictionary<PdfDictionary, ObjectEdit> _moved = new(ReferenceEqualityComparer.Instance);
    private readonly HashSet<PdfDictionary> _removedDirect = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<PdfWriterFont, string> _resourceNames = [];
    private readonly HashSet<string> _flattened = new(StringComparer.Ordinal);
    private readonly HashSet<int> _removedNodes = [];
    private readonly List<PdfNewField> _added = [];
    private readonly List<PdfNewNode> _addedTop = [];
    private readonly List<PdfNewNode> _addedNodes = [];
    private readonly Dictionary<string, (int Number, int Standard)> _standardFonts = new(StringComparer.Ordinal);
    private readonly HashSet<PdfWriterFont> _resourceFonts = [];
    private bool _treeChanged, _acroFormInline;
    private int _acroFormNumber;

    public PdfFormEditor(PdfEditor editor, PdfDocument source, DocumentWriteOptions options)
    {
        (_editor, _source, _options) = (editor, source, options);
        _form = source.Form;
        _fonts = new PdfFieldFonts(options.Forms);
    }

    /// <summary>Whether the catalog's form entry is written again (fields flattened or added; a form held in the catalog whose fields move or whose resources grow).</summary>
    public bool CatalogChanged => _treeChanged || _acroFormInline;

    // ---- filling -----------------------------------------------------------------------------------------------------

    /// <summary>Sets field <paramref name="name"/>'s value and makes each of its widgets show it.</summary>
    public void SetField(string name, ReadOnlySpan<string> values)
    {
        ArgumentNullException.ThrowIfNull(name);
        var field = Find(name);
        var given = values.ToArray();
        var look = Look(field) with { Values = Checked(field, given) };
        var node = Edit(field.Node, field.Dictionary);
        switch (field.Kind)
        {
            case DocumentFieldKind.Text:
                if (look.Values.Count == 0)
                {
                    node.Remove("V");
                }
                else
                {
                    string text = look.Values[0];
                    node.Set("V", (s, _, _) => s.Text(text));
                }

                if ((field.Field.Flags & DocumentFieldFlags.RichText) != 0 || field.Field.RichText.Count > 0)
                {
                    // Rich text: the plain value is its one run, so a reader showing /RV shows the value set.
                    if (look.Values.Count == 0)
                    {
                        node.Remove("RV");
                    }
                    else
                    {
                        string rich = PdfRichText.Write([new DocumentFieldRun(look.Values[0])]);
                        node.Set("RV", (s, _, _) => s.Text(rich));
                    }
                }

                break;
            case DocumentFieldKind.CheckBox or DocumentFieldKind.Radio:
                string state = look.Values.Count == 0 ? "Off" : StateOf(field, look.Values[0]);
                node.Set("V", (s, _, _) => s.Name(state));
                break;
            case DocumentFieldKind.List or DocumentFieldKind.Combo:
                var chosen = look.Values;
                bool multi = (field.Field.Flags & DocumentFieldFlags.MultiSelect) != 0;
                if (chosen.Count == 0)
                {
                    node.Remove("V");
                }
                else if (multi)
                {
                    node.Set("V", (s, _, _) =>
                    {
                        s.Byte((byte)'[');
                        foreach (string v in chosen)
                        {
                            s.Text(v);
                        }

                        s.Byte((byte)']');
                    });
                }
                else
                {
                    string v = chosen[0];
                    node.Set("V", (s, _, _) => s.Text(v));
                }

                var indices = chosen.Select(v => field.Field.Options.ToList().FindIndex(o => o.Export == v)).Where(i => i >= 0).Order().ToList();
                if (field.Kind == DocumentFieldKind.List && indices.Count > 0)
                {
                    node.Set("I", (s, _, _) => Integers(s, indices));
                }
                else
                {
                    node.Remove("I");
                }

                break;
        }

        Show(field, look, node);
    }

    /// <summary>
    /// Sets text field <paramref name="name"/>'s value as rich text: <c>/RV</c> written from <paramref name="runs"/>, <c>/V</c>
    /// their plain text, the field marked rich text, each widget's appearance made with the runs' styles.
    /// </summary>
    public void SetRichText(string name, IReadOnlyList<DocumentFieldRun> runs)
    {
        ArgumentNullException.ThrowIfNull(name);
        ArgumentNullException.ThrowIfNull(runs);
        var field = Find(name);
        if (field.Kind != DocumentFieldKind.Text)
        {
            throw new ArgumentException($"Field '{name}' is a {field.Kind} field: rich text is a text field's value.", nameof(name));
        }

        PdfNewField.CheckRuns(runs, name);
        string plain = PdfRichText.Plain(runs);
        var look = Look(field) with { Values = Checked(field, [plain]) };
        if (look.Values.Count > 0)
        {
            look = look with { Rich = runs };
        }

        var node = Edit(field.Node, field.Dictionary);
        if (look.Values.Count == 0)
        {
            node.Remove("V");
            node.Remove("RV");
        }
        else
        {
            string rich = PdfRichText.Write(runs);
            node.Set("V", (s, _, _) => s.Text(plain));
            node.Set("RV", (s, _, _) => s.Text(rich));
        }

        if ((field.Flags & PdfForms.RichTextFlag) == 0)
        {
            long flags = field.Flags | PdfForms.RichTextFlag;
            node.Set("Ff", (s, _, _) => s.Integer(flags));
        }

        Show(field, look, node);
    }

    // Each widget made to show the look: a button's state switched (both its appearances made when it has none for the
    // state), any other's appearance made. A text set in a face (no simple font of the field's shows it): the field's
    // default appearance names that face's font, which the form's resources list.
    private void Show(PdfFormField field, PdfFieldLook look, ObjectEdit node)
    {
        var file = _source.File;
        PdfFieldFont? font = null;
        foreach (var widget in field.Widgets)
        {
            var edit = Edit(widget.Node, widget.Dictionary);
            if (field.Kind is DocumentFieldKind.CheckBox or DocumentFieldKind.Radio)
            {
                string value = look.Values.Count == 0 ? "Off" : StateOf(field, look.Values[0]);
                string shown = widget.OnState is not null && widget.OnState == value ? value : "Off";
                edit.Set("AS", (s, _, _) => s.Name(shown));
                if (PdfForms.AppearanceStream(file, widget.Dictionary, shown) is not null && Reference(file, widget.Dictionary, shown) is { } reference)
                {
                    _shown[widget] = new PdfShownAppearance(reference, null, shown);
                    continue;
                }

                // No appearance for the state: both made, the on state named as the widget's own (or the value).
                string on = widget.OnState ?? value;
                var made = MakeCheck(look, on: true, widget);
                var off = MakeCheck(look, on: false, widget);
                edit.Set("AP", (s, _, _) => s.Raw("<</N<<"u8).Name(on).Space().Reference(made.Number).Raw("/Off "u8).Reference(off.Number).Raw(">>>>"u8));
                _shown[widget] = new PdfShownAppearance(PdfValue.Null, shown == "Off" ? off : made, shown);
                continue;
            }

            var appearance = Make(field, look, widget, font ??= Font(field, look));
            edit.Set("AP", (s, _, _) => s.Raw("<</N "u8).Reference(appearance.Number).Raw(">>"u8));
            _shown[widget] = new PdfShownAppearance(PdfValue.Null, appearance, null);
        }

        if (font is { IsSimple: false, Faces: [var face, ..] } && field.Kind is DocumentFieldKind.Text or DocumentFieldKind.List or DocumentFieldKind.Combo)
        {
            var used = _editor.Font(face);
            _resourceFonts.Add(used);
            byte[] appearance = Named(field.Appearance, ResourceName(used));
            node.Set("DA", (s, _, _) => s.Literal(appearance));
            FormChanged(fields: false, resources: true);
        }
    }

    // A default appearance naming `resource` as its font (the font operand of its last Tf replaced; one added before it
    // when it sets none), the rest as it was.
    private static byte[] Named(byte[]? appearance, string resource)
    {
        byte[] da = appearance ?? "/Helv 0 Tf 0 g"u8.ToArray();
        var lexer = new PdfContentLexer(da);
        int nameStart = -1, nameEnd = -1, fontStart = -1, fontEnd = -1;
        while (true)
        {
            var token = lexer.Next(out int start, out int length, out _);
            if (token == PdfToken.End)
            {
                break;
            }

            if (token == PdfToken.Name)
            {
                (nameStart, nameEnd) = (start - 1, start + length);            // the lexer's start is past the name's '/'
            }
            else if (token == PdfToken.Keyword && da.AsSpan(start, length).SequenceEqual("Tf"u8) && nameStart >= 0)
            {
                (fontStart, fontEnd) = (nameStart, nameEnd);
            }
        }

        byte[] name = Encoding.ASCII.GetBytes("/" + resource);
        if (fontStart < 0)
        {
            return [.. name, .. " 0 Tf "u8.ToArray(), .. da];
        }

        return [.. da.AsSpan(0, fontStart).ToArray(), .. name, .. da.AsSpan(fontEnd).ToArray()];
    }

    // ---- adding ------------------------------------------------------------------------------------------------------

    /// <summary>Adds <paramref name="field"/>: its nodes under the tree's (a parent named by its dotted name's start), its widgets on the edited document's pages.</summary>
    public void AddField(DocumentField field, int pageCount)
    {
        PdfNewField.Check(field, pageCount);
        if (_form.Find(field.Name) is not null || _added.Exists(a => a.Field.Name == field.Name))
        {
            throw new ArgumentException($"A field '{field.Name}' is there already: a name is one field's.", nameof(field));
        }

        string[] parts = field.Name.Split('.');
        PdfValue sourceParent = PdfValue.Null;
        PdfNewNode? newParent = null;
        string prefix = "";
        for (int i = 0; i < parts.Length - 1; i++)
        {
            prefix = prefix.Length == 0 ? parts[i] : prefix + "." + parts[i];
            if (newParent is null && _form.Nodes.FindLast(n => n.Name == prefix && !Removed(n.Node)) is { Dictionary: not null } existing)
            {
                if (existing.Terminal)
                {
                    throw new ArgumentException($"'{prefix}' is a field with a value: it cannot hold the field '{field.Name}'.", nameof(field));
                }

                sourceParent = existing.Node;
                continue;
            }

            var made = _addedNodes.Find(n => n.Name == prefix);
            if (made is null)
            {
                made = new PdfNewNode(prefix, parts[i]) { Parent = newParent, SourceParent = newParent is null ? sourceParent : PdfValue.Null };
                _addedNodes.Add(made);
                Attach(made, newParent, sourceParent);
            }

            newParent = made;
        }

        var added = new PdfNewField(field, parts[^1], newParent, sourceParent);
        added.Look(_fonts, field, _editor.Font, _options.Forms, StandardFont, ResourceName);
        _added.Add(added);
        Attach(added.Node, newParent, sourceParent);
        foreach (var widget in added.Widgets)
        {
            var page = _editor.Pages[widget.Page];
            (page.Annots ??= new PdfAnnotsEdit()).Added.Add(() => widget.Number);
            widget.Rect = Rect(widget.Box, page);
            widget.EditPage = page;
        }

        if (added.Fonts is { } used)
        {
            _resourceFonts.UnionWith(used);
        }

        _treeChanged = true;
    }

    // A new node joins its parent: a new node's kids, a source node's /Kids (written again), or the form's top.
    private void Attach(PdfNewNode node, PdfNewNode? newParent, PdfValue sourceParent)
    {
        if (newParent is not null)
        {
            newParent.Kids.Add(node);
            return;
        }

        if (!sourceParent.IsNull && _source.File.Dictionary(sourceParent) is { } parent)
        {
            var edit = Edit(sourceParent, parent);
            edit.Added.Add(node);
            edit.Set("Kids", (s, copier, source) => Kids(s, copier, source, parent, edit.Added));
            if (edit.Moved)
            {
                node.HeldParent = () => edit.Number;                           // a parent held inside another object, moved: named by its number
            }

            return;
        }

        _addedTop.Add(node);
    }

    // A source parent's /Kids: its own (those not flattened; one moved to an object of its own by its number) and the
    // nodes added under it.
    private void Kids(PdfSyntax s, PdfEditCopier copier, PdfEditCopier.Source source, PdfDictionary parent, List<PdfNewNode> added)
    {
        s.Byte((byte)'[');
        foreach (var kid in _source.File.Resolve(parent["Kids"u8]).AsArray?.Items ?? [])
        {
            Item(s, copier, source, kid);
        }

        foreach (var node in added)
        {
            s.Reference(node.Number).Space();
        }

        s.Byte((byte)']');
    }

    // A standard font a field added uses: a font object of the update's, named in the form's resources (one for each).
    private PdfFieldFont StandardFont(string resource, int standard)
    {
        if (!_standardFonts.ContainsKey(resource))
        {
            _standardFonts[resource] = (0, standard);
        }

        return PdfFieldFont.Standard(Encoding.ASCII.GetBytes(resource), standard, 0);
    }

    // ---- flattening --------------------------------------------------------------------------------------------------

    /// <summary>
    /// Flattens the fields <paramref name="names"/> (every field when none): each visible widget's appearance (as it
    /// shows now, made when it has none) drawn over its page, the widgets left out of their pages, the fields out of the tree.
    /// </summary>
    public void Flatten(ReadOnlySpan<string> names)
    {
        var fields = new List<PdfFormField>();
        if (names.IsEmpty)
        {
            fields.AddRange(_form.Fields.Where(f => !_flattened.Contains(f.Name)));
        }

        foreach (string name in names)
        {
            ArgumentNullException.ThrowIfNull(name);
            var found = _form.Fields.Where(f => !_flattened.Contains(f.Name) && (f.Name == name || f.Name.StartsWith(name + ".", StringComparison.Ordinal))).ToList();
            if (found.Count == 0)
            {
                throw NoField(name);
            }

            fields.AddRange(found.Where(f => !fields.Contains(f)));
        }

        var file = _source.File;
        foreach (var field in fields)
        {
            foreach (var widget in field.Widgets)
            {
                var pages = EditorPages(widget.Page);
                if (widget.Hidden || pages.Count == 0)
                {
                    Remove(widget, pages);
                    continue;
                }

                var shown = Shown(field, widget);
                if (shown is not null)
                {
                    var made = shown.Made;
                    PdfMatrix? placement = made is not null
                        ? new PdfMatrix(1, 0, 0, 1, widget.Rect.Left, widget.Rect.Bottom)
                        : file.Resolve(shown.Source).AsStream is { } stream ? PdfForms.Placement(file, stream, widget.Rect) : null;
                    if (placement is { } at)
                    {
                        foreach (int page in pages)
                        {
                            ((PdfPageBuilder)_editor.Overlay(page)).Form(shown.Source, made is null ? null : () => made.Number, at);
                        }
                    }
                }

                Remove(widget, pages);
            }

            _flattened.Add(field.Name);
            RemoveNode(field.Node, field.Dictionary);
            _treeChanged = true;
        }

        // Parents left with no field: removed too, up the tree.
        bool removed = true;
        while (removed)
        {
            removed = false;
            foreach (var node in _form.Nodes)
            {
                if (node.Terminal || Removed(node.Node))
                {
                    continue;
                }

                var kids = _form.Nodes.Where(n => Same(n.Parent, node.Node)).ToList();
                bool added = Edited(node.Node) is { Added.Count: > 0 };
                if (kids.Count > 0 && kids.All(k => Removed(k.Node)) && !added)
                {
                    RemoveNode(node.Node, node.Dictionary);
                    removed = true;
                }
            }
        }

        // Parents that keep fields: their /Kids written again without the removed.
        foreach (var node in _form.Nodes)
        {
            if (node.Terminal || Removed(node.Node))
            {
                continue;
            }

            var dictionary = node.Dictionary;
            if ((file.Resolve(dictionary["Kids"u8]).AsArray?.Items ?? []).Any(Removed))
            {
                var edit = Edit(node.Node, dictionary);
                edit.Set("Kids", (s, copier, source) => Kids(s, copier, source, dictionary, edit.Added));
            }
        }
    }

    // A node of the tree removed with its field: by its number, or, held inside another object, by its dictionary (and
    // its move to an object of its own undone: no longer listed on its pages).
    private void RemoveNode(PdfValue node, PdfDictionary dictionary)
    {
        if (node.IsReference)
        {
            _removedNodes.Add(node.AsReference.Number);
            return;
        }

        _removedDirect.Add(dictionary);
        if (_moved.TryGetValue(dictionary, out var moved))
        {
            foreach (var (page, number) in moved.Listed)
            {
                _editor.Pages[page].Annots?.Added.Remove(number);
            }
        }
    }

    // Whether a node (a reference, or a dictionary held inside another object) is removed.
    private bool Removed(PdfValue node) =>
        node.IsReference ? _removedNodes.Contains(node.AsReference.Number) : node.AsDictionary is { } d && _removedDirect.Contains(d);

    // Whether two nodes are the same object (the same number, or the same dictionary held inside another object).
    private static bool Same(PdfValue a, PdfValue b) =>
        a.IsReference ? b.IsReference && a.AsReference.Number == b.AsReference.Number : !b.IsReference && a.AsDictionary is { } d && ReferenceEquals(d, b.AsDictionary);

    // A node's edit when it has one.
    private ObjectEdit? Edited(PdfValue node) =>
        node.IsReference ? _edits.GetValueOrDefault(node.AsReference.Number) : node.AsDictionary is { } d ? _moved.GetValueOrDefault(d) : null;

    // What a widget shows now: a value set's appearance, else the source's for its state; one made when it has none.
    private PdfShownAppearance? Shown(PdfFormField field, PdfFormWidget widget)
    {
        var file = _source.File;
        if (_shown.TryGetValue(widget, out var known))
        {
            return known;
        }

        if (PdfForms.AppearanceStream(file, widget.Dictionary, widget.State) is not null && Reference(file, widget.Dictionary, widget.State) is { } reference)
        {
            return new PdfShownAppearance(reference, null, widget.State);
        }

        // None (a form that asks the reader to make them): made from the field's value now.
        var look = Look(field);
        if (field.Kind is DocumentFieldKind.CheckBox or DocumentFieldKind.Radio)
        {
            bool on = widget.State is { } state && state != "Off";
            return new PdfShownAppearance(PdfValue.Null, MakeCheck(look, on, widget), widget.State);
        }

        if (field.Field.RichText.Count > 0 && look.Values.Count > 0)
        {
            look = look with { Rich = field.Field.RichText };                  // a rich text value shown styled
        }

        return new PdfShownAppearance(PdfValue.Null, Make(field, look, widget, Font(field, look)), null);
    }

    // A flattened widget out of its pages' annotations (and out of the tree, with its field).
    private void Remove(PdfFormWidget widget, List<int> pages)
    {
        if (widget.Number == 0)
        {
            RemoveNode(widget.Node, widget.Dictionary);                        // held inside its field: out with it
            return;
        }

        foreach (int page in pages)
        {
            (_editor.Pages[page].Annots ??= new PdfAnnotsEdit()).Removed.Add(widget.Number);
        }

        _removedNodes.Add(widget.Number);
    }

    // The editor's pages that show page `index` of the source.
    private List<int> EditorPages(int index)
    {
        var pages = new List<int>();
        if (index < 0)
        {
            return pages;
        }

        for (int i = 0; i < _editor.Pages.Count; i++)
        {
            if (_editor.Pages[i].Document == _source && _editor.Pages[i].Index == index)
            {
                pages.Add(i);
            }
        }

        return pages;
    }

    // ---- writing -----------------------------------------------------------------------------------------------------

    /// <summary>
    /// Numbers what the changes make (appearances, fields and widgets added, standard fonts) and, for a rewrite, tells the
    /// copier which source objects it writes anew and which it leaves out; called before anything refers to them.
    /// </summary>
    public void Prepare(PdfFileWriter file, PdfEditCopier copier, PdfEditCopier.Source own, bool rewrite)
    {
        foreach (var made in _made)
        {
            made.Number = file.Allocate();
        }

        foreach (var (dictionary, moved) in _moved)
        {
            if (!_removedDirect.Contains(dictionary))
            {
                moved.Number = file.Allocate();
            }
        }

        foreach (var node in _addedNodes)
        {
            node.Number = file.Allocate();
        }

        foreach (var added in _added)
        {
            added.Allocate(file);
        }

        foreach (string key in _standardFonts.Keys.ToList())
        {
            _standardFonts[key] = (file.Allocate(), _standardFonts[key].Standard);
        }

        if (!rewrite)
        {
            return;
        }

        foreach (var (number, edit) in _edits)
        {
            if (number > 0 && !_removedNodes.Contains(number) && !(_treeChanged && number == _acroFormNumber))
            {
                own.Replaced[number] = s => edit.Write(s, copier, own);
            }
        }

        own.Dropped.UnionWith(_removedNodes);
    }

    /// <summary>
    /// Writes what the changes made: appearances, standard fonts, the fields and widgets added (a widget's page by
    /// <paramref name="pageNumber"/>), and, in an update, every source object changed again under its own number.
    /// </summary>
    public void Write(PdfFileWriter file, PdfEditCopier copier, PdfEditCopier.Source own, bool rewrite, Func<PdfNewWidget, (int Number, int Generation)> pageNumber)
    {
        foreach (var made in _made)
        {
            made.Appearance.Write(file, made.Number, (s, font) => SimpleFont(s, copier, own, font), (s, v) => copier.Value(s, own, v));
        }

        foreach (var (_, (number, standard)) in _standardFonts)
        {
            PdfNewField.WriteStandardFont(file, number, standard);
        }

        foreach (var node in _addedNodes)
        {
            copier.WriteObject(node.Number, s => node.Write(s, copier, own));
        }

        foreach (var added in _added)
        {
            added.Write(file, copier, own, pageNumber, (s, font) => SimpleFont(s, copier, own, font));
        }

        foreach (var (dictionary, moved) in _moved)
        {
            if (!_removedDirect.Contains(dictionary))
            {
                copier.WriteObject(moved.Number, s => moved.Write(s, copier, own));   // held inside another object: one of its own now
            }
        }

        if (rewrite)
        {
            return;
        }

        foreach (var (number, edit) in _edits)
        {
            if (number < 0 || _removedNodes.Contains(number) || (_treeChanged && number == _acroFormNumber))
            {
                continue;
            }

            file.Reuse(number, edit.Generation);
            copier.WriteObject(number, s => edit.Write(s, copier, own));
        }
    }

    /// <summary>
    /// The catalog's form entry as it now stands: the source's (its other entries kept; its XFA packets dropped when its
    /// tree changed) with its fields those not flattened (one moved to an object of its own by its number) and those added,
    /// the standard and embedded fonts fields name in its resources; nothing when no field is left.
    /// </summary>
    public void WriteAcroForm(PdfSyntax s, PdfEditCopier copier, PdfEditCopier.Source own)
    {
        var file = _source.File;
        var kept = (_form.Dictionary is { } acro ? file.Resolve(acro["Fields"u8]).AsArray?.Items ?? [] : [])
            .Where(f => !Removed(f) && !(f.IsReference && own.Removed.Contains(f.AsReference.Number))).ToList();
        if (kept.Count == 0 && _addedTop.Count == 0)
        {
            return;                                                            // no field left: no form
        }

        s.Raw("/AcroForm<</Fields"u8);
        Fields(s, copier, own);
        bool resources = false;
        var dictionary = _form.Dictionary ?? PdfDictionary.Empty;
        for (int i = 0; i < dictionary.Count; i++)
        {
            var key = dictionary.KeyAt(i);
            if (key.Is("Fields"u8) || (_treeChanged && key.Is("XFA"u8)))
            {
                continue;
            }

            if (key.Is("DR"u8))
            {
                resources = true;
                Resources(s, copier, own, dictionary.ValueAt(i));
                continue;
            }

            copier.Entry(s, own, key.Bytes, dictionary.ValueAt(i));
        }

        if (!resources && (_standardFonts.Count > 0 || _resourceFonts.Count > 0))
        {
            Resources(s, copier, own, PdfValue.Null);
        }

        if (!dictionary.Contains("DA"u8) && _addedTop.Count > 0)
        {
            s.Raw("/DA"u8).Literal("/Helv 0 Tf 0 g"u8);
        }

        s.Raw(">>"u8);
    }

    // The form's /Fields: the source's not flattened (one held inside it moved to an object of its own by its number),
    // then the fields added at the top.
    private void Fields(PdfSyntax s, PdfEditCopier copier, PdfEditCopier.Source own)
    {
        s.Byte((byte)'[');
        foreach (var field in _form.Dictionary is { } acro ? _source.File.Resolve(acro["Fields"u8]).AsArray?.Items ?? [] : [])
        {
            if (!(field.IsReference && own.Removed.Contains(field.AsReference.Number)))
            {
                Item(s, copier, own, field);
            }
        }

        foreach (var node in _addedTop)
        {
            s.Reference(node.Number).Space();
        }

        s.Byte((byte)']');
    }

    // An item of a list of nodes (/Kids, /Fields) and a space: none when it was removed (flattened, or redacted with every
    // mention of it), a dictionary moved to an object of its own by its number, else as it is.
    private void Item(PdfSyntax s, PdfEditCopier copier, PdfEditCopier.Source source, PdfValue item)
    {
        if (Removed(item) || (item.IsReference && source.Removed.Contains(item.AsReference.Number)))
        {
            return;
        }

        if (!item.IsReference && item.AsDictionary is { } held && _moved.TryGetValue(held, out var moved))
        {
            s.Reference(moved.Number).Space();
            return;
        }

        copier.Value(s, source, item);
        s.Space();
    }

    // The form's resources with the fonts added fields name added to its /Font.
    private void Resources(PdfSyntax s, PdfEditCopier copier, PdfEditCopier.Source own, PdfValue value)
    {
        if (_standardFonts.Count == 0 && _resourceFonts.Count == 0)
        {
            copier.Entry(s, own, "DR"u8, value);
            return;
        }

        s.Raw("/DR"u8);
        ResourcesValue(s, copier, own, value);
    }

    // The resources' dictionary: the source's entries, its /Font with the fonts added (the source's of the same name left out).
    private void ResourcesValue(PdfSyntax s, PdfEditCopier copier, PdfEditCopier.Source own, PdfValue value)
    {
        var file = _source.File;
        var dictionary = file.Dictionary(value) ?? PdfDictionary.Empty;
        s.Raw("<<"u8);
        bool fonts = false;
        for (int i = 0; i < dictionary.Count; i++)
        {
            var key = dictionary.KeyAt(i);
            if (!key.Is("Font"u8))
            {
                copier.Entry(s, own, key.Bytes, dictionary.ValueAt(i));
                continue;
            }

            fonts = true;
            s.Raw("/Font<<"u8);
            var own2 = file.Dictionary(dictionary.ValueAt(i)) ?? PdfDictionary.Empty;
            for (int k = 0; k < own2.Count; k++)
            {
                if (!_standardFonts.ContainsKey(own2.KeyAt(k).ToString()))
                {
                    copier.Entry(s, own, own2.KeyAt(k).Bytes, own2.ValueAt(k));
                }
            }

            AddedFonts(s);
            s.Raw(">>"u8);
        }

        if (!fonts)
        {
            s.Raw("/Font<<"u8);
            AddedFonts(s);
            s.Raw(">>"u8);
        }

        s.Raw(">>"u8);
    }

    private void AddedFonts(PdfSyntax s)
    {
        foreach (var (name, (number, _)) in _standardFonts)
        {
            s.Name(name).Space().Reference(number);
        }

        foreach (var font in _resourceFonts)
        {
            s.Name(ResourceName(font)).Space().Reference(font.Number);
        }
    }

    // The name a face's font has in the form's resources: F and its index, or, when the source's resources hold that name,
    // the first F<index>_<n> they do not.
    private string ResourceName(PdfWriterFont font)
    {
        if (_resourceNames.TryGetValue(font, out var name))
        {
            return name;
        }

        var file = _source.File;
        var taken = _form.Resources is { } dr ? file.Dictionary(dr["Font"u8]) : null;
        string first = "F" + font.Index.ToString(CultureInfo.InvariantCulture);
        name = first;
        for (int n = 1; taken?.Contains(Encoding.ASCII.GetBytes(name)) == true || _standardFonts.ContainsKey(name) || _resourceNames.ContainsValue(name); n++)
        {
            name = string.Create(CultureInfo.InvariantCulture, $"{first}_{n}");
        }

        _resourceNames[font] = name;
        return name;
    }

    // A simple font's resource value: the source's object through the copier, or a standard font the update adds.
    private void SimpleFont(PdfSyntax s, PdfEditCopier copier, PdfEditCopier.Source own, PdfFieldFont font)
    {
        if (!font.Source.IsNull)
        {
            copier.Value(s, own, font.Source);
        }
        else if (font.Resource is { } name && _standardFonts.TryGetValue(Encoding.ASCII.GetString(name), out var added))
        {
            s.Reference(added.Number);
        }
        else
        {
            s.Raw("null"u8);
        }
    }

    // ---- the parts -----------------------------------------------------------------------------------------------------

    private PdfFormField Find(string name)
    {
        if (_flattened.Contains(name))
        {
            throw new ArgumentException($"The field '{name}' was flattened: it is no longer a field.", nameof(name));
        }

        return _form.Find(name) ?? throw NoField(name);
    }

    private ArgumentException NoField(string name)
    {
        var names = _form.Fields.Select(f => f.Name).Where(n => !_flattened.Contains(n)).ToList();
        string listed = names.Count == 0 ? "it has none" : string.Join(", ", names.Take(12)) + (names.Count > 12 ? $", ... ({names.Count} in all)" : "");
        return new ArgumentException($"{_source.File.Name} has no form field '{name}' ({listed}).", nameof(name));
    }

    // The source object's edit, made on its first change. A dictionary held inside another object (none of its own) is
    // moved to an object of its own, numbered when the document is written: what holds it is written again naming it.
    private ObjectEdit Edit(PdfValue node, PdfDictionary dictionary)
    {
        if (node.IsReference)
        {
            int number = node.AsReference.Number;
            if (!_edits.TryGetValue(number, out var edit))
            {
                edit = new ObjectEdit(node.AsReference.Generation, dictionary);
                _edits.Add(number, edit);
            }

            return edit;
        }

        if (!_moved.TryGetValue(dictionary, out var moved))
        {
            moved = new ObjectEdit(0, dictionary) { Moved = true };
            _moved.Add(dictionary, moved);
            _edits.Add(-_moved.Count, moved);
            Hold(dictionary, moved);
        }

        return moved;
    }

    // A dictionary moved to an object of its own: its holder (its parent's /Kids, a field's /Kids holding it as a widget,
    // or the form's /Fields) written again naming it; the objects under it that name their parent naming it; a widget
    // added to its pages' annotations (a dictionary held inside another object could not be listed there).
    private void Hold(PdfDictionary dictionary, ObjectEdit moved)
    {
        var file = _source.File;
        PdfValue holder = PdfValue.Null;
        bool found = false;
        foreach (var node in _form.Nodes)
        {
            if (ReferenceEquals(node.Dictionary, dictionary))
            {
                (holder, found) = (node.Parent, true);
                break;
            }
        }

        PdfFormWidget? asWidget = null;
        foreach (var field in _form.Fields)
        {
            foreach (var widget in field.Widgets)
            {
                if (ReferenceEquals(widget.Dictionary, dictionary))
                {
                    asWidget = widget;
                    if (!found)
                    {
                        (holder, found) = (field.Node, true);                  // a widget kid: its field's /Kids hold it
                    }
                }
            }
        }

        if (!found || holder.IsNull)
        {
            FormChanged(fields: true, resources: false);
        }
        else if (file.Dictionary(holder) is { } parent)
        {
            var edit = Edit(holder, parent);
            edit.Set("Kids", (s, copier, source) => Kids(s, copier, source, parent, edit.Added));
            if (edit.Moved)
            {
                moved.Set("Parent", (s, _, _) => s.Reference(edit.Number));    // its parent moved too: named by its new number
            }
        }

        // The objects under it that name their parent: named by its new number.
        foreach (var kid in file.Resolve(dictionary["Kids"u8]).AsArray?.Items ?? [])
        {
            if (kid.IsReference && file.Dictionary(kid) is { } under)
            {
                Edit(kid, under).Set("Parent", (s, _, _) => s.Reference(moved.Number));
            }
        }

        if (asWidget is not null)
        {
            foreach (int page in EditorPages(asWidget.Page))
            {
                Func<int> number = () => moved.Number;
                (_editor.Pages[page].Annots ??= new PdfAnnotsEdit()).Added.Add(number);
                moved.Listed.Add((page, number));
            }
        }
    }

    // The form's changes that keep its tree: fields moved to objects of their own (its /Fields written again) or fonts its
    // fields' default appearances name added to its resources: its own object written again, or, held in the catalog, the
    // catalog's entry.
    private void FormChanged(bool fields, bool resources)
    {
        if (!_form.Value.IsReference || _form.Dictionary is not { } acro)
        {
            _acroFormInline = true;
            return;
        }

        _acroFormNumber = _form.Value.AsReference.Number;
        var edit = Edit(_form.Value, acro);
        if (fields)
        {
            edit.Set("Fields", (s, copier, source) => Fields(s, copier, source));
        }

        if (resources)
        {
            edit.Set("DR", (s, copier, source) => ResourcesValue(s, copier, source, acro["DR"u8]));
        }
    }

    // A field's look as the document gives it (its default appearance, alignment, length, appearance characteristics and
    // its first widget's border).
    private PdfFieldLook Look(PdfFormField field)
    {
        var file = _source.File;
        var f = field.Field;
        int top = file.Resolve(field.Dictionary["TI"u8]).TryInteger(out long ti) ? (int)Math.Clamp(ti, 0, int.MaxValue) : 0;
        return new PdfFieldLook(f.Kind, f.Flags)
        {
            Values = f.Values, Options = f.Options, Alignment = f.Alignment, MaxLength = f.MaxLength, FontSize = f.FontSize, TextColour = f.TextColour,
            Background = f.Background, Border = f.Border, BorderWidth = f.BorderWidth ?? _options.Forms.BorderWidth, BorderStyle = f.BorderStyle, Dashes = f.Dashes,
            Mark = f.Mark, Caption = f.Caption, TopIndex = top,
        };
    }

    // The look as `widget` draws it: its own border when it states one (a field's widgets may differ).
    private PdfFieldLook Bordered(PdfFieldLook look, PdfFormWidget widget)
    {
        var file = _source.File;
        if (PdfForms.SafeDictionary(file, widget.Dictionary["BS"u8]) is null)
        {
            return look;
        }

        var (style, width, dashes) = PdfForms.Border(file, widget.Dictionary);
        return look with { BorderStyle = style, BorderWidth = width ?? _options.Forms.BorderWidth, Dashes = dashes };
    }

    // A widget's appearance box: its size, turned a quarter when its /MK /R says so.
    private static (float Width, float Height) Frame(PdfFormWidget widget)
    {
        var (width, height) = widget.Size;
        return widget.Rotation is 90 or 270 ? (height, width) : (width, height);
    }

    // The values a field takes from `given`: checked against its kind, choices and length.
    private static IReadOnlyList<string> Checked(PdfFormField field, string[] given)
    {
        var f = field.Field;
        foreach (string v in given)
        {
            ArgumentNullException.ThrowIfNull(v, "values");
        }

        switch (f.Kind)
        {
            case DocumentFieldKind.PushButton:
                throw new ArgumentException($"Field '{f.Name}' is a push button: it holds no value.", "values");
            case DocumentFieldKind.Signature:
                throw new ArgumentException($"Field '{f.Name}' is a signature field: its value is a signature (IDocumentEditor.Sign, phase D12 of the documents plan).", "values");
            case DocumentFieldKind.Text:
                if (given.Length > 1)
                {
                    throw new ArgumentException($"Field '{f.Name}' is a text box: it takes one value, not {given.Length}.", "values");
                }

                if (given.Length == 1 && f.MaxLength is { } max && given[0].EnumerateRunes().Count() > max)
                {
                    throw new ArgumentException($"Field '{f.Name}' takes at most {max} characters, not {given[0].EnumerateRunes().Count()} (\"{given[0]}\").", "values");
                }

                return given.Length == 1 && given[0].Length == 0 ? [] : given;
            case DocumentFieldKind.CheckBox or DocumentFieldKind.Radio:
                if (given.Length > 1)
                {
                    throw new ArgumentException($"Field '{f.Name}' is a {(f.Kind == DocumentFieldKind.Radio ? "radio group" : "check box")}: it takes one value, not {given.Length}.", "values");
                }

                if (given.Length == 0 || given[0] is "" or "Off")
                {
                    return [];
                }

                var states = field.Widgets.Select(w => w.OnState).Where(s => s is not null).Select(s => s!).Distinct().ToList();
                if (!states.Contains(given[0]) && !f.Options.Any(o => o.Export == given[0]) && !(states.Count == 0 && f.Kind == DocumentFieldKind.CheckBox))
                {
                    throw new ArgumentException(
                        $"Field '{f.Name}' takes {string.Join(", ", states.Concat(f.Options.Select(o => o.Export)).Distinct().Select(s => $"'{s}'"))} or 'Off', not '{given[0]}'.", "values");
                }

                return given;
            default:
                if (given.Length == 0 || (given.Length == 1 && given[0].Length == 0))
                {
                    return [];                                                 // cleared
                }

                if (given.Length > 1 && (f.Flags & DocumentFieldFlags.MultiSelect) == 0)
                {
                    throw new ArgumentException($"Field '{f.Name}' takes one choice, not {given.Length}.", "values");
                }

                var exports = new List<string>();
                var options = f.Options.ToList();
                foreach (string v in given)
                {
                    int at = options.FindIndex(o => o.Export == v);
                    at = at >= 0 ? at : options.FindIndex(o => o.Text == v);
                    if (at >= 0)
                    {
                        exports.Add(options[at].Export);
                    }
                    else if (f.Kind == DocumentFieldKind.Combo && (f.Flags & DocumentFieldFlags.Editable) != 0)
                    {
                        exports.Add(v);
                    }
                    else
                    {
                        throw new ArgumentException($"Field '{f.Name}' takes {string.Join(", ", f.Options.Select(o => $"'{o.Export}'"))}, not '{v}'.", "values");
                    }
                }

                return exports;
        }
    }

    // The appearance state a button's value names (its export through /Opt when its states are indices).
    private static string StateOf(PdfFormField field, string value)
    {
        int index = field.Field.Options.ToList().FindIndex(o => o.Export == value);
        if (index >= 0 && field.Widgets.Any(w => w.OnState == index.ToString(CultureInfo.InvariantCulture)))
        {
            return index.ToString(CultureInfo.InvariantCulture);
        }

        return value;
    }

    // A text or choice widget's appearance made to show the look in `font` (a rich text's runs each in its own), turned
    // and bordered as the widget says: over the widget's own appearance when that has a text section (its producer's box
    // and border kept, 12.7.4.3), else whole.
    private PdfMadeAppearance Make(PdfFormField field, PdfFieldLook look, PdfFormWidget widget, PdfFieldFont font)
    {
        var file = _source.File;
        look = Bordered(look, widget);
        var (width, height) = Frame(widget);
        var kept = field.Kind is DocumentFieldKind.Text or DocumentFieldKind.List or DocumentFieldKind.Combo
                   && file.Resolve(file.Dictionary(widget.Dictionary["AP"u8])?["N"u8] ?? PdfValue.Null).AsStream is { } existing
            ? PdfAppearanceBase.Of(file, existing, width, height, widget.Rotation)
            : null;
        var shown = kept is null ? look : look with { Framed = false };
        var appearance = field.Kind switch
        {
            DocumentFieldKind.List => PdfFieldAppearances.List(shown, font, width, height, _editor.Font, _options.Forms),
            DocumentFieldKind.PushButton => PdfFieldAppearances.Button(shown, font, width, height, _editor.Font, _options.Forms),
            DocumentFieldKind.Signature => PdfFieldAppearances.Box(shown, width, height, _editor.Font, _options.Forms),
            _ when Styled(look) => PdfFieldAppearances.Rich(shown, font, (run, text) => RunFont(field, font, run, text), width, height, _editor.Font, _options.Forms),
            _ => PdfFieldAppearances.Text(shown, font, width, height, _editor.Font, _options.Forms),
        };
        appearance.Rotation = widget.Rotation;
        if (kept is not null)
        {
            appearance = appearance.Over(kept);
        }

        var made = new PdfMadeAppearance(appearance);
        _made.Add(made);
        return made;
    }

    // A check box's or radio button's appearance, on or off, turned and bordered as the widget says.
    private PdfMadeAppearance MakeCheck(PdfFieldLook look, bool on, PdfFormWidget widget)
    {
        var (width, height) = Frame(widget);
        var appearance = PdfFieldAppearances.Check(Bordered(look, widget), on, width, height, _editor.Font, _options.Forms);
        appearance.Rotation = widget.Rotation;
        var made = new PdfMadeAppearance(appearance);
        _made.Add(made);
        return made;
    }

    /// <summary>Whether a look shows rich text styled (a text box's runs; a comb's cells and a password's asterisks plain).</summary>
    public static bool Styled(PdfFieldLook look) =>
        look.Rich is { Count: > 0 } && look.Kind == DocumentFieldKind.Text && (look.Flags & (DocumentFieldFlags.Comb | DocumentFieldFlags.Password)) == 0;

    // The font a run of a rich text is shown in, the field's own family its default.
    private PdfFieldFont RunFont(PdfFormField field, PdfFieldFont own, DocumentFieldRun run, string text) =>
        PdfRichFonts.Choose(run, text, own, run.Family ?? FieldFont(field).Font?.Name ?? field.Field.Font, _fonts, StandardFont, field.Name, _options.Forms);

    // The simple font a field's default appearance names in the form's resources: its resource name, the font read, its object.
    private (string? Resource, PdfFont? Font, PdfValue Reference) FieldFont(PdfFormField field)
    {
        var file = _source.File;
        string? resource = field.Field.Font;
        if (resource is not null && _form.Resources is { } dr && file.Dictionary(dr["Font"u8]) is { } fonts)
        {
            var reference = fonts[Encoding.UTF8.GetBytes(resource)];
            if (file.Dictionary(reference) is { } dictionary)
            {
                return (resource, PdfFonts.Read(file, dictionary), reference);
            }

            return (resource, null, reference);
        }

        return (resource, null, PdfValue.Null);
    }

    // The font a field's text is shown in: its default appearance's font when it is a simple font with a code for each
    // character, else faces that show the text.
    private PdfFieldFont Font(PdfFormField field, PdfFieldLook look)
    {
        string text = field.Kind switch
        {
            DocumentFieldKind.List => string.Join("\n", look.Options.Select(o => o.Text)),
            DocumentFieldKind.PushButton => look.Caption ?? "",
            _ => look.Shown(),
        };
        var (resource, font, reference) = FieldFont(field);
        PdfFieldFont? simple = font is { Composite: false } && resource is not null ? PdfFieldFont.Simple(Encoding.UTF8.GetBytes(resource), font, reference) : null;
        if (simple is not null && simple.Encodes(text))
        {
            return simple;
        }

        if (_fonts.Faces(text, font?.Name ?? resource, font?.Face) is { } faces)
        {
            return PdfFieldFont.OfFaces(faces);
        }

        if (text.Length == 0 || PdfFieldFonts.FirstMissing(text, _ => false) < 0)
        {
            return simple ?? PdfFieldFont.Standard("Helv"u8.ToArray(), PdfFontTables.StandardFontIndex("Helvetica"), 0);   // nothing to show: no font named
        }

        int missing = PdfFieldFonts.FirstMissing(text, c => simple?.Encodes(char.ConvertFromUtf32(c)) == true);
        throw new InvalidOperationException(string.Create(CultureInfo.InvariantCulture,
            $"Field '{field.Name}': no font shows \"{text}\" (U+{missing:X4} first): its font /{resource ?? "(none)"} has no code for it, and no face of DocumentWriteOptions.Forms.Faces or of the fonts folder (DocumentWriteOptions.Forms.Fonts, {DocumentDrawOptions.FontsVariable}{(_options.Forms.Fonts is null ? ": not set" : $": {_options.Forms.Fonts}")}) has its glyph."));
    }

    // A widget's appearance for a state as its /AP /N holds it (the stream's reference), or null.
    private static PdfValue? Reference(PdfFile file, PdfDictionary widget, string? state)
    {
        var normal = file.Dictionary(widget["AP"u8])?["N"u8] ?? PdfValue.Null;
        var resolved = file.Resolve(normal);
        if (resolved.Kind == PdfKind.Stream)
        {
            return normal.IsReference ? normal : null;
        }

        if (resolved.Kind != PdfKind.Dictionary || resolved.AsDictionary is not { } states)
        {
            return null;
        }

        var value = state is not null ? states[Encoding.UTF8.GetBytes(state)] : states.Count == 1 ? states.ValueAt(0) : PdfValue.Null;
        return value.IsReference ? value : null;
    }

    // A box on an editor's page as a rectangle of its user space.
    private static (double Left, double Bottom, double Right, double Top) Rect(DocumentBox box, PdfEditPage page)
    {
        var toUser = page.Document is { } document
            ? PdfEditPages.Inverse(document.Frame(page.Index, page.Rotation).Display)
            : page.Content!.ToUserSpace;
        return PdfNewField.Corners(box, toUser);
    }

    private static void Integers(PdfSyntax s, IReadOnlyList<int> values)
    {
        s.Byte((byte)'[');
        for (int i = 0; i < values.Count; i++)
        {
            if (i > 0)
            {
                s.Space();
            }

            s.Integer(values[i]);
        }

        s.Byte((byte)']');
    }

    // A source object written anew: its own entries but those set or removed, then those set. A dictionary held inside
    // another object is written as an object of its own (moved), under a number given when the document is written.
    private sealed class ObjectEdit(int generation, PdfDictionary original)
    {
        private readonly List<(string Key, Action<PdfSyntax, PdfEditCopier, PdfEditCopier.Source> Write)> _set = [];
        private readonly HashSet<string> _removed = new(StringComparer.Ordinal);

        public int Generation { get; } = generation;

        /// <summary>New nodes added under it (a parent's kids).</summary>
        public List<PdfNewNode> Added { get; } = [];

        /// <summary>Whether it was held inside another object and is written as one of its own.</summary>
        public bool Moved { get; init; }

        /// <summary>A moved object's number (0 until the document is written).</summary>
        public int Number { get; set; }

        /// <summary>A moved widget's places in its pages' annotations (the editor's page and what gives its number).</summary>
        public List<(int Page, Func<int> Number)> Listed { get; } = [];

        public void Set(string key, Action<PdfSyntax, PdfEditCopier, PdfEditCopier.Source> write)
        {
            _set.RemoveAll(e => e.Key == key);
            _removed.Remove(key);
            _set.Add((key, write));
        }

        public void Remove(string key)
        {
            _set.RemoveAll(e => e.Key == key);
            _removed.Add(key);
        }

        public void Write(PdfSyntax s, PdfEditCopier copier, PdfEditCopier.Source source)
        {
            s.Raw("<<"u8);
            for (int i = 0; i < original.Count; i++)
            {
                string key = original.KeyAt(i).ToString();
                if (_removed.Contains(key) || _set.Exists(e => e.Key == key))
                {
                    continue;
                }

                copier.Entry(s, source, original.KeyAt(i).Bytes, original.ValueAt(i));
            }

            foreach (var (key, write) in _set)
            {
                s.Name(key).Space();
                write(s, copier, source);
            }

            s.Raw(">>"u8);
        }
    }
}

/// <summary>A node a writer adds to a form's tree that holds fields (a dotted name's parent part): its name, parent and kids.</summary>
internal sealed class PdfNewNode(string name, string partial)
{
    /// <summary>Its full, dotted name.</summary>
    public string Name { get; } = name;

    /// <summary>Its own part of the name (<c>/T</c>).</summary>
    public string Partial { get; } = partial;

    /// <summary>Its object number (given when the document is written).</summary>
    public int Number { get; set; }

    /// <summary>A new node it is under, or null.</summary>
    public PdfNewNode? Parent { get; init; }

    /// <summary>A source node it is under (an editor's), or null.</summary>
    public PdfValue SourceParent { get; init; }

    /// <summary>The number of a source node it is under that was held inside another object and moved to one of its own, or null.</summary>
    public Func<int>? HeldParent { get; set; }

    /// <summary>The nodes under it.</summary>
    public List<PdfNewNode> Kids { get; } = [];

    /// <summary>Writes it: <c>/T</c>, its parent and its kids.</summary>
    public void Write(PdfSyntax s, PdfEditCopier? copier, PdfEditCopier.Source? source)
    {
        s.Raw("<</T"u8).Text(Partial);
        PdfNewField.ParentEntry(s, Parent, SourceParent, copier, source, HeldParent);
        s.Raw("/Kids["u8);
        foreach (var kid in Kids)
        {
            s.Reference(kid.Number).Space();
        }

        s.Raw("]>>"u8);
    }
}

/// <summary>A widget a writer adds: its page, box and rectangle, and its appearances (the normal one, or a button's on and off).</summary>
internal sealed class PdfNewWidget(int page, DocumentBox box, string? onState)
{
    public int Page { get; } = page;

    public DocumentBox Box { get; } = box;

    /// <summary>A check box's or radio button's on state, or null.</summary>
    public string? OnState { get; } = onState;

    /// <summary>How far its content is turned counter-clockwise (0, 90, 180 or 270; <c>/MK /R</c>).</summary>
    public int Rotation { get; init; }

    /// <summary>The editor's page it is on (pages may move after it is added), or null for a builder's.</summary>
    public PdfEditPage? EditPage { get; set; }

    public int Number { get; set; }

    /// <summary>Its rectangle in its page's user space (set when its page is known).</summary>
    public (double Left, double Bottom, double Right, double Top) Rect { get; set; }

    /// <summary>Its normal appearance (or a button's on appearance).</summary>
    public PdfMadeAppearance? On { get; set; }

    /// <summary>A button's off appearance.</summary>
    public PdfMadeAppearance? Off { get; set; }
}

/// <summary>
/// A field a writer adds (a builder's or an editor's): written from the contract's <see cref="DocumentField"/> as a terminal
/// field (<c>/FT /T /Ff /V /RV /DV /TU /MaxLen /Q /DA /Opt /I</c>; rich text runs as <c>/RV</c>, their plain text <c>/V</c>,
/// the field marked rich text) with its widgets (<c>/Rect /P /F /AP /AS /MK /BS</c>; <c>/MK /R</c> a turned one's, <c>/BS</c>
/// a border of a style, width or dashes given): one widget written in the field's own object, several as its kids. Each
/// widget's appearance is made from the field's look in the font that shows its text, turned as the widget is.
/// </summary>
internal sealed class PdfNewField(DocumentField field, string partial, PdfNewNode? parent, PdfValue sourceParent)
{
    public DocumentField Field { get; } = field;

    /// <summary>Its node (its number is the field's object's), under a new node or a source node (or at the form's top).</summary>
    public PdfNewNode Node { get; } = new(field.Name, partial) { Parent = parent, SourceParent = parent is null ? sourceParent : PdfValue.Null };

    public List<PdfNewWidget> Widgets { get; } = [];

    /// <summary>The default appearance it writes (its font's resource name, size, colour).</summary>
    public string Appearance { get; private set; } = "";

    /// <summary>The faces' fonts its default appearance names (the form's resources list them), or null.</summary>
    public IReadOnlyList<PdfWriterFont>? Fonts { get; private set; }

    // Its border's width (its own, else the options').
    private float _borderWidth;

    /// <summary>Refuses a field that cannot be written: no name, a widget off the document's pages, a value of the wrong kind.</summary>
    public static void Check(DocumentField field, int pageCount)
    {
        ArgumentNullException.ThrowIfNull(field);
        if (string.IsNullOrWhiteSpace(field.Name) || field.Name.Split('.').Any(p => p.Length == 0))
        {
            throw new ArgumentException($"A field has a name of parts joined by dots, none empty ('{field.Name}').", nameof(field));
        }

        foreach (var widget in field.Widgets)
        {
            if (widget.Page < 0 || widget.Page >= pageCount)
            {
                throw new ArgumentException($"Field '{field.Name}': a widget on page {widget.Page}, past the document's {pageCount} pages (indices from 0).", nameof(field));
            }

            var b = widget.Box;
            if (!(b.Width > 0 && b.Height > 0) || !float.IsFinite(b.X) || !float.IsFinite(b.Y) || !float.IsFinite(b.Width) || !float.IsFinite(b.Height))
            {
                throw new ArgumentException($"Field '{field.Name}': a widget's box has a finite place and a width and height above 0, not {b}.", nameof(field));
            }
        }

        if (field.Kind == DocumentFieldKind.Radio && field.Widgets.Count > 0 && field.Widgets.Count != field.Widgets.Select((w, i) => OnStateOf(field, w, i)).Distinct().Count())
        {
            throw new ArgumentException($"Field '{field.Name}': a radio group's buttons each have an on state of their own.", nameof(field));
        }

        if (!float.IsFinite(field.FontSize) || field.FontSize < 0)
        {
            throw new ArgumentException($"Field '{field.Name}': a font size is 0 (fitted) or a finite number of points, not {field.FontSize}.", nameof(field));
        }

        if (field.Widgets.Any(w => w.Rotation is not (0 or 90 or 180 or 270)))
        {
            throw new ArgumentException($"Field '{field.Name}': a widget is turned 0, 90, 180 or 270 degrees, not {field.Widgets.First(w => w.Rotation is not (0 or 90 or 180 or 270)).Rotation}.", nameof(field));
        }

        if (field.BorderWidth is { } width && !(width >= 0 && float.IsFinite(width)) || field.Dashes is null || field.Dashes.Any(d => !(d >= 0 && float.IsFinite(d))))
        {
            throw new ArgumentException($"Field '{field.Name}': a border's width and dashes are finite numbers of points, 0 or more.", nameof(field));
        }

        if (field.RichText.Count > 0)
        {
            if (field.Kind != DocumentFieldKind.Text)
            {
                throw new ArgumentException($"Field '{field.Name}' is a {field.Kind} field: rich text is a text field's value.", nameof(field));
            }

            CheckRuns(field.RichText, field.Name);
        }
    }

    /// <summary>Refuses runs of rich text that cannot be written: a run or its text missing, a size not above 0.</summary>
    public static void CheckRuns(IReadOnlyList<DocumentFieldRun> runs, string field)
    {
        foreach (var run in runs)
        {
            if (run?.Text is null)
            {
                throw new ArgumentException($"Field '{field}': a run of rich text, and its text, are given.", nameof(runs));
            }

            if (run.Size is { } size && !(size > 0 && float.IsFinite(size)))
            {
                throw new ArgumentException($"Field '{field}': a run's size is a finite number of points above 0, not {size}.", nameof(runs));
            }
        }
    }

    /// <summary>A button's on state: the widget's, else the field's choice of its index, else "Yes" (a check box) or "Choice" and its number.</summary>
    public static string OnStateOf(DocumentField field, DocumentWidget widget, int index) =>
        widget.OnState is { Length: > 0 } own ? own
        : field.Kind == DocumentFieldKind.Radio && index < field.Options.Count ? field.Options[index].Export
        : field.Kind == DocumentFieldKind.CheckBox ? "Yes" : string.Create(CultureInfo.InvariantCulture, $"Choice{index + 1}");

    /// <summary>
    /// Makes its widgets' appearances (and chooses its font) for the field's value; a face's font its default appearance
    /// names is named in the form's resources by <paramref name="resourceName"/>.
    /// </summary>
    public void Look(PdfFieldFonts fonts, DocumentField field, Func<IFontFace, PdfWriterFont> faceFont, DocumentFormOptions options, Func<string, int, PdfFieldFont> standard,
        Func<PdfWriterFont, string> resourceName)
    {
        var look = new PdfFieldLook(field.Kind, field.Flags)
        {
            Values = Values(field), Options = field.Options, Alignment = field.Alignment, MaxLength = field.MaxLength, FontSize = field.FontSize,
            TextColour = field.TextColour, Background = field.Background, Border = field.Border, BorderWidth = field.BorderWidth ?? options.BorderWidth,
            BorderStyle = field.BorderStyle, Dashes = field.Dashes, Mark = field.Mark, Caption = field.Caption,
            Rich = field.Kind == DocumentFieldKind.Text && field.RichText.Count > 0 ? field.RichText : null,
        };
        _borderWidth = look.BorderWidth;
        string text = field.Kind switch
        {
            DocumentFieldKind.List => string.Join("\n", field.Options.Select(o => o.Text)),
            DocumentFieldKind.PushButton => field.Caption ?? "",
            _ => look.Shown(),
        };
        PdfFieldFont? font = null;
        if (field.Kind is DocumentFieldKind.CheckBox or DocumentFieldKind.Radio)
        {
            int zapf = PdfFontTables.StandardFontIndex("ZapfDingbats");
            standard("ZaDb", zapf);
            Appearance = "/ZaDb 0 Tf " + Colour(field.TextColour);
        }
        else if (field.Kind != DocumentFieldKind.Signature)
        {
            font = Font(fonts, field, text, options, standard);
            string resource;
            if (font.IsSimple)
            {
                resource = Encoding.ASCII.GetString(font.Resource!);
            }
            else
            {
                var used = faceFont(font.Faces![0]);
                Fonts = [used];
                resource = resourceName(used);
            }

            Appearance = string.Create(CultureInfo.InvariantCulture, $"/{resource} {field.FontSize:0.###} Tf {Colour(field.TextColour)}");
        }

        for (int i = 0; i < field.Widgets.Count; i++)
        {
            var w = field.Widgets[i];
            string? on = field.Kind is DocumentFieldKind.CheckBox or DocumentFieldKind.Radio ? OnStateOf(field, w, i) : null;
            var widget = new PdfNewWidget(w.Page, w.Box, on) { Rotation = w.Rotation };
            var (width, height) = w.Rotation is 90 or 270 ? (w.Box.Height, w.Box.Width) : (w.Box.Width, w.Box.Height);
            switch (field.Kind)
            {
                case DocumentFieldKind.CheckBox or DocumentFieldKind.Radio:
                    widget.On = new PdfMadeAppearance(PdfFieldAppearances.Check(look, on: true, width, height, faceFont, options));
                    widget.Off = new PdfMadeAppearance(PdfFieldAppearances.Check(look, on: false, width, height, faceFont, options));
                    break;
                case DocumentFieldKind.List:
                    widget.On = new PdfMadeAppearance(PdfFieldAppearances.List(look, font!, width, height, faceFont, options));
                    break;
                case DocumentFieldKind.PushButton:
                    widget.On = new PdfMadeAppearance(PdfFieldAppearances.Button(look, font!, width, height, faceFont, options));
                    break;
                case DocumentFieldKind.Signature:
                    widget.On = new PdfMadeAppearance(PdfFieldAppearances.Box(look, width, height, faceFont, options));
                    break;
                default:
                    widget.On = new PdfMadeAppearance(PdfFormEditor.Styled(look)
                        ? PdfFieldAppearances.Rich(look, font!, (run, text) => PdfRichFonts.Choose(run, text, font!, run.Family ?? field.Font ?? PdfRichFonts.FamilyOf(font!), fonts, standard, field.Name, options),
                            width, height, faceFont, options)
                        : PdfFieldAppearances.Text(look, font!, width, height, faceFont, options));
                    break;
            }

            widget.On!.Appearance.Rotation = w.Rotation;
            if (widget.Off is { } off)
            {
                off.Appearance.Rotation = w.Rotation;
            }

            Widgets.Add(widget);
        }
    }

    // The values the field's appearance shows: a list's several, a rich text's plain text, else its one value.
    private static IReadOnlyList<string> Values(DocumentField field) =>
        field.Kind == DocumentFieldKind.Text && field.RichText.Count > 0 ? [PdfRichText.Plain(field.RichText)]
        : field.Values.Count > 0 ? field.Values : field.Value is { } value ? [value] : [];

    // The font a field added shows its text in: a standard font it names (or Helvetica, when none is named and no face is
    // given) when that has a code for every character; else faces that show the text.
    private static PdfFieldFont Font(PdfFieldFonts fonts, DocumentField field, string text, DocumentFormOptions options, Func<string, int, PdfFieldFont> standard)
    {
        // A standard font that has a code for every character: the one named, or Helvetica when none is (and no face is given).
        PdfFieldFont? Standard(string? name)
        {
            int index = PdfFontTables.StandardFontIndex(name);
            if (index < 0 || index == PdfFontTables.Symbol || index == PdfFontTables.ZapfDingbats)
            {
                return null;
            }

            string resource = PdfFontTables.StandardFonts[index].Replace("-", "", StringComparison.Ordinal);
            resource = resource == "Helvetica" ? "Helv" : resource;
            return PdfFieldFont.Standard(Encoding.ASCII.GetBytes(resource), index, 0).Encodes(text) ? standard(resource, index) : null;   // its object only when used
        }

        if (Standard(field.Font ?? (options.Faces.Count == 0 ? "Helvetica" : null)) is { } named)
        {
            return named;
        }

        if (fonts.Faces(text, field.Font, null) is { } faces)
        {
            return PdfFieldFont.OfFaces(faces);
        }

        if (field.Font is null && Standard("Helvetica") is { } plain)
        {
            return plain;                                                      // faces given, none showing the text: Helvetica does
        }

        if (text.Length == 0 || PdfFieldFonts.FirstMissing(text, _ => false) < 0)
        {
            return standard("Helv", PdfFontTables.StandardFontIndex("Helvetica"));
        }

        throw new InvalidOperationException(string.Create(CultureInfo.InvariantCulture,
            $"Field '{field.Name}': no font shows \"{text}\" (U+{PdfFieldFonts.FirstMissing(text, _ => false):X4} first): {(field.Font is null ? "no font is named" : $"its font '{field.Font}' is no standard font that has it")}, and no face of DocumentWriteOptions.Forms.Faces or of the fonts folder (DocumentWriteOptions.Forms.Fonts, {DocumentDrawOptions.FontsVariable}{(options.Fonts is null ? ": not set" : $": {options.Fonts}")}) has its glyph."));
    }

    private static string Colour(DocumentColour colour)
    {
        var parts = Enumerable.Range(0, colour.Count).Select(c => colour[c].ToString("0.###", CultureInfo.InvariantCulture));
        return string.Join(' ', parts) + colour.Space switch { DocumentColourSpace.Grey => " g", DocumentColourSpace.Rgb => " rg", _ => " k" };
    }

    /// <summary>Numbers its objects: the field (with its widget when it has one), its widgets and appearances.</summary>
    public void Allocate(PdfFileWriter file)
    {
        Node.Number = file.Allocate();
        for (int i = 0; i < Widgets.Count; i++)
        {
            var widget = Widgets[i];
            widget.Number = Widgets.Count == 1 ? Node.Number : file.Allocate();
            if (widget.On is { } on)
            {
                on.Number = file.Allocate();
            }

            if (widget.Off is { } off)
            {
                off.Number = file.Allocate();
            }
        }
    }

    /// <summary>
    /// Writes its appearances, then the field and its widgets (each widget's page by <paramref name="pageNumber"/>; a
    /// simple font's resource value by <paramref name="simpleFont"/>).
    /// </summary>
    public void Write(PdfFileWriter file, PdfEditCopier? copier, PdfEditCopier.Source? source, Func<PdfNewWidget, (int Number, int Generation)> pageNumber,
        Action<PdfSyntax, PdfFieldFont> simpleFont)
    {
        WriteAppearances(file, simpleFont);
        WriteField(file, copier, source, pageNumber, Widgets.Count != 1);
        if (Widgets.Count != 1)
        {
            foreach (var widget in Widgets)
            {
                WriteWidget(file, widget, pageNumber, alone: true);
            }
        }
    }

    /// <summary>Writes the widgets' appearances.</summary>
    public void WriteAppearances(PdfFileWriter file, Action<PdfSyntax, PdfFieldFont> simpleFont)
    {
        foreach (var widget in Widgets)
        {
            foreach (var made in new[] { widget.On, widget.Off })
            {
                if (made is not null)
                {
                    made.Appearance.Write(file, made.Number, simpleFont);
                }
            }
        }
    }

    /// <summary>Writes the field's object (with its one widget's entries when <paramref name="kids"/> is false).</summary>
    public void WriteField(PdfFileWriter file, PdfEditCopier? copier, PdfEditCopier.Source? source, Func<PdfNewWidget, (int Number, int Generation)> pageNumber, bool kids)
    {
        void Body(PdfSyntax s)
        {
            var f = Field;
            s.Raw("<<"u8);
            s.Raw(f.Kind switch
            {
                DocumentFieldKind.Text => "/FT/Tx"u8,
                DocumentFieldKind.List or DocumentFieldKind.Combo => "/FT/Ch"u8,
                DocumentFieldKind.Signature => "/FT/Sig"u8,
                _ => "/FT/Btn"u8,
            });
            s.Raw("/T"u8).Text(Node.Partial);
            ParentEntry(s, Node.Parent, Node.SourceParent, copier, source, Node.HeldParent);
            bool rich = f.Kind == DocumentFieldKind.Text && f.RichText.Count > 0;
            long ff = PdfForms.FieldFlags(f.Kind, f.Flags) | (rich ? PdfForms.RichTextFlag : 0);
            if (ff != 0)
            {
                s.Raw("/Ff "u8).Integer(ff);
            }

            var values = Values(f);
            switch (f.Kind)
            {
                case DocumentFieldKind.Text when values.Count > 0:
                    s.Raw("/V"u8).Text(values[0]);
                    if (rich)
                    {
                        s.Raw("/RV"u8).Text(PdfRichText.Write(f.RichText));
                    }

                    break;
                case DocumentFieldKind.CheckBox or DocumentFieldKind.Radio:
                    string state = values.Count > 0 && Widgets.Find(w => w.OnState == values[0]) is { } chosen ? chosen.OnState! : "Off";
                    s.Raw("/V"u8).Name(state);
                    break;
                case DocumentFieldKind.List or DocumentFieldKind.Combo when values.Count > 0:
                    if ((f.Flags & DocumentFieldFlags.MultiSelect) != 0)
                    {
                        s.Raw("/V["u8);
                        foreach (string v in values)
                        {
                            s.Text(v);
                        }

                        s.Byte((byte)']');
                    }
                    else
                    {
                        s.Raw("/V"u8).Text(values[0]);
                    }

                    var indices = values.Select(v => f.Options.ToList().FindIndex(o => o.Export == v)).Where(i => i >= 0).Order().ToList();
                    if (f.Kind == DocumentFieldKind.List && indices.Count > 0)
                    {
                        s.Raw("/I["u8);
                        foreach (int i in indices)
                        {
                            s.Integer(i).Space();
                        }

                        s.Byte((byte)']');
                    }

                    break;
            }

            if (f.DefaultValue is { } defaultValue)
            {
                if (f.Kind is DocumentFieldKind.CheckBox or DocumentFieldKind.Radio)
                {
                    s.Raw("/DV"u8).Name(defaultValue);
                }
                else
                {
                    s.Raw("/DV"u8).Text(defaultValue);
                }
            }

            if (f.Tooltip is { } tooltip)
            {
                s.Raw("/TU"u8).Text(tooltip);
            }

            if (f.Kind == DocumentFieldKind.Text && f.MaxLength is { } max)
            {
                s.Raw("/MaxLen "u8).Integer(max);
            }

            if (f.Alignment != DocumentFieldAlignment.Left)
            {
                s.Raw("/Q "u8).Integer((int)f.Alignment);
            }

            if (Appearance.Length > 0)
            {
                s.Raw("/DA"u8).Literal(Encoding.ASCII.GetBytes(Appearance));
            }

            if (f.Kind is DocumentFieldKind.List or DocumentFieldKind.Combo && f.Options.Count > 0)
            {
                s.Raw("/Opt["u8);
                foreach (var option in f.Options)
                {
                    if (option.Export == option.Text)
                    {
                        s.Text(option.Text);
                    }
                    else
                    {
                        s.Byte((byte)'[').Text(option.Export).Text(option.Text).Byte((byte)']');
                    }
                }

                s.Byte((byte)']');
            }

            if (kids)
            {
                s.Raw("/Kids["u8);
                foreach (var widget in Widgets)
                {
                    s.Reference(widget.Number).Space();
                }

                s.Byte((byte)']');
            }
            else
            {
                WidgetEntries(s, Widgets[0], pageNumber);
            }

            s.Raw(">>"u8);
        }

        if (copier is not null)
        {
            copier.WriteObject(Node.Number, Body);
            return;
        }

        var target = file.BeginSmall(Node.Number);
        Body(target);
        file.EndSmall();
    }

    /// <summary>Writes a widget of its own (a kid of the field): <c>/Parent</c> and its widget entries.</summary>
    public void WriteWidget(PdfFileWriter file, PdfNewWidget widget, Func<PdfNewWidget, (int Number, int Generation)> pageNumber, bool alone)
    {
        var s = file.BeginSmall(widget.Number);
        s.Raw("<<"u8);
        if (alone)
        {
            s.Raw("/Parent "u8).Reference(Node.Number);
        }

        WidgetEntries(s, widget, pageNumber);
        s.Raw(">>"u8);
        file.EndSmall();
    }

    // A widget's entries: annotation, rectangle, page, flags, appearances and state, its characteristics and border.
    private void WidgetEntries(PdfSyntax s, PdfNewWidget widget, Func<PdfNewWidget, (int Number, int Generation)> pageNumber)
    {
        var f = Field;
        var (page, generation) = pageNumber(widget);
        s.Raw("/Type/Annot/Subtype/Widget/Rect["u8).Number(widget.Rect.Left, 4).Space().Number(widget.Rect.Bottom, 4).Space().Number(widget.Rect.Right, 4).Space()
            .Number(widget.Rect.Top, 4).Raw("]/F 4"u8);
        if (page > 0)
        {
            s.Raw("/P "u8).Integer(page).Space().Integer(generation).Raw(" R"u8);   // none when its page was removed after it was added
        }
        if (widget.Off is { } off && widget.On is { } on)
        {
            s.Raw("/AP<</N<<"u8).Name(widget.OnState!).Space().Reference(on.Number).Raw("/Off "u8).Reference(off.Number).Raw(">>>>"u8);
            var values = Values(f);
            s.Raw("/AS"u8).Name(values.Count > 0 && values[0] == widget.OnState ? widget.OnState! : "Off");
        }
        else if (widget.On is { } normal)
        {
            s.Raw("/AP<</N "u8).Reference(normal.Number).Raw(">>"u8);
        }

        bool mark = f.Kind is DocumentFieldKind.CheckBox or DocumentFieldKind.Radio;
        if (f.Background is not null || f.Border is not null || mark || f.Caption is not null || widget.Rotation != 0)
        {
            s.Raw("/MK<<"u8);
            if (f.Background is { } background)
            {
                s.Raw("/BG"u8);
                Colour(s, background);
            }

            if (f.Border is { } border)
            {
                s.Raw("/BC"u8);
                Colour(s, border);
            }

            if (mark)
            {
                s.Raw("/CA"u8).Text(PdfForms.MarkCharacter(f.Mark));
            }
            else if (f.Caption is { } caption)
            {
                s.Raw("/CA"u8).Text(caption);
            }

            if (widget.Rotation != 0)
            {
                s.Raw("/R "u8).Integer(widget.Rotation);
            }

            s.Raw(">>"u8);
        }

        if (f.BorderStyle != DocumentFieldBorderStyle.Solid || f.BorderWidth is not null || f.Dashes.Count > 0)
        {
            s.Raw("/BS<</W "u8).Number(_borderWidth).Raw(f.BorderStyle switch
            {
                DocumentFieldBorderStyle.Dashed => "/S/D"u8,
                DocumentFieldBorderStyle.Beveled => "/S/B"u8,
                DocumentFieldBorderStyle.Inset => "/S/I"u8,
                DocumentFieldBorderStyle.Underline => "/S/U"u8,
                _ => "/S/S"u8,
            });
            if (f.Dashes.Count > 0)
            {
                s.Raw("/D["u8);
                for (int i = 0; i < f.Dashes.Count; i++)
                {
                    (i > 0 ? s.Space() : s).Number(f.Dashes[i]);
                }

                s.Byte((byte)']');
            }

            s.Raw(">>"u8);
        }
    }

    private static void Colour(PdfSyntax s, DocumentColour colour)
    {
        s.Byte((byte)'[');
        for (int c = 0; c < colour.Count; c++)
        {
            if (c > 0)
            {
                s.Space();
            }

            s.Number(colour[c]);
        }

        s.Byte((byte)']');
    }

    /// <summary>
    /// A node's <c>/Parent</c>: a new node by its number, a moved source node by its new number (<paramref name="held"/>), a
    /// source node through the copier (as itself without one), or none.
    /// </summary>
    public static void ParentEntry(PdfSyntax s, PdfNewNode? parent, PdfValue sourceParent, PdfEditCopier? copier, PdfEditCopier.Source? source, Func<int>? held = null)
    {
        if (parent is not null)
        {
            s.Raw("/Parent "u8).Reference(parent.Number);
        }
        else if (held is not null)
        {
            s.Raw("/Parent "u8).Reference(held());
        }
        else if (!sourceParent.IsNull && copier is not null && source is not null)
        {
            s.Raw("/Parent "u8);
            copier.Value(s, source, sourceParent);
        }
    }

    /// <summary>A standard font object in the WinAnsi encoding (ZapfDingbats in its own).</summary>
    public static void WriteStandardFont(PdfFileWriter file, int number, int standard)
    {
        var s = file.BeginSmall(number);
        s.Raw("<</Type/Font/Subtype/Type1/BaseFont"u8).Name(PdfFontTables.StandardFonts[standard]);
        if (standard != PdfFontTables.ZapfDingbats && standard != PdfFontTables.Symbol)
        {
            s.Raw("/Encoding/WinAnsiEncoding"u8);
        }

        s.Raw(">>"u8);
        file.EndSmall();
    }

    /// <summary>A box of a page (points from its top-left corner) as a rectangle of its user space by <paramref name="toUser"/>.</summary>
    public static (double Left, double Bottom, double Right, double Top) Corners(DocumentBox box, PdfMatrix toUser)
    {
        double x0 = double.MaxValue, y0 = double.MaxValue, x1 = double.MinValue, y1 = double.MinValue;
        foreach (var (x, y) in new (double X, double Y)[] { (box.X, box.Y), (box.Right, box.Y), (box.Right, box.Bottom), (box.X, box.Bottom) })
        {
            double px = toUser.A * x + toUser.C * y + toUser.E, py = toUser.B * x + toUser.D * y + toUser.F;
            (x0, y0, x1, y1) = (Math.Min(x0, px), Math.Min(y0, py), Math.Max(x1, px), Math.Max(y1, py));
        }

        return (x0, y0, x1, y1);
    }
}

/// <summary>
/// The form a PDF builder writes (C24, <see cref="IDocumentBuilder.AddField"/>): fields added while their pages are being
/// built (the page in hand or one added later), each field's objects numbered when it is added and its appearances
/// written then; each widget written with its page (which lists it in its <c>/Annots</c>), the fields' and their parents'
/// objects and the catalog's form entry at the end.
/// </summary>
internal sealed class PdfFormBuilder(PdfFileWriter file, DocumentWriteOptions options, Func<IFontFace, PdfWriterFont> faceFont)
{
    private readonly PdfFieldFonts _fonts = new(options.Forms);
    private readonly List<PdfNewField> _fields = [];
    private readonly List<PdfNewNode> _nodes = [];
    private readonly List<PdfNewNode> _top = [];
    private readonly Dictionary<string, int> _standardFonts = new(StringComparer.Ordinal);
    private readonly HashSet<PdfWriterFont> _resourceFonts = [];
    private readonly Dictionary<int, int> _pageNumbers = [];

    /// <summary>Whether any field was added.</summary>
    public bool Any => _fields.Count > 0;

    /// <summary>Adds <paramref name="field"/>; <paramref name="firstOpen"/> is the first page not written yet, <paramref name="pages"/> the pages there are with the one in hand.</summary>
    public void Add(DocumentField field, int firstOpen, int pages, Func<int, PdfPageBuilder?> page)
    {
        PdfNewField.Check(field, int.MaxValue);
        foreach (var widget in field.Widgets)
        {
            if (widget.Page < firstOpen)
            {
                throw new ArgumentException(
                    $"Field '{field.Name}': a widget on page {widget.Page}, written already (the document is on page {pages - 1}): a field is added while the pages its widgets are on are being built.",
                    nameof(field));
            }
        }

        if (_fields.Exists(f => f.Field.Name == field.Name) || _nodes.Exists(n => n.Name == field.Name))
        {
            throw new ArgumentException($"A field '{field.Name}' is there already: a name is one field's.", nameof(field));
        }

        string[] parts = field.Name.Split('.');
        PdfNewNode? parent = null;
        string prefix = "";
        for (int i = 0; i < parts.Length - 1; i++)
        {
            prefix = prefix.Length == 0 ? parts[i] : prefix + "." + parts[i];
            if (_fields.Exists(f => f.Field.Name == prefix))
            {
                throw new ArgumentException($"'{prefix}' is a field with a value: it cannot hold the field '{field.Name}'.", nameof(field));
            }

            var node = _nodes.Find(n => n.Name == prefix);
            if (node is null)
            {
                node = new PdfNewNode(prefix, parts[i]) { Parent = parent, Number = file.Allocate() };
                _nodes.Add(node);
                (parent is null ? _top : parent.Kids).Add(node);
            }

            parent = node;
        }

        var added = new PdfNewField(field, parts[^1], parent, PdfValue.Null);
        added.Look(_fonts, field, faceFont, options.Forms, Standard, font => "F" + font.Index.ToString(CultureInfo.InvariantCulture));
        added.Allocate(file);
        (parent is null ? _top : parent.Kids).Add(added.Node);
        if (added.Fonts is { } used)
        {
            _resourceFonts.UnionWith(used);
        }

        added.WriteAppearances(file, (s, font) => s.Reference(font.Number));
        foreach (var widget in added.Widgets)
        {
            if (page(widget.Page) is { } open)
            {
                widget.Rect = PdfNewField.Corners(widget.Box, open.ToUserSpace);
            }
        }

        _fields.Add(added);
    }

    /// <summary>The widgets on page <paramref name="index"/> (its <c>/Annots</c>), their rectangles set from its frame.</summary>
    public List<int>? Annotations(int index, PdfPageBuilder page)
    {
        List<int>? numbers = null;
        foreach (var field in _fields)
        {
            foreach (var widget in field.Widgets)
            {
                if (widget.Page == index)
                {
                    widget.Rect = PdfNewField.Corners(widget.Box, page.ToUserSpace);
                    (numbers ??= []).Add(widget.Number);
                }
            }
        }

        return numbers;
    }

    /// <summary>Writes the widgets on page <paramref name="index"/>, now written as object <paramref name="number"/>.</summary>
    public void PageWritten(int index, int number)
    {
        _pageNumbers[index] = number;
        foreach (var field in _fields)
        {
            if (field.Widgets.Count > 1)
            {
                foreach (var widget in field.Widgets)
                {
                    if (widget.Page == index)
                    {
                        field.WriteWidget(file, widget, PageNumber, alone: true);
                    }
                }
            }
            else if (field.Widgets.Count == 1 && field.Widgets[0].Page == index)
            {
                field.WriteField(file, null, null, PageNumber, kids: false);
            }
        }
    }

    /// <summary>The pages widgets were put on that the document never got (a field naming a page never added), or empty.</summary>
    public IReadOnlyList<string> Unplaced(int pageCount) =>
        [.. _fields.SelectMany(f => f.Widgets.Where(w => w.Page >= pageCount).Select(w => $"'{f.Field.Name}' on page {w.Page}"))];

    /// <summary>Writes the fields with several widgets or none, their parents, and the standard fonts they use.</summary>
    public void Finish()
    {
        foreach (var field in _fields)
        {
            if (field.Widgets.Count != 1)
            {
                field.WriteField(file, null, null, PageNumber, kids: true);
            }
        }

        foreach (var node in _nodes)
        {
            var s = file.BeginSmall(node.Number);
            node.Write(s, null, null);
            file.EndSmall();
        }

        foreach (var (name, number) in _standardFonts)
        {
            PdfNewField.WriteStandardFont(file, number, PdfFontTables.StandardFontIndex(StandardName(name)));
        }
    }

    /// <summary>The catalog's form entry: its fields, its resources' fonts, a default appearance.</summary>
    public void WriteAcroForm(PdfSyntax s)
    {
        s.Raw("/AcroForm<</Fields["u8);
        foreach (var node in _top)
        {
            s.Reference(node.Number).Space();
        }

        s.Raw("]/DA"u8).Literal("/Helv 0 Tf 0 g"u8);
        if (_standardFonts.Count > 0 || _resourceFonts.Count > 0)
        {
            s.Raw("/DR<</Font<<"u8);
            foreach (var (name, number) in _standardFonts)
            {
                s.Name(name).Space().Reference(number);
            }

            foreach (var font in _resourceFonts)
            {
                s.Raw("/F"u8).Integer(font.Index).Space().Reference(font.Number);
            }

            s.Raw(">>>>"u8);
        }

        s.Raw(">>"u8);
    }

    private (int Number, int Generation) PageNumber(PdfNewWidget widget) => (_pageNumbers.TryGetValue(widget.Page, out int number) ? number : 0, 0);

    // A standard font a field uses: its object numbered once, named in the form's resources.
    private PdfFieldFont Standard(string resource, int standard)
    {
        if (!_standardFonts.TryGetValue(resource, out int number))
        {
            number = file.Allocate();
            _standardFonts[resource] = number;
        }

        return PdfFieldFont.Standard(Encoding.ASCII.GetBytes(resource), standard, number);
    }

    // The standard font a resource name stands for ("Helv" Helvetica, "ZaDb" ZapfDingbats, else the name without its dashes).
    private static string StandardName(string resource) => resource switch
    {
        "Helv" => "Helvetica",
        "ZaDb" => "ZapfDingbats",
        _ => PdfFontTables.StandardFonts.First(n => n.Replace("-", "", StringComparison.Ordinal) == resource),
    };
}
