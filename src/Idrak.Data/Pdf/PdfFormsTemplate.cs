// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Xml;

namespace Idrak.Data.Pdf;

/// <summary>
/// Reads an XFA form's template packet (XFA 3.3, read as data: XML with no DTD, nothing outside fetched, no layout and no
/// script run) into the contract's <see cref="DocumentFormTemplate"/>: its sections (named subforms and page areas, an
/// unnamed subform passed through), its fields (each <c>field</c>, and each <c>exclGroup</c> as one radio group of its
/// buttons) with their captions, kinds from their <c>ui</c> (text, numeric and date edits as text, a password edit, a
/// check button, a choice list as a list or a drop-down list, a button, a signature), choices from their <c>items</c>
/// (shown, and saved when a second list says <c>save="1"</c>), flags (read only, required, several lines, several
/// choices, typed text), and their binding to the form's data: by name under their sections' data (normal binding), by
/// <c>bind ref</c> (<c>$</c>, <c>$record</c>, <c>$data</c>), by name anywhere (<c>global</c>), or none; each linked to the
/// data's value (the template's default without one) and to the document's field that holds it (its AcroForm twin, or the
/// data value read as a field of its own). Every script (in an <c>event</c>, a <c>calculate</c>, a <c>validate</c> or the
/// <c>variables</c>) is listed as an issue: none is run.
/// </summary>
internal static class PdfFormTemplates
{
    /// <summary>The template of <paramref name="document"/>'s XFA form (<paramref name="form"/>), or null when it has no XFA template.</summary>
    /// <exception cref="InvalidDataException">The template nests its containers deeper than <see cref="DocumentOptions.MaxDepth"/> (named).</exception>
    public static DocumentFormTemplate? Read(PdfDocument document, PdfForm form)
    {
        var file = document.File;
        if (form.Dictionary is not { } acro || file.Resolve(acro["XFA"u8]).IsNull
            || PdfForms.Packet(file, acro["XFA"u8], "template", "<template"u8, "</template>"u8) is not { } xml)
        {
            return null;
        }

        var scripts = new List<(Node Holder, string Activity, string Type)>();
        Node root;
        try
        {
            root = Parse(xml, file.MaxStreamBytes, file.MaxDepth, scripts);
        }
        catch (XmlException e)
        {
            return new DocumentFormTemplate { Issues = [new DocumentIssue(DocumentIssue.DamagedStream, $"the XFA template is not well-formed XML: {e.Message}")] };
        }

        return new Resolution(form).Resolve(root, scripts);
    }

    // A container or field of the template as read: what the resolution needs of it.
    private sealed class Node(string element, string? name, Node? parent)
    {
        public string Element { get; } = element;

        public string? Name { get; } = name;

        public Node? Parent { get; } = parent;

        public List<Node> Children { get; } = [];

        public string? BindMatch { get; set; }

        public string? BindRef { get; set; }

        public string? Caption { get; set; }

        public string? Ui { get; set; }

        public bool MultiLine { get; set; }

        public string? ChoiceOpen { get; set; }

        public bool TextEntry { get; set; }

        public bool ReadOnly { get; set; }

        public bool Required { get; set; }

        public List<string>? Items { get; set; }

        public List<string>? SavedItems { get; set; }

        public string? Default { get; set; }

        public bool IsField => Element is "field" or "exclGroup";

        public bool IsSection => Element is "subform" or "pageArea" && Name is not null;
    }

    // An element open while the template is read: its name, the node it made (a container or field), and what collects
    // its text (a caption, a default value, an item).
    private sealed class Open(string local, Node? node)
    {
        public string Local { get; } = local;

        public Node? Node { get; } = node;

        public StringBuilder? Text { get; set; }

        public string? Activity { get; set; }
    }

    // Elements read as containers.
    private static bool Container(string local) => local is "subform" or "subformSet" or "area" or "exclGroup" or "pageSet" or "pageArea" or "field";

    // Elements whose content is not part of the form's design as data (drawing, layout, prototypes).
    private static bool Skipped(string local) => local is "proto" or "draw" or "border" or "font" or "para" or "margin" or "keep" or "traversal" or "assist"
        or "occur" or "medium" or "contentArea" or "breakBefore" or "breakAfter" or "overflow" or "format" or "desc" or "extras" or "connect" or "setProperty";

    private static Node Parse(byte[] xml, long maxCharacters, int maxDepth, List<(Node Holder, string Activity, string Type)> scripts)
    {
        var settings = new XmlReaderSettings
        {
            DtdProcessing = DtdProcessing.Prohibit, XmlResolver = null, IgnoreComments = true, IgnoreProcessingInstructions = true, IgnoreWhitespace = true,
            MaxCharactersInDocument = Math.Max(1, maxCharacters), MaxCharactersFromEntities = 1024,
        };
        using var stream = new MemoryStream(xml, writable: false);
        using var reader = XmlReader.Create(stream, settings);
        var root = new Node("template", null, null);
        var open = new List<Open>();
        int containers = 0;
        bool read = true;
        while (!reader.EOF)
        {
            if (read && !reader.Read())
            {
                break;
            }

            read = true;
            switch (reader.NodeType)
            {
                case XmlNodeType.Element:
                {
                    string local = reader.LocalName;
                    var parent = open.Count > 0 ? open[^1] : null;
                    Node? holder = null;
                    for (int i = open.Count - 1; i >= 0 && holder is null; i--)
                    {
                        holder = open[i].Node;
                    }

                    holder ??= root;
                    if (Skipped(local))
                    {
                        reader.Skip();
                        read = false;
                        continue;
                    }

                    var entry = new Open(local, null);
                    if (Container(local) && parent?.Local is not ("ui" or "items" or "value" or "caption"))
                    {
                        if (++containers > 1_000_000)
                        {
                            throw PdfLimit.Error("an XFA template of more than a million containers.");
                        }

                        var node = new Node(local, reader.GetAttribute("name") is { Length: > 0 } n ? n : null, holder);
                        if (Depth(node) > maxDepth)
                        {
                            throw PdfLimit.Error($"an XFA template's containers nested deeper than {maxDepth} levels, the limit {nameof(DocumentOptions.MaxDepth)} (DocumentOptions.MaxDepth).");
                        }

                        node.ReadOnly = reader.GetAttribute("access") is "readOnly" or "protected" or "nonInteractive";
                        holder.Children.Add(node);
                        entry = new Open(local, node);
                    }
                    else if (parent?.Local == "ui" && parent.Node is null && holder.Ui is null)
                    {
                        holder.Ui = local;
                        holder.MultiLine = reader.GetAttribute("multiLine") == "1";
                        holder.ChoiceOpen = reader.GetAttribute("open");
                        holder.TextEntry = reader.GetAttribute("textEntry") == "1";
                    }
                    else if (local == "bind")
                    {
                        (holder.BindMatch, holder.BindRef) = (reader.GetAttribute("match"), reader.GetAttribute("ref"));
                    }
                    else if (local is "caption" || (local == "value" && parent?.Node is not null && parent.Node.IsField))
                    {
                        entry.Text = new StringBuilder();
                    }
                    else if (local == "items" && parent?.Node is not null && parent.Node.IsField)
                    {
                        if (reader.GetAttribute("save") == "1")
                        {
                            holder.SavedItems ??= [];
                        }
                        else
                        {
                            holder.Items ??= [];
                        }

                        entry.Activity = reader.GetAttribute("save") == "1" ? "save" : "shown";
                    }
                    else if (parent?.Local == "items" && parent.Activity is { } list)
                    {
                        (list == "save" ? holder.SavedItems : holder.Items)!.Add("");
                        entry.Text = new StringBuilder();
                        entry.Activity = list;
                    }
                    else if (local is "event" or "calculate" or "validate" or "variables")
                    {
                        entry.Activity = local == "event" ? reader.GetAttribute("activity") ?? "event" : local;
                        if (local == "validate" && reader.GetAttribute("nullTest") == "error")
                        {
                            holder.Required = true;
                        }
                    }
                    else if (local == "script")
                    {
                        string activity = "script";
                        for (int i = open.Count - 1; i >= 0; i--)
                        {
                            if (open[i].Activity is { } a && open[i].Local is "event" or "calculate" or "validate" or "variables")
                            {
                                activity = a;
                                break;
                            }
                        }

                        scripts.Add((holder, activity, reader.GetAttribute("contentType") ?? "application/x-formcalc"));
                        reader.Skip();
                        read = false;
                        continue;
                    }

                    if (!reader.IsEmptyElement)
                    {
                        open.Add(entry);
                    }
                    else
                    {
                        Close(entry, open, holder);
                    }

                    break;
                }

                case XmlNodeType.EndElement:
                    if (open.Count > 0)
                    {
                        var closing = open[^1];
                        open.RemoveAt(open.Count - 1);
                        Node? holder = null;
                        for (int i = open.Count - 1; i >= 0 && holder is null; i--)
                        {
                            holder = open[i].Node;
                        }

                        Close(closing, open, holder ?? root);
                    }

                    break;
                case XmlNodeType.Text or XmlNodeType.CDATA or XmlNodeType.SignificantWhitespace:
                    for (int i = open.Count - 1; i >= 0; i--)
                    {
                        if (open[i].Text is { } text)
                        {
                            text.Append(reader.Value);
                            break;
                        }
                    }

                    break;
            }
        }

        return root;
    }

    // An element closed: the text it collected given to what it belongs to (a caption, a default value, an item).
    private static void Close(Open closing, List<Open> open, Node holder)
    {
        if (closing.Text is not { } text)
        {
            return;
        }

        string value = text.ToString().Trim();
        var parent = open.Count > 0 ? open[^1] : null;
        if (closing.Local == "caption")
        {
            holder.Caption ??= value.Length > 0 ? value : null;
        }
        else if (closing.Local == "value" && parent?.Node is not null)
        {
            holder.Default ??= value;
        }
        else if (closing.Activity is { } list)
        {
            var items = list == "save" ? holder.SavedItems : holder.Items;
            if (items is { Count: > 0 })
            {
                items[^1] = value;
            }
        }
        else if (parent?.Text is { } outer)
        {
            outer.Append(text);                                                // an inner element's text (a caption's value's text) to the collector around it
        }
    }

    private static int Depth(Node node)
    {
        int depth = 0;
        for (var n = node; n is not null; n = n.Parent)
        {
            depth++;
        }

        return depth;
    }

    // The names, data bindings, values and links of the template's nodes, walked in document order.
    private sealed class Resolution
    {
        private readonly PdfForm _form;
        private readonly Dictionary<string, string> _data = new(StringComparer.Ordinal);
        private readonly Dictionary<string, string> _held = new(StringComparer.Ordinal);
        private readonly Dictionary<string, int> _dataCounts = new(StringComparer.Ordinal);
        private Dictionary<string, string>? _byName;
        private readonly List<DocumentFormSection> _sections = [];
        private readonly List<DocumentTemplateField> _fields = [];
        private readonly Dictionary<Node, string> _paths = [];

        public Resolution(PdfForm form)
        {
            _form = form;
            foreach (var (path, value) in form.XfaData)
            {
                _data.TryAdd(path, value);
            }

            foreach (var field in form.Public)
            {
                _held.TryAdd(PdfForms.Unindexed(field.Name), field.Name);
            }
        }

        public DocumentFormTemplate Resolve(Node root, List<(Node Holder, string Activity, string Type)> scripts)
        {
            Walk(root, "", new Dictionary<string, int>(StringComparer.Ordinal), "", null);
            var issues = new DocumentIssue[scripts.Count];
            for (int i = 0; i < scripts.Count; i++)
            {
                var (holder, activity, type) = scripts[i];
                string where = Where(holder);
                issues[i] = new DocumentIssue(DocumentIssue.NotSupported, $"the XFA template's script of {where} ({activity}, {type}) is not run: a form's scripts never run");
            }

            return new DocumentFormTemplate { Sections = [.. _sections], Fields = [.. _fields], Issues = issues };
        }

        // A node's children: each named one's path among its siblings (unnamed containers pass through), its data scope
        // (null inside the page set: a page's fields bind to no data by name), and its section.
        private void Walk(Node node, string path, Dictionary<string, int> names, string? scope, string? section)
        {
            foreach (var child in node.Children)
            {
                string childPath = path;
                var childNames = names;
                if (child.Name is { } name)
                {
                    names.TryGetValue(name, out int seen);
                    names[name] = seen + 1;
                    childPath = Join(path, seen == 0 ? name : string.Create(CultureInfo.InvariantCulture, $"{name}[{seen}]"));
                    childNames = new Dictionary<string, int>(StringComparer.Ordinal);
                }

                _paths[child] = childPath;
                string? binding = Bind(child, scope);
                if (child.IsField && child.Name is not null)
                {
                    _fields.Add(Field(child, childPath, binding, section));
                    if (child.Element == "exclGroup")
                    {
                        continue;                                              // its buttons are its choices, not fields of their own
                    }
                }

                string? childScope = child.IsField ? scope : child.Element is "pageSet" or "pageArea" ? null : binding ?? scope;
                string? childSection = section;
                if (child.IsSection)
                {
                    _sections.Add(new DocumentFormSection(child.Name!, childPath) { Parent = section });
                    childSection = childPath;
                }

                Walk(child, childPath, childNames, childScope, childSection);
            }
        }

        // The data path a node binds to (its scope for a container's children), or null for none.
        private string? Bind(Node node, string? scope)
        {
            switch (node.BindMatch)
            {
                case "none":
                    return node.IsField ? null : scope;
                case "global":
                    return node.Name is { } global ? Global(global) : scope;
                case "dataRef" when node.BindRef is { } reference:
                    return Reference(reference, scope ?? "");
            }

            if (node.BindRef is { } given && node.BindMatch is null)
            {
                return Reference(given, scope ?? "");
            }

            if (node.Name is not { } name || scope is null)
            {
                return node.IsField ? null : scope;                            // unnamed: passed through; a page's fields bind to nothing by name
            }

            string key = scope + "\u0000" + name;
            _dataCounts.TryGetValue(key, out int seen);
            _dataCounts[key] = seen + 1;
            return Join(scope, seen == 0 ? name : string.Create(CultureInfo.InvariantCulture, $"{name}[{seen}]"));
        }

        // A bind reference: $ (the scope), $record and $data (the data's root), "!" names and a SOM path of names with [n] indices.
        private static string Reference(string reference, string scope)
        {
            string r = reference.Trim();
            string path;
            if (r == "$")
            {
                return scope;
            }

            if (r.StartsWith("$.", StringComparison.Ordinal))
            {
                path = Join(scope, r[2..]);
            }
            else if (r.StartsWith("$record.", StringComparison.Ordinal))
            {
                path = Join(First(scope), r[8..]);
            }
            else if (r.StartsWith("$data.", StringComparison.Ordinal) || r.StartsWith("!", StringComparison.Ordinal))
            {
                path = r[(r.IndexOf('.') + 1)..];
            }
            else
            {
                path = Join(scope, r);
            }

            return Normal(path);
        }

        // A path in the data's convention: [0] dropped, [*] read as the first.
        private static string Normal(string path) => path.Replace("[0]", "", StringComparison.Ordinal).Replace("[*]", "", StringComparison.Ordinal);

        // The record a scope is in: its first name (the data's top element).
        private static string First(string scope) => scope.IndexOf('.') is >= 0 and var dot ? scope[..dot] : scope;

        // The first data value named `name` anywhere (global binding).
        private string? Global(string name)
        {
            if (_byName is null)
            {
                _byName = new Dictionary<string, string>(StringComparer.Ordinal);
                foreach (var (path, _) in _form.XfaData)
                {
                    string last = path[(path.LastIndexOf('.') + 1)..];
                    int bracket = last.IndexOf('[');
                    _byName.TryAdd(bracket >= 0 ? last[..bracket] : last, path);
                }
            }

            return _byName.TryGetValue(name, out var found) ? found : null;
        }

        private DocumentTemplateField Field(Node node, string path, string? binding, string? section)
        {
            var kind = Kind(node);
            var options = new List<DocumentFieldOption>();
            var flags = DocumentFieldFlags.None;
            if (node.Element == "exclGroup")
            {
                // A radio group: each button's on value (its first item), shown by its caption.
                foreach (var button in node.Children)
                {
                    if (button.Element == "field" && (button.SavedItems ?? button.Items) is { Count: > 0 } on)
                    {
                        options.Add(new DocumentFieldOption(button.Caption ?? on[0], on[0]));
                    }
                }
            }
            else if (node.Items is { } shown)
            {
                var saved = node.SavedItems ?? shown;
                for (int i = 0; i < shown.Count; i++)
                {
                    options.Add(new DocumentFieldOption(shown[i], i < saved.Count ? saved[i] : shown[i]));
                }
            }
            else if (node.SavedItems is { } only)
            {
                foreach (string item in only)
                {
                    options.Add(new DocumentFieldOption(item, item));
                }
            }

            if (node.ReadOnly)
            {
                flags |= DocumentFieldFlags.ReadOnly;
            }

            if (node.Required)
            {
                flags |= DocumentFieldFlags.Required;
            }

            if (node.MultiLine)
            {
                flags |= DocumentFieldFlags.Multiline;
            }

            if (node.Ui == "passwordEdit")
            {
                flags |= DocumentFieldFlags.Password;
            }

            if (node.Ui == "choiceList" && node.ChoiceOpen == "multiSelect")
            {
                flags |= DocumentFieldFlags.MultiSelect;
            }

            if (kind == DocumentFieldKind.Combo && node.TextEntry)
            {
                flags |= DocumentFieldFlags.Editable;
            }

            string? value = binding is not null && _data.TryGetValue(binding, out var data) ? data : node.Default;
            string? held = _held.TryGetValue(path, out var twin) ? twin : binding is not null && _held.TryGetValue(binding, out var own) ? own : null;
            return new DocumentTemplateField(node.Name!, path, kind)
            {
                Section = section, Caption = node.Caption, Binding = binding, Value = value, Options = options, Flags = flags, Field = held,
            };
        }

        private static DocumentFieldKind Kind(Node node) => node.Element == "exclGroup" ? DocumentFieldKind.Radio : node.Ui switch
        {
            "checkButton" => DocumentFieldKind.CheckBox,
            "choiceList" => node.ChoiceOpen is "always" or "multiSelect" ? DocumentFieldKind.List : DocumentFieldKind.Combo,
            "button" or "imageEdit" => DocumentFieldKind.PushButton,
            "signature" => DocumentFieldKind.Signature,
            _ => DocumentFieldKind.Text,
        };

        // Where a script stands: its holder's path, or the template's top.
        private string Where(Node holder)
        {
            for (var n = holder; n is not null; n = n.Parent)
            {
                if (_paths.TryGetValue(n, out var path) && path.Length > 0)
                {
                    return path;
                }
            }

            return "the template";
        }

        private static string Join(string a, string b) => a.Length == 0 ? b : string.Concat(a, ".", b);
    }
}
