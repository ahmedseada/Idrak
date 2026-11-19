// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using System.Text;

namespace Idrak.Data.Pdf;

/// <summary>How far what changed after a signature goes, in the order of DocMDP's permissions (ISO 32000-2, 12.8.2.2).</summary>
internal enum PdfChangeLevel : byte
{
    /// <summary>Nothing changed.</summary>
    None = 0,

    /// <summary>Validation data added (the DSS, document timestamps, metadata, the information): allowed by every permission.</summary>
    Validation = 1,

    /// <summary>Forms filled and signature fields signed (permission 2).</summary>
    FormFilling = 2,

    /// <summary>Annotations added, changed or removed (permission 3).</summary>
    Annotations = 3,

    /// <summary>Anything else: what no permission allows.</summary>
    Other = 4,
}

/// <summary>What changed after a signature: allowed or not, how far it goes, and what it was, for a person.</summary>
internal sealed record PdfChangeReport(DocumentSignatureChanges Changes, PdfChangeLevel Level, string Detail)
{
    public static PdfChangeReport Nothing { get; } = new(DocumentSignatureChanges.None, PdfChangeLevel.None, "nothing changed after signing: the signature covers the whole file");
}

/// <summary>
/// Changes after signing found by comparing the signed revision with the current one: the revision is opened alone (the
/// file's bytes to where the signature's end), every object a later revision writes (or frees) is compared with what it was,
/// and each one that differs is classified by its place in the current document (found by walking it from the catalog:
/// the DSS, the information and the metadata; the form, its fields and widgets and their appearances, signature
/// dictionaries; pages, their annotations and appearances, their content and resources; anything else), as a validator
/// does: validation data, form filling (field values and appearances, new signature fields), annotations, or other. The
/// changes are allowed unless they are "other" or go past the signature's DocMDP permission (pyHanko's rule); a field its
/// FieldMDP lock names changed is "other". New objects nothing reaches change nothing.
/// </summary>
internal static class PdfSignatureChanges
{
    private enum Role : byte
    {
        Catalog,
        Info,
        Metadata,
        Dss,
        AcroForm,
        Field,
        Widget,
        Signature,
        FormResources,
        FormAppearance,
        PageTree,
        Page,
        AnnotsArray,
        Annotation,
        AnnotationAppearance,
        Content,
        Resources,
        Other,
    }

    public static PdfChangeReport Analyse(PdfSignatureEntry entry, int? permission)
    {
        if (entry.RangeProblem is { } problem)
        {
            return new PdfChangeReport(DocumentSignatureChanges.NotAllowed, PdfChangeLevel.Other, $"not allowed: what was signed cannot be told: {problem}");
        }

        if (entry.Whole)
        {
            return PdfChangeReport.Nothing;
        }

        var document = entry.Context.Document;
        var file = document.File;
        var xref = file.Xref;
        if (xref.Repaired || xref.Sections.Count == 0)
        {
            return new PdfChangeReport(DocumentSignatureChanges.NotAllowed, PdfChangeLevel.Other,
                "not allowed: the file's cross-reference was rebuilt by scanning, so the revision signed cannot be told apart from later ones");
        }

        var later = xref.Sections.Where(s => s.Offset + xref.Shift >= entry.SignedEnd).ToList();
        if (later.Count == 0)
        {
            using var tail = PdfWindow.Read(file.Source, entry.SignedEnd, Math.Min(4096, file.Source.Length - entry.SignedEnd));
            return file.Source.Length - entry.SignedEnd <= 4096 && tail.Span.Trim(" \r\n\t\f\0"u8).IsEmpty
                ? new PdfChangeReport(DocumentSignatureChanges.Allowed, PdfChangeLevel.None, "allowed: nothing but line ends follow the signed bytes")
                : new PdfChangeReport(DocumentSignatureChanges.NotAllowed, PdfChangeLevel.Other, "not allowed: bytes that are no revision follow the signed ones");
        }

        var changed = new SortedSet<int>();
        var freed = new SortedSet<int>();
        foreach (var section in later)
        {
            foreach (var (number, item) in section.Entries)
            {
                if (number <= 0)
                {
                    continue;
                }

                if (item.Kind == PdfEntryKind.Free)
                {
                    freed.Add(number);
                }
                else
                {
                    changed.Add(number);
                }
            }
        }

        PdfFile signed;
        Stream prefix;
        try
        {
            prefix = PdfPrefixStream.For(file.Source, entry.SignedEnd);
            signed = PdfFile.Open(prefix, file.Name + " (the revision signed)", document.Options);
            PdfSecurity.Attach(signed, document.Options.Password);
        }
        catch (Exception e) when (e is InvalidDataException or NotSupportedException)
        {
            return new PdfChangeReport(DocumentSignatureChanges.NotAllowed, PdfChangeLevel.Other, $"not allowed: the revision signed does not open alone ({e.Message})");
        }

        try
        {
            return Compare(entry, permission, document, signed, changed, freed);
        }
        catch (InvalidDataException e)
        {
            return new PdfChangeReport(DocumentSignatureChanges.NotAllowed, PdfChangeLevel.Other, $"not allowed: the revisions cannot be compared ({e.Message})");
        }
        finally
        {
            signed.Dispose();
            prefix.Dispose();
        }
    }

    private static PdfChangeReport Compare(PdfSignatureEntry entry, int? permission, PdfDocument document, PdfFile signed, SortedSet<int> changed, SortedSet<int> freed)
    {
        var file = document.File;
        var found = new List<(PdfChangeLevel Level, string What)>();
        var differs = new List<int>();
        foreach (int number in changed)
        {
            var now = file.Get(number);
            var stream = now.AsStream;
            if (stream is not null && (stream.Dictionary["Type"u8].IsName("XRef"u8) || stream.Dictionary["Type"u8].IsName("ObjStm"u8)))
            {
                continue;                                                      // the file's structure, not the document
            }

            var before = signed.Get(number);
            if (!before.IsNull && Same(signed, before, file, now, 0))
            {
                continue;                                                      // written again as it was
            }

            differs.Add(number);
        }

        var roles = Roles(file, differs);
        foreach (int number in differs)
        {
            var before = signed.Get(number);
            var now = file.Get(number);
            if (!roles.TryGetValue(number, out var role))
            {
                continue;                                                      // reached by nothing: changes nothing
            }

            found.Add(Classify(entry, signed, file, number, role.Role, role.Name, before, now));
        }

        foreach (int number in freed)
        {
            var before = signed.Get(number);
            if (before.IsNull)
            {
                continue;
            }

            var dictionary = before.AsDictionary;
            bool annotation = dictionary is not null && dictionary.Contains("Rect"u8) && dictionary.Contains("Subtype"u8) && !dictionary["Subtype"u8].IsName("Widget"u8);
            found.Add(annotation ? (PdfChangeLevel.Annotations, $"an annotation (object {number}) deleted") : (PdfChangeLevel.Other, $"object {number} deleted"));
        }

        var level = found.Count == 0 ? PdfChangeLevel.None : found.Max(f => f.Level);
        bool allowed = !(level == PdfChangeLevel.Other || (permission is { } p && (int)level > p));
        var reasons = found.Where(f => f.Level == level).Select(f => f.What).Distinct().ToList();
        string what = level switch
        {
            PdfChangeLevel.None => "no change (later revisions wrote what was signed again as it was)",
            PdfChangeLevel.Validation => "validation data added",
            PdfChangeLevel.FormFilling => "form filling",
            PdfChangeLevel.Annotations => "annotations",
            _ => "other changes",
        };

        var detail = new StringBuilder(allowed ? "allowed: " : "not allowed: ").Append(what);
        if (reasons.Count > 0)
        {
            detail.Append(" (").Append(string.Join("; ", reasons.Take(4)));
            if (reasons.Count > 4)
            {
                detail.Append("; ").Append(reasons.Count - 4).Append(" more");
            }

            detail.Append(')');
        }

        if (permission is { } perm)
        {
            detail.Append(", under ").Append(entry.Certification ? "its certification's" : "the").Append(" permission ").Append(perm).Append(perm switch
            {
                1 => " (no changes)",
                2 => " (form filling)",
                _ => " (form filling and annotations)",
            });
        }

        return new PdfChangeReport(allowed ? DocumentSignatureChanges.Allowed : DocumentSignatureChanges.NotAllowed, level, detail.ToString());
    }

    // One object that differs from the revision signed, classified by its role.
    private static (PdfChangeLevel, string) Classify(PdfSignatureEntry entry, PdfFile signed, PdfFile file, int number, Role role, string? name, PdfValue before, PdfValue now)
    {
        bool added = before.IsNull;
        var old = before.AsDictionary;
        var current = now.AsDictionary;
        switch (role)
        {
            case Role.Info or Role.Metadata or Role.Dss:
                return (PdfChangeLevel.Validation, role switch { Role.Dss => "the DSS's validation data", Role.Info => "the information", _ => "the metadata" });
            case Role.Catalog:
                var level = PdfChangeLevel.None;
                var parts = new List<string>();
                var wasForm = old?["AcroForm"u8] ?? PdfValue.Null;
                var isForm = current?["AcroForm"u8] ?? PdfValue.Null;
                foreach (string key in ChangedKeys(signed, old, file, current))
                {
                    var (keyLevel, text) = key switch
                    {
                        "DSS" or "Extensions" or "Metadata" => (PdfChangeLevel.Validation, $"the catalog's /{key}"),
                        "AcroForm" when wasForm.IsNull => (PdfChangeLevel.FormFilling, "a form added"),
                        "AcroForm" when wasForm.IsReference && isForm.IsReference && wasForm.AsReference == isForm.AsReference => (PdfChangeLevel.None, ""),
                        "AcroForm" when wasForm.Kind == PdfKind.Dictionary && isForm.Kind == PdfKind.Dictionary => AcroForm(signed, file, wasForm.AsDictionary, isForm.AsDictionary),
                        _ => (PdfChangeLevel.Other, $"the catalog's /{key}"),
                    };
                    level = (PdfChangeLevel)Math.Max((int)level, (int)keyLevel);
                    if (text.Length > 0)
                    {
                        parts.Add(text);
                    }
                }

                return (level, parts.Count == 0 ? "the catalog" : string.Join(", ", parts) + " changed");
            case Role.AcroForm:
                return AcroForm(signed, file, old, current);
            case Role.Signature:
                return added
                    ? (current?["Type"u8].IsName("DocTimeStamp"u8) == true ? PdfChangeLevel.Validation : PdfChangeLevel.FormFilling,
                       current?["Type"u8].IsName("DocTimeStamp"u8) == true ? "a document timestamp" : "a signature added")
                    : (PdfChangeLevel.Other, $"a signature dictionary (object {number}) changed");
            case Role.Field or Role.Widget:
                return Field(entry, signed, file, number, role, name ?? "", old, current);
            case Role.FormResources or Role.FormAppearance:
                return (PdfChangeLevel.FormFilling, added ? $"an appearance of the field '{name}'" : $"an appearance of the field '{name}' rewritten");
            case Role.Annotation or Role.AnnotationAppearance:
                return (PdfChangeLevel.Annotations, added ? "an annotation added" : "an annotation changed");
            case Role.Page or Role.AnnotsArray:
                return Page(signed, file, number, role, old, before, current, now);
            case Role.Content:
                return (PdfChangeLevel.Other, $"a page's content (object {number}) {(added ? "added" : "changed")}");
            case Role.Resources:
                return (PdfChangeLevel.Other, $"a page's resources (object {number}) {(added ? "added" : "changed")}");
            case Role.PageTree:
                return (PdfChangeLevel.Other, $"the page tree (object {number}) changed");
            default:
                return (PdfChangeLevel.Other, $"object {number} {(added ? "added" : "changed")}");
        }
    }

    private static (PdfChangeLevel, string) AcroForm(PdfFile signed, PdfFile file, PdfDictionary? old, PdfDictionary? current)
    {
        var level = PdfChangeLevel.None;
        var parts = new List<string>();
        foreach (string key in ChangedKeys(signed, old, file, current))
        {
            switch (key)
            {
                case "Fields":
                    var before = References(signed, old?["Fields"u8] ?? PdfValue.Null);
                    var after = References(file, current?["Fields"u8] ?? PdfValue.Null);
                    if (before.Except(after).Any())
                    {
                        level = PdfChangeLevel.Other;
                        parts.Add("a field removed from the form");
                        break;
                    }

                    foreach (var added in after.Except(before))
                    {
                        bool signature = SignatureField(file, file.Dictionary(PdfValue.Reference(added.Number, added.Generation)));
                        level = (PdfChangeLevel)Math.Max((int)level, (int)(signature ? PdfChangeLevel.FormFilling : PdfChangeLevel.Other));
                        parts.Add(signature ? "a signature field added" : "a form field added");
                    }

                    break;
                case "SigFlags" or "NeedAppearances" or "DA" or "DR" or "Q":
                    level = (PdfChangeLevel)Math.Max((int)level, (int)PdfChangeLevel.FormFilling);
                    parts.Add($"the form's /{key}");
                    break;
                default:
                    level = PdfChangeLevel.Other;
                    parts.Add($"the form's /{key}");
                    break;
            }
        }

        return (level, parts.Count == 0 ? "the form" : string.Join(", ", parts.Distinct()));
    }

    private static (PdfChangeLevel, string) Field(PdfSignatureEntry entry, PdfFile signed, PdfFile file, int number, Role role, string name, PdfDictionary? old, PdfDictionary? current)
    {
        bool signature = SignatureField(file, current);
        if (old is null)
        {
            return signature ? (PdfChangeLevel.FormFilling, "a signature field added")
                : role == Role.Widget ? (PdfChangeLevel.FormFilling, $"a widget of the field '{name}' added") : (PdfChangeLevel.Other, $"the field '{name}' added");
        }

        var level = PdfChangeLevel.None;
        var parts = new List<string>();
        foreach (string key in ChangedKeys(signed, old, file, current))
        {
            switch (key)
            {
                case "V" when signature && !old["V"u8].IsNull:
                    level = PdfChangeLevel.Other;
                    parts.Add($"the signed field '{name}' given another signature");
                    break;
                case "V":
                    if (Locked(entry, name))
                    {
                        level = PdfChangeLevel.Other;
                        parts.Add($"the field '{name}', which the signature locks, filled");
                        break;
                    }

                    level = (PdfChangeLevel)Math.Max((int)level, (int)PdfChangeLevel.FormFilling);
                    parts.Add(signature ? $"the signature field '{name}' signed" : $"the field '{name}' filled");
                    break;
                case "AP" or "AS" or "MK" or "Kids":
                    level = (PdfChangeLevel)Math.Max((int)level, (int)PdfChangeLevel.FormFilling);
                    parts.Add($"the field '{name}''s /{key}");
                    break;
                default:
                    level = PdfChangeLevel.Other;
                    parts.Add($"the field '{name}''s /{key}");
                    break;
            }
        }

        return (level, parts.Count == 0 ? $"the field '{name}' (object {number})" : string.Join(", ", parts.Distinct()));
    }

    private static bool Locked(PdfSignatureEntry entry, string name) =>
        entry.LocksAll ? !entry.LockExcluded.Contains(name, StringComparer.Ordinal) : entry.Locked.Contains(name, StringComparer.Ordinal);

    private static (PdfChangeLevel, string) Page(PdfFile signed, PdfFile file, int number, Role role, PdfDictionary? old, PdfValue before, PdfDictionary? current, PdfValue now)
    {
        if (role == Role.Page && (old is null || current is null))
        {
            return (PdfChangeLevel.Other, old is null ? $"a page (object {number}) added" : $"a page (object {number}) changed");
        }

        var level = PdfChangeLevel.None;
        var parts = new List<string>();
        bool annotations = role == Role.AnnotsArray;
        if (role == Role.Page)
        {
            foreach (string key in ChangedKeys(signed, old, file, current))
            {
                if (key == "Annots")
                {
                    annotations = true;
                }
                else
                {
                    level = PdfChangeLevel.Other;
                    parts.Add($"the page's /{key}");
                }
            }
        }

        if (annotations)
        {
            var was = References(signed, role == Role.Page ? old!["Annots"u8] : before);
            var nowList = References(file, role == Role.Page ? current!["Annots"u8] : now);
            foreach (var removed in was.Except(nowList))
            {
                bool widget = signed.Dictionary(PdfValue.Reference(removed.Number, removed.Generation))?["Subtype"u8].IsName("Widget"u8) == true;
                level = (PdfChangeLevel)Math.Max((int)level, (int)(widget ? PdfChangeLevel.Other : PdfChangeLevel.Annotations));
                parts.Add(widget ? "a field's widget removed from a page" : "an annotation removed");
            }

            foreach (var added in nowList.Except(was))
            {
                var annotation = file.Dictionary(PdfValue.Reference(added.Number, added.Generation));
                bool widget = annotation?["Subtype"u8].IsName("Widget"u8) == true;
                var (addedLevel, text) = !widget ? (PdfChangeLevel.Annotations, "an annotation added")
                    : SignatureField(file, annotation) ? (PdfChangeLevel.FormFilling, "a signature field's widget added")
                    : (PdfChangeLevel.Other, "a form field's widget added");
                level = (PdfChangeLevel)Math.Max((int)level, (int)addedLevel);
                parts.Add(text);
            }
        }

        return (level, parts.Count == 0 ? $"a page (object {number})" : string.Join(", ", parts.Distinct()));
    }

    // Whether a field (or a widget of one) is a signature field: /FT /Sig on it or a parent.
    private static bool SignatureField(PdfFile file, PdfDictionary? field)
    {
        for (int depth = 0; field is not null && depth < 32; depth++)
        {
            var type = file.Resolve(field["FT"u8]);
            if (!type.IsNull)
            {
                return type.IsName("Sig"u8);
            }

            field = file.Dictionary(field["Parent"u8]);
        }

        return false;
    }

    // The references an array (direct or indirect) lists.
    private static HashSet<PdfReference> References(PdfFile file, PdfValue value)
    {
        var set = new HashSet<PdfReference>();
        foreach (var item in file.Resolve(value).AsArray?.Items ?? [])
        {
            if (item.IsReference)
            {
                set.Add(item.AsReference);
            }
        }

        return set;
    }

    // The keys whose values differ between two dictionaries (added and removed ones too).
    private static List<string> ChangedKeys(PdfFile oldFile, PdfDictionary? old, PdfFile newFile, PdfDictionary? current)
    {
        var keys = new List<string>();
        old ??= PdfDictionary.Empty;
        current ??= PdfDictionary.Empty;
        for (int i = 0; i < current.Count; i++)
        {
            var key = current.KeyAt(i);
            var value = current.ValueAt(i);
            if (!old.TryGet(key.Bytes, out var was) ? !value.IsNull : !Same(oldFile, was, newFile, value, 0))
            {
                keys.Add(key.ToString());
            }
        }

        for (int i = 0; i < old.Count; i++)
        {
            var key = old.KeyAt(i);
            if (!old.ValueAt(i).IsNull && !current.Contains(key.Bytes) && !keys.Contains(key.ToString()))
            {
                keys.Add(key.ToString());
            }
        }

        return keys;
    }

    // Whether two values are the same: numbers by value, references by number and generation, containers member by member
    // (a dictionary's keys in any order), streams by their dictionaries (their stored form aside) and decoded bytes.
    private static bool Same(PdfFile oldFile, PdfValue a, PdfFile newFile, PdfValue b, int depth)
    {
        if (depth > 64)
        {
            return false;
        }

        if (a.TryNumber(out double x) && b.TryNumber(out double y))
        {
            return x == y;
        }

        if (a.Kind != b.Kind)
        {
            return false;
        }

        switch (a.Kind)
        {
            case PdfKind.Null:
                return true;
            case PdfKind.Boolean:
                return a.AsBoolean == b.AsBoolean;
            case PdfKind.Reference:
                return a.AsReference == b.AsReference;
            case PdfKind.Name:
                return a.AsName!.Bytes.SequenceEqual(b.AsName!.Bytes);
            case PdfKind.String:
                return a.AsString!.Bytes.AsSpan().SequenceEqual(b.AsString!.Bytes);
            case PdfKind.Array:
                var left = a.AsArray!.Items;
                var right = b.AsArray!.Items;
                if (left.Length != right.Length)
                {
                    return false;
                }

                for (int i = 0; i < left.Length; i++)
                {
                    if (!Same(oldFile, left[i], newFile, right[i], depth + 1))
                    {
                        return false;
                    }
                }

                return true;
            case PdfKind.Dictionary:
                return SameDictionary(oldFile, a.AsDictionary!, newFile, b.AsDictionary!, depth, stream: false);
            case PdfKind.Stream:
                if (!SameDictionary(oldFile, a.AsDictionary!, newFile, b.AsDictionary!, depth, stream: true))
                {
                    return false;
                }

                try
                {
                    using var first = oldFile.Decode(a.AsStream!);
                    using var second = newFile.Decode(b.AsStream!);
                    return first.Span.SequenceEqual(second.Span);
                }
                catch (InvalidDataException)
                {
                    return false;
                }

            default:
                return false;
        }
    }

    private static bool SameDictionary(PdfFile oldFile, PdfDictionary a, PdfFile newFile, PdfDictionary b, int depth, bool stream)
    {
        static bool Stored(PdfName key, bool stream) => stream && (key.Is("Length"u8) || key.Is("Filter"u8) || key.Is("DecodeParms"u8) || key.Is("DL"u8));

        for (int i = 0; i < a.Count; i++)
        {
            var key = a.KeyAt(i);
            if (Stored(key, stream) || a.ValueAt(i).IsNull)
            {
                continue;
            }

            if (!b.TryGet(key.Bytes, out var other) || !Same(oldFile, a.ValueAt(i), newFile, other, depth + 1))
            {
                return false;
            }
        }

        for (int i = 0; i < b.Count; i++)
        {
            var key = b.KeyAt(i);
            if (!Stored(key, stream) && !b.ValueAt(i).IsNull && !a.Contains(key.Bytes))
            {
                return false;
            }
        }

        return true;
    }

    // Each object of `wanted` placed in the current document by walking it from the trailer: validation data first, then
    // the form, then pages and their annotations, then pages' content and resources, then the rest of the catalog; the walk
    // stops once every one is placed.
    private static Dictionary<int, (Role Role, string? Name)> Roles(PdfFile file, List<int> wanted)
    {
        var roles = new Dictionary<int, (Role, string?)>();
        var want = new HashSet<int>(wanted);
        int placed = 0;
        bool Done() => placed == want.Count;

        void Mark(PdfValue value, Role role, string? name = null)
        {
            if (value.IsReference && !roles.ContainsKey(value.AsReference.Number))
            {
                roles[value.AsReference.Number] = (role, name);
                if (want.Contains(value.AsReference.Number))
                {
                    placed++;
                }
            }
        }

        // Every object a value reaches (not yet placed) gets `role`.
        void Subtree(PdfValue root, Role role, string? name = null)
        {
            var stack = new Stack<(PdfValue Value, int Depth)>();
            stack.Push((root, 0));
            while (stack.Count > 0 && !Done())
            {
                var (value, depth) = stack.Pop();
                if (depth > file.MaxDepth)
                {
                    continue;
                }

                if (value.IsReference)
                {
                    if (roles.ContainsKey(value.AsReference.Number))
                    {
                        continue;
                    }

                    Mark(value, role, name);
                    value = Safe(file, value);
                }

                if (value.AsArray is { } array)
                {
                    foreach (var item in array.Items)
                    {
                        stack.Push((item, depth + 1));
                    }
                }
                else if (value.AsDictionary is { } dictionary)
                {
                    for (int i = 0; i < dictionary.Count; i++)
                    {
                        if (!dictionary.KeyAt(i).Is("Parent"u8) && !dictionary.KeyAt(i).Is("P"u8))
                        {
                            stack.Push((dictionary.ValueAt(i), depth + 1));
                        }
                    }
                }
            }
        }

        var trailer = file.Trailer;
        Mark(trailer["Root"u8], Role.Catalog);
        Subtree(trailer["Info"u8], Role.Info);
        var catalog = file.Dictionary(trailer["Root"u8]) ?? PdfDictionary.Empty;
        Subtree(catalog["DSS"u8], Role.Dss);
        Subtree(catalog["Metadata"u8], Role.Metadata);

        // The form: the dictionary, every field and widget with its name, signature dictionaries, appearances, resources.
        var formValue = catalog["AcroForm"u8];
        Mark(formValue, Role.AcroForm);
        if (Safe(file, formValue).AsDictionary is { } form && !Done())
        {
            Mark(form["Fields"u8], Role.AcroForm);
            var stack = new Stack<(PdfValue Node, string Prefix, int Depth)>();
            foreach (var field in file.Resolve(form["Fields"u8]).AsArray?.Items ?? [])
            {
                stack.Push((field, "", 1));
            }

            var seen = new HashSet<int>();
            while (stack.Count > 0 && !Done())
            {
                var (node, prefix, depth) = stack.Pop();
                if ((node.IsReference && !seen.Add(node.AsReference.Number)) || depth > file.MaxDepth || Safe(file, node).AsDictionary is not { } field)
                {
                    continue;
                }

                string? own = file.Resolve(field["T"u8]).AsString is { } t ? PdfText.Decode(t.Bytes) : null;
                string name = own is null ? prefix : prefix.Length == 0 ? own : prefix + "." + own;
                Mark(node, own is null ? Role.Widget : Role.Field, name);
                Subtree(field["AP"u8], Role.FormAppearance, name);
                Subtree(field["MK"u8], Role.FormAppearance, name);
                if (SignatureField(file, field))
                {
                    Subtree(field["V"u8], Role.Signature, name);
                }

                foreach (var kid in file.Resolve(field["Kids"u8]).AsArray?.Items ?? [])
                {
                    stack.Push((kid, name, depth + 1));
                }
            }

            Subtree(form["DR"u8], Role.FormResources, "(the form's resources)");
        }

        // Pages and their annotations, then their content and resources.
        var pages = new List<PdfValue>();
        var nodes = new Stack<(PdfValue Node, int Depth)>();
        nodes.Push((catalog["Pages"u8], 1));
        var visited = new HashSet<int>();
        while (nodes.Count > 0)
        {
            var (node, depth) = nodes.Pop();
            if ((node.IsReference && !visited.Add(node.AsReference.Number)) || depth > file.MaxDepth || Safe(file, node).AsDictionary is not { } dictionary)
            {
                continue;
            }

            if (file.Resolve(dictionary["Kids"u8]).AsArray is { } kids && !dictionary["Type"u8].IsName("Page"u8))
            {
                Mark(node, Role.PageTree);
                Mark(dictionary["Kids"u8], Role.PageTree);
                for (int i = kids.Count - 1; i >= 0; i--)
                {
                    nodes.Push((kids[i], depth + 1));
                }
            }
            else
            {
                Mark(node, Role.Page);
                pages.Add(node);
            }
        }

        foreach (var page in pages)
        {
            if (Done())
            {
                break;
            }

            var dictionary = Safe(file, page).AsDictionary ?? PdfDictionary.Empty;
            Mark(dictionary["Annots"u8], Role.AnnotsArray);
            foreach (var annotation in file.Resolve(dictionary["Annots"u8]).AsArray?.Items ?? [])
            {
                bool widget = Safe(file, annotation).AsDictionary?["Subtype"u8].IsName("Widget"u8) == true;
                if (widget && annotation.IsReference && roles.ContainsKey(annotation.AsReference.Number))
                {
                    continue;                                                  // a field's widget, placed with the form
                }

                Mark(annotation, widget ? Role.Widget : Role.Annotation, widget ? "(a widget outside the form)" : null);
                if (Safe(file, annotation).AsDictionary is { } a)
                {
                    Subtree(a["AP"u8], widget ? Role.FormAppearance : Role.AnnotationAppearance, widget ? "(a widget outside the form)" : null);
                    Subtree(a["Popup"u8], Role.Annotation);
                }
            }
        }

        foreach (var page in pages)
        {
            if (Done())
            {
                break;
            }

            var dictionary = Safe(file, page).AsDictionary ?? PdfDictionary.Empty;
            Subtree(dictionary["Contents"u8], Role.Content);
            Subtree(dictionary["Resources"u8], Role.Resources);
        }

        for (int i = 0; i < catalog.Count && !Done(); i++)
        {
            var key = catalog.KeyAt(i);
            if (!(key.Is("Pages"u8) || key.Is("AcroForm"u8) || key.Is("DSS"u8) || key.Is("Metadata"u8)))
            {
                Subtree(catalog.ValueAt(i), Role.Other);
            }
        }

        return roles;
    }

    // A value resolved, or null when it cannot be read (the walk passes over damage; a limit still stops it).
    private static PdfValue Safe(PdfFile file, PdfValue value)
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
}

/// <summary>
/// The first bytes of a document's source as a stream of their own (a revision signed, opened alone): in place for a file
/// in memory, else read through the source at any offset. The source stays its document's.
/// </summary>
internal sealed class PdfPrefixStream : Stream
{
    private readonly PdfSource _source;
    private readonly long _length;
    private long _position;

    private PdfPrefixStream(PdfSource source, long length) => (_source, _length) = (source, length);

    public static Stream For(PdfSource source, long length)
    {
        if (source.InMemory && MemoryMarshal.TryGetArray(source.Memory, out var segment) && segment.Array is { } array)
        {
            return new MemoryStream(array, segment.Offset, (int)Math.Min(length, segment.Count), writable: false, publiclyVisible: true);
        }

        return new PdfPrefixStream(source, Math.Min(length, source.Length));
    }

    public override bool CanRead => true;

    public override bool CanSeek => true;

    public override bool CanWrite => false;

    public override long Length => _length;

    public override long Position
    {
        get => _position;
        set => _position = Math.Clamp(value, 0, _length);
    }

    public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));

    public override int Read(Span<byte> buffer)
    {
        int wanted = (int)Math.Min(buffer.Length, _length - _position);
        if (wanted <= 0)
        {
            return 0;
        }

        int read = _source.Read(_position, buffer[..wanted]);
        _position += read;
        return read;
    }

    public override long Seek(long offset, SeekOrigin origin) => Position = origin switch
    {
        SeekOrigin.Begin => offset,
        SeekOrigin.Current => _position + offset,
        _ => _length + offset,
    };

    public override void Flush()
    {
    }

    public override void SetLength(long value) => throw new NotSupportedException("A revision is read, not written.");

    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException("A revision is read, not written.");
}
