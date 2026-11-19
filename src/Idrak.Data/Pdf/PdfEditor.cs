// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Security.Cryptography;

namespace Idrak.Data.Pdf;

/// <summary>A page of a document being edited: a page of a PDF document (the one edited or another), or one the editor adds.</summary>
internal sealed class PdfEditPage(PdfDocument? document, int index)
{
    /// <summary>The document the page is of; null for a page the editor adds.</summary>
    public PdfDocument? Document { get; } = document;

    /// <summary>Its index in <see cref="Document"/>.</summary>
    public int Index { get; } = index;

    /// <summary>Its rotation as the editor turned it; null for its own.</summary>
    public int? Rotation { get; set; }

    /// <summary>Content drawn over it (a page of a document), or its own (a page the editor adds).</summary>
    public PdfPageBuilder? Content { get; set; }

    /// <summary>Its annotations changed by form edits (widgets of flattened fields left out, those of added fields added), or null.</summary>
    public PdfAnnotsEdit? Annots { get; set; }

    /// <summary>Its annotations added or removed and its content redacted (<see cref="PdfPageEdits"/>), or null.</summary>
    public PdfPageEdits? Edits { get; set; }
}

/// <summary>
/// The library's PDF editor (contract C23, from <see cref="PdfWriter.Edit"/>): the information replaced, pages inserted
/// from other PDF documents, removed, moved, turned and added, content drawn over a page (a stamp, a watermark, page
/// numbers: the page builder's calls, written as a stream after the page's own), and the document saved:
/// <list type="bullet">
/// <item><see cref="DocumentSaveMode.Incremental"/>: the source's bytes copied as they are, then what changed appended under
/// its own numbers and what is new under new ones, with one cross-reference section of the source's kind (a table after a
/// table, a stream over object streams after a stream) pointing back by <c>/Prev</c>; so what earlier signatures cover is
/// untouched. A file that keeps its encryption keeps it (strings and streams encrypted by its own key).</item>
/// <item><see cref="DocumentSaveMode.Rewrite"/>: one clean file of what the catalog reaches, renumbered (pages first), in
/// object streams, streams deflated again where it pays and duplicates written once (<see cref="PdfEditCopier"/>);
/// encrypted as the source was, or as <see cref="Encrypt"/> asks (AES-256), or not at all when the owner password opened
/// the source and the encryption is removed.</item>
/// </list>
/// The documents pages are inserted from stay open until <see cref="Save"/>, and an image given to a page is read then.
/// Form fields are filled, added and flattened through <see cref="PdfFormEditor"/> (D9). <see cref="Sign"/> (D12) adds
/// each signature as an update of its own after the others (<see cref="PdfSigning"/>): over the source as it is when
/// nothing else changed, else over the edited file, written first to a temporary file. Annotations are added (their look
/// drawn) and removed, bookmarks added (the source's kept), and pages redacted for real (<see cref="Redact"/>: saved as a
/// rewrite only; D10). Not safe from several threads.
/// </summary>
internal sealed partial class PdfEditor : IDocumentEditor
{
    private readonly IDocument _given;
    private readonly PdfDocument _source;
    private readonly DocumentWriteOptions _options;
    private readonly List<PdfEditPage> _pages = [];
    private readonly List<PdfSignRequest> _signatures = [];
    private Stream _destination;
    private readonly Dictionary<IFontFace, PdfWriterFont> _fonts = new(ReferenceEqualityComparer.Instance);
    private readonly List<PdfWriterFont> _fontOrder = [];
    private readonly List<(ImageData? Image, DocumentImageEncoding Encoding, int Number)> _images = [];
    private DocumentInfo? _info;
    private DocumentEncryption? _encryption;
    private PdfFormEditor? _forms;
    private bool _encryptSet, _saved, _disposed;

    public PdfEditor(IDocument given, PdfDocument source, Stream destination, DocumentWriteOptions options)
    {
        (_given, _source, _destination, _options) = (given, source, destination, options);
        for (int i = 0; i < source.PageCount; i++)
        {
            _pages.Add(new PdfEditPage(source, i));
        }
    }

    public IDocument Source => _given;

    public int PageCount => _pages.Count;

    public void SetInfo(DocumentInfo info)
    {
        ArgumentNullException.ThrowIfNull(info);
        ThrowIfDone();
        _info = info;
    }

    /// <summary>The editor's pages, in order (the form editor finds a widget's pages and their annotations here).</summary>
    internal List<PdfEditPage> Pages => _pages;

    public void SetField(string name, params ReadOnlySpan<string> values)
    {
        ThrowIfDone();
        Forms.SetField(name, values);
    }

    public void AddField(DocumentField field)
    {
        ArgumentNullException.ThrowIfNull(field);
        ThrowIfDone();
        Forms.AddField(field, _pages.Count);
    }

    public void Flatten(params ReadOnlySpan<string> names)
    {
        ThrowIfDone();
        Forms.Flatten(names);
    }

    // The source's form as the edits change it, read on the first form call.
    private PdfFormEditor Forms => _forms ??= new PdfFormEditor(this, _source, _options);

    /// <summary>
    /// Signs the document in the signature field <paramref name="field"/> (an empty one of the form, or a new one on
    /// <see cref="SigningOptions.Page"/>) when it is saved: each signature an incremental update of its own, after the
    /// other changes, in the order asked.
    /// </summary>
    public void Sign(string field, SigningOptions options)
    {
        ThrowIfDone();
        PdfSigning.Validate(field, options);
        _signatures.Add(new PdfSignRequest(field, options));
    }

    public void InsertPages(int index, IDocument from, params ReadOnlySpan<int> pages)
    {
        ArgumentNullException.ThrowIfNull(from);
        ThrowIfDone();
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(index, _pages.Count);
        var document = PdfWriter.Own(from, "inserts pages of");
        var added = new PdfEditPage[pages.Length];
        for (int i = 0; i < pages.Length; i++)
        {
            if ((uint)pages[i] >= (uint)document.PageCount)
            {
                throw new ArgumentOutOfRangeException(nameof(pages), pages[i], $"The document pages are inserted from has {document.PageCount} pages (indices from 0).");
            }

            added[i] = new PdfEditPage(document, pages[i]);
        }

        _pages.InsertRange(index, added);
    }

    public void RemovePages(params ReadOnlySpan<int> pages)
    {
        ThrowIfDone();
        var removed = new SortedSet<int>();
        foreach (int page in pages)
        {
            CheckIndex(page, nameof(pages));
            if (!removed.Add(page))
            {
                throw new ArgumentException($"Page {page} is named twice.", nameof(pages));
            }
        }

        foreach (int page in removed.Reverse())
        {
            _pages[page].Content?.Release();
            _pages.RemoveAt(page);
        }
    }

    public void MovePage(int from, int to)
    {
        ThrowIfDone();
        CheckIndex(from, nameof(from));
        CheckIndex(to, nameof(to));
        var page = _pages[from];
        _pages.RemoveAt(from);
        _pages.Insert(to, page);
    }

    public void RotatePage(int index, int rotation)
    {
        ThrowIfDone();
        CheckIndex(index, nameof(index));
        if (rotation % 90 != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(rotation), rotation, "A page is turned to a multiple of 90 degrees (0, 90, 180 or 270).");
        }

        _pages[index].Rotation = (rotation % 360 + 360) % 360;
    }

    public IDocumentPageBuilder AddPage(float width, float height)
    {
        ThrowIfDone();
        if (!(width > 0 && height > 0) || !float.IsFinite(width) || !float.IsFinite(height))
        {
            throw new ArgumentOutOfRangeException(nameof(width), $"A page is {width} x {height} points: sizes are finite and above 0.");
        }

        var content = new PdfPageBuilder(Font, WriteImage, _pages.Count, width, height);
        _pages.Add(new PdfEditPage(null, -1) { Content = content });
        return content;
    }

    public IDocumentPageBuilder Overlay(int index)
    {
        ThrowIfDone();
        CheckIndex(index, nameof(index));
        var page = _pages[index];
        if (page.Content is { } content)
        {
            return content;
        }

        var (display, width, height, _) = page.Document!.Frame(page.Index, page.Rotation);
        page.Content = new PdfPageBuilder(Font, WriteImage, index, width, height, PdfEditPages.Inverse(display), PdfEditPages.Prefix(page.Document, page.Index));
        return page.Content;
    }

    public void Encrypt(DocumentEncryption? encryption)
    {
        ThrowIfDone();
        if (_source.File.Security is { Owner: false })
        {
            throw new InvalidOperationException(
                $"{_source.File.Name} was opened with its user password: its encryption is changed or removed only when its owner password opened it (DocumentOptions.Password).");
        }

        (_encryptSet, _encryption) = (true, encryption);
    }

    public void Save(DocumentSaveMode mode)
    {
        ThrowIfDone();
        if (mode is not (DocumentSaveMode.Incremental or DocumentSaveMode.Rewrite))
        {
            throw new ArgumentOutOfRangeException(nameof(mode), mode, "Not a save mode.");
        }

        if (mode == DocumentSaveMode.Incremental && Redacting)
        {
            throw new InvalidOperationException(
                "A redacted document is saved as a rewrite (DocumentSaveMode.Rewrite): an incremental update keeps the source's bytes, and what was removed with them.");
        }

        _saved = true;
        if (_signatures.Count > 0)
        {
            SaveSigned(mode);
            return;
        }

        if (mode == DocumentSaveMode.Incremental)
        {
            SaveIncremental();
        }
        else
        {
            SaveRewrite();
        }

        _destination.Flush();
    }

    // The signatures, each its own update: over the source as it is when nothing else changed; else the edited file is
    // written to a temporary file (as an update or a rewrite), opened, and signed onto the destination.
    private void SaveSigned(DocumentSaveMode mode)
    {
        bool edited = _info is not null || _encryptSet || _fontOrder.Count > 0 || _images.Count > 0 || !SamePages()
                      || _pages.Exists(p => p.Rotation is not null || p.Content is not null || p.Edits is { Any: true }) || _outlines is not null;
        if (mode == DocumentSaveMode.Incremental && !edited)
        {
            PdfSigning.WriteSigned(_source, _signatures, _destination, _options);
            return;
        }

        var destination = _destination;
        using var spill = PdfSpillStream.Create();
        _destination = spill;
        try
        {
            if (mode == DocumentSaveMode.Incremental)
            {
                SaveIncremental();
            }
            else
            {
                SaveRewrite();
            }
        }
        finally
        {
            _destination = destination;
        }

        spill.Flush();
        // The edited file opens with the password of its encryption: a new one's owner (else user) password, none when it
        // was removed, else the source's own.
        string? password = _encryptSet ? _encryption?.OwnerPassword ?? _encryption?.UserPassword : _source.Options.Password;
        using var based = PdfDocument.Open(spill, _source.File.Name, _source.Options with { Password = password });
        PdfSigning.WriteSigned(based, _signatures, destination, _options);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        foreach (var page in _pages)
        {
            page.Content?.Release();
        }

        _images.Clear();
    }

    /// <summary>The face's font in the edited document, made on its first use (numbered and written when it is saved).</summary>
    public PdfWriterFont Font(IFontFace face)
    {
        if (!_fonts.TryGetValue(face, out var font))
        {
            font = new PdfWriterFont(face, 0, _fontOrder.Count + 1);
            _fonts.Add(face, font);
            _fontOrder.Add(font);
        }

        return font;
    }

    /// <summary>Takes <paramref name="image"/> for the document (written when it is saved); the number a page's resources name it by until then.</summary>
    public int WriteImage(ImageData image, DocumentImageEncoding encoding)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (image.Channels is not (1 or 3))
        {
            throw new ArgumentException($"A document's image is grey (1 channel) or RGB (3), not {image.Channels} channels.", nameof(image));
        }

        _images.Add((image, encoding, 0));
        return -_images.Count;
    }

    // The update: the source's bytes, then what changed and what is new, then a section of the source's kind.
    private void SaveIncremental()
    {
        var file = _source.File;
        if (_encryptSet && (_encryption is not null || file.Security is not null))
        {
            throw new InvalidOperationException(
                "An incremental update keeps the source's bytes, and so its encryption: encryption is set or removed by a rewrite (DocumentSaveMode.Rewrite).");
        }

        var xref = file.Xref;
        bool standsAlone = xref.Repaired || xref.Sections.Count == 0;          // a rebuilt cross-reference: the update lists every object
        var carried = new List<(int Number, PdfEntry Entry)>();
        if (standsAlone)
        {
            for (int n = 1; n < xref.Count; n++)
            {
                if (xref.Entries[n].Kind is PdfEntryKind.InUse or PdfEntryKind.Compressed)
                {
                    carried.Add((n, xref.Entries[n]));
                }
            }
        }

        bool classic = standsAlone ? !carried.Exists(c => c.Entry.Kind == PdfEntryKind.Compressed) : xref.Sections[0].Kind is "table" or "hybrid";
        long size = Math.Max(xref.Count, file.Trailer["Size"u8].TryInteger(out long stated) ? stated : 0);
        long prefix = CopyPrefix(file);
        using var writer = PdfFileWriter.Append(_destination, prefix, xref.Shift, (int)Math.Min(size, int.MaxValue - 1), _options.Compression,
            packing: !classic && _options.ObjectStreams);
        writer.Security = file.Security;
        using var copier = new PdfEditCopier(writer, _options.Compression, recompress: false);
        var own = copier.For(_source, identity: true);

        var catalogValue = file.Trailer["Root"u8];
        var catalog = file.Dictionary(catalogValue) ?? throw new InvalidDataException($"{file.Name}: no document catalog to update (the trailer's /Root is {catalogValue}).");
        var pagesValue = catalog["Pages"u8];
        var rootNode = DictionaryOf(file, pagesValue);
        bool changed = !SamePages();
        int root = 0, rootGeneration = 0;
        bool catalogAgain = false;
        if (pagesValue.IsReference)
        {
            (root, rootGeneration) = (pagesValue.AsReference.Number, pagesValue.AsReference.Generation);
            if (changed)
            {
                writer.Reuse(root, rootGeneration);
            }
        }
        else if (changed)
        {
            if (!catalogValue.IsReference)
            {
                throw new InvalidDataException($"{file.Name}: a catalog that is not an object of its own cannot be updated: save the document as a rewrite.");
            }

            root = writer.Allocate();                                          // a page tree held in the catalog: one of its own now
            catalogAgain = true;
            writer.Reuse(catalogValue.AsReference.Number, catalogValue.AsReference.Generation);
        }

        copier.Root = root;
        bool parentCrops = rootNode.Contains("CropBox"u8), parentRotates = rootNode.Contains("Rotate"u8);
        foreach (var font in _fontOrder)
        {
            font.Number = writer.Allocate();
        }

        _forms?.Prepare(writer, copier, own, rewrite: false);

        // Each page's number: a page of the source keeps its own (written again only when it changes), the others are new.
        // With the source's pages as they were, none comes twice: no set of those met is kept.
        var numbers = new int[_pages.Count];
        var generations = new int[_pages.Count];
        var first = changed ? new HashSet<(PdfDocument, int)>() : null;
        for (int i = 0; i < _pages.Count; i++)
        {
            var page = _pages[i];
            if (page.Document is null)
            {
                numbers[i] = writer.Allocate();
                continue;
            }

            var node = page.Document.PageObject(page.Index).Node;
            bool once = node.IsReference && (first is null || first.Add((page.Document, node.AsReference.Number)));
            if (once && page.Document == _source)
            {
                (numbers[i], generations[i]) = (node.AsReference.Number, node.AsReference.Generation);
                continue;
            }

            numbers[i] = writer.Allocate();
            if (once)
            {
                PdfEditCopier.MapPage(copier.For(page.Document), node.AsReference.Number, numbers[i]);
            }
        }

        int save = 0;
        int Save() => save != 0 ? save : save = PdfEditPages.WriteSave(writer);
        int Image(int placeholder) => ImageNumber(writer, placeholder);
        Targets(numbers, generations, copier, writer);
        for (int i = 0; i < _pages.Count; i++)
        {
            var page = _pages[i];
            if (page.Document is null)
            {
                PdfEditPages.WriteNew(copier, writer, numbers[i], root, page.Content!, page.Rotation ?? 0, parentCrops, parentRotates, Image, page.Annots, page.Edits);
                continue;
            }

            var node = page.Document.PageObject(page.Index).Node;
            bool ownPage = page.Document == _source && node.IsReference && numbers[i] == node.AsReference.Number;
            if (ownPage && !changed && page.Rotation is null && page.Content is null && page.Annots is null && page.Edits is not { Any: true })
            {
                continue;                                                      // as it was, under the parent it had: nothing of it read
            }

            var source = copier.For(page.Document);
            int turned = page.Document.Frame(page.Index).Rotation;
            int rotation = page.Rotation ?? turned;
            if (ownPage)
            {
                var (parent, parentGeneration) = Parent(file, node);
                bool moved = changed && (parent != root || parentGeneration != rootGeneration);
                if (rotation == turned && page.Content is null && !moved && page.Annots is null && page.Edits is not { Any: true })
                {
                    continue;                                                  // as it was
                }

                writer.Reuse(numbers[i], generations[i]);
                var how = moved || parent == 0
                    ? new PdfPageWrite(numbers[i], root, rotation, Materialize: moved, ParentCrops: moved && parentCrops, ParentRotates: moved && parentRotates, rootGeneration)
                    : new PdfPageWrite(numbers[i], parent, rotation, Materialize: false, ParentCrops: false, ParentRotates: false, parentGeneration);
                PdfEditPages.Write(copier, writer, source, page.Index, how, page.Content, Image, Save, page.Annots, page.Edits);
                continue;
            }

            PdfEditPages.Write(copier, writer, source, page.Index, new PdfPageWrite(numbers[i], root, rotation, true, parentCrops, parentRotates, rootGeneration),
                page.Content, Image, Save, page.Annots, page.Edits);
        }

        int outlines = WriteOutlines(writer, numbers, generations);
        _forms?.Write(writer, copier, own, rewrite: false, w => w.EditPage is { } at && _pages.IndexOf(at) is >= 0 and var k ? (numbers[k], generations[k]) : (0, 0));
        Action<PdfSyntax>? acroForm = _forms is { CatalogChanged: true } forms ? s => forms.WriteAcroForm(s, copier, own) : null;
        if (changed)
        {
            copier.WriteObject(root, s => PageTree(s, copier, own, rootNode, numbers, generations));
        }

        if (catalogAgain || acroForm is not null || outlines > 0)
        {
            if (!catalogValue.IsReference)
            {
                throw new InvalidDataException($"{file.Name}: a catalog that is not an object of its own cannot be updated: save the document as a rewrite.");
            }

            int catalogRoot = pagesValue.IsReference && !catalogAgain ? pagesValue.AsReference.Number : root;
            int catalogRootGeneration = pagesValue.IsReference && !catalogAgain ? pagesValue.AsReference.Generation : 0;
            if (catalogRoot == 0)
            {
                throw new InvalidDataException($"{file.Name}: a page tree held in the catalog cannot be updated with the form or bookmarks: save the document as a rewrite.");
            }

            if (!catalogAgain)
            {
                writer.Reuse(catalogValue.AsReference.Number, catalogValue.AsReference.Generation);
            }

            copier.WriteObject(catalogValue.AsReference.Number, s => Catalog(s, copier, own, catalog, catalogRoot, catalogRootGeneration, aes256: false, keepVersion: true, acroForm, outlines));
        }

        foreach (var font in _fontOrder)
        {
            font.Write(writer, _options.EmbedFonts);
        }

        var infoValue = file.Trailer["Info"u8];
        int info = 0, infoGeneration = 0;
        if (_info is not null)
        {
            var entries = PdfDocumentBuilder.InfoEntries(Stamped(_info));
            if (infoValue.IsReference)
            {
                (info, infoGeneration) = (infoValue.AsReference.Number, infoValue.AsReference.Generation);
                writer.Reuse(info, infoGeneration);
            }
            else if (entries.Count > 0)
            {
                info = writer.Allocate();
            }

            if (info > 0)
            {
                copier.WriteObject(info, s => PdfDocumentBuilder.WriteInfo(s, entries));
            }
        }

        copier.Drain();
        var ids = Identifiers(file, keepSecond: true);
        writer.FinishUpdate(classic, standsAlone ? -1 : xref.Sections[0].Offset, size, carried, s =>
        {
            copier.Entry(s, own, "Root"u8, catalogValue);
            if (info > 0)
            {
                s.Raw("/Info "u8).Integer(info).Space().Integer(infoGeneration).Raw(" R"u8);
            }
            else if (!infoValue.IsNull)
            {
                copier.Entry(s, own, "Info"u8, infoValue);
            }

            if (!file.Trailer["Encrypt"u8].IsNull)
            {
                copier.Entry(s, own, "Encrypt"u8, file.Trailer["Encrypt"u8]);
            }

            if (ids is { } id)
            {
                s.Raw("/ID["u8).HexString(id.First).HexString(id.Second).Byte((byte)']');
            }
        });
    }

    // The clean file: the catalog, the page tree and the pages first, then what they reach, each once.
    private void SaveRewrite()
    {
        var file = _source.File;
        string version = _options.Version ?? _source.StatedVersion;
        if (!PdfDocumentBuilder.IsVersion(version))
        {
            if (_options.Version is not null)
            {
                throw new ArgumentException($"A PDF version is 1.0 to 1.7 or 2.0, not '{version}' (DocumentWriteOptions.Version).", "options");
            }

            version = "1.7";                                                   // a source stating none the writer writes
        }

        PdfSecurity? security = null, created = null;
        PdfValue kept = PdfValue.Null;
        if (_encryptSet)
        {
            if (_encryption is { } encryption)
            {
                security = created = PdfSecurity.Create(encryption);
            }
        }
        else if (file.Security is { } own)
        {
            (security, kept) = (own, file.Trailer["Encrypt"u8]);               // written again with the source's own key
        }

        try
        {
            if (created is not null && string.CompareOrdinal(version, "1.7") < 0)
            {
                version = "1.7";                                               // AES-256: PDF 2.0, or 1.7 with Adobe's extension level 8
            }

            bool classic = string.CompareOrdinal(version, "1.5") < 0;
            using var writer = new PdfFileWriter(_destination, version, _options.Compression, packing: !classic && _options.ObjectStreams);
            using var copier = new PdfEditCopier(writer, _options.Compression, recompress: true);
            int catalogNumber = writer.Allocate(), root = writer.Allocate();
            copier.Root = root;
            var numbers = new int[_pages.Count];
            for (int i = 0; i < numbers.Length; i++)
            {
                numbers[i] = writer.Allocate();
            }

            int encrypt = 0;
            if (security is not null)
            {
                encrypt = writer.Allocate();
                (writer.Security, writer.PlainNumber, writer.EncryptNumber) = (security, encrypt, encrypt);
            }

            var source = copier.For(_source);
            var first = new HashSet<(PdfDocument, int)>();
            for (int i = 0; i < _pages.Count; i++)
            {
                var page = _pages[i];
                var node = page.Document?.PageObject(page.Index).Node ?? PdfValue.Null;
                if (node.IsReference && first.Add((page.Document!, node.AsReference.Number)))
                {
                    PdfEditCopier.MapPage(copier.For(page.Document!), node.AsReference.Number, numbers[i]);
                }
            }

            foreach (var font in _fontOrder)
            {
                font.Number = writer.Allocate();
            }

            _forms?.Prepare(writer, copier, source, rewrite: true);
            Action<PdfSyntax>? acroForm = _forms is { CatalogChanged: true } forms ? s => forms.WriteAcroForm(s, copier, source) : null;
            var catalog = DictionaryOf(file, file.Trailer["Root"u8]);
            bool extension = created is not null && version != "2.0";
            var generations = new int[numbers.Length];
            int outlines = WriteOutlines(writer, numbers, generations);
            copier.WriteObject(catalogNumber, s => Catalog(s, copier, source, catalog, root, 0, extension, keepVersion: false, acroForm, outlines));
            copier.WriteObject(root, s =>
            {
                s.Raw("<</Type/Pages/Count "u8).Integer(numbers.Length).Raw("/Kids["u8);
                for (int i = 0; i < numbers.Length; i++)
                {
                    if (i > 0)
                    {
                        s.Space();
                    }

                    s.Reference(numbers[i]);
                }

                s.Raw("]>>"u8);
            });

            int save = 0;
            int Save() => save != 0 ? save : save = PdfEditPages.WriteSave(writer);
            int Image(int placeholder) => ImageNumber(writer, placeholder);
            Targets(numbers, generations, copier, writer);
            for (int i = 0; i < _pages.Count; i++)
            {
                var page = _pages[i];
                if (page.Document is null)
                {
                    PdfEditPages.WriteNew(copier, writer, numbers[i], root, page.Content!, page.Rotation ?? 0, false, false, Image, page.Annots, page.Edits);
                    continue;
                }

                int rotation = page.Rotation ?? page.Document.Frame(page.Index).Rotation;
                PdfEditPages.Write(copier, writer, copier.For(page.Document), page.Index, new PdfPageWrite(numbers[i], root, rotation, true, false, false),
                    page.Content, Image, Save, page.Annots, page.Edits);
            }

            _forms?.Write(writer, copier, source, rewrite: true, w => w.EditPage is { } at && _pages.IndexOf(at) is >= 0 and var k ? (numbers[k], 0) : (0, 0));

            foreach (var font in _fontOrder)
            {
                font.Write(writer, _options.EmbedFonts);
            }

            int info = 0;
            var infoValue = file.Trailer["Info"u8];
            if (_info is not null)
            {
                var entries = PdfDocumentBuilder.InfoEntries(Stamped(_info));
                if (entries.Count > 0)
                {
                    info = writer.Allocate();
                    copier.WriteObject(info, s => PdfDocumentBuilder.WriteInfo(s, entries));
                }
            }
            else if (infoValue.IsReference)
            {
                info = Math.Max(0, copier.Reference(source, infoValue.AsReference).Number);
            }
            else if (infoValue.Kind == PdfKind.Dictionary)
            {
                info = writer.Allocate();
                copier.WriteObject(info, s => copier.Value(s, source, infoValue));
            }

            if (created is not null)
            {
                created.WriteDictionary(writer.BeginObject(encrypt));
                writer.EndObject();
            }
            else if (security is not null)
            {
                var dictionary = file.Resolve(kept);                           // its strings as stored: the dictionary is never encrypted
                copier.WriteObject(encrypt, s => copier.Value(s, source, dictionary));
            }

            copier.Drain();
            var ids = security is not null && created is null ? Identifiers(file, keepSecond: true, keep: true) : Identifiers(file, keepSecond: false, made: created is not null);
            writer.Finish(catalogNumber, info, ids?.First ?? [], classic, ids?.Second ?? []);
        }
        finally
        {
            created?.Dispose();                                                // the new file key cleared
        }
    }

    // The flat page tree an update writes: every page, under the root's number, with what the root passed on kept.
    private void PageTree(PdfSyntax s, PdfEditCopier copier, PdfEditCopier.Source own, PdfDictionary rootNode, int[] numbers, int[] generations)
    {
        s.Raw("<</Type/Pages/Count "u8).Integer(numbers.Length).Raw("/Kids["u8);
        for (int i = 0; i < numbers.Length; i++)
        {
            if (i > 0)
            {
                s.Space();
            }

            s.Integer(numbers[i]).Space().Integer(generations[i]).Raw(" R"u8);
        }

        s.Byte((byte)']');
        for (int i = 0; i < rootNode.Count; i++)
        {
            var key = rootNode.KeyAt(i);
            if (!(key.Is("Type"u8) || key.Is("Kids"u8) || key.Is("Count"u8) || key.Is("Parent"u8)))
            {
                copier.Entry(s, own, key.Bytes, rootNode.ValueAt(i));
            }
        }

        s.Raw(">>"u8);
    }

    // The catalog written again: its page tree the result's; its other entries copied (its /Version only when the header
    // stays, in an update; its form as `acroForm` writes it when the form's tree changed; its bookmarks the tree written
    // again when bookmarks were added); Adobe's extension level 8 when AES-256 is written before PDF 2.0.
    private static void Catalog(PdfSyntax s, PdfEditCopier copier, PdfEditCopier.Source source, PdfDictionary catalog, int root, int rootGeneration, bool aes256, bool keepVersion,
        Action<PdfSyntax>? acroForm = null, int outlines = 0)
    {
        s.Raw("<</Type/Catalog/Pages "u8).Integer(root).Space().Integer(rootGeneration).Raw(" R"u8);
        if (outlines > 0)
        {
            s.Raw("/Outlines "u8).Reference(outlines);                       // bookmarks added: the tree written again
        }

        bool extended = false;
        acroForm?.Invoke(s);                                                   // the form as edited (none left: no entry)
        for (int i = 0; i < catalog.Count; i++)
        {
            var key = catalog.KeyAt(i);
            if (key.Is("Type"u8) || key.Is("Pages"u8) || (!keepVersion && key.Is("Version"u8)) || (acroForm is not null && key.Is("AcroForm"u8))
                || (outlines > 0 && key.Is("Outlines"u8)))
            {
                continue;
            }

            if (aes256 && key.Is("Extensions"u8) && source.File.Resolve(catalog.ValueAt(i)).AsDictionary is { } extensions)
            {
                s.Raw("/Extensions<<"u8);
                for (int k = 0; k < extensions.Count; k++)
                {
                    if (!extensions.KeyAt(k).Is("ADBE"u8))
                    {
                        copier.Entry(s, source, extensions.KeyAt(k).Bytes, extensions.ValueAt(k));
                    }
                }

                s.Raw("/ADBE<</BaseVersion/1.7/ExtensionLevel 8>>>>"u8);
                extended = true;
                continue;
            }

            copier.Entry(s, source, key.Bytes, catalog.ValueAt(i));
        }

        if (aes256 && !extended)
        {
            s.Raw(PdfEditPages.Aes256Extension);
        }

        s.Raw(">>"u8);
    }

    // The image an editor's page named by `placeholder`, written on its first use; its number.
    private int ImageNumber(PdfFileWriter file, int placeholder)
    {
        int k = -placeholder - 1;
        var (image, encoding, number) = _images[k];
        if (number == 0)
        {
            number = file.Allocate();
            PdfWriterImages.Write(file, number, image!, encoding, _options);
            _images[k] = (null, encoding, number);                             // the pixels let go
        }

        return number;
    }

    // The information as given, its modification date now when the output need not be the same every time.
    private DocumentInfo Stamped(DocumentInfo info) => _options.Deterministic ? info : info with { Modified = info.Modified ?? DateTimeOffset.Now };

    // The identifiers the trailer gives: the options' Id when given (as the second of an update, or both); the source's
    // kept (its first always, as its encryption may need; its second in an update unless a new one is wanted); one made
    // when the file is newly encrypted or need not be the same every time; else none.
    private (byte[] First, byte[] Second)? Identifiers(PdfFile file, bool keepSecond, bool keep = false, bool made = false)
    {
        byte[]? first = null, second = null;
        if (file.Resolve(file.Trailer["ID"u8]).AsArray is { Count: > 0 } ids)
        {
            first = file.Resolve(ids[0]).AsString?.Bytes;
            second = ids.Count > 1 ? file.Resolve(ids[1]).AsString?.Bytes : first;
        }

        byte[]? given = _options.Id.IsEmpty ? null : _options.Id.ToArray();
        if (keep)
        {
            return first is null ? null : (first, second ?? first);           // the source's encryption kept: its identifiers too (its key may use the first)
        }

        if (keepSecond && first is not null)
        {
            return (first, given ?? (_options.Deterministic ? second ?? first : RandomNumberGenerator.GetBytes(16)));
        }

        if (given is not null)
        {
            return (given, given);
        }

        if (first is not null && !made)
        {
            return (first, second ?? first);
        }

        if (made || !_options.Deterministic)
        {
            byte[] id = RandomNumberGenerator.GetBytes(16);
            return (id, id);
        }

        return null;
    }

    // The source's bytes written to the destination as they are, then a line end when they lack one; how many.
    private long CopyPrefix(PdfFile file)
    {
        var source = file.Source;
        long length = source.Length;
        if (source.InMemory)
        {
            _destination.Write(source.Memory.Span);
        }
        else
        {
            byte[] buffer = ArrayPool<byte>.Shared.Rent(1 << 20);
            try
            {
                for (long at = 0; at < length;)
                {
                    int read = source.Read(at, buffer.AsSpan(0, (int)Math.Min(buffer.Length, length - at)));
                    if (read <= 0)
                    {
                        throw new IOException($"{file.Name}: the file ended at byte {at:N0} of {length:N0} while it was copied.");
                    }

                    _destination.Write(buffer, 0, read);
                    at += read;
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(buffer);
            }
        }

        Span<byte> last = stackalloc byte[1];
        if (length > 0 && source.Read(length - 1, last) == 1 && last[0] is not ((byte)'\n' or (byte)'\r'))
        {
            _destination.WriteByte((byte)'\n');
            return length + 1;
        }

        return length;
    }

    // Whether the pages are the source's own, in order.
    private bool SamePages()
    {
        if (_pages.Count != _source.PageCount)
        {
            return false;
        }

        for (int i = 0; i < _pages.Count; i++)
        {
            if (_pages[i].Document != _source || _pages[i].Index != i)
            {
                return false;
            }
        }

        return true;
    }

    // A page's /Parent as a number and generation (0 when it names none).
    private static (int Number, int Generation) Parent(PdfFile file, PdfValue node)
    {
        var parent = DictionaryOf(file, node)["Parent"u8];
        return parent.IsReference ? (parent.AsReference.Number, parent.AsReference.Generation) : (0, 0);
    }

    // A dictionary of the file, or an empty one when it is none or cannot be read.
    private static PdfDictionary DictionaryOf(PdfFile file, PdfValue value)
    {
        try
        {
            return file.Dictionary(value) ?? PdfDictionary.Empty;
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            return PdfDictionary.Empty;
        }
    }

    private void CheckIndex(int index, string name)
    {
        if ((uint)index >= (uint)_pages.Count)
        {
            throw new ArgumentOutOfRangeException(name, index, $"The document has {_pages.Count} pages (indices from 0).");
        }
    }

    private void ThrowIfDone()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_saved)
        {
            throw new InvalidOperationException("The edited PDF document is saved: an editor saves once.");
        }
    }
}
