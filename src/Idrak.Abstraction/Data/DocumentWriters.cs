// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;

namespace Idrak.Abstraction.Data;

/// <summary>How a writer stores an image of a page.</summary>
public enum DocumentImageEncoding : byte
{
    /// <summary>
    /// Measured per image: photographs and scans as JPEG, line art and images of few levels exact
    /// (<see cref="DocumentWriteOptions.LineArtLevels"/>, then <see cref="DocumentWriteOptions.SmoothSteps"/>).
    /// </summary>
    Auto,

    /// <summary>Exact pixels, compressed without loss.</summary>
    Flate,

    /// <summary>JPEG, through <see cref="ImageEncoders"/>' "jpeg", at <see cref="DocumentWriteOptions.JpegQuality"/>.</summary>
    Jpeg,
}

/// <summary>How a writer embeds the fonts of its text.</summary>
public enum DocumentFontEmbedding : byte
{
    /// <summary>Only the glyphs used, written through <see cref="FontWriters"/> (a page in one font is not the font's whole size).</summary>
    Subset,

    /// <summary>The whole font file, as read.</summary>
    Whole,
}

/// <summary>How an edited document is saved.</summary>
public enum DocumentSaveMode : byte
{
    /// <summary>The source's bytes kept as they are and the changes appended, so earlier signatures stay valid.</summary>
    Incremental,

    /// <summary>One clean file: only what the document uses, written anew (earlier signatures do not survive).</summary>
    Rewrite,
}

/// <summary>
/// How a document is written. The same input writes the same bytes (<see cref="Deterministic"/>): no date or identifier is
/// written unless given. The environment variables named below set the image encoding, the JPEG quality, the font
/// embedding and the save mode.
/// </summary>
public sealed record DocumentWriteOptions
{
    /// <summary>The variable that sets <see cref="ImageEncoding"/> (auto, flate or jpeg): "IDRAK_DOCUMENT_IMAGE_ENCODING".</summary>
    public const string ImageEncodingVariable = "IDRAK_DOCUMENT_IMAGE_ENCODING";

    /// <summary>The variable that sets <see cref="JpegQuality"/> (1 to 100): "IDRAK_DOCUMENT_JPEG_QUALITY".</summary>
    public const string JpegQualityVariable = "IDRAK_DOCUMENT_JPEG_QUALITY";

    /// <summary>The variable that sets <see cref="EmbedFonts"/> (subset or whole): "IDRAK_DOCUMENT_EMBED_FONTS".</summary>
    public const string EmbedFontsVariable = "IDRAK_DOCUMENT_EMBED_FONTS";

    /// <summary>The variable that sets <see cref="SaveMode"/> (incremental or rewrite): "IDRAK_DOCUMENT_SAVE_MODE".</summary>
    public const string SaveModeVariable = "IDRAK_DOCUMENT_SAVE_MODE";

    private readonly int _compression = 6, _jpegQuality = DocumentSettings.Int(JpegQualityVariable, 1, 100) ?? 90, _lineArtLevels = 32;
    private readonly float _smoothSteps = 0.25f;
    private readonly DocumentFormOptions _forms = new();

    /// <summary>How hard streams are compressed without loss: 0 (stored) to 9 (smallest), 6 unless set.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Not from 0 to 9.</exception>
    public int Compression
    {
        get => _compression;
        init => _compression = value is >= 0 and <= 9 ? value : throw new ArgumentOutOfRangeException(nameof(Compression), value, "Compression is from 0 to 9.");
    }

    /// <summary>How images are stored (<see cref="DocumentImageEncoding.Auto"/> unless <see cref="ImageEncodingVariable"/> sets it).</summary>
    public DocumentImageEncoding ImageEncoding { get; init; } = DocumentSettings.Choice<DocumentImageEncoding>(ImageEncodingVariable) ?? DocumentImageEncoding.Auto;

    /// <summary>The quality of JPEG images, 1 to 100 (90 unless <see cref="JpegQualityVariable"/> sets it).</summary>
    /// <exception cref="ArgumentOutOfRangeException">Not from 1 to 100.</exception>
    public int JpegQuality
    {
        get => _jpegQuality;
        init => _jpegQuality = value is >= 1 and <= 100 ? value : throw new ArgumentOutOfRangeException(nameof(JpegQuality), value, "JPEG quality is from 1 to 100.");
    }

    /// <summary>
    /// <see cref="DocumentImageEncoding.Auto"/>'s first rule: an image with at most this many levels in each channel (line
    /// art, text, a few colours) is stored exact; one with more is measured by <see cref="SmoothSteps"/> (32).
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not from 1 to 256.</exception>
    public int LineArtLevels
    {
        get => _lineArtLevels;
        init => _lineArtLevels = value is >= 1 and <= 256 ? value : throw new ArgumentOutOfRangeException(nameof(LineArtLevels), value, "Levels are from 1 to 256.");
    }

    /// <summary>
    /// <see cref="DocumentImageEncoding.Auto"/>'s second rule, for an image of more levels than <see cref="LineArtLevels"/>:
    /// the share of neighbouring pixel pairs (each pixel against its right and its lower neighbour, the largest channel
    /// difference) that differ by 1 to 16 levels. At or above it the image is a photograph or a scan (smooth shading,
    /// sensor noise, paper grain: JPEG); below it, line art with smoothed edges (flat areas and sharp steps: stored exact)
    /// (0.25).
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not from 0 to 1.</exception>
    public float SmoothSteps
    {
        get => _smoothSteps;
        init => _smoothSteps = value is >= 0f and <= 1f ? value : throw new ArgumentOutOfRangeException(nameof(SmoothSteps), value, "A share is from 0 to 1.");
    }

    /// <summary>How fonts are embedded (<see cref="DocumentFontEmbedding.Subset"/> unless <see cref="EmbedFontsVariable"/> sets it).</summary>
    public DocumentFontEmbedding EmbedFonts { get; init; } = DocumentSettings.Choice<DocumentFontEmbedding>(EmbedFontsVariable) ?? DocumentFontEmbedding.Subset;

    /// <summary>The format version written ("1.7"), or null for the writer's own.</summary>
    public string? Version { get; init; }

    /// <summary>Whether the same input writes the same bytes: no date or identifier unless given (true).</summary>
    public bool Deterministic { get; init; } = true;

    /// <summary>The file's identifier to write, or empty for none (a writer that is not <see cref="Deterministic"/> may make one).</summary>
    public ReadOnlyMemory<byte> Id { get; init; }

    /// <summary>Whether small objects are packed into compressed streams where the format can (smaller files; true).</summary>
    public bool ObjectStreams { get; init; } = true;

    /// <summary>How an edit is saved when the app does not say (<see cref="DocumentSaveMode.Incremental"/> unless <see cref="SaveModeVariable"/> sets it).</summary>
    public DocumentSaveMode SaveMode { get; init; } = DocumentSettings.Choice<DocumentSaveMode>(SaveModeVariable) ?? DocumentSaveMode.Incremental;

    /// <summary>How form fields filled, added or flattened are made to look: their fonts, fitted sizes and spacing.</summary>
    public DocumentFormOptions Forms
    {
        get => _forms;
        init => _forms = value ?? throw new ArgumentNullException(nameof(Forms));
    }
}

// The checks of a run's glyph matrices (IDocumentPageBuilder.Glyphs with matrices).
internal static class DocumentGlyphMatrices
{
    /// <summary>Checks that there is one matrix for all the glyphs or one a glyph, each finite and turning no glyph flat.</summary>
    /// <exception cref="ArgumentException">The message says which.</exception>
    public static void Check(int glyphs, ReadOnlySpan<DocumentMatrix> matrices)
    {
        if (matrices.Length != 1 && matrices.Length != glyphs)
        {
            throw new ArgumentException($"A run's glyphs take one matrix for all of them or one each: {matrices.Length} for {glyphs} glyphs.", nameof(matrices));
        }

        foreach (var m in matrices)
        {
            double area = (double)m.A * m.D - (double)m.B * m.C;
            if (!float.IsFinite(m.A) || !float.IsFinite(m.B) || !float.IsFinite(m.C) || !float.IsFinite(m.D) || area == 0 || !double.IsFinite(area))
            {
                throw new ArgumentException($"A glyph's matrix ({m.A}, {m.B}, {m.C}, {m.D}) is finite and does not turn the glyph flat.", nameof(matrices));
            }
        }
    }

    /// <summary>The run's size when every matrix is upright at one size above 0, (size, 0, 0, -size); else null.</summary>
    public static float? UprightSize(ReadOnlySpan<DocumentMatrix> matrices)
    {
        if (matrices.IsEmpty)
        {
            return null;
        }

        float size = matrices[0].A;
        foreach (var m in matrices)
        {
            if (!(m.B == 0f && m.C == 0f && m.A == size && m.D == -size))
            {
                return null;
            }
        }

        return size > 0f ? size : null;
    }
}

/// <summary>
/// One page being written: images, paths and text in the order given (each later one over the earlier), the graphics state
/// saved and restored around clips, and the page's annotations and links. The page is written to the destination when the
/// next page starts or the document is finished.
/// </summary>
public interface IDocumentPageBuilder
{
    /// <summary>The page's index.</summary>
    int Index { get; }

    /// <summary>Its width in points.</summary>
    float Width { get; }

    /// <summary>Its height in points.</summary>
    float Height { get; }

    /// <summary>
    /// Places <paramref name="image"/> in <paramref name="box"/> (page points from the top-left corner): its values in
    /// [0, 1] become 8-bit levels (rounded), grey or RGB, stored as <paramref name="encoding"/> asks.
    /// </summary>
    void Image(ImageData image, DocumentBox box, DocumentImageEncoding encoding);

    /// <summary>Draws <paramref name="path"/>: filled and stroked as it says, under the current clip.</summary>
    void Path(DocumentPath path);

    /// <summary>
    /// Writes a run of glyphs of <paramref name="face"/> at <paramref name="size"/> points, each where
    /// <paramref name="glyphs"/> places it, standing for <paramref name="text"/> (in logical order: the reader gets it back
    /// as text that can be searched and copied, in every script), painted as <paramref name="mode"/> says in
    /// <paramref name="colour"/>. The caller shapes the text (<see cref="TextShapers"/>); the writer embeds the font.
    /// </summary>
    void Glyphs(IFontFace face, float size, ReadOnlySpan<DocumentGlyph> glyphs, string text, DocumentTextMode mode, DocumentColour colour);

    /// <summary>
    /// Writes a run of glyphs of <paramref name="face"/> as <see cref="Glyphs(IFontFace, float, ReadOnlySpan{DocumentGlyph}, string, DocumentTextMode, DocumentColour)"/>
    /// does, each drawn through its matrix instead of upright: <paramref name="matrices"/> holds one for every glyph, or
    /// one for all, each mapping the glyph's outline at one unit an em (font units over the face's units per em, y up)
    /// onto the page (points, y down) from the glyph's origin, its own shift not used, as
    /// <see cref="DocumentTextRun.GlyphMatrix"/> reads it back: text along a turned line or round a ring, slanted or
    /// stretched glyphs. (size, 0, 0, -size) is upright at size points: what the call without matrices writes. The glyphs
    /// are written in the order given (a reader takes them in that order).
    /// </summary>
    /// <exception cref="ArgumentException">Neither one matrix nor one a glyph, or a matrix that is not finite or turns a glyph flat.</exception>
    /// <exception cref="NotSupportedException">The writer writes upright glyphs only and a matrix is not upright at the run's one size (the message says so).</exception>
    void Glyphs(IFontFace face, ReadOnlySpan<DocumentGlyph> glyphs, ReadOnlySpan<DocumentMatrix> matrices, string text, DocumentTextMode mode, DocumentColour colour)
    {
        ArgumentNullException.ThrowIfNull(face);
        DocumentGlyphMatrices.Check(glyphs.Length, matrices);
        if (glyphs.IsEmpty)
        {
            return;                                                            // nothing to show
        }

        if (DocumentGlyphMatrices.UprightSize(matrices) is not { } size)
        {
            throw new NotSupportedException($"This document writer's pages ({GetType().Name}) write glyphs upright only, at one size a run; a turned, slanted or stretched glyph is not written.");
        }

        Glyphs(face, size, glyphs, text, mode, colour);
    }

    /// <summary>Saves the graphics state (the clip), to be restored by <see cref="Restore"/>.</summary>
    void Save();

    /// <summary>Restores the graphics state last saved.</summary>
    /// <exception cref="InvalidOperationException">Nothing is saved.</exception>
    void Restore();

    /// <summary>Clips what follows to <paramref name="path"/>'s inside (by its <see cref="DocumentPath.Fill"/> rule), within the current clip.</summary>
    void Clip(DocumentPath path);

    /// <summary>Adds an annotation to the page.</summary>
    /// <exception cref="NotSupportedException">The writer does not write annotations yet (the message says so).</exception>
    void AddAnnotation(DocumentAnnotation annotation);

    /// <summary>Adds a link to the page.</summary>
    /// <exception cref="NotSupportedException">The writer does not write links yet (the message says so).</exception>
    void AddLink(DocumentLink link);
}

/// <summary>
/// A document being written: pages added in order (each written to the destination when the next starts), files attached,
/// form fields, bookmarks, page labels, encryption and a signature; <see cref="Finish"/> writes what ends the file. A
/// builder disposed without <see cref="Finish"/> leaves the file unfinished.
/// </summary>
public interface IDocumentBuilder : IDisposable
{
    /// <summary>What the document says about itself (only what is given is written).</summary>
    DocumentInfo Info { get; set; }

    /// <summary>The pages added so far.</summary>
    int PageCount { get; }

    /// <summary>Adds a page of <paramref name="width"/> x <paramref name="height"/> points (the page before it is written now).</summary>
    IDocumentPageBuilder AddPage(float width, float height);

    /// <summary>Adds page <paramref name="index"/> of <paramref name="from"/> with what it uses (merging and splitting documents); more content can go over it.</summary>
    /// <exception cref="NotSupportedException">The writer cannot copy pages of that document's format (the message says why).</exception>
    IDocumentPageBuilder CopyPage(IDocument from, int index);

    /// <summary>Attaches a file: <paramref name="content"/> read to its end now (the stream stays the caller's).</summary>
    void Attach(string name, string mediaType, Stream content);

    /// <summary>
    /// Adds a form field: its widgets name their pages (the page being built or one added later: a page written already
    /// cannot take a widget) and each gets an appearance showing its value (<see cref="DocumentWriteOptions.Forms"/>).
    /// </summary>
    /// <exception cref="ArgumentException">The field cannot be written as given (the message says why).</exception>
    /// <exception cref="NotSupportedException">The writer does not write forms.</exception>
    void AddField(DocumentField field);

    /// <summary>Adds a bookmark at the top level (with its children).</summary>
    /// <exception cref="NotSupportedException">The writer does not write bookmarks yet.</exception>
    void AddOutline(DocumentOutline outline);

    /// <summary>Sets the page labels.</summary>
    /// <exception cref="NotSupportedException">The writer does not write page labels yet.</exception>
    void SetPageLabels(IReadOnlyList<DocumentPageLabel> labels);

    /// <summary>Encrypts the document as it is written.</summary>
    /// <exception cref="NotSupportedException">The writer does not encrypt yet.</exception>
    void Encrypt(DocumentEncryption encryption);

    /// <summary>Signs the document in the signature field <paramref name="field"/> (made when it does not exist) when it is finished.</summary>
    /// <exception cref="NotSupportedException">The writer does not sign yet.</exception>
    void Sign(string field, SigningOptions options);

    /// <summary>Writes the last page and what ends the file.</summary>
    void Finish();
}

/// <summary>
/// A document being changed and saved to a new destination: fields filled, annotations added and removed, pages inserted,
/// removed, moved and rotated, content over a page, encryption and a signature; <see cref="Save"/> writes it, keeping the
/// source's bytes (an incremental update) or as one clean file.
/// </summary>
public interface IDocumentEditor : IDisposable
{
    /// <summary>The document being changed.</summary>
    IDocument Source { get; }

    /// <summary>The number of pages, with the changes so far.</summary>
    int PageCount { get; }

    /// <summary>Replaces what the document says about itself.</summary>
    void SetInfo(DocumentInfo info);

    /// <summary>
    /// Sets field <paramref name="name"/>'s value (several for a list that takes them; none clears it); its appearance is
    /// made to show it (<see cref="DocumentWriteOptions.Forms"/>), so a reader shows the value without making its own.
    /// </summary>
    /// <exception cref="ArgumentException">No such field, or a value it does not take.</exception>
    void SetField(string name, params ReadOnlySpan<string> values);

    /// <summary>
    /// Sets text field <paramref name="name"/>'s value as rich text: <paramref name="runs"/>, each with its own style (bold,
    /// italic, size, colour, font family, underline, strike-out; what a run does not set is the field's own). The field
    /// keeps the runs and their plain text, and its appearance is made to show them styled
    /// (<see cref="DocumentWriteOptions.Forms"/>). No runs clears it.
    /// </summary>
    /// <exception cref="ArgumentException">No such field, not a text field, or a value it does not take.</exception>
    /// <exception cref="NotSupportedException">The editor does not write rich text (the message says so).</exception>
    void SetRichText(string name, IReadOnlyList<DocumentFieldRun> runs) =>
        throw new NotSupportedException($"This document editor ({GetType().Name}) does not write rich text.");

    /// <summary>Adds a form field (its widgets name their pages; each gets an appearance showing its value).</summary>
    void AddField(DocumentField field);

    /// <summary>
    /// Flattens form fields: each one's appearance (as shown now, made when it has none) drawn into its page's content, and
    /// the field and its widgets removed; <paramref name="names"/> names the fields (a parent's name takes its children),
    /// none flattens every field. What the form showed stays on its pages as plain content.
    /// </summary>
    /// <exception cref="ArgumentException">No such field.</exception>
    /// <exception cref="NotSupportedException">The editor does not flatten forms (the message says so).</exception>
    void Flatten(params ReadOnlySpan<string> names) =>
        throw new NotSupportedException($"This document editor ({GetType().Name}) does not flatten form fields.");

    /// <summary>Adds an annotation (to its page).</summary>
    void AddAnnotation(DocumentAnnotation annotation);

    /// <summary>Removes an annotation the document holds (by its page and <see cref="DocumentAnnotation.Id"/>).</summary>
    void RemoveAnnotation(DocumentAnnotation annotation);

    /// <summary>Adds a bookmark at the top level.</summary>
    void AddOutline(DocumentOutline outline);

    /// <summary>Inserts <paramref name="pages"/> of <paramref name="from"/> (with what they use) before page <paramref name="index"/>.</summary>
    void InsertPages(int index, IDocument from, params ReadOnlySpan<int> pages);

    /// <summary>Removes <paramref name="pages"/>.</summary>
    void RemovePages(params ReadOnlySpan<int> pages);

    /// <summary>Moves page <paramref name="from"/> to <paramref name="to"/>.</summary>
    void MovePage(int from, int to);

    /// <summary>Turns page <paramref name="index"/> to <paramref name="rotation"/> degrees clockwise (0, 90, 180 or 270).</summary>
    void RotatePage(int index, int rotation);

    /// <summary>Adds a page at the end.</summary>
    IDocumentPageBuilder AddPage(float width, float height);

    /// <summary>New content over page <paramref name="index"/>'s (a stamp, a watermark, page numbers).</summary>
    IDocumentPageBuilder Overlay(int index);

    /// <summary>Encrypts the saved document, or, with null, saves it without encryption (the owner password having opened it).</summary>
    void Encrypt(DocumentEncryption? encryption);

    /// <summary>Signs the document in the signature field <paramref name="field"/> when it is saved (always as an incremental update).</summary>
    void Sign(string field, SigningOptions options);

    /// <summary>Writes the changed document to the destination.</summary>
    void Save(DocumentSaveMode mode);

    /// <summary>
    /// Removes for real what lies under <paramref name="areas"/> on page <paramref name="index"/> (the document as it
    /// stands): the glyphs whose boxes overlap an area leave the text, a path inside one leaves the page and one crossing it
    /// keeps only its parts outside, an image's pixels whose centres fall inside one take the area's fill (or black), the
    /// annotations over one are removed; then each area is filled. A page redacted again takes the new areas with its
    /// earlier ones. Saved as a rewrite only (<see cref="DocumentSaveMode.Rewrite"/>), so nothing removed survives in an
    /// earlier revision.
    /// </summary>
    /// <returns>What was removed from the page (all its areas so far).</returns>
    /// <exception cref="NotSupportedException">The editor does not redact (the message says why).</exception>
    DocumentRedactionResult Redact(int index, IReadOnlyList<DocumentRedaction> areas) =>
        throw new NotSupportedException($"This document editor ({GetType().Name}) does not redact pages.");
}

/// <summary>
/// Writes one document format; register it with <see cref="DocumentWriters.Register"/>. The reading side of a format is its
/// <see cref="IDocumentFormat"/> in <see cref="DocumentFormats"/>: a format may read without writing or write without reading.
/// </summary>
public interface IDocumentWriter
{
    /// <summary>The format's name (names ignore case).</summary>
    string Name { get; }

    /// <summary>The file extensions it writes, with the dot; <see cref="DocumentWriters.For"/> picks a writer by them.</summary>
    IReadOnlyCollection<string> Extensions { get; }

    /// <summary>The media types it writes.</summary>
    IReadOnlyCollection<string> MediaTypes { get; }

    /// <summary>
    /// A new document written to <paramref name="destination"/> as it is built (the stream stays the caller's; it may not
    /// seek, so a writer that must go back keeps what it must change).
    /// </summary>
    IDocumentBuilder Create(Stream destination, DocumentWriteOptions options);

    /// <summary><paramref name="source"/> changed and written to <paramref name="destination"/> (the source and the stream stay the caller's).</summary>
    /// <exception cref="NotSupportedException">The writer cannot edit documents (yet), or not that one's format; the message says which.</exception>
    IDocumentEditor Edit(IDocument source, Stream destination, DocumentWriteOptions options);
}

/// <summary>
/// The document writers, by name (ignoring case). Built in (Idrak.Data): "pdf" (pages of images, vector paths and text
/// with their fonts embedded). A package registers its own with <see cref="Register"/>, and one registered under a library
/// name takes its place, which stays behind it (see <see cref="SetPolicy"/>).
/// </summary>
public static class DocumentWriters
{
    private static readonly SlotTable<string, IDocumentWriter> Registry =
        new(nameof(DocumentWriters), (slot, app, library) => new GuardedWriter(slot, app, library), StringComparer.OrdinalIgnoreCase);

    static DocumentWriters() => LibraryDefaults.Ensure(typeof(DocumentWriters));

    /// <summary>Registers <paramref name="writer"/> under its name; under a library name it takes that writer's place, which stays behind it (see <see cref="SetPolicy"/>).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IDocumentWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);
        ArgumentException.ThrowIfNullOrWhiteSpace(writer.Name);
        Registry.Register(writer.Name, writer, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's writer <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered writer names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The extensions the registered writers write (with the dot, lower case).</summary>
    public static IReadOnlyCollection<string> Extensions => [.. Registry.Values.SelectMany(w => w.Extensions).Select(e => e.ToLowerInvariant()).Distinct()];

    /// <summary>The writer registered as <paramref name="name"/> (any case).</summary>
    /// <exception cref="NotRegisteredException">None is: the message names the registered writers and what to do.</exception>
    public static IDocumentWriter Get(string name) => Registry.Find(name)
        ?? throw new NotRegisteredException(nameof(DocumentWriters), name, Registry.Keys,
            $"No document writer '{name}' is registered (registered: {Listed()}); add it with DocumentWriters.Register, or reference a package that registers it.");

    /// <summary>The writer that writes files named like <paramref name="path"/> (by its extension).</summary>
    /// <exception cref="NotSupportedException">No registered writer lists the extension; the message names the extensions there are.</exception>
    public static IDocumentWriter For(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        string extension = Path.GetExtension(path);
        return Registry.Values.FirstOrDefault(w => extension.Length > 0 && w.Extensions.Contains(extension, StringComparer.OrdinalIgnoreCase))
            ?? throw new NotSupportedException($"No document writer writes '{extension}' files (registered: {Listed()}; extensions: {string.Join(", ", Extensions)}); add one with DocumentWriters.Register.");
    }

    /// <summary>The library's writer <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IDocumentWriter? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the writer <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>
    /// What happens when the app's writer <paramref name="name"/> fails to start a document (<see cref="SlotPolicy.Throw"/>
    /// unless set; <see cref="SlotPolicy.FallBack"/>: the library's writes it; a builder writes as it goes, so a failure past
    /// the start reaches the caller), or whether it only runs beside the library's (<see cref="SlotPolicy.Shadow"/>: the
    /// library's bytes are written, the app's built from the same calls and compared byte for byte; an edit, written once,
    /// is the library's).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(name, policy, shadowRate);

    /// <summary>A new document written to the file <paramref name="path"/> (its folder made) by the writer of its extension; the file is closed with the builder.</summary>
    /// <exception cref="NotSupportedException">No registered writer writes the extension.</exception>
    public static IDocumentBuilder Create(string path, DocumentWriteOptions? options = null)
    {
        var writer = For(path);
        string? folder = Path.GetDirectoryName(Path.GetFullPath(path));
        if (folder is not null)
        {
            Directory.CreateDirectory(folder);
        }

        var stream = new FileStream(path, FileMode.Create, FileAccess.ReadWrite, FileShare.None);
        try
        {
            return new OwningBuilder(writer.Create(stream, options ?? new DocumentWriteOptions()), stream);
        }
        catch
        {
            stream.Dispose();
            throw;
        }
    }

    /// <summary>A new document written to <paramref name="destination"/> by the writer <paramref name="name"/> (the stream stays the caller's).</summary>
    /// <exception cref="NotRegisteredException">No writer of that name is registered.</exception>
    public static IDocumentBuilder Create(Stream destination, string name, DocumentWriteOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(destination);
        return Get(name).Create(destination, options ?? new DocumentWriteOptions());
    }

    private static string Listed() => Registry.Keys.Count == 0 ? "none" : string.Join(", ", Registry.Keys);

    // A builder the registry made over a file it opened: the file is closed after the builder.
    private sealed class OwningBuilder(IDocumentBuilder inner, Stream stream) : IDocumentBuilder
    {
        public DocumentInfo Info
        {
            get => inner.Info;
            set => inner.Info = value;
        }

        public int PageCount => inner.PageCount;

        public IDocumentPageBuilder AddPage(float width, float height) => inner.AddPage(width, height);

        public IDocumentPageBuilder CopyPage(IDocument from, int index) => inner.CopyPage(from, index);

        public void Attach(string name, string mediaType, Stream content) => inner.Attach(name, mediaType, content);

        public void AddField(DocumentField field) => inner.AddField(field);

        public void AddOutline(DocumentOutline outline) => inner.AddOutline(outline);

        public void SetPageLabels(IReadOnlyList<DocumentPageLabel> labels) => inner.SetPageLabels(labels);

        public void Encrypt(DocumentEncryption encryption) => inner.Encrypt(encryption);

        public void Sign(string field, SigningOptions options) => inner.Sign(field, options);

        public void Finish() => inner.Finish();

        public void Dispose()
        {
            try
            {
                inner.Dispose();
            }
            finally
            {
                stream.Dispose();
            }
        }
    }

    // An app's writer over a library one: a document that fails to start falls back; under Shadow the library's writes the
    // destination and the app's, on a sample of documents, writes the same calls to a counter of bytes, the two compared.
    private sealed class GuardedWriter(Slot slot, IDocumentWriter app, IDocumentWriter library) : IDocumentWriter
    {
        public string Name => app.Name;

        public IReadOnlyCollection<string> Extensions => app.Extensions;

        public IReadOnlyCollection<string> MediaTypes => app.MediaTypes;

        public IDocumentBuilder Create(Stream destination, DocumentWriteOptions options)
        {
            if (slot.Policy == SlotPolicy.Shadow)
            {
                if (!slot.Samples())
                {
                    return library.Create(destination, options);
                }

                var written = new HashingStream(destination);
                var answer = library.Create(written, options);
                var sink = new HashingStream(null);
                IDocumentBuilder? other = null;
                Exception? error = null;
                try
                {
                    other = app.Create(sink, options);
                }
                catch (Exception e) when (e is not OperationCanceledException)
                {
                    error = e;
                }

                return new ShadowBuilder(slot, answer, written, other, sink, error);
            }

            long start = destination.CanSeek ? destination.Position : -1;
            try
            {
                return app.Create(destination, options);
            }
            catch (Exception e) when (slot.Failed(e))
            {
                Rewind(destination, start);
                return library.Create(destination, options);
            }
        }

        public IDocumentEditor Edit(IDocument source, Stream destination, DocumentWriteOptions options)
        {
            if (slot.Policy == SlotPolicy.Shadow)
            {
                return library.Edit(source, destination, options);       // an edit is written once: the library's
            }

            long start = destination.CanSeek ? destination.Position : -1;
            try
            {
                return app.Edit(source, destination, options);
            }
            catch (Exception e) when (slot.Failed(e))
            {
                Rewind(destination, start);
                return library.Edit(source, destination, options);
            }
        }

        // What the app's writer wrote before it failed is dropped, where the stream allows.
        private static void Rewind(Stream destination, long start)
        {
            if (start >= 0 && destination.Position != start)
            {
                destination.Position = start;
                try
                {
                    destination.SetLength(start);
                }
                catch (NotSupportedException)
                {
                    // A stream that cannot be cut keeps the bytes past the start; the library's writer writes over them.
                }
            }
        }
    }

    // The library's builder answers and writes the destination; the app's gets the same calls (its first failure stops it)
    // and writes to a counter; at Finish the two are compared by length and SHA-256, with each side's time.
    private sealed class ShadowBuilder(Slot slot, IDocumentBuilder library, HashingStream written, IDocumentBuilder? app, HashingStream sink, Exception? error) : IDocumentBuilder
    {
        private long _libraryTicks, _appTicks;
        private Exception? _error = error;
        private bool _reported;

        public DocumentInfo Info
        {
            get => library.Info;
            set
            {
                Library(() => library.Info = value);
                App(() => app!.Info = value);
            }
        }

        public int PageCount => library.PageCount;

        public IDocumentPageBuilder AddPage(float width, float height)
        {
            var page = Library(() => library.AddPage(width, height));
            return new ShadowPageBuilder(this, page, App(() => app!.AddPage(width, height)));
        }

        public IDocumentPageBuilder CopyPage(IDocument from, int index)
        {
            var page = Library(() => library.CopyPage(from, index));
            return new ShadowPageBuilder(this, page, App(() => app!.CopyPage(from, index)));
        }

        public void Attach(string name, string mediaType, Stream content)
        {
            if (Active)
            {
                using var copy = new MemoryStream();
                content.CopyTo(copy);
                copy.Position = 0;
                Library(() => library.Attach(name, mediaType, copy));
                App(() => app!.Attach(name, mediaType, new MemoryStream(copy.GetBuffer(), 0, (int)copy.Length, writable: false)));
                return;
            }

            Library(() => library.Attach(name, mediaType, content));
        }

        public void AddField(DocumentField field)
        {
            Library(() => library.AddField(field));
            App(() => app!.AddField(field));
        }

        public void AddOutline(DocumentOutline outline)
        {
            Library(() => library.AddOutline(outline));
            App(() => app!.AddOutline(outline));
        }

        public void SetPageLabels(IReadOnlyList<DocumentPageLabel> labels)
        {
            Library(() => library.SetPageLabels(labels));
            App(() => app!.SetPageLabels(labels));
        }

        public void Encrypt(DocumentEncryption encryption)
        {
            Library(() => library.Encrypt(encryption));
            App(() => app!.Encrypt(encryption));
        }

        public void Sign(string field, SigningOptions options)
        {
            Library(() => library.Sign(field, options));
            App(() => app!.Sign(field, options));
        }

        public void Finish()
        {
            Library(library.Finish);
            App(() => app!.Finish());
            if (!_reported)
            {
                _reported = true;
                string? difference = _error is not null ? null
                    : written.Written == sink.Written && written.Hash().AsSpan().SequenceEqual(sink.Hash()) ? null
                    : $"the library wrote {written.Written} bytes, the override {sink.Written}: they differ";
                slot.ReportComparison(difference, Stopwatch.GetElapsedTime(0, _libraryTicks), Stopwatch.GetElapsedTime(0, _appTicks), error: _error);
            }
        }

        public void Dispose()
        {
            try
            {
                library.Dispose();
            }
            finally
            {
                try
                {
                    app?.Dispose();
                }
                catch (Exception e) when (e is not OperationCanceledException)
                {
                    _error ??= e;
                }

                written.Dispose();
                sink.Dispose();
            }
        }

        // Whether the app's builder still gets calls.
        internal bool Active => app is not null && _error is null;

        internal void Library(Action call)
        {
            long start = Stopwatch.GetTimestamp();
            try
            {
                call();
            }
            finally
            {
                _libraryTicks += Stopwatch.GetTimestamp() - start;
            }
        }

        internal T Library<T>(Func<T> call)
        {
            long start = Stopwatch.GetTimestamp();
            try
            {
                return call();
            }
            finally
            {
                _libraryTicks += Stopwatch.GetTimestamp() - start;
            }
        }

        internal void App(Action call) => RunApp(call);

        internal T? App<T>(Func<T> call) where T : class
        {
            T? result = null;
            RunApp(() => result = call());                                    // not App: this lambda would bind to App<T> again
            return result;
        }

        // The app's call, timed; its first failure stops the app's builder getting calls.
        private void RunApp(Action call)
        {
            if (!Active)
            {
                return;
            }

            long start = Stopwatch.GetTimestamp();
            try
            {
                call();
            }
            catch (Exception e) when (e is not OperationCanceledException)
            {
                _error = e;
            }
            finally
            {
                _appTicks += Stopwatch.GetTimestamp() - start;
            }
        }
    }

    // A page of a shadowed document: each call to the library's page, then to the app's while it still gets calls.
    private sealed class ShadowPageBuilder(ShadowBuilder owner, IDocumentPageBuilder library, IDocumentPageBuilder? app) : IDocumentPageBuilder
    {
        public int Index => library.Index;

        public float Width => library.Width;

        public float Height => library.Height;

        public void Image(ImageData image, DocumentBox box, DocumentImageEncoding encoding)
        {
            owner.Library(() => library.Image(image, box, encoding));
            Other(page => page.Image(image, box, encoding));
        }

        public void Path(DocumentPath path)
        {
            owner.Library(() => library.Path(path));
            Other(page => page.Path(path));
        }

        public void Glyphs(IFontFace face, float size, ReadOnlySpan<DocumentGlyph> glyphs, string text, DocumentTextMode mode, DocumentColour colour)
        {
            DocumentGlyph[] copy = glyphs.ToArray();                          // a span cannot be held by the calls below
            owner.Library(() => library.Glyphs(face, size, copy, text, mode, colour));
            Other(page => page.Glyphs(face, size, copy, text, mode, colour));
        }

        public void Glyphs(IFontFace face, ReadOnlySpan<DocumentGlyph> glyphs, ReadOnlySpan<DocumentMatrix> matrices, string text, DocumentTextMode mode, DocumentColour colour)
        {
            DocumentGlyph[] copy = glyphs.ToArray();                          // spans cannot be held by the calls below
            DocumentMatrix[] maps = matrices.ToArray();
            owner.Library(() => library.Glyphs(face, copy, maps, text, mode, colour));
            Other(page => page.Glyphs(face, copy, maps, text, mode, colour));
        }

        public void Save()
        {
            owner.Library(library.Save);
            Other(page => page.Save());
        }

        public void Restore()
        {
            owner.Library(library.Restore);
            Other(page => page.Restore());
        }

        public void Clip(DocumentPath path)
        {
            owner.Library(() => library.Clip(path));
            Other(page => page.Clip(path));
        }

        public void AddAnnotation(DocumentAnnotation annotation)
        {
            owner.Library(() => library.AddAnnotation(annotation));
            Other(page => page.AddAnnotation(annotation));
        }

        public void AddLink(DocumentLink link)
        {
            owner.Library(() => library.AddLink(link));
            Other(page => page.AddLink(link));
        }

        private void Other(Action<IDocumentPageBuilder> call)
        {
            if (app is not null)
            {
                owner.App(() => call(app));
            }
        }
    }
}

// A write-only stream that hashes what passes (SHA-256) and counts it, on its way to another stream or to nothing: the
// guards compare what two writers wrote without keeping it.
internal sealed class HashingStream(Stream? inner) : Stream
{
    private readonly IncrementalHash _hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);

    /// <summary>The bytes written so far.</summary>
    public long Written { get; private set; }

    public override bool CanRead => false;

    public override bool CanSeek => false;

    public override bool CanWrite => true;

    public override long Length => Written;

    public override long Position
    {
        get => Written;
        set => throw new NotSupportedException("A hashing stream is written in order.");
    }

    /// <summary>The SHA-256 of what was written so far.</summary>
    public byte[] Hash() => _hash.GetCurrentHash();

    public override void Write(byte[] buffer, int offset, int count) => Write(buffer.AsSpan(offset, count));

    public override void Write(ReadOnlySpan<byte> buffer)
    {
        _hash.AppendData(buffer);
        inner?.Write(buffer);
        Written += buffer.Length;
    }

    public override void Flush() => inner?.Flush();

    public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException("A hashing stream is written, not read.");

    public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException("A hashing stream is written in order.");

    public override void SetLength(long value) => throw new NotSupportedException("A hashing stream is written in order.");

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _hash.Dispose();
        }

        base.Dispose(disposing);
    }
}
