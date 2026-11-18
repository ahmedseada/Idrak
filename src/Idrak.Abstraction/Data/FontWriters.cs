// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Data;

/// <summary>A font's name strings, by what they hold (the font file formats number them this way).</summary>
public enum FontNameKind
{
    /// <summary>The copyright notice.</summary>
    Copyright = 0,

    /// <summary>The family ("DejaVu Sans").</summary>
    Family = 1,

    /// <summary>The style within the family ("Bold").</summary>
    Style = 2,

    /// <summary>A unique identifier of the font.</summary>
    UniqueId = 3,

    /// <summary>The full name ("DejaVu Sans Bold").</summary>
    FullName = 4,

    /// <summary>The version ("Version 2.37").</summary>
    Version = 5,

    /// <summary>The name without spaces programs refer to it by.</summary>
    PostScriptName = 6,

    /// <summary>The trademark notice.</summary>
    Trademark = 7,

    /// <summary>Who made the font.</summary>
    Manufacturer = 8,

    /// <summary>Who designed it.</summary>
    Designer = 9,

    /// <summary>A description.</summary>
    Description = 10,

    /// <summary>The maker's web address.</summary>
    VendorUrl = 11,

    /// <summary>The designer's web address.</summary>
    DesignerUrl = 12,

    /// <summary>The licence.</summary>
    License = 13,

    /// <summary>The licence's web address.</summary>
    LicenseUrl = 14,

    /// <summary>The family a large family groups its styles under.</summary>
    TypographicFamily = 16,

    /// <summary>The style within that family.</summary>
    TypographicStyle = 17,

    /// <summary>A sample of text in the font.</summary>
    SampleText = 19,
}

/// <summary>
/// How a face is written as a font file: whole or a subset, with or without its layout tables, which name strings, and how
/// curves are converted when the format takes another kind. <see cref="CurveToleranceVariable"/> sets the tolerance.
/// </summary>
public sealed record FontWriteOptions
{
    /// <summary>The variable that sets <see cref="CurveTolerance"/> (font units, more than 0): "IDRAK_FONT_CURVE_TOLERANCE".</summary>
    public const string CurveToleranceVariable = "IDRAK_FONT_CURVE_TOLERANCE";

    private readonly float _curveTolerance = DocumentSettings.Float(CurveToleranceVariable, 1e-3f, 1000f) ?? 1f;

    /// <summary>
    /// A subset: these glyph ids, the parts of composite glyphs among them and, with <see cref="KeepLayout"/>, every glyph
    /// the layout tables reach from them; null writes every glyph. The ids stay the same: the glyphs left out are written
    /// empty (no outline, their advance kept), so text already placed by glyph id draws the same.
    /// </summary>
    public IReadOnlyList<int>? Glyphs { get; init; }

    /// <summary>
    /// Whether the layout tables (substitutions, positioning, glyph classes) are written (true): whole, or for a subset with
    /// every glyph they reach; false drops them (what an embedded subset of a document needs).
    /// </summary>
    public bool KeepLayout { get; init; } = true;

    /// <summary>
    /// Name strings written over the source face's: the face's own name records are kept (its copyright and licence
    /// included), these replace theirs or add to them; nothing else is made up.
    /// </summary>
    public IReadOnlyDictionary<FontNameKind, string>? Names { get; init; }

    /// <summary>
    /// How far a curve converted to another kind (cubic to quadratic for a format of quadratic outlines) may stray from the
    /// original, in font units (1 unless <see cref="CurveToleranceVariable"/> sets it).
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not positive and finite.</exception>
    public float CurveTolerance
    {
        get => _curveTolerance;
        init => _curveTolerance = value > 0f && float.IsFinite(value) ? value : throw new ArgumentOutOfRangeException(nameof(CurveTolerance), value, $"A curve tolerance is more than 0 font units ({CurveToleranceVariable}).");
    }

    /// <summary>Whether the hinting instructions are kept (false: dropped; they are kept only when the face is written whole).</summary>
    public bool KeepHinting { get; init; }
}

/// <summary>
/// Writes a face as one font file format; register it with <see cref="FontWriters.Register"/>. The reading side of a format
/// is its <see cref="IFontFormat"/> in <see cref="FontFormats"/>: what a writer writes reads back there with the same
/// outlines, advances, characters and shaping.
/// </summary>
public interface IFontWriter
{
    /// <summary>The format's name (names ignore case).</summary>
    string Name { get; }

    /// <summary>The file extensions it writes, with the dot; <see cref="FontWriters.For"/> picks a writer by them.</summary>
    IReadOnlyCollection<string> Extensions { get; }

    /// <summary>
    /// Writes <paramref name="face"/> to <paramref name="destination"/> as <paramref name="options"/> say (the stream stays
    /// the caller's; it may not seek, so a writer that must go back keeps the file until it is whole).
    /// </summary>
    /// <exception cref="NotSupportedException">The face holds what the format cannot (the message says what).</exception>
    void Write(IFontFace face, Stream destination, FontWriteOptions options);
}

/// <summary>
/// The font file writers, by name (ignoring case), beside <see cref="FontFormats"/>. None is built in yet; a package
/// registers its own with <see cref="Register"/>, and one registered under a library name takes its place, which stays
/// behind it (see <see cref="SetPolicy"/>).
/// </summary>
public static class FontWriters
{
    private static readonly SlotTable<string, IFontWriter> Registry =
        new(nameof(FontWriters), (slot, app, library) => new GuardedWriter(slot, app, library), StringComparer.OrdinalIgnoreCase);

    static FontWriters() => LibraryDefaults.Ensure(typeof(FontWriters));

    /// <summary>Registers <paramref name="writer"/> under its name; under a library name it takes that writer's place, which stays behind it (see <see cref="SetPolicy"/>).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IFontWriter writer)
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
    public static IFontWriter Get(string name) => Registry.Find(name)
        ?? throw new NotRegisteredException(nameof(FontWriters), name, Registry.Keys,
            $"No font writer '{name}' is registered (registered: {Listed()}); add it with FontWriters.Register, or reference a package that registers it.");

    /// <summary>The writer that writes files named like <paramref name="path"/> (by its extension).</summary>
    /// <exception cref="NotSupportedException">No registered writer lists the extension; the message names the extensions there are.</exception>
    public static IFontWriter For(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        string extension = Path.GetExtension(path);
        return Registry.Values.FirstOrDefault(w => extension.Length > 0 && w.Extensions.Contains(extension, StringComparer.OrdinalIgnoreCase))
            ?? throw new NotSupportedException($"No font writer writes '{extension}' files (registered: {Listed()}; extensions: {string.Join(", ", Extensions)}); add one with FontWriters.Register.");
    }

    /// <summary>The library's writer <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IFontWriter? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the writer <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>
    /// What happens when the app's writer <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/>: the app's file is kept until it is whole, and the library's writes it when the app's
    /// fails), or whether it only runs beside the library's (<see cref="SlotPolicy.Shadow"/>: the library's bytes are
    /// written, the app's compared with them byte for byte).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(name, policy, shadowRate);

    /// <summary>Writes <paramref name="face"/> to the file <paramref name="path"/> (its folder made) with the writer of its extension.</summary>
    /// <exception cref="NotSupportedException">No registered writer writes the extension.</exception>
    public static void Save(IFontFace face, string path, FontWriteOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(face);
        var writer = For(path);
        string? folder = Path.GetDirectoryName(Path.GetFullPath(path));
        if (folder is not null)
        {
            Directory.CreateDirectory(folder);
        }

        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        writer.Write(face, stream, options ?? new FontWriteOptions());
    }

    /// <summary>Writes <paramref name="face"/> to <paramref name="destination"/> with the writer <paramref name="name"/> (the stream stays the caller's).</summary>
    /// <exception cref="NotRegisteredException">No writer of that name is registered.</exception>
    public static void Write(IFontFace face, Stream destination, string name, FontWriteOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(face);
        ArgumentNullException.ThrowIfNull(destination);
        Get(name).Write(face, destination, options ?? new FontWriteOptions());
    }

    private static string Listed() => Registry.Keys.Count == 0 ? "none" : string.Join(", ", Registry.Keys);

    // An app's writer over the library's. Under FallBack the app's file is kept in memory until whole (a font is small), so a
    // failure leaves nothing in the destination for the library's to write after; under Shadow the library's writes the
    // destination and the app's a counter, compared by length and SHA-256.
    private sealed class GuardedWriter(Slot slot, IFontWriter app, IFontWriter library) : IFontWriter
    {
        public string Name => app.Name;

        public IReadOnlyCollection<string> Extensions => app.Extensions;

        public void Write(IFontFace face, Stream destination, FontWriteOptions options)
        {
            if (slot.Policy == SlotPolicy.Shadow)
            {
                var run = slot.Shadow();
                using var written = new HashingStream(destination);
                library.Write(face, written, options);
                if (run is not null)
                {
                    run.Answered();
                    try
                    {
                        using var sink = new HashingStream(null);
                        app.Write(face, sink, options);
                        run.Done(written.Written == sink.Written && written.Hash().AsSpan().SequenceEqual(sink.Hash()) ? null
                            : $"the library wrote {written.Written} bytes, the override {sink.Written}: they differ");
                    }
                    catch (Exception e) when (e is not OperationCanceledException)
                    {
                        run.Failed(e);
                    }
                }

                return;
            }

            if (slot.Policy == SlotPolicy.FallBack)
            {
                using var file = new MemoryStream();
                try
                {
                    app.Write(face, file, options);
                }
                catch (Exception e) when (slot.Failed(e))
                {
                    library.Write(face, destination, options);
                    return;
                }

                file.Position = 0;
                file.CopyTo(destination);
                return;
            }

            try
            {
                app.Write(face, destination, options);
            }
            catch (Exception e) when (slot.Failed(e))
            {
                library.Write(face, destination, options);
            }
        }
    }
}
