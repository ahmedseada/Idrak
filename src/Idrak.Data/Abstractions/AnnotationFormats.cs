// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Data.Abstractions;

/// <summary>How an annotation format reads a dataset (<see cref="IAnnotationFormat.Read"/>); each format reads what it needs.</summary>
public sealed record AnnotationReadOptions
{
    /// <summary>The class names, in index order, when the files do not name them (YOLO) or to fix their order; null to take them from the files.</summary>
    public IReadOnlyList<string>? Classes { get; init; }

    /// <summary>The folder the images are in, when not the format's usual place.</summary>
    public string? ImagesRoot { get; init; }
}

/// <summary>
/// A detection or segmentation dataset format (COCO, YOLO text, Pascal VOC, ...): reads its annotation files into an
/// <see cref="AnnotatedDataset"/> (images named, not decoded) and writes one back, so a dataset written and read again is
/// the same. Register new ones with <see cref="AnnotationFormats.Register"/>.
/// </summary>
public interface IAnnotationFormat
{
    /// <summary>The format's name (the library's: "coco", "yolo", "voc").</summary>
    string Name { get; }

    /// <summary>What the format reads (a file or a folder, and its layout), for listings and error messages.</summary>
    string Summary { get; }

    /// <summary>Whether <paramref name="path"/> looks like a dataset of this format (by its name and first bytes; nothing is read in full).</summary>
    bool CanRead(string path);

    /// <summary>The dataset at <paramref name="path"/>; a malformed file is an <see cref="InvalidDataException"/> naming it.</summary>
    AnnotatedDataset Read(string path, AnnotationReadOptions? options = null);

    /// <summary>Writes <paramref name="dataset"/>'s annotations at <paramref name="path"/> (the images are not copied).</summary>
    void Write(AnnotatedDataset dataset, string path);
}

/// <summary>
/// The detection and segmentation dataset formats, by name (ignoring case). Idrak.Data registers "coco" (an instances
/// JSON file: boxes, categories, polygon and run-length masks, crowds), "yolo" (a folder of images/ and labels/, one
/// text file per image of normalized centre and size, or polygons; class names from classes.txt or data.yaml) and "voc"
/// (Pascal VOC: a folder of Annotations/*.xml, read with System.Xml). Add others, or replace these, with <see cref="Register"/>.
/// </summary>
public static class AnnotationFormats
{
    private static readonly SlotTable<string, IAnnotationFormat> Registry =
        new(nameof(AnnotationFormats), (slot, app, library) => new Guarded(slot, app, library), StringComparer.OrdinalIgnoreCase);

    static AnnotationFormats() => Overrides.AsLibraryDefaults(AnnotationFiles.RegisterAll);   // the built-in formats, on first use

    /// <summary>Registers <paramref name="format"/> (names ignore case); under a library name it shadows the library's until <see cref="Unregister"/>.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IAnnotationFormat format)
    {
        ArgumentNullException.ThrowIfNull(format);
        ArgumentException.ThrowIfNullOrWhiteSpace(format.Name);
        Registry.Register(format.Name, format, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's format <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered format names, in registration order.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The format registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names those that are.</exception>
    public static IAnnotationFormat Get(string name) => Registry.Find(name)
        ?? throw new NotSupportedException($"No annotation format '{name}' is registered ({string.Join(", ", Names)}); add it with AnnotationFormats.Register.");

    /// <summary>The first registered format that reads <paramref name="path"/>, or null.</summary>
    public static IAnnotationFormat? Find(string path) => Registry.Values.FirstOrDefault(f => f.CanRead(path));

    /// <summary>The library's format <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IAnnotationFormat? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the format <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>
    /// What happens when the app's format <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/> retries on the library's), or whether its reading only runs beside the library's
    /// (<see cref="SlotPolicy.Shadow"/>: the library's answers, the images and objects are compared; writing is never done twice).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(name, policy, shadowRate);

    /// <summary>The dataset at <paramref name="path"/> in the format <paramref name="format"/> (null: the first that reads it).</summary>
    /// <exception cref="NotSupportedException">No registered format reads the path.</exception>
    public static AnnotatedDataset Read(string path, string? format = null, AnnotationReadOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        var reader = format is null ? Find(path) ?? throw new NotSupportedException($"No registered annotation format reads {path} ({string.Join(", ", Names)}); name the format.")
            : Get(format);
        return reader.Read(path, options);
    }

    /// <summary>Writes <paramref name="dataset"/> at <paramref name="path"/> in the format <paramref name="format"/>.</summary>
    public static void Write(AnnotatedDataset dataset, string path, string format) => Get(format).Write(dataset, path);

    private sealed class Guarded(Slot slot, IAnnotationFormat app, IAnnotationFormat library) : IAnnotationFormat
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public bool CanRead(string path) => slot.Call(() => app.CanRead(path), () => library.CanRead(path), Comparisons.Exact);

        public AnnotatedDataset Read(string path, AnnotationReadOptions? options = null) => slot.Call(() => app.Read(path, options), () => library.Read(path, options), (a, b) =>
            Comparisons.Exact((a.Classes.Count, a.Images.Count), (b.Classes.Count, b.Images.Count))
            ?? Comparisons.Difference([.. a.Images.Select(i => i.Objects.Count)], [.. b.Images.Select(i => i.Objects.Count)]));

        public void Write(AnnotatedDataset dataset, string path) => slot.Call(() =>
        {
            app.Write(dataset, path);
            return 0;
        }, () =>
        {
            library.Write(dataset, path);
            return 0;
        }, effects: true);
    }
}
