// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;

namespace Idrak.Data.Abstractions;

/// <summary>
/// A kind of data file a dataset reads (<c>DatasetRows</c> and <c>DataFiles</c> in Idrak.Data): its name, the extensions it is chosen by and how its rows are read.
/// Register new ones with <see cref="DataFileFormats.Register"/>.
/// </summary>
public interface IDataFileFormat
{
    /// <summary>The format's name (the built-ins use the names of <see cref="DataFormat"/>, e.g. "Csv").</summary>
    string Name { get; }

    /// <summary>The extensions files of this format have, with the dot (".csv"); a trailing ".gz" is stripped before matching.</summary>
    IReadOnlyCollection<string> Extensions { get; }

    /// <summary>The rows of a file of this format; <paramref name="open"/> opens it (already decompressed), <paramref name="path"/> is shown in rows and errors.</summary>
    IEnumerable<JsonObject> Read(Func<Stream> open, string path, ReadOptions options);
}

/// <summary>
/// The data file formats datasets read (<c>DataFiles</c> in Idrak.Data), by name and by extension. Idrak.Data
/// registers JSON Lines, JSON, CSV, TSV, Parquet, text and source code (named as <see cref="DataFormat"/>); add others,
/// or replace these, with <see cref="Register"/>.
/// </summary>
public static class DataFileFormats
{
    // In registration order; a later format takes the extensions it shares with earlier ones.
    private static readonly SlotTable<string, IDataFileFormat> Registry = new(nameof(DataFileFormats), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "rows are read lazily, so a failure comes half-way through a file, after rows were handed out");

    // The formats by extension, from the registry's current formats.
    private static (IReadOnlyList<IDataFileFormat> Source, Dictionary<string, IDataFileFormat> Map) byExtension = ([], []);

    static DataFileFormats() => Overrides.AsLibraryDefaults(DataFiles.RegisterAll);   // the built-in formats, on first use

    /// <summary>
    /// Registers <paramref name="format"/> (names ignore case); under a built-in name it overrides the library's format
    /// until <see cref="Unregister"/>. Rows are read lazily, so a format does not fall back to the library's when it fails.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IDataFileFormat format)
    {
        ArgumentNullException.ThrowIfNull(format);
        Registry.Register(format.Name, format, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's format <paramref name="name"/> (a built-in name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered format names, in registration order.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The format registered as <paramref name="name"/> (ignoring case).</summary>
    public static IDataFileFormat Get(string name) => Registry.Find(name)
        ?? throw new NotSupportedException($"No data file format '{name}' is registered ({string.Join(", ", Names)}); add it with DataFileFormats.Register.");

    /// <summary>The library's format <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IDataFileFormat? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the format <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>
    /// The format of <paramref name="path"/> by its extension (.gz stripped), or null for an unknown one. Source code
    /// (the "Code" format) is only chosen when <paramref name="includeCode"/>.
    /// </summary>
    public static IDataFileFormat? Find(string path, bool includeCode = true)
    {
        string name = path.EndsWith(".gz", StringComparison.OrdinalIgnoreCase) ? path[..^3] : path;
        string extension = Path.GetExtension(name);
        return ByExtension().TryGetValue(extension, out var format) && (includeCode || !Is(format, DataFormat.Code)) ? format : null;
    }

    // Whether the format has the name of a built-in (it may be a replacement registered under that name).
    private static bool Is(IDataFileFormat format, DataFormat builtIn) =>
        string.Equals(format.Name, builtIn.ToString(), StringComparison.OrdinalIgnoreCase);

    private static Dictionary<string, IDataFileFormat> ByExtension()
    {
        var formats = Registry.Values;
        var cache = byExtension;
        if (ReferenceEquals(cache.Source, formats))
        {
            return cache.Map;
        }

        var map = new Dictionary<string, IDataFileFormat>(StringComparer.OrdinalIgnoreCase);
        foreach (var format in formats)
        {
            foreach (var extension in format.Extensions)
            {
                map[extension] = format;
            }
        }

        byExtension = (formats, map);
        return map;
    }
}
