// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Datasets;

/// <summary>
/// A kind of data file <see cref="Dataset"/> reads: its name, the extensions it is chosen by and how its rows are read.
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
/// The data file formats <see cref="DataFiles"/> knows, by name and by extension. JSON Lines, JSON, CSV, TSV, Parquet,
/// text and source code are registered (named as <see cref="DataFormat"/>); add others, or replace these, with
/// <see cref="Register"/>.
/// </summary>
public static class DataFileFormats
{
    // In registration order; a later format takes the extensions it shares with earlier ones.
    private static readonly List<IDataFileFormat> Registry =
    [
        new BuiltIn(nameof(DataFormat.JsonLines), [".jsonl", ".ndjson"], (open, path, _) => DataFiles.JsonLines(open, path)),
        new BuiltIn(nameof(DataFormat.Json), [".json"], DataFiles.Json),
        new BuiltIn(nameof(DataFormat.Csv), [".csv"], (open, _, options) => DataFiles.Delimited(open, ',', options)),
        new BuiltIn(nameof(DataFormat.Tsv), [".tsv"], (open, _, options) => DataFiles.Delimited(open, '\t', options)),
        new BuiltIn(nameof(DataFormat.Parquet), [".parquet"], (open, _, _) => DataFiles.ParquetRows(open)),
        new BuiltIn(nameof(DataFormat.Text), [".txt", ".text", ".md", ".markdown", ".rst"], DataFiles.Text),
        new BuiltIn(nameof(DataFormat.Code), [.. DataFiles.CodeLanguages.Keys], DataFiles.Code),
    ];

    private static Dictionary<string, IDataFileFormat> byExtension = ByExtension();

    /// <summary>Registers <paramref name="format"/>, replacing a format of the same name (names ignore case).</summary>
    public static void Register(IDataFileFormat format)
    {
        ArgumentNullException.ThrowIfNull(format);
        lock (Registry)
        {
            int index = Registry.FindIndex(f => string.Equals(f.Name, format.Name, StringComparison.OrdinalIgnoreCase));
            if (index >= 0)
            {
                Registry[index] = format;
            }
            else
            {
                Registry.Add(format);
            }

            byExtension = ByExtension();
        }
    }

    /// <summary>Removes the format registered as <paramref name="name"/> (ignoring case); returns whether there was one.</summary>
    public static bool Unregister(string name)
    {
        lock (Registry)
        {
            bool removed = Registry.RemoveAll(f => string.Equals(f.Name, name, StringComparison.OrdinalIgnoreCase)) > 0;
            byExtension = ByExtension();
            return removed;
        }
    }

    /// <summary>The registered format names, in registration order.</summary>
    public static IReadOnlyCollection<string> Names
    {
        get
        {
            lock (Registry)
            {
                return [.. Registry.Select(f => f.Name)];
            }
        }
    }

    /// <summary>The format registered as <paramref name="name"/> (ignoring case).</summary>
    public static IDataFileFormat Get(string name)
    {
        lock (Registry)
        {
            return Registry.Find(f => string.Equals(f.Name, name, StringComparison.OrdinalIgnoreCase))
                ?? throw new NotSupportedException($"No data file format '{name}' is registered ({string.Join(", ", Registry.Select(f => f.Name))}); add it with DataFileFormats.Register.");
        }
    }

    /// <summary>
    /// The format of <paramref name="path"/> by its extension (.gz stripped), or null for an unknown one. Source code
    /// (the "Code" format) is only chosen when <paramref name="includeCode"/>.
    /// </summary>
    public static IDataFileFormat? Find(string path, bool includeCode = true)
    {
        string name = path.EndsWith(".gz", StringComparison.OrdinalIgnoreCase) ? path[..^3] : path;
        string extension = Path.GetExtension(name);
        lock (Registry)
        {
            return byExtension.TryGetValue(extension, out var format) && (includeCode || !Is(format, DataFormat.Code)) ? format : null;
        }
    }

    // Whether the format has the name of a built-in (it may be a replacement registered under that name).
    internal static bool Is(IDataFileFormat format, DataFormat builtIn) =>
        string.Equals(format.Name, builtIn.ToString(), StringComparison.OrdinalIgnoreCase);

    private static Dictionary<string, IDataFileFormat> ByExtension()
    {
        var map = new Dictionary<string, IDataFileFormat>(StringComparer.OrdinalIgnoreCase);
        foreach (var format in Registry)
        {
            foreach (var extension in format.Extensions)
            {
                map[extension] = format;
            }
        }

        return map;
    }

    private sealed class BuiltIn(string name, string[] extensions, Func<Func<Stream>, string, ReadOptions, IEnumerable<JsonObject>> read) : IDataFileFormat
    {
        public string Name => name;

        public IReadOnlyCollection<string> Extensions => extensions;

        public IEnumerable<JsonObject> Read(Func<Stream> open, string path, ReadOptions options) => read(open, path, options);
    }
}
