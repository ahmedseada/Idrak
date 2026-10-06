// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Data;

/// <summary>
/// The sample sources this assembly ships, registered in <see cref="SampleSources"/> (Idrak.Abstraction) by
/// <see cref="LibraryRegistrations"/>: "csv", "images", "tokens" and "npy".
/// </summary>
internal static class LibrarySampleSources
{
    public static void RegisterAll()
    {
        SampleSources.Register("csv", OpenCsv);
        SampleSources.Register("images", OpenImages);
        SampleSources.Register("tokens", OpenTokens);
        SampleSources.Register("npy", OpenNpy);
    }

    private static CsvSource OpenCsv(string path, IReadOnlyDictionary<string, string> options)
    {
        Known(options, "csv", "target", "ignore", "delimiter", "header");
        string target = options.GetValueOrDefault("target") ?? throw new ArgumentException("The csv source needs the option 'target' (the target columns, comma-separated).");
        return CsvSource.Open(path, new CsvOptions
        {
            TargetColumns = List(target),
            IgnoreColumns = options.TryGetValue("ignore", out var ignore) ? List(ignore) : [],
            Delimiter = options.TryGetValue("delimiter", out var d) ? (d == "\\t" ? '\t' : d.Length == 1 ? d[0] : throw new ArgumentException($"delimiter '{d}': one character.")) : ',',
            HasHeader = !options.TryGetValue("header", out var header) || bool.Parse(header),
        });
    }

    private static ImageFolderSource OpenImages(string path, IReadOnlyDictionary<string, string> options)
    {
        Known(options, "images", "channels", "height", "width");
        int? Number(string name) => options.TryGetValue(name, out var v) ? int.Parse(v, CultureInfo.InvariantCulture) : null;
        return ImageFolderSource.Open(path, Number("channels"), Number("height"), Number("width"));
    }

    private static TokenFileSource OpenTokens(string path, IReadOnlyDictionary<string, string> options)
    {
        Known(options, "tokens", "length", "stride", "type");
        int length = int.Parse(options.GetValueOrDefault("length") ?? throw new ArgumentException("The tokens source needs the option 'length' (tokens per window)."), CultureInfo.InvariantCulture);
        int? stride = options.TryGetValue("stride", out var s) ? int.Parse(s, CultureInfo.InvariantCulture) : null;
        var type = options.TryGetValue("type", out var t) ? Enum.Parse<TokenType>(t, ignoreCase: true) : TokenType.UInt16;
        return new TokenFileSource(path, length, stride, type);
    }

    private static NpySource OpenNpy(string path, IReadOnlyDictionary<string, string> options)
    {
        Known(options, "npy", "targets", "classes");
        int? classes = options.TryGetValue("classes", out var c) ? int.Parse(c, CultureInfo.InvariantCulture) : null;
        return new NpySource(path, options.GetValueOrDefault("targets"), classes);
    }

    private static void Known(IReadOnlyDictionary<string, string> options, string source, params string[] names)
    {
        foreach (string key in options.Keys)
        {
            if (!names.Contains(key, StringComparer.OrdinalIgnoreCase))
            {
                throw new ArgumentException($"The {source} source has no option '{key}' ({string.Join(", ", names)}).");
            }
        }
    }

    private static string[] List(string text) => text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
}
