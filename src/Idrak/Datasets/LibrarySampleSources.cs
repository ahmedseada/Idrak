// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Datasets;

/// <summary>
/// The sample sources this assembly ships, registered in <see cref="SampleSources"/> (Idrak.Abstraction) by
/// <see cref="LibraryRegistrations"/>: "csv", "tokens" and "npy" (Idrak.Imaging adds "images").
/// </summary>
internal static class LibrarySampleSources
{
    public static void RegisterAll()
    {
        SampleSources.Register(new SourceKind("csv", "CsvSource: rows of a CSV file (target required, comma-separated)", new OptionKeys("target", "ignore", "delimiter", "header"), OpenCsv));
        SampleSources.Register(new SourceKind("tokens", "TokenFileSource: windows of a token file (length required)", new OptionKeys("length", "stride", "type"), OpenTokens));
        SampleSources.Register(new SourceKind("npy", "NpySource: the rows of a .npy file, with targets from a second one", new OptionKeys("targets", "classes"), OpenNpy));
    }

    // One of the sources: its options' keys (any other is refused, naming them) and how it opens a path.
    private sealed class SourceKind(string name, string summary, OptionKeys keys, Func<string, IReadOnlyDictionary<string, string>, ISampleSource> open) : ISampleSourceKind
    {
        public string Name => name;

        public string Summary => summary;

        public IReadOnlySet<string> Keys => keys;

        public ISampleSource Create(SampleSourceRequest request)
        {
            ArgumentNullException.ThrowIfNull(request);
            Known(request.Options, name, keys);
            return open(request.Path, request.Options);
        }
    }

    private static CsvSource OpenCsv(string path, IReadOnlyDictionary<string, string> options)
    {
        string target = options.GetValueOrDefault("target") ?? throw new ArgumentException("The csv source needs the option 'target' (the target columns, comma-separated).");
        return CsvSource.Open(path, new CsvOptions
        {
            TargetColumns = List(target),
            IgnoreColumns = options.TryGetValue("ignore", out var ignore) ? List(ignore) : [],
            Delimiter = options.TryGetValue("delimiter", out var d) ? (d == "\\t" ? '\t' : d.Length == 1 ? d[0] : throw new ArgumentException($"delimiter '{d}': one character.")) : ',',
            HasHeader = !options.TryGetValue("header", out var header) || bool.Parse(header),
        });
    }

    private static TokenFileSource OpenTokens(string path, IReadOnlyDictionary<string, string> options)
    {
        int length = int.Parse(options.GetValueOrDefault("length") ?? throw new ArgumentException("The tokens source needs the option 'length' (tokens per window)."), CultureInfo.InvariantCulture);
        int? stride = options.TryGetValue("stride", out var s) ? int.Parse(s, CultureInfo.InvariantCulture) : null;
        var type = options.TryGetValue("type", out var t) ? Enum.Parse<TokenType>(t, ignoreCase: true) : TokenType.UInt16;
        return new TokenFileSource(path, length, stride, type);
    }

    private static NpySource OpenNpy(string path, IReadOnlyDictionary<string, string> options)
    {
        int? classes = options.TryGetValue("classes", out var c) ? int.Parse(c, CultureInfo.InvariantCulture) : null;
        return new NpySource(path, options.GetValueOrDefault("targets"), classes);
    }

    private static void Known(IReadOnlyDictionary<string, string> options, string source, OptionKeys names)
    {
        foreach (string key in options.Keys)
        {
            if (!Has(names, key))
            {
                throw new ArgumentException($"The {source} source has no option '{key}' ({names}).");
            }
        }

        // Option names ignore case.
        static bool Has(OptionKeys names, string key)
        {
            if (names.Contains(key))
            {
                return true;
            }

            foreach (string name in names.Ordered)
            {
                if (string.Equals(name, key, StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }

            return false;
        }
    }

    private static string[] List(string text) => text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
}
