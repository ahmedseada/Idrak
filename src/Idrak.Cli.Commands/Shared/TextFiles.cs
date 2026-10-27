// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Cli.Shared;

/// <summary>The text files arguments name (shells that do not expand wildcards, such as PowerShell, included).</summary>
internal static class TextFiles
{
    /// <summary>
    /// The files <paramref name="inputs"/> name: files as they are, folders searched (with their subfolders) for .txt
    /// files, and file names with * or ? matched in their folder; in that order, each once.
    /// </summary>
    /// <exception cref="UsageException">An input is neither a file, a folder nor a pattern.</exception>
    public static IReadOnlyList<string> Expand(IEnumerable<string> inputs)
    {
        var files = new List<string>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (string input in inputs)
        {
            IEnumerable<string> found = Directory.Exists(input)
                ? Directory.EnumerateFiles(input, "*.txt", SearchOption.AllDirectories).Order(StringComparer.Ordinal)
                : input.IndexOfAny(['*', '?']) >= 0
                    ? (Path.GetDirectoryName(input) is { Length: > 0 } d ? d : ".") is var folder && Directory.Exists(folder)
                        ? Directory.EnumerateFiles(folder, Path.GetFileName(input)).Order(StringComparer.Ordinal) : []
                    : File.Exists(input) ? [input] : throw new UsageException($"File not found: {input}");
            files.AddRange(found.Where(seen.Add));
        }

        return files;
    }
}
