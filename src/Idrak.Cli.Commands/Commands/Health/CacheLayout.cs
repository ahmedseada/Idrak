// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Cli.Commands.Health;

/// <summary>
/// What lives in the cache folder, by kind: downloaded models, tuning choices measured on each device, compiled kernels,
/// and other downloads (datasets). The paths are the ones the libraries use under <c>IDRAK_CACHE</c>.
/// </summary>
internal static class CacheLayout
{
    /// <summary>One part of the cache: its kind, its paths (folders or files) relative to the cache folder.</summary>
    public sealed record Part(string Kind, string Meaning, string[] Paths);

    /// <summary>The kinds <c>cache clear</c> takes (besides "all").</summary>
    public static IReadOnlyList<Part> Parts { get; } =
    [
        new("models", "downloaded models", [Path.Combine("downloads", "huggingface", "models"), "models"]),
        new("tuning", "kernel choices measured on each device", ["tuning", Path.Combine("cpu", "tuning.tsv"), Path.Combine("vulkan", "tuning.tsv")]),
        new("kernels", "compiled kernels", [Path.Combine("hip", "kernels"), "kernels"]),
        new("downloads", "other downloads (datasets, files)", ["downloads"]),
    ];

    /// <summary>The size in bytes of a file or folder (0 when missing).</summary>
    public static long Size(string path)
    {
        if (File.Exists(path))
        {
            return new FileInfo(path).Length;
        }

        if (!Directory.Exists(path))
        {
            return 0;
        }

        long total = 0;
        foreach (var file in new DirectoryInfo(path).EnumerateFiles("*", new EnumerationOptions { RecurseSubdirectories = true, IgnoreInaccessible = true }))
        {
            total += file.Length;
        }

        return total;
    }

    /// <summary>The existing paths of a part under <paramref name="root"/>; "downloads" leaves out the models inside it.</summary>
    public static IEnumerable<string> Existing(string root, Part part) =>
        part.Paths.Select(p => Path.Combine(root, p)).Where(p => File.Exists(p) || Directory.Exists(p));

    /// <summary>The size of a part (for "downloads", without the models it holds).</summary>
    public static long Size(string root, Part part)
    {
        long size = Existing(root, part).Sum(Size);
        return part.Kind == "downloads" ? size - Size(Path.Combine(root, "downloads", "huggingface", "models")) : size;
    }

    /// <summary>Deletes a file or folder; returns the bytes freed.</summary>
    public static long Delete(string path)
    {
        long size = Size(path);
        if (File.Exists(path))
        {
            File.Delete(path);
        }
        else if (Directory.Exists(path))
        {
            Directory.Delete(path, recursive: true);
        }

        return size;
    }
}
