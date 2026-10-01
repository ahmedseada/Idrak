// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;

// The library's name: nothing in the repository still uses the one it had before (sources, projects, docs, tools).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] NamingGroup =
    [
        ("naming: no file in the repository uses the library's former name or its tool names", FormerNamesGone),
    ];

    private static void FormerNamesGone(Device device)
    {
        _ = device;
        string root = RepositoryRoot();
        string[] former = ["Neural" + "Sharp", "ns" + "tune", "ns" + "data"];   // split so this file does not match
        string[] skipped = ["bin", "obj", ".git", ".vs", ".idea", "__pycache__"];
        string[] binary = [".gguf", ".safetensors", ".parquet", ".onnx", ".png", ".jpg", ".bin", ".gz", ".zip", ".ikm", ".ikw"];
        var found = new List<string>();
        foreach (string path in Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories))
        {
            string relative = Path.GetRelativePath(root, path);
            if (relative.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar).Any(skipped.Contains)
                || binary.Contains(Path.GetExtension(path), StringComparer.OrdinalIgnoreCase))
            {
                continue;
            }

            string text = relative + "\n" + File.ReadAllText(path);
            found.AddRange(former.Where(n => text.Contains(n, StringComparison.OrdinalIgnoreCase)).Select(n => $"{relative}: {n}"));
        }

        Check(found.Count == 0, $"former names left: {string.Join(", ", found.Take(10))}");
    }

    // The folder holding Idrak.slnx, found from the build output folder.
    private static string RepositoryRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "Idrak.slnx")))
            {
                return dir.FullName;
            }
        }

        throw new DirectoryNotFoundException($"Idrak.slnx not found above {AppContext.BaseDirectory}.");
    }
}
