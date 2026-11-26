// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Formats.Tar;
using System.IO.Compression;
using System.Text;
using Idrak.Data;
using Idrak.Data.Abstractions;

// Plan 27, R4b part 1: no programming language in the library: Idrak.Data skips only .git and the editors' folders
// unless told.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] CodingLanguageGroup =
    [
        ("data files: Idrak.Data alone skips only .git, .vs and .idea in folders and archives (ReadOptions.SkippedFolders adds more)", d => { if (d == Device.Cpu) DataFilesSkippedFolders(); }),
    ];

    private static string LanguageTemp(string name)
    {
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-languages-{name}-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        return folder;
    }

    private static void DataFilesSkippedFolders()
    {
        string root = LanguageTemp("data");
        try
        {
            string[] files =
            [
                "src/App.cs", "web/app.ts", "bin/Debug/gen.cs", "obj/x.cs", "node_modules/lib/index.js", "dist/main.js", ".angular/cache.js", "__pycache__/m.py", ".venv/lib/site.py",
                ".git/hooks/pre.py", ".vs/state.json", ".idea/workspace.xml", "coverage/report.js", "out/o.js",
            ];
            foreach (var file in files)
            {
                string path = Path.Combine(root, "repo", file);
                Directory.CreateDirectory(Path.GetDirectoryName(path)!);
                File.WriteAllText(path, "x = 1");
            }

            string folder = Path.Combine(root, "repo");
            List<string> Paths(IEnumerable<System.Text.Json.Nodes.JsonObject> rows) => [.. rows.Select(r => ((string)r["path"]!).Replace('\\', '/')).Order(StringComparer.Ordinal)];
            var documents = new ReadOptions { Documents = true, Text = TextRows.Document };
            var alone = Paths(DatasetRows.FromFolder(folder, options: documents));
            Check(ReadOptions.DefaultSkippedFolders.SequenceEqual([".git", ".vs", ".idea"]) && ReadOptions.Default.SkippedFolders.SequenceEqual([".git", ".vs", ".idea"]), "the library's default");
            Check(alone.SequenceEqual(files.Where(f => !f.StartsWith(".git/", StringComparison.Ordinal) && !f.StartsWith(".vs/", StringComparison.Ordinal) && !f.StartsWith(".idea/", StringComparison.Ordinal))
                    .Order(StringComparer.Ordinal)), $"Idrak.Data alone skips only .git, .vs and .idea: {string.Join(", ", alone)}");
            var some = Paths(DatasetRows.FromFolder(folder, options: documents with { SkippedFolders = [.. ReadOptions.DefaultSkippedFolders, "BIN"] }));
            Check(!some.Any(p => p.StartsWith("bin/", StringComparison.Ordinal)) && some.Count == alone.Count - 1, "a folder added to SkippedFolders is skipped (ignoring case)");

            string archive = Path.Combine(root, "repo.tar.gz");
            using (var file = File.Create(archive))
            using (var gzip = new GZipStream(file, CompressionLevel.Fastest))
            using (var tar = new TarWriter(gzip))
            {
                foreach (var name in files)
                {
                    tar.WriteEntry(new PaxTarEntry(TarEntryType.RegularFile, "repo-abc/" + name) { DataStream = new MemoryStream(Encoding.UTF8.GetBytes("x = 1")) });
                }
            }

            var inArchive = DatasetRows.FromFile(archive, documents).Count();
            Check(inArchive == alone.Count, $"in an archive the same: Idrak.Data alone {inArchive}");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }
}
