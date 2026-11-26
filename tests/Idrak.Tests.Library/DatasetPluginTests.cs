// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Data;
using Idrak.Data.Abstractions;

// Dataset plug-ins: a file format added from outside the library, given to a reading directly.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] DatasetPluginGroup =
    [
        ("dataset plug-ins: a file format of one's own reads a file when ReadOptions.FileFormat names it", DatasetCustomFormat),
    ];

    // ".lines" files: each line a row {"line", "number"}.
    internal sealed class LinesFormat : IDataFileFormat
    {
        public int Reads { get; private set; }

        public string Name => "Lines";

        public IReadOnlyCollection<string> Extensions => [".lines"];

        public IEnumerable<JsonObject> Read(Func<Stream> open, string path, ReadOptions options)
        {
            Reads++;
            using var reader = new StreamReader(open());
            int number = 0;
            while (reader.ReadLine() is { } line)
            {
                yield return new JsonObject { ["line"] = line, ["number"] = ++number };
            }
        }
    }

    internal static void DatasetCustomFormat(Device device)
    {
        _ = device;
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-plugin-format-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        var lines = new LinesFormat();
        try
        {
            File.WriteAllText(Path.Combine(folder, "b.txt"), "x\ny\nz\n");
            var forced = DatasetRows.FromFile(Path.Combine(folder, "b.txt"), new ReadOptions { FileFormat = lines }).ToList();
            Check(forced.Count == 3 && forced[0]["line"] is not null, "ReadOptions.FileFormat chooses it");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }
}
