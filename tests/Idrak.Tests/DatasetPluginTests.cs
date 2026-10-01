// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Datasets;
using RegisteredSources = Idrak.Datasets.DatasetSources;

// Dataset plug-ins: file formats, sources and Parquet codecs added from outside the library through their registries.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] DatasetPluginGroup =
    [
        ("dataset plug-ins: every built-in file format, source and Parquet codec is reached through its registry", DatasetBuiltInsRegistered),
        ("dataset plug-ins: a registered file format is chosen by extension, in folders and by name in recipes", DatasetCustomFormat),
        ("dataset plug-ins: a registered source opens its prefix, with its own options, before the built-ins", DatasetCustomSource),
        ("dataset plug-ins: a registered Parquet codec decompresses the pages of its codec id", DatasetCustomCodec),
    ];

    private static void DatasetBuiltInsRegistered(Device device)
    {
        _ = device;
        foreach (var format in Enum.GetValues<DataFormat>())
        {
            Check(DataFileFormats.Get(format.ToString()).Name == format.ToString(), $"format {format} registered");
            Check(DataFileFormats.Get(format.ToString().ToLowerInvariant()).Name == format.ToString(), $"format {format} found ignoring case");
        }

        string[] files = ["a.jsonl", "a.ndjson", "a.json", "a.csv", "a.TSV", "a.parquet", "a.txt", "a.md.gz", "a.cs", "a.py"];
        DataFormat[] expected = [DataFormat.JsonLines, DataFormat.JsonLines, DataFormat.Json, DataFormat.Csv, DataFormat.Tsv, DataFormat.Parquet,
            DataFormat.Text, DataFormat.Text, DataFormat.Code, DataFormat.Code];
        for (int i = 0; i < files.Length; i++)
        {
            Check(DataFiles.FormatOf(files[i]) == expected[i] && DataFileFormats.Find(files[i])!.Name == expected[i].ToString(), $"{files[i]} is {expected[i]}");
        }

        Check(DataFileFormats.Find("a.cs", includeCode: false) is null && DataFiles.FormatOf("a.cs", includeCode: false) is null, "code only when asked");
        Check(DataFileFormats.Find("a.unknown") is null && DataFiles.FormatOf("a.unknown") is null, "unknown extension");

        Check(string.Join(",", RegisteredSources.Names) == "hf,github,kaggle,zenodo,http,folder,file", $"sources in order: {string.Join(",", RegisteredSources.Names)}");
        string here = AppContext.BaseDirectory, file = typeof(Tests).Assembly.Location;
        foreach (var (source, name) in new[] { ("hf:org/name", "hf"), ("GitHub:o/r@v1", "github"), ("kaggle:o/d", "kaggle"), ("zenodo:1", "zenodo"),
                     ("https://example.com/a.csv", "http"), ("http://example.com/a.csv", "http"), (here, "folder"), (file, "file") })
        {
            Check(RegisteredSources.Find(source)?.Name == name && RegisteredSources.Get(name).Name == name, $"{source} opened by {name}");
        }

        Check(RegisteredSources.Find("nowhere:at/all") is null, "an unknown source is found by none");
        try
        {
            _ = DatasetSpec.Parse("nowhere:at/all").Open();
            Check(false, "an unknown source should be refused");
        }
        catch (FileNotFoundException ex)
        {
            Check(ex.Message.Contains("known source", StringComparison.Ordinal), "unknown source explained");
        }

        Check(string.Join(",", ParquetCodecs.Ids) == "0,1,2,4,7" && string.Join(",", ParquetCodecs.Names) == "Uncompressed,Snappy,Gzip,Brotli,Lz4Raw",
            $"codecs: {string.Join(",", ParquetCodecs.Names)}");
        Check(ParquetCodecs.Get(0).Decompress([1, 2, 3], 3).SequenceEqual(new byte[] { 1, 2, 3 }), "uncompressed copies");
        foreach (var id in new[] { 3, 5, 6 })
        {
            try
            {
                _ = ParquetCodecs.Get(id);
                Check(false, $"codec {id} should not be registered");
            }
            catch (NotSupportedException ex)
            {
                Check(ex.Message.Contains("ParquetCodecs.Register", StringComparison.Ordinal) && (id != 6 || ex.Message.Contains("Zstandard", StringComparison.Ordinal)),
                    $"codec {id} explained: {ex.Message}");
            }
        }
    }

    // ".lines" files: each line a row {"line", "number"}.
    private sealed class LinesFormat : IDataFileFormat
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

    private static void DatasetCustomFormat(Device device)
    {
        _ = device;
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-plugin-format-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        var lines = new LinesFormat();
        var csv = DataFileFormats.Get("Csv");
        try
        {
            File.WriteAllText(Path.Combine(folder, "a.lines"), "first\nsecond\n");
            File.WriteAllText(Path.Combine(folder, "b.txt"), "x\ny\nz\n");
            Check(DataFileFormats.Find("a.lines") is null, "not known before it is registered");

            DataFileFormats.Register(lines);
            Check(DataFileFormats.Names.Last() == "Lines" && DataFileFormats.Find("A.LINES.gz") == lines, "registered and found by extension");
            Check(DataFiles.FormatOf("a.lines") is null, "a registered format has no DataFormat value");

            var rows = Dataset.FromFile(Path.Combine(folder, "a.lines")).ToList();
            Check(rows.Count == 2 && (string)rows[1]["line"]! == "second" && (int)rows[1]["number"]! == 2, "read by the registered format");

            var inFolder = Dataset.FromFolder(folder, "*.lines").ToList();
            Check(inFolder.Count == 2, $"folders include the registered format: {inFolder.Count} rows");

            var named = DatasetSpec.Parse(Path.Combine(folder, "b.txt") + "?format=lines").Open().ToList();
            Check(named.Count == 3 && (int)named[2]["number"]! == 3, "a recipe chooses the format by its name");
            var forced = Dataset.FromFile(Path.Combine(folder, "b.txt"), new ReadOptions { FileFormat = lines }).ToList();
            Check(forced.Count == 3 && forced[0]["line"] is not null, "ReadOptions.FileFormat chooses it");

            // Replacing a built-in: the enum value and the extension both reach the replacement.
            var counted = new CountingFormat(csv);
            DataFileFormats.Register(counted);
            File.WriteAllText(Path.Combine(folder, "c.csv"), "a,b\n1,2\n");
            var csvRows = Dataset.FromFile(Path.Combine(folder, "c.csv")).ToList();
            _ = DataFiles.ReadStream(() => new MemoryStream("a\n1\n"u8.ToArray()), "t.csv", DataFormat.Csv, ReadOptions.Default).ToList();
            Check(counted.Reads == 2 && csvRows.Count == 1 && (long)csvRows[0]["b"]! == 2, $"a replaced built-in is used ({counted.Reads} reads)");
            Check(string.Join(",", DataFileFormats.Names.Take(7)) == "JsonLines,Json,Csv,Tsv,Parquet,Text,Code", "a replacement keeps its place");
        }
        finally
        {
            DataFileFormats.Register(csv);
            DataFileFormats.Unregister("Lines");
            Directory.Delete(folder, true);
        }

        Check(DataFileFormats.Find("a.lines") is null && DataFileFormats.Get("Csv") == csv && DataFileFormats.Names.Count == 7, "registrations restored");
    }

    private sealed class CountingFormat(IDataFileFormat inner) : IDataFileFormat
    {
        public int Reads { get; private set; }

        public string Name => inner.Name;

        public IReadOnlyCollection<string> Extensions => inner.Extensions;

        public IEnumerable<JsonObject> Read(Func<Stream> open, string path, ReadOptions options)
        {
            Reads++;
            return inner.Read(open, path, options);
        }
    }

    // "test:name" serves files from one local folder, with its own option "shout" (upper-cases the text column).
    private sealed class FolderSource(string folder) : IDatasetSource
    {
        public string Name => "test";

        public IReadOnlyCollection<string> Options => ["shout"];

        public bool CanOpen(string source) => source.StartsWith("test:", StringComparison.OrdinalIgnoreCase);

        public Dataset Open(DatasetSpec spec, ReadOptions options, Downloader? downloader)
        {
            var data = Dataset.FromFile(Path.Combine(folder, spec.Source[5..]), options);
            return spec.Options.TryGetValue("shout", out var shout) && shout == "true"
                ? Dataset.FromRows(data.Select(r => { r["text"] = ((string)r["text"]!).ToUpperInvariant(); return r; }))
                : data;
        }
    }

    private static void DatasetCustomSource(Device device)
    {
        _ = device;
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-plugin-source-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        try
        {
            File.WriteAllText(Path.Combine(folder, "rows.jsonl"), "{\"text\":\"one\"}\n{\"text\":\"two\"}\n{\"text\":\"three\"}\n");
            RegisteredSources.Register(new FolderSource(folder));
            Check(RegisteredSources.Names.First() == "test" && RegisteredSources.Find("TEST:rows.jsonl")?.Name == "test", "registered first");

            var rows = DatasetSpec.Parse("test:rows.jsonl?skip=1&shout=true").Open().ToList();
            Check(rows.Count == 2 && (string)rows[0]["text"]! == "TWO", "opened by the registered source, with its own option and the common ones");
            try
            {
                _ = DatasetSpec.Parse("test:rows.jsonl?loud=true").Open();
                Check(false, "an unknown option should be refused");
            }
            catch (ArgumentException ex)
            {
                Check(ex.Message.Contains("shout", StringComparison.Ordinal), $"the source's options are listed as known: {ex.Message}");
            }

            try
            {
                _ = DatasetSpec.Parse(Path.Combine(folder, "rows.jsonl") + "?shout=true").Open();
                Check(false, "another source's option should be refused");
            }
            catch (ArgumentException)
            {
            }

            var recipe = new DatasetRecipe { Sources = [DatasetSpec.Parse("test:rows.jsonl")], Shuffle = false };
            var (train, _) = recipe.Build();
            Check(train.Count() == 3, "recipes use the registered source");
        }
        finally
        {
            RegisteredSources.Unregister("test");
            Directory.Delete(folder, true);
        }

        Check(string.Join(",", RegisteredSources.Names) == "hf,github,kaggle,zenodo,http,folder,file", "registrations restored");
    }

    private sealed class CountingCodec(IParquetCodec inner) : IParquetCodec
    {
        public int Pages { get; private set; }

        public int Id => inner.Id;

        public string Name => inner.Name;

        public byte[] Decompress(ReadOnlySpan<byte> input, int uncompressedSize)
        {
            Pages++;
            return inner.Decompress(input, uncompressedSize);
        }
    }

    private sealed class RefusingCodec : IParquetCodec
    {
        public int Id => 6;

        public string Name => "Zstandard";

        public byte[] Decompress(ReadOnlySpan<byte> input, int uncompressedSize) => throw new InvalidOperationException("refusing codec called");
    }

    private static void DatasetCustomCodec(Device device)
    {
        _ = device;
        var snappy = ParquetCodecs.Get(1);
        string Rows(string name) => string.Join("\n", Dataset.FromFile(TestData($"parquet/{name}.parquet")).Select(r => r.ToJsonString()));
        string before = Rows("snappy-v1");
        var counting = new CountingCodec(snappy);
        try
        {
            ParquetCodecs.Register(counting);
            Check(ParquetCodecs.Get(1) == counting, "replaced");
            Check(Rows("snappy-v1") == before, "rows unchanged through the wrapper");
            Check(counting.Pages > 0, $"the registered codec decompressed {counting.Pages} pages");

            ParquetCodecs.Register(new RefusingCodec());
            try
            {
                _ = Dataset.FromFile(TestData("parquet/zstd.parquet")).ToList();
                Check(false, "the zstd file should reach the registered codec");
            }
            catch (InvalidOperationException ex)
            {
                Check(ex.Message == "refusing codec called", "a codec id registered from outside is dispatched to");
            }
        }
        finally
        {
            ParquetCodecs.Register(snappy);
            ParquetCodecs.Unregister(6);
        }

        Check(string.Join(",", ParquetCodecs.Ids) == "0,1,2,4,7" && ParquetCodecs.Get(1) == snappy, "registrations restored");
        try
        {
            _ = Dataset.FromFile(TestData("parquet/zstd.parquet")).ToList();
            Check(false, "zstd should be refused again");
        }
        catch (NotSupportedException ex)
        {
            Check(ex.Message.Contains("Zstandard", StringComparison.Ordinal), "zstd explained");
        }
    }
}
