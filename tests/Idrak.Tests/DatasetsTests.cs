// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Formats.Tar;
using System.IO.Compression;
using System.Net;
using System.Text.Json.Nodes;
using System.Text;
using Idrak.Data.Abstractions;
using Idrak.Data;
using Idrak;

// Datasets: file formats (Parquet checked against pyarrow), archives, operations, the download cache.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] DatasetsGroup =
    [
        ("datasets: Parquet files (nested lists, structs, maps; v1/v2 pages; dictionary, delta, byte-stream-split; Snappy/Gzip/Brotli/LZ4) read as pyarrow reads them", ParquetMatchesPyarrow),
        ("datasets: JSON Lines, JSON, CSV/TSV, text and code files, also in .gz, .zip and .tar.gz", DatasetFormats),
        ("datasets: CSV and JSON Lines read in blocks as a character (line) at a time: quotes, line breaks and records across block edges, byte order marks, invalid UTF-8", DatasetReadersMatchReference),
        ("datasets: select, filter, shuffle, deduplicate, split and mix are lazy, streamed and reproducible", DatasetOperations),
        ("datasets: downloads are cached, resumed, retried, and explain missing access", DatasetDownloads),
        ("datasets: rows of common layouts (messages, ShareGPT, Alpaca, question/answer, TRL, templates) become conversations or text; specs and recipes", DatasetChatAndRecipes),
        ("datasets: recipes hold out prompts (a question and all its answers, however spaced or cased, on one side) and count repeats dropped", RecipeHeldOutPrompts),
        ("datasets: Hugging Face (splits, pages, tokens, Parquet fallback), GitHub (repositories, files, releases), Kaggle and Zenodo against a fake server", DatasetSources),
    ];

    private static void RecipeHeldOutPrompts(Device device)
    {
        _ = device;
        string folder = Path.Combine(Path.GetTempPath(), $"ns-recipe-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        try
        {
            // 300 questions, each asked three ways (cased, spaced, repeated exactly) with answers that sometimes disagree.
            string file = Path.Combine(folder, "data.csv");
            var lines = new List<string> { "question,answer" };
            for (int i = 0; i < 300; i++)
            {
                lines.Add($"what is item {i},a{i % 3}");
                lines.Add($"What is  Item {i},a{(i + 1) % 3}");
                lines.Add($"what is item {i},a{i % 3}");
            }

            File.WriteAllLines(file, lines);
            var recipe = new DatasetRecipe
            {
                Sources = [DatasetSpec.Parse(file + "?user={question}&assistant={answer}")], EvaluationFraction = 0.2, Seed = 3, Shuffle = false,
            };
            var counts = new RecipeCounts();
            var (train, evaluation) = recipe.Build(null, counts);
            string Question(JsonObject row) => string.Join(' ', ((string)row["messages"]![0]!["content"]!).Split(' ', StringSplitOptions.RemoveEmptyEntries)).ToLowerInvariant();
            var trainQuestions = train.Select(Question).ToHashSet();
            Check(counts.Rows == 900 && counts.Duplicates == 300 && counts.Kept == 600, $"counts: {counts.Rows} rows, {counts.Duplicates} repeats, {counts.Kept} kept");
            var heldOut = evaluation!.Select(Question).ToList();
            Check(heldOut.Count is > 60 and < 180, $"about a fifth held out: {heldOut.Count} of 600");
            Check(heldOut.All(q => !trainQuestions.Contains(q)), "no held-out question is also trained on (in any casing or spacing, with any answer)");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    private static void DatasetChatAndRecipes(Device device)
    {
        _ = device;
        JsonObject Row(string json) => JsonNode.Parse(json)!.AsObject();
        string? Chat(string json, ChatMapping? mapping = null, RowKind kind = RowKind.Auto, string? system = null) =>
            ChatRows.Normalize(Row(json), kind, mapping, system)?.ToJsonString();

        Check(Chat("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":\"yo\",\"reasoning_content\":\"r\"}],\"tools\":\"[{\\\"type\\\":\\\"function\\\",\\\"function\\\":{\\\"name\\\":\\\"f\\\"}}]\"}")
              == "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":\"yo\",\"reasoning_content\":\"r\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\"}}]}",
            "messages kept, reasoning kept, tools parsed from a JSON string");
        Check(Chat("{\"conversations\":[{\"from\":\"system\",\"value\":\"s\"},{\"from\":\"human\",\"value\":\"q\"},{\"from\":\"gpt\",\"value\":\"a\"}]}")
              == "{\"messages\":[{\"role\":\"system\",\"content\":\"s\"},{\"role\":\"user\",\"content\":\"q\"},{\"role\":\"assistant\",\"content\":\"a\"}]}", "ShareGPT");
        Check(Chat("{\"instruction\":\"Translate\",\"input\":\"hola\",\"output\":\"hello\"}")
              == "{\"messages\":[{\"role\":\"user\",\"content\":\"Translate\\n\\nhola\"},{\"role\":\"assistant\",\"content\":\"hello\"}]}", "Alpaca");
        Check(Chat("{\"instruction\":\"Say hi\",\"input\":\"\",\"output\":\"hi\"}")!.Contains("\"content\":\"Say hi\"", StringComparison.Ordinal), "Alpaca without input");
        Check(Chat("{\"question\":\"two and two?\",\"answer\":\"4\"}", system: "Be exact.")
              == "{\"messages\":[{\"role\":\"system\",\"content\":\"Be exact.\"},{\"role\":\"user\",\"content\":\"two and two?\"},{\"role\":\"assistant\",\"content\":\"4\"}]}", "question/answer with a system prompt");
        Check(Chat("{\"prompt\":[{\"role\":\"user\",\"content\":\"p\"}],\"completion\":[{\"role\":\"assistant\",\"content\":\"c\"}]}")!.Contains("\"content\":\"c\"", StringComparison.Ordinal), "TRL prompt/completion");
        Check(Chat("{\"text\":\"plain words\"}") == "{\"text\":\"plain words\"}" && Chat("{\"text\":\"plain\"}", kind: RowKind.Chat) is null, "text rows");
        Check(Chat("{\"question\":\"q\",\"answer\":\"a\"}", kind: RowKind.Text) == "{\"text\":\"q\\n\\na\"}", "conversations as text");
        Check(Chat("{\"label\":1}") is null, "unrecognized rows are dropped");
        var mapping = new ChatMapping { User = "Title: {title}\n\n{body}\n\n{missing_col}", Assistant = "{summary}", System = "{nothing}" };
        Check(Chat("{\"title\":\"T\",\"body\":\"B\",\"summary\":\"S\",\"nothing\":\"\"}", mapping)
              == "{\"messages\":[{\"role\":\"user\",\"content\":\"Title: T\\n\\nB\\n\\n{missing_col}\"},{\"role\":\"assistant\",\"content\":\"S\"}]}", "templates");
        Check(ChatRows.Describe(Row("{\"instruction\":\"i\",\"output\":\"o\"}")) == "alpaca" && ChatRows.Describe(Row("{\"conversations\":[{\"from\":\"human\",\"value\":\"x\"}]}")) == "sharegpt (conversations)",
            "layouts described");

        var spec = DatasetSpec.Parse("hf:openai/gsm8k?config=main&split=test&user={question}&assistant={answer}&weight=0.5&take=10");
        Check(spec.Source == "hf:openai/gsm8k" && spec.Options["split"] == "test" && spec.Weight == 0.5 && spec.Mapping is { User: "{question}", Assistant: "{answer}" }, "spec options");
        Check(DatasetSpec.Parse("https://host/data.jsonl?sig=abc").Source == "https://host/data.jsonl?sig=abc" && DatasetSpec.Parse("https://host/d.jsonl?take=5").Source == "https://host/d.jsonl",
            "URLs keep their own query");

        string root = Path.Combine(Path.GetTempPath(), "ns-recipe-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            File.WriteAllLines(Path.Combine(root, "qa.jsonl"), Enumerable.Range(0, 300).Select(i => $"{{\"q\":\"question {i % 250}\",\"a\":\"answer {i % 250}\"}}"));
            File.WriteAllLines(Path.Combine(root, "chat.jsonl"), Enumerable.Range(0, 100).Select(i => $"{{\"messages\":[{{\"role\":\"user\",\"content\":\"hi {i}\"}},{{\"role\":\"assistant\",\"content\":\"hello {i}\"}}]}}"));
            File.WriteAllLines(Path.Combine(root, "notes.txt"), ["short", "a line long enough to keep", "another line long enough"]);
            File.WriteAllText(Path.Combine(root, "recipe.json"), """
                {
                  "sources": [
                    {"source": "qa.jsonl", "user": "{q}", "assistant": "{a}"},
                    "chat.jsonl?take=50",
                    "notes.txt"
                  ],
                  "system": "Be kind.", "seed": 4, "min_chars": 10, "eval_fraction": 0.1
                }
                """);
            var recipe = DatasetRecipe.Load(Path.Combine(root, "recipe.json"));
            var (train, evaluation) = recipe.Build();
            var trainRows = train.ToList();
            var evalRows = evaluation!.ToList();
            int total = trainRows.Count + evalRows.Count;
            Check(total == 250 + 50 + 2, $"recipe rows: {total} (duplicates and short rows dropped)");
            Check(evalRows.Count is > 10 and < 60 && !trainRows.Select(r => r.ToJsonString()).Intersect(evalRows.Select(r => r.ToJsonString())).Any(), $"held out {evalRows.Count}");
            Check(trainRows.Where(r => r.ContainsKey("messages")).All(r => (string?)r["messages"]![0]!["content"] == "Be kind.") && trainRows.Count(r => r.ContainsKey("text")) <= 2, "system prompt, text rows");
            Check(recipe.Build().Train.Select(r => r.ToJsonString()).SequenceEqual(trainRows.Select(r => r.ToJsonString())), "a recipe builds the same rows each time");
            var weighted = DatasetRecipe.Of(Path.Combine(root, "qa.jsonl") + "?weight=3&user={q}&assistant={a}", Path.Combine(root, "chat.jsonl") + "?weight=1") with { Deduplicate = false, Stop = MixStop.FirstExhausted };
            var mixed = weighted.Build().Train.ToList();
            Check(mixed.Count < 400 && mixed.Count(r => ((string)r["messages"]![0]!["content"]!).StartsWith("question", StringComparison.Ordinal)) > mixed.Count / 2, "weights mix the sources");
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }

    private static void DatasetSources(Device device)
    {
        _ = device;
        string cache = Path.Combine(Path.GetTempPath(), "ns-cache-" + Guid.NewGuid().ToString("N"));
        string? kaggleUser = Environment.GetEnvironmentVariable("KAGGLE_USERNAME"), kaggleKey = Environment.GetEnvironmentVariable("KAGGLE_KEY");
        try
        {
            var parquet = File.ReadAllBytes(TestData("parquet/snappy-v1.parquet"));
            var web = new FakeRouter();
            var downloader = new Downloader(new HttpClient(web), cache) { Attempts = 1 };

            // Hugging Face: a repository with split-named Parquet shards, listed over two pages.
            const string hf = "https://huggingface.co";
            const string commit = "0123456789abcdef0123456789abcdef01234567";
            foreach (var name in new[] { "chat", "scripted", "sft" })
            {
                web.Json($"{hf}/api/datasets/org/{name}/revision/main", $"{{\"sha\":\"{commit}\"}}");
            }

            web.Json($"{hf}/api/datasets/org/chat/tree/{commit}?recursive=true",
                "[{\"type\":\"file\",\"path\":\"README.md\",\"size\":10},{\"type\":\"file\",\"path\":\".gitattributes\"},{\"type\":\"directory\",\"path\":\"data\"},"
                + "{\"type\":\"file\",\"path\":\"data/train-00000-of-00002.parquet\"}]",
                next: $"{hf}/api/datasets/org/chat/tree/{commit}?recursive=true&cursor=2");
            web.Json($"{hf}/api/datasets/org/chat/tree/{commit}?recursive=true&cursor=2",
                "[{\"type\":\"file\",\"path\":\"data/train-00001-of-00002.parquet\"},{\"type\":\"file\",\"path\":\"data/test-00000-of-00001.parquet\"},"
                + "{\"type\":\"file\",\"path\":\"data/train.jsonl\"}]");
            foreach (var shard in new[] { "train-00000-of-00002", "train-00001-of-00002", "test-00000-of-00001" })
            {
                web.Bytes($"{hf}/datasets/org/chat/resolve/{commit}/data/{shard}.parquet", parquet, requireToken: "hf_secret");
            }

            var train = HuggingFace.Dataset("org/chat", token: "hf_secret", downloader: downloader);
            Check(train.Count() == 80 && train.First()["messages"] is JsonArray, $"hf train split: two shards, Parquet preferred over JSON Lines ({train.Count()} rows)");
            Check(HuggingFace.Dataset("org/chat", split: "test", token: "hf_secret", downloader: downloader).Count() == 40, "hf test split");
            var files = HuggingFace.Dataset("org/chat", token: "hf_secret", downloader: downloader).Download();
            Check(files.Count == 2 && files.All(File.Exists), "download: the local files of a source");
            Check(files[0] == Path.Combine(cache, "huggingface", "datasets", "org", "chat", "0123456789ab", "data", "train-00000-of-00002.parquet"), $"cache laid out by source: {files[0]}");
            Check(DatasetRows.FromFile(TestData("parquet/delta.parquet")).Download().Single().EndsWith("delta.parquet", StringComparison.Ordinal), "download: a local file is itself");
            Check(HuggingFace.Dataset("org/chat", token: "hf_secret", maxFiles: 1, downloader: downloader).Count() == 40, "hf first files only");
            web.Json($"{hf}/api/datasets/org/sft/tree/{commit}?recursive=true",
                "[{\"type\":\"file\",\"path\":\"data/train_sft-00000-of-00001-ab12.parquet\"},{\"type\":\"file\",\"path\":\"data/train_gen-00000-of-00001-cd34.parquet\"},"
                + "{\"type\":\"file\",\"path\":\"data/test_sft-00000-of-00001-ef56.parquet\"}]");
            web.Bytes($"{hf}/datasets/org/sft/resolve/{commit}/data/train_sft-00000-of-00001-ab12.parquet", parquet);
            Check(HuggingFace.Dataset("org/sft", split: "train_sft", downloader: downloader).Count() == 40, "hf split with its own name (train_sft)");
            Check(HuggingFace.Dataset("org/chat", files: "data/test-*", token: "hf_secret", downloader: downloader).Count() == 40, "hf files by pattern");
            try
            {
                _ = HuggingFace.Dataset("org/chat", split: "validation", token: "hf_secret", downloader: downloader).Count();
                Check(false, "a missing split should fail");
            }
            catch (HttpRequestException ex)
            {
                Check(ex.Message.Contains("/parquet", StringComparison.Ordinal), $"missing split falls back to the Parquet index: {ex.Message}");
            }

            // A dataset without plain data files (a loading script): the Hub's Parquet copy.
            web.Json($"{hf}/api/datasets/org/scripted/tree/{commit}?recursive=true", "[{\"type\":\"file\",\"path\":\"scripted.py\"}]");
            web.Json($"{hf}/api/datasets/org/scripted/parquet", $"{{\"en\":{{\"train\":[\"{hf}/api/datasets/org/scripted/parquet/en/train/0.parquet\"]}}}}");
            web.Bytes($"{hf}/api/datasets/org/scripted/parquet/en/train/0.parquet", parquet);
            Check(HuggingFace.Dataset("org/scripted", downloader: downloader).Count() == 40, "hf converted Parquet");

            web.Json($"{hf}/api/datasets/org/gated/revision/main", "{}", status: HttpStatusCode.Unauthorized);
            try
            {
                _ = HuggingFace.Dataset("org/gated", downloader: downloader).Count();
                Check(false, "gated without a token should fail");
            }
            catch (HttpRequestException ex)
            {
                Check(ex.Message.Contains("sign-in needed", StringComparison.Ordinal), ex.Message);
            }

            // GitHub: a repository snapshot as a code dataset.
            var tarball = new MemoryStream();
            using (var gzip = new GZipStream(tarball, CompressionLevel.Fastest, leaveOpen: true))
            using (var tar = new TarWriter(gzip))
            {
                foreach (var (name, text) in new[] { ("owner-app-1a2b3c/src/Program.cs", "Console.WriteLine();"), ("owner-app-1a2b3c/web/main.ts", "bootstrap();"),
                             ("owner-app-1a2b3c/bin/Debug/app.cs", "generated"), ("owner-app-1a2b3c/data/rows.jsonl", "{\"x\":1}") })
                {
                    tar.WriteEntry(new PaxTarEntry(TarEntryType.RegularFile, name) { DataStream = new MemoryStream(Encoding.UTF8.GetBytes(text)) });
                }
            }

            web.Json("https://api.github.com/repos/owner/app/commits/HEAD", "1a2b3c4d5e6f7a8b9c0d");
            web.Bytes("https://api.github.com/repos/owner/app/tarball/1a2b3c4d5e6f7a8b9c0d", tarball.ToArray(), requireToken: "gh_secret");
            var code = GitHub.Repository("owner/app", token: "gh_secret", downloader: downloader).ToList();
            Check(code.Select(r => (string)r["path"]!).Order().SequenceEqual(["data/rows.jsonl", "src/Program.cs", "web/main.ts"]) && code.All(r => (string?)r["repo"] == "owner/app"),
                $"github repository: {string.Join(", ", code.Select(r => r["path"]))}");
            Check(GitHub.Repository("owner/app", pattern: "*.cs", token: "gh_secret", downloader: downloader).Single()["language"]!.GetValue<string>() == "csharp", "github pattern");

            web.Json("https://api.github.com/repos/owner/data/commits/HEAD", "fedcba9876543210fedc");
            web.Json("https://api.github.com/repos/owner/data/git/trees/fedcba9876543210fedc?recursive=1",
                "{\"tree\":[{\"type\":\"blob\",\"path\":\"sets/a.jsonl\"},{\"type\":\"blob\",\"path\":\"sets/b.jsonl\"},{\"type\":\"tree\",\"path\":\"sets\"},{\"type\":\"blob\",\"path\":\"README.md\"}]}");
            web.Bytes("https://api.github.com/repos/owner/data/contents/sets/a.jsonl?ref=fedcba9876543210fedc", Encoding.UTF8.GetBytes("{\"q\":1}\n{\"q\":2}\n"));
            web.Bytes("https://api.github.com/repos/owner/data/contents/sets/b.jsonl?ref=fedcba9876543210fedc", Encoding.UTF8.GetBytes("{\"q\":3}\n"));
            Check(GitHub.Files("owner/data", "sets/*.jsonl", downloader: downloader).Count() == 3, "github files");
            Check(GitHub.Files("owner/data", "sets/*.jsonl", downloader: downloader).Download()[0] == Path.Combine(cache, "github", "owner", "data", "fedcba987654", "sets", "a.jsonl"),
                "github files cached per commit");
            Check(GitHub.Repository("owner/app", token: "gh_secret", downloader: downloader).Download().Single() == Path.Combine(cache, "github", "owner", "app", "1a2b3c4d5e6f.tar.gz"),
                "github snapshot cached per commit");

            web.Json("https://api.github.com/repos/owner/data/releases/latest",
                "{\"tag_name\":\"v2\",\"assets\":[{\"name\":\"rows.csv\",\"url\":\"https://api.github.com/repos/owner/data/releases/assets/7\"},{\"name\":\"notes.pdf\",\"url\":\"x\"}]}");
            web.Bytes("https://api.github.com/repos/owner/data/releases/assets/7", Encoding.UTF8.GetBytes("a,b\n1,2\n3,4\n"));
            Check(GitHub.Release("owner/data", "*.csv", downloader: downloader).Count() == 2, "github release assets");
            Check(File.Exists(Path.Combine(cache, "github", "owner", "data", "releases", "v2", "rows.csv")), "release assets under their tag");

            // Kaggle: a zip, with the account's key.
            Environment.SetEnvironmentVariable("KAGGLE_USERNAME", "me");
            Environment.SetEnvironmentVariable("KAGGLE_KEY", "k123");
            var zip = new MemoryStream();
            using (var archive = new ZipArchive(zip, ZipArchiveMode.Create, leaveOpen: true))
            {
                using (var writer = new StreamWriter(archive.CreateEntry("reviews.csv").Open()))
                {
                    writer.Write("text,stars\ngood,5\nbad,1\n");
                }

                using (var writer = new StreamWriter(archive.CreateEntry("other.json").Open()))
                {
                    writer.Write("[{\"x\":1}]");
                }
            }

            web.Bytes("https://www.kaggle.com/api/v1/datasets/download/someone/reviews", zip.ToArray(),
                requireAuthorization: "Basic " + Convert.ToBase64String(Encoding.UTF8.GetBytes("me:k123")));
            Check(Kaggle.Dataset("someone/reviews", "*.csv", downloader: downloader).Count() == 2, "kaggle");
            Check(File.Exists(Path.Combine(cache, "kaggle", "someone", "reviews", "latest", "reviews.zip")), "kaggle cache path");

            web.Json("https://zenodo.org/api/records/123",
                "{\"files\":[{\"key\":\"data.jsonl\",\"links\":{\"self\":\"https://zenodo.org/api/records/123/files/data.jsonl/content\"}},{\"key\":\"paper.pdf\",\"links\":{\"self\":\"y\"}}]}");
            web.Bytes("https://zenodo.org/api/records/123/files/data.jsonl/content", Encoding.UTF8.GetBytes("{\"a\":1}\n"));
            Check(Zenodo.Record("123", downloader: downloader).Count() == 1, "zenodo");
            Check(File.Exists(Path.Combine(cache, "zenodo", "123", "data.jsonl")), "zenodo cache path");
        }
        finally
        {
            Environment.SetEnvironmentVariable("KAGGLE_USERNAME", kaggleUser);
            Environment.SetEnvironmentVariable("KAGGLE_KEY", kaggleKey);
            if (Directory.Exists(cache))
            {
                Directory.Delete(cache, true);
            }
        }
    }

    // Answers fixed URLs, as the real services would.
    private sealed class FakeRouter : HttpMessageHandler
    {
        private readonly Dictionary<string, Func<HttpRequestMessage, HttpResponseMessage>> _routes = [];

        public void Json(string url, string json, string? next = null, HttpStatusCode status = HttpStatusCode.OK) => _routes[url] = _ =>
        {
            var response = new HttpResponseMessage(status) { Content = new StringContent(json, Encoding.UTF8, "application/json") };
            if (next is not null)
            {
                response.Headers.TryAddWithoutValidation("Link", $"<{next}>; rel=\"next\"");
            }

            return response;
        };

        public void Bytes(string url, byte[] body, string? requireToken = null, string? requireAuthorization = null) => _routes[url] = request =>
        {
            string? given = request.Headers.Authorization?.ToString();
            string? wanted = requireAuthorization ?? (requireToken is null ? null : $"Bearer {requireToken}");
            if (wanted is not null && given != wanted)
            {
                return new HttpResponseMessage(HttpStatusCode.Unauthorized) { Content = new StringContent("unauthorized") };
            }

            var response = new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(body) };
            response.Content.Headers.ContentLength = body.Length;
            return response;
        };

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
            Task.FromResult(_routes.TryGetValue(request.RequestUri!.ToString(), out var route)
                ? route(request)
                : new HttpResponseMessage(HttpStatusCode.NotFound) { Content = new StringContent($"no route for {request.RequestUri}") });
    }

    // tests/Idrak.Tests/data, found from the build output folder.
    private static string TestData(string relative)
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "tests", "Idrak.Tests", "data", relative);
            if (File.Exists(candidate) || Directory.Exists(candidate))
            {
                return candidate;
            }
        }

        throw new FileNotFoundException($"Test data '{relative}' not found above {AppContext.BaseDirectory}.");
    }

    // JSON equality with numbers compared by value (1.0 == 1) and object keys in any order.
    private static bool SameJson(JsonNode? a, JsonNode? b) => (a, b) switch
    {
        (null, null) => true,
        (JsonObject x, JsonObject y) => x.Count == y.Count && x.All(p => y.TryGetPropertyValue(p.Key, out var v) && SameJson(p.Value, v)),
        (JsonArray x, JsonArray y) => x.Count == y.Count && x.Zip(y).All(p => SameJson(p.First, p.Second)),
        (JsonValue x, JsonValue y) when x.GetValueKind() == System.Text.Json.JsonValueKind.Number && y.GetValueKind() == System.Text.Json.JsonValueKind.Number =>
            Math.Abs(double.Parse(x.ToJsonString(), System.Globalization.CultureInfo.InvariantCulture) - double.Parse(y.ToJsonString(), System.Globalization.CultureInfo.InvariantCulture)) < 1e-9,
        (JsonValue x, JsonValue y) => x.ToJsonString() == y.ToJsonString(),
        _ => false,
    };

    private static void ParquetMatchesPyarrow(Device device)
    {
        _ = device;
        foreach (var name in new[] { "snappy-v1", "gzip-v2", "brotli-groups", "lz4-plain", "delta" })
        {
            var rows = DatasetRows.FromFile(TestData($"parquet/{name}.parquet")).ToList();
            var expected = File.ReadAllLines(TestData($"parquet/{name}.jsonl")).Select(l => JsonNode.Parse(l)!).ToList();
            Check(rows.Count == expected.Count, $"{name}: {rows.Count} rows, pyarrow {expected.Count}");
            for (int i = 0; i < rows.Count; i++)
            {
                Check(SameJson(rows[i], expected[i]), $"{name} row {i}:\n  ours    {rows[i].ToJsonString()}\n  pyarrow {expected[i].ToJsonString()}");
            }
        }

        using (var stream = File.OpenRead(TestData("parquet/brotli-groups.parquet")))
        {
            var (count, columns) = ParquetFile.Describe(stream);
            Check(count == 300 && columns[0] == "id" && columns.Contains("messages"), "footer: rows and columns");
            var projected = ParquetFile.ReadRows(stream, ["id", "messages"]).First();
            Check(projected.Count == 2 && projected["messages"] is JsonArray, "column projection");
        }

        try
        {
            _ = DatasetRows.FromFile(TestData("parquet/zstd.parquet")).ToList();
            Check(false, "zstd should be refused");
        }
        catch (NotSupportedException ex)
        {
            Check(ex.Message.Contains("Zstandard", StringComparison.Ordinal), "zstd explained");
        }
    }

    private static void DatasetFormats(Device device)
    {
        _ = device;
        string root = Path.Combine(Path.GetTempPath(), "ns-data-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            string P(string name) => Path.Combine(root, name);
            File.WriteAllText(P("a.jsonl"), "{\"q\":\"one\",\"n\":1}\n\n{\"q\":\"two\",\"n\":2}\n");
            File.WriteAllText(P("b.json"), "[{\"q\":\"x\"},{\"q\":\"y\"}]");
            File.WriteAllText(P("c.json"), "{\"version\":1,\"data\":[{\"q\":\"x\"},{\"q\":\"y\"},{\"q\":\"z\"}]}");
            File.WriteAllText(P("d.json"), "{\"q\":[\"x\",\"y\"],\"n\":[1,2]}");
            File.WriteAllText(P("e.json"), "{\"q\":\"x\"}\n{\"q\":\"y\"}\n");
            File.WriteAllText(P("f.csv"), "name,count,ratio,ok,zip,note\r\nmug,3,0.5,true,01234,\"says \"\"hi\"\", then\nleaves\"\r\npan,,1e3,FALSE,7,\r\n");
            File.WriteAllText(P("g.tsv"), "a\tb\n1\tx y\n");
            File.WriteAllText(P("h.txt"), "first line\n\nsecond\nthird\n\n\nfourth\n");

            Check(DatasetRows.FromFile(P("a.jsonl")).Select(r => r).Count() == 2, "jsonl skips blank lines");
            Check(DatasetRows.FromFile(P("b.json")).Count() == 2 && DatasetRows.FromFile(P("c.json")).Count() == 3, "json arrays, wrapped arrays");
            Check(DatasetRows.FromFile(P("d.json")).Last().ToJsonString() == "{\"q\":\"y\",\"n\":2}", "json columns");
            Check(DatasetRows.FromFile(P("e.json")).Count() == 2, ".json holding JSON Lines");
            var csv = DatasetRows.FromFile(P("f.csv")).ToList();
            Check(csv.Count == 2 && (long)csv[0]["count"]! == 3 && (double)csv[0]["ratio"]! == 0.5 && (bool)csv[0]["ok"]! && (string?)csv[0]["zip"] == "01234",
                $"csv types: {csv[0].ToJsonString()}");
            Check((string?)csv[0]["note"] == "says \"hi\", then\nleaves" && csv[1]["count"] is null && (double)csv[1]["ratio"]! == 1000 && !(bool)csv[1]["ok"]!, "csv quoting and empty cells");
            Check(DatasetRows.FromFile(P("g.tsv")).Single().ToJsonString() == "{\"a\":1,\"b\":\"x y\"}", "tsv");
            Check(DatasetRows.FromFile(P("h.txt")).Count() == 4, "text lines");
            Check(DatasetRows.FromFile(P("h.txt"), new ReadOptions { Text = TextRows.Paragraphs }).Select(r => r).Count() == 3, "text paragraphs");
            Check(((string?)DatasetRows.FromFile(P("h.txt"), new ReadOptions { Text = TextRows.Document }).Single()["text"])!.StartsWith("first line", StringComparison.Ordinal), "text document");

            using (var gz = new GZipStream(File.Create(P("i.jsonl.gz")), CompressionLevel.Fastest))
            {
                gz.Write(Encoding.UTF8.GetBytes("{\"q\":1}\n{\"q\":2}\n{\"q\":3}\n"));
            }

            Check(DatasetRows.FromFile(P("i.jsonl.gz")).Count() == 3, "gzip");

            using (var zip = ZipFile.Open(P("j.zip"), ZipArchiveMode.Create))
            {
                zip.CreateEntryFromFile(P("a.jsonl"), "data/a.jsonl");
                zip.CreateEntryFromFile(P("f.csv"), "data/f.csv");
                zip.CreateEntryFromFile(TestData("parquet/snappy-v1.parquet"), "data/rows.parquet");
                zip.CreateEntry("README").Open().Dispose();
            }

            Check(DatasetRows.FromFile(P("j.zip")).Count() == 2 + 2 + 40, "zip: every data file");
            Check(DatasetRows.FromFile(P("j.zip"), new ReadOptions { Pattern = "*.parquet", IncludeFile = true }).First()["_file"]!.GetValue<string>() == "data/rows.parquet",
                "zip pattern, source column");

            // A repository snapshot, as GitHub's tarballs are: code files become documents; dependencies are skipped.
            using (var tarFile = File.Create(P("k.tar.gz")))
            using (var gzip = new GZipStream(tarFile, CompressionLevel.Fastest))
            using (var tar = new TarWriter(gzip))
            {
                void Add(string name, string text)
                {
                    var entry = new PaxTarEntry(TarEntryType.RegularFile, name) { DataStream = new MemoryStream(Encoding.UTF8.GetBytes(text)) };
                    tar.WriteEntry(entry);
                }

                Add("repo-abc/src/App.cs", "class App { }");
                Add("repo-abc/web/app.component.ts", "export class AppComponent {}");
                Add("repo-abc/node_modules/x/index.js", "module.exports = 1;");
                Add("repo-abc/logo.png", "\0\0binary");
                Add("repo-abc/README.md", "# Title\n\nText.");
            }

            var code = DatasetRows.FromFile(P("k.tar.gz"), new ReadOptions { IncludeCode = true, Text = TextRows.Document }).ToList();
            Check(code.Count == 3 && code.Any(r => (string?)r["language"] == "csharp") && code.Any(r => (string?)r["language"] == "typescript")
                  && code.All(r => !((string)r["path"]!).Contains("node_modules", StringComparison.Ordinal)), $"tar.gz code: {string.Join(", ", code.Select(r => r["path"]))}");

            Directory.CreateDirectory(P("nested/deeper"));
            File.WriteAllText(P("nested/deeper/x.jsonl"), "{\"q\":1}\n");
            Check(DatasetRows.FromFolder(root, "**/*.jsonl").Count() == 2 + 1, "folder with a glob");
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }

    private static void DatasetOperations(Device device)
    {
        _ = device;
        int reads = 0;
        var numbers = new DatasetRows(() => { reads++; return Enumerable.Range(0, 1000).Select(i => new JsonObject { ["n"] = i, ["even"] = i % 2 == 0 }); }, "numbers");
        var chained = numbers.Where(r => (bool)r["even"]!).Select(r => { r["half"] = (int)r["n"]! / 2; return r; }).Skip(10).Take(5);
        Check(reads == 0, "operations are lazy");
        Check(string.Join(",", chained.Select(r => (int)r["half"]!)) == "10,11,12,13,14" && reads == 1, "filter, map, skip, take");

        var shuffled = numbers.Shuffle(seed: 7).Select(r => (int)r["n"]!).ToList();
        Check(shuffled.Order().SequenceEqual(Enumerable.Range(0, 1000)) && !shuffled.SequenceEqual(Enumerable.Range(0, 1000)), "shuffle is a permutation");
        Check(numbers.Shuffle(seed: 7).Select(r => (int)r["n"]!).SequenceEqual(shuffled) && !numbers.Shuffle(seed: 8).Select(r => (int)r["n"]!).SequenceEqual(shuffled), "shuffle seeds");
        var local = numbers.Shuffle(seed: 1, buffer: 50).Select(r => (int)r["n"]!).ToList();
        Check(local.Order().SequenceEqual(Enumerable.Range(0, 1000)), "buffered shuffle keeps every row");

        var texts = DatasetRows.FromRows([new() { ["t"] = "Hello  World" }, new() { ["t"] = "hello world" }, new() { ["t"] = "Hello  World" }, new() { ["t"] = "other" }]);
        Check(texts.Deduplicate().Count() == 3, "deduplicate: exact values");
        var edges = DatasetRows.FromRows([
            new() { ["a"] = "ab", ["b"] = "c" }, new() { ["a"] = "a", ["b"] = "bc" },          // same characters, other columns
            new() { ["a"] = "", ["b"] = "x" }, new() { ["b"] = "x" },                          // empty text and a missing column
            new() { ["a"] = "5", ["b"] = "x" }, new() { ["a"] = 5, ["b"] = "x" },               // text and a number
            new() { ["a"] = "ab", ["b"] = "c" },                                                 // a real repeat
            new() { ["a"] = "طويل جداً " + new string('x', 1000), ["b"] = "c" }, new() { ["a"] = "طويل جداً " + new string('x', 1000), ["b"] = "c" }]);
        Check(edges.Deduplicate(["a", "b"]).Count() == 7, "deduplicate: column boundaries, missing vs empty, number vs text differ; repeats go");
        var spacing = new CaseAndSpacing();
        Check(texts.Normalize(spacing, "t").Deduplicate(["t"]).Select(r => (string)r["t"]!).SequenceEqual(["hello world", "other"]),
            "normalize in place, then deduplicate");
        var keyed = texts.Normalize(spacing, "t", into: "key").Deduplicate(["key"]).ToList();
        Check(keyed.Count == 2 && (string)keyed[0]["t"]! == "Hello  World" && (string)keyed[0]["key"]! == "hello world",
            "normalize into a key column: the text stays as written, the key deduplicates");
        Check(texts.Deduplicate().Select(r => (string)r["t"]!).First() == "Hello  World", "normalizing is a separate step: the source rows are untouched");
        var nonText = DatasetRows.FromRows([new() { ["t"] = 5 }, new() { ["u"] = "x" }]).Normalize(spacing, "t").ToList();
        Check((int)nonText[0]["t"]! == 5 && !nonText[1].ContainsKey("t"), "normalize leaves values that are not text, and missing columns, as they are");

        var (train, evaluation) = numbers.Split(0.1, seed: 3);
        var trainSet = train.Select(r => (int)r["n"]!).ToHashSet();
        var evalSet = evaluation.Select(r => (int)r["n"]!).ToHashSet();
        Check(trainSet.Count + evalSet.Count == 1000 && !trainSet.Overlaps(evalSet) && evalSet.Count is > 60 and < 140, $"split sizes {trainSet.Count}/{evalSet.Count}");
        Check(numbers.Shuffle(5).Split(0.1, seed: 3).Evaluation.Select(r => (int)r["n"]!).ToHashSet().SetEquals(evalSet), "split does not depend on order");

        var a = new DatasetRows(() => Enumerable.Range(0, 10_000).Select(i => new JsonObject { ["from"] = "a" }), "a");
        var b = new DatasetRows(() => Enumerable.Range(0, 10_000).Select(i => new JsonObject { ["from"] = "b" }), "b");
        var mixed = DatasetRows.Mix([(a, 3), (b, 1)], seed: 2).Take(4000).Select(r => (string)r["from"]!).ToList();
        double shareA = mixed.Count(f => f == "a") / 4000.0;
        Check(shareA is > 0.72 and < 0.78, $"mix proportions {shareA:F3}");
        var small = new DatasetRows(() => Enumerable.Range(0, 5).Select(i => new JsonObject { ["from"] = "s" }), "s");
        Check(DatasetRows.Mix([(small, 1), (a, 1)], seed: 1).Count() < 30, "mix stops at the first exhausted source");
        Check(DatasetRows.Mix([(small, 1), (a.Take(20), 1)], seed: 1, stop: MixStop.AllExhausted).Count() == 25, "mix can run every source out");

        var row = DatasetRows.FromRows([new() { ["a"] = 1, ["b"] = 2, ["c"] = 3 }]);
        Check(row.SelectColumns("c", "a").Single().ToJsonString() == "{\"c\":3,\"a\":1}" && row.RemoveColumns("b").Single().ToJsonString() == "{\"a\":1,\"c\":3}"
              && row.RenameColumn("a", "z").Single().ContainsKey("z") && string.Join(",", row.Columns()) == "a,b,c", "columns");

        string file = Path.Combine(Path.GetTempPath(), $"ns-rows-{Guid.NewGuid():N}.jsonl");
        try
        {
            var unicode = DatasetRows.FromRows([new() { ["t"] = "é \"q\" <b>" }]);
            Check(unicode.WriteJsonLines(file) == 1 && File.ReadAllText(file) == "{\"t\":\"é \\\"q\\\" <b>\"}\n", "json lines output is readable UTF-8");
            Check(SameJson(DatasetRows.FromFile(file).Single(), unicode.Single()), "round trip");
        }
        finally
        {
            File.Delete(file);
        }
    }

    private static void DatasetDownloads(Device device)
    {
        _ = device;
        string cache = Path.Combine(Path.GetTempPath(), "ns-cache-" + Guid.NewGuid().ToString("N"));
        try
        {
            var server = new FakeFileServer();
            server.Files["https://data.example/rows.jsonl"] = Encoding.UTF8.GetBytes(string.Concat(Enumerable.Range(0, 100).Select(i => $"{{\"i\":{i}}}\n")));
            var downloader = new Downloader(new HttpClient(server), cache) { Attempts = 3, RetryDelay = TimeSpan.FromMilliseconds(10) };
            var data = DatasetRows.FromUrl("https://data.example/rows.jsonl", downloader: downloader, headers: new Dictionary<string, string> { ["Authorization"] = "Bearer secret" });
            Check(data.Count() == 100 && data.Count() == 100 && server.Requests == 1, $"cached ({server.Requests} requests)");
            Check(server.LastAuthorization == "Bearer secret", "headers sent");
            Check(File.Exists(Path.Combine(cache, "urls", "data.example", "rows.jsonl")), "plain URLs under urls/<host>/<path>");

            // Resume: a partial file continues with a Range request.
            server.Files["https://data.example/big.jsonl"] = Encoding.UTF8.GetBytes(string.Concat(Enumerable.Range(0, 1000).Select(i => $"{{\"i\":{i}}}\n")));
            server.FailAfterBytes = 3000;
            var resumed = DatasetRows.FromUrl("https://data.example/big.jsonl", downloader: downloader);
            Check(resumed.Count() == 1000 && server.RangeRequests >= 1, $"resumed ({server.RangeRequests} range requests)");

            server.FailAfterBytes = null;
            server.Files["https://data.example/flaky.jsonl"] = Encoding.UTF8.GetBytes("{\"ok\":1}\n");
            server.Errors = 2;
            Check(DatasetRows.FromUrl("https://data.example/flaky.jsonl", downloader: downloader).Count() == 1, "retried after server errors");

            try
            {
                _ = DatasetRows.FromUrl("https://data.example/private.jsonl", downloader: downloader).Count();
                Check(false, "missing file should fail");
            }
            catch (HttpRequestException ex)
            {
                Check(ex.StatusCode == HttpStatusCode.NotFound && ex.Message.Contains("private without a token", StringComparison.Ordinal), ex.Message);
            }
        }
        finally
        {
            if (Directory.Exists(cache))
            {
                Directory.Delete(cache, true);
            }
        }
    }

    private sealed class FakeFileServer : HttpMessageHandler
    {
        public Dictionary<string, byte[]> Files { get; } = [];

        public int Requests { get; private set; }

        public int RangeRequests { get; private set; }

        public string? LastAuthorization { get; private set; }

        public int? FailAfterBytes { get; set; }

        public int Errors { get; set; }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Requests++;
            LastAuthorization = request.Headers.Authorization?.ToString();
            if (Errors > 0)
            {
                Errors--;
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.ServiceUnavailable) { Content = new StringContent("busy") });
            }

            string url = request.RequestUri!.ToString();
            if (!Files.TryGetValue(url, out var body))
            {
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound) { Content = new StringContent("{\"error\":\"not found\"}") });
            }

            long from = request.Headers.Range?.Ranges.First().From ?? 0;
            RangeRequests += from > 0 ? 1 : 0;
            var slice = body.AsSpan((int)from).ToArray();
            Stream content = new MemoryStream(slice);
            if (FailAfterBytes is { } limit && from == 0)
            {
                content = new BreakingStream(slice, limit);
            }

            var response = new HttpResponseMessage(from > 0 ? HttpStatusCode.PartialContent : HttpStatusCode.OK) { Content = new StreamContent(content) };
            response.Content.Headers.ContentLength = slice.Length;
            return Task.FromResult(response);
        }
    }

    // Delivers some bytes, then fails as a dropped connection does.
    private sealed class BreakingStream(byte[] data, int limit) : MemoryStream(data)
    {
        public override int Read(byte[] buffer, int offset, int count)
        {
            if (Position >= limit)
            {
                throw new IOException("connection reset");
            }

            return base.Read(buffer, offset, (int)Math.Min(count, limit - Position));
        }

        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            if (Position >= limit)
            {
                throw new IOException("connection reset");
            }

            return base.ReadAsync(buffer[..(int)Math.Min(buffer.Length, limit - Position)], cancellationToken);
        }
    }
}

// A consumer's normalizer for the tests: lower case, one space between words.
internal sealed class CaseAndSpacing : ITextNormalizer
{
    public string Normalize(string text) => string.Join(' ', text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)).ToLowerInvariant();
}
