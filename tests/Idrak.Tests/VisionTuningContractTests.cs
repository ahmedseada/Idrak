// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.IO.Compression;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Gemma3Vision;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Nlp;
using Idrak.Nlp.Abstractions;

// Plan 12, phase 1: the contracts and registries of vision-language fine-tuning (no family in the library): tuning data
// formats with images, the images' preparation saved with the adapters, the trainable parts a family offers, the feature
// cache and the metrics on generated text.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] VisionTuningContractGroup =
    [
        ("vision tuning contracts: sharegpt reads LlamaFactory's JSON array and JSON Lines, each <image> the next image in place, paths from the data file's folder, an images folder or a zip read in place (images only, nothing unpacked); a marker count that disagrees names the record; the kit's data-format suite passes", ShareGptWithImages),
        ("vision tuning contracts: messages reads chat JSON Lines with image parts (a path, a data URL, base64, bare parts and <image> markers taking the record's images; text-only records as before); formats are detected, an unknown one names the registry; the kit's data-format suite passes", MessagesWithImages),
        ("vision tuning contracts: TuningImages parses the command line's transforms, options and grayscale, checks options against a family's keys, saves tuning_images.json and reads it back equal; a bad file is refused naming it", TuningImagesRoundTrip),
        ("vision tuning contracts: cer and wer on known strings (English, Arabic with and without its marks, NFC, a character outside the basic plane, case), corpus sums, options checked; the edit distance takes memory of the shorter text only; an app's metric registers and an unknown one names the registry", TuningMetricValues),
        ("vision tuning contracts: the memory and disk feature caches pass the kit's suite; the memory budget is measured and the least recently used goes first; disk entries are safetensors files that outlive the cache, a file under another key is a miss; an unknown cache names the registry", FeatureCacheRoundTrips),
        ("vision tuning contracts: the tiny Gemma 3 (plug-in) offers its projector and tower: features from the tower's output equal the encoder's, gradients reach the projector (finite differences), the tower and the pixels; bfloat16 weights and another family's encoder are refused", VisionTuningPartGemma3),
        ("vision tuning contracts: the tiny LLaVA (outside plug-in) offers its projector only (features equal the encoder's, gradients by finite differences); asking for its tower fails naming the parts offered; a family without IVisionTuningPart trains the language model only", VisionTuningPartLlava),
    ];

    // ------------------------------------------------------------------------------------------------ data formats

    private static void ShareGptWithImages(Device device)
    {
        _ = device;
        string folder = TempFolder();
        try
        {
            string images = Path.Combine(folder, "images");
            Directory.CreateDirectory(images);
            File.Copy(TestData("vlm/image.png"), Path.Combine(images, "a.png"));
            File.Copy(TestData("vlm/image.jpg"), Path.Combine(images, "b.jpg"));
            var a = ChatImage.FromFile(Path.Combine(images, "a.png"));
            var b = ChatImage.FromFile(Path.Combine(images, "b.jpg"));

            JsonObject Record(string first, string answer, string[] paths, string? system = null)
            {
                var record = new JsonObject
                {
                    ["conversations"] = new JsonArray(new JsonObject { ["from"] = "human", ["value"] = first }, new JsonObject { ["from"] = "gpt", ["value"] = answer }),
                    ["images"] = new JsonArray([.. paths.Select(p => (JsonNode)p)]),
                };
                if (system is not null)
                {
                    record["system"] = system;
                }

                return record;
            }

            JsonObject[] records =
            [
                Record("<image>Extract details to JSON.", "{\"رقم\": 12}", ["images/a.png"]),
                Record("Compare <image> with <image> please", "ok", ["images/b.jpg", "images/a.png"], system: "Read the scan."),
                Record("مرحبا", "أهلا", []),
            ];
            IReadOnlyList<ChatMessage>[] expected =
            [
                [new("user", [a, new ChatText("Extract details to JSON.")]), new("assistant", "{\"رقم\": 12}")],
                [new("system", "Read the scan."), new("user", [new ChatText("Compare "), b, new ChatText(" with "), a, new ChatText(" please")]), new("assistant", "ok")],
                [new("user", "مرحبا"), new("assistant", "أهلا")],
            ];
            string array = Path.Combine(folder, "train.json"), lines = Path.Combine(folder, "train.jsonl");
            File.WriteAllText(array, new JsonArray([.. records.Select(r => (JsonNode)r.DeepClone())]).ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
            File.WriteAllLines(lines, records.Select(r => r.ToJsonString()));

            var format = TuningDataFormats.Get(TuningDataFormats.ShareGpt);
            Check(TuningDataFormats.Detect(array).Name == "sharegpt" && TuningDataFormats.Detect(lines).Name == "sharegpt", "sharegpt detected in both layouts");
            foreach (string file in new[] { array, lines })
            {
                var read = format.Read(file).ToList();
                Check(Same(expected, read.Select(t => t.Messages).ToList()) is null, $"{Path.GetFileName(file)}: {Same(expected, read.Select(t => t.Messages).ToList())}");
            }

            // Images under another folder (--images), the data elsewhere.
            string data = Path.Combine(folder, "data");
            Directory.CreateDirectory(data);
            string elsewhere = Path.Combine(data, "train.json");
            File.WriteAllText(elsewhere, new JsonArray(Record("<image>Read.", "done", ["a.png"])).ToJsonString());
            var fromFolder = format.Read(elsewhere, new TuningDataOptions { Images = images }).Single();
            Check(fromFolder.Messages[0].Parts[0] is ChatImage got && got.Equals(a), "an image relative to the images folder");
            var missing = Failure<FileNotFoundException>(() => format.Read(elsewhere).ToList());
            Check(missing.Message.Contains("record 1", StringComparison.Ordinal) && missing.Message.Contains("a.png", StringComparison.Ordinal), $"a missing image names the record: {missing.Message}");

            // A zip as the images root: only image entries are read, in place.
            string zip = Path.Combine(folder, "downloaded_images.zip");
            using (var archive = ZipFile.Open(zip, ZipArchiveMode.Create))
            {
                archive.CreateEntryFromFile(Path.Combine(images, "a.png"), "downloaded_images/a.png");
                archive.CreateEntryFromFile(Path.Combine(images, "b.jpg"), "downloaded_images/b.jpg");
                using var pickle = new StreamWriter(archive.CreateEntry("downloaded_images/model.pkl").Open());
                pickle.Write("not to be opened");
            }

            string zipped = Path.Combine(data, "zipped.jsonl");
            File.WriteAllLines(zipped, [Record("<image> and <image>", "two", ["downloaded_images/a.png", "b.jpg"]).ToJsonString()]);
            var before = Directory.GetFileSystemEntries(folder, "*", SearchOption.AllDirectories).Order().ToList();
            var fromZip = format.Read(zipped, new TuningDataOptions { Images = zip }).Single();
            Check(fromZip.Messages[0].Parts is [ChatImage first, ChatText { Text: " and " }, ChatImage second] && first.Equals(a) && second.Equals(b),
                $"images from the zip, by full name and under its one top folder: {string.Join(", ", fromZip.Messages[0].Parts)}");
            Check(Directory.GetFileSystemEntries(folder, "*", SearchOption.AllDirectories).Order().SequenceEqual(before), "nothing unpacked to disk");
            // Paths from the machine the data was made on: found by their longest trailing part naming one entry or file.
            string pages = Path.Combine(folder, "pages.zip");
            using (var archive = ZipFile.Open(pages, ZipArchiveMode.Create))
            {
                archive.CreateEntryFromFile(Path.Combine(images, "a.png"), "downloaded_images/pdf_images/0012/page_013.png");
                archive.CreateEntryFromFile(Path.Combine(images, "b.jpg"), "downloaded_images/pdf_images/0013/page_013.png");
            }

            string foreign = Path.Combine(data, "foreign.jsonl");
            File.WriteAllLines(foreign, [Record("<image> <image>", "two", ["/workspace/pdf_images/0012/page_013.png", "C:\\work\\pdf_images\\0013\\page_013.png"]).ToJsonString()]);
            var fromPages = format.Read(foreign, new TuningDataOptions { Images = pages }).Single();
            Check(fromPages.Messages[0].Parts is [ChatImage p1, ChatText, ChatImage p2] && p1.Equals(a) && p2.Equals(b), $"other machines' paths in a zip: {string.Join(", ", fromPages.Messages[0].Parts)}");
            string folderRoot = Path.Combine(folder, "unzipped");
            Directory.CreateDirectory(Path.Combine(folderRoot, "pdf_images", "0012"));
            File.Copy(Path.Combine(images, "a.png"), Path.Combine(folderRoot, "pdf_images", "0012", "page_013.png"));
            string foreignOne = Path.Combine(data, "foreign-one.jsonl");
            File.WriteAllLines(foreignOne, [Record("<image>", "one", ["/workspace/pdf_images/0012/page_013.png"]).ToJsonString()]);
            Check(format.Read(foreignOne, new TuningDataOptions { Images = folderRoot }).Single().Messages[0].Parts[0] is ChatImage f1 && f1.Equals(a), "another machine's path under a folder");
            string vague = Path.Combine(data, "vague.jsonl");
            File.WriteAllLines(vague, [Record("<image>", "x", ["/elsewhere/page_013.png"]).ToJsonString()]);
            var several = Failure<FileNotFoundException>(() => format.Read(vague, new TuningDataOptions { Images = pages }).ToList());
            Check(several.Message.Contains("'/page_013.png' (in several entries)", StringComparison.Ordinal), $"a name several entries end in finds none: {several.Message}");

            string pickled = Path.Combine(data, "pickled.jsonl");
            File.WriteAllLines(pickled, [Record("<image>", "x", ["downloaded_images/model.pkl"]).ToJsonString()]);
            var refused = Failure<InvalidDataException>(() => format.Read(pickled, new TuningDataOptions { Images = zip }).ToList());
            Check(refused.Message.Contains("line 1", StringComparison.Ordinal) && refused.Message.Contains("not an image", StringComparison.Ordinal), $"a non-image entry is refused unread: {refused.Message}");

            // The counts must agree, and the error names the record; reading is lazy (a bad record later does not stop the first).
            string mismatch = Path.Combine(folder, "mismatch.json");
            File.WriteAllText(mismatch, new JsonArray(records[0].DeepClone(), Record("<image><image> two pages", "x", ["images/a.png"]), records[2].DeepClone()).ToJsonString());
            var wrong = Failure<InvalidDataException>(() => format.Read(mismatch).ToList());
            Check(wrong.Message.Contains("mismatch.json, record 2", StringComparison.Ordinal) && wrong.Message.Contains("2 <image> markers for 1 entry", StringComparison.Ordinal),
                $"the mismatch names the record: {wrong.Message}");
            Check(format.Read(mismatch).First().Messages[0].Parts[0] is ChatImage, "the first record reads before the bad one is reached");
            string extra = Path.Combine(folder, "extra.jsonl");
            File.WriteAllLines(extra, [new JsonObject { ["id"] = "page-7", ["conversations"] = new JsonArray(new JsonObject { ["from"] = "human", ["value"] = "no marker" }), ["images"] = new JsonArray("images/a.png") }.ToJsonString()]);
            var unused = Failure<InvalidDataException>(() => format.Read(extra).ToList());
            Check(unused.Message.Contains("line 1 (id page-7)", StringComparison.Ordinal) && unused.Message.Contains("0 <image> markers for 1 entry", StringComparison.Ordinal), $"an unused image: {unused.Message}");
            Check(Failure<DirectoryNotFoundException>(() => format.Read(array, new TuningDataOptions { Images = Path.Combine(folder, "nowhere") })).Message.Contains("nowhere", StringComparison.Ordinal),
                "a missing images root is refused when the reading starts");

            // The testing kit's suite: the expected transcripts, both layouts, and the refused file.
            var options = new TuningDataOptions();
            Conformance.CheckTuningDataFormat(path => format.Read(path, options).Select(t => t.Messages),
            [
                new TuningDataSample(array, expected),
                new TuningDataSample(lines, expected),
                new TuningDataSample(mismatch, Error: "markers"),
                new TuningDataSample(zipped.Replace("zipped", "pickled", StringComparison.Ordinal), Error: "not found"),
            ]).ThrowIfFailed();
            Console.WriteLine($"    sharegpt: {records.Length} records as a JSON array and JSON Lines, 2 images from a zip, mismatches named");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void MessagesWithImages(Device device)
    {
        _ = device;
        string folder = TempFolder();
        try
        {
            File.Copy(TestData("vlm/image.png"), Path.Combine(folder, "a.png"));
            File.Copy(TestData("vlm/image.jpg"), Path.Combine(folder, "b.jpg"));
            var a = ChatImage.FromFile(Path.Combine(folder, "a.png"));
            var b = ChatImage.FromFile(Path.Combine(folder, "b.jpg"));
            string[] lines =
            [
                """{"messages": [{"role": "user", "content": [{"type": "image", "path": "a.png"}, {"type": "text", "text": "Read it."}]}, {"role": "assistant", "content": "done"}]}""",
                """{"messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "and <image>"}]}, {"role": "assistant", "content": "two"}], "images": ["a.png", "b.jpg"]}""",
                $$$"""{"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "{{{a.ToDataUrl()}}}"}}, {"type": "image", "media_type": "image/jpeg", "data": "{{{Convert.ToBase64String(b.Data.Span)}}}"}]}, {"role": "assistant", "content": "x"}]}""",
                """{"messages": [{"role": "user", "content": "hi <image> as text"}, {"role": "assistant", "content": "hello"}], "tools": []}""",
            ];
            string file = Path.Combine(folder, "train.jsonl");
            File.WriteAllLines(file, lines);
            IReadOnlyList<ChatMessage>[] expected =
            [
                [new("user", [a, new ChatText("Read it.")]), new("assistant", "done")],
                [new("user", [a, new ChatText("and "), b]), new("assistant", "two")],
                [new("user", [a, b]), new("assistant", "x")],
                [new("user", "hi <image> as text"), new("assistant", "hello")],
            ];
            var format = TuningDataFormats.Get("Messages");                     // names ignore case
            var read = format.Read(file).ToList();
            Check(Same(expected, read.Select(t => t.Messages).ToList()) is null, $"messages: {Same(expected, read.Select(t => t.Messages).ToList())}");
            var plain = ChatTranscript.FromJson(JsonNode.Parse(lines[3])!.AsObject());
            Check(read[3].Messages.SequenceEqual(plain.Messages), "a text-only record reads as ChatTranscript.FromJson reads it");
            Check(TuningDataFormats.Detect(file).Name == "messages" && TuningDataFormats.Read(file).Count() == 4, "detected and read through the registry");

            string bare = Path.Combine(folder, "bare.jsonl");
            File.WriteAllLines(bare, [lines[0], """{"messages": [{"role": "user", "content": [{"type": "image"}]}]}"""]);
            var error = Failure<InvalidDataException>(() => format.Read(bare).ToList());
            Check(error.Message.Contains("bare.jsonl, line 2", StringComparison.Ordinal) && error.Message.Contains("no \"images\" list", StringComparison.Ordinal), $"a bare image part without images: {error.Message}");
            string counted = Path.Combine(folder, "counted.jsonl");
            File.WriteAllLines(counted, ["""{"messages": [{"role": "user", "content": "<image><image>"}], "images": ["a.png"]}"""]);
            Check(Failure<InvalidDataException>(() => format.Read(counted).ToList()).Message.Contains("2 image placeholders", StringComparison.Ordinal), "the count of placeholders is checked");
            string unknown = Path.Combine(folder, "unknown.jsonl");
            File.WriteAllLines(unknown, ["""{"prompt": "x", "completion": "y"}"""]);
            Check(Failure<InvalidDataException>(() => TuningDataFormats.Detect(unknown)).Message.Contains("messages, sharegpt", StringComparison.Ordinal), "an unknown record names the formats");
            var notRegistered = Failure<NotSupportedException>(() => TuningDataFormats.Get("alpaca"));
            Check(notRegistered.Message.Contains("'alpaca' is not registered", StringComparison.Ordinal) && notRegistered.Message.Contains("TuningDataFormats.Register", StringComparison.Ordinal), notRegistered.Message);
            Check(TuningDataFormats.Origin("sharegpt") == Overrides.Library && TuningDataFormats.Describe().Contains("sharegpt:", StringComparison.Ordinal), "library defaults");

            Conformance.CheckTuningDataFormat(path => format.Read(path).Select(t => t.Messages),
                [new TuningDataSample(file, expected), new TuningDataSample(bare, Error: "\"images\"")]).ThrowIfFailed();
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // ------------------------------------------------------------------------------------------------ preparation

    private static void TuningImagesRoundTrip(Device device)
    {
        _ = device;
        string folder = TempFolder();
        try
        {
            var images = TuningImages.Parse("max_width=1024,contrast=1.5", ["do_pan_and_scan=true", "pan_and_scan_max_num_crops=3"], grayscale: true);
            Check(images.Pipeline.ToString() == "grayscale,max_width=1024,contrast=1.5" && images.Transforms.ToString() == "max_width=1024,contrast=1.5",
                $"grayscale goes first: {images.Pipeline}");
            Check(TuningImages.Parse("grayscale,contrast=1.5", grayscale: true).Pipeline.ToString() == "grayscale,contrast=1.5", "grayscale is not added twice");
            images.ThrowIfUnknown("Gemma3ForConditionalGeneration", Gemma3PanAndScan.Keys);
            var unknown = Failure<ArgumentException>(() => TuningImages.Parse(null, ["crops=4"]).ThrowIfUnknown("Gemma3ForConditionalGeneration", Gemma3PanAndScan.Keys));
            Check(unknown.Message.Contains("'crops'", StringComparison.Ordinal) && unknown.Message.Contains("do_pan_and_scan", StringComparison.Ordinal), $"an unknown option names the family's keys: {unknown.Message}");
            Check(Failure<ArgumentException>(() => TuningImages.Parse("blur=2")).Message.Contains("registered", StringComparison.Ordinal), "an unknown transform names the registered ones");

            Check(TuningImages.Read(folder) is null && !TuningImages.Exists(folder), "no file, no preparation");
            images.Save(folder);
            Check(File.Exists(Path.Combine(folder, "tuning_images.json")), "saved as tuning_images.json");
            var back = TuningImages.Read(folder)!;
            Check(back.Equals(images) && back.GetHashCode() == images.GetHashCode() && back.VisionOptions.Integer("pan_and_scan_max_num_crops", 0) == 3 && back.Grayscale,
                $"read back equal: {back}");
            TuningImages.None.Save(folder);
            Check(TuningImages.Read(folder)!.IsEmpty && TuningImages.Read(folder)!.Equals(TuningImages.None), "nothing round-trips as nothing");
            Check(!images.Equals(images with { Grayscale = false }) && !images.Equals(images with { VisionOptions = VisionOptions.Empty }), "equality sees every field");

            var image = ImageCodecs.Decode(TestData("vlm/image.png"));
            var applied = images.Apply(image);
            var piped = images.Pipeline.Apply(image);
            Check(applied.Channels == piped.Channels && applied.Pixels.AsSpan().SequenceEqual(piped.Pixels), "Apply runs the pipeline, grayscale first");

            File.WriteAllText(Path.Combine(folder, "tuning_images.json"), """{"format": "idrak-tuning-images/9"}""");
            var bad = Failure<InvalidDataException>(() => TuningImages.Read(folder));
            Check(bad.Message.Contains("tuning_images.json", StringComparison.Ordinal) && bad.Message.Contains("idrak-tuning-images/1", StringComparison.Ordinal), $"another format is refused: {bad.Message}");
            File.WriteAllText(Path.Combine(folder, "tuning_images.json"), """{"image_transforms": "blur=2"}""");
            Check(Failure<InvalidDataException>(() => TuningImages.Read(folder)).Message.Contains("blur", StringComparison.Ordinal), "an unknown transform in the file is refused");

            // The feature cache's key follows the preparation.
            var chat = ChatImage.FromFile(TestData("vlm/image.png"));
            var key = FeatureCacheKey.For(chat, images, "Gemma3ForConditionalGeneration", "abc", "float32");
            Check(key.Transforms == "grayscale,max_width=1024,contrast=1.5" && key.Image == chat.Hash && key.Stage == "tower", $"key {key}");
            Check(key.Id != FeatureCacheKey.For(chat, images with { Grayscale = false }, "Gemma3ForConditionalGeneration", "abc", "float32").Id, "grayscale changes the key");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // ------------------------------------------------------------------------------------------------ metrics

    private static void TuningMetricValues(Device device)
    {
        _ = device;
        var cer = TuningMetrics.Get("CER");
        var wer = TuningMetrics.Get(TuningMetrics.WordErrorRate);
        void Is(TuningScore score, double numerator, double denominator, string what) =>
            Check(score.Numerator == numerator && score.Denominator == denominator, $"{what}: {score}, expected {numerator} / {denominator}");

        Is(cer.Score("kitten", "sitting"), 3, 7, "cer kitten/sitting");
        Is(cer.Score("sitting", "sitting"), 0, 7, "cer equal");
        Is(wer.Score("the cat sat", "the cat sat down"), 1, 4, "wer one word missing");
        Is(wer.Score("  the   cat\tsat \n", "the cat sat"), 0, 3, "wer splits at runs of white space");
        Is(wer.Score("the dog sat", "the cat sat"), 1, 3, "wer one word wrong");

        // Arabic: harakat are characters (6 missing of 15) unless stripped; NFC joins alef and hamza above.
        Is(cer.Score("كتب الولد", "كَتَبَ الوَلَدُ"), 6, 15, "cer Arabic without its marks");
        Is(cer.Score("كتب الولد", "كَتَبَ الوَلَدُ", new Dictionary<string, string> { ["strip_diacritics"] = "true" }), 0, 9, "cer Arabic, marks stripped");
        Is(wer.Score("كتب الولد", "كَتَبَ الوَلَدُ"), 2, 2, "wer Arabic without its marks");
        Is(cer.Score("أحمد", "أحمد"), 0, 4, "cer: alef + hamza above equals أ (NFC)");
        Is(cer.Score("été", "été"), 0, 3, "cer: e + acute equals é (NFC)");
        Is(cer.Score("cafe", "café", new Dictionary<string, string> { ["strip_diacritics"] = "yes" }), 0, 4, "cer: accents stripped when asked");
        Is(cer.Score("a😀b", "ab"), 1, 2, "cer: an emoji is one character");
        Is(cer.Score("ABC", "abc"), 3, 3, "cer: case counts");
        Is(cer.Score("ABC", "abc", new Dictionary<string, string> { ["ignore_case"] = "true" }), 0, 3, "cer: case ignored when asked");
        Is(cer.Score("abc", ""), 3, 0, "cer: an empty reference");
        Check(cer.Score("abc", "").Value == 1 && cer.Score("", "").Value == 0, "an empty reference: all wrong, or nothing to get wrong");
        Is(TuningMetrics.Score("cer", [("kitten", "sitting"), ("abc", "abc")]), 3, 10, "the corpus: errors over every reference's characters");
        Check(Math.Abs(TuningMetrics.Score("cer", [("kitten", "sitting"), ("abc", "abc")]).Value - 0.3) < 1e-12 && cer.LowerIsBetter, "the corpus value");
        var option = Failure<ArgumentException>(() => cer.Score("a", "a", new Dictionary<string, string> { ["normalize"] = "nfkc" }));
        Check(option.Message.Contains("strip_diacritics", StringComparison.Ordinal) && option.Message.Contains("'normalize'", StringComparison.Ordinal), $"an unknown option names the keys: {option.Message}");
        Check(Failure<ArgumentException>(() => cer.Score("a", "a", new Dictionary<string, string> { ["ignore_case"] = "maybe" })).Message.Contains("true or false", StringComparison.Ordinal), "a bad flag");

        // The edit distance in linear memory: two texts of 4,000 characters (a full table would be 64 MB).
        var random = new Random(3);
        int[] x = [.. Enumerable.Range(0, 4000).Select(_ => random.Next(30))];
        int[] y = [.. x[..1000], 99, .. x[1000..3000], .. x[3001..]];
        long allocated = GC.GetAllocatedBytesForCurrentThread();
        int distance = TuningMetrics.EditDistance<int>(x, y);
        allocated = GC.GetAllocatedBytesForCurrentThread() - allocated;
        Check(distance == 2 && TuningMetrics.EditDistance<int>(x, x) == 0 && TuningMetrics.EditDistance<int>(x, []) == 4000, $"edit distance {distance}");
        Check(allocated < 100_000, $"the edit distance allocated {allocated:N0} bytes for 4,000 x 4,000");

        // An app's metric, and an unknown name.
        var mine = new ExactMetric();
        TuningMetrics.Register(mine);
        try
        {
            Check(TuningMetrics.Get("exact").Score("a", "a").Value == 1 && TuningMetrics.Origin("exact") == typeof(ExactMetric).Assembly.GetName().Name, "an app's metric");
        }
        finally
        {
            Check(TuningMetrics.Unregister("exact") && TuningMetrics.Find("exact") is null, "unregistered");
        }

        var notRegistered = Failure<NotSupportedException>(() => TuningMetrics.Get("bleu"));
        Check(notRegistered.Message.Contains("'bleu' is not registered", StringComparison.Ordinal) && notRegistered.Message.Contains("cer, wer", StringComparison.Ordinal), notRegistered.Message);
    }

    private sealed class ExactMetric : ITuningMetric
    {
        public string Name => "exact";

        public string Summary => "1 when the answer is the reference";

        public bool LowerIsBetter => false;

        public IReadOnlyCollection<string> Keys => [];

        public TuningScore Score(string answer, string reference, IReadOnlyDictionary<string, string>? options = null) => new(answer == reference ? 1 : 0, 1);
    }

    // ------------------------------------------------------------------------------------------------ feature caches

    private static FeatureCacheUnderTest Wrap(IFeatureCache cache)
    {
        static FeatureCacheKey Key(FeatureKeyFields f) => new(f.Image, f.Transforms, f.VisionOptions, f.Family, f.Checkpoint, f.DType, f.Stage);
        return new FeatureCacheUnderTest((k, v) => cache.Put(Key(k), v), (k, d) => cache.TryGet(Key(k), d, out var v) ? v : null, cache.Clear, cache);
    }

    private static void FeatureCacheRoundTrips(Device device)
    {
        string folder = TempFolder();
        try
        {
            using (var memory = Wrap(FeatureCaches.Create("memory")))
            {
                Conformance.CheckFeatureCache(memory, device).ThrowIfFailed();
            }

            using (var disk = Wrap(FeatureCaches.Create(FeatureCaches.Disk, new FeatureCacheOptions { Folder = Path.Combine(folder, "disk") })))
            {
                Conformance.CheckFeatureCache(disk, device).ThrowIfFailed();
            }

            // The memory budget: half of what the machine reports free, unless given; the least recently used goes first.
            using (var measured = (MemoryFeatureCache)FeatureCaches.Create("memory"))
            {
                long available = Device.Cpu.Backend.AvailableMemory() ?? 0;
                Check(measured.Budget > 0 && measured.Budget <= available && measured.Name == "memory", $"budget {measured.Budget:N0} of {available:N0} available");
            }

            static FeatureCacheKey Key(string image) => new(image, "", "none", "Tiny", "c0ffee", "float32");
            using (var small = FeatureCaches.Create("memory", new FeatureCacheOptions { Budget = 48 }))
            {
                foreach (string k in new[] { "1", "2", "3" })
                {
                    using var value = Tensor.From([1f, 2f, 3f, float.Parse(k, System.Globalization.CultureInfo.InvariantCulture)], [4], Device.Cpu);
                    Check(small.Put(Key(k), value), $"put {k}");
                }

                Check(small.TryGet(Key("1"), Device.Cpu, out var one) && one.ToArray()[3] == 1f, "1 is kept");
                one!.Dispose();
                using (var fourth = Tensor.From([4f, 4f, 4f, 4f], [4], Device.Cpu))
                {
                    small.Put(Key("4"), fourth);
                }

                Check(small.Contains(Key("1")) && !small.Contains(Key("2")) && small.Contains(Key("3")) && small.Contains(Key("4")) && small.Count == 3 && small.Bytes == 48,
                    "the least recently used (2) was evicted");
                using var large = Tensor.From(new float[13], [13], Device.Cpu);
                Check(!small.Put(Key("5"), large) && !small.Contains(Key("5")) && small.Count == 3, "a value over the budget is not kept");
            }

            // Disk: files under the folder, read by a new cache; a file under another key's name is a miss.
            string diskFolder = Path.Combine(folder, "persist");
            var key = Key("persisted");
            using (var disk = FeatureCaches.Create("disk", new FeatureCacheOptions { Folder = diskFolder }))
            using (var value = Tensor.From([.. Enumerable.Range(0, 24).Select(i => i * 0.5f)], [2, 3, 4], Device.Cpu))
            {
                Check(disk.Put(key, value) && disk.Count == 1 && disk.Bytes > 96, $"disk put: {disk.Count} files, {disk.Bytes} bytes");
            }

            string file = Path.Combine(diskFolder, key.Id[..2], key.Id + ".safetensors");
            Check(File.Exists(file), $"the entry is {file}");
            using (var reader = SafeTensorsReader.Open(file))
            {
                Check(reader.Tensors["features"].Shape.SequenceEqual([2, 3, 4]) && reader.Metadata["key"] == key.ToJson().ToJsonString(), "a safetensors file with the key in its metadata");
            }

            using (var again = FeatureCaches.Create("disk", new FeatureCacheOptions { Folder = diskFolder }))
            {
                Check(again.TryGet(key, device, out var back) && back.Shape.SequenceEqual([2, 3, 4]) && back.ToArray()[23] == 11.5f, "a new cache reads the file");
                back!.Dispose();
                var other = Key("other");
                string misplaced = Path.Combine(diskFolder, other.Id[..2], other.Id + ".safetensors");
                Directory.CreateDirectory(Path.GetDirectoryName(misplaced)!);
                File.Copy(file, misplaced);
                Check(again.Contains(other) && !again.TryGet(other, device, out _), "a file whose metadata names another key is a miss");
                using var big = Tensor.From(new float[1000], [1000], Device.Cpu);
                using var bounded = FeatureCaches.Create("disk", new FeatureCacheOptions { Folder = diskFolder, Budget = 1000 });
                Check(!bounded.Put(Key("big"), big), "a bounded disk cache refuses what does not fit");
                again.Clear();
                Check(again.Count == 0 && !again.Contains(key), "cleared");
            }

            Check(FeatureCaches.DefaultFolder.EndsWith("features", StringComparison.Ordinal)
                  && (Environment.GetEnvironmentVariable("IDRAK_CACHE") is not { Length: > 0 } cache || FeatureCaches.DefaultFolder.StartsWith(cache, StringComparison.Ordinal)),
                $"the default folder {FeatureCaches.DefaultFolder}");
            var notRegistered = Failure<NotSupportedException>(() => FeatureCaches.Create("redis"));
            Check(notRegistered.Message.Contains("'redis' is not registered", StringComparison.Ordinal) && notRegistered.Message.Contains("memory, disk", StringComparison.Ordinal), notRegistered.Message);
            Check(Key("a").Id != (Key("a") with { Stage = "features" }).Id && Key("a").Id == Key("a").Id && Key("a").Id.Length == 64, "ids follow the fields");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // ------------------------------------------------------------------------------------------------ trainable parts

    private static void VisionTuningPartGemma3(Device device)
    {
        RegisterGemma3Vision();
        using var model = PretrainedModel.Load(TestData("vlm/tiny-gemma3"), new PretrainedOptions { Device = device });
        var vision = model.Vision!;
        Check(VisionTuningParts.Offered(vision).SequenceEqual(["projector", "tower"]), $"offered: {string.Join(", ", VisionTuningParts.Offered(vision))}");
        Check(VisionTuningParts.For(vision, []) is null && VisionTuningParts.For(vision, ["Projector", "tower"]) is { } both && ReferenceEquals(both, vision), "asked for nothing, or for what it offers");
        var tuning = VisionTuningParts.For(vision, ["projector"])!;
        var notOffered = Failure<NotSupportedException>(() => VisionTuningParts.For(vision, ["projector", "lm_head"]));
        Check(notOffered.Message.Contains("'lm_head'", StringComparison.Ordinal) && notOffered.Message.Contains("it offers: projector, tower", StringComparison.Ordinal), notOffered.Message);

        using var encoder = vision.CreateEncoder(new VisionEncoderOptions { Device = device, Weights = EncoderWeights.Float32 });
        var stages = (IVisionEncoderStages)encoder;
        var image = ImageCodecs.Decode(TestData("vlm/image.png"));
        CheckTrainableProjector(tuning, encoder, stages, image, device, "tiny Gemma 3");

        // The tower: its output with gradients equals the stage's, and gradients reach its weights and the pixels.
        using var pixels = stages.PixelValues(image);
        using var frozen = stages.Tower(pixels);
        var input = Tensor.From(pixels.ToArray(), pixels.Shape, device, requiresGrad: true);
        var towerParameters = tuning.Parameters(encoder, "tower");
        Check(towerParameters.Count > 4, $"{towerParameters.Count} tower parameters");
        foreach (var p in towerParameters)
        {
            p.RequiresGrad = true;
        }

        var tower = tuning.Tower(encoder, input);
        AssertClose(frozen.ToArray(), tower.ToArray(), 1e-5f, "the tower's output with gradients");
        var features = tuning.Features(encoder, tower);
        var weights = Tensor.From(Weights(features.Size, 5), features.Shape, device);
        (features * weights).Sum().Backward();
        Check(input.Grad is { } pixelGrad && pixelGrad.ToArray().Any(v => v != 0f) && pixelGrad.ToArray().All(float.IsFinite), "gradients reach the pixels");
        Check(towerParameters.Count(p => p.Grad is { } g && g.ToArray().Any(v => v != 0f)) >= towerParameters.Count / 2, "gradients reach the tower's weights");

        // Refusals: bfloat16 weights, another family's encoder, a part not offered.
        using (var packed = vision.CreateEncoder(new VisionEncoderOptions { Device = device, Weights = EncoderWeights.BFloat16 }))
        {
            Check(Failure<InvalidOperationException>(() => tuning.Parameters(packed, "projector")).Message.Contains("EncoderWeights.Float32", StringComparison.Ordinal), "bfloat16 weights do not train");
        }

        Check(Failure<NotSupportedException>(() => tuning.Parameters(encoder, "decoder")).Message.Contains("it offers: projector, tower", StringComparison.Ordinal), "a part not offered");
        Idrak.PluginTests.LlavaPlugin.Register();
        using var llava = PretrainedModel.Load(LlavaData("tiny-llava"), new PretrainedOptions { Device = device });
        using var otherEncoder = llava.Vision!.CreateEncoder(new VisionEncoderOptions { Device = device });
        Check(Failure<ArgumentException>(() => tuning.Features(otherEncoder, frozen)).Message.Contains("not an encoder of this Gemma 3 vision part", StringComparison.Ordinal), "another family's encoder");
    }

    private static void VisionTuningPartLlava(Device device)
    {
        Idrak.PluginTests.LlavaPlugin.Register();
        var image = ImageCodecs.Decode(TestData("vlm/image.png"));
        foreach (string folder in new[] { "tiny-llava", "tiny-llava-full" })
        {
            using var model = PretrainedModel.Load(LlavaData(folder), new PretrainedOptions { Device = device });
            var vision = model.Vision!;
            Check(VisionTuningParts.Offered(vision).SequenceEqual(["projector"]), $"{folder}: offered {string.Join(", ", VisionTuningParts.Offered(vision))}");
            var tower = Failure<NotSupportedException>(() => VisionTuningParts.For(vision, ["tower"]));
            Check(tower.Message.Contains("LlavaForConditionalGeneration does not offer 'tower'", StringComparison.Ordinal) && tower.Message.Contains("it offers: projector", StringComparison.Ordinal),
                $"{folder}: {tower.Message}");
            var tuning = VisionTuningParts.For(vision, ["projector"])!;
            using var encoder = vision.CreateEncoder(new VisionEncoderOptions { Device = device });
            CheckTrainableProjector(tuning, encoder, (IVisionEncoderStages)encoder, image, device, folder);
            using var pixels = ((IVisionEncoderStages)encoder).PixelValues(image);
            Check(Failure<NotSupportedException>(() => tuning.Tower(encoder, pixels)).Message.Contains("it offers: projector", StringComparison.Ordinal), $"{folder}: no tower forward");
            Check(Failure<NotSupportedException>(() => tuning.Parameters(encoder, "tower")).Message.Contains("it offers: projector", StringComparison.Ordinal), $"{folder}: no tower parameters");
        }

        // A family without IVisionTuningPart trains the language model only.
        var plain = new PlainVision();
        Check(VisionTuningParts.Offered(plain).Count == 0 && VisionTuningParts.For(plain, []) is null, "a plain family offers nothing");
        var none = Failure<NotSupportedException>(() => VisionTuningParts.For(plain, ["projector"]));
        Check(none.Message.Contains("PlainVisionForTest offers no vision parts to train (it trains the language model only)", StringComparison.Ordinal), none.Message);
    }

    // The projector trains: features from the tower's output equal the encoder's stage; gradients reach every projector
    // parameter and agree with central finite differences on the largest one.
    private static void CheckTrainableProjector(IVisionTuningPart tuning, IVisionEncoder encoder, IVisionEncoderStages stages, ImageData image, Device device, string what)
    {
        using var pixels = stages.PixelValues(image);
        using var expected = stages.Features(pixels);
        using var towerOutput = stages.Tower(pixels);
        var parameters = tuning.Parameters(encoder, "projector");
        Check(parameters.Count >= 2, $"{what}: {parameters.Count} projector parameters");
        foreach (var p in parameters)
        {
            p.ZeroGrad();
            p.RequiresGrad = true;
        }

        var features = tuning.Features(encoder, towerOutput);
        AssertClose(expected.ToArray(), features.ToArray(), 1e-5f, $"{what}: features from the tower's output");
        var weights = Tensor.From(Weights(features.Size, 11), features.Shape, device);
        var loss = (features * weights).Sum();
        loss.Backward();
        Check(parameters.All(p => p.Grad is { } g && g.ToArray().All(float.IsFinite) && g.ToArray().Any(v => v != 0f)), $"{what}: every projector parameter has a gradient");

        float Loss()
        {
            using var noGrad = Autograd.NoGrad();
            using var f = tuning.Features(encoder, towerOutput);
            using var product = f * weights;
            using var sum = product.Sum();
            return sum.ToArray()[0];
        }

        foreach (var p in parameters.Take(2))
        {
            float[] values = p.ToArray(), gradient = p.Grad!.ToArray();
            int at = Enumerable.Range(0, gradient.Length).MaxBy(i => MathF.Abs(gradient[i]));
            const float Step = 1e-2f;
            float original = values[at];
            values[at] = original + Step;
            p.Load(values);
            float up = Loss();
            values[at] = original - Step;
            p.Load(values);
            float down = Loss();
            values[at] = original;
            p.Load(values);
            float numeric = (up - down) / (2 * Step);
            Check(MathF.Abs(numeric - gradient[at]) <= 2e-2f * MathF.Max(1f, MathF.Abs(gradient[at])),
                $"{what}: d loss / d {Tensor.FormatShape(p.Shape)}[{at}] is {gradient[at]}, finite differences {numeric}");
        }
    }

    private static float[] Weights(int count, int seed)
    {
        var random = new Random(seed);
        return [.. Enumerable.Range(0, count).Select(_ => (float)(random.NextDouble() * 2 - 1))];
    }

    // A vision family that does not implement IVisionTuningPart.
    private sealed class PlainVision : PretrainedVision
    {
        public override string Family => "PlainVisionForTest";

        public override int Width => 8;

        public override IImagePromptFormat PromptFormat => throw new NotSupportedException();

        public override IImageAttentionRule Attention => ImageAttentionRules.Get(ImageAttentionRules.Causal);

        public override IReadOnlyCollection<string> StoredTensors => [];

        public override IVisionEncoder CreateEncoder(VisionEncoderOptions? options = null) => throw new NotSupportedException();
    }

    // ------------------------------------------------------------------------------------------------ helpers

    private static T Failure<T>(Action action) where T : Exception
    {
        try
        {
            action();
        }
        catch (T e)
        {
            return e;
        }

        throw new InvalidOperationException($"expected {typeof(T).Name}");
    }

    private static T Failure<T>(Func<object?> action) where T : Exception => Failure<T>(() => { _ = action(); });

    // Where two lists of conversations differ (roles and parts), or null.
    private static string? Same(IReadOnlyList<IReadOnlyList<ChatMessage>> expected, IReadOnlyList<IReadOnlyList<ChatMessage>> actual)
    {
        if (expected.Count != actual.Count)
        {
            return $"{actual.Count} conversations, expected {expected.Count}";
        }

        for (int i = 0; i < expected.Count; i++)
        {
            if (expected[i].Count != actual[i].Count)
            {
                return $"conversation {i + 1}: {actual[i].Count} messages, expected {expected[i].Count}";
            }

            for (int j = 0; j < expected[i].Count; j++)
            {
                if (expected[i][j].Role != actual[i][j].Role || !expected[i][j].Parts.SequenceEqual(actual[i][j].Parts))
                {
                    return $"conversation {i + 1}, message {j + 1}: {actual[i][j].Role} [{string.Join(" | ", actual[i][j].Parts)}], expected {expected[i][j].Role} [{string.Join(" | ", expected[i][j].Parts)}]";
                }
            }
        }

        return null;
    }
}
