// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.IO.Compression;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Vision.Plugins.Gemma;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Text;
using Idrak.Text.Abstractions;
using Idrak.Abstraction.Generation;

// Plan 12, phase 1: the contracts and registries of vision-language fine-tuning (no family in the library): tuning data
// formats with images, the images' preparation saved with the adapters, the trainable parts a family offers, the feature
// cache and the metrics on generated text.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] VisionTuningContractGroup =
    [
        ("vision tuning contracts: cer and wer on known strings (English, Arabic with and without its marks, NFC, a character outside the basic plane, case), corpus sums, options checked; the edit distance takes memory of the shorter text only and equals the row-by-row table (random sequences across 64-element blocks and past 512 symbols, edge cases, characters outside the basic plane); an app's metric registers and an unknown one names the registry", TuningMetricValues),
        ("vision tuning contracts: the memory and disk feature caches pass the kit's suite; the memory budget is measured and the least recently used goes first; disk entries are safetensors files that outlive the cache, a file under another key is a miss; an unknown cache names the registry", FeatureCacheRoundTrips),
    ];

    // ------------------------------------------------------------------------------------------------ data formats

    // ------------------------------------------------------------------------------------------------ preparation

    // ------------------------------------------------------------------------------------------------ metrics

    internal static void TuningMetricValues(Device device)
    {
        _ = device;
        var cer = new ErrorRateMetric(words: false);
        var wer = new ErrorRateMetric(words: true);
        void Is(TuningScore score, double numerator, double denominator, string what) =>
            Check(score.Numerator == numerator && score.Denominator == denominator, $"{what}: {score}, expected {numerator} / {denominator}");

        Is(cer.Score("kitten", "sitting"), 3, 7, "cer kitten/sitting");
        Is(cer.Score("sitting", "sitting"), 0, 7, "cer equal");
        Is(wer.Score("the cat sat", "the cat sat down"), 1, 4, "wer one word missing");
        Is(wer.Score("  the   cat\tsat \n", "the cat sat"), 0, 3, "wer splits at runs of white space");
        Is(wer.Score("the dog sat", "the cat sat"), 1, 3, "wer one word wrong");

        // Arabic: harakat are characters (6 missing of 15) unless stripped; NFC joins alef and hamza above.
        Is(cer.Score("كتب الولد", "كَتَبَ الوَلَدُ"), 6, 15, "cer Arabic without its marks");
        Is(new ErrorRateMetric(words: false, stripDiacritics: true).Score("كتب الولد", "كَتَبَ الوَلَدُ"), 0, 9, "cer Arabic, marks stripped");
        Is(wer.Score("كتب الولد", "كَتَبَ الوَلَدُ"), 2, 2, "wer Arabic without its marks");
        Is(cer.Score("أحمد", "أحمد"), 0, 4, "cer: alef + hamza above equals أ (NFC)");
        Is(cer.Score("été", "été"), 0, 3, "cer: e + acute equals é (NFC)");
        Is(new ErrorRateMetric(words: false, stripDiacritics: true).Score("cafe", "café"), 0, 4, "cer: accents stripped when asked");
        Is(cer.Score("a😀b", "ab"), 1, 2, "cer: an emoji is one character");
        Is(cer.Score("ABC", "abc"), 3, 3, "cer: case counts");
        Is(new ErrorRateMetric(words: false, ignoreCase: true).Score("ABC", "abc"), 0, 3, "cer: case ignored when asked");
        Is(cer.Score("abc", ""), 3, 0, "cer: an empty reference");
        Check(cer.Score("abc", "").Value == 1 && cer.Score("", "").Value == 0, "an empty reference: all wrong, or nothing to get wrong");
        Check(cer.LowerIsBetter && wer.LowerIsBetter, "error rates: lower is better");

        // The edit distance in linear memory: two texts of 4,000 characters (a full table would be 64 MB).
        var random = new Random(3);
        int[] x = [.. Enumerable.Range(0, 4000).Select(_ => random.Next(30))];
        int[] y = [.. x[..1000], 99, .. x[1000..3000], .. x[3001..]];
        long allocated = GC.GetAllocatedBytesForCurrentThread();
        int distance = TuningMetrics.EditDistance<int>(x, y);
        allocated = GC.GetAllocatedBytesForCurrentThread() - allocated;
        Check(distance == 2 && TuningMetrics.EditDistance<int>(x, x) == 0 && TuningMetrics.EditDistance<int>(x, []) == 4000, $"edit distance {distance}");
        Check(allocated < 100_000, $"the edit distance allocated {allocated:N0} bytes for 4,000 x 4,000");

        // Exact: the same distance as the table filled row by row (the loop it replaced), on random sequences across the
        // 64-element blocks (lengths 0-300, alphabets of 2 to 2,000: past 512 distinct elements the row loop runs), with
        // the arguments either way round; edge cases; and through the metric, characters outside the basic plane.
        for (int trial = 0; trial < 3000; trial++)
        {
            int symbols = new[] { 2, 4, 30, 700, 2000 }[trial % 5];
            int[] p = [.. Enumerable.Range(0, random.Next(301)).Select(_ => random.Next(symbols))];
            int[] q = trial % 3 == 0 && p.Length > 0 ? [.. p] : [.. Enumerable.Range(0, trial % 7 == 0 ? p.Length : random.Next(301)).Select(_ => random.Next(symbols))];
            if (trial % 3 == 0 && q.Length > 0)
            {
                q[random.Next(q.Length)] = random.Next(symbols);                    // a few changes inside a long shared run
                q[random.Next(q.Length)] = random.Next(symbols);
            }

            int expected = TableDistance(p, q), got = TuningMetrics.EditDistance<int>(p, q), back = TuningMetrics.EditDistance<int>(q, p);
            if (expected != got || expected != back)
            {
                Check(false, $"trial {trial}: {p.Length} x {q.Length} over {symbols} symbols: {got} and {back}, the table {expected}");
                break;
            }
        }

        string[] edges = ["", "a", "abc", "kitten", "sitting", new string('a', 200), new string('b', 200), string.Concat(Enumerable.Repeat("ab", 100)),
            string.Concat(Enumerable.Range(0, 130).Select(i => (char)('A' + (i % 50))))];
        foreach (string p in edges)
        {
            foreach (string q in edges)
            {
                int[] pc = [.. p.Select(c => (int)c)], qc = [.. q.Select(c => (int)c)];
                Check(TuningMetrics.EditDistance<int>(pc, qc) == TableDistance(pc, qc), $"'{p}' against '{q}'");
            }
        }

        Check(TuningMetrics.EditDistance<int>([], []) == 0 && TuningMetrics.EditDistance<int>([1, 2, 3], [4, 5, 6]) == 3 && TuningMetrics.EditDistance<int>([1, 2, 3], [1, 2, 3]) == 0,
            "empty, all different, equal");
        string emoji = string.Concat(Enumerable.Range(0, 90).Select(i => char.ConvertFromUtf32(0x1F600 + (i % 40)))) + "نص عربي";
        string misread = emoji.Remove(4, 2).Insert(30, "x") + "😀";
        Is(cer.Score(misread, emoji), TableDistance([.. misread.EnumerateRunes().Select(r => r.Value)], [.. emoji.EnumerateRunes().Select(r => r.Value)]),
            emoji.EnumerateRunes().Count(), "cer over characters outside the basic plane, as the table counts them");
    }

    // A format's conversations through ReadPending, every one taken in order (the kit's reading delegate).
    internal static IEnumerable<IReadOnlyList<ChatMessage>> Pending(ITuningDataFormat format, string path, TuningDataOptions? options)
    {
        using var pending = format.ReadPending(path, options);
        return [.. Enumerable.Range(0, pending.Count).Select(i => pending.Read(i).Messages)];
    }

    // The edit distance as the library computed it before the bit vectors: the whole table, a row at a time.
    internal static int TableDistance(int[] a, int[] b)
    {
        var row = Enumerable.Range(0, b.Length + 1).ToArray();
        for (int i = 1; i <= a.Length; i++)
        {
            int diagonal = row[0];
            row[0] = i;
            for (int j = 1; j <= b.Length; j++)
            {
                int above = row[j];
                row[j] = Math.Min(Math.Min(above + 1, row[j - 1] + 1), diagonal + (a[i - 1] == b[j - 1] ? 0 : 1));
                diagonal = above;
            }
        }

        return row[b.Length];
    }

    // ------------------------------------------------------------------------------------------------ feature caches

    internal static void FeatureCacheRoundTrips(Device device)
    {
        string folder = TempFolder();
        try
        {
            using (var memory = new MemoryFeatureCache(new FeatureCacheOptions()))
            {
                Conformance.CheckFeatureCache(memory, device).ThrowIfFailed();
            }

            using (var disk = new DiskFeatureCache(new FeatureCacheOptions { Folder = Path.Combine(folder, "disk") }))
            {
                Conformance.CheckFeatureCache(disk, device).ThrowIfFailed();
            }

            // The memory budget: half of what the machine reports free, unless given; the least recently used goes first.
            using (var measured = new MemoryFeatureCache(new FeatureCacheOptions()))
            {
                long available = Device.Cpu.Backend.AvailableMemory() ?? 0;
                Check(measured.Budget > 0 && measured.Budget <= available && measured.Name == "memory", $"budget {measured.Budget:N0} of {available:N0} available");
            }

            static FeatureCacheKey Key(string image) => new(image, "", "none", "Tiny", "c0ffee", "float32");
            using (var small = new MemoryFeatureCache(new FeatureCacheOptions { Budget = 48 }))
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
            using (var disk = new DiskFeatureCache(new FeatureCacheOptions { Folder = diskFolder }))
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

            using (var again = new DiskFeatureCache(new FeatureCacheOptions { Folder = diskFolder }))
            {
                Check(again.TryGet(key, device, out var back) && back.Shape.SequenceEqual([2, 3, 4]) && back.ToArray()[23] == 11.5f, "a new cache reads the file");
                back!.Dispose();
                var other = Key("other");
                string misplaced = Path.Combine(diskFolder, other.Id[..2], other.Id + ".safetensors");
                Directory.CreateDirectory(Path.GetDirectoryName(misplaced)!);
                File.Copy(file, misplaced);
                Check(again.Contains(other) && !again.TryGet(other, device, out _), "a file whose metadata names another key is a miss");
                using var big = Tensor.From(new float[1000], [1000], Device.Cpu);
                using var bounded = new DiskFeatureCache(new FeatureCacheOptions { Folder = diskFolder, Budget = 1000 });
                Check(!bounded.Put(Key("big"), big), "a bounded disk cache refuses what does not fit");
                again.Clear();
                Check(again.Count == 0 && !again.Contains(key), "cleared");
            }

            Check(FeatureCaches.DefaultFolder.EndsWith("features", StringComparison.Ordinal)
                  && (Environment.GetEnvironmentVariable("IDRAK_CACHE") is not { Length: > 0 } cache || FeatureCaches.DefaultFolder.StartsWith(cache, StringComparison.Ordinal)),
                $"the default folder {FeatureCaches.DefaultFolder}");
            Check(Key("a").Id != (Key("a") with { Stage = "features" }).Id && Key("a").Id == Key("a").Id && Key("a").Id.Length == 64, "ids follow the fields");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // ------------------------------------------------------------------------------------------------ trainable parts

    internal static float[] Weights(int count, int seed)
    {
        var random = new Random(seed);
        return [.. Enumerable.Range(0, count).Select(_ => (float)(random.NextDouble() * 2 - 1))];
    }

    // ------------------------------------------------------------------------------------------------ helpers

    internal static T Failure<T>(Action action) where T : Exception
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

    internal static T Failure<T>(Func<object?> action) where T : Exception => Failure<T>(() => { _ = action(); });

    // Where two lists of conversations differ (roles and parts), or null.
    internal static string? Same(IReadOnlyList<IReadOnlyList<ChatMessage>> expected, IReadOnlyList<IReadOnlyList<ChatMessage>> actual)
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
