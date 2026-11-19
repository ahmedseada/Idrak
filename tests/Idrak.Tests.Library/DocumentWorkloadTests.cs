// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Data.Pdf;

// Plan 28, phase D13: any workload. The limits (DocumentOptions) measured from the machine's memory and settable, and memory
// bounded at any size, on the large set tools/documents/make_fixtures.py --large writes beside the samples
// (test-output/documents/large/d13: 3,000 pages, 1,000 small files, one huge image, deep forms): a large file read page
// by page within its cache, written and drawn with one page's buffers, many files read in parallel within
// ComputeResources and checked to pay (rule 79).
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] DocumentWorkloadGroup =
    [
        ("documents workload: every limit defaults from the memory measured when a document opens, within ComputeResources.CpuMemoryLimit (a stream a quarter, the cache an eighth, an object a KiB; the nesting a count of levels), each set by its option or its variable, the option winning; a bad variable refused naming it; a limit set low trips naming its option and variable; the large set's deep forms read within the default nesting and refused under a lower one", WorkloadLimits),
        ("documents workload: a 3,000-page file read page by page (text and images) stays within its cache's bound and one page: the managed memory sampled after every page grows no more than the cache and one page's buffers, under the measured cache and a small one, every page's text as the generator hashed it", WorkloadReadLarge),
        ("documents workload: a 3,000-page file written back page by page (its images and text) holds one page of buffers; its pages drawn one by one hold one page's pixels; the huge image's page drawn without the image twice", WorkloadWriteDrawLarge),
        ("documents workload: 1,000 small files, and a 3,000-page file's pages, read in parallel within ComputeResources (no more at once than MaxCpuThreads, the same text as one by one, the managed memory bounded), checked to pay against one by one (rule 79)", WorkloadParallel),
    ];

    // A member of the large set (beside the samples' folder, as make_fixtures.py --large writes it): missing, the group fails naming the command.
    private static string WorkloadLarge(string name)
    {
        string samples = Environment.GetEnvironmentVariable("IDRAK_DOCUMENTS_SAMPLES") is { Length: > 0 } set ? set : RepositoryPath("test-output", "documents", "samples");
        string path = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(samples).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar))!, "large", "d13", name);
        Check(File.Exists(path) || Directory.Exists(path),
            $"{path} is not there. Make the large set with: python -I tools/documents/make_fixtures.py -o test-output/documents/samples --large");
        return path;
    }

    private static JsonObject WorkloadChecks(string name) =>
        (JsonNode.Parse(File.ReadAllText(Path.ChangeExtension(WorkloadLarge(name), ".json")))!["checks"] as JsonObject)!;

    private static PdfDocument WorkloadOpen(string path, DocumentOptions options, out FileStream stream)
    {
        stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1, FileOptions.RandomAccess);
        return PdfDocument.Open(stream, Path.GetFileName(path), options);
    }

    private static string WorkloadHash(string text) => Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(text)));

    private static void WorkloadLimits(Device device)
    {
        CpuOnly(device);
        long? before = ComputeResources.CpuMemoryLimit;
        try
        {
            // Within a memory limit (less than this machine has free), each size limit is its share of it: measured, not fixed.
            foreach (long limit in new[] { 64L << 20, 192L << 20 })
            {
                ComputeResources.CpuMemoryLimit = limit;
                var measured = new DocumentOptions();
                Check(measured.MaxStreamBytes == limit / 4 && measured.CacheBytes == limit / 8 && measured.MaxObjects == limit / 1024 && measured.MaxDepth == DocumentOptions.DefaultMaxDepth,
                    $"within {limit:N0} bytes: a stream {measured.MaxStreamBytes:N0}, the cache {measured.CacheBytes:N0}, {measured.MaxObjects:N0} objects, nesting {measured.MaxDepth}");
            }

            // Without one, from what the machine reports free when the limits are read (between two readings around them).
            ComputeResources.CpuMemoryLimit = null;
            long low = DocumentSettings.FreeMemory();
            var free = new DocumentOptions();
            long stream = free.MaxStreamBytes, cache = free.CacheBytes, objects = free.MaxObjects;
            long high = DocumentSettings.FreeMemory();
            (low, high) = (Math.Min(low, high), Math.Max(low, high));
            long slack = 64L << 20;   // the free memory moves while the test runs
            Check(stream >= Math.Min(Array.MaxLength, (low - slack) / 4) && stream <= Math.Min(Array.MaxLength, (high + slack) / 4)
                  && cache >= (low - slack) / 8 && cache <= (high + slack) / 8 && objects >= (low - slack) / 1024 && objects <= (high + slack) / 1024,
                $"from the free memory ({low:N0} to {high:N0} bytes): a stream {stream:N0}, the cache {cache:N0}, {objects:N0} objects");
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"    measured here: {high / 1048576.0:F0} MiB free: a stream {stream / 1048576.0:F0} MiB, a document's cache {cache / 1048576.0:F0} MiB, {objects:N0} objects, nesting {free.MaxDepth}"));
        }
        finally
        {
            ComputeResources.CpuMemoryLimit = before;
        }

        // Each settable by its variable, the option winning; a bad value refused naming the variable.
        WithEnvironment(new Dictionary<string, string?>
        {
            [DocumentOptions.MaxStreamBytesVariable] = "3m", [DocumentOptions.CacheBytesVariable] = "512k", [DocumentOptions.MaxObjectsVariable] = "5000",
            [DocumentOptions.MaxDepthVariable] = "12",
        }, () =>
        {
            var set = new DocumentOptions();
            Check(set.MaxStreamBytes == 3L << 20 && set.CacheBytes == 512L << 10 && set.MaxObjects == 5000 && set.MaxDepth == 12,
                $"the variables: {set.MaxStreamBytes}, {set.CacheBytes}, {set.MaxObjects}, {set.MaxDepth}");
            var given = new DocumentOptions { MaxStreamBytes = 1 << 20, CacheBytes = 0, MaxObjects = 9, MaxDepth = 30 };
            Check(given.MaxStreamBytes == 1 << 20 && given.CacheBytes == 0 && given.MaxObjects == 9 && given.MaxDepth == 30,
                $"the options win over the variables (a cache of 0 keeps nothing): {given.MaxStreamBytes}, {given.CacheBytes}, {given.MaxObjects}, {given.MaxDepth}");
        });
        WithEnvironment(new Dictionary<string, string?> { [DocumentOptions.CacheBytesVariable] = "0" },
            () => Check(new DocumentOptions().CacheBytes == 0, $"{DocumentOptions.CacheBytesVariable}=0: no cache"));
        foreach (var (variable, value) in new[]
                 {
                     (DocumentOptions.MaxObjectsVariable, "many"), (DocumentOptions.MaxObjectsVariable, "0"), (DocumentOptions.MaxDepthVariable, "-3"),
                     (DocumentOptions.MaxDepthVariable, (DocumentOptions.MostNesting + 1L).ToString(CultureInfo.InvariantCulture)), (DocumentOptions.CacheBytesVariable, "lots"),
                 })
        {
            WithEnvironment(new Dictionary<string, string?> { [variable] = value },
                () => ExpectRefusal<ArgumentException>(() => _ = new DocumentOptions(), variable, $"{variable}={value}"));
        }

        ExpectRefusal<ArgumentOutOfRangeException>(() => _ = new DocumentOptions { MaxDepth = DocumentOptions.MostNesting + 1 }, "MaxDepth", "a nesting limit past the most");

        // A limit set low trips naming its option and its variable: the 3,000-page file has more objects than 1,000.
        string pages = WorkloadLarge("pages-3000.pdf");
        WithEnvironment(new Dictionary<string, string?> { [DocumentOptions.MaxObjectsVariable] = "1000" }, () => ExpectRefusal<InvalidDataException>(
            () => DocumentFormats.Open(pages).Dispose(), DocumentOptions.MaxObjectsVariable, "the object limit from its variable"));

        // The deep forms: 8, 16 and 24 levels within the default nesting; a lower one stops the 24 levels deep page, naming it.
        string deep = WorkloadLarge("deep-forms.pdf");
        var checks = WorkloadChecks("deep-forms.pdf");
        using (var document = DocumentFormats.Open(deep))
        {
            foreach (var entry in checks["textPages"]!.AsArray())
            {
                int index = (int)entry![0]!;
                string text = document.ReadPage(index, DocumentParts.Text).Text;
                Check(text == (string)entry[1]!, $"deep-forms page {index}: '{text}', not '{entry[1]}'");
            }
        }

        WithEnvironment(new Dictionary<string, string?> { [DocumentOptions.MaxDepthVariable] = "20" }, () =>
        {
            using var document = DocumentFormats.Open(deep);
            Check(document.ReadPage(1, DocumentParts.Text).Text == "Nested 16 forms deep", "16 levels within a nesting limit of 20");
            ExpectRefusal<InvalidDataException>(() => document.ReadPage(2, DocumentParts.Text), "MaxDepth", "24 levels past a nesting limit of 20");
        });
    }

    private static void WorkloadReadLarge(Device device)
    {
        CpuOnly(device);
        string path = WorkloadLarge("pages-3000.pdf");
        var expected = WorkloadChecks("pages-3000.pdf")["pageTextSha256"]!.AsArray().Select(n => (string)n!).ToArray();

        // A run: every page read with its text and images listed, its text checked; the managed memory after each page
        // (garbage included) and after a full collection every 250 pages (what is live), its growth from page 20 to the end.
        (double Milliseconds, long Growth, long Live, long Heap, long Cached, long Limit) Read(DocumentOptions options)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            using var document = WorkloadOpen(path, options, out var stream);
            using (stream)
            {
                long settled = 0, live = 0, heap = 0, measuring = 0, start = Stopwatch.GetTimestamp();
                Check(document.PageCount == expected.Length, $"{document.PageCount} pages");
                for (int p = 0; p < document.PageCount; p++)
                {
                    var page = document.ReadPage(p, DocumentParts.Text | DocumentParts.Images);
                    Check(WorkloadHash(page.Text) == expected[p], $"page {p}'s text: {page.Text}");
                    heap = Math.Max(heap, GC.GetTotalMemory(forceFullCollection: false));
                    if (p == 20 || p % 250 == 249)
                    {
                        long at = Stopwatch.GetTimestamp();
                        long now = GC.GetTotalMemory(forceFullCollection: true);
                        measuring += Stopwatch.GetTimestamp() - at;
                        live = Math.Max(live, now);
                        settled = p == 20 ? now : settled;
                    }
                }

                double milliseconds = Stopwatch.GetElapsedTime(start).TotalMilliseconds - Stopwatch.GetElapsedTime(0, measuring).TotalMilliseconds;
                long growth = GC.GetTotalMemory(forceFullCollection: true) - settled;
                return (milliseconds, growth, live, heap, document.File.CachedBytes, options.CacheBytes);
            }
        }

        const long OnePage = 4L << 20;   // a page's buffers, and what the pools keep between pages
        foreach (var (name, options) in new[] { ("the measured cache", new DocumentOptions()), ("a 2 MiB cache", new DocumentOptions { CacheBytes = 2L << 20 }), ("no cache", new DocumentOptions { CacheBytes = 0 }) })
        {
            var run = Read(options);
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"    3,000 pages read with {name} ({run.Limit / 1048576.0:F0} MiB): {run.Milliseconds:F0} ms ({run.Milliseconds * 1000 / expected.Length:F0} µs a page); the cache holds {run.Cached / 1048576.0:F2} MiB; "
                + $"live managed memory at most {run.Live / 1048576.0:F1} MiB, with its garbage {run.Heap / 1048576.0:F1} MiB; {run.Growth / 1048576.0:F2} MiB more after page 3,000 than after page 20"));
            Check(run.Cached <= run.Limit, $"{name}: the cache holds {run.Cached:N0} bytes, past its {run.Limit:N0}");
            Check(run.Growth <= Math.Min(run.Limit, run.Cached + OnePage) + OnePage,
                $"{name}: the managed memory grew {run.Growth:N0} bytes from page 20 to 3,000, more than the cache's {run.Cached:N0} (of {run.Limit:N0}) and one page");
        }
    }

    private static void WorkloadWriteDrawLarge(Device device)
    {
        CpuOnly(device);
        string path = WorkloadLarge("pages-3000.pdf");
        var face = SubsetFace();
        var reading = new DocumentOptions { CacheBytes = 2L << 20 };

        // Written back: each page's images (decoded, stored by the options' rule) and its text in a test face, one page after
        // another into a stream that counts; the managed memory sampled per page, its growth from page 20 to 3,000 bounded.
        GC.Collect();
        GC.WaitForPendingFinalizers();
        long settled = 0, heap = 0, written, start = Stopwatch.GetTimestamp();
        long allocated = GC.GetTotalAllocatedBytes(precise: true);
        int pages;
        using (var source = WorkloadOpen(path, reading, out var stream))
        using (stream)
        {
            pages = source.PageCount;
            using var counter = new PdfByteCounter();
            using (var builder = DocumentWriters.Create(counter, "pdf", new DocumentWriteOptions()))
            {
                var glyphs = new DocumentGlyph[256];
                for (int p = 0; p < pages; p++)
                {
                    var page = source.ReadPage(p, DocumentParts.Text | DocumentParts.Images);
                    var target = builder.AddPage(page.Width, page.Height);
                    foreach (var image in page.Images)
                    {
                        target.Image(image.Decode(), image.Box, DocumentImageEncoding.Auto);
                    }

                    foreach (var run in page.Runs)
                    {
                        float scale = run.FontSize / face.Metrics.UnitsPerEm, x = run.Box.X, y = run.Baseline ?? run.Box.Bottom;
                        int count = 0;
                        foreach (char c in run.Text.AsSpan(0, Math.Min(run.Text.Length, glyphs.Length)))
                        {
                            int glyph = face.Glyph(c);
                            glyphs[count++] = new DocumentGlyph(glyph, x, y, face.Advance(glyph) * scale);
                            x += face.Advance(glyph) * scale;
                        }

                        if (count > 0)
                        {
                            target.Glyphs(face, run.FontSize, glyphs.AsSpan(0, count), run.Text[..count], DocumentTextMode.Fill, DocumentColour.Black);
                        }
                    }

                    heap = Math.Max(heap, GC.GetTotalMemory(forceFullCollection: false));
                    if (p == 20)
                    {
                        settled = GC.GetTotalMemory(forceFullCollection: true);
                    }
                }

                builder.Finish();
            }

            written = counter.Length;
        }

        double writeMs = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        allocated = GC.GetTotalAllocatedBytes(precise: true) - allocated;
        long growth = GC.GetTotalMemory(forceFullCollection: true) - settled;
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    3,000 pages written back (images and text): {writeMs:F0} ms ({writeMs * 1000 / pages:F0} µs a page), {written / 1048576.0:F1} MiB written, {allocated / pages / 1024.0:F0} KiB allocated a page; "
            + $"the heap with its garbage at most {heap / 1048576.0:F1} MiB, {growth / 1048576.0:F2} MiB more after page 3,000 than after page 20 (the source's document closed)"));
        Check(growth < 4L << 20, $"writing: the managed memory grew {growth:N0} bytes from page 20 to 3,000 (one page's buffers, not every page's)");

        // Drawn: every fifth page at 72 dpi, one after another; what stays from page to page is not the pages drawn.
        using (var source = WorkloadOpen(path, reading, out var stream))
        using (stream)
        {
            var options = new DocumentDrawOptions { Dpi = 72 };
            long drawSettled = 0, drawHeap = 0, drawStart = Stopwatch.GetTimestamp();
            int drawn = 0;
            for (int p = 0; p < source.PageCount; p += 5, drawn++)
            {
                var drawing = source.DrawPage(p, options);
                Check(drawing.Image.Width == 612 && drawing.Image.Height == 792, $"page {p} drawn {drawing.Image.Width} x {drawing.Image.Height}");
                drawHeap = Math.Max(drawHeap, GC.GetTotalMemory(forceFullCollection: false));
                if (drawn == 20)
                {
                    drawSettled = GC.GetTotalMemory(forceFullCollection: true);
                }
            }

            double drawMs = Stopwatch.GetElapsedTime(drawStart).TotalMilliseconds;
            long drawGrowth = GC.GetTotalMemory(forceFullCollection: true) - drawSettled;
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"    {drawn} pages drawn at 72 dpi: {drawMs:F0} ms ({drawMs / drawn:F1} ms a page); the heap with its garbage at most {drawHeap / 1048576.0:F1} MiB, {drawGrowth / 1048576.0:F2} MiB more after the last page than after page 20's"));
            Check(drawGrowth < 4L << 20, $"drawing: the managed memory grew {drawGrowth:N0} bytes from the 20th page drawn to the last");
        }

        // The huge image's page: the image decoded once (its 8-bit planes) and the page's pixels, nothing more than bands besides.
        string huge = WorkloadLarge("huge-image.pdf");
        var expectedImage = WorkloadChecks("huge-image.pdf")["image"]!;
        long decoded = (long)expectedImage["decodedBytes"]!;
        using (var document = DocumentFormats.Open(huge))
        {
            var options = new DocumentDrawOptions { Dpi = 36 };
            GC.Collect();
            GC.WaitForPendingFinalizers();
            long before = GC.GetTotalAllocatedBytes(precise: true), at = Stopwatch.GetTimestamp();
            var drawing = document.DrawPage(0, options);
            double ms = Stopwatch.GetElapsedTime(at).TotalMilliseconds;
            long used = GC.GetTotalAllocatedBytes(precise: true) - before;
            long pixels = (long)drawing.Image.Width * drawing.Image.Height * drawing.Image.Channels;
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"    the huge image's page (an A0 page over a {(int)expectedImage["pixelWidth"]!} x {(int)expectedImage["pixelHeight"]!} JPEG, {decoded / 1048576.0:F0} MiB decoded) drawn at 36 dpi ({drawing.Image.Width} x {drawing.Image.Height}): {ms:F0} ms, "
                + $"{used / 1048576.0:F0} MiB allocated: {(used - decoded - pixels) / 1048576.0:F1} MiB besides the decoded image and the page's pixels"));
            Check(used < decoded + pixels + Math.Max(64L << 20, decoded / 4), $"the huge page allocated {used:N0} bytes: the image ({decoded:N0} bytes) twice?");
        }
    }

    private static void WorkloadParallel(Device device)
    {
        CpuOnly(device);
        string folder = WorkloadLarge("many-small");
        var expected = (JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "many-small.json")))!["checks"]!["files"]!.AsArray())
            .Select(f => (File: Path.Combine(folder, (string)f!["file"]!), Text: (string)f["text"]!)).ToArray();
        int threads = ComputeResources.MaxCpuThreads;

        // One file read: opened, its page's text; the most read at once counted.
        int active = 0, most = 0;
        string ReadOne(string file)
        {
            int now = Interlocked.Increment(ref active);
            InterlockedMax(ref most, now);
            try
            {
                using var document = DocumentFormats.Open(file);
                return document.ReadPage(0, DocumentParts.Text).Text;
            }
            finally
            {
                Interlocked.Decrement(ref active);
            }
        }

        static void InterlockedMax(ref int target, int value)
        {
            int seen;
            while ((seen = Volatile.Read(ref target)) < value && Interlocked.CompareExchange(ref target, value, seen) != seen)
            {
            }
        }

        // Each way timed twice (the first warms the files and the code), the second kept.
        (double Ms, string[] Texts, long Heap) Pass(bool parallel)
        {
            var texts = new string[expected.Length];
            long heap = 0;
            double ms = 0;
            for (int round = 0; round < 2; round++)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                long baseline = GC.GetTotalMemory(forceFullCollection: true), start = Stopwatch.GetTimestamp();
                (active, most) = (0, 0);
                if (parallel)
                {
                    Parallel.For(0, expected.Length, ComputeResources.ParallelOptions, i => texts[i] = ReadOne(expected[i].File));
                }
                else
                {
                    for (int i = 0; i < expected.Length; i++)
                    {
                        texts[i] = ReadOne(expected[i].File);
                    }
                }

                ms = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                heap = GC.GetTotalMemory(forceFullCollection: false) - baseline;
            }

            return (ms, texts, heap);
        }

        var one = Pass(parallel: false);
        Check(most == 1, $"one by one: {most} at once");
        var many = Pass(parallel: true);
        int mostAtOnce = most;
        for (int i = 0; i < expected.Length; i++)
        {
            Check(one.Texts[i] == expected[i].Text && many.Texts[i] == one.Texts[i], $"{Path.GetFileName(expected[i].File)}: '{many.Texts[i]}', '{one.Texts[i]}', not '{expected[i].Text}'");
        }

        Check(mostAtOnce <= threads, $"{mostAtOnce} files read at once, more than ComputeResources.MaxCpuThreads ({threads})");
        int before = ComputeResources.MaxCpuThreads;
        try
        {
            ComputeResources.MaxCpuThreads = Math.Min(2, Environment.ProcessorCount);
            var two = Pass(parallel: true);
            Check(most <= ComputeResources.MaxCpuThreads && two.Texts.SequenceEqual(one.Texts), $"within {ComputeResources.MaxCpuThreads} threads: {most} at once");
        }
        finally
        {
            ComputeResources.MaxCpuThreads = before;
        }

        // A 3,000-page file's pages over the same threads: one document read from several threads at once.
        string pagesFile = WorkloadLarge("pages-3000.pdf");
        var expectedPages = WorkloadChecks("pages-3000.pdf")["pageTextSha256"]!.AsArray().Select(n => (string)n!).ToArray();
        (double Ms, string[] Hashes) Pages(bool parallel)
        {
            var hashes = new string[expectedPages.Length];
            double ms = 0;
            for (int round = 0; round < 2; round++)
            {
                using var document = DocumentFormats.Open(pagesFile);
                long start = Stopwatch.GetTimestamp();
                if (parallel)
                {
                    Parallel.For(0, document.PageCount, ComputeResources.ParallelOptions, p => hashes[p] = WorkloadHash(document.ReadPage(p, DocumentParts.Text).Text));
                }
                else
                {
                    for (int p = 0; p < document.PageCount; p++)
                    {
                        hashes[p] = WorkloadHash(document.ReadPage(p, DocumentParts.Text).Text);
                    }
                }

                ms = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            }

            return (ms, hashes);
        }

        var pagesOne = Pages(parallel: false);
        var pagesMany = Pages(parallel: true);
        Check(pagesOne.Hashes.SequenceEqual(expectedPages) && pagesMany.Hashes.SequenceEqual(expectedPages), "the pages' text read in parallel is the generator's");
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    {expected.Length:N0} small files on {threads} threads: one by one {one.Ms:F0} ms, in parallel {many.Ms:F0} ms ({one.Ms / many.Ms:F2}x, {mostAtOnce} at once), "
            + $"the managed memory after {one.Heap / 1048576.0:F1} and {many.Heap / 1048576.0:F1} MiB; 3,000 pages of one file: one by one {pagesOne.Ms:F0} ms, in parallel {pagesMany.Ms:F0} ms ({pagesOne.Ms / pagesMany.Ms:F2}x)"));
        Check(many.Heap < (64L << 20) * Math.Max(1, threads), $"many files at once left {many.Heap:N0} bytes of managed memory");
        if (Environment.ProcessorCount >= 2 && threads >= 2)
        {
            Check(many.Ms < one.Ms, $"reading many files in parallel did not pay: {many.Ms:F0} ms against {one.Ms:F0} ms one by one");
            Check(pagesMany.Ms < pagesOne.Ms, $"reading a file's pages in parallel did not pay: {pagesMany.Ms:F0} ms against {pagesOne.Ms:F0} ms one by one");
        }
    }
}
