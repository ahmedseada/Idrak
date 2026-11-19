// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;

// Plan 28, phase D5 (with D13's real-files clause): the owner's own PDFs. Every *.pdf under the folder IDRAK_DOCUMENTS_REAL
// names (its subfolders too, in ordinal order of their paths; nothing is written into it) is opened, read page by page
// within the default limits (text through the lines and tagged layouts, images decoded, paths, annotations; fields,
// bookmarks, named destinations, page labels, attachments and signatures listed, each signature checked), drawn at the
// drawer's default DPI, and written back as a rewrite and as an incremental update that reads back the same; each step
// is timed cold and warm apart (rule 76). The summary goes to test-output/documents/real (summary.json, summary.txt).
// Without the variable every test is a SKIP. The test-only variables (read here alone, as IDRAK_DOCUMENTS_SAMPLES is):
//   IDRAK_DOCUMENTS_REAL            the folder of PDFs
//   IDRAK_DOCUMENTS_REAL_REFERENCE  a folder of a reference extractor's text (tools/documents/reference_text.py; default
//                                   test-output/documents/real-reference when it is there): NAME.reference.txt at the
//                                   PDF's path under the folder, or beside the PDF; the differences are listed, not failed
//   IDRAK_DOCUMENTS_REAL_SAVE       pages, files or both (comma-separated; all): the drawn pages (PNG) and the written-back
//                                   files kept under test-output/documents/real/drawn and .../written; none by default
//   IDRAK_DOCUMENTS_REAL_WARM       the warm passes a step takes, the median reported (3; 0 to 100, 0: cold alone)
// An encrypted file's password is the first line of NAME.password.txt beside it (no variable: a secret stays out of the
// environment, plan 28's D2 decision 2); one that needs a password and has none is listed as skipped.
// A file fails the group only for a crash (an exception the contract does not document), a page or part that cannot be
// read with no reason given, or a written-back file that does not read the same; each failure names the file and step.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] DocumentRealGroup =
    [
        ("documents real: every PDF in IDRAK_DOCUMENTS_REAL opens and every page reads within the default limits (text through the lines and tagged layouts, images decoded, paths, annotations; fields, bookmarks, destinations, labels, attachments, signatures listed and checked), a broken, hostile or locked file reporting why; differences from a reference extractor's text listed; each step timed cold and warm, the summary in test-output/documents/real", PdfRealRead),
        ("documents real: every page of every file drawn at the drawer's default DPI (text, images, paths, annotations), timed cold and warm; the pixels kept under test-output/documents/real/drawn only when IDRAK_DOCUMENTS_REAL_SAVE names pages", PdfRealDraw),
        ("documents real: every file written back as a rewrite and as an incremental update (its title changed) reads back to the same pages (sizes, turns), text, runs and images; the update keeps the source as its prefix and its signatures' digests; timed cold and warm", PdfRealWriteBack),
    ];

    private const string PdfRealVariable = "IDRAK_DOCUMENTS_REAL", PdfRealReferenceVariable = "IDRAK_DOCUMENTS_REAL_REFERENCE";
    private const string PdfRealSaveVariable = "IDRAK_DOCUMENTS_REAL_SAVE", PdfRealWarmVariable = "IDRAK_DOCUMENTS_REAL_WARM";
    private const DocumentParts PdfRealDrawnParts = DocumentParts.Text | DocumentParts.Images | DocumentParts.Paths | DocumentParts.Annotations;

    private static readonly Lock PdfRealGate = new();
    private static PdfRealRun? s_pdfRealRun;

    private static readonly JsonSerializerOptions PdfRealJson = new() { WriteIndented = true, Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

    // The run: the folder's files and what each step found, kept across the group's tests so the summary holds them all.
    private sealed class PdfRealRun(string key, string folder, string? reference, int warm, bool savePages, bool saveFiles, string output, List<PdfRealFile> files, DocumentOptions limits)
    {
        private long _written = Stopwatch.GetTimestamp();

        public string Key { get; } = key;

        public string Folder { get; } = folder;

        public string? Reference { get; } = reference;

        public int Warm { get; } = warm;

        public bool SavePages { get; } = savePages;

        public bool SaveFiles { get; } = saveFiles;

        public string Output { get; } = output;

        public List<PdfRealFile> Files { get; } = files;

        public DateTimeOffset Started { get; } = DateTimeOffset.UtcNow;

        public DocumentDrawOptions Draw { get; } = new() { Parts = PdfRealDrawnParts };

        // The limits every file is read within: C8's defaults, those measured from the free memory measured once when the run
        // starts (PdfRealLimits), so every file, pass and read-back reads under the same ones and the summary names them.
        public DocumentOptions Limits { get; } = limits;

        // How signatures are checked: nothing trusted (only the certificates an app gives are, plan 28's decision 16) and
        // nothing fetched, so a signature's digest, signature and changes are what the check proves here.
        public SignatureCheckOptions Trust { get; } = new() { Online = false };

        // How files are written back: the defaults.
        public DocumentWriteOptions WriteOptions { get; } = new();

        // The summary written again when a few seconds have passed since the last time (a long run can be followed in it).
        public void WriteSoon()
        {
            if (Stopwatch.GetElapsedTime(_written).TotalSeconds >= 5)
            {
                Write();
            }
        }

        public void Write()
        {
            Directory.CreateDirectory(Output);
            PdfRealWriteFile(Path.Combine(Output, "summary.json"), PdfRealSummaryJson(this).ToJsonString(PdfRealJson) + "\n");
            PdfRealWriteFile(Path.Combine(Output, "summary.txt"), PdfRealSummaryText(this));
            _written = Stopwatch.GetTimestamp();
        }
    }

    // A step's times: the cold pass (the file's first of the step in this run) and the warm passes' median, with the managed
    // bytes each allocated on every thread.
    private sealed record PdfRealTiming(double ColdMs, long ColdBytes, double? WarmMs, long? WarmBytes);

    // A reported failure: the step, the page (-1: none), and the exception's type and message.
    private sealed record PdfRealError(string Step, int Page, string Error);

    // A page as read for writing back: what both written files must read back to.
    private sealed record PdfRealPage(float Width, float Height, int Rotation, string Text, int Runs, int Images);

    private sealed class PdfRealFile(string fullPath, string relative, long bytes, string? password, DocumentOptions limits)
    {
        private readonly Dictionary<(int Page, string Kind, string Detail), int> _issues = [];

        public string FullPath { get; } = fullPath;

        // Its path under the folder, with / between folders.
        public string Name { get; } = relative.Replace('\\', '/');

        // Its path under the folder as the system writes it (the reference folder and the output's folders mirror it).
        public string Relative { get; } = relative;

        public long Bytes { get; } = bytes;

        public string? Password { get; } = password;

        public bool? Opened { get; set; }

        public int Pages { get; private set; }

        public string? Version { get; private set; }

        public bool Encrypted { get; private set; }

        public bool Repaired { get; private set; }

        public string? Skipped { get; private set; }

        public string? NotOpened { get; private set; }

        public OrderedDictionary<string, long> Counts { get; } = new(StringComparer.Ordinal);

        public SortedDictionary<string, long> AnnotationKinds { get; } = new(StringComparer.Ordinal);

        public SortedDictionary<string, long> FieldKinds { get; } = new(StringComparer.Ordinal);

        public SortedDictionary<string, long> DrawIssues { get; } = new(StringComparer.Ordinal);

        public List<PdfRealError> Errors { get; } = [];

        public List<string> Failures { get; } = [];

        public OrderedDictionary<string, PdfRealTiming> Steps { get; } = new(StringComparer.Ordinal);

        public List<JsonObject> Signatures { get; } = [];

        public OrderedDictionary<string, JsonObject> WrittenBack { get; } = new(StringComparer.Ordinal);

        // Each page's text through the lines layout (null: not read), for the reference.
        public string?[]? Texts { get; set; }

        public JsonObject? Reference { get; set; }

        // The run's pinned limits (PdfRealRun.Limits), its password and a layout.
        public DocumentOptions Options(string layout) => limits with { Password = Password, Layout = layout };

        public void Open(int pages, DocumentInfo info)
        {
            Opened = true;
            Pages = pages;
            Version = info.Version;
            Encrypted = info.Encrypted;
            Repaired = info.Repaired;
        }

        // It does not open: a password it needs (skipped), or the reason it gives (damaged, hostile, not supported).
        public void NotOpen(Exception e)
        {
            Opened = false;
            string why = PdfRealDescribe(e);
            if (string.IsNullOrWhiteSpace(e.Message))
            {
                Fail("open", $"it does not open and says nothing why ({e.GetType().Name})");
            }
            else if (e.Message.Contains("DocumentOptions.Password", StringComparison.Ordinal))
            {
                string sidecar = Path.GetFileNameWithoutExtension(FullPath) + ".password.txt";
                Skipped = Password is null
                    ? $"encrypted: it opens only with a password (its first line in {sidecar} beside it)"
                    : $"encrypted: the password in {sidecar} does not open it";
            }
            else
            {
                NotOpened = why;
            }
        }

        public void Count(string key, long value) => Counts[key] = (Counts.TryGetValue(key, out long known) ? known : 0) + value;

        public void Error(string step, int page, Exception e)
        {
            if (string.IsNullOrWhiteSpace(e.Message))
            {
                Fail(step, $"{(page >= 0 ? $"page {page + 1} " : "")}cannot be read and says nothing why ({e.GetType().Name})");
                return;
            }

            Errors.Add(new PdfRealError(step, page, PdfRealDescribe(e)));
        }

        public void Fail(string step, string what) => Failures.Add($"{step}: {what}");

        // A page's issues: each (kind, detail) once a page, with the larger count when several reads report it.
        public void AddIssues(int page, IReadOnlyList<DocumentIssue> issues)
        {
            foreach (var issue in issues)
            {
                var key = (page, issue.Kind, issue.Detail);
                _issues[key] = Math.Max(_issues.TryGetValue(key, out int known) ? known : 0, issue.Count);
            }
        }

        public SortedDictionary<string, long> IssuesByKind()
        {
            var kinds = new SortedDictionary<string, long>(StringComparer.Ordinal);
            foreach (var ((_, kind, _), count) in _issues)
            {
                kinds[kind] = (kinds.TryGetValue(kind, out long known) ? known : 0) + count;
            }

            return kinds;
        }

        // Each issue's kind and detail, with its count and the pages it is on (first pages first).
        public List<(string Kind, string Detail, long Count, List<int> Pages)> IssueDetails() =>
        [
            .. _issues.GroupBy(i => (i.Key.Kind, i.Key.Detail))
                .Select(g => (g.Key.Kind, g.Key.Detail, (long)g.Sum(i => i.Value), g.Select(i => i.Key.Page).Order().ToList()))
                .OrderBy(i => i.Item4[0]).ThenBy(i => i.Kind, StringComparer.Ordinal).ThenBy(i => i.Detail, StringComparer.Ordinal),
        ];
    }

    // A pass's clock: started and stopped around the work it times (the open and what is only recorded left out), with the
    // managed bytes allocated in between on every thread (rule 63: drawing spreads its bands over the cores).
    private sealed class PdfRealMeter
    {
        private long _start, _allocated;
        private bool _running;

        public double Ms { get; private set; }

        public long Bytes { get; private set; }

        public void Start()
        {
            _allocated = GC.GetTotalAllocatedBytes(precise: true);
            _start = Stopwatch.GetTimestamp();
            _running = true;
        }

        public void Stop()
        {
            if (!_running)
            {
                return;
            }

            Ms += Stopwatch.GetElapsedTime(_start).TotalMilliseconds;
            Bytes += GC.GetTotalAllocatedBytes(precise: true) - _allocated;
            _running = false;
        }
    }

    // The run for the variables as they are now; a SKIP when IDRAK_DOCUMENTS_REAL names no folder.
    private static PdfRealRun PdfRealRunFor()
    {
        string? named = Environment.GetEnvironmentVariable(PdfRealVariable);
        if (string.IsNullOrWhiteSpace(named))
        {
            Skip($"{PdfRealVariable} names no folder: the owner's real PDFs are read only when it names one (tools/documents/README.md, \"Real files\")");
        }

        string folder = Path.GetFullPath(named!.Trim());
        Check(Directory.Exists(folder), $"{PdfRealVariable}={named} is not a folder");

        string? reference = Environment.GetEnvironmentVariable(PdfRealReferenceVariable) is { } given && !string.IsNullOrWhiteSpace(given) ? Path.GetFullPath(given.Trim()) : null;
        Check(reference is null || Directory.Exists(reference), $"{PdfRealReferenceVariable}={reference} is not a folder");
        reference ??= RepositoryPath("test-output", "documents", "real-reference") is { } fallback && Directory.Exists(fallback) ? fallback : null;

        int warm = 3;
        if (Environment.GetEnvironmentVariable(PdfRealWarmVariable) is { } warmText && !string.IsNullOrWhiteSpace(warmText))
        {
            Check(int.TryParse(warmText.Trim(), NumberStyles.None, CultureInfo.InvariantCulture, out warm) && warm is >= 0 and <= 100,
                $"{PdfRealWarmVariable}={warmText} is not a whole number of warm passes from 0 to 100");
        }

        bool savePages = false, saveFiles = false;
        foreach (string part in (Environment.GetEnvironmentVariable(PdfRealSaveVariable) ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            switch (part.ToLowerInvariant())
            {
                case "pages":
                    savePages = true;
                    break;
                case "files":
                    saveFiles = true;
                    break;
                case "all":
                    savePages = saveFiles = true;
                    break;
                case "none":
                    break;
                default:
                    Check(false, $"{PdfRealSaveVariable}: '{part}' is not pages, files, all or none");
                    break;
            }
        }

        string key = string.Join('\n', folder, reference, warm, savePages, saveFiles);
        lock (PdfRealGate)
        {
            if (s_pdfRealRun is { } known && known.Key == key)
            {
                return known;
            }

            var paths = Directory.EnumerateFiles(folder, "*.pdf", new EnumerationOptions { RecurseSubdirectories = true, IgnoreInaccessible = true, MatchCasing = MatchCasing.CaseInsensitive })
                .Select(p => (Full: p, Relative: Path.GetRelativePath(folder, p)))
                .OrderBy(p => p.Relative.Replace('\\', '/'), StringComparer.Ordinal)
                .ToList();
            Check(paths.Count > 0, $"no .pdf file in {folder} or the folders under it");
            var limits = PdfRealLimits();
            var files = paths.Select(p => new PdfRealFile(p.Full, p.Relative, new FileInfo(p.Full).Length, PdfRealPassword(p.Full), limits)).ToList();

            // The run's own outputs start empty (a page or file of an earlier run never mixes in); the summary is replaced.
            string output = RepositoryPath("test-output", "documents", "real");
            foreach (string own in new[] { "drawn", "written" })
            {
                string path = Path.Combine(output, own);
                if (Directory.Exists(path))
                {
                    Directory.Delete(path, recursive: true);
                }
            }

            var run = new PdfRealRun(key, folder, reference, warm, savePages, saveFiles, output, files, limits);
            s_pdfRealRun = run;
            return run;
        }
    }

    // C8's default limits with each one measured from the free memory resolved once, now (its variable's value where one is
    // set): the options measure again at every open, so a file near a limit (a flate bomb, a huge stream) would read or be
    // refused by what the machine had free at that moment, and a file written back could be read under other limits than
    // its source.
    private static DocumentOptions PdfRealLimits()
    {
        var defaults = new DocumentOptions();
        return defaults with { MaxStreamBytes = defaults.MaxStreamBytes, MaxObjects = defaults.MaxObjects, CacheBytes = defaults.CacheBytes };
    }

    // The password beside a file (the first line of NAME.password.txt), or null.
    private static string? PdfRealPassword(string path)
    {
        string sidecar = Path.Combine(Path.GetDirectoryName(path)!, Path.GetFileNameWithoutExtension(path) + ".password.txt");
        return File.Exists(sidecar) ? File.ReadLines(sidecar, Encoding.UTF8).FirstOrDefault() ?? "" : null;
    }

    // The failures the contract documents for a file or part that cannot be read as asked (damaged, hostile, past a limit,
    // locked by a password, a part no registered codec reads): listed with the reason they give. Anything else is a crash.
    private static bool PdfRealReported(Exception e) => e switch
    {
        InvalidDataException or NotSupportedException => true,
        AggregateException all => all.InnerExceptions.Count > 0 && all.InnerExceptions.All(PdfRealReported),
        _ => false,
    };

    // An exception as one line: its type and message (an aggregate's first inner one), at most 400 characters.
    private static string PdfRealDescribe(Exception e)
    {
        if (e is AggregateException { InnerExceptions.Count: > 0 } all)
        {
            e = all.InnerExceptions[0];
        }

        string text = $"{e.GetType().Name}: {e.Message}".ReplaceLineEndings(" ");
        return text.Length <= 400 ? text : text[..400] + "...";
    }

    // Times `pass` on `file`: cold (record: true, the results kept) and the run's warm passes (record: false), the median kept.
    private static void PdfRealTime(PdfRealRun run, PdfRealFile file, string step, Action<bool, PdfRealMeter> pass)
    {
        var cold = new PdfRealMeter();
        pass(true, cold);
        var warm = new List<PdfRealMeter>();
        for (int i = 0; i < run.Warm; i++)
        {
            var meter = new PdfRealMeter();
            pass(false, meter);
            warm.Add(meter);
        }

        warm.Sort((a, b) => a.Ms.CompareTo(b.Ms));
        var median = warm.Count > 0 ? warm[warm.Count / 2] : null;
        file.Steps[step] = new PdfRealTiming(cold.Ms, cold.Bytes, median?.Ms, median?.Bytes);
    }

    // One step of a file: a failure the contract documents is listed with its reason, any other exception is a crash.
    private static void PdfRealStep(PdfRealFile file, string step, Action body)
    {
        try
        {
            body();
        }
        catch (Exception e) when (PdfRealReported(e))
        {
            file.Error(step, -1, e);
        }
        catch (Exception e)
        {
            file.Fail(step, $"crashed: {PdfRealDescribe(e)}");
        }
    }

    // Page `index` read with `parts`, or null when it cannot be (listed, or a crash failing the file, on the recording pass).
    private static DocumentPage? PdfRealReadPage(PdfRealFile file, string step, IDocument document, int index, DocumentParts parts, bool record)
    {
        try
        {
            return document.ReadPage(index, parts);
        }
        catch (Exception e) when (PdfRealReported(e))
        {
            if (record)
            {
                file.Error(step, index, e);
            }
        }
        catch (Exception e)
        {
            if (record)
            {
                file.Fail(step, $"page {index + 1}: crashed: {PdfRealDescribe(e)}");
            }
        }

        return null;
    }

    // Whether the file opens, learnt once (untimed) when the read test has not opened it.
    private static bool PdfRealOpen(PdfRealFile file)
    {
        if (file.Opened is null)
        {
            try
            {
                using var document = DocumentFormats.Open(file.FullPath, file.Options(DocumentTextLayouts.Lines));
                file.Open(document.PageCount, document.Info);
            }
            catch (Exception e) when (PdfRealReported(e))
            {
                file.NotOpen(e);
            }
            catch (Exception e)
            {
                file.Opened = false;
                file.Fail("open", $"crashed: {PdfRealDescribe(e)}");
            }
        }

        return file.Opened == true;
    }

    // Runs `work` on every file in order (a line each), writing the summary as it goes and at the end; then fails naming
    // every file and step that failed in this test.
    private static void PdfRealEach(PdfRealRun run, string test, Action<PdfRealFile> work)
    {
        var failures = new List<string>();
        for (int i = 0; i < run.Files.Count; i++)
        {
            var file = run.Files[i];
            int before = file.Failures.Count;
            long start = Stopwatch.GetTimestamp();
            try
            {
                work(file);
            }
            catch (Exception e)
            {
                file.Fail(test, $"crashed: {PdfRealDescribe(e)}");
            }

            failures.AddRange(file.Failures.Skip(before).Select(f => $"{file.Name}: {f}"));
            string state = file.Skipped is { } skipped ? $"skipped ({skipped})" : file.NotOpened is { } why ? $"does not open ({why})" : $"{file.Pages} pages";
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"    {test} [{i + 1}/{run.Files.Count}] {file.Name}: {state}, {file.Failures.Count - before} failures ({Stopwatch.GetElapsedTime(start).TotalSeconds:F1} s)"));
            run.WriteSoon();
        }

        run.Write();
        string summary = Path.Combine(run.Output, "summary.txt");
        Console.WriteLine($"    the summary: {summary}");
        Check(failures.Count == 0, $"{failures.Count} failure(s) in {test}: {string.Join("; ", failures.Take(20))}{(failures.Count > 20 ? $"; and {failures.Count - 20} more" : "")} (all of them in {summary})");
    }

    private static void PdfRealRead(Device device)
    {
        CpuOnly(device);
        var run = PdfRealRunFor();
        PdfRealEach(run, "read", file => PdfRealReadFile(run, file));
    }

    private static void PdfRealDraw(Device device)
    {
        CpuOnly(device);
        var run = PdfRealRunFor();
        PdfRealEach(run, "draw", file => PdfRealDrawFile(run, file));
    }

    private static void PdfRealWriteBack(Device device)
    {
        CpuOnly(device);
        var run = PdfRealRunFor();
        PdfRealEach(run, "write back", file => PdfRealWriteBackFile(run, file));
    }

    private static void PdfRealReadFile(PdfRealRun run, PdfRealFile file)
    {
        // Opened (its information and page count): the run's first open of the file is its cold one.
        try
        {
            PdfRealTime(run, file, "open", (record, meter) =>
            {
                meter.Start();
                using var document = DocumentFormats.Open(file.FullPath, file.Options(DocumentTextLayouts.Lines));
                var info = document.Info;
                int pages = document.PageCount;
                meter.Stop();
                if (record)
                {
                    file.Open(pages, info);
                }
            });
        }
        catch (Exception e) when (PdfRealReported(e))
        {
            file.NotOpen(e);
            return;
        }
        catch (Exception e)
        {
            file.Opened = false;
            file.Fail("open", $"crashed: {PdfRealDescribe(e)}");
            return;
        }

        // Text through the lines layout: every page's runs, lines and text.
        var texts = new string?[file.Pages];
        file.Texts = texts;
        PdfRealStep(file, "text", () => PdfRealTime(run, file, "text", (record, meter) =>
        {
            using var document = DocumentFormats.Open(file.FullPath, file.Options(DocumentTextLayouts.Lines));
            meter.Start();
            for (int i = 0; i < file.Pages; i++)
            {
                if (PdfRealReadPage(file, "text", document, i, DocumentParts.Text, record) is not { } page || !record)
                {
                    continue;
                }

                file.Count("runs", page.Runs.Count);
                file.Count("hiddenRuns", page.Runs.Count(r => r.Hidden));
                file.Count("lines", page.Lines.Count);
                file.Count("characters", page.Text.Length);
                file.AddIssues(i, page.Issues);
                texts[i] = page.Text;
            }

            meter.Stop();
        }));

        // Text through the tagged layout (the order a tagged file gives; untagged pages read as lines).
        PdfRealStep(file, "tagged", () => PdfRealTime(run, file, "tagged", (record, meter) =>
        {
            using var document = DocumentFormats.Open(file.FullPath, file.Options(DocumentTextLayouts.Tagged));
            meter.Start();
            for (int i = 0; i < file.Pages; i++)
            {
                if (PdfRealReadPage(file, "tagged", document, i, DocumentParts.Text, record) is not { } page || !record)
                {
                    continue;
                }

                file.Count("taggedLines", page.Lines.Count);
                file.Count("taggedRuns", page.Runs.Count(r => r.Tag is { IsArtifact: false }));
                file.Count("taggedPagesInAnotherOrder", texts[i] is { } lines && lines != page.Text ? 1 : 0);
            }

            meter.Stop();
        }));

        // The other parts of the page: images (listed), paths, clips, shadings, annotations.
        PdfRealStep(file, "parts", () => PdfRealTime(run, file, "parts", (record, meter) =>
        {
            using var document = DocumentFormats.Open(file.FullPath, file.Options(DocumentTextLayouts.Lines));
            meter.Start();
            for (int i = 0; i < file.Pages; i++)
            {
                if (PdfRealReadPage(file, "parts", document, i, DocumentParts.Images | DocumentParts.Paths | DocumentParts.Annotations, record) is not { } page || !record)
                {
                    continue;
                }

                file.Count("images", page.Images.Count);
                file.Count("paths", page.Paths.Count);
                file.Count("clips", page.Clips.Count);
                file.Count("shadings", page.Shadings.Count);
                file.Count("annotations", page.Annotations.Count);
                foreach (var annotation in page.Annotations)
                {
                    file.AnnotationKinds[annotation.Kind] = (file.AnnotationKinds.TryGetValue(annotation.Kind, out long known) ? known : 0) + 1;
                }

                file.AddIssues(i, page.Issues);
            }

            meter.Stop();
        }));

        // Every image decoded, with its mask.
        PdfRealStep(file, "images", () => PdfRealTime(run, file, "images", (record, meter) =>
        {
            using var document = DocumentFormats.Open(file.FullPath, file.Options(DocumentTextLayouts.Lines));
            meter.Start();
            if (record)
            {
                file.Count("imagesDecoded", 0);
                file.Count("imagesNotDecoded", 0);
                file.Count("imagePixels", 0);
                file.Count("masks", 0);
            }

            for (int i = 0; i < file.Pages; i++)
            {
                if (PdfRealReadPage(file, "images", document, i, DocumentParts.Images, record) is not { } page)
                {
                    continue;
                }

                foreach (var image in page.Images)
                {
                    try
                    {
                        var pixels = image.Decode();
                        var mask = image.Mask();
                        if (record)
                        {
                            file.Count("imagesDecoded", 1);
                            file.Count("imagePixels", (long)pixels.Width * pixels.Height);
                            file.Count("masks", mask is null ? 0 : 1);
                        }
                    }
                    catch (Exception e) when (PdfRealReported(e))
                    {
                        if (record)
                        {
                            file.Count("imagesNotDecoded", 1);
                            file.Error("images", i, e);
                        }
                    }
                    catch (Exception e)
                    {
                        if (record)
                        {
                            file.Fail("images", $"page {i + 1}: an image ({image.Encoding}, {image.PixelWidth} x {image.PixelHeight}) crashed: {PdfRealDescribe(e)}");
                        }
                    }
                }
            }

            meter.Stop();
        }));

        // The document's own parts: fields, bookmarks, named destinations, page labels, attachments (each read to its end).
        PdfRealStep(file, "navigation", () => PdfRealTime(run, file, "navigation", (record, meter) =>
        {
            using var document = DocumentFormats.Open(file.FullPath, file.Options(DocumentTextLayouts.Lines));
            T Part<T>(string what, Func<T> read, T none)
            {
                try
                {
                    return read();
                }
                catch (Exception e) when (PdfRealReported(e))
                {
                    if (record)
                    {
                        file.Error($"navigation ({what})", -1, e);
                    }
                }
                catch (Exception e)
                {
                    if (record)
                    {
                        file.Fail("navigation", $"{what} crashed: {PdfRealDescribe(e)}");
                    }
                }

                return none;
            }

            meter.Start();
            var fields = Part("fields", () => document.Fields, []);
            var outline = Part("bookmarks", () => document.Outline, []);
            var destinations = Part("named destinations", () => document.Destinations, []);
            int labelled = Part("page labels", () => Enumerable.Range(0, file.Pages).Count(i => document.PageLabel(i) is not null), 0);
            var attachments = Part("attachments", () => document.Attachments, []);
            long attached = Part("attachments' bytes", () =>
            {
                long total = 0;
                foreach (var attachment in attachments)
                {
                    using var content = attachment.Read();
                    using var counter = new PdfByteCounter();
                    content.CopyTo(counter);
                    total += counter.Length;
                }

                return total;
            }, 0L);
            meter.Stop();
            if (!record)
            {
                return;
            }

            static int Bookmarks(IReadOnlyList<DocumentOutline> items) => items.Sum(i => 1 + Bookmarks(i.Children));
            file.Count("fields", fields.Count);
            file.Count("widgets", fields.Sum(f => f.Widgets.Count));
            foreach (var field in fields)
            {
                string kind = field.Kind.ToString();
                file.FieldKinds[kind] = (file.FieldKinds.TryGetValue(kind, out long known) ? known : 0) + 1;
            }

            file.Count("bookmarks", Bookmarks(outline));
            file.Count("namedDestinations", destinations.Count);
            file.Count("labelledPages", labelled);
            file.Count("attachments", attachments.Count);
            file.Count("attachmentBytes", attached);
        }));

        // Signatures listed and checked (nothing trusted, nothing fetched).
        PdfRealStep(file, "signatures", () => PdfRealTime(run, file, "signatures", (record, meter) =>
        {
            using var document = DocumentFormats.Open(file.FullPath, file.Options(DocumentTextLayouts.Lines));
            meter.Start();
            var signatures = document.Signatures;
            var checks = signatures.Select(s => s.Check(run.Trust)).ToList();
            meter.Stop();
            if (!record)
            {
                return;
            }

            file.Count("signatures", signatures.Count);
            for (int i = 0; i < signatures.Count; i++)
            {
                var steps = new JsonObject();
                foreach (var step in checks[i].Steps)
                {
                    steps[step.Name] = new JsonObject { ["passed"] = step.Passed, ["detail"] = step.Detail };
                }

                file.Signatures.Add(new JsonObject
                {
                    ["field"] = signatures[i].Field, ["kind"] = signatures[i].Kind.ToString(), ["subFilter"] = signatures[i].KindName,
                    ["coversWholeFile"] = signatures[i].CoversWholeFile, ["changes"] = signatures[i].Changes.ToString(),
                    ["intact"] = PdfRealIntact(checks[i]), ["valid"] = checks[i].Valid, ["steps"] = steps,
                });
            }
        }));

        // The reference extractor's text, when there is one: its differences listed.
        if (PdfRealReferenceFile(run, file) is { } reference)
        {
            var compared = PdfRealCompare(reference, texts);
            compared["file"] = reference;
            file.Reference = compared;
        }
    }

    // A signature's digest and signature both check: the signed bytes are as signed (whatever the trust).
    private static bool PdfRealIntact(SignatureCheck check) =>
        check.Steps.Any(s => s.Name == SignatureCheck.Digest && s.Passed) && check.Steps.Any(s => s.Name == SignatureCheck.Signature && s.Passed);

    private static void PdfRealDrawFile(PdfRealRun run, PdfRealFile file)
    {
        if (!PdfRealOpen(file))
        {
            return;
        }

        string? folder = run.SavePages ? Path.Combine(run.Output, "drawn") : null;
        PdfRealStep(file, "draw", () => PdfRealTime(run, file, "draw", (record, meter) =>
        {
            using var document = DocumentFormats.Open(file.FullPath, file.Options(DocumentTextLayouts.Lines));
            if (record)
            {
                file.Count("pagesDrawn", 0);
                file.Count("drawnPixels", 0);
            }

            for (int i = 0; i < file.Pages; i++)
            {
                DocumentDrawing? drawing = null;
                meter.Start();
                try
                {
                    drawing = document.DrawPage(i, run.Draw);
                }
                catch (Exception e) when (PdfRealReported(e))
                {
                    if (record)
                    {
                        file.Error("draw", i, e);
                    }
                }
                catch (Exception e)
                {
                    if (record)
                    {
                        file.Fail("draw", $"page {i + 1}: crashed: {PdfRealDescribe(e)}");
                    }
                }

                meter.Stop();
                if (drawing is null || !record)
                {
                    continue;
                }

                file.Count("pagesDrawn", 1);
                file.Count("drawnPixels", (long)drawing.Image.Width * drawing.Image.Height);
                foreach (var issue in drawing.Issues)
                {
                    file.DrawIssues[issue.Kind] = (file.DrawIssues.TryGetValue(issue.Kind, out long known) ? known : 0) + issue.Count;
                }

                if (folder is not null)
                {
                    string png = Path.Combine(folder, Path.ChangeExtension(file.Relative, null) + $".p{i + 1}.png");
                    Directory.CreateDirectory(Path.GetDirectoryName(png)!);
                    File.WriteAllBytes(png, ImageEncoders.Get("png").Encode(drawing.Image));
                }
            }
        }));
    }

    private static void PdfRealWriteBackFile(PdfRealRun run, PdfRealFile file)
    {
        if (!PdfRealOpen(file))
        {
            return;
        }

        // The source as read: what both written files must read back to (a page the source cannot read is left out).
        List<PdfRealPage?> source;
        List<(string Field, bool Digest, bool Signature)> signed;
        string title;
        using (var document = DocumentFormats.Open(file.FullPath, file.Options(DocumentTextLayouts.Lines)))
        {
            source = PdfRealPages(document, out _);
            signed = PdfRealSigned(run, document);
            title = $"{document.Info.Title ?? Path.GetFileNameWithoutExtension(file.FullPath)} (written back by Idrak)";
        }

        foreach (var mode in new[] { DocumentSaveMode.Rewrite, DocumentSaveMode.Incremental })
        {
            string step = mode == DocumentSaveMode.Rewrite ? "rewrite" : "incremental";
            bool written = false;
            long bytes = 0;
            PdfRealStep(file, step, () =>
            {
                PdfRealTime(run, file, step, (record, meter) =>
                {
                    using var document = DocumentFormats.Open(file.FullPath, file.Options(DocumentTextLayouts.Lines));
                    using var counter = new PdfByteCounter();
                    meter.Start();
                    using (var editor = DocumentWriters.Get("pdf").Edit(document, counter, run.WriteOptions))
                    {
                        editor.SetInfo(document.Info with { Title = title });
                        editor.Save(mode);
                    }

                    meter.Stop();
                    if (record)
                    {
                        bytes = counter.Length;
                    }
                });
                written = true;
            });

            if (written)
            {
                file.WrittenBack[step] = PdfRealCheckWritten(run, file, step, mode, title, bytes, source, signed);
            }
        }
    }

    // Each page as read with its text and images (null: the page cannot be read; `failure` the first reason).
    private static List<PdfRealPage?> PdfRealPages(IDocument document, out string? failure)
    {
        failure = null;
        var pages = new List<PdfRealPage?>(document.PageCount);
        for (int i = 0; i < document.PageCount; i++)
        {
            try
            {
                var page = document.ReadPage(i, DocumentParts.Text | DocumentParts.Images);
                pages.Add(new PdfRealPage(page.Width, page.Height, page.Rotation, page.Text, page.Runs.Count, page.Images.Count));
            }
            catch (Exception e)
            {
                failure ??= $"page {i + 1}: {PdfRealDescribe(e)}";
                pages.Add(null);
            }
        }

        return pages;
    }

    // The document's signatures with whether each one's digest and signature check.
    private static List<(string Field, bool Digest, bool Signature)> PdfRealSigned(PdfRealRun run, IDocument document)
    {
        try
        {
            return [.. document.Signatures.Select(s => s.Check(run.Trust)).Zip(document.Signatures, (check, s) =>
                (s.Field, check.Steps.Any(c => c.Name == SignatureCheck.Digest && c.Passed), check.Steps.Any(c => c.Name == SignatureCheck.Signature && c.Passed)))];
        }
        catch (Exception e) when (PdfRealReported(e))
        {
            return [];
        }
    }

    // The file written back once more, to a file this time, and read back: the same pages, sizes, turns, text, runs and
    // images as the source, the title given; an update also keeps the source as its prefix and its signatures' digests.
    private static JsonObject PdfRealCheckWritten(PdfRealRun run, PdfRealFile file, string step, DocumentSaveMode mode, string title, long bytes,
        List<PdfRealPage?> source, List<(string Field, bool Digest, bool Signature)> signed)
    {
        var result = new JsonObject { ["bytes"] = bytes };
        if (mode == DocumentSaveMode.Incremental)
        {
            result["appended"] = bytes - file.Bytes;
        }

        string path = Path.Combine(run.Output, "written", Path.ChangeExtension(file.Relative, null) + $".{step}.pdf");
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var differences = new List<string>();
        try
        {
            try
            {
                using var document = DocumentFormats.Open(file.FullPath, file.Options(DocumentTextLayouts.Lines));
                using var output = File.Create(path);
                using var editor = DocumentWriters.Get("pdf").Edit(document, output, run.WriteOptions);
                editor.SetInfo(document.Info with { Title = title });
                editor.Save(mode);
            }
            catch (Exception e)
            {
                differences.Add($"written to a file it failed where the timed passes did not: {PdfRealDescribe(e)}");
                return PdfRealWrittenResult(file, step, result, differences);
            }

            IDocument back;
            try
            {
                back = DocumentFormats.Open(path, file.Options(DocumentTextLayouts.Lines));
            }
            catch (Exception e)
            {
                differences.Add($"it does not open: {PdfRealDescribe(e)}");
                return PdfRealWrittenResult(file, step, result, differences);
            }

            using (back)
            {
                if (back.PageCount != source.Count)
                {
                    differences.Add($"{back.PageCount} pages, the source {source.Count}");
                }

                if (back.Info.Title != title)
                {
                    differences.Add($"its title reads \"{back.Info.Title}\", written \"{title}\"");
                }

                var pages = PdfRealPages(back, out string? unread);
                int differing = 0;
                for (int i = 0; i < Math.Min(pages.Count, source.Count); i++)
                {
                    if (source[i] is not { } was)
                    {
                        continue;
                    }

                    string? what = pages[i] is not { } now ? "it cannot be read"
                        : MathF.Abs(now.Width - was.Width) > 1e-3f || MathF.Abs(now.Height - was.Height) > 1e-3f ? string.Create(CultureInfo.InvariantCulture, $"its size {now.Width} x {now.Height}, the source's {was.Width} x {was.Height}")
                        : now.Rotation != was.Rotation ? $"turned {now.Rotation}, the source {was.Rotation}"
                        : now.Text != was.Text ? $"its text differs (\"{PdfRealClip(now.Text)}\", the source \"{PdfRealClip(was.Text)}\")"
                        : now.Runs != was.Runs ? $"{now.Runs} runs, the source {was.Runs}"
                        : now.Images != was.Images ? $"{now.Images} images, the source {was.Images}"
                        : null;
                    if (what is not null && ++differing <= 5)
                    {
                        differences.Add($"page {i + 1}: {what}{(what == "it cannot be read" && unread is not null ? $" ({unread})" : "")}");
                    }
                }

                if (differing > 5)
                {
                    differences.Add($"and {differing - 5} more pages differ");
                }

                if (mode == DocumentSaveMode.Incremental)
                {
                    bool kept = PdfRealPrefix(file.FullPath, path);
                    result["sourceKept"] = kept;
                    if (!kept)
                    {
                        differences.Add("the source's bytes are not its prefix");
                    }

                    if (signed.Count > 0)
                    {
                        var after = PdfRealSigned(run, back);
                        bool same = after.Count == signed.Count && after.Zip(signed).All(p => p.First.Field == p.Second.Field && p.First.Digest == p.Second.Digest && p.First.Signature == p.Second.Signature);
                        result["signaturesKept"] = same;
                        if (!same)
                        {
                            differences.Add($"its signatures check otherwise: {string.Join(", ", after.Select(s => $"{s.Field} digest {s.Digest} signature {s.Signature}"))}, the source {string.Join(", ", signed.Select(s => $"{s.Field} digest {s.Digest} signature {s.Signature}"))}");
                        }
                    }
                }
            }

            return PdfRealWrittenResult(file, step, result, differences);
        }
        finally
        {
            if (!run.SaveFiles && File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    private static JsonObject PdfRealWrittenResult(PdfRealFile file, string step, JsonObject result, List<string> differences)
    {
        result["readsTheSame"] = differences.Count == 0;
        if (differences.Count > 0)
        {
            result["differences"] = new JsonArray([.. differences.Select(d => (JsonNode?)d)]);
            foreach (string difference in differences)
            {
                file.Fail(step, $"written back as {(step == "rewrite" ? "a rewrite" : "an incremental update")}, {difference}");
            }
        }

        return result;
    }

    // Whether `written` starts with every byte of `source`.
    private static bool PdfRealPrefix(string source, string written)
    {
        using var a = File.OpenRead(source);
        using var b = File.OpenRead(written);
        if (b.Length < a.Length)
        {
            return false;
        }

        byte[] left = new byte[1 << 20], right = new byte[1 << 20];
        while (true)
        {
            int read = a.ReadAtLeast(left, left.Length, throwOnEndOfStream: false);
            if (read == 0)
            {
                return true;
            }

            if (b.ReadAtLeast(right, read, throwOnEndOfStream: false) < read || !left.AsSpan(0, read).SequenceEqual(right.AsSpan(0, read)))
            {
                return false;
            }
        }
    }

    private static string PdfRealClip(string text)
    {
        text = text.ReplaceLineEndings("/");
        return text.Length <= 80 ? text : text[..80] + "...";
    }

    // The reference text for a file: NAME.reference.txt beside it, else at its path under the reference folder.
    private static string? PdfRealReferenceFile(PdfRealRun run, PdfRealFile file)
    {
        string beside = Path.Combine(Path.GetDirectoryName(file.FullPath)!, Path.GetFileNameWithoutExtension(file.FullPath) + ".reference.txt");
        if (File.Exists(beside))
        {
            return beside;
        }

        string? under = run.Reference is { } folder ? Path.Combine(folder, Path.ChangeExtension(file.Relative, null) + ".reference.txt") : null;
        return under is not null && File.Exists(under) ? under : null;
    }

    // The reference's pages (split at form feeds) against Idrak's, whitespace collapsed: the same; the same but for spaces;
    // the same words in another order; the same words, some of them as drawn (a right-to-left word's characters in the
    // order they are drawn, as a reader that does not reorder them gives it); or words that differ (the reference's missing
    // from Idrak's and Idrak's extra, and the share of words in common). Listed, never failed: the reference fixes what
    // the text must say, not how fast Idrak may be.
    private static JsonObject PdfRealCompare(string referenceFile, string?[] texts)
    {
        string[] reference = File.ReadAllText(referenceFile, Encoding.UTF8).Split('\f');
        int same = 0, spacing = 0, order = 0, drawn = 0, words = 0, notRead = 0, listed = 0;
        var differences = new JsonArray();
        int count = Math.Max(reference.Length, texts.Length);
        for (int i = 0; i < count; i++)
        {
            string? theirs = i < reference.Length ? reference[i] : null, ours = i < texts.Length ? texts[i] : null;
            JsonObject? difference = null;
            if (theirs is null || ours is null)
            {
                notRead++;
                difference = new JsonObject { ["page"] = i + 1, ["kind"] = theirs is null ? "not in the reference" : i < texts.Length ? "not read by Idrak" : "not in Idrak's pages" };
            }
            else
            {
                string a = PdfRealCollapse(theirs), b = PdfRealCollapse(ours);
                if (a == b)
                {
                    same++;
                }
                else if (a.Replace(" ", "", StringComparison.Ordinal) == b.Replace(" ", "", StringComparison.Ordinal))
                {
                    spacing++;
                    difference = new JsonObject { ["page"] = i + 1, ["kind"] = "spacing" };
                }
                else
                {
                    var (missing, extra, share, reversed) = PdfRealWords(a, b);
                    if (missing.Count == 0 && extra.Count == 0 && reversed == 0)
                    {
                        order++;
                        difference = new JsonObject { ["page"] = i + 1, ["kind"] = "order" };
                    }
                    else if (missing.Count == 0 && extra.Count == 0)
                    {
                        drawn++;
                        difference = new JsonObject { ["page"] = i + 1, ["kind"] = "drawn order", ["wordsAsDrawn"] = reversed };
                    }
                    else
                    {
                        words++;
                        difference = new JsonObject
                        {
                            ["page"] = i + 1, ["kind"] = "words", ["wordsInCommon"] = Math.Round(share, 3), ["wordsAsDrawn"] = reversed,
                            ["missing"] = new JsonArray([.. missing.Take(8).Select(w => (JsonNode?)w)]), ["extra"] = new JsonArray([.. extra.Take(8).Select(w => (JsonNode?)w)]),
                        };
                    }
                }
            }

            if (difference is not null && ++listed <= 50)
            {
                differences.Add(difference);
            }
        }

        return new JsonObject
        {
            ["pages"] = reference.Length, ["same"] = same, ["spacing"] = spacing, ["order"] = order, ["drawnOrder"] = drawn, ["words"] = words, ["notRead"] = notRead,
            ["differences"] = differences, ["moreDifferences"] = Math.Max(0, listed - 50),
        };
    }

    // Whitespace runs as one space, trimmed.
    private static string PdfRealCollapse(string text) => string.Join(' ', text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));

    // The reference's words Idrak lacks and Idrak's the reference lacks (each once, in order), the share in common, and how
    // many of the reference's words matched only with their characters (text elements) reversed: as drawn.
    private static (List<string> Missing, List<string> Extra, double Share, int Reversed) PdfRealWords(string reference, string idrak)
    {
        string[] theirs = reference.Split(' ', StringSplitOptions.RemoveEmptyEntries), ours = idrak.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var left = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (string word in ours)
        {
            left[word] = left.GetValueOrDefault(word) + 1;
        }

        int common = 0, reversed = 0;
        var missing = new List<string>();
        foreach (string word in theirs)
        {
            string turned;
            if (left.GetValueOrDefault(word) > 0)
            {
                left[word]--;
                common++;
            }
            else if (left.GetValueOrDefault(turned = PdfRealReversed(word)) > 0)
            {
                left[turned]--;
                common++;
                reversed++;
            }
            else if (!missing.Contains(word))
            {
                missing.Add(word);
            }
        }

        var extra = new List<string>();
        foreach (string word in ours)
        {
            if (left.GetValueOrDefault(word) > 0 && !extra.Contains(word))
            {
                extra.Add(word);
            }
        }

        return (missing, extra, theirs.Length + ours.Length == 0 ? 1 : 2.0 * common / (theirs.Length + ours.Length), reversed);
    }

    // A word's text elements (grapheme clusters) in the other order.
    private static string PdfRealReversed(string word)
    {
        var elements = new List<string>();
        var walk = StringInfo.GetTextElementEnumerator(word);
        while (walk.MoveNext())
        {
            elements.Add(walk.GetTextElement());
        }

        elements.Reverse();
        return string.Concat(elements);
    }

    private static void PdfRealWriteFile(string path, string text)
    {
        string partial = path + ".partial";
        File.WriteAllText(partial, text, new UTF8Encoding(false));
        File.Move(partial, path, overwrite: true);
    }

    private static JsonObject PdfRealTimingJson(PdfRealTiming timing, int pages)
    {
        int per = Math.Max(1, pages);
        var json = new JsonObject
        {
            ["coldMs"] = Math.Round(timing.ColdMs, 3), ["coldMsPerPage"] = Math.Round(timing.ColdMs / per, 3), ["coldKiBPerPage"] = Math.Round(timing.ColdBytes / 1024.0 / per, 1),
        };
        if (timing.WarmMs is { } warm)
        {
            json["warmMs"] = Math.Round(warm, 3);
            json["warmMsPerPage"] = Math.Round(warm / per, 3);
            json["warmKiBPerPage"] = Math.Round((timing.WarmBytes ?? 0) / 1024.0 / per, 1);
        }

        return json;
    }

    private static JsonObject PdfRealCounts(IEnumerable<KeyValuePair<string, long>> counts)
    {
        var json = new JsonObject();
        foreach (var (key, value) in counts)
        {
            json[key] = value;
        }

        return json;
    }

    private static JsonObject PdfRealSummaryJson(PdfRealRun run)
    {
        var files = new JsonArray();
        foreach (var file in run.Files)
        {
            var steps = new JsonObject();
            foreach (var (step, timing) in file.Steps)
            {
                steps[step] = PdfRealTimingJson(timing, file.Pages);
            }

            var errors = new JsonArray();
            foreach (var group in file.Errors.GroupBy(e => (e.Step, e.Error)))
            {
                var pages = group.Where(e => e.Page >= 0).Select(e => e.Page + 1).Distinct().Order().ToList();
                errors.Add(new JsonObject { ["step"] = group.Key.Step, ["error"] = group.Key.Error, ["pages"] = new JsonArray([.. pages.Select(p => (JsonNode)p)]) });
            }

            var details = new JsonArray();
            foreach (var (kind, detail, count, pages) in file.IssueDetails().Take(50))
            {
                details.Add(new JsonObject { ["kind"] = kind, ["detail"] = detail, ["count"] = count, ["pages"] = new JsonArray([.. pages.Take(20).Select(p => (JsonNode)(p + 1))]) });
            }

            var written = new JsonObject();
            foreach (var (step, result) in file.WrittenBack)
            {
                written[step] = result.DeepClone();
            }

            files.Add(new JsonObject
            {
                ["file"] = file.Name, ["bytes"] = file.Bytes, ["pages"] = file.Opened == true ? file.Pages : null,
                ["version"] = file.Version, ["encrypted"] = file.Encrypted, ["repaired"] = file.Repaired, ["password"] = file.Password is not null,
                ["skipped"] = file.Skipped, ["notOpened"] = file.NotOpened,
                ["counts"] = PdfRealCounts(file.Counts), ["annotationKinds"] = PdfRealCounts(file.AnnotationKinds), ["fieldKinds"] = PdfRealCounts(file.FieldKinds),
                ["issues"] = PdfRealCounts(file.IssuesByKind()), ["issueDetails"] = details, ["drawIssues"] = PdfRealCounts(file.DrawIssues),
                ["signatures"] = new JsonArray([.. file.Signatures.Select(s => s.DeepClone())]),
                ["steps"] = steps, ["writtenBack"] = written, ["reference"] = file.Reference?.DeepClone(),
                ["errors"] = errors, ["failures"] = new JsonArray([.. file.Failures.Select(f => (JsonNode?)f)]),
            });
        }

        var totals = PdfRealTotals(run);
        return new JsonObject
        {
            ["schema"] = 1, ["folder"] = run.Folder, ["started"] = run.Started.ToString("u", CultureInfo.InvariantCulture),
            ["machine"] = new JsonObject { ["cores"] = Environment.ProcessorCount, ["os"] = RuntimeInformation.OSDescription, ["runtime"] = RuntimeInformation.FrameworkDescription },
            ["settings"] = new JsonObject
            {
                ["warmPasses"] = run.Warm, ["dpi"] = run.Draw.Dpi, ["drawnParts"] = run.Draw.Parts.ToString(), ["fonts"] = run.Draw.Fonts, ["reference"] = run.Reference,
                ["savePages"] = run.SavePages, ["saveFiles"] = run.SaveFiles,
                ["limits"] = new JsonObject { ["maxStreamBytes"] = run.Limits.MaxStreamBytes, ["maxObjects"] = run.Limits.MaxObjects, ["maxDepth"] = run.Limits.MaxDepth, ["cacheBytes"] = run.Limits.CacheBytes },
            },
            ["files"] = files, ["totals"] = totals,
        };
    }

    private static JsonObject PdfRealTotals(PdfRealRun run)
    {
        var opened = run.Files.Where(f => f.Opened == true).ToList();
        var counts = new OrderedDictionary<string, long>(StringComparer.Ordinal);
        var issues = new SortedDictionary<string, long>(StringComparer.Ordinal);
        var drawIssues = new SortedDictionary<string, long>(StringComparer.Ordinal);
        var steps = new OrderedDictionary<string, (double Cold, double Warm, int Pages, int Files, int Warmed)>(StringComparer.Ordinal);
        foreach (var file in opened)
        {
            foreach (var (key, value) in file.Counts)
            {
                counts[key] = (counts.TryGetValue(key, out long known) ? known : 0) + value;
            }

            foreach (var (key, value) in file.IssuesByKind())
            {
                issues[key] = (issues.TryGetValue(key, out long known) ? known : 0) + value;
            }

            foreach (var (key, value) in file.DrawIssues)
            {
                drawIssues[key] = (drawIssues.TryGetValue(key, out long known) ? known : 0) + value;
            }

            foreach (var (step, timing) in file.Steps)
            {
                var sum = steps.TryGetValue(step, out var known) ? known : default;
                steps[step] = (sum.Cold + timing.ColdMs, sum.Warm + (timing.WarmMs ?? 0), sum.Pages + file.Pages, sum.Files + 1, sum.Warmed + (timing.WarmMs is null ? 0 : 1));
            }
        }

        var stepsJson = new JsonObject();
        foreach (var (step, sum) in steps)
        {
            int per = Math.Max(1, sum.Pages);
            var json = new JsonObject { ["files"] = sum.Files, ["pages"] = sum.Pages, ["coldMs"] = Math.Round(sum.Cold, 3), ["coldMsPerPage"] = Math.Round(sum.Cold / per, 3) };
            if (sum.Warmed == sum.Files)
            {
                json["warmMs"] = Math.Round(sum.Warm, 3);
                json["warmMsPerPage"] = Math.Round(sum.Warm / per, 3);
            }

            stepsJson[step] = json;
        }

        var signatures = run.Files.SelectMany(f => f.Signatures).ToList();
        var written = new JsonObject();
        foreach (string step in new[] { "rewrite", "incremental" })
        {
            var results = run.Files.Where(f => f.WrittenBack.ContainsKey(step)).Select(f => f.WrittenBack[step]).ToList();
            written[step] = new JsonObject { ["files"] = results.Count, ["readTheSame"] = results.Count(r => r["readsTheSame"]?.GetValue<bool>() == true) };
        }

        var compared = run.Files.Where(f => f.Reference is not null).Select(f => f.Reference!).ToList();
        int Sum(string key) => compared.Sum(r => r[key]?.GetValue<int>() ?? 0);
        return new JsonObject
        {
            ["files"] = run.Files.Count, ["opened"] = opened.Count, ["skipped"] = run.Files.Count(f => f.Skipped is not null),
            ["notOpened"] = run.Files.Count(f => f.NotOpened is not null), ["failed"] = run.Files.Count(f => f.Failures.Count > 0),
            ["pages"] = opened.Sum(f => f.Pages), ["bytes"] = run.Files.Sum(f => f.Bytes),
            ["counts"] = PdfRealCounts(counts), ["issues"] = PdfRealCounts(issues), ["drawIssues"] = PdfRealCounts(drawIssues),
            ["signatures"] = new JsonObject
            {
                ["count"] = signatures.Count, ["intact"] = signatures.Count(s => s["intact"]?.GetValue<bool>() == true), ["valid"] = signatures.Count(s => s["valid"]?.GetValue<bool>() == true),
            },
            ["steps"] = stepsJson, ["writtenBack"] = written,
            ["reference"] = new JsonObject
            {
                ["files"] = compared.Count, ["pages"] = Sum("pages"), ["same"] = Sum("same"), ["spacing"] = Sum("spacing"), ["order"] = Sum("order"), ["drawnOrder"] = Sum("drawnOrder"),
                ["words"] = Sum("words"), ["notRead"] = Sum("notRead"),
            },
            ["errors"] = run.Files.Sum(f => f.Errors.Count), ["failures"] = run.Files.Sum(f => f.Failures.Count),
        };
    }

    private static string PdfRealSummaryText(PdfRealRun run)
    {
        var text = new StringBuilder();
        void Line(FormattableString line) => text.Append(line.ToString(CultureInfo.InvariantCulture)).Append('\n');
        static string Sizes(long bytes) => bytes >= 1L << 30 ? string.Create(CultureInfo.InvariantCulture, $"{bytes / (double)(1L << 30):F1} GiB")
            : bytes >= 1L << 20 ? string.Create(CultureInfo.InvariantCulture, $"{bytes / (double)(1L << 20):F1} MiB")
            : string.Create(CultureInfo.InvariantCulture, $"{bytes:N0} bytes");
        static string Of(IEnumerable<KeyValuePair<string, long>> counts) =>
            counts.Any() ? string.Join(", ", counts.Select(c => string.Create(CultureInfo.InvariantCulture, $"{c.Key} {c.Value:N0}"))) : "none";
        long Count(PdfRealFile file, string key) => file.Counts.TryGetValue(key, out long value) ? value : 0;
        static string N(long count, string one) => string.Create(CultureInfo.InvariantCulture, $"{count:N0} {one}{(count == 1 ? "" : "s")}");
        static string Listed(JsonNode? counts) => counts is JsonObject { Count: > 0 } all ? string.Join(", ", all.Select(c => $"{c.Key} {c.Value}")) : "none";

        var limits = run.Limits;
        Line($"documents real: {run.Files.Count} PDF files in {run.Folder} ({run.Started:u}; {Environment.ProcessorCount} cores, {RuntimeInformation.OSDescription})");
        Line($"Read within the default limits, measured once when the run started: streams up to {Sizes(limits.MaxStreamBytes)} decoded, {limits.MaxObjects:N0} objects, nesting {limits.MaxDepth}, a cache of {Sizes(limits.CacheBytes)}.");
        Line($"Pages drawn at {run.Draw.Dpi:0.#} dpi ({run.Draw.Parts}); fonts for text not embedded: {run.Draw.Fonts ?? "none given (such text is left out and counted)"}.");
        Line($"Times in milliseconds: cold (the file's first pass of the step in this run; the first file's also compiles the code) / warm (the median of {run.Warm} passes), then a page.");
        Line($"Reference text: {run.Reference ?? "none (NAME.reference.txt beside a file only)"}.");
        text.Append('\n');
        foreach (var file in run.Files)
        {
            if (file.Opened != true)
            {
                string why = file.Skipped is { } skipped ? $"skipped: {skipped}" : file.NotOpened is { } reason ? $"does not open: {reason}" : "not opened";
                Line($"{file.Name}: {Sizes(file.Bytes)}; {why}");
            }
            else
            {
                var flags = new List<string> { $"PDF {file.Version}" };
                if (file.Encrypted)
                {
                    flags.Add(file.Password is null ? "encrypted (opened with the empty password)" : "encrypted (opened with its password)");
                }

                if (file.Repaired)
                {
                    flags.Add("repaired");
                }

                Line($"{file.Name}: {N(file.Pages, "page")}, {Sizes(file.Bytes)}, {string.Join(", ", flags)}");
                Line($"  text: {N(Count(file, "runs"), "run")} ({Count(file, "hiddenRuns"):N0} hidden), {N(Count(file, "lines"), "line")}, {N(Count(file, "characters"), "character")}; tagged: {N(Count(file, "taggedRuns"), "run")} tagged, {N(Count(file, "taggedPagesInAnotherOrder"), "page")} in another order");
                Line($"  parts: {N(Count(file, "images"), "image")} ({Count(file, "imagesDecoded"):N0} decoded, {Count(file, "imagesNotDecoded"):N0} not, {Count(file, "imagePixels") / 1e6:F1} megapixels, {N(Count(file, "masks"), "mask")}), {N(Count(file, "paths"), "path")}, {N(Count(file, "clips"), "clip")}, {N(Count(file, "shadings"), "shading")}, {N(Count(file, "annotations"), "annotation")}{(file.AnnotationKinds.Count > 0 ? $" ({Of(file.AnnotationKinds)})" : "")}");
                Line($"  document: {N(Count(file, "fields"), "field")}{(file.FieldKinds.Count > 0 ? $" ({Of(file.FieldKinds)})" : "")}, {N(Count(file, "bookmarks"), "bookmark")}, {N(Count(file, "namedDestinations"), "named destination")}, {N(Count(file, "labelledPages"), "page")} labelled, {N(Count(file, "attachments"), "attachment")} ({Sizes(Count(file, "attachmentBytes"))}), {N(Count(file, "signatures"), "signature")}");
                foreach (var signature in file.Signatures)
                {
                    var failed = (signature["steps"] as JsonObject)?.Where(s => s.Value?["passed"]?.GetValue<bool>() == false).Select(s => $"{s.Key}: {s.Value?["detail"]}").ToList() ?? [];
                    Line($"    signature {signature["field"]} ({signature["kind"]}, {signature["subFilter"]}): {((bool)signature["intact"]! ? "intact" : "NOT intact")}, {((bool)signature["coversWholeFile"]! ? "covers the whole file" : $"changes after it {signature["changes"]}")}{(failed.Count > 0 ? $"; not passed: {string.Join("; ", failed)}" : "")}");
                }

                if (file.Steps.ContainsKey("draw"))
                {
                    Line($"  drawn: {N(Count(file, "pagesDrawn"), "page")}, {Count(file, "drawnPixels") / 1e6:F1} megapixels; drawing issues: {Of(file.DrawIssues)}");
                }

                Line($"  issues: {Of(file.IssuesByKind())}");
                foreach (var (kind, detail, count, pages) in file.IssueDetails().Take(10))
                {
                    Line($"    {kind}: {detail} ({count:N0}; page {string.Join(", ", pages.Take(10).Select(p => p + 1))}{(pages.Count > 10 ? ", ..." : "")})");
                }

                if (file.Steps.Count > 0)
                {
                    int per = Math.Max(1, file.Pages);
                    Line($"  times: {string.Join("; ", file.Steps.Select(s => string.Create(CultureInfo.InvariantCulture,
                        $"{s.Key} {s.Value.ColdMs:F2} / {(s.Value.WarmMs is { } w ? w.ToString("F2", CultureInfo.InvariantCulture) : "-")} ({s.Value.ColdMs / per:F2} / {(s.Value.WarmMs is { } wp ? (wp / per).ToString("F2", CultureInfo.InvariantCulture) : "-")} a page)")))}");
                }

                foreach (var (step, result) in file.WrittenBack)
                {
                    bool same = result["readsTheSame"]?.GetValue<bool>() == true;
                    string extra = result["appended"] is { } appended
                        ? string.Create(CultureInfo.InvariantCulture, $" ({appended.GetValue<long>():N0} appended{(result["sourceKept"]?.GetValue<bool>() == true ? ", the source its prefix" : "")}{(result["signaturesKept"] is { } kept ? kept.GetValue<bool>() ? ", the signatures' digests kept" : ", the signatures NOT kept" : "")})")
                        : "";
                    Line($"  written back as {step}: {Sizes(result["bytes"]!.GetValue<long>())}{extra}, {(same ? "reads the same" : "does NOT read the same")}");
                }

                if (file.Reference is { } reference)
                {
                    Line($"  reference: {N(reference["pages"]!.GetValue<int>(), "page")}: {reference["same"]} the same, {reference["spacing"]} in spacing only, {reference["order"]} in order only, {reference["drawnOrder"]} with words as drawn, {reference["words"]} in words, {reference["notRead"]} not compared");
                    foreach (var difference in (reference["differences"] as JsonArray ?? new JsonArray()).OfType<JsonObject>().Where(d => (string?)d["kind"] == "words").Take(5))
                    {
                        Line($"    page {difference["page"]}: {(double)difference["wordsInCommon"]! * 100:F1}% of the words in common ({difference["wordsAsDrawn"]} as drawn); missing {difference["missing"]!.ToJsonString(PdfRealJsonLine)}; extra {difference["extra"]!.ToJsonString(PdfRealJsonLine)}");
                    }
                }
            }

            foreach (var group in file.Errors.GroupBy(e => (e.Step, e.Error)).Take(10))
            {
                var pages = group.Where(e => e.Page >= 0).Select(e => e.Page + 1).Distinct().Order().ToList();
                Line($"  reported ({group.Key.Step}{(pages.Count > 0 ? $", page {string.Join(", ", pages.Take(10))}{(pages.Count > 10 ? ", ..." : "")}" : "")}): {group.Key.Error}");
            }

            foreach (string failure in file.Failures)
            {
                Line($"  FAILED {failure}");
            }
        }

        var totals = PdfRealTotals(run);
        text.Append('\n');
        Line($"Totals: {totals["files"]} files: {totals["opened"]} opened, {totals["skipped"]} skipped (a password), {totals["notOpened"]} not opened (their reasons above); {totals["failed"]} failed; {totals["pages"]} pages, {Sizes(totals["bytes"]!.GetValue<long>())}");
        Line($"  counts: {Listed(totals["counts"])}");
        Line($"  issues: {Listed(totals["issues"])}; drawing issues: {Listed(totals["drawIssues"])}");
        var signed = totals["signatures"]!;
        Line($"  signatures: {signed["count"]}, {signed["intact"]} intact (digest and signature), {signed["valid"]} valid (nothing is trusted here, so the chain fails unless the app gives trust)");
        foreach (var (step, node) in totals["steps"] as JsonObject ?? new JsonObject())
        {
            if (node is not JsonObject sum)
            {
                continue;
            }

            Line($"  {step}: {sum["files"]} files, {sum["pages"]} pages: cold {sum["coldMs"]} ms ({sum["coldMsPerPage"]} a page), warm {sum["warmMs"]?.ToString() ?? "-"} ms ({sum["warmMsPerPage"]?.ToString() ?? "-"} a page)");
        }

        var writtenBack = totals["writtenBack"]!;
        Line($"  written back: rewrite {writtenBack["rewrite"]!["readTheSame"]} of {writtenBack["rewrite"]!["files"]} read the same, incremental {writtenBack["incremental"]!["readTheSame"]} of {writtenBack["incremental"]!["files"]}");
        var compared = totals["reference"]!;
        Line($"  reference: {compared["files"]} files, {compared["pages"]} pages: {compared["same"]} the same, {compared["spacing"]} in spacing only, {compared["order"]} in order only, {compared["drawnOrder"]} with words as drawn, {compared["words"]} in words, {compared["notRead"]} not compared");
        Line($"  {totals["errors"]} reported failures (each with its reason, above), {totals["failures"]} failures");
        return text.ToString();
    }

    private static readonly JsonSerializerOptions PdfRealJsonLine = new() { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };
}
