// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Text.Json.Nodes;

namespace Idrak.Cli.Commands.Measure;

/// <summary>
/// A set of documents <c>idrak bench documents</c> runs on: its name, its folder, its PDF files in order, and what it
/// learns while it runs (the files no format opens, each file's page count, the files the signing case wrote).
/// </summary>
internal sealed class DocumentBenchSet(string name, string root, IReadOnlyList<string> files, int leftOut)
{
    public string Name { get; } = name;

    public string Root { get; } = root;

    public IReadOnlyList<string> Files { get; } = files;

    /// <summary>The files the folder's manifest marks hostile (made to fail), left out.</summary>
    public int LeftOut { get; } = leftOut;

    /// <summary>The files no format opens (a password, a file damaged beyond repair), with why: found by the first case that opens them, then left out of every run.</summary>
    public Dictionary<string, string> Refused { get; } = new(StringComparer.Ordinal);

    /// <summary>Each file's page count, once a case has opened it.</summary>
    public Dictionary<string, int> Pages { get; } = new(StringComparer.Ordinal);

    /// <summary>What the signing case's last run wrote (each file signed), for the checking case.</summary>
    public List<(string File, byte[] Bytes)>? Signed { get; set; }

    /// <summary>
    /// The files a case takes: at most <paramref name="most"/> of the set's files spread evenly over them (0: all), less
    /// those refused. The spread is over every file, refused or not, so every run of every case at one cap takes the same
    /// files, whatever an earlier run found it could not open (a spread over the files still open moved with each refusal,
    /// and the runs of one case read other files).
    /// </summary>
    public IReadOnlyList<string> Take(int most) => [.. DocumentBench.Spread(Files, most).Where(f => !Refused.ContainsKey(f))];

    /// <summary>A file's path under the set's folder, with / between folders (the key of its output in the results, the same on every system).</summary>
    public string Relative(string file) => Path.GetRelativePath(Root, file).Replace('\\', '/');
}

/// <summary>What <c>idrak bench documents</c> runs: the cases, the drawing resolutions, the caps and the pieces some cases need.</summary>
internal sealed record DocumentBenchOptions
{
    /// <summary>The cases, in the order they run (draw stands for one case a resolution).</summary>
    public required IReadOnlyList<string> Cases { get; init; }

    /// <summary>The resolutions pages are drawn at, in dots per inch.</summary>
    public required IReadOnlyList<float> Dpis { get; init; }

    /// <summary>The warm runs of each case after its cold one.</summary>
    public int Repeats { get; init; } = 3;

    /// <summary>The most files a drawing case takes from a set, spread evenly (0: all).</summary>
    public int DrawFiles { get; init; } = 50;

    /// <summary>The most pages a drawing case draws of a file (0: all).</summary>
    public int DrawPages { get; init; } = 10;

    /// <summary>The most files every case takes from a set (0: all): the short mode's few.</summary>
    public int Files { get; init; }

    /// <summary>The most pages every case takes of a file (0: all): the short mode's few.</summary>
    public int Pages { get; init; }

    /// <summary>
    /// Whether every run's output (text, pixels, written bytes) is hashed (SHA-256), file by file, so runs and saved results
    /// show the same output; an encrypted output by what it reads back to (<see cref="DocumentBenchReadBack"/>), not its bytes.
    /// </summary>
    public bool Hash { get; init; }

    /// <summary>
    /// The options every run of every case opens files with: the library's settings as the environment gives them, each
    /// limit measured from the memory pinned once for the whole bench run (<see cref="DocumentBench.Pin"/>), so no run
    /// reads under limits another did not.
    /// </summary>
    public required DocumentOptions Read { get; init; }

    /// <summary>The face text whose font a file does not embed is written in by the vector case, or null (such text is left out).</summary>
    public IFontFace? Font { get; init; }

    /// <summary>How the signing case signs (the generator's test certificate, never a real one), or null when no certificate was found.</summary>
    public SigningOptions? Signing { get; init; }

    /// <summary>The root the checking case trusts.</summary>
    public X509Certificate2? Trust { get; init; }

    /// <summary>When the checking case checks the chain (inside the test certificate's validity).</summary>
    public DateTimeOffset? CheckAt { get; init; }
}

/// <summary>One run of a case: the time (refused files' time left out), what it allocated, the process's working set at its highest, the JIT's share, and what it did.</summary>
internal sealed record DocumentBenchRun(double Milliseconds, long Allocated, long PeakWorkingSet, double JitMilliseconds, long JitMethods, DocumentBenchWork Work);

/// <summary>A file whose output differs between the runs of one case: its output in each run, cold first (null where a run did not read it: refused, or not taken).</summary>
internal sealed record DocumentBenchDifference(string File, IReadOnlyList<string?> Outputs);

/// <summary>
/// What one run of a case did: its units (the case's own: files, pages, signatures), the other counts, the bytes written
/// and the output's hash, whole and file by file. A file's work hashes what it adds; a run's hashes its files' hashes in
/// order (<see cref="Finish"/>), an encrypted output's taken from what it reads back to once the run is measured
/// (<see cref="ReadBack"/>).
/// </summary>
internal sealed class DocumentBenchWork
{
    private readonly IncrementalHash? _hash;

    // A file's work: its output (written encrypted), kept to be hashed by what it reads back to in place of its bytes.
    private ReadOnlyMemory<byte>? _readBack;

    // A run's work: the outputs kept to be read back, each with its place in FileOutputs.
    private List<(int At, string File, ReadOnlyMemory<byte> Output)>? _readBacks;

    public DocumentBenchWork(bool hash) => _hash = hash ? IncrementalHash.CreateHash(HashAlgorithmName.SHA256) : null;

    public long Units { get; set; }

    public long Files { get; set; }

    public long Pages { get; set; }

    /// <summary>The case's own items: images, fields, signatures, parts written.</summary>
    public long Items { get; set; }

    public long Pixels { get; set; }

    public long Written { get; set; }

    /// <summary>Parts a writing case left out (text without a face to write it in, a clip of text, a part the writer refused).</summary>
    public long LeftOut { get; set; }

    /// <summary>The time spent on files refused during the run and on keeping outputs to read back (left out of its time), in timestamp ticks.</summary>
    public long ExcludedTicks { get; set; }

    /// <summary>The bytes allocated keeping outputs to read back (left out of the run's allocation, as their time is of its time).</summary>
    public long ExcludedBytes { get; set; }

    /// <summary>The files whose output was hashed by what it reads back to (encrypted outputs), in the order they went through; filled by <see cref="ReadBack"/>.</summary>
    public List<string> ReadBackFiles { get; } = [];

    /// <summary>The files refused during the run, with why.</summary>
    public List<(string File, string Reason)> Refused { get; } = [];

    /// <summary>Each file's output (its SHA-256, hex) in the order the files went through; hashed runs only. The run's output hashes these in order.</summary>
    public List<(string File, string Output)> FileOutputs { get; } = [];

    /// <summary>The files that met one of the library's limits in the run: refused by one, or a part of a page left out naming one.</summary>
    public HashSet<string> LimitFiles { get; } = new(StringComparer.Ordinal);

    /// <summary>Whether this file's work met one of the library's limits (a page's issue, or a part's error, naming one).</summary>
    public bool MetLimit { get; set; }

    /// <summary>The output's SHA-256 (hex), once <see cref="Finish"/> ran; null without hashing.</summary>
    public string? Output { get; private set; }

    public bool Hashing => _hash is not null;

    /// <summary>Notes a page's or a drawing's issues: one that names a limit marks the work as having met it.</summary>
    public void Note(IReadOnlyList<DocumentIssue> issues)
    {
        foreach (var issue in issues)
        {
            MetLimit |= DocumentBench.NamesLimit(issue.Detail);
        }
    }

    /// <summary>Notes a file refused during the run (its time left out), with why; one refused by a limit, or after meeting one, also met it.</summary>
    public void Refuse(string file, string reason, bool metLimit = false)
    {
        Refused.Add((file, reason));
        if (metLimit || DocumentBench.NamesLimit(reason))
        {
            LimitFiles.Add(file);
        }
    }

    public void Add(ReadOnlySpan<byte> bytes) => _hash?.AppendData(bytes);

    public void Add(string text)
    {
        if (_hash is null)
        {
            return;
        }

        Span<byte> small = stackalloc byte[256];
        int count = Encoding.UTF8.GetMaxByteCount(text.Length);
        if (count <= small.Length)
        {
            _hash.AppendData(small[..Encoding.UTF8.GetBytes(text, small)]);
        }
        else
        {
            _hash.AppendData(Encoding.UTF8.GetBytes(text));
        }

        _hash.AppendData("\n"u8);
    }

    public void Add(long value)
    {
        Span<byte> bytes = stackalloc byte[8];
        BinaryPrimitives.WriteInt64LittleEndian(bytes, value);
        _hash?.AppendData(bytes);
    }

    /// <summary>An image's pixels: its 8-bit planes when it holds them, else its values.</summary>
    public void Add(ImageData image)
    {
        Pixels += (long)image.Width * image.Height;
        if (_hash is null)
        {
            return;
        }

        Add(image.Channels);
        Add(image.Height);
        Add(image.Width);
        if (image.TryGetPlanes(out var planes, out _))
        {
            foreach (var plane in planes)
            {
                _hash.AppendData(plane.Span);
            }
        }
        else
        {
            _hash.AppendData(MemoryMarshal.AsBytes(image.Pixels.AsSpan()));
        }
    }

    /// <summary>
    /// Keeps <paramref name="output"/> (this file's whole output, written encrypted) to be hashed by what it reads back to
    /// once the run is measured, in place of its bytes (an encrypted save repeats nothing byte for byte: a fresh random IV).
    /// </summary>
    public void ReadBackLater(ReadOnlyMemory<byte> output) => _readBack = output;

    /// <summary>Takes <paramref name="file"/>'s work in (its counts, whether it met a limit and, hashed, its output's hash, kept under its name too).</summary>
    public void Merge(string file, DocumentBenchWork part)
    {
        Units += part.Units;
        Files += part.Files;
        Pages += part.Pages;
        Items += part.Items;
        Pixels += part.Pixels;
        Written += part.Written;
        LeftOut += part.LeftOut;
        ExcludedTicks += part.ExcludedTicks;
        ExcludedBytes += part.ExcludedBytes;
        if (part.MetLimit)
        {
            LimitFiles.Add(file);
        }

        if (_hash is not null && part._hash is not null)
        {
            if (part._readBack is { } output)
            {
                (_readBacks ??= []).Add((FileOutputs.Count, file, output));
                FileOutputs.Add((file, ""));     // its hash once read back
            }
            else
            {
                Span<byte> digest = stackalloc byte[SHA256.HashSizeInBytes];
                part._hash.GetHashAndReset(digest);
                FileOutputs.Add((file, Convert.ToHexStringLower(digest)));
            }
        }

        part._hash?.Dispose();
    }

    /// <summary>
    /// Hashes each output kept to be read back (<see cref="ReadBackLater"/>) by what it reads back to
    /// (<see cref="DocumentBenchReadBack"/>), opened with <paramref name="read"/> (the options, and so the password, the case
    /// opened its source with). Called once the run is measured, so the reading is in neither its time, its allocation nor
    /// its working set; an output that opens unencrypted keeps its bytes' hash.
    /// </summary>
    public void ReadBack(DocumentOptions read)
    {
        if (_readBacks is null)
        {
            return;
        }

        foreach (var (at, file, output) in _readBacks)
        {
            var (hash, readBack) = DocumentBenchReadBack.Hash(output, file, read);
            FileOutputs[at] = (file, hash);
            if (readBack)
            {
                ReadBackFiles.Add(file);
            }
        }

        _readBacks = null;
    }

    /// <summary>
    /// Ends the work: its output's hash taken (once), the hash released. A file's is the hash of what it added; a run's
    /// the hash of its files' hashes in the order they went through (once every kept output was read back).
    /// </summary>
    public void Finish()
    {
        if (_hash is null || Output is not null)
        {
            return;
        }

        if (_readBacks is { Count: > 0 })
        {
            throw new InvalidOperationException($"{_readBacks.Count} outputs kept to be read back were not: ReadBack comes before Finish.");
        }

        Span<byte> digest = stackalloc byte[SHA256.HashSizeInBytes];
        foreach (var (_, output) in FileOutputs)
        {
            Convert.FromHexString(output, digest, out _, out _);
            _hash.AppendData(digest);
        }

        Output = Convert.ToHexStringLower(_hash.GetHashAndReset());
        _hash.Dispose();
    }
}

/// <summary>
/// The cases of <c>idrak bench documents</c>, through the library's public document API only (any format,
/// writer and drawer): opening, text through the <c>lines</c> layout and with none, the same text read over the cores,
/// images decoded, pages drawn at chosen resolutions, an image PDF and a vector PDF written from each file's pages,
/// incremental and rewritten saves, forms filled, signing and checking. Each case runs once cold (its first run on the
/// set; the first set's also compiles the case's code) and <see cref="DocumentBenchOptions.Repeats"/> times warm, each
/// run after a full collection, timed, its managed allocation counted on every thread and the process's working set
/// sampled through it (rules 75, 76).
/// </summary>
internal static class DocumentBench
{
    /// <summary>The cases, in the order they run.</summary>
    public static readonly string[] Cases =
        ["open", "text-lines", "text-content", "text-parallel", "images", "draw", "write-image", "write-vector", "edit-incremental", "edit-rewrite", "fill", "sign", "check"];

    /// <summary>The method saved results record (<c>timing</c>), so a comparison can tell results timed another way.</summary>
    public const string Method = "documents-cold-and-median";

    private const string SignatureField = "BenchSignature";

    // Parts of a page in their content order.
    private static readonly Comparer<(int Order, int Path, int Run)> ByOrder = Comparer<(int Order, int Path, int Run)>.Create(static (a, b) => a.Order.CompareTo(b.Order));

    // The library's limits as their messages name them ("... the limit MaxStreamBytes (DocumentOptions.MaxStreamBytes, or
    // IDRAK_DOCUMENT_MAX_STREAM_BYTES)"): a refusal or an issue naming one met that limit.
    private static readonly SearchValues<string> LimitNames = SearchValues.Create(
        [$"DocumentOptions.{nameof(DocumentOptions.MaxStreamBytes)}", $"DocumentOptions.{nameof(DocumentOptions.MaxObjects)}", $"DocumentOptions.{nameof(DocumentOptions.MaxDepth)}", $"DocumentOptions.{nameof(DocumentOptions.CacheBytes)}"],
        StringComparison.Ordinal);

    /// <summary>
    /// <paramref name="options"/> with each limit measured from the memory (<see cref="DocumentOptions.MaxStreamBytes"/>,
    /// <see cref="DocumentOptions.MaxObjects"/>, <see cref="DocumentOptions.CacheBytes"/>) resolved once, now: the value its
    /// option or variable sets, else its share of the memory free at this moment. Every document opened with the result
    /// reads under the same limits; the options themselves measure again at each open, so a file near a limit (a flate
    /// bomb, a huge stream) would be refused or read by what the machine had free at that moment.
    /// </summary>
    public static DocumentOptions Pin(DocumentOptions options) =>
        options with { MaxStreamBytes = options.MaxStreamBytes, MaxObjects = options.MaxObjects, CacheBytes = options.CacheBytes };

    /// <summary>Whether <paramref name="text"/> (a refusal's reason, an issue's detail) names one of the library's limits.</summary>
    public static bool NamesLimit(string text) => text.AsSpan().ContainsAny(LimitNames);

    /// <summary>
    /// The files whose output differs between the runs of one case (hashed runs only), in the order the runs first met
    /// them: a file whose hash is not the same in every run, or that one run read and another did not. Empty when every
    /// run gave the same output file by file.
    /// </summary>
    public static List<DocumentBenchDifference> Differences(DocumentBenchRun cold, IReadOnlyList<DocumentBenchRun> warm)
    {
        int runs = 1 + warm.Count;
        var outputs = new Dictionary<string, string?[]>(StringComparer.Ordinal);
        var order = new List<string>();
        for (int r = 0; r < runs; r++)
        {
            foreach (var (file, output) in (r == 0 ? cold : warm[r - 1]).Work.FileOutputs)
            {
                if (!outputs.TryGetValue(file, out var row))
                {
                    outputs[file] = row = new string?[runs];
                    order.Add(file);
                }

                row[r] = output;
            }
        }

        var differing = new List<DocumentBenchDifference>();
        foreach (string file in order)
        {
            var row = outputs[file];
            if (Array.Exists(row, o => o != row[0]))
            {
                differing.Add(new DocumentBenchDifference(file, row));
            }
        }

        return differing;
    }

    /// <summary>What each case's unit is.</summary>
    public static string Unit(string @case) => @case switch
    {
        "open" or "edit-incremental" or "edit-rewrite" or "sign" => "file",
        "fill" => "form",
        "check" => "signature",
        _ => "page",
    };

    /// <summary>At most <paramref name="most"/> of <paramref name="items"/> spread evenly over them, in order (0 or fewer: all).</summary>
    public static IReadOnlyList<T> Spread<T>(IReadOnlyList<T> items, int most)
    {
        if (most <= 0 || items.Count <= most)
        {
            return items;
        }

        var taken = new T[most];
        for (int i = 0; i < most; i++)
        {
            taken[i] = items[(int)((long)i * items.Count / most)];
        }

        return taken;
    }

    /// <summary>
    /// Runs <paramref name="case"/> once on <paramref name="set"/> (at <paramref name="dpi"/> for a drawing case),
    /// measured: a full collection first, then the time, the bytes allocated on every thread, the JIT's share and the
    /// working set sampled while it runs. Hashed, an encrypted output is read back and hashed after all of it is measured.
    /// </summary>
    public static DocumentBenchRun Measure(string @case, float dpi, DocumentBenchSet set, DocumentBenchOptions options, Action<long> progress)
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        var jitTime = System.Runtime.JitInfo.GetCompilationTime(currentThread: false);
        long jitMethods = System.Runtime.JitInfo.GetCompiledMethodCount(currentThread: false);
        using var sampler = WorkingSetSampler.Start();
        long allocated = GC.GetTotalAllocatedBytes(precise: true);
        long start = Stopwatch.GetTimestamp();
        var work = Run(@case, dpi, set, options, progress);
        long ticks = Stopwatch.GetTimestamp() - start - work.ExcludedTicks;
        allocated = GC.GetTotalAllocatedBytes(precise: true) - allocated;
        var (peak, samplerAllocated) = sampler.Stop();
        double jitMs = (System.Runtime.JitInfo.GetCompilationTime(currentThread: false) - jitTime).TotalMilliseconds;
        jitMethods = System.Runtime.JitInfo.GetCompiledMethodCount(currentThread: false) - jitMethods;
        work.ReadBack(options.Read);     // outside everything measured above
        work.Finish();
        return new DocumentBenchRun(Math.Max(0, ticks) * 1000.0 / Stopwatch.Frequency, Math.Max(0, allocated - samplerAllocated - work.ExcludedBytes), peak, jitMs, jitMethods, work);
    }

    /// <summary>The title the editing cases write over each file's (and its subject below): a change every edit makes.</summary>
    public const string EditedTitle = "Edited copy";

    /// <summary>The subject the editing cases write.</summary>
    public const string EditedSubject = "A copy saved by the benchmark";

    // Whether a case that saves `document` again keeps its output to be read back: hashed, and the source encrypted (an
    // editor keeps a file's own encryption, so the output is written encrypted, with a fresh random IV every time).
    private static bool ReadsBack(IDocument document, DocumentBenchWork work) => work.Hashing && document.Info.Encrypted;

    private static DocumentBenchWork Run(string @case, float dpi, DocumentBenchSet set, DocumentBenchOptions options, Action<long> progress)
    {
        var work = new DocumentBenchWork(options.Hash);
        var read = options.Read;
        switch (@case)
        {
            case "open":
                EachFile(set, set.Take(options.Files), work, (file, w) =>
                {
                    using var document = Open(set, file, read);
                    w.Units = w.Files = 1;
                    w.Pages = document.PageCount;
                    w.Add(document.PageCount);
                    w.Add(document.Info.Title ?? "");
                    w.Add(document.Info.Version ?? "");
                    progress(1);
                });
                break;
            case "text-lines" or "text-content":
                var layout = read with { Layout = @case == "text-lines" ? DocumentTextLayouts.Lines : DocumentTextLayouts.Content };
                EachFile(set, set.Take(options.Files), work, (file, w) =>
                {
                    using var document = Open(set, file, layout);
                    w.Files = 1;
                    foreach (int page in PagesOf(document, options.Pages))
                    {
                        var pageText = document.ReadPage(page, DocumentParts.Text);
                        w.Add(pageText.Text);
                        w.Note(pageText.Issues);
                        w.Units++;
                        w.Pages++;
                        progress(1);
                    }
                });
                break;
            case "text-parallel":
                TextParallel(set, options, work, read with { Layout = DocumentTextLayouts.Lines }, progress);
                break;
            case "images":
                EachFile(set, set.Take(options.Files), work, (file, w) =>
                {
                    using var document = Open(set, file, read);
                    w.Files = 1;
                    foreach (int page in PagesOf(document, options.Pages))
                    {
                        var pageImages = document.ReadPage(page, DocumentParts.Images);
                        w.Note(pageImages.Issues);
                        foreach (var image in pageImages.Images)
                        {
                            w.Add(NotOnTheNewDesign.Yet<ImageData>("Decoding a document's image through the registered image codecs"));
                            w.Items++;
                        }

                        w.Units++;
                        w.Pages++;
                        progress(1);
                    }
                });
                break;
            case "draw":
                var draw = new DocumentDrawOptions { Dpi = dpi };
                int drawPages = options.Pages > 0 ? options.Pages : options.DrawPages;
                EachFile(set, set.Take(options.Files > 0 ? options.Files : options.DrawFiles), work, (file, w) =>
                {
                    using var document = Open(set, file, read);
                    w.Files = 1;
                    foreach (int page in PagesOf(document, drawPages))
                    {
                        var drawing = document.DrawPage(page, draw);
                        w.Add(drawing.Image);
                        w.Items += drawing.Issues.Count;
                        w.Note(drawing.Issues);
                        w.Units++;
                        w.Pages++;
                        progress(1);
                    }
                });
                break;
            case "write-image" or "write-vector":
                bool images = @case == "write-image";
                var writing = new DocumentWriteOptions();
                EachFile(set, set.Take(options.Files), work, (file, w) =>
                {
                    using var document = Open(set, file, read);
                    using var destination = new CountingStream(w);
                    using (var builder = NotOnTheNewDesign.Yet<IDocumentWriter>($"Picking the document writer '{document.Format}' by its name").Create(destination, writing))
                    {
                        foreach (int page in PagesOf(document, options.Pages))
                        {
                            var source = document.ReadPage(page, images ? DocumentParts.Images : DocumentParts.Paths | DocumentParts.Text | DocumentParts.Glyphs);
                            w.Note(source.Issues);
                            if (images)
                            {
                                WriteImages(source, builder, writing.ImageEncoding, w);
                            }
                            else
                            {
                                WriteVector(source, builder, options.Font, w);
                            }

                            w.Units++;
                            w.Pages++;
                            progress(1);
                        }

                        builder.Finish();
                    }

                    w.Files = 1;
                    w.Written = destination.Length;
                });
                break;
            case "edit-incremental" or "edit-rewrite":
                var mode = @case == "edit-incremental" ? DocumentSaveMode.Incremental : DocumentSaveMode.Rewrite;
                EachFile(set, set.Take(options.Files), work, (file, w) =>
                {
                    using var document = Open(set, file, read);
                    using var destination = new CountingStream(w, keep: ReadsBack(document, w));
                    using (var editor = NotOnTheNewDesign.Yet<IDocumentWriter>($"Picking the document writer '{document.Format}' by its name").Edit(document, destination, new DocumentWriteOptions()))
                    {
                        editor.SetInfo(document.Info with { Title = EditedTitle, Subject = EditedSubject });
                        editor.Save(mode);
                    }

                    destination.ReadBackLater();
                    w.Units = w.Files = 1;
                    w.Pages = document.PageCount;
                    w.Written = destination.Length;
                    progress(1);
                });
                break;
            case "fill":
                EachFile(set, set.Take(options.Files), work, (file, w) =>
                {
                    using var document = Open(set, file, read);
                    var fields = document.Fields.Where(f => f.Kind is DocumentFieldKind.Text && (f.Flags & DocumentFieldFlags.ReadOnly) == 0).ToList();
                    if (fields.Count == 0)
                    {
                        return;     // no form: not a unit of this case
                    }

                    using var destination = new CountingStream(w, keep: ReadsBack(document, w));
                    using (var editor = NotOnTheNewDesign.Yet<IDocumentWriter>($"Picking the document writer '{document.Format}' by its name").Edit(document, destination, new DocumentWriteOptions()))
                    {
                        foreach (var field in fields)
                        {
                            const string Value = "Example entry 42";
                            editor.SetField(field.Name, field.MaxLength is { } most && most < Value.Length ? Value[..Math.Max(0, most)] : Value);
                        }

                        editor.Save(DocumentSaveMode.Incremental);
                    }

                    destination.ReadBackLater();
                    w.Units = w.Files = 1;
                    w.Items = fields.Count;
                    w.Written = destination.Length;
                    progress(1);
                });
                break;
            case "sign":
                var signing = options.Signing ?? throw new InvalidOperationException("no certificate to sign with");
                var signed = new List<(string, byte[])>();
                EachFile(set, set.Take(options.Files), work, (file, w) =>
                {
                    using var document = Open(set, file, read);
                    using var destination = new MemoryStream();
                    using (var editor = NotOnTheNewDesign.Yet<IDocumentWriter>($"Picking the document writer '{document.Format}' by its name").Edit(document, destination, new DocumentWriteOptions()))
                    {
                        editor.Sign(SignatureField, signing);
                        editor.Save(DocumentSaveMode.Incremental);
                    }

                    byte[] bytes = destination.ToArray();
                    if (ReadsBack(document, w))
                    {
                        w.ReadBackLater(bytes);     // kept for the checking case anyway
                    }
                    else
                    {
                        w.Add(bytes);
                    }

                    w.Units = w.Files = 1;
                    w.Written = bytes.Length;
                    lock (signed)
                    {
                        signed.Add((file, bytes));
                    }

                    progress(1);
                });
                set.Signed = signed;
                break;
            case "check":
                Check(set, options, work, read, progress);
                break;
            default:
                throw new ArgumentException($"no case '{@case}'", nameof(@case));
        }

        return work;
    }

    // Every file in turn: its work merged when it went through, else the file refused (its time left out of the run's).
    private static void EachFile(DocumentBenchSet set, IReadOnlyList<string> files, DocumentBenchWork work, Action<string, DocumentBenchWork> each)
    {
        foreach (string file in files)
        {
            if (set.Refused.ContainsKey(file))
            {
                continue;
            }

            var part = new DocumentBenchWork(work.Hashing);
            long start = Stopwatch.GetTimestamp();
            try
            {
                each(file, part);
                work.Merge(file, part);
            }
            catch (Exception e) when (e is not OutOfMemoryException)
            {
                work.ExcludedTicks += Stopwatch.GetTimestamp() - start;
                work.ExcludedBytes += part.ExcludedBytes;
                work.Refuse(file, Reason(e), part.MetLimit);
                part.Finish();
            }
        }
    }

    // The file opened; one no format opens is refused for every case from now on.
    private static IDocument Open(DocumentBenchSet set, string file, DocumentOptions options)
    {
        try
        {
            var document = NotOnTheNewDesign.Yet<IDocument>($"Picking the document format of {file}");
            lock (set.Pages)
            {
                set.Pages[file] = document.PageCount;
            }

            return document;
        }
        catch (Exception e) when (e is InvalidDataException or NotSupportedException or IOException or UnauthorizedAccessException)
        {
            lock (set.Refused)
            {
                set.Refused[file] = Reason(e);
            }

            throw;
        }
    }

    private static string Reason(Exception e)
    {
        string message = e.Message.ReplaceLineEndings(" ");
        return e is InvalidDataException or NotSupportedException ? message : $"{e.GetType().Name}: {message}";
    }

    private static IEnumerable<int> PagesOf(IDocument document, int most) => Enumerable.Range(0, most > 0 ? Math.Min(most, document.PageCount) : document.PageCount);

    // The text of every page read over the cores (ComputeResources.ParallelOptions): the files at once when there are as
    // many as the threads, else each file's pages at once; hashed in order afterwards, so the output is the sequential one.
    private static void TextParallel(DocumentBenchSet set, DocumentBenchOptions options, DocumentBenchWork work, DocumentOptions layout, Action<long> progress)
    {
        var files = set.Take(options.Files);
        var texts = new string[files.Count][];
        var refused = new string?[files.Count];
        var limited = new bool[files.Count];     // a page's issue named a limit
        void ReadFile(int f, bool pagesAtOnce)
        {
            try
            {
                using var document = Open(set, files[f], layout);
                var pages = PagesOf(document, options.Pages).ToArray();
                var text = new string[pages.Length];
                void ReadOne(int p)
                {
                    var page = document.ReadPage(pages[p], DocumentParts.Text);
                    text[p] = page.Text;
                    foreach (var issue in page.Issues)
                    {
                        if (NamesLimit(issue.Detail))
                        {
                            Volatile.Write(ref limited[f], true);
                        }
                    }

                    progress(1);
                }

                if (pagesAtOnce)
                {
                    Parallel.For(0, pages.Length, ComputeResources.ParallelOptions, ReadOne);
                }
                else
                {
                    for (int p = 0; p < pages.Length; p++)
                    {
                        ReadOne(p);
                    }
                }

                texts[f] = text;
            }
            catch (Exception e) when (e is not OutOfMemoryException)
            {
                refused[f] = Reason(e);
            }
        }

        if (files.Count >= ComputeResources.MaxCpuThreads)
        {
            Parallel.For(0, files.Count, ComputeResources.ParallelOptions, f => ReadFile(f, pagesAtOnce: false));
        }
        else
        {
            for (int f = 0; f < files.Count; f++)
            {
                ReadFile(f, pagesAtOnce: true);
            }
        }

        for (int f = 0; f < files.Count; f++)
        {
            if (refused[f] is { } reason)
            {
                work.Refuse(files[f], reason, limited[f]);
                continue;
            }

            var part = new DocumentBenchWork(work.Hashing) { Files = 1, MetLimit = limited[f] };
            foreach (string text in texts[f])
            {
                part.Add(text);
                part.Units++;
                part.Pages++;
            }

            work.Merge(files[f], part);
        }
    }

    // The page's images decoded and written as an image page, each in its box.
    private static void WriteImages(DocumentPage page, IDocumentBuilder builder, DocumentImageEncoding encoding, DocumentBenchWork work)
    {
        var target = builder.AddPage(page.Width, page.Height);
        foreach (var image in page.Images)
        {
            try
            {
                var pixels = NotOnTheNewDesign.Yet<ImageData>("Decoding a document's image through the registered image codecs");
                target.Image(pixels, image.Box, encoding);
                work.Items++;
                work.Pixels += (long)pixels.Width * pixels.Height;
            }
            catch (Exception e) when (e is NotSupportedException or InvalidDataException or ArgumentException)
            {
                work.LeftOut++;
                work.MetLimit |= NamesLimit(e.Message);
            }
        }
    }

    // The page's vector content written back in content order: its paths, and its text in the fonts it embeds (else in
    // the bench's face), each under the clips it was drawn under (a clip made of text is left out).
    private static void WriteVector(DocumentPage page, IDocumentBuilder builder, IFontFace? font, DocumentBenchWork work)
    {
        var target = builder.AddPage(page.Width, page.Height);
        // Each part: its place in the content, and a path (its index) or a run (its index, written with its own face or ours).
        var parts = new (int Order, int Path, int Run)[page.Paths.Count + page.Runs.Count];
        int count = 0;
        for (int i = 0; i < page.Paths.Count; i++)
        {
            if (!page.Paths[i].IsEmpty)
            {
                parts[count++] = (page.Paths[i].Order, i, -1);
            }
        }

        for (int i = 0; i < page.Runs.Count; i++)
        {
            var run = page.Runs[i];
            if ((run.Face is not null && run.Glyphs.Length > 0) || (font is not null && run.Text.Length > 0))
            {
                parts[count++] = (run.Order, -1, i);
            }
            else
            {
                work.LeftOut++;
            }
        }

        Array.Sort(parts, 0, count, ByOrder);
        Span<int> chain = stackalloc int[16];
        foreach (var (_, pathIndex, runIndex) in parts.AsSpan(0, count))
        {
            int clip = pathIndex >= 0 ? page.Paths[pathIndex].Clip : page.Runs[runIndex].Clip, depth = 0;
            for (int c = clip; c >= 0 && c < page.Clips.Count && depth < chain.Length; c = page.Clips[c].Clip)
            {
                chain[depth++] = c;
            }

            try
            {
                if (depth > 0)
                {
                    target.Save();
                }

                try
                {
                    for (int k = depth - 1; k >= 0; k--)
                    {
                        var shape = page.Clips[chain[k]];
                        if (shape.ClipRun < 0 && !shape.IsEmpty)
                        {
                            target.Clip(shape);
                        }
                    }

                    if (pathIndex >= 0)
                    {
                        target.Path(page.Paths[pathIndex]);
                    }
                    else if (page.Runs[runIndex] is { Face: { } face, Glyphs.Length: > 0 } run)
                    {
                        target.Glyphs(face, run.FontSize, run.Glyphs.Span, run.Text, run.Mode, run.Colour);
                    }
                    else
                    {
                        WritePlaced(target, font!, page.Runs[runIndex]);
                    }

                    work.Items++;
                }
                finally
                {
                    if (depth > 0)
                    {
                        target.Restore();
                    }
                }
            }
            catch (Exception e) when (e is ArgumentException or NotSupportedException or InvalidDataException)
            {
                work.LeftOut++;
            }
        }
    }

    // A run's characters as the face's glyphs, one a character, from its box's left on its baseline.
    private static void WritePlaced(IDocumentPageBuilder target, IFontFace face, DocumentTextRun run)
    {
        float scale = run.FontSize / Math.Max(1, face.Metrics.UnitsPerEm), x = run.Box.X, y = run.Baseline ?? run.Box.Bottom;
        var glyphs = System.Buffers.ArrayPool<DocumentGlyph>.Shared.Rent(run.Text.Length);
        try
        {
            int count = 0;
            foreach (var rune in run.Text.EnumerateRunes())
            {
                int glyph = face.Glyph(rune.Value);
                float advance = face.Advance(glyph) * scale;
                glyphs[count++] = new DocumentGlyph(glyph, x, y, advance);
                x += advance;
            }

            target.Glyphs(face, run.FontSize, glyphs.AsSpan(0, count), run.Text, run.Mode, run.Colour);
        }
        finally
        {
            System.Buffers.ArrayPool<DocumentGlyph>.Shared.Return(glyphs);
        }
    }

    // Every signature the signing case wrote (each file's new one and those it held), or, without that case, the set's own.
    private static void Check(DocumentBenchSet set, DocumentBenchOptions options, DocumentBenchWork work, DocumentOptions read, Action<long> progress)
    {
        var check = new SignatureCheckOptions { Trusted = options.Trust is { } root ? [root] : [], At = options.CheckAt, Online = false };
        if (set.Signed is { } signed)
        {
            foreach (var (file, bytes) in signed)
            {
                var part = new DocumentBenchWork(work.Hashing);
                try
                {
                    using var document = NotOnTheNewDesign.Yet<IDocument>($"Picking the document format of {file}");
                    CheckAll(document, check, part, progress);
                    work.Merge(file, part);
                }
                catch (Exception e) when (e is not OutOfMemoryException)
                {
                    work.Refuse(file, Reason(e));
                    part.Finish();
                }
            }

            return;
        }

        EachFile(set, set.Take(options.Files), work, (file, w) =>
        {
            using var document = Open(set, file, read);
            CheckAll(document, check, w, progress);
        });
    }

    private static void CheckAll(IDocument document, SignatureCheckOptions options, DocumentBenchWork work, Action<long> progress)
    {
        work.Files = 1;
        foreach (var signature in document.Signatures)
        {
            var result = signature.Check(options);
            work.Add(signature.Field);
            work.Add(result.Valid ? 1 : 0);
            foreach (var step in result.Steps)
            {
                work.Add(step.Passed ? 1 : 0);
            }

            work.Units++;
            work.Items++;
            progress(1);
        }
    }

    /// <summary>
    /// The test certificate the sample generator writes beside its signed samples (<c>pki.signer.pem</c> and its key, the
    /// chain's <c>pki.intermediate.cer</c>, the root <c>pki.root.cer</c>), looked for under <paramref name="folder"/>.
    /// </summary>
    public static (string Certificate, string Key, string? Chain, string? Root)? GeneratorCertificate(string folder)
    {
        if (!Directory.Exists(folder))
        {
            return null;
        }

        foreach (string certificate in Directory.EnumerateFiles(folder, "pki.signer.pem", SearchOption.AllDirectories).Order(StringComparer.Ordinal))
        {
            string dir = Path.GetDirectoryName(certificate)!, key = Path.Combine(dir, "pki.signer.key.pem");
            if (File.Exists(key))
            {
                string chain = Path.Combine(dir, "pki.intermediate.cer"), root = Path.Combine(dir, "pki.root.cer");
                return (certificate, key, File.Exists(chain) ? chain : null, File.Exists(root) ? root : null);
            }
        }

        return null;
    }

    /// <summary>
    /// The PDF files of a set's folder, in order of their paths, less those the folder's <c>manifest.json</c> (the sample
    /// generator's) marks hostile: files made to fail, which time their refusal, not reading.
    /// </summary>
    public static (List<string> Files, int LeftOut) Files(string folder)
    {
        var files = Directory.EnumerateFiles(folder, "*.pdf", SearchOption.AllDirectories)
            .OrderBy(f => Path.GetRelativePath(folder, f).Replace('\\', '/'), StringComparer.Ordinal).ToList();
        string manifest = Path.Combine(folder, "manifest.json");
        if (!File.Exists(manifest))
        {
            return (files, 0);
        }

        var hostile = new HashSet<string>(StringComparer.Ordinal);
        try
        {
            if (JsonNode.Parse(File.ReadAllText(manifest))?["samples"] is JsonArray samples)
            {
                foreach (var sample in samples)
                {
                    if (sample?["hostile"] is JsonValue flag && flag.TryGetValue(out bool isHostile) && isHostile && (string?)sample["file"] is { } file)
                    {
                        hostile.Add(file);
                    }
                }
            }
        }
        catch (System.Text.Json.JsonException)
        {
            return (files, 0);     // not the generator's manifest: every file is taken
        }

        int before = files.Count;
        files.RemoveAll(f => hostile.Contains(Path.GetRelativePath(folder, f).Replace('\\', '/')));
        return (files, before - files.Count);
    }

    // A stream that counts (and, hashing, hashes) what is written: the bench measures the writer, not a disk. One told to
    // keep (an encrypted output, read back once the run is measured) keeps the bytes instead of hashing them, the
    // keeping's time and allocation left out of the run's (counted on the writing thread around each copy).
    private sealed class CountingStream(DocumentBenchWork work, bool keep = false) : Stream
    {
        private long _length;
        private MemoryStream? _kept;

        // Hands what was kept to the work, to be read back (nothing when not keeping).
        public void ReadBackLater()
        {
            if (keep)
            {
                work.ReadBackLater(_kept is null ? ReadOnlyMemory<byte>.Empty : _kept.GetBuffer().AsMemory(0, (int)_kept.Length));
            }
        }

        public override bool CanRead => false;

        public override bool CanSeek => false;

        public override bool CanWrite => true;

        public override long Length => _length;

        public override long Position
        {
            get => _length;
            set => throw new NotSupportedException();
        }

        public override void Write(byte[] buffer, int offset, int count) => Write(buffer.AsSpan(offset, count));

        public override void Write(ReadOnlySpan<byte> buffer)
        {
            _length += buffer.Length;
            if (!keep)
            {
                work.Add(buffer);
                return;
            }

            long start = Stopwatch.GetTimestamp(), allocated = GC.GetAllocatedBytesForCurrentThread();
            (_kept ??= new MemoryStream()).Write(buffer);
            work.ExcludedBytes += GC.GetAllocatedBytesForCurrentThread() - allocated;
            work.ExcludedTicks += Stopwatch.GetTimestamp() - start;
        }

        public override void Flush()
        {
        }

        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();

        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

        public override void SetLength(long value) => throw new NotSupportedException();

        // What was kept stays readable through the memory handed on (a memory stream's buffer outlives it).
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _kept?.Dispose();
            }

            base.Dispose(disposing);
        }
    }

    // The process's working set sampled on a thread of its own while a run lasts (every 10 ms, and at its start and end):
    // its highest value, and what the sampling allocated (left out of the run's count).
    private sealed class WorkingSetSampler : IDisposable
    {
        private readonly Thread _thread;
        private volatile bool _running = true;
        private long _peak, _allocated;

        private WorkingSetSampler()
        {
            _thread = new Thread(Loop) { IsBackground = true, Name = "bench working set" };
        }

        public static WorkingSetSampler Start()
        {
            var sampler = new WorkingSetSampler();
            sampler._thread.Start();
            return sampler;
        }

        public (long Peak, long Allocated) Stop()
        {
            _running = false;
            _thread.Join();
            return (Interlocked.Read(ref _peak), Interlocked.Read(ref _allocated));
        }

        public void Dispose()
        {
            if (_running)
            {
                Stop();
            }
        }

        private void Loop()
        {
            long before = GC.GetAllocatedBytesForCurrentThread();
            long peak = Environment.WorkingSet;
            while (_running)
            {
                Thread.Sleep(10);
                peak = Math.Max(peak, Environment.WorkingSet);
            }

            peak = Math.Max(peak, Environment.WorkingSet);
            Interlocked.Exchange(ref _peak, peak);
            Interlocked.Exchange(ref _allocated, GC.GetAllocatedBytesForCurrentThread() - before);
        }
    }
}
