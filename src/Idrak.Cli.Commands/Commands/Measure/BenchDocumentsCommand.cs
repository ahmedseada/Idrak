// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography.X509Certificates;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands.Measure;

/// <summary>
/// <c>idrak bench documents</c> (<c>idrak b documents</c>; plan 28, phase D13): the speed and memory of the library's
/// document work (<see cref="DocumentBench"/>'s cases) on sets of documents: the sample generator's samples and its large
/// set by default, or any folders and files. Each case is measured cold (its first run on a set) and warm (the median of
/// the runs after it) apart: the time, the managed bytes allocated and the process's working set at its highest. The
/// library's document settings are used as the environment sets them (its <c>IDRAK_DOCUMENT_*</c> variables), the limits
/// they leave to the memory measured once when the bench starts and kept for every run, and recorded;
/// <c>--save NAME</c> and <c>--compare NAME</c> work as <c>idrak bench</c>'s do, and <c>--compare</c> also says when the
/// saved run read under other limits and whether they explain a changed output.
/// </summary>
internal sealed class BenchDocumentsCommand : Command
{
    /// <summary>The cases run when <c>--cases</c> names none (comma-separated, or all).</summary>
    public const string CasesVariable = "IDRAK_BENCH_DOCUMENTS_CASES";

    /// <summary>The sets when <c>--sets</c> names none.</summary>
    public const string SetsVariable = "IDRAK_BENCH_DOCUMENTS_SETS";

    /// <summary>The drawing resolutions when <c>--dpi</c> names none.</summary>
    public const string DpiVariable = "IDRAK_BENCH_DOCUMENTS_DPI";

    /// <summary>The warm runs of each case when <c>--repeat</c> is not given.</summary>
    public const string RepeatVariable = "IDRAK_BENCH_DOCUMENTS_REPEAT";

    /// <summary>The most files a drawing case takes from a set when <c>--draw-files</c> is not given.</summary>
    public const string DrawFilesVariable = "IDRAK_BENCH_DOCUMENTS_DRAW_FILES";

    /// <summary>The most pages a drawing case draws of a file when <c>--draw-pages</c> is not given.</summary>
    public const string DrawPagesVariable = "IDRAK_BENCH_DOCUMENTS_DRAW_PAGES";

    /// <summary>The font the vector case writes text in whose font a file does not embed, when <c>--font</c> is not given.</summary>
    public const string FontVariable = "IDRAK_BENCH_DOCUMENTS_FONT";

    /// <summary>The sets run when none is named: the samples and the large set, where the sample generator writes them.</summary>
    public const string DefaultSets = "samples,large";

    /// <summary>The resolutions pages are drawn at when none is named: the library's default and a print resolution.</summary>
    public const string DefaultDpi = "150,300";

    private const string Generator = "python -I tools/documents/make_fixtures.py -o test-output/documents/samples --large";

    public override string Name => "bench documents";

    public override IReadOnlyCollection<string> Aliases => ["b documents"];

    public override string Summary => "Speed and memory of reading, drawing, writing, editing and signing documents, cold and warm";

    public override string Usage => """
        [options]

        Runs the library's document work on sets of PDF files, each case once cold (its first run on a set; on the first
        set it also compiles the case's code) and then --repeat times warm, a full collection before each run: the time
        (the median of the warm runs), the managed bytes allocated on every thread and the process's working set at its
        highest (sampled through the run). Writing cases write to a stream that counts, so the writer is measured, not a
        disk. The library's settings come from their variables (IDRAK_DOCUMENT_*, IDRAK_CPU_THREADS, ...); the limits
        they leave unset are measured from the machine's memory once, when the bench starts, and every run of every
        case reads under those same limits (the saved results record them).

        Cases (in order): open (files opened, their pages counted); text-lines, text-content (every page's text through
        the lines layout, and with none); text-parallel (text-lines over the CPU threads: the files at once, or a file's
        pages); images (every image decoded); draw (pages drawn at each --dpi); write-image (each file's images written
        as an image PDF); write-vector (each file's paths and text written back as a vector PDF); edit-incremental,
        edit-rewrite (the information changed and saved); fill (every text field filled, saved); sign (signed with the
        sample generator's test certificate, saved incrementally); check (every signature sign made, checked).

        Sets: samples (test-output/documents/samples, the hostile samples its manifest names left out), large
        (test-output/documents/large: each of its files and folders a set of its own), or any folder or PDF file.
        A file no format opens (a password, damage) is listed as refused and left out of every run. A cap on the files
        (--draw-files, --short) spreads over every file of the set, refused or not, so every run takes the same files.

        Options:
              --cases LIST     the cases, comma-separated, or all (default all; IDRAK_BENCH_DOCUMENTS_CASES)
              --sets LIST      samples, large, or folders and files, comma-separated (default samples,large;
                               IDRAK_BENCH_DOCUMENTS_SETS)
              --dpi LIST       the resolutions pages are drawn at (default 150,300; IDRAK_BENCH_DOCUMENTS_DPI)
          -n, --repeat N       warm runs of each case after its cold one; the median is kept (default 3;
                               IDRAK_BENCH_DOCUMENTS_REPEAT)
              --draw-files N   the most files a drawing case takes from a set, spread evenly (default 50, 0: all;
                               IDRAK_BENCH_DOCUMENTS_DRAW_FILES)
              --draw-pages N   the most pages drawn of a file (default 10, 0: all; IDRAK_BENCH_DOCUMENTS_DRAW_PAGES)
              --font FILE      a font for text whose font a file does not embed, in write-vector (default none: such
                               text is left out and counted; IDRAK_BENCH_DOCUMENTS_FONT)
              --certificate F  sign's certificate (PEM; default the sample generator's test certificate under the
                               samples, never a real one)
              --key FILE       its private key (PEM; default the file beside it, or the certificate's own file)
              --chain FILE     a certificate between it and the root, embedded in the signature
              --trust FILE     the root check trusts
              --short          a quick check: at most 6 files of each set, 2 pages of each, one warm run, one resolution
              --hash           hash every run's output (text, pixels, written bytes; SHA-256), file by file, into the
                               results, so two runs or two saved results show the same output; a case whose runs
                               differ names each file that differs and its hash in each run (the hashing is inside
                               the runs' time). An encrypted output (a fresh random IV every save) is hashed by what
                               it reads back to (its information and encryption, pages, fields, outline, attachments,
                               signatures' checks), read after the run is measured: "read back: N encrypted"
              --save NAME      keep the results as NAME under the cache folder (bench/NAME.json)
              --compare NAME   show the change against the results saved as NAME: whether it read under other limits,
                               and, hashed, which outputs changed, their files, and whether the other limits explain it
                               (a file read back in one and hashed by its bytes in the other is named, not compared)

        Shows its progress: the case and set, the run, how far it is, how fast and the time left.

        Examples:
          idrak bench documents --save laptop
          idrak bench documents --compare laptop --cases text-lines,draw --dpi 150
          idrak b documents --sets ./reports,./newsletters --cases open,text-lines,text-parallel -n 5
          idrak bench documents --sets large --cases write-image,write-vector --font fonts/sans.ttf --hash
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } =
        ["--cases", "--sets", "--dpi", "--repeat", "--draw-files", "--draw-pages", "--font", "--certificate", "--key", "--chain", "--trust", "--save", "--compare"];

    public override IReadOnlyCollection<string> Flags { get; } = ["--short", "--hash"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-n"] = "--repeat" };

    /// <summary>One case measured on one set: its cold run and its warm runs.</summary>
    private sealed record Measured(string Case, string Label, DocumentBenchSet Set, DocumentBenchRun Cold, IReadOnlyList<DocumentBenchRun> Warm)
    {
        public string Unit => DocumentBench.Unit(Case);

        public double WarmMs => Median(Warm.Select(r => r.Milliseconds));

        public long WarmAllocated => (long)Median(Warm.Select(r => (double)r.Allocated));

        public long WarmPeak => Warm.Count == 0 ? Cold.PeakWorkingSet : Warm.Max(r => r.PeakWorkingSet);

        public DocumentBenchWork Work => (Warm.Count > 0 ? Warm[^1] : Cold).Work;

        /// <summary>Whether every run gave the cold run's output (null without hashing).</summary>
        public bool? SameOutput => Cold.Work.Output is null ? null : Warm.All(r => r.Work.Output == Cold.Work.Output);

        /// <summary>The files whose output differs between the runs (hashed runs only; none when every run gave the same).</summary>
        public List<DocumentBenchDifference> Differing { get; } = DocumentBench.Differences(Cold, Warm);

        /// <summary>The files whose output a run hashed by what it reads back to (encrypted outputs).</summary>
        public HashSet<string> ReadBack { get; } = FilesReadBack(Cold, Warm);
    }

    public override int Run(CommandContext context)
    {
        if (context.Positional.Count > 0)
        {
            throw new UsageException($"bench documents takes no arguments (name folders and files with --sets), not '{context.Positional[0]}'.");
        }

        var options = Options(context, out var sets, out string? certificateNote);
        string? compareName = context.Option("--compare");
        JsonObject? saved = compareName is null ? null : BenchCommand.LoadSaved(context, compareName);
        var read = options.Read;
        context.Write(string.Create(CultureInfo.InvariantCulture,
            $"documents on the CPU ({Environment.ProcessorCount} cores, {ComputeResources.MaxCpuThreads} threads): {string.Join(", ", options.Cases)}; "
            + $"cold: a case's first run on a set, warm: the median of {options.Repeats} runs after it"));
        context.Write("  limits for every run (measured from the memory once, now, where no variable sets them): " + Limits(read));
        foreach (var set in sets)
        {
            context.Write($"  {set.Name}: {set.Files.Count} files in {set.Root}" + (set.LeftOut > 0 ? $" ({set.LeftOut} hostile samples left out)" : ""));
        }

        if (certificateNote is not null)
        {
            context.Write("  " + certificateNote);
        }

        // Each case on each set (drawing once a resolution), in the cases' order, set by set.
        var plan = new List<(DocumentBenchSet Set, string Case, float Dpi, string Label)>();
        foreach (var set in sets)
        {
            foreach (string @case in options.Cases)
            {
                if (@case is "sign" && options.Signing is null)
                {
                    continue;
                }

                if (@case == "draw")
                {
                    plan.AddRange(options.Dpis.Select(dpi => (set, @case, dpi, string.Create(CultureInfo.InvariantCulture, $"draw {dpi:0.##} dpi"))));
                }
                else
                {
                    plan.Add((set, @case, 0f, @case));
                }
            }
        }

        var measured = new List<Measured>();
        int runs = plan.Count * (1 + options.Repeats), run = 0;
        using (var progress = new ProgressLine(context, "documents", null, ProgressUnit.Items))
        {
            progress.Detail = () => string.Create(CultureInfo.InvariantCulture, $"run {run} of {runs}");
            foreach (var (set, @case, dpi, label) in plan)
            {
                DocumentBenchRun? cold = null;
                var warm = new List<DocumentBenchRun>();
                for (int k = 0; k <= options.Repeats; k++)
                {
                    run++;
                    long? total = cold is null ? null : cold.Work.Units;
                    progress.NextStage($"{label}, {set.Name}, {(k == 0 ? "cold" : $"warm {k}/{options.Repeats}")}", total, ProgressUnit.Items, keepLine: false);
                    var result = DocumentBench.Measure(@case, dpi, set, options, n => progress.Advance(n));
                    if (cold is null)
                    {
                        cold = result;
                    }
                    else
                    {
                        warm.Add(result);
                    }
                }

                var done = new Measured(@case, label, set, cold!, warm);
                foreach (var w in warm)
                {
                    w.Work.FileOutputs.Clear();     // compared (done.Differing); the cold run's are the ones recorded
                }

                measured.Add(done);
                progress.Erase();
                context.Write(Line(done));
                foreach (string line in DifferenceLines(done.Label, set, done.Cold, done.Warm, done.Differing))
                {
                    context.Write(line);
                }

                if (context.Verbose)
                {
                    foreach (var (file, reason) in done.Cold.Work.Refused)
                    {
                        context.Write($"    refused {Path.GetRelativePath(set.Root, file)}: {reason}");
                    }
                }
            }

            progress.Clear();
        }

        var results = Results(measured);
        context.Write("");
        context.Table(["Case", "Set", "Work", "Warm ms", "Cold ms", "Warm a unit", "Allocated warm / cold", "Working set warm / cold"],
            measured.Select(m => (IReadOnlyList<string>)
            [
                m.Label, m.Set.Name, Work(m), BenchCommand.Format(m.WarmMs), BenchCommand.Format(m.Cold.Milliseconds), PerUnit(m),
                $"{Size(m.WarmAllocated)} / {Size(m.Cold.Allocated)}", $"{Size(m.WarmPeak)} / {Size(m.Cold.PeakWorkingSet)}",
            ]));
        var document = Document(options, sets, measured, results);
        if (saved is not null)
        {
            document["compare"] = BenchCommand.Compare(context, compareName!, saved, [.. results.Select(r => ("", r))], DocumentBench.Method);
            var (limits, sameLimits) = CompareLimits(context, compareName!, saved, read);
            document["compare"]!["limits"] = limits;
            document["compare"]!["outputs"] = CompareOutputs(context, compareName!, saved, measured, sameLimits);
        }

        if (context.Option("--save") is { } saveName)
        {
            string path = BenchCommand.SavedPath(context, saveName);
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, document.ToJsonString(CommandContext.JsonOutput));
            document["saved"] = path;
            context.Write($"\nSaved as '{saveName}' ({path}); compare later with: idrak bench documents --compare {saveName}");
        }

        context.WriteJson(document);
        return measured.Count == 0 ? ExitCodes.Failed : ExitCodes.Ok;
    }

    // The options, each from its option, else its variable, else its default; the sets found; the certificate's note.
    private static DocumentBenchOptions Options(CommandContext context, out List<DocumentBenchSet> sets, out string? certificateNote)
    {
        (string Text, string From)? Given(string option, string variable) =>
            context.Option(option) is { } text ? (text, option)
            : Environment.GetEnvironmentVariable(variable) is { Length: > 0 } value && value.Trim().Length > 0 ? (value.Trim(), variable) : null;
        int Whole(string option, string variable, int least, int otherwise) => Given(option, variable) is not { } given ? otherwise
            : int.TryParse(given.Text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) && value >= least ? value
            : throw new UsageException($"{given.From} needs a whole number of at least {least}, not '{given.Text}'.");
        static List<string> Split(string text) => [.. text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)];

        bool quick = context.Flag("--short");
        var casesText = Given("--cases", CasesVariable);
        List<string> cases = casesText is { } chosen ? Split(chosen.Text).Select(c => c.ToLowerInvariant()).ToList() : ["all"];
        if (cases.Contains("all"))
        {
            cases = [.. DocumentBench.Cases];
        }

        string? unknown = cases.FirstOrDefault(c => !DocumentBench.Cases.Contains(c));
        if (cases.Count == 0 || unknown is not null)
        {
            throw new UsageException($"{casesText?.From ?? "--cases"} takes all or {string.Join(", ", DocumentBench.Cases)} (comma-separated), not '{unknown ?? ""}'.");
        }

        cases = [.. DocumentBench.Cases.Where(cases.Contains)];     // in the cases' own order: check follows sign
        var dpiText = Given("--dpi", DpiVariable);
        var dpis = new List<float>();
        foreach (string item in Split(dpiText?.Text ?? DefaultDpi))
        {
            dpis.Add(float.TryParse(item, NumberStyles.Float, CultureInfo.InvariantCulture, out float dpi) && dpi > 0 && dpi <= 9600
                ? dpi : throw new UsageException($"{dpiText?.From ?? "--dpi"} takes resolutions from 1 to 9600 dots per inch, comma-separated, not '{item}'."));
        }

        if (dpis.Count == 0)
        {
            throw new UsageException($"{dpiText?.From ?? "--dpi"} needs a resolution.");
        }

        if (quick && dpiText is null)
        {
            dpis = [dpis[0]];
        }

        var setsText = Given("--sets", SetsVariable);
        sets = Sets(Split(setsText?.Text ?? DefaultSets), setsText?.From ?? "--sets");
        IFontFace? font = null;
        if (Given("--font", FontVariable) is { } fontFile)
        {
            try
            {
                font = NotOnTheNewDesign.Yet<IReadOnlyList<IFontFace>>($"Picking the font format of {fontFile.Text}") is { Count: > 0 } faces ? faces[0] : throw new UsageException($"{fontFile.From}: {fontFile.Text} holds no font.");
            }
            catch (Exception e) when (e is IOException or InvalidDataException or NotSupportedException or UnauthorizedAccessException or ArgumentException)
            {
                throw new UsageException($"{fontFile.From}: {fontFile.Text}: {e.Message}");
            }
        }

        var (signing, trust, at) = Signing(context, cases, sets, out certificateNote);
        return new DocumentBenchOptions
        {
            Read = DocumentBench.Pin(new DocumentOptions()),     // the limits measured once: every run reads under the same ones
            Cases = cases,
            Dpis = dpis,
            Repeats = Whole("--repeat", RepeatVariable, 1, quick ? 1 : 3),
            DrawFiles = Whole("--draw-files", DrawFilesVariable, 0, 50),
            DrawPages = Whole("--draw-pages", DrawPagesVariable, 0, 10),
            Files = quick ? 6 : 0,
            Pages = quick ? 2 : 0,
            Hash = context.Flag("--hash"),
            Font = font,
            Signing = signing,
            Trust = trust,
            CheckAt = at,
        };
    }

    // The sets: samples (one set), large (each of its members a set), any other folder or file one set.
    private static List<DocumentBenchSet> Sets(List<string> names, string from)
    {
        if (names.Count == 0)
        {
            throw new UsageException($"{from} needs samples, large, or folders and files.");
        }

        var sets = new List<DocumentBenchSet>();
        foreach (string name in names)
        {
            if (name.Equals("samples", StringComparison.OrdinalIgnoreCase) || name.Equals("large", StringComparison.OrdinalIgnoreCase))
            {
                bool large = name.Equals("large", StringComparison.OrdinalIgnoreCase);
                string folder = Path.GetFullPath(Path.Combine("test-output", "documents", large ? "large" : "samples"));
                if (!Directory.Exists(folder))
                {
                    throw new UsageException($"{from}: the {name.ToLowerInvariant()} set is not in {folder}. Make it with: {Generator}");
                }

                if (!large)
                {
                    var (files, leftOut) = DocumentBench.Files(folder);
                    sets.Add(new DocumentBenchSet("samples", folder, files, leftOut));
                    continue;
                }

                // The large set's members, each a set: the folder's files and folders (through a folder holding one folder alone).
                string inside = folder;
                while (Directory.GetFiles(inside, "*.pdf").Length == 0 && Directory.GetDirectories(inside) is [var only])
                {
                    inside = only;
                }

                foreach (string file in Directory.GetFiles(inside, "*.pdf").Order(StringComparer.Ordinal))
                {
                    sets.Add(new DocumentBenchSet(Path.GetFileNameWithoutExtension(file), Path.GetDirectoryName(file)!, [file], 0));
                }

                foreach (string member in Directory.GetDirectories(inside).Order(StringComparer.Ordinal))
                {
                    var (files, leftOut) = DocumentBench.Files(member);
                    if (files.Count > 0)
                    {
                        sets.Add(new DocumentBenchSet(Path.GetFileName(member), member, files, leftOut));
                    }
                }

                if (sets.Count == 0 || !sets.Any(s => s.Root.StartsWith(folder, StringComparison.Ordinal)))
                {
                    throw new UsageException($"{from}: {folder} holds no PDF file. Make the large set with: {Generator}");
                }

                continue;
            }

            string path = Path.GetFullPath(name);
            if (File.Exists(path))
            {
                sets.Add(new DocumentBenchSet(Path.GetFileNameWithoutExtension(path), Path.GetDirectoryName(path)!, [path], 0));
            }
            else if (Directory.Exists(path))
            {
                var (files, leftOut) = DocumentBench.Files(path);
                if (files.Count == 0)
                {
                    throw new UsageException($"{from}: {path} holds no PDF file.");
                }

                sets.Add(new DocumentBenchSet(Path.GetFileName(path.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)), path, files, leftOut));
            }
            else
            {
                throw new UsageException($"{from}: {name} is not a folder or a file, nor samples or large.");
            }
        }

        return sets;
    }

    // How sign signs and check checks: --certificate and its key, else the sample generator's test certificate under the
    // samples; signed at the certificate's first day (a fixed time inside its validity, so the same file signs to the same bytes).
    private static (SigningOptions? Signing, X509Certificate2? Trust, DateTimeOffset? At) Signing(CommandContext context, List<string> cases, List<DocumentBenchSet> sets, out string? note)
    {
        note = null;
        if (!cases.Contains("sign") && !cases.Contains("check"))
        {
            return (null, null, null);
        }

        string? certificate = context.Option("--certificate"), key = context.Option("--key"), chain = context.Option("--chain"), root = context.Option("--trust");
        if (certificate is null)
        {
            string samples = sets.FirstOrDefault(s => s.Name == "samples")?.Root ?? Path.GetFullPath(Path.Combine("test-output", "documents", "samples"));
            if (DocumentBench.GeneratorCertificate(samples) is { } found)
            {
                (certificate, key, chain, root) = (found.Certificate, key ?? found.Key, chain ?? found.Chain, root ?? found.Root);
            }
            else
            {
                note = $"sign and check: no certificate (--certificate, or the sample generator's test certificate under {samples}): sign is left out";
                return (null, null, null);
            }
        }

        try
        {
            var signer = X509Certificate2.CreateFromPemFile(certificate, key);
            var at = new DateTimeOffset(signer.NotBefore.ToUniversalTime(), TimeSpan.Zero).AddDays(1);
            var signing = new SigningOptions(signer)
            {
                Chain = chain is null ? [] : [X509CertificateLoader.LoadCertificateFromFile(chain)],
                Level = DocumentSignatureLevel.BaselineB,
                SigningTime = at,
                Reason = "Benchmark",
                Location = "Test bench",
                Online = false,
            };
            note = $"sign and check: {signer.Subject}" + (root is null ? ", no root trusted" : $", trusting {Path.GetFileName(root)}");
            return (signing, root is null ? null : X509CertificateLoader.LoadCertificateFromFile(root), at.AddHours(1));
        }
        catch (Exception e) when (e is IOException or System.Security.Cryptography.CryptographicException or ArgumentException or UnauthorizedAccessException)
        {
            throw new UsageException($"--certificate {certificate}: {e.Message}");
        }
    }

    // The results --save keeps and --compare matches: each case's time, allocation and working set on each set, cold beside warm.
    private static List<BenchResult> Results(List<Measured> measured)
    {
        var results = new List<BenchResult>();
        foreach (var m in measured)
        {
            string name = $"{m.Label} {m.Set.Name}";
            results.Add(new BenchResult(name, m.WarmMs, "ms", false, $"{Work(m)}, {PerUnit(m)}", Cold: m.Cold.Milliseconds, ColdJitMs: m.Cold.JitMilliseconds, ColdJitMethods: m.Cold.JitMethods));
            results.Add(new BenchResult($"{name}: allocated", m.WarmAllocated / 1048576.0, "MiB", false, "managed bytes allocated a run, every thread", Cold: m.Cold.Allocated / 1048576.0));
            results.Add(new BenchResult($"{name}: working set", m.WarmPeak / 1048576.0, "MiB", false, "the process's working set at its highest in a run", Cold: m.Cold.PeakWorkingSet / 1048576.0));
        }

        return results;
    }

    private static string Line(Measured m)
    {
        var cold = m.Cold;
        string refused = cold.Work.Refused.Count > 0 ? $" ({cold.Work.Refused.Count} refused)" : "";
        string output = m.SameOutput switch
        {
            true => ", the same output every run",
            false => m.Differing.Count > 0
                ? string.Create(CultureInfo.InvariantCulture, $", OUTPUT DIFFERS BETWEEN RUNS ({m.Differing.Count:N0} file{(m.Differing.Count == 1 ? "" : "s")}, below)")
                : ", OUTPUT DIFFERS BETWEEN RUNS (every file the same alone: their order or number)",
            null => "",
        };
        int readBack = cold.Work.ReadBackFiles.Count;
        string back = readBack > 0 ? string.Create(CultureInfo.InvariantCulture, $" (read back: {readBack:N0} encrypted)") : "";
        return string.Create(CultureInfo.InvariantCulture,
            $"  {m.Label} on {m.Set.Name}: {Work(m)}{refused}; warm {BenchCommand.Format(m.WarmMs)} ms ({PerUnit(m)}), {Size(m.WarmAllocated)} allocated, working set {Size(m.WarmPeak)}; "
            + $"cold {BenchCommand.Format(cold.Milliseconds)} ms ({BenchCommand.Format(cold.JitMilliseconds)} ms compiling {cold.JitMethods:N0} methods), {Size(cold.Allocated)}, {Size(cold.PeakWorkingSet)}{output}{back}");
    }

    private static string Work(Measured m)
    {
        var w = m.Work;
        string unit = m.Unit;
        var parts = new List<string> { string.Create(CultureInfo.InvariantCulture, $"{w.Units:N0} {unit}{(w.Units == 1 ? "" : "s")}") };
        if (unit != "file" && w.Files > 0)
        {
            parts.Add(string.Create(CultureInfo.InvariantCulture, $"{w.Files:N0} files"));
        }

        if (m.Case is "images" or "write-image" && w.Items > 0)
        {
            parts.Add(string.Create(CultureInfo.InvariantCulture, $"{w.Items:N0} images, {w.Pixels / 1e6:0.#} Mpx"));
        }

        if (m.Case is "write-vector" && w.Items + w.LeftOut > 0)
        {
            parts.Add(string.Create(CultureInfo.InvariantCulture, $"{w.Items:N0} parts written, {w.LeftOut:N0} left out"));
        }

        if (m.Case is "fill" && w.Items > 0)
        {
            parts.Add(string.Create(CultureInfo.InvariantCulture, $"{w.Items:N0} fields"));
        }

        if (w.Written > 0)
        {
            parts.Add(Size(w.Written) + " written");
        }

        return string.Join(", ", parts);
    }

    private static string PerUnit(Measured m) => m.Work.Units == 0 ? "-"
        : string.Create(CultureInfo.InvariantCulture, $"{BenchCommand.Format(m.WarmMs * 1000 / m.Work.Units)} µs a {m.Unit}");

    private static string Size(long bytes) => bytes switch
    {
        >= 1L << 30 => string.Create(CultureInfo.InvariantCulture, $"{bytes / (double)(1L << 30):0.00} GiB"),
        >= 1L << 20 => string.Create(CultureInfo.InvariantCulture, $"{bytes / (double)(1L << 20):0.0} MiB"),
        _ => string.Create(CultureInfo.InvariantCulture, $"{bytes / 1024.0:0} KiB"),
    };

    private static double Median(IEnumerable<double> values)
    {
        var list = values.ToList();
        return BenchTiming.Median(list);
    }

    // The limits as the results record them (settings.limits) and as a line shows them.
    private static JsonObject LimitsJson(DocumentOptions read) => new()
    {
        ["max_stream_bytes"] = read.MaxStreamBytes,
        ["max_objects"] = read.MaxObjects,
        ["max_depth"] = read.MaxDepth,
        ["cache_bytes"] = read.CacheBytes,
    };

    private static string Limits(DocumentOptions read) => string.Join(", ", LimitValues(read).Select(l => Limit(l.Key, l.Value)));

    private static (string Key, long Value)[] LimitValues(DocumentOptions read) =>
        [("max_stream_bytes", read.MaxStreamBytes), ("cache_bytes", read.CacheBytes), ("max_objects", read.MaxObjects), ("max_depth", read.MaxDepth)];

    private static string Limit(string key, long value) => key switch
    {
        "max_stream_bytes" => "a stream " + Size(value),
        "cache_bytes" => "a document's cache " + Size(value),
        "max_objects" => string.Create(CultureInfo.InvariantCulture, $"{value:N0} objects"),
        _ => string.Create(CultureInfo.InvariantCulture, $"nesting {value}"),
    };

    /// <summary>
    /// The lines that name each file whose output differs between a case's runs (at most 20; the results hold them all),
    /// with its output in every run (the hash's first 16 digits) or why a run did not read it, and "(encrypted, read back)"
    /// when that output was hashed by what it reads back to. Every run reads under the same limits and takes the same
    /// files, so such a file is a finding to look into, not the bench's noise.
    /// </summary>
    internal static List<string> DifferenceLines(string label, DocumentBenchSet set, DocumentBenchRun cold, IReadOnlyList<DocumentBenchRun> warm, IReadOnlyList<DocumentBenchDifference> differing)
    {
        const int Shown = 20;
        var lines = new List<string>();
        var readBack = FilesReadBack(cold, warm);
        foreach (var difference in differing.Take(Shown))
        {
            var runs = new List<string>(difference.Outputs.Count);
            for (int r = 0; r < difference.Outputs.Count; r++)
            {
                string run = r == 0 ? "cold" : string.Create(CultureInfo.InvariantCulture, $"warm {r}");
                var refused = (r == 0 ? cold : warm[r - 1]).Work.Refused.FirstOrDefault(x => x.File == difference.File);
                runs.Add(difference.Outputs[r] is { } output ? $"{run} {output[..Math.Min(16, output.Length)]}"
                    : refused.Reason is { } reason ? $"{run} refused ({reason})"
                    : $"{run} not read");
            }

            string how = readBack.Contains(difference.File) ? " (encrypted, read back)" : "";
            lines.Add($"    differs in {label}: {set.Relative(difference.File)}{how}: {string.Join(", ", runs)}");
        }

        if (differing.Count > Shown)
        {
            lines.Add(string.Create(CultureInfo.InvariantCulture, $"    and {differing.Count - Shown:N0} more files (each in the results' differing_files)"));
        }

        return lines;
    }

    // The files whose output any of a case's runs hashed by what it reads back to.
    private static HashSet<string> FilesReadBack(DocumentBenchRun cold, IReadOnlyList<DocumentBenchRun> warm)
    {
        var files = new HashSet<string>(cold.Work.ReadBackFiles, StringComparer.Ordinal);
        foreach (var run in warm)
        {
            files.UnionWith(run.Work.ReadBackFiles);
        }

        return files;
    }

    // Whether the saved run read under the limits this one does: a line naming each limit that differs (now and saved),
    // and the JSON (both sets, the same or not). Null when the saved results record no limits.
    private static (JsonObject Json, bool? Same) CompareLimits(CommandContext context, string name, JsonObject saved, DocumentOptions read)
    {
        if (saved["settings"]?["limits"] is not JsonObject before)
        {
            context.Write($"  limits: '{name}' recorded none, so other limits cannot be told from a changed output");
            return (new JsonObject { ["same"] = null, ["saved"] = null, ["now"] = LimitsJson(read) }, null);
        }

        var shown = new List<string>();
        var keys = new JsonArray();
        foreach (var (key, value) in LimitValues(read))
        {
            long? then = before[key] is JsonValue v && v.TryGetValue(out long x) ? x : null;
            if (then != value)
            {
                shown.Add($"{Limit(key, value)} (saved: {(then is { } t ? Limit(key, t) : "none")})");
                keys.Add(item: key);
            }
        }

        if (shown.Count > 0)
        {
            context.Write($"  limits: other than '{name}' read under: {string.Join(", ", shown)}; a file near a limit can read otherwise");
        }

        return (new JsonObject { ["same"] = shown.Count == 0, ["changed"] = keys, ["saved"] = before.DeepClone(), ["now"] = LimitsJson(read) }, shown.Count == 0);
    }

    // Whether each hashed case gave the output it gave when saved (rule 71: a change that makes things faster changes no
    // output). A case that did not names the files that changed (each file's output is saved) and, when the saved run read
    // under other limits, whether they explain it: every changed file met a limit in one of the two runs (refused by one,
    // or a part left out naming one). A changed case is listed as changed either way: other limits never hide a difference.
    // Like is compared with like: a file hashed by what it reads back to in one run (an encrypted output) and by its bytes
    // in the other (a save made before encrypted outputs were read back) is named as hashed another way, not compared and
    // not a change; a case whose only differences are such files is neither the same nor changed.
    private static JsonArray CompareOutputs(CommandContext context, string name, JsonObject saved, List<Measured> measured, bool? sameLimits)
    {
        var before = (saved["cases"] as JsonArray ?? []).OfType<JsonObject>()
            .Where(c => (string?)c["output"] is not null)
            .GroupBy(c => $"{(string?)c["label"]} {(string?)c["set"]}")
            .ToDictionary(g => g.Key, g => g.First());
        var rows = new JsonArray();
        var changed = new List<string>();
        var otherWay = new List<string>();
        var details = new List<string>();
        int same = 0;
        foreach (var m in measured.Where(m => m.Cold.Work.Output is not null))
        {
            string key = $"{m.Label} {m.Set.Name}";
            if (!before.TryGetValue(key, out var old))
            {
                continue;
            }

            bool equal = (string?)old["output"] == m.Cold.Work.Output;
            var row = new JsonObject { ["case"] = m.Label, ["set"] = m.Set.Name, ["same"] = equal };
            rows.Add(item: row);
            if (equal)
            {
                same++;
                continue;
            }

            // How each file was hashed here and when saved: read back (an encrypted output) or by its bytes. Results saved
            // before encrypted outputs were read back hold no read_back_files: every file there was hashed by its bytes.
            var readBackNow = new HashSet<string>(m.Cold.Work.ReadBackFiles.Select(m.Set.Relative), StringComparer.Ordinal);
            var readBackThen = new HashSet<string>((old["read_back_files"] as JsonArray ?? []).Select(f => (string?)f).OfType<string>(), StringComparer.Ordinal);
            if (old["file_outputs"] is not JsonObject savedFiles)
            {
                row["files"] = null;
                row["limits_explain"] = false;
                if (readBackNow.Count > 0 && old["read_back_files"] is null)
                {
                    row["same"] = null;
                    row["hashed_another_way"] = null;
                    otherWay.Add(key);
                    details.Add(string.Create(CultureInfo.InvariantCulture,
                        $"    {key}: {readBackNow.Count:N0} encrypted output{(readBackNow.Count == 1 ? "" : "s")} read back here, hashed by their bytes in '{name}', which holds no file's own output: not compared"));
                    continue;
                }

                changed.Add(key);
                details.Add($"    {key}: '{name}' holds no file's own output, so the files are not named; a change all the same");
                continue;
            }

            // The files whose output is not the same (or that one of the two runs did not read), in the order they ran, and
            // those read by both but hashed another way.
            var files = new List<string>();
            var another = new List<string>();
            var now = new HashSet<string>(StringComparer.Ordinal);
            foreach (var (file, output) in m.Cold.Work.FileOutputs)
            {
                string relative = m.Set.Relative(file);
                now.Add(relative);
                string? then = (string?)savedFiles[relative];
                if (then != output)
                {
                    (then is not null && readBackNow.Contains(relative) != readBackThen.Contains(relative) ? another : files).Add(relative);
                }
            }

            files.AddRange(savedFiles.Select(p => p.Key).Where(f => !now.Contains(f)));
            if (another.Count > 0)
            {
                row["hashed_another_way"] = new JsonArray([.. another.Select(f => (JsonNode)f)]);
            }

            string how = another.All(readBackNow.Contains) ? $"read back here (encrypted), hashed by their bytes in '{name}'"
                : another.All(readBackThen.Contains) ? $"hashed by their bytes here, read back in '{name}'"
                : "read back in one of the two runs, hashed by their bytes in the other";
            string anotherText = string.Create(CultureInfo.InvariantCulture,
                $"{another.Count:N0} file{(another.Count == 1 ? "" : "s")} hashed another way ({string.Join(", ", another.Take(10))}{(another.Count > 10 ? $", and {another.Count - 10:N0} more" : "")}): {how}, not compared");
            bool sameOrder = savedFiles.Select(p => p.Key).SequenceEqual(m.Cold.Work.FileOutputs.Select(f => m.Set.Relative(f.File)), StringComparer.Ordinal);
            if (files.Count == 0 && another.Count > 0 && sameOrder)
            {
                row["same"] = null;
                row["files"] = new JsonArray();
                row["limits_explain"] = false;
                otherWay.Add(key);
                details.Add($"    {key}: {anotherText}; every other file's output the same as saved");
                continue;
            }

            changed.Add(key);

            // The files that met a limit in either run: in this case, or refused by one when the set's files were opened
            // (left out of every case after it).
            var limited = new HashSet<string>(m.Cold.Work.LimitFiles.Select(m.Set.Relative), StringComparer.Ordinal);
            limited.UnionWith(m.Set.Refused.Where(p => DocumentBench.NamesLimit(p.Value)).Select(p => m.Set.Relative(p.Key)));
            limited.UnionWith((old["limit_files"] as JsonArray ?? []).Select(f => (string?)f).OfType<string>());
            limited.UnionWith((saved["sets"] as JsonArray ?? []).OfType<JsonObject>().Where(set => (string?)set["name"] == m.Set.Name)
                .SelectMany(set => (set["refused"] as JsonArray ?? []).OfType<JsonObject>())
                .Where(r => (string?)r["reason"] is { } reason && DocumentBench.NamesLimit(reason))
                .Select(r => ((string?)r["file"] ?? "").Replace('\\', '/')));
            bool explained = sameLimits == false && files.Count > 0 && files.All(limited.Contains);
            row["files"] = new JsonArray([.. files.Select(f => (JsonNode)f)]);
            row["limits_explain"] = explained;
            int unexplained = files.Count(f => !limited.Contains(f));
            string why = sameLimits switch
            {
                true => "read under the same limits",
                false when explained => "the other limits explain it: each of them met a limit in one of the two runs",
                false => string.Create(CultureInfo.InvariantCulture, $"NOT explained by the other limits: {unexplained:N0} of them met none"),
                null => "the saved limits unknown",
            };
            string also = another.Count > 0 ? $"; besides, {anotherText}" : "";
            details.Add(files.Count == 0
                ? $"    {key}: every file's output the same as saved, their order or number not; {why}{also}"
                : string.Create(CultureInfo.InvariantCulture,
                    $"    {key}: {files.Count:N0} file{(files.Count == 1 ? "" : "s")} changed ({string.Join(", ", files.Take(10))}{(files.Count > 10 ? $", and {files.Count - 10:N0} more" : "")}); {why}{also}"));
        }

        if (rows.Count > 0)
        {
            context.Write(changed.Count == 0 && otherWay.Count == 0
                ? $"  outputs: all {same} hashed cases the same as '{name}'"
                : $"  outputs: {same} of {rows.Count} hashed cases the same as '{name}'"
                  + (changed.Count > 0 ? $"; CHANGED: {string.Join(", ", changed)}" : "")
                  + (otherWay.Count > 0 ? $"; hashed another way, not compared: {string.Join(", ", otherWay)}" : ""));
            foreach (string line in details)
            {
                context.Write(line);
            }
        }

        return rows;
    }

    private static JsonObject Document(DocumentBenchOptions options, List<DocumentBenchSet> sets, List<Measured> measured, List<BenchResult> results)
    {
        // The library's document settings as the environment gives them (every variable the table names for this command,
        // and the CPU's), beside the limits every run read under (measured once where no variable sets them).
        var ordered = new JsonObject();
        var names = EnvironmentVariables.For("bench documents").Select(v => v.Name).Concat(["IDRAK_CPU_THREADS", "IDRAK_CPU_MEMORY_LIMIT_MB", "IDRAK_FONT_CACHE"]);
        foreach (string name in names.Distinct().Order(StringComparer.Ordinal))
        {
            if (Environment.GetEnvironmentVariable(name) is { Length: > 0 } value)
            {
                ordered[name] = value;
            }
        }

        static JsonObject Run(DocumentBenchRun r) => new()
        {
            ["ms"] = r.Milliseconds,
            ["allocated_bytes"] = r.Allocated,
            ["peak_working_set_bytes"] = r.PeakWorkingSet,
            ["jit_ms"] = r.JitMilliseconds,
            ["jit_methods"] = r.JitMethods,
            ["output"] = r.Work.Output,
        };

        using var process = Process.GetCurrentProcess();
        return new JsonObject
        {
            ["kind"] = "documents",
            ["date"] = DateTimeOffset.UtcNow.ToString("o", CultureInfo.InvariantCulture),
            ["timing"] = DocumentBench.Method,
            ["device"] = "cpu",
            ["settings"] = new JsonObject
            {
                ["cases"] = new JsonArray([.. options.Cases.Select(c => (JsonNode)c)]),
                ["dpi"] = new JsonArray([.. options.Dpis.Select(d => (JsonNode)d)]),
                ["repeat"] = options.Repeats,
                ["draw_files"] = options.DrawFiles,
                ["draw_pages"] = options.DrawPages,
                ["files"] = options.Files,
                ["pages"] = options.Pages,
                ["hash"] = options.Hash,
                ["font"] = options.Font is { } font ? $"{font.Family} {font.Style}" : null,
                ["signer"] = options.Signing?.Certificate.Subject,
                ["processors"] = Environment.ProcessorCount,
                ["cpu_threads"] = ComputeResources.MaxCpuThreads,
                ["cpu_memory_limit_bytes"] = ComputeResources.CpuMemoryLimit,
                ["limits"] = LimitsJson(options.Read),
                ["variables"] = ordered,
            },
            ["sets"] = new JsonArray([.. sets.Select(s => (JsonNode)new JsonObject
            {
                ["name"] = s.Name,
                ["root"] = s.Root,
                ["files"] = s.Files.Count,
                ["left_out"] = s.LeftOut,
                ["pages"] = s.Pages.Values.Sum(),
                ["refused"] = new JsonArray([.. s.Refused.OrderBy(p => p.Key, StringComparer.Ordinal)
                    .Select(p => (JsonNode)new JsonObject { ["file"] = s.Relative(p.Key), ["reason"] = p.Value })]),
            })]),
            ["cases"] = new JsonArray([.. measured.Select(m => (JsonNode)new JsonObject
            {
                ["case"] = m.Case,
                ["label"] = m.Label,
                ["set"] = m.Set.Name,
                ["unit"] = m.Unit,
                ["units"] = m.Work.Units,
                ["files"] = m.Work.Files,
                ["pages"] = m.Work.Pages,
                ["items"] = m.Work.Items,
                ["pixels"] = m.Work.Pixels,
                ["written_bytes"] = m.Work.Written,
                ["left_out"] = m.Work.LeftOut,
                ["refused"] = new JsonArray([.. m.Cold.Work.Refused.Select(r => (JsonNode)new JsonObject { ["file"] = m.Set.Relative(r.File), ["reason"] = r.Reason })]),
                ["output"] = m.Cold.Work.Output,
                ["same_output"] = m.SameOutput,
                ["file_outputs"] = m.Cold.Work.Output is null ? null : FileOutputs(m),
                // The files whose output was hashed by what it reads back to (encrypted outputs), so a comparison hashes like with like.
                ["read_back"] = m.Cold.Work.Output is null ? null : JsonValue.Create(m.Cold.Work.ReadBackFiles.Count),
                ["read_back_files"] = m.Cold.Work.Output is null ? null
                    : new JsonArray([.. m.Cold.Work.ReadBackFiles.Select(m.Set.Relative).Order(StringComparer.Ordinal).Select(f => (JsonNode)f)]),
                ["limit_files"] = new JsonArray([.. m.Cold.Work.LimitFiles.Select(m.Set.Relative).Order(StringComparer.Ordinal).Select(f => (JsonNode)f)]),
                ["differing_files"] = m.Differing.Count == 0 ? null : new JsonArray([.. m.Differing.Select(d => (JsonNode)new JsonObject
                {
                    ["file"] = m.Set.Relative(d.File),
                    ["read_back"] = m.ReadBack.Contains(d.File),
                    ["outputs"] = new JsonArray([.. d.Outputs.Select(o => (JsonNode?)o)]),
                })]),
                ["cold"] = Run(m.Cold),
                ["warm"] = new JsonObject
                {
                    ["ms"] = m.WarmMs,
                    ["allocated_bytes"] = m.WarmAllocated,
                    ["peak_working_set_bytes"] = m.WarmPeak,
                    ["runs"] = new JsonArray([.. m.Warm.Select(r => (JsonNode)Run(r))]),
                },
            })]),
            ["results"] = new JsonArray([.. results.Select(r => (JsonNode)BenchCommand.ToJson(r, ""))]),
            ["process_peak_working_set_bytes"] = process.PeakWorkingSet64,
        };
    }

    // The cold run's output file by file (each file's path under its set's folder), so a comparison can name what changed.
    private static JsonObject FileOutputs(Measured m)
    {
        var files = new JsonObject();
        foreach (var (file, output) in m.Cold.Work.FileOutputs)
        {
            files[m.Set.Relative(file)] = output;
        }

        return files;
    }
}
