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
/// library's document settings are used as the environment sets them (its <c>IDRAK_DOCUMENT_*</c> variables) and
/// recorded with the measured limits; <c>--save NAME</c> and <c>--compare NAME</c> work as <c>idrak bench</c>'s do.
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
        disk. The library's settings come from their variables (IDRAK_DOCUMENT_*, IDRAK_CPU_THREADS, ...), and the saved
        results record them with the limits measured from the machine's memory.

        Cases (in order): open (files opened, their pages counted); text-lines, text-content (every page's text through
        the lines layout, and with none); text-parallel (text-lines over the CPU threads: the files at once, or a file's
        pages); images (every image decoded); draw (pages drawn at each --dpi); write-image (each file's images written
        as an image PDF); write-vector (each file's paths and text written back as a vector PDF); edit-incremental,
        edit-rewrite (the information changed and saved); fill (every text field filled, saved); sign (signed with the
        sample generator's test certificate, saved incrementally); check (every signature sign made, checked).

        Sets: samples (test-output/documents/samples, the hostile samples its manifest names left out), large
        (test-output/documents/large: each of its files and folders a set of its own), or any folder or PDF file.
        A file no format opens (a password, damage) is listed as refused and left out of every run.

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
              --hash           hash every run's output (text, pixels, written bytes; SHA-256) into the results, so two
                               runs or two saved results show the same output (the hashing is inside the runs' time)
              --save NAME      keep the results as NAME under the cache folder (bench/NAME.json)
              --compare NAME   show the change against the results saved as NAME (and, hashed, whether outputs changed)

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
        var read = new DocumentOptions();
        context.Write(string.Create(CultureInfo.InvariantCulture,
            $"documents on the CPU ({Environment.ProcessorCount} cores, {ComputeResources.MaxCpuThreads} threads): {string.Join(", ", options.Cases)}; "
            + $"cold: a case's first run on a set, warm: the median of {options.Repeats} runs after it"));
        context.Write(string.Create(CultureInfo.InvariantCulture,
            $"  limits measured from the memory: a stream {Size(read.MaxStreamBytes)}, a document's cache {Size(read.CacheBytes)}, {read.MaxObjects:N0} objects, nesting {read.MaxDepth}"));
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
                measured.Add(done);
                progress.Erase();
                context.Write(Line(done));
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
        var document = Document(options, sets, measured, results, read);
        if (saved is not null)
        {
            document["compare"] = BenchCommand.Compare(context, compareName!, saved, [.. results.Select(r => ("", r))], DocumentBench.Method);
            document["compare"]!["outputs"] = CompareOutputs(context, compareName!, saved, measured);
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
                font = FontFormats.Read(fontFile.Text) is { Count: > 0 } faces ? faces[0] : throw new UsageException($"{fontFile.From}: {fontFile.Text} holds no font.");
            }
            catch (Exception e) when (e is IOException or InvalidDataException or NotSupportedException or UnauthorizedAccessException or ArgumentException)
            {
                throw new UsageException($"{fontFile.From}: {fontFile.Text}: {e.Message}");
            }
        }

        var (signing, trust, at) = Signing(context, cases, sets, out certificateNote);
        return new DocumentBenchOptions
        {
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
        string output = m.SameOutput switch { true => ", the same output every run", false => ", OUTPUT DIFFERS BETWEEN RUNS", null => "" };
        return string.Create(CultureInfo.InvariantCulture,
            $"  {m.Label} on {m.Set.Name}: {Work(m)}{refused}; warm {BenchCommand.Format(m.WarmMs)} ms ({PerUnit(m)}), {Size(m.WarmAllocated)} allocated, working set {Size(m.WarmPeak)}; "
            + $"cold {BenchCommand.Format(cold.Milliseconds)} ms ({BenchCommand.Format(cold.JitMilliseconds)} ms compiling {cold.JitMethods:N0} methods), {Size(cold.Allocated)}, {Size(cold.PeakWorkingSet)}{output}");
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

    // Whether each hashed case gave the output it gave when saved (rule 71: a change that makes things faster changes no output).
    private static JsonArray CompareOutputs(CommandContext context, string name, JsonObject saved, List<Measured> measured)
    {
        var before = (saved["cases"] as JsonArray ?? []).OfType<JsonObject>()
            .Where(c => (string?)c["output"] is not null)
            .GroupBy(c => $"{(string?)c["label"]} {(string?)c["set"]}")
            .ToDictionary(g => g.Key, g => (string)g.First()["output"]!);
        var rows = new JsonArray();
        var changed = new List<string>();
        int same = 0;
        foreach (var m in measured.Where(m => m.Cold.Work.Output is not null))
        {
            string key = $"{m.Label} {m.Set.Name}";
            if (!before.TryGetValue(key, out string? output))
            {
                continue;
            }

            bool equal = output == m.Cold.Work.Output;
            same += equal ? 1 : 0;
            if (!equal)
            {
                changed.Add(key);
            }

            rows.Add(item: new JsonObject { ["case"] = m.Label, ["set"] = m.Set.Name, ["same"] = equal });
        }

        if (rows.Count > 0)
        {
            context.Write(changed.Count == 0
                ? $"  outputs: all {same} hashed cases the same as '{name}'"
                : $"  outputs: {same} of {rows.Count} hashed cases the same as '{name}'; CHANGED: {string.Join(", ", changed)}");
        }

        return rows;
    }

    private static JsonObject Document(DocumentBenchOptions options, List<DocumentBenchSet> sets, List<Measured> measured, List<BenchResult> results, DocumentOptions read)
    {
        // The library's document settings as the environment gives them (every variable the table names for this command,
        // and the CPU's), beside what they come to on this machine.
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
                ["limits"] = new JsonObject
                {
                    ["max_stream_bytes"] = read.MaxStreamBytes,
                    ["max_objects"] = read.MaxObjects,
                    ["max_depth"] = read.MaxDepth,
                    ["cache_bytes"] = read.CacheBytes,
                },
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
                    .Select(p => (JsonNode)new JsonObject { ["file"] = Path.GetRelativePath(s.Root, p.Key), ["reason"] = p.Value })]),
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
                ["refused"] = new JsonArray([.. m.Cold.Work.Refused.Select(r => (JsonNode)new JsonObject { ["file"] = Path.GetRelativePath(m.Set.Root, r.File), ["reason"] = r.Reason })]),
                ["output"] = m.Cold.Work.Output,
                ["same_output"] = m.SameOutput,
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
}
