// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Data.Abstractions;

namespace Idrak.Data.Code;

/// <summary>What a code extraction wrote: its folder, its manifest and the counts in it.</summary>
/// <param name="Output">The output folder.</param>
/// <param name="ManifestPath">Its <c>manifest.json</c>.</param>
/// <param name="Manifest">The manifest as written.</param>
public sealed record CodeExtraction(string Output, string ManifestPath, JsonObject Manifest)
{
    /// <summary>Rows written (every family's, after the row filters).</summary>
    public long Rows => (long?)Manifest["totals"]?["rows"] ?? 0;

    /// <summary>Drafts the families proposed.</summary>
    public long Drafts => (long?)Manifest["totals"]?["drafts"] ?? 0;

    /// <summary>Task folders written, each proven by the verifier.</summary>
    public long Tasks => (long?)Manifest["totals"]?["tasks"] ?? 0;

    /// <summary>Drafts written unproven (no verifier).</summary>
    public long Unproven => (long?)Manifest["totals"]?["unproven"] ?? 0;

    /// <summary>Drafts rejected (the verifier's reason, or the layout's).</summary>
    public long Rejected => (long?)Manifest["totals"]?["rejected"] ?? 0;

    /// <summary>Why the run stopped early ("max tasks", "max time"), or null when it read everything.</summary>
    public string? Stopped => (string?)Manifest["stopped"];
}

/// <summary>
/// Turns a repository or project into data (plan 36, <c>idrak data code</c>): it walks the source once (rule 72; files are
/// told apart by their bytes, rule 74), runs the chosen task families (<see cref="TaskFamilies"/>) whose languages match the
/// files', writes their rows through the row filters (<see cref="RowFilters"/>) and their drafts as task folders, and writes
/// <c>manifest.json</c>: the families, the counts asked, kept and rejected with the reasons, and the time.
/// </summary>
/// <remarks>
/// Under the output folder: <c>rows/{family}.jsonl</c>; <c>tasks/{family}/{id}/</c> for the drafts the verifier proved
/// (<see cref="TaskVerifiers"/>: the check fails on the workspace and passes with the solution); <c>drafts/{family}/{id}/</c>,
/// marked <c>"proven": false</c>, when no verifier is chosen or registered (never under <c>tasks</c>); the proof's output of each
/// rejected draft under <c>rejected/{family}/{id}.txt</c>. A task folder is the coding tasks' layout: <c>task.json</c>
/// (prompt, language, the source as its base, setup, verify, environment, tags), <c>workspace/</c>, <c>verify/</c>,
/// <c>solution/</c> (the source's own files when the draft names none).
/// </remarks>
/// <param name="options">The run's settings; null for the defaults.</param>
public sealed class CodeExtractor(CodeExtractorOptions? options = null)
{
    /// <summary>The run's settings.</summary>
    public CodeExtractorOptions Options { get; } = options ?? new CodeExtractorOptions();

    /// <summary>Where the run shows what it does (the walk, each family's rows and drafts: how far, how fast, time left); null for nowhere.</summary>
    public IProgressReporter? Progress { get; init; }

    // The bytes of a file's start read to tell a binary file (a zero byte in them, as git tells one).
    private const int SniffBytes = 8000;

    /// <summary>
    /// The files of <paramref name="source"/> as a run walks them: in name order (ordinal), folders after the files of their
    /// parent; folders in <paramref name="skippedFolders"/> (by name, ignoring case; null: the source's read options') and
    /// links to folders not entered; binary files (a zero byte in the first 8,000) and files over <paramref name="maxFileBytes"/>
    /// (null: the source's read options' largest document) left out.
    /// </summary>
    public static IReadOnlyList<CodeSourceFile> Walk(CodeSource source, IReadOnlyCollection<string>? skippedFolders = null, long? maxFileBytes = null) =>
        Walk(source, skippedFolders, maxFileBytes, excluded: null, skipped: null);

    private static List<CodeSourceFile> Walk(CodeSource source, IReadOnlyCollection<string>? skippedFolders, long? maxFileBytes, string? excluded, Dictionary<string, long>? skipped)
    {
        ArgumentNullException.ThrowIfNull(source);
        string root = Path.GetFullPath(source.Root);
        if (!Directory.Exists(root))
        {
            throw new DirectoryNotFoundException($"The source '{source.Root}' is not a folder.");
        }

        var folders = new HashSet<string>(skippedFolders ?? source.ReadOptions.SkippedFolders, StringComparer.OrdinalIgnoreCase);
        long largest = maxFileBytes ?? source.ReadOptions.MaxDocumentBytes;
        var files = new List<CodeSourceFile>();
        var head = new byte[SniffBytes];
        var pending = new Stack<string>();
        pending.Push(root);
        while (pending.Count > 0)
        {
            string folder = pending.Pop();
            string[] entries;
            string[] children;
            try
            {
                entries = Directory.GetFiles(folder);
                children = Directory.GetDirectories(folder);
            }
            catch (Exception e) when (e is UnauthorizedAccessException or IOException)
            {
                Count(skipped, "a folder that could not be read");
                continue;
            }

            Array.Sort(entries, StringComparer.Ordinal);
            foreach (string path in entries)
            {
                var info = new FileInfo(path);
                if (info.Length > largest)
                {
                    Count(skipped, string.Create(CultureInfo.InvariantCulture, $"larger than {largest} bytes"));
                    continue;
                }

                if (IsBinary(path, head))
                {
                    Count(skipped, "binary (a zero byte in its first 8,000)");
                    continue;
                }

                string relative = Path.GetRelativePath(root, path).Replace('\\', '/');
                files.Add(new CodeSourceFile(relative, DataFiles.LanguageOf(path), info.Length));
            }

            Array.Sort(children, StringComparer.Ordinal);
            for (int i = children.Length - 1; i >= 0; i--)                 // pushed in reverse: walked in name order
            {
                string child = children[i];
                if (folders.Contains(Path.GetFileName(child)) || new DirectoryInfo(child).LinkTarget is not null
                    || (excluded is not null && Path.GetFullPath(child).Equals(excluded, StringComparison.OrdinalIgnoreCase)))
                {
                    continue;
                }

                pending.Push(child);
            }
        }

        return files;
    }

    // A zero byte in the file's first bytes: a binary file. The bytes are read, never decoded (rule 74).
    private static bool IsBinary(string path, byte[] head)
    {
        try
        {
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1, FileOptions.SequentialScan);
            int read = stream.ReadAtLeast(head, head.Length, throwOnEndOfStream: false);
            return head.AsSpan(0, read).Contains((byte)0);
        }
        catch (Exception e) when (e is UnauthorizedAccessException or IOException)
        {
            return true;
        }
    }

    private static void Count(Dictionary<string, long>? counts, string reason)
    {
        if (counts is not null)
        {
            lock (counts)
            {
                counts[reason] = counts.GetValueOrDefault(reason) + 1;
            }
        }
    }

    /// <summary>
    /// Runs the extraction of <paramref name="source"/> into <paramref name="output"/> (created; files of an earlier run there
    /// are replaced as they are written) and writes its manifest.
    /// </summary>
    /// <exception cref="NotSupportedException">A family or verifier nobody registered ("not registered"), or a row format that cannot be written.</exception>
    public async Task<CodeExtraction> RunAsync(CodeSource source, string output, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentException.ThrowIfNullOrWhiteSpace(output);
        var options = Options;
        output = Path.GetFullPath(output);
        string root = Path.GetFullPath(source.Root);

        // Everything named is checked before any work: the families, the verifier, the formats.
        var families = (options.Families ?? [.. TaskFamilies.Names]).Distinct(StringComparer.OrdinalIgnoreCase).Select(TaskFamilies.Get).ToList();
        var verifier = TaskVerifiers.Choose(options.Verifier);
        foreach (string format in options.Formats)
        {
            var known = DataFileFormats.Get(format);
            if (!known.Name.Equals(nameof(DataFormat.JsonLines), StringComparison.OrdinalIgnoreCase))
            {
                throw new NotSupportedException($"Rows are written as {nameof(DataFormat.JsonLines)}; the data format '{known.Name}' is read only in Idrak.Data, so a code extraction cannot write it.");
            }
        }

        var started = DateTimeOffset.UtcNow;
        var clock = Stopwatch.StartNew();
        Directory.CreateDirectory(output);
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        string? stopped = null;
        if (options.MaxTime is { } limit)
        {
            stop.CancelAfter(limit);
        }

        // The walk, once, for every family.
        var progress = Progress;
        progress?.NextStage($"walking {root}", null, ProgressUnit.Items, keepLine: false);
        var skippedFiles = new Dictionary<string, long>(StringComparer.Ordinal);
        string? inside = output.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase) ? output : null;
        var files = Walk(source, options.SkippedFolders, options.MaxFileBytes, inside, skippedFiles);
        progress?.Report(files.Count);

        var filters = RowFilters.Names.Select(RowFilters.Get).ToList();
        var state = new RunState(options.MaxTasks);
        var reports = new JsonArray();
        var byLanguages = new Dictionary<string, IReadOnlyList<CodeSourceFile>>(StringComparer.OrdinalIgnoreCase);   // one list per language set
        foreach (var family in families)
        {
            var report = new FamilyReport(family);
            reports.Add((JsonNode)report.Json);
            if (stopped is null && stop.IsCancellationRequested)
            {
                stopped = cancellationToken.IsCancellationRequested ? null : StopReason(options, state);
            }

            if (stopped is not null)
            {
                report.Skip($"not run: the run stopped ({stopped})");
                report.Finish();
                continue;
            }

            if (family.NeedsHistory && source.History is null)
            {
                report.Skip("not run: it needs the repository's history, and this source has none");
                report.Finish();
                continue;
            }

            string key = string.Join(",", family.Languages.Order(StringComparer.OrdinalIgnoreCase));
            if (!byLanguages.TryGetValue(key, out var mine))
            {
                mine = family.Languages.Count == 0 ? files
                    : [.. files.Where(f => f.Language is not null && family.Languages.Contains(f.Language, StringComparer.OrdinalIgnoreCase))];
                byLanguages[key] = mine;
            }

            report.Files = mine.Count;
            var familySource = source with { Root = root, Files = mine };
            try
            {
                await WriteRowsAsync(family, familySource, output, filters, report, progress, stop.Token).ConfigureAwait(false);
                await WriteDraftsAsync(family, familySource, output, verifier, report, state, progress, stop, stop.Token).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
            {
                stopped = StopReason(options, state);
            }

            report.Finish();
        }

        cancellationToken.ThrowIfCancellationRequested();
        stopped ??= state.Full ? "max tasks" : null;
        TryDelete(Path.Combine(output, ".staging"));
        TryDelete(Path.Combine(output, ".work"));
        progress?.Finish();

        var manifest = new JsonObject
        {
            ["source"] = root,
            ["output"] = output,
            ["started"] = started.ToString("O", CultureInfo.InvariantCulture),
            ["finished"] = DateTimeOffset.UtcNow.ToString("O", CultureInfo.InvariantCulture),
            ["seconds"] = Math.Round(clock.Elapsed.TotalSeconds, 3),
            ["stopped"] = stopped,
            ["verifier"] = verifier?.Name,
            ["options"] = OptionsJson(options),
            ["files"] = new JsonObject
            {
                ["walked"] = files.Count,
                ["skipped"] = Counts(skippedFiles),
                ["languages"] = Counts(files.Where(f => f.Language is not null).GroupBy(f => f.Language!, StringComparer.Ordinal).ToDictionary(g => g.Key, g => (long)g.Count(), StringComparer.Ordinal)),
            },
            ["rowFilters"] = new JsonArray([.. filters.Select(f => (JsonNode)f.Name)]),
            ["families"] = reports,
            ["totals"] = new JsonObject
            {
                ["rows"] = Sum(reports, "rows", "written"),
                ["drafts"] = Sum(reports, "drafts", "asked"),
                ["tasks"] = Sum(reports, "drafts", "tasks"),
                ["unproven"] = Sum(reports, "drafts", "unproven"),
                ["rejected"] = reports.Sum(r => ((JsonObject?)r!["drafts"]!["rejected"])?.Sum(p => (long?)p.Value ?? 0) ?? 0),
            },
            ["splits"] = "not made in this run: group splits and the leakage check are plan 36's G2 (the split settings are recorded above)",
        };
        string manifestPath = Path.Combine(output, "manifest.json");
        await File.WriteAllTextAsync(manifestPath, manifest.ToJsonString(new JsonSerializerOptions { WriteIndented = true }) + "\n", cancellationToken).ConfigureAwait(false);
        return new CodeExtraction(output, manifestPath, manifest);
    }

    private static string StopReason(CodeExtractorOptions options, RunState state) => state.Full ? "max tasks"
        : string.Create(CultureInfo.InvariantCulture, $"max time ({options.MaxTime?.TotalSeconds:0.###} s)");

    private static long Sum(JsonArray reports, string part, string field) => reports.Sum(r => (long?)r![part]![field] ?? 0);

    private static JsonObject Counts(Dictionary<string, long> counts)
    {
        var json = new JsonObject();
        foreach (var (key, value) in counts.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            json[key] = value;
        }

        return json;
    }

    private static JsonObject OptionsJson(CodeExtractorOptions o) => new()
    {
        ["families"] = o.Families is null ? (JsonNode)"all" : new JsonArray([.. o.Families.Select(f => (JsonNode)f)]),
        ["skippedFolders"] = o.SkippedFolders is null ? null : new JsonArray([.. o.SkippedFolders.Select(f => (JsonNode)f)]),
        ["maxFileBytes"] = o.MaxFileBytes,
        ["dedupe"] = o.DedupeThreshold,
        ["seed"] = o.Seed,
        ["splitBy"] = o.SplitBy,
        ["evalFraction"] = o.EvalFraction,
        ["evalGroups"] = new JsonArray([.. o.EvalGroups.Select(g => (JsonNode)g)]),
        ["maxTasks"] = o.MaxTasks,
        ["maxTime"] = o.MaxTime?.TotalSeconds,
        ["formats"] = new JsonArray([.. o.Formats.Select(f => (JsonNode)f)]),
        ["concurrency"] = o.EffectiveConcurrency,
        ["verifier"] = o.Verifier,
    };

    // ------------------------------------------------------------------ rows

    // The family's rows through the row filters into rows/{family}.jsonl (the file made only when a row is written). The
    // rows come as an async stream and the filters read an enumerable: a bounded hand-over joins them, so a repository's
    // rows are never all held at once.
    private async Task WriteRowsAsync(ITaskFamily family, CodeSource source, string output, List<IRowFilter> filters, FamilyReport report,
        IProgressReporter? progress, CancellationToken cancellationToken)
    {
        progress?.NextStage($"{family.Name}: rows", source.Files.Count, ProgressUnit.Items, keepLine: false);
        using var handOver = new BlockingCollection<JsonObject>(boundedCapacity: 64);
        var context = new RowFilterContext { Threshold = Options.DedupeThreshold, Rejected = (_, why) => report.RowRejected(why) };
        string path = Path.Combine(output, "rows", SafeName(family.Name) + ".jsonl");
        TryDeleteFile(path);
        var writer = Task.Run(() =>
        {
            IEnumerable<JsonObject> rows = handOver.GetConsumingEnumerable(cancellationToken);
            foreach (var filter in filters)
            {
                rows = filter.Apply(rows, context);
            }

            StreamWriter? file = null;
            try
            {
                foreach (var row in rows)
                {
                    if (file is null)
                    {
                        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
                        file = new StreamWriter(path, append: false, new UTF8Encoding(false));
                    }

                    file.Write(row.ToJsonString());
                    file.Write('\n');
                    report.RowWritten();
                }
            }
            finally
            {
                file?.Dispose();
            }
        }, cancellationToken);
        try
        {
            await foreach (var row in family.RowsAsync(source, cancellationToken).ConfigureAwait(false))
            {
                handOver.Add(row, cancellationToken);
                report.RowAsked();
                progress?.Advance();
            }
        }
        finally
        {
            handOver.CompleteAdding();
            await writer.ConfigureAwait(false);
        }

        if (report.RowsWritten > 0)
        {
            report.RowsPath = Path.GetRelativePath(output, path).Replace('\\', '/');
        }
    }

    // ------------------------------------------------------------------ drafts

    private async Task WriteDraftsAsync(ITaskFamily family, CodeSource source, string output, ITaskVerifier? verifier, FamilyReport report, RunState state,
        IProgressReporter? progress, CancellationTokenSource stop, CancellationToken cancellationToken)
    {
        progress?.NextStage($"{family.Name}: {(verifier is null ? "drafts (unproven: no verifier)" : $"tasks (proven by {verifier.Name})")}", Options.MaxTasks, ProgressUnit.Items, keepLine: false);
        if (progress is not null)
        {
            progress.Detail = () => report.Line;
        }

        var ids = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var parallel = new ParallelOptions { MaxDegreeOfParallelism = Options.EffectiveConcurrency, CancellationToken = cancellationToken };
        await Parallel.ForEachAsync(family.DraftAsync(source, cancellationToken), parallel, async (draft, token) =>
        {
            report.DraftAsked();
            string id;
            lock (ids)
            {
                id = SafeName(draft.Id);
                for (int n = 2; !ids.Add(id); n++)
                {
                    id = string.Create(CultureInfo.InvariantCulture, $"{SafeName(draft.Id)}-{n}");
                }
            }

            if (draft.Workspace.Any(f => f.Delete))
            {
                report.Rejected("the workspace removes a file (a task folder's workspace only writes files)");
                return;
            }

            if (!state.TryReserve())
            {
                await stop.CancelAsync().ConfigureAwait(false);
                return;
            }

            string folder = SafeName(draft.Family.Length > 0 ? draft.Family : report.Name);
            string staging = Path.Combine(output, ".staging", folder, id);
            bool kept = false;
            try
            {
                TryDelete(staging);
                WriteTaskFolder(staging, draft, source, proven: verifier is not null);
                if (verifier is null)
                {
                    Move(staging, Path.Combine(output, "drafts", folder, id));
                    report.Unproven();
                    kept = true;
                    return;
                }

                string work = Path.Combine(output, ".work", folder, id);
                TryDelete(work);
                TaskProof proof;
                try
                {
                    proof = await verifier.ProveAsync(staging, work, token).ConfigureAwait(false);
                }
                catch (Exception e) when (e is not OperationCanceledException)
                {
                    proof = new TaskProof(false, false, $"the verifier failed: {e.GetType().Name}: {e.Message}");
                    report.Rejected($"the verifier failed ({e.GetType().Name})");
                    WriteRejected(output, folder, id, proof);
                    return;
                }
                finally
                {
                    TryDelete(work);
                }

                if (proof.Proven)
                {
                    Move(staging, Path.Combine(output, "tasks", folder, id));
                    report.Kept();
                    kept = true;
                }
                else
                {
                    report.Rejected(proof.Reason!);
                    WriteRejected(output, folder, id, proof);
                }
            }
            finally
            {
                if (!kept)
                {
                    state.Release();
                }

                TryDelete(staging);
                progress?.Advance();
            }
        }).ConfigureAwait(false);
    }

    private static void WriteRejected(string output, string family, string id, TaskProof proof)
    {
        string path = Path.Combine(output, "rejected", family, id + ".txt");
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, $"{proof.Reason}\n\n{proof.Output}");
    }

    // The task's folder in the coding tasks' layout: task.json, workspace/, verify/, solution/.
    private static void WriteTaskFolder(string folder, TaskDraft draft, CodeSource source, bool proven)
    {
        Directory.CreateDirectory(folder);
        WriteFiles(Path.Combine(folder, "workspace"), draft.Workspace);
        WriteFiles(Path.Combine(folder, "verify"), draft.Verify);
        var solution = draft.Solution.Count > 0 ? draft.Solution
            : [.. draft.Workspace.Select(f => Path.Combine(source.Root, f.Path)).Where(File.Exists)
                .Select(p => new CodeFile(Path.GetRelativePath(source.Root, p).Replace('\\', '/'), File.ReadAllBytes(p)))];
        WriteFiles(Path.Combine(folder, "solution"), solution);
        var environment = new JsonObject();
        foreach (var (name, value) in draft.Environment.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            environment[name] = value;
        }

        var task = new JsonObject
        {
            ["id"] = draft.Id,
            ["family"] = draft.Family,
            ["prompt"] = draft.Prompt,
            ["language"] = draft.Language ?? source.Languages.FirstOrDefault(),
            ["base"] = Path.GetFullPath(source.Root),
            ["setup"] = new JsonArray([.. draft.Setup.Select(c => (JsonNode)c)]),
            ["verify"] = new JsonArray([.. draft.VerifyCommands.Select(c => (JsonNode)c)]),
            ["environment"] = environment,
            ["tags"] = new JsonArray([.. draft.Tags.Select(t => (JsonNode)t)]),
            ["provenance"] = new JsonObject
            {
                ["path"] = draft.Provenance.Path, ["commit"] = draft.Provenance.Commit, ["test"] = draft.Provenance.Test, ["group"] = draft.Provenance.Group,
            },
            ["proven"] = proven,
        };
        File.WriteAllText(Path.Combine(folder, "task.json"), task.ToJsonString(new JsonSerializerOptions { WriteIndented = true }) + "\n");
    }

    private static void WriteFiles(string folder, IReadOnlyList<CodeFile> files)
    {
        foreach (var file in files.Where(f => !f.Delete))
        {
            string path = Path.GetFullPath(Path.Combine(folder, file.Path));
            if (!path.StartsWith(Path.GetFullPath(folder) + Path.DirectorySeparatorChar, StringComparison.Ordinal))
            {
                throw new InvalidDataException($"The task file '{file.Path}' is outside its folder.");
            }

            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            using var stream = File.Create(path);
            stream.Write(file.Bytes.Span);
        }
    }

    private static void Move(string from, string to)
    {
        TryDelete(to);
        Directory.CreateDirectory(Path.GetDirectoryName(to)!);
        Directory.Move(from, to);
    }

    private static void TryDelete(string folder)
    {
        try
        {
            if (Directory.Exists(folder))
            {
                Directory.Delete(folder, recursive: true);
            }
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            // Left for the next run (a file a process still holds): it replaces what it writes.
        }
    }

    private static void TryDeleteFile(string path)
    {
        if (File.Exists(path))
        {
            File.Delete(path);
        }
    }

    /// <summary>A name made safe for a folder: letters, digits, '.', '-' and '_' kept, others as '-', at most 120 characters (a hash of the rest after that).</summary>
    public static string SafeName(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        var text = new StringBuilder(name.Length);
        foreach (char c in name)
        {
            text.Append(char.IsAsciiLetterOrDigit(c) || c is '.' or '-' or '_' ? c : '-');
        }

        string safe = text.ToString().Trim('-', '.');
        if (safe.Length == 0)
        {
            safe = "task";
        }

        if (safe.Length > 120)
        {
            uint hash = 2166136261;
            foreach (char c in name)
            {
                hash = (hash ^ c) * 16777619;
            }

            safe = string.Create(CultureInfo.InvariantCulture, $"{safe[..110]}-{hash:x8}");
        }

        return safe;
    }

    // The budget of tasks (and unproven drafts) a run keeps, shared by the families' parallel proofs.
    private sealed class RunState(int? maxTasks)
    {
        private int _reserved;

        public bool Full => maxTasks is { } max && Volatile.Read(ref _reserved) >= max;

        public bool TryReserve()
        {
            if (maxTasks is not { } max)
            {
                return true;
            }

            while (true)
            {
                int now = Volatile.Read(ref _reserved);
                if (now >= max)
                {
                    return false;
                }

                if (Interlocked.CompareExchange(ref _reserved, now + 1, now) == now)
                {
                    return true;
                }
            }
        }

        public void Release()
        {
            if (maxTasks is not null)
            {
                Interlocked.Decrement(ref _reserved);
            }
        }
    }

    // One family's part of the manifest, counted from parallel proofs.
    private sealed class FamilyReport(ITaskFamily family)
    {
        private readonly Lock _gate = new();
        private readonly Dictionary<string, long> _rowsRejected = new(StringComparer.Ordinal), _rejected = new(StringComparer.Ordinal);
        private long _rowsAsked, _rowsWritten, _drafts, _tasks, _unproven;

        public string Name => family.Name;

        public JsonObject Json { get; } = new() { ["name"] = family.Name, ["summary"] = family.Summary, ["languages"] = new JsonArray([.. family.Languages.Select(l => (JsonNode)l)]) };

        public int Files { get; set; }

        public string? RowsPath { get; set; }

        public long RowsWritten => Interlocked.Read(ref _rowsWritten);

        public string Line
        {
            get
            {
                long rejected;
                lock (_gate)
                {
                    rejected = _rejected.Values.Sum();
                }

                return string.Create(CultureInfo.InvariantCulture, $"{Interlocked.Read(ref _tasks)} kept, {Interlocked.Read(ref _unproven)} unproven, {rejected} rejected");
            }
        }

        public void Skip(string why) => Json["skipped"] = why;

        public void RowAsked() => Interlocked.Increment(ref _rowsAsked);

        public void RowWritten() => Interlocked.Increment(ref _rowsWritten);

        public void RowRejected(string why)
        {
            lock (_gate)
            {
                _rowsRejected[why] = _rowsRejected.GetValueOrDefault(why) + 1;
            }
        }

        public void DraftAsked() => Interlocked.Increment(ref _drafts);

        public void Kept() => Interlocked.Increment(ref _tasks);

        public void Unproven() => Interlocked.Increment(ref _unproven);

        public void Rejected(string why)
        {
            lock (_gate)
            {
                _rejected[why] = _rejected.GetValueOrDefault(why) + 1;
            }
        }

        public void Finish()
        {
            lock (_gate)
            {
                Fill();
            }
        }

        private void Fill()
        {
            Json["files"] = Files;
            Json["rows"] = new JsonObject { ["asked"] = _rowsAsked, ["written"] = _rowsWritten, ["rejected"] = Counts(_rowsRejected), ["path"] = RowsPath };
            Json["drafts"] = new JsonObject { ["asked"] = _drafts, ["tasks"] = _tasks, ["unproven"] = _unproven, ["rejected"] = Counts(_rejected) };
        }
    }
}
