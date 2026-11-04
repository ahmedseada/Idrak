// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using Idrak.Data;
using Idrak.Data.Abstractions;

// Plan 36, G0: the code dataset tool's contracts in Idrak.Data.Abstractions (task families, repository histories, row
// filters, their registries) and CodeExtractorOptions with its IDRAK_CODE_* variables. No extraction yet (G1).
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] DataCodeContractsGroup =
    [
        ("data code contracts: a task family registered is found by name (ignoring case), listed, described, its origin the registering assembly, and drafts as its contract says; unregistering leaves TaskFamilies as found", d => { if (d == Device.Cpu) TaskFamiliesRegistry(); }),
        ("data code contracts: a repository history and a row filter registered are found, chosen (a history by the root it reads), applied and unregistered, each registry left as found", d => { if (d == Device.Cpu) HistoriesAndFiltersRegistries(); }),
        ("data code contracts: an unknown task family, repository history or row filter is refused with \"not registered\", naming it and the registered ones", d => { if (d == Device.Cpu) CodeContractsNotRegistered(); }),
        ("data code contracts: CodeExtractorOptions' defaults are the documented ones, each IDRAK_CODE_* variable sets its setting (0 or all for no limit), a bad value is refused naming the variable, and the records hold their values (a file as bytes, a removal, a draft's defaults)", d => { if (d == Device.Cpu) CodeExtractorOptionsRead(); }),
    ];

    private static void TaskFamiliesRegistry()
    {
        var before = TaskFamilies.Names.ToList();
        var family = new TestTaskFamily();
        TaskFamilies.Register(family);
        try
        {
            Check(TaskFamilies.Get("CODE-TEST-FAMILY") == family && TaskFamilies.Find("code-test-family") == family && TaskFamilies.Names.Contains("code-test-family"),
                $"the family is found ignoring case and listed: {string.Join(", ", TaskFamilies.Names)}");
            Check(TaskFamilies.Origin("code-test-family") == typeof(DesktopTests).Assembly.GetName().Name && TaskFamilies.Default("code-test-family") is null,
                $"its origin is the registering assembly ({TaskFamilies.Origin("code-test-family")}), and the library has none of that name");
            Check(TaskFamilies.Describe().Contains("code-test-family: drafts one task per file", StringComparison.Ordinal), $"described: {TaskFamilies.Describe()}");
            ITaskFamily contract = family;
            Check(contract.Languages.Count == 0 && !contract.NeedsHistory, "a family's defaults: any language, no history needed");

            string root = Path.Combine(Path.GetTempPath(), "idrak-code-contracts-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(root);
            try
            {
                File.WriteAllText(Path.Combine(root, "a.txt"), "one");
                File.WriteAllText(Path.Combine(root, "b.txt"), "two");
                var drafts = Drafts(family, new CodeSource(root));
                Check(drafts.Count == 2 && drafts[0] is { Id: "a.txt", Family: "code-test-family", Provenance.Path: "a.txt" } && drafts[0].Solution[0].Bytes.Span.SequenceEqual("one"u8),
                    $"its drafts: {string.Join("; ", drafts.Select(d => $"{d.Id} {d.Family}"))}");
            }
            finally
            {
                Directory.Delete(root, recursive: true);
            }
        }
        finally
        {
            TaskFamilies.Unregister("code-test-family");
        }

        Check(TaskFamilies.Names.SequenceEqual(before) && TaskFamilies.Find("code-test-family") is null, "unregistering leaves the registry as found");
    }

    private static List<TaskDraft> Drafts(ITaskFamily family, CodeSource source)
    {
        var drafts = new List<TaskDraft>();
        var enumerator = family.DraftAsync(source).GetAsyncEnumerator();
        try
        {
            while (enumerator.MoveNextAsync().AsTask().GetAwaiter().GetResult())
            {
                drafts.Add(enumerator.Current);
            }
        }
        finally
        {
            enumerator.DisposeAsync().AsTask().GetAwaiter().GetResult();
        }

        return drafts;
    }

    private static void HistoriesAndFiltersRegistries()
    {
        var histories = RepositoryHistories.Names.ToList();
        var filters = RowFilters.Names.ToList();
        var history = new TestHistory();
        var filter = new TestFilter();
        RepositoryHistories.Register(history);
        RowFilters.Register(filter);
        try
        {
            Check(RepositoryHistories.Get("Code-Test-History") == history && RepositoryHistories.For("/code-test-repo") == history && RepositoryHistories.For("/elsewhere") is null,
                "the history is found by name and chosen by the root it reads");
            var commits = new List<CodeCommit>();
            var enumerator = history.CommitsAsync("/code-test-repo", new HistoryQuery { MaxCount = 1 }).GetAsyncEnumerator();
            while (enumerator.MoveNextAsync().AsTask().GetAwaiter().GetResult())
            {
                commits.Add(enumerator.Current);
            }

            enumerator.DisposeAsync().AsTask().GetAwaiter().GetResult();
            Check(commits is [{ Id: "c1", Changes: [{ Path: "a.cs", Kind: CodeChangeKind.Modified }] }], $"its commits as the query asks: {commits.Count}");
            using (var diff = history.OpenDiffAsync("/code-test-repo", "c1").AsTask().GetAwaiter().GetResult())
            {
                Check(new StreamReader(diff).ReadToEnd() == "diff c1", "its diff is a stream");
            }

            Check(history.OpenFileAsync("/code-test-repo", "c1", "missing.cs").AsTask().GetAwaiter().GetResult() is null, "a file the commit lacks is null");

            Check(RowFilters.Get("CODE-TEST-FILTER") == filter, "the filter is found by name");
            var rejected = new List<string>();
            JsonObject[] rows = [new() { ["text"] = "keep" }, new() { ["text"] = "" }, new() { ["text"] = "also" }];
            var kept = filter.Apply(rows, new RowFilterContext { Rejected = (_, why) => rejected.Add(why) }).ToList();
            Check(kept.Count == 2 && (string?)kept[1]["text"] == "also" && rejected.SequenceEqual(["empty"]), "the filter keeps rows in order and tells the dropped one with its reason");
        }
        finally
        {
            RepositoryHistories.Unregister("code-test-history");
            RowFilters.Unregister("code-test-filter");
        }

        Check(RepositoryHistories.Names.SequenceEqual(histories) && RowFilters.Names.SequenceEqual(filters) && RepositoryHistories.Find("code-test-history") is null
              && RowFilters.Find("code-test-filter") is null, "unregistering leaves both registries as found");
    }

    private static void CodeContractsNotRegistered()
    {
        TaskFamilies.Register(new TestTaskFamily());
        try
        {
            ThrowsNaming<NotRegisteredException>(() => TaskFamilies.Get("code-test-nobody"), "an unknown task family", "'code-test-nobody'", "not registered", "code-test-family", "TaskFamilies.Register");
        }
        finally
        {
            TaskFamilies.Unregister("code-test-family");
        }

        ThrowsNaming<NotRegisteredException>(() => RepositoryHistories.Get("code-test-nobody"), "an unknown repository history", "'code-test-nobody'", "not registered", "RepositoryHistories.Register");
        ThrowsNaming<NotRegisteredException>(() => RowFilters.Get("code-test-nobody"), "an unknown row filter", "'code-test-nobody'", "not registered", "RowFilters.Register");
        try
        {
            TaskFamilies.Get("code-test-nobody");
        }
        catch (NotRegisteredException e)
        {
            Check(e.Registry == nameof(TaskFamilies) && e is NotSupportedException, $"the exception names its registry ({e.Registry}) and is a NotSupportedException");
        }
    }

    private static void CodeExtractorOptionsRead()
    {
        string[] variables =
        [
            CodeExtractorOptions.FamiliesVariable, CodeExtractorOptions.SkippedFoldersVariable, CodeExtractorOptions.MaxFileBytesVariable, CodeExtractorOptions.DedupeVariable,
            CodeExtractorOptions.SeedVariable, CodeExtractorOptions.SplitByVariable, CodeExtractorOptions.EvalFractionVariable, CodeExtractorOptions.EvalGroupsVariable,
            CodeExtractorOptions.MaxTasksVariable, CodeExtractorOptions.MaxTimeVariable, CodeExtractorOptions.FormatsVariable, CodeExtractorOptions.ConcurrencyVariable,
        ];
        Check(variables.All(v => v.StartsWith("IDRAK_CODE_", StringComparison.Ordinal)) && variables.Distinct().Count() == 12, "twelve IDRAK_CODE_* variables");

        var unset = variables.ToDictionary(v => v, _ => (string?)null);
        WithEnvironment(unset, () =>
        {
            var defaults = CodeExtractorOptions.FromEnvironment();
            Check(defaults == new CodeExtractorOptions() || defaults is { Families: null, SkippedFolders: null, Seed: 0, EvalGroups.Count: 0, MaxTasks: null, MaxTime: null, Concurrency: 0 },
                "unset: the defaults");
            Check(defaults is { MaxFileBytes: 1L << 20, DedupeThreshold: CodeExtractorOptions.DefaultDedupeThreshold, SplitBy: "project", EvalFraction: CodeExtractorOptions.DefaultEvalFraction }
                  && defaults.Formats.SequenceEqual(["JsonLines"]) && defaults.EffectiveConcurrency == ComputeResources.MaxCpuThreads,
                $"the documented defaults (the reader's largest file, 0.85, project, 0.1, JSON Lines, the CPU threads): {defaults}");
        });

        var set = new Dictionary<string, string?>
        {
            [CodeExtractorOptions.FamiliesVariable] = "a, b", [CodeExtractorOptions.SkippedFoldersVariable] = "bin,obj", [CodeExtractorOptions.MaxFileBytesVariable] = "4096",
            [CodeExtractorOptions.DedupeVariable] = "0.9", [CodeExtractorOptions.SeedVariable] = "7", [CodeExtractorOptions.SplitByVariable] = "test-group",
            [CodeExtractorOptions.EvalFractionVariable] = "0.25", [CodeExtractorOptions.EvalGroupsVariable] = "g1,g2", [CodeExtractorOptions.MaxTasksVariable] = "50",
            [CodeExtractorOptions.MaxTimeVariable] = "90", [CodeExtractorOptions.FormatsVariable] = "JsonLines, Parquet", [CodeExtractorOptions.ConcurrencyVariable] = "3",
        };
        WithEnvironment(set, () =>
        {
            var o = CodeExtractorOptions.FromEnvironment();
            Check(o.Families!.SequenceEqual(["a", "b"]) && o.SkippedFolders!.SequenceEqual(["bin", "obj"]) && o.MaxFileBytes == 4096 && o.DedupeThreshold == 0.9 && o.Seed == 7
                  && o.SplitBy == "test-group" && o.EvalFraction == 0.25 && o.EvalGroups.SequenceEqual(["g1", "g2"]) && o.MaxTasks == 50 && o.MaxTime == TimeSpan.FromSeconds(90)
                  && o.Formats.SequenceEqual(["JsonLines", "Parquet"]) && o.Concurrency == 3 && o.EffectiveConcurrency == 3, $"each variable sets its setting: {o}");
        });

        WithEnvironment(new Dictionary<string, string?>(unset) { [CodeExtractorOptions.FamiliesVariable] = "all", [CodeExtractorOptions.MaxTasksVariable] = "0", [CodeExtractorOptions.MaxTimeVariable] = "0" }, () =>
            Check(CodeExtractorOptions.FromEnvironment(new CodeExtractorOptions { Families = ["x"], MaxTasks = 5, MaxTime = TimeSpan.FromMinutes(1) }) is { Families: null, MaxTasks: null, MaxTime: null },
                "all and 0 mean every family and no limit, over the defaults given"));

        foreach (var (variable, value, takes) in new[]
                 {
                     (CodeExtractorOptions.MaxFileBytesVariable, "0", "above 0"), (CodeExtractorOptions.DedupeVariable, "1.5", "at most 1"), (CodeExtractorOptions.SeedVariable, "x", "whole number"),
                     (CodeExtractorOptions.EvalFractionVariable, "1", "below 1"), (CodeExtractorOptions.MaxTasksVariable, "-1", "0 for no limit"), (CodeExtractorOptions.MaxTimeVariable, "soon", "seconds"),
                     (CodeExtractorOptions.FormatsVariable, ",", "data format names"), (CodeExtractorOptions.ConcurrencyVariable, "-2", "CPU threads"),
                 })
        {
            WithEnvironment(new Dictionary<string, string?>(unset) { [variable] = value }, () =>
                ThrowsNaming<ArgumentException>(() => CodeExtractorOptions.FromEnvironment(), $"{variable}={value}", variable, $"'{value}'", takes));
        }

        // The records: a file as bytes, a removal, a draft's defaults, a source's.
        var file = new CodeFile("src/a.cs", "x"u8.ToArray());
        Check(file is { Path: "src/a.cs", Delete: false } && file.Bytes.Length == 1 && CodeFile.Removed("old.cs") is { Delete: true, Bytes.Length: 0 }, "a file as bytes, and a removal");
        var draft = new TaskDraft("t1", "f", "Fix it");
        Check(draft is { Language: null, Workspace.Count: 0, Verify.Count: 0, Solution.Count: 0, Setup.Count: 0, VerifyCommands.Count: 0, Tags.Count: 0, Provenance.Path: null },
            "a draft's defaults are empty");
        Check(new CodeSource("/r") is { Languages.Count: 0, History: null } source && source.ReadOptions == ReadOptions.Default && new HistoryQuery() is { From: "HEAD", MaxCount: null, Merges: false },
            "a source's and a query's defaults");
    }

    // A family of the test: one draft per file of the root, its solution the file as it is.
    private sealed class TestTaskFamily : ITaskFamily
    {
        public string Name => "code-test-family";

        public string Summary => "drafts one task per file";

        public async IAsyncEnumerable<TaskDraft> DraftAsync(CodeSource source, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            foreach (string path in Directory.EnumerateFiles(source.Root).Order(StringComparer.Ordinal))
            {
                cancellationToken.ThrowIfCancellationRequested();
                string name = Path.GetFileName(path);
                byte[] bytes = await File.ReadAllBytesAsync(path, cancellationToken).ConfigureAwait(false);
                yield return new TaskDraft(name, Name, $"Write {name}.")
                {
                    Workspace = [CodeFile.Removed(name)], Solution = [new CodeFile(name, bytes)], Provenance = new TaskProvenance { Path = name },
                };
            }
        }
    }

    private sealed class TestHistory : IRepositoryHistory
    {
        public string Name => "code-test-history";

        public string Summary => "one commit of the test";

        public bool CanRead(string root) => root == "/code-test-repo";

        public async IAsyncEnumerable<CodeCommit> CommitsAsync(string root, HistoryQuery? query = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await Task.Yield();
            yield return new CodeCommit("c1", "Fix a", DateTimeOffset.UnixEpoch) { Changes = [new CodeChange("a.cs", CodeChangeKind.Modified)] };
        }

        public ValueTask<Stream> OpenDiffAsync(string root, string commit, CancellationToken cancellationToken = default) =>
            ValueTask.FromResult<Stream>(new MemoryStream("diff c1"u8.ToArray()));

        public ValueTask<Stream?> OpenFileAsync(string root, string commit, string path, CancellationToken cancellationToken = default) => ValueTask.FromResult<Stream?>(null);
    }

    private sealed class TestFilter : IRowFilter
    {
        public string Name => "code-test-filter";

        public string Summary => "drops empty text";

        public IEnumerable<JsonObject> Apply(IEnumerable<JsonObject> rows, RowFilterContext context)
        {
            foreach (var row in rows)
            {
                if ((string?)row[context.TextColumn] is { Length: > 0 })
                {
                    yield return row;
                }
                else
                {
                    context.Rejected?.Invoke(row, "empty");
                }
            }
        }
    }
}
