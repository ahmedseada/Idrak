// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using Idrak.Data;
using Idrak.Data.Abstractions;

// Plan 36, G0: the code dataset tool's CodeExtractorOptions with its IDRAK_CODE_* variables.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] DataCodeContractsGroup =
    [
        ("data code contracts: CodeExtractorOptions' defaults are the documented ones, each IDRAK_CODE_* variable sets its setting (0 or all for no limit), a bad value is refused naming the variable, and the records hold their values (a file as bytes, a removal, a draft's defaults)", d => { if (d == Device.Cpu) CodeExtractorOptionsRead(); }),
    ];

    private static void CodeExtractorOptionsRead()
    {
        string[] variables =
        [
            CodeExtractorOptions.FamiliesVariable, CodeExtractorOptions.SkippedFoldersVariable, CodeExtractorOptions.MaxFileBytesVariable, CodeExtractorOptions.DedupeVariable,
            CodeExtractorOptions.SeedVariable, CodeExtractorOptions.SplitByVariable, CodeExtractorOptions.EvalFractionVariable, CodeExtractorOptions.EvalGroupsVariable,
            CodeExtractorOptions.MaxTasksVariable, CodeExtractorOptions.MaxTimeVariable, CodeExtractorOptions.FormatsVariable, CodeExtractorOptions.ConcurrencyVariable,
            CodeExtractorOptions.VerifierVariable,
        ];
        Check(variables.All(v => v.StartsWith("IDRAK_CODE_", StringComparison.Ordinal)) && variables.Distinct().Count() == 13, "thirteen IDRAK_CODE_* variables");

        var unset = variables.ToDictionary(v => v, _ => (string?)null);
        WithEnvironment(unset, () =>
        {
            var defaults = CodeExtractorOptions.FromEnvironment();
            Check(defaults == new CodeExtractorOptions() || defaults is { Families: null, SkippedFolders: null, Seed: 0, EvalGroups.Count: 0, MaxTasks: null, MaxTime: null, Concurrency: 0, Verifier: null },
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
            [CodeExtractorOptions.VerifierVariable] = "none",
        };
        WithEnvironment(set, () =>
        {
            var o = CodeExtractorOptions.FromEnvironment();
            Check(o.Families!.SequenceEqual(["a", "b"]) && o.SkippedFolders!.SequenceEqual(["bin", "obj"]) && o.MaxFileBytes == 4096 && o.DedupeThreshold == 0.9 && o.Seed == 7
                  && o.SplitBy == "test-group" && o.EvalFraction == 0.25 && o.EvalGroups.SequenceEqual(["g1", "g2"]) && o.MaxTasks == 50 && o.MaxTime == TimeSpan.FromSeconds(90)
                  && o.Formats.SequenceEqual(["JsonLines", "Parquet"]) && o.Concurrency == 3 && o.EffectiveConcurrency == 3 && o.Verifier == "none", $"each variable sets its setting: {o}");
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
}
