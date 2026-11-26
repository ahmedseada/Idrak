// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Abstraction.Reinforcement;
using Idrak.Abstraction.Testing;
using Idrak.Reinforcement.Plugins;
using Idrak.Reinforcement.Plugins.Coding;
using Idrak.Reinforcement.Plugins.CSharp;
using Idrak.Text.Abstractions;
using Idrak.Text.Plugins.CSharp;

// Plan 27, R4b part 2: Idrak.Reinforcement.Plugins.CSharp, agentic coding on C# projects: the rewards from the build's
// logs and the tests' TRX results, those results and logs read, and the csharp environment's variant (dotnet's settings,
// defaults and protected files).
internal abstract partial class DesktopTests : Tests
{
    internal static readonly (string Name, Action<Device> Run)[] CSharpCodingGroup =
    [
        ("plugins: csharp: the rewards score the share of tests passed and the build less its errors (its options checked) and refuse an answer without the csharp environment's info; the TRX results and the build's logs read; the C# variant's variables, defaults and protected files", d => { if (d == Device.Cpu) CSharpRewardsAndResults(); }),
    ];

    private static EpisodeInfo CSharpInfoOf(int total, int passed, bool built, int errors, int warnings) => new EpisodeInfo()
        .Set(CSharpInfo.TestsTotal, total).Set(CSharpInfo.TestsPassed, passed).Set(CSharpInfo.BuildPassed, built)
        .Set(CSharpInfo.BuildErrors, errors).Set(CSharpInfo.BuildWarnings, warnings);

    private static void CSharpRewardsAndResults()
    {
        // The rewards: their scores from the environment's info.
        RewardInput Input(int total, int passed, bool built, int errors, int warnings) => new("Fix it.", "Done.", null, CSharpInfoOf(total, passed, built, errors, warnings));
        RewardInput[] samples = [Input(10, 7, true, 0, 2), Input(4, 4, true, 0, 0), Input(0, 0, false, 3, 1), Input(5, 0, true, 0, 0)];
        IRewardFunction tests = CSharpRewards.All[0], build = CSharpRewards.All[1];
        float[] Score(IRewardFunction reward, IReadOnlyDictionary<string, string>? options = null) => reward.ScoreAsync(samples, options).AsTask().GetAwaiter().GetResult();
        AssertClose([0.7f, 1f, 0f, 0f], Score(tests), 1e-6f, "csharp-tests: the share passed (seven of ten is 0.7), none run is 0");
        AssertClose([1f, 1f, -0.3f, 1f], Score(build), 1e-6f, "csharp-build: 1 built, less 0.1 an error");
        AssertClose([0.9f, 1f, -0.35f, 1f], Score(build, new Dictionary<string, string> { ["warning"] = "0.05" }), 1e-6f, "csharp-build with a weight per warning");
        AssertClose([1f, 1f, -0.2f, 1f], Score(build, new Dictionary<string, string> { ["floor"] = "-0.2" }), 1e-6f, "csharp-build at least its floor");
        ThrowsNaming<ArgumentException>(() => Score(build, new Dictionary<string, string> { ["weight"] = "1" }), "an option csharp-build does not take", "'weight'", "error, warning, floor");
        ThrowsNaming<ArgumentException>(() => tests.ScoreAsync([new RewardInput("p", "a")]).AsTask().GetAwaiter().GetResult(),
            "an answer without the csharp environment's info", CSharpInfo.TestsTotal.Name, "csharp environment");

        // The machine-readable output: TRX counters summed over the files, MSBuild's error and warning lines counted once.
        string folder = Path.Combine(Path.GetTempPath(), "idrak-csharp-results-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(folder, "results", "more"));
        try
        {
            const string Trx = "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<TestRun xmlns=\"http://microsoft.com/schemas/VisualStudio/TeamTest/2010\"><ResultSummary outcome=\"Failed\">"
                               + "<Counters total=\"{0}\" executed=\"{0}\" passed=\"{1}\" failed=\"1\" notExecuted=\"0\" /></ResultSummary></TestRun>";
            File.WriteAllText(Path.Combine(folder, "results", "a.trx"), string.Format(System.Globalization.CultureInfo.InvariantCulture, Trx, 6, 5));
            File.WriteAllText(Path.Combine(folder, "results", "more", "b.trx"), string.Format(System.Globalization.CultureInfo.InvariantCulture, Trx, 4, 2));
            File.WriteAllText(Path.Combine(folder, "results", "broken.trx"), "<TestRun><Counters total=");
            Check(CSharpResults.Tests(Path.Combine(folder, "results")) == (10, 7), $"TRX: {CSharpResults.Tests(Path.Combine(folder, "results"))}");
            Check(CSharpResults.Tests(Path.Combine(folder, "none")) == (0, 0), "no results: none ran");
            File.WriteAllLines(Path.Combine(folder, "errors.log"),
            [
                "   2:2>/src/Lib/Broken.cs(4,23): error CS0103: The name 'y' does not exist in the current context [/src/Lib/Lib.csproj]",
                "/src/Lib/Broken.cs(4,23): error CS0103: The name 'y' does not exist in the current context [/src/Lib/Lib.csproj]",
                "   2:2>/src/Lib/Broken.cs(5,23): error CS0103: The name 'z' does not exist in the current context [/src/Lib/Lib.csproj]",
                "MSBUILD : error MSB1009: Project file does not exist.",
                "/src/Lib/A.cs(1,1): warning CS0219: an error: in a warning's text",
            ]);
            Check(CSharpResults.Diagnostics(Path.Combine(folder, "errors.log"), "error") == 3 && CSharpResults.Diagnostics(Path.Combine(folder, "errors.log"), "warning") == 1,
                "errors counted once each (a node's prefix aside), a warning that says error is a warning");
            Check(CSharpResults.Diagnostics(Path.Combine(folder, "missing.log"), "error") == 0, "no log: none");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }

        var variant = new CSharpEnvironmentVariant();
        var variables = variant.Variables(new Dictionary<string, string>());
        Check(variables.Count == 1 && variables["MSBUILDDISABLENODEREUSE"] == "1" || Environment.GetEnvironmentVariable(CSharpEnvironmentVariant.PackagesVariable) is not null,
            "by default: build nodes end with their build, the machine's package folder, the shared compiler server");
        Check(variant.Variables(new Dictionary<string, string> { ["node-reuse"] = "true", ["build-server"] = "false" }) is { } other && !other.ContainsKey("MSBUILDDISABLENODEREUSE")
              && other["UseSharedCompilation"] == "false", "node-reuse and build-server as settings");
        ThrowsNaming<ArgumentException>(() => variant.Check(new Dictionary<string, string> { ["node-reuse"] = "sometimes" }), "a bad flag", "node-reuse=sometimes");
        Check(variant.Defaults()["reward"] == CSharpRewards.Tests && variant.Defaults()["languages"] == "csharp"
              && (Environment.GetEnvironmentVariable(CSharpEnvironmentVariant.ProtectVariable) is not null || variant.Defaults()["protect"] == string.Join(",", CSharpEnvironmentVariant.DefaultProtect)),
            "its defaults for the coding options");
        WithEnvironment(new Dictionary<string, string?> { [CSharpEnvironmentVariant.ProtectVariable] = "a/**; b.props" }, () =>
            Check(CSharpEnvironmentVariant.Protect.SequenceEqual(["a/**", "b.props"]), "IDRAK_CSHARP_PROTECT replaces the protected globs"));
    }
}
