// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The app's tests, with the testing kit (Idrak.Abstraction.Testing) on the public API alone, as an app's test project uses
// the NuGet packages. The app no longer overrides the token sampler (see the app's Program.cs): these tests check that
// the library's default does what the app's own sampler once did for it, so the app notices if an update changes that.
// A plain runner like the repository's (no test framework): the kit returns reports, which any framework can assert on.
//
//   dotnet run -c Release --project samples/Idrak.Samples.Override.Tests

using Idrak.Abstraction.Testing;

// The kit checks a sampler kind (ITokenSamplerKind) against the library's default: the one the app's generations make now.
var current = TokenSamplers.Get(TokenSamplers.DefaultName);
var samplers = new TokenSamplerSuite();

(string Name, Action Run)[] tests =
[
    ("the app overrides nothing: the sampler is the library's default, version 2 or later", () =>
    {
        Check(Overrides.Report().Count == 0, $"overrides: {string.Join(", ", Overrides.Report())}");
        Check(TokenSamplers.Origin(TokenSamplers.DefaultName) == Overrides.Library && TokenSamplers.DefaultVersion(TokenSamplers.DefaultName) >= 2,
            $"version {TokenSamplers.DefaultVersion(TokenSamplers.DefaultName)}");
    }),
    ("the default keeps the token-sampler contract, logits that are not finite included (Conformance.Check)", () =>
    {
        var report = Conformance.Check(current, samplers);
        report.ThrowIfFailed();
        Check(report.Entries.Any(e => e.Name == "greedy takes the best finite token" && e.Status == CheckStatus.Passed), "the non-finite cases ran");
    }),
    ("a stress run: four threads, large vocabularies, no memory left behind (Stress.Run)", () =>
        Stress.Run(current, samplers, new StressOptions { Iterations = 80, Threads = 4 }).ThrowIfFailed()),
    ("the way back: version 1 of the default is still reachable, and still refuses non-finite logits", () =>
    {
        var version1 = TokenSamplers.Default(TokenSamplers.DefaultName, 1);
        Check(version1 is not null, "no version 1");
        Check(!Conformance.Check(version1!, samplers, new ContractCheckOptions { RandomCases = 0 }).Passed, "version 1 passes the non-finite cases");
    }),
];

int failed = 0;
foreach (var (name, run) in tests)
{
    try
    {
        run();
        Console.WriteLine($"  PASS {name}");
    }
    catch (Exception ex)
    {
        failed++;
        Console.WriteLine($"  FAIL {name}: {ex.Message}");
    }
}

Console.WriteLine($"{tests.Length - failed} passed, {failed} failed");
return failed == 0 ? 0 : 1;

static void Check(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}
