// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The app's own tests of its override, with the testing kit (Idrak.Abstraction.Testing) on the public API alone, as an
// app's test project uses the NuGet packages. A plain runner like the repository's (no test framework): the kit
// returns reports, which any framework can assert on as well.
//
//   dotnet run -c Release --project samples/Idrak.Samples.Override.Tests

using System.Runtime.InteropServices;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Generation;
using Idrak.Generation.Abstractions;
using Idrak.Samples.Override;

// The token-sampler contract is Idrak.Nlp's; the kit drives a sampler through a few delegates (Drive, below), and
// compares it with the library's TokenSampler.
var app = Drive(SanitizingSampler.Create);
var library = Drive(TokenSampler.Create);
var samplers = new TokenSamplerSuite(library);
string regressions = Path.Combine(AppContext.BaseDirectory, "regressions");

(string Name, Action Run)[] tests =
[
    ("the app's sampler keeps the token-sampler contract (Conformance.Check)", () => Conformance.Check(app, samplers).ThrowIfFailed()),
    ("it samples the library's very tokens and probabilities (it wraps the library's sampler)", () =>
        Conformance.Check(app, new TokenSamplerSuite(library) { Exact = true }, new ContractCheckOptions { RandomCases = 40 }).ThrowIfFailed()),
    ("a stress run: four threads, large vocabularies, no memory left behind (Stress.Run)", () =>
        Stress.Run(app, samplers, new StressOptions { Iterations = 80, Threads = 4 }).ThrowIfFailed()),
    ("the case that made the app write it: non-finite logits break the library's sampler, not the app's", () =>
    {
        var suite = new NonFiniteLogitsSuite();
        var failing = Conformance.Check(TokenSampler.Create, suite);
        Check(!failing.Passed, $"the library's sampler should fail the app's case:\n{failing}");
        Conformance.Check(SanitizingSampler.Create, suite).ThrowIfFailed();

        // The failing cases, saved, become the app's regression tests.
        string folder = Path.Combine(Path.GetTempPath(), $"override-cases-{Guid.NewGuid():N}");
        try
        {
            Check(failing.SaveFailures(folder).Count > 0, "failures saved");
            Check(!Regression.Replay(folder, TokenSampler.Create, suite).Passed, "the saved cases fail on the library's sampler");
            Regression.Replay(folder, SanitizingSampler.Create, suite).ThrowIfFailed();
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }),
    ("the cases saved in regressions/ (what once broke the app) still pass", () =>
    {
        var replay = Regression.Replay(regressions, SanitizingSampler.Create, new NonFiniteLogitsSuite());
        replay.ThrowIfFailed();
        Check(replay.Cases > 0, "no saved case was found");
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

// An ITokenSampler factory as the kit's sampler checks drive it.
static Func<SamplerSettings, SamplerUnderTest> Drive(Func<SamplerRequest, ITokenSampler> create) => settings =>
{
    var sampler = create(new SamplerRequest(settings.Device, settings.Rows, settings.Vocabulary, settings.Steps, settings.HistoryCapacity, new GenerationOptions
    {
        Temperature = settings.Temperature, TopK = settings.TopK, TopP = settings.TopP, MinP = settings.MinP, RepeatPenalty = settings.RepeatPenalty,
        RepeatLastN = settings.RepeatLastN, PresencePenalty = settings.PresencePenalty, FrequencyPenalty = settings.FrequencyPenalty, Seed = settings.Seed,
    }));
    return new SamplerUnderTest(sampler.Rows, sampler.Vocabulary, sampler.SetHistory, logits =>
    {
        sampler.Sample(logits);
        return sampler.Ids.ToArray();
    }, sampler.Reset, (from, to) => [.. sampler.Read(from, to).Select(step => step.Select(t => new SampledStatistics(t.Id, t.Probability, t.Entropy)).ToArray())], sampler);
};

static void Check(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

/// <summary>
/// The app's own contract case, added to the kit's suites the way any contract is (a <see cref="ContractSuite{T}"/>):
/// greedy sampling over logits some of which are NaN or infinite must choose the best finite token (or a +∞ one).
/// </summary>
internal sealed class NonFiniteLogitsSuite : ContractSuite<Func<SamplerRequest, ITokenSampler>>
{
    public override string Name => "non-finite-logits";

    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return Case("one NaN", [0.5f, float.NaN, 1f, 4f, 2f]);
        yield return Case("NaN first", [float.NaN, 1f, 3f, 2f]);
        yield return Case("-∞ and NaN", [float.NegativeInfinity, 2f, float.NaN, 1f]);
        yield return Case("+∞", [1f, float.PositiveInfinity, 3f]);
    }

    public override ContractCase RandomCase(Random random, bool large)
    {
        var logits = Enumerable.Range(0, random.Next(2, large ? 5000 : 60)).Select(_ => random.NextSingle() * 10f - 5f).ToArray();
        for (int i = 0, bad = random.Next(1, Math.Max(2, logits.Length / 4)); i < bad; i++)
        {
            logits[random.Next(logits.Length)] = random.Next(3) == 0 ? float.NegativeInfinity : float.NaN;
        }

        logits[random.Next(logits.Length)] = 6f;                                    // a finite best token
        return Case($"random ({logits.Length} tokens)", logits);
    }

    public override void Run(Func<SamplerRequest, ITokenSampler> implementation, ContractCase @case, CaseChecks checks)
    {
        var logits = MemoryMarshal.Cast<byte, float>(Convert.FromBase64String((string)@case.Data["logits"]!)).ToArray();
        int best = Array.IndexOf(logits, float.PositiveInfinity) is >= 0 and var infinite ? infinite
            : Enumerable.Range(0, logits.Length).Where(i => float.IsFinite(logits[i])).MaxBy(i => logits[i]);
        var request = new SamplerRequest(Device.Cpu, 1, logits.Length, 1, 1, new GenerationOptions { TopK = 1, Seed = 1 });
        using var sampler = implementation(request);
        using var step = Tensor.From(logits, [1, logits.Length], Device.Cpu);
        sampler.Sample(step);
        int id = (int)sampler.Ids.ToArray()[0];
        checks.Check("greedy takes the best finite token", id == best, $"token {id} ({logits.ElementAtOrDefault(id)}), the best is {best} ({logits[best]})");
    }

    private static ContractCase Case(string name, float[] logits) =>
        new(name, new JsonObject { ["logits"] = Convert.ToBase64String(MemoryMarshal.AsBytes(logits.AsSpan())) });
}
