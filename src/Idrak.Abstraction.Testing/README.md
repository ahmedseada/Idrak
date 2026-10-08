# Idrak.Abstraction.Testing

The conformance and stress kit for the contracts of [Idrak](https://www.nuget.org/packages/Idrak): it checks your
implementation of a contract (a device, a token sampler, a tokenizer, a RoPE scaling, or a contract of your own) against
the library default, stresses it, and keeps the cases that once failed as regression tests. Built on
`Idrak.Abstraction` alone, so an outside device or an app's override is checked without the rest of the library.

```bash
dotnet add package Idrak.Abstraction.Testing   # in your test project
```

Nothing here asserts or depends on a test framework: every method returns a report (`Passed`, `Failures`, `Entries`,
a readable `ToString()`, `ThrowIfFailed()`), so xunit, NUnit, MSTest or a plain console runner can use it.

## Conformance

```csharp
using Idrak.Abstraction.Testing;

Conformance.Check(Device.Get("mydevice")).ThrowIfFailed();          // a device, operation by operation
Conformance.Check(myTokenizer, reference: libraryTokenizer).ThrowIfFailed();
Conformance.CheckRopeScaling("yarn", MyYarn).ThrowIfFailed();        // against the library's own "yarn"
```

| Check | What it compares |
|---|---|
| Devices (`Check(Device)`, `Check(Backend)`) | The built-in cases (`DeviceCases.All`: element-wise, products, norms, rotary positions, convolution, layout, optimizer steps, packed weights and their fused products, gated activations, key/value caches, every attention kernel, sampling, autograd) run on the CPU with every operation call recorded; each call then runs on your device through its dispatcher (`Backend.Name(...)`: a registered kernel, the device's own, a composition, or the host fallback) on copies of the same values, and everything it writes must agree with the CPU within the operation's tolerance. Operations the CPU has no single kernel for (the fused products) are compared with their composition, and those it has none for at all (FP8 weights and products, attention read in place from strided rows and its gradients) with plain-loop references written from their contracts. The report has one entry per operation with the kernel that ran (`Kernels.Chain`); operations with no kernel on the device, or that no case reaches, are listed as skipped. |
| Token samplers (`TokenSamplerSuite`) | The asked shape; ids in range; greedy tokens identical to the reference's (the library's `TokenSampler`), penalties included; sampled tokens among those top-k, top-p and min-p allow; a reset reproducing the same tokens; statistics matching the tokens; with `Exact`, the reference's very tokens and probabilities; logits that are not finite (NaN and -∞ never sampled, greedy takes the best finite token, +∞ wins and shares the probability evenly, a row with nothing finite is reported by `Sample` or `Read`), wide rows included. Pass a device to run it there. |
| Tokenizers (`TokenizerSuite`) | Ids in range; the same ids every time; decoding a span as a list; the round trip (exact, or stable); `TokenOf` past the end; with a reference, the same ids and text. Texts: empty, whitespace, accents, Arabic, Chinese, emoji, combining marks, control characters, long runs, random Unicode, and yours (`ExtraTexts`). |
| Plug-in operations (`Check(PluginOperation<TKernel>, Device, run)`) | The kernel the operation runs on your device (`KernelFor`: the one registered for its kind, or the default) against the operation's default kernel on the CPU, on cases your `run(backend, kernel, seed)` makes from a seed (inputs uploaded, the kernel called, outputs read back); the entry names the kernel that ran. |
| RoPE scalings (`RopeScalingSuite`) | As many frequencies, finite and positive; the library method's frequencies and attention factor (`RopeScalings.Default`); the same per position; safe from several threads. |

The token-sampler contract (`ITokenSampler`) lives in Idrak.Nlp, and this kit depends on Idrak.Abstraction alone, so
a test passes the sampler it checks and the reference as delegates (`SamplerUnderTest`: set the history, sample a step,
reset, read the statistics):

```csharp
static Func<SamplerSettings, SamplerUnderTest> Drive(Func<SamplerRequest, ITokenSampler> create) => settings =>
{
    var sampler = create(new SamplerRequest(settings.Device, settings.Rows, settings.Vocabulary, settings.Steps, settings.HistoryCapacity,
        new GenerationOptions { Temperature = settings.Temperature, TopK = settings.TopK, /* ... */ Seed = settings.Seed }));
    return new SamplerUnderTest(sampler.Rows, sampler.Vocabulary, sampler.SetHistory, logits => { sampler.Sample(logits); return sampler.Ids.ToArray(); },
        sampler.Reset, (from, to) => [.. sampler.Read(from, to).Select(s => s.Select(t => new SampledStatistics(t.Id, t.Probability, t.Entropy)).ToArray())], sampler);
};

Conformance.Check(Drive(MySampler.Create), new TokenSamplerSuite(Drive(TokenSampler.Create))).ThrowIfFailed();
```

Fixed cases come first, then random ones (`DeviceCheckOptions.RandomRuns`, `ContractCheckOptions.RandomCases`, a seed).
Tolerances are per operation (`DeviceCheckOptions.Tolerances`); the comparisons themselves are public (`Comparisons`, in Idrak.Abstraction, which the override loop's shadow mode uses too).

### A contract of your own

Subclass `ContractSuite<T>`: fixed cases and random ones as data (`ContractCase`, a name and a JSON object), and how to
run one, recording each property in `CaseChecks`. `Conformance.Check`, `Stress.Run` and `Regression.Replay` then work
with it:

```csharp
sealed class MyParserSuite : ContractSuite<IToolCallParser> { ... }
Conformance.Check(myParser, new MyParserSuite()).ThrowIfFailed();
```

A device check can take cases of yours too (`DeviceCheckOptions.Cases = [.. DeviceCases.All, mine]`): a `DeviceCase`
is a small program on the CPU, and `DeviceCaseContext.Composed` gives an operation the CPU has no kernel for a reference
made of other operations.

## Stress

```csharp
Stress.Run(device, new StressOptions { Duration = TimeSpan.FromMinutes(5), Threads = 4 }).ThrowIfFailed();
Stress.Run(myTokenizer, new TokenizerSuite(), new StressOptions { Iterations = 10_000 }).ThrowIfFailed();
```

Generated inputs (odd and large shapes, long texts, wide vocabularies) from several threads at once, for a time or a
number of iterations, compared as the conformance check compares; the memory in use (`MemoryAccountant`, through
`Backend.GetMemoryUsage`) must come back to where it was, also when the run is cancelled (`CancellationToken`,
`StressOptions.CancelAfter`). The report lists the failures and the slowest cases.

## Regression cases

```csharp
var report = Conformance.Check(device);
report.SaveFailures("regressions");                         // one small JSON file per failing case
Regression.Replay("regressions", device).ThrowIfFailed();   // in your tests from then on
Regression.Replay("regressions", Drive(MySampler.Create), new TokenSamplerSuite(Drive(TokenSampler.Create))).ThrowIfFailed();
```

A saved operation call holds the operation, its arguments and the values of its storages before and after (exact
bits); a saved contract case holds the case's data. The case that made an app write its override stays its test, and
the library's once the fix moves there.

`samples/Idrak.Samples.Override` (an app with its own token sampler) and its tests
(`samples/Idrak.Samples.Override.Tests`) show the whole loop; `tests/Idrak.PluginTests` has a plain-loop device written
outside the library that passes the device check.

## Status

Preview, while the version is 0.y.z. Plan: [plans/10-abstraction.md](https://github.com/ahmedseada/Idrak/blob/main/plans/10-abstraction.md)
("The testing kit as the app's stress harness").

Licensed under the Apache License 2.0.
