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
| Token samplers (`TokenSamplerSuite`, a sampler kind) | The asked shape; ids in range; greedy tokens identical to the reference's (the library's `TokenSampler`), penalties included; sampled tokens among those top-k, top-p and min-p allow; a reset reproducing the same tokens; statistics matching the tokens; with `Exact`, the reference's very tokens and probabilities; logits that are not finite (NaN and -∞ never sampled, greedy takes the best finite token, +∞ wins and shares the probability evenly, a row with nothing finite is reported by `Sample` or `Read`), wide rows included. Pass a device to run it there. |
| Tokenizers (`TokenizerSuite`) | Ids in range; the same ids every time; decoding a span as a list; the round trip (exact, or stable); `TokenOf` past the end; with a reference, the same ids and text. Texts: empty, whitespace, accents, Arabic, Chinese, emoji, combining marks, control characters, long runs, random Unicode, and yours (`ExtraTexts`). |
| Plug-in operations (`Check(PluginOperation<TKernel>, Device, run)`) | The kernel the operation runs on your device (`KernelFor`: the one registered for its kind, or the default) against the operation's default kernel on the CPU, on cases your `run(backend, kernel, seed)` makes from a seed (inputs uploaded, the kernel called, outputs read back); the entry names the kernel that ran. |
| RoPE scalings (`RopeScalingSuite`) | As many frequencies, finite and positive; the library method's frequencies and attention factor (`RopeScalings.Default`); the same per position; safe from several threads. |
| Chat part kinds (`ChatPartKindSuite`, `Conformance.CheckChatPartKind`) | Reads parts of its kind; the JSON it writes keeps the type and reads back to an equal part, the same every time; the library kind's parts and JSON (`ChatParts.Default`). Text: empty, Arabic, quotes, control characters, emoji; images: PNG, JPEG, unknown types, random bytes; yours (`samples`). |
| Fine-tuning data formats (`TuningDataFormatSuite`, `Conformance.CheckTuningDataFormat`) | On data files of the format's own (`TuningDataSample`): valid files read without failing, every conversation with messages of known roles and image parts with their bytes; the same conversations on a second reading, from two readings at once, and when a reading stops early; the expected conversations when given; invalid files refused with an error naming the file and saying what is wrong. |
| Image model families (`ImageModelFamilySuite`, `Conformance.CheckImageModelFamily`) | On checkpoints of the family's own (`ImageFamilySample`): each loads, with a known task and labels that are not empty (a classifier one output per label); every image's outputs finite, the same on a second prediction, from two predictions at once, alone as in the batch, and after loading again; the reference's outputs when given; random images of many sizes and channel counts mixed in; invalid checkpoints refused saying why. |
| Feature caches (`FeatureCacheSuite`, `Conformance.CheckFeatureCache`) | Misses when empty and after a clear; a small value kept, given back exactly (values and shape) on the device asked for, as a new tensor each time, from the cache's own copy; a key differing in any field (image, transforms, vision options, family, checkpoint, precision, stage) misses; a second put replaces the first; puts and gets from several threads, every hit exactly its key's value. |
| Environment kinds (`EnvironmentSuite<TObservation, TAction>`, `Conformance.CheckEnvironmentKind`) | On episodes played with actions the test chooses: the kind's types, its options refused unless in `Keys`; no step before a reset or after an episode ended; observations in their space, chosen actions in theirs, finite rewards; a seeded replay giving the same steps in the environment reset again, in a new one and in two run at once; a cancelled step; `Dispose` twice. |
| Reward functions (`RewardFunctionSuite`, `Conformance.CheckRewardFunction`) | On answers of the reward's own (`RewardInput`): a finite score per input and none for none; a batch scoring what its inputs score one by one; the same scores again and from two calls at once; options refused unless in `Keys`; a cancelled call; the library reward's scores (`RewardFunctions.Default`). |
| Advantage estimators (`AdvantageEstimatorSuite`, `Conformance.CheckAdvantageEstimator`) | Episodes in groups, with and without values and truncated episodes: every step's advantage and return written and finite; the same output bit for bit again and from several threads (a fixed summation order); wrong lengths, a missing critic and options refused; the library estimator's output (`AdvantageEstimators.Default`). |
| Policy losses (`PolicyLossSuite`, `Conformance.CheckPolicyLoss`) | On- and off-policy mini-batches, with and without a reference and a value head: a finite one-element loss, the same every time; a gradient that reaches the log-probabilities and agrees with central differences where the loss is smooth (a clip's kinks left out); options refused; the library loss's value (`PolicyLosses.Default`). |

A suite whose contract is in Idrak.Abstraction takes the contract itself and calls it as the library does, so it
measures the implementation's real work and allocations. The token samplers are checked by their kind
(`ITokenSamplerKind`, as registered in `TokenSamplers`), against the library's default unless another reference is given:

```csharp
Conformance.Check(new MySamplerKind(), new TokenSamplerSuite()).ThrowIfFailed();
Conformance.Check(new MySamplerKind(), new TokenSamplerSuite(device: Device.Get("cuda")) { Exact = true }).ThrowIfFailed();
```

The feature caches (`IFeatureCache`, in Idrak.Abstraction with Idrak.Text's built-ins) and the progress reporters
(`IProgressReporterKind`) are checked the same way. A contract that lives in another package (the fine-tuning data
formats, `ITuningDataFormat`, in Idrak.Text) is given as its methods, since this kit depends on Idrak.Abstraction alone:
a format as its reading.

```csharp
using var cache = FeatureCaches.Create("disk", new FeatureCacheOptions { Folder = "cache" });
Conformance.CheckFeatureCache(cache).ThrowIfFailed();
Conformance.CheckProgressReporter("console").ThrowIfFailed();

var format = TuningDataFormats.Get("sharegpt");
Conformance.CheckTuningDataFormat(path => format.Read(path, new TuningDataOptions { Images = "images.zip" }).Select(t => t.Messages),
    [new TuningDataSample("train.json"), new TuningDataSample("bad.json", Error: "markers")]).ThrowIfFailed();
```

An annotation format (Idrak.Data's `IAnnotationFormat`) is given as its reading and writing, with what it declares it
keeps (`Keeps`, an `AnnotationFormatTraits`), so the suite checks those parts and no others:

```csharp
var coco = AnnotationFormats.Get("coco");
Conformance.Check(new AnnotationFormatUnderTest(path => coco.Read(path), coco.Write, coco.Keeps), new AnnotationFormatSuite()).ThrowIfFailed();
```

An image model family (Idrak.Vision's `IImageModelFamily`, registered in `ImageModelFamilies`) is checked as its loading: a
checkpoint's path in, the loaded model wrapped in `ImageModelUnderTest` (its task, labels and each image's outputs) out.
The suite (`ImageModelFamilySuite`) checks that it loads, predicts finite outputs, the same every time, from two threads,
alone and in a batch and after loading again, the reference's outputs when given, and that invalid checkpoints are
refused saying why:

```csharp
Conformance.CheckImageModelFamily(path =>
{
    var model = ImageModels.Load(path);
    var outputs = model.Outputs().Build();
    return new ImageModelUnderTest("classification", model.Labels, outputs.Predict, model);
}, [new ImageFamilySample("models/tiny-resnet", images, expectedLogits), new ImageFamilySample("models/broken", Error: "conv1.weight")]).ThrowIfFailed();
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
Regression.Replay("regressions", new MySamplerKind(), new TokenSamplerSuite()).ThrowIfFailed();
```

A saved operation call holds the operation, its arguments and the values of its storages before and after (exact
bits); a saved contract case holds the case's data. The case that made an app write its override stays its test, and
the library's once the fix moves there.

`samples/Idrak.Samples.Override` (an app with its own token sampler) and its tests
(`samples/Idrak.Samples.Override.Tests`) show the whole loop; `tests/Idrak.PluginTests` has a plain-loop device written
outside the library that passes the device check.

## Status

Preview, while the version is 0.y.z. Plan: [plans/done/10-abstraction.md](https://github.com/ahmedseada/Idrak/blob/main/plans/10-abstraction.md)
("The testing kit as the app's stress harness").

Licensed under the Apache License 2.0.
