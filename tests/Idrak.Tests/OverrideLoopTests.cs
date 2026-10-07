// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Diagnostics;
using Idrak.Abstraction.Generation;
using Idrak.Data.Abstractions;
using Idrak.Generation;
using Idrak.Generation.Abstractions;
using Idrak.Layers.Abstractions;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Onnx.Abstractions;

// Plan 10, phase 6b (the override loop): an app overrides a sampler, a tokenizer part and a RoPE scaling. Each keeps the
// library default behind it: a failure falls back to it (FallBack, the default) and is reported to telemetry, Shadow
// lets the default answer and compares, Throw lets the failure through, and Unregister brings the default back.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] OverrideLoopGroup =
    [
        ("override loop: the library's built-ins are the slots' defaults in every registry, not overrides", OverrideLoopLibraryDefaults),
        ("override loop: under the default policy (Throw) an app's sampler, tokenizer part and RoPE scaling that throw reach the caller, reported with a hint naming FallBack, Shadow and IDRAK_OVERRIDE_POLICY", OverrideLoopDefaultThrows),
        ("override loop: under FallBack (opted into per slot) an app's sampler, tokenizer part and RoPE scaling each throw once and fall back to the default, reported to telemetry", OverrideLoopFallBack),
        ("override loop: under Shadow the default answers and the app's sampler, tokenizer part and RoPE scaling are compared (outputs, times, allocations)", OverrideLoopShadow),
        ("override loop: under an explicit Throw the app's failure reaches the caller; Unregister restores the default; the report lists overrides with origin and policy", OverrideLoopThrowAndReport),
        ("override loop: IDRAK_OVERRIDE_POLICY gives slots their first policy: one for every slot, or Registry/name=policy entries", OverrideLoopPolicyVariable),
        ("override loop: a slot table keeps the default under an app's entry, keeps order, hands the default out as it is, and cannot set a policy without a guard", OverrideLoopSlotTable),
        ("override loop: library defaults have versions; older ones stay reachable; the report says when an override was built against a release older than the default it shadows", OverrideLoopVersions),
    ];

    private static void OverrideLoopVersions(Device device)
    {
        _ = device;
        Check(TokenSamplers.DefaultVersion(TokenSamplers.DefaultName) == 2 && TokenSamplers.Default(TokenSamplers.DefaultName, 1) is not null
              && TokenSamplers.Default(TokenSamplers.DefaultName, 3) is null
              && TokenSamplers.Default(TokenSamplers.DefaultName, 2) == TokenSamplers.Default(TokenSamplers.DefaultName), "the sampler default is version 2; version 1 is kept");
        Check(RopeScalings.Default(LinearType) is not null && Overrides.Report().All(o => o.Registry != "RopeScalings" || o.DefaultVersion >= 1), "other defaults are version 1");

        // A table whose default (from Idrak.Nlp, which this assembly references) moved to version 2 in a later release.
        Func<SamplerRequest, ITokenSampler> first = TokenSamplers.Default(TokenSamplers.DefaultName, 1)!, second = TokenSampler.Create,
            mine = request => TokenSampler.Create(request);
        string built = typeof(Tests).Assembly.GetReferencedAssemblies().Single(a => a.Name == "Idrak.Nlp").Version!.ToString(3);
        foreach (var (since, outdated) in new[] { ("99.0.0", true), (built, false) })
        {
            var table = new SlotTable<string, Func<SamplerRequest, ITokenSampler>>("TestVersions");
            table.RegisterDefault("s", second, version: 2, since: since);
            table.RegisterDefault("s", first);                                     // registered in any order: the highest version is the default
            Check(ReferenceEquals(table.Default("s"), second) && ReferenceEquals(table.Default("s", 1), first) && table.Default("s", 7) is null
                  && table.DefaultVersions("s").SequenceEqual([1, 2]) && table.DefaultVersion("s") == 2, "versions are kept, the highest answers");
            table.Register("s", mine);
            var row = table.Report().Single();
            Check(row.DefaultVersion == 2 && row.DefaultSince == since && row.BuiltAgainst == built && row.Outdated == outdated
                  && (row.OutdatedNote?.Contains($"built against {built}", StringComparison.Ordinal) ?? !outdated)
                  && row.ToString().Contains("version 2", StringComparison.Ordinal) == true, $"since {since}: {row}");
            Check(table.Unregister("s") && ReferenceEquals(table.Find("s"), second) && table.DefaultVersions("s").Count == 2, "unregistering keeps every version");
        }

        try
        {
            new SlotTable<string, int>("TestVersions").RegisterDefault("x", 1, version: 2, since: "next");
            Check(false, "a release that is not a number is refused");
        }
        catch (ArgumentException)
        {
        }
    }

    // The overridden slots, with what the tests registered over them.
    private const string LowercaseType = "Lowercase", LinearType = "linear";

    // Records the override loop's telemetry.
    private sealed class OverrideRecorder : ITelemetryHook
    {
        public List<OverrideFailed> Failed { get; } = [];

        public List<OverrideCompared> Compared { get; } = [];

        public TelemetryLevel Levels => TelemetryLevel.Overrides;

        public void OnOverrideFailed(in OverrideFailed e)
        {
            lock (Failed)
            {
                Failed.Add(e);
            }
        }

        public void OnOverrideCompared(in OverrideCompared e)
        {
            lock (Compared)
            {
                Compared.Add(e);
            }
        }
    }

    // An app's sampler: the library's, choosing from negated logits (so greedy picks the least likely token).
    private sealed class NegatingSampler(ITokenSampler inner) : ITokenSampler
    {
        public int Rows => inner.Rows;

        public int Vocabulary => inner.Vocabulary;

        public Tensor Ids => inner.Ids;

        public bool Recordable => inner.Recordable;

        public void Sample(Tensor logits) => inner.Sample(-logits);

        public void SetHistory(IReadOnlyList<int> tokens) => inner.SetHistory(tokens);

        public void Reset() => inner.Reset();

        public SampledToken[][] Read(int fromStep, int toStep) => inner.Read(fromStep, toStep);

        public void Dispose() => inner.Dispose();
    }

    // Logits whose most likely token is 1 and least likely 3, and a greedy request for one row of them.
    private static (Tensor Logits, SamplerRequest Request) SamplerCase(Device device) =>
        (Tensor.From([0.1f, 2f, 0.3f, -1f, 0.5f, 0f, 0f, 0f], [1, 8], device),
         new SamplerRequest(device, 1, 8, 4, 1, new GenerationOptions { Temperature = 0, RepeatPenalty = 1, Seed = 1 }));

    // The token a sampler made for the request chooses from the logits.
    private static int Choose(Func<SamplerRequest, ITokenSampler> create, Device device)
    {
        var (logits, request) = SamplerCase(device);
        using var sampler = create(request);
        sampler.SetHistory([0]);
        sampler.Sample(logits);
        return (int)sampler.Ids.ToArray()[0];
    }

    private static JsonObject LowercaseTokenizer() => new()
    {
        ["added_tokens"] = new JsonArray(),
        ["normalizer"] = new JsonObject { ["type"] = LowercaseType },
        ["model"] = new JsonObject
        {
            ["type"] = "BPE",
            ["vocab"] = new JsonObject { ["a"] = 0, ["b"] = 1, ["A"] = 2, ["B"] = 3, ["ab"] = 4 },
            ["merges"] = new JsonArray("a b"),
        },
    };

    private static RopeSettings LinearRope() => new(10000f, Scaling: RopeScaling.Linear(2));

    // A normalizer from a function.
    private sealed class FunctionNormalizer(Func<string, string> normalize) : ITokenizerNormalizer
    {
        public string Normalize(string text) => normalize(text);
    }

    // Removes what the tests registered, whatever happened.
    private static void RemoveOverrides()
    {
        TokenSamplers.Unregister(TokenSamplers.DefaultName);
        TokenizerComponents.UnregisterNormalizer(LowercaseType);
        RopeScalings.Unregister(LinearType);
        TokenSamplers.SetPolicy(TokenSamplers.DefaultName, SlotPolicy.Throw);
        TokenizerComponents.SetNormalizerPolicy(LowercaseType, SlotPolicy.Throw);
        RopeScalings.SetPolicy(LinearType, SlotPolicy.Throw);
    }

    private static void OverrideLoopLibraryDefaults(Device device)
    {
        _ = device;
        const string library = Overrides.Library;
        Check(RopeScalings.Origin(LinearType) == library && TokenSamplers.Origin(TokenSamplers.DefaultName) == library
              && TokenizerComponents.NormalizerOrigin(LowercaseType) == library && TokenizerComponents.PreTokenizerOrigin("ByteLevel") == library
              && TokenizerComponents.DecoderOrigin("ByteFallback") == library, "the slots of the acceptance test start at the library's");
        // Built-ins registered on first use, from Abstraction, core, Gpu, Data and Nlp.
        Check(Idrak.Abstraction.Generation.ToolCallFormats.Origin("json") == library && Idrak.Abstraction.Generation.ChatTemplates.Origin("jinja") == library && PackedWeight.Origin("int4") == library
              && KeyValueLayouts.Origin("int8") == library && ModelSources.Origin("huggingface") == library && ModelSources.Origin("folder") == library
              && DeviceProviders.Origin("vulkan") == library && DeviceProviders.Origin("minimal") == "Idrak.Tests", "Abstraction's registries");
        Check(CheckpointFormats.Origin("gguf") == library && GgufTypes.Origin(8) == library && GgufArchitectures.Origin("llama") == library
              && GgufPreTokenizers.Origin("qwen2") == library && PretrainedArchitectures.Origin("LlamaForCausalLM") == library
              && GraphOps.Origin("relu") == library && NetworkOps.Origin("linear") == library && LayerTypes.Origin("linear") == library
              && ImageCodecs.Origin("png") == library && SampleSources.Origin("csv") == library && OnnxImportOps.Origin("Relu") == library
              && OnnxExportOps.GraphOpOrigin("relu") == library, $"core's registries; overridden now: {string.Join(", ", Overrides.Report().Select(o => $"{o.Registry}/{o.Name}"))}");
        Check(Idrak.Data.Abstractions.ParquetCodecs.Origin(1) == library && Idrak.Data.Abstractions.DataFileFormats.Origin("Csv") == library
              && Idrak.Data.Abstractions.DatasetSources.Origin("hf") == library, "Data's registries");
        var origins = Overrides.Report().Select(o => o.Origin).ToHashSet();
        Check(!origins.Overlaps(["Idrak", "Idrak.Abstraction", "Idrak.Gpu", "Idrak.Data", "Idrak.Nlp", "Idrak.Vision"]),
            $"no library assembly is reported as overriding: {string.Join(", ", origins)}");
        Check(RopeScalings.Get(LinearType) == RopeScalings.Default(LinearType), "with only its default, a slot hands it out as it is");
    }

    private static void OverrideLoopDefaultThrows(Device device)
    {
        var recorder = new OverrideRecorder();
        using var subscription = Telemetry.Subscribe(recorder);
        try
        {
            Check(TokenSamplers.Default(TokenSamplers.DefaultName) is not null && RopeScalings.Default(LinearType) is not null
                  && Overrides.Report().All(o => o.Policy == SlotPolicy.Throw || o.Registry is not ("RopeScalings" or "TokenSamplers")), "Throw is the default policy");
            TokenSamplers.Register(TokenSamplers.DefaultName, _ => throw new InvalidOperationException("the app's sampler broke"));
            TokenizerComponents.RegisterNormalizer(LowercaseType, _ => new FunctionNormalizer(_ => throw new FormatException("the app's normalizer broke")));
            RopeScalings.Register(LinearType, _ => throw new ArithmeticException("the app's scaling broke"));
            Check(Throws<InvalidOperationException>(() => Choose(TokenSamplers.Create, device)), "the sampler's failure reaches the caller");
            var tokenizer = BpeTokenizer.FromJson(LowercaseTokenizer());
            Check(Throws<FormatException>(() => tokenizer.Encode("AB")), "the normalizer's failure reaches the caller");
            Check(Throws<ArithmeticException>(() => LinearRope().Frequencies(8)), "the scaling's failure reaches the caller");

            var failed = recorder.Failed.Select(e => (e.Registry, e.Slot, e.Exception.GetType(), e.FellBack)).ToList();
            Check(failed.SequenceEqual([
                ("TokenSamplers", "default", typeof(InvalidOperationException), false),
                ("TokenizerComponents.Normalizers", LowercaseType, typeof(FormatException), false),
                ("RopeScalings", LinearType, typeof(ArithmeticException), false)]), $"one event per failure: {string.Join("; ", failed)}");
            var hints = recorder.Failed.Select(e => e.Hint ?? "").ToList();
            Check(hints[0].Contains("TokenSamplers.SetPolicy(\"default\", SlotPolicy.FallBack)", StringComparison.Ordinal)
                  && hints[1].Contains("TokenizerComponents.SetNormalizerPolicy(\"Lowercase\", SlotPolicy.FallBack)", StringComparison.Ordinal)
                  && hints[2].Contains("RopeScalings.SetPolicy(\"linear\", SlotPolicy.Shadow, shadowRate: 0.01)", StringComparison.Ordinal)
                  && hints[2].Contains("IDRAK_OVERRIDE_POLICY=RopeScalings/linear=fallback", StringComparison.Ordinal),
                $"each event carries the hint: {hints[2]}");
            var printed = new StringWriter();
            using (Telemetry.Subscribe(new ConsoleLogger(TelemetryLevel.Training, output: printed)))
            {
                _ = Throws<ArithmeticException>(() => LinearRope().Frequencies(8));
            }

            Check(printed.ToString().Contains("the error reached the caller", StringComparison.Ordinal)
                  && printed.ToString().Contains("IDRAK_OVERRIDE_POLICY", StringComparison.Ordinal), $"a console logger prints it with the hint: {printed}");
            Check(Overrides.Report().Single(o => o.Registry == "RopeScalings" && o.Name == LinearType) is { Failures: 2, FallBacks: 0, PolicyText: "throw" },
                "the report counts the failures");
        }
        finally
        {
            RemoveOverrides();
        }
    }

    private static void OverrideLoopFallBack(Device device)
    {
        var recorder = new OverrideRecorder();
        using var subscription = Telemetry.Subscribe(recorder);
        try
        {
            TokenSamplers.SetPolicy(TokenSamplers.DefaultName, SlotPolicy.FallBack);
            TokenizerComponents.SetNormalizerPolicy(LowercaseType, SlotPolicy.FallBack);
            RopeScalings.SetPolicy(LinearType, SlotPolicy.FallBack);
            // The sampler: made by the app, which breaks the first time; the library's is kept to wrap.
            var builtIn = TokenSamplers.Default(TokenSamplers.DefaultName)!;
            int failures = 1;
            TokenSamplers.Register(TokenSamplers.DefaultName, request =>
                failures-- > 0 ? throw new InvalidOperationException("the app's sampler broke") : new NegatingSampler(builtIn(request)));
            Check(TokenSamplers.Origin(TokenSamplers.DefaultName) == "Idrak.Tests" && TokenSamplers.Get(TokenSamplers.DefaultName) != builtIn,
                "the app's sampler is registered over the default");
            Check(Choose(TokenSamplers.Create, device) == 1, "the first sampler fell back to the library's (greedy: token 1)");
            Check(Choose(TokenSamplers.Create, device) == 3, "the next one is the app's (token 3)");

            // The tokenizer part: its making fails once, then a call fails once; the built-in answers both times.
            int making = 1, calls = 0;
            TokenizerComponents.RegisterNormalizer(LowercaseType, _ =>
            {
                if (making-- > 0)
                {
                    throw new InvalidDataException("the app's normalizer cannot be made");
                }

                return new FunctionNormalizer(s => calls++ == 0 ? throw new FormatException("the app's normalizer broke") : s.ToLowerInvariant());
            });
            Check(BpeTokenizer.FromJson(LowercaseTokenizer()).Encode("AB").SequenceEqual([4]), "made with the built-in normalizer");
            var tokenizer = BpeTokenizer.FromJson(LowercaseTokenizer());
            Check(tokenizer.Encode("AB").SequenceEqual([4]) && tokenizer.Encode("AB").SequenceEqual([4]) && calls == 2,
                "a call that throws is answered by the built-in, the next by the app's");

            // The RoPE scaling: the first call throws, the second is the app's own.
            int ropeCalls = 0;
            var linear = RopeScalings.Default(LinearType)!;
            RopeScalings.Register(LinearType, input => ropeCalls++ == 0 ? throw new ArithmeticException("the app's scaling broke") : linear(input));
            var expected = new RopeSettings(10000f).Frequencies(8).Select(f => f / 2).ToArray();
            Check(Comparisons.Difference(expected, LinearRope().Frequencies(8), 1e-9) is null && ropeCalls == 1, "the failing call fell back to the library's scaling");
            Check(Comparisons.Difference(expected, LinearRope().Frequencies(8), 1e-9) is null && ropeCalls == 2, "the next call is the app's");

            var fellBack = recorder.Failed.Where(e => e.FellBack && e.Hint is null).Select(e => (e.Registry, e.Slot, e.Exception.GetType())).ToList();
            Check(fellBack.SequenceEqual([
                ("TokenSamplers", "default", typeof(InvalidOperationException)),
                ("TokenizerComponents.Normalizers", LowercaseType, typeof(InvalidDataException)),
                ("TokenizerComponents.Normalizers", LowercaseType, typeof(FormatException)),
                ("RopeScalings", LinearType, typeof(ArithmeticException))]), $"one telemetry event per fallback: {string.Join("; ", fellBack)}");
            Check(recorder.Failed.Count == 4 && recorder.Failed.All(e => e.Origin == "Idrak.Tests" && e.Implementation.StartsWith("Tests", StringComparison.Ordinal)),
                $"events name the app's implementation: {string.Join(", ", recorder.Failed.Select(e => e.Implementation))}");
            var report = Overrides.Report();
            Check(report.Single(o => o.Registry == "RopeScalings" && o.Name == LinearType) is { FallBacks: 1, Failures: 1, PolicyText: "fall back" }
                  && report.Single(o => o.Registry == "TokenizerComponents.Normalizers").FallBacks == 2, "the report counts the fallbacks");
        }
        finally
        {
            RemoveOverrides();
        }
    }

    private static void OverrideLoopShadow(Device device)
    {
        var recorder = new OverrideRecorder();
        using var subscription = Telemetry.Subscribe(recorder);
        try
        {
            // Every call compared (a rate of 1 keeps the test deterministic); the library answers each time.
            TokenSamplers.SetPolicy(TokenSamplers.DefaultName, SlotPolicy.Shadow, shadowRate: 1);
            TokenizerComponents.SetNormalizerPolicy(LowercaseType, SlotPolicy.Shadow, shadowRate: 1);
            RopeScalings.SetPolicy(LinearType, SlotPolicy.Shadow, shadowRate: 1);
            var builtIn = TokenSamplers.Default(TokenSamplers.DefaultName)!;
            TokenSamplers.Register(TokenSamplers.DefaultName, request => new NegatingSampler(builtIn(request)));
            TokenizerComponents.RegisterNormalizer(LowercaseType, _ => new FunctionNormalizer(s => s));        // forgets to lower the case
            var linear = RopeScalings.Default(LinearType)!;
            RopeScalings.Register(LinearType, input => linear(input with { Parameters = new JsonObject { ["factor"] = 4 } }));   // twice the factor

            Check(Choose(TokenSamplers.Create, device) == 1, "the library's sampler chooses");
            Check(BpeTokenizer.FromJson(LowercaseTokenizer()).Encode("AB").SequenceEqual([4]), "the built-in normalizer answers");
            var expected = new RopeSettings(10000f).Frequencies(8).Select(f => f / 2).ToArray();
            Check(Comparisons.Difference(expected, LinearRope().Frequencies(8), 1e-9) is null, "the library's scaling answers");

            var compared = recorder.Compared.ToDictionary(e => e.Registry);
            Check(compared.Count == 3 && compared.Values.All(e => !e.Agreed && e.Origin == "Idrak.Tests" && e.OverrideTime > TimeSpan.Zero && e.LibraryTime > TimeSpan.Zero),
                $"one comparison per slot, each differing: {string.Join("; ", recorder.Compared.Select(e => $"{e.Registry}: {e.Difference}"))}");
            Check(compared["TokenSamplers"].Difference!.Contains("1 of 1 steps", StringComparison.Ordinal)
                  && compared["TokenizerComponents.Normalizers"].Difference == "\"AB\", expected \"ab\""
                  && compared["RopeScalings"].Difference!.StartsWith("frequencies: element 0", StringComparison.Ordinal),
                "each difference says what differed");
            Check(recorder.Failed.Count == 0, "nothing failed in front of the caller");

            // An app's version that agrees, and one that throws: in Shadow neither reaches the caller.
            recorder.Compared.Clear();
            RopeScalings.Register(LinearType, linear);
            TokenizerComponents.RegisterNormalizer(LowercaseType, _ => new FunctionNormalizer(_ => throw new InvalidOperationException("broken")));
            Check(Comparisons.Difference(expected, LinearRope().Frequencies(8), 1e-9) is null && BpeTokenizer.FromJson(LowercaseTokenizer()).Encode("AB").SequenceEqual([4]),
                "the library answers");
            Check(recorder.Compared.Count == 2 && recorder.Compared[0].Agreed && recorder.Compared[0].Difference is null
                  && !recorder.Compared[1].Agreed && recorder.Compared[1].Exception is InvalidOperationException,
                $"an agreeing scaling, a throwing normalizer: {string.Join("; ", recorder.Compared.Select(e => $"{e.Registry}: {e.Agreed} {e.Difference}"))}");

            // A rate of 0 compares nothing.
            recorder.Compared.Clear();
            RopeScalings.SetPolicy(LinearType, SlotPolicy.Shadow, shadowRate: 0);
            _ = LinearRope().Frequencies(8);
            Check(recorder.Compared.Count == 0, "a rate of 0 compares no call");
        }
        finally
        {
            RemoveOverrides();
        }
    }

    private static void OverrideLoopThrowAndReport(Device device)
    {
        try
        {
            TokenSamplers.SetPolicy(TokenSamplers.DefaultName, SlotPolicy.Throw);
            TokenizerComponents.SetNormalizerPolicy(LowercaseType, SlotPolicy.Throw);
            RopeScalings.SetPolicy(LinearType, SlotPolicy.Throw);
            TokenSamplers.Register(TokenSamplers.DefaultName, _ => throw new InvalidOperationException("sampler"));
            TokenizerComponents.RegisterNormalizer(LowercaseType, _ => throw new InvalidDataException("normalizer"));
            RopeScalings.Register(LinearType, _ => throw new ArithmeticException("scaling"));
            Check(Throws<InvalidOperationException>(() => Choose(TokenSamplers.Create, device)), "the sampler's failure reaches the caller");
            Check(Throws<InvalidDataException>(() => BpeTokenizer.FromJson(LowercaseTokenizer())), "the normalizer's failure reaches the caller");
            Check(Throws<ArithmeticException>(() => LinearRope().Frequencies(8)), "the scaling's failure reaches the caller");

            // The report: each override, what it replaces, where it comes from and its policy; an added name too.
            RopeScalings.Register("test-added", input => new RopeScalingResult(input.Frequencies));
            var report = Overrides.Report();
            var rope = report.Single(o => o.Registry == "RopeScalings" && o.Name == LinearType);
            Check(rope is { ReplacesDefault: true, Guarded: true, Policy: SlotPolicy.Throw, Origin: "Idrak.Tests" } && rope.PolicyText == "throw"
                  && rope.ToString().Contains("RopeScalings/linear", StringComparison.Ordinal), $"the scaling's row: {rope}");
            Check(report.Single(o => o.Registry == "RopeScalings" && o.Name == "test-added") is { ReplacesDefault: false, PolicyText: "none (no library default)" },
                "a name only the app registered is listed as added");
            Check(report.Any(o => o.Registry == "TokenSamplers" && o.Name == "default") && report.Any(o => o.Registry == "TokenizerComponents.Normalizers" && o.Name == LowercaseType),
                "the sampler and the normalizer are listed");
            var rows = CliJson("overrides", "-j").AsArray();
            Check(rows.Any(r => (string?)r!["registry"] == "RopeScalings" && (string?)r["name"] == LinearType && (string?)r["policy"] == "Throw"
                                && (string?)r["origin"] == "Idrak.Tests" && (bool?)r["replaces_default"] == true)
                  && rows.Any(r => (string?)r!["name"] == "test-added" && r["policy"] is null), $"idrak overrides: {rows.ToJsonString()}");
            var (exit, text, _) = Cli("overrides");
            Check(exit == 0 && text.Contains("RopeScalings", StringComparison.Ordinal) && text.Contains("library default", StringComparison.Ordinal)
                  && text.Contains("throw", StringComparison.Ordinal), $"idrak overrides prints a table: {text}");
            Check(RopeScalings.Unregister("test-added") && !RopeScalings.Contains("test-added"), "an added name is removed");

            // Unregister brings each default back (and only once); Default is the library's all along.
            Check(TokenSamplers.Unregister(TokenSamplers.DefaultName) && TokenizerComponents.UnregisterNormalizer(LowercaseType) && RopeScalings.Unregister(LinearType),
                "the overrides are removed");
            Check(!TokenSamplers.Unregister(TokenSamplers.DefaultName) && !RopeScalings.Unregister(LinearType), "a library default is never removed");
            Check(TokenSamplers.Origin(TokenSamplers.DefaultName) == Overrides.Library && RopeScalings.Get(LinearType) == RopeScalings.Default(LinearType)
                  && TokenizerComponents.FindNormalizer(LowercaseType) is null && TokenizerComponents.NormalizerOrigin(LowercaseType) == Overrides.Library,
                "each slot is the library's again");
            Check(Choose(TokenSamplers.Create, device) == 1 && BpeTokenizer.FromJson(LowercaseTokenizer()).Encode("AB").SequenceEqual([4])
                  && Comparisons.Difference([.. new RopeSettings(10000f).Frequencies(8).Select(f => f / 2)], LinearRope().Frequencies(8), 1e-9) is null,
                "and answers as before, under the Throw policy it kept");
            Check(!Overrides.Report().Any(o => o.Registry is "RopeScalings" or "TokenSamplers" or "TokenizerComponents.Normalizers"), "the report is empty for them");
        }
        finally
        {
            RemoveOverrides();
        }
    }

    private static void OverrideLoopPolicyVariable(Device device)
    {
        _ = device;
        var defaults = (SlotPolicy.Throw, Slot.DefaultShadowRate);
        Check(Overrides.PolicyFrom(null, "RopeScalings", "yarn") == defaults && Overrides.PolicyFrom("", "RopeScalings", "yarn") == defaults,
            "not set: Throw");
        Check(Overrides.PolicyFrom("fallback", "RopeScalings", "yarn") == (SlotPolicy.FallBack, Slot.DefaultShadowRate)
              && Overrides.PolicyFrom("Shadow:0.25", "GraphOps", "relu") == (SlotPolicy.Shadow, 0.25), "a policy for every slot, with a rate");
        Check(Overrides.PolicyFrom("shadow, ropescalings/YARN=fallback", "RopeScalings", "yarn") == (SlotPolicy.FallBack, Slot.DefaultShadowRate)
              && Overrides.PolicyFrom("shadow, RopeScalings/yarn=fallback", "RopeScalings", "linear") == (SlotPolicy.Shadow, Slot.DefaultShadowRate),
            "a slot's own entry wins over the policy for every slot (names ignore case)");
        Check(Overrides.PolicyFrom("sometimes, RopeScalings/yarn=shadow:2", "RopeScalings", "yarn") == defaults, "entries it cannot read are ignored");
    }

    private static void OverrideLoopSlotTable(Device device)
    {
        _ = device;
        Func<int, int> Twice = x => 2 * x, Thrice = x => 3 * x, Broken = _ => throw new InvalidOperationException();
        var table = new SlotTable<string, Func<int, int>>("Test", (slot, app, library) => x => slot.Call(() => app(x), () => library(x), Comparisons.Exact),
            StringComparer.OrdinalIgnoreCase, newestFirst: true);
        Overrides.AsLibraryDefaults(() =>
        {
            table.Register("b", Twice);
            table.Register("a", Twice);
        });
        Check(table.Keys.SequenceEqual(["a", "b"]) && table.Origin("A") == Overrides.Library && ReferenceEquals(table.Find("a"), Twice),
            "library registrations are defaults, newest first, handed out as they are");
        table.SetPolicy("b", SlotPolicy.FallBack);
        table.Register("b", Broken);
        table.Register("c", Thrice);
        Check(table.Keys.SequenceEqual(["c", "a", "b"]) && table.Find("b")!(5) == 10 && table.Find("c")!(5) == 15 && table.Origin("b") == "Idrak.Tests",
            "the app's entry takes the default's place and falls back to it (FallBack); a new name goes first");
        Check(table.Report().Select(o => (o.Name, o.ReplacesDefault)).SequenceEqual([("c", false), ("b", true)]) && ReferenceEquals(table.Default("b"), Twice),
            "the table's report and default");
        Check(table.Unregister("b") && table.Unregister("c") && !table.Unregister("a") && table.Keys.SequenceEqual(["a", "b"]) && ReferenceEquals(table.Find("b"), Twice),
            "unregistering restores the default and removes the added name");

        var data = new SlotTable<int, string?>("TestData", unguarded: "data");
        try
        {
            data.SetPolicy(1, SlotPolicy.Shadow);
            Check(false, "a table without a guard has no policy");
        }
        catch (NotSupportedException e)
        {
            Check(e.Message.Contains("data", StringComparison.Ordinal), e.Message);
        }

        data.Register(1, null, typeof(Tests).Assembly);
        Check(data.TryGet(1, out var value) && value is null && data.Origin(1) == "Idrak.Tests" && data.Unregister(1) && !data.Contains(1),
            "a description registered by an assembly, null included");
    }
}
