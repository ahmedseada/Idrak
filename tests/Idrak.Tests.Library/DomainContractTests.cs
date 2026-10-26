// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Diagnostics;
using Idrak.Abstraction.Generation;
using Idrak.Abstraction.Reinforcement;
using Idrak.Audio.Abstractions;
using Idrak.Data.Abstractions;
using Idrak.Models.Abstractions;
using Idrak.Text;
using Idrak.Text.Abstractions;
using Idrak.Vision.Abstractions;

// Plan 33, C5: the domain packages' contracts as interfaces. An episode's info typed by its keys (no flag or count through
// text), options declared once and parsed once from spans, capabilities by interface (a live audio processor, a sound or
// feature augmentation, a fine-tuning loss's needs, a binary part's record keys, a data format's kind), advisors typed
// by their subject, and a teacher that must say what its vocabulary is.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] DomainContractGroup =
    [
        ("domain contracts: an episode's info keeps each value in its type under an EpisodeInfoKey<T> (a flag a bool, a count an int), prints them as before (true, invariant numbers, sorted by name), refuses a key of another type and a type it does not hold, and crosses processes with its types (Write, Read); reading a value allocates nothing", EpisodeInfoContract),
        ("domain contracts: an option schema declares each option's type, range and default once; Parse checks every key, parses each value from its span once (the same dictionary gives the same values), refuses a bad value naming what it takes, keeps text as given; the rewards, estimators, losses and classic environments declare theirs (IOptionSchema, their Keys the schema's)", OptionSchemaContract),
        ("domain contracts: capabilities are interfaces: a processor is live exactly when it is an ILiveAudioProcessor, an augmentation varies the sound or the features by its interface, a fine-tuning loss says what it needs (DPO pairs and the reference, distillation a teacher, a sum both terms'), a data format its kind (Parquet binary, code code), the audio part kind its record keys", CapabilityContracts),
        ("domain contracts: a typed advisor answers only for its subject's type; a teacher says what its vocabulary is (a fingerprint checked, an unknown one with its reason); the library's decoders' network parts and adapter targets; kinds registered by their names (keyword indexes, detection decoders, vision losses, matchers and metrics, vocoders, feature extractors)", TypedAdvisorsAndKinds),
    ];

    internal static readonly EpisodeInfoKey<bool> TestPassed = new("test_passed");
    internal static readonly EpisodeInfoKey<double> TestScore = new("test_score");
    internal static readonly EpisodeInfoKey<string> TestTask = new("test_task");

    internal static void EpisodeInfoContract(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(EpisodeInfoContract)))
        {
            return;
        }

        var info = new EpisodeInfo().Set(EpisodeInfoKeys.Rounds, 4).Set(TestPassed, true).Set(TestScore, 0.25).Set(TestTask, "csharp/hello");
        Check(info.Get(EpisodeInfoKeys.Rounds) == 4 && info.Get(TestPassed) && info.Get(TestScore) == 0.25 && info.Get(TestTask) == "csharp/hello" && info.Count == 4,
            $"the values in their types: {info}");
        Check(info.ToString() == "rounds=4, test_passed=true, test_score=0.25, test_task=csharp/hello", $"printed as the text info was: {info}");
        Check(!info.TryGet(EpisodeInfoKeys.Tokens, out _) && info.GetOrDefault(EpisodeInfoKeys.Tokens, -1) == -1, "a key not set");
        ThrowsNaming<InvalidOperationException>(() => info.Get(new EpisodeInfoKey<string>("rounds")), "a key of another type", "rounds", "Int32", "String");
        ThrowsNaming<KeyNotFoundException>(() => info.Get(EpisodeInfoKeys.Tokens), "a key not set", "tokens", "rounds");
        ThrowsNaming<NotSupportedException>(() => _ = new EpisodeInfoKey<decimal>("money"), "a type the info does not hold", "Decimal");

        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            info.Write(writer);
        }

        stream.Position = 0;
        using var reader = new BinaryReader(stream);
        var back = EpisodeInfo.Read(reader);
        Check(back.ToString() == info.ToString() && back.Get(EpisodeInfoKeys.Rounds) == 4 && back.Get(TestPassed) && back.Get(TestScore) == 0.25, $"read back with its types: {back}");

        int rounds = 0;
        for (int i = 0; i < 1000; i++)
        {
            rounds += info.Get(EpisodeInfoKeys.Rounds);                     // warm
        }

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 1000; i++)
        {
            rounds += info.Get(EpisodeInfoKeys.Rounds) + (info.Get(TestPassed) ? 1 : 0);
        }

        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Check(allocated == 0 && rounds > 0, $"reading typed values allocates nothing: {allocated} bytes");
    }

    internal static void OptionSchemaContract(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(OptionSchemaContract)))
        {
            return;
        }

        var rate = OptionSpec.Number("rate", 0.5f, 0f, 1f, summary: "a rate");
        var steps = OptionSpec.Whole("steps", 10, min: 1);
        var mode = OptionSpec.Choice("mode", "fast", "fast", "slow");
        var on = OptionSpec.Flag("on");
        var name = OptionSpec.Text("name");
        var schema = new OptionSchema("test kind", rate, steps, mode, on, name);
        Check(schema.Keys.Ordered.SequenceEqual(["rate", "steps", "mode", "on", "name"]) && schema.Keys.Contains("mode".AsSpan()), $"the keys in order: {schema.Keys}");
        var given = new Dictionary<string, string> { ["rate"] = " 0.25 ", ["mode"] = "SLOW", ["on"] = "yes", ["name"] = " kept " };
        var values = schema.Parse(given);
        Check(values.Get(rate) == 0.25f && values.Get(steps) == 10 && values.Get(mode) == "slow" && values.Get(on) && values.Get(name) == " kept "
              && values.IsGiven(rate) && !values.IsGiven(steps), "the values parsed, the defaults kept, text as given");
        Check(ReferenceEquals(schema.Parse(given), values), "the same dictionary is parsed once");
        ThrowsNaming<ArgumentException>(() => schema.Parse(new Dictionary<string, string> { ["speed"] = "1" }), "an undeclared key",
            "The test kind does not take the option 'speed'; it takes: rate, steps, mode, on, name.");
        ThrowsNaming<ArgumentException>(() => schema.Parse(new Dictionary<string, string> { ["rate"] = "2" }), "a value out of range", "The test kind's option 'rate' is '2': a number from 0 to 1.");
        ThrowsNaming<ArgumentException>(() => schema.Parse(new Dictionary<string, string> { ["mode"] = "medium" }), "a choice not declared", "one of fast, slow");
        Check(new OptionSchema("blank kind", true, steps).Parse(new Dictionary<string, string> { ["steps"] = "  " }).Get(steps) == 10, "a blank value is the default where the schema says so");
        Check(schema.Parse(new Dictionary<string, string> { ["x.y"] = "1" }, key => key.Contains('.', StringComparison.Ordinal)).Rest["x.y"] == "1", "keys the caller also takes are kept as given");

        foreach (var reward in new[] { "metric", "regex", "json", "length", "sum" })
        {
            var kind = RewardFunctions.Get(reward);
            Check(kind is IOptionSchema declared && ReferenceEquals(declared.Options.Keys, kind.Keys), $"the reward {reward} declares its options once");
        }

        foreach (var estimator in AdvantageEstimators.Names)
        {
            Check(AdvantageEstimators.Get(estimator) is IOptionSchema, $"the estimator {estimator} declares its options");
        }

        ThrowsNaming<ArgumentException>(() => RewardFunctions.Get("length").ScoreAsync([new RewardInput("p", "a")], new Dictionary<string, string> { ["budget"] = "lots" }).AsTask().GetAwaiter().GetResult(),
            "a budget that is not a number", "The reward length's option 'budget' is 'lots'");
    }

    internal static void CapabilityContracts(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(CapabilityContracts)))
        {
            return;
        }

        foreach (string name in AudioProcessors.Names)
        {
            var processor = AudioProcessors.Get(name);
            Check(processor.Live == processor is ILiveAudioProcessor && (processor is ILiveAudioProcessor) != (processor is IRecordingAudioProcessor),
                $"{name}: live exactly when it is an ILiveAudioProcessor, and one of the two");
        }

        Check(AudioAugmentations.Get(AudioAugmentations.Speed) is ISoundAugmentation { Stage: AudioAugmentationStage.Sound }
              && AudioAugmentations.Get(AudioAugmentations.SpecMask) is IFeatureAugmentation { Stage: AudioAugmentationStage.Features }, "the augmentations' stages follow their interfaces");

        var dpo = FineTuningLosses.Dpo();
        var distillation = FineTuningLosses.Distillation();
        Check(dpo is { NeedsPairs: true, NeedsReference: true, NeedsTeacher: false } && FineTuningLosses.Orpo() is { NeedsPairs: true, NeedsReference: false }
              && distillation is { NeedsTeacher: true, NeedsPairs: false } && FineTuningLosses.Distillation(alpha: 0f) is { NeedsTeacher: false }
              && FineTuningLosses.TokenCrossEntropy is { NeedsPairs: false, NeedsReference: false, NeedsTeacher: false }, "each loss says what it needs");
        Check(FineTuningLosses.Add(FineTuningLosses.TokenCrossEntropy, dpo) is { NeedsPairs: true, NeedsReference: true }
              && FineTuningLosses.Add(FineTuningLosses.TokenCrossEntropy, distillation, 0f) is { NeedsTeacher: false }, "a sum needs what its weighted terms need");

        Check(DataFileFormats.Get("Parquet").Kind == DataFileKind.Binary && DataFileFormats.Get("Code").Kind == DataFileKind.Code && DataFileFormats.Get("Csv").Kind == DataFileKind.Table,
            "the data formats' kinds");

        var audio = ChatParts.BinaryKinds.SingleOrDefault(k => k.Name == ChatParts.Audio);
        Check(audio?.RecordKeys is { List: "audios", Marker: "<audio>", InputPart: "input_audio" } keys && keys.Formats["mp3"] == "audio/mpeg",
            $"the audio part kind's record keys: {audio?.RecordKeys}");
    }

    internal static void TypedAdvisorsAndKinds(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(TypedAdvisorsAndKinds)))
        {
            return;
        }

        ISettingAdvisor advisor = new SubjectAdvisor();
        Check(advisor.Suggest(new AdviceSituation(Device.Cpu, "a text subject", null)).Single().Value == "a text subject"
              && !advisor.Suggest(new AdviceSituation(Device.Cpu, new object(), null)).Any(), "a typed advisor answers for its subject's type only");

        var unknown = TeacherVocabulary.Unknown("the logits were written without a fingerprint");
        Check(!unknown.IsKnown && unknown.ToString().Contains("without a fingerprint", StringComparison.Ordinal), $"an unknown vocabulary says why: {unknown}");
        var student = new CharTokenizer("abc");
        unknown.Check(student);
        var other = TeacherVocabulary.Fingerprinted("17-0000000000000000");
        ThrowsNaming<InvalidOperationException>(() => other.Check(student), "another vocabulary's fingerprint", "do not share a vocabulary", "fingerprint");
        TeacherVocabulary.Fingerprinted(DistillationTeacher.VocabularyFingerprint(student)).Check(student);
        TeacherVocabulary.Of(student).Check(student);

        var parts = NetworkParts.Decoder;
        Check(parts.PartOf("embed") == INetworkParts.Embeddings && parts.PartOf("pos.weight") == INetworkParts.Embeddings && parts.PartOf("layers.3.attn.q") == INetworkParts.Decoder
              && parts.PartOf("head") == INetworkParts.Head && parts.AdapterTargets.SequenceEqual(["q", "k", "v", "o", "gate", "up", "down"]), "the library's decoders' parts");

        Check(KeywordIndexes.Get("BM25") is { Name: KeywordIndexes.Bm25 } && DetectionDecoders.Get(DetectionDecoders.BoxesScores).Keys.Contains("layout")
              && VisionLosses.Get("giou").Name == "giou" && BoxMatchers.Get("HUNGARIAN").Name == BoxMatchers.Hungarian && VisionMetrics.Get("coco") is { Task: ImageTask.Detection }
              && Vocoders.Find(Vocoders.DefaultName) is { } griffin && griffin.Accepts(new AudioFeatureOptions()) && AudioFeatureExtractors.Default("log-mel") is { Name: "log-mel" },
            "kinds found by their names");
        ThrowsNaming<ArgumentException>(() => KeywordIndexes.Create(KeywordIndexes.Bm25, ["a"], new KeywordIndexOptions { Parameters = new Dictionary<string, double> { ["boost"] = 2 } }),
            "a setting bm25 does not take", "'boost'");
    }

    // A typed advisor: it suggests the subject's own text.
    internal sealed class SubjectAdvisor : ISettingAdvisor<string>
    {
        public string Name => "domain-contract-test";

        public string Summary => "a test's own";

        public IEnumerable<SettingSuggestion> Suggest(AdviceSituation situation, string subject) => [new("Test.Subject", subject, "the subject itself", "cpu", Name)];
    }
}
