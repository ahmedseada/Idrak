// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Generation;
using Idrak.Abstraction.Training;

// Plan 33, C0: the shared kind contract (IKind, IConfigurableKind, IKind<TRequest, T>, OptionKeys), KindTable, and
// SlotTable's lookup of string names by span.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] KindTableGroup =
    [
        ("slot table span lookup: a table of string names finds by span what it finds by string, in both layers and under the comparer's case rule, and allocates nothing once warm", SlotTableSpanLookup),
        ("slot table span lookup: a comparer that cannot take spans still finds the same entries (through a string), and a table of value-type keys refuses a span", SlotTableSpanFallback),
        ("kind table: kinds register by their name (one or several at once), are found by span, list names and summaries in order, and an unknown name is refused as every registry words it, with a NotRegisteredException naming the registry, the name and the registered ones", KindTableRegistry),
        ("kind table: option keys keep their declared order for listings and are checked by span; a kind's request may be a ref struct; the library's kind registries find their kinds by span", KindTableKeysAndRegistries),
        ("kind table: not registered: every registry of the library and its packages refuses a name nobody registered with a NotRegisteredException naming the registry, the name and its registered names, its message worded as before (the kind tables' word for word)", NotRegisteredRefusals),
    ];

    internal static void SlotTableSpanLookup(Device device)
    {
        _ = device;
        foreach (var comparer in new IEqualityComparer<string>?[] { null, StringComparer.Ordinal, StringComparer.OrdinalIgnoreCase })
        {
            string label = comparer is null ? "the default comparer" : comparer == StringComparer.Ordinal ? "Ordinal" : "OrdinalIgnoreCase";
            var table = new SlotTable<string, string>("TestSpans", comparer: comparer, unguarded: "test");
            Overrides.AsLibraryDefaults(() =>
            {
                table.Register("alpha", "library alpha");
                table.Register("beta", "library beta");
            });
            table.Register("beta", "app beta");
            table.Register("gamma", "app gamma");

            const string Line = "--kind=alpha,beta,gamma,delta,ALPHA";
            foreach (string name in new[] { "alpha", "beta", "gamma", "delta", "ALPHA", "Beta" })
            {
                int at = Line.IndexOf(name, StringComparison.Ordinal);
                var span = at >= 0 ? Line.AsSpan(at, name.Length) : name.AsSpan();
                bool byString = table.TryGet(name, out var found), bySpan = table.TryGet(span, out var spanFound);
                Check(byString == bySpan && found == spanFound && table.Find(span) == table.Find(name) && table.Contains(span) == table.Contains(name),
                    $"{label}: '{name}' by span ({bySpan}, {spanFound}) as by string ({byString}, {found})");
            }

            Check(table.Find("beta".AsSpan()) == "app beta" && table.Find("alpha".AsSpan()) == "library alpha" && table.Find("gamma".AsSpan()) == "app gamma",
                $"{label}: the app's registration over the library's, the library's alone, and the app's alone");
            Check(table.Find("ALPHA".AsSpan()) == (comparer == StringComparer.OrdinalIgnoreCase ? "library alpha" : null), $"{label}: the comparer's case rule");
            Check(table.Unregister("beta") && table.Find("beta".AsSpan()) == "library beta", $"{label}: a new snapshot's span lookup sees the default come back");

            // Warm: no string is made, no lookup is built per call.
            var names = Line.AsMemory(7);
            int hits = 0;
            void Lookups()
            {
                foreach (var range in names.Span.Split(','))
                {
                    hits += table.Contains(names.Span[range]) ? 1 : 0;
                    hits += table.TryGet(names.Span[range], out _) ? 1 : 0;
                    hits += table.Find(names.Span[range]) is null ? 0 : 1;
                }
            }

            for (int i = 0; i < 2_000; i++)
            {
                Lookups();
            }

            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 1_000; i++)
            {
                Lookups();
            }

            long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            Check(allocated == 0 && hits > 0, $"{label}: 15,000 span lookups allocated {allocated} bytes once warm{(allocated == 0 ? "" : $" ({AllocatedTypes(Lookups)})")}");
        }
    }

    internal static void SlotTableSpanFallback(Device device)
    {
        _ = device;
        var table = new SlotTable<string, int>("TestUpper", comparer: new UpperComparer(), unguarded: "test");
        Overrides.AsLibraryDefaults(() => table.Register("alpha", 1));
        table.Register("Beta", 2);
        Check(table.Find("ALPHA".AsSpan()) == 1 && table.TryGet("beta".AsSpan(), out int beta) && beta == 2 && !table.Contains("gamma".AsSpan()),
            "a comparer without a span lookup finds the entries by its own rule, through a string made from the span");

        var numbered = new SlotTable<int, string>("TestNumbered", unguarded: "test");
        numbered.Register(1, "one");
        try
        {
            _ = numbered.Find("1".AsSpan());
            Check(false, "a table of int keys refuses a span");
        }
        catch (NotSupportedException e)
        {
            Check(e.Message.Contains("strings", StringComparison.Ordinal) && e.Message.Contains("Int32", StringComparison.Ordinal), e.Message);
        }
    }

    internal static void KindTableRegistry(Device device)
    {
        _ = device;
        var table = new KindTable<Greeting>("TestKinds", "Test kind", "add it with TestKinds.Register", comparer: StringComparer.OrdinalIgnoreCase, unguarded: "test");
        Greeting alpha = new("alpha", "the first"), beta = new("beta", "the second"), mine = new("Beta", "the app's second");
        Overrides.AsLibraryDefaults(() => table.Register(alpha, beta));
        Check(table.Names.SequenceEqual(["alpha", "beta"]) && table.Kinds.SequenceEqual([alpha, beta]) && table.Origin("alpha") == Overrides.Library,
            "several kinds registered at once are library defaults, in order, each under its own name");
        Check(ReferenceEquals(table.Find("ALPHA".AsSpan()), alpha) && table.TryGet("Beta".AsSpan(), out var found) && ReferenceEquals(found, beta)
              && table.Contains("beta") && !table.Contains("gamma".AsSpan()) && ReferenceEquals(table.Get("alpha"), alpha),
            "found by span and by string, ignoring case as the table compares");

        table.Register(mine);
        Check(ReferenceEquals(table.Find("beta".AsSpan()), mine) && ReferenceEquals(table.Default("beta"), beta) && table.Origin("beta") == typeof(Tests).Assembly.GetName().Name
              && table.Names.SequenceEqual(["alpha", "Beta"]) && string.Equals(table.Slots.Report().Single().Name, "beta", StringComparison.OrdinalIgnoreCase),
            $"an app's kind under a library name shadows it, in its place, and is reported: names {string.Join(", ", table.Names)}, origin {table.Origin("beta")}, "
            + $"report {string.Join("; ", table.Slots.Report())}");
        Check(table.Describe() == "alpha: the first\nBeta: the app's second" && table.Listed() == "alpha, Beta",
            $"names and summaries listed in order: {table.Describe()}");
        Check(table.Unregister("BETA") && ReferenceEquals(table.Find("beta".AsSpan()), beta), "unregistering brings the library's kind back");

        // The name a table lists is spelled as its last registration or removal spelled it (the table ignores case).
        ThrowsNaming<NotSupportedException>(() => table.Get("gamma".AsSpan()), "an unknown kind",
            $"Test kind 'gamma' is not registered (registered: {table.Listed()}); add it with TestKinds.Register.", "registered: alpha, ");
        try
        {
            table.Get("gamma".AsSpan());
            Check(false, "an unknown kind is refused");
        }
        catch (NotRegisteredException e)
        {
            Check(e.Registry == "TestKinds" && e.Name == "gamma" && e.Registered.SequenceEqual(table.Names) && e.Message == table.NotRegistered("gamma"),
                $"the refusal is a NotRegisteredException (a NotSupportedException) naming the registry, the name and the registered ones: {e.Registry} {e.Name} [{string.Join(", ", e.Registered)}]");
        }

        ThrowsNaming<ArgumentException>(() => table.Register(new Greeting(" ", "nameless")), "a kind without a name", "kind");
        var empty = new KindTable<Greeting>("TestEmpty", "Test kind", "add one", unguarded: "test");
        Check(empty.Listed() == "none" && empty.NotRegistered("x") == "Test kind 'x' is not registered (registered: none); add one.", empty.NotRegistered("x"));
    }

    internal static void NotRegisteredRefusals(Device device)
    {
        _ = device;
        const string None = "KindTestNobody";
        static string Listed(IEnumerable<string> names) => string.Join(", ", names);
        static string Id(int id) => id.ToString(System.Globalization.CultureInfo.InvariantCulture);
        static string Typed(Type t) => t.Name;

        // Each registry, the name it is asked for, the names it holds, the start of its message (the whole message for the
        // kind tables, which word it through KindTable.NotRegistered) and the lookup that refuses.
        (string Registry, string Name, IEnumerable<string> Registered, string Message, Action Get)[] refusals =
        [
            ("PretrainedArchitectures", None, Idrak.Models.Abstractions.PretrainedArchitectures.Names, $"No architecture '{None}' is registered (",
                () => Idrak.Models.Abstractions.PretrainedArchitectures.Get(None)),
            ("PretrainedEncoders", None, Idrak.Models.Abstractions.PretrainedEncoders.Names, $"Encoder family '{None}' is not registered (registered: ",
                () => Idrak.Models.Abstractions.PretrainedEncoders.Get(None)),
            ("PretrainedSeq2Seq", None, Idrak.Models.Abstractions.PretrainedSeq2Seq.Names, $"Encoder-decoder family '{None}' is not registered (registered: ",
                () => Idrak.Models.Abstractions.PretrainedSeq2Seq.Get(None)),
            ("EncoderHeads", None, Idrak.Models.Abstractions.EncoderHeads.Names,
                $"Encoder head '{None}' is not registered (registered: {Listed(Idrak.Models.Abstractions.EncoderHeads.Names)}); add it with EncoderHeads.Register.",
                () => Idrak.Models.Abstractions.EncoderHeads.Get(None)),
            ("AttentionPositionBiases", None, AttentionPositionBiases.Names,
                $"Attention position bias '{None}' is not registered (registered: {Listed(AttentionPositionBiases.Names)}); "
                + "register it with AttentionPositionBiases.Register (the plug-in that brings the kind does).",
                () => AttentionPositionBiases.Get(None)),
            ("FeatureCaches", None, FeatureCaches.Names,
                $"Feature cache '{None}' is not registered (registered: {Listed(FeatureCaches.Names)}); add it with FeatureCaches.Register.", () => FeatureCaches.Get(None)),
            ("ProgressReporters", None, ProgressReporters.Names,
                $"Progress reporter '{None}' is not registered (registered: {Listed(ProgressReporters.Names)}); add it with ProgressReporters.Register.",
                () => ProgressReporters.Get(None)),
            ("TokenizerModels", None, Idrak.Models.Abstractions.TokenizerModels.Names,
                $"Tokenizer model '{None}' is not registered (registered: {Listed(Idrak.Models.Abstractions.TokenizerModels.Names)}); add it with TokenizerModels.Register.",
                () => Idrak.Models.Abstractions.TokenizerModels.Get(None)),
            ("GgufTokenizers", None, Idrak.Models.Abstractions.GgufTokenizers.Names,
                $"GGUF tokenizer '{None}' is not registered (registered: {Listed(Idrak.Models.Abstractions.GgufTokenizers.Names)}); add it with GgufTokenizers.Register.",
                () => Idrak.Models.Abstractions.GgufTokenizers.Get(None)),
            ("GgufChatTemplates", None, Idrak.Models.Abstractions.GgufChatTemplates.Names, $"GGUF chat template '{None}' is not registered (registered: ",
                () => Idrak.Models.Abstractions.GgufChatTemplates.Get(None)),
            ("GgufArchitectures", None, Idrak.Models.Abstractions.GgufArchitectures.Names, $"No GGUF architecture '{None}' is registered (",
                () => Idrak.Models.Abstractions.GgufArchitectures.Get(None)),
            ("GgufTypes", "999", Idrak.Models.Abstractions.GgufTypes.Ids.Select(Id), "No ggml type #999 is registered (", () => Idrak.Models.Abstractions.GgufTypes.Get(999)),
            ("GgufTypes", "999", Idrak.Models.Abstractions.GgufTypes.Ids.Select(Id), "ggml type #999 is not supported yet (supported: ",
                () => Idrak.Models.GgufFile.Dequantize(999, [], [])),
            ("ToolCallFormats", None, Idrak.Abstraction.Generation.ToolCallFormats.Names, $"No tool-call format '{None}' is registered (",
                () => Idrak.Abstraction.Generation.ToolCallFormats.Get(None)),
            ("RopeScalings", None, RopeScalings.Names, $"No RoPE scaling '{None}' is registered (", () => RopeScalings.Get(None)),
            ("TokenSamplers", None, TokenSamplers.Names, $"No token sampler '{None}' is registered (", () => TokenSamplers.Get(None)),
            ("PackedWeightFormats", None, PackedWeightFormats.Names, $"No packed format '{None}' is registered (", () => PackedWeightFormats.Get(None)),
            ("CtcDecoders", None, CtcDecoders.Names, $"No CTC decoder '{None}' is registered (", () => CtcDecoders.Get(None)),
            ("KeyValueLayouts", None, KeyValueLayouts.Names, $"No key/value cache format '{None}' is registered (", () => KeyValueLayouts.Get(None)),
            ("ImageAttentionRules", None, ImageAttentionRules.Names, $"No image attention rule '{None}' is registered (", () => ImageAttentionRules.Get(None)),
            ("ImagePositionRules", None, ImagePositionRules.Names, $"No image position rule '{None}' is registered (", () => ImagePositionRules.Get(None)),
            ("CheckpointFormats", None, Idrak.Abstraction.Formats.CheckpointFormats.Names, $"No checkpoint format '{None}' is registered (",
                () => Idrak.Abstraction.Formats.CheckpointFormats.Get(None)),
            ("ModelSources", None, Idrak.Abstraction.Formats.ModelSources.Names, $"No model source '{None}' is registered (", () => Idrak.Abstraction.Formats.ModelSources.Get(None)),
            ("SampleSources", None, SampleSources.Names, $"No sample source '{None}' is registered (", () => SampleSources.Get(None)),
            ("ScanProfiles", None, ScanProfiles.Names, $"No scan profile '{None}' is registered (", () => ScanProfiles.Get(None)),
            ("ImageCodecs", None, ImageCodecs.Names, $"No image codec '{None}' is registered (", () => ImageCodecs.Get(None)),
            ("PageLayouts", None, PageLayouts.Names, $"No page layout '{None}' is registered (", () => PageLayouts.Get(None)),
            ("ImageEncoders", None, ImageEncoders.Names, $"No image encoder '{None}' is registered (", () => ImageEncoders.Get(None)),
            ("FontFormats", None, FontFormats.Names, $"No font format '{None}' is registered (", () => FontFormats.Get(None)),
            ("OutlineRasterizers", None, OutlineRasterizers.Names, $"No outline rasterizer '{None}' is registered (", () => OutlineRasterizers.Get(None)),
            ("IdrakPlugins", None, Idrak.Abstraction.Plugins.IdrakPlugins.Names, $"No plug-in '{None}' is registered (registered: ",
                () => Idrak.Abstraction.Plugins.IdrakPlugins.Get(None)),
            ("GraphOps", None, Idrak.Layers.Abstractions.GraphOps.Names, $"No graph operation '{None}' is registered (", () => Idrak.Layers.Abstractions.GraphOps.Get(None)),
            ("NetworkOps", None, Idrak.Layers.Abstractions.NetworkOps.Names, $"No network step '{None}' is registered (", () => Idrak.Layers.Abstractions.NetworkOps.Get(None)),
            ("OnnxImportOps", None, Idrak.Onnx.Abstractions.OnnxImportOps.Names, $"No ONNX import op '{None}' is registered (", () => Idrak.Onnx.Abstractions.OnnxImportOps.Get(None)),
            ("OnnxExportOps.GraphOps", None, Idrak.Onnx.Abstractions.OnnxExportOps.GraphOpNames, $"No ONNX export translator for the graph operation '{None}' is registered (",
                () => Idrak.Onnx.Abstractions.OnnxExportOps.GetGraphOp(None)),
            ("OnnxExportOps.Modules", nameof(NotRegisteredModule), Idrak.Onnx.Abstractions.OnnxExportOps.Types.Select(Typed).Order(),
                $"No ONNX export translator for {nameof(NotRegisteredModule)} is registered (", () => Idrak.Onnx.Abstractions.OnnxExportOps.Get<NotRegisteredModule>()),
            ("AudioCodecs", None, Idrak.Audio.Abstractions.AudioCodecs.Names, $"No audio codec '{None}' is registered (", () => Idrak.Audio.Abstractions.AudioCodecs.Get(None)),
            ("DataFileFormats", None, Idrak.Data.Abstractions.DataFileFormats.Names, $"No data file format '{None}' is registered (",
                () => Idrak.Data.Abstractions.DataFileFormats.Get(None)),
            ("DatasetSources", None, Idrak.Data.Abstractions.DatasetSources.Names, $"No dataset source '{None}' is registered (", () => Idrak.Data.Abstractions.DatasetSources.Get(None)),
            ("AnnotationFormats", None, Idrak.Data.Abstractions.AnnotationFormats.Names, $"No annotation format '{None}' is registered (",
                () => Idrak.Data.Abstractions.AnnotationFormats.Get(None)),
        ];
        foreach (var (registry, name, registered, message, get) in refusals)
        {
            try
            {
                get();
                Check(false, $"{registry}: '{name}' is refused");
            }
            catch (NotRegisteredException e)
            {
                Check(e.Registry == registry && e.Name == name && e.Registered.SequenceEqual(registered) && e.Message.StartsWith(message, StringComparison.Ordinal),
                    $"{registry}: the refusal of '{name}' is a NotRegisteredException naming the registry, the name and the registered ones, its message as before: "
                    + $"{e.Registry} {e.Name} [{string.Join(", ", e.Registered)}]: {e.Message}");
            }
        }
    }

    // A module type no ONNX export translator is registered for.
    internal sealed class NotRegisteredModule : Module
    {
        protected override Tensor ForwardCore(Tensor input) => input;
    }

    internal static void KindTableKeysAndRegistries(Device device)
    {
        _ = device;
        var keys = new OptionKeys("max-steps", "gamma", "alpha", "gamma", "beta");
        Check(keys.SequenceEqual(["max-steps", "gamma", "alpha", "beta"]) && keys.Count == 4 && keys.ToString() == "max-steps, gamma, alpha, beta",
            $"declared order kept, a repeated key once: {keys}");
        const string Option = "alpha=1";
        Check(keys.Contains(Option.AsSpan(0, 5)) && !keys.Contains(Option.AsSpan(0, 4)) && keys.Contains("beta") && !keys.Contains("Beta")
              && keys.SetEquals(["alpha", "beta", "gamma", "max-steps"]) && OptionKeys.None.Count == 0 && OptionKeys.None.ToString() == "none",
            "keys checked exactly, by span and by string");
        long before = GC.GetAllocatedBytesForCurrentThread();
        int hits = 0;
        for (int i = 0; i < 10_000; i++)
        {
            hits += keys.Contains(Option.AsSpan(0, 5)) ? 1 : 0;
        }

        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Check(allocated == 0 && hits == 10_000, $"10,000 key checks by span allocated {allocated} bytes");

        IKind<ReadOnlySpan<char>, string> greeting = new Greeting("hello", "greets");
        Check(greeting.Create("world".AsSpan()) == "hello, world" && greeting.Keys.Count == 1, "a kind made from a ref struct request");

        Check(OptimizerKinds.Find("ADAMW".AsSpan())?.Name == "adamw" && LearningRateSchedules.Find("Cosine".AsSpan())?.Name == "cosine"
              && ChatParts.Find("image".AsSpan())?.Name == ChatParts.Image && ChatParts.Find("Image".AsSpan()) is null,
            "the library's kind registries find their kinds by span, each under its comparer");
        ThrowsNaming<NotSupportedException>(() => ChatParts.Get("nobody"), "an unknown chat part kind",
            "Chat part kind 'nobody' is not registered (registered: text, image", "ChatParts.Register");
        using var made = OptimizerKinds.Get("sgd").Create(new OptimizerRequest([Tensor.From([1f, 2f], [2])], new OptimizerSettings(0.1f)));
        Check(made is Sgd, $"an optimizer made from a request record: {made.GetType().Name}");
    }

    // A kind made from a span of text: proves a request may be a ref struct.
    internal sealed class Greeting(string name, string summary) : IKind<ReadOnlySpan<char>, string>
    {
        public string Name => name;

        public string Summary => summary;

        public IReadOnlySet<string> Keys { get; } = new OptionKeys("punctuation");

        public string Create(ReadOnlySpan<char> request) => $"{name}, {request}";
    }

    // Compares names by their upper-case form, without a span lookup of its own.
    internal sealed class UpperComparer : IEqualityComparer<string>
    {
        public bool Equals(string? x, string? y) => string.Equals(x?.ToUpperInvariant(), y?.ToUpperInvariant(), StringComparison.Ordinal);

        public int GetHashCode(string obj) => obj.ToUpperInvariant().GetHashCode(StringComparison.Ordinal);
    }
}
