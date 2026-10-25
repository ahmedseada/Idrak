// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Generation;
using Idrak.Abstraction.Training;

// Plan 33, C0: the shared kind contract (IKind, IConfigurableKind, IKind<TRequest, T>, OptionKeys), KindTable, and
// SlotTable's lookup of string names by span.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] KindTableGroup =
    [
        ("slot table span lookup: a table of string names finds by span what it finds by string, in both layers and under the comparer's case rule, and allocates nothing once warm", SlotTableSpanLookup),
        ("slot table span lookup: a comparer that cannot take spans still finds the same entries (through a string), and a table of value-type keys refuses a span", SlotTableSpanFallback),
        ("kind table: kinds register by their name (one or several at once), are found by span, list names and summaries in order, and an unknown name is refused as every registry words it", KindTableRegistry),
        ("kind table: option keys keep their declared order for listings and are checked by span; a kind's request may be a ref struct; the library's kind registries find their kinds by span", KindTableKeysAndRegistries),
    ];

    private static void SlotTableSpanLookup(Device device)
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

    private static void SlotTableSpanFallback(Device device)
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

    private static void KindTableRegistry(Device device)
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
        Check(ReferenceEquals(table.Find("beta".AsSpan()), mine) && ReferenceEquals(table.Default("beta"), beta) && table.Origin("beta") == "Idrak.Tests"
              && table.Names.SequenceEqual(["alpha", "Beta"]) && string.Equals(table.Slots.Report().Single().Name, "beta", StringComparison.OrdinalIgnoreCase),
            $"an app's kind under a library name shadows it, in its place, and is reported: names {string.Join(", ", table.Names)}, origin {table.Origin("beta")}, "
            + $"report {string.Join("; ", table.Slots.Report())}");
        Check(table.Describe() == "alpha: the first\nBeta: the app's second" && table.Listed() == "alpha, Beta",
            $"names and summaries listed in order: {table.Describe()}");
        Check(table.Unregister("BETA") && ReferenceEquals(table.Find("beta".AsSpan()), beta), "unregistering brings the library's kind back");

        // The name a table lists is spelled as its last registration or removal spelled it (the table ignores case).
        ThrowsNaming<NotSupportedException>(() => table.Get("gamma".AsSpan()), "an unknown kind",
            $"Test kind 'gamma' is not registered (registered: {table.Listed()}); add it with TestKinds.Register.", "registered: alpha, ");
        ThrowsNaming<ArgumentException>(() => table.Register(new Greeting(" ", "nameless")), "a kind without a name", "kind");
        var empty = new KindTable<Greeting>("TestEmpty", "Test kind", "add one", unguarded: "test");
        Check(empty.Listed() == "none" && empty.NotRegistered("x") == "Test kind 'x' is not registered (registered: none); add one.", empty.NotRegistered("x"));
    }

    private static void KindTableKeysAndRegistries(Device device)
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
    private sealed class Greeting(string name, string summary) : IKind<ReadOnlySpan<char>, string>
    {
        public string Name => name;

        public string Summary => summary;

        public IReadOnlySet<string> Keys { get; } = new OptionKeys("punctuation");

        public string Create(ReadOnlySpan<char> request) => $"{name}, {request}";
    }

    // Compares names by their upper-case form, without a span lookup of its own.
    private sealed class UpperComparer : IEqualityComparer<string>
    {
        public bool Equals(string? x, string? y) => string.Equals(x?.ToUpperInvariant(), y?.ToUpperInvariant(), StringComparison.Ordinal);

        public int GetHashCode(string obj) => obj.ToUpperInvariant().GetHashCode(StringComparison.Ordinal);
    }
}
