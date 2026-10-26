// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using Idrak;
using Idrak.Abstraction.Cluster;
using Idrak.Abstraction.Operations;
using Idrak.Cluster;
using Idrak.Gpu.Vulkan;

// Phase 3: the ranks of a cluster run rank 0's measured kernel choices (ClusterOptions.ShareTuning, IDRAK_CLUSTER_SHARE_TUNING),
// through the backends' IMeasuredChoicesProvider (MeasuredChoiceProviders): in one process each rank's backend follows rank
// 0's before the ranks' threads start; between processes rank 0 sends what it knows as the ranks join. A rank on another
// device or driver keeps its own and says so. Every cluster here is an in-process one of devices carved from one device.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] ClusterTuningGroup =
    [
        ("cluster tuning: through the contract (a provider of the tests' own on carved CPU devices): every rank runs rank 0's choices, a rank of another device keeps its own and says so (its result and the telemetry), IDRAK_CLUSTER_SHARE_TUNING=0 and ShareTuning off leave each rank its own, the exchange between processes takes rank 0's over the transport, and every follower has its own back after the run; the choices round trip as bytes", ClusterTuningContract),
        ("cluster tuning: Vulkan: a rank whose own measured attention path differs from rank 0's runs rank 0's (its kernels traced) and gives one process's attention bit for bit; with sharing off it runs its own; rank 0's choices sent as between processes are taken without a line written to the cache, and choices of another device or driver are refused", ClusterTuningVulkan),
    ];

    // A provider of the tests' own: "choices" kept per backend, handled for the backends the test gives a table only.
    internal sealed class TestChoices : IMeasuredChoicesProvider
    {
        public sealed class Table(string identity, Dictionary<string, string> own)
        {
            public string Identity { get; } = identity;

            public Dictionary<string, string> Own { get; } = own;

            public Table? Leader { get; set; }

            public IReadOnlyDictionary<string, string>? Imported { get; set; }

            public IReadOnlyDictionary<string, string> Effective => Leader?.Effective ?? Imported ?? Own;
        }

        private int _follows, _imports, _leaves;

        public ConditionalWeakTable<Backend, Table> Tables { get; } = new();

        public int Follows => Volatile.Read(ref _follows);

        public int Imports => Volatile.Read(ref _imports);

        public int Leaves => Volatile.Read(ref _leaves);

        public string Name => "cluster-tuning-test";

        public string Summary => "the cluster tuning tests' choices";

        public bool Handles(Backend backend) => Tables.TryGetValue(backend, out _);

        public MeasuredChoices Export(Backend backend) =>
            new("test", Tables.GetValue(backend, _ => throw new InvalidOperationException()).Identity, backend.Name,
                new Dictionary<string, string>(Tables.GetValue(backend, _ => throw new InvalidOperationException()).Effective));

        public MeasuredChoicesResult Import(Backend backend, MeasuredChoices choices)
        {
            Interlocked.Increment(ref _imports);
            var table = Tables.GetValue(backend, _ => throw new InvalidOperationException());
            if (choices.Identity != table.Identity)
            {
                return new MeasuredChoicesResult(false, 0, $"kept its own: rank 0's are of {choices.Identity}, this is {table.Identity}");
            }

            (table.Leader, table.Imported) = (null, choices.Entries);
            return new MeasuredChoicesResult(true, choices.Entries.Count, "took rank 0's");
        }

        public MeasuredChoicesResult Follow(Backend follower, Backend leader)
        {
            Interlocked.Increment(ref _follows);
            var table = Tables.GetValue(follower, _ => throw new InvalidOperationException());
            if (!Tables.TryGetValue(leader, out var lead) || lead.Identity != table.Identity)
            {
                return new MeasuredChoicesResult(false, 0, $"kept its own: rank 0's are of {lead?.Identity ?? "none"}, this is {table.Identity}");
            }

            table.Leader = lead;
            return new MeasuredChoicesResult(true, lead.Effective.Count, "shares rank 0's");
        }

        public void Leave(Backend follower)
        {
            Interlocked.Increment(ref _leaves);
            var table = Tables.GetValue(follower, _ => throw new InvalidOperationException());
            (table.Leader, table.Imported) = (null, null);
        }
    }

    // Keeps every MeasuredChoicesShared published.
    internal sealed class ChoicesSharedHook : ITelemetryHook
    {
        public List<MeasuredChoicesShared> Events { get; } = [];

        public TelemetryLevel Levels => TelemetryLevel.Network;

        public void OnChoicesShared(in MeasuredChoicesShared e)
        {
            lock (Events)
            {
                Events.Add(e);
            }
        }
    }

    internal static void ClusterTuningContract(Device device)
    {
        if (!FirstRun(nameof(ClusterTuningContract)))
        {
            return;
        }

        _ = device;
        var topology = DpTopology(["cpu", "cpu", "cpu"]);
        var backends = Enumerable.Range(0, 3).Select(r => SimulatedDevices.Carve(Device.Parse("cpu"), null, r).Backend).ToArray();
        var provider = new TestChoices();
        // Rank 1 measured another choice than rank 0 on the same device; rank 2 is another device (its own identity).
        provider.Tables.Add(backends[0], new TestChoices.Table("device A", new() { ["path"] = "spans", ["splits"] = "4" }));
        provider.Tables.Add(backends[1], new TestChoices.Table("device A", new() { ["path"] = "composed", ["splits"] = "2" }));
        provider.Tables.Add(backends[2], new TestChoices.Table("device B", new() { ["path"] = "composed", ["splits"] = "2" }));
        string Run(ClusterRank rank) => string.Join(",", provider.Tables.GetValue(rank.Device.Backend, _ => throw new InvalidOperationException()).Effective
            .OrderBy(p => p.Key, StringComparer.Ordinal).Select(p => $"{p.Key}={p.Value}"));

        MeasuredChoiceProviders.Register(provider);
        var hook = new ChoicesSharedHook();
        try
        {
            Check(MeasuredChoiceProviders.Find("vulkan") is not null && MeasuredChoiceProviders.Find("cuda") is not null && MeasuredChoiceProviders.For(backends[0]) == provider
                  && MeasuredChoiceProviders.For(Device.Cpu.Backend) is null, $"the GPU backends register their providers; a CPU backend has none: {MeasuredChoiceProviders.Describe()}");

            // Sharing on (the default): rank 1 runs rank 0's, rank 2 (another device) its own, each told why.
            Check(new ClusterOptions().ShareTuning, "sharing is on by default");
            MeasuredChoicesResult?[] results;
            string[] seen;
            using (Telemetry.Subscribe(hook))
            {
                var ran = DpRun(topology, DpOptions(), rank => (Seen: Run(rank), rank.SharedChoices));
                (seen, results) = ([.. ran.Select(r => r.Seen)], [.. ran.Select(r => r.SharedChoices)]);
            }

            Check(seen[0] == "path=spans,splits=4" && seen[1] == seen[0], $"rank 1 runs rank 0's choices, not its own: {string.Join(" | ", seen)}");
            Check(seen[2] == "path=composed,splits=2", $"rank 2, on another device, keeps its own: {seen[2]}");
            Check(results[0] is null && results[1] is { Taken: true } && results[2] is { Taken: false } refused && refused.Detail.Contains("device B", StringComparison.Ordinal),
                $"each rank's result says what it runs and why: {string.Join(" | ", results.Select(r => r?.ToString() ?? "null"))}");
            Check(hook.Events.Count == 2 && hook.Events.Any(e => e is { Rank: 1, Taken: true }) && hook.Events.Any(e => e is { Rank: 2, Taken: false } && e.Detail.Contains("kept its own", StringComparison.Ordinal)),
                $"the telemetry tells each rank's outcome: {string.Join(" | ", hook.Events)}");
            Check(provider.Follows == 2 && provider.Leaves == 1 && provider.Tables.GetValue(backends[1], _ => null!).Leader is null,
                $"followed before the ranks ran (rank 0 never follows), given back after: {provider.Follows} follows, {provider.Leaves} leaves");

            // Off: each rank its own, by the option and by its variable.
            var off = DpRun(topology, DpOptions() with { ShareTuning = false }, rank => (Seen: Run(rank), rank.SharedChoices));
            Check(off[1].Seen == "path=composed,splits=2" && off.All(r => r.SharedChoices is null) && provider.Follows == 2,
                $"ShareTuning off: rank 1 runs its own: {string.Join(" | ", off.Select(r => r.Seen))}");
            ClusterOptions? fromVariable = null;
            WithEnvironment(new Dictionary<string, string?> { ["IDRAK_CLUSTER_SHARE_TUNING"] = "0" }, () => fromVariable = ClusterOptions.FromEnvironment(DpOptions()));
            Check(fromVariable is { ShareTuning: false }, "IDRAK_CLUSTER_SHARE_TUNING=0 turns it off");
            var offByVariable = DpRun(topology, fromVariable!, rank => Run(rank));
            Check(offByVariable[1] == "path=composed,splits=2" && provider.Follows == 2, $"IDRAK_CLUSTER_SHARE_TUNING=0: rank 1 runs its own: {offByVariable[1]}");

            // Between processes: rank 0 sends its choices over the transport as the ranks join, each rank takes them.
            var exchanged = DpRun(topology, DpOptions() with { ShareTuning = false }, rank => (Result: ChoiceSharing.Exchange(rank.Transport, rank.Device, DpOptions()), Seen: Run(rank)));
            Check(exchanged[0].Result is null && exchanged[1].Result is { Taken: true } && exchanged[1].Seen == "path=spans,splits=4" && exchanged[2].Result is { Taken: false }
                  && exchanged[2].Seen == "path=composed,splits=2" && provider.Imports == 2,
                $"the exchange between processes: rank 1 takes rank 0's, rank 2 keeps its own: {string.Join(" | ", exchanged.Select(e => $"{e.Result} {e.Seen}"))}");
            provider.Leave(backends[1]);
            var silent = DpRun(topology, DpOptions() with { ShareTuning = false }, rank => ChoiceSharing.Exchange(rank.Transport, rank.Device, DpOptions() with { ShareTuning = false }));
            Check(silent.All(r => r is null) && provider.Imports == 2, "with sharing off rank 0 sends nothing and no rank takes anything (no rank waits either)");

            // The bytes between processes.
            var choices = new MeasuredChoices("vulkan", "vulkan/uuid/driver/v2", "a device", new Dictionary<string, string> { ["width"] = "64", ["Rows 1 2 3 0 0 0 0"] = "1" });
            var back = MeasuredChoices.FromBytes(choices.ToBytes());
            Check(back.Kind == choices.Kind && back.Identity == choices.Identity && back.Device == choices.Device && back.Entries.Count == 2 && back.Entries["Rows 1 2 3 0 0 0 0"] == "1",
                "measured choices round trip as bytes");
            Throws<InvalidDataException>(() => MeasuredChoices.FromBytes("not choices"u8), "bytes that are not measured choices");
            Throws<ArgumentException>(() => new MeasuredChoices("vulkan", "a\tb", "d", new Dictionary<string, string>()).ToBytes(), "a tab in a field");
        }
        finally
        {
            MeasuredChoiceProviders.Unregister(provider.Name);
            foreach (var backend in backends)
            {
                if (provider.Tables.TryGetValue(backend, out var table))
                {
                    (table.Leader, table.Imported) = (null, null);
                }
            }
        }
    }

    internal static void ClusterTuningVulkan(Device device)
    {
        if (device.Backend is not VulkanBackend main)
        {
            Console.WriteLine($"    ({device} is not a Vulkan device: the contract's test covers it)");
            return;
        }

        // A shape no other test meets, so the choices set here reach nothing else: 3 heads of 5 rows, every key seen.
        const int Heads = 3, Rows = 5, Dim = 8;
        var key = new VulkanTuneKey(VulkanTuneOp.AttentionPath, (int)MixedPrecision.Current, Heads, Rows, Rows, Dim, 0);
        var random = new Random(41);
        float[] q = RandomArray(random, Heads * Rows * Dim), k = RandomArray(random, Heads * Rows * Dim), v = RandomArray(random, Heads * Rows * Dim);
        const float Scale = 0.35f;
        var spans = KeySpans.Bidirectional(Rows, Rows);

        // The attention on `on` as a layer runs it (the path its backend chose), and whether the key ranges ran.
        (float[] Values, bool Spans, bool Composed) Attention(Device on)
        {
            using var tq = Tensor.From(q, [Heads, Rows, Dim], on);
            using var tk = Tensor.From(k, [Heads, Rows, Dim], on);
            using var tv = Tensor.From(v, [Heads, Rows, Dim], on);
            var (starts, ends) = spans.ToTensors(on);
            using (starts)
            using (ends)
            using (var trace = Kernels.Trace(on.Backend))
            using (var y = Tensor.AttentionFastest(tq, tk, tv, starts, ends, Scale, everyKey: true))
            {
                return (y.ToArray(), trace.Calls(Ops.AttentionSpans) > 0, trace.Calls(Ops.AttentionSpans) == 0 && trace.Calls(Ops.ScaleMaskSoftmax) > 0);
            }
        }

        var topology = DpTopology(device, 2);
        var ranks = Enumerable.Range(0, 2).Select(r => (VulkanBackend)SimulatedDevices.Carve(device, null, r).Backend).ToArray();
        string? savedCache = VulkanBackend.TuningCacheFile;
        var savedForced = AttentionPaths.Forced;
        string cache = Path.Combine(Path.GetTempPath(), $"idrak-cluster-tuning-{Environment.ProcessId}.tsv");
        try
        {
            // One process: the key ranges (measured as faster, say); the composed scores for comparison.
            AttentionPaths.Forced = AttentionPath.Measured;
            main.SetTunedForTests(key, 0);
            var alone = Attention(device);
            AttentionPaths.Forced = AttentionPath.Composed;
            var composedAlone = Attention(device);
            AttentionPaths.Forced = AttentionPath.Measured;
            bool composes = composedAlone.Composed;                       // the device reports the memory the composed path needs
            bool differs = !SameBits(alone.Values, composedAlone.Values);
            Console.WriteLine($"    the two paths on {device}: {(composes ? "both run" : "the composed one cannot run here")}; {(differs ? $"they differ by up to {Gap(alone.Values, composedAlone.Values):E2}" : "the same bits")}");

            // Rank 0 measured the key ranges, rank 1 the composed scores (its own choice, as a rank measuring alone could).
            ranks[0].SetTunedForTests(key, 0);
            ranks[1].SetTunedForTests(key, 1);
            var shared = DpRun(topology, DpOptions(), rank =>
            {
                var backend = (VulkanBackend)rank.Device.Backend;
                bool known = backend.TryTunedForTests(key, out int choice);
                return (Known: known, Choice: choice, Run: Attention(rank.Device), rank.SharedChoices, Follows: backend.Leader is not null);
            });
            Check(shared.All(r => r.Known && r.Choice == 0) && shared[1].Follows && shared[1].SharedChoices is { Taken: true },
                $"every rank runs rank 0's attention path: {string.Join(" | ", shared.Select(r => $"{r.Choice} {r.SharedChoices}"))}");
            Check(shared.All(r => SameBits(r.Run.Values, alone.Values)), $"every rank's attention is one process's bit for bit: gaps {string.Join(", ", shared.Select(r => Gap(r.Run.Values, alone.Values)))}");
            if (composes)
            {
                Check(shared.All(r => r.Run.Spans && !r.Run.Composed), "every rank ran the key ranges (traced), rank 1 too");
            }

            Check(ranks[1].Leader is null, "after the run rank 1 has its own table back");

            // Sharing off: rank 1 runs its own choice.
            ranks[1].SetTunedForTests(key, 1);
            var own = DpRun(topology, DpOptions() with { ShareTuning = false }, rank =>
            {
                ((VulkanBackend)rank.Device.Backend).TryTunedForTests(key, out int choice);
                return (Choice: choice, Run: Attention(rank.Device), rank.SharedChoices);
            });
            Check(own[0].Choice == 0 && own[1].Choice == 1 && own.All(r => r.SharedChoices is null), $"sharing off: each rank its own path: {own[0].Choice}, {own[1].Choice}");
            if (composes)
            {
                Check(own[1].Run.Composed && SameBits(own[1].Run.Values, composedAlone.Values), "sharing off: rank 1 ran the composed scores");
                Check(!differs || !SameBits(own[1].Run.Values, alone.Values), "sharing off: rank 1's attention is not one process's (the test sees the difference sharing removes)");
            }

            // As between processes: rank 0's choices sent over the transport and taken, nothing written to the cache file.
            VulkanBackend.TuningCacheFile = cache;
            File.Delete(cache);
            ranks[1].SetTunedForTests(key, 1);
            var exchanged = DpRun(topology, DpOptions() with { ShareTuning = false }, rank =>
            {
                var result = ChoiceSharing.Exchange(rank.Transport, rank.Device, DpOptions());
                var backend = (VulkanBackend)rank.Device.Backend;
                backend.TryTunedForTests(key, out int choice);
                return (Result: result, Choice: choice, backend.Width, Imported: backend.ImportedChoices);
            });
            Check(exchanged[1].Result is { Taken: true, Choices: > 0 } && exchanged[1].Choice == 0 && exchanged[1].Imported && exchanged[1].Width == exchanged[0].Width,
                $"rank 1 takes rank 0's choices and width sent as between processes: {exchanged[1].Result}, choice {exchanged[1].Choice}, width {exchanged[1].Width} (rank 0's {exchanged[0].Width})");
            Check(!File.Exists(cache) || !File.ReadLines(cache).Any(l => l.StartsWith("kernels/", StringComparison.Ordinal)), $"taking them wrote nothing to the cache file {cache}");

            // Another device or driver: refused, its own kept.
            var provider = MeasuredChoiceProviders.Find("vulkan")!;
            var export = provider.Export(ranks[0]);
            ranks[1].LeaveChoices();
            ranks[1].SetTunedForTests(key, 1);
            var foreign = provider.Import(ranks[1], export with { Identity = export.Identity.Replace("vulkan/", "vulkan/another-", StringComparison.Ordinal) });
            Check(!foreign.Taken && foreign.Detail.Contains("another device or driver", StringComparison.Ordinal) && ranks[1].TryTunedForTests(key, out int kept) && kept == 1,
                $"choices of another device or driver are refused, the rank's own kept: {foreign}");
            var otherKind = provider.Follow(ranks[1], Device.Cpu.Backend);
            Check(!otherKind.Taken && otherKind.Detail.Contains("another kind of device", StringComparison.Ordinal), $"rank 0 on another kind of device: {otherKind}");
        }
        finally
        {
            ranks[1].LeaveChoices();
            VulkanBackend.TuningCacheFile = savedCache;
            AttentionPaths.Forced = savedForced;
            File.Delete(cache);
            main.ForgetTunedForTests(key);
            foreach (var backend in ranks)
            {
                backend.ResetTuningForTests();                             // read again from the cache file (not this test's) when next needed
            }
        }
    }
}
