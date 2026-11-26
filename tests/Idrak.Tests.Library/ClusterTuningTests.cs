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

    // The runs here keep the ranks' carved devices (ReleaseDevices off): the tests set choices on them before the runs and
    // read them across runs, then release them at the end.
    private static ClusterOptions KeepingDevices(ClusterOptions options) => options with { Simulation = options.Simulation with { ReleaseDevices = false } };

    internal static void ClusterTuningContract(Device device)
    {
        if (!FirstRun(nameof(ClusterTuningContract)))
        {
            return;
        }

        _ = device;
        var topology = DpTopology(["cpu", "cpu", "cpu"]);
        var carved = Enumerable.Range(0, 3).Select(r => SimulatedDevices.Carve(Device.Parse("cpu"), null, r)).ToArray();
        var backends = carved.Select(d => d.Backend).ToArray();
        ClusterOptions Options() => KeepingDevices(DpOptions());
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
                var ran = DpRun(topology, Options(), rank => (Seen: Run(rank), rank.SharedChoices));
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
            var off = DpRun(topology, Options() with { ShareTuning = false }, rank => (Seen: Run(rank), rank.SharedChoices));
            Check(off[1].Seen == "path=composed,splits=2" && off.All(r => r.SharedChoices is null) && provider.Follows == 2,
                $"ShareTuning off: rank 1 runs its own: {string.Join(" | ", off.Select(r => r.Seen))}");
            ClusterOptions? fromVariable = null;
            WithEnvironment(new Dictionary<string, string?> { ["IDRAK_CLUSTER_SHARE_TUNING"] = "0" }, () => fromVariable = ClusterOptions.FromEnvironment(Options()));
            Check(fromVariable is { ShareTuning: false }, "IDRAK_CLUSTER_SHARE_TUNING=0 turns it off");
            var offByVariable = DpRun(topology, fromVariable!, rank => Run(rank));
            Check(offByVariable[1] == "path=composed,splits=2" && provider.Follows == 2, $"IDRAK_CLUSTER_SHARE_TUNING=0: rank 1 runs its own: {offByVariable[1]}");

            // Between processes: rank 0 sends its choices over the transport as the ranks join, each rank takes them.
            var exchanged = DpRun(topology, Options() with { ShareTuning = false }, rank => (Result: ChoiceSharing.Exchange(rank.Transport, rank.Device, DpOptions()), Seen: Run(rank)));
            Check(exchanged[0].Result is null && exchanged[1].Result is { Taken: true } && exchanged[1].Seen == "path=spans,splits=4" && exchanged[2].Result is { Taken: false }
                  && exchanged[2].Seen == "path=composed,splits=2" && provider.Imports == 2,
                $"the exchange between processes: rank 1 takes rank 0's, rank 2 keeps its own: {string.Join(" | ", exchanged.Select(e => $"{e.Result} {e.Seen}"))}");
            provider.Leave(backends[1]);
            var silent = DpRun(topology, Options() with { ShareTuning = false }, rank => ChoiceSharing.Exchange(rank.Transport, rank.Device, DpOptions() with { ShareTuning = false }));
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

            foreach (var rank in carved)
            {
                SimulatedDevices.Release(rank);
            }
        }
    }

}
