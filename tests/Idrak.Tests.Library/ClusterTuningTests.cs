// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Abstraction.Cluster;
using Idrak.Abstraction.Operations;
using Idrak.Cluster;
using Idrak.Gpu.Vulkan;

// Phase 3: the ranks of a cluster run rank 0's measured kernel choices (ClusterOptions.ShareTuning, IDRAK_CLUSTER_SHARE_TUNING):
// in one process each rank's backend follows rank 0's before the ranks' threads start; between processes rank 0 sends
// what it knows as the ranks join. A rank on another device or driver keeps its own and says so. Every cluster here is
// an in-process one.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] ClusterTuningGroup =
    [
        ("cluster tuning: sharing is on by default; IDRAK_CLUSTER_SHARE_TUNING=0 and ShareTuning off leave each rank its own, and with sharing off the exchange between processes sends nothing and no rank takes anything; the choices round trip as bytes", ClusterTuningContract),
    ];

    internal static void ClusterTuningContract(Device device)
    {
        if (!FirstRun(nameof(ClusterTuningContract)))
        {
            return;
        }

        _ = device;
        var topology = DpTopology(["cpu", "cpu", "cpu"]);
        Check(new ClusterOptions().ShareTuning, "sharing is on by default");

        // Off: each rank its own, by the option and by its variable.
        var off = DpRun(topology, DpOptions() with { ShareTuning = false }, rank => rank.SharedChoices);
        Check(off.All(r => r is null), $"ShareTuning off: each rank runs its own: {string.Join(" | ", off.Select(r => r?.ToString() ?? "null"))}");
        ClusterOptions? fromVariable = null;
        WithEnvironment(new Dictionary<string, string?> { ["IDRAK_CLUSTER_SHARE_TUNING"] = "0" }, () => fromVariable = ClusterOptions.FromEnvironment(DpOptions()));
        Check(fromVariable is { ShareTuning: false }, "IDRAK_CLUSTER_SHARE_TUNING=0 turns it off");

        // Between processes, with sharing off: rank 0 sends nothing and no rank takes anything.
        var silent = DpRun(topology, DpOptions() with { ShareTuning = false }, rank => ChoiceSharing.Exchange(rank.Transport, rank.Device, DpOptions() with { ShareTuning = false }));
        Check(silent.All(r => r is null), "with sharing off rank 0 sends nothing and no rank takes anything (no rank waits either)");

        // The bytes between processes.
        var choices = new MeasuredChoices("vulkan", "vulkan/uuid/driver/v2", "a device", new Dictionary<string, string> { ["width"] = "64", ["Rows 1 2 3 0 0 0 0"] = "1" });
        var back = MeasuredChoices.FromBytes(choices.ToBytes());
        Check(back.Kind == choices.Kind && back.Identity == choices.Identity && back.Device == choices.Device && back.Entries.Count == 2 && back.Entries["Rows 1 2 3 0 0 0 0"] == "1",
            "measured choices round trip as bytes");
        Throws<InvalidDataException>(() => MeasuredChoices.FromBytes("not choices"u8), "bytes that are not measured choices");
        Throws<ArgumentException>(() => new MeasuredChoices("vulkan", "a\tb", "d", new Dictionary<string, string>()).ToBytes(), "a tab in a field");
    }
}
