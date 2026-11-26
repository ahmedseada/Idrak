// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Cluster;
using Idrak.Abstraction.Testing;
using Idrak.Cli;
using Idrak.Cli.Shared;
using Idrak.Cluster;
using Idrak.Cluster.Abstractions;
using Idrak.Networking.Selection;

// Plan 23, phase C1: Idrak.Cluster's virtual clock and topology builder, Idrak.Network's fastest selector, and the
// cluster init command.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] ClusterSimGroup =
    [
        ("cluster: the virtual clock charges bytes / bandwidth + latency, one transfer at a time each way, a lost message again, and is the same for a seed", ClusterVirtualClock),
        ("cluster: the fastest selector takes the first available, fastest first, and names every reason when none can", ClusterSelector),
        ("cluster: the topology builder asks every question with a description and a default from the machine's facts, takes answers given by key, refuses a bad answer saying why (a cap without a unit), splits a card among its ranks whichever backend names it, and writes a commented file that reads back", ClusterBuilder),
        ("cluster: cli cluster init asked through the test input (in colour with --color always) and with --yes and options writes the same topology; without a terminal the usage error names --yes; a bad option names it", ClusterCliInit),
    ];

    private static void ClusterVirtualClock(Device device)
    {
        if (!FirstRun(nameof(ClusterVirtualClock)))
        {
            return;
        }

        var link = new LinkProfile("test", 1e6, TimeSpan.FromMilliseconds(1));
        var clock = new VirtualClock(2, (_, _) => link, seed: 1);
        var (sent, first) = clock.Send(0, 1, 1000, 0, 0);
        var (_, second) = clock.Send(0, 1, 1000, 0, 1);
        var (_, back) = clock.Send(1, 0, 1000, 0, 0);
        Check(sent == 0 && Math.Abs(first - 0.002) < 1e-12 && Math.Abs(second - 0.003) < 1e-12 && Math.Abs(back - 0.002) < 1e-12,
            $"1,000 bytes at 1 MB/s after 1 ms: arrive at {first}; the next waits for the link: {second}; the other way is free: {back}");
        clock.Arrive(1, second);
        Check(clock.Now(1) == second && clock.Now(0) == 0, "a receive moves the receiver's clock only");
        clock.Synchronize();
        Check(clock.Now(0) == second, "a barrier moves every clock to the latest");

        var noisy = link with { Jitter = TimeSpan.FromMilliseconds(5), LossRate = 0.5 };
        double[] Play(int seed)
        {
            var c = new VirtualClock(2, (_, _) => noisy, seed);
            return [.. Enumerable.Range(0, 50).Select(i => c.Send(0, 1, 1000, i % 3, i).Arrives)];
        }

        Check(Play(7).SequenceEqual(Play(7)) && !Play(7).SequenceEqual(Play(8)), "jitter and loss are the same for a seed, and differ for another");
        Check(Play(7).Zip(Play(7).Skip(1)).Any(p => p.Second - p.First > 0.002), "a lost message costs another transfer");
    }

    private static void ClusterSelector(Device device)
    {
        if (!FirstRun(nameof(ClusterSelector)))
        {
            return;
        }

        var selector = new FastestSelector();
        var topology = ClusterTopology.Create([new ClusterNodeSpec("box", null, [new ClusterDeviceSpec("cpu", null), new ClusterDeviceSpec("cpu", null)])], null, null);
        var (chosen, reason) = selector.Choose(0, 1, topology,
            [new("tcp", true, "a route"), new("sim", false, "not in process"), new("mine", true, "a plug-in"), new("shared-memory", true, "one machine")]);
        Check(chosen == "shared-memory" && reason.Contains("one machine"), $"fastest first: {chosen} ({reason})");
        Check(selector.Choose(0, 1, topology, [new("mine", true, "a plug-in"), new("udp", false, "lossy")]).Transport == "mine", "a plug-in comes after the known ones but before none");
        ThrowsNaming<NotSupportedException>(() => selector.Choose(0, 1, topology, [new("tcp", false, "no route"), new("sim", false, "not in process")]), "none available", "no route", "not in process");
    }

    private static readonly MachineFacts ClusterFacts = new("desk", "192.168.1.10", "eth0", 1e9,
        [new MachineDevice("cpu", DeviceType.Cpu, 32L << 30, "CPU"), new MachineDevice("cuda:0", DeviceType.Cuda, 16L << 30, "a GPU")]);

    private static void ClusterBuilder(Device device)
    {
        if (!FirstRun(nameof(ClusterBuilder)))
        {
            return;
        }

        var builder = new TopologyBuilder(ClusterFacts);
        var asked = new List<TopologyQuestion>();
        string[] answers = ["3", "", "", "mini", "old", "cuda:0,cuda:0", "", "", "", "", "", "6GB", "", "", "", ""];
        Check(builder.Next!.Key == "nodes" && builder.Answer("zero") is { } refused && refused.Contains("number"), "a bad answer is refused saying why");
        foreach (string answer in answers)
        {
            var question = builder.Next ?? throw new Exception($"no question left for '{answer}'; asked: {string.Join(", ", asked.Select(q => q.Key))}");
            asked.Add(question);
            Check(builder.Answer(answer) is null, $"{question.Key} takes '{answer}'");
        }

        Check(builder.Next is null, $"every question answered: {string.Join(", ", asked.Select(q => q.Key))}");
        Check(asked.All(q => q.Description.Length > 20 && (q.Default is not null || q.Suggestion is not null)), "every question has a description and a default or a suggestion");
        var byKey = asked.ToDictionary(q => q.Key);
        Check(byKey["node.1.name"].Default == "desk" && byKey["node.1.address"].Default == "192.168.1.10:29500" && byKey["node.1.devices"].Default == "cuda:0"
              && byKey["node.1.device.1.memory"].Default == "8GB" && byKey["node.2.device.1.memory"].Default == "16GB" && byKey["inter-node"].Default == "ethernet-1g"
              && byKey["inter-node"].Suggestion!.Contains("eth0"),
            "the defaults come from the facts: the host name, its address, the GPU, its memory split between the ranks on it, the adapter's speed");
        Check(!byKey.ContainsKey("intra-node"), "no intra-node question when no node has different devices");
        var topology = builder.Build();
        Check(topology.WorldSize == 4 && topology.Coordinator == "192.168.1.10:29500" && topology.Nodes[2].Devices[0].MemoryBytes == 6L << 30, "the topology the answers describe");
        string json = builder.ToJson();
        Check(json.Contains("// rank 3", StringComparison.Ordinal) && ClusterTopology.Parse(json).ToJson() == topology.ToJson(), "a commented file that reads back");

        var given = new TopologyBuilder(ClusterFacts, new Dictionary<string, string> { ["nodes"] = "1", ["node.1.devices"] = "cpu,cuda:0", ["intra-node"] = "pcie" });
        Check(given.Next!.Key == "node.1.name", "answers given by key skip their questions");
        given.AnswerRest();
        Check(given.Build() is { WorldSize: 2, IntraNode: "pcie" }, "the rest take their defaults");
        ThrowsNaming<ArgumentException>(() => _ = new TopologyBuilder(ClusterFacts, new Dictionary<string, string> { ["nodes"] = "-1" }), "a bad given answer", "nodes");
        ThrowsNaming<ArgumentException>(() => _ = new TopologyBuilder(ClusterFacts, new Dictionary<string, string> { ["gpus"] = "1" }), "an unknown key", "gpus");

        // Ranks on one card share its memory whichever backend each names (vulkan:0 beside cuda:0); a bare number is refused.
        var twins = ClusterFacts with { Devices = [.. ClusterFacts.Devices, new MachineDevice("vulkan:0", DeviceType.Vulkan, 16L << 30, "a GPU (the card cuda:0 drives)", "cuda:0")] };
        var shared = new TopologyBuilder(twins, new Dictionary<string, string> { ["nodes"] = "1", ["node.1.devices"] = "cuda:0,cuda:0,vulkan:0" });
        while (shared.Next is { } q && !q.Key.EndsWith(".memory", StringComparison.Ordinal))
        {
            Check(shared.Answer(q.Default ?? "") is null, $"{q.Key} takes its default");
        }

        string third = ClusterTopology.FormatMemory((16L << 30) / 3 / (1 << 20) * (1 << 20));
        Check(shared.Next is { Default: var split, Suggestion: var why } && split == third && why!.Contains("3 ranks", StringComparison.Ordinal),
            $"three ranks on one card through two backends split it in three: {shared.Next?.Default} ({shared.Next?.Suggestion})");
        Check(shared.Answer("4048") is { } unitless && unitless.Contains("4048MB", StringComparison.Ordinal), "a cap without a unit is refused, naming one");
    }

    private static void ClusterCliInit(Device device)
    {
        if (!FirstRun(nameof(ClusterCliInit)))
        {
            return;
        }

        string folder = CliTemp(), typed = Path.Combine(folder, "typed.json"), options = Path.Combine(folder, "options.json");
        try
        {
            string[] given = ["--nodes", "2", "--names", "desk,mini", "--address", "10.0.0.1:29500", "--devices", "cpu,cpu;cpu", "--memory", "64MB,64MB;64MB", "--inter", "ethernet-1g"];
            var (exit, output, error) = Cli(["cluster", "init", "-y", "-o", options, .. given]);
            Check(exit == 0 && output.Contains("2 rank(s)") == false && output.Contains("3 rank(s)") && File.Exists(options), $"cluster init -y: {output} {error}");
            const string answers = "zero\n2\ndesk\n10.0.0.1:29500\nmini\ncpu,cpu\n64\n64MB\n64MB\ncpu\n64MB\nethernet-1g\n\n\n\n";
            Terminal.TestInput = new StringReader(answers);
            try
            {
                (exit, output, error) = Cli("cluster", "init", "-o", typed);
            }
            finally
            {
                Terminal.TestInput = null;
            }

            Check(exit == 0 && output.Contains("a number of machines") && output.Contains("suggestion:") && output.Contains("with its unit") && !output.Contains('\u001b'),
                $"cluster init asked, refusing a cap without a unit, in no colour off a terminal: {output} {error}");
            Terminal.TestInput = new StringReader(answers);
            try
            {
                (exit, output, error) = Cli("cluster", "init", "-o", Path.Combine(folder, "coloured.json"), "--color", "always");
            }
            finally
            {
                Terminal.TestInput = null;
            }

            Check(exit == 0 && output.Contains("\u001b[2mA node is", StringComparison.Ordinal) && output.Contains("\u001b[32msuggestion:", StringComparison.Ordinal)
                  && output.Contains("\u001b[1mHow many machines", StringComparison.Ordinal) && output.Contains("\u001b[36m[", StringComparison.Ordinal)
                  && output.Contains("\u001b[33m", StringComparison.Ordinal) && output.Contains("\u001b[31m", StringComparison.Ordinal),
                $"in colour the description is dim, the suggestion green, the question bold, its default cyan, the answer yellow, a refusal red: {output.Replace("\u001b", "ESC", StringComparison.Ordinal)}");
            Check(ClusterTopology.Load(typed).ToJson() == ClusterTopology.Load(options).ToJson(), "asked and given write the same topology");
            var (noTerminal, _, refusal) = Cli("cluster", "init", "-o", Path.Combine(folder, "x.json"));
            Check(noTerminal == 2 && refusal.Contains("--yes", StringComparison.Ordinal), $"no terminal: {refusal}");
            var (bad, _, badError) = Cli("cluster", "init", "-y", "-o", Path.Combine(folder, "y.json"), "--nodes", "0");
            Check(bad == 2 && badError.Contains("nodes", StringComparison.Ordinal), $"a bad option: {badError}");
            var dry = Cli(["cluster", "init", "-y", "--dry-run", "-o", Path.Combine(folder, "z.json"), .. given]);
            Check(dry.Exit == 0 && dry.Out.Contains("\"format\"") && !File.Exists(Path.Combine(folder, "z.json")), "--dry-run prints the file");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }
}
