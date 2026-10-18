// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Cluster;
using Idrak.Abstraction.Testing;
using Idrak.Cli;
using Idrak.Cli.Shared;
using Idrak.Cluster;

// Plan 23, phase C1: Idrak.Cluster (carved devices, the sim transport, the virtual clock, the in-process cluster, the
// built-in links, the builder, the cluster probe) and Idrak.Network (naive and ring, the fastest selector, the network
// probe), and the cluster init and inspect commands.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ClusterSimGroup =
    [
        ("cluster: the packages register sim devices, the sim transport, the built-in links, the none model, naive and ring, the fastest selector and both probes; carved devices are never listed; Idrak.Cluster and Idrak.Network reference Idrak.Abstraction only", ClusterRegistrations),
        ("cluster: a carved device is its physical device's type and kernels with its own pool: past its cap an allocation fails naming it; the same carving gives the same device; a carved device is not carved again", ClusterCarvedDevice),
        ("cluster: the virtual clock charges bytes / bandwidth + latency, one transfer at a time each way, a lost message again, and is the same for a seed", ClusterVirtualClock),
        ("cluster: the sim transport passes the kit's TransportSuite", ClusterSimTransportSuite),
        ("cluster: ring and naive pass the kit's CollectivesSuite over sim on devices carved from the device", ClusterSimCollectivesSuite),
        ("cluster: an in-process ring all-reduce agrees with naive (exactly on whole numbers), every rank and rerun identical, and its simulated time is the clock's played prediction and the same on every run", ClusterInProcessRing),
        ("cluster: a bounded pipe makes a fast rank wait for its peer; a ring all-reduce publishes nothing while no hook listens, one CollectiveCompleted per rank with its bytes while one does, and its device memory does not grow over 100 calls", ClusterPipeTelemetryMemory),
        ("cluster: a rank that dies leaves its peers an error naming it at once, not a timeout", ClusterRankDies),
        ("cluster: the built-in links carry a rate only where the name states it, measure same-device, shared-memory and pcie here, refuse to predict the rest; the kit's LinkProfileSuite passes them all", ClusterBuiltInLinks),
        ("cluster: the fastest selector takes the first available, fastest first, and names every reason when none can", ClusterSelector),
        ("cluster: the network and cluster probes record a simulated run's collectives, transfers and network time through Profiler.Run", ClusterProbes),
        ("cluster: the topology builder asks every question with a description and a default from the machine's facts, takes answers given by key, refuses a bad answer saying why, and writes a commented file that reads back", ClusterBuilder),
        ("cluster: cli cluster init asked through the test input and with --yes and options writes the same topology; without a terminal the usage error names --yes; a bad option names it", ClusterCliInit),
        ("cluster: cli cluster inspect lists the ranks, the links with their figures and the predicted all-reduce times, in text and JSON; a bad file names the key", ClusterCliInspect),
    ];

    // A group of `world` sim ranks on one node, every rank naming `device` (so every pair is same-device).
    private static (IReadOnlyList<ITransport> Transports, ClusterTopology Topology, ClusterOptions Options) SimGroup(int world, TimeSpan timeout, string device = "cpu", int inFlight = 2)
    {
        var topology = ClusterTopology.Create([new ClusterNodeSpec("box", null, [.. Enumerable.Repeat(new ClusterDeviceSpec(device, null), world)])], null, null);
        var options = new ClusterOptions { Coordinator = "sim://test-" + Guid.NewGuid().ToString("N"), Timeout = timeout, InFlight = inFlight };
        var provider = Transports.Get("sim");
        var connecting = Enumerable.Range(0, world).Select(r => Task.Factory.StartNew(() => provider.Connect(new ClusterContext(topology, r, options)), TaskCreationOptions.LongRunning)).ToArray();
        Task.WaitAll(connecting);
        return ([.. connecting.Select(t => t.Result)], topology, options);
    }

    private static IReadOnlyList<ICollectives> SimCollectives(string algorithm, int world, TimeSpan timeout, Device device)
    {
        var (transports, topology, options) = SimGroup(world, timeout, device.ToString());
        var provider = Collectives.Get(algorithm);
        return [.. transports.Select((t, r) => provider.Create(t, SimulatedDevices.Carve(device, null, r), new ClusterContext(topology, r, options)))];
    }

    private static void ClusterRegistrations(Device device)
    {
        if (!FirstRun(nameof(ClusterRegistrations)))
        {
            return;
        }

        Check(DeviceProviders.Find("sim") is not null && Transports.Find("sim") is not null && FailureModels.Find("none") is not null, "sim devices, the sim transport and the none model");
        Check(Collectives.Find("naive") is not null && Collectives.Find("ring") is not null && TransportSelectors.Find("fastest") is not null, "naive, ring and fastest");
        Check(ProfileProbes.Names.Contains("network") && ProfileProbes.Names.Contains("cluster"), $"the probes: {string.Join(", ", ProfileProbes.Names)}");
        foreach (var assembly in new[] { typeof(SimulatedDevices).Assembly, Collectives.Get("ring").GetType().Assembly })
        {
            var idrak = assembly.GetReferencedAssemblies().Select(a => a.Name!).Where(n => n.StartsWith("Idrak", StringComparison.Ordinal)).ToArray();
            Check(idrak is ["Idrak.Abstraction"], $"{assembly.GetName().Name} references {string.Join(", ", idrak)}; plan 23 allows Idrak.Abstraction only");
        }

        foreach (string name in new[] { "ethernet-1g", "ethernet-2.5g", "ethernet-10g", "ethernet-100g", "infiniband", "roce", "nvlink", "wifi", "direct-link", "same-device", "shared-memory", "pcie" })
        {
            Check(LinkProfiles.Find(name) is not null && LinkProfiles.Origin(name) == Overrides.Library, $"the built-in link {name}");
        }

        _ = SimulatedDevices.Carve(Device.Cpu, 1L << 20, 0);
        Check(Device.Available.All(d => d.Kind != SimulatedDevices.Kind), "carved devices are never listed");
    }

    private static void ClusterCarvedDevice(Device device)
    {
        long cap = 8L << 20;
        var carved = SimulatedDevices.Carve(device, cap, slot: 0);
        Check(carved.Kind == SimulatedDevices.Kind && carved.Type == device.Type && carved.Backend.Kind == device.Backend.Kind && !ReferenceEquals(carved.Backend, device.Backend),
            $"{carved} is a {carved.Type} device running {carved.Backend.Kind} kernels on a backend of its own");
        Check(ReferenceEquals(SimulatedDevices.Carve(device, cap, 0), carved) && !ReferenceEquals(SimulatedDevices.Carve(device, cap, 1), carved)
              && SimulatedDevices.PhysicalOf(carved) == device && SimulatedDevices.CapOf(carved) == cap, "the same carving gives the same device; another slot another");
        ThrowsNaming<ArgumentException>(() => SimulatedDevices.Carve(carved, cap, 0), "carving a carved device", "carved");
        var backend = carved.Backend;
        var half = backend.Allocate((int)(cap / 2 / sizeof(float)), zeroed: true);
        try
        {
            Check(backend.GetMemoryUsage().InUse <= cap, $"half the cap fits: {backend.GetMemoryUsage()}");
            ThrowsNaming<ResourceLimitExceededException>(() => backend.Allocate((int)(cap / sizeof(float)), zeroed: false).Release(), "past the cap", carved.ToString());
        }
        finally
        {
            half.Release();
        }

        using var scope = new TensorScope();
        var a = Tensor.From([1f, 2, 3, 4, 5, 6], [2, 3], carved);
        var b = Tensor.From([1f, 0, 0, 1, 1, 1], [3, 2], carved);
        Check(a.MatMul(b).ToArray().SequenceEqual([4f, 5, 10, 11]), "a carved device runs its physical device's kernels");
        if (device.Type != DeviceType.Cpu)
        {
            Check(DeviceProviders.Find("sim")!.DeviceUuid(carved.Ordinal) == DeviceProviders.Find(device.Kind)!.DeviceUuid(device.Ordinal), "a carved device reports its card's UUID");
        }
    }

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

    private static void ClusterSimTransportSuite(Device device)
    {
        if (!FirstRun(nameof(ClusterSimTransportSuite)))
        {
            return;
        }

        Conformance.CheckTransport(new TransportUnderTest("sim", (w, t) => SimGroup(w, t).Transports), new ContractCheckOptions { RandomCases = 2 }).ThrowIfFailed();
    }

    private static void ClusterSimCollectivesSuite(Device device)
    {
        foreach (string algorithm in new[] { "ring", "naive" })
        {
            Conformance.CheckCollectives(new CollectivesUnderTest(algorithm, (w, t) => SimCollectives(algorithm, w, t, device)), new ContractCheckOptions { RandomCases = 2 }).ThrowIfFailed();
        }
    }

    private static void ClusterInProcessRing(Device device)
    {
        if (!FirstRun(nameof(ClusterInProcessRing)))
        {
            return;
        }

        var topology = ClusterTopology.Create(
            [new ClusterNodeSpec("desk", null, [new ClusterDeviceSpec("cpu", 64L << 20), new ClusterDeviceSpec("cpu", 64L << 20)]), new ClusterNodeSpec("mini", null, [new ClusterDeviceSpec("cpu", 64L << 20)])],
            null, "ethernet-1g");
        const int count = 3 * 4096;
        float[][] Run(string algorithm, bool whole, out TimeSpan[] simulated)
        {
            var results = new float[3][];
            var run = InProcessCluster.Run(topology, new ClusterOptions { Collective = algorithm }, r =>
            {
                var random = new Random(100 + r.Rank);
                var values = Enumerable.Range(0, count).Select(_ => whole ? random.Next(-8, 9) : (float)(random.NextDouble() - 0.5)).ToArray();
                var storage = r.Device.Backend.Allocate(count, zeroed: false);
                try
                {
                    r.Device.Backend.Upload(values, storage);
                    r.Collectives.AllReduce(storage, count, ReduceOp.Sum);
                    results[r.Rank] = new float[count];
                    r.Device.Backend.Download(storage, results[r.Rank]);
                }
                finally
                {
                    storage.Release();
                }
            });
            run.ThrowIfFailed();
            simulated = [.. run.Simulated];
            return results;
        }

        var ring = Run("ring", whole: false, out var ringTime);
        var again = Run("ring", whole: false, out var againTime);
        var naive = Run("naive", whole: false, out _);
        Check(ring.All(r => r.AsSpan().SequenceEqual(ring[0])) && again[0].AsSpan().SequenceEqual(ring[0]), "every rank and a rerun hold identical bytes");
        Check(ring[0].Zip(naive[0]).All(p => Math.Abs(p.First - p.Second) <= 1e-6f), "ring agrees with naive within float rounding");
        Check(Run("ring", whole: true, out _)[0].AsSpan().SequenceEqual(Run("naive", whole: true, out _)[0]), "exactly on whole numbers");
        Check(ringTime.SequenceEqual(againTime) && ringTime.Max() > TimeSpan.Zero, $"the simulated time is the same on every run: {string.Join(", ", ringTime)}");
        var predicted = TopologyInspection.PredictAllReduce(topology, "ring", count * sizeof(float));
        Check(Math.Abs((ringTime.Max() - predicted).TotalSeconds) <= 1e-9 * Math.Max(1, predicted.TotalSeconds),
            $"the ring's simulated time {ringTime.Max().TotalMilliseconds} ms is the clock's played prediction {predicted.TotalMilliseconds} ms");
    }

    private static void ClusterPipeTelemetryMemory(Device device)
    {
        if (FirstRun(nameof(ClusterPipeTelemetryMemory)))
        {
            var (transports, _, _) = SimGroup(2, TimeSpan.FromSeconds(60), inFlight: 1);
            try
            {
                var sending = Task.Factory.StartNew(() =>
                {
                    for (int i = 0; i < 3; i++)
                    {
                        transports[0].Send(1, [(byte)i], 0);
                    }
                }, TaskCreationOptions.LongRunning);
                Check(!sending.Wait(TimeSpan.FromMilliseconds(300)), "with one message in flight, a rank sending three waits for its peer");
                var got = new byte[1];
                for (int i = 0; i < 3; i++)
                {
                    Check(transports[1].Receive(0, got, 0) == 1 && got[0] == i, $"message {i} arrives in order");
                }

                Check(sending.Wait(TimeSpan.FromSeconds(60)), "and goes on once its peer receives");
            }
            finally
            {
                foreach (var t in transports)
                {
                    t.Dispose();
                }
            }
        }

        // A ring all-reduce on devices carved from this one: quiet without a hook, one event per rank with one, flat memory.
        const int world = 3, count = (3 * 1000) + 1;
        string name = device.ToString();
        var topology = ClusterTopology.Create([new ClusterNodeSpec("box", null, [.. Enumerable.Repeat(new ClusterDeviceSpec(name, 64L << 20), world)])], null, null);
        var options = new ClusterOptions { Collective = "ring", Timeout = TimeSpan.FromSeconds(120) };
        var held = new long[world, 2];
        Check(!Telemetry.IsEnabled(TelemetryLevel.Network), "no hook listens at the network level");
        var run = InProcessCluster.Run(topology, options, r =>
        {
            var storage = r.Device.Backend.Allocate(count, zeroed: true);
            try
            {
                r.Collectives.AllReduce(storage, count, ReduceOp.Sum);                 // warm
                held[r.Rank, 0] = r.Device.Backend.GetMemoryUsage().InUse;
                for (int i = 0; i < 100; i++)
                {
                    r.Collectives.AllReduce(storage, count, ReduceOp.Sum);
                }

                held[r.Rank, 1] = r.Device.Backend.GetMemoryUsage().InUse;
            }
            finally
            {
                storage.Release();
            }
        });
        run.ThrowIfFailed();
        for (int r = 0; r < world; r++)
        {
            Check(held[r, 1] == held[r, 0], $"rank {r}'s device memory over 100 all-reduces: {held[r, 0]} B, then {held[r, 1]} B");
        }

        var metrics = new MetricsRecorder(TelemetryLevel.Network);
        using (Telemetry.Subscribe(metrics))
        {
            InProcessCluster.Run(topology, options, r =>
            {
                var storage = r.Device.Backend.Allocate(count, zeroed: true);
                try
                {
                    r.Collectives.AllReduce(storage, count, ReduceOp.Sum);
                }
                finally
                {
                    storage.Release();
                }
            }).ThrowIfFailed();
        }

        // Each rank sends 2·(N−1) slices of 1,000 or 1,001 values, as float32.
        var events = metrics.Collectives;
        long least = 2L * (world - 1) * (count / world) * sizeof(float), most = 2L * (world - 1) * ((count / world) + 1) * sizeof(float);
        Check(events.Count == world && events.All(e => e is { Collective: "ring", Operation: "all-reduce", WorldSize: world } && e.BytesSent >= least && e.BytesSent <= most)
              && events.Select(e => e.Rank).Order().SequenceEqual(Enumerable.Range(0, world)) && events.Sum(e => e.BytesSent) == events.Sum(e => e.BytesReceived),
            $"one event per rank with its bytes: {string.Join("; ", events.Select(e => $"rank {e.Rank} sent {e.BytesSent} received {e.BytesReceived}"))}");
    }

    private static void ClusterRankDies(Device device)
    {
        if (!FirstRun(nameof(ClusterRankDies)))
        {
            return;
        }

        var topology = ClusterTopology.Create([new ClusterNodeSpec("box", null, [new ClusterDeviceSpec("cpu", null), new ClusterDeviceSpec("cpu", null)])], null, null);
        long start = Stopwatch.GetTimestamp();
        var run = InProcessCluster.Run(topology, new ClusterOptions { Timeout = TimeSpan.FromSeconds(60) }, r =>
        {
            if (r.Rank == 1)
            {
                throw new InvalidOperationException("rank 1 fails on purpose");
            }

            var storage = r.Device.Backend.Allocate(16, zeroed: true);
            try
            {
                r.Collectives.AllReduce(storage, 16, ReduceOp.Sum);
            }
            finally
            {
                storage.Release();
            }
        });
        var took = Stopwatch.GetElapsedTime(start);
        Check(!run.Succeeded && run.Errors[1]!.Message.Contains("on purpose") && run.Errors[0]?.Message.Contains("rank 1 left", StringComparison.Ordinal) == true && took < TimeSpan.FromSeconds(30),
            $"rank 0 learns at once that rank 1 left ({took.TotalSeconds:F1} s): {run.Errors[0]?.Message}");
        ThrowsNaming<AggregateException>(run.ThrowIfFailed, "a failed run", "rank 1", "rank 0");
    }

    private static void ClusterBuiltInLinks(Device device)
    {
        if (!FirstRun(nameof(ClusterBuiltInLinks)))
        {
            return;
        }

        Check(LinkProfiles.Find("ethernet-1g")!.BandwidthBytesPerSecond == 125e6 && LinkProfiles.Find("ethernet-100g")!.BandwidthBytesPerSecond == 12.5e9
              && LinkProfiles.Find("infiniband")!.BandwidthBytesPerSecond == 0 && LinkProfiles.Resolve("infiniband-200g").BandwidthBytesPerSecond == 25e9,
            "a rate only where the name states it");
        foreach (string name in new[] { "same-device", "shared-memory", "pcie" })
        {
            var measured = ((MeasuredLink)LinkProfiles.Find(name)!).On(Device.Cpu);
            Check(measured.BandwidthBytesPerSecond > 0 && measured.Latency >= TimeSpan.Zero && ReferenceEquals(measured, ((MeasuredLink)LinkProfiles.Find(name)!).On(Device.Cpu)),
                $"{name} measured here once: {measured.BandwidthBytesPerSecond / 1e9:F2} GB/s");
            Conformance.CheckLinkProfile(measured).ThrowIfFailed();
        }

        foreach (var profile in LinkProfiles.All)
        {
            Conformance.CheckLinkProfile(profile).ThrowIfFailed();
        }
    }

    private static void ClusterSelector(Device device)
    {
        if (!FirstRun(nameof(ClusterSelector)))
        {
            return;
        }

        var selector = TransportSelectors.Get("fastest");
        var topology = ClusterTopology.Create([new ClusterNodeSpec("box", null, [new ClusterDeviceSpec("cpu", null), new ClusterDeviceSpec("cpu", null)])], null, null);
        var (chosen, reason) = selector.Choose(0, 1, topology,
            [new("tcp", true, "a route"), new("sim", false, "not in process"), new("mine", true, "a plug-in"), new("shared-memory", true, "one machine")]);
        Check(chosen == "shared-memory" && reason.Contains("one machine"), $"fastest first: {chosen} ({reason})");
        Check(selector.Choose(0, 1, topology, [new("mine", true, "a plug-in"), new("udp", false, "lossy")]).Transport == "mine", "a plug-in comes after the known ones but before none");
        ThrowsNaming<NotSupportedException>(() => selector.Choose(0, 1, topology, [new("tcp", false, "no route"), new("sim", false, "not in process")]), "none available", "no route", "not in process");
    }

    private static void ClusterProbes(Device device)
    {
        if (!FirstRun(nameof(ClusterProbes)))
        {
            return;
        }

        var topology = ClusterTopology.Create([new ClusterNodeSpec("box", null, [new ClusterDeviceSpec("cpu", null), new ClusterDeviceSpec("cpu", null), new ClusterDeviceSpec("cpu", null)])], null, null);
        var report = Profiler.Run("all-reduce", Device.Cpu, _ => InProcessCluster.Run(topology, null, r =>
        {
            var storage = r.Device.Backend.Allocate(4096, zeroed: true);
            try
            {
                r.Collectives.AllReduce(storage, 4096, ReduceOp.Sum);
            }
            finally
            {
                storage.Release();
            }
        }).ThrowIfFailed(), new ProfileOptions { WarmRuns = 1, Probes = ["network", "cluster"] });
        Check(report["collectives"] is { Value: 3 } && report["network sent"] is { Value: > 0 } && report["simulated network"] is { Value: > 0 } && report["simulated transfers"] is { Value: > 0 },
            $"the probes' rows: {string.Join("; ", report.Summary.Select(s => $"{s.Name} {s.Value}"))}");
        Check(report.Runs.Any(r => r.Tables.Any(t => t.Name == "simulated links")) && report.Runs.Any(r => r.Tables.Any(t => t.Name == "collectives")), "the probes' tables");
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
            Terminal.TestInput = new StringReader("zero\n2\ndesk\n10.0.0.1:29500\nmini\ncpu,cpu\n64MB\n64MB\ncpu\n64MB\nethernet-1g\n\n\n\n");
            try
            {
                (exit, output, error) = Cli("cluster", "init", "-o", typed);
            }
            finally
            {
                Terminal.TestInput = null;
            }

            Check(exit == 0 && output.Contains("a number of machines") && output.Contains("suggestion:"), $"cluster init asked: {output} {error}");
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

    private static void ClusterCliInspect(Device device)
    {
        if (!FirstRun(nameof(ClusterCliInspect)))
        {
            return;
        }

        string folder = CliTemp(), file = Path.Combine(folder, "topology.json");
        try
        {
            Check(Cli("cluster", "init", "-y", "-o", file, "--nodes", "2", "--names", "desk,mini", "--devices", "cpu,cpu;cpu", "--memory", "64MB,64MB;64MB", "--inter", "ethernet-1g").Exit == 0, "a topology to inspect");
            var (exit, output, error) = Cli("cluster", "inspect", file);
            Check(exit == 0 && output.Contains("2 node(s), 3 rank(s)") && output.Contains("64MB (64 MiB)") && output.Contains("ethernet-1g") && output.Contains("same-device")
                  && output.Contains("all-reduce of 100MB with ring:"), $"cluster inspect: {output} {error}");
            var json = CliJson("cluster", "inspect", file, "--bytes", "1GB", "-j");
            Check((int)json["ranks"]! == 3 && json["links"]!.AsArray().Count == 2 && (double)json["all_reduce_seconds"]!["ring"]! > 0
                  && (double)json["all_reduce_seconds"]!["naive"]! > (double)json["all_reduce_seconds"]!["ring"]!, $"cluster inspect -j: {json}");
            File.WriteAllText(file, """{ "nodes": [ { "name": "a", "gpu": [] } ] }""");
            var (badExit, _, badError) = Cli("cluster", "inspect", file);
            Check(badExit == 1 && badError.Contains("unknown key", StringComparison.Ordinal), $"a bad file: {badError}");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }
}
