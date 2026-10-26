// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Cluster;
using Idrak.Abstraction.Testing;
using Idrak.Cluster;

// Plan 23, phase C3: the hierarchical collective and its predicted win on a two-level topology, the transport selector
// across mixed pipes, the seeded faults (slow worker, dropped worker, link loss, partition, late join) with their
// telemetry, the event log and replay.
internal abstract partial class DesktopTests : Tests
{
    internal static readonly (string Name, Action<Device> Run)[] ClusterTopologyGroup =
    [
        ("cluster: hierarchical equals ring within float rounding, exactly on whole numbers, every rank and rerun identical; the kit's CollectivesSuite passes it over sim on nodes of different sizes", ClusterHierarchicalEqualsRing),
        ("cluster: hierarchical beats ring in simulated time on a two-level topology (two machines of four fake devices, nvlink inside, ethernet-1g between): the ratio printed and above 1, each run's time the clock's played prediction; cluster inspect shows it as the default", ClusterHierarchicalBeatsRing),
        ("cluster: the selector picks per pair in the fastest-first order across mixed pipes and publishes TransportChosen once per pair; a forced transport wins; a forced one that cannot join a pair is refused naming the pair", ClusterSelectorAcrossPipes),
        ("cluster: the default collective is hierarchical across nodes and ring on one; the built-in failure models draw from (seed, step) alone with FailureSettings read from every variable, a bad value naming it", ClusterFailureModels),
        ("cluster: a slow worker shows as Blocked on every other rank and SlowestRank names it, the same on every run", ClusterSlowWorker),
        ("cluster: a dropped worker ends the step on every rank naming it", ClusterDroppedWorker),
        ("cluster: link loss is resent and the step completes with the same values, later", ClusterLinkLoss),
        ("cluster: a partition times out naming both sides on every rank; one shorter than the timeout holds the messages and heals", ClusterPartition),
        ("cluster: a late join connects at step N: the others run the steps before as a group without it, and its clock joins theirs", ClusterLateJoin),
        ("cluster: every fault and rank-state change is a telemetry record with its seed and step, in the JSON lines log and the event log", ClusterFaultTelemetry),
        ("cluster: replay --seed S --step N reproduces the fault schedule and the clock bit for bit (two replays, equal logs); the event log reads back and refuses another format", ClusterReplayBitForBit),
    ];

    // Nodes of 1, 2, 3, 1, 2, … ranks until `world`, so every world of 2 or more spans several nodes of different sizes.
    private static int[] NodeSizes(int world)
    {
        var sizes = new List<int>();
        for (int i = 0, left = world; left > 0; i++)
        {
            int n = Math.Min(left, 1 + (i % 3));
            sizes.Add(n);
            left -= n;
        }

        return [.. sizes];
    }

    // A topology of nodes of `sizes` ranks, every rank on `device`, joined by `inter` between nodes.
    private static ClusterTopology NodesOf(IReadOnlyList<int> sizes, string device, string inter = "ethernet-10g") =>
        ClusterTopology.Create([.. sizes.Select((n, i) => new ClusterNodeSpec($"node{i}", null, [.. Enumerable.Repeat(new ClusterDeviceSpec(device, null), n)]))],
            null, sizes.Count > 1 ? inter : null);

    // Every rank's collectives of `algorithm` over the sim transport on `topology`, on devices carved from `device`.
    private static IReadOnlyList<ICollectives> SimCollectivesOn(ClusterTopology topology, string algorithm, TimeSpan timeout, Device device)
    {
        var options = new ClusterOptions { Coordinator = "sim://test-" + Guid.NewGuid().ToString("N"), Timeout = timeout };
        var provider = Transports.Get("sim");
        var connecting = Enumerable.Range(0, topology.WorldSize)
            .Select(r => Task.Factory.StartNew(() => provider.Connect(new ClusterContext(topology, r, options)), TaskCreationOptions.LongRunning)).ToArray();
        Task.WaitAll(connecting);
        var collectives = Collectives.Get(algorithm);
        return [.. connecting.Select((t, r) => collectives.Create(t.Result, SimulatedDevices.Carve(device, null, r), new ClusterContext(topology, r, options)))];
    }

    // The file the plan's test builds: two machines of four fake devices, nvlink inside (a rate given: built-in links carry
    // none until named), ethernet-1g between. The devices are names only; the run carves every one from the test's device.
    private const string TwoMachinesJson = """
        {
          "format": "idrak-topology/1",
          "nodes": [
            { "name": "m1", "gpus": [ { "device": "cuda:0" }, { "device": "cuda:1" }, { "device": "cuda:2" }, { "device": "cuda:3" } ] },
            { "name": "m2", "gpus": [ { "device": "cuda:0" }, { "device": "cuda:1" }, { "device": "cuda:2" }, { "device": "cuda:3" } ] },
          ],
          "links": { "intra-node": "nvlink-900g", "inter-node": "ethernet-1g" },
        }
        """;

    // An all-reduce of `count` values per rank (random, or whole numbers) with `algorithm`: every rank's result.
    private static float[][] AllReduceRun(IClusterTopology topology, ClusterOptions options, string algorithm, int count, bool whole, out ClusterRunResult run)
    {
        var results = new float[topology.WorldSize][];
        run = InProcessCluster.Run(topology, options with { Collective = algorithm }, r =>
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
        return results;
    }

    private static void ClusterHierarchicalEqualsRing(Device device)
    {
        Conformance.CheckCollectives(new CollectivesUnderTest("hierarchical", (w, t) => SimCollectivesOn(NodesOf(NodeSizes(w), device.ToString()), "hierarchical", t, device)),
            new ContractCheckOptions { RandomCases = 2 }).ThrowIfFailed();

        var topology = ClusterTopology.Parse(TwoMachinesJson);
        var options = new ClusterOptions { Simulation = new SimulationOptions { CarveFrom = device.ToString() } };
        const int count = (8 * 4096) + 5;                               // not a multiple of the ranks or the nodes
        var hierarchical = AllReduceRun(topology, options, "hierarchical", count, whole: false, out _);
        var again = AllReduceRun(topology, options, "hierarchical", count, whole: false, out _);
        var ring = AllReduceRun(topology, options, "ring", count, whole: false, out _);
        Check(hierarchical.All(r => r.AsSpan().SequenceEqual(hierarchical[0])) && again.All(r => r.AsSpan().SequenceEqual(hierarchical[0])),
            "hierarchical: every rank and a rerun hold identical bytes (a fixed summation order, rule 86)");
        float worst = hierarchical[0].Zip(ring[0]).Max(p => Math.Abs(p.First - p.Second));
        Check(worst <= 2e-6f, $"hierarchical agrees with ring within float rounding: the largest difference over {count} values is {worst:G3}");
        Check(AllReduceRun(topology, options, "hierarchical", count, whole: true, out _)[0].AsSpan().SequenceEqual(AllReduceRun(topology, options, "ring", count, whole: true, out _)[0]),
            "hierarchical equals ring exactly on whole numbers");
    }

    private static void ClusterHierarchicalBeatsRing(Device device)
    {
        string folder = CliTemp(), file = Path.Combine(folder, "two-machines.json");
        try
        {
            File.WriteAllText(file, TwoMachinesJson);
            var topology = ClusterTopology.Load(file);
            const int count = 8 * 32768;                                // 1 MiB a rank
            var seconds = new Dictionary<string, double>();
            foreach (string algorithm in new[] { "ring", "hierarchical" })
            {
                var options = new ClusterOptions { Collective = algorithm, Simulation = new SimulationOptions { CarveFrom = device.ToString() } };
                var report = Profiler.Run($"all-reduce ({algorithm})", device, _ => InProcessCluster.Run(topology, options, r =>
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
                }).ThrowIfFailed(), new ProfileOptions { Cold = false, WarmRuns = 1, Probes = ["cluster"] });
                double measured = (report["simulated network"]?.Value ?? 0) / 1e3;
                double predicted = TopologyInspection.PredictAllReduce(topology, algorithm, count * sizeof(float)).TotalSeconds;
                Check(measured > 0 && Math.Abs(measured - predicted) <= 1e-9 * Math.Max(1, predicted),
                    $"{algorithm}: the profile's simulated network time {measured * 1e3:F4} ms is the clock's played prediction {predicted * 1e3:F4} ms");
                seconds[algorithm] = measured;
            }

            double ratio = seconds["ring"] / seconds["hierarchical"];
            Console.WriteLine($"    hierarchical against ring, 1 MiB a rank on 2 machines x 4 devices (nvlink-900g inside, ethernet-1g between): ring {seconds["ring"] * 1e3:F3} ms, hierarchical {seconds["hierarchical"] * 1e3:F3} ms of simulated network time, ratio {ratio:F2}");
            Check(ratio > 1, $"hierarchical beats ring on the two-level topology: ring {seconds["ring"] * 1e3:F3} ms, hierarchical {seconds["hierarchical"] * 1e3:F3} ms (ratio {ratio:F2})");

            if (FirstRun(nameof(ClusterHierarchicalBeatsRing)))
            {
                var (exit, output, error) = Cli("cluster", "inspect", file, "--bytes", "1MB");
                Check(exit == 0 && output.Contains("with hierarchical:", StringComparison.Ordinal) && output.Contains("(the default here)", StringComparison.Ordinal)
                      && output.Contains("sim for 28 pair(s)", StringComparison.Ordinal), $"cluster inspect shows the hierarchical prediction, the default and the transports: {output} {error}");
                var json = CliJson("cluster", "inspect", file, "--bytes", "1MB", "-j");
                Check((string)json["default_collective"]! == "hierarchical" && (double)json["all_reduce_seconds"]!["hierarchical"]! < (double)json["all_reduce_seconds"]!["ring"]!
                      && json["transports"]!.AsArray().Count == 28, $"cluster inspect -j: {json}");
            }
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    // Collects the network events a test asks about.
    private sealed class NetworkEvents : ITelemetryHook
    {
        private readonly Lock _gate = new();
        private readonly List<TransportChosen> _chosen = [];
        private readonly List<RankStateChanged> _states = [];
        private readonly List<FaultInjected> _faults = [];
        private readonly List<CollectiveCompleted> _collectives = [];

        public TelemetryLevel Levels => TelemetryLevel.Network;

        public IReadOnlyList<TransportChosen> Chosen => Copy(_chosen);

        public IReadOnlyList<RankStateChanged> States => Copy(_states);

        public IReadOnlyList<FaultInjected> Faults => Copy(_faults);

        public IReadOnlyList<CollectiveCompleted> CollectivesDone => Copy(_collectives);

        public void OnTransportChosen(in TransportChosen e) => Add(_chosen, e);

        public void OnRankState(in RankStateChanged e) => Add(_states, e);

        public void OnFault(in FaultInjected e) => Add(_faults, e);

        public void OnCollective(in CollectiveCompleted e) => Add(_collectives, e);

        private void Add<T>(List<T> list, T e)
        {
            lock (_gate)
            {
                list.Add(e);
            }
        }

        private List<T> Copy<T>(List<T> list)
        {
            lock (_gate)
            {
                return [.. list];
            }
        }
    }

    // A transport for the selector's test: available for a pair when `joins` says so; never connected.
    private sealed class PairTransport(string name, Func<ClusterContext, int, bool> joins, string reason) : ITransportProvider
    {
        public string Name => name;

        public string Summary => "a test's";

        public TransportCapabilities Capabilities => default;

        public TransportAvailability Availability(ClusterContext context, int peer) =>
            joins(context, peer) ? new(name, true, reason) : new(name, false, $"not {reason}");

        public ITransport Connect(ClusterContext context, CancellationToken cancellationToken = default) => throw new NotSupportedException("a test's transport is never connected");
    }

    private static void ClusterSelectorAcrossPipes(Device device)
    {
        if (!FirstRun(nameof(ClusterSelectorAcrossPipes)))
        {
            return;
        }

        var topology = NodesOf([2, 2], "cpu");
        bool SameNode(ClusterContext c, int peer) => c.Topology.Place(c.Rank).Node == c.Topology.Place(peer).Node;
        Transports.Register(new PairTransport("nvlink", SameNode, "on one node"));
        Transports.Register(new PairTransport("tcp", (_, _) => true, "a route"));
        var events = new NetworkEvents();
        try
        {
            var away = new ClusterOptions { Simulation = new SimulationOptions { Mode = SimulationMode.Localhost } };   // sim is not available
            TransportPlan plan;
            using (Telemetry.Subscribe(events))
            {
                plan = TransportPlan.Make(topology, away);
            }

            Check(plan[0, 1].Transport == "nvlink" && plan[2, 3].Transport == "nvlink" && plan[1, 0] == plan[0, 1]
                  && new[] { (0, 2), (0, 3), (1, 2), (1, 3) }.All(p => plan[p.Item1, p.Item2].Transport == "tcp") && plan.Single is null && plan.Used.SequenceEqual(["nvlink", "tcp"]),
                $"fastest first per pair: nvlink on a node, tcp between nodes: {string.Join("; ", events.Chosen.Select(c => $"{c.RankA}-{c.RankB} {c.Transport}"))}");
            Check(plan[0, 2].Reason.Contains("a route", StringComparison.Ordinal), $"the reason names the transport's own: {plan[0, 2].Reason}");
            var chosen = events.Chosen;
            Check(chosen.Count == 6 && chosen.Select(c => (c.RankA, c.RankB)).Distinct().Count() == 6 && chosen.All(c => c.Transport == plan[c.RankA, c.RankB].Transport),
                $"TransportChosen once per pair, as planned: {chosen.Count} event(s)");

            var forced = TransportPlan.Make(topology, away with { Transport = "tcp" });
            Check(forced.Single == "tcp" && forced[0, 1].Reason.Contains("forced", StringComparison.Ordinal), $"a forced transport wins: {forced[0, 1]}");
            ThrowsNaming<NotSupportedException>(() => TransportPlan.Make(topology, away with { Transport = "nvlink" }), "a forced transport that cannot join a pair",
                "'nvlink' is forced", "ranks 0 and 2", "not on one node");
            ThrowsNaming<NotSupportedException>(() => TransportPlan.Make(topology, away with { Transport = "no-such-pipe" }), "an unregistered forced transport", "not registered");

            var inProcess = new NetworkEvents();
            using (Telemetry.Subscribe(inProcess))
            {
                InProcessCluster.Run(NodesOf([3], "cpu"), null, _ => { }).ThrowIfFailed();
            }

            Check(inProcess.Chosen is { Count: 3 } c3 && c3.All(c => c.Transport == "sim") && c3.Select(c => (c.RankA, c.RankB)).Distinct().Count() == 3,
                $"an in-process run publishes TransportChosen once per pair, sim: {string.Join("; ", inProcess.Chosen)}");
        }
        finally
        {
            Transports.Unregister("nvlink");
            Transports.Unregister("tcp");
        }
    }

    private static void ClusterFailureModels(Device device)
    {
        if (!FirstRun(nameof(ClusterFailureModels)))
        {
            return;
        }

        var options = new ClusterOptions();
        Check(options.CollectiveFor(NodesOf([2, 1], "cpu")) == "hierarchical" && options.CollectiveFor(NodesOf([3], "cpu")) == "ring"
              && (options with { Collective = "naive" }).CollectiveFor(NodesOf([2, 1], "cpu")) == "naive",
            "the default collective: hierarchical across nodes, ring on one node, the option's when set (IDRAK_CLUSTER_COLLECTIVE)");

        var good = new Dictionary<string, string>
        {
            ["IDRAK_CLUSTER_FAULT_RATE"] = "0.5", ["IDRAK_CLUSTER_SLOW_FACTOR"] = "8", ["IDRAK_CLUSTER_FAULT_DURATION"] = "0.25", ["IDRAK_CLUSTER_LOSS_SHARE"] = "0.3",
            ["IDRAK_CLUSTER_LATE_STEPS"] = "3",
        };
        Check(good.Keys.Order().SequenceEqual(FailureSettings.Variables.Order()), "every failure variable has a test value");
        var read = FailureSettings.From(n => good.GetValueOrDefault(n), null);
        Check(read == new FailureSettings { Rate = 0.5, SlowFactor = 8, Duration = TimeSpan.FromSeconds(0.25), LossShare = 0.3, LateSteps = 3 }, $"every value read: {read}");
        Check(FailureSettings.From(_ => null, null) == new FailureSettings(), "no variable leaves the defaults");
        foreach (var (name, value, expected) in new[] { ("IDRAK_CLUSTER_FAULT_RATE", "2", "0 to 1"), ("IDRAK_CLUSTER_SLOW_FACTOR", "0.5", "1 or more"), ("IDRAK_CLUSTER_LATE_STEPS", "0", "1 or more") })
        {
            ThrowsNaming<ArgumentException>(() => FailureSettings.From(n => n == name ? value : null, null), $"{name}={value}", name, expected);
        }

        var topology = NodesOf([1, 2, 1], "cpu");
        foreach (string name in BuiltInFailures.Names.Where(n => n != "none"))
        {
            var model = BuiltInFailures.Create(name, new FailureSettings { Rate = 1, Duration = TimeSpan.FromMilliseconds(2) });
            var first = ClusterReplayScheduleOf(model, topology, 7);
            Check(first.Count > 0 && first.Select(Describe).SequenceEqual(ClusterReplayScheduleOf(model, topology, 7).Select(Describe))
                  && Enumerable.Range(8, 8).Any(seed => !first.Select(Describe).SequenceEqual(ClusterReplayScheduleOf(model, topology, seed).Select(Describe))),
                $"{name}: the same faults for a seed, others for another: {string.Join("; ", first.Select(Describe))}");
            Check(name != "late-join" || first.All(f => f.Step == 0), "a late join is drawn at step 0 only");
        }

        ThrowsNaming<NotSupportedException>(() => ClusterReplay.Schedule(topology, "no-such-model", 1, 3), "an unregistered failure model", "not registered", "FailureModels.Register");
    }

    private static IReadOnlyList<Fault> ClusterReplayScheduleOf(IFailureModel model, IClusterTopology topology, int seed) => FaultInjector.Schedule(model, topology, seed, 20);

    private static string Describe(Fault f) => $"{f.Kind}@{f.Step}[{string.Join(",", f.Ranks)}]x{f.Factor:G4}/{f.Duration.TotalMilliseconds:G4}ms";

    // A failure model drawing exactly the faults given, at their steps.
    private sealed class ScriptedFailures(string name, Fault[] faults) : IFailureModel
    {
        public string Name => name;

        public string Summary => "a test's schedule";

        public IReadOnlyList<Fault> Draw(long step, IClusterTopology topology, Random seeded) => [.. faults.Where(f => f.Step == step)];
    }

    private static IDisposable Scripted(string name, params Fault[] faults)
    {
        FailureModels.Register(new ScriptedFailures(name, faults));
        return new Undo(() => FailureModels.Unregister(name));
    }

    // The built-in "mixed" drawing a fault every step, over the library's for the test.
    private static IDisposable MixedEveryStep()
    {
        FailureModels.Register(BuiltInFailures.Create("mixed", new FailureSettings { Rate = 1, Duration = TimeSpan.FromMilliseconds(1) }));
        return new Undo(() => FailureModels.Unregister("mixed"));
    }

    // `steps` steps of an all-reduce of `count` values (rank + 1 + step each, whole numbers); each rank's results per step.
    private static Action<ClusterRank> Steps(int steps, int count, ConcurrentDictionary<(int Rank, long Step), (int World, float First)>? results = null) => r =>
    {
        var storage = r.Device.Backend.Allocate(count, zeroed: false);
        var values = new float[count];
        try
        {
            for (long step = 0; step < steps; step++)
            {
                if (!r.BeginStep(step))
                {
                    continue;
                }

                Array.Fill(values, (float)(r.Rank + 1 + step));
                r.Device.Backend.Upload(values, storage);
                r.Collectives.AllReduce(storage, count, ReduceOp.Sum);
                r.Device.Backend.Download(storage, values);
                results?.TryAdd((r.Rank, step), (r.Collectives.WorldSize, values[0]));
            }
        }
        finally
        {
            storage.Release();
        }
    };

    private static ClusterOptions Faulty(string failures, int seed) =>
        new() { Timeout = TimeSpan.FromSeconds(60), Simulation = new SimulationOptions { Failures = failures, Seed = seed } };

    private static void ClusterSlowWorker(Device device)
    {
        if (!FirstRun(nameof(ClusterSlowWorker)))
        {
            return;
        }

        var topology = NodesOf([1, 1, 1, 1], "cpu");
        using var slow = Scripted("test-slow", new Fault(FaultKind.SlowWorker, 0, [2], TimeSpan.Zero, 8));
        IReadOnlyList<CollectiveCompleted> Collect(string failures, out NetworkEvents events)
        {
            events = new NetworkEvents();
            using (Telemetry.Subscribe(events))
            {
                InProcessCluster.Run(topology, Faulty(failures, 5), Steps(1, 4096)).ThrowIfFailed();
            }

            return [.. events.CollectivesDone.OrderBy(c => c.Rank)];
        }

        var plain = Collect("none", out _);
        var slowed = Collect("test-slow", out var seen);
        var again = Collect("test-slow", out _);
        Check(plain.Count == 4 && slowed.Count == 4, $"one all-reduce a rank: {plain.Count} and {slowed.Count}");
        for (int q = 0; q < 4; q++)
        {
            if (q == 2)
            {
                continue;
            }

            Check(slowed[q].Blocked > plain[q].Blocked && slowed[q].SlowestRank == 2,
                $"rank {q}: blocked {slowed[q].Blocked.TotalMicroseconds:F1} us with rank 2 slow, {plain[q].Blocked.TotalMicroseconds:F1} us without; the slowest named is rank {slowed[q].SlowestRank}");
        }

        Check(slowed.Zip(again).All(p => p.First.Blocked == p.Second.Blocked && p.First.SlowestRank == p.Second.SlowestRank && p.First.Simulated == p.Second.Simulated),
            "the same blocked times and slowest ranks on every run");
        Check(seen.States.Any(s => s is { Rank: 2, State: RankState.Slow, Step: 0, Seed: 5 }), $"rank 2's state: slow at step 0, seed 5: {string.Join("; ", seen.States)}");
    }

    private static void ClusterDroppedWorker(Device device)
    {
        if (!FirstRun(nameof(ClusterDroppedWorker)))
        {
            return;
        }

        using var drop = Scripted("test-drop", new Fault(FaultKind.DroppedWorker, 1, [1], TimeSpan.Zero, 0));
        var events = new NetworkEvents();
        long start = System.Diagnostics.Stopwatch.GetTimestamp();
        ClusterRunResult run;
        using (Telemetry.Subscribe(events))
        {
            run = InProcessCluster.Run(NodesOf([1, 1, 1], "cpu"), Faulty("test-drop", 3), Steps(3, 1024));
        }

        var took = System.Diagnostics.Stopwatch.GetElapsedTime(start);
        Check(!run.Succeeded && run.Errors.All(e => e?.Message.Contains("rank 1 was dropped at step 1", StringComparison.Ordinal) == true) && took < TimeSpan.FromSeconds(30),
            $"every rank's step ends naming the dropped rank ({took.TotalSeconds:F1} s): {string.Join(" | ", run.Errors.Select(e => e?.Message))}");
        Check(events.States.Count(s => s is { Rank: 1, State: RankState.Dropped, Step: 1, Seed: 3 }) == 1, $"rank 1 dropped at step 1 once, with the seed: {string.Join("; ", events.States)}");
        Check(run.Log is { } log && log.Faults is [{ Kind: FaultKind.DroppedWorker, Step: 1 }] && log.ClockAt(1, 1) is null, "the log holds the fault; rank 1 never began step 1");
    }

    private static void ClusterLinkLoss(Device device)
    {
        if (!FirstRun(nameof(ClusterLinkLoss)))
        {
            return;
        }

        var topology = NodesOf([1, 1, 1], "cpu");
        using var loss = Scripted("test-loss", new Fault(FaultKind.LinkLoss, 0, [0, 1], TimeSpan.Zero, 0.6));
        var plainResults = new ConcurrentDictionary<(int Rank, long Step), (int World, float First)>();
        var lossyResults = new ConcurrentDictionary<(int Rank, long Step), (int World, float First)>();
        var plain = InProcessCluster.Run(topology, Faulty("none", 9), Steps(2, 2048, plainResults));
        var lossy = InProcessCluster.Run(topology, Faulty("test-loss", 9), Steps(2, 2048, lossyResults));
        plain.ThrowIfFailed();
        lossy.ThrowIfFailed();
        Check(lossyResults.Count == 6 && plainResults.OrderBy(p => p.Key).SequenceEqual(lossyResults.OrderBy(p => p.Key)) && lossyResults.All(p => p.Value.First == 6 + (3 * p.Key.Step)),
            "the lossy run gives the same values (each step's sum of rank + 1 + step over 3 ranks)");
        Check(lossy.Log!.Resends > 0 && plain.Log!.Resends == 0 && lossy.Log.TransferRecords.Where(t => t.Resends > 0).All(t => (t.From, t.To) is (0, 1) or (1, 0)),
            $"lost messages are sent again on the lossy link only: {lossy.Log.Resends} resend(s)");
        Check(lossy.Simulated.Max() > plain.Simulated.Max(), $"the step completes later: {lossy.Simulated.Max().TotalMicroseconds:F1} us against {plain.Simulated.Max().TotalMicroseconds:F1} us");
    }

    private static void ClusterPartition(Device device)
    {
        if (!FirstRun(nameof(ClusterPartition)))
        {
            return;
        }

        var topology = NodesOf([1, 1, 1], "cpu");
        using (Scripted("test-cut", new Fault(FaultKind.Partition, 1, [0], TimeSpan.Zero, 0)))
        {
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            var run = InProcessCluster.Run(topology, Faulty("test-cut", 4), Steps(3, 1024));
            var took = System.Diagnostics.Stopwatch.GetElapsedTime(start);
            Check(!run.Succeeded && run.Errors.All(e => e is TimeoutException t && t.Message.Contains("separates rank(s) 0 from rank(s) 1, 2", StringComparison.Ordinal)) && took < TimeSpan.FromSeconds(30),
                $"every rank times out naming both sides ({took.TotalSeconds:F1} s): {string.Join(" | ", run.Errors.Select(e => $"{e?.GetType().Name}: {e?.Message}"))}");
        }

        using (Scripted("test-heal", new Fault(FaultKind.Partition, 1, [0], TimeSpan.FromMilliseconds(5), 0)))
        {
            var events = new NetworkEvents();
            ClusterRunResult healed;
            var results = new ConcurrentDictionary<(int Rank, long Step), (int World, float First)>();
            using (Telemetry.Subscribe(events))
            {
                healed = InProcessCluster.Run(topology, Faulty("test-heal", 4), Steps(4, 1024, results));
            }

            healed.ThrowIfFailed();
            var plain = InProcessCluster.Run(topology, Faulty("none", 4), Steps(4, 1024));
            Check(results.Count == 12 && results.All(p => p.Value.First == 6 + (3 * p.Key.Step)), "a partition shorter than the timeout only delays the step");
            Check(healed.Simulated.Min() - plain.Simulated.Max() >= TimeSpan.FromMilliseconds(4), $"the messages across were held: {healed.Simulated.Min().TotalMilliseconds:F3} ms against {plain.Simulated.Max().TotalMilliseconds:F3} ms");
            var states = events.States;
            Check(Enumerable.Range(0, 3).All(r => states.Any(s => s is { State: RankState.Partitioned, Step: 1, Seed: 4 } && s.Rank == r)
                                                && states.Any(s => s.State == RankState.Rejoined && s.Rank == r && s.Step is 2 or 3)),
                $"every rank partitioned at step 1 and rejoined after: {string.Join("; ", states.Where(s => s.Step >= 0))}");
        }
    }

    private static void ClusterLateJoin(Device device)
    {
        if (!FirstRun(nameof(ClusterLateJoin)))
        {
            return;
        }

        using var late = Scripted("test-late", new Fault(FaultKind.LateJoin, 0, [2], TimeSpan.Zero, 2));
        var results = new ConcurrentDictionary<(int Rank, long Step), (int World, float First)>();
        var events = new NetworkEvents();
        ClusterRunResult run;
        using (Telemetry.Subscribe(events))
        {
            run = InProcessCluster.Run(NodesOf([1, 1, 1], "cpu"), Faulty("test-late", 6), Steps(4, 512, results));
        }

        run.ThrowIfFailed();
        Check(!results.ContainsKey((2, 0)) && !results.ContainsKey((2, 1)), "rank 2 sits out steps 0 and 1");
        for (long step = 0; step < 4; step++)
        {
            int world = step < 2 ? 2 : 3;
            float sum = step < 2 ? 3 + (2 * step) : 6 + (3 * step);
            for (int r = 0; r < world; r++)
            {
                Check(results.TryGetValue((r, step), out var got) && got == (world, sum), $"rank {r} at step {step}: a group of {world} summing to {sum}; got {(results.TryGetValue((r, step), out var g) ? g : default)}");
            }
        }

        Check(events.States.Any(s => s is { Rank: 2, State: RankState.Joined, Step: 2, Seed: 6 }), $"rank 2 joined at step 2: {string.Join("; ", events.States.Where(s => s.Step >= 0))}");
        var log = run.Log!;
        double? joined = log.ClockAt(2, 2);
        Check(joined > 0 && Enumerable.Range(0, 3).All(r => log.ClockAt(r, 2) is { } c && BitConverter.DoubleToInt64Bits(c) == BitConverter.DoubleToInt64Bits(joined!.Value))
              && log.ClockAt(2, 0) is null && log.ClockAt(2, 1) is null,
            $"its clock joins the others' at step 2: {string.Join(", ", Enumerable.Range(0, 3).Select(r => log.ClockAt(r, 2)))}");
    }

    private static void ClusterFaultTelemetry(Device device)
    {
        if (!FirstRun(nameof(ClusterFaultTelemetry)))
        {
            return;
        }

        string path = Path.Combine(Path.GetTempPath(), $"idrak-faults-{Guid.NewGuid():N}.jsonl");
        try
        {
            using var mixed = MixedEveryStep();
            var events = new NetworkEvents();
            ClusterRunResult run;
            var json = new JsonLinesLogger(path, TelemetryLevel.Network);
            using (Telemetry.Subscribe(json))
            using (Telemetry.Subscribe(events))
            {
                run = InProcessCluster.Run(NodesOf([1, 1, 1, 1], "cpu"), Faulty("mixed", 13), Steps(4, 256));
            }

            json.Dispose();
            run.ThrowIfFailed();
            var faults = events.Faults;
            Check(faults.Count == 4 && faults.Select(f => f.Step).SequenceEqual([0L, 1, 2, 3]) && faults.All(f => f.Seed == 13),
                $"one fault a step, each with its seed and step: {string.Join("; ", faults.Select(f => $"{f.Kind}@{f.Step} seed {f.Seed}"))}");
            Check(run.Log!.Faults.Select(Describe).SequenceEqual(faults.Select(f => Describe(new Fault(f.Kind, f.Step, f.Ranks, f.Duration, f.Factor)))), "the event log holds the same faults");
            var fromFaults = events.States.Where(s => s.Step >= 0 && s.State is not RankState.Finished).ToList();
            Check((fromFaults.Count > 0 || faults.All(f => f.Kind == FaultKind.LinkLoss)) && fromFaults.All(s => s.Seed == 13),
                $"every state change a fault made carries the seed and its step: {string.Join("; ", fromFaults)}");
            var logged = run.Log.States.Select(s => (s.Rank, s.State, s.Step)).Order().ToList();
            Check(logged.SequenceEqual(fromFaults.Select(s => (s.Rank, s.State, s.Step)).Order()) && run.Log.States.All(s => s.Seed == 13),
                $"the event log holds the same state changes: {string.Join("; ", logged)}");
            var lines = File.ReadAllLines(path).Select(l => JsonNode.Parse(l)!).ToList();
            Check(lines.Count(l => (string?)l["event"] == "fault" && (int)l["seed"]! == 13 && l["step"] is not null) == 4
                  && lines.Where(l => (string?)l["event"] == "rank_state" && (long)l["step"]! >= 0 && (string?)l["state"] != "finished").All(l => (int?)l["seed"] == 13),
                "the JSON lines log writes each fault and each state change with its seed and step");
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static void ClusterReplayBitForBit(Device device)
    {
        if (!FirstRun(nameof(ClusterReplayBitForBit)))
        {
            return;
        }

        using var mixed = MixedEveryStep();
        var topology = NodesOf([2, 1, 2], "cpu");
        string folder = CliTemp(), file = Path.Combine(folder, "run.events");
        try
        {
            var options = Faulty("mixed", 11) with { Collective = "hierarchical" };
            var full = InProcessCluster.Run(topology, options with { Simulation = options.Simulation with { EventLog = file } }, Steps(6, 1024));
            full.ThrowIfFailed();
            var log = full.Log!;
            Check(File.Exists(file) && EventLog.Load(file).Bytes.Span.SequenceEqual(log.Bytes.Span), "IDRAK_CLUSTER_EVENT_LOG's file holds the log");
            const long step = 3;
            var first = ClusterReplay.Run(topology, options, 11, step, Steps(6, 1024));
            var second = ClusterReplay.Run(topology, options, 11, step, Steps(6, 1024));
            first.ThrowIfFailed();
            second.ThrowIfFailed();
            var a = first.Log!;
            Check(a.Bytes.Span.SequenceEqual(second.Log!.Bytes.Span), $"two replays write equal logs ({a.Bytes.Length:N0} bytes)");
            var schedule = log.Faults.Where(f => f.Step <= step).Select(Describe).ToList();
            Check(schedule.Count > 0 && a.Faults.Select(Describe).SequenceEqual(schedule) && ClusterReplay.Schedule(topology, "mixed", 11, step).Select(Describe).SequenceEqual(schedule),
                $"the replay draws the run's faults through step {step}: {string.Join("; ", schedule)}");
            for (int r = 0; r < topology.WorldSize; r++)
            {
                for (long s = 0; s <= step; s++)
                {
                    Check(Bits(a.ClockAt(r, s)) == Bits(log.ClockAt(r, s)), $"rank {r}'s clock at step {s}: {a.ClockAt(r, s):R} in the replay, {log.ClockAt(r, s):R} in the run");
                }

                Check(a.ClockAt(r, step + 1) is null, $"rank {r} stops after step {step}");
                var replayed = a.TransferRecords.Where(t => t.From == r).ToList();
                Check(replayed.Count > 0 && replayed.SequenceEqual(log.TransferRecords.Where(t => t.From == r).Take(replayed.Count)),
                    $"rank {r}'s {replayed.Count} transfers are the run's first ones, bit for bit");
            }

            var bytes = log.Bytes.ToArray();
            Check(EventLog.Parse(bytes).Bytes.Span.SequenceEqual(bytes), "the log reads back");
            bytes[8] = 9;
            ThrowsNaming<InvalidDataException>(() => EventLog.Parse(bytes), "another version", "version 9", "version 1");
            ThrowsNaming<InvalidDataException>(() => EventLog.Parse("not a log at all"u8), "another format", "IDRKLOG");
            ThrowsNaming<InvalidDataException>(() => EventLog.Parse(log.Bytes.Span[..^3]), "a log cut short", "ends early");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    private static long? Bits(double? value) => value is { } v ? BitConverter.DoubleToInt64Bits(v) : null;
}
