// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Abstraction.Cluster;
using Idrak.Cli;
using Idrak.Cluster;
using Idrak.Layers;
using Idrak.Networking.Abstractions;

// Plan 23, phase C8a: the command line and calibration. idrak transports; cluster replay over C3's ClusterReplay;
// cluster calibrate from a run's log and from a bench's JSON (a shared adapter included), the fitted profile then used by
// inspect and listed by tuning show; the measured chunk size kept once per transport; the event log readable while a run
// goes; an address on a link override (a cable on its own subnet) through init, inspect and direct-link; cluster sim for
// tune and train, with --profile and with --placement pipeline; profile --cluster's hops; runs show's network share;
// --cluster on the training commands with the rank and world-size rules of decision 3. Every run is in-process or a
// bounded child process; nothing is left running.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ClusterCliGroup =
    [
        ("cluster: cli transports lists every registered transport with who registered it, its capabilities and, per setting here, whether it joins two ranks and why", ClusterCliTransports),
        ("cluster: cli an override's address (a cable on its own subnet) is asked by cluster init, shown by inspect and dialled by direct-link", ClusterCliOverrideAddress),
        ("cluster: cli calibrate corrects a profile from a run whose log it reads (--dry-run keeps nothing); inspect then uses the fitted profile and tuning show lists it", ClusterCliCalibrateRun),
        ("cluster: cli calibrate from a bench's JSON fits each link and finds an adapter shared by a machine's ranks; the topology gives each pair its share", ClusterCliCalibrateBench),
        ("cluster: cli bench --concurrent measures pairs at once over localhost processes and calibrate reads its JSON; no process left", ClusterCliBenchConcurrent),
        ("cluster: cli the measured chunk size is kept once per transport (a second group reads it), a forced ChunkBytes skips it, and tuning show lists it", ClusterCliChunkKept),
        ("cluster: cli the event log is readable while a run goes and the finished file is the in-memory log's bytes", ClusterCliEventLogLive),
        ("cluster: cli replay --seed S --step N reproduces the fault schedule and the clock of a cluster sim run bit for bit; runs show charts the run's network share", ClusterCliReplay),
        ("cluster: cli cluster sim runs train and tune on two ranks (rank 0 writing the run) and --profile prints the probes' rows with the predicted step beside the measured, also over a link cheaper than the clock's tick", ClusterCliSimTrainTune),
        ("cluster: cli cluster sim --placement pipeline runs pretrain as a pipeline; another command with it is a usage error", ClusterCliSimPipeline),
        ("cluster: cli profile --cluster shows a token's hops through a simulated pipeline", ClusterCliProfileHops),
        ("cluster: cli --cluster on the training commands: a one-worker node needs no rank, a two-worker node names the missing --rank (IDRAK_RANK on tune, whose --rank stays the adapter's), a contradicting world size names both, --world-size without --coordinator names it", ClusterCliHandStarted),
    ];

    private static (int Code, string Output, string Error) ClusterCli(params string[] args)
    {
        using var output = new StringWriter(CultureInfo.InvariantCulture);
        using var error = new StringWriter(CultureInfo.InvariantCulture);
        int code = CommandLine.Run(args, output, error);
        return (code, output.ToString(), error.ToString());
    }

    // Runs `body` with the environment variables given (null: unset), putting the earlier values back after.
    private static void WithVariables(IReadOnlyDictionary<string, string?> variables, Action body)
    {
        var before = variables.Keys.ToDictionary(k => k, Environment.GetEnvironmentVariable);
        try
        {
            foreach (var (name, value) in variables)
            {
                Environment.SetEnvironmentVariable(name, value);
            }

            body();
        }
        finally
        {
            foreach (var (name, value) in before)
            {
                Environment.SetEnvironmentVariable(name, value);
            }
        }
    }

    private static void ClusterCliTransports(Device device)
    {
        if (!FirstRun(nameof(ClusterCliTransports)))
        {
            return;
        }

        _ = device;
        var (code, text, error) = ClusterCli("transports");
        Check(code == 0 && text.Contains("sim", StringComparison.Ordinal) && text.Contains("tcp", StringComparison.Ordinal) && text.Contains("shared-memory", StringComparison.Ordinal)
              && text.Contains("one process", StringComparison.Ordinal) && text.Contains("two machines", StringComparison.Ordinal), $"idrak transports lists the transports and the settings: {code} {error}\n{text}");
        Check(text.Contains("sim, one process: yes", StringComparison.Ordinal) && text.Contains("shared-memory, two machines: no", StringComparison.Ordinal),
            $"each setting's answer comes with its reason: {text}");
        Check(text.Contains("rdma", StringComparison.Ordinal) && text.Contains("-P PLUGIN.dll", StringComparison.Ordinal), $"the names a plug-in provides are named with the -P hint: {text}");
        var json = JsonNode.Parse(ClusterCli("transports", "tcp", "-j").Output)!;
        var tcp = json["transports"]!.AsArray().Single()!;
        Check((string?)tcp["name"] == "tcp" && (bool?)tcp["reliable"] == true && (bool?)tcp["cross_machine"] == true
              && (bool?)tcp["available"]!["processes here"]!["available"] == true && ((string?)tcp["available"]!["two machines"]!["reason"])!.Length > 0,
            $"-j gives the capabilities and every setting's reason: {json}");
        var missing = ClusterCli("transports", "no-such-transport");
        Check(missing.Code != 0 && missing.Error.Contains("not registered", StringComparison.Ordinal) && missing.Error.Contains("tcp", StringComparison.Ordinal), $"an unknown name lists the registered ones: {missing.Error}");
    }

    private static void ClusterCliOverrideAddress(Device device)
    {
        if (!FirstRun(nameof(ClusterCliOverrideAddress)))
        {
            return;
        }

        _ = device;
        string folder = TempFolder("cluster-cli-cable");
        try
        {
            string file = Path.Combine(folder, "lan.json");
            var (code, text, error) = ClusterCli("cluster", "init", "-y", "-o", file, "--nodes", "2", "--names", "desk,mini", "--address", "192.168.1.10:29500",
                "--devices", "cpu;cpu", "--memory", "64MB;64MB", "--inter", "ethernet-1g", "--override", "desk,mini=direct-link-10g", "--override-address", "10.0.0.1,10.0.0.2");
            Check(code == 0 && File.Exists(file), $"cluster init -y writes the file with the cable's addresses: {code} {error}\n{text}");
            Check(File.ReadAllText(file).Contains("\"address\": [\"10.0.0.1\", \"10.0.0.2\"]", StringComparison.Ordinal), $"the override carries each end's address: {File.ReadAllText(file)}");
            var topology = ClusterTopology.Load(file);
            Check(topology.Overrides.Single() is { AddressA: "10.0.0.1", AddressB: "10.0.0.2" } && topology.AddressOf(1, 0) == "10.0.0.2" && topology.AddressOf(0, 1) == "10.0.0.1",
                "the reader gives each end's address on the cable to the other end");
            Check(ClusterTopology.Parse(topology.ToJson()).AddressOf(1, 0) == "10.0.0.2", "the writer keeps the addresses");

            var builder = new TopologyBuilder(MachineFacts.Gather(), new Dictionary<string, string>
            {
                ["nodes"] = "2", ["node.1.name"] = "desk", ["node.1.address"] = "", ["node.2.name"] = "mini", ["node.1.devices"] = "cpu", ["node.2.devices"] = "cpu",
                ["node.1.device.1.memory"] = "", ["node.2.device.1.memory"] = "", ["inter-node"] = "ethernet-1g", ["overrides"] = "desk,mini=direct-link",
            });
            var question = builder.Next!;
            Check(question.Key == "override.1.address" && question.Prompt.Contains("desk", StringComparison.Ordinal) && question.Suggestion is { Length: > 0 }
                  && question.Description.Contains("subnet", StringComparison.Ordinal), $"the builder asks the cable's addresses with a suggestion: {question}");
            Check(builder.Answer("10.0.0.1 x") is not null && builder.Answer(",10.0.0.9") is null, "a bad answer is refused; one side may be left to its node's address");

            var (_, inspected, _) = ClusterCli("cluster", "inspect", file);
            Check(inspected.Contains("own link desk-mini (direct-link-10g): desk at 10.0.0.1 (on the link), mini at 10.0.0.2 (on the link)", StringComparison.Ordinal),
                $"inspect shows where each end is reached on the cable: {inspected}");

            var options = new ClusterOptions
            {
                Key = ClusterKey.FromText("a key of sixteen letters"), Coordinator = "192.168.1.10:29500", Simulation = new SimulationOptions { Mode = SimulationMode.Localhost },
            };
            var availability = Transports.Get("direct-link").Availability(new ClusterContext(topology, 0, options), 1);
            Check(availability.Available && availability.Reason.Contains("10.0.0.2", StringComparison.Ordinal), $"direct-link dials the peer's address on the cable: {availability}");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // A two-machine topology of one CPU rank each, joined by ethernet-1g.
    private static string TwoMachines(string folder, int perNode = 1)
    {
        string file = Path.Combine(folder, $"two-{perNode}.json");
        var devices = Enumerable.Repeat(new ClusterDeviceSpec("cpu", null), perNode).ToList();
        File.WriteAllText(file, ClusterTopology.Create([new ClusterNodeSpec("desk", null, devices), new ClusterNodeSpec("mini", null, devices)], null, "ethernet-1g").ToJson());
        return file;
    }

    private static void ClusterCliCalibrateRun(Device device)
    {
        if (!FirstRun(nameof(ClusterCliCalibrateRun)))
        {
            return;
        }

        _ = device;
        string folder = TempFolder("cluster-cli-calibrate");
        string profiles = Path.Combine(folder, "profiles.txt");
        try
        {
            WithVariables(new Dictionary<string, string?> { ["IDRAK_CLUSTER_PROFILES_FILE"] = profiles }, () =>
            {
                string topology = TwoMachines(folder);
                string run = Path.Combine(folder, "run");
                Directory.CreateDirectory(run);
                // A run's log: transfers over tcp that took 50 us + bytes / 40 MB/s, beside the records calibration skips.
                const double Bandwidth = 40e6, Latency = 50e-6;
                var log = new StringBuilder();
                foreach (long bytes in new long[] { 1024, 65536, 1 << 20, 4 << 20, 65536, 1 << 20 })
                {
                    using var buffer = new MemoryStream();
                    using (var writer = new Utf8JsonWriter(buffer))
                    {
                        TelemetryJson.Write(writer, new TelemetryRecord(DateTimeOffset.Now, new TransferCompleted("tcp", 1, 0, -11, bytes, TimeSpan.FromSeconds(Latency + bytes / Bandwidth), TimeSpan.Zero)));
                    }

                    log.Append(Encoding.UTF8.GetString(buffer.ToArray())).Append('\n');
                }

                log.Append("{\"event\": \"transfer\", \"transport\": \"sim\", \"from\": 0, \"to\": 1, \"bytes\": 9, \"wall_ms\": 9}\nnot json\n");
                File.WriteAllText(Path.Combine(run, "log.jsonl"), log.ToString());

                var dry = ClusterCli("cluster", "calibrate", "--from", run, "--topology", topology, "--dry-run");
                Check(dry.Code == 0 && dry.Output.Contains("ethernet-1g", StringComparison.Ordinal) && dry.Output.Contains("6 measured transfer(s)", StringComparison.Ordinal) && !File.Exists(profiles),
                    $"--dry-run prints the deltas and keeps nothing: {dry.Code} {dry.Error}\n{dry.Output}");
                var kept = ClusterCli("cluster", "calibrate", "--from", run, "--topology", topology, "-j");
                var json = JsonNode.Parse(kept.Output)!;
                Check(kept.Code == 0 && File.Exists(profiles) && (bool?)json["written"] == true, $"calibrate keeps the fitted profile: {kept.Error}\n{kept.Output}");
                var fitted = ClusterCache.Calibrated("ethernet-1g")!;
                Check(Math.Abs(fitted.BandwidthBytesPerSecond / Bandwidth - 1) < 1e-6 && Math.Abs(fitted.Latency.TotalSeconds - Latency) < 1e-9 && fitted.Residual < 1e-9 && fitted.Samples == 6,
                    $"the fit recovers the run's bandwidth and latency: {fitted}");
                Check(Math.Abs((double)json["links"]![0]!["bandwidth_before"]! - 125e6) < 1 && File.ReadAllText(profiles).StartsWith(ClusterCache.ProfilesHeader, StringComparison.Ordinal),
                    "the delta is against the name's rate, and the file has its header");
                Check(LinkProfiles.Resolve("ethernet-1g").BandwidthBytesPerSecond == fitted.BandwidthBytesPerSecond && LinkProfiles.Find("ethernet-1g")!.BandwidthBytesPerSecond == 125e6
                      && ClusterTopology.Load(topology).Link(0, 1).Latency == fitted.Latency, "the calibration resolves in the built-in's place; the registered profile stays behind it");
                var (_, inspected, _) = ClusterCli("cluster", "inspect", topology, "--unit", "MB");
                Check(inspected.Contains("calibrated", StringComparison.Ordinal) && inspected.Contains("38.15 MB/s", StringComparison.Ordinal), $"inspect uses the fitted profile: {inspected}");
                var (_, tuning, _) = ClusterCli("tuning", "show", "--backend", "cluster");
                Check(tuning.Contains("ethernet-1g", StringComparison.Ordinal) && tuning.Contains("calibrated links", StringComparison.Ordinal), $"tuning show lists the calibration: {tuning}");
                var back = ClusterCache.ParseProfiles(ClusterCache.SerializeProfiles([fitted])).Single();
                Check(ClusterCache.ParseProfiles("idrak-cluster-profiles 2\nethernet-1g 1 0 0 0 1 0 2026-10-18T00:00:00Z x\n").Count == 0
                      && back.Link == fitted.Link && back.BandwidthBytesPerSecond == fitted.BandwidthBytesPerSecond && back.Latency == fitted.Latency && back.Samples == fitted.Samples
                      && back.Source == fitted.Source, "a file of another version is ignored; one written reads back");
                var missing = ClusterCli("cluster", "calibrate", "--from", run);
                Check(missing.Code == ExitCodes.Usage && missing.Error.Contains("--topology", StringComparison.Ordinal), $"a run's log without its topology names --topology: {missing.Error}");
            });
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void ClusterCliCalibrateBench(Device device)
    {
        if (!FirstRun(nameof(ClusterCliCalibrateBench)))
        {
            return;
        }

        _ = device;
        string folder = TempFolder("cluster-cli-calibrate-bench");
        string profiles = Path.Combine(folder, "profiles.txt");
        try
        {
            WithVariables(new Dictionary<string, string?> { ["IDRAK_CLUSTER_PROFILES_FILE"] = profiles }, () =>
            {
                // Two machines of two ranks: the cross pairs' link moves 100 MB/s after 100 us; two pairs at once only 110 MB/s.
                string topology = TwoMachines(folder, perNode: 2);
                const double Bandwidth = 100e6, Latency = 100e-6;
                const long Bytes = 16 << 20;
                var rows = new JsonArray();
                foreach (var (a, b) in new[] { (0, 2), (0, 3), (1, 2), (1, 3) })
                {
                    rows.Add(new JsonObject
                    {
                        ["rank_a"] = a, ["rank_b"] = b, ["transport"] = "tcp", ["bytes"] = Bytes, ["rounds"] = 3,
                        ["cold_bytes_per_second"] = Bandwidth / 2, ["warm_bytes_per_second"] = Bytes / (2 * Latency + Bytes / Bandwidth),
                        ["cold_latency_seconds"] = 2 * Latency, ["warm_latency_seconds"] = Latency,
                    });
                }

                var bench = new JsonObject
                {
                    ["topology"] = topology, ["bytes"] = Bytes, ["rounds"] = 3, ["rows"] = rows,
                    ["concurrent"] = new JsonArray(new JsonObject
                    {
                        ["rank_a"] = 0, ["rank_b"] = 2, ["pairs"] = 2, ["transport"] = "tcp", ["bytes"] = Bytes, ["rounds"] = 3,
                        ["cold_bytes_per_second"] = 100e6, ["warm_bytes_per_second"] = 110e6,
                    }),
                };
                string file = Path.Combine(folder, "bench.json");
                File.WriteAllText(file, bench.ToJsonString());
                var (code, text, error) = ClusterCli("cluster", "calibrate", "--from", file);
                Check(code == 0 && text.Contains("one adapter shared by the machine's ranks", StringComparison.Ordinal), $"calibrate reads the bench and finds the shared adapter: {code} {error}\n{text}");
                var fitted = ClusterCache.Calibrated("ethernet-1g")!;
                Check(Math.Abs(fitted.BandwidthBytesPerSecond / Bandwidth - 1) < 1e-6 && Math.Abs(fitted.Latency.TotalSeconds - Latency) < 1e-9 && fitted.AdapterBytesPerSecond == 110e6,
                    $"the bench's pairs give the link's bandwidth and latency, the concurrent row its adapter: {fitted}");
                var two = ClusterTopology.Load(topology);
                Check(two.Link(0, 2).BandwidthBytesPerSecond == 55e6 && two.Link(1, 3).BandwidthBytesPerSecond == 55e6,
                    $"each of a machine's two ranks crossing gets half the adapter: {two.Link(0, 2).BandwidthBytesPerSecond}");
                var once = TopologyInspection.PredictAllReduce(two, "hierarchical", 1 << 20);
                var again = TopologyInspection.PredictAllReduce(two, "hierarchical", 1 << 20);
                Check(once == again && once > TimeSpan.Zero, "the shared adapter is a static share: the clock plays the same every time");
                var alone = LinkCalibration.AdaptersFromBench(bench, two, tolerance: 0.5);
                Check(alone.Single() is { Shared: false, Pairs: 2 }, "within the tolerance, the pairs have an adapter each");
            });
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void ClusterCliBenchConcurrent(Device device)
    {
        if (!FirstRun(nameof(ClusterCliBenchConcurrent)))
        {
            return;
        }

        _ = device;
        string folder = TempFolder("cluster-cli-bench-concurrent");
        string profiles = Path.Combine(folder, "profiles.txt");
        try
        {
            WithVariables(new Dictionary<string, string?> { ["IDRAK_CLUSTER_PROFILES_FILE"] = profiles, ["IDRAK_CLUSTER_THROTTLE"] = "0", ["IDRAK_CLUSTER_TIMEOUT"] = "60" }, () =>
            {
                string topology = Path.Combine(folder, "four.json");
                var devices = new List<ClusterDeviceSpec> { new("cpu", null), new("cpu", null) };
                File.WriteAllText(topology, ClusterTopology.Create([new ClusterNodeSpec("desk", null, devices), new ClusterNodeSpec("mini", null, devices)], null, "ethernet-10g").ToJson());
                var (code, text, error) = ClusterCli("cluster", "bench", "--topology", topology, "--localhost", "--concurrent", "--bytes", "64KB", "--rounds", "2", "-j", "--timeout", "180s");
                JsonNode? json = null;
                try
                {
                    json = JsonNode.Parse(text);
                }
                catch (JsonException)
                {
                    // Checked below with the output.
                }

                Check(code == 0 && json?["concurrent"]?.AsArray() is { Count: 1 } together && (int?)together[0]!["pairs"] == 2 && (double?)together[0]!["warm_bytes_per_second"] > 0
                      && json["rows"]!.AsArray().Count == 6, $"the bench measures every pair and the two pairs between the machines at once: {code} {error}\n{text}");
                string file = Path.Combine(folder, "bench.json");
                File.WriteAllText(file, text);
                var calibrated = ClusterCli("cluster", "calibrate", "--from", file, "--dry-run");
                Check(calibrated.Code == 0 && calibrated.Output.Contains("ethernet-10g", StringComparison.Ordinal) && calibrated.Output.Contains("pairs at once", StringComparison.Ordinal),
                    $"calibrate reads the measured bench: {calibrated.Error}\n{calibrated.Output}");
            });
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void ClusterCliChunkKept(Device device)
    {
        if (!FirstRun(nameof(ClusterCliChunkKept)))
        {
            return;
        }

        _ = device;
        string folder = TempFolder("cluster-cli-chunk");
        string tuning = Path.Combine(folder, "tuning.txt");
        try
        {
            // Two ranks over tcp in this process; the first host-value all-reduce measures (or reads) the chunk.
            long[] Run(ClusterOptions overrides)
            {
                var (transports, topology, options) = RealGroup("tcp", 2, TimeSpan.FromSeconds(60));
                options = options with { TuningFile = overrides.TuningFile, ChunkBytes = overrides.ChunkBytes, ChunkProbeBytes = overrides.ChunkProbeBytes };
                var chunks = new long[2];
                var ranks = Enumerable.Range(0, 2).Select(r => Task.Factory.StartNew(() =>
                {
                    using var collectives = Collectives.Get("ring").Create(transports[r], Device.Cpu, new ClusterContext(topology, r, options));
                    var values = Enumerable.Range(0, 1000).Select(i => (float)(i + r)).ToArray();
                    Check(((IHostValueCollectives)collectives).TryAllReduce(values, ReduceOp.Sum, WireType.Float32) && values[10] == 21f, "the all-reduce sums");
                    chunks[r] = ((HostCollectives)collectives).ChunkBytes;
                }, TaskCreationOptions.LongRunning)).ToArray();
                try
                {
                    Task.WaitAll(ranks);
                }
                finally
                {
                    foreach (var t in transports)
                    {
                        t.Dispose();
                    }
                }

                return chunks;
            }

            var probe = new ClusterOptions { TuningFile = tuning, ChunkProbeBytes = 1 << 20 };
            int before = HostCollectives.ChunkMeasurements;
            var first = Run(probe);
            int measured = HostCollectives.ChunkMeasurements - before;
            var entry = ClusterCache.FindTuning("chunk", "tcp", tuning);
            Check(measured == 2 && entry is not null && first[0] == first[1] && first[0] == entry.Value && entry.Detail.Contains("65536 B", StringComparison.Ordinal),
                $"the first group measures the chunk on each rank alike and keeps it: {measured} measurements, chunks {first[0]} and {first[1]}, kept {entry}");
            var second = Run(probe);
            Check(HostCollectives.ChunkMeasurements - before == 2 && second.All(c => c == entry!.Value), "a second group reads the kept size and measures nothing");
            string other = Path.Combine(folder, "other.txt");
            var forced = Run(new ClusterOptions { TuningFile = other, ChunkBytes = 8192, ChunkProbeBytes = 1 << 20 });
            Check(HostCollectives.ChunkMeasurements - before == 2 && forced.All(c => c == 8192) && !File.Exists(other), "a forced ChunkBytes skips the measurement and keeps nothing");
            WithVariables(new Dictionary<string, string?> { ["IDRAK_CLUSTER_TUNING_FILE"] = tuning }, () =>
            {
                var (_, text, _) = ClusterCli("tuning", "show", "--backend", "cluster");
                Check(text.Contains("measured chunk sizes", StringComparison.Ordinal) && text.Contains("tcp", StringComparison.Ordinal), $"tuning show lists the measured chunk: {text}");
            });
            Check(ClusterCache.ParseTuning(ClusterCache.SerializeTuning([entry!])).Single().Value == entry!.Value, "the sizes' file reads back");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void ClusterCliEventLogLive(Device device)
    {
        if (!FirstRun(nameof(ClusterCliEventLogLive)))
        {
            return;
        }

        _ = device;
        string folder = TempFolder("cluster-cli-events");
        try
        {
            var topology = DpTopology(Device.Cpu, 2);
            string path = Path.Combine(folder, "events.idrklog");
            var options = DpOptions() with { Simulation = new SimulationOptions { Seed = 3, EventLog = path, EventLogFlushSteps = 1 } };
            int midClocks = 0, midRecords = -1;
            double? midClock = null;
            void Work(ClusterRank rank, bool look)
            {
                var host = (IHostValueCollectives)rank.Collectives;
                var values = new float[256];
                for (long step = 0; step < 6; step++)
                {
                    rank.BeginStep(step);
                    Array.Fill(values, rank.Rank + 1);
                    host.TryAllReduce(values, ReduceOp.Sum, WireType.Float32);
                    if (look && step == 3)
                    {
                        var seen = EventLog.Load(path);                    // the run is still going
                        (midClocks, midClock, midRecords) = (seen.Clocks.Count, seen.ClockAt(0, 3), (int)seen.Transfers);
                    }
                }
            }

            var live = InProcessCluster.Run(topology, options, rank => Work(rank, rank.Rank == 0));
            live.ThrowIfFailed();
            var memory = InProcessCluster.Run(topology, options with { Simulation = options.Simulation with { EventLog = null } }, rank => Work(rank, false));
            memory.ThrowIfFailed();
            Check(midClock is not null && midClocks >= 4 && midClocks < live.Log!.Clocks.Count && midRecords > 0,
                $"mid-run the file holds the clocks and transfers flushed so far: {midClocks} clocks of {live.Log!.Clocks.Count}, rank 0 at step 3 {midClock}");
            Check(File.ReadAllBytes(path).AsSpan().SequenceEqual(memory.Log!.Bytes.Span) && live.Log.Bytes.Span.SequenceEqual(memory.Log.Bytes.Span),
                "the finished file is the bytes a run kept in memory gives, bit for bit");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // pretrain's arguments for a few steps of the tiny model into `run`.
    private static string[] TinyPretrain(string run, int steps = 3) =>
        ["pretrain", TinyLlama, "--tokens", PretrainData("tokens.bin"), "--steps", steps.ToString(CultureInfo.InvariantCulture), "--max-length", "32", "--batch-tokens", "64",
         "--optimizer", "sgd", "--lr", "0.2", "--no-graphs", "--log-every", "1", "-o", run];

    private static void ClusterCliReplay(Device device)
    {
        if (!FirstRun(nameof(ClusterCliReplay)))
        {
            return;
        }

        _ = device;
        string folder = TempFolder("cluster-cli-replay");
        try
        {
            string topology = Path.Combine(folder, "two.json");
            File.WriteAllText(topology, DpTopology(Device.Cpu, 2).ToJson());
            string run = Path.Combine(folder, "run");
            var (code, text, error) = ClusterCli(["cluster", "sim", "--topology", topology, "--failures", "slow-worker", "--seed", "3", "--", .. TinyPretrain(run)]);
            string log = Path.Combine(run, "cluster-events.idrklog");
            Check(code == 0 && File.Exists(log) && text.Contains("event log:", StringComparison.Ordinal), $"cluster sim writes the event log into the run folder: {code} {error}\n{text}");

            var schedule = ClusterCli("cluster", "replay", "--topology", topology, "--step", "2", "--from", log, "-j");
            var json = JsonNode.Parse(schedule.Output)!;
            Check(schedule.Code == 0 && (bool?)json["matches"] == true && (int?)json["seed"] == 3 && (string?)json["failures"] == "slow-worker",
                $"the schedule from the log's seed and model matches the run's faults: {schedule.Error}\n{schedule.Output}");
            var rerun = ClusterCli(["cluster", "replay", "--topology", topology, "--step", "1", "--from", log, "--", .. TinyPretrain(Path.Combine(folder, "again"))]);
            Check(rerun.Code == 0 && rerun.Output.Contains("matches", StringComparison.Ordinal) && rerun.Output.Contains("rank 1: clock", StringComparison.Ordinal),
                $"the rerun through step 1 has every rank's clock of the run bit for bit: {rerun.Code} {rerun.Error}\n{rerun.Output}");
            var other = ClusterCli("cluster", "replay", "--topology", topology, "--step", "1", "--seed", "4", "--from", log);
            Check(other.Code == ExitCodes.Usage && other.Error.Contains("seed 3", StringComparison.Ordinal) && other.Error.Contains("seed 4", StringComparison.Ordinal),
                $"a replay of another seed than the log's is refused naming both: {other.Error}");
            var noStep = ClusterCli("cluster", "replay", "--topology", topology);
            Check(noStep.Code == ExitCodes.Usage && noStep.Error.Contains("--step", StringComparison.Ordinal), $"a replay without a step names --step: {noStep.Error}");

            var (shown, chart, _) = ClusterCli("runs", "show", run);
            Check(shown == 0 && chart.Contains("Network ", StringComparison.Ordinal) && chart.Contains("network %", StringComparison.Ordinal),
                $"runs show charts the network share of the run's steps: {chart}");
            var shownJson = JsonNode.Parse(ClusterCli("runs", "show", run, "-j").Output)!;
            Check(shownJson["network"]!.AsArray().Count == 3, $"-j has the network share of each step: {shownJson["network"]}");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void ClusterCliSimTrainTune(Device device)
    {
        if (!FirstRun(nameof(ClusterCliSimTrainTune)))
        {
            return;
        }

        _ = device;
        string folder = TempFolder("cluster-cli-sim-train");
        string model = WriteChatModel(TinyChatSpec());
        try
        {
            string topology = Path.Combine(folder, "two.json");
            File.WriteAllText(topology, DpTopology(Device.Cpu, 2).ToJson());
            var random = new Random(5);
            var csv = new StringBuilder("x1,x2,y\n");
            for (int i = 0; i < 96; i++)
            {
                double a = random.NextDouble() * 4 - 2, b = random.NextDouble() * 4 - 2;
                csv.Append(CultureInfo.InvariantCulture, $"{a:F4},{b:F4},{3 * a - 2 * b + 1:F4}\n");
            }

            File.WriteAllText(Path.Combine(folder, "reg.csv"), csv.ToString());
            File.WriteAllText(Path.Combine(folder, "reg.json"), Network.Input(2).Linear(8).ReLU().Linear(1).Named("reg").ToJson().ToJsonString());
            string run = Path.Combine(folder, "run");
            string[] Train(string into) => ["train", Path.Combine(folder, "reg.json"), "--data", Path.Combine(folder, "reg.csv"), "--epochs", "3", "--batch", "16",
                "--run", into, "-o", Path.Combine(into + "-model", "reg.ikm")];
            var (code, text, error) = ClusterCli(["cluster", "sim", "--topology", topology, "--", .. Train(run)]);
            Check(code == 0 && File.Exists(Path.Combine(run, "run.json")) && File.Exists(Path.Combine(run, "log.jsonl")) && File.Exists(Path.Combine(run + "-model", "reg.ikm"))
                  && text.Contains("network time on the virtual clock: rank 0", StringComparison.Ordinal), $"cluster sim runs train on two ranks, rank 0 writing the run: {code} {error}\n{text}");

            string profiled = Path.Combine(folder, "profiled");
            var (pcode, ptext, perror) = ClusterCli(["cluster", "sim", "--topology", topology, "--profile", "--", .. Train(profiled)]);
            Check(pcode == 0 && ptext.Contains("collectives", StringComparison.Ordinal) && ptext.Contains("simulated network", StringComparison.Ordinal)
                  && ptext.Contains("gradient buckets", StringComparison.Ordinal) && ptext.Contains("step: measured", StringComparison.Ordinal) && ptext.Contains("predicted", StringComparison.Ordinal)
                  && ptext.Contains("cpu", StringComparison.Ordinal), $"--profile prints the probes' rows and the predicted step beside the measured: {pcode} {perror}\n{ptext}");
            Check(Directory.Exists(profiled) && !Directory.EnumerateDirectories(folder, "profiled.profile-*").Any(), "the warm run's folder is removed after");

            // A link cheaper than the clock's smallest tick (a fast same-device copy on a fast CPU measures that way): every
            // simulated transfer reports 0, and the run is still a simulated one with its predicted step.
            LinkProfiles.Register(new LinkProfile(ClusterTopology.SameDevice, 1e18, TimeSpan.Zero));
            try
            {
                string free = Path.Combine(folder, "free");
                var (fcode, ftext, ferror) = ClusterCli(["cluster", "sim", "--topology", topology, "--profile", "--", .. Train(free)]);
                Check(fcode == 0 && ftext.Contains("step: measured", StringComparison.Ordinal) && ftext.Contains("predicted", StringComparison.Ordinal),
                    $"--profile over a link that costs nothing on the clock still prints the predicted step: {fcode} {ferror}\n{ftext}");
            }
            finally
            {
                LinkProfiles.Unregister(ClusterTopology.SameDevice);
            }
            var localhost = ClusterCli(["cluster", "sim", "--topology", topology, "--mode", "localhost", "--profile", "--", .. Train(Path.Combine(folder, "x"))]);
            Check(localhost.Code == ExitCodes.Usage && localhost.Error.Contains("in-process", StringComparison.Ordinal), $"--profile with --mode localhost names why: {localhost.Error}");

            string data = Path.Combine(folder, "chats.jsonl");
            File.WriteAllLines(data, Enumerable.Range(0, 8).Select(i => new JsonObject { ["question"] = $"What is {i} plus {i}?", ["answer"] = $"{i + i}" }.ToJsonString()));
            string adapter = Path.Combine(folder, "adapter");
            var tuned = ClusterCli("cluster", "sim", "--topology", topology, "--", "tune", "train", data, "-b", model, "-o", adapter, "--rank", "4", "--alpha", "8",
                "--max-length", "64", "--batch-tokens", "256", "--epochs", "1", "--eval-fraction", "0");
            Check(tuned.Code == 0 && File.Exists(Path.Combine(adapter, "idrak-tuning.json")) && tuned.Output.Contains("network time on the virtual clock", StringComparison.Ordinal),
                $"cluster sim runs tune on two ranks, rank 0 writing the adapter: {tuned.Code} {tuned.Error}\n{tuned.Output}");
            var noRun = ClusterCli("cluster", "sim", "--topology", topology, "--", "train", Path.Combine(folder, "reg.json"), "--data", Path.Combine(folder, "reg.csv"));
            Check(noRun.Code == ExitCodes.Usage && noRun.Error.Contains("--run DIR", StringComparison.Ordinal), $"train without its run folder names --run: {noRun.Error}");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
            Directory.Delete(model, recursive: true);
        }
    }

    private static void ClusterCliSimPipeline(Device device)
    {
        if (!FirstRun(nameof(ClusterCliSimPipeline)))
        {
            return;
        }

        _ = device;
        string folder = TempFolder("cluster-cli-sim-pipeline");
        try
        {
            string topology = Path.Combine(folder, "two.json");
            File.WriteAllText(topology, DpTopology(Device.Cpu, 2).ToJson());
            string run = Path.Combine(folder, "run");
            var (code, text, error) = ClusterCli(["cluster", "sim", "--topology", topology, "--placement", "pipeline", "--stages", "2", "--", .. TinyPretrain(run, steps: 2)]);
            Check(code == 0 && text.Contains("pipeline of 2 stage(s)", StringComparison.Ordinal) && Directory.EnumerateDirectories(run, "checkpoint-*").Any(),
                $"cluster sim --placement pipeline trains the tiny model as two stages, rank 0 writing the checkpoint: {code} {error}\n{text}");
            var (_, shown, _) = ClusterCli("runs", "show", run);
            Check(shown.Contains("Network ", StringComparison.Ordinal), $"runs show charts a pipeline's blocked share: {shown}");
            var refused = ClusterCli("cluster", "sim", "--topology", topology, "--placement", "pipeline", "--", "train", "x.json", "--data", "x.csv", "--run", run);
            Check(refused.Code == ExitCodes.Usage && refused.Error.Contains("pretrain", StringComparison.Ordinal), $"another command with a pipeline is refused naming pretrain: {refused.Error}");
            var stages = ClusterCli(["cluster", "sim", "--topology", topology, "--placement", "pipeline", "--stages", "3", "--", .. TinyPretrain(Path.Combine(folder, "y"))]);
            Check(stages.Code == ExitCodes.Usage && stages.Error.Contains("--stages 3", StringComparison.Ordinal), $"stages that do not divide the ranks are named: {stages.Error}");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void ClusterCliProfileHops(Device device)
    {
        if (!FirstRun(nameof(ClusterCliProfileHops)))
        {
            return;
        }

        _ = device;
        string folder = TempFolder("cluster-cli-profile-hops");
        try
        {
            string topology = Path.Combine(folder, "two.json");
            File.WriteAllText(topology, DpTopology(Device.Cpu, 2).ToJson());
            var (code, text, error) = ClusterCli("profile", TinyLlama, "--cluster", topology, "--steps", "2", "--prompt-tokens", "8", "-d", "cpu");
            Check(code == 0 && text.Contains("rank 0 -> rank 1", StringComparison.Ordinal) && text.Contains("rank 1 -> rank 0", StringComparison.Ordinal)
                  && text.Contains("pipeline of 2 stage(s)", StringComparison.Ordinal), $"profile --cluster lists a token's hops through the stages: {code} {error}\n{text}");
            var json = JsonNode.Parse(ClusterCli("profile", TinyLlama, "--cluster", topology, "--steps", "2", "--prompt-tokens", "8", "-d", "cpu", "-j").Output)!;
            var hops = json["hops"]!.AsArray();
            Check(hops.Count == 2 && hops.All(h => (double?)h!["bytes"] > 0 && (double?)h["messages"] >= 1), $"-j gives each hop's messages and bytes per token: {json}");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void ClusterCliHandStarted(Device device)
    {
        if (!FirstRun(nameof(ClusterCliHandStarted)))
        {
            return;
        }

        _ = device;
        string folder = TempFolder("cluster-cli-hand");
        try
        {
            int port;
            using (var probe = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp))
            {
                probe.Bind(new IPEndPoint(IPAddress.Loopback, 0));
                port = ((IPEndPoint)probe.LocalEndPoint!).Port;              // free, and nobody listens once closed
            }

            string host = Environment.MachineName;
            // This host is the second node, of one worker (rank 1); the coordinator is a loopback port nobody listens on.
            string oneWorker = Path.Combine(folder, "one.json");
            File.WriteAllText(oneWorker, ClusterTopology.Create([new ClusterNodeSpec("elsewhere", $"127.0.0.1:{port}", [new ClusterDeviceSpec("cpu", null)]),
                new ClusterNodeSpec(host, null, [new ClusterDeviceSpec("cpu", null)])], null, "ethernet-1g").ToJson());
            // This host holds two workers.
            string twoWorkers = Path.Combine(folder, "two.json");
            File.WriteAllText(twoWorkers, ClusterTopology.Create([new ClusterNodeSpec(host, $"127.0.0.1:{port}", [new ClusterDeviceSpec("cpu", null), new ClusterDeviceSpec("cpu", null)])], null, null).ToJson());
            WithVariables(new Dictionary<string, string?>
            {
                ["IDRAK_RANK"] = null, ["IDRAK_WORLD_SIZE"] = null, ["IDRAK_COORDINATOR"] = null, ["IDRAK_CLUSTER_NODE"] = null, ["IDRAK_CLUSTER_TOPOLOGY"] = null,
                ["IDRAK_CLUSTER_RETRIES"] = "1", ["IDRAK_CLUSTER_RETRY_DELAY"] = "0.05", ["IDRAK_CLUSTER_TIMEOUT"] = "3",
            }, () =>
            {
                string model = Path.Combine(folder, "no-model");
                var one = ClusterCli("pretrain", model, "--tokens", "t.bin", "-o", Path.Combine(folder, "run"), "--cluster", oneWorker);
                Check(one.Code != ExitCodes.Usage && one.Code != ExitCodes.Ok && one.Error.Contains("rank 1", StringComparison.Ordinal) && one.Error.Contains("could not reach", StringComparison.Ordinal),
                    $"a one-worker node needs no rank: rank 1 is found from the file and dials the coordinator: {one.Code} {one.Error}");
                var two = ClusterCli("pretrain", model, "--tokens", "t.bin", "-o", Path.Combine(folder, "run"), "--cluster", twoWorkers);
                Check(two.Code == ExitCodes.Usage && two.Error.Contains("--rank N", StringComparison.Ordinal) && two.Error.Contains("2 workers", StringComparison.Ordinal),
                    $"a two-worker node names the missing --rank: {two.Error}");
                var tune = ClusterCli("tune", "train", "chats.jsonl", "-b", model, "-o", Path.Combine(folder, "adapter"), "--rank", "4", "--cluster", twoWorkers);
                Check(tune.Code == ExitCodes.Usage && tune.Error.Contains("IDRAK_RANK", StringComparison.Ordinal) && tune.Error.Contains("adapter's rank", StringComparison.Ordinal),
                    $"tune names IDRAK_RANK: its --rank is the adapter's: {tune.Error}");
                WithVariables(new Dictionary<string, string?> { ["IDRAK_RANK"] = "1" }, () =>
                {
                    var joined = ClusterCli("tune", "train", "chats.jsonl", "-b", model, "-o", Path.Combine(folder, "adapter"), "--rank", "4", "--cluster", twoWorkers);
                    Check(joined.Code != ExitCodes.Usage && joined.Code != ExitCodes.Ok && !joined.Error.Contains("Rank 4", StringComparison.Ordinal),
                        $"with IDRAK_RANK tune joins as rank 1 (its peer never comes) and --rank 4 stays the adapter's: {joined.Code} {joined.Error}");
                });
                var world = ClusterCli("pretrain", model, "--tokens", "t.bin", "-o", Path.Combine(folder, "run"), "--cluster", oneWorker, "--world-size", "3");
                Check(world.Code == ExitCodes.Usage && world.Error.Contains("--world-size 3", StringComparison.Ordinal) && world.Error.Contains("2 rank(s)", StringComparison.Ordinal),
                    $"a world size that contradicts the file is refused naming both: {world.Error}");
                WithVariables(new Dictionary<string, string?> { ["IDRAK_WORLD_SIZE"] = "5" }, () =>
                {
                    var variable = ClusterCli("train", "x.json", "--data", "x.csv", "--cluster", oneWorker);
                    Check(variable.Code == ExitCodes.Usage && variable.Error.Contains("IDRAK_WORLD_SIZE=5", StringComparison.Ordinal) && variable.Error.Contains(oneWorker, StringComparison.Ordinal),
                        $"IDRAK_WORLD_SIZE that contradicts the file is refused naming both: {variable.Error}");
                });
                var bare = ClusterCli("pretrain", model, "--tokens", "t.bin", "-o", Path.Combine(folder, "run"), "--world-size", "2", "--rank", "0");
                Check(bare.Code == ExitCodes.Usage && bare.Error.Contains("--coordinator", StringComparison.Ordinal), $"--world-size without --coordinator names it: {bare.Error}");
                var node = ClusterCli("resume", Path.Combine(folder, "run"), "--cluster", oneWorker, "--node", "nowhere");
                Check(node.Code == ExitCodes.Usage && node.Error.Contains("nowhere", StringComparison.Ordinal), $"a node the file does not have is named: {node.Error}");
            });
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }
}
