// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Net;
using System.Net.Sockets;
using Idrak.Abstraction.Cluster;
using Idrak.Cli;
using Idrak.Cluster;
using Idrak.Networking.Abstractions;
using Idrak.Networking.Collectives;
using Idrak.Networking.Transports;

// Plan 23, phase C8a: the command line. The measured chunk size kept once per transport; an address on a link override
// (a cable on its own subnet) through init; cluster sim's usage with --placement pipeline; --cluster on the training
// commands with the rank and world-size rules of decision 3. Nothing is left running.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] ClusterCliGroup =
    [
        ("cluster: cli an override's address (a cable on its own subnet) is asked by cluster init and kept by the topology file", ClusterCliOverrideAddress),
        ("cluster: cli the measured chunk size is kept once per transport (a second group reads it), a forced ChunkBytes skips it, and tuning show lists it", ClusterCliChunkKept),
        ("cluster: cli cluster sim --placement pipeline: another command with it, and stages that do not divide the ranks, are usage errors", ClusterCliSimPipeline),
        ("cluster: cli --cluster on the training commands: a two-worker node names the missing --rank (IDRAK_RANK on tune, whose --rank stays the adapter's), a contradicting world size names both, --world-size without --coordinator names it", ClusterCliHandStarted),
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
                var (transports, topology, options) = RealGroup(new TcpProvider(), 2, TimeSpan.FromSeconds(60));
                options = options with { TuningFile = overrides.TuningFile, ChunkBytes = overrides.ChunkBytes, ChunkProbeBytes = overrides.ChunkProbeBytes };
                var chunks = new long[2];
                var ranks = Enumerable.Range(0, 2).Select(r => Task.Factory.StartNew(() =>
                {
                    using var collectives = new RingCollectives(transports[r], Device.Cpu, new ClusterContext(topology, r, options));
                    var values = Enumerable.Range(0, 1000).Select(i => (float)(i + r)).ToArray();
                    Check(((IHostValueCollectives)collectives).TryAllReduce(values, ReduceOp.Sum) && values[10] == 21f, "the all-reduce sums");
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

    // pretrain's arguments for a few steps of the tiny model into `run`.
    private static string[] TinyPretrain(string run, int steps = 3) =>
        ["pretrain", TinyLlama, "--tokens", PretrainData("tokens.bin"), "--steps", steps.ToString(CultureInfo.InvariantCulture), "--max-length", "32", "--batch-tokens", "64",
         "--optimizer", "sgd", "--lr", "0.2", "--no-graphs", "--log-every", "1", "-o", run];

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
                var two = ClusterCli("pretrain", model, "--tokens", "t.bin", "-o", Path.Combine(folder, "run"), "--cluster", twoWorkers);
                Check(two.Code == ExitCodes.Usage && two.Error.Contains("--rank N", StringComparison.Ordinal) && two.Error.Contains("2 workers", StringComparison.Ordinal),
                    $"a two-worker node names the missing --rank: {two.Error}");
                var tune = ClusterCli("tune", "train", "chats.jsonl", "-b", model, "-o", Path.Combine(folder, "adapter"), "--rank", "4", "--cluster", twoWorkers);
                Check(tune.Code == ExitCodes.Usage && tune.Error.Contains("IDRAK_RANK", StringComparison.Ordinal) && tune.Error.Contains("adapter's rank", StringComparison.Ordinal),
                    $"tune names IDRAK_RANK: its --rank is the adapter's: {tune.Error}");
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
