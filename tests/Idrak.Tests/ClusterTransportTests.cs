// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Cluster;
using Idrak.Abstraction.Testing;
using Idrak.Cli;
using Idrak.Cluster;
using Idrak.Networking.Abstractions;
using Idrak.Networking.Bench;
using Idrak.Networking.Transports;
using Idrak.Nlp;

// Plan 23, phase C5: the real transports (shared-memory, tcp with its rendezvous and keyed hello, direct-link, pcie-host,
// nvlink), one group over several transports, Backend.CopyTo, the localhost cluster of child processes with its throttle,
// the link bench, and idrak cluster start, bench and sim --mode localhost. Child processes are this test program again
// (--cluster-child) or the CLI; every one is ended by the launcher's finally, and each watches its launcher's process.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ClusterTransportGroup =
    [
        ("cluster: the real transports are registered (shared-memory, pcie-host, nvlink and p2p, direct-link, tcp), each says per pair whether it joins two ranks and why; direct-link and tcp share one carrier", ClusterTransportsRegistered),
        ("cluster: shared-memory, tcp, pcie-host and nvlink pass the kit's TransportSuite in one process (2 to 8 ranks, messages larger than a shared-memory ring)", ClusterRealTransportSuite),
        ("cluster: ring and naive pass the kit's CollectivesSuite over tcp and shared-memory, and hierarchical over a group routed across shared-memory and tcp", ClusterRealCollectivesSuite),
        ("cluster: tcp rendezvous retries then fails naming the rank and address, each attempt as long as one refused connect on this machine, not the timeout", ClusterRendezvousRetries),
        ("cluster: a hello with a wrong key is refused on tcp and shared-memory, the key never in a message; nothing listens off loopback without a key", ClusterWrongKey),
        ("cluster: a rank whose process ends leaves its peers an error naming it at once over tcp and shared-memory", ClusterRealRankDies),
        ("cluster: CopyTo default equals Download then Upload on every device (inside a device, between two backends of it, to and from the CPU), values past the count untouched", ClusterCopyTo),
        ("cluster: nvlink falls back to pcie-host where peer access is refused and the telemetry says so once per pair; pcie-host hands device memory through the host", ClusterNvlinkFallback),
        ("cluster: TransportSuite and CollectivesSuite work in child processes: 2 ranks on shared-memory, 3 forced onto tcp, 3 on two nodes routed over both; every result bit-identical to the in-process run", ClusterChildProcesses),
        ("cluster: localhost mode with the throttle reproduces the profile's bandwidth within 20% (cluster bench --localhost, cold and warm apart)", ClusterThrottle),
        ("cluster: the localhost run and the in-process run give bit-identical weights (cluster sim --mode localhost runs pretrain in processes); rank 0's log holds every rank's per-step summary", ClusterLocalhostWeights),
        ("cluster: cli cluster start finds this host's node, starts one process per rank with its rank, world size, node, device and coordinator, and stops all when one fails, none left running", ClusterCliStart),
    ];

    // The child processes of these tests: `--cluster-child SCENARIO FOLDER`.
    internal static int ClusterChild(string scenario, string[] arguments)
    {
        string folder = arguments.Length > 0 ? arguments[0] : Path.GetTempPath();
        try
        {
            switch (scenario)
            {
                case "suite":
                {
                    using var joined = LocalhostCluster.Join();
                    var rank = joined.Rank;
                    File.WriteAllText(Path.Combine(folder, $"plan-{rank.Rank}.txt"), string.Join("+", joined.Plan.Used));
                    if (ChildExchange(rank.Transport) is { } problem)
                    {
                        Console.Error.WriteLine(problem);
                        return 4;
                    }

                    var values = ClusterWork(rank);
                    File.WriteAllBytes(Path.Combine(folder, $"rank-{rank.Rank}.bin"), MemoryMarshal.AsBytes(values.AsSpan()).ToArray());
                    rank.Transport.Barrier();
                    return 0;
                }

                case "env":
                {
                    string rank = Environment.GetEnvironmentVariable("IDRAK_RANK") ?? "none";
                    File.WriteAllLines(Path.Combine(folder, $"env-{rank}.txt"),
                        [.. new[] { "IDRAK_RANK", "IDRAK_WORLD_SIZE", "IDRAK_CLUSTER_NODE", "IDRAK_CLUSTER_DEVICE", "IDRAK_COORDINATOR", "IDRAK_CLUSTER_RUN", "IDRAK_CLUSTER_TOPOLOGY" }
                            .Select(n => $"{n}={Environment.GetEnvironmentVariable(n)}")]);
                    Console.WriteLine($"rank {rank} wrote its environment");
                    return 0;
                }

                case "fail-one":
                {
                    string rank = Environment.GetEnvironmentVariable("IDRAK_RANK") ?? "none";
                    File.WriteAllText(Path.Combine(folder, $"pid-{rank}.txt"), Environment.ProcessId.ToString(CultureInfo.InvariantCulture));
                    if (rank == "1")
                    {
                        Thread.Sleep(500);
                        Console.Error.WriteLine("rank 1 fails on purpose");
                        return 7;
                    }

                    Thread.Sleep(TimeSpan.FromSeconds(60));          // ended by the launcher long before
                    return 0;
                }

                case "die":
                {
                    using var joined = LocalhostCluster.Join();
                    if (joined.Rank.Rank == 1)
                    {
                        Thread.Sleep(300);
                        Environment.Exit(0);                         // no goodbye (and no failure, so the launcher lets rank 0 notice)
                    }

                    try
                    {
                        joined.Rank.Transport.Receive(1, new byte[8], 3);
                        return 1;
                    }
                    catch (Exception e)
                    {
                        File.WriteAllText(Path.Combine(folder, "rank-0-error.txt"), $"{e.GetType().Name}: {e.Message}");
                        return 0;
                    }
                }

                case "shm-speed":
                    return SharedMemorySpeedChild(folder);       // phase C6: the doorbell's latency and a ring's rate

                default:
                    Console.Error.WriteLine($"no cluster child scenario '{scenario}'");
                    return 2;
            }
        }
        catch (Exception e)
        {
            Console.Error.WriteLine($"{e.GetType().Name}: {e.Message}");
            return 1;
        }
    }

    // Every rank sends every other rank messages of several sizes (one larger than a shared-memory ring) under tags 0-2,
    // half by awaited sends, and checks what arrives, the later tags first; null when all arrived whole and in order.
    private static string? ChildExchange(ITransport transport)
    {
        int[] sizes = [0, 1, 7, 65_537, 300_000];
        int me = transport.Rank, world = transport.WorldSize;
        static byte[] Payload(int from, int to, int k, int length)
        {
            var bytes = new byte[length];
            for (int j = 0; j < length; j++)
            {
                bytes[j] = (byte)((from * 31) + (to * 17) + (k * 13) + (j * 5) + (j >> 8));
            }

            return bytes;
        }

        // One task per peer sends its messages in order (a tag's messages arrive in the order they were sent), half awaited.
        var sends = Enumerable.Range(0, world).Where(to => to != me).Select(to => Task.Run(async () =>
        {
            for (int k = 0; k < sizes.Length; k++)
            {
                var payload = Payload(me, to, k, sizes[k]);
                if (k % 2 == 0)
                {
                    await transport.SendAsync(to, payload, k % 3);
                }
                else
                {
                    transport.Send(to, payload, k % 3);
                }
            }
        })).ToList();

        var buffer = new byte[sizes.Max()];
        for (int from = 0; from < world; from++)
        {
            for (int tag = 2; tag >= 0 && from != me; tag--)
            {
                for (int k = 0; k < sizes.Length; k++)
                {
                    if (k % 3 != tag)
                    {
                        continue;
                    }

                    int length = k % 2 == 0 ? transport.Receive(from, buffer, tag) : transport.ReceiveAsync(from, buffer, tag).AsTask().GetAwaiter().GetResult();
                    if (!buffer.AsSpan(0, length).SequenceEqual(Payload(from, me, k, sizes[k])))
                    {
                        return $"rank {me} got {length} bytes from rank {from} under tag {tag} where message {k} of {sizes[k]} bytes was due";
                    }
                }
            }
        }

        Task.WaitAll([.. sends]);
        transport.Barrier();
        return null;
    }

    // Every collective operation on seeded values of this rank, in order, as one array of floats (identical across modes).
    private static float[] ClusterWork(ClusterRank rank)
    {
        var c = rank.Collectives;
        var backend = rank.Device.Backend;
        int world = c.WorldSize, me = c.Rank;
        var random = new Random(1000 + me);
        float[] Random(int count, bool whole) => [.. Enumerable.Range(0, count).Select(_ => whole ? random.Next(-8, 9) : (float)((random.NextDouble() - 0.5) * (1 + me) * Math.Pow(10, random.Next(-3, 4))))];
        var all = new List<float>();
        void Run(float[] input, int outputLength, Action<Storage, Storage> op)
        {
            var buffer = backend.Allocate(Math.Max(1, input.Length), zeroed: false);
            var output = outputLength == 0 ? buffer : backend.Allocate(outputLength, zeroed: true);
            try
            {
                backend.Upload(input, buffer);
                op(buffer, output);
                var result = new float[outputLength == 0 ? input.Length : outputLength];
                backend.Download(output, result);
                all.AddRange(result);
            }
            finally
            {
                buffer.Release();
                if (!ReferenceEquals(output, buffer))
                {
                    output.Release();
                }
            }
        }

        var values = Random(1_023, whole: false);
        Run(values, 0, (b, _) => c.AllReduce(b, values.Length, ReduceOp.Sum));
        var whole = Random(7, whole: true);
        Run(whole, 0, (b, _) => c.AllReduce(b, whole.Length, ReduceOp.Sum, WireType.BFloat16));
        Run(values, 0, (b, _) => c.AllReduce(b, values.Length, ReduceOp.Max));
        Run(values, 0, (b, _) => c.AllReduce(b, values.Length, ReduceOp.Min));
        var piece = Random(5, whole: false);
        Run(piece, world * 5, (b, o) => c.AllGather(b, 5, o));
        var scattered = Random(world * 9, whole: false);
        Run(scattered, 9, (b, o) => c.ReduceScatter(b, 9, o, ReduceOp.Sum));
        var sent = Random(11, whole: false);
        Run(sent, 0, (b, _) => c.Broadcast(b, 11, world - 1));
        c.Barrier();
        return [.. all];
    }

    // A topology of one machine per entry of `perNode`, every device the CPU.
    private static ClusterTopology RealTopology(int[] perNode, string? inter = null, string? firstAddress = null) =>
        ClusterTopology.Create([.. perNode.Select((n, i) => new ClusterNodeSpec(i == 0 ? "box" : $"box{i + 1}", i == 0 ? firstAddress : null, [.. Enumerable.Repeat(new ClusterDeviceSpec("cpu", null), n)]))],
            null, perNode.Length > 1 ? inter ?? "ethernet-1g" : null);

    private static ClusterOptions RealOptions(TimeSpan timeout, SimulationMode mode = SimulationMode.InProcess) => new()
    {
        Coordinator = "127.0.0.1:0", RunId = "test-" + Guid.NewGuid().ToString("N"), Timeout = timeout, SharedMemoryBytes = 64 << 10,
        Retries = 5, RetryDelay = TimeSpan.FromMilliseconds(50), Simulation = new SimulationOptions { Mode = mode },
    };

    // A group of `world` ranks of one machine joined by the transport named, each rank connecting on a thread of its own.
    private static (IReadOnlyList<ITransport> Transports, IClusterTopology Topology, ClusterOptions Options) RealGroup(string transport, int world, TimeSpan timeout)
    {
        var topology = RealTopology([world]);
        var options = RealOptions(timeout);
        var provider = Transports.Get(transport);
        return (ConnectAll(world, r => provider.Connect(new ClusterContext(topology, r, options))), topology, options);
    }

    // A group of `world` ranks on two machines of this one (a localhost simulation), routed per pair: shared-memory inside, tcp between.
    private static (IReadOnlyList<ITransport> Transports, IClusterTopology Topology, ClusterOptions Options) RoutedGroup(int world, TimeSpan timeout)
    {
        var topology = RealTopology(world == 2 ? [1, 1] : [(world + 1) / 2, world / 2]);
        var options = RealOptions(timeout, SimulationMode.Localhost);
        var plan = TransportPlan.Make(topology, options);
        return (ConnectAll(world, r => plan.Connect(new ClusterContext(topology, r, options))), topology, options);
    }

    private static ITransport[] ConnectAll(int world, Func<int, ITransport> connect)
    {
        var connecting = Enumerable.Range(0, world).Select(r => Task.Factory.StartNew(() => connect(r), TaskCreationOptions.LongRunning)).ToArray();
        try
        {
            Task.WaitAll(connecting);
        }
        catch
        {
            foreach (var task in connecting.Where(t => t.IsCompletedSuccessfully))
            {
                task.Result.Dispose();
            }

            throw;
        }

        return [.. connecting.Select(t => t.Result)];
    }

    // Collectives that own their transport (the suites dispose the collectives only).
    private sealed class OwningCollectives(ICollectives inner, ITransport transport) : ICollectives
    {
        public string Name => inner.Name;

        public int Rank => inner.Rank;

        public int WorldSize => inner.WorldSize;

        public Device Device => inner.Device;

        public void AllReduce(Storage buffer, int count, ReduceOp op, WireType wire = WireType.Float32) => inner.AllReduce(buffer, count, op, wire);

        public void AllGather(Storage input, int count, Storage output) => inner.AllGather(input, count, output);

        public void ReduceScatter(Storage input, int count, Storage output, ReduceOp op) => inner.ReduceScatter(input, count, output, op);

        public void Broadcast(Storage buffer, int count, int root) => inner.Broadcast(buffer, count, root);

        public void Barrier() => inner.Barrier();

        public void Dispose()
        {
            inner.Dispose();
            transport.Dispose();
        }
    }

    private static IReadOnlyList<ICollectives> RealCollectives(string algorithm, (IReadOnlyList<ITransport> Transports, IClusterTopology Topology, ClusterOptions Options) group, Device device)
    {
        var provider = Collectives.Get(algorithm);
        return [.. group.Transports.Select((t, r) => (ICollectives)new OwningCollectives(
            provider.Create(t, SimulatedDevices.Carve(device, null, r), new ClusterContext(group.Topology, r, group.Options)), t))];
    }

    private static void ClusterTransportsRegistered(Device device)
    {
        if (!FirstRun(nameof(ClusterTransportsRegistered)))
        {
            return;
        }

        foreach (string name in new[] { "shared-memory", "pcie-host", "nvlink", "p2p", "direct-link", "tcp" })
        {
            Check(Transports.Find(name) is not null && Transports.Origin(name) == Overrides.Library, $"Idrak.Network registers {name} as a library default ({string.Join(", ", Transports.Names)})");
        }

        Check(Transports.Get("direct-link").Carrier == "tcp" && Transports.Get("tcp").Carrier == "tcp" && Transports.Get("shared-memory").Carrier == "shared-memory",
            "direct-link rides on the tcp carrier; shared-memory is its own");
        var two = RealTopology([2, 1]);
        var local = RealOptions(TimeSpan.FromSeconds(5), SimulationMode.Localhost);
        var answers = Transports.All.Select(t => (t.Name, Same: t.Availability(new ClusterContext(two, 0, local), 1), Across: t.Availability(new ClusterContext(two, 0, local), 2)))
            .ToDictionary(a => a.Name);
        Check(answers["shared-memory"].Same.Available && !answers["shared-memory"].Across.Available && answers["shared-memory"].Across.Reason.Contains("different machines", StringComparison.Ordinal),
            $"shared-memory joins one machine's ranks: {answers["shared-memory"].Same.Reason} / {answers["shared-memory"].Across.Reason}");
        Check(answers["tcp"].Same.Available && answers["tcp"].Across.Available, $"tcp joins every pair of a localhost run: {answers["tcp"].Across.Reason}");
        Check(!answers["pcie-host"].Same.Available && answers["pcie-host"].Same.Reason.Contains("separate processes", StringComparison.Ordinal)
              && !answers["nvlink"].Same.Available && !answers["sim"].Same.Available, "the one-process transports refuse ranks that are processes of their own");
        Check(!answers["direct-link"].Across.Available, $"no cable in the topology: {answers["direct-link"].Across.Reason}");
        var plan = TransportPlan.Make(two, local);
        Check(plan[0, 1].Transport == "shared-memory" && plan[0, 2].Transport == "tcp" && plan[1, 2].Transport == "tcp" && plan.Single is null,
            $"fastest per pair: shared-memory inside a node, tcp between: {plan[0, 1]}, {plan[0, 2]}");

        // On machines of their own (the coordinator off loopback): tcp between them needs the key; a cable is taken where the file has one.
        var lan = ClusterTopology.Create([new ClusterNodeSpec("desk", "192.0.2.10:29500", [new ClusterDeviceSpec("cpu", null)]), new ClusterNodeSpec("mini", "192.0.2.11:29500", [new ClusterDeviceSpec("cpu", null)]),
            new ClusterNodeSpec("old", "192.0.2.12:29500", [new ClusterDeviceSpec("cpu", null)])], null, "ethernet-1g", [new ClusterLinkOverride("desk", "mini", "direct-link-10g")]);
        var away = new ClusterOptions { Simulation = new SimulationOptions { Mode = SimulationMode.Localhost } };
        var keyless = Transports.Get("tcp").Availability(new ClusterContext(lan, 0, away), 2);
        Check(!keyless.Available && keyless.Reason.Contains("IDRAK_CLUSTER_KEY", StringComparison.Ordinal), $"no key, no tcp between machines: {keyless.Reason}");
        var keyed = away with { Key = ClusterKey.FromText("a shared secret of sixteen") };
        var cable = Transports.Get("direct-link").Availability(new ClusterContext(lan, 0, keyed), 1);
        var noCable = Transports.Get("direct-link").Availability(new ClusterContext(lan, 0, keyed), 2);
        Check(cable.Available && cable.Reason.Contains("192.0.2.11", StringComparison.Ordinal) && !noCable.Available, $"direct-link where the file has a cable: {cable.Reason} / {noCable.Reason}");
        var lanPlan = TransportPlan.Make(lan, keyed);
        Check(lanPlan[0, 1].Transport == "direct-link" && lanPlan[0, 2].Transport == "tcp" && lanPlan[1, 2].Transport == "tcp",
            $"the cable for its pair, tcp for the rest: {lanPlan[0, 1]}, {lanPlan[0, 2]}");
    }

    private static void ClusterRealTransportSuite(Device device)
    {
        if (!FirstRun(nameof(ClusterRealTransportSuite)))
        {
            return;
        }

        foreach (string name in new[] { "shared-memory", "tcp", "pcie-host", "nvlink" })
        {
            Conformance.CheckTransport(new TransportUnderTest(name, (w, t) => RealGroup(name, w, t).Transports), new ContractCheckOptions { RandomCases = 1 }).ThrowIfFailed();
        }

        Conformance.CheckTransport(new TransportUnderTest("routed shared-memory+tcp", (w, t) => RoutedGroup(w, t).Transports), new ContractCheckOptions { RandomCases = 1 }).ThrowIfFailed();
    }

    private static void ClusterRealCollectivesSuite(Device device)
    {
        foreach (var (algorithm, transport) in new[] { ("ring", "tcp"), ("ring", "shared-memory"), ("naive", "tcp") })
        {
            Conformance.CheckCollectives(new CollectivesUnderTest($"{algorithm} over {transport}", (w, t) => RealCollectives(algorithm, RealGroup(transport, w, t), device)),
                new ContractCheckOptions { RandomCases = 1 }).ThrowIfFailed();
        }

        Conformance.CheckCollectives(new CollectivesUnderTest("hierarchical over shared-memory+tcp", (w, t) => RealCollectives("hierarchical", RoutedGroup(w, t), device)),
            new ContractCheckOptions { RandomCases = 1 }).ThrowIfFailed();
    }

    private static void ClusterRendezvousRetries(Device device)
    {
        if (!FirstRun(nameof(ClusterRendezvousRetries)))
        {
            return;
        }

        int port;
        using (var probe = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp))
        {
            probe.Bind(new IPEndPoint(IPAddress.Loopback, 0));
            port = ((IPEndPoint)probe.LocalEndPoint!).Port;              // free, and nobody listens once closed
        }

        var topology = RealTopology([2]);
        var options = RealOptions(TimeSpan.FromSeconds(5)) with { Coordinator = $"127.0.0.1:{port}", Retries = 3, RetryDelay = TimeSpan.FromMilliseconds(50) };
        // How long one refused connect takes here: at once on Linux, about 2 s on Windows (it resends the SYN before giving
        // up). Each attempt must cost that, not the 5 s timeout; the waits between them are 50 and 100 ms.
        var refusal = Stopwatch.StartNew();
        using (var once = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp))
        {
            try
            {
                once.Connect(new IPEndPoint(IPAddress.Loopback, port));
            }
            catch (SocketException)
            {
                // refused, as expected
            }
        }

        var refused = refusal.Elapsed;
        var watch = Stopwatch.StartNew();
        ThrowsNaming<InvalidOperationException>(() => Transports.Get("tcp").Connect(new ClusterContext(topology, 1, options)).Dispose(), "a coordinator nobody listens at",
            "rank 1", $"127.0.0.1:{port}", "after 3 attempt(s)", "IDRAK_CLUSTER_RETRIES");
        var bound = (refused * 3 * 1.5) + TimeSpan.FromMilliseconds(150) + TimeSpan.FromSeconds(1);
        Check(watch.Elapsed < bound && bound < TimeSpan.FromSeconds(15),
            $"three attempts, the wait doubling from 50 ms: {watch.Elapsed.TotalSeconds:F1} s against {bound.TotalSeconds:F1} s (a refused connect takes {refused.TotalMilliseconds:F0} ms here)");
    }

    private static void ClusterWrongKey(Device device)
    {
        if (!FirstRun(nameof(ClusterWrongKey)))
        {
            return;
        }

        const string first = "the first secret of the owner", second = "the second secret, not the first";
        var topology = RealTopology([2]);
        foreach (string transport in new[] { "tcp", "shared-memory" })
        {
            var options = RealOptions(TimeSpan.FromSeconds(3));
            var errors = new Exception?[2];
            var threads = Enumerable.Range(0, 2).Select(r => new Thread(() =>
            {
                try
                {
                    Transports.Get(transport).Connect(new ClusterContext(topology, r, options with { Key = ClusterKey.FromText(r == 0 ? first : second) })).Dispose();
                }
                catch (Exception e)
                {
                    errors[r] = e;
                }
            }) { IsBackground = true }).ToArray();
            foreach (var thread in threads)
            {
                thread.Start();
            }

            foreach (var thread in threads)
            {
                Check(thread.Join(TimeSpan.FromSeconds(30)), $"{transport}: both ranks end");
            }

            Check(errors.All(e => e is not null), $"{transport}: neither rank joins a group whose hello failed: {string.Join("; ", errors.Select(e => e?.Message ?? "joined"))}");
            Check(errors.Any(e => e!.Message.Contains("does not prove the cluster key", StringComparison.Ordinal)), $"{transport}: the refusal says why: {string.Join("; ", errors.Select(e => e!.Message))}");
            Check(errors.All(e => !e!.ToString().Contains("secret", StringComparison.Ordinal)), $"{transport}: the key is never in a message");
        }

        // Without a key nothing listens beyond loopback: a coordinator off loopback is refused before any socket opens.
        var lan = RealOptions(TimeSpan.FromSeconds(3)) with { Coordinator = "192.0.2.10:29500" };
        ThrowsNaming<InvalidOperationException>(() => Transports.Get("tcp").Connect(new ClusterContext(topology, 0, lan)).Dispose(), "a keyless coordinator off loopback",
            "IDRAK_CLUSTER_KEY", "not a loopback address");
    }

    private static void ClusterRealRankDies(Device device)
    {
        if (!FirstRun(nameof(ClusterRealRankDies)))
        {
            return;
        }

        foreach (string transport in new[] { "tcp", "shared-memory" })
        {
            string folder = TempFolder("cluster-die");
            try
            {
                string path = Path.Combine(folder, "two.json");
                File.WriteAllText(path, RealTopology([2]).ToJson());
                var (file, prefix) = LocalhostCluster.ProgramOf(typeof(Tests).Assembly);
                var launch = new LocalhostLaunch(file, [.. prefix, "--cluster-child", "die", folder])
                {
                    Environment = new Dictionary<string, string> { ["IDRAK_CLUSTER_TRANSPORT"] = transport, ["IDRAK_CLUSTER_TIMEOUT"] = "60" },
                    Limit = TimeSpan.FromSeconds(90),
                };
                var watch = Stopwatch.StartNew();
                var result = LocalhostCluster.Run(path, new ClusterOptions { Transport = transport, Timeout = TimeSpan.FromSeconds(30) }, launch);
                string error = File.Exists(Path.Combine(folder, "rank-0-error.txt")) ? File.ReadAllText(Path.Combine(folder, "rank-0-error.txt")) : "(rank 0 wrote nothing)";
                Check(error.Contains("rank 1", StringComparison.Ordinal) && !error.Contains("TimeoutException", StringComparison.Ordinal),
                    $"{transport}: rank 0 is told rank 1 is gone, not after a timeout: {error}");
                Check(watch.Elapsed < TimeSpan.FromSeconds(50), $"{transport}: at once ({watch.Elapsed.TotalSeconds:F1} s with a 60 s timeout)");
                Check(result.Succeeded, $"{transport}: both ranks ended by themselves: {string.Join(", ", result.Ranks.Select(r => $"{r.Rank}:{r.ExitCode}"))}");
            }
            finally
            {
                Directory.Delete(folder, recursive: true);
            }
        }
    }

    private static void ClusterCopyTo(Device device)
    {
        var random = new Random(5);
        float[] values = [.. Enumerable.Range(0, 1_031).Select(_ => (float)random.NextDouble())];
        var one = SimulatedDevices.Carve(device, null, 950);
        var two = SimulatedDevices.Carve(device, null, 951);
        var pairs = new List<(string What, Backend From, Backend To)> { ("inside the device", device.Backend, device.Backend), ("between two backends of the device", one.Backend, two.Backend) };
        if (device.Type != DeviceType.Cpu)
        {
            pairs.Add(("from the device to the CPU", device.Backend, Device.Cpu.Backend));
            pairs.Add(("from the CPU to the device", Device.Cpu.Backend, device.Backend));
        }

        foreach (var (what, from, to) in pairs)
        {
            foreach (int count in new[] { 0, 1, 1_000 })
            {
                var source = from.Allocate(values.Length, zeroed: false);
                var copied = to.Allocate(values.Length, zeroed: true);
                var staged = to.Allocate(values.Length, zeroed: true);
                try
                {
                    from.Upload(values, source);
                    from.CopyTo(source, copied, count);
                    // The same through the host: download, then upload.
                    var host = new float[count];
                    from.DownloadRange(source, 0, host);
                    to.Upload(host, staged);
                    var a = new float[values.Length];
                    var b = new float[values.Length];
                    to.Download(copied, a);
                    to.Download(staged, b);
                    Check(a.AsSpan().SequenceEqual(b), $"CopyTo {what} of {count} values equals a download then an upload, values past the count untouched");
                }
                finally
                {
                    source.Release();
                    copied.Release();
                    staged.Release();
                }
            }
        }

        var cpu = Device.Cpu.Backend.Allocate(4, zeroed: true);
        try
        {
            ThrowsNaming<ArgumentException>(() => one.Backend.CopyTo(cpu, cpu, 4), "a source on another backend", "not on this backend");
        }
        finally
        {
            cpu.Release();
        }
    }

    private static void ClusterNvlinkFallback(Device device)
    {
        var events = new NetworkEvents();
        var (group, _, _) = RealGroup("nvlink", 2, TimeSpan.FromSeconds(20));
        var hosted = RealGroup("pcie-host", 2, TimeSpan.FromSeconds(20));
        var devices = new[] { SimulatedDevices.Carve(device, null, 960), SimulatedDevices.Carve(device, null, 961) };
        bool direct = devices[0].Backend.CopiesDirectlyTo(devices[1].Backend);
        try
        {
            float[] values = [.. Enumerable.Range(0, 300).Select(i => i * 0.5f)];
            using (Telemetry.Subscribe(events))
            {
                foreach (var transports in new[] { group, hosted.Transports })
                {
                    for (int round = 0; round < 3; round++)
                    {
                        var source = devices[0].Backend.Allocate(values.Length, zeroed: false);
                        var target = devices[1].Backend.Allocate(values.Length, zeroed: true);
                        try
                        {
                            devices[0].Backend.Upload(values, source);
                            var sender = Task.Factory.StartNew(() => ((IDeviceTransport)transports[0]).SendDevice(1, source, values.Length, 5), TaskCreationOptions.LongRunning);
                            ((IDeviceTransport)transports[1]).ReceiveDevice(0, target, values.Length, 5);
                            Check(sender.Wait(TimeSpan.FromSeconds(20)), "the sender returns once the receiver copied");
                            var got = new float[values.Length];
                            devices[1].Backend.Download(target, got);
                            Check(got.AsSpan().SequenceEqual(values), $"{transports[0].Name}: device memory handed over, round {round}");
                        }
                        finally
                        {
                            source.Release();
                            target.Release();
                        }
                    }
                }
            }

            var fallbacks = events.Chosen.Where(c => c.Transport == "pcie-host").ToList();
            if (direct)
            {
                Check(fallbacks.Count == 0, $"{device}: two backends of one device copy directly, no fallback: {fallbacks.Count}");
                Console.WriteLine($"    {device}: two backends of one device copied device to device; the peer path between two cards needs two of them");
            }
            else
            {
                Check(fallbacks is [{ RankA: 0, RankB: 1 } only] && only.Reason.Contains("no peer access", StringComparison.Ordinal),
                    $"{device}: refused peers fall back through host memory, said once for the pair: {string.Join("; ", fallbacks)}");
            }

            // Two physical GPUs of one kind, where this machine has them: the peer path or its fallback, as the driver says.
            var gpus = Device.Available.Where(d => d.Type == device.Type && d.Type != DeviceType.Cpu).ToList();
            if (gpus.Count < 2)
            {
                Console.WriteLine(device.Type == DeviceType.Cpu
                    ? "    skipped: copies between two physical GPUs (a GPU run of this test checks them where a machine has two)"
                    : $"    skipped: copies between two physical {device.Type} devices (this machine lists {gpus.Count})");
            }
            else
            {
                var source = gpus[0].Backend.Allocate(values.Length, zeroed: false);
                var target = gpus[1].Backend.Allocate(values.Length, zeroed: true);
                try
                {
                    gpus[0].Backend.Upload(values, source);
                    gpus[0].Backend.CopyTo(source, target, values.Length);
                    var got = new float[values.Length];
                    gpus[1].Backend.Download(target, got);
                    Check(got.AsSpan().SequenceEqual(values), $"{gpus[0]} to {gpus[1]}: {(gpus[0].Backend.CopiesDirectlyTo(gpus[1].Backend) ? "a peer copy" : "through host memory")}");
                    Console.WriteLine($"    {gpus[0]} to {gpus[1]}: {(gpus[0].Backend.CopiesDirectlyTo(gpus[1].Backend) ? "peer access" : "no peer access, through host memory")}");
                }
                finally
                {
                    source.Release();
                    target.Release();
                }
            }
        }
        finally
        {
            foreach (var transport in group.Concat(hosted.Transports))
            {
                transport.Dispose();
            }
        }
    }

    private static void ClusterChildProcesses(Device device)
    {
        if (!FirstRun(nameof(ClusterChildProcesses)))
        {
            return;
        }

        var cases = new (string What, int[] Nodes, string? Forced, string Plan)[]
        {
            ("2 ranks on one machine", [2], null, "shared-memory"),
            ("3 ranks forced onto tcp", [3], "tcp", "tcp"),
            ("3 ranks on two machines", [2, 1], null, "shared-memory+tcp"),
        };
        foreach (var (what, nodes, forced, expected) in cases)
        {
            string folder = TempFolder("cluster-children");
            try
            {
                var topology = RealTopology(nodes);
                string path = Path.Combine(folder, "topology.json");
                File.WriteAllText(path, topology.ToJson());
                var (file, prefix) = LocalhostCluster.ProgramOf(typeof(Tests).Assembly);
                var environment = new Dictionary<string, string> { ["IDRAK_CLUSTER_TIMEOUT"] = "60", ["IDRAK_CLUSTER_SHM_BYTES"] = "64KB" };
                if (forced is not null)
                {
                    environment["IDRAK_CLUSTER_TRANSPORT"] = forced;
                }

                var launch = new LocalhostLaunch(file, [.. prefix, "--cluster-child", "suite", folder]) { Environment = environment, Limit = TimeSpan.FromSeconds(120) };
                var result = LocalhostCluster.Run(path, new ClusterOptions { Transport = forced ?? "auto", Timeout = TimeSpan.FromSeconds(60) }, launch);
                Check(result.Succeeded, $"{what}: every rank's exchange and collectives completed: {result.Failure} "
                                        + string.Join("; ", result.Ranks.Select(r => $"rank {r.Rank} exit {r.ExitCode}: {string.Join(" | ", r.Errors.TakeLast(3))}")));
                int world = topology.WorldSize;
                for (int r = 0; r < world; r++)
                {
                    string plan = File.ReadAllText(Path.Combine(folder, $"plan-{r}.txt"));
                    Check(plan == expected, $"{what}: rank {r} joined over {plan}, {expected} was due");
                }

                var reference = new float[world][];
                InProcessCluster.Run(topology, new ClusterOptions { Timeout = TimeSpan.FromSeconds(60) }, rank => reference[rank.Rank] = ClusterWork(rank)).ThrowIfFailed();
                for (int r = 0; r < world; r++)
                {
                    var bytes = File.ReadAllBytes(Path.Combine(folder, $"rank-{r}.bin"));
                    Check(bytes.AsSpan().SequenceEqual(MemoryMarshal.AsBytes(reference[r].AsSpan())),
                        $"{what}: rank {r}'s results ({bytes.Length / sizeof(float)} values) are bit-identical to the in-process run's ({reference[r].Length})");
                }
            }
            finally
            {
                Directory.Delete(folder, recursive: true);
            }
        }
    }

    private static (int Code, string Output, string Error) IdrakCli(params string[] args)
    {
        using var output = new StringWriter(CultureInfo.InvariantCulture);
        using var error = new StringWriter(CultureInfo.InvariantCulture);
        int code = CommandLine.Run(args, output, error);
        return (code, output.ToString(), error.ToString());
    }

    private static void ClusterThrottle(Device device)
    {
        if (!FirstRun(nameof(ClusterThrottle)))
        {
            return;
        }

        string folder = TempFolder("cluster-throttle");
        try
        {
            string path = Path.Combine(folder, "two.json");
            File.WriteAllText(path, RealTopology([1, 1], "ethernet-200m").ToJson());
            var (code, text, error) = IdrakCli("cluster", "bench", "--topology", path, "--localhost", "--bytes", "4MB", "--rounds", "3", "-j");
            Check(code == ExitCodes.Ok, $"cluster bench ran: exit {code}: {error}\n{text}");
            var report = JsonNode.Parse(text)!.AsObject();
            var row = report["rows"]!.AsArray().Single()!.AsObject();
            double warm = (double)row["warm_bytes_per_second"]!, cold = (double)row["cold_bytes_per_second"]!, profile = 200e6 / 8;
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"    throttled to ethernet-200m ({profile / 1e6:F1} MB/s) over {(string)row["transport"]!}: warm {warm / 1e6:F2} MB/s, cold {cold / 1e6:F2} MB/s, warm latency {(double)row["warm_latency_seconds"]! * 1e6:F0} us"));
            Check((bool)report["throttled"]! && (string)row["transport"]! == "tcp", $"the pair between two machines of the file runs on tcp, paced: {row}");
            Check(Math.Abs(warm / profile - 1) < 0.2, $"the warm bandwidth {warm / 1e6:F2} MB/s is within 20% of the profile's {profile / 1e6:F1} MB/s");
            Check(row["cold_latency_seconds"] is not null && row["warm_latency_seconds"] is not null && row["cold_bytes_per_second"] is not null, "cold and warm reported apart");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void ClusterLocalhostWeights(Device device)
    {
        if (!FirstRun(nameof(ClusterLocalhostWeights)))
        {
            return;
        }

        string folder = TempFolder("cluster-localhost");
        try
        {
            string topology = Path.Combine(folder, "two.json");
            File.WriteAllText(topology, RealTopology([2]).ToJson());
            string[] Pretrain(string run) => ["--", "pretrain", Path.GetFullPath(TinyLlama), "--tokens", Path.GetFullPath(PretrainData("tokens.bin")), "--steps", "3",
                "--max-length", "32", "--batch-tokens", "64", "--optimizer", "sgd", "--lr", "0.2", "--no-graphs", "-o", run];
            string inProcess = Path.Combine(folder, "in-process"), localhost = Path.Combine(folder, "localhost");
            var (code, text, error) = IdrakCli(["cluster", "sim", "--topology", topology, .. Pretrain(inProcess)]);
            Check(code == ExitCodes.Ok, $"the in-process run: exit {code}: {error}\n{text}");
            (code, text, error) = IdrakCli(["cluster", "sim", "--topology", topology, "--mode", "localhost", .. Pretrain(localhost)]);
            Check(code == ExitCodes.Ok, $"the localhost run: exit {code}: {error}\n{text}");
            Check(text.Contains("as processes on this machine", StringComparison.Ordinal) && text.Contains("rank 1 done", StringComparison.Ordinal), $"the ranks were processes: {text}");
            string Weights(string run) => Path.Combine(Pretrainer.LatestCheckpoint(run) ?? throw new InvalidOperationException($"{run} has no checkpoint"), "model.safetensors");
            var a = ReadTensors(Weights(inProcess));
            var b = ReadTensors(Weights(localhost));
            Check(a.Count > 0 && a.Keys.Order().SequenceEqual(b.Keys.Order()) && a.All(t => t.Value.Values.AsSpan().SequenceEqual(b[t.Key].Values)),
                $"the {a.Count} tensors after 3 steps are bit-identical (largest gap {a.Max(t => Gap(t.Value.Values, b[t.Key].Values)):G3})");

            // Rank 0's log: each step's summary of every rank (the ranks' telemetry is their own process's).
            var summaries = File.ReadAllLines(Path.Combine(localhost, "log.jsonl")).Where(l => l.Contains("step-summary", StringComparison.Ordinal))
                .Select(l => JsonNode.Parse(l)!).ToList();
            var ranks = summaries.Select(s => FindRank(s)).ToList();
            Check(summaries.Count >= 2 * 3 && ranks.Count(r => r == 0) == ranks.Count(r => r == 1) && ranks.Contains(1),
                $"rank 0's log holds every rank's per-step summary: {summaries.Count} line(s), ranks {string.Join(",", ranks)}");
            Check(!File.ReadAllText(Path.Combine(inProcess, "log.jsonl")).Contains("step-summary", StringComparison.Ordinal), "an in-process run needs none (one telemetry)");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }

        static int? FindRank(JsonNode node) => (int?)node["rank"];
    }

    private static void ClusterCliStart(Device device)
    {
        if (!FirstRun(nameof(ClusterCliStart)))
        {
            return;
        }

        string folder = TempFolder("cluster-start");
        try
        {
            // Two machines, this host the first with two ranks; the second never starts here.
            var topology = ClusterTopology.Create([new ClusterNodeSpec(Environment.MachineName, "127.0.0.1:29555", [new ClusterDeviceSpec("cpu", null), new ClusterDeviceSpec("cpu", null)]),
                new ClusterNodeSpec("elsewhere", null, [new ClusterDeviceSpec("cpu", null)])], null, "ethernet-1g");
            string path = Path.Combine(folder, "lan.json");
            File.WriteAllText(path, topology.ToJson());
            var (file, prefix) = LocalhostCluster.ProgramOf(typeof(Tests).Assembly);
            var (code, text, error) = IdrakCli(["cluster", "start", "--topology", path, "--", file, .. prefix, "--cluster-child", "env", folder]);
            Check(code == ExitCodes.Ok, $"cluster start ran this host's node: exit {code}: {error}\n{text}");
            Check(text.Contains("starting 2 rank(s) of node " + Environment.MachineName, StringComparison.Ordinal) && text.Contains("rank 1 (cpu): done", StringComparison.Ordinal),
                $"one process per rank of this host's node: {text}");
            var runs = new HashSet<string>();
            for (int r = 0; r < 2; r++)
            {
                var env = File.ReadAllLines(Path.Combine(folder, $"env-{r}.txt")).Select(l => l.Split('=', 2)).ToDictionary(p => p[0], p => p[1]);
                Check(env["IDRAK_RANK"] == r.ToString(CultureInfo.InvariantCulture) && env["IDRAK_WORLD_SIZE"] == "3" && env["IDRAK_CLUSTER_NODE"] == Environment.MachineName
                      && env["IDRAK_CLUSTER_DEVICE"] == "cpu" && env["IDRAK_COORDINATOR"] == "127.0.0.1:29555" && env["IDRAK_CLUSTER_TOPOLOGY"] == Path.GetFullPath(path),
                    $"rank {r} was given its rank, the world size, its node, device, coordinator and topology: {string.Join(", ", env)}");
                runs.Add(env["IDRAK_CLUSTER_RUN"]);
            }

            Check(runs.Count == 1 && runs.Single().Length > 0, "the ranks share one run name");
            var (missing, _, missingError) = IdrakCli(["cluster", "start", "--topology", path, "--node", "nowhere", "--", file]);
            Check(missing == ExitCodes.Usage && missingError.Contains("--node nowhere", StringComparison.Ordinal), $"an unknown node is a usage error naming it: {missingError}");

            // One rank fails: the other is ended at once, and no process is left.
            var watch = Stopwatch.StartNew();
            var (failed, failedText, failedError) = IdrakCli(["cluster", "start", "--topology", path, "--", file, .. prefix, "--cluster-child", "fail-one", folder]);
            Check(failed == 7, $"the failed rank's exit code is the command's: {failed}: {failedError}");
            Check(watch.Elapsed < TimeSpan.FromSeconds(40), $"the other rank was ended at once ({watch.Elapsed.TotalSeconds:F1} s; it would have slept 60 s)");
            Check(failedText.Contains("rank 1 (cpu): exit code 7", StringComparison.Ordinal) && failedText.Contains("rank 0 (cpu): ended by the launcher", StringComparison.Ordinal)
                  && failedError.Contains("rank 1 ended with exit code 7", StringComparison.Ordinal), $"each rank's end is reported: {failedText} / {failedError}");
            foreach (int r in new[] { 0, 1 })
            {
                int pid = int.Parse(File.ReadAllText(Path.Combine(folder, $"pid-{r}.txt")), CultureInfo.InvariantCulture);
                bool alive;
                try
                {
                    using var process = Process.GetProcessById(pid);
                    alive = !process.HasExited;
                }
                catch (ArgumentException)
                {
                    alive = false;
                }

                Check(!alive, $"rank {r}'s process ({pid}) does not outlive the command");
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }
}
