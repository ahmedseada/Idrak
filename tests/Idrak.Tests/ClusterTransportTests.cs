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
using Idrak.Text;

// Plan 23, phase C5: the real transports made directly (shared-memory, tcp with its rendezvous and keyed hello, pcie-host,
// nvlink) and idrak cluster start. Child processes are this test program again (--cluster-child); every one is ended by
// the launcher's finally, and each watches its launcher's process.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] ClusterTransportGroup =
    [
        ("cluster: shared-memory, tcp, pcie-host and nvlink, made directly, pass the kit's TransportSuite in one process (2 to 8 ranks, messages larger than a shared-memory ring)", ClusterRealTransportSuite),
        ("cluster: tcp rendezvous retries then fails naming the rank and address, each attempt as long as one refused connect on this machine, not the timeout", ClusterRendezvousRetries),
        ("cluster: a hello with a wrong key is refused on tcp and shared-memory, the key never in a message; nothing listens off loopback without a key", ClusterWrongKey),
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

    // A topology of one machine per entry of `perNode`, every device the CPU.
    private static ClusterTopology RealTopology(int[] perNode, string? inter = null, string? firstAddress = null) =>
        ClusterTopology.Create([.. perNode.Select((n, i) => new ClusterNodeSpec(i == 0 ? "box" : $"box{i + 1}", i == 0 ? firstAddress : null, [.. Enumerable.Repeat(new ClusterDeviceSpec("cpu", null), n)]))],
            null, perNode.Length > 1 ? inter ?? "ethernet-1g" : null);

    private static ClusterOptions RealOptions(TimeSpan timeout, SimulationMode mode = SimulationMode.InProcess) => new()
    {
        Coordinator = "127.0.0.1:0", RunId = "test-" + Guid.NewGuid().ToString("N"), Timeout = timeout, SharedMemoryBytes = 64 << 10,
        Retries = 5, RetryDelay = TimeSpan.FromMilliseconds(50), Simulation = new SimulationOptions { Mode = mode },
    };

    // A group of `world` ranks of one machine joined by the transport given, each rank connecting on a thread of its own.
    private static (IReadOnlyList<ITransport> Transports, IClusterTopology Topology, ClusterOptions Options) RealGroup(ITransportProvider provider, int world, TimeSpan timeout)
    {
        var topology = RealTopology([world]);
        var options = RealOptions(timeout);
        return (ConnectAll(world, r => provider.Connect(new ClusterContext(topology, r, options))), topology, options);
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

    private static void ClusterRealTransportSuite(Device device)
    {
        if (!FirstRun(nameof(ClusterRealTransportSuite)))
        {
            return;
        }

        foreach (var provider in new ITransportProvider[] { new SharedMemoryProvider(), new TcpProvider(), new PciHostProvider(), new NvlinkProvider() })
        {
            Conformance.CheckTransport(new TransportUnderTest(provider.Name, (w, t) => RealGroup(provider, w, t).Transports), new ContractCheckOptions { RandomCases = 1 }).ThrowIfFailed();
        }
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
        ThrowsNaming<InvalidOperationException>(() => new TcpProvider().Connect(new ClusterContext(topology, 1, options)).Dispose(), "a coordinator nobody listens at",
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
        foreach (var transport in new ITransportProvider[] { new TcpProvider(), new SharedMemoryProvider() })
        {
            var options = RealOptions(TimeSpan.FromSeconds(3));
            var errors = new Exception?[2];
            var threads = Enumerable.Range(0, 2).Select(r => new Thread(() =>
            {
                try
                {
                    transport.Connect(new ClusterContext(topology, r, options with { Key = ClusterKey.FromText(r == 0 ? first : second) })).Dispose();
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
                Check(thread.Join(TimeSpan.FromSeconds(30)), $"{transport.Name}: both ranks end");
            }

            Check(errors.All(e => e is not null), $"{transport.Name}: neither rank joins a group whose hello failed: {string.Join("; ", errors.Select(e => e?.Message ?? "joined"))}");
            Check(errors.Any(e => e!.Message.Contains("does not prove the cluster key", StringComparison.Ordinal)), $"{transport.Name}: the refusal says why: {string.Join("; ", errors.Select(e => e!.Message))}");
            Check(errors.All(e => !e!.ToString().Contains("secret", StringComparison.Ordinal)), $"{transport.Name}: the key is never in a message");
        }

        // Without a key nothing listens beyond loopback: a coordinator off loopback is refused before any socket opens.
        var lan = RealOptions(TimeSpan.FromSeconds(3)) with { Coordinator = "192.0.2.10:29500" };
        ThrowsNaming<InvalidOperationException>(() => new TcpProvider().Connect(new ClusterContext(topology, 0, lan)).Dispose(), "a keyless coordinator off loopback",
            "IDRAK_CLUSTER_KEY", "not a loopback address");
    }

    private static (int Code, string Output, string Error) IdrakCli(params string[] args)
    {
        using var output = new StringWriter(CultureInfo.InvariantCulture);
        using var error = new StringWriter(CultureInfo.InvariantCulture);
        int code = CommandLine.Run(args, output, error);
        return (code, output.ToString(), error.ToString());
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
            var (file, prefix) = LocalhostCluster.ProgramOf(typeof(DesktopTests).Assembly);
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
