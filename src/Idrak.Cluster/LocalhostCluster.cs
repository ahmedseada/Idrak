// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Reflection;
using System.Text;

namespace Idrak.Cluster;

/// <summary>A line a rank's process wrote.</summary>
/// <param name="Rank">The rank.</param>
/// <param name="Text">The line, without its end.</param>
/// <param name="IsError">Whether it came on the error output.</param>
public readonly record struct LocalhostLine(int Rank, string Text, bool IsError);

/// <summary>The program every rank's process runs, and how.</summary>
/// <param name="FileName">The program (<see cref="LocalhostCluster.ProgramOf"/> finds an assembly's).</param>
/// <param name="Arguments">Its arguments, the same for every rank: a rank reads which it is from its environment.</param>
public sealed record LocalhostLaunch(string FileName, IReadOnlyList<string> Arguments)
{
    /// <summary>More variables for every rank's process (beside the cluster's own, which the launcher sets).</summary>
    public IReadOnlyDictionary<string, string>? Environment { get; init; }

    /// <summary>The folder the processes start in, or null for this process's.</summary>
    public string? WorkingDirectory { get; init; }

    /// <summary>How long the whole run may take before every process is ended, or null for no limit.</summary>
    public TimeSpan? Limit { get; init; }

    /// <summary>The last lines of each output kept per rank for the result.</summary>
    public int KeepLines { get; init; } = 200;

    /// <summary>
    /// Steers the processes (pause, resume, stop at a step boundary; phase C9), or null: the launcher gives every rank its
    /// control file (<c>IDRAK_CLUSTER_CONTROL</c>), which rank 0 reads as each step begins.
    /// </summary>
    public LocalhostControl? Control { get; init; }
}

/// <summary>How one rank's process ended.</summary>
/// <param name="Rank">The rank.</param>
/// <param name="Device">The device it was given.</param>
/// <param name="ExitCode">Its exit code, or null when it was ended by the launcher (another rank failed, or the limit passed).</param>
/// <param name="Output">The last lines of its standard output.</param>
/// <param name="Errors">The last lines of its error output.</param>
public sealed record LocalhostRankResult(int Rank, string Device, int? ExitCode, IReadOnlyList<string> Output, IReadOnlyList<string> Errors);

/// <summary>What a run of rank processes gave.</summary>
/// <param name="Ranks">Each rank's end, in rank order.</param>
/// <param name="Wall">The time the run took.</param>
/// <param name="Failure">Why the run stopped early (a rank's failure, the limit), or null.</param>
public sealed record LocalhostRunResult(IReadOnlyList<LocalhostRankResult> Ranks, TimeSpan Wall, string? Failure)
{
    /// <summary>Whether every rank's process ended with exit code 0.</summary>
    public bool Succeeded => Failure is null && Ranks.All(r => r.ExitCode == 0);

    /// <summary>Throws naming why the run failed and each failed rank's last error line.</summary>
    /// <exception cref="InvalidOperationException">The run failed.</exception>
    public void ThrowIfFailed()
    {
        if (!Succeeded)
        {
            throw new InvalidOperationException((Failure ?? "a rank failed") + ": " + string.Join("; ", Ranks.Where(r => r.ExitCode != 0)
                .Select(r => $"rank {r.Rank} {(r.ExitCode is { } code ? $"exit {code}" : "ended")}" + (r.Errors.LastOrDefault(e => e.Length > 0) is { } last ? $" ({last})" : ""))));
        }
    }
}

/// <summary>This process as one rank of a cluster of processes (<see cref="LocalhostCluster.Join(ClusterOptions)"/>); disposing it leaves the group.</summary>
public sealed class LocalhostRank : IDisposable
{
    private bool _disposed;

    internal LocalhostRank(ClusterRank rank, TransportPlan plan)
    {
        Rank = rank;
        Plan = plan;
    }

    /// <summary>The rank: its device, transport (routed per pair, paced to the links when throttled) and collectives.</summary>
    public ClusterRank Rank { get; }

    /// <summary>The transport joining each pair.</summary>
    public TransportPlan Plan { get; }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        Rank.Collectives.Dispose();
        Rank.Transport.Dispose();
    }
}

/// <summary>
/// Runs a cluster as one process per rank on this machine (plan 23, phase C5): the real transports between processes
/// (shared memory inside a node, TCP between nodes, chosen per pair), the throttle optionally pacing them to the topology's
/// links (<see cref="SimulationOptions.Throttle"/>), so the code that runs is the code a real cluster runs. <see cref="Run"/>
/// starts the processes (every rank of a topology: a localhost simulation; one node's ranks: that machine's part of a
/// real cluster, <c>idrak cluster start</c>), each with its rank, the world size, its node, its device, the coordinator, the
/// run's name and the launcher's process id in its environment (the variables <see cref="ClusterOptions.FromEnvironment"/>
/// reads), streams their lines, ends them all when one fails, and kills any left in <c>finally</c>. A rank's process calls
/// <see cref="Join(ClusterOptions)"/>.
/// </summary>
public static class LocalhostCluster
{
    /// <summary>
    /// Starts one process per rank and waits for them: every rank of the topology at <paramref name="topologyPath"/> on
    /// this machine (a localhost simulation, the coordinator on loopback) when <paramref name="node"/> is null, else the
    /// ranks of that node only (a machine's part of a real cluster, the coordinator the file's or the options').
    /// </summary>
    /// <param name="topologyPath">The topology file.</param>
    /// <param name="options">The cluster options (null: the environment's); every rank also inherits this process's environment.</param>
    /// <param name="launch">The program each rank runs.</param>
    /// <param name="node">The node whose ranks to start, or null for every rank.</param>
    /// <param name="lines">Told each line a rank writes (from the processes' reading threads).</param>
    /// <param name="cancellationToken">Ends every rank.</param>
    /// <exception cref="NotSupportedException">No transport can join a pair (the plan is made before anything starts).</exception>
    public static LocalhostRunResult Run(string topologyPath, ClusterOptions? options, LocalhostLaunch launch, string? node = null, Action<LocalhostLine>? lines = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(topologyPath);
        ArgumentNullException.ThrowIfNull(launch);
        string path = Path.GetFullPath(topologyPath);
        var topology = ClusterTopology.Load(path);
        options ??= ClusterOptions.FromEnvironment();
        bool simulate = node is null;
        int nodeIndex = simulate ? -1 : topology.NodeOf(node!) ?? throw new ArgumentException($"The topology {path} has no node '{node}' (nodes: {string.Join(", ", topology.Nodes.Select(n => n.Name))}).", nameof(node));
        int[] ranks = simulate ? [.. Enumerable.Range(0, topology.WorldSize)] : [.. topology.Ranks(nodeIndex)];
        // A localhost run meets on a loopback port the system gives free now: rank 0 may meet its node's ranks over shared
        // memory before it listens, so every rank must know the port before any starts.
        string coordinator = simulate ? string.Create(CultureInfo.InvariantCulture, $"127.0.0.1:{FreePort()}") : options.Coordinator ?? topology.Coordinator
            ?? throw new InvalidOperationException($"The ranks of a real cluster meet at a coordinator: give the first node of {path} an address (host:port) or set IDRAK_COORDINATOR.");
        string run = Guid.NewGuid().ToString("N", CultureInfo.InvariantCulture);
        var planned = options with { Coordinator = coordinator, RunId = run, Simulation = options.Simulation with { Mode = SimulationMode.Localhost } };
        var plan = TransportPlan.Make(topology, planned);
        if (plan.Used.Any(t => Transports.Get(t).Capabilities.CrossMachine) && coordinator.EndsWith(":0", StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"The coordinator {coordinator} has no port: give the first node's address a port (host:port) so every machine meets rank 0 there.");
        }

        var results = new (int? Code, Queue<string> Output, Queue<string> Errors)[topology.WorldSize];
        for (int r = 0; r < results.Length; r++)
        {
            results[r] = (null, new Queue<string>(), new Queue<string>());
        }

        var processes = new Process?[topology.WorldSize];
        long start = Stopwatch.GetTimestamp();
        string? failure = null;
        void KillAll(object? sender, EventArgs e) => Kill(processes);
        AppDomain.CurrentDomain.ProcessExit += KillAll;            // this process ending ends its ranks too
        try
        {
            void Start(int rank, string at)
            {
                var (n, local) = topology.Place(rank);
                string device = options.Simulation.CarveFrom ?? topology.Nodes[n].Devices[local].Device;
                var info = new ProcessStartInfo(launch.FileName)
                {
                    UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true, RedirectStandardInput = true,
                    StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8,
                    WorkingDirectory = launch.WorkingDirectory ?? Environment.CurrentDirectory,
                };
                foreach (string argument in launch.Arguments)
                {
                    info.ArgumentList.Add(argument);
                }

                if (launch.Environment is { } extra)
                {
                    foreach (var (name, value) in extra)
                    {
                        info.Environment[name] = value;
                    }
                }

                info.Environment["IDRAK_RANK"] = rank.ToString(CultureInfo.InvariantCulture);
                info.Environment["IDRAK_WORLD_SIZE"] = topology.WorldSize.ToString(CultureInfo.InvariantCulture);
                info.Environment["IDRAK_CLUSTER_TOPOLOGY"] = path;
                info.Environment["IDRAK_CLUSTER_NODE"] = topology.Nodes[n].Name;
                info.Environment["IDRAK_CLUSTER_DEVICE"] = device;
                info.Environment["IDRAK_COORDINATOR"] = at;
                info.Environment["IDRAK_CLUSTER_RUN"] = run;
                info.Environment["IDRAK_CLUSTER_PARENT"] = Environment.ProcessId.ToString(CultureInfo.InvariantCulture);
                info.Environment["IDRAK_CLUSTER_MODE"] = "localhost";
                if (launch.Control is { } control)
                {
                    info.Environment["IDRAK_CLUSTER_CONTROL"] = control.Path;
                }
                if (!simulate)
                {
                    info.Environment["IDRAK_CLUSTER_THROTTLE"] = "0";   // a real cluster's links are what they are
                }

                var process = new Process { StartInfo = info, EnableRaisingEvents = true };
                int keep = Math.Max(1, launch.KeepLines);
                process.OutputDataReceived += (_, e) =>
                {
                    if (e.Data is not { } text)
                    {
                        return;
                    }

                    Keep(results[rank].Output, text, keep);
                    lines?.Invoke(new LocalhostLine(rank, text, false));
                };
                process.ErrorDataReceived += (_, e) =>
                {
                    if (e.Data is { } text)
                    {
                        Keep(results[rank].Errors, text, keep);
                        lines?.Invoke(new LocalhostLine(rank, text, true));
                    }
                };
                process.Start();
                processes[rank] = process;
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
            }

            foreach (int rank in ranks)
            {
                Start(rank, coordinator);
            }

            failure = Watch(ranks, processes, results, start, launch.Limit, cancellationToken);
        }
        finally
        {
            AppDomain.CurrentDomain.ProcessExit -= KillAll;
            var ended = ranks.Where(r => processes[r] is { HasExited: false }).ToHashSet();   // ended by the launcher, not by themselves
            Kill(processes);
            foreach (int rank in ranks)
            {
                if (processes[rank] is { } process)
                {
                    process.WaitForExit(5000);
                    if (process.HasExited)
                    {
                        process.WaitForExit();                     // the last lines read
                        if (!ended.Contains(rank))
                        {
                            results[rank].Code ??= process.ExitCode;
                        }
                    }

                    process.Dispose();
                }
            }
        }

        var rows = ranks.Select(r =>
        {
            var (n, local) = topology.Place(r);
            var (code, output, errors) = results[r];
            lock (output)
            {
                lock (errors)
                {
                    return new LocalhostRankResult(r, options.Simulation.CarveFrom ?? topology.Nodes[n].Devices[local].Device, code, [.. output], [.. errors]);
                }
            }
        }).ToList();
        return new LocalhostRunResult(rows, Stopwatch.GetElapsedTime(start), failure);
    }

    // Waits for every rank; the first to fail (or the limit, or a cancellation) ends the others. Returns why it stopped early.
    private static string? Watch(int[] ranks, Process?[] processes, (int? Code, Queue<string> Output, Queue<string> Errors)[] results, long start, TimeSpan? limit,
        CancellationToken cancellationToken)
    {
        while (true)
        {
            bool running = false;
            foreach (int rank in ranks)
            {
                var process = processes[rank]!;
                if (!process.HasExited)
                {
                    running = true;
                    continue;
                }

                if (results[rank].Code is null)
                {
                    results[rank].Code = process.ExitCode;
                    if (process.ExitCode != 0)
                    {
                        return $"rank {rank} ended with exit code {process.ExitCode}; the other ranks were ended";
                    }
                }
            }

            if (!running)
            {
                return null;
            }

            if (cancellationToken.IsCancellationRequested)
            {
                return "the run was cancelled; every rank was ended";
            }

            if (limit is { } most && Stopwatch.GetElapsedTime(start) > most)
            {
                return $"the run passed its limit of {most.TotalSeconds:F0} s; every rank was ended";
            }

            cancellationToken.WaitHandle.WaitOne(50);
        }
    }

    // A loopback port no one listens on now, for a localhost run's coordinator.
    private static int FreePort()
    {
        using var probe = new System.Net.Sockets.Socket(System.Net.Sockets.AddressFamily.InterNetwork, System.Net.Sockets.SocketType.Stream, System.Net.Sockets.ProtocolType.Tcp);
        probe.Bind(new IPEndPoint(IPAddress.Loopback, 0));
        return ((IPEndPoint)probe.LocalEndPoint!).Port;
    }

    private static void Keep(Queue<string> lines, string text, int keep)
    {
        lock (lines)
        {
            lines.Enqueue(text);
            while (lines.Count > keep)
            {
                lines.Dequeue();
            }
        }
    }

    private static void Kill(Process?[] processes)
    {
        foreach (var process in processes)
        {
            try
            {
                if (process is not null && !process.HasExited)
                {
                    process.Kill(entireProcessTree: true);
                }
            }
            catch (Exception e) when (e is InvalidOperationException or System.ComponentModel.Win32Exception or NotSupportedException)
            {
                // ended meanwhile
            }
        }
    }

    /// <summary>
    /// Joins this process to the cluster it was started for (a rank of <see cref="Run"/> or of <c>idrak cluster start</c>, or
    /// started by hand with the same variables): reads the topology, finds its rank (<see cref="ClusterOptions.Rank"/>, else
    /// the only rank of this host's node), its device (<see cref="ClusterOptions.Device"/>, else its entry's, carved with
    /// the entry's memory as a cap when it declares one), connects to every other rank as <see cref="TransportPlan"/> joins
    /// each pair, paces the links to the topology's profiles in a localhost simulation with the throttle on, and makes the
    /// collectives. While the launcher's process (<see cref="ClusterOptions.Parent"/>) lives this one does; when it is gone,
    /// this process ends.
    /// </summary>
    /// <param name="options">The options, or null for the environment's.</param>
    /// <exception cref="InvalidOperationException">No topology, no rank, or a rank, world size or node that contradicts the file (the message names both).</exception>
    public static LocalhostRank Join(ClusterOptions? options = null)
    {
        options ??= ClusterOptions.FromEnvironment();
        string path = options.Topology ?? throw new InvalidOperationException("A rank of a cluster reads its topology file from IDRAK_CLUSTER_TOPOLOGY, which is not set (idrak cluster start sets it).");
        return Join(ClusterTopology.Load(path), options, path);
    }

    /// <summary>
    /// Joins this process to a cluster of <paramref name="topology"/> as one rank (<see cref="Join(ClusterOptions)"/> for a
    /// topology given as an object: a hand-started process with a bare coordinator and world size, plan 23 C8, every rank on
    /// a node of its own); <paramref name="source"/> names it in messages.
    /// </summary>
    /// <exception cref="InvalidOperationException">No rank, or a rank, world size or node that contradicts the topology (the message names both).</exception>
    public static LocalhostRank Join(ClusterTopology topology, ClusterOptions options, string? source = null)
    {
        ArgumentNullException.ThrowIfNull(topology);
        ArgumentNullException.ThrowIfNull(options);
        string path = source ?? "the topology";
        options = options with { Simulation = options.Simulation with { Mode = SimulationMode.Localhost } };   // one process per rank
        if (options.WorldSize is { } world && world != topology.WorldSize)
        {
            throw new InvalidOperationException($"IDRAK_WORLD_SIZE={world} contradicts the topology {path}, which has {topology.WorldSize} rank(s).");
        }

        int? named = options.Node is { } nodeName ? topology.NodeOf(nodeName) ?? throw new InvalidOperationException($"IDRAK_CLUSTER_NODE={nodeName} is not a node of {path}.") : null;
        int rank = options.Rank ?? OnlyRank(topology, named, path);
        if ((uint)rank >= (uint)topology.WorldSize)
        {
            throw new InvalidOperationException($"IDRAK_RANK={rank} is not a rank of {path} (0 to {topology.WorldSize - 1}).");
        }

        if (named is { } n0 && topology.Place(rank).Node != n0)
        {
            throw new InvalidOperationException($"IDRAK_RANK={rank} contradicts IDRAK_CLUSTER_NODE={options.Node}: rank {rank} is on node {topology.Nodes[topology.Place(rank).Node].Name}.");
        }

        if (options.Parent is { } parent)
        {
            WatchParent(parent, rank);
        }

        var (node, local) = topology.Place(rank);
        var spec = topology.Nodes[node].Devices[local];
        var physical = Device.Parse(options.Device ?? options.Simulation.CarveFrom ?? spec.Device);
        var device = spec.MemoryBytes is { } cap ? SimulatedDevices.Carve(physical, cap, rank) : physical;
        var context = new ClusterContext(topology, rank, options);
        var plan = TransportPlan.Make(topology, options);
        var transport = plan.Connect(context);
        try
        {
            if (options.Simulation.Throttle && OnThisMachine(options.Coordinator ?? topology.Coordinator))
            {
                transport = ThrottledTransport.Wrap(transport, context);
            }

            var collectives = Collectives.Get(options.CollectiveFor(topology)).Create(transport, device, context);
            try
            {
                // Every rank runs rank 0's measured kernel choices, taken before this rank's work makes one (ShareTuning).
                var shared = ChoiceSharing.Exchange(transport, device, options);
                return new LocalhostRank(new ClusterRank(rank, device, transport, collectives, context) { SharedChoices = shared }, plan);
            }
            catch
            {
                collectives.Dispose();
                throw;
            }
        }
        catch
        {
            transport.Dispose();
            throw;
        }
    }

    /// <summary>
    /// How to start <paramref name="assembly"/>'s program as a child process: its own launcher beside it when there is one,
    /// else the dotnet host running it (the host this process runs on when it is one), else this process's program (a
    /// single-file app).
    /// </summary>
    public static (string FileName, IReadOnlyList<string> Arguments) ProgramOf(Assembly assembly)
    {
        ArgumentNullException.ThrowIfNull(assembly);
        string location = Path.Combine(AppContext.BaseDirectory, assembly.GetName().Name + ".dll");   // beside this app (Location is empty in a single-file app)
        if (!File.Exists(location))
        {
            return (Environment.ProcessPath ?? throw new InvalidOperationException("This process's program is not known."), []);
        }

        string launcher = Path.ChangeExtension(location, OperatingSystem.IsWindows() ? ".exe" : null);
        if (File.Exists(launcher))
        {
            return (launcher, []);
        }

        string host = Environment.ProcessPath is { } current && string.Equals(Path.GetFileNameWithoutExtension(current), "dotnet", StringComparison.OrdinalIgnoreCase)
            ? current
            : "dotnet";
        return (host, [location]);
    }

    private static int OnlyRank(IClusterTopology topology, int? named, string path)
    {
        int node = named ?? topology.NodeOf(Environment.MachineName)
            ?? throw new InvalidOperationException($"This host ({Environment.MachineName}) is not a node of {path}: set IDRAK_CLUSTER_NODE, or IDRAK_RANK.");
        var ranks = topology.Ranks(node);
        return ranks.Count == 1
            ? ranks[0]
            : throw new InvalidOperationException($"Node {topology.Nodes[node].Name} holds {ranks.Count} ranks ({string.Join(", ", ranks)}): set IDRAK_RANK (idrak cluster start sets it for each).");
    }

    private static bool OnThisMachine(string? coordinator)
    {
        int colon = coordinator?.LastIndexOf(':') ?? -1;
        if (coordinator is null || colon <= 0)
        {
            return true;
        }

        string host = coordinator[..colon].Trim('[', ']');
        return string.Equals(host, "localhost", StringComparison.OrdinalIgnoreCase) || (IPAddress.TryParse(host, out var ip) && IPAddress.IsLoopback(ip));
    }

    private static void WatchParent(int parent, int rank)
    {
        var watch = new Thread(() =>
        {
            while (true)
            {
                Thread.Sleep(1000);
                bool alive;
                try
                {
                    using var process = Process.GetProcessById(parent);
                    alive = !process.HasExited;
                }
                catch (Exception e) when (e is ArgumentException or InvalidOperationException)
                {
                    alive = false;
                }

                if (!alive)
                {
                    Console.Error.WriteLine($"rank {rank}: the launcher (process {parent}) is gone; this rank ends");
                    Environment.Exit(3);
                }
            }
        })
        { IsBackground = true, Name = "cluster launcher watch" };
        watch.Start();
    }
}
