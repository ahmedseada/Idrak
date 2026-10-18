// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Cluster;

/// <summary>How a pipeline's stages are cut (<see cref="ClusterOptions.PipelineBalance"/>).</summary>
public enum PipelineBalance
{
    /// <summary>By the ranks' declared memory when the topology declares every rank's and they differ, else by bytes.</summary>
    Auto,

    /// <summary>Stages hold equal bytes of the model's tensors.</summary>
    Bytes,

    /// <summary>Stages hold bytes in proportion to their ranks' declared memory (the smallest of a stage's replicas).</summary>
    Memory,
}

/// <summary>One stage of a pipeline: the model's top-level modules [Start, End) and the bytes of the tensors they hold or read.</summary>
/// <param name="Stage">The stage, from 0 (the model's input) to the last (its output).</param>
/// <param name="Start">Its first module.</param>
/// <param name="End">One past its last module.</param>
/// <param name="Bytes">The bytes of the distinct tensors its modules hold or read (a tied head's table counted where it is read).</param>
public readonly record struct PipelineStageRange(int Stage, int Start, int End, long Bytes)
{
    /// <summary>The number of modules.</summary>
    public int Count => End - Start;

    /// <summary>Whether module <paramref name="module"/> is in this stage.</summary>
    public bool Contains(int module) => module >= Start && module < End;
}

/// <summary>A top-level module of a model as a pipeline plan counts it: its name and the tensors it holds or reads.</summary>
/// <param name="Name">The module's name, for messages.</param>
/// <param name="Tensors">
/// Indices into the plan's tensor table of every tensor the module holds or reads (its parameters and buffers, and a tied
/// layer's table it reads from another module): a tensor several modules read counts once in a stage that holds them all.
/// </param>
public sealed record PipelineModule(string Name, IReadOnlyList<int> Tensors);

/// <summary>
/// Which ranks are which stage (plan 23, phase C7): <see cref="Stages"/> stages of <see cref="Replicas"/> ranks each, stage
/// by stage in rank order (ranks 0 to R-1 stage 0, and so on), so a stage's replicas sit together (on one node where the
/// topology lists its ranks so) and reduce their gradients there, while only activations cross between stages. Replica r of
/// every stage is one pipeline; with one replica the pipeline is the whole group.
/// </summary>
/// <param name="WorldSize">The ranks.</param>
/// <param name="Stages">The stages; a divisor of <paramref name="WorldSize"/>.</param>
public readonly record struct PipelineLayout(int WorldSize, int Stages)
{
    /// <summary>The ranks of each stage: the pipelines running side by side as data-parallel replicas.</summary>
    public int Replicas => WorldSize / Stages;

    /// <summary>The stage rank <paramref name="rank"/> runs.</summary>
    public int StageOf(int rank) => rank / Replicas;

    /// <summary>The pipeline (replica) rank <paramref name="rank"/> belongs to.</summary>
    public int ReplicaOf(int rank) => rank % Replicas;

    /// <summary>The rank running stage <paramref name="stage"/> of replica <paramref name="replica"/>.</summary>
    public int RankOf(int stage, int replica) => (stage * Replicas) + replica;

    /// <summary>The ranks of stage <paramref name="stage"/>, in replica order.</summary>
    public int[] StageRanks(int stage)
    {
        var ranks = new int[Replicas];
        for (int r = 0; r < ranks.Length; r++)
        {
            ranks[r] = RankOf(stage, r);
        }

        return ranks;
    }

    /// <summary>The layout of <paramref name="worldSize"/> ranks in <see cref="ClusterOptions.Stages"/> stages (unset: one stage per rank).</summary>
    /// <exception cref="ArgumentException">The stages do not divide the ranks; the message names IDRAK_CLUSTER_STAGES and both numbers.</exception>
    public static PipelineLayout For(int worldSize, ClusterOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(worldSize);
        int stages = options.Stages ?? worldSize;
        if (stages < 1 || stages > worldSize || worldSize % stages != 0)
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture,
                $"A pipeline of {stages} stages over {worldSize} ranks: the stages (IDRAK_CLUSTER_STAGES) must divide the ranks, each stage then held by "
                + $"world size / stages ranks that reduce its gradients as data-parallel replicas (for {worldSize} ranks: {string.Join(", ", Enumerable.Range(1, worldSize).Where(s => worldSize % s == 0))})."));
        }

        return new PipelineLayout(worldSize, stages);
    }
}

/// <summary>
/// How a model is cut into pipeline stages (plan 23, phase C7): its top-level modules (embeddings first, the output head
/// last) in contiguous ranges, every module in exactly one stage, each stage at least one module. The cut balances the
/// bytes of the tensors each stage holds (a tied head's table counted in the stage that reads it), equal by default or in
/// proportion to each stage's declared memory, or follows cuts given (<see cref="ClusterOptions.PipelineCuts"/>). The same
/// sizes give the same plan on every rank.
/// </summary>
public sealed class PipelinePlan
{
    private PipelinePlan(IReadOnlyList<PipelineModule> modules, IReadOnlyList<long> tensorBytes, PipelineStageRange[] stages, string basis)
    {
        Modules = modules;
        TensorBytes = tensorBytes;
        Stages = stages;
        Basis = basis;
    }

    /// <summary>The model's top-level modules, in order.</summary>
    public IReadOnlyList<PipelineModule> Modules { get; }

    /// <summary>The bytes of each tensor the modules index.</summary>
    public IReadOnlyList<long> TensorBytes { get; }

    /// <summary>The stages, in order.</summary>
    public IReadOnlyList<PipelineStageRange> Stages { get; }

    /// <summary>What the cut balanced: "bytes", "memory", or "cuts" (given).</summary>
    public string Basis { get; }

    /// <summary>The bytes of every distinct tensor of the model (what one device holds).</summary>
    public long TotalBytes => Bytes(0, Modules.Count);

    /// <summary>The stage holding module <paramref name="module"/>.</summary>
    public int StageOf(int module)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(module);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(module, Modules.Count);
        foreach (var stage in Stages)
        {
            if (stage.Contains(module))
            {
                return stage.Stage;
            }
        }

        throw new InvalidOperationException("unreachable: every module is in a stage");
    }

    /// <summary>
    /// The plan of <paramref name="stages"/> stages whose largest load is the least: each stage's bytes over its share of
    /// <paramref name="capacities"/> (equal shares when null), every stage at least one module. Ties go to the earliest cut,
    /// so every rank finds the same plan.
    /// </summary>
    /// <exception cref="ArgumentException">More stages than modules, or capacities not one per stage above 0.</exception>
    public static PipelinePlan Balance(IReadOnlyList<PipelineModule> modules, IReadOnlyList<long> tensorBytes, int stages, IReadOnlyList<long>? capacities = null)
    {
        Validate(modules, tensorBytes);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(stages);
        int n = modules.Count;
        if (stages > n)
        {
            throw new ArgumentException($"A pipeline of {stages} stages over a model of {n} top-level modules ({string.Join(", ", modules.Select(m => m.Name))}): each stage needs at least one; lower IDRAK_CLUSTER_STAGES.");
        }

        if (capacities is not null && (capacities.Count != stages || capacities.Any(c => c <= 0)))
        {
            throw new ArgumentException($"A memory cut needs one capacity above 0 per stage ({stages}); got [{string.Join(", ", capacities)}].", nameof(capacities));
        }

        double total = capacities?.Sum(c => (double)c) ?? stages;
        double Share(int stage) => (capacities is null ? 1.0 : capacities[stage]) / total;

        // cost[j, i]: the distinct bytes of modules [j, i), made once (a model has tens to hundreds of top-level modules).
        var cost = new long[n + 1, n + 1];
        for (int j = 0; j < n; j++)
        {
            var seen = new HashSet<int>();
            long bytes = 0;
            for (int i = j; i < n; i++)
            {
                foreach (int t in modules[i].Tensors)
                {
                    if (seen.Add(t))
                    {
                        bytes += tensorBytes[t];
                    }
                }

                cost[j, i + 1] = bytes;
            }
        }

        // best[s, i]: the least largest load of stages 0..s over modules [0, i); from[s, i]: where stage s starts.
        var best = new double[stages, n + 1];
        var from = new int[stages, n + 1];
        for (int s = 0; s < stages; s++)
        {
            for (int i = 0; i <= n; i++)
            {
                best[s, i] = double.PositiveInfinity;
            }
        }

        for (int i = 1; i <= n - (stages - 1); i++)
        {
            best[0, i] = cost[0, i] / Share(0);
        }

        for (int s = 1; s < stages; s++)
        {
            for (int i = s + 1; i <= n - (stages - 1 - s); i++)
            {
                for (int j = s; j < i; j++)
                {
                    double load = Math.Max(best[s - 1, j], cost[j, i] / Share(s));
                    if (load < best[s, i])
                    {
                        (best[s, i], from[s, i]) = (load, j);
                    }
                }
            }
        }

        var starts = new int[stages];
        for (int s = stages - 1, end = n; s > 0; s--)
        {
            starts[s] = from[s, end];
            end = starts[s];
        }

        return Make(modules, tensorBytes, starts, capacities is null ? "bytes" : "memory");
    }

    /// <summary>The plan whose stages start at <paramref name="starts"/> (the first 0, rising, each below the module count).</summary>
    /// <exception cref="ArgumentException">The starts do not cut the modules into stages of at least one; the message names IDRAK_CLUSTER_PIPELINE_CUTS.</exception>
    public static PipelinePlan FromStarts(IReadOnlyList<PipelineModule> modules, IReadOnlyList<long> tensorBytes, IReadOnlyList<int> starts)
    {
        Validate(modules, tensorBytes);
        ArgumentNullException.ThrowIfNull(starts);
        bool rising = starts.Count > 0 && starts[0] == 0 && starts.Zip(starts.Skip(1)).All(p => p.First < p.Second) && starts[^1] < modules.Count;
        if (!rising)
        {
            throw new ArgumentException($"The pipeline cuts [{string.Join(", ", starts)}] (IDRAK_CLUSTER_PIPELINE_CUTS: the first module of each stage after the first) "
                + $"must rise from 0 within the model's {modules.Count} top-level modules ({string.Join(", ", modules.Select((m, i) => $"{i} {m.Name}"))}).");
        }

        return Make(modules, tensorBytes, [.. starts], "cuts");
    }

    /// <summary>
    /// The plan <paramref name="options"/> ask for over <paramref name="layout"/>'s stages: <see cref="ClusterOptions.PipelineCuts"/>
    /// when given; else balanced by the ranks' declared memory (<see cref="ClusterOptions.PipelineBalance"/>: memory, or
    /// auto when <paramref name="topology"/> declares every rank's memory and they differ), each stage's the least of its
    /// replicas'; else by bytes.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// The cuts do not match the stages, or a memory cut is asked of a topology that does not declare a rank's memory (named).
    /// </exception>
    public static PipelinePlan For(IReadOnlyList<PipelineModule> modules, IReadOnlyList<long> tensorBytes, PipelineLayout layout, ClusterOptions options,
        IClusterTopology? topology)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (options.PipelineCuts is { } cuts)
        {
            int[] starts = [0, .. cuts];
            if (starts.Length != layout.Stages)
            {
                throw new ArgumentException($"IDRAK_CLUSTER_PIPELINE_CUTS gives {cuts.Count} cut(s), so {starts.Length} stages; the layout has {layout.Stages} (IDRAK_CLUSTER_STAGES).");
            }

            return FromStarts(modules, tensorBytes, starts);
        }

        long?[] Declared() => topology is null ? [.. Enumerable.Repeat((long?)null, layout.WorldSize)]
            : [.. Enumerable.Range(0, layout.WorldSize).Select(r => topology.Nodes[topology.Place(r).Node].Devices[topology.Place(r).Local].MemoryBytes)];
        var declared = Declared();
        bool memory = options.PipelineBalance switch
        {
            PipelineBalance.Memory => true,
            PipelineBalance.Bytes => false,
            _ => declared.All(d => d is not null) && declared.Distinct().Count() > 1,
        };
        if (!memory)
        {
            return Balance(modules, tensorBytes, layout.Stages);
        }

        int missing = Array.FindIndex(declared, d => d is null);
        if (missing >= 0)
        {
            throw new ArgumentException($"IDRAK_CLUSTER_PIPELINE_BALANCE=memory cuts by the ranks' declared memory, and the topology declares none for rank {missing}; "
                + "give every device a \"memory\" in the topology file, or cut by bytes.");
        }

        long[] capacities = [.. Enumerable.Range(0, layout.Stages).Select(s => layout.StageRanks(s).Min(r => declared[r]!.Value))];
        return Balance(modules, tensorBytes, layout.Stages, capacities);
    }

    /// <summary>"stage 0: modules 0-2 (embed … layers.0), 101 KiB; stage 1: …" (invariant culture).</summary>
    public override string ToString() => string.Join("; ", Stages.Select(s => string.Create(CultureInfo.InvariantCulture,
        $"stage {s.Stage}: modules {s.Start}-{s.End - 1} ({Modules[s.Start].Name}{(s.Count > 1 ? $" … {Modules[s.End - 1].Name}" : "")}), {s.Bytes / 1024.0:F1} KiB")));

    private long Bytes(int start, int end)
    {
        var seen = new HashSet<int>();
        long bytes = 0;
        for (int i = start; i < end; i++)
        {
            foreach (int t in Modules[i].Tensors)
            {
                if (seen.Add(t))
                {
                    bytes += TensorBytes[t];
                }
            }
        }

        return bytes;
    }

    private static PipelinePlan Make(IReadOnlyList<PipelineModule> modules, IReadOnlyList<long> tensorBytes, int[] starts, string basis)
    {
        var plan = new PipelinePlan(modules, tensorBytes, [], basis);
        var stages = new PipelineStageRange[starts.Length];
        for (int s = 0; s < starts.Length; s++)
        {
            int end = s + 1 < starts.Length ? starts[s + 1] : modules.Count;
            stages[s] = new PipelineStageRange(s, starts[s], end, plan.Bytes(starts[s], end));
        }

        return new PipelinePlan(modules, tensorBytes, stages, basis);
    }

    private static void Validate(IReadOnlyList<PipelineModule> modules, IReadOnlyList<long> tensorBytes)
    {
        ArgumentNullException.ThrowIfNull(modules);
        ArgumentNullException.ThrowIfNull(tensorBytes);
        if (modules.Count == 0)
        {
            throw new ArgumentException("A pipeline cuts a model of at least one top-level module.", nameof(modules));
        }

        if (modules.SelectMany(m => m.Tensors).Any(t => (uint)t >= (uint)tensorBytes.Count) || tensorBytes.Any(b => b < 0))
        {
            throw new ArgumentException("Every module's tensors index the tensor table, whose bytes are 0 or more.", nameof(tensorBytes));
        }
    }
}

/// <summary>A pipeline stage's work on one micro-batch: its forward pass or its backward pass.</summary>
public enum PipelineWorkKind
{
    /// <summary>The micro-batch's forward pass (receive the activation, run the stage, send it on).</summary>
    Forward,

    /// <summary>Its backward pass (receive the gradient, run the stage's backward, send it back).</summary>
    Backward,
}

/// <summary>One item of a stage's work in a step.</summary>
/// <param name="Kind">Forward or backward.</param>
/// <param name="MicroBatch">The micro-batch, from 0.</param>
public readonly record struct PipelineWork(PipelineWorkKind Kind, int MicroBatch);

/// <summary>
/// The order in which a pipeline stage runs its micro-batches' forward and backward passes in a training step (plan 23,
/// phase C7): every stage of a pipeline runs the same schedule, so the order is fixed by the stage, the stage count and the
/// micro-batches alone, and the messages between stages always meet. Registered by name in <see cref="PipelineSchedules"/>.
/// </summary>
public interface IPipelineSchedule
{
    /// <summary>The schedule's name ("gpipe").</summary>
    string Name { get; }

    /// <summary>One line for listings.</summary>
    string Summary { get; }

    /// <summary>
    /// Stage <paramref name="stage"/>'s work in a step of <paramref name="microBatches"/> micro-batches over
    /// <paramref name="stages"/> stages: every micro-batch's forward once and its backward once, the backward after the forward.
    /// </summary>
    IReadOnlyList<PipelineWork> Order(int stage, int stages, int microBatches);

    /// <summary>The share of a step a stage waits idle on stages of equal cost, by the schedule's arithmetic (the bubble the telemetry's measured one is set against).</summary>
    double IdealBubble(int stages, int microBatches);

    /// <summary>
    /// The most micro-batches whose activations a stage holds at once (memory): for <paramref name="stage"/> of
    /// <paramref name="stages"/> with <paramref name="microBatches"/>.
    /// </summary>
    int InFlight(int stage, int stages, int microBatches);
}

/// <summary>
/// The pipeline schedules, by name (plan 23, phase C7). Idrak.Network registers "gpipe" (every micro-batch's forward, then
/// every backward in reverse); none is built into Idrak.Abstraction. <see cref="ClusterOptions.Schedule"/> names the one a run uses.
/// </summary>
public static class PipelineSchedules
{
    private static readonly SlotTable<string, IPipelineSchedule> Table = new(nameof(PipelineSchedules), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "every stage of a pipeline must run the same schedule, or their messages never meet; another cannot take over a step");

    static PipelineSchedules() => LibraryDefaults.Ensure(typeof(PipelineSchedules));

    /// <summary>The registered schedules, in registration order.</summary>
    public static IReadOnlyList<IPipelineSchedule> All => Table.Values;

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>Registers <paramref name="schedule"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IPipelineSchedule schedule)
    {
        ArgumentNullException.ThrowIfNull(schedule);
        ArgumentException.ThrowIfNullOrWhiteSpace(schedule.Name);
        Table.Register(schedule.Name, schedule, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's schedule of that name (the library's, if any, answers again).</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The schedule of that name, or null.</summary>
    public static IPipelineSchedule? Find(string name) => Table.Find(name);

    /// <summary>The schedule of that name.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name (Idrak.Network registers "gpipe": ship it with the app).</exception>
    public static IPipelineSchedule Get(string name) => Table.Find(name)
        ?? throw new NotSupportedException($"Pipeline schedule '{name}' is not registered (registered: {ClusterNames.List(Table.Keys)}); add one with PipelineSchedules.Register, or reference Idrak.Network (IDRAK_CLUSTER_SCHEDULE).");

    /// <summary>The library's schedule of that name, or null.</summary>
    public static IPipelineSchedule? Default(string name) => Table.Default(name);

    /// <summary>Who registered the schedule of that name.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each schedule as "name: summary", one a line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(s => $"{s.Name}: {s.Summary}"));
}

/// <summary>
/// Groups of some of a cluster's ranks (plan 23, phase C7): a pipeline stage's data-parallel replicas reduce their
/// gradients over a group of their own, through the transport the whole cluster connected.
/// </summary>
public static class ClusterGroups
{
    /// <summary>
    /// The ranks <paramref name="members"/> of <paramref name="whole"/> (rising, this rank among them) as a transport of
    /// their own, numbered from 0 in that order: every message goes over <paramref name="whole"/> to the member's rank. It
    /// owns nothing: disposing it leaves <paramref name="whole"/> connected.
    /// </summary>
    /// <exception cref="ArgumentException">The members do not rise within the group, or leave out this rank.</exception>
    public static ITransport Transport(ITransport whole, IReadOnlyList<int> members)
    {
        ArgumentNullException.ThrowIfNull(whole);
        int[] ranks = Members(members, whole.WorldSize);
        int me = Array.IndexOf(ranks, whole.Rank);
        if (me < 0)
        {
            throw new ArgumentException($"Rank {whole.Rank} is not among the group's members [{string.Join(", ", ranks)}].", nameof(members));
        }

        return new SubsetTransport(whole, ranks, me);
    }

    /// <summary>The ranks <paramref name="members"/> of <paramref name="whole"/> as a topology of their own, numbered from 0 in that order.</summary>
    public static IClusterTopology Topology(IClusterTopology whole, IReadOnlyList<int> members)
    {
        ArgumentNullException.ThrowIfNull(whole);
        return new SubsetTopology(whole, Members(members, whole.WorldSize));
    }

    private static int[] Members(IReadOnlyList<int> members, int world)
    {
        ArgumentNullException.ThrowIfNull(members);
        int[] ranks = [.. members];
        if (ranks.Length == 0 || ranks.Any(r => (uint)r >= (uint)world) || ranks.Zip(ranks.Skip(1)).Any(p => p.First >= p.Second))
        {
            throw new ArgumentException($"A group's members [{string.Join(", ", ranks)}] rise within ranks 0 to {world - 1}.", nameof(members));
        }

        return ranks;
    }
}

// Some ranks of a transport's group as a group of their own: sends and receives go to the members' ranks; a barrier is
// messages to the first member and back, on a tag below every collective's.
internal sealed class SubsetTransport(ITransport whole, int[] members, int rank) : ITransport
{
    private const int BarrierTag = int.MinValue + 13;

    public string Name => whole.Name;

    public int Rank { get; } = rank;

    public int WorldSize => members.Length;

    public TransportCapabilities Capabilities => whole.Capabilities;

    public TimeSpan SimulatedTime => whole.SimulatedTime;

    public bool Simulated => whole.Simulated;

    public int? SimulatedCriticalRank => whole.SimulatedCriticalRank is { } critical && Array.IndexOf(members, critical) is >= 0 and var at ? at : null;

    public void Send(int to, ReadOnlySpan<byte> bytes, int tag) => whole.Send(Member(to, nameof(to)), bytes, tag);

    public int Receive(int from, Span<byte> bytes, int tag) => whole.Receive(Member(from, nameof(from)), bytes, tag);

    public ValueTask SendAsync(int to, ReadOnlyMemory<byte> bytes, int tag, CancellationToken cancellationToken = default) =>
        whole.SendAsync(Member(to, nameof(to)), bytes, tag, cancellationToken);

    public ValueTask<int> ReceiveAsync(int from, Memory<byte> bytes, int tag, CancellationToken cancellationToken = default) =>
        whole.ReceiveAsync(Member(from, nameof(from)), bytes, tag, cancellationToken);

    public void Barrier()
    {
        Span<byte> none = [];
        if (Rank != 0)
        {
            Send(0, none, BarrierTag);
            Receive(0, none, BarrierTag);
            return;
        }

        for (int r = 1; r < WorldSize; r++)
        {
            Receive(r, none, BarrierTag);
        }

        for (int r = 1; r < WorldSize; r++)
        {
            Send(r, none, BarrierTag);
        }
    }

    public void Dispose()
    {
        // The whole group's transport stays connected: its owner disposes it.
    }

    private int Member(int peer, string name) => peer != Rank && (uint)peer < (uint)members.Length
        ? members[peer]
        : throw new ArgumentOutOfRangeException(name, peer, $"rank {Rank} of the group reaches the other ranks of 0 to {members.Length - 1}");
}
