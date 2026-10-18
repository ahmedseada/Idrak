// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Abstraction.Cluster;

/// <summary>Which state a sharded run splits across the ranks (ZeRO-style, plan 23 phase C6).</summary>
public enum ShardMode
{
    /// <summary>Every rank holds everything.</summary>
    None,

    /// <summary>Each rank holds its share of the optimizer's state.</summary>
    Optimizer,

    /// <summary>Its share of the optimizer's state and of the gradients.</summary>
    Gradients,

    /// <summary>Its share of the optimizer's state, the gradients and the weights.</summary>
    Parameters,
}

/// <summary>Where a simulated cluster's ranks run.</summary>
public enum SimulationMode
{
    /// <summary>Every rank a thread of this process, joined by the "sim" transport: fast and deterministic.</summary>
    InProcess,

    /// <summary>One process per rank over localhost, through the real transports, optionally throttled to the links.</summary>
    Localhost,
}

/// <summary>Which clock a run's throughput figures use.</summary>
public enum ThroughputClock
{
    /// <summary>The time the run took here.</summary>
    Wall,

    /// <summary>The virtual clock's time: the cluster the topology describes.</summary>
    Simulated,
}

/// <summary>
/// The shared secret ranks prove to each other in their first message (an HMAC, <see cref="WireFormat.SignHello"/>). It is
/// read from <c>IDRAK_CLUSTER_KEY</c>, never from a command line, and never written: <see cref="ToString"/> hides it.
/// </summary>
public sealed class ClusterKey
{
    /// <summary>The fewest characters a key may have.</summary>
    public const int MinimumLength = 16;

    private readonly byte[] _bytes;

    private ClusterKey(byte[] bytes) => _bytes = bytes;

    internal ReadOnlySpan<byte> Bytes => _bytes;

    /// <summary>A key from its text (UTF-8).</summary>
    /// <exception cref="ArgumentException">Shorter than <see cref="MinimumLength"/> characters.</exception>
    public static ClusterKey FromText(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (text.Length < MinimumLength)
        {
            throw new ArgumentException($"A cluster key needs at least {MinimumLength} characters.", nameof(text));
        }

        return new ClusterKey(Encoding.UTF8.GetBytes(text));
    }

    /// <summary>"(secret)": the key is never written.</summary>
    public override string ToString() => "(secret)";
}

/// <summary>
/// The settings of a simulated run (plan 23). The seed and the failure model come from the topology file's "simulation"
/// block unless set here.
/// </summary>
public sealed record SimulationOptions
{
    /// <summary>The seed of jitter and faults, or null for the topology's (<c>IDRAK_CLUSTER_SEED</c>).</summary>
    public int? Seed { get; init; }

    /// <summary>Where the ranks run (<c>IDRAK_CLUSTER_MODE</c>: in-process or localhost).</summary>
    public SimulationMode Mode { get; init; } = SimulationMode.InProcess;

    /// <summary>Whether localhost transfers are paced to the topology's links (<c>IDRAK_CLUSTER_THROTTLE</c>: 1 or 0).</summary>
    public bool Throttle { get; init; } = true;

    /// <summary>The failure model, or null for the topology's (<c>IDRAK_CLUSTER_FAILURES</c>).</summary>
    public string? Failures { get; init; }

    /// <summary>The step a replay reruns, or null for a whole run (<c>IDRAK_CLUSTER_REPLAY_STEP</c>).</summary>
    public long? ReplayStep { get; init; }

    /// <summary>The clock throughput figures use (<c>IDRAK_CLUSTER_CLOCK</c>: wall or simulated).</summary>
    public ThroughputClock Clock { get; init; } = ThroughputClock.Wall;
}

/// <summary>
/// Everything a cluster run can be told (plan 23): where the topology is and which rank this process is, how ranks reach
/// each other, and the policies of the collectives and of training. Each field has an environment variable
/// (<see cref="FromEnvironment"/>); defaults are measured at run time where a size matters, and nothing names a card, a
/// link speed or a memory size. With a topology file the rank, world size and coordinator come from the file, and the
/// variables only check it.
/// </summary>
public sealed record ClusterOptions
{
    /// <summary>The topology file (<c>IDRAK_CLUSTER_TOPOLOGY</c>), or null for one node with the devices given.</summary>
    public string? Topology { get; init; }

    /// <summary>Which node of the file this machine is (<c>IDRAK_CLUSTER_NODE</c>), or null for the node named after this host.</summary>
    public string? Node { get; init; }

    /// <summary>This process's rank (<c>IDRAK_RANK</c>), or null: from the file when this node holds one rank.</summary>
    public int? Rank { get; init; }

    /// <summary>The number of ranks (<c>IDRAK_WORLD_SIZE</c>), or null: the file's device count.</summary>
    public int? WorldSize { get; init; }

    /// <summary>Where rank 0 listens (<c>IDRAK_COORDINATOR</c>, "host:port"), or null: the file's first node's address.</summary>
    public string? Coordinator { get; init; }

    /// <summary>The transport, a <see cref="Transports"/> name, or "auto" for the selector's choice per pair (<c>IDRAK_CLUSTER_TRANSPORT</c>).</summary>
    public string Transport { get; init; } = "auto";

    /// <summary>The <see cref="TransportSelectors"/> name (<c>IDRAK_CLUSTER_SELECTOR</c>).</summary>
    public string Selector { get; init; } = "fastest";

    /// <summary>The <see cref="Collectives"/> name, or null: "hierarchical" when the topology has more than one node, else "ring" (<c>IDRAK_CLUSTER_COLLECTIVE</c>).</summary>
    public string? Collective { get; init; }

    /// <summary>The shared secret of the handshake (<c>IDRAK_CLUSTER_KEY</c>), or null: nothing listens beyond this machine without one.</summary>
    public ClusterKey? Key { get; init; }

    /// <summary>The port this rank listens on for its peers; 0 lets the system pick and tells the coordinator (<c>IDRAK_CLUSTER_PORT</c>).</summary>
    public int ListenPort { get; init; }

    /// <summary>How long any wait for a peer lasts (<c>IDRAK_CLUSTER_TIMEOUT</c>, seconds).</summary>
    public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(120);

    /// <summary>Connection attempts at the rendezvous (<c>IDRAK_CLUSTER_RETRIES</c>).</summary>
    public int Retries { get; init; } = 3;

    /// <summary>The bytes of a chunk in flight, or null: measured at first use and kept in the tuning cache (<c>IDRAK_CLUSTER_CHUNK_BYTES</c>).</summary>
    public long? ChunkBytes { get; init; }

    /// <summary>Chunks in flight each way (<c>IDRAK_CLUSTER_IN_FLIGHT</c>).</summary>
    public int InFlight { get; init; } = 2;

    /// <summary>The socket buffers' bytes, or null for the system's (<c>IDRAK_CLUSTER_SOCKET_BUFFER</c>).</summary>
    public long? SocketBufferBytes { get; init; }

    /// <summary>The bytes of a gradient bucket reduced during the backward pass, or null: measured against the step (<c>IDRAK_CLUSTER_BUCKET_BYTES</c>).</summary>
    public long? BucketBytes { get; init; }

    /// <summary>How gradients travel (<c>IDRAK_CLUSTER_WIRE</c>: f32 or bf16).</summary>
    public WireType Wire { get; init; } = WireType.Float32;

    /// <summary>The share of gradient values sent (largest first), from 0 to 1; 0 sends them all (<c>IDRAK_CLUSTER_TOPK</c>).</summary>
    public double TopK { get; init; }

    /// <summary>Whether values top-k leaves out are added to the next step's (<c>IDRAK_CLUSTER_ERROR_FEEDBACK</c>: 1 or 0).</summary>
    public bool ErrorFeedback { get; init; } = true;

    /// <summary>Synchronize the weights every this many steps instead of the gradients every step; 0 for every step (<c>IDRAK_CLUSTER_LOCAL_SGD</c>).</summary>
    public int LocalSgdEvery { get; init; }

    /// <summary>Rank 0 sends its weights every this many steps (ranks of different device kinds round apart); 0 never (<c>IDRAK_CLUSTER_SYNC_WEIGHTS</c>).</summary>
    public int SyncWeightsEvery { get; init; }

    /// <summary>Re-measure each rank's speed and re-split the batch every this many steps; 0 measures the first warm steps only (<c>IDRAK_CLUSTER_REBALANCE_EVERY</c>).</summary>
    public int RebalanceEvery { get; init; }

    /// <summary>The state split across ranks (<c>IDRAK_CLUSTER_SHARD</c>: none, optimizer, gradients or parameters).</summary>
    public ShardMode Shard { get; init; } = ShardMode.None;

    /// <summary>How the model is placed: "single" (all of it on every rank) or "pipeline" (<c>IDRAK_CLUSTER_PLACEMENT</c>).</summary>
    public string Placement { get; init; } = "single";

    /// <summary>A pipeline's micro-batches in flight, or null: the gradient accumulation (<c>IDRAK_CLUSTER_MICROBATCHES</c>).</summary>
    public int? MicroBatches { get; init; }

    /// <summary>A pipeline's schedule (<c>IDRAK_CLUSTER_SCHEDULE</c>: gpipe).</summary>
    public string Schedule { get; init; } = "gpipe";

    /// <summary>The share of a step spent on the network above which the advisor suggests sending less (<c>IDRAK_CLUSTER_SUGGEST_SHARE</c>).</summary>
    public double SuggestShare { get; init; } = 0.2;

    /// <summary>The largest relative error of bfloat16 gradients the advisor accepts, or null for the reference's measured one (<c>IDRAK_CLUSTER_SUGGEST_BF16_TOLERANCE</c>).</summary>
    public double? SuggestBFloat16Tolerance { get; init; }

    /// <summary>The calibrated link profiles' file, or null for the tuning cache's (<c>IDRAK_CLUSTER_PROFILES_FILE</c>).</summary>
    public string? ProfilesFile { get; init; }

    /// <summary>The simulation's settings.</summary>
    public SimulationOptions Simulation { get; init; } = new();

    /// <summary>The environment variables these options read, in order.</summary>
    public static IReadOnlyList<string> Variables { get; } =
    [
        "IDRAK_CLUSTER_TOPOLOGY", "IDRAK_CLUSTER_NODE", "IDRAK_RANK", "IDRAK_WORLD_SIZE", "IDRAK_COORDINATOR", "IDRAK_CLUSTER_TRANSPORT",
        "IDRAK_CLUSTER_SELECTOR", "IDRAK_CLUSTER_COLLECTIVE", "IDRAK_CLUSTER_KEY", "IDRAK_CLUSTER_PORT", "IDRAK_CLUSTER_TIMEOUT",
        "IDRAK_CLUSTER_RETRIES", "IDRAK_CLUSTER_CHUNK_BYTES", "IDRAK_CLUSTER_IN_FLIGHT", "IDRAK_CLUSTER_SOCKET_BUFFER",
        "IDRAK_CLUSTER_BUCKET_BYTES", "IDRAK_CLUSTER_WIRE", "IDRAK_CLUSTER_TOPK", "IDRAK_CLUSTER_ERROR_FEEDBACK", "IDRAK_CLUSTER_LOCAL_SGD",
        "IDRAK_CLUSTER_SYNC_WEIGHTS", "IDRAK_CLUSTER_REBALANCE_EVERY", "IDRAK_CLUSTER_SHARD", "IDRAK_CLUSTER_PLACEMENT",
        "IDRAK_CLUSTER_MICROBATCHES", "IDRAK_CLUSTER_SCHEDULE", "IDRAK_CLUSTER_SUGGEST_SHARE", "IDRAK_CLUSTER_SUGGEST_BF16_TOLERANCE",
        "IDRAK_CLUSTER_PROFILES_FILE", "IDRAK_CLUSTER_SEED", "IDRAK_CLUSTER_MODE", "IDRAK_CLUSTER_THROTTLE", "IDRAK_CLUSTER_FAILURES",
        "IDRAK_CLUSTER_REPLAY_STEP", "IDRAK_CLUSTER_CLOCK",
    ];

    /// <summary>
    /// <paramref name="defaults"/> (or the defaults) with what the environment sets (<see cref="Variables"/>). Sizes take a
    /// unit (B, KB, MB, GB, powers of 1,024).
    /// </summary>
    /// <exception cref="ArgumentException">A value that does not parse; the message names the variable and what it takes.</exception>
    public static ClusterOptions FromEnvironment(ClusterOptions? defaults = null) => From(Environment.GetEnvironmentVariable, defaults);

    internal static ClusterOptions From(Func<string, string?> variable, ClusterOptions? defaults)
    {
        var o = defaults ?? new ClusterOptions();
        var s = o.Simulation;
        if (Read(variable, "IDRAK_CLUSTER_TOPOLOGY") is { } topology) o = o with { Topology = topology };
        if (Read(variable, "IDRAK_CLUSTER_NODE") is { } node) o = o with { Node = node };
        if (Read(variable, "IDRAK_RANK") is { } rank) o = o with { Rank = Whole("IDRAK_RANK", rank, 0) };
        if (Read(variable, "IDRAK_WORLD_SIZE") is { } world) o = o with { WorldSize = Whole("IDRAK_WORLD_SIZE", world, 1) };
        if (Read(variable, "IDRAK_COORDINATOR") is { } coordinator) o = o with { Coordinator = coordinator };
        if (Read(variable, "IDRAK_CLUSTER_TRANSPORT") is { } transport) o = o with { Transport = transport };
        if (Read(variable, "IDRAK_CLUSTER_SELECTOR") is { } selector) o = o with { Selector = selector };
        if (Read(variable, "IDRAK_CLUSTER_COLLECTIVE") is { } collective) o = o with { Collective = collective };
        if (variable("IDRAK_CLUSTER_KEY") is { Length: > 0 } key)
        {
            o = o with { Key = key.Length >= ClusterKey.MinimumLength ? ClusterKey.FromText(key) : throw new ArgumentException($"IDRAK_CLUSTER_KEY needs at least {ClusterKey.MinimumLength} characters.") };
        }

        if (Read(variable, "IDRAK_CLUSTER_PORT") is { } port) o = o with { ListenPort = Whole("IDRAK_CLUSTER_PORT", port, 0, 65535) };
        if (Read(variable, "IDRAK_CLUSTER_TIMEOUT") is { } timeout) o = o with { Timeout = TimeSpan.FromSeconds(Number("IDRAK_CLUSTER_TIMEOUT", timeout, "a number of seconds above 0", v => v > 0)) };
        if (Read(variable, "IDRAK_CLUSTER_RETRIES") is { } retries) o = o with { Retries = Whole("IDRAK_CLUSTER_RETRIES", retries, 1) };
        if (Read(variable, "IDRAK_CLUSTER_CHUNK_BYTES") is { } chunk) o = o with { ChunkBytes = Size("IDRAK_CLUSTER_CHUNK_BYTES", chunk) };
        if (Read(variable, "IDRAK_CLUSTER_IN_FLIGHT") is { } inFlight) o = o with { InFlight = Whole("IDRAK_CLUSTER_IN_FLIGHT", inFlight, 1) };
        if (Read(variable, "IDRAK_CLUSTER_SOCKET_BUFFER") is { } socket) o = o with { SocketBufferBytes = Size("IDRAK_CLUSTER_SOCKET_BUFFER", socket) };
        if (Read(variable, "IDRAK_CLUSTER_BUCKET_BYTES") is { } bucket) o = o with { BucketBytes = Size("IDRAK_CLUSTER_BUCKET_BYTES", bucket) };
        if (Read(variable, "IDRAK_CLUSTER_WIRE") is { } wire)
        {
            o = o with
            {
                Wire = wire.ToLowerInvariant() switch
                {
                    "f32" or "float32" => WireType.Float32,
                    "bf16" or "bfloat16" => WireType.BFloat16,
                    _ => throw Bad("IDRAK_CLUSTER_WIRE", wire, "f32 or bf16"),
                },
            };
        }

        if (Read(variable, "IDRAK_CLUSTER_TOPK") is { } topK) o = o with { TopK = Number("IDRAK_CLUSTER_TOPK", topK, "a share from 0 to 1", v => v is >= 0 and <= 1) };
        if (Read(variable, "IDRAK_CLUSTER_ERROR_FEEDBACK") is { } feedback) o = o with { ErrorFeedback = Flag("IDRAK_CLUSTER_ERROR_FEEDBACK", feedback) };
        if (Read(variable, "IDRAK_CLUSTER_LOCAL_SGD") is { } localSgd) o = o with { LocalSgdEvery = Whole("IDRAK_CLUSTER_LOCAL_SGD", localSgd, 0) };
        if (Read(variable, "IDRAK_CLUSTER_SYNC_WEIGHTS") is { } sync) o = o with { SyncWeightsEvery = Whole("IDRAK_CLUSTER_SYNC_WEIGHTS", sync, 0) };
        if (Read(variable, "IDRAK_CLUSTER_REBALANCE_EVERY") is { } rebalance) o = o with { RebalanceEvery = Whole("IDRAK_CLUSTER_REBALANCE_EVERY", rebalance, 0) };
        if (Read(variable, "IDRAK_CLUSTER_SHARD") is { } shard) o = o with { Shard = Choice<ShardMode>("IDRAK_CLUSTER_SHARD", shard, "none, optimizer, gradients or parameters") };
        if (Read(variable, "IDRAK_CLUSTER_PLACEMENT") is { } placement) o = o with { Placement = placement };
        if (Read(variable, "IDRAK_CLUSTER_MICROBATCHES") is { } micro) o = o with { MicroBatches = Whole("IDRAK_CLUSTER_MICROBATCHES", micro, 1) };
        if (Read(variable, "IDRAK_CLUSTER_SCHEDULE") is { } schedule) o = o with { Schedule = schedule };
        if (Read(variable, "IDRAK_CLUSTER_SUGGEST_SHARE") is { } share) o = o with { SuggestShare = Number("IDRAK_CLUSTER_SUGGEST_SHARE", share, "a share from 0 to 1", v => v is >= 0 and <= 1) };
        if (Read(variable, "IDRAK_CLUSTER_SUGGEST_BF16_TOLERANCE") is { } tolerance) o = o with { SuggestBFloat16Tolerance = Number("IDRAK_CLUSTER_SUGGEST_BF16_TOLERANCE", tolerance, "a number above 0", v => v > 0) };
        if (Read(variable, "IDRAK_CLUSTER_PROFILES_FILE") is { } profiles) o = o with { ProfilesFile = profiles };
        if (Read(variable, "IDRAK_CLUSTER_SEED") is { } seed) s = s with { Seed = int.TryParse(seed, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) ? v : throw Bad("IDRAK_CLUSTER_SEED", seed, "a whole number") };
        if (Read(variable, "IDRAK_CLUSTER_MODE") is { } mode) s = s with { Mode = Choice<SimulationMode>("IDRAK_CLUSTER_MODE", mode, "in-process or localhost") };
        if (Read(variable, "IDRAK_CLUSTER_THROTTLE") is { } throttle) s = s with { Throttle = Flag("IDRAK_CLUSTER_THROTTLE", throttle) };
        if (Read(variable, "IDRAK_CLUSTER_FAILURES") is { } failures) s = s with { Failures = failures };
        if (Read(variable, "IDRAK_CLUSTER_REPLAY_STEP") is { } step) s = s with { ReplayStep = long.TryParse(step, NumberStyles.None, CultureInfo.InvariantCulture, out long st) ? st : throw Bad("IDRAK_CLUSTER_REPLAY_STEP", step, "a step number, 0 or more") };
        if (Read(variable, "IDRAK_CLUSTER_CLOCK") is { } clock) s = s with { Clock = Choice<ThroughputClock>("IDRAK_CLUSTER_CLOCK", clock, "wall or simulated") };
        return o with { Simulation = s };
    }

    private static string? Read(Func<string, string?> variable, string name) => variable(name) is { } value && value.Trim() is { Length: > 0 } trimmed ? trimmed : null;

    private static int Whole(string name, string value, int min, int max = int.MaxValue) =>
        int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) && v >= min && v <= max
            ? v
            : throw Bad(name, value, max == int.MaxValue ? $"a whole number, {min} or more" : $"a whole number from {min} to {max}");

    private static double Number(string name, string value, string expected, Func<double, bool> valid) =>
        double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out double v) && double.IsFinite(v) && valid(v) ? v : throw Bad(name, value, expected);

    private static long Size(string name, string value)
    {
        try
        {
            return ClusterTopology.ParseMemory(value) is > 0 and var bytes ? bytes : throw Bad(name, value, "a size above 0 (B, KB, MB, GB)");
        }
        catch (FormatException)
        {
            throw Bad(name, value, "a size (B, KB, MB, GB, powers of 1,024)");
        }
    }

    private static bool Flag(string name, string value) => value.ToLowerInvariant() switch
    {
        "1" or "true" or "on" => true,
        "0" or "false" or "off" => false,
        _ => throw Bad(name, value, "1 or 0"),
    };

    private static T Choice<T>(string name, string value, string expected)
        where T : struct, Enum =>
        Enum.TryParse<T>(value.Replace("-", "", StringComparison.Ordinal), ignoreCase: true, out var v) && Enum.IsDefined(v) && !char.IsAsciiDigit(value[0])
            ? v
            : throw Bad(name, value, expected);

    private static ArgumentException Bad(string name, string value, string expected) => new($"{name}={value} is not {expected}.");
}
