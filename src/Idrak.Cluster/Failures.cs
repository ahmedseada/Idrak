// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Cluster;

/// <summary>
/// The figures the built-in failure models draw with (plan 23, phase C3). Each has an environment variable, read when a
/// built-in model first draws; an app gives its own with <see cref="BuiltInFailures.Create"/> and
/// <see cref="FailureModels.Register"/>. Durations are simulated time, from the start of the step a fault is drawn at.
/// </summary>
public sealed record FailureSettings
{
    /// <summary>The chance that a step draws a fault, from 0 to 1 (<c>IDRAK_CLUSTER_FAULT_RATE</c>).</summary>
    public double Rate { get; init; } = 0.1;

    /// <summary>How many times as long a slow worker's transfers take, 1 or more (<c>IDRAK_CLUSTER_SLOW_FACTOR</c>).</summary>
    public double SlowFactor { get; init; } = 4;

    /// <summary>
    /// How long a slow worker, a lossy link or a partition lasts on the virtual clock; zero for the rest of the run, which
    /// makes a partition fatal (<c>IDRAK_CLUSTER_FAULT_DURATION</c>, seconds).
    /// </summary>
    public TimeSpan Duration { get; init; }

    /// <summary>The share of messages a lossy link loses on top of its profile's, from 0 to 1 (<c>IDRAK_CLUSTER_LOSS_SHARE</c>).</summary>
    public double LossShare { get; init; } = 0.2;

    /// <summary>How many steps late a late joiner connects, 1 or more (<c>IDRAK_CLUSTER_LATE_STEPS</c>).</summary>
    public int LateSteps { get; init; } = 2;

    /// <summary>The environment variables these settings read, in order.</summary>
    public static IReadOnlyList<string> Variables { get; } =
        ["IDRAK_CLUSTER_FAULT_RATE", "IDRAK_CLUSTER_SLOW_FACTOR", "IDRAK_CLUSTER_FAULT_DURATION", "IDRAK_CLUSTER_LOSS_SHARE", "IDRAK_CLUSTER_LATE_STEPS"];

    /// <summary><paramref name="defaults"/> (or the defaults) with what the environment sets (<see cref="Variables"/>).</summary>
    /// <exception cref="ArgumentException">A value that does not parse; the message names the variable and what it takes.</exception>
    public static FailureSettings FromEnvironment(FailureSettings? defaults = null) => From(Environment.GetEnvironmentVariable, defaults);

    internal static FailureSettings From(Func<string, string?> variable, FailureSettings? defaults)
    {
        var s = defaults ?? new FailureSettings();
        if (Read(variable, "IDRAK_CLUSTER_FAULT_RATE") is { } rate) s = s with { Rate = Number("IDRAK_CLUSTER_FAULT_RATE", rate, "a share from 0 to 1", v => v is >= 0 and <= 1) };
        if (Read(variable, "IDRAK_CLUSTER_SLOW_FACTOR") is { } slow) s = s with { SlowFactor = Number("IDRAK_CLUSTER_SLOW_FACTOR", slow, "a factor of 1 or more", v => v >= 1) };
        if (Read(variable, "IDRAK_CLUSTER_FAULT_DURATION") is { } duration) s = s with { Duration = TimeSpan.FromSeconds(Number("IDRAK_CLUSTER_FAULT_DURATION", duration, "a number of seconds, 0 or more", v => v >= 0)) };
        if (Read(variable, "IDRAK_CLUSTER_LOSS_SHARE") is { } loss) s = s with { LossShare = Number("IDRAK_CLUSTER_LOSS_SHARE", loss, "a share from 0 to 1", v => v is >= 0 and <= 1) };
        if (Read(variable, "IDRAK_CLUSTER_LATE_STEPS") is { } late)
        {
            s = s with { LateSteps = int.TryParse(late, NumberStyles.Integer, CultureInfo.InvariantCulture, out int steps) && steps >= 1 ? steps : throw Bad("IDRAK_CLUSTER_LATE_STEPS", late, "a whole number of steps, 1 or more") };
        }

        return s;
    }

    /// <summary>Throws when a figure is out of range.</summary>
    /// <exception cref="ArgumentException">The figure and its range.</exception>
    public void Validate()
    {
        if (Rate is < 0 or > 1 or double.NaN || LossShare is < 0 or > 1 or double.NaN)
        {
            throw new ArgumentException($"Failure settings: the rate ({Rate}) and the loss share ({LossShare}) are shares from 0 to 1.");
        }

        if (!(SlowFactor >= 1) || double.IsInfinity(SlowFactor) || Duration < TimeSpan.Zero || LateSteps < 1)
        {
            throw new ArgumentException($"Failure settings: the slow factor ({SlowFactor}) is 1 or more, the duration ({Duration}) 0 or more, the late steps ({LateSteps}) 1 or more.");
        }
    }

    private static string? Read(Func<string, string?> variable, string name) => variable(name) is { } value && value.Trim() is { Length: > 0 } trimmed ? trimmed : null;

    private static double Number(string name, string value, string expected, Func<double, bool> valid) =>
        double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out double v) && double.IsFinite(v) && valid(v) ? v : throw Bad(name, value, expected);

    private static ArgumentException Bad(string name, string value, string expected) => new($"{name}={value} is not {expected}.");
}

/// <summary>
/// The built-in failure models (plan 23, phase C3), registered in <see cref="FailureModels"/> by Idrak.Cluster: "none";
/// one model per kind of fault ("slow-worker", "dropped-worker", "link-loss", "partition", "late-join"), each drawing at
/// most one fault a step with <see cref="FailureSettings.Rate"/>; and "mixed", which draws any kind a run survives (a slow
/// worker, a lossy link, a late join). A late join concerns the start of a run, so it is drawn at step 0 only. Every draw
/// comes from the generator the injector seeds with (seed, step), so a step's faults are the same on every run and replay.
/// </summary>
public static class BuiltInFailures
{
    /// <summary>The built-in models' names.</summary>
    public static IReadOnlyList<string> Names { get; } = ["none", "slow-worker", "dropped-worker", "link-loss", "partition", "late-join", "mixed"];

    /// <summary>The built-in model <paramref name="name"/> drawing with <paramref name="settings"/> (null: the environment's, read at its first draw).</summary>
    /// <exception cref="NotSupportedException">Not a built-in name.</exception>
    /// <exception cref="ArgumentException">A setting out of range.</exception>
    public static IFailureModel Create(string name, FailureSettings? settings = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        settings?.Validate();
        var given = settings is null ? null : new Lazy<FailureSettings>(() => settings);
        return name.ToLowerInvariant() switch
        {
            "none" => new NoFailures(),
            "slow-worker" => new SeededFailures("slow-worker", "a rank's transfers take longer (IDRAK_CLUSTER_SLOW_FACTOR times)", [FaultKind.SlowWorker], given),
            "dropped-worker" => new SeededFailures("dropped-worker", "a rank stops and its links close: the step ends on every rank naming it", [FaultKind.DroppedWorker], given),
            "link-loss" => new SeededFailures("link-loss", "a link between two ranks loses a share of its messages, each sent again", [FaultKind.LinkLoss], given),
            "partition" => new SeededFailures("partition", "the links between two sets of ranks close (for the rest of the run: the step times out naming both sides)", [FaultKind.Partition], given),
            "late-join" => new SeededFailures("late-join", "a rank connects some steps late; the others run those steps without it", [FaultKind.LateJoin], given),
            "mixed" => new SeededFailures("mixed", "any fault a run survives: a slow worker, a lossy link, a late join", [FaultKind.SlowWorker, FaultKind.LinkLoss, FaultKind.LateJoin], given),
            _ => throw new NotSupportedException($"'{name}' is not a built-in failure model ({string.Join(", ", Names)})."),
        };
    }
}

/// <summary>The "none" failure model: a run with no faults.</summary>
internal sealed class NoFailures : IFailureModel
{
    public string Name => "none";

    public string Summary => "no faults";

    public IReadOnlyList<Fault> Draw(long step, IClusterTopology topology, Random seeded) => [];
}

// A built-in model: at most one fault a step, of one of `kinds`, with the settings given or the environment's.
internal sealed class SeededFailures(string name, string summary, FaultKind[] kinds, Lazy<FailureSettings>? settings) : IFailureModel
{
    private readonly Lazy<FailureSettings> _settings = settings ?? new Lazy<FailureSettings>(() =>
    {
        var read = FailureSettings.FromEnvironment();
        read.Validate();
        return read;
    });

    public string Name => name;

    public string Summary => summary;

    public IReadOnlyList<Fault> Draw(long step, IClusterTopology topology, Random seeded)
    {
        ArgumentNullException.ThrowIfNull(topology);
        ArgumentNullException.ThrowIfNull(seeded);
        var s = _settings.Value;
        int world = topology.WorldSize;
        var possible = kinds.Where(k => k != FaultKind.LateJoin || step == 0).ToArray();
        if (world < 2 || possible.Length == 0 || !(seeded.NextDouble() < s.Rate))
        {
            return [];
        }

        var kind = possible[seeded.Next(possible.Length)];
        int rank = seeded.Next(world);
        return kind switch
        {
            FaultKind.SlowWorker => [new Fault(kind, step, [rank], s.Duration, s.SlowFactor)],
            FaultKind.DroppedWorker => [new Fault(kind, step, [rank], TimeSpan.Zero, 0)],
            FaultKind.LinkLoss => [new Fault(kind, step, [.. new[] { rank, (rank + 1 + seeded.Next(world - 1)) % world }.Order()], s.Duration, s.LossShare)],
            FaultKind.Partition => [new Fault(kind, step, [.. Enumerable.Range(0, 1 + seeded.Next(world - 1)).Select(i => (rank + i) % world).Order()], s.Duration, 0)],
            _ => [new Fault(kind, step, [rank], TimeSpan.Zero, s.LateSteps)],
        };
    }
}
