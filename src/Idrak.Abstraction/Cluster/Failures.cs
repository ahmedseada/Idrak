// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Cluster;

/// <summary>
/// What goes wrong in a fault. The simulator applies each from the step it is drawn at; a duration is the affected
/// rank's (or, for a link, the sender's) simulated time from the start of that step, zero for the rest of the run.
/// </summary>
public enum FaultKind
{
    /// <summary>
    /// A rank runs slower: its transfers take <see cref="Fault.Factor"/> times as long (1 or more), so every rank that
    /// waits on it is blocked longer and names it as the slowest.
    /// </summary>
    SlowWorker,

    /// <summary>A rank stops at the start of the step: its links close, and every other rank's step ends naming it.</summary>
    DroppedWorker,

    /// <summary>
    /// The link between <see cref="Fault.Ranks"/>' two ranks loses messages at <see cref="Fault.Factor"/> more (a share from 0
    /// to 1), both ways; each lost message is sent again, so the step completes later.
    /// </summary>
    LinkLoss,

    /// <summary>
    /// The links between <see cref="Fault.Ranks"/> and the other ranks close for <see cref="Fault.Duration"/>: a message
    /// across waits until they open again. Closed for the rest of the run, or for the timeout or longer, the step times out
    /// on every rank naming both sides.
    /// </summary>
    Partition,

    /// <summary>
    /// A rank connects <see cref="Fault.Factor"/> steps late: the others run those steps as a group without it, and it joins
    /// at the step after, its clock moved to theirs.
    /// </summary>
    LateJoin,
}

/// <summary>One fault drawn for a step.</summary>
/// <param name="Kind">What goes wrong.</param>
/// <param name="Step">The step it starts at.</param>
/// <param name="Ranks">The ranks it touches (for a partition, the ranks on one side).</param>
/// <param name="Duration">How long it lasts, or zero for the rest of the run.</param>
/// <param name="Factor">The kind's figure (see <see cref="FaultKind"/>).</param>
public readonly record struct Fault(FaultKind Kind, long Step, IReadOnlyList<int> Ranks, TimeSpan Duration, double Factor);

/// <summary>
/// Draws the faults of a simulated run, step by step, from a seeded generator (plan 23), so a failure is replayed bit for
/// bit from its seed and step (<c>idrak cluster replay</c>).
/// </summary>
public interface IFailureModel
{
    /// <summary>The model's name in <see cref="FailureModels"/> ("none", "slow-worker", "mixed", …).</summary>
    string Name { get; }

    /// <summary>One line for listings.</summary>
    string Summary { get; }

    /// <summary>The faults that start at <paramref name="step"/>, drawn from <paramref name="seeded"/> alone.</summary>
    IReadOnlyList<Fault> Draw(long step, IClusterTopology topology, Random seeded);
}

/// <summary>
/// The failure models, by name. Idrak.Cluster registers "none", "slow-worker", "dropped-worker", "link-loss", "partition",
/// "late-join" and "mixed".
/// </summary>
public static class FailureModels
{
    private static readonly SlotTable<string, IFailureModel> Table = new(nameof(FailureModels), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a model's faults are drawn from a seeded generator, so another model cannot give the same run to replay");

    static FailureModels() => LibraryDefaults.Ensure(typeof(FailureModels));

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>Registers <paramref name="model"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IFailureModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(model.Name);
        Table.Register(model.Name, model, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's model of that name (the library's, if any, answers again).</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The model of that name, or null.</summary>
    public static IFailureModel? Find(string name) => Table.Find(name);

    /// <summary>The model of that name.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name.</exception>
    public static IFailureModel Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(FailureModels), name, Table.Keys,
            $"Failure model '{name}' is not registered (registered: {ClusterNames.List(Table.Keys)}); add one with FailureModels.Register.");

    /// <summary>The library's model of that name, or null.</summary>
    public static IFailureModel? Default(string name) => Table.Default(name);

    /// <summary>Who registered the model of that name.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each model as "name: summary", one a line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(m => $"{m.Name}: {m.Summary}"));
}
