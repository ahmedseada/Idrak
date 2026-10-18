// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Cluster;

/// <summary>
/// Picks the transport for each pair of ranks from those available between them (plan 23). The choice is published as
/// <c>TransportChosen</c> once per pair; <see cref="ClusterOptions.Transport"/> forces one.
/// </summary>
public interface ITransportSelector
{
    /// <summary>The selector's name in <see cref="TransportSelectors"/> ("fastest").</summary>
    string Name { get; }

    /// <summary>One line for listings.</summary>
    string Summary { get; }

    /// <summary>
    /// The transport (a <see cref="Transports"/> name) for ranks <paramref name="rankA"/> and <paramref name="rankB"/>,
    /// from <paramref name="available"/> (every registered transport's answer for the pair), with the reason logged.
    /// </summary>
    /// <exception cref="NotSupportedException">None of them can join the pair; the message lists each one's reason.</exception>
    (string Transport, string Reason) Choose(int rankA, int rankB, IClusterTopology topology, IReadOnlyList<TransportAvailability> available);
}

/// <summary>The transport selectors, by name. Idrak.Network registers "fastest" (the first available in the order fastest first).</summary>
public static class TransportSelectors
{
    private static readonly SlotTable<string, ITransportSelector> Table = new(nameof(TransportSelectors), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a group's ranks must make the same choice for a pair, so a selector never falls back to another for one call");

    static TransportSelectors() => LibraryDefaults.Ensure(typeof(TransportSelectors));

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>Registers <paramref name="selector"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ITransportSelector selector)
    {
        ArgumentNullException.ThrowIfNull(selector);
        ArgumentException.ThrowIfNullOrWhiteSpace(selector.Name);
        Table.Register(selector.Name, selector, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's selector of that name (the library's, if any, answers again).</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The selector of that name, or null.</summary>
    public static ITransportSelector? Find(string name) => Table.Find(name);

    /// <summary>The selector of that name.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name.</exception>
    public static ITransportSelector Get(string name) => Table.Find(name)
        ?? throw new NotSupportedException($"Transport selector '{name}' is not registered (registered: {ClusterNames.List(Table.Keys)}); add one with TransportSelectors.Register.");

    /// <summary>The library's selector of that name, or null.</summary>
    public static ITransportSelector? Default(string name) => Table.Default(name);

    /// <summary>Who registered the selector of that name.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each selector as "name: summary", one a line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(s => $"{s.Name}: {s.Summary}"));
}

/// <summary>
/// The transport joining each pair of ranks of a group (plan 23, phase C3), made once when the group starts: every
/// registered transport says whether it can join the pair here (<see cref="ITransportProvider.Availability"/>), and the
/// selector <see cref="ClusterOptions.Selector"/> names picks among them, so different pairs may use different pipes (peer
/// copies inside a machine, TCP between machines). A transport <see cref="ClusterOptions.Transport"/> forces wins over the
/// selector for every pair it can join. Each choice is published once per pair as <see cref="TransportChosen"/>.
/// </summary>
public sealed class TransportPlan
{
    private readonly (string Transport, string Reason)[] _pairs;

    private TransportPlan(int worldSize, (string, string)[] pairs, IReadOnlyList<string> used)
    {
        WorldSize = worldSize;
        _pairs = pairs;
        Used = used;
    }

    /// <summary>The number of ranks.</summary>
    public int WorldSize { get; }

    /// <summary>The transports the plan uses, in the order of their first pair.</summary>
    public IReadOnlyList<string> Used { get; }

    /// <summary>The one transport every pair uses, or null when pairs use different ones (or there is no pair).</summary>
    public string? Single => Used.Count == 1 ? Used[0] : null;

    /// <summary>The transport joining <paramref name="rankA"/> and <paramref name="rankB"/> (either order), and why.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Not two different ranks of the group.</exception>
    public (string Transport, string Reason) this[int rankA, int rankB] => _pairs[Index(rankA, rankB)];

    /// <summary>
    /// Plans every pair of <paramref name="topology"/> under <paramref name="options"/> and publishes each choice once
    /// (<see cref="Telemetry.TransportChosen"/>, at <see cref="TelemetryLevel.Network"/>).
    /// </summary>
    /// <exception cref="NotSupportedException">
    /// The forced transport or the selector is not registered; the forced transport cannot join a pair (the message names the
    /// pair and the transport's reason); or no transport can join a pair (the selector names each one's reason).
    /// </exception>
    public static TransportPlan Make(IClusterTopology topology, ClusterOptions options)
    {
        ArgumentNullException.ThrowIfNull(topology);
        ArgumentNullException.ThrowIfNull(options);
        int world = topology.WorldSize;
        bool auto = string.Equals(options.Transport, "auto", StringComparison.OrdinalIgnoreCase);
        var forced = auto ? null : Transports.Get(options.Transport);
        var selector = auto ? TransportSelectors.Get(options.Selector) : null;
        IReadOnlyList<ITransportProvider> providers = auto ? Transports.All : [];
        var answers = new TransportAvailability[providers.Count];
        var pairs = new (string, string)[checked(world * (world - 1) / 2)];
        var used = new List<string>();
        bool publish = Telemetry.IsEnabled(TelemetryLevel.Network);
        for (int a = 0; a < world; a++)
        {
            var context = new ClusterContext(topology, a, options);
            for (int b = a + 1; b < world; b++)
            {
                (string Transport, string Reason) choice;
                if (forced is not null)
                {
                    var answer = forced.Availability(context, b);
                    choice = answer.Available
                        ? (forced.Name, $"forced by the options (IDRAK_CLUSTER_TRANSPORT): {answer.Reason}")
                        : throw new NotSupportedException($"Transport '{forced.Name}' is forced (ClusterOptions.Transport, IDRAK_CLUSTER_TRANSPORT) but cannot join ranks {a} and {b}: {answer.Reason}.");
                }
                else
                {
                    for (int i = 0; i < providers.Count; i++)
                    {
                        answers[i] = providers[i].Availability(context, b);
                    }

                    choice = selector!.Choose(a, b, topology, answers);
                }

                pairs[Index(world, a, b)] = choice;
                if (!used.Contains(choice.Transport, StringComparer.OrdinalIgnoreCase))
                {
                    used.Add(choice.Transport);
                }

                if (publish)
                {
                    Telemetry.TransportChosen(new TransportChosen(a, b, choice.Transport, choice.Reason));
                }
            }
        }

        return new TransportPlan(world, pairs, used);
    }

    private int Index(int rankA, int rankB)
    {
        if (rankA == rankB || (uint)rankA >= (uint)WorldSize || (uint)rankB >= (uint)WorldSize)
        {
            throw new ArgumentOutOfRangeException(nameof(rankB), $"Ranks {rankA} and {rankB} are not two different ranks of 0 to {WorldSize - 1}.");
        }

        return Index(WorldSize, Math.Min(rankA, rankB), Math.Max(rankA, rankB));
    }

    // The pair (a, b), a < b, in row order of the upper triangle.
    private static int Index(int world, int a, int b) => (a * ((2 * world) - a - 1) / 2) + (b - a - 1);
}
