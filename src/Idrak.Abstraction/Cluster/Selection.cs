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
