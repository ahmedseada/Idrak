// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics.CodeAnalysis;
using System.Reflection;

namespace Idrak.Abstraction;

/// <summary>
/// The registry of a contract's kinds: a <see cref="SlotTable{TKey, TValue}"/> of names to kinds (its two layers, the
/// library default and an app's registration over it, with their policies, versions and the overrides report) that
/// registers every kind under its own <see cref="IKind.Name"/>, finds kinds by a name read from input without making a
/// string, and words every listing and "not registered" error alike. The table is safe to read from many threads while
/// kinds are registered: lookups take no lock.
/// </summary>
/// <typeparam name="TKind">The contract (a class: a null or type test boxes nothing).</typeparam>
public sealed class KindTable<TKind> where TKind : class, IKind
{
    private readonly string _kind, _remedy;

    /// <summary>An empty table for the registry <paramref name="registry"/>.</summary>
    /// <param name="registry">The registry's name in reports, telemetry and its policy calls ("EnvironmentKinds").</param>
    /// <param name="kind">What one entry is called in messages, capitalised ("Environment kind", "Optimizer").</param>
    /// <param name="remedy">What a "not registered" error tells the caller to do ("add it with EnvironmentKinds.Register").</param>
    /// <param name="guard">How an app's registration is guarded under its slot's policy; null for kinds that cannot fall back.</param>
    /// <param name="comparer">How names compare (ordinal when null).</param>
    /// <param name="newestFirst">New names go first instead of last.</param>
    /// <param name="unguarded">Why the kinds have no guard, for messages and the report (tables without one).</param>
    /// <param name="setPolicy">The method that sets a kind's policy, as failure hints name it (<c>"Registry.SetPolicy"</c> when null).</param>
    public KindTable(string registry, string kind, string remedy, SlotGuard<TKind>? guard = null, IEqualityComparer<string>? comparer = null,
        bool newestFirst = false, string? unguarded = null, string? setPolicy = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(kind);
        ArgumentException.ThrowIfNullOrWhiteSpace(remedy);
        _kind = kind;
        _remedy = remedy;
        Slots = new SlotTable<string, TKind>(registry, guard, comparer ?? StringComparer.Ordinal, newestFirst, unguarded, setPolicy);
    }

    /// <summary>The slots behind the table: versions of the library defaults, policies and the overrides report.</summary>
    public SlotTable<string, TKind> Slots { get; }

    /// <summary>The registry's name in reports and telemetry.</summary>
    public string Registry => Slots.Registry;

    /// <summary>The registered names, in order.</summary>
    public IReadOnlyList<string> Names => Slots.Keys;

    /// <summary>What each name hands out (see <see cref="TryGet"/>), in the order of <see cref="Names"/>.</summary>
    public IReadOnlyList<TKind> Kinds => Slots.Values;

    /// <summary>
    /// Registers <paramref name="kind"/> under its <see cref="IKind.Name"/>: as its library default while the library
    /// registers its built-ins (<see cref="Overrides.AsLibraryDefaults"/>), otherwise as the app's registration, which
    /// shadows the default. Its origin in the report is the assembly its type comes from.
    /// </summary>
    /// <exception cref="ArgumentException">The kind has no name.</exception>
    public void Register(TKind kind) => Register(kind, registeredBy: null);

    /// <summary>Registers <paramref name="kind"/> as <see cref="Register(TKind)"/> does, naming <paramref name="registeredBy"/> as its origin.</summary>
    /// <param name="kind">The kind.</param>
    /// <param name="registeredBy">
    /// The assembly that registered it (a registry passes <see cref="Assembly.GetCallingAssembly"/> from its own
    /// <c>Register</c>); when null, the assembly of the kind's type.
    /// </param>
    public void Register(TKind kind, Assembly? registeredBy)
    {
        ArgumentNullException.ThrowIfNull(kind);
        ArgumentException.ThrowIfNullOrWhiteSpace(kind.Name, nameof(kind));
        Slots.Register(kind.Name, kind, registeredBy);
    }

    /// <summary>Registers each of <paramref name="kinds"/> under its name, in order (see <see cref="Register(TKind)"/>).</summary>
    public void Register(params ReadOnlySpan<TKind> kinds)
    {
        foreach (var kind in kinds)
        {
            Register(kind, registeredBy: null);
        }
    }

    /// <summary>
    /// Registers <paramref name="kind"/> as version <paramref name="version"/> of the library default of its name (see
    /// <see cref="SlotTable{TKey, TValue}.RegisterDefault"/>).
    /// </summary>
    public void RegisterDefault(TKind kind, int version = 1, string? since = null)
    {
        ArgumentNullException.ThrowIfNull(kind);
        ArgumentException.ThrowIfNullOrWhiteSpace(kind.Name, nameof(kind));
        Slots.RegisterDefault(kind.Name, kind, version, since);
    }

    /// <summary>Removes the app's kind <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public bool Unregister(string name) => Slots.Unregister(name);

    /// <summary>What <paramref name="name"/> hands out: the app's kind (guarded by its policy when it shadows a default), else the library's.</summary>
    public bool TryGet(ReadOnlySpan<char> name, [MaybeNullWhen(false)] out TKind kind) => Slots.TryGet(name, out kind);

    /// <summary>The kind registered as <paramref name="name"/>, or null.</summary>
    public TKind? Find(ReadOnlySpan<char> name) => Slots.Find(name);

    /// <summary>Whether a kind is registered as <paramref name="name"/>.</summary>
    public bool Contains(ReadOnlySpan<char> name) => Slots.Contains(name);

    /// <summary>The kind registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones and what to do (<see cref="NotRegistered"/>).</exception>
    public TKind Get(ReadOnlySpan<char> name) => Slots.Find(name) ?? throw new NotSupportedException(NotRegistered(name));

    /// <summary>The library's kind <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public TKind? Default(string name) => Slots.Default(name);

    /// <summary>Version <paramref name="version"/> of the library's kind <paramref name="name"/>; null when there is none.</summary>
    public TKind? Default(string name, int version) => Slots.Default(name, version);

    /// <summary>The version of the library's kind <paramref name="name"/> (0 when the library has none).</summary>
    public int DefaultVersion(string name) => Slots.DefaultVersion(name);

    /// <summary>Who registered the kind <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public string? Origin(string name) => Slots.Origin(name);

    /// <summary>Sets the policy of the app's kind <paramref name="name"/> (see <see cref="SlotTable{TKey, TValue}.SetPolicy"/>).</summary>
    /// <exception cref="NotSupportedException">The table has no guard: its kinds cannot fall back.</exception>
    public void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Slots.SetPolicy(name, policy, shadowRate);

    /// <summary>The registered names joined for a message, in order: "a, b, c", or "none".</summary>
    public string Listed()
    {
        var names = Slots.Keys;
        return names.Count == 0 ? "none" : string.Join(", ", names);
    }

    /// <summary>Each registered kind as "name: summary", one per line, in order (for <c>--list</c> and help pages).</summary>
    public string Describe() => string.Join("\n", Slots.Values.Select(k => $"{k.Name}: {k.Summary}"));

    /// <summary>
    /// The error for a name nothing is registered as, worded as every registry words it: "Environment kind 'x' is not
    /// registered (registered: a, b); add it with EnvironmentKinds.Register."
    /// </summary>
    public string NotRegistered(ReadOnlySpan<char> name) => $"{_kind} '{name}' is not registered (registered: {Listed()}); {_remedy}.";
}
