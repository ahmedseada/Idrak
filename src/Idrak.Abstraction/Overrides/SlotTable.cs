// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics.CodeAnalysis;

namespace Idrak.Abstraction;

/// <summary>
/// The entries of a registry, each a slot with two layers: the library default and what an app registered over it.
/// Every registry of the library keeps its entries here.
/// <list type="bullet">
/// <item>A registration made while the library registers its built-ins (<see cref="Overrides.AsLibraryDefaults"/>, or
/// <see cref="RegisterDefault"/>) is the slot's library default; any other is the app's, and it shadows the default
/// without removing it. <see cref="Unregister"/> removes the app's registration, which brings the default back; a
/// library default is never removed.</item>
/// <item>While an app's registration shadows a default, lookups get it guarded by the registry's
/// <see cref="SlotGuard{TValue}"/> under the slot's <see cref="SlotPolicy"/> (<see cref="SlotPolicy.Throw"/>, reported,
/// unless set). A slot with only its default hands the default out as it is: overriding nothing costs nothing.</item>
/// <item>A library default has a version (1 unless the library says otherwise): when the library improves a default, it
/// registers the new one under a higher version and the release it came with, and the previous ones stay reachable by
/// version (<see cref="Default(TKey, int)"/>) for an app that wants the old behavior back. The report says when an app's
/// registration was built against a release older than the default it shadows (see <see cref="SlotOverride.Outdated"/>).</item>
/// <item>A registry with no guard (its entries are used across calls, such as a key/value cache layout through a
/// sequence, or change what they are given as they run) hands the app's registration out as it is; its policy cannot be
/// set.</item>
/// </list>
/// Lookups take no lock. Entries keep their order: new names go last, or first for registries that ask the newest
/// first; an app's registration takes the place of the default it shadows.
/// </summary>
/// <typeparam name="TKey">The entry's name (or id, or type).</typeparam>
/// <typeparam name="TValue">What is registered: a delegate, an object implementing the contract, a description.</typeparam>
public sealed class SlotTable<TKey, TValue> where TKey : notnull
{
    private sealed class Entry(TKey key, Slot slot)
    {
        public TKey Key { get; } = key;

        public Slot Slot { get; } = slot;

        public bool HasLibrary { get; init; }

        public TValue Library { get; init; } = default!;

        public bool HasApp { get; init; }

        public TValue App { get; init; } = default!;

        public TValue Current { get; init; } = default!;

        // The library's versions of the default, lowest first; Library is the last one's value.
        public LibraryVersion[] Versions { get; init; } = [];
    }

    private sealed record LibraryVersion(int Version, TValue Value, string? Since, string Assembly);

    private sealed class State(Dictionary<TKey, Entry> map, Entry[] ordered)
    {
        public Dictionary<TKey, Entry> Map { get; } = map;

        public Entry[] Ordered { get; } = ordered;

        public TKey[] Keys { get; } = [.. ordered.Select(e => e.Key)];

        public TValue[] Values { get; } = [.. ordered.Select(e => e.Current)];
    }

    private readonly Lock _gate = new();
    private readonly IEqualityComparer<TKey> _comparer;
    private readonly Dictionary<TKey, Slot> _slots;
    private readonly SlotGuard<TValue>? _guard;
    private readonly bool _newestFirst;
    private readonly string _unguarded;
    private readonly string _setPolicy;
    private volatile State _state;

    /// <summary>An empty table for the registry <paramref name="registry"/>.</summary>
    /// <param name="registry">The registry's name in reports and telemetry, for example "RopeScalings".</param>
    /// <param name="guard">How an app's registration is guarded under the slot's policy; null for a registry whose entries cannot fall back.</param>
    /// <param name="comparer">How names compare (the default comparer when null).</param>
    /// <param name="newestFirst">New names go first instead of last (registries that ask their entries in order, the newest first).</param>
    /// <param name="unguarded">Why the entries have no guard, for messages and the report (registries without one).</param>
    /// <param name="setPolicy">
    /// The method that sets an entry's policy, as failure hints name it (<c>"Registry.SetPolicy"</c> when null): the
    /// registry's own method, which takes the entry's name, the policy and the shadow rate.
    /// </param>
    public SlotTable(string registry, SlotGuard<TValue>? guard = null, IEqualityComparer<TKey>? comparer = null, bool newestFirst = false, string? unguarded = null,
        string? setPolicy = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(registry);
        Registry = registry;
        _guard = guard;
        _comparer = comparer ?? EqualityComparer<TKey>.Default;
        _slots = new(_comparer);
        _newestFirst = newestFirst;
        _unguarded = unguarded ?? "its entries have no call boundary to fall back at";
        _setPolicy = setPolicy ?? registry + ".SetPolicy";
        _state = new(new(_comparer), []);
        Overrides.Track(Report);
    }

    /// <summary>The registry's name in reports and telemetry.</summary>
    public string Registry { get; }

    /// <summary>Whether an app's registration is guarded by its slot's policy (the registry gave a <see cref="SlotGuard{TValue}"/>).</summary>
    public bool Guarded => _guard is not null;

    /// <summary>The names, in order.</summary>
    public IReadOnlyList<TKey> Keys => _state.Keys;

    /// <summary>What each name hands out (see <see cref="TryGet"/>), in the order of <see cref="Keys"/>.</summary>
    public IReadOnlyList<TValue> Values => _state.Values;

    /// <summary>
    /// Registers <paramref name="value"/> as <paramref name="key"/>: as its library default while the library registers its
    /// built-ins (<see cref="Overrides.AsLibraryDefaults"/>), otherwise as the app's registration, which shadows the default.
    /// </summary>
    /// <param name="key">The name.</param>
    /// <param name="value">What to hand out.</param>
    /// <param name="registeredBy">
    /// The assembly that registered it, its <see cref="Origin"/> (a registry passes
    /// <see cref="System.Reflection.Assembly.GetCallingAssembly"/> from its own <c>Register</c>); when null, the assembly of
    /// the implementation.
    /// </param>
    /// <param name="implementation">
    /// Where the app's implementation is named from for reports (its type, a delegate's method, or a name), when not
    /// <paramref name="value"/> itself.
    /// </param>
    public void Register(TKey key, TValue value, System.Reflection.Assembly? registeredBy = null, object? implementation = null)
    {
        if (Overrides.RegisteringDefaults)
        {
            RegisterDefault(key, value);
            return;
        }

        ArgumentNullException.ThrowIfNull(key);
        var (id, origin) = Overrides.Describe(implementation ?? value);
        var assembly = registeredBy ?? Overrides.AssemblyOf(implementation ?? value);
        origin = registeredBy?.GetName().Name ?? origin;
        Update(key, old => new Entry(key, old.Slot)
        {
            HasLibrary = old.HasLibrary, Library = old.Library, Versions = old.Versions, HasApp = true, App = value,
        }, slot =>
        {
            slot.Implementation = id;
            slot.Origin = origin;
            slot.AppAssembly = assembly;
            slot.Reset();
        });
    }

    /// <summary>
    /// Registers <paramref name="value"/> as version <paramref name="version"/> of the library default of
    /// <paramref name="key"/> (for the library's own built-ins). The highest version registered is the default; the others
    /// stay reachable with <see cref="Default(TKey, int)"/>. Registering a version again replaces it.
    /// </summary>
    /// <param name="key">The name.</param>
    /// <param name="value">The library's implementation.</param>
    /// <param name="version">Its version, from 1; a better default replacing an older one takes a higher version.</param>
    /// <param name="since">The release that made this version the default ("0.4.0"); null for the first version.</param>
    public void RegisterDefault(TKey key, TValue value, int version = 1, string? since = null)
    {
        ArgumentNullException.ThrowIfNull(key);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(version);
        if (since is not null && !Version.TryParse(since, out _))
        {
            throw new ArgumentException($"'{since}' is not a release number such as 0.4.0.", nameof(since));
        }

        string assembly = Overrides.AssemblyOf(value)?.GetName().Name ?? "unknown";
        Update(key, old =>
        {
            LibraryVersion[] versions = [.. old.Versions.Where(v => v.Version != version).Append(new LibraryVersion(version, value, since, assembly)).OrderBy(v => v.Version)];
            return new Entry(key, old.Slot) { HasLibrary = true, Library = versions[^1].Value, Versions = versions, HasApp = old.HasApp, App = old.App };
        }, null);
    }

    /// <summary>
    /// Removes the app's registration of <paramref name="key"/>: a slot with a library default gets it back, a name only the
    /// app registered is gone. False when the app registered nothing under that name (a library default is never removed).
    /// </summary>
    public bool Unregister(TKey key)
    {
        lock (_gate)
        {
            if (!_state.Map.TryGetValue(key, out var old) || !old.HasApp)
            {
                return false;
            }

            Update(key, _ => old.HasLibrary ? new Entry(old.Key, old.Slot) { HasLibrary = true, Library = old.Library, Versions = old.Versions } : null, null);
            return true;
        }
    }

    /// <summary>What <paramref name="key"/> hands out: the app's registration (guarded by its policy when it shadows a default), else the library default.</summary>
    public bool TryGet(TKey key, [MaybeNullWhen(false)] out TValue value)
    {
        if (_state.Map.TryGetValue(key, out var entry))
        {
            value = entry.Current;
            return true;
        }

        value = default;
        return false;
    }

    /// <summary>What <paramref name="key"/> hands out (see <see cref="TryGet"/>), or the type's default when nothing is registered.</summary>
    public TValue? Find(TKey key) => _state.Map.TryGetValue(key, out var entry) ? entry.Current : default;

    /// <summary>Whether anything is registered as <paramref name="key"/>.</summary>
    public bool Contains(TKey key) => _state.Map.ContainsKey(key);

    /// <summary>Whether the library has a default for <paramref name="key"/>.</summary>
    public bool HasDefault(TKey key) => _state.Map.TryGetValue(key, out var entry) && entry.HasLibrary;

    /// <summary>The library default of <paramref name="key"/>, whatever an app registered over it; the type's default when the library has none.</summary>
    public TValue? Default(TKey key) => _state.Map.TryGetValue(key, out var entry) && entry.HasLibrary ? entry.Library : default;

    /// <summary>
    /// Version <paramref name="version"/> of the library default of <paramref name="key"/> (an older default the library
    /// replaced, kept for an app that wants its behavior back); the type's default when the library has no such version.
    /// </summary>
    public TValue? Default(TKey key, int version) =>
        _state.Map.TryGetValue(key, out var entry) && entry.Versions.FirstOrDefault(v => v.Version == version) is { } found ? found.Value : default;

    /// <summary>The version of the library default of <paramref name="key"/> (the highest registered); 0 when the library has none.</summary>
    public int DefaultVersion(TKey key) => _state.Map.TryGetValue(key, out var entry) && entry.Versions.Length > 0 ? entry.Versions[^1].Version : 0;

    /// <summary>The versions of the library default of <paramref name="key"/> that can be asked for, lowest first.</summary>
    public IReadOnlyList<int> DefaultVersions(TKey key) =>
        _state.Map.TryGetValue(key, out var entry) ? [.. entry.Versions.Select(v => v.Version)] : [];

    /// <summary>
    /// Who registered what <paramref name="key"/> hands out: <see cref="Overrides.Library"/>, or the name of the assembly
    /// the app's implementation comes from; null when nothing is registered.
    /// </summary>
    public string? Origin(TKey key) =>
        _state.Map.TryGetValue(key, out var entry) ? entry.HasApp ? entry.Slot.Origin : Overrides.Library : null;

    /// <summary>The policy of <paramref name="key"/>'s slot (<see cref="SlotPolicy.Throw"/> unless set).</summary>
    public SlotPolicy Policy(TKey key) => SlotOf(key).Policy;

    /// <summary>
    /// Sets the policy of <paramref name="key"/>'s slot; it applies at once, also to what was handed out before, and stays
    /// when the app registers again. Allowed before anything is registered under the name.
    /// </summary>
    /// <param name="key">The name.</param>
    /// <param name="policy">What happens to the app's calls.</param>
    /// <param name="shadowRate">The share of calls compared under <see cref="SlotPolicy.Shadow"/>, from 0 to 1.</param>
    /// <exception cref="NotSupportedException">The registry has no guard: its entries cannot fall back.</exception>
    public void SetPolicy(TKey key, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate)
    {
        if (_guard is null)
        {
            throw new NotSupportedException($"{Registry} entries have no failure policy: {_unguarded}. An app's registration is used as it is.");
        }

        SlotOf(key).Set(policy, shadowRate);
    }

    /// <summary>The slot of <paramref name="key"/>: its app implementation, policy and counts.</summary>
    public Slot SlotOf(TKey key)
    {
        ArgumentNullException.ThrowIfNull(key);
        lock (_gate)
        {
            if (!_slots.TryGetValue(key, out var slot))
            {
                string name = Convert.ToString(key, System.Globalization.CultureInfo.InvariantCulture) ?? "";
                _slots[key] = slot = new Slot(Registry, name, $"{_setPolicy}({(key is string ? $"\"{name}\"" : name)}, ");
            }

            return slot;
        }
    }

    /// <summary>Every app registration in the table, in order: what it replaces, its origin and policy (see <see cref="Overrides.Report"/>).</summary>
    public IReadOnlyList<SlotOverride> Report() =>
        [.. _state.Ordered.Where(e => e.HasApp).Select(e =>
        {
            var current = e.Versions.Length > 0 ? e.Versions[^1] : null;
            return new SlotOverride(Registry, e.Slot.Name, e.Slot.Implementation, e.Slot.Origin, e.HasLibrary, Guarded, e.Slot.Policy, e.Slot.ShadowRate,
                e.Slot.Failures, e.Slot.FallBacks, e.Slot.Compared, e.Slot.Differed)
            {
                DefaultVersion = current?.Version ?? 0,
                DefaultSince = current?.Since,
                BuiltAgainst = current is null ? null : Overrides.BuiltAgainst(e.Slot.AppAssembly, current.Assembly),
            };
        })];

    // Replaces the entry of `key` with make(old) (null: removed), under the lock, and publishes a new state.
    private void Update(TKey key, Func<Entry, Entry?> make, Action<Slot>? touch)
    {
        lock (_gate)
        {
            var state = _state;
            var slot = SlotOf(key);
            touch?.Invoke(slot);
            bool existed = state.Map.TryGetValue(key, out var old);
            var made = make(old ?? new Entry(key, slot));
            if (made is not null)
            {
                made = new Entry(key, slot)
                {
                    HasLibrary = made.HasLibrary, Library = made.Library, Versions = made.Versions, HasApp = made.HasApp, App = made.App,
                    Current = !made.HasApp ? made.Library : made.HasLibrary && _guard is not null ? _guard(slot, made.App, made.Library) : made.App,
                };
            }

            var ordered = new List<Entry>(state.Ordered);
            int at = existed ? ordered.FindIndex(e => _comparer.Equals(e.Key, key)) : -1;
            if (made is null)
            {
                ordered.RemoveAt(at);
            }
            else if (at >= 0)
            {
                ordered[at] = made;
            }
            else if (_newestFirst)
            {
                ordered.Insert(0, made);
            }
            else
            {
                ordered.Add(made);
            }

            var map = new Dictionary<TKey, Entry>(_comparer);
            foreach (var e in ordered)
            {
                map[e.Key] = e;
            }

            _state = new State(map, [.. ordered]);
        }
    }
}
