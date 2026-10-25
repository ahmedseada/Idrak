// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections;
using System.Collections.Frozen;

namespace Idrak.Abstraction;

/// <summary>
/// The option keys a kind takes (<see cref="IConfigurableKind.Keys"/>), made once: looked up through a
/// <see cref="FrozenSet{T}"/>, by string or by a span read from input (no string made), and listed in the order they were
/// declared, so help pages and "it takes: …" messages read the same on every run. Keys match exactly (ordinal); a key
/// given twice is kept once, where it first came.
/// </summary>
public sealed class OptionKeys : IReadOnlySet<string>
{
    private readonly string[] _ordered;
    private readonly FrozenSet<string> _set;
    private readonly FrozenSet<string>.AlternateLookup<ReadOnlySpan<char>> _byName;

    /// <summary>The keys <paramref name="keys"/>, in their order.</summary>
    public OptionKeys(params ReadOnlySpan<string> keys) : this(Distinct(keys))
    {
    }

    /// <summary>The keys <paramref name="keys"/>, in their order.</summary>
    public OptionKeys(IEnumerable<string> keys) : this(Distinct([.. keys ?? throw new ArgumentNullException(nameof(keys))]))
    {
    }

    private OptionKeys(string[] ordered)
    {
        _ordered = ordered;
        _set = ordered.ToFrozenSet(StringComparer.Ordinal);
        _byName = _set.GetAlternateLookup<ReadOnlySpan<char>>();
    }

    /// <summary>No keys: a kind that takes no options.</summary>
    public static OptionKeys None { get; } = new();

    /// <summary>How many keys there are.</summary>
    public int Count => _ordered.Length;

    /// <summary>The keys in the order they were declared.</summary>
    public ReadOnlySpan<string> Ordered => _ordered;

    /// <summary>The keys as a frozen set (for set operations; it lists them in no particular order).</summary>
    public FrozenSet<string> Set => _set;

    /// <summary>Whether <paramref name="key"/> is one of the keys.</summary>
    public bool Contains(string key) => _set.Contains(key);

    /// <summary>Whether the key <paramref name="key"/>, read from input, is one of the keys; no string is made.</summary>
    public bool Contains(ReadOnlySpan<char> key) => _byName.Contains(key);

    /// <summary>The keys joined for a message, in order: "a, b, c", or "none".</summary>
    public override string ToString() => _ordered.Length == 0 ? "none" : string.Join(", ", _ordered);

    /// <summary>The keys in the order they were declared.</summary>
    public IEnumerator<string> GetEnumerator() => ((IEnumerable<string>)_ordered).GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    // The keys once each, in their first order.
    private static string[] Distinct(ReadOnlySpan<string> keys)
    {
        var seen = new HashSet<string>(keys.Length, StringComparer.Ordinal);
        var ordered = new List<string>(keys.Length);
        foreach (string key in keys)
        {
            ArgumentException.ThrowIfNullOrEmpty(key, nameof(keys));
            if (seen.Add(key))
            {
                ordered.Add(key);
            }
        }

        return [.. ordered];
    }

    /// <inheritdoc />
    public bool IsProperSubsetOf(IEnumerable<string> other) => _set.IsProperSubsetOf(other);

    /// <inheritdoc />
    public bool IsProperSupersetOf(IEnumerable<string> other) => _set.IsProperSupersetOf(other);

    /// <inheritdoc />
    public bool IsSubsetOf(IEnumerable<string> other) => _set.IsSubsetOf(other);

    /// <inheritdoc />
    public bool IsSupersetOf(IEnumerable<string> other) => _set.IsSupersetOf(other);

    /// <inheritdoc />
    public bool Overlaps(IEnumerable<string> other) => _set.Overlaps(other);

    /// <inheritdoc />
    public bool SetEquals(IEnumerable<string> other) => _set.SetEquals(other);
}
