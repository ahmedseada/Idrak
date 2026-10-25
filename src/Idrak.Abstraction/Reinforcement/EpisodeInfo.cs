// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections;
using System.Globalization;
using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Reinforcement;

/// <summary>The kinds of value an episode's info holds (<see cref="EpisodeInfoKey{T}"/>).</summary>
public enum EpisodeInfoType : byte
{
    /// <summary>A <see cref="bool"/>.</summary>
    Boolean = 1,

    /// <summary>An <see cref="int"/>.</summary>
    Int32 = 2,

    /// <summary>A <see cref="long"/>.</summary>
    Int64 = 3,

    /// <summary>A <see cref="float"/>.</summary>
    Single = 4,

    /// <summary>A <see cref="double"/>.</summary>
    Double = 5,

    /// <summary>A <see cref="string"/>.</summary>
    String = 6,
}

/// <summary>
/// A key of an episode's info (<see cref="EpisodeInfo"/>): its name and the type of its value. The writer (an
/// environment) and every reader (a reward, a report) share one key, so a value is stored as what it is (a count as an
/// <see cref="int"/>, a flag as a <see cref="bool"/>) and never goes through text between them. See
/// <see cref="EpisodeInfoKey{T}"/>.
/// </summary>
public abstract class EpisodeInfoKey
{
    private protected EpisodeInfoKey(string name, EpisodeInfoType type)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        Name = name;
        Type = type;
    }

    /// <summary>The key's name ("rounds", "verify_passed"): what reports print and an exchange between processes carries.</summary>
    public string Name { get; }

    /// <summary>The type of its value.</summary>
    public EpisodeInfoType Type { get; }

    /// <inheritdoc />
    public override string ToString() => Name;
}

/// <summary>
/// A typed key of an episode's info: <c>new EpisodeInfoKey&lt;int&gt;("rounds")</c>. Values are <see cref="bool"/>,
/// <see cref="int"/>, <see cref="long"/>, <see cref="float"/>, <see cref="double"/> or <see cref="string"/>: what an
/// exchange between processes carries and a report prints, each in its own type.
/// </summary>
/// <typeparam name="T">The value's type.</typeparam>
public sealed class EpisodeInfoKey<T> : EpisodeInfoKey
{
    /// <summary>The key <paramref name="name"/> of <typeparamref name="T"/> values.</summary>
    /// <exception cref="NotSupportedException"><typeparamref name="T"/> is not one of the types an episode's info holds.</exception>
    public EpisodeInfoKey(string name) : base(name, TypeOf())
    {
    }

    private static EpisodeInfoType TypeOf() =>
        typeof(T) == typeof(bool) ? EpisodeInfoType.Boolean
        : typeof(T) == typeof(int) ? EpisodeInfoType.Int32
        : typeof(T) == typeof(long) ? EpisodeInfoType.Int64
        : typeof(T) == typeof(float) ? EpisodeInfoType.Single
        : typeof(T) == typeof(double) ? EpisodeInfoType.Double
        : typeof(T) == typeof(string) ? EpisodeInfoType.String
        : throw new NotSupportedException($"An episode's info holds bool, int, long, float, double and string values, not {typeof(T).Name}.");

    // The value as stored: integers and flags in the bits, floating-point values as a double, text as itself. Branches on
    // typeof(T), which the JIT folds, and reads with Unsafe.As: nothing is boxed (rule 89).
    internal EpisodeInfoValue Store(T value)
    {
        if (typeof(T) == typeof(bool))
        {
            return new EpisodeInfoValue(EpisodeInfoType.Boolean, Unsafe.As<T, bool>(ref value) ? 1 : 0, 0, null);
        }

        if (typeof(T) == typeof(int))
        {
            return new EpisodeInfoValue(EpisodeInfoType.Int32, Unsafe.As<T, int>(ref value), 0, null);
        }

        if (typeof(T) == typeof(long))
        {
            return new EpisodeInfoValue(EpisodeInfoType.Int64, Unsafe.As<T, long>(ref value), 0, null);
        }

        if (typeof(T) == typeof(float))
        {
            return new EpisodeInfoValue(EpisodeInfoType.Single, 0, Unsafe.As<T, float>(ref value), null);
        }

        if (typeof(T) == typeof(double))
        {
            return new EpisodeInfoValue(EpisodeInfoType.Double, 0, Unsafe.As<T, double>(ref value), null);
        }

        string text = Unsafe.As<T, string>(ref value) ?? throw new ArgumentNullException(nameof(value), $"The info '{Name}' is text: give a string, not null.");
        return new EpisodeInfoValue(EpisodeInfoType.String, 0, 0, text);
    }

    internal T Load(in EpisodeInfoValue stored)
    {
        if (stored.Type != Type)
        {
            throw new InvalidOperationException($"The info '{Name}' holds a {stored.Type} value, not a {Type}: its writer and this reader use different keys.");
        }

        if (typeof(T) == typeof(bool))
        {
            bool flag = stored.Bits != 0;
            return Unsafe.As<bool, T>(ref flag);
        }

        if (typeof(T) == typeof(int))
        {
            int number = (int)stored.Bits;
            return Unsafe.As<int, T>(ref number);
        }

        if (typeof(T) == typeof(long))
        {
            long number = stored.Bits;
            return Unsafe.As<long, T>(ref number);
        }

        if (typeof(T) == typeof(float))
        {
            float number = (float)stored.Number;
            return Unsafe.As<float, T>(ref number);
        }

        if (typeof(T) == typeof(double))
        {
            double number = stored.Number;
            return Unsafe.As<double, T>(ref number);
        }

        string text = stored.Text!;
        return Unsafe.As<string, T>(ref text);
    }
}

// One stored value of an episode's info, in its own type.
internal readonly record struct EpisodeInfoValue(EpisodeInfoType Type, long Bits, double Number, string? Text)
{
    // The value as a report prints it: flags as true or false, numbers in the invariant culture, text as it is.
    public string Format() => Type switch
    {
        EpisodeInfoType.Boolean => Bits != 0 ? "true" : "false",
        EpisodeInfoType.Int32 or EpisodeInfoType.Int64 => Bits.ToString(CultureInfo.InvariantCulture),
        EpisodeInfoType.Single => ((float)Number).ToString(CultureInfo.InvariantCulture),
        EpisodeInfoType.Double => Number.ToString(CultureInfo.InvariantCulture),
        _ => Text!,
    };
}

/// <summary>
/// What an environment reports about an episode besides its rewards (rounds used, tests passed, files changed), each value
/// in its own type under an <see cref="EpisodeInfoKey{T}"/> shared by its writer and its readers: an environment
/// <see cref="Set{T}">sets</see> values as a step ends; a reward reads them (<see cref="TryGet{T}"/>,
/// <see cref="Get{T}"/>), a report prints them (<see cref="ToString"/>, <see cref="Entries"/>), and a rollout carries them
/// to other processes (<see cref="Write"/>, <see cref="Read"/>) without turning numbers into text. Not thread-safe; an
/// episode's info is set by its environment, then only read.
/// </summary>
public sealed class EpisodeInfo : IReadOnlyCollection<KeyValuePair<string, string>>
{
    private readonly Dictionary<string, EpisodeInfoValue> _values = new(StringComparer.Ordinal);

    /// <summary>How many values it holds.</summary>
    public int Count => _values.Count;

    /// <summary>Sets <paramref name="key"/> to <paramref name="value"/> (replacing what it held); returns this info, to chain.</summary>
    public EpisodeInfo Set<T>(EpisodeInfoKey<T> key, T value)
    {
        ArgumentNullException.ThrowIfNull(key);
        _values[key.Name] = key.Store(value);
        return this;
    }

    /// <summary>Whether a value is set under <paramref name="key"/>'s name.</summary>
    public bool Contains(EpisodeInfoKey key)
    {
        ArgumentNullException.ThrowIfNull(key);
        return _values.ContainsKey(key.Name);
    }

    /// <summary>The value of <paramref name="key"/>, when one is set.</summary>
    /// <exception cref="InvalidOperationException">A value of another type is set under its name (the writer used another key).</exception>
    public bool TryGet<T>(EpisodeInfoKey<T> key, out T value)
    {
        ArgumentNullException.ThrowIfNull(key);
        if (_values.TryGetValue(key.Name, out var stored))
        {
            value = key.Load(stored);
            return true;
        }

        value = default!;
        return false;
    }

    /// <summary>The value of <paramref name="key"/>.</summary>
    /// <exception cref="KeyNotFoundException">None is set.</exception>
    /// <exception cref="InvalidOperationException">A value of another type is set under its name.</exception>
    public T Get<T>(EpisodeInfoKey<T> key) =>
        TryGet(key, out T value) ? value : throw new KeyNotFoundException($"The episode's info has no '{key.Name}' (it has: {Names()}).");

    /// <summary>The value of <paramref name="key"/>, or <paramref name="fallback"/> when none is set.</summary>
    public T GetOrDefault<T>(EpisodeInfoKey<T> key, T fallback) => TryGet(key, out T value) ? value : fallback;

    /// <summary>Removes the value of <paramref name="key"/>; false when none was set.</summary>
    public bool Remove(EpisodeInfoKey key)
    {
        ArgumentNullException.ThrowIfNull(key);
        return _values.Remove(key.Name);
    }

    /// <summary>A copy of this info (an environment's next step adds to its own).</summary>
    public EpisodeInfo Copy()
    {
        var copy = new EpisodeInfo();
        foreach (var (name, value) in _values)
        {
            copy._values[name] = value;
        }

        return copy;
    }

    /// <summary>Each value by name as a report prints it (flags true or false, numbers in the invariant culture), sorted by name.</summary>
    public IEnumerable<KeyValuePair<string, string>> Entries =>
        _values.OrderBy(p => p.Key, StringComparer.Ordinal).Select(p => new KeyValuePair<string, string>(p.Key, p.Value.Format()));

    /// <summary>The values as "name=value", sorted by name and joined by ", ".</summary>
    public override string ToString() => string.Join(", ", Entries.Select(p => $"{p.Key}={p.Value}"));

    /// <summary>
    /// Writes the values to <paramref name="writer"/>, each with its name and type, for <see cref="Read"/> in another
    /// process: numbers stay numbers.
    /// </summary>
    public void Write(BinaryWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);
        writer.Write(_values.Count);
        foreach (var (name, value) in _values)
        {
            writer.Write(name);
            writer.Write((byte)value.Type);
            switch (value.Type)
            {
                case EpisodeInfoType.Single or EpisodeInfoType.Double:
                    writer.Write(value.Number);
                    break;
                case EpisodeInfoType.String:
                    writer.Write(value.Text!);
                    break;
                default:
                    writer.Write(value.Bits);
                    break;
            }
        }
    }

    /// <summary>The info <see cref="Write"/> wrote.</summary>
    /// <exception cref="InvalidDataException">A type the info does not hold.</exception>
    public static EpisodeInfo Read(BinaryReader reader)
    {
        ArgumentNullException.ThrowIfNull(reader);
        var info = new EpisodeInfo();
        int count = reader.ReadInt32();
        for (int i = 0; i < count; i++)
        {
            string name = reader.ReadString();
            var type = (EpisodeInfoType)reader.ReadByte();
            info._values[name] = type switch
            {
                EpisodeInfoType.Single or EpisodeInfoType.Double => new EpisodeInfoValue(type, 0, reader.ReadDouble(), null),
                EpisodeInfoType.String => new EpisodeInfoValue(type, 0, 0, reader.ReadString()),
                EpisodeInfoType.Boolean or EpisodeInfoType.Int32 or EpisodeInfoType.Int64 => new EpisodeInfoValue(type, reader.ReadInt64(), 0, null),
                _ => throw new InvalidDataException($"The info '{name}' has the type {(byte)type}, which an episode's info does not hold."),
            };
        }

        return info;
    }

    /// <summary>Each value by name as text, sorted by name (see <see cref="Entries"/>).</summary>
    public IEnumerator<KeyValuePair<string, string>> GetEnumerator() => Entries.GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    private string Names() => _values.Count == 0 ? "nothing" : string.Join(", ", _values.Keys.Order(StringComparer.Ordinal));
}

/// <summary>
/// The info keys environments of different families share, so a reward reads them from any of them: the replies of the
/// episode so far (<see cref="Rounds"/>) and the tokens of its last answer (<see cref="Tokens"/>, for a length reward).
/// </summary>
public static class EpisodeInfoKeys
{
    /// <summary>The assistant's replies (the episode's steps) so far: "rounds".</summary>
    public static EpisodeInfoKey<int> Rounds { get; } = new("rounds");

    /// <summary>The tokens of the last answer, counted by the environment's tokenizer: "tokens".</summary>
    public static EpisodeInfoKey<int> Tokens { get; } = new("tokens");
}
