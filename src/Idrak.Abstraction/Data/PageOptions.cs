// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Abstraction.Data;

/// <summary>
/// The typed readers of settings written as text, shared by every reader of them (a page's <c>KEY=VALUE,...</c>
/// settings, a shape step's options, a pen's or a text binding's keys), so each words its refusals alike: a value or a
/// range <c>MIN..MAX</c>, a chance, a number, a whole number, on or off, one of some choices, and lists, their items
/// between ':' with <c>ITEM*N</c> for N of the same (<c>1:4:1*16</c>). Numbers are read in the invariant culture. Each
/// reader takes what the value is called in its refusal (<c>"Page setting size"</c>, <c>"grid's widths"</c>).
/// </summary>
public static class PageOptions
{
    // The most items a list may stand for (an ITEM*N repeat included).
    private const int MostItems = 100_000;

    /// <summary>No options (keys ignore case).</summary>
    public static IReadOnlyDictionary<string, string> None { get; } = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase).AsReadOnly();

    /// <summary>
    /// Settings written <c>KEY=VALUE,...</c> as pairs, in order: each key trimmed and in lower case, its value trimmed
    /// (empty when the item has no '='); empty items are passed over.
    /// </summary>
    public static IReadOnlyList<KeyValuePair<string, string>> Split(string? settings)
    {
        var pairs = new List<KeyValuePair<string, string>>();
        foreach (string raw in (settings ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            int eq = raw.IndexOf('=', StringComparison.Ordinal);
            pairs.Add(KeyValuePair.Create((eq < 0 ? raw : raw[..eq]).Trim().ToLowerInvariant(), eq < 0 ? "" : raw[(eq + 1)..].Trim()));
        }

        return pairs;
    }

    /// <summary>The entries of <paramref name="options"/> whose key is one of <paramref name="keys"/> (keys ignore case).</summary>
    public static IReadOnlyDictionary<string, string> Only(IReadOnlyDictionary<string, string> options, IReadOnlySet<string> keys)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(keys);
        Dictionary<string, string>? own = null;
        foreach (var (key, value) in options)
        {
            if (keys.Contains(key) || keys.Contains(key.ToLowerInvariant()))
            {
                (own ??= new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase))[key] = value;
            }
        }

        return own is null ? None : own;
    }

    /// <summary>A value, or a range <c>MIN..MAX</c>, each end from <paramref name="least"/> to <paramref name="most"/> and MIN not above MAX (a value is MIN and MAX).</summary>
    /// <exception cref="ArgumentException">Not such a value or range; the message names <paramref name="what"/>.</exception>
    public static (float Min, float Max) Range(string what, string value, float least, float most)
    {
        ArgumentNullException.ThrowIfNull(value);
        int dots = value.IndexOf("..", StringComparison.Ordinal);
        string a = dots < 0 ? value : value[..dots], b = dots < 0 ? value : value[(dots + 2)..];
        return float.TryParse(a, NumberStyles.Float, CultureInfo.InvariantCulture, out float min) && float.TryParse(b, NumberStyles.Float, CultureInfo.InvariantCulture, out float max)
            && min >= least && max <= most && min <= max
            ? (min, max)
            : throw new ArgumentException($"{what} is a value or a range MIN..MAX from {least.ToString(CultureInfo.InvariantCulture)} to {most.ToString(CultureInfo.InvariantCulture)}, not '{value}'.");
    }

    /// <summary>A chance from 0 to 1 (written as a value; of a range, its low end).</summary>
    /// <exception cref="ArgumentException">Not a chance; the message names <paramref name="what"/>.</exception>
    public static float Chance(string what, string value) => Range(what, value, 0, 1).Min;

    /// <summary>A finite number from <paramref name="least"/> to <paramref name="most"/>.</summary>
    /// <exception cref="ArgumentException">Not such a number; the message names <paramref name="what"/>.</exception>
    public static float Number(string what, string value, float least, float most)
    {
        ArgumentNullException.ThrowIfNull(value);
        return float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out float v) && float.IsFinite(v) && v >= least && v <= most
            ? v
            : throw new ArgumentException($"{what} is a number from {least.ToString(CultureInfo.InvariantCulture)} to {most.ToString(CultureInfo.InvariantCulture)}, not '{value}'.");
    }

    /// <summary>A whole number from <paramref name="least"/> to <paramref name="most"/>.</summary>
    /// <exception cref="ArgumentException">Not such a number; the message names <paramref name="what"/>.</exception>
    public static int Integer(string what, string value, int least, int most)
    {
        ArgumentNullException.ThrowIfNull(value);
        return int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) && v >= least && v <= most
            ? v
            : throw new ArgumentException($"{what} is a whole number from {least.ToString(CultureInfo.InvariantCulture)} to {most.ToString(CultureInfo.InvariantCulture)}, not '{value}'.");
    }

    /// <summary>On or off: <c>on</c>, <c>true</c>, <c>yes</c>, <c>1</c>, or <c>off</c>, <c>false</c>, <c>no</c>, <c>0</c> (ignoring case).</summary>
    /// <exception cref="ArgumentException">Neither; the message names <paramref name="what"/>.</exception>
    public static bool Switch(string what, string value)
    {
        ArgumentNullException.ThrowIfNull(value);
        string v = value.Trim();
        return v.Equals("on", StringComparison.OrdinalIgnoreCase) || v.Equals("true", StringComparison.OrdinalIgnoreCase) || v.Equals("yes", StringComparison.OrdinalIgnoreCase) || v == "1"
            || (v.Equals("off", StringComparison.OrdinalIgnoreCase) || v.Equals("false", StringComparison.OrdinalIgnoreCase) || v.Equals("no", StringComparison.OrdinalIgnoreCase) || v == "0"
                ? false
                : throw new ArgumentException($"{what} is on or off, not '{value}'."));
    }

    /// <summary>One of <paramref name="choices"/> (ignoring case), as the choice is written there.</summary>
    /// <exception cref="ArgumentException">None of them; the message names <paramref name="what"/> and the choices.</exception>
    public static string Choice(string what, string value, params ReadOnlySpan<string> choices)
    {
        ArgumentNullException.ThrowIfNull(value);
        foreach (string choice in choices)
        {
            if (choice.Equals(value.Trim(), StringComparison.OrdinalIgnoreCase))
            {
                return choice;
            }
        }

        throw new ArgumentException($"{what} is one of {string.Join(", ", choices.ToArray())}, not '{value}'.");
    }

    /// <summary>A list: items between ':', each trimmed and not empty, <c>ITEM*N</c> standing for N of the same (N from 1).</summary>
    /// <exception cref="ArgumentException">An empty item or a bad repeat; the message names <paramref name="what"/>.</exception>
    public static IReadOnlyList<string> List(string what, string value)
    {
        ArgumentNullException.ThrowIfNull(value);
        var items = new List<string>();
        foreach (string raw in value.Split(':'))
        {
            string item = raw.Trim();
            int times = 1;
            int star = item.LastIndexOf('*');
            if (star >= 0)
            {
                if (!int.TryParse(item.AsSpan(star + 1), NumberStyles.Integer, CultureInfo.InvariantCulture, out times) || times < 1)
                {
                    throw new ArgumentException($"{what} is a list of items between ':' (ITEM*N for N of the same, N from 1), not '{value}'.");
                }

                item = item[..star].Trim();
            }

            if (item.Length == 0 || items.Count + (long)times > MostItems)
            {
                throw new ArgumentException($"{what} is a list of items between ':' (ITEM*N for N of the same), not '{value}'.");
            }

            for (int i = 0; i < times; i++)
            {
                items.Add(item);
            }
        }

        return items;
    }

    /// <summary>A list (<see cref="List"/>) of numbers, each from <paramref name="least"/> to <paramref name="most"/>.</summary>
    /// <exception cref="ArgumentException">A bad list or number; the message names <paramref name="what"/>.</exception>
    public static float[] Numbers(string what, string value, float least, float most)
    {
        var items = List(what, value);
        var numbers = new float[items.Count];
        for (int i = 0; i < numbers.Length; i++)
        {
            numbers[i] = Number(what, items[i], least, most);
        }

        return numbers;
    }

    /// <summary>
    /// Whole numbers from <paramref name="least"/> to <paramref name="most"/> written as a list (<see cref="List"/>) whose
    /// items are numbers or ranges <c>A..B</c> (every number from A to B): <c>3..18</c>, <c>1:3:5..7</c>.
    /// </summary>
    /// <exception cref="ArgumentException">A bad list, number or range; the message names <paramref name="what"/>.</exception>
    public static IReadOnlySet<int> Indexes(string what, string value, int least, int most)
    {
        var set = new SortedSet<int>();
        foreach (string item in List(what, value))
        {
            int dots = item.IndexOf("..", StringComparison.Ordinal);
            int from = Integer(what, dots < 0 ? item : item[..dots].Trim(), least, most), to = dots < 0 ? from : Integer(what, item[(dots + 2)..].Trim(), least, most);
            if (to < from)
            {
                throw new ArgumentException($"{what} has a range {item} whose start is above its end.");
            }

            for (int i = from; i <= to; i++)
            {
                set.Add(i);
            }
        }

        return set;
    }
}
