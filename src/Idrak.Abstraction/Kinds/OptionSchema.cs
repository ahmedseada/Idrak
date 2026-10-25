// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Abstraction;

/// <summary>
/// A kind whose options are declared once, each with its type, range and default (<see cref="Options"/>): what help pages
/// list, and what parses a run's options from their text, every key checked and every value parsed once.
/// </summary>
public interface IOptionSchema
{
    /// <summary>The options the kind takes.</summary>
    OptionSchema Options { get; }
}

/// <summary>
/// One declared option: its key, what it takes (for help and for the message that refuses a bad value) and its default
/// as text. See <see cref="OptionSpec{T}"/> and the factories of <see cref="OptionSpec"/>.
/// </summary>
public abstract class OptionSpec
{
    private protected OptionSpec(string key, string takes, string? summary)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(key);
        ArgumentException.ThrowIfNullOrWhiteSpace(takes);
        Key = key;
        Takes = takes;
        Summary = summary ?? "";
    }

    /// <summary>The option's key.</summary>
    public string Key { get; }

    /// <summary>What a value must be ("a number from 0 to 1", "true or false"), as the message that refuses another says it.</summary>
    public string Takes { get; }

    /// <summary>What the option does, in a line ("" when not given).</summary>
    public string Summary { get; }

    /// <summary>The default as text, for help; null when the option has none (it is then required, or optional with no value).</summary>
    public abstract string? DefaultText { get; }

    /// <summary>Whether the value is parsed with its leading and trailing blanks trimmed (true unless set: text options keep them).</summary>
    public bool Trimmed { get; init; } = true;

    // Parses the trimmed text into the box the values keep; false when it is not a value of the option.
    internal abstract bool TryParse(ReadOnlySpan<char> text, out object? value);

    internal abstract object? Default { get; }

    /// <summary>A number (a finite <see cref="float"/>) from <paramref name="min"/> to <paramref name="max"/>, <paramref name="fallback"/> when not given.</summary>
    /// <param name="key">The option's key.</param>
    /// <param name="fallback">Its default.</param>
    /// <param name="min">The smallest value it takes.</param>
    /// <param name="max">The largest value it takes.</param>
    /// <param name="takes">What it takes, as messages say it; null: "a number from <paramref name="min"/> to <paramref name="max"/>".</param>
    /// <param name="summary">What it does.</param>
    public static OptionSpec<float> Number(string key, float fallback, float min = float.NegativeInfinity, float max = float.PositiveInfinity, string? takes = null,
        string? summary = null) =>
        new(key, fallback, takes ?? string.Create(CultureInfo.InvariantCulture, $"a number from {min} to {max}"), summary,
            (ReadOnlySpan<char> text, out float value) =>
                float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out value) && float.IsFinite(value) && value >= min && value <= max,
            fallback.ToString(CultureInfo.InvariantCulture));

    /// <summary>A number (a finite <see cref="double"/>) from <paramref name="min"/> to <paramref name="max"/>, <paramref name="fallback"/> when not given.</summary>
    public static OptionSpec<double> Real(string key, double fallback, double min = double.NegativeInfinity, double max = double.PositiveInfinity, string? takes = null,
        string? summary = null) =>
        new(key, fallback, takes ?? string.Create(CultureInfo.InvariantCulture, $"a finite number from {min} to {max}"), summary,
            (ReadOnlySpan<char> text, out double value) =>
                double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out value) && double.IsFinite(value) && value >= min && value <= max,
            fallback.ToString(CultureInfo.InvariantCulture));

    /// <summary>A whole number from <paramref name="min"/> to <paramref name="max"/>, <paramref name="fallback"/> when not given.</summary>
    public static OptionSpec<int> Whole(string key, int fallback, int min = int.MinValue, int max = int.MaxValue, string? takes = null, string? summary = null) =>
        new(key, fallback, takes ?? string.Create(CultureInfo.InvariantCulture, $"a whole number from {min} to {max}"), summary,
            (ReadOnlySpan<char> text, out int value) => int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out value) && value >= min && value <= max,
            fallback.ToString(CultureInfo.InvariantCulture));

    /// <summary>A switch: true, 1, yes or on; false, 0, no or off (any case); <paramref name="fallback"/> when not given.</summary>
    public static OptionSpec<bool> Flag(string key, bool fallback = false, string? takes = null, string? summary = null) =>
        new(key, fallback, takes ?? "true or false", summary, ParseFlag, fallback ? "true" : "false");

    /// <summary>One of <paramref name="choices"/> (any case; the value is the choice as declared), <paramref name="fallback"/> when not given.</summary>
    public static OptionSpec<string> Choice(string key, string fallback, params ReadOnlySpan<string> choices)
    {
        string[] declared = [.. choices];
        return new(key, fallback, $"one of {string.Join(", ", declared)}", null, (ReadOnlySpan<char> text, out string value) =>
        {
            foreach (string choice in declared)
            {
                if (text.Equals(choice, StringComparison.OrdinalIgnoreCase))
                {
                    value = choice;
                    return true;
                }
            }

            value = "";
            return false;
        }, fallback);
    }

    /// <summary>Text, as given, <paramref name="fallback"/> when not given (null: none, which the caller may require).</summary>
    public static OptionSpec<string?> Text(string key, string? fallback = null, string? takes = null, string? summary = null) =>
        new(key, fallback, takes ?? "text", summary, (ReadOnlySpan<char> text, out string? value) =>
        {
            value = text.ToString();
            return true;
        }, fallback) { Trimmed = false };

    /// <summary>Finite numbers of at least <paramref name="min"/> separated by commas; null when not given.</summary>
    public static OptionSpec<double[]?> Reals(string key, double min = double.NegativeInfinity, string? takes = null, string? summary = null) =>
        new(key, null, takes ?? string.Create(CultureInfo.InvariantCulture, $"finite numbers of at least {min}, separated by commas"), summary,
            (ReadOnlySpan<char> text, out double[]? value) =>
            {
                var values = new List<double>();
                foreach (var range in text.Split(','))
                {
                    if (!double.TryParse(text[range].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out double number) || !double.IsFinite(number) || number < min)
                    {
                        value = null;
                        return false;
                    }

                    values.Add(number);
                }

                value = [.. values];
                return true;
            }, null);

    private static bool ParseFlag(ReadOnlySpan<char> text, out bool value)
    {
        if (text.Equals("true", StringComparison.OrdinalIgnoreCase) || text.SequenceEqual("1") || text.Equals("yes", StringComparison.OrdinalIgnoreCase)
            || text.Equals("on", StringComparison.OrdinalIgnoreCase))
        {
            value = true;
            return true;
        }

        value = false;
        return text.Equals("false", StringComparison.OrdinalIgnoreCase) || text.SequenceEqual("0") || text.Equals("no", StringComparison.OrdinalIgnoreCase)
               || text.Equals("off", StringComparison.OrdinalIgnoreCase);
    }
}

/// <summary>Parses an option's value from its text, read as a span (no string made for a number or a switch).</summary>
/// <typeparam name="T">The value's type.</typeparam>
public delegate bool OptionParser<T>(ReadOnlySpan<char> text, out T value);

/// <summary>A declared option of <typeparamref name="T"/> values: its key, its default and how its text is parsed.</summary>
/// <typeparam name="T">The value's type.</typeparam>
public sealed class OptionSpec<T> : OptionSpec
{
    private readonly OptionParser<T> _parse;
    private readonly string? _defaultText;

    /// <summary>The option <paramref name="key"/> of <typeparamref name="T"/> values parsed by <paramref name="parse"/>.</summary>
    /// <param name="key">Its key.</param>
    /// <param name="fallback">Its value when not given.</param>
    /// <param name="takes">What a value must be, as messages say it.</param>
    /// <param name="summary">What it does.</param>
    /// <param name="parse">Parses a value's trimmed text; false refuses it.</param>
    /// <param name="defaultText">The default as help prints it; null for none.</param>
    public OptionSpec(string key, T fallback, string takes, string? summary, OptionParser<T> parse, string? defaultText) : base(key, takes, summary)
    {
        ArgumentNullException.ThrowIfNull(parse);
        Fallback = fallback;
        _parse = parse;
        _defaultText = defaultText;
    }

    /// <summary>The value when the option is not given.</summary>
    public T Fallback { get; }

    /// <inheritdoc />
    public override string? DefaultText => _defaultText;

    internal override object? Default => Fallback;

    internal override bool TryParse(ReadOnlySpan<char> text, out object? value)
    {
        if (_parse(text, out T parsed))
        {
            value = parsed;
            return true;
        }

        value = null;
        return false;
    }
}

/// <summary>
/// The options a kind takes, declared once (<see cref="IOptionSchema"/>): their keys (<see cref="Keys"/>, what
/// <see cref="IConfigurableKind.Keys"/> returns), their types, ranges and defaults. <see cref="Parse"/> checks a run's
/// options against them and parses each declared value from its text as a span, once: the last options parsed are kept,
/// so a kind asked again with the same dictionary (every batch of a run) parses nothing more.
/// </summary>
public sealed class OptionSchema
{
    private readonly OptionSpec[] _options;
    private readonly bool _blankIsDefault;
    private Parsed? _last;

    /// <summary>The options <paramref name="options"/> of <paramref name="what"/>.</summary>
    /// <param name="what">What takes them, as messages name it ("reward length", "estimator gae", "cartpole environment").</param>
    /// <param name="options">The options, in the order help lists them.</param>
    public OptionSchema(string what, params ReadOnlySpan<OptionSpec> options) : this(what, false, options)
    {
    }

    /// <summary>The options <paramref name="options"/> of <paramref name="what"/>; with <paramref name="blankIsDefault"/>, a value of blanks counts as not given.</summary>
    public OptionSchema(string what, bool blankIsDefault, params ReadOnlySpan<OptionSpec> options)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(what);
        What = what;
        _blankIsDefault = blankIsDefault;
        _options = [.. options];
        string[] keys = new string[_options.Length];
        for (int i = 0; i < keys.Length; i++)
        {
            keys[i] = (_options[i] ?? throw new ArgumentNullException(nameof(options))).Key;
        }

        Keys = new OptionKeys(keys);
        if (Keys.Count != keys.Length)
        {
            throw new ArgumentException($"The options of the {what} declare a key twice ({string.Join(", ", keys)}).", nameof(options));
        }
    }

    /// <summary>What takes the options, as messages name it.</summary>
    public string What { get; }

    /// <summary>The declared options, in order.</summary>
    public ReadOnlySpan<OptionSpec> Options => _options;

    /// <summary>The declared keys, in order.</summary>
    public OptionKeys Keys { get; }

    /// <summary>Each option as "key: takes (default …)", one per line, for help.</summary>
    public string Describe() => string.Join("\n", _options.Select(o =>
        $"{o.Key}: {(o.Summary.Length > 0 ? o.Summary + "; " : "")}{o.Takes}{(o.DefaultText is { } d ? $" (default {d})" : "")}"));

    /// <summary>
    /// The values of <paramref name="options"/>: every key checked against the declared ones (and
    /// <paramref name="alsoTakes"/>, whose keys are kept as given in <see cref="OptionValues.Rest"/>), every declared value
    /// parsed from its text once (trimmed, but for <see cref="OptionSpec.Trimmed"/> false). The same dictionary (and the
    /// same <paramref name="alsoTakes"/>) asked again gives the values parsed before: a run's options are parsed once.
    /// </summary>
    /// <exception cref="ArgumentException">A key not declared (the message names it and the keys), or a value the option does not take (naming what it takes).</exception>
    public OptionValues Parse(IReadOnlyDictionary<string, string>? options, Func<string, bool>? alsoTakes = null)
    {
        var last = _last;
        if (last is not null && ReferenceEquals(last.Of, options) && Equals(last.AlsoTakes, alsoTakes))
        {
            return last.Values;
        }

        var values = new object?[_options.Length];
        var set = new bool[_options.Length];
        for (int i = 0; i < _options.Length; i++)
        {
            values[i] = _options[i].Default;
        }

        Dictionary<string, string>? rest = null;
        foreach (var (key, text) in options ?? Empty)
        {
            int at = IndexOf(key);
            if (at < 0)
            {
                if (alsoTakes?.Invoke(key) == true)
                {
                    (rest ??= new Dictionary<string, string>(StringComparer.Ordinal))[key] = text;
                    continue;
                }

                throw new ArgumentException($"The {What} does not take the option '{key}'; it takes: {Keys}.", nameof(options));
            }

            var given = (text ?? "").AsSpan();
            if (_blankIsDefault && given.IsWhiteSpace())
            {
                continue;
            }

            if (!_options[at].TryParse(_options[at].Trimmed ? given.Trim() : given, out values[at]))
            {
                throw new ArgumentException($"The {What}'s option '{key}' is '{text}': {_options[at].Takes}.", nameof(options));
            }

            set[at] = true;
        }

        var parsed = new OptionValues(this, values, set, rest ?? Empty);
        _last = new Parsed(options, alsoTakes, parsed);
        return parsed;
    }

    internal int IndexOf(OptionSpec option)
    {
        for (int i = 0; i < _options.Length; i++)
        {
            if (ReferenceEquals(_options[i], option))
            {
                return i;
            }
        }

        throw new ArgumentException($"The option '{option.Key}' is not one of the {What}'s ({Keys}).", nameof(option));
    }

    private int IndexOf(string key)
    {
        for (int i = 0; i < _options.Length; i++)
        {
            if (string.Equals(_options[i].Key, key, StringComparison.Ordinal))
            {
                return i;
            }
        }

        return -1;
    }

    private static readonly IReadOnlyDictionary<string, string> Empty = new Dictionary<string, string>();

    private sealed record Parsed(IReadOnlyDictionary<string, string>? Of, Func<string, bool>? AlsoTakes, OptionValues Values);
}

/// <summary>The values of a run's options, parsed by an <see cref="OptionSchema"/>: each declared option's, or its default.</summary>
public sealed class OptionValues
{
    private readonly OptionSchema _schema;
    private readonly object?[] _values;
    private readonly bool[] _given;

    internal OptionValues(OptionSchema schema, object?[] values, bool[] given, IReadOnlyDictionary<string, string> rest)
    {
        _schema = schema;
        _values = values;
        _given = given;
        Rest = rest;
    }

    /// <summary>The value of <paramref name="option"/>: as given, or its default.</summary>
    public T Get<T>(OptionSpec<T> option) => (T)_values[_schema.IndexOf(option)]!;

    /// <summary>Whether <paramref name="option"/> was given (not left to its default).</summary>
    public bool IsGiven(OptionSpec option) => _given[_schema.IndexOf(option)];

    /// <summary>The options the schema does not declare but its caller takes (<see cref="OptionSchema.Parse"/>'s <c>alsoTakes</c>), as given.</summary>
    public IReadOnlyDictionary<string, string> Rest { get; }
}
