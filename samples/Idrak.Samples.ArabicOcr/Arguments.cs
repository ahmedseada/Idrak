// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Samples.ArabicOcr;

/// <summary>A usage error: the message is printed and the app exits with 2.</summary>
internal sealed class UsageException(string message) : Exception(message);

/// <summary>
/// A command's arguments: the words that are not options, options that take a value (repeatable) and flags. Each command
/// declares what it takes; anything else is a usage error naming what it does take.
/// </summary>
internal sealed class Arguments
{
    private static readonly Dictionary<string, string> ShortForms = new(StringComparer.Ordinal)
    {
        ["-o"] = "--out", ["-j"] = "--json", ["-v"] = "--verbose", ["-d"] = "--device", ["-h"] = "--help",
    };

    private readonly Dictionary<string, List<string>> _values = new(StringComparer.Ordinal);
    private readonly HashSet<string> _flags = new(StringComparer.Ordinal);

    private Arguments(List<string> words) => Words = words;

    /// <summary>The words that are not options, in order.</summary>
    public IReadOnlyList<string> Words { get; }

    /// <summary>Parses <paramref name="args"/>: <paramref name="valueOptions"/> take the next word (or <c>--name=value</c>), <paramref name="flags"/> none.</summary>
    public static Arguments Parse(IReadOnlyList<string> args, IReadOnlyCollection<string> valueOptions, IReadOnlyCollection<string> flags)
    {
        var words = new List<string>();
        var parsed = new Arguments(words);
        for (int i = 0; i < args.Count; i++)
        {
            string arg = args[i];
            if (arg == "--")
            {
                words.AddRange(args.Skip(i + 1));
                break;
            }

            if (arg.Length < 2 || arg[0] != '-' || char.IsDigit(arg[1]))
            {
                words.Add(arg);
                continue;
            }

            string name = arg;
            string? inline = null;
            int equals = arg.IndexOf('=', StringComparison.Ordinal);
            if (arg.StartsWith("--", StringComparison.Ordinal) && equals > 0)
            {
                (name, inline) = (arg[..equals], arg[(equals + 1)..]);
            }

            name = ShortForms.GetValueOrDefault(name, name);
            if (valueOptions.Contains(name))
            {
                string value = inline ?? (i + 1 < args.Count ? args[++i] : throw new UsageException($"{name} needs a value."));
                if (!parsed._values.TryGetValue(name, out var list))
                {
                    parsed._values[name] = list = [];
                }

                list.Add(value);
            }
            else if (flags.Contains(name) && inline is null)
            {
                parsed._flags.Add(name);
            }
            else
            {
                var known = valueOptions.Concat(flags).Order(StringComparer.Ordinal);
                throw new UsageException($"Unknown option '{arg}'. This command takes: {string.Join(", ", known)}.");
            }
        }

        return parsed;
    }

    /// <summary>The last value of <paramref name="name"/>, or null.</summary>
    public string? Option(string name) => _values.TryGetValue(name, out var list) ? list[^1] : null;

    /// <summary>Every value of <paramref name="name"/>, in order.</summary>
    public IReadOnlyList<string> Options(string name) => _values.TryGetValue(name, out var list) ? list : [];

    /// <summary>Whether the flag <paramref name="name"/> was given.</summary>
    public bool Flag(string name) => _flags.Contains(name);

    /// <summary>The option <paramref name="name"/> as a whole number in [<paramref name="min"/>, <paramref name="max"/>], or <paramref name="otherwise"/>.</summary>
    public int Integer(string name, int otherwise, int min = int.MinValue, int max = int.MaxValue)
    {
        if (Option(name) is not { } text)
        {
            return otherwise;
        }

        return int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) && value >= min && value <= max
            ? value
            : throw new UsageException($"{name} needs a whole number{(min > int.MinValue ? $" from {min}" : "")}{(max < int.MaxValue ? $" to {max}" : "")}, not '{text}'.");
    }

    /// <summary>The option <paramref name="name"/> as a number, or <paramref name="otherwise"/>.</summary>
    public double Number(string name, double otherwise, double min = double.MinValue)
    {
        if (Option(name) is not { } text)
        {
            return otherwise;
        }

        return double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double value) && double.IsFinite(value) && value >= min
            ? value
            : throw new UsageException($"{name} needs a number{(min > double.MinValue ? $" of at least {min.ToString(CultureInfo.InvariantCulture)}" : "")}, not '{text}'.");
    }

    /// <summary>The option <paramref name="name"/>, or a usage error saying what it is for.</summary>
    public string Required(string name, string what) => Option(name) ?? throw new UsageException($"{name} is required: {what}.");
}
