// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Reinforcement.Plugins.Classic;

// A classic kind's options by key: every key checked against the kind's own (refused naming it and the keys), values
// parsed with the invariant culture, a bad value refused naming its key and what it takes.
internal readonly struct ClassicOptions
{
    private readonly IReadOnlyDictionary<string, string>? _options;
    private readonly string _kind;

    public ClassicOptions(string kind, IReadOnlyCollection<string> keys, IReadOnlyDictionary<string, string>? options)
    {
        _kind = kind;
        _options = options;
        foreach (string key in options?.Keys ?? [])
        {
            if (!keys.Contains(key))
            {
                throw new ArgumentException($"The {kind} environment does not take the option '{key}'; it takes: {string.Join(", ", keys)}.", nameof(options));
            }
        }
    }

    public int Int(string key, int fallback, int min = int.MinValue, int max = int.MaxValue)
    {
        if (Text(key) is not { } text)
        {
            return fallback;
        }

        return int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) && value >= min && value <= max
            ? value
            : throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"The {_kind} environment's option '{key}' is '{text}': a whole number from {min} to {max}."));
    }

    public double Number(string key, double fallback, double min = double.NegativeInfinity, double max = double.PositiveInfinity)
    {
        if (Text(key) is not { } text)
        {
            return fallback;
        }

        return Parse(text, out double value) && value >= min && value <= max
            ? value
            : throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"The {_kind} environment's option '{key}' is '{text}': a finite number from {min} to {max}."));
    }

    public bool Bool(string key, bool fallback) => Text(key)?.ToLowerInvariant() switch
    {
        null => fallback,
        "true" or "1" or "yes" or "on" => true,
        "false" or "0" or "no" or "off" => false,
        var other => throw new ArgumentException($"The {_kind} environment's option '{key}' is '{other}': true or false."),
    };

    // A list of numbers separated by commas; null when the option is not given.
    public double[]? Numbers(string key, double min = double.NegativeInfinity)
    {
        if (Text(key) is not { } text)
        {
            return null;
        }

        var parts = text.Split(',', StringSplitOptions.TrimEntries);
        var values = new double[parts.Length];
        for (int i = 0; i < parts.Length; i++)
        {
            if (!Parse(parts[i], out values[i]) || values[i] < min)
            {
                throw new ArgumentException(string.Create(CultureInfo.InvariantCulture,
                    $"The {_kind} environment's option '{key}' is '{text}': finite numbers of at least {min}, separated by commas ('{parts[i]}' is not one)."));
            }
        }

        return values;
    }

    private string? Text(string key) => _options is not null && _options.TryGetValue(key, out string? text) && text.Trim().Length > 0 ? text.Trim() : null;

    private static bool Parse(string text, out double value) =>
        double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out value) && double.IsFinite(value);
}

// What every classic environment keeps of its episode: whether one is under way and its steps, and the checks a step
// makes before it runs.
internal struct EpisodeClock
{
    public bool Running;
    public int Steps;

    public void Start()
    {
        Running = true;
        Steps = 0;
    }

    // Counts a step of a running episode (refusing one when none is), after the token's check.
    public void Tick(string kind, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (!Running)
        {
            throw new InvalidOperationException($"The {kind} environment has no episode under way: reset it first (also after an episode ended).");
        }

        Steps++;
    }

    // Gymnasium's TimeLimit: the step that reaches the limit is truncated (whether or not it also terminated).
    public bool Truncates(int maxSteps) => maxSteps > 0 && Steps >= maxSteps;

    public void End(bool terminated, bool truncated)
    {
        if (terminated || truncated)
        {
            Running = false;
        }
    }
}
