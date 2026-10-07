// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Idrak.Abstraction.Serving;

/// <summary>Parses keep-alive values: durations such as "30m", "1h30m", "90s", "500ms", a number of seconds, 0 (unload now) or a negative value (keep forever).</summary>
public static partial class KeepAlive
{
    /// <summary>A duration, <see cref="TimeSpan.Zero"/> to unload right after use, or null to keep the model loaded indefinitely.</summary>
    public static TimeSpan? Parse(string text)
    {
        text = text.Trim();
        if (double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double seconds))
        {
            return FromSeconds(seconds);
        }

        bool negative = text.StartsWith('-');
        var parts = Unit().Matches(negative ? text[1..] : text);
        if (parts.Count == 0 || string.Concat(parts.Select(m => m.Value)) != (negative ? text[1..] : text))
        {
            throw new FormatException($"'{text}' is not a duration (examples: 30m, 1h30m, 90s, 500ms, 0, -1).");
        }

        if (negative)
        {
            return null;
        }

        double total = parts.Sum(m => double.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture) * m.Groups[2].Value switch
        {
            "h" => 3600,
            "m" => 60,
            "s" => 1,
            "ms" => 0.001,
            "us" or "µs" => 1e-6,
            _ => 1e-9,
        });
        return TimeSpan.FromSeconds(total);
    }

    /// <summary>Parses a JSON value (string or number); null or undefined give <paramref name="fallback"/>.</summary>
    public static TimeSpan? Parse(JsonElement? value, TimeSpan? fallback) => value?.ValueKind switch
    {
        JsonValueKind.String => Parse(value.Value.GetString()!),
        JsonValueKind.Number => FromSeconds(value.Value.GetDouble()),
        _ => fallback,
    };

    private static TimeSpan? FromSeconds(double seconds) => seconds < 0 ? null : TimeSpan.FromSeconds(seconds);

    [GeneratedRegex(@"(\d+(?:\.\d+)?)(ms|us|µs|ns|h|m|s)")]
    private static partial Regex Unit();
}
