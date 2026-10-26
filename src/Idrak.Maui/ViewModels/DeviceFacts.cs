// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.RegularExpressions;

namespace Idrak.Maui.ViewModels;

/// <summary>Memory free and the memory it is out of, in bytes, as a tile or a card shows it ("4.2 GB free of 11 GB").</summary>
/// <param name="Free">What is free now.</param>
/// <param name="Total">What the process may use.</param>
public readonly record struct MemoryFigure(long Free, long Total)
{
    /// <summary>The share in use, 0 to 1.</summary>
    public double Used => Total > 0 ? Math.Clamp((double)(Total - Free) / Total, 0, 1) : 0;

    /// <summary>"4.2 GB free of 11 GB".</summary>
    public string Text => $"{ProgressText.Bytes(Free)} free of {ProgressText.Bytes(Total)}";
}

/// <summary>
/// The device facts the library lists (<c>DeviceInfo.Facts</c>), read for the app's cards and tiles: the CPU's memory from
/// the system's figure (not the runtime's, which under Mono is its heap budget), its core groups and vector paths in a
/// line, and a GPU's kernels the driver refused (each run on a fallback). Only what the facts say; nothing is measured here.
/// </summary>
public static class DeviceFacts
{
    private static readonly Regex Size = new(@"(\d+(?:\.\d+)?)\s*(KiB|MiB|GiB|TiB)", RegexOptions.CultureInvariant);
    private static readonly Regex Frequency = new(@"up to (\d+(?:\.\d+)?) GHz", RegexOptions.CultureInvariant);
    private static readonly Regex Aside = new(@"\(([^():]*):[^()]*\)", RegexOptions.CultureInvariant);

    /// <summary>The fact named <paramref name="name"/> (exactly), or null.</summary>
    public static string? Fact(IEnumerable<KeyValuePair<string, string>>? facts, string name)
    {
        foreach (var (key, value) in facts ?? [])
        {
            if (string.Equals(key, name, StringComparison.Ordinal))
            {
                return value;
            }
        }

        return null;
    }

    /// <summary>
    /// The CPU's memory from the system's figure: "system memory" (its total and what is available, from /proc/meminfo)
    /// held to the "memory limit" where one is set (the limit, less what is in use under it); else the "memory" fact's
    /// figures (what the process may use and what is free of it, where the system reports none); null when neither reads.
    /// </summary>
    public static MemoryFigure? SystemMemory(IEnumerable<KeyValuePair<string, string>>? facts)
    {
        var list = facts?.ToList();
        var system = Sizes(Fact(list, "system memory"));
        if (system.Count > 0)
        {
            long total = system[0];
            long free = system.Count > 1 ? system[1] : total;
            var limit = Sizes(Fact(list, "memory limit"));
            if (limit.Count > 0 && limit[0] < total)
            {
                total = limit[0];
                free = Math.Min(free, limit.Count > 1 ? Math.Max(0, limit[0] - limit[1]) : limit[0]);
            }

            return new MemoryFigure(Math.Clamp(free, 0, total), total);
        }

        var process = Sizes(Fact(list, "memory"));
        return process.Count > 1 ? new MemoryFigure(Math.Clamp(process[1], 0, process[0]), process[0]) : null;
    }

    /// <summary>
    /// The core groups in a line: "8 cores: 4 × part 0x41:0xd05 up to 2.00 GHz, 3 × … up to 2.75 GHz" gives
    /// "4 × 2.0 GHz · 3 × 2.75 GHz" (a group without a frequency by its part); "" for none.
    /// </summary>
    public static string CoreGroups(string? groups)
    {
        var parts = GroupParts(groups);
        return string.Join(" · ", parts.Select(part =>
        {
            int times = part.IndexOf('×', StringComparison.Ordinal);
            if (times < 0)
            {
                return part;
            }

            string count = part[..times].Trim();
            var frequency = Frequency.Match(part);
            if (frequency.Success && double.TryParse(frequency.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out double ghz))
            {
                return string.Create(CultureInfo.InvariantCulture, $"{count} × {ghz:0.0#} GHz");
            }

            string what = part[(times + 1)..].Trim();
            return $"{count} × {(what.StartsWith("part ", StringComparison.Ordinal) ? what[5..] : what)}";
        }));
    }

    /// <summary>How many core groups the fact lists (0 for none).</summary>
    public static int CoreGroupCount(string? groups) => GroupParts(groups).Count;

    /// <summary>
    /// The vector paths in a line: the setting and the tiled products' kernel, an aside's explanation left out
    /// ("automatic · 8×8 in 128-bit vectors (portable vectors)"); the fact as it is when it does not read so; "" for none.
    /// </summary>
    public static string VectorPaths(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return "";
        }

        const string Tiled = "tiled products: ";
        int at = text.IndexOf(Tiled, StringComparison.Ordinal);
        int cut = text.IndexOfAny([' ', ';', '(']);
        if (at < 0 || cut <= 0)
        {
            return text;
        }

        var pieces = SplitOutside(text[(at + Tiled.Length)..], "; ");
        return $"{text[..cut]} · {Aside.Replace(pieces[0], "($1)")}";
    }

    /// <summary>
    /// The kernels a GPU's driver refused, one entry each ("int8_gemv at width 64 (…); instead: the host"), from its
    /// "refused kernels" fact; empty for "none" or no fact.
    /// </summary>
    public static IReadOnlyList<string> RefusedKernels(string? text)
    {
        if (string.IsNullOrWhiteSpace(text) || text.Trim().Equals("none", StringComparison.OrdinalIgnoreCase))
        {
            return [];
        }

        // Entries are joined by "; " and each has its own "; instead: …" after its attempts (which may hold "; " inside parentheses).
        var kernels = new List<string>();
        foreach (string piece in SplitOutside(text, "; "))
        {
            if (piece.StartsWith("instead:", StringComparison.Ordinal) && kernels.Count > 0)
            {
                kernels[^1] += "; " + piece;
            }
            else if (piece.Length > 0)
            {
                kernels.Add(piece);
            }
        }

        return kernels;
    }

    // Every size in a text ("11.0 GiB, 4.2 GiB available") in bytes, in order.
    private static List<long> Sizes(string? text)
    {
        var sizes = new List<long>();
        foreach (Match m in Size.Matches(text ?? ""))
        {
            if (double.TryParse(m.Groups[1].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out double value))
            {
                int shift = m.Groups[2].Value switch { "KiB" => 10, "MiB" => 20, "GiB" => 30, _ => 40 };
                sizes.Add((long)Math.Round(value * (1L << shift)));
            }
        }

        return sizes;
    }

    // "8 cores: a, b" → [a, b] (the groups after the count).
    private static List<string> GroupParts(string? groups)
    {
        if (string.IsNullOrWhiteSpace(groups))
        {
            return [];
        }

        int colon = groups.IndexOf(": ", StringComparison.Ordinal);
        string list = colon >= 0 ? groups[(colon + 2)..] : groups;
        return [.. SplitOutside(list, ", ").Select(p => p.Trim()).Where(p => p.Length > 0)];
    }

    // Splits at each separator outside parentheses.
    private static List<string> SplitOutside(string text, string separator)
    {
        var parts = new List<string>();
        int depth = 0, start = 0;
        for (int i = 0; i < text.Length; i++)
        {
            char c = text[i];
            if (c == '(')
            {
                depth++;
            }
            else if (c == ')')
            {
                depth = Math.Max(0, depth - 1);
            }
            else if (depth == 0 && string.CompareOrdinal(text, i, separator, 0, separator.Length) == 0)
            {
                parts.Add(text[start..i].Trim());
                start = i + separator.Length;
                i = start - 1;
            }
        }

        parts.Add(text[start..].Trim());
        return parts;
    }
}
