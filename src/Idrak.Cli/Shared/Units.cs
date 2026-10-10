// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Cli.Shared;

/// <summary>
/// Amounts for people, the same in every command (sizes, counts, times): one place instead of a formatter per group.
/// </summary>
internal static class Units
{
    /// <summary>A byte count: "512 B", "3.4 MB", "1.20 GB" (powers of 1024, as the library's downloader prints them).</summary>
    public static string Bytes(long bytes) => Idrak.Data.Downloader.Size(bytes);

    /// <summary>A byte count given as a number of any size (estimates): as <see cref="Bytes(long)"/>.</summary>
    public static string Bytes(double bytes) => Bytes((long)Math.Round(bytes));

    /// <summary>A count of parameters: "9,999", "751.6 M", "8.03 B".</summary>
    public static string Count(long count) => count switch
    {
        < 10_000 => count.ToString("N0", CultureInfo.InvariantCulture),
        < 1_000_000 => string.Create(CultureInfo.InvariantCulture, $"{count / 1e3:0.#} K"),
        < 1_000_000_000 => string.Create(CultureInfo.InvariantCulture, $"{count / 1e6:0.#} M"),
        _ => string.Create(CultureInfo.InvariantCulture, $"{count / 1e9:0.##} B"),
    };

    /// <summary>A large amount in short SI form (FLOPs, parameters in a design): "812", "45k", "1.2M", "3.4G".</summary>
    public static string Short(double value) => value switch
    {
        >= 1e9 => string.Create(CultureInfo.InvariantCulture, $"{value / 1e9:0.#}G"),
        >= 1e6 => string.Create(CultureInfo.InvariantCulture, $"{value / 1e6:0.#}M"),
        >= 1e3 => string.Create(CultureInfo.InvariantCulture, $"{value / 1e3:0.#}k"),
        _ => string.Create(CultureInfo.InvariantCulture, $"{value:0}"),
    };

    /// <summary>A duration: "4s", "3m 05s" or "1h 02m".</summary>
    public static string Duration(TimeSpan time) =>
        time.TotalHours >= 1 ? string.Create(CultureInfo.InvariantCulture, $"{(int)time.TotalHours}h {time.Minutes:D2}m")
        : time.TotalMinutes >= 1 ? string.Create(CultureInfo.InvariantCulture, $"{(int)time.TotalMinutes}m {time.Seconds:D2}s")
        : string.Create(CultureInfo.InvariantCulture, $"{Math.Max(0, (int)Math.Ceiling(time.TotalSeconds))}s");

    /// <summary>A time relative to now: "just now", "5 min ago", "3 days ago", or the date after 60 days.</summary>
    public static string Ago(DateTime utc)
    {
        var age = DateTime.UtcNow - utc;
        return age.TotalMinutes < 1 ? "just now"
            : age.TotalHours < 1 ? string.Create(CultureInfo.InvariantCulture, $"{(int)age.TotalMinutes} min ago")
            : age.TotalDays < 1 ? string.Create(CultureInfo.InvariantCulture, $"{(int)age.TotalHours} h ago")
            : age.TotalDays < 60 ? string.Create(CultureInfo.InvariantCulture, $"{(int)age.TotalDays} days ago")
            : utc.ToLocalTime().ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
    }
}
