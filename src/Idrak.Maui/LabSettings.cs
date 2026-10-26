// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Audio.Abstractions;
using Idrak.Maui.Abstractions;

namespace Idrak.Maui;

/// <summary>How a lab's setting is edited.</summary>
public enum LabSettingKind
{
    /// <summary>Free text.</summary>
    Text,

    /// <summary>A number; empty means the library's own value.</summary>
    Number,

    /// <summary>One of <see cref="ILabSetting.Choices"/>.</summary>
    Choice,
}

/// <summary>One value of a lab's options as the page edits it: its key, title, help, and the library's default shown in an empty field.</summary>
public interface ILabSetting
{
    /// <summary>The key in the settings store, after the lab's prefix ("rate").</summary>
    string Key { get; }

    /// <summary>The title on the page.</summary>
    string Title { get; }

    /// <summary>What it does, with the library's option it maps to.</summary>
    string Summary { get; }

    /// <summary>How it is edited.</summary>
    LabSettingKind Kind { get; }

    /// <summary>What an empty value means (the library's default), shown in an empty field.</summary>
    string Placeholder { get; }

    /// <summary>The values a <see cref="LabSettingKind.Choice"/> takes, the first standing for "" (the default); read when asked (the registries may grow).</summary>
    IReadOnlyList<string> Choices { get; }
}

/// <summary>One value of <typeparamref name="TOptions"/> as a setting (<see cref="ILabSetting"/>), read and written as text.</summary>
/// <typeparam name="TOptions">The lab's options.</typeparam>
public sealed class LabSetting<TOptions> : ILabSetting
{
    private readonly Func<TOptions, string> _read;
    private readonly Action<TOptions, string> _write;
    private readonly Func<IReadOnlyList<string>> _choices;

    internal LabSetting(string key, string title, string summary, LabSettingKind kind, string placeholder, Func<TOptions, string> read, Action<TOptions, string> write,
        Func<IReadOnlyList<string>>? choices = null)
    {
        Key = key;
        Title = title;
        Summary = summary;
        Kind = kind;
        Placeholder = placeholder;
        _read = read;
        _write = write;
        _choices = choices ?? (() => []);
    }

    /// <inheritdoc />
    public string Key { get; }

    /// <inheritdoc />
    public string Title { get; }

    /// <inheritdoc />
    public string Summary { get; }

    /// <inheritdoc />
    public LabSettingKind Kind { get; }

    /// <inheritdoc />
    public string Placeholder { get; }

    /// <inheritdoc />
    public IReadOnlyList<string> Choices => _choices();

    /// <summary>The value in <paramref name="options"/> as text ("" for the default).</summary>
    public string Get(TOptions options) => _read(options ?? throw new ArgumentNullException(nameof(options)));

    /// <summary>Sets the value in <paramref name="options"/> from text ("" or null for the default).</summary>
    /// <exception cref="FormatException">The text is not a value of the setting (the message says what is).</exception>
    public void Set(TOptions options, string? text) => _write(options ?? throw new ArgumentNullException(nameof(options)), text?.Trim() ?? "");
}

/// <summary>
/// The labs' settings kept between runs: each definition's value under the lab's prefix and its key in the app's
/// <see cref="ISettingsStore"/> (MAUI's Preferences); an empty value is removed (the library's own). The value parsers
/// the labs share: a decimal point or comma, whole numbers in a range, on/off/default, a registered name.
/// </summary>
public static class LabSettings
{
    /// <summary>
    /// Reads every definition's value kept under <paramref name="prefix"/> into <paramref name="options"/>; a saved value
    /// that no longer reads is skipped and named in <paramref name="skipped"/>.
    /// </summary>
    public static TOptions Load<TOptions>(ISettingsStore store, string prefix, IReadOnlyList<LabSetting<TOptions>> definitions, TOptions options, out IReadOnlyList<string> skipped)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(definitions);
        var refused = new List<string>();
        foreach (var setting in definitions)
        {
            if (store.Get(prefix + setting.Key) is not { } text)
            {
                continue;
            }

            try
            {
                setting.Set(options, text);
            }
            catch (FormatException e)
            {
                refused.Add($"{setting.Title}: {e.Message}");
            }
        }

        skipped = refused;
        return options;
    }

    /// <summary>Keeps every definition's value of <paramref name="options"/> under <paramref name="prefix"/> (an empty one removed).</summary>
    public static void Save<TOptions>(ISettingsStore store, string prefix, IReadOnlyList<LabSetting<TOptions>> definitions, TOptions options)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(definitions);
        foreach (var setting in definitions)
        {
            store.Set(prefix + setting.Key, Empty(setting.Get(options)));
        }
    }

    internal static string? Empty(string? text) => string.IsNullOrWhiteSpace(text) ? null : text.Trim();

    internal static string Show(double? value) => value?.ToString("R", CultureInfo.InvariantCulture) ?? "";

    internal static string Show(int? value) => value?.ToString(CultureInfo.InvariantCulture) ?? "";

    internal static string Show(bool? value) => value switch { true => "on", false => "off", null => "" };

    internal static bool IsDefault(string text) => text.Length == 0 || text.Equals("default", StringComparison.OrdinalIgnoreCase);

    // A decimal number, a comma read as the point (a phone's keyboard in many languages gives one).
    internal static double? Real(string text, double? above = null, double? atLeast = null, double? below = null, double? atMost = null)
    {
        if (text.Length == 0)
        {
            return null;
        }

        if (!double.TryParse(text.Replace(',', '.'), NumberStyles.Float, CultureInfo.InvariantCulture, out double value) || !double.IsFinite(value)
            || value <= above || value < atLeast || value >= below || value > atMost)
        {
            var range = new List<string>();
            if (above is { } a)
            {
                range.Add($"above {Show(a)}");
            }

            if (atLeast is { } l)
            {
                range.Add($"of {Show(l)} or more");
            }

            if (below is { } b)
            {
                range.Add($"under {Show(b)}");
            }

            if (atMost is { } m)
            {
                range.Add($"of {Show(m)} or less");
            }

            throw new FormatException($"'{text}' is not a number{(range.Count > 0 ? " " + string.Join(" and ", range) : "")}.");
        }

        return value;
    }

    internal static int? Whole(string text, int minimum, int maximum = int.MaxValue)
    {
        if (text.Length == 0)
        {
            return null;
        }

        return int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) && value >= minimum && value <= maximum ? value
            : throw new FormatException(maximum == int.MaxValue ? $"'{text}' is not a whole number of {minimum} or more." : $"'{text}' is not a whole number from {minimum} to {maximum}.");
    }

    internal static bool? Switch(string text) => text.ToLowerInvariant() switch
    {
        "" or "default" => null,
        "on" or "true" or "1" => true,
        "off" or "false" or "0" => false,
        _ => throw new FormatException($"'{text}' is not on, off or default."),
    };

    // A name of a registry (`names`), or null for the default; `what` names the registry in the refusal.
    internal static string? Registered(string text, IReadOnlyCollection<string> names, string what, StringComparer? comparer = null) =>
        IsDefault(text) ? null
        : names.Contains(text, comparer ?? StringComparer.OrdinalIgnoreCase) ? text
        : throw new FormatException($"'{text}' is not {what} (registered: {(names.Count == 0 ? "none" : string.Join(", ", names))}).");

    // One of `choices` (any case), or null for the default.
    internal static string? OneOf(string text, IReadOnlyList<string> choices) =>
        IsDefault(text) ? null
        : choices.FirstOrDefault(c => c.Equals(text, StringComparison.OrdinalIgnoreCase))
          ?? throw new FormatException($"'{text}' is not one of {string.Join(", ", choices)}.");

    // An AudioPerformance name, or null for the default.
    internal static AudioPerformance? Performance(string text) =>
        IsDefault(text) ? null
        : Enum.TryParse<AudioPerformance>(text, ignoreCase: true, out var performance) && Enum.IsDefined(performance) && !int.TryParse(text, out _) ? performance
        : throw new FormatException($"'{text}' is not {string.Join(", ", Enum.GetNames<AudioPerformance>())}.");

    /// <summary>The sample formats a saved sound takes ("16", "24", "32", "8", "float"), the first the default.</summary>
    public static IReadOnlyList<string> SampleFormats { get; } = ["16", "24", "32", "8", "float"];

    // The encoder options of a sample format ("float" is 32-bit float; null: 16-bit).
    internal static AudioEncodeOptions Encode(string? bits) => bits switch
    {
        null or "" or "16" => new AudioEncodeOptions(),
        "float" => new AudioEncodeOptions { Float = true, BitsPerSample = 32 },
        _ => new AudioEncodeOptions { BitsPerSample = int.Parse(bits, CultureInfo.InvariantCulture) },
    };
}
