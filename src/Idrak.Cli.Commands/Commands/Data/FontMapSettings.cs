// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands.Data;

/// <summary>
/// The legacy font settings of the commands that read fonts (<c>data synth</c>, <c>data fonts</c>): <c>--font-map FILE</c>
/// (again for more) registers a map tried before the library's, and <c>--font-maps KEY=VALUE,...</c> sets
/// <see cref="FontMaps"/> and its <see cref="FontMapDrawing"/> for the command; both are put back when it ends.
/// </summary>
internal sealed class FontMapSettings : IDisposable
{
    /// <summary>The options' help lines.</summary>
    public const string Help = """
              --font-map FILE     a font map (JSON), again for more: tried before the library's maps
              --font-maps SET     legacy font settings, KEY=VALUE,...: builtin (on: the library's maps), sidecars (on:
                                  FONT.map.json beside FONT.ttf), digits, marks (on: drawn into mapped fonts lacking them), letters (on: composed),
                                  blank-marks (on: a mark without ink is drawn), slant (0.15), wobble (0.012), seed (7),
                                  digit-pen, digit-height, digit-width, mark-size, mark-pen (font units; measured unless set),
                                  mark-share (0.3), mark-pen-share (0.6), pairs, style2, words (the chances a page
                                  joins letter pairs, writes their second style, writes whole-word ligatures; 0.5 each),
                                  short-marks (auto: lower marks on bases under the height they clear; a share of the
                                  tall letters' height; off)
        """;

    private static readonly string[] Keys =
        ["builtin", "sidecars", "digits", "marks", "letters", "blank-marks", "slant", "wobble", "seed", "digit-pen", "digit-height", "digit-width", "mark-size", "mark-pen", "mark-share", "mark-pen-share",
         "pairs", "style2", "words", "short-marks"];

    private readonly bool _builtin = FontMaps.Builtin, _sidecars = FontMaps.Sidecars;
    private readonly FontMapDrawing _drawing = FontMaps.Drawing;
    private readonly List<string> _added = [];

    private FontMapSettings()
    {
    }

    /// <summary>The value options this adds to a command.</summary>
    public static IReadOnlyCollection<string> Options => ["--font-map", "--font-maps"];

    /// <summary>Applies the command's font map options; dispose the result to put the settings back.</summary>
    /// <exception cref="UsageException">A map file that is not valid, or a setting that is not known or not valid.</exception>
    public static FontMapSettings Apply(CommandContext context)
    {
        var settings = new FontMapSettings();
        try
        {
            foreach (string file in context.Options("--font-map"))
            {
                var map = FontMap.Read(file);
                FontMaps.Register(map);
                settings._added.Add(map.Name);
            }

            if (context.Option("--font-maps") is { } text)
            {
                settings.Set(text);
            }
        }
        catch (Exception e) when (e is InvalidDataException or IOException or ArgumentException or FormatException)
        {
            settings.Dispose();
            throw new UsageException(e.Message);
        }

        return settings;
    }

    private void Set(string text)
    {
        var drawing = FontMaps.Drawing;
        foreach (string part in text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            int equals = part.IndexOf('=', StringComparison.Ordinal);
            string key = equals < 0 ? part : part[..equals].Trim(), value = equals < 0 ? "" : part[(equals + 1)..].Trim();
            if (Array.IndexOf(Keys, key) < 0)
            {
                throw new ArgumentException($"--font-maps has no setting '{key}' (valid: {string.Join(", ", Keys)}).");
            }

            bool Switch() => value switch
            {
                "on" or "1" or "true" or "yes" => true,
                "off" or "0" or "false" or "no" => false,
                _ => throw new ArgumentException($"--font-maps {key}={value}: on or off."),
            };
            double Number() => double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out double v) && double.IsFinite(v) && v >= 0
                ? v : throw new ArgumentException($"--font-maps {key}={value}: a number of at least 0.");
            switch (key)
            {
                case "builtin": FontMaps.Builtin = Switch(); break;
                case "sidecars": FontMaps.Sidecars = Switch(); break;
                case "digits": drawing = drawing with { Digits = Switch() }; break;
                case "marks": drawing = drawing with { Marks = Switch() }; break;
                case "letters": drawing = drawing with { Letters = Switch() }; break;
                case "blank-marks": drawing = drawing with { ReplaceBlankMarks = Switch() }; break;
                case "slant": drawing = drawing with { Slant = double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out double s) && double.IsFinite(s) ? s : throw new ArgumentException($"--font-maps slant={value}: a number.") }; break;
                case "wobble": drawing = drawing with { Wobble = Number() }; break;
                case "seed": drawing = drawing with { Seed = int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int seed) ? seed : throw new ArgumentException($"--font-maps seed={value}: a whole number.") }; break;
                case "digit-pen": drawing = drawing with { DigitPen = Number() }; break;
                case "digit-height": drawing = drawing with { DigitHeight = Number() }; break;
                case "digit-width": drawing = drawing with { DigitWidth = Number() }; break;
                case "mark-size": drawing = drawing with { MarkSize = Number() }; break;
                case "mark-pen": drawing = drawing with { MarkPen = Number() }; break;
                case "mark-share": drawing = drawing with { MarkShare = Number() }; break;
                case "pairs": drawing = drawing with { Pairs = Number() is var chance && chance <= 1 ? chance : throw new ArgumentException($"--font-maps pairs={value}: a chance from 0 to 1.") }; break;
                case "style2": drawing = drawing with { SecondStyle = Number() is var second && second <= 1 ? second : throw new ArgumentException($"--font-maps style2={value}: a chance from 0 to 1.") }; break;
                case "words": drawing = drawing with { Words = Number() is var words && words <= 1 ? words : throw new ArgumentException($"--font-maps words={value}: a chance from 0 to 1.") }; break;
                case "short-marks": drawing = drawing with { ShortMarks = value is "auto" or "on" ? null : value is "off" or "false" or "no" ? 0
                    : Number() is var share && share <= 1 ? share : throw new ArgumentException($"--font-maps short-marks={value}: auto, off or a share from 0 to 1.") }; break;
                default: drawing = drawing with { MarkPenShare = Number() }; break;
            }
        }

        FontMaps.Drawing = drawing;
    }

    /// <summary>Puts the settings back and removes the maps the command added.</summary>
    public void Dispose()
    {
        foreach (string name in _added)
        {
            FontMaps.Unregister(name);
        }

        _added.Clear();
        FontMaps.Builtin = _builtin;
        FontMaps.Sidecars = _sidecars;
        FontMaps.Drawing = _drawing;
    }
}
