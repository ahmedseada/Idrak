// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Frozen;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;

namespace Idrak.VitsSpeech;

/// <summary>
/// Turns text of a script into the Latin letters a checkpoint was trained on: MMS-TTS checkpoints that set
/// <c>is_uroman</c> in tokenizer_config.json read text romanised by uroman, so <see cref="VitsTokenizer"/> romanises
/// such text before tokenizing. Registered in <see cref="TextRomanizers"/>.
/// </summary>
public interface ITextRomanizer
{
    /// <summary>The name it is registered under ("uroman-arabic").</summary>
    string Name { get; }

    /// <summary>Whether it romanises <paramref name="rune"/> (a character of its script).</summary>
    bool Covers(Rune rune);

    /// <summary><paramref name="text"/> with the characters it covers romanised and the others as they are.</summary>
    string Romanize(string text);
}

/// <summary>
/// The romanizers, by name (ordinal). Built in: <see cref="UromanArabic"/> (uroman's romanisation of the Arabic
/// script). <see cref="For"/> picks the one that covers a text's script; register one for another script with
/// <see cref="Register"/> (one under a built-in name shadows it, which stays behind it).
/// </summary>
public static class TextRomanizers
{
    /// <summary>uroman's romanisation of Arabic script (<see cref="UromanArabicRomanizer"/>).</summary>
    public const string UromanArabic = "uroman-arabic";

    private static readonly SlotTable<string, ITextRomanizer> Table = BuiltIn();

    private static SlotTable<string, ITextRomanizer> BuiltIn()
    {
        var table = new SlotTable<string, ITextRomanizer>(nameof(TextRomanizers), comparer: StringComparer.Ordinal, newestFirst: true);
        table.RegisterDefault(UromanArabic, new UromanArabicRomanizer());
        return table;
    }

    /// <summary>Registers <paramref name="romanizer"/> under its <see cref="ITextRomanizer.Name"/> (asked before the earlier ones by <see cref="For"/>).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ITextRomanizer romanizer)
    {
        ArgumentNullException.ThrowIfNull(romanizer);
        ArgumentException.ThrowIfNullOrWhiteSpace(romanizer.Name);
        Table.Register(romanizer.Name, romanizer, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's romanizer <paramref name="name"/> (a built-in name gets the built-in back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, the most recently registered first.</summary>
    public static IReadOnlyList<string> Names => Table.Keys;

    /// <summary>The romanizer <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name: the message names the registered ones.</exception>
    public static ITextRomanizer Get(string name) => Table.Find(name)
        ?? throw new NotSupportedException($"Romanizer '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with TextRomanizers.Register.");

    /// <summary>The built-in romanizer <paramref name="name"/>, whatever an app registered over it; null when there is none.</summary>
    public static ITextRomanizer? Default(string name) => Table.Default(name);

    /// <summary>Who registered the romanizer <paramref name="name"/>: <see cref="Overrides.Library"/> for the built-in or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>
    /// The romanizer for <paramref name="text"/>: the first registered (most recent first) that covers its first
    /// non-Latin letter; null when none does.
    /// </summary>
    public static ITextRomanizer? For(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        foreach (var rune in text.EnumerateRunes())
        {
            if (VitsTokenizer.IsNonLatinLetter(rune))
            {
                foreach (var romanizer in Table.Values)
                {
                    if (romanizer.Covers(rune))
                    {
                        return romanizer;
                    }
                }

                return null;
            }
        }

        return null;
    }
}

/// <summary>
/// uroman's romanisation of the Arabic script: every code point of the Arabic blocks and presentation forms uroman
/// changes, and the few sequences its table romanises together without a language code (alef + fathatan "an", maddah
/// + alef, a word-initial alef + Persian yeh), from <c>tools/audio/make_uroman_table.py</c> (the version in
/// <see cref="Version"/>). Letters become their Latin values (ب b, ع ', ة a, ئ ye), vowel marks their vowels, shadda,
/// sukun, other tanween and tatweel nothing, Arabic-Indic digits 0 to 9; other characters (Latin, spaces, emoji) stay
/// as they are. The longest sequence that applies at a position wins over the single code point. The test checks it
/// against uroman's own output on the cases the tool writes.
/// </summary>
public sealed class UromanArabicRomanizer : ITextRomanizer
{
    private static readonly FrozenDictionary<int, string> Map = Parse(UromanArabicTable.Entries);
    private static readonly (int[] Source, string Target, char Where)[] Rules = ParseRules(UromanArabicTable.Rules);

    /// <summary>The uroman version the table was made from.</summary>
    public static string Version => UromanArabicTable.Version;

    /// <inheritdoc />
    public string Name => TextRomanizers.UromanArabic;

    /// <inheritdoc />
    public bool Covers(Rune rune) => Map.ContainsKey(rune.Value);

    /// <inheritdoc />
    public string Romanize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var runes = new List<Rune>(text.Length);
        foreach (var rune in text.EnumerateRunes())
        {
            runes.Add(rune);
        }

        var builder = new StringBuilder(text.Length + text.Length / 2);
        Span<char> units = stackalloc char[2];
        for (int i = 0; i < runes.Count;)
        {
            if (Sequence(runes, i) is { } rule)
            {
                builder.Append(rule.Target);
                i += rule.Source.Length;
                continue;
            }

            var rune = runes[i++];
            if (Map.TryGetValue(rune.Value, out string? roman))
            {
                builder.Append(roman);
            }
            else
            {
                builder.Append(units[..rune.EncodeToUtf16(units)]);
            }
        }

        return builder.ToString();
    }

    // The longest sequence rule that matches at `at` and whose place in the word holds.
    private static (int[] Source, string Target, char Where)? Sequence(List<Rune> runes, int at)
    {
        (int[] Source, string Target, char Where)? best = null;
        foreach (var rule in Rules)
        {
            int n = rule.Source.Length;
            if (at + n > runes.Count || (best is { } b && b.Source.Length >= n))
            {
                continue;
            }

            bool match = true;
            for (int k = 0; k < n && match; k++)
            {
                match = runes[at + k].Value == rule.Source[k];
            }

            if (match && (rule.Where switch
                {
                    's' => at == 0 || !InWord(runes[at - 1]),
                    'e' => at + n == runes.Count || !InWord(runes[at + n]),
                    _ => true,
                }))
            {
                best = rule;
            }
        }

        return best;
    }

    // A letter or a mark on one: part of a word for the start and end conditions.
    private static bool InWord(Rune rune) => Rune.IsLetter(rune) || Rune.GetUnicodeCategory(rune) is UnicodeCategory.NonSpacingMark or UnicodeCategory.SpacingCombiningMark;

    private static (int[] Source, string Target, char Where)[] ParseRules(string rules)
    {
        var parsed = new List<(int[], string, char)>();
        foreach (var line in rules.Split('\n', StringSplitOptions.RemoveEmptyEntries))
        {
            var fields = line.Split('\t');
            int[] source = [.. fields[0].Split(' ').Select(h => int.Parse(h, NumberStyles.HexNumber, CultureInfo.InvariantCulture))];
            parsed.Add((source, fields[1], fields[2][0]));                    // 'a'ny, 's'tart or 'e'nd of a word
        }

        return [.. parsed];
    }

    private static FrozenDictionary<int, string> Parse(string entries)
    {
        var map = new Dictionary<int, string>();
        foreach (var line in entries.AsSpan().EnumerateLines())
        {
            if (line.IsEmpty)
            {
                continue;
            }

            int tab = line.IndexOf('\t');
            map[int.Parse(line[..tab], NumberStyles.HexNumber, CultureInfo.InvariantCulture)] = line[(tab + 1)..].ToString();
        }

        return map.ToFrozenDictionary();
    }
}
