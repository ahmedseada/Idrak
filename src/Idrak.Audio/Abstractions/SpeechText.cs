// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text;

namespace Idrak.Audio.Abstractions;

/// <summary>
/// Turns text of a script into the Latin letters a checkpoint was trained on: MMS-TTS checkpoints that set
/// <c>is_uroman</c> in tokenizer_config.json read text romanised by uroman, so a synthesizer's tokenizer romanises
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
            if (IsNonLatinLetter(rune))
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

    /// <summary>Whether <paramref name="rune"/> is a letter outside the Latin blocks (past Latin Extended-B): one a romanizer may cover.</summary>
    public static bool IsNonLatinLetter(Rune rune) => Rune.IsLetter(rune) && rune.Value > 0x024F;
}

/// <summary>
/// Turns text into the phonemes a checkpoint was trained on: VITS checkpoints that set <c>phonemize</c> in
/// tokenizer_config.json (VCTK's, LJSpeech's) read IPA from espeak, as transformers' <c>VitsTokenizer</c> gets it from
/// the <c>phonemizer</c> package, so a synthesizer's tokenizer phonemizes such text before tokenizing. Registered in
/// <see cref="TextPhonemizers"/>.
/// </summary>
public interface ITextPhonemizer
{
    /// <summary>The name it is registered under ("espeak-ng").</summary>
    string Name { get; }

    /// <summary>
    /// The phonemes of <paramref name="text"/> in <paramref name="language"/> (an espeak voice name such as "en-us"):
    /// IPA with stress marks, words separated by a space, punctuation kept where it was.
    /// </summary>
    /// <exception cref="NotSupportedException">The phonemizer cannot run here (the message says what it needs).</exception>
    string Phonemize(string text, string language);
}

/// <summary>
/// The phonemizers, by name (ordinal). Built in: <see cref="EspeakNg"/> (the espeak-ng program, which the user
/// installs: the library runs it and carries none). Register another (a pure .NET G2P, a service) with
/// <see cref="Register"/>; one under a built-in name shadows it.
/// </summary>
public static class TextPhonemizers
{
    /// <summary>The espeak-ng program (<see cref="EspeakNgPhonemizer"/>).</summary>
    public const string EspeakNg = "espeak-ng";

    /// <summary>The punctuation a phonemizer keeps in place (the phonemizer package's default marks).</summary>
    public const string Marks = ";:,.!?¡¿—…\"«»“”(){}[]";

    private static readonly SlotTable<string, ITextPhonemizer> Table = BuiltIn();

    private static SlotTable<string, ITextPhonemizer> BuiltIn()
    {
        var table = new SlotTable<string, ITextPhonemizer>(nameof(TextPhonemizers), comparer: StringComparer.Ordinal, newestFirst: true);
        table.RegisterDefault(EspeakNg, new EspeakNgPhonemizer());
        return table;
    }

    /// <summary>Registers <paramref name="phonemizer"/> under its <see cref="ITextPhonemizer.Name"/> (asked before the earlier ones by <see cref="First"/>).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ITextPhonemizer phonemizer)
    {
        ArgumentNullException.ThrowIfNull(phonemizer);
        ArgumentException.ThrowIfNullOrWhiteSpace(phonemizer.Name);
        Table.Register(phonemizer.Name, phonemizer, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's phonemizer <paramref name="name"/> (a built-in name gets the built-in back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, the most recently registered first.</summary>
    public static IReadOnlyList<string> Names => Table.Keys;

    /// <summary>The phonemizer <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name: the message names the registered ones.</exception>
    public static ITextPhonemizer Get(string name) => Table.Find(name)
        ?? throw new NotSupportedException($"Phonemizer '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with TextPhonemizers.Register.");

    /// <summary>The most recently registered phonemizer (the built-in espeak-ng when an app registered none).</summary>
    public static ITextPhonemizer First() => Get(Table.Keys[0]);

    /// <summary>Who registered the phonemizer <paramref name="name"/>: <see cref="Overrides.Library"/> for the built-in or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>
    /// <paramref name="text"/> phonemized a stretch at a time by <paramref name="phonemize"/> between the
    /// <paramref name="marks"/> (punctuation), the marks kept in place and followed by a space where the text had one,
    /// as the phonemizer package's <c>preserve_punctuation</c> does; runs of white space become one space, and the ends
    /// are trimmed.
    /// </summary>
    public static string KeepingPunctuation(string text, Func<string, string> phonemize, string marks = Marks)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(phonemize);
        var result = new StringBuilder(text.Length * 2);
        int start = 0;
        for (int i = 0; i <= text.Length; i++)
        {
            if (i < text.Length && marks.IndexOf(text[i], StringComparison.Ordinal) < 0)
            {
                continue;
            }

            string stretch = text[start..i].Trim();
            if (stretch.Length > 0)
            {
                result.Append(phonemize(stretch).Trim());
            }

            if (i < text.Length)
            {
                if (result.Length > 0 && result[^1] == ' ')
                {
                    result.Length--;
                }

                result.Append(text[i]);
                if (i + 1 < text.Length && char.IsWhiteSpace(text[i + 1]))
                {
                    result.Append(' ');
                }
            }

            start = i + 1;
        }

        return Collapse(result.ToString());
    }

    /// <summary>Runs of white space as one space, the ends trimmed (what a phonemizer's output goes through before tokenizing).</summary>
    public static string Collapse(string text)
    {
        var result = new StringBuilder(text.Length);
        bool space = false;
        foreach (char c in text)
        {
            if (char.IsWhiteSpace(c))
            {
                space = result.Length > 0;
                continue;
            }

            if (space)
            {
                result.Append(' ');
                space = false;
            }

            result.Append(c);
        }

        return result.ToString();
    }
}
