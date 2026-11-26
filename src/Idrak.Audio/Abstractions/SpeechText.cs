// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Audio.Abstractions;

/// <summary>
/// Turns text of a script into the Latin letters a checkpoint was trained on: MMS-TTS checkpoints that set
/// <c>is_uroman</c> in tokenizer_config.json read text romanised by uroman, so a synthesizer's tokenizer romanises
/// such text before tokenizing.
/// </summary>
public interface ITextRomanizer
{
    /// <summary>Its name (Idrak.Audio.Plugins' VITS family's uroman Arabic is "uroman-arabic").</summary>
    string Name { get; }

    /// <summary>Whether it romanises <paramref name="rune"/> (a character of its script).</summary>
    bool Covers(Rune rune);

    /// <summary><paramref name="text"/> with the characters it covers romanised and the others as they are.</summary>
    string Romanize(string text);
}

/// <summary>
/// What the romanizers share. None is built in: a romanizer belongs to the family whose checkpoints read its letters, and
/// comes with that family's plug-in (Idrak.Audio.Plugins' VITS family has uroman's romanisation of the Arabic script).
/// </summary>
public static class TextRomanizers
{
    /// <summary>Whether <paramref name="rune"/> is a letter outside the Latin blocks (past Latin Extended-B): one a romanizer may cover.</summary>
    public static bool IsNonLatinLetter(Rune rune) => Rune.IsLetter(rune) && rune.Value > 0x024F;
}

/// <summary>
/// Turns text into the phonemes a checkpoint was trained on: VITS checkpoints that set <c>phonemize</c> in
/// tokenizer_config.json (VCTK's, LJSpeech's) read IPA from espeak, as transformers' <c>VitsTokenizer</c> gets it from
/// the <c>phonemizer</c> package, so a synthesizer's tokenizer phonemizes such text before tokenizing.
/// </summary>
public interface ITextPhonemizer
{
    /// <summary>Its name (Idrak.Audio.Plugins' espeak-ng phonemizer is "espeak-ng").</summary>
    string Name { get; }

    /// <summary>
    /// The phonemes of <paramref name="text"/> in <paramref name="language"/> (an espeak voice name such as "en-us"):
    /// IPA with stress marks, words separated by a space, punctuation kept where it was.
    /// </summary>
    /// <exception cref="NotSupportedException">The phonemizer cannot run here (the message says what it needs).</exception>
    string Phonemize(string text, string language);
}

/// <summary>
/// What the phonemizers share. None is built in: the library starts no outside program (rule 70), so a phonemizer is a
/// plug-in's (Idrak.Audio.Plugins' "espeak-ng" runs the espeak-ng program the user installs; the VITS family's phonemize
/// checkpoints read espeak's IPA).
/// </summary>
public static class TextPhonemizers
{
    /// <summary>The punctuation a phonemizer keeps in place (the phonemizer package's default marks).</summary>
    public const string Marks = ";:,.!?¡¿—…\"«»“”(){}[]";

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
