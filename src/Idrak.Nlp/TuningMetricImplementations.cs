// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using Idrak.Nlp.Abstractions;

namespace Idrak.Nlp;

// The library's error rates ("cer", "wer"): both texts NFC-normalized (optionally without nonspacing marks, optionally
// lower-cased invariantly), then the edit distance over Unicode scalar values or over white-space-separated words, over
// the reference's length.
internal sealed class ErrorRateMetric(bool words) : ITuningMetric
{
    private const string StripDiacritics = "strip_diacritics", IgnoreCase = "ignore_case";

    public string Name => words ? TuningMetrics.WordErrorRate : TuningMetrics.CharacterErrorRate;

    public string Summary => words
        ? "word error rate: edit distance between the white-space-separated words over the reference's words (NFC; strip_diacritics, ignore_case)"
        : "character error rate: edit distance between the characters (Unicode scalar values) over the reference's characters (NFC; strip_diacritics, ignore_case)";

    public bool LowerIsBetter => true;

    public IReadOnlyCollection<string> Keys { get; } = [StripDiacritics, IgnoreCase];

    public TuningScore Score(string answer, string reference, IReadOnlyDictionary<string, string>? options = null)
    {
        ArgumentNullException.ThrowIfNull(answer);
        ArgumentNullException.ThrowIfNull(reference);
        bool strip = false, fold = false;
        foreach (var (key, value) in options ?? new Dictionary<string, string>())
        {
            switch (key)
            {
                case StripDiacritics: strip = Flag(key, value); break;
                case IgnoreCase: fold = Flag(key, value); break;
                default: throw new ArgumentException($"The tuning metric {Name} does not take the option '{key}'; it takes: {string.Join(", ", Keys)}.");
            }
        }

        string a = Prepare(answer, strip, fold), r = Prepare(reference, strip, fold);
        if (words)
        {
            var ids = new Dictionary<string, int>(StringComparer.Ordinal);
            int[] answerWords = Words(a, ids), referenceWords = Words(r, ids);
            return new TuningScore(TuningMetrics.EditDistance<int>(answerWords, referenceWords), referenceWords.Length);
        }

        int[] answerChars = Scalars(a), referenceChars = Scalars(r);
        return new TuningScore(TuningMetrics.EditDistance<int>(answerChars, referenceChars), referenceChars.Length);
    }

    public override string ToString() => Name;

    private static bool Flag(string key, string value) => value.Trim().ToLowerInvariant() switch
    {
        "true" or "1" or "yes" or "on" => true,
        "false" or "0" or "no" or "off" => false,
        _ => throw new ArgumentException($"The tuning metric option {key} is true or false, not '{value}'."),
    };

    // NFC; without nonspacing marks (decomposed first, so precomposed accents go too) when asked; invariant lower case when asked.
    private static string Prepare(string text, bool strip, bool fold)
    {
        if (strip)
        {
            var kept = new StringBuilder(text.Length);
            foreach (var rune in text.Normalize(NormalizationForm.FormD).EnumerateRunes())
            {
                if (Rune.GetUnicodeCategory(rune) != UnicodeCategory.NonSpacingMark)
                {
                    kept.Append(rune.ToString());
                }
            }

            text = kept.ToString();
        }

        text = text.Normalize(NormalizationForm.FormC);
        return fold ? text.ToLowerInvariant() : text;
    }

    private static int[] Scalars(string text)
    {
        var values = new List<int>(text.Length);
        foreach (var rune in text.EnumerateRunes())
        {
            values.Add(rune.Value);
        }

        return [.. values];
    }

    // Each word as an id shared by both texts (equal words, equal ids).
    private static int[] Words(string text, Dictionary<string, int> ids)
    {
        var result = new List<int>();
        int start = -1;
        for (int i = 0; i <= text.Length; i++)
        {
            bool space = i == text.Length || char.IsWhiteSpace(text[i]);
            if (space && start >= 0)
            {
                string word = text[start..i];
                if (!ids.TryGetValue(word, out int id))
                {
                    ids[word] = id = ids.Count;
                }

                result.Add(id);
                start = -1;
            }
            else if (!space && start < 0)
            {
                start = i;
            }
        }

        return [.. result];
    }
}
