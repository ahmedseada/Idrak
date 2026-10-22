// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using Idrak.Abstraction.Text;

namespace Idrak.Abstraction.Generation;

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
            Span<char> units = stackalloc char[2];                         // a rune's UTF-16 (no string per rune)
            foreach (var rune in UnicodeNormalization.Normalize(text, NormalizationForm.FormD).EnumerateRunes())
            {
                if (Rune.GetUnicodeCategory(rune) != UnicodeCategory.NonSpacingMark)
                {
                    kept.Append(units[..rune.EncodeToUtf16(units)]);
                }
            }

            text = kept.ToString();
        }

        text = UnicodeNormalization.Normalize(text, NormalizationForm.FormC);
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

    // Each word as an id shared by both texts (equal words, equal ids); a word is looked up by its span, a string kept
    // only for a new word.
    private static int[] Words(string text, Dictionary<string, int> ids)
    {
        var lookup = ids.GetAlternateLookup<ReadOnlySpan<char>>();
        var result = new List<int>();
        int start = -1;
        for (int i = 0; i <= text.Length; i++)
        {
            bool space = i == text.Length || char.IsWhiteSpace(text[i]);
            if (space && start >= 0)
            {
                ref int id = ref CollectionsMarshal.GetValueRefOrAddDefault(lookup, text.AsSpan(start, i - start), out bool exists);
                if (!exists)
                {
                    id = ids.Count - 1;
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

// The library's answer metrics ("number", "exact", "contains", "f1"), higher is better, each answer scored out of 1: the
// scoring of chat evaluations (ChatEvaluation in Idrak.Text) as it was, with its normalization (invariant lower case,
// runs of white space as one space, final . ! ? dropped; no Unicode normalization).
internal sealed partial class AnswerMatchMetric(string name) : ITuningMetric
{
    public string Name => name;

    public string Summary => name switch
    {
        TuningMetrics.NumberMatch => "the final number of each (after \"####\" when present, else the last one) is the same: 1 or 0",
        TuningMetrics.ExactMatch => "the whole answer equals the reference, ignoring case, spacing and final punctuation: 1 or 0",
        TuningMetrics.ContainsMatch => "the reference appears in the answer, ignoring case and spacing: 1 or 0",
        _ => "word overlap of answer and reference (SQuAD's F1 of the lowered letter and digit runs), from 0 to 1",
    };

    public bool LowerIsBetter => false;

    public IReadOnlyCollection<string> Keys { get; } = [];

    public TuningScore Score(string answer, string reference, IReadOnlyDictionary<string, string>? options = null)
    {
        ArgumentNullException.ThrowIfNull(answer);
        ArgumentNullException.ThrowIfNull(reference);
        if (options is { Count: > 0 })
        {
            throw new ArgumentException($"The tuning metric {Name} takes no options; it was given '{string.Join("', '", options.Keys)}'.");
        }

        double value = name switch
        {
            TuningMetrics.NumberMatch => FinalNumber(answer) is { } a && FinalNumber(reference) is { } r && a == r ? 1 : 0,
            TuningMetrics.ExactMatch => Normalize(answer) == Normalize(reference) ? 1 : 0,
            TuningMetrics.ContainsMatch => Normalize(answer).Contains(Normalize(reference), StringComparison.Ordinal) ? 1 : 0,
            _ => F1(answer, reference),
        };
        return new TuningScore(value, 1);
    }

    public override string ToString() => Name;

    // Whether the text holds gsm8k's final-answer marker, "#### " and the answer.
    public static bool HasFinalAnswer(string text) => FinalMarker().IsMatch(text);

    // The number an answer ends with: after "####" when present, else the last number in the text.
    public static decimal? FinalNumber(string text)
    {
        var marked = FinalMarker().Match(text);
        string scope = marked.Success ? marked.Groups[1].Value : text;
        var numbers = NumberPattern().Matches(scope);
        if (numbers.Count == 0)
        {
            return null;
        }

        string value = (marked.Success ? numbers[0] : numbers[^1]).Value.Replace(",", "", StringComparison.Ordinal).TrimEnd('.');
        return decimal.TryParse(value, NumberStyles.Number, CultureInfo.InvariantCulture, out var number) ? number : null;
    }

    // string.Join(' ', text.ToLowerInvariant().Split(null, RemoveEmptyEntries)).Trim().TrimEnd('.', '!', '?') in one
    // buffer (the same casing, white space and trimming), with a string only for the result.
    private static string Normalize(string text)
    {
        char[]? rented = null;
        Span<char> buffer = text.Length <= 256 ? stackalloc char[256] : (rented = ArrayPool<char>.Shared.Rent(text.Length));
        try
        {
            var lowered = buffer[..text.AsSpan().ToLowerInvariant(buffer)];
            int length = 0;
            for (int i = 0; i < lowered.Length; i++)
            {
                if (char.IsWhiteSpace(lowered[i]))
                {
                    continue;
                }

                if (length > 0)
                {
                    lowered[length++] = ' ';                                  // one space between runs of other characters
                }

                while (i < lowered.Length && !char.IsWhiteSpace(lowered[i]))
                {
                    lowered[length++] = lowered[i++];
                }
            }

            return new string(lowered[..length].TrimEnd(".!?"));
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<char>.Shared.Return(rented);
            }
        }
    }

    // SQuAD's F1 of the lowered words ([\p{L}\p{N}]+), counted by span: no string per word, a key only per distinct
    // reference word.
    private static double F1(string answer, string reference)
    {
        char[] rented = ArrayPool<char>.Shared.Rent(Math.Max(1, answer.Length + reference.Length));
        try
        {
            var a = rented.AsSpan(0, answer.AsSpan().ToLowerInvariant(rented));
            var r = rented.AsSpan(a.Length, reference.AsSpan().ToLowerInvariant(rented.AsSpan(a.Length)));
            var counts = new Dictionary<string, int>();
            var lookup = counts.GetAlternateLookup<ReadOnlySpan<char>>();
            int referenceWords = 0;
            foreach (var match in WordPattern().EnumerateMatches(r))
            {
                referenceWords++;
                CollectionsMarshal.GetValueRefOrAddDefault(lookup, r.Slice(match.Index, match.Length), out _)++;
            }

            int answerWords = 0, common = 0;
            foreach (var match in WordPattern().EnumerateMatches(a))
            {
                answerWords++;
                ref int n = ref CollectionsMarshal.GetValueRefOrNullRef(lookup, a.Slice(match.Index, match.Length));
                if (!Unsafe.IsNullRef(ref n) && n > 0)
                {
                    n--;
                    common++;
                }
            }

            if (answerWords == 0 || referenceWords == 0)
            {
                return answerWords == referenceWords ? 1 : 0;
            }

            if (common == 0)
            {
                return 0;
            }

            double precision = (double)common / answerWords, recall = (double)common / referenceWords;
            return 2 * precision * recall / (precision + recall);
        }
        finally
        {
            ArrayPool<char>.Shared.Return(rented);
        }
    }

    [GeneratedRegex(@"####\s*(.+)$", RegexOptions.Multiline)]
    private static partial Regex FinalMarker();

    [GeneratedRegex(@"-?\d[\d,]*(\.\d+)?")]
    private static partial Regex NumberPattern();

    [GeneratedRegex(@"[\p{L}\p{N}]+")]
    private static partial Regex WordPattern();
}
