// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// A score as a ratio, so scores of many answers add up the way the metric means them to: an error rate is the errors of
/// every answer over the length of every reference (<see cref="Sum"/>), not the mean of each answer's rate.
/// </summary>
/// <param name="Numerator">What was counted (errors, matches).</param>
/// <param name="Denominator">What it is counted out of (the reference's characters or words).</param>
public readonly record struct TuningScore(double Numerator, double Denominator)
{
    /// <summary>
    /// <see cref="Numerator"/> over <see cref="Denominator"/>; when the denominator is 0 (an empty reference), 0 for a
    /// numerator of 0 and 1 otherwise (all wrong).
    /// </summary>
    public double Value => Denominator > 0 ? Numerator / Denominator : Numerator > 0 ? 1 : 0;

    /// <summary>The scores added term by term (the score of all their answers together).</summary>
    public static TuningScore operator +(TuningScore a, TuningScore b) => new(a.Numerator + b.Numerator, a.Denominator + b.Denominator);

    /// <summary>The scores of <paramref name="scores"/> added together.</summary>
    public static TuningScore Sum(IEnumerable<TuningScore> scores)
    {
        ArgumentNullException.ThrowIfNull(scores);
        var total = default(TuningScore);
        foreach (var score in scores)
        {
            total += score;
        }

        return total;
    }

    /// <summary>"0.0625 (2 / 32)".</summary>
    public override string ToString() => FormattableString.Invariant($"{Value:G4} ({Numerator:G6} / {Denominator:G6})");
}

/// <summary>
/// A score of generated text against its reference, for a fine-tune's evaluation set (an answer generated every few
/// steps
/// and compared with the expected one): character or word error rate, exact match, or a metric of one's own (the
/// library's are named in <see cref="TuningMetrics"/>: "cer", "wer", "f1"). Thread-safe: one metric scores many answers
/// at once.
/// </summary>
public interface ITuningMetric
{
    /// <summary>Whether a lower <see cref="TuningScore.Value"/> is better (an error rate), or a higher one (a match rate).</summary>
    bool LowerIsBetter { get; }

    /// <summary>The score of <paramref name="answer"/> against <paramref name="reference"/>.</summary>
    /// <param name="answer">The generated text.</param>
    /// <param name="reference">The expected text.</param>
    TuningScore Score(string answer, string reference);
}

/// <summary>
/// The names of the library's metrics a fine-tune's evaluation, a chat evaluation or a reward can score generated text
/// with (picking one by its name is not on the new design yet), and the helpers they share. The library's error rates,
/// lower is better, are Unicode-aware (both texts NFC-normalized first, so a letter
/// written precomposed or as a base and its marks counts the same; Arabic letters and their marks are characters like
/// any other): <list type="bullet"> <item><see cref="CharacterErrorRate"/> ("cer"): the edit distance between the
/// texts' characters (Unicode scalar values, so a character outside the basic plane is one) over the reference's
/// characters.</item> <item><see cref="WordErrorRate"/> ("wer"): the same over words, the text split at white space
/// (runs of it count as one split, leading and trailing ignored).</item> </list> Both take the options
/// <c>strip_diacritics</c> (true: marks that combine with a letter, Unicode's nonspacing marks such as Arabic's harakat
/// and Latin accents, are removed from both texts first; false by default: a missing or wrong mark is an error) and
/// <c>ignore_case</c> (false by default). The library's answer metrics, higher is better, score each answer out of 1
/// and take no options.
/// </summary>
public static class TuningMetrics
{
    /// <summary>The character error rate's name: "cer".</summary>
    public const string CharacterErrorRate = "cer";

    /// <summary>The word error rate's name: "wer".</summary>
    public const string WordErrorRate = "wer";

    /// <summary>The final numbers' match: "number" (gsm8k's answers, "#### n").</summary>
    public const string NumberMatch = "number";

    /// <summary>The whole answer's match: "exact".</summary>
    public const string ExactMatch = "exact";

    /// <summary>The reference found in the answer: "contains".</summary>
    public const string ContainsMatch = "contains";

    /// <summary>The words' F1: "f1".</summary>
    public const string WordF1 = "f1";

    /// <summary>
    /// The number <paramref name="text"/> ends with, as <see cref="NumberMatch"/> reads it: the first number after "####"
    /// when the text holds that marker (gsm8k's layout), else its last number; thousands separators (",") are dropped. Null
    /// when it holds none.
    /// </summary>
    public static decimal? FinalNumber(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return AnswerMatchMetric.FinalNumber(text);
    }

    /// <summary>Whether <paramref name="text"/> holds gsm8k's final-answer marker: "####", then the answer on its line.</summary>
    public static bool HasFinalAnswer(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return AnswerMatchMetric.HasFinalAnswer(text);
    }

    /// <summary>
    /// The edit distance (Levenshtein: insertions, deletions and substitutions, each 1) between <paramref name="a"/> and
    /// <paramref name="b"/>, exact, in memory proportional to the shorter: what they share at the start and the end is
    /// trimmed first, then the rest is compared 64 elements of the shorter at a time (Myers' bit-vector algorithm in
    /// Hyyrö's form, in blocks), in time proportional to the product of their lengths over 64; a shorter side of more than
    /// 512 distinct elements and over 64 long is compared an element at a time.
    /// </summary>
    public static int EditDistance<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b) where T : IEquatable<T>
    {
        // What the two share at the start and the end costs nothing (a vector compare for numbers).
        int prefix = a.CommonPrefixLength(b);
        a = a[prefix..];
        b = b[prefix..];
        int suffix = 0;
        var equal = EqualityComparer<T>.Default;
        while (suffix < a.Length && suffix < b.Length && equal.Equals(a[a.Length - 1 - suffix], b[b.Length - 1 - suffix]))
        {
            suffix++;
        }

        a = a[..^suffix];
        b = b[..^suffix];
        if (a.Length < b.Length)
        {
            var swap = a;
            a = b;
            b = swap;
        }

        return b.Length == 0 ? a.Length : BitVectors(a, b) ?? Rows(a, b);
    }

    // Myers' bit-vector edit distance in Hyyrö's form, in blocks of 64 (Hyyrö 2003, as rapidfuzz writes it): the table's
    // column over `pattern` (the shorter) is kept as vertical +1 and -1 bit vectors, VP and VN, one bit per row; each
    // element of `text` advances every row at once, block after block, the horizontal +1/-1 at a block's last row carried
    // into the next block's first; the score follows the last row. The match masks are one bit per row for each distinct
    // element of the pattern. Null when the pattern holds more than 512 distinct elements and more than one block (the
    // masks would outgrow the one-row table of Rows: 16 times its bytes at 512).
    private static int? BitVectors<T>(ReadOnlySpan<T> text, ReadOnlySpan<T> pattern) where T : IEquatable<T>
    {
        const int MostSymbols = 512;
        int m = pattern.Length, words = (m + 63) >> 6;
        var symbols = new Dictionary<Symbol<T>, int>();
        for (int i = 0; i < m; i++)
        {
            if (symbols.TryAdd(new Symbol<T>(pattern[i]), symbols.Count) && symbols.Count > MostSymbols && words > 1)
            {
                return null;
            }
        }

        int masksLength = symbols.Count * words;
        ulong[] rented = ArrayPool<ulong>.Shared.Rent(masksLength + 2 * words);
        try
        {
            var masks = rented.AsSpan(0, masksLength);
            var vp = rented.AsSpan(masksLength, words);
            var vn = rented.AsSpan(masksLength + words, words);
            masks.Clear();
            for (int i = 0; i < m; i++)
            {
                masks[symbols[new Symbol<T>(pattern[i])] * words + (i >> 6)] |= 1UL << (i & 63);
            }

            vp.Fill(ulong.MaxValue);                                                 // the first column: 0, 1, 2, ..., m
            vn.Clear();
            ulong last = 1UL << ((m - 1) & 63);
            int score = m;
            foreach (var element in text)
            {
                int at = symbols.TryGetValue(new Symbol<T>(element), out int symbol) ? symbol * words : -1;
                ulong hpCarry = 1, hnCarry = 0;                                       // the first row: one more each step
                for (int w = 0; w < words; w++)
                {
                    ulong match = at < 0 ? 0 : masks[at + w];
                    ulong p = vp[w], n = vn[w];
                    ulong x = match | hnCarry;
                    ulong d0 = (((x & p) + p) ^ p) | x | n;
                    ulong hp = n | ~(d0 | p);
                    ulong hn = d0 & p;
                    ulong hpOut, hnOut;
                    if (w < words - 1)
                    {
                        hpOut = hp >> 63;
                        hnOut = hn >> 63;
                    }
                    else
                    {
                        hpOut = (hp & last) != 0 ? 1UL : 0UL;
                        hnOut = (hn & last) != 0 ? 1UL : 0UL;
                        score += (int)hpOut - (int)hnOut;
                    }

                    hp = (hp << 1) | hpCarry;
                    hn = (hn << 1) | hnCarry;
                    vp[w] = hn | ~(d0 | hp);
                    vn[w] = hp & d0;
                    hpCarry = hpOut;
                    hnCarry = hnOut;
                }
            }

            return score;
        }
        finally
        {
            ArrayPool<ulong>.Shared.Return(rented);
        }
    }

    // An element of a sequence as a dictionary key (a null one too), equal as the sequence's elements are.
    private readonly struct Symbol<T>(T value) : IEquatable<Symbol<T>> where T : IEquatable<T>
    {
        private readonly T _value = value;

        public bool Equals(Symbol<T> other) => EqualityComparer<T>.Default.Equals(_value, other._value);

        public override bool Equals(object? obj) => obj is Symbol<T> other && Equals(other);

        // A value type is hashed as it is: a null test on T boxes it in a method's code before it is optimized
        // (optimization rule 89), and this runs on every lookup.
        public override int GetHashCode() => typeof(T).IsValueType ? _value.GetHashCode() : _value?.GetHashCode() ?? 0;
    }

    // The table a row at a time over the shorter sequence `b`.
    private static int Rows<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b) where T : IEquatable<T>
    {
        // One row over the shorter sequence (pooled: a metric scores every record): row[j] is the distance between a[..i]
        // and b[..j].
        int[] rented = ArrayPool<int>.Shared.Rent(b.Length + 1);
        try
        {
            var row = rented.AsSpan(0, b.Length + 1);
            for (int j = 0; j <= b.Length; j++)
            {
                row[j] = j;
            }

            for (int i = 1; i <= a.Length; i++)
            {
                int diagonal = row[0];
                row[0] = i;
                for (int j = 1; j <= b.Length; j++)
                {
                    int above = row[j];
                    int cost = a[i - 1].Equals(b[j - 1]) ? 0 : 1;
                    row[j] = Math.Min(Math.Min(above + 1, row[j - 1] + 1), diagonal + cost);
                    diagonal = above;
                }
            }

            return row[b.Length];
        }
        finally
        {
            ArrayPool<int>.Shared.Return(rented);
        }
    }

}
