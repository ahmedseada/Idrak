// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.CompilerServices;

namespace Idrak.Nlp.Abstractions;

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
/// A score of generated text against its reference, for a fine-tune's evaluation set (an answer generated every few steps
/// and compared with the expected one): character or word error rate, exact match, or a metric of one's own. Registered in
/// <see cref="TuningMetrics"/> under <see cref="Name"/>. Thread-safe: one metric scores many answers at once.
/// </summary>
public interface ITuningMetric
{
    /// <summary>The metric's name in <see cref="TuningMetrics"/> ("cer", "wer").</summary>
    string Name { get; }

    /// <summary>One line for help and listings.</summary>
    string Summary { get; }

    /// <summary>Whether a lower <see cref="TuningScore.Value"/> is better (an error rate), or a higher one (a match rate).</summary>
    bool LowerIsBetter { get; }

    /// <summary>The option keys <see cref="Score"/> takes (none by default); any other is refused, naming these.</summary>
    IReadOnlyCollection<string> Keys { get; }

    /// <summary>The score of <paramref name="answer"/> against <paramref name="reference"/>.</summary>
    /// <param name="answer">The generated text.</param>
    /// <param name="reference">The expected text.</param>
    /// <param name="options">The metric's options by key (<see cref="Keys"/>), values as text; null for its defaults.</param>
    /// <exception cref="ArgumentException">An option the metric does not take, or a bad value.</exception>
    TuningScore Score(string answer, string reference, IReadOnlyDictionary<string, string>? options = null);
}

/// <summary>
/// The metrics a fine-tune's evaluation can score generated text with, by name (ignoring case). The library's, Unicode-aware
/// (both texts NFC-normalized first, so a letter written precomposed or as a base and its marks counts the same; Arabic
/// letters and their marks are characters like any other):
/// <list type="bullet">
/// <item><see cref="CharacterErrorRate"/> ("cer"): the edit distance between the texts' characters (Unicode scalar
/// values, so a character outside the basic plane is one) over the reference's characters.</item>
/// <item><see cref="WordErrorRate"/> ("wer"): the same over words, the text split at white space (runs of it count as
/// one split, leading and trailing ignored).</item>
/// </list>
/// Both take the options <c>strip_diacritics</c> (true: marks that combine with a letter, Unicode's nonspacing marks such
/// as Arabic's harakat and Latin accents, are removed from both texts first; false by default: a missing or wrong mark is
/// an error) and <c>ignore_case</c> (false by default). Register another with <see cref="Register"/>; one under a library
/// name shadows the library's, which <see cref="Unregister"/> brings back.
/// </summary>
public static class TuningMetrics
{
    /// <summary>The character error rate's name: "cer".</summary>
    public const string CharacterErrorRate = "cer";

    /// <summary>The word error rate's name: "wer".</summary>
    public const string WordErrorRate = "wer";

    private static readonly SlotTable<string, ITuningMetric> Table = BuiltIn();

    private static SlotTable<string, ITuningMetric> BuiltIn()
    {
        var table = new SlotTable<string, ITuningMetric>(nameof(TuningMetrics), (slot, app, library) => new GuardedMetric(slot, app, library), StringComparer.OrdinalIgnoreCase);
        table.RegisterDefault(CharacterErrorRate, new ErrorRateMetric(words: false));
        table.RegisterDefault(WordErrorRate, new ErrorRateMetric(words: true));
        return table;
    }

    /// <summary>Registers <paramref name="metric"/> under its <see cref="ITuningMetric.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ITuningMetric metric)
    {
        ArgumentNullException.ThrowIfNull(metric);
        ArgumentException.ThrowIfNullOrWhiteSpace(metric.Name);
        Table.Register(metric.Name, metric, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's metric <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The metric registered as <paramref name="name"/>, or null.</summary>
    public static ITuningMetric? Find(string name) => Table.Find(name);

    /// <summary>The metric registered as <paramref name="name"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones.</exception>
    public static ITuningMetric Get(string name) => Table.Find(name)
        ?? throw new NotSupportedException($"Tuning metric '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with TuningMetrics.Register.");

    /// <summary>The library's metric <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static ITuningMetric? Default(string name) => Table.Default(name);

    /// <summary>Who registered the metric <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>
    /// What happens when the app's metric <paramref name="name"/> fails to score (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/> scores with the library's; <see cref="SlotPolicy.Shadow"/> lets the library's score
    /// and compares the app's on a sample of calls).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>Each registered metric as "name: summary", one per line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(m => $"{m.Name}: {m.Summary}"));

    /// <summary>The score of every answer against its reference together (<see cref="TuningScore.Sum"/>), by the metric <paramref name="name"/>.</summary>
    public static TuningScore Score(string name, IEnumerable<(string Answer, string Reference)> pairs, IReadOnlyDictionary<string, string>? options = null)
    {
        ArgumentNullException.ThrowIfNull(pairs);
        var metric = Get(name);
        return TuningScore.Sum(pairs.Select(p => metric.Score(p.Answer, p.Reference, options)));
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

        public override int GetHashCode() => _value?.GetHashCode() ?? 0;
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

    private sealed class GuardedMetric(Slot slot, ITuningMetric app, ITuningMetric library) : ITuningMetric
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public bool LowerIsBetter => app.LowerIsBetter;

        public IReadOnlyCollection<string> Keys => app.Keys;

        public TuningScore Score(string answer, string reference, IReadOnlyDictionary<string, string>? options = null) =>
            slot.Call(() => app.Score(answer, reference, options), () => library.Score(answer, reference, options),
                (a, b) => a == b ? null : $"{a} against the library's {b}");
    }
}
