// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// A text matcher's search, as a matcher under test gives it (Idrak.Text's <c>ITextMatcher.Best</c> with its options bound):
/// the span of <paramref name="text"/> from <paramref name="from"/> on nearest <paramref name="draft"/>, and the draft's
/// edit distance to the best span before any widening, or null when the draft or the rest of the text is empty.
/// </summary>
/// <param name="draft">The draft to find.</param>
/// <param name="text">The text to find it in.</param>
/// <param name="from">Where in the text the span may start (UTF-16 index).</param>
public delegate (int Start, int Length, int Distance)? TextMatchSearch(string draft, string text, int from);

/// <summary>
/// The checks of an approximate text matcher (Idrak.Text's <c>ITextMatcher</c>, given as a <see cref="TextMatchSearch"/>)
/// on texts of Latin and Arabic words and random letters, with drafts cut from them and edited:
/// <list type="bullet">
/// <item>nothing for an empty draft or nothing left of the text; else a span within the text from the start asked;</item>
/// <item>the distance is the least edit distance (over Unicode scalars) of the draft to any span of the text from that start
/// (every span tried on small cases);</item>
/// <item>a draft that is a run of the text's own words is found at distance 0, inside the span;</item>
/// <item>with <see cref="WordBoundaries"/>, the span starts and ends at white space or the text's ends;</item>
/// <item>the same span again and from several threads;</item>
/// <item>given a <see cref="Reference"/> (the library's matcher it replaces), the same spans.</item>
/// </list>
/// </summary>
/// <param name="reference">The library's matcher to agree with, or null.</param>
/// <param name="wordBoundaries">Whether the matcher widens its spans to word boundaries (as the library's does by default).</param>
public sealed class TextMatcherSuite(TextMatchSearch? reference = null, bool wordBoundaries = true) : ContractSuite<TextMatchSearch>
{
    private static readonly string[] Words = ["the", "first", "line", "here", "then", "a", "second", "and", "third", "one", "river", "النهر",
        "باب", "سم", "مدينة", "2026", "x"];

    /// <summary>The library's matcher to agree with, or null.</summary>
    public TextMatchSearch? Reference { get; } = reference;

    /// <summary>Whether spans are expected to start and end at word boundaries.</summary>
    public bool WordBoundaries { get; } = wordBoundaries;

    /// <inheritdoc />
    public override string Name => "text-matcher";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        const string Page = "the first line here then a second line and the third one";
        yield return Case("empty draft", "", Page, 0);
        yield return Case("from past the end", "line", Page, Page.Length);
        yield return Case("an exact run of words", "a second line", Page, 0);
        yield return Case("a draft with typos", "the frst line hre", Page, 0);
        yield return Case("from the middle", "the thrd one", Page, 30);
        yield return Case("nothing alike", "zzzzzzzzzzzz", Page, 0);
        yield return Case("arabic", "النهر بب", "مدينة النهر باب سم", 0);
        yield return Case("a surrogate pair", "b\U0001F600c", "a b\U0001F600c d", 0);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int words = random.Next(1, large ? 400 : 12);
        string text = string.Join(' ', Enumerable.Range(0, words).Select(_ => Words[random.Next(Words.Length)]));
        int from = random.Next(0, Math.Max(1, text.Length / 3));
        while (from > 0 && char.IsLowSurrogate(text[from]))
        {
            from--;
        }

        // A run of the text's words, then a few edits (or none).
        var parts = text.Split(' ');
        int first = random.Next(parts.Length), count = random.Next(1, Math.Min(4, parts.Length - first) + 1);
        var draft = new StringBuilder(string.Join(' ', parts.Skip(first).Take(count)));
        for (int e = random.Next(0, 3); e > 0 && draft.Length > 1; e--)
        {
            int at = random.Next(draft.Length);
            if (char.IsSurrogate(draft[at]))
            {
                continue;
            }

            switch (random.Next(3))
            {
                case 0:
                    draft.Remove(at, 1);
                    break;
                case 1:
                    draft.Insert(at, 'q');
                    break;
                default:
                    draft[at] = 'z';
                    break;
            }
        }

        return Case($"random text of {words} words", draft.ToString(), text, from);
    }

    /// <inheritdoc />
    public override void Run(TextMatchSearch implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        string draft = (string)@case.Data["draft"]!, text = (string)@case.Data["text"]!;
        int from = (int)@case.Data["from"]!;
        var found = implementation(draft, text, from);
        bool none = draft.Length == 0 || from >= text.Length;
        checks.Check("nothing for an empty draft or text", none == found is null, none ? $"found {found} for nothing to match" : "nothing found");
        if (found is not { } span)
        {
            return;
        }

        checks.Check("a span within the text from the start", span.Start >= from && span.Length >= 0 && span.Start + span.Length <= text.Length,
            $"[{span.Start}, {span.Start + span.Length}) in {text.Length} from {from}");
        if (span.Start < from || span.Start + span.Length > text.Length || span.Length < 0)
        {
            return;
        }

        int[] d = Scalars(draft), t = Scalars(text[from..]);
        if ((long)d.Length * t.Length * t.Length <= 2_000_000)
        {
            int least = Least(d, t);
            checks.Check("the least distance of any span", span.Distance == least, $"distance {span.Distance}, the least of any span {least}");
        }
        else
        {
            checks.Skip("the least distance of any span", "too long to try every span");
        }

        string piece = text.Substring(span.Start, span.Length);
        if (text.IndexOf(draft, from, StringComparison.Ordinal) >= 0)
        {
            checks.Check("a run of the text is found exactly", span.Distance == 0 && piece.Contains(draft, StringComparison.Ordinal), $"'{draft}' as '{piece}' at distance {span.Distance}");
        }

        if (WordBoundaries)
        {
            int end = span.Start + span.Length;
            checks.Check("the span's ends at word boundaries", (span.Start == from || char.IsWhiteSpace(text[span.Start - 1]) || span.Length == 0)
                                                              && (end == text.Length || char.IsWhiteSpace(text[end])),
                $"'{piece}' at [{span.Start}, {end})");
        }

        checks.Check("the same span again", implementation(draft, text, from) == found, "another span");
        var parallel = new (int, int, int)?[4];
        Parallel.For(0, parallel.Length, i => parallel[i] = implementation(draft, text, from));
        checks.Check("the same span from several threads", parallel.All(p => p == found), string.Join(", ", parallel));
        if (Reference is not null)
        {
            var expected = Reference(draft, text, from);
            checks.Check("agrees with the reference", expected == found, $"{found}, the reference {expected}");
        }
        else
        {
            checks.Skip("agrees with the reference", "no reference matcher given");
        }
    }

    private static int[] Scalars(string text) => [.. text.EnumerateRunes().Select(r => r.Value)];

    // The least edit distance of the draft to any span of the text (every start tried; the ends free by the last row).
    private static int Least(int[] draft, int[] text)
    {
        int least = draft.Length;
        var previous = new int[draft.Length + 1];
        var current = new int[draft.Length + 1];
        for (int start = 0; start <= text.Length; start++)
        {
            for (int i = 0; i <= draft.Length; i++)
            {
                previous[i] = i;
            }

            for (int j = start; j < text.Length; j++)
            {
                current[0] = j - start + 1;
                for (int i = 1; i <= draft.Length; i++)
                {
                    current[i] = Math.Min(Math.Min(previous[i] + 1, current[i - 1] + 1), previous[i - 1] + (draft[i - 1] == text[j] ? 0 : 1));
                }

                least = Math.Min(least, current[draft.Length]);
                (previous, current) = (current, previous);
            }
        }

        return least;
    }

    private static ContractCase Case(string name, string draft, string text, int from) =>
        new(name, new JsonObject { ["draft"] = draft, ["text"] = text, ["from"] = from });
}
