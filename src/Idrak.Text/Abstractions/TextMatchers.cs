// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Text.Abstractions;

/// <summary>Where a draft lies in a text: a span of the text (UTF-16 indexes) and how far the draft is from it.</summary>
/// <param name="Start">The span's first character in the text.</param>
/// <param name="Length">The span's length.</param>
/// <param name="Distance">The edit distance between the draft and the span (in Unicode scalars).</param>
public readonly record struct TextMatch(int Start, int Length, int Distance);

/// <summary>A draft snapped to its span of a text (<see cref="TextMatchers.Align"/>).</summary>
/// <param name="Text">The span's text, as it is in the text.</param>
/// <param name="Span">Where it is, and the draft's distance to it.</param>
/// <param name="Match">1 - distance / the draft's scalars: 1 for a draft that is the span exactly.</param>
public sealed record AlignedText(string Text, TextMatch Span, double Match);

/// <summary>How a draft is matched in a text (<see cref="ITextMatcher"/>); each matcher reads those it has.</summary>
public sealed record TextMatchOptions
{
    /// <summary>Whether the span's ends are moved out to word boundaries (white space or the text's ends). Default true: a line breaks between words.</summary>
    public bool WordBoundaries { get; init; } = true;

    /// <summary>
    /// For <see cref="TextMatchers.Align"/>: the largest share of a draft's characters its span may differ in (edit
    /// distance over the draft's scalars); a draft further from its best span is left unaligned. Default 0.5.
    /// </summary>
    public double MaxError { get; init; } = 0.5;

    /// <summary>Settings of a matcher of one's own, by name (empty for the library's).</summary>
    public IReadOnlyDictionary<string, double> Parameters { get; init; } = new Dictionary<string, double>();
}

/// <summary>
/// Approximate matching of a draft inside a longer text: the span of the text nearest the draft (a recognizer's reading of
/// one line inside its page's known text, a quotation inside its source). The library's is "semi-global"
/// (<see cref="SemiGlobalMatcher"/>: Sellers' edit distance, the span's ends free); another (weighted edits,
/// a confusion table of a script's letters) is an app's. Picking one by its name is not on the new design yet.
/// </summary>
public interface ITextMatcher
{
    /// <summary>Its name in <see cref="TextMatchers"/>.</summary>
    string Name { get; }

    /// <summary>
    /// The span of <paramref name="text"/>, starting at or after <paramref name="from"/>, nearest <paramref name="draft"/>;
    /// null when the draft or the rest of the text is empty.
    /// </summary>
    TextMatch? Best(string draft, string text, int from, TextMatchOptions options);
}

/// <summary>
/// The text matchers. The library's: <see cref="SemiGlobal"/> (Sellers' semi-global edit distance over Unicode scalars,
/// the span's ends at word boundaries), which <see cref="Best"/> and <see cref="Align"/> (a page's drafts snapped to its
/// text in order) reach directly.
/// </summary>
public static class TextMatchers
{
    private static readonly SemiGlobalMatcher Matcher = new();
    /// <summary>"semi-global": the nearest span by edit distance, its ends free in the text.</summary>
    public const string SemiGlobal = "semi-global";

    /// <summary>The span of <paramref name="text"/> nearest <paramref name="draft"/>, by the semi-global matcher (<see cref="SemiGlobal"/>).</summary>
    public static TextMatch? Best(string draft, string text, int from = 0, TextMatchOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(draft);
        ArgumentNullException.ThrowIfNull(text);
        ArgumentOutOfRangeException.ThrowIfNegative(from);
        return Matcher.Best(draft, text, from, options ?? new TextMatchOptions());
    }

    /// <summary>
    /// Each draft's span in <paramref name="text"/>, in order (a page's lines top to bottom): each search starts where the
    /// previous aligned span ended. A draft whose best span differs in more than <see cref="TextMatchOptions.MaxError"/> of
    /// its characters is left unaligned (null), and the next draft searches from the same place.
    /// </summary>
    /// <param name="drafts">The drafts, in the text's order.</param>
    /// <param name="text">The text they are parts of.</param>
    /// <param name="options">The matching settings (<see cref="TextMatchOptions.MaxError"/> among them).</param>
    public static IReadOnlyList<AlignedText?> Align(IReadOnlyList<string> drafts, string text, TextMatchOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(drafts);
        ArgumentNullException.ThrowIfNull(text);
        options ??= new TextMatchOptions();
        var matcher = Matcher;
        var result = new List<AlignedText?>(drafts.Count);
        int from = 0;
        foreach (string draft in drafts)
        {
            int scalars = Scalars(draft);
            if (matcher.Best(draft, text, from, options) is { } span && span.Distance <= options.MaxError * scalars)
            {
                result.Add(new AlignedText(text.Substring(span.Start, span.Length), span, 1 - span.Distance / (double)Math.Max(1, scalars)));
                from = span.Start + span.Length;
            }
            else
            {
                result.Add(null);
            }
        }

        return result;
    }

    // The draft's Unicode scalars (a lone surrogate counts as one, as EnumerateRunes reads it).
    private static int Scalars(string text)
    {
        int count = 0;
        foreach (var _ in text.EnumerateRunes())
        {
            count++;
        }

        return count;
    }
}
