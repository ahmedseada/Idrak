// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// The built-in normalizers and pre-tokenizers on spans (plan 24, L1): each gives the pieces its string form in
/// <see cref="BpeTokenizer"/> gives, without a string per call, and keeps where every character came from. A type an app
/// registered runs through its own component's span form (<see cref="ITokenizerNormalizer.Normalize(NormalizedText)"/>).
/// </summary>
internal static class SpanComponents
{
    // The normalizer a tokenizer.json names: the app's component when one is registered for its type, else the built-in.
    public static Action<NormalizedText> Normalizer(JsonObject spec) =>
        TokenizerComponents.FindNormalizer((string?)spec["type"]) is { } create ? create(spec).Normalize : BuiltInNormalizer(spec);

    // The pre-tokenizer a tokenizer.json names: the app's component when one is registered for its type, else the built-in.
    public static Action<PreTokenizedText, bool> PreTokenizer(JsonObject spec) =>
        TokenizerComponents.FindPreTokenizer((string?)spec["type"]) is { } create ? create(spec).PreTokenize : BuiltInPreTokenizer(spec);

    // The library's own normalizer of the spec's type (inside a sequence, registered types still answer for themselves).
    public static Action<NormalizedText> BuiltInNormalizer(JsonObject spec)
    {
        string type = (string)spec["type"]!;
        return type switch
        {
            "Sequence" => new SequenceNormalizer([.. spec["normalizers"]!.AsArray().OfType<JsonObject>().Select(Normalizer)]).Run,
            "NFC" => new UnicodeNormalizer(NormalizationForm.FormC).Run,
            "NFKC" => new UnicodeNormalizer(NormalizationForm.FormKC).Run,
            "NFD" => new UnicodeNormalizer(NormalizationForm.FormD).Run,
            "NFKD" => new UnicodeNormalizer(NormalizationForm.FormKD).Run,
            "Prepend" => new PrependNormalizer((string)spec["prepend"]!).Run,
            "Replace" => spec["pattern"]?["String"] is { } literal
                ? new LiteralReplaceNormalizer((string)literal!, (string)spec["content"]!).Run
                : new RegexReplaceNormalizer(new Regex((string)spec["pattern"]!["Regex"]!, BpeTokenizer.Fast), (string)spec["content"]!).Run,
            "Lowercase" => new LowercaseNormalizer().Run,
            "Strip" => new StripNormalizer((bool?)spec["strip_left"] ?? true, (bool?)spec["strip_right"] ?? true).Run,
            "BertNormalizer" => BertComponents.Normalizer(spec),
            "Precompiled" => PrecompiledCharsMap.SpanStep(spec),
            _ => throw new NotSupportedException($"Tokenizer normalizer '{type}' is not supported; add it with TokenizerComponents.RegisterNormalizer."),
        };
    }

    // The library's own pre-tokenizer of the spec's type.
    public static Action<PreTokenizedText, bool> BuiltInPreTokenizer(JsonObject spec)
    {
        string type = (string)spec["type"]!;
        switch (type)
        {
            case "Sequence":
                return new SequencePreTokenizer([.. spec["pretokenizers"]!.AsArray().OfType<JsonObject>().Select(PreTokenizer)]).Run;
            case "Split":
                var (pattern, behavior, invert) = BpeTokenizer.SplitSettings(spec);
                return new SplitPreTokenizer(pattern, behavior, invert).Run;
            case "ByteLevel":
                return new ByteLevelPreTokenizer((bool?)spec["add_prefix_space"] ?? false, (bool?)spec["use_regex"] ?? true).Run;
            case "Metaspace":
                var (replacement, scheme, split) = BpeTokenizer.MetaspaceSettings(spec);
                return new MetaspacePreTokenizer(replacement, scheme, split).Run;
            case "Digits":
                return new SplitPreTokenizer(BpeTokenizer.DigitsPattern(spec), "Isolated", false).Run;
            case "Whitespace":
                return new SplitPreTokenizer(new Regex(BpeTokenizer.WordsPattern, BpeTokenizer.Fast), "Removed", true).Run;
            case "WhitespaceSplit":
                return new SplitPreTokenizer(new Regex(BpeTokenizer.NonWhitespacePattern, BpeTokenizer.Fast), "Removed", true).Run;
            case "Punctuation":
                var (punctuation, punctuationBehavior) = BpeTokenizer.PunctuationSettings(spec);
                return new SplitPreTokenizer(punctuation, punctuationBehavior, false).Run;
            case "BertPreTokenizer":
                return BertComponents.PreTokenizer();
            default:
                throw new NotSupportedException($"Tokenizer pre-tokenizer '{type}' is not supported; add it with TokenizerComponents.RegisterPreTokenizer.");
        }
    }

    private sealed class SequenceNormalizer(Action<NormalizedText>[] steps)
    {
        public void Run(NormalizedText text)
        {
            foreach (var step in steps)
            {
                step(text);
            }
        }
    }

    private sealed class SequencePreTokenizer(Action<PreTokenizedText, bool>[] stages)
    {
        public void Run(PreTokenizedText text, bool atStart)
        {
            foreach (var stage in stages)
            {
                stage(text, atStart);
            }
        }
    }

    // Unicode normalization keeping each character's origin. Text made only of characters no context can change under the
    // form (UnicodeStability) is left as it is, with no call into the platform's normalization (which allocates on each
    // call). Otherwise the whole text is normalized once; when that changes it, each cluster (a character with the marks
    // after it, Hangul jamo together) that changes is normalized on its own, its characters aligned to the source as
    // tokenizers aligns them (AppendAligned), and the joined clusters are checked against the whole result. Where they differ (a composition across clusters) the whole result is
    // taken, its changed middle standing for the old middle.
    private sealed class UnicodeNormalizer(NormalizationForm form)
    {
        // The decomposition a form starts with: canonical for NFC and NFD, compatibility for NFKC and NFKD.
        private readonly NormalizationForm _decomposition = form is NormalizationForm.FormKC or NormalizationForm.FormKD
            ? NormalizationForm.FormKD : NormalizationForm.FormD;

        private readonly bool _composes = form is NormalizationForm.FormC or NormalizationForm.FormKC;

        public void Run(NormalizedText text)
        {
            var span = text.Span;
            if (System.Text.Ascii.IsValid(span) || UnicodeStability.AllStable(span, form))
            {
                return;
            }

            char[] whole = ArrayPool<char>.Shared.Rent(span.Length * 4 + 16);
            char[] cluster = ArrayPool<char>.Shared.Rent(64);
            try
            {
                int written = Normalize(span, ref whole, form);
                if (span.SequenceEqual(whole.AsSpan(0, written)))
                {
                    return;                                                        // marks already in order and composed
                }

                var writer = text.Rewrite();
                for (int start = 0; start < span.Length;)
                {
                    int end = ClusterEnd(span, start);
                    var characters = span[start..end];
                    if (UnicodeStability.AllStable(characters, form))
                    {
                        writer.Copy(start, end - start);
                    }
                    else
                    {
                        int count = Normalize(characters, ref cluster, form);
                        if (characters.SequenceEqual(cluster.AsSpan(0, count)))
                        {
                            writer.Copy(start, end - start);
                        }
                        else if (!AppendAligned(writer, span, start, end, cluster.AsSpan(0, count)))
                        {
                            writer.Append(cluster.AsSpan(0, count), start, end - start);
                        }
                    }

                    start = end;
                }

                if (text.Pending.SequenceEqual(whole.AsSpan(0, written)))
                {
                    writer.Commit();
                }
                else
                {
                    text.Set(whole.AsSpan(0, written));
                }
            }
            finally
            {
                ArrayPool<char>.Shared.Return(whole);
                ArrayPool<char>.Shared.Return(cluster);
            }
        }

        // Normalizes `source` into `buffer` (a pooled array, replaced by a larger one when needed); returns the length. One
        // platform call when the buffer is large enough, which it nearly always is.
        private static int Normalize(ReadOnlySpan<char> source, ref char[] buffer, NormalizationForm form)
        {
            if (source.TryNormalize(buffer, out int written, form))
            {
                return written;
            }

            int needed = source.GetNormalizedLength(form);
            ArrayPool<char>.Shared.Return(buffer);
            buffer = ArrayPool<char>.Shared.Rent(needed);
            return source.TryNormalize(buffer, out written, form) ? written
                : throw new InvalidOperationException("Unicode normalization needed more room than it reported.");
        }

        // Appends a changed cluster's normalization aligned as tokenizers aligns it (unicode_normalization_alignments, then
        // NormalizedString.transform): each source character decomposes alone, its first character kept and the rest added;
        // canonical reordering moves them; a character composed of k parts removes k - 1. Walking the output in order, a kept
        // character stands for the next source character (a composed one also passes over those it removed), an added one
        // for the source character before it (nothing at the text's start). False, with nothing appended, when the output
        // cannot be matched to the parts; the cluster then stands for each output character.
        private bool AppendAligned(NormalizedText.Rewriter writer, ReadOnlySpan<char> span, int start, int end, ReadOnlySpan<char> output)
        {
            int length = end - start;
            int[] parts = ArrayPool<int>.Shared.Rent(length * 18 + 4);       // a character decomposes to at most 18 (U+FDFA)
            int[] added = ArrayPool<int>.Shared.Rent(length * 18 + 4);
            int[] sources = ArrayPool<int>.Shared.Rent(length + 1);
            int[] plan = ArrayPool<int>.Shared.Rent(output.Length + 1);
            char[] buffer = ArrayPool<char>.Shared.Rent(32);
            try
            {
                int partCount = 0, sourceCount = 0;
                for (int at = start; at < end;)
                {
                    UnicodeStability.CodePoint(span, at, out int width);
                    sources[sourceCount++] = at;
                    int written = Normalize(span.Slice(at, width), ref buffer, _decomposition);
                    for (int i = 0; i < written;)
                    {
                        if (partCount == parts.Length)
                        {
                            return false;
                        }

                        parts[partCount] = UnicodeStability.CodePoint(buffer.AsSpan(0, written), i, out int w);
                        added[partCount++] = i == 0 ? 0 : 1;
                        i += w;
                    }

                    at += width;
                }

                sources[sourceCount] = end;
                int cursor = 0, planned = 0, used = 0;
                for (int at = 0; at < output.Length;)
                {
                    UnicodeStability.CodePoint(output, at, out int width);
                    int written = _composes && !(width == 1 && output[at] < 0x80) ? Normalize(output.Slice(at, width), ref buffer, NormalizationForm.FormD) : 0;
                    ReadOnlySpan<char> pieces = written > 0 ? buffer.AsSpan(0, written) : output.Slice(at, width);
                    int change = 0, k = 0;
                    for (int i = 0; i < pieces.Length; k++)
                    {
                        int point = UnicodeStability.CodePoint(pieces, i, out int w);
                        int match = Array.IndexOf(parts, point, 0, partCount);
                        if (match < 0)
                        {
                            return false;
                        }

                        parts[match] = -1;                                         // used: equal parts are matched in order
                        change += added[match];
                        used++;
                        i += w;
                    }

                    change -= k - 1;
                    int source = change > 0 ? cursor - 1 : cursor;
                    if (change <= 0)
                    {
                        cursor += 1 - change;
                    }

                    if (cursor > sourceCount || source >= sourceCount)
                    {
                        return false;
                    }

                    plan[planned++] = source;
                    at += width;
                }

                if (cursor != sourceCount || used != partCount)
                {
                    return false;
                }

                for (int at = 0, j = 0; at < output.Length; j++)
                {
                    UnicodeStability.CodePoint(output, at, out int width);
                    var chars = output.Slice(at, width);
                    int source = plan[j];
                    if (source >= 0)
                    {
                        writer.Append(chars, sources[source], sources[source + 1] - sources[source]);
                    }
                    else if (start > 0)
                    {
                        int from = start - (start > 1 && char.IsSurrogatePair(span[start - 2], span[start - 1]) ? 2 : 1);
                        writer.Append(chars, from, start - from);
                    }
                    else
                    {
                        foreach (char c in chars)
                        {
                            writer.AppendOrigin(c, 0, 0);
                        }
                    }

                    at += width;
                }

                return true;
            }
            finally
            {
                ArrayPool<int>.Shared.Return(parts);
                ArrayPool<int>.Shared.Return(added);
                ArrayPool<int>.Shared.Return(sources);
                ArrayPool<int>.Shared.Return(plan);
                ArrayPool<char>.Shared.Return(buffer);
            }
        }

        // Where the cluster that starts at `start` ends: the character, then every mark after it and every Hangul vowel or
        // final jamo (which compose with what precedes them). Each precomposed syllable is its own cluster, so its
        // decomposed jamo stand for it alone.
        private static int ClusterEnd(ReadOnlySpan<char> text, int start)
        {
            UnicodeStability.CodePoint(text, start, out int width);
            int end = start + width;
            while (end < text.Length)
            {
                int next = UnicodeStability.CodePoint(text, end, out int nextWidth);
                if (!UnicodeStability.IsMark(next) && next is not ((>= 0x1160 and <= 0x11FF) or (>= 0xD7B0 and <= 0xD7FF)))
                {
                    break;
                }

                end += nextWidth;
            }

            return end;
        }
    }

    // Which characters no context can change under a normalization form: their own normalization is themselves, they are
    // not marks (which reorder and compose with what precedes them), and not Hangul vowel or final jamo (which compose with
    // the jamo before them). A text of such characters is already normalized. One bit per code point, computed a block of
    // 256 at a time the first time a block is met (two threads may compute the same block; they compute the same bits).
    private static class UnicodeStability
    {
        private static readonly ulong[]?[][] Blocks = [new ulong[]?[0x1100], new ulong[]?[0x1100], new ulong[]?[0x1100], new ulong[]?[0x1100]];

        public static bool AllStable(ReadOnlySpan<char> text, NormalizationForm form)
        {
            var table = Blocks[Index(form)];
            for (int i = 0; i < text.Length;)
            {
                if (text[i] < 0x80)
                {
                    i++;                                                           // ASCII is stable in every form
                    continue;
                }

                if (char.IsSurrogate(text[i]) && !(char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1])))
                {
                    return false;                                                  // a lone surrogate: let normalization refuse it
                }

                int c = CodePoint(text, i, out int width);
                var block = Volatile.Read(ref table[c >> 8]) ?? Build(table, c >> 8, form);
                if ((block[(c & 0xFF) >> 6] & (1UL << (c & 63))) == 0)
                {
                    return false;
                }

                i += width;
            }

            return true;
        }

        public static int CodePoint(ReadOnlySpan<char> text, int at, out int width)
        {
            if (char.IsHighSurrogate(text[at]) && at + 1 < text.Length && char.IsLowSurrogate(text[at + 1]))
            {
                width = 2;
                return char.ConvertToUtf32(text[at], text[at + 1]);
            }

            width = 1;
            return text[at];
        }

        public static bool IsMark(int c) => CharUnicodeInfo.GetUnicodeCategory(c)
            is UnicodeCategory.NonSpacingMark or UnicodeCategory.SpacingCombiningMark or UnicodeCategory.EnclosingMark;

        private static int Index(NormalizationForm form) => form switch
        {
            NormalizationForm.FormC => 0,
            NormalizationForm.FormD => 1,
            NormalizationForm.FormKC => 2,
            _ => 3,
        };

        private static ulong[] Build(ulong[]?[] table, int block, NormalizationForm form)
        {
            var bits = new ulong[4];
            for (int k = 0; k < 256; k++)
            {
                if (Stable((block << 8) | k, form))
                {
                    bits[k >> 6] |= 1UL << (k & 63);
                }
            }

            Volatile.Write(ref table[block], bits);
            return bits;
        }

        private static bool Stable(int c, NormalizationForm form)
        {
            if (c is (>= 0xD800 and <= 0xDFFF) or (>= 0x1160 and <= 0x11FF) or (>= 0xD7B0 and <= 0xD7FF)
                || IsMark(c) || CharUnicodeInfo.GetUnicodeCategory(c) == UnicodeCategory.OtherNotAssigned)
            {
                return false;
            }

            string one = char.ConvertFromUtf32(c);
            try
            {
                return string.Equals(one.Normalize(form), one, StringComparison.Ordinal);
            }
            catch (ArgumentException)
            {
                return false;
            }
        }
    }

    // Prepend: the string goes before the text, standing for its first character; an empty text stays empty (as the string
    // form and Hugging Face's do).
    // The UTF-16 width of a text's first character (0 for none): what a prefix stands for, as tokenizers aligns it.
    internal static int FirstWidth(ReadOnlySpan<char> text) =>
        text.IsEmpty ? 0 : text.Length > 1 && char.IsSurrogatePair(text[0], text[1]) ? 2 : 1;

    private sealed class PrependNormalizer(string prepend)
    {
        public void Run(NormalizedText text)
        {
            int length = text.Length;
            if (length == 0)
            {
                return;
            }

            var writer = text.Rewrite();
            writer.Append(prepend, 0, FirstWidth(text.Span));                  // stands for the first character (a surrogate pair is one)
            writer.Copy(0, length);
            writer.Commit();
        }
    }

    // Replace of a literal: every occurrence, left to right, as string.Replace (ordinal); the content stands for the occurrence.
    private sealed class LiteralReplaceNormalizer(string pattern, string content)
    {
        public void Run(NormalizedText text)
        {
            ArgumentException.ThrowIfNullOrEmpty(pattern);                          // as string.Replace refuses one
            var span = text.Span;
            int at = span.IndexOf(pattern, StringComparison.Ordinal);
            if (at < 0)
            {
                return;
            }

            var writer = text.Rewrite();
            int last = 0;
            while (at >= 0)
            {
                writer.Copy(last, at - last);
                writer.Append(content, at, pattern.Length);
                last = at + pattern.Length;
                int next = span[last..].IndexOf(pattern, StringComparison.Ordinal);
                at = next < 0 ? -1 : last + next;
            }

            writer.Copy(last, span.Length - last);
            writer.Commit();
        }
    }

    // Replace of a regular expression: every match (empty ones too, as Regex.Replace); a content with substitutions ($1)
    // runs through Regex.Replace itself.
    private sealed class RegexReplaceNormalizer(Regex pattern, string content)
    {
        private readonly bool _substitutes = content.Contains('$', StringComparison.Ordinal);

        public void Run(NormalizedText text)
        {
            if (_substitutes)
            {
                string before = text.ToString(), after = pattern.Replace(before, content);
                if (!string.Equals(before, after, StringComparison.Ordinal))
                {
                    text.Set(after);
                }

                return;
            }

            var span = text.Span;
            var matches = pattern.EnumerateMatches(span);
            if (!matches.MoveNext())
            {
                return;
            }

            var writer = text.Rewrite();
            int last = 0;
            do
            {
                var match = matches.Current;
                writer.Copy(last, match.Index - last);
                writer.Append(content, match.Index, match.Length);
                last = match.Index + match.Length;
            }
            while (matches.MoveNext());

            writer.Copy(last, span.Length - last);
            writer.Commit();
        }
    }

    // Lowercase as Rust's char::to_lowercase: the invariant mapping character by character, each standing for itself,
    // except U+0130 (İ), which becomes i and a combining dot above, both standing for it.
    private sealed class LowercaseNormalizer
    {
        public void Run(NormalizedText text)
        {
            var span = text.Span;
            char[] lower = ArrayPool<char>.Shared.Rent(Math.Max(1, span.Length));
            try
            {
                int count = span.ToLowerInvariant(lower);
                if (span.SequenceEqual(lower.AsSpan(0, count)) && !span.Contains('\u0130'))
                {
                    return;
                }

                var writer = text.Rewrite();
                for (int i = 0; i < count; i++)
                {
                    if (lower[i] == '\u0130')
                    {
                        writer.Append('i', i);
                        writer.Append('\u0307', i);
                        continue;
                    }

                    writer.Append(lower[i], i);
                }

                writer.Commit();
            }
            finally
            {
                ArrayPool<char>.Shared.Return(lower);
            }
        }
    }

    // Strip: whitespace off either end (char.IsWhiteSpace, as string.Trim, the set Rust's char::is_whitespace has).
    private sealed class StripNormalizer(bool left, bool right)
    {
        public void Run(NormalizedText text)
        {
            var span = text.Span;
            int start = 0, end = span.Length;
            while (left && start < end && char.IsWhiteSpace(span[start]))
            {
                start++;
            }

            while (right && end > start && char.IsWhiteSpace(span[end - 1]))
            {
                end--;
            }

            if (start == 0 && end == span.Length)
            {
                return;
            }

            var writer = text.Rewrite();
            writer.Copy(start, end - start);
            writer.Commit();
        }
    }

    // A split of every piece by a pattern (Split, Digits, Whitespace, WhitespaceSplit, Punctuation): ranges of the same text.
    private sealed class SplitPreTokenizer(Regex pattern, string behavior, bool invert)
    {
        public void Run(PreTokenizedText text, bool atStart)
        {
            if (text.Pieces.Count > 0 && behavior is not ("Isolated" or "Removed" or "MergedWithPrevious" or "MergedWithNext"))
            {
                throw new NotSupportedException($"Split behavior '{behavior}' is not supported.");
            }

            var output = text.Scratch();
            var span = text.Text.Span;
            foreach (var piece in text.Pieces)
            {
                BpeTokenizer.SplitRange(span, piece, pattern, behavior, invert, output);
            }

            text.TakeScratch();
        }
    }

    // Metaspace: spaces become the replacement, a piece gets one in front by the prepend scheme, and (split) a piece is
    // cut before every replacement. The replacement for a space stands for the space; one put in front stands for the
    // piece's first character.
    private sealed class MetaspacePreTokenizer(string replacement, string scheme, bool split)
    {
        public void Run(PreTokenizedText text, bool atStart)
        {
            var normalized = text.Text;
            var span = normalized.Span;
            var output = text.Scratch();
            var writer = normalized.Rewrite();
            for (int i = 0; i < text.Pieces.Count; i++)
            {
                var (start, length) = text.Pieces[i];
                var piece = span.Slice(start, length);
                bool begins = (piece.Length > 0 && piece[0] == ' ') || piece.StartsWith(replacement, StringComparison.Ordinal);
                int from = writer.Position;
                if ((scheme == "always" || (scheme == "first" && i == 0 && atStart)) && !begins)
                {
                    writer.Append(replacement, start, FirstWidth(piece));
                }

                int run = start;
                for (int k = start; k < start + length; k++)
                {
                    if (span[k] == ' ')
                    {
                        writer.Copy(run, k - run);
                        writer.Append(replacement, k, 1);
                        run = k + 1;
                    }
                }

                writer.Copy(run, start + length - run);
                int count = writer.Position - from;
                if (!split)
                {
                    output.Add((from, count));
                    continue;
                }

                var written = normalized.Pending.Slice(from, count);
                int cut = 0;
                for (int j = 1; j <= count; j++)
                {
                    if (j == count || written[j..].StartsWith(replacement, StringComparison.Ordinal))
                    {
                        output.Add((from + cut, j - cut));
                        cut = j;
                    }
                }
            }

            writer.Commit();
            text.TakeScratch();
        }
    }

    // ByteLevel: each piece (with a space in front, add_prefix_space, unless it has one) cut by GPT-2's rule (use_regex),
    // then written as the byte map's characters of its UTF-8 bytes; every byte's character stands for the character it
    // is a byte of, the added space for the piece's first character.
    private sealed class ByteLevelPreTokenizer(bool prefix, bool useRegex)
    {
        private readonly Regex _gpt2 = new(BpeTokenizer.Gpt2Pattern, BpeTokenizer.Fast);

        public void Run(PreTokenizedText text, bool atStart)
        {
            var normalized = text.Text;
            var span = normalized.Span;
            var starts = normalized.Starts;
            var ends = normalized.Ends;
            var output = text.Scratch();
            int longest = 1;
            foreach (var (_, length) in text.Pieces)
            {
                longest = Math.Max(longest, length + 1);
            }

            // A piece as the pre-tokenizer sees it (the prefix space included), with each character's origin.
            char[] chars = ArrayPool<char>.Shared.Rent(longest);
            int[] from = ArrayPool<int>.Shared.Rent(longest);
            int[] to = ArrayPool<int>.Shared.Rent(longest);
            try
            {
                var writer = normalized.Rewrite();
                foreach (var (start, length) in text.Pieces)
                {
                    int count = 0;
                    if (prefix && (length == 0 || span[start] != ' '))
                    {
                        var (s, e) = normalized.OriginOf(start, FirstWidth(span.Slice(start, length)));
                        (chars[0], from[0], to[0]) = (' ', s, e);
                        count = 1;
                    }

                    span.Slice(start, length).CopyTo(chars.AsSpan(count));
                    starts.Slice(start, length).CopyTo(from.AsSpan(count));
                    ends.Slice(start, length).CopyTo(to.AsSpan(count));
                    count += length;
                    var piece = chars.AsSpan(0, count);
                    if (!useRegex)
                    {
                        output.Add(Bytes(writer, piece, from, to, 0, count));
                        continue;
                    }

                    foreach (var match in _gpt2.EnumerateMatches(piece))
                    {
                        output.Add(Bytes(writer, piece, from, to, match.Index, match.Length));
                    }
                }

                writer.Commit();
                text.TakeScratch();
            }
            finally
            {
                ArrayPool<char>.Shared.Return(chars);
                ArrayPool<int>.Shared.Return(from);
                ArrayPool<int>.Shared.Return(to);
            }
        }

        // Writes piece[at..at + length] as byte characters; returns the new piece's range.
        private static (int Start, int Length) Bytes(NormalizedText.Rewriter writer, ReadOnlySpan<char> piece, int[] from, int[] to, int at, int length)
        {
            int begin = writer.Position;
            Span<byte> utf8 = stackalloc byte[4];
            for (int k = at; k < at + length;)
            {
                Rune.DecodeFromUtf16(piece[k..(at + length)], out var rune, out int consumed);   // a lone surrogate is U+FFFD, as UTF-8 writes it
                int bytes = rune.EncodeToUtf8(utf8);
                int origin = from[k], end = to[k + consumed - 1];
                for (int b = 0; b < bytes; b++)
                {
                    writer.AppendOrigin(BpeTokenizer.ByteToChar[utf8[b]], origin, end);
                }

                k += consumed;
            }

            return (begin, writer.Position - begin);
        }
    }
}

/// <summary>The span forms of the component contracts' defaults, for components that give only strings.</summary>
internal static class TokenizerPipelineDefaults
{
    // A pre-tokenizer's string form on a PreTokenizedText: each result piece is looked for, in order, in the pieces it was
    // given; found, it is a range of the text (its characters keep their origins). When one is not found (the component
    // rewrote it), the text is rebuilt from the result pieces, each standing for the piece it came from as a whole.
    public static void PreTokenize(IPreTokenizer component, PreTokenizedText text, bool atStart)
    {
        var inputs = text.PieceStrings();
        var results = component.PreTokenize(inputs, atStart);
        var span = text.Text.Span;
        var output = text.Scratch();
        int piece = 0, cursor = text.Pieces.Count > 0 ? text.Pieces[0].Start : 0;
        bool found = true;
        foreach (string result in results)
        {
            found = false;
            while (piece < text.Pieces.Count)
            {
                var (start, length) = text.Pieces[piece];
                int from = Math.Max(cursor, start);
                int at = span[from..(start + length)].IndexOf(result, StringComparison.Ordinal);
                if (at >= 0)
                {
                    output.Add((from + at, result.Length));
                    cursor = from + at + result.Length;
                    found = true;
                    break;
                }

                piece++;
                cursor = piece < text.Pieces.Count ? text.Pieces[piece].Start : cursor;
            }

            if (!found)
            {
                break;
            }
        }

        if (found)
        {
            text.TakeScratch();
            return;
        }

        // Rewritten: result piece i stands for the input piece the results have reached by length, in order.
        output.Clear();
        var normalized = text.Text;
        var writer = normalized.Rewrite();
        long resultTotal = Math.Max(1, results.Sum(r => (long)r.Length));
        long done = 0;
        foreach (string result in results)
        {
            int index = text.Pieces.Count == 0 ? -1 : (int)Math.Min(text.Pieces.Count - 1, done * text.Pieces.Count / resultTotal);
            var (s, e) = index < 0 ? (normalized.OriginalStart, normalized.OriginalStart) : normalized.OriginOf(text.Pieces[index].Start, text.Pieces[index].Length);
            int begin = writer.Position;
            foreach (char c in result)
            {
                writer.AppendOrigin(c, s, e);
            }

            output.Add((begin, result.Length));
            done += result.Length;
        }

        writer.Commit();
        text.TakeScratch();
    }
}
