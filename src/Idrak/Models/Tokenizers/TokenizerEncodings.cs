// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json.Nodes;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// What every model's detailed encode shares (plan 24, T3), as Hugging Face's <c>tokenizers</c> does it: truncation (before
/// the post-processor, leaving room for its special tokens), the post-processor, padding, batches spread over the
/// tokenizer's workers, and the alignment of a piece's tokens to the characters they came from.
/// </summary>
internal static class TokenizerEncodings
{
    // The settings an encode runs with: the options' own, else the tokenizer.json's truncation and padding.
    private readonly record struct Settings(int? MaxLength, string Truncation, bool TruncateLeft, string Padding, int? FixedLength, bool PadLeft,
        int? PadId, int PadTypeId, string? PadToken);

    private static Settings Resolve(TokenizerEncodeOptions options, JsonObject source)
    {
        var truncation = source["truncation"] as JsonObject;
        var padding = source["padding"] as JsonObject;
        int? max = options.MaxLength ?? (int?)truncation?["max_length"];
        string strategy = options.Truncation ?? (string?)truncation?["strategy"] switch
        {
            "OnlyFirst" => "only_first",
            "OnlySecond" => "only_second",
            _ => "longest_first",
        };
        if (strategy is not ("longest_first" or "only_first" or "only_second"))
        {
            throw new ArgumentException($"Truncation '{strategy}' is not one of longest_first, only_first, only_second.", nameof(options));
        }

        bool left = options.TruncationDirection switch
        {
            null => (string?)truncation?["direction"] == "Left",
            "right" => false,
            "left" => true,
            var direction => throw new ArgumentException($"Truncation direction '{direction}' is not one of right, left.", nameof(options)),
        };
        string pad = options.Padding ?? (padding?["strategy"] switch
        {
            JsonValue value when (string?)value == "BatchLongest" => "longest",
            JsonObject fixedLength when fixedLength["Fixed"] is not null => "max_length",
            null => "none",
            _ => "none",
        });
        if (pad is not ("none" or "longest" or "max_length"))
        {
            throw new ArgumentException($"Padding '{pad}' is not one of none, longest, max_length.", nameof(options));
        }

        int? fixedPad = options.Padding is null && padding?["strategy"] is JsonObject f ? (int?)f["Fixed"] : null;
        return new Settings(max, strategy, left, pad, fixedPad ?? max, (string?)padding?["direction"] == "Left",
            options.PadId ?? (int?)padding?["pad_id"], (int?)padding?["pad_type_id"] ?? 0, (string?)padding?["pad_token"]);
    }

    /// <summary>
    /// A model's raw encode of one sequence: its ids, offsets and tokens, as sequence <c>sequence</c>. With a limit, the
    /// text is encoded a pre-token at a time until it has <c>limit</c> tokens (from its end when <c>left</c>), and the rest
    /// is not encoded.
    /// </summary>
    public delegate TokenizerEncoding RawEncoder(string text, int sequence, int limit, bool left);

    // One text (and its pair) encoded raw by the model, finished: truncated, post-processed, padded to a fixed length. As
    // tokenizers (0.23) does, a truncated sequence is encoded only up to the pre-token that reaches the maximum length
    // (all of the first text under only_second), and longest_first compares those lengths.
    public static TokenizerEncoding Finish(string text, string? pair, RawEncoder raw, ITokenizerPostProcessor? post, TokenizerEncodeOptions options,
        JsonObject source, IPretrainedTokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(options);
        var settings = Resolve(options, source);
        int Limit(int sequence) => settings.MaxLength is int m && (settings.Truncation != "only_second" || sequence != 0) ? m : int.MaxValue;
        var first = raw(text, 0, Limit(0), settings.TruncateLeft);
        var second = pair is null ? null : raw(pair, 1, Limit(1), settings.TruncateLeft);
        if (settings.MaxLength is int max)
        {
            int added = options.AddSpecialTokens ? post?.AddedTokens(second is not null) ?? 0 : 0;
            (first, second) = Truncate(first, second, Math.Max(0, max - added), settings);
        }

        var result = post is null ? PostProcessors.Plain(first, second) : post.Process(first, second, options.AddSpecialTokens);
        return settings.Padding == "max_length"
            ? Pad(result, settings.FixedLength ?? throw new ArgumentException("Padding to max_length needs MaxLength.", nameof(options)), settings, tokenizer)
            : result;
    }

    // A batch: each text finished on the tokenizer's workers, in input order; "longest" pads all to the longest.
    public static IReadOnlyList<TokenizerEncoding> Batch(IReadOnlyList<string> texts, IReadOnlyList<string>? pairs, TokenizerEncodeOptions options,
        JsonObject source, IPretrainedTokenizer tokenizer, int workers)
    {
        ArgumentNullException.ThrowIfNull(texts);
        ArgumentNullException.ThrowIfNull(options);
        if (pairs is not null && pairs.Count != texts.Count)
        {
            throw new ArgumentException($"{pairs.Count} pairs for {texts.Count} texts.", nameof(pairs));
        }

        var results = new TokenizerEncoding[texts.Count];
        Parallel.For(0, texts.Count, new ParallelOptions { MaxDegreeOfParallelism = workers <= 0 ? -1 : workers },
            i => results[i] = tokenizer.Encode(texts[i], pairs?[i], options));
        var settings = Resolve(options, source);
        if (settings.Padding == "longest" && results.Length > 0)
        {
            int longest = results.Max(r => r.Count);
            for (int i = 0; i < results.Length; i++)
            {
                results[i] = Pad(results[i], longest, settings, tokenizer);
            }
        }

        return results;
    }

    // tokenizers' truncation (utils/truncation.rs). longest_first: a single text to the length; of a pair, the shorter
    // keeps its length when the longer can take the rest, else both get half (the longer the odd token). only_first and
    // only_second: that text loses what is too many (a single text loses it under either), refused when it is too short.
    private static (TokenizerEncoding First, TokenizerEncoding? Second) Truncate(TokenizerEncoding first, TokenizerEncoding? second, int max, Settings settings)
    {
        int total = first.Count + (second?.Count ?? 0);
        if (total <= max)
        {
            return (first, second);
        }

        if (settings.Truncation == "longest_first")
        {
            if (second is null)
            {
                return (Slice(first, max, settings.TruncateLeft), null);
            }

            int n1 = first.Count, n2 = second.Count;
            bool swap = n1 > n2;
            if (swap)
            {
                (n1, n2) = (n2, n1);
            }

            n2 = n1 > max ? n1 : Math.Max(n1, max - n1);
            if (n1 + n2 > max)
            {
                n1 = max / 2;
                n2 = n1 + max % 2;
            }

            if (swap)
            {
                (n1, n2) = (n2, n1);
            }

            return (Slice(first, n1, settings.TruncateLeft), Slice(second, n2, settings.TruncateLeft));
        }

        int remove = total - max;
        bool onSecond = settings.Truncation == "only_second" && second is not null;
        var target = onSecond ? second! : first;
        if (target.Count <= remove)
        {
            throw new InvalidOperationException($"Truncation {settings.Truncation} cannot fit {total} tokens into {max}: the "
                + $"{(onSecond ? "second" : "first")} text has {target.Count} tokens, {remove} must go.");
        }

        return onSecond
            ? (first, Slice(second!, second!.Count - remove, settings.TruncateLeft))
            : (Slice(first, first.Count - remove, settings.TruncateLeft), second);
    }

    private static TokenizerEncoding Slice(TokenizerEncoding e, int keep, bool left)
    {
        if (keep >= e.Count)
        {
            return e;
        }

        int from = left ? e.Count - keep : 0;
        return new TokenizerEncoding
        {
            Ids = e.Ids[from..(from + keep)], TypeIds = e.TypeIds[from..(from + keep)], Offsets = e.Offsets[from..(from + keep)],
            AttentionMask = e.AttentionMask[from..(from + keep)], SpecialTokensMask = e.SpecialTokensMask[from..(from + keep)],
            SequenceIds = e.SequenceIds[from..(from + keep)], Tokens = e.Tokens.Length == e.Count ? e.Tokens[from..(from + keep)] : [],
        };
    }

    // Padding as tokenizers adds it: the pad id and token, the pad type id, no offsets, not attended, special.
    private static TokenizerEncoding Pad(TokenizerEncoding e, int length, Settings settings, IPretrainedTokenizer tokenizer)
    {
        int add = length - e.Count;
        if (add <= 0)
        {
            return e;
        }

        int padId = settings.PadId ?? PadIdOf(tokenizer)
            ?? throw new InvalidOperationException("Padding needs a pad id: set TokenizerEncodeOptions.PadId (the tokenizer.json names none and the tokenizer has no padding token).");
        string padToken = settings.PadToken ?? tokenizer.TokenOf(padId);
        T[] Padded<T>(T[] values, T value)
        {
            var result = new T[length];
            int at = settings.PadLeft ? add : 0;
            values.CopyTo(result, at);
            for (int i = 0; i < add; i++)
            {
                result[settings.PadLeft ? i : e.Count + i] = value;
            }

            return result;
        }

        return new TokenizerEncoding
        {
            Ids = Padded(e.Ids, padId), TypeIds = Padded(e.TypeIds, settings.PadTypeId), Offsets = Padded(e.Offsets, (0, 0)),
            AttentionMask = Padded(e.AttentionMask, 0), SpecialTokensMask = Padded(e.SpecialTokensMask, 1), SequenceIds = Padded(e.SequenceIds, -1),
            Tokens = e.Tokens.Length == e.Count ? Padded(e.Tokens, padToken) : [],
        };
    }

    // The tokenizer's padding token, by the names tokenizers use.
    private static int? PadIdOf(IPretrainedTokenizer tokenizer) =>
        tokenizer.IdOf("[PAD]") ?? tokenizer.IdOf("<pad>") ?? tokenizer.IdOf("<|endoftext|>");

    // A raw encoding: the ids from `from` on (what a left limit keeps), their offsets and tokens, one sequence whose number
    // is also its type id (as tokenizers encodes a pair's second text with type id 1), all attended, none special.
    public static TokenizerEncoding Raw(List<int> ids, List<(int Start, int End)> offsets, Func<int, string> tokenOf, int sequence, int from = 0)
    {
        var kept = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(ids)[from..];
        var tokens = new string[kept.Length];
        for (int i = 0; i < kept.Length; i++)
        {
            tokens[i] = tokenOf(kept[i]);
        }

        return new()
        {
            Ids = kept.ToArray(), Offsets = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(offsets)[from..].ToArray(), TypeIds = Filled(kept.Length, sequence),
            AttentionMask = Ones(kept.Length), SpecialTokensMask = new int[kept.Length], SequenceIds = Filled(kept.Length, sequence), Tokens = tokens,
        };
    }

    // The tokens a left limit drops, given each pre-token's first token (and whether it is an added token): tokenizers walks
    // the pre-tokens from the last, counting added tokens without stopping on them, and stops at the first other pre-token
    // that brings the count to the limit; the pre-tokens before it are not encoded.
    public static int Limited(List<(int Start, bool Added)> pieces, int count, int limit)
    {
        for (int i = pieces.Count - 1; i >= 0; i--)
        {
            if (!pieces[i].Added && count - pieces[i].Start >= limit)
            {
                return pieces[i].Start;
            }
        }

        return 0;
    }

    private static int[] Ones(int count) => Filled(count, 1);

    private static int[] Filled(int count, int value)
    {
        var values = new int[count];
        Array.Fill(values, value);
        return values;
    }

    // The ranges (in the piece's characters) of the tokens a piece encoded to, as tokenizers aligns them: a token whose text
    // comes next spans it; byte-fallback tokens (<0xNN>) of one character all span that character; an unknown token spans
    // up to where the next token's text comes (or one character when the next is unknown too).
    public static void Align(ReadOnlySpan<char> piece, ReadOnlySpan<int> ids, Func<int, string> tokenOf, bool byteFallback, List<(int Start, int Length)> ranges)
    {
        int cursor = 0, byteStart = 0, byteWidth = 0, bytesLeft = 0;
        for (int k = 0; k < ids.Length; k++)
        {
            string token = tokenOf(ids[k]);
            if (bytesLeft > 0 && IsByteToken(token))
            {
                ranges.Add((byteStart, byteWidth));
                if (--bytesLeft == 0)
                {
                    cursor = byteStart + byteWidth;
                }

                continue;
            }

            if (token.Length > 0 && piece[cursor..].StartsWith(token, StringComparison.Ordinal))
            {
                ranges.Add((cursor, token.Length));
                cursor += token.Length;
                continue;
            }

            int width = cursor < piece.Length ? Width(piece, cursor) : 0;
            if (byteFallback && IsByteToken(token) && width > 0)
            {
                (byteStart, byteWidth) = (cursor, width);
                bytesLeft = Encoding.UTF8.GetByteCount(piece.Slice(cursor, width)) - 1;
                ranges.Add((cursor, width));
                if (bytesLeft == 0)
                {
                    cursor += width;
                }

                continue;
            }

            // Unknown: up to the next token's text (a fused run of unknown characters), else one character.
            int end = cursor + width;
            if (k + 1 < ids.Length && tokenOf(ids[k + 1]) is { Length: > 0 } next && !piece[end..].StartsWith(next, StringComparison.Ordinal)
                && !(byteFallback && IsByteToken(next)))
            {
                int at = piece[end..].IndexOf(next, StringComparison.Ordinal);
                end = at < 0 ? end : end + at;                                // a next token not in the piece is unknown too
            }
            else if (k + 1 == ids.Length)
            {
                end = piece.Length;
            }

            ranges.Add((cursor, end - cursor));
            cursor = end;
        }
    }

    // Unigram's byte fallback as tokenizers does it: a run of unknown characters is fused into one unknown token first, so
    // every byte token of the run spans the whole run.
    public static void FuseByteRuns(ReadOnlySpan<int> ids, Func<int, string> tokenOf, List<(int Start, int Length)> ranges)
    {
        for (int k = 0; k < ids.Length;)
        {
            if (!IsByteToken(tokenOf(ids[k])))
            {
                k++;
                continue;
            }

            int end = k + 1;
            while (end < ids.Length && IsByteToken(tokenOf(ids[end])))
            {
                end++;
            }

            int from = ranges[k].Start, to = ranges[end - 1].Start + ranges[end - 1].Length;
            for (int i = k; i < end; i++)
            {
                ranges[i] = (from, to - from);
            }

            k = end;
        }
    }

    private static bool IsByteToken(string token) => token.Length == 6 && token.StartsWith("<0x", StringComparison.Ordinal) && token[5] == '>';

    private static int Width(ReadOnlySpan<char> text, int at) =>
        at + 1 < text.Length && char.IsSurrogatePair(text[at], text[at + 1]) ? 2 : 1;
}
