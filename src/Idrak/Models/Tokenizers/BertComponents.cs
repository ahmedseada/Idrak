// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// BERT's tokenizer.json components, as Hugging Face's <c>tokenizers</c> defines them (plan 24, T2): the
/// <c>BertNormalizer</c> (control characters out, whitespace to spaces, spaces around Chinese characters, accents stripped,
/// lowercase), the <c>BertPreTokenizer</c> (whitespace, then each punctuation character), and the <c>WordPiece</c> decoder.
/// Each has one implementation on spans; the string forms run it.
/// </summary>
internal static class BertComponents
{
    // BertNormalizer on spans: clean_text and handle_chinese_chars in one pass, then strip_accents (NFD without the
    // non-spacing marks), then lowercase; strip_accents null follows lowercase.
    public static Action<NormalizedText> Normalizer(JsonObject spec)
    {
        bool clean = (bool?)spec["clean_text"] ?? true;
        bool chinese = (bool?)spec["handle_chinese_chars"] ?? true;
        bool lowercase = (bool?)spec["lowercase"] ?? true;
        bool strip = (bool?)spec["strip_accents"] ?? lowercase;
        var decompose = strip ? SpanComponents.BuiltInNormalizer(new JsonObject { ["type"] = "NFD" }) : null;
        var lower = lowercase ? SpanComponents.BuiltInNormalizer(new JsonObject { ["type"] = "Lowercase" }) : null;
        return text =>
        {
            if (clean || chinese)
            {
                CleanAndSpace(text, clean, chinese);
            }

            if (decompose is not null)
            {
                decompose(text);
                RemoveNonSpacingMarks(text);
            }

            lower?.Invoke(text);
        };
    }

    // The string form (TokenizerComponents' library default, the BPE tokenizer's string pipeline): the span form on a text of its own.
    public static Func<string, string> StringNormalizer(JsonObject spec)
    {
        var normalize = Normalizer(spec);
        return s =>
        {
            using var text = new NormalizedText();
            text.Reset(s);
            normalize(text);
            return text.ToString();
        };
    }

    // clean_text drops U+0000, U+FFFD and control characters (Cc, Cf, Co, except tab, LF and CR) and makes whitespace a
    // space; handle_chinese_chars puts a space on each side of a CJK ideograph (both spaces stand for it).
    private static void CleanAndSpace(NormalizedText text, bool clean, bool chinese)
    {
        var span = text.Span;
        int first = -1;
        for (int i = 0; i < span.Length && first < 0;)
        {
            int c = CodePoint(span, i, out int width);
            if ((clean && (Dropped(c) || (c != ' ' && IsWhitespace(c)))) || (chinese && IsChinese(c)))
            {
                first = i;
            }

            i += width;
        }

        if (first < 0)
        {
            return;
        }

        var writer = text.Rewrite();
        writer.Copy(0, first);
        for (int i = first; i < span.Length;)
        {
            int c = CodePoint(span, i, out int width);
            if (clean && Dropped(c))
            {
                // dropped
            }
            else if (clean && IsWhitespace(c))
            {
                writer.Append(' ', i, width);
            }
            else if (chinese && IsChinese(c))
            {
                writer.Append(' ', i, width);
                writer.Copy(i, width);
                writer.Append(' ', i, width);
            }
            else
            {
                writer.Copy(i, width);
            }

            i += width;
        }

        writer.Commit();
    }

    private static void RemoveNonSpacingMarks(NormalizedText text)
    {
        var span = text.Span;
        int first = -1;
        for (int i = 0; i < span.Length && first < 0;)
        {
            if (CharUnicodeInfo.GetUnicodeCategory(CodePoint(span, i, out int width)) == UnicodeCategory.NonSpacingMark)
            {
                first = i;
            }

            i += width;
        }

        if (first < 0)
        {
            return;
        }

        var writer = text.Rewrite();
        writer.Copy(0, first);
        for (int i = first; i < span.Length;)
        {
            int c = CodePoint(span, i, out int width);
            if (CharUnicodeInfo.GetUnicodeCategory(c) != UnicodeCategory.NonSpacingMark)
            {
                writer.Copy(i, width);
            }

            i += width;
        }

        writer.Commit();
    }

    // U+0000, U+FFFD and the characters tokenizers counts as control: Cc, Cf and Co, but not tab, LF or CR (whitespace).
    private static bool Dropped(int c) => c is 0 or 0xFFFD || (c is not ('\t' or '\n' or '\r')
        && CharUnicodeInfo.GetUnicodeCategory(c) is UnicodeCategory.Control or UnicodeCategory.Format or UnicodeCategory.PrivateUse);

    // Rust's char::is_whitespace (Unicode White_Space), which .NET's char.IsWhiteSpace and Rune.IsWhiteSpace match.
    private static bool IsWhitespace(int c) => c is '\t' or '\n' or '\r' || (c <= 0xFFFF ? char.IsWhiteSpace((char)c) : Rune.IsWhiteSpace(new Rune(c)));

    // The CJK ideograph blocks BERT puts spaces around.
    private static bool IsChinese(int c) => c is (>= 0x4E00 and <= 0x9FFF) or (>= 0x3400 and <= 0x4DBF) or (>= 0x20000 and <= 0x2A6DF)
        or (>= 0x2A700 and <= 0x2B73F) or (>= 0x2B740 and <= 0x2B81F) or (>= 0x2B920 and <= 0x2CEAF) or (>= 0xF900 and <= 0xFAFF)
        or (>= 0x2F800 and <= 0x2FA1F);

    private static int CodePoint(ReadOnlySpan<char> text, int at, out int width)
    {
        if (char.IsHighSurrogate(text[at]) && at + 1 < text.Length && char.IsLowSurrogate(text[at + 1]))
        {
            width = 2;
            return char.ConvertToUtf32(text[at], text[at + 1]);
        }

        width = 1;
        return text[at];
    }

    // BertPreTokenizer: whitespace removed, then each punctuation character (ASCII punctuation and Unicode's P categories)
    // its own piece.
    public static Action<PreTokenizedText, bool> PreTokenizer() =>
        SpanComponents.BuiltInPreTokenizer(new JsonObject
        {
            ["type"] = "Sequence",
            ["pretokenizers"] = new JsonArray(new JsonObject { ["type"] = "WhitespaceSplit" }, new JsonObject { ["type"] = "Punctuation", ["behavior"] = "Isolated" }),
        });

    // The string form of BertPreTokenizer.
    public static Func<List<string>, bool, List<string>> StringPreTokenizer() =>
        BpeTokenizer.PreTokenizerStage(new JsonObject
        {
            ["type"] = "Sequence",
            ["pretokenizers"] = new JsonArray(new JsonObject { ["type"] = "WhitespaceSplit" }, new JsonObject { ["type"] = "Punctuation", ["behavior"] = "Isolated" }),
        });

    // The WordPiece decoder: a token after the first loses its continuation prefix, or gets a space before it; cleanup
    // then joins punctuation and English contractions as tokenizers' cleanup does, token by token.
    public static Func<List<string>, List<string>> WordPieceDecoder(JsonObject spec)
    {
        string prefix = (string?)spec["prefix"] ?? "##";
        bool cleanup = (bool?)spec["cleanup"] ?? true;
        return tokens =>
        {
            var result = new List<string>(tokens.Count);
            for (int i = 0; i < tokens.Count; i++)
            {
                string token = tokens[i];
                if (i != 0)
                {
                    token = prefix.Length > 0 && token.StartsWith(prefix, StringComparison.Ordinal) ? token[prefix.Length..] : " " + token;
                }

                result.Add(cleanup ? Cleanup(token) : token);
            }

            return result;
        };
    }

    // tokenizers' decoders::wordpiece::cleanup, in its order.
    private static string Cleanup(string text) => text
        .Replace(" .", ".", StringComparison.Ordinal).Replace(" ?", "?", StringComparison.Ordinal).Replace(" !", "!", StringComparison.Ordinal)
        .Replace(" ,", ",", StringComparison.Ordinal).Replace(" ' ", "'", StringComparison.Ordinal).Replace(" n't", "n't", StringComparison.Ordinal)
        .Replace(" 'm", "'m", StringComparison.Ordinal).Replace(" do not", " don't", StringComparison.Ordinal).Replace(" 's", "'s", StringComparison.Ordinal)
        .Replace(" 've", "'ve", StringComparison.Ordinal).Replace(" 're", "'re", StringComparison.Ordinal);

    // Lowercase as Rust's char::to_lowercase (tokenizers' Lowercase and BertNormalizer): .NET's invariant mapping, except
    // U+0130 (İ), whose lowercase is two characters, i and a combining dot above.
    public static string Lowercase(string text)
    {
        string lower = text.ToLowerInvariant();
        return lower.Contains('İ', StringComparison.Ordinal) ? lower.Replace("İ", "i̇", StringComparison.Ordinal) : lower;
    }
}
