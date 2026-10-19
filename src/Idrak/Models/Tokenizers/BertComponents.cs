// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Idrak.Abstraction.Text;
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

    /// <summary>
    /// BertNormalizer then BertPreTokenizer (the library's own, as a tokenizer.json names them: a BERT tokenizer's usual
    /// pipeline) in one pass over the text, for text whose every character the steps treat alone: kept as one character
    /// (lowercased), dropped (clean_text's controls, strip_accents' non-spacing marks), whitespace, or a piece of its own
    /// (punctuation, a CJK ideograph). It writes the normalized text, each character's range and the pieces the steps
    /// write, character for character, without their passes over the whole text, their regular expressions or a rewrite
    /// per step. A text with a character that needs the steps' context (under strip_accents, one NFD makes more than a
    /// letter and non-spacing marks of, or a mark that is not non-spacing; U+0130 under lowercase without it; a surrogate
    /// pair) is left to the steps (TryRun returns false).
    /// </summary>
    internal sealed class OnePass
    {
        // A character's kind (above bit 16) and the character it becomes (the low 16 bits).
        private const int Word = 0, Separator = 1, Isolated = 2, Spaced = 3, Removed = 4, Complex = 5;

        // Per setting (clean, chinese, strip, lowercase as 4 bits): per 256 UTF-16 code units, each one's kind and
        // character, made when the block is first met and shared by every tokenizer with those settings.
        private static readonly int[]?[]?[] Tables = new int[]?[]?[16];

        // The pre-tokenizer's own patterns (WhitespaceSplit's \S+ cuts at \s; Punctuation's class), asked of one character
        // at a time when a block is made, so a character's kind is what the patterns say.
        private static readonly Regex Space = new(@"\s", RegexOptions.CultureInvariant);
        private static readonly Regex Punctuation = new(BpeTokenizer.PunctuationPattern, RegexOptions.CultureInvariant);

        private readonly bool _clean, _chinese, _strip, _lowercase;
        private readonly int[]?[] _blocks;

        private OnePass(bool clean, bool chinese, bool strip, bool lowercase)
        {
            (_clean, _chinese, _strip, _lowercase) = (clean, chinese, strip, lowercase);
            int setting = (clean ? 1 : 0) | (chinese ? 2 : 0) | (strip ? 4 : 0) | (lowercase ? 8 : 0);
            var blocks = Volatile.Read(ref Tables[setting]);
            if (blocks is null)
            {
                blocks = new int[]?[256];
                blocks = Interlocked.CompareExchange(ref Tables[setting], blocks, null) ?? blocks;
            }

            _blocks = blocks;
        }

        /// <summary>
        /// The one pass for <paramref name="json"/>'s normalizer (none, or BertNormalizer) and pre-tokenizer (BertPreTokenizer),
        /// or null for any other pipeline, or when an app registered its own type for one of them (or for the
        /// pre-tokenizer's parts, WhitespaceSplit and Punctuation).
        /// </summary>
        public static OnePass? Of(JsonObject json)
        {
            if (json["pre_tokenizer"] is not JsonObject pre || (string?)pre["type"] != "BertPreTokenizer"
                || TokenizerComponents.FindPreTokenizer("BertPreTokenizer") is not null || TokenizerComponents.FindPreTokenizer("WhitespaceSplit") is not null
                || TokenizerComponents.FindPreTokenizer("Punctuation") is not null)
            {
                return null;
            }

            switch (json["normalizer"])
            {
                case null:
                    return new OnePass(false, false, false, false);
                case JsonObject spec when (string?)spec["type"] == "BertNormalizer" && TokenizerComponents.FindNormalizer("BertNormalizer") is null:
                    bool lowercase = (bool?)spec["lowercase"] ?? true;
                    return new OnePass((bool?)spec["clean_text"] ?? true, (bool?)spec["handle_chinese_chars"] ?? true, (bool?)spec["strip_accents"] ?? lowercase, lowercase);
                default:
                    return null;
            }
        }

        /// <summary>Normalizes and pre-tokenizes <paramref name="text"/>[<paramref name="start"/>..<paramref name="start"/> + <paramref name="length"/>] into <paramref name="output"/>; false (output to be rewritten) when the text needs the steps.</summary>
        public bool TryRun(string text, int start, int length, PreTokenizedText output)
        {
            output.Text.StartDirect(text, start, length, _chinese ? length * 3 : length);
            return Split(text.AsSpan(start, length), start, output);
        }

        /// <summary><see cref="TryRun(string, int, int, PreTokenizedText)"/> on a caller's buffer (as TokenizerPipeline.Run's buffer form).</summary>
        public bool TryRun(char[] text, int textLength, int start, int length, PreTokenizedText output)
        {
            output.Text.StartDirect(text, textLength, start, length, _chinese ? length * 3 : length);
            return Split(text.AsSpan(start, length), start, output);
        }

        // The pass: every character's kind from its block, its normalized characters written with its range (the steps
        // give each kept character, each space clean_text writes for whitespace and both spaces around an ideograph, the
        // range of the character it came from), the pieces cut at whitespace and around isolated characters.
        private bool Split(ReadOnlySpan<char> segment, int offset, PreTokenizedText output)
        {
            var normalized = output.Text;
            var chars = normalized.DirectChars;
            var starts = normalized.DirectStarts;
            var ends = normalized.DirectEnds;
            var pieces = output.Pieces;
            pieces.Clear();
            int n = 0, word = -1;
            for (int i = 0; i < segment.Length; i++)
            {
                char c = segment[i];
                int info = (_blocks[c >> 8] ?? Block(c >> 8))[c & 0xFF];
                int at = offset + i;
                switch (info >>> 16)
                {
                    case Word:
                        word = word < 0 ? n : word;
                        (chars[n], starts[n], ends[n]) = ((char)info, at, at + 1);
                        n++;
                        break;
                    case Separator:
                        if (word >= 0)
                        {
                            pieces.Add((word, n - word));
                            word = -1;
                        }

                        (chars[n], starts[n], ends[n]) = ((char)info, at, at + 1);
                        n++;
                        break;
                    case Isolated:
                        if (word >= 0)
                        {
                            pieces.Add((word, n - word));
                            word = -1;
                        }

                        (chars[n], starts[n], ends[n]) = ((char)info, at, at + 1);
                        pieces.Add((n, 1));
                        n++;
                        break;
                    case Spaced:
                        if (word >= 0)
                        {
                            pieces.Add((word, n - word));
                            word = -1;
                        }

                        (chars[n], starts[n], ends[n]) = (' ', at, at + 1);
                        (chars[n + 1], starts[n + 1], ends[n + 1]) = ((char)info, at, at + 1);
                        (chars[n + 2], starts[n + 2], ends[n + 2]) = (' ', at, at + 1);
                        pieces.Add((n + 1, 1));
                        n += 3;
                        break;
                    case Removed:
                        break;
                    default:
                        return false;
                }
            }

            if (word >= 0)
            {
                pieces.Add((word, n - word));
            }

            normalized.DirectLength = n;
            return true;
        }

        // A block's kinds, made once (concurrent makers write the same values; the first array published wins).
        private int[] Block(int block)
        {
            var made = new int[256];
            for (int i = 0; i < 256; i++)
            {
                made[i] = Kind((char)((block << 8) | i));
            }

            return Interlocked.CompareExchange(ref _blocks[block], made, null) ?? made;
        }

        // A character as the steps treat it: clean_text drops it or makes whitespace a space; handle_chinese_chars puts
        // spaces around an ideograph; strip_accents (NFD, then the non-spacing marks out) keeps the first character of its
        // canonical decomposition when that is a starter (combining class 0, which NFD's reordering never moves) and the rest
        // non-spacing marks (é, a letter with hamza: the letter, which the steps align to the character it came from), drops
        // a character made only of non-spacing marks, and leaves anything else to the steps; lowercase maps what is kept as
        // the Lowercase step does (U+0130, which becomes two characters, to the steps). Then the pre-tokenizer's patterns on
        // what it became: whitespace cuts, punctuation is a piece of its own.
        private int Kind(char c)
        {
            if (char.IsSurrogate(c))
            {
                return Complex << 16;
            }

            if (_clean && Dropped(c))
            {
                return Removed << 16;
            }

            if (_clean && IsWhitespace(c))
            {
                return (Separator << 16) | ' ';
            }

            bool spaced = _chinese && IsChinese(c);
            char kept = c;
            if (_strip)
            {
                Span<int> parts = stackalloc int[UnicodeNormalization.MaxDecompositionLength(compatibility: false)];
                int count = UnicodeNormalization.Decompose(c, parts, compatibility: false);
                for (int k = 1; k < count; k++)
                {
                    if (CharUnicodeInfo.GetUnicodeCategory(parts[k]) != UnicodeCategory.NonSpacingMark)
                    {
                        return Complex << 16;
                    }
                }

                if (CharUnicodeInfo.GetUnicodeCategory(parts[0]) == UnicodeCategory.NonSpacingMark)
                {
                    return Removed << 16;
                }

                if (UnicodeNormalization.CombiningClass(parts[0]) != 0 || parts[0] > char.MaxValue)
                {
                    return Complex << 16;
                }

                kept = (char)parts[0];
            }

            char lowered = kept;
            if (_lowercase)
            {
                if (kept == '\u0130')
                {
                    return Complex << 16;
                }

                Span<char> one = [kept];
                Span<char> lower = stackalloc char[1];
                ((ReadOnlySpan<char>)one).ToLowerInvariant(lower);
                lowered = lower[0];
            }

            if (spaced)
            {
                return (Spaced << 16) | lowered;
            }

            string alone = lowered.ToString();
            return Space.IsMatch(alone) ? (Separator << 16) | lowered
                : Punctuation.IsMatch(alone) ? (Isolated << 16) | lowered
                : (Word << 16) | lowered;
        }
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
