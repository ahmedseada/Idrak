// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>How a precompiled character map is applied: the two references differ on marks after a mapped letter.</summary>
internal enum PrecompiledMode
{
    /// <summary>
    /// Hugging Face tokenizers' <c>Precompiled</c>: text is taken grapheme cluster by grapheme cluster; a cluster of fewer
    /// than 6 UTF-8 bytes that starts with a key is replaced whole by the shortest such key's string (so a mark after a
    /// mapped letter is dropped), otherwise each character is replaced by its own entry, if it has one.
    /// </summary>
    HuggingFace,

    /// <summary>
    /// SentencePiece's normalizer (normalizer.cc): at each position the longest key that is a prefix of the remaining
    /// UTF-8 bytes is replaced by its string; where none is, one character is kept as it is.
    /// </summary>
    SentencePiece,
}

/// <summary>
/// SentencePiece's precompiled character map (<c>NormalizerSpec.precompiled_charsmap</c>, and the <c>Precompiled</c>
/// normalizer of a tokenizer.json): a double-array trie over UTF-8 keys whose values are offsets into a block of
/// NUL-terminated replacement strings. Built once from the map's bytes (the trie's units and that block are the only
/// arrays it keeps); normalizing allocates nothing but the result, and returns the input string itself when nothing in
/// it maps to something else.
/// </summary>
/// <remarks>
/// A tokenizer.json names the mode with an extra field of the normalizer, <c>"mode": "sentencepiece"</c>; without it the
/// mode is Hugging Face's. Hugging Face's tokenizers reads such a file too (it ignores fields it does not know, and
/// applies its own mode), so a tokenizer saved from a SentencePiece model stays loadable there.
/// </remarks>
internal sealed class PrecompiledCharsMap
{
    /// <summary>The tokenizer.json field that names the mode.</summary>
    public const string ModeField = "mode";

    /// <summary>The mode field's value for <see cref="PrecompiledMode.SentencePiece"/>.</summary>
    public const string SentencePieceMode = "sentencepiece";

    /// <summary>The mode field's value for <see cref="PrecompiledMode.HuggingFace"/> (the default when the field is absent).</summary>
    public const string HuggingFaceMode = "huggingface";

    private readonly DoubleArrayTrie _trie;
    private readonly byte[] _normalized;

    // Each ASCII character's own entry (-1: none), looked up once: an ASCII character that is a cluster on its own is
    // replaced (or not) without a grapheme split, an encode or a trie walk.
    private readonly int[] _ascii = new int[128];

    private PrecompiledCharsMap(DoubleArrayTrie trie, byte[] normalized)
    {
        _trie = trie;
        _normalized = normalized;
        Span<byte> one = stackalloc byte[1];
        for (int c = 0; c < 128; c++)
        {
            one[0] = (byte)c;
            _ascii[c] = c == 0 ? -1 : trie.ShortestPrefix(one, out _);
        }
    }

    /// <summary>Reads a precompiled charsmap (trie size, trie units, then the NUL-terminated strings).</summary>
    /// <exception cref="InvalidDataException">The bytes are not a charsmap.</exception>
    public static PrecompiledCharsMap Read(ReadOnlySpan<byte> charsmap)
    {
        var trie = DoubleArrayTrie.ReadCharsmap(charsmap, out var normalized);
        return new PrecompiledCharsMap(trie, normalized.ToArray());
    }

    /// <summary>
    /// The normalizer a tokenizer.json's <c>Precompiled</c> object describes (its base64 <c>precompiled_charsmap</c> and
    /// optional <c>mode</c>), built once, as a function.
    /// </summary>
    /// <exception cref="InvalidDataException">The charsmap is missing or not one, or the mode is unknown.</exception>
    public static Func<string, string> Step(JsonObject spec)
    {
        var map = FromSpec(spec, out var mode);
        if (map is null)
        {
            return static s => s;                          // an empty map (the identity rule) changes nothing
        }

        return mode == PrecompiledMode.SentencePiece ? map.NormalizeSentencePiece : map.NormalizeHuggingFace;
    }

    /// <summary>
    /// The same normalizer on a <see cref="NormalizedText"/> (plan 24, L1): each replacement's characters stand for the
    /// characters (or grapheme cluster) they replace; text that does not change is left as it is, with nothing allocated.
    /// </summary>
    /// <exception cref="InvalidDataException">The charsmap is missing or not one, or the mode is unknown.</exception>
    public static Action<NormalizedText> SpanStep(JsonObject spec)
    {
        var map = FromSpec(spec, out var mode);
        if (map is null)
        {
            return static _ => { };
        }

        return mode == PrecompiledMode.SentencePiece ? map.NormalizeSentencePiece : map.NormalizeHuggingFace;
    }

    // The map a Precompiled object describes, read once (null for an empty map), and its mode.
    private static PrecompiledCharsMap? FromSpec(JsonObject spec, out PrecompiledMode mode)
    {
        mode = ModeOf(spec);
        if (spec["precompiled_charsmap"] is not JsonValue value || !value.TryGetValue(out string? text))
        {
            throw new InvalidDataException("The Precompiled normalizer has no precompiled_charsmap (a base64 string).");
        }

        byte[] bytes;
        try
        {
            bytes = Convert.FromBase64String(text);
        }
        catch (FormatException e)
        {
            throw new InvalidDataException("The Precompiled normalizer's precompiled_charsmap is not base64.", e);
        }

        return bytes.Length == 0 ? null : Read(bytes);
    }

    /// <summary>The mode a tokenizer.json's <c>Precompiled</c> object names (Hugging Face's when it names none).</summary>
    /// <exception cref="InvalidDataException">It names a mode that is not one of the two.</exception>
    public static PrecompiledMode ModeOf(JsonObject spec) => spec[ModeField] switch
    {
        null => PrecompiledMode.HuggingFace,
        JsonValue v when v.TryGetValue(out string? name) && name == SentencePieceMode => PrecompiledMode.SentencePiece,
        JsonValue v when v.TryGetValue(out string? name) && name == HuggingFaceMode => PrecompiledMode.HuggingFace,
        var other => throw new InvalidDataException(
            $"The Precompiled normalizer's mode is {other.ToJsonString()}: give \"{SentencePieceMode}\" or \"{HuggingFaceMode}\" (or leave it out)."),
    };

    /// <summary><paramref name="text"/> normalized in <paramref name="mode"/>.</summary>
    public string Normalize(string text, PrecompiledMode mode) =>
        mode == PrecompiledMode.SentencePiece ? NormalizeSentencePiece(text) : NormalizeHuggingFace(text);

    /// <summary>
    /// <paramref name="text"/> normalized as Hugging Face's tokenizers does (see <see cref="PrecompiledMode.HuggingFace"/>):
    /// grapheme clusters are .NET's extended ones (UAX #29), which tokenizers' unicode-segmentation also follows. Half of
    /// a surrogate pair is looked up as U+FFFD and kept as it is when that has no entry.
    /// </summary>
    public string NormalizeHuggingFace(string text)
    {
        var output = new Output(text);
        Span<byte> utf8 = stackalloc byte[16];
        try
        {
            int i = 0;
            while (i < text.Length)
            {
                int length = StringInfo.GetNextTextElementLength(text.AsSpan(i));
                var cluster = text.AsSpan(i, length);

                // A cluster of 6 chars or more has 6 bytes or more; a shorter one is measured.
                if (length <= 5 && Encoding.UTF8.GetBytes(cluster, utf8) is var bytes and < 6
                    && _trie.ShortestPrefix(utf8[..bytes], out _) is var whole and >= 0)
                {
                    output.Replace(i, length, ValueBytes(whole));
                    i += length;
                    continue;
                }

                for (int j = i, end = i + length; j < end;)
                {
                    Rune.DecodeFromUtf16(text.AsSpan(j, end - j), out var rune, out int used);   // a lone surrogate: U+FFFD
                    int n = rune.EncodeToUtf8(utf8);
                    if (_trie.ShortestPrefix(utf8[..n], out _) is var value and >= 0)
                    {
                        output.Replace(j, used, ValueBytes(value));
                    }

                    j += used;
                }

                i += length;
            }

            return output.Finish();
        }
        finally
        {
            output.Dispose();
        }
    }

    /// <summary>
    /// <paramref name="text"/> normalized as SentencePiece's normalizer applies its charsmap (see
    /// <see cref="PrecompiledMode.SentencePiece"/>), without its whitespace handling (a tokenizer.json writes that as
    /// separate normalizers). The text is walked as UTF-8 (half of a surrogate pair is U+FFFD there).
    /// </summary>
    public string NormalizeSentencePiece(string text)
    {
        if (text.Length == 0)
        {
            return text;
        }

        int count = Encoding.UTF8.GetByteCount(text);
        byte[] rented = ArrayPool<byte>.Shared.Rent(count);
        byte[]? result = null;
        try
        {
            var bytes = rented.AsSpan(0, count);
            Encoding.UTF8.GetBytes(text, bytes);
            int written = 0, pending = 0, at = 0;
            while (at < count)
            {
                int value = _trie.LongestPrefix(bytes[at..], out int length);
                if (value < 0)
                {
                    byte lead = bytes[at];
                    at += lead < 0x80 ? 1 : lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : 4;    // valid UTF-8: .NET wrote it
                    continue;
                }

                var replacement = ValueBytes(value);
                if (!replacement.SequenceEqual(bytes.Slice(at, length)))
                {
                    // The first change: from here the result is written out (the unchanged part before it in one copy).
                    Append(ref result, ref written, bytes[pending..at], count);
                    Append(ref result, ref written, replacement, count);
                    pending = at + length;
                }

                at += length;
            }

            if (result is null)
            {
                return text;
            }

            Append(ref result, ref written, bytes[pending..count], count);
            return Encoding.UTF8.GetString(result!, 0, written);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rented);
            if (result is not null)
            {
                ArrayPool<byte>.Shared.Return(result);
            }
        }

        static void Append(ref byte[]? buffer, ref int written, ReadOnlySpan<byte> part, int hint)
        {
            buffer ??= ArrayPool<byte>.Shared.Rent(Math.Max(hint + 16, 64));
            if (written + part.Length > buffer.Length)
            {
                var larger = ArrayPool<byte>.Shared.Rent(Math.Max(buffer.Length * 2, written + part.Length));
                buffer.AsSpan(0, written).CopyTo(larger);
                ArrayPool<byte>.Shared.Return(buffer);
                buffer = larger;
            }

            part.CopyTo(buffer.AsSpan(written));
            written += part.Length;
        }
    }

    /// <summary>
    /// <see cref="NormalizeHuggingFace(string)"/> on a <see cref="NormalizedText"/>, aligned as tokenizers aligns it: the
    /// i-th character of a replacement stands for the i-th character it replaces (the last one for those beyond).
    /// </summary>
    public void NormalizeHuggingFace(NormalizedText text)
    {
        var span = text.Span;
        var output = new SpanOutput(text) { Positional = true };
        Span<byte> utf8 = stackalloc byte[16];
        try
        {
            int i = 0;
            while (i < span.Length)
            {
                // An ASCII character followed by ASCII (or the end) is a cluster on its own, but for CR LF.
                char c = span[i];
                if (c < 0x80 && (i + 1 == span.Length || (span[i + 1] < 0x80 && !(c == '\r' && span[i + 1] == '\n'))))
                {
                    if (_ascii[c] >= 0)
                    {
                        output.Replace(i, 1, ValueBytes(_ascii[c]));
                    }

                    i++;
                    continue;
                }

                int length = StringInfo.GetNextTextElementLength(span[i..]);
                var cluster = span.Slice(i, length);
                if (length <= 5 && Encoding.UTF8.GetBytes(cluster, utf8) is var bytes and < 6
                    && _trie.ShortestPrefix(utf8[..bytes], out _) is var whole and >= 0)
                {
                    output.Replace(i, length, ValueBytes(whole));
                    i += length;
                    continue;
                }

                for (int j = i, end = i + length; j < end;)
                {
                    Rune.DecodeFromUtf16(span[j..end], out var rune, out int used);
                    int n = rune.EncodeToUtf8(utf8);
                    if (_trie.ShortestPrefix(utf8[..n], out _) is var value and >= 0)
                    {
                        output.Replace(j, used, ValueBytes(value));
                    }

                    j += used;
                }

                i += length;
            }

            output.Finish();
        }
        finally
        {
            output.Dispose();
        }
    }

    /// <summary>
    /// <see cref="NormalizeSentencePiece(string)"/> on a <see cref="NormalizedText"/>: a key's replacement stands for the
    /// characters the key's bytes encode.
    /// </summary>
    public void NormalizeSentencePiece(NormalizedText text)
    {
        var span = text.Span;
        if (span.Length == 0)
        {
            return;
        }

        int count = Encoding.UTF8.GetByteCount(span);
        byte[] rented = ArrayPool<byte>.Shared.Rent(count);
        var output = new SpanOutput(text);
        try
        {
            var bytes = rented.AsSpan(0, count);
            Encoding.UTF8.GetBytes(span, bytes);
            int at = 0, character = 0;                     // the byte and the character they begin
            while (at < count)
            {
                int value = _trie.LongestPrefix(bytes[at..], out int length);
                if (value < 0)
                {
                    byte lead = bytes[at];
                    at += lead < 0x80 ? 1 : lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : 4;
                    character += lead >= 0xF0 ? 2 : 1;     // a 4-byte character is a surrogate pair
                    continue;
                }

                int chars = 0;
                for (int k = at; k < at + length; k++)
                {
                    byte b = bytes[k];
                    chars += (b & 0xC0) == 0x80 ? 0 : b >= 0xF0 ? 2 : 1;
                }

                var replacement = ValueBytes(value);
                if (!replacement.SequenceEqual(bytes.Slice(at, length)))
                {
                    output.Replace(character, chars, replacement);
                }

                at += length;
                character += chars;
            }

            output.Finish();
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rented);
            output.Dispose();
        }
    }

    // The replacement string a trie value points at: its bytes up to the next NUL.
    private ReadOnlySpan<byte> ValueBytes(int offset)
    {
        if ((uint)offset > (uint)_normalized.Length)
        {
            throw new InvalidDataException($"Precompiled charsmap: a key maps to byte {offset} of its {_normalized.Length}-byte strings, past their end.");
        }

        var rest = _normalized.AsSpan(offset);
        int end = rest.IndexOf((byte)0);
        return end < 0 ? rest : rest[..end];
    }

    // The result of a UTF-16 normalization: nothing is written until the first replacement, so text that does not change
    // is returned as it is; then the unchanged runs and the replacements go into one pooled buffer.
    private struct Output(string text) : IDisposable
    {
        private char[]? _buffer;
        private int _written;
        private int _pending;                              // the start of the text not yet copied out

        public void Replace(int start, int length, ReadOnlySpan<byte> utf8)
        {
            Append(text.AsSpan(_pending, start - _pending));
            int chars = Encoding.UTF8.GetCharCount(utf8);
            Ensure(chars);
            _written += Encoding.UTF8.GetChars(utf8, _buffer.AsSpan(_written));
            _pending = start + length;
        }

        public string Finish()
        {
            if (_buffer is null)
            {
                return text;
            }

            Append(text.AsSpan(_pending));
            return new string(_buffer, 0, _written);
        }

        public void Dispose()
        {
            if (_buffer is not null)
            {
                ArrayPool<char>.Shared.Return(_buffer);
                _buffer = null;
            }
        }

        private void Append(ReadOnlySpan<char> part)
        {
            Ensure(part.Length);
            part.CopyTo(_buffer.AsSpan(_written));
            _written += part.Length;
        }

        private void Ensure(int more)
        {
            if (_buffer is null)
            {
                _buffer = ArrayPool<char>.Shared.Rent(Math.Max(text.Length + 16, _written + more));
            }
            else if (_written + more > _buffer.Length)
            {
                var larger = ArrayPool<char>.Shared.Rent(Math.Max(_buffer.Length * 2, _written + more));
                _buffer.AsSpan(0, _written).CopyTo(larger);
                ArrayPool<char>.Shared.Return(_buffer);
                _buffer = larger;
            }
        }
    }

    // The result of a normalization on a NormalizedText: no change is begun until the first replacement, so text that does
    // not change is left as it is; then the unchanged runs are copied with their origins and each replacement's characters
    // stand for what they replace: all of it (sentencepiece), or, Positional, by position as tokenizers' Precompiled
    // aligns them (its replace: the i-th character kept, extra ones added after the last, missing ones removed).
    private struct SpanOutput(NormalizedText text) : IDisposable
    {
        public bool Positional { get; init; }

        private NormalizedText.Rewriter _writer;
        private bool _started;
        private int _pending;
        private char[]? _chars;

        public void Replace(int start, int length, ReadOnlySpan<byte> utf8)
        {
            if (!_started)
            {
                _writer = text.Rewrite();
                _started = true;
            }

            if (Positional)
            {
                CopyRun(_pending, start);
            }
            else
            {
                _writer.Copy(_pending, start - _pending);
            }

            if (_chars is null || _chars.Length < utf8.Length)
            {
                if (_chars is not null)
                {
                    ArrayPool<char>.Shared.Return(_chars);
                }

                _chars = ArrayPool<char>.Shared.Rent(Math.Max(64, utf8.Length));
            }

            int count = Encoding.UTF8.GetChars(utf8, _chars);
            if (Positional)
            {
                AppendPositional(_chars.AsSpan(0, count), start, length);
            }
            else
            {
                _writer.Append(_chars.AsSpan(0, count), start, length);
            }

            _pending = start + length;
        }

        // tokenizers' alignment (Precompiled's replace, then NormalizedString.transform): a cursor walks the source
        // characters in order; a kept character stands for the one at the cursor and moves it on, an added one (beyond the
        // characters it replaces) for the one before; characters a replacement drops move the cursor past them, unless
        // nothing has been written yet (tokenizers then has no character to charge them to, so what follows stands one
        // place earlier, which this reproduces).
        private int _cursor, _previous, _previousWidth;
        private bool _written;

        private void Keep(ReadOnlySpan<char> chars)
        {
            var source = text.Span;
            if (_cursor >= source.Length)
            {
                Add(chars);                                                    // past the source: the last character's
                return;
            }

            int width = _cursor + 1 < source.Length && char.IsSurrogatePair(source[_cursor], source[_cursor + 1]) ? 2 : 1;
            _writer.Append(chars, _cursor, width);
            (_previous, _previousWidth) = (_cursor, width);
            _cursor += width;
            _written = true;
        }

        private void Add(ReadOnlySpan<char> chars)
        {
            if (_previousWidth > 0)
            {
                _writer.Append(chars, _previous, _previousWidth);
            }
            else
            {
                foreach (char c in chars)
                {
                    _writer.AppendOrigin(c, 0, 0);
                }
            }

            _written = true;
        }

        // The unchanged characters [from, to): copied as they are while the cursor is on them, else each kept.
        private void CopyRun(int from, int to)
        {
            if (_cursor == from)
            {
                _writer.Copy(from, to - from);
                if (to > from)
                {
                    var source = text.Span;
                    _previousWidth = to - from > 1 && char.IsSurrogatePair(source[to - 2], source[to - 1]) ? 2 : 1;
                    _previous = to - _previousWidth;
                    _written = true;
                }

                _cursor = to;
                return;
            }

            var span = text.Span;
            for (int i = from; i < to;)
            {
                int w = i + 1 < to && char.IsSurrogatePair(span[i], span[i + 1]) ? 2 : 1;
                Keep(span.Slice(i, w));
                i += w;
            }
        }

        private void AppendPositional(ReadOnlySpan<char> chars, int start, int length)
        {
            var source = text.Span.Slice(start, length);
            int replaced = 0;
            for (int at = 0; at < source.Length; replaced++)
            {
                at += at + 1 < source.Length && char.IsSurrogatePair(source[at], source[at + 1]) ? 2 : 1;
            }

            int index = 0;
            for (int i = 0; i < chars.Length; index++)
            {
                int w = i + 1 < chars.Length && char.IsSurrogatePair(chars[i], chars[i + 1]) ? 2 : 1;
                if (index < replaced)
                {
                    Keep(chars.Slice(i, w));
                }
                else
                {
                    Add(chars.Slice(i, w));
                }

                i += w;
            }

            if (_written)
            {
                var span = text.Span;
                for (int k = index; k < replaced && _cursor < span.Length; k++)
                {
                    _cursor += _cursor + 1 < span.Length && char.IsSurrogatePair(span[_cursor], span[_cursor + 1]) ? 2 : 1;
                }
            }
        }

        public void Finish()
        {
            if (_started)
            {
                if (Positional)
                {
                    CopyRun(_pending, text.Length);
                }
                else
                {
                    _writer.Copy(_pending, text.Length - _pending);
                }

                _writer.Commit();
            }
        }

        public void Dispose()
        {
            if (_chars is not null)
            {
                ArrayPool<char>.Shared.Return(_chars);
                _chars = null;
            }
        }
    }
}
