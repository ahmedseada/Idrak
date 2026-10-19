// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;

namespace Idrak.Models.Abstractions;

/// <summary>
/// Text being normalized for a tokenizer, with where each of its characters came from: every character keeps the range of
/// the original text it stands for, so a token's offsets can be given in the text the caller passed, whatever the
/// normalizers did to it (lowercased, decomposed, replaced, prepended). The characters and their ranges live in pooled
/// buffers, reused from one text to the next; <see cref="Dispose"/> returns them.
/// <para>
/// A change is written through <see cref="Rewrite"/>: the new characters are appended one by one, each with the
/// characters of the current text it comes from, then committed. A character made from nothing (a prefix, say) takes the
/// range of the character it stands before.
/// </para>
/// </summary>
public sealed class NormalizedText : IDisposable
{
    private char[] _chars = [], _next = [];
    private int[] _starts = [], _ends = [], _nextStarts = [], _nextEnds = [];
    private int _length, _nextLength;
    private bool _rewriting;

    /// <summary>An empty text.</summary>
    public NormalizedText()
    {
    }

    // The text the caller passed: a string, or (Reset from a buffer) the buffer's characters, made a string only when asked for.
    private string? _original = "";
    private char[]? _originalChars;
    private int _originalLength;

    /// <summary>The text the caller passed (offsets index into it).</summary>
    public string Original => _original ??= new string(_originalChars!, 0, _originalLength);

    /// <summary>Where the normalized range begins in <see cref="Original"/>.</summary>
    public int OriginalStart { get; private set; }

    /// <summary>Where the normalized range ends in <see cref="Original"/>.</summary>
    public int OriginalEnd { get; private set; }

    /// <summary>How many characters the text has now.</summary>
    public int Length => _length;

    /// <summary>The characters now.</summary>
    public ReadOnlySpan<char> Span => _chars.AsSpan(0, _length);

    /// <summary>
    /// Starts over with <paramref name="original"/>[<paramref name="start"/>..<paramref name="start"/> + <paramref name="length"/>],
    /// each character standing for itself.
    /// </summary>
    public void Reset(string original, int start, int length)
    {
        ArgumentNullException.ThrowIfNull(original);
        ArgumentOutOfRangeException.ThrowIfNegative(start);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(length, original.Length - start);
        _original = original;
        _originalChars = null;
        Start(original.AsSpan(start, length), start);
    }

    // Starts over with original[start..start + length] of a caller's buffer, whose first originalLength characters are the
    // original text (Original makes them a string only when asked for; the caller leaves them unchanged until the next
    // Reset). For callers that decode text into a buffer of their own, so no string is made per block.
    internal void Reset(char[] original, int originalLength, int start, int length)
    {
        ArgumentNullException.ThrowIfNull(original);
        ArgumentOutOfRangeException.ThrowIfNegative(start);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(originalLength, original.Length);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(length, originalLength - start);
        _original = null;
        (_originalChars, _originalLength) = (original, originalLength);
        Start(original.AsSpan(start, length), start);
    }

    // The text's characters, each standing for itself at start + its index.
    private void Start(ReadOnlySpan<char> text, int start)
    {
        _rewriting = false;
        OriginalStart = start;
        OriginalEnd = start + text.Length;
        Ensure(ref _chars, ref _starts, ref _ends, text.Length);
        text.CopyTo(_chars);
        for (int i = 0; i < text.Length; i++)
        {
            _starts[i] = start + i;
            _ends[i] = start + i + 1;
        }

        _length = text.Length;
    }

    /// <summary>Starts over with all of <paramref name="original"/>.</summary>
    public void Reset(string original) => Reset(original, 0, original?.Length ?? 0);

    // Starts over as Reset does, but with no characters yet: a one-pass stage writes them and their ranges straight into the
    // buffers (DirectChars, DirectStarts, DirectEnds, room for `capacity`), then sets how many it wrote (DirectLength).
    internal void StartDirect(string original, int start, int length, int capacity)
    {
        ArgumentNullException.ThrowIfNull(original);
        ArgumentOutOfRangeException.ThrowIfNegative(start);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(length, original.Length - start);
        _original = original;
        _originalChars = null;
        StartEmpty(start, length, capacity);
    }

    // StartDirect on original[start..start + length] of a caller's buffer (as the buffer form of Reset).
    internal void StartDirect(char[] original, int originalLength, int start, int length, int capacity)
    {
        ArgumentNullException.ThrowIfNull(original);
        ArgumentOutOfRangeException.ThrowIfNegative(start);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(originalLength, original.Length);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(length, originalLength - start);
        _original = null;
        (_originalChars, _originalLength) = (original, originalLength);
        StartEmpty(start, length, capacity);
    }

    private void StartEmpty(int start, int length, int capacity)
    {
        _rewriting = false;
        OriginalStart = start;
        OriginalEnd = start + length;
        Ensure(ref _chars, ref _starts, ref _ends, capacity);
        _length = 0;
    }

    internal Span<char> DirectChars => _chars;

    internal Span<int> DirectStarts => _starts;

    internal Span<int> DirectEnds => _ends;

    // How many characters a one-pass stage wrote (StartDirect).
    internal int DirectLength
    {
        set
        {
            ArgumentOutOfRangeException.ThrowIfGreaterThan((uint)value, (uint)_chars.Length, nameof(value));
            _length = value;
        }
    }

    /// <summary>The range of <see cref="Original"/> the character at <paramref name="index"/> stands for.</summary>
    public (int Start, int End) OriginOf(int index)
    {
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual((uint)index, (uint)_length, nameof(index));
        return (_starts[index], _ends[index]);
    }

    /// <summary>
    /// The range of <see cref="Original"/> the characters [<paramref name="start"/>, <paramref name="start"/> + <paramref name="length"/>)
    /// stand for together; for no characters, the empty range where character <paramref name="start"/> begins (or where
    /// the text ends).
    /// </summary>
    public (int Start, int End) OriginOf(int start, int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(start);
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(start + length, _length);
        if (length == 0)
        {
            int at = start < _length ? _starts[start] : _length > 0 ? _ends[_length - 1] : OriginalEnd;
            return (at, at);
        }

        var starts = _starts.AsSpan(start, length);
        var ends = _ends.AsSpan(start, length);
        int from = starts[0], to = ends[0];
        for (int i = 1; i < starts.Length; i++)
        {
            from = Math.Min(from, starts[i]);
            to = Math.Max(to, ends[i]);
        }

        return (from, to);
    }

    // What a change has written so far (before Commit).
    internal ReadOnlySpan<char> Pending => _next.AsSpan(0, _nextLength);

    // The original ranges of the current characters, for stages that carry them through a buffer of their own.
    internal ReadOnlySpan<int> Starts => _starts.AsSpan(0, _length);

    internal ReadOnlySpan<int> Ends => _ends.AsSpan(0, _length);

    /// <summary>Begins a change: append the new text through the returned writer, then <see cref="Rewriter.Commit"/> it.</summary>
    public Rewriter Rewrite()
    {
        _rewriting = true;
        _nextLength = 0;
        return new Rewriter(this);
    }

    /// <summary>
    /// Replaces the whole text with <paramref name="text"/>: the characters before and after the first and last that differ
    /// keep their ranges, and the changed middle stands for the changed middle of the old text as a whole. For a component
    /// that gives only its result.
    /// </summary>
    public void Set(ReadOnlySpan<char> text)
    {
        var old = Span;
        int prefix = old.CommonPrefixLength(text);
        int suffix = 0;
        while (suffix < old.Length - prefix && suffix < text.Length - prefix && old[old.Length - 1 - suffix] == text[text.Length - 1 - suffix])
        {
            suffix++;
        }

        var writer = Rewrite();
        writer.Copy(0, prefix);
        var (start, end) = OriginOf(prefix, old.Length - prefix - suffix);   // the changed middle's range, once
        for (int i = prefix; i < text.Length - suffix; i++)
        {
            writer.AppendOrigin(text[i], start, end);
        }

        writer.Copy(old.Length - suffix, suffix);
        writer.Commit();
    }

    /// <summary>The text now.</summary>
    public override string ToString() => new(Span);

    /// <summary>Returns the buffers to the pool.</summary>
    public void Dispose()
    {
        Return(ref _chars, ref _starts, ref _ends);
        Return(ref _next, ref _nextStarts, ref _nextEnds);
        _length = _nextLength = 0;
    }

    private static void Ensure(ref char[] chars, ref int[] starts, ref int[] ends, int size)
    {
        if (chars.Length >= size)
        {
            return;
        }

        int capacity = Math.Max(size, Math.Max(64, chars.Length * 2));
        var newChars = ArrayPool<char>.Shared.Rent(capacity);
        var newStarts = ArrayPool<int>.Shared.Rent(capacity);
        var newEnds = ArrayPool<int>.Shared.Rent(capacity);
        Return(ref chars, ref starts, ref ends);
        (chars, starts, ends) = (newChars, newStarts, newEnds);
    }

    private static void Grow(ref char[] chars, ref int[] starts, ref int[] ends, int used, int size)
    {
        if (chars.Length >= size)
        {
            return;
        }

        int capacity = Math.Max(size, Math.Max(64, chars.Length * 2));
        var newChars = ArrayPool<char>.Shared.Rent(capacity);
        var newStarts = ArrayPool<int>.Shared.Rent(capacity);
        var newEnds = ArrayPool<int>.Shared.Rent(capacity);
        chars.AsSpan(0, used).CopyTo(newChars);
        starts.AsSpan(0, used).CopyTo(newStarts);
        ends.AsSpan(0, used).CopyTo(newEnds);
        Return(ref chars, ref starts, ref ends);
        (chars, starts, ends) = (newChars, newStarts, newEnds);
    }

    private static void Return(ref char[] chars, ref int[] starts, ref int[] ends)
    {
        if (chars.Length > 0)
        {
            ArrayPool<char>.Shared.Return(chars);
            ArrayPool<int>.Shared.Return(starts);
            ArrayPool<int>.Shared.Return(ends);
        }

        (chars, starts, ends) = ([], [], []);
    }

    private void Add(char c, int start, int end)
    {
        if (!_rewriting)
        {
            throw new InvalidOperationException("Begin a change with NormalizedText.Rewrite first.");
        }

        if (_nextLength == _next.Length)
        {
            Grow(ref _next, ref _nextStarts, ref _nextEnds, _nextLength, _nextLength + 1);
        }

        _next[_nextLength] = c;
        _nextStarts[_nextLength] = start;
        _nextEnds[_nextLength] = end;
        _nextLength++;
    }

    private void CopyRange(int from, int count)
    {
        if (!_rewriting)
        {
            throw new InvalidOperationException("Begin a change with NormalizedText.Rewrite first.");
        }

        ArgumentOutOfRangeException.ThrowIfGreaterThan(from + count, _length);
        Grow(ref _next, ref _nextStarts, ref _nextEnds, _nextLength, _nextLength + count);
        _chars.AsSpan(from, count).CopyTo(_next.AsSpan(_nextLength));
        _starts.AsSpan(from, count).CopyTo(_nextStarts.AsSpan(_nextLength));
        _ends.AsSpan(from, count).CopyTo(_nextEnds.AsSpan(_nextLength));
        _nextLength += count;
    }

    private void Swap()
    {
        if (!_rewriting)
        {
            throw new InvalidOperationException("No change was begun (NormalizedText.Rewrite).");
        }

        (_chars, _next) = (_next, _chars);
        (_starts, _nextStarts) = (_nextStarts, _starts);
        (_ends, _nextEnds) = (_nextEnds, _ends);
        _length = _nextLength;
        _nextLength = 0;
        _rewriting = false;
    }

    /// <summary>
    /// Writes the new text of a change (<see cref="Rewrite"/>): each appended character names the characters of the
    /// current text it comes from, and <see cref="Commit"/> makes it the text.
    /// </summary>
    public readonly struct Rewriter
    {
        private readonly NormalizedText _text;

        internal Rewriter(NormalizedText text) => _text = text;

        /// <summary>How many characters the new text has so far.</summary>
        public int Position => _text._nextLength;

        /// <summary>Appends <paramref name="c"/>, made of the current characters [<paramref name="from"/>, <paramref name="from"/> + <paramref name="count"/>).</summary>
        public void Append(char c, int from, int count = 1)
        {
            var (start, end) = _text.OriginOf(from, count);
            _text.Add(c, start, end);
        }

        /// <summary>Appends <paramref name="chars"/>, each made of the current characters [<paramref name="from"/>, <paramref name="from"/> + <paramref name="count"/>).</summary>
        public void Append(ReadOnlySpan<char> chars, int from, int count)
        {
            var (start, end) = _text.OriginOf(from, count);
            foreach (char c in chars)
            {
                _text.Add(c, start, end);
            }
        }

        /// <summary>Appends <paramref name="c"/> standing for the range [<paramref name="start"/>, <paramref name="end"/>) of the original text.</summary>
        public void AppendOrigin(char c, int start, int end) => _text.Add(c, start, end);

        /// <summary>Appends the current characters [<paramref name="from"/>, <paramref name="from"/> + <paramref name="count"/>) as they are.</summary>
        public void Copy(int from, int count) => _text.CopyRange(from, count);

        /// <summary>Makes the new text the text.</summary>
        public void Commit() => _text.Swap();
    }
}

/// <summary>
/// Normalized text split into the pieces a tokenizer model encodes one by one: each piece is a range of
/// <see cref="Text"/>, so pieces are never copied into strings, and their offsets in the original text follow from it.
/// </summary>
public sealed class PreTokenizedText : IDisposable
{
    private readonly List<(int Start, int Length)> _scratch = [];

    /// <summary>The normalized text the pieces are ranges of (pre-tokenizers that rewrite text change it too).</summary>
    public NormalizedText Text { get; } = new();

    /// <summary>The pieces, in order, as ranges of <see cref="Text"/>.</summary>
    public List<(int Start, int Length)> Pieces { get; } = [];

    /// <summary>A piece's characters.</summary>
    public ReadOnlySpan<char> Piece(int index) => Text.Span.Slice(Pieces[index].Start, Pieces[index].Length);

    /// <summary>The range of the original text a piece stands for.</summary>
    public (int Start, int End) OriginOf(int index) => Text.OriginOf(Pieces[index].Start, Pieces[index].Length);

    /// <summary>The pieces as strings (for a component that works on strings, and for tests).</summary>
    public List<string> PieceStrings()
    {
        var pieces = new List<string>(Pieces.Count);
        for (int i = 0; i < Pieces.Count; i++)
        {
            pieces.Add(new string(Piece(i)));
        }

        return pieces;
    }

    /// <summary>The whole normalized text as one piece.</summary>
    public void OnePiece()
    {
        Pieces.Clear();
        Pieces.Add((0, Text.Length));
    }

    // A list a stage fills with the new pieces before they replace the old (cleared; reused from call to call).
    internal List<(int Start, int Length)> Scratch()
    {
        _scratch.Clear();
        return _scratch;
    }

    // The new pieces become the pieces.
    internal void TakeScratch()
    {
        Pieces.Clear();
        Pieces.AddRange(_scratch);
        _scratch.Clear();
    }

    /// <summary>Returns the text's buffers to the pool.</summary>
    public void Dispose() => Text.Dispose();
}
