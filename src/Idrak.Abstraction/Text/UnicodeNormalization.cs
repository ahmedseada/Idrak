// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.CompilerServices;
using System.Text;

namespace Idrak.Abstraction.Text;

/// <summary>
/// Unicode normalization (UAX #15: NFC, NFD, NFKC, NFKD) in C#, from tables made from the Unicode character database
/// (<see cref="UnicodeVersion"/>), so it gives the same result on every platform. .NET's own normalization
/// (<see cref="string.Normalize(NormalizationForm)"/>) goes through ICU or Windows, allocates on each call, and returns
/// the text unchanged under invariant globalization; the library uses this class instead.
/// </summary>
/// <remarks>
/// Text is checked first (the quick-check property of each form, with the combining classes in order) and returned or
/// copied as it is when nothing can change. Otherwise only the segments that may change (a starter and what follows it,
/// up to the next code point nothing before can reach) are decomposed, reordered and, for NFC and NFKC, composed; the
/// rest is copied. Nothing is allocated but the result string of <see cref="Normalize(string, NormalizationForm)"/>
/// (pooled buffers, returned before the method returns). Invalid UTF-16 (a lone surrogate) is refused with an
/// <see cref="ArgumentException"/>, as .NET's normalization refuses it.
/// </remarks>
public static partial class UnicodeNormalization
{
    // Flag bits of RecordInfo (as tools/unicode/make_normalization_tables.py writes them).
    private const uint NfdNo = 1, NfkdNo = 2, NfcNo = 4, NfcMaybe = 8, NfkcNo = 16, NfkcMaybe = 32;
    private const uint NfdBoundary = 64, NfkdBoundary = 128, NfcBoundary = 256, NfkcBoundary = 512;

    private const int SBase = 0xAC00, LBase = 0x1100, VBase = 0x1161, TBase = 0x11A7;
    private const int LCount = 19, VCount = 21, TCount = 28, NCount = VCount * TCount, SCount = LCount * NCount;

    // Per form (C, D, KC, KD): the first code point whose record is not "a starter, Yes, a segment boundary". Text below
    // it is skipped with a vectorized search (all of ASCII; Latin-1 too for NFC).
    private static readonly int[] Plain = [FirstChanging(0), FirstChanging(1), FirstChanging(2), FirstChanging(3)];

    // Per form: one bit per UTF-16 code unit of the basic plane (8 KB a form, built once), set for a code point that is a
    // starter, Yes and a segment boundary: one load tells most letters (Arabic, CJK, Hangul syllables for NFC) apart.
    // Surrogates are never set, so a set bit is always a whole code point.
    private static readonly ulong[][] PlainBits = BuildPlainBits();

    /// <summary>The longest full decomposition of one code point: canonical (NFD, NFC) or compatibility (NFKD, NFKC).</summary>
    public static int MaxDecompositionLength(bool compatibility) => compatibility ? MaxCompatibility : MaxCanonical;

    /// <summary>The canonical combining class of a code point (0 for starters).</summary>
    public static int CombiningClass(int codePoint) => (int)(Info(codePoint) >> 16);

    /// <summary>
    /// Writes the full decomposition of one code point (canonical, or compatibility when <paramref name="compatibility"/>
    /// is set; Hangul syllables by the algorithm) into <paramref name="destination"/> and returns its length: the code
    /// point itself (length 1) when it does not decompose. The destination needs <see cref="MaxDecompositionLength"/>.
    /// </summary>
    public static int Decompose(int codePoint, Span<int> destination, bool compatibility)
    {
        if ((uint)(codePoint - SBase) < SCount)
        {
            return DecomposeHangul(codePoint, destination);
        }

        int mapping = Mapping(codePoint, compatibility);
        if (mapping == 0)
        {
            destination[0] = codePoint;
            return 1;
        }

        var parts = Mappings.Slice(mapping >> 5, mapping & 31);
        parts.CopyTo(destination);
        return parts.Length;
    }

    /// <summary>The primary composite of two code points (Hangul syllables included), or -1 when they do not compose.</summary>
    public static int Compose(int first, int second)
    {
        if ((uint)(first - LBase) < LCount && (uint)(second - VBase) < VCount)
        {
            return SBase + ((first - LBase) * VCount + second - VBase) * TCount;
        }

        if ((uint)(first - SBase) < SCount && (first - SBase) % TCount == 0 && (uint)(second - TBase - 1) < TCount - 1)
        {
            return first + second - TBase;
        }

        int at = CompositionPairs.BinarySearch((long)first << 21 | (uint)second);
        return at >= 0 ? Composites[at] : -1;
    }

    /// <summary>Whether <paramref name="text"/> is already in <paramref name="form"/>.</summary>
    public static bool IsNormalized(ReadOnlySpan<char> text, NormalizationForm form = NormalizationForm.FormC)
    {
        var rules = new Rules(form);
        return QuickCheck(text, rules, out int safe) switch
        {
            Check.Yes => true,
            Check.No => false,
            _ => SegmentsUnchanged(text, rules, safe),
        };
    }

    /// <summary>
    /// <paramref name="text"/> in <paramref name="form"/>: the same instance when it is already normalized.
    /// </summary>
    public static string Normalize(string text, NormalizationForm form = NormalizationForm.FormC)
    {
        ArgumentNullException.ThrowIfNull(text);
        var rules = new Rules(form);
        if (QuickCheck(text, rules, out int safe) == Check.Yes)
        {
            return text;
        }

        char[] buffer = ArrayPool<char>.Shared.Rent(text.Length + (text.Length >> 1) + 16);
        try
        {
            var sink = new Sink(buffer);
            Run(text, ref sink, rules, safe);
            if (sink.Overflow)
            {
                ArrayPool<char>.Shared.Return(buffer);
                buffer = ArrayPool<char>.Shared.Rent(sink.Count);
                sink = new Sink(buffer);
                Run(text, ref sink, rules, safe);
            }

            var result = buffer.AsSpan(0, sink.Count);
            return result.SequenceEqual(text) ? text : new string(result);
        }
        finally
        {
            ArrayPool<char>.Shared.Return(buffer);
        }
    }

    /// <summary>
    /// Writes <paramref name="source"/> in <paramref name="form"/> into <paramref name="destination"/>; false (with
    /// <paramref name="written"/> 0) when it does not fit (<see cref="GetNormalizedLength"/> gives the room it needs).
    /// </summary>
    public static bool TryNormalize(ReadOnlySpan<char> source, Span<char> destination, out int written,
        NormalizationForm form = NormalizationForm.FormC)
    {
        var rules = new Rules(form);
        var sink = new Sink(destination);
        Run(source, ref sink, rules, QuickCheck(source, rules, out int safe) == Check.Yes ? source.Length : safe);
        written = sink.Overflow ? 0 : sink.Count;
        return !sink.Overflow;
    }

    /// <summary>The length of <paramref name="source"/> in <paramref name="form"/>, in UTF-16 code units.</summary>
    public static int GetNormalizedLength(ReadOnlySpan<char> source, NormalizationForm form = NormalizationForm.FormC)
    {
        var rules = new Rules(form);
        if (QuickCheck(source, rules, out int safe) == Check.Yes)
        {
            return source.Length;
        }

        var sink = new Sink(Span<char>.Empty);
        Run(source, ref sink, rules, safe);
        return sink.Count;
    }

    // What a form looks at: its quick-check bits, its segment boundary bit, its decomposition and whether it composes.
    private readonly struct Rules
    {
        public readonly uint No, Maybe, Boundary;
        public readonly int Plain;
        public readonly ulong[] Bits;
        public readonly bool Compatibility, Composes;

        public Rules(NormalizationForm form)
        {
            (No, Maybe, Boundary, Compatibility, Composes, int index) = form switch
            {
                NormalizationForm.FormC => (NfcNo, NfcMaybe, NfcBoundary, false, true, 0),
                NormalizationForm.FormD => (NfdNo, 0u, NfdBoundary, false, false, 1),
                NormalizationForm.FormKC => (NfkcNo, NfkcMaybe, NfkcBoundary, true, true, 2),
                NormalizationForm.FormKD => (NfkdNo, 0u, NfkdBoundary, true, false, 3),
                _ => throw new ArgumentException($"Normalization form '{form}' is not one of C, D, KC and KD.", nameof(form)),
            };
            Plain = UnicodeNormalization.Plain[index];
            Bits = PlainBits[index];
        }
    }

    private enum Check { Yes, No, Maybe }

    // Writes into a span, then only counts once it is full (Overflow), so the room needed is known.
    private ref struct Sink(Span<char> destination)
    {
        private readonly Span<char> _destination = destination;
        public int Count;
        public bool Overflow;

        public void Write(ReadOnlySpan<char> text)
        {
            if (!Overflow && text.TryCopyTo(_destination[Count..]))
            {
                Count += text.Length;
                return;
            }

            Overflow = true;
            Count += text.Length;
        }

        public void Write(int codePoint)
        {
            int width = codePoint >= 0x10000 ? 2 : 1;
            if (!Overflow && Count + width <= _destination.Length)
            {
                if (width == 1)
                {
                    _destination[Count] = (char)codePoint;
                }
                else
                {
                    _destination[Count] = (char)(0xD7C0 + (codePoint >> 10));
                    _destination[Count + 1] = (char)(0xDC00 | (codePoint & 0x3FF));
                }
            }
            else
            {
                Overflow = true;
            }

            Count += width;
        }
    }

    // UAX #15's quick check: No at a No code point or combining classes out of order, Maybe at a Maybe code point.
    // `safe` is where the segment of the first code point that is not Yes starts: everything before it is normalized.
    private static Check QuickCheck(ReadOnlySpan<char> text, in Rules rules) => QuickCheck(text, rules, out _);

    private static Check QuickCheck(ReadOnlySpan<char> text, in Rules rules, out int safe)
    {
        safe = text.Length;
        int i = text.IndexOfAnyExceptInRange('\0', (char)(rules.Plain - 1));
        if (i < 0)
        {
            return Check.Yes;
        }

        var bits = rules.Bits;
        var result = Check.Yes;
        uint lastClass = 0;
        int segment = i > 0 ? i - 1 : 0;                                   // the plain code unit before starts the segment
        while (i < text.Length)
        {
            char c = text[i];
            if ((bits[c >> 6] & (1UL << c)) != 0)
            {
                lastClass = 0;
                segment = i;
                i++;
                continue;
            }

            int codePoint = CodePointAt(text, i, out int width);
            uint info = Info(codePoint);
            uint cls = info >> 16;
            if ((info & rules.Boundary) != 0)
            {
                segment = i;
            }

            if ((info & rules.No) != 0 || (cls != 0 && lastClass > cls))
            {
                if (result == Check.Yes)
                {
                    safe = segment;
                }

                return Check.No;
            }

            if ((info & rules.Maybe) != 0 && result == Check.Yes)
            {
                result = Check.Maybe;
                safe = segment;
            }

            lastClass = cls;
            i += width;
        }

        return result;
    }

    // Walks the text segment by segment (a segment starts before a code point with the form's boundary bit): a segment
    // that passes the quick check is copied as it is (in runs), another is normalized on its own.
    private static void Run(ReadOnlySpan<char> text, ref Sink sink, in Rules rules, int from = 0)
    {
        int copyFrom = 0, segment = from;
        bool clean = true;
        uint lastClass = 0;
        var bits = rules.Bits;
        for (int i = from; i < text.Length;)
        {
            char c = text[i];
            uint info;
            int width;
            if ((bits[c >> 6] & (1UL << c)) != 0)
            {
                if (clean)
                {
                    segment = i;                                                   // a clean segment ends here; nothing to do
                    lastClass = 0;
                    i++;
                    continue;
                }

                info = rules.Boundary;
                width = 1;
            }
            else
            {
                info = Info(CodePointAt(text, i, out width));
            }

            if ((info & rules.Boundary) != 0 && i > segment)
            {
                if (!clean)
                {
                    sink.Write(text[copyFrom..segment]);
                    NormalizeSegment(text[segment..i], ref sink, rules);
                    copyFrom = i;
                }

                segment = i;
                clean = true;
                lastClass = 0;
            }

            uint cls = info >> 16;
            if ((info & (rules.No | rules.Maybe)) != 0 || (cls != 0 && lastClass > cls))
            {
                clean = false;
            }

            lastClass = cls;
            i += width;
        }

        if (clean)
        {
            sink.Write(text[copyFrom..]);
        }
        else
        {
            sink.Write(text[copyFrom..segment]);
            NormalizeSegment(text[segment..], ref sink, rules);
        }
    }

    // After a Maybe: whether normalizing each segment that is not clean leaves it as it is.
    private static bool SegmentsUnchanged(ReadOnlySpan<char> text, in Rules rules, int from)
    {
        char[] buffer = ArrayPool<char>.Shared.Rent(64);
        try
        {
            int segment = from;
            bool clean = true;
            for (int i = from; i <= text.Length;)
            {
                uint info = rules.Boundary;
                int width = 1;
                if (i < text.Length && text[i] >= rules.Plain)
                {
                    info = Info(CodePointAt(text, i, out width));
                }

                if (i == text.Length || ((info & rules.Boundary) != 0 && i > segment))
                {
                    if (!clean)
                    {
                        var part = text[segment..i];
                        var sink = new Sink(buffer);
                        NormalizeSegment(part, ref sink, rules);
                        if (sink.Overflow)
                        {
                            ArrayPool<char>.Shared.Return(buffer);
                            buffer = ArrayPool<char>.Shared.Rent(sink.Count);
                            sink = new Sink(buffer);
                            NormalizeSegment(part, ref sink, rules);
                        }

                        if (!part.SequenceEqual(buffer.AsSpan(0, sink.Count)))
                        {
                            return false;
                        }
                    }

                    if (i == text.Length)
                    {
                        return true;
                    }

                    segment = i;
                    clean = true;
                }

                if ((info & rules.Maybe) != 0)
                {
                    clean = false;
                }

                i += width;
            }

            return true;
        }
        finally
        {
            ArrayPool<char>.Shared.Return(buffer);
        }
    }

    // One segment: full decomposition, canonical ordering, then composition for NFC and NFKC. The code points are kept
    // with their combining class (class << 21 | code point) in a pooled array.
    private static void NormalizeSegment(ReadOnlySpan<char> text, ref Sink sink, in Rules rules)
    {
        int room = text.Length * (rules.Compatibility ? MaxCompatibility : MaxCanonical);
        int[]? kept = t_points;                                            // this thread's scratch; a longer segment rents
        int[] points = kept is not null && kept.Length >= room ? kept : ArrayPool<int>.Shared.Rent(room);
        t_points = null;                                                   // not shared with a nested call while in use
        try
        {
            int count = 0;
            for (int i = 0; i < text.Length;)
            {
                int codePoint = CodePointAt(text, i, out int width);
                i += width;
                if ((uint)(codePoint - SBase) < SCount)
                {
                    count += DecomposeHangul(codePoint, points.AsSpan(count));
                    continue;
                }

                int mapping = Mapping(codePoint, rules.Compatibility);
                if (mapping == 0)
                {
                    points[count++] = (int)(Info(codePoint) >> 16 << 21) | codePoint;
                    continue;
                }

                foreach (int part in Mappings.Slice(mapping >> 5, mapping & 31))
                {
                    points[count++] = (int)(Info(part) >> 16 << 21) | part;
                }
            }

            // Canonical ordering: a stable insertion sort of each run of non-starters by combining class.
            for (int i = 1; i < count; i++)
            {
                int point = points[i];
                int cls = point >> 21;
                if (cls == 0)
                {
                    continue;
                }

                int j = i;
                while (j > 0 && points[j - 1] >> 21 > cls)
                {
                    points[j] = points[j - 1];
                    j--;
                }

                points[j] = point;
            }

            if (rules.Composes && count > 1)
            {
                count = ComposeAll(points.AsSpan(0, count));
            }

            for (int i = 0; i < count; i++)
            {
                sink.Write(points[i] & 0x1FFFFF);
            }
        }
        finally
        {
            if (ReferenceEquals(points, kept) || (kept is null && points.Length <= ScratchLimit))
            {
                t_points = points;
            }
            else
            {
                ArrayPool<int>.Shared.Return(points);
                t_points = kept;
            }
        }
    }

    // A thread keeps one segment buffer of up to ScratchLimit code points (16 KB), so a text of many short changed
    // segments (Hangul syllables under NFD, accented Latin) takes nothing from the pool per segment.
    private const int ScratchLimit = 4096;

    [ThreadStatic]
    private static int[]? t_points;

    // UAX #15's canonical composition over decomposed, ordered code points (class << 21 | code point); returns the new
    // count. A character composes with the last starter unless something between blocks it (a starter, or a class not
    // below its own).
    private static int ComposeAll(Span<int> points)
    {
        int starter = 0;
        int lastClass = points[0] >> 21 == 0 ? 0 : 256;               // text that starts with a mark composes nothing first
        int written = 1;
        for (int i = 1; i < points.Length; i++)
        {
            int point = points[i];
            int cls = point >> 21;
            int composite = lastClass < cls || lastClass == 0 ? Compose(points[starter] & 0x1FFFFF, point & 0x1FFFFF) : -1;
            if (composite >= 0)
            {
                points[starter] = (int)(Info(composite) >> 16 << 21) | composite;
                continue;
            }

            if (cls == 0)
            {
                starter = written;
            }

            lastClass = cls;
            points[written++] = point;
        }

        return written;
    }

    private static int DecomposeHangul(int syllable, Span<int> destination)
    {
        int s = syllable - SBase;
        destination[0] = LBase + s / NCount;
        destination[1] = VBase + s % NCount / TCount;
        int t = s % TCount;
        if (t == 0)
        {
            return 2;
        }

        destination[2] = TBase + t;
        return 3;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint Info(int codePoint) =>
        (uint)codePoint > 0x10FFFF ? NfdBoundary | NfkdBoundary | NfcBoundary | NfkcBoundary
            : RecordInfo[Stage2[Stage1[codePoint >> 8] << 8 | (codePoint & 0xFF)]];

    private static int Mapping(int codePoint, bool compatibility)
    {
        int record = Stage2[Stage1[codePoint >> 8] << 8 | (codePoint & 0xFF)];
        return compatibility ? RecordCompatibility[record] : RecordCanonical[record];
    }

    // The code point at `at`; a lone surrogate is refused, as .NET's normalization refuses it.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int CodePointAt(ReadOnlySpan<char> text, int at, out int width)
    {
        char c = text[at];
        if (!char.IsSurrogate(c))
        {
            width = 1;
            return c;
        }

        if (char.IsHighSurrogate(c) && at + 1 < text.Length && char.IsLowSurrogate(text[at + 1]))
        {
            width = 2;
            return char.ConvertToUtf32(c, text[at + 1]);
        }

        throw new ArgumentException($"Invalid Unicode code point found at index {at}.", nameof(text));
    }

    private static ulong[][] BuildPlainBits()
    {
        var bits = new ulong[4][];
        for (int form = 0; form < 4; form++)
        {
            bits[form] = new ulong[1024];
        }

        uint[] boundary = [NfcBoundary, NfdBoundary, NfkcBoundary, NfkdBoundary];
        uint[] bad = [NfcNo | NfcMaybe, NfdNo, NfkcNo | NfkcMaybe, NfkdNo];
        for (int c = 0; c < 0x10000; c++)
        {
            if (c is >= 0xD800 and <= 0xDFFF)
            {
                continue;
            }

            uint info = Info(c);
            for (int form = 0; form < 4; form++)
            {
                if ((info & boundary[form]) != 0 && (info & bad[form]) == 0 && info >> 16 == 0)
                {
                    bits[form][c >> 6] |= 1UL << c;
                }
            }
        }

        return bits;
    }

    private static int FirstChanging(int form)
    {
        uint wanted = form switch { 0 => NfcBoundary, 1 => NfdBoundary, 2 => NfkcBoundary, _ => NfkdBoundary };
        uint bad = form switch { 0 => NfcNo | NfcMaybe, 1 => NfdNo, 2 => NfkcNo | NfkcMaybe, _ => NfkdNo };
        for (int c = 0; c < 0xD800; c++)
        {
            uint info = Info(c);
            if ((info & wanted) == 0 || (info & bad) != 0 || info >> 16 != 0)
            {
                return c;
            }
        }

        return 0xD800;
    }
}
