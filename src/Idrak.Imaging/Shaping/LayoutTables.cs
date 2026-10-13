// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Runtime.CompilerServices;

namespace Idrak.Imaging.Shaping;

/// <summary>One glyph while shaping: what it is, where it came from, how the lookups see it, and where it goes.</summary>
internal struct ShapingGlyph
{
    public int Glyph;
    public int Cluster;            // UTF-16 index of its first character in the text
    public int CodePoint;          // the character it was mapped from (the first component's, for a ligature)
    public uint Mask;              // the features that apply to it
    public GlyphKind Kind;         // GDEF's class (or one made from the character when the font has none)
    public ushort MarkClass;       // GDEF's mark attachment class
    public int LigatureId;         // which ligature a mark (or the ligature) belongs to; 0 for none
    public int LigatureComponent;  // the component of that ligature a mark sits on (1 based); a multiple substitution's part
    public int Components;         // components of a ligature glyph (1 for others)
    public bool Multiplied;        // made by a multiple substitution
    public bool Substituted;
    public int XAdvance, YAdvance, XOffset, YOffset;
    public int AttachChain;        // the glyph it is attached to, relative (mark or cursive), 0 for none
    public AttachKind Attach;
    public byte Combining;         // (modified) canonical combining class of its character
    public bool Ignorable;         // a default-ignorable character (ZWJ, ZWNJ, bidi controls): hidden after shaping

    public readonly bool IsMark => Kind == GlyphKind.Mark;

    public readonly bool IsZwj => CodePoint == 0x200D;

    public readonly bool IsZwnj => CodePoint == 0x200C;
}

internal enum GlyphKind : byte
{
    Unclassified = 0, Base = 1, Ligature = 2, Mark = 3, Component = 4,
}

internal enum AttachKind : byte
{
    None = 0, Mark = 1, Cursive = 2,
}

/// <summary>Big-endian reads over a table's bytes; an offset outside the table reads as zero (a damaged font draws less, it does not throw).</summary>
internal static class Bytes
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int U16(ReadOnlySpan<byte> t, int at) => (uint)at <= (uint)(t.Length - 2) ? BinaryPrimitives.ReadUInt16BigEndian(t[at..]) : 0;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int I16(ReadOnlySpan<byte> t, int at) => (uint)at <= (uint)(t.Length - 2) ? BinaryPrimitives.ReadInt16BigEndian(t[at..]) : 0;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int U32(ReadOnlySpan<byte> t, int at) => (uint)at <= (uint)(t.Length - 4) ? (int)BinaryPrimitives.ReadUInt32BigEndian(t[at..]) : 0;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static uint Tag(ReadOnlySpan<byte> t, int at) => (uint)at <= (uint)(t.Length - 4) ? BinaryPrimitives.ReadUInt32BigEndian(t[at..]) : 0;

    public static uint Tag(string tag) => (uint)(tag[0] << 24 | tag[1] << 16 | tag[2] << 8 | tag[3]);

    /// <summary>The coverage index of <paramref name="glyph"/> in the coverage table at <paramref name="at"/>, or -1.</summary>
    public static int Coverage(ReadOnlySpan<byte> t, int at, int glyph)
    {
        int format = U16(t, at);
        if (format == 1)
        {
            int lo = 0, hi = U16(t, at + 2) - 1;
            while (lo <= hi)
            {
                int mid = (lo + hi) >>> 1, g = U16(t, at + 4 + 2 * mid);
                if (g == glyph)
                {
                    return mid;
                }

                if (g < glyph)
                {
                    lo = mid + 1;
                }
                else
                {
                    hi = mid - 1;
                }
            }
        }
        else if (format == 2)
        {
            int lo = 0, hi = U16(t, at + 2) - 1;
            while (lo <= hi)
            {
                int mid = (lo + hi) >>> 1, r = at + 4 + 6 * mid;
                int start = U16(t, r), end = U16(t, r + 2);
                if (glyph < start)
                {
                    hi = mid - 1;
                }
                else if (glyph > end)
                {
                    lo = mid + 1;
                }
                else
                {
                    return U16(t, r + 4) + glyph - start;
                }
            }
        }

        return -1;
    }

    /// <summary>The class of <paramref name="glyph"/> in the class definition at <paramref name="at"/> (0 when not listed or no table).</summary>
    public static int Class(ReadOnlySpan<byte> t, int at, int glyph)
    {
        if (at <= 0)
        {
            return 0;
        }

        int format = U16(t, at);
        if (format == 1)
        {
            int start = U16(t, at + 2), count = U16(t, at + 4);
            return glyph >= start && glyph < start + count ? U16(t, at + 6 + 2 * (glyph - start)) : 0;
        }

        if (format == 2)
        {
            int lo = 0, hi = U16(t, at + 2) - 1;
            while (lo <= hi)
            {
                int mid = (lo + hi) >>> 1, r = at + 4 + 6 * mid;
                int start = U16(t, r), end = U16(t, r + 2);
                if (glyph < start)
                {
                    hi = mid - 1;
                }
                else if (glyph > end)
                {
                    lo = mid + 1;
                }
                else
                {
                    return U16(t, r + 4);
                }
            }
        }

        return 0;
    }
}

/// <summary>GDEF: glyph classes, mark attachment classes and mark glyph sets.</summary>
internal sealed class GlyphDefinitions
{
    private readonly ReadOnlyMemory<byte> _table;
    private readonly int _classes, _markClasses, _markSets;

    public GlyphDefinitions(ReadOnlyMemory<byte> table)
    {
        _table = table;
        var t = table.Span;
        if (t.Length < 12)
        {
            return;
        }

        _classes = Bytes.U16(t, 4);
        _markClasses = Bytes.U16(t, 10);
        int minor = Bytes.U16(t, 2);
        _markSets = minor >= 2 && t.Length >= 14 ? Bytes.U16(t, 12) : 0;
    }

    /// <summary>Whether the font classes its glyphs (else classes come from the characters).</summary>
    public bool HasClasses => _classes != 0;

    public GlyphKind Kind(int glyph) => (GlyphKind)Bytes.Class(_table.Span, _classes, glyph);

    public ushort MarkClass(int glyph) => (ushort)Bytes.Class(_table.Span, _markClasses, glyph);

    /// <summary>Whether <paramref name="glyph"/> is in mark glyph set <paramref name="set"/>.</summary>
    public bool InMarkSet(int set, int glyph)
    {
        var t = _table.Span;
        if (_markSets == 0 || set >= Bytes.U16(t, _markSets + 2))
        {
            return false;
        }

        int coverage = Bytes.U32(t, _markSets + 4 + 4 * set);
        return coverage != 0 && Bytes.Coverage(t, _markSets + coverage, glyph) >= 0;
    }
}

/// <summary>A lookup of GSUB or GPOS: its type (extensions unwrapped per subtable), flags and subtables (absolute offsets).</summary>
internal readonly record struct Lookup(int Index, int Type, int Flags, int MarkSet, int[] Subtables)
{
    public const int RightToLeft = 1, IgnoreBases = 2, IgnoreLigatures = 4, IgnoreMarks = 8, UseMarkSet = 0x10;

    public int MarkAttachment => Flags >> 8;
}

/// <summary>
/// GSUB or GPOS read for one font: the scripts, languages and features it has, and its lookups (parsed once, when the
/// font is first shaped).
/// </summary>
internal sealed class LayoutTable
{
    private readonly int _scripts, _features;

    public LayoutTable(ReadOnlyMemory<byte> table, bool positioning)
    {
        Data = table;
        var t = table.Span;
        if (t.Length < 10)
        {
            Lookups = [];
            return;
        }

        _scripts = Bytes.U16(t, 4);
        _features = Bytes.U16(t, 6);
        int lookups = Bytes.U16(t, 8);
        int count = Bytes.U16(t, lookups);
        Lookups = new Lookup[count];
        int extension = positioning ? 9 : 7;
        for (int i = 0; i < count; i++)
        {
            int at = lookups + Bytes.U16(t, lookups + 2 + 2 * i);
            int type = Bytes.U16(t, at), flags = Bytes.U16(t, at + 2), subtables = Bytes.U16(t, at + 4);
            var offsets = new int[subtables];
            for (int s = 0; s < subtables; s++)
            {
                int sub = at + Bytes.U16(t, at + 6 + 2 * s);
                if (type == extension && Bytes.U16(t, sub) == 1)
                {
                    type = Bytes.U16(t, sub + 2);                                // every subtable of an extension names the same type
                    sub += Bytes.U32(t, sub + 4);
                }

                offsets[s] = sub;
            }

            int markSet = (flags & Lookup.UseMarkSet) != 0 ? Bytes.U16(t, at + 6 + 2 * subtables) : 0;
            Lookups[i] = new Lookup(i, type, flags, markSet, offsets);
        }
    }

    public ReadOnlyMemory<byte> Data { get; }

    public Lookup[] Lookups { get; }

    /// <summary>
    /// The feature indices of the language system for the first of <paramref name="scriptTags"/> the table has (else
    /// DFLT, dflt, latn), for <paramref name="languageTag"/> when given and present, else the script's default; and its
    /// required feature (-1 for none). Null when the table has no usable script.
    /// </summary>
    public (int[] Features, int Required)? LanguageSystem(IReadOnlyList<string> scriptTags, string? languageTag)
    {
        var t = Data.Span;
        if (_scripts == 0)
        {
            return null;
        }

        int count = Bytes.U16(t, _scripts);
        int script = -1;
        foreach (string tag in scriptTags.Concat(["DFLT", "dflt", "latn"]))
        {
            uint wanted = Bytes.Tag(tag);
            for (int i = 0; i < count && script < 0; i++)
            {
                if (Bytes.Tag(t, _scripts + 2 + 6 * i) == wanted)
                {
                    script = _scripts + Bytes.U16(t, _scripts + 2 + 6 * i + 4);
                }
            }

            if (script >= 0)
            {
                break;
            }
        }

        if (script < 0)
        {
            return null;
        }

        int langSys = Bytes.U16(t, script) is int d and > 0 ? script + d : 0;
        if (languageTag is not null)
        {
            uint wanted = Bytes.Tag(languageTag);
            int languages = Bytes.U16(t, script + 2);
            for (int i = 0; i < languages; i++)
            {
                if (Bytes.Tag(t, script + 4 + 6 * i) == wanted)
                {
                    langSys = script + Bytes.U16(t, script + 4 + 6 * i + 4);
                    break;
                }
            }
        }

        if (langSys == 0)
        {
            return ([], -1);
        }

        int required = Bytes.U16(t, langSys + 2);
        int features = Bytes.U16(t, langSys + 4);
        var indices = new int[features];
        for (int i = 0; i < features; i++)
        {
            indices[i] = Bytes.U16(t, langSys + 6 + 2 * i);
        }

        return (indices, required == 0xFFFF ? -1 : required);
    }

    /// <summary>The tag of feature <paramref name="index"/>.</summary>
    public uint FeatureTag(int index) => Bytes.Tag(Data.Span, _features + 2 + 6 * index);

    /// <summary>The lookup indices of feature <paramref name="index"/>.</summary>
    public int[] FeatureLookups(int index)
    {
        var t = Data.Span;
        int feature = _features + Bytes.U16(t, _features + 2 + 6 * index + 4);
        int count = Bytes.U16(t, feature + 2);
        var lookups = new int[count];
        for (int i = 0; i < count; i++)
        {
            lookups[i] = Bytes.U16(t, feature + 4 + 2 * i);
        }

        return lookups;
    }
}
