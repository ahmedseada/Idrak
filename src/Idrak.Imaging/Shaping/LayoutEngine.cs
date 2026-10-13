// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;
using System.Runtime.InteropServices;

namespace Idrak.Imaging.Shaping;

/// <summary>What one shaping pass works on: the glyphs, the font's GDEF, and where ligature ids come from.</summary>
internal sealed class LayoutContext(IFontFace font, GlyphDefinitions definitions, List<ShapingGlyph> glyphs, bool rightToLeft)
{
    public IFontFace Font { get; } = font;

    public GlyphDefinitions Definitions { get; } = definitions;

    public List<ShapingGlyph> Glyphs { get; } = glyphs;

    public bool RightToLeft { get; } = rightToLeft;

    private int _ligatureId;

    public int NextLigatureId() => _ligatureId = _ligatureId % 255 + 1;            // HarfBuzz's ids wrap at 3 bits; 8 here

    /// <summary>A glyph's GDEF properties after a substitution put <paramref name="glyph"/> in its place.</summary>
    public void SetGlyph(ref ShapingGlyph g, int glyph, GlyphKind fallback)
    {
        g.Glyph = glyph;
        g.Substituted = true;
        if (Definitions.HasClasses)
        {
            g.Kind = Definitions.Kind(glyph);
            g.MarkClass = Definitions.MarkClass(glyph);
        }
        else if (fallback != GlyphKind.Unclassified)
        {
            g.Kind = fallback;
        }
    }
}

/// <summary>
/// Applies GSUB and GPOS lookups to a glyph run as HarfBuzz does: lookup flags and mark filtering sets, default
/// ignorables (ZWJ, ZWNJ) skipped as its matcher skips them, contexts with nested lookups, ligature components and
/// marks between them, multiple substitutions, pair, cursive and mark attachment.
/// </summary>
internal sealed class LayoutEngine(LayoutContext context, LayoutTable table, bool positioning)
{
    private const int MaxNesting = 64;

    private readonly LayoutContext _c = context;
    private readonly LayoutTable _table = table;
    private readonly bool _positioning = positioning;
    private Lookup _lookup;
    private uint _mask;
    private bool _autoZwj, _autoZwnj;
    private int _nesting;

    private List<ShapingGlyph> G => _c.Glyphs;

    private ReadOnlySpan<byte> T => _table.Data.Span;

    /// <summary>Applies <paramref name="lookup"/> over the run to the glyphs carrying <paramref name="mask"/>.</summary>
    public void Apply(Lookup lookup, uint mask, bool autoZwj = true, bool autoZwnj = true)
    {
        (_lookup, _mask, _autoZwj, _autoZwnj) = (lookup, mask, autoZwj, autoZwnj);
        if (!_positioning && lookup.Type == 8)
        {
            for (int i = G.Count - 1; i >= 0; i--)
            {
                var g = G[i];
                if ((g.Mask & mask) != 0 && !Ignored(g, lookup))
                {
                    foreach (int sub in lookup.Subtables)
                    {
                        if (ReverseChain(sub, i))
                        {
                            break;
                        }
                    }
                }
            }

            return;
        }

        for (int i = 0; i < G.Count;)
        {
            var g = G[i];
            if ((g.Mask & mask) == 0 || Ignored(g, lookup))
            {
                i++;
                continue;
            }

            int next = ApplyAt(lookup, i);
            i = next >= 0 ? next : i + 1;
        }
    }

    // The lookup's subtables at position i, the first that applies: the position after what it handled, or -1.
    private int ApplyAt(Lookup lookup, int i)
    {
        var saved = _lookup;
        _lookup = lookup;
        try
        {
            foreach (int sub in lookup.Subtables)
            {
                int next = _positioning ? PositionAt(lookup.Type, sub, i) : SubstituteAt(lookup.Type, sub, i);
                if (next >= 0)
                {
                    return next;
                }
            }

            return -1;
        }
        finally
        {
            _lookup = saved;
        }
    }

    // ---------------------------------------------------------------- matching

    private bool Ignored(in ShapingGlyph g, Lookup lookup)
    {
        int flags = lookup.Flags;
        switch (g.Kind)
        {
            case GlyphKind.Base when (flags & Lookup.IgnoreBases) != 0:
            case GlyphKind.Ligature when (flags & Lookup.IgnoreLigatures) != 0:
                return true;
            case GlyphKind.Mark:
                if ((flags & Lookup.IgnoreMarks) != 0)
                {
                    return true;
                }

                if ((flags & Lookup.UseMarkSet) != 0)
                {
                    return !_c.Definitions.InMarkSet(lookup.MarkSet, g.Glyph);
                }

                return lookup.MarkAttachment != 0 && g.MarkClass != lookup.MarkAttachment;
            default:
                return false;
        }
    }

    // A glyph a match may step over: one the lookup ignores (always), or ZWJ / ZWNJ when this match ignores them (unless it matches).
    private enum Skip
    {
        No, Yes, Maybe,
    }

    private Skip MaySkip(in ShapingGlyph g, bool contextMatch, Lookup lookup)
    {
        if (Ignored(g, lookup))
        {
            return Skip.Yes;
        }

        if (g.Ignorable)
        {
            bool ignoreZwnj = _positioning || (contextMatch && _autoZwnj);
            bool ignoreZwj = _positioning || contextMatch || _autoZwj;
            if ((g.IsZwnj && ignoreZwnj) || (g.IsZwj && ignoreZwj) || (!g.IsZwj && !g.IsZwnj))
            {
                return Skip.Maybe;
            }
        }

        return Skip.No;
    }

    // The next glyph after `from` that a match takes (forward or backward), testing it with `matches`; -1 when none.
    private int Step(int from, bool forward, bool contextMatch, Func<ShapingGlyph, bool> matches)
    {
        var lookup = _lookup;
        for (int j = forward ? from + 1 : from - 1; forward ? j < G.Count : j >= 0; j += forward ? 1 : -1)
        {
            var g = G[j];
            var skip = MaySkip(g, contextMatch, lookup);
            if (skip == Skip.Yes)
            {
                continue;
            }

            bool maskOk = contextMatch || (g.Mask & _mask) != 0;
            bool match = maskOk && matches(g);
            if (match && skip != Skip.Yes)
            {
                return j;
            }

            if (skip == Skip.No)
            {
                return -1;
            }
        }

        return -1;
    }

    // The input sequence after the first glyph (at `start`): positions of each matched glyph, or null.
    private int[]? MatchInput(int start, int count, Func<int, ShapingGlyph, bool> matches)
    {
        var positions = new int[count];
        positions[0] = start;
        int at = start;
        for (int k = 1; k < count; k++)
        {
            int kk = k;
            at = Step(at, forward: true, contextMatch: false, g => matches(kk, g));
            if (at < 0)
            {
                return null;
            }

            positions[k] = at;
        }

        return positions;
    }

    private bool MatchBacktrack(int start, int count, Func<int, ShapingGlyph, bool> matches)
    {
        int at = start;
        for (int k = 0; k < count; k++)
        {
            int kk = k;
            at = Step(at, forward: false, contextMatch: true, g => matches(kk, g));
            if (at < 0)
            {
                return false;
            }
        }

        return true;
    }

    private bool MatchLookahead(int end, int count, Func<int, ShapingGlyph, bool> matches)
    {
        int at = end;
        for (int k = 0; k < count; k++)
        {
            int kk = k;
            at = Step(at, forward: true, contextMatch: true, g => matches(kk, g));
            if (at < 0)
            {
                return false;
            }
        }

        return true;
    }

    // Nested lookups at the matched positions (sequence index, lookup index records at `records`), then the position
    // after the matched input as it stands.
    private int ApplyRecords(int[] positions, int records, int count)
    {
        var t = T;
        int end = positions[^1] + 1;
        if (_nesting >= MaxNesting)
        {
            return end;
        }

        var recordList = new (int Sequence, int Lookup)[count];
        for (int r = 0; r < count; r++)
        {
            recordList[r] = (Bytes.U16(t, records + 4 * r), Bytes.U16(t, records + 4 * r + 2));
        }

        foreach (var (sequence, lookupIndex) in recordList)
        {
            if (sequence >= positions.Length || lookupIndex >= _table.Lookups.Length)
            {
                continue;
            }

            int at = positions[sequence];
            if (at >= G.Count)
            {
                continue;
            }

            int before = G.Count;
            _nesting++;
            try
            {
                ApplyAt(_table.Lookups[lookupIndex], at);
            }
            finally
            {
                _nesting--;
            }

            int delta = G.Count - before;
            if (delta != 0)
            {
                for (int k = sequence + 1; k < positions.Length; k++)
                {
                    positions[k] = Math.Max(at + 1, positions[k] + delta);
                }

                end = Math.Max(at + 1, end + delta);
            }
        }

        return Math.Min(end, G.Count);
    }

    // ---------------------------------------------------------------- GSUB

    private int SubstituteAt(int type, int sub, int i)
    {
        var t = T;
        int glyph = G[i].Glyph;
        switch (type)
        {
            case 1:
            {
                int format = Bytes.U16(t, sub), index = Bytes.Coverage(t, sub + Bytes.U16(t, sub + 2), glyph);
                if (index < 0)
                {
                    return -1;
                }

                int replacement = format == 1 ? (glyph + Bytes.I16(t, sub + 4)) & 0xFFFF : Bytes.U16(t, sub + 6 + 2 * index);
                Replace(i, replacement);
                return i + 1;
            }

            case 2:
            {
                int index = Bytes.Coverage(t, sub + Bytes.U16(t, sub + 2), glyph);
                if (index < 0 || index >= Bytes.U16(t, sub + 4))
                {
                    return -1;
                }

                int sequence = sub + Bytes.U16(t, sub + 6 + 2 * index);
                int count = Bytes.U16(t, sequence);
                if (count == 1)
                {
                    Replace(i, Bytes.U16(t, sequence + 2));
                    return i + 1;
                }

                var source = G[i];
                G.RemoveAt(i);
                for (int k = 0; k < count; k++)
                {
                    var part = source;
                    _c.SetGlyph(ref part, Bytes.U16(t, sequence + 2 + 2 * k), GlyphKind.Unclassified);
                    part.Multiplied = true;
                    part.LigatureId = 0;
                    part.LigatureComponent = k + 1;
                    G.Insert(i + k, part);
                }

                return i + count;
            }

            case 3:
            {
                int index = Bytes.Coverage(t, sub + Bytes.U16(t, sub + 2), glyph);
                if (index < 0 || index >= Bytes.U16(t, sub + 4))
                {
                    return -1;
                }

                int set = sub + Bytes.U16(t, sub + 6 + 2 * index);
                if (Bytes.U16(t, set) == 0)
                {
                    return -1;
                }

                Replace(i, Bytes.U16(t, set + 2));                              // the first alternate (feature value 1)
                return i + 1;
            }

            case 4:
                return Ligate(sub, i);
            case 5:
                return Context(sub, i);
            case 6:
                return ChainContext(sub, i);
            default:
                return -1;
        }
    }

    private void Replace(int i, int glyph)
    {
        var span = CollectionsMarshal.AsSpan(G);
        _c.SetGlyph(ref span[i], glyph, GlyphKind.Unclassified);
    }

    private int Ligate(int sub, int i)
    {
        var t = T;
        int index = Bytes.Coverage(t, sub + Bytes.U16(t, sub + 2), G[i].Glyph);
        if (index < 0 || index >= Bytes.U16(t, sub + 4))
        {
            return -1;
        }

        int set = sub + Bytes.U16(t, sub + 6 + 2 * index);
        int ligatures = Bytes.U16(t, set);
        for (int l = 0; l < ligatures; l++)
        {
            int ligature = set + Bytes.U16(t, set + 2 + 2 * l);
            int components = Bytes.U16(t, ligature + 2);
            int lig = ligature;
            var positions = components == 1 ? [i] : MatchInput(i, components, (k, g) => g.Glyph == Bytes.U16(T, lig + 4 + 2 * (k - 1)));
            if (positions is null)
            {
                continue;
            }

            MakeLigature(positions, Bytes.U16(t, ligature));
            return i + 1 + CountBetween(positions);
        }

        return -1;
    }

    private static int CountBetween(int[] positions) => positions[^1] - positions[0] - (positions.Length - 1);

    // The first component becomes the ligature; the other components go; marks between them stay after it, each told
    // which component it sat on; every glyph of the span takes the span's first cluster.
    private void MakeLigature(int[] positions, int ligatureGlyph)
    {
        var span = CollectionsMarshal.AsSpan(G);
        int first = positions[0], last = positions[^1];
        bool marksOnly = true;
        foreach (int p in positions)
        {
            marksOnly &= span[p].IsMark;
        }

        int cluster = int.MaxValue;
        for (int k = first; k <= last; k++)
        {
            cluster = Math.Min(cluster, span[k].Cluster);
        }

        int ligatureId = marksOnly ? 0 : _c.NextLigatureId();
        int componentsSoFar = span[first].Components;
        int lastComponents = span[first].Components;
        int lastId = span[first].LigatureId;
        ref var head = ref span[first];
        _c.SetGlyph(ref head, ligatureGlyph, marksOnly ? GlyphKind.Mark : GlyphKind.Ligature);
        int totalComponents = 0;
        foreach (int p in positions)
        {
            totalComponents += Math.Max(1, span[p].Components);
        }

        head.Components = totalComponents;
        head.LigatureId = ligatureId;
        head.LigatureComponent = 0;
        head.Cluster = cluster;

        var keep = new List<int>();
        for (int k = 1; k < positions.Length; k++)
        {
            for (int m = positions[k - 1] + 1; m < positions[k]; m++)
            {
                ref var mark = ref span[m];
                if (ligatureId != 0)
                {
                    int component = mark.LigatureComponent == 0 ? lastComponents : mark.LigatureComponent;
                    mark.LigatureComponent = componentsSoFar - lastComponents + Math.Min(component, lastComponents);
                    mark.LigatureId = ligatureId;
                }

                mark.Cluster = cluster;
                keep.Add(m);
            }

            lastId = span[positions[k]].LigatureId;
            lastComponents = Math.Max(1, span[positions[k]].Components);
            componentsSoFar += lastComponents;
        }

        // Marks right after the ligature that belonged to its last component follow it there.
        if (ligatureId != 0 && lastId != 0)
        {
            for (int m = last + 1; m < span.Length && span[m].LigatureId == lastId && span[m].LigatureComponent != 0; m++)
            {
                ref var mark = ref span[m];
                mark.LigatureComponent = componentsSoFar - lastComponents + Math.Min(mark.LigatureComponent, lastComponents);
                mark.LigatureId = ligatureId;
            }
        }

        // Remove the other components (from the end, so positions stay right).
        for (int k = positions.Length - 1; k >= 1; k--)
        {
            G.RemoveAt(positions[k]);
        }
    }

    private int Context(int sub, int i)
    {
        var t = T;
        int format = Bytes.U16(t, sub);
        int glyph = G[i].Glyph;
        switch (format)
        {
            case 1:
            {
                int index = Bytes.Coverage(t, sub + Bytes.U16(t, sub + 2), glyph);
                if (index < 0 || index >= Bytes.U16(t, sub + 4))
                {
                    return -1;
                }

                int set = sub + Bytes.U16(t, sub + 6 + 2 * index);
                int rules = Bytes.U16(t, set);
                for (int r = 0; r < rules; r++)
                {
                    int rule = set + Bytes.U16(t, set + 2 + 2 * r);
                    int count = Bytes.U16(t, rule), records = Bytes.U16(t, rule + 2);
                    int ruleAt = rule;
                    var positions = MatchInput(i, Math.Max(1, count), (k, g) => g.Glyph == Bytes.U16(T, ruleAt + 4 + 2 * (k - 1)));
                    if (positions is not null)
                    {
                        return ApplyRecords(positions, rule + 4 + 2 * (count - 1), records);
                    }
                }

                return -1;
            }

            case 2:
            {
                if (Bytes.Coverage(t, sub + Bytes.U16(t, sub + 2), glyph) < 0)
                {
                    return -1;
                }

                int classes = sub + Bytes.U16(t, sub + 4);
                int first = Bytes.Class(t, classes, glyph);
                if (first >= Bytes.U16(t, sub + 6) || Bytes.U16(t, sub + 8 + 2 * first) == 0)
                {
                    return -1;
                }

                int set = sub + Bytes.U16(t, sub + 8 + 2 * first);
                int rules = Bytes.U16(t, set);
                for (int r = 0; r < rules; r++)
                {
                    int rule = set + Bytes.U16(t, set + 2 + 2 * r);
                    int count = Bytes.U16(t, rule), records = Bytes.U16(t, rule + 2);
                    int ruleAt = rule;
                    var positions = MatchInput(i, Math.Max(1, count), (k, g) => Bytes.Class(T, classes, g.Glyph) == Bytes.U16(T, ruleAt + 4 + 2 * (k - 1)));
                    if (positions is not null)
                    {
                        return ApplyRecords(positions, rule + 4 + 2 * (count - 1), records);
                    }
                }

                return -1;
            }

            case 3:
            {
                int count = Bytes.U16(t, sub + 2), records = Bytes.U16(t, sub + 4);
                if (count == 0 || Bytes.Coverage(t, sub + Bytes.U16(t, sub + 6), glyph) < 0)
                {
                    return -1;
                }

                var positions = MatchInput(i, count, (k, g) => Bytes.Coverage(T, sub + Bytes.U16(T, sub + 6 + 2 * k), g.Glyph) >= 0);
                return positions is null ? -1 : ApplyRecords(positions, sub + 6 + 2 * count, records);
            }

            default:
                return -1;
        }
    }

    private int ChainContext(int sub, int i)
    {
        var t = T;
        int format = Bytes.U16(t, sub);
        int glyph = G[i].Glyph;
        switch (format)
        {
            case 1:
            {
                int index = Bytes.Coverage(t, sub + Bytes.U16(t, sub + 2), glyph);
                if (index < 0 || index >= Bytes.U16(t, sub + 4))
                {
                    return -1;
                }

                int set = sub + Bytes.U16(t, sub + 6 + 2 * index);
                int rules = Bytes.U16(t, set);
                for (int r = 0; r < rules; r++)
                {
                    int rule = set + Bytes.U16(t, set + 2 + 2 * r);
                    int result = ChainRule(rule, i, (_, k, g) => g.Glyph == k);
                    if (result >= 0)
                    {
                        return result;
                    }
                }

                return -1;
            }

            case 2:
            {
                if (Bytes.Coverage(t, sub + Bytes.U16(t, sub + 2), glyph) < 0)
                {
                    return -1;
                }

                int backtrackClasses = Bytes.U16(t, sub + 4) is int b and > 0 ? sub + b : 0;
                int inputClasses = Bytes.U16(t, sub + 6) is int n and > 0 ? sub + n : 0;
                int lookaheadClasses = Bytes.U16(t, sub + 8) is int a and > 0 ? sub + a : 0;
                int first = Bytes.Class(t, inputClasses, glyph);
                if (first >= Bytes.U16(t, sub + 10) || Bytes.U16(t, sub + 12 + 2 * first) == 0)
                {
                    return -1;
                }

                int set = sub + Bytes.U16(t, sub + 12 + 2 * first);
                int rules = Bytes.U16(t, set);
                for (int r = 0; r < rules; r++)
                {
                    int rule = set + Bytes.U16(t, set + 2 + 2 * r);
                    int result = ChainRule(rule, i, (part, value, g) =>
                        Bytes.Class(T, part switch { 0 => backtrackClasses, 1 => inputClasses, _ => lookaheadClasses }, g.Glyph) == value);
                    if (result >= 0)
                    {
                        return result;
                    }
                }

                return -1;
            }

            case 3:
            {
                int at = sub + 2;
                int backtrack = Bytes.U16(t, at), backtrackAt = at + 2;
                at = backtrackAt + 2 * backtrack;
                int input = Bytes.U16(t, at), inputAt = at + 2;
                at = inputAt + 2 * input;
                int lookahead = Bytes.U16(t, at), lookaheadAt = at + 2;
                at = lookaheadAt + 2 * lookahead;
                int records = Bytes.U16(t, at);
                if (input == 0 || Bytes.Coverage(t, sub + Bytes.U16(t, inputAt), glyph) < 0)
                {
                    return -1;
                }

                bool Covered(int list, int k, ShapingGlyph g) => Bytes.Coverage(T, sub + Bytes.U16(T, list + 2 * k), g.Glyph) >= 0;
                var positions = MatchInput(i, input, (k, g) => Covered(inputAt, k, g));
                if (positions is null || !MatchBacktrack(i, backtrack, (k, g) => Covered(backtrackAt, k, g))
                    || !MatchLookahead(positions[^1], lookahead, (k, g) => Covered(lookaheadAt, k, g)))
                {
                    return -1;
                }

                return ApplyRecords(positions, at + 2, records);
            }

            default:
                return -1;
        }
    }

    // A chain rule of format 1 or 2 (glyphs or classes): `matches(part, value, glyph)` with part 0 backtrack, 1 input, 2 lookahead.
    private int ChainRule(int rule, int i, Func<int, int, ShapingGlyph, bool> matches)
    {
        var t = T;
        int at = rule;
        int backtrack = Bytes.U16(t, at), backtrackAt = at + 2;
        at = backtrackAt + 2 * backtrack;
        int input = Bytes.U16(t, at), inputAt = at + 2;
        at = inputAt + 2 * Math.Max(0, input - 1);
        int lookahead = Bytes.U16(t, at), lookaheadAt = at + 2;
        at = lookaheadAt + 2 * lookahead;
        int records = Bytes.U16(t, at);
        var positions = MatchInput(i, Math.Max(1, input), (k, g) => matches(1, Bytes.U16(T, inputAt + 2 * (k - 1)), g));
        if (positions is null || !MatchBacktrack(i, backtrack, (k, g) => matches(0, Bytes.U16(T, backtrackAt + 2 * k), g))
            || !MatchLookahead(positions[^1], lookahead, (k, g) => matches(2, Bytes.U16(T, lookaheadAt + 2 * k), g)))
        {
            return -1;
        }

        return ApplyRecords(positions, at + 2, records);
    }

    private bool ReverseChain(int sub, int i)
    {
        var t = T;
        int glyph = G[i].Glyph;
        int index = Bytes.Coverage(t, sub + Bytes.U16(t, sub + 2), glyph);
        if (index < 0)
        {
            return false;
        }

        int at = sub + 4;
        int backtrack = Bytes.U16(t, at), backtrackAt = at + 2;
        at = backtrackAt + 2 * backtrack;
        int lookahead = Bytes.U16(t, at), lookaheadAt = at + 2;
        at = lookaheadAt + 2 * lookahead;
        int count = Bytes.U16(t, at);
        if (index >= count)
        {
            return false;
        }

        bool Covered(int list, int k, ShapingGlyph g) => Bytes.Coverage(T, sub + Bytes.U16(T, list + 2 * k), g.Glyph) >= 0;
        if (!MatchBacktrack(i, backtrack, (k, g) => Covered(backtrackAt, k, g)) || !MatchLookahead(i, lookahead, (k, g) => Covered(lookaheadAt, k, g)))
        {
            return false;
        }

        Replace(i, Bytes.U16(t, at + 2 + 2 * index));
        return true;
    }

    // ---------------------------------------------------------------- GPOS

    private int PositionAt(int type, int sub, int i)
    {
        switch (type)
        {
            case 1:
                return Single(sub, i);
            case 2:
                return Pair(sub, i);
            case 3:
                return Cursive(sub, i);
            case 4:
                return MarkToBase(sub, i);
            case 5:
                return MarkToLigature(sub, i);
            case 6:
                return MarkToMark(sub, i);
            case 7:
                return Context(sub, i);
            case 8:
                return ChainContext(sub, i);
            default:
                return -1;
        }
    }

    private static int ValueSize(int format) => 2 * BitOperations.PopCount((uint)(format & 0xFF));

    // A value record's placements and horizontal advance (device tables and vertical advance are not used for horizontal text).
    private void AddValue(int glyphIndex, int format, int at)
    {
        var t = T;
        var span = CollectionsMarshal.AsSpan(G);
        ref var g = ref span[glyphIndex];
        if ((format & 1) != 0)
        {
            g.XOffset += Bytes.I16(t, at);
            at += 2;
        }

        if ((format & 2) != 0)
        {
            g.YOffset += Bytes.I16(t, at);
            at += 2;
        }

        if ((format & 4) != 0)
        {
            g.XAdvance += Bytes.I16(t, at);
        }
    }

    private int Single(int sub, int i)
    {
        var t = T;
        int format = Bytes.U16(t, sub), valueFormat = Bytes.U16(t, sub + 4);
        int index = Bytes.Coverage(t, sub + Bytes.U16(t, sub + 2), G[i].Glyph);
        if (index < 0)
        {
            return -1;
        }

        AddValue(i, valueFormat, format == 1 ? sub + 6 : sub + 8 + index * ValueSize(valueFormat));
        return i + 1;
    }

    private int Pair(int sub, int i)
    {
        var t = T;
        int format = Bytes.U16(t, sub);
        int index = Bytes.Coverage(t, sub + Bytes.U16(t, sub + 2), G[i].Glyph);
        if (index < 0)
        {
            return -1;
        }

        int j = Step(i, forward: true, contextMatch: false, _ => true);
        if (j < 0)
        {
            return -1;
        }

        int format1 = Bytes.U16(t, sub + 4), format2 = Bytes.U16(t, sub + 6);
        int size1 = ValueSize(format1), size2 = ValueSize(format2);
        int second = G[j].Glyph;
        if (format == 1)
        {
            if (index >= Bytes.U16(t, sub + 8))
            {
                return -1;
            }

            int set = sub + Bytes.U16(t, sub + 10 + 2 * index);
            int count = Bytes.U16(t, set), record = 2 + size1 + size2;
            int lo = 0, hi = count - 1;
            while (lo <= hi)
            {
                int mid = (lo + hi) >>> 1, at = set + 2 + record * mid;
                int glyph = Bytes.U16(t, at);
                if (glyph == second)
                {
                    AddValue(i, format1, at + 2);
                    AddValue(j, format2, at + 2 + size1);
                    return format2 != 0 ? j + 1 : j;
                }

                if (glyph < second)
                {
                    lo = mid + 1;
                }
                else
                {
                    hi = mid - 1;
                }
            }

            return -1;
        }

        if (format == 2)
        {
            int class1 = Bytes.Class(t, sub + Bytes.U16(t, sub + 8), G[i].Glyph), class2 = Bytes.Class(t, sub + Bytes.U16(t, sub + 10), second);
            int count1 = Bytes.U16(t, sub + 12), count2 = Bytes.U16(t, sub + 14);
            if (class1 >= count1 || class2 >= count2)
            {
                return -1;
            }

            int at = sub + 16 + (class1 * count2 + class2) * (size1 + size2);
            AddValue(i, format1, at);
            AddValue(j, format2, at + size1);
            return format2 != 0 ? j + 1 : j;
        }

        return -1;
    }

    private (int X, int Y)? Anchor(int at)
    {
        if (at <= 0)
        {
            return null;
        }

        var t = T;
        return (Bytes.I16(t, at + 2), Bytes.I16(t, at + 4));
    }

    private int Cursive(int sub, int i)
    {
        var t = T;
        int coverage = sub + Bytes.U16(t, sub + 2), count = Bytes.U16(t, sub + 4);
        int thisIndex = Bytes.Coverage(t, coverage, G[i].Glyph);
        if (thisIndex < 0 || thisIndex >= count)
        {
            return -1;
        }

        int entryOffset = Bytes.U16(t, sub + 6 + 4 * thisIndex);
        if (entryOffset == 0)
        {
            return -1;
        }

        int p = Step(i, forward: false, contextMatch: false, _ => true);
        if (p < 0)
        {
            return -1;
        }

        int prevIndex = Bytes.Coverage(t, coverage, G[p].Glyph);
        if (prevIndex < 0 || prevIndex >= count)
        {
            return -1;
        }

        int exitOffset = Bytes.U16(t, sub + 6 + 4 * prevIndex + 2);
        if (exitOffset == 0)
        {
            return -1;
        }

        var (exitX, exitY) = Anchor(sub + exitOffset)!.Value;
        var (entryX, entryY) = Anchor(sub + entryOffset)!.Value;
        var span = CollectionsMarshal.AsSpan(G);
        ref var prev = ref span[p];
        ref var cur = ref span[i];
        if (!_c.RightToLeft)
        {
            prev.XAdvance = exitX + prev.XOffset;
            int d = entryX + cur.XOffset;
            cur.XAdvance -= d;
            cur.XOffset -= d;
        }
        else
        {
            int d = exitX + prev.XOffset;
            prev.XAdvance -= d;
            prev.XOffset -= d;
            cur.XAdvance = entryX + cur.XOffset;
        }

        // The child hangs on its parent: the earlier glyph on the later one for a right-to-left lookup, else the reverse.
        int child = p, parent = i, yOffset = entryY - exitY;
        if ((_lookup.Flags & Lookup.RightToLeft) == 0)
        {
            (child, parent) = (parent, child);
            yOffset = -yOffset;
        }

        ReverseCursiveChain(child, parent);
        span[child].Attach = AttachKind.Cursive;
        span[child].AttachChain = parent - child;
        span[child].YOffset = yOffset;
        if (span[parent].AttachChain == -span[child].AttachChain)
        {
            span[parent].AttachChain = 0;
            if (span[parent].Attach == AttachKind.Cursive)
            {
                span[parent].YOffset = 0;
            }

            span[parent].Attach = AttachKind.None;
        }

        return i + 1;
    }

    // A child already hung on another glyph: its old chain is turned round so the whole tree hangs on the new parent.
    private void ReverseCursiveChain(int i, int newParent)
    {
        var span = CollectionsMarshal.AsSpan(G);
        int chain = span[i].AttachChain;
        if (chain == 0 || span[i].Attach != AttachKind.Cursive)
        {
            return;
        }

        span[i].AttachChain = 0;
        int j = i + chain;
        if (j == newParent)
        {
            return;
        }

        ReverseCursiveChain(j, newParent);
        span[j].YOffset = -span[i].YOffset;
        span[j].AttachChain = -chain;
        span[j].Attach = AttachKind.Cursive;
    }

    // The glyph before a mark that it may attach to: walking back over marks (and ZWJ, ZWNJ).
    private int PreviousNonMark(int i)
    {
        for (int j = i - 1; j >= 0; j--)
        {
            var g = G[j];
            if (g.IsMark || g.Ignorable)
            {
                continue;
            }

            return j;
        }

        return -1;
    }

    private int MarkToBase(int sub, int i)
    {
        var t = T;
        int markIndex = Bytes.Coverage(t, sub + Bytes.U16(t, sub + 2), G[i].Glyph);
        if (markIndex < 0)
        {
            return -1;
        }

        int b = PreviousNonMark(i);
        if (b < 0)
        {
            return -1;
        }

        int baseIndex = Bytes.Coverage(t, sub + Bytes.U16(t, sub + 4), G[b].Glyph);
        if (baseIndex < 0)
        {
            return -1;
        }

        int classCount = Bytes.U16(t, sub + 6), markArray = sub + Bytes.U16(t, sub + 8), baseArray = sub + Bytes.U16(t, sub + 10);
        if (baseIndex >= Bytes.U16(t, baseArray))
        {
            return -1;
        }

        return MarkWithAnchors(markArray, markIndex, baseArray, baseArray + 2 + 2 * classCount * baseIndex, classCount, b, i);
    }

    private int MarkToLigature(int sub, int i)
    {
        var t = T;
        int markIndex = Bytes.Coverage(t, sub + Bytes.U16(t, sub + 2), G[i].Glyph);
        if (markIndex < 0)
        {
            return -1;
        }

        int l = PreviousNonMark(i);
        if (l < 0)
        {
            return -1;
        }

        int ligatureIndex = Bytes.Coverage(t, sub + Bytes.U16(t, sub + 4), G[l].Glyph);
        if (ligatureIndex < 0)
        {
            return -1;
        }

        int classCount = Bytes.U16(t, sub + 6), markArray = sub + Bytes.U16(t, sub + 8), ligatureArray = sub + Bytes.U16(t, sub + 10);
        if (ligatureIndex >= Bytes.U16(t, ligatureArray))
        {
            return -1;
        }

        int attach = ligatureArray + Bytes.U16(t, ligatureArray + 2 + 2 * ligatureIndex);
        int components = Bytes.U16(t, attach);
        if (components == 0)
        {
            return -1;
        }

        var lig = G[l];
        var mark = G[i];
        int component = lig.LigatureId != 0 && lig.LigatureId == mark.LigatureId && mark.LigatureComponent > 0
            ? Math.Min(components, mark.LigatureComponent) - 1 : components - 1;
        return MarkWithAnchors(markArray, markIndex, attach, attach + 2 + 2 * classCount * component, classCount, l, i);
    }

    private int MarkToMark(int sub, int i)
    {
        var t = T;
        int mark1Index = Bytes.Coverage(t, sub + Bytes.U16(t, sub + 2), G[i].Glyph);
        if (mark1Index < 0)
        {
            return -1;
        }

        // The mark before, skipping only what the lookup's mark filtering skips (not marks as such).
        var lookup = _lookup with { Flags = _lookup.Flags & ~(Lookup.IgnoreBases | Lookup.IgnoreLigatures | Lookup.IgnoreMarks) };
        int j = -1;
        for (int k = i - 1; k >= 0; k--)
        {
            var g = G[k];
            if (Ignored(g, lookup) || g.Ignorable)
            {
                continue;
            }

            j = k;
            break;
        }

        if (j < 0 || !G[j].IsMark)
        {
            return -1;
        }

        var cur = G[i];
        var prev = G[j];
        bool good = cur.LigatureId == prev.LigatureId
            ? cur.LigatureId == 0 || cur.LigatureComponent == prev.LigatureComponent
            : (cur.LigatureId > 0 && cur.LigatureComponent == 0) || (prev.LigatureId > 0 && prev.LigatureComponent == 0);
        if (!good)
        {
            return -1;
        }

        int mark2Index = Bytes.Coverage(t, sub + Bytes.U16(t, sub + 4), prev.Glyph);
        if (mark2Index < 0)
        {
            return -1;
        }

        int classCount = Bytes.U16(t, sub + 6), mark1Array = sub + Bytes.U16(t, sub + 8), mark2Array = sub + Bytes.U16(t, sub + 10);
        if (mark2Index >= Bytes.U16(t, mark2Array))
        {
            return -1;
        }

        return MarkWithAnchors(mark1Array, mark1Index, mark2Array, mark2Array + 2 + 2 * classCount * mark2Index, classCount, j, i);
    }

    // The mark's anchor from its MarkArray, the target's from its record (offsets relative to `recordBase`).
    private int MarkWithAnchors(int markArray, int markIndex, int recordBase, int record, int classCount, int target, int i)
    {
        var t = T;
        if (markIndex >= Bytes.U16(t, markArray))
        {
            return -1;
        }

        int markClass = Bytes.U16(t, markArray + 2 + 4 * markIndex);
        int markAnchorOffset = Bytes.U16(t, markArray + 2 + 4 * markIndex + 2);
        if (markClass >= classCount || markAnchorOffset == 0)
        {
            return -1;
        }

        int targetAnchorOffset = Bytes.U16(t, record + 2 * markClass);
        if (targetAnchorOffset == 0)
        {
            return -1;
        }

        var mark = Anchor(markArray + markAnchorOffset)!.Value;
        var anchor = Anchor(recordBase + targetAnchorOffset)!.Value;
        var span = CollectionsMarshal.AsSpan(G);
        ref var g = ref span[i];
        g.XOffset = anchor.X - mark.X;
        g.YOffset = anchor.Y - mark.Y;
        g.Attach = AttachKind.Mark;
        g.AttachChain = target - i;
        return i + 1;
    }
}
