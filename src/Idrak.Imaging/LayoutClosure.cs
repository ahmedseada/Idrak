// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Imaging.Shaping;

namespace Idrak.Imaging;

/// <summary>
/// What a font's layout tables reach, read from their bytes: every glyph GSUB can make from a set of glyphs (the closure
/// a subset keeps, as fontTools' subsetter computes it: the lookups the scripts' features list, applied again and again
/// until nothing is added, contextual rules only where every glyph of their context can be there, each nested lookup on
/// the glyphs its place can hold), and the longest context any GSUB or GPOS lookup reads (OS/2's usMaxContext, as
/// fontTools' maxContextCalc counts it). Offsets outside a table read as zero, so a damaged table adds less, never throws.
/// </summary>
internal sealed class LayoutClosure
{
    private const int MaxNesting = 64;

    private readonly ReadOnlyMemory<byte> _table;
    private readonly HashSet<int> _glyphs;
    private readonly int _glyphCount, _lookupList;
    private readonly Dictionary<int, (int Count, HashSet<int> Covered)> _done = [];
    private readonly Dictionary<int, bool> _nonOneToOne = [];

    private LayoutClosure(ReadOnlyMemory<byte> table, HashSet<int> glyphs, int glyphCount)
    {
        _table = table;
        _glyphs = glyphs;
        _glyphCount = glyphCount;
        _lookupList = Bytes.U16(table.Span, 8);
    }

    private ReadOnlySpan<byte> T => _table.Span;

    /// <summary>Adds to <paramref name="glyphs"/> every glyph (below <paramref name="glyphCount"/>) the GSUB table <paramref name="gsub"/> can make from them.</summary>
    public static void Close(ReadOnlyMemory<byte> gsub, HashSet<int> glyphs, int glyphCount)
    {
        if (gsub.Length < 10)
        {
            return;
        }

        var closure = new LayoutClosure(gsub, glyphs, glyphCount);
        int[] lookups = closure.FeatureLookups();
        while (true)
        {
            int before = glyphs.Count;                                          // the set only grows: its size tells whether it did
            foreach (int lookup in lookups)
            {
                closure.Lookup(lookup, null, 0);
            }

            if (glyphs.Count == before)
            {
                return;
            }
        }
    }

    /// <summary>The longest context of any lookup of <paramref name="gsub"/> and <paramref name="gpos"/> (0 when neither has one).</summary>
    public static int MaxContext(ReadOnlySpan<byte> gsub, ReadOnlySpan<byte> gpos) => Math.Max(LongestContext(gsub, positioning: false), LongestContext(gpos, positioning: true));

    // The lookups of every feature a script's language system lists (its required feature included), and of those
    // features' variations, each once, in order.
    private int[] FeatureLookups()
    {
        var t = T;
        int scriptList = Bytes.U16(t, 4), featureList = Bytes.U16(t, 6);
        var features = new SortedSet<int>();
        void LanguageSystem(int at)
        {
            int required = Bytes.U16(T, at + 2);
            if (required != 0xFFFF)
            {
                features.Add(required);
            }

            int count = Bytes.U16(T, at + 4);
            for (int i = 0; i < count; i++)
            {
                features.Add(Bytes.U16(T, at + 6 + 2 * i));
            }
        }

        if (scriptList != 0)
        {
            int scripts = Bytes.U16(t, scriptList);
            for (int s = 0; s < scripts; s++)
            {
                int script = scriptList + Bytes.U16(t, scriptList + 2 + 6 * s + 4);
                if (Bytes.U16(t, script) is int dflt and > 0)
                {
                    LanguageSystem(script + dflt);
                }

                int languages = Bytes.U16(t, script + 2);
                for (int l = 0; l < languages; l++)
                {
                    LanguageSystem(script + Bytes.U16(t, script + 4 + 6 * l + 4));
                }
            }
        }

        var lookups = new SortedSet<int>();
        void Feature(int at)
        {
            int count = Bytes.U16(T, at + 2);
            for (int i = 0; i < count; i++)
            {
                lookups.Add(Bytes.U16(T, at + 4 + 2 * i));
            }
        }

        int featureCount = featureList == 0 ? 0 : Bytes.U16(t, featureList);
        foreach (int f in features)
        {
            if (f < featureCount)
            {
                Feature(featureList + Bytes.U16(t, featureList + 2 + 6 * f + 4));
            }
        }

        // Version 1.1: the substitute features a variation's conditions bring in, for the same features.
        if (Bytes.U16(t, 2) >= 1 && Bytes.U32(t, 10) is int variations and > 0)
        {
            int records = Bytes.U32(t, variations + 4);
            for (int r = 0; r < records && r < 0x10000; r++)
            {
                int substitution = Bytes.U32(t, variations + 8 + 8 * r + 4);
                if (substitution <= 0)
                {
                    continue;
                }

                int at = variations + substitution, count = Bytes.U16(t, at + 4);
                for (int i = 0; i < count; i++)
                {
                    if (features.Contains(Bytes.U16(t, at + 6 + 6 * i)) && Bytes.U32(t, at + 6 + 6 * i + 2) is int alternate and > 0)
                    {
                        Feature(at + alternate);
                    }
                }
            }
        }

        return [.. lookups];
    }

    // The lookup's table offset and its subtables' offsets and types (an extension's unwrapped).
    private List<(int Type, int At)> Subtables(int index)
    {
        var t = T;
        var list = new List<(int, int)>();
        if (_lookupList == 0 || index >= Bytes.U16(t, _lookupList))
        {
            return list;
        }

        int lookup = _lookupList + Bytes.U16(t, _lookupList + 2 + 2 * index);
        int type = Bytes.U16(t, lookup), count = Bytes.U16(t, lookup + 4);
        for (int s = 0; s < count; s++)
        {
            int at = lookup + Bytes.U16(t, lookup + 6 + 2 * s), kind = type;
            if (type == 7 && Bytes.U16(t, at) == 1)
            {
                kind = Bytes.U16(t, at + 2);
                at += Bytes.U32(t, at + 4);
            }

            list.Add((kind, at));
        }

        return list;
    }

    // Whether the lookup can change the number of glyphs (multiple, ligature and contextual substitutions).
    private bool NonOneToOne(int index)
    {
        if (!_nonOneToOne.TryGetValue(index, out bool answer))
        {
            answer = Subtables(index).Exists(s => s.Type is 2 or 4 or 5 or 6);
            _nonOneToOne[index] = answer;
        }

        return answer;
    }

    // One lookup over `current` (null: every glyph there is), each lookup once for the glyphs it was given since the set
    // last grew.
    private void Lookup(int index, HashSet<int>? current, int depth)
    {
        if (depth > MaxNesting)
        {
            return;
        }

        current ??= [.. _glyphs];
        if (!_done.TryGetValue(index, out var done) || done.Count != _glyphs.Count)
        {
            done = (_glyphs.Count, new HashSet<int>());
            _done[index] = done;
        }

        if (current.IsSubsetOf(done.Covered))
        {
            return;
        }

        done.Covered.UnionWith(current);
        foreach (var (type, at) in Subtables(index))
        {
            Subtable(type, at, current, depth);
        }
    }

    private void Add(int glyph)
    {
        if ((uint)glyph < (uint)_glyphCount)
        {
            _glyphs.Add(glyph);
        }
    }

    private void Subtable(int type, int at, HashSet<int> current, int depth)
    {
        var t = T;
        int format = Bytes.U16(t, at);
        switch (type)
        {
            case 1:
                {
                    int delta = Bytes.I16(t, at + 4), count = Bytes.U16(t, at + 4);
                    foreach (var (index, glyph) in Coverage(at + Bytes.U16(t, at + 2)))
                    {
                        if (!current.Contains(glyph))
                        {
                            continue;
                        }

                        if (format == 1)
                        {
                            Add((glyph + delta) & 0xFFFF);
                        }
                        else if (format == 2 && index < count)
                        {
                            Add(Bytes.U16(T, at + 6 + 2 * index));
                        }
                    }

                    break;
                }

            case 2:
            case 3:
                {
                    int count = Bytes.U16(t, at + 4);
                    foreach (var (index, glyph) in Coverage(at + Bytes.U16(t, at + 2)))
                    {
                        if (index < count && current.Contains(glyph))
                        {
                            int sequence = at + Bytes.U16(T, at + 6 + 2 * index), n = Bytes.U16(T, sequence);
                            for (int k = 0; k < n; k++)
                            {
                                Add(Bytes.U16(T, sequence + 2 + 2 * k));
                            }
                        }
                    }

                    break;
                }

            case 4:
                {
                    int count = Bytes.U16(t, at + 4);
                    foreach (var (index, glyph) in Coverage(at + Bytes.U16(t, at + 2)))
                    {
                        if (index >= count || !current.Contains(glyph))
                        {
                            continue;
                        }

                        int set = at + Bytes.U16(T, at + 6 + 2 * index), ligatures = Bytes.U16(T, set);
                        for (int l = 0; l < ligatures; l++)
                        {
                            int ligature = set + Bytes.U16(T, set + 2 + 2 * l), components = Bytes.U16(T, ligature + 2);
                            bool all = true;
                            for (int c = 1; c < components && all; c++)
                            {
                                all = _glyphs.Contains(Bytes.U16(T, ligature + 4 + 2 * (c - 1)));
                            }

                            if (all)
                            {
                                Add(Bytes.U16(T, ligature));
                            }
                        }
                    }

                    break;
                }

            case 5:
            case 6:
                Context(at, chain: type == 6, current, depth);
                break;
            case 8:
                Reverse(at, current);
                break;
        }
    }

    // GSUB 8: the substitutes of the covered glyphs here, when every backtrack and lookahead coverage can match.
    private void Reverse(int at, HashSet<int> current)
    {
        var t = T;
        if (Bytes.U16(t, at) != 1)
        {
            return;
        }

        var indices = Coverage(at + Bytes.U16(t, at + 2)).Where(c => current.Contains(c.Glyph)).Select(c => c.Index).ToList();
        if (indices.Count == 0)
        {
            return;
        }

        int backtrack = Bytes.U16(t, at + 4), lookahead = Bytes.U16(t, at + 6 + 2 * backtrack);
        for (int i = 0; i < backtrack; i++)
        {
            if (!Intersects(at + Bytes.U16(t, at + 6 + 2 * i)))
            {
                return;
            }
        }

        for (int i = 0; i < lookahead; i++)
        {
            if (!Intersects(at + Bytes.U16(t, at + 8 + 2 * backtrack + 2 * i)))
            {
                return;
            }
        }

        int substitutes = at + 8 + 2 * backtrack + 2 * lookahead, count = Bytes.U16(t, substitutes);
        foreach (int index in indices)
        {
            if (index < count)
            {
                Add(Bytes.U16(T, substitutes + 2 + 2 * index));
            }
        }
    }

    // GSUB 5 and 6 (contextual and chained contextual), formats 1 to 3.
    private void Context(int at, bool chain, HashSet<int> current, int depth)
    {
        var t = T;
        int format = Bytes.U16(t, at);
        if (format == 1)
        {
            int count = Bytes.U16(t, at + 4);
            foreach (var (index, glyph) in Coverage(at + Bytes.U16(t, at + 2)))
            {
                int setOffset = index < count && current.Contains(glyph) ? Bytes.U16(T, at + 6 + 2 * index) : 0;
                if (setOffset == 0)
                {
                    continue;
                }

                int set = at + setOffset, rules = Bytes.U16(T, set);
                for (int r = 0; r < rules; r++)
                {
                    int ruleOffset = Bytes.U16(T, set + 2 + 2 * r);
                    if (ruleOffset == 0)
                    {
                        continue;
                    }

                    var rule = Rule(set + ruleOffset, chain);
                    if (!rule.Backtrack.All(_glyphs.Contains) || !rule.Input.All(_glyphs.Contains) || !rule.Lookahead.All(_glyphs.Contains))
                    {
                        continue;
                    }

                    Apply(rule.Records, rule.Input.Length + 1, seq => seq == 0 ? new HashSet<int> { glyph } : seq - 1 < rule.Input.Length ? new HashSet<int> { rule.Input[seq - 1] } : null, depth);
                }
            }

            return;
        }

        if (format == 2)
        {
            int coverage = at + Bytes.U16(t, at + 2);
            int backtrackClasses = chain ? Offset(at, at + 4) : 0, inputClasses = Offset(at, at + (chain ? 6 : 4)), lookaheadClasses = chain ? Offset(at, at + 8) : 0;
            int count = Bytes.U16(t, at + (chain ? 10 : 6)), sets = at + (chain ? 12 : 8);
            var here = Coverage(coverage).Where(c => current.Contains(c.Glyph)).Select(c => c.Glyph).ToHashSet();
            if (here.Count == 0)
            {
                return;
            }

            foreach (int cls in here.Select(g => Bytes.Class(T, inputClasses, g)).Distinct().Order())
            {
                int setOffset = cls < count ? Bytes.U16(T, sets + 2 * cls) : 0;
                if (setOffset == 0)
                {
                    continue;
                }

                int set = at + setOffset, rules = Bytes.U16(T, set);
                for (int r = 0; r < rules; r++)
                {
                    int ruleOffset = Bytes.U16(T, set + 2 + 2 * r);
                    if (ruleOffset == 0)
                    {
                        continue;
                    }

                    var rule = Rule(set + ruleOffset, chain);
                    if (!rule.Backtrack.All(k => ClassHas(backtrackClasses, k, _glyphs)) || !rule.Input.All(k => ClassHas(inputClasses, k, _glyphs))
                        || !rule.Lookahead.All(k => ClassHas(lookaheadClasses, k, _glyphs)))
                    {
                        continue;
                    }

                    int first = cls;
                    Apply(rule.Records, rule.Input.Length + 1, seq => seq == 0 ? OfClass(inputClasses, here, first)
                        : seq - 1 < rule.Input.Length ? OfClass(inputClasses, _glyphs, rule.Input[seq - 1]) : null, depth);
                }
            }

            return;
        }

        if (format == 3)
        {
            int[] backtrack, input, lookahead;
            int records;
            if (chain)
            {
                backtrack = Offsets(at, at + 2);
                int i = at + 4 + 2 * backtrack.Length;
                input = Offsets(at, i);
                i += 2 + 2 * input.Length;
                lookahead = Offsets(at, i);
                records = i + 2 + 2 * lookahead.Length;
            }
            else
            {
                int glyphCount = Bytes.U16(t, at + 2);
                backtrack = lookahead = [];
                input = new int[glyphCount];
                for (int k = 0; k < glyphCount; k++)
                {
                    input[k] = at + Bytes.U16(t, at + 6 + 2 * k);
                }

                records = at + 4;                                               // the record count before the coverages
                int lookupCount = Bytes.U16(t, records);
                var list = new (int, int)[lookupCount];
                for (int k = 0; k < lookupCount; k++)
                {
                    int r = at + 6 + 2 * glyphCount + 4 * k;
                    list[k] = (Bytes.U16(t, r), Bytes.U16(t, r + 2));
                }

                Format3(backtrack, input, lookahead, list, current, depth);
                return;
            }

            int n = Bytes.U16(t, records);
            var lookups = new (int, int)[n];
            for (int k = 0; k < n; k++)
            {
                lookups[k] = (Bytes.U16(t, records + 2 + 4 * k), Bytes.U16(t, records + 4 + 4 * k));
            }

            Format3(backtrack, input, lookahead, lookups, current, depth);
        }
    }

    private void Format3(int[] backtrack, int[] input, int[] lookahead, (int Sequence, int Lookup)[] records, HashSet<int> current, int depth)
    {
        if (input.Length == 0)
        {
            return;
        }

        var here = Coverage(input[0]).Where(c => current.Contains(c.Glyph)).Select(c => c.Glyph).ToHashSet();
        if (here.Count == 0 || !backtrack.All(Intersects) || !input.All(Intersects) || !lookahead.All(Intersects))
        {
            return;
        }

        Apply(records, input.Length, seq => seq == 0 ? here
            : seq < input.Length ? Coverage(input[seq]).Where(c => _glyphs.Contains(c.Glyph)).Select(c => c.Glyph).ToHashSet() : null, depth);
    }

    // A rule's nested lookups, in order, each on the glyphs its place can hold; a place an earlier lookup there may have
    // changed the length before (multiple, ligature, contextual) takes every glyph.
    private void Apply((int Sequence, int Lookup)[] records, int inputLength, Func<int, HashSet<int>?> place, int depth)
    {
        var chaos = new HashSet<int>();
        foreach (var (sequence, lookup) in records)
        {
            HashSet<int>? glyphs = chaos.Contains(sequence) ? null : place(sequence);
            if (!chaos.Contains(sequence) && glyphs is null)
            {
                continue;                                                       // a place past the rule's input: a damaged rule
            }

            chaos.Add(sequence);
            if (NonOneToOne(lookup))
            {
                for (int s = sequence; s <= inputLength; s++)
                {
                    chaos.Add(s);
                }
            }

            Lookup(lookup, glyphs, depth + 1);
        }
    }

    // A format 1 or 2 rule: its backtrack, input (after the first) and lookahead values (glyphs or classes) and its records.
    private (int[] Backtrack, int[] Input, int[] Lookahead, (int Sequence, int Lookup)[] Records) Rule(int at, bool chain)
    {
        var t = T;
        int[] backtrack = [], lookahead = [];
        int i = at;
        if (chain)
        {
            backtrack = Values(i);
            i += 2 + 2 * backtrack.Length;
            int inputCount = Bytes.U16(t, i);
            var input = new int[Math.Max(0, inputCount - 1)];
            for (int k = 0; k < input.Length; k++)
            {
                input[k] = Bytes.U16(t, i + 2 + 2 * k);
            }

            i += 2 + 2 * input.Length;
            lookahead = Values(i);
            i += 2 + 2 * lookahead.Length;
            return (backtrack, input, lookahead, Records(i));
        }

        int glyphCount = Bytes.U16(t, i), recordCount = Bytes.U16(t, i + 2);
        var rest = new int[Math.Max(0, glyphCount - 1)];
        for (int k = 0; k < rest.Length; k++)
        {
            rest[k] = Bytes.U16(t, i + 4 + 2 * k);
        }

        var records = new (int, int)[recordCount];
        int r = i + 4 + 2 * rest.Length;
        for (int k = 0; k < recordCount; k++)
        {
            records[k] = (Bytes.U16(t, r + 4 * k), Bytes.U16(t, r + 4 * k + 2));
        }

        return (backtrack, rest, lookahead, records);
    }

    private int[] Values(int at)
    {
        int count = Bytes.U16(T, at);
        var values = new int[count];
        for (int k = 0; k < count; k++)
        {
            values[k] = Bytes.U16(T, at + 2 + 2 * k);
        }

        return values;
    }

    private (int, int)[] Records(int at)
    {
        int count = Bytes.U16(T, at);
        var records = new (int, int)[count];
        for (int k = 0; k < count; k++)
        {
            records[k] = (Bytes.U16(T, at + 2 + 4 * k), Bytes.U16(T, at + 4 + 4 * k));
        }

        return records;
    }

    // A count followed by that many offsets (from `table`), as absolute positions.
    private int[] Offsets(int table, int at)
    {
        int count = Bytes.U16(T, at);
        var offsets = new int[count];
        for (int k = 0; k < count; k++)
        {
            offsets[k] = table + Bytes.U16(T, at + 2 + 2 * k);
        }

        return offsets;
    }

    // An offset field (from `table`) as an absolute position, 0 for a null offset.
    private int Offset(int table, int at) => Bytes.U16(T, at) is int offset and > 0 ? table + offset : 0;

    // The glyphs of a coverage table with their coverage indices.
    private List<(int Index, int Glyph)> Coverage(int at)
    {
        var t = T;
        var list = new List<(int, int)>();
        int format = Bytes.U16(t, at), count = Bytes.U16(t, at + 2);
        if (format == 1)
        {
            for (int i = 0; i < count; i++)
            {
                list.Add((i, Bytes.U16(t, at + 4 + 2 * i)));
            }
        }
        else if (format == 2)
        {
            for (int r = 0; r < count; r++)
            {
                int start = Bytes.U16(t, at + 4 + 6 * r), end = Bytes.U16(t, at + 6 + 6 * r), first = Bytes.U16(t, at + 8 + 6 * r);
                for (int g = start; g <= end; g++)
                {
                    list.Add((first + g - start, g));
                }
            }
        }

        return list;
    }

    // Whether a coverage table lists a glyph that is there.
    private bool Intersects(int coverage) => Coverage(coverage).Exists(c => _glyphs.Contains(c.Glyph));

    // Whether some glyph of `glyphs` has class `cls` (class 0: no class) in the class definition at `at` (0: none, every glyph class 0).
    private bool ClassHas(int at, int cls, HashSet<int> glyphs)
    {
        foreach (int g in glyphs)
        {
            if (Bytes.Class(T, at, g) == cls)
            {
                return true;
            }
        }

        return false;
    }

    private HashSet<int> OfClass(int at, HashSet<int> glyphs, int cls)
    {
        var set = new HashSet<int>();
        foreach (int g in glyphs)
        {
            if (Bytes.Class(T, at, g) == cls)
            {
                set.Add(g);
            }
        }

        return set;
    }

    // The longest context of one table's lookups.
    private static int LongestContext(ReadOnlySpan<byte> t, bool positioning)
    {
        if (t.Length < 10)
        {
            return 0;
        }

        int max = 0, lookupList = Bytes.U16(t, 8), lookups = lookupList == 0 ? 0 : Bytes.U16(t, lookupList);
        for (int l = 0; l < lookups; l++)
        {
            int lookup = lookupList + Bytes.U16(t, lookupList + 2 + 2 * l);
            int type = Bytes.U16(t, lookup), count = Bytes.U16(t, lookup + 4);
            for (int s = 0; s < count; s++)
            {
                int at = lookup + Bytes.U16(t, lookup + 6 + 2 * s), kind = type;
                if (type == (positioning ? 9 : 7) && Bytes.U16(t, at) == 1)
                {
                    kind = Bytes.U16(t, at + 2);
                    at += Bytes.U32(t, at + 4);
                }

                max = Math.Max(max, SubtableContext(t, kind, at, positioning));
            }
        }

        return max;
    }

    private static int SubtableContext(ReadOnlySpan<byte> t, int kind, int at, bool positioning)
    {
        if (positioning ? kind == 1 : kind is 1 or 2 or 3)
        {
            return 1;
        }

        if (positioning && kind == 2)
        {
            return 2;
        }

        if (!positioning && kind == 4)
        {
            int max = 0, sets = Bytes.U16(t, at + 4);
            for (int i = 0; i < sets; i++)
            {
                int set = at + Bytes.U16(t, at + 6 + 2 * i), ligatures = Bytes.U16(t, set);
                for (int l = 0; l < ligatures; l++)
                {
                    max = Math.Max(max, Bytes.U16(t, set + Bytes.U16(t, set + 2 + 2 * l) + 2));
                }
            }

            return max;
        }

        if (kind == (positioning ? 7 : 5) || kind == (positioning ? 8 : 6))
        {
            bool chain = kind == (positioning ? 8 : 6);
            int format = Bytes.U16(t, at);
            if (format == 3)
            {
                return chain ? ChainCount(t, at + 2, format3: true) : Bytes.U16(t, at + 2);
            }

            if (format is not (1 or 2))
            {
                return 0;
            }

            int sets = Bytes.U16(t, at + (format == 2 ? (chain ? 10 : 6) : 4)), first = at + (format == 2 ? (chain ? 12 : 8) : 6), max = 0;

            for (int i = 0; i < sets; i++)
            {
                int setOffset = Bytes.U16(t, first + 2 * i);
                if (setOffset == 0)
                {
                    continue;
                }

                int set = at + setOffset, rules = Bytes.U16(t, set);
                for (int r = 0; r < rules; r++)
                {
                    int ruleOffset = Bytes.U16(t, set + 2 + 2 * r);
                    if (ruleOffset != 0)
                    {
                        int rule = set + ruleOffset;
                        max = Math.Max(max, chain ? ChainCount(t, rule, format3: false) : Bytes.U16(t, rule));
                    }
                }
            }

            return max;
        }

        if (!positioning && kind == 8)
        {
            int backtrack = Bytes.U16(t, at + 4);
            return 1 + Bytes.U16(t, at + 6 + 2 * backtrack);
        }

        return 0;
    }

    // A chained rule's (or a format 3 table's, from its backtrack count) input and lookahead counts added: a rule lists its
    // input after the first glyph (count - 1 values), a format 3 table a coverage for each.
    private static int ChainCount(ReadOnlySpan<byte> t, int at, bool format3)
    {
        int backtrack = Bytes.U16(t, at);
        int input = at + 2 + 2 * backtrack, inputCount = Bytes.U16(t, input);
        int listed = format3 ? inputCount : Math.Max(0, inputCount - 1);
        return inputCount + Bytes.U16(t, input + 2 + 2 * listed);
    }
}
