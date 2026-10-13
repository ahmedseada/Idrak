// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Imaging;

/// <summary>
/// The outlines of a CFF table (PostScript outlines in OpenType: most .otf fonts and the CJK fonts): its INDEXes and
/// DICTs, the Private DICT's local subroutines, CID-keyed fonts' FDArray and FDSelect, and a Type 2 charstring interpreter
/// (every path operator, hints counted for hint masks, flex, subroutine calls). Outlines come out in font units (the
/// FontMatrix applied), as cubic curves.
/// </summary>
internal sealed class CffOutlines
{
    private const int MaxStack = 513, MaxCallDepth = 10;

    private readonly ReadOnlyMemory<byte> _data;
    private readonly (int Offset, int Count, int OffSize, int DataStart) _charStrings, _globalSubrs;
    private readonly (int Offset, int Count, int OffSize, int DataStart)[] _localSubrs;   // one per font DICT (one for a plain font)
    private readonly byte[]? _fdSelect;                                                   // the font DICT of each glyph (CID-keyed)
    private readonly float _scaleX, _scaleY;
    private readonly string _name;

    public CffOutlines(ReadOnlyMemory<byte> table, int unitsPerEm, int glyphCount, string name)
    {
        _data = table;
        _name = name;
        var t = table.Span;
        if (t.Length < 4 || t[0] != 1)
        {
            throw new NotSupportedException($"{name}: CFF version {(t.Length > 0 ? t[0] : 0)} is not read (CFF2, the variable fonts' outlines, is not supported yet).");
        }

        int at = t[2];
        var names = Index(at);
        var topDicts = Index(names.End);
        var strings = Index(topDicts.End);
        var global = Index(strings.End);
        _globalSubrs = (global.Offset, global.Count, global.OffSize, global.DataStart);
        if (topDicts.Count < 1)
        {
            throw new InvalidDataException($"{name}: the CFF table has no font.");
        }

        var (topStart, topEnd) = Item(topDicts, 0);
        var top = Dict(topStart, topEnd);
        if (!top.TryGetValue(17, out var charStrings))
        {
            throw new InvalidDataException($"{name}: the CFF font has no CharStrings.");
        }

        if (top.TryGetValue(1206, out var type) && type[0] != 2)
        {
            throw new NotSupportedException($"{name}: Type {type[0]} charstrings are not read (Type 2 only).");
        }

        var cs = Index((int)charStrings[0]);
        _charStrings = (cs.Offset, cs.Count, cs.OffSize, cs.DataStart);

        // FontMatrix (default 0.001 0 0 0.001): charstring units to ems, times units per em to font units.
        double[] matrix = top.TryGetValue(1207, out var m) && m.Length >= 4 ? m : [0.001, 0, 0, 0.001];
        _scaleX = (float)(matrix[0] * unitsPerEm);
        _scaleY = (float)(matrix[3] * unitsPerEm);

        if (top.ContainsKey(1230))                                                  // ROS: CID-keyed
        {
            var fdArray = Index((int)top[1236][0]);
            _localSubrs = new (int, int, int, int)[fdArray.Count];
            for (int i = 0; i < fdArray.Count; i++)
            {
                var (s, e) = Item(fdArray, i);
                _localSubrs[i] = LocalSubrs(Dict(s, e));
            }

            _fdSelect = FdSelect((int)top[1237][0], Math.Max(glyphCount, _charStrings.Count));
        }
        else
        {
            _localSubrs = [LocalSubrs(top)];
        }
    }

    /// <summary>The number of charstrings.</summary>
    public int Count => _charStrings.Count;

    /// <summary>A glyph's outline (empty for a glyph out of range or with no path).</summary>
    public GlyphOutline Outline(int glyph)
    {
        if ((uint)glyph >= (uint)_charStrings.Count)
        {
            return GlyphOutline.Empty;
        }

        var (start, end) = Item((_charStrings.Offset, _charStrings.Count, _charStrings.OffSize, _charStrings.DataStart, 0), glyph);
        int fd = _fdSelect is { } select && glyph < select.Length ? select[glyph] : 0;
        var local = _localSubrs[Math.Min(fd, _localSubrs.Length - 1)];
        var path = new PathBuilder(_scaleX, _scaleY);
        var state = new State();
        Run(start, end, local, path, ref state, 0);
        path.Close();
        return path.Build();
    }

    // ---------------------------------------------------------------- the table's structures

    private (int Offset, int Count, int OffSize, int DataStart, int End) Index(int at)
    {
        var t = _data.Span;
        int count = Bytes(t, at, 2);
        if (count == 0)
        {
            return (at, 0, 1, at + 2, at + 2);
        }

        int offSize = t[at + 2];
        int dataStart = at + 3 + (count + 1) * offSize - 1;
        int last = Bytes(t, at + 3 + count * offSize, offSize);
        return (at, count, offSize, dataStart, dataStart + last);
    }

    private (int Start, int End) Item((int Offset, int Count, int OffSize, int DataStart, int End) index, int i)
    {
        var t = _data.Span;
        int a = Bytes(t, index.Offset + 3 + i * index.OffSize, index.OffSize), b = Bytes(t, index.Offset + 3 + (i + 1) * index.OffSize, index.OffSize);
        return (index.DataStart + a, index.DataStart + b);
    }

    private static int Bytes(ReadOnlySpan<byte> t, int at, int size)
    {
        int v = 0;
        for (int k = 0; k < size; k++)
        {
            v = v << 8 | ((uint)(at + k) < (uint)t.Length ? t[at + k] : 0);
        }

        return v;
    }

    // A DICT's operators (12 x as 1200 + x) and their operands.
    private Dictionary<int, double[]> Dict(int start, int end)
    {
        var t = _data.Span;
        var dict = new Dictionary<int, double[]>();
        var operands = new List<double>();
        for (int i = start; i < end && i < t.Length;)
        {
            int b0 = t[i];
            if (b0 <= 21)
            {
                int op = b0 == 12 ? 1200 + t[i + 1] : b0;
                i += b0 == 12 ? 2 : 1;
                dict[op] = [.. operands];
                operands.Clear();
            }
            else if (b0 == 28)
            {
                operands.Add((short)(t[i + 1] << 8 | t[i + 2]));
                i += 3;
            }
            else if (b0 == 29)
            {
                operands.Add(t[i + 1] << 24 | t[i + 2] << 16 | t[i + 3] << 8 | t[i + 4]);
                i += 5;
            }
            else if (b0 == 30)
            {
                operands.Add(Real(t, ref i));
            }
            else if (b0 >= 32 && b0 <= 246)
            {
                operands.Add(b0 - 139);
                i++;
            }
            else if (b0 >= 247 && b0 <= 250)
            {
                operands.Add((b0 - 247) * 256 + t[i + 1] + 108);
                i += 2;
            }
            else if (b0 >= 251 && b0 <= 254)
            {
                operands.Add(-(b0 - 251) * 256 - t[i + 1] - 108);
                i += 2;
            }
            else
            {
                i++;
            }
        }

        return dict;
    }

    // A DICT real number: nibbles of digits, point, exponent and sign, ending at nibble 0xF.
    private static double Real(ReadOnlySpan<byte> t, ref int i)
    {
        i++;
        var text = new System.Text.StringBuilder();
        while (i < t.Length)
        {
            int b = t[i++];
            foreach (int nibble in (int[])[b >> 4, b & 0xF])
            {
                switch (nibble)
                {
                    case <= 9:
                        text.Append((char)('0' + nibble));
                        break;
                    case 0xA:
                        text.Append('.');
                        break;
                    case 0xB:
                        text.Append('E');
                        break;
                    case 0xC:
                        text.Append("E-");
                        break;
                    case 0xE:
                        text.Append('-');
                        break;
                    case 0xF:
                        return double.TryParse(text.ToString(), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out double v) ? v : 0;
                }
            }
        }

        return 0;
    }

    private (int, int, int, int) LocalSubrs(Dictionary<int, double[]> fontDict)
    {
        if (!fontDict.TryGetValue(18, out var priv) || priv.Length < 2)
        {
            return (0, 0, 1, 0);
        }

        int size = (int)priv[0], offset = (int)priv[1];
        var p = Dict(offset, offset + size);
        if (!p.TryGetValue(19, out var subrs))
        {
            return (0, 0, 1, 0);
        }

        var index = Index(offset + (int)subrs[0]);
        return (index.Offset, index.Count, index.OffSize, index.DataStart);
    }

    private byte[] FdSelect(int at, int glyphs)
    {
        var t = _data.Span;
        var select = new byte[glyphs];
        int format = t[at];
        if (format == 0)
        {
            for (int g = 0; g < glyphs; g++)
            {
                select[g] = t[at + 1 + g];
            }
        }
        else if (format == 3)
        {
            int ranges = Bytes(t, at + 1, 2);
            for (int r = 0; r < ranges; r++)
            {
                int first = Bytes(t, at + 3 + 3 * r, 2), fd = t[at + 3 + 3 * r + 2], next = Bytes(t, at + 3 + 3 * (r + 1), 2);
                for (int g = first; g < next && g < glyphs; g++)
                {
                    select[g] = (byte)fd;
                }
            }
        }

        return select;
    }

    private static int Bias(int count) => count < 1240 ? 107 : count < 33900 ? 1131 : 32768;

    // ---------------------------------------------------------------- Type 2 charstrings

    private struct State
    {
        public float[] Stack;
        public int Top;
        public int Stems;
        public bool WidthTaken;
        public bool Ended;
    }

    // Runs the charstring [start, end); returns false when it ended (endchar).
    private void Run(int start, int end, (int Offset, int Count, int OffSize, int DataStart) local, PathBuilder path, ref State s, int depth)
    {
        s.Stack ??= new float[MaxStack];
        var t = _data.Span;
        int i = start;
        while (i < end && !s.Ended)
        {
            int b0 = t[i];
            if (b0 >= 32 || b0 == 28)
            {
                float v;
                if (b0 == 28)
                {
                    v = (short)(t[i + 1] << 8 | t[i + 2]);
                    i += 3;
                }
                else if (b0 <= 246)
                {
                    v = b0 - 139;
                    i++;
                }
                else if (b0 <= 250)
                {
                    v = (b0 - 247) * 256 + t[i + 1] + 108;
                    i += 2;
                }
                else if (b0 <= 254)
                {
                    v = -(b0 - 251) * 256 - t[i + 1] - 108;
                    i += 2;
                }
                else
                {
                    v = (t[i + 1] << 24 | t[i + 2] << 16 | t[i + 3] << 8 | t[i + 4]) / 65536f;
                    i += 5;
                }

                if (s.Top < MaxStack)
                {
                    s.Stack[s.Top++] = v;
                }

                continue;
            }

            i++;
            var a = s.Stack;
            int n = s.Top;
            switch (b0)
            {
                case 1 or 3 or 18 or 23:                                         // hstem, vstem, hstemhm, vstemhm
                    TakeWidth(ref s, n % 2 == 1);
                    s.Stems += s.Top / 2;
                    s.Top = 0;
                    break;
                case 19 or 20:                                                   // hintmask, cntrmask (stems before it are vstems)
                    TakeWidth(ref s, n % 2 == 1);
                    s.Stems += s.Top / 2;
                    s.Top = 0;
                    i += (s.Stems + 7) / 8;
                    break;
                case 21:                                                         // rmoveto
                {
                    int o = TakeWidth(ref s, n > 2);
                    path.MoveBy(a[o], a[o + 1]);
                    s.Top = 0;
                    break;
                }

                case 22:                                                         // hmoveto
                {
                    int o = TakeWidth(ref s, n > 1);
                    path.MoveBy(a[o], 0);
                    s.Top = 0;
                    break;
                }

                case 4:                                                          // vmoveto
                {
                    int o = TakeWidth(ref s, n > 1);
                    path.MoveBy(0, a[o]);
                    s.Top = 0;
                    break;
                }

                case 5:                                                          // rlineto
                    for (int k = 0; k + 1 < n; k += 2)
                    {
                        path.LineBy(a[k], a[k + 1]);
                    }

                    s.Top = 0;
                    break;
                case 6 or 7:                                                     // hlineto, vlineto: alternating
                {
                    bool horizontal = b0 == 6;
                    for (int k = 0; k < n; k++, horizontal = !horizontal)
                    {
                        if (horizontal)
                        {
                            path.LineBy(a[k], 0);
                        }
                        else
                        {
                            path.LineBy(0, a[k]);
                        }
                    }

                    s.Top = 0;
                    break;
                }

                case 8:                                                          // rrcurveto
                    for (int k = 0; k + 5 < n; k += 6)
                    {
                        path.CurveBy(a[k], a[k + 1], a[k + 2], a[k + 3], a[k + 4], a[k + 5]);
                    }

                    s.Top = 0;
                    break;
                case 24:                                                         // rcurveline
                {
                    int k = 0;
                    for (; k + 5 < n - 2; k += 6)
                    {
                        path.CurveBy(a[k], a[k + 1], a[k + 2], a[k + 3], a[k + 4], a[k + 5]);
                    }

                    path.LineBy(a[k], a[k + 1]);
                    s.Top = 0;
                    break;
                }

                case 25:                                                         // rlinecurve
                {
                    int k = 0;
                    for (; k + 1 < n - 6; k += 2)
                    {
                        path.LineBy(a[k], a[k + 1]);
                    }

                    path.CurveBy(a[k], a[k + 1], a[k + 2], a[k + 3], a[k + 4], a[k + 5]);
                    s.Top = 0;
                    break;
                }

                case 26:                                                         // vvcurveto
                {
                    int k = 0;
                    float dx1 = 0;
                    if (n % 4 == 1)
                    {
                        dx1 = a[0];
                        k = 1;
                    }

                    for (; k + 3 < n; k += 4)
                    {
                        path.CurveBy(dx1, a[k], a[k + 1], a[k + 2], 0, a[k + 3]);
                        dx1 = 0;
                    }

                    s.Top = 0;
                    break;
                }

                case 27:                                                         // hhcurveto
                {
                    int k = 0;
                    float dy1 = 0;
                    if (n % 4 == 1)
                    {
                        dy1 = a[0];
                        k = 1;
                    }

                    for (; k + 3 < n; k += 4)
                    {
                        path.CurveBy(a[k], dy1, a[k + 1], a[k + 2], a[k + 3], 0);
                        dy1 = 0;
                    }

                    s.Top = 0;
                    break;
                }

                case 30 or 31:                                                   // vhcurveto, hvcurveto: alternating, a last delta at the end
                {
                    bool horizontal = b0 == 31;
                    for (int k = 0; k + 3 < n; k += 4, horizontal = !horizontal)
                    {
                        bool last = k + 4 == n - 1;
                        float extra = last ? a[k + 4] : 0;
                        if (horizontal)
                        {
                            path.CurveBy(a[k], 0, a[k + 1], a[k + 2], extra, a[k + 3]);
                        }
                        else
                        {
                            path.CurveBy(0, a[k], a[k + 1], a[k + 2], a[k + 3], extra);
                        }
                    }

                    s.Top = 0;
                    break;
                }

                case 10 or 29:                                                   // callsubr, callgsubr
                {
                    if (n == 0 || depth >= MaxCallDepth)
                    {
                        s.Top = 0;
                        break;
                    }

                    var subrs = b0 == 10 ? local : _globalSubrs;
                    int index = (int)a[--s.Top] + Bias(subrs.Count);
                    if ((uint)index < (uint)subrs.Count)
                    {
                        var (ss, se) = Item((subrs.Offset, subrs.Count, subrs.OffSize, subrs.DataStart, 0), index);
                        Run(ss, se, local, path, ref s, depth + 1);
                    }

                    break;
                }

                case 11:                                                         // return
                    return;
                case 14:                                                         // endchar (an accented seac is not composed)
                    TakeWidth(ref s, n is 1 or 5);
                    s.Top = 0;
                    s.Ended = true;
                    return;
                case 12:
                {
                    int op = t[i++];
                    switch (op)
                    {
                        case 35:                                                 // flex
                            path.CurveBy(a[0], a[1], a[2], a[3], a[4], a[5]);
                            path.CurveBy(a[6], a[7], a[8], a[9], a[10], a[11]);
                            break;
                        case 34:                                                 // hflex
                            path.CurveBy(a[0], 0, a[1], a[2], a[3], 0);
                            path.CurveBy(a[4], 0, a[5], -a[2], a[6], 0);
                            break;
                        case 36:                                                 // hflex1
                            path.CurveBy(a[0], a[1], a[2], a[3], a[4], 0);
                            path.CurveBy(a[5], 0, a[6], a[7], a[8], -(a[1] + a[3] + a[7]));
                            break;
                        case 37:                                                 // flex1: the last point by the larger of the two sums
                        {
                            float dx = a[0] + a[2] + a[4] + a[6] + a[8], dy = a[1] + a[3] + a[5] + a[7] + a[9];
                            path.CurveBy(a[0], a[1], a[2], a[3], a[4], a[5]);
                            if (Math.Abs(dx) > Math.Abs(dy))
                            {
                                path.CurveBy(a[6], a[7], a[8], a[9], a[10], -dy);
                            }
                            else
                            {
                                path.CurveBy(a[6], a[7], a[8], a[9], -dx, a[10]);
                            }

                            break;
                        }
                    }

                    s.Top = 0;
                    break;
                }

                default:
                    s.Top = 0;                                                    // a reserved or arithmetic operator: its operands go
                    break;
            }
        }
    }

    // The first stack-clearing operator may carry the advance width first: skip it.
    private static int TakeWidth(ref State s, bool hasWidth)
    {
        if (s.WidthTaken)
        {
            return 0;
        }

        s.WidthTaken = true;
        if (!hasWidth)
        {
            return 0;
        }

        Array.Copy(s.Stack, 1, s.Stack, 0, s.Top - 1);
        s.Top--;
        return 0;
    }

    // Relative path commands to an outline in font units.
    private sealed class PathBuilder(float scaleX, float scaleY)
    {
        private readonly List<OutlineVerb> _verbs = [];
        private readonly List<float> _points = [];
        private float _x, _y;
        private bool _open;

        public void MoveBy(float dx, float dy)
        {
            Close();
            _x += dx;
            _y += dy;
            _verbs.Add(OutlineVerb.Move);
            _points.Add(_x * scaleX);
            _points.Add(_y * scaleY);
            _open = true;
        }

        public void LineBy(float dx, float dy)
        {
            _x += dx;
            _y += dy;
            _verbs.Add(OutlineVerb.Line);
            _points.Add(_x * scaleX);
            _points.Add(_y * scaleY);
        }

        public void CurveBy(float dx1, float dy1, float dx2, float dy2, float dx3, float dy3)
        {
            float x1 = _x + dx1, y1 = _y + dy1, x2 = x1 + dx2, y2 = y1 + dy2;
            _x = x2 + dx3;
            _y = y2 + dy3;
            _verbs.Add(OutlineVerb.Cubic);
            _points.AddRange([x1 * scaleX, y1 * scaleY, x2 * scaleX, y2 * scaleY, _x * scaleX, _y * scaleY]);
        }

        public void Close()
        {
            if (_open)
            {
                _verbs.Add(OutlineVerb.Close);
                _open = false;
            }
        }

        public GlyphOutline Build() => _verbs.Count == 0 ? GlyphOutline.Empty : new GlyphOutline([.. _verbs], [.. _points]);
    }
}
