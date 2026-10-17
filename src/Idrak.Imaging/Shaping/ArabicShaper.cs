// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;

namespace Idrak.Imaging.Shaping;

/// <summary>
/// "arabic": cursive joining scripts (Arabic, Syriac, N'Ko, Mongolian, Mandaic, Manichaean, Psalter Pahlavi, Adlam,
/// Hanifi Rohingya, Sogdian, Old Uyghur). Each letter's form (isolated, final, medial, initial) follows from the joining
/// types of its neighbours, marks passed over; the font's isol, fina, medi and init features each run in their own stage,
/// then rlig, calt and the rest, as HarfBuzz orders them. Modifier marks (hamza above and below, ...) are moved first
/// among their class.
/// </summary>
internal sealed class ArabicShaper : OpenTypeShaper
{
    /// <summary>The scripts this shaper is registered for.</summary>
    public static readonly string[] Scripts = ["Arab", "Syrc", "Nkoo", "Mong", "Mand", "Mani", "Phlp", "Adlm", "Rohg", "Sogd", "Ougr"];

    private static readonly string[] Forms = ["isol", "fina", "fin2", "fin3", "medi", "med2", "init"];

    public override string Name => "arabic";

    protected override IReadOnlyList<FeatureRequest> SubstitutionFeatures(bool rightToLeft)
    {
        List<FeatureRequest> list =
        [
            new("rvrn", 0),
            .. rightToLeft ? (FeatureRequest[])[new("rtla", 1), new("rtlm", 1)] : [new("ltra", 1), new("ltrm", 1)],
            new("stch", 1),
            new("ccmp", 2, ManualZwj: true), new("locl", 2, ManualZwj: true),
        ];
        int stage = 3;
        foreach (string form in Forms)
        {
            list.Add(new(form, stage++, Global: false));
        }

        list.Add(new("rlig", stage++, ManualZwj: true));
        list.Add(new("rclt", stage, ManualZwj: true));
        list.Add(new("calt", stage++, ManualZwj: true));
        list.Add(new("mset", stage));
        list.Add(new("clig", stage));
        list.Add(new("liga", stage));
        return list;
    }

    // Each letter's positional form from its joining type and the nearest non-transparent letters around it (across the
    // run's edges: the text around a run is context).
    protected override void SetupMasks(ShapingPlan plan, List<ShapingGlyph> glyphs, string text, TextRun run)
    {
        var span = CollectionsMarshal.AsSpan(glyphs);
        var types = new JoiningType[span.Length];
        for (int i = 0; i < span.Length; i++)
        {
            types[i] = Joining(span[i].CodePoint);
        }

        var before = ContextType(text, run.Start - 1, -1);
        var after = ContextType(text, run.Start + run.Length, 1);
        for (int i = 0; i < span.Length; i++)
        {
            var type = types[i];
            if (type is JoiningType.T or JoiningType.U)
            {
                continue;
            }

            var previous = before;
            for (int k = i - 1; k >= 0; k--)
            {
                if (types[k] != JoiningType.T)
                {
                    previous = types[k];
                    break;
                }

                if (k == 0)
                {
                    previous = before;
                }
            }

            var following = after;
            for (int k = i + 1; k < span.Length; k++)
            {
                if (types[k] != JoiningType.T)
                {
                    following = types[k];
                    break;
                }

                if (k == span.Length - 1)
                {
                    following = after;
                }
            }

            bool joinsPrevious = type is JoiningType.R or JoiningType.D or JoiningType.C && previous is JoiningType.L or JoiningType.D or JoiningType.C;
            bool joinsNext = type is JoiningType.L or JoiningType.D or JoiningType.C && following is JoiningType.R or JoiningType.D or JoiningType.C;
            string form = joinsPrevious && joinsNext ? "medi" : joinsPrevious ? "fina" : joinsNext ? "init" : "isol";
            if (plan.Masks.TryGetValue(form, out uint mask))
            {
                span[i].Mask |= mask;
            }
        }
    }

    private static JoiningType Joining(int codePoint) => ArabicPresentationForms.JoiningTypeOf(codePoint);

    // The joining type of the first non-transparent character from `at` in `step` direction, U at the text's edge.
    private static JoiningType ContextType(string text, int at, int step)
    {
        for (int i = at; i >= 0 && i < text.Length; i += step)
        {
            int cp = text[i];
            if (char.IsSurrogate(text[i]))
            {
                if (step < 0 && char.IsLowSurrogate(text[i]) && i > 0)
                {
                    cp = char.ConvertToUtf32(text[i - 1], text[i]);
                    i--;
                }
                else if (step > 0 && char.IsHighSurrogate(text[i]) && i + 1 < text.Length)
                {
                    cp = char.ConvertToUtf32(text[i], text[i + 1]);
                    i++;
                }
            }

            var type = Joining(cp);
            if (type != JoiningType.T)
            {
                return type;
            }
        }

        return JoiningType.U;
    }

    // HarfBuzz's reorder_marks_arabic: modifier combining marks (hamza above or below, ...) of class 220 or 230 move to
    // the start of the marks of their class.
    protected override void ReorderMarks(List<ShapingGlyph> glyphs, int start, int end)
    {
        var span = CollectionsMarshal.AsSpan(glyphs);
        int i = start;
        for (int cc = 220; cc <= 230; cc += 10)
        {
            while (i < end && span[i].Combining < cc)
            {
                i++;
            }

            if (i == end)
            {
                break;
            }

            if (span[i].Combining > cc)
            {
                continue;
            }

            int j = i;
            while (j < end && span[j].Combining == cc && IsModifierMark(span[j].CodePoint))
            {
                j++;
            }

            if (i == j)
            {
                continue;
            }

            // The modifier marks [i, j) go to `start`, the marks [start, i) after them.
            var moved = span[i..j].ToArray();
            var rest = span[start..i].ToArray();
            moved.CopyTo(span[start..]);
            rest.CopyTo(span[(start + moved.Length)..]);
            MergeClusters(span, start, j);
            start += moved.Length;
            i = j;
        }
    }

    private static bool IsModifierMark(int cp) => cp is 0x0654 or 0x0655 or 0x0658 or 0x06DC or 0x06E3 or 0x06E7 or 0x06E8 or 0x08CA or 0x08CB
        or 0x08CD or 0x08CE or 0x08CF or 0x08D3 or 0x08F3;

    // A font without Arabic GSUB: the Unicode presentation forms of each letter's form, and lam-alef ligatures.
    protected override void FallbackSubstitution(IFontFace font, ShapingPlan plan, List<ShapingGlyph> glyphs, string text, TextRun run)
    {
        var span = CollectionsMarshal.AsSpan(glyphs);
        for (int i = 0; i < span.Length; i++)
        {
            ref var g = ref span[i];
            int form = FormOf(plan, g.Mask);
            if (form < 0)
            {
                continue;
            }

            // Lam followed by an alef: one ligature glyph (isolated or final), the alef removed.
            if (g.CodePoint == 0x0644 && form is 2 or 3 && i + 1 < span.Length && UnicodeText.PresentationLamAlef(span[i + 1].CodePoint, final: form == 3) is int ligature
                && font.Glyph(ligature) is int lg and > 0)
            {
                g.Glyph = lg;
                g.Kind = GlyphKind.Ligature;
                glyphs.RemoveAt(i + 1);
                span = CollectionsMarshal.AsSpan(glyphs);
                continue;
            }

            if (UnicodeText.PresentationForm(g.CodePoint, form) is int presentation && font.Glyph(presentation) is int pg and > 0)
            {
                g.Glyph = pg;
            }
        }
    }

    // 0 isolated, 1 final, 2 initial, 3 medial, or -1 when the glyph has no positional form.
    private static int FormOf(ShapingPlan plan, uint mask)
    {
        (string Tag, int Form)[] forms = [("isol", 0), ("fina", 1), ("init", 2), ("medi", 3)];
        foreach (var (tag, form) in forms)
        {
            if (plan.Masks.TryGetValue(tag, out uint m) && (mask & m) != 0 && m != plan.GlobalMask)
            {
                return form;
            }
        }

        return -1;
    }
}
