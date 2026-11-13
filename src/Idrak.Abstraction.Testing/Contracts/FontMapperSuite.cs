// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of a font mapper (<see cref="IFontMapper"/>, registered in <see cref="FontMappers"/>), on a legacy font and
/// its map and on a font that answers its letters itself:
/// <list type="bullet">
/// <item>the legacy font fits its map, and every letter given by the font's own code points answers that glyph (through a
/// fallback when the map gives one), every letter found answers a glyph;</item>
/// <item>each letter form given by code point is reachable from the letter's glyph through the made GSUB's single
/// substitution of the form's feature;</item>
/// <item>a map of glyph ids on the other font answers each mapped character with its glyph and reaches each form;</item>
/// <item>the other font does not fit the map when matched (it answers the letters itself), and a font unchanged by
/// <see cref="FontMaps.Apply(IFontFace)"/> is the same face;</item>
/// <item>the same answers from several threads;</item>
/// <item>a map with an unknown key or a bad value is refused, the message naming the source, the key and what is valid.</item>
/// </list>
/// </summary>
/// <param name="legacy">A legacy font <paramref name="map"/> fits.</param>
/// <param name="map">The map.</param>
/// <param name="unicode">A font that answers the map's letters itself (it must not be mapped when matched).</param>
public sealed class FontMapperSuite(IFontFace legacy, FontMap map, IFontFace unicode) : ContractSuite<IFontMapper>
{
    /// <inheritdoc />
    public override string Name => "font-mapper";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return new ContractCase("the legacy font and its map", new JsonObject { ["kind"] = "legacy" });
        yield return new ContractCase("a font that answers the letters itself", new JsonObject { ["kind"] = "unicode" });
        yield return new ContractCase("a map with an unknown key", new JsonObject { ["kind"] = "refused", ["json"] = """{"name": "x", "scripts": ["arab"], "letters": {"U+0628": {"isol": "#1"}}, "colour": 1}""", ["key"] = "colour" });
        yield return new ContractCase("a map with a bad form", new JsonObject { ["kind"] = "refused", ["json"] = """{"name": "x", "scripts": ["arab"], "letters": {"U+0628": {"isol": "#1", "middle": "#2"}}}""", ["key"] = "letters.U+0628.middle" });
        yield return new ContractCase("a map with a bad character", new JsonObject { ["kind"] = "refused", ["json"] = """{"name": "x", "scripts": ["arab"], "letters": {"B": {"isol": "#1"}}}""", ["key"] = "letters.B" });
        yield return new ContractCase("a map of glyph ids, three letters", IdCase(new Random(3), 3));
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        return new ContractCase("a random map of glyph ids", IdCase(random, random.Next(1, large ? 40 : 8)));
    }

    // Letters at Private Use code points, each with its plain glyph and an initial form, by glyph id of the other font.
    private JsonObject IdCase(Random random, int letters)
    {
        var data = new JsonObject { ["kind"] = "ids" };
        var list = new JsonArray();
        for (int i = 0; i < letters; i++)
        {
            int isol = 1 + random.Next(Math.Max(1, unicode.GlyphCount - 1)), init = 1 + random.Next(Math.Max(1, unicode.GlyphCount - 1));
            list.Add((JsonNode)new JsonArray(0xE000 + i, isol, init));
        }

        data["letters"] = list;
        return data;
    }

    /// <inheritdoc />
    public override void Run(IFontMapper implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var drawing = new FontMapDrawing { Digits = false, Marks = false };
        switch ((string)@case.Data["kind"]!)
        {
            case "legacy":
            {
                var mapped = implementation.Map(legacy, map, detect: true, drawing);
                checks.Check("the legacy font fits its map", mapped is not null, $"{legacy.Family} is not mapped by {map.Name}");
                if (mapped is null)
                {
                    return;
                }

                var wrong = new List<string>();
                int found = 0;
                foreach (var (code, letter) in map.Letters)
                {
                    int glyph = mapped.Glyph(code);
                    found += glyph != 0 ? 1 : 0;
                    if (Expected(letter, "isol") is int expected && expected != 0 && glyph != expected)
                    {
                        wrong.Add($"U+{code:X4}: {glyph}, the font's {expected}");
                    }
                }

                checks.Check("every letter answers its glyph", wrong.Count == 0 && found >= map.DetectShare * map.Letters.Count,
                    $"{found} of {map.Letters.Count} found; {string.Join("; ", wrong.Take(5))}");

                var gsub = mapped.Table("GSUB");
                var missing = new List<string>();
                foreach (var (code, letter) in map.Letters)
                {
                    foreach (string form in FontMap.Forms.Skip(1))
                    {
                        if (Expected(letter, form) is int to && to != 0 && to != mapped.Glyph(code) && Substitute(gsub.Span, form, mapped.Glyph(code)) != to)
                        {
                            missing.Add($"U+{code:X4}.{form}");
                        }
                    }
                }

                checks.Check("each form is reached through the made substitutions", missing.Count == 0, string.Join(", ", missing.Take(8)));
                Threads(mapped, [.. map.Letters.Keys], checks);
                break;
            }

            case "unicode":
                checks.Check("a font that answers the letters itself is not mapped when matched", implementation.Map(unicode, map, detect: true, drawing) is null,
                    $"{unicode.Family} was mapped by {map.Name}");
                checks.Check("an unmapped face is the same face", ReferenceEquals(FontMaps.Apply(unicode), unicode),
                    "FontMaps.Apply gave another face for a font that answers its letters itself");
                break;

            case "refused":
                try
                {
                    FontMap.Parse(Encoding.UTF8.GetBytes((string)@case.Data["json"]!), "case.map.json");
                    checks.Fail("a bad map is refused", "it was read");
                }
                catch (InvalidDataException e)
                {
                    string key = (string)@case.Data["key"]!;
                    checks.Check("a bad map is refused", e.Message.StartsWith("case.map.json:", StringComparison.Ordinal) && e.Message.Contains($"'{key}'", StringComparison.Ordinal),
                        e.Message);
                }

                break;

            default:
            {
                var letters = @case.Data["letters"]!.AsArray().Select(l => ((int)l![0]!, (int)l[1]!, (int)l[2]!)).ToList();
                var json = new StringBuilder("{\"name\": \"ids\", \"scripts\": [\"DFLT\"], \"letters\": {");
                json.AppendJoin(", ", letters.Select(l => string.Create(CultureInfo.InvariantCulture, $"\"U+{l.Item1:X4}\": {{\"isol\": \"#{l.Item2}\", \"init\": \"#{l.Item3}\"}}")));
                json.Append("}}");
                var ids = FontMap.Parse(Encoding.UTF8.GetBytes(json.ToString()), "ids.map.json");
                var mapped = implementation.Map(unicode, ids, detect: false, drawing);
                checks.Check("a map of glyph ids fits when given", mapped is not null, "not mapped");
                if (mapped is null)
                {
                    return;
                }

                // A glyph given for two letters keeps its first form only: compare each plain glyph's first.
                var first = new Dictionary<int, int>();
                foreach (var (_, isol, init) in letters)
                {
                    first.TryAdd(isol, init);
                }

                checks.Check("each character answers its glyph id", letters.All(l => mapped.Glyph(l.Item1) == l.Item2),
                    string.Join(", ", letters.Where(l => mapped.Glyph(l.Item1) != l.Item2).Select(l => $"U+{l.Item1:X4}: {mapped.Glyph(l.Item1)}, not {l.Item2}")));
                var gsub = mapped.Table("GSUB");
                var lost = first.Where(p => p.Key != p.Value && Substitute(gsub.Span, "init", p.Key) != p.Value).ToList();
                checks.Check("each form is reached through the made substitutions", lost.Count == 0, string.Join(", ", lost.Take(5).Select(p => $"{p.Key} → {p.Value}")));
                Threads(mapped, [.. letters.Select(l => l.Item1)], checks);
                break;
            }
        }
    }

    // The glyph of a form the map gives by the legacy font's code point (through the map's fallback); null when the
    // form is given another way (a name or an id the suite cannot read without the font's names).
    private int? Expected(FontMapLetter letter, string form)
    {
        if (!letter.Forms.TryGetValue(form, out var refs))
        {
            return null;
        }

        foreach (var r in refs)
        {
            if (r.Kind != FontGlyphRefKind.Code)
            {
                return null;
            }

            int glyph = legacy.Glyph(r.Value);
            if (glyph == 0 && map.Fallbacks.TryGetValue(r.Text, out var fallback) && fallback.Count > 0 && fallback[0].Kind == FontGlyphRefKind.Code)
            {
                glyph = legacy.Glyph(fallback[0].Value);
            }

            if (glyph != 0)
            {
                return glyph;
            }
        }

        return 0;
    }

    private static void Threads(IFontFace face, int[] codes, CaseChecks checks)
    {
        var expected = codes.Select(face.Glyph).ToArray();
        var answers = new int[4][];
        Parallel.For(0, answers.Length, i => answers[i] = [.. codes.Select(face.Glyph)]);
        checks.Check("the same answers from several threads", answers.All(a => a.SequenceEqual(expected)), "the answers differ between threads");
    }

    // What the GSUB's single substitutions of feature `tag` make of `glyph` (-1 when none applies): the feature list,
    // its lookups, their subtables of type 1 (formats 1 and 2) read by hand.
    private static int Substitute(ReadOnlySpan<byte> gsub, string tag, int glyph)
    {
        if (gsub.Length < 10)
        {
            return -1;
        }

        int U16(ReadOnlySpan<byte> t, int at) => at >= 0 && at + 2 <= t.Length ? BinaryPrimitives.ReadUInt16BigEndian(t[at..]) : 0;
        int featureList = U16(gsub, 6), lookupList = U16(gsub, 8);
        int features = U16(gsub, featureList);
        for (int f = 0; f < features; f++)
        {
            int record = featureList + 2 + 6 * f;
            if (Encoding.ASCII.GetString(gsub.Slice(record, 4)) != tag)
            {
                continue;
            }

            int feature = featureList + U16(gsub, record + 4);
            for (int k = 0; k < U16(gsub, feature + 2); k++)
            {
                int lookup = lookupList + U16(gsub, lookupList + 2 + 2 * U16(gsub, feature + 4 + 2 * k));
                if (U16(gsub, lookup) != 1)
                {
                    continue;
                }

                for (int s = 0; s < U16(gsub, lookup + 4); s++)
                {
                    int sub = lookup + U16(gsub, lookup + 6 + 2 * s);
                    int coverage = sub + U16(gsub, sub + 2);
                    int index = -1;
                    for (int c = 0; c < U16(gsub, coverage + 2) && U16(gsub, coverage) == 1; c++)
                    {
                        if (U16(gsub, coverage + 4 + 2 * c) == glyph)
                        {
                            index = c;
                        }
                    }

                    if (index >= 0)
                    {
                        return U16(gsub, sub) == 1 ? (glyph + (short)U16(gsub, sub + 4)) & 0xFFFF : U16(gsub, sub + 6 + 2 * index);
                    }
                }
            }
        }

        return -1;
    }
}
