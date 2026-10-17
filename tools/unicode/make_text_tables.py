"""Makes src/Idrak.Imaging/UnicodeText.Data.cs from a folder of the Unicode Character Database: the properties text
shaping and the bidirectional algorithm need and .NET does not give.

    python -I tools/unicode/make_text_tables.py UCD_FOLDER [OUT_FILE]

UCD_FOLDER holds one Unicode version's UnicodeData.txt, Scripts.txt, PropertyValueAliases.txt, BidiBrackets.txt,
BidiMirroring.txt, IndicSyllabicCategory.txt, IndicPositionalCategory.txt and, under extracted/, DerivedBidiClass.txt,
DerivedJoiningType.txt and DerivedCombiningClass.txt (https://www.unicode.org/Public/UCD/latest/ucd/, or the
unicodetools repository's unicodetools/data/ucd/<version>/). No Python package is needed.

Per code point, as ranges over 0..10FFFF (each range runs to the next start): the script (ISO 15924 code), the
bidirectional class, the joining type (Arabic, Syriac, N'Ko, Mongolian, ...; T for unlisted marks and format
characters as the derived file gives it), the canonical combining class, and the Indic syllabic and positional
categories; and the Arabic presentation forms, the lam-alef ligatures, the bidi mirroring pairs and the paired
brackets. Reruns give the same bytes.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "src", "Idrak.Imaging", "UnicodeText.Data.cs")
LAST = 0x10FFFF

BIDI = ["L", "R", "AL", "EN", "ES", "ET", "AN", "CS", "NSM", "BN", "B", "S", "WS", "ON", "LRE", "LRO", "RLE", "RLO", "PDF", "LRI", "RLI", "FSI", "PDI"]
JOINING = ["U", "R", "L", "D", "C", "T"]


def data_lines(path):
    """The fields of each data line (comments and blank lines skipped), stripped."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            body = line.split("#", 1)[0].strip()
            if body:
                yield [part.strip() for part in body.split(";")]


def missing_lines(path):
    """The (first, last, value) of each @missing line: the default of the code points no data line names."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            match = re.match(r"#\s*@missing:\s*([0-9A-Fa-f.]+)\s*;\s*([^;#\s]+)", line)
            if match:
                first, last = code_range(match.group(1))
                yield first, last, match.group(2)


def code_range(text):
    first, _, last = text.partition("..")
    return int(first, 16), int(last or first, 16)


def version(path):
    with open(path, encoding="utf-8") as f:
        return re.search(r"-(\d+\.\d+\.\d+)\.txt", f.readline()).group(1)


def aliases(folder, prop):
    """PropertyValueAliases.txt for one property: every name (short, long, others) to the short name."""
    names = {}
    for fields in data_lines(os.path.join(folder, "PropertyValueAliases.txt")):
        if fields[0] == prop:
            for name in fields[1:]:
                names[name] = fields[1]
    return names


def property_values(path, default, canonical):
    """A value per code point: the @missing defaults in file order, then the data lines, each through `canonical`."""
    values = [canonical(default)] * (LAST + 1)
    for first, last, value in missing_lines(path):
        values[first:last + 1] = [canonical(value)] * (last - first + 1)
    for fields in data_lines(path):
        first, last = code_range(fields[0])
        values[first:last + 1] = [canonical(fields[1])] * (last - first + 1)
    return values


def ranges(values):
    starts, out = [], []
    previous = None
    for cp, value in enumerate(values):
        if value != previous:
            starts.append(cp)
            out.append(value)
            previous = value
    return starts, out


def indexed(values, first):
    """Values as indexes into a name list that starts with `first` and grows in order of first appearance."""
    names = [first]
    out = []
    for value in values:
        if value not in names:
            names.append(value)
        out.append(names.index(value))
    return names, out


def presentation(folder):
    """The Arabic presentation forms (letter, form, code point) and the lam-alef ligatures (alef, final, code point)
    from the compatibility decompositions <isolated|final|initial|medial> of the two presentation blocks."""
    forms, lam_alef = {}, []
    names = {"<isolated>": 0, "<final>": 1, "<initial>": 2, "<medial>": 3}
    with open(os.path.join(folder, "UnicodeData.txt"), encoding="utf-8") as f:
        for line in f:
            fields = line.split(";")
            cp = int(fields[0], 16)
            if not (0xFB50 <= cp < 0xFE00 or 0xFE70 <= cp < 0xFF00):
                continue
            d = fields[5].split()
            if len(d) < 2 or d[0] not in names:
                continue
            parts = [int(x, 16) for x in d[1:]]
            if len(parts) == 1:
                forms.setdefault((parts[0], names[d[0]]), cp)
            elif len(parts) == 2 and parts[0] == 0x0644 and parts[1] in (0x0622, 0x0623, 0x0625, 0x0627) and d[0] in ("<isolated>", "<final>"):
                lam_alef.append((parts[1], 1 if d[0] == "<final>" else 0, cp))
    return sorted((b, f, c) for (b, f), c in forms.items()), sorted(lam_alef)


def block(name, starts, values, kind):
    lines = [f"    private static ReadOnlySpan<int> {name}Starts =>", "    ["]
    for i in range(0, len(starts), 12):
        lines.append("        " + ", ".join(f"0x{s:X}" for s in starts[i:i + 12]) + ",")
    lines.append("    ];")
    lines.append("")
    lines.append(f"    private static ReadOnlySpan<{kind}> {name}Values =>")
    lines.append("    [")
    for i in range(0, len(values), 24):
        lines.append("        " + ", ".join(str(v) for v in values[i:i + 24]) + ",")
    lines.append("    ];")
    return "\n".join(lines)


def rows(items, per_line):
    return "\n".join("        " + ", ".join(items[i:i + per_line]) + "," for i in range(0, len(items), per_line))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    folder = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else OUT
    extracted = os.path.join(folder, "extracted")
    ucd = version(os.path.join(extracted, "DerivedBidiClass.txt"))

    script_codes = aliases(folder, "sc")
    bidi_names = aliases(folder, "bc")
    joining_names = aliases(folder, "jt")
    positional_names = aliases(folder, "InPC")
    combining_names = aliases(folder, "ccc")                                   # Not_Reordered and the other names to their numbers
    scripts = property_values(os.path.join(folder, "Scripts.txt"), "Unknown", lambda v: script_codes[v])
    bidis = property_values(os.path.join(extracted, "DerivedBidiClass.txt"), "L", lambda v: BIDI.index(bidi_names[v]))
    joins = property_values(os.path.join(extracted, "DerivedJoiningType.txt"), "U", lambda v: JOINING.index(joining_names[v]))
    classes = property_values(os.path.join(extracted, "DerivedCombiningClass.txt"), "0", lambda v: int(combining_names.get(v, v)))
    syllabic = property_values(os.path.join(folder, "IndicSyllabicCategory.txt"), "Other", str)
    # The positional categories keep their long names but NA (the data lines use the long names, @missing the long default).
    positional = property_values(os.path.join(folder, "IndicPositionalCategory.txt"), "NA", lambda v: "NA" if positional_names.get(v) == "NA" else v)

    script_tags, scripts = indexed(scripts, "Zzzz")
    syllabic_names, syllabic = indexed(syllabic, "Other")
    positional_list, positional = indexed(positional, "NA")

    parts = []
    for name, values in (("Script", scripts), ("Bidi", bidis), ("Joining", joins), ("Combining", classes), ("Syllabic", syllabic), ("Positional", positional)):
        starts, values = ranges(values)
        parts.append(block(name, starts, values, "byte"))
    mirror = sorted((int(f[0], 16), int(f[1], 16)) for f in data_lines(os.path.join(folder, "BidiMirroring.txt")))
    brackets = sorted((int(f[0], 16), int(f[1], 16), 1 if f[2] == "o" else 0) for f in data_lines(os.path.join(folder, "BidiBrackets.txt")))
    forms, lam_alef = presentation(folder)
    names = lambda items: ", ".join(f'"{x}"' for x in items)
    text = f"""// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// <auto-generated>
// Made by tools/unicode/make_text_tables.py from the Unicode {ucd} character database (Scripts.txt, DerivedBidiClass.txt,
// DerivedJoiningType.txt, DerivedCombiningClass.txt, IndicSyllabicCategory.txt, IndicPositionalCategory.txt,
// UnicodeData.txt, BidiMirroring.txt, BidiBrackets.txt); do not edit. The data is Copyright (c) Unicode, Inc., used
// under the Unicode License v3 (https://www.unicode.org/license.txt).
// </auto-generated>

namespace Idrak.Imaging;

internal static partial class UnicodeText
{{
    /// <summary>The Unicode version the tables are made from.</summary>
    public const string Version = "{ucd}";

    /// <summary>ISO 15924 codes of the Script values (index 0: Zzzz, unknown).</summary>
    public static readonly string[] ScriptTags = [{names(script_tags)}];

    /// <summary>Indic syllabic categories of the Syllabic values (index 0: Other).</summary>
    public static readonly string[] SyllabicNames = [{names(syllabic_names)}];

    /// <summary>Indic positional categories of the Positional values (index 0: NA).</summary>
    public static readonly string[] PositionalNames = [{names(positional_list)}];

{chr(10).join(parts)}

    // Arabic presentation forms: letter, form (0 isolated, 1 final, 2 initial, 3 medial), presentation code point.
    private static ReadOnlySpan<int> PresentationForms =>
    [
{rows([f"0x{a:X}, {b}, 0x{c:X}" for a, b, c in forms], 6)}
    ];

    // Lam-alef ligatures: alef, 1 when final, presentation code point.
    private static ReadOnlySpan<int> LamAlef =>
    [
{rows([f"0x{a:X}, {b}, 0x{c:X}" for a, b, c in lam_alef], 6)}
    ];

    // Bidi mirroring pairs (BidiMirroring.txt): code point, its mirror (sorted by the first).
    private static ReadOnlySpan<int> Mirrors =>
    [
{rows([f"0x{a:X}, 0x{b:X}" for a, b in mirror], 6)}
    ];

    // Paired brackets (BidiBrackets.txt): bracket, its pair, 1 for an opening bracket (sorted by the first).
    private static ReadOnlySpan<int> Brackets =>
    [
{rows([f"0x{a:X}, 0x{b:X}, {c}" for a, b, c in brackets], 6)}
    ];
}}
"""
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"{out}: UCD {ucd}, {os.path.getsize(out)} bytes, {len(script_tags)} scripts, {len(mirror)} mirror pairs, {len(brackets)} brackets, "
          f"{len(forms)} presentation forms, {len(lam_alef)} lam-alef ligatures")


if __name__ == "__main__":
    main()
