"""Makes src/Idrak.Imaging/UnicodeText.Data.cs: the Unicode properties text shaping needs and .NET does not give (plan 18).

    python -I tools/unicode/make_text_tables.py

Per code point, as ranges over 0..10FFFF (each range runs to the next start): the script (ISO 15924 code), the
bidirectional class, the joining type (Arabic, Syriac, N'Ko, Mongolian, ...), the canonical combining class, and the Indic
syllabic and positional categories; and the bidi mirroring pairs. From unicodedataplus (Unicode 16) and regex's joining
types (pip install unicodedataplus regex). Reruns give the same bytes.
"""

import os
import unicodedata as legacy

import regex
import unicodedataplus as u

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "src", "Idrak.Imaging", "UnicodeText.Data.cs")
LAST = 0x10FFFF

SCRIPT_CODE = {name: aliases[0] for name, aliases in u.property_value_aliases["script"].items()}
BIDI = ["L", "R", "AL", "EN", "ES", "ET", "AN", "CS", "NSM", "BN", "B", "S", "WS", "ON", "LRE", "LRO", "RLE", "RLO", "PDF", "LRI", "RLI", "FSI", "PDI"]
JOINING = ["U", "R", "L", "D", "C", "T"]
JOINING_NAMES = {"R": "Right_Joining", "L": "Left_Joining", "D": "Dual_Joining", "C": "Join_Causing", "T": "Transparent"}
JOINING_PATTERNS = {k: regex.compile(r"\p{Joining_Type=" + v + "}") for k, v in JOINING_NAMES.items()}

# Unassigned code points' default bidi class (DerivedBidiClass.txt's @missing lines, the main ones).
RTL_DEFAULTS = [(0x0590, 0x05FF, "R"), (0x0600, 0x07BF, "AL"), (0x07C0, 0x085F, "R"), (0x0860, 0x08FF, "AL"), (0xFB1D, 0xFB4F, "R"),
                (0xFB50, 0xFDCF, "AL"), (0xFDF0, 0xFDFF, "AL"), (0xFE70, 0xFEFF, "AL"), (0x10800, 0x10CFF, "R"), (0x10D00, 0x10D3F, "AL"),
                (0x10D40, 0x10EBF, "R"), (0x10EC0, 0x10EFF, "AL"), (0x10F00, 0x10F2F, "R"), (0x10F30, 0x10F6F, "AL"), (0x10F70, 0x10FFF, "R"),
                (0x1E800, 0x1EC6F, "R"), (0x1EC70, 0x1ECBF, "AL"), (0x1ECC0, 0x1ECFF, "R"), (0x1ED00, 0x1ED4F, "AL"), (0x1ED50, 0x1EDFF, "R"),
                (0x1EE00, 0x1EEFF, "AL"), (0x1EF00, 0x1EFFF, "R")]


def bidi(cp):
    c = u.bidirectional(chr(cp))
    if c:
        return c
    for start, end, value in RTL_DEFAULTS:
        if start <= cp <= end:
            return value
    return "BN" if u.category(chr(cp)) == "Cn" and (0xFDD0 <= cp <= 0xFDEF or (cp & 0xFFFE) == 0xFFFE) else "L"


def joining(cp):
    ch = chr(cp)
    for key, pattern in JOINING_PATTERNS.items():
        if pattern.match(ch):
            return key
    return "U"


def ranges(values):
    starts, out = [], []
    previous = None
    for cp, value in enumerate(values):
        if value != previous:
            starts.append(cp)
            out.append(value)
            previous = value
    return starts, out


def mirrors():
    pairs = {}
    swaps = [("LEFT", "RIGHT"), ("LESS-THAN", "GREATER-THAN"), ("OPENING", "CLOSING"), ("LEFT-POINTING", "RIGHT-POINTING")]
    for cp in range(0x20, 0x2E7F + 1):
        ch = chr(cp)
        if not u.mirrored(ch):
            continue
        name = u.name(ch, "")
        for a, b in swaps:
            for x, y in ((a, b), (b, a)):
                if x in name:
                    other = name.replace(x, "\0").replace(y, x).replace("\0", y)
                    try:
                        target = ord(legacy.lookup(other))
                    except KeyError:
                        continue
                    if u.mirrored(chr(target)) and target != cp:
                        pairs.setdefault(cp, target)
    return sorted(pairs.items())


def presentation():
    forms, lam_alef = {}, []
    names = {"<isolated>": 0, "<final>": 1, "<initial>": 2, "<medial>": 3}
    for cp in list(range(0xFB50, 0xFE00)) + list(range(0xFE70, 0xFF00)):
        d = legacy.decomposition(chr(cp)).split()
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


def main():
    scripts, bidis, joins, classes, syllabic, positional = [], [], [], [], [], []
    script_tags, syllabic_names, positional_names = ["Zzzz"], ["Other"], ["NA"]
    for cp in range(LAST + 1):
        ch = chr(cp)
        tag = SCRIPT_CODE.get(u.script(ch), "Zzzz")
        if tag not in script_tags:
            script_tags.append(tag)
        scripts.append(script_tags.index(tag))
        bidis.append(BIDI.index(bidi(cp)))
        joins.append(JOINING.index(joining(cp)))
        classes.append(u.combining(ch))
        s = u.indic_syllabic_category(ch) or "Other"
        if s not in syllabic_names:
            syllabic_names.append(s)
        syllabic.append(syllabic_names.index(s))
        p = u.indic_positional_category(ch) or "NA"
        if p not in positional_names:
            positional_names.append(p)
        positional.append(positional_names.index(p))

    parts = []
    for name, values in (("Script", scripts), ("Bidi", bidis), ("Joining", joins), ("Combining", classes), ("Syllabic", syllabic), ("Positional", positional)):
        starts, out = ranges(values)
        parts.append(block(name, starts, out, "byte"))
    mirror = mirrors()
    forms, lam_alef = presentation()
    names = lambda items: ", ".join(f'"{x}"' for x in items)
    text = f"""// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// <auto-generated>
// Made by tools/unicode/make_text_tables.py from the Unicode {u.unidata_version} character database; do not edit.
// </auto-generated>

namespace Idrak.Imaging;

internal static partial class UnicodeText
{{
    /// <summary>The Unicode version the tables are made from.</summary>
    public const string Version = "{u.unidata_version}";

    /// <summary>ISO 15924 codes of the Script values (index 0: Zzzz, unknown).</summary>
    public static readonly string[] ScriptTags = [{names(script_tags)}];

    /// <summary>Indic syllabic categories of the Syllabic values (index 0: Other).</summary>
    public static readonly string[] SyllabicNames = [{names(syllabic_names)}];

    /// <summary>Indic positional categories of the Positional values (index 0: NA).</summary>
    public static readonly string[] PositionalNames = [{names(positional_names)}];

{chr(10).join(parts)}

    // Arabic presentation forms: letter, form (0 isolated, 1 final, 2 initial, 3 medial), presentation code point.
    private static ReadOnlySpan<int> PresentationForms =>
    [
{chr(10).join("        " + ", ".join(f"0x{a:X}, {b}, 0x{c:X}" for a, b, c in forms[i:i + 6]) + "," for i in range(0, len(forms), 6))}
    ];

    // Lam-alef ligatures: alef, 1 when final, presentation code point.
    private static ReadOnlySpan<int> LamAlef =>
    [
{chr(10).join("        " + ", ".join(f"0x{a:X}, {b}, 0x{c:X}" for a, b, c in lam_alef[i:i + 6]) + "," for i in range(0, len(lam_alef), 6))}
    ];

    // Bidi mirroring pairs: code point, its mirror (sorted by the first).
    private static ReadOnlySpan<int> Mirrors =>
    [
{chr(10).join("        " + ", ".join(f"0x{a:X}, 0x{b:X}" for a, b in mirror[i:i + 6]) + "," for i in range(0, len(mirror), 6))}
    ];
}}
"""
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"{OUT}: {os.path.getsize(OUT)} bytes, {len(script_tags)} scripts, {len(mirror)} mirror pairs")


if __name__ == "__main__":
    main()
