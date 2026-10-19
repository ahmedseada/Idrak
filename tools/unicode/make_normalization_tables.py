"""Makes src/Idrak.Abstraction/Text/UnicodeNormalization.Data.cs from a folder of the Unicode Character Database: the
tables Unicode normalization (UAX #15: NFC, NFD, NFKC, NFKD) needs, so the library normalizes in C# on every platform
(.NET's own normalization goes through ICU or Windows, and does nothing under invariant globalization). Also writes the
conformance test file the tests read (NormalizationTest.txt without its comments, gzipped).

    python -I tools/unicode/make_normalization_tables.py UCD_FOLDER [OUT_FILE] [TEST_FILE]

UCD_FOLDER holds one Unicode version's UnicodeData.txt, DerivedNormalizationProps.txt and NormalizationTest.txt
(https://www.unicode.org/Public/UCD/latest/ucd/, or the unicodetools repository's unicodetools/data/ucd/<version>/,
which raw.githubusercontent.com serves). No Python package is needed. Reruns give the same bytes.

Per code point, through a two-stage table (a page of 256 code points, pages shared): an index into a list of records,
each a canonical combining class, the quick-check values of the four forms, whether a normalization segment may start
before the code point in each form, and the full canonical and compatibility decompositions (Hangul syllables are left
to the algorithm). Then the primary composites: (first, second) pairs, sorted, with the code point they compose to.
"""

import gzip
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "src", "Idrak.Abstraction", "Text", "UnicodeNormalization.Data.cs")
TEST = os.path.join(ROOT, "tests", "Idrak.Tests", "data", "unicode", "normalization-test.txt.gz")
LAST = 0x10FFFF
S_BASE, L_BASE, V_BASE, T_BASE, L_COUNT, V_COUNT, T_COUNT = 0xAC00, 0x1100, 0x1161, 0x11A7, 19, 21, 28
S_COUNT = L_COUNT * V_COUNT * T_COUNT

# Flag bits, as UnicodeNormalization.cs reads them.
NFD_NO, NFKD_NO, NFC_NO, NFC_MAYBE, NFKC_NO, NFKC_MAYBE = 1, 2, 4, 8, 16, 32
NFD_BOUNDARY, NFKD_BOUNDARY, NFC_BOUNDARY, NFKC_BOUNDARY = 64, 128, 256, 512


def data_lines(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            body = line.split("#", 1)[0].strip()
            if body:
                yield [part.strip() for part in body.split(";")]


def code_range(text):
    first, _, last = text.partition("..")
    return int(first, 16), int(last or first, 16)


def version(path):
    with open(path, encoding="utf-8") as f:
        return re.search(r"-(\d+\.\d+\.\d+)\.txt", f.readline()).group(1)


def is_hangul(cp):
    return S_BASE <= cp < S_BASE + S_COUNT


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    folder = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else OUT
    test_out = sys.argv[3] if len(sys.argv) > 3 else TEST
    props_path = os.path.join(folder, "DerivedNormalizationProps.txt")
    ucd = version(props_path)

    ccc = {}
    mapping = {}                      # code point -> (compatibility?, [code points]), one level, as UnicodeData.txt gives it
    with open(os.path.join(folder, "UnicodeData.txt"), encoding="utf-8") as f:
        for line in f:
            fields = line.split(";")
            cp = int(fields[0], 16)
            if int(fields[3]):
                ccc[cp] = int(fields[3])
            d = fields[5].split()
            if d:
                compat = d[0].startswith("<")
                mapping[cp] = (compat, [int(x, 16) for x in (d[1:] if compat else d)])

    def full(cp, compat):
        """The full decomposition: mappings applied until none applies, then canonically ordered."""
        if is_hangul(cp):
            s = cp - S_BASE
            parts = [L_BASE + s // (V_COUNT * T_COUNT), V_BASE + (s % (V_COUNT * T_COUNT)) // T_COUNT]
            return parts + ([T_BASE + s % T_COUNT] if s % T_COUNT else [])
        m = mapping.get(cp)
        if m is None or (m[0] and not compat):
            return [cp]
        result = []
        for part in m[1]:
            result.extend(full(part, compat))
        return reorder(result)

    def reorder(points):
        points = list(points)
        for i in range(1, len(points)):                                      # a stable insertion sort of non-starter runs
            j = i
            while j > 0 and ccc.get(points[j], 0) != 0 and ccc.get(points[j - 1], 0) > ccc.get(points[j], 0):
                points[j - 1], points[j] = points[j], points[j - 1]
                j -= 1
        return points

    quick = {}
    exclusions = set()
    for fields in data_lines(props_path):
        first, last = code_range(fields[0])
        name = fields[1]
        for cp in range(first, last + 1):
            if name == "Full_Composition_Exclusion":
                exclusions.add(cp)
            elif name in ("NFD_QC", "NFKD_QC", "NFC_QC", "NFKC_QC"):
                quick[(name, cp)] = fields[2]

    pairs = {}
    for cp, (compat, parts) in mapping.items():
        if not compat and len(parts) == 2 and cp not in exclusions:
            pairs[(parts[0], parts[1])] = cp
    seconds = {b for (_, b) in pairs}
    seconds.update(range(V_BASE, V_BASE + V_COUNT))
    seconds.update(range(T_BASE + 1, T_BASE + T_COUNT))

    def flags(cp):
        value = 0
        q = lambda name: quick.get((name, cp), "Y")
        if q("NFD_QC") == "N": value |= NFD_NO
        if q("NFKD_QC") == "N": value |= NFKD_NO
        if q("NFC_QC") == "N": value |= NFC_NO
        if q("NFC_QC") == "M": value |= NFC_MAYBE
        if q("NFKC_QC") == "N": value |= NFKC_NO
        if q("NFKC_QC") == "M": value |= NFKC_MAYBE
        canonical, compatibility = full(cp, False), full(cp, True)
        if ccc.get(canonical[0], 0) == 0: value |= NFD_BOUNDARY
        if ccc.get(compatibility[0], 0) == 0: value |= NFKD_BOUNDARY
        # A composing segment may start before a code point that is Yes, a starter, and decomposes to a starter that
        # composes with nothing before it: nothing earlier can compose with it or across it.
        if q("NFC_QC") == "Y" and ccc.get(cp, 0) == 0 and ccc.get(canonical[0], 0) == 0 and canonical[0] not in seconds:
            value |= NFC_BOUNDARY
        if q("NFKC_QC") == "Y" and ccc.get(cp, 0) == 0 and ccc.get(compatibility[0], 0) == 0 and compatibility[0] not in seconds:
            value |= NFKC_BOUNDARY
        return value

    # The records: the default first (index 0).
    default_flags = NFD_BOUNDARY | NFKD_BOUNDARY | NFC_BOUNDARY | NFKC_BOUNDARY
    mappings = []
    mapping_at = {}

    def store(points):
        key = tuple(points)
        if key not in mapping_at:
            mapping_at[key] = len(mappings)
            mappings.extend(points)
        return mapping_at[key] << 5 | len(points)

    records = [(0, default_flags, 0, 0)]
    record_at = {records[0]: 0}
    per_code_point = [0] * (LAST + 1)
    max_canonical = max_compatibility = 1
    for cp in range(LAST + 1):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = ccc.get(cp, 0)
        if c == 0 and cp not in mapping and (("NFC_QC", cp) not in quick) and (("NFKC_QC", cp) not in quick) and not is_hangul(cp):
            f = flags(cp)
            if f == default_flags:
                continue
        f = flags(cp)
        canonical = compatibility = 0
        if not is_hangul(cp):
            d = full(cp, False)
            k = full(cp, True)
            if d != [cp]:
                canonical = store(d)
                max_canonical = max(max_canonical, len(d))
            if k != [cp]:
                compatibility = store(k)
                max_compatibility = max(max_compatibility, len(k))
        record = (c, f, canonical, compatibility)
        if record not in record_at:
            record_at[record] = len(records)
            records.append(record)
        per_code_point[cp] = record_at[record]
    assert len(records) < 0x10000 and len(mappings) < (1 << 27)

    pages, page_at, stage1 = [], {}, []
    for block in range(0, LAST + 1, 256):
        page = tuple(per_code_point[block:block + 256])
        if page not in page_at:
            page_at[page] = len(pages)
            pages.append(page)
        stage1.append(page_at[page])
    assert len(pages) < 0x10000

    pair_list = sorted(pairs.items())

    def rows(items, per_line):
        return "\n".join("        " + ", ".join(items[i:i + per_line]) + "," for i in range(0, len(items), per_line))

    stage2 = [v for page in pages for v in page]
    text = f"""// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// <auto-generated>
// Made by tools/unicode/make_normalization_tables.py from the Unicode {ucd} character database (UnicodeData.txt,
// DerivedNormalizationProps.txt); do not edit. The data is Copyright (c) Unicode, Inc., used under the Unicode License
// v3 (https://www.unicode.org/license.txt).
// </auto-generated>

namespace Idrak.Abstraction.Text;

public static partial class UnicodeNormalization
{{
    /// <summary>The Unicode version the normalization tables are made from.</summary>
    public const string UnicodeVersion = "{ucd}";

    // The longest full decompositions, canonical and compatibility (Hangul syllables decompose to at most 3).
    private const int MaxCanonical = {max(max_canonical, 3)}, MaxCompatibility = {max(max_compatibility, 3)};

    // A code point's record: Records[Stage2[Stage1[cp >> 8] << 8 | (cp & 0xFF)]].
    private static ReadOnlySpan<ushort> Stage1 =>
    [
{rows([str(v) for v in stage1], 32)}
    ];

    private static ReadOnlySpan<ushort> Stage2 =>
    [
{rows([str(v) for v in stage2], 32)}
    ];

    // Per record: the quick-check and segment flags (the constants in UnicodeNormalization.cs), plus the canonical
    // combining class shifted left 16.
    private static ReadOnlySpan<uint> RecordInfo =>
    [
{rows([str(r[1] | r[0] << 16) for r in records], 24)}
    ];

    // Per record: the full canonical decomposition (its start in Mappings shifted left 5, plus its length; 0 for none).
    private static ReadOnlySpan<int> RecordCanonical =>
    [
{rows([str(r[2]) for r in records], 24)}
    ];

    // Per record: the full compatibility decomposition, as RecordCanonical.
    private static ReadOnlySpan<int> RecordCompatibility =>
    [
{rows([str(r[3]) for r in records], 24)}
    ];

    // The decompositions' code points.
    private static ReadOnlySpan<int> Mappings =>
    [
{rows([f"0x{v:X}" for v in mappings], 16)}
    ];

    // The primary composites: (first << 21 | second), sorted, and the code point each pair composes to.
    private static ReadOnlySpan<long> CompositionPairs =>
    [
{rows([f"0x{a << 21 | b:X}" for (a, b), _ in pair_list], 12)}
    ];

    private static ReadOnlySpan<int> Composites =>
    [
{rows([f"0x{c:X}" for _, c in pair_list], 16)}
    ];
}}
"""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

    # The conformance file without comments (its five columns, the @Part lines kept).
    lines = [f"# NormalizationTest-{ucd}.txt from the Unicode character database, comments removed by",
             "# tools/unicode/make_normalization_tables.py. Copyright (c) Unicode, Inc., Unicode License v3."]
    with open(os.path.join(folder, "NormalizationTest.txt"), encoding="utf-8") as f:
        for line in f:
            body = line.split("#", 1)[0].strip()
            if body:
                lines.append(body)
    raw = ("\n".join(lines) + "\n").encode("utf-8")
    os.makedirs(os.path.dirname(test_out), exist_ok=True)
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as g:
        g.write(raw)
    with open(test_out, "wb") as f:
        f.write(buffer.getvalue())

    print(f"{out}: UCD {ucd}, {os.path.getsize(out)} bytes, {len(pages)} pages, {len(records)} records, "
          f"{len(mappings)} mapping code points, {len(pair_list)} composition pairs, longest {max_canonical}/{max_compatibility}")
    print(f"{test_out}: {len(lines) - 2} lines, {os.path.getsize(test_out)} bytes")


if __name__ == "__main__":
    main()
