"""Makes the font fixtures of the "fonts:" and "shaping:" tests (plan 18).

    python -I tools/fonts/make_font_references.py [path to DejaVuSans.ttf]

Writes under tests/Idrak.Tests/data/fonts:
  dejavu-sans-subset.ttf   DejaVu Sans cut to Arabic, Latin, digits and punctuation, its layout tables (GSUB, GPOS,
                           GDEF) kept, hinting dropped (the DejaVu license, LICENSE-DejaVu.txt, allows it)
  freetype/*.bin           FreeType's anti-aliased renders of chosen glyphs (no hinting), one byte per pixel
  freetype.json            per render: the text, glyph id, pixels per em, the bitmap's left and top and its size
  face.json                fontTools' reading of the subset: names, metrics, the character map, every advance and
                           every glyph's bounds (composites resolved)
  dejavu-sans-pair.ttc     a collection of the subset and DejaVu Sans Bold cut the same way
  dejavu-sans-nogsub.ttf   the subset without GSUB (Arabic is then shaped with Unicode's presentation forms)
  shaping.json             HarfBuzz's shaping of fixed strings in several scripts (glyph, cluster, advances, offsets)
  bidi.json                python-bidi's visual order of mixed paragraphs, each direction given and found
  features.ttf             DejaVu Sans' Latin letters with layout features of our own (FEATURES below) that use every
                           GSUB and GPOS lookup kind: contexts of every format, multiple and reverse chaining
                           substitution, extension lookups, mark filtering sets, single and pair (class) positioning,
                           cursive attachment both ways, mark to base and mark to mark, contextual positioning
Reruns give the same bytes. Needs fonttools, freetype-py, uharfbuzz and python-bidi (pip).
"""

import json
import os
import sys

import freetype
import uharfbuzz as hb
from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
from fontTools.ttLib.ttCollection import TTCollection

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "tests", "Idrak.Tests", "data", "fonts")
SOURCE = sys.argv[1] if len(sys.argv) > 1 else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT = os.path.join(OUT, "dejavu-sans-subset.ttf")

# Arabic letters, marks, digits and punctuation; Latin letters, digits and the punctuation of legal texts; é (a composite).
CODE_POINTS = (
    list(range(0x0621, 0x063B)) + list(range(0x0640, 0x0653)) + list(range(0x0660, 0x066A))
    + [0x060C, 0x061B, 0x061F, 0x0670, 0x0671, 0x067E, 0x0686, 0x06A4, 0x06AF, 0x06CC]
    + list(range(0x20, 0x7F)) + [0xE9, 0xA0]
    + list(range(0x05B0, 0x05C8)) + list(range(0x05D0, 0x05EB))         # Hebrew letters and points
    + list(range(0x0391, 0x03CA)) + list(range(0x0410, 0x0450))         # Greek and Cyrillic
    + [0x200C, 0x200D, 0x25CC, 0x0670] + list(range(0xFE70, 0xFEFD))    # ZWNJ, ZWJ, dotted circle; Arabic presentation forms
)


def make_subset(source=SOURCE, target=FONT):
    options = subset.Options()
    options.layout_features = ["*"]
    options.hinting = False
    options.glyph_names = False
    options.notdef_outline = True
    options.name_IDs = [0, 1, 2, 3, 4, 5, 6]
    options.drop_tables += ["FFTM", "MATH"]
    font = TTFont(source, recalcTimestamp=False)
    subsetter = subset.Subsetter(options)
    subsetter.populate(unicodes=CODE_POINTS)
    subsetter.subset(font)
    font.recalcTimestamp = False
    font.save(target)


def describe():
    font = TTFont(FONT)
    names = font["name"]
    glyf = font["glyf"]
    order = font.getGlyphOrder()
    bounds = []
    for name in order:
        g = glyf[name]
        if g.numberOfContours == 0:
            bounds.append(None)
            continue
        g.recalcBounds(glyf)
        bounds.append([g.xMin, g.yMin, g.xMax, g.yMax])
    face = {
        "family": names.getDebugName(16) or names.getDebugName(1), "style": names.getDebugName(17) or names.getDebugName(2),
        "unitsPerEm": font["head"].unitsPerEm, "ascender": font["hhea"].ascent, "descender": font["hhea"].descent,
        "lineGap": font["hhea"].lineGap, "glyphCount": len(order),
        "cmap": sorted([cp, font.getGlyphID(n)] for cp, n in font.getBestCmap().items()),
        "advances": [font["hmtx"][n][0] for n in order], "bounds": bounds,
    }
    with open(os.path.join(OUT, "face.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(face, f, separators=(",", ":"))
        f.write("\n")


def collection():
    bold = os.path.join(OUT, "bold.tmp.ttf")
    make_subset(SOURCE.replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"), bold)
    pair = TTCollection()
    pair.fonts = [TTFont(FONT, recalcTimestamp=False), TTFont(bold, recalcTimestamp=False)]
    for f in pair.fonts:
        f.recalcTimestamp = False
    pair.save(os.path.join(OUT, "dejavu-sans-pair.ttc"))
    os.remove(bold)


FEATURES = """
languagesystem DFLT dflt;
languagesystem latn dflt;
languagesystem arab dflt;
@LOWER = [a b c d e f g h i k l m n o p q r s t u v w x y z];
@VOWEL = [a e i o u];
@CONS = [b c d g h k l m n p r s t];
markClass [gravecomb acutecomb] <anchor 0 1500> @TOP;
markClass [uni0302] <anchor 0 1700> @HAT;

lookup UPPER_A { sub a by A; } UPPER_A;
lookup UPPER_ANY { sub @LOWER by [A B C D E F G H I K L M N O P Q R S T U V W X Y Z]; } UPPER_ANY;
lookup DOUBLE_X { sub x by x x; } DOUBLE_X;
lookup CAT useExtension { sub c a t by X; sub d o g by Y; } CAT;

feature calt {
    sub [b c] a' lookup UPPER_A;                        # chained context, one input
    sub q' lookup DOUBLE_X u;                           # chained context with lookahead
    sub @CONS @VOWEL' lookup UPPER_ANY @CONS @CONS;     # class context, backtrack and lookahead
} calt;

feature liga {
    lookup CAT;
} liga;

feature rlig {
    rsub [m n] e' [m n] by E;                           # reverse chaining
} rlig;

lookup CURSIVE_LTR {
    pos cursive o <anchor 0 300> <anchor 1100 400>;
    pos cursive w <anchor 0 500> <anchor 1500 200>;
} CURSIVE_LTR;

lookup CURSIVE_RTL {
    lookupflag RightToLeft;
    pos cursive v <anchor 0 100> <anchor 1000 300>;
    pos cursive z <anchor 0 250> <anchor 900 50>;
} CURSIVE_RTL;

lookup CURSIVE_ARAB {
    lookupflag RightToLeft;
    pos cursive uni0628 <anchor 0 100> <anchor 900 300>;
    pos cursive uni0645 <anchor 0 250> <anchor 700 0>;
    pos cursive uni062A <anchor 0 0> <anchor 600 200>;
} CURSIVE_ARAB;

lookup CURSIVE_ARAB_LTR_FLAG {
    pos cursive uni062B <anchor 0 150> <anchor 800 350>;
} CURSIVE_ARAB_LTR_FLAG;

feature kern {
    pos [A V] [V A] -150;
    pos T o -120;
    enum pos @CONS [a o] -40;
    pos x <30 0 60 0>;
} kern;

feature curs {
    lookup CURSIVE_LTR;
    lookup CURSIVE_RTL;
    lookup CURSIVE_ARAB;
    lookup CURSIVE_ARAB_LTR_FLAG;
} curs;

feature mark {
    pos base [a e o u i] <anchor 560 1150> mark @TOP <anchor 560 1300> mark @HAT;
} mark;

feature mkmk {
    lookupflag UseMarkFilteringSet [acutecomb];
    pos mark [gravecomb acutecomb] <anchor 0 2000> mark @TOP;
} mkmk;

feature dist {
    pos [k] [l]' 77 [m];                                # contextual positioning
} dist;
"""


def features_font():
    options = subset.Options()
    options.layout_features = []
    options.hinting = False
    options.glyph_names = True
    options.notdef_outline = True
    options.drop_tables += ["FFTM", "MATH", "kern", "GSUB", "GPOS", "GDEF"]
    font = TTFont(SOURCE, recalcTimestamp=False)
    subsetter = subset.Subsetter(options)
    subsetter.populate(unicodes=list(range(0x20, 0x7F)) + [0x0300, 0x0301, 0x0302, 0x0628, 0x062A, 0x062B, 0x0645])
    subsetter.subset(font)
    addOpenTypeFeaturesFromString(font, FEATURES)
    font.recalcTimestamp = False
    font.save(os.path.join(OUT, "features.ttf"))


# (font file, text, direction, script, language, features): one run each.
SHAPINGS = [
    ("dejavu-sans-subset.ttf", "بسم الله الرحمن الرحيم", "rtl", "Arab", "ar", None),
    ("dejavu-sans-subset.ttf", "المملكة العربية السعودية", "rtl", "Arab", "ar", None),
    ("dejavu-sans-subset.ttf", "ديوان رئاسة مجلس الوزراء", "rtl", "Arab", None, None),
    ("dejavu-sans-subset.ttf", "لا إله إلا الله", "rtl", "Arab", None, None),
    ("dejavu-sans-subset.ttf", "مُحَمَّدٌ وَعَلِيٌّ", "rtl", "Arab", None, None),
    ("dejavu-sans-subset.ttf", "كـتـاب", "rtl", "Arab", None, None),
    ("dejavu-sans-subset.ttf", "سلام، عليكم؛ كيف حالك؟", "rtl", "Arab", None, None),
    ("dejavu-sans-subset.ttf", "(النظام) [المادة]", "rtl", "Arab", None, None),
    ("dejavu-sans-subset.ttf", "ي بب ببب", "rtl", "Arab", None, None),
    ("dejavu-sans-subset.ttf", "پنجره‌ها گچ", "rtl", "Arab", "fa", None),
    ("dejavu-sans-subset.ttf", "١٤٢٦/٥/٢١هـ", "rtl", "Arab", None, None),
    ("dejavu-sans-subset.ttf", "أؤإئآ ة ى", "rtl", "Arab", None, None),
    ("dejavu-sans-subset.ttf", "AVATAR Wave To", "ltr", "Latn", "en", None),
    ("dejavu-sans-subset.ttf", "Hello, World! (2026) café", "ltr", "Latn", None, None),
    ("dejavu-sans-subset.ttf", "office fi", "ltr", "Latn", None, ["-liga"]),
    ("dejavu-sans-subset.ttf", "שָׁלוֹם עולם", "rtl", "Hebr", "he", None),
    ("dejavu-sans-subset.ttf", "Καλημέρα κόσμε", "ltr", "Grek", None, None),
    ("dejavu-sans-subset.ttf", "Привет, мир", "ltr", "Cyrl", "ru", None),
    ("dejavu-sans-subset.ttf", "1426/5/21", "ltr", "Zyyy", None, None),
    ("dejavu-sans-nogsub.ttf", "بسم الله الرحمن", "rtl", "Arab", None, None),
    ("features.ttf", "ba ca da", "ltr", "Latn", None, None),
    ("features.ttf", "quiet qu", "ltr", "Latn", None, None),
    ("features.ttf", "basket pitch", "ltr", "Latn", None, None),
    ("features.ttf", "cat dog catdog", "ltr", "Latn", None, None),
    ("features.ttf", "mem nen menem", "ltr", "Latn", None, None),
    ("features.ttf", "AVATAR To xx", "ltr", "Latn", None, None),
    ("features.ttf", "oowwo vvzv", "ltr", "Latn", None, None),
    ("features.ttf", "oowwo vvzv", "rtl", "Latn", None, None),
    ("features.ttf", "بمتبم ثثث بثم", "rtl", "Arab", None, None),
    ("features.ttf", "vvzv بمت", "ltr", "Arab", None, None),
    ("features.ttf", "a\u0300 e\u0301\u0300 o\u0300\u0301 u\u0302 i\u0301\u0301", "ltr", "Latn", None, None),
    ("features.ttf", "klm kl lm", "ltr", "Latn", None, None),
    ("features.ttf", "cat dog", "ltr", "Latn", None, ["-liga", "-kern"]),
]


BIDI = [
    "abc (النظام) 123 ١٤٢٦/٥/٢١", "بسم الله الرحمن الرحيم", "المادة 12 من النظام", "رقم (م/٥٥) وتاريخ ١٤٠٩/١٠/٢٣هـ",
    "Hello عالم world", "The price is 50% (خمسون) today.", "عنوان: http://x.y/a?b=1 هنا", "1-2-3 ٤-٥", "שלום, world!",
    "abc", "  مرحبا  ", "a\u200fb c", "(abc) [عربي]", "سعر 3.5 ريال، و 1,000.25 دولار", "عدد -5 و +7", "قال: «نعم».",
]


def bidi_references():
    from bidi import get_display
    rows = []
    for text in BIDI:
        for base in ("L", "R", None):
            visual = get_display(text, base_dir=base) if base else get_display(text)
            rows.append({"text": text, "base": base, "visual": visual})
    with open(os.path.join(OUT, "bidi.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
        f.write("\n")


def shape():
    nogsub = TTFont(FONT, recalcTimestamp=False)
    del nogsub["GSUB"]
    nogsub.recalcTimestamp = False
    nogsub.save(os.path.join(OUT, "dejavu-sans-nogsub.ttf"))
    rows = []
    for file, text, direction, script, language, features in SHAPINGS:
        with open(os.path.join(OUT, file), "rb") as f:
            face = hb.Face(f.read())
        font = hb.Font(face)
        buf = hb.Buffer()
        buf.add_codepoints([ord(c) for c in text])
        buf.direction = direction
        buf.script = script
        if language:
            buf.language = language
        buf.cluster_level = hb.BufferClusterLevel.MONOTONE_CHARACTERS
        feats = {}
        for change in features or []:
            feats[change[1:]] = change[0] == "+"
        hb.shape(font, buf, feats)
        glyphs = [[i.codepoint, i.cluster, p.x_advance, p.y_advance, p.x_offset, p.y_offset] for i, p in zip(buf.glyph_infos, buf.glyph_positions)]
        rows.append({"font": file, "text": text, "direction": direction, "script": script, "language": language, "features": features, "glyphs": glyphs})
    with open(os.path.join(OUT, "shaping.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
        f.write("\n")


# The renders: (text of one code point, pixels per em).
RENDERS = [("A", 24), ("g", 24), ("é", 24), ("0", 48), ("&", 48),
           ("ب", 24), ("ج", 48), ("س", 48), ("ع", 24), ("ه", 48), ("أ", 48), ("ك", 24), ("٣", 48), ("ﻻ", 0)]


def render():
    face = freetype.Face(FONT)
    cmap = TTFont(FONT).getBestCmap()
    folder = os.path.join(OUT, "freetype")
    os.makedirs(folder, exist_ok=True)
    rows = []
    for text, ppem in RENDERS:
        if ppem == 0 or ord(text) not in cmap:
            continue
        glyph = face.get_char_index(ord(text))
        face.set_pixel_sizes(0, ppem)
        face.load_glyph(glyph, freetype.FT_LOAD_NO_HINTING | freetype.FT_LOAD_NO_BITMAP)
        face.glyph.render(freetype.FT_RENDER_MODE_NORMAL)
        bitmap = face.glyph.bitmap
        name = f"u{ord(text):04x}-{ppem}.bin"
        data = bytes(bitmap.buffer[r * bitmap.pitch + c] for r in range(bitmap.rows) for c in range(bitmap.width))
        with open(os.path.join(folder, name), "wb") as f:
            f.write(data)
        rows.append({"text": text, "glyph": glyph, "ppem": ppem, "left": face.glyph.bitmap_left, "top": face.glyph.bitmap_top,
                     "width": bitmap.width, "rows": bitmap.rows, "file": f"freetype/{name}"})
    with open(os.path.join(OUT, "freetype.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
        f.write("\n")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    make_subset()
    render()
    describe()
    collection()
    features_font()
    shape()
    bidi_references()
    print(f"{FONT}: {os.path.getsize(FONT)} bytes")
