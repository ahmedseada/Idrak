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
Reruns give the same bytes. Needs fonttools and freetype-py (pip).
"""

import json
import os
import sys

import freetype
from fontTools import subset
from fontTools.ttLib import TTFont
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
    print(f"{FONT}: {os.path.getsize(FONT)} bytes")
