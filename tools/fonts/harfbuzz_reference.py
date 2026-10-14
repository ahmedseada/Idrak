"""HarfBuzz's shaping of lines in fonts, for idrak data fonts --harfbuzz (any font, any script).

    python -I tools/fonts/harfbuzz_reference.py FONTS LINES -o REFERENCES.json [--script Arab] [--direction rtl]
        [--language ar] [--clusters graphemes|characters]

FONTS is a font file or a folder (searched through, .ttf .otf .ttc; every face of a collection); LINES a UTF-8 text
file, one line a shaping (blank lines and lines of --- passed over), such as the one idrak data fonts --write-lines or
idrak data text --unit lines writes. Writes a JSON list, an entry a font and line: {"font": the file relative to FONTS,
"face": its index in a collection, "text", "script", "direction", "language", "clusters", "glyphs": [[glyph, cluster,
x advance, y advance, x offset, y offset], ...]}, in font units; clusters are UTF-16 indices, as idrak's. Needs
uharfbuzz (pip install uharfbuzz).
"""

import argparse
import json
import os

import uharfbuzz as hb


def fonts(root):
    if os.path.isfile(root):
        yield os.path.dirname(os.path.abspath(root)), os.path.abspath(root)
        return
    for folder, _, files in sorted(os.walk(root)):
        for name in sorted(files):
            if name.lower().endswith((".ttf", ".otf", ".ttc")):
                yield os.path.abspath(root), os.path.join(folder, name)


def utf16_clusters(text):
    # HarfBuzz's add_str counts code points; idrak counts UTF-16 code units.
    at, index = [], 0
    for ch in text:
        at.append(index)
        index += 2 if ord(ch) > 0xFFFF else 1
    return at


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fonts")
    parser.add_argument("lines")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("--script", default="Arab")
    parser.add_argument("--direction", default="rtl", choices=["rtl", "ltr"])
    parser.add_argument("--language", default="ar")
    parser.add_argument("--clusters", default="graphemes", choices=["graphemes", "characters"])
    args = parser.parse_args()

    with open(args.lines, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and line.strip() != "---"]
    level = hb.BufferClusterLevel.MONOTONE_GRAPHEMES if args.clusters == "graphemes" else hb.BufferClusterLevel.MONOTONE_CHARACTERS
    references = []
    for root, path in fonts(args.fonts):
        blob = hb.Blob.from_file_path(path)
        with open(path, "rb") as f:
            head = f.read(12)
        count = int.from_bytes(head[8:12], "big") if head[:4] == b"ttcf" else 1   # a collection's fonts
        for index in range(max(1, count)):
            font = hb.Font(hb.Face(blob, index))
            for text in lines:
                buf = hb.Buffer()
                buf.add_str(text)
                buf.direction = args.direction
                buf.script = args.script
                buf.language = args.language
                buf.cluster_level = level
                hb.shape(font, buf)
                at = utf16_clusters(text)
                glyphs = [[i.codepoint, at[i.cluster] if i.cluster < len(at) else i.cluster, p.x_advance, p.y_advance, p.x_offset, p.y_offset]
                          for i, p in zip(buf.glyph_infos, buf.glyph_positions)]
                references.append({"font": os.path.relpath(path, root).replace(os.sep, "/"), "face": index, "text": text, "script": args.script,
                                   "direction": args.direction, "language": args.language, "clusters": args.clusters, "glyphs": glyphs})

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(references, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    print(f"{len(references)} shapings of {len(lines)} lines written to {args.out}")


if __name__ == "__main__":
    main()
