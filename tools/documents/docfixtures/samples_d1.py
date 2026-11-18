"""D1, the file: cross-reference tables and streams, object streams, hybrid files, incremental updates, bytes before
the header, damaged tables, wrong lengths, every filter and predictor, document information, the page tree, and the
hostile files (each marked with the limit it must trip)."""
import zlib

from .common import Sample, base_expected, helvetica, text_page
from .document import Doc
from .filters import a85, ahx, flate, lzw, rle
from .images import Image, pattern
from .pdf import B, H, N, Raw, Stream, U16, Writer, ser, sha256

D1 = ["D1"]

REPORT = ["Quarterly report", "Sales rose in the third quarter.", "Costs stayed within the plan.",
          "The new warehouse opens in May."]
NEXT = ["Next steps", "Hire two engineers for the platform team.", "Review the supplier contracts."]


def _sample(name, kind, desc, data, exp, phases=D1, files=None, hostile=None):
    return Sample(name, "d1", kind, phases, desc, data, exp, files, hostile)


def _two_pages(doc):
    f = helvetica(doc)
    text_page(doc, f, REPORT)
    text_page(doc, f, NEXT)
    return f


def _plain_info(doc):
    doc.info = {"Title": B(b"Quarterly report"), "Producer": B(b"Idrak document samples")}
    return {"title": "Quarterly report", "author": None, "subject": None, "keywords": None, "creator": None,
            "producer": "Idrak document samples", "created": None, "modified": None, "other": {}}


def xref_table(seed):
    doc = Doc("xref-table", "1.4", seed)
    _two_pages(doc)
    info = _plain_info(doc)
    doc.finish()
    data, w = doc.write("table")
    desc = "Two pages, a classic cross-reference table (one subsection, CRLF entries), PDF 1.4."
    return _sample("xref-table", "xref-table", desc, data, base_expected(doc, data, "xref-table", "xref-table", D1,
                                                                         desc, w, info=info))


def xref_stream(seed):
    doc = Doc("xref-stream", "1.5", seed)
    _two_pages(doc)
    info = _plain_info(doc)
    doc.finish()
    data, w = doc.write("stream")
    desc = "A cross-reference stream (/W, PNG Up predictor 12), no object streams."
    return _sample("xref-stream", "xref-stream", desc, data,
                   base_expected(doc, data, "xref-stream", "xref-stream", D1, desc, w, info=info))


def object_streams(seed):
    doc = Doc("object-streams", "1.5", seed)
    _two_pages(doc)
    info = _plain_info(doc)
    doc.finish()
    data, w = doc.write("objstm", objstm_size=4, extends=True)
    desc = "Object streams (four objects each, the later ones /Extends the first) and a cross-reference stream."
    return _sample("object-streams", "object-streams", desc, data,
                   base_expected(doc, data, "object-streams", "object-streams", D1, desc, w, info=info))


def hybrid(seed):
    doc = Doc("hybrid", "1.5", seed)
    _two_pages(doc)
    info = _plain_info(doc)
    doc.finish()
    data, w = doc.write("hybrid", objstm_size=5)
    desc = ("A hybrid file: a classic table for the plain objects and a trailer /XRefStm pointing to a cross-reference "
            "stream that lists the objects held in object streams (absent from the table).")
    return _sample("hybrid", "hybrid", desc, data, base_expected(doc, data, "hybrid", "hybrid", D1, desc, w, info=info))


def incremental(seed):
    doc = Doc("incremental", "1.4", seed)
    f = helvetica(doc)
    p1 = text_page(doc, f, ["Draft plan", "Ship the beta in June."])
    doc.info = {"Title": B(b"Draft plan"), "Producer": B(b"Idrak document samples")}
    draft = doc.pdf.add({"Status": B(b"draft"), "Reviewers": 2})
    doc.catalog_extra["DraftNote"] = draft
    doc.finish()
    data0, w = doc.write("table")
    base_len = len(w.buf)
    # Update 1: page 1's content replaced (same object number), a second page added.
    content_ref = doc.pdf.objects[p1.ref.num][1]["Contents"]
    data = b"BT\n/F1 12 Tf\n36 252 Td\n(Revised plan) Tj\n0 -18 Td\n(Ship the beta in July.) Tj\nET\n"
    doc.pdf.put(content_ref, Stream({"Filter": N("FlateDecode")}, flate(data), data))
    page2_content = b"BT\n/F1 12 Tf\n36 252 Td\n(Budget) Tj\n0 -18 Td\n(Travel costs fall by a tenth.) Tj\nET\n"
    c2 = doc.pdf.add(Stream({"Filter": N("FlateDecode")}, flate(page2_content), page2_content))
    page2 = doc.pdf.add({"Type": N("Page"), "Parent": doc.pages_ref, "MediaBox": [0, 0, 420, 300],
                         "Resources": {"Font": {"F1": f.ref}}, "Contents": c2})
    doc.pdf.put(doc.pages_ref, {"Type": N("Pages"), "Kids": [p1.ref, page2], "Count": 2})
    w.revision([content_ref.num, c2.num, page2.num, doc.pages_ref.num], "table", doc.trailer(), prev=w.last_xref,
               table_eol=b" \n")
    first_update_end = len(w.buf)
    # Update 2: the title changed, the draft note dropped from the catalog and its object freed.
    doc.pdf.put(doc.info_ref, {"Title": B(b"Approved plan"), "Producer": B(b"Idrak document samples")})
    cat = dict(doc.pdf.get(doc.catalog))
    del cat["DraftNote"]
    doc.pdf.put(doc.catalog, cat)
    del doc.pdf.objects[draft.num]
    w.revision([doc.info_ref.num, doc.catalog.num], "table", doc.trailer(), prev=w.last_xref, free={draft.num: 1})
    data = bytes(w.buf)
    # The expected pages are the newest revision's.
    exp = base_expected(doc, data, "incremental", "incremental", D1, "", w)
    exp["pages"] = [_text_page_json(0, ["Revised plan", "Ship the beta in July."], f),
                    _text_page_json(1, ["Budget", "Travel costs fall by a tenth."], f)]
    exp["open"]["pageCount"] = 2
    exp["info"] = {"title": "Approved plan", "author": None, "subject": None, "keywords": None, "creator": None,
                   "producer": "Idrak document samples", "created": None, "modified": None, "other": {}}
    exp["structure"]["revisionEnds"] = [base_len, first_update_end, len(data)]
    exp["structure"]["freed"] = [[draft.num, 1]]
    desc = ("A base revision and two incremental updates (classic tables, /Prev): the first replaces page 1's content "
            "and adds a page, the second changes the title and frees an object (the newest revision wins).")
    exp["description"] = desc
    return _sample("incremental", "incremental", desc, data, exp)


def _text_page_json(index, lines, font, size=12, left=36, top=252, leading=18):
    """The expected page of a hand-written text page (Helvetica, one Tj a line, as text_page writes them)."""
    from .document import r4

    runs = []
    for i, line in enumerate(lines):
        gl = font.glyphs(line)
        width = sum(g.width for g in gl) / 1000 * size
        base = top - i * leading
        y_top = 300 - (base + font.ascent / 1000 * size)
        runs.append({"order": i, "text": line,
                     "box": [left, r4(y_top), r4(width), r4((font.ascent - font.descent) / 1000 * size)],
                     "origin": [left, r4(300 - base)], "end": [r4(left + width), r4(300 - base)], "fontSize": size,
                     "font": font.base_font, "fontResource": font.resname, "fontKind": font.kind,
                     "ascent": font.ascent, "descent": font.descent, "vertical": False, "hidden": False,
                     "renderMode": 0, "angle": 0.0})
    return {"index": index, "width": 420, "height": 300, "rotation": 0, "mediaBox": [0, 0, 420, 300],
            "cropBox": [0, 0, 420, 300], "text": "\n".join(lines), "lines": [{"text": t, "direction": "ltr"} for t in lines],
            "runs": runs, "images": [], "issues": {"unmappedGlyphs": 0}}


def incremental_stream(seed):
    doc = Doc("incremental-stream", "1.5", seed)
    f = helvetica(doc)
    text_page(doc, f, ["Release notes", "Version 2.1 fixes the export."])
    doc.info = {"Title": B(b"Release notes"), "Producer": B(b"Idrak document samples")}
    doc.finish()
    data0, w = doc.write("objstm", objstm_size=10)
    base_len = len(w.buf)
    # The update replaces the information dictionary, which the base holds in an object stream.
    assert w.entries[doc.info_ref.num][0] == 2
    doc.pdf.put(doc.info_ref, {"Title": B(b"Release notes 2.1"), "Producer": B(b"Idrak document samples")})
    w.revision([doc.info_ref.num], "stream", doc.trailer(), prev=w.last_xref)
    data = bytes(w.buf)
    desc = ("A cross-reference stream with object streams, then an incremental update as a cross-reference stream "
            "(/Prev) replacing an object the base held in an object stream.")
    exp = base_expected(doc, data, "incremental-stream", "incremental", D1, desc, w)
    exp["info"] = {"title": "Release notes 2.1", "author": None, "subject": None, "keywords": None, "creator": None,
                   "producer": "Idrak document samples", "created": None, "modified": None, "other": {}}
    exp["structure"]["revisionEnds"] = [base_len, len(data)]
    return _sample("incremental-stream", "incremental", desc, data, exp)


JUNK = (b"From: reports@example.com\r\nTo: team@example.com\r\nSubject: Monthly figures\r\n"
        b"Content-Type: application/pdf\r\n\r\n" + b"x" * 480 + b"\r\n")


def header_junk(seed, absolute):
    name = "header-junk-absolute" if absolute else "header-junk"
    doc = Doc(name, "1.4", seed)
    _two_pages(doc)
    info = _plain_info(doc)
    doc.finish()
    w = Writer(doc.pdf)
    if absolute:
        w.buf = bytearray(JUNK) + w.buf
    w.revision(sorted(doc.pdf.objects), "table", doc.trailer(), first=True)
    data = bytes(w.buf) if absolute else JUNK + bytes(w.buf)
    if absolute:
        desc = f"{len(JUNK)} bytes before %PDF- (a mail header); the offsets count from the file's start."
    else:
        desc = (f"{len(JUNK)} bytes before %PDF- (a mail header put in front of a finished file): the offsets count "
                "from the header, as readers that find the header shift them.")
    exp = base_expected(doc, data, name, "header-junk", D1, desc, w, info=info)
    exp["structure"]["headerOffset"] = len(JUNK)
    exp["structure"]["offsetsFrom"] = "file" if absolute else "header"
    if absolute:
        # qpdf takes offsets past junk to count from the header and so rebuilds this one; a reader that tries the
        # offset as written first reads it as it is. Either is right: the tests do not check the flag here.
        exp["open"]["repaired"] = None
    return _sample(name, "header-junk", desc, data, exp)


def broken_xref(seed):
    doc = Doc("broken-xref", "1.4", seed)
    _two_pages(doc)
    info = _plain_info(doc)
    doc.finish()
    data, w = doc.write("table")
    insert = b"%edited %%\n"
    head = data.index(b"\n", data.index(b"%\xe2")) + 1
    data = data[:head] + insert + data[head:]
    desc = ("A comment of 11 bytes inserted after the header once the file was written: every offset in the table and "
            "startxref is 11 short, so the reader must rebuild the cross-reference by scanning for objects.")
    exp = base_expected(doc, data, "broken-xref", "repair", D1, desc, w, info=info)
    exp["open"]["repaired"] = True
    exp["structure"]["damage"] = "every cross-reference offset 11 bytes short"
    return _sample("broken-xref", "repair", desc, data, exp)


def missing_xref(seed):
    doc = Doc("missing-xref", "1.4", seed)
    _two_pages(doc)
    info = _plain_info(doc)
    doc.finish()
    w = Writer(doc.pdf)
    for n in sorted(doc.pdf.objects):
        g, o = doc.pdf.objects[n]
        w.write_object(n, g, o)
    t = doc.trailer()
    t["Size"] = doc.pdf.next_num
    w.buf += b"trailer\n" + ser(t) + b"\n%%EOF\n"
    data = bytes(w.buf)
    desc = "No cross-reference table and no startxref, only the objects and a trailer: rebuilt by scanning."
    exp = base_expected(doc, data, "missing-xref", "repair", D1, desc, w, info=info)
    exp["open"]["repaired"] = True
    exp["structure"]["damage"] = "no cross-reference, no startxref"
    return _sample("missing-xref", "repair", desc, data, exp)


def wrong_length(seed):
    doc = Doc("wrong-length", "1.4", seed)
    f = helvetica(doc)
    p1 = text_page(doc, f, ["Length too short", "The stream is longer than its /Length says."])
    p2 = text_page(doc, f, ["Length too long", "The stream ends before its /Length says."])
    p3 = text_page(doc, f, ["Length indirect", "The /Length is an object written after the stream."])
    info = _plain_info(doc)
    doc.finish()
    s1, s2, s3 = (doc.pdf.get(doc.pdf.get(p.ref)["Contents"]) for p in (p1, p2, p3))
    s1.length = len(s1.data) - 10
    s1.note = "/Length 10 bytes short"
    s2.length = len(s2.data) + 25
    s2.note = "/Length 25 bytes past endstream"
    length_ref = doc.pdf.add(len(s3.data))
    s3.length = length_ref
    s3.note = "/Length indirect, written after the stream"
    data, w = doc.write("table")
    desc = ("Three content streams: /Length 10 bytes short, 25 bytes too long, and an indirect /Length written after "
            "the stream; each recovered by its endstream.")
    exp = base_expected(doc, data, "wrong-length", "wrong-length", D1, desc, w, info=info)
    return _sample("wrong-length", "wrong-length", desc, data, exp)


FILTER_PAGES = [
    ("FlateDecode", lambda d: (flate(d), {"Filter": N("FlateDecode")})),
    ("FlateDecode without the zlib header", lambda d: (flate(d, header=False), {"Filter": N("FlateDecode")})),
    ("ASCIIHexDecode", lambda d: (ahx(d), {"Filter": N("ASCIIHexDecode")})),
    ("ASCIIHexDecode with an odd last digit", lambda d: (ahx(d + b" ", lower=True, odd=True),
                                                          {"Filter": N("ASCIIHexDecode")}, d + b" ")),
    ("ASCII85Decode", lambda d: (a85(d + bytes(8)), {"Filter": N("ASCII85Decode")}, d + bytes(8))),
    ("LZWDecode", lambda d: (lzw(d), {"Filter": N("LZWDecode")})),
    ("LZWDecode with EarlyChange 0", lambda d: (lzw(d, early=0), {"Filter": N("LZWDecode"),
                                                                  "DecodeParms": {"EarlyChange": 0}})),
    ("RunLengthDecode", lambda d: (rle(d), {"Filter": N("RunLengthDecode")})),
    ("ASCII85Decode then FlateDecode", lambda d: (a85(flate(d)), {"Filter": [N("ASCII85Decode"), N("FlateDecode")]})),
    ("ASCIIHexDecode then LZWDecode", lambda d: (ahx(lzw(d)), {"Filter": [N("ASCIIHexDecode"), N("LZWDecode")]})),
    ("FlateDecode then RunLengthDecode", lambda d: (flate(rle(d)), {"Filter": [N("FlateDecode"),
                                                                                N("RunLengthDecode")]})),
    ("No filter", lambda d: (d, {})),
]


def filters(seed):
    doc = Doc("filters", "1.4", seed)
    f = helvetica(doc)
    for label, enc in FILTER_PAGES:
        p = text_page(doc, f, [f"Filter: {label}", "Sales rose in the third quarter.",
                               "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa 0000000000000000"])
        p.content_encoder = enc
    info = _plain_info(doc)
    doc.finish()
    data, w = doc.write("table")
    desc = ("One page a filter chain for its content: Flate with and without the zlib header, ASCIIHex (and an odd "
            "last digit), ASCII85 (with z groups), LZW with and without early change, RunLength, three chains, none.")
    return _sample("filters", "filters", desc, data, base_expected(doc, data, "filters", "filters", D1, desc, w,
                                                                   info=info))


PREDICTED = [
    ("grey 8-bit, PNG predictors 0-4 by row (15)", dict(cs="DeviceGray", comps=1, bpc=8, predictor=("png", (0, 1, 2, 3, 4)))),
    ("RGB 8-bit, PNG Up (12)", dict(cs="DeviceRGB", comps=3, bpc=8, predictor=("png", (2,)))),
    ("RGB 16-bit, PNG 0-4 by row (15)", dict(cs="DeviceRGB", comps=3, bpc=16, predictor=("png", (0, 1, 2, 3, 4)))),
    ("grey 1-bit, PNG 0-4 by row (15)", dict(cs="DeviceGray", comps=1, bpc=1, predictor=("png", (4, 3, 2, 1, 0)))),
    ("RGB 8-bit, TIFF 2", dict(cs="DeviceRGB", comps=3, bpc=8, predictor=("tiff", None))),
    ("grey 16-bit, TIFF 2", dict(cs="DeviceGray", comps=1, bpc=16, predictor=("tiff", None))),
    ("grey 4-bit, TIFF 2", dict(cs="DeviceGray", comps=1, bpc=4, predictor=("tiff", None))),
    ("RGB 8-bit, LZW with PNG Paeth (14)", dict(cs="DeviceRGB", comps=3, bpc=8, predictor=("png", (4,)), filters=("LZW",))),
]


def predictors(seed):
    doc = Doc("predictors", "1.5", seed)
    f = helvetica(doc)
    p = doc.page(420, 300)
    c = p.content
    c.BT()
    c.Tf(f, 10)
    c.Td(20, 280)
    c.show("Images stored with predictors")
    c.ET()
    p.line("Images stored with predictors")
    files = {}
    for i, (label, kw) in enumerate(PREDICTED):
        w, h = 23, 17
        maxv = (1 << kw["bpc"]) - 1
        img = Image(doc, w, h, rows=pattern(w, h, kw["comps"], maxv, seed=i), label=label, **kw)
        x = 20 + (i % 4) * 100
        y = 150 - (i // 4) * 110
        rec = c.image(f"Im{i + 1}", img, (80, 0, 0, 60, x, y))
        rec["pixels"]["file"] = f"predictors.im{i + 1}.png"
        files[f"predictors.im{i + 1}.png"] = img.png()
    info = _plain_info(doc)
    doc.finish()
    data, w_ = doc.write("table")
    desc = "Eight image XObjects stored with PNG predictors (each row type, 1, 8 and 16 bits) and the TIFF predictor."
    exp = base_expected(doc, data, "predictors", "predictors", ["D1", "D4"], desc, w_, info=info)
    return _sample("predictors", "predictors", desc, data, exp, ["D1", "D4"], files)


def info_full(seed):
    doc = Doc("info", "1.7", seed)
    _two_pages(doc)
    doc.info = {
        "Title": B(b"Quarterly report \x85 Q3 \x80 draft"),  # PDFDocEncoding: en dash, bullet
        "Author": B(b"Operations team"),
        "Subject": U16("التقرير الفني – Technical report"),
        "Keywords": U16("sales, logistics, \U0001F4C8 growth", hex=True),
        "Creator": Raw(b"(Report builder (version 2) \\\\ \\(beta\\) \\101\\102\\103 multi\\\nline)"),
        "Producer": H(b"Idrak document samples"),
        "CreationDate": B(b"D:20240315143000+02'00'"),
        "ModDate": B(b"D:20240316091500Z"),
        "Trapped": N("False"),
        "Department": B(b"Finance and planning"),
    }
    info = {"title": "Quarterly report – Q3 • draft", "author": "Operations team",
            "subject": "التقرير الفني – Technical report",
            "keywords": "sales, logistics, \U0001F4C8 growth",
            "creator": "Report builder (version 2) \\ (beta) ABC multiline",
            "producer": "Idrak document samples",
            "created": "2024-03-15T14:30:00+02:00", "modified": "2024-03-16T09:15:00+00:00",
            "other": {"Trapped": "False", "Department": "Finance and planning"},
            "raw": {"CreationDate": "D:20240315143000+02'00'", "ModDate": "D:20240316091500Z"}}
    doc.finish()
    data, w = doc.write("table")
    desc = ("Document information: PDFDocEncoding (en dash, bullet), UTF-16BE (Arabic and English; a character outside "
            "the BMP in a hex string), a literal with balanced parentheses, escapes, octal and a line continuation, a "
            "hex string, D: dates with an offset and with Z, a name and a custom key.")
    return _sample("info", "info", desc, data, base_expected(doc, data, "info", "info", D1, desc, w, info=info))


def info_partial(seed):
    doc = Doc("info-partial", "2.0", seed)
    _two_pages(doc)
    doc.info = {
        "Title": B(b"\xef\xbb\xbf" + "Café menu – spring".encode("utf-8")),
        "CreationDate": B(b"D:2024"),
        "ModDate": B(b"D:20240315143000-05'30"),
    }
    info = {"title": "Café menu – spring", "author": None, "subject": None, "keywords": None,
            "creator": None, "producer": None, "created": "2024-01-01T00:00:00", "modified": "2024-03-15T14:30:00-05:30",
            "other": {}, "raw": {"CreationDate": "D:2024", "ModDate": "D:20240315143000-05'30"}}
    doc.finish()
    data, w = doc.write("table")
    desc = ("PDF 2.0 information: a UTF-8 string (its byte order mark), a date with only the year, a date whose offset "
            "lacks the closing apostrophe.")
    return _sample("info-partial", "info", desc, data,
                   base_expected(doc, data, "info-partial", "info", D1, desc, w, version="2.0", info=info))


def page_tree(seed):
    doc = Doc("page-tree", "1.4", seed)
    f = helvetica(doc)
    p1 = text_page(doc, f, ["Inherited box and rotation", "This page takes its size and turn from its parent."],
                   rotate=90, inherit={"MediaBox", "Rotate", "Resources"})
    p2 = text_page(doc, f, ["Its own rotation", "This page sets /Rotate 0 over its parent's 90."],
                   inherit={"MediaBox", "Resources"})
    p2.extra["Rotate"] = 0
    p3 = text_page(doc, f, ["Under the root", "A page beside the intermediate node."], width=300, height=420)
    node = doc.pdf.alloc()
    p1.extra["Parent"] = node
    p2.extra["Parent"] = node

    def tree(d):
        d.pdf.put(node, {"Type": N("Pages"), "Parent": d.pages_ref, "Kids": [p1.ref, p2.ref], "Count": 2,
                         "MediaBox": [0, 0, 420, 300], "Rotate": 90, "Resources": p1.resources})
        d.pdf.put(d.pages_ref, {"Type": N("Pages"), "Kids": [node, p3.ref], "Count": 5})

    doc.page_tree = tree
    info = _plain_info(doc)
    doc.finish()
    data, w = doc.write("table")
    desc = ("A nested page tree: an intermediate node gives two pages their MediaBox, Resources and /Rotate 90 (the "
            "second overrides it with 0); the root's /Count says 5 where its kids hold 3 (the kids win).")
    exp = base_expected(doc, data, "page-tree", "page-tree", ["D1", "D3"], desc, w, info=info)
    exp["structure"]["countWritten"] = 5
    return _sample("page-tree", "page-tree", desc, data, exp, ["D1", "D3"])


# -- hostile ----------------------------------------------------------------------------------------------------------

def _hostile(name, desc, data, limit, options, error="InvalidDataException", trips="open", phases=D1, extra=None):
    hostile = {"limit": limit, "options": options, "error": error, "trips": trips}
    if extra:
        hostile.update(extra)
    exp = {"schema": 1, "name": name, "file": f"{name}.pdf", "kind": "hostile", "phases": phases,
           "description": desc, "size": len(data), "sha256": sha256(data), "hostile": hostile}
    return Sample(name, "d1" if phases == D1 else "d3", "hostile", phases, desc, data, exp, hostile=hostile)


def _zeros_flate(total, chunk=1 << 20):
    c = zlib.compressobj(9)
    out = bytearray()
    block = bytes(chunk)
    for _ in range(total // chunk):
        out += c.compress(block)
    out += c.flush()
    return bytes(out)


def _one_page(doc, content_stream, version="1.4"):
    """A one-page file whose page content is `content_stream` (a Stream)."""
    f = helvetica(doc)
    cref = doc.pdf.add(content_stream)
    page = doc.pdf.alloc()
    doc.pdf.put(page, {"Type": N("Page"), "Parent": doc.pages_ref, "MediaBox": [0, 0, 420, 300],
                       "Resources": {"Font": {"F1": f.ref}}, "Contents": cref})
    f.used.add(32)
    f.finish()
    doc.pdf.put(doc.pages_ref, {"Type": N("Pages"), "Kids": [page], "Count": 1})
    doc.pdf.put(doc.catalog, {"Type": N("Catalog"), "Pages": doc.pages_ref})
    doc.info_ref = None
    return page


GIB = 1 << 30


def flate_bomb(seed, inner):
    doc = Doc("hostile-flate-bomb", "1.4", seed)
    if inner is None:
        inner = _zeros_flate(GIB)
    _one_page(doc, Stream({"Filter": N("FlateDecode")}, inner, b"", note="1 GiB of zero bytes"))
    data, w = doc.write("table")
    return _hostile("hostile-flate-bomb", f"A page content stream of {len(inner)} bytes that inflates to 1 GiB of zero "
                    "bytes.", data, "MaxStreamBytes", {"MaxStreamBytes": 64 << 20}, trips="page 0 read",
                    extra={"decodedBytes": GIB}), inner


def chained_bomb(seed, inner):
    doc = Doc("hostile-chained-bomb", "1.4", seed)
    outer = flate(inner)
    _one_page(doc, Stream({"Filter": [N("FlateDecode"), N("FlateDecode")]}, outer, b"",
                          note="FlateDecode twice: 1 GiB of zero bytes"))
    data, w = doc.write("table")
    return _hostile("hostile-chained-bomb", f"A content stream of {len(outer)} bytes through FlateDecode twice: the "
                    "first stage gives about a megabyte, the second 1 GiB of zero bytes.", data, "MaxStreamBytes",
                    {"MaxStreamBytes": 64 << 20}, trips="page 0 read", extra={"decodedBytes": GIB})


def reference_cycle(seed):
    doc = Doc("hostile-reference-cycle", "1.4", seed)
    a, b = doc.pdf.alloc(), doc.pdf.alloc()
    doc.pdf.put(a, b)
    doc.pdf.put(b, a)
    f = helvetica(doc)
    f.used.add(32)
    f.finish()
    page = doc.pdf.add({"Type": N("Page"), "Parent": doc.pages_ref, "MediaBox": [0, 0, 420, 300],
                        "Resources": {"Font": {"F1": f.ref}}, "Contents": a})
    doc.pdf.put(doc.pages_ref, {"Type": N("Pages"), "Kids": [page], "Count": 1})
    doc.pdf.put(doc.catalog, {"Type": N("Catalog"), "Pages": doc.pages_ref})
    doc.info_ref = None
    data, w = doc.write("table")
    return _hostile("hostile-reference-cycle", f"The page's /Contents is object {a.num}, whose value is {b.num} 0 R, "
                    f"whose value is {a.num} 0 R.", data, "cycle", {}, trips="page 0 read")


def page_tree_cycle(seed):
    doc = Doc("hostile-page-tree-cycle", "1.4", seed)
    f = helvetica(doc)
    f.used.add(32)
    f.finish()
    node = doc.pdf.alloc()
    page = doc.pdf.add({"Type": N("Page"), "Parent": node, "MediaBox": [0, 0, 420, 300],
                        "Resources": {"Font": {"F1": f.ref}}})
    doc.pdf.put(node, {"Type": N("Pages"), "Parent": doc.pages_ref, "Kids": [page, doc.pages_ref], "Count": 2})
    doc.pdf.put(doc.pages_ref, {"Type": N("Pages"), "Kids": [node], "Count": 2})
    doc.pdf.put(doc.catalog, {"Type": N("Catalog"), "Pages": doc.pages_ref})
    doc.info_ref = None
    data, w = doc.write("table")
    return _hostile("hostile-page-tree-cycle", "An intermediate page-tree node lists the root among its kids.", data,
                    "cycle", {}, trips="open (page count)")


def prev_loop(seed):
    doc = Doc("hostile-prev-loop", "1.4", seed)
    _two_pages(doc)
    doc.finish()
    w = Writer(doc.pdf)
    w.revision(sorted(doc.pdf.objects), "table", doc.trailer(), first=True)
    # Two sections whose /Prev point at each other (the first padded so the second's offset is known before).
    def section(prev):
        return b"xref\n0 1\n0000000000 65535 f\r\ntrailer\n" + ser({**doc.trailer(), "Size": doc.pdf.next_num,
                                                                       "Prev": prev}) + b"\n"

    sec_a = len(w.buf)
    width = len(section(10 ** 9))
    sec_b = sec_a + width
    a = section(sec_b)
    w.buf += a + b" " * (width - len(a)) + section(sec_a) + b"startxref\n%d\n%%%%EOF\n" % sec_b
    data = bytes(w.buf)
    return _hostile("hostile-prev-loop", "Two cross-reference sections whose /Prev entries point at each other.", data,
                    "cycle", {}, trips="open")


def deep_nesting(seed, kind):
    doc = Doc(f"hostile-deep-{kind}", "1.4", seed)
    _two_pages(doc)
    depth = 100000
    if kind == "arrays":
        doc.catalog_extra["Nesting"] = Raw(b"[" * depth + b"1" + b"]" * depth)
    else:
        doc.catalog_extra["Nesting"] = Raw(b"<</A " * depth + b"1" + b">>" * depth)
    doc.finish()
    data, w = doc.write("table")
    what = "arrays" if kind == "arrays" else "dictionaries"
    return _hostile(f"hostile-deep-{kind}", f"The catalog holds {depth} nested {what}.", data, "MaxDepth",
                    {"MaxDepth": 64}, trips="open (the catalog)", extra={"depth": depth})


def huge_size(seed):
    doc = Doc("hostile-huge-size", "1.5", seed)
    _two_pages(doc)
    doc.finish()
    data, w = doc.write("stream")
    # The cross-reference stream claims two billion objects (its /Size and /Index) but holds only the real rows.
    count = 2000000000
    data = data.replace(b"/Size %d" % doc.pdf.next_num, b"/Size %d" % count, 1)
    return _hostile("hostile-huge-size", "A cross-reference stream whose /Size claims two billion objects.", data,
                    "MaxObjects", {"MaxObjects": 1000000}, trips="open", extra={"claimedObjects": count})


def objstm_self(seed):
    doc = Doc("hostile-objstm-self", "1.5", seed)
    _two_pages(doc)
    doc.finish()
    w = Writer(doc.pdf)
    nums = sorted(doc.pdf.objects)
    w.revision(nums, "objstm", doc.trailer(), objstm_size=50, first=True)
    data = bytes(w.buf)
    stm = w.revisions[0]["objectStreams"][0]
    return _hostile("hostile-objstm-self", f"Object {stm}, the object stream, is listed in the cross-reference stream as "
                    "object 0 of itself.", _patch_objstm_self(data, w, stm), "cycle", {}, trips="open")


def _patch_objstm_self(data, w, stm):
    """Rewrites the cross-reference stream so its row for the object stream says: in object stream `stm`, index 0."""
    from .filters import flate as _fl, png_predict

    start = data.rindex(b"startxref")
    off = int(data[start:].split()[1])
    head_end = data.index(b"stream\n", off) + 7
    end = data.index(b"\nendstream", head_end)
    d = data[off:head_end]
    widths = [int(x) for x in d[d.index(b"/W [") + 4:d.index(b"]", d.index(b"/W ["))].split()]
    idx = d[d.index(b"/Index [") + 8:d.index(b"]", d.index(b"/Index ["))].split() if b"/Index" in d else None
    rows = zlib.decompress(data[head_end:end])
    cols = sum(widths)
    # undo the Up predictor
    plain = bytearray()
    prev = bytes(cols)
    for r in range(len(rows) // (cols + 1)):
        line = rows[r * (cols + 1) + 1:(r + 1) * (cols + 1)]
        cur = bytes((line[i] + prev[i]) & 0xFF for i in range(cols))
        plain += cur
        prev = cur
    nums = []
    if idx:
        vals = [int(x) for x in idx]
        for k in range(0, len(vals), 2):
            nums += list(range(vals[k], vals[k] + vals[k + 1]))
    else:
        nums = list(range(len(plain) // cols))
    row = nums.index(stm)
    new = bytes([2]) + stm.to_bytes(widths[1], "big") + (0).to_bytes(widths[2], "big")
    plain[row * cols:(row + 1) * cols] = new
    encoded = _fl(png_predict(bytes(plain), 1, 8, cols, pattern=(2,)))
    d2 = d.replace(b"/Length %d" % (end - head_end), b"/Length %d" % len(encoded))
    return data[:off] + d2 + encoded + data[end:]


def form_recursion(seed):
    doc = Doc("hostile-form-recursion", "1.4", seed)
    f = helvetica(doc)
    form = doc.pdf.alloc()
    content = b"BT /F1 12 Tf 10 10 Td (Loop) Tj ET\n/Fm1 Do\n"
    doc.pdf.put(form, Stream({"Type": N("XObject"), "Subtype": N("Form"), "BBox": [0, 0, 420, 300],
                              "Resources": {"Font": {"F1": f.ref}, "XObject": {"Fm1": form}},
                              "Filter": N("FlateDecode")}, flate(content), content))
    page_content = b"/Fm1 Do\n"
    cref = doc.pdf.add(Stream({"Filter": N("FlateDecode")}, flate(page_content), page_content))
    f.used.update(ord(ch) for ch in "Loop")
    f.finish()
    page = doc.pdf.add({"Type": N("Page"), "Parent": doc.pages_ref, "MediaBox": [0, 0, 420, 300],
                        "Resources": {"Font": {"F1": f.ref}, "XObject": {"Fm1": form}}, "Contents": cref})
    doc.pdf.put(doc.pages_ref, {"Type": N("Pages"), "Kids": [page], "Count": 1})
    doc.pdf.put(doc.catalog, {"Type": N("Catalog"), "Pages": doc.pages_ref})
    doc.info_ref = None
    data, w = doc.write("table")
    return _hostile("hostile-form-recursion", "A form XObject that draws itself (its resources name it).", data,
                    "MaxDepth", {"MaxDepth": 64}, trips="page 0 read (text)", phases=["D3"])


def build(seed):
    out = [xref_table(seed), xref_stream(seed), object_streams(seed), hybrid(seed), incremental(seed),
           incremental_stream(seed), header_junk(seed, False), header_junk(seed, True), broken_xref(seed),
           missing_xref(seed), wrong_length(seed), filters(seed), predictors(seed), info_full(seed),
           info_partial(seed), page_tree(seed)]
    bomb, inner = flate_bomb(seed, None)
    out += [bomb, chained_bomb(seed, inner), reference_cycle(seed), page_tree_cycle(seed), prev_loop(seed),
            deep_nesting(seed, "arrays"), deep_nesting(seed, "dicts"), huge_size(seed), objstm_self(seed),
            form_recursion(seed)]
    return out
