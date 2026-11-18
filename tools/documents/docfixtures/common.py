"""What every sample shares: its record (name, phase folder, bytes, expected JSON, files beside it), a page of text
lines, and the canonical JSON form of objects that a reader's parsed objects are compared with."""
import os

from .document import r4
from .fonts import Standard14
from .pdf import Name, Raw, Ref, Stream, String, sha256

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FONTS = os.path.join(ROOT, "tests", "Idrak.Tests", "data", "fonts")
DEJAVU = os.path.join(FONTS, "dejavu-sans-subset.ttf")
AMIRI = os.path.join(FONTS, "arabic", "amiri.ttf")

SCHEMA = 1


class Sample:
    def __init__(self, name, folder, kind, phases, description, data, expected, files=None, hostile=None):
        self.name = name
        self.folder = folder
        self.kind = kind
        self.phases = phases
        self.description = description
        self.data = data
        self.expected = expected
        self.files = files or {}  # name beside the sample -> bytes (expected pixels, attachments' bytes)
        self.hostile = hostile

    @property
    def file(self):
        return f"{self.name}.pdf"


def canonical(v):
    """An object as JSON: names "/Name", strings {"string": hex}, references {"ref": [n, g]}, dictionaries
    {"dict": {...}}, arrays lists, numbers and booleans as they are."""
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return r4(v)
    if isinstance(v, Name):
        return "/" + v.value
    if isinstance(v, String):
        return {"string": v.data.hex()}
    if isinstance(v, Ref):
        return {"ref": [v.num, v.gen]}
    if isinstance(v, Raw):
        return {"raw": v.data.decode("latin-1")}
    if isinstance(v, (list, tuple)):
        return [canonical(x) for x in v]
    if isinstance(v, dict):
        return {"dict": {k: canonical(x) for k, x in v.items()}}
    raise TypeError(type(v).__name__)


def objects_json(pdf, writer, skip=()):
    """Every live object as the newest revision has it (streams by their dictionary and decoded bytes)."""
    lengths = {}
    for rec in writer.streams:
        lengths[rec["object"][0]] = rec
    out = []
    for num in sorted(pdf.objects):
        if num in skip:
            continue
        gen, obj = pdf.objects[num]
        if isinstance(obj, Stream):
            rec = lengths.get(num)
            d = dict(obj.dict)
            d["Length"] = obj.length if obj.length is not None else (rec["storedLength"] if rec else len(obj.data))
            value = {"stream": canonical(d)["dict"], "decodedLength": len(obj.decoded),
                     "decodedSha256": sha256(obj.decoded)}
        else:
            value = canonical(obj)
        out.append({"num": num, "gen": gen, "value": value})
    return out


def newest_streams(writer):
    """Each stream object as the newest revision has it (an update that replaces a stream supersedes the old one)."""
    newest = {}
    for rec in writer.streams:
        newest[rec["object"][0]] = rec
    return [newest[n] for n in sorted(newest)]


def helvetica(doc, name="Helvetica"):
    return doc.font(Standard14, name)


def text_page(doc, font, lines, size=12, left=36, top=None, leading=18, width=420, height=300, **page_kw):
    """A page with `lines` from the top left, one Tj each (Td between them)."""
    p = doc.page(width, height, **page_kw)
    c = p.content
    c.BT()
    c.Tf(font, size)
    c.Td(left, (top if top is not None else height - 48))
    for i, line in enumerate(lines):
        if i:
            c.Td(0, -leading)
        c.show(line)
        p.line(line)
    c.ET()
    return p


def base_expected(doc, data, name, kind, phases, description, writer=None, version=None, objects=True, **extra):
    exp = {
        "schema": SCHEMA, "name": name, "file": f"{name}.pdf", "kind": kind, "phases": phases,
        "description": description, "size": len(data), "sha256": sha256(data), "hostile": None,
        "open": {"password": None, "encrypted": False, "repaired": False, "version": version or doc.pdf.version,
                 "pageCount": len(doc.pages)},
        "info": None,
        "structure": {"headerOffset": 0, "offsetsFrom": "file", "damage": None,
                      "revisions": writer.revisions if writer else None},
        "streams": newest_streams(writer) if writer else [],
        "objects": objects_json(doc.pdf, writer) if (objects and writer) else None,
        "pages": doc.expected_pages(),
        "attachments": [],
    }
    exp.update(extra)
    return exp
