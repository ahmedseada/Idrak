"""D11, editing and saving: files with several incremental updates (classic tables, and cross-reference streams over
object streams) with each revision's end and what it changed; two documents to merge whose page resources are
inherited from their page trees under the same names (/F1, /Im1, /GS1, different objects), and a stamp to lay over
pages with the same names again; a file to split; a file encrypted with AES-256 (revision 6) with a user and an owner
password and permissions; and page operations (insert, remove, move, rotate) as inputs with the page order and
rotations they must give. The reference check performs every merge, split and operation with qpdf (pikepdf) and reads
the result's text with pdfminer.six."""
import io

from .common import Sample, base_expected, helvetica
from .document import Doc
from .fonts import Standard14, TrueTypeSimple
from .images import Image, pattern
from .pdf import B, N, Stream, String
from .filters import flate
from .common import DEJAVU
from .security import StandardSecurity

D11 = ["D11"]


def _page_text(doc, p, f, lines, size=14, x=40, top=None):
    c = p.content
    c.BT()
    c.Tf(f, size)
    c.Td(x, top if top is not None else p.size()[1] - 50)
    for k, line in enumerate(lines):
        if k:
            c.Td(0, -size * 1.5)
        c.show(line)
        p.line(line)
    c.ET()


def _revisions_json(w, ends, changes):
    return [{**rev, "end": end, "changes": ch} for rev, end, ch in zip(w.revisions, ends, changes)]


# -- incremental updates ---------------------------------------------------------------------------------------------------

def incremental_table(seed):
    doc = Doc("edit-incremental-table", "1.7", seed)
    f = helvetica(doc)
    p1 = doc.page(420, 300)
    p2 = doc.page(420, 300)
    _page_text(doc, p1, f, ["Project plan", "Phase one starts in April."])
    _page_text(doc, p2, f, ["Milestones", "Beta in June, release in September."])
    pdf = doc.pdf
    field = pdf.alloc()
    pdf.put(field, {"FT": N("Tx"), "T": String(b"status"), "Type": N("Annot"), "Subtype": N("Widget"),
                    "Rect": [250, 40, 400, 60], "F": 4, "P": p1.ref, "DA": String(b"/Helv 10 Tf 0 g")})
    p1.annotations.append(field)
    doc.catalog_extra["AcroForm"] = pdf.add({"Fields": [field], "DA": String(b"/Helv 0 Tf 0 g"),
                                             "DR": {"Font": {"Helv": f.ref}}})
    doc.info = {"Title": B(b"Project plan"), "Producer": B(b"Idrak document samples")}
    doc.finish()
    data0, w = doc.write("table")
    ends = [len(w.buf)]
    changes = [["the base: two pages, an empty text field 'status'"]]
    # Update 1: the field filled (value and appearance), a note added on page 1.
    ap_data = b"/Tx BMC q BT /Helv 10 Tf 0 g 2 6 Td (Approved) Tj ET Q EMC\n"
    ap = pdf.add(Stream({"Type": N("XObject"), "Subtype": N("Form"), "BBox": [0, 0, 150, 20],
                         "Resources": {"Font": {"Helv": f.ref}}, "Filter": N("FlateDecode")}, flate(ap_data), ap_data))
    widget = dict(pdf.get(field))
    widget["V"] = String(b"Approved")
    widget["AP"] = {"N": ap}
    pdf.put(field, widget)
    note = pdf.add({"Type": N("Annot"), "Subtype": N("Text"), "Rect": [380, 260, 400, 280], "F": 4, "P": p1.ref,
                    "Contents": String(b"Approved at the planning meeting"), "NM": String(b"note-1")})
    page = dict(pdf.get(p1.ref))
    page["Annots"] = [field, note]
    pdf.put(p1.ref, page)
    w.revision([ap.num, field.num, note.num, p1.ref.num], "table", doc.trailer(), prev=w.last_xref)
    ends.append(len(w.buf))
    changes.append(["field 'status' set to 'Approved' with an appearance", "a note added on page 1"])
    # Update 2: a third page added, the title changed.
    p3data = b"BT\n/F1 14 Tf\n40 250 Td\n(Budget) Tj\n0 -21 Td\n(Spending stays within the plan.) Tj\nET\n"
    c3 = pdf.add(Stream({"Filter": N("FlateDecode")}, flate(p3data), p3data))
    p3 = pdf.add({"Type": N("Page"), "Parent": doc.pages_ref, "MediaBox": [0, 0, 420, 300],
                  "Resources": {"Font": {"F1": f.ref}}, "Contents": c3})
    pdf.put(doc.pages_ref, {"Type": N("Pages"), "Kids": [p1.ref, p2.ref, p3], "Count": 3})
    pdf.put(doc.info_ref, {"Title": B(b"Project plan, approved"), "Producer": B(b"Idrak document samples")})
    w.revision([c3.num, p3.num, doc.pages_ref.num, doc.info_ref.num], "table", doc.trailer(), prev=w.last_xref)
    ends.append(len(w.buf))
    changes.append(["page 3 added", "the title changed to 'Project plan, approved'"])
    # Update 3: page 1 turned by 90 degrees, the note removed (its object freed).
    page = dict(pdf.get(p1.ref))
    page["Rotate"] = 90
    page["Annots"] = [field]
    pdf.put(p1.ref, page)
    del pdf.objects[note.num]
    w.revision([p1.ref.num], "table", doc.trailer(), prev=w.last_xref, free={note.num: 1})
    ends.append(len(w.buf))
    changes.append(["page 1 turned by /Rotate 90", "the note removed and its object freed"])
    data = bytes(w.buf)
    desc = ("A base revision and three incremental updates with classic tables: a form field filled and a note added; "
            "a page added and the title changed; page 1 turned and the note removed (freed). Each revision's end is "
            "given, so an editor's update can be checked to keep the source's bytes as its prefix.")
    exp = base_expected(doc, data, "edit-incremental-table", "incremental", D11, desc, w, objects=False)
    exp["structure"]["revisionEnds"] = ends
    exp["revisions"] = _revisions_json(w, ends, changes)
    exp["open"]["pageCount"] = 3
    exp["final"] = {"pageCount": 3, "rotations": [90, 0, 0], "title": "Project plan, approved",
                    "fields": {"status": "Approved"}, "annotations": {"0": ["Widget"], "1": [], "2": []},
                    "pageText": ["Project plan\nPhase one starts in April.", "Milestones\nBeta in June, release in "
                                 "September.", "Budget\nSpending stays within the plan."]}
    exp["pages"] = []  # the newest revision's pages are given by `final`
    return Sample("edit-incremental-table", "d11", "incremental", D11, desc, data, exp, checks=[check_final])


def incremental_stream(seed):
    doc = Doc("edit-incremental-stream", "1.7", seed)
    f = helvetica(doc)
    p1 = doc.page(420, 300)
    p2 = doc.page(420, 300)
    _page_text(doc, p1, f, ["Release notes", "Version 3.0 adds export to CSV."])
    _page_text(doc, p2, f, ["Known issues", "Large imports are slow."])
    doc.info = {"Title": B(b"Release notes"), "Producer": B(b"Idrak document samples")}
    doc.finish()
    data0, w = doc.write("objstm", objstm_size=20)
    pdf = doc.pdf
    ends = [len(w.buf)]
    changes = [["the base: object streams and a cross-reference stream"]]
    content2 = pdf.get(p2.ref)["Contents"]
    text = b"BT\n/F1 14 Tf\n40 250 Td\n(Known issues) Tj\n0 -21 Td\n(Large imports are fast since 3.0.1.) Tj\nET\n"
    pdf.put(content2, Stream({"Filter": N("FlateDecode")}, flate(text), text))
    w.revision([content2.num], "stream", doc.trailer(), prev=w.last_xref)
    ends.append(len(w.buf))
    changes.append(["page 2's content replaced"])
    outlines = pdf.alloc()
    item = pdf.add({"Title": String(b"Known issues"), "Parent": outlines, "Dest": [p2.ref, N("Fit")]})
    pdf.put(outlines, {"Type": N("Outlines"), "First": item, "Last": item, "Count": 1})
    cat = dict(pdf.get(doc.catalog))
    cat["Outlines"] = outlines
    pdf.put(doc.catalog, cat)
    w.revision([outlines.num, item.num, doc.catalog.num], "objstm", doc.trailer(), prev=w.last_xref)
    ends.append(len(w.buf))
    changes.append(["a bookmark added (in a new object stream)"])
    data = bytes(w.buf)
    desc = ("A base written with object streams and a cross-reference stream, then two incremental updates as "
            "cross-reference streams: page 2's content replaced, and a bookmark added in a new object stream (the "
            "catalog, which the base holds compressed, replaced).")
    exp = base_expected(doc, data, "edit-incremental-stream", "incremental", D11, desc, w, objects=False)
    exp["structure"]["revisionEnds"] = ends
    exp["revisions"] = _revisions_json(w, ends, changes)
    exp["final"] = {"pageCount": 2, "rotations": [0, 0], "title": "Release notes", "fields": {},
                    "outline": ["Known issues"],
                    "pageText": ["Release notes\nVersion 3.0 adds export to CSV.",
                                 "Known issues\nLarge imports are fast since 3.0.1."]}
    exp["pages"] = []
    return Sample("edit-incremental-stream", "d11", "incremental", D11, desc, data, exp, checks=[check_final])


# -- merge, stamp, split --------------------------------------------------------------------------------------------------

def _inherited_doc(name, seed, font_kind, lines, image_seed, alpha):
    """Pages whose resources come from the page tree (/F1, /Im1, /GS1 on the Pages node)."""
    doc = Doc(name, "1.7", seed)
    f = doc.font(TrueTypeSimple, DEJAVU) if font_kind == "truetype" else helvetica(doc)
    img = Image(doc, 16, 12, 8, "DeviceRGB", 3, pattern(16, 12, 3, 255, image_seed), label=f"{name}'s image")
    gs = doc.pdf.add({"Type": N("ExtGState"), "ca": alpha})
    for k, text in enumerate(lines):
        p = doc.page(420, 300, inherit={"Resources": True})
        _page_text(doc, p, f, [text, f"Page {k + 1} of {name}"])
        c = p.content
        c.q()
        c.emit("/GS1 gs")
        c.cm(80, 0, 0, 60, 300, 40)
        c.emit("/Im1 Do")
        c.Q()
    resources = {"Font": {"F1": f.ref}, "XObject": {"Im1": img.ref}, "ExtGState": {"GS1": gs}}

    def tree(d):
        d.pdf.put(d.pages_ref, {"Type": N("Pages"), "Kids": [p.ref for p in d.pages], "Count": len(d.pages),
                                "Resources": resources})

    doc.page_tree = tree
    doc.finish()
    return doc, f, img, gs


def merge_inputs(seed):
    out = []
    specs = [("merge-a", "standard14", ["Sales report", "Regional figures"], 1, 1.0),
             ("merge-b", "truetype", ["Support report", "Response times"], 7, 0.5)]
    for name, kind, lines, img_seed, alpha in specs:
        doc, f, img, gs = _inherited_doc(name, seed, kind, lines, img_seed, alpha)
        data, w = doc.write("table")
        desc = (f"Two pages whose resources are inherited from the page tree: /F1 ({'Helvetica' if kind == 'standard14' else 'embedded DejaVu Sans, TrueType'}), "
                f"/Im1 (an image), /GS1 (ca {alpha}); merge-a and merge-b use the same names for different objects.")
        exp = base_expected(doc, data, name, "merge", D11, desc, w, objects=False)
        exp["resources"] = {"inheritedFromPageTree": True, "F1": f.base_font, "Im1": img.describe()["pixels"]["sha256"],
                            "GS1": {"ca": alpha}}
        out.append(Sample(name, "d11", "merge", D11, desc, data, exp))
    # A one-page stamp using the same names again, to lay over other pages.
    doc = Doc("stamp", "1.7", seed)
    f = doc.font(Standard14, "Helvetica-Bold")
    gs = doc.pdf.add({"Type": N("ExtGState"), "ca": 0.3})
    p = doc.page(420, 300)
    c = p.content
    c.q()
    c.emit("/GS1 gs")
    c.rg(0.8, 0.1, 0.1)
    c.BT()
    c.Tf(f, 60)
    c.Tm(0.866, 0.5, -0.5, 0.866, 110, 70)
    c.show("DRAFT")
    c.ET()
    c.Q()
    p.line("DRAFT")
    p.resource("ExtGState", "GS1", gs)
    doc.finish()
    data, w = doc.write("table")
    desc = ("A stamp to lay over pages (turned 'DRAFT' at ca 0.3) whose resources are named /F1 and /GS1 again: laid "
            "over merge-a's pages, its names must not take the page's objects, nor the page's its.")
    exp = base_expected(doc, data, "stamp", "merge", D11, desc, w, objects=False)
    out.append(Sample("stamp", "d11", "merge", D11, desc, data, exp))
    return out


MERGE = {"inputs": ["merge-a", "merge-b"],
         "result": [{"from": "merge-a", "page": 0, "font": "Helvetica", "text": "Sales report\nPage 1 of merge-a"},
                    {"from": "merge-a", "page": 1, "font": "Helvetica", "text": "Regional figures\nPage 2 of merge-a"},
                    {"from": "merge-b", "page": 0, "font": "DejaVuSans", "text": "Support report\nPage 1 of merge-b"},
                    {"from": "merge-b", "page": 1, "font": "DejaVuSans", "text": "Response times\nPage 2 of merge-b"}],
         "clashingNames": ["F1", "Im1", "GS1"],
         "rule": "each page keeps the objects its own names meant: a merged page's /F1, /Im1 and /GS1 are its source's"}
STAMP = {"stamp": "stamp", "onto": "merge-a", "pages": [0, 1],
         "result": [{"text": ["Sales report", "Page 1 of merge-a", "DRAFT"]},
                    {"text": ["Regional figures", "Page 2 of merge-a", "DRAFT"]}],
         "rule": "the stamp's resources are renamed where they clash; the page's text and image stay as they were"}


def split_source(seed):
    doc = Doc("split-source", "1.7", seed)
    f = helvetica(doc)
    titles = ["Cover", "Summary", "Sales", "Costs", "Staff", "Outlook"]
    for k, t in enumerate(titles):
        p = doc.page(420, 300, rotate=90 if k == 3 else 0)
        _page_text(doc, p, f, [t, f"Section {k + 1}"])
    doc.finish()
    data, w = doc.write("table")
    desc = ("Six pages to split (page 4 turned by /Rotate 90): `split` gives the parts asked and each part's pages, "
            "text and rotations.")
    exp = base_expected(doc, data, "split-source", "split", D11, desc, w, objects=False)
    parts = [[0, 1], [2, 3, 4], [5]]
    exp["split"] = {"parts": [{"pages": pp, "text": [f"{titles[k]}\nSection {k + 1}" for k in pp],
                               "rotations": [90 if k == 3 else 0 for k in pp]} for pp in parts]}
    return Sample("split-source", "d11", "split", D11, desc, data, exp, checks=[check_split])


# -- encryption -----------------------------------------------------------------------------------------------------------

PERMISSIONS = {"print": 3, "modify": 4, "copy": 5, "annotate": 6, "fillForms": 9, "extractForAccessibility": 10,
               "assemble": 11, "printHighQuality": 12}


def permissions_value(allowed):
    p = 0xFFFFF0C0  # reserved bits set, no permission
    for name in allowed:
        p |= 1 << (PERMISSIONS[name] - 1)
    return p - (1 << 32) if p & 0x80000000 else p


def encrypted(seed):
    doc = Doc("encrypted-aes256", "2.0", seed)
    f = helvetica(doc)
    for lines in (["Salary review", "Shared with the board only."], ["Next steps", "Decisions in the May meeting."]):
        p = doc.page(420, 300)
        _page_text(doc, p, f, lines)
    doc.info = {"Title": B(b"Salary review"), "Producer": B(b"Idrak document samples")}
    doc.finish()
    allowed = ["print", "annotate", "fillForms", "extractForAccessibility", "printHighQuality"]
    perms = permissions_value(allowed)
    sec = StandardSecurity(6, "reader-2026", "editor-2026", doc.file_id, permissions=perms,
                           seed=f"{seed}|encrypted-aes256")
    data, w = doc.write("objstm", security=sec)
    desc = ("AES-256 (revision 6) with a user password 'reader-2026' and an owner password 'editor-2026', "
            "permissions: print, annotate, fill forms, extract for accessibility, print at high quality; not modify, "
            "copy or assemble. Object streams and a cross-reference stream.")
    exp = base_expected(doc, data, "encrypted-aes256", "encrypted", D11, desc, w, objects=False)
    exp["open"].update({"password": "reader-2026", "encrypted": True})
    exp["encryption"] = {**sec.describe(), "wrongPassword": "not-the-password", "encryptDictionary": doc.encrypt_ref.num,
                         "permissionsAllowed": allowed,
                         "permissionsDenied": sorted(set(PERMISSIONS) - set(allowed)),
                         "openWith": {"user": "reader-2026", "owner": "editor-2026", "none": "fails"}}
    exp["info"] = {"title": "Salary review", "author": None, "subject": None, "keywords": None, "creator": None,
                   "producer": "Idrak document samples", "created": None, "modified": None, "other": {}}
    return Sample("encrypted-aes256", "d11", "encrypted", D11, desc, data, exp, checks=[check_encrypted])


# -- page operations ------------------------------------------------------------------------------------------------------

LETTERS = "ABCDE"
ROT = [0, 90, 0, 180, 0]


def page_operations(seed):
    doc = Doc("page-operations", "1.7", seed)
    f = helvetica(doc)
    for k, letter in enumerate(LETTERS):
        p = doc.page(420, 300, rotate=ROT[k])
        _page_text(doc, p, f, [f"Page {letter}"], size=40, x=120, top=140)
    doc.finish()
    data, w = doc.write("table")
    src = [{"source": f"page-operations:{k}", "text": f"Page {LETTERS[k]}", "rotation": ROT[k]} for k in range(5)]
    b0 = {"source": "merge-b:0", "text": "Support report\nPage 1 of merge-b", "rotation": 0}
    a1 = {"source": "merge-a:1", "text": "Regional figures\nPage 2 of merge-a", "rotation": 0}

    def rot(e, by):
        return {**e, "rotation": (e["rotation"] + by) % 360}

    cases = [
        {"name": "insert", "steps": [{"op": "insert", "at": 2, "from": "merge-b", "pages": [0]}],
         "result": [src[0], src[1], b0, src[2], src[3], src[4]]},
        {"name": "remove", "steps": [{"op": "remove", "pages": [1, 3]}], "result": [src[0], src[2], src[4]]},
        {"name": "move", "steps": [{"op": "move", "from": 4, "to": 0}],
         "result": [src[4], src[0], src[1], src[2], src[3]]},
        {"name": "rotate", "steps": [{"op": "rotate", "page": 0, "by": 90}, {"op": "rotate", "page": 3, "by": -90}],
         "result": [rot(src[0], 90), src[1], src[2], rot(src[3], -90), src[4]]},
        {"name": "combined", "steps": [{"op": "remove", "pages": [0]}, {"op": "move", "from": 3, "to": 0},
                                       {"op": "rotate", "page": 1, "by": 180},
                                       {"op": "insert", "at": 4, "from": "merge-a", "pages": [1]}],
         "result": [src[4], rot(src[1], 180), src[2], src[3], a1]},
    ]
    desc = ("Five pages 'Page A' to 'Page E' (B turned 90, D turned 180), with page operations stated as steps "
            "(insert from another sample, remove, move, rotate by a multiple of 90, applied in order; indices into "
            "the document as it stands before each step) and the pages and rotations they must give.")
    exp = base_expected(doc, data, "page-operations", "page-operations", D11, desc, w, objects=False)
    exp["operations"] = cases
    return Sample("page-operations", "d11", "page-operations", D11, desc, data, exp, checks=[check_operations])


# -- reference checks (qpdf through pikepdf; pdfminer.six reads the results) ---------------------------------------------------

def _texts(data, password=""):
    """pdfminer.six's text of each page (extract_text, which keeps turned text a layout walk of text boxes drops),
    whitespace collapsed."""
    from pdfminer.high_level import extract_text

    from .reference import _quiet_logs

    _quiet_logs()
    pages = extract_text(io.BytesIO(data), password=password).split("\x0c")
    return [" ".join(t.split()) for t in pages[:-1]]


def _norm(t):
    return " ".join(t.split())


def _key(t):
    """A page's letters in order of their code points: pdfminer.six splits and reorders the words of a turned page,
    so pages are told apart by the letters they hold."""
    return "".join(sorted(t.replace(" ", "").replace("\n", "")))


def check_final(path, exp, notes):
    import pikepdf

    with pikepdf.open(path) as pdf:
        rotations = [int(p.obj.get("/Rotate", 0)) for p in pdf.pages]
        title = str(pdf.docinfo.get("/Title")) if pdf.docinfo is not None and "/Title" in pdf.docinfo else None
    with open(path, "rb") as fh:
        data = fh.read()
    texts = _texts(data)
    want = exp["final"]
    problems = []
    if rotations != want["rotations"]:
        problems.append({"rotations": [want["rotations"], rotations]})
    if title != want["title"]:
        problems.append({"title": [want["title"], title]})
    if [_key(t) for t in want["pageText"]] != [_key(t) for t in texts]:
        problems.append({"text": [want["pageText"], texts]})
    ends = exp["structure"]["revisionEnds"]
    prefix_ok = all(data[:e].rstrip().endswith(b"%%EOF") for e in ends)
    if not prefix_ok:
        problems.append({"revisionEnds": "a revision end does not follow %%EOF"})
    if problems:
        notes.append(f"{exp['name']}: qpdf and pdfminer.six read the newest revision differently: {problems}")
    return "final", {"rotations": rotations, "title": title, "pdfminerText": texts, "revisionEndsAfterEof": prefix_ok,
                     "problems": problems}, {}


def _load_sample(folder, name):
    import os

    with open(os.path.join(folder, f"{name}.pdf"), "rb") as fh:
        return fh.read()


def check_split(path, exp, notes):
    import os

    import pikepdf

    problems = []
    results = []
    for part in exp["split"]["parts"]:
        with pikepdf.open(path) as src:
            out = pikepdf.new()
            for k in part["pages"]:
                out.pages.append(src.pages[k])
            buf = io.BytesIO()
            out.save(buf, deterministic_id=True)
        data = buf.getvalue()
        texts = _texts(data)
        with pikepdf.open(io.BytesIO(data)) as res:
            rotations = [int(p.obj.get("/Rotate", 0)) for p in res.pages]
        ok = [_key(t) for t in texts] == [_key(t) for t in part["text"]] and rotations == part["rotations"]
        if not ok:
            problems.append({"pages": part["pages"], "text": texts, "rotations": rotations})
        results.append({"pages": part["pages"], "matches": ok})
    if problems:
        notes.append(f"{exp['name']}: qpdf's split differs: {problems}")
    del os
    return "split", {"tool": "qpdf (pikepdf) pages appended to a new file", "parts": results, "problems": problems}, {}


def check_encrypted(path, exp, notes):
    import pikepdf

    out = {}
    for label, pw in (("user", "reader-2026"), ("owner", "editor-2026")):
        with pikepdf.open(path, password=pw) as pdf:
            allow = pdf.allow
            out[label] = {"opened": True, "pages": len(pdf.pages), "ownerPasswordMatched": pdf.owner_password_matched,
                          "allow": {"print": allow.print_lowres, "printHighQuality": allow.print_highres,
                                    "modify": allow.modify_other, "copy": allow.extract,
                                    "annotate": allow.modify_annotation, "fillForms": allow.modify_form,
                                    "extractForAccessibility": allow.accessibility, "assemble": allow.modify_assembly}}
    try:
        pikepdf.open(path).close()
        out["none"] = {"opened": True}
    except pikepdf.PasswordError as e:
        out["none"] = {"opened": False, "error": "PasswordError"}
        del e
    with open(path, "rb") as fh:
        texts = _texts(fh.read(), "reader-2026")
    out["pdfminerText"] = texts
    allowed = exp["encryption"]["permissionsAllowed"]
    problems = [k for k, v in out["user"]["allow"].items() if v != (k in allowed)]
    if out["user"]["ownerPasswordMatched"] or not out["owner"]["ownerPasswordMatched"] or out["none"]["opened"]:
        problems.append("passwords")
    if problems:
        notes.append(f"{exp['name']}: qpdf reads the permissions or passwords differently: {problems}")
    out["problems"] = problems
    return "passwords", out, {}


def check_operations(path, exp, notes):
    import os

    import pikepdf

    folder = os.path.dirname(path)
    results = []
    problems = []
    for case in exp["operations"]:
        with pikepdf.open(path) as pdf:
            others = {}
            for step in case["steps"]:
                if step["op"] == "insert":
                    other = others.setdefault(step["from"], pikepdf.open(io.BytesIO(_load_sample(folder, step["from"]))))
                    for j, k in enumerate(step["pages"]):
                        pdf.pages.insert(step["at"] + j, other.pages[k])
                elif step["op"] == "remove":
                    for k in sorted(step["pages"], reverse=True):
                        del pdf.pages[k]
                elif step["op"] == "move":
                    page = pdf.pages[step["from"]]
                    del pdf.pages[step["from"]]
                    pdf.pages.insert(step["to"], page)
                elif step["op"] == "rotate":
                    pdf.pages[step["page"]].rotate(step["by"], relative=True)
            buf = io.BytesIO()
            pdf.save(buf, deterministic_id=True)
            for o in others.values():
                o.close()
        data = buf.getvalue()
        texts = _texts(data)
        with pikepdf.open(io.BytesIO(data)) as res:
            rotations = [int(p.obj.get("/Rotate", 0)) % 360 for p in res.pages]
        want_t = [_key(r["text"]) for r in case["result"]]
        want_r = [r["rotation"] for r in case["result"]]
        ok = [_key(t) for t in texts] == want_t and rotations == want_r
        if not ok:
            problems.append({"case": case["name"], "text": texts, "rotations": rotations})
        results.append({"case": case["name"], "matches": ok})
    if problems:
        notes.append(f"{exp['name']}: qpdf's page operations give other pages: {problems}")
    return "operations", {"tool": "qpdf (pikepdf)", "cases": results, "problems": problems}, {}


def check_merge(path, exp, notes):
    import os

    import pikepdf

    folder = path if os.path.isdir(path) else os.path.dirname(path)
    with pikepdf.open(io.BytesIO(_load_sample(folder, "merge-a"))) as a, \
            pikepdf.open(io.BytesIO(_load_sample(folder, "merge-b"))) as b:
        out = pikepdf.new()
        out.pages.extend(a.pages)
        out.pages.extend(b.pages)
        buf = io.BytesIO()
        out.save(buf, deterministic_id=True)
    data = buf.getvalue()
    texts = _texts(data)
    fonts = []
    with pikepdf.open(io.BytesIO(data)) as res:
        for p in res.pages:
            base = str(p.Resources.Font.F1.BaseFont)[1:]
            fonts.append(base.split("+")[-1])
    want = exp["merge"]["result"]
    ok = [_key(t) for t in texts] == [_key(r["text"]) for r in want] and fonts == [r["font"] for r in want]
    if not ok:
        notes.append(f"{exp['name']}: qpdf's merge gives other pages: {texts} {fonts}")
    # The stamp laid over merge-a's pages by qpdf (pikepdf's Page.add_overlay renames clashing resources).
    with pikepdf.open(io.BytesIO(_load_sample(folder, "merge-a"))) as a, \
            pikepdf.open(io.BytesIO(_load_sample(folder, "stamp"))) as s:
        stamp = pikepdf.Page(s.pages[0])
        for k in exp["stamp"]["pages"]:
            pikepdf.Page(a.pages[k]).add_overlay(a.copy_foreign(stamp.as_form_xobject()))
        buf = io.BytesIO()
        a.save(buf, deterministic_id=True)
    stamped = _texts(buf.getvalue())
    stamp_ok = all(all(w in t for w in r["text"]) for t, r in zip(stamped, exp["stamp"]["result"]))
    if not stamp_ok:
        notes.append(f"{exp['name']}: qpdf's overlay gives other text: {stamped}")
    return "merge", {"tool": "qpdf (pikepdf): pages appended; Page.add_overlay", "text": texts, "fonts": fonts,
                     "matches": ok, "stampedText": stamped, "stampMatches": stamp_ok}, {}


def merge_plan(seed):
    """The merge and stamp expectations, as a set of its own (no file: the inputs are merge-a, merge-b and stamp)."""
    desc = ("What merging merge-a and merge-b, and laying the stamp over merge-a's pages, must give: each page with "
            "the objects its own resource names meant (`merge`, `stamp`).")
    exp = {"schema": 1, "name": "merge-plan", "file": None, "kind": "merge", "phases": D11, "description": desc,
           "size": None, "sha256": None, "hostile": None, "merge": MERGE, "stamp": STAMP}
    return Sample("merge-plan", "d11", "merge", D11, desc, None, exp, checks=[check_merge])


def build(seed):
    return [incremental_table(seed), incremental_stream(seed)] + merge_inputs(seed) + \
        [merge_plan(seed), split_source(seed), encrypted(seed), page_operations(seed)]

