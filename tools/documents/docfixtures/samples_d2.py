"""D2, encryption: the standard security handler's revisions 2-6, each with an empty and with a given user password
and an owner password, read against their unencrypted twin (the same objects, numbers and text); files whose user
password is empty and whose permissions only the owner may lift; and damaged files whose catalog lies in an object
stream, whose cross-reference must be rebuilt by scanning (the file key found before the object streams are listed)."""
from .common import Sample, base_expected, helvetica, text_page
from .document import Doc
from .pdf import B, H, N, Stream, String, U16, sha256
from .security import SeededBytes, StandardSecurity

D2 = ["D2"]
OWNER = "owner-pass-1"
USER = "user-pass-1"
WRONG = "not-the-password"

XMP = b"""<?xpacket begin="\xef\xbb\xbf" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">
   <dc:title><rdf:Alt><rdf:li xml:lang="x-default">Supplier review</rdf:li></rdf:Alt></dc:title>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""

CONFIGS = [
    # name, revision, extra arguments, cross-reference kind, version
    ("r2-rc4-40", 2, {}, "table", "1.4"),
    ("r3-rc4-128", 3, {}, "table", "1.4"),
    ("r4-rc4-128", 4, {"cfm": "V2"}, "table", "1.6"),
    ("r4-aes-128", 4, {}, "table", "1.6"),
    ("r4-aes-128-metadata-plain", 4, {"encrypt_metadata": False}, "table", "1.6"),
    ("r4-aes-128-identity-strings", 4, {"string_filter": "Identity"}, "table", "1.6"),
    ("r4-aes-128-xref-stream", 4, {}, "stream", "1.6"),
    ("r5-aes-256", 5, {}, "table", "1.7"),
    ("r6-aes-256", 6, {}, "table", "2.0"),
    ("r6-aes-256-object-streams", 6, {}, "objstm", "2.0"),
]

# /P allowing a user to print (bit 3) and copy (bit 5), nothing else.
PRINT_COPY = -3904 | 4 | 16

# An empty user password and the owner password, permissions to print and copy only: given, the owner password opens
# such a file as its owner (who may change or remove its encryption), not as the user the empty password opens it as.
OWNER_ONLY = [
    ("r4-aes-128-print-copy", 4, {"permissions": PRINT_COPY}, "table", "1.6"),
    ("r6-aes-256-print-copy", 6, {"permissions": PRINT_COPY}, "table", "2.0"),
]

# Damaged files whose catalog lies in an object stream: name, revision, user password, version, damage.
DAMAGED = [
    ("r4-aes-128-object-streams-cut", 4, "", "1.6", "cut"),
    ("r6-aes-256-object-streams-shifted", 6, USER, "2.0", "shifted"),
]

INFO = {"title": "Supplier review", "author": "Purchasing team", "subject": "Prices and delivery times",
        "keywords": None, "creator": None, "producer": "Idrak document samples", "created": None, "modified": None,
        "other": {"Reference": "PR-2024-117"}}


def twin_doc(name, version, seed, metadata_crypt=True):
    doc = Doc(name, version, seed)
    f = helvetica(doc)
    text_page(doc, f, ["Supplier review", "Delivery times improved by two days.",
                       "Prices for steel parts rose by 4 percent."])
    text_page(doc, f, ["Actions", "Ask two suppliers for new quotes.", "Report back at the next meeting."])
    doc.info = {"Title": B(b"Supplier review"), "Author": U16("Purchasing team"),
                "Subject": H(b"Prices and delivery times"), "Producer": B(b"Idrak document samples"),
                "Reference": B(b"PR-2024-117")}
    doc.catalog_extra["Metadata"] = doc.pdf.add(Stream({"Type": N("Metadata"), "Subtype": N("XML")}, XMP,
                                                       crypt=metadata_crypt))
    doc.finish()
    return doc


def plain(seed):
    doc = twin_doc("plain", "1.7", seed)
    data, w = doc.write("table")
    desc = "The unencrypted twin of every encrypted sample: the same objects, numbers, text and information."
    exp = base_expected(doc, data, "plain", "twin", D2, desc, w, info=INFO)
    exp["metadataSha256"] = sha256(XMP)
    return Sample("plain", "d2", "twin", D2, desc, data, exp)


def encrypted(seed, cfg, user, label, checks=None):
    name, rev, kw, kind, version = cfg
    sample = f"{name}-{label}"
    doc = twin_doc(sample, version, seed, metadata_crypt=kw.get("encrypt_metadata", True))
    sec = StandardSecurity(rev, user, OWNER, doc.file_id, seed=f"{seed}|{sample}", **kw)
    if rev == 5:
        doc.pdf.get(doc.catalog)["Extensions"] = {"ADBE": {"BaseVersion": N("1.7"), "ExtensionLevel": 3}}
    data, w = doc.write(kind, security=sec)
    what = {"empty": "an empty user password", "user": f"the user password {user!r}",
            "unicode": f"the user password {user!r} (UTF-8 after SASLprep)"}[label]
    d = sec.describe()
    desc = (f"Standard security handler revision {rev}, {d['cipher']} ({d['lengthBits']} bits"
            + (f", crypt filter {d['cryptFilter']}" if d["cryptFilter"] else "")
            + (", strings not encrypted (/StrF /Identity)" if kw.get("string_filter") == "Identity" else "")
            + (", metadata left plain (/EncryptMetadata false)" if kw.get("encrypt_metadata") is False else "")
            + (", a user may print and copy only" if kw.get("permissions") == PRINT_COPY else "")
            + f"), {what}, owner password {OWNER!r}; cross-reference: {kind}.")
    exp = base_expected(doc, data, sample, "encrypted", D2, desc, w, info=INFO)
    exp["open"].update({"password": user, "encrypted": True})
    exp["encryption"] = {**d, "wrongPassword": WRONG, "encryptDictionary": doc.encrypt_ref.num}
    exp["twin"] = "plain"
    exp["metadataSha256"] = sha256(XMP)
    return Sample(sample, "d2", "encrypted", D2, desc, data, exp, checks=checks)


def check_owner_only(path, exp, notes):
    """qpdf (through pikepdf) on a file whose user password is empty: the empty password opens it as the user, the owner
    password as the owner, and a user may print and copy, nothing else."""
    import pikepdf

    out = {}
    for label, pw in (("empty", ""), ("owner", OWNER), ("wrong", WRONG)):
        try:
            with pikepdf.open(path, password=pw) as pdf:
                allow = pdf.allow
                out[label] = {"opened": True, "ownerPasswordMatched": pdf.owner_password_matched,
                              "userPasswordMatched": pdf.user_password_matched,
                              "allow": {"print": allow.print_lowres, "printHighQuality": allow.print_highres,
                                        "modify": allow.modify_other, "copy": allow.extract,
                                        "annotate": allow.modify_annotation, "fillForms": allow.modify_form,
                                        "extractForAccessibility": allow.accessibility,
                                        "assemble": allow.modify_assembly}}
        except pikepdf.PasswordError:
            out[label] = {"opened": False, "error": "PasswordError"}
    problems = []
    if not out["owner"].get("ownerPasswordMatched") or out["empty"].get("ownerPasswordMatched"):
        problems.append("passwords")
    allowed = {k for k, v in out["empty"].get("allow", {}).items() if v}
    if allowed != {"print", "copy"}:
        problems.append(f"permissions {sorted(allowed)}")
    if problems:
        notes.append(f"d2/{exp['name']}: qpdf reads the passwords or permissions differently: {problems}")
    out["problems"] = problems
    return "passwords", out, {}


def damaged(seed, cfg):
    """An encrypted file with object streams (the catalog in one), damaged so that its cross-reference must be rebuilt
    by scanning: the trailer's /Encrypt and /ID are found only by the scan, in the cross-reference stream's dictionary
    (never encrypted), and the file key is needed before the object streams can be listed."""
    name, rev, user, version, damage = cfg
    doc = twin_doc(name, version, seed)
    sec = StandardSecurity(rev, user, OWNER, doc.file_id, seed=f"{seed}|{name}")
    data, w = doc.write("objstm", security=sec)
    whole = data
    xref = w.revisions[-1]["offset"]
    if damage == "cut":
        start = data.index(b"stream", xref) + len(b"stream")
        start += 2 if data[start:start + 2] == b"\r\n" else 1
        end = data.index(b"endstream", start)
        data = data[:start + (end - start) // 2]
        how = ("cut in the middle of its cross-reference stream's data, as a download that stopped: no startxref, the "
               "stream's dictionary (the trailer's /Root, /Encrypt and /ID) whole")
        broken = "cut in its cross-reference stream: no startxref"
    else:
        insert = b"%edited %%\n"
        head = data.index(b"\n", data.index(b"%\xe2")) + 1
        data = data[:head] + insert + data[head:]
        how = (f"a comment of {len(insert)} bytes inserted after the header once the file was written, so every offset "
               "in its cross-reference stream and startxref is that much short")
        broken = f"every cross-reference offset {len(insert)} bytes short"
    d = sec.describe()
    what = "an empty user password" if user == "" else f"the user password {user!r}"
    desc = (f"Standard security handler revision {rev}, {d['cipher']} ({d['lengthBits']} bits), {what}, owner password "
            f"{OWNER!r}; object streams hold the catalog. Damaged: {how}. A reader rebuilds the cross-reference by "
            "scanning, finds /Encrypt and /ID in the cross-reference stream's dictionary, and needs the file key before "
            "it can list the objects in the object streams.")
    exp = base_expected(doc, data, name, "encrypted", D2, desc, w, info=INFO)
    exp["open"].update({"password": user, "encrypted": True, "repaired": True})
    exp["structure"]["damage"] = broken
    if damage == "cut":
        exp["streams"] = [s for s in exp["streams"] if s["type"] != "XRef"]   # the stream that was cut
    exp["encryption"] = {**d, "wrongPassword": WRONG, "encryptDictionary": doc.encrypt_ref.num}
    exp["twin"] = "plain"
    exp["metadataSha256"] = sha256(XMP)
    return Sample(name, "d2", "encrypted", D2, desc, data, exp, checks=[_undamaged(whole, user)])


def _undamaged(whole, password):
    """A check with the reference tools of a damaged sample's bytes as they were before the damage (in memory): qpdf
    opens them with the password, encrypted, every stream decoding as the JSON says, and pdfminer.six extracts each page's
    text as the JSON has it. The damaged file itself is beyond them (they find no trailer, or no catalog without the
    key), so this is what proves its encryption and content independently."""

    def check(path, exp, notes):
        import hashlib
        import io

        import pikepdf
        from pdfminer.high_level import extract_text

        from .reference import _quiet_logs

        out = {"opened": False}
        try:
            with pikepdf.open(io.BytesIO(whole), password=password) as pdf:
                problems = pdf.check_pdf_syntax() if hasattr(pdf, "check_pdf_syntax") else pdf.check()
                out.update({"opened": True, "pages": len(pdf.pages), "encrypted": bool(pdf.is_encrypted),
                            "problems": [str(p) for p in problems]})
                mismatches = []
                for rec in exp["streams"]:
                    if rec.get("type") in ("XRef", "ObjStm"):
                        continue
                    data = pdf.get_object(tuple(rec["object"])).read_bytes(pikepdf.StreamDecodeLevel.generalized)
                    if hashlib.sha256(data).hexdigest() != rec["decodedSha256"]:
                        mismatches.append(rec["object"])
                out["streamMismatches"] = mismatches
            _quiet_logs()
            texts = extract_text(io.BytesIO(whole), password=password).split("\x0c")[:-1]
            expected = [" ".join(p["text"].split()) for p in exp["pages"]]
            out["textMatches"] = [" ".join(t.split()) for t in texts] == expected
        except Exception as e:  # noqa: BLE001 - recorded
            out["error"] = f"{type(e).__name__}: {e}"
        if not out["opened"] or out.get("streamMismatches") or not out.get("textMatches") or out.get("problems"):
            notes.append(f"d2/{exp['name']}: undamaged, the reference tools read it otherwise: {out}")
        return "undamaged", out, {}

    return check


def unsupported_handler(seed):
    doc = twin_doc("unsupported-handler", "1.7", seed)
    rnd = SeededBytes(seed, "pubsec")
    enc = {"Filter": N("Adobe.PubSec"), "SubFilter": N("adbe.pkcs7.s5"), "V": 4, "R": 4,
           "CF": {"DefaultCryptFilter": {"CFM": N("AESV2"), "Recipients": [String(rnd.take(64), hex=True)],
                                         "Length": 16}},
           "StmF": N("DefaultCryptFilter"), "StrF": N("DefaultCryptFilter")}
    from .pdf import Writer

    w = Writer(doc.pdf)
    t = doc.trailer()
    ref = doc.pdf.add(enc)
    t["Encrypt"] = ref
    w.revision(sorted(doc.pdf.objects), "table", t, first=True)
    data = bytes(w.buf)
    desc = ("Encrypted for a public-key handler (/Filter /Adobe.PubSec): not the standard handler, so the reader "
            "refuses it naming the handler (no recipient key exists: the streams are not really encrypted).")
    exp = {"schema": 1, "name": "unsupported-handler", "file": "unsupported-handler.pdf", "kind": "unsupported",
           "phases": D2, "description": desc, "size": len(data), "sha256": sha256(data), "hostile": None,
           "openError": {"error": "NotSupportedException", "handler": "Adobe.PubSec"}}
    return Sample("unsupported-handler", "d2", "unsupported", D2, desc, data, exp)


def build(seed):
    out = [plain(seed)]
    for cfg in CONFIGS:
        out.append(encrypted(seed, cfg, "", "empty"))
        out.append(encrypted(seed, cfg, USER, "user"))
    out.append(encrypted(seed, ("r6-aes-256", 6, {}, "table", "2.0"), "Grüße-2024", "unicode"))
    for cfg in OWNER_ONLY:
        out.append(encrypted(seed, cfg, "", "empty", checks=[check_owner_only]))
    for cfg in DAMAGED:
        out.append(damaged(seed, cfg))
    out.append(unsupported_handler(seed))
    return out
