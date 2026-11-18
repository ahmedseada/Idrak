"""D2, encryption: the standard security handler's revisions 2-6, each with an empty and with a given user password
and an owner password, read against their unencrypted twin (the same objects, numbers and text)."""
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


def encrypted(seed, cfg, user, label):
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
            + f"), {what}, owner password {OWNER!r}; cross-reference: {kind}.")
    exp = base_expected(doc, data, sample, "encrypted", D2, desc, w, info=INFO)
    exp["open"].update({"password": user, "encrypted": True})
    exp["encryption"] = {**d, "wrongPassword": WRONG, "encryptDictionary": doc.encrypt_ref.num}
    exp["twin"] = "plain"
    exp["metadataSha256"] = sha256(XMP)
    return Sample(sample, "d2", "encrypted", D2, desc, data, exp)


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
    out.append(unsupported_handler(seed))
    return out
