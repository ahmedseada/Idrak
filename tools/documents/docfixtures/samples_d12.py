"""D12, signatures: a test PKI made here (pki.py: never a real certificate) and signed samples, each signature written
as its own incremental update with its byte range reserved and filled: PAdES B-B (RSA, RSA-PSS, ECDSA), B-T (a
signature timestamp from the in-process timestamp authority), B-LT (a later update adding the DSS with the
certificates, both CRLs and the OCSP response), a signer whose certificate is revoked (B-LT), a broken digest (a byte
of the signed revision changed afterwards), a wrong chain (signed under a root the check is not given as trusted),
changes after a certification signature (a form filled: allowed by its DocMDP permissions; page content replaced: not
allowed), two signatures (CMS adbe.pkcs7.detached, then PAdES), and the legacy adbe.pkcs7.sha1 and adbe.x509.rsa_sha1
forms. Every byte repeats on every run (pki.py says how).

A signature, as `signatures` lists it: `field`, `subFilter`, `kind`, `signer` and `chain` (certificates, the PKI files
beside), `byteRange`, `contentsReserved` (bytes in /Contents) and `cmsLength`, `revision` (which revision holds it,
0 the first) and `signedRevisionEnd`, `coversWholeFile`, `digestAlgorithm`, `signatureAlgorithm`, `claimedSigningTime`
(/M), `cmsSigningTime` (its signingTime attribute, when it has one), `timestamp` (time, authority, policy, serial),
`reason`, `location`, `docMdp` (a certification signature's permission), `trust` (decision 16: the only certificates a
check is given as trusted), `validationTime` (the time a check validates at), and `results`: `digest` and `signature`
(valid or invalid), `chain` (trusted or untrusted), `timestamp` (valid, or null without one), `revocation` (good,
revoked, or not-checked when the file holds no revocation data and fetching is off), `modifications` (none, allowed,
not-allowed) with `modificationLevel` (none, form-filling, annotations, other)."""
import hashlib

from asn1crypto import core

from .cms import signed_data, timestamp_attr
from .common import Sample, base_expected, helvetica
from .document import Doc
from .filters import flate
from .pdf import B, N, Raw, Stream, String
from .pki import TestPKI, T, pem_files

D12 = ["D12"]
PLACEHOLDER = b"[0 0000000000 0000000000 0000000000]"


def _pdf_date(dt):
    return dt.strftime("D:%Y%m%d%H%M%SZ").encode("ascii")


class SignedDoc:
    """A base document (a page of text, optional empty text field), then revisions: signatures, fills, changes."""

    def __init__(self, name, seed, field=False, version="1.7"):
        self.doc = doc = Doc(name, version, seed)
        self.f = helvetica(doc)
        p = doc.page(420, 300)
        c = p.content
        c.BT()
        c.Tf(self.f, 14)
        c.Td(40, 240)
        for k, line in enumerate(["Quarterly report", "Revenue rose by 12 percent.",
                                  "Approved by the finance team."]):
            if k:
                c.Td(0, -22)
            c.show(line)
            p.line(line)
        c.ET()
        self.page = p
        self.pdf = doc.pdf
        self.fields = []
        self.acroform = self.pdf.alloc()
        self.text_field = None
        if field:
            self.text_field = self.pdf.alloc()
            self.pdf.put(self.text_field, {"FT": N("Tx"), "T": String(b"notes"), "Type": N("Annot"),
                                           "Subtype": N("Widget"), "Rect": [40, 120, 380, 140], "F": 4, "P": p.ref,
                                           "DA": String(b"/Helv 10 Tf 0 g")})
            p.annotations.append(self.text_field)
            self.fields.append(self.text_field)
        self.pdf.put(self.acroform, {"Fields": list(self.fields), "DA": String(b"/Helv 0 Tf 0 g"),
                                     "DR": {"Font": {"Helv": self.f.ref}}})
        doc.catalog_extra["AcroForm"] = self.acroform
        doc.info = {"Title": B(b"Quarterly report"), "Producer": B(b"Idrak document samples")}
        doc.finish()
        self.data0, self.w = doc.write("table")
        self.revision_ends = [len(self.w.buf)]
        self.signatures = []

    def _update(self, nums, free=None):
        self.w.revision(sorted(set(nums)), "table", self.doc.trailer(), prev=self.w.last_xref, free=free)
        self.revision_ends.append(len(self.w.buf))

    def _appearance(self, text, w, h):
        content = (b"q 0.97 0.97 1 rg 0 0 %d %d re f 0.3 G 1 w 0.5 0.5 %d %d re S Q BT /Helv 9 Tf 0 0 0.4 rg 6 %d Td "
                   b"(%s) Tj ET\n" % (w, h, w - 1, h - 1, h // 2 - 3, text.encode("latin-1")))
        return self.pdf.add(Stream({"Type": N("XObject"), "Subtype": N("Form"), "BBox": [0, 0, w, h],
                                    "Resources": {"Font": {"Helv": self.f.ref}}, "Filter": N("FlateDecode")},
                                   flate(content), content))

    def sign(self, field_name, rect, subfilter, cms_maker, reason, location, when, reserve=16384, certify=None,
             cert_entry=None, label=None):
        """A signature as an incremental update: the signature dictionary with its /ByteRange and /Contents reserved,
        its field and widget, the page's /Annots and the AcroForm (/SigFlags 3); then the byte range filled and the
        signature made over it (`cms_maker(signed_bytes) -> the bytes for /Contents`)."""
        pdf = self.pdf
        sig = {"Type": N("Sig"), "Filter": N("Adobe.PPKLite"), "SubFilter": N(subfilter),
               "ByteRange": Raw(PLACEHOLDER), "Contents": Raw(b"<" + b"0" * (2 * reserve) + b">"),
               "M": String(_pdf_date(when)), "Reason": String(reason.encode("ascii")),
               "Location": String(location.encode("ascii"))}
        if cert_entry is not None:
            sig["Cert"] = cert_entry
        if certify is not None:
            sig["Reference"] = [{"Type": N("SigRef"), "TransformMethod": N("DocMDP"),
                                 "TransformParams": {"Type": N("TransformParams"), "P": certify, "V": N("1.2")}}]
        sig_ref = pdf.add(sig)
        ap = self._appearance(f"Signed by {label or 'Example Test Signer'}", int(rect[2] - rect[0]),
                              int(rect[3] - rect[1]))
        field = pdf.add({"FT": N("Sig"), "T": String(field_name.encode("ascii")), "V": sig_ref, "Type": N("Annot"),
                         "Subtype": N("Widget"), "Rect": list(rect), "F": 132, "P": self.page.ref, "AP": {"N": ap}})
        self.fields.append(field)
        page = dict(pdf.get(self.page.ref))
        page["Annots"] = list(page.get("Annots", [])) + [field]
        pdf.put(self.page.ref, page)
        acro = dict(pdf.get(self.acroform))
        acro["Fields"] = list(self.fields)
        acro["SigFlags"] = 3
        pdf.put(self.acroform, acro)
        nums = [sig_ref.num, ap.num, field.num, self.page.ref.num, self.acroform.num]
        if certify is not None:
            cat = dict(pdf.get(self.doc.catalog))
            cat["Perms"] = {"DocMDP": sig_ref}
            pdf.put(self.doc.catalog, cat)
            nums.append(self.doc.catalog.num)
        start = len(self.w.buf)
        self._update(nums)
        buf = self.w.buf
        at = buf.index(b"/ByteRange " + PLACEHOLDER, start)
        c0 = buf.index(b"/Contents <", at) + len(b"/Contents ")
        c1 = c0 + 2 * reserve + 2
        total = len(buf)
        byte_range = [0, c0, c1, total - c1]
        text = b"[%d %d %d %d]" % tuple(byte_range)
        buf[at + len(b"/ByteRange "):at + len(b"/ByteRange ") + len(PLACEHOLDER)] = text.ljust(len(PLACEHOLDER))
        signed = bytes(buf[:c0]) + bytes(buf[c1:])
        contents = cms_maker(signed)
        if len(contents) > reserve:
            raise ValueError(f"{len(contents)} bytes do not fit the {reserve} reserved")
        buf[c0 + 1:c0 + 1 + 2 * len(contents)] = contents.hex().upper().encode("ascii")
        info = {"field": field_name, "subFilter": subfilter, "byteRange": byte_range, "contentsReserved": reserve,
                "cmsLength": len(contents), "revision": len(self.revision_ends) - 1, "signedRevisionEnd": total,
                "reason": reason, "location": location, "claimedSigningTime": when.isoformat(),
                "docMdp": certify, "contentsSha1": hashlib.sha1(contents).hexdigest().upper()}
        self.signatures.append(info)
        return info, contents

    def fill(self, value):
        """An update filling the text field (its value and a new appearance): form filling."""
        pdf = self.pdf
        content = (b"/Tx BMC q BT /Helv 10 Tf 0 g 2 6 Td (" + value.encode("latin-1") + b") Tj ET Q EMC\n")
        ap = pdf.add(Stream({"Type": N("XObject"), "Subtype": N("Form"), "BBox": [0, 0, 340, 20],
                             "Resources": {"Font": {"Helv": self.f.ref}}, "Filter": N("FlateDecode")},
                            flate(content), content))
        d = dict(pdf.get(self.text_field))
        d["V"] = String(value.encode("latin-1"))
        d["AP"] = {"N": ap}
        pdf.put(self.text_field, d)
        self._update([ap.num, self.text_field.num])

    def change_content(self):
        """An update replacing the page's content stream: a change no signature permission allows."""
        pdf = self.pdf
        ref = pdf.get(self.page.ref)["Contents"]
        data = (b"BT\n/F1 14 Tf\n40 240 Td\n(Quarterly report) Tj\n0 -22 Td\n(Revenue rose by 21 percent.) Tj\n"
                b"0 -22 Td\n(Approved by the finance team.) Tj\nET\n")
        pdf.put(ref, Stream({"Filter": N("FlateDecode")}, flate(data), data))
        self._update([ref.num])

    def add_dss(self, pki, contents_list, revoked=False):
        """An update adding the DSS: the chain's certificates, both CRLs, the OCSP response, and a VRI entry per
        signature (keyed by the SHA-1 of its /Contents, upper-case hex)."""
        pdf = self.pdf

        def stream(data):
            return pdf.add(Stream({"Filter": N("FlateDecode")}, flate(data), data))

        signer = pki.revoked if revoked else pki.signer
        certs = [stream(p.der) for p in (signer, pki.intermediate, pki.root, pki.tsa)]
        crls = [stream(pki.intermediate_crl), stream(pki.root_crl)]
        ocsps = [] if revoked else [stream(pki.signer_ocsp)]
        vri = {}
        for contents in contents_list:
            vri[hashlib.sha1(contents).hexdigest().upper()] = {"Cert": certs, "CRL": crls, "OCSP": ocsps}
        dss = pdf.add({"Type": N("DSS"), "Certs": certs, "CRLs": crls, "OCSPs": ocsps,
                       "VRI": {k: pdf.add(v) for k, v in vri.items()}})
        cat = dict(pdf.get(self.doc.catalog))
        cat["DSS"] = dss
        cat["Extensions"] = {"ESIC": {"BaseVersion": N("1.7"), "ExtensionLevel": 2}}
        pdf.put(self.doc.catalog, cat)
        nums = [r.num for r in certs + crls + ocsps] + [dss.num, self.doc.catalog.num] + \
            [v.num for v in pdf.get(dss)["VRI"].values()]
        self._update(nums)
        return {"certs": [f"pki.{p.label}.cer" for p in (signer, pki.intermediate, pki.root, pki.tsa)],
                "crls": ["pki.intermediate.crl", "pki.root.crl"], "ocsps": [] if revoked else ["pki.signer.ocsp"],
                "vri": sorted(vri)}

    def data(self):
        return bytes(self.w.buf)


def _pades(pki, signer, algorithm="rsa", timestamp=False, label="", chain=None, digest="sha256"):
    def make(signed):
        dv = hashlib.sha256(signed).digest()
        unsigned = (lambda s: [timestamp_attr(pki, s, T(2026, 3, 2, 9, 30, 5), label)]) if timestamp else None
        salt = hashlib.sha256(f"{pki.seed}|pss|{label}".encode()).digest() if algorithm == "pss" else None
        return signed_data(signer, None, digest_value=dv, chain=chain if chain is not None else [pki.intermediate],
                           algorithm=algorithm, signing_time=None, ess=True, pss_salt=salt,
                           unsigned_for=unsigned).dump()
    return make


def _cms_detached(pki, signer):
    def make(signed):
        return signed_data(signer, None, digest_value=hashlib.sha256(signed).digest(), chain=[pki.intermediate],
                           signing_time=pki.SIGNING_TIME, ess=False).dump()
    return make


def _pkcs7_sha1(pki):
    def make(signed):
        inner = hashlib.sha1(signed).digest()
        return signed_data(pki.signer, inner, embed=True, digest="sha1", chain=[pki.intermediate],
                           signing_time=pki.SIGNING_TIME, ess=False).dump()
    return make


def _x509_sha1(pki):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    def make(signed):
        sig = pki.signer.key.sign(signed, padding.PKCS1v15(), hashes.SHA1())
        return core.OctetString(sig).dump()
    return make


def _signer_json(p):
    d = p.describe()
    return {"subject": d["subject"], "issuer": d["issuer"], "serial": d["serial"], "certificate": d["file"],
            "sha256": d["sha256"], "key": d["key"]}


def _expect(info, pki, signer, kind, sig_alg, digest="sha256", timestamp=None, chain=("intermediate",),
            results=None, whole=None, cms_time=None):
    out = dict(info)
    out.pop("contentsSha1", None)
    out.update({"kind": kind, "signer": _signer_json(signer), "chain": [f"pki.{c}.cer" for c in chain],
                "digestAlgorithm": digest, "signatureAlgorithm": sig_alg, "timestamp": timestamp,
                "cmsSigningTime": cms_time, "trust": ["pki.root.cer"],
                "validationTime": pki.VALIDATION_TIME.isoformat(), "coversWholeFile": whole,
                "results": results})
    return out


def _results(digest="valid", signature="valid", chain="trusted", timestamp=None, revocation="not-checked",
             modifications="none", level="none"):
    return {"digest": digest, "signature": signature, "chain": chain, "timestamp": timestamp,
            "revocation": revocation, "modifications": modifications, "modificationLevel": level}


TS = {"time": "2026-03-02T09:30:05+00:00", "authority": "C=GB,O=Example Organisation,CN=Example Test Timestamp "
      "Authority", "policy": "2.999.1.2", "certificate": "pki.tsa.cer"}


def _sample(sd, name, desc, signatures, pki, extra=None, checks=None):
    data = sd.data()
    exp = base_expected(sd.doc, data, name, "signed", D12, desc, sd.w, objects=False)
    exp["structure"]["revisionEnds"] = sd.revision_ends
    for s in signatures:
        if s["coversWholeFile"] is None:
            s["coversWholeFile"] = s["signedRevisionEnd"] == len(data)
    exp["signatures"] = signatures
    exp["pki"] = "pki"
    exp.update(extra or {})
    return Sample(name, "d12", "signed", D12, desc, data, exp, checks=checks or [check_signatures])


def build(seed):
    pki = TestPKI(seed)
    files, parties = pem_files(pki)
    out = []
    desc = ("The test PKI the D12 samples are signed with (never a real certificate): certificates (.cer DER, .pem), "
            "their throwaway private keys (unencrypted PKCS #8, for D12's signing tests), the root's and the "
            "intermediate's CRLs (the intermediate's revokes 'Example Revoked Signer' from 2026-02-01) and an OCSP "
            "response (good) for the signer; a timestamp authority with its own certificate.")
    exp = {"schema": 1, "name": "pki", "file": None, "kind": "pki", "phases": D12, "description": desc, "size": None,
           "sha256": None, "hostile": None, "parties": {p.label: p.describe() for p in parties},
           "crls": {"pki.root.crl": {"issuer": "Example Test Root CA", "revoked": []},
                    "pki.intermediate.crl": {"issuer": "Example Test Intermediate CA",
                                             "revoked": [{"serial": 0x1005, "date": "2026-02-01T00:00:00+00:00",
                                                          "reason": "keyCompromise"}]}},
           "ocsp": {"pki.signer.ocsp": {"certificate": "pki.signer.cer", "status": "good",
                                        "producedAt": "2026-03-02T09:31:00+00:00", "responder": "the intermediate"}},
           "signingTime": pki.SIGNING_TIME.isoformat(), "validationTime": pki.VALIDATION_TIME.isoformat(),
           "timestampPolicy": "2.999.1.2"}
    out.append(Sample("pki", "d12", "pki", D12, desc, None, exp, files, reference=None))
    when = pki.SIGNING_TIME

    for name, alg, signer, kind_alg in (("pades-bb", "rsa", pki.signer, "rsa-pkcs1v15-sha256"),
                                        ("pades-bb-pss", "pss", pki.signer, "rsassa-pss-sha256-salt32"),
                                        ("pades-bb-ecdsa", "ecdsa", pki.signer_ec, "ecdsa-p256-sha256")):
        sd = SignedDoc(name, seed)
        info, _ = sd.sign("Signature1", [240, 30, 400, 70], "ETSI.CAdES.detached", _pades(pki, signer, alg, label=name),
                          "Approved", "Head office", when, label=signer.cert.subject.rfc4514_string().split("CN=")[1]
                          .split(",")[0])
        sig = _expect(info, pki, signer, "PAdES B-B", kind_alg, results=_results())
        out.append(_sample(sd, name, f"PAdES baseline B-B ({kind_alg}), one approval signature over the whole file.",
                           [sig], pki))

    sd = SignedDoc("pades-bt", seed)
    info, _ = sd.sign("Signature1", [240, 30, 400, 70], "ETSI.CAdES.detached",
                      _pades(pki, pki.signer, timestamp=True, label="pades-bt"), "Approved", "Head office", when)
    out.append(_sample(sd, "pades-bt", "PAdES B-T: B-B with a signature timestamp (RFC 3161, from the test timestamp "
                       "authority, 5 seconds after the claimed time).",
                       [_expect(info, pki, pki.signer, "PAdES B-T", "rsa-pkcs1v15-sha256", timestamp=TS,
                                results=_results(timestamp="valid"))], pki))

    for name, revoked in (("pades-blt", False), ("pades-revoked", True)):
        sd = SignedDoc(name, seed)
        signer = pki.revoked if revoked else pki.signer
        info, contents = sd.sign("Signature1", [240, 30, 400, 70], "ETSI.CAdES.detached",
                                 _pades(pki, signer, timestamp=True, label=name), "Approved", "Head office", when,
                                 label=signer.cert.subject.rfc4514_string().split("CN=")[1].split(",")[0])
        dss = sd.add_dss(pki, [contents], revoked=revoked)
        res = _results(timestamp="valid", revocation="revoked" if revoked else "good",
                       chain="untrusted" if revoked else "trusted", modifications="allowed", level="none")
        sig = _expect(info, pki, signer, "PAdES B-LT", "rsa-pkcs1v15-sha256", timestamp=TS, results=res, whole=False)
        if revoked:
            desc = ("PAdES B-LT signed with a certificate the intermediate's CRL (in the DSS) revoked on 2026-02-01, "
                    "before the signature's time: the revocation check fails.")
        else:
            desc = ("PAdES B-LT: B-T, then an update adding the DSS (the chain's and the timestamp authority's "
                    "certificates, both CRLs, the signer's OCSP response, a VRI entry): revocation is checked from "
                    "the file alone. The DSS update is after the signature and allowed (it changes nothing signed).")
        out.append(_sample(sd, name, desc, [sig], pki, extra={"dss": dss}))

    sd = SignedDoc("broken-digest", seed)
    info, _ = sd.sign("Signature1", [240, 30, 400, 70], "ETSI.CAdES.detached", _pades(pki, pki.signer, label="bd"),
                      "Approved", "Head office", when)
    buf = sd.w.buf
    at = buf.index(b"(Quarterly report)", 0, sd.revision_ends[0])
    buf[at + 11] = ord("R")  # "Quarterly Report": one signed byte changed, the file's structure intact
    out.append(_sample(sd, "broken-digest", "PAdES B-B whose signed bytes were changed afterwards (one letter of the "
                       "title in the first revision): the digest no longer matches.",
                       [_expect(info, pki, pki.signer, "PAdES B-B", "rsa-pkcs1v15-sha256",
                                results=_results(digest="invalid", signature="valid"))], pki,
                       extra={"damage": {"offset": at + 11, "was": "r", "now": "R"}}))

    sd = SignedDoc("wrong-chain", seed)
    info, _ = sd.sign("Signature1", [240, 30, 400, 70], "ETSI.CAdES.detached",
                      _pades(pki, pki.other_signer, label="wc", chain=[pki.other_root]), "Approved", "Head office",
                      when, label="Example Untrusted Signer")
    out.append(_sample(sd, "wrong-chain", "PAdES B-B signed under 'Example Other Root CA', which the check is not "
                       "given as trusted (decision 16): intact and valid, but untrusted.",
                       [_expect(info, pki, pki.other_signer, "PAdES B-B", "rsa-pkcs1v15-sha256", chain=("other-root",),
                                results=_results(chain="untrusted"))], pki))

    for name, allowed in (("changed-allowed", True), ("changed-not-allowed", False)):
        sd = SignedDoc(name, seed, field=True)
        info, _ = sd.sign("Signature1", [240, 30, 400, 70], "ETSI.CAdES.detached",
                          _pades(pki, pki.signer, label=name), "Certified", "Head office", when, certify=2)
        if allowed:
            sd.fill("Checked on 3 March")
            desc = ("A certification signature (DocMDP, P 2: form filling allowed) and then an update filling the text "
                    "field 'notes' (its value and appearance): changed after signing, as the permission allows.")
            res = _results(modifications="allowed", level="form-filling")
        else:
            sd.change_content()
            desc = ("A certification signature (DocMDP, P 2) and then an update replacing the page's content (12 "
                    "percent becomes 21): a change no permission allows.")
            res = _results(modifications="not-allowed", level="other")
        sig = _expect(info, pki, pki.signer, "PAdES B-B", "rsa-pkcs1v15-sha256", results=res, whole=False)
        sig["certification"] = True
        out.append(_sample(sd, name, desc, [sig], pki))

    sd = SignedDoc("two-signatures", seed)
    info1, _ = sd.sign("Signature1", [20, 30, 180, 70], "adbe.pkcs7.detached", _cms_detached(pki, pki.signer),
                       "Reviewed", "Branch office", when)
    info2, _ = sd.sign("Signature2", [240, 30, 400, 70], "ETSI.CAdES.detached",
                       _pades(pki, pki.signer_ec, "ecdsa", label="two-2"), "Approved", "Head office",
                       T(2026, 3, 3, 14, 0, 0), label="Example Test Signer EC")
    s1 = _expect(info1, pki, pki.signer, "CMS", "rsa-pkcs1v15-sha256", whole=False, cms_time=when.isoformat(),
                 results=_results(modifications="allowed", level="form-filling"))
    s2 = _expect(info2, pki, pki.signer_ec, "PAdES B-B", "ecdsa-p256-sha256", results=_results())
    out.append(_sample(sd, "two-signatures", "Two approval signatures: a CMS one (adbe.pkcs7.detached, with a "
                       "signingTime attribute), then a PAdES one (ECDSA) in a later update adding its field. The first "
                       "covers its revision only; the second signature is a change the first allows.", [s1, s2], pki))

    sd = SignedDoc("legacy-pkcs7-sha1", seed, version="1.4")
    info, _ = sd.sign("Signature1", [240, 30, 400, 70], "adbe.pkcs7.sha1", _pkcs7_sha1(pki), "Approved",
                      "Head office", when)
    out.append(_sample(sd, "legacy-pkcs7-sha1", "The legacy adbe.pkcs7.sha1 form: the CMS encapsulates the SHA-1 digest "
                       "of the byte range (20 bytes) and signs it (SHA-1, RSA). Not PAdES; pyHanko does not read the "
                       "form, so the reference check is pyHanko's CMS check of the SignedData and the encapsulated "
                       "digest compared with the byte range's.",
                       [_expect(info, pki, pki.signer, "legacy PKCS #7 SHA-1", "rsa-pkcs1v15-sha1", digest="sha1",
                                cms_time=when.isoformat(), results=_results())], pki,
                       checks=[check_legacy]))

    sd = SignedDoc("legacy-x509-sha1", seed, version="1.4")
    cert_entry = [String(pki.signer.der, hex=True), String(pki.intermediate.der, hex=True)]
    info, _ = sd.sign("Signature1", [240, 30, 400, 70], "adbe.x509.rsa_sha1", _x509_sha1(pki), "Approved",
                      "Head office", when, reserve=512, cert_entry=cert_entry)
    out.append(_sample(sd, "legacy-x509-sha1", "The legacy adbe.x509.rsa_sha1 form: /Contents holds a DER OCTET STRING "
                       "with a PKCS #1 v1.5 SHA-1 signature of the byte range, /Cert the signer's certificate and the "
                       "intermediate's. The reference check verifies the signature with cryptography and the chain "
                       "with pyhanko-certvalidator (pyHanko does not read the form).",
                       [_expect(info, pki, pki.signer, "legacy PKCS #1 SHA-1", "rsa-pkcs1v15-sha1", digest="sha1",
                                results=_results())], pki, checks=[check_legacy]))
    return out


# -- the reference validator -----------------------------------------------------------------------------------------------

def _pki(exp_path):
    import os

    from asn1crypto import x509 as ax509

    folder = os.path.dirname(exp_path)

    def load(name):
        with open(os.path.join(folder, name), "rb") as fh:
            return fh.read()

    certs = {n: ax509.Certificate.load(load(f"pki.{n}.cer")) for n in ("root", "intermediate", "tsa", "signer",
                                                                        "signer-ec", "revoked", "other-root",
                                                                        "other-signer")}
    return certs, [load("pki.root.crl"), load("pki.intermediate.crl")], [load("pki.signer.ocsp")]


def _level(level):
    name = getattr(level, "name", str(level)).lower()
    return {"none": "none", "lta_updates": "none", "form_filling": "form-filling", "annotations": "annotations",
            "other": "other"}.get(name, name)


def _quiet_pyhanko():
    """pyHanko logs each failure the broken, wrong-chain and changed samples are made to cause, with a traceback; the
    results are recorded in the JSON, so its log is kept to critical messages."""
    import logging

    for name in ("pyhanko", "pyhanko_certvalidator"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def check_signatures(path, exp, notes):
    """pyHanko validates every signature with the trust the JSON gives (the test root alone), at the validation time,
    fetching nothing; B-LT samples also with the DSS alone, revocation required, at the timestamp's time."""
    from datetime import datetime

    _quiet_pyhanko()
    from pyhanko.pdf_utils.reader import PdfFileReader
    from pyhanko.sign.validation import DocumentSecurityStore, validate_pdf_signature
    from pyhanko_certvalidator import ValidationContext

    certs, crls, ocsps = _pki(path)
    moment = datetime.fromisoformat(exp["signatures"][0]["validationTime"])
    results = []
    problems = []
    with open(path, "rb") as fh:
        reader = PdfFileReader(fh, strict=False)
        embedded = reader.embedded_signatures
        for want, emb in zip(exp["signatures"], embedded):
            vc = ValidationContext(trust_roots=[certs["root"]], other_certs=[certs["intermediate"]], moment=moment,
                                   allow_fetching=False, revocation_mode="soft-fail")
            st = validate_pdf_signature(emb, signer_validation_context=vc, ts_validation_context=vc)
            got = {"field": emb.field_name, "intact": st.intact, "valid": st.valid, "trusted": st.trusted,
                   "coverage": st.coverage.name, "modificationLevel": _level(st.modification_level),
                   "docmdpOk": st.docmdp_ok, "bottomLine": st.bottom_line,
                   "timestamp": None if st.timestamp_validity is None else
                   {"intact": st.timestamp_validity.intact, "valid": st.timestamp_validity.valid,
                    "trusted": st.timestamp_validity.trusted,
                    "time": st.timestamp_validity.timestamp.isoformat()},
                   "mechanism": st.pkcs7_signature_mechanism, "digest": st.md_algorithm}
            if want["kind"] == "PAdES B-LT":
                # Revocation from the file alone: a context holding the DSS's certificates, CRLs and OCSP responses,
                # revocation required (hard-fail), at the timestamp's time, nothing fetched.
                dss = DocumentSecurityStore.read_dss(reader)
                ts_time = st.timestamp_validity.timestamp if st.timestamp_validity else moment
                lt_vc = dss.as_validation_context({"trust_roots": [certs["root"]], "moment": ts_time,
                                                   "allow_fetching": False, "revocation_mode": "hard-fail"})
                try:
                    lt = validate_pdf_signature(emb, signer_validation_context=lt_vc, ts_validation_context=lt_vc)
                    rd = getattr(lt, "revocation_details", None)
                    got["lt"] = {"intact": lt.intact, "valid": lt.valid, "trusted": lt.trusted,
                                 "revocation": "good" if lt.trusted else "failed", "time": ts_time.isoformat(),
                                 "details": None if rd is None else
                                 {"caRevoked": rd.ca_revoked, "revocationDate": rd.revocation_date.isoformat(),
                                  "reason": rd.revocation_reason.native}}
                except Exception as e:  # noqa: BLE001 - recorded
                    got["lt"] = {"error": f"{type(e).__name__}: {str(e)[:200]}", "revocation": "failed"}
            results.append(got)
            r = want["results"]
            mismatch = []
            if got["intact"] != (r["digest"] == "valid"):
                mismatch.append("digest")
            if got["intact"] and got["valid"] != (r["signature"] == "valid"):
                mismatch.append("signature")
            if want["kind"] != "PAdES B-LT" and got["trusted"] != (r["chain"] == "trusted") and got["intact"]:
                mismatch.append("chain")
            if (r["timestamp"] == "valid") != bool(got["timestamp"] and got["timestamp"]["valid"]
                                                    and got["timestamp"]["trusted"]):
                mismatch.append("timestamp")
            if want["kind"] == "PAdES B-LT":
                if (r["revocation"] == "good") != (got["lt"].get("revocation") == "good"):
                    mismatch.append("revocation")
            if r["modifications"] == "not-allowed" and got["docmdpOk"]:
                mismatch.append("modifications")
            if r["modifications"] in ("none", "allowed") and got["intact"] and not got["docmdpOk"]:
                mismatch.append("modifications")
            if r["modificationLevel"] in ("form-filling", "other") and got["modificationLevel"] != r["modificationLevel"]:
                mismatch.append("modificationLevel")
            whole = got["coverage"] == "ENTIRE_FILE"
            if whole != want["coversWholeFile"]:
                mismatch.append("coversWholeFile")
            if mismatch:
                problems.append({"field": want["field"], "differs": mismatch})
    if len(embedded) != len(exp["signatures"]):
        problems.append({"signatures": [len(exp["signatures"]), len(embedded)]})
    if problems:
        notes.append(f"{exp['name']}: pyHanko validates differently: {problems}")
    return "validator", {"tool": "pyHanko (pyhanko-certvalidator), trust: pki.root.cer only, no fetching; B-LT also "
                                 "with the DSS alone, revocation required, at the timestamp's time",
                         "signatures": results, "problems": problems}, {}


def check_legacy(path, exp, notes):
    """The legacy forms: pyHanko's CMS check of the SignedData (adbe.pkcs7.sha1) or cryptography's PKCS #1 check
    (adbe.x509.rsa_sha1), the signed bytes' SHA-1 compared, and the chain by pyhanko-certvalidator."""
    import asyncio
    from datetime import datetime

    from asn1crypto import cms, x509 as ax509
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from pyhanko_certvalidator import CertificateValidator, ValidationContext

    _quiet_pyhanko()
    certs, _, _ = _pki(path)
    want = exp["signatures"][0]
    with open(path, "rb") as fh:
        data = fh.read()
    a, b, c, d = want["byteRange"]
    signed = data[a:a + b] + data[c:c + d]
    contents = bytes.fromhex(data[b + 1:c - 1].decode("ascii"))
    out = {}
    if want["subFilter"] == "adbe.pkcs7.sha1":
        from pyhanko.sign.validation.generic_cms import validate_sig_integrity

        ci = cms.ContentInfo.load(contents, strict=False)
        sd = ci["content"]
        si = sd["signer_infos"][0]
        sid = si["sid"].chosen  # the signer by issuer and serial: the certificate SET OF is in DER order
        signer_cert = next(c.chosen for c in sd["certificates"]
                           if c.chosen.serial_number == sid["serial_number"].native
                           and c.chosen.issuer == sid["issuer"])
        econtent = sd["encap_content_info"]["content"].native
        intact, valid = validate_sig_integrity(si, signer_cert, expected_content_type="data",
                                               actual_digest=hashlib.sha1(econtent).digest())
        out = {"validator": "pyHanko validate_sig_integrity and the encapsulated digest",
               "encapsulatedDigestMatches": econtent == hashlib.sha1(signed).digest(), "intact": intact,
               "valid": valid}
        digest_ok = out["encapsulatedDigestMatches"] and intact
        signature_ok = valid
    else:
        from asn1crypto import core

        sig = core.OctetString.load(contents, strict=False).native  # the reserved zeros after it are left
        cert = x509.load_der_x509_certificate(certs["signer"].dump())
        try:
            cert.public_key().verify(sig, signed, padding.PKCS1v15(), hashes.SHA1())
            ok = True
        except Exception:  # noqa: BLE001 - recorded
            ok = False
        out = {"validator": "cryptography (PKCS #1 v1.5, SHA-1) over the byte range", "verified": ok}
        digest_ok = signature_ok = ok
        signer_cert = certs["signer"]
    moment = datetime.fromisoformat(want["validationTime"])
    vc = ValidationContext(trust_roots=[certs["root"]], other_certs=[certs["intermediate"]], moment=moment,
                           allow_fetching=False, revocation_mode="soft-fail")
    try:
        asyncio.run(CertificateValidator(ax509.Certificate.load(signer_cert.dump()), validation_context=vc)
                    .async_validate_usage({"digital_signature"}))
        trusted = True
    except Exception as e:  # noqa: BLE001 - recorded
        trusted = False
        out["chainError"] = f"{type(e).__name__}: {str(e)[:200]}"
    out["chainTrusted"] = trusted
    r = want["results"]
    problems = [k for k, ok in (("digest", digest_ok == (r["digest"] == "valid")),
                                ("signature", signature_ok == (r["signature"] == "valid")),
                                ("chain", trusted == (r["chain"] == "trusted"))) if not ok]
    if problems:
        notes.append(f"{exp['name']}: the legacy check differs: {problems}")
    out["problems"] = problems
    return "validator", out, {}

