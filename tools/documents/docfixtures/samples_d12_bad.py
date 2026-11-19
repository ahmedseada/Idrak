"""D12's known-bad cases (owner, 2026-11-19, 1a): signed samples whose certificate path, revocation data or timestamp a
check must refuse, each with its chain in its CMS (and its DSS) and the app's trust (the test root alone), made with the
test PKI's tooling (pki.py: every key and byte from the seed, never a real certificate, every name generic). They are D12
samples like the others (`signatures`, `results`, pyHanko's verdicts); each signature adds `knownBad`:

- `case`: the case's name; `step`: the check step that refuses it; `detail`: a part of that step's reason;
- `failsWith`: null when the step fails with the JSON's trust and time alone, or "RequireRevocation" when a revocation
  source that cannot be used leaves the step "not checked" (soft-fail) and only a check that requires revocation fails;
- `passesWith`: where an option of the check is the cause, the option and a value that lets the step pass
  (`MaxChainLength`, `MinimumRsaKeyBits`).

Where Idrak's rule differs from pyHanko's by design, the signature's `referenceDiffers` names the result and why (pyHanko
builds a path of any length; Idrak stops at its limit).

The cases: a signer expired, and not yet valid, at the signing time; a signer's certificate whose signature is not its
issuer's; an issuer that is not a CA; a path past an issuer's pathLenConstraint; an issuer without keyCertSign; a signer
without digitalSignature or nonRepudiation; a signer with an unknown critical extension; an issuer name that names no
certificate; two CAs that issued each other (a loop) and not the root; a signer revoked by its OCSP response (the CRL
case is pades-revoked); a CRL and an OCSP response, each revoking the signer, signed by the wrong key; a timestamp whose
token stamps other bytes; a path longer than the check's limit (14 certificates); a 1024-bit RSA signer (under the 2048
a check takes by default, pyHanko's rule: the signature verifies, the path is refused)."""
import hashlib

from asn1crypto import core, ocsp
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.x509.oid import ObjectIdentifier

from .cms import signed_data, timestamp_attr
from .pki import Party, T, _cert, name, rsa_key
from .samples_d12 import TS, SignedDoc, _expect, _pades, _results, _sample

WHEN = T(2026, 3, 2, 9, 30, 0)
REVOKED_AT = T(2026, 2, 15)
LEAF_USAGES = ("digital_signature", "content_commitment")
CA_USAGES = ("key_cert_sign", "crl_sign")
UNKNOWN_EXTENSION = "2.999.1.9"  # the ITU-T/ISO example arc (2.999): an extension no check reads


def _files(*parties):
    """The new parties' certificates beside the sample (DER), as the PKI's are (the PKI's own are written once)."""
    return {f"pki.{p.label}.cer": p.der for p in parties if p.label.startswith("bad-")}


def _leaf(pki, label, cn, key, serial, issuer=None, issuer_key=None, issuer_name=None, start=T(2025, 6, 1),
          end=T(2035, 6, 1), usages=LEAF_USAGES, extra=()):
    """A signer's certificate, by default the intermediate's with its list and responder addresses."""
    issuer = issuer or pki.intermediate
    addresses = issuer is pki.intermediate
    cert = _cert(name(cn), key, issuer_name or issuer.cert.subject, issuer_key or issuer.key, serial, start, end,
                 usages=usages, issuer_cert=issuer.cert,
                 crl_url="http://pki.example.com/intermediate.crl" if addresses else None,
                 ocsp_url="http://ocsp.example.com" if addresses else None, extra=extra)
    return Party(label, key, cert)


def _ca(label, cn, key, serial, issuer, ca=True, path_len=None, usages=CA_USAGES):
    cert = _cert(name(cn), key, issuer.cert.subject, issuer.key, serial, T(2025, 1, 1), T(2040, 1, 1), ca=ca,
                 path_len=path_len, usages=usages, issuer_cert=issuer.cert)
    return Party(label, key, cert)


def _crl(issuer, signing_key, revoked, number):
    """A CRL in `issuer`'s name (its name and key identifier) signed with `signing_key`, revoking `revoked` serials."""
    b = (x509.CertificateRevocationListBuilder().issuer_name(issuer.cert.subject).last_update(T(2026, 3, 1))
         .next_update(T(2036, 3, 1))
         .add_extension(x509.CRLNumber(number), critical=False)
         .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(issuer.cert.public_key()), critical=False))
    for serial in revoked:
        b = b.add_revoked_certificate(
            x509.RevokedCertificateBuilder().serial_number(serial).revocation_date(REVOKED_AT)
            .add_extension(x509.CRLReason(x509.ReasonFlags.key_compromise), critical=False).build())
    return b.sign(signing_key, hashes.SHA256()).public_bytes(serialization.Encoding.DER)


def _ocsp(pki, party, signing_key, revoked):
    """An OCSP response for `party` naming the intermediate as its responder (by its key's SHA-1), its status revoked
    (keyCompromise, 2026-02-15) or good, signed with `signing_key`; fixed times, as the PKI's own response."""
    ia = pki.intermediate.asn1
    key_hash = hashlib.sha1(ia.public_key["public_key"].contents[1:]).digest()
    cert_id = ocsp.CertId({"hash_algorithm": {"algorithm": "sha1"},
                           "issuer_name_hash": hashlib.sha1(ia.subject.dump()).digest(),
                           "issuer_key_hash": key_hash, "serial_number": party.cert.serial_number})
    status = (ocsp.CertStatus(name="revoked", value=ocsp.RevokedInfo({"revocation_time": REVOKED_AT,
                                                                      "revocation_reason": "key_compromise"}))
              if revoked else ocsp.CertStatus(name="good", value=core.Null()))
    single = ocsp.SingleResponse({"cert_id": cert_id, "cert_status": status, "this_update": T(2026, 3, 2, 9, 0, 0),
                                  "next_update": T(2036, 3, 2)})
    data = ocsp.ResponseData({"responder_id": ocsp.ResponderId(name="by_key", value=key_hash),
                              "produced_at": T(2026, 3, 2, 9, 31, 0), "responses": [single]})
    sig = signing_key.sign(data.dump(), padding.PKCS1v15(), hashes.SHA256())
    basic = ocsp.BasicOCSPResponse({"tbs_response_data": data, "signature_algorithm": {"algorithm": "sha256_rsa"},
                                    "signature": sig})
    return ocsp.OCSPResponse({"response_status": "successful",
                              "response_bytes": {"response_type": "basic_ocsp_response", "response": basic}}).dump()


def _bad(case, step, detail, fails_with=None, passes_with=None):
    return {"case": case, "step": step, "detail": detail, "failsWith": fails_with, "passesWith": passes_with}


def build_bad(seed, pki):
    out = []
    signer_key = pki.signer.key
    bad_ca_key = rsa_key(seed, "bad-ca")
    serial = iter(range(0x3001, 0x3100))

    def path_case(sample, case, desc, signer, chain, step, detail, passes_with=None, differs=None, results=None,
                  extra_files=()):
        """A B-B signature by `signer` with `chain` in its CMS, refused at `step` (the chain, by default)."""
        sd = SignedDoc(sample, seed)
        info, _ = sd.sign("Signature1", [240, 30, 400, 70], "ETSI.CAdES.detached",
                          _pades(pki, signer, label=sample, chain=chain), "Approved", "Head office", WHEN,
                          label=signer.cert.subject.rfc4514_string().split("CN=")[1].split(",")[0])
        sig = _expect(info, pki, signer, "PAdES B-B", "rsa-pkcs1v15-sha256", chain=[p.label for p in chain],
                      results=results or _results(chain="untrusted"))
        sig["knownBad"] = _bad(case, step, detail, passes_with=passes_with)
        if differs:
            sig["referenceDiffers"] = differs
        s = _sample(sd, sample, desc, [sig], pki)
        s.files.update(_files(signer, *chain, *extra_files))
        out.append(s)

    expired = _leaf(pki, "bad-expired", "Example Expired Signer", signer_key, next(serial), end=T(2026, 3, 1))
    path_case("bad-expired", "expired", "Known bad: the signer's certificate expired on 2026-03-01, the day before it "
              "signed (2026-03-02): the chain fails at the signing and the validation time.",
              expired, [pki.intermediate], "chain", "is not valid at")

    early = _leaf(pki, "bad-not-yet-valid", "Example Early Signer", signer_key, next(serial), start=T(2026, 4, 1))
    path_case("bad-not-yet-valid", "not-yet-valid", "Known bad: the signer's certificate is valid only from "
              "2026-04-01, after it signed (2026-03-02) and after the validation time.",
              early, [pki.intermediate], "chain", "is not valid at")

    forged = _leaf(pki, "bad-issuer-signature", "Example Forged Signer", signer_key, next(serial),
                   issuer_key=pki.other_root.key)
    path_case("bad-issuer-signature", "wrong-issuer-signature", "Known bad: the signer's certificate names the "
              "intermediate as its issuer (name and key identifier) but is signed by another key (the other root's).",
              forged, [pki.intermediate], "chain", "does not verify with Example Test Intermediate CA's key")

    not_ca = _ca("bad-not-ca-issuer", "Example Not CA Issuer", bad_ca_key, next(serial), pki.root, ca=False)
    under_not_ca = _leaf(pki, "bad-not-ca", "Example Signer Under A Non-CA", signer_key, next(serial), issuer=not_ca)
    path_case("bad-not-ca", "issuer-not-ca", "Known bad: the signer's issuer (under the root) has basic constraints "
              "with cA false (and keyCertSign): it may not issue certificates.",
              under_not_ca, [not_ca], "chain", "is not a CA")

    sub_ca = _ca("bad-sub-ca", "Example Test Sub CA", bad_ca_key, next(serial), pki.intermediate)
    under_sub = _leaf(pki, "bad-path-length", "Example Signer Too Deep", signer_key, next(serial), issuer=sub_ca)
    path_case("bad-path-length", "path-length", "Known bad: the intermediate's pathLenConstraint is 0, but it issued a "
              "sub-CA, which issued the signer: one CA too many below it.",
              under_sub, [sub_ca, pki.intermediate], "chain", "allows 0 CAs below it")

    no_sign = _ca("bad-no-cert-sign-issuer", "Example No Cert Sign CA", bad_ca_key, next(serial), pki.root,
                  usages=("digital_signature", "crl_sign"))
    under_no_sign = _leaf(pki, "bad-no-cert-sign", "Example Signer Under No Cert Sign", signer_key, next(serial),
                          issuer=no_sign)
    path_case("bad-no-cert-sign", "issuer-without-keycertsign", "Known bad: the signer's issuer is a CA whose key usage "
              "lacks keyCertSign (digitalSignature and cRLSign only).",
              under_no_sign, [no_sign], "chain", "may not sign certificates")

    no_usage = _leaf(pki, "bad-signer-usage", "Example Encipherment Only Signer", signer_key, next(serial),
                     usages=("key_encipherment",))
    path_case("bad-signer-usage", "signer-without-signature-usage", "Known bad: the signer's key usage is "
              "keyEncipherment alone: neither digitalSignature nor nonRepudiation.",
              no_usage, [pki.intermediate], "chain", "may not make signatures")

    critical = _leaf(pki, "bad-critical-extension", "Example Critical Extension Signer", signer_key, next(serial),
                     extra=[(x509.UnrecognizedExtension(ObjectIdentifier(UNKNOWN_EXTENSION), b"\x05\x00"), True)])
    path_case("bad-critical-extension", "unknown-critical-extension", f"Known bad: the signer's certificate carries "
              f"the extension {UNKNOWN_EXTENSION} marked critical, which no check reads.",
              critical, [pki.intermediate], "chain", f"critical extension {UNKNOWN_EXTENSION}")

    misnamed = _leaf(pki, "bad-issuer-name", "Example Misnamed Signer", signer_key, next(serial),
                     issuer_name=name("Example Test Intermediate CA", "Example Organisation Branch"))
    path_case("bad-issuer-name", "names-do-not-chain", "Known bad: the signer's certificate is signed by the "
              "intermediate's key but names another issuer (O=Example Organisation Branch): no certificate's subject "
              "is its issuer.", misnamed, [pki.intermediate], "chain", "is neither trusted nor among")

    loop_b_key = pki.other_signer.key
    loop_a_cert = _cert(name("Example Loop CA A"), bad_ca_key, name("Example Loop CA B"), loop_b_key, next(serial),
                        T(2025, 1, 1), T(2040, 1, 1), ca=True, usages=CA_USAGES)
    loop_b_cert = _cert(name("Example Loop CA B"), loop_b_key, name("Example Loop CA A"), bad_ca_key, next(serial),
                        T(2025, 1, 1), T(2040, 1, 1), ca=True, usages=CA_USAGES)
    loop_a, loop_b = Party("bad-loop-a", bad_ca_key, loop_a_cert), Party("bad-loop-b", loop_b_key, loop_b_cert)
    in_loop = _leaf(pki, "bad-loop", "Example Signer In A Loop", signer_key, next(serial), issuer=loop_a)
    path_case("bad-loop", "loop", "Known bad: two CAs issued each other (A by B, B by A) and neither chains to the "
              "trusted root; the signer is A's. A path search must end, refusing the loop.",
              in_loop, [loop_a, loop_b], "chain", "the path loops")

    chain_key = rsa_key(seed, "chain-ca")
    long_chain, issuer = [], pki.root
    for k in range(1, 13):
        issuer = _ca(f"bad-chain-{k:02d}", f"Example Chain CA {k:02d}", chain_key, next(serial), issuer)
        long_chain.append(issuer)
    deep = _leaf(pki, "bad-long-chain", "Example Signer Fourteen Deep", signer_key, next(serial), issuer=long_chain[-1])
    path_case("bad-long-chain", "chain-too-long", "Known bad: twelve CAs between the root and the signer, fourteen "
              "certificates in all: longer than the 12 a check builds (SignatureCheckOptions.MaxChainLength). pyHanko "
              "builds paths of any length and trusts it.",
              deep, list(reversed(long_chain)), "chain", "a path longer than 12 certificates",
              passes_with={"option": "MaxChainLength", "value": 16},
              differs={"chain": "pyHanko builds a path of any length; Idrak stops at SignatureCheckOptions.MaxChainLength "
                                "(12 certificates by default)"})

    weak = _leaf(pki, "bad-weak-key", "Example Weak Key Signer", rsa_key(seed, "weak", bits=1024), next(serial))
    path_case("bad-weak-key", "weak-rsa-key", "Known bad: the signer's RSA key is 1024 bits, under the 2048 a check "
              "takes (SignatureCheckOptions.MinimumRsaKeyBits; pyHanko's default rule): the signature verifies, the "
              "chain fails naming the key's size (as pyHanko: valid, not trusted).",
              weak, [pki.intermediate], "chain", "RSA key is 1024 bits",
              passes_with={"option": "MinimumRsaKeyBits", "value": 1024})

    # Revocation from the file: B-LT, the DSS holding the given lists and responses.
    def lt_case(sample, case, desc, signer, crls, ocsps, results, detail, fails_with=None):
        sd = SignedDoc(sample, seed)
        info, contents = sd.sign("Signature1", [240, 30, 400, 70], "ETSI.CAdES.detached",
                                 _pades(pki, signer, timestamp=True, label=sample), "Approved", "Head office", WHEN,
                                 label=signer.cert.subject.rfc4514_string().split("CN=")[1].split(",")[0])
        dss = sd.add_dss(pki, [contents], signer=signer, crls=crls, ocsps=ocsps)
        sig = _expect(info, pki, signer, "PAdES B-LT", "rsa-pkcs1v15-sha256", timestamp=TS, results=results,
                      whole=False)
        sig["knownBad"] = _bad(case, "revocation", detail, fails_with=fails_with)
        s = _sample(sd, sample, desc, [sig], pki, extra={"dss": dss})
        s.files.update(_files(signer))
        s.files.update({n: data for n, data in crls + ocsps if not n.startswith("pki.root")})
        out.append(s)

    root_crl = [("pki.root.crl", pki.root_crl)]
    by_ocsp = _leaf(pki, "bad-revoked-ocsp", "Example Signer Revoked By OCSP", signer_key, next(serial))
    lt_case("bad-revoked-ocsp", "revoked-by-ocsp", "Known bad: PAdES B-LT whose DSS holds the root's CRL and an OCSP "
            "response from the intermediate saying the signer was revoked on 2026-02-15 (keyCompromise), before it "
            "signed: the revocation check fails. (The signer revoked by a CRL is pades-revoked.)",
            by_ocsp, root_crl, [("pki.bad-revoked-ocsp.ocsp", _ocsp(pki, by_ocsp, pki.intermediate.key, True))],
            _results(timestamp="valid", revocation="revoked", chain="untrusted", modifications="allowed"),
            "revoked on 2026-02-15")

    crl_forged = _leaf(pki, "bad-crl-wrong-key", "Example Signer With A Forged CRL", signer_key, next(serial))
    lt_case("bad-crl-wrong-key", "crl-wrong-key", "Known bad: PAdES B-LT whose DSS holds a CRL in the intermediate's "
            "name revoking the signer but signed by another key (the other root's), and the root's CRL: the forged list "
            "cannot be used, so the signer's status is not known (soft-fail passes it as not checked; a check that "
            "requires revocation fails).",
            crl_forged, [("pki.bad-crl-wrong-key.crl", _crl(pki.intermediate, pki.other_root.key,
                                                             [crl_forged.cert.serial_number], 9))] + root_crl, [],
            _results(timestamp="valid", revocation="not-checked", modifications="allowed"),
            "list does not verify", fails_with="RequireRevocation")

    ocsp_forged = _leaf(pki, "bad-ocsp-wrong-key", "Example Signer With A Forged OCSP", signer_key, next(serial))
    lt_case("bad-ocsp-wrong-key", "ocsp-wrong-key", "Known bad: PAdES B-LT whose DSS holds an OCSP response naming the "
            "intermediate as its responder and the signer as revoked, but signed by another key (the other root's), and "
            "the root's CRL: the forged response cannot be used (soft-fail passes the signer as not checked; a check "
            "that requires revocation fails).",
            ocsp_forged, root_crl, [("pki.bad-ocsp-wrong-key.ocsp", _ocsp(pki, ocsp_forged, pki.other_root.key, True))],
            _results(timestamp="valid", revocation="not-checked", modifications="allowed"),
            "does not verify", fails_with="RequireRevocation")

    # A timestamp whose token stamps other bytes than the signature's value.
    sd = SignedDoc("bad-timestamp-imprint", seed)

    def make(signed):
        dv = hashlib.sha256(signed).digest()
        return signed_data(pki.signer, None, digest_value=dv, chain=[pki.intermediate], signing_time=None, ess=True,
                           unsigned_for=lambda s: [timestamp_attr(pki, s + b"\x00", T(2026, 3, 2, 9, 30, 5),
                                                                  "bad-timestamp-imprint")]).dump()

    info, _ = sd.sign("Signature1", [240, 30, 400, 70], "ETSI.CAdES.detached", make, "Approved", "Head office", WHEN)
    sig = _expect(info, pki, pki.signer, "PAdES B-T", "rsa-pkcs1v15-sha256", timestamp=TS,
                  results=_results(timestamp="invalid"))
    sig["knownBad"] = _bad("timestamp-imprint", "timestamp", "stamps other bytes than the signature's value")
    out.append(_sample(sd, "bad-timestamp-imprint", "Known bad: PAdES B-T whose timestamp token (from the test "
                       "authority, its own signature sound) stamps the digest of other bytes than the signature's "
                       "value: the timestamp fails, and the signature's time is the one it claims.", [sig], pki))
    return out
