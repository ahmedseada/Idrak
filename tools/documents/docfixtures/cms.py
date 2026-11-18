"""CMS SignedData (RFC 5652) for the D12 samples, written with asn1crypto: the signed attributes in DER order (each
attribute's encoding sorted, as SET OF asks), the signature over their SET encoding, unsigned attributes made from the
signature value (a signature timestamp token). Signature kinds: RSA PKCS #1 v1.5, RSA-PSS (SHA-256, MGF1-SHA-256, a
32-byte salt from the seed), ECDSA (deterministic, RFC 6979); every byte repeats on every run."""
import hashlib

from asn1crypto import algos, cms, core, tsp, x509 as ax509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, padding

HASHES = {"sha256": (hashlib.sha256, hashes.SHA256), "sha1": (hashlib.sha1, hashes.SHA1)}


def ess_v2(party):
    a = party.asn1
    return cms.CMSAttribute({"type": "signing_certificate_v2", "values": [tsp.SigningCertificateV2({"certs": [
        tsp.ESSCertIDv2({"hash_algorithm": {"algorithm": "sha256"},
                         "cert_hash": hashlib.sha256(party.der).digest(),
                         "issuer_serial": {"issuer": [ax509.GeneralName({"directory_name": a.issuer})],
                                           "serial_number": a.serial_number}})]})]})


def _sorted_attrs(attrs):
    return cms.CMSAttributes(sorted(attrs, key=lambda at: at.dump()))


def signed_data(signer, content, content_type="data", embed=False, chain=(), digest="sha256", algorithm="rsa",
                signing_time=None, ess=True, digest_value=None, pss_salt=None, unsigned_for=None, extra_certs=()):
    """A ContentInfo holding SignedData. `content` is hashed (and embedded with `embed`) unless `digest_value` gives
    the digest already (a PDF's byte range); `unsigned_for(signature) -> [CMSAttribute]` adds unsigned attributes."""
    h, H = HASHES[digest]
    md = digest_value if digest_value is not None else h(content).digest()
    attrs = [cms.CMSAttribute({"type": "content_type", "values": [content_type]}),
             cms.CMSAttribute({"type": "message_digest", "values": [md]})]
    if signing_time is not None:
        attrs.append(cms.CMSAttribute({"type": "signing_time", "values": [cms.Time({"utc_time": signing_time})]}))
    if ess:
        attrs.append(ess_v2(signer))
    signed_attrs = _sorted_attrs(attrs)
    to_sign = signed_attrs.dump()
    key = signer.key
    if algorithm == "rsa":
        signature = key.sign(to_sign, padding.PKCS1v15(), H())
        sig_alg = {"algorithm": "rsassa_pkcs1v15"}
    elif algorithm == "pss":
        from .pki import pss_sign

        signature = pss_sign(key, to_sign, pss_salt)
        sig_alg = {"algorithm": "rsassa_pss", "parameters": algos.RSASSAPSSParams({
            "hash_algorithm": {"algorithm": "sha256"},
            "mask_gen_algorithm": {"algorithm": "mgf1", "parameters": {"algorithm": "sha256"}},
            "salt_length": len(pss_salt), "trailer_field": "trailer_field_bc"})}
    elif algorithm == "ecdsa":
        signature = key.sign(to_sign, ec.ECDSA(H(), deterministic_signing=True))
        sig_alg = {"algorithm": f"{digest}_ecdsa"}
    else:
        raise ValueError(algorithm)
    info = {"version": "v1",
            "sid": cms.SignerIdentifier({"issuer_and_serial_number": cms.IssuerAndSerialNumber({
                "issuer": signer.asn1.issuer, "serial_number": signer.asn1.serial_number})}),
            "digest_algorithm": {"algorithm": digest}, "signed_attrs": signed_attrs,
            "signature_algorithm": sig_alg, "signature": signature}
    if unsigned_for is not None:
        info["unsigned_attrs"] = cms.CMSAttributes(unsigned_for(signature))
    encap = {"content_type": content_type}
    if embed:
        # asn1crypto's type for the content follows its type: data an OCTET STRING, a timestamp's TSTInfo parsed.
        encap["content"] = tsp.TSTInfo.load(content) if content_type == "tst_info" else core.OctetString(content)
    sd = cms.SignedData({"version": "v3" if content_type != "data" else "v1",
                         "digest_algorithms": [{"algorithm": digest}], "encap_content_info": encap,
                         "certificates": [cms.CertificateChoices({"certificate": p.asn1})
                                          for p in [signer, *chain, *extra_certs]],
                         "signer_infos": [cms.SignerInfo(info)]})
    return cms.ContentInfo({"content_type": "signed_data", "content": sd})


def timestamp_attr(pki, signature, gen_time, label):
    """The signature timestamp (id-aa-signatureTimeStampToken) over a signature value, from the in-process TSA."""
    req = pki.timestamp_request(hashlib.sha256(signature).digest(), label)
    resp = tsp.TimeStampResp.load(pki.timestamp(req, gen_time))
    return cms.CMSAttribute({"type": "signature_time_stamp_token", "values": [resp["time_stamp_token"]]})

