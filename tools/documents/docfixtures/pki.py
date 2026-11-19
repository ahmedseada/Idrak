"""The D12 test PKI, made here and the same on every run: never a real certificate, every name generic ("Example
Test Root CA"). A root CA, an intermediate CA, a signer (RSA), a signer with an EC key (P-256), a signer whose
certificate is revoked, a timestamp authority, and an untrusted root with its own signer (the wrong chain); the
intermediate's and the root's CRLs, an OCSP response for the signer, and the timestamp authority itself (in process:
a TimeStampReq in, a TimeStampResp out, RFC 3161).

What cannot be left random is made from the seed:
- RSA keys: primes drawn from a seeded generator (random.Random over SHA-256 of the seed and the key's label) and
  tested by Miller-Rabin with seeded bases; the EC key: the scalar from SHA-256 of the seed and its label;
- certificates, CRLs: cryptography's builders with fixed serials and times, signed with RSA PKCS #1 v1.5 (which needs
  no randomness);
- the OCSP response and the timestamp tokens: written with asn1crypto (cryptography's OCSP builder stamps the current
  time), fixed times and serials, the request nonce from the seed;
- RSA-PSS signatures: the salt from the seed (EMSA-PSS encoded here, RSA applied with the private exponent), and
  ECDSA signatures in deterministic mode (RFC 6979, cryptography's `deterministic_signing`)."""
import datetime
import hashlib
import random

from asn1crypto import core, ocsp, tsp, x509 as ax509
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

UTC = datetime.timezone.utc
TEST_POLICY = "2.999.1.2"  # the ITU-T/ISO example arc (2.999): no real policy


def T(*args):
    return datetime.datetime(*args, tzinfo=UTC)


def _rng(seed, label):
    return random.Random(hashlib.sha256(f"{seed}|pki|{label}".encode("utf-8")).digest())


SMALL_PRIMES = [p for p in range(3, 2000) if all(p % d for d in range(2, int(p ** 0.5) + 1))]


def _probable_prime(n, rnd, rounds=40):
    if any(n % p == 0 for p in SMALL_PRIMES):
        return n in SMALL_PRIMES
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for _ in range(rounds):
        a = rnd.randrange(2, n - 2)
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _prime(rnd, bits, e=65537):
    while True:
        c = rnd.getrandbits(bits) | (3 << (bits - 2)) | 1
        if (c - 1) % e and _probable_prime(c, rnd):
            return c


def rsa_key(seed, label, bits=2048):
    rnd = _rng(seed, label)
    e = 65537
    while True:
        p, q = _prime(rnd, bits // 2), _prime(rnd, bits // 2)
        if p != q and (p * q).bit_length() == bits:
            break
    if p < q:
        p, q = q, p
    d = pow(e, -1, (p - 1) * (q - 1))
    numbers = rsa.RSAPrivateNumbers(p, q, d, d % (p - 1), d % (q - 1), pow(q, -1, p),
                                    rsa.RSAPublicNumbers(e, p * q))
    return numbers.private_key()


def ec_key(seed, label):
    k = int.from_bytes(hashlib.sha256(f"{seed}|pki|{label}".encode("utf-8")).digest(), "big")
    order = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
    return ec.derive_private_key(k % (order - 1) + 1, ec.SECP256R1())


def name(cn, org="Example Organisation"):
    return x509.Name([x509.NameAttribute(NameOID.COUNTRY_NAME, "GB"),
                      x509.NameAttribute(NameOID.ORGANIZATION_NAME, org),
                      x509.NameAttribute(NameOID.COMMON_NAME, cn)])


class Party:
    def __init__(self, label, key, cert):
        self.label, self.key, self.cert = label, key, cert

    @property
    def der(self):
        return self.cert.public_bytes(serialization.Encoding.DER)

    @property
    def asn1(self):
        return ax509.Certificate.load(self.der)

    def describe(self):
        c = self.cert
        return {"file": f"pki.{self.label}.cer", "subject": c.subject.rfc4514_string(),
                "issuer": c.issuer.rfc4514_string(), "serial": c.serial_number,
                "notBefore": c.not_valid_before_utc.isoformat(), "notAfter": c.not_valid_after_utc.isoformat(),
                "sha256": hashlib.sha256(self.der).hexdigest(),
                "key": "EC P-256" if isinstance(self.key, ec.EllipticCurvePrivateKey) else
                f"RSA {self.key.key_size}"}


def _cert(subject, subject_key, issuer_name, issuer_key, serial, start, end, ca=False, path_len=None, usages=(),
          eku=None, issuer_cert=None, crl_url=None, ocsp_url=None, extra=()):
    """A certificate from cryptography's builder, signed with `issuer_key` (RSA PKCS #1 v1.5, SHA-256); `extra` more
    extensions as (extension, critical) pairs, added last (the known-bad samples' unknown critical one)."""
    pub = subject_key.public_key()
    b = (x509.CertificateBuilder().subject_name(subject).issuer_name(issuer_name).public_key(pub)
         .serial_number(serial).not_valid_before(start).not_valid_after(end)
         .add_extension(x509.BasicConstraints(ca=ca, path_length=path_len), critical=True)
         .add_extension(x509.SubjectKeyIdentifier.from_public_key(pub), critical=False))
    ku = {k: k in usages for k in ("digital_signature", "content_commitment", "key_encipherment",
                                    "data_encipherment", "key_agreement", "key_cert_sign", "crl_sign")}
    b = b.add_extension(x509.KeyUsage(encipher_only=False, decipher_only=False, **ku), critical=True)
    if issuer_cert is not None:
        b = b.add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(issuer_cert.public_key()),
                            critical=False)
    if eku:
        b = b.add_extension(x509.ExtendedKeyUsage(eku), critical=True)
    if crl_url:
        b = b.add_extension(x509.CRLDistributionPoints([x509.DistributionPoint(
            [x509.UniformResourceIdentifier(crl_url)], None, None, None)]), critical=False)
    if ocsp_url:
        b = b.add_extension(x509.AuthorityInformationAccess([x509.AccessDescription(
            x509.oid.AuthorityInformationAccessOID.OCSP, x509.UniformResourceIdentifier(ocsp_url))]), critical=False)
    for extension, critical in extra:
        b = b.add_extension(extension, critical=critical)
    return b.sign(issuer_key, hashes.SHA256())


class TestPKI:
    """Every party and every revocation answer, from the seed."""

    SIGNING_TIME = T(2026, 3, 2, 9, 30, 0)
    VALIDATION_TIME = T(2026, 3, 9, 12, 0, 0)

    def __init__(self, seed):
        self.seed = seed
        root_key = rsa_key(seed, "root")
        root_name = name("Example Test Root CA")
        root = _cert(root_name, root_key, root_name, root_key, 0x1001, T(2025, 1, 1), T(2045, 1, 1), ca=True,
                     usages=("key_cert_sign", "crl_sign"))
        self.root = Party("root", root_key, root)
        ikey = rsa_key(seed, "intermediate")
        inter = _cert(name("Example Test Intermediate CA"), ikey, root_name, root_key, 0x1002, T(2025, 1, 1),
                      T(2040, 1, 1), ca=True, path_len=0, usages=("key_cert_sign", "crl_sign"), issuer_cert=root,
                      crl_url="http://pki.example.com/root.crl")
        self.intermediate = Party("intermediate", ikey, inter)
        iname = inter.subject

        def leaf(label, cn, key, serial, eku=None):
            usages = ("digital_signature", "content_commitment")
            cert = _cert(name(cn), key, iname, ikey, serial, T(2025, 6, 1), T(2035, 6, 1), usages=usages, eku=eku,
                         issuer_cert=inter, crl_url="http://pki.example.com/intermediate.crl",
                         ocsp_url="http://ocsp.example.com")
            return Party(label, key, cert)

        self.signer = leaf("signer", "Example Test Signer", rsa_key(seed, "signer"), 0x1003)
        self.signer_ec = leaf("signer-ec", "Example Test Signer EC", ec_key(seed, "signer-ec"), 0x1004)
        self.revoked = leaf("revoked", "Example Revoked Signer", rsa_key(seed, "revoked"), 0x1005)
        self.tsa = leaf("tsa", "Example Test Timestamp Authority", rsa_key(seed, "tsa"), 0x1006,
                        eku=[ExtendedKeyUsageOID.TIME_STAMPING])
        okey = rsa_key(seed, "other-root")
        oname = name("Example Other Root CA", "Example Other Organisation")
        other = _cert(oname, okey, oname, okey, 0x2001, T(2025, 1, 1), T(2045, 1, 1), ca=True,
                      usages=("key_cert_sign", "crl_sign"))
        self.other_root = Party("other-root", okey, other)
        skey = rsa_key(seed, "other-signer")
        ocert = _cert(name("Example Untrusted Signer", "Example Other Organisation"), skey, oname, okey, 0x2002,
                      T(2025, 6, 1), T(2035, 6, 1), usages=("digital_signature", "content_commitment"),
                      issuer_cert=other)
        self.other_signer = Party("other-signer", skey, ocert)
        self.root_crl = self._crl(self.root, [], T(2026, 3, 1), T(2036, 3, 1), 1)
        self.intermediate_crl = self._crl(self.intermediate, [(0x1005, T(2026, 2, 1))], T(2026, 3, 1),
                                          T(2036, 3, 1), 7)
        self.signer_ocsp = self.ocsp_response(self.signer, T(2026, 3, 2, 9, 0, 0), T(2026, 3, 2, 9, 31, 0))
        self.tst_serial = 0x5000

    def _crl(self, issuer, revoked, this_update, next_update, number):
        b = (x509.CertificateRevocationListBuilder().issuer_name(issuer.cert.subject).last_update(this_update)
             .next_update(next_update)
             .add_extension(x509.CRLNumber(number), critical=False)
             .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(issuer.cert.public_key()),
                            critical=False))
        for serial, when in revoked:
            b = b.add_revoked_certificate(
                x509.RevokedCertificateBuilder().serial_number(serial).revocation_date(when)
                .add_extension(x509.CRLReason(x509.ReasonFlags.key_compromise), critical=False).build())
        return b.sign(issuer.key, hashes.SHA256()).public_bytes(serialization.Encoding.DER)

    def ocsp_response(self, party, this_update, produced_at):
        """A good-status OCSP response for `party`, signed by its issuer (the intermediate) directly."""
        issuer = self.intermediate
        ia = issuer.asn1
        cert_id = ocsp.CertId({
            "hash_algorithm": {"algorithm": "sha1"},
            "issuer_name_hash": hashlib.sha1(ia.subject.dump()).digest(),
            "issuer_key_hash": hashlib.sha1(ia.public_key["public_key"].contents[1:]).digest(),
            "serial_number": party.cert.serial_number})
        single = ocsp.SingleResponse({"cert_id": cert_id, "cert_status": ocsp.CertStatus(name="good",
                                                                                          value=core.Null()),
                                      "this_update": this_update, "next_update": T(2036, 3, 2)})
        data = ocsp.ResponseData({"responder_id": ocsp.ResponderId(name="by_key", value=hashlib.sha1(
            ia.public_key["public_key"].contents[1:]).digest()), "produced_at": produced_at, "responses": [single]})
        sig = issuer.key.sign(data.dump(), padding.PKCS1v15(), hashes.SHA256())
        basic = ocsp.BasicOCSPResponse({"tbs_response_data": data, "signature_algorithm": {"algorithm": "sha256_rsa"},
                                        "signature": sig})
        return ocsp.OCSPResponse({"response_status": "successful",
                                  "response_bytes": {"response_type": "basic_ocsp_response", "response": basic}}).dump()

    # -- the timestamp authority -----------------------------------------------------------------------------------------

    def timestamp_request(self, imprint, label):
        nonce = int.from_bytes(hashlib.sha256(f"{self.seed}|nonce|{label}".encode()).digest()[:8], "big")
        return tsp.TimeStampReq({"version": "v1", "message_imprint": {"hash_algorithm": {"algorithm": "sha256"},
                                                                      "hashed_message": imprint},
                                 "nonce": nonce, "cert_req": True}).dump()

    def timestamp(self, request_der, gen_time):
        """The in-process timestamp authority: a TimeStampReq's DER in, a granted TimeStampResp's DER out."""
        from .cms import signed_data

        req = tsp.TimeStampReq.load(request_der)
        self.tst_serial += 1
        info = tsp.TSTInfo({"version": "v1", "policy": TEST_POLICY, "message_imprint": req["message_imprint"],
                            "serial_number": self.tst_serial, "gen_time": gen_time, "nonce": req["nonce"].native})
        token = signed_data(self.tsa, info.dump(), content_type="tst_info", embed=True,
                            chain=[self.intermediate], signing_time=None)
        resp = tsp.TimeStampResp({"status": {"status": "granted"}, "time_stamp_token": token})
        return resp.dump()


def pss_sign(key, data, salt):
    """RSASSA-PSS with SHA-256 and MGF1-SHA-256, the salt given (EMSA-PSS encoding, RFC 8017 9.1.1), so the
    signature is the same on every run."""
    n = key.private_numbers().public_numbers.n
    d = key.private_numbers().d
    em_bits = n.bit_length() - 1
    em_len = (em_bits + 7) // 8
    m_hash = hashlib.sha256(data).digest()
    h = hashlib.sha256(b"\x00" * 8 + m_hash + salt).digest()
    ps = b"\x00" * (em_len - len(salt) - 32 - 2)
    db = ps + b"\x01" + salt
    mask = b""
    counter = 0
    while len(mask) < len(db):
        mask += hashlib.sha256(h + counter.to_bytes(4, "big")).digest()
        counter += 1
    masked = bytes(a ^ b for a, b in zip(db, mask))
    masked = bytes([masked[0] & (0xFF >> (8 * em_len - em_bits))]) + masked[1:]
    em = masked + h + b"\xbc"
    sig = pow(int.from_bytes(em, "big"), d, n).to_bytes((n.bit_length() + 7) // 8, "big")
    key.public_key().verify(sig, data, padding.PSS(padding.MGF1(hashes.SHA256()), len(salt)), hashes.SHA256())
    return sig


def pem_files(pki):
    """The PKI as files beside the samples: certificates (DER .cer and PEM), private keys (unencrypted PKCS #8 PEM,
    throwaway test keys), the CRLs and the OCSP response."""
    files = {}
    parties = [pki.root, pki.intermediate, pki.signer, pki.signer_ec, pki.revoked, pki.tsa, pki.other_root,
               pki.other_signer]
    for p in parties:
        files[f"pki.{p.label}.cer"] = p.der
        files[f"pki.{p.label}.pem"] = p.cert.public_bytes(serialization.Encoding.PEM)
        files[f"pki.{p.label}.key.pem"] = p.key.private_bytes(serialization.Encoding.PEM,
                                                              serialization.PrivateFormat.PKCS8,
                                                              serialization.NoEncryption())
    files["pki.root.crl"] = pki.root_crl
    files["pki.intermediate.crl"] = pki.intermediate_crl
    files["pki.signer.ocsp"] = pki.signer_ocsp
    return files, parties

