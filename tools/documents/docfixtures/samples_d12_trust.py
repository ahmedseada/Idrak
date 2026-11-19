"""D12's trust samples (owner, 2026-11-19, signature trust completed): signed samples whose certificate path carries name
constraints or certificate policies, or whose revocation data is a delta list, an indirect list or stale, made with the
test PKI's tooling (pki.py: every key and byte from the seed, never a real certificate, every name and address generic:
example.com, example.org, the documentation address ranges 192.0.2.0/24, 198.51.100.0/24, 2001:db8::/32 and 3fff::/20,
policies under the ITU-T/ISO example arc 2.999). They are D12 samples like the others (`signatures`, `results`,
pyHanko's verdicts), the chain in each CMS (and DSS), the root alone trusted. A sample a check must refuse carries
`knownBad` as samples_d12_bad.py's do; here `failsWith` may also be an object of check options (a policy the app asks
for), with which a path valid by default fails. A policy sample's `policies` lists the policies a check with the default
options finds the path valid for.

- Name constraints (RFC 5280 4.2.1.10) in three CAs under the root: one permitting and excluding a directoryName, an
  rfc822Name, a dNSName and a URI; one permitting and excluding IPv4 and IPv6 addresses; one excluding an otherName (a
  form a check does not compare). A signer within each (the first with names in other case, which the RFC compares
  case-insensitively, and a mailbox whose local part differs from an excluded one only in case, which it does not); a
  signer excluded and one not permitted for each form; a signer with an otherName under its constraint.
- Policies (RFC 5280 6.1): a path for one policy; a CA requiring an explicit policy over a signer with none; a CA mapping
  its policy to the signer's; a CA inhibiting mappings above one that maps; a CA asserting anyPolicy; a CA inhibiting
  anyPolicy above one that asserts it.
- Revocation lists (B-LT, the DSS holding them): a complete list and its delta (deltaCRLIndicator, the base's
  freshestCRL), the delta revoking the signer, or releasing a hold the base list puts on it (removeFromCRL); an indirect
  list (its issuing distribution point's indirectCRL) from a CRL issuer the signer's distribution point names (cRLIssuer),
  its entry naming the signer's issuer (certificateIssuer), revoking the signer or not.
- Freshness: B-LT whose OCSP response and intermediate's list had their next update before the signing (stale at the
  timestamp's and the validation time): not checked (stale), refused when revocation is required, good with a grace of 14
  days (`FreshnessGrace`)."""
import ipaddress

from asn1crypto import x509 as ax509
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.x509.oid import NameOID, ObjectIdentifier

from .pki import Party, T, _cert, name, rsa_key
from .samples_d12 import TS, SignedDoc, _expect, _pades, _results, _sample
from .samples_d12_bad import CA_USAGES, LEAF_USAGES, WHEN, _bad, _ocsp

POLICY_A, POLICY_B = "2.999.2.1", "2.999.2.2"
OTHER_NAME = "2.999.1.10"  # an otherName type under the example arc


def _dn(*parts):
    """A name in its order (country first), from (attribute, value) pairs."""
    oids = {"C": NameOID.COUNTRY_NAME, "O": NameOID.ORGANIZATION_NAME, "OU": NameOID.ORGANIZATIONAL_UNIT_NAME,
            "CN": NameOID.COMMON_NAME}
    return x509.Name([x509.NameAttribute(oids[k], v) for k, v in parts])


def _files(*parties):
    """The new parties' certificates beside the sample (DER), as the PKI's are."""
    return {f"pki.{p.label}.cer": p.der for p in parties if p.label.startswith("trust-")}


def _policies(*oids):
    return x509.CertificatePolicies([x509.PolicyInformation(ObjectIdentifier(o), None) for o in oids])


def _mapping(issuer, subject):
    der = ax509.PolicyMappings([{"issuer_domain_policy": issuer, "subject_domain_policy": subject}]).dump()
    return x509.UnrecognizedExtension(ObjectIdentifier("2.5.29.33"), der)


def build_trust(seed, pki):
    out = []
    signer_key = pki.signer.key
    ca_key = rsa_key(seed, "trust-ca")
    serial = iter(range(0x4001, 0x4100))

    def ca(label, cn, issuer=None, extra=(), path_len=None):
        issuer = issuer or pki.root
        cert = _cert(name(cn), ca_key, issuer.cert.subject, issuer.key, next(serial), T(2025, 1, 1), T(2040, 1, 1), ca=True,
                     path_len=path_len, usages=CA_USAGES, issuer_cert=issuer.cert, extra=extra)
        return Party(label, ca_key, cert)

    def leaf(label, subject, issuer, extra=(), crl_url=None, ocsp_url=None):
        cert = _cert(subject, signer_key, issuer.cert.subject, issuer.key, next(serial), T(2025, 6, 1), T(2035, 6, 1),
                     usages=LEAF_USAGES, issuer_cert=issuer.cert, crl_url=crl_url, ocsp_url=ocsp_url, extra=extra)
        return Party(label, signer_key, cert)

    def san(*names):
        return [(x509.SubjectAlternativeName(list(names)), False)]

    def signed(sample, signer, chain, desc, results, policies=None, bad=None, differs=None):
        """A B-B signature by `signer` with `chain` in its CMS, checked as `results` say (with the root alone trusted)."""
        sd = SignedDoc(sample, seed)
        info, _ = sd.sign("Signature1", [240, 30, 400, 70], "ETSI.CAdES.detached",
                          _pades(pki, signer, label=sample, chain=chain), "Approved", "Head office", WHEN,
                          label=signer.cert.subject.rfc4514_string().split("CN=")[1].split(",")[0])
        sig = _expect(info, pki, signer, "PAdES B-B", "rsa-pkcs1v15-sha256", chain=[p.label for p in chain],
                      results=results)
        if policies is not None:
            sig["policies"] = policies
        if bad is not None:
            sig["knownBad"] = bad
        if differs:
            sig["referenceDiffers"] = differs
        s = _sample(sd, sample, desc, [sig], pki)
        s.files.update(_files(signer, *chain))
        out.append(s)

    trusted, untrusted = _results(), _results(chain="untrusted")

    # -- name constraints ------------------------------------------------------------------------------------------------
    org = (("C", "GB"), ("O", "Example Organisation"))
    names_ca = ca("trust-nc-names-ca", "Example Constrained CA", extra=[(x509.NameConstraints(
        permitted_subtrees=[x509.DirectoryName(_dn(*org)), x509.RFC822Name("example.com"), x509.DNSName("example.com"),
                            x509.UniformResourceIdentifier(".example.com")],
        excluded_subtrees=[x509.DirectoryName(_dn(*org, ("OU", "Example Restricted Unit"))),
                           x509.RFC822Name("restricted@example.com"), x509.DNSName("restricted.example.com"),
                           x509.UniformResourceIdentifier("restricted.example.com")]), True)])
    signer_dn = _dn(*org, ("CN", "Example Constrained Signer"))
    valid = leaf("trust-nc-names-valid", _dn(("C", "GB"), ("O", "example  ORGANISATION"), ("CN", "Example Constrained Signer")),
                 names_ca, extra=san(x509.RFC822Name("signer@Example.COM"), x509.RFC822Name("Restricted@example.com"),
                                     x509.DNSName("Signer.Example.com"), x509.UniformResourceIdentifier("https://WWW.example.com/signer")))
    signed("trust-nc-names-valid", valid, [names_ca], "Name constraints: the CA permits C=GB, O=Example Organisation, the "
           "mailboxes at example.com, the hosts under example.com (dNSName) and below it (a URI's host), and excludes "
           "the OU Example Restricted Unit, restricted@example.com, restricted.example.com; the signer's subject and names "
           "are within it in other case (example  ORGANISATION, Signer.Example.com, signer@Example.COM, WWW.example.com), "
           "which the RFC compares case-insensitively, and its mailbox Restricted@example.com differs from the excluded "
           "one only in its local part's case, which the RFC compares exactly: valid.", trusted)
    cases = [
        ("dn", "excluded", _dn(*org, ("OU", "Example Restricted Unit"), ("CN", "Example Restricted Unit Signer")),
         [x509.DNSName("signer.example.com")], "the subject's OU is Example Restricted Unit, an excluded directory subtree",
         "which Example Constrained CA's name constraints exclude"),
        ("dn", "not-permitted", _dn(("C", "GB"), ("O", "Example Other Organisation"), ("CN", "Example Outside Signer")),
         [x509.DNSName("signer.example.com")], "the subject's O is Example Other Organisation, outside the permitted "
         "directory subtree", "is outside the names Example Constrained CA's name constraints permit"),
        ("email", "excluded", signer_dn, [x509.RFC822Name("restricted@example.com")], "its mailbox is "
         "restricted@example.com, an excluded mailbox", "which Example Constrained CA's name constraints exclude"),
        ("email", "not-permitted", signer_dn, [x509.RFC822Name("signer@example.org")], "its mailbox is at example.org, "
         "not the permitted host example.com", "is outside the names Example Constrained CA's name constraints permit"),
        ("dns", "excluded", signer_dn, [x509.DNSName("host.restricted.example.com")], "its dNSName "
         "host.restricted.example.com is under the excluded restricted.example.com",
         "which Example Constrained CA's name constraints exclude"),
        ("dns", "not-permitted", signer_dn, [x509.DNSName("signer.example.org")], "its dNSName is under example.org, "
         "not the permitted example.com", "is outside the names Example Constrained CA's name constraints permit"),
        ("uri", "excluded", signer_dn, [x509.UniformResourceIdentifier("https://restricted.example.com/signer")],
         "its URI's host is restricted.example.com, an excluded host", "which Example Constrained CA's name constraints exclude"),
        ("uri", "not-permitted", signer_dn, [x509.UniformResourceIdentifier("https://www.example.org/signer")],
         "its URI's host is under example.org, not below the permitted .example.com",
         "is outside the names Example Constrained CA's name constraints permit"),
    ]
    for form, how, subject, names, why, detail in cases:
        sample = f"trust-nc-{form}-{how}"
        signer = leaf(sample, subject, names_ca, extra=san(*names))
        signed(sample, signer, [names_ca], f"Known bad: name constraints ({form}): {why}; the chain fails.", untrusted,
               bad=_bad(f"name-constraints-{form}-{how}", "chain", detail))

    address_ca = ca("trust-nc-address-ca", "Example Address Constrained CA", extra=[(x509.NameConstraints(
        permitted_subtrees=[x509.IPAddress(ipaddress.ip_network("192.0.2.0/24")),
                            x509.IPAddress(ipaddress.ip_network("2001:db8::/32"))],
        excluded_subtrees=[x509.IPAddress(ipaddress.ip_network("192.0.2.128/25")),
                           x509.IPAddress(ipaddress.ip_network("2001:db8:ffff::/48"))]), True)])
    no_ip = {"chain": "pyHanko does not evaluate iPAddress name constraints (pyhanko_certvalidator's name_trees has no "
                      "checker for them): it stops with NotImplementedError on a path whose addresses meet one"}
    valid = leaf("trust-nc-ip-valid", name("Example Address Signer"), address_ca,
                 extra=san(x509.IPAddress(ipaddress.ip_address("192.0.2.10")), x509.IPAddress(ipaddress.ip_address("2001:db8::10"))))
    signed("trust-nc-ip-valid", valid, [address_ca], "Name constraints (iPAddress): the CA permits 192.0.2.0/24 and "
           "2001:db8::/32 and excludes 192.0.2.128/25 and 2001:db8:ffff::/48; the signer's addresses 192.0.2.10 and "
           "2001:db8::10 are within: valid.", trusted, differs=no_ip)
    for family, how, address, why, detail in [
        ("ipv4", "excluded", "192.0.2.200", "192.0.2.200 is in the excluded 192.0.2.128/25",
         "which Example Address Constrained CA's name constraints exclude"),
        ("ipv4", "not-permitted", "198.51.100.10", "198.51.100.10 is outside the permitted 192.0.2.0/24",
         "is outside the names Example Address Constrained CA's name constraints permit"),
        ("ipv6", "excluded", "2001:db8:ffff::1", "2001:db8:ffff::1 is in the excluded 2001:db8:ffff::/48",
         "which Example Address Constrained CA's name constraints exclude"),
        ("ipv6", "not-permitted", "3fff::1", "3fff::1 is outside the permitted 2001:db8::/32",
         "is outside the names Example Address Constrained CA's name constraints permit"),
    ]:
        sample = f"trust-nc-{family}-{how}"
        signer = leaf(sample, name("Example Address Signer"), address_ca,
                      extra=san(x509.IPAddress(ipaddress.ip_address(address))))
        signed(sample, signer, [address_ca], f"Known bad: name constraints ({family}): the signer's address {why}; the "
               "chain fails.", untrusted, bad=_bad(f"name-constraints-{family}-{how}", "chain", detail), differs=no_ip)

    other_ca = ca("trust-nc-other-ca", "Example Other Name Constrained CA", extra=[(x509.NameConstraints(
        permitted_subtrees=[x509.DNSName("example.com")],
        excluded_subtrees=[x509.OtherName(ObjectIdentifier(OTHER_NAME), b"\x0c\x0aRestricted")]), True)])
    no_other = {"chain": "pyHanko cannot read an otherName name constraint (pyhanko_certvalidator hashes its value, "
                         "which it cannot: TypeError), whether or not a name below holds one"}
    plain = leaf("trust-nc-othername-unused", name("Example Plain Signer"), other_ca,
                 extra=san(x509.DNSName("signer.example.com")))
    signed("trust-nc-othername-unused", plain, [other_ca], "Name constraints: the CA excludes an otherName (a form a "
           "check does not compare) and permits example.com; the signer holds no otherName, so the constraint asks "
           "nothing of it: valid.", trusted, differs=no_other)
    other = leaf("trust-nc-othername", name("Example Other Name Signer"), other_ca,
                 extra=san(x509.DNSName("signer.example.com"), x509.OtherName(ObjectIdentifier(OTHER_NAME), b"\x0c\x05Plain")))
    signed("trust-nc-othername", other, [other_ca], "Known bad: name constraints on otherName names, a form a check "
           "does not compare, and a signer holding one: the chain fails, naming the form (RFC 5280: a constraint on a "
           "form a validator cannot process refuses a certificate holding a name of it).", untrusted,
           bad=_bad("name-constraints-unsupported-form", "chain", "limit otherName names, which this check does not evaluate"),
           differs=no_other)

    # -- policies --------------------------------------------------------------------------------------------------------
    def pol_leaf(label, cn, issuer, *policies):
        return leaf(label, name(cn), issuer, extra=[(_policies(*policies), False)] if policies else ())

    policy_ca = ca("trust-policy-ca", "Example Policy CA", extra=[(_policies(POLICY_A), False)])
    signer = pol_leaf("trust-policy-valid", "Example Policy Signer", policy_ca, POLICY_A)
    signed("trust-policy-valid", signer, [policy_ca], f"Policies: the CA and the signer assert the policy {POLICY_A}: "
           f"valid for it. A check that requires the policy {POLICY_B} explicitly refuses the path.", trusted,
           policies=[POLICY_A], bad=_bad("policy-not-the-initial", "chain", f"valid for none of the policies {POLICY_B}",
                                         fails_with={"InitialPolicies": [POLICY_B], "RequireExplicitPolicy": True}))

    explicit_ca = ca("trust-policy-explicit-ca", "Example Explicit Policy CA", extra=[
        (_policies("2.5.29.32.0"), False), (x509.PolicyConstraints(require_explicit_policy=0, inhibit_policy_mapping=None), True)])
    signer = pol_leaf("trust-policy-explicit", "Example Policy-Free Signer", explicit_ca)
    signed("trust-policy-explicit", signer, [explicit_ca], "Known bad: the CA's policy constraints require an explicit "
           "policy (requireExplicitPolicy 0) and the signer asserts no policy: the chain fails.", untrusted,
           bad=_bad("policy-explicit-required", "chain", "leaves the path no valid certificate policy"))

    mapping_ca = ca("trust-policy-mapping-ca", "Example Mapping CA",
                    extra=[(_policies(POLICY_A), False), (_mapping(POLICY_A, POLICY_B), True)])
    signer = pol_leaf("trust-policy-mapping", "Example Mapped Signer", mapping_ca, POLICY_B)
    signed("trust-policy-mapping", signer, [mapping_ca], f"Policies: the CA asserts {POLICY_A} and maps it to {POLICY_B}, "
           f"the signer asserts {POLICY_B}: valid for {POLICY_A} (the trusted side's name). A check inhibiting policy "
           "mapping and requiring an explicit policy refuses the path.", trusted, policies=[POLICY_A],
           bad=_bad("policy-mapping-inhibited-by-the-check", "chain", "leaves the path no valid certificate policy",
                    fails_with={"InhibitPolicyMapping": True, "RequireExplicitPolicy": True}))

    inhibit_ca = ca("trust-policy-inhibit-mapping-ca", "Example Inhibit Mapping CA", extra=[
        (_policies("2.5.29.32.0"), False), (x509.PolicyConstraints(require_explicit_policy=0, inhibit_policy_mapping=0), True)])
    mapping_sub = ca("trust-policy-mapping-sub-ca", "Example Mapping Sub CA", issuer=inhibit_ca,
                     extra=[(_policies(POLICY_A), False), (_mapping(POLICY_A, POLICY_B), True)])
    signer = pol_leaf("trust-policy-mapping-inhibited", "Example Mapped Signer Below", mapping_sub, POLICY_B)
    signed("trust-policy-mapping-inhibited", signer, [mapping_sub, inhibit_ca], "Known bad: a CA inhibits policy mapping "
           "(inhibitPolicyMapping 0) and requires an explicit policy above a CA that maps its policy to the signer's: the "
           "mapped policy leaves the path, and the chain fails.", untrusted,
           bad=_bad("policy-mapping-inhibited", "chain", "leaves the path no valid certificate policy"))

    any_ca = ca("trust-policy-any-ca", "Example Any Policy CA", extra=[(_policies("2.5.29.32.0"), False)])
    signer = pol_leaf("trust-policy-any", "Example Any Policy Signer", any_ca, POLICY_A)
    signed("trust-policy-any", signer, [any_ca], f"Policies: the CA asserts anyPolicy, the signer {POLICY_A}: valid for "
           f"{POLICY_A}. A check inhibiting anyPolicy and requiring an explicit policy refuses the path.", trusted,
           policies=[POLICY_A], bad=_bad("policy-any-inhibited-by-the-check", "chain", "leaves the path no valid certificate policy",
                                         fails_with={"InhibitAnyPolicy": True, "RequireExplicitPolicy": True}))

    inhibit_any_ca = ca("trust-policy-inhibit-any-ca", "Example Inhibit Any Policy CA", extra=[
        (_policies("2.5.29.32.0"), False), (x509.PolicyConstraints(require_explicit_policy=0, inhibit_policy_mapping=None), True),
        (x509.InhibitAnyPolicy(0), True)])
    any_sub = ca("trust-policy-any-sub-ca", "Example Any Policy Sub CA", issuer=inhibit_any_ca,
                 extra=[(_policies("2.5.29.32.0"), False)])
    signer = pol_leaf("trust-policy-any-inhibited", "Example Any Policy Signer Below", any_sub, POLICY_A)
    signed("trust-policy-any-inhibited", signer, [any_sub, inhibit_any_ca], "Known bad: a CA inhibits anyPolicy "
           "(inhibitAnyPolicy 0) and requires an explicit policy above a CA asserting anyPolicy alone: the CA below "
           "leaves no policy, and the chain fails.", untrusted,
           bad=_bad("policy-any-inhibited", "chain", "Example Any Policy Sub CA's certificate leaves the path no valid certificate policy"))

    # -- delta and indirect lists, stale data (B-LT) ---------------------------------------------------------------------
    def crl(issuer_party, number, entries, this_update=T(2026, 3, 1), next_update=T(2036, 3, 1), extensions=()):
        b = (x509.CertificateRevocationListBuilder().issuer_name(issuer_party.cert.subject).last_update(this_update)
             .next_update(next_update).add_extension(x509.CRLNumber(number), critical=False)
             .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(issuer_party.cert.public_key()), critical=False))
        for extension, critical in extensions:
            b = b.add_extension(extension, critical=critical)
        for serial_number, when, extra in entries:
            r = x509.RevokedCertificateBuilder().serial_number(serial_number).revocation_date(when)
            for extension, critical in extra:
                r = r.add_extension(extension, critical=critical)
            b = b.add_revoked_certificate(r.build())
        return b.sign(issuer_party.key, hashes.SHA256()).public_bytes(serialization.Encoding.DER)

    def reason(flag):
        return (x509.CRLReason(flag), False)

    def lt(sample, signer, desc, crls, ocsps, results, bad=None, extra_certs=()):
        sd = SignedDoc(sample, seed)
        info, contents = sd.sign("Signature1", [240, 30, 400, 70], "ETSI.CAdES.detached",
                                 _pades(pki, signer, timestamp=True, label=sample), "Approved", "Head office", WHEN,
                                 label=signer.cert.subject.rfc4514_string().split("CN=")[1].split(",")[0])
        dss = sd.add_dss(pki, [contents], signer=signer, crls=crls, ocsps=ocsps, extra_certs=extra_certs)
        sig = _expect(info, pki, signer, "PAdES B-LT", "rsa-pkcs1v15-sha256", timestamp=TS, results=results, whole=False)
        if bad is not None:
            sig["knownBad"] = bad
        s = _sample(sd, sample, desc, [sig], pki, extra={"dss": dss})
        s.files.update(_files(signer, *extra_certs))
        s.files.update({n: data for n, data in crls + ocsps if not n.startswith("pki.root")})
        out.append(s)

    root_crl = [("pki.root.crl", pki.root_crl)]
    revoked_results = _results(timestamp="valid", revocation="revoked", chain="untrusted", modifications="allowed")
    good_results = _results(timestamp="valid", revocation="good", modifications="allowed")
    delta_url = "http://pki.example.com/intermediate-delta.crl"
    freshest = (x509.FreshestCRL([x509.DistributionPoint([x509.UniformResourceIdentifier(delta_url)], None, None, None)]), False)

    def delta_signer(label, cn):
        return leaf(label, name(cn), pki.intermediate, crl_url="http://pki.example.com/intermediate.crl")

    revoked = delta_signer("trust-crl-delta-revoked", "Example Signer Revoked By A Delta")
    base = crl(pki.intermediate, 20, [], extensions=[freshest])
    delta = crl(pki.intermediate, 21, [(revoked.cert.serial_number, T(2026, 2, 15), [reason(x509.ReasonFlags.key_compromise)])],
                this_update=T(2026, 3, 2, 8, 0, 0), extensions=[(x509.DeltaCRLIndicator(20), True)])
    lt("trust-crl-delta-revoked", revoked, "Known bad: PAdES B-LT whose DSS holds the intermediate's complete list (number "
       "20, naming its delta's address) that does not name the signer, and its delta list (number 21, deltaCRLIndicator "
       "20) revoking the signer on 2026-02-15 (keyCompromise): the revocation check fails.",
       [("pki.trust-crl-delta-revoked.crl", base), ("pki.trust-crl-delta-revoked.delta.crl", delta)] + root_crl, [],
       revoked_results, bad=_bad("delta-crl-revoked", "revocation", "revoked on 2026-02-15"))

    held = delta_signer("trust-crl-delta-released", "Example Signer Released By A Delta")
    base = crl(pki.intermediate, 20, [(held.cert.serial_number, T(2026, 2, 10), [reason(x509.ReasonFlags.certificate_hold)])],
               extensions=[freshest])
    delta = crl(pki.intermediate, 21, [(held.cert.serial_number, T(2026, 3, 2, 7, 0, 0), [reason(x509.ReasonFlags.remove_from_crl)])],
                this_update=T(2026, 3, 2, 8, 0, 0), extensions=[(x509.DeltaCRLIndicator(20), True)])
    lt("trust-crl-delta-released", held, "PAdES B-LT whose DSS holds the intermediate's complete list (number 20) putting "
       "the signer on hold since 2026-02-10 (certificateHold), and its delta list (number 21) taking it off "
       "(removeFromCRL): the signer is good.",
       [("pki.trust-crl-delta-released.crl", base), ("pki.trust-crl-delta-released.delta.crl", delta)] + root_crl, [],
       good_results)

    crl_issuer = Party("trust-crl-issuer", ca_key, _cert(name("Example Test CRL Issuer"), ca_key, pki.root.cert.subject,
                                                        pki.root.key, next(serial), T(2025, 1, 1), T(2040, 1, 1),
                                                        usages=("digital_signature", "crl_sign"), issuer_cert=pki.root.cert))
    indirect_url = "http://pki.example.com/indirect.crl"
    point = (x509.CRLDistributionPoints([x509.DistributionPoint([x509.UniformResourceIdentifier(indirect_url)], None, None,
                                                                [x509.DirectoryName(crl_issuer.cert.subject)])]), False)
    scope = (x509.IssuingDistributionPoint(full_name=[x509.UniformResourceIdentifier(indirect_url)], relative_name=None,
                                           only_contains_user_certs=False, only_contains_ca_certs=False,
                                           only_some_reasons=None, indirect_crl=True, only_contains_attribute_certs=False), True)
    certificate_issuer = (x509.CertificateIssuer([x509.DirectoryName(pki.intermediate.cert.subject)]), True)
    for how in ("revoked", "good"):
        sample = f"trust-crl-indirect-{how}"
        signer = leaf(sample, name(f"Example Signer With An Indirect List ({how})"), pki.intermediate, extra=[point])
        listed = signer.cert.serial_number if how == "revoked" else 0x4FFF
        entries = [(listed, T(2026, 2, 15), [certificate_issuer, reason(x509.ReasonFlags.key_compromise)])]
        indirect = crl(crl_issuer, 5, entries, extensions=[scope])
        if how == "revoked":
            desc = ("Known bad: PAdES B-LT whose DSS holds an indirect list (indirectCRL) from Example Test CRL Issuer, which "
                    "the signer's distribution point names as its lists' issuer (cRLIssuer); its entry names the "
                    "intermediate as the certificate's issuer (certificateIssuer) and revokes the signer on 2026-02-15: "
                    "the revocation check fails.")
            lt(sample, signer, desc, [(f"pki.{sample}.crl", indirect)] + root_crl, [], revoked_results,
               bad=_bad("indirect-crl-revoked", "revocation", "revoked on 2026-02-15"), extra_certs=(crl_issuer,))
        else:
            desc = ("PAdES B-LT whose DSS holds an indirect list from Example Test CRL Issuer (named by the signer's "
                    "distribution point), revoking another of the intermediate's certificates but not the signer, and "
                    "the CRL issuer's certificate (under the root, whose list covers it): the signer is good.")
            lt(sample, signer, desc, [(f"pki.{sample}.crl", indirect)] + root_crl, [], good_results, extra_certs=(crl_issuer,))

    stale_ocsp = _ocsp(pki, pki.signer, pki.intermediate.key, False, this_update=T(2026, 2, 20), next_update=T(2026, 2, 27),
                       produced_at=T(2026, 2, 20))
    stale_crl = crl(pki.intermediate, 30, [], this_update=T(2026, 2, 1), next_update=T(2026, 2, 28))
    lt("trust-stale-revocation", pki.signer, "Known bad (with revocation required): PAdES B-LT whose DSS holds an OCSP "
       "response for the signer (next update 2026-02-27) and the intermediate's list (next update 2026-02-28), both "
       "stale before the signature (2026-03-02): the signer is not checked (stale), which fails a check requiring "
       "revocation; with a grace of 14 days (FreshnessGrace) both are fresh and the signer good.",
       [("pki.trust-stale-revocation.crl", stale_crl)] + root_crl, [("pki.trust-stale-revocation.ocsp", stale_ocsp)],
       _results(timestamp="valid", revocation="not-checked", modifications="allowed"),
       bad=_bad("stale-revocation", "revocation", "is stale at", fails_with="RequireRevocation",
                passes_with={"option": "FreshnessGrace", "value": 14 * 86400}))
    return out
