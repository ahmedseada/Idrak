// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Formats.Asn1;
using System.Numerics;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;

namespace Idrak.Data.Pdf;

/// <summary>
/// An X.509 certificate (RFC 5280) read on its DER here, the same on every platform: what a signature check needs of it
/// (the signed part, the names and alternative names, the validity, the key, the extensions a path is checked by: its
/// constraints on names and policies, its revocation lists' distribution points among them), and the
/// <see cref="X509Certificate2"/> the contract hands out, made when asked. Immutable; safe from several threads.
/// </summary>
internal sealed class CmsCertificate
{
    // Key usage bits (RFC 5280, 4.2.1.3).
    public const int DigitalSignature = 0, NonRepudiation = 1, KeyCertSign = 5, CrlSign = 6;

    private readonly byte[] _raw;
    private X509Certificate2? _certificate;
    private byte[]? _sha256;
    private (BigInteger Modulus, BigInteger Exponent, int Bits)? _rsa;

    private CmsCertificate(byte[] raw) => _raw = raw;

    /// <summary>The certificate's DER.</summary>
    public ReadOnlyMemory<byte> Raw => _raw;

    /// <summary>The signed part (TBSCertificate), as encoded.</summary>
    public ReadOnlyMemory<byte> Tbs { get; private set; }

    public CmsAlgorithm SignatureAlgorithm { get; private set; }

    public byte[] Signature { get; private set; } = [];

    /// <summary>The serial number's INTEGER content (two's complement, big-endian).</summary>
    public ReadOnlyMemory<byte> SerialBytes { get; private set; }

    public BigInteger Serial { get; private set; }

    /// <summary>The issuer's and the subject's names, as encoded.</summary>
    public ReadOnlyMemory<byte> Issuer { get; private set; }

    public ReadOnlyMemory<byte> Subject { get; private set; }

    public DateTimeOffset NotBefore { get; private set; }

    public DateTimeOffset NotAfter { get; private set; }

    /// <summary>The SubjectPublicKeyInfo, as encoded, and its parts: the key's algorithm, its parameters and the key's bits.</summary>
    public ReadOnlyMemory<byte> PublicKeyInfo { get; private set; }

    public CmsAlgorithm KeyAlgorithm { get; private set; }

    public byte[] PublicKey { get; private set; } = [];

    /// <summary>The extensions: identifier, critical, value (the OCTET STRING's content).</summary>
    public IReadOnlyList<(string Oid, bool Critical, ReadOnlyMemory<byte> Value)> Extensions { get; private set; } = [];

    /// <summary>Basic constraints: whether a CA (null: no extension), and the path length it allows (null: no limit).</summary>
    public bool? IsCa { get; private set; }

    public int? PathLength { get; private set; }

    /// <summary>The key usage bits, or null without the extension.</summary>
    public byte[]? KeyUsage { get; private set; }

    /// <summary>The extended key usages, or null without the extension.</summary>
    public IReadOnlyList<string>? ExtendedKeyUsage { get; private set; }

    public byte[]? SubjectKeyId { get; private set; }

    public byte[]? AuthorityKeyId { get; private set; }

    /// <summary>The OCSP responders (authority information access) and revocation list addresses (distribution points) it names.</summary>
    public IReadOnlyList<string> OcspAddresses { get; private set; } = [];

    public IReadOnlyList<string> CrlAddresses { get; private set; } = [];

    /// <summary>Whether it carries id-pkix-ocsp-nocheck (a responder's certificate needing no status of its own).</summary>
    public bool OcspNoCheck { get; private set; }

    /// <summary>Its CRL distribution points as they are written (names, reasons, CRL issuers), for a list's scope.</summary>
    public IReadOnlyList<CmsDistributionPoint> CrlPoints { get; private set; } = [];

    /// <summary>The addresses of its delta lists (freshest CRL, RFC 5280 4.2.1.15).</summary>
    public IReadOnlyList<string> FreshestAddresses { get; private set; } = [];

    /// <summary>The subject alternative names (null without the extension), and why they cannot be read (null when they can).</summary>
    public IReadOnlyList<CmsGeneralName>? AltNames { get; private set; }

    public string? AltNamesProblem { get; private set; }

    /// <summary>The name constraints (null without the extension).</summary>
    public CmsNameConstraints? NameConstraints { get; private set; }

    /// <summary>The certificate policies' identifiers (null without the extension).</summary>
    public IReadOnlyList<string>? Policies { get; private set; }

    /// <summary>The policy mappings: issuer domain policy to subject domain policy (null without the extension).</summary>
    public IReadOnlyList<(string Issuer, string Subject)>? PolicyMappings { get; private set; }

    /// <summary>The policy constraints' requireExplicitPolicy and inhibitPolicyMapping (null when absent), and inhibit-any-policy's skip count.</summary>
    public int? RequireExplicitPolicy { get; private set; }

    public int? InhibitPolicyMapping { get; private set; }

    public int? InhibitAnyPolicy { get; private set; }

    /// <summary>
    /// What makes it unfit for a path though it reads (an extension twice, an empty extended key usage, name constraints or
    /// policy extensions that cannot be used), or null.
    /// </summary>
    public string? Problem { get; private set; }

    public string SubjectName => CmsName.Describe(Subject);

    public string IssuerName => CmsName.Describe(Issuer);

    /// <summary>The subject's common name, for messages.</summary>
    public string Named => CmsName.Common(Subject);

    public bool SelfIssued => Issuer.Span.SequenceEqual(Subject.Span);

    public byte[] Sha256 => _sha256 ??= SHA256.HashData(_raw);

    /// <summary>The certificate as the base library holds it (made on the first call).</summary>
    public X509Certificate2 Certificate => _certificate ??= X509CertificateLoader.LoadCertificate(_raw);

    /// <summary>Whether the key usage allows <paramref name="bit"/> (true when the certificate has no key usage).</summary>
    public bool Allows(int bit) => KeyUsage is not { } bits || Asserts(bit);

    /// <summary>Whether the key usage names <paramref name="bit"/> (false when the certificate has no key usage).</summary>
    public bool Asserts(int bit) => KeyUsage is { } bits && bit / 8 < bits.Length && (bits[bit / 8] & (0x80 >> (bit % 8))) != 0;

    /// <summary>Whether the extended key usage lists <paramref name="oid"/> (or any purpose).</summary>
    public bool HasPurpose(string oid) => ExtendedKeyUsage is { } usages && (usages.Contains(oid) || usages.Contains(CmsOids.AnyExtendedKeyUsage));

    /// <summary>Whether this is the same certificate as <paramref name="other"/> (byte for byte).</summary>
    public bool Same(CmsCertificate other) => ReferenceEquals(this, other) || _raw.AsSpan().SequenceEqual(other._raw);

    /// <summary>Whether one of its distribution points names <paramref name="crlIssuer"/> (a name's encoding) as its lists' issuer (cRLIssuer: an indirect list's).</summary>
    public bool NamesCrlIssuer(ReadOnlyMemory<byte> crlIssuer)
    {
        CmsNameAttribute[][]? wanted = null;
        foreach (var point in CrlPoints)
        {
            foreach (var name in point.CrlIssuer ?? [])
            {
                if (name.Form == CmsNameForm.Directory
                    && (name.Value.Span.SequenceEqual(crlIssuer.Span) || ((wanted ??= CmsDn.Read(crlIssuer)) is { } dn && CmsDn.Read(name.Value) is { } named && CmsDn.Same(named, dn))))
                {
                    return true;
                }
            }
        }

        return false;
    }

    /// <summary>Whether <paramref name="issuer"/> could have issued this one: its subject is this one's issuer, and the key identifiers agree where both are given.</summary>
    public bool NamesIssuer(CmsCertificate issuer) =>
        Issuer.Span.SequenceEqual(issuer.Subject.Span) && (AuthorityKeyId is null || issuer.SubjectKeyId is null || AuthorityKeyId.AsSpan().SequenceEqual(issuer.SubjectKeyId));

    /// <summary>The RSA key's modulus and exponent (null when the key is not RSA, or cannot be read).</summary>
    public (BigInteger Modulus, BigInteger Exponent, int Bits)? Rsa()
    {
        if (_rsa is { } known)
        {
            return known;
        }

        if (KeyAlgorithm.Oid is not (CmsOids.RsaEncryption or CmsOids.RsaPss))
        {
            return null;
        }

        try
        {
            var key = CmsAsn.Reader(PublicKey).ReadSequence();
            var modulus = key.ReadInteger();
            var exponent = key.ReadInteger();
            if (modulus.Sign <= 0 || exponent.Sign <= 0)
            {
                return null;
            }

            var rsa = (modulus, exponent, (int)modulus.GetBitLength());
            _rsa = rsa;
            return rsa;
        }
        catch (Exception e) when (CmsAsn.IsDamage(e))
        {
            return null;
        }
    }

    /// <summary>The certificate <paramref name="der"/> holds (the bytes are copied).</summary>
    /// <exception cref="InvalidDataException">It is not a certificate's DER (the message says what is wrong).</exception>
    public static CmsCertificate Parse(ReadOnlySpan<byte> der)
    {
        var certificate = new CmsCertificate(der.ToArray());
        try
        {
            certificate.Read();
        }
        catch (Exception e) when (CmsAsn.IsDamage(e) && e is not InvalidDataException)
        {
            throw CmsAsn.Damaged("a certificate", e);
        }

        return certificate;
    }

    /// <summary>The certificate <paramref name="der"/> holds, or null when it is not one.</summary>
    public static CmsCertificate? TryParse(ReadOnlySpan<byte> der)
    {
        try
        {
            return Parse(der);
        }
        catch (InvalidDataException)
        {
            return null;
        }
    }

    /// <summary>A certificate the base library holds, read here.</summary>
    public static CmsCertificate From(X509Certificate2 certificate)
    {
        var read = Parse(certificate.RawData);
        read._certificate = certificate;
        return read;
    }

    private void Read()
    {
        var reader = CmsAsn.Reader(_raw);
        var certificate = reader.ReadSequence();
        Tbs = certificate.ReadEncodedValue();
        SignatureAlgorithm = CmsAlgorithm.Read(certificate);
        Signature = certificate.ReadBitString(out int unused);
        if (unused != 0)
        {
            throw new InvalidDataException("a certificate's signature that is not whole bytes");
        }

        var tbs = CmsAsn.Reader(Tbs).ReadSequence();
        if (CmsAsn.Next(tbs, CmsAsn.Context0))
        {
            tbs.ReadSequence(CmsAsn.Context0).ReadInteger();                 // the version
        }

        SerialBytes = tbs.ReadIntegerBytes();
        Serial = new BigInteger(SerialBytes.Span, isUnsigned: false, isBigEndian: true);
        var inner = CmsAlgorithm.Read(tbs);
        if (inner.Oid != SignatureAlgorithm.Oid)
        {
            throw new InvalidDataException($"a certificate whose signed part names {inner.Oid} and whose signature {SignatureAlgorithm.Oid}");
        }

        Issuer = tbs.ReadEncodedValue();
        var validity = tbs.ReadSequence();
        NotBefore = CmsAsn.ReadTime(validity);
        NotAfter = CmsAsn.ReadTime(validity);
        Subject = tbs.ReadEncodedValue();
        PublicKeyInfo = tbs.ReadEncodedValue();
        var spki = CmsAsn.Reader(PublicKeyInfo).ReadSequence();
        KeyAlgorithm = CmsAlgorithm.Read(spki);
        PublicKey = spki.ReadBitString(out _);
        var extensions = new List<(string, bool, ReadOnlyMemory<byte>)>();
        while (tbs.HasData)
        {
            var tag = tbs.PeekTag();
            if (tag.HasSameClassAndValue(CmsAsn.Context3))
            {
                var list = tbs.ReadSequence(CmsAsn.Context3).ReadSequence();
                while (list.HasData)
                {
                    var extension = list.ReadSequence();
                    string oid = extension.ReadObjectIdentifier();
                    bool critical = CmsAsn.Next(extension, Asn1Tag.Boolean) && extension.ReadBoolean();
                    var value = extension.ReadOctetString();
                    extensions.Add((oid, critical, value));
                }
            }
            else
            {
                tbs.ReadEncodedValue();                                        // the unique identifiers ([1], [2])
            }
        }

        Extensions = extensions;
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var (oid, _, value) in extensions)
        {
            if (!seen.Add(oid))
            {
                Problem ??= $"the extension {oid} twice (a certificate holds each at most once, RFC 5280 4.2)";
            }

            ReadExtension(oid, value);
        }
    }

    private void ReadExtension(string oid, ReadOnlyMemory<byte> value)
    {
        var reader = CmsAsn.Reader(value);
        switch (oid)
        {
            case CmsOids.BasicConstraints:
                var constraints = reader.ReadSequence();
                IsCa = CmsAsn.Next(constraints, Asn1Tag.Boolean) && constraints.ReadBoolean();
                if (constraints.HasData && constraints.TryReadInt32(out int length))
                {
                    PathLength = length;
                }

                break;
            case CmsOids.KeyUsage:
                KeyUsage = reader.ReadBitString(out _);
                break;
            case CmsOids.ExtendedKeyUsage:
                var usages = new List<string>();
                var sequence = reader.ReadSequence();
                while (sequence.HasData)
                {
                    usages.Add(sequence.ReadObjectIdentifier());
                }

                if (usages.Count == 0)
                {
                    Problem ??= "an extended key usage naming no purpose (it names one or more, RFC 5280 4.2.1.12)";
                }

                ExtendedKeyUsage = usages;
                break;
            case CmsOids.SubjectKeyId:
                SubjectKeyId = reader.ReadOctetString();
                break;
            case CmsOids.AuthorityKeyId:
                var authority = reader.ReadSequence();
                var keyId = new Asn1Tag(TagClass.ContextSpecific, 0);
                if (CmsAsn.Next(authority, keyId))
                {
                    AuthorityKeyId = authority.ReadOctetString(keyId);
                }

                break;
            case CmsOids.AuthorityInfoAccess:
                var addresses = new List<string>();
                var access = reader.ReadSequence();
                while (access.HasData)
                {
                    var description = access.ReadSequence();
                    string method = description.ReadObjectIdentifier();
                    if (method == CmsOids.OcspAccess && Uri(description) is { } uri)
                    {
                        addresses.Add(uri);
                    }
                }

                OcspAddresses = addresses;
                break;
            case CmsOids.CrlDistributionPoints:
                var crls = new List<string>();
                var points = reader.ReadSequence();
                while (points.HasData)
                {
                    var point = points.ReadSequence();
                    if (CmsAsn.Next(point, CmsAsn.Context0))
                    {
                        var name = point.ReadSequence(CmsAsn.Context0);
                        if (CmsAsn.Next(name, CmsAsn.Context0))
                        {
                            var full = name.ReadSequence(CmsAsn.Context0);
                            while (full.HasData)
                            {
                                if (Uri(full) is { } uri)
                                {
                                    crls.Add(uri);
                                }
                            }
                        }
                    }
                }

                CrlAddresses = crls;
                break;
            case CmsOids.OcspNoCheck:
                OcspNoCheck = true;
                break;
        }

        Constraint(oid, value);
    }

    // The extensions a path's constraints and a list's scope are read from: one that cannot be read leaves the certificate
    // read, and fails a path it is on (Problem) or a check of its names (AltNamesProblem); a distribution point that cannot be
    // read leaves its addresses as read above.
    private void Constraint(string oid, ReadOnlyMemory<byte> value)
    {
        try
        {
            switch (oid)
            {
                case CmsOids.SubjectAltName:
                    var names = CmsAsn.Reader(value);
                    AltNames = CmsGeneralName.ReadAll(names.ReadSequence());
                    names.ThrowIfNotEmpty();
                    break;
                case CmsOids.NameConstraints:
                    NameConstraints = CmsNameConstraints.Read(value, out string? problem);
                    if (problem is not null)
                    {
                        Problem ??= problem;
                    }

                    break;
                case CmsOids.CertificatePolicies:
                    var policies = new List<string>();
                    var list = CmsAsn.Reader(value).ReadSequence();
                    while (list.HasData)
                    {
                        var information = list.ReadSequence();
                        string policy = information.ReadObjectIdentifier();
                        if (!policies.Contains(policy))
                        {
                            policies.Add(policy);                              // the qualifiers are not read
                        }
                    }

                    Policies = policies.Count > 0 ? policies : throw new InvalidDataException("certificate policies naming none (RFC 5280 4.2.1.4: one or more)");
                    break;
                case CmsOids.PolicyMappings:
                    var mappings = new List<(string, string)>();
                    var pairs = CmsAsn.Reader(value).ReadSequence();
                    while (pairs.HasData)
                    {
                        var pair = pairs.ReadSequence();
                        mappings.Add((pair.ReadObjectIdentifier(), pair.ReadObjectIdentifier()));
                    }

                    PolicyMappings = mappings.Count > 0 ? mappings : throw new InvalidDataException("policy mappings naming none (RFC 5280 4.2.1.5: one or more)");
                    break;
                case CmsOids.PolicyConstraints:
                    var constraints = CmsAsn.Reader(value).ReadSequence();
                    var require = new Asn1Tag(TagClass.ContextSpecific, 0);
                    var inhibit = new Asn1Tag(TagClass.ContextSpecific, 1);
                    if (CmsAsn.Next(constraints, require))
                    {
                        RequireExplicitPolicy = SkipCount(constraints.ReadInteger(require));
                    }

                    if (CmsAsn.Next(constraints, inhibit))
                    {
                        InhibitPolicyMapping = SkipCount(constraints.ReadInteger(inhibit));
                    }

                    if (RequireExplicitPolicy is null && InhibitPolicyMapping is null)
                    {
                        throw new InvalidDataException("policy constraints naming neither requireExplicitPolicy nor inhibitPolicyMapping (RFC 5280 4.2.1.11)");
                    }

                    break;
                case CmsOids.InhibitAnyPolicy:
                    InhibitAnyPolicy = SkipCount(CmsAsn.Reader(value).ReadInteger());
                    break;
                case CmsOids.CrlDistributionPoints:
                    CrlPoints = CmsDistributionPoint.ReadAll(value);
                    break;
                case CmsOids.FreshestCrl:
                    FreshestAddresses = [.. CmsDistributionPoint.ReadAll(value).SelectMany(p => p.Addresses)];
                    break;
            }
        }
        catch (Exception e) when (CmsAsn.IsDamage(e))
        {
            switch (oid)
            {
                case CmsOids.SubjectAltName:
                    AltNamesProblem = e.Message;
                    AltNames = [];
                    break;
                case CmsOids.CrlDistributionPoints or CmsOids.FreshestCrl:
                    break;                                                     // the addresses read above stay
                default:
                    Problem ??= $"the extension {oid}, which cannot be read ({e.Message})";
                    break;
            }
        }
    }

    // A skip count (RFC 5280 4.2.1.11 and 4.2.1.14): 0 or more; a count past what a path can hold is as good as no limit.
    private static int SkipCount(BigInteger count) =>
        count.Sign < 0 ? throw new InvalidDataException("a negative skip count") : count > 1024 ? 1024 : (int)count;

    // A GeneralName read: its text when it is a uniformResourceIdentifier ([6] IA5String), else null (passed over).
    private static string? Uri(AsnReader reader)
    {
        var uri = new Asn1Tag(TagClass.ContextSpecific, 6);
        if (CmsAsn.Next(reader, uri))
        {
            return reader.ReadCharacterString(UniversalTagNumber.IA5String, uri);
        }

        reader.ReadEncodedValue();
        return null;
    }
}
