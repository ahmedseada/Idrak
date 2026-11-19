// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Formats.Asn1;
using System.Numerics;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;

namespace Idrak.Data.Pdf;

/// <summary>An attribute of a CMS signer: its type and its values, as encoded.</summary>
internal sealed record CmsAttribute(string Oid, ReadOnlyMemory<byte>[] Values);

/// <summary>One signer of a CMS SignedData (RFC 5652, 5.3): who, with which digest and algorithm, its attributes and its signature.</summary>
internal sealed class CmsSignerInfo
{
    public int Version { get; init; }

    /// <summary>The signer by its certificate's issuer and serial number, or by its subject key identifier.</summary>
    public ReadOnlyMemory<byte> Issuer { get; init; }

    public BigInteger? Serial { get; init; }

    public byte[]? SubjectKeyId { get; init; }

    public required CmsAlgorithm DigestAlgorithm { get; init; }

    /// <summary>The signed attributes as they are signed (the DER of a SET), or null when there are none.</summary>
    public byte[]? SignedAttributesDer { get; init; }

    public IReadOnlyList<CmsAttribute> SignedAttributes { get; init; } = [];

    public required CmsAlgorithm SignatureAlgorithm { get; init; }

    public required byte[] Signature { get; init; }

    public IReadOnlyList<CmsAttribute> UnsignedAttributes { get; init; } = [];

    public CmsAttribute? Signed(string oid) => SignedAttributes.FirstOrDefault(a => a.Oid == oid);

    public CmsAttribute? Unsigned(string oid) => UnsignedAttributes.FirstOrDefault(a => a.Oid == oid);

    /// <summary>The certificate of <paramref name="certificates"/> this signer names, or null.</summary>
    public CmsCertificate? FindCertificate(IEnumerable<CmsCertificate> certificates)
    {
        foreach (var certificate in certificates)
        {
            if (Serial is { } serial ? certificate.Serial == serial && certificate.Issuer.Span.SequenceEqual(Issuer.Span)
                    : SubjectKeyId is { } key && certificate.SubjectKeyId is { } own && own.AsSpan().SequenceEqual(key))
            {
                return certificate;
            }
        }

        return null;
    }

    /// <summary>The digest this signer used, or null when it is not one read here.</summary>
    public HashAlgorithmName? Digest => CmsDigest.FromOid(DigestAlgorithm.Oid);

    /// <summary>The content type its signed attributes state, or null.</summary>
    public string? ContentType()
    {
        try
        {
            return Signed(CmsOids.ContentType) is { Values.Length: 1 } attribute ? CmsAsn.Reader(attribute.Values[0]).ReadObjectIdentifier() : null;
        }
        catch (Exception e) when (CmsAsn.IsDamage(e))
        {
            return null;
        }
    }

    /// <summary>The message digest its signed attributes state, or null.</summary>
    public byte[]? MessageDigest()
    {
        try
        {
            return Signed(CmsOids.MessageDigest) is { Values.Length: 1 } attribute ? CmsAsn.Reader(attribute.Values[0]).ReadOctetString() : null;
        }
        catch (Exception e) when (CmsAsn.IsDamage(e))
        {
            return null;
        }
    }

    /// <summary>The signing time its signed attributes state, or null.</summary>
    public DateTimeOffset? SigningTime()
    {
        try
        {
            return Signed(CmsOids.SigningTime) is { Values.Length: 1 } attribute ? CmsAsn.ReadTime(CmsAsn.Reader(attribute.Values[0])) : null;
        }
        catch (Exception e) when (CmsAsn.IsDamage(e))
        {
            return null;
        }
    }

    /// <summary>
    /// The signing certificate the signed attributes bind (ESS signing-certificate-v2, or the first version): null when
    /// there is none; else whether <paramref name="signer"/> is it, and what was found.
    /// </summary>
    public (bool Matches, string Detail)? SigningCertificate(CmsCertificate signer)
    {
        bool v2 = Signed(CmsOids.SigningCertificateV2) is not null;
        var attribute = Signed(CmsOids.SigningCertificateV2) ?? Signed(CmsOids.SigningCertificate);
        if (attribute is not { Values.Length: >= 1 })
        {
            return null;
        }

        try
        {
            // SigningCertificateV2 ::= SEQUENCE { certs SEQUENCE OF ESSCertIDv2, policies OPTIONAL }; the first names the signer.
            var certs = CmsAsn.Reader(attribute.Values[0]).ReadSequence().ReadSequence();
            var id = certs.ReadSequence();
            var digest = HashAlgorithmName.SHA1;
            if (v2)
            {
                digest = HashAlgorithmName.SHA256;
                if (CmsAsn.Next(id, Asn1Tag.Sequence))
                {
                    var algorithm = CmsAlgorithm.Read(id);
                    digest = CmsDigest.FromOid(algorithm.Oid) ?? throw new NotSupportedException($"a signing certificate hashed with {algorithm.Oid}");
                }
            }

            byte[] hash = id.ReadOctetString();
            if (!CmsDigest.Of(digest, signer.Raw.Span).AsSpan().SequenceEqual(hash))
            {
                return (false, $"the signed attributes bind another certificate than {signer.Named}'s ({(v2 ? "signing-certificate-v2" : "signing-certificate")}: its {CmsDigest.Named(digest)} differs)");
            }

            if (id.HasData)
            {
                // IssuerSerial ::= SEQUENCE { issuer GeneralNames, serialNumber }
                var issuerSerial = id.ReadSequence();
                issuerSerial.ReadEncodedValue();
                if (issuerSerial.ReadInteger() != signer.Serial)
                {
                    return (false, $"the signed attributes bind another serial number than {signer.Named}'s");
                }
            }

            return (true, $"{(v2 ? "signing-certificate-v2" : "signing-certificate")} binds {signer.Named}'s certificate");
        }
        catch (NotSupportedException e)
        {
            return (false, $"the signing certificate attribute: {e.Message}, which is not read");
        }
        catch (Exception e) when (CmsAsn.IsDamage(e))
        {
            return (false, $"the signing certificate attribute is not valid DER ({e.Message})");
        }
    }
}

/// <summary>
/// A CMS SignedData (RFC 5652) read on its DER (BER taken: what signers write): its content's type and, when it carries it,
/// the content; the certificates and revocation lists it holds; its signers. Immutable.
/// </summary>
internal sealed class CmsSignedData
{
    public required string ContentType { get; init; }

    /// <summary>The encapsulated content's octets, or null when detached.</summary>
    public byte[]? Content { get; init; }

    public required IReadOnlyList<CmsCertificate> Certificates { get; init; }

    /// <summary>The revocation lists it holds (CertificateList DER).</summary>
    public IReadOnlyList<ReadOnlyMemory<byte>> RevocationLists { get; init; } = [];

    public required IReadOnlyList<CmsSignerInfo> Signers { get; init; }

    /// <summary>The SignedData a ContentInfo holds (the bytes after it, a PDF's reserved zeros, are left).</summary>
    /// <exception cref="InvalidDataException">It is not a ContentInfo holding SignedData, or its DER is damaged (the message says where).</exception>
    public static CmsSignedData Parse(ReadOnlyMemory<byte> contentInfo)
    {
        try
        {
            var info = CmsAsn.Reader(contentInfo).ReadSequence();
            string type = info.ReadObjectIdentifier();
            if (type != CmsOids.SignedData)
            {
                throw new InvalidDataException($"a ContentInfo of {type}, not SignedData");
            }

            var signed = info.ReadSequence(CmsAsn.Context0).ReadSequence();
            signed.ReadInteger();                                              // the version
            var digests = signed.ReadSetOf(skipSortOrderValidation: true);
            while (digests.HasData)
            {
                digests.ReadEncodedValue();
            }

            var encapsulated = signed.ReadSequence();
            string contentType = encapsulated.ReadObjectIdentifier();
            byte[]? content = null;
            if (CmsAsn.Next(encapsulated, CmsAsn.Context0))
            {
                content = encapsulated.ReadSequence(CmsAsn.Context0).ReadOctetString();
            }

            var certificates = new List<CmsCertificate>();
            var lists = new List<ReadOnlyMemory<byte>>();
            if (CmsAsn.Next(signed, CmsAsn.Context0))
            {
                var set = signed.ReadSetOf(skipSortOrderValidation: true, CmsAsn.Context0);
                while (set.HasData)
                {
                    bool plain = set.PeekTag().HasSameClassAndValue(Asn1Tag.Sequence);
                    var encoded = set.ReadEncodedValue();
                    if (plain && CmsCertificate.TryParse(encoded.Span) is { } certificate)
                    {
                        certificates.Add(certificate);                         // attribute and other certificates are passed over
                    }
                }
            }

            if (CmsAsn.Next(signed, CmsAsn.Context1))
            {
                var set = signed.ReadSetOf(skipSortOrderValidation: true, CmsAsn.Context1);
                while (set.HasData)
                {
                    bool plain = set.PeekTag().HasSameClassAndValue(Asn1Tag.Sequence);
                    var encoded = set.ReadEncodedValue();
                    if (plain)
                    {
                        lists.Add(encoded);
                    }
                }
            }

            var signers = new List<CmsSignerInfo>();
            var infos = signed.ReadSetOf(skipSortOrderValidation: true);
            while (infos.HasData)
            {
                signers.Add(ReadSigner(infos.ReadSequence()));
            }

            return new CmsSignedData { ContentType = contentType, Content = content, Certificates = certificates, RevocationLists = lists, Signers = signers };
        }
        catch (Exception e) when (CmsAsn.IsDamage(e) && e is not InvalidDataException)
        {
            throw CmsAsn.Damaged("the CMS signature", e);
        }
    }

    private static CmsSignerInfo ReadSigner(AsnReader reader)
    {
        int version = (int)reader.ReadInteger();
        ReadOnlyMemory<byte> issuer = default;
        BigInteger? serial = null;
        byte[]? keyId = null;
        var keyIdTag = new Asn1Tag(TagClass.ContextSpecific, 0);
        if (CmsAsn.Next(reader, keyIdTag))
        {
            keyId = reader.ReadOctetString(keyIdTag);
        }
        else
        {
            var issuerAndSerial = reader.ReadSequence();
            issuer = issuerAndSerial.ReadEncodedValue();
            serial = issuerAndSerial.ReadInteger();
        }

        var digest = CmsAlgorithm.Read(reader);
        byte[]? signedDer = null;
        IReadOnlyList<CmsAttribute> signed = [];
        if (CmsAsn.Next(reader, CmsAsn.Context0))
        {
            signedDer = CmsAsn.AsSet(reader.ReadEncodedValue());
            signed = Attributes(CmsAsn.Reader(signedDer).ReadSetOf(skipSortOrderValidation: true));
        }

        var algorithm = CmsAlgorithm.Read(reader);
        byte[] signature = reader.ReadOctetString();
        IReadOnlyList<CmsAttribute> unsigned = [];
        if (CmsAsn.Next(reader, CmsAsn.Context1))
        {
            unsigned = Attributes(reader.ReadSetOf(skipSortOrderValidation: true, CmsAsn.Context1));
        }

        return new CmsSignerInfo
        {
            Version = version, Issuer = issuer, Serial = serial, SubjectKeyId = keyId, DigestAlgorithm = digest, SignedAttributesDer = signedDer,
            SignedAttributes = signed, SignatureAlgorithm = algorithm, Signature = signature, UnsignedAttributes = unsigned,
        };
    }

    private static List<CmsAttribute> Attributes(AsnReader set)
    {
        var attributes = new List<CmsAttribute>();
        while (set.HasData)
        {
            var attribute = set.ReadSequence();
            string oid = attribute.ReadObjectIdentifier();
            var values = attribute.ReadSetOf(skipSortOrderValidation: true);
            var list = new List<ReadOnlyMemory<byte>>();
            while (values.HasData)
            {
                list.Add(values.ReadEncodedValue());
            }

            attributes.Add(new CmsAttribute(oid, [.. list]));
        }

        return attributes;
    }
}

/// <summary>
/// One CMS signer checked: the digest its signed attributes state against the content's (the signed bytes' digest, or an
/// encapsulated content's), and its signature over them with its certificate's key.
/// </summary>
internal static class CmsVerifier
{
    /// <summary>
    /// The digest step: whether the signer's message digest is <paramref name="contentDigest"/> (the content hashed with the
    /// signer's digest), with what was found; with no signed attributes the signature itself is over the content (checked
    /// by <see cref="Signature"/>).
    /// </summary>
    public static (bool Passed, string Detail) Digest(CmsSignerInfo signer, ReadOnlySpan<byte> contentDigest, string content)
    {
        if (signer.Digest is not { } digest)
        {
            return (false, $"the signer's digest {signer.DigestAlgorithm.Oid} is not one read here (SHA-1, SHA-256, SHA-384, SHA-512)");
        }

        if (signer.SignedAttributesDer is null)
        {
            return (true, $"the {CmsDigest.Named(digest)} digest of {content} is signed directly (no signed attributes)");
        }

        if (signer.MessageDigest() is not { } stated)
        {
            return (false, "the signed attributes state no message digest");
        }

        return stated.AsSpan().SequenceEqual(contentDigest)
            ? (true, $"the {CmsDigest.Named(digest)} digest of {content} is the one signed")
            : (false, $"the {CmsDigest.Named(digest)} digest of {content} is not the one signed: {content} changed after signing (signed {Convert.ToHexString(stated)}, now {Convert.ToHexString(contentDigest)})");
    }

    /// <summary>
    /// The signature step: <paramref name="signer"/>'s signature over its signed attributes (or, with none, over the content
    /// whose digest is <paramref name="contentDigest"/>) verified with <paramref name="certificate"/>'s key; the signed
    /// attributes' content type (<paramref name="contentType"/>) and, when <paramref name="requireSigningCertificate"/>, the
    /// signing certificate they bind (ESS, PAdES).
    /// </summary>
    public static (bool Passed, string Detail) Signature(CmsSignerInfo signer, CmsCertificate? certificate, ReadOnlySpan<byte> contentDigest, string contentType,
        bool requireSigningCertificate)
    {
        if (certificate is null)
        {
            return (false, "the signer's certificate is not in the signature (nor in the document's validation data)");
        }

        var scheme = CmsKeys.Scheme(signer.SignatureAlgorithm, signer.Digest, out string? problem);
        if (scheme is not { } s)
        {
            return (false, $"the signature is made with {problem}");
        }

        var notes = new List<string>();
        string? failed;
        if (signer.SignedAttributesDer is { } attributes)
        {
            if (signer.ContentType() is not { } stated || stated != contentType)
            {
                return (false, $"the signed attributes state the content type {signer.ContentType() ?? "(none)"}, not {contentType}");
            }

            failed = CmsKeys.Verify(certificate, s, CmsDigest.Of(s.Digest, attributes), signer.Signature);
            if (signer.SigningCertificate(certificate) is { } bound)
            {
                if (!bound.Matches)
                {
                    return (false, bound.Detail);
                }

                notes.Add(bound.Detail);
            }
            else if (requireSigningCertificate)
            {
                return (false, "the signed attributes bind no signing certificate (ESS signing-certificate-v2, which PAdES asks for)");
            }
        }
        else
        {
            failed = s.Digest == signer.Digest ? CmsKeys.Verify(certificate, s, contentDigest, signer.Signature)
                : $"a signature with {CmsDigest.Named(s.Digest)} over content digested with {CmsDigest.Named(signer.Digest ?? s.Digest)}";
        }

        if (failed is not null)
        {
            return (false, failed);
        }

        notes.Insert(0, $"{s} by {certificate.Named} verifies");
        return (true, string.Join("; ", notes));
    }
}

/// <summary>
/// Writes a CMS SignedData on the DER (RFC 5652, through <see cref="AsnWriter"/>): one signer by issuer and serial number,
/// the signed attributes in DER order (content type, message digest, ESS signing-certificate-v2, any given), the signature
/// over them, unsigned attributes made from the signature (a timestamp), the certificates given.
/// </summary>
internal static class CmsWriter
{
    /// <summary>
    /// The signed attributes (the DER of their SET, as it is signed) for content of <paramref name="contentType"/> whose
    /// digest is <paramref name="contentDigest"/>, binding <paramref name="signer"/>'s certificate.
    /// </summary>
    public static byte[] SignedAttributes(string contentType, ReadOnlySpan<byte> contentDigest, CmsCertificate signer, HashAlgorithmName digest,
        IReadOnlyList<(string Oid, byte[] Value)>? more = null)
    {
        byte[] stated = contentDigest.ToArray();
        var writer = new AsnWriter(AsnEncodingRules.DER);
        using (writer.PushSetOf())
        {
            Attribute(writer, CmsOids.ContentType, w => w.WriteObjectIdentifier(contentType));
            Attribute(writer, CmsOids.MessageDigest, w => w.WriteOctetString(stated));
            Attribute(writer, CmsOids.SigningCertificateV2, w =>
            {
                using (w.PushSequence())                                       // SigningCertificateV2
                {
                    using (w.PushSequence())                                   // certs
                    {
                        using (w.PushSequence())                               // ESSCertIDv2
                        {
                            if (digest != HashAlgorithmName.SHA256)
                            {
                                CmsAsn.WriteAlgorithm(w, CmsDigest.Oid(digest));
                            }

                            w.WriteOctetString(CmsDigest.Of(digest, signer.Raw.Span));
                            using (w.PushSequence())                           // IssuerSerial
                            {
                                using (w.PushSequence())                       // GeneralNames
                                {
                                    using (w.PushSequence(new Asn1Tag(TagClass.ContextSpecific, 4, isConstructed: true)))
                                    {
                                        w.WriteEncodedValue(signer.Issuer.Span);
                                    }
                                }

                                w.WriteInteger(signer.Serial);
                            }
                        }
                    }
                }
            });
            foreach (var (oid, value) in more ?? [])
            {
                Attribute(writer, oid, w => w.WriteEncodedValue(value));
            }
        }

        return writer.Encode();
    }

    /// <summary>
    /// A ContentInfo holding SignedData: <paramref name="signedAttributes"/> (from <see cref="SignedAttributes"/>) signed as
    /// <paramref name="signature"/> by <paramref name="signer"/> under <paramref name="scheme"/>; <paramref name="content"/>
    /// encapsulated when given (a timestamp's TSTInfo), else detached; the certificates; unsigned attributes.
    /// </summary>
    public static byte[] SignedData(string contentType, byte[]? content, CmsCertificate signer, HashAlgorithmName digest, byte[] signedAttributes, CmsScheme scheme,
        byte[] signature, IReadOnlyList<CmsCertificate> certificates, IReadOnlyList<(string Oid, byte[] Value)>? unsigned = null)
    {
        var writer = new AsnWriter(AsnEncodingRules.DER);
        using (writer.PushSequence())
        {
            writer.WriteObjectIdentifier(CmsOids.SignedData);
            using (writer.PushSequence(CmsAsn.Context0))
            {
                using (writer.PushSequence())
                {
                    writer.WriteInteger(contentType == CmsOids.Data ? 1 : 3);
                    using (writer.PushSetOf())
                    {
                        CmsAsn.WriteAlgorithm(writer, CmsDigest.Oid(digest));
                    }

                    using (writer.PushSequence())
                    {
                        writer.WriteObjectIdentifier(contentType);
                        if (content is not null)
                        {
                            using (writer.PushSequence(CmsAsn.Context0))
                            {
                                writer.WriteOctetString(content);
                            }
                        }
                    }

                    using (writer.PushSetOf(CmsAsn.Context0))
                    {
                        var written = new List<CmsCertificate>();
                        foreach (var certificate in certificates)
                        {
                            if (!written.Exists(c => c.Same(certificate)))
                            {
                                writer.WriteEncodedValue(certificate.Raw.Span);
                                written.Add(certificate);
                            }
                        }
                    }

                    using (writer.PushSetOf())
                    {
                        using (writer.PushSequence())
                        {
                            writer.WriteInteger(1);
                            using (writer.PushSequence())
                            {
                                writer.WriteEncodedValue(signer.Issuer.Span);
                                writer.WriteInteger(signer.Serial);
                            }

                            CmsAsn.WriteAlgorithm(writer, CmsDigest.Oid(digest));
                            byte[] attributes = (byte[])signedAttributes.Clone();
                            attributes[0] = 0xA0;                              // [0] IMPLICIT in the signer, a SET when signed
                            writer.WriteEncodedValue(attributes);
                            CmsKeys.WriteAlgorithm(writer, scheme);
                            writer.WriteOctetString(signature);
                            if (unsigned is { Count: > 0 })
                            {
                                using (writer.PushSetOf(CmsAsn.Context1))
                                {
                                    foreach (var (oid, value) in unsigned)
                                    {
                                        Attribute(writer, oid, w => w.WriteEncodedValue(value));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return writer.Encode();
    }

    /// <summary>A whole SignedData made with <paramref name="key"/>'s private key (the timestamp authority stand-ins and tokens use it).</summary>
    public static byte[] Sign(string contentType, byte[]? content, byte[] contentDigest, X509Certificate2 key, HashAlgorithmName digest, RSASignaturePadding padding,
        IReadOnlyList<CmsCertificate> chain, IReadOnlyList<(string Oid, byte[] Value)>? unsigned = null)
    {
        var signer = CmsCertificate.From(key);
        byte[] attributes = SignedAttributes(contentType, contentDigest, signer, digest);
        var (signature, scheme) = CmsKeys.Sign(key, digest, padding, CmsDigest.Of(digest, attributes));
        return SignedData(contentType, content, signer, digest, attributes, scheme, signature, [signer, .. chain], unsigned);
    }

    private static void Attribute(AsnWriter writer, string oid, Action<AsnWriter> value)
    {
        using (writer.PushSequence())
        {
            writer.WriteObjectIdentifier(oid);
            using (writer.PushSetOf())
            {
                value(writer);
            }
        }
    }
}
