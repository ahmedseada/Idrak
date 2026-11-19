// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Formats.Asn1;
using System.Numerics;
using System.Security.Cryptography;

namespace Idrak.Data.Pdf;

/// <summary>What a revocation source says of one certificate at a time.</summary>
internal enum CmsStatus : byte
{
    /// <summary>Nothing usable: no source names the certificate, or none was valid at the time.</summary>
    Unknown,

    /// <summary>Not revoked at the time.</summary>
    Good,

    /// <summary>Revoked at or before the time.</summary>
    Revoked,
}

/// <summary>A certificate's status from one source, and what the source was.</summary>
internal readonly record struct CmsStatusAnswer(CmsStatus Status, string Detail, DateTimeOffset? RevokedAt = null, string? Reason = null);

/// <summary>
/// A certificate revocation list (RFC 5280, 5) read on its DER: its issuer, its times, the serial numbers it revokes (with
/// their dates and reasons), and the signature over its signed part. Only direct, complete lists are used: a delta list, an
/// indirect one (entries naming another issuer) or one with a critical extension not read here says so, and is not used.
/// </summary>
internal sealed class CmsCrl
{
    private readonly Dictionary<BigInteger, (DateTimeOffset Date, int? Reason)> _revoked = [];

    private CmsCrl(ReadOnlyMemory<byte> raw) => Raw = raw;

    public ReadOnlyMemory<byte> Raw { get; }

    public ReadOnlyMemory<byte> Tbs { get; private set; }

    public CmsAlgorithm SignatureAlgorithm { get; private set; }

    public byte[] Signature { get; private set; } = [];

    public ReadOnlyMemory<byte> Issuer { get; private set; }

    public DateTimeOffset ThisUpdate { get; private set; }

    public DateTimeOffset? NextUpdate { get; private set; }

    /// <summary>Why the list is not used (a delta list, an indirect one, an unread critical extension), or null.</summary>
    public string? NotUsable { get; private set; }

    public int Count => _revoked.Count;

    /// <summary>The list <paramref name="der"/> holds.</summary>
    /// <exception cref="InvalidDataException">It is not a CertificateList's DER.</exception>
    public static CmsCrl Parse(ReadOnlyMemory<byte> der)
    {
        var crl = new CmsCrl(der);
        try
        {
            var list = CmsAsn.Reader(der).ReadSequence();
            crl.Tbs = list.ReadEncodedValue();
            crl.SignatureAlgorithm = CmsAlgorithm.Read(list);
            crl.Signature = list.ReadBitString(out _);
            var tbs = CmsAsn.Reader(crl.Tbs).ReadSequence();
            if (CmsAsn.Next(tbs, Asn1Tag.Integer))
            {
                tbs.ReadInteger();                                             // the version
            }

            CmsAlgorithm.Read(tbs);
            crl.Issuer = tbs.ReadEncodedValue();
            crl.ThisUpdate = CmsAsn.ReadTime(tbs);
            if (tbs.HasData && (tbs.PeekTag().HasSameClassAndValue(Asn1Tag.UtcTime) || tbs.PeekTag().HasSameClassAndValue(Asn1Tag.GeneralizedTime)))
            {
                crl.NextUpdate = CmsAsn.ReadTime(tbs);
            }

            if (CmsAsn.Next(tbs, Asn1Tag.Sequence))
            {
                var entries = tbs.ReadSequence();
                while (entries.HasData)
                {
                    var entry = entries.ReadSequence();
                    var serial = entry.ReadInteger();
                    var date = CmsAsn.ReadTime(entry);
                    int? reason = null;
                    if (entry.HasData)
                    {
                        foreach (var (oid, critical, value) in Extensions(entry.ReadSequence()))
                        {
                            if (oid == CmsOids.CrlReason)
                            {
                                reason = CmsAsn.ReadEnumerated(CmsAsn.Reader(value));
                            }
                            else if (oid == CmsOids.CertificateIssuer || (critical && oid != CmsOids.InvalidityDate))
                            {
                                crl.NotUsable ??= $"an entry with the extension {oid} (an indirect list), which is not read";
                            }
                        }
                    }

                    crl._revoked[serial] = (date, reason);
                }
            }

            if (CmsAsn.Next(tbs, CmsAsn.Context0))
            {
                foreach (var (oid, critical, _) in Extensions(tbs.ReadSequence(CmsAsn.Context0).ReadSequence()))
                {
                    if (oid == CmsOids.DeltaCrlIndicator)
                    {
                        crl.NotUsable ??= "a delta list, which is not read";
                    }
                    else if (critical && oid is not (CmsOids.CrlNumber or CmsOids.AuthorityKeyId))
                    {
                        crl.NotUsable ??= $"the critical extension {oid}, which is not read";
                    }
                }
            }
        }
        catch (Exception e) when (CmsAsn.IsDamage(e) && e is not InvalidDataException)
        {
            throw CmsAsn.Damaged("a revocation list", e);
        }

        return crl;
    }

    /// <summary>
    /// What this list says of <paramref name="certificate"/> at <paramref name="at"/>, when <paramref name="issuer"/> (the
    /// certificate's issuer) signed it; <see cref="CmsStatus.Unknown"/> with why when it says nothing usable.
    /// </summary>
    public CmsStatusAnswer StatusOf(CmsCertificate certificate, CmsCertificate issuer, DateTimeOffset at)
    {
        if (!Issuer.Span.SequenceEqual(certificate.Issuer.Span))
        {
            return new CmsStatusAnswer(CmsStatus.Unknown, "another issuer's list");
        }

        if (NotUsable is not null)
        {
            return new CmsStatusAnswer(CmsStatus.Unknown, $"{issuer.Named}'s list is {NotUsable}");
        }

        if (NextUpdate is { } next && next < at)
        {
            return new CmsStatusAnswer(CmsStatus.Unknown, $"{issuer.Named}'s list of {CmsText.Time(ThisUpdate)} was out of date at {CmsText.Time(at)} (next update {CmsText.Time(next)})");
        }

        if (!issuer.Allows(CmsCertificate.CrlSign))
        {
            return new CmsStatusAnswer(CmsStatus.Unknown, $"{issuer.Named}'s key may not sign revocation lists (key usage)");
        }

        var scheme = CmsKeys.Scheme(SignatureAlgorithm, null, out string? problem);
        string? failed = scheme is { } s ? CmsKeys.Verify(issuer, s, CmsDigest.Of(s.Digest, Tbs.Span), Signature) : problem;
        if (failed is not null)
        {
            return new CmsStatusAnswer(CmsStatus.Unknown, $"{issuer.Named}'s list does not verify: {failed}");
        }

        if (_revoked.TryGetValue(certificate.Serial, out var revoked) && revoked.Reason != 8)   // removeFromCRL (8) takes a hold back
        {
            string reason = CmsRevocation.ReasonName(revoked.Reason);
            return revoked.Date <= at
                ? new CmsStatusAnswer(CmsStatus.Revoked, $"{certificate.Named} revoked on {CmsText.Time(revoked.Date)} ({reason}) by {issuer.Named}'s list of {CmsText.Time(ThisUpdate)}", revoked.Date, reason)
                : new CmsStatusAnswer(CmsStatus.Good, $"{certificate.Named} revoked only on {CmsText.Time(revoked.Date)} ({reason}), after {CmsText.Time(at)}", revoked.Date, reason);
        }

        return new CmsStatusAnswer(CmsStatus.Good, $"{certificate.Named} not on {issuer.Named}'s list of {CmsText.Time(ThisUpdate)}");
    }

    internal static List<(string Oid, bool Critical, ReadOnlyMemory<byte> Value)> Extensions(AsnReader sequence)
    {
        var list = new List<(string, bool, ReadOnlyMemory<byte>)>();
        while (sequence.HasData)
        {
            var extension = sequence.ReadSequence();
            string oid = extension.ReadObjectIdentifier();
            bool critical = CmsAsn.Next(extension, Asn1Tag.Boolean) && extension.ReadBoolean();
            list.Add((oid, critical, extension.ReadOctetString()));
        }

        return list;
    }
}

/// <summary>
/// An OCSP response (RFC 6960) read on its DER: a successful basic response's signed data (who answers, when, each
/// certificate's status), its signature and the certificates it carries.
/// </summary>
internal sealed class CmsOcsp
{
    private CmsOcsp(ReadOnlyMemory<byte> raw) => Raw = raw;

    public ReadOnlyMemory<byte> Raw { get; }

    public ReadOnlyMemory<byte> Tbs { get; private set; }

    public CmsAlgorithm SignatureAlgorithm { get; private set; }

    public byte[] Signature { get; private set; } = [];

    /// <summary>The responder by name (its encoding) or by its key's SHA-1.</summary>
    public ReadOnlyMemory<byte> ResponderName { get; private set; }

    public byte[]? ResponderKeyHash { get; private set; }

    public DateTimeOffset ProducedAt { get; private set; }

    public IReadOnlyList<CmsCertificate> Certificates { get; private set; } = [];

    public IReadOnlyList<Answer> Responses { get; private set; } = [];

    /// <summary>The response's status when it is not successful (named), else null.</summary>
    public string? Failure { get; private set; }

    /// <summary>One certificate's answer: who it names (the issuer's name and key hashes, the serial), its status and times.</summary>
    public sealed record Answer(HashAlgorithmName? Digest, byte[] IssuerNameHash, byte[] IssuerKeyHash, BigInteger Serial, CmsStatus Status, DateTimeOffset? RevokedAt,
        int? Reason, DateTimeOffset ThisUpdate, DateTimeOffset? NextUpdate);

    /// <summary>The response <paramref name="der"/> holds.</summary>
    /// <exception cref="InvalidDataException">It is not an OCSPResponse's DER, or not a basic one.</exception>
    public static CmsOcsp Parse(ReadOnlyMemory<byte> der)
    {
        var ocsp = new CmsOcsp(der);
        try
        {
            var response = CmsAsn.Reader(der).ReadSequence();
            int status = CmsAsn.ReadEnumerated(response);
            if (status != 0)
            {
                ocsp.Failure = status switch
                {
                    1 => "malformedRequest",
                    2 => "internalError",
                    3 => "tryLater",
                    5 => "sigRequired",
                    6 => "unauthorized",
                    _ => $"status {status}",
                };
                return ocsp;
            }

            var bytes = response.ReadSequence(CmsAsn.Context0).ReadSequence();
            string type = bytes.ReadObjectIdentifier();
            if (type != CmsOids.OcspBasic)
            {
                throw new InvalidDataException($"an OCSP response of the type {type}, not a basic one");
            }

            var basic = CmsAsn.Reader(bytes.ReadOctetString()).ReadSequence();
            ocsp.Tbs = basic.ReadEncodedValue();
            ocsp.SignatureAlgorithm = CmsAlgorithm.Read(basic);
            ocsp.Signature = basic.ReadBitString(out _);
            var certificates = new List<CmsCertificate>();
            if (CmsAsn.Next(basic, CmsAsn.Context0))
            {
                var list = basic.ReadSequence(CmsAsn.Context0).ReadSequence();
                while (list.HasData)
                {
                    if (CmsCertificate.TryParse(list.ReadEncodedValue().Span) is { } certificate)
                    {
                        certificates.Add(certificate);
                    }
                }
            }

            ocsp.Certificates = certificates;
            var data = CmsAsn.Reader(ocsp.Tbs).ReadSequence();
            if (CmsAsn.Next(data, CmsAsn.Context0))
            {
                data.ReadSequence(CmsAsn.Context0).ReadInteger();              // the version
            }

            if (CmsAsn.Next(data, CmsAsn.Context1))
            {
                ocsp.ResponderName = data.ReadSequence(CmsAsn.Context1).ReadEncodedValue();
            }
            else
            {
                ocsp.ResponderKeyHash = data.ReadSequence(CmsAsn.Context2).ReadOctetString();
            }

            ocsp.ProducedAt = data.ReadGeneralizedTime();
            var answers = new List<Answer>();
            var responses = data.ReadSequence();
            while (responses.HasData)
            {
                var single = responses.ReadSequence();
                var id = single.ReadSequence();
                var digest = CmsDigest.FromOid(CmsAlgorithm.Read(id).Oid);
                byte[] nameHash = id.ReadOctetString(), keyHash = id.ReadOctetString();
                var serial = id.ReadInteger();
                var tag = single.PeekTag();
                CmsStatus answer;
                DateTimeOffset? revokedAt = null;
                int? reason = null;
                if (tag.TagValue == 0 && tag.TagClass == TagClass.ContextSpecific)
                {
                    single.ReadNull(new Asn1Tag(TagClass.ContextSpecific, 0));
                    answer = CmsStatus.Good;
                }
                else if (tag.TagValue == 1 && tag.TagClass == TagClass.ContextSpecific)
                {
                    var revoked = single.ReadSequence(new Asn1Tag(TagClass.ContextSpecific, 1, isConstructed: true));
                    revokedAt = revoked.ReadGeneralizedTime();
                    if (CmsAsn.Next(revoked, CmsAsn.Context0))
                    {
                        reason = CmsAsn.ReadEnumerated(revoked.ReadSequence(CmsAsn.Context0));
                    }

                    answer = CmsStatus.Revoked;
                }
                else
                {
                    single.ReadEncodedValue();
                    answer = CmsStatus.Unknown;
                }

                var thisUpdate = single.ReadGeneralizedTime();
                DateTimeOffset? nextUpdate = CmsAsn.Next(single, CmsAsn.Context0) ? single.ReadSequence(CmsAsn.Context0).ReadGeneralizedTime() : null;
                answers.Add(new Answer(digest, nameHash, keyHash, serial, answer, revokedAt, reason, thisUpdate, nextUpdate));
            }

            ocsp.Responses = answers;
        }
        catch (Exception e) when (CmsAsn.IsDamage(e) && e is not InvalidDataException)
        {
            throw CmsAsn.Damaged("an OCSP response", e);
        }

        return ocsp;
    }

    /// <summary>
    /// What this response says of <paramref name="certificate"/> at <paramref name="at"/>: when it names the certificate (by
    /// its issuer's name and key and its serial), is signed by <paramref name="issuer"/> itself or by a responder
    /// <paramref name="issuer"/> issued for OCSP signing, and was not out of date at the time.
    /// </summary>
    public CmsStatusAnswer StatusOf(CmsCertificate certificate, CmsCertificate issuer, DateTimeOffset at)
    {
        if (Failure is not null)
        {
            return new CmsStatusAnswer(CmsStatus.Unknown, $"an OCSP response that is not successful ({Failure})");
        }

        Answer? found = null;
        foreach (var single in Responses)
        {
            if (single.Serial == certificate.Serial && single.Digest is { } digest
                && CmsDigest.Of(digest, issuer.Subject.Span).AsSpan().SequenceEqual(single.IssuerNameHash)
                && CmsDigest.Of(digest, issuer.PublicKey).AsSpan().SequenceEqual(single.IssuerKeyHash))
            {
                found = single;
                break;
            }
        }

        if (found is null)
        {
            return new CmsStatusAnswer(CmsStatus.Unknown, "a response for another certificate");
        }

        if (Responder(issuer) is not { } responder)
        {
            return new CmsStatusAnswer(CmsStatus.Unknown, $"an OCSP response for {certificate.Named} signed neither by {issuer.Named} nor by a responder it authorized");
        }

        var scheme = CmsKeys.Scheme(SignatureAlgorithm, null, out string? problem);
        string? failed = scheme is { } s ? CmsKeys.Verify(responder, s, CmsDigest.Of(s.Digest, Tbs.Span), Signature) : problem;
        if (failed is not null)
        {
            return new CmsStatusAnswer(CmsStatus.Unknown, $"the OCSP response for {certificate.Named} does not verify: {failed}");
        }

        if (found.NextUpdate is { } next && next < at)
        {
            return new CmsStatusAnswer(CmsStatus.Unknown, $"the OCSP response for {certificate.Named} of {CmsText.Time(found.ThisUpdate)} was out of date at {CmsText.Time(at)}");
        }

        string by = ReferenceEquals(responder, issuer) ? issuer.Named : $"{responder.Named} for {issuer.Named}";
        switch (found.Status)
        {
            case CmsStatus.Good:
                return new CmsStatusAnswer(CmsStatus.Good, $"{certificate.Named} good in {by}'s OCSP response of {CmsText.Time(found.ThisUpdate)}");
            case CmsStatus.Revoked when found.RevokedAt is { } when:
                string reason = CmsRevocation.ReasonName(found.Reason);
                return when <= at
                    ? new CmsStatusAnswer(CmsStatus.Revoked, $"{certificate.Named} revoked on {CmsText.Time(when)} ({reason}) in {by}'s OCSP response", when, reason)
                    : new CmsStatusAnswer(CmsStatus.Good, $"{certificate.Named} revoked only on {CmsText.Time(when)} ({reason}), after {CmsText.Time(at)}", when, reason);
            default:
                return new CmsStatusAnswer(CmsStatus.Unknown, $"{by}'s OCSP response does not know {certificate.Named}");
        }
    }

    // The certificate that signed the response: the issuer itself, or one the issuer issued for OCSP signing (RFC 6960, 4.2.2.2).
    private CmsCertificate? Responder(CmsCertificate issuer)
    {
        if (Names(issuer))
        {
            return issuer;
        }

        foreach (var candidate in Certificates)
        {
            if (Names(candidate) && candidate.NamesIssuer(issuer) && candidate.HasPurpose(CmsOids.OcspSigning) && candidate.Allows(CmsCertificate.DigitalSignature)
                && CmsPath.VerifiedBy(candidate, issuer) is null)
            {
                return candidate;
            }
        }

        return null;
    }

    private bool Names(CmsCertificate certificate) =>
        ResponderKeyHash is { } hash ? SHA1.HashData(certificate.PublicKey).AsSpan().SequenceEqual(hash) : ResponderName.Span.SequenceEqual(certificate.Subject.Span);
}

/// <summary>Revocation reasons by name, and the OCSP request a responder is asked with (RFC 6960, 4.1).</summary>
internal static class CmsRevocation
{
    public static string ReasonName(int? reason) => reason switch
    {
        null => "no reason given",
        0 => "unspecified",
        1 => "keyCompromise",
        2 => "cACompromise",
        3 => "affiliationChanged",
        4 => "superseded",
        5 => "cessationOfOperation",
        6 => "certificateHold",
        8 => "removeFromCRL",
        9 => "privilegeWithdrawn",
        10 => "aACompromise",
        _ => $"reason {reason}",
    };

    /// <summary>An OCSPRequest for <paramref name="certificate"/> issued by <paramref name="issuer"/> (its CertID by SHA-1, as responders expect), with a nonce.</summary>
    public static byte[] Request(CmsCertificate certificate, CmsCertificate issuer)
    {
        var writer = new AsnWriter(AsnEncodingRules.DER);
        using (writer.PushSequence())                                          // OCSPRequest
        {
            using (writer.PushSequence())                                      // TBSRequest
            {
                using (writer.PushSequence())                                  // requestList
                {
                    using (writer.PushSequence())                              // Request
                    {
                        using (writer.PushSequence())                          // CertID
                        {
                            CmsAsn.WriteAlgorithm(writer, CmsOids.Sha1, nullParameters: true);
                            writer.WriteOctetString(SHA1.HashData(issuer.Subject.Span));
                            writer.WriteOctetString(SHA1.HashData(issuer.PublicKey));
                            writer.WriteInteger(certificate.Serial);
                        }
                    }
                }

                using (writer.PushSequence(CmsAsn.Context2))                   // requestExtensions
                {
                    using (writer.PushSequence())
                    {
                        using (writer.PushSequence())
                        {
                            writer.WriteObjectIdentifier(CmsOids.OcspNonce);
                            var nonce = new AsnWriter(AsnEncodingRules.DER);
                            nonce.WriteOctetString(RandomNumberGenerator.GetBytes(16));
                            writer.WriteOctetString(nonce.Encode());
                        }
                    }
                }
            }
        }

        return writer.Encode();
    }
}
