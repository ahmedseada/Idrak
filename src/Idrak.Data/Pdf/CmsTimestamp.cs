// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Formats.Asn1;
using System.Numerics;
using System.Security.Cryptography;

namespace Idrak.Data.Pdf;

/// <summary>
/// An RFC 3161 timestamp token: a CMS SignedData whose content is a TSTInfo (the authority's policy, the digest it stamped,
/// its serial number, the time), read on its DER.
/// </summary>
internal sealed class CmsTimestamp
{
    private CmsTimestamp(CmsSignedData token, string policy, CmsAlgorithm imprintAlgorithm, byte[] imprint, BigInteger serial, DateTimeOffset time, BigInteger? nonce) =>
        (Token, Policy, ImprintAlgorithm, Imprint, Serial, Time, Nonce) = (token, policy, imprintAlgorithm, imprint, serial, time, nonce);

    public CmsSignedData Token { get; }

    public string Policy { get; }

    public CmsAlgorithm ImprintAlgorithm { get; }

    /// <summary>The digest the authority stamped (of a signature's value, or of a document's signed bytes).</summary>
    public byte[] Imprint { get; }

    public BigInteger Serial { get; }

    /// <summary>When the authority says it stamped the digest (genTime).</summary>
    public DateTimeOffset Time { get; }

    public BigInteger? Nonce { get; }

    /// <summary>The token <paramref name="contentInfo"/> holds.</summary>
    /// <exception cref="InvalidDataException">It is not a timestamp token's DER.</exception>
    public static CmsTimestamp Parse(ReadOnlyMemory<byte> contentInfo)
    {
        var token = CmsSignedData.Parse(contentInfo);
        if (token.ContentType != CmsOids.TstInfo || token.Content is not { } content)
        {
            throw new InvalidDataException($"a timestamp token whose content is {token.ContentType}{(token.Content is null ? " (detached)" : "")}, not a TSTInfo");
        }

        try
        {
            var info = CmsAsn.Reader(content).ReadSequence();
            info.ReadInteger();                                                // the version
            string policy = info.ReadObjectIdentifier();
            var imprint = info.ReadSequence();
            var algorithm = CmsAlgorithm.Read(imprint);
            byte[] hashed = imprint.ReadOctetString();
            var serial = info.ReadInteger();
            var time = info.ReadGeneralizedTime();
            BigInteger? nonce = null;
            while (info.HasData)
            {
                var tag = info.PeekTag();
                if (tag.HasSameClassAndValue(Asn1Tag.Integer))
                {
                    nonce = info.ReadInteger();
                }
                else
                {
                    info.ReadEncodedValue();                                   // accuracy, ordering, tsa, extensions
                }
            }

            return new CmsTimestamp(token, policy, algorithm, hashed, serial, time, nonce);
        }
        catch (Exception e) when (CmsAsn.IsDamage(e) && e is not InvalidDataException)
        {
            throw CmsAsn.Damaged("a timestamp's TSTInfo", e);
        }
    }

    /// <summary>
    /// Whether the token stamps what <paramref name="digestOf"/> digests (with the imprint's digest: a signature's value, a
    /// document's signed bytes); null when it does, else why not.
    /// </summary>
    public string? Stamps(Func<HashAlgorithmName, byte[]> digestOf, string what)
    {
        if (CmsDigest.FromOid(ImprintAlgorithm.Oid) is not { } digest)
        {
            return $"the timestamp stamps a digest by {ImprintAlgorithm.Oid}, which is not read";
        }

        return digestOf(digest).AsSpan().SequenceEqual(Imprint) ? null
            : $"the timestamp stamps other bytes than {what} (its {CmsDigest.Named(digest)} message imprint differs)";
    }

    /// <summary>
    /// The token itself checked: its signer's digest and signature over the TSTInfo verify and its signer is a timestamp
    /// authority (the timeStamping purpose); null when it is sound, else why not. The authority's chain is the caller's
    /// (<see cref="Signer"/>).
    /// </summary>
    public string? CheckToken()
    {
        if (Token.Signers is not [var signer])
        {
            return $"a timestamp token with {Token.Signers.Count} signers (one is expected)";
        }

        var certificate = Signer;
        if (signer.Digest is not { } tokenDigest)
        {
            return $"the timestamp's signer digests with {signer.DigestAlgorithm.Oid}, which is not read";
        }

        byte[] contentDigest = CmsDigest.Of(tokenDigest, Token.Content);
        var (digestOk, digestDetail) = CmsVerifier.Digest(signer, contentDigest, "the timestamp's TSTInfo");
        if (!digestOk)
        {
            return digestDetail;
        }

        var (signatureOk, signatureDetail) = CmsVerifier.Signature(signer, certificate, contentDigest, CmsOids.TstInfo, requireSigningCertificate: false);
        if (!signatureOk)
        {
            return $"the timestamp's signature: {signatureDetail}";
        }

        if (certificate is not null && !certificate.HasPurpose(CmsOids.TimeStamping))
        {
            return $"the timestamp is signed by {certificate.Named}, whose certificate is not for timestamping (no timeStamping purpose)";
        }

        return null;
    }

    /// <summary>The authority's certificate, from the token's own certificates; null when it does not carry it.</summary>
    public CmsCertificate? Signer => Token.Signers is [var signer] ? signer.FindCertificate(Token.Certificates) : null;

    /// <summary>A TimeStampReq (RFC 3161, 2.4.1) for the digest <paramref name="hash"/>, asking for the authority's certificate, with <paramref name="nonce"/>.</summary>
    public static byte[] Request(HashAlgorithmName digest, ReadOnlySpan<byte> hash, BigInteger nonce)
    {
        var writer = new AsnWriter(AsnEncodingRules.DER);
        using (writer.PushSequence())
        {
            writer.WriteInteger(1);
            using (writer.PushSequence())
            {
                CmsAsn.WriteAlgorithm(writer, CmsDigest.Oid(digest));
                writer.WriteOctetString(hash);
            }

            writer.WriteInteger(nonce);
            writer.WriteBoolean(true);                                         // certReq: the authority's certificate in the token
        }

        return writer.Encode();
    }

    /// <summary>
    /// The token a TimeStampResp (RFC 3161, 2.4.2) holds, checked against the request: granted, a token stamping
    /// <paramref name="hash"/> with <paramref name="nonce"/>.
    /// </summary>
    /// <exception cref="IOException">The authority refused, or answered with a token for another request (the message says which).</exception>
    public static (byte[] Token, CmsTimestamp Read) FromResponse(ReadOnlyMemory<byte> response, ReadOnlySpan<byte> hash, BigInteger nonce, string authority)
    {
        byte[] token;
        try
        {
            var reader = CmsAsn.Reader(response).ReadSequence();
            var status = reader.ReadSequence();
            int granted = (int)status.ReadInteger();
            string? text = null;
            if (CmsAsn.Next(status, Asn1Tag.Sequence))
            {
                var strings = status.ReadSequence();
                if (strings.HasData)
                {
                    text = strings.ReadCharacterString(UniversalTagNumber.UTF8String);
                }
            }

            if (granted is not (0 or 1))
            {
                throw new IOException($"The timestamp authority {authority} refused the request (status {granted}{(text is null ? "" : $": {text}")}).");
            }

            if (!reader.HasData)
            {
                throw new IOException($"The timestamp authority {authority} granted the request but sent no token.");
            }

            token = reader.ReadEncodedValue().ToArray();
        }
        catch (Exception e) when (CmsAsn.IsDamage(e))
        {
            throw new IOException($"The timestamp authority {authority} answered with bytes that are not a TimeStampResp ({e.Message}).", e);
        }

        CmsTimestamp read;
        try
        {
            read = Parse(token);
        }
        catch (InvalidDataException e)
        {
            throw new IOException($"The timestamp authority {authority} answered with a token that cannot be read ({e.Message}).", e);
        }

        if (!read.Imprint.AsSpan().SequenceEqual(hash) || read.Nonce != nonce)
        {
            throw new IOException($"The timestamp authority {authority} answered with a token for another request (its digest or nonce differs).");
        }

        return (token, read);
    }
}
