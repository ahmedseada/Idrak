// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Formats.Asn1;
using System.Numerics;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;

namespace Idrak.Data.Pdf;

/// <summary>How a signature was made: RSA with PKCS #1 v1.5 or PSS padding, or ECDSA; its digest; PSS's mask digest and salt length.</summary>
internal readonly record struct CmsScheme(CmsSchemeKind Kind, HashAlgorithmName Digest, HashAlgorithmName MaskDigest = default, int SaltLength = 0)
{
    /// <summary>The scheme's name for messages ("RSA PKCS #1 v1.5 with SHA-256").</summary>
    public override string ToString() => Kind switch
    {
        CmsSchemeKind.RsaPkcs1 => $"RSA PKCS #1 v1.5 with {CmsDigest.Named(Digest)}",
        CmsSchemeKind.RsaPss => $"RSA-PSS with {CmsDigest.Named(Digest)} (MGF1 {CmsDigest.Named(MaskDigest)}, a {SaltLength}-byte salt)",
        _ => $"ECDSA with {CmsDigest.Named(Digest)}",
    };
}

internal enum CmsSchemeKind : byte
{
    RsaPkcs1,
    RsaPss,
    Ecdsa,
}

/// <summary>
/// Signatures verified and made. RSA signatures are verified here, on <see cref="BigInteger"/> (RFC 8017's public
/// operation, then EMSA-PKCS1-v1_5 or EMSA-PSS checked), so every platform gives the same answer for every digest (SHA-1's
/// legacy forms too, which some platforms' policies refuse) and every PSS salt length; ECDSA through the base library's
/// <see cref="ECDsa"/>. Signing uses the app's key as the base library holds it (a key in a store signs where it is).
/// </summary>
internal static class CmsKeys
{
    // The DigestInfo prefixes of RFC 8017 9.2, note 1, with the NULL parameters and without (both are read).
    private static readonly byte[] Sha1Info = Convert.FromHexString("3021300906052B0E03021A05000414");
    private static readonly byte[] Sha256Info = Convert.FromHexString("3031300D060960864801650304020105000420");
    private static readonly byte[] Sha384Info = Convert.FromHexString("3041300D060960864801650304020205000430");
    private static readonly byte[] Sha512Info = Convert.FromHexString("3051300D060960864801650304020305000440");
    private static readonly byte[] Sha1InfoBare = Convert.FromHexString("301F300706052B0E03021A0414");
    private static readonly byte[] Sha256InfoBare = Convert.FromHexString("302F300B06096086480165030402010420");
    private static readonly byte[] Sha384InfoBare = Convert.FromHexString("303F300B06096086480165030402020430");
    private static readonly byte[] Sha512InfoBare = Convert.FromHexString("304F300B06096086480165030402030440");

    /// <summary>
    /// The scheme <paramref name="algorithm"/> names (a signer's signatureAlgorithm), its digest from
    /// <paramref name="digest"/> when the algorithm names only the key (rsaEncryption, id-ecPublicKey); or why it is not one
    /// read here.
    /// </summary>
    public static CmsScheme? Scheme(CmsAlgorithm algorithm, HashAlgorithmName? digest, out string? problem)
    {
        problem = null;
        switch (algorithm.Oid)
        {
            case CmsOids.RsaEncryption:
                return digest is { } d ? new CmsScheme(CmsSchemeKind.RsaPkcs1, d) : Fail("rsaEncryption without a digest", out problem);
            case CmsOids.Sha1WithRsa or CmsOids.Sha1WithRsaOiw:
                return new CmsScheme(CmsSchemeKind.RsaPkcs1, HashAlgorithmName.SHA1);
            case CmsOids.Sha256WithRsa:
                return new CmsScheme(CmsSchemeKind.RsaPkcs1, HashAlgorithmName.SHA256);
            case CmsOids.Sha384WithRsa:
                return new CmsScheme(CmsSchemeKind.RsaPkcs1, HashAlgorithmName.SHA384);
            case CmsOids.Sha512WithRsa:
                return new CmsScheme(CmsSchemeKind.RsaPkcs1, HashAlgorithmName.SHA512);
            case CmsOids.RsaPss:
                return Pss(algorithm.Parameters, out problem);
            case CmsOids.EcPublicKey:
                return digest is { } e ? new CmsScheme(CmsSchemeKind.Ecdsa, e) : Fail("id-ecPublicKey without a digest", out problem);
            case CmsOids.EcdsaSha1:
                return new CmsScheme(CmsSchemeKind.Ecdsa, HashAlgorithmName.SHA1);
            case CmsOids.EcdsaSha256:
                return new CmsScheme(CmsSchemeKind.Ecdsa, HashAlgorithmName.SHA256);
            case CmsOids.EcdsaSha384:
                return new CmsScheme(CmsSchemeKind.Ecdsa, HashAlgorithmName.SHA384);
            case CmsOids.EcdsaSha512:
                return new CmsScheme(CmsSchemeKind.Ecdsa, HashAlgorithmName.SHA512);
            default:
                return Fail($"the signature algorithm {algorithm.Oid}, which is not read (RSA PKCS #1 v1.5, RSA-PSS and ECDSA with SHA-1 or SHA-2 are)", out problem);
        }
    }

    private static CmsScheme? Fail(string why, out string problem)
    {
        problem = why;
        return null;
    }

    // RSASSA-PSS-params (RFC 8017, A.2.3): the digest [0] (SHA-1), the mask [1] (MGF1 with SHA-1), the salt [2] (20), the trailer [3] (1).
    private static CmsScheme? Pss(ReadOnlyMemory<byte> parameters, out string? problem)
    {
        problem = null;
        var digest = HashAlgorithmName.SHA1;
        var mask = HashAlgorithmName.SHA1;
        int salt = 20;
        try
        {
            if (!parameters.IsEmpty)
            {
                var sequence = CmsAsn.Reader(parameters).ReadSequence();
                if (CmsAsn.Next(sequence, CmsAsn.Context0))
                {
                    var algorithm = CmsAlgorithm.Read(sequence.ReadSequence(CmsAsn.Context0));
                    digest = CmsDigest.FromOid(algorithm.Oid) ?? throw new NotSupportedException($"RSA-PSS with the digest {algorithm.Oid}");
                }

                if (CmsAsn.Next(sequence, CmsAsn.Context1))
                {
                    var generator = CmsAlgorithm.Read(sequence.ReadSequence(CmsAsn.Context1));
                    if (generator.Oid != CmsOids.Mgf1)
                    {
                        throw new NotSupportedException($"RSA-PSS with the mask {generator.Oid} (MGF1 is read)");
                    }

                    var inner = CmsAlgorithm.Read(CmsAsn.Reader(generator.Parameters));
                    mask = CmsDigest.FromOid(inner.Oid) ?? throw new NotSupportedException($"RSA-PSS with MGF1 over {inner.Oid}");
                }

                if (CmsAsn.Next(sequence, CmsAsn.Context2))
                {
                    salt = (int)sequence.ReadSequence(CmsAsn.Context2).ReadInteger();
                }

                if (CmsAsn.Next(sequence, CmsAsn.Context3) && sequence.ReadSequence(CmsAsn.Context3).ReadInteger() != 1)
                {
                    throw new NotSupportedException("RSA-PSS with a trailer field other than 1");
                }
            }
        }
        catch (NotSupportedException e)
        {
            problem = e.Message;
            return null;
        }
        catch (Exception e) when (CmsAsn.IsDamage(e))
        {
            problem = $"RSA-PSS parameters that are not valid DER ({e.Message})";
            return null;
        }

        if (salt is < 0 or > 1024)
        {
            problem = $"RSA-PSS with a {salt}-byte salt";
            return null;
        }

        return new CmsScheme(CmsSchemeKind.RsaPss, digest, mask, salt);
    }

    /// <summary>Whether <paramref name="signature"/> is <paramref name="key"/>'s signature of the digest <paramref name="hash"/>; null when it is, else why not.</summary>
    public static string? Verify(CmsCertificate key, CmsScheme scheme, ReadOnlySpan<byte> hash, ReadOnlySpan<byte> signature)
    {
        if (scheme.Kind == CmsSchemeKind.Ecdsa)
        {
            if (key.KeyAlgorithm.Oid != CmsOids.EcPublicKey)
            {
                return $"an ECDSA signature, but {key.Named}'s key is {key.KeyAlgorithm.Oid}";
            }

            try
            {
                using var ecdsa = ECDsa.Create();
                ecdsa.ImportSubjectPublicKeyInfo(key.PublicKeyInfo.Span, out _);
                return ecdsa.VerifyHash(hash, signature, DSASignatureFormat.Rfc3279DerSequence) ? null : $"the {scheme} signature does not verify with {key.Named}'s key";
            }
            catch (CryptographicException e)
            {
                return $"{key.Named}'s EC key cannot be used here ({e.Message})";
            }
        }

        if (key.Rsa() is not { } rsa)
        {
            return $"an RSA signature, but {key.Named}'s key is not an RSA key ({key.KeyAlgorithm.Oid})";
        }

        var encoded = PublicOperation(rsa, signature, out string? problem);
        if (encoded is null)
        {
            return problem;
        }

        return scheme.Kind == CmsSchemeKind.RsaPkcs1
            ? (Pkcs1Matches(encoded, scheme.Digest, hash) ? null : $"the {scheme} signature does not verify with {key.Named}'s key")
            : (PssMatches(encoded, rsa.Bits, scheme, hash) ? null : $"the {scheme} signature does not verify with {key.Named}'s key");
    }

    /// <summary>
    /// The digest an RSA PKCS #1 v1.5 signature carries (its DigestInfo opened with <paramref name="key"/>'s key), or null
    /// when the key does not open it to a DigestInfo of a digest read here: tells a signature that verifies over other
    /// bytes from one that does not verify at all (the legacy PKCS #1 form, whose signature is its only digest).
    /// </summary>
    public static (HashAlgorithmName Digest, byte[] Hash)? Recover(CmsCertificate key, ReadOnlySpan<byte> signature)
    {
        if (key.Rsa() is not { } rsa || PublicOperation(rsa, signature, out _) is not { } encoded)
        {
            return null;
        }

        // 00 01 FF..FF 00 DigestInfo
        if (encoded.Length < 11 || encoded[0] != 0 || encoded[1] != 1)
        {
            return null;
        }

        int at = 2;
        while (at < encoded.Length && encoded[at] == 0xFF)
        {
            at++;
        }

        if (at - 2 < 8 || at >= encoded.Length || encoded[at] != 0)
        {
            return null;
        }

        var info = encoded.AsMemory(at + 1);
        try
        {
            var sequence = CmsAsn.Reader(info).ReadSequence();
            var algorithm = CmsAlgorithm.Read(sequence);
            byte[] hash = sequence.ReadOctetString();
            return CmsDigest.FromOid(algorithm.Oid) is { } digest && hash.Length == CmsDigest.Length(digest) ? (digest, hash) : null;
        }
        catch (Exception e) when (CmsAsn.IsDamage(e))
        {
            return null;
        }
    }

    // RSAVP1 (RFC 8017, 5.2.2) and I2OSP into the modulus's length: s^e mod n, or null when s is not below n.
    private static byte[]? PublicOperation((BigInteger Modulus, BigInteger Exponent, int Bits) rsa, ReadOnlySpan<byte> signature, out string? problem)
    {
        problem = null;
        int length = (rsa.Bits + 7) / 8;
        if (signature.Length != length)
        {
            problem = $"an RSA signature of {signature.Length} bytes for a {rsa.Bits}-bit key ({length} bytes)";
            return null;
        }

        var s = new BigInteger(signature, isUnsigned: true, isBigEndian: true);
        if (s >= rsa.Modulus)
        {
            problem = "an RSA signature not below its key's modulus";
            return null;
        }

        var m = BigInteger.ModPow(s, rsa.Exponent, rsa.Modulus);
        byte[] encoded = new byte[length];
        int size = m.GetByteCount(isUnsigned: true);
        m.TryWriteBytes(encoded.AsSpan(length - size), out _, isUnsigned: true, isBigEndian: true);
        return encoded;
    }

    // EMSA-PKCS1-v1_5 (RFC 8017, 9.2): 00 01 PS 00 DigestInfo, with or without the NULL parameters.
    private static bool Pkcs1Matches(ReadOnlySpan<byte> encoded, HashAlgorithmName digest, ReadOnlySpan<byte> hash)
    {
        var (info, bare) = digest == HashAlgorithmName.SHA1 ? (Sha1Info, Sha1InfoBare)
            : digest == HashAlgorithmName.SHA384 ? (Sha384Info, Sha384InfoBare)
            : digest == HashAlgorithmName.SHA512 ? (Sha512Info, Sha512InfoBare)
            : (Sha256Info, Sha256InfoBare);
        return hash.Length == CmsDigest.Length(digest) && (Pkcs1Matches(encoded, info, hash) || Pkcs1Matches(encoded, bare, hash));
    }

    private static bool Pkcs1Matches(ReadOnlySpan<byte> encoded, ReadOnlySpan<byte> info, ReadOnlySpan<byte> hash)
    {
        int tail = info.Length + hash.Length;
        int padding = encoded.Length - tail - 3;
        if (padding < 8 || encoded[0] != 0 || encoded[1] != 1 || encoded[2 + padding] != 0)
        {
            return false;
        }

        foreach (byte b in encoded.Slice(2, padding))
        {
            if (b != 0xFF)
            {
                return false;
            }
        }

        return encoded.Slice(3 + padding, info.Length).SequenceEqual(info) && encoded[^hash.Length..].SequenceEqual(hash);
    }

    // EMSA-PSS-VERIFY (RFC 8017, 9.1.2) over the message's digest.
    private static bool PssMatches(byte[] encoded, int bits, CmsScheme scheme, ReadOnlySpan<byte> hash)
    {
        int emBits = bits - 1, emLength = (emBits + 7) / 8;
        int hashLength = CmsDigest.Length(scheme.Digest), salt = scheme.SaltLength;
        var em = encoded.AsSpan(encoded.Length - emLength);
        if (encoded.Length > emLength && encoded[0] != 0)
        {
            return false;                                                      // the byte above emBits is not zero
        }

        if (hash.Length != hashLength || emLength < hashLength + salt + 2 || em[^1] != 0xBC)
        {
            return false;
        }

        int dbLength = emLength - hashLength - 1;
        var masked = em[..dbLength];
        var h = em.Slice(dbLength, hashLength);
        int topBits = 8 * emLength - emBits;
        if ((masked[0] & (0xFF << (8 - topBits)) & 0xFF) != 0)
        {
            return false;
        }

        byte[] db = Mgf1(h, dbLength, scheme.MaskDigest);
        for (int i = 0; i < dbLength; i++)
        {
            db[i] ^= masked[i];
        }

        db[0] &= (byte)(0xFF >> topBits);
        int zeros = emLength - hashLength - salt - 2;
        for (int i = 0; i < zeros; i++)
        {
            if (db[i] != 0)
            {
                return false;
            }
        }

        if (db[zeros] != 1)
        {
            return false;
        }

        byte[] prime = new byte[8 + hashLength + salt];
        hash.CopyTo(prime.AsSpan(8));
        db.AsSpan(dbLength - salt).CopyTo(prime.AsSpan(8 + hashLength));
        return CmsDigest.Of(scheme.Digest, prime).AsSpan().SequenceEqual(h);
    }

    // MGF1 (RFC 8017, B.2.1).
    private static byte[] Mgf1(ReadOnlySpan<byte> seed, int length, HashAlgorithmName digest)
    {
        int hashLength = CmsDigest.Length(digest);
        byte[] mask = new byte[length];
        byte[] block = new byte[seed.Length + 4];
        seed.CopyTo(block);
        for (int counter = 0, at = 0; at < length; counter++, at += hashLength)
        {
            System.Buffers.Binary.BinaryPrimitives.WriteInt32BigEndian(block.AsSpan(seed.Length), counter);
            byte[] part = CmsDigest.Of(digest, block);
            part.AsSpan(0, Math.Min(hashLength, length - at)).CopyTo(mask.AsSpan(at));
        }

        return mask;
    }

    /// <summary>
    /// Signs the digest <paramref name="hash"/> with <paramref name="certificate"/>'s private key: RSA with
    /// <paramref name="padding"/>, or ECDSA (DER, as CMS holds it); the signature, and the signatureAlgorithm it is written under.
    /// </summary>
    /// <exception cref="ArgumentException">The certificate has no private key, or one that is neither RSA nor EC.</exception>
    public static (byte[] Signature, CmsScheme Scheme) Sign(X509Certificate2 certificate, HashAlgorithmName digest, RSASignaturePadding padding, ReadOnlySpan<byte> hash)
    {
        using (var rsa = certificate.GetRSAPrivateKey())
        {
            if (rsa is not null)
            {
                byte[] signature = rsa.SignHash(hash, digest, padding);
                return (signature, padding == RSASignaturePadding.Pss
                    ? new CmsScheme(CmsSchemeKind.RsaPss, digest, digest, CmsDigest.Length(digest))
                    : new CmsScheme(CmsSchemeKind.RsaPkcs1, digest));
            }
        }

        using (var ecdsa = certificate.GetECDsaPrivateKey())
        {
            if (ecdsa is not null)
            {
                return (ecdsa.SignHash(hash, DSASignatureFormat.Rfc3279DerSequence), new CmsScheme(CmsSchemeKind.Ecdsa, digest));
            }
        }

        throw new ArgumentException(certificate.HasPrivateKey
            ? $"The signing certificate '{certificate.Subject}' has a private key that is neither RSA nor EC (SigningOptions.Certificate)."
            : $"The signing certificate '{certificate.Subject}' has no private key (SigningOptions.Certificate: a certificate with its key).", "options");
    }

    /// <summary>Writes the signatureAlgorithm a CMS signer is written under for <paramref name="scheme"/>.</summary>
    public static void WriteAlgorithm(AsnWriter writer, CmsScheme scheme)
    {
        switch (scheme.Kind)
        {
            case CmsSchemeKind.RsaPkcs1:
                string oid = scheme.Digest == HashAlgorithmName.SHA384 ? CmsOids.Sha384WithRsa
                    : scheme.Digest == HashAlgorithmName.SHA512 ? CmsOids.Sha512WithRsa
                    : scheme.Digest == HashAlgorithmName.SHA1 ? CmsOids.Sha1WithRsa
                    : CmsOids.Sha256WithRsa;
                CmsAsn.WriteAlgorithm(writer, oid, nullParameters: true);
                break;
            case CmsSchemeKind.RsaPss:
                using (writer.PushSequence())
                {
                    writer.WriteObjectIdentifier(CmsOids.RsaPss);
                    using (writer.PushSequence())
                    {
                        using (writer.PushSequence(CmsAsn.Context0))
                        {
                            CmsAsn.WriteAlgorithm(writer, CmsDigest.Oid(scheme.Digest));
                        }

                        using (writer.PushSequence(CmsAsn.Context1))
                        {
                            using (writer.PushSequence())
                            {
                                writer.WriteObjectIdentifier(CmsOids.Mgf1);
                                CmsAsn.WriteAlgorithm(writer, CmsDigest.Oid(scheme.MaskDigest));
                            }
                        }

                        using (writer.PushSequence(CmsAsn.Context2))
                        {
                            writer.WriteInteger(scheme.SaltLength);
                        }
                    }
                }

                break;
            default:
                string ecdsa = scheme.Digest == HashAlgorithmName.SHA384 ? CmsOids.EcdsaSha384
                    : scheme.Digest == HashAlgorithmName.SHA512 ? CmsOids.EcdsaSha512
                    : scheme.Digest == HashAlgorithmName.SHA1 ? CmsOids.EcdsaSha1
                    : CmsOids.EcdsaSha256;
                CmsAsn.WriteAlgorithm(writer, ecdsa);
                break;
        }
    }

    /// <summary>The longest signature <paramref name="certificate"/>'s key makes (to reserve room for it).</summary>
    public static int MostSignatureBytes(X509Certificate2 certificate)
    {
        using (var rsa = certificate.GetRSAPublicKey())
        {
            if (rsa is not null)
            {
                return (rsa.KeySize + 7) / 8;
            }
        }

        using var ecdsa = certificate.GetECDsaPublicKey();
        return ecdsa is null ? 1024 : 2 * ((ecdsa.KeySize + 7) / 8) + 16;      // two integers, their tags and lengths
    }
}
