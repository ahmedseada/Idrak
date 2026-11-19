// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Formats.Asn1;
using System.Globalization;
using System.Numerics;
using System.Security.Cryptography;

namespace Idrak.Data.Pdf;

/// <summary>
/// The object identifiers signatures use (RFC 5652 CMS, RFC 5280 certificates and revocation lists, RFC 6960 OCSP,
/// RFC 3161 timestamps, RFC 5035 and RFC 2634 signing certificates, RFC 8017 RSA, RFC 5758 ECDSA).
/// </summary>
internal static class CmsOids
{
    public const string Data = "1.2.840.113549.1.7.1", SignedData = "1.2.840.113549.1.7.2", TstInfo = "1.2.840.113549.1.9.16.1.4";

    public const string ContentType = "1.2.840.113549.1.9.3", MessageDigest = "1.2.840.113549.1.9.4", SigningTime = "1.2.840.113549.1.9.5";

    public const string SigningCertificate = "1.2.840.113549.1.9.16.2.12", SigningCertificateV2 = "1.2.840.113549.1.9.16.2.47";

    public const string SignatureTimeStampToken = "1.2.840.113549.1.9.16.2.14", AdobeRevocation = "1.2.840.113583.1.1.8";

    public const string Sha1 = "1.3.14.3.2.26", Sha256 = "2.16.840.1.101.3.4.2.1", Sha384 = "2.16.840.1.101.3.4.2.2", Sha512 = "2.16.840.1.101.3.4.2.3";

    public const string RsaEncryption = "1.2.840.113549.1.1.1", RsaPss = "1.2.840.113549.1.1.10", Mgf1 = "1.2.840.113549.1.1.8";

    public const string Sha1WithRsa = "1.2.840.113549.1.1.5", Sha256WithRsa = "1.2.840.113549.1.1.11", Sha384WithRsa = "1.2.840.113549.1.1.12",
        Sha512WithRsa = "1.2.840.113549.1.1.13", Sha1WithRsaOiw = "1.3.14.3.2.29";

    public const string EcPublicKey = "1.2.840.10045.2.1", EcdsaSha1 = "1.2.840.10045.4.1", EcdsaSha256 = "1.2.840.10045.4.3.2",
        EcdsaSha384 = "1.2.840.10045.4.3.3", EcdsaSha512 = "1.2.840.10045.4.3.4";

    public const string BasicConstraints = "2.5.29.19", KeyUsage = "2.5.29.15", ExtendedKeyUsage = "2.5.29.37", SubjectKeyId = "2.5.29.14",
        AuthorityKeyId = "2.5.29.35", CertificatePolicies = "2.5.29.32", SubjectAltName = "2.5.29.17", IssuerAltName = "2.5.29.18",
        CrlDistributionPoints = "2.5.29.31", AuthorityInfoAccess = "1.3.6.1.5.5.7.1.1", SubjectInfoAccess = "1.3.6.1.5.5.7.1.11",
        FreshestCrl = "2.5.29.46", PrivateKeyUsagePeriod = "2.5.29.16";

    public const string NameConstraints = "2.5.29.30", PolicyMappings = "2.5.29.33", PolicyConstraints = "2.5.29.36", InhibitAnyPolicy = "2.5.29.54",
        AnyPolicy = "2.5.29.32.0", EmailAddress = "1.2.840.113549.1.9.1";

    public const string CrlNumber = "2.5.29.20", CrlReason = "2.5.29.21", InvalidityDate = "2.5.29.24", DeltaCrlIndicator = "2.5.29.27",
        IssuingDistributionPoint = "2.5.29.28", CertificateIssuer = "2.5.29.29";

    public const string OcspAccess = "1.3.6.1.5.5.7.48.1", OcspBasic = "1.3.6.1.5.5.7.48.1.1", OcspNonce = "1.3.6.1.5.5.7.48.1.2",
        OcspNoCheck = "1.3.6.1.5.5.7.48.1.5", OcspArchiveCutoff = "1.3.6.1.5.5.7.48.1.6", OcspCrlId = "1.3.6.1.5.5.7.48.1.3";

    public const string TimeStamping = "1.3.6.1.5.5.7.3.8", OcspSigning = "1.3.6.1.5.5.7.3.9", AnyExtendedKeyUsage = "2.5.29.37.0";
}

/// <summary>An algorithm identifier: its object identifier and its parameters' DER (empty when absent).</summary>
internal readonly record struct CmsAlgorithm(string Oid, ReadOnlyMemory<byte> Parameters)
{
    public static CmsAlgorithm Read(AsnReader reader)
    {
        var sequence = reader.ReadSequence();
        string oid = sequence.ReadObjectIdentifier();
        var parameters = sequence.HasData ? sequence.ReadEncodedValue() : ReadOnlyMemory<byte>.Empty;
        return new CmsAlgorithm(oid, parameters);
    }

    /// <summary>Whether the parameters are absent or NULL.</summary>
    public bool NoParameters => Parameters.IsEmpty || Parameters.Span is [0x05, 0x00];
}

/// <summary>The digests signatures name: SHA-1, SHA-256, SHA-384 and SHA-512 (through the base library), by object identifier.</summary>
internal static class CmsDigest
{
    /// <summary>The digest <paramref name="oid"/> names, or null when it is not one read here.</summary>
    public static HashAlgorithmName? FromOid(string oid) => oid switch
    {
        CmsOids.Sha1 => HashAlgorithmName.SHA1,
        CmsOids.Sha256 => HashAlgorithmName.SHA256,
        CmsOids.Sha384 => HashAlgorithmName.SHA384,
        CmsOids.Sha512 => HashAlgorithmName.SHA512,
        _ => null,
    };

    /// <summary>The object identifier of <paramref name="digest"/>.</summary>
    /// <exception cref="NotSupportedException">Not a digest written here.</exception>
    public static string Oid(HashAlgorithmName digest) =>
        digest == HashAlgorithmName.SHA256 ? CmsOids.Sha256
        : digest == HashAlgorithmName.SHA384 ? CmsOids.Sha384
        : digest == HashAlgorithmName.SHA512 ? CmsOids.Sha512
        : digest == HashAlgorithmName.SHA1 ? CmsOids.Sha1
        : throw new NotSupportedException($"The digest {digest.Name} is not one a signature is made with here (SHA-256, SHA-384, SHA-512; SHA-1 is read only).");

    /// <summary>The bytes a digest is long.</summary>
    public static int Length(HashAlgorithmName digest) =>
        digest == HashAlgorithmName.SHA1 ? 20 : digest == HashAlgorithmName.SHA384 ? 48 : digest == HashAlgorithmName.SHA512 ? 64 : 32;

    /// <summary>The digest of <paramref name="data"/>.</summary>
    public static byte[] Of(HashAlgorithmName digest, ReadOnlySpan<byte> data) =>
        digest == HashAlgorithmName.SHA1 ? SHA1.HashData(data)
        : digest == HashAlgorithmName.SHA384 ? SHA384.HashData(data)
        : digest == HashAlgorithmName.SHA512 ? SHA512.HashData(data)
        : SHA256.HashData(data);

    /// <summary>The digest's name for messages ("SHA-256").</summary>
    public static string Named(HashAlgorithmName digest) => digest.Name switch
    {
        "SHA1" => "SHA-1",
        "SHA256" => "SHA-256",
        "SHA384" => "SHA-384",
        "SHA512" => "SHA-512",
        var other => other ?? "none",
    };
}

/// <summary>Small readers and writers over <see cref="System.Formats.Asn1"/> the signature code shares.</summary>
internal static class CmsAsn
{
    public static readonly Asn1Tag Context0 = new(TagClass.ContextSpecific, 0, isConstructed: true);
    public static readonly Asn1Tag Context1 = new(TagClass.ContextSpecific, 1, isConstructed: true);
    public static readonly Asn1Tag Context2 = new(TagClass.ContextSpecific, 2, isConstructed: true);
    public static readonly Asn1Tag Context3 = new(TagClass.ContextSpecific, 3, isConstructed: true);

    /// <summary>A reader over <paramref name="data"/> taking BER (what signers write; DER is a part of it).</summary>
    public static AsnReader Reader(ReadOnlyMemory<byte> data) => new(data, AsnEncodingRules.BER);

    /// <summary>A UTCTime (years 1950-2049, as X.509 reads them) or a GeneralizedTime.</summary>
    public static DateTimeOffset ReadTime(AsnReader reader)
    {
        var tag = reader.PeekTag();
        return tag.HasSameClassAndValue(Asn1Tag.UtcTime) ? reader.ReadUtcTime(2049) : reader.ReadGeneralizedTime();
    }

    /// <summary>An ENUMERATED's value (a reason code, a response status).</summary>
    public static int ReadEnumerated(AsnReader reader) => (int)new BigInteger(reader.ReadEnumeratedBytes().Span, isUnsigned: false, isBigEndian: true);

    /// <summary>Whether the next element has <paramref name="tag"/>.</summary>
    public static bool Next(AsnReader reader, Asn1Tag tag) => reader.HasData && reader.PeekTag().HasSameClassAndValue(tag);

    /// <summary>The element's encoding with its tag replaced by a universal SET (a CMS signed attributes' [0] as it is signed).</summary>
    public static byte[] AsSet(ReadOnlyMemory<byte> encoded)
    {
        byte[] copy = encoded.ToArray();
        copy[0] = 0x31;
        return copy;
    }

    /// <summary>Writes an algorithm identifier with no parameters (or with NULL ones).</summary>
    public static void WriteAlgorithm(AsnWriter writer, string oid, bool nullParameters = false)
    {
        using (writer.PushSequence())
        {
            writer.WriteObjectIdentifier(oid);
            if (nullParameters)
            {
                writer.WriteNull();
            }
        }
    }

    /// <summary>Wraps an <see cref="AsnContentException"/> or a <see cref="CryptographicException"/> as damage named by <paramref name="what"/>.</summary>
    public static InvalidDataException Damaged(string what, Exception e) => new($"{what} is not valid DER ({e.Message})", e);

    /// <summary>Whether <paramref name="e"/> is what a reader of damaged DER throws.</summary>
    public static bool IsDamage(Exception e) => e is AsnContentException or CryptographicException or InvalidDataException or ArgumentException or OverflowException;
}

/// <summary>
/// A distinguished name as text, written here the same on every platform (RFC 4514's order: the last attribute first;
/// CN, O, OU, C, L, ST, DC, the e-mail address and the serial number by their short names, others by their identifiers).
/// </summary>
internal static class CmsName
{
    public static string Describe(ReadOnlyMemory<byte> name)
    {
        try
        {
            var parts = new List<string>();
            var sequence = CmsAsn.Reader(name).ReadSequence();
            while (sequence.HasData)
            {
                var set = sequence.ReadSetOf(skipSortOrderValidation: true);
                while (set.HasData)
                {
                    var attribute = set.ReadSequence();
                    string oid = attribute.ReadObjectIdentifier();
                    string value = Value(attribute);
                    parts.Add($"{Short(oid)}={value}");
                }
            }

            parts.Reverse();
            return parts.Count == 0 ? "(an empty name)" : string.Join(", ", parts);
        }
        catch (Exception e) when (CmsAsn.IsDamage(e))
        {
            return "(a damaged name)";
        }
    }

    /// <summary>The name's common name (CN), or the whole name when it has none.</summary>
    public static string Common(ReadOnlyMemory<byte> name)
    {
        string described = Describe(name);
        foreach (string part in described.Split(", "))
        {
            if (part.StartsWith("CN=", StringComparison.Ordinal))
            {
                return part[3..];
            }
        }

        return described;
    }

    private static string Value(AsnReader attribute)
    {
        var tag = attribute.PeekTag();
        if (tag.TagClass == TagClass.Universal)
        {
            switch ((UniversalTagNumber)tag.TagValue)
            {
                case UniversalTagNumber.UTF8String or UniversalTagNumber.PrintableString or UniversalTagNumber.IA5String or UniversalTagNumber.BMPString
                    or UniversalTagNumber.NumericString or UniversalTagNumber.VisibleString or UniversalTagNumber.T61String:
                    return attribute.ReadCharacterString((UniversalTagNumber)tag.TagValue);
            }
        }

        return "#" + Convert.ToHexString(attribute.ReadEncodedValue().Span);
    }

    private static string Short(string oid) => oid switch
    {
        "2.5.4.3" => "CN",
        "2.5.4.10" => "O",
        "2.5.4.11" => "OU",
        "2.5.4.6" => "C",
        "2.5.4.7" => "L",
        "2.5.4.8" => "ST",
        "2.5.4.5" => "SERIALNUMBER",
        "0.9.2342.19200300.100.1.25" => "DC",
        "1.2.840.113549.1.9.1" => "E",
        _ => oid,
    };
}

/// <summary>Times as signature messages give them: UTC, to the second, invariant.</summary>
internal static class CmsText
{
    public static string Time(DateTimeOffset time) => time.ToUniversalTime().ToString("yyyy-MM-dd HH:mm:ss 'UTC'", CultureInfo.InvariantCulture);

    public static string Serial(ReadOnlySpan<byte> serial) => Convert.ToHexString(serial);

    /// <summary>A span of time for messages: whole days, hours, minutes or seconds ("7 days", "90 minutes"), invariant.</summary>
    public static string Span(TimeSpan span) =>
        span.Ticks % TimeSpan.TicksPerDay == 0 ? Unit(span.Ticks / TimeSpan.TicksPerDay, "day")
        : span.Ticks % TimeSpan.TicksPerHour == 0 ? Unit(span.Ticks / TimeSpan.TicksPerHour, "hour")
        : span.Ticks % TimeSpan.TicksPerMinute == 0 ? Unit(span.Ticks / TimeSpan.TicksPerMinute, "minute")
        : Unit((long)span.TotalSeconds, "second");

    private static string Unit(long count, string unit) => string.Create(CultureInfo.InvariantCulture, $"{count} {unit}{(count == 1 ? "" : "s")}");
}
