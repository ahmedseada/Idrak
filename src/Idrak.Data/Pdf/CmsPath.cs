// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Data.Pdf;

/// <summary>What a certificate at the start of a path is checked for.</summary>
internal enum CmsUsage : byte
{
    /// <summary>A document's signer: a key that may make digital signatures (or non-repudiation ones).</summary>
    Signer,

    /// <summary>A timestamp authority: the timeStamping purpose.</summary>
    Timestamp,
}

/// <summary>
/// Certification paths built and checked here (RFC 5280, 6.1's basic path validation), the same on every platform and with
/// nothing fetched: from a certificate, through issuers found among the certificates given (the signature's, the document's
/// validation data), to a certificate the app trusts (decision 16: only those; never one the document carries). Each link's
/// signature, each certificate's validity at the time, the CA's basic constraints, path length and key usage, the first
/// certificate's usage, and critical extensions this does not read (which fail the path, naming them).
/// </summary>
internal static class CmsPath
{
    private const int MaxLength = 12;

    // Extensions a path is checked by, or whose meaning does not limit a signature's path; any other critical one fails it.
    private static readonly HashSet<string> Known =
    [
        CmsOids.BasicConstraints, CmsOids.KeyUsage, CmsOids.ExtendedKeyUsage, CmsOids.SubjectKeyId, CmsOids.AuthorityKeyId, CmsOids.CertificatePolicies,
        CmsOids.SubjectAltName, CmsOids.IssuerAltName, CmsOids.CrlDistributionPoints, CmsOids.AuthorityInfoAccess, CmsOids.SubjectInfoAccess,
        CmsOids.FreshestCrl, CmsOids.PrivateKeyUsagePeriod, CmsOids.OcspNoCheck,
    ];

    /// <summary>Whether <paramref name="issuer"/>'s key verifies <paramref name="certificate"/>'s signature; null when it does, else why not.</summary>
    public static string? VerifiedBy(CmsCertificate certificate, CmsCertificate issuer)
    {
        var scheme = CmsKeys.Scheme(certificate.SignatureAlgorithm, null, out string? problem);
        return scheme is { } s ? CmsKeys.Verify(issuer, s, CmsDigest.Of(s.Digest, certificate.Tbs.Span), certificate.Signature) : $"{certificate.Named}'s certificate is signed with {problem}";
    }

    /// <summary>
    /// A path from <paramref name="first"/> to one of <paramref name="anchors"/> through <paramref name="pool"/>, checked at
    /// <paramref name="at"/>: the path (first to anchor) and what was found, or null and why none is valid.
    /// </summary>
    public static (List<CmsCertificate>? Path, string Detail) Build(CmsCertificate first, IReadOnlyList<CmsCertificate> pool, IReadOnlyList<CmsCertificate> anchors,
        DateTimeOffset at, CmsUsage usage)
    {
        if (anchors.Count == 0)
        {
            return (null, "no certificate is trusted: a check trusts only the certificates the app gives (SignatureCheckOptions.Trusted, or the machine's store with UseMachineStore)");
        }

        string? best = null;
        var path = new List<CmsCertificate> { first };
        var found = Walk(path, pool, anchors, at, usage, ref best);
        if (found is not null)
        {
            string chain = string.Join(" < ", found.Select(c => c.Named));
            return (found, $"{chain}, trusted, valid at {CmsText.Time(at)}");
        }

        return (null, best ?? $"no path from {first.Named} to a trusted certificate: its issuer '{first.IssuerName}' is neither trusted nor among the certificates the signature and the document carry");
    }

    // Depth first: each issuer the names (and key identifiers) allow, whose key verifies the link, until a trusted one;
    // the first whole path that checks is taken, else the reason the deepest attempt failed.
    private static List<CmsCertificate>? Walk(List<CmsCertificate> path, IReadOnlyList<CmsCertificate> pool, IReadOnlyList<CmsCertificate> anchors, DateTimeOffset at,
        CmsUsage usage, ref string? best)
    {
        var current = path[^1];
        if (anchors.Any(a => a.Same(current)))
        {
            string? failed = Check(path, at, usage);
            if (failed is null)
            {
                return [.. path];
            }

            best ??= failed;
            return null;
        }

        if (path.Count > MaxLength)
        {
            best ??= $"a path longer than {MaxLength} certificates";
            return null;
        }

        foreach (bool trusted in (ReadOnlySpan<bool>)[true, false])
        {
            foreach (var candidate in trusted ? anchors : pool)
            {
                if (!current.NamesIssuer(candidate) || path.Exists(c => c.Same(candidate)) || (!trusted && anchors.Any(a => a.Same(candidate))))
                {
                    continue;
                }

                if (VerifiedBy(current, candidate) is { } broken)
                {
                    best ??= $"{current.Named}'s certificate does not verify with {candidate.Named}'s key: {broken}";
                    continue;
                }

                path.Add(candidate);
                var found = Walk(path, pool, anchors, at, usage, ref best);
                path.RemoveAt(path.Count - 1);
                if (found is not null)
                {
                    return found;
                }
            }
        }

        if (current.SelfIssued && best is null)
        {
            best = $"{current.Named}'s certificate is a root that is not trusted (the check trusts only the certificates the app gives)";
        }

        return null;
    }

    // The basic path checks (RFC 5280, 6.1.3 and 6.1.4) on a path from its first certificate to its anchor; null when it passes.
    private static string? Check(List<CmsCertificate> path, DateTimeOffset at, CmsUsage usage)
    {
        for (int i = 0; i < path.Count; i++)
        {
            var certificate = path[i];
            bool anchor = i == path.Count - 1 && path.Count > 1;
            if (!anchor && (at < certificate.NotBefore || at > certificate.NotAfter))
            {
                return $"{certificate.Named}'s certificate is not valid at {CmsText.Time(at)} (from {CmsText.Time(certificate.NotBefore)} to {CmsText.Time(certificate.NotAfter)})";
            }

            foreach (var (oid, critical, _) in certificate.Extensions)
            {
                if (critical && !Known.Contains(oid))
                {
                    return $"{certificate.Named}'s certificate has the critical extension {oid}, which this check does not read";
                }
            }

            if (i == 0)
            {
                continue;
            }

            if (certificate.IsCa != true)
            {
                return $"{certificate.Named}'s certificate issued {path[i - 1].Named}'s but is not a CA (basic constraints)";
            }

            if (!certificate.Allows(CmsCertificate.KeyCertSign))
            {
                return $"{certificate.Named}'s key may not sign certificates (key usage)";
            }

            int below = path.Take(i).Skip(1).Count(c => !c.SelfIssued);       // the intermediate CAs between it and the first
            if (certificate.PathLength is { } limit && below > limit)
            {
                return $"{certificate.Named}'s certificate allows {limit} CAs below it, the path has {below}";
            }
        }

        var first = path[0];
        if (usage == CmsUsage.Signer && !first.Allows(CmsCertificate.DigitalSignature) && !first.Allows(CmsCertificate.NonRepudiation))
        {
            return $"{first.Named}'s key may not make signatures (key usage)";
        }

        if (usage == CmsUsage.Timestamp && !first.HasPurpose(CmsOids.TimeStamping))
        {
            return $"{first.Named}'s certificate is not for timestamping (no timeStamping purpose)";
        }

        return null;
    }
}
