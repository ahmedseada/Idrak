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
/// certificate's usage, critical extensions this does not read and name or policy constraints critical or not (which fail
/// the path, naming them: never a path they would limit passed unread), RSA keys shorter than the
/// check takes, and the path's length (<see cref="SignatureCheckOptions.MinimumRsaKeyBits"/>,
/// <see cref="SignatureCheckOptions.MaxChainLength"/>). The search verifies each link once, names a loop, and tries a bounded
/// number of issuers, so the certificates a document carries cannot keep it searching. Checked against the D12 known-bad
/// samples and x509-limbo's path cases.
/// </summary>
internal static class CmsPath
{
    // The issuers a search may try for each certificate a path may hold (a document's own certificates bound no search).
    private const int TriesPerPlace = 64;

    // Extensions that limit a path in ways this check does not read (the names a CA may certify, its policies): a certificate
    // of a path carrying one, critical or not, fails it, naming it, rather than a path it would limit passing.
    private static readonly Dictionary<string, string> Limiting = new(StringComparer.Ordinal)
    {
        [CmsOids.NameConstraints] = "name constraints",
        [CmsOids.PolicyConstraints] = "policy constraints",
        [CmsOids.PolicyMappings] = "policy mappings",
        [CmsOids.InhibitAnyPolicy] = "an inhibit-any-policy limit",
    };

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
    /// <paramref name="at"/> within <paramref name="options"/>' limits (its key size and path length): the path (first to
    /// anchor) and what was found, or null and why none is valid.
    /// </summary>
    public static (List<CmsCertificate>? Path, string Detail) Build(CmsCertificate first, IReadOnlyList<CmsCertificate> pool, IReadOnlyList<CmsCertificate> anchors,
        DateTimeOffset at, CmsUsage usage, SignatureCheckOptions options)
    {
        if (anchors.Count == 0)
        {
            return (null, "no certificate is trusted: a check trusts only the certificates the app gives (SignatureCheckOptions.Trusted, or the machine's store with UseMachineStore)");
        }

        var search = new Search(pool, anchors, at, usage, options);
        var found = search.Walk([first]);
        if (found is not null)
        {
            string chain = string.Join(" < ", found.Select(c => c.Named));
            return (found, $"{chain}, trusted, valid at {CmsText.Time(at)}");
        }

        return (null, search.Best ?? $"no path from {first.Named} to a trusted certificate: its issuer '{first.IssuerName}' is neither trusted nor among the certificates the signature and the document carry");
    }

    // One path search: depth first, each issuer the names (and key identifiers) allow whose key verifies the link, until a
    // trusted one; the first whole path that checks is taken, else the reason the first failed attempt gave.
    private sealed class Search(IReadOnlyList<CmsCertificate> pool, IReadOnlyList<CmsCertificate> anchors, DateTimeOffset at, CmsUsage usage, SignatureCheckOptions options)
    {
        private readonly Dictionary<(CmsCertificate Certificate, CmsCertificate Issuer), string?> _links = [];
        private int _tried;
        private bool _stopped;

        /// <summary>Why the search found no path (the first failure met), or null.</summary>
        public string? Best { get; private set; }

        public List<CmsCertificate>? Walk(List<CmsCertificate> path)
        {
            var current = path[^1];
            if (anchors.Any(a => a.Same(current)))
            {
                string? failed = Check(path, at, usage, options.MinimumRsaKeyBits);
                if (failed is null)
                {
                    return [.. path];
                }

                Best ??= failed;
                return null;
            }

            if (path.Count >= options.MaxChainLength)
            {
                Best ??= $"a path longer than {options.MaxChainLength} certificates (SignatureCheckOptions.MaxChainLength, {SignatureCheckOptions.MaxChainLengthVariable})";
                return null;
            }

            foreach (bool trusted in (ReadOnlySpan<bool>)[true, false])
            {
                foreach (var candidate in trusted ? anchors : pool)
                {
                    if (!current.NamesIssuer(candidate) || (!trusted && anchors.Any(a => a.Same(candidate))))
                    {
                        continue;
                    }

                    if (path.Exists(c => c.Same(candidate)))
                    {
                        if (!candidate.Same(current))
                        {
                            Best ??= $"the path loops: {current.Named}'s issuer {candidate.Named} is already in it";
                        }

                        continue;
                    }

                    if (++_tried > TriesPerPlace * options.MaxChainLength)
                    {
                        // A search cut short says so first (the paths it did not try may hold one), then the first failure it met.
                        if (!_stopped)
                        {
                            _stopped = true;
                            Best = $"the path search stopped after {TriesPerPlace * options.MaxChainLength} issuers tried ({TriesPerPlace} for each certificate a path may hold, SignatureCheckOptions.MaxChainLength){(Best is null ? "" : $"; the first failure met: {Best}")}";
                        }

                        return null;
                    }

                    if (Link(current, candidate) is { } broken)
                    {
                        Best ??= $"{current.Named}'s certificate does not verify with {candidate.Named}'s key: {broken}";
                        continue;
                    }

                    path.Add(candidate);
                    var found = Walk(path);
                    path.RemoveAt(path.Count - 1);
                    if (found is not null)
                    {
                        return found;
                    }
                }
            }

            if (current.SelfIssued && Best is null)
            {
                Best = $"{current.Named}'s certificate is a root that is not trusted (the check trusts only the certificates the app gives)";
            }

            return null;
        }

        // Each link verified once a search, however many paths pass through it.
        private string? Link(CmsCertificate certificate, CmsCertificate issuer)
        {
            if (!_links.TryGetValue((certificate, issuer), out string? broken))
            {
                broken = VerifiedBy(certificate, issuer);
                _links[(certificate, issuer)] = broken;
            }

            return broken;
        }
    }

    // The basic path checks (RFC 5280, 6.1.3 and 6.1.4) on a path from its first certificate to its anchor; null when it passes.
    private static string? Check(List<CmsCertificate> path, DateTimeOffset at, CmsUsage usage, int minimumRsaBits)
    {
        // A certificate's dates are whole seconds (RFC 5280, 4.1.2.5, inclusive): the time is compared floored to its second,
        // so the last second of a validity holds whole and the one before its first does not (x509-limbo's reading).
        var second = new DateTimeOffset(at.UtcTicks - (at.UtcTicks % TimeSpan.TicksPerSecond), TimeSpan.Zero);
        for (int i = 0; i < path.Count; i++)
        {
            var certificate = path[i];
            bool anchor = i == path.Count - 1 && path.Count > 1;
            if (!anchor && (second < certificate.NotBefore || second > certificate.NotAfter))
            {
                return $"{certificate.Named}'s certificate is not valid at {CmsText.Time(at)} (from {CmsText.Time(certificate.NotBefore)} to {CmsText.Time(certificate.NotAfter)})";
            }

            foreach (var (oid, critical, _) in certificate.Extensions)
            {
                if (Limiting.TryGetValue(oid, out string? limiting))
                {
                    return $"{certificate.Named}'s certificate has {limiting} ({oid}), which this check does not read: a path they may limit is refused";
                }

                if (critical && !Known.Contains(oid))
                {
                    return $"{certificate.Named}'s certificate has the critical extension {oid}, which this check does not read";
                }
            }

            if (certificate.Problem is { } problem)
            {
                return $"{certificate.Named}'s certificate has {problem}";
            }

            if (CmsKeys.Weak(certificate, minimumRsaBits) is { } weak)
            {
                return weak;
            }

            if (i == 0)
            {
                if (certificate.Asserts(CmsCertificate.KeyCertSign) && certificate.IsCa != true)
                {
                    return $"{certificate.Named}'s key usage names keyCertSign, but its certificate is not a CA (basic constraints; RFC 5280 4.2.1.3)";
                }

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
