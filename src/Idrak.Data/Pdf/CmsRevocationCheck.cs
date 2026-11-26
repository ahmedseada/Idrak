// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Data.Pdf;

/// <summary>
/// The revocation sources one check reads (the document's validation data, the signature's own lists and responses; online
/// ones only when <see cref="SignatureCheckOptions.Online"/>), and what they say of the certificates of a path (RFC 5280
/// 6.3, RFC 6960): OCSP responses first, then complete lists, each with the newest delta list that applies to it; an
/// indirect list (its issuing distribution point's indirectCRL) answers for a certificate whose distribution point names
/// its issuer, once that issuer's certificate chains to the same trusted certificate and is not revoked. A source whose next
/// update, with <see cref="SignatureCheckOptions.FreshnessGrace"/>, is before the time is stale: fetched again when online,
/// else the certificate is "not checked (stale)". Each list's and response's signature is verified once a check, however
/// many certificates ask about it (never across checks). Used by one check, on one thread at a time.
/// </summary>
internal sealed class CmsRevocationCheck
{
    private readonly List<CmsCrl> _crls = [];
    private readonly List<CmsCrl> _deltas = [];
    private readonly List<CmsOcsp> _ocsps = [];
    private readonly List<string> _notes;
    private readonly IReadOnlyList<CmsCertificate> _pool;
    private readonly SignatureCheckOptions _options;
    private readonly long _maxBytes;
    private readonly CmsVerifications _verified = new();
    private readonly List<CmsCertificate> _asking = new(4);
    private HashSet<object>? _fetched;

    public CmsRevocationCheck(IEnumerable<CmsCrl> crls, IEnumerable<CmsOcsp> ocsps, List<string> notes, IReadOnlyList<CmsCertificate> pool, SignatureCheckOptions options,
        long maxBytes)
    {
        foreach (var crl in crls)
        {
            Add(crl);
        }

        _ocsps.AddRange(ocsps);
        (_notes, _pool, _options, _maxBytes) = (notes, pool, options, maxBytes);
    }

    private void Add(CmsCrl crl) => (crl.IsDelta ? _deltas : _crls).Add(crl);

    /// <summary>Why a certificate of <paramref name="path"/> is revoked at <paramref name="at"/>, or null (what is not known is not revoked).</summary>
    public string? Revoked(List<CmsCertificate> path, DateTimeOffset at)
    {
        for (int i = 0; i + 1 < path.Count; i++)
        {
            var (status, detail, _) = StatusOf(path[i], path[i + 1], path[^1], at);
            if (status == CmsStatus.Revoked)
            {
                return detail;
            }
        }

        return null;
    }

    /// <summary>The revocation step over a path (first to anchor): revoked, not checked (stale or without data), or good.</summary>
    public SignatureCheckStep Step(List<CmsCertificate>? path, DateTimeOffset at)
    {
        if (path is null)
        {
            return new SignatureCheckStep(SignatureCheck.Revocation, !_options.RequireRevocation, "not checked: there is no path to a trusted certificate to check");
        }

        var details = new List<string>();
        bool revoked = false, unknown = false, stale = false;
        for (int i = 0; i + 1 < path.Count; i++)
        {
            if (path[i].OcspNoCheck)
            {
                details.Add($"{path[i].Named} needs no check (ocsp-nocheck)");
                continue;
            }

            var (status, detail, old) = StatusOf(path[i], path[i + 1], path[^1], at);
            revoked |= status == CmsStatus.Revoked;
            unknown |= status == CmsStatus.Unknown;
            stale |= status == CmsStatus.Unknown && old;
            details.Add(detail);
        }

        details.AddRange(_notes);
        string all = string.Join("; ", details);
        if (revoked)
        {
            return new SignatureCheckStep(SignatureCheck.Revocation, false, "revoked: " + all);
        }

        if (unknown)
        {
            string online = _options.Online ? "" : " (fetching is off: SignatureCheckOptions.Online, or IDRAK_DOCUMENT_SIGNATURE_ONLINE)";
            return new SignatureCheckStep(SignatureCheck.Revocation, !_options.RequireRevocation, $"not checked{(stale ? " (stale)" : "")}: {all}{online}");
        }

        return new SignatureCheckStep(SignatureCheck.Revocation, true, path.Count == 1 ? "good: the signer's certificate is trusted itself" : "good: " + all);
    }

    // A certificate's status from every source: revoked when one says so, good when one says so, else unknown (fetched
    // online first, when allowed); whether a source was stale.
    private (CmsStatus Status, string Detail, bool Stale) StatusOf(CmsCertificate certificate, CmsCertificate issuer, CmsCertificate anchor, DateTimeOffset at)
    {
        if (Asking(certificate))
        {
            return (CmsStatus.Unknown, $"{certificate.Named}'s status is being checked already (an indirect list's issuer on its own path)", false);
        }

        _asking.Add(certificate);

        try
        {
            // A source fetched for another certificate of the check (the timestamp authority's path is checked first) answers
            // for this one too, and says it was fetched.
            var (status, detail, stale, online) = Known(certificate, issuer, anchor, at);
            if (status != CmsStatus.Unknown || !_options.Online)
            {
                return (status, online ? $"{detail} (fetched online)" : detail, stale);
            }

            string fetched = Fetch(certificate, issuer);
            (status, detail, stale, _) = Known(certificate, issuer, anchor, at);
            return status == CmsStatus.Unknown ? (status, $"{detail} ({fetched})", stale) : (status, $"{detail} (fetched online)", false);
        }
        finally
        {
            _asking.RemoveAt(_asking.Count - 1);
        }
    }

    // Whether the certificate's status is being asked already, further up this check (an indirect list's issuer's path).
    private bool Asking(CmsCertificate certificate)
    {
        foreach (var asked in _asking)
        {
            if (ReferenceEquals(asked, certificate))
            {
                return true;
            }
        }

        return false;
    }

    // What the sources held say of a certificate (OCSP responses first, then complete lists): its status, why, whether a
    // source was stale, and whether the answer came from a source fetched online. The lists are walked by index: a fetch
    // (online, for an indirect list's issuer) only adds to their ends.
    private (CmsStatus, string, bool, bool) Known(CmsCertificate certificate, CmsCertificate issuer, CmsCertificate anchor, DateTimeOffset at)
    {
        var tally = new Tally();
        for (int i = 0; i < _ocsps.Count; i++)
        {
            if (tally.Add(_ocsps[i].StatusOf(certificate, issuer, at, _options.FreshnessGrace, _verified), _fetched?.Contains(_ocsps[i]) == true))
            {
                return (CmsStatus.Revoked, tally.Revoked!, false, tally.Online);
            }
        }

        for (int i = 0; i < _crls.Count; i++)
        {
            if (tally.Add(ListStatus(_crls[i], certificate, issuer, anchor, at), _fetched?.Contains(_crls[i]) == true))
            {
                return (CmsStatus.Revoked, tally.Revoked!, false, tally.Online);
            }
        }

        if (tally.Good is { } found)
        {
            return (CmsStatus.Good, found, false, tally.Online);
        }

        if (tally.Covered == CmsCrl.AllReasons)
        {
            return (CmsStatus.Good, $"{tally.Partial} (lists covering some reasons each, every reason between them)", false, false);
        }

        string? why = tally.Why ?? (tally.Partial is null ? null : $"{tally.Partial}, but its lists cover only some revocation reasons");
        return (CmsStatus.Unknown, why ?? $"no revocation data for {certificate.Named}", tally.Stale, false);
    }

    // The answers of the sources for one certificate: the first revoked (which ends the walk), the first good, the reasons
    // lists covering only some cover between them, and the first reason a source that concerns the certificate says nothing.
    private struct Tally
    {
        public string? Revoked, Good, Partial, Why;
        public bool Online, Stale;
        public int Covered;

        public bool Add(CmsStatusAnswer answer, bool online)
        {
            switch (answer.Status)
            {
                case CmsStatus.Revoked:
                    (Revoked, Online) = (answer.Detail, online);
                    return true;
                case CmsStatus.Good when answer.Reasons == CmsCrl.AllReasons:
                    if (Good is null)
                    {
                        (Good, Online) = (answer.Detail, online);
                    }

                    break;
                case CmsStatus.Good:
                    Covered |= answer.Reasons;
                    Partial = Partial is null ? answer.Detail : $"{Partial}, {answer.Detail}";
                    break;
                default:
                    if (!answer.Detail.StartsWith("a response for another", StringComparison.Ordinal) && !answer.Detail.StartsWith("another issuer", StringComparison.Ordinal))
                    {
                        Why ??= answer.Detail;
                        Stale |= answer.Stale;
                    }

                    break;
            }

            return false;
        }
    }

    // What one complete list says of a certificate: a direct list through its issuer; an indirect one through the list's issuer
    // the certificate's distribution point names, once that issuer's certificate is found and checked.
    private CmsStatusAnswer ListStatus(CmsCrl crl, CmsCertificate certificate, CmsCertificate issuer, CmsCertificate anchor, DateTimeOffset at)
    {
        var grace = _options.FreshnessGrace;
        if (crl.Issuer.Span.SequenceEqual(certificate.Issuer.Span) || !crl.Indirect || !certificate.NamesCrlIssuer(crl.Issuer))
        {
            return crl.StatusOf(certificate, issuer, at, grace, _verified, _deltas);
        }

        var (signer, why) = IndirectIssuer(crl, anchor, at);
        return signer is null ? new CmsStatusAnswer(CmsStatus.Unknown, why!) : crl.StatusOf(certificate, signer, at, grace, _verified, _deltas);
    }

    // An indirect list's issuer (RFC 5280 6.3.3 (f)): a certificate of its name that may sign lists, whose key verifies the
    // list, with a path to the same trusted certificate as the path asked about (RFC 5280's defaults for its policies), none
    // of it revoked (nor unknown, when revocation is required); or why there is none.
    private (CmsCertificate? Signer, string? Why) IndirectIssuer(CmsCrl crl, CmsCertificate anchor, DateTimeOffset at)
    {
        string? first = null;
        string named = CmsName.Common(crl.Issuer);
        foreach (var candidate in _pool.Append(anchor))
        {
            if (!candidate.Subject.Span.SequenceEqual(crl.Issuer.Span) || Asking(candidate))
            {
                continue;
            }

            if (!candidate.Allows(CmsCertificate.CrlSign))
            {
                first ??= $"{candidate.Named}'s key may not sign revocation lists (key usage)";
                continue;
            }

            if (_verified.Verify(crl, candidate) is { } broken)
            {
                first ??= $"{candidate.Named}'s indirect list does not verify: {broken}";
                continue;
            }

            var (path, detail) = CmsPath.Build(candidate, _pool, [anchor], at, CmsUsage.CrlSigner, _options);
            if (path is null)
            {
                first ??= $"{candidate.Named}, an indirect list's issuer, has no valid path to {anchor.Named}: {detail}";
                continue;
            }

            string? refused = null;
            for (int i = 0; i + 1 < path.Count && refused is null; i++)
            {
                var (status, said, _) = StatusOf(path[i], path[i + 1], anchor, at);
                if (status == CmsStatus.Revoked || (status == CmsStatus.Unknown && _options.RequireRevocation))
                {
                    refused = $"{candidate.Named}, an indirect list's issuer, cannot be relied on: {said}";
                }
            }

            if (refused is null)
            {
                return (candidate, null);
            }

            first ??= refused;
        }

        return (null, first ?? $"{named}'s indirect list, whose issuer's certificate is neither trusted nor among the certificates the signature and the document carry");
    }

    // The signature services were registered by their scheme (SignatureServices) and asked through it: not on the new design yet.
    private static byte[] Send(SignatureServiceRequest request) =>
        throw new NotSupportedException("A timestamp or revocation request through the signature services by scheme is not on the new design yet.");

    // Revocation data fetched from the addresses the certificate names (OCSP first, then its lists with their deltas), through
    // SignatureServices.
    private string Fetch(CmsCertificate certificate, CmsCertificate issuer)
    {
        var tried = new List<string>();
        foreach (string address in certificate.OcspAddresses)
        {
            try
            {
                byte[] answer = Send(new SignatureServiceRequest(new Uri(address), SignatureServiceKind.Ocsp)
                {
                    Body = CmsRevocation.Request(certificate, issuer), Timeout = _options.Timeout, MaxResponseBytes = _maxBytes,
                });
                var parsed = CmsOcsp.Parse(answer);
                _ocsps.Add(parsed);
                (_fetched ??= []).Add(parsed);

                return $"asked {address}";
            }
            catch (Exception e) when (Failed(e))
            {
                tried.Add($"{address}: {e.Message}");
            }
        }

        foreach (string address in certificate.CrlAddresses)
        {
            try
            {
                var list = CmsCrl.Parse(Get(address));
                Add(list);
                (_fetched ??= []).Add(list);

                // The list's deltas (the certificate's and the list's freshest-CRL addresses): a delta that cannot be had
                // leaves the complete list's answer.
                foreach (string delta in certificate.FreshestAddresses.Concat(list.FreshestAddresses).Distinct(StringComparer.Ordinal))
                {
                    try
                    {
                        var fetched = CmsCrl.Parse(Get(delta));
                        Add(fetched);
                        (_fetched ??= []).Add(fetched);
                    }
                    catch (Exception e) when (Failed(e))
                    {
                        _notes.Add($"the delta list at {delta} cannot be had ({e.Message})");
                    }
                }

                return $"fetched {address}";
            }
            catch (Exception e) when (Failed(e))
            {
                tried.Add($"{address}: {e.Message}");
            }
        }

        return tried.Count == 0 ? $"{certificate.Named}'s certificate names no address to ask" : "fetching failed: " + string.Join("; ", tried);
    }

    private byte[] Get(string address) =>
        Send(new SignatureServiceRequest(new Uri(address), SignatureServiceKind.RevocationList) { Timeout = _options.Timeout, MaxResponseBytes = _maxBytes });

    private static bool Failed(Exception e) => e is IOException or InvalidDataException or UriFormatException or NotSupportedException or OperationCanceledException or TimeoutException;
}
