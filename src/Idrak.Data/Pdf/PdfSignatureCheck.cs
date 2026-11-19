// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;

namespace Idrak.Data.Pdf;

/// <summary>
/// A PDF signature checked, every step reported and why it failed (contract C22): the <b>digest</b> of the signed bytes (the
/// byte range read from the file) against the one signed; the <b>signature</b> over the signed attributes (content type,
/// message digest, the ESS signing certificate PAdES asks for) with the signer's key; the signer's <b>chain</b> to a
/// certificate the app trusts (decision 16: <see cref="SignatureCheckOptions.Trusted"/>, the machine's store only when
/// asked), at the trusted time; the <b>timestamp</b> (its imprint, its own signature, its authority's chain); the chain's
/// <b>revocation</b> from the document's validation data (DSS) and the signature's own, fetched online only when
/// <see cref="SignatureCheckOptions.Online"/>; and the <b>changes</b> made after signing. The legacy forms
/// (<c>adbe.pkcs7.sha1</c>, <c>adbe.x509.rsa_sha1</c>) and document timestamps (<c>ETSI.RFC3161</c>) are checked by their own
/// rules. A signature never makes a document trusted on its own. Safe from several threads.
/// </summary>
internal static class PdfSignatureCheck
{
    private const int Window = 1 << 20;

    public static SignatureCheck Run(PdfSignatureEntry entry, SignatureCheckOptions options)
    {
        var document = entry.Context.Document;
        ObjectDisposedException.ThrowIf(document.Disposed, document);
        var steps = new List<SignatureCheckStep>(6);
        var notes = new List<string>();
        var anchors = Anchors(options, notes);
        var data = entry.Context.Dss;
        var pool = new List<CmsCertificate>();
        pool.AddRange(entry.Cms?.Certificates ?? []);
        pool.AddRange(entry.CertEntry);
        pool.AddRange(data.Certificates);
        if (entry.Timestamp is { } stamp && !ReferenceEquals(stamp.Token, entry.Cms))
        {
            pool.AddRange(stamp.Token.Certificates);
        }

        var certificate = entry.SignerCertificate ?? entry.Signer?.FindCertificate(data.Certificates);
        var (digest, signature) = entry.SubFilter switch
        {
            "adbe.x509.rsa_sha1" => Pkcs1(entry, certificate),
            "adbe.pkcs7.sha1" => Pkcs7Sha1(entry, certificate),
            "ETSI.RFC3161" => DocumentTimestamp(entry, certificate),
            _ => Detached(entry, certificate),
        };
        steps.Add(digest);
        steps.Add(signature);

        // The timestamp first (its time is the chain's when the app gives none), reported after the chain.
        var revocations = new Revocations(entry, data, options, entry.Context.File.MaxStreamBytes);
        var (stampStep, trustedTime) = Timestamp(entry, anchors, pool, options, revocations);
        var at = options.At ?? trustedTime ?? DateTimeOffset.UtcNow;
        List<CmsCertificate>? path = null;
        if (certificate is null)
        {
            steps.Add(new SignatureCheckStep(SignatureCheck.Chain, false, "no signer's certificate to build a chain from"));
        }
        else
        {
            var usage = entry.IsDocumentTimestamp ? CmsUsage.Timestamp : CmsUsage.Signer;
            (path, string detail) = CmsPath.Build(certificate, pool, anchors, at, usage);
            string when = options.At is not null ? "" : trustedTime is not null ? " (the timestamp's time)" : " (now: no trusted time)";
            steps.Add(new SignatureCheckStep(SignatureCheck.Chain, path is not null, (path is null ? detail : detail + when) + Noted(notes)));
        }

        steps.Add(stampStep);
        steps.Add(revocations.Step(path, at));
        steps.Add(new SignatureCheckStep(SignatureCheck.Changes, entry.Changes.Changes != DocumentSignatureChanges.NotAllowed, entry.Changes.Detail));
        return new SignatureCheck(steps);
    }

    private static string Noted(List<string> notes) => notes.Count == 0 ? "" : " (" + string.Join("; ", notes) + ")";

    // The trust anchors: the app's certificates, and the machine's roots only when the app asks (decision 16).
    private static List<CmsCertificate> Anchors(SignatureCheckOptions options, List<string> notes)
    {
        var anchors = new List<CmsCertificate>();
        foreach (var trusted in options.Trusted)
        {
            try
            {
                anchors.Add(CmsCertificate.From(trusted));
            }
            catch (InvalidDataException e)
            {
                notes.Add($"a trusted certificate cannot be read: {e.Message}");
            }
        }

        if (options.UseMachineStore)
        {
            foreach (var location in (ReadOnlySpan<StoreLocation>)[StoreLocation.CurrentUser, StoreLocation.LocalMachine])
            {
                try
                {
                    using var store = new X509Store(StoreName.Root, location);
                    store.Open(OpenFlags.ReadOnly | OpenFlags.OpenExistingOnly);
                    foreach (var root in store.Certificates)
                    {
                        if (CmsCertificate.TryParse(root.RawData) is { } read)
                        {
                            anchors.Add(read);
                        }
                    }
                }
                catch (CryptographicException e)
                {
                    notes.Add($"the machine's {location} roots cannot be read: {e.Message}");
                }
            }
        }

        return anchors;
    }

    // The digest of the signed bytes: the byte range read from the file in windows.
    private static byte[] RangeDigest(PdfSignatureEntry entry, HashAlgorithmName digest)
    {
        var source = entry.Context.File.Source;
        using var hash = IncrementalHash.CreateHash(digest);
        byte[] buffer = ArrayPool<byte>.Shared.Rent(Window);
        try
        {
            for (int k = 0; k + 1 < entry.ByteRange.Length; k += 2)
            {
                long at = entry.ByteRange[k], end = at + entry.ByteRange[k + 1];
                while (at < end)
                {
                    int read = source.Read(at, buffer.AsSpan(0, (int)Math.Min(buffer.Length, end - at)));
                    if (read <= 0)
                    {
                        throw new InvalidDataException($"the file ends at byte {at:N0}, inside the signed bytes");
                    }

                    hash.AppendData(buffer, 0, read);
                    at += read;
                }
            }

            return hash.GetHashAndReset();
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    // The detached CMS forms (PAdES, adbe.pkcs7.detached): the message digest against the signed bytes', the signature over
    // the signed attributes; PAdES also asks for the ESS signing certificate.
    private static (SignatureCheckStep Digest, SignatureCheckStep Signature) Detached(PdfSignatureEntry entry, CmsCertificate? certificate)
    {
        if (Unreadable(entry) is { } failed)
        {
            return failed;
        }

        var signer = entry.Signer!;
        if (signer.Digest is not { } digest)
        {
            string why = $"the signer's digest {signer.DigestAlgorithm.Oid} is not one read here (SHA-1, SHA-256, SHA-384, SHA-512)";
            return (Fail(SignatureCheck.Digest, why), Fail(SignatureCheck.Signature, why));
        }

        if (entry.Cms!.Content is not null)
        {
            string why = "the CMS carries its content, but a detached signature of the file's bytes carries none";
            return (Fail(SignatureCheck.Digest, why), Fail(SignatureCheck.Signature, why));
        }

        byte[] signed = entry.RangeProblem is null ? RangeDigest(entry, digest) : [];
        var digestStep = entry.RangeProblem is { } range ? Fail(SignatureCheck.Digest, range) : Step(SignatureCheck.Digest, CmsVerifier.Digest(signer, signed, "the signed bytes"));
        var signatureStep = Step(SignatureCheck.Signature, CmsVerifier.Signature(signer, certificate, signed, CmsOids.Data, requireSigningCertificate: entry.SubFilter == "ETSI.CAdES.detached"));
        return (digestStep, signatureStep);
    }

    // adbe.pkcs7.sha1: the CMS carries the SHA-1 digest of the signed bytes as its content, and signs that content.
    private static (SignatureCheckStep Digest, SignatureCheckStep Signature) Pkcs7Sha1(PdfSignatureEntry entry, CmsCertificate? certificate)
    {
        if (Unreadable(entry) is { } failed)
        {
            return failed;
        }

        var signer = entry.Signer!;
        if (entry.Cms!.Content is not { } carried || signer.Digest is not { } digest)
        {
            string why = entry.Cms.Content is null ? "the legacy SHA-1 form carries no digest of the signed bytes" : $"the signer's digest {signer.DigestAlgorithm.Oid} is not read";
            return (Fail(SignatureCheck.Digest, why), Fail(SignatureCheck.Signature, why));
        }

        byte[] contentDigest = CmsDigest.Of(digest, carried);
        SignatureCheckStep digestStep;
        if (entry.RangeProblem is { } range)
        {
            digestStep = Fail(SignatureCheck.Digest, range);
        }
        else
        {
            byte[] sha1 = RangeDigest(entry, HashAlgorithmName.SHA1);
            var inner = CmsVerifier.Digest(signer, contentDigest, "the carried digest");
            digestStep = !sha1.AsSpan().SequenceEqual(carried)
                ? Fail(SignatureCheck.Digest, $"the SHA-1 digest of the signed bytes is not the one the signature carries: the signed bytes changed after signing (carried {Convert.ToHexString(carried)}, now {Convert.ToHexString(sha1)})")
                : inner.Passed ? new SignatureCheckStep(SignatureCheck.Digest, true, "the SHA-1 digest of the signed bytes is the one carried, and " + inner.Detail)
                : Step(SignatureCheck.Digest, inner);
        }

        return (digestStep, Step(SignatureCheck.Signature, CmsVerifier.Signature(signer, certificate, contentDigest, CmsOids.Data, requireSigningCertificate: false)));
    }

    // adbe.x509.rsa_sha1: /Contents holds a DER OCTET STRING with an RSA PKCS #1 v1.5 signature of the signed bytes' digest;
    // the digest is the one the signature carries, so a signature that opens with the key to another digest is a digest
    // that differs, one that does not open at all a signature that fails.
    private static (SignatureCheckStep Digest, SignatureCheckStep Signature) Pkcs1(PdfSignatureEntry entry, CmsCertificate? certificate)
    {
        if (entry.ContentsProblem is { } problem)
        {
            return (Fail(SignatureCheck.Digest, problem), Fail(SignatureCheck.Signature, problem));
        }

        if (certificate is null)
        {
            const string why = "the legacy PKCS #1 form names no signer's certificate (/Cert)";
            return (Fail(SignatureCheck.Digest, why), Fail(SignatureCheck.Signature, why));
        }

        byte[] value;
        try
        {
            value = CmsAsn.Reader(entry.Contents).ReadOctetString();
        }
        catch (Exception e) when (CmsAsn.IsDamage(e))
        {
            string why = $"/Contents is not the DER OCTET STRING the legacy PKCS #1 form holds ({e.Message})";
            return (Fail(SignatureCheck.Digest, why), Fail(SignatureCheck.Signature, why));
        }

        if (CmsKeys.Recover(certificate, value) is not { } carried)
        {
            string why = $"the RSA PKCS #1 v1.5 signature does not open with {certificate.Named}'s key";
            return (Fail(SignatureCheck.Digest, "no digest can be read from a signature that does not verify"), Fail(SignatureCheck.Signature, why));
        }

        var scheme = new CmsScheme(CmsSchemeKind.RsaPkcs1, carried.Digest);
        var signatureStep = CmsKeys.Verify(certificate, scheme, carried.Hash, value) is { } broken
            ? Fail(SignatureCheck.Signature, broken)
            : new SignatureCheckStep(SignatureCheck.Signature, true, $"{scheme} by {certificate.Named} verifies");
        if (entry.RangeProblem is { } range)
        {
            return (Fail(SignatureCheck.Digest, range), signatureStep);
        }

        byte[] signed = RangeDigest(entry, carried.Digest);
        var digestStep = signed.AsSpan().SequenceEqual(carried.Hash)
            ? new SignatureCheckStep(SignatureCheck.Digest, true, $"the {CmsDigest.Named(carried.Digest)} digest of the signed bytes is the one signed")
            : Fail(SignatureCheck.Digest, $"the {CmsDigest.Named(carried.Digest)} digest of the signed bytes is not the one signed: the signed bytes changed after signing");
        return (digestStep, signatureStep);
    }

    // ETSI.RFC3161: a document timestamp, whose token stamps the signed bytes.
    private static (SignatureCheckStep Digest, SignatureCheckStep Signature) DocumentTimestamp(PdfSignatureEntry entry, CmsCertificate? certificate)
    {
        string? problem = entry.ContentsProblem;
        if (problem is not null || entry.Timestamp is null)
        {
            string why = problem ?? "the document timestamp cannot be read";
            return (Fail(SignatureCheck.Digest, why), Fail(SignatureCheck.Signature, why));
        }

        var stamp = entry.Timestamp;
        var digestStep = entry.RangeProblem is { } range ? Fail(SignatureCheck.Digest, range)
            : stamp.Stamps(d => RangeDigest(entry, d), "the signed bytes") is { } differs ? Fail(SignatureCheck.Digest, differs + ": the signed bytes changed after the timestamp")
            : new SignatureCheckStep(SignatureCheck.Digest, true, "the timestamp stamps the digest of the signed bytes");
        var signatureStep = stamp.CheckToken() is { } broken ? Fail(SignatureCheck.Signature, broken)
            : new SignatureCheckStep(SignatureCheck.Signature, true, $"the timestamp token's signature by {certificate?.Named ?? "its authority"} verifies");
        return (digestStep, signatureStep);
    }

    private static (SignatureCheckStep, SignatureCheckStep)? Unreadable(PdfSignatureEntry entry)
    {
        string? why = entry.ContentsProblem ?? (entry.Signer is null ? "the signature holds no signer" : null);
        return why is null ? null : (Fail(SignatureCheck.Digest, why), Fail(SignatureCheck.Signature, why));
    }

    // The timestamp step: none (passed), or its imprint over the signature's value, its own signature, its authority's
    // chain and revocation; the trusted time when it is valid.
    private static (SignatureCheckStep Step, DateTimeOffset? Time) Timestamp(PdfSignatureEntry entry, List<CmsCertificate> anchors, List<CmsCertificate> pool,
        SignatureCheckOptions options, Revocations revocations)
    {
        if (entry.TimestampProblem is { } problem)
        {
            return (Fail(SignatureCheck.Timestamp, $"the timestamp cannot be read ({problem})"), null);
        }

        if (entry.Timestamp is not { } stamp)
        {
            return (new SignatureCheckStep(SignatureCheck.Timestamp, true, "none: the signature holds no timestamp (its time is the one the signer claims)"), null);
        }

        if (!entry.IsDocumentTimestamp)
        {
            var signer = entry.Signer!;
            if (stamp.Stamps(d => CmsDigest.Of(d, signer.Signature), "the signature's value") is { } other)
            {
                return (Fail(SignatureCheck.Timestamp, other), null);
            }

            if (stamp.CheckToken() is { } broken)
            {
                return (Fail(SignatureCheck.Timestamp, broken), null);
            }
        }
        else if (stamp.CheckToken() is not null)
        {
            return (Fail(SignatureCheck.Timestamp, "the document timestamp's token does not verify (see the signature step)"), null);
        }

        if (stamp.Signer is not { } authority)
        {
            return (Fail(SignatureCheck.Timestamp, "the timestamp token does not carry its authority's certificate"), null);
        }

        var at = options.At ?? stamp.Time;
        var (path, detail) = CmsPath.Build(authority, [.. pool, .. stamp.Token.Certificates], anchors, at, CmsUsage.Timestamp);
        if (path is null)
        {
            return (Fail(SignatureCheck.Timestamp, $"the timestamp authority is not trusted: {detail}"), null);
        }

        var revoked = revocations.Revoked(path, at);
        if (revoked is not null)
        {
            return (Fail(SignatureCheck.Timestamp, $"the timestamp authority's chain is revoked: {revoked}"), null);
        }

        string kind = entry.IsDocumentTimestamp ? "a document timestamp" : "valid";
        return (new SignatureCheckStep(SignatureCheck.Timestamp, true, $"{kind}: {CmsText.Time(stamp.Time)} from {authority.Named} (policy {stamp.Policy}; {detail})"), stamp.Time);
    }

    private static SignatureCheckStep Step(string name, (bool Passed, string Detail) result) => new(name, result.Passed, result.Detail);

    private static SignatureCheckStep Fail(string name, string detail) => new(name, false, detail);

    /// <summary>
    /// The revocation sources a check reads (the document's DSS, the signature's own revocation lists and Adobe's
    /// revocation attribute; online ones only when allowed), and the revocation step over a path.
    /// </summary>
    private sealed class Revocations
    {
        private readonly List<CmsCrl> _crls = [];
        private readonly List<CmsOcsp> _ocsps = [];
        private readonly List<string> _notes = [];
        private readonly SignatureCheckOptions _options;
        private readonly long _maxBytes;

        public Revocations(PdfSignatureEntry entry, PdfValidationData data, SignatureCheckOptions options, long maxBytes)
        {
            (_options, _maxBytes) = (options, maxBytes);
            _crls.AddRange(data.Crls);
            _ocsps.AddRange(data.Ocsps);
            foreach (var list in entry.Cms?.RevocationLists ?? [])
            {
                Add(() => _crls.Add(CmsCrl.Parse(list)), "a revocation list in the signature");
            }

            // Adobe's revocation attribute (RevocationInfoArchival: [0] lists, [1] OCSP responses), signed with the signature.
            if (entry.Signer?.Signed(CmsOids.AdobeRevocation) is { Values.Length: >= 1 } archival)
            {
                Add(() =>
                {
                    var sequence = CmsAsn.Reader(archival.Values[0]).ReadSequence();
                    while (sequence.HasData)
                    {
                        var tag = sequence.PeekTag();
                        var items = sequence.ReadSequence(tag).ReadSequence();
                        while (items.HasData)
                        {
                            var item = items.ReadEncodedValue();
                            if (tag.TagValue == 0)
                            {
                                _crls.Add(CmsCrl.Parse(item));
                            }
                            else if (tag.TagValue == 1)
                            {
                                _ocsps.Add(CmsOcsp.Parse(item));
                            }
                        }
                    }
                }, "the signature's revocation attribute");
            }
        }

        private void Add(Action parse, string what)
        {
            try
            {
                parse();
            }
            catch (Exception e) when (CmsAsn.IsDamage(e))
            {
                _notes.Add($"{what} cannot be read ({e.Message})");
            }
        }

        /// <summary>Why a certificate of <paramref name="path"/> is revoked at <paramref name="at"/>, or null (what is not known is not revoked).</summary>
        public string? Revoked(List<CmsCertificate> path, DateTimeOffset at)
        {
            for (int i = 0; i + 1 < path.Count; i++)
            {
                var (status, detail) = StatusOf(path[i], path[i + 1], at);
                if (status == CmsStatus.Revoked)
                {
                    return detail;
                }
            }

            return null;
        }

        public SignatureCheckStep Step(List<CmsCertificate>? path, DateTimeOffset at)
        {
            if (path is null)
            {
                return new SignatureCheckStep(SignatureCheck.Revocation, !_options.RequireRevocation, "not checked: there is no path to a trusted certificate to check");
            }

            var details = new List<string>();
            bool revoked = false, unknown = false;
            for (int i = 0; i + 1 < path.Count; i++)
            {
                if (path[i].OcspNoCheck)
                {
                    details.Add($"{path[i].Named} needs no check (ocsp-nocheck)");
                    continue;
                }

                var (status, detail) = StatusOf(path[i], path[i + 1], at);
                revoked |= status == CmsStatus.Revoked;
                unknown |= status == CmsStatus.Unknown;
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
                return new SignatureCheckStep(SignatureCheck.Revocation, !_options.RequireRevocation, $"not checked: {all}{online}");
            }

            return new SignatureCheckStep(SignatureCheck.Revocation, true, path.Count == 1 ? "good: the signer's certificate is trusted itself" : "good: " + all);
        }

        // A certificate's status from every source: revoked when one says so, good when one says so, else unknown (fetched
        // online first, when allowed).
        private (CmsStatus Status, string Detail) StatusOf(CmsCertificate certificate, CmsCertificate issuer, DateTimeOffset at)
        {
            var (status, detail) = Known(certificate, issuer, at);
            if (status != CmsStatus.Unknown || !_options.Online)
            {
                return (status, detail);
            }

            var fetched = Fetch(certificate, issuer);
            (status, detail) = Known(certificate, issuer, at);
            return status == CmsStatus.Unknown ? (status, $"{detail} ({fetched})") : (status, $"{detail} (fetched online)");
        }

        private (CmsStatus, string) Known(CmsCertificate certificate, CmsCertificate issuer, DateTimeOffset at)
        {
            CmsStatusAnswer? good = null;
            string? why = null;
            lock (_ocsps)
            {
                foreach (var answer in _ocsps.Select(o => o.StatusOf(certificate, issuer, at)).Concat(_crls.Select(c => c.StatusOf(certificate, issuer, at))))
                {
                    switch (answer.Status)
                    {
                        case CmsStatus.Revoked:
                            return (CmsStatus.Revoked, answer.Detail);
                        case CmsStatus.Good:
                            good ??= answer;
                            break;
                        default:
                            if (!answer.Detail.StartsWith("a response for another", StringComparison.Ordinal) && !answer.Detail.StartsWith("another issuer", StringComparison.Ordinal))
                            {
                                why ??= answer.Detail;
                            }

                            break;
                    }
                }
            }

            return good is { } found ? (CmsStatus.Good, found.Detail) : (CmsStatus.Unknown, why ?? $"no revocation data for {certificate.Named}");
        }

        // Revocation data fetched from the addresses the certificate names (OCSP first, then its lists), through SignatureServices.
        private string Fetch(CmsCertificate certificate, CmsCertificate issuer)
        {
            var tried = new List<string>();
            foreach (string address in certificate.OcspAddresses)
            {
                try
                {
                    byte[] answer = SignatureServices.Send(new SignatureServiceRequest(new Uri(address), SignatureServiceKind.Ocsp)
                    {
                        Body = CmsRevocation.Request(certificate, issuer), Timeout = _options.Timeout, MaxResponseBytes = _maxBytes,
                    });
                    lock (_ocsps)
                    {
                        _ocsps.Add(CmsOcsp.Parse(answer));
                    }

                    return $"asked {address}";
                }
                catch (Exception e) when (e is IOException or InvalidDataException or UriFormatException or NotSupportedException or OperationCanceledException or TimeoutException)
                {
                    tried.Add($"{address}: {e.Message}");
                }
            }

            foreach (string address in certificate.CrlAddresses)
            {
                try
                {
                    byte[] answer = SignatureServices.Send(new SignatureServiceRequest(new Uri(address), SignatureServiceKind.RevocationList)
                    {
                        Timeout = _options.Timeout, MaxResponseBytes = _maxBytes,
                    });
                    lock (_ocsps)
                    {
                        _crls.Add(CmsCrl.Parse(answer));
                    }

                    return $"fetched {address}";
                }
                catch (Exception e) when (e is IOException or InvalidDataException or UriFormatException or NotSupportedException or OperationCanceledException or TimeoutException)
                {
                    tried.Add($"{address}: {e.Message}");
                }
            }

            return tried.Count == 0 ? $"{certificate.Named}'s certificate names no address to ask" : "fetching failed: " + string.Join("; ", tried);
        }
    }
}
