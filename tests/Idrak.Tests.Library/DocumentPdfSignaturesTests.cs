// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Formats.Asn1;
using System.Globalization;
using System.Numerics;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Data.Pdf;
using Idrak.Imaging;

// Plan 28, phase D12: signatures read and checked (CMS, the byte range, the chain against the trust the app gives,
// timestamps, revocation from the file's DSS, changes after signing, the legacy forms) on the D12 samples
// tools/documents/make_fixtures.py writes into the ignored test-output/documents/samples (with their test PKI: never a real
// certificate), and signing (PAdES B-B: RSA, RSA-PSS and ECDSA) with the PKI's throwaway keys. Every file signed here goes to
// test-output/documents/signed (NAME.pdf, and NAME.json with what each signature must check as) for
// tools/documents/check_signed.py (pyHanko, the reference validator).
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] DocumentPdfSignaturesGroup =
    [
        ("documents pdf: signatures: timings: every D12 sample's signatures listed and checked (cold, then warm; time and managed allocation a signature) and a page signed at B-B (cold, warm)", PdfSignatureTimings),
        ("documents pdf: signatures: every D12 sample's signatures read and check as its JSON says (kind, field, signer, times, byte range, coverage; digest, signature, chain against the JSON's trust alone, timestamp, revocation from the DSS, changes after signing and their level) and as pyHanko found them, alike from several threads", PdfSignatureSamples),
        ("documents pdf: signatures: each step says why it failed (a changed byte, an untrusted root, a revoked signer, content changed after certification); nothing trusted that the app did not give; the trust anchor, the time and required revocation as the options say", PdfSignatureReasons),
        ("documents pdf: signatures: signing as incremental updates: PAdES B-B (RSA, RSA-PSS, ECDSA) with appearances, a second signature keeping the first valid, the builder's Sign at Finish, after other edits and after a rewrite: each checks valid, the source's bytes its prefix", PdfSignatureSigning),
        ("documents pdf: signatures: misuse refused (no options, no private key, SHA-1, B-T without an authority, B-LT without revocation data, a field that is not a signature field or is signed already, a dotted new name, a page out of range, the builder signing after its first page)", PdfSignatureRefusals),
        ("documents pdf: signatures: trust: the policies a path is valid for named (by default and for the options' initial policies and flags), an indirect and a delta list answering, stale data not checked offline and taken with a grace, each list's and response's signature verified once a check", PdfSignatureTrust),
        ("documents pdf: signatures: known bad: Project Wycheproof's RSA PKCS #1 v1.5 vectors (2048, 3072 and 4096 bits; SHA-1, SHA-256, -384, -512): every valid case verifies and every invalid one fails, through CmsKeys and the legacy form's route; the acceptable ones listed with Idrak's answer", PdfKnownBadPkcs1),
        ("documents pdf: signatures: known bad: Project Wycheproof's RSA-PSS vectors (salt lengths 0 to 64, MGF1 digests, keys certified with their parameters): every valid case verifies, every invalid one fails", PdfKnownBadPss),
        ("documents pdf: signatures: known bad: Project Wycheproof's ECDSA vectors (P-256, P-384, P-521, secp256k1, Brainpool; the DER signature read by Idrak, r and s ranged by the curve's order): every valid case verifies, every invalid one fails; a curve this platform's ECDsa does not take is a named limit", PdfKnownBadEcdsa),
        ("documents pdf: signatures: known bad: this platform's RSA (RSA.VerifyHash, a test-only guard) gives every Wycheproof valid and invalid case it takes the verdict Idrak gives (PSS where its salt is as long as the digest; an input it refuses printed and passed over)", PdfKnownBadPlatformRsa),
        ("documents pdf: signatures: known bad: the generator's known-bad samples (expired and not yet valid, a forged issuer signature, an issuer not a CA, a path past its length, an issuer without keyCertSign, a signer without a signing usage, an unknown critical extension, names that do not chain, a loop, revoked by OCSP, a CRL and an OCSP response signed by the wrong key, a timestamp of other bytes, a path past the limit, a weak RSA key): each refused at its step saying why; the option that is the cause lets it pass", PdfKnownBadPaths),
        ("documents pdf: signatures: known bad: x509-limbo's path and revocation-list cases through Idrak's path builder give limbo's verdict; the cases a document's check answers otherwise named with the rule it applies, the limits named", PdfKnownBadLimbo),
        ("documents pdf: signatures: known bad: certificates that issued each other many times over end a path search within its bound, quickly, naming it; the trusted one among them found at once", PdfKnownBadSearchBound),
    ];

    private static readonly DateTimeOffset PdfSigningTime = new(2026, 3, 2, 9, 30, 0, TimeSpan.Zero);
    private static readonly DateTimeOffset PdfValidationTime = new(2026, 3, 9, 12, 0, 0, TimeSpan.Zero);
    private static readonly Uri PdfTestAuthority = new("idrak-test://timestamps.example.com/");

    // The D12 test PKI the generator writes beside the samples (pki.LABEL.cer, pki.LABEL.key.pem, the lists, the response).
    private sealed class PdfPki
    {
        private readonly string _folder;

        public PdfPki(string folder) => _folder = folder;

        public string Folder => _folder;

        public X509Certificate2 Certificate(string label) => X509CertificateLoader.LoadCertificateFromFile(Path.Combine(_folder, $"pki.{label}.cer"));

        public X509Certificate2 WithKey(string label) => X509Certificate2.CreateFromPemFile(Path.Combine(_folder, $"pki.{label}.pem"), Path.Combine(_folder, $"pki.{label}.key.pem"));

        public byte[] File(string name) => System.IO.File.ReadAllBytes(Path.Combine(_folder, name));

        public SignatureCheckOptions Trust(DateTimeOffset? at = null) => new() { Trusted = [Certificate("root")], At = at ?? PdfValidationTime, Online = false };
    }

    private static PdfPki PdfSignaturePki()
    {
        var sample = PdfEditSample("pades-bb");
        string folder = Path.GetDirectoryName(sample.Path)!;
        Check(File.Exists(Path.Combine(folder, "pki.root.cer")) && File.Exists(Path.Combine(folder, "pki.signer.key.pem")), $"the test PKI is beside the D12 samples in {folder}: make the samples again (tools/documents/make_fixtures.py)");
        return new PdfPki(folder);
    }

    private static SignatureCheckStep PdfStep(SignatureCheck check, string name) =>
        check.Steps.SingleOrDefault(s => s.Name == name) is { Name: not null } step ? step : throw new InvalidOperationException($"the check has no step '{name}': {PdfDescribe(check)}");

    private static string PdfDescribe(SignatureCheck check) => string.Join(" | ", check.Steps.Select(s => $"{s.Name} {(s.Passed ? "passed" : "FAILED")}: {s.Detail}"));

    private static DocumentSignatureKind PdfSignatureKindOf(string kind) => kind switch
    {
        "PAdES B-B" => DocumentSignatureKind.BaselineB,
        "PAdES B-T" => DocumentSignatureKind.BaselineT,
        "PAdES B-LT" => DocumentSignatureKind.BaselineLt,
        "CMS" => DocumentSignatureKind.Cms,
        "legacy PKCS #7 SHA-1" => DocumentSignatureKind.LegacyPkcs7Sha1,
        "legacy PKCS #1 SHA-1" => DocumentSignatureKind.LegacyPkcs1,
        _ => throw new InvalidOperationException($"a sample kind the test does not know: {kind}"),
    };

    // The D12 samples: every sample whose JSON lists signatures.
    private static List<PdfSampleEntry> PdfSignedSamples()
    {
        var samples = PdfSamples().Where(s => s.Expected["signatures"] is JsonArray { Count: > 0 } && !s.Encrypted).ToList();
        Check(samples.Count >= 13, $"{samples.Count} signed samples (13 or more expected): make the samples again (tools/documents/make_fixtures.py)");
        return samples;
    }

    private static void PdfSignatureSamples(Device device)
    {
        CpuOnly(device);
        var pki = PdfSignaturePki();
        int checkedCount = 0;
        foreach (var sample in PdfSignedSamples())
        {
            byte[] bytes = File.ReadAllBytes(sample.Path);
            using var document = PdfEditOpen(bytes, sample.Name);
            var expected = sample.Expected["signatures"]!.AsArray();
            var signatures = document.Signatures;
            Check(signatures.Count == expected.Count && ReferenceEquals(signatures, document.Signatures), $"{sample.Name}: {signatures.Count} signatures, {expected.Count} expected");
            var reference = sample.Expected["reference"]?["validator"]?["signatures"] as JsonArray;
            for (int i = 0; i < expected.Count; i++)
            {
                var want = expected[i]!;
                var signature = signatures[i];
                string what = $"{sample.Name}: {signature.Field}";
                var results = want["results"]!;
                Check(signature.Field == (string)want["field"]! && signature.KindName == (string)want["subFilter"]! && signature.Kind == PdfSignatureKindOf((string)want["kind"]!),
                    $"{what}: field {signature.Field}, {signature.KindName}, {signature.Kind}; expected {want["field"]}, {want["subFilter"]}, {want["kind"]}");
                Check(signature.ByteRange.SequenceEqual(want["byteRange"]!.AsArray().Select(v => (long)v!)) && signature.CoversWholeFile == (bool)want["coversWholeFile"]!,
                    $"{what}: byte range {string.Join(" ", signature.ByteRange)}, covers the whole file {signature.CoversWholeFile}");
                Check(signature.Signer is { } signer && Convert.ToHexStringLower(SHA256.HashData(signer.RawData)) == (string)want["signer"]!["sha256"]!,
                    $"{what}: the signer is {signature.Signer?.Subject}, expected {want["signer"]!["subject"]}");
                Check(signature.SignedAt == DateTimeOffset.Parse((string)want["claimedSigningTime"]!, CultureInfo.InvariantCulture)
                      && signature.Reason == (string?)want["reason"] && signature.Location == (string?)want["location"],
                    $"{what}: signed at {signature.SignedAt}, reason {signature.Reason}, location {signature.Location}");
                var stamp = want["timestamp"];
                Check(stamp is null ? signature.TimestampedAt is null : signature.TimestampedAt == DateTimeOffset.Parse((string)stamp["time"]!, CultureInfo.InvariantCulture),
                    $"{what}: timestamped at {signature.TimestampedAt}, expected {stamp?["time"]}");
                string modifications = (string)results["modifications"]!;
                var changes = modifications switch { "none" => DocumentSignatureChanges.None, "allowed" => DocumentSignatureChanges.Allowed, _ => DocumentSignatureChanges.NotAllowed };
                Check(signature.Changes == changes, $"{what}: changes {signature.Changes}, expected {modifications}");

                var trust = want["trust"]!.AsArray().Select(t => X509CertificateLoader.LoadCertificateFromFile(Path.Combine(pki.Folder, (string)t!))).ToList();
                var at = DateTimeOffset.Parse((string)want["validationTime"]!, CultureInfo.InvariantCulture);
                var check = signature.Check(new SignatureCheckOptions { Trusted = trust, At = at, Online = false });
                PdfExpectResults(what, check, results);
                if (want["policies"] is JsonArray policies)
                {
                    Check(check.Policies.SequenceEqual(policies.Select(p => (string)p!)), $"{what}: valid for the policies {string.Join(", ", check.Policies)}, expected {policies.ToJsonString()}");
                }

                // The same when checked at the trusted time (the timestamp's, else now), and alike from several threads.
                var parallel = new SignatureCheck[4];
                Parallel.For(0, parallel.Length, k => parallel[k] = signature.Check(new SignatureCheckOptions { Trusted = trust, At = at, Online = false }));
                Check(parallel.All(p => PdfDescribe(p) == PdfDescribe(check)), $"{what}: checked from several threads alike");
                if (stamp is not null)
                {
                    PdfExpectResults(what + " (at the timestamp's time)", signature.Check(new SignatureCheckOptions { Trusted = trust, Online = false }), results);
                }

                // pyHanko's verdicts recorded with the sample (not for the legacy forms, which it does not read).
                if (reference is not null && i < reference.Count && reference[i] is JsonObject pyhanko && pyhanko["intact"] is not null)
                {
                    bool intact = (bool)pyhanko["intact"]!;
                    Check(PdfStep(check, SignatureCheck.Digest).Passed == intact, $"{what}: digest {PdfStep(check, SignatureCheck.Digest).Passed}, pyHanko's intact {intact}");
                    if (intact)
                    {
                        Check(PdfStep(check, SignatureCheck.Signature).Passed == (bool)pyhanko["valid"]!, $"{what}: signature as pyHanko's valid {pyhanko["valid"]}");
                        Check(PdfStep(check, SignatureCheck.Changes).Passed == (bool)pyhanko["docmdpOk"]!, $"{what}: changes as pyHanko's docmdpOk {pyhanko["docmdpOk"]}");
                    }
                }

                checkedCount++;
            }
        }

        Check(checkedCount >= 14, $"{checkedCount} signatures checked");
    }

    // A check against a sample's results: each step passed or not as they say, the detail naming what was found.
    private static void PdfExpectResults(string what, SignatureCheck check, JsonNode results)
    {
        var digest = PdfStep(check, SignatureCheck.Digest);
        var signature = PdfStep(check, SignatureCheck.Signature);
        var chain = PdfStep(check, SignatureCheck.Chain);
        var timestamp = PdfStep(check, SignatureCheck.Timestamp);
        var revocation = PdfStep(check, SignatureCheck.Revocation);
        var changes = PdfStep(check, SignatureCheck.Changes);
        string all = PdfDescribe(check);
        Check(check.Steps.Select(s => s.Name).SequenceEqual(new[] { SignatureCheck.Digest, SignatureCheck.Signature, SignatureCheck.Chain, SignatureCheck.Timestamp, SignatureCheck.Revocation, SignatureCheck.Changes }),
            $"{what}: the steps in order: {all}");
        Check(digest.Passed == ((string)results["digest"]! == "valid"), $"{what}: digest {results["digest"]}: {all}");
        Check(signature.Passed == ((string)results["signature"]! == "valid"), $"{what}: signature {results["signature"]}: {all}");

        // The JSON's "untrusted" for a revoked signer is pyHanko's verdict on the whole path; here the path checks and its
        // revocation step fails.
        string revoked = (string)results["revocation"]!;
        bool trusted = (string)results["chain"]! == "trusted";
        Check(revoked == "revoked" ? chain.Passed && !revocation.Passed : chain.Passed == trusted, $"{what}: chain {results["chain"]} (revocation {revoked}): {all}");
        Check(revoked switch
        {
            "good" => revocation.Passed && revocation.Detail.StartsWith("good", StringComparison.Ordinal),
            "revoked" => !revocation.Passed && revocation.Detail.StartsWith("revoked", StringComparison.Ordinal),
            _ => revocation.Passed && revocation.Detail.StartsWith("not checked", StringComparison.Ordinal),
        }, $"{what}: revocation {revoked}: {revocation.Detail}");
        string? stamped = (string?)results["timestamp"];
        Check(stamped switch
        {
            null => timestamp.Passed && timestamp.Detail.StartsWith("none", StringComparison.Ordinal),
            "valid" => timestamp.Passed && timestamp.Detail.StartsWith("valid", StringComparison.Ordinal),
            _ => !timestamp.Passed,
        }, $"{what}: timestamp {stamped ?? "none"}: {timestamp.Detail}");
        string modifications = (string)results["modifications"]!, level = (string)results["modificationLevel"]!;
        Check(changes.Passed == (modifications != "not-allowed"), $"{what}: changes {modifications}: {changes.Detail}");
        string words = level switch
        {
            "form-filling" => "form filling",
            "annotations" => "annotations",
            "other" => "other changes",
            _ => modifications == "none" ? "nothing changed" : "validation data",
        };
        Check(changes.Detail.Contains(words, StringComparison.Ordinal), $"{what}: the changes' level {level}: {changes.Detail}");
        bool valid = digest.Passed && signature.Passed && trusted && revoked != "revoked" && modifications != "not-allowed" && (stamped is null or "valid");
        Check(check.Valid == valid, $"{what}: valid {check.Valid}, expected {valid}: {all}");
    }

    private static void PdfSignatureReasons(Device device)
    {
        CpuOnly(device);
        var pki = PdfSignaturePki();
        SignatureCheck First(string sample, SignatureCheckOptions options)
        {
            using var opened = PdfEditOpen(File.ReadAllBytes(PdfEditSample(sample).Path), sample);
            return opened.Signatures[0].Check(options);
        }

        var broken = First("broken-digest", pki.Trust());
        Check(PdfStep(broken, SignatureCheck.Digest).Detail.Contains("changed after signing", StringComparison.Ordinal) && PdfStep(broken, SignatureCheck.Signature).Passed,
            $"a changed byte: {PdfDescribe(broken)}");
        var wrong = First("wrong-chain", pki.Trust());
        Check(PdfStep(wrong, SignatureCheck.Chain).Detail.Contains("Example Other Root CA", StringComparison.Ordinal) && PdfStep(wrong, SignatureCheck.Chain).Detail.Contains("not trusted", StringComparison.Ordinal),
            $"an untrusted root named: {PdfStep(wrong, SignatureCheck.Chain).Detail}");
        var revoked = First("pades-revoked", pki.Trust());
        string revocation = PdfStep(revoked, SignatureCheck.Revocation).Detail;
        Check(revocation.Contains("Example Revoked Signer revoked on 2026-02-01", StringComparison.Ordinal) && revocation.Contains("keyCompromise", StringComparison.Ordinal),
            $"a revoked signer, its date and reason: {revocation}");
        var changed = First("changed-not-allowed", pki.Trust());
        Check(PdfStep(changed, SignatureCheck.Changes).Detail.Contains("page's content", StringComparison.Ordinal) && PdfStep(changed, SignatureCheck.Changes).Detail.Contains("permission 2", StringComparison.Ordinal),
            $"content changed after certification: {PdfStep(changed, SignatureCheck.Changes).Detail}");

        // Decision 16: nothing is trusted that the app did not give; a document's own root never is.
        var none = First("pades-bb", new SignatureCheckOptions { At = PdfValidationTime, Online = false });
        Check(!none.Valid && PdfStep(none, SignatureCheck.Chain).Detail.Contains("no certificate is trusted", StringComparison.Ordinal), $"no trust given: {PdfDescribe(none)}");
        var other = First("wrong-chain", new SignatureCheckOptions { Trusted = [pki.Certificate("root")], At = PdfValidationTime, Online = false, UseMachineStore = true });
        Check(!other.Valid, $"the machine's store does not trust the test PKI's other root: {PdfDescribe(other)}");
        var intermediate = First("pades-bb", new SignatureCheckOptions { Trusted = [pki.Certificate("intermediate")], At = PdfValidationTime, Online = false });
        Check(intermediate.Valid && PdfStep(intermediate, SignatureCheck.Chain).Detail.Contains("Example Test Intermediate CA", StringComparison.Ordinal), $"the intermediate trusted: {PdfDescribe(intermediate)}");
        var itself = First("pades-bb", new SignatureCheckOptions { Trusted = [pki.Certificate("signer")], At = PdfValidationTime, Online = false });
        Check(itself.Valid, $"the signer's own certificate trusted: {PdfDescribe(itself)}");
        var late = First("pades-bb", pki.Trust(new DateTimeOffset(2036, 1, 1, 0, 0, 0, TimeSpan.Zero)));
        Check(!PdfStep(late, SignatureCheck.Chain).Passed && PdfStep(late, SignatureCheck.Chain).Detail.Contains("not valid at 2036-01-01", StringComparison.Ordinal), $"checked after the signer's certificate expired: {PdfStep(late, SignatureCheck.Chain).Detail}");
        var required = First("pades-bb", pki.Trust() with { RequireRevocation = true });
        Check(!required.Valid && !PdfStep(required, SignatureCheck.Revocation).Passed && PdfStep(required, SignatureCheck.Revocation).Detail.StartsWith("not checked", StringComparison.Ordinal),
            $"revocation required without data: {PdfStep(required, SignatureCheck.Revocation).Detail}");
        var blt = First("pades-blt", pki.Trust() with { RequireRevocation = true });
        Check(blt.Valid, $"B-LT with revocation required, from the file alone: {PdfDescribe(blt)}");

        // A disposed document's signature is not checked.
        var document = PdfEditOpen(File.ReadAllBytes(PdfEditSample("pades-bb").Path), "pades-bb");
        var signature = document.Signatures[0];
        document.Dispose();
        ExpectRefusal<ObjectDisposedException>(() => signature.Check(pki.Trust()), "", "a disposed document's signature checked");
    }

    // Signature trust completed (2026-11-19) on the generator's trust samples: the policies a check names, the options for
    // the initial policies and flags, an indirect and a delta list in the revocation step, stale data (not checked offline,
    // fetched again online, fresh with a grace), and each source's signature verified once a check.
    private static void PdfSignatureTrust(Device device)
    {
        CpuOnly(device);
        var pki = PdfSignaturePki();
        SignatureCheck First(string sample, SignatureCheckOptions options)
        {
            using var opened = PdfEditOpen(File.ReadAllBytes(PdfEditSample(sample).Path), sample);
            return opened.Signatures[0].Check(options);
        }

        // The policies a path is valid for: by default, for the policies the app asks for, none for a path that states none.
        var mapped = First("trust-policy-mapping", pki.Trust());
        Check(mapped.Valid && mapped.Policies.SequenceEqual(["2.999.2.1"]) && PdfStep(mapped, SignatureCheck.Chain).Detail.Contains("for the policy 2.999.2.1", StringComparison.Ordinal),
            $"a mapped policy named by its trusted side's identifier: {string.Join(", ", mapped.Policies)}; {PdfStep(mapped, SignatureCheck.Chain).Detail}");
        var asked = First("trust-policy-mapping", pki.Trust() with { InitialPolicies = ["2.999.2.1"], RequireExplicitPolicy = true });
        Check(asked.Valid && asked.Policies.SequenceEqual(["2.999.2.1"]), $"the policy asked for, required: {PdfDescribe(asked)}");
        var other = First("trust-policy-any", pki.Trust() with { InitialPolicies = ["2.999.2.2"] });
        Check(other.Valid && other.Policies.Count == 0, $"a path for none of the policies asked for passes unless one is required: {PdfDescribe(other)}");
        var plain = First("pades-bb", pki.Trust());
        Check(plain.Valid && plain.Policies.Count == 0 && !PdfStep(plain, SignatureCheck.Chain).Detail.Contains("polic", StringComparison.Ordinal),
            $"a path stating no policy says nothing of them: {PdfStep(plain, SignatureCheck.Chain).Detail}");
        var required = First("pades-bb", pki.Trust() with { RequireExplicitPolicy = true });
        string requiredWhy = PdfStep(required, SignatureCheck.Chain).Detail;
        Check(!required.Valid && requiredWhy.Contains("leaves the path no valid certificate policy", StringComparison.Ordinal)
              && requiredWhy.Contains(SignatureCheckOptions.RequireExplicitPolicyVariable, StringComparison.Ordinal), $"an explicit policy required of a path stating none: {requiredWhy}");

        // An indirect list and a delta list answering for the signer.
        var indirect = First("trust-crl-indirect-good", pki.Trust() with { RequireRevocation = true });
        Check(indirect.Valid && PdfStep(indirect, SignatureCheck.Revocation).Detail.Contains("not on Example Test CRL Issuer's indirect list", StringComparison.Ordinal),
            $"an indirect list: {PdfDescribe(indirect)}");
        var released = First("trust-crl-delta-released", pki.Trust() with { RequireRevocation = true });
        Check(released.Valid && PdfStep(released, SignatureCheck.Revocation).Detail.Contains("taken off Example Test Intermediate CA's list of 2026-03-01 00:00:00 UTC (removeFromCRL) by Example Test Intermediate CA's delta list", StringComparison.Ordinal),
            $"a hold the delta list releases: {PdfDescribe(released)}");

        // Stale data: not checked (stale) offline, fresh with a grace long enough, fetched again online.
        var stale = First("trust-stale-revocation", pki.Trust());
        string staleWhy = PdfStep(stale, SignatureCheck.Revocation).Detail;
        Check(stale.Valid && staleWhy.StartsWith("not checked (stale): ", StringComparison.Ordinal) && staleWhy.Contains("is stale at 2026-03-09 12:00:00 UTC (next update 2026-02-27", StringComparison.Ordinal),
            $"stale data offline: {staleWhy}");
        var shortGrace = First("trust-stale-revocation", pki.Trust() with { RequireRevocation = true, FreshnessGrace = TimeSpan.FromDays(3) });
        Check(!shortGrace.Valid && PdfStep(shortGrace, SignatureCheck.Revocation).Detail.Contains("taken 3 days past it", StringComparison.Ordinal),
            $"a grace too short: {PdfStep(shortGrace, SignatureCheck.Revocation).Detail}");
        var graced = First("trust-stale-revocation", pki.Trust() with { RequireRevocation = true, FreshnessGrace = TimeSpan.FromDays(14) });
        Check(graced.Valid && PdfStep(graced, SignatureCheck.Revocation).Detail.StartsWith("good", StringComparison.Ordinal), $"a grace of 14 days: {PdfDescribe(graced)}");
        // Each list's and response's signature verified once a check: pades-blt's signer response and two lists, asked about
        // five times between the timestamp authority's path and the signer's; the next check verifies them again.
        using var document = PdfEditOpen(File.ReadAllBytes(PdfEditSample("pades-blt").Path), "pades-blt");
        var signature = document.Signatures[0];
        CmsVerifications.Made = 0;
        var first = signature.Check(pki.Trust() with { RequireRevocation = true });
        int once = CmsVerifications.Made;
        var second = signature.Check(pki.Trust() with { RequireRevocation = true });
        int again = CmsVerifications.Made - once;
        Check(first.Valid && second.Valid && once == 3 && again == 3, $"the sources' signatures verified {once} times in a check and {again} in the next (3 each expected): {PdfDescribe(first)}");
    }

    // A document of two pages of generic text to sign.
    private static byte[] PdfSigningSource()
    {
        var face = SubsetFace();
        using var destination = new MemoryStream();
        using (var builder = new PdfWriter().Create(destination, PdfWriteDefaults))
        {
            builder.Info = new DocumentInfo { Title = "Quarterly report" };
            foreach (string line in new[] { "Revenue rose by 12 percent.", "Approved by the finance team." })
            {
                var page = builder.AddPage(420, 300);
                page.Glyphs(face, 14, PdfShaped(face, line, "Latn", false, 40, 60, 14), line, DocumentTextMode.Fill, DocumentColour.Black);
            }

            builder.Finish();
        }

        return destination.ToArray();
    }

    private static SigningOptions PdfSigningOptions(PdfPki pki, DocumentSignatureLevel level, string key = "signer", bool pss = false) =>
        new(pki.WithKey(key))
        {
            Chain = [pki.Certificate("intermediate")], Level = level, RsaPadding = pss ? RSASignaturePadding.Pss : RSASignaturePadding.Pkcs1,
            TimestampAuthority = level == DocumentSignatureLevel.BaselineB ? null : PdfTestAuthority, SigningTime = PdfSigningTime,
            Revocation = level == DocumentSignatureLevel.BaselineLt ? [pki.File("pki.root.crl"), pki.File("pki.intermediate.crl"), pki.File("pki.signer.ocsp")] : [],
            Reason = "Approved", Location = "Head office", Page = 0, Box = new DocumentBox(220, 210, 170, 50),
            AppearanceText = $"Signed by Example Test Signer\n{PdfSigningTime:yyyy-MM-dd}",
        };

    // A signed file saved for pyHanko: NAME.pdf, and NAME.json with each signature's field and what it must check as.
    private static void PdfSaveSigned(string name, byte[] bytes, IReadOnlyList<DocumentSignature> signatures, IReadOnlyList<SignatureCheck> checks, string folder = "signed")
    {
        string path = RepositoryPath("test-output", "documents", folder);
        Directory.CreateDirectory(path);
        File.WriteAllBytes(Path.Combine(path, name + ".pdf"), bytes);
        var list = new JsonArray();
        for (int i = 0; i < signatures.Count; i++)
        {
            list.Add(new JsonObject
            {
                ["field"] = signatures[i].Field, ["kind"] = signatures[i].Kind.ToString(), ["coversWholeFile"] = signatures[i].CoversWholeFile,
                ["valid"] = checks[i].Valid, ["timestamp"] = signatures[i].TimestampedAt is not null,
                ["revocation"] = PdfStep(checks[i], SignatureCheck.Revocation).Detail.StartsWith("good", StringComparison.Ordinal) ? "good" : "not-checked",
                ["changes"] = signatures[i].Changes.ToString(),
            });
        }

        var expected = new JsonObject
        {
            ["schema"] = 1, ["name"] = name, ["file"] = name + ".pdf", ["trust"] = new JsonArray("pki.root.cer"), ["validationTime"] = PdfValidationTime.ToString("O", CultureInfo.InvariantCulture),
            ["pki"] = Path.GetFullPath(PdfSignaturePki().Folder), ["signatures"] = list,
        };
        File.WriteAllText(Path.Combine(path, name + ".json"), expected.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }) + "\n");
    }

    // A signed file checked: every signature valid, every step passed (none failed), and saved for pyHanko.
    private static List<SignatureCheck> PdfExpectSigned(string name, byte[] bytes, PdfPki pki, string? password = null, string folder = "signed")
    {
        using var document = PdfEditOpen(bytes, name, password);
        var checks = document.Signatures.Select(s => s.Check(pki.Trust())).ToList();
        for (int i = 0; i < checks.Count; i++)
        {
            Check(checks[i].Valid, $"{name}: {document.Signatures[i].Field} checks valid: {PdfDescribe(checks[i])}");
        }

        PdfSaveSigned(name, bytes, document.Signatures, checks, folder);
        return checks;
    }

    private static void PdfSignatureSigning(Device device)
    {
        CpuOnly(device);
        var pki = PdfSignaturePki();
        byte[] source = PdfSigningSource();
        var cases = new (string Name, DocumentSignatureLevel Level, string Key, bool Pss, DocumentSignatureKind Kind)[]
        {
            ("signed-bb-rsa", DocumentSignatureLevel.BaselineB, "signer", false, DocumentSignatureKind.BaselineB),
            ("signed-bb-rsa-pss", DocumentSignatureLevel.BaselineB, "signer", true, DocumentSignatureKind.BaselineB),
            ("signed-bb-ecdsa", DocumentSignatureLevel.BaselineB, "signer-ec", false, DocumentSignatureKind.BaselineB),
        };
        byte[]? first = null;
        foreach (var (name, level, key, pss, kind) in cases)
        {
            using var document = PdfEditOpen(source, name);
            byte[] signed = PdfEditSave(document, e => e.Sign("Signature1", PdfSigningOptions(pki, level, key, pss)), DocumentSaveMode.Incremental);
            Check(signed.AsSpan(0, source.Length).SequenceEqual(source), $"{name}: the source's bytes are its prefix");
            var checks = PdfExpectSigned(name, signed, pki);
            using var read = PdfEditOpen(signed, name);
            var signature = read.Signatures.Single();
            Check(signature.Kind == kind && signature.Field == "Signature1" && signature.SignedAt == PdfSigningTime && signature.Reason == "Approved"
                  && signature.Changes == (level == DocumentSignatureLevel.BaselineLt ? DocumentSignatureChanges.Allowed : DocumentSignatureChanges.None),
                $"{name}: {signature.Kind}, {signature.Changes}, signed at {signature.SignedAt}");
            var check = checks.Single();
            Check(PdfStep(check, SignatureCheck.Revocation).Detail.StartsWith(level == DocumentSignatureLevel.BaselineLt ? "good" : "not checked", StringComparison.Ordinal)
                  && PdfStep(check, SignatureCheck.Timestamp).Detail.StartsWith(level == DocumentSignatureLevel.BaselineB ? "none" : "valid", StringComparison.Ordinal),
                $"{name}: {PdfDescribe(check)}");
            first ??= name == "signed-bb-rsa" ? signed : null;
            Check(read.PageCount == 2, $"{name}: {read.PageCount} pages");
        }

        // A second signature (ECDSA, a field of its own on page 2) keeps the first valid: its changes are form filling.
        using (var once = PdfEditOpen(first!, "signed-bb-rsa"))
        {
            byte[] twice = PdfEditSave(once, e => e.Sign("Signature2", PdfSigningOptions(pki, DocumentSignatureLevel.BaselineB, "signer-ec") with { Page = 1 }), DocumentSaveMode.Incremental);
            Check(twice.AsSpan(0, first!.Length).SequenceEqual(first), "a second signature keeps the first's bytes");
            PdfExpectSigned("signed-twice", twice, pki);
            using var read = PdfEditOpen(twice, "signed-twice");
            var firstSignature = read.Signatures[0];
            var checkFirst = firstSignature.Check(pki.Trust());
            Check(read.Signatures.Count == 2 && firstSignature.Changes == DocumentSignatureChanges.Allowed && read.Signatures[1].CoversWholeFile
                  && PdfStep(checkFirst, SignatureCheck.Changes).Detail.Contains("form filling", StringComparison.Ordinal), $"the first signature after the second: {PdfDescribe(checkFirst)}");
        }

        // The builder signs at Finish: the file written whole, then the signature as an update.
        using (var destination = new MemoryStream())
        {
            var face = SubsetFace();
            using (var builder = new PdfWriter().Create(destination, PdfWriteDefaults))
            {
                builder.Sign("Approval", PdfSigningOptions(pki, DocumentSignatureLevel.BaselineB) with { Box = null, AppearanceText = null });
                var page = builder.AddPage(300, 200);
                page.Glyphs(face, 12, PdfShaped(face, "Signed when finished", "Latn", false, 20, 40, 12), "Signed when finished", DocumentTextMode.Fill, DocumentColour.Black);
                builder.Finish();
            }

            byte[] built = destination.ToArray();
            PdfExpectSigned("signed-by-builder", built, pki);
            using var read = PdfEditOpen(built, "signed-by-builder");
            Check(read.Signatures.Single().Field == "Approval" && read.File.Xref.Sections.Count == 2 && read.ReadPage(0, DocumentParts.Text).Text == "Signed when finished",
                $"the builder's signature, as an update ({read.File.Xref.Sections.Count} sections)");
        }

        // After other edits (the information, a page turned) saved as an update, and after a rewrite: the signature covers all.
        using (var edited = PdfEditOpen(source, "edited"))
        {
            byte[] signed = PdfEditSave(edited, e =>
            {
                e.SetInfo(new DocumentInfo { Title = "Quarterly report, approved" });
                e.RotatePage(1, 90);
                e.Sign("Signature1", PdfSigningOptions(pki, DocumentSignatureLevel.BaselineB));
            }, DocumentSaveMode.Incremental);
            Check(signed.AsSpan(0, source.Length).SequenceEqual(source), "edited then signed: the source's bytes its prefix");
            PdfExpectSigned("signed-after-edits", signed, pki);
            using var read = PdfEditOpen(signed, "signed-after-edits");
            Check(read.Info.Title == "Quarterly report, approved" && read.ReadPage(1, DocumentParts.None).Rotation == 90 && read.Signatures.Single().CoversWholeFile, "the edits and the signature");
            byte[] rewritten = PdfEditSave(read, e => e.Sign("Signature2", PdfSigningOptions(pki, DocumentSignatureLevel.BaselineB) with { Page = 1 }), DocumentSaveMode.Rewrite);
            using var clean = PdfEditOpen(rewritten, "signed-after-rewrite");
            var signatures = clean.Signatures;
            var added = signatures.SingleOrDefault(s => s.Field == "Signature2");
            Check(signatures.Count == 2 && added is not null && added.CoversWholeFile && added.Check(pki.Trust()).Valid,
                $"a rewrite then signed: {signatures.Count} signatures, the new one valid");
        }
    }

    private static void PdfSignatureRefusals(Device device)
    {
        CpuOnly(device);
        var pki = PdfSignaturePki();
        byte[] source = PdfSigningSource();
        var options = PdfSigningOptions(pki, DocumentSignatureLevel.BaselineB);
        using var document = PdfEditOpen(source, "refusals");
        using (var editor = new PdfWriter().Edit(document, new MemoryStream(), PdfWriteDefaults))
        {
            ExpectRefusal<ArgumentNullException>(() => editor.Sign("Signature1", null!), "options", "no options");
            ExpectRefusal<ArgumentException>(() => editor.Sign("Signature1", new SigningOptions(pki.Certificate("signer"))), "no private key", "a certificate without its key");
            ExpectRefusal<ArgumentException>(() => editor.Sign("Signature1", options with { Digest = HashAlgorithmName.SHA1 }), "SHA-1 is read only", "SHA-1");
            ExpectRefusal<ArgumentException>(() => editor.Sign("Signature1", options with { Level = DocumentSignatureLevel.BaselineT }), "SigningOptions.TimestampAuthority", "B-T without an authority");
            ExpectRefusal<ArgumentException>(() => editor.Sign("Signature1", options with { Level = DocumentSignatureLevel.BaselineLt, TimestampAuthority = PdfTestAuthority, Online = false }),
                "SigningOptions.Revocation", "B-LT without revocation data");
        }

        void Refused<TException>(byte[] bytes, string field, SigningOptions signing, string text, string what) where TException : Exception
        {
            using var target = PdfEditOpen(bytes, what);
            ExpectRefusal<TException>(() => PdfEditSave(target, e => e.Sign(field, signing), DocumentSaveMode.Incremental), text, what);
        }

        byte[] form = File.ReadAllBytes(PdfEditSample("forms-empty").Path);
        Refused<ArgumentException>(form, "name", options, "not a signature field", "a text field");
        Refused<InvalidOperationException>(File.ReadAllBytes(PdfEditSample("pades-bb").Path), "Signature1", options, "signed already", "a field signed already");
        Refused<ArgumentException>(source, "approvals.first", options, "without periods", "a dotted new field");
        Refused<ArgumentOutOfRangeException>(source, "Signature1", options with { Page = 5 }, "2 pages", "a page out of range");
        using var builder = new PdfWriter().Create(new MemoryStream(), PdfWriteDefaults);
        builder.AddPage(100, 100);
        ExpectRefusal<InvalidOperationException>(() => builder.Sign("Signature1", options), "before the first page", "the builder signing after its first page");
        builder.Finish();
    }

    private static void PdfSignatureTimings(Device device)
    {
        CpuOnly(device);
        var pki = PdfSignaturePki();
        var samples = PdfSignedSamples().Select(s => (s.Name, Bytes: File.ReadAllBytes(s.Path), Trust: pki.Trust())).ToList();

        // Listing (the signatures read, their byte ranges, what changed after each) and checking, every signature.
        (int Signatures, long Listed, long Checked) Pass(out double listMs, out double checkMs)
        {
            int count = 0;
            long listed = 0, checkedBytes = 0;
            listMs = checkMs = 0;
            foreach (var (name, bytes, trust) in samples)
            {
                long before = GC.GetAllocatedBytesForCurrentThread();
                long start = Stopwatch.GetTimestamp();
                using var document = PdfDocument.Open(new MemoryStream(bytes, writable: false), name, new DocumentOptions());
                var signatures = document.Signatures;
                listMs += Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                listed += GC.GetAllocatedBytesForCurrentThread() - before;
                before = GC.GetAllocatedBytesForCurrentThread();
                start = Stopwatch.GetTimestamp();
                foreach (var signature in signatures)
                {
                    _ = signature.Check(trust);
                    count++;
                }

                checkMs += Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                checkedBytes += GC.GetAllocatedBytesForCurrentThread() - before;
            }

            return (count, listed, checkedBytes);
        }

        var cold = Pass(out double coldList, out double coldCheck);
        var warmList = new List<double>();
        var warmCheck = new List<double>();
        (int Signatures, long Listed, long Checked) warm = default;
        for (int i = 0; i < 5; i++)
        {
            warm = Pass(out double list, out double check);
            warmList.Add(list);
            warmCheck.Add(check);
        }

        warmList.Sort();
        warmCheck.Sort();
        int n = warm.Signatures;
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    {samples.Count} D12 samples, {n} signatures: listed (read, byte ranges, changes) cold {coldList:F1} ms, warm {warmList[2]:F2} ms ({warmList[2] * 1000 / n:F0} µs and {warm.Listed / 1024.0 / n:F1} KiB a signature)"));
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    checked (every step): cold {coldCheck:F1} ms, warm {warmCheck[2]:F2} ms ({warmCheck[2] * 1000 / n:F0} µs and {warm.Checked / 1024.0 / n:F1} KiB a signature)"));
        Check(cold.Signatures == n && n >= 14, $"{n} signatures timed");

        // Signing a page at B-B.
        byte[] source = PdfSigningSource();
        foreach (var level in new[] { DocumentSignatureLevel.BaselineB })
        {
            var options = PdfSigningOptions(pki, level);
            long Sign()
            {
                using var document = PdfDocument.Open(new MemoryStream(source, writable: false), "to sign", new DocumentOptions());
                using var counter = new PdfByteCounter();
                using (var editor = new PdfWriter().Edit(document, counter, PdfWriteDefaults))
                {
                    editor.Sign("Signature1", options);
                    editor.Save(DocumentSaveMode.Incremental);
                }

                return counter.Length;
            }

            long start = Stopwatch.GetTimestamp();
            long written = Sign();
            double first = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            var times = new List<double>();
            long allocated = 0;
            for (int i = 0; i < 5; i++)
            {
                long before = GC.GetAllocatedBytesForCurrentThread();
                start = Stopwatch.GetTimestamp();
                Sign();
                times.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds);
                allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            }

            times.Sort();
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"    a page signed at {level}: {written - source.Length:N0} bytes added to {source.Length:N0}, cold {first:F1} ms, warm {times[2]:F2} ms, {allocated / 1024.0:F0} KiB allocated"));
        }
    }
}
