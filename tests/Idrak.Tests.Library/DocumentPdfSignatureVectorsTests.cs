// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Formats.Asn1;
using System.Globalization;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Data.Pdf;

// Plan 28, D12 against known-bad cases (owner, 2026-11-19, decision 1a: Idrak keeps its own RSA verification and path
// building, the same on every platform, and is tested against public known-bad cases; every failure is fixed in the library,
// or named as a known limit: printed "LIMIT: <case>: <why>", in src/Idrak.Data/README.md and in plans/leftovers.md). Project
// Wycheproof's RSA PKCS #1 v1.5, RSA-PSS and ECDSA vectors (tests/Idrak.Tests/data/wycheproof, their notice and commit in its
// README) through CmsKeys; this platform's RSA cross-checked on the cases it takes (a test-only guard, never a library path);
// the generator's known-bad certificate paths (D12's bad-* samples in the ignored test-output/documents/samples);
// x509-limbo's path and revocation-list cases (tests/Idrak.Tests/data/x509-limbo), name constraints and policies among them
// (signature trust completed, 2026-11-19); the generator's trust samples refused (name constraints, policies, delta and
// indirect lists, stale data); the path search's bound. Each test keeps what it found in
// test-output/documents/known-bad/NAME.txt (every case counted, acceptable cases and limits named).
internal abstract partial class Tests
{
    private static readonly string[] WycheproofPkcs1Files =
    [
        "rsa_signature_2048_sha256_test", "rsa_signature_2048_sha384_test", "rsa_signature_2048_sha512_test",
        "rsa_signature_3072_sha256_test", "rsa_signature_3072_sha384_test", "rsa_signature_3072_sha512_test",
        "rsa_signature_4096_sha256_test", "rsa_signature_4096_sha384_test", "rsa_signature_4096_sha512_test",
        "rsa_pkcs1_2048_sig_gen_test", "rsa_pkcs1_3072_sig_gen_test", "rsa_pkcs1_4096_sig_gen_test",
    ];

    private static readonly string[] WycheproofPssFiles =
    [
        "rsa_pss_2048_sha1_mgf1_20_test", "rsa_pss_2048_sha256_mgf1_0_test", "rsa_pss_2048_sha256_mgf1_32_test",
        "rsa_pss_2048_sha256_mgf1sha1_20_test", "rsa_pss_2048_sha384_mgf1_48_test", "rsa_pss_3072_sha256_mgf1_32_test",
        "rsa_pss_4096_sha256_mgf1_32_test", "rsa_pss_4096_sha384_mgf1_48_test", "rsa_pss_4096_sha512_mgf1_32_test",
        "rsa_pss_4096_sha512_mgf1_64_test", "rsa_pss_misc_test", "rsa_pss_2048_sha256_mgf1_32_params_test", "rsa_pss_misc_params_test",
    ];

    private static readonly string[] WycheproofEcdsaFiles =
    [
        "ecdsa_secp256r1_sha256_test", "ecdsa_secp256r1_sha512_test", "ecdsa_secp384r1_sha384_test", "ecdsa_secp384r1_sha512_test",
        "ecdsa_secp521r1_sha512_test", "ecdsa_secp256k1_sha256_test", "ecdsa_brainpoolP256r1_sha256_test", "ecdsa_brainpoolP384r1_sha384_test",
        "ecdsa_brainpoolP512r1_sha512_test",
    ];

    // What a test found, printed and kept in test-output/documents/known-bad/NAME.txt for the owner (when the run has the
    // repository), written even when the test fails.
    private sealed class KnownBadReport(string name) : IDisposable
    {
        private readonly List<string> _lines = [];

        public void Line(string line)
        {
            Console.WriteLine(line);
            _lines.Add(line);
        }

        public void Dispose()
        {
            if (TestPaths.Root is { } root)
            {
                string folder = Path.Combine(root, "test-output", "documents", "known-bad");
                Directory.CreateDirectory(folder);
                File.WriteAllLines(Path.Combine(folder, name + ".txt"), _lines);
            }
        }
    }

    // One Wycheproof file's cases counted by their result and Idrak's answer; a valid case refused or an invalid one taken is
    // a failure; an acceptable one is listed with the answer.
    private sealed class VectorTally(string file)
    {
        private readonly SortedDictionary<string, int> _notRead = new(StringComparer.Ordinal);
        private int _valid, _validTaken, _invalid, _invalidRefused, _acceptable, _acceptableTaken;

        public List<string> Acceptable { get; } = [];

        public int Cases => _valid + _invalid + _acceptable;

        public void NotRead(string what, int count) => _notRead[what] = _notRead.GetValueOrDefault(what) + count;

        public void Add(JsonNode test, string? refused, List<string> failures)
        {
            string result = (string)test["result"]!;
            string flags = string.Join(", ", test["flags"]!.AsArray().Select(f => (string)f!));
            string named = $"{file} #{(int)test["tcId"]!} [{flags}]{((string?)test["comment"] is { Length: > 0 } comment ? $" {comment}" : "")}";
            switch (result)
            {
                case "valid":
                    _valid++;
                    _validTaken += refused is null ? 1 : 0;
                    if (refused is not null)
                    {
                        failures.Add($"{named}: valid, refused ({refused})");
                    }

                    break;
                case "invalid":
                    _invalid++;
                    _invalidRefused += refused is null ? 0 : 1;
                    if (refused is null)
                    {
                        failures.Add($"{named}: invalid, accepted");
                    }

                    break;
                default:
                    _acceptable++;
                    _acceptableTaken += refused is null ? 1 : 0;
                    Acceptable.Add($"  acceptable {named}: Idrak {(refused is null ? "accepts" : $"refuses ({refused})")}");
                    break;
            }
        }

        public string Summary() =>
            $"wycheproof {file}: {Cases} cases: {_valid} valid ({_validTaken} verified), {_invalid} invalid ({_invalidRefused} refused), "
            + $"{_acceptable} acceptable ({_acceptableTaken} accepted){(_notRead.Count == 0 ? "" : $"; not read by Idrak: {string.Join(", ", _notRead.Select(n => $"{n.Value} with {n.Key}"))}")}";
    }

    private static JsonObject WycheproofFile(string name)
    {
        using var stream = new GZipStream(File.OpenRead(TestData($"wycheproof/{name}.json.gz")), CompressionMode.Decompress);
        return JsonNode.Parse(stream)!.AsObject();
    }

    private static HashAlgorithmName? WycheproofDigest(string? name) => name switch
    {
        "SHA-1" => HashAlgorithmName.SHA1,
        "SHA-256" => HashAlgorithmName.SHA256,
        "SHA-384" => HashAlgorithmName.SHA384,
        "SHA-512" => HashAlgorithmName.SHA512,
        _ => null,
    };

    private static byte[] WycheproofHex(JsonNode? node) => Convert.FromHexString((string?)node ?? "");

    // A certificate holding only a key (the vectors give keys, CmsKeys takes certificates): the key's SubjectPublicKeyInfo as
    // given, with a placeholder name, dates and signature (CmsCertificate reads them; nothing here verifies them).
    private static CmsCertificate WycheproofKey(byte[] subjectPublicKeyInfo)
    {
        static void Name(AsnWriter writer)
        {
            using (writer.PushSequence())
            {
                using (writer.PushSetOf())
                {
                    using (writer.PushSequence())
                    {
                        writer.WriteObjectIdentifier("2.5.4.3");
                        writer.WriteCharacterString(UniversalTagNumber.UTF8String, "Wycheproof key");
                    }
                }
            }
        }

        static void Algorithm(AsnWriter writer)
        {
            using (writer.PushSequence())
            {
                writer.WriteObjectIdentifier("1.2.840.113549.1.1.11");
                writer.WriteNull();
            }
        }

        var tbs = new AsnWriter(AsnEncodingRules.DER);
        using (tbs.PushSequence())
        {
            using (tbs.PushSequence(new Asn1Tag(TagClass.ContextSpecific, 0, isConstructed: true)))
            {
                tbs.WriteInteger(2);
            }

            tbs.WriteInteger(1);
            Algorithm(tbs);
            Name(tbs);
            using (tbs.PushSequence())
            {
                tbs.WriteUtcTime(new DateTimeOffset(2020, 1, 1, 0, 0, 0, TimeSpan.Zero));
                tbs.WriteUtcTime(new DateTimeOffset(2040, 1, 1, 0, 0, 0, TimeSpan.Zero));
            }

            Name(tbs);
            tbs.WriteEncodedValue(subjectPublicKeyInfo);
        }

        var certificate = new AsnWriter(AsnEncodingRules.DER);
        using (certificate.PushSequence())
        {
            certificate.WriteEncodedValue(tbs.Encode());
            Algorithm(certificate);
            certificate.WriteBitString([0]);
        }

        return CmsCertificate.Parse(certificate.Encode());
    }

    private static void PdfKnownBadPkcs1(Device device)
    {
        CpuOnly(device);
        using var report = new KnownBadReport("wycheproof-rsa-pkcs1");
        var failures = new List<string>();
        int cases = 0;
        foreach (string file in WycheproofPkcs1Files)
        {
            var tally = new VectorTally(file);
            foreach (var groupNode in WycheproofFile(file)["testGroups"]!.AsArray())
            {
                var group = groupNode!;
                var tests = group["tests"]!.AsArray();
                if (WycheproofDigest((string?)group["sha"]) is not { } digest)
                {
                    tally.NotRead((string)group["sha"]!, tests.Count);
                    continue;
                }

                var key = WycheproofKey(WycheproofHex(group["publicKeyDer"] ?? group["keyDer"]));
                var scheme = new CmsScheme(CmsSchemeKind.RsaPkcs1, digest);
                foreach (var testNode in tests)
                {
                    var test = testNode!;
                    byte[] message = WycheproofHex(test["msg"]), signature = WycheproofHex(test["sig"]);
                    byte[] hash = CmsDigest.Of(digest, message);
                    string? refused = CmsKeys.Verify(key, scheme, hash, signature);

                    // The legacy PKCS #1 form's route (adbe.x509.rsa_sha1): the digest the signature opens to, verified as it.
                    bool legacy = CmsKeys.Recover(key, signature) is { } carried && carried.Digest == digest && carried.Hash.AsSpan().SequenceEqual(hash)
                                  && CmsKeys.Verify(key, new CmsScheme(CmsSchemeKind.RsaPkcs1, carried.Digest), carried.Hash, signature) is null;
                    if (legacy != (refused is null))
                    {
                        failures.Add($"{file} #{(int)test["tcId"]!}: the legacy form's route {(legacy ? "accepts" : "refuses")}, Verify {(refused is null ? "accepts" : "refuses")}");
                    }

                    tally.Add(test, refused, failures);
                }
            }

            report.Line(tally.Summary());
            tally.Acceptable.ForEach(report.Line);
            cases += tally.Cases;
        }

        report.Line($"RSA PKCS #1 v1.5: {cases} cases in {WycheproofPkcs1Files.Length} files, {failures.Count} differ from Wycheproof's verdict");
        Check(cases > 2400, $"{cases} Wycheproof PKCS #1 cases read (more than 2400 expected)");
        Check(failures.Count == 0, $"{failures.Count} Wycheproof PKCS #1 cases differ: {string.Join("; ", failures.Take(20))}");
    }

    private static void PdfKnownBadPss(Device device)
    {
        CpuOnly(device);
        using var report = new KnownBadReport("wycheproof-rsa-pss");
        var failures = new List<string>();
        int cases = 0;
        foreach (string file in WycheproofPssFiles)
        {
            var tally = new VectorTally(file);
            foreach (var groupNode in WycheproofFile(file)["testGroups"]!.AsArray())
            {
                var group = groupNode!;
                var tests = group["tests"]!.AsArray();
                string sha = (string)group["sha"]!, mgf = (string)group["mgf"]!, mgfSha = (string)group["mgfSha"]!;
                int salt = (int)group["sLen"]!;
                if (WycheproofDigest(sha) is not { } digest || WycheproofDigest(mgfSha) is not { } mask || mgf != "MGF1")
                {
                    tally.NotRead($"{sha}, {mgf} {mgfSha}", tests.Count);
                    continue;
                }

                var key = WycheproofKey(WycheproofHex(group["publicKeyDer"]));
                var expected = new CmsScheme(CmsSchemeKind.RsaPss, digest, mask, salt);
                var scheme = expected;
                if ((string)group["type"]! == "RsassaPssWithParametersVerify")
                {
                    // The key is certified for RSA-PSS with its parameters (RFC 4055): the scheme is the one Idrak reads from them.
                    if (CmsKeys.Scheme(key.KeyAlgorithm, null, out string? problem) is not { } read || read != expected)
                    {
                        failures.Add($"{file}: the key's RSA-PSS parameters read as {(problem ?? CmsKeys.Scheme(key.KeyAlgorithm, null, out _).ToString())}, the group says {expected}");
                        continue;
                    }

                    scheme = read;
                }

                foreach (var testNode in tests)
                {
                    var test = testNode!;
                    byte[] hash = CmsDigest.Of(digest, WycheproofHex(test["msg"]));
                    tally.Add(test, CmsKeys.Verify(key, scheme, hash, WycheproofHex(test["sig"])), failures);
                }
            }

            report.Line(tally.Summary());
            tally.Acceptable.ForEach(report.Line);
            cases += tally.Cases;
        }

        report.Line($"RSA-PSS: {cases} cases in {WycheproofPssFiles.Length} files, {failures.Count} differ from Wycheproof's verdict");
        Check(cases > 1000, $"{cases} Wycheproof PSS cases read (more than 1000 expected)");
        Check(failures.Count == 0, $"{failures.Count} Wycheproof PSS cases differ: {string.Join("; ", failures.Take(20))}");
    }

    private static void PdfKnownBadEcdsa(Device device)
    {
        CpuOnly(device);
        using var report = new KnownBadReport("wycheproof-ecdsa");
        var failures = new List<string>();
        int cases = 0;
        foreach (string file in WycheproofEcdsaFiles)
        {
            var tally = new VectorTally(file);
            string? limit = null;
            foreach (var groupNode in WycheproofFile(file)["testGroups"]!.AsArray())
            {
                var group = groupNode!;
                var tests = group["tests"]!.AsArray();
                if (WycheproofDigest((string?)group["sha"]) is not { } digest)
                {
                    tally.NotRead((string)group["sha"]!, tests.Count);
                    continue;
                }

                // The curve arithmetic is the base library's: a curve this platform's ECDsa does not take is a named limit.
                byte[] spki = WycheproofHex(group["publicKeyDer"]);
                string curve = (string)group["publicKey"]!["curve"]!;
                try
                {
                    using var platform = ECDsa.Create();
                    platform.ImportSubjectPublicKeyInfo(spki, out _);
                }
                catch (Exception e) when (e is CryptographicException or PlatformNotSupportedException)
                {
                    limit ??= $"LIMIT: wycheproof {file}: this platform's ECDsa does not take {curve} keys ({e.Message}): Idrak verifies ECDSA through the base library's curve arithmetic, so the curves are the platform's (P-256, P-384 and P-521 on every platform)";
                    tally.NotRead($"{curve} (a limit here)", tests.Count);
                    continue;
                }

                var key = WycheproofKey(spki);
                var scheme = new CmsScheme(CmsSchemeKind.Ecdsa, digest);
                foreach (var testNode in tests)
                {
                    var test = testNode!;
                    byte[] hash = CmsDigest.Of(digest, WycheproofHex(test["msg"]));
                    tally.Add(test, CmsKeys.Verify(key, scheme, hash, WycheproofHex(test["sig"])), failures);
                }
            }

            report.Line(tally.Summary());
            tally.Acceptable.ForEach(report.Line);
            if (limit is not null)
            {
                report.Line(limit);
            }

            cases += tally.Cases;
        }

        report.Line($"ECDSA: {cases} cases in {WycheproofEcdsaFiles.Length} files, {failures.Count} differ from Wycheproof's verdict");
        Check(cases > 1500, $"{cases} Wycheproof ECDSA cases read (more than 1500 expected: P-256, P-384 and P-521 are on every platform)");
        Check(failures.Count == 0, $"{failures.Count} Wycheproof ECDSA cases differ: {string.Join("; ", failures.Take(20))}");
    }

    // A test-only guard, never a library path: this platform's RSA (RSA.VerifyHash) on every Wycheproof RSA case it takes as
    // input, its verdict required to be Idrak's on the valid and invalid cases (the acceptable ones listed where they differ:
    // Wycheproof allows either). PSS only where the platform can take the scheme: a salt as long as the digest, MGF1 with the
    // same digest. A key or an input the platform refuses (an exception) is passed over with a printed line.
    private static void PdfKnownBadPlatformRsa(Device device)
    {
        CpuOnly(device);
        using var report = new KnownBadReport("platform-rsa");
        var failures = new List<string>();
        int compared = 0, refusedInputs = 0, refusedKeys = 0, otherSchemes = 0;
        foreach (string file in WycheproofPkcs1Files.Concat(WycheproofPssFiles))
        {
            bool pss = file.StartsWith("rsa_pss", StringComparison.Ordinal);
            foreach (var groupNode in WycheproofFile(file)["testGroups"]!.AsArray())
            {
                var group = groupNode!;
                var tests = group["tests"]!.AsArray();
                if (WycheproofDigest((string?)group["sha"]) is not { } digest)
                {
                    continue;
                }

                var key = WycheproofKey(WycheproofHex(group["publicKeyDer"] ?? group["keyDer"]));
                CmsScheme scheme;
                if (pss)
                {
                    if (WycheproofDigest((string?)group["mgfSha"]) != digest || (int)group["sLen"]! != CmsDigest.Length(digest))
                    {
                        otherSchemes += tests.Count;
                        continue;
                    }

                    scheme = new CmsScheme(CmsSchemeKind.RsaPss, digest, digest, CmsDigest.Length(digest));
                }
                else
                {
                    scheme = new CmsScheme(CmsSchemeKind.RsaPkcs1, digest);
                }

                using var platform = RSA.Create();
                try
                {
                    var (modulus, exponent, _) = key.Rsa()!.Value;
                    platform.ImportParameters(new RSAParameters
                    {
                        Modulus = modulus.ToByteArray(isUnsigned: true, isBigEndian: true), Exponent = exponent.ToByteArray(isUnsigned: true, isBigEndian: true),
                    });
                }
                catch (CryptographicException e)
                {
                    report.Line($"  passed over: {file}, a group's key ({tests.Count} cases): this platform's RSA does not take it ({e.Message})");
                    refusedKeys += tests.Count;
                    continue;
                }

                foreach (var testNode in tests)
                {
                    var test = testNode!;
                    byte[] hash = CmsDigest.Of(digest, WycheproofHex(test["msg"])), signature = WycheproofHex(test["sig"]);
                    bool idrak = CmsKeys.Verify(key, scheme, hash, signature) is null, theirs;
                    try
                    {
                        theirs = platform.VerifyHash(hash, signature, digest, pss ? RSASignaturePadding.Pss : RSASignaturePadding.Pkcs1);
                    }
                    catch (CryptographicException e)
                    {
                        report.Line($"  passed over: {file} #{(int)test["tcId"]!}: this platform's RSA does not take the input ({e.Message})");
                        refusedInputs++;
                        continue;
                    }

                    compared++;
                    if (idrak != theirs)
                    {
                        string line = $"{file} #{(int)test["tcId"]!} ({(string)test["result"]!}): Idrak {(idrak ? "accepts" : "refuses")}, this platform's RSA {(theirs ? "accepts" : "refuses")}";
                        if ((string)test["result"]! == "acceptable")
                        {
                            report.Line($"  differs on an acceptable case: {line}");
                        }
                        else
                        {
                            failures.Add(line);
                        }
                    }
                }
            }
        }

        report.Line($"this platform's RSA ({RuntimeInformationText()}): {compared} cases compared, {failures.Count} valid or invalid ones differ; passed over: {refusedInputs} inputs and {refusedKeys} cases' keys it refuses, {otherSchemes} PSS cases whose salt or mask it cannot be asked for");
        Check(compared > 2000, $"{compared} cases compared with this platform's RSA (more than 2000 expected)");
        Check(failures.Count == 0, $"{failures.Count} cases where this platform's RSA and Idrak differ: {string.Join("; ", failures.Take(20))}");
    }

    private static string RuntimeInformationText() => $"{System.Runtime.InteropServices.RuntimeInformation.OSDescription.Trim()}, {System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription}";

    // The generator's known-bad samples (D12's bad-* and trust-*): each refused at the step its JSON names, saying why (with
    // the options it names, when it fails only with them); with the option that is the cause raised, it passes.
    private static void PdfKnownBadPaths(Device device)
    {
        CpuOnly(device);
        var pki = PdfSignaturePki();
        using var report = new KnownBadReport("paths");
        string[] expectedCases =
        [
            "expired", "not-yet-valid", "wrong-issuer-signature", "issuer-not-ca", "path-length", "issuer-without-keycertsign", "signer-without-signature-usage",
            "unknown-critical-extension", "names-do-not-chain", "loop", "chain-too-long", "weak-rsa-key", "revoked-by-ocsp", "crl-wrong-key", "ocsp-wrong-key",
            "timestamp-imprint",
            "name-constraints-dn-excluded", "name-constraints-dn-not-permitted", "name-constraints-email-excluded", "name-constraints-email-not-permitted",
            "name-constraints-dns-excluded", "name-constraints-dns-not-permitted", "name-constraints-uri-excluded", "name-constraints-uri-not-permitted",
            "name-constraints-ipv4-excluded", "name-constraints-ipv4-not-permitted", "name-constraints-ipv6-excluded", "name-constraints-ipv6-not-permitted",
            "name-constraints-unsupported-form", "policy-not-the-initial", "policy-explicit-required", "policy-mapping-inhibited-by-the-check",
            "policy-mapping-inhibited", "policy-any-inhibited-by-the-check", "policy-any-inhibited", "delta-crl-revoked", "indirect-crl-revoked",
            "stale-revocation",
        ];
        var found = new List<string>();
        foreach (var sample in PdfSignedSamples().Where(s => s.Expected["signatures"]!.AsArray().Any(g => g?["knownBad"] is not null)))
        {
            using var document = PdfEditOpen(File.ReadAllBytes(sample.Path), sample.Name);
            var expected = sample.Expected["signatures"]!.AsArray();
            var reference = sample.Expected["reference"]?["validator"]?["signatures"] as JsonArray;
            for (int i = 0; i < expected.Count; i++)
            {
                if (expected[i]!["knownBad"] is not JsonObject bad)
                {
                    continue;
                }

                var want = expected[i]!;
                var signature = document.Signatures[i];
                string kind = (string)bad["case"]!, stepName = (string)bad["step"]!, detail = (string)bad["detail"]!;
                var trust = want["trust"]!.AsArray().Select(t => X509CertificateLoader.LoadCertificateFromFile(Path.Combine(pki.Folder, (string)t!))).ToList();
                var options = new SignatureCheckOptions { Trusted = trust, At = DateTimeOffset.Parse((string)want["validationTime"]!, CultureInfo.InvariantCulture), Online = false };
                var check = signature.Check(options);
                PdfExpectResults(sample.Name, check, want["results"]!);
                var failing = options;
                string? failsWith = null;
                if (bad["failsWith"] is JsonValue required && (string?)required == "RequireRevocation")
                {
                    // A source that cannot be used leaves the status unknown: soft-fail passes it as not checked, saying why.
                    var soft = PdfStep(check, stepName);
                    Check(soft.Passed && soft.Detail.StartsWith("not checked", StringComparison.Ordinal) && soft.Detail.Contains(detail, StringComparison.Ordinal),
                        $"{sample.Name} ({kind}): soft-fail passes the {stepName} step as not checked, saying '{detail}': {soft.Detail}");
                    failing = options with { RequireRevocation = true };
                    failsWith = "RequireRevocation";
                }
                else if (bad["failsWith"] is JsonObject asked)
                {
                    // A path valid with the default options, refused with the policy options the app asks for.
                    Check(PdfStep(check, stepName).Passed, $"{sample.Name} ({kind}): the {stepName} step passes with the default options: {PdfDescribe(check)}");
                    failing = PdfWithOptions(options, asked);
                    failsWith = asked.ToJsonString();
                }

                check = signature.Check(failing);
                var step = PdfStep(check, stepName);
                Check(!step.Passed && !check.Valid && step.Detail.Contains(detail, StringComparison.Ordinal),
                    $"{sample.Name} ({kind}): the {stepName} step fails saying '{detail}': {PdfDescribe(check)}");
                string pyhanko = reference is not null && i < reference.Count && reference[i] is JsonObject r
                    ? $" (pyHanko: {(r["error"] is { } error ? $"stops: {error}" : $"intact {r["intact"]}, valid {r["valid"]}, trusted {r["trusted"]}")}{(r["lt"] is JsonObject lt ? $", from the DSS {lt["revocation"]}" : "")}{(r["withOptions"] is JsonObject o ? $", with those options trusted {o["trusted"]}" : "")}{(r["differsByDesign"] is JsonObject d ? $"; differs by design: {d.ToJsonString()}" : "")})"
                    : "";
                report.Line($"path case {kind} ({sample.Name}){(failsWith is null ? "" : $" with {failsWith}")}: {stepName} refused: {step.Detail}{pyhanko}");
                if (bad["passesWith"] is JsonObject passes)
                {
                    string option = (string)passes["option"]!;
                    int value = (int)passes["value"]!;
                    var lifted = option switch
                    {
                        "MaxChainLength" => failing with { MaxChainLength = value },
                        "MinimumRsaKeyBits" => failing with { MinimumRsaKeyBits = value },
                        "FreshnessGrace" => failing with { FreshnessGrace = TimeSpan.FromSeconds(value) },
                        _ => throw new InvalidOperationException($"an option the test does not know: {option}"),
                    };
                    var again = signature.Check(lifted);
                    Check(PdfStep(again, stepName).Passed && again.Valid, $"{sample.Name} ({kind}) with {option} = {value}: {PdfDescribe(again)}");
                    report.Line($"  with {option} = {value}: the {stepName} step passes ({PdfStep(again, stepName).Detail})");
                }

                found.Add(kind);
            }
        }

        // A signer revoked by the file's CRL is D12's own pades-revoked.
        using (var revoked = PdfEditOpen(File.ReadAllBytes(PdfEditSample("pades-revoked").Path), "pades-revoked"))
        {
            var step = PdfStep(revoked.Signatures[0].Check(pki.Trust()), SignatureCheck.Revocation);
            Check(!step.Passed && step.Detail.Contains("revoked on 2026-02-01", StringComparison.Ordinal), $"pades-revoked: {step.Detail}");
            report.Line($"path case revoked-by-crl (pades-revoked): revocation refused: {step.Detail}");
        }

        report.Line($"{found.Count + 1} known-bad path cases, each refused at its step");
        Check(found.Order(StringComparer.Ordinal).SequenceEqual(expectedCases.Order(StringComparer.Ordinal)),
            $"the known-bad samples' cases: {string.Join(", ", found)}; expected {string.Join(", ", expectedCases)} (make the samples again: tools/documents/make_fixtures.py)");
    }

    // A check's options with the policy options a sample's JSON names (InitialPolicies, RequireExplicitPolicy,
    // InhibitPolicyMapping, InhibitAnyPolicy).
    private static SignatureCheckOptions PdfWithOptions(SignatureCheckOptions options, JsonObject asked)
    {
        foreach (var (name, value) in asked)
        {
            options = name switch
            {
                "InitialPolicies" => options with { InitialPolicies = [.. value!.AsArray().Select(p => (string)p!)] },
                "RequireExplicitPolicy" => options with { RequireExplicitPolicy = (bool)value! },
                "InhibitPolicyMapping" => options with { InhibitPolicyMapping = (bool)value! },
                "InhibitAnyPolicy" => options with { InhibitAnyPolicy = (bool)value! },
                _ => throw new InvalidOperationException($"an option the test does not know: {name}"),
            };
        }

        return options;
    }

    // x509-limbo's cases whose verdict a document's check gives otherwise on purpose: the rule it applies instead.
    private static readonly (string Case, string Why)[] LimboByDesign =
    [
        ("rfc5280::validity::expired-root", "a trust anchor's own dates are outside the path (RFC 5280 6.1; pyHanko, the reference, alike): the app's trusted certificate is the anchor"),
        ("rfc5280::aki::", "the key identifiers' presence and criticality are rules for how a CA writes certificates; a check uses them to find an issuer where given (pyHanko alike)"),
        ("rfc5280::ski::", "the key identifiers' presence and criticality are rules for how a CA writes certificates; a check uses them to find an issuer where given (pyHanko alike)"),
        ("rfc5280::serial::", "a serial number's sign and length are rules for how a CA writes certificates; a check reads any serial"),
        ("rfc5280::san::", "a subject alternative name's form matters to the server or mailbox it names; a document check names its signer by the subject"),
        ("rfc5280::eku::ee-wrong-eku", "the case asks the leaf for a server's purpose; a document's signer is asked for its key usage (digitalSignature or nonRepudiation), no extended purpose (pyHanko's default alike)"),
        ("rfc5280::root-non-critical-basic-constraints", "the criticality of a CA's basic constraints is a rule for how a CA writes certificates; a check reads them either way"),
        ("rfc5280::ee-critical-aia-invalid", "the criticality of authority information access is a rule for how a CA writes certificates; a check reads it either way"),
        ("rfc5280::ca-empty-subject", "an empty subject is a rule for how a CA writes certificates; a check chains names byte for byte"),
        ("rfc5280::ca-as-leaf", "the leaf is a CA certificate whose key usage is keyCertSign alone: a document's signer must allow digitalSignature or nonRepudiation"),
        ("pathlen::validation-ignores-pathlen-in-leaf", "the leaf is a CA certificate whose key usage is keyCertSign alone: a document's signer must allow digitalSignature or nonRepudiation"),
        ("pathlen::max-chain-depth-1-self-issued", "the case's depth leaves self-issued certificates out; SignatureCheckOptions.MaxChainLength bounds every certificate of a path (the search's work)"),
        ("crl::temporal::this-update-in-future", "a document check uses a list issued after the time it checks at: a later list proves the status then (PAdES long-term validation)"),
        ("crl::temporal::revocation-date-in-future", "a certificate revoked after the time a document is checked at was not revoked then: its signature made before stays valid (PAdES long-term validation)"),
        ("crl::crlnumber-", "the CRL number's presence and criticality are rules for how a CA writes lists; a check reads a list either way"),
        ("rfc5280::nc::permitted-dns-match-noncritical", "name constraints are applied whether marked critical or not (pyHanko alike; the CA/Browser Forum lets a CA mark them non-critical, limbo tags the case rfc5280-incompatible-with-webpki): the leaf is within them"),
        ("rfc5280::pc::ica-noncritical-pc", "policy constraints are applied whether marked critical or not (pyHanko alike): the CA's requireExplicitPolicy 3 does not reach this two-certificate path"),
    ];

    // x509-limbo's cases Idrak cannot answer as limbo does yet: a named limit (src/Idrak.Data/README.md, plans/leftovers.md).
    private static readonly (string Case, string Why)[] LimboLimits =
    [
        ("rfc9881::", "ML-DSA keys are not read (RSA and ECDSA are): a path signed with one fails"),
    ];

    private static string? LimboRule((string Case, string Why)[] table, string id) =>
        table.FirstOrDefault(r => r.Case.EndsWith("::", StringComparison.Ordinal) || r.Case.EndsWith('-') ? id.StartsWith(r.Case, StringComparison.Ordinal) : id == r.Case).Why;

    private static byte[] PemBytes(string pem)
    {
        var fields = PemEncoding.Find(pem);
        return Convert.FromBase64String(pem[fields.Base64Data]);
    }

    // One x509-limbo case through Idrak: the leaf as a document's signer, the intermediates as the certificates a document
    // carries, the trusted certificates as the app's trust, at the case's time (its depth as MaxChainLength: the leaf and the
    // anchor more); then each of the case's lists read for each link: a link revoked, or a list naming its issuer that
    // cannot be used, refuses the path.
    private static (bool Passed, string Detail) LimboRun(JsonNode limbo, SignatureCheckOptions? given = null)
    {
        static List<CmsCertificate> Read(JsonNode? list) =>
            list is JsonArray certificates ? [.. certificates.Select(p => CmsCertificate.TryParse(PemBytes((string)p!))).OfType<CmsCertificate>()] : [];

        if (CmsCertificate.TryParse(PemBytes((string)limbo["peer_certificate"]!)) is not { } leaf)
        {
            return (false, "the leaf's certificate cannot be read");
        }

        var at = (string?)limbo["validation_time"] is { } time ? DateTimeOffset.Parse(time, CultureInfo.InvariantCulture) : DateTimeOffset.UtcNow;
        var options = given ?? new SignatureCheckOptions { Online = false };
        if ((int?)limbo["max_chain_depth"] is { } depth)
        {
            options = options with { MaxChainLength = Math.Min(depth + 2, 64) };
        }

        var (path, detail) = CmsPath.Build(leaf, Read(limbo["untrusted_intermediates"]), Read(limbo["trusted_certs"]), at, CmsUsage.Signer, options);
        if (path is null)
        {
            return (false, detail);
        }

        foreach (var pem in limbo["crls"] as JsonArray ?? new JsonArray())
        {
            CmsCrl crl;
            try
            {
                crl = CmsCrl.Parse(PemBytes((string)pem!));
            }
            catch (InvalidDataException e)
            {
                return (false, $"a list cannot be read: {e.Message}");
            }

            for (int i = 0; i + 1 < path.Count; i++)
            {
                var answer = crl.StatusOf(path[i], path[i + 1], at);
                if (answer.Status == CmsStatus.Revoked || (answer.Status == CmsStatus.Unknown && !answer.Detail.StartsWith("another issuer", StringComparison.Ordinal)))
                {
                    return (false, answer.Detail);
                }
            }
        }

        return (true, detail);
    }

    private static void PdfKnownBadLimbo(Device device)
    {
        CpuOnly(device);
        using var report = new KnownBadReport("x509-limbo");
        JsonObject data;
        using (var stream = new GZipStream(File.OpenRead(TestData("x509-limbo/limbo-paths.json.gz")), CompressionMode.Decompress))
        {
            data = JsonNode.Parse(stream)!.AsObject();
        }

        var failures = new List<string>();
        int matched = 0, limits = 0, byDesign = 0;
        foreach (var limboNode in data["cases"]!.AsArray())
        {
            var limbo = limboNode!;
            string id = (string)limbo["id"]!, expected = (string)limbo["expected_result"]!;
            var (passed, detail) = LimboRun(limbo);
            string answer = passed ? "valid" : $"refused: {detail}";
            if (passed == (expected == "SUCCESS"))
            {
                matched++;
                report.Line($"{id}: {answer} (limbo: {expected})");
            }
            else if (LimboRule(LimboByDesign, id) is { } why)
            {
                byDesign++;
                report.Line($"not applied: x509-limbo {id} (limbo: {expected}; Idrak: {answer}): {why}");
            }
            else if (LimboRule(LimboLimits, id) is { } limit)
            {
                limits++;
                report.Line($"LIMIT: x509-limbo {id}: {limit} (limbo: {expected}; Idrak: {answer})");
            }
            else
            {
                failures.Add($"{id}: limbo {expected}, Idrak {answer}");
            }
        }

        // The name-constraint bound is the cause of the pathological cases' refusal: raised, the first (technically valid) passes.
        var dos = data["cases"]!.AsArray().Single(c => (string)c!["id"]! == "pathological::nc-dos-1")!;
        var (raised, raisedWhy) = LimboRun(dos, new SignatureCheckOptions { Online = false, MaxNameConstraintChecks = 1 << 30 });
        report.Line($"pathological::nc-dos-1 with MaxNameConstraintChecks = {1 << 30}: {(raised ? "valid" : $"refused: {raisedWhy}")}");
        Check(raised, $"pathological::nc-dos-1 passes once the name comparisons are allowed: {raisedWhy}");

        foreach (var (name, count) in data["otherNamespaces"]!.AsObject())
        {
            report.Line($"not applied: x509-limbo's {name} namespace ({(int)count!} cases): a server's or a mailbox's names checked against what it answers for; a document check names its signer by the subject");
        }

        report.Line($"x509-limbo ({(string)data["commit"]!}): {data["cases"]!.AsArray().Count} cases: {matched} as limbo says, {byDesign} not applied by design, {limits} limits, {failures.Count} differ");
        Check(matched >= 100, $"{matched} x509-limbo cases as limbo says (100 or more expected)");
        Check(failures.Count == 0, $"{failures.Count} x509-limbo cases differ: {string.Join("; ", failures.Take(20))}");
    }

    // Certificates that issued each other many times over (six keys under one name, each certifying each: 36 certificates, no
    // key identifiers to tell them apart): a search for a path to an unrelated root ends within its bound, quickly, naming it;
    // trusting the mesh's first self-signed certificate finds the path at once.
    private static void PdfKnownBadSearchBound(Device device)
    {
        CpuOnly(device);
        using var report = new KnownBadReport("search-bound");
        var name = new X500DistinguishedName("CN=Example Mesh CA, O=Example Organisation");
        var start = new DateTimeOffset(2025, 1, 1, 0, 0, 0, TimeSpan.Zero);
        var end = new DateTimeOffset(2040, 1, 1, 0, 0, 0, TimeSpan.Zero);
        var keys = Enumerable.Range(0, 6).Select(_ => ECDsa.Create(ECCurve.NamedCurves.nistP256)).ToList();
        try
        {
            var mesh = new List<CmsCertificate>();
            int serial = 1;
            foreach (var issuer in keys)
            {
                foreach (var subject in keys)
                {
                    var request = new CertificateRequest(name, subject, HashAlgorithmName.SHA256);
                    request.CertificateExtensions.Add(new X509BasicConstraintsExtension(true, false, 0, true));
                    request.CertificateExtensions.Add(new X509KeyUsageExtension(X509KeyUsageFlags.KeyCertSign | X509KeyUsageFlags.CrlSign, true));
                    using var made = request.Create(name, X509SignatureGenerator.CreateForECDsa(issuer), start, end, [(byte)serial++]);
                    mesh.Add(CmsCertificate.Parse(made.RawData));
                }
            }

            using var leafKey = ECDsa.Create(ECCurve.NamedCurves.nistP256);
            var leafRequest = new CertificateRequest("CN=Example Mesh Signer, O=Example Organisation", leafKey, HashAlgorithmName.SHA256);
            leafRequest.CertificateExtensions.Add(new X509KeyUsageExtension(X509KeyUsageFlags.DigitalSignature, true));
            using var leafMade = leafRequest.Create(name, X509SignatureGenerator.CreateForECDsa(keys[0]), start, end, [0x7F]);
            var leaf = CmsCertificate.Parse(leafMade.RawData);
            using var rootKey = ECDsa.Create(ECCurve.NamedCurves.nistP256);
            var rootRequest = new CertificateRequest("CN=Example Unrelated Root CA, O=Example Organisation", rootKey, HashAlgorithmName.SHA256);
            rootRequest.CertificateExtensions.Add(new X509BasicConstraintsExtension(true, false, 0, true));
            rootRequest.CertificateExtensions.Add(new X509KeyUsageExtension(X509KeyUsageFlags.KeyCertSign, true));
            using var rootMade = rootRequest.CreateSelfSigned(start, end);
            var unrelated = CmsCertificate.Parse(rootMade.RawData);
            var at = new DateTimeOffset(2026, 3, 9, 12, 0, 0, TimeSpan.Zero);
            var options = new SignatureCheckOptions { Online = false };

            var clock = Stopwatch.StartNew();
            var (path, detail) = CmsPath.Build(leaf, mesh, [unrelated], at, CmsUsage.Signer, options);
            double ms = clock.Elapsed.TotalMilliseconds;
            report.Line(string.Create(CultureInfo.InvariantCulture, $"a mesh of {mesh.Count} certificates, no trusted path: {detail} ({ms:F0} ms)"));
            Check(path is null && detail.Contains("the path search stopped after 768 issuers tried", StringComparison.Ordinal), $"the mesh's search: {detail}");
            Check(ms < 20_000, string.Create(CultureInfo.InvariantCulture, $"the mesh's search took {ms:F0} ms (well under 20 s expected)"));

            var trusted = mesh[0];                                             // keys[0] certifying itself
            var (found, why) = CmsPath.Build(leaf, mesh, [trusted], at, CmsUsage.Signer, options);
            report.Line($"the mesh with its first self-signed certificate trusted: {why}");
            Check(found is { Count: 2 } && found[1].Same(trusted), $"the trusted mesh certificate found: {why}");
        }
        finally
        {
            keys.ForEach(k => k.Dispose());
        }
    }
}
