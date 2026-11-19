// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Data.Pdf;

// Plan 28, phase D2: PDF encryption (the standard security handler, revisions 2-6), checked on the samples
// tools/documents/make_fixtures.py writes (d2/, and d11/encrypted-aes256) against their JSON and their unencrypted twin;
// and its edges: damaged encrypted files whose catalog lies in an object stream, and the owner password of a file whose
// user password is empty.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] DocumentPdfEncryptionGroup =
    [
        ("documents pdf: encryption: every encrypted sample (revisions 2-6: RC4 of 40 and 128 bits, AES-128 through crypt filters, /StrF /Identity, /EncryptMetadata false, cross-reference and object streams, AES-256; damaged ones repaired) opens with its JSON's password, its user and its owner password (a non-ASCII one too; the owner password as the owner, also where the user password is empty), says it is encrypted and how it was opened, and reads its cross-reference, every object, every stream, its metadata and its information as its JSON has them and as its unencrypted twin holds them (the same objects but each stream's stored /Length and revision 5's /Extensions, the same pages and text), its objects read alike from several threads", PdfEncryptedSamples),
        ("documents pdf: encryption: a wrong password, or none where the user password is not empty, fails with an InvalidDataException naming DocumentOptions.Password; a wrong one opens a file whose user password is empty as its user, every page read naming it (wrong-password); the public-key handler is refused naming Adobe.PubSec; the document format suite passes on the encrypted samples, and damaged encrypted files fail only as damaged or as a variant not read", PdfEncryptionRefusals),
        ("documents pdf: encryption: RC4 on published vectors, passwords prepared (PDFDocEncoding; SASLprep: characters mapped to nothing, other spaces, NFKC; UTF-8 cut at 127 bytes), a revision, algorithm, crypt filter method or key length the handler does not read refused naming it, the stream limit holding on encrypted streams, and the file key cleared when the document is disposed", PdfEncryptionPieces),
        ("documents pdf: encryption: edges: a damaged encrypted file whose catalog lies in an object stream (AES-128 cut in its cross-reference stream, AES-256 with every offset short, an encryption dictionary away from where its cross-reference says) opens repaired, its key found from the trailer the scan finds before the object streams are listed, and reads its twin's text, refused without its password or with a wrong one naming the option; a file whose user password is empty opens as its owner by the owner password (DocumentInfo.Security.OwnerAccess: a rewrite changes its permissions and removes its encryption), as its user without one and by a wrong one; a password that is both is the owner's", PdfEncryptionEdges),
    ];

    // The encrypted samples (their JSON's open.encrypted), and the one a public-key handler encrypts.
    private static List<PdfSampleEntry> PdfEncryptedSamplesOf(List<PdfSampleEntry> samples) =>
        [.. samples.Where(s => s.Expected["open"]?["encrypted"]?.GetValue<bool>() == true)];

    private static void PdfEncryptedSamples(Device device)
    {
        CpuOnly(device);
        var samples = PdfSamples();
        var encrypted = PdfEncryptedSamplesOf(samples);
        Check(encrypted.Count >= 26, $"{encrypted.Count} encrypted samples (the generator writes 25 for D2 and one for D11)");
        int opens = 0, objects = 0, streams = 0, twins = 0;
        foreach (var sample in encrypted)
        {
            var expected = sample.Expected;
            var encryption = expected["encryption"]!;
            string user = (string)encryption["userPassword"]!, owner = (string)encryption["ownerPassword"]!;
            var twin = (string?)expected["twin"] is { } twinName ? samples.Single(s => s.Name == twinName) : null;
            bool first = true;
            foreach (string password in new[] { (string)expected["open"]!["password"]!, user, owner }.Distinct())
            {
                string what = $"{sample.Name} (password \"{password}\")";
                using var source = File.OpenRead(sample.Path);
                using var document = (PdfDocument)new PdfFormat().Open(source, sample.Name, new DocumentOptions { Password = password.Length == 0 ? null : password });
                var file = document.File;
                var info = document.Info;
                var open = expected["open"]!;
                bool repaired = open["repaired"]?.GetValue<bool>() == true;
                Check(info.Encrypted && info.Repaired == repaired && document.PageCount == (int)open["pageCount"]! && info.Version == (string?)open["version"],
                    $"{what}: encrypted {info.Encrypted}, repaired {info.Repaired}, {document.PageCount} pages, version {info.Version}; expected {open.ToJsonString()}");
                var security = file.Security!;
                bool asOwner = owner.Length > 0 && password == owner;   // tried as the owner's first: also where the user password is empty
                Check(security.Revision == (int)encryption["R"]! && security.Owner == asOwner && info.Security?.OwnerAccess == asOwner && !security.PasswordRefused,
                    $"{what}: revision {security.Revision}, opened as {(security.Owner ? "owner" : "user")} (the information says owner access {info.Security?.OwnerAccess}); expected {encryption["R"]}, as {(asOwner ? "owner" : "user")}");

                // The information says how it is encrypted, as its JSON states it: nothing random (no key, salt or IV).
                string cipher = (string)encryption["cipher"]!;
                string strings = (string?)encryption["stringFilter"] == "Identity" ? "none" : cipher;
                Check(info.Security is { Handler: "Standard" } described && described.Revision == (int)encryption["R"]! && described.KeyBits == (int)encryption["lengthBits"]!
                      && described.Streams == cipher && described.Strings == strings && described.EmbeddedFiles == cipher
                      && described.Permissions == PdfSecurity.ToPermissions((int)encryption["permissions"]!) && described.MetadataEncrypted == (bool)encryption["encryptMetadata"]!,
                    $"{what}: the information's encryption {info.Security}; expected revision {encryption["R"]}, {encryption["lengthBits"]} bits, {cipher} (strings {strings}), /P {encryption["permissions"]}, metadata encrypted {encryption["encryptMetadata"]}");
                if (expected["info"] is JsonObject stated)
                {
                    PdfCheckInfo(what, info, stated);
                }

                foreach (var page in expected["pages"]!.AsArray())
                {
                    int index = (int)page!["index"]!;
                    var read = document.ReadPage(index, DocumentParts.All);
                    Check(MathF.Abs(read.Width - (float)page["width"]!) < 1e-3f && MathF.Abs(read.Height - (float)page["height"]!) < 1e-3f && read.Issues.Count == 0,
                        $"{what}: page {index} is {read.Width} x {read.Height} with {read.Issues.Count} issues ({string.Join("; ", read.Issues.Select(i => i.Detail))})");
                }

                PdfCheckStructure(what, file, expected);
                if (expected["objects"] is JsonArray listed)
                {
                    objects += PdfCheckObjects(what, file, listed);
                }

                foreach (var stream in expected["streams"]!.AsArray())
                {
                    int number = (int)stream!["object"]![0]!;
                    var read = file.Get(number).AsStream;
                    Check(read is not null && read.Length == (long)stream["storedLength"]! && !read.LengthRecovered, $"{what}: object {number} stores {read?.Length} bytes, expected {stream["storedLength"]}");
                    using var decoded = file.Decode(read!);
                    string hash = Convert.ToHexStringLower(SHA256.HashData(decoded.Span));
                    Check(decoded.Length == (long)stream["decodedLength"]! && hash == (string?)stream["decodedSha256"] && !decoded.Damaged,
                        $"{what}: object {number} ({stream["type"]}, {stream["filters"]!.ToJsonString()}) decodes to {decoded.Length} bytes ({hash}), expected {stream["decodedLength"]} ({stream["decodedSha256"]})");
                    streams++;
                }

                if (expected["metadataSha256"] is { } metadataHash)
                {
                    var catalog = file.Dictionary(file.Trailer["Root"u8])!;
                    using var metadata = file.Decode(file.Resolve(catalog["Metadata"u8]).AsStream!);
                    string xml = Encoding.UTF8.GetString(metadata.Span);
                    Check(Convert.ToHexStringLower(SHA256.HashData(metadata.Span)) == (string?)metadataHash,
                        $"{what}: the metadata (/EncryptMetadata {encryption["encryptMetadata"]}) reads as \"{xml[..Math.Min(40, xml.Length)]}...\"");
                }

                if (first && twin is not null)
                {
                    PdfCheckTwin(what, document, twin);
                    twins++;
                }

                if (first)
                {
                    // Every live object read from several threads at once, by a document of its own (nothing cached yet).
                    using var fresh = (PdfDocument)new PdfFormat().Open(new MemoryStream(File.ReadAllBytes(sample.Path), writable: false), sample.Name,
                        new DocumentOptions { Password = password });
                    int[] numbers = [.. fresh.File.LiveObjects().Select(o => o.Number)];
                    var parallel = new string[numbers.Length];
                    Parallel.For(0, numbers.Length, i => parallel[i] = PdfCanonical(fresh.File, fresh.File.Get(numbers[i]))?.ToJsonString() ?? "null");
                    for (int i = 0; i < numbers.Length; i++)
                    {
                        string once = PdfCanonical(file, file.Get(numbers[i]))?.ToJsonString() ?? "null";
                        Check(parallel[i] == once, $"{what}: object {numbers[i]} read from several threads is {parallel[i]}, one by one {once}");
                    }
                }

                first = false;
                opens++;
            }
        }

        Check(opens >= 44 && objects >= 420 && streams >= 130 && twins >= 21, $"{opens} opens, {objects} objects, {streams} streams and {twins} twins checked");
    }

    // Each listed object (the newest revision's, decrypted) in the canonical form; a live object the list leaves out is
    // an object stream or a cross-reference stream. Returns how many were checked.
    private static int PdfCheckObjects(string what, PdfFile file, JsonArray listed)
    {
        var live = file.LiveObjects().ToDictionary(o => o.Number, o => o.Generation);
        var numbers = new HashSet<int>();
        foreach (var entry in listed)
        {
            int number = (int)entry!["num"]!;
            numbers.Add(number);
            Check(live.TryGetValue(number, out int generation) && generation == (int)entry["gen"]!, $"{what}: object {number} {entry["gen"]} is not live");
            if (PdfDifference(entry["value"], PdfCanonical(file, file.Get(number)), file, $"object {number}") is { } difference)
            {
                Check(false, $"{what}: {difference}");
            }
        }

        foreach (int number in live.Keys.Where(n => !numbers.Contains(n)))
        {
            var type = file.Get(number).AsDictionary?["Type"u8];
            Check(type is { } t && (t.IsName("ObjStm"u8) || t.IsName("XRef"u8)), $"{what}: object {number} is live but not in the sample's objects ({file.Get(number)})");
        }

        return listed.Count;
    }

    // The unencrypted twin's objects as this file holds them (a stream's stored /Length aside: encryption changes it; and
    // the catalog's /Extensions, where revision 5 declares itself), its information, and the same pages and text.
    private static void PdfCheckTwin(string what, PdfDocument document, PdfSampleEntry twin)
    {
        var file = document.File;
        foreach (var entry in twin.Expected["objects"]!.AsArray())
        {
            int number = (int)entry!["num"]!;
            var actual = PdfCanonical(file, file.Get(number));
            if (PdfDifference(PdfWithoutEncryption(entry["value"]?.DeepClone()), PdfWithoutEncryption(actual), file, $"object {number}") is { } difference)
            {
                Check(false, $"{what}: as its twin {twin.Name} holds it, {difference}");
            }
        }

        PdfCheckInfo($"{what}, as its twin", document.Info, twin.Expected["info"]!.AsObject());
        using var plain = DocumentFormats.Open(twin.Path);
        Check(plain.PageCount == document.PageCount && !plain.Info.Encrypted && plain.Info.Security is null, $"{what}: {document.PageCount} pages, its twin {plain.PageCount}, its twin's encryption {plain.Info.Security}");
        for (int i = 0; i < plain.PageCount; i++)
        {
            var mine = document.ReadPage(i, DocumentParts.Text);
            var theirs = plain.ReadPage(i, DocumentParts.Text);
            Check(mine.Text == theirs.Text && mine.Runs.Select(r => r.Text).SequenceEqual(theirs.Runs.Select(r => r.Text)) && mine.Width == theirs.Width && mine.Height == theirs.Height,
                $"{what}: page {i} reads \"{mine.Text}\" in {mine.Runs.Count} runs, its twin \"{theirs.Text}\" in {theirs.Runs.Count}");
        }
    }

    // An object's canonical form without what encryption changes: a stream's /Length, the catalog's /Extensions.
    private static JsonNode? PdfWithoutEncryption(JsonNode? value)
    {
        if (value is JsonObject { } stream && stream["stream"] is JsonObject dictionary)
        {
            dictionary.Remove("Length");
        }

        if (value is JsonObject { } holder && holder["dict"] is JsonObject catalog && (string?)catalog["Type"] == "/Catalog")
        {
            catalog.Remove("Extensions");
        }

        return value;
    }

    private static void PdfEncryptionRefusals(Device device)
    {
        CpuOnly(device);
        var samples = PdfSamples();
        var encrypted = PdfEncryptedSamplesOf(samples);
        foreach (var sample in encrypted)
        {
            var encryption = sample.Expected["encryption"]!;
            string user = (string)encryption["userPassword"]!, wrong = (string)encryption["wrongPassword"]!;
            if (user.Length == 0)
            {
                // Neither password: the empty user password opens it, as its user, and every page read says so.
                using var document = DocumentFormats.Open(sample.Path, new DocumentOptions { Password = wrong });
                Check(document.Info is { Encrypted: true, Security.OwnerAccess: false }, $"{sample.Name}: a wrong password given, the empty user password opens it as its user");
                for (int i = 0; i < document.PageCount; i++)
                {
                    var issues = document.ReadPage(i, DocumentParts.Text).Issues;
                    Check(issues.Count == 1 && issues[0].Kind == DocumentIssue.WrongPassword && issues[0].Detail.Contains("DocumentOptions.Password", StringComparison.Ordinal)
                          && issues[0].Detail.Contains($"page {i + 1}", StringComparison.Ordinal),
                        $"{sample.Name}, a wrong password: page {i + 1}'s issues {string.Join("; ", issues.Select(x => $"{x.Kind}: {x.Detail}"))}");
                }

                continue;
            }

            foreach (string? password in new[] { wrong, null, "" })
            {
                try
                {
                    using var document = DocumentFormats.Open(sample.Path, new DocumentOptions { Password = password });
                    Check(false, $"{sample.Name}: opened with the password \"{password}\"");
                }
                catch (InvalidDataException e)
                {
                    bool none = string.IsNullOrEmpty(password);
                    Check(e.Message.Contains("DocumentOptions.Password", StringComparison.Ordinal) && e.Message.Contains(Path.GetFileName(sample.Path), StringComparison.Ordinal)
                          && e.Message.Contains(none ? "opens only with a password" : "neither the user nor the owner password", StringComparison.Ordinal),
                        $"{sample.Name}, the password \"{password}\": \"{e.Message}\"");
                }
            }
        }

        var unsupported = samples.Single(s => s.Kind == "unsupported");
        string handler = (string)unsupported.Expected["openError"]!["handler"]!;
        foreach (string? password in new[] { null, "user-pass-1" })
        {
            try
            {
                using var document = DocumentFormats.Open(unsupported.Path, new DocumentOptions { Password = password });
                Check(false, $"{unsupported.Name}: a file encrypted by {handler} opened");
            }
            catch (NotSupportedException e)
            {
                Check(e.Message.Contains(handler, StringComparison.Ordinal) && e.Message.Contains("encrypted", StringComparison.Ordinal), $"{unsupported.Name}: \"{e.Message}\" does not name {handler}");
            }
        }

        // The format suite with the D2 user password: the files it (or the empty one) opens are read whole, pages in
        // parallel, with tight limits; the others are refused naming the option, the public-key one naming its handler.
        const string Password = "user-pass-1";
        var suite = new List<DocumentSample>();
        foreach (var sample in encrypted.Append(unsupported).Append(samples.Single(s => s.Kind == "twin")))
        {
            string? user = (string?)sample.Expected["encryption"]?["userPassword"];
            suite.Add(sample.Kind == "unsupported" ? new DocumentSample(sample.Path, Error: handler)
                : user is null or "" or Password ? new DocumentSample(sample.Path, Pages: (int)sample.Expected["open"]!["pageCount"]!)
                : new DocumentSample(sample.Path, Error: "DocumentOptions.Password"));
        }

        Conformance.CheckDocumentFormat(DocumentFormats.Get(PdfFormat.FormatName), suite, documentOptions: new DocumentOptions { Password = Password },
            options: new ContractCheckOptions { RandomCases = 0 }).ThrowIfFailed();

        // Damaged encrypted files (cut, or a few bytes changed): read, or failing as damaged (InvalidDataException) or as a
        // variant not read when a changed byte names one (a NotSupportedException saying the file is encrypted).
        int damaged = 0;
        foreach (var sample in encrypted)
        {
            byte[] bytes = File.ReadAllBytes(sample.Path);
            var random = new Random(sample.Name.Length * 7919 + bytes.Length);
            var options = new DocumentOptions { Password = (string)sample.Expected["open"]!["password"]! };
            for (int round = 0; round < 6; round++)
            {
                byte[] changed = round % 2 == 0 ? bytes[..random.Next(bytes.Length)] : (byte[])bytes.Clone();
                for (int k = 0, n = round % 2 == 0 ? 0 : 1 + random.Next(8); k < n; k++)
                {
                    changed[random.Next(changed.Length)] = (byte)random.Next(256);
                }

                try
                {
                    using var document = DocumentFormats.Open(changed.AsMemory(), sample.Name + ".pdf", options);
                    _ = document.Info;
                    for (int i = 0; i < document.PageCount; i++)
                    {
                        _ = document.ReadPage(i, DocumentParts.All);
                    }
                }
                catch (InvalidDataException)
                {
                    // Damaged: as the contract says.
                }
                catch (NotSupportedException e) when (e.Message.Contains("encrypted", StringComparison.Ordinal))
                {
                    // A changed byte that names another handler, revision or algorithm.
                }
                catch (Exception e)
                {
                    Check(false, $"{sample.Name}, damaged case {round} ({changed.Length} bytes): {e.GetType().Name}: {e.Message}");
                }

                damaged++;
            }
        }

        Check(damaged >= 120, $"{damaged} damaged cases");
    }

    private static void PdfEncryptionPieces(Device device)
    {
        CpuOnly(device);

        // RC4 on the published vectors, and in place.
        foreach (var (key, plain, cipher) in new[] { ("Key", "Plaintext", "BBF316E8D940AF0AD3"), ("Wiki", "pedia", "1021BF0420"), ("Secret", "Attack at dawn", "45A01F645FC35B383552544B9BF5") })
        {
            byte[] data = Encoding.ASCII.GetBytes(plain), output = new byte[data.Length];
            PdfCrypt.Rc4(Encoding.ASCII.GetBytes(key), data, output);
            Check(Convert.ToHexString(output) == cipher, $"RC4 of \"{plain}\" under \"{key}\" is {Convert.ToHexString(output)}, expected {cipher}");
            PdfCrypt.Rc4(Encoding.ASCII.GetBytes(key), output, output);
            Check(output.AsSpan().SequenceEqual(data), $"RC4 in place does not give \"{plain}\" back");
        }

        // Passwords as each revision takes them.
        Span<byte> padded = stackalloc byte[32];
        PdfCrypt.Pad("abc"u8, padded);
        Check(padded[..3].SequenceEqual("abc"u8) && padded[3..].SequenceEqual(PdfCrypt.Padding[..29]), "a password padded to 32 bytes");
        Span<byte> bytes = stackalloc byte[PdfCrypt.PasswordBytes];
        int legacy = PdfCrypt.LegacyPassword("\u20ACuro \u00FC\u2022", bytes);
        Check(legacy == 7 && bytes[..legacy].SequenceEqual((ReadOnlySpan<byte>)[0xA0, (byte)'u', (byte)'r', (byte)'o', (byte)' ', 0xFC, 0x80]) && PdfCrypt.LegacyPassword("\u65E5\u672C", bytes) == -1
              && PdfCrypt.LegacyPassword(new string('a', 40), bytes) == 32, "PDFDocEncoding passwords: the euro, a bullet, Latin-1, none for characters outside it, at most 32 bytes");
        Span<char> prepared = stackalloc char[64];
        foreach (var (given, expected) in new[] { ("pass\u00ADword", "password"), ("a\u00A0b\u2003c", "a b c"), ("\uFB01le\u200B", "file"), ("Gr\u00FC\u00DFe-2024", "Gr\u00FC\u00DFe-2024"), ("e\u0301t\u00E9", "\u00E9t\u00E9") })
        {
            int n = PdfCrypt.SaslPrep(given, prepared);
            Check(n >= 0 && prepared[..n].SequenceEqual(expected), $"SASLprep of \"{given}\" is \"{(n < 0 ? "(none)" : prepared[..n].ToString())}\", expected \"{expected}\"");
        }

        Check(PdfCrypt.SaslPrep("a\uD800b", prepared) == -1, "a lone surrogate is not prepared (the password is tried as it is)");
        Check(PdfCrypt.Utf8Password(new string('\u00E9', 100), bytes, PdfCrypt.PasswordBytes) == 127, "a UTF-8 password cut at 127 bytes");

        // Variants the handler does not read, and a key length it does not allow: the same bytes but one entry.
        var samples = PdfSamples();
        byte[] Changed(string name, string from, string to)
        {
            byte[] file = File.ReadAllBytes(samples.Single(s => s.Name == name).Path);
            int at = file.AsSpan().IndexOf(Encoding.ASCII.GetBytes(from));
            Check(at >= 0 && from.Length == to.Length, $"{name}: no \"{from}\" to change");
            Encoding.ASCII.GetBytes(to).CopyTo(file, at);
            return file;
        }

        foreach (var (sample, from, to, error, says) in new[]
                 {
                     ("r2-rc4-40-empty", "/V 1/R 2", "/V 3/R 2", typeof(NotSupportedException), "/V 3"),
                     ("r2-rc4-40-empty", "/V 1/R 2", "/V 1/R 7", typeof(NotSupportedException), "revision 7"),
                     ("r3-rc4-128-empty", "/R 3/Length 128", "/R 3/Length 129", typeof(InvalidDataException), "129 bits"),
                     ("r4-aes-128-empty", "/CFM /AESV2", "/CFM /AESV9", typeof(NotSupportedException), "AESV9"),
                     ("r6-aes-256-empty", "/Filter /Standard", "/Filter /Standarx", typeof(NotSupportedException), "Standarx"),
                 })
        {
            try
            {
                using var document = DocumentFormats.Open(Changed(sample, from, to).AsMemory(), sample + ".pdf");
                Check(false, $"{sample} with {to}: opened");
            }
            catch (Exception e) when (e is InvalidDataException or NotSupportedException)
            {
                Check(e.GetType() == error && e.Message.Contains(says, StringComparison.Ordinal) && e.Message.Contains("encrypted", StringComparison.Ordinal),
                    $"{sample} with {to}: {e.GetType().Name}: \"{e.Message}\" (expected a {error.Name} saying \"{says}\")");
            }
        }

        // The stream limit holds on encrypted streams (their stored bytes, which decrypt to no more).
        var aes = samples.Single(s => s.Name == "r4-aes-128-empty");
        using (var document = DocumentFormats.Open(aes.Path, new DocumentOptions { MaxStreamBytes = 64 }))
        {
            try
            {
                document.ReadPage(0, DocumentParts.Text);
                Check(false, "a page's encrypted content of more than 64 bytes read under a limit of 64");
            }
            catch (InvalidDataException e)
            {
                Check(e.Message.Contains("MaxStreamBytes", StringComparison.Ordinal), e.Message);
            }
        }

        // The file key is held while the document is open, and cleared when it is disposed.
        var r6 = samples.Single(s => s.Name == "r6-aes-256-user");
        var opened = (PdfDocument)new PdfFormat().Open(new MemoryStream(File.ReadAllBytes(r6.Path), writable: false), r6.Name, new DocumentOptions { Password = "user-pass-1" });
        var security = opened.File.Security!;
        Check(security is { Cleared: false, Revision: 6, Owner: false, Strings: PdfCryptMethod.Aes256, Streams: PdfCryptMethod.Aes256 }, "the r6 sample opened by its user password, its key held while open");
        opened.Dispose();
        Check(security.Cleared, "the file key is cleared when the document is disposed");
    }

    private static void PdfEncryptionEdges(Device device)
    {
        CpuOnly(device);
        var samples = PdfSamples();
        var encrypted = PdfEncryptedSamplesOf(samples);
        List<string> twin;
        using (var plain = DocumentFormats.Open(samples.Single(s => s.Name == "plain").Path))
        {
            twin = [.. Enumerable.Range(0, plain.PageCount).Select(i => plain.ReadPage(i, DocumentParts.Text).Text)];
        }

        // Damaged files whose catalog lies in an object stream: the cross-reference rebuilt by scanning, the trailer's
        // /Encrypt and /ID found in the cross-reference stream's dictionary, the key found before the object streams are listed.
        var damaged = encrypted.Where(s => s.Expected["open"]!["repaired"]?.GetValue<bool>() == true).ToList();
        Check(damaged.Count >= 2 && damaged.Any(s => (int)s.Expected["encryption"]!["R"]! == 4) && damaged.Any(s => (int)s.Expected["encryption"]!["R"]! == 6),
            $"{damaged.Count} damaged encrypted samples ({string.Join(", ", damaged.Select(s => s.Name))}), an AES-128 and an AES-256 one");
        foreach (var sample in damaged)
        {
            PdfEdgeRepaired(sample.Name, File.ReadAllBytes(sample.Path), sample.Expected, twin);
        }

        // An encryption dictionary away from where the cross-reference (read as written) says: reading it repairs the file
        // while its key is looked for, and the repair finds the key before it lists the object streams.
        foreach (string name in new[] { "r6-aes-256-object-streams-user", "r6-aes-256-object-streams-empty" })
        {
            var sample = samples.Single(s => s.Name == name);
            byte[] moved = PdfEdgeMoveEncryption(File.ReadAllBytes(sample.Path), (int)sample.Expected["encryption"]!["encryptDictionary"]!, name);
            PdfEdgeRepaired($"{name}, its encryption dictionary moved", moved, sample.Expected, twin);
        }

        // A file whose user password is empty: the owner password opens it as its owner (tried first), who may change its
        // permissions and remove its encryption; no password, or the empty one, opens it as its user, who may not.
        var ownerOnly = encrypted.Where(s => (string)s.Expected["encryption"]!["userPassword"]! == "" && ((string?)s.Expected["encryption"]!["ownerPassword"])?.Length > 0).ToList();
        Check(ownerOnly.Count >= 13 && ownerOnly.Count(s => s.Name.Contains("print-copy", StringComparison.Ordinal)) == 2,
            $"{ownerOnly.Count} samples whose user password is empty and owner password is not (the print-copy ones among them)");
        foreach (var sample in ownerOnly)
        {
            string owner = (string)sample.Expected["encryption"]!["ownerPassword"]!;
            int bits = (int)sample.Expected["encryption"]!["permissions"]!;
            byte[] bytes = File.ReadAllBytes(sample.Path);
            using (var asUser = PdfEditOpen(bytes, sample.Name))
            {
                Check(asUser.File.Security is { Owner: false, PasswordRefused: false } && asUser.Info.Security is { OwnerAccess: false },
                    $"{sample.Name}: no password opens it as its user");
                ExpectRefusal<InvalidOperationException>(() => PdfEditSave(asUser, e => e.Encrypt(null), DocumentSaveMode.Rewrite), "owner password", $"{sample.Name}: its encryption removed by its user");
            }

            using var asOwner = PdfEditOpen(bytes, sample.Name, owner);
            var read = PdfEditPagesRead(asOwner);
            Check(asOwner.File.Security is { Owner: true, PasswordRefused: false } && asOwner.Info.Security is { OwnerAccess: true } security
                  && security.Permissions == PdfSecurity.ToPermissions(bits) && Enumerable.Range(0, asOwner.PageCount).All(i => asOwner.ReadPage(i, DocumentParts.Text).Issues.Count == 0),
                $"{sample.Name}: the owner password \"{owner}\" opens it as its owner: {asOwner.Info.Security}");

            byte[] decrypted = PdfEditSave(asOwner, e => e.Encrypt(null), DocumentSaveMode.Rewrite);
            using (var opened = PdfEditOpen(decrypted, $"{sample.Name} decrypted"))
            {
                Check(!opened.Info.Encrypted && opened.Info.Security is null && opened.File.Trailer["Encrypt"u8].IsNull && PdfEditPagesRead(opened).SequenceEqual(read),
                    $"{sample.Name}: its encryption removed by its owner, it reads as before: {PdfEditDescribe(PdfEditPagesRead(opened))}");
            }

            // New permissions (printing only), the user password still empty: the user may print, the owner opens it as the owner.
            const string NewOwner = "new-owner-2026";
            byte[] printOnly = PdfEditSave(asOwner, e => e.Encrypt(new DocumentEncryption { OwnerPassword = NewOwner, Permissions = DocumentPermissions.Print }), DocumentSaveMode.Rewrite);
            using (var user = PdfEditOpen(printOnly, $"{sample.Name} print only"))
            {
                Check(user.Info.Security is { OwnerAccess: false, Permissions: DocumentPermissions.Print, Revision: 6 } && PdfEditPagesRead(user).SequenceEqual(read),
                    $"{sample.Name}: its permissions changed by its owner: {user.Info.Security}");
            }

            using (var owned = PdfEditOpen(printOnly, $"{sample.Name} print only", NewOwner))
            {
                Check(owned.Info.Security is { OwnerAccess: true, Permissions: DocumentPermissions.Print }, $"{sample.Name}, new permissions: the new owner password opens it as the owner: {owned.Info.Security}");
            }
        }

        // A password that is both the user's and the owner's is the owner's (Algorithm 2.A tries the owner's first).
        using var destination = new MemoryStream();
        using (var builder = DocumentWriters.Create(destination, "pdf", PdfWriteDefaults))
        {
            builder.Encrypt(new DocumentEncryption { UserPassword = "same-2026", OwnerPassword = "same-2026", Permissions = DocumentPermissions.None });
            builder.AddPage(200, 100);
            builder.Finish();
        }

        using var both = PdfEditOpen(destination.ToArray(), "same passwords", "same-2026");
        Check(both.File.Security is { Owner: true } && both.Info.Security is { OwnerAccess: true }, "a password that is both the user's and the owner's opens as the owner");
    }

    // A damaged encrypted file read with its user and its owner password: repaired, the catalog in an object stream, opened
    // as the password says, each page reading its twin's text; refused without a password and with a wrong one (the
    // password's error, not a repair's), or, where its user password is empty, opened as its user by a wrong one.
    private static void PdfEdgeRepaired(string what, byte[] bytes, JsonObject expected, List<string> twin)
    {
        var encryption = expected["encryption"]!;
        string user = (string)encryption["userPassword"]!, owner = (string)encryption["ownerPassword"]!, wrong = (string)encryption["wrongPassword"]!;
        foreach (var (password, asOwner) in new[] { (user, false), (owner, true) })
        {
            using var document = PdfEditOpen(bytes, what, password.Length == 0 ? null : password);
            var file = document.File;
            var root = file.Trailer["Root"u8];
            Check(document.Info is { Encrypted: true, Repaired: true } && document.Info.Security?.OwnerAccess == asOwner && file.Security!.Owner == asOwner
                  && root.IsReference && file.Xref[root.AsReference.Number].Kind == PdfEntryKind.Compressed && document.PageCount == twin.Count,
                $"{what} (password \"{password}\"): encrypted {document.Info.Encrypted}, repaired {document.Info.Repaired}, owner access {document.Info.Security?.OwnerAccess}, the catalog {root} {(root.IsReference ? file.Xref[root.AsReference.Number].Kind.ToString() : "")}, {document.PageCount} pages");
            for (int i = 0; i < document.PageCount; i++)
            {
                var page = document.ReadPage(i, DocumentParts.Text);
                Check(page.Text == twin[i] && page.Issues.Count == 0, $"{what} (password \"{password}\"): page {i + 1} reads \"{page.Text}\", its twin \"{twin[i]}\"; {page.Issues.Count} issues");
            }
        }

        foreach (string? password in new[] { null, wrong })
        {
            if (user.Length == 0)
            {
                if (password is not null)
                {
                    using var document = PdfEditOpen(bytes, what, password);
                    Check(document.Info.Repaired && document.File.Security is { Owner: false, PasswordRefused: true }
                          && document.ReadPage(0, DocumentParts.Text).Issues.Any(x => x.Kind == DocumentIssue.WrongPassword),
                        $"{what}: a wrong password opens it as its user, saying so");
                }

                continue;
            }

            try
            {
                using var document = PdfEditOpen(bytes, what, password);
                Check(false, $"{what}: opened with the password \"{password}\"");
            }
            catch (InvalidDataException e)
            {
                Check(e.Message.Contains("DocumentOptions.Password", StringComparison.Ordinal)
                      && e.Message.Contains(password is null ? "opens only with a password" : "neither the user nor the owner password", StringComparison.Ordinal),
                    $"{what}, the password \"{password}\": \"{e.Message}\"");
            }
        }
    }

    // `bytes` with "junk" put before object `number` (the encryption dictionary) and startxref moved past it: the
    // cross-reference stream (after it) is read as written, but says the dictionary is where it no longer is.
    private static byte[] PdfEdgeMoveEncryption(byte[] bytes, int number, string what)
    {
        string text = Encoding.Latin1.GetString(bytes);
        int at = text.IndexOf($"\n{number} 0 obj", StringComparison.Ordinal) + 1;
        int keyword = text.LastIndexOf("startxref", StringComparison.Ordinal);
        int digits = keyword + "startxref".Length;
        while (digits < text.Length && !char.IsAsciiDigit(text[digits]))
        {
            digits++;
        }

        int end = digits;
        while (end < text.Length && char.IsAsciiDigit(text[end]))
        {
            end++;
        }

        long startxref = long.Parse(text[digits..end], CultureInfo.InvariantCulture);
        const string Junk = "junk\n";
        Check(at > 0 && keyword > at && startxref > at, $"{what}: object {number} at {at - 1} before the cross-reference stream at {startxref}");
        return Encoding.Latin1.GetBytes(text[..at] + Junk + text[at..digits] + (startxref + Junk.Length).ToString(CultureInfo.InvariantCulture) + text[end..]);
    }
}
