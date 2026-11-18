// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Data.Pdf;

// Plan 28, phase D1: the PDF file (objects, cross-references, object streams, incremental updates, repair, filters,
// limits, the information and the pages' sizes), checked on the samples tools/documents/make_fixtures.py writes into the
// ignored test-output/documents/samples (IDRAK_DOCUMENTS_SAMPLES moves them), and on hand-made bytes.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] DocumentPdfFileGroup =
    [
        ("documents pdf: file: the library's pdf format is registered (name, extension, media type, the header within the first KiB) and opens every sample that is not encrypted or hostile with its page count, version, information (PDFDocEncoding, UTF-16, UTF-8, escapes, dates whole, partial and odd), each page's size and rotation (inherited, crop boxes, the kids' count winning over /Count) and the repaired flag", PdfSamplesOpen),
        ("documents pdf: file: each D1 sample's cross-reference as written (tables, streams, object streams, hybrids, incremental updates through /Prev with a freed object gone, offsets after junk), every live object in the canonical form, read alike from several threads, and every listed stream of every sample decoded through its filters and predictors to its SHA-256", PdfSamplesObjects),
        ("documents pdf: file: hostile samples fail fast with an InvalidDataException naming their limit (MaxStreamBytes, MaxDepth, MaxObjects) or the cycle (references, the page tree, /Prev, an object stream holding itself), where their JSON says; encrypted samples are refused naming phase D2, the public-key handler naming its handler", PdfHostileAndEncrypted),
        ("documents pdf: file: the document format suite passes on pdf with the samples (pages read in parallel the same, tight limits named, cut and changed files failing only as damaged)", PdfFormatSuite),
        ("documents pdf: file: the lexer on hand-made bytes (names with #xx, literal strings with escapes, octal, nested parentheses, line ends and continuations, hex strings, numbers and references); hand-made files read alike from a file, memory and a plain stream, with no cache, a stream bomb and a deep page tree failing naming their limits", PdfHandMade),
    ];

    private static readonly Lock PdfSamplesGate = new();
    private static (string Folder, List<PdfSampleEntry> Samples)? s_pdfSamples;

    // One sample of the manifest: its file, phases and expected JSON.
    private sealed record PdfSampleEntry(string Name, string Path, string Kind, string[] Phases, bool Hostile, JsonObject Expected)
    {
        public bool D1 => Phases.Contains("D1") || Kind == "twin";

        public bool Encrypted => Expected["open"]?["encrypted"]?.GetValue<bool>() == true || Expected["openError"] is not null;

        public bool Readable => !Hostile && !Encrypted && Expected["open"] is not null;
    }

    // The samples' folder (IDRAK_DOCUMENTS_SAMPLES, else test-output/documents/samples), each sample checked against the
    // manifest's SHA-256; a missing folder fails at once naming the command that makes it.
    private static List<PdfSampleEntry> PdfSamples()
    {
        string folder = Environment.GetEnvironmentVariable("IDRAK_DOCUMENTS_SAMPLES") is { Length: > 0 } set ? set : RepositoryPath("test-output", "documents", "samples");
        lock (PdfSamplesGate)
        {
            if (s_pdfSamples is { } known && known.Folder == folder)
            {
                return known.Samples;
            }

            string manifestPath = Path.Combine(folder, "manifest.json");
            Check(File.Exists(manifestPath),
                $"The PDF samples are not in {folder}. Make them with: python -I tools/documents/make_fixtures.py -o test-output/documents/samples "
                + "(after python -m pip install -r tools/documents/requirements.txt; see tools/documents/README.md), or set IDRAK_DOCUMENTS_SAMPLES to where they are.");
            var manifest = JsonNode.Parse(File.ReadAllText(manifestPath))!.AsObject();
            var samples = new List<PdfSampleEntry>();
            foreach (var entry in manifest["samples"]!.AsArray())
            {
                string path = Path.Combine(folder, (string)entry!["file"]!);
                Check(File.Exists(path), $"{path} is in the manifest but not on disk: make the samples again (tools/documents/make_fixtures.py).");
                string hash = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
                Check(hash == (string?)entry["sha256"], $"{path}'s SHA-256 is not the manifest's: make the samples again (tools/documents/make_fixtures.py).");
                var expected = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, (string)entry["expected"]!)))!.AsObject();
                samples.Add(new PdfSampleEntry((string)entry["name"]!, path, (string)entry["kind"]!, [.. entry["phases"]!.AsArray().Select(p => (string)p!)], (bool)entry["hostile"]!, expected));
            }

            Check(samples.Count(s => s.D1) >= 16 && samples.Any(s => s.Hostile), $"{manifestPath}: {samples.Count} samples, fewer D1 samples than the generator writes");
            s_pdfSamples = (folder, samples);
            return samples;
        }
    }

    private static void PdfSamplesOpen(Device device)
    {
        CpuOnly(device);
        var samples = PdfSamples();
        Check(DocumentFormats.Names.Contains(PdfFormat.FormatName) && DocumentFormats.Origin(PdfFormat.FormatName) == Overrides.Library
              && DocumentFormats.Default(PdfFormat.FormatName) is PdfFormat && DocumentFormats.Extensions.Contains(".pdf") && DocumentFormats.MediaTypes.Contains("application/pdf")
              && DocumentFormats.CanOpen("letter.PDF"), $"the pdf format registered by the library: {string.Join(", ", DocumentFormats.Names)}");
        var format = new PdfFormat();
        byte[] junk = new byte[1100];
        junk.AsSpan().Fill((byte)'x');
        "%PDF-1.7"u8.CopyTo(junk.AsSpan(1023));
        Check(format.Accepts("%PDF-1.4\n"u8) && format.Accepts(junk.AsSpan(0, 1031)) && DocumentFormats.Detect("%PDF-2.0\n"u8) is { Name: PdfFormat.FormatName }, "a header at the start or within the first KiB");
        "%PDF-1.7"u8.CopyTo(junk.AsSpan(1024));
        junk[1023] = (byte)'x';
        Check(!format.Accepts(junk) && !format.Accepts("%!PS-Adobe-3.0\n"u8) && !format.Accepts([]), "no header within the first KiB, PostScript, nothing");

        int opened = 0;
        foreach (var sample in samples.Where(s => s.Readable))
        {
            var open = sample.Expected["open"]!;
            using var document = DocumentFormats.Open(sample.Path);
            string what = sample.Name;
            Check(document.Format == PdfFormat.FormatName && document.PageCount == (int)open["pageCount"]!, $"{what}: {document.Format}, {document.PageCount} pages, expected {open["pageCount"]}");
            var info = document.Info;
            Check(info.Version == (string?)open["version"] && !info.Encrypted, $"{what}: version {info.Version}, expected {open["version"]}");
            if (open["repaired"] is { } repaired)
            {
                Check(info.Repaired == repaired.GetValue<bool>(), $"{what}: repaired {info.Repaired}, expected {repaired}");
            }

            if (sample.Expected["info"] is JsonObject stated)
            {
                PdfCheckInfo(what, info, stated);   // the D3 and D4 samples state none
            }
            foreach (var page in sample.Expected["pages"]!.AsArray())
            {
                int index = (int)page!["index"]!;
                var read = document.ReadPage(index, DocumentParts.None);
                Check(read.Index == index && MathF.Abs(read.Width - (float)page["width"]!) < 1e-3f && MathF.Abs(read.Height - (float)page["height"]!) < 1e-3f && read.Rotation == (int)page["rotation"]!,
                    $"{what}: page {index} is {read.Width} x {read.Height} turned {read.Rotation}, expected {page["width"]} x {page["height"]} turned {page["rotation"]}");
                Check(read.Runs.Count == 0 && read.Images.Count == 0 && read.Issues.Count == 0, $"{what}: page {index}: nothing read yet but its size (D3 and D4 read its text and images)");
            }

            opened++;
        }

        Check(opened >= 40, $"only {opened} samples opened");
    }

    // The information as the sample's JSON has it: the six texts, the dates (with the offset they give, or none), the other keys.
    private static void PdfCheckInfo(string what, DocumentInfo info, JsonObject expected)
    {
        (string Key, string? Actual)[] texts = [("title", info.Title), ("author", info.Author), ("subject", info.Subject), ("keywords", info.Keywords), ("creator", info.Creator), ("producer", info.Producer)];
        foreach (var (key, actual) in texts)
        {
            Check(actual == (string?)expected[key], $"{what}: the information's {key} is \"{actual}\", expected \"{expected[key]}\"");
        }

        foreach (var (key, actual) in new[] { ("created", info.Created), ("modified", info.Modified) })
        {
            string? written = (string?)expected[key];
            if (written is null)
            {
                Check(actual is null, $"{what}: the information's {key} is {actual}, expected none");
                continue;
            }

            bool hasOffset = written.Length > 19;
            var date = DateTimeOffset.Parse(hasOffset ? written : written + "+00:00", CultureInfo.InvariantCulture);
            Check(actual is { } a && a.DateTime == date.DateTime && a.Offset == date.Offset, $"{what}: the information's {key} is {actual:O}, expected {written}");
        }

        var others = expected["other"]!.AsObject();
        Check(info.Entries.Count == others.Count && others.All(o => info.Entries.TryGetValue(o.Key, out var v) && v == (string?)o.Value),
            $"{what}: the other entries {string.Join(", ", info.Entries.Select(e => $"{e.Key}={e.Value}"))}, expected {others.ToJsonString()}");
    }

    private static void PdfSamplesObjects(Device device)
    {
        CpuOnly(device);
        var samples = PdfSamples();
        int objects = 0, streams = 0;
        foreach (var sample in samples.Where(s => s.Readable))
        {
            using var source = File.OpenRead(sample.Path);
            using var document = (PdfDocument)new PdfFormat().Open(source, sample.Name, new DocumentOptions());
            var file = document.File;
            string what = sample.Name;
            if (sample.D1)
            {
                PdfCheckStructure(what, file, sample.Expected);
                var live = file.LiveObjects().ToDictionary(o => o.Number, o => o.Generation);
                var listed = new HashSet<int>();
                foreach (var entry in sample.Expected["objects"]!.AsArray())
                {
                    int number = (int)entry!["num"]!;
                    listed.Add(number);
                    Check(live.TryGetValue(number, out int generation) && generation == (int)entry["gen"]!, $"{what}: object {number} {entry["gen"]} is not live (live: {string.Join(", ", live.Keys)})");
                    var actual = PdfCanonical(file, file.Get(number));
                    if (PdfDifference(entry["value"], actual, file, $"object {number}") is { } difference)
                    {
                        Check(false, $"{what}: {difference}");
                    }

                    objects++;
                }

                // Live objects the JSON leaves out are the file's own machinery: object streams and cross-reference streams.
                foreach (int number in live.Keys.Where(n => !listed.Contains(n)))
                {
                    var type = file.Get(number).AsDictionary?["Type"u8];
                    Check(type is { } t && (t.IsName("ObjStm"u8) || t.IsName("XRef"u8)), $"{what}: object {number} is live but not in the sample's objects ({file.Get(number)})");
                }

                if (sample.Expected["structure"]?["freed"] is JsonArray freed)
                {
                    foreach (var pair in freed)
                    {
                        int number = (int)pair![0]!;
                        Check(!live.ContainsKey(number) && file.Get(number).IsNull, $"{what}: object {number}, freed by an update, still reads");
                    }
                }

                // The same objects read from several threads at once, by a document of its own (nothing cached yet).
                using var fresh = (PdfDocument)new PdfFormat().Open(new MemoryStream(File.ReadAllBytes(sample.Path), writable: false), sample.Name, new DocumentOptions());
                int[] numbers = [.. live.Keys.Order()];
                var parallel = new string[numbers.Length];
                Parallel.For(0, numbers.Length, i => parallel[i] = PdfCanonical(fresh.File, fresh.File.Get(numbers[i]))?.ToJsonString() ?? "null");
                for (int i = 0; i < numbers.Length; i++)
                {
                    string once = PdfCanonical(file, file.Get(numbers[i]))?.ToJsonString() ?? "null";
                    Check(parallel[i] == once, $"{what}: object {numbers[i]} read from several threads is {parallel[i]}, one by one {once}");
                }
            }

            foreach (var listedStream in sample.Expected["streams"]?.AsArray() ?? new JsonArray())
            {
                int number = (int)listedStream!["object"]![0]!;
                var stream = file.Get(number).AsStream;
                Check(stream is not null, $"{what}: object {number} is not a stream");
                Check(stream!.Length == (long)listedStream["storedLength"]!, $"{what}: object {number} stores {stream.Length} bytes, expected {listedStream["storedLength"]} ({listedStream["note"]})");
                string? note = (string?)listedStream["note"];
                bool wrong = note is not null && (note.Contains("short", StringComparison.Ordinal) || note.Contains("past", StringComparison.Ordinal));
                Check(stream.LengthRecovered == wrong, $"{what}: object {number}'s length {(stream.LengthRecovered ? "was" : "was not")} found by endstream ({note ?? "its /Length is right"})");
                using var decoded = file.Decode(stream);
                string hash = Convert.ToHexStringLower(SHA256.HashData(decoded.Span));
                Check(decoded.Length == (long)listedStream["decodedLength"]! && hash == (string?)listedStream["decodedSha256"] && !decoded.Damaged,
                    $"{what}: object {number} through {listedStream["filters"]!.ToJsonString()} decodes to {decoded.Length} bytes ({hash}), expected {listedStream["decodedLength"]} ({listedStream["decodedSha256"]})");
                streams++;
            }
        }

        Check(objects >= 160 && streams >= 170, $"{objects} objects and {streams} streams checked");
    }

    // The cross-reference: the header's offset; each section (oldest first in the JSON, newest first in the file) its kind,
    // offset as written, compressed and free objects, every object it lists; a repaired file has none.
    private static void PdfCheckStructure(string what, PdfFile file, JsonObject expected)
    {
        var structure = expected["structure"]!;
        Check(file.HeaderOffset == (long)structure["headerOffset"]!, $"{what}: the header at {file.HeaderOffset}, expected {structure["headerOffset"]}");
        if (expected["open"]!["repaired"]?.GetValue<bool>() == true)
        {
            Check(file.Repaired && file.Xref.Sections.Count == 0, $"{what}: repaired {file.Repaired}, {file.Xref.Sections.Count} sections read");
            return;
        }

        if (file.Repaired)
        {
            return;   // either is right (offsets that count from a header with junk before it)
        }

        var revisions = structure["revisions"]!.AsArray();
        var sections = file.Xref.Sections.Reverse().ToList();
        Check(sections.Count == revisions.Count, $"{what}: {sections.Count} cross-reference sections, expected {revisions.Count}");
        for (int i = 0; i < revisions.Count; i++)
        {
            var revision = revisions[i]!;
            var section = sections[i];
            Check(section.Kind == (string?)revision["kind"] && section.Offset == (long)revision["offset"]!,
                $"{what}: section {i} is a {section.Kind} at {section.Offset}, expected a {revision["kind"]} at {revision["offset"]}");
            int[] Of(PdfEntryKind kind) => [.. section.Entries.Where(e => e.Entry.Kind == kind).Select(e => e.Number).Distinct().Order()];
            int[] Expected(string key) => [.. revision[key]!.AsArray().Select(n => (int)n!).Order()];
            Check(Of(PdfEntryKind.Compressed).SequenceEqual(Expected("compressed")) && Of(PdfEntryKind.Free).SequenceEqual(Expected("free")),
                $"{what}: section {i} compresses {string.Join(" ", Of(PdfEntryKind.Compressed))} and frees {string.Join(" ", Of(PdfEntryKind.Free))}, expected {revision["compressed"]!.ToJsonString()} and {revision["free"]!.ToJsonString()}");
            var listed = section.Entries.Where(e => e.Entry.Kind is PdfEntryKind.InUse or PdfEntryKind.Compressed).Select(e => e.Number).ToHashSet();
            Check(Expected("objects").All(listed.Contains), $"{what}: section {i} lists {string.Join(" ", listed.Order())}, which lacks some of {revision["objects"]!.ToJsonString()}");
        }
    }

    // An object in the samples' canonical form: names "/Name", strings {"string": hex}, references {"ref": [n, g]},
    // dictionaries {"dict": {...}}, streams {"stream": dictionary, "decodedLength", "decodedSha256"}.
    private static JsonNode? PdfCanonical(PdfFile file, PdfValue value)
    {
        switch (value.Kind)
        {
            case PdfKind.Null:
                return null;
            case PdfKind.Boolean:
                return JsonValue.Create(value.AsBoolean);
            case PdfKind.Integer:
                value.TryInteger(out long integer);
                return JsonValue.Create(integer);
            case PdfKind.Real:
                value.TryNumber(out double real);
                return JsonValue.Create(real);
            case PdfKind.Name:
                return JsonValue.Create("/" + value.AsName);
            case PdfKind.String:
                return new JsonObject { ["string"] = Convert.ToHexStringLower(value.AsString!.Bytes) };
            case PdfKind.Reference:
                return new JsonObject { ["ref"] = new JsonArray { value.AsReference.Number, value.AsReference.Generation } };
            case PdfKind.Array:
                var array = new JsonArray();
                foreach (var item in value.AsArray!.Items)
                {
                    array.Add(PdfCanonical(file, item));
                }

                return array;
            case PdfKind.Dictionary:
                return new JsonObject { ["dict"] = PdfCanonicalDictionary(file, value.AsDictionary!) };
            default:
                var stream = value.AsStream!;
                using (var decoded = file.Decode(stream))
                {
                    return new JsonObject
                    {
                        ["stream"] = PdfCanonicalDictionary(file, stream.Dictionary), ["decodedLength"] = decoded.Length,
                        ["decodedSha256"] = Convert.ToHexStringLower(SHA256.HashData(decoded.Span)),
                    };
                }
        }
    }

    private static JsonObject PdfCanonicalDictionary(PdfFile file, PdfDictionary dictionary)
    {
        var json = new JsonObject();
        for (int i = 0; i < dictionary.Count; i++)
        {
            json[dictionary.KeyAt(i).ToString()] = PdfCanonical(file, dictionary.ValueAt(i));
        }

        return json;
    }

    // Where `actual` differs from `expected` (numbers within 1e-4; {"raw": text} parsed as the object it writes), or null.
    private static string? PdfDifference(JsonNode? expected, JsonNode? actual, PdfFile file, string path)
    {
        if (expected is JsonObject { Count: 1 } raw && raw["raw"] is { } text)
        {
            var parser = new PdfParser(Encoding.Latin1.GetBytes((string)text!), 64, complete: true);
            expected = PdfCanonical(file, parser.ReadValue());
        }

        switch (expected)
        {
            case null:
                return actual is null ? null : $"{path} is {actual.ToJsonString()}, expected null";
            case JsonObject e:
                if (actual is not JsonObject a || a.Count != e.Count || e.Any(p => !a.ContainsKey(p.Key)))
                {
                    return $"{path} is {actual?.ToJsonString()}, expected {e.ToJsonString()}";
                }

                foreach (var (key, value) in e)
                {
                    if (PdfDifference(value, a[key], file, $"{path}/{key}") is { } inner)
                    {
                        return inner;
                    }
                }

                return null;
            case JsonArray e:
                if (actual is not JsonArray list || list.Count != e.Count)
                {
                    return $"{path} is {actual?.ToJsonString()}, expected {e.ToJsonString()}";
                }

                for (int i = 0; i < e.Count; i++)
                {
                    if (PdfDifference(e[i], list[i], file, $"{path}[{i}]") is { } inner)
                    {
                        return inner;
                    }
                }

                return null;
            default:
                var expectedValue = expected.AsValue();
                if (expectedValue.TryGetValue(out double number) && expectedValue.GetValueKind() == System.Text.Json.JsonValueKind.Number)
                {
                    return actual is JsonValue v && v.GetValueKind() == System.Text.Json.JsonValueKind.Number
                           && Math.Abs(double.Parse(v.ToJsonString(), CultureInfo.InvariantCulture) - number) <= 1e-4
                        ? null : $"{path} is {actual?.ToJsonString()}, expected {number.ToString(CultureInfo.InvariantCulture)}";
                }

                return actual?.ToJsonString() == expected.ToJsonString() ? null : $"{path} is {actual?.ToJsonString()}, expected {expected.ToJsonString()}";
        }
    }

    private static void PdfHostileAndEncrypted(Device device)
    {
        CpuOnly(device);
        var samples = PdfSamples();
        int hostile = 0;
        foreach (var sample in samples.Where(s => s.Hostile && s.Phases.Contains("D1")))
        {
            var rule = sample.Expected["hostile"]!;
            string limit = (string)rule["limit"]!, trips = (string)rule["trips"]!;
            var options = new DocumentOptions();
            foreach (var (key, value) in rule["options"]!.AsObject())
            {
                options = key switch
                {
                    "MaxStreamBytes" => options with { MaxStreamBytes = (long)value! },
                    "MaxDepth" => options with { MaxDepth = (int)value! },
                    "MaxObjects" => options with { MaxObjects = (long)value! },
                    _ => throw new InvalidOperationException($"{sample.Name}: an option {key} the test does not set"),
                };
            }

            var clock = Stopwatch.StartNew();
            string where = "nowhere";
            try
            {
                using var document = DocumentFormats.Open(sample.Path, options);
                where = "page 0 read";
                document.ReadPage(0, DocumentParts.All);
                Check(false, $"{sample.Name}: read without an error (expected one naming {limit}, {trips})");
            }
            catch (InvalidDataException e)
            {
                bool atOpen = where == "nowhere";
                Check(atOpen == trips.StartsWith("open", StringComparison.Ordinal), $"{sample.Name}: failed at {(atOpen ? "open" : where)}, expected at {trips}: {e.Message}");
                Check(e.Message.Contains(limit, StringComparison.Ordinal) && (limit == "cycle" || e.Message.Contains("DocumentOptions." + limit, StringComparison.Ordinal))
                      && e.Message.Contains(Path.GetFileName(sample.Path), StringComparison.Ordinal), $"{sample.Name}: \"{e.Message}\" does not name {limit} (and its option) and the file");
            }

            Check(clock.Elapsed.TotalSeconds < 20, $"{sample.Name}: took {clock.Elapsed.TotalSeconds:F1} s to fail, not fast");
            hostile++;
        }

        Check(hostile >= 9, $"{hostile} hostile D1 samples");

        int encrypted = 0;
        foreach (var sample in samples.Where(s => s.Encrypted))
        {
            try
            {
                using var document = DocumentFormats.Open(sample.Path);
                Check(false, $"{sample.Name}: an encrypted file opened before phase D2");
            }
            catch (NotSupportedException e)
            {
                string handler = (string?)sample.Expected["openError"]?["handler"] ?? "D2";
                Check(e.Message.Contains(handler, StringComparison.Ordinal) && e.Message.Contains("encrypted", StringComparison.Ordinal), $"{sample.Name}: \"{e.Message}\" does not name {handler}");
            }

            encrypted++;
        }

        Check(encrypted >= 20, $"{encrypted} encrypted samples");
    }

    private static void PdfFormatSuite(Device device)
    {
        CpuOnly(device);
        var all = PdfSamples();
        var samples = new List<DocumentSample>();
        foreach (var sample in all)
        {
            if (sample.Readable)
            {
                samples.Add(new DocumentSample(sample.Path, Pages: (int)sample.Expected["open"]!["pageCount"]!));
            }
            else if (sample.Encrypted)
            {
                samples.Add(new DocumentSample(sample.Path, Error: (string?)sample.Expected["openError"]?["handler"] ?? "D2"));
            }
            else if (sample.Hostile && sample.Phases.Contains("D1") && (string)sample.Expected["hostile"]!["limit"]! is "cycle" or "MaxDepth")
            {
                // The limits these trip hold at the defaults; the bombs are checked with their own limits (above), not the measured ones.
                samples.Add(new DocumentSample(sample.Path, Error: (string)sample.Expected["hostile"]!["limit"]!));
            }
        }

        Conformance.CheckDocumentFormat(DocumentFormats.Get(PdfFormat.FormatName), samples, options: new ContractCheckOptions { RandomCases = 60 }).ThrowIfFailed();
    }

    private static void PdfHandMade(Device device)
    {
        CpuOnly(device);

        // The lexer: one dictionary holding every kind of object.
        byte[] syntax = Encoding.Latin1.GetBytes("<< /A#20B (a\\(b\\)c\\\\d\\101\\7x\\n\\r\\t\\b\\f(nested (deep)) end) /H <48 65 6c 6C 6f 7> /N [1 -2 +3 4.5 -.5 0 0 R 12 0 R 7 8] "
                                                 + "/L (line1\r\nline2\rline3\\\nmore) /T true /F false /Z null /E [] /D <<>> % a comment\n /Last 4 >>");
        var parser = new PdfParser(syntax, 64, complete: true);
        var value = parser.ReadValue();
        var dictionary = value.AsDictionary!;
        Check(dictionary.Count == 10 && dictionary.KeyAt(0).ToString() == "A B", $"the dictionary: {value}");
        Check(dictionary["A B"u8].AsString!.Bytes.AsSpan().SequenceEqual("a(b)c\\dA\u0007x\n\r\t\b\f(nested (deep)) end"u8), $"a literal string: {Encoding.Latin1.GetString(dictionary["A B"u8].AsString!.Bytes)}");
        Check(dictionary["H"u8].AsString!.Bytes.AsSpan().SequenceEqual("Hellop"u8), "a hex string with an odd last digit");
        Check(dictionary["L"u8].AsString!.Bytes.AsSpan().SequenceEqual("line1\nline2\nline3more"u8), "line ends inside a string, and a continuation");
        var numbers = dictionary["N"u8].AsArray!;
        Check(numbers.Count == 9 && numbers[0].TryInteger(out long one) && one == 1 && numbers[1].TryInteger(out long minus) && minus == -2 && numbers[2].TryInteger(out long plus) && plus == 3
              && numbers[3].TryNumber(out double half) && half == 4.5 && numbers[4].TryNumber(out double small) && small == -0.5 && numbers[5].AsReference == new PdfReference(0, 0)
              && numbers[6].AsReference == new PdfReference(12, 0) && numbers[7].TryInteger(out long seven) && seven == 7 && numbers[8].TryInteger(out long eight) && eight == 8,
            $"numbers and references: {dictionary["N"u8]}");
        Check(dictionary["T"u8].AsBoolean && !dictionary["F"u8].AsBoolean && dictionary["Z"u8].IsNull && dictionary.TryGet("Z"u8, out _) && !dictionary.Contains("Z"u8)
              && dictionary["E"u8].AsArray!.Count == 0 && dictionary["D"u8].AsDictionary!.Count == 0 && dictionary["Last"u8].TryInteger(out long last) && last == 4,
            "booleans, null, empty containers, a comment");
        Check(PdfParser.TryNumber("--5"u8, out _, out double twice, out _) && twice == 5 && PdfParser.TryNumber("1.2.3"u8, out _, out double points, out _) && points == 1.2
              && !PdfParser.TryNumber("1x"u8, out _, out _, out _), "numbers as damaged files write them");
        try
        {
            new PdfParser(Encoding.Latin1.GetBytes(new string('[', 100) + new string(']', 100)), 64, complete: true).ReadValue();
            Check(false, "100 nested arrays were read under a limit of 64");
        }
        catch (InvalidDataException e)
        {
            Check(e.Message.Contains("MaxDepth", StringComparison.Ordinal), e.Message);
        }

        Check(PdfText.Decode([0x80, 0x85, 0x41, 0xA0]) == "•–A€" && PdfText.Decode([0xFE, 0xFF, 0x06, 0x27, 0xD8, 0x3D, 0xDC, 0xC8]) == "ا\U0001F4C8"
              && PdfText.Decode([0xEF, 0xBB, 0xBF, 0x43, 0x61, 0x66, 0xC3, 0xA9]) == "Café" && PdfText.Decode([0xFE, 0xFF, 0x00, 0x1B, 0x65, 0x6E, 0x00, 0x1B, 0x00, 0x41]) == "A",
            "PDFDocEncoding, UTF-16 with a character outside the BMP, UTF-8, a language mark dropped");
        Check(PdfText.Date("D:20240315143000+02'00'"u8) == new DateTimeOffset(2024, 3, 15, 14, 30, 0, TimeSpan.FromHours(2)) && PdfText.Date("D:2024"u8) == new DateTimeOffset(2024, 1, 1, 0, 0, 0, TimeSpan.Zero)
              && PdfText.Date("D:20240315143000-05'30"u8)!.Value.Offset == new TimeSpan(-5, -30, 0) && PdfText.Date("20240316091500Z"u8)!.Value.Offset == TimeSpan.Zero
              && PdfText.Date("D:20241332"u8) is null && PdfText.Date("yesterday"u8) is null, "dates: whole, the year alone, an offset without its last apostrophe, Z, out of range, none");

        // A two-page file written here: its pages read alike from a file, memory and a plain stream, and with no cache.
        byte[] content = Encoding.ASCII.GetBytes("BT /F1 12 Tf 72 720 Td (A letter of thanks) Tj ET");
        byte[] plain = PdfHandMadeFile([
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 7 /MediaBox [0 0 612 792] /Rotate 90 >>",
            "<< /Type /Page /Parent 2 0 R /Contents 5 0 R >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 400] /CropBox [10 10 210 310] /Rotate -90 /UserUnit 2 >>",
            PdfHandMadeStream("/Filter /FlateDecode", PdfDeflate(content, header: false)),
        ]);
        string folder = Path.Combine(Path.GetTempPath(), "idrak-documents-pdf", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(folder);
        try
        {
            string path = Path.Combine(folder, "letter.pdf");
            File.WriteAllBytes(path, plain);
            var expected = new[] { (792f, 612f, 90), (600f, 400f, 270) };
            void Pages(IDocument document, string how)
            {
                Check(document.PageCount == 2 && !document.Info.Repaired, $"{how}: {document.PageCount} pages (the kids win over /Count 7), repaired {document.Info.Repaired}");
                for (int i = 0; i < 2; i++)
                {
                    var page = document.ReadPage(i, DocumentParts.All);
                    Check((page.Width, page.Height, page.Rotation) == expected[i] && page.Issues.Count == 0, $"{how}: page {i} is {page.Width} x {page.Height} turned {page.Rotation}");
                }
            }

            using (var fromFile = DocumentFormats.Open(path))
            {
                Pages(fromFile, "a file");
            }

            using (var fromMemory = DocumentFormats.Open(plain.AsMemory(), "letter.pdf"))
            {
                Pages(fromMemory, "memory");
            }

            using (var stream = new BufferedStream(new MemoryStream(plain, writable: false)))
            using (var fromStream = DocumentFormats.Open(stream, "letter.pdf", new DocumentOptions { CacheBytes = 0 }))
            {
                Pages(fromStream, "a plain stream, no cache");
            }

            // The same file with its table's offsets shifted: rebuilt by scanning, and said so.
            byte[] shifted = [.. "%PDF-1.7\n% a comment put in after the file was written\n"u8, .. plain.AsSpan(9)];
            using (var repaired = DocumentFormats.Open(shifted.AsMemory(), "shifted.pdf"))
            {
                Check(repaired.Info.Repaired && repaired.PageCount == 2 && repaired.ReadPage(1, DocumentParts.None).Width == 600f, "a shifted table: repaired, two pages");
            }

            // A stream bomb (8 MiB of zeros in a few KiB) under a 1 MiB limit, from each source; a page tree 70 levels deep under 64.
            byte[] bomb = PdfHandMadeFile([
                "<< /Type /Catalog /Pages 2 0 R >>",
                "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>",
                PdfHandMadeStream("/Filter /FlateDecode", PdfDeflate(new byte[8 << 20], header: true)),
            ]);
            File.WriteAllBytes(path, bomb);
            var tight = new DocumentOptions { MaxStreamBytes = 1 << 20 };
            foreach (var (how, open) in new (string, Func<IDocument>)[]
                     {
                         ("a file", () => DocumentFormats.Open(path, tight)), ("memory", () => DocumentFormats.Open(bomb.AsMemory(), "bomb.pdf", tight)),
                         ("a plain stream", () => DocumentFormats.Open(new BufferedStream(new MemoryStream(bomb, writable: false)), "bomb.pdf", tight)),
                     })
            {
                using var document = open();
                Check(document.ReadPage(0, DocumentParts.None).Issues.Count == 0, $"{how}: the page's size reads without its content");
                try
                {
                    document.ReadPage(0, DocumentParts.Text);
                    Check(false, $"{how}: a stream of 8 MiB was decoded under a limit of 1 MiB");
                }
                catch (InvalidDataException e)
                {
                    Check(e.Message.Contains("MaxStreamBytes", StringComparison.Ordinal) && e.Message.Contains(DocumentOptions.MaxStreamBytesVariable, StringComparison.Ordinal)
                          && e.Message.Contains("object 4", StringComparison.Ordinal), $"{how}: {e.Message}");
                }
            }

            var deep = new List<string> { "<< /Type /Catalog /Pages 2 0 R >>" };
            for (int level = 0; level < 70; level++)
            {
                deep.Add($"<< /Type /Pages /Kids [{level + 3} 0 R] /Count 1 >>");
            }

            deep.Add("<< /Type /Page /MediaBox [0 0 612 792] >>");
            try
            {
                using var document = DocumentFormats.Open(PdfHandMadeFile(deep).AsMemory(), "deep.pdf");
                Check(false, "a page tree 70 levels deep opened under a limit of 64");
            }
            catch (InvalidDataException e)
            {
                Check(e.Message.Contains("MaxDepth", StringComparison.Ordinal) && e.Message.Contains("page tree", StringComparison.Ordinal), e.Message);
            }

            using (var deeper = DocumentFormats.Open(PdfHandMadeFile(deep).AsMemory(), "deep.pdf", new DocumentOptions { MaxDepth = 80 }))
            {
                Check(deeper.PageCount == 1, "a limit raised to 80 reads it");
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // A file of the given objects (numbered from 1), a classic table and a trailer naming object 1 its catalog.
    private static byte[] PdfHandMadeFile(IReadOnlyList<string> objects)
    {
        var file = new MemoryStream();
        file.Write("%PDF-1.7\n%âã\n"u8);
        var offsets = new long[objects.Count];
        for (int i = 0; i < objects.Count; i++)
        {
            offsets[i] = file.Position;
            file.Write(Encoding.Latin1.GetBytes($"{i + 1} 0 obj\n{objects[i]}\nendobj\n"));
        }

        long table = file.Position;
        var text = new StringBuilder();
        text.Append(CultureInfo.InvariantCulture, $"xref\n0 {objects.Count + 1}\n0000000000 65535 f\r\n");
        foreach (long offset in offsets)
        {
            text.Append(CultureInfo.InvariantCulture, $"{offset:D10} 00000 n\r\n");
        }

        text.Append(CultureInfo.InvariantCulture, $"trailer\n<< /Size {objects.Count + 1} /Root 1 0 R >>\nstartxref\n{table}\n%%EOF\n");
        file.Write(Encoding.Latin1.GetBytes(text.ToString()));
        return file.ToArray();
    }

    // A stream object's text (its bytes as Latin-1, one character a byte).
    private static string PdfHandMadeStream(string entries, byte[] data) => $"<< {entries} /Length {data.Length} >>\nstream\n{Encoding.Latin1.GetString(data)}\nendstream";

    private static byte[] PdfDeflate(byte[] data, bool header)
    {
        var output = new MemoryStream();
        using (Stream compressor = header ? new ZLibStream(output, CompressionLevel.SmallestSize, leaveOpen: true) : new DeflateStream(output, CompressionLevel.SmallestSize, leaveOpen: true))
        {
            compressor.Write(data);
        }

        return output.ToArray();
    }
}
