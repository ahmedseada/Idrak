// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using BenchDocumentsCommand = global::Idrak.Cli.Commands.Measure.BenchDocumentsCommand;
using DocumentBench = global::Idrak.Cli.Commands.Measure.DocumentBench;
using DocumentBenchOptions = global::Idrak.Cli.Commands.Measure.DocumentBenchOptions;
using DocumentBenchRun = global::Idrak.Cli.Commands.Measure.DocumentBenchRun;
using DocumentBenchSet = global::Idrak.Cli.Commands.Measure.DocumentBenchSet;
using DocumentBenchWork = global::Idrak.Cli.Commands.Measure.DocumentBenchWork;

// Plan 28, phase D13: idrak bench documents, run in-process on the samples tools/documents/make_fixtures.py writes into the
// ignored test-output/documents/samples, in its short mode (a few files and pages, one warm run): its cases, what each
// records (cold and warm apart), its settings, --save and --compare; and that one bench run's runs are comparable (the
// same files, the limits pinned once), --compare telling other limits from a real change. Speeds are not checked here,
// only the output's shape; the real numbers come from the owner's devices.
internal abstract partial class DesktopTests
{
    internal static readonly (string Name, Action<Device> Run)[] DocumentBenchGroup =
    [
        ("documents bench: every case runs on the samples in the short mode, each cold and warm apart (time, managed bytes allocated, the working set at its highest), the same output every run (hashed); the library's document variables and its measured limits recorded; forms filled on the form samples", DocumentBenchShort),
        ("documents bench: --save and --compare round-trip (each case's time, allocation and working set matched, cold beside warm, the hashed outputs compared); the options' variables, the option winning; usage errors exit with 2", DocumentBenchSaveCompare),
        ("documents bench: one bench run is comparable: the measured limits pinned once (every run reads under them, the saved results hold them) and a capped case takes the same files in every run whatever the cold run refused, so the samples drawn with --hash give the same output every run; a file whose output differs between runs is named with its case and its hash in each run", DocumentBenchSameEveryRun),
        ("documents bench: --compare notes limits other than the saved ones and says whether they explain a changed output (each changed file named); a deliberately changed output is reported as changed under the same limits and under other ones", DocumentBenchCompareLimits),
    ];

    // The samples (IDRAK_DOCUMENTS_SAMPLES, else test-output/documents/samples): missing, the group fails naming the command that makes them.
    private static string BenchSamples()
    {
        string folder = Environment.GetEnvironmentVariable("IDRAK_DOCUMENTS_SAMPLES") is { Length: > 0 } set ? set : RepositoryPath("test-output", "documents", "samples");
        Check(File.Exists(Path.Combine(folder, "manifest.json")), $"{folder} holds no samples. Make them with: python -I tools/documents/make_fixtures.py -o test-output/documents/samples");
        return Path.GetFullPath(folder);
    }

    // The start of a JSON answer, for a failure's message.
    private static string BenchHead(JsonNode node) => node.ToJsonString() is { Length: > 400 } text ? text[..400] + "..." : node.ToJsonString();

    private static void DocumentBenchShort(Device device)
    {
        CpuOnly(device);
        string samples = BenchSamples(), cache = MeasureTemp();
        try
        {
            JsonObject? json = null;
            WithEnvironment(new Dictionary<string, string?> { [DocumentOptions.CacheBytesVariable] = "64m" },
                () => json = IdrakJson(cache, "bench", "documents", "--short", "--hash", "--sets", samples, "--dpi", "50", "--font", TestData("fonts/dejavu-sans-subset.ttf")));
            Check((string?)json!["kind"] == "documents" && (string?)json["timing"] == DocumentBench.Method && (string?)json["device"] == "cpu", BenchHead(json));
            var settings = json["settings"]!;
            Check((long)settings["limits"]!["cache_bytes"]! == 64L << 20 && (string?)settings["variables"]![DocumentOptions.CacheBytesVariable] == "64m"
                  && (long)settings["limits"]!["max_stream_bytes"]! > 0 && (long)settings["limits"]!["max_objects"]! > 0 && (int)settings["limits"]!["max_depth"]! == DocumentOptions.DefaultMaxDepth,
                $"the library's settings as the variables give them, and the limits measured: {settings.ToJsonString()}");
            Check((int)settings["repeat"]! == 1 && (int)settings["files"]! == 6 && (int)settings["pages"]! == 2 && settings["dpi"]!.AsArray().Count == 1 && (string?)settings["signer"] is { } signer
                  && signer.Contains("Example Test Signer", StringComparison.Ordinal), $"the short mode and the generator's test certificate: {settings.ToJsonString()}");
            var set = json["sets"]!.AsArray().Single()!;
            Check((string?)set["name"] == "samples" && (int)set["files"]! > 100 && (int)set["left_out"]! >= 10, $"the samples, their hostile ones left out: {set.ToJsonString()}");

            var cases = json["cases"]!.AsArray().Select(c => c!.AsObject()).ToList();
            var labels = cases.Select(c => (string)c["label"]!).ToList();
            var expected = DocumentBench.Cases.Select(c => c == "draw" ? "draw 50 dpi" : c).ToList();
            Check(labels.SequenceEqual(expected), $"every case in order: {string.Join(", ", labels)}");
            foreach (var c in cases)
            {
                string label = (string)c["label"]!;
                var cold = c["cold"]!;
                var warm = c["warm"]!;
                Check((double)cold["ms"]! > 0 && (long)cold["allocated_bytes"]! > 0 && (long)cold["peak_working_set_bytes"]! > 0 && (double)cold["jit_ms"]! >= 0,
                    $"{label}: its cold run measured: {cold.ToJsonString()}");
                Check(warm["runs"]!.AsArray().Count == 1 && (double)warm["ms"]! > 0 && (long)warm["peak_working_set_bytes"]! > 0, $"{label}: one warm run measured apart: {warm.ToJsonString()}");
                Check(label == "fill" || (long)c["units"]! > 0, $"{label}: did nothing: {c.ToJsonString()}");
                Check((bool?)c["same_output"] == true && ((string?)c["output"])?.Length == 64, $"{label}: the same output cold and warm: {c["output"]}, {c["same_output"]}");
                Check(c["file_outputs"] is JsonObject files && files.All(f => ((string?)f.Value)?.Length == 64) && ((long)c["units"]! == 0 || files.Count > 0)
                      && c["limit_files"] is JsonArray && c["differing_files"] is null, $"{label}: each file's output hashed, none differing: {c.ToJsonString()[..Math.Min(400, c.ToJsonString().Length)]}");
            }

            JsonObject Case(string label) => cases.Single(c => (string)c["label"]! == label);
            Check((long)Case("text-lines")["pages"]! >= 6 && (long)Case("text-lines")["units"]! == (long)Case("text-parallel")["units"]!
                  && (string?)Case("text-lines")["output"] == (string?)Case("text-parallel")["output"], "the text read over the cores is the text read one page after another");
            Check((long)Case("write-vector")["items"]! > 0 && (long)Case("write-vector")["written_bytes"]! > 0 && (long)Case("write-image")["written_bytes"]! > 0,
                $"the writing cases wrote: {Case("write-vector").ToJsonString()}");
            Check((long)Case("check")["units"]! >= (long)Case("sign")["units"]!, $"check took every signature sign made: {Case("check")["units"]}, {Case("sign")["units"]}");
            var results = json["results"]!.AsArray();
            Check(results.Count == 3 * cases.Count && results.All(r => r!["cold"] is not null), $"three results a case, each with its cold value: {results.Count}");

            // The form samples: every text field filled and saved.
            var forms = IdrakJson(cache, "bench", "documents", "--short", "--cases", "fill", "--sets", Path.Combine(samples, "d9"));
            var fill = forms["cases"]!.AsArray().Single()!;
            Check((long)fill["units"]! >= 2 && (long)fill["items"]! >= 4 && (long)fill["written_bytes"]! > 0, $"fill on the form samples: {fill.ToJsonString()}");

            // The text output: a line a case, then the table.
            var (code, text, error) = MeasureIdrak(cache, "bench", "documents", "--short", "--cases", "open,text-lines", "--sets", samples);
            Check(code == 0 && text.Contains("text-lines on samples: ", StringComparison.Ordinal) && text.Contains("Warm ms", StringComparison.Ordinal)
                  && text.Contains("limits for every run (measured from the memory once", StringComparison.Ordinal), $"the text output: {error}{text}");
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }

    private static void DocumentBenchSaveCompare(Device device)
    {
        CpuOnly(device);
        string samples = BenchSamples(), cache = MeasureTemp();
        try
        {
            string[] args = ["bench", "documents", "--short", "--hash", "--cases", "open,text-lines", "--sets", samples];
            var first = IdrakJson(cache, [.. args, "--save", "first"]);
            Check(File.Exists(Path.Combine(cache, "bench", "first.json")) && (string?)first["saved"] is not null, "saved under the cache folder");
            var compared = IdrakJson(cache, [.. args, "--compare", "first"]);
            var rows = compared["compare"]!["results"]!.AsArray();
            Check(rows.Count == 6 && (bool)compared["compare"]!["same_timing"]! && rows.All(r => r!["saved"] is not null && r["saved_cold"] is not null && r["cold_change"] is not null),
                $"each case's three results matched, cold beside warm: {compared["compare"]!.ToJsonString()}");
            var outputs = compared["compare"]!["outputs"]!.AsArray();
            Check(outputs.Count == 2 && outputs.All(o => (bool)o!["same"]!), $"the outputs the same as saved: {outputs.ToJsonString()}");
            var (code, text, _) = MeasureIdrak(cache, [.. args, "--compare", "first"]);
            Check(code == 0 && text.Contains("Against 'first'", StringComparison.Ordinal) && text.Contains("outputs: all 2 hashed cases the same as 'first'", StringComparison.Ordinal), text);

            // The variables, and the options over them.
            JsonObject? fromVariables = null, overVariables = null;
            WithEnvironment(new Dictionary<string, string?>
            {
                [BenchDocumentsCommand.CasesVariable] = "open", [BenchDocumentsCommand.RepeatVariable] = "2", [BenchDocumentsCommand.SetsVariable] = samples,
                [BenchDocumentsCommand.DpiVariable] = "40,60", [BenchDocumentsCommand.DrawPagesVariable] = "1", [BenchDocumentsCommand.DrawFilesVariable] = "2",
            }, () =>
            {
                fromVariables = IdrakJson(cache, "bench", "documents");
                overVariables = IdrakJson(cache, "bench", "documents", "--cases", "draw", "-n", "1", "--sets", Path.Combine(samples, "d7"));
            });
            Check(fromVariables!["cases"]!.AsArray().Select(c => (string)c!["label"]!).SequenceEqual(["open"]) && fromVariables["cases"]![0]!["warm"]!["runs"]!.AsArray().Count == 2
                  && (string?)fromVariables["sets"]![0]!["name"] == "samples", $"the cases, repeats and sets from their variables: {BenchHead(fromVariables)}");
            var draws = overVariables!["cases"]!.AsArray();
            Check(draws.Select(c => (string)c!["label"]!).SequenceEqual(["draw 40 dpi", "draw 60 dpi"]) && draws.All(c => (long)c!["units"]! == 2 && c["warm"]!["runs"]!.AsArray().Count == 1)
                  && (int)overVariables["settings"]!["draw_pages"]! == 1 && (int)overVariables["settings"]!["draw_files"]! == 2,
                $"the options over the variables, the resolutions and caps from theirs: {draws.ToJsonString()}");

            // Usage errors.
            Check(MeasureIdrak(cache, "bench", "documents", "--cases", "nothing").Code == 2, "an unknown case");
            Check(MeasureIdrak(cache, "bench", "documents", "--sets", Path.Combine(cache, "missing")).Code == 2, "a set that is not there");
            Check(MeasureIdrak(cache, "bench", "documents", "--sets", samples, "--dpi", "0").Code == 2, "a resolution of 0");
            Check(MeasureIdrak(cache, "bench", "documents", "extra").Code == 2, "an argument");
            Check(MeasureIdrak(cache, "bench", "documents", "--sets", samples, "-n", "0").Code == 2, "no warm run");
            Check(MeasureIdrak(cache, "bench", "documents", "--sets", samples, "--compare", "missing").Error.Contains("No saved results", StringComparison.Ordinal), "missing saved results");
            Check(MeasureIdrak(cache, "b", "documents", "--help").Output.Contains("idrak bench documents", StringComparison.Ordinal), "the alias b documents");
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }

    private static void DocumentBenchSameEveryRun(Device device)
    {
        CpuOnly(device);
        string samples = BenchSamples(), cache = MeasureTemp();
        try
        {
            // Pinned: the library's options measure a limit again at every open; the bench's resolve each once.
            WithEnvironment(new Dictionary<string, string?> { [DocumentOptions.MaxStreamBytesVariable] = null, [DocumentOptions.MaxObjectsVariable] = null, [DocumentOptions.CacheBytesVariable] = null }, () =>
            {
                var measured = new DocumentOptions();
                var pinned = DocumentBench.Pin(measured);
                Check(measured.ToString().Contains("MaxStreamBytes = measured", StringComparison.Ordinal) && !pinned.ToString().Contains("measured", StringComparison.Ordinal)
                      && pinned.MaxStreamBytes > 0 && pinned.MaxObjects > 0 && pinned.CacheBytes >= 0 && pinned.MaxDepth == DocumentOptions.DefaultMaxDepth && pinned.Layout == measured.Layout,
                    $"the limits pinned: {measured} against {pinned}");
            });

            // Every run reads under the bench's options, not options of its own: a stream limit of 64 bytes pinned there
            // stops a sample's 205-byte content (refused or left out, naming the limit), the measured one does not.
            string file = Path.Combine(samples, "d3", "encodings.pdf");
            DocumentBenchRun TextLines(DocumentOptions read) => DocumentBench.Measure("text-lines", 0f, new DocumentBenchSet("encodings", Path.GetDirectoryName(file)!, [file], 0),
                new DocumentBenchOptions { Cases = ["text-lines"], Dpis = [150f], Hash = true, Read = read }, _ => { });
            var tight = TextLines(DocumentBench.Pin(new DocumentOptions { MaxStreamBytes = 64 }));
            var roomy = TextLines(DocumentBench.Pin(new DocumentOptions()));
            Check(tight.Work.LimitFiles.Contains(file) && tight.Work.Refused.All(r => DocumentBench.NamesLimit(r.Reason)), $"64 bytes pinned: the file met the limit: refused {string.Join("; ", tight.Work.Refused.Select(r => r.Reason))}");
            Check(roomy.Work.LimitFiles.Count == 0 && roomy.Work.Refused.Count == 0 && roomy.Work.Units > 0 && roomy.Work.FileOutputs.Single().File == file && roomy.Work.Output != tight.Work.Output,
                $"the measured limits pinned: read, its output hashed: {roomy.Work.Units} pages, {string.Join("; ", roomy.Work.Refused.Select(r => r.Reason))}");

            // The saved results hold the limits every run read under: here, the ones the variables pin.
            WithEnvironment(new Dictionary<string, string?> { [DocumentOptions.MaxStreamBytesVariable] = "300m", [DocumentOptions.MaxObjectsVariable] = "2000000", [DocumentOptions.CacheBytesVariable] = "0" },
                () => IdrakJson(cache, "bench", "documents", "--short", "--cases", "open", "--sets", samples, "--save", "pinned"));
            var limits = JsonNode.Parse(File.ReadAllText(Path.Combine(cache, "bench", "pinned.json")))!["settings"]!["limits"]!;
            Check((long)limits["max_stream_bytes"]! == 300L << 20 && (long)limits["max_objects"]! == 2_000_000 && (long)limits["cache_bytes"]! == 0 && (int)limits["max_depth"]! == DocumentOptions.DefaultMaxDepth,
                $"the saved limits: {limits.ToJsonString()}");

            // The samples drawn (at most 50 files spread over them, the encrypted ones among them refused in the cold run):
            // every run takes the same files, so every run gives the same output, file by file.
            var drawn = IdrakJson(cache, "bench", "documents", "--cases", "draw", "--dpi", "36", "-n", "2", "--hash", "--sets", samples);
            var draw = drawn["cases"]!.AsArray().Single()!;
            Check(draw["refused"]!.AsArray().Count > 0, $"the cold run refused files (the case that gave a false alarm): {draw.ToJsonString()[..Math.Min(400, draw.ToJsonString().Length)]}");
            Check((bool?)draw["same_output"] == true && draw["differing_files"] is null && draw["file_outputs"]!.AsObject().Count == (long)draw["files"]!
                  && draw["warm"]!["runs"]!.AsArray().All(r => (string?)r!["output"] == (string?)draw["output"]),
                $"the same files and the same output in every run: {draw["output"]}, {draw["same_output"]}, {draw["files"]} files, {draw["file_outputs"]!.AsObject().Count} hashed");

            // A file whose output differs between runs (made here: one run's output changed, another's file refused) is
            // named with its case and its hash in each run.
            string root = Path.Combine(samples, "d3");
            string a = Path.Combine(root, "first.pdf"), b = Path.Combine(root, "second.pdf"), c = Path.Combine(root, "third.pdf");
            static DocumentBenchRun Fake(params (string File, string Text)[] files)
            {
                var work = new DocumentBenchWork(hash: true);
                foreach (var (name, text) in files)
                {
                    var part = new DocumentBenchWork(hash: true) { Units = 1, Files = 1 };
                    part.Add(text);
                    work.Merge(name, part);
                }

                work.Finish();
                return new DocumentBenchRun(1, 0, 0, 0, 0, work);
            }

            var cold = Fake((a, "the same"), (b, "a page"), (c, "kept"));
            var warm1 = Fake((a, "the same"), (b, "a page"), (c, "kept"));
            var warm2 = Fake((a, "the same"), (b, "another page"));
            warm2.Work.Refuse(c, "a reason given for the test");
            Check(DocumentBench.Differences(cold, [warm1]).Count == 0 && cold.Work.Output == warm1.Work.Output, "runs alike: nothing differs");
            var differing = DocumentBench.Differences(cold, [warm1, warm2]);
            var lines = BenchDocumentsCommand.DifferenceLines("draw 36 dpi", new DocumentBenchSet("d3", root, [a, b, c], 0), cold, [warm1, warm2], differing);
            Check(cold.Work.Output != warm2.Work.Output && differing.Select(d => d.File).SequenceEqual([b, c]) && differing[0].Outputs[0] == differing[0].Outputs[1]
                  && differing[0].Outputs[2] != differing[0].Outputs[0] && differing[1].Outputs[2] is null,
                $"the files that differ, run by run: {string.Join("; ", differing.Select(d => $"{d.File}: {string.Join(", ", d.Outputs)}"))}");
            Check(lines.Count == 2 && lines[0].StartsWith("    differs in draw 36 dpi: second.pdf: cold ", StringComparison.Ordinal) && lines[0].Contains(", warm 2 ", StringComparison.Ordinal)
                  && lines[1].StartsWith("    differs in draw 36 dpi: third.pdf: ", StringComparison.Ordinal) && lines[1].EndsWith("warm 2 refused (a reason given for the test)", StringComparison.Ordinal),
                $"named with the step and each run's hash: {string.Join(" | ", lines)}");
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }

    private static void DocumentBenchCompareLimits(Device device)
    {
        CpuOnly(device);
        string samples = BenchSamples(), cache = MeasureTemp();
        try
        {
            // The limits pinned by their variables, the stream's other between the save and the comparison.
            static Dictionary<string, string?> Limits(string stream) => new()
            {
                [DocumentOptions.MaxStreamBytesVariable] = stream, [DocumentOptions.MaxObjectsVariable] = "1000000", [DocumentOptions.CacheBytesVariable] = "64m",
            };

            string[] args = ["bench", "documents", "--short", "--hash", "--cases", "open,text-lines", "--sets", samples];
            WithEnvironment(Limits("1g"), () => IdrakJson(cache, [.. args, "--save", "limits-1g"]));
            JsonObject? other = null;
            (int Code, string Output, string Error) otherText = (0, "", "");
            WithEnvironment(Limits("2g"), () =>
            {
                other = IdrakJson(cache, [.. args, "--compare", "limits-1g"]);
                otherText = MeasureIdrak(cache, [.. args, "--compare", "limits-1g"]);
            });
            var limits = other!["compare"]!["limits"]!;
            Check((bool?)limits["same"] == false && limits["changed"]!.AsArray().Select(k => (string?)k).SequenceEqual(["max_stream_bytes"])
                  && (long)limits["saved"]!["max_stream_bytes"]! == 1L << 30 && (long)limits["now"]!["max_stream_bytes"]! == 2L << 30,
                $"the other limits noted: {limits.ToJsonString()}");
            Check(other["compare"]!["outputs"]!.AsArray().All(o => (bool)o!["same"]!)
                  && otherText.Output.Contains("limits: other than 'limits-1g' read under: a stream 2.00 GiB (saved: a stream 1.00 GiB)", StringComparison.Ordinal)
                  && otherText.Output.Contains("outputs: all 2 hashed cases the same as 'limits-1g'", StringComparison.Ordinal),
                $"no file near the limits, the outputs the same: {otherText.Output}");

            // Other limits that change an output explain it: a file refused (or cut) by a stream limit pinned at 64 bytes,
            // read under a gigabyte.
            string file = Path.Combine(samples, "d3", "encodings.pdf");
            string[] one = ["bench", "documents", "--hash", "-n", "1", "--cases", "text-lines", "--sets", file];
            WithEnvironment(Limits("64"), () => IdrakJson(cache, [.. one, "--save", "limits-64"]));
            JsonObject? explained = null;
            (int Code, string Output, string Error) explainedText = (0, "", "");
            WithEnvironment(Limits("1g"), () =>
            {
                explained = IdrakJson(cache, [.. one, "--compare", "limits-64"]);
                explainedText = MeasureIdrak(cache, [.. one, "--compare", "limits-64"]);
            });
            var row = explained!["compare"]!["outputs"]!.AsArray().Single()!;
            Check((bool?)row["same"] == false && row["files"]!.AsArray().Select(f => (string?)f).SequenceEqual(["encodings.pdf"]) && (bool?)row["limits_explain"] == true
                  && explainedText.Output.Contains("CHANGED: text-lines encodings", StringComparison.Ordinal)
                  && explainedText.Output.Contains("1 file changed (encodings.pdf); the other limits explain it", StringComparison.Ordinal),
                $"a change the other limits explain, listed as changed: {row.ToJsonString()} {explainedText.Output}");

            // A deliberately changed output: one file's saved output (and so its case's) altered. Under the same limits it is
            // a change, its file named; under other limits too, since that file met none.
            string savedPath = Path.Combine(cache, "bench", "limits-1g.json");
            var tampered = JsonNode.Parse(File.ReadAllText(savedPath))!.AsObject();
            var textCase = tampered["cases"]!.AsArray().Single(c => (string?)c!["label"] == "text-lines")!.AsObject();
            var fileOutputs = textCase["file_outputs"]!.AsObject();
            string victim = fileOutputs.First().Key;
            fileOutputs[victim] = new string('0', 64);
            textCase["output"] = new string('0', 64);
            File.WriteAllText(Path.Combine(cache, "bench", "tampered.json"), tampered.ToJsonString());
            foreach (string stream in new[] { "1g", "2g" })
            {
                JsonObject? compared = null;
                (int Code, string Output, string Error) text = (0, "", "");
                WithEnvironment(Limits(stream), () =>
                {
                    compared = IdrakJson(cache, [.. args, "--compare", "tampered"]);
                    text = MeasureIdrak(cache, [.. args, "--compare", "tampered"]);
                });
                var outputs = compared!["compare"]!["outputs"]!.AsArray();
                var changed = outputs.Single(o => (string?)o!["case"] == "text-lines")!;
                Check((bool?)outputs.Single(o => (string?)o!["case"] == "open")!["same"] == true && (bool?)changed["same"] == false
                      && changed["files"]!.AsArray().Select(f => (string?)f).SequenceEqual([victim]) && (bool?)changed["limits_explain"] == false
                      && text.Output.Contains("outputs: 1 of 2 hashed cases the same as 'tampered'; CHANGED: text-lines samples", StringComparison.Ordinal)
                      && text.Output.Contains($"text-lines samples: 1 file changed ({victim}); "
                          + (stream == "1g" ? "read under the same limits" : "NOT explained by the other limits: 1 of them met none"), StringComparison.Ordinal),
                    $"a changed output reported under a {stream} stream limit: {changed.ToJsonString()} {text.Output}");
            }
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }
}
