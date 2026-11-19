// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using BenchDocumentsCommand = global::Idrak.Cli.Commands.Measure.BenchDocumentsCommand;
using DocumentBench = global::Idrak.Cli.Commands.Measure.DocumentBench;

// Plan 28, phase D13: idrak bench documents, run in-process on the samples tools/documents/make_fixtures.py writes into the
// ignored test-output/documents/samples, in its short mode (a few files and pages, one warm run): its cases, what each
// records (cold and warm apart), its settings, --save and --compare. Speeds are not checked here, only the output's shape;
// the real numbers come from the owner's devices.
internal abstract partial class DesktopTests
{
    internal static readonly (string Name, Action<Device> Run)[] DocumentBenchGroup =
    [
        ("documents bench: every case runs on the samples in the short mode, each cold and warm apart (time, managed bytes allocated, the working set at its highest), the same output every run (hashed); the library's document variables and its measured limits recorded; forms filled on the form samples", DocumentBenchShort),
        ("documents bench: --save and --compare round-trip (each case's time, allocation and working set matched, cold beside warm, the hashed outputs compared); the options' variables, the option winning; usage errors exit with 2", DocumentBenchSaveCompare),
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
                  && text.Contains("limits measured from the memory", StringComparison.Ordinal), $"the text output: {error}{text}");
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
}
