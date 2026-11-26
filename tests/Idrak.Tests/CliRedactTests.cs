// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Plan 28, phase D10 (decision 17): idrak data redact on the D10 redaction sample tools/documents/make_fixtures.py writes
// into the ignored test-output/documents/samples, and on a file of many pages made here. The redacted files are kept in
// test-output/documents/redacted for tools/documents/check_written.py.
internal abstract partial class DesktopTests
{
    internal static readonly (string Name, Action<Device> Run)[] CliRedactGroup =
    [
        ("cli: data redact: usage errors: no areas, no output, the output the input, a missing file, a boxes file that is not areas, a page past the end, a colour that is not one", CliRedactUsage),
    ];

    private static (int Code, string Out, string Err) RedactCli(params string[] args)
    {
        var output = new StringWriter();
        var error = new StringWriter();
        int code = Idrak.Cli.CommandLine.Run([.. args, "-C", Path.Combine(Path.GetTempPath(), "idrak-cli-redact-no-config.json")], output, error);
        return (code, output.ToString(), error.ToString());
    }

    // The D10 redaction sample (IDRAK_DOCUMENTS_SAMPLES, else test-output/documents/samples): missing, the group fails naming the command that makes it.
    private static string RedactSample(string name)
    {
        string folder = Environment.GetEnvironmentVariable("IDRAK_DOCUMENTS_SAMPLES") is { Length: > 0 } set ? set : RepositoryPath("test-output", "documents", "samples");
        string path = Path.Combine(folder, "d10", name + ".pdf");
        Check(File.Exists(path), $"{path} is not there. Make the samples with: python -I tools/documents/make_fixtures.py -o test-output/documents/samples");
        return path;
    }

    private static string RedactFolder() => Directory.CreateDirectory(RepositoryPath("test-output", "documents", "redacted")).FullName;

    private static void CliRedactUsage(Device device)
    {
        CpuOnly(device);
        string sample = RedactSample("redaction"), folder = RedactFolder();
        string output = Path.Combine(folder, "usage.pdf");
        void Refused(string why, string text, params string[] args)
        {
            var (code, _, error) = RedactCli(["data", "redact", .. args]);
            Check(code == 2 && error.Contains(text, StringComparison.Ordinal), $"{why}: exit {code}: {error}");
        }

        Refused("no areas", "--boxes FILE, --text TEXT or --marked", sample, "-o", output);
        Refused("no output", "-o, --out", sample, "--marked");
        Refused("the output the input", "another file", sample, "--marked", "-o", sample);
        Refused("a missing file", "is not there", Path.Combine(folder, "nope.pdf"), "--marked", "-o", output);
        string bad = Path.Combine(folder, "bad-areas.json");
        File.WriteAllText(bad, """{"page": 1}""");
        Refused("a boxes file that is not an array", "JSON array", sample, "--boxes", bad, "-o", output);
        File.WriteAllText(bad, """[{"page": 1, "box": [1, 2, 3]}]""");
        Refused("an area without four numbers", "area 1", sample, "--boxes", bad, "-o", output);
        File.WriteAllText(bad, """[{"page": 9, "box": [1, 2, 3, 4]}]""");
        Refused("a page past the end", "page 9", sample, "--boxes", bad, "-o", output);
        Refused("a colour that is not one", "not a colour", sample, "--marked", "--fill", "teal-ish", "-o", output);
        Check(!File.Exists(output) && !File.Exists(output + ".partial"), "nothing written when refused");
    }
}
