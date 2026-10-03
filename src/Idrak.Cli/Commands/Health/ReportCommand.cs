// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.IO.Compression;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Diagnostics;

namespace Idrak.Cli.Commands.Health;

/// <summary>
/// <c>idrak report</c>: one report (Markdown and JSON) with the machine, the devices, the drivers and, on request, the
/// test list's results (<c>--tests</c>, from a source checkout) and kernel benchmarks (<c>--bench</c>), for the README's
/// tested-on tables. <c>--zip</c> puts both files (and the test log) in one archive.
/// </summary>
internal sealed class ReportCommand : Command
{
    public override string Name => "report";

    public override string Summary => "A report (Markdown and JSON) of the machine, devices, drivers, tests and benchmarks";

    public override string Usage => """
        [--tests] [--bench] [--readme] [-o FILE] [--zip] [--source DIR]

        Options:
              --tests       run the library's test list (from a source checkout, on --device or every listed device)
                            and add the results; IDRAK_FILTER and IDRAK_TIMEOUT pass through to the test runner
              --bench       time matrix products on each device (GFLOP/s) and add the results
              --readme      print the README's tested-on rows for this machine, ready to paste (files only with
                            -o or --zip)
          -o, --out FILE    where to write (default idrak-report.md, with idrak-report.json beside it; a .zip name
                            implies --zip)
              --zip         one .zip with report.md, report.json and the test log
              --source DIR  the source checkout for --tests (default: found from the current folder upwards)

        Without --tests and --bench the report has the machine, devices and drivers only. The benchmarks are the kernel
        products only; model speeds come from idrak bench.

        Examples:
          idrak report
          idrak report --bench -d vulkan:0
          idrak report --tests --readme
          idrak report --tests --bench --zip -o phone-report.zip
        """;

    public override IReadOnlyCollection<string> ValueOptions => ["--out", "--source"];

    public override IReadOnlyCollection<string> Flags => ["--tests", "--bench", "--zip", "--readme"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-o"] = "--out" };

    public override int Run(CommandContext context)
    {
        bool zip = context.Flag("--zip") || context.Option("--out")?.EndsWith(".zip", StringComparison.OrdinalIgnoreCase) == true;
        string output = Path.GetFullPath(context.Option("--out") ?? (zip ? "idrak-report.zip" : "idrak-report.md"));
        string? source = context.Flag("--tests") ? FindSource(context.Option("--source")) : null;

        var report = new JsonObject
        {
            ["date"] = DateTime.UtcNow.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture),
            ["tool"] = Machine.ToolVersion,
            ["libraries"] = new JsonObject([.. Machine.Libraries().Select(l => KeyValuePair.Create(l.Name, (JsonNode?)l.Version))]),
            ["machine"] = Machine.Json(),
        };
        context.Detail("reading the devices");
        var devices = Machine.Devices(context);
        report["devices"] = new JsonArray([.. devices.Select(d => (JsonNode)Machine.Json(d))]);
        report["drivers"] = new JsonObject([.. VersionCommand.Drivers().Select(d => KeyValuePair.Create(d.Device, (JsonNode?)d.Driver))]);

        string? testLog = null;
        if (source is not null)
        {
            (report["tests"], testLog) = RunTests(context, source);
        }

        if (context.Flag("--bench"))
        {
            report["benchmarks"] = Bench(context, devices.Where(d => d.Error is null && (d.Listed || context.Option("--device") is not null)).ToList());
        }

        string markdown = Markdown(report);
        bool readme = context.Flag("--readme");
        if (readme)
        {
            report["readme"] = Readme(report);
            if (!context.Json)
            {
                context.Output.Write((string)report["readme"]!);
            }
        }

        // --readme alone prints the rows; the files are written otherwise, or when -o or --zip asks for them too.
        bool files = !readme || zip || context.Option("--out") is not null;
        if (files)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(output)!);
        }

        if (files && zip)
        {
            File.Delete(output);
            using var archive = ZipFile.Open(output, ZipArchiveMode.Create);
            Add(archive, "report.md", markdown);
            Add(archive, "report.json", report.ToJsonString(CommandContext.JsonOutput));
            if (testLog is not null)
            {
                Add(archive, "tests.log", testLog);
            }

            context.Write($"Wrote {output}");
        }
        else if (files)
        {
            string json = Path.ChangeExtension(output, ".json");
            File.WriteAllText(output, markdown);
            File.WriteAllText(json, report.ToJsonString(CommandContext.JsonOutput));
            context.Write($"Wrote {output} and {json}");
        }

        report["file"] = files ? output : null;
        context.WriteJson(report);
        return report["tests"]?["failed"]?.GetValue<int>() is > 0 ? ExitCodes.Failed : ExitCodes.Ok;
    }

    private static void Add(ZipArchive archive, string name, string text)
    {
        using var writer = new StreamWriter(archive.CreateEntry(name).Open(), new UTF8Encoding(false));
        writer.Write(text);
    }

    // The checkout holding tests/Idrak.Tests: the given folder, or the current folder or one above it.
    private static string FindSource(string? given)
    {
        const string Project = "tests/Idrak.Tests/Idrak.Tests.csproj";
        if (given is not null)
        {
            return File.Exists(Path.Combine(given, Project)) ? Path.GetFullPath(given)
                : throw new UsageException($"{Path.GetFullPath(given)} is not an Idrak source checkout (no {Project}).");
        }

        for (var folder = new DirectoryInfo(Environment.CurrentDirectory); folder is not null; folder = folder.Parent)
        {
            if (File.Exists(Path.Combine(folder.FullName, Project)))
            {
                return folder.FullName;
            }
        }

        throw new UsageException($"--tests runs the test list from a source checkout: run it inside one, or give --source DIR (git clone the repository first).");
    }

    // Runs the test runner as a child process and reads its PASS / FAIL lines and summary.
    private static (JsonObject Result, string Log) RunTests(CommandContext context, string source)
    {
        var start = new ProcessStartInfo("dotnet") { RedirectStandardOutput = true, RedirectStandardError = true, WorkingDirectory = source };
        foreach (string arg in (string[])["run", "-c", "Release", "--project", Path.Combine("tests", "Idrak.Tests")])
        {
            start.ArgumentList.Add(arg);
        }

        if (context.Option("--device") is { } device)
        {
            start.Environment["IDRAK_DEVICES"] = device;
        }

        context.Write($"Running the test list in {source} (this takes several minutes)...");
        using var process = Process.Start(start) ?? throw new InvalidOperationException("could not start dotnet; is the .NET SDK installed?");
        using var interrupt = new Interrupt(context);
        using var registration = interrupt.Token.Register(() => process.Kill(entireProcessTree: true));
        var log = new StringBuilder();
        var perDevice = new JsonObject();
        JsonObject? current = null;
        int passed = 0, failed = 0;
        var errors = process.StandardError.ReadToEndAsync();
        using var progress = new ProgressLine(context, "tests");
        while (process.StandardOutput.ReadLine() is { } line)
        {
            log.AppendLine(line);
            context.Detail(line);
            string trimmed = line.TrimStart();
            if (line.StartsWith("== ", StringComparison.Ordinal))
            {
                string name = line[3..];
                int colon = name.IndexOf(':', name.IndexOf(':') + 1);
                string key = colon > 0 ? name[..colon] : name;
                current = new JsonObject { ["name"] = colon > 0 ? name[(colon + 1)..].Trim() : "", ["passed"] = 0, ["failed"] = 0, ["failures"] = new JsonArray() };
                perDevice[key] = current;
            }
            else if (trimmed.StartsWith("PASS ", StringComparison.Ordinal) && current is not null)
            {
                passed++;
                current["passed"] = current["passed"]!.GetValue<int>() + 1;
                progress.Advance();
            }
            else if (trimmed.StartsWith("FAIL ", StringComparison.Ordinal) || trimmed.StartsWith("HANG ", StringComparison.Ordinal))
            {
                failed++;
                if (current is not null)
                {
                    current["failed"] = current["failed"]!.GetValue<int>() + 1;
                    current["failures"]!.AsArray().Add(trimmed[5..]);
                }

                progress.Advance();
            }
        }

        process.WaitForExit();
        progress.Finish($"tests: {passed} passed, {failed} failed");
        log.Append(errors.Result);
        if (perDevice.Count == 0 && process.ExitCode != 0)
        {
            throw new InvalidOperationException($"the test runner did not start (exit code {process.ExitCode}): {errors.Result.Trim()}");
        }

        context.Write($"Tests: {passed} passed, {failed} failed");
        return (new JsonObject { ["passed"] = passed, ["failed"] = failed, ["devices"] = perDevice, ["interrupted"] = interrupt.Requested }, log.ToString());
    }

    // Matrix products of a few sizes on each device: the median of five timed runs after one warm-up.
    internal static JsonArray Bench(CommandContext context, IReadOnlyList<DeviceInfo> devices)
    {
        int[] sizes = [256, 512];
        var results = new JsonArray();
        using var progress = new ProgressLine(context, "bench", devices.Count * sizes.Length);
        foreach (var info in devices)
        {
            var device = Device.Parse(info.Device);
            foreach (int n in sizes)
            {
                progress.Label = $"bench {info.Device} {n}x{n}x{n}";
                double gflops;
                using (new TensorScope())
                {
                    var random = new Random(1);
                    var a = Tensor.Uniform([n, n], -1, 1, random, device);
                    var b = Tensor.Uniform([n, n], -1, 1, random, device);
                    a.MatMul(b).Dispose();
                    device.Synchronize();
                    var times = new List<double>();
                    for (int i = 0; i < 5; i++)
                    {
                        var watch = Stopwatch.StartNew();
                        a.MatMul(b).Dispose();
                        device.Synchronize();
                        times.Add(watch.Elapsed.TotalSeconds);
                    }

                    times.Sort();
                    gflops = 2.0 * n * n * n / times[2] / 1e9;
                }

                context.Detail($"{info.Device} matmul {n}: {gflops:F1} GFLOP/s");
                results.Add(new JsonObject { ["device"] = info.Device, ["benchmark"] = $"matmul {n}x{n}x{n} float32", ["gflops"] = Math.Round(gflops, 2) });
                progress.Advance();
            }
        }

        return results;
    }

    /// <summary>The report as Markdown, for pasting into the README's tested-on tables.</summary>
    internal static string Markdown(JsonObject report)
    {
        var md = new StringBuilder();
        md.Append("# Idrak report\n\n");
        md.Append($"{report["date"]}, idrak {report["tool"]}\n\n");
        var m = report["machine"]!;
        md.Append("## Machine\n\n");
        md.Append($"- OS: {m["os"]} ({m["architecture"]})\n- Processors: {m["processors"]}\n- Memory: {Units.Bytes(m["memoryBytes"]!.GetValue<long>())}\n- Runtime: {m["runtime"]}\n");
        md.Append($"- Libraries: {string.Join(", ", report["libraries"]!.AsObject().Select(p => $"{p.Key} {p.Value}"))}\n\n");

        md.Append("## Devices\n\n| Device | Backend | Name | Memory | CUs | Lanes | Width | Matrix units | In a plain run |\n|---|---|---|---|---|---|---|---|---|\n");
        foreach (var d in report["devices"]!.AsArray())
        {
            string name = d!["error"] is { } error ? $"cannot start: {error}" : (string)d["name"]!;
            md.Append($"| {d["device"]} | {d["backend"]} | {Cell(name)} | {Machine.Bytes((long?)d["memoryBytes"])} | {Machine.Number((int?)d["computeUnits"])} | " +
                      $"{Machine.Number((int?)d["subgroupSize"])} | {Machine.Number((int?)d["kernelWidth"])} | {((bool)d["matrixUnits"]! ? "yes" : "no")} | {((bool)d["listed"]! ? "yes" : "by name")} |\n");
        }

        md.Append("\n## Drivers\n\n");
        foreach (var (device, driver) in report["drivers"]!.AsObject())
        {
            md.Append($"- {device}: {driver}\n");
        }

        if (report["tests"] is JsonObject tests)
        {
            md.Append($"\n## Tests\n\n{tests["passed"]} passed, {tests["failed"]} failed\n\n| Device | Name | Passed | Failed |\n|---|---|---|---|\n");
            foreach (var (device, result) in tests["devices"]!.AsObject())
            {
                md.Append($"| {device} | {Cell((string?)result!["name"] ?? "")} | {result["passed"]} | {result["failed"]} |\n");
            }

            foreach (var (device, result) in tests["devices"]!.AsObject().Where(p => p.Value!["failed"]!.GetValue<int>() > 0))
            {
                md.Append($"\nFailed on {device}:\n\n");
                foreach (var failure in result!["failures"]!.AsArray())
                {
                    md.Append($"- {failure}\n");
                }
            }
        }

        if (report["benchmarks"] is JsonArray bench)
        {
            md.Append("\n## Benchmarks\n\n| Device | Benchmark | GFLOP/s |\n|---|---|---|\n");
            foreach (var b in bench)
            {
                md.Append($"| {b!["device"]} | {b["benchmark"]} | {((double)b["gflops"]!).ToString("F1", CultureInfo.InvariantCulture)} |\n");
            }
        }

        return md.ToString();
    }

    /// <summary>
    /// The README's tested-on rows for this machine: a row of the "Tested on" table per GPU (or one for the CPU only),
    /// and a row of the per-device HTML table for each device.
    /// </summary>
    internal static string Readme(JsonObject report)
    {
        var m = report["machine"]!;
        string date = (string)report["date"]!, library = (string)report["tool"]!;
        var devices = report["devices"]!.AsArray().Select(d => d!.AsObject()).Where(d => d["error"] is null).ToList();
        var tests = report["tests"] as JsonObject;
        var cpu = devices.FirstOrDefault(d => (string?)d["device"] == "cpu");
        string system = $"{m["os"]} ({((string)m["architecture"]!).ToLowerInvariant()}), {m["processors"]}-thread CPU" +
                        $"{(cpu?["subgroupSize"] is { } lanes ? $" with {lanes}-wide SIMD" : "")}, {m["runtime"]}";
        string Result(string? device)
        {
            if (tests is null)
            {
                return "not run (idrak report --tests)";
            }

            if (device is not null && tests["devices"]?[device] is JsonObject one)
            {
                int passed = (int)one["passed"]!, failed = (int)one["failed"]!;
                return $"{passed} of {passed + failed}";
            }

            int all = (int)tests["passed"]! + (int)tests["failed"]!;
            return $"{tests["passed"]} of {all}";
        }

        string Memory(JsonObject d) => d["memoryBytes"] is { } bytes ? $", {Units.Bytes((long)bytes)}" : "";
        var md = new StringBuilder("Tested on (README \"Tested on\" table):\n\n| Date | Library | GPU | Compute | Driver | System | Result |\n|---|---|---|---|---|---|---|\n");
        var gpus = devices.Where(d => (string?)d["device"] != "cpu").ToList();
        foreach (var d in gpus)
        {
            string units = d["computeUnits"] is { } cus ? $", {cus} compute units" : "";
            md.Append($"| {date} | {library} | {Cell((string)d["name"]!)}{Memory(d)}{units} | {d["backend"]} | {Cell((string?)d["driver"] ?? "–")} | {Cell(system)} | {Result(null)} |\n");
        }

        if (gpus.Count == 0)
        {
            md.Append($"| {date} | {library} | none (CPU only) | – | – | {Cell(system)} | {Result(null)} |\n");
        }

        md.Append("\nEach run in detail (README per-device table):\n\n");
        foreach (var d in devices)
        {
            string kind = (string?)d["kind"] ?? "–";
            md.Append($"    <tr><td>{date}</td><td><b>{Html((string)d["name"]!)}</b>{Html(Memory(d))}<br>{Html((string)d["backend"]!)}</td><td>{kind}</td><td>–</td>" +
                      $"<td>{((bool)d["matrixUnits"]! ? "matrix units" : "–")}</td><td>{Html(Result((string?)d["device"]))}</td></tr>\n");
        }

        return md.ToString();
    }

    private static string Html(string text) => System.Net.WebUtility.HtmlEncode(text);

    private static string Cell(string text) => text.Replace("|", "\\|", StringComparison.Ordinal);
}
