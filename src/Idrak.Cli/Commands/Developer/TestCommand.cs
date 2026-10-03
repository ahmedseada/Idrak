// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;

namespace Idrak.Cli.Commands.Developer;

/// <summary>
/// <c>idrak test [--filter TEXT] [--device NAME]</c>: runs the library's test runner (tests/Idrak.Tests) from the source
/// checkout around the current folder, as the README gives it: <c>dotnet run -c Release --project tests/Idrak.Tests</c>
/// with <c>IDRAK_FILTER</c> and <c>IDRAK_DEVICES</c> set from the options.
/// </summary>
internal sealed class TestCommand : Command
{
    /// <summary>The test runner's project, relative to the repository root.</summary>
    internal const string RunnerProject = "tests/Idrak.Tests/Idrak.Tests.csproj";

    /// <summary>Where the search for the repository starts (the current folder unless a test sets it).</summary>
    internal static string? StartFolder { get; set; }

    public override string Name => "test";

    public override string Summary => "Runs the library's tests from a source checkout";

    public override string Usage =>
        "[--filter TEXT] [--device NAME]\n\n" +
        "Runs tests/Idrak.Tests of the Idrak source checkout that holds the current folder (dotnet run -c Release).\n\n" +
        "Options:\n" +
        "      --filter TEXT   only tests whose name contains TEXT (IDRAK_FILTER)\n" +
        "  -d, --device NAME   only this device, or several separated by commas (IDRAK_DEVICES); default every device found\n\n" +
        "Examples:\n" +
        "  idrak test\n" +
        "  idrak test --filter \"cli dev\" -d cpu\n" +
        "  idrak test -d vulkan:0 -j";

    public override IReadOnlyCollection<string> ValueOptions => ["--filter"];

    public override int Run(CommandContext context)
    {
        if (context.Positional.Count > 0)
        {
            throw new UsageException($"Unexpected argument '{context.Positional[0]}'; pass test names with --filter.");
        }

        string start = StartFolder ?? Environment.CurrentDirectory;
        if (FindRepositoryRoot(start) is not { } root)
        {
            context.Error($"{start} is not inside an Idrak source checkout (no Idrak.slnx with {RunnerProject} above it). " +
                "Run idrak test from a clone: git clone https://github.com/ahmedseada/Idrak && cd Idrak && idrak test");
            return ExitCodes.Failed;
        }

        string? filter = context.Option("--filter");
        string? devices = context.Option("--device");
        var info = new ProcessStartInfo("dotnet") { WorkingDirectory = root, RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false };
        foreach (string arg in new[] { "run", "-c", "Release", "--project", RunnerProject })
        {
            info.ArgumentList.Add(arg);
        }

        if (filter is not null)
        {
            info.Environment["IDRAK_FILTER"] = filter;
        }

        if (devices is not null)
        {
            info.Environment["IDRAK_DEVICES"] = devices;
        }

        context.Detail($"in {root}: {(filter is null ? "" : $"IDRAK_FILTER=\"{filter}\" ")}{(devices is null ? "" : $"IDRAK_DEVICES={devices} ")}dotnet {string.Join(' ', info.ArgumentList)}");
        int passed = 0, failed = 0;
        var failures = new List<string>();
        using var process = Process.Start(info) ?? throw new InvalidOperationException("Could not start dotnet; is the .NET SDK installed and on PATH?");
        process.ErrorDataReceived += (_, e) =>
        {
            if (e.Data is { } line)
            {
                context.ErrorOutput.WriteLine(line);
            }
        };
        process.BeginErrorReadLine();
        while (process.StandardOutput.ReadLine() is { } line)
        {
            string trimmed = line.TrimStart();
            if (trimmed.StartsWith("PASS ", StringComparison.Ordinal))
            {
                passed++;
            }
            else if (trimmed.StartsWith("FAIL ", StringComparison.Ordinal) || trimmed.StartsWith("HANG ", StringComparison.Ordinal))
            {
                failed++;
                failures.Add(trimmed);
            }

            context.Write(line);
        }

        process.WaitForExit();
        context.WriteJson(new JsonObject
        {
            ["root"] = root,
            ["filter"] = filter,
            ["devices"] = devices,
            ["passed"] = passed,
            ["failed"] = failed,
            ["failures"] = new JsonArray([.. failures.Select(f => (JsonNode)f)]),
            ["exitCode"] = process.ExitCode,
        });
        return process.ExitCode == 0 ? ExitCodes.Ok : ExitCodes.Failed;
    }

    /// <summary>The nearest folder at or above <paramref name="start"/> holding Idrak.slnx and the test runner, or null.</summary>
    internal static string? FindRepositoryRoot(string start)
    {
        for (var folder = new DirectoryInfo(Path.GetFullPath(start)); folder is not null; folder = folder.Parent)
        {
            if (File.Exists(Path.Combine(folder.FullName, "Idrak.slnx")) && File.Exists(Path.Combine(folder.FullName, RunnerProject)))
            {
                return folder.FullName;
            }
        }

        return null;
    }
}
