// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Idrak.Maui's tests on the CPU, on its net10.0 target (no phone: in-memory stores, a manual memory-pressure signal, a
// queued dispatcher standing in for the UI thread). A plain runner like the repository's: IDRAK_FILTER runs the tests whose
// name contains any of its "|"-separated parts (ignoring case), IDRAK_DEVICES the device (its first entry; the CPU by
// default), --failures-only shows only the FAIL lines and the summary, --failures-file PATH writes them to a file too.
//
//   IDRAK_DEVICES=cpu dotnet run -c Release --project tests/Idrak.Maui.Tests
//   IDRAK_DEVICES=cpu IDRAK_FILTER="maui progress|maui settings" dotnet run -c Release --project tests/Idrak.Maui.Tests
//   IDRAK_DEVICES=cpu dotnet run -c Release --no-build --project tests/Idrak.Maui.Tests -- --failures-only --failures-file test-output/maui-fails.txt
//   IDRAK_UPDATE_API=1 IDRAK_FILTER="public API" …   rewrites api/Idrak.Maui.txt after an intended change

using System.Diagnostics;
using Idrak.Maui.Tests;

string device = Environment.GetEnvironmentVariable("IDRAK_DEVICES") is { Length: > 0 } devices ? devices.Split(',')[0].Trim() : "cpu";
string[]? filters = Environment.GetEnvironmentVariable("IDRAK_FILTER") is { Length: > 0 } filter
    ? filter.Split('|', StringSplitOptions.RemoveEmptyEntries)
    : null;
bool failuresOnly = args.Contains("--failures-only");
string? failuresPath = Array.IndexOf(args, "--failures-file") is var at and >= 0
    ? (at + 1 < args.Length ? args[at + 1] : throw new InvalidOperationException("--failures-file needs a path."))
    : null;
if (failuresPath is not null)
{
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(failuresPath))!);
}

using var failuresFile = failuresPath is null ? null : new StreamWriter(failuresPath, append: false, new System.Text.UTF8Encoding(false)) { AutoFlush = true };
var selected = MauiTests.All.Where(t => filters is null || filters.Any(f => t.Name.Contains(f, StringComparison.OrdinalIgnoreCase))).ToArray();
string header = $"Idrak.Maui tests on {device}: {selected.Length} of {MauiTests.All.Length}{(filters is null ? "" : $" (IDRAK_FILTER={string.Join('|', filters)})")}";
Console.WriteLine(header);
failuresFile?.WriteLine(header);
if (device != "cpu")
{
    Console.WriteLine($"  (these tests run on the CPU; {device} is not used)");
}

int failed = 0;
var run = Stopwatch.StartNew();
foreach (var (name, test) in selected)
{
    var clock = Stopwatch.StartNew();
    try
    {
        test();
        if (!failuresOnly)
        {
            Console.WriteLine($"  PASS {name} ({clock.Elapsed.TotalSeconds:F1} s)");
        }
    }
    catch (Exception ex)
    {
        failed++;
        string line = $"  FAIL {name}: {(ex is MauiTests.CheckException ? ex.Message : ex.ToString())}";
        Console.WriteLine(line);
        failuresFile?.WriteLine(line);
    }
}

string summary = $"{selected.Length - failed} passed, {failed} failed ({run.Elapsed.TotalSeconds:F1} s, device {device})";
Console.WriteLine(summary);
failuresFile?.WriteLine(summary);
return failed == 0 ? 0 : 1;
