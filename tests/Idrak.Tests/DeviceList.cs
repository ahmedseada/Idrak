// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Backends;

// --list-devices: every device the library can run on (the CPU, each CUDA and Vulkan GPU, test backends), whether a
// plain test run includes it, and how to run the tests on one of them.
internal static partial class Tests
{
    internal static int ListDevices()
    {
        var rows = new List<(string Device, string Kind, string Name, string Tested, string Note)>
        {
            ("cpu", "CPU", Describe(Device.Cpu), "yes", Device.Default == Device.Cpu ? "default device" : ""),
        };

        foreach (var provider in DeviceProviders.All)
        {
            if (provider.Count == 0)
            {
                rows.Add(($"{provider.Kind}:-", provider.Display, $"none found: {provider.UnavailableReason}", "-", ""));
                continue;
            }

            for (int i = 0; i < provider.Count; i++)
            {
                var device = Device.Get(provider.Kind, i);
                string note = device == Device.Default ? "default device" : provider.Note(i) ?? "";
                rows.Add((device.ToString(), provider.Display, Describe(device), provider.Listed(i) ? "yes" : "by name", note));
            }
        }

        string[] header = ["Device", "Backend", "Name", "In a plain run", "Note"];
        int[] widths =
        [
            Math.Max(header[0].Length, rows.Max(r => r.Device.Length)),
            Math.Max(header[1].Length, rows.Max(r => r.Kind.Length)),
            Math.Max(header[2].Length, rows.Max(r => r.Name.Length)),
            header[3].Length,
            Math.Max(header[4].Length, rows.Max(r => r.Note.Length)),
        ];
        string Line(string a, string b, string c, string d, string e) =>
            $"| {a.PadRight(widths[0])} | {b.PadRight(widths[1])} | {c.PadRight(widths[2])} | {d.PadRight(widths[3])} | {e.PadRight(widths[4])} |";
        string rule = "|" + string.Join("|", widths.Select(w => new string('-', w + 2))) + "|";

        Console.WriteLine(Line(header[0], header[1], header[2], header[3], header[4]));
        Console.WriteLine(rule);
        foreach (var r in rows)
        {
            Console.WriteLine(Line(r.Device, r.Kind, r.Name, r.Tested, r.Note));
        }

        // The example: a Vulkan GPU in plain runs (the newest backend), else any GPU, else the CPU.
        var runnable = rows.Where(r => !r.Device.EndsWith(":-", StringComparison.Ordinal) && !r.Name.StartsWith("cannot", StringComparison.Ordinal)).ToList();
        string example = runnable.FirstOrDefault(r => r.Device.StartsWith("vulkan", StringComparison.Ordinal) && r.Tested == "yes").Device
                         ?? runnable.FirstOrDefault(r => r.Device.StartsWith("vulkan", StringComparison.Ordinal) || r.Device.StartsWith("cuda", StringComparison.Ordinal)).Device
                         ?? "cpu";
        bool windows = OperatingSystem.IsWindows();
        Console.WriteLine();
        Console.WriteLine("A plain run tests every device marked \"yes\":");
        Console.WriteLine("  dotnet run -c Release --project tests/Idrak.Tests");
        Console.WriteLine($"Only some devices (any of the names above, comma-separated), e.g. {example}:");
        Console.WriteLine(windows
            ? $"  $env:IDRAK_DEVICES=\"{example}\"; dotnet run -c Release --project tests/Idrak.Tests; Remove-Item Env:IDRAK_DEVICES"
            : $"  IDRAK_DEVICES={example} dotnet run -c Release --project tests/Idrak.Tests");
        Console.WriteLine(windows
            ? $"  $env:IDRAK_DEVICES=\"cpu,{example}\"; dotnet run -c Release --project tests/Idrak.Tests; Remove-Item Env:IDRAK_DEVICES"
            : $"  IDRAK_DEVICES=cpu,{example} dotnet run -c Release --project tests/Idrak.Tests");
        Console.WriteLine("Only some tests: add IDRAK_FILTER=<text in the test names> the same way.");
        return 0;
    }

    // The device's name; starting its backend can fail on a broken driver, which the listing reports instead.
    private static string Describe(Device device)
    {
        try
        {
            return device.Name;
        }
        catch (Exception e)
        {
            return $"cannot start: {e.Message}";
        }
    }
}
