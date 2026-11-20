// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using Idrak.Adad.Devices.Cuda;
using Idrak.Adad.Measure;

namespace Idrak.Adad.Tests;

/// <summary>A test's checks: each failed check is kept with its message.</summary>
internal sealed class Checks
{
    public List<string> Failures { get; } = [];

    public int Count { get; private set; }

    public void Check(bool condition, string message)
    {
        Count++;
        if (!condition && Failures.Count < 20)
        {
            Failures.Add(message);
        }
    }
}

/// <summary>What the CUDA tests share: the device, its product (lanes prepared once) and its tuner.</summary>
internal sealed class Gpu : IDisposable
{
    private Gpu(CudaDevice device)
    {
        Device = device;
        Tuner = new Tuner(device);
        Product = new SuperProduct(device, Tuner);
    }

    public CudaDevice Device { get; }

    public Tuner Tuner { get; }

    public SuperProduct Product { get; }

    private static Gpu? s_gpu;
    private static string? s_reason;

    /// <summary>The device, or null with the reason (no NVIDIA driver, no device).</summary>
    public static Gpu? Get(out string reason)
    {
        if (s_gpu is null && s_reason is null)
        {
            var device = CudaDevice.TryOpen(0, out string why);
            if (device is null)
            {
                s_reason = why;
            }
            else
            {
                s_gpu = new Gpu(device);
            }
        }

        reason = s_reason ?? "";
        return s_gpu;
    }

    public void Dispose() => Device.Dispose();
}

// Adad's tests: a console runner (no packages). `--filter a,b` (or ADAD_FILTER) picks groups; `--failures FILE` writes the
// failed checks.
internal static class Program
{
    private static readonly (string Group, string Name, Action<Checks> Run)[] All =
    [
        ("types", "q4_k's encoder gives the source's bytes (the seeded rows' SHA-256 recorded from the source)", TypeTests.EncoderKeepsTheSourceBytes),
        ("types", "q4_k's decoder gives the source's floats bit for bit (seeded and random valid blocks)", TypeTests.DecoderMatchesTheSource),
        ("types", "q4_k's description id is FNV-1a over its code, masked to 0x7FFFFFFF, and its kernels' name part the source's", TypeTests.DescriptionId),
        ("ptx", "the control's kernels are the source's texts (native kernel's 9 shapes, x pack, q4_k widening, splits' sum)", PtxTests.ControlKernelsAreTheSource),
        ("control", "the control lane gives the source's output bit for bit (its first shape unsplit, and the source's measured shape and split)", ProductTests.ControlMatchesTheSource),
        ("a", "A gives the control's output bit for bit at every shape and split", ProductTests.DecodeMatchesControl),
        ("b", "B's x rounding matches its CPU model bit for bit", ProductTests.QuantizeMatchesModel),
        ("b", "B gives its CPU model's output bit for bit and the float reference within the stated bound", ProductTests.Int8MatchesModel),
        ("measure", "the product's measured choice is a candidate, stored by the description's id and kept for the next call", ProductTests.MeasuredChoiceIsStored),
    ];

    private static int Main(string[] args)
    {
        string? filter = Environment.GetEnvironmentVariable("ADAD_FILTER");
        string? failuresFile = null;
        for (int i = 0; i + 1 < args.Length; i += 2)
        {
            if (args[i] == "--filter")
            {
                filter = args[i + 1];
            }
            else if (args[i] == "--failures")
            {
                failuresFile = args[i + 1];
            }
        }

        var groups = filter?.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var failed = new List<string>();
        int passed = 0, ran = 0;
        var clock = Stopwatch.StartNew();
        foreach (var (group, name, run) in All)
        {
            if (groups is not null && !groups.Contains(group, StringComparer.OrdinalIgnoreCase))
            {
                continue;
            }

            ran++;
            var checks = new Checks();
            var started = Stopwatch.StartNew();
            Console.Write($"[{ran}] {group}: {name} ... ");
            try
            {
                run(checks);
            }
            catch (Exception e)
            {
                checks.Failures.Add($"threw {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
            }

            string time = string.Create(CultureInfo.InvariantCulture, $"{started.Elapsed.TotalSeconds:F1} s, {checks.Count} checks");
            if (checks.Failures.Count == 0)
            {
                passed++;
                Console.WriteLine($"PASS ({time})");
            }
            else
            {
                Console.WriteLine($"FAIL ({time})");
                foreach (string f in checks.Failures)
                {
                    Console.WriteLine("    " + f);
                }

                failed.Add($"FAIL {group}: {name}\n" + string.Join("\n", checks.Failures.Select(f => "    " + f)));
            }
        }

        string summary = string.Create(CultureInfo.InvariantCulture, $"{passed} of {ran} passed, {ran - passed} failed ({clock.Elapsed.TotalSeconds:F1} s)");
        Console.WriteLine(summary);
        if (failuresFile is not null)
        {
            File.WriteAllText(failuresFile, string.Join("\n", failed.Append(summary)) + "\n");
        }

        Gpu.Get(out _)?.Dispose();
        return failed.Count == 0 ? 0 : 1;
    }
}
