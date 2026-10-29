// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using Idrak.Diagnostics;
using Idrak.Gpu.Hip;

// dotnet run -c Release --project tests/Idrak.Tests -- --bench-hip [launch|copies|fill …]
//
// Every HIP device found, or those IDRAK_DEVICES names (hip:0, …), section by section (all of them, or those named), as
// --bench-vulkan is for Vulkan. Each time is the first call (cold: the code object written and loaded, the first launch,
// the first copy) apart from the warm ones that follow (rule 76). Sizes come from the memory the device reports, never a
// card's. Without a HIP device, or when the runtime refuses Idrak's GCN code object, one line says why and the sections
// that need no kernel still run.
//   launch   GCN's fill_f32: its code object written and loaded (cold, then again), one launch over one float (the
//            first, then queued, then a launch and a wait), its device time between two events (the profiler's), and one
//            launch over the large buffer
//   copies   upload and download of the large buffer (cold, then warm), and a copy on the device
//   fill     fill bandwidth over the large buffer: GCN's fill_f32 beside the runtime's own fill (its memset), each result
//            read back and checked
// Each later phase of plan 35 adds the sections of its kernels to HipBenchSections (rule 78).
// Settings (environment variables):
//   IDRAK_BENCH_HIP_MB        the large buffer in MB (default: a sixteenth of the memory the device has free, at most
//                             256 MB)
//   IDRAK_BENCH_HIP_REPEATS   the warm calls a row times (default 20; the queued tiny launches are a hundred times more)
internal abstract partial class DesktopTests
{
    private static readonly (string Name, Action<Device> Run)[] HipBenchGroup =
    [
        ("hip bench: --bench-hip [launch|copies|fill] without a HIP device (or with IDRAK_DEVICES naming none) prints why in one line and exits cleanly; an unknown section is refused naming the sections", HipBenchWithoutDevice),
    ];

    private static void HipBenchWithoutDevice(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(HipBenchWithoutDevice)))
        {
            return;
        }

        Check(HipBenchSections.Select(s => s.Name).SequenceEqual(["launch", "copies", "fill"]), "the sections: launch cost, copies, fill");
        (int Exit, string Out, string Err) Bench(string[] sections, string? devices)
        {
            var (savedOut, savedError) = (Console.Out, Console.Error);
            var (output, error) = (new StringWriter(), new StringWriter());
            int exit = -1;
            Console.SetOut(output);
            Console.SetError(error);
            try
            {
                WithEnvironment(new Dictionary<string, string?> { ["IDRAK_DEVICES"] = devices }, () => exit = BenchHip(sections));
            }
            finally
            {
                Console.SetOut(savedOut);
                Console.SetError(savedError);
            }

            return (exit, output.ToString(), error.ToString());
        }

        // IDRAK_DEVICES naming no HIP device: one line, with why there is none when the runtime found none at all.
        var none = Bench([], "cpu");
        string why = HipBackend.DeviceCount == 0 ? HipBackend.UnavailableReason : "IDRAK_DEVICES names none";
        Check(none.Exit == 0 && none.Out.Trim() == $"--bench-hip: no HIP device ({why}); nothing timed" && none.Err.Length == 0,
            $"IDRAK_DEVICES naming no HIP device: one line, exit 0: {none.Exit} '{none.Out}' '{none.Err}'");
        if (HipBackend.DeviceCount == 0)
        {
            var missing = Bench(["launch", "fill"], null);
            Check(missing.Exit == 0 && missing.Out.Trim() == $"--bench-hip: no HIP device ({HipBackend.UnavailableReason}); nothing timed",
                $"no HIP device here: the reason in one line, exit 0: {missing.Exit} '{missing.Out}'");
            Console.WriteLine($"    {missing.Out.Trim()}");
        }

        var unknown = Bench(["launch", "nope"], "cpu");
        Check(unknown.Exit == 2 && unknown.Out.Length == 0 && unknown.Err.Contains("unknown section 'nope' (sections: launch, copies, fill)", StringComparison.Ordinal),
            $"an unknown section refused before anything runs: {unknown.Exit} '{unknown.Err}'");
    }

    /// <summary>A section of <c>--bench-hip</c>: its name, what it times and its run.</summary>
    internal sealed record HipBenchSection(string Name, string Summary, Action<HipBench> Run);

    /// <summary>The sections in the order they run; each later phase adds its kernels' rows here.</summary>
    internal static readonly HipBenchSection[] HipBenchSections =
    [
        new("launch", "GCN's fill_f32: the code object's load (cold and warm) and a launch's cost, tiny and large", BenchHipLaunch),
        new("copies", "upload, download and a copy on the device over the large buffer (cold and warm)", BenchHipCopies),
        new("fill", "fill bandwidth: GCN's fill_f32 beside the runtime's memset, each result checked", BenchHipFill),
    ];

    internal static int BenchHip(string[] sections)
    {
        foreach (string section in sections)
        {
            if (!HipBenchSections.Any(s => s.Name.Equals(section, StringComparison.OrdinalIgnoreCase)))
            {
                Console.Error.WriteLine($"--bench-hip: unknown section '{section}' (sections: {string.Join(", ", HipBenchSections.Select(s => s.Name))})");
                return 2;
            }
        }

        List<Device> devices = Environment.GetEnvironmentVariable("IDRAK_DEVICES") is { Length: > 0 } chosen
            ? [.. chosen.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(Device.Parse).Where(d => d.Type == DeviceType.Hip)]
            : [.. Enumerable.Range(0, HipBackend.DeviceCount).Select(i => Device.Get("hip", i))];
        if (devices.Count == 0)
        {
            Console.WriteLine($"--bench-hip: no HIP device ({(HipBackend.DeviceCount == 0 ? HipBackend.UnavailableReason : "IDRAK_DEVICES names none")}); nothing timed");
            return 0;
        }

        var chosenSections = HipBenchSections.Where(s => sections.Length == 0 || sections.Contains(s.Name, StringComparer.OrdinalIgnoreCase)).ToList();
        foreach (var device in devices)
        {
            using var bench = new HipBench(device);
            bench.Header();
            foreach (var section in chosenSections)
            {
                Console.WriteLine($"  -- {section.Name}: {section.Summary}");
                section.Run(bench);
            }
        }

        return 0;
    }

    private static void BenchHipLaunch(HipBench b)
    {
        // The code object: written for the device's target and loaded, the first time in the process (cold), then again.
        long start = Stopwatch.GetTimestamp();
        HipBackend.GcnFunction? cold = b.TryLoadFill(out string? refused);
        if (cold is null)
        {
            b.Row("fill_f32's code object", $"refused: {refused}");
            return;
        }

        double coldMs = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        start = Stopwatch.GetTimestamp();
        using (b.Hip.LoadGcn(GcnKernels.Fill))
        {
            b.Row("fill_f32's code object, written and loaded", $"first {coldMs:F2} ms, again {Stopwatch.GetElapsedTime(start).TotalMilliseconds:F2} ms (format {cold.Format}, {cold.Target})");
        }

        b.Fill = cold;
        var one = b.Hip.Allocate(1, zeroed: true);
        try
        {
            b.Device.Synchronize();
            start = Stopwatch.GetTimestamp();
            b.Hip.FillGcn(cold, one, 1, 1f);
            b.Device.Synchronize();
            b.Row("fill_f32 over 1 float, first launch and wait", $"{Stopwatch.GetElapsedTime(start).TotalMicroseconds:F1} µs");
            b.Row("fill_f32 over 1 float, queued", $"{b.Micros(() => b.Hip.FillGcn(cold, one, 1, 2f), b.Repeats * 100):F2} µs a launch");
            b.Row("fill_f32 over 1 float, launch and wait (round trip)", $"{b.Micros(() => { b.Hip.FillGcn(cold, one, 1, 3f); b.Device.Synchronize(); }, b.Repeats * 10):F1} µs");
            if (b.Hip.ProfileUnavailableReason is { } why)
            {
                b.Row("fill_f32 over 1 float, on the device", $"not timed: {why}");
            }
            else
            {
                GpuProfiler.Start(b.Device);
                for (int i = 0; i < b.Repeats * 10; i++)
                {
                    b.Hip.FillGcn(cold, one, 1, i);
                }

                var timed = GpuProfiler.Stop(b.Device).FirstOrDefault(e => e.Name == "fill_f32");
                b.Row("fill_f32 over 1 float, on the device (between two events)", timed is null ? "no launch timed" : $"{timed.Milliseconds * 1000 / timed.Calls:F2} µs a launch");
            }
        }
        finally
        {
            one.Release();
        }

        var large = b.Hip.Allocate(b.LargeElements, zeroed: false);
        try
        {
            double us = b.Micros(() => b.Hip.FillGcn(cold, large, b.LargeElements, 1f), b.Repeats);
            b.Row($"fill_f32 over {b.LargeText}", $"{us:F1} µs a launch, {b.LargeBytes / (us * 1e3):F1} GB/s written");
        }
        finally
        {
            large.Release();
        }
    }

    private static void BenchHipCopies(HipBench b)
    {
        int n = b.LargeElements;
        var host = new float[n];
        for (int i = 0; i < n; i++)
        {
            host[i] = (i % 1000) * 0.001f;
        }

        var a = b.Hip.Allocate(n, zeroed: false);
        var c = b.Hip.Allocate(n, zeroed: false);
        try
        {
            long start = Stopwatch.GetTimestamp();
            b.Hip.Upload(host, a);                                                   // waits for its copy
            double coldUp = Stopwatch.GetElapsedTime(start).TotalMicroseconds;
            double up = b.Micros(() => b.Hip.Upload(host, a), b.Repeats);
            b.Row($"upload {b.LargeText}", $"cold {b.LargeBytes / (coldUp * 1e3):F2} GB/s, warm {b.LargeBytes / (up * 1e3):F2} GB/s ({up / 1000:F2} ms)");
            var back = new float[n];
            start = Stopwatch.GetTimestamp();
            b.Hip.Download(a, back);
            double coldDown = Stopwatch.GetElapsedTime(start).TotalMicroseconds;
            double down = b.Micros(() => b.Hip.Download(a, back), b.Repeats);
            b.Row($"download {b.LargeText}", $"cold {b.LargeBytes / (coldDown * 1e3):F2} GB/s, warm {b.LargeBytes / (down * 1e3):F2} GB/s ({down / 1000:F2} ms)");
            double copy = b.Micros(() => b.Hip.Copy(a, c, n), b.Repeats);
            b.Row($"copy on the device, {b.LargeText}", $"{2.0 * b.LargeBytes / (copy * 1e3):F1} GB/s read and written ({copy / 1000:F2} ms)");
            b.Hip.Download(c, back);
            b.Row("  (the copies read back)", back.AsSpan().SequenceEqual(host) ? "equal to what was uploaded" : "DIFFERENT from what was uploaded");
        }
        finally
        {
            a.Release();
            c.Release();
        }
    }

    private static void BenchHipFill(HipBench b)
    {
        int n = b.LargeElements;
        var large = b.Hip.Allocate(n, zeroed: false);
        var back = new float[n];
        try
        {
            string Checked(float value)
            {
                b.Hip.Download(large, back);
                int wrong = back.AsSpan().IndexOfAnyExcept(value);
                return wrong < 0 ? "checked" : string.Create(CultureInfo.InvariantCulture, $"WRONG at {wrong}: {back[wrong]} for {value}");
            }

            double memset = b.Micros(() => b.Hip.FillKernel(large, n, 5f), b.Repeats);
            b.Row($"the runtime's fill (memset), {b.LargeText}", $"{b.LargeBytes / (memset * 1e3):F1} GB/s ({memset:F1} µs), {Checked(5f)}");
            string? refused = null;
            var fill = b.Fill ?? b.TryLoadFill(out refused);
            if (fill is null)
            {
                b.Row($"fill_f32, {b.LargeText}", $"not run: the code object was refused ({refused})");
                return;
            }

            b.Fill = fill;
            b.Device.Synchronize();
            long start = Stopwatch.GetTimestamp();
            b.Hip.FillGcn(fill, large, n, 6f);
            b.Device.Synchronize();
            double cold = Stopwatch.GetElapsedTime(start).TotalMicroseconds;
            double warm = b.Micros(() => b.Hip.FillGcn(fill, large, n, 7f), b.Repeats);
            b.Row($"fill_f32, {b.LargeText}", $"cold {cold:F1} µs, warm {b.LargeBytes / (warm * 1e3):F1} GB/s ({warm:F1} µs), {Checked(7f)}");
        }
        finally
        {
            large.Release();
        }
    }
}

/// <summary>What a <c>--bench-hip</c> section works with: the device, its backend, the large buffer's size, fill_f32 once loaded, and the timing helpers.</summary>
internal sealed class HipBench : IDisposable
{
    internal HipBench(Device device)
    {
        Device = device;
        Hip = (HipBackend)device.Backend;
        Repeats = Environment.GetEnvironmentVariable("IDRAK_BENCH_HIP_REPEATS") is { Length: > 0 } repeats ? int.Parse(repeats, CultureInfo.InvariantCulture) : 20;
        ArgumentOutOfRangeException.ThrowIfLessThan(Repeats, 1, "IDRAK_BENCH_HIP_REPEATS");
        long bytes = Environment.GetEnvironmentVariable("IDRAK_BENCH_HIP_MB") is { Length: > 0 } mb
            ? long.Parse(mb, CultureInfo.InvariantCulture) << 20
            : Math.Min(256L << 20, ((IMemoryReport)Hip).AvailableMemory() is { } available ? available / 16 : 64L << 20);
        long elements = Math.Clamp(bytes / sizeof(float), GcnKernels.FillBlock, Math.Min(GcnKernels.FillMostElements, int.MaxValue / 2));
        LargeElements = (int)(elements / GcnKernels.FillBlock * GcnKernels.FillBlock);
    }

    internal Device Device { get; }

    internal HipBackend Hip { get; }

    /// <summary>The warm calls a row times (IDRAK_BENCH_HIP_REPEATS).</summary>
    internal int Repeats { get; }

    /// <summary>The large buffer's floats (IDRAK_BENCH_HIP_MB, else from the memory the device has free).</summary>
    internal int LargeElements { get; }

    internal double LargeBytes => LargeElements * 4.0;

    internal string LargeText => string.Create(CultureInfo.InvariantCulture, $"{LargeBytes / (1 << 20):F0} MB");

    /// <summary>fill_f32 once a section loaded it (null before, and when the runtime refused it).</summary>
    internal HipBackend.GcnFunction? Fill { get; set; }

    internal void Header()
    {
        Console.WriteLine($"== {Device}: {Hip.Name}");
        Console.WriteLine($"   HIP runtime {HipBackend.FormatVersion(Hip.RuntimeVersion)}, driver {Hip.DriverVersion}, from {HipRuntime.LoadedFrom}; target {Hip.DeviceLimits.Architecture}");
        Console.WriteLine($"   operations: {Hip.KernelsUnavailableReason}");
        Console.WriteLine($"   kernel timing: {Hip.ProfileUnavailableReason ?? "each launch between two events while profiling"}");
        Console.WriteLine($"   large buffer {LargeText}, {Repeats} warm calls a row");
    }

    /// <summary>fill_f32 written and loaded for this device, or null with the runtime's refusal.</summary>
    internal HipBackend.GcnFunction? TryLoadFill(out string? refused)
    {
        try
        {
            refused = null;
            return Hip.LoadGcn(GcnKernels.Fill);
        }
        catch (HipException e)
        {
            refused = e.Message;
            return null;
        }
    }

    internal void Row(string name, string value) => Console.WriteLine($"  {name,-60} {value}");

    /// <summary>The average time of <paramref name="run"/> in µs over <paramref name="repeats"/> calls, after one warm-up call, the device drained before and after.</summary>
    internal double Micros(Action run, int repeats)
    {
        run();
        Device.Synchronize();
        long start = Stopwatch.GetTimestamp();
        for (int i = 0; i < repeats; i++)
        {
            run();
        }

        Device.Synchronize();
        return Stopwatch.GetElapsedTime(start).TotalMicroseconds / repeats;
    }

    public void Dispose() => Fill?.Dispose();
}
