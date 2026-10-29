// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using Idrak.Diagnostics;
using Idrak.Gpu.Hip;
using GcnWeights = Idrak.Gpu.Hip.GcnKernels.PackedWeights;

// dotnet run -c Release --project tests/Idrak.Tests -- --bench-hip [launch|copies|fill|rows|pointwise|products|attention|shapes|decoding|training …]
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
//   rows     the kernel set (every GCN kernel assembled into one code object: cold, into an empty cache, then warm, read
//            from it), then each row operation on its kernel over rows of 1,024 and 8,192 columns (as many rows as the
//            large buffer holds, at most 8,192): the first call apart from the warm ones, GB/s and GFLOP/s from
//            GcnKernels.Rows.cs's table (HipBackend.Cost), and a check that none took the host fallback
//   pointwise each element-wise operation on its kernel over the large buffer: the first call apart from the warm ones,
//            GB/s and GFLOP/s from GcnKernels.Pointwise.cs's table, none through the host fallback
//   products the float32 product (nn, nt, tn, tt; a batch; with a bias), the few-row float32 products, the packed few-row
//            products (int8, int4, bfloat16, float16; 1, 4 and 8 rows; the leftover-8.10 shapes int8 1 x 3072 -> 1024 and
//            int4 1 x 1024 -> 32000), more int8 rows, the expansions and bfloat16 packing: the first call apart from the
//            warm ones, GB/s and GFLOP/s from GcnKernels.Products.cs's table (HipBackend.ProductCost, Cost), the k splits
//            each packed product took (IDRAK_HIP_GEMV_SPLITS forces them), none through the host fallback
//   attention decoding over float32, bfloat16 and int8 caches (one row a head, the cache's positions split), the tiled
//            form with the log-sum-exp, attention over causal key ranges with grouped heads, and the int8 cache's scores
//            and weighted values (BenchHip.Attention.cs): cold apart from warm, GB/s and GFLOP/s from
//            GcnKernels.Attention.cs's table, none through the host fallback
//   shapes   the cache writes (float32, bfloat16, int8), the decoder mask, the gathers, a permute and the adding strided
//            copy over the large buffer (BenchHip.Attention.cs): cold apart from warm, none through the host fallback
//   decoding the fused packed products (gated, of one input, a gate/up pair, with the residual addition and norms; int8,
//            int4, bfloat16; 1 and 8 rows), attention over rows of different lengths and over packed sequences, the rows
//            of decoding (arg-max, class match, the residual add with its norm, rotary embedding alone and after the norm,
//            penalties, one-hot, the history) and the element-wise operations with parameters (BenchHip.Decoding.cs):
//            cold apart from warm, GB/s and GFLOP/s from GcnKernels.Decoding.cs's and .Pointwise.cs's tables, ending with
//            the host fallbacks the section took (none expected)
//   training the backward passes (unary, gated, extremes, softmax, RMS norm), SGD, Adam, fused AdamW, dropout, the sums and
//            the tiled attention gradients (BenchHip.Training.cs): cold apart from warm, GB/s and GFLOP/s from
//            GcnKernels.Training.cs's table, none through the host fallback
// Each later phase of plan 35 adds the sections of its kernels to HipBenchSections (rule 78).
// Settings (environment variables):
//   IDRAK_BENCH_HIP_MB        the large buffer in MB (default: a sixteenth of the memory the device has free, at most
//                             256 MB)
//   IDRAK_BENCH_HIP_REPEATS   the warm calls a row times (default 20; the queued tiny launches are a hundred times more)
internal abstract partial class DesktopTests
{
    private static readonly (string Name, Action<Device> Run)[] HipBenchGroup =
    [
        ("hip bench: --bench-hip [launch|copies|fill|rows|pointwise|products|attention|shapes|decoding|training] without a HIP device (or with IDRAK_DEVICES naming none) prints why in one line and exits cleanly; an unknown section is refused naming the sections", HipBenchWithoutDevice),
    ];

    private static void HipBenchWithoutDevice(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(HipBenchWithoutDevice)))
        {
            return;
        }

        Check(HipBenchSections.Select(s => s.Name).SequenceEqual(["launch", "copies", "fill", "rows", "pointwise", "products", "attention", "shapes", "decoding", "training"]),
            "the sections: launch cost, copies, fill, rows, element-wise, products, attention, shapes, decoding, training");
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
        Check(unknown.Exit == 2 && unknown.Out.Length == 0 && unknown.Err.Contains("unknown section 'nope' (sections: launch, copies, fill, rows, pointwise, products, attention, shapes, decoding, training)", StringComparison.Ordinal),
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
        new("rows", "the kernel set (assembled cold, then from the cache warm) and each row operation on its GCN kernel (cold and warm, GB/s, GFLOP/s)", BenchHipRows),
        new("pointwise", "each element-wise operation on its GCN kernel over the large buffer (cold and warm, GB/s, GFLOP/s)", BenchHipPointwise),
        new("products", "the float32, few-row and packed few-row products, the expansions and bfloat16 packing on their GCN kernels (cold and warm, GB/s, GFLOP/s)", BenchHipProducts),
        new("attention", "decoding over float32, bfloat16 and int8 caches, the tiled form and attention over key ranges on the GCN kernels (cold and warm, GB/s, GFLOP/s)", BenchHipAttention),
        new("shapes", "the cache writes, the decoder mask, the gathers, a permute and the adding strided copy on the GCN kernels (cold and warm, GB/s)", BenchHipShapes),
        new("decoding", "the fused packed products, attention over rows and packed sequences, the rows of decoding and the element-wise operations with parameters on the GCN kernels (cold and warm, GB/s, GFLOP/s)", BenchHipDecoding),
        new("training", "the backward passes, the optimizers, the sums and the attention gradients on the GCN kernels (cold and warm, GB/s, GFLOP/s)", BenchHipTraining),
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

internal abstract partial class DesktopTests
{
    // The kernel set timed cold (assembled into an empty cache) and warm (read back from it), apart from the live one, then
    // the live one loaded: false with a row saying why when the kernels are unavailable.
    private static bool BenchKernelSet(HipBench b)
    {
        if (b.SetTimed)
        {
            return b.Hip.KernelsUnavailableReason is null;
        }

        b.SetTimed = true;
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-bench-hip-{Environment.ProcessId}-{Guid.NewGuid():N}");
        try
        {
            var cache = new HipKernelCache(folder);
            for (int pass = 0; pass < 2; pass++)
            {
                long start = Stopwatch.GetTimestamp();
                using var set = b.Hip.LoadKernelSet(cache, out string? why);
                double ms = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                if (set is null)
                {
                    b.Row("the kernel set", $"unavailable: {why}");
                    return false;
                }

                b.Row(pass == 0 ? $"the kernel set ({set.Functions.Count} kernels), assembled and loaded (cold)" : "the kernel set, read from the cache and loaded (warm)",
                    pass == 0 ? $"{ms:F1} ms (assembly {set.Assembly.TotalMilliseconds:F1} ms), {set.Bytes:N0} bytes, format {set.Format}, {set.Target}"
                        : $"{ms:F1} ms{(set.FromCache ? "" : " (NOT from the cache)")}");
            }
        }
        catch (HipException e)
        {
            b.Row("the kernel set", $"refused: {e.Message}");
            return false;
        }
        finally
        {
            if (Directory.Exists(folder))
            {
                Directory.Delete(folder, recursive: true);
            }
        }

        if (b.Hip.KernelsUnavailableReason is { } off)
        {
            b.Row("the device's kernels", $"unavailable: {off}");
            return false;
        }

        return true;
    }

    // One operation timed: its first call (cold, with a wait) and the warm average, its rate from flops and bytes per
    // element over `elements`.
    private static void BenchOperation(HipBench b, string name, long elements, (double Flops, double Bytes) cost, Action run)
    {
        b.Device.Synchronize();
        long start = Stopwatch.GetTimestamp();
        run();
        b.Device.Synchronize();
        double cold = Stopwatch.GetElapsedTime(start).TotalMicroseconds;
        double warm = b.Micros(run, b.Repeats);
        b.Row(name, string.Create(CultureInfo.InvariantCulture,
            $"cold {cold:F1} µs, warm {warm:F1} µs: {cost.Bytes * elements / (warm * 1e3):F1} GB/s, {cost.Flops * elements / (warm * 1e3):F1} GFLOP/s"));
    }

    private static void BenchHipRows(HipBench b)
    {
        if (!BenchKernelSet(b))
        {
            return;
        }

        var hip = b.Hip;
        var random = new Random(5);
        long hostBefore = Idrak.Abstraction.Operations.Kernels.HostCalls(hip);
        foreach (int cols in new[] { 1024, 8192 })
        {
            int rows = (int)Math.Clamp(b.LargeElements / (4L * cols), 1, 8192);
            long n = (long)rows * cols;
            const int K = 8;
            var owned = new List<Storage>();
            Storage On(int count, float low = -2f, float high = 2f)
            {
                var values = new float[count];
                for (int i = 0; i < count; i++)
                {
                    values[i] = low + random.NextSingle() * (high - low);
                }

                var s = hip.Allocate(count, zeroed: false);
                owned.Add(s);
                hip.Upload(values, s);
                return s;
            }

            try
            {
                Storage x = On((int)n), y = On((int)n), dy = On((int)n), dx = On((int)n), gain = On(cols), beta = On(cols), mask = On(cols, -1f, 0f);
                Storage inv = On(rows), stats = On(2 * rows), dgamma = On(cols), dbeta = On(cols), picks = On(rows * K), columns = On(rows * K);
                string shape = string.Create(CultureInfo.InvariantCulture, $"{rows}x{cols}");
                BenchOperation(b, $"RMS norm {shape}", n, HipBackend.Cost("rms_norm_f32"), () => hip.RmsNorm(x, y, inv, rows, cols, 1e-6f));
                BenchOperation(b, $"RMS norm with gain {shape}", n, HipBackend.Cost("rms_norm_affine_f32"), () => hip.RmsNormAffine(x, gain, y, rows, cols, 1e-6f, 1f));
                BenchOperation(b, $"layer norm {shape}", n, HipBackend.Cost("layernorm_fused_f32"), () => hip.LayerNormFused(x, gain, beta, y, rows, cols, 1e-5f));
                BenchOperation(b, $"layer norm, training {shape}", n, HipBackend.Cost("layernorm_train_f32"), () => hip.LayerNormTrain(x, gain, beta, y, stats, rows, cols, 1e-5f));
                var (bwd, parameters) = (HipBackend.Cost("layernorm_bwd_f32"), HipBackend.Cost("layernorm_bwd_params_f32"));
                BenchOperation(b, $"layer norm gradients {shape}", n, (bwd.Flops + parameters.Flops, bwd.Bytes + parameters.Bytes),
                    () => hip.LayerNormBackward(x, gain, dy, stats, dx, dgamma, dbeta, rows, cols));
                BenchOperation(b, $"softmax {shape}", n, HipBackend.Cost("softmax_f32"), () => hip.Softmax(x, y, rows, cols, false));
                BenchOperation(b, $"log-softmax {shape}", n, HipBackend.Cost("softmax_f32", variant: true), () => hip.Softmax(x, y, rows, cols, true));
                BenchOperation(b, $"masked softmax {shape}", n, HipBackend.Cost("scale_mask_softmax_f32"), () => hip.ScaleMaskSoftmax(x, mask, y, rows, cols, 1, 0.125f));
                BenchOperation(b, $"log-softmax top-{K} {shape}", n, HipBackend.Cost("log_softmax_topk_f32", K), () => hip.LogSoftmaxTopK(x, picks, columns, rows, cols, K));
            }
            finally
            {
                foreach (var s in owned)
                {
                    s.Release();
                }
            }
        }

        long host = Idrak.Abstraction.Operations.Kernels.HostCalls(hip) - hostBefore;
        b.Row("  (host fallbacks during the section)", host == 0 ? "none: every operation ran on its kernel" : $"{host} (SHOULD BE NONE)");
    }

    private static void BenchHipPointwise(HipBench b)
    {
        if (!BenchKernelSet(b))
        {
            return;
        }

        var hip = b.Hip;
        int n = b.LargeElements;
        var host = new float[n];
        for (int i = 0; i < n; i++)
        {
            host[i] = 0.25f + (i % 997) * 0.003f;                                  // positive: log and sqrt stay finite
        }

        var x = hip.Allocate(n, zeroed: false);
        var y = hip.Allocate(n, zeroed: false);
        var z = hip.Allocate(n, zeroed: false);
        try
        {
            hip.Upload(host, x);
            hip.Upload(host, y);
            long before = Idrak.Abstraction.Operations.Kernels.HostCalls(hip);
            b.Row("  (blocks a launch takes)", string.Create(CultureInfo.InvariantCulture,
                $"{hip.PointwiseBlocks(n)} of 256 for {n:N0} elements ({hip.DeviceLimits.ComputeUnits} compute units, {HipBackend.PointwiseBlocksVariable})"));
            foreach (var op in Enum.GetValues<UnaryOp>())
            {
                if (HipBackend.UnaryKernelName(op) is { } kernel)
                {
                    BenchOperation(b, $"{kernel} {b.LargeText}", n, HipBackend.Cost(kernel), () => hip.Unary(op, x, z, n));
                }
            }

            foreach (var op in Enum.GetValues<BinaryOp>())
            {
                if (HipBackend.BinaryKernelName(op) is { } kernel)
                {
                    BenchOperation(b, $"{kernel} {b.LargeText}", n, HipBackend.Cost(kernel), () => hip.Binary(op, x, y, z, n));
                }
            }

            BenchOperation(b, $"affine_f32 {b.LargeText}", n, HipBackend.Cost("affine_f32"), () => hip.Affine(x, z, n, 0.5f, 0.25f));
            BenchOperation(b, $"axpy_f32 {b.LargeText}", n, HipBackend.Cost("axpy_f32"), () => hip.Axpy(x, z, n, 1e-6f));
            BenchOperation(b, $"muladd_f32 {b.LargeText}", n, HipBackend.Cost("muladd_f32"), () => hip.MulAdd(x, y, z, n));
            int cols = 4096, rows = n / cols;
            if (rows > 0)
            {
                BenchOperation(b, $"add_rowvec_f32 {rows}x{cols}", (long)rows * cols, HipBackend.Cost("add_rowvec_f32"), () => hip.AddRowVector(x, y, z, rows, cols));
            }

            long fallbacks = Idrak.Abstraction.Operations.Kernels.HostCalls(hip) - before;
            b.Row("  (host fallbacks during the section)", fallbacks == 0 ? "none: every operation ran on its kernel" : $"{fallbacks} (SHOULD BE NONE)");
        }
        finally
        {
            x.Release();
            y.Release();
            z.Release();
        }
    }
}

internal abstract partial class DesktopTests
{
    // One product timed: its first call apart from the warm ones, its rates from the whole call's flops and bytes.
    private static void BenchProduct(HipBench b, string name, (double Flops, double Bytes) cost, Action run) => BenchOperation(b, name, 1, cost, run);

    private static void BenchHipProducts(HipBench b)
    {
        if (!BenchKernelSet(b))
        {
            return;
        }

        var hip = b.Hip;
        var random = new Random(7);
        var owned = new List<Storage>();
        Storage On(long count)
        {
            var values = new float[Math.Max(1, count)];
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = random.NextSingle() - 0.5f;
            }

            var s = hip.Allocate(values.Length, zeroed: false);
            owned.Add(s);
            hip.Upload(values, s);
            return s;
        }

        // Packed weights of random bits: small integers for int8 and int4, the high halves of floats for bfloat16 and float16.
        Storage Packed(long words, bool halves)
        {
            var values = new float[Math.Max(1, words)];
            var bits = System.Runtime.InteropServices.MemoryMarshal.Cast<float, uint>(values.AsSpan());
            for (int i = 0; i < bits.Length; i++)
            {
                bits[i] = halves ? 0x3C003C00u ^ (uint)random.Next(0, 0x03FF03FF) : (uint)random.Next();
            }

            var s = hip.Allocate(values.Length, zeroed: false);
            owned.Add(s);
            hip.Upload(values, s);
            return s;
        }

        long before = Idrak.Abstraction.Operations.Kernels.HostCalls(hip);
        try
        {
            // The float32 product: the side the large buffer holds three of, at most 1,024.
            int side = (int)Math.Clamp(Math.Sqrt(b.LargeElements / 3.0), 64, 1024) / 64 * 64;
            Storage a = On((long)side * side), w = On((long)side * side), c = On((long)side * side), bias = On(side);
            foreach (var (ta, tb) in new[] { (false, false), (false, true), (true, false), (true, true) })
            {
                BenchProduct(b, string.Create(CultureInfo.InvariantCulture, $"matmul_{(ta ? 't' : 'n')}{(tb ? 't' : 'n')} {side}x{side}x{side}"),
                    HipBackend.ProductCost("matmul_f32", side, side, side), () => hip.MatMul(a, w, c, side, side, side, ta, tb, 0f));
            }

            BenchProduct(b, string.Create(CultureInfo.InvariantCulture, $"matmul_nn+bias {side}x{side}x{side}"), HipBackend.ProductCost("matmul_f32", side, side, side),
                () => hip.MatMulBiasKernel(a, w, bias, c, side, side, side));
            int quarter = side / 4;
            BenchProduct(b, string.Create(CultureInfo.InvariantCulture, $"matmul_nt {quarter}x{quarter}x{side} batch 16"), HipBackend.ProductCost("matmul_f32", quarter, quarter, side, batch: 16),
                () => hip.BatchedMatMul(a, w, c, 16, quarter, quarter, side, false, true, 0f));

            // Few rows (decoding) through the float32 kernels and the packed ones.
            int k = (int)Math.Clamp(b.LargeElements / 4096L, 256, 4096) / 64 * 64, n = 4096;
            Storage weights = On((long)k * n), x = On(8L * Math.Max(k, 3072)), y = On(8L * 32000);
            foreach (int m in new[] { 1, 8 })
            {
                BenchProduct(b, string.Create(CultureInfo.InvariantCulture, $"gemv_nn {m}x{n}x{k}"), HipBackend.ProductCost("gemv_nn_f32", m, n, k),
                    () => hip.MatMul(x, weights, y, m, n, k, false, false, 0f));
                BenchProduct(b, string.Create(CultureInfo.InvariantCulture, $"gemv_nt {m}x{k}x{n}"), HipBackend.ProductCost("gemv_nt_f32", m, k, n),
                    () => hip.MatMul(x, weights, y, m, k, n, false, true, 0f));
            }

            (GcnWeights Format, int K, int N)[] shapes =
            [
                (GcnWeights.Int8, 3072, 1024), (GcnWeights.Int8, k, n), (GcnWeights.Int4, 1024, 32000), (GcnWeights.Int4, k, n), (GcnWeights.BFloat16, k, n), (GcnWeights.Float16, k, n),
            ];
            foreach (var (format, kk, nn) in shapes)
            {
                int perWord = GcnKernels.ColumnsPerWord(format);
                long words = (nn + perWord - 1) / perWord;
                Storage packed = Packed(kk * words, format is GcnWeights.BFloat16 or GcnWeights.Float16);
                Storage scales = On(format == GcnWeights.Int4 ? (kk + 31) / 32 * 8 * words : nn);
                foreach (int m in new[] { 1, 4, 8 })
                {
                    string kernel = GcnKernels.PackedGemvKernel(format, m);
                    int splits = HipBackend.GemvSplitCount(hip.DeviceLimits.ComputeUnits, (int)((words + 63) / 64), kk, HipBackend.GemvSplitsSetting(Environment.GetEnvironmentVariable(HipBackend.GemvSplitsVariable)));
                    var cost = HipBackend.ProductCost(kernel, m, nn, kk, splits);
                    Action run = format switch
                    {
                        GcnWeights.Int8 => () => hip.Int8MatMul(x, packed, scales, y, m, nn, kk),
                        GcnWeights.Int4 => () => hip.Int4MatMul(x, packed, scales, y, m, nn, kk),
                        GcnWeights.BFloat16 => () => hip.BFloat16MatMul(x, packed, y, m, nn, kk),
                        _ => () => hip.Float16MatMul(x, packed, y, m, nn, kk),
                    };
                    BenchProduct(b, string.Create(CultureInfo.InvariantCulture, $"{kernel} {m}x{kk}->{nn} (about {splits} splits)"), cost, run);
                }
            }

            int rows = 64;
            {
                long words = (n + 3) / 4;
                Storage packed = Packed(k * words, false), scales = On(n), many = On((long)rows * k), output = On((long)rows * n);
                BenchProduct(b, string.Create(CultureInfo.InvariantCulture, $"int8_matmul_f32 {rows}x{k}->{n}"), HipBackend.ProductCost("int8_matmul_f32", rows, n, k),
                    () => hip.Int8MatMul(many, packed, scales, output, rows, n, k));
            }

            // The expansions over k x n weights, and bfloat16 packing over the large buffer.
            Storage expanded = On((long)k * n);
            foreach (var format in new[] { GcnWeights.Int8, GcnWeights.Int4, GcnWeights.BFloat16, GcnWeights.Float16 })
            {
                int perWord = GcnKernels.ColumnsPerWord(format);
                long words = (n + perWord - 1) / perWord;
                Storage packed = Packed(k * words, format is GcnWeights.BFloat16 or GcnWeights.Float16);
                Storage scales = On(format == GcnWeights.Int4 ? (k + 31) / 32 * 8 * words : n);
                string kernel = GcnKernels.DequantKernel(format);
                Action run = format switch
                {
                    GcnWeights.Int8 => () => hip.Int8Dequantize(packed, scales, expanded, k, n),
                    GcnWeights.Int4 => () => hip.Int4Dequantize(packed, scales, expanded, k, n),
                    GcnWeights.BFloat16 => () => hip.BFloat16Dequantize(packed, expanded, k, n),
                    _ => () => hip.Float16Dequantize(packed, expanded, k, n),
                };
                BenchOperation(b, string.Create(CultureInfo.InvariantCulture, $"{kernel} {k}x{n}"), (long)k * n, HipBackend.Cost(kernel), run);
            }

            Storage source = On(b.LargeElements), target = On(b.LargeElements / 2);
            BenchOperation(b, $"bf16_pack_f32 {b.LargeText}", b.LargeElements, HipBackend.Cost("bf16_pack_f32"), () => hip.PackBFloat16(source, target, b.LargeElements));
            long fallbacks = Idrak.Abstraction.Operations.Kernels.HostCalls(hip) - before;
            b.Row("  (host fallbacks during the section)", fallbacks == 0 ? "none: every operation ran on its kernel" : $"{fallbacks} (SHOULD BE NONE)");
        }
        finally
        {
            foreach (var s in owned)
            {
                s.Release();
            }
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

    /// <summary>Whether a section timed the kernel set's cold and warm loads already.</summary>
    internal bool SetTimed { get; set; }

    /// <summary>fill_f32 once a section loaded it (null before, and when the runtime refused it).</summary>
    internal HipBackend.GcnFunction? Fill { get; set; }

    internal void Header()
    {
        Console.WriteLine($"== {Device}: {Hip.Name}");
        Console.WriteLine($"   HIP runtime {HipBackend.FormatVersion(Hip.RuntimeVersion)}, driver {Hip.DriverVersion}, from {HipRuntime.LoadedFrom}; target {Hip.DeviceLimits.Architecture}");
        Console.WriteLine($"   operations: {(Hip.KernelsUnavailableReason is { } off ? $"every one through the host fallback ({off})" : "rows, element-wise, products, attention, shapes and decoding on Idrak's GCN kernels, the rest through the host fallback")}");
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
