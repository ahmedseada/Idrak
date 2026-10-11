// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using Idrak;
using Idrak.Abstraction.Operations;
using Idrak.Data;
using Idrak.Diagnostics;
using Idrak.Generation;
using Idrak.Models;

// A page read profiled (plan 17, H3): the app's flow (the prepared page for the preview, then the read, greedy) on each
// device IDRAK_DEVICES names (default the CPU). By default the tiny Gemma 3 fixture and a seeded A4 scan (JPEG); the
// owner's model and pages through the environment:
//   IDRAK_BENCH_MODEL       a checkpoint folder (default tests' vlm/tiny-gemma3)
//   IDRAK_BENCH_PAGE        an image file (default the seeded 1,654 x 2,339 scan of --bench-image)
//   IDRAK_BENCH_TRANSFORMS  the card's pipeline (default grayscale,max_width=1024,contrast=1.5)
//   IDRAK_BENCH_PROMPT      the instruction (default "Extract details to JSON.")
//   IDRAK_BENCH_TOKENS      the answer's most tokens (default 64)
//   IDRAK_BENCH_READS       warm reads after the cold one (default 3; the median is shown)
//   IDRAK_BENCH_WEIGHTS     f32, bf16, int8 or int4 (default f32);  IDRAK_BENCH_CACHE f32 or bf16;  IDRAK_BENCH_ENCODER f32 or bf16
//   IDRAK_BENCH_CONTEXT     the context window (default the model's)
// Shown per read, the cold one apart from the warm ones (rule 76): the stages' times (prepare: decode and transforms;
// encode: the image through the encoder and the prompt built, the read's time less the generation's; prefill; decode),
// the speeds (prompt and answer tokens a second, ms a token), estimated compute (2 x parameters a token: GFLOP/s, and the
// weights read once a decoded token: GB/s), device memory (in use, cached, peak), the bytes allocated on the managed heap,
// collections, and the CPU's cores kept busy. Then one traced read (the operations run, per answer token, and every host
// fallback by operation) and, on a device that times kernels (CUDA), one profiled read: each kernel's calls, time, share
// and TFLOPS. The traced and profiled reads are not in the timings (tracing and profiling slow a read).
internal static partial class Tests
{
    private sealed record ReadSample(double PrepareMs, double ReadMs, double FirstTextMs, double EncodeMs, double PrefillMs, int PromptTokens,
        double DecodeMs, int Tokens, double AllocatedMb, int Gen0, int Gen1, int Gen2, double CpuMs, MemoryUsage Memory, string? DoneReason);

    internal static int BenchRead()
    {
        RegisterGemma3Vision();
        string Setting(string name, string fallback) => Environment.GetEnvironmentVariable(name) is { Length: > 0 } value ? value : fallback;
        string folder = Setting("IDRAK_BENCH_MODEL", TestData("vlm/tiny-gemma3"));
        string weights = Setting("IDRAK_BENCH_WEIGHTS", "f32").ToLowerInvariant();
        bool Bf16(string name) => Setting(name, "f32").ToLowerInvariant() is "bf16" or "bfloat16";
        int tokens = int.Parse(Setting("IDRAK_BENCH_TOKENS", "64"), CultureInfo.InvariantCulture);
        int reads = Math.Max(1, int.Parse(Setting("IDRAK_BENCH_READS", "3"), CultureInfo.InvariantCulture));
        int? context = Environment.GetEnvironmentVariable("IDRAK_BENCH_CONTEXT") is { Length: > 0 } c ? int.Parse(c, CultureInfo.InvariantCulture) : null;
        string transforms = Setting("IDRAK_BENCH_TRANSFORMS", "grayscale,max_width=1024,contrast=1.5");
        string prompt = Setting("IDRAK_BENCH_PROMPT", "Extract details to JSON.");
        var page = Environment.GetEnvironmentVariable("IDRAK_BENCH_PAGE") is { Length: > 0 } file
            ? ChatImage.FromFile(file)
            : ChatImage.FromBytes(JpegEncoder.Encode(ScanPage(1654, 2339, new Random(17)), quality: 90), "image/jpeg");
        List<Device> devices = Environment.GetEnvironmentVariable("IDRAK_DEVICES") is { Length: > 0 } chosen
            ? [.. chosen.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(Device.Parse)]
            : [Device.Cpu];
        var size = ChatImageDecoder.Size(page.Data.Span);
        Console.WriteLine($"Read profile: {Path.GetFileName(folder.TrimEnd('/', '\\'))}, weights {weights}, cache {(Bf16("IDRAK_BENCH_CACHE") ? "bf16" : "f32")}, "
                          + $"encoder {(Bf16("IDRAK_BENCH_ENCODER") ? "bf16" : "f32")}; page {page.MediaType} {page.Data.Length / 1024} KiB"
                          + $"{(size is var (w, h) ? $" {w} x {h}" : "")}; transforms \"{transforms}\"; up to {tokens} tokens; {Environment.ProcessorCount} CPU threads");

        // The page alone, before any model is loaded: what the prepare costs without anything else in the process.
        var pipeline = ImageTransformPipeline.Parse(transforms);
        var decoded = ChatImageDecoder.Decode(page);
        double aloneDecodeMs = Timed(() => ChatImageDecoder.Decode(page)), aloneTransformMs = Timed(() => pipeline.Apply(decoded));

        foreach (var device in devices)
        {
            var load = Stopwatch.StartNew();
            using var model = PretrainedModel.Load(folder, new PretrainedOptions
            {
                Device = device,
                BFloat16 = weights is "bf16" or "bfloat16",
                Int8 = weights == "int8",
                Int4 = weights == "int4",
                MaxPositions = context,
            });
            double loadSeconds = load.Elapsed.TotalSeconds;
            using var reader = new ImageReader(model, new ImageReaderOptions
            {
                Transforms = ImageTransformPipeline.Parse(transforms),
                CacheFormat = Bf16("IDRAK_BENCH_CACHE") ? KeyValueFormat.BFloat16 : KeyValueFormat.Float32,
                EncoderPrecision = Bf16("IDRAK_BENCH_ENCODER") ? MatMulPrecision.BFloat16 : null,
                ContextLength = context,
            });
            long parameters = model.Network.ParameterCount, weightBytes = ModelMemory.TensorBytes(model.Network);
            Console.WriteLine();
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"{device} ({device.Backend.Name}): loaded in {loadSeconds:F2} s; decoder {parameters:N0} parameters, {weightBytes / (double)(1 << 20):F0} MB of weights; "
                + $"device memory {Mb(ComputeResources.GetMemoryUsage(device).InUse)} MB in use after loading"));
            var request = new ImageReadRequest(page, prompt) { MaxTokens = tokens };

            var cold = ReadOnce(device, reader, request);
            var warm = Enumerable.Range(0, reads).Select(_ => ReadOnce(device, reader, request)).ToList();
            ReadSample Median(Func<ReadSample, double> by) => warm.OrderBy(by).ElementAt(warm.Count / 2);
            var typical = Median(s => s.ReadMs);

            Console.WriteLine($"{"read",-6} {"prepare",8} {"encode",8} {"prefill",8} {"decode",8} {"total",8} {"1st text",8}  {"prompt",14} {"answer",16} {"ms/tok",7}  "
                              + $"{"in use",7} {"cached",7} {"peak",7}  {"heap MB",8} {"GC 0/1/2",9} {"cores",5}");
            Row("cold", cold);
            foreach (var (sample, i) in warm.Select((s, i) => (s, i)))
            {
                Row($"warm {i + 1}", sample);
            }

            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"warm median: prepare {Median(s => s.PrepareMs).PrepareMs:F1} ms, encode {Median(s => s.EncodeMs).EncodeMs:F1} ms, prefill {Median(s => s.PrefillMs).PrefillMs:F1} ms, "
                + $"decode {Median(s => s.DecodeMs).DecodeMs:F1} ms, read {typical.ReadMs:F1} ms (the cold read {cold.ReadMs - typical.ReadMs:F0} ms more); answer ended by {typical.DoneReason ?? "?"}"));

            // The prepare by part, warm, now that the model has run: the codec, the transforms, and the reader's whole step.
            double decodeMs = Timed(() => ChatImageDecoder.Decode(page)), transformMs = Timed(() => pipeline.Apply(decoded));
            double preparedMs = Timed(() =>
            {
                _ = reader.Prepared(request);                            // decoded and transformed, kept for the read
                _ = reader.Prepared(request);                            // taken, so the next one does the work again
            });
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"prepare by part (warm, median of 3, after the model ran): decode {decodeMs:F1} ms, transforms {transformMs:F1} ms; the reader's prepare {preparedMs:F1} ms "
                + $"(before any model: decode {aloneDecodeMs:F1} ms, transforms {aloneTransformMs:F1} ms)"));

            // Estimated compute: a token through the decoder is about 2 x parameters of multiply-adds (attention not counted),
            // and a decoded token reads every weight once.
            double prefillFlops = 2.0 * parameters * typical.PromptTokens, decodeFlops = 2.0 * parameters * typical.Tokens;
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"compute (est., 2 x parameters a token): prefill {Rate(prefillFlops, typical.PrefillMs) / 1e9:F1} GFLOP/s, decode {Rate(decodeFlops, typical.DecodeMs) / 1e9:F1} GFLOP/s; "
                + $"weights read while decoding {Rate((double)weightBytes * typical.Tokens, typical.DecodeMs) / 1e9:F1} GB/s"));

            // One traced read: which operations ran, how often, and what fell back to the host.
            KernelTrace trace;
            ReadSample traced;
            using (trace = Kernels.Trace(device.Backend))
            {
                traced = ReadOnce(device, reader, request);
            }

            var calls = Ops.All.Concat(PluginOperations.All).Select(op => (op.Name, Calls: trace.Calls(op))).Where(x => x.Calls > 0).OrderByDescending(x => x.Calls).ToList();
            long total = calls.Sum(x => x.Calls);
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"operations: {total:N0} calls in a read ({(traced.Tokens > 0 ? total / (double)(traced.Tokens + 1) : 0):F0} a token), {calls.Count} kinds; the most called: "
                + $"{string.Join(", ", calls.Take(12).Select(x => string.Create(CultureInfo.InvariantCulture, $"{x.Name} {x.Calls:N0}")))}"));
            var fallbacks = trace.HostCallsByOperation;
            Console.WriteLine(fallbacks.Count == 0
                ? "host fallbacks: none (every operation ran on the device's own kernels)"
                : string.Create(CultureInfo.InvariantCulture, $"host fallbacks: {trace.HostCalls:N0} calls copied to the CPU and back: "
                    + $"{string.Join(", ", fallbacks.OrderByDescending(f => f.Value).Select(f => string.Create(CultureInfo.InvariantCulture, $"{f.Key} {f.Value:N0}")))}"));

            // One profiled read where the device times its kernels.
            if (device.Backend.Capabilities.Profiling)
            {
                GpuProfiler.Start(device);
                var profiled = ReadOnce(device, reader, request);
                var kernels = GpuProfiler.Stop(device);
                double gpu = kernels.Sum(k => k.Milliseconds);
                Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                    $"kernels of one read (each waited for, so the read took {profiled.ReadMs:F0} ms): {gpu:F1} ms on the GPU, {kernels.Sum(k => k.Calls):N0} launches"));
                Console.Write(GpuProfiler.Format(kernels, 30));
            }
            else
            {
                Console.WriteLine($"kernel timing: {device} does not time its kernels (CUDA does)");
            }
        }

        return 0;

        // The median of three calls after a first one.
        static double Timed(Action run)
        {
            run();
            var times = new double[3];
            for (int i = 0; i < times.Length; i++)
            {
                long at = Stopwatch.GetTimestamp();
                run();
                times[i] = Stopwatch.GetElapsedTime(at).TotalMilliseconds;
            }

            Array.Sort(times);
            return times[1];
        }

        static double Rate(double amount, double ms) => ms > 0 ? amount / (ms / 1000) : 0;

        static void Row(string name, ReadSample s)
        {
            var invariant = CultureInfo.InvariantCulture;
            string prompt = string.Create(invariant, $"{s.PromptTokens} @ {Rate(s.PromptTokens, s.PrefillMs):F0}/s");
            string answer = string.Create(invariant, $"{s.Tokens} @ {Rate(s.Tokens, s.DecodeMs):F1}/s");
            string collections = string.Create(invariant, $"{s.Gen0}/{s.Gen1}/{s.Gen2}");
            Console.WriteLine(string.Create(invariant,
                $"{name,-6} {s.PrepareMs,8:F1} {s.EncodeMs,8:F1} {s.PrefillMs,8:F1} {s.DecodeMs,8:F1} {s.ReadMs + s.PrepareMs,8:F1} {s.FirstTextMs,8:F1}  "
                + $"{prompt,14} {answer,16} {(s.Tokens > 0 ? s.DecodeMs / s.Tokens : 0),7:F2}  "
                + $"{Mb(s.Memory.InUse),7} {Mb(s.Memory.Cached),7} {Mb(s.Memory.Peak),7}  {s.AllocatedMb,8:F1} {collections,9} {s.CpuMs / Math.Max(1e-9, s.ReadMs + s.PrepareMs),5:F1}"));
        }
    }

    private static long Mb(long bytes) => bytes >> 20;

    // One read as the app does it: the prepared page (its preview), then the read; timed, with what it allocated and used.
    private static ReadSample ReadOnce(Device device, ImageReader reader, ImageReadRequest request)
    {
        ComputeResources.ResetPeakMemoryUsage(device);
        var process = Process.GetCurrentProcess();
        var cpu = process.TotalProcessorTime;
        int gen0 = GC.CollectionCount(0), gen1 = GC.CollectionCount(1), gen2 = GC.CollectionCount(2);
        long allocated = GC.GetTotalAllocatedBytes(precise: true);
        long start = Stopwatch.GetTimestamp();
        _ = reader.Prepared(request);
        double prepare = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        var reading = reader.Read(request);
        double heap = (GC.GetTotalAllocatedBytes(precise: true) - allocated) / (double)(1 << 20);
        process.Refresh();
        double cpuMs = (process.TotalProcessorTime - cpu).TotalMilliseconds;
        var stats = reading.Stats;
        double total = reading.Elapsed.TotalMilliseconds, generation = stats?.TotalDuration.TotalMilliseconds ?? 0;
        return new ReadSample(prepare, total, reading.FirstText?.TotalMilliseconds ?? 0, total - generation, stats?.PromptDuration.TotalMilliseconds ?? 0,
            stats?.PromptTokens ?? 0, stats?.GenerationDuration.TotalMilliseconds ?? 0, stats?.GeneratedTokens ?? 0, heap,
            GC.CollectionCount(0) - gen0, GC.CollectionCount(1) - gen1, GC.CollectionCount(2) - gen2, cpuMs, ComputeResources.GetMemoryUsage(device), reading.DoneReason);
    }
}
