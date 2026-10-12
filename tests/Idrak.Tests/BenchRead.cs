// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using Idrak;
using Idrak.Data;
using Idrak.Diagnostics;
using Idrak.Generation;
using Idrak.Models;

// A page read profiled (plan 17, H3) through the library's profiler (Profiler, ProfileProbes, ProfileWriters): the app's
// flow (the prepared page for the preview, then the read, greedy) on each device IDRAK_DEVICES names (default the CPU).
// What the read is:
//   IDRAK_BENCH_MODEL       a checkpoint folder, or a Hugging Face id ("bakrianoo/arabic-legal-documents-ocr-1.0": the copy the
//                           app downloaded, from the same cache; HF_TOKEN for gated ones) (default tests' vlm/tiny-gemma3)
//   IDRAK_BENCH_PAGE        an image file (default the seeded 1,654 x 2,339 scan of --bench-image, JPEG)
//   IDRAK_BENCH_TRANSFORMS  the card's pipeline (default grayscale,max_width=1024,contrast=1.5)
//   IDRAK_BENCH_PROMPT      the instruction (default "Extract details to JSON.")
//   IDRAK_BENCH_TOKENS      the answer's most tokens (default 64)
//   IDRAK_BENCH_WEIGHTS     f32, bf16, int8 or int4 (default f32);  IDRAK_BENCH_CACHE f32 or bf16;  IDRAK_BENCH_ENCODER f32 or bf16;
//                           IDRAK_BENCH_PREFILL f32 or bf16 (the prompt's pass)
//   IDRAK_BENCH_HEAD        the head's format beside int8 or int4 weights: bf16 (the head is the only vocabulary table, as the
//                           app's HeadWeights) or empty (as the weights, beside a bfloat16 token table: the default)
//   IDRAK_LOOKUP_TABLES     where that separate token table lives: auto (on the device until it is full), host or device
//   IDRAK_BENCH_CONTEXT     the context window (default the model's)
//   IDRAK_BENCH_BATCH       pages read together (ImageReader.ReadBatch; default 1: the page alone): the page this many times,
//                           with the time a page and the tokens a second of the whole batch
//   IDRAK_BENCH_BATCH_FIT   split (the default: pages that do not fit together are read in halves) or fail;
//                           IDRAK_BENCH_MAX_BATCH the most pages together (default as many as fit)
//   IDRAK_BENCH_SUGGEST     1 (the default): suggested settings (TelemetryLevel.Suggestions) printed as the reader makes them; 0: none
// How it is measured and written (ProfileOptions.FromEnvironment): IDRAK_PROFILE_WARM (3), IDRAK_PROFILE_COLD (1),
// IDRAK_PROFILE_PROBES (all: bandwidth, cpu, heap, memory, operations, kernels, stages), IDRAK_PROFILE_STATISTIC (median),
// IDRAK_PROFILE_FORMAT (table, markdown, csv, json), IDRAK_PROFILE_OUT (a file, appended).
// Two profiles: the page alone before any model (decode, transforms), then the read: its stages (prepare; encode, the
// image through the encoder and the prompt built: the read's time less the generation's; prefill; decode; first text),
// the speeds (prompt and answer tokens a second, ms a token) and estimated compute (2 x parameters a token: GFLOP/s; the
// weights read once a decoded token: GB/s), with what the probes measure around it. The read's memory: the decoder's
// weights (MB), the device's in use, cached and peak ("memory"), and in the "stages" run the caches allocated, the
// encoder's output and its peak above the run's start (its activations), cold and warm.
internal static partial class Tests
{
    internal static int BenchRead()
    {
        RegisterGemma3Vision();
        using var suggestions = Environment.GetEnvironmentVariable("IDRAK_BENCH_SUGGEST") is "0" or "false" ? null
            : Telemetry.Subscribe(new ConsoleLogger(TelemetryLevel.Suggestions));
        string Setting(string name, string fallback) => Environment.GetEnvironmentVariable(name) is { Length: > 0 } value ? value : fallback;
        string folder = Setting("IDRAK_BENCH_MODEL", TestData("vlm/tiny-gemma3"));
        if (!Directory.Exists(folder) && folder.Count(ch => ch == '/') == 1 && !Path.IsPathRooted(folder))
        {
            folder = Idrak.Data.HuggingFaceModels.DownloadAsync(folder, token: Idrak.Data.HuggingFace.Token()).GetAwaiter().GetResult();   // cached once, as the app does
            Console.WriteLine($"model: {folder}");
        }

        string weights = Setting("IDRAK_BENCH_WEIGHTS", "f32").ToLowerInvariant();
        string? head = weights is "int8" or "int4" && Setting("IDRAK_BENCH_HEAD", "").ToLowerInvariant() is "bf16" or "bfloat16" ? "bfloat16" : null;
        bool Bf16(string name) => Setting(name, "f32").ToLowerInvariant() is "bf16" or "bfloat16";
        int tokens = int.Parse(Setting("IDRAK_BENCH_TOKENS", "64"), CultureInfo.InvariantCulture);
        int batch = Math.Max(1, int.Parse(Setting("IDRAK_BENCH_BATCH", "1"), CultureInfo.InvariantCulture));
        int? context = Environment.GetEnvironmentVariable("IDRAK_BENCH_CONTEXT") is { Length: > 0 } c ? int.Parse(c, CultureInfo.InvariantCulture) : null;
        string transforms = Setting("IDRAK_BENCH_TRANSFORMS", "grayscale,max_width=1024,contrast=1.5");
        string prompt = Setting("IDRAK_BENCH_PROMPT", "Extract details to JSON.");
        var page = Environment.GetEnvironmentVariable("IDRAK_BENCH_PAGE") is { Length: > 0 } file
            ? ChatImage.FromFile(file)
            : ChatImage.FromBytes(JpegEncoder.Encode(ScanPage(1654, 2339, new Random(17)), quality: 90), "image/jpeg");
        List<Device> devices = Environment.GetEnvironmentVariable("IDRAK_DEVICES") is { Length: > 0 } chosen
            ? [.. chosen.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(Device.Parse)]
            : [Device.Cpu];
        var options = ProfileOptions.FromEnvironment();
        string pageText = string.Create(CultureInfo.InvariantCulture, $"{page.MediaType} {page.Data.Length / 1024} KiB, \"{transforms}\"");
        string setup = $"{Path.GetFileName(folder.TrimEnd('/', '\\'))}, weights {weights}{(head is null ? "" : ", head bf16")}, lookup tables {ComputeResources.LookupTables}, cache {(Bf16("IDRAK_BENCH_CACHE") ? "bf16" : "f32")}, "
                       + string.Create(CultureInfo.InvariantCulture, $"encoder {(Bf16("IDRAK_BENCH_ENCODER") ? "bf16" : "f32")}, prefill {(Bf16("IDRAK_BENCH_PREFILL") ? "bf16" : "f32")}, {pageText}, up to {tokens} tokens")
                       + (batch > 1 ? string.Create(CultureInfo.InvariantCulture, $", {batch} pages together") : "");

        // The page alone, before any model is loaded: what preparing it costs with nothing else in the process (no trace
        // or kernel timing: it runs on the CPU's own code).
        var pipeline = ImageTransformPipeline.Parse(transforms);
        Profiler.Run($"page alone: {pageText}", Device.Cpu, run =>
        {
            var decoded = run.Stage("decode", () => ChatImageDecoder.Decode(page));
            run.Stage("transforms", () => pipeline.Apply(decoded));
            run.Record("pixels", decoded.Width * (double)decoded.Height / 1e6, "Mpx");
        }, options with { Probes = [.. (options.Probes ?? ProfileProbes.Names).Where(p => !ProfileProbes.Get(p).Intrusive)] }).Write();

        foreach (var device in devices)
        {
            long loading = Stopwatch.GetTimestamp();
            using var model = PretrainedModel.Load(folder, new PretrainedOptions
            {
                Device = device,
                BFloat16 = weights is "bf16" or "bfloat16",
                Int8 = weights == "int8",
                Int4 = weights == "int4",
                HeadFormatName = head,
                MaxPositions = context,
            });
            double loadSeconds = Stopwatch.GetElapsedTime(loading).TotalSeconds;
            long loadPeak;                                                     // the process's host memory peak so far: the load's
            using (var process = Process.GetCurrentProcess())
            {
                loadPeak = process.PeakWorkingSet64;
            }
            using var reader = new ImageReader(model, new ImageReaderOptions
            {
                Transforms = pipeline,
                CacheFormat = Bf16("IDRAK_BENCH_CACHE") ? KeyValueFormat.BFloat16 : KeyValueFormat.Float32,
                EncoderPrecision = Bf16("IDRAK_BENCH_ENCODER") ? MatMulPrecision.BFloat16 : null,
                PrefillPrecision = Bf16("IDRAK_BENCH_PREFILL") ? MatMulPrecision.BFloat16 : null,
                ContextLength = context,
                BatchFit = Setting("IDRAK_BENCH_BATCH_FIT", "split").ToLowerInvariant() == "fail" ? BatchFit.Fail : BatchFit.Split,
                MaxBatch = Environment.GetEnvironmentVariable("IDRAK_BENCH_MAX_BATCH") is { Length: > 0 } most ? int.Parse(most, CultureInfo.InvariantCulture) : null,
            });
            long parameters = ModelMemory.WeightCount(model.Network), weightBytes = ModelMemory.TensorBytes(model.Network);
            var request = new ImageReadRequest(page, prompt) { MaxTokens = tokens };
            var report = batch > 1 ? Profiler.Run($"read: {setup}", device, run =>
            {
                // The page `batch` times, read together: the batch's time, a page's share of it, and the tokens of all.
                var readings = run.Stage("read", () => reader.ReadBatch([.. Enumerable.Repeat(request, batch)]));
                int answered = readings.Sum(r => r.Stats?.GeneratedTokens ?? 0);
                double decode = readings.Max(r => r.Stats?.GenerationDuration.TotalMilliseconds ?? 0);
                run.Record("page", run.Value("read")!.Value / batch, "ms");
                run.Record("prefill", readings.Max(r => r.Stats?.PromptDuration.TotalMilliseconds ?? 0), "ms");
                run.Record("decode", decode, "ms");
                run.Record("answer tokens", answered, "tok");
                run.Record("decode speed", Rate(answered, decode), "tok/s");
                run.Record("same answers", readings.Select(r => r.Text).Distinct().Count() == 1 ? 1 : 0);
                run.Record("pages together", readings.Min(r => r.BatchSize));
            }, options) : Profiler.Run($"read: {setup}", device, run =>
            {
                run.Stage("prepare", () => reader.Prepared(request));
                var reading = reader.Read(request);
                run.Record("decoder weights", weightBytes / 1048576.0, "MB");
                var stats = reading.Stats;
                double prefill = stats?.PromptDuration.TotalMilliseconds ?? 0, decode = stats?.GenerationDuration.TotalMilliseconds ?? 0;
                int promptTokens = stats?.PromptTokens ?? 0, answerTokens = stats?.GeneratedTokens ?? 0;
                run.Stage("encode", reading.Elapsed - (stats?.TotalDuration ?? TimeSpan.Zero));
                run.Record("prefill", prefill, "ms");
                run.Record("decode", decode, "ms");
                run.Stage("read", reading.Elapsed);
                run.Record("first text", reading.FirstText?.TotalMilliseconds ?? 0, "ms");
                run.Record("prompt tokens", promptTokens, "tok");
                run.Record("answer tokens", answerTokens, "tok");
                run.Record("prefill speed", Rate(promptTokens, prefill), "tok/s");
                run.Record("decode speed", Rate(answerTokens, decode), "tok/s");
                run.Record("ms per token", answerTokens > 0 ? decode / answerTokens : 0, "ms");

                // Estimated compute: a token through the decoder is about 2 x parameters of multiply-adds (attention not
                // counted), and a decoded token reads every weight once.
                run.Record("prefill compute", Rate(2.0 * parameters * promptTokens, prefill) / 1e9, "GFLOP/s");
                run.Record("decode compute", Rate(2.0 * parameters * answerTokens, decode) / 1e9, "GFLOP/s");
                run.Record("decode weights read", Rate((double)weightBytes * answerTokens, decode) / 1e9, "GB/s");
            }, options);
            Console.WriteLine();
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"{device}: loaded in {loadSeconds:F2} s ({model.LoadTimes}), host peak {loadPeak / 1048576.0:F0} MB; decoder {parameters:N0} weights, {weightBytes / 1048576.0:F0} MB of weights"));
            report.Write();
        }

        return 0;

        static double Rate(double amount, double ms) => ms > 0 ? amount / (ms / 1000) : 0;
    }
}
