// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Backends.Cuda;
using Idrak.Layers;

// Card-dependent choices (k splits, kernel, tile) are measured on the device in use, not taken from one card's benchmarks.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] TuningGroup =
    [
        ("tuning: card-dependent choices (few-row kernel and splits, prompt splits, tensor-core splits, tiles, decoding-attention splits) are measured on the device and give the results of the formulas' choices", TunedChoicesMatch),
        ("tuning: candidates are compared by their median time, so a lucky timing does not win and a close one keeps the formula's choice", TuneMedians),
        ("tuning: the cache file of measured choices round-trips and is ignored when the GPU, driver, library build or format differs", TuningCacheFile),
        ("tuning: measured choices are kept per GPU in the cache folder and read back instead of measured again", TuningPersisted),
    ];

    private static void TunedChoicesMatch(Device device)
    {
        if (device.Backend is not CudaBackend cuda)
        {
            return;                                                              // the CPU has nothing card-dependent to measure
        }

        var r = new Random(21);
        float[] Values(int n) => [.. Enumerable.Range(0, n).Select(_ => r.NextSingle() * 2 - 1)];
        var (w, xs, a, b) = (Values(512 * 1536), Values(64 * 512), Values(256 * 4096), Values(4096 * 256));
        var (keys, values, query) = (Values(8 * 1024 * 64), Values(8 * 1024 * 64), Values(8 * 2 * 64));
        float[][] Run()
        {
            using var weights = Tensor.From(w, [512, 1536], device);
            using var int8 = Int8Weight.Quantize(weights);
            var results = new List<float[]>();
            using (MixedPrecision.BFloat16())
            {
                foreach (int rows in new[] { 1, 4, 8, 40 })                     // GEMV splits, few-row kernel, prompt splits
                {
                    using var x = Tensor.From(xs.AsSpan(0, rows * 512), [rows, 512], device);
                    using var y = x.MatMulInt8(int8);
                    results.Add(y.ToArray());
                }

                using var ta = Tensor.From(a, [256, 4096], device);              // a long k over few output tiles: k splits
                using var tb = Tensor.From(b, [4096, 256], device);
                using var product = ta.MatMul(tb);
                results.Add(product.ToArray());
            }

            using var fa = Tensor.From(a, [256, 4096], device);                  // float tiles (no tensor cores)
            using var fb = Tensor.From(b, [4096, 256], device);
            using var plain = fa.MatMul(fb);
            results.Add(plain.ToArray());

            // Decoding attention over 700 of 1024 cached positions (splits measured at a full cache).
            using var cache = new KeyValueCache(8, 1024, 64, device, KeyValueFormat.Float32);
            device.Backend.Upload(keys, cache.Keys.Storage);
            device.Backend.Upload(values, cache.Values.Storage);
            using var q = Tensor.From(query, [8, 2, 64], device);
            using var position = Tensor.From([699f], [1], device);
            using var attention = Tensor.AttentionDecode(q, cache, position, 1, 0.125f);
            results.Add(attention.ToArray());
            return [.. results];
        }

        bool autotune = CudaBackend.Autotune, persist = CudaBackend.PersistTuning;
        try
        {
            CudaBackend.PersistTuning = false;                                   // measure here, whatever an earlier run kept
            CudaBackend.Autotune = false;
            cuda.ForgetTuning();
            var formulas = Run();
            Check(cuda.TunedCount == 0, "nothing measured with autotuning off");

            CudaBackend.Autotune = true;
            var measured = Run();
            Check(cuda.TunedCount > 0, $"choices measured on the device ({cuda.TunedCount})");
            int known = cuda.TunedCount;
            var again = Run();
            Check(cuda.TunedCount == known, "each shape is measured once");
            for (int i = 0; i < formulas.Length; i++)
            {
                AssertClose(formulas[i], measured[i], 2e-3f * Math.Max(1f, formulas[i].Max(Math.Abs)), $"product {i}: measured choice against the formula's");
                AssertClose(measured[i], again[i], 2e-3f * Math.Max(1f, measured[i].Max(Math.Abs)), $"product {i}: the kept choice");
            }
        }
        finally
        {
            CudaBackend.Autotune = autotune;
            CudaBackend.PersistTuning = persist;
            cuda.ForgetTuning();
        }
    }

    // Pure (no GPU): the comparison with an injected timer.
    private static void TuneMedians(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                              // device-independent: once is enough
        }

        // What --bench-gemv showed on "o + add + norm": 16 splits 6.6 µs, 8 splits 7.7 µs, but one timing of 8 came out
        // low and two of 16 high. The best time of each picked 8; the median picks 16.
        int[] candidates = [1, 2, 4, 8, 16];
        float[][] timings =
        [
            [20f, 20.1f, 19.9f, 20f, 20.2f, 20f, 19.8f],
            [12f, 12.1f, 11.9f, 12f, 12.2f, 12f, 11.8f],
            [9f, 9.1f, 8.9f, 9f, 9.2f, 9f, 8.8f],
            [7.7f, 7.8f, 5.0f, 7.7f, 7.6f, 7.7f, 7.8f],
            [6.6f, 6.7f, 6.5f, 11f, 6.6f, 12f, 6.6f],
        ];
        var round = new int[candidates.Length];
        var order = new List<int>();
        float Time(int c)
        {
            order.Add(c);
            return timings[c][round[c]++];
        }

        int chosen = TuneTiming.Choose(candidates, 8, Time);
        Check(candidates[chosen] == 16, $"the median picks 16 splits (picked {candidates[chosen]})");
        Check(round.All(n => n == TuneTiming.Rounds), "every candidate timed once per round");
        Check(order.Take(candidates.Length).SequenceEqual([0, 1, 2, 3, 4]) && order.Skip(candidates.Length).Take(candidates.Length).SequenceEqual([4, 3, 2, 1, 0]),
            "candidates take turns forwards, then backwards");

        // Within the margin of the formula's choice: the formula's choice stays.
        int close = TuneTiming.Choose([8, 16], 8, c => c == 0 ? 7.7f : 7.6f);
        Check(close == 0, "a candidate less than 3% faster does not replace the formula's choice");
        int clear = TuneTiming.Choose([8, 16], 8, c => c == 0 ? 7.7f : 6.6f);
        Check(clear == 1, "a clearly faster candidate replaces it");
        int noFormula = TuneTiming.Choose([2, 4], 3, c => c == 0 ? 5f : 4.9f);
        Check(noFormula == 1, "the fastest median wins when the formula's choice is not a candidate");

        Check(TuneTiming.Median([3f, 1f, 2f]) == 2f && TuneTiming.Median([4f, 1f, 3f, 2f]) == 2.5f, "median of odd and even counts");
    }

    // Pure (no GPU apart from the backend-free helpers): the file format, identity checks, saving and loading.
    private static void TuningCacheFile(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        var identity = new TuningIdentity("NVIDIA Test GPU; compute 8.6; 30 SMs; 6144 MiB", "CUDA 13000; 580.82.07", "0.1.7+abc; kernels 0123456789abcdef");
        var entries = new Dictionary<TuneKey, int>
        {
            [new TuneKey(TuneOp.GemvSplits, 0, 1, 2048, 1024)] = 16,
            [new TuneKey(TuneOp.TensorSplits, 9, 256, 256, 4096, 256, -1, 7)] = 6,
            [new TuneKey(TuneOp.DecodeSplits, 2, 16, 4096, 128)] = 24,
        };

        string text = TuningCache.Serialize(identity, entries);
        Check(text.StartsWith($"idrak-tuning {TuningCache.FormatVersion}\n", StringComparison.Ordinal), "the file starts with its format version");
        var parsed = TuningCache.Parse(text, identity);
        Check(parsed is not null && parsed.Count == entries.Count && entries.All(e => parsed.TryGetValue(e.Key, out int v) && v == e.Value), "round trip");
        Check(TuningCache.Serialize(identity, parsed!) == text, "the same text again (stable order)");
        Check(text.Replace("\n", "\r\n", StringComparison.Ordinal) is var crlf && TuningCache.Parse(crlf, identity)?.Count == entries.Count, "Windows line ends");

        Check(TuningCache.Parse(text, identity with { Device = "NVIDIA Test GPU; compute 8.6; 20 SMs; 6144 MiB" }) is null, "another GPU: ignored");
        Check(TuningCache.Parse(text, identity with { Driver = "CUDA 13000; 580.95.05" }) is null, "another driver release: ignored");
        Check(TuningCache.Parse(text, identity with { Driver = "CUDA 12090; 580.82.07" }) is null, "another CUDA version: ignored");
        Check(TuningCache.Parse(text, identity with { Library = "0.1.7+abc; kernels fedcba9876543210" }) is null, "other kernels: ignored");
        Check(TuningCache.Parse(text.Replace($"idrak-tuning {TuningCache.FormatVersion}", "idrak-tuning 999", StringComparison.Ordinal), identity) is null, "another format version: ignored");
        Check(TuningCache.Parse("", identity) is null && TuningCache.Parse("garbage", identity) is null, "not a cache file: ignored");
        var unknown = TuningCache.Parse(text + "NoSuchOp 0 1 2 3 0 0 0 = 4\nGemvSplits 0 1 x 3 0 0 0 = 4\nGemvSplits 1 1 2 3 0 0 0 = 8\n", identity);
        Check(unknown?.Count == entries.Count + 1, "lines that do not parse are skipped, the rest kept");

        string name = TuningCache.FileName("NVIDIA GeForce RTX 3060 Laptop GPU; compute 8.6; 30 SMs; 6144 MiB");
        Check(name == "NVIDIA_GeForce_RTX_3060_Laptop_GPU_compute_8.6_30_SMs_6144_MiB.txt", $"file name ({name})");
        Check(TuningCache.FileName("a/b\\c:d*e?f\"g<h>i|j") == "a_b_c_d_e_f_g_h_i_j.txt", "characters a file system rejects are replaced");

        Check(TuningCache.Folder(v => v == "IDRAK_TUNING_CACHE" ? "0" : null) is null, "IDRAK_TUNING_CACHE=0 turns the cache off");
        Check(TuningCache.Folder(v => v == "IDRAK_TUNING_CACHE" ? "false" : null) is null, "IDRAK_TUNING_CACHE=false turns the cache off");
        Check(TuningCache.Folder(v => v == "IDRAK_TUNING_CACHE" ? "/some/folder" : null) == "/some/folder", "IDRAK_TUNING_CACHE=<folder> chooses the folder");
        Check(TuningCache.Folder(v => v == "IDRAK_CACHE" ? "/cache" : null) == Path.Combine("/cache", "tuning", "cuda"), "under IDRAK_CACHE by default");
        Check(TuningCache.Folder(v => v == "IDRAK_TUNING_CACHE" ? "1" : v == "IDRAK_CACHE" ? "/cache" : null) == Path.Combine("/cache", "tuning", "cuda"), "IDRAK_TUNING_CACHE=1: the default folder");

        string folder = Path.Combine(Path.GetTempPath(), $"idrak-tuning-test-{Environment.ProcessId}-{Guid.NewGuid():N}");
        try
        {
            string path = Path.Combine(folder, "nested", TuningCache.FileName(identity.Device));
            Check(TuningCache.Load(path, identity).Count == 0, "no file yet: nothing kept");
            Check(TuningCache.Save(path, identity, entries), "saved (the folder is created)");
            var loaded = TuningCache.Load(path, identity);
            Check(loaded.Count == entries.Count && entries.All(e => loaded[e.Key] == e.Value), "loaded back");

            // Another process of the same identity adds a choice: both kept, ours win where both measured.
            var more = new Dictionary<TuneKey, int> { [new TuneKey(TuneOp.FloatTile, 0, 256, 256, 4096, 1)] = 128, [new TuneKey(TuneOp.GemvSplits, 0, 1, 2048, 1024)] = 8 };
            Check(TuningCache.Save(path, identity, more), "saved again");
            loaded = TuningCache.Load(path, identity);
            Check(loaded.Count == entries.Count + 1 && loaded[new TuneKey(TuneOp.GemvSplits, 0, 1, 2048, 1024)] == 8, "merged with what the file held");

            // A driver update: the old file is ignored and rewritten with the new identity only.
            var updated = identity with { Driver = "CUDA 13010; 590.10" };
            Check(TuningCache.Load(path, updated).Count == 0, "after a driver update nothing is kept");
            Check(TuningCache.Save(path, updated, new Dictionary<TuneKey, int> { [new TuneKey(TuneOp.PackedTile, 0, 180, 1024, 1024)] = 64 }), "rewritten");
            Check(TuningCache.Load(path, updated).Count == 1 && TuningCache.Load(path, identity).Count == 0, "only the new driver's choices remain");
            Check(!Directory.EnumerateFiles(Path.GetDirectoryName(path)!, "*.tmp").Any(), "no temporary file left");
        }
        finally
        {
            try
            {
                Directory.Delete(folder, recursive: true);
            }
            catch (IOException)
            {
            }
        }
    }

    // On a GPU: choices measured once are written to the cache file and read back by a fresh start (simulated by
    // forgetting what this process measured), without measuring again; a file of another identity is not used.
    private static void TuningPersisted(Device device)
    {
        if (device.Backend is not CudaBackend cuda)
        {
            return;
        }

        string folder = Path.Combine(Path.GetTempPath(), $"idrak-tuning-gpu-{Environment.ProcessId}-{Guid.NewGuid():N}");
        string? setting = Environment.GetEnvironmentVariable("IDRAK_TUNING_CACHE");
        bool autotune = CudaBackend.Autotune, persist = CudaBackend.PersistTuning;
        var r = new Random(5);
        var w = Enumerable.Range(0, 1024 * 768).Select(_ => r.NextSingle() - 0.5f).ToArray();
        var xs = Enumerable.Range(0, 1024).Select(_ => r.NextSingle() - 0.5f).ToArray();
        float[] Product()
        {
            using var weights = Tensor.From(w, [1024, 768], device);
            using var int8 = Int8Weight.Quantize(weights);
            using var x = Tensor.From(xs, [1, 1024], device);
            using var y = x.MatMulInt8(int8);
            return y.ToArray();
        }

        try
        {
            Environment.SetEnvironmentVariable("IDRAK_TUNING_CACHE", folder);
            CudaBackend.Autotune = true;
            CudaBackend.PersistTuning = true;
            cuda.ForgetTuning();
            int measuredBefore = cuda.MeasuredCount;
            var first = Product();
            Check(cuda.MeasuredCount > measuredBefore, "measured on first use");
            string? file = cuda.TuningFile;
            Check(file is not null && file.StartsWith(folder, StringComparison.Ordinal) && File.Exists(file), $"written to the cache folder ({file})");
            var kept = TuningCache.Load(file!, cuda.TuningIdentity);
            Check(kept.Count > 0, $"the file holds the choices ({kept.Count})");

            cuda.ForgetTuning();                                                 // a fresh start
            int measured = cuda.MeasuredCount;
            var second = Product();
            Check(cuda.MeasuredCount == measured && cuda.TunedCount > 0, "read back instead of measured again");
            AssertClose(first, second, 1e-5f, "the same choice gives the same result");

            // The same file under another driver's identity: not used.
            File.WriteAllText(file!, TuningCache.Serialize(cuda.TuningIdentity with { Driver = "CUDA 1; other" }, kept));
            cuda.ForgetTuning();
            measured = cuda.MeasuredCount;
            Product();
            Check(cuda.MeasuredCount > measured, "a file of another driver is ignored and the choices measured again");
            Check(TuningCache.Load(file!, cuda.TuningIdentity).Count > 0, "and the file rewritten for this one");
        }
        finally
        {
            Environment.SetEnvironmentVariable("IDRAK_TUNING_CACHE", setting);
            CudaBackend.Autotune = autotune;
            CudaBackend.PersistTuning = persist;
            cuda.ForgetTuning();
            try
            {
                Directory.Delete(folder, recursive: true);
            }
            catch (IOException)
            {
            }
        }
    }
}
