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
        ("tuning: few-row products are measured over 1, 2, 3, 4, 6, 8, 12, ... splits with the kernel that then runs (fused add-and-normalize kept apart from the plain kernel), after a warm-up, relative to the formula's choice per round", GemvSelection),
        ("tuning: the cache file of measured choices round-trips and is ignored when the GPU, driver, library build or format differs", TuningCacheFile),
        ("tuning: measured choices are kept per GPU in the cache folder and read back instead of measured again", TuningPersisted),
        ("kernel shapes: derived from the device's reported limits; today's cards (12.0 and 8.6) give today's PTX byte for byte, other limits valid PTX with the same kernels, or a reason they cannot run", KernelShapesFromLimits),
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

    // Pure (no GPU): the few-row split choice as CudaBackend.PackedFewRows makes it (candidates, key, formula, TuneTiming)
    // with fake timers shaped like --bench-gemv's results on an RTX 5070 Ti and an RTX 3060 Laptop.
    private static void GemvSelection(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        // Candidates: the bench's columns (any count runs: the kernels take a 64-row step, then single rows).
        Check(CudaBackend.GemvSplitCandidates(2048, 1).SequenceEqual([1, 2, 3, 4, 6, 8, 12, 16, 24, 32]), "k 2048: 1 … 32 with 3, 6, 12, 24");
        Check(CudaBackend.GemvSplitCandidates(3072, 1).SequenceEqual([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]), "k 3072: up to 48");
        Check(CudaBackend.GemvSplitCandidates(1024, 1).SequenceEqual([1, 2, 3, 4, 6, 8, 12, 16]), "k 1024: up to 16");
        var int4 = CudaBackend.GemvSplitCandidates(3072, 64);
        Check(int4.Select(w => (3072 + ((3072 + w - 1) / w + 63) / 64 * 64 - 1) / (((3072 + w - 1) / w + 63) / 64 * 64)).Distinct().Count() == int4.Length,
            $"int4 (64-row groups): no two candidates run the same chunks ({string.Join(",", int4)})");
        foreach (int k in new[] { 64, 1024, 2048, 3072, 4096 })
        {
            foreach (int sms in new[] { 20, 30, 70, 132 })
            {
                foreach (int blocks in new[] { 1, 8, 24, 48, 1187 })
                {
                    int formula = CudaBackend.GemvSplitCount(sms, blocks, k);
                    Check(CudaBackend.GemvSplitCandidates(k, 1).Contains(formula) && CudaBackend.GemvSplitCandidates(k, 64).Contains(formula),
                        $"the formula's choice ({formula}; k {k}, {sms} SMs, {blocks} blocks) is a candidate");
                }
            }
        }

        // Keys: per kernel variant and shape, so the add-and-normalize kernel no longer takes the plain kernel's choice.
        var keys = new HashSet<TuneKey>();
        for (int variant = 0; variant < 12; variant++)
        {
            keys.Add(CudaBackend.GemvSplitsKey(variant, 1, 1024, 2048));
        }

        keys.Add(CudaBackend.GemvSplitsKey(0, 1, 1024, 3072));
        keys.Add(CudaBackend.GemvSplitsKey(0, 2, 1024, 2048));
        Check(keys.Count == 14, "every kernel variant and shape has its own key");
        var identity = new TuningIdentity("GPU", "driver", "library");
        var plainKey = CudaBackend.GemvSplitsKey(CudaBackend.GemvPlain, 1, 1024, 2048);
        var addNormKey = CudaBackend.GemvSplitsKey(CudaBackend.GemvAddNorm, 1, 1024, 2048);
        var file = TuningCache.Parse(TuningCache.Serialize(identity, new Dictionary<TuneKey, int> { [plainKey] = 8, [addNormKey] = 16 }), identity)!;
        Check(file[plainKey] == 8 && file[addNormKey] == 16, "and the cache file keeps them apart");

        // Measured per key with each kernel's own timer, as PackedFewRows does (the fake curves: µs per call).
        Dictionary<TuneKey, int> Measure(int sms, int k, params (TuneKey Key, Dictionary<int, float> Curve)[] kernels)
        {
            var kept = new Dictionary<TuneKey, int>();
            foreach (var (key, curve) in kernels)
            {
                int[] candidates = CudaBackend.GemvSplitCandidates(k, 1);
                int formula = CudaBackend.GemvSplitCount(sms, 8, k);           // 1024 int8 columns: 8 column blocks
                kept.TryAdd(key, candidates[TuneTiming.Choose(candidates, formula, c => curve[candidates[c]])]);
            }

            return kept;
        }

        // RTX 5070 Ti (70 SMs), o + add + norm 2048 → 1024: the fused kernel is fastest at 16 splits (7.5 µs against 8.8 at
        // 8); the plain kernel of the shape at 8. Before, the fused kernel took the plain kernel's 8.
        var plain5070 = new Dictionary<int, float> { [1] = 28f, [2] = 15f, [3] = 11f, [4] = 9.5f, [6] = 8f, [8] = 7.0f, [12] = 7.3f, [16] = 7.6f, [24] = 8.2f, [32] = 8.8f };
        var addNorm5070 = new Dictionary<int, float> { [1] = 30f, [2] = 16f, [3] = 12f, [4] = 10.5f, [6] = 9.3f, [8] = 8.8f, [12] = 8.0f, [16] = 7.5f, [24] = 7.9f, [32] = 8.3f };
        Check(CudaBackend.GemvSplitCount(70, 8, 2048) == 16, "5070 Ti: the formula gives 16");
        var chosen = Measure(70, 2048, (plainKey, plain5070), (addNormKey, addNorm5070));
        Check(chosen[plainKey] == 8 && chosen[addNormKey] == 16, $"5070 Ti: plain {chosen[plainKey]}, add + norm {chosen[addNormKey]} (8 and 16)");
        var collided = Measure(70, 2048, (plainKey, plain5070), (plainKey, addNorm5070));
        Check(collided[plainKey] == 8, "with the old shared key the fused kernel would run the plain kernel's 8 (8.8 µs, not 7.5)");

        // RTX 3060 Laptop (30 SMs): 6 splits 15 µs, the formula's 8 about 20; 6 was not a candidate before.
        var addNorm3060 = new Dictionary<int, float> { [1] = 40f, [2] = 26f, [3] = 19f, [4] = 17f, [6] = 15f, [8] = 20f, [12] = 19f, [16] = 21f, [24] = 25f, [32] = 29f };
        Check(CudaBackend.GemvSplitCount(30, 8, 2048) == 8, "3060: the formula gives 8");
        Check(Measure(30, 2048, (addNormKey, addNorm3060))[addNormKey] == 6, "3060: 6 splits chosen");

        // A clock that rises 3x part way through the measurement (an idle laptop GPU): with ratios per round the choice
        // is 6 wherever the step falls; medians of the times themselves chose differently by where it fell.
        int[] list = CudaBackend.GemvSplitCandidates(2048, 1);
        int total = list.Length * TuneTiming.Rounds;
        int rawWrong = 0;
        for (int step = 0; step <= total; step++)
        {
            int calls = 0;
            var times = new List<(int C, float T)>();
            float Timer(int c)
            {
                float t = addNorm3060[list[c]] * (calls++ < step ? 3f : 1f);
                times.Add((c, t));
                return t;
            }

            int pick = list[TuneTiming.Choose(list, 8, Timer)];
            Check(pick == 6, $"clock step after {step} of {total} timings: 6 chosen (chose {pick})");
            var raw = Enumerable.Range(0, list.Length).Select(c => TuneTiming.Median([.. times.Where(x => x.C == c).Select(x => x.T)])).ToArray();
            int fastest = Array.IndexOf(raw, raw.Min()), formula = Array.IndexOf(list, 8);
            rawWrong += list[raw[fastest] >= raw[formula] * TuneTiming.Margin ? formula : fastest] != 6 ? 1 : 0;
        }

        Check(rawWrong > 0, $"medians of the raw times are thrown off by some step positions ({rawWrong} of {total + 1})");

        // The warm-up: stops soon on a settled GPU, waits out a clock ramp, and is bounded.
        float steady = TuneTiming.Warm(() => 1f);
        Check(steady >= TuneTiming.WarmMinMs && steady <= TuneTiming.WarmMinMs + 10f, $"settled GPU: about the minimum ({steady} ms)");
        float spent = 0f;
        float ramped = TuneTiming.Warm(() =>
        {
            float t = 1f + 3f * Math.Max(0f, 1f - spent / 60f);                 // 4x slower at first, settled after 60 ms
            spent += t;
            return t;
        });
        Check(ramped >= 60f && ramped <= 100f, $"a ramp of 60 ms is waited out ({ramped} ms)");
        int n = 0;
        float noisy = TuneTiming.Warm(() => n++ / 5 % 2 == 0 ? 1f : 2f);
        Check(noisy >= TuneTiming.WarmMaxMs && noisy < TuneTiming.WarmMaxMs + 3f, $"never settling: stops at the bound ({noisy} ms)");
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

    // Pure (no GPU): shapes and the checks of what kernels need, from synthetic limit records.
    private static void KernelShapesFromLimits(Device device)
    {
        if (device.Backend is CudaBackend cuda)
        {
            // What this GPU reports gives the shapes every CUDA GPU so far gives (1024 threads per block, 32-lane warps).
            Check(cuda.Shapes == KernelShapes.Default, $"this GPU's limits give today's shapes: {cuda.Limits}");
            Check(cuda.Limits.SharedPerBlockOptin >= cuda.Limits.SharedPerBlock && cuda.Limits.MaxGridZ > 0 && cuda.Limits.RegistersPerBlock > 0, $"limits read: {cuda.Limits}");
            return;
        }

        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        // The cards the defaults were measured on, and the other compute capabilities as documented.
        var cards = new (string Name, CudaDeviceLimits Limits)[]
        {
            ("RTX 5070 Ti (12.0, 70 SMs)", CudaDeviceLimits.Documented(12, 0, 70, 16L << 30, 48 << 20)),
            ("RTX 5050 Laptop (12.0, 20 SMs)", CudaDeviceLimits.Documented(12, 0, 20, 8L << 30, 32 << 20)),
            ("RTX 3060 Laptop (8.6, 30 SMs)", CudaDeviceLimits.Documented(8, 6, 30, 6L << 30, 3 << 20)),
            ("8.0", CudaDeviceLimits.Documented(8, 0, 108)),
            ("8.9", CudaDeviceLimits.Documented(8, 9, 76)),
            ("9.0", CudaDeviceLimits.Documented(9, 0, 132)),
            ("10.0", CudaDeviceLimits.Documented(10, 0, 148)),
            ("7.5", CudaDeviceLimits.Documented(7, 5, 40)),
        };
        foreach (var (name, limits) in cards)
        {
            var shapes = KernelShapes.Derive(limits, out string? reason);
            Check(shapes == KernelShapes.Default, $"{name}: today's shapes ({reason})");
            Check(ReferenceEquals(PtxKernels.SourceFor(shapes!), PtxKernels.Source), $"{name}: today's PTX, byte for byte (one module text)");
        }

        var defaults = KernelShapes.Default;
        Check(defaults is { BlockSize: 256, SamplerThreads: 1024, MaxWarpsPerBlock: 32 }, "the defaults are the sizes the kernels had");
        var sharedBytes = PtxKernels.StaticSharedBytes(PtxKernels.Source);
        Check(sharedBytes["sum_f32"] == 256 * 4 && sharedBytes["norm_stats_f32"] == 2 * 256 * 4 && sharedBytes["fill_f32"] == 0, "static shared memory read from the PTX");

        // A device with larger blocks: the shape-generic kernels take them, the others keep their geometry.
        var large = CudaDeviceLimits.Documented(12, 0, 70) with { MaxThreadsPerBlock = 2048 };
        var wide = KernelShapes.Derive(large, out string? largeReason);
        Check(wide is { BlockSize: 512, SamplerThreads: 2048, MaxWarpsPerBlock: 64 }, $"2048 threads per block: larger shapes ({largeReason})");
        string source = PtxKernels.SourceFor(wide!);
        Check(source != PtxKernels.Source && source.Contains("sdata[512]", StringComparison.Ordinal) && source.Contains("s1[512]", StringComparison.Ordinal)
              && source.Contains("_rv[64]", StringComparison.Ordinal) && source.Contains("mad.lo.u32 %r7, %r5, 512, %r6;", StringComparison.Ordinal), "the PTX generated for them");
        var signatures = System.Text.RegularExpressions.Regex.Matches(source, @"\.entry\s+(\w+)\s*\(([^)]*)\)")
            .ToDictionary(m => m.Groups[1].Value, m => System.Text.RegularExpressions.Regex.Count(m.Groups[2].Value, @"\.param\b"));
        Check(signatures.Count == PtxKernels.ParameterCounts.Count && signatures.All(e => PtxKernels.ParameterCounts.TryGetValue(e.Key, out int n) && n == e.Value),
            "the same kernels with the same parameters");
        Check(PtxKernels.Fits(source, large) is null, "and they fit");

        // Devices the kernels cannot run on: a reason, no shapes.
        Check(KernelShapes.Derive(large with { MaxThreadsPerBlock = 512 }, out string? small) is null && small!.Contains("1024", StringComparison.Ordinal), $"512 threads per block: {small}");
        Check(KernelShapes.Derive(large with { WarpSize = 64 }, out string? warp) is null && warp!.Contains("32-lane", StringComparison.Ordinal), $"64-lane warps: {warp}");
        Check(KernelShapes.Derive(large with { SharedPerBlock = 16 << 10, SharedPerBlockOptin = 16 << 10 }, out string? shared) is null && shared!.Contains("shared memory", StringComparison.Ordinal),
            $"16 KiB of shared memory: {shared}");

        // Tensor-core modules: judged by the shared memory a block may use (static, and with opting in).
        var modules = PtxKernels.TensorCoreModules.ToDictionary(m => m.Name, m => m.Source);
        var ampere = CudaDeviceLimits.Documented(8, 6, 30);
        Check(modules.Values.All(m => PtxKernels.Fits(m, ampere) is null), "every tensor-core module fits 8.6 (99 KiB per block)");
        var tight = ampere with { SharedPerBlockOptin = 48 << 10 };
        Check(PtxKernels.Fits(modules["products"], tight) is null, "the products need no more than 48 KiB");
        Check(PtxKernels.Fits(modules["attention d128"], tight) is { } flash && flash.Contains("flash_tc_bwd", StringComparison.Ordinal), "the flash backward (d128) needs more than 48 KiB");
        Check(PtxKernels.Fits(modules["int8 products"], tight) is not null, "the 8-bit products need more than 48 KiB");
        Check(PtxKernels.DynamicSharedBytes("flash_tc_bwd_kv_d64") == PtxKernels.FlashTensorBackwardKvShared(64)
              && PtxKernels.DynamicSharedBytes("flash_tc_bwd_q_d128") == PtxKernels.FlashTensorBackwardQShared(128)
              && PtxKernels.DynamicSharedBytes("gemm8_s8_f32") == PtxKernels.EightBitShared && PtxKernels.DynamicSharedBytes("gemm_tc_nn_f32") == 0,
            "dynamic shared memory per kernel");
    }
}
