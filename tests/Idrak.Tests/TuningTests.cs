// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Gpu.Cuda;
using Idrak.Layers;

// Card-dependent choices (k splits, kernel, tile) are measured on the device in use, not taken from one card's benchmarks.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] TuningGroup =
    [
        ("tuning: card-dependent choices (few-row kernel and splits, prompt splits, tensor-core splits, tiles, decoding-attention splits) are measured on the device and give the results of the formulas' choices", TunedChoicesMatch),
        ("tuning: candidates are compared by their median time, so a lucky timing does not win and a close one keeps the formula's choice; a winner is timed again and kept only if it wins twice (noisy short prompt products)", TuneMedians),
        ("tuning: candidates are timed in pairs with the formula's choice, so a drifting clock does not move the choice (the 151936-column head kept 2-4% slower splits under a rising clock); decoding-sized products are timed with a cold L2 cache", TuneDrift),
        ("tuning: few-row products are measured over 1, 2, 3, 4, 6, 8, 12, ... splits with the kernel that then runs (fused add-and-normalize kept apart from the plain kernel), after a warm-up, in pairs with the formula's choice", GemvSelection),
        ("tuning: the cache file of measured choices round-trips and is ignored when the GPU, driver, library build or format differs", TuningCacheFile),
        ("tuning: measured choices are kept per GPU in the cache folder and read back instead of measured again", TuningPersisted),
        ("tuning: decoding attention reads at least a measured number of positions per block (short caches on fewer blocks): every least chunk, split count and cache format gives the one-block result; lengths 64 ... capacity and chunk candidates", DecodeMinChunks),
        ("tuning: every backend keeps one set of measured choices per power source (mains or battery, as the system reports it)", PowerSourceKeys),
        ("tuning: sizes that follow the data are keyed by size class (exact up to 32, then four classes a doubling, each within a quarter), so convolutions over batches of every width share a few measured choices", TuneSizeClasses),
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

            // Decoding attention over 700 of 1024 cached positions (splits measured over filled lengths 64 ... 1024).
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

        // What --bench-gemv showed on "o + add + norm": 16 splits 6.6 µs, 8 splits 7.7 µs, but three timings of 16 came out
        // high and one of 4 low. The best time of each picked 4; the median of the ratios picks 16.
        int[] candidates = [1, 2, 4, 8, 16];
        float[] steady = [20f, 12f, 9f, 7.7f, 6.6f];
        var outliers = new Dictionary<(int C, int Nth), float> { [(4, 1)] = 11f, [(4, 3)] = 12f, [(4, 5)] = 11.5f, [(2, 2)] = 5f };
        var timed = new int[candidates.Length];
        var order = new List<int>();
        float Time(int c)
        {
            order.Add(c);
            return outliers.TryGetValue((c, timed[c]++), out float t) ? t : steady[c];
        }

        int chosen = TuneTiming.Choose(candidates, 8, Time);
        Check(candidates[chosen] == 16, $"the median picks 16 splits (picked {candidates[chosen]})");
        Check(timed.Select((n, c) => c switch { 3 => n == TuneTiming.Rounds * 5, 4 => n == TuneTiming.Rounds * 2, _ => n == TuneTiming.Rounds }).All(x => x),
            $"every candidate timed once per round, the formula's choice once with each, the winner again in as many new pairs ({string.Join(",", timed)})");
        Check(order.Take(8).SequenceEqual([3, 0, 3, 1, 3, 2, 3, 4]) && order.Skip(8).Take(8).SequenceEqual([4, 3, 2, 3, 1, 3, 0, 3]),
            "in pairs with the formula's choice: first, forwards, then second, backwards");

        // Within the margin of the formula's choice: the formula's choice stays.
        int close = TuneTiming.Choose([8, 16], 8, c => c == 0 ? 7.7f : 7.6f);
        Check(close == 0, "a candidate less than 3% faster does not replace the formula's choice");
        int clear = TuneTiming.Choose([8, 16], 8, c => c == 0 ? 7.7f : 6.6f);
        Check(clear == 1, "a clearly faster candidate replaces it");
        int noFormula = TuneTiming.Choose([2, 4], 3, c => c == 0 ? 5f : 4.9f);
        Check(noFormula == 1, "the fastest median wins when the formula's choice is not a candidate");

        Check(TuneTiming.Median([3f, 1f, 2f]) == 2f && TuneTiming.Median([4f, 1f, 3f, 2f]) == 2.5f, "median of odd and even counts");
        var medians = new float[candidates.Length];
        TuneTiming.Choose(candidates, 8, c => steady[c], medians: medians);
        Check(MathF.Abs(medians[4] - 6.6f / 7.7f) < 1e-5f && medians[3] == 1f, "the medians reported for the log: time over the formula's choice's");
        var confirmation = new float[1];
        TuneTiming.Choose(candidates, 8, c => steady[c], confirmation: confirmation);
        Check(MathF.Abs(confirmation[0] - 6.6f / 7.7f) < 1e-5f, "the winner's median when timed again, for the log");
        TuneTiming.Choose([8, 16], 8, c => c == 0 ? 7.7f : 7.6f, confirmation: confirmation);
        Check(float.IsNaN(confirmation[0]), "nothing timed again when the formula's choice stays");

        // Noisy timings, shaped like the prompt splits --bench-gemv logged on an RTX 5070 Ti (180 rows, 1024 -> 1024, 64-row
        // tiles, warm, about 13 µs): the formula's 4 splits, 6 splits 6% faster, the others as fast or slower, and each
        // timing off by up to 20% either way. Two processes kept 2 and 3 splits; an earlier one 1 (22 µs against 12.7).
        int[] prompt = [1, 2, 3, 4, 6, 8];
        float[] truth = [1.6f, 1.02f, 1.0f, 1f, 0.94f, 1.06f];
        int slowerOnce = 0, slowerConfirmed = 0, gainOnce = 0, gainConfirmed = 0;
        const int trials = 400;
        for (int seed = 0; seed < trials; seed++)
        {
            foreach (bool confirm in new[] { false, true })
            {
                var noise = new Random(seed);
                int pick = TuneTiming.Choose(prompt, 4, c => truth[c] * (0.8f + 0.4f * noise.NextSingle()), confirm: confirm);
                bool slower = truth[pick] > 1f, gain = truth[pick] < 1f;
                (slowerOnce, slowerConfirmed) = confirm ? (slowerOnce, slowerConfirmed + (slower ? 1 : 0)) : (slowerOnce + (slower ? 1 : 0), slowerConfirmed);
                (gainOnce, gainConfirmed) = confirm ? (gainOnce, gainConfirmed + (gain ? 1 : 0)) : (gainOnce + (gain ? 1 : 0), gainConfirmed);
            }
        }

        Console.WriteLine($"    noisy prompt splits ({trials} measurements): slower than the formula's choice {slowerOnce} once, {slowerConfirmed} confirmed; the 6% gain {gainOnce} once, {gainConfirmed} confirmed");
        Check(slowerConfirmed * 4 <= slowerOnce && slowerConfirmed <= trials / 20,
            $"timed again, a candidate slower than the formula's choice is rarely kept ({slowerConfirmed} of {trials}, {slowerOnce} without)");
        Check(gainConfirmed * 2 >= gainOnce, $"and most real gains still are ({gainConfirmed}, {gainOnce} without)");
    }

    // Pure (no GPU): a clock that drifts through the whole measurement (a laptop GPU still raising its clocks, or one
    // lowering them under a power limit) does not move the choice. Shaped like the 151936-column head on an RTX 5050
    // Laptop (20 SMs, the formula's 1 split): --bench-gemv found 1 split fastest (657 µs; 2 splits 765), yet the
    // measured choice ran 7-9% slower.
    private static void TuneDrift(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        int[] list = CudaBackend.GemvSplitCandidates(1024, 1);
        Check(list.SequenceEqual([1, 2, 3, 4, 6, 8, 12, 16]), "head (k 1024): 1 … 16");
        int formula = CudaBackend.GemvSplitCount(20, (151936 / 4 + 31) / 32, 1024);
        Check(formula == 1, "head on 20 SMs: the formula gives 1 split");
        var head = new Dictionary<int, float> { [1] = 657f, [2] = 765f, [3] = 700f, [4] = 690f, [6] = 684f, [8] = 680f, [12] = 676f, [16] = 672f };

        // The comparison before (each candidate once per round against the reference's timing of that round, forwards
        // then backwards, seven rounds), kept here to show what the pairs fix.
        static int Before(int[] candidates, int fallback, Func<int, float> time)
        {
            const int rounds = 7;
            var samples = new float[candidates.Length, rounds];
            for (int round = 0; round < rounds; round++)
            {
                for (int i = 0; i < candidates.Length; i++)
                {
                    int c = round % 2 == 0 ? i : candidates.Length - 1 - i;
                    samples[c, round] = time(c);
                }
            }

            int reference = Math.Max(Array.IndexOf(candidates, fallback), 0);
            var medians = Enumerable.Range(0, candidates.Length).Select(c => c == reference ? 1f
                : TuneTiming.Median([.. Enumerable.Range(0, rounds).Select(r => samples[c, r] / samples[reference, r])])).ToArray();
            int fastest = Array.IndexOf(medians, medians.Min());
            return medians[fastest] >= medians[reference] * TuneTiming.Margin ? reference : fastest;
        }

        foreach (float drift in new[] { 0.99f, 0.995f, 1.005f, 1.01f })                // per timing: rising or falling clocks
        {
            float clock = 1f;
            float Timer(int c) => head[list[c]] * (clock *= drift);
            int pick = list[TuneTiming.Choose(list, formula, Timer)];
            Check(pick == 1, $"clock x{drift} per timing: 1 split kept (chose {pick})");
            clock = 1f;
            int before = list[Before(list, formula, Timer)];
            if (drift == 0.99f)
            {
                Check(before != 1, $"the comparison before kept {before} splits under a rising clock, 1-4% slower than 1");
            }
        }

        // And a candidate clearly faster still wins under the drift.
        var faster = new Dictionary<int, float>(head) { [3] = 600f };
        float rising = 1f;
        Check(list[TuneTiming.Choose(list, formula, c => faster[list[c]] * (rising *= 0.99f))] == 3, "a candidate 9% faster wins under a rising clock");

        // Cold timings (decoding-sized products): the L2 read before each run is twice the size the device reports, and
        // the runs per timing are fewer, since each then costs that read as well.
        Check(CudaBackend.L2FlushFloats(3 << 20) == 2L * (3 << 20) / 4 && CudaBackend.L2FlushFloats(48 << 20) == 2L * (48 << 20) / 4, "flush: twice the L2 size");
        Check(CudaBackend.L2FlushFloats(0) == 0, "no L2 size reported: warm timings");
        Check(CudaBackend.TimingRepeats(0.66f, cold: true) == 1 && CudaBackend.TimingRepeats(0.66f, cold: false) == 1, "a 0.66 ms head: one run per timing");
        Check(CudaBackend.TimingRepeats(0.007f, cold: false) == 16 && CudaBackend.TimingRepeats(0.007f, cold: true) == 4, "7 µs: 16 back to back, 4 cold");
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

        // A clock that rises 3x part way through the measurement (an idle laptop GPU): with ratios of pairs the choice
        // is 6 wherever the step falls; medians of the times themselves chose differently by where it fell.
        int[] list = CudaBackend.GemvSplitCandidates(2048, 1);
        int total = 0;
        TuneTiming.Choose(list, 8, _ => { total++; return 1f; });
        int rawWrong = 0;
        for (int step = 0; step <= total; step++)
        {
            int calls = 0;
            var times = new List<(int C, float T)>();
            float Timer(int c)
            {
                float t = addNorm3060[list[c]] * (calls < step ? 3f : 1f);
                if (calls++ < total)
                {
                    times.Add((c, t));                                          // the first pass (not the winner's timings again)
                }

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

        // The module parsers (span searches) read what the regular expressions they replaced read, on every module.
        foreach (string module in (string[])[PtxKernels.Source, source, .. PtxKernels.TensorCoreModules.Select(m => m.Source)])
        {
            var expected = new Dictionary<string, int>();
            var entries = System.Text.RegularExpressions.Regex.Matches(module, @"\.entry\s+(\w+)\s*\(");
            for (int i = 0; i < entries.Count; i++)
            {
                int end = i + 1 < entries.Count ? entries[i + 1].Index : module.Length;
                expected[entries[i].Groups[1].Value] = System.Text.RegularExpressions.Regex.Matches(module[entries[i].Index..end], @"\.shared\s+(?:\.align\s+\d+\s+)?\.(\w+)\s+\w+\[(\d+)\]")
                    .Sum(a => (a.Groups[1].Value switch { "b8" or "u8" or "s8" => 1, "b16" or "u16" or "s16" or "f16" => 2, "b64" or "u64" or "s64" or "f64" => 8, _ => 4 })
                              * int.Parse(a.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture));
            }

            var read = PtxKernels.StaticSharedBytes(module);
            var wrong = expected.Where(e => !read.TryGetValue(e.Key, out int b) || b != e.Value).Select(e => $"{e.Key}: {(read.TryGetValue(e.Key, out int b) ? b : -1)} read, {e.Value} declared").ToList();
            Check(read.Count == expected.Count && wrong.Count == 0, $"static shared memory per kernel as declared ({read.Count} kernels read, {expected.Count} declared): {string.Join("; ", wrong.Take(5))}");
            var counts = PtxKernels.ParameterCountsOf(module);
            Check(counts.Count == entries.Count && counts.All(c => CudaKernel.ReadParameters(module, c.Key)?.Length == c.Value),
                $"every kernel's parameters read alike by the module's counts and the launch check ({counts.Count} kernels)");
        }

        // Kernels written for a block larger than the row kernels' declare it (.maxntid, between the parameters and the
        // body), so ptxas keeps their registers within what that block may use; the sampler's follows the shapes.
        static int MaxThreads(string ptx, string kernel)
        {
            var entry = PtxKernels.Entries(ptx).Single(e => e.Name == kernel);
            var after = ptx.AsSpan(entry.Parameters!.Value.End.Value + 1).TrimStart();
            return after.StartsWith(".maxntid ", StringComparison.Ordinal)
                ? int.Parse(after[".maxntid ".Length..after.IndexOf(',')], System.Globalization.CultureInfo.InvariantCulture)
                : 0;
        }

        var declared = new (string Kernel, string Ptx, int Threads)[]
        {
            ("gemv_nn_f32", PtxKernels.Source, PtxKernels.GemvThreads), ("gemv_multi_f32", PtxKernels.Source, PtxKernels.GemvThreads),
            ("int8_gemv_f32", PtxKernels.Source, PtxKernels.Int8GemvThreads), ("int4_gemv_multi_act_f32", PtxKernels.Source, PtxKernels.Int8GemvThreads),
            ("softmax_ce_rows_f32", PtxKernels.Source, PtxKernels.SoftmaxCrossEntropyThreads),
            ("sample_rows_f32", PtxKernels.Source, defaults.SamplerThreads), ("sample_rows_f32", source, wide!.SamplerThreads),
        };
        var undeclared = declared.Where(d => MaxThreads(d.Ptx, d.Kernel) != d.Threads).Select(d => $"{d.Kernel}: .maxntid {MaxThreads(d.Ptx, d.Kernel)}, launched with {d.Threads}").ToList();
        Check(undeclared.Count == 0, $"kernels written for a fixed block declare it: {string.Join("; ", undeclared)}");

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

    private static void TuneSizeClasses(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                              // nothing device-specific: once
        }

        for (int size = 0; size <= Idrak.Gpu.TuneSizes.Exact; size++)
        {
            Check(Idrak.Gpu.TuneSizes.Class(size) == size, $"{size} is its own class");
        }

        int[] expected = [40, 40, 48, 56, 64, 80, 1024, 1280, 1280, 1536];
        int[] sizes = [33, 40, 41, 50, 64, 65, 1000, 1025, 1280, 1281];
        for (int i = 0; i < sizes.Length; i++)
        {
            Check(Idrak.Gpu.TuneSizes.Class(sizes[i]) == expected[i], $"class of {sizes[i]}: {Idrak.Gpu.TuneSizes.Class(sizes[i])}, expected {expected[i]}");
        }

        int classes = 0, previous = -1;
        for (int size = 1; size <= 1 << 20; size++)
        {
            int c = Idrak.Gpu.TuneSizes.Class(size);
            Check(c >= size && c <= size + Math.Max(0, size / 4), $"class of {size} ({c}) within a quarter above it");
            Check(c >= previous, "classes never decrease");
            Check(Idrak.Gpu.TuneSizes.Class(c) == c, $"a class ({c}) is its own class");
            classes += c != previous ? 1 : 0;
            previous = c;
        }

        Check(classes == Idrak.Gpu.TuneSizes.Exact + 4 * 15, $"{classes} classes up to 2^20: the exact sizes and four a doubling above them");

        // Lines of text at height 48: batches 830 and 890 pixels wide (class 896) share a convolution's key (one
        // measurement); 1100 pixels and a batch of 14 lines (small batches are exact) do not; a model's sizes (channels,
        // filters, the window) stay exact.
        var narrow = new ConvGeometry(16, 32, 48, 830, 3, 3, 1, 1, 1, 1);
        Check(Idrak.Gpu.ConvolutionShapes.Key(in narrow, 64, 1, out var a), "a key");
        Check(Idrak.Gpu.ConvolutionShapes.Key(narrow with { W = 890 }, 64, 1, out var b) && a == b, "830 and 890 wide: one key");
        Check(Idrak.Gpu.ConvolutionShapes.Key(narrow with { N = 14 }, 64, 1, out var c0) && c0 != a, "14 lines: another key");
        Check(Idrak.Gpu.ConvolutionShapes.Key(narrow with { W = 1100 }, 64, 1, out var c1) && c1 != a, "1100 wide: another key");
        Check(Idrak.Gpu.ConvolutionShapes.Key(narrow with { C = 33 }, 64, 1, out var c2) && c2 != a, "another channel count: another key");
        Check(Idrak.Gpu.ConvolutionShapes.Key(in narrow, 65, 1, out var c3) && c3 != a, "another filter count: another key");
    }

    private static void PowerSourceKeys(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                              // nothing device-specific: once
        }

        // The Linux report: a discharging battery and no supply online is battery power; anything else is mains.
        string root = Path.Combine(Path.GetTempPath(), "idrak-power-" + Guid.NewGuid().ToString("N"));
        void Supply(string name, string type, string file, string value)
        {
            Directory.CreateDirectory(Path.Combine(root, name));
            File.WriteAllText(Path.Combine(root, name, "type"), type + "\n");
            File.WriteAllText(Path.Combine(root, name, file), value + "\n");
        }

        try
        {
            Check(Idrak.Abstraction.Devices.PowerSource.Linux(Path.Combine(root, "missing")) == "ac", "no report: mains");
            Supply("BAT0", "Battery", "status", "Discharging");
            Check(Idrak.Abstraction.Devices.PowerSource.Linux(root) == "battery", "a discharging battery, no supply online: battery");
            Supply("AC", "Mains", "online", "1");
            Check(Idrak.Abstraction.Devices.PowerSource.Linux(root) == "ac", "mains online: mains");
            File.WriteAllText(Path.Combine(root, "AC", "online"), "0\n");
            Supply("ucsi", "USB_C", "online", "1");
            Check(Idrak.Abstraction.Devices.PowerSource.Linux(root) == "ac", "USB-C supply online: mains");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }

        // The cache keys differ by power source.
        string? before = Environment.GetEnvironmentVariable("IDRAK_POWER_SOURCE");
        try
        {
            Environment.SetEnvironmentVariable("IDRAK_POWER_SOURCE", "ac");
            string ac = Idrak.Abstraction.Devices.Cpu.CpuTuning.CacheKey(4);
            Environment.SetEnvironmentVariable("IDRAK_POWER_SOURCE", "battery");
            string battery = Idrak.Abstraction.Devices.Cpu.CpuTuning.CacheKey(4);
            Check(ac != battery && ac.EndsWith("power ac", StringComparison.Ordinal) && battery.EndsWith("power battery", StringComparison.Ordinal),
                $"CPU cache keys per power source ({ac} / {battery})");
        }
        finally
        {
            Environment.SetEnvironmentVariable("IDRAK_POWER_SOURCE", before);
        }
    }

    // The least positions per block of decoding attention (CudaBackend.DecodeSplit): any least chunk with any split count
    // gives the result of one block reading every position (the blocks past the filled length add empty parts), for the
    // float32, int8 and bfloat16 caches and filled lengths from one position to most of the capacity.
    private static void DecodeMinChunks(Device device)
    {
        if (device.Type == DeviceType.Cpu)
        {
            Check(CudaBackend.DecodeTuneLengths(4096).SequenceEqual([64, 128, 256, 512, 1024, 2048, 4096]), "lengths: 64 ... 4096");
            Check(CudaBackend.DecodeTuneLengths(1000).SequenceEqual([64, 128, 256, 512, 1000]), "a capacity that is no power of two comes last");
            Check(CudaBackend.DecodeTuneLengths(32).SequenceEqual([32]), "a small capacity: itself");
            Check(CudaBackend.DecodeMinChunks(4096, 24).SequenceEqual([1, 8, 16, 32, 64, 128]), "4096 positions, 24 splits: 1, then 8 ... 128 (a full cache keeps all 24 blocks busy)");
            Check(CudaBackend.DecodeMinChunks(1024, 16).SequenceEqual([1, 8, 16, 32]), "1024 positions, 16 splits: up to 32");
            Check(CudaBackend.DecodeMinChunks(64, 8).SequenceEqual([1]), "64 positions, 8 splits: plain chunks only");
            return;
        }

        if (device.Backend is not CudaBackend)
        {
            return;
        }

        const int rows = 8, capacity = 1024, dim = 64;
        var r = new Random(5);
        float[] Words(int n, KeyValueFormat format) => [.. Enumerable.Range(0, n).Select(_ => format switch
        {
            KeyValueFormat.Float32 => r.NextSingle() * 2 - 1,
            KeyValueFormat.BFloat16 => BitConverter.Int32BitsToSingle((int)((BitConverter.SingleToUInt32Bits(r.NextSingle() - 0.5f) >> 16)
                | (BitConverter.SingleToUInt32Bits(r.NextSingle() - 0.5f) & 0xFFFF0000u))),
            _ => BitConverter.Int32BitsToSingle(r.Next() & 0x7F7F7F7F),
        })];
        int? splits = CudaBackend.DecodeSplits, chunk = CudaBackend.DecodeMinChunk;
        try
        {
            foreach (var format in new[] { KeyValueFormat.Float32, KeyValueFormat.Int8, KeyValueFormat.BFloat16 })
            {
                using var cache = new KeyValueCache(rows, capacity, dim, device, format);
                device.Backend.Upload(Words(cache.Keys.Size, format), cache.Keys.Storage);
                device.Backend.Upload(Words(cache.Values.Size, format), cache.Values.Storage);
                if (format == KeyValueFormat.Int8)
                {
                    device.Backend.Upload([.. Enumerable.Repeat(0.01f, cache.KeyScales!.Size)], cache.KeyScales.Storage);
                    device.Backend.Upload([.. Enumerable.Repeat(0.01f, cache.ValueScales!.Size)], cache.ValueScales.Storage);
                }

                using var q = Tensor.From([.. Enumerable.Range(0, rows * 2 * dim).Select(_ => r.NextSingle() - 0.5f)], [rows, 2, dim], device);
                float[] Attend(int length)
                {
                    using var position = Tensor.From([length - 1f], [1], device);
                    using var y = format switch
                    {
                        KeyValueFormat.Float32 => Tensor.AttentionDecode(q, cache, position, 1, 0.125f),
                        KeyValueFormat.Int8 => Tensor.AttentionInt8(q, cache, position, 1, 0.125f, tiled: false),
                        _ => Tensor.AttentionBFloat16(q, cache, position, 1, 0.125f, tiled: false),
                    };
                    return y.ToArray();
                }

                foreach (int length in new[] { 1, 5, 70, 700 })
                {
                    (CudaBackend.DecodeSplits, CudaBackend.DecodeMinChunk) = (1, 1);
                    var one = Attend(length);
                    foreach (int s in new[] { 4, 16 })
                    {
                        foreach (int least in new[] { 1, 8, 64, 512 })
                        {
                            (CudaBackend.DecodeSplits, CudaBackend.DecodeMinChunk) = (s, least);
                            AssertClose(one, Attend(length), 1e-4f * Math.Max(1f, one.Max(Math.Abs)), $"{format}, {length} positions, {s} splits, at least {least} per block");
                        }
                    }
                }
            }
        }
        finally
        {
            (CudaBackend.DecodeSplits, CudaBackend.DecodeMinChunk) = (splits, chunk);
        }
    }
}
