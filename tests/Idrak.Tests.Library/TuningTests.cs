// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Abstraction.Diagnostics;
using Idrak.Gpu.Cuda;
using Idrak.Layers;

// Card-dependent choices (k splits, kernel, tile) are measured on the device in use, not taken from one card's benchmarks.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] TuningGroup =
    [
        ("tuning: candidates are compared by their median time, so a lucky timing does not win and a close one keeps the formula's choice; a winner is timed again and kept only if it wins twice (noisy short prompt products)", TuneMedians),
        ("tuning: candidates are timed in pairs with the formula's choice, so a drifting clock does not move the choice (the 151936-column head kept 2-4% slower splits under a rising clock); decoding-sized products are timed with a cold L2 cache", TuneDrift),
        ("tuning: a clock that rises part way through a measurement does not move the choice (ratios of pairs with the formula's choice), and the warm-up stops soon on a settled GPU, waits out a clock ramp and is bounded", GemvSelection),
        ("tuning: the cache file of measured choices round-trips and is ignored when the GPU, driver, library build or format differs", TuningCacheFile),
        ("tuning: decoding attention reads at least a measured number of positions per block (short caches on fewer blocks): every least chunk and split count gives the one-block result; lengths 64 ... capacity and chunk candidates", DecodeMinChunks),
        ("tuning: the power source is read as the system reports it (mains or battery)", PowerSourceKeys),
        ("tuning: sizes that follow the data are keyed by size class (exact up to 32, then four classes a doubling, each within a quarter), so convolutions over batches of every width share a few measured choices", TuneSizeClasses),
        ("tuning: attention over rows of different lengths and decoding attention key the cache's size and the rows by size class: caches of 1000 and 1010 positions (float32, decode and prompt steps) share one measured choice, the second measures nothing and matches the CPU", RowsTuneSizeClasses),
        ("tuning: a convolution or a float32 product at a size class none measured takes the choice of the nearest measured class of its shape within ComputeResources.TuningNeighbourOctaves (IDRAK_TUNE_NEIGHBOUR_OCTAVES) instead of measuring, a class farther away is measured, 0 measures every class, and the lent choice gives the CPU's result; each measured tuning is a TuningCompleted telemetry event", ConvNeighbourClasses),
    ];

    internal static void ConvNeighbourClasses(Device device)
    {
        if (device.Backend is not (CudaBackend or Idrak.Gpu.Vulkan.VulkanBackend))
        {
            return;                                                              // the CPU measures nothing
        }

        var r = new Random(23);
        float[] values = [.. Enumerable.Range(0, 64 * 25000).Select(_ => r.NextSingle() * 2 - 1)];
        float[] Run(Device on, int width)
        {
            using var layer = new Conv1d(64, 64, 5, padding: 2, device: on, random: new Random(5));
            using var x = Tensor.From(values.AsSpan(0, 64 * width), [1, 64, width], on);
            using var y = layer.Forward(x);
            return y.ToArray();
        }

        float[] Product(Device on, int rows)
        {
            using var a = Tensor.From(values.AsSpan(0, rows * 256), [rows, 256], on);
            using var b = Tensor.From(values.AsSpan(1000, 256 * 256), [256, 256], on);
            using var y = a.MatMul(b);
            return y.ToArray();
        }

        float[] Scores(Device on, int n)                                         // attention scores: the length in n, exact in the key
        {
            using var a = Tensor.From(values.AsSpan(0, 256 * 256), [256, 256], on);
            using var b = Tensor.From(values.AsSpan(1000, 256 * n), [256, n], on);
            using var y = a.MatMul(b);
            return y.ToArray();
        }

        var tunings = new TuningCounter();
        var cuda = device.Backend as CudaBackend;
        var vulkan = device.Backend as Idrak.Gpu.Vulkan.VulkanBackend;
        void Forget()
        {
            cuda?.ForgetTuning();
            vulkan?.ForgetTuning();
        }

        bool autotune = CudaBackend.Autotune, persist = CudaBackend.PersistTuning, vulkanAutotune = Idrak.Gpu.Vulkan.VulkanBackend.Autotune;
        string? vulkanCache = Idrak.Gpu.Vulkan.VulkanBackend.TuningCacheFile;
        double reach = ComputeResources.TuningNeighbourOctaves;
        string cache = Path.Combine(Path.GetTempPath(), $"idrak-neighbour-tuning-{Environment.ProcessId}.tsv");
        try
        {
            CudaBackend.PersistTuning = false;                                   // measure here, whatever an earlier run kept
            CudaBackend.Autotune = true;
            Idrak.Gpu.Vulkan.VulkanBackend.Autotune = true;
            Idrak.Gpu.Vulkan.VulkanBackend.TuningCacheFile = cache;
            ComputeResources.TuningNeighbourOctaves = 1;
            Forget();
            using (Telemetry.Subscribe(tunings))
            {
                Run(device, 4000);                                               // width class 4096: measured
                int first = tunings.Convolutions;
                if (first == 0)
                {
                    Skip($"the shape offers one candidate on {device}: nothing to measure");
                }

                var near = Run(device, 5000);                                    // class 5120, a quarter octave away: lent
                Check(tunings.Convolutions == first, $"a class within an octave borrows the choice: {tunings.Convolutions - first} convolution tunings measured for it");
                var cpu = Run(Device.Cpu, 5000);
                AssertClose(near, cpu, 1e-3f * Math.Max(1f, cpu.Max(Math.Abs)), "the lent choice against the CPU");
                Run(device, 20000);                                              // class 20480, 2.3 octaves away: measured
                Check(tunings.Convolutions > first, "a class farther than the reach is measured");

                Product(device, 1000);                                           // rows class 1024
                int tiles = tunings.Count("FloatTile");
                if (tiles > 0)
                {
                    var lent = Product(device, 1200);                            // class 1280: lent
                    Check(tunings.Count("FloatTile") == tiles, $"a product's rows class within an octave borrows the tile: {tunings.Count("FloatTile") - tiles} measured for it");
                    var cpuProduct = Product(Device.Cpu, 1200);
                    AssertClose(lent, cpuProduct, 1e-3f * Math.Max(1f, cpuProduct.Max(Math.Abs)), "the lent tile against the CPU");
                    Product(device, 6000);                                       // class 6144, 2.5 octaves away: measured
                    Check(tunings.Count("FloatTile") > tiles, "a product's rows class farther than the reach is measured");

                    Scores(device, 1000);
                    tiles = tunings.Count("FloatTile");
                    var lentScores = Scores(device, 1300);                       // n a third longer: lent
                    Check(tunings.Count("FloatTile") == tiles, $"a product whose n follows the data borrows the tile within an octave: {tunings.Count("FloatTile") - tiles} measured");
                    var cpuScores = Scores(Device.Cpu, 1300);
                    AssertClose(lentScores, cpuScores, 1e-3f * Math.Max(1f, cpuScores.Max(Math.Abs)), "the lent tile of the scores against the CPU");
                    Scores(device, 3000);                                        // n three times longer: measured
                    Check(tunings.Count("FloatTile") > tiles, "a product whose n is farther than the reach is measured");
                }
                else
                {
                    Console.WriteLine($"    (the float32 product offers one tile on {device}: its lending not exercised)");
                }

                ComputeResources.TuningNeighbourOctaves = 0;
                Forget();
                int before = tunings.Convolutions;
                Run(device, 4000);
                int measured = tunings.Convolutions;
                Run(device, 5000);
                Check(measured > before && tunings.Convolutions > measured, "with a reach of 0 every class is measured");
            }
        }
        finally
        {
            CudaBackend.Autotune = autotune;
            CudaBackend.PersistTuning = persist;
            Idrak.Gpu.Vulkan.VulkanBackend.Autotune = vulkanAutotune;
            Idrak.Gpu.Vulkan.VulkanBackend.TuningCacheFile = vulkanCache;
            ComputeResources.TuningNeighbourOctaves = reach;
            Forget();
            File.Delete(cache);
        }
    }

    // Attention over rows of different lengths (each row from its own start) on a float32 cache of `capacity` positions
    // filled to all but the step's own, with seeded data (the same on every device for one shape).
    internal static float[] RowsAttention(Device on, int capacity, int steps)
    {
        const int heads = 8, dim = 64;
        var random = new Random(capacity * 31 + steps);
        float[] Values(int n) => [.. Enumerable.Range(0, n).Select(_ => (float)(random.NextDouble() * 2 - 1))];
        var backend = on.Backend;
        using var q = Tensor.From(Values(heads * steps * dim), [heads, steps, dim], on);
        using var k = Tensor.From(Values(heads * capacity * dim), [heads, capacity, dim], on);
        using var v = Tensor.From(Values(heads * capacity * dim), [heads, capacity, dim], on);
        using var position = Tensor.From([(float)(capacity - steps)], [1], on);
        using var starts = Tensor.From([.. Enumerable.Range(0, heads * steps).Select(i => (float)(i / steps % 3 * 7))], [heads * steps], on);
        using var y = Tensor.From(new float[heads * steps * dim], [heads, steps, dim], on);
        Check(backend.AttentionRows(q.Storage, k.Storage, v.Storage, position.Storage, y.Storage, starts.Storage, heads, 1, steps, steps, capacity, dim, 0.125f),
            $"float32 rows run on {on}");
        return y.ToArray();
    }

    internal static void RowsTuneSizeClasses(Device device)
    {
        var cuda = device.Backend as CudaBackend;
        var vulkan = device.Backend as Idrak.Gpu.Vulkan.VulkanBackend;
        if (cuda is null && vulkan is null)
        {
            return;                                                              // the CPU measures nothing
        }

        bool autotune = CudaBackend.Autotune, persist = CudaBackend.PersistTuning, vulkanAutotune = Idrak.Gpu.Vulkan.VulkanBackend.Autotune;
        string? vulkanCache = Idrak.Gpu.Vulkan.VulkanBackend.TuningCacheFile;
        string cache = Path.Combine(Path.GetTempPath(), $"idrak-rows-tuning-{Environment.ProcessId}.tsv");
        var tunings = new TuningCounter();
        try
        {
            CudaBackend.PersistTuning = false;                                   // measure here, whatever an earlier run kept
            CudaBackend.Autotune = true;
            Idrak.Gpu.Vulkan.VulkanBackend.Autotune = true;
            Idrak.Gpu.Vulkan.VulkanBackend.TuningCacheFile = cache;
            cuda?.ForgetTuning();
            vulkan?.ForgetTuning();
            using (Telemetry.Subscribe(tunings))
            {
                foreach (int steps in new[] { 1, 9 })
                {
                    string what = $"float32 rows, {(steps == 1 ? "a decoding step" : $"{steps} prompt steps")}";
                    RowsAttention(device, 1000, steps);                          // class 1024: measured where the shape offers choices
                    int first = tunings.Total;
                    var near = RowsAttention(device, 1010, steps);
                    Check(tunings.Total == first, $"{what}: a cache of 1010 positions (class 1024) after 1000 measured {tunings.Total - first} choices again");
                    var cpu = RowsAttention(Device.Cpu, 1010, steps);
                    AssertClose(near, cpu, 1e-2f * Math.Max(1f, cpu.Max(Math.Abs)), $"{what}: the class-mate's choice against the CPU");
                }
            }
        }
        finally
        {
            CudaBackend.Autotune = autotune;
            CudaBackend.PersistTuning = persist;
            Idrak.Gpu.Vulkan.VulkanBackend.Autotune = vulkanAutotune;
            Idrak.Gpu.Vulkan.VulkanBackend.TuningCacheFile = vulkanCache;
            cuda?.ForgetTuning();
            vulkan?.ForgetTuning();
            if (File.Exists(cache))
            {
                File.Delete(cache);
            }
        }
    }

    internal sealed class TuningCounter : ITelemetryHook
    {
        private readonly Dictionary<string, int> _counts = new(StringComparer.Ordinal);

        public TelemetryLevel Levels => TelemetryLevel.Devices;

        public int Convolutions => Count("Convolution");

        public int Total => _counts.Values.Sum();

        public int Count(string operation) => _counts.GetValueOrDefault(operation);

        public void OnTuning(in TuningCompleted e) => _counts[e.Operation] = Count(e.Operation) + 1;
    }

    // Pure (no GPU): the comparison with an injected timer.
    internal static void TuneMedians(Device device)
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

        // Noisy timings, shaped like the prompt splits --bench-gemv logged on a 70-SM card (180 rows, 1024 -> 1024, 64-row
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
    // lowering them under a power limit) does not move the choice. Shaped like the 151936-column head on a 20-SM
    // card ( the formula's 1 split): --bench-gemv found 1 split fastest (657 µs; 2 splits 765), yet the
    // measured choice ran 7-9% slower.
    internal static void TuneDrift(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        int[] list = [1, 2, 3, 4, 6, 8, 12, 16];                                // the head's split counts (k 1024)
        int formula = 1;                                                        // the head on 20 SMs: 1 split
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

    // Pure (no GPU): TuneTiming's choice under a clock step and its warm-up, with fake timers shaped like --bench-gemv's
    // results on a 30-SM card.
    internal static void GemvSelection(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        // Times shaped like a 30-SM card's: 6 splits 15 µs, the formula's 8 about 20.
        var addNorm30 = new Dictionary<int, float> { [1] = 40f, [2] = 26f, [3] = 19f, [4] = 17f, [6] = 15f, [8] = 20f, [12] = 19f, [16] = 21f, [24] = 25f, [32] = 29f };

        // A clock that rises 3x part way through the measurement (an idle laptop GPU): with ratios of pairs the choice
        // is 6 wherever the step falls; medians of the times themselves chose differently by where it fell.
        int[] list = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32];                         // the split counts of k 2048
        int total = 0;
        TuneTiming.Choose(list, 8, _ => { total++; return 1f; });
        int rawWrong = 0;
        for (int step = 0; step <= total; step++)
        {
            int calls = 0;
            var times = new List<(int C, float T)>();
            float Timer(int c)
            {
                float t = addNorm30[list[c]] * (calls < step ? 3f : 1f);
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
    internal static void TuningCacheFile(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        var identity = new TuningIdentity("NVIDIA Test GPU; compute 8.6; 30 SMs; 6144 MiB", "CUDA 13000; 580.82.07", "0.1.7+abc; kernels 0123456789abcdef");
        var entries = new Dictionary<TuneKey, int>
        {
            [new TuneKey(TuneOp.Convolution, 9, 256, 256, 4096, 256, -1, 7)] = 6,
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
        var unknown = TuningCache.Parse(text + "NoSuchOp 0 1 2 3 0 0 0 = 4\nDecodeSplits 0 1 x 3 0 0 0 = 4\nDecodeSplits 1 1 2 3 0 0 0 = 8\n", identity);
        Check(unknown?.Count == entries.Count + 1, "lines that do not parse are skipped, the rest kept");

        string name = TuningCache.FileName("Vendor GPU 1 Laptop; compute 8.6; 30 SMs; 6144 MiB");
        Check(name == "Vendor_GPU_1_Laptop_compute_8.6_30_SMs_6144_MiB.txt", $"file name ({name})");
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
            var more = new Dictionary<TuneKey, int> { [new TuneKey(TuneOp.FloatTile, 0, 256, 256, 4096, 1)] = 128, [new TuneKey(TuneOp.DecodeSplits, 2, 16, 4096, 128)] = 8 };
            Check(TuningCache.Save(path, identity, more), "saved again");
            loaded = TuningCache.Load(path, identity);
            Check(loaded.Count == entries.Count + 1 && loaded[new TuneKey(TuneOp.DecodeSplits, 2, 16, 4096, 128)] == 8, "merged with what the file held");

            // A driver update: the old file is ignored and rewritten with the new identity only.
            var updated = identity with { Driver = "CUDA 13010; 590.10" };
            Check(TuningCache.Load(path, updated).Count == 0, "after a driver update nothing is kept");
            Check(TuningCache.Save(path, updated, new Dictionary<TuneKey, int> { [new TuneKey(TuneOp.FloatTile, 0, 180, 1024, 1024)] = 64 }), "rewritten");
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

    internal static void TuneSizeClasses(Device device)
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

    internal static void PowerSourceKeys(Device device)
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
    }

    // The least positions per block of decoding attention (CudaBackend.DecodeSplit): any least chunk with any split count
    // gives the result of one block reading every position (the blocks past the filled length add empty parts), for the
    // float32 cache and filled lengths from one position to most of the capacity.
    internal static void DecodeMinChunks(Device device)
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
        float[] Words(int n) => [.. Enumerable.Range(0, n).Select(_ => r.NextSingle() * 2 - 1)];
        int? splits = CudaBackend.DecodeSplits, chunk = CudaBackend.DecodeMinChunk;
        try
        {
            {
                using var cache = new KeyValueCache(rows, capacity, dim, device, KeyValueFormat.Float32);
                device.Backend.Upload(Words(cache.Keys.Size), cache.Keys.Storage);
                device.Backend.Upload(Words(cache.Values.Size), cache.Values.Storage);

                using var q = Tensor.From([.. Enumerable.Range(0, rows * 2 * dim).Select(_ => r.NextSingle() - 0.5f)], [rows, 2, dim], device);
                float[] Attend(int length)
                {
                    using var position = Tensor.From([length - 1f], [1], device);
                    using var y = Tensor.AttentionDecode(q, cache, position, 1, 0.125f);
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
                            AssertClose(one, Attend(length), 1e-4f * Math.Max(1f, one.Max(Math.Abs)), $"{length} positions, {s} splits, at least {least} per block");
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
