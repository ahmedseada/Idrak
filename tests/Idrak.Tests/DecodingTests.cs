// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Generation;
using Idrak.Layers;
using Idrak;

// Tests for fused inference kernels, KV-cache decoding, on-device sampling and compute graphs.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] Decoding =
    [
        ("fused kernels match their unfused equivalents", FusedKernels),
        ("KV-cache decoding matches the full forward pass at every position", CachedDecodingMatchesFull),
        ("batched cached decoding matches sequences decoded one by one", BatchedDecoding),
        ("graph replay gives the same tokens as direct execution", GraphReplay),
        ("sampler: distribution, top-k, temperature, determinism, CPU parity", SamplerBehaviour),
        ("sampler: top-p, min-p, repeat/presence/frequency penalties, history", SamplerFilters),
        ("sampler: the CPU kernel's one-pass top-k, top-p and top-5 equal the scan-per-choice reference exactly", SamplerSinglePass),
        ("cuda: every kernel's declared parameter count is known (launch argument check)", KernelSignatures),
    ];

    private static Sequential TinyGpt(Device device, int vocabulary = 11, int dim = 16, int context = 12)
    {
        var random = new Random(50);
        var model = new Sequential
        {
            new Embedding(vocabulary, dim, device, random),
            new PositionalEncoding(context, dim, device),
            new TransformerEncoderLayer(dim, 4, ffDim: 32, dropout: 0f, causal: true, device: device, random: random),
            new TransformerEncoderLayer(dim, 4, ffDim: 32, dropout: 0f, causal: true, device: device, random: random),
            new LayerNorm(dim, device: device),
            new Linear(dim, vocabulary, device: device, random: random),
        };
        model.Eval();
        return model;
    }

    private static void FusedKernels(Device device)
    {
        var random = new Random(51);
        using var x = Tensor.From(RandomArray(random, 3 * 4 * 6, 3f), [3, 4, 6], device);
        using var mask = Tensor.From(RandomArray(random, 4 * 6, 2f), [4, 6], device);
        float[] reference, fused;
        using (Autograd.NoGrad())
        {
            fused = x.ScaleMaskSoftmax(0.7f, mask).ToArray();
        }

        reference = (x * 0.7f + mask).Softmax().ToArray();
        AssertClose(reference, fused, 1e-5f, "scale + mask + softmax");

        var norm = new LayerNorm(6, device: device);
        norm.Gamma.Load(RandomArray(random, 6, 2f));
        norm.Beta.Load(RandomArray(random, 6));
        reference = norm.Forward(x).ToArray();                  // autograd on: the three-kernel path
        using (Autograd.NoGrad())
        {
            fused = norm.Forward(x).ToArray();                  // inference: the fused kernel
        }

        AssertClose(reference, fused, 1e-4f, "fused LayerNorm");

        using var bias = Tensor.From(RandomArray(random, 6), device);
        reference = (x + bias).Gelu().ToArray();
        using (Autograd.NoGrad())
        {
            fused = x.BiasGelu(bias).ToArray();
        }

        AssertClose(reference, fused, 1e-5f, "bias + GELU");
    }

    private static void CachedDecodingMatchesFull(Device device)
    {
        const int T = 10, V = 11;
        using var model = TinyGpt(device);
        var ids = new float[T];
        var random = new Random(52);
        for (int t = 0; t < T; t++)
        {
            ids[t] = random.Next(V);
        }

        using (Autograd.NoGrad())
        {
            using var sequence = Tensor.From(ids, [1, T], device);
            var full = model.Forward(sequence).ToArray();                                   // [1, T, V]

            using var context = new DecodingContext(device, 1, 12);
            using var prompt = Tensor.From(ids.AsSpan(0, 4), [1, 4], device);
            var prefill = model.ForwardCached(prompt, context).ToArray();                   // positions 0-3 at once
            AssertClose(full[..(4 * V)], prefill, 1e-4f, "prefill logits");
            for (int t = 4; t < T; t++)
            {
                using var next = Tensor.From(ids.AsSpan(t, 1), [1, 1], device);
                var step = model.ForwardCached(next, context).ToArray();                    // one new position
                AssertClose(full[(t * V)..((t + 1) * V)], step, 1e-4f, $"cached step {t}");
            }

            Check(context.Length == T, "context length");

            // An int8 cache: prefill and steps close to the full pass (int8 keys and values round to ~1% of their range).
            using var context8 = new DecodingContext(device, 1, 12, KeyValueFormat.Int8);
            float range = full.Max(MathF.Abs);
            AssertClose(full[..(4 * V)], model.ForwardCached(prompt, context8).ToArray(), 0.05f * range, "int8 prefill logits");
            for (int t = 4; t < T; t++)
            {
                using var next = Tensor.From(ids.AsSpan(t, 1), [1, 1], device);
                AssertClose(full[(t * V)..((t + 1) * V)], model.ForwardCached(next, context8).ToArray(), 0.05f * range, $"int8 cached step {t}");
            }
        }
    }

    private static void BatchedDecoding(Device device)
    {
        const int B = 3, T = 7, V = 11;
        using var model = TinyGpt(device);
        var random = new Random(53);
        var ids = new float[B * T];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = random.Next(V);
        }

        using (Autograd.NoGrad())
        {
            using var batchContext = new DecodingContext(device, B, 12);
            var batched = new List<float[]>();
            using (var prompt = Tensor.From([.. Enumerable.Range(0, B).SelectMany(b => ids.Skip(b * T).Take(2))], [B, 2], device))
            {
                model.ForwardCached(prompt, batchContext);
            }

            for (int t = 2; t < T; t++)
            {
                using var next = Tensor.From([.. Enumerable.Range(0, B).Select(b => ids[b * T + t])], [B, 1], device);
                batched.Add(model.ForwardCached(next, batchContext).ToArray());
            }

            for (int b = 0; b < B; b++)
            {
                using var context = new DecodingContext(device, 1, 12);
                using (var prompt = Tensor.From(ids.AsSpan(b * T, 2), [1, 2], device))
                {
                    model.ForwardCached(prompt, context);
                }

                for (int t = 2; t < T; t++)
                {
                    using var next = Tensor.From(ids.AsSpan(b * T + t, 1), [1, 1], device);
                    var single = model.ForwardCached(next, context).ToArray();
                    AssertClose(single, batched[t - 2][(b * V)..((b + 1) * V)], 1e-4f, $"sequence {b} step {t}");
                }
            }
        }
    }

    private static void GraphReplay(Device device)
    {
        const int B = 2, V = 11, Steps = 8;
        using var model = TinyGpt(device);
        int[][] Run(bool useGraph)
        {
            using var context = new DecodingContext(device, B, 12);
            using var sampler = new TokenSampler(device, B, V, Steps + 1) { Temperature = 0.9f, Seed = 7 };
            using (Autograd.NoGrad())
            {
                using (var scope = new TensorScope())
                {
                    sampler.Sample(model.ForwardCached(Tensor.From([1f, 2f, 3f, 4f], [B, 2], device), context));
                }

                void Step() => sampler.Sample(model.ForwardCached(sampler.Ids.Reshape(B, 1), context));
                using var graph = useGraph ? context.CaptureStep(Step) : null;
                for (int s = 0; s < Steps; s++)
                {
                    if (graph is not null)
                    {
                        context.ReplayStep(graph);
                    }
                    else
                    {
                        using var scope = new TensorScope();
                        Step();
                    }
                }

                if (graph is not null && device.Type == DeviceType.Cuda)
                {
                    Check(graph.IsRecorded, $"CUDA graph recording failed: {graph.FailureReason}");
                }
            }

            return [.. sampler.Read(0, Steps + 1).Select(step => step.Select(t => t.Id).ToArray())];
        }

        var direct = Run(useGraph: false);
        var replayed = Run(useGraph: true);
        for (int s = 0; s < direct.Length; s++)
        {
            Check(direct[s].SequenceEqual(replayed[s]), $"step {s}: direct [{string.Join(",", direct[s])}] vs graph [{string.Join(",", replayed[s])}]");
        }
    }

    private static void SamplerBehaviour(Device device)
    {
        const int Rows = 2000, V = 5;
        float[] logits = [2f, 1f, 0f, -1f, 0.5f];
        var all = new float[Rows * V];
        for (int r = 0; r < Rows; r++)
        {
            logits.CopyTo(all, r * V);
        }

        using var x = Tensor.From(all, [Rows, V], device);
        float[] Frequencies(TokenSampler sampler)
        {
            sampler.Sample(x);
            var counts = new float[V];
            foreach (var t in sampler.Read(0, 1)[0])
            {
                counts[t.Id]++;
            }

            return [.. counts.Select(c => c / Rows)];
        }

        using var plain = new TokenSampler(device, Rows, V, 1) { Temperature = 1f, Seed = 3 };
        var expected = logits.Select(MathF.Exp).ToArray();
        float sum = expected.Sum();
        var observed = Frequencies(plain);
        for (int j = 0; j < V; j++)
        {
            Check(MathF.Abs(observed[j] - expected[j] / sum) < 0.035f, $"token {j}: frequency {observed[j]:F3}, probability {expected[j] / sum:F3}");
        }

        var stats = plain.Read(0, 1)[0][0];
        AssertClose([expected[stats.Id] / sum], [stats.Probability], 1e-4f, "reported probability");
        double entropy = -expected.Sum(e => e / sum * Math.Log2(e / sum));
        AssertClose([(float)entropy], [stats.Entropy], 1e-3f, "reported entropy");
        Check(stats.Alternatives[0].Id == 0 && stats.Alternatives[1].Id == 1 && stats.Alternatives[2].Id == 4, "top alternatives in order");

        using var topTwo = new TokenSampler(device, Rows, V, 1) { TopK = 2, Seed = 3 };
        var restricted = Frequencies(topTwo);
        Check(restricted[2] == 0 && restricted[3] == 0 && restricted[4] == 0, "top-k must exclude all but the two best tokens");
        Check(MathF.Abs(restricted[0] - MathF.E / (MathF.E + 1)) < 0.035f, "top-k renormalizes");

        using var cold = new TokenSampler(device, Rows, V, 1) { Temperature = 0.05f, Seed = 3 };
        Check(Frequencies(cold)[0] == 1f, "low temperature is greedy");

        using var again = new TokenSampler(device, Rows, V, 1) { Temperature = 1f, Seed = 3 };
        Check(Frequencies(again).SequenceEqual(observed), "same seed, same samples");

        if (device.Type != DeviceType.Cpu)
        {
            using var cpuX = Tensor.From(all, [Rows, V], Device.Cpu);
            using var cpu = new TokenSampler(Device.Cpu, Rows, V, 1) { Temperature = 1f, Seed = 3 };
            cpu.Sample(cpuX);
            var cpuIds = cpu.Read(0, 1)[0].Select(t => t.Id).ToArray();
            var deviceIds = plain.Read(0, 1)[0].Select(t => t.Id).ToArray();
            int differences = cpuIds.Zip(deviceIds).Count(p => p.First != p.Second);
            Check(differences <= Rows / 200, $"{differences} of {Rows} samples differ between CPU and {device} (rounding only)");
        }

        // A large vocabulary with top-k: the device may pick candidates block by block first. Same tokens as the CPU
        // sampler, including rows where every score ties, or many do (more candidates than fit a slice's slots).
        const int LargeRows = 48, LargeV = 30000;
        var random = new Random(11);
        var large = new float[LargeRows * LargeV];
        for (int i = 0; i < large.Length; i++)
        {
            large[i] = (float)(random.NextDouble() * 6 - 3);
        }

        for (int i = 0; i < LargeV; i++)
        {
            large[i] = 0.5f;                                                   // row 0: all tied
        }

        // Row 1: a slice of low scores with 300 ties (like a checkpoint's padding tokens) that top-k never keeps.
        Array.Fill(large, -5f, LargeV + 3 * 2048, 2048);
        Array.Fill(large, -4f, LargeV + 3 * 2048, 300);
        // Row 2: the same ties are among the top-k scores (three higher scores, then the ties, then everything else).
        Array.Fill(large, -3f, 2 * LargeV, LargeV);
        Array.Fill(large, -0.5f, 2 * LargeV + 5 * 2048, 300);
        (large[2 * LargeV + 10], large[2 * LargeV + 11], large[2 * LargeV + 12]) = (2f, 1.5f, 1f);
        // Row 3: 100 ties at the row's largest score.
        Array.Fill(large, 10f, 3 * LargeV + 7 * 2048, 100);

        using var largeX = Tensor.From(large, [LargeRows, LargeV], device);
        using var largeCpuX = Tensor.From(large, [LargeRows, LargeV], Device.Cpu);
        foreach (var (topK, topP, minP) in new[] { (1, 1f, 0f), (5, 1f, 0f), (40, 0.9f, 0f), (64, 0.95f, 0.02f) })
        {
            using var onDevice = new TokenSampler(device, LargeRows, LargeV, 1) { TopK = topK, TopP = topP, MinP = minP, Seed = 5 };
            using var onCpu = new TokenSampler(Device.Cpu, LargeRows, LargeV, 1) { TopK = topK, TopP = topP, MinP = minP, Seed = 5 };
            onDevice.Sample(largeX);
            onCpu.Sample(largeCpuX);
            var got = onDevice.Read(0, 1)[0];
            var want = onCpu.Read(0, 1)[0];
            int differ = got.Zip(want).Count(p => p.First.Id != p.Second.Id);
            Check(differ <= 1, $"top-k {topK}: {differ} of {LargeRows} tokens differ from the CPU sampler");
            for (int r = 1; r < LargeRows; r++)
            {
                if (got[r].Id == want[r].Id)
                {
                    AssertClose([want[r].Probability], [got[r].Probability], 1e-3f, $"top-k {topK}, row {r}: probability");
                    Check(got[r].Alternatives[0].Id == want[r].Alternatives[0].Id, $"top-k {topK}, row {r}: best alternative");
                }
            }
        }
    }

    // Runs without a GPU: the PTX is generated on the host. CudaBackend.Launch compares every launch's argument count
    // with these numbers, so a missing argument fails with a clear message instead of CUDA_ERROR_INVALID_VALUE.
    private static void KernelSignatures(Device device)
    {
        _ = device;
        var counts = Idrak.Backends.Cuda.PtxKernels.ParameterCounts;
        string[] all = [.. Idrak.Backends.Cuda.PtxKernels.Names, .. Idrak.Backends.Cuda.PtxKernels.AdvancedNames,
            .. Idrak.Backends.Cuda.PtxKernels.DecodingNames, .. Idrak.Backends.Cuda.PtxKernels.QuantizedNames, .. Idrak.Backends.Cuda.PtxKernels.DecoderNames, .. Idrak.Backends.Cuda.PtxKernels.RowNames];
        Check(counts.Count == all.Length && all.All(k => counts.TryGetValue(k, out int n) && n > 0), $"{counts.Count} kernels parsed, {all.Length} expected");
        // Parameter names must be unique within a kernel (a duplicate makes the whole module fail to load on the GPU).
        foreach (var entry in System.Text.RegularExpressions.Regex.Matches(Idrak.Backends.Cuda.PtxKernels.Source, @"\.entry (\w+)\(([^)]*)\)"))
        {
            var match = (System.Text.RegularExpressions.Match)entry;
            var names = System.Text.RegularExpressions.Regex.Matches(match.Groups[2].Value, @"\.param \.\w+ (\w+)").Select(m => m.Groups[1].Value).ToList();
            Check(names.Count == names.Distinct().Count(), $"{match.Groups[1].Value}: duplicate parameter names {string.Join(", ", names)}");
        }

        Check(counts["sample_rows_f32"] == 14 && counts["penalize_rows_f32"] == 14 && counts["history_push_f32"] == 6,
            $"sampler kernels: {counts["sample_rows_f32"]}, {counts["penalize_rows_f32"]}, {counts["history_push_f32"]}");
    }

    private static void SamplerFilters(Device device)
    {
        const int Rows = 2000, V = 5;
        float[] logits = [2f, 1f, 0f, -1f, 0.5f];     // probabilities ≈ 0.563, 0.207, 0.076, 0.028, 0.126
        var all = new float[Rows * V];
        for (int r = 0; r < Rows; r++)
        {
            logits.CopyTo(all, r * V);
        }

        using var x = Tensor.From(all, [Rows, V], device);
        int[] Sample(TokenSampler sampler)
        {
            sampler.Sample(x);
            return [.. sampler.Read(0, 1)[0].Select(t => t.Id)];
        }

        using (var topP = new TokenSampler(device, Rows, V, 1) { TopP = 0.7f, Seed = 5 })
        {
            var ids = Sample(topP);
            Check(ids.All(i => i is 0 or 1), "top-p 0.7 keeps exactly the two most likely tokens (0.563 + 0.207)");
            Check(ids.Contains(1), "top-p keeps the second token");
        }

        using (var narrow = new TokenSampler(device, Rows, V, 1) { TopP = 0.5f, Seed = 5 })
        {
            Check(Sample(narrow).All(i => i == 0), "top-p below the best token's probability is greedy");
        }

        using (var minP = new TokenSampler(device, Rows, V, 1) { MinP = 0.2f, Seed = 5 })
        {
            var ids = Sample(minP);
            Check(ids.All(i => i is 0 or 1 or 4), "min-p 0.2 drops tokens less than 0.2 x as likely as the best");
            Check(ids.Contains(4), "min-p keeps token 4 (ratio 0.22)");
        }

        // Penalties, greedy, one row: history [0, 0, 1].
        using var one = Tensor.From(logits, [1, V], device);
        int Greedy(Action<TokenSampler> configure)
        {
            using var sampler = new TokenSampler(device, 1, V, 1) { Temperature = 0.01f, Seed = 1 };
            configure(sampler);
            sampler.SetHistory([0, 0, 1]);
            sampler.Sample(one);
            return sampler.Read(0, 1)[0][0].Id;
        }

        Check(Greedy(s => s.RepeatPenalty = 2f) == 0, "repeat penalty 2: scores 1, 0.5, 0, -1, 0.5 -> token 0");
        Check(Greedy(s => s.PresencePenalty = 3f) == 4, "presence penalty 3: scores -1, -2, 0, -1, 0.5 -> token 4");
        Check(Greedy(s => { s.RepeatPenalty = 2f; s.FrequencyPenalty = 1f; }) == 4, "repeat 2 + frequency 1: scores -1, -0.5, 0, -1, 0.5 -> token 4");
        Check(Greedy(s => { s.PresencePenalty = 3f; s.RepeatLastN = 1; }) == 0, "a window of 1 only sees token 1: token 0 stays best");

        // Each sampled token joins the history: a large frequency penalty forces a new token every step.
        using var cycle = new TokenSampler(device, 1, V, 5) { Temperature = 0.01f, FrequencyPenalty = 10f, Seed = 1 };
        for (int step = 0; step < 5; step++)
        {
            cycle.Sample(one);
        }

        var sequence = cycle.Read(0, 5).Select(s => s[0].Id).ToArray();
        Check(sequence.SequenceEqual([0, 1, 4, 2, 3]), $"history-driven penalties give 0 1 4 2 3, got {string.Join(" ", sequence)}");

        if (device.Type != DeviceType.Cpu)
        {
            using var cpuX = Tensor.From(all, [Rows, V], Device.Cpu);
            using var cpu = new TokenSampler(Device.Cpu, Rows, V, 1) { TopP = 0.9f, MinP = 0.05f, RepeatPenalty = 1.3f, Seed = 9 };
            using var gpu = new TokenSampler(device, Rows, V, 1) { TopP = 0.9f, MinP = 0.05f, RepeatPenalty = 1.3f, Seed = 9 };
            cpu.SetHistory([0, 4]);
            gpu.SetHistory([0, 4]);
            cpu.Sample(cpuX);
            gpu.Sample(x);
            var a = cpu.Read(0, 1)[0].Select(t => t.Id).ToArray();
            var b = gpu.Read(0, 1)[0].Select(t => t.Id).ToArray();
            int differences = a.Zip(b).Count(p => p.First != p.Second);
            Check(differences <= Rows / 200, $"{differences} of {Rows} filtered samples differ between CPU and {device}");
        }
    }

    private static void SamplerSinglePass(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        var random = new Random(91);
        const int Rows = 3, Vocabulary = 700;
        foreach (var (topK, topP, minP, ties) in new[] { (0, 0f, 0f, false), (20, 0f, 0f, false), (20, 0.9f, 0f, true), (0, 0.5f, 0.05f, false),
                                                         (650, 0.95f, 0f, true), (5, 0f, 0f, true), (40, 0.8f, 0.01f, true), (699, 0f, 0f, false) })
        {
            // Ties: scores from a few values (repeated cut-off scores and equal weights among the top five).
            float[] logits = [.. Enumerable.Range(0, Rows * Vocabulary).Select(_ => ties ? random.Next(-12, 12) * 0.5f : random.NextSingle() * 16 - 8)];
            if (topK == 699)
            {
                logits = [.. logits.Select(v => MathF.Round(v))];                  // fewer distinct scores than k
            }

            using var z = Tensor.From(logits, [Rows, Vocabulary], device);
            using var ids = Tensor.Zeros([Rows], device);
            using var stats = Tensor.Zeros([2 * Rows * 13], device);
            using var step = Tensor.From([1f], [1], device);
            device.Backend.SampleRows(z.Storage, ids.Storage, stats.Storage, step.Storage, Rows, Vocabulary, Vocabulary, 0, 0.8f, topK, topP, minP, 5);
            var (expectedIds, expectedStats) = SampleReference(logits, Rows, Vocabulary, 1, 0.8f, topK, topP, minP, 5);
            Check(ids.ToArray().SequenceEqual(expectedIds), $"top-k {topK}, top-p {topP}, min-p {minP}: chosen tokens");
            Check(stats.ToArray().Skip(Rows * 13).SequenceEqual(expectedStats), $"top-k {topK}, top-p {topP}, min-p {minP}: statistics");
        }
    }

    // The CPU sampler as it was: a full scan per top-k step, per bisection step and per top-5 entry.
    private static (float[] Ids, float[] Stats) SampleReference(float[] lv, int rows, int vocabulary, uint stepNumber, float temperature,
        int topK, float topP, float minP, uint seed)
    {
        float invT = 1f / MathF.Max(temperature, 1e-3f);
        var e = new float[vocabulary];
        var iv = new float[rows];
        var sv = new float[rows * 13];
        Span<int> taken = stackalloc int[5];
        for (int r = 0; r < rows; r++)
        {
            var z = lv.AsSpan(r * vocabulary, vocabulary);
            float max = float.NegativeInfinity;
            foreach (float v in z)
            {
                max = MathF.Max(max, v * invT);
            }

            float threshold = float.NegativeInfinity;
            if (topK > 0 && topK < vocabulary)
            {
                threshold = float.PositiveInfinity;
                for (int k = 0; k < topK; k++)
                {
                    float next = float.NegativeInfinity;
                    foreach (float v in z)
                    {
                        float s = v * invT;
                        if (s < threshold && s > next)
                        {
                            next = s;
                        }
                    }

                    threshold = next;
                }
            }

            if (minP > 0f)
            {
                threshold = MathF.Max(threshold, max + MathF.Log(minP));
            }

            if (topP > 0f && topP < 1f)
            {
                float total = 0f;
                foreach (float v in z)
                {
                    float s = v * invT;
                    total += s >= threshold ? MathF.Exp(s - max) : 0f;
                }

                float goal = total * topP, lo = MathF.Max(max - 40f, threshold), hi = max;
                for (int it = 0; it < 24; it++)
                {
                    float mid = (lo + hi) * 0.5f, mass = 0f;
                    foreach (float v in z)
                    {
                        float s = v * invT;
                        mass += s >= threshold && s >= mid ? MathF.Exp(s - max) : 0f;
                    }

                    if (mass >= goal)
                    {
                        lo = mid;
                    }
                    else
                    {
                        hi = mid;
                    }
                }

                threshold = lo;
            }

            float sum = 0f;
            for (int j = 0; j < vocabulary; j++)
            {
                float s = z[j] * invT;
                e[j] = s >= threshold ? MathF.Exp(s - max) : 0f;
                sum += e[j];
            }

            float target = Idrak.Abstraction.Devices.CounterRandom.Uniform(seed, stepNumber, (uint)r) * sum;
            int chosen = -1;
            float cumulative = 0f;
            for (int j = 0; j < vocabulary; j++)
            {
                if (e[j] > 0f)
                {
                    chosen = j;
                    cumulative += e[j];
                    if (cumulative > target)
                    {
                        break;
                    }
                }
            }

            float entropy = 0f;
            for (int j = 0; j < vocabulary; j++)
            {
                if (e[j] > 0f)
                {
                    float p = e[j] / sum;
                    entropy -= p * MathF.Log2(p);
                }
            }

            iv[r] = chosen;
            int o = r * 13;
            sv[o] = chosen;
            sv[o + 1] = e[chosen] / sum;
            sv[o + 2] = entropy;
            for (int a = 0; a < 5; a++)
            {
                int best = -1;
                for (int j = 0; j < vocabulary; j++)
                {
                    if (e[j] > 0f && (best < 0 || e[j] > e[best]) && taken[..a].IndexOf(j) < 0)
                    {
                        best = j;
                    }
                }

                taken[a] = best;
                sv[o + 3 + 2 * a] = best;
                sv[o + 4 + 2 * a] = best < 0 ? 0f : e[best] / sum;
            }
        }

        return (iv, sv);
    }
}
