// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Datasets.Abstractions;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Text;
using Idrak.Text.Abstractions;

internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] FineTuning =
    [
        ("fine-tuning: chunked token cross-entropy (all rows or trained rows only) and the frozen-transpose product and the fused LoRA term match dense results and gradients", TokenLoss),
        ("memory: a full GPU raises a clear error, or with offloading places tensors in system memory the kernels still use", HostOffload),
        ("fine-tuning: activation checkpointing gives the same loss and gradients (adapters and input)", CheckpointingGradients),
        ("fine-tuning: answers from a short list are balanced (rarer ones repeated up to a limit, deterministically, spread among the others); free text is left alone; batches mix neighbouring sequences", BalancedAnswers),
        ("scoring: log-probabilities of chosen tokens computed on the device match a log-softmax of the full logits (chunked, repeated rows)", TokenLogProbabilities),
        ("scoring: a cached forward pass without autograd frees each layer's intermediate results (only the cache and the output stay)", CachedForwardFreesLayers),
        ("fine-tuning: releasing results no backward step reads, and recomputing feed-forward activations, give the same loss and gradients with less memory (RoPE, biases, q/k norms, post norms, parallel blocks, layer norms, dropout)", ReleasedActivations),
        ("fine-tuning: the activation estimate per rung grows with rows × positions and falls down the ladder (held, recomputed feed-forward, checkpointing); a rung round-trips through the stop state", ActivationEstimatePerRung),
        ("fine-tuning options: DPO, conservative DPO, ORPO and SimPO match their formulas for fixed log-probabilities (and DPO its gradient)", PreferenceFormulas),
        ("fine-tuning options: a DoRA layer gives the reference output and magnitude gradient; merging keeps the output", DoraLayer),
    ];

    internal static void HostOffload(Device device)
    {
        if (device.Type != DeviceType.Cuda)
        {
            return;                                                              // the CPU's memory is system memory already
        }

        var r = new Random(95);
        float[] Values(int n) => [.. Enumerable.Range(0, n).Select(_ => (float)(r.NextDouble() * 2 - 1))];
        var (av, bv) = (Values(512 * 512), Values(512 * 512));
        using var ca = Tensor.From(av, [512, 512], Device.Cpu);
        using var cb = Tensor.From(bv, [512, 512], Device.Cpu);
        var expected = ca.MatMul(cb).ToArray();
        long? limit = ComputeResources.GpuMemoryLimit;
        bool offload = ComputeResources.OffloadToHostMemory;
        ComputeResources.ReleaseCachedMemory(device);
        try
        {
            ComputeResources.GpuMemoryLimit = ComputeResources.GetMemoryUsage(device).InUse + (256L << 10);   // less than one 1 MiB tensor
            ComputeResources.OffloadToHostMemory = false;
            try
            {
                using var tooBig = Tensor.From(av, [512, 512], device);
                Check(false, "a tensor beyond the limit is refused");
            }
            catch (ResourceLimitExceededException)
            {
            }

            ComputeResources.OffloadToHostMemory = true;
            using (var scope = new TensorScope())
            {
                var a = Tensor.From(av, [512, 512], device);
                var b = Tensor.From(bv, [512, 512], device);
                var c = a.MatMul(b) + 0f;
                Check(ComputeResources.GetMemoryUsage(device).Offloaded >= 3L * 512 * 512 * 4, $"offloaded: {ComputeResources.GetMemoryUsage(device)}");
                AssertClose(expected, c.ToArray(), 1e-3f, "a product computed in system memory");
            }

            Check(ComputeResources.GetMemoryUsage(device).Offloaded == 0, "offloaded blocks are returned");
        }
        finally
        {
            ComputeResources.GpuMemoryLimit = limit;
            ComputeResources.OffloadToHostMemory = offload;
            ComputeResources.ReleaseCachedMemory(device);
        }
    }

    internal static void CheckpointingGradients(Device device)
    {
        var spec = SmallSpec with { QkNorm = true };
        var r = new Random(92);
        var inputValues = Enumerable.Range(0, 2 * 7 * spec.Dim).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
        var weightValues = Enumerable.Range(0, 2 * 7 * spec.Dim).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
        (float Loss, float[] Input, float[] Adapters) Run(bool checkpointed)
        {
            using var model = spec.Build(new RandomWeights(93), new DecoderBuildOptions { Device = device });
            model.AddLora(rank: 2, alpha: 4, targets: l => l.Name is "q" or "v" or "down", freezeBase: true, random: new Random(94));
            foreach (var adapter in model.Descendants().OfType<Linear>().Select(l => l.Adapter).OfType<LoraAdapter>())
            {
                adapter.B.Load([.. Enumerable.Range(0, adapter.B.Size).Select(i => MathF.Sin(i))]);
            }

            model.Train();
            using var scope = new TensorScope();
            var x = Tensor.From(inputValues, [2, 7, spec.Dim], device, requiresGrad: true);
            var hidden = x;
            foreach (var block in model.OfType<DecoderBlock>())
            {
                hidden = checkpointed ? block.ForwardCheckpointed(hidden) : block.Forward(hidden);
            }

            var loss = (hidden * Tensor.From(weightValues, [2, 7, spec.Dim], device)).Sum();
            loss.Backward();
            var adapters = model.TrainableParameters().SelectMany(p => p.Grad!.ToArray()).ToArray();
            return (loss.Item(), x.Grad!.ToArray(), adapters);
        }

        var plain = Run(false);
        var checkpointed = Run(true);
        AssertClose([plain.Loss], [checkpointed.Loss], 1e-4f, "loss");
        AssertClose(plain.Input, checkpointed.Input, 1e-4f, "input gradient");
        AssertClose(plain.Adapters, checkpointed.Adapters, 1e-4f, "adapter gradients");
    }

    internal static void ReleasedActivations(Device device)
    {
        var specs = new (string Name, DecoderSpec Spec)[]
        {
            ("rotary, q/k/v biases", SmallSpec with { QkvBias = true }),
            ("one key/value head (its rearrangement is a view)", SmallSpec with { KvHeads = 1 }),
            ("as many key/value heads as query heads", SmallSpec with { KvHeads = 4 }),
            ("q/k norms, post norms, tied", SmallSpec with { QkNorm = true, PostNorms = true, TieEmbeddings = true }),
            ("parallel blocks, layer norms, learned positions, dropout", SmallSpec with
            {
                ParallelBlocks = true, Norm = DecoderNorm.Layer, Rope = null, LearnedPositions = true, Dropout = 0.1f, FeedForwardBias = true, OutputBias = true,
            }),
        };
        var r = new Random(81);
        foreach (var (name, spec) in specs)
        {
            var tokens = Enumerable.Range(0, 3 * 11).Select(_ => (float)r.Next(spec.Vocabulary)).ToArray();
            var weights = Enumerable.Range(0, 3 * 11 * spec.Vocabulary).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
            (float Loss, float[] Gradients, long Held) Run(bool release, bool recompute)
            {
                bool previous = ActivationMemory.ReleaseUnused;
                ActivationMemory.ReleaseUnused = release;
                try
                {
                    using var model = spec.Build(new RandomWeights(82), new DecoderBuildOptions { Device = device });
                    model.AddLora(rank: 2, alpha: 4, targets: _ => true, freezeBase: true, random: new Random(83));
                    foreach (var adapter in model.Descendants().OfType<Linear>().Select(l => l.Adapter).OfType<LoraAdapter>())
                    {
                        adapter.B.Load([.. Enumerable.Range(0, adapter.B.Size).Select(i => 0.1f * MathF.Sin(i))]);
                    }

                    model.Train();
                    using var scope = new TensorScope();
                    using var recomputing = recompute ? ActivationMemory.Recompute() : (ActivationMemory.Scope?)null;
                    // Tensors other tests left to the garbage collector free their memory when finalized; let that happen
                    // now, not between the two readings (it made "held" negative whenever a collection fell in between).
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    long before = ComputeResources.GetMemoryUsage(device).InUse;
                    var logits = model.Forward(Tensor.From(tokens, [3, 11], device));
                    var loss = (logits * Tensor.From(weights, [3, 11, spec.Vocabulary], device)).Sum();
                    long held = ComputeResources.GetMemoryUsage(device).InUse - before;
                    loss.Backward();
                    return (loss.Item(), [.. model.TrainableParameters().SelectMany(p => p.Grad!.ToArray())], held);
                }
                finally
                {
                    ActivationMemory.ReleaseUnused = previous;
                }
            }

            var kept = Run(release: false, recompute: false);
            var released = Run(release: true, recompute: false);
            var recomputed = Run(release: true, recompute: true);
            foreach (var (what, run) in new[] { ("released", released), ("recomputed", recomputed) })
            {
                AssertClose([kept.Loss], [run.Loss], 1e-5f, $"{name}, {what}: loss");
                AssertClose(kept.Gradients, run.Gradients, 1e-5f, $"{name}, {what}: adapter gradients");
            }

            Check(released.Held < kept.Held, $"{name}: releasing holds less after the forward pass ({released.Held:N0} vs {kept.Held:N0} bytes)");

            Check(spec.Gated == false || recomputed.Held < released.Held, $"{name}: recomputing holds less still ({recomputed.Held:N0} vs {released.Held:N0} bytes)");
        }
    }

    internal static void CachedForwardFreesLayers(Device device)
    {
        var spec = new DecoderSpec
        {
            Vocabulary = 260, Dim = 128, Layers = 4, Heads = 2, KvHeads = 1, HeadDim = 64, FfDim = 512, MaxPositions = 256,
            Rope = new RopeSettings(10000f), NormEpsilon = 1e-6f,
        };
        using var model = spec.Build(new RandomWeights(61), new DecoderBuildOptions { Device = device });
        model.Eval();
        const int Rows = 4, Steps = 64;
        var tokens = Enumerable.Range(0, Rows * Steps).Select(i => (float)(i * 7 % 260)).ToArray();
        using var context = new DecodingContext(device, Rows, Steps);
        using (Autograd.NoGrad())
        using (new TensorScope())
        {
            long before = ComputeResources.GetMemoryUsage(device).InUse;
            var hidden = model.ForwardCached(Tensor.From(tokens, [Rows, Steps], device), context, model.Count - 1);
            long held = ComputeResources.GetMemoryUsage(device).InUse - before - context.CacheBytes;
            long layer = (long)Rows * Steps * spec.FfDim * sizeof(float);                    // one feed-forward activation
            Check(hidden.Shape.SequenceEqual([Rows, Steps, spec.Dim]), $"hidden states {string.Join("×", hidden.Shape.ToArray())}");
            Check(held < 2 * layer, $"only the cache and the output are held after the pass ({held:N0} bytes besides the cache; one feed-forward activation is {layer:N0})");
        }
    }

    internal static void BalancedAnswers(Device device)
    {
        _ = device;
        static TrainingSequence Row(int prompt, int[] answer) =>
            new([prompt, 7, .. answer, 1], [false, false, .. answer.Select(_ => true), true]);
        var rows = Enumerable.Range(0, 10).Select(i => Row(i, [40])).Concat(Enumerable.Range(0, 3).Select(i => Row(i, [41, 42])))
            .Concat([Row(0, [43])]).ToList();
        var (balanced, answers) = FineTuner.BalanceAnswers(rows, maxRepeats: 4, seed: 3);
        string Counts(IEnumerable<(int[] Answer, int Before, int After)> a) => string.Join(" ", a.Select(x => $"{x.Answer[0]}:{x.Before}->{x.After}"));
        Check(Counts(answers) == "40:10->10 41:3->10 43:1->4", $"counts {Counts(answers)}");
        Check(balanced.Count == 24 && rows.All(balanced.Contains), $"{balanced.Count} sequences, every original kept");
        Check(balanced.Count(s => s.Tokens[2] == 41) == 10 && balanced.Count(s => s.Tokens[2] == 43) == 4, "repeats per answer");
        var again = FineTuner.BalanceAnswers(rows, maxRepeats: 4, seed: 3).Sequences;
        Check(again.Select(s => s.Tokens[0] * 100 + s.Tokens[2]).SequenceEqual(balanced.Select(s => s.Tokens[0] * 100 + s.Tokens[2])), "the same for the same seed");

        var free = Enumerable.Range(0, 100).Select(i => Row(i, [100 + i])).ToList();
        Check(FineTuner.BalanceAnswers(free).Sequences.Count == 100, "free text (100 distinct answers) is left alone");

        // Neighbours in the list do not fill rows together: 200 sequences of one length, the first half one answer and
        // the second half another, packed 20 to a row, give rows of both (a row of one answer teaches that answer alone).
        var halves = Enumerable.Range(0, 200).Select(i => Row(i, [i < 100 ? 50 : 51])).ToList();
        var packed = FineTuner.PackedBatches(halves, 20 * 4, 1, new Random(5));
        int mixed = packed.Count(b => b[0].Select(i => halves[i].Tokens[2]).Distinct().Count() == 2);
        Check(packed.Count > 1 && mixed == packed.Count, $"{mixed} of {packed.Count} rows hold both answers");
        var padded = FineTuner.Batches(halves, 20 * 5, new Random(5));
        Check(padded.Count(b => b.Select(i => halves[i].Tokens[2]).Distinct().Count() == 2) >= padded.Count - 1, "padded batches mix them too");
    }

    internal static void TokenLogProbabilities(Device device)
    {
        var r = new Random(53);
        const int Positions = 37, Dim = 24, Vocabulary = 301;
        float[] Values(int n, float scale) => [.. Enumerable.Range(0, n).Select(_ => (float)(r.NextDouble() * 2 - 1) * scale)];
        using var scope = new TensorScope();
        var hiddenValues = Values(Positions * Dim, 1f);
        var hidden = Tensor.From(hiddenValues, [Positions, Dim], device);
        var head = new Linear(Dim, Vocabulary, bias: true, device: device, random: new Random(54));
        int[] rows = [0, 5, 5, 36, 12, 7, 20, 20, 3];
        int[] targets = [.. rows.Select(_ => r.Next(Vocabulary))];
        float[] expected;
        using (Autograd.NoGrad())
        {
            var logits = head.Forward(hidden).ToArray();
            expected = [.. rows.Select((row, i) =>
            {
                var span = logits.AsSpan(row * Vocabulary, Vocabulary);
                double max = double.NegativeInfinity, sum = 0;
                foreach (float v in span)
                {
                    max = Math.Max(max, v);
                }

                foreach (float v in span)
                {
                    sum += Math.Exp(v - max);
                }

                return (float)(span[targets[i]] - max - Math.Log(sum));
            })];
        }

        foreach (int chunk in new[] { 1024, 4 })
        {
            AssertClose(expected, Losses.TokenLogProbabilities(hidden, head.Forward, rows, targets, chunk), 1e-2f, $"chunks of {chunk}");
        }

        Check(Losses.TokenLogProbabilities(hidden, head.Forward, [], []).Length == 0, "no rows, no values");
    }

    internal static void ActivationEstimatePerRung(Device device)
    {
        var spec = new DecoderSpec
        {
            Vocabulary = 260, Dim = 128, Layers = 4, Heads = 4, KvHeads = 2, HeadDim = 32, FfDim = 512, MaxPositions = 512,
            Rope = new RopeSettings(10000f), NormEpsilon = 1e-6f, QkNorm = true, TieEmbeddings = true,
        };
        var capabilities = device.Backend.Capabilities;
        ActivationRung[] ladder = [new(false, false), new(true, false), new(true, true)];
        long Bytes(int rows, int length, ActivationRung rung) => ActivationEstimate.Bytes(spec, capabilities, rows, length, rung, 512);
        for (int i = 0; i < ladder.Length; i++)
        {
            Check(Bytes(2, 256, ladder[i]) > Bytes(1, 256, ladder[i]) && Bytes(1, 512, ladder[i]) > Bytes(1, 256, ladder[i]),
                $"{ladder[i]}: more rows or positions need more ({Bytes(1, 256, ladder[i])}, {Bytes(2, 256, ladder[i])}, {Bytes(1, 512, ladder[i])} bytes)");
            Check(i == 0 || Bytes(4, 256, ladder[i]) < Bytes(4, 256, ladder[i - 1]),
                $"{ladder[i]} needs less than {(i == 0 ? "" : ladder[i - 1].ToString())}: {Bytes(4, 256, ladder[i])} and {(i == 0 ? 0 : Bytes(4, 256, ladder[i - 1]))} bytes");
            Check(ActivationRung.FromJson(System.Text.Json.Nodes.JsonNode.Parse(ladder[i].ToJson().ToJsonString())) == ladder[i], $"{ladder[i]} round-trips");
        }

        Check(ActivationEstimate.Bytes(spec, capabilities, 1, 256, ladder[0], 512, pendingGradients: 1000) == Bytes(1, 256, ladder[0]) + 4000,
            "the gradients a first step makes are counted");
        Check(ActivationRung.FromJson(null) is null, "a stop state from before rungs were kept holds none");
    }

    internal sealed class Unreachable : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
            throw new HttpRequestException("No such host is known.");
    }

    internal static string WriteChatModel(DecoderSpec spec) => WriteChatModelWith(spec, Qwen3Template);

    // The tiny chat model with the chat template `template` (Jinja), its weights from seed 91.
    internal static string WriteChatModelWith(DecoderSpec spec, string template)
    {
        string folder = TempFolder();
        var weights = new RandomWeights(91);
        using (spec.Build(weights, new DecoderBuildOptions { Device = Device.Cpu }))
        {
        }

        WriteCheckpoint(folder, spec, weights, "Qwen3ForCausalLM", sharded: false);
        var printable = Enumerable.Range('!', '~' - '!' + 1).Concat(Enumerable.Range('¡', '¬' - '¡' + 1)).Concat(Enumerable.Range('®', 'ÿ' - '®' + 1)).ToHashSet();
        var vocab = new JsonObject();
        for (int b = 0, n = 0; b < 256; b++)
        {
            vocab[((char)(printable.Contains(b) ? b : 256 + n++)).ToString()] = b;
        }

        var tokenizer = new JsonObject
        {
            ["added_tokens"] = new JsonArray(
                new JsonObject { ["id"] = 256, ["content"] = "<|im_start|>", ["special"] = true },
                new JsonObject { ["id"] = 257, ["content"] = "<|im_end|>", ["special"] = true },
                new JsonObject { ["id"] = 258, ["content"] = "<|endoftext|>", ["special"] = true }),
            ["model"] = new JsonObject { ["type"] = "BPE", ["vocab"] = vocab, ["merges"] = new JsonArray() },
            ["pre_tokenizer"] = new JsonObject { ["type"] = "ByteLevel", ["add_prefix_space"] = false, ["use_regex"] = true },
            ["decoder"] = new JsonObject { ["type"] = "ByteLevel" },
        };
        File.WriteAllText(Path.Combine(folder, "tokenizer.json"), tokenizer.ToJsonString());
        File.WriteAllText(Path.Combine(folder, "tokenizer_config.json"), new JsonObject
        {
            ["chat_template"] = template, ["eos_token"] = "<|im_end|>", ["bos_token"] = null,
        }.ToJsonString());
        return folder;
    }

    internal static void TokenLoss(Device device)
    {
        const int Rows = 11, Dim = 6, Vocabulary = 37;
        var r = new Random(81);
        float[] Random(int n) => [.. Enumerable.Range(0, n).Select(_ => (float)(r.NextDouble() * 2 - 1))];
        var hiddenValues = Random(Rows * Dim);
        var ids = Enumerable.Range(0, Rows).Select(_ => (float)r.Next(Vocabulary)).ToArray();
        var weights = Enumerable.Range(0, Rows).Select(i => i % 4 == 0 ? 0f : 1f).ToArray();
        float normalizer = weights.Sum();
        var adapterB = Random(2 * Vocabulary);

        float[] Run(int mode, out float[] hiddenGrad, out float[] adapterGrad)
        {
            using var head = new Linear(Dim, Vocabulary, bias: true, device, new Random(82));
            head.AddLora(rank: 2, alpha: 2, targets: _ => true, freezeBase: true, random: new Random(83));
            head.Lora!.B.Load(adapterB);
            using var scope = new TensorScope();
            var hidden = Tensor.From(hiddenValues, [Rows, Dim], device, requiresGrad: true);
            Tensor loss;
            if (mode == 2)
            {
                // Only the rows with a weight: the head never sees the others.
                int[] rows = [.. Enumerable.Range(0, Rows).Where(i => weights[i] != 0f)];
                loss = Losses.TokenCrossEntropyRows(hidden, h => head.Forward(h), rows, [.. rows.Select(i => ids[i])], [.. rows.Select(i => weights[i])],
                    normalizer, chunkRows: 3);
            }
            else if (mode == 1)
            {
                using var targets = Tensor.From(ids, [Rows], device);
                using var w = Tensor.From(weights, [Rows], device);
                loss = Losses.TokenCrossEntropy(hidden, h => head.Forward(h), targets, w, normalizer, chunkRows: 4);
            }
            else
            {
                var weighted = new float[Rows * Vocabulary];
                for (int i = 0; i < Rows; i++)
                {
                    weighted[i * Vocabulary + (int)ids[i]] = weights[i];
                }

                var onehot = Tensor.From(weighted, [Rows, Vocabulary], device);
                loss = (head.Forward(hidden).LogSoftmax() * onehot).Sum() * (-1f / normalizer);
            }

            loss.Backward();
            hiddenGrad = hidden.Grad!.ToArray();
            adapterGrad = head.Lora!.A.Grad!.ToArray();
            return loss.ToArray();
        }

        var expected = Run(0, out var expectedHidden, out var expectedAdapter);
        var actual = Run(1, out var actualHidden, out var actualAdapter);
        AssertClose(expected, actual, 1e-4f, "loss");
        AssertClose(expectedHidden, actualHidden, 1e-4f, "hidden gradient");
        AssertClose(expectedAdapter, actualAdapter, 1e-4f, "adapter gradient");
        var rowsOnly = Run(2, out var rowsHidden, out var rowsAdapter);
        AssertClose(expected, rowsOnly, 1e-4f, "loss on trained rows only");
        AssertClose(expectedHidden, rowsHidden, 1e-4f, "hidden gradient, trained rows only");
        AssertClose(expectedAdapter, rowsAdapter, 1e-4f, "adapter gradient, trained rows only");

        // The fused LoRA term: same output and gradients (input, A, B, base weight) as product + x·A·B·scale.
        var w0 = Random(Dim * Vocabulary);
        var a0 = Random(Dim * 3);
        var b0 = Random(3 * Vocabulary);
        float[][] Lora(bool fused)
        {
            using var scope = new TensorScope();
            var x = Tensor.From(hiddenValues, [Rows, Dim], device, requiresGrad: true);
            var w = Tensor.From(w0, [Dim, Vocabulary], device, requiresGrad: true);
            var a = Tensor.From(a0, [Dim, 3], device, requiresGrad: true);
            var b = Tensor.From(b0, [3, Vocabulary], device, requiresGrad: true);
            var product = x.MatMul(w);
            var y = fused ? Tensor.AddLowRank(product, x, a, b, 0.75f) : product + x.MatMul(a).MatMul(b) * 0.75f;
            (y * y).Sum().Backward();
            return [y.ToArray(), x.Grad!.ToArray(), w.Grad!.ToArray(), a.Grad!.ToArray(), b.Grad!.ToArray()];
        }

        var plainLora = Lora(false);
        var fusedLora = Lora(true);
        string[] parts = ["output", "input gradient", "base weight gradient", "A gradient", "B gradient"];
        for (int i = 0; i < parts.Length; i++)
        {
            AssertClose(plainLora[i], fusedLora[i], 1e-3f, $"fused LoRA: {parts[i]}");
        }
    }

    internal static double Norm(float[] values) => Math.Sqrt(values.Sum(v => (double)v * v));

    // ‖expected - actual‖ ≤ tolerance · ‖expected‖.
    internal static void CloseByNorm(float[] expected, float[] actual, float tolerance, string what)
    {
        Check(expected.Length == actual.Length, $"{what}: length {actual.Length}, expected {expected.Length}");
        double difference = 0, norm = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            difference += (double)(expected[i] - actual[i]) * (expected[i] - actual[i]);
            norm += (double)expected[i] * expected[i];
        }

        double error = Math.Sqrt(difference / Math.Max(norm, 1e-30));
        Check(error <= tolerance, $"{what}: relative error {error:G3}");
    }

    // Reports on the calling thread (Progress<T> posts to the thread pool, so its reports can arrive late).
    internal sealed class SynchronousProgress<T>(Action<T> report) : IProgress<T>
    {
        public void Report(T value) => report(value);
    }

    // A progress that keeps every report in order, on the reporting thread (Progress<T> would post them).
    private sealed class Recorded<T> : IProgress<T>
    {
        public List<T> Values { get; } = [];

        public void Report(T value)
        {
            lock (Values)
            {
                Values.Add(value);
            }
        }
    }
}
