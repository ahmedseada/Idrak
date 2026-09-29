using System.Text.Json.Nodes;
using Idrak;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Pretrained;

internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] FineTuning =
    [
        ("fine-tuning: chunked token cross-entropy (all rows or trained rows only) and the frozen-transpose product and the fused LoRA term match dense results and gradients", TokenLoss),
        ("fine-tuning: LoRA terms inside the tensor-core products (float32, bfloat16 and 4-bit bases, one layer or merged) and rank-16 products match the separate computation, with gradients", LoraInsideProducts),
        ("fine-tuning: packed sequences (several per row, rotary or learned positions) give each sequence the logits and gradients it gets alone; packing fills rows first-fit", PackedSequencesMatch),
        ("fine-tuning: a training step recorded as a CUDA graph and replayed gives the losses and adapters of ordinary steps", GraphTraining),
        ("generation: prompts of different lengths decoded together (left-padded, per-row starts) give each row its own logits and greedy replies", RaggedBatchDecoding),
        ("memory: a full GPU raises a clear error, or with offloading places tensors in system memory the kernels still use", HostOffload),
        ("fine-tuning: activation checkpointing gives the same loss and gradients (adapters and input)", CheckpointingGradients),
        ("fine-tuning: the input gradient through bfloat16 weights reads them as stored (with and without the adapter's term, odd widths, split k)", BFloat16InputGradient),
        ("fine-tuning: answers from a short list are balanced (rarer ones repeated up to a limit, deterministically, spread among the others); free text is left alone; batches mix neighbouring sequences", BalancedAnswers),
        ("scoring: log-probabilities of chosen tokens computed on the device match a log-softmax of the full logits (chunked, repeated rows)", TokenLogProbabilities),
        ("scoring: a cached forward pass without autograd frees each layer's intermediate results (only the cache and the output stay)", CachedForwardFreesLayers),
        ("scoring: given answers to chat prompts are scored as the model's own full forward pass scores them, each prompt run once from a cache or with every answer, whatever the batching; long messages are shortened to fit", AnswerScoring),
        ("fine-tuning: the gated activation reads and writes bfloat16 words as packing and unpacking around the float kernels would (SiLU, GELU, ReLU, odd sizes)", PackedGatedActivation),
        ("fine-tuning: releasing results no backward step reads, and recomputing feed-forward activations, give the same loss and gradients with less memory (RoPE, biases, q/k norms, post norms, parallel blocks, layer norms, dropout)", ReleasedActivations),
        ("fine-tuning: checkpointing is off by default and turns on (the step run again) when a step runs out of memory; a lighter setting that fits is timed against checkpointing and the faster kept", AutomaticCheckpointing),
        ("models: a Hugging Face model id is downloaded (only the files the library reads, sharded weights) into the cache, loads, and works offline", ModelDownload),
        ("fine-tuning: agent transcripts through the chat template, assistant-only tokens, LoRA and QLoRA training, PEFT adapters, merged export", AgentFineTuning),
        ("fine-tuning: a conversation too long for the maximum length keeps its whole answer (the user message is shortened, its start kept); rows encode on all cores in order", LongMessageKeepsAnswer),
        ("fine-tuning: an adapter folder's manifest round-trips, names a local base model by its full path, and loads the model with the adapter merged", ManifestRoundTrip),
    ];

    private static void HostOffload(Device device)
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

    private static void CheckpointingGradients(Device device)
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

    private static void ReleasedActivations(Device device)
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
            (float Loss, float[] Gradients, long Held) Run(bool release, bool recompute, bool compress = false)
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
                    using var compressing = compress ? ActivationMemory.CompressToBFloat16() : (ActivationMemory.Scope?)null;
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

            // Held as bfloat16 between the passes: the same loss (computed before), gradients within bfloat16's rounding,
            // less memory still; also with the feed-forward activations recomputed from compressed gate and up.
            foreach (var (what, run) in new[] { ("bfloat16", Run(true, false, compress: true)), ("bfloat16, recomputed", Run(true, true, compress: true)) })
            {
                AssertClose([kept.Loss], [run.Loss], 1e-5f, $"{name}, {what}: loss");
                CloseByNorm(kept.Gradients, run.Gradients, 2e-2f, $"{name}, {what}: adapter gradients");
                Check(run.Held < released.Held, $"{name}, {what}: holds less than released alone ({run.Held:N0} vs {released.Held:N0} bytes)");
            }
            Check(spec.Gated == false || recomputed.Held < released.Held, $"{name}: recomputing holds less still ({recomputed.Held:N0} vs {released.Held:N0} bytes)");
        }
    }

    private static void AnswerScoring(Device device)
    {
        var small = new DecoderSpec
        {
            Vocabulary = 260, Dim = 32, Layers = 2, Heads = 4, KvHeads = 2, HeadDim = 8, FfDim = 64, MaxPositions = 256,
            Rope = new RopeSettings(10000f), NormEpsilon = 1e-6f, QkNorm = true, TieEmbeddings = true,
        };

        // Head size 64: what CUDA's attention over rows of different lengths needs, so the GPU also runs prompts once.
        AnswerScoring(device, small);
        AnswerScoring(device, small with { Dim = 128, Heads = 2, KvHeads = 1, HeadDim = 64, FfDim = 256 });
    }

    private static void AnswerScoring(Device device, DecoderSpec spec)
    {
        string folder = WriteChatModel(spec);
        try
        {
            using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
            model.Network.Eval();
            string[] answers = ["yes", "no", "maybe later"];
            IReadOnlyList<ChatMessage>[] prompts =
            [
                [new ChatMessage("system", "Answer yes or no."), new ChatMessage("user", "Is it raining?")],
                [new ChatMessage("user", "Short.")],
                [new ChatMessage("system", "Answer yes or no."), new ChatMessage("user", new string('x', 2000))],   // shortened to fit
            ];
            var scorer = new AnswerScorer(model, maxLength: 96);
            var scores = scorer.LogLikelihoods(prompts, answers);
            var single = new AnswerScorer(model, maxLength: 96) { PromptsPerPass = 1 }.LogLikelihoods(prompts, answers);
            var bounded = new AnswerScorer(model, maxLength: 96) { PositionsPerPass = 1 }.LogLikelihoods(prompts, answers);   // one prompt a pass
            var unshared = new AnswerScorer(model, maxLength: 96) { SharePrompts = false }.LogLikelihoods(prompts, answers);
            // With the default precision (bfloat16 tensor cores where the GPU has them), each prompt runs once.
            bool mustShare = device.Type == DeviceType.Cpu || spec.HeadDim == 64 && MixedPrecision.TensorCoresUnavailable(device) is null;
            Check(scorer.SharesPrompts || !mustShare, $"head size {spec.HeadDim}: each prompt runs once for all its answers ({scorer.EffectivePrecision})");
            using var precision = MixedPrecision.Use(scorer.EffectivePrecision);                  // the reference at the same precision

            // The same values from the whole network's logits over each (fitted prompt, answer) sequence alone.
            for (int p = 0; p < prompts.Length; p++)
            {
                var fitted = scorer.Fit(prompts[p], answers);
                Check(fitted is not null, $"prompt {p} fits");
                for (int a = 0; a < answers.Length; a++)
                {
                    var sequence = scorer.Encoder.Encode(new ChatTranscript([.. fitted!, new ChatMessage("assistant", answers[a])], [], false), 96)!;
                    Check(sequence.Tokens.Length <= 97, $"prompt {p}, answer {a}: within the length ({sequence.Tokens.Length} tokens)");
                    double expected = 0;
                    using (Autograd.NoGrad())
                    using (new TensorScope())
                    {
                        int n = sequence.Tokens.Length - 1;
                        var logits = model.Network.Forward(Tensor.From([.. sequence.Tokens.Take(n).Select(t => (float)t)], [1, n], device)).ToArray();
                        int vocabulary = logits.Length / n;
                        for (int t = 0; t < n; t++)
                        {
                            if (!sequence.Trained[t + 1])
                            {
                                continue;
                            }

                            var row = logits.AsSpan(t * vocabulary, vocabulary);
                            double max = double.NegativeInfinity, sum = 0;
                            foreach (float v in row)
                            {
                                max = Math.Max(max, v);
                            }

                            foreach (float v in row)
                            {
                                sum += Math.Exp(v - max);
                            }

                            expected += row[sequence.Tokens[t + 1]] - max - Math.Log(sum);
                        }
                    }

                    AssertClose([(float)expected], [(float)scores[p][a]], 2e-2f, $"prompt {p}, answer {a}: log-likelihood");
                    AssertClose([(float)scores[p][a]], [(float)single[p][a]], 2e-2f, $"prompt {p}, answer {a}: one prompt per pass");
                    AssertClose([(float)scores[p][a]], [(float)bounded[p][a]], 2e-2f, $"prompt {p}, answer {a}: passes bounded by positions");
                    AssertClose([(float)scores[p][a]], [(float)unshared[p][a]], 2e-2f, $"prompt {p}, answer {a}: every row running the whole prompt");
                }
            }

            var choices = scorer.Choose(prompts, answers);
            foreach (var choice in choices)
            {
                AssertClose([1f], [(float)choice.Probabilities.Sum()], 1e-5f, "probabilities sum to one");
                Check(choice.Best == Array.IndexOf([.. choice.LogLikelihoods], choice.LogLikelihoods.Max()), "the best is the most likely");
            }

            Check(new AnswerScorer(model, maxLength: 4).Choose([prompts[0]], answers)[0].Probabilities.All(x => Math.Abs(x - 1.0 / 3) < 1e-9),
                "a prompt that cannot fit gets even probabilities");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    private static void CachedForwardFreesLayers(Device device)
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

    private static void BalancedAnswers(Device device)
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

    private static void TokenLogProbabilities(Device device)
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

    private static void PackedGatedActivation(Device device)
    {
        var r = new Random(52);
        float[] Values(int n) => [.. Enumerable.Range(0, n).Select(_ => (float)(r.NextDouble() * 6 - 3))];
        foreach (int n in new[] { 1, 7, 1000, 4097 })
        {
            for (int kind = 0; kind < 3; kind++)
            {
                using var scope = new TensorScope();
                var gate = Tensor.From(Values(n), [n], device);
                var up = Tensor.From(Values(n), [n], device);
                var backend = gate.Backend;
                int words = (n + 1) / 2;
                float[] Unpacked(Tensor packed)
                {
                    var values = Tensor.Zeros([n], device);
                    backend.BFloat16Dequantize(packed.Storage, values.Storage, 1, n);
                    return values.ToArray();
                }

                Tensor Packed(Tensor values)
                {
                    var packed = Tensor.Zeros([words], device);
                    backend.PackBFloat16(values.Storage, packed.Storage, n);
                    return packed;
                }

                // Forward: y from the floats, gate, up and y packed in the same pass.
                var expected = Tensor.Zeros([n], device);
                backend.GatedActivation(gate.Storage, up.Storage, expected.Storage, n, kind);
                var (y, pg, pu, py) = (Tensor.Zeros([n], device), Tensor.Zeros([words], device), Tensor.Zeros([words], device), Tensor.Zeros([words], device));
                backend.GatedActivationPacked(gate.Storage, up.Storage, pg.Storage, pu.Storage, y.Storage, py.Storage, n, kind, 2 | 4 | 8);
                string what = $"n {n}, kind {kind}";
                AssertClose(expected.ToArray(), y.ToArray(), 1e-5f, $"{what}: y");
                AssertClose(Unpacked(Packed(gate)), Unpacked(pg), 0f, $"{what}: packed gate");
                AssertClose(Unpacked(Packed(up)), Unpacked(pu), 0f, $"{what}: packed up");
                AssertClose(Unpacked(Packed(expected)), Unpacked(py), 1e-2f, $"{what}: packed y");

                // Recomputed from the words: as the float kernel over the unpacked values.
                var (g16, u16) = (Tensor.From(Unpacked(pg), [n], device), Tensor.From(Unpacked(pu), [n], device));
                backend.GatedActivation(g16.Storage, u16.Storage, expected.Storage, n, kind);
                var recomputed = Tensor.Zeros([n], device);
                backend.GatedActivationPacked(recomputed.Storage, recomputed.Storage, pg.Storage, pu.Storage, recomputed.Storage, recomputed.Storage, n, kind, 1 | 4);
                AssertClose(expected.ToArray(), recomputed.ToArray(), 1e-5f, $"{what}: recomputed");

                // Backward from the words, accumulating (flags: gate, up, both).
                for (int flags = 1; flags <= 3; flags++)
                {
                    var dy = Tensor.From(Values(n), [n], device);
                    var start = Values(n);
                    var (dg, du) = (Tensor.From(start, [n], device), Tensor.From(start, [n], device));
                    var (eg, eu) = (Tensor.From(start, [n], device), Tensor.From(start, [n], device));
                    backend.GatedActivationBackward(g16.Storage, u16.Storage, dy.Storage, eg.Storage, eu.Storage, n, kind, flags);
                    backend.GatedActivationBackwardPacked(pg.Storage, pu.Storage, dy.Storage, dg.Storage, du.Storage, n, kind, flags);
                    AssertClose(eg.ToArray(), dg.ToArray(), 1e-5f, $"{what}, flags {flags}: dgate");
                    AssertClose(eu.ToArray(), du.ToArray(), 1e-5f, $"{what}, flags {flags}: dup");

                    // The first gradient written over whatever the buffers hold (flags 4 and 8), float and packed kernels.
                    int written = flags | 4 | 8;
                    var nan = Enumerable.Repeat(float.NaN, n).ToArray();
                    var (zg, zu) = (Tensor.Zeros([n], device), Tensor.Zeros([n], device));
                    backend.GatedActivationBackward(g16.Storage, u16.Storage, dy.Storage, zg.Storage, zu.Storage, n, kind, flags);
                    var (wg, wu) = (Tensor.From(nan, [n], device), Tensor.From(nan, [n], device));
                    backend.GatedActivationBackward(g16.Storage, u16.Storage, dy.Storage, wg.Storage, wu.Storage, n, kind, written);
                    var (pwg, pwu) = (Tensor.From(nan, [n], device), Tensor.From(nan, [n], device));
                    backend.GatedActivationBackwardPacked(pg.Storage, pu.Storage, dy.Storage, pwg.Storage, pwu.Storage, n, kind, written);
                    foreach (var (expectedWritten, actual, part) in new[] { (zg, wg, "dgate"), (zu, wu, "dup"), (zg, pwg, "packed dgate"), (zu, pwu, "packed dup") })
                    {
                        bool used = part.EndsWith("dgate") ? (flags & 1) != 0 : (flags & 2) != 0;
                        if (used)
                        {
                            AssertClose(expectedWritten.ToArray(), actual.ToArray(), 1e-5f, $"{what}, flags {written}: {part} written");
                        }
                    }
                }
            }
        }
    }

    private static void BFloat16InputGradient(Device device)
    {
        if (device.Type != DeviceType.Cuda || MixedPrecision.TensorCoresUnavailable(device) is not null)
        {
            return;                                                              // a tensor-core kernel; others expand the weight
        }

        using var precision = MixedPrecision.BFloat16();
        var r = new Random(51);
        float[] Values(int n) => [.. Enumerable.Range(0, n).Select(_ => (float)(r.NextDouble() * 2 - 1))];
        foreach (var (m, inputs, outputs) in new[] { (300, 150, 201), (4096, 96, 2049) })
        {
            using var scope = new TensorScope();
            using var weight = BFloat16Weight.FromValues(Values(inputs * outputs), inputs, outputs, device);     // W [inputs, outputs]
            var g = Tensor.From(Values(m * outputs), [m, outputs], device);
            var dt = Tensor.From(Values(m * 16), [m, 16], device);
            var a = Tensor.From(Values(inputs * 16), [inputs, 16], device);
            var start = Values(m * inputs);
            foreach (bool lowRank in new[] { false, true })
            {
                var direct = Tensor.From(start, [m, inputs], device);
                Check(g.Backend.BFloat16TransposedMatMul(g.Storage, weight.Packed.Storage, direct.Storage, m, inputs, outputs, 1f,
                    lowRank ? dt.Storage : null, lowRank ? a.Storage : null, 16), "the kernel ran");
                var expected = Tensor.From(start, [m, inputs], device);
                using var w = weight.Dequantize();
                g.Backend.BatchedMatMul(g.Storage, w.Storage, expected.Storage, 1, m, inputs, outputs, false, true, 1f);
                if (lowRank)
                {
                    g.Backend.BatchedMatMul(dt.Storage, a.Storage, expected.Storage, 1, m, inputs, 16, false, true, 1f);
                }

                CloseByNorm(expected.ToArray(), direct.ToArray(), 1e-3f, $"{m}×{inputs}×{outputs}{(lowRank ? " + low rank" : "")}");
            }
        }
    }

    private static void AutomaticCheckpointing(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                              // the CPU memory limit makes a small full device
        }

        var spec = new DecoderSpec
        {
            Vocabulary = 260, Dim = 128, Layers = 4, Heads = 4, KvHeads = 2, HeadDim = 32, FfDim = 512, MaxPositions = 512,
            Rope = new RopeSettings(10000f), NormEpsilon = 1e-6f, QkNorm = true, TieEmbeddings = true,
        };
        string folder = WriteChatModel(spec);
        var random = new Random(71);
        var sequences = Enumerable.Range(0, 16).Select(_ =>
            new TrainingSequence([.. Enumerable.Range(0, 257).Select(_ => random.Next(256))], [.. Enumerable.Range(0, 257).Select(i => i > 128)])).ToList();
        var options = new FineTuningOptions { Rank = 4, Alpha = 8, BatchTokens = 2048, Seed = 1, Packing = false };
        (bool Done, string Trace, List<float> Losses) Train(bool? checkpointing, long? extra, int epochs = 1)
        {
            using var pretrained = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
            var trace = new System.Text.StringBuilder();
            var losses = new List<float>();
            long before = ComputeResources.GetMemoryUsage(device).InUse;
            ComputeResources.CpuMemoryLimit = extra is { } e ? before + e : null;
            try
            {
                FineTuner.Train(pretrained, sequences, null, options with { Checkpointing = checkpointing, Epochs = epochs },
                    progress: new SynchronousProgress<FineTuningProgress>(p => losses.Add(p.Loss)), trace: line => trace.AppendLine(line));
                return (true, trace.ToString(), losses);
            }
            catch (ResourceLimitExceededException)
            {
                return (false, trace.ToString(), losses);
            }
            finally
            {
                ComputeResources.CpuMemoryLimit = null;
            }
        }

        try
        {
            // A memory budget that fits the training with checkpointing but not without.
            long? budget = null;
            foreach (long megabytes in new long[] { 4, 8, 12, 16, 24, 32, 48, 64, 96, 128 })
            {
                if (Train(true, megabytes << 20).Done)
                {
                    budget = Train(false, megabytes << 20).Done ? null : megabytes << 20;
                    break;
                }
            }

            Check(budget is not null, "a budget where only checkpointed training fits");
            var automatic = Train(null, budget);
            Check(automatic.Done && automatic.Trace.Contains("checkpointing is on"), $"automatic: finished {automatic.Done}, trace:\n{automatic.Trace}");
            var checkpointed = Train(true, null);
            Check(automatic.Losses.Count == checkpointed.Losses.Count, $"steps: {automatic.Losses.Count} and {checkpointed.Losses.Count}");
            CloseByNorm([.. checkpointed.Losses], [.. automatic.Losses], 1e-4f, "the same losses as checkpointed training");
            var roomy = Train(null, null);
            Check(roomy.Done && !roomy.Trace.Contains("checkpointing is on"), "with room, training runs without checkpointing");

            // A budget where recomputing the feed-forward activations (or also holding them as bfloat16) fits: that
            // setting and checkpointing are each timed, and training finishes with one of them.
            var plain = Enumerable.Range(1, 32).Select(i => budget!.Value + ((long)i << 23)).First(b => Train(false, b).Done);
            var lighter = Enumerable.Range(1, 15).Select(i => plain - (plain - budget!.Value) * i / 16)
                .Select(b => Train(null, b, epochs: 3)).FirstOrDefault(t => t.Trace.Contains("measuring:"));
            Check(lighter.Done && lighter.Trace.Contains("keeping "), $"a lighter setting is measured against checkpointing, one kept: {lighter.Trace}");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    // A tiny Qwen3-style model folder: byte-level tokenizer with ChatML tokens, Qwen3's chat template, random weights.
    private static void ModelDownload(Device device)
    {
        var spec = new DecoderSpec
        {
            Vocabulary = 260, Dim = 32, Layers = 2, Heads = 4, KvHeads = 2, HeadDim = 8, FfDim = 64, MaxPositions = 256,
            Rope = new RopeSettings(10000f), NormEpsilon = 1e-6f, QkNorm = true, TieEmbeddings = true,
        };
        string source = WriteChatModel(spec);
        string cache = Path.Combine(Path.GetTempPath(), "ns-models-" + Guid.NewGuid().ToString("N"));
        try
        {
            // Re-save the weights as two shards with an index, as larger models ship.
            foreach (var file in Directory.GetFiles(source, "*.safetensors"))
            {
                File.Delete(file);
            }

            var weights = new RandomWeights(91);
            using (spec.Build(weights, new DecoderBuildOptions { Device = Device.Cpu }))
            {
            }

            WriteCheckpoint(source, spec, weights, "Qwen3ForCausalLM", SafeTensorType.F32, sharded: true);
            var served = Directory.GetFiles(source).Select(Path.GetFileName).ToList();
            Check(served.Contains("model.safetensors.index.json") && served.Count(f => f!.EndsWith(".safetensors", StringComparison.Ordinal)) >= 2, "sharded source");

            const string hf = "https://huggingface.co", commit = "abcdefabcdefabcdefabcdefabcdefabcdefabcd";
            var web = new FakeRouter();
            web.Json($"{hf}/api/models/org/tiny/revision/main", $"{{\"sha\":\"{commit}\"}}");
            var listing = new JsonArray();
            foreach (var name in served.Concat(["pytorch_model.bin", "README.md", "onnx/model.onnx", "original/consolidated.safetensors"]))
            {
                listing.Add((JsonNode)new JsonObject { ["type"] = "file", ["path"] = name, ["size"] = 10 });
            }

            web.Json($"{hf}/api/models/org/tiny/tree/{commit}?recursive=true", listing.ToJsonString());
            foreach (var name in served)
            {
                web.Bytes($"{hf}/org/tiny/resolve/{commit}/{name}", File.ReadAllBytes(Path.Combine(source, name!)), requireToken: "hf_model");
            }

            var downloader = new Idrak.Datasets.Downloader(new HttpClient(web), cache) { Attempts = 1 };
            string folder = ModelSource.DownloadAsync("org/tiny", token: "hf_model", downloader: downloader).GetAwaiter().GetResult();
            Check(folder == Path.Combine(cache, "huggingface", "models", "org", "tiny", commit[..12]), $"model folder {folder}");
            Check(Directory.GetFiles(folder).Select(Path.GetFileName).Order().SequenceEqual(served.Order()), "only the files the library reads");

            using (var original = PretrainedModel.Load(source, new PretrainedOptions { Device = device }))
            using (var downloaded = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
            {
                int[] ids = [.. downloaded.Tokenizer!.Encode("hello there, general kenobi")];
                TrainingSequence[] probe = [new TrainingSequence(ids, [.. ids.Select(_ => true)])];
                AssertClose([FineTuner.Evaluate(original, probe)], [FineTuner.Evaluate(downloaded, probe)], 0f, "same loss as the original folder");
            }

            // Without a network, the downloaded copy is used.
            var offline = new Idrak.Datasets.Downloader(new HttpClient(new Unreachable()), cache) { Attempts = 1 };
            Check(ModelSource.Resolve("org/tiny", downloader: offline) == folder, "offline: the cached copy");
            Check(ModelSource.IsModelId("Qwen/Qwen3-0.6B") && !ModelSource.IsModelId(source) && !ModelSource.IsModelId("a/b/c"), "ids and folders");
        }
        finally
        {
            Directory.Delete(source, true);
            if (Directory.Exists(cache))
            {
                Directory.Delete(cache, true);
            }
        }
    }

    private sealed class Unreachable : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
            throw new HttpRequestException("No such host is known.");
    }

    private static string WriteChatModel(DecoderSpec spec)
    {
        string folder = TempFolder();
        var weights = new RandomWeights(91);
        using (spec.Build(weights, new DecoderBuildOptions { Device = Device.Cpu }))
        {
        }

        WriteCheckpoint(folder, spec, weights, "Qwen3ForCausalLM", SafeTensorType.F32, sharded: false);
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
            ["chat_template"] = Qwen3Template, ["eos_token"] = "<|im_end|>", ["bos_token"] = null,
        }.ToJsonString());
        return folder;
    }

    private static void ManifestRoundTrip(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        var spec = new DecoderSpec
        {
            Vocabulary = 260, Dim = 32, Layers = 1, Heads = 2, KvHeads = 1, HeadDim = 16, FfDim = 32, MaxPositions = 128,
            Rope = new RopeSettings(10000f), NormEpsilon = 1e-6f, QkNorm = true, TieEmbeddings = true,
        };
        string folder = WriteChatModel(spec), adapter = Path.Combine(folder, "adapter");
        string previous = Directory.GetCurrentDirectory();
        try
        {
            using (var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
            {
                model.AddAdapters(2, 4, ["q", "v"], seed: 5);
                foreach (var a in model.Network.Descendants().OfType<Linear>().Select(l => l.Adapter).OfType<LoraAdapter>())
                {
                    a.B.Load([.. Enumerable.Range(0, a.B.Size).Select(i => 0.05f * MathF.Sin(i))]);
                }

                model.SaveAdapter(adapter);
            }

            // Named relative to the working directory, as a user types it.
            Directory.SetCurrentDirectory(Path.GetDirectoryName(folder)!);
            new TuningManifest { BaseModel = Path.GetFileName(folder), System = "Be brief.", MaxLength = 100 }.Save(adapter);
            Directory.SetCurrentDirectory(previous);
            var read = TuningManifest.Read(adapter)!;
            Check(read.BaseModel == folder && read.System == "Be brief." && read.MaxLength == 100 && TuningManifest.Exists(adapter),
                $"read back: {read.BaseModel}, {read.System}, {read.MaxLength}");
            Check(TuningManifest.Read(folder) is null, "a folder without a manifest has none");
            new TuningManifest { BaseModel = "owner/model" }.Save(adapter);
            Check(TuningManifest.Read(adapter)!.BaseModel == "owner/model", "a model id is kept as written");

            new TuningManifest { BaseModel = folder }.Save(adapter);
            using var merged = TuningManifest.Read(adapter)!.LoadModel(adapter, device);
            using var plain = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, MergeAdapter = adapter });
            plain.Network.Eval();
            using var scope = new TensorScope();
            var tokens = Tensor.From([1f, 2f, 3f, 4f], [1, 4], device);
            float[] a1, a2;
            using (Autograd.NoGrad())
            {
                a1 = merged.Network.Forward(tokens).ToArray();
                a2 = plain.Network.Forward(tokens).ToArray();
            }

            Check(a1.SequenceEqual(a2), "LoadModel is the base model with the adapter merged");
        }
        finally
        {
            Directory.SetCurrentDirectory(previous);
            Directory.Delete(folder, true);
        }
    }

    private static void LongMessageKeepsAnswer(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                              // tokenization only
        }

        var spec = new DecoderSpec
        {
            Vocabulary = 260, Dim = 32, Layers = 1, Heads = 2, KvHeads = 1, HeadDim = 16, FfDim = 32, MaxPositions = 512,
            Rope = new RopeSettings(10000f), NormEpsilon = 1e-6f, QkNorm = true, TieEmbeddings = true,
        };
        string folder = WriteChatModel(spec);
        try
        {
            using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
            var tokenizer = model.Tokenizer!;
            string message = "Please find the sales report " + string.Concat(Enumerable.Repeat("and the figures for every month ", 40));
            var transcript = new ChatTranscript([new ChatMessage("system", "Answer with one word."), new ChatMessage("user", message),
                new ChatMessage("assistant", "retrieve")], []);
            string Trained(TrainingSequence s) => tokenizer.Decode(s.Tokens.Where((_, i) => s.Trained[i]));

            var encoder = new ChatTranscriptEncoder(model.ChatTemplate!, tokenizer);
            var whole = encoder.Encode(transcript, 2000)!;
            Check(whole.Tokens.Length > 200, $"the conversation is long ({whole.Tokens.Length} tokens)");
            var fitted = encoder.Encode(transcript, 120);
            Check(fitted is not null && fitted.Tokens.Length <= 121 && Trained(fitted).EndsWith("retrieve<|im_end|>", StringComparison.Ordinal),
                $"shortened: {fitted?.Tokens.Length} tokens, trained '{(fitted is null ? "" : Trained(fitted))}'");
            var shortened = encoder.Fit(transcript, 120)!;
            string kept = shortened.Messages[1].Content;
            Check(kept.Length > 20 && kept.Length < message.Length && message.StartsWith(kept, StringComparison.Ordinal)
                  && shortened.Messages[0] == transcript.Messages[0] && shortened.Messages[2] == transcript.Messages[2],
                $"the message's start is kept ({kept.Length} of {message.Length} characters), the rest unchanged");
            Check(ReferenceEquals(encoder.Fit(transcript, 2000), transcript), "a transcript that fits is left as it is");

            // Many rows on all cores: the same sequences, in the same order, as one by one.
            var rows = Enumerable.Range(0, 400).Select(i => (JsonObject)JsonNode.Parse(i % 5 == 0
                ? $"{{\"text\": \"plain text number {i} {new string('x', i % 37)}\"}}"
                : $"{{\"messages\": [{{\"role\": \"user\", \"content\": \"question {i} {new string('y', i % 53)}\"}}, {{\"role\": \"assistant\", \"content\": \"answer {i}\"}}]}}")!).ToList();
            var parallel = encoder.EncodeRows(rows, 120).ToList();
            Check(parallel.Count == rows.Count && parallel.Select(p => p.Row).SequenceEqual(rows), "every row, in order");
            for (int i = 0; i < rows.Count; i++)
            {
                var one = encoder.EncodeRow(rows[i], 120).ToList();
                Check(one.Count == parallel[i].Sequences.Count && one.Zip(parallel[i].Sequences).All(p => p.First.Tokens.SequenceEqual(p.Second.Tokens)
                      && p.First.Trained.SequenceEqual(p.Second.Trained)), $"row {i}: the same sequences as encoded alone");
            }

            var cutting = new ChatTranscriptEncoder(model.ChatTemplate!, tokenizer) { ShortenToFit = false };
            Check(cutting.Encode(transcript, 120) is null, "without shortening, cutting the end leaves no answer to train");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    private static void AgentFineTuning(Device device)
    {
        var spec = new DecoderSpec
        {
            Vocabulary = 260, Dim = 32, Layers = 2, Heads = 4, KvHeads = 2, HeadDim = 8, FfDim = 64, MaxPositions = 1024,
            Rope = new RopeSettings(10000f), NormEpsilon = 1e-6f, QkNorm = true, TieEmbeddings = true,
        };
        string folder = WriteChatModel(spec), data = Path.Combine(folder, "train.jsonl"), adapter = Path.Combine(folder, "adapter"),
            exported = Path.Combine(folder, "merged");
        try
        {
            File.WriteAllLines(data,
            [
                """{"messages": [{"role": "system", "content": "You fix code."}, {"role": "user", "content": "Fix a.py"}, {"role": "assistant", "content": "", "reasoning_content": "Read it.", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{\"path\": \"a.py\"}"}}]}, {"role": "tool", "tool_call_id": "c1", "content": "x = 1/0"}, {"role": "assistant", "content": "Divide by one."}], "tools": [{"type": "function", "function": {"name": "read", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]}""",
                "",
                """{"messages": [{"role": "user", "content": [{"type": "text", "text": "Say hi"}]}, {"role": "assistant", "content": "hi"}]}""",
                """{"conversations": [{"from": "human", "value": "2+2?"}, {"from": "gpt", "value": "4"}]}""",
            ]);
            var transcripts = ChatTranscript.ReadJsonLines(data).ToList();
            Check(transcripts.Count == 3 && transcripts[0].Tools.Count == 1 && transcripts[0].Messages[2].ToolCalls![0].Name == "read"
                  && (string?)transcripts[0].Messages[2].ToolCalls![0].Arguments["path"] == "a.py" && transcripts[0].Messages[3].ToolName == "read"
                  && transcripts[0].Messages[2].Thinking == "Read it." && transcripts[1].Messages[0].Content == "Say hi" && transcripts[2].Messages[1].Role == "assistant",
                "transcripts parse (tools, string arguments, tool_call_id, reasoning, content parts, ShareGPT)");

            using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
            var encoder = new ChatTranscriptEncoder(model.ChatTemplate!, model.Tokenizer!);
            Check(encoder.AssistantHeader == "<|im_start|>assistant\n" && encoder.AssistantEnd == "<|im_end|>",
                $"assistant markers '{encoder.AssistantHeader}' / '{encoder.AssistantEnd}'");
            var sequences = transcripts.Select(t => encoder.Encode(t, 1000)!).ToList();
            var tokenizer = model.Tokenizer!;
            string Trained(TrainingSequence s) => tokenizer.Decode(s.Tokens.Where((_, i) => s.Trained[i]));
            string first = Trained(sequences[0]);
            Check(first.Contains("<tool_call>") && first.Contains("\"path\": \"a.py\"") && first.Contains("Divide by one.<|im_end|>")
                  && first.Contains("Read it.") && !first.Contains("Fix a.py") && !first.Contains("x = 1/0") && !first.Contains("You fix code"),
                $"only the assistant's turns are trained: {first}");
            Check(Trained(sequences[1]).EndsWith("hi<|im_end|>") && sequences.All(s => s.Tokens.Length <= 1001), "short transcripts");
            Check(encoder.Encode(transcripts[0], 20) is null || encoder.Encode(transcripts[0], 20)!.Tokens.Length == 21, "cut to the maximum length");

            // Rows from datasets: conversations as before; plain text trains every token, long texts in chunks.
            var row = JsonNode.Parse(File.ReadLines(data).First())!.AsObject();
            Check(encoder.EncodeRow(row, 1000).Single().Tokens.SequenceEqual(sequences[0].Tokens), "a conversation row encodes as its transcript");
            string text = string.Concat(Enumerable.Repeat("the quick brown fox jumps over the lazy dog. ", 6));
            var whole = encoder.EncodeText(text, 10_000).Single();
            var chunks = encoder.EncodeRow(new JsonObject { ["text"] = text }, 16).ToList();
            Check(whole.Trained.All(t => t) && chunks.Count > 1 && chunks.All(c => c.Tokens.Length <= 17 && c.Trained.All(t => t))
                  && chunks.Sum(c => c.TrainedTokens) == whole.TrainedTokens && chunks[0].Tokens.Concat(chunks[1].Tokens.Skip(1)).SequenceEqual(whole.Tokens.Take(chunks[0].Tokens.Length + chunks[1].Tokens.Length - 1)),
                $"text rows: {chunks.Count} chunks predict each of the {whole.TrainedTokens} tokens once");
            Check(!encoder.EncodeRow(new JsonObject { ["label"] = 3 }, 100).Any(), "rows without text or messages give nothing");

            // LoRA: the transcripts are learned (loss falls), and the adapters round-trip through the PEFT files.
            var options = new FineTuningOptions { Rank = 4, Alpha = 8, LearningRate = 1e-2f, Epochs = 40, BatchTokens = 1024, WarmupFraction = 0f, Seed = 5 };
            float before = FineTuner.Evaluate(model, sequences);
            var progress = new List<FineTuningProgress>();
            FineTuner.Train(model, sequences, null, options, adapter, new Progress<FineTuningProgress>(progress.Add));
            float after = FineTuner.Evaluate(model, sequences);
            Check(after < before * 0.5f, $"loss {before:F3} → {after:F3}");
            Check(File.Exists(Path.Combine(adapter, "adapter_model.safetensors")) && File.Exists(Path.Combine(adapter, "adapter_config.json")), "PEFT files");
            var peft = JsonNode.Parse(File.ReadAllText(Path.Combine(adapter, "adapter_config.json")))!;
            Check((int)peft["r"]! == 4 && (float)peft["lora_alpha"]! == 8f && peft["target_modules"]!.AsArray().Count == 7, $"adapter config {peft}");
            using (var reader = SafeTensorsReader.Open(Path.Combine(adapter, "adapter_model.safetensors")))
            {
                Check(reader.Tensors["base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight"].Shape.SequenceEqual([4, 32])
                      && reader.Tensors["base_model.model.model.layers.1.mlp.down_proj.lora_B.weight"].Shape.SequenceEqual([32, 4]), "PEFT names and layouts");
            }

            using (var reloaded = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
            {
                Check(reloaded.LoadAdapter(adapter) == 7 * spec.Layers, "every adapter loads");
                AssertClose([after], [FineTuner.Evaluate(reloaded, sequences)], 1e-4f, "loss with the reloaded adapters");
                reloaded.SaveHuggingFace(exported, SafeTensorType.F32);
            }

            using (var merged = PretrainedModel.Load(exported, new PretrainedOptions { Device = device }))
            {
                Check(!merged.Network.Descendants().OfType<Linear>().Any(l => l.Adapter is not null), "the export has no adapters");
                AssertClose([after], [FineTuner.Evaluate(merged, sequences)], 1e-3f, "loss of the merged export");
            }

            using (var merged = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, MergeAdapter = adapter }))
            {
                Check(!merged.Network.Descendants().OfType<Linear>().Any(l => l.Adapter is not null)
                      && merged.Notes.Any(n => n.Contains($"merged into {7 * spec.Layers} weights")), "merged while loading");
                AssertClose([after], [FineTuner.Evaluate(merged, sequences)], 1e-3f, "loss with the adapter merged while loading");
            }

            using (var merged = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, MergeAdapter = adapter, Int8 = true }))
            {
                float loss = FineTuner.Evaluate(merged, sequences);
                Check(loss < before * 0.5f, $"int8 weights with the adapter merged while loading: {loss:F3}");
            }

            // QLoRA: adapters on a 4-bit base also learn.
            using var quantized = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, Int4 = true });
            float q0 = FineTuner.Evaluate(quantized, sequences);
            FineTuner.Train(quantized, sequences, sequences, options with { Epochs = 25 });
            float q1 = FineTuner.Evaluate(quantized, sequences);
            Check(q1 < q0 * 0.6f, $"QLoRA loss {q0:F3} → {q1:F3}");
            Check(progress.Count > 0 && progress[^1].Step == progress[^1].TotalSteps && progress.All(p => float.IsFinite(p.Loss)), "progress reports");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void TokenLoss(Device device)
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
            head.Adapter!.B.Load(adapterB);
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
            adapterGrad = head.Adapter.A.Grad!.ToArray();
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

        // A frozen matrix used through a transposed bfloat16 copy: the product of x and the table rounded to bfloat16 (as the
        // tensor cores round it), and the input gradient through the table as stored.
        var tableValues = Random(Vocabulary * Dim);
        static float Round(float v)
        {
            int bits = BitConverter.SingleToInt32Bits(v);
            bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & unchecked((int)0xFFFF0000);
            return BitConverter.Int32BitsToSingle(bits);
        }

        float[] Tied(bool cached, out float[] inputGrad)
        {
            using var scope = new TensorScope();
            var table = Tensor.From(tableValues, [Vocabulary, Dim], device);
            var x = Tensor.From(hiddenValues, [Rows, Dim], device, requiresGrad: true);
            if (cached)
            {
                using var transposed = BFloat16Weight.FromValues([.. Enumerable.Range(0, Dim * Vocabulary).Select(i => tableValues[i % Vocabulary * Dim + i / Vocabulary])], Dim, Vocabulary, device);
                var y = Tensor.MatMulFrozenTransposed(x, table, transposed);
                (y * y).Sum().Backward();
                inputGrad = x.Grad!.ToArray();
                return y.ToArray();
            }

            // Reference: y with the rounded table, dx = 2y · E with the table as stored.
            var rounded = Tensor.From([.. tableValues.Select(Round)], [Vocabulary, Dim], device);
            using (Autograd.NoGrad())
            {
                var y = x.MatMul(rounded, transposeB: true);
                inputGrad = (y * 2f).MatMul(table).ToArray();
                return y.ToArray();
            }
        }

        AssertClose(Tied(false, out var plainGrad), Tied(true, out var cachedGrad), 1e-4f, "frozen transposed product (bfloat16 copy)");
        AssertClose(plainGrad, cachedGrad, 1e-3f, "frozen transposed product: input gradient");

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

    private static void LoraInsideProducts(Device device)
    {
        const int Batch = 2, Steps = 100, In = 96, Out = 384, Rank = 16;
        var random = new Random(71);
        float[] Values(int n) => RandomArray(random, n);
        if (device.Type != DeviceType.Cuda || MixedPrecision.TensorCoresUnavailable(device) is not null)
        {
            // Devices without the fused kernels decline, and callers compute the term separately.
            using var scope = new TensorScope();
            var layer = new Linear(In, Out, bias: false, device: device, random: random);
            layer.Adapter = new LoraAdapter(Tensor.From(Values(In * Rank), [In, Rank], device), Tensor.From(Values(Rank * Out), [Rank, Out], device), Rank, 2f);
            Check(Tensor.LoraProducts(Tensor.From(Values(Steps * In), [Steps, In], device), [layer]) is null, "no fused LoRA products on this device");
            return;
        }

        static double RelativeError(float[] expected, float[] actual)
        {
            double difference = 0, norm = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                difference += (double)(expected[i] - actual[i]) * (expected[i] - actual[i]);
                norm += (double)expected[i] * expected[i];
            }

            return Math.Sqrt(difference / Math.Max(norm, 1e-30));
        }

        void Close(float[] expected, float[] actual, string what, double tolerance = 2e-2)
        {
            Check(expected.Length == actual.Length, $"{what}: length {actual.Length}, expected {expected.Length}");
            double error = RelativeError(expected, actual);
            Check(error < tolerance, $"{what}: relative error {error:G3}");
        }

        using var precision = MixedPrecision.BFloat16();

        // Skinny products (one side 16) on tensor cores: as the CPU computes them, in every layout.
        foreach (var (m, n, k, ta, tb) in new[] { (300, 16, 2000, false, false), (2000, 16, 300, true, false), (300, 16, 2000, false, true),
                     (16, 2000, 300, true, false), (300, 2000, 16, false, false), (300, 2000, 16, false, true) })
        {
            var av = Values(m * k);
            var bv = Values(k * n);
            var cv = Values(m * n);
            float[] Product(Device on, float beta)
            {
                using var a = Tensor.From(av, ta ? [k, m] : [m, k], on);
                using var b = Tensor.From(bv, tb ? [n, k] : [k, n], on);
                using var c = Tensor.From(cv, [m, n], on);
                on.Backend.BatchedMatMul(a.Storage, b.Storage, c.Storage, 1, m, n, k, ta, tb, beta);
                return c.ToArray();
            }

            foreach (float beta in new[] { 0f, 1f })
            {
                Close(Product(Device.Cpu, beta), Product(device, beta), $"{m}x{n}x{k} {(ta ? 't' : 'n')}{(tb ? 't' : 'n')} beta {beta}");
            }
        }

        bool fp8Available = device.Backend.Float8PaddedK(In) > 0;
        foreach (string format in new[] { "float32", "bfloat16", "int4", "bfloat16 + fp8", "float32 + fp8" })
        {
            bool fp8 = format.EndsWith("fp8", StringComparison.Ordinal);
            if (fp8 && !fp8Available)
            {
                continue;
            }

            foreach (int count in format.StartsWith("float32", StringComparison.Ordinal) ? new[] { 1 } : new[] { 1, 2, 3 })
            {
                var weights = Enumerable.Range(0, count).Select(_ => Values(In * Out)).ToArray();
                var aValues = Enumerable.Range(0, count).Select(_ => Values(In * Rank)).ToArray();
                var bValues = Enumerable.Range(0, count).Select(_ => Values(Rank * Out)).ToArray();
                var coefficients = Enumerable.Range(0, count).Select(_ => Values(Batch * Steps * Out)).ToArray();
                var xValues = Values(Batch * Steps * In);
                float[][] Run(bool fused)
                {
                    using var scope = new TensorScope();
                    var layers = new Linear[count];
                    for (int j = 0; j < count; j++)
                    {
                        layers[j] = Linear.FromWeights(Tensor.From(weights[j], [In, Out], device));
                        if (format.StartsWith("bfloat16", StringComparison.Ordinal))
                        {
                            layers[j].ToBFloat16();
                        }
                        else if (format == "int4")
                        {
                            layers[j].QuantizeInt4();
                        }

                        layers[j].Adapter = new LoraAdapter(Tensor.From(aValues[j], [In, Rank], device, requiresGrad: true),
                            Tensor.From(bValues[j], [Rank, Out], device, requiresGrad: true), Rank, 1.5f);
                    }

                    var x = Tensor.From(xValues, [Batch, Steps, In], device, requiresGrad: true);
                    Tensor[] ys;
                    if (fused)
                    {
                        if (fp8)
                        {
                            Check(layers.All(l => l.AttachFloat8()), $"{format}: FP8 copies of the weights");
                        }

                        ys = Tensor.LoraProducts(x, layers) ?? throw new Exception($"{format} × {count}: no fused LoRA products on {device}");
                        foreach (var layer in layers)
                        {
                            layer.DetachFloat8();
                        }
                    }
                    else
                    {
                        ys = [.. layers.Select(l =>
                        {
                            var product = l.BFloat16 is { } h ? x.MatMulBFloat16(h) : l.Int4 is { } q ? x.MatMulInt4(q) : x.MatMul(l.Weight);
                            return product + x.MatMul(l.Adapter!.A).MatMul(l.Adapter.B) * l.Adapter.Scale;
                        })];
                    }

                    var loss = ys.Select((y, j) => (y * Tensor.From(coefficients[j], [Batch, Steps, Out], device)).Sum()).Aggregate((p, q) => p + q);
                    loss.Backward();
                    return [.. ys.Select(y => y.ToArray()), x.Grad!.ToArray(), .. layers.Select(l => l.Adapter!.A.Grad!.ToArray()),
                        .. layers.Select(l => l.Adapter!.B.Grad!.ToArray())];
                }

                var expected = Run(false);
                var actual = Run(true);
                for (int i = 0; i < expected.Length; i++)
                {
                    string part = i < count ? $"output {i}" : i == count ? "input gradient" : i <= 2 * count ? $"A gradient {i - count - 1}" : $"B gradient {i - 2 * count - 1}";
                    Close(expected[i], actual[i], $"{format} × {count}: {part}", fp8 && i < count ? 1e-1 : 2e-2);
                }
            }
        }
    }

    private static void PackedSequencesMatch(Device device)
    {
        if (device.Type == DeviceType.Cuda && MixedPrecision.TensorCoresUnavailable(device) is not null)
        {
            return;                                                              // packed attention runs on tensor cores
        }

        using var precision = MixedPrecision.Use(device.Type == DeviceType.Cuda ? MatMulPrecision.BFloat16 : MatMulPrecision.Float32);
        int[][] rows = [[37, 50, 20], [70, 30]];
        const int Length = 110;
        var random = new Random(97);
        var tokens = rows.Select(r => r.Select(n => Enumerable.Range(0, n).Select(_ => (float)random.Next(50)).ToArray()).ToArray()).ToArray();
        var coefficients = rows.Select(r => r.Select(n => RandomArray(random, n * 50)).ToArray()).ToArray();
        foreach (bool rotary in new[] { true, false })
        {
            var spec = new DecoderSpec
            {
                Vocabulary = 50, Dim = 128, Layers = 2, Heads = 4, KvHeads = 2, HeadDim = 64, FfDim = 128, MaxPositions = 256,
                Rope = rotary ? new RopeSettings(10000f) : null, LearnedPositions = !rotary,
            };
            using var model = spec.Build(new RandomWeights(98), new DecoderBuildOptions { Device = device });
            model.AddLora(rank: 2, alpha: 4, targets: _ => true, freezeBase: true, random: new Random(99));
            foreach (var adapter in model.Descendants().OfType<Linear>().Select(l => l.Adapter).OfType<LoraAdapter>())
            {
                adapter.B.Load([.. Enumerable.Range(0, adapter.B.Size).Select(i => 0.1f * MathF.Sin(i))]);
            }

            Check(PackedSequences.Supports(model), "the model runs packed batches");
            model.Train();

            // Packed: both rows in one [2, 110] batch (the ends of the rows are padding).
            var packedLogits = new List<float[]>();
            float[] packedGradients;
            float packedLoss;
            using (var scope = new TensorScope())
            {
                var values = new float[rows.Length * Length];
                var weights = new float[rows.Length * Length * 50];
                for (int r = 0; r < rows.Length; r++)
                {
                    int offset = 0;
                    for (int j = 0; j < rows[r].Length; j++)
                    {
                        tokens[r][j].CopyTo(values, r * Length + offset);
                        coefficients[r][j].CopyTo(weights, (r * Length + offset) * 50);
                        offset += rows[r][j];
                    }
                }

                using var packing = PackedSequences.Create(rows, Length, device);
                using (packing.Use())
                {
                    var logits = model.Forward(Tensor.From(values, [rows.Length, Length], device));
                    var loss = (logits * Tensor.From(weights, [rows.Length, Length, 50], device)).Sum();
                    loss.Backward();
                    packedLoss = loss.Item();
                    var all = logits.ToArray();
                    for (int r = 0; r < rows.Length; r++)
                    {
                        int offset = 0;
                        foreach (int n in rows[r])
                        {
                            packedLogits.Add(all[((r * Length + offset) * 50)..((r * Length + offset + n) * 50)]);
                            offset += n;
                        }
                    }
                }

                packedGradients = [.. model.TrainableParameters().SelectMany(p => p.Grad!.ToArray())];
            }

            // Alone: each sequence as its own batch, gradients summed; in the test's precision, and (on CUDA) in float32 too,
            // which measures how far bfloat16 alone moves the results.
            (List<float[]> Logits, float Loss, float[] Gradients) Alone(MatMulPrecision? precision)
            {
                using var scoped = precision is { } p ? MixedPrecision.Use(p) : default(MixedPrecision.Scope?);
                foreach (var parameter in model.TrainableParameters())
                {
                    parameter.ZeroGrad();
                }

                var logitsList = new List<float[]>();
                float total = 0f;
                using var scope = new TensorScope();
                for (int r = 0; r < rows.Length; r++)
                {
                    for (int j = 0; j < rows[r].Length; j++)
                    {
                        int n = rows[r][j];
                        var logits = model.Forward(Tensor.From(tokens[r][j], [1, n], device));
                        var loss = (logits * Tensor.From(coefficients[r][j], [1, n, 50], device)).Sum();
                        loss.Backward();
                        total += loss.Item();
                        logitsList.Add(logits.ToArray());
                    }
                }

                return (logitsList, total, [.. model.TrainableParameters().SelectMany(p => p.Grad!.ToArray())]);
            }

            var alone = Alone(null);
            var exact = device.Type == DeviceType.Cuda ? Alone(MatMulPrecision.Float32) : alone;
            string what = rotary ? "rotary positions" : "learned positions";
            for (int i = 0; i < alone.Logits.Count; i++)
            {
                WithinPrecision(exact.Logits[i], alone.Logits[i], packedLogits[i], device, $"{what}: logits of sequence {i}");
            }

            // The loss sums logits times random coefficients of both signs, which nearly cancel: its error is bounded by the
            // logits' error times the size of the terms (Cauchy-Schwarz), not by the loss itself.
            var weightsBySequence = rows.SelectMany((row, r) => row.Select((_, j) => coefficients[r][j])).ToList();
            double terms = exact.Logits.Select((l, i) => Norm(l) * Norm(weightsBySequence[i])).Sum();
            float own = device.Type == DeviceType.Cuda ? exact.Logits.Select((l, i) => RelativeError(l, alone.Logits[i])).Max() : 0f;
            float allowed = Math.Max(device.Type == DeviceType.Cuda ? 1e-2f : 1e-3f, 3f * own);
            Check(Math.Abs(packedLoss - exact.Loss) <= allowed * terms,
                $"{what}: loss {packedLoss} against {exact.Loss} (float32), difference {Math.Abs(packedLoss - exact.Loss):G3}, allowed {allowed * terms:G3}");
            WithinPrecision(exact.Gradients, alone.Gradients, packedGradients, device, $"{what}: adapter gradients");
        }

        // First fit, longest first: 6+4, 5+5, 3+2 in rows of 10.
        var sequences = new[] { 5, 3, 6, 4, 5, 2 }.Select(n => new TrainingSequence(new int[n + 1], new bool[n + 1])).ToList();
        var batches = FineTuner.PackedBatches(sequences, 10, 2, random: null);
        Check(batches.Count == 2 && batches[0].Length == 2 && batches[1].Length == 1, $"packed batches: {batches.Count}");
        var filled = batches.SelectMany(b => b).Select(r => r.Sum(i => sequences[i].Tokens.Length - 1)).ToArray();
        Check(filled.SequenceEqual([10, 10, 5]) && batches.SelectMany(b => b).SelectMany(r => r).Order().SequenceEqual(Enumerable.Range(0, 6)),
            $"rows filled {string.Join(", ", filled)}");
    }

    // actual (from a batched, packed or ragged path) against exact (float32, the plain path): on the CPU within 1e-3; on
    // CUDA within three times the error bfloat16 itself gives the plain path (reduced, its own error), at least 1e-2. A wrong
    // position, mask or start gives errors of order one; rounding differences stay near the plain path's own.
    private static void WithinPrecision(float[] exact, float[] reduced, float[] actual, Device device, string what)
    {
        if (device.Type != DeviceType.Cuda)
        {
            CloseByNorm(exact, actual, 1e-3f, what);
            return;
        }

        float own = RelativeError(exact, reduced);
        float error = RelativeError(exact, actual);
        Check(error <= Math.Max(1e-2f, 3f * own), $"{what}: relative error {error:G3} against float32; bfloat16 alone gives {own:G3}");
    }

    private static double Norm(float[] values) => Math.Sqrt(values.Sum(v => (double)v * v));

    private static float RelativeError(float[] expected, float[] actual)
    {
        double difference = 0, norm = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            difference += (double)(expected[i] - actual[i]) * (expected[i] - actual[i]);
            norm += (double)expected[i] * expected[i];
        }

        return (float)Math.Sqrt(difference / Math.Max(norm, 1e-30));
    }

    // ‖expected - actual‖ ≤ tolerance · ‖expected‖ (products in bfloat16 differ element by element, not overall).
    private static void CloseByNorm(float[] expected, float[] actual, float tolerance, string what)
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

    private static void GraphTraining(Device device)
    {
        if (device.Type != DeviceType.Cuda || !device.Backend.SupportsGraphs || MixedPrecision.TensorCoresUnavailable(device) is not null)
        {
            return;                                                              // graphs: CUDA; packed attention: tensor cores
        }

        using var precision = MixedPrecision.BFloat16();
        var spec = new DecoderSpec
        {
            Vocabulary = 260, Dim = 128, Layers = 2, Heads = 2, KvHeads = 1, HeadDim = 64, FfDim = 256, MaxPositions = 512,
            Rope = new RopeSettings(10000f), NormEpsilon = 1e-6f, QkNorm = true, TieEmbeddings = true,   // a Qwen3 checkpoint: query/key norms
        };
        string folder = WriteChatModel(spec);
        try
        {
            var random = new Random(61);
            var sequences = Enumerable.Range(0, 24).Select(_ =>
            {
                int n = random.Next(20, 120);
                return new TrainingSequence([.. Enumerable.Range(0, n).Select(_ => random.Next(256))], [.. Enumerable.Range(0, n).Select(i => i >= n / 2)]);
            }).ToList();
            (List<float> Losses, float[] Adapters, string Trace) Run(bool graphs)
            {
                using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
                var options = new FineTuningOptions { Rank = 4, Alpha = 8, LearningRate = 1e-3f, BatchTokens = 256, Seed = 1, CudaGraphs = graphs };
                var losses = new List<float>();
                var trace = new System.Text.StringBuilder();
                FineTuner.Train(model, sequences, null, options, progress: new SynchronousProgress<FineTuningProgress>(p => losses.Add(p.Loss)),
                    trace: line => trace.AppendLine(line));
                return (losses, [.. model.Network.TrainableParameters().SelectMany(p => p.ToArray())], trace.ToString());
            }

            var ordinary = Run(false);
            var replayed = Run(true);
            Check(replayed.Trace.Contains("recorded one training step as a CUDA graph") && replayed.Trace.Contains("replayed"),
                $"the graph was recorded and replayed:\n{replayed.Trace}");
            Check(ordinary.Losses.Count >= 5 && ordinary.Losses.Count == replayed.Losses.Count, $"steps: {ordinary.Losses.Count} and {replayed.Losses.Count}");
            CloseByNorm([.. ordinary.Losses], [.. replayed.Losses], 2e-3f, "losses per step");
            CloseByNorm(ordinary.Adapters, replayed.Adapters, 2e-3f, "trained adapters");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    // Reports on the calling thread (Progress<T> posts to the thread pool, so its reports can arrive late).
    private sealed class SynchronousProgress<T>(Action<T> report) : IProgress<T>
    {
        public void Report(T value) => report(value);
    }

    private static void RaggedBatchDecoding(Device device)
    {
        if (device.Type == DeviceType.Cuda && MixedPrecision.TensorCoresUnavailable(device) is not null)
        {
            return;                                                              // per-row starts: tensor-core attention
        }

        using var precision = MixedPrecision.Use(device.Type == DeviceType.Cuda ? MatMulPrecision.BFloat16 : MatMulPrecision.Float32);
        var spec = new DecoderSpec
        {
            Vocabulary = 260, Dim = 128, Layers = 2, Heads = 4, KvHeads = 2, HeadDim = 64, FfDim = 128, MaxPositions = 256,
            Rope = new RopeSettings(10000f), NormEpsilon = 1e-6f, TieEmbeddings = true,
        };
        var random = new Random(41);
        int[] lengths = [5, 12, 9];
        var prompts = lengths.Select(n => Enumerable.Range(0, n).Select(_ => (float)random.Next(250)).ToArray()).ToArray();
        var next = lengths.Select(_ => Enumerable.Range(0, 4).Select(_ => (float)random.Next(250)).ToArray()).ToArray();
        using (var model = spec.Build(new RandomWeights(42), new DecoderBuildOptions { Device = device }))
        using (Autograd.NoGrad())
        {
            model.Eval();
            int longest = lengths.Max();
            var batched = new List<float[]>[lengths.Length];
            using (var scope = new TensorScope())
            using (var context = new DecodingContext(device, lengths.Length, 32))
            {
                context.SetRowStarts([.. lengths.Select(n => longest - n)]);
                context.LastPositionOnly = true;
                var input = new float[lengths.Length * longest];
                for (int r = 0; r < lengths.Length; r++)
                {
                    prompts[r].CopyTo(input, r * longest + longest - lengths[r]);
                    batched[r] = [];
                }

                void Collect(Tensor logits)
                {
                    var values = logits.ToArray();
                    for (int r = 0; r < lengths.Length; r++)
                    {
                        batched[r].Add(values[(r * 260)..((r + 1) * 260)]);
                    }
                }

                Collect(model.ForwardCached(Tensor.From(input, [lengths.Length, longest], device), context));
                for (int s = 0; s < 4; s++)
                {
                    Collect(model.ForwardCached(Tensor.From([.. next.Select(n => n[s])], [lengths.Length, 1], device), context));
                }
            }

            // Each row alone: exact (float32, the plain path), and on CUDA also through the batched path's own bfloat16 kernels
            // (per-row starts, one row starting at 0), which measures how far bfloat16 alone moves that computation.
            List<float[]> Alone(int r, MatMulPrecision? precision, bool rowStarts)
            {
                using var scoped = precision is { } p ? MixedPrecision.Use(p) : default(MixedPrecision.Scope?);
                using var scope = new TensorScope();
                using var context = new DecodingContext(device, 1, 32) { LastPositionOnly = true };
                if (rowStarts)
                {
                    context.SetRowStarts([0]);
                }

                var steps = new List<float[]> { model.ForwardCached(Tensor.From(prompts[r], [1, lengths[r]], device), context).ToArray() };
                for (int s = 0; s < 4; s++)
                {
                    steps.Add(model.ForwardCached(Tensor.From([next[r][s]], [1, 1], device), context).ToArray());
                }

                return steps;
            }

            for (int r = 0; r < lengths.Length; r++)
            {
                bool cuda = device.Type == DeviceType.Cuda;
                var exact = Alone(r, cuda ? MatMulPrecision.Float32 : null, rowStarts: false);
                var alone = cuda ? Alone(r, null, rowStarts: true) : exact;
                for (int s = 0; s < alone.Count; s++)
                {
                    WithinPrecision(exact[s], alone[s], batched[r][s], device, $"row {r} (prompt of {lengths[r]}), step {s}: logits");
                }
            }
        }

        if (device.Type != DeviceType.Cpu)
        {
            return;                                                              // greedy choices can flip on bfloat16 near-ties
        }

        // Whole greedy replies through the chat template (a Qwen3-style model): batched as generated one by one.
        string folder = WriteChatModel(spec with { QkNorm = true });
        try
        {
            using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
            var chat = model.CreateChat();
            var options = new GenerationOptions { Temperature = 0f, TopK = 1, TopP = 1f, RepeatPenalty = 1f, NumPredict = 12, NumCtx = 200 };
            var requests = new[] { "hi", "tell me a longer story about the sea", "2+2?" }
                .Select(q => new ChatRequest([new ChatMessage("user", q)], null, null, options)).ToList();
            var together = chat.ChatBatch(requests);
            for (int i = 0; i < requests.Count; i++)
            {
                var alone = chat.Chat(requests[i]);
                Check(together[i].Message!.Content == alone.Message!.Content && together[i].DoneReason == alone.DoneReason,
                    $"reply {i}: batched '{together[i].Message!.Content}' ({together[i].DoneReason}), alone '{alone.Message!.Content}' ({alone.DoneReason})");
            }

            // Streamed together, a token at a time: the pieces add up to the batched replies, each request ends once.
            foreach (int chunkSize in new[] { 1, 3 })
            {
                var streamed = requests.Select(r => r with { Options = r.Options! with { ChunkSize = chunkSize } }).ToList();
                var content = requests.Select(_ => new System.Text.StringBuilder()).ToArray();
                var finals = new ChatChunk?[requests.Count];
                foreach (var (index, chunk) in chat.StreamBatch(streamed))
                {
                    Check(finals[index] is null, $"chunk size {chunkSize}, reply {index}: nothing after its final chunk");
                    content[index].Append(chunk.Delta.Content);
                    finals[index] = chunk.Done ? chunk : null;
                }

                for (int i = 0; i < requests.Count; i++)
                {
                    Check(finals[i] is { } last && last.Message!.Content == together[i].Message!.Content && last.DoneReason == together[i].DoneReason
                          && content[i].ToString().Trim() == together[i].Message!.Content,
                        $"chunk size {chunkSize}, reply {i}: streamed '{content[i]}' ({finals[i]?.DoneReason}), batched '{together[i].Message!.Content}'");
                }
            }

            // Raw text with a stop sequence: the streamed pieces are the batched texts, cut before the stop.
            var raw = new[] { "once upon", "a b c d", "zzz" };
            var stopping = options with { Stop = ["e"], ChunkSize = 1 };
            var texts = chat.Generator.GenerateBatch(raw, stopping);
            var pieces = raw.Select(_ => new System.Text.StringBuilder()).ToArray();
            foreach (var chunk in chat.Generator.StreamBatch(raw, stopping))
            {
                pieces[chunk.Index].Append(chunk.Text);
            }

            for (int i = 0; i < raw.Length; i++)
            {
                Check(pieces[i].ToString() == texts[i].Text && !texts[i].Text.Contains('e'), $"text {i}: streamed '{pieces[i]}', batched '{texts[i].Text}' ({texts[i].DoneReason})");
            }
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }
}
