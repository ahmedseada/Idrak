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
        ("scoring: given answers to chat prompts are scored as the model's own full forward pass scores them, each prompt run once from a cache or with every answer, whatever the batching; long messages are shortened to fit", AnswerScoring),
        ("fine-tuning: releasing results no backward step reads, and recomputing feed-forward activations, give the same loss and gradients with less memory (RoPE, biases, q/k norms, post norms, parallel blocks, layer norms, dropout)", ReleasedActivations),
        ("fine-tuning: checkpointing is off by default and turns on (the step run again) when a step runs out of memory; a lighter setting that fits is timed against checkpointing and the faster kept", AutomaticCheckpointing),
        ("fine-tuning: when the first step's bytes per position say the largest batch will not fit, recomputed feed-forward activations are on before any step fails; with room nothing changes", FirstRungFromMeasuredBytes),
        ("fine-tuning: the activation estimate per rung grows with rows × positions and falls down the ladder (held, recomputed feed-forward, checkpointing); a rung round-trips through the stop state", ActivationEstimatePerRung),
        ("fine-tuning: with the activation estimate the first step starts at the rung that fits a small device (no out-of-memory step-down); with it off the ladder steps down; a resumed session starts at the saved rung", ActivationEstimateFirstRung),
        ("fine-tuning: a model tuned in place goes back as found: a loaded adapter leaves the base frozen (a session over it trains the adapter alone), RemoveAdapters frees every adapter and its gradients and the model scores as its base again; BatchTokensThatFit halves the batch until the estimate fits the device's room, never below one sequence", AdaptersInPlace),
        ("models: a Hugging Face model id is downloaded (only the files the library reads, sharded weights) into the cache with its bytes of total told, loads with its tensors read of total told, and works offline", ModelDownload),
        ("models: offline, a model id takes the newest finished download, never a folder a pull left unfinished (its pull marker, or a partial file); a download that fails keeps its marker", OfflineSkipsUnfinished),
        ("fine-tuning: agent transcripts through the chat template, assistant-only tokens, LoRA training, PEFT adapters, merged export", AgentFineTuning),
        ("fine-tuning: a conversation too long for the maximum length keeps its whole answer (the user message is shortened, its start kept); an answer longer than the budget alone gives the end cut at once; rows encode on all cores in order", LongMessageKeepsAnswer),
        ("fine-tuning: an adapter folder's manifest round-trips, names a local base model by its full path, and loads the model with the adapter merged", ManifestRoundTrip),
        ("fine-tuning options: AdamW given explicitly matches the default; another optimizer's recorded steps match its ordinary ones; an optimizer that cannot be recorded runs ordinary steps", OptimizerAndSchedule),
        ("fine-tuning options: the token cross-entropy as a loss delegate trains as the fused default does (with gradient accumulation)", LossHook),
        ("fine-tuning options: a step whose loss is not finite drops its gradients (the adapters keep their values); past NonFiniteSteps in a row the run stops", NonFiniteLossSkipped),
        ("fine-tuning options: DPO, conservative DPO, ORPO and SimPO match their formulas for fixed log-probabilities (and DPO its gradient)", PreferenceFormulas),
        ("fine-tuning options: DPO (reference: adapters disabled), ORPO and SimPO lower their loss on preference pairs; DPO starts at log 2; disabled adapters give the base model", PreferenceTraining),
        ("fine-tuning options: preference rows in TRL's layouts read as one shape; only the answers train", PreferenceRows),
        ("fine-tuning options: a DoRA layer gives the reference output and magnitude gradient; merging keeps the output", DoraLayer),
        ("fine-tuning options: DoRA adapters start as the identity, train, round-trip through the PEFT files (use_dora, magnitude vectors), merge while loading and export", DoraTraining),
        ("fine-tuning options: PEFT adapters with another peft_type, unsupported options or tensors for layers not adapted are refused; use_rslora scales by alpha / sqrt(r)", PeftConfigChecks),
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

    internal static void AnswerScoring(Device device)
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

    internal static void AnswerScoring(Device device, DecoderSpec spec)
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

    internal static void AutomaticCheckpointing(Device device)
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
        // The out-of-memory ladder alone (the estimate off; ActivationEstimateFirstRung tests it).
        var options = new FineTuningOptions { Rank = 4, Alpha = 8, BatchTokens = 2048, Seed = 1, Packing = false, EstimateActivations = false };
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

            // A budget where recomputing the feed-forward activations fits: that
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

    // A run whose first batch is smaller than its largest: the first step's bytes per position say the largest will not
    // fit, so the ladder's first rung (recomputed feed-forward activations) is taken before any step fails. Nine sequences
    // of one length in batches of four rows leave a batch of one; seeds that put it first take the measured rung.
    internal static void FirstRungFromMeasuredBytes(Device device)
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
        var random = new Random(72);
        var sequences = Enumerable.Range(0, 9).Select(_ =>
            new TrainingSequence([.. Enumerable.Range(0, 129).Select(_ => random.Next(256))], [.. Enumerable.Range(0, 129).Select(i => i > 64)])).ToList();
        var options = new FineTuningOptions { Rank = 4, Alpha = 8, BatchTokens = 512, Packing = false, Seed = 1, EstimateActivations = false };
        (bool Done, string Trace) Train(FineTuningOptions run, long? extra)
        {
            using var pretrained = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
            var trace = new System.Text.StringBuilder();
            long before = ComputeResources.GetMemoryUsage(device).InUse;
            ComputeResources.CpuMemoryLimit = extra is { } e ? before + e : null;
            try
            {
                FineTuner.Train(pretrained, sequences, null, run, trace: line => trace.AppendLine(line));
                return (true, trace.ToString());
            }
            catch (ResourceLimitExceededException)
            {
                return (false, trace.ToString());
            }
            finally
            {
                ComputeResources.CpuMemoryLimit = null;
            }
        }

        // The least budget (in steps of 256 KiB) a fixed setting trains under.
        long Least(FineTuningOptions run)
        {
            long low = 0, high = 256L << 20;
            while (high - low > 256 << 10)
            {
                long middle = (low + high) / 2;
                (low, high) = Train(run, middle).Done ? (low, middle) : (middle, high);
            }

            return high;
        }

        try
        {
            long recomputed = Least(options with { RecomputeFeedForward = true, Checkpointing = false });
            long plain = Least(options with { RecomputeFeedForward = false, Checkpointing = false });
            Check(recomputed < plain, $"recomputed feed-forward activations need less ({recomputed >> 10} KiB) than plain training ({plain >> 10} KiB)");
            long budget = (recomputed + plain) / 2;
            int measured = 0;
            foreach (int seed in Enumerable.Range(1, 12))
            {
                var run = Train(options with { Seed = seed }, budget);
                Check(run.Done, $"seed {seed}: training finishes under {budget >> 10} KiB; trace:\n{run.Trace}");
                if (run.Trace.Contains("the largest batch", StringComparison.Ordinal))
                {
                    measured++;
                    Check(!run.Trace.Contains("out of device memory", StringComparison.Ordinal), $"seed {seed}: the rung taken from the measured bytes, no step failed first:\n{run.Trace}");
                }
            }

            Check(measured > 0, "some seed starts with the batch of one row and takes the rung from its measured bytes");
            Check(!Train(options, null).Trace.Contains("the largest batch", StringComparison.Ordinal), "with room, no rung is taken");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
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

    // A model tuned in place, as an app that shares one copy between its users does (Craft's learning plug-in, B7): the
    // adapter it goes on from loads with the base frozen, the round's adapters are removed with their memory, and the batch
    // is sized to the room the device reports.
    internal static void AdaptersInPlace(Device device)
    {
        var spec = new DecoderSpec
        {
            Vocabulary = 260, Dim = 64, Layers = 2, Heads = 2, KvHeads = 1, HeadDim = 32, FfDim = 128, MaxPositions = 256,
            Rope = new RopeSettings(10000f), NormEpsilon = 1e-6f, QkNorm = true, TieEmbeddings = true,
        };
        string folder = WriteChatModel(spec);
        string adapter = Path.Combine(Path.GetTempPath(), "ns-inplace-" + Guid.NewGuid().ToString("N"));
        var random = new Random(5);
        var sequences = Enumerable.Range(0, 4).Select(_ =>
            new TrainingSequence([.. Enumerable.Range(0, 33).Select(_ => random.Next(256))], [.. Enumerable.Range(0, 33).Select(i => i > 16)])).ToList();
        var options = new FineTuningOptions { Rank = 4, Alpha = 8, BatchTokens = 256, LearningRate = 1e-2f, Seed = 1 };
        try
        {
            using (var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
            {
                using (var session = new FineTuningSession(model, options, totalSteps: 2))
                {
                    session.Step(sequences);
                }

                model.SaveAdapter(adapter);
            }

            using var reloaded = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
            Check(reloaded.Network.TrainableParameters().Any(), "a loaded model's weights come trainable");
            int loaded = reloaded.LoadAdapter(adapter);
            var adapters = reloaded.Network.Descendants().OfType<Idrak.Abstraction.Modules.IAdaptableLinear>().Where(l => l.Adapter is not null)
                .SelectMany(l => l.Adapter!.Parameters).ToHashSet();
            var trainable = reloaded.Network.TrainableParameters().ToList();
            Check(loaded > 0 && trainable.Count == adapters.Count && trainable.All(adapters.Contains),
                $"after LoadAdapter only the adapters train ({trainable.Count} trainable tensors, {adapters.Count} adapter tensors; before, every weight trained)");
            long held = ComputeResources.GetMemoryUsage(device).InUse;
            using (var session = new FineTuningSession(reloaded, options, totalSteps: 1))
            {
                session.Step(sequences);
            }

            Check(reloaded.Network.TrainableParameters().All(adapters.Contains), "a session over the loaded adapter trains the adapter alone");
            int removed = reloaded.Network.RemoveAdapters();
            long after = ComputeResources.GetMemoryUsage(device).InUse;
            Check(removed == loaded && !reloaded.Network.Descendants().OfType<Idrak.Abstraction.Modules.IAdaptableLinear>().Any(l => l.Adapter is not null),
                $"every adapter removed ({removed} of {loaded})");
            Check(after < held, $"the adapters' tensors and gradients are freed ({held:N0} → {after:N0} bytes in use)");
            using (var fresh = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
            {
                AssertClose([FineTuner.Evaluate(fresh, sequences)], [FineTuner.Evaluate(reloaded, sequences)], 1e-5f, "without its adapters the model scores as its base model");
            }

            if (device.Type != DeviceType.Cpu)
            {
                return;                                                          // the CPU memory limit makes a small device
            }

            // The batch: halved while the estimate is above the room (a CPU limit just above what is in use), one sequence at least.
            var run = options with { BatchTokens = 4096 };
            long Bytes(int tokens) => ActivationEstimate.Bytes(reloaded.Spec, device.Backend.Capabilities, tokens / 64, 64, ActivationRung.Of(run), run.LossChunkRows);
            long inUse = ComputeResources.GetMemoryUsage(device).InUse;
            try
            {
                ComputeResources.CpuMemoryLimit = inUse + Bytes(1024) + 1;
                int fits = FineTuner.BatchTokensThatFit(reloaded, run, 64);
                Check(fits == 1024, $"a room of the 1,024-token estimate fits 1,024 tokens ({fits}; estimates {Bytes(4096):N0}, {Bytes(2048):N0}, {Bytes(1024):N0} bytes)");
                ComputeResources.CpuMemoryLimit = inUse + 1;
                Check(FineTuner.BatchTokensThatFit(reloaded, run, 64) == 64, "no room: one sequence, never less (the out-of-memory ladder stays behind it)");
                ComputeResources.CpuMemoryLimit = inUse + Bytes(4096) * 2;
                Check(FineTuner.BatchTokensThatFit(reloaded, run, 64) == 4096, "with room the batch is the options' own");
            }
            finally
            {
                ComputeResources.CpuMemoryLimit = null;
            }
        }
        finally
        {
            if (Directory.Exists(adapter))
            {
                Directory.Delete(adapter, recursive: true);
            }
        }
    }

    // The automatic-checkpointing model and data: a CPU budget only checkpointed training fits (found with the estimate
    // off), under which the estimate starts the first step at checkpointing; the ladder alone steps down to it; a session
    // checkpointed there resumes at checkpointing.
    internal static void ActivationEstimateFirstRung(Device device)
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
        string saved = Path.Combine(Path.GetTempPath(), "ns-rung-" + Guid.NewGuid().ToString("N"));
        var random = new Random(73);
        var sequences = Enumerable.Range(0, 16).Select(_ =>
            new TrainingSequence([.. Enumerable.Range(0, 257).Select(_ => random.Next(256))], [.. Enumerable.Range(0, 257).Select(i => i > 128)])).ToList();
        var options = new FineTuningOptions { Rank = 4, Alpha = 8, BatchTokens = 2048, Seed = 1, Packing = false };
        (bool Done, string Trace) Train(FineTuningOptions run, long? extra)
        {
            using var pretrained = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
            var trace = new System.Text.StringBuilder();
            long before = ComputeResources.GetMemoryUsage(device).InUse;
            ComputeResources.CpuMemoryLimit = extra is { } e ? before + e : null;
            try
            {
                FineTuner.Train(pretrained, sequences, null, run, trace: line => trace.AppendLine(line));
                return (true, trace.ToString());
            }
            catch (ResourceLimitExceededException)
            {
                return (false, trace.ToString());
            }
            finally
            {
                ComputeResources.CpuMemoryLimit = null;
            }
        }

        try
        {
            var ladder = options with { EstimateActivations = false };
            var lightest = ladder with { RecomputeFeedForward = true, Checkpointing = false };
            long? budget = null;
            foreach (long megabytes in new long[] { 4, 8, 12, 16, 24, 32, 48, 64, 96, 128 })
            {
                if (Train(ladder with { Checkpointing = true }, megabytes << 20).Done)
                {
                    budget = Train(lightest, megabytes << 20).Done ? null : megabytes << 20;
                    break;
                }
            }

            Check(budget is not null, "a budget where only checkpointed training fits");
            var estimated = Train(options, budget);
            Check(estimated.Done && estimated.Trace.Contains("activations: checkpointing from the start", StringComparison.Ordinal)
                  && !estimated.Trace.Contains("out of device memory", StringComparison.Ordinal),
                $"the estimate starts at checkpointing under {budget >> 20} MiB, no step fails first:\n{estimated.Trace}");
            var stepped = Train(ladder, budget);
            Check(stepped.Done && stepped.Trace.Contains("out of device memory", StringComparison.Ordinal) && !stepped.Trace.Contains("activations:", StringComparison.Ordinal),
                $"with the estimate off the ladder steps down:\n{stepped.Trace}");
            var roomy = Train(options, null);
            Check(roomy.Done && !roomy.Trace.Contains("from the start", StringComparison.Ordinal) && !roomy.Trace.Contains("checkpointing is on", StringComparison.Ordinal),
                $"with room the steps start with the activations held:\n{roomy.Trace}");

            // A session that settled on checkpointing writes it with its checkpoint; one resumed from it starts there.
            using (var pretrained = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
            {
                long before = ComputeResources.GetMemoryUsage(device).InUse;
                ComputeResources.CpuMemoryLimit = before + budget!.Value;
                try
                {
                    using var session = new FineTuningSession(pretrained, options, totalSteps: 4, sequences);
                    session.Step(sequences.Take(7).ToList());
                    Check(session.Settings.StartsWith("checkpointing", StringComparison.Ordinal), $"the session settled on checkpointing: {session.Settings}");
                    session.Checkpoint(saved);
                }
                finally
                {
                    ComputeResources.CpuMemoryLimit = null;
                }
            }

            Check(FineTuner.ReadStopRung(saved) is { Checkpointing: true }, $"the stop state holds the rung: {File.ReadAllText(Path.Combine(saved, FineTuner.StopStateFile))}");
            using (var pretrained = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
            {
                var trace = new System.Text.StringBuilder();
                using var resumed = new FineTuningSession(pretrained, options with { ResumeFrom = saved }, totalSteps: 4, sequences, trace: line => trace.AppendLine(line));
                Check(resumed.Settings.StartsWith("checkpointing", StringComparison.Ordinal) && trace.ToString().Contains("the rung the checkpoint", StringComparison.Ordinal),
                    $"the resumed session starts at checkpointing ({resumed.Settings}):\n{trace}");
                resumed.Step(sequences.Take(7).ToList());
            }
        }
        finally
        {
            Directory.Delete(folder, true);
            if (Directory.Exists(saved))
            {
                Directory.Delete(saved, true);
            }
        }
    }

    // A tiny Qwen3-style model folder: byte-level tokenizer with ChatML tokens, Qwen3's chat template, random weights.
    internal static void ModelDownload(Device device)
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

            WriteCheckpoint(source, spec, weights, "Qwen3ForCausalLM", sharded: true);
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

            var downloader = new Idrak.Data.Downloader(new HttpClient(web), cache) { Attempts = 1 };
            var pulled = new Recorded<ModelPullProgress>();
            string folder = Idrak.Data.HuggingFaceModels.DownloadFolderAsync("org/tiny", "", "main", "hf_model", downloader, pulled).GetAwaiter().GetResult();
            long bytes = served.Sum(name => new FileInfo(Path.Combine(source, name!)).Length);
            Check(pulled.Values.Count >= served.Count + 1 && pulled.Values[0] is { Received: 0, FilesDone: 0 } && pulled.Values[^1].FilesDone == served.Count && pulled.Values[^1].Files == served.Count,
                $"the pull's progress: nothing first, each file done ({pulled.Values.Count} reports, last {pulled.Values[^1]})");
            Check(pulled.Values.Zip(pulled.Values.Skip(1)).All(pair => pair.Second.Received >= pair.First.Received) && pulled.Values.All(v => v.Model == "org/tiny"),
                "the bytes received never go back");
            Check(pulled.Values[^1].Fraction == 1 || pulled.Values[^1].Received == pulled.Values[^1].Total, $"the last report is the whole: {pulled.Values[^1]} (the listing's sizes are 10 bytes a file, {bytes} served)");
            Check(folder == Path.Combine(cache, "huggingface", "models", "org", "tiny", commit[..12]), $"model folder {folder}");
            Check(Directory.GetFiles(folder).Select(Path.GetFileName).Order().SequenceEqual(served.Order()), "only the files the library reads");

            var loading = new Recorded<PretrainedLoadProgress>();
            using (var original = PretrainedModel.Load(source, new PretrainedOptions { Device = device }))
            using (var downloaded = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, Progress = loading }))
            {
                var told = loading.Values;
                Check(told.Count >= 3 && told[0] is { Tensors: 0, Values: 0 } && told[0].TotalTensors > 0 && told[0].TotalValues > 0, $"the load says nothing is read first ({told[0]})");
                Check(told.Zip(told.Skip(1)).All(pair => pair.Second.Values >= pair.First.Values && pair.Second.Fraction >= pair.First.Fraction), "the share read only grows");
                Check(told[^1].Fraction == 1 && told[^1].Tensors == told[^1].TotalTensors && told.Count > 2, $"the end: every tensor ({told[^1]}, {told.Count} reports)");
                int[] ids = [.. downloaded.Tokenizer!.Encode("hello there, general kenobi")];
                TrainingSequence[] probe = [new TrainingSequence(ids, [.. ids.Select(_ => true)])];
                AssertClose([FineTuner.Evaluate(original, probe)], [FineTuner.Evaluate(downloaded, probe)], 0f, "same loss as the original folder");
            }

            // Without a network, the downloaded copy is used.
            var offline = new Idrak.Data.Downloader(new HttpClient(new Unreachable()), cache) { Attempts = 1 };
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

    internal static void OfflineSkipsUnfinished(Device device)
    {
        if (!FirstRun(nameof(OfflineSkipsUnfinished)))
        {
            return;                                                             // folders and files only: the same on every device
        }

        string cache = Path.Combine(Path.GetTempPath(), "ns-unfinished-" + Guid.NewGuid().ToString("N"));
        try
        {
            string models = Path.Combine(cache, "huggingface", "models", "org");
            string Revision(string model, string revision, int minutesAgo, params string[] files)
            {
                string folder = Path.Combine(models, model, revision);
                Directory.CreateDirectory(folder);
                foreach (string file in files.Prepend("config.json"))
                {
                    File.WriteAllText(Path.Combine(folder, file), "{}");
                }

                Directory.SetLastWriteTimeUtc(folder, DateTime.UtcNow.AddMinutes(-minutesAgo));
                return folder;
            }

            // The oldest is finished; a newer one holds the pull marker, the newest a partial weights file.
            string finished = Revision("half", "aaaaaaaaaaaa", 30, "model.safetensors");
            string marked = Revision("half", "bbbbbbbbbbbb", 20, "model.safetensors", Idrak.Data.HuggingFaceModels.PullMarker);
            string partial = Revision("half", "cccccccccccc", 10, "model.safetensors.part");
            var offline = new Idrak.Data.Downloader(new HttpClient(new Unreachable()), cache) { Attempts = 1 };
            Check(Idrak.Data.HuggingFaceModels.IsFinished(finished) && !Idrak.Data.HuggingFaceModels.IsFinished(marked) && !Idrak.Data.HuggingFaceModels.IsFinished(partial),
                "finished: no marker and no partial file");
            Check(ModelSource.Resolve("org/half", downloader: offline) == finished, $"offline: {ModelSource.Resolve("org/half", downloader: offline)}, not the finished {finished}");
            Check(ModelSource.Resolve("org/half", downloader: offline, download: false) == finished, "without downloading: the finished one");
            File.Delete(Path.Combine(marked, Idrak.Data.HuggingFaceModels.PullMarker));
            Check(ModelSource.Resolve("org/half", downloader: offline) == marked, "once its pull ended, the newer folder is taken (the partial one still not)");

            // The library's own download marks its folder from before its first file: one that fails after it stays unfinished.
            const string hf = "https://huggingface.co", commit = "0123456789abcdef0123456789abcdef01234567";
            var web = new FakeRouter();
            web.Json($"{hf}/api/models/org/fail/revision/main", $"{{\"sha\":\"{commit}\"}}");
            web.Json($"{hf}/api/models/org/fail/tree/{commit}?recursive=true", new JsonArray(
                (JsonNode)new JsonObject { ["type"] = "file", ["path"] = "config.json", ["size"] = 2 },
                new JsonObject { ["type"] = "file", ["path"] = "model.safetensors", ["size"] = 10 }).ToJsonString());
            web.Bytes($"{hf}/org/fail/resolve/{commit}/config.json", "{}"u8.ToArray());     // the weights are not served: 404
            var online = new Idrak.Data.Downloader(new HttpClient(web), cache) { Attempts = 1 };
            string failed = Path.Combine(models, "fail", commit[..12]);
            Throws<HttpRequestException>(() => Idrak.Data.HuggingFaceModels.DownloadAsync("org/fail", downloader: online).GetAwaiter().GetResult(), "a missing weights file");
            Check(File.Exists(Path.Combine(failed, "config.json")) && File.Exists(Path.Combine(failed, Idrak.Data.HuggingFaceModels.PullMarker)),
                "the failed download keeps its config and its marker");
            Throws<HttpRequestException>(() => ModelSource.Resolve("org/fail", downloader: offline), "offline, the unfinished download");
        }
        finally
        {
            if (Directory.Exists(cache))
            {
                Directory.Delete(cache, true);
            }
        }
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

    internal static void ManifestRoundTrip(Device device)
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
            // The working directory comes back with links resolved (macOS: /var is /private/var), so the full path
            // written is the folder as the working directory names it.
            Directory.SetCurrentDirectory(folder);
            string resolved = Directory.GetCurrentDirectory();
            Directory.SetCurrentDirectory(Path.GetDirectoryName(folder)!);
            new TuningManifest { BaseModel = Path.GetFileName(folder), System = "Be brief.", MaxLength = 100 }.Save(adapter);
            Directory.SetCurrentDirectory(previous);
            var read = TuningManifest.Read(adapter)!;
            Check((read.BaseModel == folder || read.BaseModel == resolved) && read.System == "Be brief." && read.MaxLength == 100 && TuningManifest.Exists(adapter),
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

    internal static void LongMessageKeepsAnswer(Device device)
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

            var encoder = new ChatTranscriptEncoder(model.JinjaTemplate!, tokenizer);
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

            // An answer longer than the budget on its own: no shortening can fit, so none is tried; the sequence is the
            // end cut, exactly as without shortening.
            string answer = string.Concat(Enumerable.Repeat("the answer goes on and on ", 30));
            var longAnswer = new ChatTranscript([new ChatMessage("user", "Read the whole report aloud."), new ChatMessage("assistant", answer)], []);
            var givenUp = encoder.Encode(longAnswer, 60);
            var endCut = new ChatTranscriptEncoder(model.JinjaTemplate!, tokenizer) { ShortenToFit = false }.Encode(longAnswer, 60);
            Check(encoder.Fit(longAnswer, 60) is null && givenUp is not null && endCut is not null && givenUp.Tokens.SequenceEqual(endCut.Tokens)
                  && givenUp.Trained.SequenceEqual(endCut.Trained), $"an answer over the budget: {givenUp?.Tokens.Length} tokens, the end cut {endCut?.Tokens.Length}");

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

            var cutting = new ChatTranscriptEncoder(model.JinjaTemplate!, tokenizer) { ShortenToFit = false };
            Check(cutting.Encode(transcript, 120) is null, "without shortening, cutting the end leaves no answer to train");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    internal static void AgentFineTuning(Device device)
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
            var encoder = new ChatTranscriptEncoder(model.JinjaTemplate!, model.Tokenizer!);
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
            var options = new FineTuningOptions { Rank = 4, Alpha = 8, LearningRate = 1e-2f, Epochs = 40, BatchTokens = 2048, WarmupFraction = 0f, Seed = 5 };
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
                reloaded.SaveHuggingFace(exported);
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

            using (var merged = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, MergeAdapter = adapter }))
            {
                float loss = FineTuner.Evaluate(merged, sequences);
                Check(loss < before * 0.5f, $"the adapter merged while loading: {loss:F3}");
            }

            Check(progress.Count > 0 && progress[^1].Step == progress[^1].TotalSteps && progress.All(p => float.IsFinite(p.Loss)), "progress reports");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
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

    internal static float RelativeError(float[] expected, float[] actual)
    {
        double difference = 0, norm = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            difference += (double)(expected[i] - actual[i]) * (expected[i] - actual[i]);
            norm += (double)expected[i] * expected[i];
        }

        return (float)Math.Sqrt(difference / Math.Max(norm, 1e-30));
    }

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
