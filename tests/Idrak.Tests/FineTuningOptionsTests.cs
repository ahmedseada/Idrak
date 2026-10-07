// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Datasets;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Nlp;
using Idrak.Optimizers;

internal static partial class Tests
{
    private static DecoderSpec TinyChatSpec() => new()
    {
        Vocabulary = 260, Dim = 32, Layers = 2, Heads = 2, KvHeads = 1, HeadDim = 16, FfDim = 64, MaxPositions = 256,
        Rope = new RopeSettings(10000f), NormEpsilon = 1e-6f, QkNorm = true, TieEmbeddings = true,
    };

    private static List<TrainingSequence> RandomSequences(int count, int seed)
    {
        var random = new Random(seed);
        return [.. Enumerable.Range(0, count).Select(_ =>
        {
            int n = random.Next(12, 60);
            return new TrainingSequence([.. Enumerable.Range(0, n).Select(_ => random.Next(256))], [.. Enumerable.Range(0, n).Select(i => i >= n / 2)]);
        })];
    }

    private static (List<float> Losses, float[] Adapters, string Trace) TrainTiny(string folder, Device device, IReadOnlyList<TrainingSequence> sequences,
        FineTuningOptions options)
    {
        using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
        var losses = new List<float>();
        var trace = new System.Text.StringBuilder();
        FineTuner.Train(model, sequences, null, options, progress: new SynchronousProgress<FineTuningProgress>(p => losses.Add(p.Loss)),
            trace: line => trace.AppendLine(line));
        return (losses, [.. model.Network.TrainableParameters().SelectMany(p => p.ToArray())], trace.ToString());
    }

    private static void OptimizerAndSchedule(Device device)
    {
        string folder = WriteChatModel(TinyChatSpec());
        try
        {
            var sequences = RandomSequences(24, 71);
            var options = new FineTuningOptions { Rank = 4, Alpha = 8, LearningRate = 3e-3f, WeightDecay = 0.01f, BatchTokens = 160, Epochs = 2, Seed = 3 };

            // The default is AdamW with a cosine schedule: the same run when both are given explicitly.
            var builtIn = TrainTiny(folder, device, sequences, options);
            var explicitRun = TrainTiny(folder, device, sequences, options with
            {
                Optimizer = ps => new AdamW(ps, options.LearningRate, weightDecay: options.WeightDecay),
                Scheduler = FineTuningSchedules.Cosine(options.WarmupFraction, options.MinLearningRate),
            });
            Check(builtIn.Losses.Count >= 6 && builtIn.Losses.Count == explicitRun.Losses.Count, $"steps {builtIn.Losses.Count} and {explicitRun.Losses.Count}");
            CloseByNorm([.. builtIn.Losses], [.. explicitRun.Losses], 1e-5f, "explicit AdamW and cosine: losses");
            CloseByNorm(builtIn.Adapters, explicitRun.Adapters, 1e-5f, "explicit AdamW and cosine: adapters");
            bool recorded = builtIn.Trace.Contains("recorded one training step as a graph");
            Check(recorded == explicitRun.Trace.Contains("recorded one training step as a graph"), "an explicit AdamW is recorded as the default is");

            // An optimizer of one's own: recorded steps (where the device records them) give what ordinary steps give.
            var sgd = options with { Optimizer = ps => new Sgd(ps, 0.05f, momentum: 0.9f), Scheduler = FineTuningSchedules.WarmupStableDecay(0.1f, 0.3f) };
            var sgdGraphs = TrainTiny(folder, device, sequences, sgd);
            var sgdOrdinary = TrainTiny(folder, device, sequences, sgd with { CudaGraphs = false });
            Check(recorded == sgdGraphs.Trace.Contains("recorded one training step as a graph") && (!recorded || sgdGraphs.Trace.Contains("replayed")),
                $"SGD steps are recorded where AdamW's are ({recorded}):\n{sgdGraphs.Trace}");
            Check(!sgdOrdinary.Trace.Contains("recorded"), "no graph when graphs are off");
            CloseByNorm([.. sgdOrdinary.Losses], [.. sgdGraphs.Losses], 2e-4f, "SGD recorded and ordinary: losses");
            CloseByNorm(sgdOrdinary.Adapters, sgdGraphs.Adapters, 2e-4f, "SGD recorded and ordinary: adapters");
            Check(sgdOrdinary.Losses[^1] < sgdOrdinary.Losses[0], $"SGD trains: {sgdOrdinary.Losses[0]:F4} → {sgdOrdinary.Losses[^1]:F4}");

            // An optimizer that cannot be recorded (the CPU update releases the gradients): ordinary steps, same results.
            var host = TrainTiny(folder, device, sequences, options with
            {
                Optimizer = ps => new HostOptimizer(ps, cpu => new AdamW(cpu, options.LearningRate, weightDecay: options.WeightDecay)),
            });
            Check(!host.Trace.Contains("recorded one training step"), $"the CPU update is not recorded:\n{host.Trace}");
            CloseByNorm([.. builtIn.Losses], [.. host.Losses], 2e-3f, "AdamW on the CPU through FineTuningOptions.Optimizer: losses");
            CloseByNorm(builtIn.Adapters, host.Adapters, 2e-3f, "AdamW on the CPU through FineTuningOptions.Optimizer: adapters");

            if (device.Type != DeviceType.Cpu)
            {
                return;
            }

            // The built-in schedules' rates, step by step (10 steps, 2 of warm-up).
            float[] Rates(Func<Optimizer, int, LearningRateScheduler> schedule)
            {
                using var p = Tensor.Persistent([0f], [1], Device.Cpu, requiresGrad: true);
                using var optimizer = new Sgd([p], 1f);
                var s = schedule(optimizer, 10);
                var rates = new float[10];
                for (int i = 0; i < 10; i++)
                {
                    rates[i] = optimizer.LearningRate;
                    s.Step();
                }

                return rates;
            }

            AssertClose([1 / 3f, 2 / 3f, 1f, 0.875f, 0.75f, 0.625f, 0.5f, 0.375f, 0.25f, 0.125f], Rates(FineTuningSchedules.Linear(0.2f)), 1e-6f, "linear schedule");
            AssertClose([1 / 3f, 2 / 3f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f], Rates(FineTuningSchedules.Constant(0.2f)), 1e-6f, "constant schedule");
            AssertClose([1 / 3f, 2 / 3f, 1f, 1f, 1f, 1f, 1f, 2 / 3f, 1 / 3f, 0f], Rates(FineTuningSchedules.WarmupStableDecay(0.2f, 0.3f)), 1e-6f, "warm-up, stable, decay");
            Check(FineTuningOptimizers.Names.All(n => FineTuningOptimizers.Create(n, 1e-3f) is not null), "every named optimizer");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    private static void LossHook(Device device)
    {
        string folder = WriteChatModel(TinyChatSpec());
        try
        {
            var sequences = RandomSequences(20, 72);
            var options = new FineTuningOptions { Rank = 4, Alpha = 8, LearningRate = 3e-3f, BatchTokens = 200, Epochs = 2, Seed = 4, CudaGraphs = false };
            var fused = TrainTiny(folder, device, sequences, options);
            var hooked = TrainTiny(folder, device, sequences, options with { Loss = FineTuningLosses.TokenCrossEntropy });
            CloseByNorm([.. fused.Losses], [.. hooked.Losses], 1e-4f, "the token cross-entropy as a loss delegate: losses");
            CloseByNorm(fused.Adapters, hooked.Adapters, 1e-3f, "the token cross-entropy as a loss delegate: adapters");

            // Gradient accumulation: the step totals cover both batches, so the step's loss is the same mean.
            var accumulated = TrainTiny(folder, device, sequences, options with { GradientAccumulation = 2 });
            var accumulatedHook = TrainTiny(folder, device, sequences, options with { GradientAccumulation = 2, Loss = FineTuningLosses.TokenCrossEntropy });
            CloseByNorm([.. accumulated.Losses], [.. accumulatedHook.Losses], 1e-4f, "accumulated steps through the loss delegate");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    private static void PreferenceFormulas(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                              // the losses run on the CPU on every device
        }

        int[] counts = [3, 2, 4, 1, 2, 2];
        var random = new Random(9);
        float[] policy = [.. Enumerable.Range(0, counts.Sum()).Select(_ => -0.2f - 2f * random.NextSingle())];
        float[] reference = [.. policy.Select(v => v + 0.6f * (random.NextSingle() - 0.5f))];
        double[] Sums(float[] v)
        {
            var sums = new double[counts.Length];
            for (int s = 0, t = 0; s < counts.Length; s++)
            {
                for (int i = 0; i < counts[s]; i++, t++)
                {
                    sums[s] += v[t];
                }
            }

            return sums;
        }

        static double LogSigmoid(double x) => -Math.Log(1 + Math.Exp(-x));
        var sp = Sums(policy);
        var sr = Sums(reference);
        double[] means = [.. sp.Select((v, s) => v / counts[s])];
        int pairs = counts.Length / 2;
        double dpo = 0, dpoSmooth = 0, orpo = 0, simpo = 0;
        for (int p = 0; p < pairs; p++)
        {
            double h = (sp[2 * p] - sr[2 * p]) - (sp[2 * p + 1] - sr[2 * p + 1]);
            dpo += -LogSigmoid(0.1 * h);
            dpoSmooth += -0.9 * LogSigmoid(0.1 * h) - 0.1 * LogSigmoid(-0.1 * h);
            double Odds(double a) => a - Math.Log(1 - Math.Exp(a));
            orpo += -means[2 * p] - 0.25 * LogSigmoid(Odds(means[2 * p]) - Odds(means[2 * p + 1]));
            simpo += -LogSigmoid(2 * (means[2 * p] - means[2 * p + 1]) - 0.5);
        }

        float Run(FineTuningLoss loss, out float[] gradient)
        {
            using var scope = new TensorScope();
            var leaf = Tensor.From(policy, [policy.Length], Device.Cpu, requiresGrad: true);
            var input = new FineTuningLossInput(leaf, counts, Tensor.From(reference, [reference.Length], Device.Cpu), pairs: true);
            var value = loss(input);
            value.Backward();
            gradient = leaf.Grad!.ToArray();
            return value.Item();
        }

        AssertClose([(float)(dpo / pairs)], [Run(FineTuningLosses.Dpo(0.1f), out var dpoGradient)], 1e-5f, "DPO against its formula");
        AssertClose([(float)(dpoSmooth / pairs)], [Run(FineTuningLosses.Dpo(0.1f, 0.1f), out _)], 1e-5f, "conservative DPO against its formula");
        AssertClose([(float)(orpo / pairs)], [Run(FineTuningLosses.Orpo(0.25f), out _)], 1e-5f, "ORPO against its formula");
        AssertClose([(float)(simpo / pairs)], [Run(FineTuningLosses.SimPo(2f, 0.5f), out _)], 1e-5f, "SimPO against its formula");

        // DPO's gradient: −β σ(−βh) / pairs on each chosen token, the opposite on each rejected one.
        var expected = new float[policy.Length];
        for (int s = 0, t = 0; s < counts.Length; s++)
        {
            int p = s / 2;
            double h = (sp[2 * p] - sr[2 * p]) - (sp[2 * p + 1] - sr[2 * p + 1]);
            double g = -0.1 / (1 + Math.Exp(0.1 * h)) / pairs * (s % 2 == 0 ? 1 : -1);
            for (int i = 0; i < counts[s]; i++, t++)
            {
                expected[t] = (float)g;
            }
        }

        AssertClose(expected, dpoGradient, 1e-6f, "DPO gradient");
        using (var scope = new TensorScope())
        {
            var x = Tensor.From([0f, -100f, 100f, 1.5f], [4], Device.Cpu, requiresGrad: true);
            var y = FineTuningLosses.LogSigmoid(x);
            y.Sum().Backward();
            AssertClose([MathF.Log(0.5f), -100f, 0f, (float)LogSigmoid(1.5)], y.ToArray(), 1e-5f, "log-sigmoid values (no overflow)");
            AssertClose([0.5f, 1f, 0f, (float)(1 / (1 + Math.Exp(1.5)))], x.Grad!.ToArray(), 1e-5f, "log-sigmoid gradient (½ at 0)");
        }
        using (var scope = new TensorScope())
        {
            var plain = new FineTuningLossInput(Tensor.From(policy, [policy.Length], Device.Cpu), counts);
            Check(Refused(() => FineTuningLosses.Dpo()(plain)), "DPO without pairs is refused");
        }
    }

    private static bool Refused(Action action)
    {
        try
        {
            action();
            return false;
        }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or InvalidDataException or ArgumentException)
        {
            return true;
        }
    }

    // Pairs of random prompts with two answer styles: the chosen answers count up from the prompt's first token, the
    // rejected ones repeat it; only the answers are trained.
    private static List<PreferencePair> RandomPairs(int count, int seed)
    {
        var random = new Random(seed);
        TrainingSequence Sequence(int[] prompt, int[] answer) =>
            new([.. prompt, .. answer], [.. prompt.Select(_ => false), .. answer.Select(_ => true)]);
        return [.. Enumerable.Range(0, count).Select(_ =>
        {
            int[] prompt = [.. Enumerable.Range(0, random.Next(4, 12)).Select(_ => random.Next(10, 250))];
            int n = random.Next(3, 7);
            return new PreferencePair(Sequence(prompt, [.. Enumerable.Range(1, n).Select(i => (prompt[0] + i) % 256)]),
                Sequence(prompt, [.. Enumerable.Range(0, n + random.Next(-1, 2)).Select(_ => prompt[0])]));
        })];
    }

    private static void PreferenceTraining(Device device)
    {
        string folder = WriteChatModel(TinyChatSpec());
        try
        {
            var pairs = RandomPairs(16, 73);
            var held = RandomPairs(8, 74);
            foreach (var (name, loss, rate) in new (string, FineTuningLoss?, float)[]
                     {
                         ("DPO", null, 5e-3f), ("ORPO", FineTuningLosses.Orpo(), 5e-3f), ("SimPO", FineTuningLosses.SimPo(), 5e-3f),
                     })
            {
                using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
                var options = new FineTuningOptions { Rank = 4, Alpha = 8, LearningRate = rate, BatchTokens = 256, Epochs = 12, WarmupFraction = 0f, Seed = 2, Loss = loss };
                var losses = new List<float>();
                var evaluations = FineTuner.Train(model, pairs, held, options,
                    progress: new SynchronousProgress<FineTuningProgress>(p => losses.Add(p.Loss)));
                float first = losses.Take(3).Average(), last = losses.TakeLast(3).Average();
                Check(evaluations.Count == options.Epochs && evaluations.All(float.IsFinite), $"{name}: an evaluation per epoch, {string.Join(", ", evaluations)}");
                Check(last < first * 0.8f, $"{name}: training loss {first:F4} → {last:F4}");
                if (name == "DPO")
                {
                    // The adapters start as the identity, so the model is its own reference: every pair's loss is log 2.
                    AssertClose([MathF.Log(2f)], [losses[0]], 1e-4f, "DPO's first step");
                    // Pairs the model never saw lean the same way.
                    Check(evaluations[^1] < 0.98f * MathF.Log(2f), $"DPO held-out loss {evaluations[^1]:F4} (log 2 before training)");
                }
            }

            // With the adapters disabled the model is its base model again.
            using var tuned = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
            tuned.AddAdapters(2, 4, ["q", "v", "down"], seed: 1);
            foreach (var adapter in tuned.Network.Descendants().OfType<Linear>().Select(l => l.Adapter).OfType<LoraAdapter>())
            {
                adapter.B.Load([.. Enumerable.Range(0, adapter.B.Size).Select(i => 0.1f * MathF.Cos(i))]);
            }

            using var plain = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
            tuned.Network.Eval();
            plain.Network.Eval();
            using (var scope = new TensorScope())
            using (Autograd.NoGrad())
            {
                var tokens = Tensor.From([5f, 6f, 7f, 8f, 9f], [1, 5], device);
                float[] base0 = plain.Network.Forward(tokens).ToArray(), adapted = tuned.Network.Forward(tokens).ToArray(), disabled;
                using (tuned.Network.DisableAdapters())
                {
                    disabled = tuned.Network.Forward(tokens).ToArray();
                }

                Check(RelativeError(base0, adapted) > 1e-3f, "the adapters change the output");
                CloseByNorm(base0, disabled, 1e-5f, "adapters disabled: the base model's output");
                CloseByNorm(adapted, tuned.Network.Forward(tokens).ToArray(), 1e-6f, "adapters back after the scope");
            }
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    private static void PreferenceRows(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                              // tokenization only
        }

        // The layouts ChatRows reads as preference pairs, all to one shape.
        var expected = JsonNode.Parse("""{"prompt": [{"role": "user", "content": "2+2?"}], "chosen": [{"role": "assistant", "content": "4"}], "rejected": [{"role": "assistant", "content": "5"}]}""");
        string[] layouts =
        [
            """{"prompt": "2+2?", "chosen": "4", "rejected": "5"}""",
            """{"prompt": [{"role": "user", "content": "2+2?"}], "chosen": [{"role": "assistant", "content": "4"}], "rejected": [{"role": "assistant", "content": "5"}]}""",
            """{"chosen": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}], "rejected": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "5"}]}""",
            """{"prompt": "2+2?", "chosen": [{"from": "human", "value": "2+2?"}, {"from": "gpt", "value": "4"}], "rejected": "[{\"role\": \"user\", \"content\": \"2+2?\"}, {\"role\": \"assistant\", \"content\": \"5\"}]"}""",
        ];
        foreach (var layout in layouts)
        {
            var row = ChatRows.Preference(JsonNode.Parse(layout)!.AsObject());
            Check(JsonNode.DeepEquals(row, expected), $"{layout} → {row?.ToJsonString()}");
        }

        Check(ChatRows.Preference(JsonNode.Parse("""{"messages": [{"role": "user", "content": "a"}]}""")!.AsObject()) is null, "a conversation is not a pair");
        Check(ChatRows.Normalize(JsonNode.Parse(layouts[0])!.AsObject(), RowKind.Preference, system: "Be brief.")!["prompt"]![0]!["content"]!.GetValue<string>() == "Be brief.",
            "RowKind.Preference adds the system prompt");

        string folder = WriteChatModel(TinyChatSpec());
        try
        {
            using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
            var encoder = new ChatTranscriptEncoder(model.ChatTemplate!, model.Tokenizer!);
            var multi = JsonNode.Parse("""{"prompt": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi there"}, {"role": "user", "content": "2+2?"}], "chosen": "four", "rejected": "five, I think"}""")!.AsObject();
            var encoded = encoder.EncodePreference(multi, 200);
            Check(encoded is not null, $"the multi-turn pair encodes: {ChatRows.Preference(multi)?.ToJsonString()}");
            var pair = encoded!;
            string Trained(TrainingSequence s) => model.Tokenizer!.Decode(s.Tokens.Where((_, i) => s.Trained[i]));
            Check(Trained(pair.Chosen).EndsWith("four<|im_end|>") && Trained(pair.Rejected).EndsWith("five, I think<|im_end|>")
                  && !Trained(pair.Chosen).Contains("hi there") && !Trained(pair.Rejected).Contains("hi there"),
                $"only the answers train (not the assistant turn in the prompt): '{Trained(pair.Chosen)}', '{Trained(pair.Rejected)}'");
            int Prompt(TrainingSequence s) => Array.IndexOf(s.Trained, true);
            Check(Prompt(pair.Chosen) == Prompt(pair.Rejected) && pair.Chosen.Tokens.Take(Prompt(pair.Chosen)).SequenceEqual(pair.Rejected.Tokens.Take(Prompt(pair.Chosen))),
                "both answers follow the same prompt");
            Check(encoder.EncodePreference(JsonNode.Parse(layouts[0])!.AsObject(), 200) is not null && encoder.EncodePreference(new JsonObject { ["text"] = "x" }, 200) is null,
                "string layouts encode; other rows give nothing");
            var cut = encoder.EncodePreference(new JsonObject { ["prompt"] = new string('x', 400), ["chosen"] = "yes", ["rejected"] = "no" }, 80)!;
            Check(cut.Chosen.Tokens.Length <= 81 && Trained(cut.Chosen).EndsWith("yes<|im_end|>"), "a long prompt is shortened, the answer kept");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    private static void PeftConfigChecks(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        string folder = WriteChatModel(TinyChatSpec()), adapter = Path.Combine(folder, "adapter");
        try
        {
            using (var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
            {
                model.AddAdapters(2, 4, ["q", "v"], seed: 5);
                model.SaveAdapter(adapter);
            }

            string configPath = Path.Combine(adapter, "adapter_config.json"), weightsPath = Path.Combine(adapter, "adapter_model.safetensors");
            string original = File.ReadAllText(configPath);
            byte[] weights = File.ReadAllBytes(weightsPath);
            void Expect<T>(string change, Action<JsonObject> edit, string message) where T : Exception
            {
                var config = JsonNode.Parse(original)!.AsObject();
                edit(config);
                File.WriteAllText(configPath, config.ToJsonString());
                foreach (var (how, load) in new (string, Action)[]
                         {
                             ("LoadAdapter", () => { using var m = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }); m.LoadAdapter(adapter); }),
                             ("MergeAdapter", () => { using var m = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, MergeAdapter = adapter }); }),
                         })
                {
                    try
                    {
                        load();
                        Check(false, $"{change}: {how} loaded");
                    }
                    catch (T ex)
                    {
                        Check(ex.Message.Contains(message), $"{change}: {how} said {ex.Message}");
                    }
                }

                File.WriteAllText(configPath, original);
            }

            Expect<NotSupportedException>("peft_type IA3", c => c["peft_type"] = "IA3", "peft_type \"IA3\"");
            Expect<NotSupportedException>("rank_pattern", c => c["rank_pattern"] = new JsonObject { ["q_proj"] = 8 }, "rank_pattern");
            Expect<NotSupportedException>("trained biases", c => c["bias"] = "all", "bias \"all\"");
            Expect<NotSupportedException>("modules to save", c => c["modules_to_save"] = new JsonArray("lm_head"), "modules_to_save");
            Expect<InvalidDataException>("use_dora without magnitude vectors", c => c["use_dora"] = true, "lora_magnitude_vector");

            // A tensor for a layer the model does not adapt.
            List<(string, int[], float[])> tensors;
            using (var reader = SafeTensorsReader.Open(weightsPath))
            {
                tensors = [.. reader.Tensors.Keys.Select(k => (k, reader.Tensors[k].Shape.ToArray(), reader.Read(k)))];
            }

            tensors.Add(("base_model.model.model.embed_tokens.lora_embedding_A", [2, 260], new float[520]));
            SafeTensorsWriter.Write(weightsPath, tensors, SafeTensorType.F32);

            Expect<InvalidDataException>("an embedding adapter", _ => { }, "lora_embedding_A");
            File.WriteAllBytes(weightsPath, weights);

            // use_rslora: scale alpha / √r.
            var rs = JsonNode.Parse(original)!.AsObject();
            rs["use_rslora"] = true;
            File.WriteAllText(configPath, rs.ToJsonString());
            using (var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
            {
                Check(model.LoadAdapter(adapter) == 2 * 2, "the adapter loads");
                Check(model.Network.Descendants().OfType<Linear>().Select(l => l.Adapter).OfType<LoraAdapter>().All(a => Math.Abs(a.Scale - 4f / MathF.Sqrt(2f)) < 1e-6f),
                    "use_rslora scales by alpha / √r");
            }
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    private static void DoraLayer(Device device)
    {
        const int In = 24, Out = 20, Rank = 3, Rows = 5;
        const float Scale = 0.7f;
        var random = new Random(31);
        float[] Values(int n, float range) => [.. Enumerable.Range(0, n).Select(_ => (random.NextSingle() * 2f - 1f) * range)];
        float[] w = Values(In * Out, 0.3f), bias = Values(Out, 0.1f), a = Values(In * Rank, 0.4f), b = Values(Rank * Out, 0.4f), x = Values(Rows * In, 1f);
        float[] m = [.. Enumerable.Range(0, Out).Select(_ => 0.5f + random.NextSingle())], c = Values(Rows * Out, 1f);

        foreach (bool int4 in new[] { false, true })
        {
            using var scope = new TensorScope();
            var layer = Linear.FromWeights(Tensor.Persistent(w, [In, Out], device, requiresGrad: false), Tensor.Persistent(bias, [Out], device, requiresGrad: false));
            if (int4)
            {
                layer.QuantizeInt4();
            }

            float[] dense = layer.WeightValues();                        // the weights as the layer holds them
            var dora = new DoraAdapter(Tensor.Persistent(a, [In, Rank], device, requiresGrad: true), Tensor.Persistent(b, [Rank, Out], device, requiresGrad: true),
                Tensor.Persistent(m, [Out], device, requiresGrad: true), Rank, Scale);
            layer.Adapter = dora;
            var input = Tensor.From(x, [Rows, In], device);
            var y = layer.Forward(input);
            (y * Tensor.From(c, [Rows, Out], device)).Sum().Backward();

            // The reference on the host, in double: z = x·W + s·x·A·B, n = ‖W + s·A·B‖ per column, y = z·m/n + b;
            // d/dm (with the norm a constant, as peft) = Σ_rows c·z / n.
            var expected = new float[Rows * Out];
            var magnitudeGradient = new float[Out];
            var norms = new double[Out];
            for (int o = 0; o < Out; o++)
            {
                for (int i = 0; i < In; i++)
                {
                    double ab = 0;
                    for (int r = 0; r < Rank; r++)
                    {
                        ab += a[i * Rank + r] * b[r * Out + o];
                    }

                    double v = dense[i * Out + o] + Scale * ab;
                    norms[o] += v * v;
                }

                norms[o] = Math.Sqrt(norms[o]);
            }

            for (int row = 0; row < Rows; row++)
            {
                for (int o = 0; o < Out; o++)
                {
                    double z = 0;
                    for (int i = 0; i < In; i++)
                    {
                        double ab = 0;
                        for (int r = 0; r < Rank; r++)
                        {
                            ab += a[i * Rank + r] * b[r * Out + o];
                        }

                        z += x[row * In + i] * (dense[i * Out + o] + Scale * ab);
                    }

                    expected[row * Out + o] = (float)(z * m[o] / norms[o] + bias[o]);
                    magnitudeGradient[o] += (float)(c[row * Out + o] * z / norms[o]);
                }
            }

            string kind = int4 ? "4-bit base" : "float base";
            CloseByNorm(expected, y.ToArray(), 2e-3f, $"DoRA output, {kind}");
            CloseByNorm(magnitudeGradient, dora.Magnitude.Grad!.ToArray(), 2e-3f, $"DoRA magnitude gradient, {kind}");
            Check(dora.A.Grad!.ToArray().Any(v => v != 0f) && dora.B.Grad!.ToArray().Any(v => v != 0f), $"A and B receive gradients, {kind}");
            if (!int4)
            {
                float[] adapted;
                using (Autograd.NoGrad())
                {
                    adapted = layer.Forward(input).ToArray();
                    layer.MergeAdapter();
                    Check(layer.Adapter is null, "merged: no adapter");
                    CloseByNorm(adapted, layer.Forward(input).ToArray(), 1e-5f, "merged DoRA weight gives the adapted output");
                }
            }
        }
    }

    private static void DoraTraining(Device device)
    {
        string folder = WriteChatModel(TinyChatSpec()), adapter = Path.Combine(folder, "dora"), exported = Path.Combine(folder, "merged");
        try
        {
            var sequences = RandomSequences(16, 75);
            float before, after;
            using (var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
            {
                before = FineTuner.Evaluate(model, sequences);
                model.AddAdapters(4, 8, ["q", "v", "gate", "down"], seed: 3, dora: true);
                AssertClose([before], [FineTuner.Evaluate(model, sequences)], 1e-5f, "DoRA adapters start as the identity");
                var options = new FineTuningOptions { Rank = 4, Alpha = 8, LearningRate = 1e-2f, Epochs = 8, BatchTokens = 512, WarmupFraction = 0f, Dora = true };
                FineTuner.Train(model, sequences, null, options, adapter);
                after = FineTuner.Evaluate(model, sequences);
                Check(after < before * 0.9f, $"DoRA trains: loss {before:F4} → {after:F4}");
                Check(model.Network.Descendants().OfType<Linear>().Count(l => l.Adapter is DoraAdapter) == 4 * 2, "DoRA adapters on the chosen layers");
            }

            var config = JsonNode.Parse(File.ReadAllText(Path.Combine(adapter, "adapter_config.json")))!;
            Check((bool)config["use_dora"]! && (string)config["peft_type"]! == "LORA", $"PEFT config: {config}");
            using (var reader = SafeTensorsReader.Open(Path.Combine(adapter, "adapter_model.safetensors")))
            {
                Check(reader.Tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_magnitude_vector"].Shape.SequenceEqual([32])
                      && reader.Tensors["base_model.model.model.layers.1.mlp.down_proj.lora_magnitude_vector"].Shape.SequenceEqual([32]), "magnitude vectors in the PEFT layout");
            }

            using (var reloaded = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
            {
                Check(reloaded.LoadAdapter(adapter) == 4 * 2, "every DoRA adapter loads");
                AssertClose([after], [FineTuner.Evaluate(reloaded, sequences)], 1e-4f, "loss with the reloaded DoRA adapters");
                reloaded.SaveHuggingFace(exported, SafeTensorType.F32);
            }

            using (var merged = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, MergeAdapter = adapter }))
            {
                AssertClose([after], [FineTuner.Evaluate(merged, sequences)], 1e-3f, "loss with DoRA merged while loading");
            }

            using (var merged = PretrainedModel.Load(exported, new PretrainedOptions { Device = device }))
            {
                AssertClose([after], [FineTuner.Evaluate(merged, sequences)], 1e-3f, "loss of the merged DoRA export");
            }

            // QLoRA with DoRA: a 4-bit base trains too.
            using var quantized = PretrainedModel.Load(folder, new PretrainedOptions { Device = device, Int4 = true });
            float q0 = FineTuner.Evaluate(quantized, sequences);
            FineTuner.Train(quantized, sequences, null, new FineTuningOptions { Rank = 4, Alpha = 8, LearningRate = 1e-2f, Epochs = 6, BatchTokens = 512, Dora = true });
            float q1 = FineTuner.Evaluate(quantized, sequences);
            Check(q1 < q0 * 0.95f, $"DoRA on a 4-bit base: loss {q0:F4} → {q1:F4}");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }
}
