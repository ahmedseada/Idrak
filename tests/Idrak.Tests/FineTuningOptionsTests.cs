// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Datasets;
using Idrak.Layers;
using Idrak.LanguageModels;
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
            Expect<NotSupportedException>("use_dora", c => c["use_dora"] = true, "use_dora");

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
}
