// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Idrak;
using Idrak.Abstraction.Cluster;
using Idrak.Cli;
using Idrak.Cluster;
using Idrak.Datasets;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Text;
using Idrak.Text.Abstractions;
using Idrak.Optimizers;
using Idrak.Training;
using Idrak.Training.Abstractions;

// Plan 23, phase C2: data parallel on the simulator: the gradient synchronisation (Idrak.Network's "all-reduce" through
// Idrak.Abstraction's GradientSyncs), the Cluster hook of the Trainer, the fine-tuner and the pretrainer, the HostOptimizer
// reducing on its host pieces, the split by measured speed, and idrak cluster sim. Every run is an in-process cluster of
// devices carved from the device under test.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ClusterTrainingGroup =
    [
        ("cluster: gradient synchronisations are a registry (all-reduce from Idrak.Network); a step's draws go round the ranks while their shares are equal and in proportion once measured", ClusterDrawShares),
        ("cluster: the all-reduce sync gives every rank the token-weighted gradient and loss, bucket by bucket with a parameter cut across buckets, on f32 and bf16 wires; a rank with no rows adds nothing and gets the gradients it lacked; normalized losses are summed; a step no rank had rows in ends it", ClusterSyncWeighted),
        ("cluster: the starting state: rank 0's values sent to every rank, or every rank's hash checked and a rank that differs named", ClusterStartingState),
        ("cluster: data parallel 2 ranks on sim matches the ddp fixture: each rank's losses, the step 1 gradients within 1e-4 of the largest, three SGD and folded AdamW steps' losses within 1e-5 and weights within 1e-5 (AdamW: the fixture's measured growth of the gradient gap)", ClusterDdpTrainer),
        ("cluster: bf16 wire within the fixture's tolerance: the step 1 gradients, three SGD steps' losses and weights against bf16/ within its measured gaps", ClusterDdpBFloat16),
        ("cluster: 2 ranks equal one process with the global batch (rule 71, Idrak against Idrak): the Trainer, the fine-tuner (accumulation 1 on 2 ranks as 2 in one) and the pretrainer", ClusterEqualsOneProcess),
        ("cluster: 3 ranks with unequal token counts give the token-weighted gradient: the pretrainer's losses and weights as one process's", ClusterUnequalRanks),
        ("cluster: rank 0 alone writes the checkpoint; a 2-rank run resumes as 3 ranks giving the 3-rank run's losses", ClusterResumeOtherWorld),
        ("cluster: mixed kinds (cpu and a fake device of every listed kind) sync weights every K steps: rank 0's weights on every rank at each K-th step, bit for bit", ClusterSyncWeights),
        ("cluster: batches split by measured speed give the one-process losses (a slow rank takes fewer rows)", ClusterSplitBySpeed),
        ("cluster: HostOptimizer reduces on its downloaded pieces: the host gradients reduced, the device ones untouched, the update alike on every rank; the pretrainer with it as without", ClusterHostOptimizer),
        ("cluster: the Spirals sample's Trainer with a cluster: 2 ranks learn the spirals as one process does, the weights and batch-norm statistics the same on every rank", ClusterSpirals),
        ("cluster: cli cluster sim runs pretrain on every rank of a topology file: rank 0 writes the run, each rank's network time is reported; another command and a missing run folder are usage errors", ClusterCliSim),
    ];

    // `world` ranks of one node, every one carved from `device`.
    private static ClusterTopology DpTopology(Device device, int world) => DpTopology([.. Enumerable.Repeat(device.ToString(), world)]);

    // Different devices in the node (cpu beside a GPU) are joined by pcie, measured here; ranks on one device by same-device.
    private static ClusterTopology DpTopology(IReadOnlyList<string> devices) =>
        ClusterTopology.Create([new ClusterNodeSpec("box", null, [.. devices.Select(d => new ClusterDeviceSpec(d, null))])],
            devices.Distinct(StringComparer.OrdinalIgnoreCase).Count() > 1 ? "pcie" : null, null);

    // Equal splits (no speed measured), so a rerun splits alike.
    private static ClusterOptions DpOptions(WireType wire = WireType.Float32) => new() { RebalanceSteps = 0, Wire = wire, Timeout = TimeSpan.FromSeconds(300) };

    // Runs `work` on every rank of an in-process cluster and returns each rank's result, failing with every rank's error.
    private static T[] DpRun<T>(IClusterTopology topology, ClusterOptions options, Func<ClusterRank, T> work)
    {
        var results = new T[topology.WorldSize];
        InProcessCluster.Run(topology, options, rank => results[rank.Rank] = work(rank)).ThrowIfFailed();
        return results;
    }

    private static float Gap(float[] a, float[] b) => a.Zip(b, (x, y) => MathF.Abs(x - y)).DefaultIfEmpty(0f).Max();

    private static void ClusterDrawShares(Device device)
    {
        if (!FirstRun(nameof(ClusterDrawShares)))
        {
            return;
        }

        Check(GradientSyncs.Find("all-reduce") is not null && GradientSyncs.Origin("all-reduce") == Overrides.Library && GradientSyncs.Describe().Contains("all-reduce:", StringComparison.Ordinal),
            $"Idrak.Network registers all-reduce: {GradientSyncs.Describe()}");
        ThrowsNaming<NotSupportedException>(() => GradientSyncs.Get("no-such-sync"), "an unknown synchronisation", "is not registered", "GradientSyncs.Register");
        long[] sizes = [5, 9, 5, 2, 5, 5, 7];
        Check(DrawShares.Owners(sizes, DrawShares.Even(3)).SequenceEqual([0, 1, 2, 0, 1, 2, 0]), "equal shares: rank r takes draws k·W + r whatever their sizes");
        var owners = DrawShares.Owners([.. Enumerable.Repeat(10L, 100)], [0.75, 0.25]);
        Check(owners.Count(o => o == 0) == 75, $"shares of 3:1 give rank 0 three draws in four: {owners.Count(o => o == 0)}");
        Check(DrawShares.Owners(sizes, [0.6, 0.4]).SequenceEqual(DrawShares.Owners(sizes, [0.6, 0.4])), "the same shares give the same owners");
    }

    private static void ClusterSyncWeighted(Device device)
    {
        foreach (var wire in new[] { WireType.Float32, WireType.BFloat16 })
        {
            // Values every step of the sum holds exactly (bfloat16 too), so the checks are exact.
            int size = wire == WireType.Float32 ? 50 : 16;
            var options = DpOptions(wire) with { BucketBytes = wire == WireType.Float32 ? 64 : 40 };   // parameter 0 cut across buckets
            double[] tokens = [1, 3, 0];
            var results = DpRun(DpTopology(device, 3), options, rank =>
            {
                int r = rank.Rank;
                using var p0 = Tensor.Persistent(new float[size], [size], rank.Device, requiresGrad: true);
                using var p1 = Tensor.Persistent(new float[7], [7], rank.Device, requiresGrad: true);
                using var optimizer = new Sgd([p0, p1], 0f);
                using var sync = GradientSyncs.Create(rank.Collectives, optimizer, options);
                void Gradients()
                {
                    p0.Backend.Upload([.. Enumerable.Range(0, size).Select(j => (float)((r + 1) * j))], p0.GradStorage());
                    if (r < 2)
                    {
                        p1.Backend.Upload([.. Enumerable.Range(0, 7).Select(j => (float)((10 * (r + 1)) + j))], p1.GradStorage());   // rank 2 has none
                    }
                }

                Gradients();
                var weighted = sync.Reduce(tokens[r], r == 2 ? 999 : 2 + (4 * r));     // rank 2 had no rows: its loss counts for nothing
                float[] g0 = p0.Grad!.ToArray(), g1 = p1.Grad?.ToArray() ?? [];
                Gradients();
                var summed = sync.Reduce(tokens[r], r == 2 ? 0 : 1.5 + r, normalized: true);
                var s0 = p0.Grad!.ToArray();
                var none = sync.Reduce(0, 0);
                return (Weighted: weighted, G0: g0, G1: g1, Summed: summed, S0: s0, None: none, After: p0.Grad!.ToArray());
            });

            string w = wire.ToString();
            float[] expected0 = [.. Enumerable.Range(0, size).Select(j => 1.75f * j)], expected1 = [.. Enumerable.Range(0, 7).Select(j => 17.5f + j)];
            foreach (var (result, r) in results.Select((x, r) => (x, r)))
            {
                Check(result.Weighted is { Tokens: 4, Loss: 5 }, $"{w}: rank {r}: the step's tokens and token-weighted loss: {result.Weighted}");
                Check(result.G0.SequenceEqual(expected0), $"{w}: rank {r}: (1·g0 + 3·g1) / 4 cut across buckets: [{string.Join(", ", result.G0.Take(8))}…]");
                Check(result.G1.SequenceEqual(expected1), $"{w}: rank {r}: a parameter rank 2 had no gradient for gets one: [{string.Join(", ", result.G1)}]");
                Check(result.Summed is { Tokens: 4, Loss: 4 } && result.S0.SequenceEqual(Enumerable.Range(0, size).Select(j => 3f * j)),
                    $"{w}: rank {r}: normalized losses and gradients summed as they are: {result.Summed}");
                Check(result.None.Tokens == 0 && result.After.SequenceEqual(result.S0), $"{w}: rank {r}: a step no rank had rows in ends it, nothing moved");
            }
        }
    }

    private static void ClusterStartingState(Device device)
    {
        var topology = DpTopology(device, 2);
        float[][] Start(ClusterOptions options, bool differ) => DpRun(topology, options, rank =>
        {
            using var p = Tensor.Persistent([.. Enumerable.Repeat(differ ? rank.Rank + 1f : 1f, 6)], [2, 3], rank.Device, requiresGrad: true);
            using var optimizer = new Sgd([p], 0.1f);
            using var sync = GradientSyncs.Create(rank.Collectives, optimizer, options);
            sync.Start();
            return p.ToArray();
        });

        Check(Start(DpOptions(), differ: true).All(v => v.All(x => x == 1f)), "broadcast: every rank starts from rank 0's values");
        var check = DpOptions() with { StartingState = StartingState.Check };
        Check(Start(check, differ: false).All(v => v.All(x => x == 1f)), "check: equal values pass");
        var failed = InProcessCluster.Run(topology, check, rank =>
        {
            using var p = Tensor.Persistent([.. Enumerable.Repeat(rank.Rank + 1f, 6)], [2, 3], rank.Device, requiresGrad: true);
            using var optimizer = new Sgd([p], 0.1f);
            using var sync = GradientSyncs.Create(rank.Collectives, optimizer, check);
            sync.Start();
        });
        Check(!failed.Succeeded && failed.Errors.All(e => e is InvalidOperationException && e.Message.Contains("Rank 1", StringComparison.Ordinal)
                                                         && e.Message.Contains("parameter 0 of 1", StringComparison.Ordinal) && e.Message.Contains("IDRAK_CLUSTER_START", StringComparison.Ordinal)),
            $"check: different values name the rank, the parameter and the setting: {string.Join(" | ", failed.Errors.Select(e => e?.Message))}");
    }

    // The fixture's eight windows (33 ids each, window i at 32·i of the plan 19 token file): 257 ids.
    private static string DdpWindows(string folder)
    {
        string path = Path.Combine(folder, "windows8.bin");
        File.WriteAllBytes(path, File.ReadAllBytes(PretrainData("tokens.bin"))[..(257 * 2)]);
        return path;
    }

    // Two ranks of the tiny model training with the Trainer on the fixture's windows (rank r's loader shard: windows r, r+2,
    // r+4, r+6), batch 4 a rank, one step an epoch; rank 0 saves the weights. Returns the global losses, each rank's own.
    private static (float[] Losses, float[][] RankLosses, string Saved) DdpTrain(Device device, string windows, string folder, string name, int epochs,
        Func<IReadOnlyList<Tensor>, Optimizer> make, WireType wire = WireType.Float32)
    {
        string saved = Path.Combine(folder, name);
        var options = DpOptions(wire);
        var ranks = DpRun(DpTopology(device, 2), options, rank =>
        {
            using var model = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = rank.Device });
            var parameters = model.Network.Parameters().Distinct().ToList();   // the tied head once
            foreach (var p in parameters)
            {
                p.RequiresGrad = true;
            }

            using var optimizer = make(parameters);
            var losses = new List<float>();
            var trainer = new Trainer(model.Network, optimizer, (logits, targets) =>
            {
                var loss = Losses.SparseCrossEntropy(logits, targets);
                losses.Add(loss.Item());
                return loss;
            })
            { Cluster = rank.Collectives, ClusterOptions = options };
            using var tokens = new TokenFileSource(windows, 32);
            var history = trainer.Fit(new DataLoader(tokens, batchSize: 4, device: rank.Device), epochs);
            if (rank.Rank == 0)
            {
                model.SaveHuggingFace(saved, SafeTensorType.F32);
            }

            return (Global: history.Epochs.Select(e => (float)e.Loss).ToArray(), Own: losses.ToArray());
        });
        return (ranks[0].Global, [.. ranks.Select(r => r.Own)], saved);
    }

    // The gradients of step 1 as the weights moved under SGD at `rate`.
    private static Dictionary<string, float[]> MovedGradients(string saved, float rate)
    {
        var initial = ReadTensors(Path.Combine(TinyLlama, "model.safetensors"));
        var after = ReadTensors(Path.Combine(saved, "model.safetensors"));
        return initial.ToDictionary(p => p.Key, p => p.Value.Values.Zip(after[p.Key].Values, (a, b) => (a - b) / rate).ToArray(), StringComparer.Ordinal);
    }

    private static void ClusterDdpTrainer(Device device)
    {
        string folder = TempFolder("cluster-ddp");
        try
        {
            string windows = DdpWindows(folder);
            var ddp = JsonNode.Parse(File.ReadAllText(ClusterData("ddp/ddp.json")))!;
            using (var source = new TokenFileSource(windows, 32))
            {
                var x = new float[32];
                var y = new float[32];
                source.Read(7, x, y);
                var ids = ddp["window_ids"]!.AsArray()[7]!.AsArray().Select(i => (float)(int)i!).ToArray();
                Check(source.Count == 8 && x.Concat([y[^1]]).SequenceEqual(ids), "the windows are the fixture's");
            }

            var (_, _, one) = DdpTrain(device, windows, folder, "sgd-1", 1, ps => new Sgd(ps, 0.2f));
            var moved = MovedGradients(one, 0.2f);
            var grads = ReadTensors(ClusterData("ddp/grads-step1.safetensors"));
            float largest = grads.Values.Max(g => g.Values.Max(MathF.Abs));
            float gradientGap = grads.Max(g => Gap(g.Value.Values, moved[g.Key]));
            Check(gradientGap <= 1e-4f * largest, $"step 1's gradients (DDP's average) within 1e-4 of the largest, {largest}: {gradientGap:G3}");

            foreach (var (run, make, file) in new (string, Func<IReadOnlyList<Tensor>, Optimizer>, string)[]
                     {
                         ("sgd", ps => new Sgd(ps, 0.2f), "ddp/weights-step3.safetensors"),
                         ("adamw_folded", ps => new AdamW(ps, 1e-3f, 0.9f, 0.95f, 1e-8f, 0.1f), "ddp/adamw-folded/weights-step3.safetensors"),
                     })
            {
                var (losses, rankLosses, saved) = DdpTrain(device, windows, folder, run, 3, make);
                var expected = ddp[run]!["losses"]!.AsArray().Select(l => (float)(double)l!).ToArray();
                AssertClose(expected, losses, 1e-5f, $"{run}: the three steps' losses (the ranks' mean)");
                if (run == "adamw_folded")
                {
                    AssertClose([.. ddp["adamw"]!["losses"]!.AsArray().Select(l => (float)(double)l!)], losses, 1e-5f, "adamw: torch's own AdamW losses");
                }

                for (int r = 0; r < 2; r++)
                {
                    AssertClose([.. ddp[run]!["rank_losses"]!.AsArray()[r]!.AsArray().Select(l => (float)(double)l!)], rankLosses[r], 1e-5f, $"{run}: rank {r}'s own losses (windows {r}, {r + 2}, {r + 4}, {r + 6})");
                }

                // SGD's weights move with the gradients: 1e-5. AdamW divides by the root of the second moment, so a last-bit
                // gradient difference grows where that moment is small: the fixture measured it (its own DDP against one
                // process, vs_single: step 1 gradients 1.3e-7 apart gave folded AdamW weights 6.7e-6 apart), so the bound is
                // that measured growth applied to this run's measured gradient gap, never below 1e-5.
                var single = ddp["vs_single"]!;
                float growth = (float)((double)single["adamw_folded"]!["weights-step3"]! / (double)single["sgd"]!["grads-step1"]!);
                float tolerance = run == "sgd" ? 1e-5f : Math.Max(1e-5f, growth * gradientGap);
                var after = ReadTensors(Path.Combine(saved, "model.safetensors"));
                float gap = ReadTensors(ClusterData(file)).Max(t => Gap(t.Value.Values, after[t.Key].Values));
                Check(gap <= tolerance, $"{run}: the weights after three steps within {tolerance:G3} of DDP's: {gap:G3}");
                Console.WriteLine($"    measured: {run} weights {gap:G3} (bound {tolerance:G3}), step 1 gradients {gradientGap:G3}");
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void ClusterDdpBFloat16(Device device)
    {
        string folder = TempFolder("cluster-bf16");
        try
        {
            string windows = DdpWindows(folder);
            var bf16 = JsonNode.Parse(File.ReadAllText(ClusterData("bf16/bf16.json")))!;
            // The fixture's measured gaps of a bf16 wire against a float32 one: the tolerances of a bf16 run.
            var measured = bf16["vs_ddp"]!;
            float gradients = (float)(double)measured["grads-step1"]!, losses = (float)(double)measured["losses"]!, weights = (float)(double)measured["weights-step3"]!;
            var (_, _, one) = DdpTrain(device, windows, folder, "bf16-1", 1, ps => new Sgd(ps, 0.2f), WireType.BFloat16);
            var moved = MovedGradients(one, 0.2f);
            float gradientGap = ReadTensors(ClusterData("bf16/grads-step1.safetensors")).Max(g => Gap(g.Value.Values, moved[g.Key]));
            Check(gradientGap <= gradients, $"bf16: step 1's gradients within the fixture's measured {gradients:G3}: {gradientGap:G3}");
            var (curve, _, saved) = DdpTrain(device, windows, folder, "bf16-3", 3, ps => new Sgd(ps, 0.2f), WireType.BFloat16);
            var expected = bf16["sgd"]!["losses"]!.AsArray().Select(l => (float)(double)l!).ToArray();
            float lossGap = Gap(expected, curve);
            Check(lossGap <= losses, $"bf16: three steps' losses within the fixture's measured {losses:G3}: {lossGap:G3}");
            var after = ReadTensors(Path.Combine(saved, "model.safetensors"));
            float weightGap = ReadTensors(ClusterData("bf16/weights-step3.safetensors")).Max(t => Gap(t.Value.Values, after[t.Key].Values));
            Check(weightGap <= weights, $"bf16: the weights after three steps within the fixture's measured {weights:G3}: {weightGap:G3}");
            Console.WriteLine($"    measured: bf16 gradients {gradientGap:G3}, losses {lossGap:G3}, weights {weightGap:G3}");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // A small classifier without batch norm or dropout, so one process and its ranks compute the same function.
    private static Sequential DpClassifier(Device device)
    {
        var init = new Random(3);
        return new Sequential { new Linear(2, 16, device: device, random: init), new ReLU(), new Linear(16, 3, device: device, random: init) };
    }

    private static PretrainingOptions DpPretraining(string tokens, int steps, ICollectives? cluster, ClusterOptions? options) => new()
    {
        Stages = [new PretrainingStage("text") { Sources = [new PretrainingSource(PretrainingSources.Tokens, [tokens])], Steps = steps, LearningRate = 0.2f, Schedule = "constant", WarmupFraction = 0 }],
        MaxLength = 32, BatchTokens = 64, GradientAccumulation = 2, Optimizer = "sgd", WeightDecay = 0f, Momentum = 0f, MaxGradientNorm = 0f, CudaGraphs = false,
        KeepCheckpoints = 0, Seed = 3, Cluster = cluster, ClusterOptions = options,
    };

    // A pretraining run's losses and its last checkpoint's weights (concatenated in name order).
    private static (float[] Losses, float[] Weights) DpPretrain(Device device, PretrainingOptions options, string folder, Action<string>? trace = null, PretrainingCheckpoint? resume = null,
        string? model = null)
    {
        using var loaded = PretrainedModel.Load(model ?? TinyLlama, new PretrainedOptions { Device = device });
        var result = Pretrainer.Train(loaded, options, folder, resume, trace: trace);
        float[] weights = result.LastCheckpoint is { } last && File.Exists(Path.Combine(last, "model.safetensors"))
            ? [.. ReadTensors(Path.Combine(last, "model.safetensors")).OrderBy(t => t.Key, StringComparer.Ordinal).SelectMany(t => t.Value.Values)]
            : [];
        return ([.. result.Losses], weights);
    }

    private static void ClusterEqualsOneProcess(Device device)
    {
        // The Trainer: batch 20 in one process; batch 10 on each of 2 ranks (rank r takes the epoch's samples 2k + r).
        var data = Spirals(50, 3, 1);
        (float[] Curve, float[] Weights) Fit(Device on, int batch, ICollectives? cluster, ClusterOptions? options)
        {
            using var model = DpClassifier(on);
            using var optimizer = new Sgd(model.Parameters(), 0.1f);
            var trainer = new Trainer(model, optimizer, Losses.CrossEntropy) { Cluster = cluster, ClusterOptions = options, Metrics = { Metric.Accuracy } };
            var history = trainer.Fit(new DataLoader(data, batch, shuffle: true, device: on, seed: 7), 3, validation: new DataLoader(data, 32, device: on));
            return ([.. history.Epochs.SelectMany(e => new[] { (float)e.Loss, (float)e.ValidationLoss!.Value, (float)e.Metrics["accuracy"] })], [.. model.Parameters().SelectMany(p => p.ToArray())]);
        }

        var alone = Fit(device, 20, null, null);
        var ranks = DpRun(DpTopology(device, 2), DpOptions(), rank => Fit(rank.Device, 10, rank.Collectives, DpOptions()));
        Check(ranks[0].Weights.SequenceEqual(ranks[1].Weights) && ranks[0].Curve.SequenceEqual(ranks[1].Curve), "Trainer: every rank holds the same weights and reports the same losses, bit for bit");
        AssertClose(alone.Curve, ranks[0].Curve, 1e-5f, "Trainer: the epochs' training and validation losses and accuracy against one process with batch 20");
        AssertClose(alone.Weights, ranks[0].Weights, 1e-5f, "Trainer: the weights after three epochs against one process");

        string folder = TempFolder("cluster-one");
        try
        {
            // The fine-tuner: accumulation 1 on 2 ranks is one process's accumulation 2 (the same batches from the seed).
            var tokens = File.ReadAllBytes(PretrainData("tokens.bin"));
            var sequences = Enumerable.Range(0, 8).Select(w => Enumerable.Range(0, 33).Select(i => (int)BitConverter.ToUInt16(tokens, ((w * 32) + i) * 2)).ToArray())
                .Select(ids => new TrainingSequence(ids, [.. ids.Select(_ => true)])).ToList();
            var tuning = new FineTuningOptions { Rank = 4, Alpha = 8, LearningRate = 1e-3f, Epochs = 1, MaxLength = 64, BatchTokens = 64, CudaGraphs = false, Seed = 1, WarmupFraction = 0 };
            (float[] Curve, float[] Weights) Tune(Device on, FineTuningOptions options)
            {
                using var model = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = on });
                var curve = new List<float>();
                FineTuner.Train(model, sequences, null, options, progress: new SynchronousProgress<FineTuningProgress>(p => curve.Add(p.Loss)));
                return ([.. curve], [.. model.Network.TrainableParameters().SelectMany(p => p.ToArray())]);
            }

            var tunedAlone = Tune(device, tuning with { GradientAccumulation = 2 });
            var tuned = DpRun(DpTopology(device, 2), DpOptions(), rank => Tune(rank.Device, tuning with { Cluster = rank.Collectives, ClusterOptions = DpOptions() }));
            Check(tunedAlone.Curve.Length >= 2 && tuned[0].Weights.SequenceEqual(tuned[1].Weights), $"fine-tuner: {tunedAlone.Curve.Length} steps; every rank's adapters the same");
            AssertClose(tunedAlone.Curve, tuned[0].Curve, 1e-5f, "fine-tuner: the steps' losses against one process");
            AssertClose(tunedAlone.Weights, tuned[0].Weights, 1e-5f, "fine-tuner: the adapters against one process");

            // The pretrainer: the options describe the step (4 windows); each of 2 ranks trains 2 of them.
            string corpus = PretrainData("tokens.bin");
            var pretrainedAlone = DpPretrain(device, DpPretraining(corpus, 3, null, null), Path.Combine(folder, "alone"));
            var pretrained = DpRun(DpTopology(device, 2), DpOptions(), rank =>
                DpPretrain(rank.Device, DpPretraining(corpus, 3, rank.Collectives, DpOptions()), Path.Combine(folder, "two")));
            Check(pretrained[0].Losses.SequenceEqual(pretrained[1].Losses), "pretrainer: every rank reports the same losses");
            AssertClose(pretrainedAlone.Losses, pretrained[0].Losses, 1e-5f, "pretrainer: the steps' losses against one process");
            AssertClose(pretrainedAlone.Weights, pretrained[0].Weights, 1e-5f, "pretrainer: the checkpoint's weights against one process");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void ClusterUnequalRanks(Device device)
    {
        string folder = TempFolder("cluster-unequal");
        try
        {
            // Four sequences a step over 3 ranks: rank 0 trains two, ranks 1 and 2 one each; a micro-batch a sequence.
            string corpus = PretrainData("tokens.bin");
            PretrainingOptions Options(ICollectives? cluster) => DpPretraining(corpus, 3, cluster, DpOptions()) with { SequencesPerStep = 4, BatchTokens = 32, GradientAccumulation = 1 };
            var alone = DpPretrain(device, Options(null), Path.Combine(folder, "alone"));
            var ranks = DpRun(DpTopology(device, 3), DpOptions(), rank => DpPretrain(rank.Device, Options(rank.Collectives), Path.Combine(folder, "three")));
            AssertClose(alone.Losses, ranks[0].Losses, 1e-5f, "the token-weighted losses of ranks with 2, 1 and 1 sequences against one process");
            AssertClose(alone.Weights, ranks[0].Weights, 1e-5f, "the weights against one process");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void ClusterResumeOtherWorld(Device device)
    {
        string folder = TempFolder("cluster-resume");
        try
        {
            // A step of three windows (accumulation 3 of one row) at every world size; a checkpoint every second step.
            string corpus = PretrainData("tokens.bin");
            PretrainingOptions Options(int steps, ICollectives? cluster) =>
                DpPretraining(corpus, steps, cluster, DpOptions()) with { BatchTokens = 32, GradientAccumulation = 3, SaveEvery = 2 };
            var whole = DpRun(DpTopology(device, 3), DpOptions(), rank => DpPretrain(rank.Device, Options(4, rank.Collectives), Path.Combine(folder, "three")));

            // Two ranks for two steps: rank 0 writes the run (rank 1 is given a folder of its own, to see it writes nothing).
            string two = Path.Combine(folder, "two");
            DpRun(DpTopology(device, 2), DpOptions(), rank => DpPretrain(rank.Device, Options(2, rank.Collectives), rank.Rank == 0 ? two : Path.Combine(folder, "two-rank1")));
            Check(Directory.Exists(Path.Combine(two, "checkpoint-2")) && Directory.GetDirectories(Path.Combine(folder, "two-rank1")).Length == 0,
                "rank 0 alone wrote the checkpoint");

            string latest = Pretrainer.LatestCheckpoint(two)!;
            var checkpoint = Pretrainer.ReadCheckpoint(latest);
            Check(checkpoint.Step == 2 && checkpoint.Sources[0].Drawn == 6, $"the checkpoint is where every rank's draws were: step {checkpoint.Step}, drawn {checkpoint.Sources[0].Drawn}");
            var resumed = DpRun(DpTopology(device, 3), DpOptions(), rank => DpPretrain(rank.Device, Options(4, rank.Collectives), two, resume: checkpoint, model: latest));
            Check(resumed[0].Losses.Length == 4, $"the resumed run has the four steps' losses: {resumed[0].Losses.Length}");
            AssertClose(whole[0].Losses, resumed[0].Losses, 1e-5f, "2 ranks resumed as 3: the 3-rank run's losses");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // Each rank's weights after every step; rank 1's are moved after step 1, as a rank of another kind rounds apart.
    private sealed class WeightProbe : ITrainerCallback
    {
        public List<float[]> After { get; } = [];

        public void OnBatchEnd(TrainerContext context, BatchCompleted batch)
        {
            var parameters = context.Model.Parameters().ToList();
            if (context.Rank == 1 && context.Step == 1)
            {
                var values = parameters[0].ToArray();
                values[0] += 0.5f;
                parameters[0].Load(values);
            }

            After.Add([.. parameters.SelectMany(p => p.ToArray())]);
        }
    }

    private static void ClusterSyncWeights(Device device)
    {
        if (!FirstRun(nameof(ClusterSyncWeights)))
        {
            return;
        }

        // The CPU and one device of every other kind listed here (only the CPU on a machine without a GPU: two CPU ranks).
        var devices = new List<string> { "cpu" };
        devices.AddRange(Device.Available.Where(d => d.Type != DeviceType.Cpu).GroupBy(d => d.Kind).Select(g => g.First().ToString()));
        if (devices.Count == 1)
        {
            devices.Add("cpu");
        }

        int world = devices.Count, batch = (int)Math.Ceiling(60.0 / world / 4);
        var options = DpOptions() with { SyncWeightsEvery = 2 };
        var data = Spirals(20, 3, 1);
        var probes = DpRun(DpTopology(devices), options, rank =>
        {
            using var model = DpClassifier(rank.Device);
            using var optimizer = new Sgd(model.Parameters(), 0.1f);
            var probe = new WeightProbe();
            var trainer = new Trainer(model, optimizer, Losses.CrossEntropy) { Cluster = rank.Collectives, ClusterOptions = options, Callbacks = { probe } };
            trainer.Fit(new DataLoader(data, batch, device: rank.Device), 1);
            return probe.After;
        });

        Check(probes.All(p => p.Count == 4), $"{world} ranks on {string.Join(", ", devices)}: four steps each ({string.Join(", ", probes.Select(p => p.Count))})");
        Check(!probes[1][0].SequenceEqual(probes[0][0]), "rank 1 moved apart after step 1");
        foreach (int step in new[] { 2, 4 })
        {
            Check(probes.All(p => p[step - 1].SequenceEqual(probes[0][step - 1])), $"step {step}: every rank holds rank 0's weights, bit for bit");
        }
    }

    private static void ClusterSplitBySpeed(Device device)
    {
        string folder = TempFolder("cluster-speed");
        try
        {
            // Eight sequences a step, one a micro-batch; rank 1 sleeps at every batch, so its measured speed is lower.
            string corpus = PretrainData("tokens.bin");
            PretrainingOptions Options(ICollectives? cluster, ClusterOptions? options) =>
                DpPretraining(corpus, 8, cluster, options) with { SequencesPerStep = 8, BatchTokens = 32, GradientAccumulation = 1 };
            var alone = DpPretrain(device, Options(null, null), Path.Combine(folder, "alone"));
            var measuring = DpOptions() with { RebalanceSteps = 2, RebalanceTolerance = 0.1 };
            var lines = new List<string>();
            var ranks = DpRun(DpTopology(device, 2), measuring, rank => DpPretrain(rank.Device, Options(rank.Collectives, measuring), Path.Combine(folder, "two"), line =>
            {
                if (rank.Rank == 1 && line.EndsWith('…'))
                {
                    Thread.Sleep(25);
                }
                else if (rank.Rank == 0)
                {
                    lines.Add(line);
                }
            }));
            string? split = lines.FirstOrDefault(l => l.StartsWith("rows split by the ranks' measured speeds", StringComparison.Ordinal));
            var share = split is null ? null : Regex.Match(split, @"rank 0 (\d+)");
            Check(share is { Success: true } && int.Parse(share.Groups[1].Value, CultureInfo.InvariantCulture) > 50, $"the slow rank takes fewer rows: {split ?? "no split traced"}");
            AssertClose(alone.Losses, ranks[0].Losses, 1e-5f, "the losses with rows split by speed against one process");
            AssertClose(alone.Weights, ranks[0].Weights, 1e-5f, "the weights against one process");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void ClusterHostOptimizer(Device device)
    {
        var options = DpOptions();
        double[] tokens = [1, 3];
        var results = DpRun(DpTopology(device, 2), options, rank =>
        {
            int r = rank.Rank;
            using var p = Tensor.Persistent([.. Enumerable.Repeat(10f, 12)], [12], rank.Device, requiresGrad: true);
            using var optimizer = new HostOptimizer([p], ps => new Sgd(ps, 1f));
            using var sync = GradientSyncs.Create(rank.Collectives, optimizer, options);
            float[] own = [.. Enumerable.Range(0, 12).Select(j => (float)((r + 1) * j))];
            p.Backend.Upload(own, p.GradStorage());
            var reduced = sync.Reduce(tokens[r], 1);
            float[] host = optimizer.HostGradient(0).ToArray(), onDevice = p.Grad!.ToArray();
            optimizer.ClipAndStep(0f);
            return (Reduced: reduced, Host: host, OnDevice: onDevice, Own: own, After: p.ToArray());
        });
        float[] expected = [.. Enumerable.Range(0, 12).Select(j => 1.75f * j)];
        foreach (var (result, r) in results.Select((x, r) => (x, r)))
        {
            Check(result.Reduced.Tokens == 4 && result.Host.SequenceEqual(expected), $"rank {r}: the host gradient is the token-weighted sum: [{string.Join(", ", result.Host)}]");
            Check(result.OnDevice.SequenceEqual(result.Own), $"rank {r}: the device gradient was not moved (the sum was made on the host pieces)");
            Check(result.After.SequenceEqual(expected.Select(g => 10f - g)), $"rank {r}: the update read the reduced host values: [{string.Join(", ", result.After)}]");
        }

        string folder = TempFolder("cluster-host");
        try
        {
            string corpus = PretrainData("tokens.bin");
            var plain = DpRun(DpTopology(device, 2), options, rank => DpPretrain(rank.Device, DpPretraining(corpus, 3, rank.Collectives, options), Path.Combine(folder, "plain")));
            var host = DpRun(DpTopology(device, 2), options, rank =>
                DpPretrain(rank.Device, DpPretraining(corpus, 3, rank.Collectives, options) with { HostOptimizer = true }, Path.Combine(folder, "host")));
            AssertClose(plain[0].Losses, host[0].Losses, 1e-5f, "the pretrainer's losses with the HostOptimizer against without");
            AssertClose(plain[0].Weights, host[0].Weights, 1e-5f, "the weights with the HostOptimizer against without");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void ClusterSpirals(Device device)
    {
        // The sample's data, model and settings (samples/Idrak.Samples.Spirals): three spirals of 300 points, batch norm,
        // AdamW with a cosine schedule, cross-entropy with label smoothing; fewer epochs. Its batch of 64 is 32 a rank.
        const int Classes = 3, PerClass = 300, Epochs = 30;
        var random = new Random(1);
        var features = new float[Classes * PerClass, 2];
        var labels = new int[Classes * PerClass];
        for (int c = 0; c < Classes; c++)
        {
            for (int i = 0; i < PerClass; i++)
            {
                int row = (c * PerClass) + i;
                double radius = i / (double)PerClass, angle = (c * 2 * Math.PI / Classes) + (radius * 5) + (random.NextDouble() * 0.25);
                (features[row, 0], features[row, 1], labels[row]) = ((float)(radius * Math.Cos(angle)), (float)(radius * Math.Sin(angle)), c);
            }
        }

        var (train, test) = Dataset.FromClassLabels(features, labels, Classes).Split(0.8, seed: 2);
        (double Accuracy, float[] Weights, float[] Statistics) Fit(Device on, int batch, ICollectives? cluster)
        {
            var init = new Random(3);
            using var model = new Sequential
            {
                new Linear(2, 128, device: on, random: init), new BatchNorm(128, device: on), new ReLU(),
                new Linear(128, 64, device: on, random: init), new BatchNorm(64, device: on), new ReLU(),
                new Linear(64, Classes, device: on, random: init),
            };
            using var optimizer = new AdamW(model.Parameters(), learningRate: 0.01f, weightDecay: 1e-4f);
            var trainer = new Trainer(model, optimizer, (logits, targets) => Losses.CrossEntropy(logits, targets, labelSmoothing: 0.05f))
            {
                Metrics = { Metric.Accuracy },
                Scheduler = new CosineAnnealing(optimizer, totalEpochs: Epochs, warmupEpochs: 5),
                Cluster = cluster,
                ClusterOptions = DpOptions(),
            };
            var history = trainer.Fit(new DataLoader(train, batch, shuffle: true, device: on, seed: 4), Epochs, validation: new DataLoader(test, 512, device: on));
            return (history.Epochs[^1].ValidationMetrics!["accuracy"], [.. model.Parameters().SelectMany(p => p.ToArray())], [.. model.Buffers().SelectMany(b => b.ToArray())]);
        }

        var alone = Fit(device, 64, null);
        var ranks = DpRun(DpTopology(device, 2), DpOptions(), rank => Fit(rank.Device, 32, rank.Collectives));
        Check(ranks[0].Weights.SequenceEqual(ranks[1].Weights) && ranks[0].Statistics.SequenceEqual(ranks[1].Statistics) && ranks[0].Accuracy == ranks[1].Accuracy,
            "every rank holds the same weights and batch-norm statistics and reports the same accuracy");
        Check(ranks[0].Accuracy >= alone.Accuracy - 0.05 && ranks[0].Accuracy > 0.7, $"2 ranks learn the spirals as one process does: {ranks[0].Accuracy:P1} against {alone.Accuracy:P1}");
    }

    private static void ClusterCliSim(Device device)
    {
        if (!FirstRun(nameof(ClusterCliSim)))
        {
            return;
        }

        string folder = TempFolder("cluster-cli-sim");
        try
        {
            string topology = Path.Combine(folder, "two.json");
            File.WriteAllText(topology, DpTopology(Device.Cpu, 2).ToJson());
            string run = Path.Combine(folder, "run");
            (int Code, string Output, string Error) Idrak(params string[] args)
            {
                using var output = new StringWriter(CultureInfo.InvariantCulture);
                using var error = new StringWriter(CultureInfo.InvariantCulture);
                int code = CommandLine.Run(args, output, error);
                return (code, output.ToString(), error.ToString());
            }

            var (code, text, errors) = Idrak("cluster", "sim", "--topology", topology, "--", "pretrain", TinyLlama, "--tokens", PretrainData("tokens.bin"), "--steps", "2",
                "--max-length", "32", "--batch-tokens", "64", "--optimizer", "sgd", "--lr", "0.2", "--no-graphs", "-o", run);
            Check(code == ExitCodes.Ok, $"cluster sim ran pretrain on two ranks: exit {code}: {errors}\n{text}");
            Check(text.Contains("simulating 2 rank(s)", StringComparison.Ordinal) && text.Contains("network time on the virtual clock: rank 0", StringComparison.Ordinal)
                  && text.Contains("rank 1", StringComparison.Ordinal), $"the ranks and their network time are reported: {text}");
            Check(File.Exists(Path.Combine(run, "run.json")) && File.Exists(Path.Combine(run, "log.jsonl")) && Pretrainer.LatestCheckpoint(run) is { } last && last.EndsWith("checkpoint-2", StringComparison.Ordinal),
                "rank 0 wrote the run: run.json, the log and the stage's checkpoint");

            var (other, _, otherError) = Idrak("cluster", "sim", "--topology", topology, "--", "chat", TinyLlama, "-o", run);
            Check(other == ExitCodes.Usage && otherError.Contains("pretrain", StringComparison.Ordinal), $"another command is a usage error naming the one it runs: {otherError}");
            var (missing, _, missingError) = Idrak("cluster", "sim", "--topology", topology, "--", "pretrain", TinyLlama, "--tokens", PretrainData("tokens.bin"));
            Check(missing == ExitCodes.Usage && missingError.Contains("-o RUN", StringComparison.Ordinal), $"a missing run folder is a usage error naming -o: {missingError}");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }
}
