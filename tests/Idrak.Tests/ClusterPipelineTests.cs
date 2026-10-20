// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak;
using Idrak.Abstraction.Cluster;
using Idrak.Cluster;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Text;
using Idrak.Text.Abstractions;
using Idrak.Text.Generation;
using Idrak.Training;

// Plan 23, phase C7: pipeline parallelism. The plan cutting a model into stages, the boundary module (its forward sends
// the activation, its backward receives the gradient), GPipe's micro-batches, a rank holding only its stage, the pretrainer
// through a pipeline (with data-parallel replicas of each stage), and generation through the stages, each with its own
// cache. Every model is plan 19's tiny one, loaded on the host and placed on devices carved from the device under test.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ClusterPipelineGroup =
    [
        ("cluster: a plan covers every block once, embeddings first, head last: cut by bytes (a tied head counting its table), by declared memory or by hand; the gpipe schedule is a registry entry; wrong stages, cuts and layouts are refused by name", ClusterPipelinePlan),
        ("cluster: pipeline 2 stages on sim gives the one-device logits, losses and gradients of the tiny model (2 micro-batches; the tied table's gradient summed alike on both stages)", ClusterPipelineEqualsOneDevice),
        ("cluster: micro-batches 1, 2 and 4 give the same loss: the pretrainer through 2 stages as one process (losses, clipped losses and the gathered checkpoint's weights), and 2 stages × 2 replicas; the bubble is in the telemetry", ClusterPipelineMicroBatches),
        ("cluster: a rank holds only its stage's tensors: InUse per rank against the stage's bytes and the whole model's, the other stages' tensors released", ClusterPipelineHoldsStage),
        ("cluster: chat through a 2-stage pipeline answers as one device does (greedy replies, a second turn reusing the caches, each stage keeping its own)", ClusterPipelineChat),
    ];

    private static ClusterOptions PipeOptions(int? stages = null, int? microBatches = null) =>
        DpOptions() with { Placement = "pipeline", Stages = stages, MicroBatches = microBatches };

    // The tiny model on the host, every parameter trained.
    private static PretrainedModel HostTiny()
    {
        var model = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = Device.Cpu });
        foreach (var p in model.Network.Parameters())
        {
            p.RequiresGrad = true;
        }

        return model;
    }

    private static void ClusterPipelinePlan(Device device)
    {
        if (!FirstRun(nameof(ClusterPipelinePlan)))
        {
            return;
        }

        _ = device;
        Check(PipelineSchedules.Find("gpipe") is { } gpipe && PipelineSchedules.Origin("gpipe") == Overrides.Library
              && gpipe.Order(1, 2, 3).SequenceEqual(new PipelineWork[] { new(PipelineWorkKind.Forward, 0), new(PipelineWorkKind.Forward, 1), new(PipelineWorkKind.Forward, 2),
                  new(PipelineWorkKind.Backward, 2), new(PipelineWorkKind.Backward, 1), new(PipelineWorkKind.Backward, 0) })
              && Math.Abs(gpipe.IdealBubble(2, 4) - 0.2) < 1e-12 && gpipe.InFlight(0, 2, 4) == 4,
            $"Idrak.Network registers gpipe: every forward, then the backward passes in reverse; bubble (S-1)/(M+S-1): {PipelineSchedules.Describe()}");
        ThrowsNaming<NotSupportedException>(() => PipelineSchedules.Get("1f1b"), "an unregistered schedule", "is not registered", "PipelineSchedules.Register", "IDRAK_CLUSTER_SCHEDULE");

        using var model = HostTiny();
        var names = model.Network.Select(m => m.Name).ToList();
        int blocks = model.Network.OfType<DecoderBlock>().Count();
        var single = PipelineStage.PlanFor(model.Network, new PipelineLayout(1, 1), PipeOptions());
        long whole = single.TotalBytes;
        Check(single.Stages is [{ Start: 0 } only] && only.End == names.Count && only.Bytes == whole, $"one stage holds the whole model: {single}");
        for (int stages = 1; stages <= names.Count; stages++)
        {
            var plan = PipelineStage.PlanFor(model.Network, new PipelineLayout(stages, stages), PipeOptions());
            var covered = plan.Stages.SelectMany(s => Enumerable.Range(s.Start, s.Count)).ToList();
            Check(plan.Stages.Count == stages && covered.SequenceEqual(Enumerable.Range(0, names.Count)) && plan.Stages.All(s => s.Count >= 1),
                $"{stages} stage(s): every module once, in order, each stage at least one: {plan}");
            Check(names[plan.Stages[0].Start] == "embed" && names[plan.Stages[^1].End - 1] == "head", $"{stages} stage(s): the embeddings first, the head last: {plan}");
            Check(plan.Stages.Sum(s => model.Network.Skip(s.Start).Take(s.Count).OfType<DecoderBlock>().Count()) == blocks, $"{stages} stage(s): every block in one stage");
            // The best cut by bytes: no other cut into the same number of stages has a smaller largest stage.
            long largest = plan.Stages.Max(s => s.Bytes);
            foreach (var starts in Cuts(names.Count, stages))
            {
                var other = PipelinePlan.FromStarts(plan.Modules, plan.TensorBytes, starts);
                Check(other.Stages.Max(s => s.Bytes) >= largest, $"{stages} stage(s): the cut at [{string.Join(", ", starts)}] holds less at most than {plan}");
            }
        }

        // The tied head reads the embedding's table: the last stage counts it, so two stages each hold well under the whole.
        var two = PipelineStage.PlanFor(model.Network, new PipelineLayout(2, 2), PipeOptions());
        long table = model.Network.OfType<Embedding>().First().Weight.Size * 4L;
        Check(two.Stages[1].Bytes >= table && two.Stages.All(s => s.Bytes < whole) && two.Basis == "bytes",
            $"2 stages by bytes (the head's table in the last): {two}; whole {whole:N0} B, table {table:N0} B");
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"    measured: 2 stages of the tiny model: {two.Stages[0].Bytes:N0} and {two.Stages[1].Bytes:N0} B of {whole:N0} B"));

        // By declared memory: a rank declaring three times the other's takes the larger share.
        var memory = PipelinePlan.Balance(two.Modules, two.TensorBytes, 2, [1L << 20, 3L << 20]);
        Check(memory.Basis == "memory" && memory.Stages[1].Count > two.Stages[1].Count, $"memory 1:3 puts more in the second stage: {memory} (bytes: {two})");
        var topology = ClusterTopology.Create([new ClusterNodeSpec("box", null, [new ClusterDeviceSpec("cpu", 1L << 20), new ClusterDeviceSpec("cpu", 3L << 20)])], null, null);
        var auto = PipelineStage.PlanFor(model.Network, new PipelineLayout(2, 2), PipeOptions(), topology);
        Check(auto.Basis == "memory" && auto.Stages.Select(s => s.Start).SequenceEqual(memory.Stages.Select(s => s.Start)),
            $"auto balance follows the topology's declared memory when it differs: {auto}");
        Check(PipelineStage.PlanFor(model.Network, new PipelineLayout(2, 2), PipeOptions() with { PipelineBalance = PipelineBalance.Bytes }, topology).Basis == "bytes",
            "IDRAK_CLUSTER_PIPELINE_BALANCE=bytes ignores the declared memory");
        var undeclared = ClusterTopology.Create([new ClusterNodeSpec("box", null, [new ClusterDeviceSpec("cpu", null), new ClusterDeviceSpec("cpu", 3L << 20)])], null, null);
        ThrowsNaming<ArgumentException>(() => PipelineStage.PlanFor(model.Network, new PipelineLayout(2, 2), PipeOptions() with { PipelineBalance = PipelineBalance.Memory }, undeclared),
            "a memory cut without a rank's memory", "rank 0", "memory");

        // By hand, and what is refused.
        var cut = PipelineStage.PlanFor(model.Network, new PipelineLayout(2, 2), PipeOptions() with { PipelineCuts = [3] });
        Check(cut.Basis == "cuts" && cut.Stages[1].Start == 3, $"IDRAK_CLUSTER_PIPELINE_CUTS=3: {cut}");
        ThrowsNaming<ArgumentException>(() => PipelineStage.PlanFor(model.Network, new PipelineLayout(3, 3), PipeOptions() with { PipelineCuts = [3] }), "a cut for another stage count",
            "IDRAK_CLUSTER_PIPELINE_CUTS", "IDRAK_CLUSTER_STAGES");
        ThrowsNaming<ArgumentException>(() => PipelinePlan.FromStarts(two.Modules, two.TensorBytes, [0, 3, 3]), "cuts that do not rise", "IDRAK_CLUSTER_PIPELINE_CUTS", "rise");
        ThrowsNaming<ArgumentException>(() => PipelinePlan.Balance(two.Modules, two.TensorBytes, names.Count + 1), "more stages than modules", "IDRAK_CLUSTER_STAGES", "embed");
        ThrowsNaming<ArgumentException>(() => PipelineLayout.For(6, PipeOptions(stages: 4)), "stages that do not divide the ranks", "IDRAK_CLUSTER_STAGES", "1, 2, 3, 6");
        var layout = PipelineLayout.For(6, PipeOptions(stages: 3));
        Check(layout is { Replicas: 2 } && layout.StageOf(3) == 1 && layout.ReplicaOf(3) == 1 && layout.RankOf(2, 0) == 4 && layout.StageRanks(1).SequenceEqual([2, 3]),
            $"6 ranks in 3 stages: each stage's 2 replicas together, stage by stage: {layout}");
        Check(ClusterOptions.From(n => n == "IDRAK_CLUSTER_PIPELINE_CUTS" ? "3, 6" : null, null).PipelineCuts!.SequenceEqual([3, 6]), "the cuts parse from the environment");
        ThrowsNaming<ArgumentException>(() => ClusterOptions.From(n => n == "IDRAK_CLUSTER_PLACEMENT" ? "tensor" : null, null), "an unknown placement", "IDRAK_CLUSTER_PLACEMENT", "single or pipeline");

        // Every way to cut `modules` into `stages` non-empty ranges, as each stage's first module.
        static IEnumerable<int[]> Cuts(int modules, int stages)
        {
            if (stages == 1)
            {
                yield return [0];
                yield break;
            }

            foreach (var rest in Cuts(modules, stages - 1))
            {
                for (int start = rest[^1] + 1; start < modules; start++)
                {
                    if (modules - start >= 1)
                    {
                        yield return [.. rest, start];
                    }
                }
            }
        }
    }

    // Four windows of the plan 19 token file: ids [4, 32] and their next tokens.
    private static (float[] Ids, float[] Targets) PipeWindows()
    {
        var tokens = File.ReadAllBytes(PretrainData("tokens.bin"));
        var ids = new float[4 * 32];
        var targets = new float[4 * 32];
        for (int w = 0; w < 4; w++)
        {
            for (int i = 0; i < 32; i++)
            {
                ids[(w * 32) + i] = BitConverter.ToUInt16(tokens, ((w * 32) + i) * 2);
                targets[(w * 32) + i] = BitConverter.ToUInt16(tokens, ((w * 32) + i + 1) * 2);
            }
        }

        return (ids, targets);
    }

    private static void ClusterPipelineEqualsOneDevice(Device device)
    {
        var (ids, targets) = PipeWindows();
        float[] logits;
        float loss;
        float[][] gradients;
        using (var one = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = device }))
        using (var scope = new TensorScope())
        {
            var parameters = one.Network.Parameters().Distinct().ToList();
            foreach (var p in parameters)
            {
                p.RequiresGrad = true;
            }

            var output = one.Network.Forward(Tensor.From(ids, [4, 32], device));
            var value = Losses.SparseCrossEntropy(output, Tensor.From(targets, [4, 32], device));
            value.Backward();
            (logits, loss, gradients) = (output.ToArray(), value.Item(), [.. parameters.Select(p => p.Grad!.ToArray())]);
        }

        // Two stages, two micro-batches of two rows; each micro-batch's mean loss weighs a half.
        var options = PipeOptions();
        var ranks = DpRun(DpTopology(device, 2), options, rank =>
        {
            using var model = HostTiny();
            using var pipe = PipelineStage.Place(model.Network, rank.Collectives, options);
            var seen = new float[2][];
            var result = pipe.Step(2,
                m => Tensor.From(ids.AsSpan(m * 64, 64), [2, 32], pipe.Device),
                (m, output) =>
                {
                    seen[m] = output.ToArray();
                    return Losses.SparseCrossEntropy(output, Tensor.From(targets.AsSpan(m * 64, 64), [2, 32], pipe.Device)) * 0.5f;
                });
            var grads = pipe.Parameters.Select((p, i) => (Index: pipe.ParameterIndices[i], Values: p.Grad!.ToArray())).ToList();
            return (pipe.Stage, Logits: pipe.IsLast ? seen.SelectMany(s => s).ToArray() : [], result, Grads: grads, Shared: Enumerable.Range(0, pipe.Parameters.Count).Count(i => !pipe.Owns(i)));
        });

        var last = ranks.Single(r => r.Stage == 1);
        float logitGap = Gap(logits, last.Logits), largest = logits.Max(MathF.Abs);
        Check(last.Logits.Length == logits.Length && logitGap <= 1e-5f * Math.Max(1f, largest), $"the last stage's logits are one device's: largest gap {logitGap:G3} of {largest:G3}");
        Check(ranks.All(r => Math.Abs(r.result.Loss - loss) <= 1e-5 * Math.Max(1, Math.Abs(loss))), $"every stage reports one device's loss {loss}: {string.Join(", ", ranks.Select(r => r.result.Loss))}");
        var held = ranks.SelectMany(r => r.Grads.Select(g => (r.Stage, g.Index, g.Values))).ToList();
        Check(held.Select(h => h.Index).Distinct().Order().SequenceEqual(Enumerable.Range(0, gradients.Length)), "every parameter is held by a stage");
        float scale = gradients.Max(g => g.Max(MathF.Abs));
        foreach (var (stage, index, values) in held)
        {
            float gap = Gap(gradients[index], values);
            Check(gap <= 1e-5f * scale, $"stage {stage}: parameter {index}'s gradient within 1e-5 of the largest ({scale:G3}) of one device's: {gap:G3}");
        }

        var tied = held.GroupBy(h => h.Index).Where(g => g.Count() > 1).ToList();
        Check(tied.Count == 1 && tied[0].First().Values.SequenceEqual(tied[0].Last().Values) && ranks.Sum(r => r.Shared) == 1,
            $"the tied table is held by both stages with the same summed gradient, counted by the first alone ({tied.Count} shared)");
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    measured: logits {logitGap:G3}, gradients {held.Max(h => Gap(gradients[h.Index], h.Values)):G3}; bubble {string.Join(", ", ranks.Select(r => $"stage {r.Stage} {r.result.Bubble:P0}"))} (gpipe's {ranks[0].result.IdealBubble:P0})"));
    }

    // A pretraining run through a pipeline: rank r on its carved device, the model loaded on the host; each rank's losses and
    // rank 0's last checkpoint (the whole model, gathered).
    private static (float[] Losses, float[] Weights) PipePretrain(ClusterRank rank, PretrainingOptions options, string folder)
    {
        using var model = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = Device.Cpu });
        var result = Pretrainer.Train(model, options with { Cluster = rank.Collectives }, folder);
        float[] weights = rank.Rank == 0 && result.LastCheckpoint is { } saved
            ? [.. ReadTensors(Path.Combine(saved, "model.safetensors")).OrderBy(t => t.Key, StringComparer.Ordinal).SelectMany(t => t.Value.Values)]
            : [];
        return ([.. result.Losses], weights);
    }

    private static void ClusterPipelineMicroBatches(Device device)
    {
        string folder = TempFolder("cluster-pipeline");
        try
        {
            // A step of four windows (two batches of two rows); no packing, so every micro-batch is padded rows alike.
            string corpus = PretrainData("tokens.bin");
            PretrainingOptions Options(ClusterOptions? cluster, float clip = 0f) => DpPretraining(corpus, 3, null, cluster) with { Packing = false, MaxGradientNorm = clip, SaveEvery = 3 };
            var alone = DpPretrain(device, Options(null), Path.Combine(folder, "alone"));
            var clipped = DpPretrain(device, Options(null, 0.5f), Path.Combine(folder, "alone-clipped"));
            var bubbles = new List<string>();
            foreach (int m in new[] { 1, 2, 4 })
            {
                var cluster = PipeOptions(microBatches: m);
                var (ranks, seen) = OverlapRun(DpTopology(device, 2), cluster, rank => PipePretrain(rank, Options(cluster), Path.Combine(folder, $"m{m}")));
                Check(ranks[0].Losses.SequenceEqual(ranks[1].Losses), $"{m} micro-batch(es): both stages report the same losses");
                AssertClose(alone.Losses, ranks[0].Losses, 1e-5f, $"{m} micro-batch(es): the losses against one process");
                AssertClose(alone.Weights, ranks[0].Weights, 1e-5f, $"{m} micro-batch(es): the gathered checkpoint's weights against one process");
                var steps = seen.PipelineSteps;
                Check(steps.Count == 6 && steps.All(s => s.MicroBatches == m && s.Stages == 2 && s.Schedule == "gpipe" && s.BytesSent > 0 && s.Bubble is >= 0 and <= 1),
                    $"{m} micro-batch(es): each rank's step in the telemetry ({steps.Count})");
                // The first, cold step aside (rule 76).
                double Bubble(int stage) => steps.Where(s => s.Stage == stage && s.Step > 0).Average(s => s.Bubble);
                bubbles.Add(string.Create(CultureInfo.InvariantCulture, $"M={m}: stage 0 {Bubble(0):P0}, stage 1 {Bubble(1):P0} (gpipe's {steps[0].IdealBubble:P0})"));
            }

            Console.WriteLine($"    measured: bubble {string.Join("; ", bubbles)}");
            var clippedPipe = DpRun(DpTopology(device, 2), PipeOptions(microBatches: 2), rank => PipePretrain(rank, Options(PipeOptions(microBatches: 2), 0.5f), Path.Combine(folder, "clipped")));
            AssertClose(clipped.Losses, clippedPipe[0].Losses, 1e-5f, "clipped at 0.5 by the norm over both stages: the losses against one process");
            AssertClose(clipped.Weights, clippedPipe[0].Weights, 1e-5f, "clipped: the weights against one process");

            // Two stages of two replicas: each pipeline trains half the step's windows, the stages' replicas sum their gradients.
            var replicated = PipeOptions(stages: 2, microBatches: 1);
            var four = DpRun(DpTopology(device, 4), replicated, rank => PipePretrain(rank, Options(replicated), Path.Combine(folder, "replicas")));
            Check(four.All(r => r.Losses.SequenceEqual(four[0].Losses)), "2 stages × 2 replicas: every rank reports the same losses");
            AssertClose(alone.Losses, four[0].Losses, 1e-5f, "2 stages × 2 replicas: the losses against one process");
            AssertClose(alone.Weights, four[0].Weights, 1e-5f, "2 stages × 2 replicas: the weights against one process");

            // What a pipeline does not carry is refused by name.
            var refused = InProcessCluster.Run(DpTopology(device, 2), PipeOptions(), rank =>
            {
                using var model = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = Device.Cpu });
                var tuning = new FineTuningOptions { Rank = 4, Alpha = 8, Epochs = 1, MaxLength = 64, BatchTokens = 64, CudaGraphs = false, Cluster = rank.Collectives, ClusterOptions = PipeOptions() };
                FineTuner.Train(model, [new TrainingSequence([1, 2, 3, 4], [true, true, true, true])], null, tuning);
            });
            Check(refused.Errors.All(e => e is NotSupportedException && e.Message.Contains("Pretrainer", StringComparison.Ordinal) && e.Message.Contains("IDRAK_CLUSTER_PLACEMENT", StringComparison.Ordinal)),
                $"the fine-tuner refuses pipeline placement naming the pretrainer: {string.Join(" | ", refused.Errors.Select(e => e?.Message))}");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void ClusterPipelineHoldsStage(Device device)
    {
        long whole;
        using (var model = HostTiny())
        {
            whole = PipelineStage.PlanFor(model.Network, new PipelineLayout(1, 1), PipeOptions()).TotalBytes;
        }

        // Earlier tests' tensors dropped without a release go back to their devices when the finalizers run: before the measure
        // (plan 23, C8b: CUDA's stage 0 once showed 252 B less than its own tensors, a release inside the measured window, or a
        // tensor of the stage not on its device; the message below names which). Storages are tracked from here on.
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        using var tracking = StorageTracking.Track();
        var ranks = DpRun(DpTopology(device, 2), PipeOptions(), rank =>
        {
            var held = rank.Device.Backend.LiveStorages().Select(l => l.Storage).ToHashSet(ReferenceEqualityComparer.Instance);
            long before = rank.Device.Backend.GetMemoryUsage().InUse;
            using var model = HostTiny();
            var all = model.Network.Parameters().Distinct().ToList();
            using var pipe = PipelineStage.Place(model.Network, rank.Collectives, PipeOptions());
            long inUse = rank.Device.Backend.GetMemoryUsage().InUse - before;
            var others = model.Network.Parameters().Distinct().Where(p => !pipe.Parameters.Contains(p)).ToList();

            // Each of the stage's tensors (parameters, buffers, a tied head's table) where it is: on this rank's device and
            // in memory, or not (named with its bytes and where it is instead).
            var tensors = pipe.Modules.SelectMany(m => m.Parameters().Concat(m.Buffers())
                    .Concat(m.Descendants().OfType<Linear>().Select(l => l.TiedTo).OfType<Module>().SelectMany(t => t.Parameters().Concat(t.Buffers()))))
                .Distinct(ReferenceEqualityComparer.Instance).Cast<Tensor>().ToList();
            var names = new Dictionary<Tensor, string>(ReferenceEqualityComparer.Instance);
            foreach (var (path, module) in model.NamedModules())
            {
                foreach (var t in module.Parameters().Concat(module.Buffers()))
                {
                    names[t] = path;                                        // the innermost module holding it, walked last
                }
            }

            string Where(Tensor t) => t.IsDisposed ? "released" : t.Device != rank.Device ? $"on {t.Device}" : t.Storage.Evicted ? "evicted" : !t.Storage.Alive ? "returned" : "resident";
            var away = tensors.Where(t => Where(t) != "resident")
                .Select(t => $"{names.GetValueOrDefault(t, "?")} {Tensor.FormatShape(t.Shape)} {(long)t.Size * sizeof(float):N0} B {Where(t)}").ToList();
            long resident = tensors.Where(t => Where(t) == "resident").Select(t => t.Storage).Distinct(ReferenceEqualityComparer.Instance).Cast<Storage>().Sum(st => (long)st.Length * sizeof(float));
            var now = rank.Device.Backend.LiveStorages();
            string changed = $"made while placing: {StorageTracking.Describe([.. now.Where(l => !held.Contains(l.Storage))])}; "
                + $"tracked before and released meanwhile: {held.Count(h => !now.Any(l => ReferenceEquals(l.Storage, h)))}";
            return (pipe.Stage, pipe.Bytes, InUse: inUse, OnDevice: pipe.Parameters.All(p => p.Device == rank.Device),
                Released: others.All(p => p.IsDisposed), Modules: pipe.Modules.Select(m => m.Name).ToArray(), Away: away, Resident: resident, Changed: changed);
        });

        foreach (var r in ranks)
        {
            Check(r.OnDevice && r.Released, $"stage {r.Stage} ({string.Join(", ", r.Modules)}): its parameters on its device, every other stage's released");
            Check(r.Away.Count == 0 && r.Resident >= r.Bytes, $"stage {r.Stage}: its tensors in memory on its device ({r.Resident:N0} B of its {r.Bytes:N0} B); not: [{string.Join("; ", r.Away)}]");
            Check(r.InUse >= r.Bytes && r.InUse < whole, $"stage {r.Stage}: {r.InUse:N0} B in use on its device, at least its {r.Bytes:N0} B and under the whole model's {whole:N0} B; "
                + $"tensors not in memory there: [{string.Join("; ", r.Away)}]; {r.Changed} (IDRAK_TRACK_STORAGES=1 names what earlier tests left too)");
        }

        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    measured: in use per rank {string.Join(", ", ranks.Select(r => $"stage {r.Stage} {r.InUse:N0} B ({r.InUse * 100.0 / whole:F0}%)"))} of the whole model's {whole:N0} B"));
    }

    private static void ClusterPipelineChat(Device device)
    {
        var options = new GenerationOptions { Temperature = 0f, TopK = 1, TopP = 1f, RepeatPenalty = 1f, NumPredict = 8 };
        var requests = new[] { "hi", "2+2?" }.Select(q => new ChatRequest([new ChatMessage("user", q)], null, null, options)).ToList();
        // A second turn: the first answer and a new question, so the kept cache is cut back to the shared prefix.
        string[] Converse(ChatGenerator chat)
        {
            var first = requests.Select(r => chat.Chat(r).Message!.Content).ToList();
            var turn = new ChatRequest([new ChatMessage("user", "hi"), new ChatMessage("assistant", first[0]), new ChatMessage("user", "more")], null, null, options);
            return [.. first, chat.Chat(turn).Message!.Content];
        }

        string[] alone;
        using (var model = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = device }))
        {
            alone = Converse(model.CreateChat());
        }

        var ranks = DpRun(DpTopology(device, 2), PipeOptions(), rank =>
        {
            using var model = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = Device.Cpu });
            using var pipe = PipelineStage.Place(model.Network, rank.Collectives, PipeOptions());
            if (!pipe.IsFirst)
            {
                pipe.Serve();
                return [];
            }

            var answers = Converse(new ChatGenerator(new TextGenerator(pipe.Network, model.Tokenizer!, model.MaxPositions), model.ChatTemplate));
            pipe.StopServing();
            return answers;
        });

        Check(ranks[0].SequenceEqual(alone), $"the pipeline's replies are one device's: [{string.Join(" | ", ranks[0])}] against [{string.Join(" | ", alone)}]");
    }
}
