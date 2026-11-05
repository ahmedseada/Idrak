// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak;
using Idrak.Abstraction.Cluster;
using Idrak.Cli;
using Idrak.Cluster;
using Idrak.Datasets;
using Idrak.Datasets.Abstractions;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Networking.Abstractions;
using Idrak.Networking.Compression;
using Idrak.Text;
using Idrak.Text.Abstractions;
using Idrak.Training;

// Plan 23, phase C4: gradient buckets reduced while the backward pass continues (a hook per parameter), the bucket size
// measured on the first step, compression by name (bf16, top-k with error feedback), local SGD beside all-reduce; and the
// open item of C2 and C3: the trainers' step loop takes part in ClusterRank.BeginStep, so a simulated training run gets
// its faults (a slow worker, a late joiner, a replay's end) with the right group and token weighting.
internal abstract partial class DesktopTests : Tests
{
    // Measured on the CPU (2026-10-18, see plans/done/23-cluster.md "C4 as built"): top-k 10% with error feedback against the
    // full gradient on the Spirals sample, the last epoch's training loss.
    private const float TopKLossGap = 0.02f;                           // measured 0.0167

    private static readonly (string Name, Action<Device> Run)[] ClusterOverlapGroup =
    [
        ("cluster: the gradient-ready hook names each parameter once its backward pass finished it, the last layers first; a checkpointed block's as not final; none outside the scope", ClusterGradientReadyHook),
        ("cluster: bucketed overlap equals the unbucketed step bit for bit: buckets sent during the backward pass against one bucket after it (2 ranks ring, 3 ranks naive) and against the same buckets after it (3 ranks ring); the telemetry counts the buckets sent during the backward pass and names a lost overlap", ClusterOverlapEqualsAfter),
        ("cluster: overlap shortens the simulated step on a slow link (ethernet-1g): the exposed network time a step, the simulator's number, is less with the buckets sent during the backward pass; the bucket size is measured once on the first step, the same on every run", ClusterOverlapShortens),
        ("cluster: compressions are a registry picked by name (none, bf16, top-k, int8, fp8); top-k sends each rank's largest values as pairs summed in rank order and keeps the rest as a residual for the next step", ClusterCompressions),
        ("cluster: fp8 E4M3 codes as the format defines them: every code's value, every finite code back to itself, the nearest code with ties to even (every midpoint, its neighbours, and values over every size against a search of all codes), saturation at 448, NaN", ClusterFloat8Codes),
        ("cluster: int8 and fp8 compressions on 2 and 3 ranks, bit for bit as their definition (written apart) gives: one byte a value times a scale, by the way of fewer bytes: each piece to its owner (all-to-all), summed there in rank order, rounded again and all-gathered (3 ranks), or every rank's codes all-gathered (2 ranks, small buckets, collectives without an all-to-all); the vector and scalar paths alike; both roundings sent with the next step (error feedback) and none without it; a rank with no rows adds nothing and keeps its residual; a NaN travels and drops the residual; 4x fewer bytes on the wire than float32 on 2, 4 and 8 ranks, every rank alike", ClusterQuantizedCompressions),
        ("cluster: all-to-all (ring on 1 and 3 ranks, naive on 3, hierarchical on 2 nodes of 2) gives each rank every rank's block for it, bit for bit (NaN patterns too), published as all-to-all with a block to each other rank", ClusterAllToAll),
        ("cluster: top-k (10%) with error feedback converges on the Spirals sample within 0.02 of the full-gradient training loss after 20 epochs (the tolerance measured on the CPU), sending under half the bytes", ClusterTopKSpirals),
        ("cluster: local SGD every k steps equals one process at k = 1 (plain SGD, the weights averaged by each rank's rows) and runs k times fewer weight averages at k = 4, the weights alike on every rank after each", ClusterLocalSgd),
        ("cluster: a simulated training run with a slow worker shows it as blocked time naming it: the pretrainer's gradient all-reduces on every other rank wait on rank 2, the losses as without the fault", ClusterTrainingSlowWorker),
        ("cluster: a late joiner joins at step N: the pretrainer's other ranks train steps 0 to N-1 over every step's draws, the joiner takes the trained weights and optimizer state, and the run equals the run without the fault (losses and weights within 1e-5)", ClusterTrainingLateJoin),
        ("cluster: a late joiner across the Trainer's epoch end: each step's samples shared among the ranks that take part, so the run equals the run without the fault and every rank ends alike; a source sharded by epoch fails at once naming the cause", ClusterTrainerLateJoin),
        ("cluster: a replay stops at a step bit for bit: the pretrainer's checkpoint, losses and clocks through step N as the whole run's; a run the cluster advisor measured replays from its log's setting alone; the Trainer's weights as one step's", ClusterTrainingReplay),
        ("cluster: cli cluster sim --failures MODEL --seed S injects the faults into the training and lists them with their replay key; an unknown model is named", ClusterCliSimFailures),
        ("cluster: a rank with no rows in a step takes part in the bucketed reduction as the others do (3 ranks, buckets of a set size sent during the backward pass, two sequences a step): the pretrainer's losses and weights as one process's", ClusterBucketsRankWithoutRows),
    ];

    // A classifier of a few layers (several parameters to bucket), without batch norm or dropout.
    private static Sequential OverlapModel(Device device, int width = 32, int layers = 2)
    {
        var init = new Random(5);
        var model = new Sequential { new Linear(2, width, device: device, random: init), new ReLU() };
        for (int i = 1; i < layers; i++)
        {
            model.Add(new Linear(width, width, device: device, random: init));
            model.Add(new ReLU());
        }

        model.Add(new Linear(width, 3, device: device, random: init));
        return model;
    }

    private static void ClusterGradientReadyHook(Device device)
    {
        using var model = OverlapModel(device);
        var parameters = model.Parameters().ToList();
        var input = Tensor.From([.. Enumerable.Range(0, 16).Select(i => (i % 7) * 0.1f)], [8, 2], device);
        var targets = Tensor.From([.. Enumerable.Range(0, 24).Select(i => i % 3 == (i / 3) % 3 ? 1f : 0f)], [8, 3], device);   // one-hot rows
        var seen = new List<(int Parameter, bool Outermost)>();
        using (var scope = new TensorScope())
        {
            var loss = Losses.CrossEntropy(model.Forward(input), targets);
            using (Autograd.OnGradientReady((t, outer) => seen.Add((parameters.IndexOf(t), outer))))
            {
                loss.Backward();
            }
        }

        Check(seen.Count == parameters.Count && seen.Select(s => s.Parameter).Order().SequenceEqual(Enumerable.Range(0, parameters.Count)) && seen.All(s => s.Outermost),
            $"each parameter once, in the outermost pass: {string.Join(", ", seen)}");
        Check(seen.FindIndex(s => s.Parameter >= parameters.Count - 2) < seen.FindIndex(s => s.Parameter < 2),
            $"the last layer's parameters are finished before the first layer's: {string.Join(", ", seen.Select(s => s.Parameter))}");

        // A checkpointed first layer: its parameters are finished inside the checkpoint's backward step, a nested pass.
        var first = (Linear)model[0];
        seen.Clear();
        int outside = 0;
        using (var scope = new TensorScope())
        {
            var hidden = Checkpointing.Checkpoint(x => first.Forward(x), input);
            var rest = model[1].Forward(hidden);
            for (int i = 2; i < model.Count; i++)
            {
                rest = model[i].Forward(rest);
            }

            var loss = Losses.CrossEntropy(rest, targets);
            using (Autograd.OnGradientReady((_, _) => outside++))
            {
            }

            using (Autograd.OnGradientReady((t, outer) => seen.Add((parameters.IndexOf(t), outer))))
            {
                loss.Backward();
            }
        }

        Check(seen.Where(s => s.Parameter < 2).All(s => !s.Outermost) && seen.Count(s => s.Parameter < 2) == 2 && seen.Where(s => s.Parameter >= 2).All(s => s.Outermost) && outside == 0,
            $"a checkpointed block's parameters are not final in its nested pass; the others are; nothing is heard outside a scope: {string.Join(", ", seen)}");
        input.Dispose();
        targets.Dispose();
    }

    private static void ClusterOverlapEqualsAfter(Device device)
    {
        var data = Spirals(50, 3, 1);
        (float[] Curve, float[] Weights) Fit(ClusterRank rank, ClusterOptions options)
        {
            using var model = OverlapModel(rank.Device);
            using var optimizer = new Sgd(model.Parameters(), 0.1f, momentum: 0.9f);
            var trainer = new Trainer(model, optimizer, Losses.CrossEntropy) { Cluster = rank.Collectives, ClusterOptions = options };
            var history = trainer.Fit(new DataLoader(data, 10, shuffle: true, device: rank.Device, seed: 7), 2);
            return ([.. history.Epochs.Select(e => (float)e.Loss)], [.. model.Parameters().SelectMany(p => p.ToArray())]);
        }

        var small = DpOptions() with { BucketBytes = 256 };                 // 64 values: about 20 buckets of the model's 1,251
        var whole = DpOptions() with { BucketBytes = 1L << 20, Overlap = false };
        foreach (var (world, algorithm, after) in new[] { (2, "ring", whole), (3, "naive", whole), (3, "ring", small with { Overlap = false }) })
        {
            var topology = DpTopology(device, world);
            string label = $"{world} ranks {algorithm}, against {(after.BucketBytes == small.BucketBytes ? "the same buckets" : "one bucket")} after the backward pass";
            var (overlapped, seen) = OverlapRun(topology, small with { Collective = algorithm }, rank => Fit(rank, small with { Collective = algorithm }));
            var (plain, plainSeen) = OverlapRun(topology, after with { Collective = algorithm }, rank => Fit(rank, after with { Collective = algorithm }));
            for (int r = 0; r < world; r++)
            {
                Check(overlapped[r].Weights.SequenceEqual(plain[r].Weights) && overlapped[r].Curve.SequenceEqual(plain[r].Curve),
                    $"{label}: rank {r}'s weights and losses bit for bit (largest gap {Gap(overlapped[r].Weights, plain[r].Weights):G3})");
            }

            var syncs = seen.Synchronisations;
            Check(syncs.Count > 0 && Enumerable.Range(0, world).All(r => syncs.Where(s => s.Rank == r).Sum(s => s.Overlapped) > 0) && syncs.All(s => s.Buckets > 1),
                $"{label}: every rank sent buckets during the backward pass ({string.Join(", ", Enumerable.Range(0, world).Select(r => syncs.Where(s => s.Rank == r).Sum(s => s.Overlapped)))} of {syncs.Sum(s => s.Buckets)})");
            Check(plainSeen.Synchronisations.Where(s => s.Backward > TimeSpan.Zero).All(s => s.Overlapped == 0 && s.Lost is { } lost && lost.Contains("overlap is off", StringComparison.Ordinal)),
                $"{label}: with overlap off the telemetry names it: {plainSeen.Synchronisations.FirstOrDefault().Lost}");
        }

        // A step whose gradients cannot go during its backward pass (a recorded step replayed) names why.
        var lost = OverlapRun(DpTopology(device, 2), DpOptions(), rank =>
        {
            using var p = Tensor.Persistent(new float[8], [8], rank.Device, requiresGrad: true);
            using var optimizer = new Sgd([p], 0f);
            using var sync = GradientSyncs.Create(rank.Collectives, optimizer, DpOptions());
            p.Backend.Upload([.. Enumerable.Range(0, 8).Select(i => (float)(i + rank.Rank))], p.GradStorage());
            sync.OverlapLost("a test's recorded step");
            return sync.Reduce(1, 0.5).Tokens;
        }).Metrics.Synchronisations;
        Check(lost.Count == 2 && lost.All(s => s.Lost == "a test's recorded step"), $"the lost overlap is in the step's telemetry: {string.Join("; ", lost.Select(s => s.Lost))}");

        // On a device that records steps as graphs (CUDA), the fine-tuner's replayed steps reduce after the replay and say so.
        if (device.Backend.SupportsGraphs)
        {
            var tokens = File.ReadAllBytes(PretrainData("tokens.bin"));
            var sequences = Enumerable.Range(0, 8).Select(w => Enumerable.Range(0, 33).Select(i => (int)BitConverter.ToUInt16(tokens, ((w * 32) + i) * 2)).ToArray())
                .Select(ids => new TrainingSequence(ids, [.. ids.Select(_ => true)])).ToList();
            var tuning = new FineTuningOptions { Rank = 4, Alpha = 8, LearningRate = 1e-3f, Epochs = 4, MaxLength = 64, BatchTokens = 64, CudaGraphs = true, Seed = 1, WarmupFraction = 0 };
            int replays = 0;
            var (_, graphSeen) = OverlapRun(DpTopology(device, 2), DpOptions(), rank =>
            {
                using var model = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = rank.Device });
                FineTuner.Train(model, sequences, null, tuning with { Cluster = rank.Collectives, ClusterOptions = DpOptions() },
                    trace: line => { if (rank.Rank == 0 && line.Contains("replayed", StringComparison.Ordinal)) { Interlocked.Increment(ref replays); } });
                return 0;
            });
            int named = graphSeen.Synchronisations.Count(s => s.Rank == 0 && s.Lost is { } why && why.Contains("recorded step", StringComparison.Ordinal));
            Check(named == replays, $"each replayed step's telemetry names the lost overlap: {named} of {replays} replayed steps");
            Console.WriteLine($"    graphs: {replays} replayed step(s) on rank 0, each reduced after its replay");
        }
    }

    private static void ClusterOverlapShortens(Device device)
    {
        var data = Spirals(200, 3, 1);
        var topology = NodesOf([1, 1], device.ToString(), "ethernet-1g");
        IReadOnlyList<GradientsSynced> Run(ClusterOptions options)
        {
            var (_, seen) = OverlapRun(topology, options, rank =>
            {
                using var model = OverlapModel(rank.Device, width: 256, layers: 3);
                using var optimizer = new Sgd(model.Parameters(), 0.05f);
                var trainer = new Trainer(model, optimizer, Losses.CrossEntropy) { Cluster = rank.Collectives, ClusterOptions = options };
                trainer.Fit(new DataLoader(data, 128, shuffle: true, device: rank.Device, seed: 3), 2);
                return 0;
            });
            return [.. seen.Synchronisations.Where(s => s.Rank == 0 && s.Backward > TimeSpan.Zero)];   // the steps that trained (an epoch's end adds none)
        }

        var slow = DpOptions() with { Collective = "ring", BucketBytes = 64 << 10 };
        var overlapped = Run(slow);
        var after = Run(slow with { Overlap = false });
        // The first, cold step aside (rule 76).
        double Exposed(IReadOnlyList<GradientsSynced> steps) => steps.Skip(1).Sum(s => s.Exposed.TotalMilliseconds) / Math.Max(1, steps.Count - 1);
        double Network(IReadOnlyList<GradientsSynced> steps) => steps.Skip(1).Sum(s => s.Network.TotalMilliseconds) / Math.Max(1, steps.Count - 1);
        double with = Exposed(overlapped), without = Exposed(after);
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    overlap on ethernet-1g, 2 ranks, {overlapped[0].Bytes / 1024.0:F0} KiB of gradients in {overlapped[0].Buckets} buckets: exposed {with:F3} ms a step with overlap, {without:F3} ms without "
            + $"(network {Network(overlapped):F3} and {Network(after):F3} ms simulated, backward {overlapped.Skip(1).Average(s => s.Backward.TotalMilliseconds):F3} ms measured), {(1 - (with / without)) * 100:F1}% shorter"));
        Check(overlapped.Count >= 4 && overlapped.Skip(1).All(s => s.Overlapped > 0), $"buckets went during the backward pass on every warm step: {string.Join(", ", overlapped.Select(s => s.Overlapped))}");
        Check(with < without, $"overlap shortens the simulated step's exposed network time: {with:F3} ms against {without:F3} ms");

        // Unset, the bucket size is the first probe on the first step, then the measured size on every later one, and the
        // same on every run (on a simulated transport it depends on the virtual clock alone).
        var measured = Run(DpOptions() with { Collective = "ring" });
        var again = Run(DpOptions() with { Collective = "ring" });
        long chosen = measured[^1].BucketBytes;
        Check(measured.Skip(1).All(s => s.BucketBytes == chosen) && chosen <= measured[0].Bytes && measured.Select(s => s.BucketBytes).SequenceEqual(again.Select(s => s.BucketBytes)),
            $"the bucket size measured once and kept, the same on a rerun: {string.Join(", ", measured.Select(s => s.BucketBytes))} and {string.Join(", ", again.Select(s => s.BucketBytes))}");
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"    measured bucket size: {chosen / 1024.0:F1} KiB of {measured[0].Bytes / 1024.0:F0} KiB ({measured[^1].Buckets} buckets)"));
    }

    private static void ClusterCompressions(Device device)
    {
        if (FirstRun(nameof(ClusterCompressions)))
        {
            Check(new[] { "none", "bf16", "top-k", "int8", "fp8" }.All(n => GradientCompressions.Find(n) is not null && GradientCompressions.Origin(n) == Overrides.Library)
                  && GradientCompressions.Describe().Contains("top-k:", StringComparison.Ordinal) && GradientCompressions.Describe().Contains("int8:", StringComparison.Ordinal)
                  && GradientCompressions.Describe().Contains("fp8:", StringComparison.Ordinal), $"the built-in compressions: {GradientCompressions.Describe()}");
            ThrowsNaming<NotSupportedException>(() => GradientCompressions.Get("zip"), "an unknown compression", "is not registered", "GradientCompressions.Register");
            Check(GradientSyncs.Find("local-sgd") is not null && GradientSyncs.Origin("local-sgd") == Overrides.Library, $"Idrak.Network registers local-sgd: {GradientSyncs.Describe()}");
        }

        // Two ranks, one parameter of 10 values, 20% sent: each rank's two largest by magnitude, summed in rank order.
        float[][] mine = [[10f, -1f, 2f, 0.5f, -7f, 0f, 3f, 0f, 0f, 1f], [0f, 4f, 0f, -6f, 0f, 9f, 0f, 2.5f, 0f, 0f]];
        var options = DpOptions() with { Compression = "top-k", TopK = 0.2 };
        var results = DpRun(DpTopology(device, 2), options, rank =>
        {
            using var p = Tensor.Persistent(new float[10], [10], rank.Device, requiresGrad: true);
            using var compressor = GradientCompressions.Create(rank.Collectives, [p], options);
            GradientPiece[] pieces = [new(0, 0, 10, 0)];
            var first = (float[])mine[rank.Rank].Clone();
            compressor.Reduce(first, 2f, pieces);                               // weighted by 2 (its rows)
            var second = new float[10];
            compressor.Reduce(second, 1f, pieces);                              // the residual's two largest go next
            ThrowsNaming<ArgumentException>(() => GradientCompressions.Create(rank.Collectives, [p], DpOptions() with { Compression = "top-k" }), "top-k without a share", "IDRAK_CLUSTER_TOPK");
            return (First: first, Second: second);
        });

        float[] expectedFirst = [20f, 0f, 0f, -12f, -14f, 18f, 0f, 0f, 0f, 0f];   // rank 0: 10 and -7; rank 1: 9 and -6; times 2
        float[] expectedSecond = [0f, 4f, 2f, 0f, 0f, 0f, 3f, 2.5f, 0f, 0f];     // the residuals' two largest: rank 0's 3 and 2, rank 1's 4 and 2.5
        foreach (var (result, r) in results.Select((x, r) => (x, r)))
        {
            Check(result.First.SequenceEqual(expectedFirst), $"rank {r}: the step's sum of each rank's two largest, weighted: [{string.Join(", ", result.First)}]");
            Check(result.Second.SequenceEqual(expectedSecond), $"rank {r}: the next step sends what was left, largest first (error feedback): [{string.Join(", ", result.Second)}]");
        }
    }

    // E4M3 by its definition, written apart from the library: (-1)^s · 2^(e-7) · (1 + m/8), subnormal 2^-6 · m/8, S.1111.111 NaN.
    private static double E4M3Value(int code)
    {
        int e = (code >> 3) & 15, m = code & 7;
        double magnitude = e == 15 && m == 7 ? double.NaN : e == 0 ? Math.Pow(2, -6) * (m / 8.0) : Math.Pow(2, e - 7) * (1 + (m / 8.0));
        return (code & 0x80) != 0 ? -magnitude : magnitude;
    }

    // The nearest finite E4M3 code to x by a search of every code, ties to the even code; ±448 beyond it; NaN for NaN.
    private static int E4M3Nearest(float x)
    {
        int sign = float.IsNegative(x) ? 0x80 : 0;
        if (float.IsNaN(x))
        {
            return sign | 0x7F;
        }

        double a = Math.Abs((double)x);
        if (a >= 448)
        {
            return sign | 0x7E;
        }

        int best = 0;
        double distance = double.PositiveInfinity;
        for (int c = 0; c <= 0x7E; c++)
        {
            double d = Math.Abs(a - E4M3Value(c));
            if (d < distance || (d == distance && (c & 1) == 0))
            {
                (best, distance) = (c, d);
            }
        }

        return sign | best;
    }

    private static void ClusterFloat8Codes(Device device)
    {
        if (!FirstRun(nameof(ClusterFloat8Codes)))
        {
            return;
        }

        // Every code's value, against the definition and at the known points.
        var wrong = Enumerable.Range(0, 256).Where(c => !(double.IsNaN(E4M3Value(c)) ? float.IsNaN(Float8E4M3.Decode((byte)c)) : Float8E4M3.Decode((byte)c) == E4M3Value(c))).ToList();
        Check(wrong.Count == 0, $"codes whose value is not the format's: {string.Join(", ", wrong.Select(c => $"0x{c:X2} = {Float8E4M3.Decode((byte)c)}"))}");
        Check(Float8E4M3.Decode(0x00) == 0f && BitConverter.SingleToUInt32Bits(Float8E4M3.Decode(0x80)) == 0x80000000u && Float8E4M3.Decode(0x01) == MathF.ScaleB(1f, -9)
              && Float8E4M3.Decode(0x07) == 7 * MathF.ScaleB(1f, -9) && Float8E4M3.Decode(0x08) == MathF.ScaleB(1f, -6) && Float8E4M3.Decode(0x38) == 1f && Float8E4M3.Decode(0x40) == 2f
              && Float8E4M3.Decode(0x7E) == 448f && Float8E4M3.Decode(0xFE) == -448f && float.IsNaN(Float8E4M3.Decode(0x7F)) && float.IsNaN(Float8E4M3.Decode(0xFF)) && Float8E4M3.MaxValue == 448f,
            "E4M3: 0 and -0, the smallest subnormal 2^-9, the largest 7·2^-9, the smallest normal 2^-6, 1, 2, ±448, the NaN codes");

        // Every finite code back to itself.
        var lost = Enumerable.Range(0, 256).Where(c => (c & 0x7F) != 0x7F && Float8E4M3.Encode(Float8E4M3.Decode((byte)c)) != c).ToList();
        Check(lost.Count == 0, $"finite codes that do not come back: {string.Join(", ", lost.Select(c => $"0x{c:X2} -> 0x{Float8E4M3.Encode(Float8E4M3.Decode((byte)c)):X2}"))}");

        // Each midpoint goes to the even code, its neighbours one float away to the nearer code; both signs.
        var ties = new List<string>();
        for (int c = 0; c < 0x7E; c++)
        {
            float mid = (float)((E4M3Value(c) + E4M3Value(c + 1)) / 2);         // exact: a few bits
            foreach (int sign in new[] { 1, -1 })
            {
                int s = sign < 0 ? 0x80 : 0;
                int even = s | ((c & 1) == 0 ? c : c + 1);
                if (Float8E4M3.Encode(sign * mid) != even || Float8E4M3.Encode(sign * MathF.BitDecrement(mid)) != (s | c) || Float8E4M3.Encode(sign * MathF.BitIncrement(mid)) != (s | (c + 1)))
                {
                    ties.Add($"0x{c:X2}/0x{c + 1:X2} sign {sign}: {Float8E4M3.Encode(sign * mid):X2}, below {Float8E4M3.Encode(sign * MathF.BitDecrement(mid)):X2}, above {Float8E4M3.Encode(sign * MathF.BitIncrement(mid)):X2}");
                }
            }
        }

        Check(ties.Count == 0, $"midpoints not to the even code, or neighbours not to the nearer: {string.Join("; ", ties.Take(8))}");

        // Values over every size (float32 subnormals to beyond 448) against the search of every code.
        var random = new Random(37);
        var misses = new List<string>();
        for (int i = 0; i < 200_000; i++)
        {
            float x = i % 4 == 0 ? BitConverter.Int32BitsToSingle(random.Next()) % 1024f
                : (float)((random.NextDouble() < 0.5 ? -1 : 1) * Math.Pow(2, (random.NextDouble() * 22) - 12) * (1 + random.NextDouble()));
            if (Float8E4M3.Encode(x) != E4M3Nearest(x) && misses.Count < 8)
            {
                misses.Add($"{x:R} -> 0x{Float8E4M3.Encode(x):X2}, the nearest is 0x{E4M3Nearest(x):X2}");
            }
        }

        Check(misses.Count == 0, $"values not encoded to the nearest code: {string.Join("; ", misses)}");
        Check(Float8E4M3.Encode(448f) == 0x7E && Float8E4M3.Encode(449f) == 0x7E && Float8E4M3.Encode(1e30f) == 0x7E && Float8E4M3.Encode(float.PositiveInfinity) == 0x7E
              && Float8E4M3.Encode(float.NegativeInfinity) == 0xFE && Float8E4M3.Encode(-500f) == 0xFE && float.IsNaN(Float8E4M3.Decode(Float8E4M3.Encode(float.NaN)))
              && Float8E4M3.Encode(MathF.ScaleB(1f, -10)) == 0x00 && Float8E4M3.Encode(MathF.BitIncrement(MathF.ScaleB(1f, -10))) == 0x01 && Float8E4M3.Encode(-1e-30f) == 0x80
              && Float8E4M3.Encode(float.Epsilon) == 0x00,
            "E4M3: beyond 448 and infinity saturate to ±448, NaN stays NaN, half the smallest subnormal ties to 0, a tiny negative is -0");
    }

    // By the definition, written apart from the library: finite values as codes of one scale, largest / 127 (or 448), each
    // code the nearest to v · (127 / largest) (or 448 / largest), ties to even; returns the scale (0 when every value is
    // zero or too small to scale), writes the decoded values (code · scale) and leaves in `values` what they do not carry.
    private static float QuantizeReference(Span<float> values, Span<float> decoded, bool fp8)
    {
        float top = fp8 ? 448f : 127f, largest = 0f;
        foreach (float x in values)
        {
            largest = Math.Max(largest, MathF.Abs(x));
        }

        float inverse = top / largest, scale = largest / top;
        decoded.Clear();
        if (largest == 0f || !float.IsFinite(inverse))
        {
            return 0f;
        }

        for (int i = 0; i < values.Length; i++)
        {
            float code = fp8 ? (float)E4M3Value(E4M3Nearest(values[i] * inverse)) : MathF.Round(Math.Clamp(values[i] * inverse, -127f, 127f), MidpointRounding.ToEven);
            decoded[i] = code * scale;
            values[i] -= decoded[i];
        }

        return scale;
    }

    // One step of an int8 or fp8 compression over every rank, by its definition: each rank's values weighted, its residual
    // added. Gathered: each rank's whole bucket at one scale, every rank's decoded values summed in rank order. Exchanged:
    // the bucket cut in one piece per rank (as a ring cuts it), each rank's piece at a scale of its own; owner p sums every
    // rank's decoded piece p in rank order, rounds the sum again at its own scale and keeps that rounding's rest too. A rank
    // with no rows sends zeros and keeps its residual; a NaN or infinity makes every value NaN and drops the residual of the
    // rank that had it (and the owners' second rest).
    private static float[] QuantizedStepReference(float[][] values, float[] weights, float[][] residuals, bool fp8, bool feedback, bool exchanged)
    {
        int world = values.Length, n = values[0].Length;
        int Start(int p) => (p * (n / world)) + Math.Min(p, n % world);
        int Size(int p) => (n / world) + (p < n % world ? 1 : 0);
        var decoded = new float[world][];
        for (int r = 0; r < world; r++)
        {
            decoded[r] = new float[n];
            if (weights[r] == 0f)
            {
                continue;
            }

            var v = new float[n];
            for (int i = 0; i < n; i++)
            {
                v[i] = (values[r][i] * weights[r]) + (feedback ? residuals[r][i] : 0f);
            }

            if (v.Any(x => !float.IsFinite(x)))
            {
                Array.Clear(residuals[r]);
                Array.Fill(decoded[r], float.NaN);
                continue;
            }

            for (int p = 0; p < (exchanged ? world : 1); p++)
            {
                int start = exchanged ? Start(p) : 0, size = exchanged ? Size(p) : n;
                QuantizeReference(v.AsSpan(start, size), decoded[r].AsSpan(start, size), fp8);
            }

            if (feedback)
            {
                v.CopyTo(residuals[r]);
            }
        }

        var result = new float[n];
        if (!exchanged)
        {
            for (int r = 0; r < world; r++)
            {
                for (int i = 0; i < n; i++)
                {
                    result[i] += decoded[r][i];
                }
            }

            return result;
        }

        for (int p = 0; p < world; p++)
        {
            int start = Start(p), size = Size(p);
            var sum = new float[size];
            for (int r = 0; r < world; r++)
            {
                for (int i = 0; i < size; i++)
                {
                    sum[i] += decoded[r][start + i];
                }
            }

            var second = new float[size];
            if (sum.All(float.IsFinite))
            {
                QuantizeReference(sum, second, fp8);
                for (int i = 0; feedback && i < size; i++)
                {
                    residuals[p][start + i] += sum[i];
                }
            }
            else
            {
                Array.Fill(second, float.NaN);
            }

            for (int i = 0; i < size; i++)
            {
                result[start + i] += second[i];
            }
        }

        return result;
    }

    // Collectives that take host values but exchange no blocks (as a plug-in's may), over a rank's own: the int8 and fp8
    // compressions all-gather every rank's codes over them.
    private sealed class GatherOnly(ICollectives inner) : ICollectives, IHostValueCollectives
    {
        private readonly IHostValueCollectives _host = (IHostValueCollectives)inner;

        public string Name => inner.Name;

        public int Rank => inner.Rank;

        public int WorldSize => inner.WorldSize;

        public Device Device => inner.Device;

        public bool SupportsHostValues => _host.SupportsHostValues;

        public TimeSpan SimulatedTime => _host.SimulatedTime;

        public void AllReduce(Storage buffer, int count, ReduceOp op, WireType wire = WireType.Float32) => inner.AllReduce(buffer, count, op, wire);

        public void AllGather(Storage input, int count, Storage output) => inner.AllGather(input, count, output);

        public void ReduceScatter(Storage input, int count, Storage output, ReduceOp op) => inner.ReduceScatter(input, count, output, op);

        public void Broadcast(Storage buffer, int count, int root) => inner.Broadcast(buffer, count, root);

        public void Barrier() => inner.Barrier();

        public bool TryAllReduce(Span<float> values, ReduceOp op, WireType wire) => _host.TryAllReduce(values, op, wire);

        public bool TryAllGather(Span<float> all, int count) => _host.TryAllGather(all, count);

        public bool TryBroadcast(Span<float> values, int root) => _host.TryBroadcast(values, root);

        public void Dispose()
        {
            // The rank's collectives are the run's.
        }
    }

    private static void ClusterQuantizedCompressions(Device device)
    {
        // Rank 0's first four values make the bytes FF FF FF 7F at scale 1 (-1, -1, -1, 127) on the gathered path: a float
        // whose bits are a NaN, which the float32 wire carries unchanged. 1,000 values over five sizes take the vector
        // paths and their tails; on 3 ranks the pieces are uneven (14, 13, 13 and 334, 333, 333).
        float[][] rows = [[-1f, -1f, -1f, 127f, -63.5f, 0.5f, 2.25f, 0f, 10f, -10f], [254f, 3f, -5f, 1f, 0f, 0f, 0f, -254f, 7f, 0.5f],
            [0.001f, -2f, 0f, 3.5f, -127f, 64.25f, 1e-3f, -0.5f, 9f, 100f]];
        float[][] small = [.. rows.Select(r => r.Concat(r.Select(x => x * 0.5f)).Concat(r.Select(x => -x)).Concat(r.Select(x => x * 0.01f)).ToArray())];
        var random = new Random(23);
        float[][] large = [.. Enumerable.Range(0, 3).Select(_ => Enumerable.Range(0, 1000).Select(_ => (float)((random.NextDouble() - 0.5) * Math.Pow(10, random.Next(-4, 1)))).ToArray())];
        float[][] withNaN = [.. small.Select(v => (float[])v.Clone())];
        withNaN[1][3] = float.NaN;

        // The steps: (parameter, each rank's values or null for zeros, each rank's weight, forget the residual after it).
        var steps = new (int Parameter, float[][]? Values, float[] Weights, bool Forget)[]
        {
            (0, small, [1f, 1f, 1f], false),
            (0, null, [1f, 1f, 1f], false),                                      // the residual goes
            (1, large, [2f, 2f, 2f], false),
            (1, null, [1f, 1f, 1f], false),
            (0, small, [2f, 0f, 2f], false),                                     // rank 1 has no rows: adds nothing, keeps its residual
            (0, null, [1f, 1f, 1f], false),
            (0, withNaN, [1f, 1f, 1f], false),                                   // the NaN travels; rank 1's residual dropped
            (0, null, [1f, 1f, 1f], false),
            (0, small, [1f, 1f, 1f], true),                                      // the bucket went again whole: the residual forgotten
            (0, null, [1f, 1f, 1f], false),
        };

        // The way a bucket goes, as the compressor picks it: exchanged where the collectives offer it and two blocks of a
        // piece's codes (a scale, then the codes four to a float) are fewer floats than one block of the whole bucket's.
        static int Floats(int codes) => 1 + ((codes + 3) / 4);
        static bool Exchanges(int n, int world, bool offered) => offered && world > 1 && 2 * Floats((n + world - 1) / world) < Floats(n);
        foreach (var (world, offered) in new[] { (2, true), (3, true), (3, false) })
        {
            foreach (var (name, fp8) in new[] { ("int8", false), ("fp8", true) })
            {
                foreach (bool feedback in new[] { true, false })
                {
                    var options = DpOptions() with { Compression = name, ErrorFeedback = feedback };
                    var results = DpRun(DpTopology(device, world), options, rank =>
                    {
                        using var p = Tensor.Persistent(new float[40], [40], rank.Device, requiresGrad: true);
                        using var q = Tensor.Persistent(new float[1000], [1000], rank.Device, requiresGrad: true);
                        using var collectives = offered ? null : new GatherOnly(rank.Collectives);
                        using var compressor = GradientCompressions.Create(collectives ?? rank.Collectives, [p, q], options);
                        Check(compressor is QuantizedCompressor c && c.Exchanges(40) == Exchanges(40, world, offered) && c.Exchanges(1000) == Exchanges(1000, world, offered)
                              && c.Exchanges(4) == false,
                            $"{name} on {world} ranks{(offered ? "" : " without an all-to-all")}: buckets of 40 and 1,000 values {(Exchanges(40, world, offered) ? "exchanged" : "all-gathered")}, the way of fewer bytes; 4 values all-gathered");
                        var outputs = new List<float[]>();
                        foreach (var step in steps)
                        {
                            int n = step.Parameter == 0 ? 40 : 1000;
                            var values = step.Values is { } given ? (float[])given[rank.Rank].Clone() : new float[n];
                            GradientPiece[] pieces = [new(step.Parameter, 0, n, 0)];
                            compressor.Reduce(values, step.Weights[rank.Rank], pieces);
                            if (step.Forget)
                            {
                                compressor.Forget(pieces);
                            }

                            outputs.Add(values);
                        }

                        return outputs;
                    });

                    // The reference: every rank's step by the definition.
                    float[][][] residuals = [.. Enumerable.Range(0, world).Select(_ => new[] { new float[40], new float[1000] })];
                    for (int k = 0; k < steps.Length; k++)
                    {
                        var step = steps[k];
                        int n = step.Parameter == 0 ? 40 : 1000;
                        bool exchanged = Exchanges(n, world, offered);
                        float[][] given = [.. Enumerable.Range(0, world).Select(r => step.Values?[r] ?? new float[n])];
                        var sum = QuantizedStepReference(given, step.Weights[..world], [.. residuals.Select(r => r[step.Parameter])], fp8, feedback, exchanged);
                        if (step.Forget)
                        {
                            foreach (var r in residuals)
                            {
                                Array.Clear(r[step.Parameter]);
                            }
                        }

                        string what = $"{name} on {world} ranks{(offered ? "" : " without an all-to-all")} ({(exchanged ? "exchanged" : "all-gathered")}){(feedback ? "" : " without error feedback")}, step {k + 1}";
                        for (int r = 0; r < world; r++)
                        {
                            var got = results[r][k];
                            int bad = Enumerable.Range(0, n).FirstOrDefault(i => !(float.IsNaN(sum[i]) ? float.IsNaN(got[i]) : BitConverter.SingleToInt32Bits(got[i]) == BitConverter.SingleToInt32Bits(sum[i])), -1);
                            Check(bad < 0, bad < 0 ? what : $"{what}, rank {r}: value {bad} is {got[bad]:R}, the definition gives {sum[bad]:R}");
                        }
                    }

                    string run = $"{name} on {world} ranks{(offered ? "" : " without an all-to-all")}";
                    if (feedback)
                    {
                        Check(results[0][1].Any(x => x != 0f) && results[0][9].All(x => x == 0f), $"{run}: the step after sends the rounding error left over; after a forget, nothing");
                    }
                    else
                    {
                        Check(results[0][1].All(x => x == 0f), $"{run} without error feedback: nothing kept for the next step");
                    }
                }
            }
        }

        // Bytes on the wire a rank sends for a bucket of 65,536 values (a ring): float32's all-reduce, 2 (W - 1) / W · 4n,
        // against one byte a value and a scale: on 2 ranks every rank's codes all-gathered, W - 1 blocks of the whole
        // bucket's; beyond, all-to-all then all-gathered, 2 (W - 1) blocks of a piece's.
        const int Values = 65536;
        (long Bytes, float[][] Results) Run(int world, string compression)
        {
            var options = DpOptions() with { Compression = compression };
            var (results, metrics) = OverlapRun(DpTopology(device, world), options, rank =>
            {
                using var p = Tensor.Persistent(new float[Values], [Values], rank.Device, requiresGrad: true);
                using var compressor = GradientCompressions.Create(rank.Collectives, [p], options);
                var values = Enumerable.Range(0, Values).Select(i => MathF.Sin(i + rank.Rank)).ToArray();
                compressor.Reduce(values, 1f, [new(0, 0, Values, 0)]);
                return values;
            });
            return (metrics.Collectives.Where(c => c.Rank == 0 && c.Operation is "all-reduce" or "all-gather" or "all-to-all").Sum(c => c.BytesSent), results);
        }

        foreach (int world in new[] { 2, 4, 8 })
        {
            var plain = Run(world, "none");
            var int8 = Run(world, "int8");
            var fp8Run = Run(world, "fp8");
            long expected = 4L * (world - 1) * (Exchanges(Values, world, true) ? 2 * Floats(Values / world) : Floats(Values));
            Console.WriteLine($"    a bucket of {Values:N0} values on {world} ranks: float32 {plain.Bytes:N0} B sent by rank 0, int8 {int8.Bytes:N0} B, fp8 {fp8Run.Bytes:N0} B ({(double)plain.Bytes / int8.Bytes:F3}x fewer)");
            Check(plain.Bytes == 2L * (world - 1) * 4 * (Values / world) && int8.Bytes == expected && fp8Run.Bytes == expected,
                $"{world} ranks: one byte a value and its scale (the bucket's to each other rank on 2, a piece's twice beyond) against float32's piece twice: int8 {int8.Bytes}, fp8 {fp8Run.Bytes}, expected {expected}, float32 {plain.Bytes}");
            Check(plain.Bytes >= 3.99 * int8.Bytes, $"{world} ranks: 4x fewer bytes on the wire than float32: {plain.Bytes:N0} against {int8.Bytes:N0}");
            Check(int8.Results.All(r => r.AsSpan().SequenceEqual(int8.Results[0])) && fp8Run.Results.All(r => r.AsSpan().SequenceEqual(fp8Run.Results[0])),
                $"{world} ranks: every rank ends with the same values, bit for bit");
        }
    }

    private static void ClusterAllToAll(Device device)
    {
        // Rank r's block for rank j: float bits that encode (r, j, i), half of them NaN patterns (a signalling NaN's too),
        // so a value that is not carried bit for bit shows.
        static float Sent(int from, int to, int i) => BitConverter.Int32BitsToSingle(i % 2 == 0 ? 0x7F800001 + (from << 16) + (to << 8) + i : (from * 1000) + (to * 100) + i);
        const int Count = 37;
        foreach (var (topology, algorithm) in new[] { (DpTopology(device, 1), "ring"), (DpTopology(device, 3), "ring"), (DpTopology(device, 3), "naive"), (NodesOf([2, 2], device.ToString()), "hierarchical") })
        {
            int world = topology.WorldSize;
            var options = DpOptions() with { Collective = algorithm };
            var (results, metrics) = OverlapRun(topology, options, rank =>
            {
                var exchange = (IAllToAllHostValueCollectives)rank.Collectives;
                var send = new float[world * Count];
                var receive = new float[world * Count];
                for (int j = 0; j < world; j++)
                {
                    for (int i = 0; i < Count; i++)
                    {
                        send[(j * Count) + i] = Sent(rank.Rank, j, i);
                    }
                }

                bool done = exchange.SupportsAllToAll && exchange.TryAllToAll(send, receive, Count);
                return (Done: done, Received: receive);
            });

            for (int r = 0; r < world; r++)
            {
                var (done, received) = results[r];
                int wrong = Enumerable.Range(0, world * Count)
                    .Count(k => BitConverter.SingleToInt32Bits(received[k]) != BitConverter.SingleToInt32Bits(Sent(k / Count, r, k % Count)));
                Check(done && wrong == 0, $"{algorithm} on {world} ranks, rank {r}: every rank's block for it, bit for bit ({wrong} values differ)");
                var sent = metrics.Collectives.Where(c => c.Rank == r && c.Operation == "all-to-all").ToList();
                Check(sent.Count == 1 && sent[0].BytesSent == 4L * Count * (world - 1) && sent[0].BytesReceived == 4L * Count * (world - 1),
                    $"{algorithm} on {world} ranks, rank {r}: one all-to-all published, its block to each other rank: {string.Join(", ", sent.Select(c => $"{c.BytesSent} B sent, {c.BytesReceived} B received"))}");
            }
        }
    }

    // The Spirals sample's model and data (as ClusterSpirals), on two ranks, with the given cluster options.
    private static (float Loss, double Accuracy, long Bytes) SpiralsOnTwo(Device device, ClusterOptions options, int epochs)
    {
        const int Classes = 3, PerClass = 300;
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
        var (results, metrics) = OverlapRun(DpTopology(device, 2), options, rank =>
        {
            var init = new Random(3);
            using var model = new Sequential
            {
                new Linear(2, 128, device: rank.Device, random: init), new BatchNorm(128, device: rank.Device), new ReLU(),
                new Linear(128, 64, device: rank.Device, random: init), new BatchNorm(64, device: rank.Device), new ReLU(),
                new Linear(64, Classes, device: rank.Device, random: init),
            };
            using var optimizer = new AdamW(model.Parameters(), learningRate: 0.01f, weightDecay: 1e-4f);
            var trainer = new Trainer(model, optimizer, (logits, targets) => Losses.CrossEntropy(logits, targets, labelSmoothing: 0.05f))
            {
                Metrics = { Metric.Accuracy },
                Scheduler = new CosineAnnealing(optimizer, totalEpochs: epochs, warmupEpochs: 5),
                Cluster = rank.Collectives,
                ClusterOptions = options,
            };
            var history = trainer.Fit(new DataLoader(train, 32, shuffle: true, device: rank.Device, seed: 4), epochs, validation: new DataLoader(test, 512, device: rank.Device));
            return ((float)history.Epochs[^1].Loss, history.Epochs[^1].ValidationMetrics!["accuracy"]);
        });

        return (results[0].Item1, results[0].Item2, metrics.Collectives.Where(c => c.Rank == 0).Sum(c => c.BytesSent));
    }

    private static void ClusterTopKSpirals(Device device)
    {
        const int Epochs = 20;
        var full = SpiralsOnTwo(device, DpOptions(), Epochs);
        // One bucket of every gradient value, so top-k picks over the whole gradient whatever size the sizing would measure.
        var topK = SpiralsOnTwo(device, DpOptions() with { Compression = "top-k", TopK = 0.1, BucketBytes = 1L << 20 }, Epochs);
        var plain = SpiralsOnTwo(device, DpOptions() with { Compression = "top-k", TopK = 0.1, ErrorFeedback = false, BucketBytes = 1L << 20 }, Epochs);
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    Spirals, 2 ranks, {Epochs} epochs: full gradient loss {full.Loss:F4} (accuracy {full.Accuracy:P1}, {full.Bytes / 1024.0:F0} KiB sent by rank 0); top-k 10% with error feedback {topK.Loss:F4} "
            + $"({topK.Accuracy:P1}, {topK.Bytes / 1024.0:F0} KiB), gap {topK.Loss - full.Loss:F4}; without error feedback {plain.Loss:F4} ({plain.Accuracy:P1})"));
        Check(topK.Loss - full.Loss <= TopKLossGap, $"top-k with error feedback within {TopKLossGap} of the full-gradient loss: {topK.Loss:F4} against {full.Loss:F4}");
        Check(topK.Bytes < full.Bytes / 2, $"top-k sends under half the bytes: {topK.Bytes:N0} against {full.Bytes:N0}");
    }

    private static void ClusterLocalSgd(Device device)
    {
        var data = Spirals(50, 3, 1);
        (float[] Curve, float[] Weights) Fit(Device on, int batch, ICollectives? cluster, ClusterOptions? options)
        {
            using var model = DpClassifier(on);
            using var optimizer = new Sgd(model.Parameters(), 0.1f);
            var trainer = new Trainer(model, optimizer, Losses.CrossEntropy) { Cluster = cluster, ClusterOptions = options };
            var history = trainer.Fit(new DataLoader(data, batch, shuffle: true, device: on, seed: 7), 3);
            return ([.. history.Epochs.Select(e => (float)e.Loss)], [.. model.Parameters().SelectMany(p => p.ToArray())]);
        }

        var alone = Fit(device, 20, null, null);
        var one = DpOptions() with { LocalSgdEvery = 1 };
        var (ranks, seen) = OverlapRun(DpTopology(device, 2), one, rank => Fit(rank.Device, 10, rank.Collectives, one));
        Check(ranks[0].Weights.SequenceEqual(ranks[1].Weights), "k = 1: the weights alike on every rank after each average");
        AssertClose(alone.Curve, ranks[0].Curve, 1e-5f, "local SGD at k = 1: the epochs' losses against one process with batch 20");
        AssertClose(alone.Weights, ranks[0].Weights, 1e-5f, "local SGD at k = 1: the weights against one process");

        var four = DpOptions() with { LocalSgdEvery = 4 };
        var (fourRanks, fourSeen) = OverlapRun(DpTopology(device, 2), four, rank => Fit(rank.Device, 10, rank.Collectives, four));
        int Averages(MetricsRecorder m) => m.Synchronisations.Count(s => s.Rank == 0 && s.Buckets > 0);
        int averagesOne = Averages(seen), averagesFour = Averages(fourSeen);
        long Steps(MetricsRecorder m) => m.Synchronisations.Count(s => s.Rank == 0 && s.Buckets == 0);
        Check(seen.Synchronisations.All(s => s.Sync == "local-sgd") && averagesOne == Steps(seen) && averagesOne == 4 * averagesFour && averagesFour > 0,
            $"a weight average every step at k = 1 ({averagesOne} in {Steps(seen)} steps), k times fewer at k = 4 ({averagesFour} in {Steps(fourSeen)})");
        Check(fourRanks[0].Weights.SequenceEqual(fourRanks[1].Weights), "k = 4: the last step averaged, the weights alike on every rank");
        Console.WriteLine($"    local SGD: {averagesOne} weight averages at k = 1, {averagesFour} at k = 4, over {Steps(seen)} steps");
    }

    // A pretraining run's losses (this rank's), its model's weights in memory and its last checkpoint.
    private static (float[] Losses, float[] Weights, string? Last) ClusterPretrain(Device device, PretrainingOptions options, string folder)
    {
        using var loaded = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = device });
        var result = Pretrainer.Train(loaded, options, folder);
        return ([.. result.Losses], [.. loaded.Network.Parameters().SelectMany(p => p.ToArray())], result.LastCheckpoint);
    }

    private static void ClusterTrainingSlowWorker(Device device)
    {
        string folder = TempFolder("cluster-slow-training");
        using var slow = Scripted("test-slow-training", new Fault(FaultKind.SlowWorker, 0, [2], TimeSpan.Zero, 8));
        try
        {
            string corpus = PretrainData("tokens.bin");
            (float[][] Losses, IReadOnlyList<CollectiveCompleted> Reduces) Run(string failures, string name)
            {
                // Buckets of a set size: the same collectives with and without the fault (a measured size follows the slow link).
                var options = DpOptions() with { Collective = "ring", BucketBytes = 64 << 10, Simulation = new SimulationOptions { Failures = failures, Seed = 5 } };
                var (results, seen) = OverlapRun(DpTopology(device, 3), options, rank =>
                    ClusterPretrain(rank.Device, DpPretraining(corpus, 3, rank.Collectives, options), Path.Combine(folder, name)).Losses);
                return (results, [.. seen.Collectives.Where(c => c.Operation == "all-reduce")]);
            }

            var (plainLosses, plain) = Run("none", "plain");
            var (slowLosses, slowed) = Run("test-slow-training", "slow");
            Check(plainLosses.Zip(slowLosses).All(p => p.First.SequenceEqual(p.Second)), "a slow worker changes the time, not the values: every rank's losses bit for bit");
            foreach (int r in new[] { 0, 1 })
            {
                var mine = slowed.Where(c => c.Rank == r).ToList();
                var before = plain.Where(c => c.Rank == r).ToList();
                Check(mine.Count > 0 && mine.Count == before.Count && mine.All(c => c.SlowestRank == 2),
                    $"rank {r}: every gradient all-reduce names rank 2 as the slowest: {string.Join(", ", mine.Select(c => c.SlowestRank))}");
                double blocked = mine.Sum(c => c.Blocked.TotalMilliseconds), was = before.Sum(c => c.Blocked.TotalMilliseconds);
                Check(blocked > was, $"rank {r}: blocked {blocked:F3} ms on the slow worker, {was:F3} ms without: "
                    + string.Join("; ", mine.Zip(before).Take(12).Select(p => string.Create(CultureInfo.InvariantCulture,
                        $"{p.First.BytesSent} B sim {p.First.Simulated.TotalMicroseconds:F1}/{p.Second.Simulated.TotalMicroseconds:F1} us blocked {p.First.Blocked.TotalMicroseconds:F1}/{p.Second.Blocked.TotalMicroseconds:F1}"))));
                Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"    rank {r}: blocked {blocked:F3} ms of simulated time with rank 2 slow (x8), {was:F3} ms without"));
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void ClusterTrainingLateJoin(Device device)
    {
        string folder = TempFolder("cluster-late-training");
        using var late = Scripted("test-late-training", new Fault(FaultKind.LateJoin, 0, [2], TimeSpan.Zero, 2));
        try
        {
            string corpus = PretrainData("tokens.bin");
            var events = new NetworkEvents();
            (float[] Losses, float[] Weights, string? Last)[] Run(string failures, string name)
            {
                var options = DpOptions() with { Collective = "ring", Simulation = new SimulationOptions { Failures = failures, Seed = 5 } };
                return DpRun(DpTopology(device, 3), options, rank =>
                    ClusterPretrain(rank.Device, DpPretraining(corpus, 4, rank.Collectives, options) with { Momentum = 0.9f }, Path.Combine(folder, name)));
            }

            var whole = Run("none", "whole");
            (float[] Losses, float[] Weights, string? Last)[] joined;
            using (Telemetry.Subscribe(events))
            {
                joined = Run("test-late-training", "late");
            }

            Check(whole[0].Losses.Length == 4 && joined[0].Losses.Length == 4 && joined[2].Losses.Length == 2,
                $"ranks 0 and 1 train the 4 steps, rank 2 the 2 from its join: {joined[0].Losses.Length}, {joined[1].Losses.Length}, {joined[2].Losses.Length}");
            Check(joined[2].Losses.SequenceEqual(joined[0].Losses[2..]) && joined[0].Weights.SequenceEqual(joined[2].Weights) && joined[1].Weights.SequenceEqual(joined[2].Weights),
                $"after the join every rank reports the same losses and holds the same weights (the joiner took the trained weights and momentum; largest gap {Gap(joined[0].Weights, joined[2].Weights):G3})");
            AssertClose(whole[0].Losses, joined[0].Losses, 1e-5f, "the late-join run's losses against the run without the fault (2 ranks share steps 0 and 1)");
            AssertClose(whole[0].Weights, joined[0].Weights, 1e-5f, "the late-join run's weights against the run without the fault");
            Check(events.States.Any(s => s is { Rank: 2, State: RankState.Joined, Step: 2 }), $"rank 2 joined at step 2: {string.Join("; ", events.States.Where(s => s.Rank == 2))}");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // A batch source that is not a DataLoader (sharded by epoch in data parallel).
    private sealed class ForwardedBatches(IBatchSource inner) : IBatchSource
    {
        public IEnumerator<Batch> GetEnumerator() => inner.GetEnumerator();

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
    }

    private static void ClusterTrainerLateJoin(Device device)
    {
        // 150 samples, 3 ranks of batch 10: 5 steps an epoch; rank 2 joins at step 7, in the second epoch.
        using var late = Scripted("test-late-trainer", new Fault(FaultKind.LateJoin, 0, [2], TimeSpan.Zero, 7));
        var data = Spirals(50, 3, 1);
        (float[] Weights, double Loss) Fit(Device on, int batch, ICollectives? cluster, ClusterOptions? options, bool forwarded = false)
        {
            using var model = DpClassifier(on);
            using var optimizer = new Sgd(model.Parameters(), 0.1f, momentum: 0.9f);
            var loader = new DataLoader(data, batch, shuffle: true, device: on, seed: 7);
            var history = new Trainer(model, optimizer, Losses.CrossEntropy) { Cluster = cluster, ClusterOptions = options }.Fit(forwarded ? new ForwardedBatches(loader) : loader, 3);
            return ([.. model.Parameters().SelectMany(p => p.ToArray())], history.Epochs[^1].Loss);
        }

        var none = DpOptions() with { Collective = "ring", Simulation = new SimulationOptions { Failures = "none", Seed = 5 } };
        var faulty = none with { Simulation = none.Simulation with { Failures = "test-late-trainer" } };
        var alone = Fit(device, 30, null, null);
        var whole = DpRun(DpTopology(device, 3), none, rank => Fit(rank.Device, 10, rank.Collectives, none));
        var joined = DpRun(DpTopology(device, 3), faulty, rank => Fit(rank.Device, 10, rank.Collectives, faulty));
        Check(joined.All(r => r.Weights.SequenceEqual(joined[0].Weights)), "every rank ends with the same weights (the joiner took the trained weights and momentum)");
        AssertClose(whole[0].Weights, joined[0].Weights, 1e-5f, "the late-join run's weights against the run without the fault");
        AssertClose(alone.Weights, joined[0].Weights, 1e-5f, "the late-join run's weights against one process with batch 30");
        Check(Math.Abs(joined[0].Loss - whole[0].Loss) <= 1e-5 * Math.Max(1, Math.Abs(whole[0].Loss)), $"the last epoch's loss: {joined[0].Loss} against {whole[0].Loss}");

        var started = System.Diagnostics.Stopwatch.StartNew();
        var failed = InProcessCluster.Run(DpTopology(device, 3), faulty, rank => Fit(rank.Device, 10, rank.Collectives, faulty, forwarded: true));
        Check(!failed.Succeeded && failed.Errors[2] is { } error && error.Message.Contains("sits out step 0", StringComparison.Ordinal) && error.Message.Contains("sharded by epoch", StringComparison.Ordinal),
            $"a source sharded by epoch fails at once on the late joiner, naming why: {failed.Errors[2]?.Message}");
        Check(started.Elapsed < TimeSpan.FromSeconds(60) && failed.Errors.All(e => e is not null), $"every rank ends at once, not at the timeout: {started.Elapsed.TotalSeconds:F1} s; {string.Join("; ", failed.Errors.Select(e => e?.Message))}");
    }

    private static void ClusterTrainingReplay(Device device)
    {
        string folder = TempFolder("cluster-replay-training");
        try
        {
            string corpus = PretrainData("tokens.bin");
            var topology = DpTopology(device, 3);
            var options = DpOptions() with { Collective = "ring" };
            var wholeLosses = new float[3][];
            var replayed = new (float[] Losses, float[] Weights, string? Last)[3];
            var whole = InProcessCluster.Run(topology, options with { Simulation = new SimulationOptions { Seed = 9 } }, rank =>
                wholeLosses[rank.Rank] = ClusterPretrain(rank.Device, DpPretraining(corpus, 4, rank.Collectives, options) with { SaveEvery = 1 }, Path.Combine(folder, "whole")).Losses);
            whole.ThrowIfFailed();
            var replay = ClusterReplay.Run(topology, options, seed: 9, step: 1, rank =>
                replayed[rank.Rank] = ClusterPretrain(rank.Device, DpPretraining(corpus, 4, rank.Collectives, options) with { SaveEvery = 1 }, Path.Combine(folder, "replay")));
            replay.ThrowIfFailed();
            Check(replayed.All(r => r.Losses.SequenceEqual(wholeLosses[0][..2])), $"the replay trains steps 0 and 1 as the run did: {string.Join(" / ", replayed.Select(r => string.Join(", ", r.Losses)))}");
            Check(replayed.All(r => r.Weights.SequenceEqual(replayed[0].Weights)) && replayed[0].Last is { } last && last.EndsWith("checkpoint-2", StringComparison.Ordinal),
                $"every rank sat out steps 2 and 3 with the same weights; the last checkpoint is step 2's: {replayed[0].Last}");
            byte[] Weights(string run) => File.ReadAllBytes(Path.Combine(folder, run, "checkpoint-2", "model.safetensors"));
            Check(Weights("replay").AsSpan().SequenceEqual(Weights("whole")), "the replay's checkpoint after step 1 is the whole run's, byte for byte");
            for (int r = 0; r < 3; r++)
            {
                for (long s = 0; s <= 1; s++)
                {
                    Check(replay.Log!.ClockAt(r, s) is { } at && at == whole.Log!.ClockAt(r, s), $"rank {r}'s clock as step {s} began: {replay.Log!.ClockAt(r, s)} against {whole.Log!.ClockAt(r, s)}");
                }
            }

            // A run whose ranks measured for the cluster advisor (someone listened; one warm step, so its two slice collectives
            // go at the end of step 1) replays bit for bit from its log alone: the log's header holds the advisor's setting,
            // and the replay takes it, nobody listening and IDRAK_CLUSTER_SUGGEST unset. Without the log the clocks differ.
            var advised = options with { SuggestSteps = 1 };
            var listener = new MetricsRecorder(TelemetryLevel.Suggestions);
            ClusterRunResult measured;
            using (Telemetry.Subscribe(listener))
            {
                measured = InProcessCluster.Run(topology, advised with { Simulation = new SimulationOptions { Seed = 9 } }, rank =>
                    ClusterPretrain(rank.Device, DpPretraining(corpus, 4, rank.Collectives, advised), Path.Combine(folder, "advised")));
            }

            measured.ThrowIfFailed();
            var fromLog = ClusterReplay.Run(topology, options, seed: 9, step: 2, rank =>
                ClusterPretrain(rank.Device, DpPretraining(corpus, 4, rank.Collectives, options), Path.Combine(folder, "from-log")), measured.Log!);
            fromLog.ThrowIfFailed();
            var unlogged = ClusterReplay.Run(topology, options, seed: 9, step: 2, rank =>
                ClusterPretrain(rank.Device, DpPretraining(corpus, 4, rank.Collectives, options), Path.Combine(folder, "unlogged")));
            unlogged.ThrowIfFailed();
            Check(measured.Log!.Suggest && measured.Log.SuggestSteps == 1 && !unlogged.Log!.Suggest, "the log's header holds the advisor's setting: on, one warm step");
            for (int r = 0; r < 3; r++)
            {
                for (long s = 0; s <= 2; s++)
                {
                    Check(fromLog.Log!.ClockAt(r, s) is { } at && at == measured.Log.ClockAt(r, s),
                        $"rank {r}'s clock as step {s} began, replayed from the advised run's log: {fromLog.Log!.ClockAt(r, s)} against {measured.Log.ClockAt(r, s)}");
                }
            }

            Check(Enumerable.Range(0, 3).Any(r => unlogged.Log!.ClockAt(r, 2) != measured.Log.ClockAt(r, 2)),
                "without the log the replay does not measure, and its clocks after the advisor's step differ: the log's setting is what matched them");

            // The Trainer: one step an epoch (75 samples a rank); a replay through step 0 of three epochs leaves one step's weights.
            var data = Spirals(50, 3, 1);
            float[] Fit(ClusterRank rank, int epochs)
            {
                using var model = DpClassifier(rank.Device);
                using var optimizer = new Sgd(model.Parameters(), 0.1f);
                new Trainer(model, optimizer, Losses.CrossEntropy) { Cluster = rank.Collectives, ClusterOptions = options }
                    .Fit(new DataLoader(data, 75, shuffle: true, device: rank.Device, seed: 7), epochs);
                return [.. model.Parameters().SelectMany(p => p.ToArray())];
            }

            var oneStep = DpRun(DpTopology(device, 2), options, rank => Fit(rank, 1));
            var stopped = new float[2][];
            ClusterReplay.Run(DpTopology(device, 2), options, seed: 1, step: 0, rank => stopped[rank.Rank] = Fit(rank, 3)).ThrowIfFailed();
            Check(stopped.All(w => w.SequenceEqual(oneStep[0])), $"the Trainer replayed through step 0 holds one step's weights, bit for bit (largest gap {Gap(stopped[0], oneStep[0]):G3})");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void ClusterCliSimFailures(Device device)
    {
        if (!FirstRun(nameof(ClusterCliSimFailures)))
        {
            return;
        }

        string folder = TempFolder("cluster-cli-failures");
        using var slow = Scripted("test-cli-slow", new Fault(FaultKind.SlowWorker, 0, [1], TimeSpan.Zero, 4));
        try
        {
            string topology = Path.Combine(folder, "two.json");
            File.WriteAllText(topology, DpTopology(Device.Cpu, 2).ToJson());
            (int Code, string Output, string Error) Idrak(params string[] args)
            {
                using var output = new StringWriter(CultureInfo.InvariantCulture);
                using var error = new StringWriter(CultureInfo.InvariantCulture);
                int code = CommandLine.Run(args, output, error);
                return (code, output.ToString(), error.ToString());
            }

            string[] Pretrain(string run) => ["pretrain", TinyLlama, "--tokens", PretrainData("tokens.bin"), "--steps", "2", "--max-length", "32", "--batch-tokens", "64", "--optimizer", "sgd",
                "--lr", "0.2", "--no-graphs", "-o", run];
            var (code, text, errors) = Idrak(["cluster", "sim", "--topology", topology, "--failures", "test-cli-slow", "--seed", "5", "--", .. Pretrain(Path.Combine(folder, "run"))]);
            Check(code == ExitCodes.Ok, $"cluster sim ran pretrain with a slow worker: exit {code}: {errors}\n{text}");
            Check(text.Contains("failures test-cli-slow, seed 5", StringComparison.Ordinal)
                  && text.Contains("faults injected (1): SlowWorker at step 0 on rank(s) 1 (replay with --seed 5 --step 0)", StringComparison.Ordinal),
                $"the model, the seed and the fault with its replay key are printed: {text}");
            var (unknown, _, unknownError) = Idrak(["cluster", "sim", "--topology", topology, "--failures", "no-such-model", "--", .. Pretrain(Path.Combine(folder, "other"))]);
            Check(unknown != ExitCodes.Ok && unknownError.Contains("no-such-model", StringComparison.Ordinal) && unknownError.Contains("not registered", StringComparison.Ordinal),
                $"an unknown failure model is named: {unknownError}");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // Found in C9: a rank with no rows ran no backward pass, so it never opened the overlap and sent the header before its
    // buckets while the others sent their buckets first; every collective then met a mismatched one. Every rank now opens
    // the overlap each step, rows or not.
    private static void ClusterBucketsRankWithoutRows(Device device)
    {
        string folder = TempFolder("cluster-no-rows");
        try
        {
            // Two sequences a step over 3 ranks: rank 2 trains none; buckets of 4 KiB, many of them, sent during the backward pass.
            string corpus = PretrainData("tokens.bin");
            var options = DpOptions() with { Collective = "ring", BucketBytes = 4L << 10 };
            PretrainingOptions Options(ICollectives? cluster) => DpPretraining(corpus, 3, cluster, options) with { SequencesPerStep = 2, BatchTokens = 32, GradientAccumulation = 1 };
            var alone = DpPretrain(device, Options(null), Path.Combine(folder, "alone"));
            var (ranks, seen) = OverlapRun(DpTopology(device, 3), options, rank => DpPretrain(rank.Device, Options(rank.Collectives), Path.Combine(folder, "three")));
            AssertClose(alone.Losses, ranks[0].Losses, 1e-5f, "the token-weighted losses of ranks with 1, 1 and 0 sequences against one process");
            AssertClose(alone.Weights, ranks[0].Weights, 1e-5f, "the weights against one process");
            var synced = seen.Synchronisations.Where(s => s.Rank == 2).ToList();
            Check(synced.Count == 3 && synced.All(s => s.Buckets > 1), $"the rank without rows reduced every step's buckets ({string.Join(", ", synced.Select(s => s.Buckets))})");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }
}
