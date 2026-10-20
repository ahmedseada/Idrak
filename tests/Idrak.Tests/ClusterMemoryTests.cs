// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using Idrak;
using Idrak.Abstraction.Cluster;
using Idrak.Cluster;
using Idrak.Datasets;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Text;
using Idrak.Text.Abstractions;
using Idrak.Training;

// Plan 23, phase C8b: memory and speed. A pipeline stage and a parameter shard read only their own tensors from the
// checkpoint (a deferred load), a pipelined or sharded checkpoint is streamed to rank 0 a tensor at a time in the
// one-process format, activations cross device to device where the transport hands device memory over, a receive is woken
// directly by the thread that queued its message, and a stopped pipeline writes its checkpoint. Every model is plan 19's
// tiny one; every run an in-process cluster carved from the device under test, real transports in one process, or child
// processes of this program.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ClusterMemoryGroup =
    [
        ("cluster: a pipeline rank reads only its stage's tensors from the checkpoint (bytes read against its share, nothing at the load), holds them on its device against the whole model's, and its weights are one process's bit for bit", ClusterMemoryStageLoad),
        ("cluster: a parameters-shard rank reads only its own tensors from the checkpoint (the ranks' reads add up to the file once) and trains to the whole-load run's losses and checkpoint, byte for byte", ClusterMemoryShardLoad),
        ("cluster: a pipelined and a sharded checkpoint streamed to rank 0 a tensor at a time are the joined checkpoints byte for byte, load in one process (whole or deferred) with the same weights, and the one-process format still loads", ClusterMemoryStreamedCheckpoint),
        ("cluster: activations cross between stages device to device over pcie-host, bit-identical to staging through the host; the telemetry says which", ClusterMemoryDeviceActivations),
        ("cluster: a receive waiting on shared memory is woken directly by the reader thread (the idle-gap latency measured both ways, under half a timer tick)", ClusterMemoryDirectWake),
        ("cluster: a stopped pipeline writes its checkpoint (each stage's part streamed to rank 0): the uninterrupted run's files, and a resumed run ends as the uninterrupted one", ClusterMemoryPipelineStop),
        ("cluster: a stopped Trainer and a stopped fine-tuner (sharded and not) write the checkpoint of their last finished step, and a run resumed from it ends with the uninterrupted run's weights, bit for bit", ClusterMemoryTrainerStop),
    ];

    // Runs `work` on 2 ranks steered by a control that stops it after a few steps: the step every rank stopped before.
    private static int StoppedRun(Device device, ClusterOptions options, Action<ClusterRank> work, string what)
    {
        // A pause asked as a rank begins step 2 holds every rank before step 3 (a fast run cannot end first), then the stop.
        var control = new ClusterRunControl { Observed = true };
        control.RankStepped += e =>
        {
            if (e.Step >= 2)
            {
                control.Pause();
            }
        };
        var running = Task.Run(() => InProcessCluster.Run(DpTopology(device, 2), options, work, control));
        Check(UiWait(() => control.State == ClusterRunState.Paused, 120), $"{what}: both ranks wait before step 3 ({control.State}, last step {control.LastStep})");
        long last = control.Stop();
        Check(running.Wait(TimeSpan.FromMinutes(3)), $"{what}: the stopped run ends");
        var result = running.Result;
        Check(result.Errors.All(e => e is ClusterRunStoppedException s && s.Step == last + 1),
            $"{what}: every rank stopped as it began step {last + 1}: {string.Join("; ", result.Errors.Select(e => e?.Message ?? "finished"))}");
        return (int)last + 1;
    }

    private static void ClusterMemoryTrainerStop(Device device)
    {
        string folder = TempFolder("cluster-trainer-stop");
        try
        {
            // The Trainer: 4 epochs of the spirals, 5 steps an epoch over 2 ranks, Adam.
            var data = Spirals(30, 3, 1);
            float[] Fit(ClusterRank rank, ClusterOptions options, string? stopFolder, string? resumeFrom)
            {
                using var model = ShardModel(rank.Device);
                using var optimizer = new Adam(model.Parameters(), 0.01f);
                var trainer = new Trainer(model, optimizer, Losses.CrossEntropy) { Cluster = rank.Collectives, ClusterOptions = options, StopFolder = stopFolder, ResumeFrom = resumeFrom };
                trainer.Fit(new DataLoader(data, 10, shuffle: true, device: rank.Device, seed: 7), 4);
                return [.. model.Parameters().SelectMany(p => p.ToArray())];
            }

            foreach (var shard in new[] { ShardMode.None, ShardMode.Parameters })
            {
                var options = DpOptions() with { Collective = "naive", Shard = shard };
                string stops = Path.Combine(folder, $"train-{shard}");
                int steps = StoppedRun(device, options, rank => Fit(rank, options, stops, null), $"Trainer, shard {shard}");
                string checkpoint = Path.Combine(stops, $"checkpoint-{steps}");
                Check(File.Exists(Path.Combine(checkpoint, "train-state.json")), $"Trainer, shard {shard}: the stop wrote {checkpoint}");
                var resumed = DpRun(DpTopology(device, 2), options, rank => Fit(rank, options, null, checkpoint));
                var straight = DpRun(DpTopology(device, 2), options, rank => Fit(rank, options, null, null));
                Check(resumed[0].SequenceEqual(straight[0]) && resumed[1].SequenceEqual(straight[1]),
                    $"Trainer, shard {shard}: stopped after {steps} steps and resumed, the weights after 4 epochs are the uninterrupted run's (largest gap {Gap(resumed[0], straight[0]):G3})");
            }

            // The command line takes them under the dashboard now (C8a refused a steered tune and train: no checkpoint on a stop).
            Idrak.Cli.Commands.Cluster.ClusterSimCommand.CheckTrainer(["tune", "train", "tiny", "data.jsonl", "-o", "adapters"]);
            Idrak.Cli.Commands.Cluster.ClusterSimCommand.CheckTrainer(["train", "data.csv", "--run", "run"]);

            // The fine-tuner: LoRA on the tiny model, 2 epochs of 16 windows.
            var tokens = File.ReadAllBytes(PretrainData("tokens.bin"));
            var sequences = Enumerable.Range(0, 16).Select(w => new TrainingSequence(
                [.. Enumerable.Range(0, 24).Select(i => (int)BitConverter.ToUInt16(tokens, ((w * 24) + i) * 2))], [.. Enumerable.Repeat(true, 24)])).ToList();
            float[] Tune(ClusterRank rank, ClusterOptions options, string output, string? resumeFrom)
            {
                using var model = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = rank.Device });
                var tuning = new FineTuningOptions
                {
                    Rank = 4, Alpha = 8, Epochs = 2, MaxLength = 32, BatchTokens = 48, CudaGraphs = false, Seed = 5, Cluster = rank.Collectives, ClusterOptions = options,
                    ResumeFrom = resumeFrom,
                };
                FineTuner.Train(model, sequences, null, tuning, output);
                return [.. model.Network.Parameters().Where(p => p.RequiresGrad).SelectMany(p => p.ToArray())];
            }

            foreach (var shard in new[] { ShardMode.None, ShardMode.Optimizer })
            {
                var options = DpOptions() with { Collective = "naive", Shard = shard };
                string stops = Path.Combine(folder, $"tune-{shard}");
                int steps = StoppedRun(device, options, rank => Tune(rank, options, Path.Combine(stops, $"rank-{rank.Rank}"), null), $"fine-tuner, shard {shard}");
                string checkpoint = Path.Combine(stops, "rank-0", $"checkpoint-{steps}");
                Check(File.Exists(Path.Combine(checkpoint, "tune-state.json")), $"fine-tuner, shard {shard}: the stop wrote {checkpoint}");
                var resumed = DpRun(DpTopology(device, 2), options, rank => Tune(rank, options, Path.Combine(folder, $"tune-resumed-{shard}-{rank.Rank}"), checkpoint));
                var straight = DpRun(DpTopology(device, 2), options, rank => Tune(rank, options, Path.Combine(folder, $"tune-straight-{shard}-{rank.Rank}"), null));
                Check(resumed[0].Length > 0 && resumed[0].SequenceEqual(straight[0]),
                    $"fine-tuner, shard {shard}: stopped after {steps} steps and resumed, the adapters are the uninterrupted run's (largest gap {Gap(resumed[0], straight[0]):G3})");
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // The checkpoint's tensors: each one's bytes in the file.
    private static Dictionary<string, long> FileTensorBytes(string folder)
    {
        using var reader = SafeTensorsReader.Open(folder);
        return reader.Tensors.ToDictionary(t => t.Key, t => t.Value.Length, StringComparer.Ordinal);
    }

    private static void ClusterMemoryStageLoad(Device device)
    {
        long whole = FileTensorBytes(TinyLlama).Values.Sum();
        float[][] reference;
        using (var one = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = Device.Cpu }))
        {
            reference = [.. one.Network.Parameters().Distinct().Select(p => p.ToArray())];
        }

        var options = PipeOptions();
        var ranks = DpRun(DpTopology(device, 2), options, rank =>
        {
            long before = rank.Device.Backend.GetMemoryUsage().InUse;
            using var model = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = rank.Device, Deferred = true });
            long atLoad = model.BytesRead ?? -1;
            bool deferred = model.IsDeferred;
            using var pipe = PipelineStage.Place(model.Network, rank.Collectives, options);
            long read = model.BytesRead ?? -1;
            long inUse = rank.Device.Backend.GetMemoryUsage().InUse - before;
            long own = pipe.Parameters.Sum(p => (long)p.Size * sizeof(float));
            var weights = pipe.Parameters.Select((p, i) => (Index: pipe.ParameterIndices[i], Values: p.ToArray())).ToList();
            return (pipe.Stage, Deferred: deferred, AtLoad: atLoad, Read: read, Own: own, InUse: inUse, pipe.Bytes, Weights: weights);
        });

        foreach (var r in ranks)
        {
            Check(r.Deferred && r.AtLoad == 0, $"stage {r.Stage}: the load read no weight ({r.AtLoad:N0} B), the model deferred");
            Check(r.Read == r.Own && r.Read < whole, $"stage {r.Stage}: read {r.Read:N0} B from the checkpoint, its own tensors' {r.Own:N0} B, of the file's {whole:N0} B");
            Check(r.InUse >= r.Own && r.InUse < whole, $"stage {r.Stage}: {r.InUse:N0} B in use on its device, at least its {r.Own:N0} B and under the whole model's {whole:N0} B");
            foreach (var (index, values) in r.Weights)
            {
                Check(values.SequenceEqual(reference[index]), $"stage {r.Stage}: parameter {index} as one process loads it, bit for bit");
            }
        }

        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    measured: bytes read per rank {string.Join(", ", ranks.Select(r => $"stage {r.Stage} {r.Read:N0} B ({r.Read * 100.0 / whole:F0}%)"))} of the file's {whole:N0} B; "
            + $"in use {string.Join(", ", ranks.Select(r => $"{r.InUse:N0} B"))}; the deferred device keeps {DeferredTensors.Device.Backend.GetMemoryUsage().InUse:N0} B (computed tables of unplaced modules)"));
    }

    // A pretraining run of 3 steps on 2 ranks sharding the parameters, its model loaded deferred or whole: each rank's bytes
    // read and device peak, the losses, and rank 0's checkpoint files.
    private static (long Read, long Peak, float[] Losses, byte[] Model, byte[] Optimizer)[] ShardedPretrain(Device device, bool deferred, ClusterOptions options, string folder)
    {
        string corpus = PretrainData("tokens.bin");
        return DpRun(DpTopology(device, 2), options, rank =>
        {
            rank.Device.Backend.ResetPeakMemoryUsage();
            using var model = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = rank.Device, Deferred = deferred });
            var settings = DpPretraining(corpus, 3, rank.Collectives, options) with { Optimizer = "adamw", BatchTokens = 32, GradientAccumulation = 2 };
            var result = Pretrainer.Train(model, settings, Path.Combine(folder, $"rank-{rank.Rank}"));
            long read = model.BytesRead ?? -1;
            byte[] Bytes(string file) => rank.Rank == 0 && result.LastCheckpoint is { } last ? File.ReadAllBytes(Path.Combine(last, file)) : [];
            return (read, rank.Device.Backend.GetMemoryUsage().Peak, result.Losses.ToArray(), Bytes("model.safetensors"), Bytes("optimizer.safetensors"));
        });
    }

    private static void ClusterMemoryShardLoad(Device device)
    {
        string folder = TempFolder("cluster-shard-load");
        try
        {
            long whole = FileTensorBytes(TinyLlama).Values.Sum();
            var options = DpOptions() with { Collective = "ring", Shard = ShardMode.Parameters };
            var own = ShardedPretrain(device, deferred: true, options, Path.Combine(folder, "own"));
            var all = ShardedPretrain(device, deferred: false, options, Path.Combine(folder, "whole"));
            Check(own.Sum(r => r.Read) == whole && own.All(r => r.Read > 0 && r.Read < whole),
                $"each rank read its own tensors, together the file once: {string.Join(", ", own.Select(r => $"{r.Read:N0} B"))} of {whole:N0} B");
            Check(all.All(r => r.Read == whole), $"loaded whole, every rank read the whole file: {string.Join(", ", all.Select(r => $"{r.Read:N0} B"))}");
            Check(own[0].Losses.SequenceEqual(all[0].Losses) && own[1].Losses.SequenceEqual(all[1].Losses), "the losses are the whole-load run's, bit for bit");
            Check(own[0].Model.AsSpan().SequenceEqual(all[0].Model) && own[0].Optimizer.AsSpan().SequenceEqual(all[0].Optimizer),
                "rank 0's checkpoint (model and optimizer state) is the whole-load run's, byte for byte");
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"    measured: bytes read per rank {string.Join(", ", own.Select(r => $"{r.Read:N0} B ({r.Read * 100.0 / whole:F0}%)"))} against {whole:N0} B each loaded whole; "
                + $"device peak per rank {string.Join(", ", own.Select(r => $"{r.Peak:N0} B"))} against {string.Join(", ", all.Select(r => $"{r.Peak:N0} B"))} loaded whole"));
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void ClusterMemoryStreamedCheckpoint(Device device)
    {
        string folder = TempFolder("cluster-streamed");
        try
        {
            string corpus = PretrainData("tokens.bin");
            // A pipeline of 2 stages, its checkpoint streamed and joined (C7's path).
            string Pipelined(bool streamed)
            {
                var options = PipeOptions(microBatches: 2) with { StreamedSave = streamed };
                string run = Path.Combine(folder, streamed ? "pipe-streamed" : "pipe-joined");
                DpRun(DpTopology(device, 2), options, rank =>
                {
                    using var model = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = rank.Device, Deferred = true });
                    return Pretrainer.Train(model, DpPretraining(corpus, 3, rank.Collectives, options) with { Packing = false, Optimizer = "adamw", SaveEvery = 3 }, run).LastCheckpoint;
                });
                return Path.Combine(run, "checkpoint-3");
            }

            string pipeStreamed = Pipelined(true), pipeJoined = Pipelined(false);
            foreach (string file in new[] { "model.safetensors", "optimizer.safetensors" })
            {
                Check(File.ReadAllBytes(Path.Combine(pipeStreamed, file)).AsSpan().SequenceEqual(File.ReadAllBytes(Path.Combine(pipeJoined, file))),
                    $"the pipeline's streamed {file} is the joined one, byte for byte");
            }

            // A parameters shard, its checkpoint streamed and joined (C6's path).
            var shard = DpOptions() with { Collective = "ring", Shard = ShardMode.Parameters };
            var streamedShard = ShardedPretrain(device, deferred: true, shard, Path.Combine(folder, "shard-streamed"));
            var joinedShard = ShardedPretrain(device, deferred: true, shard with { StreamedSave = false }, Path.Combine(folder, "shard-joined"));
            static string Differs(byte[] a, byte[] b) => a.AsSpan().SequenceEqual(b) ? "equal"
                : $"{a.Length:N0} against {b.Length:N0} B, first difference at byte {Enumerable.Range(0, Math.Min(a.Length, b.Length)).FirstOrDefault(i => a[i] != b[i], Math.Min(a.Length, b.Length)):N0}";
            Check(streamedShard[0].Model.AsSpan().SequenceEqual(joinedShard[0].Model) && streamedShard[0].Optimizer.AsSpan().SequenceEqual(joinedShard[0].Optimizer)
                    && streamedShard[0].Losses.SequenceEqual(joinedShard[0].Losses),
                $"the shard's streamed checkpoint is the joined one, byte for byte: model {Differs(streamedShard[0].Model, joinedShard[0].Model)}, "
                + $"optimizer state {Differs(streamedShard[0].Optimizer, joinedShard[0].Optimizer)}, losses {(streamedShard[0].Losses.SequenceEqual(joinedShard[0].Losses) ? "equal" : "differ")}");

            // The streamed checkpoint in one process, loaded whole and deferred; and the one-process format of the start.
            float[] Weights(string path, bool deferred)
            {
                using var model = PretrainedModel.Load(path, new PretrainedOptions { Device = Device.Cpu, Deferred = deferred });
                model.Place();
                return [.. model.Network.Parameters().Distinct().SelectMany(p => p.ToArray())];
            }

            var saved = ReadTensors(Path.Combine(pipeStreamed, "model.safetensors"));
            var loaded = Weights(pipeStreamed, deferred: false);
            Check(loaded.SequenceEqual(Weights(pipeStreamed, deferred: true)), "the streamed checkpoint loads in one process alike whole and deferred, bit for bit");
            Check(loaded.Length == saved.Values.Sum(t => t.Values.Length), $"every tensor of the streamed checkpoint is loaded ({loaded.Length:N0} values)");
            Check(Weights(TinyLlama, deferred: false).SequenceEqual(Weights(TinyLlama, deferred: true)), "the one-process format loads whole and deferred alike, bit for bit");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void ClusterMemoryDeviceActivations(Device device)
    {
        var (ids, targets) = PipeWindows();
        (float[] Logits, float[][] Grads, string?[] Staging, long Bytes) Run(bool onDevice)
        {
            var options = RealOptions(TimeSpan.FromSeconds(60)) with { DeviceActivations = onDevice, Placement = "pipeline" };
            var topology = RealTopology([2]);
            var group = CreditGroup("pcie-host", 2, options);
            var metrics = new MetricsRecorder(TelemetryLevel.Network);
            var logits = new float[2][];
            var grads = new float[2][][];
            using (Telemetry.Subscribe(metrics))
            {
                var ranks = Enumerable.Range(0, 2).Select(r => Task.Factory.StartNew(() =>
                {
                    using var collectives = Collectives.Get("ring").Create(group[r], SimulatedDevices.Carve(device, null, 60 + r), new ClusterContext(topology, r, options));
                    using var model = HostTiny();
                    using var pipe = PipelineStage.Place(model.Network, collectives, options);
                    var seen = new float[2][];
                    pipe.Step(2,
                        m => Tensor.From(ids.AsSpan(m * 64, 64), [2, 32], pipe.Device),
                        (m, output) =>
                        {
                            seen[m] = output.ToArray();
                            return Losses.SparseCrossEntropy(output, Tensor.From(targets.AsSpan(m * 64, 64), [2, 32], pipe.Device)) * 0.5f;
                        });
                    logits[r] = pipe.IsLast ? [.. seen.SelectMany(s => s)] : [];
                    grads[r] = [.. pipe.Parameters.Select(p => p.Grad!.ToArray())];
                }, TaskCreationOptions.LongRunning)).ToArray();
                try
                {
                    Check(Task.WaitAll(ranks, TimeSpan.FromSeconds(120)), "both stages finished their step");
                }
                finally
                {
                    foreach (var t in group)
                    {
                        t.Dispose();
                    }
                }
            }

            var steps = metrics.PipelineSteps;
            return ([.. logits.SelectMany(l => l)], [.. grads.SelectMany(g => g)], [.. steps.Select(s => s.Staging)], steps.Sum(s => s.BytesSent));
        }

        var direct = Run(onDevice: true);
        var staged = Run(onDevice: false);
        Check(direct.Logits.Length > 0 && direct.Logits.SequenceEqual(staged.Logits), "the last stage's logits device to device are the host-staged ones, bit for bit");
        Check(direct.Grads.Length == staged.Grads.Length && direct.Grads.Zip(staged.Grads).All(g => g.First.SequenceEqual(g.Second)),
            "every stage's gradients device to device are the host-staged ones, bit for bit");
        Check(direct.Staging is ["device", "device"] && staged.Staging is ["host", "host"],
            $"the telemetry says how they crossed: [{string.Join(", ", direct.Staging)}] and [{string.Join(", ", staged.Staging)}]");
        Check(direct.Bytes == staged.Bytes, $"the same bytes cross either way ({direct.Bytes:N0} and {staged.Bytes:N0})");
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"    measured: {direct.Bytes:N0} B of activations and gradients a step, device to device and through the host alike"));
    }

    private static void ClusterMemoryDirectWake(Device device)
    {
        if (!FirstRun(nameof(ClusterMemoryDirectWake)))
        {
            return;
        }

        var ticks = new double[15];
        for (int i = 0; i < ticks.Length; i++)
        {
            long start = Stopwatch.GetTimestamp();
            Thread.Sleep(1);
            ticks[i] = Stopwatch.GetElapsedTime(start).TotalSeconds;
        }

        Array.Sort(ticks);
        double tick = ticks[ticks.Length / 2];
        (double Median, double Worst) Measure(bool direct)
        {
            string folder = TempFolder(direct ? "cluster-wake-direct" : "cluster-wake-pool");
            try
            {
                string path = Path.Combine(folder, "topology.json");
                File.WriteAllText(path, RealTopology([2]).ToJson());
                var (file, prefix) = LocalhostCluster.ProgramOf(typeof(Tests).Assembly);
                var launch = new LocalhostLaunch(file, [.. prefix, "--cluster-child", "shm-speed", folder])
                {
                    Environment = new Dictionary<string, string> { ["IDRAK_CLUSTER_TIMEOUT"] = "60", ["IDRAK_CLUSTER_DIRECT_WAKE"] = direct ? "1" : "0" },
                    Limit = TimeSpan.FromSeconds(120),
                };
                var result = LocalhostCluster.Run(path, new ClusterOptions { Timeout = TimeSpan.FromSeconds(60) }, launch);
                Check(result.Succeeded, $"the two processes ran: {result.Failure} " + string.Join("; ", result.Ranks.Select(r => $"rank {r.Rank} exit {r.ExitCode}: {string.Join(" | ", r.Errors.TakeLast(3))}")));
                var parts = File.ReadAllText(Path.Combine(folder, "speed.txt")).Split(' ');
                return (double.Parse(parts[1], CultureInfo.InvariantCulture), double.Parse(parts[2], CultureInfo.InvariantCulture));
            }
            finally
            {
                Directory.Delete(folder, recursive: true);
            }
        }

        // The inbox's own wake, apart from the shared-memory doorbell: two threads of one process over pcie-host (a frame is
        // queued by the sender's thread), a round trip after an idle gap of 2 ms, the first rounds (cold) aside.
        (double Median, double Worst) InProcess(bool direct)
        {
            var group = CreditGroup("pcie-host", 2, RealOptions(TimeSpan.FromSeconds(30)) with { DirectWake = direct });
            try
            {
                const int Rounds = 80, Cold = 5;
                var echo = Task.Factory.StartNew(() =>
                {
                    for (int i = 0; i < Rounds; i++)
                    {
                        group[1].Receive(0, [], 7);
                        group[1].Send(0, [], 7);
                    }
                }, TaskCreationOptions.LongRunning);
                var latencies = new List<double>();
                for (int i = 0; i < Rounds; i++)
                {
                    Thread.Sleep(2);
                    long start = Stopwatch.GetTimestamp();
                    group[0].Send(1, [], 7);
                    group[0].Receive(1, [], 7);
                    if (i >= Cold)
                    {
                        latencies.Add(Stopwatch.GetElapsedTime(start).TotalSeconds / 2);
                    }
                }

                Check(echo.Wait(TimeSpan.FromSeconds(30)), "the echoing rank finished");
                latencies.Sort();
                return (latencies[latencies.Count / 2], latencies[^1]);
            }
            finally
            {
                foreach (var t in group)
                {
                    t.Dispose();
                }
            }
        }

        var pooledHere = InProcess(direct: false);
        var wokenHere = InProcess(direct: true);
        var pooled = Measure(direct: false);
        var woken = Measure(direct: true);
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    measured: a message after an idle gap between two processes (shared memory), woken through the thread pool {pooled.Median * 1e6:F0} us median ({pooled.Worst * 1e6:F0} us worst), "
            + $"woken directly {woken.Median * 1e6:F0} us median ({woken.Worst * 1e6:F0} us worst); between two threads (pcie-host) {pooledHere.Median * 1e6:F1} us "
            + $"({pooledHere.Worst * 1e6:F0} us worst) through the pool, {wokenHere.Median * 1e6:F1} us ({wokenHere.Worst * 1e6:F0} us worst) directly; a timer tick {tick * 1e6:F0} us"));
        Check(woken.Median < tick / 2, $"woken directly, a message after an idle gap arrives in far less than a timer tick: {woken.Median * 1e6:F0} us against {tick * 1e6:F0} us");
    }

    private static void ClusterMemoryPipelineStop(Device device)
    {
        string folder = TempFolder("cluster-pipeline-stop");
        try
        {
            string corpus = PretrainData("tokens.bin");
            var options = PipeOptions();
            PretrainingOptions Options(int steps, ICollectives cluster) => DpPretraining(corpus, steps, cluster, options) with { Packing = false, Optimizer = "adamw" };
            (float[] Losses, string? Last) Pretrain(ClusterRank rank, int steps, string run, string? from = null)
            {
                using var model = PretrainedModel.Load(from ?? TinyLlama, new PretrainedOptions { Device = rank.Device, Deferred = true });
                var result = Pretrainer.Train(model, Options(steps, rank.Collectives), run, from is null ? null : Pretrainer.ReadCheckpoint(from));
                return ([.. result.Losses], result.LastCheckpoint);
            }

            string stopped = Path.Combine(folder, "stopped");
            var control = new ClusterRunControl();
            control.Pause();
            var running = Task.Run(() => InProcessCluster.Run(DpTopology(device, 2), options, rank => Pretrain(rank, 40, stopped), control));
            Check(UiWait(() => control.State == ClusterRunState.Paused, 120), "both stages wait before step 0");
            control.Resume();
            Check(UiWait(() => control.LastStep >= 1, 120), "two steps begun");
            long last = control.Stop();
            Check(running.Wait(TimeSpan.FromMinutes(3)), "the stopped pipeline ends");
            var result = running.Result;
            Check(result.Errors.All(e => e is ClusterRunStoppedException s && s.Step == last + 1),
                $"both stages stopped as they began step {last + 1}: {string.Join("; ", result.Errors.Select(e => e?.Message ?? "finished"))}");
            int steps = (int)last + 1;
            string checkpoint = Path.Combine(stopped, $"checkpoint-{steps}");
            Check(File.Exists(Path.Combine(checkpoint, "pretrain-state.json")), $"the stop wrote the checkpoint of its last finished step ({checkpoint})");

            string straight = Path.Combine(folder, "straight");
            DpRun(DpTopology(device, 2), options, rank => Pretrain(rank, steps, straight));
            string whole = Path.Combine(straight, $"checkpoint-{steps}");
            foreach (string file in new[] { "model.safetensors", "optimizer.safetensors" })
            {
                Check(File.ReadAllBytes(Path.Combine(checkpoint, file)).AsSpan().SequenceEqual(File.ReadAllBytes(Path.Combine(whole, file))),
                    $"the stop's {file} is the uninterrupted run's after {steps} steps, byte for byte (each stage's part)");
            }

            // Resumed from the stop's checkpoint (each stage reading its own part) to 3 more steps: the run of them all at once.
            int total = steps + 3;
            var resumed = DpRun(DpTopology(device, 2), options, rank => Pretrain(rank, total, Path.Combine(folder, "resumed"), checkpoint));
            var uninterrupted = DpRun(DpTopology(device, 2), options, rank => Pretrain(rank, total, Path.Combine(folder, "all")));
            Check(resumed[0].Losses.SequenceEqual(uninterrupted[0].Losses), $"the resumed run's losses are the uninterrupted run's ({resumed[0].Losses.Length} steps)");
            Check(File.ReadAllBytes(Path.Combine(resumed[0].Last!, "model.safetensors")).AsSpan().SequenceEqual(File.ReadAllBytes(Path.Combine(uninterrupted[0].Last!, "model.safetensors"))),
                $"and its weights after {total} steps, byte for byte");
            Console.WriteLine($"    pipeline stopped after {steps} steps; the checkpoint as the uninterrupted run's, resumed to {total} steps as one run");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }
}
