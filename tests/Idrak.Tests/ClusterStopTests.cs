// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using Idrak;
using Idrak.Abstraction.Cluster;
using Idrak.Abstraction.Reinforcement;
using Idrak.Cluster;
using Idrak.Datasets;
using Idrak.Models;
using Idrak.Networking.Training;
using Idrak.Reinforcement;
using Idrak.Text;
using Idrak.Text.Abstractions;
using Idrak.Text.Reinforcement;
using Idrak.Training;
using Idrak.Training.Abstractions;

// Plan 27, R7: the ranks agree on a stop. A cancellation raised on one rank (between steps, inside a step, during a rollout,
// on a slow rank) ends every rank at the same step boundary, with no rank left in a collective; rank 0 writes one checkpoint
// and a run resumed from it ends as the uninterrupted one. Every run is an in-process cluster carved from the device under
// test (the "shared" agreement by default, "all-reduce" forced where it is the one under test), each awaited far within the
// collectives' timeout; one process stops as before.
internal abstract partial class DesktopTests : Tests
{
    internal static readonly (string Name, Action<Device> Run)[] ClusterStopGroup =
    [
        ("cluster stop: the settings (IDRAK_CLUSTER_STOP, _STOP_EVERY, _STOP_WAIT; a bad value named) and the registry (all-reduce from Idrak.Network, shared from Idrak.Cluster, auto by the collectives); on 3 ranks both agreements stop every rank at the same boundary, a carried flag needs no message, StopEvery spaces the exchanges, a reset starts afresh, and a rank that waits past StopWait gives up loudly", StopAgreementRules),
        ("cluster stop: a Trainer cancelled on one rank of 2 (after a batch, and beside a slow rank) stops every rank at the same step under both agreements, rank 0's checkpoint resumes to the uninterrupted run's weights bit for bit; on 3 ranks an unstopped run's weights and losses are the same with either agreement (the carried flag changes no result)", StopTrainer),
        ("cluster stop: a fine-tuner cancelled on one rank of 2 (inside a step, and at a step's end) stops every rank at the same step under both agreements, rank 0 alone writes one checkpoint, and a run resumed from it ends with the uninterrupted run's adapters bit for bit; unstopped, both agreements give the same adapters; one process cancelled throws as before", StopFineTuner),
        ("cluster stop: a pretrainer (data parallel, and 2 pipeline stages) cancelled on one rank stops every rank at the same step, rank 0's checkpoint is the last finished step's, and the data-parallel run resumed from it gives the uninterrupted run's losses", StopPretrainer),
        ("cluster stop: reinforcement on 2 ranks cancelled on one rank (between iterations, and during a rollout) stops every rank at the same iteration, rank 0 writes the checkpoint once, and the run resumed from it ends with the uninterrupted run's adapters; one process cancelled stops as before", StopReinforcement),
        ("cluster stop: cli reinforce under cluster sim with --timeout on each rank: every rank stops at the same iteration, rank 0 prints and writes once (the adapter, the checkpoint), and --resume goes on from it", StopCliReinforce),
        ("cluster stop: the cost on the simulated cluster: the flag carried in the gradient header (no message of its own after the first boundary; network time per step with and without it) and the boundary's own one-value all-reduce (wall and virtual-clock time per boundary)", StopCost),
    ];

    // In-process ranks are awaited this long at most: far within the collectives' timeout of the options below.
    private static readonly TimeSpan StopBound = TimeSpan.FromSeconds(150);

    private static ClusterOptions StopOptions(string agreement = StopAgreements.Automatic) => DpOptions() with { StopAgreement = agreement, Timeout = TimeSpan.FromMinutes(10) };

    // Runs `work` on `world` ranks and waits for them, far within the collectives' timeout.
    private static ClusterRunResult StopRun(Device device, int world, ClusterOptions options, Action<ClusterRank> work, string what)
    {
        var running = Task.Run(() => InProcessCluster.Run(DpTopology(device, world), options, work));
        Check(running.Wait(StopBound), $"{what}: every rank ended within {StopBound.TotalSeconds:F0} s (the collectives' timeout is {options.Timeout.TotalSeconds:F0} s)");
        return running.Result;
    }

    // The step every rank stopped before (each left with an agreed stop), checked to be one step.
    private static long AgreedStop(ClusterRunResult result, string what)
    {
        var steps = result.Errors.Select(e => e is ClusterRunStoppedException { Agreed: true } s ? s.Step : -1).ToArray();
        Check(steps.All(s => s >= 0 && s == steps[0]),
            $"{what}: every rank stopped at the same boundary (an agreed stop): {string.Join("; ", result.Errors.Select((e, r) => $"rank {r}: {e?.GetType().Name} {e?.Message ?? "finished"}"))}");
        return steps[0];
    }

    private static void StopAgreementRules(Device device)
    {
        if (!FirstRun(nameof(StopAgreementRules)))
        {
            return;
        }

        // The settings and their variables.
        var defaults = new ClusterOptions();
        Check(defaults.StopAgreement == StopAgreements.Automatic && defaults.StopEvery == 1 && defaults.StopWait == TimeSpan.Zero, "defaults: auto, every boundary, no give-up");
        var set = ClusterOptions.From(n => n switch
        {
            "IDRAK_CLUSTER_STOP" => "all-reduce",
            "IDRAK_CLUSTER_STOP_EVERY" => "3",
            "IDRAK_CLUSTER_STOP_WAIT" => "1.5",
            _ => null,
        }, null);
        Check(set.StopAgreement == "all-reduce" && set.StopEvery == 3 && set.StopWait == TimeSpan.FromSeconds(1.5), "the variables set them");
        ThrowsNaming<ArgumentException>(() => ClusterOptions.From(n => n == "IDRAK_CLUSTER_STOP_EVERY" ? "0" : null, null), "a stop check every 0 boundaries", "IDRAK_CLUSTER_STOP_EVERY=0");
        ThrowsNaming<ArgumentException>(() => ClusterOptions.From(n => n == "IDRAK_CLUSTER_STOP_WAIT" ? "-1" : null, null), "a negative wait", "IDRAK_CLUSTER_STOP_WAIT=-1");
        Check(new[] { "IDRAK_CLUSTER_STOP", "IDRAK_CLUSTER_STOP_EVERY", "IDRAK_CLUSTER_STOP_WAIT" }.All(ClusterOptions.Variables.Contains), "ClusterOptions lists the variables");

        // The registry.
        Check(StopAgreements.Origin("all-reduce") == Overrides.Library && StopAgreements.Origin("shared") == Overrides.Library
              && StopAgreements.Describe().Contains("all-reduce:", StringComparison.Ordinal) && StopAgreements.Describe().Contains("shared:", StringComparison.Ordinal),
            $"the library registers all-reduce and shared: {StopAgreements.Describe()}");
        ThrowsNaming<NotSupportedException>(() => StopAgreements.Get("no-such-agreement"), "an unknown agreement", "is not registered", "StopAgreements.Register");

        // On 3 ranks, a barrier a step (the step's collective): rank 1 asks after its boundary 3; every rank stops at 4.
        foreach (string name in new[] { StopAgreements.Automatic, "all-reduce" })
        {
            var options = StopOptions(name);
            var seen = DpRun(DpTopology(device, 3), options, rank =>
            {
                var stop = StopAgreements.For(rank.Collectives, options);
                bool same = ReferenceEquals(stop, StopAgreements.For(rank.Collectives));
                stop.Reset();
                long at = -1;
                for (long b = 0; b < 40 && at < 0; b++)
                {
                    if (stop.Agree(b))
                    {
                        at = b;
                        break;
                    }

                    if (rank.Rank == 1 && b == 3)
                    {
                        stop.Request("a test");
                        stop.Request("a second reason, not kept");
                    }

                    rank.Collectives.Barrier();
                }

                var stopped = stop.Stopped(at);
                bool sticky = stop.Agree(at + 1);

                // A reset starts afresh: 5 boundaries, nobody asks, nobody stops.
                stop.Reset();
                bool fresh = true;
                for (long b = 0; b < 5; b++)
                {
                    fresh &= !stop.Agree(b);
                    rank.Collectives.Barrier();
                }

                return (stop.Name, Same: same, At: at, stopped.Agreed, stopped.Step, stopped.Message, Sticky: sticky, Fresh: fresh, Reason: rank.Rank == 1 ? stopped.Message : "");
            });

            string expected = name == "all-reduce" ? "all-reduce" : "shared";
            Check(seen.All(s => s.Name == expected && s.Same), $"{name}: every rank's agreement is {expected}, one per rank's collectives: {string.Join(", ", seen.Select(s => s.Name))}");
            Check(seen.All(s => s.At == 4 && s.Agreed && s.Step == 4 && s.Sticky && s.Fresh),
                $"{name}: rank 1 asked after boundary 3, every rank stopped at boundary 4 (agreed, sticky until a reset; afresh after it): {string.Join(", ", seen.Select(s => s.At))}");
            Check(seen.All(s => s.Message.Contains("1 rank of 3 asked", StringComparison.Ordinal)) && seen[1].Reason.Contains("rank 1 asked: a test", StringComparison.Ordinal),
                $"{name}: the message says how many asked and, on the asker, why: {seen[1].Message}");
        }

        // all-reduce: a carried flag decides the next boundary with no message; StopEvery 3 exchanges at every third boundary.
        var spaced = StopOptions("all-reduce") with { StopEvery = 3 };
        var counts = DpRun(DpTopology(device, 2), spaced, rank =>
        {
            var agreement = StopAgreements.For(rank.Collectives, spaced);
            Check(agreement is ICarriedStopAgreement, "the all-reduce agreement's flag may travel in a collective the ranks make anyway (ICarriedStopAgreement)");
            var stop = (AllReduceStopAgreement)agreement;
            stop.Reset();
            for (long b = 0; b < 6; b++)
            {
                _ = stop.Agree(b);                               // boundaries 0 and 3 exchange
            }

            long exchanged = stop.Exchanges;
            if (rank.Rank == 0)
            {
                stop.Request("carried");
            }

            // The flag summed in a collective the ranks make anyway (as the gradient header carries it).
            Span<float> carried = [stop.Contribution];
            Check(((IHostValueCollectives)rank.Collectives).TryAllReduce(carried, ReduceOp.Sum, WireType.Float32), "the simulated ranks take host values");
            stop.Carried(carried[0]);
            bool agreed = stop.Agree(7);
            return (Exchanged: exchanged, After: stop.Exchanges, Agreed: agreed);
        });
        Check(counts.All(c => c.Exchanged == 2 && c.After == 2 && c.Agreed),
            $"StopEvery 3: 2 exchanges in 6 boundaries; the carried flag stopped the next boundary with no exchange: {string.Join(", ", counts.Select(c => $"{c.Exchanged}/{c.After}/{c.Agreed}"))}");

        // A rank that asked and waits past StopWait gives up: its step's token is cancelled, and it says why, loudly.
        var waiting = StopOptions() with { StopWait = TimeSpan.FromSeconds(0.2) };
        var gaveUp = DpRun(DpTopology(device, 2), waiting, rank =>
        {
            var stop = StopAgreements.For(rank.Collectives, waiting);
            stop.Reset();
            _ = stop.Agree(0);
            if (rank.Rank == 1)
            {
                stop.Request("a test that waits");
                Thread.Sleep(1000);                              // the step goes on past the wait
            }

            rank.Collectives.Barrier();
            bool cancelled = stop.GiveUpToken.IsCancellationRequested;
            var error = stop.GaveUpError(new OperationCanceledException(stop.GiveUpToken));
            bool agreed = stop.Agree(1);
            return (Cancelled: cancelled, stop.GaveUp, Error: error, Agreed: agreed);
        });
        Check(gaveUp[1].Cancelled && gaveUp[1].GaveUp is { } why && why.Contains("IDRAK_CLUSTER_STOP_WAIT", StringComparison.Ordinal) && gaveUp[1].Error is TimeoutException,
            $"rank 1 gave up after 0.2 s: its token cancelled, a TimeoutException naming the setting: {gaveUp[1].GaveUp}");
        Check(!gaveUp[0].Cancelled && gaveUp[0].GaveUp is null && gaveUp.All(g => g.Agreed), "rank 0 did not give up; the ranks still agreed at the next boundary");
    }

    // Cancels a token at a step's end (after the batch of step `at` on one rank), and slows a rank down.
    private sealed class StopAtBatch(CancellationTokenSource? cancel, long at, TimeSpan slow) : ITrainerCallback
    {
        public void OnBatchEnd(TrainerContext context, BatchCompleted batch)
        {
            if (slow > TimeSpan.Zero)
            {
                Thread.Sleep(slow);
            }

            if (cancel is not null && batch.Step == at)
            {
                cancel.Cancel();
            }
        }
    }

    private static void StopTrainer(Device device)
    {
        string folder = TempFolder("cluster-stop-trainer");
        try
        {
            // The Trainer: 4 epochs of the spirals, 5 steps an epoch over 2 ranks, Adam.
            var data = Spirals(30, 3, 1);
            (float[] Weights, double[] Losses) Fit(ClusterRank rank, ClusterOptions options, string? stopFolder, string? resumeFrom, long cancelAt = -1, bool slow = false)
            {
                using var cancel = new CancellationTokenSource();
                using var model = ShardModel(rank.Device);
                using var optimizer = new Adam(model.Parameters(), 0.01f);
                var trainer = new Trainer(model, optimizer, Losses.CrossEntropy) { Cluster = rank.Collectives, ClusterOptions = options, StopFolder = stopFolder, ResumeFrom = resumeFrom };
                trainer.Callbacks.Add(new StopAtBatch(rank.Rank == 1 && cancelAt >= 0 ? cancel : null, cancelAt, slow && rank.Rank == 0 ? TimeSpan.FromMilliseconds(40) : TimeSpan.Zero));
                var history = trainer.Fit(new DataLoader(data, 10, shuffle: true, device: rank.Device, seed: 7), 4, cancellationToken: cancel.Token);
                return ([.. model.Parameters().SelectMany(p => p.ToArray())], [.. history.Epochs.Select(e => e.Loss)]);
            }

            foreach (var (name, slow) in new[] { (StopAgreements.Automatic, false), ("all-reduce", true) })
            {
                var options = StopOptions(name);
                string stops = Path.Combine(folder, $"stops-{name}");
                string what = $"Trainer, {name}{(slow ? ", rank 0 slow" : "")}";
                long step = AgreedStop(StopRun(device, 2, options, rank => Fit(rank, options, stops, null, cancelAt: 7, slow), what), what);
                // Asked after 7 steps (their reductions done): all-reduce carries the flag in the next step's header and stops before
                // step 8; shared stops after the last step any rank began (before step 7 or 8).
                Check(name == "all-reduce" ? step == 8 : step is 7 or 8, $"{what}: rank 1 asked after step 7, the ranks stopped before step {step}");
                string checkpoint = Path.Combine(stops, $"checkpoint-{step}");
                Check(File.Exists(Path.Combine(checkpoint, "train-state.json")) && Directory.GetDirectories(stops).Length == 1, $"{what}: one checkpoint, {checkpoint}");
                var resumed = DpRun(DpTopology(device, 2), options, rank => Fit(rank, options, null, checkpoint));
                var straight = DpRun(DpTopology(device, 2), options, rank => Fit(rank, options, null, null));
                Check(SameBits(resumed[0].Weights, straight[0].Weights) && SameBits(resumed[1].Weights, straight[1].Weights),
                    $"{what}: resumed from the stop, the weights after 4 epochs are the uninterrupted run's bit for bit (largest gap {Gap(resumed[0].Weights, straight[0].Weights):G3})");
            }

            // Unstopped, on 3 ranks (ring: a header slice per rank): the carried flag changes no result.
            var shared = DpRun(DpTopology(device, 3), StopOptions(), rank => Fit(rank, StopOptions(), null, null));
            var reduced = DpRun(DpTopology(device, 3), StopOptions("all-reduce"), rank => Fit(rank, StopOptions("all-reduce"), null, null));
            Check(Enumerable.Range(0, 3).All(r => SameBits(shared[r].Weights, reduced[r].Weights) && shared[r].Losses.SequenceEqual(reduced[r].Losses)),
                $"3 ranks unstopped: the same weights and losses bit for bit with the flag carried (all-reduce) and without (shared): {string.Join(", ", shared[0].Losses.Select(l => l.ToString("G6", CultureInfo.InvariantCulture)))}");

            // One process: a cancelled token ends the loop and throws, as before (no agreement).
            using var alone = ShardModel(device);
            using var adam = new Adam(alone.Parameters(), 0.01f);
            using var cancelled = new CancellationTokenSource();
            cancelled.Cancel();
            try
            {
                new Trainer(alone, adam, Losses.CrossEntropy).Fit(new DataLoader(data, 10, shuffle: true, device: device, seed: 7), 2, cancellationToken: cancelled.Token);
                Check(false, "one process: a cancelled Fit throws");
            }
            catch (OperationCanceledException e)
            {
                Check(e is not ClusterRunStoppedException, $"one process: a plain cancellation, as before ({e.GetType().Name})");
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void StopFineTuner(Device device)
    {
        string folder = TempFolder("cluster-stop-tune");
        try
        {
            // LoRA on the tiny model, 2 epochs of 16 windows: 4 steps an epoch on 2 ranks.
            var tokens = File.ReadAllBytes(PretrainData("tokens.bin"));
            var sequences = Enumerable.Range(0, 16).Select(w => new TrainingSequence(
                [.. Enumerable.Range(0, 24).Select(i => (int)BitConverter.ToUInt16(tokens, ((w * 24) + i) * 2))], [.. Enumerable.Repeat(true, 24)])).ToList();
            FineTuningOptions Tuning(ClusterRank? rank, ClusterOptions? options, string? resumeFrom) => new()
            {
                Rank = 4, Alpha = 8, Epochs = 2, MaxLength = 32, BatchTokens = 48, CudaGraphs = false, Seed = 5, Cluster = rank?.Collectives, ClusterOptions = options, ResumeFrom = resumeFrom,
            };

            // On rank 1: cancelled inside a step (its trace line of a batch of `inside`), or at a step's end (its progress report of `after`).
            float[] Tune(ClusterRank rank, ClusterOptions options, string output, string? resumeFrom = null, string? inside = null, int after = 0)
            {
                using var cancel = new CancellationTokenSource();
                using var model = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = rank.Device });
                bool asker = rank.Rank == 1;
                void Trace(string line)
                {
                    if (asker && inside is not null && line.StartsWith(inside, StringComparison.Ordinal))
                    {
                        cancel.Cancel();
                    }
                }

                var progress = new SynchronousProgress<FineTuningProgress>(p =>
                {
                    if (asker && after > 0 && p.Step == after)
                    {
                        cancel.Cancel();
                    }
                });
                FineTuner.Train(model, sequences, null, Tuning(rank, options, resumeFrom), output, progress, cancel.Token, Trace);
                return [.. model.Network.Parameters().Where(p => p.RequiresGrad).SelectMany(p => p.ToArray())];
            }

            float[]? reference = null;
            var straights = new Dictionary<string, float[][]>(StringComparer.Ordinal);
            foreach (var (name, inside, after) in new[] { (StopAgreements.Automatic, "step 3/", 0), ("all-reduce", "step 3/", 0), (StopAgreements.Automatic, (string?)null, 4), ("all-reduce", null, 4) })
            {
                var options = StopOptions(name);
                string run = Path.Combine(folder, $"{name}-{inside?.Trim('/', ' ') ?? $"after-{after}"}");
                string what = $"fine-tuner, {name}, cancelled {(inside is not null ? "inside step 3" : $"after step {after}")}";
                long step = AgreedStop(StopRun(device, 2, options, rank => Tune(rank, options, Path.Combine(run, $"rank-{rank.Rank}"), inside: inside, after: after), what), what);
                // Asked inside step 3 (index 2, before its reduction): every agreement stops before step index 3. Asked after step 4's
                // end: all-reduce carries it in the next step's header (stops before 5), shared after the last step begun (4 or 5).
                bool expected = inside is not null ? step == 3 : name == "all-reduce" ? step == after + 1 : step == after || step == after + 1;
                Check(expected, $"{what}: the ranks stopped before step {step}");
                string checkpoint = Path.Combine(run, "rank-0", $"checkpoint-{step}");
                Check(File.Exists(Path.Combine(checkpoint, "tune-state.json")) && Directory.GetDirectories(Path.Combine(run, "rank-0"), "checkpoint-*").Length == 1
                      && (!Directory.Exists(Path.Combine(run, "rank-1")) || Directory.GetDirectories(Path.Combine(run, "rank-1"), "checkpoint-*").Length == 0),
                    $"{what}: rank 0 alone wrote one checkpoint, {checkpoint}");
                var resumed = DpRun(DpTopology(device, 2), options, rank => Tune(rank, options, Path.Combine(run, $"resumed-{rank.Rank}"), checkpoint));
                if (!straights.TryGetValue(name, out var straight))
                {
                    straights[name] = straight = DpRun(DpTopology(device, 2), options, rank => Tune(rank, options, Path.Combine(run, $"straight-{rank.Rank}")));
                }

                Check(resumed[0].Length > 0 && SameBits(resumed[0], straight[0]) && SameBits(resumed[1], straight[0]),
                    $"{what}: resumed from the stop, the adapters are the uninterrupted run's bit for bit (largest gap {Gap(resumed[0], straight[0]):G3})");
                reference ??= straight[0];
                Check(SameBits(straight[0], reference), $"{what}: unstopped, the adapters are the same with either agreement (the carried flag changes no result)");
            }

            // One process: a cancelled token throws a plain cancellation before the first step, as before.
            using var alone = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = device });
            using var cancelled = new CancellationTokenSource();
            cancelled.Cancel();
            try
            {
                FineTuner.Train(alone, sequences, null, Tuning(null, null, null), null, null, cancelled.Token);
                Check(false, "one process: a cancelled tune throws");
            }
            catch (OperationCanceledException e)
            {
                Check(e is not ClusterRunStoppedException, $"one process: a plain cancellation, as before ({e.GetType().Name})");
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void StopPretrainer(Device device)
    {
        string folder = TempFolder("cluster-stop-pretrain");
        try
        {
            string corpus = PretrainData("tokens.bin");
            (float[] Losses, string? Last) Pretrain(ClusterRank rank, ClusterOptions options, int steps, string run, string? from = null, int cancelAfter = 0)
            {
                using var cancel = new CancellationTokenSource();
                using var model = PretrainedModel.Load(from ?? TinyLlama, new PretrainedOptions { Device = rank.Device });
                var progress = new SynchronousProgress<PretrainingProgress>(p =>
                {
                    if (rank.Rank == 1 && cancelAfter > 0 && p.Step == cancelAfter)
                    {
                        cancel.Cancel();
                    }
                });
                var settings = DpPretraining(corpus, steps, rank.Collectives, options) with { Packing = false };
                var result = Pretrainer.Train(model, settings, run, from is null ? null : Pretrainer.ReadCheckpoint(from), progress, cancel.Token);
                return ([.. result.Losses], result.LastCheckpoint);
            }

            // Data parallel: rank 1 asks after step 3.
            var options = StopOptions();
            string stopped = Path.Combine(folder, "stopped");
            long step = AgreedStop(StopRun(device, 2, options, rank => Pretrain(rank, options, 8, Path.Combine(stopped, $"rank-{rank.Rank}"), cancelAfter: 3), "pretrainer"), "pretrainer");
            Check(step is 3 or 4, $"pretrainer: rank 1 asked after step 3, the ranks stopped before step {step}");
            string checkpoint = Path.Combine(stopped, "rank-0", $"checkpoint-{step}");
            Check(File.Exists(Path.Combine(checkpoint, "pretrain-state.json")) && Pretrainer.LatestCheckpoint(Path.Combine(stopped, "rank-0")) == checkpoint,
                $"pretrainer: rank 0's last checkpoint is the stop's, {checkpoint} ({Pretrainer.LatestCheckpoint(Path.Combine(stopped, "rank-0"))})");
            var resumed = DpRun(DpTopology(device, 2), options, rank => Pretrain(rank, options, 8, Path.Combine(folder, $"resumed-{rank.Rank}"), checkpoint));
            var straight = DpRun(DpTopology(device, 2), options, rank => Pretrain(rank, options, 8, Path.Combine(folder, $"straight-{rank.Rank}")));
            Check(resumed[0].Losses.SequenceEqual(straight[0].Losses), $"pretrainer: the resumed run's losses are the uninterrupted run's ({string.Join(", ", resumed[0].Losses)})");

            // 2 pipeline stages: no collective of every rank carries the flag; the boundary's own exchange (or the shared memory) agrees.
            foreach (string name in new[] { StopAgreements.Automatic, "all-reduce" })
            {
                var pipe = PipeOptions() with { StopAgreement = name, Timeout = TimeSpan.FromMinutes(10) };
                string piped = Path.Combine(folder, $"pipeline-{name}");
                string what = $"pipeline, {name}";
                long at = AgreedStop(StopRun(device, 2, pipe, rank => Pretrain(rank, pipe, 20, piped, cancelAfter: 2), what), what);
                Check((name == "all-reduce" ? at == 2 : at is 2 or 3) && File.Exists(Path.Combine(piped, $"checkpoint-{at}", "pretrain-state.json")), $"{what}: both stages stopped before step {at}, its checkpoint written");
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    // The language-model policy, its acts counted (a cancellation raised during a rollout); its ranks and agreement forwarded.
    private sealed class StopPolicy(LanguageModelPolicy inner, Action<int>? acted) : IPolicy<ChatObservation, ChatMessage>, IRolloutRanks<ChatObservation, ChatMessage>
    {
        private int _acts;

        public int Rank => inner.Rank;

        public int WorldSize => inner.WorldSize;

        public IStopAgreement? StopAgreement => inner.StopAgreement;

        public ValueTask<PolicyActions<ChatMessage>> ActAsync(IReadOnlyList<ChatObservation> observations, ActionSampling sampling, CancellationToken cancellationToken = default)
        {
            acted?.Invoke(++_acts);
            return inner.ActAsync(observations, sampling, cancellationToken);
        }

        public IReadOnlyList<Trajectory<ChatObservation, ChatMessage>> Gather(IReadOnlyList<Trajectory<ChatObservation, ChatMessage>> mine, CancellationToken cancellationToken = default) =>
            inner.Gather(mine, cancellationToken);

        public UpdateStats Update(RolloutBatch<ChatObservation, ChatMessage> batch, IPolicyLoss loss, IReadOnlyList<int>? episodes = null, IReadOnlyDictionary<string, string>? lossOptions = null) =>
            inner.Update(batch, loss, episodes, lossOptions);

        public void Save(string path) => inner.Save(path);

        public void Load(string path) => inner.Load(path);

        public void Dispose() => inner.Dispose();
    }

    private static void StopReinforcement(Device device)
    {
        string folder = WriteChatModelWith(TinyChatSpec(), ChatMlTemplate);
        try
        {
            var kind = new PromptsEnvironmentKind([new PromptTask([new ChatMessage("user", "Q")], "yes"), new PromptTask([new ChatMessage("user", "P")], "no")], name: "cluster-stop-prompts");
            const int Iterations = 6;
            var tuning = SessionOptions(1e-2f) with { BatchTokens = 64 };
            // On rank 1: cancelled after iteration `after` (between iterations), or at its act number `act` (during a rollout).
            (ReinforcementResult Result, float[] Adapters) Run(ClusterRank rank, ClusterOptions options, string checkpoints, bool resume = false, int after = 0, int act = 0)
            {
                using var cancel = new CancellationTokenSource();
                using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = rank.Device });
                var session = new FineTuningSession(model, tuning with { Cluster = rank.Collectives, ClusterOptions = options }, totalSteps: 100);
                bool asker = rank.Rank == 1;
                using var policy = new StopPolicy(new LanguageModelPolicy(session, new LanguageModelPolicyOptions { MaxNewTokens = 4 }), n =>
                {
                    if (asker && act > 0 && n == act)
                    {
                        cancel.Cancel();
                    }
                });
                var trainer = new ReinforcementTrainer<ChatObservation, ChatMessage>(kind, policy, new ReinforcementOptions
                {
                    Algorithm = "grpo", Estimator = "group", GroupSize = 4, EpisodesPerIteration = 8, Iterations = Iterations, Epochs = 1, Seed = 3, Environments = 1,
                    ParallelSteps = false, Shards = 2, CheckpointFolder = checkpoints, Resume = resume ? checkpoints : null,
                    EnvironmentOptions = new Dictionary<string, string> { ["reward"] = "metric", ["metric.metric"] = "exact" },
                })
                {
                    OnIteration = s =>
                    {
                        if (asker && after > 0 && s.Iteration == after)
                        {
                            cancel.Cancel();
                        }
                    },
                };
                var result = trainer.Train(cancel.Token);
                return (result, TrainedValues(model));
            }

            static string[] Saved(string path) => Directory.Exists(path) ? [.. Directory.GetDirectories(path).Select(d => Path.GetFileName(d)).Order(StringComparer.Ordinal)] : [];
            float[]? straight = null;
            foreach (var (name, after, act) in new[] { (StopAgreements.Automatic, 2, 0), ("all-reduce", 0, 5) })
            {
                var options = StopOptions(name);
                string checkpoints = Path.Combine(folder, $"checkpoints-{name}-{after}-{act}");
                string what = $"reinforcement, {name}, cancelled {(after > 0 ? $"after iteration {after}" : $"at rank 1's act {act} (inside a rollout)")}";
                var results = new (ReinforcementResult Result, float[] Adapters)[2];
                var run = StopRun(device, 2, options, rank => results[rank.Rank] = Run(rank, options, checkpoints, after: after, act: act), what);
                run.ThrowIfFailed();
                int stoppedAt = results[0].Result.Iterations;
                Check(results.All(r => r.Result.Cancelled && r.Result.Iterations == stoppedAt) && stoppedAt is > 0 and < Iterations,
                    $"{what}: every rank stopped after the same iteration: {string.Join(", ", results.Select(r => $"{r.Result.Iterations} (cancelled {r.Result.Cancelled})"))}");
                string last = $"iteration-{stoppedAt:D6}";
                Check(Saved(checkpoints).SequenceEqual([last]) && File.Exists(Path.Combine(checkpoints, last, "state.json")), $"{what}: one checkpoint, {last}: {string.Join(", ", Saved(checkpoints))}");
                var resumed = DpRun(DpTopology(device, 2), options, rank => Run(rank, options, checkpoints, resume: true));
                Check(resumed.All(r => !r.Result.Cancelled && r.Result.Iterations == Iterations && r.Result.History.Count == Iterations - stoppedAt),
                    $"{what}: the resumed run went on from iteration {stoppedAt + 1} to {Iterations}");
                straight ??= DpRun(DpTopology(device, 2), options, rank => Run(rank, options, Path.Combine(folder, $"straight-{name}")))[0].Adapters;
                Check(SameBits(resumed[0].Adapters, straight) && SameBits(resumed[1].Adapters, straight),
                    $"{what}: resumed, the adapters are the uninterrupted run's bit for bit (largest gap {Gap(resumed[0].Adapters, straight):G3})");
            }

            // One process: a cancelled token stops before the first iteration, as before.
            using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
            using var policy = new LanguageModelPolicy(new FineTuningSession(model, tuning, totalSteps: 10), new LanguageModelPolicyOptions { MaxNewTokens = 4 });
            Check(policy.StopAgreement is null, "one process: the policy has no agreement");
            using var cancelled = new CancellationTokenSource();
            cancelled.Cancel();
            var alone = new ReinforcementTrainer<ChatObservation, ChatMessage>(kind, policy, new ReinforcementOptions
            {
                Algorithm = "grpo", GroupSize = 4, EpisodesPerIteration = 8, Iterations = 2, Seed = 3,
                EnvironmentOptions = new Dictionary<string, string> { ["reward"] = "metric", ["metric.metric"] = "exact" },
            }).Train(cancelled.Token);
            Check(alone.Cancelled && alone.Iterations == 0, $"one process: cancelled before the first iteration ({alone.Iterations})");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void StopCliReinforce(Device device)
    {
        string folder = TempFolder("cluster-stop-cli");
        var (model, data) = ReinforceFixture(folder);
        try
        {
            string topology = Path.Combine(folder, "two.json");
            File.WriteAllText(topology, DpTopology(device, 2).ToJson());
            string output = Path.Combine(folder, "ranks");
            string[] Reinforce(int iterations, params string[] more) => ["reinforce", model, "--env", "prompts", "--data", data, "--reward", "cer",
                "--algorithm", "grpo", "--group", "4", "--episodes", "8", "--iterations", iterations.ToString(CultureInfo.InvariantCulture), "--seed", "3", "-o", output,
                "--shards", "2", .. TinySession(device, folder), .. more];
            var variables = new Dictionary<string, string?> { ["IDRAK_RL_ENVIRONMENTS"] = "1", ["IDRAK_RL_PARALLEL_STEPS"] = "off", ["IDRAK_CLUSTER_TIMEOUT"] = "600" };
            (int Code, string Out, string Err) run = (0, "", "");
            var watch = Stopwatch.StartNew();
            WithEnvironment(variables, () => run = ClusterCli(["cluster", "sim", "--topology", topology, "--", .. Reinforce(1000, "--timeout", "4s")]));
            Check(run.Code == 0 && run.Out.Contains("rank 1", StringComparison.Ordinal) && watch.Elapsed < StopBound,
                $"every rank ended after its --timeout (exit {run.Code}, {watch.Elapsed.TotalSeconds:F1} s; the collectives' timeout 600 s)\n{run.Out}\n{run.Err}");
            var stopped = run.Out.Split('\n').Where(l => l.StartsWith("stopped after ", StringComparison.Ordinal)).ToList();
            Check(stopped.Count == 1, $"rank 0 alone says it stopped: {run.Out}");
            int iterations = int.Parse(stopped[0]["stopped after ".Length..].Split(' ')[0], CultureInfo.InvariantCulture);
            Check(iterations is > 0 and < 1000 && File.Exists(Path.Combine(output, "adapter_model.safetensors")),
                $"the ranks stopped after {iterations} iterations, the adapter written: {stopped[0]}");
            string[] saved = [.. Directory.GetDirectories(Path.Combine(output, "checkpoints")).Select(d => Path.GetFileName(d)).Order(StringComparer.Ordinal)];
            Check(saved.SequenceEqual([$"iteration-{iterations:D6}"]), $"one checkpoint, the stop's: {string.Join(", ", saved)}");

            // --resume goes on from it for one more iteration.
            WithEnvironment(variables, () => run = ClusterCli(["cluster", "sim", "--topology", topology, "--", .. Reinforce(iterations + 1, "--resume")]));
            string next = $"iteration {iterations + 1}/{iterations + 1}: reward";
            Check(run.Code == 0 && run.Out.Split('\n').Count(l => l.StartsWith(next, StringComparison.Ordinal)) == 1 && run.Out.Contains($"trained {iterations + 1} iterations", StringComparison.Ordinal),
                $"--resume goes on at iteration {iterations + 1}: exit {run.Code}\n{run.Out}\n{run.Err}");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
            Directory.Delete(model, recursive: true);
        }
    }

    private static void StopCost(Device device)
    {
        var inv = CultureInfo.InvariantCulture;
        string Us(TimeSpan t, int n) => (t.TotalMilliseconds * 1000 / n).ToString("F1", inv);
        // Two ranks on one device (the same-device link) and on two nodes joined by ethernet-1g (the virtual clock charges its
        // latency and bandwidth).
        var topologies = new (string Name, ClusterTopology Topology)[] { ("same device", DpTopology(device, 2)), ("2 nodes, ethernet-1g", NodesOf([1, 1], device.ToString(), "ethernet-1g")) };

        // The boundary's own exchange: 200 boundaries, nothing carried, an all-reduce of one value each.
        const int Boundaries = 200;
        var lines = new List<string>();
        foreach (var (name, topology) in topologies)
        {
            var exchange = DpRun(topology, StopOptions("all-reduce"), rank =>
            {
                var stop = (AllReduceStopAgreement)StopAgreements.For(rank.Collectives, StopOptions("all-reduce"));
                stop.Reset();
                for (int b = 0; b < 20; b++)
                {
                    _ = stop.Agree(b);                           // warm
                }

                var before = rank.Transport.SimulatedTime;
                long exchanges = stop.Exchanges;
                var watch = Stopwatch.StartNew();
                for (int b = 0; b < Boundaries; b++)
                {
                    _ = stop.Agree(b);
                }

                return (Wall: watch.Elapsed, Simulated: rank.Transport.SimulatedTime - before, Exchanges: stop.Exchanges - exchanges);
            });
            Check(exchange.All(e => e.Exchanges == Boundaries), $"{name}: every boundary exchanged when nothing carried the flag ({exchange[0].Exchanges})");
            lines.Add($"{name}: {Us(exchange[0].Wall, Boundaries)} us wall, {Us(exchange[0].Simulated, Boundaries)} us on the virtual clock");
        }

        // The carried flag: the Trainer's 20 steps on 2 ranks with all-reduce (4 bytes more in each step's header) and shared
        // (no flag), each run 3 times in turn after a warm run; the wall's median, the virtual clock (deterministic).
        var data = Spirals(30, 3, 1);
        (TimeSpan Wall, TimeSpan Simulated, long Exchanges) Fit(ClusterRank rank, ClusterOptions options)
        {
            using var model = ShardModel(rank.Device);
            using var optimizer = new Adam(model.Parameters(), 0.01f);
            var trainer = new Trainer(model, optimizer, Losses.CrossEntropy) { Cluster = rank.Collectives, ClusterOptions = options };
            var watch = Stopwatch.StartNew();
            trainer.Fit(new DataLoader(data, 10, shuffle: true, device: rank.Device, seed: 7), 4);
            long exchanges = StopAgreements.For(rank.Collectives) is AllReduceStopAgreement a ? a.Exchanges : 0;
            return (watch.Elapsed, rank.Transport.SimulatedTime, exchanges);
        }

        const int Steps = 20;
        var cost = new List<string>();
        foreach (var (name, topology) in topologies)
        {
            var walls = new Dictionary<string, List<TimeSpan>>(StringComparer.Ordinal) { ["shared"] = [], ["all-reduce"] = [] };
            var clock = new Dictionary<string, TimeSpan>(StringComparer.Ordinal);
            _ = DpRun(topology, StopOptions(), rank => Fit(rank, StopOptions()));    // warm
            for (int round = 0; round < 3; round++)
            {
                foreach (string agreement in new[] { "shared", "all-reduce" })
                {
                    var options = StopOptions(agreement);
                    var ranks = DpRun(topology, options, rank => Fit(rank, options));
                    walls[agreement].Add(ranks[0].Wall);
                    clock[agreement] = ranks[0].Simulated;
                    Check(agreement == "shared" || ranks.All(c => c.Exchanges == 1),
                        $"{name}: the Trainer's steps carry the flag in the gradient header: one exchange in {Steps} steps (the first boundary's), {ranks[0].Exchanges}");
                }
            }

            TimeSpan Median(List<TimeSpan> values) => values.Order().ElementAt(values.Count / 2);
            cost.Add($"{name}: network time on the virtual clock {Us(clock["shared"], Steps)} us a step without the flag, {Us(clock["all-reduce"], Steps)} us with it "
                     + $"(+{Us(clock["all-reduce"] - clock["shared"], Steps)} us, the first boundary's exchange included); wall median {Us(Median(walls["shared"]), Steps)} us and "
                     + $"{Us(Median(walls["all-reduce"]), Steps)} us a step");
        }

        Console.WriteLine($"    measured: the boundary's own one-value all-reduce on 2 in-process ranks, a boundary: {string.Join("; ", lines)}");
        Console.WriteLine($"    measured: the Trainer's {Steps} steps on 2 ranks (spirals), shared (no flag) against all-reduce (the flag carried): {string.Join("; ", cost)}");
    }
}
