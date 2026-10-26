// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Abstraction.Cluster;
using Idrak.Abstraction;
using Idrak.Abstraction.Generation;
using Idrak.Abstraction.Reinforcement;
using Idrak.Cli.Commands.Cluster;
using Idrak.Cluster;
using Idrak.Models;
using Idrak.Reinforcement;
using Idrak.Reinforcement.Plugins.Classic;
using Idrak.Text;
using Idrak.Text.Reinforcement;

// Plan 27, R11: the cluster dashboard steers reinforce. A trainer's iteration (or an off-policy round) is the control point:
// the dashboard's in-process control or a launcher's control file holds every rank at the same boundary while paused (a
// resume goes on bit for bit) and stops every rank there through the ranks' agreement, with one checkpoint. In-process ranks
// and in-process command runs only; every wait is bounded and every run awaited far within the collectives' timeout.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] ClusterControlGroup =
    [
        ("cluster control: reinforce: the run steerings (in-process from Idrak.Cluster, then control-file from Idrak.Network) and the pass rate's setting; nothing steers a run without a control; one steering per rank's collectives, whose step gate then lets every step through while the steering holds the rank; in one process the control file pauses, goes on and stops at a boundary (read at most every poll, a cancellation ending a pause), rank 0's state file saying so", ControlRegistry),
        ("cluster control: reinforce on 2 in-process ranks, steered by the run's control (the dashboard's in-process run) and by a control file (the all-reduce agreement): started paused and paused again at an iteration boundary with every rank there, then resumed, the run ends with the uninterrupted run's adapters bit for bit; stopped, every rank ends at the same iteration, rank 0 writes one checkpoint, the iterations before the stop are the uninterrupted run's bit for bit and the run resumed from it ends within R3b's 1e-4 of its adapters", ControlReinforceRanks),
        ("cluster control: reinforce in one process through a control file: the language model's run, the numeric path's (ppo on cartpole) and dqn's (cartpole, at a round) pause and go on bit for bit and stop with one checkpoint; resumed, ppo and dqn end with the uninterrupted run's weights bit for bit, the language model within 1e-4; idrak reinforce honours IDRAK_CLUSTER_CONTROL on both paths and writes its log to DIR/log.jsonl", ControlReinforceOneProcess),
        ("cluster control: reinforce started from the dashboard on 2 in-process ranks: the run listing shows its iterations, reward and pass rate from its JSON log and the events stream them; paused before iteration 0 and at a later iteration, resumed and stopped from the API: one checkpoint and the adapter so far, its log the uninterrupted run's line for line, and cluster sim --resume goes on to within 1e-4 of the uninterrupted run's adapter", ControlReinforceDashboard),
    ];

    // A resumed language-model run against the uninterrupted one: the session's packing order (FineTuner's tie order and bin
    // shuffle, drawn from the session's own Random) is not in a reinforcement checkpoint, so a resumed run packs its later steps
    // in another order: the same values within R3b's tolerance, not bit for bit (a finding of R11, beside its steering).
    private const float ResumeTolerance = 1e-4f;

    private static float[] ControlAdapter(string folder)
    {
        using var reader = SafeTensorsReader.Open(Path.Combine(folder, "adapter_model.safetensors"));
        return [.. reader.Tensors.Keys.Order(StringComparer.Ordinal).SelectMany(n => reader.Read(n))];
    }

    private static string[] ControlSaved(string path) => Directory.Exists(path) ? [.. Directory.GetDirectories(path).Select(d => Path.GetFileName(d)).Order(StringComparer.Ordinal)] : [];

    private static bool ControlState(string file, string state, long step) => ClusterControlFile.ReadState(file) is { } said && said.State == state && said.Step == step;

    // A control file's helper for a run in one process: it waits for the run to pause before boundary 0, checks nothing ran,
    // lets it go; then waits for the pause the run asks at boundary `at` (its callback writes "pause"), checks the run did `at`
    // and no more while held, and lets it go again. True when both pauses held as they should.
    private static Task<string?> ControlPauses(string file, Func<long> done, long at) => Task.Run(() =>
    {
        if (!UiWait(() => ControlState(file, "paused", 0), 120))
        {
            return $"no pause before boundary 0 ({ClusterControlFile.ReadState(file)})";
        }

        Thread.Sleep(200);
        if (done() != 0)
        {
            return $"{done()} done while paused before the first boundary";
        }

        ClusterControlFile.Write(file, ClusterControlCommand.Run);
        if (!UiWait(() => ControlState(file, "paused", at), 120))
        {
            return $"no pause at boundary {at} ({ClusterControlFile.ReadState(file)})";
        }

        Thread.Sleep(200);
        long held = done();
        ClusterControlFile.Write(file, ClusterControlCommand.Run);
        return held == at ? null : $"{held} done while paused at boundary {at}";
    });

    private static void ControlRegistry(Device device)
    {
        if (!FirstRun(nameof(ControlRegistry)))
        {
            return;
        }

        Check(RunSteerings.Names.SequenceEqual(["in-process", "control-file"]) && RunSteerings.Origin("in-process") == Overrides.Library
              && RunSteerings.Origin("control-file") == Overrides.Library && RunSteerings.Describe().Contains("control-file:", StringComparison.Ordinal),
            $"the library registers in-process, then control-file: {RunSteerings.Describe()}");
        ThrowsNaming<NotSupportedException>(() => RunSteerings.Get("no-such-steering"), "an unknown steering", "is not registered", "RunSteerings.Register");
        Check(RunSteerings.For(null, new ClusterOptions()) is null, "one process without a control file: nothing steers the run");

        // The pass rate's setting.
        Check(new ReinforcementOptions().PassReturn == 1f, "an episode passes at a return of 1 by default");
        WithEnvironment(new Dictionary<string, string?> { [ReinforcementOptions.PassReturnVariable] = "0.5" }, () => Check(ReinforcementOptions.FromEnvironment().PassReturn == 0.5f, "IDRAK_RL_PASS_RETURN=0.5"));
        WithEnvironment(new Dictionary<string, string?> { [ReinforcementOptions.PassReturnVariable] = "none" }, () => Check(ReinforcementOptions.FromEnvironment().PassReturn is null, "IDRAK_RL_PASS_RETURN=none: no pass rate"));
        WithEnvironment(new Dictionary<string, string?> { [ReinforcementOptions.PassReturnVariable] = "most" },
            () => ThrowsNaming<ArgumentException>(() => ReinforcementOptions.FromEnvironment(), "a pass return that is not a number", "IDRAK_RL_PASS_RETURN=most"));

        // Simulated ranks without a control: nothing steers them. With the run's control: an in-process steering per rank's
        // collectives, the same one again; the rank's step gate lets step 0 through although the run is paused before it, and
        // the steering holds the rank at its boundary until the stop, which it says on every rank.
        var bare = DpRun(DpTopology(device, 2), DpOptions(), rank => RunSteerings.For(rank.Collectives) is null && !RunSteerings.Steered(rank.Collectives));
        Check(bare.All(b => b), "a simulated run without a control: nothing steers it, its step gates are its own");
        var control = new ClusterRunControl();
        control.Pause();
        var names = new string?[2];
        var same = new bool[2];
        var began = new bool[2];
        var answers = new ClusterControlCommand[2];
        var running = Task.Run(() => InProcessCluster.Run(DpTopology(device, 2), DpOptions(), rank =>
        {
            var steering = RunSteerings.For(rank.Collectives)!;
            names[rank.Rank] = steering.Name;
            same[rank.Rank] = ReferenceEquals(steering, RunSteerings.For(rank.Collectives)) && RunSteerings.Steered(rank.Collectives);
            began[rank.Rank] = rank.BeginStep(0);                   // the step gate does not hold a steered rank
            answers[rank.Rank] = steering.Boundary(0);
        }, control));
        Check(UiWait(() => control.State == ClusterRunState.Paused, 60) && began.All(b => b) && control.LastStep == -1,
            $"both ranks began step 0 past the paused gate and wait at boundary 0 (state {control.State}, began {string.Join(", ", began)})");
        control.Stop();
        Check(running.Wait(TimeSpan.FromSeconds(60)), "the stop lets the held ranks go");
        running.Result.ThrowIfFailed();
        Check(names.All(n => n == "in-process") && same.All(s => s) && answers.All(a => a == ClusterControlCommand.Stop),
            $"in-process steering, one per rank, the stop said on every rank: {string.Join(", ", names)}; {string.Join(", ", answers)}");

        // One process: the control file.
        string folder = TempFolder("cluster-control");
        try
        {
            string file = Path.Combine(folder, "control");
            ClusterOptions Options(double poll) => new() { Simulation = new SimulationOptions { Control = file, ControlPoll = TimeSpan.FromSeconds(poll) } };
            var steering = RunSteerings.For(null, Options(0.01))!;
            Check(steering.Name == "control-file" && steering.Boundary(0) == ClusterControlCommand.Run && ClusterControlFile.ReadState(file) is null,
                "no file yet: the run goes on, nothing reported");
            ClusterControlFile.Write(file, ClusterControlCommand.Pause);
            Thread.Sleep(30);
            var resume = Task.Run(() =>
            {
                bool held = UiWait(() => ControlState(file, "paused", 1), 30);
                ClusterControlFile.Write(file, ClusterControlCommand.Run);
                return held;
            });
            Check(steering.Boundary(1) == ClusterControlCommand.Run && resume.Result && ControlState(file, "running", 1), "paused at boundary 1, rank 0 reports it, then goes on");
            ClusterControlFile.Write(file, ClusterControlCommand.Stop);
            Thread.Sleep(30);
            Check(steering.Boundary(2) == ClusterControlCommand.Stop && ControlState(file, "stopped", 2), "stopped at boundary 2, reported");
            Thread.Sleep(30);
            Check(steering.Stops(3, null) && !((IRunSteering?)null).Stops(3, null), "in one process a stop is the trainer's at once; nothing steering: no stop");

            // Read at most every poll: a stop written right after a read waits for the next.
            ClusterControlFile.Write(file, ClusterControlCommand.Run);
            var slow = RunSteerings.For(null, Options(5))!;
            Check(slow.Boundary(0) == ClusterControlCommand.Run, "the first boundary reads the file");
            ClusterControlFile.Write(file, ClusterControlCommand.Stop);
            Check(slow.Boundary(1) == ClusterControlCommand.Run, "within the poll the file is not read again");

            // A cancellation while paused ends the pause as a stop.
            ClusterControlFile.Write(file, ClusterControlCommand.Pause);
            using var cancel = new CancellationTokenSource(TimeSpan.FromMilliseconds(300));
            Check(RunSteerings.For(null, Options(0.05))!.Boundary(4, cancel.Token) == ClusterControlCommand.Stop && ControlState(file, "stopped", 4),
                "Ctrl+C while paused stops the run there");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void ControlReinforceRanks(Device device)
    {
        string folder = WriteChatModelWith(TinyChatSpec(), ChatMlTemplate);
        try
        {
            var kind = new PromptsEnvironmentKind([new PromptTask([new ChatMessage("user", "Q")], "yes"), new PromptTask([new ChatMessage("user", "P")], "no")], name: "cluster-control-prompts");
            const int Iterations = 6;
            var tuning = SessionOptions(1e-2f) with { BatchTokens = 64 };
            var done = new int[2];                                   // each rank's iterations finished, in the steered run
            // One rank's run, steered as its options and the run's control say; `lead` called on rank 0 after each iteration.
            (ReinforcementResult Result, float[] Adapters) Run(ClusterRank rank, ClusterOptions options, string checkpoints, bool resume = false, Action<IterationSummary>? lead = null)
            {
                using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = rank.Device });
                var session = new FineTuningSession(model, tuning with { Cluster = rank.Collectives, ClusterOptions = options }, totalSteps: 100);
                using var policy = new LanguageModelPolicy(session, new LanguageModelPolicyOptions { MaxNewTokens = 4 });
                var trainer = new ReinforcementTrainer<ChatObservation, ChatMessage>(kind, policy, new ReinforcementOptions
                {
                    // Two epochs of two mini-batches: four updates an iteration, so a stop at a gradient step would cut one.
                    Algorithm = "grpo", Estimator = "group", GroupSize = 4, EpisodesPerIteration = 8, Iterations = Iterations, Epochs = 2, MiniBatch = 4, Seed = 3,
                    Environments = 1, ParallelSteps = false, Shards = 2, CheckpointFolder = checkpoints, Resume = resume ? checkpoints : null,
                    EnvironmentOptions = new Dictionary<string, string> { ["reward"] = "cer" },     // a spread: the updates move the adapters
                })
                {
                    Steering = RunSteerings.For(rank.Collectives, options),
                    OnIteration = s =>
                    {
                        Interlocked.Increment(ref done[rank.Rank]);
                        if (rank.Rank == 0)
                        {
                            lead?.Invoke(s);
                        }
                    },
                };
                var result = trainer.Train();
                return (result, TrainedValues(model));
            }

            var plain = StopOptions();
            var straight = DpRun(DpTopology(device, 2), plain, rank => Run(rank, plain, Path.Combine(folder, "straight")));
            Check(straight[0].Result.History.Any(h => h.RewardSpread > 0), "the rewards differ within the groups, so the updates move the adapters");
            foreach (var (file, stops) in new[] { (false, false), (false, true), (true, true) })
            {
                string what = (file ? "a control file, all-reduce" : "the run's control, shared") + (stops ? ", stopped" : ", paused and resumed");
                string checkpoints = Path.Combine(folder, $"{(file ? "file" : "control")}-{(stops ? "stop" : "pause")}"), controlFile = Path.Combine(folder, "control-file");
                var options = file ? StopOptions("all-reduce") with { Simulation = plain.Simulation with { Control = controlFile, ControlPoll = TimeSpan.FromMilliseconds(20) } } : plain;
                var control = file ? null : new ClusterRunControl();
                Array.Clear(done);
                if (file)
                {
                    ClusterControlFile.Write(controlFile, ClusterControlCommand.Pause);
                    File.Delete(ClusterControlFile.StatePath(controlFile));
                }
                else
                {
                    control!.Pause();
                }

                // Rank 0 asks a pause after iteration 2 and (when the case stops) a stop after iteration 4, as the page would.
                void Lead(IterationSummary s)
                {
                    if (s.Iteration == 2 || (stops && s.Iteration == 4))
                    {
                        if (file)
                        {
                            ClusterControlFile.Write(controlFile, s.Iteration == 2 ? ClusterControlCommand.Pause : ClusterControlCommand.Stop);
                        }
                        else if (s.Iteration == 2)
                        {
                            control!.Pause();
                        }
                        else
                        {
                            control!.Stop();
                        }
                    }
                }

                var results = new (ReinforcementResult Result, float[] Adapters)[2];
                var running = Task.Run(() => InProcessCluster.Run(DpTopology(device, 2), options, rank => results[rank.Rank] = Run(rank, options, checkpoints, lead: Lead), control));
                bool PausedAt(Func<long, bool> at) => file ? ClusterControlFile.ReadState(controlFile) is { State: "paused" } said && at(said.Step)
                    : control!.State == ClusterRunState.Paused && control.PauseAt is { } pause && at(pause);
                void Resume()
                {
                    if (file)
                    {
                        ClusterControlFile.Write(controlFile, ClusterControlCommand.Run);
                    }
                    else
                    {
                        control!.Resume();
                    }
                }

                Check(UiWait(() => PausedAt(s => s == 0), 120), $"{what}: every rank waits before iteration 0");
                Thread.Sleep(200);
                Check(done.All(d => d == 0), $"{what}: no rank played an iteration while paused before the first");
                Resume();
                Check(UiWait(() => PausedAt(s => s > 0), 120), $"{what}: the run pauses at an iteration boundary");
                long at = file ? ClusterControlFile.ReadState(controlFile)!.Value.Step : control!.PauseAt!.Value;
                Thread.Sleep(300);
                Check(done.All(d => d == at) && (file ? at == 2 : control!.LastStep == at - 1),
                    $"{what}: every rank finished {at} iterations and none began the next while paused: {string.Join(", ", done)}");
                Resume();
                Check(running.Wait(StopBound), $"{what}: every rank ended within {StopBound.TotalSeconds:F0} s");
                running.Result.ThrowIfFailed();
                var steered = results[0].Result.History;
                var unsteered = straight[0].Result.History;
                if (!stops)
                {
                    Check(results.All(r => !r.Result.Cancelled && r.Result.Iterations == Iterations)
                          && steered.Zip(unsteered).All(p => p.First.Returns.SequenceEqual(p.Second.Returns) && p.First.Loss.Equals(p.Second.Loss))
                          && SameBits(results[0].Adapters, straight[0].Adapters) && SameBits(results[1].Adapters, straight[0].Adapters),
                        $"{what}: the run ends as the uninterrupted one, every iteration and the adapters bit for bit (largest gap {Gap(results[0].Adapters, straight[0].Adapters):G3})");
                    Console.WriteLine($"    {what}: paused before 0 and at {at}; the uninterrupted run's iterations and adapters bit for bit");
                    continue;
                }

                int stoppedAt = results[0].Result.Iterations;
                Check(results.All(r => r.Result.Cancelled && r.Result.Iterations == stoppedAt) && stoppedAt is >= 4 and < Iterations,
                    $"{what}: every rank stopped after the same iteration: {string.Join(", ", results.Select(r => $"{r.Result.Iterations} (cancelled {r.Result.Cancelled})"))}");
                Check(file ? stoppedAt == 4 && ControlState(controlFile, "stopped", 4) : stoppedAt == control!.StopAfter + 1,
                    $"{what}: the stop lands at the boundary after the iteration it was asked in ({stoppedAt}; {(file ? ClusterControlFile.ReadState(controlFile)?.ToString() : control!.StopAfter?.ToString(CultureInfo.InvariantCulture))})");
                string last = $"iteration-{stoppedAt:D6}";
                Check(ControlSaved(checkpoints).SequenceEqual([last]) && File.Exists(Path.Combine(checkpoints, last, "state.json")),
                    $"{what}: rank 0 wrote one checkpoint, {last}: {string.Join(", ", ControlSaved(checkpoints))}");
                Check(steered.Count == stoppedAt && steered.Zip(unsteered).All(p => p.First.Returns.SequenceEqual(p.Second.Returns) && p.First.Loss.Equals(p.Second.Loss)),
                    $"{what}: the iterations before the stop are the uninterrupted run's bit for bit, the pauses changed nothing");

                var resumed = DpRun(DpTopology(device, 2), plain, rank => Run(rank, plain, checkpoints, resume: true));
                Check(resumed.All(r => !r.Result.Cancelled && r.Result.Iterations == Iterations && r.Result.History.Count == Iterations - stoppedAt),
                    $"{what}: the resumed run went on from iteration {stoppedAt + 1} to {Iterations}");
                Check(SameBits(resumed[0].Adapters, resumed[1].Adapters), $"{what}: resumed, both ranks hold the same adapters");
                AssertClose(straight[0].Adapters, resumed[0].Adapters, ResumeTolerance, $"{what}: resumed, the adapters against the uninterrupted run's");
                Console.WriteLine($"    {what}: paused before 0 and at {at}, stopped after {stoppedAt} of {Iterations}; resumed, the adapters' largest gap to the uninterrupted run's {Gap(resumed[0].Adapters, straight[0].Adapters):E2}");
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void ControlReinforceOneProcess(Device device)
    {
        string root = TempFolder("cluster-control-one");
        string model = WriteChatModelWith(TinyChatSpec(), ChatMlTemplate);
        try
        {
            string file = Path.Combine(root, "control");
            var steering = new ClusterOptions { Simulation = new SimulationOptions { Control = file, ControlPoll = TimeSpan.FromTicks(1) } };   // read at every boundary

            // The language model's run in one process: paused before iteration 0 and after iteration 2, stopped after iteration 4.
            var kind = new PromptsEnvironmentKind([new PromptTask([new ChatMessage("user", "Q")], "yes"), new PromptTask([new ChatMessage("user", "P")], "no")], name: "control-one-prompts");
            var tuning = SessionOptions(1e-2f) with { BatchTokens = 64 };
            long done = 0;
            (ReinforcementResult Result, float[] Weights) Text(string checkpoints, bool resume, bool steered)
            {
                using var loaded = PretrainedModel.Load(model, new PretrainedOptions { Device = device });
                using var policy = new LanguageModelPolicy(new FineTuningSession(loaded, tuning, totalSteps: 100), new LanguageModelPolicyOptions { MaxNewTokens = 4 });
                var trainer = new ReinforcementTrainer<ChatObservation, ChatMessage>(kind, policy, new ReinforcementOptions
                {
                    Algorithm = "grpo", GroupSize = 4, EpisodesPerIteration = 8, Iterations = 6, Epochs = 2, MiniBatch = 4, Seed = 3, Environments = 1, ParallelSteps = false,
                    CheckpointFolder = checkpoints, Resume = resume ? checkpoints : null, EnvironmentOptions = new Dictionary<string, string> { ["reward"] = "cer" },
                })
                {
                    Steering = steered ? RunSteerings.For(null, steering) : null,
                    OnIteration = s => Steer(s.Iteration, 2, 4, steered),
                };
                return (trainer.Train(), TrainedValues(loaded));
            }

            // After `iteration`: a pause asked after `pause`, a stop after `stop`.
            void Steer(long iteration, long pause, long stop, bool steered)
            {
                Interlocked.Exchange(ref done, iteration);
                if (steered && (iteration == pause || iteration == stop))
                {
                    ClusterControlFile.Write(file, iteration == pause ? ClusterControlCommand.Pause : ClusterControlCommand.Stop);
                }
            }

            // The numeric path's policy: ppo on cartpole.
            var cartpole = ClassicControlPlugin.CartPoleKind;
            var (observations, actions) = PolicyNetworks.Spaces(cartpole);
            (ReinforcementResult Result, float[] Weights) Numbers(string checkpoints, bool resume, bool steered)
            {
                using var policy = PolicyNetworks.Categorical(observations, actions, new ModulePolicyOptions { Hidden = [16, 16], LearningRate = 3e-3f }, valueHead: true, device, seed: 7);
                var trainer = new ReinforcementTrainer<float[], int>(cartpole, policy, new ReinforcementOptions
                {
                    Algorithm = "ppo", EpisodesPerIteration = 8, MiniBatch = 4, Iterations = 6, Seed = 7, Environments = 1, ParallelSteps = false,
                    CheckpointFolder = checkpoints, Resume = resume ? checkpoints : null,
                })
                {
                    Steering = steered ? RunSteerings.For(null, steering) : null,
                    OnIteration = s => Steer(s.Iteration, 2, 4, steered),
                };
                return (trainer.Train(), [.. ModuleWeights(policy.Actor), .. ModuleWeights(policy.Critic!)]);
            }

            foreach (var (name, run, exact) in new (string, Func<string, bool, bool, (ReinforcementResult Result, float[] Weights)>, bool)[]
                     { ("the language model", Text, false), ("ppo on cartpole", Numbers, true) })
            {
                string Folder(string what) => Path.Combine(root, $"{name.Replace(' ', '-')}-{what}");
                var straight = run(Folder("straight"), false, false);
                ClusterControlFile.Write(file, ClusterControlCommand.Pause);
                File.Delete(ClusterControlFile.StatePath(file));
                Interlocked.Exchange(ref done, 0);
                var pauses = ControlPauses(file, () => Interlocked.Read(ref done), 2);
                var stopped = run(Folder("stopped"), false, true);
                Check(pauses.Wait(TimeSpan.FromSeconds(150)) && pauses.Result is null, $"{name}: {pauses.Result ?? "both pauses held"}");
                Check(stopped.Result.Cancelled && stopped.Result.Iterations == 4 && ControlState(file, "stopped", 4)
                      && ControlSaved(Folder("stopped")).SequenceEqual(["iteration-000004"]),
                    $"{name}: stopped after iteration 4 as the file asked, one checkpoint: {stopped.Result.Iterations}, {string.Join(", ", ControlSaved(Folder("stopped")))}");
                Check(straight.Result.History.Any(h => h.RewardSpread > 0) && stopped.Result.History.Zip(straight.Result.History)
                          .All(p => p.First.Returns.SequenceEqual(p.Second.Returns) && p.First.Loss.Equals(p.Second.Loss)),
                    $"{name}: the iterations before the stop are the uninterrupted run's bit for bit (the rewards spread, the weights move)");
                ClusterControlFile.Write(file, ClusterControlCommand.Run);
                var resumed = run(Folder("stopped"), true, false);
                Check(!resumed.Result.Cancelled && resumed.Result.Iterations == 6, $"{name}: resumed to iteration 6");
                if (exact)
                {
                    Check(SameBits(resumed.Weights, straight.Weights), $"{name}: resumed, the weights are the uninterrupted run's bit for bit (largest gap {Gap(resumed.Weights, straight.Weights):G3})");
                }
                else
                {
                    AssertClose(straight.Weights, resumed.Weights, ResumeTolerance, $"{name}: resumed, the adapters against the uninterrupted run's");
                }

                Console.WriteLine($"    {name} in one process: paused before 0 and at 2, stopped after 4; resumed, the largest gap to the uninterrupted run {Gap(resumed.Weights, straight.Weights):E2}");
            }

            // dqn on cartpole: a round's boundary is its step (the steps done); a report every 500 steps.
            var dqn = SmallDqn(11);
            float[] whole;
            using (var trainer = SmallTrainer(device, dqn with { CheckpointFolder = Path.Combine(root, "dqn-whole") }))
            {
                trainer.Train();
                whole = ModuleWeights(trainer.Network);
            }

            ClusterControlFile.Write(file, ClusterControlCommand.Pause);
            File.Delete(ClusterControlFile.StatePath(file));
            long reports = 0;
            var held = ControlPauses(file, () => Interlocked.Read(ref reports) * 500, 1_000);
            OffPolicyResult cut;
            using (var trainer = new OffPolicyTrainer(ClassicControlPlugin.CartPoleKind, ModulePolicy.Perceptron(4, 2, [16, 16], device, new Random(5)),
                       ModulePolicy.Perceptron(4, 2, [16, 16], device, new Random(105)), dqn with { CheckpointFolder = Path.Combine(root, "dqn-stopped") })
                   {
                       Steering = RunSteerings.For(null, steering),
                       OnReport = s =>
                       {
                           Interlocked.Increment(ref reports);
                           if (s.Step is 1_000 or 2_000)
                           {
                               ClusterControlFile.Write(file, s.Step == 1_000 ? ClusterControlCommand.Pause : ClusterControlCommand.Stop);
                           }
                       },
                   })
            {
                cut = trainer.Train();
            }

            Check(held.Wait(TimeSpan.FromSeconds(150)) && held.Result is null, $"dqn: {held.Result ?? "both pauses held"}");
            Check(cut.Cancelled && cut.Steps == 2_000 && ControlState(file, "stopped", 2_000) && ControlSaved(Path.Combine(root, "dqn-stopped")).SequenceEqual(["step-000000002000"]),
                $"dqn: stopped at step 2,000 as the file asked, one checkpoint: {cut.Steps}, {string.Join(", ", ControlSaved(Path.Combine(root, "dqn-stopped")))}");
            using (var trainer = SmallTrainer(device, dqn with { Resume = Path.Combine(root, "dqn-stopped"), CheckpointFolder = Path.Combine(root, "dqn-resumed") }, weights: 99))
            {
                var resumed = trainer.Train();
                Check(!resumed.Cancelled && resumed.Steps == dqn.Steps && SameBits(ModuleWeights(trainer.Network), whole),
                    $"dqn: resumed to step {dqn.Steps}, the network is the uninterrupted run's bit for bit");
            }

            // The command: IDRAK_CLUSTER_CONTROL on the numeric path (ppo and dqn) and the language model's, in one process.
            var (chatModel, data) = ReinforceFixture(root);
            try
            {
                var variables = new Dictionary<string, string?>
                {
                    ["IDRAK_CLUSTER_CONTROL"] = file, ["IDRAK_CLUSTER_CONTROL_POLL"] = "0.01", ["IDRAK_RL_LOG"] = null, ["IDRAK_RL_ENVIRONMENTS"] = "1", ["IDRAK_RL_PARALLEL_STEPS"] = "off",
                };
                string[] Ppo(string output) => ["reinforce", "--env", "cartpole", "--algorithm", "ppo", "--iterations", "3", "--episodes", "4", "--hidden", "8", "--seed", "7",
                    "-d", device.ToString(), "-o", output];
                string[] Dqn(string output) => ["reinforce", "--env", "bandit", "--algorithm", "dqn", "--steps", "400", "--hidden", "none", "--report-every", "100", "--seed", "3",
                    "-d", device.ToString(), "-o", output];
                string[] Chat(string output) => ["reinforce", chatModel, "--env", "prompts", "--data", data, "--reward", "exact", "--algorithm", "grpo", "--group", "4",
                    "--episodes", "4", "--iterations", "2", "--seed", "3", "-o", output, .. TinySession(device, root)];
                (int Code, string Out, string Err) cli = (0, "", "");
                ClusterControlFile.Write(file, ClusterControlCommand.Stop);
                foreach (var (what, args, stopped) in new[] { ("ppo", Ppo(Path.Combine(root, "cli-ppo-stop")), "stopped after 0 iterations"),
                             ("dqn", Dqn(Path.Combine(root, "cli-dqn-stop")), "stopped after 0 steps"), ("the language model", Chat(Path.Combine(root, "cli-chat-stop")), "stopped after 0 iterations") })
                {
                    WithEnvironment(variables, () => cli = TrainCli(args));
                    Check(cli.Code == 0 && cli.Out.Contains(stopped, StringComparison.Ordinal), $"reinforce ({what}) with the control file saying stop: {stopped}\n{cli.Out}\n{cli.Err}");
                }

                // Paused before the first iteration, then let go: the run trains to its end, its log in DIR/log.jsonl.
                ClusterControlFile.Write(file, ClusterControlCommand.Pause);
                File.Delete(ClusterControlFile.StatePath(file));
                string output = Path.Combine(root, "cli-ppo");
                var let = Task.Run(() =>
                {
                    bool paused = UiWait(() => ControlState(file, "paused", 0), 120);
                    ClusterControlFile.Write(file, ClusterControlCommand.Run);
                    return paused;
                });
                WithEnvironment(variables, () => cli = TrainCli(Ppo(output)));
                string log = Path.Combine(output, "log.jsonl");
                var lines = File.Exists(log) ? File.ReadAllLines(log).Select(l => (JsonObject)JsonNode.Parse(l)!).ToList() : [];
                Check(let.Wait(TimeSpan.FromSeconds(150)) && let.Result && cli.Code == 0 && cli.Out.Contains("trained 3 iterations", StringComparison.Ordinal),
                    $"reinforce paused by the control file before iteration 0, then let go: {cli.Out}\n{cli.Err}");
                Check(lines.Count == 3 && lines.All(l => (string?)l["event"] == "iteration" && l["pass_rate"] is not null && (int?)l["iterations"] == 3)
                      && lines.Select(l => (int)l["iteration"]!).SequenceEqual([1, 2, 3]), $"DIR/log.jsonl: a line per iteration with its pass rate: {string.Join(" | ", lines)}");
            }
            finally
            {
                Directory.Delete(chatModel, recursive: true);
            }
        }
        finally
        {
            Directory.Delete(root, recursive: true);
            Directory.Delete(model, recursive: true);
        }
    }

    private static void ControlReinforceDashboard(Device device)
    {
        string folder = TempFolder("cluster-control-ui");
        var (model, data) = ReinforceFixture(folder);
        var variables = new Dictionary<string, string?>
        {
            ["IDRAK_RL_ENVIRONMENTS"] = "1", ["IDRAK_RL_PARALLEL_STEPS"] = "off", ["IDRAK_CLUSTER_TIMEOUT"] = "600", ["IDRAK_RL_LOG"] = null, ["IDRAK_CLUSTER_CONTROL"] = null,
        };
        try
        {
            WithEnvironment(variables, () => ControlDashboardRun(device, folder, model, data));
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
            Directory.Delete(model, recursive: true);
        }
    }

    private static void ControlDashboardRun(Device device, string folder, string model, string data)
    {
        {
            string topology = Path.Combine(folder, "two.json");
            File.WriteAllText(topology, DpTopology(device, 2).ToJson());
            string output = Path.Combine(folder, "steered"), plain = Path.Combine(folder, "plain"), log = Path.Combine(output, "log.jsonl");
            string[] Reinforce(string into, int iterations) => ["reinforce", model, "--env", "prompts", "--data", data, "--reward", "cer", "--algorithm", "grpo", "--group", "4",
                "--episodes", "8", "--iterations", iterations.ToString(CultureInfo.InvariantCulture), "--seed", "3", "-o", into, "--shards", "2", .. TinySession(device, folder)];
            var (server, http) = UiServer(topology);
            int stoppedAt;
            try
            {
                var started = UiOk(http, "POST", "/api/runs", new JsonObject { ["command"] = UiArgs(Reinforce(output, 1000)), ["mode"] = "in-process", ["paused"] = true });
                string id = (string)started["id"]!;
                var control = server.Runs.Find(id)!.Control!;
                JsonObject Listed() => (JsonObject)UiOk(http, "GET", "/api/runs")["runs"]!.AsArray().First(r => (string?)r!["id"] == id)!;
                int Done() => (int?)Listed()["figures"]?["iteration"] ?? -1;
                Check((string?)started["unit"] == "iteration" && (int?)started["figures"]?["iteration"] == 0 && (string?)started["figures"]?["log"] == log
                      && started["controls"]!["stop"] is null, $"a reinforce run is steered and counts iterations from its log: {started}");
                Check(UiWait(() => UiState(http, id) == "paused", 240) && control.LastStep == -1, $"every rank waits before iteration 0: {UiState(http, id)}");
                Thread.Sleep(300);
                Check(Done() == 0 && !File.Exists(log), "no iteration while paused before the first");
                UiOk(http, "POST", $"/api/runs/{id}/resume");
                Check(UiWait(() => Done() >= 2, 300), $"two iterations listed: {Listed()}");

                UiOk(http, "POST", $"/api/runs/{id}/pause");
                Check(UiWait(() => UiState(http, id) == "paused", 300), "the pause is reached at an iteration boundary");
                int held = Done();
                Thread.Sleep(500);
                Check(Done() == held && control.LastStep == control.PauseAt - 1 && held == control.PauseAt,
                    $"every rank waits before iteration {control.PauseAt} with {held} done (the highest begun {control.LastStep})");
                UiOk(http, "POST", $"/api/runs/{id}/resume");
                Check(UiWait(() => Done() > held, 300), "the run goes on after the resume");
                UiOk(http, "POST", $"/api/runs/{id}/stop");
                Check(UiWait(() => UiState(http, id) is "stopped" or "failed", 300), "the run ends after the stop");
                var listed = Listed();
                var figures = (JsonObject)listed["figures"]!;
                stoppedAt = (int)figures["iteration"]!;
                Check((string?)listed["state"] == "stopped" && ((string?)listed["detail"])!.Contains($"stopped after iteration {stoppedAt}", StringComparison.Ordinal),
                    $"the run stopped as asked: {listed}");
                Check(stoppedAt > held && stoppedAt < 1000 && (int?)figures["iterations"] == 1000 && figures["reward_mean"] is not null && figures["first_reward_mean"] is not null
                      && figures["pass_rate"] is not null && figures["reward_spread"] is not null, $"the listing's figures: {figures}");
                Check(ControlSaved(Path.Combine(output, "checkpoints")).SequenceEqual([$"iteration-{stoppedAt:D6}"]) && File.Exists(Path.Combine(output, "adapter_model.safetensors")),
                    $"one checkpoint, the stop's, and the adapter so far: {string.Join(", ", ControlSaved(Path.Combine(output, "checkpoints")))}");
                var lines = SharedLines(log).Select(l => (JsonObject)JsonNode.Parse(l)!).ToList();
                Check(lines.Count == stoppedAt && lines.Select(l => (int)l["iteration"]!).SequenceEqual(Enumerable.Range(1, stoppedAt))
                      && lines.All(l => (string?)l["event"] == "iteration" && l["pass_rate"] is not null), $"the log: a line per iteration done: {lines.Count}");
                using (var page = new UiListener(server.Url!, id))
                {
                    Check(page.WaitFor(r => Is(r, "iteration") && (int?)r["iteration"] == stoppedAt, 30), "the events carry the run's iterations from its log");
                }

                Console.WriteLine($"    the dashboard: paused before 0 and at {held}, stopped after {stoppedAt}; figures {figures.ToJsonString()}");
            }
            finally
            {
                http.Dispose();
                server.Dispose();
            }

            // Resumed through the command, and the run uninterrupted for as many iterations: the same adapter, byte for byte.
            var resumed = ClusterCli(["cluster", "sim", "--topology", topology, "--", .. Reinforce(output, stoppedAt + 1), "--resume"]);
            Check(resumed.Code == 0 && resumed.Output.Split('\n').Count(l => l.StartsWith($"iteration {stoppedAt + 1}/{stoppedAt + 1}: reward", StringComparison.Ordinal)) == 1
                  && resumed.Output.Contains($"trained {stoppedAt + 1} iterations", StringComparison.Ordinal), $"--resume goes on at iteration {stoppedAt + 1}: {resumed.Output}\n{resumed.Error}");
            var whole = ClusterCli(["cluster", "sim", "--topology", topology, "--", .. Reinforce(plain, stoppedAt + 1)]);
            Check(whole.Code == 0, $"the uninterrupted run: {whole.Output}\n{whole.Error}");
            static string Figures(string line) => string.Join(" ", new[] { "iteration", "episodes", "reward_mean", "reward_spread", "reward_min", "reward_max", "pass_rate", "loss" }
                .Select(k => JsonNode.Parse(line)![k]!.ToJsonString()));
            var steeredLines = File.ReadAllLines(Path.Combine(output, "log.jsonl")).Select(Figures).ToList();
            var wholeLines = File.ReadAllLines(Path.Combine(plain, "log.jsonl")).Select(Figures).ToList();
            Check(steeredLines.Count == stoppedAt + 1 && wholeLines.Count == stoppedAt + 1 && steeredLines.Take(stoppedAt).SequenceEqual(wholeLines.Take(stoppedAt)),
                $"the steered run's {stoppedAt} iterations are the uninterrupted run's line for line, the pauses changed nothing:\n{string.Join("\n", steeredLines)}\n--\n{string.Join("\n", wholeLines)}");
            AssertClose(ControlAdapter(plain), ControlAdapter(output), ResumeTolerance, "the stopped and resumed run's adapter against the uninterrupted run's");
            Console.WriteLine($"    the dashboard's run resumed: the adapter's largest gap to the uninterrupted run {Gap(ControlAdapter(plain), ControlAdapter(output)):E2}");
        }
    }
}
