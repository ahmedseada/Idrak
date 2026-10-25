// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Reinforcement;
using Idrak.Reinforcement;
using Idrak.Reinforcement.Abstractions;
using Idrak.Reinforcement.Plugins.Classic;

// Plan 27, phase R6: off-policy learning in Idrak.Reinforcement: the replay buffer, the Q-learning losses (QLosses, "dqn"
// with double Q-learning), and the off-policy trainer (DQN): reproducible on 1 and 4 threads, resumed bit for bit, memory
// that does not grow, its settings, progress, telemetry and log. Its runs are on the classic plug-in's environments.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] OffPolicyGroup =
    [
        ("reinforcement: off-policy: the replay buffer is a fixed ring: the memory allocated when it is made is the transitions' size exactly (no pool's rounding) and does not grow after it is full, the oldest transition is overwritten, sampling is seeded, Save and Load keep it bit for bit", d => { if (d == Device.Cpu) OffPolicyReplayBuffer(); }),
        ("reinforcement: off-policy: dqn is QLosses' library default; it matches a direct computation (Huber of δ 1 and 2, the squared error, double Q-learning, weights), its gradient reaches only the actions taken, and unknown names, options and inputs are refused; an app's loss falls back on it", OffPolicyLoss),
        ("reinforcement: off-policy: a seeded DQN run gives the same returns, losses and weights twice, on 1 and 4 threads, with its environments stepped in parallel or not; another seed differs; memory held does not grow after the buffer is full", OffPolicyReproducible),
        ("reinforcement: off-policy: a resumed DQN run matches an uninterrupted one bit for bit (returns, losses, weights, the target network, the replay buffer, the schedule's rate), episodes under way at the checkpoint replayed from their seeds and actions", OffPolicyResume),
        ("reinforcement: off-policy: ε falls linearly to its end; the target network is copied or Polyak-averaged; progress, telemetry and the JSON log report each window; options read their IDRAK_RL_DQN_* variables and refuse bad values naming them", OffPolicyReports),
    ];

    private static float[] ModuleWeights(Module module) => [.. module.Parameters().SelectMany(p => p.ToArray())];

    // A small DQN run on CartPole: two environments side by side, a 16 x 16 network, a buffer that fills.
    private static OffPolicyOptions SmallDqn(int seed) => new()
    {
        Steps = 4_000, Seed = seed, Environments = 2, ParallelSteps = false, ReplayCapacity = 1_000, BatchSize = 32, LearningStarts = 200, TrainEvery = 4, GradientSteps = 2,
        TargetEvery = 200, LearningRate = 1e-3f, ReportEvery = 500, EpsilonFraction = 0.3f,
    };

    private static OffPolicyTrainer SmallTrainer(Device device, OffPolicyOptions options, int weights = 5) =>
        new(ClassicControlPlugin.CartPoleKind, ModulePolicy.Perceptron(4, 2, [16, 16], device, new Random(weights)), ModulePolicy.Perceptron(4, 2, [16, 16], device, new Random(weights + 100)), options);

    private static string? SameDqnRun(OffPolicyResult expected, float[] expectedWeights, OffPolicyResult actual, float[] actualWeights, int skip = 0)
    {
        if (actual.History.Count + skip != expected.History.Count)
        {
            return $"{actual.History.Count} reports, {expected.History.Count - skip} expected";
        }

        for (int i = 0; i < actual.History.Count; i++)
        {
            var (a, e) = (actual.History[i], expected.History[i + skip]);
            if (SameBits(e.Returns, a.Returns, $"report {i + skip + 1}'s returns") is { } returns)
            {
                return returns;
            }

            if (BitConverter.DoubleToInt64Bits(a.Loss) != BitConverter.DoubleToInt64Bits(e.Loss) || a.Step != e.Step || a.Updates != e.Updates || a.Episodes != e.Episodes)
            {
                return FormattableString.Invariant($"report {i + skip + 1}: step {a.Step}, updates {a.Updates}, episodes {a.Episodes}, loss {a.Loss:R}; expected {e.Step}, {e.Updates}, {e.Episodes}, {e.Loss:R}");
            }
        }

        return SameBits(expectedWeights, actualWeights, "the weights");
    }

    // ---------------------------------------------------------------- the replay buffer

    private static void OffPolicyReplayBuffer()
    {
        using var buffer = new ReplayBuffer(5, 2);
        long bytes = buffer.Bytes;
        Check(buffer.Capacity == 5 && buffer.Count == 0 && bytes == 5 * (2 * 2 * 4 + 4 + 4 + 1), $"a buffer of 5 transitions of 2 floats holds 125 bytes, not {bytes}");
        ThrowsNaming<InvalidOperationException>(() => buffer.Sample(new Random(1), new int[3]), "sampling an empty buffer", "empty");
        using (var warm = new ReplayBuffer(5, 2))
        {
            warm.Add([1f, 2f], 0, 0f, [3f, 4f], false);                      // the first call compiled before allocations are counted
        }

        // A large buffer allocates its transitions' size and a constant (the five arrays' headers), not a pool's power of two (R8).
        long start = GC.GetAllocatedBytesForCurrentThread();
        using (var large = new ReplayBuffer(100_000, 4))
        {
            long made = GC.GetAllocatedBytesForCurrentThread() - start;
            Check(large.Bytes == 4_100_000 && made >= large.Bytes && made - large.Bytes < 1_024,
                $"a buffer of 100,000 CartPole transitions holds 4,100,000 bytes ({large.Bytes}) and allocates them and under 1 KB more when made ({made} bytes)");
        }

        long allocated = 0;
        for (int i = 0; i < 12; i++)
        {
            long before = GC.GetAllocatedBytesForCurrentThread();
            buffer.Add([i, -i], i % 3, i * 0.5f, [i + 1, -i - 1], i % 4 == 3);
            allocated += GC.GetAllocatedBytesForCurrentThread() - before;
            Check(buffer.Count == Math.Min(i + 1, 5) && buffer.Bytes == bytes, $"after {i + 1} adds: {buffer.Count} held, {buffer.Bytes} bytes");
        }

        Check(allocated == 0 && buffer.Position == 2 && buffer.Added == 12, $"adding allocates nothing ({allocated} bytes); the ring writes at {buffer.Position} after 12 adds");
        var observations = new float[10];
        var next = new float[10];
        var actions = new int[5];
        var rewards = new float[5];
        var terminated = new bool[5];
        buffer.Gather([0, 1, 2, 3, 4], observations, actions, rewards, next, terminated);
        Check(observations.SequenceEqual([10f, -10f, 11f, -11f, 7f, -7f, 8f, -8f, 9f, -9f]) && actions.SequenceEqual([1, 2, 1, 2, 0]) && rewards.SequenceEqual([5f, 5.5f, 3.5f, 4f, 4.5f])
              && next[0] == 11f && terminated.SequenceEqual([false, true, true, false, false]), $"the oldest overwritten, in ring order: {string.Join(", ", observations)}");
        int[] drawn = new int[64], again = new int[64];
        buffer.Sample(new Random(9), drawn);
        buffer.Sample(new Random(9), again);
        Check(drawn.SequenceEqual(again) && drawn.All(i => i is >= 0 and < 5) && drawn.Distinct().Count() == 5, "seeded draws: the same seed the same indices, every transition reachable");
        ThrowsNaming<ArgumentOutOfRangeException>(() => buffer.Gather([5], observations, actions, rewards, next, terminated), "an index past the transitions held", "holds 5");
        ThrowsNaming<ArgumentException>(() => buffer.Add([1f], 0, 0f, [1f, 2f], false), "an observation of another size", "2 floats");
        ThrowsNaming<ArgumentException>(() => buffer.Add([1f, 2f], 0, float.NaN, [1f, 2f], false), "a NaN reward", "finite");

        using var stream = new MemoryStream();
        buffer.Save(stream);
        stream.Position = 0;
        using var copy = new ReplayBuffer(5, 2);
        copy.Load(stream);
        var copied = (new float[10], new int[5], new float[5], new float[10], new bool[5]);
        copy.Gather([0, 1, 2, 3, 4], copied.Item1, copied.Item2, copied.Item3, copied.Item4, copied.Item5);
        Check(copy.Count == 5 && copy.Position == 2 && copy.Added == 12 && copied.Item1.SequenceEqual(observations) && copied.Item2.SequenceEqual(actions) && copied.Item3.SequenceEqual(rewards)
              && copied.Item4.SequenceEqual(next) && copied.Item5.SequenceEqual(terminated), "Load restores what Save wrote");
        stream.Position = 0;
        using var other = new ReplayBuffer(6, 2);
        ThrowsNaming<InvalidDataException>(() => other.Load(stream), "a buffer of another capacity", "5 transitions", "6");
        buffer.Dispose();
        buffer.Dispose();
    }

    // ---------------------------------------------------------------- the loss

    private static void OffPolicyLoss(Device device)
    {
        Check(QLosses.Names.Contains(QLosses.Dqn) && QLosses.Origin(QLosses.Dqn) == Overrides.Library && QLosses.Default(QLosses.Dqn) is not null, "dqn is the library's default");
        ThrowsNaming<NotSupportedException>(() => QLosses.Get("rainbow"), "an unknown loss", "'rainbow'", "not registered", "dqn");
        var loss = QLosses.Get(QLosses.Dqn);
        Check(loss.Keys.Contains("huber") && loss.Keys.Contains("double") && !loss.NeedsNextValues() && loss.NeedsNextValues(new Dictionary<string, string> { ["double"] = "true" }),
            "options huber and double; the online next values only for double Q-learning");

        // Three transitions over three actions; the second terminated.
        float[] q = [1f, 2f, 0.5f, -1f, 0f, 3f, 0.25f, 0.75f, 2f];
        int[] taken = [1, 2, 0];
        float[] rewards = [0.5f, -1f, 2f], discounts = [0.9f, 0f, 0.5f];
        float[] nextTarget = [1f, 4f, 2f, 9f, 9f, 9f, 3f, 1f, -2f], nextOnline = [5f, 0f, 1f, 0f, 0f, 0f, -1f, 2f, 0f];
        float[] weights = [1f, 2f, 0.5f];
        float Expected(float delta, bool doubleQ, bool weighted)
        {
            double sum = 0;
            for (int i = 0; i < 3; i++)
            {
                var row = nextTarget.AsSpan(i * 3, 3);
                int a = doubleQ ? DqnArgMax(nextOnline[(i * 3)..(i * 3 + 3)]) : DqnArgMax(row.ToArray());
                double target = rewards[i] + (double)discounts[i] * (discounts[i] == 0 ? 0 : row[a]);
                double x = q[i * 3 + taken[i]] - (float)target, size = Math.Abs(x);
                double l = delta == 0 ? 0.5 * x * x : size <= delta ? 0.5 * x * x : delta * (size - 0.5 * delta);
                sum += (weighted ? weights[i] : 1) * l;
            }

            return (float)(sum / 3);
        }

        foreach (var (delta, doubleQ, weighted) in new[] { (1f, false, false), (2f, false, false), (0f, false, true), (1f, true, false), (1f, true, true) })
        {
            using var scope = new TensorScope();
            var values = Tensor.From(q, [3, 3], device, requiresGrad: true);
            var input = new QLossInput(values, taken, rewards, discounts, nextTarget, doubleQ ? nextOnline : default, weighted ? weights : default);
            var options = new Dictionary<string, string> { ["huber"] = delta.ToString("R", CultureInfo.InvariantCulture), ["double"] = doubleQ ? "true" : "false" };
            var total = loss.Loss(input, options);
            float expected = Expected(delta, doubleQ, weighted);
            Check(total.Size == 1 && Math.Abs(total.Item() - expected) < 1e-5f * Math.Max(1f, Math.Abs(expected)), $"δ {delta}, double {doubleQ}, weighted {weighted}: {total.Item():R}, directly {expected:R}");
            total.Backward();
            var gradient = values.Grad!.ToArray();
            for (int i = 0; i < 9; i++)
            {
                Check((i % 3 == taken[i / 3]) || gradient[i] == 0f, $"δ {delta}: the gradient reaches only the actions taken (element {i}: {gradient[i]})");
            }
        }

        using (var scope = new TensorScope())
        {
            var values = Tensor.From(q, [3, 3], device);
            ThrowsNaming<ArgumentException>(() => loss.Loss(new QLossInput(values, taken, rewards, discounts, nextTarget), new Dictionary<string, string> { ["double"] = "true" }),
                "double Q-learning without the online next values", "NextValues");
            ThrowsNaming<ArgumentException>(() => loss.Loss(new QLossInput(values, taken, rewards, discounts, nextTarget), new Dictionary<string, string> { ["gamma"] = "1" }), "an unknown option", "'gamma'", "huber, double");
            ThrowsNaming<ArgumentException>(() => new QLossInput(values, new[] { 0, 3, 0 }, rewards, discounts, nextTarget), "an action outside the space", "action 3");
            ThrowsNaming<ArgumentException>(() => new QLossInput(values, taken, rewards, discounts, nextTarget[..6]), "next values of another size", "9 next values");
        }

        // An app's loss over the library's: with FallBack, its failures are answered by the library's.
        var app = new FailingQLoss();
        try
        {
            QLosses.Register(app);
            Check(QLosses.Origin(QLosses.Dqn) != Overrides.Library, "the app's loss takes the name");
            QLosses.SetPolicy(QLosses.Dqn, SlotPolicy.FallBack);
            using var scope = new TensorScope();
            var values = Tensor.From(q, [3, 3], device);
            float answered = QLosses.Get(QLosses.Dqn).Loss(new QLossInput(values, taken, rewards, discounts, nextTarget)).Item();
            Check(Math.Abs(answered - Expected(1f, false, false)) < 1e-5f * Math.Max(1f, Math.Abs(answered)), $"the library's loss answered for the failing app's: {answered}");
        }
        finally
        {
            QLosses.SetPolicy(QLosses.Dqn, SlotPolicy.Throw);
            QLosses.Unregister(QLosses.Dqn);
        }

        Check(QLosses.Origin(QLosses.Dqn) == Overrides.Library, "Unregister brings the library's back");
    }

    private sealed class FailingQLoss : IQLoss
    {
        public string Name => QLosses.Dqn;

        public string Summary => "a test's loss that fails";

        public IReadOnlySet<string> Keys { get; } = new OptionKeys("huber", "double");

        public Tensor Loss(QLossInput input, IReadOnlyDictionary<string, string>? options = null) => throw new InvalidOperationException("the test's loss fails");
    }

    // ---------------------------------------------------------------- reproducible, memory

    private static void OffPolicyReproducible(Device device)
    {
        int threads = ComputeResources.MaxCpuThreads;
        try
        {
            ComputeResources.MaxCpuThreads = 1;
            (OffPolicyResult Result, float[] Weights) Run(OffPolicyOptions options)
            {
                using var trainer = SmallTrainer(device, options);
                var result = trainer.Train();
                return (result, ModuleWeights(trainer.Network));
            }

            var first = Run(SmallDqn(7));
            var again = Run(SmallDqn(7));
            Check(SameDqnRun(first.Result, first.Weights, again.Result, again.Weights) is null, $"the same seed twice: {SameDqnRun(first.Result, first.Weights, again.Result, again.Weights)}");
            ComputeResources.MaxCpuThreads = 4;
            var four = Run(SmallDqn(7));
            Check(SameDqnRun(first.Result, first.Weights, four.Result, four.Weights) is null, $"on 4 threads: {SameDqnRun(first.Result, first.Weights, four.Result, four.Weights)}");
            var parallel = Run(SmallDqn(7) with { ParallelSteps = true });
            Check(SameDqnRun(first.Result, first.Weights, parallel.Result, parallel.Weights) is null, $"stepped in parallel: {SameDqnRun(first.Result, first.Weights, parallel.Result, parallel.Weights)}");
            var measured = Run(SmallDqn(7) with { ParallelSteps = null });
            Check(SameDqnRun(first.Result, first.Weights, measured.Result, measured.Weights) is null, $"stepped as measured: {SameDqnRun(first.Result, first.Weights, measured.Result, measured.Weights)}");
            var other = Run(SmallDqn(8));
            Check(SameDqnRun(first.Result, first.Weights, other.Result, other.Weights) is not null, "another seed plays other episodes");
            Check(first.Result.Steps == 4_000 && first.Result.Updates == 2 * 475 && first.Result.History.Count == 8 && first.Result.History.Sum(h => h.Returns.Length) > 20,
                $"4,000 steps in rounds of 2, updates every 4 rounds once 200 steps are in: {first.Result.Steps} steps, {first.Result.Updates} updates, {first.Result.History.Count} reports");
        }
        finally
        {
            ComputeResources.MaxCpuThreads = threads;
        }

        // Memory held after the buffer is full does not grow.
        long heapFull = 0, heapEnd = 0, deviceFull = 0, deviceEnd = 0, bytesFull = 0;
        OffPolicyTrainer? running = null;
        using var trainer = new OffPolicyTrainer(ClassicControlPlugin.CartPoleKind, ModulePolicy.Perceptron(4, 2, [16, 16], device, new Random(5)),
            ModulePolicy.Perceptron(4, 2, [16, 16], device, new Random(6)), SmallDqn(9) with { Steps = 12_000 })
        {
            OnReport = s =>
            {
                if (s.Step is 2_000 or 12_000)
                {
                    long heap = GC.GetTotalMemory(forceFullCollection: true), held = ComputeResources.GetMemoryUsage(device).InUse;
                    if (s.Step == 2_000)
                    {
                        (heapFull, deviceFull, bytesFull) = (heap, held, running!.Replay.Bytes);
                    }
                    else
                    {
                        (heapEnd, deviceEnd) = (heap, held);
                    }
                }
            },
        };
        running = trainer;
        trainer.Train();
        Check(trainer.Replay.Count == 1_000 && trainer.Replay.Added == 12_000 && trainer.Replay.Bytes == bytesFull && bytesFull == 1_000 * (2 * 4 * 4 + 9),
            $"the buffer full at 1,000 of 12,000 transitions, {trainer.Replay.Bytes} bytes (41,000 expected)");
        Check(deviceEnd == deviceFull, $"device memory in use when the buffer is full (step 2,000): {deviceFull} bytes; at step 12,000: {deviceEnd}");
        Check(heapEnd - heapFull < 512 * 1024, $"the managed heap at step 2,000: {heapFull / 1024} KB; at 12,000: {heapEnd / 1024} KB (grew {(heapEnd - heapFull) / 1024} KB)");
        Console.WriteLine($"    off-policy memory: replay {bytesFull} bytes, device {deviceFull} → {deviceEnd} bytes, heap {heapFull / 1024} → {heapEnd / 1024} KB over 10,000 steps");
    }

    // ---------------------------------------------------------------- resume

    private static void OffPolicyResume(Device device)
    {
        string root = Path.Combine(Path.GetTempPath(), $"idrak-dqn-resume-{Environment.ProcessId}");
        try
        {
            var options = SmallDqn(11) with { Schedule = LearningRateSchedules.Linear, SaveEvery = 2_000, MaxEpisodeSteps = 300 };
            OffPolicyResult whole;
            float[] wholeWeights, wholeTarget;
            using (var trainer = SmallTrainer(device, options with { CheckpointFolder = Path.Combine(root, "whole") }))
            {
                whole = trainer.Train();
                (wholeWeights, wholeTarget) = (ModuleWeights(trainer.Network), ModuleWeights(trainer.Target));
            }

            OffPolicyResult stopped;
            using (var stop = new CancellationTokenSource())
            using (var trainer = new OffPolicyTrainer(ClassicControlPlugin.CartPoleKind, ModulePolicy.Perceptron(4, 2, [16, 16], device, new Random(5)),
                       ModulePolicy.Perceptron(4, 2, [16, 16], device, new Random(105)), options with { CheckpointFolder = Path.Combine(root, "stopped") })
                   {
                       OnReport = s =>
                       {
                           if (s.Step == 2_000)
                           {
                               stop.Cancel();
                           }
                       },
                   })
            {
                stopped = trainer.Train(stop.Token);
            }

            string checkpoint = Path.Combine(root, "stopped", "step-000000002000");
            Check(stopped.Cancelled && stopped.Steps == 2_000 && stopped.Checkpoint == checkpoint, $"the run stopped after 2,000 of 4,000 steps: {stopped.Steps}, {stopped.Checkpoint}");
            foreach (string file in new[] { "network.ikw", "target.ikw", "optimizer.safetensors", "replay.bin", "state.json" })
            {
                Check(File.Exists(Path.Combine(checkpoint, file)), $"a checkpoint holds {file}");
            }

            var state = JsonNode.Parse(File.ReadAllText(Path.Combine(checkpoint, "state.json")))!;
            var lanes = state["environments"]!.AsArray();
            Check(lanes.Count == 2 && lanes.Any(l => l!["actions"]!.AsArray().Count > 0) && (long)state["rounds"]! == 1_000,
                $"each environment's episode under way kept as its seed and actions: {string.Join(", ", lanes.Select(l => l!["actions"]!.AsArray().Count))} actions");
            Check(Directory.Exists(Path.Combine(root, "whole", "step-000000002000")) && Directory.Exists(Path.Combine(root, "whole", "step-000000004000")), "checkpoints after 2,000 and 4,000 steps");

            OffPolicyResult resumed;
            float[] resumedWeights, resumedTarget;
            using (var trainer = SmallTrainer(device, options with { Resume = Path.Combine(root, "stopped"), CheckpointFolder = Path.Combine(root, "resumed") }, weights: 99))
            {
                resumed = trainer.Train();
                (resumedWeights, resumedTarget) = (ModuleWeights(trainer.Network), ModuleWeights(trainer.Target));
                Check(BitConverter.SingleToInt32Bits(trainer.Optimizer.LearningRate) == BitConverter.SingleToInt32Bits(whole.History[^1].LearningRate),
                    $"the schedule's rate resumes where it was: {trainer.Optimizer.LearningRate:R}, {whole.History[^1].LearningRate:R}");
                using var resumedReplay = File.OpenRead(Path.Combine(root, "resumed", "step-000000004000", "replay.bin"));
                using var wholeReplay = File.OpenRead(Path.Combine(root, "whole", "step-000000004000", "replay.bin"));
                Check(new BinaryReader(resumedReplay).ReadBytes((int)resumedReplay.Length).SequenceEqual(new BinaryReader(wholeReplay).ReadBytes((int)wholeReplay.Length)),
                    "the replay buffers at the end are the same, byte for byte");
            }

            Check(resumed.Steps == 4_000 && resumed.History.Count == 4 && resumed.History[0].Step == 2_500, $"the resumed run continues at step 2,000: {resumed.History.Count} reports from {resumed.History[0].Step}");
            Check(SameDqnRun(whole, wholeWeights, resumed, resumedWeights, skip: 4) is null, $"resumed after 2,000 of 4,000 steps, as the run that did not stop: {SameDqnRun(whole, wholeWeights, resumed, resumedWeights, skip: 4)}");
            Check(SameBits(wholeTarget, resumedTarget, "the target network") is null, $"{SameBits(wholeTarget, resumedTarget, "the target network")}");
            void Resume(OffPolicyOptions o)
            {
                using var trainer = SmallTrainer(device, o);
                trainer.Train();
            }

            ThrowsNaming<FileNotFoundException>(() => Resume(options with { Resume = Path.Combine(root, "nothing") }), "a folder with no checkpoint", "no off-policy checkpoint");
            ThrowsNaming<InvalidDataException>(() => Resume(options with { Resume = Path.Combine(root, "stopped"), Seed = 12 }), "another run's checkpoint", "seed 11", "seed 12");
        }
        finally
        {
            if (Directory.Exists(root))
            {
                Directory.Delete(root, recursive: true);
            }
        }
    }

    // ---------------------------------------------------------------- reports, the target, ε, settings

    private static void OffPolicyReports(Device device)
    {
        // ε: linear from its start to its end over the fraction, then the end; reported at each window.
        using (var trainer = SmallTrainer(device, SmallDqn(3) with { Steps = 2_000, ReportEvery = 200, EpsilonStart = 1f, EpsilonEnd = 0.1f, EpsilonFraction = 0.5f }))
        {
            var history = trainer.Train().History;
            foreach (var report in history)
            {
                double progress = (report.Step - 2) / 2_000.0;
                double expected = progress >= 0.5 ? 0.1 : 1 - progress * 0.9 / 0.5;
                Check(Math.Abs(report.Epsilon - expected) < 1e-5, string.Create(CultureInfo.InvariantCulture, $"ε at step {report.Step}: {report.Epsilon}, expected {expected:G6}"));
            }
        }

        // The target network: copied when τ is 1 (at the start too), moved by τ otherwise.
        using (var trainer = SmallTrainer(device, SmallDqn(3) with { Steps = 400, TargetEvery = 1_000_000, LearningStarts = 0 }))
        {
            Check(SameBits(ModuleWeights(trainer.Network), ModuleWeights(trainer.Target), "the target at the start") is null, "the target network starts as the online one");
            trainer.Train();
            Check(SameBits(ModuleWeights(trainer.Network), ModuleWeights(trainer.Target), "after updates") is not null, "updates move the online network, not the target");
        }

        float[] start, online, target;
        using (var trainer = SmallTrainer(device, SmallDqn(3) with { Steps = 8, TargetEvery = 8, LearningStarts = 0, TrainEvery = 4, GradientSteps = 1, Tau = 0.25f }))
        {
            start = ModuleWeights(trainer.Target);
            trainer.Train();
            (online, target) = (ModuleWeights(trainer.Network), ModuleWeights(trainer.Target));
        }

        float worst = 0;
        for (int i = 0; i < start.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(target[i] - (0.75f * start[i] + 0.25f * online[i])));
        }

        Check(worst < 1e-6f, $"Polyak averaging: θ' ← 0.25 θ + 0.75 θ' (largest difference {worst})");

        // Progress, telemetry and the JSON log.
        string root = Path.Combine(Path.GetTempPath(), $"idrak-dqn-report-{Environment.ProcessId}");
        Directory.CreateDirectory(root);
        try
        {
            var hook = new RlHook();
            var writer = new StringWriter();
            string log = Path.Combine(root, "run.jsonl");
            var progress = ProgressReporters.Start("dqn", options: new ProgressOptions { Reporter = ProgressReporters.JsonReporter, Output = writer, Interval = TimeSpan.Zero });
            OffPolicyResult reported;
            using (Telemetry.Subscribe(hook))
            using (var trainer = new OffPolicyTrainer(ClassicControlPlugin.CartPoleKind, ModulePolicy.Perceptron(4, 2, [16], device, new Random(1)),
                       ModulePolicy.Perceptron(4, 2, [16], device, new Random(2)), SmallDqn(4) with { Steps = 1_000, Log = log, EvaluateEvery = 500, EvaluationEpisodes = 2 }) { Progress = progress })
            {
                reported = trainer.Train();
            }

            Check(progress.Done == 1_000 && progress.Total == 1_000 && writer.ToString().Contains("\"detail\"", StringComparison.Ordinal) && writer.ToString().Contains("return", StringComparison.Ordinal),
                $"progress in steps: {progress.Done} of {progress.Total}, {writer}");
            Check(hook.Started == 1 && hook.Completed == 1 && hook.Batches == reported.Updates && hook.Epochs.Count == 2, $"telemetry: {hook.Started} start, {hook.Batches} updates, {hook.Epochs.Count} reports, {hook.Completed} end");
            foreach (string name in new[] { "steps", "episodes", "reward-mean", "epsilon", "updates", "q", "gradient-norm" })
            {
                Check(hook.Epochs[^1].Metrics.ContainsKey(name), $"the report's telemetry has {name}: {string.Join(", ", hook.Epochs[^1].Metrics.Keys)}");
            }

            Check(hook.Epochs[^1].ValidationMetrics?["reward-mean"] == reported.History[^1].EvaluationReward && reported.History.All(h => h.EvaluationReward is not null),
                "the evaluation reward is the report's validation metric");
            var lines = File.ReadAllLines(log);
            Check(lines.Length == 2 && (long)JsonNode.Parse(lines[1])!["step"]! == 1_000 && JsonNode.Parse(lines[1])!["evaluation_reward"] is not null && JsonNode.Parse(lines[1])!["q"] is not null,
                $"a JSON line per report: {string.Join(" | ", lines)}");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }

        // Refusals: kinds and networks that do not fit, settings that cannot work.
        ThrowsNaming<ArgumentException>(() => new OffPolicyTrainer(ClassicControlPlugin.PendulumKind, ModulePolicy.Perceptron(3, 1, [], device), ModulePolicy.Perceptron(3, 1, [], device), SmallDqn(1)),
            "a box action", "discrete");
        ThrowsNaming<ArgumentException>(() => new OffPolicyTrainer(ClassicControlPlugin.CartPoleKind, ModulePolicy.Perceptron(4, 3, [], device), ModulePolicy.Perceptron(4, 3, [], device), SmallDqn(1)),
            "three values for two actions", "3 values", "2 actions");
        ThrowsNaming<ArgumentException>(() => new OffPolicyTrainer(ClassicControlPlugin.CartPoleKind, ModulePolicy.Perceptron(4, 2, [8], device), ModulePolicy.Perceptron(4, 2, [], device), SmallDqn(1)),
            "a target of another architecture", "same architecture");
        ThrowsNaming<ArgumentException>(() => SmallTrainer(device, SmallDqn(1) with { Tau = 0f }), "τ of 0", "Tau", "(0, 1]");
        ThrowsNaming<NotSupportedException>(() =>
        {
            using var trainer = SmallTrainer(device, SmallDqn(1) with { Algorithm = "rainbow" });
            trainer.Train();
        }, "an unknown algorithm", "'rainbow'", "not registered");

        // The variables.
        string[] names = ["IDRAK_RL_DQN_LOSS", "IDRAK_RL_DQN_DOUBLE", "IDRAK_RL_DQN_STEPS", "IDRAK_RL_DQN_REPLAY", "IDRAK_RL_DQN_TAU", "IDRAK_RL_DQN_EPSILON_END", "IDRAK_RL_GAMMA", "IDRAK_RL_SEED",
            "IDRAK_RL_DQN_LEARNING_STARTS", "IDRAK_RL_DQN_BATCH", "IDRAK_RL_DQN_HUBER"];
        try
        {
            Check(OffPolicyOptions.FromEnvironment() == new OffPolicyOptions(), "with no variable set, the defaults");
            Environment.SetEnvironmentVariable("IDRAK_RL_DQN_LOSS", "dqn");
            Environment.SetEnvironmentVariable("IDRAK_RL_DQN_DOUBLE", "on");
            Environment.SetEnvironmentVariable("IDRAK_RL_DQN_STEPS", "50000");
            Environment.SetEnvironmentVariable("IDRAK_RL_DQN_REPLAY", "2000");
            Environment.SetEnvironmentVariable("IDRAK_RL_DQN_TAU", "0.005");
            Environment.SetEnvironmentVariable("IDRAK_RL_DQN_EPSILON_END", "0.02");
            Environment.SetEnvironmentVariable("IDRAK_RL_GAMMA", "0.98");
            Environment.SetEnvironmentVariable("IDRAK_RL_SEED", "-4");
            Environment.SetEnvironmentVariable("IDRAK_RL_DQN_HUBER", "0");
            var read = OffPolicyOptions.FromEnvironment(new OffPolicyOptions { BatchSize = 7 });
            Check(read.DoubleQ == true && read.Steps == 50_000 && read.ReplayCapacity == 2_000 && read.Tau == 0.005f && read.EpsilonEnd == 0.02f && read.Gamma == 0.98f && read.Seed == -4
                  && read.Huber == 0f && read.BatchSize == 7, $"the variables set their options, the others keep the start's: {read}");
            Environment.SetEnvironmentVariable("IDRAK_RL_DQN_LEARNING_STARTS", "-1");
            ThrowsNaming<ArgumentException>(() => OffPolicyOptions.FromEnvironment(), "a negative count", "IDRAK_RL_DQN_LEARNING_STARTS=-1");
            Environment.SetEnvironmentVariable("IDRAK_RL_DQN_LEARNING_STARTS", null);
            Environment.SetEnvironmentVariable("IDRAK_RL_DQN_BATCH", "many");
            ThrowsNaming<ArgumentException>(() => OffPolicyOptions.FromEnvironment(), "a bad batch", "IDRAK_RL_DQN_BATCH=many");
        }
        finally
        {
            foreach (string name in names)
            {
                Environment.SetEnvironmentVariable(name, null);
            }
        }

        var table = Idrak.Cli.Shared.EnvironmentVariables.All.Where(v => v.Name.StartsWith("IDRAK_RL_DQN_", StringComparison.Ordinal)).Select(v => v.Name).ToHashSet(StringComparer.Ordinal);
        Check(names.Where(n => n.StartsWith("IDRAK_RL_DQN_", StringComparison.Ordinal)).All(table.Contains) && table.Count == 20, $"the IDRAK_RL_DQN_* variables are in the tool's table: {table.Count}");
    }
}
