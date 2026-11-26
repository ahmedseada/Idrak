// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Reinforcement;
using Idrak.Reinforcement;
using Idrak.Reinforcement.Abstractions;
using Idrak.Reinforcement.Plugins.Classic;

// Plan 27, phase R6: off-policy learning in Idrak.Reinforcement: the replay buffer, and the off-policy trainer's helpers
// the cluster's control tests share.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] OffPolicyGroup =
    [
        ("reinforcement: off-policy: the replay buffer is a fixed ring: the memory allocated when it is made is the transitions' size exactly (no pool's rounding) and does not grow after it is full, the oldest transition is overwritten, sampling is seeded, Save and Load keep it bit for bit", d => { CpuOnly(d); OffPolicyReplayBuffer(); }),
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

    // ---------------------------------------------------------------- reproducible, memory

    // ---------------------------------------------------------------- resume

    // ---------------------------------------------------------------- reports, the target, ε, settings
}
