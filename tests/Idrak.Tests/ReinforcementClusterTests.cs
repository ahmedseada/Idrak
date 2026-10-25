// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Abstraction.Cluster;
using Idrak.Abstraction.Reinforcement;
using Idrak.Models;
using Idrak.Reinforcement;
using Idrak.Text;
using Idrak.Text.Reinforcement;

// Plan 27, R3b: rollouts on a cluster. The language-model policy on a fine-tuning session opened with a cluster spreads a
// rollout's episodes over the ranks (R2's collector cuts it into shards, each rank plays its shards' episodes and the policy
// gathers every rank's in rank order), so a seeded run replays on 1, 2 and 4 ranks. Every cluster here is an in-process one
// (Idrak.Cluster, a thread per rank on devices carved from the device under test, the "sim" transport): no child process.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ReinforcementClusterGroup =
    [
        ("reinforcement cluster: the split: shards from the episodes' numbers (blocks and interleaved, sizes within one, duplicates refused), shards to ranks in runs; the settings and their variables; the exchange gives every rank each rank's bytes and values as sent, and text episodes round trip (messages and tool lists shared as they were, images refused)", ClusterSpreadRules),
        ("reinforcement cluster: a seeded grpo run under R2's trainer on the prompts environment gives the same episodes, rewards and advantages on 1, 2 and 4 ranks (4 shards, blocks and interleaved) and adapters within the documented tolerance; every rank plays its share; the first ratios are exactly 1 on every rank; spreading the passes changes no bit; rank 0 alone writes the checkpoint and the log; held device memory does not grow", ClusterRolloutsReplay),
    ];

    private static void ClusterSpreadRules(Device device)
    {
        // Shards from the numbers, whatever order the episodes come in.
        int[] numbers = [.. Enumerable.Range(10, 10).Reverse()];
        var blocks = RolloutSpread.Shards(numbers, 4);
        var interleaved = RolloutSpread.Shards(numbers, 4, "Interleaved");
        Check(numbers.Select((n, i) => (n, s: blocks[i])).OrderBy(p => p.n).Select(p => p.s).SequenceEqual([0, 0, 0, 1, 1, 2, 2, 2, 3, 3]),
            $"blocks: runs in number order, sizes within one: {string.Join(",", blocks)}");
        Check(numbers.Select((n, i) => interleaved[i] == (n - 10) % 4).All(x => x), $"interleaved: episode k in shard k mod 4: {string.Join(",", interleaved)}");
        Check(RolloutSpread.Shards(numbers, 1).All(s => s == 0) && RolloutSpread.Shards(numbers, 12).Distinct().Count() == 10, "one shard holds all; more shards than episodes leave some empty");
        Throws<ArgumentException>(() => RolloutSpread.Shards([1, 2, 1], 2), "two episodes of one number");
        Throws<ArgumentException>(() => RolloutSpread.Parse("round-robin"), "an unknown split");
        Check(Enumerable.Range(0, 4).Select(s => RolloutSpread.Rank(s, 4, 2)).SequenceEqual([0, 0, 1, 1])
              && Enumerable.Range(0, 2).Select(s => RolloutSpread.Rank(s, 2, 4)).SequenceEqual([0, 2])
              && Enumerable.Range(0, 4).Select(s => RolloutSpread.Rank(s, 4, 4)).SequenceEqual([0, 1, 2, 3]), "shards to ranks: runs of shards, shard s on rank ⌊s·W/S⌋");

        // The settings and their variables.
        string?[] saved = [Environment.GetEnvironmentVariable(RolloutSpread.ShardsVariable), Environment.GetEnvironmentVariable(RolloutSpread.SpreadVariable),
            Environment.GetEnvironmentVariable(LanguageModelPolicyOptions.SpreadPassesVariable)];
        try
        {
            Check(new ReinforcementOptions() is { Shards: 0, Spread: RolloutSpread.Blocks } && new LanguageModelPolicyOptions().SpreadPasses, "defaults: a shard per rank, blocks, the passes spread");
            Environment.SetEnvironmentVariable(RolloutSpread.ShardsVariable, "4");
            Environment.SetEnvironmentVariable(RolloutSpread.SpreadVariable, "interleaved");
            Environment.SetEnvironmentVariable(LanguageModelPolicyOptions.SpreadPassesVariable, "off");
            Check(ReinforcementOptions.FromEnvironment() is { Shards: 4, Spread: RolloutSpread.Interleaved } && !new LanguageModelPolicyOptions().SpreadPasses, "the variables set them");
            Environment.SetEnvironmentVariable(RolloutSpread.SpreadVariable, "diagonal");
            try
            {
                _ = ReinforcementOptions.FromEnvironment();
                Check(false, "a bad split is refused");
            }
            catch (ArgumentException e)
            {
                Check(e.Message.Contains(RolloutSpread.SpreadVariable, StringComparison.Ordinal), $"a bad split is refused naming its variable: {e.Message}");
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable(RolloutSpread.ShardsVariable, saved[0]);
            Environment.SetEnvironmentVariable(RolloutSpread.SpreadVariable, saved[1]);
            Environment.SetEnvironmentVariable(LanguageModelPolicyOptions.SpreadPassesVariable, saved[2]);
        }

        // The exchange on three ranks: bytes of every value and of uneven lengths (one rank sends none), floats as their bits.
        static byte[] Bytes(int rank) => rank == 1 ? [] : [.. Enumerable.Range(0, rank == 0 ? 7 : 1000).Select(i => (byte)((i * 37) + rank))];
        static float[] Floats(int rank) => rank == 2 ? [] : [-0f, float.Epsilon, float.MaxValue, -1e-30f, rank + 0.1f, 1f / 3f];
        var gathered = DpRun(DpTopology(device, 3), DpOptions(), rank =>
            (Bytes: RolloutExchange.Gather(rank.Collectives, Bytes(rank.Rank)), Floats: RolloutExchange.Gather(rank.Collectives, (ReadOnlySpan<float>)Floats(rank.Rank))));
        for (int at = 0; at < 3; at++)
        {
            for (int r = 0; r < 3; r++)
            {
                Check(gathered[at].Bytes[r].SequenceEqual(Bytes(r)), $"rank {at} holds rank {r}'s {Bytes(r).Length} bytes as sent");
                Check(SameBits(gathered[at].Floats[r], Floats(r)), $"rank {at} holds rank {r}'s values bit for bit");
            }
        }

        // Text episodes round trip: two steps with tools, a tool call, reasoning, info, a context and none.
        IReadOnlyList<ToolDefinition> tools = [new ToolDefinition("grep", "searches", JsonNode.Parse("""{"type":"object","properties":{"q":{"type":"string"}}}"""))];
        var system = new ChatMessage("system", "be brief");
        var user = new ChatMessage("user", "find x");
        var call = new ChatMessage("assistant", "", "look first", [new ToolCall("grep", new JsonObject { ["q"] = "x" })]);
        var result = new ChatMessage("tool", "x at 3", ToolName: "grep");
        var answer = new ChatMessage("assistant", "line 3");
        ChatObservation first = new([system, user], tools, true), second = new([system, user, call, result], tools), last = new([system, user, call, result, answer], tools);
        var trajectory = new Trajectory<ChatObservation, ChatMessage>(41, 7, [first, second, last], [call, answer],
            [new ActionRecord([5, 6, 7], [-0.5f, -1.25f, -0f], float.NaN, [1, 2, 3, 4]), new ActionRecord([9], [-2f])], [0f, 1.5f], terminated: true, truncated: false, group: 40,
            info: new EpisodeInfo().Set(new EpisodeInfoKey<string>("task"), "find").Set(EpisodeInfoKeys.Rounds, 2));
        var back = RolloutExchange.Read(RolloutExchange.Write([trajectory])).Single();
        static string Text(ChatMessage m) => $"{m.Role}|{m.Content}|{m.Thinking}|{m.ToolName}|{string.Join(";", (m.ToolCalls ?? []).Select(c => c.Name + c.Arguments.ToJsonString()))}";
        Check(back is { Index: 41, Seed: 7, Group: 40, Terminated: true, Truncated: false, Steps: 2 } && float.IsNaN(back.FinalValue) && back.Rewards.ToArray().SequenceEqual([0f, 1.5f])
              && back.Info!.ToString() == trajectory.Info!.ToString() && back.Info.Get(EpisodeInfoKeys.Rounds) == 2, "the episode's number, seed, group, ends, rewards and info (each value in its type)");
        Check(back.Observations.Zip(trajectory.Observations).All(p => p.First.Think == p.Second.Think && p.First.Messages.Select(Text).SequenceEqual(p.Second.Messages.Select(Text))
              && p.First.Tools!.Single() is { Name: "grep", Description: "searches" } tool && tool.Parameters!.ToJsonString() == tools[0].Parameters!.ToJsonString())
              && back.Actions.Select(Text).SequenceEqual(trajectory.Actions.Select(Text)), "the observations (messages, tools, reasoning mode) and the actions");
        Check(ReferenceEquals(back.Observations[0].Tools, back.Observations[2].Tools) && ReferenceEquals(back.Observations[0].Messages[1], back.Observations[2].Messages[1])
              && ReferenceEquals(back.Actions[0], back.Observations[1].Messages[2]), "a tool list and a message are sent once and shared as they were");
        Check(back.Records.Zip(trajectory.Records).All(p => p.First.Tokens.SequenceEqual(p.Second.Tokens) && SameBits(p.First.LogProbabilities, p.Second.LogProbabilities)
              && BitConverter.SingleToInt32Bits(p.First.Value) == BitConverter.SingleToInt32Bits(p.Second.Value)
              && (p.First.Context ?? [-1]).SequenceEqual(p.Second.Context ?? [-1])), "the records: ids, log-probabilities and values bit for bit, the context or none");
        var image = new ChatMessage("user", [ChatImage.FromBytes([0x89, 0x50, 0x4E, 0x47])]);
        Throws<NotSupportedException>(() => RolloutExchange.Write([new Trajectory<ChatObservation, ChatMessage>(0, 0, [new([image]), new([image, answer])], [answer], [new ActionRecord([1], [-1f])], [0f], true, false)]),
            "an image is not sent between ranks");
    }

    // One rank's record of a run (or the one process's).
    private sealed class ClusterRunRecord
    {
        public List<string[]> Episodes { get; } = [];             // per iteration, each episode as text (number, seed, group, ids, reward)

        public List<float[][]> Sampled { get; } = [];             // per iteration, each episode's sampler log-probabilities

        public List<float[]> Advantages { get; } = [];

        public List<int> Played { get; } = [];

        public float[] Adapters { get; set; } = [];

        public long HeldAfterSecond { get; set; }

        public long HeldAtEnd { get; set; }

        public TimeSpan Rollout { get; set; }                     // warm iterations (the first aside, rule 76)

        public TimeSpan Update { get; set; }
    }

    // A policy that counts the observations it acted on (a rank's episodes, for one-turn episodes), handing everything else on.
    private sealed class CountingPolicy(LanguageModelPolicy inner) : IPolicy<ChatObservation, ChatMessage>, IRolloutRanks<ChatObservation, ChatMessage>
    {
        public int Acted { get; set; }

        public int Rank => inner.Rank;

        public int WorldSize => inner.WorldSize;

        public ValueTask<PolicyActions<ChatMessage>> ActAsync(IReadOnlyList<ChatObservation> observations, ActionSampling sampling, CancellationToken cancellationToken = default)
        {
            Acted += observations.Count;
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

    // A library loss under a test name recording the largest |log p − behaviour| of every call (from every rank's thread).
    private sealed class GapLoss(string name, IPolicyLoss inner) : IPolicyLoss
    {
        private readonly Lock _gate = new();

        public float Gap { get; private set; }

        public int Calls { get; private set; }

        public bool SawReference { get; private set; }

        public string Name => name;

        public string Summary => $"{inner.Name}, its ratios recorded (the reinforcement cluster tests)";

        public IReadOnlySet<string> Keys => inner.Keys;

        public bool NeedsReference(IReadOnlyDictionary<string, string>? options = null) => inner.NeedsReference(options);

        public Tensor Loss(PolicyLossInput input, IReadOnlyDictionary<string, string>? options = null)
        {
            float gap = input.LogProbabilities.ToArray().Zip(input.BehaviorLogProbabilities.ToArray()).Max(p => Math.Abs(p.First - p.Second));
            lock (_gate)
            {
                Gap = Math.Max(Gap, gap);
                Calls++;
                SawReference |= input.ReferenceLogProbabilities is not null;
            }

            return inner.Loss(input, options);
        }
    }

    private static void ClusterRolloutsReplay(Device device)
    {
        string folder = WriteChatModelWith(TinyChatSpec(), ChatMlTemplate);
        string warm = Path.Combine(folder, "warm");
        var loss = new GapLoss("reinforcement-cluster-test-grpo", PolicyLosses.Get("grpo"));
        PolicyLosses.Register(loss);
        // Small batches (one row of at most 64 positions each), so a step has several and the ranks share them.
        var tuning = SessionOptions(1e-2f) with { BatchTokens = 64 };
        try
        {
            IReadOnlyList<ChatMessage>[] prompts = [[new ChatMessage("user", "Q")], [new ChatMessage("user", "P")]];
            // A warm-up on both answers to both prompts in one process, checkpointed, so every run starts from the same bits
            // and the reward has something to choose.
            using (var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
            using (var session = new FineTuningSession(model, tuning, totalSteps: 100))
            {
                var answers = prompts.SelectMany(p => new[] { AnswerSequence(model, p, "yes"), AnswerSequence(model, p, "no") }).ToList();
                for (int i = 0; i < 30; i++)
                {
                    session.Step(answers);
                }

                session.Checkpoint(warm);
            }

            var kind = new PromptsEnvironmentKind([new PromptTask(prompts[0], "yes"), new PromptTask(prompts[1], "no")], name: "reinforcement-cluster-test-prompts");
            const int Episodes = 16, Iterations = 4;
            ClusterRunRecord Run(Device on, ICollectives? cluster, int shards, string spread, bool spreadPasses, int iterations, string? checkpoints = null, string? log = null)
            {
                using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = on });
                var session = new FineTuningSession(model, tuning with { ResumeFrom = warm, Cluster = cluster, ClusterOptions = cluster is null ? null : DpOptions() }, totalSteps: 100);
                using var policy = new CountingPolicy(new LanguageModelPolicy(session, new LanguageModelPolicyOptions { MaxNewTokens = 6, SpreadPasses = spreadPasses }));
                var record = new ClusterRunRecord();
                var options = new ReinforcementOptions
                {
                    Algorithm = loss.Name, Estimator = "group", GroupSize = 4, EpisodesPerIteration = Episodes, Iterations = iterations, Epochs = 1, KlWeight = 0.01f,
                    Seed = 3, Environments = 1, ParallelSteps = false, Shards = shards, Spread = spread, CheckpointFolder = checkpoints, Log = log,
                    EnvironmentOptions = new Dictionary<string, string> { ["reward"] = "metric", ["metric.metric"] = "exact" },
                };
                var trainer = new ReinforcementTrainer<ChatObservation, ChatMessage>(kind, policy, options)
                {
                    OnBatch = (_, batch) =>
                    {
                        record.Episodes.Add([.. batch.Trajectories.Select(t => string.Create(CultureInfo.InvariantCulture,
                            $"{t.Index}/{t.Seed}/{t.Group}: {string.Join(" ", t.Records[0].Context!)} | {string.Join(" ", t.Records[0].Tokens)} → {t.Actions[0].Content} = {t.Return:R}"))]);
                        record.Sampled.Add([.. batch.Trajectories.Select(t => t.Records[0].LogProbabilities)]);
                        record.Advantages.Add(batch.Advantages.ToArray());
                        record.Played.Add(policy.Acted);
                        policy.Acted = 0;
                    },
                    OnIteration = summary =>
                    {
                        long held = model.Device.Backend.GetMemoryUsage().InUse;
                        if (summary.Iteration == 2)
                        {
                            record.HeldAfterSecond = held;
                        }

                        record.HeldAtEnd = held;
                        if (summary.Iteration > 1)
                        {
                            record.Rollout += summary.RolloutTime;
                            record.Update += summary.UpdateTime;
                        }
                    },
                };
                trainer.Train();
                record.Adapters = TrainedValues(model);
                return record;
            }

            string checkpoints = Path.Combine(folder, "checkpoints-two"), log = Path.Combine(folder, "two.jsonl");
            var one = Run(device, null, 4, RolloutSpread.Blocks, true, Iterations);
            var two = DpRun(DpTopology(device, 2), DpOptions(), rank => Run(rank.Device, rank.Collectives, 4, RolloutSpread.Blocks, true, Iterations, checkpoints, log));
            var four = DpRun(DpTopology(device, 4), DpOptions(), rank => Run(rank.Device, rank.Collectives, 4, RolloutSpread.Blocks, true, Iterations));
            var whole = DpRun(DpTopology(device, 4), DpOptions(), rank => Run(rank.Device, rank.Collectives, 4, RolloutSpread.Blocks, false, Iterations));
            var oneInterleaved = Run(device, null, 4, RolloutSpread.Interleaved, true, 2);
            var fourInterleaved = DpRun(DpTopology(device, 4), DpOptions(), rank => Run(rank.Device, rank.Collectives, 4, RolloutSpread.Interleaved, true, 2));
            Check(true, "a language-model policy opens on a session with a cluster (the refusal is gone)");

            bool cpu = device.Type == DeviceType.Cpu;
            float adapterGap = 0f, sampledGap = 0f;
            void Same(string what, ClusterRunRecord alone, ClusterRunRecord[] ranks)
            {
                int world = ranks.Length;
                for (int r = 0; r < world; r++)
                {
                    var rank = ranks[r];
                    Check(rank.Episodes.Count == alone.Episodes.Count, $"{what}, rank {r}: {rank.Episodes.Count} iterations");
                    for (int i = 0; i < alone.Episodes.Count; i++)
                    {
                        Check(rank.Episodes[i].SequenceEqual(alone.Episodes[i]),
                            $"{what}, rank {r}, iteration {i + 1}: the same episodes (numbers, seeds, groups, prompt and reply ids, replies, rewards) as one process:\n  "
                            + string.Join("\n  ", rank.Episodes[i].Zip(alone.Episodes[i]).Where(p => p.First != p.Second).Take(3).Select(p => $"{p.First}\n  vs {p.Second}")));
                        Check(SameBits(rank.Advantages[i], alone.Advantages[i]), $"{what}, rank {r}, iteration {i + 1}: the same advantages, bit for bit");
                        Check(rank.Played[i] == Episodes / world, $"{what}, rank {r}, iteration {i + 1}: played {rank.Played[i]} episodes, its share of {Episodes} on {world} ranks");
                        var sampled = rank.Sampled[i].SelectMany(v => v).ToArray();
                        var reference = alone.Sampled[i].SelectMany(v => v).ToArray();
                        if (i == 0)
                        {
                            Check(SameBits(sampled, reference), $"{what}, rank {r}: the first rollout's sampler log-probabilities bit for bit (the same weights)");
                        }

                        sampledGap = Math.Max(sampledGap, Gap(sampled, reference));
                    }

                    Check(SameBits(rank.Adapters, ranks[0].Adapters), $"{what}: rank {r}'s adapters equal rank 0's, bit for bit");
                    Check(rank.HeldAtEnd <= rank.HeldAfterSecond + (1L << 20), $"{what}, rank {r}: device memory held after iteration 2 ({rank.HeldAfterSecond} bytes) does not grow by the last ({rank.HeldAtEnd} bytes)");
                }

                adapterGap = Math.Max(adapterGap, Gap(ranks[0].Adapters, alone.Adapters));
                AssertClose(alone.Adapters, ranks[0].Adapters, 1e-4f, $"{what}: the adapters against one process (the steps' gradient sums are ordered by rank)");
            }

            Check(one.Played.All(p => p == Episodes) && one.Episodes[0].Length == Episodes, "one process plays all 16 episodes, in 4 shards");
            Same("2 ranks", one, two);
            Same("4 ranks", one, four);
            Same("4 ranks, the passes run whole on every rank", one, whole);
            Same("interleaved, 4 ranks", oneInterleaved, fourInterleaved);
            Check(!oneInterleaved.Episodes[0].SequenceEqual(one.Episodes[0]), "interleaved shards act in other company than blocks: other draws");
            Check(loss.Calls > 0 && loss.SawReference, $"grpo was given the reference (its KL): {loss.Calls} loss calls");
            if (cpu)
            {
                Check(loss.Gap == 0f, $"every update's ratios are exactly 1 on every rank and in one process (the behaviour recomputed, spread over the ranks): largest |log p − behaviour| {loss.Gap}");
                Check(SameBits(whole[0].Adapters, four[0].Adapters), "spreading the behaviour and reference passes over the ranks changes no bit of the adapters (CPU)");
            }

            // Rank 0 alone wrote the checkpoint's folder and state (every rank saved the policy) and the log.
            string written = Path.Combine(checkpoints, "iteration-" + Iterations.ToString("D6", CultureInfo.InvariantCulture));
            Check(File.Exists(Path.Combine(written, "state.json")) && Directory.EnumerateFiles(Path.Combine(written, "policy")).Any()
                  && !Directory.EnumerateDirectories(checkpoints).Any(d => d.EndsWith(".writing", StringComparison.Ordinal)), $"the 2-rank run's checkpoint {written}");
            int lines = File.ReadAllLines(log).Length;
            Check(lines == Iterations, $"the log holds a line per iteration, rank 0's alone: {lines}");

            static string Seconds(TimeSpan t) => t.TotalSeconds.ToString("F2", CultureInfo.InvariantCulture);
            Console.WriteLine($"    the gaps against one process: adapters {adapterGap:E2}, later rollouts' sampler log-probabilities {sampledGap:E2}");
            Console.WriteLine($"    rollout / update seconds over iterations 2-{Iterations}: 1 process {Seconds(one.Rollout)} / {Seconds(one.Update)}; 2 ranks {Seconds(two[0].Rollout)} / {Seconds(two[0].Update)}; "
                              + $"4 ranks {Seconds(four[0].Rollout)} / {Seconds(four[0].Update)}; 4 ranks, the passes whole {Seconds(whole[0].Rollout)} / {Seconds(whole[0].Update)}; "
                              + $"interleaved (iteration 2): 1 process {Seconds(oneInterleaved.Rollout)} / {Seconds(oneInterleaved.Update)}, 4 ranks {Seconds(fourInterleaved[0].Rollout)} / {Seconds(fourInterleaved[0].Update)}");
        }
        finally
        {
            PolicyLosses.Unregister(loss.Name);
            Directory.Delete(folder, true);
        }
    }
}
