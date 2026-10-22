// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak;
using Idrak.Abstraction.Cluster;
using Idrak.Abstraction.Reinforcement;
using Idrak.Cli;
using Idrak.Cluster;
using Idrak.Models;
using Idrak.Text;
using Idrak.Text.Abstractions;
using Idrak.Text.Generation;
using Idrak.Text.Reinforcement;

// Plan 27, R10: on-policy distillation on a cluster. The distiller cuts each step's prompts into R3b's shards, each rank
// samples (and encodes the data's answers of) its shards' prompts, the sequences are exchanged and sorted by prompt number,
// and the step is the session's data-parallel step; a stop is agreed at one step (R7). Every cluster here is an in-process
// one (a thread per rank on devices carved from the device under test, the "sim" transport); the CLI runs in this process.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] DistillationClusterGroup =
    [
        ("distillation cluster: the numbered sequences' exchange round trips ids and trained flags (empty, odd lengths) and refuses other parts; one process with its default shard samples and trains exactly as R3's loop did (the old loop in the test, bit for bit); on 1, 2 and 4 ranks with 4 shards (blocks, and interleaved on 1 and 4) the same samples and data answers every step, the divergences within float rounding, the adapters the same bits on every rank and within R3b's tolerance of one process; every rank samples its shards' share; held device memory does not grow", DistillationClusterReplay),
        ("distillation cluster: a stop asked on one rank of 2 (rank 1's token cancelled after step 2, both agreements) ends every rank before the same step with an agreed stop, the sessions at the same step, and SaveOnce writes one adapter from rank 0; one process cancelled throws a plain cancellation as before", DistillationClusterStop),
        ("distillation cluster: cli: cluster sim -- distill --on-policy on 2 ranks (--shards 4) against one process with --shards 4: both exit 0, rank 0 alone prints, the output holds what one process's holds (one adapter and its manifest, no event log), the first divergence equal and the adapters within 1e-4; reinforce --algorithm distill runs on the cluster; cluster sim -- distill without --on-policy is refused naming why; a bad --spread and --shards are usage errors", DistillationClusterCli),
    ];

    // Five prompts with answers (the mix trains on some of them).
    private static List<ChatTranscript> DistillationPrompts() =>
        [.. new[] { ("Q", "yes"), ("P", "no"), ("R", "yes"), ("S", "no"), ("T", "maybe") }.Select(p => new ChatTranscript([new ChatMessage("user", p.Item1), new ChatMessage("assistant", p.Item2)], []))];

    // A sequence as text: its ids and which train.
    private static string SequenceText(TrainingSequence s) => $"{string.Join(" ", s.Tokens)} | {string.Concat(s.Trained.Select(t => t ? '1' : '0'))}";

    // One rank's record of a distillation run (or the one process's).
    private sealed class DistillRecord
    {
        public List<string[]> Steps { get; } = [];                // per step, its sequences as text, in the order trained

        public List<float> Losses { get; } = [];

        public List<int> SampledHere { get; } = [];

        public List<int> Sampled { get; } = [];

        public int FromData { get; set; }

        public float[] Adapters { get; set; } = [];

        public long HeldAfterSecond { get; set; }

        public long HeldAtEnd { get; set; }
    }

    private static void DistillationClusterReplay(Device device)
    {
        // The exchange of numbered sequences: ids and trained flags as sent (no flag, 1, 9 and 17 of them), images refused.
        (int, TrainingSequence)[] sent =
        [
            (5, new TrainingSequence([], [])), (0, new TrainingSequence([7], [true])),
            (3, new TrainingSequence([.. Enumerable.Range(0, 9).Select(i => (i * 977) + 70000)], [.. Enumerable.Range(0, 9).Select(i => i % 3 == 1)])),
            (9, new TrainingSequence([.. Enumerable.Range(0, 17)], [.. Enumerable.Range(0, 17).Select(i => i is 0 or 8 or 16)])),
        ];
        var back = RolloutExchange.ReadSequences(RolloutExchange.WriteSequences(sent));
        Check(back.Count == sent.Length && back.Zip(sent).All(p => p.First.Number == p.Second.Item1 && SequenceText(p.First.Sequence) == SequenceText(p.Second.Item2)),
            $"numbered sequences round trip: {string.Join("; ", back.Select(b => $"{b.Number}: {SequenceText(b.Sequence)}"))}");
        var withPart = new TrainingSequence([1, 2], [false, true]) { Parts = [new TrainingPart(new ChatText("x"), [(0, 1)])] };
        Throws<NotSupportedException>(() => RolloutExchange.WriteSequences([(0, withPart)]), "a sequence with a part that is not ids is not sent between ranks");

        string student = WriteChatModelWith(TinyChatSpec(), ChatMlTemplate), teacher = WriteChatModelWith(TinyTeacherSpec(), ChatMlTemplate);
        try
        {
            var data = DistillationPrompts();
            const int Steps = 4, Prompts = 8;
            // Small batches (one row of at most 64 positions each), so a step has several and the ranks share them.
            var tuning = SessionOptions(1e-2f) with { BatchTokens = 64 };
            OnPolicyDistillationOptions Settings(int shards, string spread, int steps) =>
                new() { Steps = steps, PromptsPerStep = Prompts, MaxNewTokens = 6, Mix = 0.25f, Seed = 3, Shards = shards, Spread = spread };

            DistillRecord Run(Device on, ICollectives? cluster, int shards, string spread, int steps = Steps)
            {
                using var model = PretrainedModel.Load(student, new PretrainedOptions { Device = on });
                using var teacherModel = PretrainedModel.Load(teacher, new PretrainedOptions { Device = on });
                using var source = DistillationTeachers.FromModel(teacherModel);
                using var session = new FineTuningSession(model, tuning with { Teacher = source, Cluster = cluster, ClusterOptions = cluster is null ? null : DpOptions() }, totalSteps: 100);
                var record = new DistillRecord();
                OnPolicyDistiller.Train(session, data, Settings(shards, spread, steps), new SynchronousProgress<OnPolicyDistillationProgress>(p =>
                {
                    record.Steps.Add([.. p.Sequences.Select(SequenceText)]);
                    record.Losses.Add(p.Loss);
                    record.SampledHere.Add(p.SampledHere);
                    record.Sampled.Add(p.Sampled);
                    record.FromData += p.FromData;
                    long held = model.Device.Backend.GetMemoryUsage().InUse;
                    if (p.Step == 2)
                    {
                        record.HeldAfterSecond = held;
                    }

                    record.HeldAtEnd = held;
                }));
                record.Adapters = TrainedValues(model);
                return record;
            }

            // Rule 71: one process with its default shard (one) samples and trains as R3's loop did. The old loop, written out
            // here on the session's public members, against the distiller on another session of the same models.
            DistillRecord Old()
            {
                using var model = PretrainedModel.Load(student, new PretrainedOptions { Device = device });
                using var teacherModel = PretrainedModel.Load(teacher, new PretrainedOptions { Device = device });
                using var source = DistillationTeachers.FromModel(teacherModel);
                using var session = new FineTuningSession(model, tuning with { Teacher = source }, totalSteps: 100);
                var options = Settings(0, RolloutSpread.Blocks, Steps);
                var loss = FineTuningLosses.OnPolicyDistillation(options.Divergence, options.Temperature, options.Beta);
                var encoder = new ChatTranscriptEncoder(model.JinjaTemplate!, model.Tokenizer!);
                var chat = session.Generator();
                var random = new Random(options.Seed);
                var order = Enumerable.Range(0, data.Count).ToArray();
                int next = order.Length;
                var record = new DistillRecord();
                for (int step = 0; step < options.Steps; step++)
                {
                    var fromData = new List<ChatTranscript>();
                    var sampled = new List<PromptTask>();
                    for (int p = 0; p < options.PromptsPerStep; p++)
                    {
                        if (next == order.Length)
                        {
                            random.Shuffle(order);
                            next = 0;
                        }

                        var transcript = data[order[next++]];
                        if (random.NextDouble() < options.Mix && transcript.Messages is [.., { Role: "assistant" }])
                        {
                            fromData.Add(transcript);
                        }
                        else
                        {
                            sampled.Add(PromptTask.FromTranscript(transcript));
                        }
                    }

                    var sequences = fromData.Select(t => encoder.Encode(t, options.MaxLength)).OfType<TrainingSequence>().ToList();
                    var generation = new GenerationOptions
                    {
                        Temperature = options.SamplingTemperature, TopK = 0, TopP = 1f, MinP = 0f, RepeatPenalty = 1f, Seed = options.Seed + step,
                        NumPredict = options.MaxNewTokens, NumCtx = model.MaxPositions, ReturnTokens = true,
                    };
                    foreach (var reply in sampled.Count == 0 ? [] : chat.ChatBatch([.. sampled.Select(t => new ChatRequest(t.Prompt, t.Tools, t.Think, generation))]))
                    {
                        sequences.Add(EpisodeTurns.Lay([new ActionRecord(reply.Tokens!.Reply, new float[reply.Tokens.Reply.Length], context: reply.Tokens.Prompt)], EpisodeTurns.Turn).Sequences[0]);
                    }

                    record.Losses.Add(session.Step(sequences, loss).Loss);
                    record.Steps.Add([.. sequences.Select(SequenceText)]);
                }

                record.Adapters = TrainedValues(model);
                return record;
            }

            var old = Old();
            var alone = Run(device, null, 0, RolloutSpread.Blocks);
            Check(alone.Steps.Count == Steps && alone.Steps.Zip(old.Steps).All(p => p.First.SequenceEqual(p.Second)) && SameBits([.. alone.Losses], [.. old.Losses])
                  && SameBits(alone.Adapters, old.Adapters), $"one process, its default shard: the same samples, losses and adapters as R3's loop, bit for bit (losses {string.Join(", ", alone.Losses)} | {string.Join(", ", old.Losses)})");
            Check(alone.SampledHere.SequenceEqual(alone.Sampled), "one process samples every sampled prompt itself");

            var one = Run(device, null, 4, RolloutSpread.Blocks);
            var two = DpRun(DpTopology(device, 2), DpOptions(), rank => Run(rank.Device, rank.Collectives, 4, RolloutSpread.Blocks));
            var four = DpRun(DpTopology(device, 4), DpOptions(), rank => Run(rank.Device, rank.Collectives, 4, RolloutSpread.Blocks));
            var oneInterleaved = Run(device, null, 4, RolloutSpread.Interleaved, 2);
            var fourInterleaved = DpRun(DpTopology(device, 4), DpOptions(), rank => Run(rank.Device, rank.Collectives, 4, RolloutSpread.Interleaved, 2));
            Check(one.FromData > 0 && one.Sampled.Sum() > 0, $"the mix trains some prompts on the data's answers ({one.FromData}) and samples the others ({one.Sampled.Sum()})");
            Check(!one.Steps[0].SequenceEqual(alone.Steps[0]), "4 shards answer in other company and from other seeds than one: other samples");

            float adapterGap = 0f, lossGap = 0f;
            void Same(string what, DistillRecord reference, DistillRecord[] ranks)
            {
                int world = ranks.Length;
                for (int r = 0; r < world; r++)
                {
                    var rank = ranks[r];
                    Check(rank.Steps.Count == reference.Steps.Count, $"{what}, rank {r}: {rank.Steps.Count} steps");
                    for (int i = 0; i < reference.Steps.Count; i++)
                    {
                        Check(rank.Steps[i].SequenceEqual(reference.Steps[i]),
                            $"{what}, rank {r}, step {i + 1}: the same sequences (the data's answers, then the samples, by prompt number) as one process:\n  "
                            + string.Join("\n  ", rank.Steps[i].Zip(reference.Steps[i]).Where(p => p.First != p.Second).Take(3).Select(p => $"{p.First}\n  vs {p.Second}")));
                        Check(rank.Sampled[i] == reference.Sampled[i] && ranks.Sum(k => k.SampledHere[i]) == reference.Sampled[i] && rank.SampledHere[i] <= Prompts / world,
                            $"{what}, rank {r}, step {i + 1}: sampled {rank.SampledHere[i]} of the step's {reference.Sampled[i]}, within its shards' {Prompts / world} prompts");
                    }

                    if (r > 0)
                    {
                        Check(SameBits(rank.Adapters, ranks[0].Adapters) && SameBits([.. rank.Losses], [.. ranks[0].Losses]), $"{what}: rank {r}'s adapters and losses equal rank 0's, bit for bit");
                    }

                    Check(rank.HeldAtEnd <= rank.HeldAfterSecond + (1L << 20), $"{what}, rank {r}: device memory held after step 2 ({rank.HeldAfterSecond} bytes) does not grow by the last ({rank.HeldAtEnd} bytes)");
                }

                Check(ranks.All(k => k.SampledHere.Sum() > 0), $"{what}: every rank sampled: {string.Join(", ", ranks.Select(k => k.SampledHere.Sum()))}");
                lossGap = Math.Max(lossGap, Gap([.. reference.Losses], [.. ranks[0].Losses]));
                adapterGap = Math.Max(adapterGap, Gap(reference.Adapters, ranks[0].Adapters));
                AssertClose([.. reference.Losses], [.. ranks[0].Losses], 1e-4f, $"{what}: the divergences against one process (summed per rank, then across the ranks)");
                AssertClose(reference.Adapters, ranks[0].Adapters, 1e-4f, $"{what}: the adapters against one process (the steps' gradient sums are ordered by rank)");
            }

            Same("2 ranks", one, two);
            Same("4 ranks", one, four);
            Same("interleaved, 4 ranks", oneInterleaved, fourInterleaved);
            Check(!oneInterleaved.Steps[0].SequenceEqual(one.Steps[0]), "interleaved shards answer in other company than blocks: other samples");
            Console.WriteLine($"    the gaps against one process: divergences {lossGap:E2}, adapters {adapterGap:E2}");
        }
        finally
        {
            Directory.Delete(student, true);
            Directory.Delete(teacher, true);
        }
    }

    private static void DistillationClusterStop(Device device)
    {
        string student = WriteChatModelWith(TinyChatSpec(), ChatMlTemplate), teacher = WriteChatModelWith(TinyTeacherSpec(), ChatMlTemplate);
        string folder = TempFolder("distillation-cluster-stop");
        try
        {
            var data = DistillationPrompts();
            var settings = new OnPolicyDistillationOptions { Steps = 8, PromptsPerStep = 4, MaxNewTokens = 4, Mix = 0.25f, Seed = 5, Shards = 2 };
            FineTuningSession Open(PretrainedModel model, DistillationTeacher source, ClusterRank? rank, ClusterOptions? options) =>
                new(model, SessionOptions(1e-2f) with { BatchTokens = 64, Teacher = source, Cluster = rank?.Collectives, ClusterOptions = options }, totalSteps: 100);

            foreach (string name in new[] { StopAgreements.Automatic, "all-reduce" })
            {
                var options = StopOptions(name);
                string target = Path.Combine(folder, name);
                var sessionSteps = new int[2];
                var wrote = new bool[2];
                string what = $"distillation, {name}, rank 1 cancelled after step 2";
                var result = StopRun(device, 2, options, rank =>
                {
                    using var cancel = new CancellationTokenSource();
                    using var model = PretrainedModel.Load(student, new PretrainedOptions { Device = rank.Device });
                    using var teacherModel = PretrainedModel.Load(teacher, new PretrainedOptions { Device = rank.Device });
                    using var source = DistillationTeachers.FromModel(teacherModel);
                    using var session = Open(model, source, rank, options);
                    try
                    {
                        OnPolicyDistiller.Train(session, data, settings, new SynchronousProgress<OnPolicyDistillationProgress>(p =>
                        {
                            if (rank.Rank == 1 && p.Step == 2)
                            {
                                cancel.Cancel();
                            }
                        }), cancel.Token);
                    }
                    finally
                    {
                        sessionSteps[rank.Rank] = session.Steps;
                        wrote[rank.Rank] = session.SaveOnce(target);             // the adapter so far, as the command writes it after a stop
                    }
                }, what);
                long step = AgreedStop(result, what);
                // Asked after step 2's update: all-reduce carries the flag in step 3's gradient header and stops before step
                // index 3; shared stops after the last boundary any rank passed (2 or 3).
                Check(name == "all-reduce" ? step == 3 : step is 2 or 3, $"{what}: the ranks stopped before step index {step}");
                Check(sessionSteps.All(s => s == step), $"{what}: every session did the steps before the stop: {string.Join(", ", sessionSteps)}");
                Check(wrote.SequenceEqual([true, false]) && File.Exists(Path.Combine(target, "adapter_model.safetensors")), $"{what}: rank 0 alone wrote the adapter: {string.Join(", ", wrote)}");
            }

            // One process: a cancelled token throws a plain cancellation before the first step, as before.
            using var alone = PretrainedModel.Load(student, new PretrainedOptions { Device = device });
            using var aloneTeacher = PretrainedModel.Load(teacher, new PretrainedOptions { Device = device });
            using var aloneSource = DistillationTeachers.FromModel(aloneTeacher);
            using var aloneSession = Open(alone, aloneSource, null, null);
            using var cancelled = new CancellationTokenSource();
            cancelled.Cancel();
            try
            {
                OnPolicyDistiller.Train(aloneSession, data, settings, null, cancelled.Token);
                Check(false, "one process: a cancelled distillation throws");
            }
            catch (OperationCanceledException e)
            {
                Check(e is not ClusterRunStoppedException && aloneSession.Steps == 0, $"one process: a plain cancellation before the first step, as before ({e.GetType().Name})");
            }
        }
        finally
        {
            Directory.Delete(folder, true);
            Directory.Delete(student, true);
            Directory.Delete(teacher, true);
        }
    }

    private static void DistillationClusterCli(Device device)
    {
        string folder = TempFolder();
        string student = WriteChatModelWith(TinyChatSpec(), ChatMlTemplate), teacher = WriteChatModelWith(TinyTeacherSpec(), ChatMlTemplate);
        try
        {
            string data = Path.Combine(folder, "chats.jsonl");
            File.WriteAllLines(data, ["{\"question\": \"Q\", \"answer\": \"yes\"}", "{\"question\": \"R\", \"answer\": \"no\"}", "{\"question\": \"S\", \"answer\": \"maybe\"}"]);
            string topology = Path.Combine(folder, "two.json");
            File.WriteAllText(topology, DpTopology(device, 2).ToJson());
            string[] Session() => ["--rank", "4", "--lora-alpha", "8", "--lr", "1e-2", "--max-length", "64", "--batch-tokens", "64", "-d", device.ToString(), "--cache", folder];
            string[] Distill(string output, params string[] more) => ["distill", "--teacher", teacher, "--student", student, "--data", data, "--on-policy", "--mix", "0.25",
                "--steps", "2", "--batch-size", "8", "--max-new", "4", "--seed", "3", "-o", output, .. Session(), .. more];
            string alone = Path.Combine(folder, "alone"), ranks = Path.Combine(folder, "ranks");
            var one = TrainCli(Distill(alone, "--shards", "4"));
            var two = ClusterCli(["cluster", "sim", "--topology", topology, "--", .. Distill(ranks, "--shards", "4")]);
            Check(one.Code == 0, $"one process, 4 shards: exit {one.Code}\n{one.Out}\n{one.Err}");
            Check(two.Code == 0 && two.Output.Contains("network time on the virtual clock: rank 0", StringComparison.Ordinal) && two.Output.Contains("rank 1", StringComparison.Ordinal),
                $"cluster sim runs distill --on-policy to the end on both ranks: exit {two.Code}\n{two.Output}\n{two.Error}");
            static string[] Lines(string text, string start) => [.. text.Split('\n').Select(l => l.TrimEnd('\r')).Where(l => l.StartsWith(start, StringComparison.Ordinal))];
            var doneAlone = Lines(one.Out, "distilled in 2 steps");
            var doneRanks = Lines(two.Output, "distilled in 2 steps");
            Check(doneAlone.Length == 1 && doneRanks.Length == 1 && doneRanks[0].Contains("adapter written to", StringComparison.Ordinal)
                  && Lines(two.Output, "on-policy distillation: 2 steps of 8 prompts").Single().Contains("4 shards (blocks)", StringComparison.Ordinal),
                $"rank 0 alone prints its lines (once each): {two.Output}");

            // The first step's divergence: the same samples on the same weights (printed to 4 places).
            static float First(string line) => float.Parse(line[(line.IndexOf("divergence ", StringComparison.Ordinal) + 11)..].Split(' ')[0], CultureInfo.InvariantCulture);
            Check(MathF.Abs(First(doneAlone[0]) - First(doneRanks[0])) <= 2e-4f, $"the first divergence is one process's: {doneRanks[0]} | {doneAlone[0]}");

            // The output folder: what one process writes, once (no event log).
            static string[] Entries(string path) => [.. Directory.EnumerateFileSystemEntries(path).Select(e => Path.GetFileName(e)).Order(StringComparer.Ordinal)];
            Check(Entries(ranks).SequenceEqual(Entries(alone)) && Entries(ranks).Contains("adapter_model.safetensors") && Entries(ranks).Contains("idrak-tuning.json")
                  && !Entries(ranks).Contains("cluster-events.idrklog"), $"the cluster's output holds what one process's holds: {string.Join(", ", Entries(ranks))} | {string.Join(", ", Entries(alone))}");
            static float[] Adapter(string path)
            {
                using var reader = SafeTensorsReader.Open(Path.Combine(path, "adapter_model.safetensors"));
                return [.. reader.Tensors.Keys.Order(StringComparer.Ordinal).SelectMany(n => reader.Read(n))];
            }

            float[] aloneValues = Adapter(alone), ranksValues = Adapter(ranks);
            AssertClose(aloneValues, ranksValues, 1e-4f, "the 2-rank adapter against one process's");
            Console.WriteLine($"    cluster sim -- distill --on-policy: the adapter's largest gap to one process {Gap(aloneValues, ranksValues):E2}");

            // reinforce --algorithm distill on the cluster: the same distiller, rank 0 writing the adapter.
            string reinforced = Path.Combine(folder, "reinforced");
            var reinforce = ClusterCli(["cluster", "sim", "--topology", topology, "--", "reinforce", student, "--env", "prompts", "--data", data, "--algorithm", "distill",
                "--teacher", teacher, "--iterations", "2", "--episodes", "4", "--shards", "4", "-o", reinforced, .. TinySession(device, folder)]);
            Check(reinforce.Code == 0 && File.Exists(Path.Combine(reinforced, "adapter_model.safetensors")) && Lines(reinforce.Output, "distilled in 2 steps").Length == 1,
                $"reinforce --algorithm distill runs on the cluster: exit {reinforce.Code}\n{reinforce.Output}\n{reinforce.Error}");

            // Usage errors: distill without --on-policy on the simulated cluster, a bad split, a negative shard count.
            var offline = ClusterCli("cluster", "sim", "--topology", topology, "--", "distill", "--teacher", teacher, "--student", student, "--data", data, "-o", Path.Combine(folder, "x"));
            Check(offline.Code == ExitCodes.Usage && offline.Error.Contains("--on-policy only", StringComparison.Ordinal), $"cluster sim refuses distill's other ways naming why: {offline.Error}");
            var spread = TrainCli(Distill(Path.Combine(folder, "s"), "--spread", "diagonal"));
            Check(spread.Code == ExitCodes.Usage && spread.Err.Contains("--spread diagonal", StringComparison.Ordinal), $"a bad --spread is named: {spread.Err}");
            var shards = TrainCli(Distill(Path.Combine(folder, "n"), "--shards", "-1"));
            Check(shards.Code == ExitCodes.Usage, $"a negative --shards is a usage error: {shards.Err}");
        }
        finally
        {
            Directory.Delete(folder, true);
            Directory.Delete(student, true);
            Directory.Delete(teacher, true);
        }
    }
}
