// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Cli;
using Idrak.Text.Abstractions;
using Idrak.Text.Reinforcement;

// Plan 27, R10: on-policy distillation on a cluster. The sequences each rank samples are exchanged between the ranks as
// numbered sequences; distill's usage on a cluster is checked before a model loads. The CLI runs in this process.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] DistillationClusterGroup =
    [
        ("distillation cluster: the numbered sequences' exchange round trips ids and trained flags (empty, odd lengths) and refuses other parts", DistillationClusterReplay),
        ("distillation cluster: cli: cluster sim -- distill without --on-policy is refused naming why; a bad --spread and --shards are usage errors", DistillationClusterCli),
    ];

    // A sequence as text: its ids and which train.
    private static string SequenceText(TrainingSequence s) => $"{string.Join(" ", s.Tokens)} | {string.Concat(s.Trained.Select(t => t ? '1' : '0'))}";

    private static void DistillationClusterReplay(Device device)
    {
        _ = device;
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
