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

// Plan 27, R3b: rollouts on a cluster. R2's collector cuts a rollout's episodes into shards and the shards into runs over
// the ranks, by the settings and their variables; the text episodes the ranks exchange round trip.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] ReinforcementClusterGroup =
    [
        ("reinforcement cluster: the split: shards from the episodes' numbers (blocks and interleaved, sizes within one, duplicates refused), shards to ranks in runs; the settings and their variables; text episodes round trip (messages and tool lists shared as they were, images refused)", ClusterSpreadRules),
    ];

    internal static void ClusterSpreadRules(Device device)
    {
        _ = device;

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
}
