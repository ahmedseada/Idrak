// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Cluster;
using Idrak.Abstraction.Reinforcement;

namespace Idrak.Text.Reinforcement;

// What the ranks of a cluster exchange for a language-model policy's rollouts (Plan 27, R3b): every rank's values or
// bytes gathered on every rank in rank order (an all-gather of each rank's count, then of the values padded to the longest
// rank's), and the codec of text episodes sent as bytes. Gathered values arrive as sent: an all-gather copies them (no
// sum, no wire narrowing), and bytes travel three to a value as an integer below 2^24, which every float holds exactly.
internal static class RolloutExchange
{
    private const int Version = 1;

    // Every rank's `mine`, in rank order (a collective: every rank calls it at the same point).
    public static float[][] Gather(ICollectives collectives, ReadOnlySpan<float> mine)
    {
        int world = collectives.WorldSize, rank = collectives.Rank;
        var result = new float[world][];
        if (world == 1)
        {
            result[0] = mine.ToArray();
            return result;
        }

        // Each rank's count as two values below 2^24 (exact as floats), then the values padded to the longest count.
        var counts = new float[2 * world];
        counts[2 * rank] = mine.Length >> 20;
        counts[(2 * rank) + 1] = mine.Length & 0xFFFFF;
        AllGather(collectives, counts, 2);
        var sizes = new int[world];
        int most = 0;
        for (int r = 0; r < world; r++)
        {
            sizes[r] = ((int)counts[2 * r] << 20) | (int)counts[(2 * r) + 1];
            most = Math.Max(most, sizes[r]);
        }

        if (most == 0)
        {
            Array.Fill(result, []);
            return result;
        }

        long total = (long)most * world;
        if (total > Array.MaxLength)
        {
            throw new InvalidOperationException($"The ranks' values ({world} × {most}) are more than one array holds.");
        }

        float[] buffer = ArrayPool<float>.Shared.Rent((int)total);
        try
        {
            var all = buffer.AsSpan(0, (int)total);
            all.Clear();
            mine.CopyTo(all.Slice(rank * most));
            AllGather(collectives, all, most);
            for (int r = 0; r < world; r++)
            {
                result[r] = all.Slice(r * most, sizes[r]).ToArray();
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }

        return result;
    }

    // Every rank's bytes, in rank order (a collective): three bytes a value, the byte count first.
    public static byte[][] Gather(ICollectives collectives, ReadOnlySpan<byte> mine)
    {
        int values = (mine.Length + 2) / 3;
        float[] packed = ArrayPool<float>.Shared.Rent(values + 2);
        float[][] gathered;
        try
        {
            packed[0] = mine.Length >> 20;
            packed[1] = mine.Length & 0xFFFFF;
            for (int i = 0, at = 0; i < values; i++, at += 3)
            {
                int value = mine[at];
                value |= at + 1 < mine.Length ? mine[at + 1] << 8 : 0;
                value |= at + 2 < mine.Length ? mine[at + 2] << 16 : 0;
                packed[i + 2] = value;
            }

            gathered = Gather(collectives, packed.AsSpan(0, values + 2));
        }
        finally
        {
            ArrayPool<float>.Shared.Return(packed);
        }

        var result = new byte[gathered.Length][];
        for (int r = 0; r < gathered.Length; r++)
        {
            var from = gathered[r];
            int length = ((int)from[0] << 20) | (int)from[1];
            var bytes = new byte[length];
            for (int at = 0; at < length; at++)
            {
                bytes[at] = (byte)((int)from[2 + (at / 3)] >> (8 * (at % 3)));
            }

            result[r] = bytes;
        }

        return result;
    }

    // `all` holds this rank's `count` values at Rank · count; every rank's afterwards: straight from the host where the
    // collectives take host values, else through two device buffers made for the call (returned to the device's pool).
    private static void AllGather(ICollectives collectives, Span<float> all, int count)
    {
        if (collectives is IHostValueCollectives { SupportsHostValues: true } host && host.TryAllGather(all, count))
        {
            return;
        }

        var backend = collectives.Device.Backend;
        using var input = Tensor.PersistentZeros([count], collectives.Device);
        using var output = Tensor.PersistentZeros([all.Length], collectives.Device);
        backend.Upload(all.Slice(collectives.Rank * count, count), input.Storage);
        collectives.AllGather(input.Storage, count, output.Storage);
        backend.Download(output.Storage, all);
    }

    // The episodes as bytes: the tool lists of their observations once each (by reference), then each episode with its
    // messages once each (by reference: an episode's observations share their earlier messages), its observations as
    // indices into them, its actions, records, rewards, ends and info.
    public static byte[] Write(IReadOnlyList<Trajectory<ChatObservation, ChatMessage>> trajectories)
    {
        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(Version);
            var tools = new Dictionary<IReadOnlyList<ToolDefinition>, int>(ReferenceEqualityComparer.Instance);
            foreach (var trajectory in trajectories)
            {
                foreach (var observation in trajectory.Observations)
                {
                    if (observation?.Tools is { } list && !tools.ContainsKey(list))
                    {
                        tools[list] = tools.Count;
                    }
                }
            }

            writer.Write(tools.Count);
            foreach (var list in tools.Keys)
            {
                writer.Write(list.Count);
                foreach (var tool in list)
                {
                    writer.Write(tool.Name);
                    Optional(writer, tool.Description);
                    Optional(writer, tool.Parameters?.ToJsonString());
                }
            }

            writer.Write(trajectories.Count);
            var messages = new Dictionary<ChatMessage, int>(ReferenceEqualityComparer.Instance);
            foreach (var trajectory in trajectories)
            {
                writer.Write(trajectory.Index);
                writer.Write(trajectory.Seed);
                writer.Write(trajectory.Group);
                writer.Write(trajectory.Terminated);
                writer.Write(trajectory.Truncated);
                writer.Write(trajectory.FinalValue);
                writer.Write(trajectory.Steps);
                foreach (float reward in trajectory.Rewards)
                {
                    writer.Write(reward);
                }

                writer.Write(trajectory.Info?.Count ?? -1);
                foreach (var (key, value) in trajectory.Info ?? new Dictionary<string, string>())
                {
                    writer.Write(key);
                    writer.Write(value);
                }

                messages.Clear();
                foreach (var observation in trajectory.Observations)
                {
                    foreach (var message in (observation ?? throw new InvalidOperationException($"Episode {trajectory.Index} holds a null observation.")).Messages)
                    {
                        messages.TryAdd(message, messages.Count);
                    }
                }

                foreach (var action in trajectory.Actions)
                {
                    messages.TryAdd(action ?? throw new InvalidOperationException($"Episode {trajectory.Index} holds a null action."), messages.Count);
                }

                writer.Write(messages.Count);
                foreach (var message in messages.Keys)
                {
                    WriteMessage(writer, message);
                }

                foreach (var observation in trajectory.Observations)
                {
                    writer.Write(observation.Messages.Count);
                    foreach (var message in observation.Messages)
                    {
                        writer.Write(messages[message]);
                    }

                    writer.Write(observation.Tools is { } list ? tools[list] : -1);
                    writer.Write((byte)(observation.Think switch { null => 0, false => 1, true => 2 }));
                }

                foreach (var action in trajectory.Actions)
                {
                    writer.Write(messages[action]);
                }

                foreach (var record in trajectory.Records)
                {
                    Ints(writer, record.Tokens);
                    writer.Write(record.LogProbabilities.Length);
                    foreach (float value in record.LogProbabilities)
                    {
                        writer.Write(value);
                    }

                    writer.Write(record.Value);
                    if (record.Context is { } context)
                    {
                        Ints(writer, context);
                    }
                    else
                    {
                        writer.Write(-1);
                    }
                }
            }
        }

        return stream.ToArray();
    }

    // The episodes Write wrote.
    public static List<Trajectory<ChatObservation, ChatMessage>> Read(byte[] data)
    {
        using var reader = new BinaryReader(new MemoryStream(data, writable: false), System.Text.Encoding.UTF8);
        int version = reader.ReadInt32();
        if (version != Version)
        {
            throw new InvalidDataException($"A rank sent episodes in exchange format {version}; this one reads {Version} (every rank runs the same Idrak).");
        }

        var tools = new IReadOnlyList<ToolDefinition>[reader.ReadInt32()];
        for (int t = 0; t < tools.Length; t++)
        {
            var list = new ToolDefinition[reader.ReadInt32()];
            for (int i = 0; i < list.Length; i++)
            {
                string name = reader.ReadString();
                string? description = Optional(reader);
                list[i] = new ToolDefinition(name, description, Optional(reader) is { } parameters ? JsonNode.Parse(parameters) : null);
            }

            tools[t] = list;
        }

        int episodes = reader.ReadInt32();
        var trajectories = new List<Trajectory<ChatObservation, ChatMessage>>(episodes);
        while (trajectories.Count < episodes)
        {
            int index = reader.ReadInt32(), seed = reader.ReadInt32(), group = reader.ReadInt32();
            bool terminated = reader.ReadBoolean(), truncated = reader.ReadBoolean();
            float finalValue = reader.ReadSingle();
            int steps = reader.ReadInt32();
            var rewards = new float[steps];
            for (int s = 0; s < steps; s++)
            {
                rewards[s] = reader.ReadSingle();
            }

            int infos = reader.ReadInt32();
            Dictionary<string, string>? info = null;
            if (infos >= 0)
            {
                info = new Dictionary<string, string>(infos, StringComparer.Ordinal);
                for (int i = 0; i < infos; i++)
                {
                    string key = reader.ReadString();
                    info[key] = reader.ReadString();
                }
            }

            var messages = new ChatMessage[reader.ReadInt32()];
            for (int m = 0; m < messages.Length; m++)
            {
                messages[m] = ReadMessage(reader);
            }

            var observations = new ChatObservation[steps + 1];
            for (int o = 0; o < observations.Length; o++)
            {
                var conversation = new ChatMessage[reader.ReadInt32()];
                for (int m = 0; m < conversation.Length; m++)
                {
                    conversation[m] = messages[reader.ReadInt32()];
                }

                int list = reader.ReadInt32();
                byte think = reader.ReadByte();
                observations[o] = new ChatObservation(conversation, list >= 0 ? tools[list] : null, think switch { 0 => null, 1 => false, _ => true });
            }

            var actions = new ChatMessage[steps];
            for (int s = 0; s < steps; s++)
            {
                actions[s] = messages[reader.ReadInt32()];
            }

            var records = new ActionRecord[steps];
            for (int s = 0; s < steps; s++)
            {
                int[] tokens = Ints(reader) ?? [];
                var logProbabilities = new float[reader.ReadInt32()];
                for (int i = 0; i < logProbabilities.Length; i++)
                {
                    logProbabilities[i] = reader.ReadSingle();
                }

                float value = reader.ReadSingle();
                records[s] = new ActionRecord(tokens, logProbabilities, value, Ints(reader));
            }

            trajectories.Add(new Trajectory<ChatObservation, ChatMessage>(index, seed, observations, actions, records, rewards, terminated, truncated, group, finalValue, info));
        }

        return trajectories;
    }

    private static void WriteMessage(BinaryWriter writer, ChatMessage message)
    {
        writer.Write(message.Role);
        writer.Write(message.Parts.Count);
        foreach (var part in message.Parts)
        {
            writer.Write(part is ChatText text ? text.Text
                : throw new NotSupportedException($"Rollouts spread over a cluster's ranks send text conversations: a message part of kind '{part.Kind}' is not sent between ranks."));
        }

        Optional(writer, message.Thinking);
        writer.Write(message.ToolCalls?.Count ?? -1);
        foreach (var call in message.ToolCalls ?? [])
        {
            writer.Write(call.Name);
            writer.Write(call.Arguments.ToJsonString());
        }

        Optional(writer, message.ToolName);
    }

    private static ChatMessage ReadMessage(BinaryReader reader)
    {
        string role = reader.ReadString();
        var parts = new ChatPart[reader.ReadInt32()];
        for (int p = 0; p < parts.Length; p++)
        {
            parts[p] = new ChatText(reader.ReadString());
        }

        string? thinking = Optional(reader);
        int calls = reader.ReadInt32();
        ToolCall[]? toolCalls = calls < 0 ? null : new ToolCall[calls];
        for (int c = 0; c < calls; c++)
        {
            string name = reader.ReadString();
            toolCalls![c] = new ToolCall(name, JsonNode.Parse(reader.ReadString()) as JsonObject ?? new JsonObject());
        }

        return new ChatMessage(role, parts, thinking, toolCalls, Optional(reader));
    }

    private static void Optional(BinaryWriter writer, string? text)
    {
        writer.Write(text is not null);
        if (text is not null)
        {
            writer.Write(text);
        }
    }

    private static string? Optional(BinaryReader reader) => reader.ReadBoolean() ? reader.ReadString() : null;

    private static void Ints(BinaryWriter writer, int[] values)
    {
        writer.Write(values.Length);
        foreach (int value in values)
        {
            writer.Write(value);
        }
    }

    // A list Ints wrote, or null for the -1 written in its place.
    private static int[]? Ints(BinaryReader reader)
    {
        int count = reader.ReadInt32();
        if (count < 0)
        {
            return null;
        }

        var values = new int[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = reader.ReadInt32();
        }

        return values;
    }
}
