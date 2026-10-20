// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;
using Idrak.Abstraction.Cluster;

namespace Idrak.Text;

/// <summary>
/// A checkpoint streamed to rank 0 (plan 23, C8b): the ranks holding a model's parts (pipeline stages, parameter shards)
/// answer rank 0's requests one tensor at a time while it writes the one-process files, so no rank ever holds more than
/// one tensor of another's and no rank gathers the whole weights or optimizer state. Rank 0 asks, in the order it writes:
/// an owner's optimizer entries (names and shapes, for the header), then each weight and each entry's values; then it says
/// the save ended. Every rank's part goes over the transport the run already has (no shared file system needed: the ranks
/// of a cluster write nothing but rank 0's folder).
/// </summary>
internal static class StreamedCheckpoint
{
    // Tags of their own beside the pipeline's (1 << 20 to 7 << 20): a request, then a reply's length and bytes.
    private const int RequestTag = 8 << 20, ReplyTag = (8 << 20) + 2;
    private const int End = -1, Weight = 0, Entries = 1, Entry = 2;

    /// <summary>Rank 0: <paramref name="owner"/>'s optimizer entries, names and shapes, in its order.</summary>
    public static List<(string Name, int[] Shape)> AskEntries(ITransport transport, int owner)
    {
        Ask(transport, owner, Entries, 0);
        var bytes = Reply(transport, owner);
        using var reader = new BinaryReader(new MemoryStream(bytes), Encoding.UTF8);
        int count = reader.ReadInt32();
        var entries = new List<(string, int[])>(count);
        for (int i = 0; i < count; i++)
        {
            string name = reader.ReadString();
            var shape = new int[reader.ReadInt32()];
            for (int d = 0; d < shape.Length; d++)
            {
                shape[d] = reader.ReadInt32();
            }

            entries.Add((name, shape));
        }

        return entries;
    }

    /// <summary>Rank 0: the values of <paramref name="owner"/>'s weight <paramref name="index"/> (the owner's numbering).</summary>
    public static float[] AskWeight(ITransport transport, int owner, int index)
    {
        Ask(transport, owner, Weight, index);
        return MemoryMarshal.Cast<byte, float>(Reply(transport, owner).AsSpan()).ToArray();
    }

    /// <summary>Rank 0: the values of <paramref name="owner"/>'s optimizer entry <paramref name="index"/> (in <see cref="AskEntries"/>' order).</summary>
    public static float[] AskEntry(ITransport transport, int owner, int index)
    {
        Ask(transport, owner, Entry, index);
        return MemoryMarshal.Cast<byte, float>(Reply(transport, owner).AsSpan()).ToArray();
    }

    /// <summary>Rank 0: the save ended; <paramref name="owner"/> stops answering.</summary>
    public static void Finish(ITransport transport, int owner) => Ask(transport, owner, End, 0);

    /// <summary>
    /// An owner: answers <paramref name="writer"/>'s requests until it says the save ended: weight i's values from
    /// <paramref name="weight"/>, the entries of <paramref name="state"/> (null: none).
    /// </summary>
    public static void Serve(ITransport transport, int writer, Func<int, float[]> weight, IReadOnlyDictionary<string, OptimizerStateValue>? state)
    {
        var entries = state?.ToList() ?? [];
        Span<byte> request = stackalloc byte[2 * sizeof(int)];
        while (true)
        {
            int got = transport.Receive(writer, request, RequestTag);
            if (got != request.Length)
            {
                throw new InvalidDataException($"A checkpoint request of {got} bytes from rank {writer}, not {request.Length}.");
            }

            int kind = BinaryPrimitives.ReadInt32LittleEndian(request), index = BinaryPrimitives.ReadInt32LittleEndian(request[sizeof(int)..]);
            switch (kind)
            {
                case End:
                    return;
                case Weight:
                    Send(transport, writer, MemoryMarshal.AsBytes(weight(index).AsSpan()));
                    break;
                case Entries:
                    using (var stream = new MemoryStream())
                    {
                        using (var w = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true))
                        {
                            w.Write(entries.Count);
                            foreach (var (name, value) in entries)
                            {
                                w.Write(name);
                                w.Write(value.Shape.Length);
                                foreach (int d in value.Shape)
                                {
                                    w.Write(d);
                                }
                            }
                        }

                        Send(transport, writer, stream.ToArray());
                    }

                    break;
                case Entry:
                    Send(transport, writer, MemoryMarshal.AsBytes(entries[index].Value.Values.AsSpan()));
                    break;
                default:
                    throw new InvalidDataException($"An unknown checkpoint request {kind} from rank {writer}.");
            }
        }
    }

    private static void Ask(ITransport transport, int owner, int kind, int index)
    {
        Span<byte> request = stackalloc byte[2 * sizeof(int)];
        BinaryPrimitives.WriteInt32LittleEndian(request, kind);
        BinaryPrimitives.WriteInt32LittleEndian(request[sizeof(int)..], index);
        transport.Send(owner, request, RequestTag);
    }

    private static void Send(ITransport transport, int to, ReadOnlySpan<byte> bytes)
    {
        Span<byte> length = stackalloc byte[sizeof(int)];
        BinaryPrimitives.WriteInt32LittleEndian(length, bytes.Length);
        transport.Send(to, length, ReplyTag);
        transport.Send(to, bytes, ReplyTag + 1);
    }

    private static byte[] Reply(ITransport transport, int from)
    {
        Span<byte> length = stackalloc byte[sizeof(int)];
        transport.Receive(from, length, ReplyTag);
        var bytes = new byte[BinaryPrimitives.ReadInt32LittleEndian(length)];
        transport.Receive(from, bytes, ReplyTag + 1);
        return bytes;
    }
}
