// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Globalization;

namespace Idrak.Abstraction.Cluster;

/// <summary>
/// Each rank's memory as the ranks join a cluster (plan 37, N2): measured on the rank's own device at run time
/// (<see cref="Measure"/>) and reported to every other rank (<see cref="Exchange"/>), so a pipeline's stages are cut by the
/// memory each machine has (<see cref="ClusterContext.RankMemory"/>, <see cref="PipelinePlan.For"/>). Nothing here names a
/// card or a memory size: a simulated rank's memory is its carved device's cap (the topology file's "memory").
/// </summary>
public static class ClusterMemory
{
    // The exchange's messages: below every collective's and program's tag (the measured choices take int.MinValue + 21).
    private const int Tag = int.MinValue + 22;

    /// <summary>
    /// The bytes new tensors could take on <paramref name="device"/> now: what the device reports free
    /// (<see cref="Devices.Backend.AvailableMemory"/>), held to its cap less what it holds in use where it has one (a carved
    /// device's, <see cref="ComputeResources"/>' limits); null when it reports neither.
    /// </summary>
    public static long? Measure(Device device)
    {
        ArgumentNullException.ThrowIfNull(device);
        var backend = device.Backend;
        long? available = backend.AvailableMemory();
        var usage = backend.GetMemoryUsage();
        long? room = usage.Limit is { } limit ? Math.Max(0, limit - usage.InUse) : null;
        return available is { } free ? (room is { } capped ? Math.Min(free, capped) : free) : room;
    }

    /// <summary>
    /// Every rank's measured memory, in rank order, the same on every rank: this rank sends <paramref name="mine"/> to each
    /// other rank and takes theirs. Every rank of <paramref name="transport"/>'s group calls it once as it joins.
    /// </summary>
    /// <exception cref="InvalidDataException">A rank's report came short.</exception>
    public static long?[] Exchange(ITransport transport, long? mine)
    {
        ArgumentNullException.ThrowIfNull(transport);
        var all = new long?[transport.WorldSize];
        all[transport.Rank] = mine;
        Span<byte> bytes = stackalloc byte[sizeof(long)];
        BinaryPrimitives.WriteInt64LittleEndian(bytes, mine is { } value && value >= 0 ? value : -1);
        for (int r = 0; r < transport.WorldSize; r++)
        {
            if (r != transport.Rank)
            {
                transport.Send(r, bytes, Tag);
            }
        }

        for (int r = 0; r < transport.WorldSize; r++)
        {
            if (r == transport.Rank)
            {
                continue;
            }

            int got = transport.Receive(r, bytes, Tag);
            if (got != sizeof(long))
            {
                throw new InvalidDataException(string.Create(CultureInfo.InvariantCulture,
                    $"rank {transport.Rank}: rank {r}'s measured memory came short ({got} of {sizeof(long)} bytes)"));
            }

            long reported = BinaryPrimitives.ReadInt64LittleEndian(bytes);
            all[r] = reported >= 0 ? reported : null;
        }

        return all;
    }

    /// <summary>A memory size for people: "3.0 MiB", "11.62 GiB" (invariant culture), or "not reported".</summary>
    public static string Format(long? bytes) => bytes switch
    {
        null => "not reported",
        >= 1L << 30 => string.Create(CultureInfo.InvariantCulture, $"{bytes.Value / (double)(1L << 30):F2} GiB"),
        >= 1L << 20 => string.Create(CultureInfo.InvariantCulture, $"{bytes.Value / (double)(1L << 20):F1} MiB"),
        >= 1L << 10 => string.Create(CultureInfo.InvariantCulture, $"{bytes.Value / (double)(1L << 10):F1} KiB"),
        _ => string.Create(CultureInfo.InvariantCulture, $"{bytes.Value} B"),
    };
}
