// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;

namespace Idrak.Cluster;

/// <summary>
/// Every rank of a cluster run runs rank 0's measured kernel choices (<see cref="ClusterOptions.ShareTuning"/>), through
/// the backends' <c>IMeasuredChoicesProvider</c>s only (none on the new design yet: nothing is shared), before any rank's work makes or stores a choice of its own:
/// ranks that each measured their own could pick different algorithms for one shape (results that differ in the last
/// bits), so a seeded run on N ranks would not run the kernels one process runs. Rank 0 measures (or reads its cache) as
/// always. In one process (<see cref="InProcessCluster"/>) each rank's backend follows rank 0's live, from before the ranks'
/// threads start to the end of the run; between processes (<see cref="LocalhostCluster.Join(ClusterOptions)"/>) rank 0 sends
/// what it knows as the ranks join and each rank takes it. A rank on another device or driver than rank 0's keeps its own
/// (choices are per device and driver: another device's do not apply, and a cluster of different GPUs is a valid run), and
/// says so (<see cref="MeasuredChoicesShared"/>, and <see cref="ClusterRank.SharedChoices"/>).
/// </summary>
internal static class ChoiceSharing
{
    // The exchange's messages between processes: below every collective's and program's tag.
    private const int Tag = int.MinValue + 21;

    /// <summary>
    /// In one process, before the ranks' threads start: each rank's outcome (null: nothing shared). No backend shares its
    /// choices on the new design yet (the providers' registry is gone), so nothing is shared.
    /// </summary>
    public static Following Follow(IReadOnlyList<Device> devices, ClusterOptions options)
    {
        ArgumentNullException.ThrowIfNull(devices);
        ArgumentNullException.ThrowIfNull(options);
        return new Following(new MeasuredChoicesResult?[devices.Count]);
    }

    /// <summary>
    /// Between processes, as a rank joins (every rank calls it, whatever its setting, so no rank waits for a message that
    /// never comes): rank 0 sends what its backend knows (nothing when sharing is off or its device measures nothing) to every
    /// other rank, and each takes it when its own setting shares. Rank 0's result is null; another rank's is null when nothing
    /// came or its setting is off.
    /// </summary>
    public static MeasuredChoicesResult? Exchange(ITransport transport, Device device, ClusterOptions options)
    {
        ArgumentNullException.ThrowIfNull(transport);
        Span<byte> header = stackalloc byte[sizeof(int)];
        if (transport.WorldSize < 2)
        {
            return null;
        }

        if (transport.Rank == 0)
        {
            byte[] payload = [];                                        // no backend shares its choices on the new design yet
            BinaryPrimitives.WriteInt32LittleEndian(header, payload.Length);
            for (int r = 1; r < transport.WorldSize; r++)
            {
                transport.Send(r, header, Tag);
                if (payload.Length > 0)
                {
                    transport.Send(r, payload, Tag);
                }
            }

            return null;
        }

        if (transport.Receive(0, header, Tag) != header.Length)
        {
            throw new InvalidDataException($"rank {transport.Rank}: rank 0's measured choices came without their length");
        }

        int length = BinaryPrimitives.ReadInt32LittleEndian(header);
        if (length <= 0)
        {
            return null;
        }

        var bytes = new byte[length];
        int got = transport.Receive(0, bytes, Tag);
        if (got != length)
        {
            throw new InvalidDataException($"rank {transport.Rank}: rank 0's measured choices came short ({got} of {length} bytes)");
        }

        if (!options.ShareTuning)
        {
            return null;
        }

        var choices = MeasuredChoices.FromBytes(bytes);
        var mine = device.Backend;
        var result = new MeasuredChoicesResult(false, 0, $"kept its own: rank 0's choices are {choices.Kind} choices ({choices.Device}); {mine.Name} measures none of its own to replace");
        Report(transport.Rank, device, result);
        return result;
    }

    private static void Report(int rank, Device device, MeasuredChoicesResult? result)
    {
        if (result is not null && Telemetry.IsEnabled(TelemetryLevel.Network))
        {
            Telemetry.ChoicesShared(new MeasuredChoicesShared(rank, device.ToString(), result.Taken, result.Choices, result.Detail));
        }
    }

    /// <summary>The ranks' outcomes in one process.</summary>
    public sealed class Following(MeasuredChoicesResult?[] results) : IDisposable
    {
        /// <summary>Each rank's outcome (null: nothing shared).</summary>
        public IReadOnlyList<MeasuredChoicesResult?> Results => results;

        public void Dispose()
        {
        }
    }
}
