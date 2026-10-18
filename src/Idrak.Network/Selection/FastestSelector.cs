// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Networking.Selection;

/// <summary>
/// "fastest" (plan 23): for a pair of ranks, the first transport available between them in the order fastest first
/// (in-process simulation, peer copies, host-staged copies, shared memory, RDMA, a direct cable, TCP, datagrams), then any
/// other registered transport in registration order; the reason names the transport's own.
/// </summary>
internal sealed class FastestSelector : ITransportSelector
{
    private static readonly string[] Order = ["sim", "nvlink", "pcie-host", "shared-memory", "rdma", "direct-link", "tcp", "udp"];

    public string Name => "fastest";

    public string Summary => "the first available of sim, nvlink, pcie-host, shared-memory, rdma, direct-link, tcp, udp, then any other";

    public (string Transport, string Reason) Choose(int rankA, int rankB, IClusterTopology topology, IReadOnlyList<TransportAvailability> available)
    {
        ArgumentNullException.ThrowIfNull(available);
        var ranked = available.Select((a, i) => (Answer: a, Rank: Array.FindIndex(Order, o => o.Equals(a.Transport, StringComparison.OrdinalIgnoreCase)) is var at and >= 0 ? at : Order.Length + i));
        foreach (var (answer, _) in ranked.OrderBy(r => r.Rank))
        {
            if (answer.Available)
            {
                return (answer.Transport, $"fastest available: {answer.Reason}");
            }
        }

        throw new NotSupportedException($"No transport joins ranks {rankA} and {rankB}: " +
            (available.Count == 0 ? "none is registered" : string.Join("; ", available.Select(a => $"{a.Transport}: {a.Reason}"))) + ".");
    }
}
