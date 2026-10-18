// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Networking.Abstractions;

namespace Idrak.Networking.Transports;

/// <summary>
/// "pcie-host" (plan 23): ranks of one process on devices of one machine; device memory goes down to host memory and up to
/// the other device (the path consumer boards have), bytes are handed over in the process.
/// </summary>
internal sealed class PciHostTransport(LocalHub hub, int rank) : LocalTransport("pcie-host", hub, rank)
{
    public override TransportCapabilities Capabilities => PciHostProvider.Estimate;

    protected override void Copy(int from, Storage source, Storage destination, int count) => CopyThroughHost(source, destination, count);
}

internal sealed class PciHostProvider : ITransportProvider
{
    public static readonly TransportCapabilities Estimate = new(0, TimeSpan.Zero, ZeroCopy: false, GpuDirect: false, Reliable: true, CrossProcess: false, CrossMachine: false);

    public string Name => "pcie-host";

    public string Summary => "GPUs of one machine driven by one process: device memory staged through host memory";

    public TransportCapabilities Capabilities => Estimate;

    public TransportAvailability Availability(ClusterContext context, int peer) => LocalPairs.Check(Name, context, peer, needsPeerAccess: false);

    public ITransport Connect(ClusterContext context, CancellationToken cancellationToken = default) =>
        LocalTransport.JoinGroup(context, Name, (hub, rank) => new PciHostTransport(hub, rank), cancellationToken);
}

/// <summary>Whether two ranks can share a process and its devices, for the in-process transports' availability.</summary>
internal static class LocalPairs
{
    public static TransportAvailability Check(string transport, ClusterContext context, int peer, bool needsPeerAccess)
    {
        var topology = context.Topology;
        int me = context.Rank;
        if (context.Options.Simulation.Mode != SimulationMode.InProcess)
        {
            return new(transport, false, "the ranks are separate processes (IDRAK_CLUSTER_MODE=localhost); this transport joins the ranks of one process");
        }

        if (topology.Place(me).Node != topology.Place(peer).Node)
        {
            return new(transport, false, $"ranks {me} and {peer} are on different machines");
        }

        string a = DeviceOf(context, me), b = DeviceOf(context, peer);
        Device first, second;
        try
        {
            first = Device.Parse(a);
            second = Device.Parse(b);
        }
        catch (Exception e) when (e is ArgumentException or NotSupportedException or InvalidOperationException or FormatException)
        {
            return new(transport, false, $"{a} or {b} is not a device here: {e.Message}");
        }

        if (first.Type == DeviceType.Cpu || second.Type == DeviceType.Cpu)
        {
            return new(transport, false, $"{a} and {b} are not two GPUs");
        }

        if (!needsPeerAccess)
        {
            return new(transport, true, $"{a} and {b} in one process, through host memory");
        }

        if (string.Equals(a, b, StringComparison.OrdinalIgnoreCase))
        {
            return new(transport, false, $"{a} is one device: same-device copies need no peer access");
        }

        try
        {
            return first.Backend.CopiesDirectlyTo(second.Backend)
                ? new(transport, true, $"{a} and {b} reach each other's memory (peer access)")
                : new(transport, false, $"the driver refuses peer access between {a} and {b}: pcie-host carries them");
        }
        catch (Exception e) when (e is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            return new(transport, false, $"{a} or {b} cannot be opened here: {e.Message}");
        }
    }

    public static string DeviceOf(ClusterContext context, int rank)
    {
        var (node, local) = context.Topology.Place(rank);
        return context.Options.Simulation.CarveFrom ?? context.Topology.Nodes[node].Devices[local].Device;
    }
}
