// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Networking.Abstractions;

namespace Idrak.Networking.Transports;

/// <summary>
/// "nvlink" (alias "p2p", plan 23): ranks of one process on GPUs that reach each other's memory; device memory is copied
/// device to device by the backend (<see cref="Backend.CopyTo"/>: a peer copy where the driver grants peer access). Where
/// it refuses, the copy goes through host memory as "pcie-host" would, and the telemetry says so once per pair (a
/// <see cref="TransportChosen"/> naming pcie-host and the reason).
/// </summary>
internal sealed class NvlinkTransport(string name, LocalHub hub, int rank) : LocalTransport(name, hub, rank)
{
    public override TransportCapabilities Capabilities => NvlinkProvider.Estimate;

    protected override void Copy(int from, Storage source, Storage destination, int count)
    {
        var backend = source.Backend;
        if (backend.CopiesDirectlyTo(destination.Backend))
        {
            backend.CopyTo(source, destination, count);
            return;
        }

        var pair = (Math.Min(from, Rank), Math.Max(from, Rank));
        if (Hub.Reported.TryAdd(pair, true) && Telemetry.IsEnabled(TelemetryLevel.Network))
        {
            Telemetry.TransportChosen(new TransportChosen(pair.Item1, pair.Item2, "pcie-host",
                $"{Name} falls back: {backend.Name} and {destination.Backend.Name} do not reach each other's memory (no peer access), so copies go through host memory"));
        }

        CopyThroughHost(source, destination, count);
    }
}

internal sealed class NvlinkProvider(string name = "nvlink") : ITransportProvider
{
    public static readonly TransportCapabilities Estimate = new(0, TimeSpan.Zero, ZeroCopy: true, GpuDirect: true, Reliable: true, CrossProcess: false, CrossMachine: false);

    public string Name { get; } = name;

    public string Summary => Name == "nvlink"
        ? "GPUs of one process with peer access: device-to-device copies (refused peers go through host memory, said once per pair)"
        : "another name of nvlink: peer copies between the GPUs of one process";

    public TransportCapabilities Capabilities => Estimate;

    public TransportAvailability Availability(ClusterContext context, int peer) => LocalPairs.Check(Name, context, peer, needsPeerAccess: true);

    public ITransport Connect(ClusterContext context, CancellationToken cancellationToken = default) =>
        LocalTransport.JoinGroup(context, Name, (hub, rank) => new NvlinkTransport(Name, hub, rank), cancellationToken);
}
