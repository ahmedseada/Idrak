// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Cluster.Abstractions;

/// <summary>
/// A link whose rate is this machine's own, measured on the device a rank is on (two ranks on one device, two processes
/// of one machine, a device's copies through the host) instead of declared: it has no rate of its own
/// (<see cref="ILinkProfile.BandwidthBytesPerSecond"/> 0), the simulator and the inspection measure it for each rank's
/// device, and a throttled transport leaves it unthrottled.
/// </summary>
public interface IDeviceMeasuredLink : ILinkProfile
{
    /// <summary>The profile as measured on <paramref name="device"/> (measured once per device, on first use).</summary>
    ILinkProfile On(Device device);
}

/// <summary>
/// A failure model whose faults are asked for while the run goes (the dashboard), not drawn from the seed: a run whose
/// own model it is logs and publishes every fault it draws as live.
/// </summary>
public interface ILiveFailureModel : IFailureModel
{
    /// <summary>Faults queued and not yet drawn.</summary>
    int Pending { get; }

    /// <summary>Queues <paramref name="fault"/> for the next step drawn (its own step is replaced by that step's).</summary>
    void Queue(Fault fault);
}
