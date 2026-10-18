// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Networking.Abstractions;

/// <summary>
/// A transport whose ranks share one process and can hand device memory over (plan 23, phase C5: "pcie-host" and "nvlink"):
/// the receiver copies straight from the sender's storage into its own (<see cref="Backend.CopyTo"/>), a peer copy where
/// the devices reach each other, through host memory where they do not. Bytes still travel through
/// <see cref="ITransport.Send"/> and <see cref="ITransport.Receive"/>.
/// </summary>
public interface IDeviceTransport : ITransport
{
    /// <summary>
    /// Hands the first <paramref name="count"/> values of <paramref name="source"/> to rank <paramref name="to"/> under
    /// <paramref name="tag"/> and returns once that rank has copied them (the storage may change after the call).
    /// </summary>
    /// <exception cref="TimeoutException">The rank did not take them within <see cref="ClusterOptions.Timeout"/>; the message names it.</exception>
    void SendDevice(int to, Storage source, int count, int tag);

    /// <summary>
    /// Copies the values rank <paramref name="from"/> handed over under <paramref name="tag"/> into the first
    /// <paramref name="count"/> values of <paramref name="destination"/>.
    /// </summary>
    /// <exception cref="TimeoutException">Nothing was handed over within <see cref="ClusterOptions.Timeout"/>; the message names the rank.</exception>
    /// <exception cref="InvalidDataException">The sender handed over a different count.</exception>
    void ReceiveDevice(int from, Storage destination, int count, int tag);
}
