// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using static Idrak.Gpu.Cuda.CudaDriver;

namespace Idrak.Gpu.Cuda;

// Copies between the devices of two ranks (plan 23, phase C5: the "nvlink" transport): device to device inside one card
// (two backends carved from it share its memory), a peer copy between two cards the driver lets reach each other (peer
// access enabled once per pair, remembered), and through host memory otherwise (Backend.CopyTo's own path).
internal sealed unsafe partial class CudaBackend : IPeerCopyBackend
{
    // (from device, to device) -> whether the first reaches the second's memory, asked and enabled once per pair.
    private static readonly ConcurrentDictionary<(int From, int To), bool> PeerAccess = new();

    bool IPeerCopyBackend.CopiesDirectlyTo(Backend destination) =>
        destination is CudaBackend other && (other._deviceHandle == _deviceHandle || Peer(other));

    // Backend.CopyTo has checked the arguments and found the destination on another CUDA backend this one reaches.
    void IPeerCopyBackend.CopyTo(Storage source, Storage destination, int count)
    {
        var other = (CudaBackend)destination.Backend;
        other.Synchronize();                                       // the destination's queued work finishes before the copy writes
        using var operation = EnterOperation();                    // addresses read until the work is queued (the move guard)
        using var theirs = other.EnterOperation();
        MakeCurrent();
        using var use = UseStream();
        nuint bytes = (nuint)count * sizeof(float);
        if (other._deviceHandle == _deviceHandle)
        {
            Check(cuMemcpyDtoDAsync(P(destination), P(source), bytes, _stream), nameof(cuMemcpyDtoDAsync));
        }
        else
        {
            Check(cuMemcpyPeerAsync(P(destination), other._context, P(source), _context, bytes, _stream), nameof(cuMemcpyPeerAsync));
        }

        Check(cuStreamSynchronize(_stream), nameof(cuStreamSynchronize));
    }

    private bool Peer(CudaBackend other) => PeerAccess.GetOrAdd((_deviceHandle, other._deviceHandle), _ =>
    {
        if (cuDeviceCanAccessPeer(out int can, _deviceHandle, other._deviceHandle) != 0 || can == 0)
        {
            return false;
        }

        MakeCurrent();
        int result = cuCtxEnablePeerAccess(other._context, 0);
        return result == 0 || result == ErrorPeerAccessAlreadyEnabled;
    });
}
