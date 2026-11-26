// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using Idrak.Abstraction.Devices.Cpu;

namespace Idrak.Abstraction.Devices;

/// <summary>
/// Runs one operation of a device that has no kernel of its own for it on the CPU's own implementation (the default of
/// each <see cref="Backend"/> operation the CPU runs itself): no registry and no slot, the CPU backend's operation called
/// directly. Each storage the operation is given is mirrored into a CPU storage once (an argument passed twice stays one
/// mirror), holding the same words of the same type (<see cref="Storage.Type"/>, so the CPU reads it through that type
/// as it reads its own); the CPU runs the operation on the mirrors; the mirrors whose words changed go back. A storage the
/// CPU holds already is taken as it is; one in host memory (<see cref="IHostStorage"/>) is mirrored from its values in
/// place and written back there, with no copy through its device. Correct for any operation whatever its inputs and
/// outputs, and slow (every operand crosses to the host and back): a new device runs every model with memory and copies
/// alone, then replaces these fallbacks with kernels of its own.
/// </summary>
internal sealed class HostCall : IDisposable
{
    private readonly Backend _device;
    private readonly List<(Storage Device, Storage Host, float[] Before)> _mirrors = [];

    /// <summary>Starts a fallback of the operation named <paramref name="operation"/> (counted under that name).</summary>
    public HostCall(Backend device, string operation)
    {
        _device = device;
        device.HostCalled(operation);                                      // Kernels.HostCalls, KernelTrace, telemetry
    }

    /// <summary>The CPU's storage for <paramref name="storage"/>: itself where the CPU holds it, else its mirror.</summary>
    public Storage this[Storage storage]
    {
        get
        {
            if (storage.Backend is CpuBackend)
            {
                return storage;                                            // the CPU's own: its kernels read it as it is
            }

            foreach (var (device, host, _) in _mirrors)
            {
                if (ReferenceEquals(device, storage))
                {
                    return host;
                }
            }

            var mirror = CpuBackend.Instance.Allocate(storage.Length, zeroed: false);
            if (storage.Typed)
            {
                mirror.SetType(storage.Type, storage.ElementCount);        // the same words, read as the same type
            }

            try
            {
                var values = CpuBackend.D(mirror).AsSpan(0, storage.Length);
                if (storage is IHostStorage host)
                {
                    host.HostMemory.CopyTo(values);                        // in host memory already: no copy from the device
                }
                else
                {
                    _device.Download(storage, values);
                }

                var before = ArrayPool<float>.Shared.Rent(storage.Length);
                values.CopyTo(before);
                _mirrors.Add((storage, mirror, before));
            }
            catch
            {
                mirror.Release();                                          // a failed download keeps no mirror
                throw;
            }

            return mirror;
        }
    }

    /// <summary>The CPU's storage for <paramref name="storage"/>, or null when there is none.</summary>
    public Storage? Maybe(Storage? storage) => storage is null ? null : this[storage];

    /// <summary>Copies the mirrors the operation changed back to their device and frees them.</summary>
    public void Dispose()
    {
        // Every mirror is freed and its snapshot returned even when a copy back fails; the first failure is rethrown.
        Exception? failure = null;
        foreach (var (device, host, before) in _mirrors)
        {
            try
            {
                var values = CpuBackend.D(host).AsSpan(0, device.Length);
                // Compared as bits: a word holding an integer or a typed value's bytes (or a float's -0 against 0) is
                // changed when its bits are.
                if (!MemoryMarshal.Cast<float, int>(values).SequenceEqual(MemoryMarshal.Cast<float, int>(before.AsSpan(0, device.Length))))
                {
                    if (device is IHostStorage inPlace)
                    {
                        values.CopyTo(inPlace.HostMemory);
                    }
                    else
                    {
                        _device.Upload(values, device);
                    }
                }
            }
            catch (Exception e)
            {
                failure ??= e;
            }
            finally
            {
                ArrayPool<float>.Shared.Return(before);
                host.Release();
            }
        }

        _mirrors.Clear();
        if (failure is not null)
        {
            ExceptionDispatchInfo.Throw(failure);
        }
    }
}
