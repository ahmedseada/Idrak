// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.ExceptionServices;
using Idrak.Abstraction.Devices.Cpu;

namespace Idrak.Abstraction.Devices;

/// <summary>
/// Runs one operation of a device that has no kernel of its own for it on the CPU (the default of every
/// <see cref="Backend"/> operation beyond memory and copies). Each storage the operation is given is copied into a CPU
/// mirror, once (an argument passed twice stays one buffer); the CPU backend runs the operation on the mirrors; the
/// mirrors whose values changed are copied back. Correct for any operation, whatever its inputs and outputs, and slow
/// (every operand crosses to the host and back): a new device runs every model with memory and copies alone, then
/// replaces these fallbacks with kernels of its own, the most used first.
/// </summary>
internal sealed class HostCall : IDisposable
{
    private readonly Backend _device;
    private readonly List<(Storage Device, Storage Host, float[] Before)> _mirrors = [];

    /// <summary>
    /// Starts a fallback of the operation <paramref name="operation"/> (by default the calling method's name, the
    /// device's own kernel <c>NameKernel</c>, counted as the operation's name).
    /// </summary>
    public HostCall(Backend device, [CallerMemberName] string operation = "")
    {
        _device = device;
        device.HostCalled(operation.EndsWith("Kernel", StringComparison.Ordinal) ? operation[..^"Kernel".Length] : operation);   // Kernels.HostCalls, KernelTrace
    }

    /// <summary>The CPU mirror of <paramref name="storage"/>, holding its values.</summary>
    public Storage this[Storage storage]
    {
        get
        {
            if (storage.Backend is CpuBackend)
            {
                return storage;                                            // already on the host
            }

            foreach (var (device, host, _) in _mirrors)
            {
                if (ReferenceEquals(device, storage))
                {
                    return host;
                }
            }

            var mirror = CpuBackend.Instance.Allocate(storage.Length, zeroed: false);
            try
            {
                var values = CpuBackend.D(mirror).AsSpan(0, storage.Length);
                _device.Download(storage, values);
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

    /// <summary>The CPU mirror of <paramref name="storage"/>, or null when there is none.</summary>
    public Storage? Maybe(Storage? storage) => storage is null ? null : this[storage];

    /// <summary>Copies the mirrors the operation changed back to the device and frees them.</summary>
    public void Dispose()
    {
        // Every mirror is freed and its snapshot returned even when a copy back fails; the first failure is rethrown.
        Exception? failure = null;
        foreach (var (device, host, before) in _mirrors)
        {
            try
            {
                var values = CpuBackend.D(host).AsSpan(0, device.Length);
                if (!values.SequenceEqual(before.AsSpan(0, device.Length)))
                {
                    _device.Upload(values, device);
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
