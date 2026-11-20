// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using static Idrak.Adad.Devices.Cuda.CudaDriver;

namespace Idrak.Adad.Devices.Cuda;

/// <summary>A block of device memory: its address and the bytes it was asked for (a value: no object a rental).</summary>
public readonly record struct DeviceBuffer(ulong Pointer, long Bytes)
{
    /// <summary>No buffer.</summary>
    public static DeviceBuffer None => default;

    /// <summary>Whether this is a buffer.</summary>
    public bool Exists => Pointer != 0;
}

/// <summary>A loaded kernel: its function, its dynamic shared bytes, and what the driver's compiler reported for it.</summary>
public sealed record CudaKernel(string Name, IntPtr Function, int DynamicShared, int Registers, int SpilledBytes);

/// <summary>
/// One CUDA device for Adad (copied from Idrak's backend, the parts the kernels need): the device's primary context, one
/// work stream, device memory from a size-keyed pool (a block of a size is reused by the next request of that size; the
/// driver is asked only when none is free), PTX modules loaded by the driver's compiler with its log kept, launches with
/// their arguments on the stack, and event timing on the work stream. Its limits are the device's own, read at open: no
/// card is known in code.
/// </summary>
public sealed unsafe class CudaDevice : IDisposable
{
    private readonly int _ordinal;
    private readonly IntPtr _context;
    private readonly IntPtr _stream;
    private readonly Dictionary<long, Stack<ulong>> _pool = [];
    private readonly IntPtr _start, _end;
    private bool _disposed;

    private CudaDevice(int ordinal, IntPtr context, IntPtr stream, string name, int multiprocessors, int major, int minor, int sharedPerBlock, int sharedOptin, int l2, long memory)
    {
        _ordinal = ordinal;
        _context = context;
        _stream = stream;
        Name = name;
        Multiprocessors = multiprocessors;
        ComputeMajor = major;
        ComputeMinor = minor;
        SharedPerBlock = sharedPerBlock;
        SharedPerBlockOptin = sharedOptin;
        L2Bytes = l2;
        TotalMemory = memory;
        Check(cuEventCreate(out _start, 0), nameof(cuEventCreate));
        Check(cuEventCreate(out _end, 0), nameof(cuEventCreate));
    }

    /// <summary>The device's name, as the driver reports it.</summary>
    public string Name { get; }

    /// <summary>Its multiprocessors.</summary>
    public int Multiprocessors { get; }

    /// <summary>Its compute capability.</summary>
    public int ComputeMajor { get; }

    /// <summary>Its compute capability's minor number.</summary>
    public int ComputeMinor { get; }

    /// <summary>Static shared memory a block may declare.</summary>
    public int SharedPerBlock { get; }

    /// <summary>Shared memory a block may take in all (static and dynamic, opted in).</summary>
    public int SharedPerBlockOptin { get; }

    /// <summary>The L2 cache's bytes (0 when not reported).</summary>
    public int L2Bytes { get; }

    /// <summary>The device's memory.</summary>
    public long TotalMemory { get; }

    /// <summary>Bytes the pool holds from the driver (in use and free).</summary>
    public long PoolBytes { get; private set; }

    /// <summary>The most bytes the pool has held at once.</summary>
    public long PeakPoolBytes { get; private set; }

    /// <summary>Driver allocations made (a pool miss each).</summary>
    public long DriverAllocations { get; private set; }

    /// <summary>Launches made.</summary>
    public long Launches { get; private set; }

    /// <summary>Opens device <paramref name="ordinal"/>, or returns null with the reason.</summary>
    public static CudaDevice? TryOpen(int ordinal, out string reason)
    {
        if (!TryLoad(out reason))
        {
            return null;
        }

        int result = cuInit(0);
        if (result != 0)
        {
            reason = $"cuInit failed: {Describe(result)}";
            return null;
        }

        Check(cuDeviceGetCount(out int count), nameof(cuDeviceGetCount));
        if (ordinal >= count)
        {
            reason = $"no CUDA device {ordinal} ({count} found)";
            return null;
        }

        Check(cuDeviceGet(out int device, ordinal), nameof(cuDeviceGet));
        byte* name = stackalloc byte[256];
        Check(cuDeviceGetName(name, 256, device), nameof(cuDeviceGetName));
        int Get(int attribute)
        {
            Check(cuDeviceGetAttribute(out int value, attribute, device), nameof(cuDeviceGetAttribute));
            return value;
        }

        Check(cuDeviceTotalMem(out nuint memory, device), nameof(cuDeviceTotalMem));
        Check(cuDevicePrimaryCtxRetain(out IntPtr context, device), nameof(cuDevicePrimaryCtxRetain));
        Check(cuCtxSetCurrent(context), nameof(cuCtxSetCurrent));
        Check(cuStreamCreate(out IntPtr stream, StreamNonBlocking), nameof(cuStreamCreate));
        int shared = Get(AttributeMaxSharedPerBlock);
        reason = "";
        return new CudaDevice(device, context, stream, Marshal.PtrToStringAnsi((IntPtr)name) ?? "CUDA device", Get(AttributeMultiprocessorCount),
            Get(AttributeComputeCapabilityMajor), Get(AttributeComputeCapabilityMinor), shared, Math.Max(shared, Get(AttributeMaxSharedPerBlockOptin)),
            Get(AttributeL2Size), (long)memory);
    }

    /// <summary>The driver's version (for the bench's header).</summary>
    public static int DriverVersion => cuDriverGetVersion(out int v) == 0 ? v : 0;

    /// <summary>Makes the device's context current on this thread.</summary>
    public void MakeCurrent() => Check(cuCtxSetCurrent(_context), nameof(cuCtxSetCurrent));

    // ------------------------------------------------------------------ memory

    /// <summary>A block of at least <paramref name="bytes"/> bytes: a free block of that size from the pool, else the driver's.</summary>
    public DeviceBuffer Allocate(long bytes)
    {
        long size = Math.Max(256, (bytes + 255) / 256 * 256);
        if (_pool.TryGetValue(size, out var free) && free.Count > 0)
        {
            return new DeviceBuffer(free.Pop(), bytes);
        }

        int result = cuMemAlloc(out ulong pointer, (nuint)size);
        if (result == ErrorOutOfMemory)
        {
            TrimPool();
            result = cuMemAlloc(out pointer, (nuint)size);
        }

        Check(result, $"cuMemAlloc({size.ToString(CultureInfo.InvariantCulture)})");
        DriverAllocations++;
        PoolBytes += size;
        PeakPoolBytes = Math.Max(PeakPoolBytes, PoolBytes);
        return new DeviceBuffer(pointer, bytes);
    }

    /// <summary>Gives <paramref name="buffer"/> back to the pool (kept for the next request of its size).</summary>
    public void Release(DeviceBuffer buffer)
    {
        if (!buffer.Exists)
        {
            return;
        }

        long size = Math.Max(256, (buffer.Bytes + 255) / 256 * 256);
        if (!_pool.TryGetValue(size, out var free))
        {
            _pool[size] = free = new Stack<ulong>();
        }

        free.Push(buffer.Pointer);
    }

    /// <summary>Frees every block the pool holds unused.</summary>
    public void TrimPool()
    {
        foreach (var (size, free) in _pool)
        {
            while (free.Count > 0)
            {
                Check(cuMemFree(free.Pop()), nameof(cuMemFree));
                PoolBytes -= size;
            }
        }
    }

    /// <summary>Copies <paramref name="source"/> into <paramref name="buffer"/> (waits for the work queued before).</summary>
    public void Upload<T>(DeviceBuffer buffer, ReadOnlySpan<T> source) where T : unmanaged
    {
        Synchronize();
        fixed (T* p = source)
        {
            Check(cuMemcpyHtoD(buffer.Pointer, p, (nuint)(source.Length * sizeof(T))), nameof(cuMemcpyHtoD));
        }
    }

    /// <summary>Copies <paramref name="buffer"/>'s first values into <paramref name="destination"/> once the queued work is done.</summary>
    public void Download<T>(DeviceBuffer buffer, Span<T> destination) where T : unmanaged
    {
        Synchronize();
        fixed (T* p = destination)
        {
            Check(cuMemcpyDtoH(p, buffer.Pointer, (nuint)(destination.Length * sizeof(T))), nameof(cuMemcpyDtoH));
        }
    }

    /// <summary>Sets <paramref name="words"/> 32-bit words of <paramref name="buffer"/> to <paramref name="value"/> on the work stream.</summary>
    public void Fill(DeviceBuffer buffer, uint value, long words) =>
        Check(cuMemsetD32Async(buffer.Pointer, value, (nuint)words, _stream), nameof(cuMemsetD32Async));

    /// <summary>Waits for the work stream.</summary>
    public void Synchronize() => Check(cuStreamSynchronize(_stream), nameof(cuStreamSynchronize));

    // ------------------------------------------------------------------ modules

    /// <summary>
    /// Loads <paramref name="source"/> (PTX) and finds <paramref name="entries"/>, each opted in to its dynamic shared
    /// bytes; null with the driver's reason when the driver refuses the module (a kernel the device cannot run is no
    /// candidate). <paramref name="log"/> receives the compiler's info log.
    /// </summary>
    public CudaKernel[]? LoadModule(string source, string[] entries, Func<string, int> dynamicShared, out string reason, out string log)
    {
        MakeCurrent();
        const int LogBytes = 64 * 1024;
        var infoBytes = new byte[LogBytes];
        var errorBytes = new byte[LogBytes];
        int* options = stackalloc int[5];
        void** values = stackalloc void*[5];
        byte[] image = Encoding.ASCII.GetBytes(source + "\0");
        IntPtr module;
        int result;
        fixed (byte* info = infoBytes, errors = errorBytes, p = image)
        {
            options[0] = JitInfoLogBuffer;
            values[0] = info;
            options[1] = JitInfoLogBufferSizeBytes;
            values[1] = (void*)LogBytes;
            options[2] = JitErrorLogBuffer;
            values[2] = errors;
            options[3] = JitErrorLogBufferSizeBytes;
            values[3] = (void*)LogBytes;
            options[4] = JitLogVerbose;
            values[4] = (void*)1;
            result = cuModuleLoadDataEx(out module, p, 5, options, values);
            log = Marshal.PtrToStringAnsi((IntPtr)info) ?? "";
            if (result != 0)
            {
                reason = $"the driver refused the module: {Describe(result)} {Marshal.PtrToStringAnsi((IntPtr)errors)}".Trim();
                return null;
            }
        }

        var kernels = new CudaKernel[entries.Length];
        for (int i = 0; i < entries.Length; i++)
        {
            byte[] name = Encoding.ASCII.GetBytes(entries[i] + "\0");
            fixed (byte* p = name)
            {
                Check(cuModuleGetFunction(out IntPtr function, module, p), $"cuModuleGetFunction({entries[i]})");
                int dynamic = dynamicShared(entries[i]);
                if (dynamic > 48 * 1024)
                {
                    result = cuFuncSetAttribute(function, FunctionAttributeMaxDynamicSharedSizeBytes, dynamic);
                    if (result != 0)
                    {
                        reason = $"{entries[i]}: {dynamic} bytes of shared memory refused: {Describe(result)}";
                        return null;
                    }
                }

                Check(cuFuncGetAttribute(out int registers, FunctionAttributeNumRegs, function), nameof(cuFuncGetAttribute));
                Check(cuFuncGetAttribute(out int local, FunctionAttributeLocalSizeBytes, function), nameof(cuFuncGetAttribute));
                kernels[i] = new CudaKernel(entries[i], function, dynamic > 48 * 1024 ? dynamic : 0, registers, local);
            }
        }

        reason = "";
        return kernels;
    }

    // ------------------------------------------------------------------ launching

    /// <summary>
    /// Launches <paramref name="kernel"/> on the work stream with its dynamic shared bytes; each argument a 64-bit slot
    /// (a 32-bit parameter reads its low half), on the stack: no allocation a launch.
    /// </summary>
    public void Launch(CudaKernel kernel, uint gridX, uint gridY, uint gridZ, uint block, ReadOnlySpan<ulong> args)
    {
        int n = args.Length;
        ulong* slots = stackalloc ulong[n];
        void** pointers = stackalloc void*[n];
        for (int i = 0; i < n; i++)
        {
            slots[i] = args[i];
            pointers[i] = &slots[i];
        }

        int result = cuLaunchKernel(kernel.Function, gridX, gridY, gridZ, block, 1, 1, (uint)kernel.DynamicShared, _stream, pointers, null);
        if (result != 0)
        {
            Check(result, $"cuLaunchKernel({kernel.Name})");
        }

        Launches++;
    }

    // ------------------------------------------------------------------ timing

    /// <summary>Marks the start of a timed span on the work stream.</summary>
    public void TimeStart() => Check(cuEventRecord(_start, _stream), nameof(cuEventRecord));

    /// <summary>Ends the span started by <see cref="TimeStart"/>, waits for it, and returns its milliseconds.</summary>
    public float TimeEnd()
    {
        Check(cuEventRecord(_end, _stream), nameof(cuEventRecord));
        Check(cuEventSynchronize(_end), nameof(cuEventSynchronize));
        Check(cuEventElapsedTime(out float ms, _start, _end), nameof(cuEventElapsedTime));
        return ms;
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        MakeCurrent();
        Synchronize();
        TrimPool();
        _ = cuEventDestroy(_start);
        _ = cuEventDestroy(_end);
        _ = cuStreamDestroy(_stream);
        _ = cuDevicePrimaryCtxRelease(_ordinal);
    }
}
