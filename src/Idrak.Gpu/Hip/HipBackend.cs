// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using static Idrak.Gpu.Hip.HipRuntime;

namespace Idrak.Gpu.Hip;

internal sealed class HipStorage(HipBackend backend, ulong pointer, int length, int capacity) : Storage(backend, length)
{
    public ulong Pointer = pointer;                                     // 0 while evicted (Backend.Evict)

    // The floats its block holds: the length, or more when a longer cached block served it (HipBackend.TakeCached).
    public int Capacity = capacity;
}

/// <summary>
/// A GPU driven through the HIP runtime (ROCm on Linux, the HIP SDK on Windows): the third backend family after CUDA
/// and Vulkan, started from the minimum backend. Memory, uploads, downloads, copies and fills run on the device; a
/// first set of kernels compiled at run time by hipRTC (<see cref="HipKernels"/>) runs element-wise arithmetic, a few
/// activations, row RMS norms and int8 products; every other operation takes the host fallback (<see cref="HostCall"/>),
/// so every model runs from the start. All work goes on one stream; uploads and downloads wait for it, so their host
/// memory is never read or written after they return.
/// </summary>
internal sealed unsafe partial class HipBackend : Backend
{
    private static readonly Lazy<(HipDevice[] Devices, string Reason)> Probe = new(ProbeDevices);
    private static readonly Lazy<Lazy<HipBackend>[]> Instances = new(() =>
        [.. Probe.Value.Devices.Select(d => new Lazy<HipBackend>(() => new HipBackend(d)))]);

    // The runtime device bound to the calling thread, plus one (0: none yet).
    [ThreadStatic]
    private static int t_currentDevice;

    private readonly HipDevice _device;
    private readonly IntPtr _stream;
    private readonly MemoryAccountant _memory;

    // Freed blocks by length, reused before new memory is asked for (as on the other GPUs: a training loop stops
    // calling hipMalloc after its first step).
    private readonly Dictionary<int, Stack<ulong>> _pool = [];

    // Lengths with cached blocks, for best-fit reuse when no block of the exact length is cached (as the CUDA pool).
    private readonly SortedSet<int> _poolSizes = [];

    // A cached block up to this fraction longer than a request (of at least BestFitMinimum floats) may serve it.
    private const int BestFitSlack = 4;                                          // 1 / 4: at most 25% longer
    private const int BestFitMinimum = 1024;

    /// <summary>A device the probe found: its runtime ordinal and what it reports.</summary>
    internal sealed record HipDevice(int RuntimeOrdinal, HipDeviceLimits Limits, int RuntimeVersion, int DriverVersion);

    private HipBackend(HipDevice device)
    {
        _device = device;
        MakeCurrent();
        IntPtr stream;
        Check(hipStreamCreate(&stream), nameof(hipStreamCreate));
        _stream = stream;
        var limits = device.Limits;
        string architecture = limits.Architecture.Length > 0 ? $", {limits.Architecture.Split(':')[0]}" : "";
        Name = $"{limits.Name} ({limits.TotalMemory / (1024 * 1024)} MiB, {limits.ComputeUnits} compute units{architecture}, "
               + $"{limits.WarpSize}-lane wavefronts, HIP {FormatVersion(device.RuntimeVersion)})";
        _memory = new MemoryAccountant(() => ComputeResources.GpuMemoryLimit, $"hip:{Ordinal}");
    }

    /// <summary>The number of HIP devices found (0 when there is no runtime or no device).</summary>
    public static int DeviceCount => Probe.Value.Devices.Length;

    /// <summary>Why there are no HIP devices ("" when there are).</summary>
    public static string UnavailableReason => Probe.Value.Reason;

    public static HipBackend Get(int ordinal) => Instances.Value[ordinal].Value;

    public static bool IsInitialized(int ordinal) => Probe.IsValueCreated && Instances.IsValueCreated && Instances.Value[ordinal].IsValueCreated;

    /// <summary>What device <paramref name="ordinal"/> reports, read without starting its backend.</summary>
    internal static HipDeviceLimits Limits(int ordinal) => Probe.Value.Devices[ordinal].Limits;

    public override string Kind => "hip";

    public override string Name { get; }

    public override BackendHardware Hardware => new()
    {
        MemoryBytes = DeviceLimits.TotalMemory,
        ComputeUnits = DeviceLimits.ComputeUnits,
        SubgroupSize = DeviceLimits.WarpSize,
        KernelWidth = HipKernels.BlockSizeFor(DeviceLimits),
        HardwareKind = DeviceLimits.Integrated ? "integrated" : "discrete",
    };

    /// <summary>What this device reports (tests and diagnostics).</summary>
    internal HipDeviceLimits DeviceLimits => _device.Limits;

    // The device's index among HIP devices (hip:N).
    private int Ordinal => Array.IndexOf(Probe.Value.Devices, _device);

    // Every operation without a kernel of its own takes the host fallback, which runs the CPU's code: the layers follow
    // the CPU's limits so the two agree.
    public override BackendCapabilities Capabilities => Device.Cpu.Backend.Capabilities;

    /// <summary>"6.2.41134" for a runtime version 60241134 (major · 10⁷ + minor · 10⁵ + patch).</summary>
    internal static string FormatVersion(int version) => $"{version / 10_000_000}.{version / 100_000 % 100}.{version % 100_000}";

    private static (HipDevice[], string) ProbeDevices()
    {
        if (Environment.GetEnvironmentVariable("IDRAK_DISABLE_HIP") is "1" or "true")
        {
            return ([], "disabled by the IDRAK_DISABLE_HIP environment variable");
        }

        if (!TryLoad(out string reason))
        {
            return ([], reason);
        }

        using var quiet = DeviceException.Handled();                     // a failed probe is a reason the backend reports, not an error
        try
        {
            int result = hipInit(0);
            if (result == ErrorNoDevice)
            {
                return ([], "the HIP runtime reports no devices");
            }

            if (result != Success)
            {
                return ([], $"hipInit failed: {Describe(result)} (no supported GPU, or on Linux the user is not in the render and video groups)");
            }

            int count = 0;
            result = hipGetDeviceCount(&count);
            if (result == ErrorNoDevice || (result == Success && count == 0))
            {
                return ([], "the HIP runtime reports no devices");
            }

            Check(result, nameof(hipGetDeviceCount));
            int runtime = 0, driver = 0;
            hipRuntimeGetVersion(&runtime);
            hipDriverGetVersion(&driver);
            var devices = new List<HipDevice>();
            var skipped = new List<string>();
            for (int i = 0; i < count; i++)
            {
                try
                {
                    devices.Add(new HipDevice(i, HipDeviceLimits.Read(i), runtime, driver));
                }
                catch (HipException e)
                {
                    skipped.Add($"device {i}: {e.Message}");
                }
            }

            return devices.Count > 0 ? ([.. devices], "") : ([], $"no usable HIP device ({string.Join("; ", skipped)})");
        }
        catch (Exception ex) when (ex is HipException or DllNotFoundException or EntryPointNotFoundException)
        {
            return ([], ex.Message);
        }
    }

    /// <summary>Binds this device to the calling thread (the runtime keeps a current device per thread).</summary>
    private void MakeCurrent()
    {
        if (t_currentDevice != _device.RuntimeOrdinal + 1)
        {
            Check(hipSetDevice(_device.RuntimeOrdinal), nameof(hipSetDevice));
            t_currentDevice = _device.RuntimeOrdinal + 1;
        }
    }

    private static long BlockBytes(int length) => (long)Math.Max(length, 1) * sizeof(float);

    /// <summary>Memory kept free on this GPU (ComputeResources.GpuMemoryReserve, else a sixteenth of it, at least 256 MiB, as on CUDA).</summary>
    private long MemoryReserve => ComputeResources.GpuMemoryReserve ?? Math.Max(256L << 20, _device.Limits.TotalMemory / 16);

    public override Storage Allocate(int length, bool zeroed)
    {
        MakeCurrent();
        long bytes = BlockBytes(length);
        bool releaseCache = _memory.MustReleaseCacheFor(bytes);              // throws when over the in-use limit
        ulong pointer = 0;
        int capacity = length;
        lock (_pool)
        {
            if (TakeCached(length, out capacity) is var cached && cached != 0)
            {
                pointer = cached;
                _memory.Reused(BlockBytes(capacity));
            }
        }

        if (pointer == 0)
        {
            if (releaseCache)
            {
                ReleaseCachedMemory();
            }

            pointer = AllocateDevice(bytes);
            _memory.Allocated(bytes);
        }

        if (zeroed && length > 0)
        {
            Check(hipMemsetD32Async(pointer, 0, (nuint)length, _stream), nameof(hipMemsetD32Async));
        }

        return new HipStorage(this, pointer, length, capacity);
    }

    // A cached block for `length` floats: the exact length, else the smallest cached length at most 25% longer (varying
    // shapes then reuse blocks instead of caching one of every length); 0 when none. Under the pool lock.
    private ulong TakeCached(int length, out int capacity)
    {
        capacity = length;
        if (!_pool.TryGetValue(length, out var bucket) || bucket.Count == 0)
        {
            if (length < BestFitMinimum || length == int.MaxValue || _poolSizes.Count == 0)
            {
                return 0;
            }

            var fitting = _poolSizes.GetViewBetween(length + 1, (int)Math.Min(int.MaxValue, (long)length + length / BestFitSlack));
            if (fitting.Count == 0)
            {
                return 0;
            }

            capacity = fitting.Min;
            bucket = _pool[capacity];
        }

        ulong pointer = bucket.Pop();
        if (bucket.Count == 0)
        {
            _poolSizes.Remove(capacity);
        }

        return pointer;
    }

    // New device memory; a block fits when the GPU keeps MemoryReserve free afterwards. Full: give back the cache and
    // what unreachable tensors hold, and try once more.
    private ulong AllocateDevice(long bytes)
    {
        bool Fits()
        {
            nuint free, total;
            return hipMemGetInfo(&free, &total) != Success || (long)free - bytes >= MemoryReserve;
        }

        int result = ErrorOutOfMemory;
        ulong pointer = 0;
        for (int attempt = 0; attempt < 2 && result == ErrorOutOfMemory; attempt++)
        {
            if (attempt == 1)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                ReleaseCachedMemory();
            }

            result = Fits() ? hipMalloc(&pointer, (nuint)bytes) : ErrorOutOfMemory;
        }

        if (result == ErrorOutOfMemory)
        {
            nuint free = 0, total = 0;
            hipMemGetInfo(&free, &total);
            throw new ResourceLimitExceededException(
                $"{Name} is out of memory: {bytes:N0} more bytes needed, {free:N0} of {total:N0} free with {MemoryReserve:N0} kept in reserve " +
                $"({_memory.Usage}). Use smaller batches or shorter sequences.");
        }

        Check(result, nameof(hipMalloc));
        return pointer;
    }

    // Called when the last reference is released, possibly from the finalizer thread, so it only touches the pool.
    public override void Return(Storage storage)
    {
        var s = (HipStorage)storage;
        long bytes = BlockBytes(s.Capacity);
        lock (_pool)
        {
            if (s.Pointer != 0)
            {
                if (!_pool.TryGetValue(s.Capacity, out var bucket))
                {
                    _pool[s.Capacity] = bucket = new Stack<ulong>();
                }

                bucket.Push(s.Pointer);
                _poolSizes.Add(s.Capacity);
            }
        }

        _memory.Returned(bytes);
    }

    protected override void Detach(Storage storage) => ((HipStorage)storage).Pointer = 0;

    protected override void Attach(Storage storage, Storage fresh) =>
        (((HipStorage)storage).Pointer, ((HipStorage)storage).Capacity) = (((HipStorage)fresh).Pointer, ((HipStorage)fresh).Capacity);

    public override MemoryUsage GetMemoryUsage() => _memory.Usage;

    public override void ResetPeakMemoryUsage() => _memory.ResetPeak();

    /// <summary>Frees every cached block, after the work queued earlier (which may still use one) has finished.</summary>
    public override void ReleaseCachedMemory()
    {
        MakeCurrent();
        Check(hipStreamSynchronize(_stream), nameof(hipStreamSynchronize));
        lock (_pool)
        {
            foreach (var (length, bucket) in _pool)
            {
                while (bucket.Count > 0)
                {
                    Check(hipFree(bucket.Pop()), nameof(hipFree));
                    _memory.Freed(BlockBytes(length));
                }
            }

            _poolSizes.Clear();
        }
    }

    public override void Upload(ReadOnlySpan<float> source, Storage destination)
    {
        if (source.IsEmpty)
        {
            return;
        }

        MakeCurrent();
        fixed (float* p = source)
        {
            Check(hipMemcpyAsync((void*)P(destination), p, (nuint)source.Length * sizeof(float), MemcpyHostToDevice, _stream), nameof(hipMemcpyAsync));
            Check(hipStreamSynchronize(_stream), nameof(hipStreamSynchronize));          // the span is not ours after return
        }
    }

    public override void Download(Storage source, Span<float> destination) => DownloadRange(source, 0, destination);

    public override void DownloadRange(Storage source, int offset, Span<float> destination)
    {
        if (destination.IsEmpty)
        {
            return;
        }

        MakeCurrent();
        fixed (float* p = destination)
        {
            Check(hipMemcpyAsync(p, (void*)(P(source) + (ulong)offset * sizeof(float)), (nuint)destination.Length * sizeof(float), MemcpyDeviceToHost, _stream),
                nameof(hipMemcpyAsync));
            Check(hipStreamSynchronize(_stream), nameof(hipStreamSynchronize));
        }
    }

    public override void FillKernel(Storage y, int n, float value)
    {
        if (n > 0)
        {
            MakeCurrent();
            Check(hipMemsetD32Async(P(y), BitConverter.SingleToInt32Bits(value), (nuint)n, _stream), nameof(hipMemsetD32Async));
        }
    }

    public override void Copy(Storage x, Storage y, int n)
    {
        if (n > 0 && !ReferenceEquals(x, y))
        {
            MakeCurrent();
            Check(hipMemcpyAsync((void*)P(y), (void*)P(x), (nuint)n * sizeof(float), MemcpyDeviceToDevice, _stream), nameof(hipMemcpyAsync));
        }
    }

    public override void Copy2D(Storage src, int srcOffset, int srcStride, Storage dst, int dstOffset, int dstStride, int rows, int cols, bool accumulate)
    {
        // A strided copy is one 2-D copy on the device; adding, overlapping rows (one storage) and pitches narrower
        // than a row take the host fallback.
        if (accumulate || ReferenceEquals(src, dst) || srcStride < cols || dstStride < cols)
        {
            base.Copy2D(src, srcOffset, srcStride, dst, dstOffset, dstStride, rows, cols, accumulate);
            return;
        }

        if (rows > 0 && cols > 0)
        {
            MakeCurrent();
            Check(hipMemcpy2DAsync((void*)(P(dst) + (ulong)dstOffset * sizeof(float)), (nuint)dstStride * sizeof(float),
                (void*)(P(src) + (ulong)srcOffset * sizeof(float)), (nuint)srcStride * sizeof(float),
                (nuint)cols * sizeof(float), (nuint)rows, MemcpyDeviceToDevice, _stream), nameof(hipMemcpy2DAsync));
        }
    }

    public override void Synchronize()
    {
        MakeCurrent();
        Check(hipStreamSynchronize(_stream), nameof(hipStreamSynchronize));
    }

    private static ulong P(Storage s)
    {
        ulong pointer = ((HipStorage)s).Pointer;
        return pointer != 0 ? pointer : throw new InvalidOperationException("The storage was evicted (its memory was given back) and not restored.");
    }
}

/// <summary>
/// HIP devices: listed (a plain test run covers them), never the default unless IDRAK_HIP_DEFAULT=1 opts in, and then
/// after CUDA (registered first, ranked higher) and before Vulkan, discrete GPUs before integrated ones as reported.
/// </summary>
internal sealed class HipProvider : DeviceProvider
{
    public override string Kind => "hip";

    public override string Display => "HIP";

    public override DeviceType Type => DeviceType.Hip;

    public override int Count => HipBackend.DeviceCount;

    public override string? UnavailableReason => HipBackend.UnavailableReason;

    public override Backend Create(int ordinal) => HipBackend.Get(ordinal);

    public override bool IsStarted(int ordinal) => HipBackend.IsInitialized(ordinal);

    public override Guid? DeviceUuid(int ordinal) => (uint)ordinal < (uint)Count ? HipBackend.Limits(ordinal).Uuid : null;

    // Untested on real hardware so far, and most operations still take the host fallback: by name or by opting in.
    public override int? DefaultRank(int ordinal) => Environment.GetEnvironmentVariable("IDRAK_HIP_DEFAULT") != "1" ? null
        : HipBackend.Limits(ordinal).Integrated ? 45 : 60;

    public override string? Note(int ordinal) =>
        DefaultRank(ordinal) is null ? "not the default device (IDRAK_HIP_DEFAULT=1 to prefer it)" : null;
}
