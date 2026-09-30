using System.Runtime.InteropServices;
using System.Text;
using static Idrak.Backends.Cuda.CudaDriver;

namespace Idrak.Backends.Cuda;

internal sealed class CudaStorage(CudaBackend backend, ulong pointer, int length, int capacity) : Storage(backend, length)
{
    public ulong Pointer = pointer;                                     // 0 while evicted (Backend.Evict)

    /// <summary>Floats in the device block (at least <see cref="Storage.Length"/>: a cached block a little larger may be reused).</summary>
    public int Capacity = capacity;

    public CudaStorage(CudaBackend backend, ulong pointer, int length) : this(backend, pointer, length, length)
    {
    }
}

/// <summary>
/// CUDA implementation. Memory comes from a caching allocator (freed blocks are kept per size
/// and reused, so a training loop stops calling cuMemAlloc after its first iteration), and all
/// kernels are the PTX from <see cref="PtxKernels"/>, launched on the default stream.
/// </summary>
internal sealed unsafe partial class CudaBackend : Backend
{
    private static readonly Lazy<(int Count, string Reason)> Probe = new(ProbeDriver);
    private static readonly Lazy<CudaBackend>[] Instances = CreateInstances();

    [ThreadStatic]
    private static IntPtr t_currentContext;

    private readonly IntPtr _context;

    /// <summary>
    /// All work runs on this (blocking) stream rather than the legacy default stream, because only work on an
    /// explicit stream can be recorded into a CUDA Graph. Synchronous copies on the legacy stream still order
    /// correctly with it.
    /// </summary>
    private readonly IntPtr _stream;

    /// <summary>Non-null while a graph is being recorded: blocks freed during capture, owned by the graph.</summary>
    private Dictionary<int, Stack<ulong>>? _captureFree;

    /// <summary>
    /// Work on the stream (launches, copies, allocation, synchronization) holds this for reading; recording a graph
    /// holds it for writing from BeginCapture to EndCapture. Recording captures everything issued on the stream, and
    /// the driver refuses synchronous copies and synchronization while it records, so other threads wait until the
    /// recording ends rather than landing in the graph or failing. The recording thread's own work still goes through.
    /// </summary>
    private readonly ReaderWriterLockSlim _streamGate = new(LockRecursionPolicy.SupportsRecursion);

    /// <summary>Managed thread id of the thread recording a graph (0 when none).</summary>
    private int _captureThread;

    private StreamUse UseStream()
    {
        _streamGate.EnterReadLock();
        return new StreamUse(_streamGate);
    }

    private readonly struct StreamUse(ReaderWriterLockSlim gate) : IDisposable
    {
        public void Dispose() => gate.ExitReadLock();
    }
    private readonly Dictionary<int, Stack<ulong>> _pool = [];

    // Sizes with cached blocks, for best-fit reuse when no block of the exact size is cached.
    private readonly SortedSet<int> _poolSizes = [];

    // A cached block up to this fraction larger than a request (of at least BestFitMinimum floats) may serve it.
    private const int BestFitSlack = 4;                                          // 1 / 4: at most 25% larger
    private const int BestFitMinimum = 1024;
    private readonly MemoryAccountant _memory;
    private readonly int _multiprocessors;
    private readonly int _computeMajor, _computeMinor;

    // The tensor-core module (compute capability 8.0 and newer), loaded on first use; null when unavailable.
    private Dictionary<string, IntPtr>? _tensorCore;
    private bool _tensorCoreTried;

    /// <summary>Why bfloat16 tensor-core products are unavailable on this GPU (null when they are available or untried).</summary>
    public string? TensorCoreUnavailableReason { get; private set; }

    /// <summary>Tensor-core modules the driver could not load, with its message (empty when all loaded).</summary>
    public IReadOnlyList<string> TensorCoreModuleErrors { get; private set; } = [];

    // Every tensor-core kernel that loaded (attention even when the products did not).
    private Dictionary<string, IntPtr> _tensorCoreAny = [];

    /// <summary>
    /// Tensor-core products copy a transposed operand into [m, k] / [k, n] layout first. Off: the transposing kernels
    /// measured as fast as the plain one (tests --bench-gemm), so the copy only costs time and memory.
    /// </summary>
    internal static bool PretransposeForTensorCores;

    // Dynamic shared memory of the next launch on this thread (consumed by Launch).
    [ThreadStatic]
    private static uint t_sharedBytes;

    // Kernel times per name while GpuProfiler runs (null otherwise); matrix products are keyed by kernel and shape.
    private Dictionary<string, (long Calls, long Ticks, double Flops)>? _profile;
    private (IntPtr Start, IntPtr End) _profileEvents;

    [ThreadStatic]
    private static string? _profileLabel;

    [ThreadStatic]
    private static double _profileFlops;

    public void StartProfile() => _profile = [];

    public Dictionary<string, (long Calls, long Ticks, double Flops)> StopProfile()
    {
        var profile = _profile ?? [];
        _profile = null;
        return profile;
    }

    /// <summary>Number of tensor-core GEMM launches (for tests and diagnostics).</summary>
    internal long TensorCoreLaunches;

    public string? TensorCoresUnavailable()
    {
        lock (_signatures)
        {
            return TensorCoreKernels() is null ? TensorCoreUnavailableReason ?? "the tensor-core products did not load" : null;
        }
    }

    // A loaded tensor-core kernel by name, or null.
    private IntPtr? TensorKernel(string name)
    {
        TensorCoreKernels();
        return _tensorCoreAny.TryGetValue(name, out var function) ? function : null;
    }

    private Dictionary<string, IntPtr>? TensorCoreKernels()
    {
        if (_tensorCoreTried)
        {
            return _tensorCore;
        }

        _tensorCoreTried = true;
        if (_computeMajor < 8)
        {
            TensorCoreUnavailableReason = $"compute capability {_computeMajor}.x has no bfloat16 tensor cores (8.0 or newer needed)";
            return null;
        }

        // Each module on its own: a driver that rejects one (a JIT compiler error) still runs the others.
        MakeCurrent();
        var kernels = new Dictionary<string, IntPtr>();
        var errors = new List<string>();
        foreach (var (moduleName, names, source) in PtxKernels.TensorCoreModules)
        {
            if (moduleName.StartsWith("fp8", StringComparison.Ordinal) && _computeMajor * 10 + _computeMinor < 89)
            {
                continue;                                           // FP8 tensor cores: compute capability 8.9 and newer
            }

            try
            {
                IntPtr module = LoadModule(source);
                foreach (var kernel in names)
                {
                    byte[] bytes = Encoding.ASCII.GetBytes(kernel + "\0");
                    fixed (byte* p = bytes)
                    {
                        Check(cuModuleGetFunction(out IntPtr function, module, p), $"cuModuleGetFunction({kernel})");
                        _signatures[function] = (kernel, PtxKernels.TensorCoreParameterCounts[kernel]);
                        if (kernel.StartsWith("flash_tc_bwd", StringComparison.Ordinal))
                        {
                            int dim = kernel.EndsWith("d64", StringComparison.Ordinal) ? 64 : 128;
                            int sharedBytes = kernel.Contains("_kv_", StringComparison.Ordinal) ? PtxKernels.FlashTensorBackwardKvShared(dim) : PtxKernels.FlashTensorBackwardQShared(dim);
                            Check(cuFuncSetAttribute(function, FunctionAttributeMaxDynamicSharedSizeBytes, sharedBytes), $"cuFuncSetAttribute({kernel})");
                        }

                        if (kernel.StartsWith("gemm8_", StringComparison.Ordinal))
                        {
                            Check(cuFuncSetAttribute(function, FunctionAttributeMaxDynamicSharedSizeBytes, PtxKernels.EightBitShared), $"cuFuncSetAttribute({kernel})");
                        }

                        kernels[kernel] = function;
                    }
                }
            }
            catch (CudaException e)
            {
                errors.Add($"{moduleName}: {e.Message}");
            }
        }

        TensorCoreModuleErrors = errors;
        if (errors.Count > 0)
        {
            TensorCoreUnavailableReason = string.Join(" | ", errors);
        }

        _tensorCore = kernels.ContainsKey("gemm_tc_nn_f32") ? kernels : null;
        _tensorCoreAny = kernels;
        return _tensorCore;
    }

    // Kernel name and declared parameter count per loaded function: a launch with the wrong number of arguments
    // would make the driver read past the argument array (CUDA_ERROR_INVALID_VALUE or silent garbage).
    private readonly Dictionary<IntPtr, (string Name, int Parameters)> _signatures = [];
    private readonly IntPtr _fill, _affine, _axpy, _mulAdd, _add, _sub, _mul, _clipFactor, _packBFloat16;
    private readonly IntPtr _sigmoid, _tanh, _relu, _square, _abs;
    private readonly IntPtr _sigmoidBwd, _tanhBwd, _reluBwd, _squareBwd, _absBwd, _dropout;
    private readonly IntPtr _addRowVec, _addScalar, _sumRows, _sum, _sgdMomentum, _adam, _matmul;

    private CudaBackend(int ordinal)
    {
        Check(cuDeviceGet(out int device, ordinal), nameof(cuDeviceGet));
        Check(cuDevicePrimaryCtxRetain(out _context, device), nameof(cuDevicePrimaryCtxRetain));
        MakeCurrent();
        Check(cuStreamCreate(out _stream, 0), nameof(cuStreamCreate));

        byte* name = stackalloc byte[256];
        Check(cuDeviceGetName(name, 256, device), nameof(cuDeviceGetName));
        Check(cuDeviceTotalMem(out nuint memory, device), nameof(cuDeviceTotalMem));
        Check(cuDeviceGetAttribute(out _multiprocessors, AttributeMultiprocessorCount, device), nameof(cuDeviceGetAttribute));
        Check(cuDeviceGetAttribute(out _computeMajor, AttributeComputeCapabilityMajor, device), nameof(cuDeviceGetAttribute));
        Check(cuDeviceGetAttribute(out _computeMinor, AttributeComputeCapabilityMinor, device), nameof(cuDeviceGetAttribute));
        // The CUDA version the installed driver supports (13000 = 13.0), not the driver's own release number.
        string cuda = cuDriverGetVersion(out int version) == 0 ? $", CUDA {version / 1000}.{version % 1000 / 10} driver" : "";
        Name = $"{Marshal.PtrToStringAnsi((IntPtr)name)} ({memory / (1024 * 1024)} MiB, {_multiprocessors} SMs, compute {_computeMajor}.{_computeMinor}{cuda})";
        _memory = new MemoryAccountant(() => ComputeResources.GpuMemoryLimit, $"cuda:{ordinal}");

        IntPtr module = LoadModule(PtxKernels.Source);
        IntPtr Fn(string kernel)
        {
            byte[] bytes = Encoding.ASCII.GetBytes(kernel + "\0");
            fixed (byte* p = bytes)
            {
                Check(cuModuleGetFunction(out IntPtr function, module, p), $"cuModuleGetFunction({kernel})");
                _signatures[function] = (kernel, PtxKernels.ParameterCounts[kernel]);
                return function;
            }
        }

        _fill = Fn("fill_f32");
        _affine = Fn("affine_f32");
        _clipFactor = Fn("clip_factor_f32");
        _packBFloat16 = Fn("bf16_pack_f32");
        _axpy = Fn("axpy_f32");
        _mulAdd = Fn("muladd_f32");
        _add = Fn("add_f32");
        _sub = Fn("sub_f32");
        _mul = Fn("mul_f32");
        _sigmoid = Fn("sigmoid_f32");
        _tanh = Fn("tanh_f32");
        _relu = Fn("relu_f32");
        _square = Fn("square_f32");
        _abs = Fn("abs_f32");
        _sigmoidBwd = Fn("sigmoid_bwd_f32");
        _tanhBwd = Fn("tanh_bwd_f32");
        _reluBwd = Fn("relu_bwd_f32");
        _squareBwd = Fn("square_bwd_f32");
        _absBwd = Fn("abs_bwd_f32");
        _dropout = Fn("dropout_f32");
        _addRowVec = Fn("add_rowvec_f32");
        _addScalar = Fn("add_scalar_f32");
        _sumRows = Fn("sum_rows_f32");
        _sum = Fn("sum_f32");
        _sgdMomentum = Fn("sgd_momentum_f32");
        _adam = Fn("adam_f32");
        _matmul = Fn("matmul_f32");
        _kernels = PtxKernels.AdvancedNames.Concat(PtxKernels.DecodingNames).Concat(PtxKernels.QuantizedNames).Concat(PtxKernels.DecoderNames).Concat(PtxKernels.RowNames).Append("add_dropout_f32").ToDictionary(k => k, Fn);
    }

    public static int DeviceCount => Probe.Value.Count;

    public static string UnavailableReason => Probe.Value.Reason;

    public string Name { get; }

    public static CudaBackend Get(int ordinal) => Instances[ordinal].Value;

    public static bool IsInitialized(int ordinal) => Instances[ordinal].IsValueCreated;

    private static (int, string) ProbeDriver()
    {
        if (Environment.GetEnvironmentVariable("IDRAK_DISABLE_CUDA") is "1" or "true")
        {
            return (0, "disabled by the IDRAK_DISABLE_CUDA environment variable");
        }

        if (!TryLoad(out string reason))
        {
            return (0, reason);
        }

        try
        {
            int result = cuInit(0);
            if (result != 0)
            {
                return (0, $"cuInit failed with error {result} (no usable NVIDIA GPU?)");
            }

            Check(cuDeviceGetCount(out int count), nameof(cuDeviceGetCount));
            return count > 0 ? (count, "") : (0, "the NVIDIA driver reports no CUDA devices");
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException or CudaException)
        {
            return (0, ex.Message);
        }
    }

    private static Lazy<CudaBackend>[] CreateInstances()
    {
        int count = DeviceCount;
        var instances = new Lazy<CudaBackend>[count];
        for (int i = 0; i < count; i++)
        {
            int ordinal = i;
            instances[i] = new Lazy<CudaBackend>(() => new CudaBackend(ordinal));
        }

        return instances;
    }

    private static IntPtr LoadModule(string ptx)
    {
        const int LogSize = 16 * 1024;
        byte[] image = Encoding.ASCII.GetBytes(ptx + "\0");
        byte* log = stackalloc byte[LogSize];
        log[0] = 0;
        int* options = stackalloc int[] { JitErrorLogBuffer, JitErrorLogBufferSizeBytes };
        void** values = stackalloc void*[] { log, (void*)LogSize };
        fixed (byte* p = image)
        {
            int result = cuModuleLoadDataEx(out IntPtr module, p, 2, options, values);
            if (result != 0)
            {
                string details = Marshal.PtrToStringAnsi((IntPtr)log) ?? "";
                throw new CudaException($"The CUDA driver could not JIT-compile the Idrak kernels (error {result}). {details}");
            }

            return module;
        }
    }

    /// <summary>Binds this device's context to the calling thread (cheap no-op when already bound).</summary>
    private void MakeCurrent()
    {
        if (t_currentContext != _context)
        {
            Check(cuCtxSetCurrent(_context), nameof(cuCtxSetCurrent));
            t_currentContext = _context;
        }
    }

    public override Storage Allocate(int length, bool zeroed)
    {
        MakeCurrent();
        using var use = UseStream();
        long bytes = BlockBytes(length);
        bool releaseCache;
        try
        {
            releaseCache = _memory.MustReleaseCacheFor(bytes); // throws when over the in-use limit
        }
        catch (ResourceLimitExceededException) when (ComputeResources.OffloadToHostMemory)
        {
            return AllocateHost(length, zeroed);
        }

        ulong pointer = 0;
        int capacity = length;
        lock (_pool)
        {
            if (_captureFree is not null && _captureFree.TryGetValue(length, out var captured) && captured.Count > 0)
            {
                // Reuse a block freed earlier in this capture: stream order keeps the recorded uses apart.
                pointer = captured.Pop();
                if (_hostBlocks.ContainsKey(pointer))
                {
                    _memory.Offloaded(bytes);
                }
                else
                {
                    _memory.Reused(bytes);
                }
            }
            else if (TakeCached(length, out capacity) is var cached && cached != 0)
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

            pointer = AllocateDevice(length);
            if (pointer == 0)
            {
                return AllocateHost(length, zeroed);
            }

            _memory.Allocated(bytes);
        }

        if (zeroed && length > 0)
        {
            Check(cuMemsetD32Async(pointer, 0, (nuint)length, _stream), nameof(cuMemsetD32Async));
        }

        return new CudaStorage(this, pointer, length, capacity);
    }

    // A cached block for `length` floats: the exact size, else the smallest cached size at most 25% larger (varying
    // shapes, e.g. prompt lengths, then reuse blocks instead of caching one of every size). Called under the pool lock.
    private ulong TakeCached(int length, out int capacity)
    {
        capacity = length;
        if (!_pool.TryGetValue(length, out var bucket) || bucket.Count == 0)
        {
            if (length < BestFitMinimum || _poolSizes.Count == 0)
            {
                return 0;
            }

            var fitting = _poolSizes.GetViewBetween(length + 1, length + length / BestFitSlack);
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

    // New GPU memory, or 0 when the GPU is full and offloading is on (the caller then uses system memory). A block
    // "fits" when the GPU keeps ComputeResources.GpuMemoryReserve free afterwards, so the driver never pages GPU
    // memory out on its own.
    private ulong AllocateDevice(int length)
    {
        nuint bytes = (nuint)BlockBytes(length);
        bool Fits() => cuMemGetInfo(out nuint free, out _) != 0 || (long)free - (long)bytes >= ComputeResources.GpuMemoryReserve;
        int result = ErrorOutOfMemory;
        ulong pointer = 0;
        for (int attempt = 0; attempt < 2 && result == ErrorOutOfMemory; attempt++)
        {
            if (attempt == 1)
            {
                // Give back cached blocks and anything held only by unreachable tensors, then retry once.
                GC.Collect();
                GC.WaitForPendingFinalizers();
                ReleaseCachedMemory();
            }

            result = Fits() ? cuMemAlloc(out pointer, bytes) : ErrorOutOfMemory;
        }

        if (result == ErrorOutOfMemory)
        {
            if (ComputeResources.OffloadToHostMemory)
            {
                return 0;
            }

            cuMemGetInfo(out nuint free, out nuint total);
            throw new ResourceLimitExceededException(
                $"{Name} is out of memory: {bytes:N0} more bytes needed, {free:N0} of {total:N0} free with {ComputeResources.GpuMemoryReserve:N0} kept in reserve " +
                $"({_memory.Usage}). Use smaller batches or shorter sequences, or set ComputeResources.OffloadToHostMemory (IDRAK_OFFLOAD=1) " +
                "to keep what does not fit in system memory (slower).");
        }

        Check(result, nameof(cuMemAlloc));
        return pointer;
    }

    // Pinned system memory the GPU reads and writes over PCIe, addressed by the kernels like GPU memory (unified
    // addressing): used when the GPU is full and ComputeResources.OffloadToHostMemory is on.
    private CudaStorage AllocateHost(int length, bool zeroed)
    {
        ulong pointer = 0;
        lock (_pool)
        {
            if (_hostPool.TryGetValue(length, out var bucket) && bucket.Count > 0)
            {
                pointer = bucket.Pop();
            }
        }

        if (pointer == 0)
        {
            Check(cuMemHostAlloc(out IntPtr host, (nuint)BlockBytes(length), HostAllocPortable | HostAllocDeviceMap), nameof(cuMemHostAlloc));
            Check(cuMemHostGetDevicePointer(out pointer, host, 0), nameof(cuMemHostGetDevicePointer));
            lock (_pool)
            {
                _hostBlocks[pointer] = host;
            }
        }

        _memory.Offloaded(BlockBytes(length));
        if (zeroed && length > 0)
        {
            Check(cuMemsetD32Async(pointer, 0, (nuint)length, _stream), nameof(cuMemsetD32Async));
        }

        return new CudaStorage(this, pointer, length, length);
    }

    // System-memory blocks by device address (to their host address), and the unused ones by size.
    private readonly Dictionary<ulong, IntPtr> _hostBlocks = [];
    private readonly Dictionary<int, Stack<ulong>> _hostPool = [];

    private static long BlockBytes(int length) => (long)Math.Max(length, 1) * sizeof(float);

    public override MemoryUsage GetMemoryUsage() => _memory.Usage;

    /// <summary>Frees every cached (currently unused) device block back to the driver.</summary>
    public override void ReleaseCachedMemory()
    {
        MakeCurrent();

        // A cached block was released by the host, but kernels queued earlier may still read or write it: wait for
        // them before the memory goes back to the driver.
        Check(cuStreamSynchronize(_stream), nameof(cuStreamSynchronize));
        lock (_pool)
        {
            foreach (var (length, bucket) in _pool)
            {
                while (bucket.Count > 0)
                {
                    Check(cuMemFree(bucket.Pop()), nameof(cuMemFree));
                    _memory.Freed(BlockBytes(length));
                }
            }

            _poolSizes.Clear();
            foreach (var (_, bucket) in _hostPool)
            {
                while (bucket.Count > 0)
                {
                    ulong pointer = bucket.Pop();
                    Check(cuMemFreeHost(_hostBlocks[pointer]), nameof(cuMemFreeHost));
                    _hostBlocks.Remove(pointer);
                }
            }
        }
    }

    // Called when the last reference is released, possibly from the finalizer thread, so it only
    // touches the pool and never the driver.
    private protected override void Detach(Storage storage) => ((CudaStorage)storage).Pointer = 0;

    private protected override void Attach(Storage storage, Storage fresh)
    {
        var (s, f) = ((CudaStorage)storage, (CudaStorage)fresh);
        (s.Pointer, s.Capacity) = (f.Pointer, f.Capacity);                // the fresh storage object is dropped, its block kept
    }

    public override void Return(Storage storage)
    {
        var s = (CudaStorage)storage;
        lock (_pool)
        {
            bool recording = _captureFree is not null && Environment.CurrentManagedThreadId == _captureThread;
            if (_hostBlocks.ContainsKey(s.Pointer) && recording)
            {
                // Freed while a graph is recorded: the graph owns it (see TakeCaptureBlocks).
                if (!_captureFree!.TryGetValue(s.Capacity, out var captured))
                {
                    _captureFree[s.Capacity] = captured = new Stack<ulong>();
                }

                captured.Push(s.Pointer);
                _memory.Offloaded(-BlockBytes(s.Capacity));
                return;
            }

            if (_hostBlocks.ContainsKey(s.Pointer))
            {
                if (!_hostPool.TryGetValue(s.Capacity, out var hostBucket))
                {
                    _hostPool[s.Capacity] = hostBucket = new Stack<ulong>();
                }

                hostBucket.Push(s.Pointer);
                _memory.Offloaded(-BlockBytes(s.Capacity));
                return;
            }

            // While recording a graph, blocks the recording frees belong to the graph: returning them to the shared pool
            // would let unrelated tensors reuse memory the graph writes on every replay. (Other threads' blocks,
            // including the finalizer's, go back to the pool.)
            var target = _captureFree is not null && Environment.CurrentManagedThreadId == _captureThread ? _captureFree : _pool;
            if (!target.TryGetValue(s.Capacity, out var bucket))
            {
                target[s.Capacity] = bucket = new Stack<ulong>();
            }

            bucket.Push(s.Pointer);
            if (ReferenceEquals(target, _pool))
            {
                _poolSizes.Add(s.Capacity);
            }

            _memory.Returned(BlockBytes(s.Capacity));
        }
    }

    public override void Upload(ReadOnlySpan<float> source, Storage destination)
    {
        MakeCurrent();
        using var use = UseStream();
        fixed (float* p = source)
        {
            Check(cuMemcpyHtoD(P(destination), p, (nuint)source.Length * sizeof(float)), nameof(cuMemcpyHtoD));
        }
    }

    public override void Download(Storage source, Span<float> destination) => DownloadRange(source, 0, destination);

    public override void DownloadRange(Storage source, int offset, Span<float> destination)
    {
        MakeCurrent();
        using var use = UseStream();
        fixed (float* p = destination)
        {
            Check(cuMemcpyDtoH(p, P(source) + (ulong)offset * sizeof(float), (nuint)destination.Length * sizeof(float)), nameof(cuMemcpyDtoH));
        }
    }

    public override void Fill(Storage y, int n, float value) => Launch1D(_fill, n, P(y), F(value), U(n));

    public override void Copy(Storage x, Storage y, int n)
    {
        MakeCurrent();
        using var use = UseStream();
        Check(cuMemcpyDtoDAsync(P(y), P(x), (nuint)n * sizeof(float), _stream), nameof(cuMemcpyDtoDAsync));
    }

    public override void Unary(UnaryOp op, Storage x, Storage y, int n)
    {
        IntPtr fn = op switch
        {
            UnaryOp.Sigmoid => _sigmoid,
            UnaryOp.Tanh => _tanh,
            UnaryOp.Relu => _relu,
            UnaryOp.Square => _square,
            UnaryOp.Abs => _abs,
            UnaryOp.Exp => _kernels["exp_f32"],
            UnaryOp.Log => _kernels["log_f32"],
            UnaryOp.Gelu => _kernels["gelu_f32"],
            _ => throw new ArgumentOutOfRangeException(nameof(op)),
        };
        Launch1D(fn, n, P(x), P(y), U(n));
    }

    public override void UnaryBackward(UnaryOp op, Storage x, Storage y, Storage dy, Storage dx, int n)
    {
        IntPtr fn = op switch
        {
            UnaryOp.Sigmoid => _sigmoidBwd,
            UnaryOp.Tanh => _tanhBwd,
            UnaryOp.Relu => _reluBwd,
            UnaryOp.Square => _squareBwd,
            UnaryOp.Abs => _absBwd,
            UnaryOp.Exp => _kernels["exp_bwd_f32"],
            UnaryOp.Log => _kernels["log_bwd_f32"],
            UnaryOp.Gelu => _kernels["gelu_bwd_f32"],
            _ => throw new ArgumentOutOfRangeException(nameof(op)),
        };
        Launch1D(fn, n, P(x), P(y), P(dy), P(dx), U(n));
    }

    public override void Binary(BinaryOp op, Storage a, Storage b, Storage c, int n)
    {
        IntPtr fn = op switch
        {
            BinaryOp.Add => _add,
            BinaryOp.Sub => _sub,
            BinaryOp.Mul => _mul,
            _ => throw new ArgumentOutOfRangeException(nameof(op)),
        };
        Launch1D(fn, n, P(a), P(b), P(c), U(n));
    }

    public override void Affine(Storage x, Storage y, int n, float alpha, float beta) => Launch1D(_affine, n, P(x), P(y), F(alpha), F(beta), U(n));

    public override void PackBFloat16(Storage x, Storage packed, int n) =>
        Launch1D(_packBFloat16, (n + 1) / 2, P(x), P(packed), U(n), U((n + 1) / 2));

    public override void ClipFactor(Storage sumSquares, Storage factor, float maxNorm) => Launch1D(_clipFactor, 1, P(sumSquares), P(factor), F(maxNorm), U(1));

    public override void Axpy(Storage x, Storage y, int n, float alpha) => Launch1D(_axpy, n, P(x), P(y), F(alpha), U(n));

    public override void MulAdd(Storage a, Storage b, Storage c, int n) => Launch1D(_mulAdd, n, P(a), P(b), P(c), U(n));

    public override void AddRowVector(Storage a, Storage v, Storage c, int rows, int cols)
    {
        int n = rows * cols;
        Launch1D(_addRowVec, n, P(a), P(v), P(c), U(cols), U(n));
    }

    public override void SumRows(Storage x, Storage y, int rows, int cols)
    {
        if (rows >= 256)
        {
            SumColumns(x, 0, cols, y, rows, cols);                  // chunks of rows in parallel (a bias gradient)
            return;
        }

        Launch1D(_sumRows, cols, P(x), P(y), U(rows), U(cols));
    }

    public override void Sum(Storage x, Storage result, int n, float scale)
    {
        MakeCurrent();
        using (UseStream())
        {
            Check(cuMemsetD32Async(P(result), 0, 1, _stream), nameof(cuMemsetD32Async));
        }

        if (n == 0)
        {
            return;
        }

        // Enough blocks to fill the GPU; the grid-stride loop covers the rest.
        uint blocks = (uint)Math.Min((n + PtxKernels.BlockSize - 1) / PtxKernels.BlockSize, Math.Max(1, _multiprocessors) * 8);
        Launch(_sum, blocks, 1, PtxKernels.BlockSize, 1, P(x), P(result), U(n), F(scale));
    }

    public override void AxpyAt(Storage x, Storage y, int offset, float alpha) =>
        Launch1D(_addScalar, 1, P(x), P(y) + (ulong)offset * sizeof(float), F(alpha), U(1));

    public override void AddBroadcastScalar(Storage s, Storage y, int n, float scale) => Launch1D(_addScalar, n, P(s), P(y), F(scale), U(n));

    public override void MatMulMany(Storage a, int m, int k, ReadOnlySpan<(Storage Weight, Storage? Bias, Storage Output, int Columns)> products)
    {
        if (products.Length is 0 or > 3 || m > PtxKernels.GemvRows)
        {
            base.MatMulMany(a, m, k, products);
            return;
        }

        // One launch for all of them: grid y picks the product (up to three), x covers the widest.
        var args = new ulong[3 + 4 * 3];
        args[0] = P(a);
        args[1] = U(m);
        args[2] = U(k);
        int widest = 0;
        for (int j = 0; j < 3; j++)
        {
            var (weight, bias, output, columns) = products[Math.Min(j, products.Length - 1)];
            bool used = j < products.Length;
            args[3 + 4 * j] = P(weight);
            args[4 + 4 * j] = bias is null ? 0UL : P(bias);
            args[5 + 4 * j] = P(output);
            args[6 + 4 * j] = used ? U(columns) : 0UL;
            widest = Math.Max(widest, used ? columns : 0);
        }

        Launch(K("gemv_multi_f32"), (uint)((widest + 31) / 32), (uint)products.Length, 1, PtxKernels.GemvThreads, 1, args);
    }

    public override void BatchedMatMul(Storage a, Storage b, Storage c, int batch, int m, int n, int k, bool transA, bool transB, float beta)
    {
        if (m == 0 || n == 0 || batch == 0)
        {
            return;
        }

        const int T = PtxKernels.Tile;
        const int MaxGridZ = 65535;
        ulong mk = (ulong)m * (ulong)k, kn = (ulong)k * (ulong)n, mn = (ulong)m * (ulong)n;
        // Token-by-token decoding: few rows through a large matrix read each weight once. (Smaller products keep the
        // tiled kernel, whose sums do not depend on the number of rows, so small models predict identically in any batch.)
        bool few = m <= PtxKernels.GemvRows && !transA && (long)n * k >= 1 << 16;

        // Larger products: register-blocked tiles, 128 × 128 when that still gives every multiprocessor a block, else
        // 64 × 64. (Small ones keep the 16 × 16 kernel; all add k terms in the same order, so results are identical.)
        int gemmTile = 0;
        if (!few && m >= 64 && n >= 64 && k >= 8)
        {
            long tiles128 = (long)((m + 127) / 128) * ((n + 127) / 128) * batch;
            gemmTile = tiles128 >= Math.Max(1, _multiprocessors) ? 128 : 64;
        }
        // bfloat16 tensor cores (MixedPrecision): products large enough to fill 128 × 128 tiles, and skinny ones with a
        // side of 8-63 (a LoRA adapter's rank-16 products: x·A, g·Bᵀ, xᵀ·dt, u·B), which the float kernels ran at 1-3
        // TFLOPS; with k split across blocks they read their wide operand about as fast as memory allows.
        bool large = m >= 64 && n >= 64 && k >= 32;
        // (Also a long k over a small output, such as the rank-16 gradient of a 128-wide key/value projection: 16 × 128 ×
        // tokens, which the 16 × 16 float kernel ran at 0.1 TFLOPS with 8 blocks.)
        bool skinny = batch == 1 && Math.Min(m, n) >= 8 && k >= 16
            && (Math.Max(m, n) >= 256 && (long)m * n * k >= 1 << 24 || Math.Max(m, n) >= 64 && k >= 2048);
        var tensorCore = !few && (large || skinny) && MixedPrecision.UsesTensorCores ? TensorCoreKernels() : null;
        if (tensorCore is not null && large && MixedPrecision.Current == MatMulPrecision.Float8 && EightBitReady(fp8: true) && batch <= 64)
        {
            for (int i = 0; i < batch; i++)
            {
                ulong at = (ulong)i * mk * 4, bt = (ulong)i * kn * 4, ct = (ulong)i * mn * 4;
                Gemm8(true, P(a) + at, transA ? m : k, transA, P(b) + bt, transB ? k : n, transB, P(c) + ct, n, m, n, k, beta, 0UL, GemmEpilogue.None, 0UL);
            }

            return;
        }

        if (tensorCore is not null && (transA || transB) && batch <= MaxGridZ && PretransposeForTensorCores)
        {
            // The tensor-core kernel is fastest with both operands as stored ([m, k] · [k, n]): a transposed operand is
            // copied into that layout first (a memory-bound pass, far cheaper than the product it speeds up).
            Storage? at = transA ? Allocate(batch * m * k, zeroed: false) : null;
            Storage? bt = transB ? Allocate(batch * k * n, zeroed: false) : null;
            try
            {
                if (at is not null)
                {
                    TransposeBatched(a, at, batch, k, m);
                }

                if (bt is not null)
                {
                    TransposeBatched(b, bt, batch, n, k);
                }

                BatchedMatMul(at ?? a, bt ?? b, c, batch, m, n, k, false, false, beta);
            }
            finally
            {
                at?.Release();
                bt?.Release();
            }

            return;
        }

        for (int first = 0; first < batch; first += MaxGridZ)
        {
            int count = Math.Min(MaxGridZ, batch - first);
            ulong offset = (ulong)first * sizeof(float);
            if (few)
            {
                uint columnsPerBlock = transB ? 8u : 32u;
                Launch(K(transB ? "gemv_nt_f32" : "gemv_nn_f32"), (uint)((n + columnsPerBlock - 1) / columnsPerBlock), 1, (uint)count,
                    (uint)(transB ? PtxKernels.RowThreads : PtxKernels.GemvThreads), 1, P(a) + offset * mk, P(b) + offset * kn, P(c) + offset * mn, U(m), U(n), U(k), F(beta), mk, kn, mn);
                continue;
            }

            if (_profile is not null)
            {
                string kind = tensorCore is not null ? "gemm_tc" : few ? "gemv" : gemmTile > 0 ? $"gemm{gemmTile}" : "matmul16";
                _profileLabel = $"{kind}_{(transA ? 't' : 'n')}{(transB ? 't' : 'n')} {m}x{n}x{k}{(count > 1 ? $" batch {count}" : "")}";
                _profileFlops = 2.0 * m * n * k * count;
            }

            if (tensorCore is not null)
            {
                string kernel = transA ? (transB ? "gemm_tc_tt_f32" : "gemm_tc_tn_f32") : (transB ? "gemm_tc_nt_f32" : "gemm_tc_nn_f32");
                int splits = count == 1 ? TensorSplits(m, n, k, beta, P(c), n) : 1;
                Launch(tensorCore[kernel], (uint)((n + PtxKernels.TensorTile - 1) / PtxKernels.TensorTile), (uint)((m + PtxKernels.TensorTile - 1) / PtxKernels.TensorTile),
                    (uint)(splits > 1 ? splits : count), PtxKernels.TensorThreads, 1, P(a) + offset * mk, P(b) + offset * kn, P(c) + offset * mn,
                    U(m), U(n), U(k), F(beta), splits > 1 ? 0UL : mk, splits > 1 ? 0UL : kn, splits > 1 ? 0UL : mn, 0UL, U(transA ? m : k), U(transB ? k : n), U(n), 0UL);
                Interlocked.Increment(ref TensorCoreLaunches);
                continue;
            }

            if (gemmTile > 0)
            {
                Launch(K(gemmTile == 128 ? "gemm128_f32" : "gemm64_f32"), (uint)((n + gemmTile - 1) / gemmTile), (uint)((m + gemmTile - 1) / gemmTile),
                    (uint)count, PtxKernels.GemmThreads, 1, P(a) + offset * mk, P(b) + offset * kn, P(c) + offset * mn,
                    U(m), U(n), U(k), U(transA ? 1 : 0), U(transB ? 1 : 0), F(beta), mk, kn, mn);
                continue;
            }

            Launch(_matmul, (uint)((n + T - 1) / T), (uint)((m + T - 1) / T), (uint)count, T, T,
                P(a) + offset * mk, P(b) + offset * kn, P(c) + offset * mn, U(m), U(n), U(k), U(transA ? 1 : 0), U(transB ? 1 : 0), F(beta),
                mk, kn, mn);
        }
    }

    internal void TransposeForBenchmark(Storage x, Storage y, int rows, int cols) => TransposeBatched(x, y, 1, rows, cols);

    // y[b] = x[b]ᵀ for x [batch][rows, cols].
    private void TransposeBatched(Storage x, Storage y, int batch, int rows, int cols) =>
        Launch(K("transpose_f32"), (uint)((cols + 31) / 32), (uint)((rows + 31) / 32), (uint)batch, 32, 8,
            P(x), P(y), U(rows), U(cols), (ulong)rows * (ulong)cols, (ulong)rows * (ulong)cols);

    public override bool MatMulBias(Storage a, Storage b, Storage bias, Storage c, int m, int n, int k)
    {
        if (m < 64 || n < 64 || k < 32 || !MixedPrecision.UsesTensorCores || TensorCoreKernels() is not { } tensorCore)
        {
            return false;
        }

        if (MixedPrecision.Current == MatMulPrecision.Float8 && EightBitReady(fp8: true))
        {
            Gemm8(true, P(a), k, false, P(b), n, false, P(c), n, m, n, k, 0f, P(bias), GemmEpilogue.None, 0UL);
            return true;
        }

        if (_profile is not null)
        {
            _profileLabel = $"gemm_tc_nn+bias {m}x{n}x{k}";
            _profileFlops = 2.0 * m * n * k;
        }

        Launch(tensorCore["gemm_tc_nn_f32"], (uint)((n + PtxKernels.TensorTile - 1) / PtxKernels.TensorTile), (uint)((m + PtxKernels.TensorTile - 1) / PtxKernels.TensorTile),
            1, PtxKernels.TensorThreads, 1, P(a), P(b), P(c), U(m), U(n), U(k), F(0f), 0UL, 0UL, 0UL, P(bias), U(k), U(n), U(n), 0UL);
        Interlocked.Increment(ref TensorCoreLaunches);
        return true;
    }

    public override bool MatMulLowRank(Storage a, Storage b, Storage c, int m, int n, int k, bool transB, float beta, Storage u, Storage v, int rank)
    {
        if (m == 0 || n == 0 || k < 16 || rank is < 1 or > 32 || !MixedPrecision.UsesTensorCores || MixedPrecision.Current == MatMulPrecision.Float8
            || TensorCoreKernels() is not { } tensorCore || !tensorCore.TryGetValue(transB ? "gemm_tc_nt_lr_f32" : "gemm_tc_nn_lr_f32", out var function))
        {
            return false;
        }

        if (_profile is not null)
        {
            _profileLabel = $"gemm_tc_{(transB ? "nt" : "nn")}_lr {m}x{n}x{k}+{rank}";
            _profileFlops = 2.0 * m * n * (k + rank);
        }

        int splits = TensorSplits(m, n, k, beta, P(c), n);
        Launch(function, (uint)((n + PtxKernels.TensorTile - 1) / PtxKernels.TensorTile), (uint)((m + PtxKernels.TensorTile - 1) / PtxKernels.TensorTile),
            (uint)splits, PtxKernels.TensorThreads, 1, P(a), P(b), P(c), U(m), U(n), U(k), F(beta), 0UL, 0UL, 0UL, 0UL,
            U(k), U(transB ? k : n), U(n), 0UL, P(u), P(v), U(rank));
        Interlocked.Increment(ref TensorCoreLaunches);
        return true;
    }

    public override bool BFloat16TransposedMatMul(Storage a, Storage packed, Storage c, int m, int n, int k, float beta, Storage? u, Storage? v, int rank)
    {
        if (m == 0 || n == 0 || k < 16 || u is not null && rank is < 1 or > 32 || !MixedPrecision.UsesTensorCores
            || MixedPrecision.Current == MatMulPrecision.Float8 || TensorCoreKernels() is not { } tensorCore
            || !tensorCore.TryGetValue("gemm_tc_nt_bf16w_lr_f32", out var function))
        {
            return false;
        }

        if (_profile is not null)
        {
            _profileLabel = $"gemm_tc_nt_bf16w{(u is null ? "" : "_lr")} {m}x{n}x{k}{(u is null ? "" : $"+{rank}")}";
            _profileFlops = 2.0 * m * n * (k + (u is null ? 0 : rank));
        }

        int splits = TensorSplits(m, n, k, beta, P(c), n);
        Launch(function, (uint)((n + PtxKernels.TensorTile - 1) / PtxKernels.TensorTile), (uint)((m + PtxKernels.TensorTile - 1) / PtxKernels.TensorTile),
            (uint)splits, PtxKernels.TensorThreads, 1, P(a), P(packed), P(c), U(m), U(n), U(k), F(beta), 0UL, 0UL, 0UL, 0UL,
            U(k), U((k + 1) / 2), U(n), 0UL, u is null ? 0UL : P(u), v is null ? 0UL : P(v), U(u is null ? 0 : rank));
        Interlocked.Increment(ref TensorCoreLaunches);
        return true;
    }

    public override bool GemmStrided(Storage a, long aOffset, int lda, bool transA, Storage b, long bOffset, int ldb, bool transB,
        Storage c, long cOffset, int ldc, int m, int n, int k, float beta, Storage? bias = null, GemmEpilogue epilogue = GemmEpilogue.None,
        Storage? aux = null, long auxOffset = 0)
    {
        if (m == 0 || n == 0 || TensorCoreKernels() is not { } tensorCore)
        {
            return false;
        }

        if (MixedPrecision.Current == MatMulPrecision.Float8 && EightBitReady(fp8: true) && k >= 16)
        {
            Gemm8(true, P(a) + (ulong)aOffset * 4, lda, transA, P(b) + (ulong)bOffset * 4, ldb, transB, P(c) + (ulong)cOffset * 4, ldc, m, n, k, beta,
                bias is null ? 0UL : P(bias), epilogue, aux is null ? 0UL : P(aux) + (ulong)auxOffset * 4);
            return true;
        }

        if (PtxKernels.GemmKernel(transA, transB, epilogue) is not { } kernel || !tensorCore.TryGetValue(kernel, out var function))
        {
            return false;
        }

        if (_profile is not null)
        {
            _profileLabel = $"gemm_tc_{(transA ? 't' : 'n')}{(transB ? 't' : 'n')}{(epilogue == GemmEpilogue.None ? "" : epilogue == GemmEpilogue.Gelu ? "+gelu" : "+gelu'")}"
                            + $"{(bias is null ? "" : "+bias")} {m}x{n}x{k}";
            _profileFlops = 2.0 * m * n * k;
        }

        int splits = epilogue == GemmEpilogue.None ? TensorSplits(m, n, k, beta, P(c) + (ulong)cOffset * 4, ldc) : 1;
        Launch(function, (uint)((n + PtxKernels.TensorTile - 1) / PtxKernels.TensorTile), (uint)((m + PtxKernels.TensorTile - 1) / PtxKernels.TensorTile),
            (uint)splits, PtxKernels.TensorThreads, 1, P(a) + (ulong)aOffset * 4, P(b) + (ulong)bOffset * 4, P(c) + (ulong)cOffset * 4, U(m), U(n), U(k), F(beta),
            0UL, 0UL, 0UL, bias is null ? 0UL : P(bias), U(lda), U(ldb), U(ldc), aux is null ? 0UL : P(aux) + (ulong)auxOffset * 4);
        Interlocked.Increment(ref TensorCoreLaunches);
        return true;
    }

    /// <summary>Benchmarks only: the k splits of plain tensor-core products instead of the heuristic's (1: none).</summary>
    internal static int? TensorSplitsOverride { get; set; }

    // k splits of a plain tensor-core product with fewer than 4 output tiles per SM and a long k (weight gradients):
    // about 16 blocks per SM, at most 8 splits, chunks of 1024 k or more (measured with --bench-gemm on an RTX 5070 Ti:
    // 768×768×12288 tn 28.6 → 59.5 TFLOPS, 3072×768×12288 49.3 → 72.5, 1024×3072×8192 65.1 → 74.2). The blocks add their
    // partial sums into c atomically, so beta must be 1, or 0 with c zeroed here (only when its rows are contiguous).
    private int TensorSplits(int m, int n, int k, float beta, ulong c, int ldc)
    {
        if (beta != 1f && (beta != 0f || ldc != n))
        {
            return 1;
        }

        int sms = Math.Max(1, _multiprocessors);
        int tiles = ((m + PtxKernels.TensorTile - 1) / PtxKernels.TensorTile) * ((n + PtxKernels.TensorTile - 1) / PtxKernels.TensorTile);
        int wanted = TensorSplitsOverride ?? (tiles >= 4 * sms ? 1 : (16 * sms + tiles - 1) / tiles);
        // A few output tiles over a long k (a LoRA adapter's gradients, [rank, width] or [width, rank] over every token):
        // up to 64 chunks of 256 or more, so the blocks cover the multiprocessors; otherwise at most 8 of 1024 or more.
        int limit = tiles <= 8 ? Math.Min(64, k / 256) : Math.Min(8, k / 1024);
        int splits = Math.Clamp(wanted, 1, Math.Max(1, limit));
        if (splits > 1 && beta == 0f)
        {
            Check(cuMemsetD32Async(c, 0, (nuint)((long)m * n), _stream), nameof(cuMemsetD32Async));
        }

        return splits;
    }

    // The 8-bit products and their quantizers loaded (FP8: compute capability 8.9+; INT8: 8.0+).
    // Benchmarks: x [rows][ld] quantized by rows (the a operand's quantizer).
    internal void QuantizeRowsForBenchmark(bool fp8, Storage x, int ld, Storage output, Storage scale, int rows, int k) =>
        QuantizeOperand(fp8, P(x), ld, byRows: true, output, scale, rows, k, PtxKernels.EightBitPaddedK(k));

    internal bool EightBitReady(bool fp8)
    {
        string format = fp8 ? "e4m3" : "s8";
        return TensorKernel($"gemm8_{format}_f32") is not null && TensorKernel($"quant_rows_{format}") is not null;
    }

    // c = beta·c + f(op(a)·op(b)) on 8-bit tensor cores (addresses in bytes, strides in floats): op(a) is quantized per row
    // and op(b) per column into k-major bytes (padded to a multiple of 64 k), then the 8-bit product applies the scales.
    internal void Gemm8(bool fp8, ulong a, int lda, bool transA, ulong b, int ldb, bool transB, ulong c, int ldc, int m, int n, int k,
        float beta, ulong bias, GemmEpilogue epilogue, ulong aux)
    {
        int kp = PtxKernels.EightBitPaddedK(k);
        // op(a) [m, k]: stored [m][lda] (rows) or, transposed, [k][lda] (columns). op(b) needs [n][k]: stored
        // transposed [n][ldb] (rows) or as is, [k][ldb] (columns).
        var (a8, sa, ownA) = Quantized(fp8, a, lda, byRows: !transA, m, k, kp);
        var (b8, sb, ownB) = Quantized(fp8, b, ldb, byRows: transB, n, k, kp);
        try
        {
            if (_profile is not null)
            {
                _profileLabel = $"gemm8_{(fp8 ? "e4m3" : "s8")}_{(transA ? 't' : 'n')}{(transB ? 't' : 'n')}"
                                + $"{(epilogue == GemmEpilogue.None ? "" : epilogue == GemmEpilogue.Gelu ? "+gelu" : "+gelu'")}{(bias == 0 ? "" : "+bias")} {m}x{n}x{k}";
                _profileFlops = 2.0 * m * n * k;
            }

            t_sharedBytes = PtxKernels.EightBitShared;
            Launch(TensorKernel(PtxKernels.EightBitKernel(fp8, epilogue))!.Value, (uint)((n + PtxKernels.TensorTile - 1) / PtxKernels.TensorTile),
                (uint)((m + PtxKernels.TensorTile - 1) / PtxKernels.TensorTile), 1, PtxKernels.TensorThreads, 1,
                P(a8), P(b8), c, U(m), U(n), U(kp), F(beta), P(sa), P(sb), bias, U(kp), U(kp), U(ldc), aux);
            Interlocked.Increment(ref TensorCoreLaunches);
        }
        finally
        {
            if (ownA)
            {
                a8.Release();
                sa.Release();
            }

            if (ownB)
            {
                b8.Release();
                sb.Release();
            }
        }
    }

    public override int Float8PaddedK(int k) => EightBitReady(fp8: true) ? PtxKernels.EightBitPaddedK(k) : 0;

    public override bool Float8QuantizeWeight(Storage w, int k, int n, Storage values, Storage scales)
    {
        if (!EightBitReady(fp8: true))
        {
            return false;
        }

        // The columns of w [k][n] become rows of k bytes: exact maxima (not the delayed ones), once.
        var amax = Allocate(n, zeroed: true);
        try
        {
            Launch(TensorKernel("absmax_cols_e4m3")!.Value, (uint)((n + 255) / 256), (uint)((k + 63) / 64), 1, 256, 1, P(w), P(amax), U(n), U(k), U(n), U(64));
            QuantizeColumnsWithMaxima(true, P(w), n, values, scales, n, k, PtxKernels.EightBitPaddedK(k), amax, null);
        }
        finally
        {
            amax.Release();
        }

        return true;
    }

    public override bool Float8MatMul(Storage x, int m, int k, Storage values, Storage scales, int n, Storage y, float beta)
    {
        if (m == 0 || n == 0 || !EightBitReady(fp8: true))
        {
            return false;
        }

        int kp = PtxKernels.EightBitPaddedK(k);
        var (x8, sx, owned) = Quantized(true, P(x), k, byRows: true, m, k, kp);
        try
        {
            if (_profile is not null)
            {
                _profileLabel = $"gemm8_e4m3_nn_fp8w {m}x{n}x{k}";
                _profileFlops = 2.0 * m * n * k;
            }

            t_sharedBytes = PtxKernels.EightBitShared;
            Launch(TensorKernel(PtxKernels.EightBitKernel(true, GemmEpilogue.None))!.Value, (uint)((n + PtxKernels.TensorTile - 1) / PtxKernels.TensorTile),
                (uint)((m + PtxKernels.TensorTile - 1) / PtxKernels.TensorTile), 1, PtxKernels.TensorThreads, 1,
                P(x8), P(values), P(y), U(m), U(n), U(kp), F(beta), P(sx), P(scales), 0UL, U(kp), U(kp), U(n), 0UL);
            Interlocked.Increment(ref TensorCoreLaunches);
        }
        finally
        {
            if (owned)
            {
                x8.Release();
                sx.Release();
            }
        }

        return true;
    }

    // Quantized operands kept while a ReuseQuantizedOperands scope is open (on the thread that opened it).
    [ThreadStatic]
    private static Dictionary<(ulong Address, int Ld, bool ByRows, int Rows, int K, bool Fp8), (Storage Values, Storage Scales)>? t_reuse;

    // The operand quantized (or the kept copy): Owned = the caller releases it.
    private (Storage Values, Storage Scales, bool Owned) Quantized(bool fp8, ulong x, int ld, bool byRows, int rows, int k, int kp)
    {
        var key = (x, ld, byRows, rows, k, fp8);
        if (t_reuse is { } reuse && reuse.TryGetValue(key, out var kept))
        {
            return (kept.Values, kept.Scales, false);
        }

        var values = Allocate(Math.Max(1, rows * kp / 4), zeroed: false);
        var scales = Allocate(rows, zeroed: false);
        QuantizeOperand(fp8, x, ld, byRows, values, scales, rows, k, kp);
        if (t_reuse is { } open)
        {
            open[key] = (values, scales);
            return (values, scales, false);
        }

        return (values, scales, true);
    }

    public override IDisposable? ReuseQuantizedOperands()
    {
        if (t_reuse is not null || MixedPrecision.Current != MatMulPrecision.Float8)
        {
            return null;                                            // nested, or nothing is quantized
        }

        t_reuse = [];
        return new ReuseScope();
    }

    private sealed class ReuseScope : IDisposable
    {
        public void Dispose()
        {
            if (t_reuse is { } reuse)
            {
                t_reuse = null;
                foreach (var (values, scales) in reuse.Values)
                {
                    values.Release();
                    scales.Release();
                }
            }
        }
    }

    // out [rows][kp] bytes (k-major) + scale [rows] from x: byRows: x rows [rows][ld] of `k` values; else x [k][ld] whose
    // `rows` columns become the output rows.
    private void QuantizeOperand(bool fp8, ulong x, int ld, bool byRows, Storage output, Storage scale, int rows, int k, int kp)
    {
        string format = fp8 ? "e4m3" : "s8";
        if (byRows)
        {
            Launch(TensorKernel($"quant_rows_{format}")!.Value, (uint)rows, 1, 1, 256, 1, x, P(output), P(scale), U(ld), U(kp), U(rows), U(k));
            return;
        }

        const int Chunk = 64;
        bool delayed = fp8 && DelayedColumnScaling && _captureThread == 0;
        var key = (x, ld, rows, k);
        if (delayed)
        {
            Storage? kept;
            lock (_keptMaxima)
            {
                _keptMaxima.Remove(key, out kept);
            }

            if (kept is not null)
            {
                var record = Allocate(rows, zeroed: true);
                QuantizeColumnsDelayed(fp8, x, ld, output, scale, rows, k, kp, kept, record);
                kept.Release();
                Keep(key, record);
                return;
            }
        }

        var amax = Allocate(rows, zeroed: true);
        bool keepAmax = false;
        try
        {
            Launch(TensorKernel($"absmax_cols_{format}")!.Value, (uint)((rows + 255) / 256), (uint)((k + Chunk - 1) / Chunk), 1, 256, 1,
                x, P(amax), U(ld), U(k), U(rows), U(Chunk));
            QuantizeColumnsWithMaxima(fp8, x, ld, output, scale, rows, k, kp, amax, null);
            if (delayed)
            {
                Keep(key, amax);                                    // the first time: exact maxima, kept for the next
                keepAmax = true;
            }
        }
        finally
        {
            if (!keepAmax)
            {
                amax.Release();
            }
        }
    }

    private void Keep((ulong X, int Ld, int Rows, int K) key, Storage maxima)
    {
        lock (_keptMaxima)
        {
            if (_keptMaxima.Count >= 4096)
            {
                foreach (var old in _keptMaxima.Values)
                {
                    old.Release();
                }

                _keptMaxima.Clear();
            }

            if (_keptMaxima.Remove(key, out var previous))
            {
                previous.Release();
            }

            _keptMaxima[key] = maxima;
        }
    }

    // The column quantization with given maxima (one per output row: the absmax_cols result, or maxima kept from an
    // earlier pass for delayed scaling: with `record` set they are doubled as headroom, larger values saturating);
    // `record` (zeroed) receives this x's maxima.
    internal void QuantizeColumnsWithMaxima(bool fp8, ulong x, int ld, Storage output, Storage scale, int rows, int k, int kp, Storage maxima,
        Storage? record)
    {
        string format = fp8 ? "e4m3" : "s8";
        Launch(TensorKernel($"quant_cols_{format}")!.Value, (uint)((rows + 31) / 32), (uint)(kp / 32), 1, 32, 8,
            x, P(output), P(maxima), P(scale), U(ld), U(kp), U(k), U(rows), record is null ? 0UL : P(record), 0UL);
    }

    // Delayed scaling: quantized with twice the kept maxima (headroom: step-to-step growth does not saturate) while
    // recording x's own, then the correction pass quantizes again, with the recorded maxima, the 32-column strips where a
    // value more than doubled and saturated (a few blocks per strip, which return at once when none did). FP8's exponent
    // keeps the relative precision the same under the larger scale.
    internal void QuantizeColumnsDelayed(bool fp8, ulong x, int ld, Storage output, Storage scale, int rows, int k, int kp, Storage kept,
        Storage record)
    {
        QuantizeColumnsWithMaxima(fp8, x, ld, output, scale, rows, k, kp, kept, record);
        string format = fp8 ? "e4m3" : "s8";
        Launch(TensorKernel($"quant_cols_{format}")!.Value, (uint)((rows + 31) / 32), (uint)Math.Min(kp / 32, 4), 1, 32, 8,
            x, P(output), P(record), P(scale), U(ld), U(kp), U(k), U(rows), 0UL, P(kept));
    }

    /// <summary>
    /// FP8 training: column operands are quantized with the maxima the same operand (address and shape) had the last
    /// time, measured in the same pass, plus a correction pass for columns that grew (one read of x instead of two).
    /// Off by default; IDRAK_FP8_DELAYED=1 turns it on.
    /// </summary>
    internal static bool DelayedColumnScaling { get; set; } = Environment.GetEnvironmentVariable("IDRAK_FP8_DELAYED") is "1" or "true";

    // Kept column maxima per operand (address, leading dimension, columns, rows), bounded: cleared past 4096 entries.
    private readonly Dictionary<(ulong X, int Ld, int Rows, int K), Storage> _keptMaxima = [];

    // Benchmarks and tests: the column quantization of x [k][ld] (rows = its columns) by the two-launch pair (variant 0),
    // with given maxima, recording x's own (2), or that plus the correction pass (3: delayed scaling as training uses it;
    // record zeroed).
    internal void QuantizeColumnsVariant(bool fp8, Storage x, int ld, Storage output, Storage scale, int rows, int k, int variant,
        Storage? maxima = null, Storage? record = null)
    {
        if (variant == 2)
        {
            QuantizeColumnsWithMaxima(fp8, P(x), ld, output, scale, rows, k, PtxKernels.EightBitPaddedK(k), maxima!, record);
            return;
        }

        if (variant == 3)
        {
            QuantizeColumnsDelayed(fp8, P(x), ld, output, scale, rows, k, PtxKernels.EightBitPaddedK(k), maxima!, record!);
            return;
        }

        QuantizeOperand(fp8, P(x), ld, byRows: false, output, scale, rows, k, PtxKernels.EightBitPaddedK(k));
    }

    public override void SgdStep(Storage p, Storage g, Storage? v, int n, float lr, float momentum)
    {
        if (v is null)
        {
            Axpy(g, p, n, -lr);
        }
        else
        {
            Launch1D(_sgdMomentum, n, P(p), P(g), P(v), F(-lr), F(momentum), U(n));
        }
    }

    public override void AdamStep(Storage p, Storage g, Storage m, Storage v, int n, float lr, float beta1, float beta2, float eps) =>
        Launch1D(_adam, n, P(p), P(g), P(m), P(v), F(lr), F(beta1), F(beta2), F(1f - beta1), F(1f - beta2), F(eps), U(n));

    public override void AdamStep8Bit(Storage p, Storage g, Storage m, Storage v, Storage absMax, Storage map, int n, float lr, float beta1, float beta2, float eps,
        float gradientScale, float decay)
    {
        int blocks = (n + Optimizers.AdamW8Bit.BlockSize - 1) / Optimizers.AdamW8Bit.BlockSize;
        LaunchRows(K("adam8_f32"), blocks, P(p), P(g), P(m), P(v), P(absMax), P(map),
            F(lr), F(beta1), F(beta2), F(1f - beta1), F(1f - beta2), F(eps), U(n), F(gradientScale), F(decay), U(blocks));
    }

    // The device tables of FusedAdamW: chunks (tensor, first element) of 4096 elements, rebuilt when the sizes change; the
    // tensors' pointers and sizes, uploaded again only when a pointer changes; the norm / factor value.
    private sealed class FusedAdamState(CudaBackend backend) : IDisposable
    {
        public int[] Sizes = [];
        public ulong[] Pointers = [];
        public Storage? Chunks, Table, Factor;
        public int ChunkCount;

        public void Dispose()
        {
            foreach (var storage in new[] { Chunks, Table, Factor })
            {
                if (storage is not null)
                {
                    backend.Return(storage);
                }
            }

            (Chunks, Table, Factor) = (null, null, null);
        }
    }

    public override bool FusedAdamW(ReadOnlySpan<(Storage P, Storage G, Storage M, Storage V, int N)> tensors, ref IDisposable? cache, float maxNorm,
        float lr, float decay, float beta1, float beta2, float eps, bool zeroGradients)
    {
        if (tensors.Length == 0 || !_kernels.TryGetValue("multi_adamw_f32", out var adam) || !_kernels.TryGetValue("multi_sumsq_f32", out var sumSquares))
        {
            return false;
        }

        var state = cache as FusedAdamState;
        if (state is null)
        {
            cache?.Dispose();
            cache = state = new FusedAdamState(this);
        }

        bool resized = state.Sizes.Length != tensors.Length;
        for (int t = 0; t < tensors.Length && !resized; t++)
        {
            resized = state.Sizes[t] != tensors[t].N;
        }

        if (resized)
        {
            state.Sizes = new int[tensors.Length];
            for (int t = 0; t < tensors.Length; t++)
            {
                state.Sizes[t] = tensors[t].N;
            }

            var chunks = new List<uint>();
            for (int t = 0; t < tensors.Length; t++)
            {
                for (int start = 0; start < Math.Max(1, tensors[t].N); start += 4096)
                {
                    chunks.Add((uint)t);
                    chunks.Add((uint)start);
                }
            }

            if (state.Chunks is not null)
            {
                Return(state.Chunks);
            }

            state.ChunkCount = chunks.Count / 2;
            state.Chunks = Allocate(chunks.Count, zeroed: false);
            Upload(System.Runtime.InteropServices.MemoryMarshal.Cast<uint, float>(chunks.ToArray()), state.Chunks);
            state.Pointers = [];
        }

        var pointers = new ulong[tensors.Length * 5];
        for (int t = 0; t < tensors.Length; t++)
        {
            (pointers[5 * t], pointers[5 * t + 1], pointers[5 * t + 2], pointers[5 * t + 3], pointers[5 * t + 4]) =
                (P(tensors[t].P), P(tensors[t].G), P(tensors[t].M), P(tensors[t].V), (ulong)(uint)tensors[t].N);
        }

        if (!pointers.AsSpan().SequenceEqual(state.Pointers))
        {
            state.Pointers = pointers;
            state.Table ??= Allocate(tensors.Length * 10, zeroed: false);
            if (state.Table.Length != tensors.Length * 10)
            {
                Return(state.Table);
                state.Table = Allocate(tensors.Length * 10, zeroed: false);
            }

            Upload(System.Runtime.InteropServices.MemoryMarshal.Cast<ulong, float>(pointers), state.Table);
        }

        state.Factor ??= Allocate(1, zeroed: false);
        if (maxNorm > 0f)
        {
            Check(cuMemsetD32Async(P(state.Factor), 0, 1, _stream), nameof(cuMemsetD32Async));
            LaunchRows(sumSquares, state.ChunkCount, P(state.Chunks!), P(state.Table!), P(state.Factor), U(state.ChunkCount));
            ClipFactor(state.Factor, state.Factor, maxNorm);
        }
        else
        {
            Fill(state.Factor, 1, 1f);
        }

        LaunchRows(adam, state.ChunkCount, P(state.Chunks!), P(state.Table!), P(state.Factor), F(lr), F(decay), F(beta1), F(beta2), F(1f - beta1),
            F(1f - beta2), F(eps), U(zeroGradients ? 1 : 0), U(state.ChunkCount));
        return true;
    }

    public override void SumSquares(Storage x, Storage total, int n)
    {
        if (n > 0)
        {
            int blocks = Math.Min(1024, (n + PtxKernels.RowThreads - 1) / PtxKernels.RowThreads);
            LaunchRows(K("sumsq_f32"), blocks, P(x), P(total), U(n), U(blocks));
        }
    }

    public override void Dropout(Storage x, Storage y, int n, float p, uint seed) =>
        Launch1D(_dropout, n, P(x), P(y), F(p), F(1f / (1f - p)), seed, 0UL, U(n));

    public override void AddDropout(Storage residual, Storage x, Storage output, int n, float p, uint seed) =>
        Launch1D(K("add_dropout_f32"), n, P(residual), P(x), P(output), F(p), F(1f / (1f - p)), seed, U(n));

    public override void DropoutBackward(Storage dy, Storage dx, int n, float p, uint seed) =>
        Launch1D(_dropout, n, P(dy), P(dx), F(p), F(1f / (1f - p)), seed, 1UL, U(n));

    public override void Synchronize()
    {
        MakeCurrent();
        using var use = UseStream();
        Check(cuCtxSynchronize(), nameof(cuCtxSynchronize));
    }

    // One block of PtxKernels.RowThreads threads per row (PtxKernels.RowBlock kernels).
    private void LaunchRows(IntPtr function, int rows, params ReadOnlySpan<ulong> args) => LaunchRows(function, rows, PtxKernels.RowThreads, args);

    private void LaunchRows(IntPtr function, int rows, int threads, params ReadOnlySpan<ulong> args)
    {
        if (rows > 0)
        {
            Launch(function, (uint)rows, 1, 1, (uint)threads, 1, args);
        }
    }

    private void Launch1D(IntPtr function, int n, params ReadOnlySpan<ulong> args)
    {
        if (n > 0)
        {
            Launch(function, (uint)((n + PtxKernels.BlockSize - 1) / PtxKernels.BlockSize), 1, PtxKernels.BlockSize, 1, args);
        }
    }

    /// <summary>
    /// Launches a kernel. Every argument is widened to a 64-bit slot; the driver reads each
    /// parameter's actual size from the slot's start, which on little-endian hosts is the value itself.
    /// </summary>
    private void Launch(IntPtr function, uint gridX, uint gridY, uint blockX, uint blockY, params ReadOnlySpan<ulong> args) =>
        Launch(function, gridX, gridY, 1, blockX, blockY, args);

    private void Launch(IntPtr function, uint gridX, uint gridY, uint gridZ, uint blockX, uint blockY, params ReadOnlySpan<ulong> args)
    {
        if (_signatures.TryGetValue(function, out var signature) && signature.Parameters != args.Length)
        {
            throw new InvalidOperationException(
                $"Kernel {signature.Name} declares {signature.Parameters} parameters but was launched with {args.Length} arguments.");
        }

        uint shared = t_sharedBytes;
        t_sharedBytes = 0;
        MakeCurrent();
        ulong* values = stackalloc ulong[args.Length];
        void** pointers = stackalloc void*[args.Length];
        for (int i = 0; i < args.Length; i++)
        {
            values[i] = args[i];
            pointers[i] = &values[i];
        }

        using var use = UseStream();
        if (_profile is { } profile && _captureFree is null)
        {
            // Profiling (GpuProfiler): the kernel's GPU time between two events (host launch overhead excluded).
            if (_profileEvents.Start == IntPtr.Zero)
            {
                Check(cuEventCreate(out var start, 0), nameof(cuEventCreate));
                Check(cuEventCreate(out var end, 0), nameof(cuEventCreate));
                _profileEvents = (start, end);
            }

            Check(cuEventRecord(_profileEvents.Start, _stream), nameof(cuEventRecord));
            Check(cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, 1, shared, _stream, pointers, null), nameof(cuLaunchKernel));
            Check(cuEventRecord(_profileEvents.End, _stream), nameof(cuEventRecord));
            Check(cuEventSynchronize(_profileEvents.End), nameof(cuEventSynchronize));
            Check(cuEventElapsedTime(out float elapsed, _profileEvents.Start, _profileEvents.End), nameof(cuEventElapsedTime));
            long ticks = (long)(elapsed * (System.Diagnostics.Stopwatch.Frequency / 1000.0));
            string key = _profileLabel ?? (_signatures.TryGetValue(function, out var named) ? named.Name : $"0x{function:X}");
            lock (profile)
            {
                var entry = profile.GetValueOrDefault(key);
                profile[key] = (entry.Calls + 1, entry.Ticks + ticks, entry.Flops + _profileFlops);
            }

            _profileLabel = null;
            _profileFlops = 0;
            return;
        }

        Check(cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, 1, shared, _stream, pointers, null), nameof(cuLaunchKernel));
        if (DebugLaunches && _captureFree is null)
        {
            // Debugging aid: wait for every kernel so a fault is reported by the kernel that caused it.
            int result = cuStreamSynchronize(_stream);
            if (result != 0)
            {
                string name = _signatures.TryGetValue(function, out var failed) ? failed.Name : $"0x{function:X}";
                var shown = new List<string>();
                for (int i = 0; i < args.Length; i++)
                {
                    shown.Add($"0x{args[i]:X}");
                }

                throw new CudaException($"Kernel {name} failed (grid {gridX}×{gridY}×{gridZ}, block {blockX}×{blockY}; arguments {string.Join(", ", shown)}): "
                    + CudaDriver.Describe(result));
            }
        }
    }

    // IDRAK_CUDA_DEBUG=1: synchronize after every kernel launch and name the kernel that failed (slow).
    private static readonly bool DebugLaunches = Environment.GetEnvironmentVariable("IDRAK_CUDA_DEBUG") is "1" or "true";

    private static ulong P(Storage s) => ((CudaStorage)s).Pointer;

    private static ulong F(float value) => BitConverter.SingleToUInt32Bits(value);

    private static ulong U(int value) => (uint)value;
}
