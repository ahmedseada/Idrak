// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using Idrak.Abstraction.Devices.Cpu;
using Idrak.Abstraction.Operations;

namespace Idrak.Abstraction.Devices;

/// <summary>An element-wise function of one input (<see cref="Backend.Unary"/>, <see cref="Backend.UnaryBackward"/>).</summary>
public enum UnaryOp
{
    /// <summary>1 / (1 + e^-x).</summary>
    Sigmoid,

    /// <summary>The hyperbolic tangent.</summary>
    Tanh,

    /// <summary>max(x, 0).</summary>
    Relu,

    /// <summary>x².</summary>
    Square,

    /// <summary>|x|.</summary>
    Abs,

    /// <summary>e^x.</summary>
    Exp,

    /// <summary>The natural logarithm.</summary>
    Log,

    /// <summary>GELU, tanh approximation: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))).</summary>
    Gelu,

    /// <summary>√x.</summary>
    Sqrt,

    /// <summary>sin x.</summary>
    Sin,

    /// <summary>cos x.</summary>
    Cos,

    /// <summary>x · sigmoid(x).</summary>
    Silu,

    /// <summary>-1, 0 or 1 (its gradient is zero).</summary>
    Sign,
}

/// <summary>An element-wise function of two inputs (<see cref="Backend.Binary"/>, <see cref="Backend.ExtremumBackward"/>).</summary>
public enum BinaryOp
{
    /// <summary>a + b.</summary>
    Add,

    /// <summary>a - b.</summary>
    Sub,

    /// <summary>a · b.</summary>
    Mul,

    /// <summary>max(a, b).</summary>
    Maximum,

    /// <summary>min(a, b).</summary>
    Minimum,
}

/// <summary>
/// Shape parameters of a 2-D convolution or pooling window over NCHW data (<see cref="Backend.Im2Col"/>,
/// <see cref="Backend.MaxPool"/>).
/// </summary>
/// <param name="N">Images in the batch.</param>
/// <param name="C">Channels.</param>
/// <param name="H">Input height.</param>
/// <param name="W">Input width.</param>
/// <param name="KH">Window height.</param>
/// <param name="KW">Window width.</param>
/// <param name="SH">Vertical stride.</param>
/// <param name="SW">Horizontal stride.</param>
/// <param name="PH">Zero padding above and below.</param>
/// <param name="PW">Zero padding left and right.</param>
public readonly record struct ConvGeometry(
    int N, int C, int H, int W, int KH, int KW, int SH, int SW, int PH, int PW)
{
    /// <summary>Output height: (H + 2·PH - KH) / SH + 1.</summary>
    public int OH => (H + 2 * PH - KH) / SH + 1;

    /// <summary>Output width: (W + 2·PW - KW) / SW + 1.</summary>
    public int OW => (W + 2 * PW - KW) / SW + 1;

    /// <summary>Columns of the unfolded matrix: C * KH * KW.</summary>
    public int PatchSize => C * KH * KW;

    /// <summary>Rows of the unfolded matrix: N * OH * OW.</summary>
    public int Positions => N * OH * OW;
}

/// <summary>
/// What the attention kernels do beyond plain causal attention: a sliding window (a query at position p sees keys at
/// positions c with p - <see cref="Window"/> &lt; c ≤ p, as transformers masks them; 0 for every earlier position) and a
/// soft-cap on the scaled scores, cap · tanh(score / cap) before the softmax (0 for none). The default is plain causal
/// attention, which every kernel computes exactly as before these were added.
/// </summary>
/// <param name="Window">Positions a query sees, its own included (0: every earlier position).</param>
/// <param name="Softcap">The cap on the scaled scores (0: none).</param>
public readonly record struct AttentionVariant(int Window, float Softcap)
{
    /// <summary>Plain causal attention: no window and no cap.</summary>
    public bool IsPlain => Window <= 0 && Softcap <= 0f;

    /// <summary>The first position a query whose causal limit is <paramref name="end"/> (exclusive) sees.</summary>
    public int Start(int end) => Window > 0 ? Math.Max(0, end - Window) : 0;

    /// <summary>A scaled score, soft-capped when a cap is set.</summary>
    public float Cap(float score) => Softcap > 0f ? MathF.Tanh(score / Softcap) * Softcap : score;

    /// <summary>The derivative of <see cref="Cap"/> given its result: 1 - (capped / cap)² (1 without a cap).</summary>
    public float Slope(float capped) => Softcap > 0f ? 1f - capped / Softcap * (capped / Softcap) : 1f;
}

/// <summary>
/// A reference-counted block of device memory holding <see cref="Length"/> floats: what a device's kernels read and write.
/// A device subclasses it to hold its own handle (a pointer, a buffer and offset) and creates it in
/// <see cref="Backend.Allocate"/>; when the last reference is released the block goes back to its device
/// (<see cref="Backend.Return"/>).
/// </summary>
/// <param name="backend">The device the memory belongs to.</param>
/// <param name="length">The number of floats.</param>
public abstract class Storage(Backend backend, int length)
{
    private int _refs = 1;
    private int _version;

    /// <summary>The device the memory belongs to.</summary>
    public Backend Backend { get; } = backend;

    /// <summary>The number of floats.</summary>
    public int Length { get; } = length;

    /// <summary>Adds a reference: the block stays until <see cref="Release"/> has been called once more than this.</summary>
    public void AddRef() => Interlocked.Increment(ref _refs);

    /// <summary>Drops a reference; the last one gives the memory back to the device.</summary>
    public void Release()
    {
        if (Interlocked.Decrement(ref _refs) == 0)
        {
            if (!Evicted)
            {
                Backend.Return(this);
            }

            Released?.Invoke();
            Released = null;
            Packed = null;
        }
    }

    /// <summary>
    /// Counts the public in-place writes into the storage (<c>Tensor.CopyFrom(ReadOnlySpan&lt;float&gt;)</c> and the
    /// others): a backward step whose operation read the storage before such a write refuses to run on the changed values.
    /// </summary>
    public int Version => Volatile.Read(ref _version);

    // Counts one in-place write (see Version).
    internal void Written() => Interlocked.Increment(ref _version);

    /// <summary>How readily the storage moves to system memory when the device fills up (see <see cref="IMemoryOffload"/>).</summary>
    public OffloadPriority OffloadPriority { get; set; }

    /// <summary>Moved to system memory on request, to stay there (<see cref="IMemoryOffload.Rebalance"/> leaves it).</summary>
    public bool KeepOnHost { get; set; }

    /// <summary>Whether the memory was given back by <see cref="Backend.Evict"/> (the values are recomputed on <see cref="Backend.Restore"/>).</summary>
    public bool Evicted { get; internal set; }

    /// <summary>Recomputes the values into this storage after <see cref="Backend.Restore"/> gave it memory again.</summary>
    public Action<Storage>? Recompute { get; internal set; }

    /// <summary>The tensor whose values are recomputed (its own backward step does not read them), or null.</summary>
    public object? RecomputedFor { get; internal set; }

    /// <summary>Called once when the storage is released for good (for what its recomputation keeps, such as a packed copy).</summary>
    public Action? Released { get; internal set; }

    /// <summary>
    /// A bfloat16 copy of the values (<see cref="Backend.PackBFloat16"/> words) kept while the storage is evicted, which
    /// kernels that read bfloat16 inputs use without unpacking it; null when there is none.
    /// </summary>
    public Storage? Packed { get; internal set; }

    /// <summary>Whether a tensor still holds the storage (it has not been released for good).</summary>
    public bool Alive => Volatile.Read(ref _refs) > 0;

    /// <summary>
    /// The values, for a storage in memory the process reads directly (the CPU device's): the first <see cref="Length"/>
    /// floats. Other devices copy with <see cref="Backend.Download"/> and <see cref="Backend.Upload"/>.
    /// </summary>
    /// <exception cref="NotSupportedException">The storage is in device memory.</exception>
    public virtual Span<float> HostMemory => throw new NotSupportedException($"A {Backend.Kind} storage is in device memory; copy it with Download and Upload.");
}

/// <summary>
/// A device: its memory, copies and synchronization, what its kernels can do, and a kernel for each operation of the
/// device contract (plan 9). Every operation is a virtual method <c>NameKernel</c> whose default body is the host
/// fallback (the CPU runs it on copies of the operands), a composition of other operations, or "no such kernel" (it
/// returns false and the caller takes another path); a device overrides the ones it runs itself, the most used first.
/// Callers run an operation through <c>Name(...)</c> (<see cref="Fill"/>, <see cref="Softmax"/>, ...), which runs the
/// kernel registered for the device's <see cref="Kind"/> in <see cref="Kernels"/> where there is one, else
/// <c>NameKernel</c>. A device comes with a <see cref="DeviceProvider"/>, which starts it.
/// </summary>
/// <remarks>
/// Every operation works on contiguous row-major float32 buffers. Methods named "...Backward", <see cref="Axpy"/>,
/// <see cref="MulAdd"/>, <see cref="SumRows"/> and <see cref="AddBroadcastScalar"/> accumulate into their output (+=) so
/// gradients from several paths add up.
/// </remarks>
[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicMethods)]   // Kernels.Chain sees which kernels a device overrides
public abstract partial class Backend
{
    // The kernels registered for this device by operation index, null when there are none and nothing is traced (every
    // call then runs the device's own kernel after one field read), or an empty array when they must be resolved again.
    private Delegate?[]? _kernels = Operations.Kernels.Unresolved();

    // The trace counting this device's calls (Kernels.Trace), or null.
    private KernelTrace? _trace;

    // Operations this device has run through the host fallback (HostCall), read through Kernels.HostCalls.
    private long _hostCalls;

    // RetryOnHost: 1 on, 0 off, -1 not read yet from IDRAK_RETRY_ON_HOST (read on first use, when Kind is safe to call).
    private int _retryOnHost = -1;

    // Which operations this device has a kernel of its own for (by operation index), read on the first retry.
    private bool[]? _ownKernels;

    /// <summary>Starts a device; <see cref="Kernels"/> tells it when the kernels registered for it change.</summary>
    protected Backend() => Operations.Kernels.Track(this);

    /// <summary>The kind of device this drives ("cpu", "cuda", "vulkan", …): the kernels registered for it run here.</summary>
    public abstract string Kind { get; }

    /// <summary>Marks the registered kernels as changed: the next call to an operation resolves them again.</summary>
    internal void KernelsChanged()
    {
        Volatile.Write(ref _kernels, Operations.Kernels.Unresolved());
        PluginKernelsChanged();
    }

    /// <summary>The trace counting this device's calls, or null; setting it makes every call resolve its kernel again.</summary>
    internal KernelTrace? Trace
    {
        get => Volatile.Read(ref _trace);
        set
        {
            Volatile.Write(ref _trace, value);
            KernelsChanged();
        }
    }

    /// <summary>
    /// Whether an operation whose kernel on this device fails (throws a <see cref="DeviceException"/>) runs again on the
    /// CPU, through the host fallback, instead of throwing. Off by default (plan 10, decision 13: a silent retry would hide
    /// driver bugs and change speed by orders of magnitude); its first value comes from the environment variable
    /// <c>IDRAK_RETRY_ON_HOST</c>: <c>1</c> or <c>all</c> turns it on for every device but the CPU, a list of kinds
    /// (<c>cuda,vulkan</c>) for those kinds.
    /// </summary>
    /// <remarks>
    /// Each retry is reported to telemetry (<see cref="DeviceFailed"/>, <see cref="DeviceFailed.RetriedOnHost"/>) and
    /// counted as a host call (<see cref="Operations.Kernels.HostCalls"/>). Only operations whose default is the host
    /// fallback are retried: a composed operation retries its parts one by one, an operation with no fallback throws, and
    /// so does a kernel registered with <see cref="Operations.Kernels.Register"/>; an operation the device has no kernel of
    /// its own for runs its host fallback once, as without the retry. An error the GPU reports after the operation
    /// returned (found at a later synchronize or copy) can't be retried. While it is on, calls leave the inlined fast
    /// path, as under a trace.
    /// <para>
    /// A retry runs the whole operation again on the host. An operation that adds into its output (a gradient
    /// accumulation, a <c>+=</c> product: its kernel reads the output as well as writing it) adds twice to whatever part the
    /// failed kernel had already written. The library's GPU kernels fail when they are launched (before writing anything)
    /// or report the error at a later synchronization (which is not retried), so this needs a kernel that fails half-way,
    /// such as a plug-in's; such a kernel should leave its output untouched when it throws.
    /// </para>
    /// </remarks>
    public bool RetryOnHost
    {
        get
        {
            int retry = Volatile.Read(ref _retryOnHost);
            if (retry < 0)
            {
                retry = RetryOnHostFrom(Environment.GetEnvironmentVariable("IDRAK_RETRY_ON_HOST"), Kind) ? 1 : 0;
                retry = Interlocked.CompareExchange(ref _retryOnHost, retry, -1) is var before and >= 0 ? before : retry;
            }

            return retry == 1;
        }

        set
        {
            Volatile.Write(ref _retryOnHost, value ? 1 : 0);
            KernelsChanged();
        }
    }

    // Whether this device has a kernel of its own for the operation: without one its NameKernel is the host fallback
    // already, which a retry would only run twice.
    private bool OwnsKernel(int operation) => (_ownKernels ??= Operations.Kernels.OwnKernels(this))[operation];

    /// <summary>
    /// Whether <c>IDRAK_RETRY_ON_HOST</c> set to <paramref name="setting"/> turns the retry on for a device of
    /// <paramref name="kind"/>: <c>1</c>, <c>true</c> or <c>all</c> for every kind but "cpu", or a comma list of kinds
    /// (any case); not set, empty, <c>0</c> or <c>false</c>: off.
    /// </summary>
    internal static bool RetryOnHostFrom(string? setting, string kind)
    {
        if (string.IsNullOrWhiteSpace(setting))
        {
            return false;
        }

        foreach (string item in setting.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            if (item is "1" || item.Equals("all", StringComparison.OrdinalIgnoreCase) || item.Equals("true", StringComparison.OrdinalIgnoreCase))
            {
                return !kind.Equals("cpu", StringComparison.OrdinalIgnoreCase);
            }

            if (item.Equals(kind, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }

    // The dispatcher's retry of one call (Backend.NameRegistered, for an operation whose default is the host fallback):
    // while the device's kernel runs, a DeviceException does not report itself; Failed reports it once, with the
    // operation, before the call runs again on the host (NameOnHost), where errors report themselves again. A mutable
    // struct, so a plain local (not a `using` one, which would be read-only).
    private struct HostRetry
    {
        private readonly Operation _operation;
        private bool _ended;

        public HostRetry(Operation operation)
        {
            _operation = operation;
            DeviceException.EnterRetry();
        }

        // The device's kernel threw: ends the quiet part and reports the retry.
        public void Failed(DeviceException failure)
        {
            End();
            if (Telemetry.IsEnabled(TelemetryLevel.Devices))
            {
                Telemetry.DeviceFailed(new DeviceFailed(failure.Device, _operation.Name, failure.Message, RetriedOnHost: true, Hint: null));
            }
        }

        public void End()
        {
            if (!_ended)
            {
                _ended = true;
                DeviceException.LeaveRetry();
            }
        }
    }

    /// <summary>Operations run through the host fallback so far.</summary>
    internal long HostCalls => Interlocked.Read(ref _hostCalls);

    /// <summary>Counts one host fallback of the operation named <paramref name="operation"/>.</summary>
    internal void HostCalled(string operation)
    {
        Interlocked.Increment(ref _hostCalls);
        Volatile.Read(ref _trace)?.HostCalled(operation);
    }

    // The kernel registered for an operation on this device, or null for the device's own (off the fast path: counted
    // when traced).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private Delegate? Kernel(int operation)
    {
        Volatile.Read(ref _trace)?.Called(operation);
        var kernels = _kernels;
        return kernels is null ? null : kernels.Length != 0 ? kernels[operation] : ResolveKernels(kernels)[operation];
    }

    // Resolves the slots; stores them unless Kernels changed meanwhile (then the next call resolves again). With nothing
    // registered, no trace and no retry on the host the slots are dropped, so calls take the fast path again.
    private Delegate?[] ResolveKernels(Delegate?[] unresolved)
    {
        var slots = Operations.Kernels.Resolve(this);
        bool direct = Array.TrueForAll(slots, k => k is null) && Volatile.Read(ref _trace) is null && !RetryOnHost;
        Interlocked.CompareExchange(ref _kernels, direct ? null : slots, unresolved);
        return slots;
    }

    /// <summary>
    /// A block of <paramref name="length"/> floats of device memory (from the device's pool where it keeps one); zeroed
    /// when <paramref name="zeroed"/>, else holding whatever was there.
    /// </summary>
    public abstract Storage Allocate(int length, bool zeroed);

    /// <summary>What this device's kernels can do (see <see cref="BackendCapabilities"/>).</summary>
    public abstract BackendCapabilities Capabilities { get; }

    /// <summary>A human-readable name of the device, such as the GPU model.</summary>
    public abstract string Name { get; }

    /// <summary>What the device reports about its hardware, for listings and diagnostics; empty when it reports nothing.</summary>
    public virtual BackendHardware Hardware => new();

    /// <summary>Null when this device runs bfloat16 products on matrix units (tensor cores); otherwise why not.</summary>
    public virtual string? TensorCoresUnavailable() => "the device computes matrix products in float32";

    /// <summary>Starts timing every kernel (where <see cref="BackendCapabilities.Profiling"/>).</summary>
    public virtual void StartProfile()
    {
    }

    /// <summary>Stops timing; the kernels by name with their calls, time (stopwatch ticks) and floating-point operations.</summary>
    public virtual Dictionary<string, (long Calls, long Ticks, double Flops)> StopProfile() => [];

    /// <summary>Keeping tensors in system memory when the device is full, or null when the device has no such support.</summary>
    public virtual IMemoryOffload? Offload => null;

    /// <summary>Pinned buffers for background copies to and from system memory, or null when the device has none (see <see cref="IHostStaging"/>).</summary>
    public virtual IHostStaging? CreateHostStaging(int slots, int slotFloats) => null;

    /// <summary>
    /// Takes back the memory of a storage whose last reference was released (<see cref="Storage.Release"/>), or that
    /// <see cref="Evict"/> empties, to its pool or to the driver.
    /// </summary>
    public abstract void Return(Storage storage);

    /// <summary>
    /// Gives the storage's memory back to the pool while the storage (shared by every view of it) stays: kernels given it
    /// fail until <see cref="Restore"/> gives it memory again and <paramref name="recompute"/> refills it (null: the values
    /// are gone for good).
    /// </summary>
    public void Evict(Storage storage, Action<Storage>? recompute)
    {
        if (storage.Evicted)
        {
            return;
        }

        storage.Recompute = recompute;
        Return(storage);
        Detach(storage);
        storage.Evicted = true;
    }

    /// <summary>Gives an evicted storage memory again and recomputes its values; false when it was not evicted or cannot be recomputed.</summary>
    public bool Restore(Storage storage)
    {
        if (!storage.Evicted || storage.Recompute is null)
        {
            return false;
        }

        Attach(storage, Allocate(storage.Length, zeroed: false));
        storage.Evicted = false;
        storage.Recompute!(storage);
        return true;
    }

    /// <summary>
    /// Points <paramref name="storage"/> at no memory, after <see cref="Evict"/> gave its memory back with
    /// <see cref="Return"/>; kernels given it fail until <see cref="Attach"/>.
    /// </summary>
    protected abstract void Detach(Storage storage);

    /// <summary>
    /// Points <paramref name="storage"/> (evicted) at the memory of <paramref name="fresh"/>, a block just allocated for
    /// it, which is not used on its own afterwards.
    /// </summary>
    protected abstract void Attach(Storage storage, Storage fresh);

    /// <summary>The bytes in use, cached and offloaded, and the limit (<see cref="ComputeResources.GetMemoryUsage"/>).</summary>
    public abstract MemoryUsage GetMemoryUsage();

    /// <summary>Frees pooled blocks that no tensor is using.</summary>
    public abstract void ReleaseCachedMemory();

    /// <summary>Copies <paramref name="source"/> into the first source.Length floats of <paramref name="destination"/>.</summary>
    public abstract void Upload(ReadOnlySpan<float> source, Storage destination);

    /// <summary>Copies the first destination.Length floats of <paramref name="source"/> into <paramref name="destination"/>.</summary>
    public abstract void Download(Storage source, Span<float> destination);

    /// <summary>Copies destination.Length floats starting at element <paramref name="offset"/>.</summary>
    public abstract void DownloadRange(Storage source, int offset, Span<float> destination);

    /// <summary>y[i] = value for i &lt; n.</summary>
    public virtual void FillKernel(Storage y, int n, float value)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.Fill(h[y], n, value);
    }

    /// <summary>y[i] = x[i] for i &lt; n (a copy within the device).</summary>
    public virtual void Copy(Storage x, Storage y, int n)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.Copy(h[x], h[y], n);
    }

    /// <summary>y = op(x).</summary>
    public virtual void UnaryKernel(UnaryOp op, Storage x, Storage y, int n)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.Unary(op, h[x], h[y], n);
    }

    /// <summary>dx += dy * op'(x), where y = op(x) is passed in for ops whose derivative is cheaper from y.</summary>
    public virtual void UnaryBackwardKernel(UnaryOp op, Storage x, Storage y, Storage dy, Storage dx, int n)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.UnaryBackward(op, h[x], h[y], h[dy], h[dx], n);
    }

    /// <summary>c = a op b, element-wise.</summary>
    public virtual void BinaryKernel(BinaryOp op, Storage a, Storage b, Storage c, int n)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.Binary(op, h[a], h[b], h[c], n);
    }

    /// <summary>
    /// dx += dy where a wins (<see cref="BinaryOp.Maximum"/>: a ≥ b; <see cref="BinaryOp.Minimum"/>: a ≤ b), db += dy
    /// elsewhere: the backward step of c = max(a, b) or min(a, b). A null gradient is skipped.
    /// </summary>
    public virtual void ExtremumBackwardKernel(BinaryOp op, Storage a, Storage b, Storage dy, Storage? da, Storage? db, int n)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.ExtremumBackward(op, h[a], h[b], h[dy], h.Maybe(da), h.Maybe(db), n);
    }

    /// <summary>y = x^exponent, element-wise.</summary>
    public virtual void PowKernel(Storage x, Storage y, int n, float exponent)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.Pow(h[x], h[y], n, exponent);
    }

    /// <summary>dx += dy · exponent · x^(exponent - 1).</summary>
    public virtual void PowBackwardKernel(Storage x, Storage dy, Storage dx, int n, float exponent)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.PowBackward(h[x], h[dy], h[dx], n, exponent);
    }

    /// <summary>y = x limited to [min, max] (either may be infinite).</summary>
    public virtual void ClampKernel(Storage x, Storage y, int n, float min, float max)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.Clamp(h[x], h[y], n, min, max);
    }

    /// <summary>dx += dy where min ≤ x ≤ max.</summary>
    public virtual void ClampBackwardKernel(Storage x, Storage dy, Storage dx, int n, float min, float max)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.ClampBackward(h[x], h[dy], h[dx], n, min, max);
    }

    /// <summary>y = condition ≠ 0 ? a : b, element-wise.</summary>
    public virtual void WhereKernel(Storage condition, Storage a, Storage b, Storage y, int n)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.Where(h[condition], h[a], h[b], h[y], n);
    }

    /// <summary>da += dy where condition ≠ 0, db += dy elsewhere (a null gradient is skipped).</summary>
    public virtual void WhereBackwardKernel(Storage condition, Storage dy, Storage? da, Storage? db, int n)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.WhereBackward(h[condition], h[dy], h.Maybe(da), h.Maybe(db), n);
    }

    /// <summary>y = alpha * x + beta.</summary>
    public virtual void AffineKernel(Storage x, Storage y, int n, float alpha, float beta)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.Affine(h[x], h[y], n, alpha, beta);
    }

    /// <summary>y += alpha * x.</summary>
    public virtual void AxpyKernel(Storage x, Storage y, int n, float alpha)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.Axpy(h[x], h[y], n, alpha);
    }

    /// <summary>c += a * b, element-wise.</summary>
    public virtual void MulAddKernel(Storage a, Storage b, Storage c, int n)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.MulAdd(h[a], h[b], h[c], n);
    }

    /// <summary>c[r, j] = a[r, j] + v[j].</summary>
    public virtual void AddRowVectorKernel(Storage a, Storage v, Storage c, int rows, int cols)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.AddRowVector(h[a], h[v], h[c], rows, cols);
    }

    /// <summary>y[j] += sum over r of x[r, j].</summary>
    public virtual void SumRowsKernel(Storage x, Storage y, int rows, int cols)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.SumRows(h[x], h[y], rows, cols);
    }

    /// <summary>result[0] = scale * sum(x).</summary>
    public virtual void SumKernel(Storage x, Storage result, int n, float scale)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.Sum(h[x], h[result], n, scale);
    }

    /// <summary>y[offset] += alpha * x[0] (used to accumulate scalar losses without leaving the device).</summary>
    public virtual void AxpyAtKernel(Storage x, Storage y, int offset, float alpha)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.AxpyAt(h[x], h[y], offset, alpha);
    }

    /// <summary>y[i] += scale * s[0].</summary>
    public virtual void AddBroadcastScalarKernel(Storage s, Storage y, int n, float scale)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.AddBroadcastScalar(h[s], h[y], n, scale);
    }

    /// <summary>
    /// c[m, n] = op(a) * op(b) + beta * c, where op(a) is [m, k] and op(b) is [k, n].
    /// With <paramref name="transA"/> a is stored as [k, m]; with <paramref name="transB"/> b is stored as [n, k].
    /// </summary>
    public void MatMul(Storage a, Storage b, Storage c, int m, int n, int k, bool transA, bool transB, float beta) =>
        BatchedMatMul(a, b, c, 1, m, n, k, transA, transB, beta);

    /// <summary>
    /// y = x · w for many rows (prompts) with packed weights w (in <paramref name="format"/>), expanding w as it is read instead of into a float copy. Returns false when the
    /// device has no such kernel or the shape is too small for it (callers then expand w first).
    /// </summary>
    public virtual bool PackedMatMulLargeKernel(PackedFormat format, Storage x, Storage packed, Storage? scales, Storage y, int m, int n, int k) => false;

    /// <summary>
    /// True when <see cref="PackedMatMulLarge"/> is faster than the few-rows kernels for this shape although the rows are
    /// few enough for them (callers still fall back to those kernels when the packed product returns false). Given the
    /// real operands, so a device can measure both on itself; it may write <paramref name="y"/>, which the caller's product
    /// then writes again.
    /// </summary>
    public virtual bool PrefersPackedMatMul(PackedFormat format, Storage x, Storage packed, Storage? scales, Storage y, int m, int n, int k) => false;

    /// <summary>
    /// y = (act(gate) · up) · w for few rows with packed weights w (in <paramref name="format"/>; activation 0 = SiLU, 1 = GELU tanh): the gated feed-forward's down projection
    /// without a separate activation pass. Returns false when the device has no fused version.
    /// </summary>
    public virtual bool PackedMatMulGatedKernel(PackedFormat format, int activation, Storage gate, Storage up, Storage packed, Storage? scales, Storage y,
        int m, int n, int k) => false;

    /// <summary>
    /// y = x · w for few rows with packed weights w (in <paramref name="format"/>), then
    /// sum = residual + y and normalized = its RMS normalization · (gain + offset) per row (a residual addition and the
    /// next normalization) in the same pass. Returns false when the device has no fused version.
    /// </summary>
    public virtual bool PackedMatMulAddRmsNormKernel(PackedFormat format, Storage x, Storage packed, Storage? scales, Storage y, int m, int n, int k,
        Storage residual, Storage sum, Storage gain, Storage normalized, float eps, float offset) => false;

    /// <summary>
    /// Several few-row products of packed weights sharing one input x [m, k]: y_j = x · w_j (+ bias_j), in <paramref name="format"/>
    /// (int8 as in <see cref="Int8MatMul"/>, 4-bit as in <see cref="Int4MatMul"/>, bfloat16 as in
    /// <see cref="BFloat16MatMul"/>, which has no scales). Returns false when the device has no single-pass version (callers then
    /// run the products one by one).
    /// </summary>
    public virtual bool PackedMatMulManyKernel(PackedFormat format, Storage x, int m, int k,
        ReadOnlySpan<(Storage Packed, Storage? Scales, Storage? Bias, Storage Output, int Columns)> products) => false;

    /// <summary>
    /// <see cref="PackedMatMulMany"/> for a feed-forward block's gate and up projections (two products of equal widths),
    /// also writing hidden = act(gate) · up (activation 0 = SiLU, 1 = GELU tanh, 2 = ReLU) in the same pass. Returns
    /// false when the device has no fused version.
    /// </summary>
    public virtual bool PackedMatMulGatedPairKernel(PackedFormat format, int activation, Storage x, int m, int k,
        ReadOnlySpan<(Storage Packed, Storage? Scales, Storage? Bias, Storage Output, int Columns)> products, Storage hidden) => false;

    /// <summary>
    /// Several products sharing one input: y_j = a · w_j (+ bias_j) for a [m, k] and w_j [k, n_j] (the query, key and
    /// value projections, say). The default computes them one by one; devices may do them in one pass.
    /// </summary>
    public virtual void MatMulManyKernel(Storage a, int m, int k, ReadOnlySpan<(Storage Weight, Storage? Bias, Storage Output, int Columns)> products)
    {
        foreach (var (weight, bias, output, columns) in products)
        {
            MatMul(a, weight, output, m, columns, k, false, false, 0f);
            if (bias is not null)
            {
                AddRowVector(output, bias, output, m, columns);
            }
        }
    }

    /// <summary>
    /// c = a·b + bias (bias [n] added to every row) in one pass, for a [m, k] and b [k, n] as stored. Returns false when
    /// the device has no such pass for these sizes (callers then multiply and add the bias separately).
    /// </summary>
    public virtual bool MatMulBiasKernel(Storage a, Storage b, Storage bias, Storage c, int m, int n, int k) => false;

    /// <summary>
    /// c = beta·c + a·op(b) + u·op(v) in one product (a LoRA adapter's term as one more k step): a [m, k], b [k, n]
    /// (transposed: [n, k]), u [m, rank], v [rank, n] (with <paramref name="transB"/>: [n, rank]), rank ≤ 32, all rows
    /// contiguous. Returns false when the device has no such pass (callers then compute the two products separately).
    /// </summary>
    public virtual bool MatMulLowRankKernel(Storage a, Storage b, Storage c, int m, int n, int k, bool transB, float beta, Storage u, Storage v, int rank) => false;

    /// <summary>
    /// c = beta·c + a · Wᵀ (+ u · vᵀ) with W a bfloat16 weight [n, k] as <c>BFloat16Weight</c> packs it (two
    /// values per word along k), read as stored: the input gradient through a frozen bfloat16 layer (and its adapter's
    /// dt · Aᵀ with u = dt [m, rank], v = A [n, rank]) without expanding the weight to float. u null: no low-rank term.
    /// False when the device has no such kernel (callers then expand the weight).
    /// </summary>
    public virtual bool BFloat16TransposedMatMulKernel(Storage a, Storage packed, Storage c, int m, int n, int k, float beta, Storage? u, Storage? v, int rank) => false;

    /// <summary>
    /// k rounded up for <see cref="Float8QuantizeWeight"/>'s values (0 when the device has no FP8 products): the values
    /// take n · paddedK bytes, n · paddedK / 4 floats of storage. The library's devices round up to a multiple of 64, the
    /// padding the conformance kit's cases lay the values out with.
    /// </summary>
    public virtual int Float8PaddedK(int k) => 0;

    /// <summary>
    /// Quantizes a frozen weight w [k, n] (float32, finite) once for <see cref="Float8MatMul"/>: one scale per column,
    /// scales[j] = a / 448 for a = max over r of |w[r, j]| (1 when a is 0), and the FP8 (e4m3) code of w[r, j] · (1 /
    /// scales[j]) (the division, the reciprocal and the product each a float32 operation rounded to nearest), rounded to
    /// nearest with ties to even and saturated to ±448, the sign kept (-0 included). e4m3: a sign, 4 exponent bits with
    /// bias 7 and 3 mantissa bits; no infinities, 0x7F and 0xFF are NaN, 0x7E is 448, codes 1-7 are the subnormals
    /// m · 2⁻⁹. The codes are bytes k-major per column: byte j · paddedK + r (paddedK = <see cref="Float8PaddedK"/>(k),
    /// four bytes per float of <paramref name="values"/>, the first in the low byte), 0 for r ≥ k. Fixed to the bit:
    /// the conformance kit compares the bytes exactly. Returns false when unsupported.
    /// </summary>
    public virtual bool Float8QuantizeWeightKernel(Storage w, int k, int n, Storage values, Storage scales) => false;

    /// <summary>
    /// y = beta·y + x · w on FP8 matrix units for x [m, k] and a weight quantized by <see cref="Float8QuantizeWeight"/>
    /// (its values and scales, paddedK = <see cref="Float8PaddedK"/>(k)): each row of x is quantized as it is read the way
    /// <see cref="Float8QuantizeWeight"/> quantizes a column (sx[i] = max over p of |x[i, p]| / 448, or 1; the e4m3 codes
    /// of x[i, p] · (1 / sx[i])), then y[i, j] = beta·y[i, j] + sx[i] · scales[j] · Σ_p x8[i, p] · w8[j, p], the
    /// products of codes summed in float32 (y is not read when beta is 0). Returns false when unsupported.
    /// </summary>
    public virtual bool Float8MatMulKernel(Storage x, int m, int k, Storage values, Storage scales, int n, Storage y, float beta) => false;

    /// <summary>
    /// Products of one input x [m, k] through 1-3 packed layers (in <paramref name="format"/>),
    /// each with a low-rank term: y_j = x · w_j + u_j · v_j for u_j [m, rank] and v_j [rank, n_j], rank ≤ 32, in one pass.
    /// Returns false when the device has no such pass.
    /// </summary>
    public virtual bool PackedMatMulLowRankKernel(PackedFormat format, Storage x, int m, int k,
        ReadOnlySpan<(Storage Packed, Storage? Scales, Storage Output, int Columns, Storage U, Storage V)> products, int rank) => false;

    /// <summary>
    /// c = beta·c + op(a)·op(b) on bfloat16 tensor cores, with row strides: a [m, k] stored with <paramref name="lda"/>
    /// floats per row (transposed: [k, m] rows), b [k, n] with <paramref name="ldb"/> (transposed: [n, k] rows), c [m, n]
    /// with <paramref name="ldc"/>; offsets in elements. <paramref name="bias"/> [n] is added to every row (modes None and
    /// Gelu). <see cref="GemmEpilogue.Gelu"/> writes gelu(product + bias) and, when <paramref name="aux"/> is given, the
    /// pre-activations into it (same layout as c); <see cref="GemmEpilogue.GeluGradient"/> multiplies the product by
    /// gelu'(aux). Returns false when the device has no tensor cores (callers use the composed operations).
    /// </summary>
    public virtual bool GemmStridedKernel(Storage a, long aOffset, int lda, bool transA, Storage b, long bOffset, int ldb, bool transB,
        Storage c, long cOffset, int ldc, int m, int n, int k, float beta, Storage? bias = null, GemmEpilogue epilogue = GemmEpilogue.None,
        Storage? aux = null, long auxOffset = 0) => false;

    /// <summary>
    /// Causal attention (positions c ≤ t) read in place from [batch, steps, *] rows: query head h (= kv · group + g, g &lt;
    /// group) of batch b at step t at q[qOffset + (b·steps + t)·qRow + h·dim], keys and values of kv head kv at k /
    /// v[offset + (b·steps + c)·kRow + kv·dim] (offsets and row strides in floats, multiples of 4). With the scores s_c =
    /// scale · q·k_c for c ≤ t, writes y[(b·steps + t)·heads·dim + h·dim] = Σ_c softmax(s)_c · v_c ([batch, steps,
    /// heads·dim], heads = kvHeads · group) and, when given, the log-sum-exp ln Σ_c exp(s_c) of the scaled scores at
    /// [(b·kvHeads + kv)·group·steps + g·steps + t] ([batch·kvHeads, group·steps]) for the backward pass. Devices may run it
    /// on bfloat16 matrix units (q, k, v and the probabilities rounded to bfloat16, float32 sums), so it agrees with
    /// float32 attention to about 1e-2. Returns false when the device has no such kernels, or none for this head size
    /// (callers rearrange the heads and use <see cref="AttentionTiled"/>).
    /// </summary>
    public virtual bool AttentionStridedKernel(Storage q, long qOffset, Storage k, long kOffset, Storage v, long vOffset, int qRow, int kRow,
        Storage y, Storage? logSumExp, int batch, int kvHeads, int group, int steps, int dim, float scale) => false;

    /// <summary>
    /// Gradients of <see cref="AttentionStrided"/> given its y and log-sum-exp and dOutput (y's layout): with P_tc =
    /// exp(s_tc - logSumExp_t) (the scores recomputed from q and k), D_t = Σ_d y_t · dOutput_t and dS_tc = P_tc ·
    /// (dOutput_t · v_c - D_t), adds scale · Σ_c dS_tc · k_c to dq_t, scale · Σ_t dS_tc · q_t to dk_c and Σ_t P_tc · dOutput_t
    /// to dv_c (dk and dv summed over the query heads of their group), each laid out as q, k, v (the same row strides,
    /// their own offsets). q, k and v may be one storage at three offsets, and dq, dk and dv another, as
    /// <c>Tensor.CausalAttentionPacked</c> passes them. Returns false when the device has no such kernels.
    /// </summary>
    public virtual bool AttentionStridedBackwardKernel(Storage q, long qOffset, Storage k, long kOffset, Storage v, long vOffset, int qRow, int kRow,
        Storage y, Storage logSumExp, Storage dOutput, Storage dq, long dqOffset, Storage dk, long dkOffset, Storage dv, long dvOffset,
        int batch, int kvHeads, int group, int steps, int dim, float scale) => false;

    /// <summary>y[j] += Σ_r x[offset + r·ld + j] for j &lt; cols (column sums of a strided block; the bias gradient of a slice).</summary>
    public virtual void SumColumnsKernel(Storage x, long offset, int ld, Storage y, int rows, int cols)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.SumColumns(h[x], offset, ld, h[y], rows, cols);
    }

    /// <summary>
    /// Until disposed, 8-bit products on this device reuse the quantized form of an operand they have already quantized
    /// (same memory, layout and size), for callers that multiply one unchanged tensor several times (the query, key and
    /// value projections of one input). Null when the device quantizes nothing.
    /// </summary>
    public virtual IDisposable? ReuseQuantizedOperands() => null;

    /// <summary>output = residual + dropout(x) (the mask of <see cref="Dropout"/> with <paramref name="seed"/>).</summary>
    public virtual void AddDropoutKernel(Storage residual, Storage x, Storage output, int n, float p, uint seed)
    {
        Dropout(x, output, n, p, seed);
        Axpy(residual, output, n, 1f);
    }

    /// <summary><see cref="MatMul"/> for <paramref name="batch"/> independent, contiguous matrix triples.</summary>
    public virtual void BatchedMatMulKernel(Storage a, Storage b, Storage c, int batch, int m, int n, int k, bool transA, bool transB, float beta)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.BatchedMatMul(h[a], h[b], h[c], batch, m, n, k, transA, transB, beta);
    }

    /// <summary>Row-wise softmax (or log-softmax) over the last dimension: y[r, :] = softmax(x[r, :]).</summary>
    public virtual void SoftmaxKernel(Storage x, Storage y, int rows, int cols, bool log)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.Softmax(h[x], h[y], rows, cols, log);
    }

    /// <summary>
    /// Softmax: dx += y * (dy - Σ dy·y). Log-softmax: dx += dy - exp(y) * Σ dy. Sums are per row; y is the forward output.
    /// </summary>
    public virtual void SoftmaxBackwardKernel(Storage y, Storage dy, Storage dx, int rows, int cols, bool log)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.SoftmaxBackward(h[y], h[dy], h[dx], rows, cols, log);
    }

    /// <summary>y[r] = index of the largest element of row r.</summary>
    public virtual void ArgMaxKernel(Storage x, Storage y, int rows, int cols)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.ArgMax(h[x], h[y], rows, cols);
    }

    /// <summary>
    /// y[r] = 1 when the prediction in row r is right, else 0. For one column: (p ≥ threshold) == (t ≥ 0.5);
    /// otherwise argmax(p) == argmax(t).
    /// </summary>
    public virtual void ClassMatchKernel(Storage predictions, Storage targets, Storage y, int rows, int cols, float threshold)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.ClassMatch(h[predictions], h[targets], h[y], rows, cols, threshold);
    }

    // Grouped normalization. Data is viewed as [outer, groups, inner]; group g holds the M = outer * inner
    // elements x[o, g, i]. BatchNorm uses groups = channels; LayerNorm uses outer = 1, groups = rows, inner = features.

    /// <summary>Per-group mean, (biased) variance and 1 / sqrt(variance + eps).</summary>
    public virtual void NormStatsKernel(Storage x, Storage mean, Storage variance, Storage invStd, int outer, int groups, int inner, float eps)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.NormStats(h[x], h[mean], h[variance], h[invStd], outer, groups, inner, eps);
    }

    /// <summary>y = (x - mean[g]) * invStd[g].</summary>
    public virtual void NormApplyKernel(Storage x, Storage mean, Storage invStd, Storage y, int outer, int groups, int inner)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.NormApply(h[x], h[mean], h[invStd], h[y], outer, groups, inner);
    }

    /// <summary>dx += invStd[g] / M * (M * dxhat - sum1[g] - xhat * sum2[g]).</summary>
    public virtual void NormBackwardKernel(Storage dxhat, Storage xhat, Storage sum1, Storage sum2, Storage invStd, Storage dx, int outer, int groups, int inner)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.NormBackward(h[dxhat], h[xhat], h[sum1], h[sum2], h[invStd], h[dx], outer, groups, inner);
    }

    /// <summary>
    /// y (+)= x * scale[g] + shift[g] with g = (i / inner) % groups; a null scale means 1, a null shift means 0.
    /// </summary>
    public virtual void GroupScaleShiftKernel(Storage x, Storage? scale, Storage? shift, Storage y, int n, int groups, int inner, bool accumulate)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.GroupScaleShift(h[x], h.Maybe(scale), h.Maybe(shift), h[y], n, groups, inner, accumulate);
    }

    /// <summary>sumA[g] += Σ a; sumAB[g] += Σ a·b over each group (see NormStats for the layout). b/sumAB may be null.</summary>
    public virtual void GroupReduceKernel(Storage a, Storage? b, Storage sumA, Storage? sumAB, int outer, int groups, int inner)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.GroupReduce(h[a], h.Maybe(b), h[sumA], h.Maybe(sumAB), outer, groups, inner);
    }

    /// <summary>y = 1 / sqrt(x + eps).</summary>
    public virtual void InvSqrtKernel(Storage x, Storage y, int n, float eps)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.InvSqrt(h[x], h[y], n, eps);
    }

    /// <summary>Embedding lookup: y[i, :] = table[indices[i], :] for count indices of width dim.</summary>
    public virtual void GatherKernel(Storage table, Storage indices, Storage y, int count, int dim, int vocabulary)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.Gather(h[table], h[indices], h[y], count, dim, vocabulary);
    }

    /// <summary><see cref="Gather"/> from a bfloat16 table packed as in <see cref="BFloat16MatMul"/> ([vocabulary, dim]).</summary>
    public virtual void GatherBFloat16Kernel(Storage packed, Storage indices, Storage y, int count, int dim, int vocabulary)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.GatherBFloat16(h[packed], h[indices], h[y], count, dim, vocabulary);
    }

    /// <summary>
    /// <see cref="Gather"/> of columns: y[i, :] = column indices[i] of a bfloat16 table [dim, vocabulary] packed as in
    /// <see cref="BFloat16MatMul"/> (rows of ⌈vocabulary / 2⌉ words): the embedding lookup of a model whose tied output
    /// head holds the only copy of the table, as its [dim, vocabulary] weight.
    /// </summary>
    public virtual void GatherBFloat16ColumnsKernel(Storage packed, Storage indices, Storage y, int count, int dim, int vocabulary)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.GatherBFloat16Columns(h[packed], h[indices], h[y], count, dim, vocabulary);
    }

    /// <summary>One-hot rows: y[i, :] = 0 except y[i, indices[i]] = 1, for count indices over classes columns.</summary>
    public virtual void OneHotKernel(Storage indices, Storage y, int count, int classes)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.OneHot(h[indices], h[y], count, classes);
    }

    /// <summary>dtable[indices[i], :] += dy[i, :].</summary>
    public virtual void ScatterAddKernel(Storage dy, Storage indices, Storage dtable, int count, int dim, int vocabulary)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.ScatterAdd(h[dy], h[indices], h[dtable], count, dim, vocabulary);
    }

    /// <summary>Unfolds image patches: cols[(n, oh, ow), (c, kh, kw)] = x[n, c, oh*sh - ph + kh, ow*sw - pw + kw] (0 outside).</summary>
    public virtual void Im2ColKernel(Storage x, Storage cols, in ConvGeometry g)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.Im2Col(h[x], h[cols], in g);
    }

    /// <summary>The adjoint of <see cref="Im2Col"/>: dx += fold(dcols).</summary>
    public virtual void Col2ImKernel(Storage dcols, Storage dx, in ConvGeometry g)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.Col2Im(h[dcols], h[dx], in g);
    }

    /// <summary>Max pooling; argmax receives the flat input index of each maximum (as raw int bits).</summary>
    public virtual void MaxPoolKernel(Storage x, Storage y, Storage argmax, in ConvGeometry g)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.MaxPool(h[x], h[y], h[argmax], in g);
    }

    /// <summary>dx[argmax[i]] += dy[i].</summary>
    public virtual void MaxPoolBackwardKernel(Storage dy, Storage argmax, Storage dx, int count)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.MaxPoolBackward(h[dy], h[argmax], h[dx], count);
    }

    /// <summary>
    /// dx[argmax[i]] += dy[i] for the windows of <paramref name="g"/> (count = N * C * OH * OW): the same as the overload
    /// taking a count; a device that adds the gradients by gathering over the windows needs the geometry.
    /// </summary>
    public virtual void MaxPoolBackwardKernel(Storage dy, Storage argmax, Storage dx, in ConvGeometry g) =>
        MaxPoolBackward(dy, argmax, dx, g.N * g.C * g.OH * g.OW);

    /// <summary>
    /// y (+)= x permuted: output element at coordinates (c0..c[r-1]) of <paramref name="outShape"/> comes from
    /// input offset Σ c_k * inStrides[k] (the input strides already reordered by the permutation). Rank ≤ 6.
    /// </summary>
    public virtual void PermuteKernel(Storage x, Storage y, ReadOnlySpan<int> outShape, ReadOnlySpan<int> inStrides, bool accumulate)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.Permute(h[x], h[y], outShape, inStrides, accumulate);
    }

    /// <summary>
    /// Strided block copy: dst[dstOffset + r*dstStride + c] (+)= src[srcOffset + r*srcStride + c] for r &lt; rows, c &lt; cols.
    /// Implements narrow, concatenation and their gradients.
    /// </summary>
    public virtual void Copy2D(Storage src, int srcOffset, int srcStride, Storage dst, int dstOffset, int dstStride, int rows, int cols, bool accumulate)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.Copy2D(h[src], srcOffset, srcStride, h[dst], dstOffset, dstStride, rows, cols, accumulate);
    }

    /// <summary>y[o, i] (+)= scale * Σ_d x[o, d, i] for a [outer, dim, inner] view.</summary>
    public virtual void SumAxisKernel(Storage x, Storage y, int outer, int dim, int inner, float scale, bool accumulate)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.SumAxis(h[x], h[y], outer, dim, inner, scale, accumulate);
    }

    /// <summary>dx[o, d, i] += scale * dy[o, i] (the gradient of <see cref="SumAxis"/>).</summary>
    public virtual void BroadcastAxisKernel(Storage dy, Storage dx, int outer, int dim, int inner, float scale)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.BroadcastAxis(h[dy], h[dx], outer, dim, inner, scale);
    }

    /// <summary>SGD with optional momentum: v = momentum * v + g; p -= lr * v (v is null when momentum is 0).</summary>
    public virtual void SgdStepKernel(Storage p, Storage g, Storage? v, int n, float lr, float momentum)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.SgdStep(h[p], h[g], h.Maybe(v), n, lr, momentum);
    }

    /// <summary>Adam: m, v moments updated in place; p -= lr * m / (sqrt(v) + eps). lr is already bias-corrected.</summary>
    public virtual void AdamStepKernel(Storage p, Storage g, Storage m, Storage v, int n, float lr, float beta1, float beta2, float eps)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.AdamStep(h[p], h[g], h[m], h[v], n, lr, beta1, beta2, eps);
    }

    /// <summary>
    /// <see cref="AdamStep"/> with 8-bit moments (see <c>Optimizers.AdamW8Bit</c>): m and v hold one byte per element
    /// (n bytes, packed four per float), codes into <paramref name="map"/> (256 signed values for m, then 256 unsigned for
    /// v, both in [-1, 1]) scaled per block of <see cref="EightBitMoments.BlockSize"/> elements by
    /// <paramref name="absMax"/> (the blocks' m scales, then their v scales). The moments are decoded, updated, and
    /// encoded again to the nearest code with new block scales.
    /// </summary>
    /// <remarks>The gradient is multiplied by <paramref name="gradientScale"/> as it is read (gradient clipping) and the
    /// parameter by <paramref name="decay"/> before the update (decoupled weight decay, 1 - lr·λ).</remarks>
    public virtual void AdamStep8BitKernel(Storage p, Storage g, Storage m, Storage v, Storage absMax, Storage map, int n, float lr, float beta1, float beta2, float eps,
        float gradientScale, float decay)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.AdamStep8Bit(h[p], h[g], h[m], h[v], h[absMax], h[map], n, lr, beta1, beta2, eps, gradientScale, decay);
    }

    /// <summary>total[0] += Σ x² over <paramref name="n"/> elements (a gradient norm without temporary tensors).</summary>
    public virtual void SumSquaresKernel(Storage x, Storage total, int n)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.SumSquares(h[x], h[total], n);
    }

    /// <summary>
    /// AdamW over many tensors in a few passes: the global gradient norm clipped to <paramref name="maxNorm"/> (0: no
    /// clipping), p ← p · <paramref name="decay"/>, then Adam with the bias-corrected <paramref name="lr"/>, as
    /// <see cref="AdamStep"/> computes it; with <paramref name="zeroGradients"/> the gradients are zeroed for the next step.
    /// <paramref name="cache"/> keeps the device tables between calls (dispose it when done). False when the device has no
    /// such pass.
    /// </summary>
    public virtual bool FusedAdamWKernel(ReadOnlySpan<(Storage P, Storage G, Storage M, Storage V, int N)> tensors, ref IDisposable? cache, float maxNorm,
        float lr, float decay, float beta1, float beta2, float eps, bool zeroGradients)
    {
        using var h = new HostCall(this);
        var mirrors = new (Storage P, Storage G, Storage M, Storage V, int N)[tensors.Length];
        for (int i = 0; i < tensors.Length; i++)
        {
            var (p, g, m, v, n) = tensors[i];
            mirrors[i] = (h[p], h[g], h[m], h[v], n);
        }

        IDisposable? none = null;                                          // the CPU keeps no tables
        return CpuBackend.Instance.FusedAdamW(mirrors, ref none, maxNorm, lr, decay, beta1, beta2, eps, zeroGradients);
    }

    /// <summary>
    /// factor[0] = min(1, maxNorm / √sumSquares[0]) (1 when the sum is 0): the gradient-clipping factor computed where the
    /// gradients are, so clipping needs no host read of the norm.
    /// </summary>
    public virtual void ClipFactorKernel(Storage sumSquares, Storage factor, float maxNorm)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.ClipFactor(h[sumSquares], h[factor], maxNorm);
    }

    /// <summary>Inverted dropout: y = keep(i) ? x / (1 - p) : 0, where keep(i) comes from <see cref="DropoutMask"/>.</summary>
    public virtual void DropoutKernel(Storage x, Storage y, int n, float p, uint seed)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.Dropout(h[x], h[y], n, p, seed);
    }

    /// <summary>dx += keep(i) ? dy / (1 - p) : 0, regenerating the same mask from the seed.</summary>
    public virtual void DropoutBackwardKernel(Storage dy, Storage dx, int n, float p, uint seed)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.DropoutBackward(h[dy], h[dx], n, p, seed);
    }

    /// <summary>Waits until every kernel and copy issued so far has finished.</summary>
    public abstract void Synchronize();

    // ---------------------------------------------------------------- fused inference kernels

    /// <summary>y[r, :] = softmax(scale * x[r, :] + mask[r % maskRows, :]) (mask optional).</summary>
    public virtual void ScaleMaskSoftmaxKernel(Storage x, Storage? mask, Storage y, int rows, int cols, int maskRows, float scale)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.ScaleMaskSoftmax(h[x], h.Maybe(mask), h[y], rows, cols, maskRows, scale);
    }

    /// <summary>y[r, :] = (x[r, :] - mean) / sqrt(var + eps) * gamma + beta over the last dimension.</summary>
    public virtual void LayerNormFusedKernel(Storage x, Storage gamma, Storage beta, Storage y, int rows, int cols, float eps)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.LayerNormFused(h[x], h[gamma], h[beta], h[y], rows, cols, eps);
    }

    /// <summary><see cref="LayerNormFused"/> that also stores each row's mean (stats[r]) and 1 / sqrt(var + eps) (stats[rows + r]).</summary>
    public virtual void LayerNormTrainKernel(Storage x, Storage gamma, Storage beta, Storage y, Storage stats, int rows, int cols, float eps)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.LayerNormTrain(h[x], h[gamma], h[beta], h[y], h[stats], rows, cols, eps);
    }

    /// <summary>
    /// Gradients of <see cref="LayerNormTrain"/> given dy: adds to dx (when given), dgamma += Σ_r dy ∘ x̂ and dbeta += Σ_r dy.
    /// </summary>
    public virtual void LayerNormBackwardKernel(Storage x, Storage gamma, Storage dy, Storage stats, Storage? dx, Storage? dgamma, Storage? dbeta, int rows, int cols)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.LayerNormBackward(h[x], h[gamma], h[dy], h[stats], h.Maybe(dx), h.Maybe(dgamma), h.Maybe(dbeta), rows, cols);
    }

    /// <summary>y[i] = gelu(x[i] + bias[i % cols]).</summary>
    public virtual void BiasGeluKernel(Storage x, Storage bias, Storage y, int n, int cols)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.BiasGelu(h[x], h[bias], h[y], n, cols);
    }

    /// <summary>
    /// y[m, n] = x[m, k] · w, where w[k, j] = q[k, j] · scales[j] and q holds signed bytes packed four per 32-bit element
    /// along each row (rows padded to ceil(n / 4) elements). Suited to few rows (token-by-token decoding).
    /// </summary>
    public virtual void Int8MatMulKernel(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.Int8MatMul(h[x], h[q], h[scales], h[y], m, n, k);
    }

    /// <summary>
    /// y[m, n] = x[m, k] · w[k, n] with w stored as bfloat16 pairs packed into 32-bit words along each row (word c of
    /// row r holds columns 2c in its low half and 2c + 1 in its high half; rows have ⌈n / 2⌉ words).
    /// </summary>
    public virtual void BFloat16MatMulKernel(Storage x, Storage packed, Storage y, int m, int n, int k)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.BFloat16MatMul(h[x], h[packed], h[y], m, n, k);
    }

    /// <summary>w[k, n] = the float32 values of bfloat16 weights packed as in <see cref="BFloat16MatMul"/>.</summary>
    public virtual void BFloat16DequantizeKernel(Storage packed, Storage w, int k, int n)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.BFloat16Dequantize(h[packed], h[w], k, n);
    }

    /// <summary>
    /// Rounds x [n] to bfloat16 (to nearest, ties to even), two values per word as <see cref="BFloat16Dequantize"/> reads
    /// them back with k = 1: packed holds (n + 1) / 2 words.
    /// </summary>
    public virtual void PackBFloat16Kernel(Storage x, Storage packed, int n)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.PackBFloat16(h[x], h[packed], n);
    }

    /// <summary>w[k, n] = q[k, n] · scales[n] (see <see cref="Int8MatMul"/> for the packing).</summary>
    public virtual void Int8DequantizeKernel(Storage q, Storage scales, Storage w, int k, int n)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.Int8Dequantize(h[q], h[scales], h[w], k, n);
    }

    /// <summary>
    /// y[m, n] = x[m, k] · w with 4-bit weights: w[r, j] = q[r, j] · scales[r / 32, j], where q holds signed nibbles packed
    /// eight per 32-bit word along each row (nibble c of word w is column 8w + c; rows have ⌈n / 8⌉ words) and scales has
    /// one row of 8·⌈n / 8⌉ values per group of 32 weight rows.
    /// </summary>
    public virtual void Int4MatMulKernel(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.Int4MatMul(h[x], h[q], h[scales], h[y], m, n, k);
    }

    /// <summary>w[k, n] = the float values of 4-bit weights packed as in <see cref="Int4MatMul"/>.</summary>
    public virtual void Int4DequantizeKernel(Storage q, Storage scales, Storage w, int k, int n)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.Int4Dequantize(h[q], h[scales], h[w], k, n);
    }

    // Int8 KV cache: each cached row (one head, one position) is dim bytes packed four per element, words = ceil(dim / 4)
    // elements, with one scale per row in scales[head, position].

    /// <summary>y = x · inv per row, inv[r] = 1 / sqrt(mean(x[r]²) + eps) (stored for the backward pass).</summary>
    public virtual void RmsNormKernel(Storage x, Storage y, Storage inv, int rows, int cols, float eps)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.RmsNorm(h[x], h[y], h[inv], rows, cols, eps);
    }

    /// <summary>dx += inv[r] · (dy - y · mean(dy · y)) per row, where y is the normalized forward output.</summary>
    public virtual void RmsNormBackwardKernel(Storage dy, Storage y, Storage inv, Storage dx, int rows, int cols)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.RmsNormBackward(h[dy], h[y], h[inv], h[dx], rows, cols);
    }

    /// <summary>
    /// Rotary position embedding of x [rows = batch·steps·heads, dim] into y (which must already hold x): pair p of a row at
    /// step t rotates by the angle whose cos/sin are cos/sin[positions[t], p]. Pairs are (2p, 2p+1) when interleaved,
    /// else (p, p + half). sign = -1 rotates backwards (the gradient).
    /// </summary>
    public virtual void RopeKernel(Storage x, Storage y, Storage cos, Storage sin, Storage positions, int rows, int heads, int steps, int dim, int half, bool interleaved, float sign)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.Rope(h[x], h[y], h[cos], h[sin], h[positions], rows, heads, steps, dim, half, interleaved, sign);
    }

    /// <summary>y = x · inv · (gain[c] + offset) per row with inv = 1 / sqrt(mean(x²) + eps): normalization and gain in one pass (inference).</summary>
    public virtual void RmsNormAffineKernel(Storage x, Storage gain, Storage y, int rows, int cols, float eps, float offset)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.RmsNormAffine(h[x], h[gain], h[y], rows, cols, eps, offset);
    }

    /// <summary>sum = a + b and y = RMS-normalized sum · (gain[c] + offset) per row, in one pass (inference).</summary>
    public virtual void AddRmsNormAffineKernel(Storage a, Storage b, Storage sum, Storage gain, Storage y, int rows, int cols, float eps, float offset)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.AddRmsNormAffine(h[a], h[b], h[sum], h[gain], h[y], rows, cols, eps, offset);
    }

    /// <summary>
    /// RMS normalization with gain of each row (a head's vector), then the rotary embedding as <see cref="Rope"/> with
    /// sign 1 (rows are batch·steps·heads; dimensions beyond 2·half are only normalized). Inference.
    /// </summary>
    public virtual void RmsNormRopeKernel(Storage x, Storage gain, Storage cos, Storage sin, Storage positions, Storage y, int rows, int cols,
        float eps, float offset, int heads, int steps, int half, bool interleaved)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.RmsNormRope(h[x], h[gain], h[cos], h[sin], h[positions], h[y], rows, cols, eps, offset, heads, steps, half, interleaved);
    }

    /// <summary>
    /// <see cref="RmsNormRope"/> for two tensors sharing the positions and rotary tables (queries and keys): rows1 rows
    /// of x (heads per step: heads) and rows2 of x2 (heads2). Devices may do both in one pass.
    /// </summary>
    public virtual void RmsNormRopePairKernel(Storage x, Storage gain, Storage y, int rows1, float eps, float offset, int heads,
        Storage x2, Storage gain2, Storage y2, int rows2, float eps2, float offset2, int heads2,
        Storage cos, Storage sin, Storage positions, int cols, int steps, int half, bool interleaved)
    {
        RmsNormRope(x, gain, cos, sin, positions, y, rows1, cols, eps, offset, heads, steps, half, interleaved);
        RmsNormRope(x2, gain2, cos, sin, positions, y2, rows2, cols, eps2, offset2, heads2, steps, half, interleaved);
    }

    /// <summary>
    /// The attention layer's queries, keys and values [batch, steps, heads·cols] (its projections) put in the layouts
    /// attention reads, in one pass (inference): query heads RMS-normalized with gain (when <paramref name="gainQ"/> is
    /// set; keys then with <paramref name="gainK"/>) and rotated as <see cref="Rope"/> (half 0: no rotation) into
    /// yq [batch, heads, steps, cols]; keys the same and values unchanged into yk and yv [batch, kvHeads, capacity, stride]
    /// at row position[0] + s (a cache; row s when <paramref name="position"/> is null), as floats or bfloat16 pairs.
    /// Returns false when the device has no such kernel (callers then run the steps one by one).
    /// </summary>
    public virtual bool NormRopeHeadsKernel(Storage q, Storage k, Storage v, int batch, int steps, int heads, int kvHeads, int cols,
        Storage? gainQ, float epsQ, float offsetQ, Storage? gainK, float epsK, float offsetK, Storage? cos, Storage? sin, Storage? positions,
        int half, bool interleaved, Storage yq, Storage yk, Storage yv, Storage? position, int capacity, int stride, bool bfloat16) => false;

    /// <summary>
    /// Token cross-entropy for language-model training, per row r of logits [rows, vocabulary] with target class t_r and
    /// weight w_r (0 masks the row): losses[r] = w_r · (logsumexp(x_r) - x_r[t_r]); the logits are overwritten with their
    /// gradient scale · w_r · (softmax(x_r) - onehot(t_r)). Targets and weights are float arrays of rows.
    /// </summary>
    public virtual void SoftmaxCrossEntropyRowsKernel(Storage logits, Storage targets, Storage weights, Storage losses, int rows, int vocabulary, float scale)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.SoftmaxCrossEntropyRows(h[logits], h[targets], h[weights], h[losses], rows, vocabulary, scale);
    }

    /// <summary>y = act(gate) · up element-wise; kind 0 = SiLU, 1 = GELU (tanh approximation), 2 = ReLU.</summary>
    public virtual void GatedActivationKernel(Storage gate, Storage up, Storage y, int n, int kind)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.GatedActivation(h[gate], h[up], h[y], n, kind);
    }

    /// <summary>dgate += dy · up · act'(gate) when flags has bit 0, dup += dy · act(gate) when it has bit 1; bits 2 and 3 write dgate and dup (= instead of +=).</summary>
    public virtual void GatedActivationBackwardKernel(Storage gate, Storage up, Storage dy, Storage dgate, Storage dup, int n, int kind, int flags)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.GatedActivationBackward(h[gate], h[up], h[dy], h[dgate], h[dup], n, kind, flags);
    }

    /// <summary>
    /// <see cref="GatedActivation"/> with bfloat16 words (two values each, low half first, as <see cref="PackBFloat16"/>
    /// writes them): gate and up read from <paramref name="packedGate"/> and <paramref name="packedUp"/> when flags has
    /// bit 0 (else from <paramref name="gate"/> and <paramref name="up"/>); y written when it has bit 2, gate and up packed
    /// when it has bit 1, y packed when it has bit 3. Storages a flag does not use may be any storage.
    /// </summary>
    public virtual void GatedActivationPackedKernel(Storage gate, Storage up, Storage packedGate, Storage packedUp, Storage y, Storage packedY, int n, int kind, int flags)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.GatedActivationPacked(h[gate], h[up], h[packedGate], h[packedUp], h[y], h[packedY], n, kind, flags);
    }

    /// <summary><see cref="GatedActivationBackward"/> reading gate and up as bfloat16 words.</summary>
    public virtual void GatedActivationBackwardPackedKernel(Storage packedGate, Storage packedUp, Storage dy, Storage dgate, Storage dup, int n, int kind, int flags)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.GatedActivationBackwardPacked(h[packedGate], h[packedUp], h[dy], h[dgate], h[dup], n, kind, flags);
    }

    /// <summary>Quantizes source [heads·steps, dim] into the int8 cache at positions position[0] + step.</summary>
    public virtual void KeyValueWriteInt8Kernel(Storage source, Storage cache, Storage scales, Storage position, int heads, int steps, int capacity, int dim)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.KeyValueWriteInt8(h[source], h[cache], h[scales], h[position], heads, steps, capacity, dim);
    }

    /// <summary>y[r, t, c] = scales[r, c] · Σ_d q[r, t, d] · keys[r, c, d] for every cached position c.</summary>
    public virtual void AttentionScoresInt8Kernel(Storage q, Storage cache, Storage scales, Storage y, int rows, int steps, int capacity, int dim)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.AttentionScoresInt8(h[q], h[cache], h[scales], h[y], rows, steps, capacity, dim);
    }

    /// <summary>y[r, t, d] = Σ_c weights[r, t, c] · scales[r, c] · values[r, c, d].</summary>
    public virtual void AttentionContextInt8Kernel(Storage weights, Storage cache, Storage scales, Storage y, int rows, int steps, int capacity, int dim)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.AttentionContextInt8(h[weights], h[cache], h[scales], h[y], rows, steps, capacity, dim);
    }

    /// <summary>
    /// Attention over a key/value cache filled up to the device position: for head h and row i of q [heads, rowsPerHead,
    /// dim], y[h, i] = Σ_c softmax(scale · q[h, i] · keys[h, c]) · values[h, c] over positions c = 0 … position[0] +
    /// (i % steps) (the causal limit of that row's step). Keys and values are [heads, capacity, dim]; unfilled
    /// positions are never read, so the cost follows the context length rather than the capacity. With a
    /// <paramref name="variant"/>, each row starts at its window's first position and its scores are soft-capped (every
    /// attention method below takes one; the default is plain causal attention).
    /// </summary>
    public virtual void AttentionDecodeKernel(Storage q, Storage keys, Storage values, Storage position, Storage y, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.AttentionDecode(h[q], h[keys], h[values], h[position], h[y], heads, rowsPerHead, steps, capacity, dim, scale, variant);
    }

    /// <summary>
    /// <see cref="AttentionDecode"/> (tiled: <paramref name="tiled"/>, for many query rows) over an int8 cache: keys and
    /// values [heads, capacity, ⌈dim / 4⌉ words] of packed bytes with one scale per cached row (keyScales, valueScales
    /// [heads, capacity]).
    /// </summary>
    public virtual void AttentionInt8Kernel(Storage q, Storage keys, Storage values, Storage keyScales, Storage valueScales, Storage position,
        Storage y, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale, bool tiled, AttentionVariant variant = default)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.AttentionInt8(h[q], h[keys], h[values], h[keyScales], h[valueScales], h[position], h[y], heads, rowsPerHead, steps, capacity, dim, scale, tiled, variant);
    }

    /// <summary>
    /// <see cref="AttentionDecode"/> (tiled: <paramref name="tiled"/>) over a bfloat16 cache: keys and values
    /// [heads, capacity, ⌈dim / 2⌉ words], dimension d in the low (even d) or high (odd d) half of word d / 2.
    /// </summary>
    public virtual void AttentionBFloat16Kernel(Storage q, Storage keys, Storage values, Storage position, Storage y, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale, bool tiled, AttentionVariant variant = default)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.AttentionBFloat16(h[q], h[keys], h[values], h[position], h[y], heads, rowsPerHead, steps, capacity, dim, scale, tiled, variant);
    }

    /// <summary><see cref="KeyValueWrite"/> into a bfloat16 cache (values rounded to nearest, ties to even).</summary>
    public virtual void KeyValueWriteBFloat16Kernel(Storage source, Storage cache, Storage position, int heads, int steps, int capacity, int dim)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.KeyValueWriteBFloat16(h[source], h[cache], h[position], heads, steps, capacity, dim);
    }

    /// <summary>
    /// Gradient of <see cref="AttentionTiled"/> with causal offset 0 (training): given the output, each row's log-sum-exp
    /// and dOutput, adds to dq [heads, rowsPerHead, dim] and dkeys, dvalues [heads, capacity, dim]. The attention weights
    /// are recomputed, never stored.
    /// </summary>
    public virtual void AttentionTiledBackwardKernel(Storage q, Storage keys, Storage values, Storage output, Storage logSumExp, Storage dOutput,
        Storage dq, Storage dkeys, Storage dvalues, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.AttentionTiledBackward(h[q], h[keys], h[values], h[output], h[logSumExp], h[dOutput], h[dq], h[dkeys], h[dvalues], heads, rowsPerHead, steps, capacity, dim, scale, variant);
    }

    /// <summary>
    /// Causal attention over packed sequences (training): as <see cref="AttentionTiled"/> with offset 0 and keys and values
    /// [heads, steps, dim], except that several sequences share each row of <paramref name="steps"/> positions, so row i of
    /// head h sees positions c with starts[b·steps + t] ≤ c ≤ t, where t = i % steps and b = h / <paramref name="headsPerRow"/>
    /// is the packed row; starts and ends hold, per position of each packed row, where its sequence begins and stops
    /// (exclusive), as floats. Writes the log-sum-exp when given. Returns false when the device has no such pass.
    /// </summary>
    public virtual bool AttentionSegmentedKernel(Storage q, Storage keys, Storage values, Storage y, Storage? logSumExp, Storage starts, Storage ends,
        int heads, int headsPerRow, int rowsPerHead, int steps, int dim, float scale, AttentionVariant variant = default)
    {
        using var h = new HostCall(this);
        return CpuBackend.Instance.AttentionSegmented(h[q], h[keys], h[values], h[y], h.Maybe(logSumExp), h[starts], h[ends], heads, headsPerRow, rowsPerHead, steps, dim, scale, variant);
    }

    /// <summary>
    /// <see cref="AttentionTiled"/> (no log-sum-exp) for rows of different lengths decoded together: row i of head h
    /// sees cached positions c with starts[(h / headsPerRow)·steps + i % steps] ≤ c ≤ position[0] + i % steps (a row with
    /// none, padding, gets zeros). Returns false when the device has no such pass.
    /// </summary>
    public virtual bool AttentionRowsKernel(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage starts, int heads, int headsPerRow,
        int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        using var h = new HostCall(this);
        return CpuBackend.Instance.AttentionRows(h[q], h[keys], h[values], h[position], h[y], h[starts], heads, headsPerRow, rowsPerHead, steps, capacity, dim, scale, variant);
    }

    /// <summary>Whether <see cref="AttentionSegmented"/> and its gradient run for this head size (with the current <c>MixedPrecision</c>).</summary>
    public virtual bool SupportsSegmentedAttention(int dim, AttentionVariant variant = default) => CpuBackend.Instance.SupportsSegmentedAttention(dim, variant);

    /// <summary>The gradient of <see cref="AttentionSegmented"/> (as <see cref="AttentionTiledBackward"/>); false when unsupported.</summary>
    public virtual bool AttentionSegmentedBackwardKernel(Storage q, Storage keys, Storage values, Storage output, Storage logSumExp, Storage dOutput,
        Storage dq, Storage dkeys, Storage dvalues, Storage starts, Storage ends, int heads, int headsPerRow, int rowsPerHead, int steps, int dim, float scale,
        AttentionVariant variant = default)
    {
        using var h = new HostCall(this);
        return CpuBackend.Instance.AttentionSegmentedBackward(h[q], h[keys], h[values], h[output], h[logSumExp], h[dOutput], h[dq], h[dkeys], h[dvalues],
            h[starts], h[ends], heads, headsPerRow, rowsPerHead, steps, dim, scale, variant);
    }

    /// <summary>
    /// The same attention as <see cref="AttentionDecode"/> for many query rows at once (a prompt, a training sequence),
    /// tiled so query rows share each key and value read; also writes each row's log-sum-exp of the scaled scores to
    /// <paramref name="logSumExp"/> [heads, rowsPerHead] when given (training). The default runs
    /// <see cref="AttentionDecode"/> for inference and the host fallback for training.
    /// </summary>
    public virtual void AttentionTiledKernel(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage? logSumExp, int heads,
        int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        if (logSumExp is null)
        {
            AttentionDecode(q, keys, values, position, y, heads, rowsPerHead, steps, capacity, dim, scale, variant);
            return;
        }

        using var h = new HostCall(this);
        CpuBackend.Instance.AttentionTiled(h[q], h[keys], h[values], h[position], h[y], h[logSumExp], heads, rowsPerHead, steps, capacity, dim, scale, variant);
    }

    /// <summary>
    /// Attention in which each query row sees one range of keys, read from the device: for query head h and row i of q
    /// [heads, rows, dim], y[h, i] = Σ_c softmax_c(s_c) · values[g, c] with s_c = scale · q[h, i] · keys[g, c] (soft-capped
    /// with the variant's cap), over the keys c with starts[b·rows + i] ≤ c &lt; ends[b·rows + i] (a half-open range, both
    /// clamped to [0, keyRows]), where g = h / (heads / kvHeads) is the key/value head the query head reads (grouped-query
    /// attention; heads is a multiple of kvHeads) and b = h / headsPerTable the table of ranges it reads (heads is a
    /// multiple of headsPerTable; starts and ends hold [heads / headsPerTable, rows] integers as floats). Keys and values
    /// are [kvHeads, keyRows, dim]. A row whose range is empty gets zeros, and a log-sum-exp of -∞. Writes each row's
    /// log-sum-exp of its scores (the natural log of Σ_c exp(s_c)) to <paramref name="logSumExp"/> [heads, rows] when
    /// given (training). Only the variant's soft-cap is read: a window is a rule for the ranges, like every mask that is
    /// not plainly causal (bidirectional, causal, sliding window, packed sequences, image blocks: <c>KeySpans</c>). The
    /// scores are never stored whole: the cost in memory follows rows + keys, not rows · keys.
    /// </summary>
    public virtual void AttentionSpansKernel(Storage q, Storage keys, Storage values, Storage starts, Storage ends, Storage y, Storage? logSumExp, int heads,
        int kvHeads, int headsPerTable, int rows, int keyRows, int dim, float scale, AttentionVariant variant = default)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.AttentionSpans(h[q], h[keys], h[values], h[starts], h[ends], h[y], h.Maybe(logSumExp), heads, kvHeads, headsPerTable, rows,
            keyRows, dim, scale, variant);
    }

    /// <summary>
    /// The gradients of <see cref="AttentionSpans"/> given its output, each row's log-sum-exp and dOutput (output's
    /// layout): with P_c = exp(s_c - logSumExp) over the row's range and Δ = dOutput · output, adds Σ_c scale · P_c ·
    /// (dOutput · values[g, c] - Δ) · keys[g, c] (times the cap's slope with a cap) to dq [heads, rows, dim], and to dkeys
    /// and dvalues [kvHeads, keyRows, dim] the matching terms (the query heads of a group add into their key/value head).
    /// The weights are recomputed, never stored; rows with an empty range add nothing.
    /// </summary>
    public virtual void AttentionSpansBackwardKernel(Storage q, Storage keys, Storage values, Storage starts, Storage ends, Storage output, Storage logSumExp,
        Storage dOutput, Storage dq, Storage dkeys, Storage dvalues, int heads, int kvHeads, int headsPerTable, int rows, int keyRows, int dim, float scale,
        AttentionVariant variant = default)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.AttentionSpansBackward(h[q], h[keys], h[values], h[starts], h[ends], h[output], h[logSumExp], h[dOutput], h[dq], h[dkeys],
            h[dvalues], heads, kvHeads, headsPerTable, rows, keyRows, dim, scale, variant);
    }

    /// <summary>
    /// True when attention over key ranges runs faster on this device composed — the full scores [heads, rows, keyRows]
    /// through <see cref="BatchedMatMul"/>, <see cref="ScaleMaskSoftmax"/> (with <paramref name="mask"/>, [rows, keyRows],
    /// or none when every row sees every key) and <see cref="BatchedMatMul"/> again (<see cref="ComposedAttention"/>) —
    /// than through <see cref="AttentionSpans"/>, for this shape and the current <see cref="MixedPrecision"/>. Given the
    /// real operands (one table of ranges, as many key/value heads as query heads, no soft-cap), so a device measures both
    /// on itself (into scratch memory of its own), once per shape and precision, and keeps the choice as it keeps its
    /// other measured choices (per device and driver). Called only when the device's free memory holds the scores with
    /// margin (<see cref="AvailableMemory"/>; <c>Tensor.AttentionFastest</c> checks). The default: false, AttentionSpans
    /// (it never holds the scores, so it never runs out of memory for them).
    /// </summary>
    public virtual bool PrefersComposedAttention(Storage q, Storage keys, Storage values, Storage starts, Storage ends, Storage? mask, int heads, int rows,
        int keyRows, int dim, float scale) => false;

    /// <summary>
    /// Attention through the full scores: scores = q · keysᵀ ([heads, rows, keyRows]), weights = softmax(scale · scores +
    /// mask) (<paramref name="mask"/> [rows, keyRows] repeated over the heads, or none), y = weights · values; q, y
    /// [heads, rows, dim], keys, values [heads, keyRows, dim]. <paramref name="scores"/> and <paramref name="weights"/>
    /// hold heads · rows · keyRows floats each. What <see cref="PrefersComposedAttention"/> measures against
    /// <see cref="AttentionSpans"/>.
    /// </summary>
    protected void ComposedAttention(Storage q, Storage keys, Storage values, Storage? mask, Storage scores, Storage weights, Storage y, int heads, int rows,
        int keyRows, int dim, float scale)
    {
        BatchedMatMul(q, keys, scores, heads, rows, keyRows, dim, transA: false, transB: true, beta: 0f);
        ScaleMaskSoftmax(scores, mask, weights, heads * rows, keyRows, mask is null ? 1 : rows, scale);
        BatchedMatMul(weights, values, y, heads, rows, dim, keyRows, transA: false, transB: false, beta: 0f);
    }

    /// <summary>
    /// Bytes of device memory new tensors can take now, as the device reports it (its free memory and the blocks its pool
    /// keeps for reuse, within <see cref="ComputeResources.GpuMemoryLimit"/>), or null when it reports none. Guards paths
    /// that need much memory at once (the composed attention's scores).
    /// </summary>
    public virtual long? AvailableMemory() => null;

    /// <summary>Starts counting the peak of the bytes in use (<see cref="MemoryUsage.Peak"/>) again from what is in use now.</summary>
    public virtual void ResetPeakMemoryUsage()
    {
    }

    // ---------------------------------------------------------------- incremental decoding (positions live on the device)

    /// <summary>mask[i, j] = j ≤ position + i ? 0 : -1e9 for a [rows, capacity] mask; position is read from device memory.</summary>
    public virtual void DecoderMaskKernel(Storage position, Storage mask, int rows, int capacity)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.DecoderMask(h[position], h[mask], rows, capacity);
    }

    /// <summary>cache[bh, position + t, :] = source[bh, t, :] for [heads, steps, dim] → [heads, capacity, dim].</summary>
    public virtual void KeyValueWriteKernel(Storage source, Storage cache, Storage position, int heads, int steps, int capacity, int dim)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.KeyValueWrite(h[source], h[cache], h[position], heads, steps, capacity, dim);
    }

    /// <summary>
    /// Draws one token per row from softmax(logits / temperature), restricted in turn to the top-k scores (topK &gt; 0),
    /// to the smallest set holding topP of the probability (0 &lt; topP &lt; 1; the cut-off is found by bisection on the
    /// score) and to tokens at least minP times as likely as the best one (minP &gt; 0), using the counter-based random
    /// stream (seed, step, row). Writes the token to ids[row] and 13 statistics to stats[(step * rows + row) * 13]:
    /// id, probability, entropy (bits), then the top-5 (id, probability) pairs. The step number is read from device
    /// memory. Row r's logits start at element r * rowStride + rowOffset.
    /// <para>
    /// Scores that are not finite follow one rule on every device: a NaN score is never sampled; when a row has a +∞ score
    /// (a +∞ logit, or a finite one that overflows when divided by the temperature) it draws uniformly among its +∞
    /// tokens, and top-k, top-p and min-p do not apply; a row with no finite score and no +∞ one has nothing to sample:
    /// ids[row] gets 0 (so the next step's input stays valid) and its statistics record id -1, probability 0, entropy 0
    /// and no alternatives, which the sampler reading them reports as an error.
    /// </para>
    /// </summary>
    public virtual void SampleRowsKernel(Storage logits, Storage ids, Storage stats, Storage step, int rows, int vocabulary,
        int rowStride, int rowOffset, float temperature, int topK, float topP, float minP, uint seed)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.SampleRows(h[logits], h[ids], h[stats], h[step], rows, vocabulary, rowStride, rowOffset, temperature, topK, topP, minP, seed);
    }

    /// <summary>
    /// Copies each row's logits (row r at r * rowStride + rowOffset) to work[r, :] and applies repetition penalties for
    /// the last min(length, lastN) tokens of the row's history ring history[r, pos % capacity] (length read from device
    /// memory). Each distinct token in the window is penalized once: repeat (x &gt; 0 ? x / repeat : x * repeat), then
    /// x -= presence + frequency * (occurrences in the window).
    /// </summary>
    public virtual void PenalizeRowsKernel(Storage logits, Storage work, Storage history, Storage length, int rows, int vocabulary,
        int rowStride, int rowOffset, int capacity, int lastN, float repeat, float presence, float frequency)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.PenalizeRows(h[logits], h[work], h[history], h[length], rows, vocabulary, rowStride, rowOffset, capacity, lastN, repeat, presence, frequency);
    }

    /// <summary>history[r, length % capacity] = ids[r] for every row (length read from device memory, not advanced).</summary>
    public virtual void HistoryPushKernel(Storage ids, Storage history, Storage length, int rows, int capacity)
    {
        using var h = new HostCall(this);
        CpuBackend.Instance.HistoryPush(h[ids], h[history], h[length], rows, capacity);
    }

    // ---------------------------------------------------------------- graphs

    /// <summary>Whether this device can record and replay work as a graph.</summary>
    public virtual bool SupportsGraphs => false;

    /// <summary>Starts recording all subsequent work on this device instead of running it.</summary>
    public virtual void BeginCapture() => throw new NotSupportedException();

    /// <summary>Stops recording and returns a replayable graph plus the device blocks it owns.</summary>
    public virtual (IntPtr Executable, IntPtr Graph, List<Storage> Owned) EndCapture() => throw new NotSupportedException();

    /// <summary>Stops recording after a failure, discarding the partial graph.</summary>
    public virtual List<Storage> AbortCapture() => [];

    /// <summary>Runs a graph returned by <see cref="EndCapture"/> again (its storages hold the inputs of the run).</summary>
    public virtual void ReplayGraph(IntPtr executable) => throw new NotSupportedException();

    /// <summary>Frees a graph returned by <see cref="EndCapture"/>.</summary>
    public virtual void DestroyGraph(IntPtr executable, IntPtr graph)
    {
    }
}

/// <summary>Stateless counter-based random numbers shared by every backend (the sampler's random stream).</summary>
internal static class CounterRandom
{
    /// <summary>A uniform float in [0, 1) from (seed, step, row) via the MurmurHash3 finalizer.</summary>
    public static float Uniform(uint seed, uint step, uint row)
    {
        uint h = seed ^ (step * 0x9E3779B9u) ^ (row * 0x85EBCA6Bu);
        h ^= h >> 16;
        h *= 0x85EBCA6Bu;
        h ^= h >> 13;
        h *= 0xC2B2AE35u;
        h ^= h >> 16;
        return (h >> 8) * (1f / 16777216f);
    }
}

/// <summary>
/// Stateless per-element random mask for dropout, identical on every backend: a MurmurHash3
/// finalizer of (index * golden ratio) ^ seed gives 24 random bits. Because the mask is a pure
/// function of (seed, index) it never has to be stored for the backward pass.
/// </summary>
internal static class DropoutMask
{
    public static bool Keep(uint seed, uint index, float p)
    {
        uint h = (index * 0x9E3779B9u) ^ seed;
        h ^= h >> 16;
        h *= 0x85EBCA6Bu;
        h ^= h >> 13;
        h *= 0xC2B2AE35u;
        h ^= h >> 16;
        return (h >> 8) * (1f / 16777216f) >= p;
    }
}

/// <summary>
/// Thread-safe byte accounting for a device's caching allocator: the bytes in use, kept in its pool and placed in system
/// memory, checked against a limit (<see cref="ComputeResources.GpuMemoryLimit"/>, say) before each new allocation.
/// </summary>
/// <param name="limit">The current limit in bytes, or null for none (read on every check, so a change applies at once).</param>
/// <param name="deviceName">The device's name in the error raised when an allocation would exceed the limit.</param>
public sealed class MemoryAccountant(Func<long?> limit, string deviceName)
{
    private long _inUse;
    private long _cached;
    private long _offloaded;
    private long _peak;

    /// <summary>The bytes in use, cached and offloaded, the limit and the peak in use, as <see cref="Backend.GetMemoryUsage"/> reports them.</summary>
    public MemoryUsage Usage => new(Interlocked.Read(ref _inUse), Interlocked.Read(ref _cached), limit(), Interlocked.Read(ref _offloaded),
        Math.Max(Interlocked.Read(ref _peak), Interlocked.Read(ref _inUse)));

    /// <summary>Starts the peak again from the bytes in use now (<see cref="Backend.ResetPeakMemoryUsage"/>).</summary>
    public void ResetPeak() => Interlocked.Exchange(ref _peak, Interlocked.Read(ref _inUse));

    // Raises the peak to `inUse` when higher.
    private void Peak(long inUse)
    {
        long peak = Interlocked.Read(ref _peak);
        while (inUse > peak)
        {
            long seen = Interlocked.CompareExchange(ref _peak, inUse, peak);
            if (seen == peak)
            {
                return;
            }

            peak = seen;
        }
    }

    /// <summary>Counts bytes placed in system memory for this device (negative when released).</summary>
    public void Offloaded(long bytes) => Interlocked.Add(ref _offloaded, bytes);

    /// <summary>
    /// Called before allocating <paramref name="bytes"/> of new memory. Returns true when the cache
    /// should be released first to stay under the limit.
    /// </summary>
    /// <exception cref="ResourceLimitExceededException">Even with the cache released, the allocation would exceed the limit.</exception>
    public bool MustReleaseCacheFor(long bytes)
    {
        if (limit() is not { } max)
        {
            return false;
        }

        long inUse = Interlocked.Read(ref _inUse);
        if (inUse + bytes > max)
        {
            throw new ResourceLimitExceededException(
                $"Allocating {bytes:N0} bytes on {deviceName} would exceed its memory limit of {max:N0} bytes ({inUse:N0} in use). " +
                "Raise the limit in ComputeResources, use smaller batches, or dispose tensors you no longer need.");
        }

        return inUse + Interlocked.Read(ref _cached) + bytes > max;
    }

    /// <summary>Counts newly allocated bytes as in use.</summary>
    public void Allocated(long bytes) => Peak(Interlocked.Add(ref _inUse, bytes));

    /// <summary>Moves bytes taken from the pool from cached to in use.</summary>
    public void Reused(long bytes)
    {
        Interlocked.Add(ref _cached, -bytes);
        Peak(Interlocked.Add(ref _inUse, bytes));
    }

    /// <summary>Moves bytes given back to the pool from in use to cached.</summary>
    public void Returned(long bytes)
    {
        Interlocked.Add(ref _inUse, -bytes);
        Interlocked.Add(ref _cached, bytes);
    }

    /// <summary>Counts cached bytes freed to the driver.</summary>
    public void Freed(long bytes) => Interlocked.Add(ref _cached, -bytes);
}

/// <summary>What <see cref="Backend.GemmStrided"/> does with each product before storing it.</summary>
public enum GemmEpilogue
{
    /// <summary>Store it (plus the bias).</summary>
    None = 0,

    /// <summary>Store gelu(product + bias) (tanh approximation), keeping product + bias in the auxiliary tensor when given.</summary>
    Gelu = 1,

    /// <summary>Store product · gelu'(auxiliary): the gradient through a GELU whose inputs the auxiliary tensor holds.</summary>
    GeluGradient = 2,
}
