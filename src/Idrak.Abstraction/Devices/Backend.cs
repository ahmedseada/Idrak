// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Diagnostics;
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

    /// <summary>GELU, exact: 0.5·x·(1 + erf(x/√2)), the error function within 1.5e-7 (Abramowitz and Stegun 7.1.26).</summary>
    GeluErf,
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
/// How <see cref="Backend.Interpolate2d"/> reads an output position from the input (PyTorch's <c>F.interpolate</c> modes).
/// </summary>
public enum InterpolationMode
{
    /// <summary>The input position floor(o · scale), the last one at most (PyTorch's "nearest").</summary>
    Nearest,

    /// <summary>The two nearest rows and columns, weighted by distance (PyTorch's "bilinear").</summary>
    Bilinear,
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
/// <param name="PH">Zero padding above (and below, unless <see cref="PadBottom"/> says otherwise).</param>
/// <param name="PW">Zero padding left (and right, unless <see cref="PadRight"/> says otherwise).</param>
/// <remarks>
/// A dilated window (<see cref="DH"/>, <see cref="DW"/> above 1, set with <c>with</c>) reads every DH-th row and DW-th
/// column: window position (kh, kw) is input row oh·SH - PH + kh·DH and column ow·SW - PW + kw·DW. Pooling takes no dilation.
/// Padding below and right (<see cref="PadBottom"/>, <see cref="PadRight"/>, set with <c>with</c>) may differ from PH and
/// PW (ONNX's four pads, a pooling window's ceil mode): it changes only how many windows there are (<see cref="OH"/>,
/// <see cref="OW"/>), never where they start, so every kernel that reads OH and OW and skips positions outside the input
/// runs it unchanged.
/// </remarks>
public readonly record struct ConvGeometry(
    int N, int C, int H, int W, int KH, int KW, int SH, int SW, int PH, int PW)
{
    // Stored as dilation - 1, so default(ConvGeometry) and every geometry made without it have dilation 1.
    private readonly int _dh, _dw;

    // Stored as padding + 1, 0 meaning "the same as PH / PW" (also when set to it, so equal geometries compare equal), so
    // every geometry made without them is symmetric.
    private readonly int _pb, _pr;

    /// <summary>Zero padding below (PH, the default, unless set).</summary>
    public int PadBottom
    {
        get => _pb == 0 ? PH : _pb - 1;
        init => _pb = value == PH ? 0 : value >= 0 ? value + 1 : throw new ArgumentOutOfRangeException(nameof(PadBottom), value, "Padding is not negative.");
    }

    /// <summary>Zero padding right (PW, the default, unless set).</summary>
    public int PadRight
    {
        get => _pr == 0 ? PW : _pr - 1;
        init => _pr = value == PW ? 0 : value >= 0 ? value + 1 : throw new ArgumentOutOfRangeException(nameof(PadRight), value, "Padding is not negative.");
    }

    /// <summary>Whether the padding below or right differs from the padding above or left.</summary>
    public bool Asymmetric => PadBottom != PH || PadRight != PW;

    /// <summary>Vertical dilation: the step between the rows a window reads (1, the default: adjacent rows).</summary>
    public int DH
    {
        get => _dh + 1;
        init => _dh = value >= 1 ? value - 1 : throw new ArgumentOutOfRangeException(nameof(DH), value, "A dilation is at least 1.");
    }

    /// <summary>Horizontal dilation: the step between the columns a window reads (1, the default: adjacent columns).</summary>
    public int DW
    {
        get => _dw + 1;
        init => _dw = value >= 1 ? value - 1 : throw new ArgumentOutOfRangeException(nameof(DW), value, "A dilation is at least 1.");
    }

    /// <summary>Whether a dilation is above 1 (devices without a dilated kernel run such windows on the host).</summary>
    public bool Dilated => _dh != 0 || _dw != 0;

    /// <summary>Output height: (H + PH + PadBottom - DH·(KH - 1) - 1) / SH + 1 (PadBottom is PH unless set).</summary>
    public int OH => (H + PH + PadBottom - DH * (KH - 1) - 1) / SH + 1;

    /// <summary>Output width: (W + PW + PadRight - DW·(KW - 1) - 1) / SW + 1 (PadRight is PW unless set).</summary>
    public int OW => (W + PW + PadRight - DW * (KW - 1) - 1) / SW + 1;

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
    /// <summary>
    /// Where the output rows go: 0 (the default) as the queries hold them, [heads, rowsPerHead, dim]; else the key/value
    /// heads per sequence, the output written token-major, [sequences, steps, heads·dim], what an output projection reads
    /// (<see cref="OutputRow"/>), so no copy rearranges the heads after attention. Given only to a device whose
    /// <see cref="Backend.MergesAttention"/> says it writes so; it changes no value, only where each row is written.
    /// </summary>
    public int Merged { get; init; }

    /// <summary>Plain causal attention: no window and no cap.</summary>
    public bool IsPlain => Window <= 0 && Softcap <= 0f;

    /// <summary>
    /// The output row query row <paramref name="row"/> (row i of head h: h · <paramref name="rowsPerHead"/> + i) is written
    /// to: itself, or with <see cref="Merged"/> (kv key/value heads per sequence, query heads sharing one stacked along the
    /// rows: i = g · steps + s), the row of step s of sequence b = h / kv at query head (h % kv) · group + g:
    /// ((b · steps + s) · kv + h % kv) · group + g, group = rowsPerHead / steps. With one step, each row is itself.
    /// </summary>
    public long OutputRow(long row, int rowsPerHead, int steps)
    {
        if (Merged <= 0)
        {
            return row;
        }

        long h = row / rowsPerHead;
        int i = (int)(row - h * rowsPerHead), group = rowsPerHead / steps, g = i / steps, s = i - g * steps;
        long b = h / Merged, kv = h - b * Merged;
        return ((b * steps + s) * Merged + kv) * group + g;
    }

    /// <summary>The first position a query whose causal limit is <paramref name="end"/> (exclusive) sees.</summary>
    public int Start(int end) => Window > 0 ? Math.Max(0, end - Window) : 0;

    /// <summary>A scaled score, soft-capped when a cap is set.</summary>
    public float Cap(float score) => Softcap > 0f ? Cpu.CpuMath.Tanh(score / Softcap) * Softcap : score;   // the CPU's own tanh (rule 70)

    /// <summary>The derivative of <see cref="Cap"/> given its result: 1 - (capped / cap)² (1 without a cap).</summary>
    public float Slope(float capped) => Softcap > 0f ? 1f - capped / Softcap * (capped / Softcap) : 1f;
}

/// <summary>
/// A reference-counted block of device memory holding <see cref="Length"/> floats: what a device's kernels read and write.
/// A device subclasses it to hold its own handle (a pointer, a buffer and offset) and creates it in
/// <see cref="Backend.Allocate(int, bool)"/>; when the last reference is released the block goes back to its device
/// (<see cref="Backend.Return"/>). A storage in memory the process reads directly (the CPU device's) implements
/// <see cref="IHostStorage"/>; any other is read and written with <see cref="Backend.Download"/> and
/// <see cref="Backend.Upload"/>.
/// </summary>
public abstract class Storage
{
    private int _refs = 1;
    private int _version;

    /// <summary>A storage of <paramref name="length"/> floats on <paramref name="backend"/> (recorded while <see cref="StorageTracking"/> is on).</summary>
    /// <param name="backend">The device the memory belongs to.</param>
    /// <param name="length">The number of floats.</param>
    protected Storage(Backend backend, int length)
    {
        Backend = backend;
        Length = length;
        if (StorageTracking.On)
        {
            StorageTracking.Created(this);
        }
    }

    /// <summary>The device the memory belongs to.</summary>
    public Backend Backend { get; }

    /// <summary>The number of floats.</summary>
    public int Length { get; }

    // The type of the values its words hold when it was allocated with one (Backend.Allocate(IElementType, ...)), and how
    // many values they are.
    private IElementType? _type;
    private long _elements = -1;

    /// <summary>
    /// The type of the values its <see cref="Length"/> 4-byte words hold: the type it was allocated with
    /// (<see cref="Backend.Allocate(IElementType, long, bool)"/>), whose <see cref="ElementCount"/> values the words hold as
    /// the type stores them; else the device's own float32 (<see cref="Types.F32"/>), one value a word.
    /// </summary>
    public IElementType Type => _type ?? Types.F32.Type;

    /// <summary>Whether it was allocated with a type (<see cref="Backend.Allocate(IElementType, long, bool)"/>), its words holding that type's stored bytes; false for the device's own floats.</summary>
    public bool Typed => _type is not null;

    /// <summary>
    /// The description of <see cref="Type"/> for a storage allocated with a type (<see cref="Typed"/>), which a device's
    /// described lane reads; null for a storage of the device's own floats.
    /// </summary>
    public TypeDescription? Description => _type?.Description;

    /// <summary>The number of values it holds: those of its <see cref="Type"/> where it was allocated with one, else <see cref="Length"/>.</summary>
    public long ElementCount => _elements >= 0 ? _elements : Length;

    /// <summary>The bytes of memory it holds: <see cref="Length"/> 4-byte words.</summary>
    public long ByteLength => (long)Length * sizeof(float);

    // Marks a storage just allocated as holding `count` values of `type` (its words hold them: Backend.Allocate).
    internal void SetType(IElementType type, long count)
    {
        _type = type;
        _elements = count;
    }

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

    /// <summary>Called once when the storage is released for good (for what its recomputation keeps).</summary>
    public Action? Released { get; internal set; }

    /// <summary>Whether a tensor still holds the storage (it has not been released for good).</summary>
    public bool Alive => Volatile.Read(ref _refs) > 0;

    // The views reading this storage through their layouts that were not copied yet (plan 37, V1), or null: a tensor that
    // hands this storage out for a direct access (Tensor.Storage, which may write it) copies them first, so they keep the
    // values they were made with, as the copies before views did.
    internal List<WeakReference<Tensor>>? PendingViews;

    // Its values were released while views still read them (ActivationMemory.Release, plan 37, V1b): evicted once the
    // last of those views is copied to its own or gone.
    internal bool DropWhenUnviewed;
}

/// <summary>
/// A device: its memory, copies and synchronization, what its kernels can do, and a kernel for each operation of the
/// device contract (plan 9). Every operation is a virtual method <c>NameKernel</c> whose default body is a composition
/// of
/// other operations, or "no such kernel" (it returns false and the caller takes another path, or it throws); a device
/// overrides the ones it runs itself.
/// Callers run an operation through <c>Name(...)</c> (<see cref="Fill"/>, <see cref="Softmax"/>, ...), which runs
/// <c>NameKernel</c>. What every device provides is <see cref="IDeviceBackend"/>; the optional capabilities are
/// interfaces
/// of their own (<see cref="IGraphCaptureBackend"/>, <see cref="IRecurrentKernels"/>, …),
/// read once when the backend starts: the operations they hold reach the device's kernels through them.
/// </summary>
/// <remarks>
/// Every operation works on contiguous row-major float32 buffers. Methods named "...Backward", <see cref="Axpy"/>,
/// <see cref="MulAdd"/>, <see cref="SumRows"/> and <see cref="AddBroadcastScalar"/> accumulate into their output (+=)
/// so
/// gradients from several paths add up.
/// </remarks>
[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicMethods)]   // OwnKernels reads the method each kernel's delegate binds
public abstract partial class Backend : IDeviceBackend, IDisposable
{
    // 1 once disposed (Dispose): every operation, allocation and copy then throws, naming the backend.
    private int _disposed;

    // Stored in _kernels once disposed: never null and never the guarded slots, so every generated call leaves the fast
    // path and enters an operation (EnterOperation), which throws.
    private static readonly Delegate?[] s_disposedKernels = [];

    /// <summary>
    /// Gives back what this backend holds on its device (memory, kernels, queues or streams, its share of the device's
    /// context), once; a second call does nothing. Afterwards every operation, allocation and copy on it throws
    /// <see cref="ObjectDisposedException"/> naming it, so a tensor that outlives the backend cannot touch freed memory
    /// (releasing such a tensor is still safe). The device's own backend (<see cref="Device.Backend"/>) lives for the process:
    /// disposing it does nothing. A separate backend (<see cref="Device.CreateSeparateBackend"/>) is its caller's to dispose.
    /// </summary>
    public void Dispose()
    {
        if (LivesForProcess || Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        Volatile.Write(ref _kernels, s_disposedKernels);
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Frees what the device gave this backend (<see cref="Dispose()"/> calls it once, after marking the backend disposed:
    /// the device's own calls inside it must not go through the checks that now throw). Nothing by default.
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
    }

    /// <summary>Whether this backend lives for the process (the device's own, <see cref="Device.Backend"/>): <see cref="Dispose()"/> then does nothing.</summary>
    protected virtual bool LivesForProcess => false;

    /// <summary>Whether <see cref="Dispose()"/> gave this backend's resources back.</summary>
    public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

    /// <summary>Throws <see cref="ObjectDisposedException"/> naming the backend when it was disposed.</summary>
    protected void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0)
        {
            ThrowDisposed();
        }
    }

    private void ThrowDisposed() =>
        throw new ObjectDisposedException(GetType().Name, $"The {Kind} backend '{Name}' was disposed: its memory and kernels were given back, so its storages and tensors can no longer be used.");

    // The kernel slots by operation index (Kernels.Resolve leaves every one empty), null when nothing is traced (every
    // call then runs the device's own kernel after one field read), or an empty array when they must be resolved again.
    private Delegate?[]? _kernels = Operations.Kernels.Unresolved();

    // The trace counting this device's calls (Kernels.Trace), or null.
    private KernelTrace? _trace;

    // Operations this device has run through the host fallback, read through Kernels.HostCalls.
    private long _hostCalls;

    // The operations whose first host fallback was reported to telemetry (HostFallback), made on the first report.
    private ConcurrentDictionary<string, bool>? _hostFallbacksReported;

    // The host fallbacks by operation counted while someone listens at TelemetryLevel.Devices (Kernels.HostCallsByOperation).
    private ConcurrentDictionary<string, long>? _hostCallsByOperation;

    // Which operations this device has a kernel of its own for, read on first use (per device type, OwnKernelsOf).
    private OwnKernelSet? _ownKernels;

    // The capabilities this device implements, read once when it starts (rule 75): the operations they hold reach the
    // device's kernels through these, and SupportsGraphs and FusedKernels follow from them.
    private readonly IGraphCaptureBackend? _graphs;
    private readonly IRowAttentionBackend? _rows;
    private readonly ISegmentedAttentionBackend? _segmented;
    private readonly IRecurrentKernels? _recurrent;
    private readonly IOffloadingBackend? _offloading;
    private readonly IPeerCopyBackend? _peers;
    private readonly IMemoryReport? _memoryReport;
    private readonly ILongLivedAllocations? _longLived;

    // Moves of live storages against the operations reading their addresses, or null: a device without offloading
    // (IOffloadingBackend) never moves a live storage, and its operations pay nothing for the guard.
    private readonly MoveGate? _moves;

    /// <summary>
    /// Starts a device: reads which capability interfaces it implements, and guards its moves when it offloads (every
    /// operation then runs as one, <see cref="EnterOperation"/>).
    /// </summary>
    protected Backend()
    {
        _graphs = this as IGraphCaptureBackend;
        _rows = this as IRowAttentionBackend;
        _segmented = this as ISegmentedAttentionBackend;
        _recurrent = this as IRecurrentKernels;
        _offloading = this as IOffloadingBackend;
        _peers = this as IPeerCopyBackend;
        _memoryReport = this as IMemoryReport;
        _longLived = this as ILongLivedAllocations;
        if (_offloading is not null)
        {
            _moves = new MoveGate();
        }
    }

    /// <summary>The kind of device this drives ("cpu", "cuda", "vulkan", …).</summary>
    public abstract string Kind { get; }

    // The default of an operation this device has no kernel of its own for.
    private NotSupportedException NoKernel(string operation) => new($"The {Kind} device '{Name}' has no kernel for {operation}.");

    /// <summary>Marks the kernel slots as changed: the next call to an operation resolves them again.</summary>
    internal void KernelsChanged()
    {
        Volatile.Write(ref _kernels, IsDisposed ? s_disposedKernels : Operations.Kernels.Unresolved());
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

    // RefusedKernelsFallBack: 1 on, 0 off, -1 not read yet (BuildsKernelsWhenUsed is virtual: read on first use).
    private int _refusedFallBack = -1;

    // IDRAK_REFUSED_KERNELS=throw: a kernel the driver refuses reaches the caller instead of falling back (to find the fault).
    private static readonly bool s_refusedThrow = Environment.GetEnvironmentVariable("IDRAK_REFUSED_KERNELS") is { } refused
        && refused.Trim().Equals("throw", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Whether this device builds its kernels when they are first used (a Vulkan driver compiles a pipeline on its first
    /// dispatch), so its driver may refuse one (<see cref="KernelUnavailableException"/>). On such a device, unless
    /// IDRAK_REFUSED_KERNELS=throw, an operation whose kernel is refused falls back instead of failing: the operation's
    /// composed path, or "no kernel" (the caller's own path), whichever its default is. With nothing else taking its calls
    /// off the inlined fast path (no trace, no guarded moves), each
    /// call runs the device's kernel through NameGuarded: one call whose only addition is a try region (free until it
    /// throws), so the first refusal is caught too; which operations the device owns is looked up only when one is refused.
    /// Off by default.
    /// </summary>
    protected virtual bool BuildsKernelsWhenUsed => false;

    /// <summary>
    /// Called once per operation and refused kernel when the dispatcher ran <paramref name="operation"/> through its
    /// <paramref name="fallback"/> ("its composed path", "the caller's path") because
    /// <paramref name="refused"/>'s kernel is unavailable here (<see cref="BuildsKernelsWhenUsed"/>): for the device to list
    /// what each refused kernel fell back to. Nothing by default.
    /// </summary>
    protected virtual void KernelFellBack(string operation, KernelUnavailableException refused, string fallback)
    {
    }

    // Whether a refused kernel falls back (BuildsKernelsWhenUsed, unless IDRAK_REFUSED_KERNELS=throw).
    private bool RefusedKernelsFallBack
    {
        get
        {
            int on = Volatile.Read(ref _refusedFallBack);
            if (on < 0)
            {
                on = BuildsKernelsWhenUsed && !s_refusedThrow ? 1 : 0;
                Volatile.Write(ref _refusedFallBack, on);
            }

            return on == 1;
        }
    }

    // The slots of a device whose only reason to leave the fast path is RefusedKernelsFallBack (no trace, no guarded
    // moves): every slot empty, and the generated Name(...) sends its calls to NameGuarded
    // directly when it finds this instance. Stored in _kernels like the slots it stands for, so a change of the kernels
    // (KernelsChanged) ends it atomically.
    private static readonly Delegate?[] s_guardedKernels = new Delegate?[OperationIndex.Count];

    // Reports a refused kernel's fallback once per operation and kernel (KernelFellBack).
    private void FellBack(string operation, KernelUnavailableException refused, string fallback)
    {
        if (LazyInitializer.EnsureInitialized(ref _fellBack, static () => new(StringComparer.Ordinal)).TryAdd(operation + "\0" + refused.Kernel, true))
        {
            KernelFellBack(operation, refused, fallback);
        }
    }

    private ConcurrentDictionary<string, bool>? _fellBack;

    // Whether this device has a kernel of its own for the operation (a refused one falls back only then).
    private bool OwnsKernel(int operation) => (_ownKernels ??= Operations.Kernels.OwnKernels(this)).Has(operation);

    /// <summary>
    /// The library's operations (<see cref="Ops"/>) this device has a kernel of its own for: its <c>NameKernel</c> overrides
    /// <see cref="Backend"/>'s, or the operation belongs to a capability interface the device implements
    /// (<see cref="IRecurrentKernels.LstmCell"/> for <see cref="Ops.LstmCell"/>, say). Built once per device type, from
    /// delegates bound to the device's kernels (no method is looked up by name); what <see cref="Operations.Kernels.Chain"/>
    /// reads.
    /// </summary>
    public IReadOnlySet<Operation> OwnKernels => _ownKernels ??= Operations.Kernels.OwnKernels(this);

    // Whether a kernel bound to this device is its own: the device's type (or a type between it and Backend) declares it.
    private static bool Overrides(Delegate kernel) => kernel.Method.DeclaringType != typeof(Backend);

    /// <summary>Operations run through the host fallback so far.</summary>
    internal long HostCalls => Interlocked.Read(ref _hostCalls);

    /// <summary>The host fallbacks by operation counted while someone listened at <see cref="TelemetryLevel.Devices"/>.</summary>
    internal IReadOnlyDictionary<string, long> HostCallsByOperation =>
        Volatile.Read(ref _hostCallsByOperation) is { } counts ? new Dictionary<string, long>(counts, StringComparer.Ordinal) : new Dictionary<string, long>(StringComparer.Ordinal);

    /// <summary>
    /// Counts one host fallback of the operation named <paramref name="operation"/>, and reports its first to telemetry.
    /// </summary>
    internal void HostCalled(string operation)
    {
        Interlocked.Increment(ref _hostCalls);
        Volatile.Read(ref _trace)?.HostCalled(operation);
        if (Telemetry.IsEnabled(TelemetryLevel.Devices))
        {
            LazyInitializer.EnsureInitialized(ref _hostCallsByOperation, static () => new(StringComparer.Ordinal)).AddOrUpdate(operation, 1, static (_, n) => n + 1);
            if (LazyInitializer.EnsureInitialized(ref _hostFallbacksReported, static () => new(StringComparer.Ordinal)).TryAdd(operation, true))
            {
                // once per operation, while someone listens; with why the kernels are off when they all are
                Telemetry.HostFallback(new HostFallback(Kind, operation, (this as IKernelAvailability)?.KernelsUnavailable()));
            }
        }
    }

    [ThreadStatic]
    private static int t_tuningDepth;

    /// <summary>
    /// Marks the start of a tuning this device measures: the timestamp <see cref="TuningMeasured"/> takes, the depth counted
    /// so a tuning a candidate's run starts itself reports as nested.
    /// </summary>
    protected static long TuningBegan()
    {
        t_tuningDepth++;
        return Stopwatch.GetTimestamp();
    }

    /// <summary>
    /// Reports a tuning this device measured (<see cref="TuningCompleted"/>: <paramref name="operation"/> at a new shape
    /// class, <paramref name="candidates"/> choices timed since <paramref name="began"/>) to telemetry, while someone listens.
    /// </summary>
    protected void TuningMeasured(string operation, int candidates, long began)
    {
        t_tuningDepth = Math.Max(0, t_tuningDepth - 1);
        if (Telemetry.IsEnabled(TelemetryLevel.Devices))
        {
            Telemetry.Tuning(new TuningCompleted(Kind, operation, candidates, Stopwatch.GetElapsedTime(began), t_tuningDepth > 0));
        }
    }

    // A call off the fast path: counted when traced, and the path the next calls take resolved when it is not yet.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void Traced(int operation)
    {
        Volatile.Read(ref _trace)?.Called(operation);
        if (_kernels is { Length: 0 } unresolved)
        {
            _ = ResolveKernels(unresolved);
        }
    }

    // Resolves the slots; stores them unless Kernels changed meanwhile (then the next call resolves again). With every
    // slot empty and no trace the slots are dropped, so calls take the fast path again; or, on a
    // device whose refused kernels fall back, replaced by s_guardedKernels, so calls take NameGuarded (unless the device
    // guards its moves: then every call is an operation, NameScoped, which runs NameGuarded inside it).
    private Delegate?[] ResolveKernels(Delegate?[] unresolved)
    {
        ThrowIfDisposed();                                                 // a disposed backend's sentinel is never resolved
        var slots = Operations.Kernels.Resolve(this);
        bool direct = Array.TrueForAll(slots, k => k is null) && Volatile.Read(ref _trace) is null;
        var kept = !direct ? slots : !RefusedKernelsFallBack ? null : _moves is null ? s_guardedKernels : slots;
        Interlocked.CompareExchange(ref _kernels, kept, unresolved);
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

    /// <summary>Whether this device can record and replay work as a graph now (<see cref="IGraphCaptureBackend"/>).</summary>
    public bool SupportsGraphs => _graphs is { } graphs && graphs.CaptureEnabled;

    /// <summary>
    /// Whether this device's tiled attention <paramref name="operation"/> (<see cref="Operations.Ops.AttentionTiled"/>) writes its
    /// output token-major when its <see cref="AttentionVariant.Merged"/> asks, for head size <paramref name="dim"/> and the
    /// <paramref name="variant"/>'s window and cap: only then is it asked.
    /// None does by default; the attention's caller then rearranges the heads itself.
    /// </summary>
    public virtual bool MergesAttention(Operation operation, int dim, AttentionVariant variant) => false;

    /// <summary>What the device reports about its hardware, for listings and diagnostics; empty when it reports nothing.</summary>
    public virtual BackendHardware Hardware => new();

    /// <summary>Keeping tensors in system memory when the device is full (<see cref="IOffloadingBackend"/>), or null when the device has no such support.</summary>
    public IMemoryOffload? Offload => _offloading?.Offload;

    /// <summary>
    /// The scope of an operation that reads storages' addresses outside the operations of this class (uploads, downloads,
    /// copies, a recording, an app's kernel): from reading the addresses until the work using them is queued. On a device
    /// that offloads (<see cref="IOffloadingBackend"/>), a move of live storages (<see cref="BeginMove"/>) waits until no
    /// other thread is inside one, so no thread queues work with an address a move has given up meanwhile; free on any
    /// other device.
    /// </summary>
    protected OperationScope EnterOperation()
    {
        ThrowIfDisposed();
        _moves?.Enter();
        return new OperationScope(_moves);
    }

    /// <summary>
    /// Starts a move of live storages (a device that offloads, <see cref="IOffloadingBackend"/>): true once no other thread is inside an operation
    /// (end it with <see cref="EndMove"/>); false when the move may not run now. With <paramref name="wait"/> false it
    /// never waits: it runs only if no other thread is inside an operation (a caller holding a lock those operations may
    /// need passes false). From inside an operation a move never waits either, except a placement of lookup tables
    /// (<paramref name="lookupTablesOnly"/>: storages <see cref="IMemoryOffload.MarkLookupTable"/> marked, which no kernel
    /// reads before an allocation of its own): it waits for the other operations, a general move still waiting for this
    /// thread gives way to it, and while another thread places lookup tables this one steps aside and gets false (its
    /// allocation tries again). Moves by one thread nest. Always true on a device that does not guard its moves.
    /// </summary>
    protected bool BeginMove(bool wait, bool lookupTablesOnly = false) => _moves is null || _moves.BeginMove(wait, lookupTablesOnly);

    /// <summary>Ends a move started by <see cref="BeginMove"/> (when it returned true).</summary>
    protected void EndMove() => _moves?.EndMove();

    /// <summary>
    /// Whether the addresses this thread reads stay valid: it is inside an operation or a move, or the device does not
    /// offload (<see cref="IOffloadingBackend"/>).
    /// </summary>
    protected bool HoldsAddresses => _moves is null || _moves.Holds;

    /// <summary>An operation's scope (<see cref="EnterOperation"/>): disposing it ends the operation.</summary>
    protected readonly struct OperationScope : IDisposable
    {
        private readonly MoveGate? _gate;

        internal OperationScope(MoveGate? gate) => _gate = gate;

        /// <summary>Ends the operation.</summary>
        public void Dispose() => _gate?.Exit();
    }

    /// <summary>
    /// Until the result is disposed, this thread's allocations on the device are long-lived (weights, attention caches):
    /// the device may place them together (<see cref="ILongLivedAllocations"/>). Null on devices that place every block
    /// alike. Scopes nest.
    /// </summary>
    public IDisposable? LongLivedAllocations() => _longLived?.LongLivedAllocations();

    /// <summary>Pinned buffers for background copies to and from system memory, or null when the device has none (see <see cref="IHostStaging"/>).</summary>
    public virtual IHostStaging? CreateHostStaging(int slots, int slotFloats) => null;

    /// <summary>
    /// Copies the first <paramref name="count"/> values of <paramref name="source"/> (on this backend) into
    /// <paramref name="destination"/>, on this backend or any other (plan 23: a copy between the devices of two ranks). The
    /// default copies inside the device when both are this backend, else through host memory (a download, then an upload
    /// to the destination's device: the path every machine has); a backend that reaches another device's memory
    /// directly (peer copies, <see cref="IPeerCopyBackend"/>) copies there itself. The copy has finished when the call
    /// returns.
    /// </summary>
    /// <exception cref="ArgumentException">The source is not on this backend, or a storage holds fewer than <paramref name="count"/> values.</exception>
    public void CopyTo(Storage source, Storage destination, int count)
    {
        CheckCopy(source, destination, count);
        if (count == 0)
        {
            return;
        }

        if (ReferenceEquals(destination.Backend, this))
        {
            Copy(source, destination, count);
            Synchronize();
            return;
        }

        if (_peers is { } peers && peers.CopiesDirectlyTo(destination.Backend))
        {
            peers.CopyTo(source, destination, count);
            return;
        }

        CopyThroughHost(source, destination, count);
    }

    /// <summary>
    /// Whether <see cref="CopyTo"/> reaches <paramref name="destination"/>'s memory without a trip through host memory: true
    /// for this backend itself, and where a backend says so (<see cref="IPeerCopyBackend"/>: another backend of the same
    /// device, a peer device).
    /// </summary>
    public bool CopiesDirectlyTo(Backend destination) => ReferenceEquals(destination, this) || _peers is { } peers && peers.CopiesDirectlyTo(destination);

    /// <summary>The copy of <see cref="CopyTo"/> through host memory: one pooled host buffer (returned in finally), down, then up.</summary>
    protected void CopyThroughHost(Storage source, Storage destination, int count)
    {
        CheckCopy(source, destination, count);
        if (count == 0)
        {
            return;
        }

        var host = System.Buffers.ArrayPool<float>.Shared.Rent(count);
        try
        {
            DownloadRange(source, 0, host.AsSpan(0, count));
            destination.Backend.Upload(host.AsSpan(0, count), destination);
            destination.Backend.Synchronize();
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(host);
        }
    }

    private void CheckCopy(Storage source, Storage destination, int count)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destination);
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        if (!ReferenceEquals(source.Backend, this))
        {
            throw new ArgumentException($"The source is on {source.Backend.Name}, not on this backend ({Name}).", nameof(source));
        }

        if (count > source.Length || count > destination.Length)
        {
            throw new ArgumentException($"A copy of {count} values from a storage of {source.Length} into one of {destination.Length}.", nameof(count));
        }
    }

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

        var fresh = Allocate(storage.Length, zeroed: false);
        Attach(storage, fresh);
        if (StorageTracking.On)
        {
            StorageTracking.Moved(fresh);                          // its memory is the restored storage's now
        }

        storage.Evicted = false;
        storage.Recompute!(storage);
        return true;
    }

    /// <summary>
    /// Sets what restores <paramref name="storage"/>'s values from now on: for an evicted storage, its recomputation is
    /// replaced (a deferred load's read from a checkpoint becomes a gather from the rank owning it, say); a storage in memory
    /// is evicted with it (<see cref="Evict"/>).
    /// </summary>
    public void SetRecompute(Storage storage, Action<Storage>? recompute)
    {
        ArgumentNullException.ThrowIfNull(storage);
        if (storage.Evicted)
        {
            storage.Recompute = recompute;
            return;
        }

        Evict(storage, recompute);
    }

    /// <summary>
    /// Points <paramref name="storage"/> at no memory, after <see cref="Evict"/> gave its memory back with
    /// <see cref="Return"/>; kernels given it fail until <see cref="Attach"/>.
    /// </summary>
    protected abstract void Detach(Storage storage);

    /// <inheritdoc />
    void IDeviceBackend.Detach(Storage storage) => Detach(storage);

    /// <summary>
    /// Points <paramref name="storage"/> (evicted) at the memory of <paramref name="fresh"/>, a block just allocated for
    /// it, which is not used on its own afterwards.
    /// </summary>
    protected abstract void Attach(Storage storage, Storage fresh);

    /// <inheritdoc />
    void IDeviceBackend.Attach(Storage storage, Storage fresh) => Attach(storage, fresh);

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

    /// <summary>
    /// Copies <paramref name="source"/> into floats [<paramref name="offset"/>, offset + source.Length) of
    /// <paramref name="destination"/>. Devices override it with a copy straight to the offset; the default goes through a
    /// block of the source's size taken from the pool and given back at once (nothing kept), then a copy into place.
    /// </summary>
    public virtual void UploadRange(ReadOnlySpan<float> source, Storage destination, int offset)
    {
        ArgumentNullException.ThrowIfNull(destination);
        if ((uint)offset > (uint)destination.Length || source.Length > destination.Length - offset)
        {
            throw new ArgumentOutOfRangeException(nameof(offset), $"Floats {offset} to {offset + source.Length} of a storage of {destination.Length}.");
        }

        if (source.IsEmpty)
        {
            return;
        }

        if (offset == 0)
        {
            Upload(source, destination);
            return;
        }

        var block = Allocate(source.Length, zeroed: false);
        try
        {
            Upload(source, block);
            Copy2D(block, 0, source.Length, destination, offset, source.Length, 1, source.Length, accumulate: false);
        }
        finally
        {
            block.Release();
        }
    }

    /// <summary>
    /// The storages this device holds, largest first, with who holds each (a debug listing: only storages made while
    /// <see cref="StorageTracking"/> is on are known; others are in <see cref="GetMemoryUsage"/>'s bytes in use but not
    /// here). A device may add blocks of its own (scratch it keeps) by overriding it.
    /// </summary>
    /// <param name="name">Names a storage's owner where the caller knows it (a parameter's name), or null.</param>
    public virtual IReadOnlyList<LiveStorage> LiveStorages(Func<Storage, string?>? name = null) => StorageTracking.List(this, name);

    /// <summary>y[i] = value for i &lt; n.</summary>
    public virtual void FillKernel(Storage y, int n, float value)
    {
        using var h = new HostCall(this, nameof(Fill));
        Cpu.CpuBackend.Instance.Fill(h[y], n, value);
    }

    /// <summary>y[i] = x[i] for i &lt; n (a copy within the device).</summary>
    public virtual void Copy(Storage x, Storage y, int n)
    {
        using var h = new HostCall(this, nameof(Copy));
        Cpu.CpuBackend.Instance.Copy(h[x], h[y], n);
    }

    /// <summary>y = op(x).</summary>
    public virtual void UnaryKernel(UnaryOp op, Storage x, Storage y, int n)
    {
        using var h = new HostCall(this, nameof(Unary));
        Cpu.CpuBackend.Instance.Unary(op, h[x], h[y], n);
    }

    /// <summary>dx += dy * op'(x), where y = op(x) is passed in for ops whose derivative is cheaper from y.</summary>
    public virtual void UnaryBackwardKernel(UnaryOp op, Storage x, Storage y, Storage dy, Storage dx, int n)
    {
        using var h = new HostCall(this, nameof(UnaryBackward));
        Cpu.CpuBackend.Instance.UnaryBackward(op, h[x], h[y], h[dy], h[dx], n);
    }

    /// <summary>c = a op b, element-wise.</summary>
    public virtual void BinaryKernel(BinaryOp op, Storage a, Storage b, Storage c, int n)
    {
        using var h = new HostCall(this, nameof(Binary));
        Cpu.CpuBackend.Instance.Binary(op, h[a], h[b], h[c], n);
    }

    /// <summary>
    /// dx += dy where a wins (<see cref="BinaryOp.Maximum"/>: a ≥ b; <see cref="BinaryOp.Minimum"/>: a ≤ b), db += dy
    /// elsewhere: the backward step of c = max(a, b) or min(a, b). A null gradient is skipped.
    /// </summary>
    public virtual void ExtremumBackwardKernel(BinaryOp op, Storage a, Storage b, Storage dy, Storage? da, Storage? db, int n)
    {
        using var h = new HostCall(this, nameof(ExtremumBackward));
        Cpu.CpuBackend.Instance.ExtremumBackward(op, h[a], h[b], h[dy], h.Maybe(da), h.Maybe(db), n);
    }

    /// <summary>y = x^exponent, element-wise.</summary>
    public virtual void PowKernel(Storage x, Storage y, int n, float exponent)
    {
        using var h = new HostCall(this, nameof(Pow));
        Cpu.CpuBackend.Instance.Pow(h[x], h[y], n, exponent);
    }

    /// <summary>dx += dy · exponent · x^(exponent - 1).</summary>
    public virtual void PowBackwardKernel(Storage x, Storage dy, Storage dx, int n, float exponent)
    {
        using var h = new HostCall(this, nameof(PowBackward));
        Cpu.CpuBackend.Instance.PowBackward(h[x], h[dy], h[dx], n, exponent);
    }

    /// <summary>y = x limited to [min, max] (either may be infinite).</summary>
    public virtual void ClampKernel(Storage x, Storage y, int n, float min, float max)
    {
        using var h = new HostCall(this, nameof(Clamp));
        Cpu.CpuBackend.Instance.Clamp(h[x], h[y], n, min, max);
    }

    /// <summary>dx += dy where min ≤ x ≤ max.</summary>
    public virtual void ClampBackwardKernel(Storage x, Storage dy, Storage dx, int n, float min, float max)
    {
        using var h = new HostCall(this, nameof(ClampBackward));
        Cpu.CpuBackend.Instance.ClampBackward(h[x], h[dy], h[dx], n, min, max);
    }

    /// <summary>y = x where x &gt; 0, slope · x elsewhere: the leaky ReLU, for a slope in [0, 1).</summary>
    public virtual void LeakyReluKernel(Storage x, Storage y, int n, float slope)
    {
        using var h = new HostCall(this, nameof(LeakyRelu));
        Cpu.CpuBackend.Instance.LeakyRelu(h[x], h[y], n, slope);
    }

    /// <summary>dx += dy where x &gt; 0, slope · dy elsewhere.</summary>
    public virtual void LeakyReluBackwardKernel(Storage x, Storage dy, Storage dx, int n, float slope)
    {
        using var h = new HostCall(this, nameof(LeakyReluBackward));
        Cpu.CpuBackend.Instance.LeakyReluBackward(h[x], h[dy], h[dx], n, slope);
    }

    /// <summary>y = condition ≠ 0 ? a : b, element-wise.</summary>
    public virtual void WhereKernel(Storage condition, Storage a, Storage b, Storage y, int n)
    {
        using var h = new HostCall(this, nameof(Where));
        Cpu.CpuBackend.Instance.Where(h[condition], h[a], h[b], h[y], n);
    }

    /// <summary>da += dy where condition ≠ 0, db += dy elsewhere (a null gradient is skipped).</summary>
    public virtual void WhereBackwardKernel(Storage condition, Storage dy, Storage? da, Storage? db, int n)
    {
        using var h = new HostCall(this, nameof(WhereBackward));
        Cpu.CpuBackend.Instance.WhereBackward(h[condition], h[dy], h.Maybe(da), h.Maybe(db), n);
    }

    /// <summary>y = alpha * x + beta.</summary>
    public virtual void AffineKernel(Storage x, Storage y, int n, float alpha, float beta)
    {
        using var h = new HostCall(this, nameof(Affine));
        Cpu.CpuBackend.Instance.Affine(h[x], h[y], n, alpha, beta);
    }

    /// <summary>y += alpha * x.</summary>
    public virtual void AxpyKernel(Storage x, Storage y, int n, float alpha)
    {
        using var h = new HostCall(this, nameof(Axpy));
        Cpu.CpuBackend.Instance.Axpy(h[x], h[y], n, alpha);
    }

    /// <summary>c += a * b, element-wise.</summary>
    public virtual void MulAddKernel(Storage a, Storage b, Storage c, int n)
    {
        using var h = new HostCall(this, nameof(MulAdd));
        Cpu.CpuBackend.Instance.MulAdd(h[a], h[b], h[c], n);
    }

    /// <summary>c[r, j] = a[r, j] + v[j].</summary>
    public virtual void AddRowVectorKernel(Storage a, Storage v, Storage c, int rows, int cols)
    {
        using var h = new HostCall(this, nameof(AddRowVector));
        Cpu.CpuBackend.Instance.AddRowVector(h[a], h[v], h[c], rows, cols);
    }

    /// <summary>y[j] += sum over r of x[r, j].</summary>
    public virtual void SumRowsKernel(Storage x, Storage y, int rows, int cols)
    {
        using var h = new HostCall(this, nameof(SumRows));
        Cpu.CpuBackend.Instance.SumRows(h[x], h[y], rows, cols);
    }

    /// <summary>result[0] = scale * sum(x).</summary>
    public virtual void SumKernel(Storage x, Storage result, int n, float scale)
    {
        using var h = new HostCall(this, nameof(Sum));
        Cpu.CpuBackend.Instance.Sum(h[x], h[result], n, scale);
    }

    /// <summary>y[offset] += alpha * x[0] (used to accumulate scalar losses without leaving the device).</summary>
    public virtual void AxpyAtKernel(Storage x, Storage y, int offset, float alpha)
    {
        using var h = new HostCall(this, nameof(AxpyAt));
        Cpu.CpuBackend.Instance.AxpyAt(h[x], h[y], offset, alpha);
    }

    /// <summary>y[i] += scale * s[0].</summary>
    public virtual void AddBroadcastScalarKernel(Storage s, Storage y, int n, float scale)
    {
        using var h = new HostCall(this, nameof(AddBroadcastScalar));
        Cpu.CpuBackend.Instance.AddBroadcastScalar(h[s], h[y], n, scale);
    }

    /// <summary>
    /// c[m, n] = op(a) * op(b) + beta * c, where op(a) is [m, k] and op(b) is [k, n].
    /// With <paramref name="transA"/> a is stored as [k, m]; with <paramref name="transB"/> b is stored as [n, k].
    /// </summary>
    public void MatMul(Storage a, Storage b, Storage c, int m, int n, int k, bool transA, bool transB, float beta) =>
        BatchedMatMul(a, b, c, 1, m, n, k, transA, transB, beta);

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

    /// <summary>y[j] += Σ_r x[offset + r·ld + j] for j &lt; cols (column sums of a strided block; the bias gradient of a slice).</summary>
    public virtual void SumColumnsKernel(Storage x, long offset, int ld, Storage y, int rows, int cols)
    {
        using var h = new HostCall(this, nameof(SumColumns));
        Cpu.CpuBackend.Instance.SumColumns(h[x], offset, ld, h[y], rows, cols);
    }

    /// <summary>output = residual + dropout(x) (the mask of <see cref="Dropout"/> with <paramref name="seed"/>).</summary>
    public virtual void AddDropoutKernel(Storage residual, Storage x, Storage output, int n, float p, uint seed)
    {
        Dropout(x, output, n, p, seed);
        Axpy(residual, output, n, 1f);
    }

    /// <summary><see cref="MatMul"/> for <paramref name="batch"/> independent, contiguous matrix triples.</summary>
    public virtual void BatchedMatMulKernel(Storage a, Storage b, Storage c, int batch, int m, int n, int k, bool transA, bool transB, float beta)
    {
        using var h = new HostCall(this, nameof(BatchedMatMul));
        Cpu.CpuBackend.Instance.BatchedMatMul(h[a], h[b], h[c], batch, m, n, k, transA, transB, beta);
    }

    /// <summary>Row-wise softmax (or log-softmax) over the last dimension: y[r, :] = softmax(x[r, :]).</summary>
    public virtual void SoftmaxKernel(Storage x, Storage y, int rows, int cols, bool log)
    {
        using var h = new HostCall(this, nameof(Softmax));
        Cpu.CpuBackend.Instance.Softmax(h[x], h[y], rows, cols, log);
    }

    /// <summary>
    /// Softmax: dx += y * (dy - Σ dy·y). Log-softmax: dx += dy - exp(y) * Σ dy. Sums are per row; y is the forward output.
    /// </summary>
    public virtual void SoftmaxBackwardKernel(Storage y, Storage dy, Storage dx, int rows, int cols, bool log)
    {
        using var h = new HostCall(this, nameof(SoftmaxBackward));
        Cpu.CpuBackend.Instance.SoftmaxBackward(h[y], h[dy], h[dx], rows, cols, log);
    }

    /// <summary>y[r] = index of the largest element of row r.</summary>
    public virtual void ArgMaxKernel(Storage x, Storage y, int rows, int cols)
    {
        using var h = new HostCall(this, nameof(ArgMax));
        Cpu.CpuBackend.Instance.ArgMax(h[x], h[y], rows, cols);
    }

    /// <summary>
    /// The best k log-probabilities of each row of x [rows, cols] and their columns (a beam search's candidates, read back
    /// instead of the whole rows): with shift = max + log Σ exp(x - max) over the row (the max ignoring NaN, the sum the
    /// entries other than -∞), each entry's log-probability is x - shift in float32; values[r, j] is the row's j-th best of
    /// those that are neither NaN nor -∞ (larger first, the smaller column first on equal values) and indices[r, j] its column
    /// (in the device's index type), j &lt; k; past the row's last such entry values gets -∞ and indices -1. A row with no finite logit, a
    /// NaN or a +∞ one has a NaN shift, so none. The CPU sums in double (its shift rounded to float32 once); a device's sum
    /// may differ from it by float rounding.
    /// </summary>
    public virtual void LogSoftmaxTopKKernel(Storage x, Storage values, Storage indices, int rows, int cols, int k)
    {
        using var h = new HostCall(this, nameof(LogSoftmaxTopK));
        Cpu.CpuBackend.Instance.LogSoftmaxTopK(h[x], h[values], h[indices], rows, cols, k);
    }

    /// <summary>
    /// y[r] = 1 when the prediction in row r is right, else 0. For one column: (p ≥ threshold) == (t ≥ 0.5);
    /// otherwise argmax(p) == argmax(t).
    /// </summary>
    public virtual void ClassMatchKernel(Storage predictions, Storage targets, Storage y, int rows, int cols, float threshold)
    {
        using var h = new HostCall(this, nameof(ClassMatch));
        Cpu.CpuBackend.Instance.ClassMatch(h[predictions], h[targets], h[y], rows, cols, threshold);
    }

    // Grouped normalization. Data is viewed as [outer, groups, inner]; group g holds the M = outer * inner
    // elements x[o, g, i]. BatchNorm uses groups = channels; LayerNorm uses outer = 1, groups = rows, inner = features.

    /// <summary>Per-group mean, (biased) variance and 1 / sqrt(variance + eps).</summary>
    public virtual void NormStatsKernel(Storage x, Storage mean, Storage variance, Storage invStd, int outer, int groups, int inner, float eps)
    {
        using var h = new HostCall(this, nameof(NormStats));
        Cpu.CpuBackend.Instance.NormStats(h[x], h[mean], h[variance], h[invStd], outer, groups, inner, eps);
    }

    /// <summary>y = (x - mean[g]) * invStd[g].</summary>
    public virtual void NormApplyKernel(Storage x, Storage mean, Storage invStd, Storage y, int outer, int groups, int inner)
    {
        using var h = new HostCall(this, nameof(NormApply));
        Cpu.CpuBackend.Instance.NormApply(h[x], h[mean], h[invStd], h[y], outer, groups, inner);
    }

    /// <summary>dx += invStd[g] / M * (M * dxhat - sum1[g] - xhat * sum2[g]).</summary>
    public virtual void NormBackwardKernel(Storage dxhat, Storage xhat, Storage sum1, Storage sum2, Storage invStd, Storage dx, int outer, int groups, int inner)
    {
        using var h = new HostCall(this, nameof(NormBackward));
        Cpu.CpuBackend.Instance.NormBackward(h[dxhat], h[xhat], h[sum1], h[sum2], h[invStd], h[dx], outer, groups, inner);
    }

    /// <summary>
    /// y (+)= x * scale[g] + shift[g] with g = (i / inner) % groups; a null scale means 1, a null shift means 0.
    /// </summary>
    public virtual void GroupScaleShiftKernel(Storage x, Storage? scale, Storage? shift, Storage y, int n, int groups, int inner, bool accumulate)
    {
        using var h = new HostCall(this, nameof(GroupScaleShift));
        Cpu.CpuBackend.Instance.GroupScaleShift(h[x], h.Maybe(scale), h.Maybe(shift), h[y], n, groups, inner, accumulate);
    }

    /// <summary>sumA[g] += Σ a; sumAB[g] += Σ a·b over each group (see NormStats for the layout). b/sumAB may be null.</summary>
    public virtual void GroupReduceKernel(Storage a, Storage? b, Storage sumA, Storage? sumAB, int outer, int groups, int inner)
    {
        using var h = new HostCall(this, nameof(GroupReduce));
        Cpu.CpuBackend.Instance.GroupReduce(h[a], h.Maybe(b), h[sumA], h.Maybe(sumAB), outer, groups, inner);
    }

    /// <summary>
    /// Weight normalization along the first axis: weight[r, :] = gain[r] · direction[r, :] / ‖direction[r, :]‖ over rows rows
    /// of columns elements (the row norm and the scaling in one pass; no squared copy or row sums kept).
    /// </summary>
    public virtual void WeightNormKernel(Storage gain, Storage direction, Storage weight, int rows, int columns)
    {
        using var h = new HostCall(this, nameof(WeightNorm));
        Cpu.CpuBackend.Instance.WeightNorm(h[gain], h[direction], h[weight], rows, columns);
    }

    /// <summary>
    /// The gradients of WeightNormKernel from dweight: with n = ‖direction[r, :]‖ and dot = Σ dweight · direction over row r,
    /// dgain[r] += dot / n and ddirection[r, :] += gain[r] / n · (dweight − direction · dot / n²). Either gradient may be null.
    /// </summary>
    public virtual void WeightNormBackwardKernel(Storage gain, Storage direction, Storage dweight, Storage? dgain, Storage? ddirection, int rows, int columns)
    {
        using var h = new HostCall(this, nameof(WeightNormBackward));
        Cpu.CpuBackend.Instance.WeightNormBackward(h[gain], h[direction], h[dweight], h.Maybe(dgain), h.Maybe(ddirection), rows, columns);
    }

    /// <summary>y = 1 / sqrt(x + eps).</summary>
    public virtual void InvSqrtKernel(Storage x, Storage y, int n, float eps)
    {
        using var h = new HostCall(this, nameof(InvSqrt));
        Cpu.CpuBackend.Instance.InvSqrt(h[x], h[y], n, eps);
    }

    /// <summary>Embedding lookup: y[i, :] = table[indices[i], :] for count indices of width dim.</summary>
    public virtual void GatherKernel(Storage table, Storage indices, Storage y, int count, int dim, int vocabulary)
    {
        using var h = new HostCall(this, nameof(Gather));
        Cpu.CpuBackend.Instance.Gather(h[table], h[indices], h[y], count, dim, vocabulary);
    }

    /// <summary>One-hot rows: y[i, :] = 0 except y[i, indices[i]] = 1, for count indices over classes columns.</summary>
    public virtual void OneHotKernel(Storage indices, Storage y, int count, int classes)
    {
        using var h = new HostCall(this, nameof(OneHot));
        Cpu.CpuBackend.Instance.OneHot(h[indices], h[y], count, classes);
    }

    /// <summary>dtable[indices[i], :] += dy[i, :].</summary>
    public virtual void ScatterAddKernel(Storage dy, Storage indices, Storage dtable, int count, int dim, int vocabulary)
    {
        using var h = new HostCall(this, nameof(ScatterAdd));
        Cpu.CpuBackend.Instance.ScatterAdd(h[dy], h[indices], h[dtable], count, dim, vocabulary);
    }

    /// <summary>Unfolds image patches: cols[(n, oh, ow), (c, kh, kw)] = x[n, c, oh*sh - ph + kh*dh, ow*sw - pw + kw*dw] (0 outside; dh, dw the dilation).</summary>
    public virtual void Im2ColKernel(Storage x, Storage cols, in ConvGeometry g)
    {
        using var h = new HostCall(this, nameof(Im2Col));
        Cpu.CpuBackend.Instance.Im2Col(h[x], h[cols], g);
    }

    /// <summary>The adjoint of <see cref="Im2Col"/>: dx += fold(dcols).</summary>
    public virtual void Col2ImKernel(Storage dcols, Storage dx, in ConvGeometry g)
    {
        using var h = new HostCall(this, nameof(Col2Im));
        Cpu.CpuBackend.Instance.Col2Im(h[dcols], h[dx], g);
    }

    /// <summary>Max pooling; argmax receives the flat input index of each maximum (as raw int bits).</summary>
    public virtual void MaxPoolKernel(Storage x, Storage y, Storage argmax, in ConvGeometry g)
    {
        using var h = new HostCall(this, nameof(MaxPool));
        Cpu.CpuBackend.Instance.MaxPool(h[x], h[y], h[argmax], g);
    }

    /// <summary>dx[argmax[i]] += dy[i].</summary>
    public virtual void MaxPoolBackwardKernel(Storage dy, Storage argmax, Storage dx, int count)
    {
        using var h = new HostCall(this, nameof(MaxPoolBackward));
        Cpu.CpuBackend.Instance.MaxPoolBackward(h[dy], h[argmax], h[dx], count);
    }

    /// <summary>
    /// dx[argmax[i]] += dy[i] for the windows of <paramref name="g"/> (count = N * C * OH * OW): the same as the overload
    /// taking a count; a device that adds the gradients by gathering over the windows needs the geometry.
    /// </summary>
    public virtual void MaxPoolBackwardKernel(Storage dy, Storage argmax, Storage dx, in ConvGeometry g) =>
        MaxPoolBackward(dy, argmax, dx, g.N * g.C * g.OH * g.OW);

    /// <summary>
    /// 2-D convolution of NCHW images: y[n, f, oh, ow] = act(bias[f] + Σ_{c, kh, kw} weight[f, (c·KH + kh)·KW + kw] ·
    /// x[n, q·Cg + c, oh·SH - PH + kh·DH, ow·SW - PW + kw·DW]) with 0 outside the image, where filter f is in group
    /// q = f / (filters / groups) and reads that group's Cg = C / groups channels. weight is [filters, Cg·KH·KW]
    /// (PyTorch's [filters, C / groups, KH, KW]), bias [filters] or null, y [N, filters, OH, OW]; act is
    /// <paramref name="activation"/> (the identity for <see cref="ConvActivation.None"/>).
    /// </summary>
    public virtual void ConvolutionKernel(Storage x, Storage weight, Storage? bias, Storage y, in ConvGeometry g, int filters, int groups, ConvActivation activation) =>
        ComposedConvolution(x, weight, bias, y, in g, filters, groups, activation);

    /// <summary>
    /// The input gradient of <see cref="ConvolutionKernel"/> (without its activation): dx += the transposed convolution of
    /// dy [N, filters, OH, OW] with weight, dx[n, q·Cg + c, ih, iw] += Σ weight[f, (c·KH + kh)·KW + kw] · dy[n, f, oh, ow] over
    /// the filters f of group q and the window positions with ih = oh·SH - PH + kh·DH and iw = ow·SW - PW + kw·DW.
    /// </summary>
    public virtual void ConvolutionBackwardInputKernel(Storage dy, Storage weight, Storage dx, in ConvGeometry g, int filters, int groups) =>
        ComposedConvolutionBackwardInput(dy, weight, dx, in g, filters, groups);

    /// <summary>
    /// The weight gradient of <see cref="ConvolutionKernel"/>: dweight[f, (c·KH + kh)·KW + kw] += Σ_{n, oh, ow} dy[n, f, oh, ow] ·
    /// x[n, q·Cg + c, oh·SH - PH + kh·DH, ow·SW - PW + kw·DW] (0 outside the image) for each filter f of group q.
    /// </summary>
    public virtual void ConvolutionBackwardWeightKernel(Storage x, Storage dy, Storage dweight, in ConvGeometry g, int filters, int groups) =>
        ComposedConvolutionBackwardWeight(x, dy, dweight, in g, filters, groups);

    /// <summary>
    /// Average pooling of NCHW images, as PyTorch's <c>AvgPool2d</c>: y[n, c, oh, ow] = the sum of x over the window
    /// (padded positions add 0, in row order) divided, when <paramref name="countIncludePad"/>, by the window's rows and
    /// columns up to the padded end (H + <paramref name="padBottom"/>, W + <paramref name="padRight"/>: the padding a
    /// divisor counts, which a ceil-mode geometry's PadBottom and PadRight may pass), else by the input positions the
    /// window covers (at least 1).
    /// </summary>
    public virtual void AvgPoolKernel(Storage x, Storage y, in ConvGeometry g, bool countIncludePad, int padBottom, int padRight)
    {
        using var h = new HostCall(this, nameof(AvgPool));
        Cpu.CpuBackend.Instance.AvgPool(h[x], h[y], g, countIncludePad, padBottom, padRight);
    }

    /// <summary>
    /// The gradient of <see cref="AvgPoolKernel"/>: dx[n, c, ih, iw] += Σ dy[n, c, oh, ow] / divisor over the windows that
    /// cover the element, in window order (the divisor as the forward pass takes it).
    /// </summary>
    public virtual void AvgPoolBackwardKernel(Storage dy, Storage dx, in ConvGeometry g, bool countIncludePad, int padBottom, int padRight)
    {
        using var h = new HostCall(this, nameof(AvgPoolBackward));
        Cpu.CpuBackend.Instance.AvgPoolBackward(h[dy], h[dx], g, countIncludePad, padBottom, padRight);
    }

    /// <summary>
    /// Image resampling and per-channel normalization in one pass. Each of <paramref name="planes"/> planes of x ([planes,
    /// height, width]; plane p is channel p % channels) is resampled to [outHeight, outWidth] by separable filters, across then
    /// down, and mapped per channel into y [planes, outHeight, outWidth]. <paramref name="coefficients"/> holds, for each
    /// output column, its first input column and its tap count (ints as float bits) then <paramref name="xTaps"/> weights,
    /// and after them the same for each output row with <paramref name="yTaps"/> weights; a pass with 0 taps is skipped
    /// (that size kept, no coefficients). Floats (<paramref name="bytes"/> false): x and the weights are floats, each
    /// output's taps are summed in order, and y = value · values[2c] + values[2c + 1]. Bytes: x holds 8-bit values four to
    /// a float word (value i in bits 8·(i % 4) of word i / 4), the weights are integers with 22 fractional bits (int bits),
    /// each pass sums 2^21 and its products as integers, shifts right by 22 and clips to [0, 255] (Pillow's
    /// <c>ImagingResample</c>), and y = values[256·c + value].
    /// </summary>
    public virtual void ResizeNormalizeKernel(Storage x, Storage coefficients, Storage values, Storage y, int planes, int channels, int height, int width,
        int outHeight, int outWidth, int xTaps, int yTaps, bool bytes)
    {
        using var h = new HostCall(this, nameof(ResizeNormalize));
        Cpu.CpuBackend.Instance.ResizeNormalize(h[x], h[coefficients], h[values], h[y], planes, channels, height, width, outHeight, outWidth, xTaps, yTaps, bytes);
    }

    /// <summary>
    /// Resamples each of <paramref name="planes"/> [height, width] planes to [outHeight, outWidth] as PyTorch's
    /// <c>F.interpolate</c>: output row o reads input row floor(o · scaleHeight) (nearest; the last row at most) or, bilinear,
    /// the two rows around o · scaleHeight (alignCorners) or around max((o + 0.5) · scaleHeight - 0.5, 0), weighted by
    /// distance; columns likewise. The scales are input positions per output position (1 / the scale factor, or the input
    /// size over the output size; with alignCorners (input - 1) / (output - 1)).
    /// </summary>
    public virtual void Interpolate2dKernel(Storage x, Storage y, int planes, int height, int width, int outHeight, int outWidth, InterpolationMode mode,
        bool alignCorners, float scaleHeight, float scaleWidth)
    {
        using var h = new HostCall(this, nameof(Interpolate2d));
        Cpu.CpuBackend.Instance.Interpolate2d(h[x], h[y], planes, height, width, outHeight, outWidth, mode, alignCorners, scaleHeight, scaleWidth);
    }

    /// <summary>The gradient of <see cref="Interpolate2dKernel"/>: dx += each output position's dy, spread over the input positions it read with their weights.</summary>
    public virtual void Interpolate2dBackwardKernel(Storage dy, Storage dx, int planes, int height, int width, int outHeight, int outWidth, InterpolationMode mode,
        bool alignCorners, float scaleHeight, float scaleWidth)
    {
        using var h = new HostCall(this, nameof(Interpolate2dBackward));
        Cpu.CpuBackend.Instance.Interpolate2dBackward(h[dy], h[dx], planes, height, width, outHeight, outWidth, mode, alignCorners, scaleHeight, scaleWidth);
    }

    /// <summary>
    /// Adaptive average pooling of <paramref name="planes"/> [height, width] planes to [outHeight, outWidth] (PyTorch's
    /// <c>AdaptiveAvgPool2d</c>): output row o averages input rows floor(o · height / outHeight) up to, not including,
    /// ceil((o + 1) · height / outHeight); columns likewise.
    /// </summary>
    public virtual void AdaptiveAvgPoolKernel(Storage x, Storage y, int planes, int height, int width, int outHeight, int outWidth)
    {
        using var h = new HostCall(this, nameof(AdaptiveAvgPool));
        Cpu.CpuBackend.Instance.AdaptiveAvgPool(h[x], h[y], planes, height, width, outHeight, outWidth);
    }

    /// <summary>The gradient of <see cref="AdaptiveAvgPoolKernel"/>: dx += dy of each window that covers the position, divided by the window's size.</summary>
    public virtual void AdaptiveAvgPoolBackwardKernel(Storage dy, Storage dx, int planes, int height, int width, int outHeight, int outWidth)
    {
        using var h = new HostCall(this, nameof(AdaptiveAvgPoolBackward));
        Cpu.CpuBackend.Instance.AdaptiveAvgPoolBackward(h[dy], h[dx], planes, height, width, outHeight, outWidth);
    }

    /// <summary>
    /// Adaptive max pooling over the windows of <see cref="AdaptiveAvgPoolKernel"/>; argmax receives the flat input index of
    /// each maximum (as raw int bits, the first in row order), so <see cref="Backend.MaxPoolBackward(Storage, Storage, Storage, int)"/> is its gradient.
    /// </summary>
    public virtual void AdaptiveMaxPoolKernel(Storage x, Storage y, Storage argmax, int planes, int height, int width, int outHeight, int outWidth)
    {
        using var h = new HostCall(this, nameof(AdaptiveMaxPool));
        Cpu.CpuBackend.Instance.AdaptiveMaxPool(h[x], h[y], h[argmax], planes, height, width, outHeight, outWidth);
    }

    /// <summary>
    /// The gradient of <see cref="AdaptiveMaxPoolKernel"/>: dx[argmax[i]] += dy[i] for its planes · outHeight · outWidth outputs,
    /// the same as <see cref="Backend.MaxPoolBackward(Storage, Storage, Storage, int)"/>; a device that adds the gradients by
    /// gathering over the windows (no atomics, the same bits every run) needs the sizes.
    /// </summary>
    public virtual void AdaptiveMaxPoolBackwardKernel(Storage dy, Storage argmax, Storage dx, int planes, int height, int width, int outHeight, int outWidth) =>
        MaxPoolBackward(dy, argmax, dx, planes * outHeight * outWidth);

    /// <summary>
    /// Connectionist temporal classification (Graves et al. 2006): losses[n] = -log of the probability, summed over every
    /// alignment, that the first inputLengths[n] steps of sequence n read as its labels (with <paramref name="blank"/>
    /// between them and repeats collapsed). logProbs holds log-probabilities over <paramref name="classes"/> per step and
    /// sequence, [steps, batch, classes] or, with <paramref name="batchFirst"/>, [batch, steps, classes]; the labels of
    /// sequence n are targets[targetOffsets[n] ..] (targetLengths[n] ids as floats). A sequence no alignment fits has an
    /// infinite loss, or 0 with <paramref name="zeroInfinity"/>.
    /// </summary>
    public virtual void CtcLossKernel(Storage logProbs, Storage targets, Storage losses, ReadOnlySpan<int> inputLengths, ReadOnlySpan<int> targetLengths,
        ReadOnlySpan<int> targetOffsets, int steps, int batch, int classes, int blank, bool batchFirst, bool zeroInfinity)
    {
        using var h = new HostCall(this, nameof(CtcLoss));
        Cpu.CpuBackend.Instance.CtcLoss(h[logProbs], h[targets], h[losses], inputLengths, targetLengths, targetOffsets, steps, batch, classes, blank, batchFirst, zeroInfinity);
    }

    /// <summary>
    /// The gradient of <see cref="CtcLossKernel"/>: dLogProbs += lossGrads[n] · ∂losses[n] / ∂logProbs for every sequence n
    /// (steps past inputLengths[n] and classes outside the labels get nothing; with <paramref name="zeroInfinity"/> a
    /// sequence of infinite loss gets nothing either).
    /// </summary>
    public virtual void CtcLossBackwardKernel(Storage logProbs, Storage targets, Storage lossGrads, Storage dLogProbs, ReadOnlySpan<int> inputLengths,
        ReadOnlySpan<int> targetLengths, ReadOnlySpan<int> targetOffsets, int steps, int batch, int classes, int blank, bool batchFirst, bool zeroInfinity)
    {
        using var h = new HostCall(this, nameof(CtcLossBackward));
        Cpu.CpuBackend.Instance.CtcLossBackward(h[logProbs], h[targets], h[lossGrads], h[dLogProbs], inputLengths, targetLengths, targetOffsets, steps, batch, classes, blank, batchFirst, zeroInfinity);
    }

    /// <summary>
    /// One time step of a long short-term memory cell (PyTorch's gate order: input, forget, cell, output), for every row n
    /// &lt; <paramref name="batch"/> and unit j &lt; H = <paramref name="hiddenSize"/>. The sequences are [batch, steps,
    /// width] row-major and the step reads and writes their row (n, <paramref name="step"/>); the states are [batch, width].
    /// With z_k = projected[n, step, k·H + j] + recurrent[n, k·H + j] (the step's input through the input weights and bias,
    /// and the previous hidden state through the recurrent weights): i = σ(z_0), f = σ(z_1), g = tanh(z_2), o = σ(z_3);
    /// c = f · cell[n, j] + i · g and h = o · tanh(c). Writes c into cell[n, j] (in place) and, when given, cells[n, step,
    /// j]; h into hidden[n, j] and output[n, step, j]; and, when given, i, f, g and o into gates[n, step, k·H + j] (what
    /// <see cref="LstmCellBackwardKernel"/> reads, with the cells). With <paramref name="batch"/> 0 nothing is read or
    /// written: callers ask that way whether the device has the kernel. Returns false when the device has none (callers
    /// then compose the step from products and element-wise operations).
    /// </summary>
    public bool LstmCellKernel(Storage projected, Storage recurrent, Storage cell, Storage hidden, Storage output, Storage? gates, Storage? cells,
        int step, int steps, int batch, int hiddenSize) =>
        _recurrent is { } own && own.LstmCell(projected, recurrent, cell, hidden, output, gates, cells, step, steps, batch, hiddenSize);

    /// <summary>
    /// The gradient of one step of <see cref="LstmCellKernel"/>, the steps taken from the last to the first. From the saved
    /// gates (i, f, g, o) and cells of the step, the previous step's cell c_prev = cells[n, previous, j] (0 when
    /// <paramref name="previous"/> is negative: the zero initial state), and the incoming dh = dHidden[n, j] + dOutput[n,
    /// step, j] (no dOutput: 0) and dc = dCell[n, j]: with t = tanh(c), dc' = dc + dh · o · (1 - t²), the pre-activation
    /// gradients dz = (dc'·g · i(1 - i), dc'·c_prev · f(1 - f), dc'·i · (1 - g²), dh·t · o(1 - o)) are written (not added)
    /// into dGates[n, step, k·H + j] and dStep[n, k·H + j], and dc'·f into dCell[n, j] (the cell gradient the previous step
    /// takes). dHidden is only read: the caller sets it to dStep · Uᵀ, the recurrent part of the previous step's dh. With
    /// <paramref name="batch"/> 0 nothing is read or written. Returns false when the device has no such kernel.
    /// </summary>
    public bool LstmCellBackwardKernel(Storage gates, Storage cells, Storage? dOutput, Storage dHidden, Storage dCell, Storage dGates, Storage dStep,
        int step, int previous, int steps, int batch, int hiddenSize) =>
        _recurrent is { } own && own.LstmCellBackward(gates, cells, dOutput, dHidden, dCell, dGates, dStep, step, previous, steps, batch, hiddenSize);

    /// <summary>
    /// One time step of a gated recurrent unit (PyTorch's gate order: reset, update, candidate; the reset gate scales the
    /// candidate's recurrent term after its product, PyTorch's GRU and ONNX's linear_before_reset), for every row n &lt;
    /// <paramref name="batch"/> and unit j &lt; H = <paramref name="hiddenSize"/>, with the layouts of
    /// <see cref="LstmCellKernel"/>. With p_k = projected[n, step, k·H + j] and q_k = recurrent[n, k·H + j]: r = σ(p_0 +
    /// q_0), u = σ(p_1 + q_1), a = q_2 + hiddenBias[j] (no hiddenBias: q_2), c = tanh(p_2 + r · a) and h = (1 - u) · c + u
    /// · hidden[n, j]. Writes h into hidden[n, j] (in place) and output[n, step, j] and, when given, r, u, c and a into
    /// gates[n, step, k·H + j] (a [batch, steps, 4H] sequence, what <see cref="GruCellBackwardKernel"/> reads). With
    /// <paramref name="batch"/> 0 nothing is read or written. Returns false when the device has no such kernel.
    /// </summary>
    public bool GruCellKernel(Storage projected, Storage recurrent, Storage? hiddenBias, Storage hidden, Storage output, Storage? gates,
        int step, int steps, int batch, int hiddenSize) =>
        _recurrent is { } own && own.GruCell(projected, recurrent, hiddenBias, hidden, output, gates, step, steps, batch, hiddenSize);

    /// <summary>
    /// The gradient of one step of <see cref="GruCellKernel"/>, the steps taken from the last to the first. From the saved
    /// r, u, c and a of the step, the previous hidden state h_prev = output[n, previous, j] (0 when
    /// <paramref name="previous"/> is negative) and the incoming dh = dHidden[n, j] + dOutput[n, step, j] (no dOutput: 0):
    /// with dc = dh · (1 - u) · (1 - c²), the input part dp = (dc·a · r(1 - r), dh·(h_prev - c) · u(1 - u), dc) is written
    /// (not added) into dGates[n, step, k·H + j], the recurrent part dq = (dp_0, dp_1, dc · r) (the gradient of the hidden
    /// product, and of the candidate bias in its last third) into dRecurrent[n, step, k·H + j] and dStep[n, k·H + j], and
    /// dh · u into dHidden[n, j] (the direct part of the previous step's dh; the caller adds dStep · Uᵀ to it). With
    /// <paramref name="batch"/> 0 nothing is read or written. Returns false when the device has no such kernel.
    /// </summary>
    public bool GruCellBackwardKernel(Storage gates, Storage output, Storage? dOutput, Storage dHidden, Storage dGates, Storage dRecurrent, Storage dStep,
        int step, int previous, int steps, int batch, int hiddenSize) =>
        _recurrent is { } own && own.GruCellBackward(gates, output, dOutput, dHidden, dGates, dRecurrent, dStep, step, previous, steps, batch, hiddenSize);

    /// <summary>
    /// One time step of <see cref="LstmCellKernel"/> with the recurrent product inside: recurrent[n, k·H + j] = Σ_m h[m] ·
    /// weights[m, k·H + j] (weights U [H, 4H] row-major; the device fixes the order of the sum, the same on every call) with
    /// h = output[n, <paramref name="previous"/>, ·], the
    /// previous step's hidden state (no product when <paramref name="previous"/> is negative: the zero initial state). The
    /// cell state is read from and written to cell[n, j] in place; h goes to output[n, step, j]; gates and cells as in
    /// <see cref="LstmCellKernel"/>. One launch a step, where the cell kernel needs a product before it. With
    /// <paramref name="batch"/> 0 nothing is read or written. Returns false when the device has no such kernel (callers
    /// then use <see cref="LstmCellKernel"/> after a product).
    /// </summary>
    public bool LstmStepKernel(Storage projected, Storage weights, Storage cell, Storage output, Storage? gates, Storage? cells, int step, int previous,
        int steps, int batch, int hiddenSize) =>
        _recurrent is { } own && own.LstmStep(projected, weights, cell, output, gates, cells, step, previous, steps, batch, hiddenSize);

    /// <summary>
    /// The gradient of one step of <see cref="LstmStepKernel"/>, the steps taken from the last to the first: as
    /// <see cref="LstmCellBackwardKernel"/> with dHidden[n, j] = Σ_g dGates[n, <paramref name="next"/>, g] ·
    /// weightsT[g, j] (weightsT = Uᵀ [4H, H] row-major: the recurrent part of this step's dh, from the gradient the later
    /// step wrote; 0 when <paramref name="next"/> is negative, the last step taken), summed as <see cref="LstmStepKernel"/>
    /// sums its product. Writes dGates[n, step, ·] and dCell[n, j] (in place) only. With <paramref name="batch"/> 0 nothing
    /// is read or written. Returns false when the device has no such kernel.
    /// </summary>
    public bool LstmStepBackwardKernel(Storage gates, Storage cells, Storage? dOutput, Storage weightsT, Storage dCell, Storage dGates, int step, int next,
        int previous, int steps, int batch, int hiddenSize) =>
        _recurrent is { } own && own.LstmStepBackward(gates, cells, dOutput, weightsT, dCell, dGates, step, next, previous, steps, batch, hiddenSize);

    /// <summary>
    /// One time step of <see cref="GruCellKernel"/> with the recurrent product inside, summed as
    /// <see cref="LstmStepKernel"/> sums it (weights U [H, 3H]); the previous hidden state is output[n,
    /// <paramref name="previous"/>, ·] (0 when <paramref name="previous"/> is negative), and h goes to output[n, step, j]
    /// only. With <paramref name="batch"/> 0 nothing is read or written. Returns false when the device has no such kernel.
    /// </summary>
    public bool GruStepKernel(Storage projected, Storage weights, Storage? hiddenBias, Storage output, Storage? gates, int step, int previous, int steps,
        int batch, int hiddenSize) =>
        _recurrent is { } own && own.GruStep(projected, weights, hiddenBias, output, gates, step, previous, steps, batch, hiddenSize);

    /// <summary>
    /// The gradient of one step of <see cref="GruStepKernel"/>, the steps taken from the last to the first: as
    /// <see cref="GruCellBackwardKernel"/> with the incoming dh = (Σ_g dRecurrent[n, <paramref name="next"/>, g] ·
    /// weightsT[g, j] + dHidden[n, j]) + dOutput[n, step, j] (weightsT = Uᵀ [3H, H]; no sum when <paramref name="next"/> is
    /// negative) and dHidden[n, j] overwritten with this step's direct part dh · u. Writes dGates and dRecurrent at the
    /// step's row and dHidden only. With <paramref name="batch"/> 0 nothing is read or written. Returns false when the
    /// device has no such kernel.
    /// </summary>
    public bool GruStepBackwardKernel(Storage gates, Storage output, Storage? dOutput, Storage weightsT, Storage dHidden, Storage dGates, Storage dRecurrent,
        int step, int next, int previous, int steps, int batch, int hiddenSize) =>
        _recurrent is { } own && own.GruStepBackward(gates, output, dOutput, weightsT, dHidden, dGates, dRecurrent, step, next, previous, steps, batch,
            hiddenSize);

    /// <summary>
    /// Every step of <see cref="LstmStepKernel"/> in one call: for i = 0, 1, …, <paramref name="steps"/> - 1 the step t = i
    /// (<paramref name="reverse"/>: t = steps - 1 - i) after the step taken before it (none for the first), the same values
    /// as those calls in that order, the cell state carried in cell[n, j] (the zero initial state when the caller zeroes
    /// it). One launch a sequence, where the step kernel takes one a step. With <paramref name="batch"/> 0 nothing is read or
    /// written. Returns false when the device has no such kernel for these sizes (callers then take the step kernels).
    /// </summary>
    public bool LstmSequenceKernel(Storage projected, Storage weights, Storage cell, Storage output, Storage? gates, Storage? cells, int steps, int batch,
        int hiddenSize, bool reverse) =>
        _recurrent is { } own && own.LstmSequence(projected, weights, cell, output, gates, cells, steps, batch, hiddenSize, reverse);

    /// <summary>
    /// Every step of <see cref="LstmStepBackwardKernel"/> in one call, the steps from the last taken by
    /// <see cref="LstmSequenceKernel"/> to the first, each with the step taken before it and after it (none at the ends):
    /// the same values as those calls in that order. dCell starts as the caller gives it (zero for the loss's end). With
    /// <paramref name="batch"/> 0 nothing is read or written. Returns false when the device has no such kernel for these
    /// sizes.
    /// </summary>
    public bool LstmSequenceBackwardKernel(Storage gates, Storage cells, Storage? dOutput, Storage weightsT, Storage dCell, Storage dGates, int steps,
        int batch, int hiddenSize, bool reverse) =>
        _recurrent is { } own && own.LstmSequenceBackward(gates, cells, dOutput, weightsT, dCell, dGates, steps, batch, hiddenSize, reverse);

    /// <summary>
    /// Every step of <see cref="GruStepKernel"/> in one call, the steps taken as <see cref="LstmSequenceKernel"/> takes
    /// them: the same values as those calls in that order. With <paramref name="batch"/> 0 nothing is read or written.
    /// Returns false when the device has no such kernel for these sizes.
    /// </summary>
    public bool GruSequenceKernel(Storage projected, Storage weights, Storage? hiddenBias, Storage output, Storage? gates, int steps, int batch,
        int hiddenSize, bool reverse) =>
        _recurrent is { } own && own.GruSequence(projected, weights, hiddenBias, output, gates, steps, batch, hiddenSize, reverse);

    /// <summary>
    /// Every step of <see cref="GruStepBackwardKernel"/> in one call, the steps taken as
    /// <see cref="LstmSequenceBackwardKernel"/> takes them: the same values as those calls in that order. dHidden starts as
    /// the caller gives it (zero for the loss's end). With <paramref name="batch"/> 0 nothing is read or written. Returns
    /// false when the device has no such kernel for these sizes.
    /// </summary>
    public bool GruSequenceBackwardKernel(Storage gates, Storage output, Storage? dOutput, Storage weightsT, Storage dHidden, Storage dGates,
        Storage dRecurrent, int steps, int batch, int hiddenSize, bool reverse) =>
        _recurrent is { } own && own.GruSequenceBackward(gates, output, dOutput, weightsT, dHidden, dGates, dRecurrent, steps, batch, hiddenSize, reverse);

    /// <summary>
    /// Box overlap losses, torchvision's formulas: for each of <paramref name="count"/> pairs of boxes given by their corners
    /// (x1, y1, x2, y2; predicted and target are [count, 4]), losses[i] = 1 - IoU (<see cref="BoxOverlap.IoU"/>), plus the
    /// enclosing box's empty share (GIoU), plus the centres' squared distance over the enclosing box's squared diagonal
    /// (DIoU), plus α·v with v the aspect-ratio term and α = v / (1 - IoU + v + eps) taken as a constant (CIoU).
    /// <paramref name="eps"/> keeps the divisions finite (torchvision's 1e-7).
    /// </summary>
    public virtual void BoxIouLossKernel(Storage predicted, Storage target, Storage losses, int count, BoxOverlap overlap, float eps)
    {
        using var h = new HostCall(this, nameof(BoxIouLoss));
        Cpu.CpuBackend.Instance.BoxIouLoss(h[predicted], h[target], h[losses], count, overlap, eps);
    }

    /// <summary>
    /// The gradient of <see cref="BoxIouLossKernel"/> with respect to the predicted corners: dPredicted[i] += lossGrads[i] ·
    /// ∂losses[i] / ∂predicted[i] (the targets are constants; a tie of a minimum or maximum shares the gradient in halves,
    /// as PyTorch's).
    /// </summary>
    public virtual void BoxIouLossBackwardKernel(Storage predicted, Storage target, Storage lossGrads, Storage dPredicted, int count, BoxOverlap overlap, float eps)
    {
        using var h = new HostCall(this, nameof(BoxIouLossBackward));
        Cpu.CpuBackend.Instance.BoxIouLossBackward(h[predicted], h[target], h[lossGrads], h[dPredicted], count, overlap, eps);
    }

    /// <summary>
    /// Sigmoid focal loss (Lin et al. 2017), torchvision's formula, element by element over <paramref name="count"/> logits:
    /// losses = α_t · (1 - p_t)^γ · BCE(logits, targets), p = σ(logits), p_t = p·t + (1 - p)(1 - t),
    /// α_t = α·t + (1 - α)(1 - t), with no α weighting when <paramref name="alpha"/> is negative. Targets in [0, 1].
    /// </summary>
    public virtual void SigmoidFocalLossKernel(Storage logits, Storage targets, Storage losses, int count, float alpha, float gamma)
    {
        using var h = new HostCall(this, nameof(SigmoidFocalLoss));
        Cpu.CpuBackend.Instance.SigmoidFocalLoss(h[logits], h[targets], h[losses], count, alpha, gamma);
    }

    /// <summary>The gradient of <see cref="SigmoidFocalLossKernel"/>: dLogits[i] += lossGrads[i] · ∂losses[i] / ∂logits[i] (the targets are constants).</summary>
    public virtual void SigmoidFocalLossBackwardKernel(Storage logits, Storage targets, Storage lossGrads, Storage dLogits, int count, float alpha, float gamma)
    {
        using var h = new HostCall(this, nameof(SigmoidFocalLossBackward));
        Cpu.CpuBackend.Instance.SigmoidFocalLossBackward(h[logits], h[targets], h[lossGrads], h[dLogits], count, alpha, gamma);
    }

    /// <summary>
    /// y (+)= x permuted: output element at coordinates (c0..c[r-1]) of <paramref name="outShape"/> comes from
    /// input offset Σ c_k * inStrides[k] (the input strides already reordered by the permutation). Rank ≤ 6.
    /// </summary>
    public virtual void PermuteKernel(Storage x, Storage y, ReadOnlySpan<int> outShape, ReadOnlySpan<int> inStrides, bool accumulate)
    {
        using var h = new HostCall(this, nameof(Permute));
        Cpu.CpuBackend.Instance.Permute(h[x], h[y], outShape, inStrides, accumulate);
    }

    /// <summary>
    /// Strided block copy: dst[dstOffset + r*dstStride + c] (+)= src[srcOffset + r*srcStride + c] for r &lt; rows, c &lt; cols.
    /// Implements narrow, concatenation and their gradients.
    /// </summary>
    public virtual void Copy2D(Storage src, int srcOffset, int srcStride, Storage dst, int dstOffset, int dstStride, int rows, int cols, bool accumulate)
    {
        using var h = new HostCall(this, nameof(Copy2D));
        Cpu.CpuBackend.Instance.Copy2D(h[src], srcOffset, srcStride, h[dst], dstOffset, dstStride, rows, cols, accumulate);
    }

    /// <summary>y[o, i] (+)= scale * Σ_d x[o, d, i] for a [outer, dim, inner] view.</summary>
    public virtual void SumAxisKernel(Storage x, Storage y, int outer, int dim, int inner, float scale, bool accumulate)
    {
        using var h = new HostCall(this, nameof(SumAxis));
        Cpu.CpuBackend.Instance.SumAxis(h[x], h[y], outer, dim, inner, scale, accumulate);
    }

    /// <summary>dx[o, d, i] += scale * dy[o, i] (the gradient of <see cref="SumAxis"/>).</summary>
    public virtual void BroadcastAxisKernel(Storage dy, Storage dx, int outer, int dim, int inner, float scale)
    {
        using var h = new HostCall(this, nameof(BroadcastAxis));
        Cpu.CpuBackend.Instance.BroadcastAxis(h[dy], h[dx], outer, dim, inner, scale);
    }

    /// <summary>SGD with optional momentum: v = momentum * v + g; p -= lr * v (v is null when momentum is 0).</summary>
    public virtual void SgdStepKernel(Storage p, Storage g, Storage? v, int n, float lr, float momentum)
    {
        using var h = new HostCall(this, nameof(SgdStep));
        Cpu.CpuBackend.Instance.SgdStep(h[p], h[g], h.Maybe(v), n, lr, momentum);
    }

    /// <summary>Adam: m, v moments updated in place; p -= lr * m / (sqrt(v) + eps). lr is already bias-corrected.</summary>
    public virtual void AdamStepKernel(Storage p, Storage g, Storage m, Storage v, int n, float lr, float beta1, float beta2, float eps)
    {
        using var h = new HostCall(this, nameof(AdamStep));
        Cpu.CpuBackend.Instance.AdamStep(h[p], h[g], h[m], h[v], n, lr, beta1, beta2, eps);
    }

    /// <summary>total[0] += Σ x² over <paramref name="n"/> elements (a gradient norm without temporary tensors).</summary>
    public virtual void SumSquaresKernel(Storage x, Storage total, int n)
    {
        using var h = new HostCall(this, nameof(SumSquares));
        Cpu.CpuBackend.Instance.SumSquares(h[x], h[total], n);
    }

    /// <summary>
    /// AdamW over many tensors in a few passes: the global gradient norm clipped to <paramref name="maxNorm"/> (0: no
    /// clipping), p ← p · <paramref name="decay"/>, then Adam with the bias-corrected <paramref name="lr"/>, as
    /// <see cref="AdamStep"/> computes it; with <paramref name="zeroGradients"/> the gradients are zeroed for the next step.
    /// <paramref name="cache"/> keeps the device tables between calls (dispose it when done). False when the device has no
    /// such pass.
    /// </summary>
    public virtual bool FusedAdamWKernel(ReadOnlySpan<(Storage P, Storage G, Storage M, Storage V, int N)> tensors, ref IDisposable? cache, float maxNorm,
        float lr, float decay, float beta1, float beta2, float eps, bool zeroGradients) => false;

    /// <summary>
    /// factor[0] = min(1, maxNorm / √sumSquares[0]) (1 when the sum is 0): the gradient-clipping factor computed where the
    /// gradients are, so clipping needs no host read of the norm.
    /// </summary>
    public virtual void ClipFactorKernel(Storage sumSquares, Storage factor, float maxNorm)
    {
        using var h = new HostCall(this, nameof(ClipFactor));
        Cpu.CpuBackend.Instance.ClipFactor(h[sumSquares], h[factor], maxNorm);
    }

    /// <summary>Inverted dropout: y = keep(i) ? x / (1 - p) : 0, where keep(i) comes from <see cref="DropoutMask"/>.</summary>
    public virtual void DropoutKernel(Storage x, Storage y, int n, float p, uint seed)
    {
        using var h = new HostCall(this, nameof(Dropout));
        Cpu.CpuBackend.Instance.Dropout(h[x], h[y], n, p, seed);
    }

    /// <summary>dx += keep(i) ? dy / (1 - p) : 0, regenerating the same mask from the seed.</summary>
    public virtual void DropoutBackwardKernel(Storage dy, Storage dx, int n, float p, uint seed)
    {
        using var h = new HostCall(this, nameof(DropoutBackward));
        Cpu.CpuBackend.Instance.DropoutBackward(h[dy], h[dx], n, p, seed);
    }

    /// <summary>Waits until every kernel and copy issued so far has finished.</summary>
    public abstract void Synchronize();

    // ---------------------------------------------------------------- fused inference kernels

    /// <summary>y[r, :] = softmax(scale * x[r, :] + mask[r % maskRows, :]) (mask optional).</summary>
    public virtual void ScaleMaskSoftmaxKernel(Storage x, Storage? mask, Storage y, int rows, int cols, int maskRows, float scale)
    {
        using var h = new HostCall(this, nameof(ScaleMaskSoftmax));
        Cpu.CpuBackend.Instance.ScaleMaskSoftmax(h[x], h.Maybe(mask), h[y], rows, cols, maskRows, scale);
    }

    /// <summary>y[r, :] = (x[r, :] - mean) / sqrt(var + eps) * gamma + beta over the last dimension.</summary>
    public virtual void LayerNormFusedKernel(Storage x, Storage gamma, Storage beta, Storage y, int rows, int cols, float eps)
    {
        using var h = new HostCall(this, nameof(LayerNormFused));
        Cpu.CpuBackend.Instance.LayerNormFused(h[x], h[gamma], h[beta], h[y], rows, cols, eps);
    }

    /// <summary><see cref="LayerNormFused"/> that also stores each row's mean (stats[r]) and 1 / sqrt(var + eps) (stats[rows + r]).</summary>
    public virtual void LayerNormTrainKernel(Storage x, Storage gamma, Storage beta, Storage y, Storage stats, int rows, int cols, float eps)
    {
        using var h = new HostCall(this, nameof(LayerNormTrain));
        Cpu.CpuBackend.Instance.LayerNormTrain(h[x], h[gamma], h[beta], h[y], h[stats], rows, cols, eps);
    }

    /// <summary>
    /// Gradients of <see cref="LayerNormTrain"/> given dy: adds to dx (when given), dgamma += Σ_r dy ∘ x̂ and dbeta += Σ_r dy.
    /// </summary>
    public virtual void LayerNormBackwardKernel(Storage x, Storage gamma, Storage dy, Storage stats, Storage? dx, Storage? dgamma, Storage? dbeta, int rows, int cols)
    {
        using var h = new HostCall(this, nameof(LayerNormBackward));
        Cpu.CpuBackend.Instance.LayerNormBackward(h[x], h[gamma], h[dy], h[stats], h.Maybe(dx), h.Maybe(dgamma), h.Maybe(dbeta), rows, cols);
    }

    /// <summary>y[i] = gelu(x[i] + bias[i % cols]).</summary>
    public virtual void BiasGeluKernel(Storage x, Storage bias, Storage y, int n, int cols)
    {
        using var h = new HostCall(this, nameof(BiasGelu));
        Cpu.CpuBackend.Instance.BiasGelu(h[x], h[bias], h[y], n, cols);
    }

    /// <summary>y = x · inv per row, inv[r] = 1 / sqrt(mean(x[r]²) + eps) (stored for the backward pass).</summary>
    public virtual void RmsNormKernel(Storage x, Storage y, Storage inv, int rows, int cols, float eps)
    {
        using var h = new HostCall(this, nameof(RmsNorm));
        Cpu.CpuBackend.Instance.RmsNorm(h[x], h[y], h[inv], rows, cols, eps);
    }

    /// <summary>dx += inv[r] · (dy - y · mean(dy · y)) per row, where y is the normalized forward output.</summary>
    public virtual void RmsNormBackwardKernel(Storage dy, Storage y, Storage inv, Storage dx, int rows, int cols)
    {
        using var h = new HostCall(this, nameof(RmsNormBackward));
        Cpu.CpuBackend.Instance.RmsNormBackward(h[dy], h[y], h[inv], h[dx], rows, cols);
    }

    /// <summary>
    /// Rotary position embedding of x [rows = batch·steps·heads, dim] into y (which must already hold x): pair p of a row at
    /// step t rotates by the angle whose cos/sin are cos/sin[positions[t], p]. Pairs are (2p, 2p+1) when interleaved,
    /// else (p, p + half). sign = -1 rotates backwards (the gradient).
    /// </summary>
    public virtual void RopeKernel(Storage x, Storage y, Storage cos, Storage sin, Storage positions, int rows, int heads, int steps, int dim, int half, bool interleaved, float sign)
    {
        using var h = new HostCall(this, nameof(Rope));
        Cpu.CpuBackend.Instance.Rope(h[x], h[y], h[cos], h[sin], h[positions], rows, heads, steps, dim, half, interleaved, sign);
    }

    /// <summary>y = x · inv · (gain[c] + offset) per row with inv = 1 / sqrt(mean(x²) + eps): normalization and gain in one pass (inference).</summary>
    public virtual void RmsNormAffineKernel(Storage x, Storage gain, Storage y, int rows, int cols, float eps, float offset)
    {
        using var h = new HostCall(this, nameof(RmsNormAffine));
        Cpu.CpuBackend.Instance.RmsNormAffine(h[x], h[gain], h[y], rows, cols, eps, offset);
    }

    /// <summary>sum = a + b and y = RMS-normalized sum · (gain[c] + offset) per row, in one pass (inference).</summary>
    public virtual void AddRmsNormAffineKernel(Storage a, Storage b, Storage sum, Storage gain, Storage y, int rows, int cols, float eps, float offset)
    {
        using var h = new HostCall(this, nameof(AddRmsNormAffine));
        Cpu.CpuBackend.Instance.AddRmsNormAffine(h[a], h[b], h[sum], h[gain], h[y], rows, cols, eps, offset);
    }

    /// <summary>
    /// RMS normalization with gain of each row (a head's vector), then the rotary embedding as <see cref="Rope"/> with
    /// sign 1 (rows are batch·steps·heads; dimensions beyond 2·half are only normalized). Inference.
    /// </summary>
    public virtual void RmsNormRopeKernel(Storage x, Storage gain, Storage cos, Storage sin, Storage positions, Storage y, int rows, int cols,
        float eps, float offset, int heads, int steps, int half, bool interleaved)
    {
        using var h = new HostCall(this, nameof(RmsNormRope));
        Cpu.CpuBackend.Instance.RmsNormRope(h[x], h[gain], h[cos], h[sin], h[positions], h[y], rows, cols, eps, offset, heads, steps, half, interleaved);
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
    /// at row position[0] + s (a cache; row s when <paramref name="position"/> is null).
    /// Returns false when the device has no such kernel (callers then run the steps one by one).
    /// </summary>
    public virtual bool NormRopeHeadsKernel(Storage q, Storage k, Storage v, int batch, int steps, int heads, int kvHeads, int cols,
        Storage? gainQ, float epsQ, float offsetQ, Storage? gainK, float epsK, float offsetK, Storage? cos, Storage? sin, Storage? positions,
        int half, bool interleaved, Storage yq, Storage yk, Storage yv, Storage? position, int capacity, int stride) => false;

    /// <summary>
    /// Token cross-entropy for language-model training, per row r of logits [rows, vocabulary] with target class t_r and
    /// weight w_r (0 masks the row): losses[r] = w_r · (logsumexp(x_r) - x_r[t_r]); the logits are overwritten with their
    /// gradient scale · w_r · (softmax(x_r) - onehot(t_r)). Targets and weights are float arrays of rows.
    /// </summary>
    public virtual void SoftmaxCrossEntropyRowsKernel(Storage logits, Storage targets, Storage weights, Storage losses, int rows, int vocabulary, float scale)
    {
        using var h = new HostCall(this, nameof(SoftmaxCrossEntropyRows));
        Cpu.CpuBackend.Instance.SoftmaxCrossEntropyRows(h[logits], h[targets], h[weights], h[losses], rows, vocabulary, scale);
    }

    /// <summary>y = act(gate) · up element-wise; kind 0 = SiLU, 1 = GELU (tanh approximation), 2 = ReLU, 3 = tanh(gate) · sigmoid(up) (the up side through its own function).</summary>
    public virtual void GatedActivationKernel(Storage gate, Storage up, Storage y, int n, int kind)
    {
        using var h = new HostCall(this, nameof(GatedActivation));
        Cpu.CpuBackend.Instance.GatedActivation(h[gate], h[up], h[y], n, kind);
    }

    /// <summary>dgate += dy · up · act'(gate) when flags has bit 0, dup += dy · act(gate) when it has bit 1 (kind 3: up read as sigmoid(up), and dup += dy · tanh(gate) · sigmoid'(up)); bits 2 and 3 write dgate and dup (= instead of +=).</summary>
    public virtual void GatedActivationBackwardKernel(Storage gate, Storage up, Storage dy, Storage dgate, Storage dup, int n, int kind, int flags)
    {
        using var h = new HostCall(this, nameof(GatedActivationBackward));
        Cpu.CpuBackend.Instance.GatedActivationBackward(h[gate], h[up], h[dy], h[dgate], h[dup], n, kind, flags);
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
        using var h = new HostCall(this, nameof(AttentionDecode));
        Cpu.CpuBackend.Instance.AttentionDecode(h[q], h[keys], h[values], h[position], h[y], heads, rowsPerHead, steps, capacity, dim, scale, variant);
    }

    /// <summary>
    /// Gradient of <see cref="AttentionTiled"/> with causal offset 0 (training): given the output, each row's log-sum-exp
    /// and dOutput, adds to dq [heads, rowsPerHead, dim] and dkeys, dvalues [heads, capacity, dim]. The attention weights
    /// are recomputed, never stored.
    /// </summary>
    public virtual void AttentionTiledBackwardKernel(Storage q, Storage keys, Storage values, Storage output, Storage logSumExp, Storage dOutput,
        Storage dq, Storage dkeys, Storage dvalues, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        using var h = new HostCall(this, nameof(AttentionTiledBackward));
        Cpu.CpuBackend.Instance.AttentionTiledBackward(h[q], h[keys], h[values], h[output], h[logSumExp], h[dOutput], h[dq], h[dkeys], h[dvalues], heads, rowsPerHead, steps, capacity, dim, scale, variant);
    }

    /// <summary>
    /// Causal attention over packed sequences (training): as <see cref="AttentionTiled"/> with offset 0 and keys and values
    /// [heads, steps, dim], except that several sequences share each row of <paramref name="steps"/> positions, so row i of
    /// head h sees positions c with starts[b·steps + t] ≤ c ≤ t, where t = i % steps and b = h / <paramref name="headsPerRow"/>
    /// is the packed row; starts and ends hold, per position of each packed row, where its sequence begins and stops
    /// (exclusive), as floats. Writes the log-sum-exp when given. Returns false when the device has no such pass.
    /// </summary>
    public bool AttentionSegmentedKernel(Storage q, Storage keys, Storage values, Storage y, Storage? logSumExp, Storage starts, Storage ends,
        int heads, int headsPerRow, int rowsPerHead, int steps, int dim, float scale, AttentionVariant variant = default) =>
        _segmented is { } own && own.AttentionSegmented(q, keys, values, y, logSumExp, starts, ends, heads, headsPerRow, rowsPerHead, steps, dim, scale, variant);

    /// <summary>
    /// <see cref="AttentionTiled"/> (no log-sum-exp) for rows of different lengths decoded together: row i of head h
    /// sees cached positions c with starts[(h / headsPerRow)·steps + i % steps] ≤ c ≤ position[0] + i % steps (a row with
    /// none, padding, gets zeros). Returns false when the device has no such pass.
    /// </summary>
    public bool AttentionRowsKernel(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage starts, int heads, int headsPerRow,
        int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant = default) =>
        _rows is { } own && own.AttentionRows(q, keys, values, position, y, starts, heads, headsPerRow, rowsPerHead, steps, capacity, dim, scale, variant);

    /// <summary>
    /// Whether <see cref="AttentionSegmented"/> and its gradient run for this head size (with the current
    /// variant): the device's own answer (<see cref="ISegmentedAttentionBackend"/>), else false.
    /// </summary>
    public bool SupportsSegmentedAttention(int dim, AttentionVariant variant = default) =>
        _segmented is { } own && own.SupportsSegmented(dim, variant);

    /// <summary>
    /// Whether <see cref="AttentionRows"/> runs on this device for head size <paramref name="dim"/> and
    /// <paramref name="variant"/> (rows of different lengths decoded together), so callers batch such rows instead of
    /// running them apart: the device's own answer (<see cref="IRowAttentionBackend"/>), else as
    /// <see cref="SupportsSegmentedAttention"/> (one pass serves both where a device has one).
    /// </summary>
    public bool SupportsAttentionRows(int dim, AttentionVariant variant = default) =>
        _rows is { } own ? own.SupportsRows(dim, variant) : SupportsSegmentedAttention(dim, variant);

    /// <summary>The gradient of <see cref="AttentionSegmented"/> (as <see cref="AttentionTiledBackward"/>); false when unsupported.</summary>
    public bool AttentionSegmentedBackwardKernel(Storage q, Storage keys, Storage values, Storage output, Storage logSumExp, Storage dOutput,
        Storage dq, Storage dkeys, Storage dvalues, Storage starts, Storage ends, int heads, int headsPerRow, int rowsPerHead, int steps, int dim, float scale,
        AttentionVariant variant = default) =>
        _segmented is { } own && own.AttentionSegmentedBackward(q, keys, values, output, logSumExp, dOutput, dq, dkeys, dvalues, starts, ends, heads, headsPerRow, rowsPerHead, steps, dim, scale, variant);

    /// <summary>
    /// The same attention as <see cref="AttentionDecode"/> for many query rows at once (a prompt, a training sequence),
    /// tiled so query rows share each key and value read; also writes each row's log-sum-exp of the scaled scores to
    /// <paramref name="logSumExp"/> [heads, rowsPerHead] when given (training). The default runs
    /// <see cref="AttentionDecode"/> for inference; training has no default.
    /// </summary>
    public virtual void AttentionTiledKernel(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage? logSumExp, int heads,
        int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        if (logSumExp is null)
        {
            AttentionDecode(q, keys, values, position, y, heads, rowsPerHead, steps, capacity, dim, scale, variant);
            return;
        }

        throw NoKernel(nameof(AttentionTiled));
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
        using var h = new HostCall(this, nameof(AttentionSpans));
        Cpu.CpuBackend.Instance.AttentionSpans(h[q], h[keys], h[values], h[starts], h[ends], h[y], h.Maybe(logSumExp), heads, kvHeads, headsPerTable, rows, keyRows, dim, scale, variant);
    }

    /// <summary>
    /// <see cref="AttentionSpans"/> of a self-attention layer read in place from rows [batch, rows, *] (its packed
    /// projections: no copies of the heads), for inference (no log-sum-exp): query head h &lt; heads of batch b at row i is
    /// q[qOffset + (b·rows + i)·qRow + h·dim .. + dim], its key/value head g = h / (heads / kvHeads) reads keys and values
    /// k / v[offset + (b·rows + c)·kRow + g·dim .. + dim] for c &lt; rows, and the output is written to y[(b·rows + i)·heads·dim
    /// + h·dim .. + dim] ([batch, rows, heads·dim]). Head b·heads + h reads the ranges of table (b·heads + h) /
    /// headsPerTable, as AttentionSpans' heads do. Offsets and row strides in floats; q, k and v may be one storage at three
    /// offsets. By default the heads are copied out to [batch·heads, rows, dim] (keys and values to [batch·kvHeads, rows,
    /// dim]), AttentionSpans runs on them and its output is copied back: a device's own kernel gives those values.
    /// </summary>
    public virtual void AttentionSpansStridedKernel(Storage q, long qOffset, Storage k, long kOffset, Storage v, long vOffset, int qRow, int kRow,
        Storage starts, Storage ends, Storage y, int batch, int heads, int kvHeads, int headsPerTable, int rows, int dim, float scale)
    {
        if (batch <= 0 || heads <= 0 || rows <= 0 || dim <= 0)
        {
            return;
        }

        long queries = (long)batch * heads * rows * dim, keyValues = (long)batch * kvHeads * rows * dim;
        if (queries > int.MaxValue || keyValues > int.MaxValue || Math.Max(qOffset, Math.Max(kOffset, vOffset)) + (long)batch * rows * Math.Max(qRow, kRow) > int.MaxValue)
        {
            throw new ArgumentException($"AttentionSpansStrided: {batch} × {rows} rows of {heads} heads of {dim} do not fit one storage's offsets.");
        }

        var (qc, kc, vc, yc) = (Allocate((int)queries, zeroed: false), Allocate((int)keyValues, zeroed: false), Allocate((int)keyValues, zeroed: false),
            Allocate((int)queries, zeroed: false));
        try
        {
            for (int b = 0; b < batch; b++)
            {
                for (int h = 0; h < heads; h++)
                {
                    Copy2D(q, (int)(qOffset + (long)b * rows * qRow + h * dim), qRow, qc, (b * heads + h) * rows * dim, dim, rows, dim, accumulate: false);
                }

                for (int g = 0; g < kvHeads; g++)
                {
                    Copy2D(k, (int)(kOffset + (long)b * rows * kRow + g * dim), kRow, kc, (b * kvHeads + g) * rows * dim, dim, rows, dim, accumulate: false);
                    Copy2D(v, (int)(vOffset + (long)b * rows * kRow + g * dim), kRow, vc, (b * kvHeads + g) * rows * dim, dim, rows, dim, accumulate: false);
                }
            }

            AttentionSpans(qc, kc, vc, starts, ends, yc, null, batch * heads, batch * kvHeads, headsPerTable, rows, rows, dim, scale);
            for (int b = 0; b < batch; b++)
            {
                for (int h = 0; h < heads; h++)
                {
                    Copy2D(yc, (b * heads + h) * rows * dim, dim, y, b * rows * heads * dim + h * dim, heads * dim, rows, dim, accumulate: false);
                }
            }
        }
        finally
        {
            qc.Release();
            kc.Release();
            vc.Release();
            yc.Release();
        }
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
        using var h = new HostCall(this, nameof(AttentionSpansBackward));
        Cpu.CpuBackend.Instance.AttentionSpansBackward(h[q], h[keys], h[values], h[starts], h[ends], h[output], h[logSumExp], h[dOutput], h[dq], h[dkeys], h[dvalues], heads, kvHeads, headsPerTable, rows, keyRows, dim, scale, variant);
    }

    /// <summary>
    /// True when attention over key ranges runs faster on this device composed — the full scores [heads, rows, keyRows]
    /// through <see cref="BatchedMatMul"/>, <see cref="ScaleMaskSoftmax"/> (with <paramref name="mask"/>, [rows, keyRows],
    /// or none when every row sees every key) and <see cref="BatchedMatMul"/> again (<see cref="ComposedAttention"/>) —
    /// than through <see cref="AttentionSpans"/>, for this shape. Given the
    /// real operands (one table of ranges, as many key/value heads as query heads, no soft-cap), so a device measures both
    /// on itself (into scratch memory of its own), once per shape, and keeps the choice as it keeps its
    /// other measured choices (per device and driver). Called only when the device's free memory holds the scores with
    /// margin (<see cref="AvailableMemory"/>; <c>Tensor.AttentionFastest</c> checks). With <paramref name="training"/>
    /// (the gradient is recorded) the choice is the faster of the two with their gradients, forward and backward
    /// (<see cref="AttentionSpansBackward"/> against the composed path's products and softmax gradient,
    /// <see cref="ComposedAttentionTraining"/>), measured and kept apart from the choice for inference. The default:
    /// false, AttentionSpans (it never holds the scores, so it never runs out of memory for them).
    /// </summary>
    public virtual bool PrefersComposedAttention(Storage q, Storage keys, Storage values, Storage starts, Storage ends, Storage? mask, int heads, int rows,
        int keyRows, int dim, float scale, bool training = false) => false;

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
    /// <see cref="ComposedAttention"/> and its gradient, as a training step runs it, for measuring
    /// (<see cref="PrefersComposedAttention"/> with training): the forward pass, then with <paramref name="dOutput"/>
    /// [heads, rows, dim] dvalues = weightsᵀ · dOutput, dweights = dOutput · valuesᵀ (into <paramref name="scores"/>), the
    /// softmax's gradient (into <paramref name="gradients"/>, heads · rows · keyRows floats), dq = dscores · keys and
    /// dkeys = dscoresᵀ · q. Writes only its scratch arguments; the scale's and the mask's element-wise passes are left out.
    /// </summary>
    protected void ComposedAttentionTraining(Storage q, Storage keys, Storage values, Storage? mask, Storage scores, Storage weights, Storage gradients,
        Storage y, Storage dOutput, Storage dq, Storage dkeys, Storage dvalues, int heads, int rows, int keyRows, int dim, float scale)
    {
        ComposedAttention(q, keys, values, mask, scores, weights, y, heads, rows, keyRows, dim, scale);
        BatchedMatMul(weights, dOutput, dvalues, heads, keyRows, dim, rows, transA: true, transB: false, beta: 0f);
        BatchedMatMul(dOutput, values, scores, heads, rows, keyRows, dim, transA: false, transB: true, beta: 0f);
        SoftmaxBackward(weights, scores, gradients, heads * rows, keyRows, log: false);
        BatchedMatMul(gradients, keys, dq, heads, rows, dim, keyRows, transA: false, transB: false, beta: 0f);
        BatchedMatMul(gradients, q, dkeys, heads, keyRows, dim, rows, transA: true, transB: false, beta: 0f);
    }

    /// <summary>
    /// Bytes of device memory new tensors can take now, as the device reports it (its free memory and the blocks its pool
    /// keeps for reuse, within <see cref="ComputeResources.GpuMemoryLimit"/>), or null when it reports none. Guards paths
    /// that need much memory at once (the composed attention's scores). Read from <see cref="IMemoryReport"/>.
    /// </summary>
    public long? AvailableMemory() => _memoryReport?.AvailableMemory();

    /// <summary>
    /// Bytes of device memory the device's driver reports free now (outside every pool, before any reserve), or null when
    /// it reports none. With <see cref="BackendHardware.MemoryBytes"/> and <see cref="GetMemoryUsage"/> it shows what the
    /// library does not hold: the driver's context, its loaded kernels and other programs. Read from <see cref="IMemoryReport"/>.
    /// </summary>
    public long? FreeMemory() => _memoryReport?.FreeMemory();

    /// <summary>Starts counting the peak of the bytes in use (<see cref="MemoryUsage.Peak"/>) again from what is in use now.</summary>
    public virtual void ResetPeakMemoryUsage()
    {
    }

    // ---------------------------------------------------------------- incremental decoding (positions live on the device)

    /// <summary>mask[i, j] = j ≤ position + i ? 0 : -1e9 for a [rows, capacity] mask; position is read from device memory.</summary>
    public virtual void DecoderMaskKernel(Storage position, Storage mask, int rows, int capacity)
    {
        using var h = new HostCall(this, nameof(DecoderMask));
        Cpu.CpuBackend.Instance.DecoderMask(h[position], h[mask], rows, capacity);
    }

    /// <summary>cache[bh, position + t, :] = source[bh, t, :] for [heads, steps, dim] → [heads, capacity, dim].</summary>
    public virtual void KeyValueWriteKernel(Storage source, Storage cache, Storage position, int heads, int steps, int capacity, int dim)
    {
        using var h = new HostCall(this, nameof(KeyValueWrite));
        Cpu.CpuBackend.Instance.KeyValueWrite(h[source], h[cache], h[position], heads, steps, capacity, dim);
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
        using var h = new HostCall(this, nameof(SampleRows));
        Cpu.CpuBackend.Instance.SampleRows(h[logits], h[ids], h[stats], h[step], rows, vocabulary, rowStride, rowOffset, temperature, topK, topP, minP, seed);
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
        using var h = new HostCall(this, nameof(PenalizeRows));
        Cpu.CpuBackend.Instance.PenalizeRows(h[logits], h[work], h[history], h[length], rows, vocabulary, rowStride, rowOffset, capacity, lastN, repeat, presence, frequency);
    }

    /// <summary>history[r, length % capacity] = ids[r] for every row (length read from device memory, not advanced).</summary>
    public virtual void HistoryPushKernel(Storage ids, Storage history, Storage length, int rows, int capacity)
    {
        using var h = new HostCall(this, nameof(HistoryPush));
        Cpu.CpuBackend.Instance.HistoryPush(h[ids], h[history], h[length], rows, capacity);
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
    private long _largest;

    /// <summary>The bytes in use, cached and offloaded, the limit and the peak in use, as <see cref="Backend.GetMemoryUsage"/> reports them.</summary>
    public MemoryUsage Usage => new(Interlocked.Read(ref _inUse), Interlocked.Read(ref _cached), limit(), Interlocked.Read(ref _offloaded),
        Math.Max(Interlocked.Read(ref _peak), Interlocked.Read(ref _inUse)), Interlocked.Read(ref _largest));

    /// <summary>Starts the peak again from the bytes in use now, and the largest block from none (<see cref="Backend.ResetPeakMemoryUsage"/>).</summary>
    public void ResetPeak()
    {
        Interlocked.Exchange(ref _peak, Interlocked.Read(ref _inUse));
        Interlocked.Exchange(ref _largest, 0);
    }

    // Raises the largest block taken to `bytes` when larger.
    private void Largest(long bytes)
    {
        long largest = Interlocked.Read(ref _largest);
        while (bytes > largest)
        {
            long seen = Interlocked.CompareExchange(ref _largest, bytes, largest);
            if (seen == largest)
            {
                return;
            }

            largest = seen;
        }
    }

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
    public void Allocated(long bytes)
    {
        Largest(bytes);
        Peak(Interlocked.Add(ref _inUse, bytes));
    }

    /// <summary>Moves bytes taken from the pool from cached to in use.</summary>
    public void Reused(long bytes)
    {
        Largest(bytes);
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

/// <summary>The activation <see cref="Backend.Convolution"/> applies to each output after the bias (a fused inference epilogue).</summary>
public enum ConvActivation
{
    /// <summary>None: the output is the sum plus the bias.</summary>
    None = 0,

    /// <summary>max(v, 0), as <see cref="UnaryOp.Relu"/>.</summary>
    Relu = 1,

    /// <summary>1 / (1 + e^-v), as <see cref="UnaryOp.Sigmoid"/>.</summary>
    Sigmoid = 2,

    /// <summary>tanh v, as <see cref="UnaryOp.Tanh"/>.</summary>
    Tanh = 3,

    /// <summary>GELU (tanh approximation), as <see cref="UnaryOp.Gelu"/>.</summary>
    Gelu = 4,

    /// <summary>v · sigmoid(v), as <see cref="UnaryOp.Silu"/>.</summary>
    Silu = 5,
}
