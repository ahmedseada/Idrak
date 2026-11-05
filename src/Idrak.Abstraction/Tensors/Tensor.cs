// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;
using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Diagnostics;

namespace Idrak.Abstraction;

/// <summary>
/// An n-dimensional array of 32-bit floats stored on a <see cref="Idrak.Abstraction.Device"/>, with automatic
/// differentiation. Tensors own device memory: dispose them (or create them inside a
/// <see cref="TensorScope"/>) to recycle it promptly, especially on the GPU.
/// </summary>
/// <example>
/// <code>
/// using var x = Tensor.From(new float[,] { { 1, 2 }, { 3, 4 } }, requiresGrad: true);
/// using var y = (x * x).Sum();
/// y.Backward();              // x.Grad is now 2 * x
/// </code>
/// </example>
public sealed partial class Tensor : IDisposable
{
    internal readonly int[] _shape;
    private Tensor[]? _parents;
    private Action<Tensor>? _backward;
    private string? _operation;
    private int _disposed;

    // Set when a backward pass ran this tensor's step and freed it: a later pass that brings it a gradient would drop it.
    private bool _consumed;

    private Tensor(int[] shape, Storage storage, Device device, bool track)
    {
        _shape = shape;
        _storage = storage;
        Device = device;
        Size = checked((int)storage.ElementCount);                         // elements of its type (plan 37, T0)
        if (track)
        {
            TensorScope.Track(this);
        }
    }

    // A view (plan 37, V1): `layout` over `storage` (another tensor's, referenced once more by the caller), its values read
    // through the layout until something takes its storage directly (Materialize). `shape` is the layout's own, or (a
    // reshape strides cannot express) another of as many elements, read in the layout's row-major order.
    private Tensor(TensorLayout layout, int[] shape, Storage storage, Device device)
    {
        _shape = shape;
        _storage = storage;
        _layout = layout;
        _pending = true;
        Device = device;
        Size = checked((int)layout.Elements);
        TensorScope.Track(this);
    }

    /// <summary>Returns device memory to the pool if the tensor was never disposed.</summary>
    ~Tensor() => _storage.Release();

    // The storage this tensor holds: its own, or (a view not copied yet) the one it reads through _layout.
    private Storage _storage;

    // The layout: null for a contiguous tensor from element 0 (made on demand by Layout), else a view's.
    private TensorLayout? _layout;

    // A view whose values still sit in another tensor's storage (read through _layout).
    private volatile bool _pending;

    // The version the storage a view read had when the view was copied to its own (see VersionOf).
    private int _versionOffset;

    /// <summary>
    /// Where the elements sit in <see cref="Storage"/> (plan 37, V0): contiguous from element 0 for every tensor but a view
    /// not copied yet, whose elements sit in another tensor's storage along its strides (see <see cref="IsView"/>). The
    /// elements in row-major order of <see cref="Shape"/> are those of the layout in its row-major order; its shape is
    /// <see cref="Shape"/> but for a reshape of a view that strides do not express (<see cref="TensorLayout.Reshape"/>),
    /// which keeps the view's layout.
    /// </summary>
    public TensorLayout Layout => _layout ??= TensorLayout.Contiguous(_shape);

    // The layout of the storage this tensor's elements sit in (its own when contiguous, a view's source otherwise), its
    // row-major order this tensor's: what operations that read strides (BackendCapabilities.StridedInputs) are given, with
    // _storage, in place of Storage.
    internal TensorLayout ElementOrder => _pending ? _layout! : Layout;

    /// <summary>
    /// Whether this tensor is a view not copied yet (<see cref="Permute"/>, <see cref="Narrow"/>, <see cref="Select"/>,
    /// <see cref="Expand"/> on a device that reads strides, and reshapes of those): its elements are another tensor's, read through
    /// <see cref="Layout"/>. Reading <see cref="Storage"/> copies them to a block of its own once (what the copies before
    /// views were); operations that read strides do not.
    /// </summary>
    public bool IsView => _pending;

    /// <summary>
    /// This tensor as a contiguous one: itself when it is (every tensor but a strided view), else a copy of its elements in
    /// row-major order, made by the device's strided copy, through which gradients flow back to this tensor.
    /// </summary>
    public Tensor Contiguous()
    {
        ThrowIfDisposed();
        if (!_pending || _layout!.IsContiguousFromStart)
        {
            return this;
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(_shape, Device);
        var layout = _layout;
        Backend.StridedCopy(_storage, layout.Offset, layout.Shape, layout.Strides, y.Storage);
        TensorViews.CountScratch((long)Size * sizeof(float));
        if (WillRecord(this))
        {
            var x = this;
            y.Record("contiguous", g => x.Backend.Axpy(g.Storage, x.GradStorage(), x.Size, 1f), x);
        }

        return Traced("contiguous", y, start);
    }

    // Whether views are made on this tensor's device: on (TensorViews.Enabled) where its kernels read strides, except while
    // recorded activations are kept as bfloat16 (ActivationMemory.CompressToBFloat16): there a copy is packed to half its
    // size, where a view would keep its source's float values for as long as it lives.
    internal bool MakesViews => TensorViews.Enabled && !(ActivationMemory.BFloat16 && Autograd.IsEnabled) && IsFloat32
        && Backend.Capabilities.StridedInputs.Contains(Operations.Ops.StridedCopy);

    // A view of this tensor's elements through `layout` (over the storage this tensor's elements sit in); not tracked for
    // gradients here (the caller records its backward step).
    internal Tensor View(TensorLayout layout, int[]? shape = null)
    {
        var storage = _storage;
        storage.AddRef();
        var view = new Tensor(layout, shape ?? layout.Shape.ToArray(), storage, Device);
        lock (storage)
        {
            var pending = storage.PendingViews ??= [];
            if (pending.Count >= 32)
            {
                pending.RemoveAll(w => !w.TryGetTarget(out var t) || !t._pending);   // views copied or gone
            }

            pending.Add(new WeakReference<Tensor>(view));
        }

        TensorViews.CountMade();
        return view;
    }

    // This tensor's layout over the storage its elements sit in, of its own shape (what a view of it builds on): its own
    // when contiguous, a view's otherwise; a reshape of a view that strides do not express is copied to its own first.
    internal TensorLayout SourceLayout
    {
        get
        {
            if (!_pending)
            {
                return Layout;
            }

            var layout = _layout!;
            if (layout.Shape.SequenceEqual(_shape))
            {
                return layout;
            }

            if (layout.Reshape(_shape) is { } strided)
            {
                return strided;
            }

            Materialize();
            return Layout;
        }
    }

    // Copies a view's elements to a storage of its own, once: from now on it is an ordinary contiguous tensor. A source whose
    // values were released while views read them is evicted after the last of them, unless this copy is made because the
    // source is being handed out (flushing).
    private Storage Materialize(bool flushing = false)
    {
        Storage source, own;
        lock (this)
        {
            if (!_pending)
            {
                return _storage;
            }

            source = _storage;
            var layout = _layout!;
            own = Backend.Allocate(Size, zeroed: false);
            Backend.StridedCopy(source, layout.Offset, layout.Shape, layout.Strides, own);
            _versionOffset = source.Version;
            _storage = own;
            _layout = null;
            _pending = false;
            source.Release();
            TensorViews.CountMaterialized((long)Size * sizeof(float));
        }

        if (!flushing)
        {
            DropUnviewed(source);
        }

        return own;
    }

    // Before a tensor hands out a storage with views of it not copied yet (a direct access may write it): the views copy
    // their elements first, so each keeps the values it was made with.
    private static void FlushViews(Storage storage)
    {
        List<WeakReference<Tensor>>? pending;
        lock (storage)
        {
            pending = storage.PendingViews;
            storage.PendingViews = null;
        }

        if (pending is null)
        {
            return;
        }

        foreach (var weak in pending)
        {
            if (weak.TryGetTarget(out var view) && view._pending && view._disposed == 0)
            {
                view.Materialize(flushing: true);
            }
        }
    }

    /// <summary>
    /// A storage holding this tensor's elements contiguous from element 0, for an operation that only reads them: its own
    /// storage (no views copied first: reading leaves them as they are), a view's source when the view is contiguous from
    /// its start, else a short-lived copy given in <paramref name="scratch"/>, which the caller releases after the operation.
    /// </summary>
    internal Storage ReadStorage(out Storage? scratch)
    {
        scratch = null;
        if (!_pending)
        {
            return _storage;
        }

        var layout = _layout!;
        if (layout.IsContiguousFromStart)
        {
            return _storage;
        }

        lock (this)
        {
            if (!_pending)
            {
                return _storage;
            }

            scratch = Backend.Allocate(Size, zeroed: false);
            Backend.StridedCopy(_storage, layout.Offset, layout.Shape, layout.Strides, scratch);
        }

        TensorViews.CountScratch((long)Size * sizeof(float));
        return scratch;
    }

    // The in-place writes counted on the storage this tensor's values are in (see Storage.Version), stable across a view's copy.
    private long VersionOf => _storage.Version + _versionOffset;

    /// <summary>The device this tensor's data lives on.</summary>
    public Device Device { get; }

    /// <summary>The length of each dimension, outermost first.</summary>
    public ReadOnlySpan<int> Shape => _shape;

    /// <summary>The number of dimensions (0 for a scalar).</summary>
    public int Rank => _shape.Length;

    /// <summary>The total number of elements.</summary>
    public int Size { get; }

    /// <summary>
    /// What the elements are (plan 37, T0): float32 for every tensor an operation makes; int32 or int64 for index tensors
    /// (<see cref="Indices(ReadOnlySpan{int}, Device)"/>, a top-k's indices, a sampler's ids), which the operations that
    /// take indices read. Float operations read float32 tensors; <see cref="ToArray"/> gives any type's values as floats.
    /// </summary>
    public ElementType ElementType => _storage.ElementType;

    // Whether the elements are float32 (what every float operation reads).
    internal bool IsFloat32 => ReferenceEquals(_storage.ElementType, ElementType.Float32);

    /// <summary>
    /// Whether gradients flow back to this tensor. Set it on inputs you want to differentiate with
    /// respect to; results of operations inherit it from their inputs.
    /// </summary>
    public bool RequiresGrad
    {
        get;
        set
        {
            if (!IsLeaf)
            {
                throw new InvalidOperationException("RequiresGrad can only be changed on leaf tensors (ones not produced by an operation). Use Detach() first.");
            }

            if (field != value && _storage is { } storage && storage.Backend.Offload is not null)
            {
                TensorOffloading.Hooks?.TrainableChanged(this, value);                  // frozen weights move to system memory first
            }

            field = value;
        }
    }

    /// <summary>The accumulated gradient after <see cref="Backward()"/>, or null if none has been computed.</summary>
    public Tensor? Grad { get; private set; }

    /// <summary>True when this tensor was created directly rather than as the output of a recorded operation.</summary>
    public bool IsLeaf => _backward is null;

    /// <summary>
    /// The device memory holding the values contiguous from element 0 (shared by reshapes of the same values), for code that
    /// runs kernels on it. A view not copied yet (<see cref="IsView"/>) copies its elements to a block of its own first; a
    /// tensor whose storage has such views makes them copy theirs first (the access may write it).
    /// </summary>
    public Storage Storage
    {
        get
        {
            if (_pending)
            {
                return Materialize();
            }

            var storage = _storage;
            if (storage.PendingViews is not null)
            {
                FlushViews(storage);
            }

            return storage;
        }
    }

    /// <summary>The backend of <see cref="Device"/>, which runs kernels on <see cref="Storage"/>.</summary>
    public Backend Backend => Device.Backend;

    // ---------------------------------------------------------------- creation

    /// <summary>Creates a tensor from a flat array with the given shape.</summary>
    public static Tensor From(ReadOnlySpan<float> values, ReadOnlySpan<int> shape, Device? device = null, bool requiresGrad = false)
    {
        var t = Empty(shape, device ?? Device.Default);
        if (values.Length != t.Size)
        {
            t.Dispose();
            throw new ArgumentException($"{values.Length} values cannot fill a tensor of shape {FormatShape(shape)} ({t.Size} elements).", nameof(values));
        }

        t.Backend.Upload(values, t.Storage);
        t.RequiresGrad = requiresGrad;
        return t;
    }

    /// <summary>Creates a 1-D tensor from an array.</summary>
    public static Tensor From(float[] values, Device? device = null, bool requiresGrad = false) =>
        From(values, [values.Length], device, requiresGrad);

    /// <summary>Creates a 2-D tensor (rows x columns) from a rectangular array.</summary>
    public static Tensor From(float[,] values, Device? device = null, bool requiresGrad = false) =>
        From(MemoryMarshalHelpers.Flatten(values), [values.GetLength(0), values.GetLength(1)], device, requiresGrad);

    /// <summary>Creates a scalar (rank-0) tensor.</summary>
    public static Tensor Scalar(float value, Device? device = null, bool requiresGrad = false) =>
        From([value], [], device, requiresGrad);

    /// <summary>Creates a tensor filled with zeros.</summary>
    public static Tensor Zeros(ReadOnlySpan<int> shape, Device? device = null, bool requiresGrad = false)
    {
        var t = Empty(shape, device ?? Device.Default, zeroed: true);
        t.RequiresGrad = requiresGrad;
        return t;
    }

    /// <summary>Creates a tensor filled with ones.</summary>
    public static Tensor Ones(ReadOnlySpan<int> shape, Device? device = null, bool requiresGrad = false) =>
        Full(shape, 1f, device, requiresGrad);

    /// <summary>Creates a tensor with every element set to <paramref name="value"/>.</summary>
    public static Tensor Full(ReadOnlySpan<int> shape, float value, Device? device = null, bool requiresGrad = false)
    {
        var t = Empty(shape, device ?? Device.Default);
        t.Backend.Fill(t.Storage, t.Size, value);
        t.RequiresGrad = requiresGrad;
        return t;
    }

    /// <summary>Creates a tensor of values drawn uniformly from [<paramref name="low"/>, <paramref name="high"/>).</summary>
    public static Tensor Uniform(ReadOnlySpan<int> shape, float low, float high, Random? random = null, Device? device = null, bool requiresGrad = false)
    {
        random ??= Random.Shared;
        var values = new float[ElementCount(shape)];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = low + (high - low) * random.NextSingle();
        }

        return From(values, shape, device, requiresGrad);
    }

    /// <summary>Creates a tensor of normally distributed values (Box-Muller).</summary>
    public static Tensor Normal(ReadOnlySpan<int> shape, float mean = 0f, float std = 1f, Random? random = null, Device? device = null, bool requiresGrad = false)
    {
        random ??= Random.Shared;
        var values = new float[ElementCount(shape)];
        for (int i = 0; i < values.Length; i += 2)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = random.NextDouble();
            double r = Math.Sqrt(-2.0 * PortableMath.Log(u1));
            values[i] = (float)(mean + std * r * PortableMath.Cos(2 * Math.PI * u2));
            if (i + 1 < values.Length)
            {
                values[i + 1] = (float)(mean + std * r * PortableMath.Sin(2 * Math.PI * u2));
            }
        }

        return From(values, shape, device, requiresGrad);
    }

    /// <summary>
    /// Creates a tensor that no <see cref="TensorScope"/> captures, for long-lived state such as parameters, buffers of a
    /// cache or a packed weight format, or optimizer state: it lives until it is disposed, even when created inside a scope.
    /// </summary>
    public static Tensor Persistent(ReadOnlySpan<float> values, ReadOnlySpan<int> shape, Device? device = null, bool requiresGrad = false)
    {
        var t = Empty(shape, device ?? Device.Default, track: false);
        if (values.Length != t.Size)
        {
            t.Dispose();
            throw new ArgumentException($"{values.Length} values cannot fill a tensor of shape {FormatShape(shape)} ({t.Size} elements).", nameof(values));
        }

        t.Backend.Upload(values, t.Storage);
        t.RequiresGrad = requiresGrad;
        return t;
    }

    /// <summary>
    /// Creates a tensor of zeros that no <see cref="TensorScope"/> captures (see <see cref="Persistent"/>), allocated on the
    /// device without a copy from the host.
    /// </summary>
    public static Tensor PersistentZeros(ReadOnlySpan<int> shape, Device? device = null, bool requiresGrad = false)
    {
        var t = Empty(shape, device ?? Device.Default, zeroed: true, track: false);
        t.RequiresGrad = requiresGrad;
        return t;
    }

    // A tensor no scope captures whose memory is given back at once, its values made by `recompute` on its first restore
    // (Backend.Restore): a deferred tensor moved inside DeferredTensors.StayDeferred.
    internal static Tensor Evicted(ReadOnlySpan<int> shape, Device device, bool requiresGrad, Action<Storage> recompute)
    {
        var t = Empty(shape, device, track: false);
        device.Backend.Evict(t.Storage, recompute);
        t.RequiresGrad = requiresGrad;
        return t;
    }

    /// <summary>Overwrites this tensor's data in place with <paramref name="values"/> (same element count).</summary>
    public void Load(ReadOnlySpan<float> values)
    {
        ThrowIfDisposed();
        Backend.Upload(values, Storage);
    }

    /// <summary>
    /// A tensor of <paramref name="shape"/> on <paramref name="device"/> whose values a kernel is about to write: zeroed
    /// when <paramref name="zeroed"/>, else holding whatever the memory held; owned by the current <see cref="TensorScope"/>
    /// when <paramref name="track"/>.
    /// </summary>
    public static Tensor Empty(ReadOnlySpan<int> shape, Device device, bool zeroed = false, bool track = true)
    {
        int size = ElementCount(shape);
        return new Tensor(shape.ToArray(), device.Backend.Allocate(size, zeroed), device, track);
    }

    /// <summary>
    /// <see cref="Empty(ReadOnlySpan{int}, Device, bool, bool)"/> of elements of <paramref name="type"/> (plan 37, T0): the
    /// storage holds them in its words (<see cref="Backend.Allocate(ElementType, int, bool)"/>).
    /// </summary>
    public static Tensor Empty(ReadOnlySpan<int> shape, Device device, ElementType type, bool zeroed = false, bool track = true)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(type);
        int size = ElementCount(shape);
        return new Tensor(shape.ToArray(), device.Backend.Allocate(type, size, zeroed), device, track);
    }

    // A tensor of `shape` over `storage`, which it takes (its one reference), made by the device outside Allocate (a file's
    // mapping, plan 37 W3: IMappedStorageBackend.MapStorage); not tracked by a scope.
    internal static Tensor OverStorage(Storage storage, ReadOnlySpan<int> shape, Device device)
    {
        if (storage.ElementCount != ElementCount(shape))
        {
            storage.Release();
            throw new ArgumentException($"A storage of {storage.ElementCount} {storage.ElementType.Name} elements cannot hold {FormatShape(shape)}.", nameof(shape));
        }

        return new Tensor(shape.ToArray(), storage, device, track: false);
    }

    private static int ElementCount(ReadOnlySpan<int> shape)
    {
        int size = 1;
        foreach (int d in shape)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(d, nameof(shape));
            size = checked(size * d);
        }

        return size;
    }

    // ---------------------------------------------------------------- reading data

    /// <summary>Copies the elements to a new array (row-major).</summary>
    public float[] ToArray()
    {
        ThrowIfDisposed();
        if (!IsFloat32)
        {
            return Widened(0, Size);                                       // another type's values as floats (plan 37, T0)
        }

        var result = GC.AllocateUninitializedArray<float>(Size);
        var storage = ReadStorage(out var scratch);
        try
        {
            Backend.Download(storage, result);
        }
        finally
        {
            scratch?.Release();
        }

        return result;
    }

    /// <summary>Copies a 2-D tensor to a rectangular array.</summary>
    public float[,] ToArray2D()
    {
        if (Rank != 2)
        {
            throw new InvalidOperationException($"ToArray2D needs a 2-D tensor, but the shape is {FormatShape(_shape)}.");
        }

        var flat = ToArray();
        var result = new float[_shape[0], _shape[1]];
        MemoryMarshalHelpers.Unflatten(flat, result);
        return result;
    }

    /// <summary>Returns the value of a single-element tensor, such as a loss.</summary>
    public float Item()
    {
        if (Size != 1)
        {
            throw new InvalidOperationException($"Item() needs a tensor with exactly one element, but the shape is {FormatShape(_shape)}.");
        }

        ThrowIfDisposed();
        if (!IsFloat32)
        {
            return Widened(0, 1)[0];
        }

        Span<float> value = stackalloc float[1];
        var storage = ReadStorage(out var scratch);
        try
        {
            Backend.Download(storage, value);
        }
        finally
        {
            scratch?.Release();
        }

        return value[0];
    }

    /// <summary>
    /// Returns this tensor on <paramref name="device"/>: the same instance if it is already there,
    /// otherwise a copy (which does not track gradients back to this tensor).
    /// </summary>
    public Tensor To(Device device)
    {
        ThrowIfDisposed();
        if (device == Device)
        {
            return this;
        }

        if (!IsFloat32)
        {
            return FromBytes(ToBytes(), ElementType, _shape, device);       // the same elements, of the same type
        }

        return From(ToArray(), _shape, device, RequiresGrad && IsLeaf);
    }

    /// <summary>Returns a copy of the data that is not connected to the autograd graph.</summary>
    public Tensor Clone()
    {
        ThrowIfDisposed();
        var t = IsFloat32 ? Empty(_shape, Device) : Empty(_shape, Device, ElementType);
        var storage = ReadStorage(out var scratch);
        try
        {
            Backend.Copy(storage, t.Storage, t.Storage.Length);            // Size floats, or a typed tensor's words
        }
        finally
        {
            scratch?.Release();
        }

        return t;
    }

    /// <summary>Whether <see cref="Dispose"/> has released this tensor.</summary>
    public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

    /// <summary>Returns a tensor sharing this tensor's data but cut off from the autograd graph.</summary>
    public Tensor Detach()
    {
        ThrowIfDisposed();
        Storage.AddRef();
        return new Tensor(_shape, Storage, Device, track: true);
    }

    /// <summary>Resets the accumulated gradient to zero.</summary>
    public void ZeroGrad()
    {
        if (Grad is not null)
        {
            Backend.Fill(Grad.Storage, Grad.Size, 0f);
        }
    }

    /// <summary>Releases the gradient's memory (the next backward pass writes a fresh one without zero-filling it).</summary>
    public void ReleaseGrad()
    {
        Grad?.Dispose();
        Grad = null;
    }

    /// <summary>Releases the tensor's memory (and its gradient's) back to the device pool.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        GC.SuppressFinalize(this);
        Grad?.Dispose();
        Grad = null;
        _parents = null;
        _backward = null;
        bool view;
        var storage = _storage;
        lock (this)
        {
            view = _pending;
            _pending = false;                                            // a view's source is no longer read through it
            storage.Release();
        }

        if (view)
        {
            DropUnviewed(storage);
        }
    }

    /// <inheritdoc />
    public override string ToString()
    {
        if (_disposed != 0)
        {
            return $"Tensor(shape={FormatShape(_shape)}, device={Device}, disposed)";
        }

        const int MaxShown = 64;
        var data = ToArray();
        var sb = new StringBuilder();
        sb.Append("Tensor(shape=").Append(FormatShape(_shape)).Append(", device=").Append(Device);
        if (RequiresGrad)
        {
            sb.Append(", requiresGrad");
        }

        sb.Append(")\n");
        if (Rank == 2 && data.Length <= MaxShown)
        {
            for (int r = 0; r < _shape[0]; r++)
            {
                sb.Append(r == 0 ? "[[" : " [");
                for (int c = 0; c < _shape[1]; c++)
                {
                    sb.Append(c == 0 ? "" : ", ").Append(data[r * _shape[1] + c].ToString("0.0000", CultureInfo.InvariantCulture));
                }

                sb.Append(r == _shape[0] - 1 ? "]]" : "]\n");
            }
        }
        else
        {
            sb.Append('[');
            for (int i = 0; i < Math.Min(data.Length, MaxShown); i++)
            {
                sb.Append(i == 0 ? "" : ", ").Append(data[i].ToString("0.0000", CultureInfo.InvariantCulture));
            }

            sb.Append(data.Length > MaxShown ? ", ...]" : "]");
        }

        return sb.ToString();
    }

    /// <summary>A shape as messages show it: [2, 3].</summary>
    public static string FormatShape(ReadOnlySpan<int> shape)
    {
        // Written once into a stack buffer (one string allocated), not copied to an array and joined.
        var text = new DefaultInterpolatedStringHandler(2 * shape.Length + 2, shape.Length, CultureInfo.InvariantCulture, stackalloc char[64]);
        text.AppendLiteral("[");
        for (int i = 0; i < shape.Length; i++)
        {
            if (i > 0)
            {
                text.AppendLiteral(", ");
            }

            text.AppendFormatted(shape[i]);
        }

        text.AppendLiteral("]");
        return text.ToStringAndClear();
    }

    /// <summary>
    /// Releases the memory of this tensor's values while it stays in the autograd graph (its gradient still flows through
    /// it): for a training result whose values no backward step reads, once its last forward use is done, so it is not
    /// kept until the end of the step. The memory is shared with every view of the values, which lose them too: call it
    /// only where every tensor sharing the values is done with them and no consumer's backward reads them. Kernels given
    /// the values afterwards fail.
    /// </summary>
    internal void DropValue()
    {
        if (_disposed != 0)
        {
            return;
        }

        var storage = _storage;
        if (!_pending && HasLiveViews(storage, null))
        {
            storage.DropWhenUnviewed = true;                             // views read them still: dropped after the last (plan 37, V1b)
            return;
        }

        Backend.Evict(Storage, null);
    }

    // The storage this tensor holds as it is (a view's source while the view is not copied): for comparing which tensors
    // share memory without copying a view (ActivationMemory.Release).
    internal Storage Held => _storage;

    // Whether views not copied yet, other than `except`, read `storage`.
    private static bool HasLiveViews(Storage storage, Tensor? except)
    {
        lock (storage)
        {
            foreach (var weak in storage.PendingViews ?? [])
            {
                if (weak.TryGetTarget(out var view) && !ReferenceEquals(view, except) && view._pending && view._disposed == 0)
                {
                    return true;
                }
            }
        }

        return false;
    }

    // After a view stopped reading `source` (copied to its own, or disposed): a source whose values were released while
    // views read them is evicted once none does.
    private void DropUnviewed(Storage source)
    {
        if (source.DropWhenUnviewed && source.Alive && !source.Evicted && !HasLiveViews(source, this))
        {
            source.DropWhenUnviewed = false;
            source.Backend.Evict(source, null);
        }
    }

    /// <summary>
    /// Releases the memory of this tensor's values like <see cref="DropValue"/>, but <paramref name="recompute"/> fills
    /// them in again when a backward step needs them (see <see cref="Backward(Tensor)"/>): activations cheaper to recompute
    /// than to keep. The recomputation must only read tensors that are still kept.
    /// </summary>
    public void Evict(Action<Tensor> recompute)
    {
        if (_disposed == 0)
        {
            var self = this;
            Backend.Evict(Storage, _ => recompute(self));
            Storage.RecomputedFor = self;                                // its own backward step needs no recomputation
        }
    }

    // Operations whose backward step does not read their own output (only inputs, or nothing): an evicted output is not
    // restored for them.
    private static readonly HashSet<string> OwnValuesUnread =
    [
        "reshape", "permute", "add_bias", "add_dropout", "rope", "group_affine", "lora", "lora_fused", "gated_activation",
        "matmul", "matmul_bias", "matmul_frozen", "matmul_bf16", "matmul_int8", "matmul_int4", "matmul_packed", "embedding",
    ];

    // Operations whose backward step reads its inputs' bfloat16 words (Storage.Packed) itself: an input evicted with a
    // packed copy is not unpacked for them.
    private static readonly HashSet<string> ReadsPackedInputs = ["gated_activation"];

    // The storages a node's backward step may read (its own and its inputs'), into reads.
    private static void Reads(Tensor node, List<Storage> reads)
    {
        reads.Clear();
        if (node._operation == "reshape")
        {
            return;                                                      // a view passes its gradient on, reading no values
        }

        if (!ReferenceEquals(node._storage.RecomputedFor, node) && !OwnValuesUnread.Contains(node._operation ?? ""))
        {
            reads.Add(node._storage);
        }

        bool packedInputs = ReadsPackedInputs.Contains(node._operation ?? "");
        foreach (var parent in node._parents ?? [])
        {
            if (!(packedInputs && parent._storage.Evicted && parent._storage.Packed is not null))
            {
                reads.Add(parent._storage);
            }
        }
    }

    /// <summary>
    /// Keeps this tensor's values as bfloat16 until a backward step reads them: a packed copy (half the memory) replaces
    /// the float values, which are unpacked when needed (<see cref="Evict"/>). For activations that only backward steps
    /// read, and whose precision bfloat16 keeps (inputs of tensor-core products, which round them to bfloat16 anyway).
    /// </summary>
    internal void CompressToBFloat16()
    {
        if (_disposed != 0 || Storage.Evicted || Storage.Packed is not null || Size < 2)
        {
            return;
        }

        var packed = Backend.Allocate((Size + 1) / 2, zeroed: false);
        Backend.PackBFloat16(Storage, packed, Size);
        EvictToPacked(packed);
    }

    /// <summary>
    /// Releases the values, keeping <paramref name="packed"/> (their bfloat16 words, which this tensor now owns) to unpack
    /// them from when a backward step needs them, or for kernels that read the words themselves (<see cref="Storage.Packed"/>).
    /// </summary>
    public void EvictToPacked(Storage packed)
    {
        if (_disposed != 0 || Storage.Evicted || Storage.Packed is not null)
        {
            packed.Release();
            return;
        }

        int n = Size;
        Backend.Evict(Storage, s => s.Backend.BFloat16Dequantize(packed, s, 1, n));
        Storage.Packed = packed;
        Storage.Released = packed.Release;
    }

    /// <summary>Runs <paramref name="action"/> with the values of <paramref name="tensors"/> in memory (evicted ones restored, then evicted again).</summary>
    internal static void WithValues(ReadOnlySpan<Tensor> tensors, Action action)
    {
        var restored = new List<Storage>();
        foreach (var tensor in tensors)
        {
            if (tensor.Storage.Evicted && tensor.Backend.Restore(tensor.Storage))
            {
                restored.Add(tensor.Storage);
            }
        }

        action();
        foreach (var storage in restored)
        {
            storage.Backend.Evict(storage, storage.Recompute);
        }
    }

    /// <summary>Throws <see cref="ObjectDisposedException"/> when the tensor was disposed.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void ThrowIfDisposed()
    {
        if (_disposed != 0)
        {
            ThrowDisposed();
        }
    }

    private static void ThrowDisposed() => throw new ObjectDisposedException(nameof(Tensor));

    // ---------------------------------------------------------------- autograd

    /// <summary>
    /// Computes the gradient of this scalar with respect to every tensor that requires gradients
    /// and accumulates it into their <see cref="Grad"/>. The graph is freed afterwards: a later pass that reaches one of
    /// its results throws (<see cref="Backward(bool)"/> keeps it).
    /// </summary>
    public void Backward() => Backward(retainGraph: false);

    /// <summary>
    /// Computes the gradient of this scalar as <see cref="Backward()"/> does; with <paramref name="retainGraph"/> the graph
    /// is kept, so another pass may run through the same results (two losses sharing an activation, each back-propagated
    /// in turn). Intermediate gradients are released either way; leaves' gradients accumulate.
    /// </summary>
    public void Backward(bool retainGraph)
    {
        if (Size != 1)
        {
            throw new InvalidOperationException($"Backward() without an argument needs a single-element tensor (such as a loss), but the shape is {FormatShape(_shape)}.");
        }

        using var seed = Full(_shape, 1f, Device);
        Backward(seed, retainGraph);
    }

    /// <summary>Back-propagates starting from the given gradient of this tensor (the graph is freed afterwards).</summary>
    public void Backward(Tensor gradient) => Backward(gradient, retainGraph: false);

    /// <summary>
    /// Back-propagates starting from the given gradient of this tensor; with <paramref name="retainGraph"/> the graph is kept
    /// for another pass.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// A result this pass reaches had its step freed by an earlier pass (run that one with <paramref name="retainGraph"/>):
    /// its gradient would otherwise be dropped silently.
    /// </exception>
    public void Backward(Tensor gradient, bool retainGraph)
    {
        ThrowIfDisposed();
        if (!RequiresGrad)
        {
            throw new InvalidOperationException("This tensor does not require gradients: none of its inputs had RequiresGrad = true (or it was computed under Autograd.NoGrad()).");
        }

        CheckSameShape(this, gradient);
        Backend.Axpy(gradient.Storage, GradStorage(), Size, 1f);

        var (reads, held) = (new List<Storage>(), new List<Storage>());
        Autograd.Backward(begins: true);
        try
        {
            BackwardNodes(reads, held, retainGraph);
        }
        finally
        {
            Autograd.Backward(begins: false);
            foreach (var storage in held)
            {
                Reevict(storage);
            }
        }
    }

    // Gives a restored activation's memory back again until another step needs it (unless it was released meanwhile).
    private static void Reevict(Storage storage)
    {
        if (storage.Alive)
        {
            storage.Backend.Evict(storage, storage.Recompute);
        }
    }

    private void BackwardNodes(List<Storage> reads, List<Storage> held, bool retainGraph)
    {
        var order = TopologicalOrder();
        using var staging = TensorOffloading.Hooks is { } hooks && Device.Backend.Offload is { OffloadedCount: > 0 }
            ? hooks.ForBackward(Device, order.ConvertAll(n => n._stage))                  // the layer each node was recorded under
            : null;
        // Someone listens for finished gradients (a data-parallel overlap): each parameter is told right after the last step
        // of this pass that reads it, so the last layers' are told first.
        var finishes = Autograd.ListensForGradients ? Finishes(order) : null;
        for (int index = 0; index < order.Count; index++)
        {
            if (finishes is not null && index > 0)
            {
                Finished(finishes[index - 1]);
            }

            var node = order[index];
            if (node._backward is null)
            {
                if (node._consumed && node.Grad is { } dropped)
                {
                    // A result whose step an earlier pass ran and freed: its gradient would stop here, never reaching its inputs.
                    dropped.Dispose();
                    node.Grad = null;
                    throw new InvalidOperationException($"Backward reached the result of '{node._operation ?? "an operation"}' (shape {FormatShape(node._shape)}) "
                        + "whose backward step an earlier Backward already ran and freed, so its gradient would be dropped. Back-propagate through it "
                        + "once, or run the earlier pass with Backward(retainGraph: true) to keep the graph for another pass.");
                }

                continue;
            }

            if (node.Grad is null)
            {
                continue;
            }

            staging?.Before(index);

            long start = Telemetry.Start(TelemetryLevel.Operations);

            // Recomputed (or unpacked) activations this step reads: given memory for it, and kept while the following steps
            // read them too (the projections of one input, one after another), then released again.
            Reads(node, reads);
            for (int i = held.Count - 1; i >= 0; i--)
            {
                if (!reads.Contains(held[i]))
                {
                    Reevict(held[i]);
                    held.RemoveAt(i);
                }
            }

            foreach (var storage in reads)
            {
                if (storage.Evicted && storage.Backend.Restore(storage))
                {
                    held.Add(storage);
                }
            }

            node.CheckVersions();
            node._backward(node.Grad);

            if (start != 0)
            {
                Telemetry.Operation(node._operation ?? "?", node, start, backward: true);
            }

            // Intermediate results do not keep their gradient, nor their graph unless asked to (PyTorch's retain_graph).
            node.Grad.Dispose();
            node.Grad = null;
            if (!retainGraph)
            {
                node._backward = null;
                node._parents = null;
                node._stage = null;
                node._consumed = true;
            }
        }

        if (finishes is not null && order.Count > 0)
        {
            Finished(finishes[order.Count - 1]);
        }
    }

    // The parameters (leaves that need gradients) whose last reader in `order` is each step, in the order they are read;
    // a leaf nothing reads (the root itself) at its own place.
    private static List<Tensor>?[] Finishes(List<Tensor> order)
    {
        var last = new Dictionary<Tensor, int>(ReferenceEqualityComparer.Instance);
        for (int i = 0; i < order.Count; i++)
        {
            foreach (var parent in order[i]._parents ?? [])
            {
                if (parent._backward is null && parent.RequiresGrad)
                {
                    last[parent] = i;
                }
            }
        }

        var finishes = new List<Tensor>?[order.Count + 1];
        for (int i = 0; i < order.Count; i++)
        {
            foreach (var parent in order[i]._parents ?? [])
            {
                if (last.TryGetValue(parent, out int at) && at == i && !(finishes[i]?.Contains(parent) ?? false))
                {
                    (finishes[i] ??= []).Add(parent);
                }
            }

            if (order[i]._backward is null && !last.ContainsKey(order[i]))
            {
                (finishes[i] ??= []).Add(order[i]);
            }
        }

        return finishes;
    }

    private static void Finished(List<Tensor>? leaves)
    {
        foreach (var leaf in leaves ?? [])
        {
            if (leaf.Grad is not null)
            {
                Autograd.GradientFinished(leaf);
            }
        }
    }

    // The layer (or fused layers) whose forward recorded this node while weights were offloaded (see Offloading), or null.
    private object? _stage;

    /// <summary>Nodes reachable from this tensor that require gradients, outputs before their inputs.</summary>
    private List<Tensor> TopologicalOrder()
    {
        var postOrder = new List<Tensor>();
        var visited = new HashSet<Tensor>(ReferenceEqualityComparer.Instance);
        var stack = new Stack<(Tensor Node, bool Expanded)>();
        stack.Push((this, false));
        while (stack.TryPop(out var item))
        {
            if (item.Expanded)
            {
                postOrder.Add(item.Node);
                continue;
            }

            if (!visited.Add(item.Node))
            {
                continue;
            }

            stack.Push((item.Node, true));
            foreach (var parent in item.Node._parents ?? [])
            {
                if (parent.RequiresGrad && !visited.Contains(parent))
                {
                    stack.Push((parent, false));
                }
            }
        }

        postOrder.Reverse();
        return postOrder;
    }

    /// <summary>
    /// Adds <paramref name="gradient"/> (same size) to this tensor's gradient. With <paramref name="adopt"/>, a tensor that
    /// has no gradient yet takes <paramref name="gradient"/>'s buffer itself instead of a zeroed copy it is added to (no
    /// memory traffic). Only for the gradient of an intermediate result, which nothing writes once its backward step has
    /// run, and for one receiver per gradient (a second would share the buffer).
    /// </summary>
    public void AddGradient(Tensor gradient, bool adopt)
    {
        if (adopt && Grad is null && gradient.Size == Size)
        {
            gradient.Storage.AddRef();
            Grad = new Tensor(_shape, gradient.Storage, Device, track: false);
            return;
        }

        if (Grad is null && gradient.Size == Size)
        {
            // The first gradient: copied (one pass), not added to a zero-filled buffer (a fill and a read-add-write pass).
            Grad = Empty(_shape, Device, track: false);
            Backend.Copy(gradient.Storage, Grad.Storage, Size);
            return;
        }

        Backend.Axpy(gradient.Storage, GradStorage(), Size, 1f);
    }

    /// <summary>
    /// The gradient buffer for a kernel that can either overwrite or add to it: the first gradient is written into a
    /// buffer that is not zeroed (<paramref name="beta"/> 0: every element must be written), later ones are added
    /// (<paramref name="beta"/> 1). Saves a zero fill and a read of the buffer for the gradient that is the first.
    /// </summary>
    public Storage GradientTarget(out float beta)
    {
        beta = Grad is null ? 0f : 1f;
        Grad ??= Empty(_shape, Device, zeroed: false, track: false);
        return Grad.Storage;
    }

    /// <summary>The gradient buffer, allocated (zeroed and outside any scope) on first use.</summary>
    public Storage GradStorage()
    {
        Grad ??= Empty(_shape, Device, zeroed: true, track: false);
        return Grad.Storage;
    }

    /// <summary>Records how to back-propagate into this freshly computed tensor. Callers check <see cref="WillRecord(Tensor)"/> first.</summary>
    public void Record(string operation, Action<Tensor> backward, params Tensor[] inputs)
    {
        RequiresGrad = true; // must precede _backward: the setter only accepts leaves
        _operation = operation;
        _stage = TensorOffloading.CurrentLayer;
        _parents = inputs;
        _backward = backward;
        _versions = VersionSum(this, inputs);
    }

    // The in-place writes into the storages a node's backward step may read, summed (each count only grows, so any write
    // after the operation was recorded changes the sum).
    private long _versions;

    private static long VersionSum(Tensor node, Tensor[] inputs)
    {
        long sum = node.VersionOf;
        foreach (var input in inputs)
        {
            sum += input.VersionOf;
        }

        return sum;
    }

    // Refuses to back-propagate through values changed in place after their operation was recorded (the gradient would
    // be computed from the new values).
    private void CheckVersions()
    {
        if (_parents is { } parents && VersionSum(this, parents) != _versions)
        {
            throw new InvalidOperationException(
                $"A tensor that the backward step of '{_operation}' reads was changed in place (CopyFrom, Fill, Scale or AddScaled) after "
                + "the operation was recorded, so its gradient cannot be computed. Write into tensors only after Backward(), or into a copy (Clone()).");
        }
    }

    /// <summary>Publishes an operation event when <paramref name="start"/> is non-zero (operations telemetry on).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor Traced(string operation, Tensor output, long start)
    {
        if (start != 0)
        {
            Telemetry.Operation(operation, output, start, backward: false);
        }

        return output;
    }

    /// <summary>True when an operation on these inputs will be recorded, so callers can skip building closures.</summary>
    public static bool WillRecord(Tensor a) => a.RequiresGrad && Autograd.IsEnabled;

    /// <summary>True when an operation on these inputs will be recorded, so callers can skip building closures.</summary>
    public static bool WillRecord(Tensor a, Tensor b) => (a.RequiresGrad || b.RequiresGrad) && Autograd.IsEnabled;}

internal static class MemoryMarshalHelpers
{
    public static float[] Flatten(float[,] values)
    {
        var flat = new float[values.Length];
        System.Runtime.InteropServices.MemoryMarshal.CreateReadOnlySpan(
            ref Unsafe.As<byte, float>(ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference(values)), values.Length).CopyTo(flat);
        return flat;
    }

    public static void Unflatten(float[] flat, float[,] destination) =>
        flat.AsSpan().CopyTo(System.Runtime.InteropServices.MemoryMarshal.CreateSpan(
            ref Unsafe.As<byte, float>(ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference(destination)), destination.Length));
}
