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

    private Tensor(int[] shape, Storage storage, Device device, bool track)
    {
        _shape = shape;
        Storage = storage;
        Device = device;
        Size = storage.Length;
        if (track)
        {
            TensorScope.Track(this);
        }
    }

    /// <summary>Returns device memory to the pool if the tensor was never disposed.</summary>
    ~Tensor() => Storage.Release();

    /// <summary>The device this tensor's data lives on.</summary>
    public Device Device { get; }

    /// <summary>The length of each dimension, outermost first.</summary>
    public ReadOnlySpan<int> Shape => _shape;

    /// <summary>The number of dimensions (0 for a scalar).</summary>
    public int Rank => _shape.Length;

    /// <summary>The total number of elements.</summary>
    public int Size { get; }

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

            if (field != value && Storage is { } storage && storage.Backend.Offload is not null)
            {
                TensorOffloading.TrainableChanged?.Invoke(this, value);                 // frozen weights move to system memory first
            }

            field = value;
        }
    }

    /// <summary>The accumulated gradient after <see cref="Backward()"/>, or null if none has been computed.</summary>
    public Tensor? Grad { get; private set; }

    /// <summary>True when this tensor was created directly rather than as the output of a recorded operation.</summary>
    public bool IsLeaf => _backward is null;

    internal Storage Storage { get; }

    internal Backend Backend => Device.Backend;

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
            double r = Math.Sqrt(-2.0 * Math.Log(u1));
            values[i] = (float)(mean + std * r * Math.Cos(2 * Math.PI * u2));
            if (i + 1 < values.Length)
            {
                values[i + 1] = (float)(mean + std * r * Math.Sin(2 * Math.PI * u2));
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

    /// <summary>Overwrites this tensor's data in place with <paramref name="values"/> (same element count).</summary>
    internal void Load(ReadOnlySpan<float> values)
    {
        ThrowIfDisposed();
        Backend.Upload(values, Storage);
    }

    internal static Tensor Empty(ReadOnlySpan<int> shape, Device device, bool zeroed = false, bool track = true)
    {
        int size = ElementCount(shape);
        return new Tensor(shape.ToArray(), device.Backend.Allocate(size, zeroed), device, track);
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
        var result = GC.AllocateUninitializedArray<float>(Size);
        Backend.Download(Storage, result);
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
        Span<float> value = stackalloc float[1];
        Backend.Download(Storage, value);
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

        return From(ToArray(), _shape, device, RequiresGrad && IsLeaf);
    }

    /// <summary>Returns a copy of the data that is not connected to the autograd graph.</summary>
    public Tensor Clone()
    {
        ThrowIfDisposed();
        var t = Empty(_shape, Device);
        Backend.Copy(Storage, t.Storage, Size);
        return t;
    }

    /// <summary>Whether <see cref="Dispose"/> has released this tensor.</summary>
    internal bool IsDisposed => Volatile.Read(ref _disposed) != 0;

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
    internal void ReleaseGrad()
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
        Storage.Release();
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

    internal static string FormatShape(ReadOnlySpan<int> shape) => "[" + string.Join(", ", shape.ToArray()) + "]";

    /// <summary>
    /// Releases the memory of this tensor's values while it stays in the autograd graph (its gradient still flows through
    /// it): for a training result whose values no backward step reads, once its last forward use is done, so it is not
    /// kept until the end of the step. The memory is shared with every view of the values, which lose them too: call it
    /// only where every tensor sharing the values is done with them and no consumer's backward reads them. Kernels given
    /// the values afterwards fail.
    /// </summary>
    internal void DropValue()
    {
        if (_disposed == 0)
        {
            Backend.Evict(Storage, null);
        }
    }

    /// <summary>
    /// Releases the memory of this tensor's values like <see cref="DropValue"/>, but <paramref name="recompute"/> fills
    /// them in again when a backward step needs them (see <see cref="Backward(Tensor)"/>): activations cheaper to recompute
    /// than to keep. The recomputation must only read tensors that are still kept.
    /// </summary>
    internal void Evict(Action<Tensor> recompute)
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

        if (!ReferenceEquals(node.Storage.RecomputedFor, node) && !OwnValuesUnread.Contains(node._operation ?? ""))
        {
            reads.Add(node.Storage);
        }

        bool packedInputs = ReadsPackedInputs.Contains(node._operation ?? "");
        foreach (var parent in node._parents ?? [])
        {
            if (!(packedInputs && parent.Storage.Evicted && parent.Storage.Packed is not null))
            {
                reads.Add(parent.Storage);
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
    internal void EvictToPacked(Storage packed)
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal void ThrowIfDisposed()
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
    /// and accumulates it into their <see cref="Grad"/>. The graph is freed afterwards.
    /// </summary>
    public void Backward()
    {
        if (Size != 1)
        {
            throw new InvalidOperationException($"Backward() without an argument needs a single-element tensor (such as a loss), but the shape is {FormatShape(_shape)}.");
        }

        using var seed = Full(_shape, 1f, Device);
        Backward(seed);
    }

    /// <summary>Back-propagates starting from the given gradient of this tensor.</summary>
    public void Backward(Tensor gradient)
    {
        ThrowIfDisposed();
        if (!RequiresGrad)
        {
            throw new InvalidOperationException("This tensor does not require gradients: none of its inputs had RequiresGrad = true (or it was computed under Autograd.NoGrad()).");
        }

        CheckSameShape(this, gradient);
        Backend.Axpy(gradient.Storage, GradStorage(), Size, 1f);

        var (reads, held) = (new List<Storage>(), new List<Storage>());
        try
        {
            BackwardNodes(reads, held);
        }
        finally
        {
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

    private void BackwardNodes(List<Storage> reads, List<Storage> held)
    {
        var order = TopologicalOrder();
        using var staging = TensorOffloading.ForBackward?.Invoke(Device, order);     // null unless weights are offloaded
        for (int index = 0; index < order.Count; index++)
        {
            var node = order[index];
            if (node._backward is null || node.Grad is null)
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

            // Intermediate results do not keep their gradient or graph (like PyTorch without retain_graph).
            node.Grad.Dispose();
            node.Grad = null;
            node._backward = null;
            node._parents = null;
            node._stage = null;
        }
    }

    // The layer (or fused layers) whose forward recorded this node while weights were offloaded (see Offloading), or null.
    private object? _stage;

    internal object? StageGroup => _stage;

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
    internal void AddGradient(Tensor gradient, bool adopt)
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
    internal Storage GradientTarget(out float beta)
    {
        beta = Grad is null ? 0f : 1f;
        Grad ??= Empty(_shape, Device, zeroed: false, track: false);
        return Grad.Storage;
    }

    /// <summary>The gradient buffer, allocated (zeroed and outside any scope) on first use.</summary>
    internal Storage GradStorage()
    {
        Grad ??= Empty(_shape, Device, zeroed: true, track: false);
        return Grad.Storage;
    }

    /// <summary>Records how to back-propagate into this freshly computed tensor. Callers check <see cref="WillRecord(Tensor)"/> first.</summary>
    internal void Record(string operation, Action<Tensor> backward, params Tensor[] inputs)
    {
        RequiresGrad = true; // must precede _backward: the setter only accepts leaves
        _operation = operation;
        _stage = TensorOffloading.Current;
        _parents = inputs;
        _backward = backward;
        _versions = VersionSum(this, inputs);
    }

    // The in-place writes into the storages a node's backward step may read, summed (each count only grows, so any write
    // after the operation was recorded changes the sum).
    private long _versions;

    private static long VersionSum(Tensor node, Tensor[] inputs)
    {
        long sum = node.Storage.Version;
        foreach (var input in inputs)
        {
            sum += input.Storage.Version;
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
    internal static Tensor Traced(string operation, Tensor output, long start)
    {
        if (start != 0)
        {
            Telemetry.Operation(operation, output, start, backward: false);
        }

        return output;
    }

    /// <summary>True when an operation on these inputs will be recorded, so callers can skip building closures.</summary>
    internal static bool WillRecord(Tensor a) => a.RequiresGrad && Autograd.IsEnabled;

    internal static bool WillRecord(Tensor a, Tensor b) => (a.RequiresGrad || b.RequiresGrad) && Autograd.IsEnabled;}

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
