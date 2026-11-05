// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.InteropServices;

namespace Idrak.Abstraction;

// Typed tensors (plan 37, T0 and T1): index tensors in the device's index type, any registered type's tensor from its
// stored bytes, and their elements read back as integers, bytes or floats.
public sealed partial class Tensor
{
    /// <summary>
    /// A tensor of integer indices [count] on <paramref name="device"/>, as the operations that take indices read them
    /// (<see cref="EmbeddingLookup(Tensor)"/>, <see cref="OneHot"/>, ...): in the device's index type
    /// (<see cref="Devices.BackendCapabilities.IndexType"/>: int32 on the CPU, every index exact). On a device whose index
    /// kernels still read float32 they are floats while every one is exact there (up to 2²⁴), else int32, which such a
    /// device refuses with a message naming it.
    /// </summary>
    public static Tensor Indices(ReadOnlySpan<int> indices, Device device)
    {
        ArgumentNullException.ThrowIfNull(device);
        var type = device.Backend.Capabilities.IndexType;
        if (ReferenceEquals(type, ElementType.Float32))
        {
            var values = new float[indices.Length];
            bool exact = true;
            for (int i = 0; i < values.Length; i++)
            {
                int v = indices[i];
                exact &= v is >= -(1 << 24) and <= 1 << 24;
                values[i] = v;
            }

            if (exact)
            {
                return From(values, [values.Length], device);
            }

            type = ElementType.Int32;
        }

        return FromBytes(MemoryMarshal.AsBytes(indices), ElementType.Int32, [indices.Length], device, type);
    }

    /// <summary>A tensor of int64 indices [count] on <paramref name="device"/>: the operations that take indices read them through a counted conversion to the device's index type.</summary>
    public static Tensor Indices(ReadOnlySpan<long> indices, Device device)
    {
        ArgumentNullException.ThrowIfNull(device);
        return FromBytes(MemoryMarshal.AsBytes(indices), ElementType.Int64, [indices.Length], device);
    }

    /// <summary>
    /// A tensor of <paramref name="shape"/> holding <paramref name="bytes"/>: its elements stored in
    /// <paramref name="type"/> (<see cref="ElementType.ByteCount"/> bytes), as they are.
    /// </summary>
    /// <exception cref="ArgumentException">The bytes are not as many as the shape's elements of the type take.</exception>
    public static Tensor FromBytes(ReadOnlySpan<byte> bytes, ElementType type, ReadOnlySpan<int> shape, Device? device = null) =>
        FromBytes(bytes, type, shape, device ?? Device.Default, type);

    // A tensor of `stored` holding `bytes` of `type`'s elements, converted on the host when the types differ.
    private static Tensor FromBytes(ReadOnlySpan<byte> bytes, ElementType type, ReadOnlySpan<int> shape, Device device, ElementType stored)
    {
        ArgumentNullException.ThrowIfNull(type);
        var t = Empty(shape, device, stored);
        long expected = type.ByteCount(t.Size);
        if (bytes.Length != expected)
        {
            t.Dispose();
            throw new ArgumentException($"{bytes.Length} bytes cannot fill a {type.Name} tensor of shape {FormatShape(shape)} ({t.Size} elements, {expected} bytes).", nameof(bytes));
        }

        var storage = t.Storage;
        float[] words = ArrayPool<float>.Shared.Rent(Math.Max(1, storage.Length));
        try
        {
            var target = MemoryMarshal.AsBytes(words.AsSpan(0, storage.Length));
            target.Clear();
            if (ReferenceEquals(type, stored))
            {
                bytes.CopyTo(target);
            }
            else
            {
                ElementConversions.Convert(type, bytes, stored, target, t.Size);
            }

            t.Backend.Upload(words.AsSpan(0, storage.Length), storage);
        }
        catch
        {
            t.Dispose();
            throw;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(words);
        }

        return t;
    }

    /// <summary>
    /// A new tensor of the same shape on the same device holding these elements in <paramref name="type"/> (plan 37, T2):
    /// bfloat16 or float16 storage of a float32 tensor at half its memory, int16 or uint16 samples and token ids, float64
    /// values for a check, and back. Converted on the device by its own conversion (<c>Backend.ConvertElements</c>),
    /// rounded to nearest (ties to even) and saturated as the type's own conversion does; integers between float32, int32
    /// and int64 exactly or refused. The same type gives a copy. Not recorded for gradients.
    /// </summary>
    public Tensor ConvertTo(ElementType type)
    {
        ArgumentNullException.ThrowIfNull(type);
        ThrowIfDisposed();
        if (ReferenceEquals(type, ElementType))
        {
            return Clone();
        }

        var result = ReferenceEquals(type, ElementType.Float32) ? Empty(_shape, Device) : Empty(_shape, Device, type);
        var storage = ReadStorage(out var scratch);
        try
        {
            Backend.ConvertElements(storage, result.Storage);
        }
        catch
        {
            result.Dispose();
            throw;
        }
        finally
        {
            scratch?.Release();
        }

        return result;
    }

    /// <summary>The elements' stored bytes (<see cref="ElementType"/>.ByteCount(<see cref="Size"/>) of them), row-major.</summary>
    public byte[] ToBytes()
    {
        ThrowIfDisposed();
        if (IsFloat32)
        {
            return MemoryMarshal.AsBytes(ToArray().AsSpan()).ToArray();
        }

        var words = Words();
        return MemoryMarshal.AsBytes(words.AsSpan()).Slice(0, (int)ElementType.ByteCount(Size)).ToArray();
    }

    /// <summary>
    /// The elements as int32, exactly: an int32 tensor's as they are, any other's converted (a float that is not a whole
    /// number, or a value past int32's range, is refused).
    /// </summary>
    /// <exception cref="ArgumentException">An element int32 does not hold exactly.</exception>
    public int[] ToInt32Array() => ToIntegers<int>(ElementType.Int32);

    /// <summary>The elements as int64, exactly (a float that is not a whole number is refused).</summary>
    /// <exception cref="ArgumentException">An element int64 does not hold exactly.</exception>
    public long[] ToInt64Array() => ToIntegers<long>(ElementType.Int64);

    private T[] ToIntegers<T>(ElementType type)
        where T : unmanaged
    {
        ThrowIfDisposed();
        var result = new T[Size];
        var source = MemoryMarshal.AsBytes((IsFloat32 ? ToArray() : Words()).AsSpan());
        ElementConversions.Convert(ElementType, source, type, MemoryMarshal.AsBytes(result.AsSpan()), Size);
        return result;
    }

    // The storage's words, read from the device (a typed tensor is never a view).
    private float[] Words()
    {
        var storage = Storage;
        var words = new float[storage.Length];
        Backend.Download(storage, words);
        return words;
    }

    // Elements [offset, offset + count) of a tensor of another type than float32, widened by its type's conversion.
    private float[] Widened(int offset, int count)
    {
        var all = new float[Size];
        ElementType.Decode(MemoryMarshal.AsBytes(Words().AsSpan()), all);
        return offset == 0 && count == all.Length ? all : all.AsSpan(offset, count).ToArray();
    }

    /// <summary>
    /// Overwrites this index tensor's elements in place with <paramref name="indices"/>, stored in its element type (a
    /// sampler's history, a recordable pass's fixed row buffer; plan 37, T1 and T1b), as <see cref="Load"/> does for floats.
    /// </summary>
    /// <exception cref="ArgumentException">Not one index per element, or an index its type cannot hold exactly.</exception>
    public void LoadIndices(ReadOnlySpan<int> indices)
    {
        ThrowIfDisposed();
        if (indices.Length != Size)
        {
            throw new ArgumentException($"{indices.Length} indices cannot fill a tensor of {Size} elements.", nameof(indices));
        }

        var storage = Storage;
        float[] words = ArrayPool<float>.Shared.Rent(Math.Max(1, storage.Length));
        try
        {
            var target = MemoryMarshal.AsBytes(words.AsSpan(0, storage.Length));
            target.Clear();
            ElementConversions.Convert(ElementType.Int32, MemoryMarshal.AsBytes(indices), ElementType, target, Size);
            Backend.Upload(words.AsSpan(0, storage.Length), storage);
            storage.Written();
        }
        finally
        {
            ArrayPool<float>.Shared.Return(words);
        }
    }

    /// <summary>
    /// A tensor of <paramref name="shape"/> holding <paramref name="values"/> in <paramref name="type"/>, or when null in
    /// the type the run keeps its values in on <paramref name="device"/> (<see cref="ElementTypes.Choose"/>: a run's
    /// <c>--dtype</c> passed by the app, <see cref="ElementTypes.EnvironmentVariable"/>, else auto, what the device computes
    /// natively): each value stored by the type's own conversion (rounded to nearest, ties to even). Owned by the current
    /// <see cref="TensorScope"/> when <paramref name="track"/>; long-lived state (a layer's parameters) passes false.
    /// </summary>
    /// <exception cref="ArgumentException">The values do not fill the shape.</exception>
    public static Tensor FromValues(ReadOnlySpan<float> values, ReadOnlySpan<int> shape, ElementType? type = null, Device? device = null, bool requiresGrad = false,
        bool track = true)
    {
        device ??= Device.Default;
        type ??= ElementTypes.Choose(device: device).Type;
        var t = ReferenceEquals(type, ElementType.Float32) ? From(values, shape, device) : FromBytes(MemoryMarshal.AsBytes(values), ElementType.Float32, shape, device, type);
        if (!track)
        {
            TensorScope.Untrack(t);
        }

        t.RequiresGrad = requiresGrad;
        return t;
    }

    /// <summary>
    /// A tensor of ids or indices of <paramref name="shape"/> (token ids, embedding and gather indices, expert indices,
    /// positions) in <paramref name="type"/>, or when null in the smallest type that holds them
    /// (<see cref="ElementTypes.ChooseIds"/>: <see cref="ElementTypes.IdEnvironmentVariable"/> when set, else auto from the
    /// data's range; an app passes its <c>--id-dtype</c> through <see cref="ElementTypes.ChooseIds"/>'s choice). An
    /// operation whose kernel reads another index type gets them through the dispatcher's counted conversion.
    /// </summary>
    /// <exception cref="ArgumentException">The ids do not fill the shape, or an id the type does not hold.</exception>
    public static Tensor Ids(ReadOnlySpan<int> ids, ReadOnlySpan<int> shape, ElementType? type = null, Device? device = null)
    {
        device ??= Device.Default;
        if (type is null)
        {
            var (smallest, largest) = Range(ids);
            type = ElementTypes.ChooseIds(largest, smallest, device: device).Type;
        }

        return FromBytes(MemoryMarshal.AsBytes(ids), ElementType.Int32, shape, device, type);
    }

    /// <summary><see cref="Ids(ReadOnlySpan{int}, ReadOnlySpan{int}, ElementType?, Device?)"/> of int64 ids.</summary>
    /// <exception cref="ArgumentException">The ids do not fill the shape, or an id the type does not hold.</exception>
    public static Tensor Ids(ReadOnlySpan<long> ids, ReadOnlySpan<int> shape, ElementType? type = null, Device? device = null)
    {
        device ??= Device.Default;
        if (type is null)
        {
            var (smallest, largest) = Range(ids);
            type = ElementTypes.ChooseIds(largest, smallest, device: device).Type;
        }

        return FromBytes(MemoryMarshal.AsBytes(ids), ElementType.Int64, shape, device, type);
    }

    // The smallest and largest of `values` in one pass ((0, 0) when empty).
    private static (long Smallest, long Largest) Range<T>(ReadOnlySpan<T> values)
        where T : unmanaged, System.Numerics.IBinaryInteger<T>
    {
        if (values.IsEmpty)
        {
            return (0, 0);
        }

        T smallest = values[0], largest = values[0];
        foreach (var v in values)
        {
            smallest = T.Min(smallest, v);
            largest = T.Max(largest, v);
        }

        return (long.CreateTruncating(smallest), long.CreateTruncating(largest));
    }

    /// <summary>
    /// The elements as <typeparamref name="T"/>, checked (plan 37, decision 6): <typeparamref name="T"/> must be this
    /// tensor's type's C# type (<see cref="ElementTypes.Of{T}"/>: <see cref="float"/> for float32, <see cref="ushort"/>
    /// for uint16, <see cref="Half"/> for float16, ...), else the read is refused naming both. Read where they lie when the
    /// storage is in the host's memory (the CPU: no copy; valid while the tensor lives and nothing writes it), else copied
    /// from the device once. A type with no C# type (bfloat16, fp8, int4, the block formats) is read with
    /// <see cref="ToBytes"/>, or converted first (<see cref="ConvertTo"/>).
    /// </summary>
    /// <exception cref="InvalidOperationException"><typeparamref name="T"/> is not this tensor's type's C# type.</exception>
    public ReadOnlySpan<T> Span<T>()
        where T : unmanaged
    {
        ThrowIfDisposed();
        var type = ElementTypes.Of<T>();
        if (!ReferenceEquals(type, ElementType))
        {
            throw new InvalidOperationException(type is null
                ? $"A {ElementType.Name} tensor read as {typeof(T).Name}, which stores no registered type: read its bytes (ToBytes) or convert it (ConvertTo)."
                : $"A {ElementType.Name} tensor read as {typeof(T).Name} ({type.Name}): read it as its own type's C# type where it has one, its bytes (ToBytes), or convert it first (ConvertTo({type.Name})).");
        }

        var storage = Storage;
        if (storage is Devices.IHostStorage host && !storage.Evicted)
        {
            return MemoryMarshal.Cast<float, T>(host.HostMemory[..storage.Length])[..Size];
        }

        return MemoryMarshal.Cast<byte, T>(ToBytes());
    }
}
