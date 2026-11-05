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

    // Overwrites an index tensor's elements with `indices`, stored in its type (a sampler's history).
    internal void LoadIndices(ReadOnlySpan<int> indices)
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
}
