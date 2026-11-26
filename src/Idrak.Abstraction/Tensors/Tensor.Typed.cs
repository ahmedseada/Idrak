// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.InteropServices;
using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction;

// Typed tensors: a tensor knows the type of its values (its storage's), and a tensor of any type holds its values as the
// type stores them (a checkpoint's weights as stored, never widened), read and written through the type alone.
public sealed partial class Tensor
{
    /// <summary>
    /// The type of the tensor's values (its storage's, <see cref="Storage.Type"/>): float32 (<see cref="Types.F32"/>),
    /// the compute type of activations, for every tensor an operation makes; a stored weight's own type for a tensor made
    /// by <see cref="Stored(IElementType, ReadOnlySpan{byte}, ReadOnlySpan{int}, Device?)"/>.
    /// </summary>
    public IElementType Type => _storage.Type;

    /// <summary>
    /// A tensor of <paramref name="shape"/> on <paramref name="device"/> holding <paramref name="stored"/>, values of
    /// <paramref name="type"/> as the type stores them (its <see cref="IElementType.ByteCount"/> bytes for the shape's
    /// values: a checkpoint's bytes as they are, never widened). No <see cref="TensorScope"/> captures it (a weight, like
    /// <see cref="Persistent"/>).
    /// </summary>
    /// <param name="type">The values' type.</param>
    /// <param name="stored">The stored bytes.</param>
    /// <param name="shape">The tensor's shape (its values, not its bytes).</param>
    /// <param name="device">Where it lives (the default device when null).</param>
    /// <exception cref="ArgumentException">The bytes are fewer than the shape's values take.</exception>
    public static Tensor Stored(IElementType type, ReadOnlySpan<byte> stored, ReadOnlySpan<int> shape, Device? device = null)
    {
        ArgumentNullException.ThrowIfNull(type);
        device ??= Device.Default;
        int count = ElementCount(shape);
        long bytes = type.ByteCount(count);
        if (stored.Length < bytes)
        {
            throw new ArgumentException($"{count} values of {type.Name} take {bytes} bytes; {stored.Length} are given.", nameof(stored));
        }

        var storage = device.Backend.Allocate(type, count, zeroed: false);
        try
        {
            float[] words = ArrayPool<float>.Shared.Rent(Math.Max(1, storage.Length));
            try
            {
                var into = MemoryMarshal.AsBytes(words.AsSpan(0, storage.Length));
                stored[..(int)bytes].CopyTo(into);
                into[(int)bytes..].Clear();
                device.Backend.Upload(words.AsSpan(0, storage.Length), storage);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(words);
            }

            return new Tensor(shape.ToArray(), storage, device, track: false);
        }
        catch
        {
            storage.Release();
            throw;
        }
    }

    /// <summary>
    /// A tensor of <paramref name="shape"/> holding <paramref name="values"/> written as <paramref name="type"/> (by the
    /// type's writer, <see cref="IElementType.Write(ReadOnlySpan{float}, Span{byte})"/>): a weight stored as any type.
    /// </summary>
    /// <param name="type">The type the values are written as.</param>
    /// <param name="values">The values.</param>
    /// <param name="shape">The tensor's shape.</param>
    /// <param name="device">Where it lives (the default device when null).</param>
    /// <exception cref="ArgumentException">The values do not fill the shape.</exception>
    public static Tensor Stored(IElementType type, ReadOnlySpan<float> values, ReadOnlySpan<int> shape, Device? device = null)
    {
        ArgumentNullException.ThrowIfNull(type);
        if (values.Length != ElementCount(shape))
        {
            throw new ArgumentException($"{values.Length} values cannot fill a tensor of shape {FormatShape(shape)}.", nameof(values));
        }

        byte[] stored = new byte[type.ByteCount(values.Length)];
        type.Write(values, stored);
        return Stored(type, stored, shape, device);
    }

    /// <summary>The tensor's values as its type stores them (<see cref="IElementType.ByteCount"/> bytes for its values).</summary>
    public byte[] ToBytes()
    {
        ThrowIfDisposed();
        var storage = ReadStorage(out var scratch);
        try
        {
            long bytes = Type.ByteCount(Size);
            float[] words = ArrayPool<float>.Shared.Rent(Math.Max(1, storage.Length));
            try
            {
                Backend.Download(storage, words.AsSpan(0, storage.Length));
                return MemoryMarshal.AsBytes(words.AsSpan(0, storage.Length))[..(int)bytes].ToArray();
            }
            finally
            {
                ArrayPool<float>.Shared.Return(words);
            }
        }
        finally
        {
            scratch?.Release();
        }
    }

    // A typed storage's values as float32, read through its type (ToArray of a tensor whose storage holds another type).
    private static void ReadTyped(Backend backend, Storage storage, Span<float> values)
    {
        float[] words = ArrayPool<float>.Shared.Rent(Math.Max(1, storage.Length));
        try
        {
            backend.Download(storage, words.AsSpan(0, storage.Length));
            storage.Type.Read(MemoryMarshal.AsBytes(words.AsSpan(0, storage.Length)), values);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(words);
        }
    }
}
