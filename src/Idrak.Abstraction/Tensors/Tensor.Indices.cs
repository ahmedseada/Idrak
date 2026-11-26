// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.InteropServices;

namespace Idrak.Abstraction;

// Index tensors: integer indices of the exact int32 type (Types.I32), written through the type, as the operations that
// take indices read them (each storage's indices as its type holds them).
public sealed partial class Tensor
{
    /// <summary>
    /// A tensor of integer indices [count] on <paramref name="device"/>, as the operations that take indices read them
    /// (<see cref="EmbeddingLookup(Tensor)"/>, <see cref="OneHot"/>, ...): values of the int32 type
    /// (<see cref="Types.I32"/>), every index exact.
    /// </summary>
    public static Tensor Indices(ReadOnlySpan<int> indices, Device device)
    {
        ArgumentNullException.ThrowIfNull(device);
        var type = Types.I32.Type;
        var storage = device.Backend.Allocate(type, indices.Length, zeroed: false);
        try
        {
            Write(device.Backend, type, indices, storage);
            return new Tensor([indices.Length], storage, device, track: true);
        }
        catch
        {
            storage.Release();
            throw;
        }
    }

    /// <summary>
    /// Overwrites this index tensor's elements in place with <paramref name="indices"/> (a sampler's history, a recordable
    /// pass's fixed row buffer), as its type holds values: an int32 tensor's written through its type (every index
    /// exact), a float32 tensor's as float values (exact up to 2²⁴), as <see cref="Load"/> does for floats.
    /// </summary>
    /// <exception cref="ArgumentException">Not one index per element, or an index a float value does not hold exactly.</exception>
    public void LoadIndices(ReadOnlySpan<int> indices)
    {
        ThrowIfDisposed();
        if (indices.Length != Size)
        {
            throw new ArgumentException($"{indices.Length} indices cannot fill a tensor of {Size} elements.", nameof(indices));
        }

        if (!Storage.Typed)
        {
            Load(Floats(indices));
            return;
        }

        Write(Backend, Storage.Type, indices, Storage);
        Storage.Written();
    }

    // The indices written as `type` stores values (its own encoder, from exact float64 values, a group of whole units at a
    // time) into the storage's words, uploaded once.
    private static void Write(Devices.Backend backend, IElementType type, ReadOnlySpan<int> indices, Devices.Storage storage)
    {
        float[] words = ArrayPool<float>.Shared.Rent(Math.Max(1, storage.Length));
        try
        {
            var bytes = MemoryMarshal.AsBytes(words.AsSpan(0, storage.Length));
            bytes.Clear();
            int unit = Math.Max(1, type.Description.UnitValues), chunk = Math.Max(unit, 1024 / unit * unit);
            Span<double> values = stackalloc double[chunk];
            for (int start = 0; start < indices.Length; start += chunk)
            {
                int count = Math.Min(chunk, indices.Length - start);
                for (int i = 0; i < count; i++)
                {
                    values[i] = indices[start + i];
                }

                long at = type.ByteCount(start);
                type.Write(values[..count], bytes.Slice((int)at, (int)type.ByteCount(count)));
            }

            backend.Upload(words.AsSpan(0, storage.Length), storage);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(words);
        }
    }

    // The indices as float values, each exact (up to 2²⁴).
    private static float[] Floats(ReadOnlySpan<int> indices)
    {
        var values = new float[indices.Length];
        for (int i = 0; i < values.Length; i++)
        {
            int v = indices[i];
            values[i] = v is >= -(1 << 24) and <= 1 << 24 ? v : throw new ArgumentException($"Index {v} (element {i}) is past what a float value holds exactly (2^24).", nameof(indices));
        }

        return values;
    }
}
