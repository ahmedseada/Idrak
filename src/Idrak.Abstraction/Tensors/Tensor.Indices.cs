// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction;

// Index tensors: integer indices held as float values, as the operations that take indices read them.
public sealed partial class Tensor
{
    /// <summary>
    /// A tensor of integer indices [count] on <paramref name="device"/>, as the operations that take indices read them
    /// (<see cref="EmbeddingLookup(Tensor)"/>, <see cref="OneHot"/>, ...): each index a float value, exact up to 2²⁴.
    /// </summary>
    /// <exception cref="ArgumentException">An index a float value does not hold exactly (past 2²⁴).</exception>
    public static Tensor Indices(ReadOnlySpan<int> indices, Device device)
    {
        ArgumentNullException.ThrowIfNull(device);
        return From(Floats(indices), [indices.Length], device);
    }

    /// <summary>
    /// Overwrites this index tensor's elements in place with <paramref name="indices"/> (a sampler's history, a recordable
    /// pass's fixed row buffer), each a float value, as <see cref="Load"/> does for floats.
    /// </summary>
    /// <exception cref="ArgumentException">Not one index per element, or an index a float value does not hold exactly.</exception>
    public void LoadIndices(ReadOnlySpan<int> indices)
    {
        ThrowIfDisposed();
        if (indices.Length != Size)
        {
            throw new ArgumentException($"{indices.Length} indices cannot fill a tensor of {Size} elements.", nameof(indices));
        }

        Load(Floats(indices));
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
