// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction;

public sealed partial class Tensor
{
    /// <summary>
    /// A tensor of integer indices [count] on <paramref name="device"/>, as the gathers (<see cref="EmbeddingLookup(Tensor)"/>) take
    /// them: each value a float holding the index exactly (indices up to 2²⁴).
    /// </summary>
    public static Tensor Indices(ReadOnlySpan<int> indices, Device device)
    {
        ArgumentNullException.ThrowIfNull(device);
        var values = new float[indices.Length];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = indices[i];
        }

        return From(values, [values.Length], device);
    }
}
