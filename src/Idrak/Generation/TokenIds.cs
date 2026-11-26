// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;

namespace Idrak.Generation;

/// <summary>
/// Token ids as a model's input: the generators' prompt, window and step tensors and the training batches' inputs, as
/// float32 values (exact up to 2^24).
/// </summary>
internal static class TokenIds
{
    /// <summary><paramref name="ids"/> (whole numbers) as a tensor of <paramref name="shape"/> on <paramref name="device"/>.</summary>
    public static Tensor Of(ReadOnlySpan<float> ids, ReadOnlySpan<int> shape, Device device)
    {
        ArgumentNullException.ThrowIfNull(device);
        return Tensor.From(ids, shape, device);
    }

    /// <summary><paramref name="ids"/> as a tensor of <paramref name="shape"/> on <paramref name="device"/>.</summary>
    public static Tensor Of(ReadOnlySpan<int> ids, ReadOnlySpan<int> shape, Device device)
    {
        ArgumentNullException.ThrowIfNull(device);
        return FloatIds(ids, shape, device);
    }

    private static Tensor FloatIds(ReadOnlySpan<int> ids, ReadOnlySpan<int> shape, Device device)
    {
        float[] values = ArrayPool<float>.Shared.Rent(Math.Max(1, ids.Length));
        try
        {
            for (int i = 0; i < ids.Length; i++)
            {
                values[i] = ids[i];
            }

            return Tensor.From(values.AsSpan(0, ids.Length), shape, device);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(values);
        }
    }
}
