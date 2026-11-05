// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using Idrak.Abstraction.Operations;

namespace Idrak.Generation;

/// <summary>
/// Token ids as a model's input (plan 37, T1b): the generators' prompt, window and step tensors and the training batches'
/// inputs, made by the ids factory (<see cref="Tensor.Ids(ReadOnlySpan{int}, ReadOnlySpan{int}, ElementType?, Device?)"/>).
/// Their type is the run's id setting when one is given (--id-dtype, <see cref="ElementTypes.IdEnvironmentVariable"/>:
/// <see cref="ElementTypes.ChooseIds"/>, auto the smallest that holds them), else the type the device's embedding gather
/// reads indices in (int32 on every built-in device): the lookup then takes no conversion at every step, which the few
/// bytes a smaller type would save never pay back.
/// </summary>
internal static class TokenIds
{
    /// <summary><paramref name="ids"/> (whole numbers) as a tensor of <paramref name="shape"/> on <paramref name="device"/>.</summary>
    /// <exception cref="ArgumentException">An id the type asked for does not hold (the message names both).</exception>
    public static Tensor Of(ReadOnlySpan<float> ids, ReadOnlySpan<int> shape, Device device)
    {
        int[] whole = ArrayPool<int>.Shared.Rent(Math.Max(1, ids.Length));
        try
        {
            for (int i = 0; i < ids.Length; i++)
            {
                whole[i] = (int)ids[i];
            }

            return Of(whole.AsSpan(0, ids.Length), shape, device);
        }
        finally
        {
            ArrayPool<int>.Shared.Return(whole);
        }
    }

    /// <summary><paramref name="ids"/> as a tensor of <paramref name="shape"/> on <paramref name="device"/>.</summary>
    /// <exception cref="ArgumentException">An id the type asked for does not hold (the message names both).</exception>
    public static Tensor Of(ReadOnlySpan<int> ids, ReadOnlySpan<int> shape, Device device)
    {
        ArgumentNullException.ThrowIfNull(device);
        ElementType type;
        if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable(ElementTypes.IdEnvironmentVariable)))
        {
            type = device.Backend.IndexTypeOf(Ops.Gather);
        }
        else
        {
            var (smallest, largest) = Range(ids);
            type = ElementTypes.ChooseIds(largest, smallest, device: device).Type;    // the run's setting (auto: the smallest that holds them)
        }

        // A device whose gather reads float32 indices takes the ids as floats (exact up to 2^24).
        return ReferenceEquals(type, ElementType.Float32) ? FloatIds(ids, shape, device) : Tensor.Ids(ids, shape, type, device);
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

    // The smallest and largest id ((0, 0) for none).
    private static (long Smallest, long Largest) Range(ReadOnlySpan<int> ids)
    {
        if (ids.IsEmpty)
        {
            return (0, 0);
        }

        int low = ids[0], high = ids[0];
        foreach (int id in ids)
        {
            low = Math.Min(low, id);
            high = Math.Max(high, id);
        }

        return (low, high);
    }
}
