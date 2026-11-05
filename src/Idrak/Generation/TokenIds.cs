// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Runtime.InteropServices;
using Idrak.Abstraction.Operations;

namespace Idrak.Generation;

/// <summary>
/// Token ids as a model's input (plan 37, T1b): the generators' prompt, window and step tensors, in the element type the
/// device's embedding gather reads indices in (int32 on every built-in device), so the lookup takes no conversion.
/// </summary>
internal static class TokenIds
{
    /// <summary>
    /// <paramref name="ids"/> (whole numbers) as a tensor of <paramref name="shape"/> on <paramref name="device"/>: int32,
    /// or float32 on a device whose gather reads its indices so.
    /// </summary>
    public static Tensor Of(ReadOnlySpan<float> ids, ReadOnlySpan<int> shape, Device device)
    {
        var type = device.Backend.IndexTypeOf(Ops.Gather);
        if (ReferenceEquals(type, ElementType.Float32))
        {
            return Tensor.From(ids, shape, device);
        }

        int[] whole = ArrayPool<int>.Shared.Rent(Math.Max(1, ids.Length));
        try
        {
            for (int i = 0; i < ids.Length; i++)
            {
                whole[i] = (int)ids[i];
            }

            return Tensor.FromBytes(MemoryMarshal.AsBytes(whole.AsSpan(0, ids.Length)), ElementType.Int32, shape, device);
        }
        finally
        {
            ArrayPool<int>.Shared.Return(whole);
        }
    }
}
