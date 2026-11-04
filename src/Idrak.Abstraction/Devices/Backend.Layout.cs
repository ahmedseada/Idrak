// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices.Cpu;

namespace Idrak.Abstraction.Devices;

// Tensor layouts (plan 37, V0, V1 and V1b): the copy that makes a strided view contiguous, and the operations that read
// views where they are (the matrix product and the tiled attention, forward and backward). A device that reads strided
// storage says so in its capabilities (BackendCapabilities.StridedInputs); tensors on it are views, the operations it lists
// read them through their layouts, and the copy runs once where something else reads a view's storage directly.
// tools/operations/generate.py reads this file after Backend.cs and Backend.Routing.cs: its operations come after theirs
// in the slot order.
public abstract partial class Backend
{
    /// <summary>
    /// y[i] = x[offset + Σₖ iₖ·strides[k]] for every index i of <paramref name="shape"/> in row-major order: a strided
    /// view of <paramref name="x"/> (a permutation, a slice, a broadcast with stride 0) copied to the contiguous
    /// <paramref name="y"/> of shape's elements. Every element is copied as it is.
    /// </summary>
    public virtual void StridedCopyKernel(Storage x, long offset, ReadOnlySpan<int> shape, ReadOnlySpan<long> strides, Storage y)
    {
        using var h = new HostCall(this, nameof(StridedCopy));
        CpuBackend.Instance.StridedCopy(h[x], offset, shape, strides, h[y]);
    }

    /// <summary>
    /// <see cref="BatchedMatMul"/> with each operand read through its layout (plan 37, V1b): the elements of
    /// <paramref name="a"/> [batch, m, k] (stored [batch, k, m] when <paramref name="transA"/>) are those of
    /// <paramref name="aLayout"/> over <paramref name="a"/> in its row-major order, and the same for b; a view is read as it
    /// is, with no contiguous copy made first. The results are <see cref="BatchedMatMul"/>'s on the operands' elements, bit
    /// for bit. A device that reads strides in its products lists this operation in
    /// <see cref="BackendCapabilities.StridedInputs"/>.
    /// </summary>
    public virtual void BatchedMatMulStridedKernel(Storage a, TensorLayout aLayout, Storage b, TensorLayout bLayout, Storage c, int batch, int m, int n, int k,
        bool transA, bool transB, float beta)
    {
        using var h = new HostCall(this, nameof(BatchedMatMulStrided));
        CpuBackend.Instance.BatchedMatMulStrided(h[a], aLayout, h[b], bLayout, h[c], batch, m, n, k, transA, transB, beta);
    }

    /// <summary>
    /// <see cref="AttentionTiled"/> with q, keys and values read through their layouts (plan 37, V1b): row r of q
    /// [heads, rowsPerHead, dim] is the r-th run of dim elements of <paramref name="qLayout"/> over <paramref name="q"/> in
    /// its row-major order (heads split from a projection and permuted, read where they are), and the same for keys and
    /// values [heads, capacity, dim]; y is contiguous. Each layout holds its rows one after another
    /// (its last dimension a multiple of dim, stride 1). The results are <see cref="AttentionTiled"/>'s, bit for bit.
    /// </summary>
    public virtual void AttentionTiledStridedKernel(Storage q, TensorLayout qLayout, Storage keys, TensorLayout keysLayout, Storage values, TensorLayout valuesLayout,
        Storage position, Storage y, Storage? logSumExp, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        using var h = new HostCall(this, nameof(AttentionTiledStrided));
        CpuBackend.Instance.AttentionTiledStrided(h[q], qLayout, h[keys], keysLayout, h[values], valuesLayout, h[position], h[y], h.Maybe(logSumExp), heads,
            rowsPerHead, steps, capacity, dim, scale, variant);
    }

    /// <summary>
    /// <see cref="AttentionTiledBackward"/> with q, keys and values read through their layouts as
    /// <see cref="AttentionTiledStrided"/> reads them; output, dOutput and the gradients dq, dkeys, dvalues are contiguous.
    /// The results are <see cref="AttentionTiledBackward"/>'s, bit for bit.
    /// </summary>
    public virtual void AttentionTiledStridedBackwardKernel(Storage q, TensorLayout qLayout, Storage keys, TensorLayout keysLayout, Storage values,
        TensorLayout valuesLayout, Storage output, Storage logSumExp, Storage dOutput, Storage dq, Storage dkeys, Storage dvalues, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        using var h = new HostCall(this, nameof(AttentionTiledStridedBackward));
        CpuBackend.Instance.AttentionTiledStridedBackward(h[q], qLayout, h[keys], keysLayout, h[values], valuesLayout, h[output], h[logSumExp], h[dOutput],
            h[dq], h[dkeys], h[dvalues], heads, rowsPerHead, steps, capacity, dim, scale, variant);
    }
}
