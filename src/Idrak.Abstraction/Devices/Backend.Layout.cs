// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices.Cpu;

namespace Idrak.Abstraction.Devices;

// Tensor layouts (plan 37, V0 and V1): the copy that makes a strided view contiguous. A device that reads strided storage
// says so in its capabilities (BackendCapabilities.StridedInputs); tensors on it are views, and this copy runs once where
// something reads a view's storage directly, or for an operation that does not take strides. tools/operations/generate.py
// reads this file after Backend.cs and Backend.Routing.cs: its operation comes after theirs in the slot order.
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
}
