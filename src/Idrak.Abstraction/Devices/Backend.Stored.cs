// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices.Cpu;

namespace Idrak.Abstraction.Devices;

// Weights kept in their file's format (plan 37, Q0): the product of a weight stored as a registered element type (a GGUF
// block format, float8 or int8 codes with scales) expanded a tile at a time inside the kernel, and the whole matrix
// expanded to float32. A device that runs them with kernels of its own says which types in its capabilities
// (BackendCapabilities.StoresAnyFormat, StoredFormats); a load expands weights of other types at load for it, so the host
// fallbacks below run only for a caller that asks for them. tools/operations/generate.py reads this file after
// Backend.Elements.cs: its operations come after those in the slot order.
public abstract partial class Backend
{
    /// <summary>
    /// y [m, n] = x [m, k] · W, W [k, n] stored as <paramref name="weight"/>: n rows of k values of its element type
    /// (<see cref="Storage.ElementType"/>, which widens them to float32), each value times its scale from
    /// <paramref name="scales"/> as <paramref name="layout"/> places them (minus its zero point from
    /// <paramref name="zeros"/> first, when given), as <see cref="ExpandStored"/> expands them. Every element of y is what
    /// <see cref="MatMul"/> gives over the expanded weights, bit for bit (the same values, each element one FMA chain over k
    /// from zero on the CPU): the weights are expanded a tile of outputs at a time, never whole.
    /// </summary>
    public virtual void StoredMatMulKernel(Storage x, Storage weight, Storage? scales, Storage? zeros, Storage y, int m, int n, int k, StoredScales layout)
    {
        using var h = new HostCall(this, nameof(StoredMatMul));
        CpuBackend.Instance.StoredMatMul(h[x], h[weight], h.Maybe(scales), h.Maybe(zeros), h[y], m, n, k, layout);
    }

    /// <summary>
    /// w [k, n] = the float32 weights of <paramref name="weight"/> (n stored rows of k values of its element type, as
    /// <see cref="StoredMatMul"/> reads them), transposed to the layer's [inputs, outputs]: each value widened by its type,
    /// then (value − zero) · scale where <paramref name="scales"/> are given (the value times its scale without zero
    /// points), in float32, as the stores compute them when they read the weights as float32.
    /// </summary>
    public virtual void ExpandStoredKernel(Storage weight, Storage? scales, Storage? zeros, Storage w, int n, int k, StoredScales layout)
    {
        using var h = new HostCall(this, nameof(ExpandStored));
        CpuBackend.Instance.ExpandStored(h[weight], h.Maybe(scales), h.Maybe(zeros), h[w], n, k, layout);
    }
}
