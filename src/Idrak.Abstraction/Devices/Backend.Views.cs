// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.


namespace Idrak.Abstraction.Devices;

// More operations that read views where they are (plan 37, V1b; after Backend.Layout.cs's): the key/value cache's write,
// whose keys and values come split from a projection and permuted (a view), so a decoding pass writes them into the cache
// with no contiguous copy made first. tools/operations/generate.py reads this file last: its operations' slots come after
// every other's.
public abstract partial class Backend
{
    /// <summary>
    /// <see cref="KeyValueWrite"/> with the source read through its layout (plan 37, V1b): row r (of
    /// <paramref name="dim"/> elements) of the source [heads, steps, dim] is the r-th run of dim elements of
    /// <paramref name="sourceLayout"/> over <paramref name="source"/> in its row-major order (keys or values split from a
    /// projection and permuted, read where they are), written into the [heads, capacity, dim] cache at the device-side
    /// position as <see cref="KeyValueWrite"/> writes it, bit for bit. A device that reads strides here lists this
    /// operation in <see cref="BackendCapabilities.StridedInputs"/>.
    /// </summary>
    public virtual void KeyValueWriteStridedKernel(Storage source, TensorLayout sourceLayout, Storage cache, Storage position, int heads, int steps, int capacity, int dim)
    {
        using var h = new HostCall(this, nameof(KeyValueWriteStrided));
        Cpu.CpuBackend.Instance.KeyValueWriteStrided(h[source], sourceLayout, h[cache], h[position], heads, steps, capacity, dim);
    }
}
