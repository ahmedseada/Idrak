// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices.Cpu;

// Weights stored as their file lays them out on the CPU: the product by the described lane (CpuBackend.Lanes.cs), which
// reads any type from its description alone and writes no weight's value anywhere; where it does not read a weight (a
// description or a row length it has no plan for, a float32 weight), the default: the product reading W through its type
// as it packs it (y = x · Wᵀ, W [n, k]), in float64 where W's values need it.
internal sealed partial class CpuBackend
{
    // The described lane reads float32 rows where they lie: 93 GB/s at 1x2048->32768 on 24 threads, where the values
    // transposed are read a run of every row at a time.
    public override bool StoredFloat32 => true;

    public override void StoredMatMulKernel(Storage x, Storage weight, Storage y, int m, int n, int k)
    {
        if (m == 0 || n == 0)
        {
            return;
        }

        if (k == 0)
        {
            D(y).AsSpan(0, m * n).Clear();
            return;
        }

        if (StoredRouted(x, weight, y, m, n, k))
        {
            return;
        }

        if (weight is MappedStorage)
        {
            throw new NotSupportedException($"The CPU's described lane does not read this weight: rows of {k} values of its description, read from a file's mapping.");
        }

        BatchedMatMulKernel(x, weight, y, 1, m, n, k, false, true, 0f);
    }
}
