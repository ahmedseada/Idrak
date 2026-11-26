// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices.Cpu;

// Weights stored as their file lays them out on the CPU: the product by the described lane (CpuBackend.Lanes.cs), which
// reads any type from its description alone and writes no weight's value anywhere.
internal sealed partial class CpuBackend
{
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

        throw new NotSupportedException(weight.Description is null
            ? "The CPU multiplies stored weights by their type's description; this weight's storage has none."
            : $"The CPU's described lane does not read this weight: rows of {k} values of its description{(weight is MappedStorage ? ", read from a file's mapping" : "")}.");
    }
}
