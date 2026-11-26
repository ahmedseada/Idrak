// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices;

// Weights stored as their file lays them out (StoredWeight): the product of a weight whose values a type description
// reads (Storage.Description), run by a device's described lane, which reads any type from its description alone.
// tools/operations/generate.py reads this file after Backend.Elements.cs: its operations come after those in the slot order.
public abstract partial class Backend
{
    /// <summary>
    /// y [m, n] = x [m, k] · Wᵀ, W stored as <paramref name="weight"/>: n rows of k values read by its description
    /// (<see cref="Storage.Description"/>), each value times its scale from <paramref name="scales"/> as
    /// <paramref name="layout"/> places them (minus its zero point from <paramref name="zeros"/> first, when given).
    /// </summary>
    public virtual void StoredMatMulKernel(Storage x, Storage weight, Storage? scales, Storage? zeros, Storage y, int m, int n, int k, StoredScales layout) => throw NoKernel(nameof(StoredMatMul));
}
