// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Hip;

// Data movement with a kernel of its own (HipKernels.Pointwise.cs): the embedding gathers and permutes (an attention
// layer's head splits and merges). Copies, fills and strided copies are in HipBackend.cs.
internal sealed unsafe partial class HipBackend
{
    // An index outside the table is held to its first or last row, as on the other GPUs (the CPU reports it).
    public override void GatherKernel(Storage table, Storage indices, Storage y, int count, int dim, int vocabulary)
    {
        if (vocabulary <= 0 || dim <= 0 || (long)count * dim > int.MaxValue
            || !Elements("gather_f32", (long)count * dim, P(table), P(indices), P(y), U(dim), U(vocabulary - 1), U(count * dim)))
        {
            base.GatherKernel(table, indices, y, count, dim, vocabulary);
        }
    }

    public override void GatherBFloat16Kernel(Storage packed, Storage indices, Storage y, int count, int dim, int vocabulary)
    {
        if (vocabulary <= 0 || dim <= 0 || (long)count * dim > int.MaxValue
            || !Elements("gather_bf16_f32", (long)count * dim, P(packed), P(indices), P(y), U(dim), U(vocabulary - 1), U((dim + 1) / 2), U(count * dim)))
        {
            base.GatherBFloat16Kernel(packed, indices, y, count, dim, vocabulary);
        }
    }

    public override void GatherBFloat16ColumnsKernel(Storage packed, Storage indices, Storage y, int count, int dim, int vocabulary)
    {
        if (vocabulary <= 0 || dim <= 0 || (long)count * dim > int.MaxValue
            || !Elements("gather_bf16_cols_f32", (long)count * dim, P(packed), P(indices), P(y), U(dim), U(vocabulary - 1), U((int)(((long)vocabulary + 1) / 2)),
                U(count * dim)))
        {
            base.GatherBFloat16ColumnsKernel(packed, indices, y, count, dim, vocabulary);
        }
    }

    // Ranks up to the kernel's six, padded in front with dimensions of size 1.
    public override void PermuteKernel(Storage x, Storage y, ReadOnlySpan<int> outShape, ReadOnlySpan<int> inStrides, bool accumulate)
    {
        const int Rank = 6;
        int rank = outShape.Length;
        long total = 1;
        foreach (int d in outShape)
        {
            total *= d;
        }

        if (rank is < 1 or > Rank || inStrides.Length != rank || total > int.MaxValue || Kernel("permute_f32") is null)
        {
            base.PermuteKernel(x, y, outShape, inStrides, accumulate);
            return;
        }

        Span<ulong> args = stackalloc ulong[2 + 2 * Rank + 2];
        args[0] = P(x);
        args[1] = P(y);
        for (int d = 0; d < Rank; d++)
        {
            int source = d - (Rank - rank);
            args[2 + d] = U(source >= 0 ? outShape[source] : 1);
            args[2 + Rank + d] = U(source >= 0 ? inStrides[source] : 0);
        }

        args[2 + 2 * Rank] = U((int)total);
        args[3 + 2 * Rank] = U(accumulate ? 1 : 0);
        Elements("permute_f32", total, args);
    }
}
