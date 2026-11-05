// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;

namespace Idrak.Abstraction.Devices.Cpu;

// The strided copy and the operations that read views where they are (Backend.Layout.cs) on the CPU. The copy writes its
// output in order (rule 72) and reads the input along its strides; a run of the last dimension with stride 1 is copied as
// a block, outer rows spread over the cores when the copy is large. The product and the attention compute exactly what
// their contiguous kernels compute on the same elements: the attention reads each row where its layout puts it, the
// product packs an operand it cannot read in place into a pooled buffer of the order its kernel reads.
internal sealed partial class CpuBackend
{
    public override void StridedCopyKernel(Storage x, long offset, ReadOnlySpan<int> shape, ReadOnlySpan<long> strides, Storage y) =>
        CopyStrided(D(x), offset, shape, strides, D(y));

    private void CopyStrided(float[] xv, long offset, ReadOnlySpan<int> shape, ReadOnlySpan<long> strides, float[] yv)
    {
        int rank = shape.Length;
        if (rank == 0)
        {
            yv[0] = xv[offset];
            return;
        }

        int inner = shape[rank - 1];
        long innerStride = strides[rank - 1];
        long count = 1;
        foreach (int d in shape)
        {
            count *= d;
        }

        if (count == 0)
        {
            return;
        }

        int rows = (int)(count / Math.Max(1, inner));
        int[] outer = shape[..^1].ToArray();
        long[] outerStrides = strides[..^1].ToArray();
        For(rows, count, (first, last) =>
        {
            // The input element of the first row of this range, then stepped along the outer dimensions row by row.
            Span<int> index = outer.Length <= 16 ? stackalloc int[outer.Length] : new int[outer.Length];
            int remainder = first;
            for (int d = outer.Length - 1; d >= 0; d--)
            {
                index[d] = remainder % outer[d];
                remainder /= outer[d];
            }

            for (int row = first; row < last; row++)
            {
                long at = offset;
                for (int d = 0; d < outer.Length; d++)
                {
                    at += index[d] * outerStrides[d];
                }

                var destination = yv.AsSpan(row * inner, inner);
                if (innerStride == 1)
                {
                    xv.AsSpan((int)at, inner).CopyTo(destination);
                }
                else
                {
                    for (int j = 0; j < inner; j++)
                    {
                        destination[j] = xv[at + j * innerStride];
                    }
                }

                for (int d = outer.Length - 1; d >= 0; d--)
                {
                    if (++index[d] < outer[d])
                    {
                        break;
                    }

                    index[d] = 0;
                }
            }
        });
    }

    public override void BatchedMatMulStridedKernel(Storage a, TensorLayout aLayout, Storage b, TensorLayout bLayout, Storage c, int batch, int m, int n, int k,
        bool transA, bool transB, float beta)
    {
        float[] av = D(a), bv = D(b);
        float[]? packedA = null, packedB = null;
        try
        {
            int ra = transA ? k : m, ca = transA ? m : k, rb = transB ? n : k, cb = transB ? k : n;   // as stored: [batch, ra, ca], [batch, rb, cb]
            if (!aLayout.IsContiguousFromStart)
            {
                // A transposed view of contiguous elements is read in place with its flag turned over (no sum changes: the
                // flag only says where each element is); any other view is packed in its order.
                if (aLayout.Reshape([batch, ra, ca])?.Permute([0, 2, 1]) is { IsContiguousFromStart: true })
                {
                    transA = !transA;
                }
                else
                {
                    av = packedA = Pack(av, aLayout);
                }
            }

            if (!bLayout.IsContiguousFromStart)
            {
                // B stored [n, k] reaches the product transposed to [k, n] (CpuMatMul.Multiply), unless few rows read it in place
                // (lane sums) or the product runs in chunks of k (CpuMatMul.SplitsK, from the shape and the threads): there it
                // is packed straight to [k, n] here, the product's own transpose skipped, every sum the same. Otherwise it is
                // packed in its order, its flag as it was.
                bool chunks = CpuMatMul.SplitsK(m, n, k);
                if (transB && !chunks && m > CpuTuning.TransposedInPlace && bLayout.Reshape([batch, rb, cb]) is { } stored)
                {
                    bv = packedB = Pack(bv, stored.Permute([0, 2, 1]));
                    transB = false;
                }
                else
                {
                    bv = packedB = Pack(bv, bLayout);
                }
            }

            BatchedProduct(av, bv, D(c), batch, m, n, k, transA, transB, beta);
        }
        finally
        {
            if (packedA is not null)
            {
                ArrayPool<float>.Shared.Return(packedA);
            }

            if (packedB is not null)
            {
                ArrayPool<float>.Shared.Return(packedB);
            }
        }
    }

    // The elements of `layout` over `source` in its row-major order, in a pooled buffer the caller returns.
    private float[] Pack(float[] source, TensorLayout layout)
    {
        var packed = ArrayPool<float>.Shared.Rent(checked((int)Math.Max(1, layout.Elements)));
        CopyStrided(source, layout.Offset, layout.Shape, layout.Strides, packed);
        return packed;
    }

    public override void AttentionTiledStridedKernel(Storage q, TensorLayout qLayout, Storage keys, TensorLayout keysLayout, Storage values,
        TensorLayout valuesLayout, Storage position, Storage y, Storage? logSumExp, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale,
        AttentionVariant variant = default)
    {
        long[]? qRows = null, kRows = null, vRows = null;
        try
        {
            qRows = Rows(qLayout, heads * rowsPerHead, dim);
            kRows = Rows(keysLayout, heads * capacity, dim);
            vRows = Rows(valuesLayout, heads * capacity, dim);
            AttentionTiledRows(D(q), qRows, D(keys), kRows, D(values), vRows, D(position), D(y), logSumExp is null ? null : D(logSumExp), heads, rowsPerHead,
                steps, capacity, dim, scale, variant);
        }
        finally
        {
            ReturnRows(qRows, kRows, vRows);
        }
    }

    public override void AttentionTiledStridedBackwardKernel(Storage q, TensorLayout qLayout, Storage keys, TensorLayout keysLayout, Storage values,
        TensorLayout valuesLayout, Storage output, Storage logSumExp, Storage dOutput, Storage dq, Storage dkeys, Storage dvalues, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        long[]? qRows = null, kRows = null, vRows = null;
        try
        {
            qRows = Rows(qLayout, heads * rowsPerHead, dim);
            kRows = Rows(keysLayout, heads * capacity, dim);
            vRows = Rows(valuesLayout, heads * capacity, dim);
            AttentionTiledBackwardRows(D(q), qRows, D(keys), kRows, D(values), vRows, D(output), D(logSumExp), D(dOutput), D(dq), D(dkeys), D(dvalues), heads,
                rowsPerHead, steps, capacity, dim, scale, variant);
        }
        finally
        {
            ReturnRows(qRows, kRows, vRows);
        }
    }

    // Where each of `rows` runs of `dim` elements of `layout` starts (pooled; the caller returns it), or null when they sit
    // one after another from element 0.
    private static long[]? Rows(TensorLayout layout, int rows, int dim)
    {
        if (layout.IsContiguousFromStart)
        {
            return null;
        }

        if (!layout.HasRows(dim) || layout.Elements < (long)rows * dim)
        {
            throw new ArgumentException($"The attention reads rows of {dim} elements one after another; the layout {layout} does not hold {rows} of them.", nameof(layout));
        }

        var offsets = ArrayPool<long>.Shared.Rent(Math.Max(1, rows));
        layout.RowOffsets(dim, offsets.AsSpan(0, rows));
        return offsets;
    }

    private static void ReturnRows(params ReadOnlySpan<long[]?> tables)
    {
        foreach (var table in tables)
        {
            if (table is not null)
            {
                ArrayPool<long>.Shared.Return(table);
            }
        }
    }
}
