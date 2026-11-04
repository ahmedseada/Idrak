// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices.Cpu;

// The strided copy (Backend.Layout.cs) on the CPU: the output written in order (rule 72), the input read along its strides;
// a run of the last dimension with stride 1 copied as a block, outer rows spread over the cores when the copy is large.
internal sealed partial class CpuBackend
{
    public override void StridedCopyKernel(Storage x, long offset, ReadOnlySpan<int> shape, ReadOnlySpan<long> strides, Storage y)
    {
        float[] xv = D(x), yv = D(y);
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
}
