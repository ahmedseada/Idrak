// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Idrak.Backends.Cpu;

// Few-row bfloat16 products with the rows' sums in registers. The weights are read in chunks of k rows
// (CpuTuning.KChunk: half a cache line of each input row's values); within a chunk, each panel of columns (two vectors
// wide) keeps the sums of up to CpuTuning.RegisterRows rows in two vector registers each, so a widened weight vector
// feeds every row and the sums go through memory once per chunk instead of once per weight row. With AVX-512 the panels
// are 512 bits wide even where Vector<T> is 256. Each output is the same FMA chain over k as the streaming loop this
// replaces (rows whose input is 0 skipped, sums from +0, a sum stored and reloaded between chunks unchanged), and the
// columns past the last whole pair of Vector<float> are computed one at a time as there, so the results are identical.
internal sealed partial class CpuBackend
{
    private static void BFloat16FewRowsTiled(float[] packed, float[] xv, float[] yv, int m, int n, int k, int stride, int blockSize, int blocks)
    {
        int w = Vector<float>.Count;
        int rowsPerPass = CpuTuning.RegisterRows;
        int chunk = CpuTuning.KChunk;
        bool wide = CpuTuning.WidePanels && Vector512.IsHardwareAccelerated;
        For(blocks, (long)m * n * k, (first, last) =>
        {
            ref ushort halves = ref Unsafe.As<float, ushort>(ref MemoryMarshal.GetArrayDataReference(packed));
            var sums = new float[m * blockSize];
            ref float acc = ref MemoryMarshal.GetArrayDataReference(sums);
            for (int block = first; block < last; block++)
            {
                int j0 = block * blockSize, width = Math.Min(blockSize, n - j0), whole = width / (2 * w) * (2 * w);
                int wideWhole = wide ? whole / (2 * Vector512<float>.Count) * (2 * Vector512<float>.Count) : 0;
                Array.Clear(sums);
                for (int k0 = 0; k0 < k; k0 += chunk)
                {
                    int k1 = Math.Min(k, k0 + chunk);
                    int j = 0;
                    for (; j < wideWhole; j += 2 * Vector512<float>.Count)
                    {
                        for (int r0 = 0; r0 < m; r0 += 8)
                        {
                            Panel512(ref halves, stride, xv, ref Unsafe.Add(ref acc, r0 * blockSize + j), blockSize, k, r0, Math.Min(8, m - r0), j0 + j, k0, k1);
                        }
                    }

                    for (; j < whole; j += 2 * w)
                    {
                        for (int r0 = 0; r0 < m; r0 += rowsPerPass)
                        {
                            ref float target = ref Unsafe.Add(ref acc, r0 * blockSize + j);
                            if (rowsPerPass == 8)
                            {
                                Panel8(ref halves, stride, xv, ref target, blockSize, k, r0, Math.Min(8, m - r0), j0 + j, k0, k1);
                            }
                            else
                            {
                                Panel4(ref halves, stride, xv, ref target, blockSize, k, r0, Math.Min(4, m - r0), j0 + j, k0, k1);
                            }
                        }
                    }
                }

                for (int j = whole; j < width; j++)
                {
                    int col = j0 + j;
                    for (int r = 0; r < m; r++)
                    {
                        float sum = 0f;
                        for (int kk = 0; kk < k; kk++)
                        {
                            float weight = BitConverter.Int32BitsToSingle(Unsafe.Add(ref halves, (nint)kk * stride + col) << 16);
                            sum = MathF.FusedMultiplyAdd(weight, xv[r * k + kk], sum);
                        }

                        sums[r * blockSize + j] = sum;
                    }
                }

                for (int r = 0; r < m; r++)
                {
                    sums.AsSpan(r * blockSize, width).CopyTo(yv.AsSpan(r * n + j0, width));
                }
            }
        });
    }

    // Up to 8 rows × 32 columns in 16 AVX-512 registers.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Panel512(ref ushort halves, int stride, float[] x, ref float acc, int accStride, int k, int r0, int rows, int col, int k0, int k1)
    {
        const nuint w = 16;
        Vector512<float> a0 = default, b0 = default, a1 = default, b1 = default, a2 = default, b2 = default, a3 = default, b3 = default, a4 = default, b4 = default, a5 = default, b5 = default, a6 = default, b6 = default, a7 = default, b7 = default;
        { a0 = Vector512.LoadUnsafe(ref acc, (nuint)(0 * accStride)); b0 = Vector512.LoadUnsafe(ref acc, (nuint)(0 * accStride) + w); }
        if (rows > 1) { a1 = Vector512.LoadUnsafe(ref acc, (nuint)(1 * accStride)); b1 = Vector512.LoadUnsafe(ref acc, (nuint)(1 * accStride) + w); }
        if (rows > 2) { a2 = Vector512.LoadUnsafe(ref acc, (nuint)(2 * accStride)); b2 = Vector512.LoadUnsafe(ref acc, (nuint)(2 * accStride) + w); }
        if (rows > 3) { a3 = Vector512.LoadUnsafe(ref acc, (nuint)(3 * accStride)); b3 = Vector512.LoadUnsafe(ref acc, (nuint)(3 * accStride) + w); }
        if (rows > 4) { a4 = Vector512.LoadUnsafe(ref acc, (nuint)(4 * accStride)); b4 = Vector512.LoadUnsafe(ref acc, (nuint)(4 * accStride) + w); }
        if (rows > 5) { a5 = Vector512.LoadUnsafe(ref acc, (nuint)(5 * accStride)); b5 = Vector512.LoadUnsafe(ref acc, (nuint)(5 * accStride) + w); }
        if (rows > 6) { a6 = Vector512.LoadUnsafe(ref acc, (nuint)(6 * accStride)); b6 = Vector512.LoadUnsafe(ref acc, (nuint)(6 * accStride) + w); }
        if (rows > 7) { a7 = Vector512.LoadUnsafe(ref acc, (nuint)(7 * accStride)); b7 = Vector512.LoadUnsafe(ref acc, (nuint)(7 * accStride) + w); }

        ref float rx = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(x), (nint)r0 * k);
        for (int kk = k0; kk < k1; kk++)
        {
            var (low, high) = Vector512.Widen(Vector512.LoadUnsafe(ref halves, (nuint)((nint)kk * stride + col)));
            var w0 = Vector512.ShiftLeft(low, 16).AsSingle();
            var w1 = Vector512.ShiftLeft(high, 16).AsSingle();
            ref float xk = ref Unsafe.Add(ref rx, kk);
            float v = xk;
            if (v != 0f)
            {
                var s = Vector512.Create(v);
                a0 = Vector512.FusedMultiplyAdd(w0, s, a0);
                b0 = Vector512.FusedMultiplyAdd(w1, s, b0);
            }

            if (rows > 1 && (v = Unsafe.Add(ref xk, 1 * k)) != 0f)
            {
                var s = Vector512.Create(v);
                a1 = Vector512.FusedMultiplyAdd(w0, s, a1);
                b1 = Vector512.FusedMultiplyAdd(w1, s, b1);
            }

            if (rows > 2 && (v = Unsafe.Add(ref xk, 2 * k)) != 0f)
            {
                var s = Vector512.Create(v);
                a2 = Vector512.FusedMultiplyAdd(w0, s, a2);
                b2 = Vector512.FusedMultiplyAdd(w1, s, b2);
            }

            if (rows > 3 && (v = Unsafe.Add(ref xk, 3 * k)) != 0f)
            {
                var s = Vector512.Create(v);
                a3 = Vector512.FusedMultiplyAdd(w0, s, a3);
                b3 = Vector512.FusedMultiplyAdd(w1, s, b3);
            }

            if (rows > 4 && (v = Unsafe.Add(ref xk, 4 * k)) != 0f)
            {
                var s = Vector512.Create(v);
                a4 = Vector512.FusedMultiplyAdd(w0, s, a4);
                b4 = Vector512.FusedMultiplyAdd(w1, s, b4);
            }

            if (rows > 5 && (v = Unsafe.Add(ref xk, 5 * k)) != 0f)
            {
                var s = Vector512.Create(v);
                a5 = Vector512.FusedMultiplyAdd(w0, s, a5);
                b5 = Vector512.FusedMultiplyAdd(w1, s, b5);
            }

            if (rows > 6 && (v = Unsafe.Add(ref xk, 6 * k)) != 0f)
            {
                var s = Vector512.Create(v);
                a6 = Vector512.FusedMultiplyAdd(w0, s, a6);
                b6 = Vector512.FusedMultiplyAdd(w1, s, b6);
            }

            if (rows > 7 && (v = Unsafe.Add(ref xk, 7 * k)) != 0f)
            {
                var s = Vector512.Create(v);
                a7 = Vector512.FusedMultiplyAdd(w0, s, a7);
                b7 = Vector512.FusedMultiplyAdd(w1, s, b7);
            }
        }

        { a0.StoreUnsafe(ref acc, (nuint)(0 * accStride)); b0.StoreUnsafe(ref acc, (nuint)(0 * accStride) + w); }
        if (rows > 1) { a1.StoreUnsafe(ref acc, (nuint)(1 * accStride)); b1.StoreUnsafe(ref acc, (nuint)(1 * accStride) + w); }
        if (rows > 2) { a2.StoreUnsafe(ref acc, (nuint)(2 * accStride)); b2.StoreUnsafe(ref acc, (nuint)(2 * accStride) + w); }
        if (rows > 3) { a3.StoreUnsafe(ref acc, (nuint)(3 * accStride)); b3.StoreUnsafe(ref acc, (nuint)(3 * accStride) + w); }
        if (rows > 4) { a4.StoreUnsafe(ref acc, (nuint)(4 * accStride)); b4.StoreUnsafe(ref acc, (nuint)(4 * accStride) + w); }
        if (rows > 5) { a5.StoreUnsafe(ref acc, (nuint)(5 * accStride)); b5.StoreUnsafe(ref acc, (nuint)(5 * accStride) + w); }
        if (rows > 6) { a6.StoreUnsafe(ref acc, (nuint)(6 * accStride)); b6.StoreUnsafe(ref acc, (nuint)(6 * accStride) + w); }
        if (rows > 7) { a7.StoreUnsafe(ref acc, (nuint)(7 * accStride)); b7.StoreUnsafe(ref acc, (nuint)(7 * accStride) + w); }
    }

    // Up to 8 rows × 2 Vector<float> columns in 16 registers (instruction sets with 32 vector registers).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Panel8(ref ushort halves, int stride, float[] x, ref float acc, int accStride, int k, int r0, int rows, int col, int k0, int k1)
    {
        nuint w = (nuint)Vector<float>.Count;
        Vector<float> a0 = default, b0 = default, a1 = default, b1 = default, a2 = default, b2 = default, a3 = default, b3 = default, a4 = default, b4 = default, a5 = default, b5 = default, a6 = default, b6 = default, a7 = default, b7 = default;
        { a0 = Vector.LoadUnsafe(ref acc, (nuint)(0 * accStride)); b0 = Vector.LoadUnsafe(ref acc, (nuint)(0 * accStride) + w); }
        if (rows > 1) { a1 = Vector.LoadUnsafe(ref acc, (nuint)(1 * accStride)); b1 = Vector.LoadUnsafe(ref acc, (nuint)(1 * accStride) + w); }
        if (rows > 2) { a2 = Vector.LoadUnsafe(ref acc, (nuint)(2 * accStride)); b2 = Vector.LoadUnsafe(ref acc, (nuint)(2 * accStride) + w); }
        if (rows > 3) { a3 = Vector.LoadUnsafe(ref acc, (nuint)(3 * accStride)); b3 = Vector.LoadUnsafe(ref acc, (nuint)(3 * accStride) + w); }
        if (rows > 4) { a4 = Vector.LoadUnsafe(ref acc, (nuint)(4 * accStride)); b4 = Vector.LoadUnsafe(ref acc, (nuint)(4 * accStride) + w); }
        if (rows > 5) { a5 = Vector.LoadUnsafe(ref acc, (nuint)(5 * accStride)); b5 = Vector.LoadUnsafe(ref acc, (nuint)(5 * accStride) + w); }
        if (rows > 6) { a6 = Vector.LoadUnsafe(ref acc, (nuint)(6 * accStride)); b6 = Vector.LoadUnsafe(ref acc, (nuint)(6 * accStride) + w); }
        if (rows > 7) { a7 = Vector.LoadUnsafe(ref acc, (nuint)(7 * accStride)); b7 = Vector.LoadUnsafe(ref acc, (nuint)(7 * accStride) + w); }

        ref float rx = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(x), (nint)r0 * k);
        for (int kk = k0; kk < k1; kk++)
        {
            Vector.Widen(Vector.LoadUnsafe(ref halves, (nuint)((nint)kk * stride + col)), out var low, out var high);
            var w0 = Vector.AsVectorSingle(low << 16);
            var w1 = Vector.AsVectorSingle(high << 16);
            ref float xk = ref Unsafe.Add(ref rx, kk);
            float v = xk;
            if (v != 0f)
            {
                var s = new Vector<float>(v);
                a0 = Vector.FusedMultiplyAdd(w0, s, a0);
                b0 = Vector.FusedMultiplyAdd(w1, s, b0);
            }

            if (rows > 1 && (v = Unsafe.Add(ref xk, 1 * k)) != 0f)
            {
                var s = new Vector<float>(v);
                a1 = Vector.FusedMultiplyAdd(w0, s, a1);
                b1 = Vector.FusedMultiplyAdd(w1, s, b1);
            }

            if (rows > 2 && (v = Unsafe.Add(ref xk, 2 * k)) != 0f)
            {
                var s = new Vector<float>(v);
                a2 = Vector.FusedMultiplyAdd(w0, s, a2);
                b2 = Vector.FusedMultiplyAdd(w1, s, b2);
            }

            if (rows > 3 && (v = Unsafe.Add(ref xk, 3 * k)) != 0f)
            {
                var s = new Vector<float>(v);
                a3 = Vector.FusedMultiplyAdd(w0, s, a3);
                b3 = Vector.FusedMultiplyAdd(w1, s, b3);
            }

            if (rows > 4 && (v = Unsafe.Add(ref xk, 4 * k)) != 0f)
            {
                var s = new Vector<float>(v);
                a4 = Vector.FusedMultiplyAdd(w0, s, a4);
                b4 = Vector.FusedMultiplyAdd(w1, s, b4);
            }

            if (rows > 5 && (v = Unsafe.Add(ref xk, 5 * k)) != 0f)
            {
                var s = new Vector<float>(v);
                a5 = Vector.FusedMultiplyAdd(w0, s, a5);
                b5 = Vector.FusedMultiplyAdd(w1, s, b5);
            }

            if (rows > 6 && (v = Unsafe.Add(ref xk, 6 * k)) != 0f)
            {
                var s = new Vector<float>(v);
                a6 = Vector.FusedMultiplyAdd(w0, s, a6);
                b6 = Vector.FusedMultiplyAdd(w1, s, b6);
            }

            if (rows > 7 && (v = Unsafe.Add(ref xk, 7 * k)) != 0f)
            {
                var s = new Vector<float>(v);
                a7 = Vector.FusedMultiplyAdd(w0, s, a7);
                b7 = Vector.FusedMultiplyAdd(w1, s, b7);
            }
        }

        { a0.StoreUnsafe(ref acc, (nuint)(0 * accStride)); b0.StoreUnsafe(ref acc, (nuint)(0 * accStride) + w); }
        if (rows > 1) { a1.StoreUnsafe(ref acc, (nuint)(1 * accStride)); b1.StoreUnsafe(ref acc, (nuint)(1 * accStride) + w); }
        if (rows > 2) { a2.StoreUnsafe(ref acc, (nuint)(2 * accStride)); b2.StoreUnsafe(ref acc, (nuint)(2 * accStride) + w); }
        if (rows > 3) { a3.StoreUnsafe(ref acc, (nuint)(3 * accStride)); b3.StoreUnsafe(ref acc, (nuint)(3 * accStride) + w); }
        if (rows > 4) { a4.StoreUnsafe(ref acc, (nuint)(4 * accStride)); b4.StoreUnsafe(ref acc, (nuint)(4 * accStride) + w); }
        if (rows > 5) { a5.StoreUnsafe(ref acc, (nuint)(5 * accStride)); b5.StoreUnsafe(ref acc, (nuint)(5 * accStride) + w); }
        if (rows > 6) { a6.StoreUnsafe(ref acc, (nuint)(6 * accStride)); b6.StoreUnsafe(ref acc, (nuint)(6 * accStride) + w); }
        if (rows > 7) { a7.StoreUnsafe(ref acc, (nuint)(7 * accStride)); b7.StoreUnsafe(ref acc, (nuint)(7 * accStride) + w); }
    }

    // Up to 4 rows × 2 Vector<float> columns in 8 registers (instruction sets with 16 vector registers).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Panel4(ref ushort halves, int stride, float[] x, ref float acc, int accStride, int k, int r0, int rows, int col, int k0, int k1)
    {
        nuint w = (nuint)Vector<float>.Count;
        Vector<float> a0 = default, b0 = default, a1 = default, b1 = default, a2 = default, b2 = default, a3 = default, b3 = default;
        { a0 = Vector.LoadUnsafe(ref acc, (nuint)(0 * accStride)); b0 = Vector.LoadUnsafe(ref acc, (nuint)(0 * accStride) + w); }
        if (rows > 1) { a1 = Vector.LoadUnsafe(ref acc, (nuint)(1 * accStride)); b1 = Vector.LoadUnsafe(ref acc, (nuint)(1 * accStride) + w); }
        if (rows > 2) { a2 = Vector.LoadUnsafe(ref acc, (nuint)(2 * accStride)); b2 = Vector.LoadUnsafe(ref acc, (nuint)(2 * accStride) + w); }
        if (rows > 3) { a3 = Vector.LoadUnsafe(ref acc, (nuint)(3 * accStride)); b3 = Vector.LoadUnsafe(ref acc, (nuint)(3 * accStride) + w); }

        ref float rx = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(x), (nint)r0 * k);
        for (int kk = k0; kk < k1; kk++)
        {
            Vector.Widen(Vector.LoadUnsafe(ref halves, (nuint)((nint)kk * stride + col)), out var low, out var high);
            var w0 = Vector.AsVectorSingle(low << 16);
            var w1 = Vector.AsVectorSingle(high << 16);
            ref float xk = ref Unsafe.Add(ref rx, kk);
            float v = xk;
            if (v != 0f)
            {
                var s = new Vector<float>(v);
                a0 = Vector.FusedMultiplyAdd(w0, s, a0);
                b0 = Vector.FusedMultiplyAdd(w1, s, b0);
            }

            if (rows > 1 && (v = Unsafe.Add(ref xk, 1 * k)) != 0f)
            {
                var s = new Vector<float>(v);
                a1 = Vector.FusedMultiplyAdd(w0, s, a1);
                b1 = Vector.FusedMultiplyAdd(w1, s, b1);
            }

            if (rows > 2 && (v = Unsafe.Add(ref xk, 2 * k)) != 0f)
            {
                var s = new Vector<float>(v);
                a2 = Vector.FusedMultiplyAdd(w0, s, a2);
                b2 = Vector.FusedMultiplyAdd(w1, s, b2);
            }

            if (rows > 3 && (v = Unsafe.Add(ref xk, 3 * k)) != 0f)
            {
                var s = new Vector<float>(v);
                a3 = Vector.FusedMultiplyAdd(w0, s, a3);
                b3 = Vector.FusedMultiplyAdd(w1, s, b3);
            }
        }

        { a0.StoreUnsafe(ref acc, (nuint)(0 * accStride)); b0.StoreUnsafe(ref acc, (nuint)(0 * accStride) + w); }
        if (rows > 1) { a1.StoreUnsafe(ref acc, (nuint)(1 * accStride)); b1.StoreUnsafe(ref acc, (nuint)(1 * accStride) + w); }
        if (rows > 2) { a2.StoreUnsafe(ref acc, (nuint)(2 * accStride)); b2.StoreUnsafe(ref acc, (nuint)(2 * accStride) + w); }
        if (rows > 3) { a3.StoreUnsafe(ref acc, (nuint)(3 * accStride)); b3.StoreUnsafe(ref acc, (nuint)(3 * accStride) + w); }
    }
}
