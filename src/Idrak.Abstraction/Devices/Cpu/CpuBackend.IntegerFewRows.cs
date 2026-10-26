// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Idrak.Abstraction.Devices.Cpu;

// Few-row int8 and int4 products with the rows' sums in registers, shaped as BFloat16FewRowsTiled: the weights are read in
// chunks of k rows (CpuTuning.KChunk), and within a chunk each panel of columns keeps the sums of several input rows in
// vector registers, so a weight vector is widened (int8: bytes to floats; int4: nibbles to floats times their scales) once
// and used for every row, and the sums go through memory once per chunk instead of once per weight row. With one row
// (decoding) two panels run side by side, sharing each weight row's loop work, and a weight row whose input is 0 is not read.
//
// Each output keeps the operations and the order of the streaming loops these replace (still run for
// CpuTuning.StreamingFewRows or without SIMD), so the results are identical, bit for bit: the sum starts at +0, and for
// k ascending, every row whose input is not 0 adds weight · input as a multiply and then an add (no FMA; int8: the byte as
// a float, int4: nibble · scale as a float), a sum stored and reloaded between chunks unchanged; int8 multiplies by the
// column's scale at the end. Columns past the last whole panel run the same operations one at a time.
internal sealed partial class CpuBackend
{
    // The int4 lanes' shifts: lane l of the v-th float vector of a panel holds column v·w + l of its first word, and
    // shifting the word left by 28 - 4·(column mod 8) moves that column's nibble to the top. First and Second hold the
    // counts where the CPU shifts each lane by its own count (ShiftLanes), else 2^count to multiply by (the same bits).
    private static class Int4Lanes
    {
        // Whether ShiftLanes takes the CPU's per-lane shift: fixed when the int4 kernels first run (ComputeResources.CpuVectors).
        internal static readonly bool Shifts = LaneShifts;

        internal static readonly Vector<int> First = Operands(0, counts: Shifts), Second = Operands(1, counts: Shifts);

        // Lanes 0-7 from the first of two words (16-lane vectors only).
        internal static readonly Vector<int> LowWord = Vector.LessThan(Lane(), new Vector<int>(8));

        private static Vector<int> Lane()
        {
            Span<int> lanes = stackalloc int[Vector<int>.Count];
            for (int l = 0; l < lanes.Length; l++)
            {
                lanes[l] = l;
            }

            return new Vector<int>(lanes);
        }

        private static Vector<int> Operands(int v, bool counts)
        {
            Span<int> lanes = stackalloc int[Vector<int>.Count];
            for (int l = 0; l < lanes.Length; l++)
            {
                int count = 28 - 4 * ((v * lanes.Length + l) & 7);
                lanes[l] = counts ? count : 1 << count;
            }

            return new Vector<int>(lanes);
        }
    }

    // Whether the int4 panels can place their lanes: one word per four, eight or sixteen float lanes.
    private static bool Int4PanelsFit => Vector<int>.Count is 4 or 8 or 16;

    private static void Int8FewRowsTiled(float[] q, float[] xv, float[] sv, float[] yv, int m, int n, int k, int stride, int blockSize, int blocks) =>
        For(blocks, (long)m * n * k, new Int8Blocks(q, xv, sv, yv, m, n, k, stride, blockSize));

    // A range of column blocks of the int8 product (a struct: no closure a call).
    private readonly struct Int8Blocks(float[] q, float[] xv, float[] sv, float[] yv, int m, int n, int k, int stride, int blockSize) : IRangeBody
    {
        public void Run(int first, int last)
        {
            var sums = ArrayPool<float>.Shared.Rent(m * blockSize);
            try
            {
                for (int block = first; block < last; block++)
                {
                    Int8Block(q, xv, sv, yv, sums, m, n, k, stride, blockSize, block);
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(sums);
            }
        }
    }

    // One column block of the int8 product: panels one byte vector wide (four float vectors), up to half the register rows
    // a pass (four accumulators per row).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Int8Block(float[] q, float[] xv, float[] sv, float[] yv, float[] sums, int m, int n, int k, int stride, int blockSize, int block)
    {
        int panel = Vector<sbyte>.Count;
        int rowsPerPass = Math.Max(2, CpuTuning.RegisterRows / 2);
        int chunk = CpuTuning.KChunk;
        int j0 = block * blockSize, width = Math.Min(blockSize, n - j0), whole = width / panel * panel;
        ref sbyte bytes = ref Unsafe.As<float, sbyte>(ref MemoryMarshal.GetArrayDataReference(q));
        ref float acc = ref MemoryMarshal.GetArrayDataReference(sums);
        sums.AsSpan(0, m * blockSize).Clear();
        for (int k0 = 0; k0 < k; k0 += chunk)
        {
            int k1 = Math.Min(k, k0 + chunk);
            int j = 0;
            if (m == 1)
            {
                // One row (decoding): two panels side by side, the work of each weight row shared by eight vectors.
                for (; j + 2 * panel <= whole; j += 2 * panel)
                {
                    Int8Panel1Wide(ref bytes, stride, xv, ref Unsafe.Add(ref acc, j), j0 + j, k0, k1);
                }
            }

            for (; j < whole; j += panel)
            {
                for (int r0 = 0; r0 < m; r0 += rowsPerPass)
                {
                    ref float target = ref Unsafe.Add(ref acc, r0 * blockSize + j);
                    if (rowsPerPass >= 4)
                    {
                        Int8Panel4(ref bytes, stride, xv, ref target, blockSize, k, r0, Math.Min(4, m - r0), j0 + j, k0, k1);
                    }
                    else
                    {
                        Int8Panel2(ref bytes, stride, xv, ref target, blockSize, k, r0, Math.Min(2, m - r0), j0 + j, k0, k1);
                    }
                }
            }
        }

        if (whole < width)
        {
            for (int kk = 0; kk < k; kk++)
            {
                ref sbyte row = ref Unsafe.Add(ref bytes, (nint)kk * stride + j0);
                for (int r = 0; r < m; r++)
                {
                    float xk = xv[r * k + kk];
                    if (xk == 0f)
                    {
                        continue;
                    }

                    ref float rowSums = ref Unsafe.Add(ref acc, r * blockSize);
                    for (int j = whole; j < width; j++)
                    {
                        Unsafe.Add(ref rowSums, j) += xk * Unsafe.Add(ref row, j);
                    }
                }
            }
        }

        // y = sum · the column's scale.
        int w = Vector<float>.Count;
        for (int r = 0; r < m; r++)
        {
            var rowSums = sums.AsSpan(r * blockSize, width);
            var scale = sv.AsSpan(j0, width);
            var output = yv.AsSpan(r * n + j0, width);
            int j = 0;
            for (; j <= width - w; j += w)
            {
                (new Vector<float>(rowSums[j..]) * new Vector<float>(scale[j..])).CopyTo(output[j..]);
            }

            for (; j < width; j++)
            {
                output[j] = rowSums[j] * scale[j];
            }
        }
    }

    // One row × 8 Vector<float> columns (two panels), each byte vector widened and added in turn.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Int8Panel1Wide(ref sbyte bytes, int stride, float[] x, ref float acc, int col, int k0, int k1)
    {
        nuint w = (nuint)Vector<float>.Count, panel = (nuint)Vector<sbyte>.Count;
        var a0 = Vector.LoadUnsafe(ref acc);
        var b0 = Vector.LoadUnsafe(ref acc, w);
        var c0 = Vector.LoadUnsafe(ref acc, 2 * w);
        var d0 = Vector.LoadUnsafe(ref acc, 3 * w);
        var a1 = Vector.LoadUnsafe(ref acc, 4 * w);
        var b1 = Vector.LoadUnsafe(ref acc, 5 * w);
        var c1 = Vector.LoadUnsafe(ref acc, 6 * w);
        var d1 = Vector.LoadUnsafe(ref acc, 7 * w);
        ref float rx = ref MemoryMarshal.GetArrayDataReference(x);
        ref sbyte start = ref Unsafe.Add(ref bytes, col);
        for (int kk = k0; kk < k1; kk++)
        {
            float v = Unsafe.Add(ref rx, kk);
            if (v == 0f)
            {
                continue;
            }

            var s = new Vector<float>(v);
            ref sbyte row = ref Unsafe.Add(ref start, (nint)kk * stride);
            Vector.Widen(Vector.LoadUnsafe(ref row), out var lowShorts, out var highShorts);
            Vector.Widen(lowShorts, out var i0, out var i1);
            Vector.Widen(highShorts, out var i2, out var i3);
            a0 = Vector.ConvertToSingle(i0) * s + a0;
            b0 = Vector.ConvertToSingle(i1) * s + b0;
            c0 = Vector.ConvertToSingle(i2) * s + c0;
            d0 = Vector.ConvertToSingle(i3) * s + d0;
            Vector.Widen(Vector.LoadUnsafe(ref row, panel), out lowShorts, out highShorts);
            Vector.Widen(lowShorts, out i0, out i1);
            Vector.Widen(highShorts, out i2, out i3);
            a1 = Vector.ConvertToSingle(i0) * s + a1;
            b1 = Vector.ConvertToSingle(i1) * s + b1;
            c1 = Vector.ConvertToSingle(i2) * s + c1;
            d1 = Vector.ConvertToSingle(i3) * s + d1;
        }

        a0.StoreUnsafe(ref acc);
        b0.StoreUnsafe(ref acc, w);
        c0.StoreUnsafe(ref acc, 2 * w);
        d0.StoreUnsafe(ref acc, 3 * w);
        a1.StoreUnsafe(ref acc, 4 * w);
        b1.StoreUnsafe(ref acc, 5 * w);
        c1.StoreUnsafe(ref acc, 6 * w);
        d1.StoreUnsafe(ref acc, 7 * w);
    }

    // Up to 4 rows × 4 Vector<float> columns in 16 registers (instruction sets with 32 vector registers).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Int8Panel4(ref sbyte bytes, int stride, float[] x, ref float acc, int accStride, int k, int r0, int rows, int col, int k0, int k1)
    {
        nuint w = (nuint)Vector<float>.Count;
        Vector<float> a0, b0, c0, d0, a1 = default, b1 = default, c1 = default, d1 = default, a2 = default, b2 = default, c2 = default, d2 = default, a3 = default, b3 = default, c3 = default, d3 = default;
        ref float t0 = ref acc;
        { a0 = Vector.LoadUnsafe(ref t0); b0 = Vector.LoadUnsafe(ref t0, w); c0 = Vector.LoadUnsafe(ref t0, 2 * w); d0 = Vector.LoadUnsafe(ref t0, 3 * w); }
        if (rows > 1) { ref float t = ref Unsafe.Add(ref acc, 1 * accStride); a1 = Vector.LoadUnsafe(ref t); b1 = Vector.LoadUnsafe(ref t, w); c1 = Vector.LoadUnsafe(ref t, 2 * w); d1 = Vector.LoadUnsafe(ref t, 3 * w); }
        if (rows > 2) { ref float t = ref Unsafe.Add(ref acc, 2 * accStride); a2 = Vector.LoadUnsafe(ref t); b2 = Vector.LoadUnsafe(ref t, w); c2 = Vector.LoadUnsafe(ref t, 2 * w); d2 = Vector.LoadUnsafe(ref t, 3 * w); }
        if (rows > 3) { ref float t = ref Unsafe.Add(ref acc, 3 * accStride); a3 = Vector.LoadUnsafe(ref t); b3 = Vector.LoadUnsafe(ref t, w); c3 = Vector.LoadUnsafe(ref t, 2 * w); d3 = Vector.LoadUnsafe(ref t, 3 * w); }

        ref float rx = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(x), (nint)r0 * k);
        for (int kk = k0; kk < k1; kk++)
        {
            // The byte vector widened once (sbyte → short → int → float), as AddScaled widens it.
            Vector.Widen(Vector.LoadUnsafe(ref bytes, (nuint)((nint)kk * stride + col)), out var lowShorts, out var highShorts);
            Vector.Widen(lowShorts, out var i0, out var i1);
            Vector.Widen(highShorts, out var i2, out var i3);
            var w0 = Vector.ConvertToSingle(i0);
            var w1 = Vector.ConvertToSingle(i1);
            var w2 = Vector.ConvertToSingle(i2);
            var w3 = Vector.ConvertToSingle(i3);
            ref float xk = ref Unsafe.Add(ref rx, kk);
            float v = xk;
            if (v != 0f)
            {
                var s = new Vector<float>(v);
                a0 = w0 * s + a0; b0 = w1 * s + b0; c0 = w2 * s + c0; d0 = w3 * s + d0;
            }

            if (rows <= 1)
            {
                continue;
            }

            if ((v = Unsafe.Add(ref xk, 1 * k)) != 0f)
            {
                var s = new Vector<float>(v);
                a1 = w0 * s + a1; b1 = w1 * s + b1; c1 = w2 * s + c1; d1 = w3 * s + d1;
            }

            if (rows <= 2)
            {
                continue;
            }

            if ((v = Unsafe.Add(ref xk, 2 * k)) != 0f)
            {
                var s = new Vector<float>(v);
                a2 = w0 * s + a2; b2 = w1 * s + b2; c2 = w2 * s + c2; d2 = w3 * s + d2;
            }

            if (rows <= 3)
            {
                continue;
            }

            if ((v = Unsafe.Add(ref xk, 3 * k)) != 0f)
            {
                var s = new Vector<float>(v);
                a3 = w0 * s + a3; b3 = w1 * s + b3; c3 = w2 * s + c3; d3 = w3 * s + d3;
            }
        }

        { a0.StoreUnsafe(ref t0); b0.StoreUnsafe(ref t0, w); c0.StoreUnsafe(ref t0, 2 * w); d0.StoreUnsafe(ref t0, 3 * w); }
        if (rows > 1) { ref float t = ref Unsafe.Add(ref acc, 1 * accStride); a1.StoreUnsafe(ref t); b1.StoreUnsafe(ref t, w); c1.StoreUnsafe(ref t, 2 * w); d1.StoreUnsafe(ref t, 3 * w); }
        if (rows > 2) { ref float t = ref Unsafe.Add(ref acc, 2 * accStride); a2.StoreUnsafe(ref t); b2.StoreUnsafe(ref t, w); c2.StoreUnsafe(ref t, 2 * w); d2.StoreUnsafe(ref t, 3 * w); }
        if (rows > 3) { ref float t = ref Unsafe.Add(ref acc, 3 * accStride); a3.StoreUnsafe(ref t); b3.StoreUnsafe(ref t, w); c3.StoreUnsafe(ref t, 2 * w); d3.StoreUnsafe(ref t, 3 * w); }
    }

    // Up to 2 rows × 4 Vector<float> columns in 8 registers (instruction sets with 16 vector registers).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Int8Panel2(ref sbyte bytes, int stride, float[] x, ref float acc, int accStride, int k, int r0, int rows, int col, int k0, int k1)
    {
        nuint w = (nuint)Vector<float>.Count;
        Vector<float> a0, b0, c0, d0, a1 = default, b1 = default, c1 = default, d1 = default;
        ref float t0 = ref acc;
        ref float t1 = ref Unsafe.Add(ref acc, accStride);
        { a0 = Vector.LoadUnsafe(ref t0); b0 = Vector.LoadUnsafe(ref t0, w); c0 = Vector.LoadUnsafe(ref t0, 2 * w); d0 = Vector.LoadUnsafe(ref t0, 3 * w); }
        if (rows > 1) { a1 = Vector.LoadUnsafe(ref t1); b1 = Vector.LoadUnsafe(ref t1, w); c1 = Vector.LoadUnsafe(ref t1, 2 * w); d1 = Vector.LoadUnsafe(ref t1, 3 * w); }

        ref float rx = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(x), (nint)r0 * k);
        for (int kk = k0; kk < k1; kk++)
        {
            Vector.Widen(Vector.LoadUnsafe(ref bytes, (nuint)((nint)kk * stride + col)), out var lowShorts, out var highShorts);
            Vector.Widen(lowShorts, out var i0, out var i1);
            Vector.Widen(highShorts, out var i2, out var i3);
            var w0 = Vector.ConvertToSingle(i0);
            var w1 = Vector.ConvertToSingle(i1);
            var w2 = Vector.ConvertToSingle(i2);
            var w3 = Vector.ConvertToSingle(i3);
            ref float xk = ref Unsafe.Add(ref rx, kk);
            float v = xk;
            if (v != 0f)
            {
                var s = new Vector<float>(v);
                a0 = w0 * s + a0; b0 = w1 * s + b0; c0 = w2 * s + c0; d0 = w3 * s + d0;
            }

            if (rows <= 1)
            {
                continue;
            }

            if ((v = Unsafe.Add(ref xk, k)) != 0f)
            {
                var s = new Vector<float>(v);
                a1 = w0 * s + a1; b1 = w1 * s + b1; c1 = w2 * s + c1; d1 = w3 * s + d1;
            }
        }

        { a0.StoreUnsafe(ref t0); b0.StoreUnsafe(ref t0, w); c0.StoreUnsafe(ref t0, 2 * w); d0.StoreUnsafe(ref t0, 3 * w); }
        if (rows > 1) { a1.StoreUnsafe(ref t1); b1.StoreUnsafe(ref t1, w); c1.StoreUnsafe(ref t1, 2 * w); d1.StoreUnsafe(ref t1, 3 * w); }
    }

    private static void Int4FewRowsTiled(float[] q, float[] scales, float[] xv, float[] yv, int m, int n, int k, int blockSize, int blocks) =>
        For(blocks, (long)m * n * k, new Int4Blocks(q, scales, xv, yv, m, n, k, blockSize));

    // A range of column blocks of the 4-bit product (a struct: no closure a call).
    private readonly struct Int4Blocks(float[] q, float[] scales, float[] xv, float[] yv, int m, int n, int k, int blockSize) : IRangeBody
    {
        public void Run(int first, int last)
        {
            var sums = ArrayPool<float>.Shared.Rent(m * blockSize);
            try
            {
                for (int block = first; block < last; block++)
                {
                    Int4Block(q, scales, xv, yv, sums, m, n, k, blockSize, block);
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(sums);
            }
        }
    }

    // One column block of the 4-bit product: panels two float vectors wide, up to the register rows a pass, and k chunks
    // that stay within one group of 32 weight rows (one row of scales).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Int4Block(float[] q, float[] scales, float[] xv, float[] yv, float[] sums, int m, int n, int k, int blockSize, int block)
    {
        int w = Vector<float>.Count, panel = 2 * w;
        int rowsPerPass = CpuTuning.RegisterRows >= 8 ? 8 : 4;
        int chunk = CpuTuning.KChunk;
        int words = (n + 7) / 8;
        int j0 = block * blockSize, width = Math.Min(blockSize, n - j0), whole = width / panel * panel;
        ref uint packed = ref Unsafe.As<float, uint>(ref MemoryMarshal.GetArrayDataReference(q));
        ref float acc = ref MemoryMarshal.GetArrayDataReference(sums);
        sums.AsSpan(0, m * blockSize).Clear();
        for (int k0 = 0, k1; k0 < k; k0 = k1)
        {
            k1 = Math.Min(Math.Min(k, k0 + chunk), (k0 / 32 + 1) * 32);
            ref float groupScales = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(scales), (nint)(k0 / 32) * words * 8);
            int j = 0;
            if (m == 1)
            {
                // One row (decoding): two panels side by side, the work of each weight row shared by four vectors.
                for (; j + 2 * panel <= whole; j += 2 * panel)
                {
                    Int4Panel1Wide(ref packed, words, ref groupScales, xv, ref Unsafe.Add(ref acc, j), j0 + j, k0, k1);
                }
            }

            for (; j < whole; j += panel)
            {
                for (int r0 = 0; r0 < m; r0 += rowsPerPass)
                {
                    ref float target = ref Unsafe.Add(ref acc, r0 * blockSize + j);
                    if (rowsPerPass == 8)
                    {
                        Int4Panel8(ref packed, words, ref groupScales, xv, ref target, blockSize, k, r0, Math.Min(8, m - r0), j0 + j, k0, k1);
                    }
                    else
                    {
                        Int4Panel4(ref packed, words, ref groupScales, xv, ref target, blockSize, k, r0, Math.Min(4, m - r0), j0 + j, k0, k1);
                    }
                }
            }
        }

        if (whole < width)
        {
            for (int kk = 0; kk < k; kk++)
            {
                ref uint row = ref Unsafe.Add(ref packed, (nint)kk * words);
                ref float s = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(scales), (nint)(kk / 32) * words * 8);
                for (int j = whole; j < width; j++)
                {
                    int col = j0 + j;
                    int nibble = (int)(Unsafe.Add(ref row, col >> 3) << (28 - 4 * (col & 7))) >> 28;
                    float value = nibble * Unsafe.Add(ref s, col);
                    for (int r = 0; r < m; r++)
                    {
                        float xk = xv[r * k + kk];
                        if (xk != 0f)
                        {
                            sums[r * blockSize + j] += xk * value;
                        }
                    }
                }
            }
        }

        for (int r = 0; r < m; r++)
        {
            sums.AsSpan(r * blockSize, width).CopyTo(yv.AsSpan(r * n + j0, width));
        }
    }

    // The two float vectors of 4-bit weights of one panel row (panel column `col` a multiple of 8): each lane takes its
    // column's word, the multiply moves the nibble to the top, the arithmetic shift brings it back signed, and the float
    // times the column's scale is the weight (Int4Row's value).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Int4Weights(ref uint row, Vector<int> first, Vector<int> second, Vector<int> lowWord, Vector<float> s0, Vector<float> s1,
        out Vector<float> w0, out Vector<float> w1)
    {
        Vector<int> l0, l1;
        if (Vector<int>.Count == 4)
        {
            l0 = l1 = new Vector<int>((int)row);                          // one word: columns 0-3, then 4-7
        }
        else if (Vector<int>.Count == 8)
        {
            l0 = new Vector<int>((int)row);
            l1 = new Vector<int>((int)Unsafe.Add(ref row, 1));
        }
        else
        {
            l0 = Vector.ConditionalSelect(lowWord, new Vector<int>((int)row), new Vector<int>((int)Unsafe.Add(ref row, 1)));
            l1 = Vector.ConditionalSelect(lowWord, new Vector<int>((int)Unsafe.Add(ref row, 2)), new Vector<int>((int)Unsafe.Add(ref row, 3)));
        }

        w0 = Vector.ConvertToSingle(Vector.ShiftRightArithmetic(ShiftLanes(l0, first), 28)) * s0;
        w1 = Vector.ConvertToSingle(Vector.ShiftRightArithmetic(ShiftLanes(l1, second), 28)) * s1;
    }

    // Whether ShiftLanes shifts each lane by its own count (one instruction) instead of multiplying by 2^count: where the
    // runtime reports the intrinsic and the vector paths allow it (ComputeResources.CpuVectors); the same bits either way.
    private static bool LaneShifts => CpuTuning.IntrinsicsAllowed &&
        ((Vector<int>.Count == 8 && Avx2.IsSupported) || (Vector<int>.Count == 16 && Avx512F.IsSupported) || (Vector<int>.Count == 4 && AdvSimd.IsSupported));

    /// <summary>Whether the int4 kernels shift each lane by its own count (listings).</summary>
    internal static bool Int4LaneShifts => Int4Lanes.Shifts;

    // lanes << each lane's count (operand: Int4Lanes.First or Second): the CPU's shift per lane where it has one, else
    // lanes · 2^count, the same bits.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<int> ShiftLanes(Vector<int> lanes, Vector<int> operand)
    {
        if (Int4Lanes.Shifts)
        {
            if (Vector<int>.Count == 8 && Avx2.IsSupported)
            {
                return Avx2.ShiftLeftLogicalVariable(lanes.AsVector256(), operand.AsVector256().AsUInt32()).AsVector();
            }

            if (Vector<int>.Count == 16 && Avx512F.IsSupported)
            {
                return Avx512F.ShiftLeftLogicalVariable(lanes.AsVector512(), operand.AsVector512().AsUInt32()).AsVector();
            }

            if (Vector<int>.Count == 4 && AdvSimd.IsSupported)
            {
                return AdvSimd.ShiftLogical(lanes.AsVector128(), operand.AsVector128()).AsVector();
            }
        }

        return lanes * operand;
    }

    // One row × 4 Vector<float> columns (two panels): x[kk] · each panel's weights, in the panels' own order.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Int4Panel1Wide(ref uint packed, int words, ref float scales, float[] x, ref float acc, int col, int k0, int k1)
    {
        nuint w = (nuint)Vector<float>.Count;
        Vector<int> first = Int4Lanes.First, second = Int4Lanes.Second, lowWord = Int4Lanes.LowWord;
        var s0 = Vector.LoadUnsafe(ref scales, (nuint)col);
        var s1 = Vector.LoadUnsafe(ref scales, (nuint)col + w);
        var s2 = Vector.LoadUnsafe(ref scales, (nuint)col + 2 * w);
        var s3 = Vector.LoadUnsafe(ref scales, (nuint)col + 3 * w);
        var a0 = Vector.LoadUnsafe(ref acc);
        var b0 = Vector.LoadUnsafe(ref acc, w);
        var c0 = Vector.LoadUnsafe(ref acc, 2 * w);
        var d0 = Vector.LoadUnsafe(ref acc, 3 * w);
        int panelWords = (int)(2 * w / 8);                                // words of one two-vector panel
        ref float rx = ref MemoryMarshal.GetArrayDataReference(x);
        ref uint start = ref Unsafe.Add(ref packed, col >> 3);
        for (int kk = k0; kk < k1; kk++)
        {
            float v = Unsafe.Add(ref rx, kk);
            if (v == 0f)
            {
                continue;
            }

            ref uint row = ref Unsafe.Add(ref start, (nint)kk * words);
            Int4Weights(ref row, first, second, lowWord, s0, s1, out var w0, out var w1);
            Int4Weights(ref Unsafe.Add(ref row, panelWords), first, second, lowWord, s2, s3, out var w2, out var w3);
            var s = new Vector<float>(v);
            a0 = w0 * s + a0;
            b0 = w1 * s + b0;
            c0 = w2 * s + c0;
            d0 = w3 * s + d0;
        }

        a0.StoreUnsafe(ref acc);
        b0.StoreUnsafe(ref acc, w);
        c0.StoreUnsafe(ref acc, 2 * w);
        d0.StoreUnsafe(ref acc, 3 * w);
    }

    // Up to 8 rows × 2 Vector<float> columns in 16 registers (instruction sets with 32 vector registers).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Int4Panel8(ref uint packed, int words, ref float scales, float[] x, ref float acc, int accStride, int k, int r0, int rows, int col, int k0, int k1)
    {
        nuint w = (nuint)Vector<float>.Count;
        Vector<int> first = Int4Lanes.First, second = Int4Lanes.Second, lowWord = Int4Lanes.LowWord;
        var s0 = Vector.LoadUnsafe(ref scales, (nuint)col);
        var s1 = Vector.LoadUnsafe(ref scales, (nuint)col + w);
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
        ref uint panelWords = ref Unsafe.Add(ref packed, col >> 3);
        for (int kk = k0; kk < k1; kk++)
        {
            Int4Weights(ref Unsafe.Add(ref panelWords, (nint)kk * words), first, second, lowWord, s0, s1, out var w0, out var w1);
            ref float xk = ref Unsafe.Add(ref rx, kk);
            float v = xk;
            if (v != 0f)
            {
                var s = new Vector<float>(v);
                a0 = w0 * s + a0;
                b0 = w1 * s + b0;
            }

            if (rows <= 1)
            {
                continue;
            }

            if ((v = Unsafe.Add(ref xk, 1 * k)) != 0f)
            {
                var s = new Vector<float>(v);
                a1 = w0 * s + a1;
                b1 = w1 * s + b1;
            }

            if (rows <= 2)
            {
                continue;
            }

            if ((v = Unsafe.Add(ref xk, 2 * k)) != 0f)
            {
                var s = new Vector<float>(v);
                a2 = w0 * s + a2;
                b2 = w1 * s + b2;
            }

            if (rows <= 3)
            {
                continue;
            }

            if ((v = Unsafe.Add(ref xk, 3 * k)) != 0f)
            {
                var s = new Vector<float>(v);
                a3 = w0 * s + a3;
                b3 = w1 * s + b3;
            }

            if (rows <= 4)
            {
                continue;
            }

            if ((v = Unsafe.Add(ref xk, 4 * k)) != 0f)
            {
                var s = new Vector<float>(v);
                a4 = w0 * s + a4;
                b4 = w1 * s + b4;
            }

            if (rows <= 5)
            {
                continue;
            }

            if ((v = Unsafe.Add(ref xk, 5 * k)) != 0f)
            {
                var s = new Vector<float>(v);
                a5 = w0 * s + a5;
                b5 = w1 * s + b5;
            }

            if (rows <= 6)
            {
                continue;
            }

            if ((v = Unsafe.Add(ref xk, 6 * k)) != 0f)
            {
                var s = new Vector<float>(v);
                a6 = w0 * s + a6;
                b6 = w1 * s + b6;
            }

            if (rows <= 7)
            {
                continue;
            }

            if ((v = Unsafe.Add(ref xk, 7 * k)) != 0f)
            {
                var s = new Vector<float>(v);
                a7 = w0 * s + a7;
                b7 = w1 * s + b7;
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
    private static void Int4Panel4(ref uint packed, int words, ref float scales, float[] x, ref float acc, int accStride, int k, int r0, int rows, int col, int k0, int k1)
    {
        nuint w = (nuint)Vector<float>.Count;
        Vector<int> first = Int4Lanes.First, second = Int4Lanes.Second, lowWord = Int4Lanes.LowWord;
        var s0 = Vector.LoadUnsafe(ref scales, (nuint)col);
        var s1 = Vector.LoadUnsafe(ref scales, (nuint)col + w);
        Vector<float> a0 = default, b0 = default, a1 = default, b1 = default, a2 = default, b2 = default, a3 = default, b3 = default;
        { a0 = Vector.LoadUnsafe(ref acc, (nuint)(0 * accStride)); b0 = Vector.LoadUnsafe(ref acc, (nuint)(0 * accStride) + w); }
        if (rows > 1) { a1 = Vector.LoadUnsafe(ref acc, (nuint)(1 * accStride)); b1 = Vector.LoadUnsafe(ref acc, (nuint)(1 * accStride) + w); }
        if (rows > 2) { a2 = Vector.LoadUnsafe(ref acc, (nuint)(2 * accStride)); b2 = Vector.LoadUnsafe(ref acc, (nuint)(2 * accStride) + w); }
        if (rows > 3) { a3 = Vector.LoadUnsafe(ref acc, (nuint)(3 * accStride)); b3 = Vector.LoadUnsafe(ref acc, (nuint)(3 * accStride) + w); }

        ref float rx = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(x), (nint)r0 * k);
        ref uint panelWords = ref Unsafe.Add(ref packed, col >> 3);
        for (int kk = k0; kk < k1; kk++)
        {
            Int4Weights(ref Unsafe.Add(ref panelWords, (nint)kk * words), first, second, lowWord, s0, s1, out var w0, out var w1);
            ref float xk = ref Unsafe.Add(ref rx, kk);
            float v = xk;
            if (v != 0f)
            {
                var s = new Vector<float>(v);
                a0 = w0 * s + a0;
                b0 = w1 * s + b0;
            }

            if (rows <= 1)
            {
                continue;
            }

            if ((v = Unsafe.Add(ref xk, 1 * k)) != 0f)
            {
                var s = new Vector<float>(v);
                a1 = w0 * s + a1;
                b1 = w1 * s + b1;
            }

            if (rows <= 2)
            {
                continue;
            }

            if ((v = Unsafe.Add(ref xk, 2 * k)) != 0f)
            {
                var s = new Vector<float>(v);
                a2 = w0 * s + a2;
                b2 = w1 * s + b2;
            }

            if (rows <= 3)
            {
                continue;
            }

            if ((v = Unsafe.Add(ref xk, 3 * k)) != 0f)
            {
                var s = new Vector<float>(v);
                a3 = w0 * s + a3;
                b3 = w1 * s + b3;
            }
        }

        { a0.StoreUnsafe(ref acc, (nuint)(0 * accStride)); b0.StoreUnsafe(ref acc, (nuint)(0 * accStride) + w); }
        if (rows > 1) { a1.StoreUnsafe(ref acc, (nuint)(1 * accStride)); b1.StoreUnsafe(ref acc, (nuint)(1 * accStride) + w); }
        if (rows > 2) { a2.StoreUnsafe(ref acc, (nuint)(2 * accStride)); b2.StoreUnsafe(ref acc, (nuint)(2 * accStride) + w); }
        if (rows > 3) { a3.StoreUnsafe(ref acc, (nuint)(3 * accStride)); b3.StoreUnsafe(ref acc, (nuint)(3 * accStride) + w); }
    }
}
