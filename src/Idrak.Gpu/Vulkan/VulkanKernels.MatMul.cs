// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// Matrix products: the float32 product tiled through workgroup memory.
internal static partial class VulkanKernels
{
    private static IEnumerable<(string, Func<SpirvKernel>)> MatMulKernels()
    {
        // The tiled float32 products: register-blocked (4 × 4 outputs per invocation) and one output per invocation, for
        // workgroups of side × side invocations (side = MatSide(width)); the backend measures which (or the small kernel)
        // is fastest per shape.
        yield return ("batched_matmul", BatchedMatMul);
        yield return ("batched_matmul_tile", BatchedMatMulTile);

        // batched_matmul for small products (attention heads, few rows): one invocation per output, no workgroup memory or
        // barriers; any number of groups (grid stride over batch · m · n).
        yield return ("batched_matmul_small", () =>
        {
            var k = new KernelBuilder("batched_matmul_small", Block);
            var (a, b, c) = (k.Buffer("a"), k.Buffer("b"), k.Buffer("c"));
            var (batch, m, n, depth) = (k.PushInt("batch"), k.PushInt("m"), k.PushInt("n"), k.PushInt("k"));
            var (transA, transB, beta) = (k.PushInt("transA"), k.PushInt("transB"), k.PushFloat("beta"));
            var (ta, tb) = (transA.Ne(0), transB.Ne(0));
            Grid(k, batch * m * n, o =>
            {
                var (bi, rest) = (o / (m * n), o % (m * n));
                var (row, col) = (rest / n, rest % n);
                var aBase = bi * m * depth;
                var bBase = bi * depth * n;
                // op(a)[row, kk] = a[aBase + aStart + kk · aStep]; op(b)[kk, col] = b[bBase + bStart + kk · bStep].
                var (aStart, aStep) = (k.Select(ta, row, row * depth), k.Select(ta, m, k.Int(1)));
                var (bStart, bStep) = (k.Select(tb, col * depth, col), k.Select(tb, k.Int(1), n));
                var acc = k.Local(0f);
                k.For(k.Int(0), depth, 1, kk => acc.V = k.Fma(a[aBase + aStart + kk * aStep], b[bBase + bStart + kk * bStep], acc.V));
                k.If(beta.Eq(0), () => c[o] = acc.V, () => c[o] = k.Fma(beta, c[o], acc.V));
            });
            return k.Build();
        });
    }

    /// <summary>
    /// Side of the square workgroup of the tiled products for a width: the largest power of two whose square is at most
    /// the width (16 for 256 invocations, 8 for 64 or 128).
    /// </summary>
    public static int MatSide(int width)
    {
        int side = 1;
        while ((side * 2) * (side * 2) <= width)
        {
            side *= 2;
        }

        return side;
    }

    /// <summary>Outputs along each side an invocation of <c>batched_matmul</c> keeps (4 × 4): its block of c is 4 · side.</summary>
    public const int MatPer = 4;

    // Rows of k staged per step of batched_matmul.
    private const int MatDepth = 16;

    // c = op(a) · op(b) + beta · c for batch contiguous triples; a [m, k] (transA: stored [k, m]), b [k, n] (transB: [n, k]).
    // A workgroup of S × S invocations (S = MatSide(width), 16 for 256) computes one T × T block of c, T = 4S:
    // invocation (tx, ty) = (i % S, i / S) keeps the 4 × 4 outputs (ty + S r, tx + S s) in registers. Per step of 16
    // along k, the workgroup stages op(a)[T × 16] and op(b)[16 × T] in workgroup memory (each invocation loading 64 / S
    // elements of each, consecutive invocations reading consecutive addresses of a and b, transposed or not; zeros
    // outside), then each invocation reads 4 + 4 staged values per k for 16 multiply-adds. Every output adds its k terms
    // in order. beta = 0 writes the product without reading c (whatever c held).
    //
    // Workgroup memory: tileA[kk · (T + 1) + r] and tileB[kk · (T + 1) + col] (rows padded by one: the stores along k
    // and the reads along m or n fall in different banks) are written only between the barrier ending the previous
    // step's reads and the barrier before this step's reads, each element by exactly one invocation.
    private static SpirvKernel BatchedMatMul()
    {
        int side = MatSide(Block), threads = side * side, Per = MatPer, tileEdge = Per * side, stride = tileEdge + 1;
        int sideShift = System.Numerics.BitOperations.Log2((uint)side), edgeShift = System.Numerics.BitOperations.Log2((uint)tileEdge);
        var k = new KernelBuilder("batched_matmul", threads);
        var (a, b, c) = (k.Buffer("a"), k.Buffer("b"), k.Buffer("c"));
        var (batch, m, n, depth) = (k.PushInt("batch"), k.PushInt("m"), k.PushInt("n"), k.PushInt("k"));
        var (transA, transB, beta) = (k.PushInt("transA"), k.PushInt("transB"), k.PushFloat("beta"));
        var tileA = k.Shared("tileA", MatDepth * stride);
        var tileB = k.Shared("tileB", MatDepth * stride);
        var tid = k.LocalX;
        var (tx, ty) = (tid & (side - 1), tid >> sideShift);
        var (ta, tb) = (transA.Ne(0), transB.Ne(0));
        var rowBase = k.GroupY * tileEdge;
        var colBase = k.GroupX * tileEdge;
        var acc = new Var[Per * Per];
        for (int i = 0; i < acc.Length; i++)
        {
            acc[i] = k.Local(ScalarKind.Float);
        }

        // Where invocation tid's p-th element of each staged tile comes from: (r, kk) of op(a), (kk, col) of op(b).
        // Untransposed a and transposed b are read along k (16 consecutive invocations per row), the others along m or n.
        var loads = new (Val R, Val KA, Val Col, Val KB)[tileEdge * MatDepth / threads];
        for (int p = 0; p < loads.Length; p++)
        {
            var l = tid + p * threads;
            loads[p] = (k.Select(ta, l & (tileEdge - 1), l >> 4), k.Select(ta, l >> edgeShift, l & (MatDepth - 1)),
                k.Select(tb, l >> 4, l & (tileEdge - 1)), k.Select(tb, l & (MatDepth - 1), l >> edgeShift));
        }

        k.For(k.GroupZ, batch, bi =>
        {
            var aBase = bi * m * depth;
            var bBase = bi * depth * n;
            foreach (var v in acc)
            {
                v.V = k.Float(0f);
            }

            k.For(k.Int(0), depth, MatDepth, t =>
            {
                foreach (var (r, ka, col, kb) in loads)
                {
                    var (row, kka) = (rowBase + r, t + ka);
                    var va = k.Local(0f);
                    k.If((row < m) & (kka < depth), () => va.V = a[aBase + k.Select(ta, kka * m + row, row * depth + kka)]);
                    tileA[ka * stride + r] = va.V;
                    var (column, kkb) = (colBase + col, t + kb);
                    var vb = k.Local(0f);
                    k.If((kkb < depth) & (column < n), () => vb.V = b[bBase + k.Select(tb, column * depth + kkb, kkb * n + column)]);
                    tileB[kb * stride + col] = vb.V;
                }

                k.Barrier();
                var sums = acc.Select(v => v.V).ToArray();
                for (int e = 0; e < MatDepth; e++)
                {
                    var av = new Val[Per];
                    var bv = new Val[Per];
                    for (int i = 0; i < Per; i++)
                    {
                        av[i] = tileA[k.Int(e * stride + side * i) + ty];
                        bv[i] = tileB[k.Int(e * stride + side * i) + tx];
                    }

                    for (int i = 0; i < Per; i++)
                    {
                        for (int j = 0; j < Per; j++)
                        {
                            sums[i * Per + j] = k.Fma(av[i], bv[j], sums[i * Per + j]);
                        }
                    }
                }

                for (int i = 0; i < acc.Length; i++)
                {
                    acc[i].V = sums[i];
                }

                k.Barrier();
            });

            for (int i = 0; i < Per; i++)
            {
                for (int j = 0; j < Per; j++)
                {
                    var (row, col, value) = (rowBase + ty + side * i, colBase + tx + side * j, acc[i * Per + j]);
                    k.If((row < m) & (col < n), () =>
                    {
                        var o = bi * m * n + row * n + col;
                        k.If(beta.Eq(0), () => c[o] = value.V, () => c[o] = k.Fma(beta, c[o], value.V));
                    });
                }
            }
        }, k.GroupsZ);
        return k.Build();
    }

    // batched_matmul with one output per invocation: each S × S workgroup (S = MatSide(width)) computes one S × S block of
    // c, staging S × S tiles of a and b in workgroup memory (fewer registers, more workgroups: better where the blocks of
    // batched_matmul are too few or too large for the device).
    //
    // Loads are coalesced for every layout (rule 83): consecutive invocations (tx) read consecutive addresses, along k for
    // untransposed a and transposed b, along m or n for the others; the tiles are stored transposed where the load was, so
    // the multiply-adds read them alike. Every output adds its k terms in order, as before.
    //
    // Workgroup memory: tileA[r · (S + 1) + kk] (op(a)'s row r, step column kk) and tileB[kk · (S + 1) + col] (rows padded
    // by one: a transposed store's column falls in different banks), each element written by exactly one invocation,
    // between the barrier ending the previous step's reads and the barrier before this step's reads.
    private static SpirvKernel BatchedMatMulTile()
    {
        int side = MatSide(Block), stride = side + 1;
        var k = new KernelBuilder("batched_matmul_tile", side, side);
        var (a, b, c) = (k.Buffer("a"), k.Buffer("b"), k.Buffer("c"));
        var (batch, m, n, depth) = (k.PushInt("batch"), k.PushInt("m"), k.PushInt("n"), k.PushInt("k"));
        var (transA, transB, beta) = (k.PushInt("transA"), k.PushInt("transB"), k.PushFloat("beta"));
        var tileA = k.Shared("tileA", side * stride);
        var tileB = k.Shared("tileB", side * stride);
        var (tx, ty) = (k.LocalX, k.LocalY);
        var (ta, tb) = (transA.Ne(0), transB.Ne(0));
        var (rowBase, colBase) = (k.GroupY * side, k.GroupX * side);
        var row = rowBase + ty;
        var col = colBase + tx;
        // Invocation (tx, ty) loads op(a)[rowBase + aR, t + aK] and op(b)[t + bK, colBase + bC]: tx runs along the stored rows.
        var (aR, aK) = (k.Select(ta, tx, ty), k.Select(ta, ty, tx));
        var (bK, bC) = (k.Select(tb, tx, ty), k.Select(tb, ty, tx));
        var (aAt, bAt) = (aR * stride + aK, bK * stride + bC);
        k.For(k.GroupZ, batch, bi =>
        {
            var aBase = bi * m * depth;
            var bBase = bi * depth * n;
            var acc = k.Local(0f);
            k.For(k.Int(0), depth, side, t =>
            {
                // tileA[r, kk] = op(a)[rowBase + r, t + kk]; tileB[kk, j] = op(b)[t + kk, colBase + j]; zeros outside.
                var (aRow, ka) = (rowBase + aR, t + aK);
                var va = k.Local(0f);
                k.If((aRow < m) & (ka < depth), () => va.V = a[aBase + k.Select(ta, ka * m + aRow, aRow * depth + ka)]);
                tileA[aAt] = va.V;
                var (kb, bCol) = (t + bK, colBase + bC);
                var vb = k.Local(0f);
                k.If((kb < depth) & (bCol < n), () => vb.V = b[bBase + k.Select(tb, bCol * depth + kb, kb * n + bCol)]);
                tileB[bAt] = vb.V;
                k.Barrier();
                var sum = acc.V;
                for (int e = 0; e < side; e++)
                {
                    sum = k.Fma(tileA[ty * stride + e], tileB[k.Int(e * stride) + tx], sum);
                }

                acc.V = sum;
                k.Barrier();
            });
            k.If((row < m) & (col < n), () =>
            {
                var o = bi * m * n + row * n + col;
                k.If(beta.Eq(0), () => c[o] = acc.V, () => c[o] = k.Fma(beta, c[o], acc.V));
            });
        }, k.GroupsZ);
        return k.Build();
    }

}
