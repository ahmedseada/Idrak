// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Backends.Vulkan;

// Matrix products: the float32 product tiled through workgroup memory, and the packed weight products (int8, int4,
// bfloat16) with their expansions to float.
internal static partial class VulkanKernels
{
    private static IEnumerable<(string, Func<SpirvKernel>)> MatMulKernels()
    {
        yield return ("batched_matmul", BatchedMatMul);

        // y[m, n] = x[m, k] · (q[k, n] · scales[n]): q signed bytes, four per word along each row of ⌈n / 4⌉ words.
        // One invocation per output (column fastest, so neighbors read neighboring bytes). TODO(tuning): split k across a
        // workgroup and keep x in workgroup memory for the few-row (decoding) shapes.
        yield return ("int8_matmul", () => PackedMatMul("int8_matmul", scaled: true, (k, q, scales, n, kk, j) =>
            Int8At(k, q, kk * ((n + 3) / 4), j)));

        // w[r, j] = q[r, j] · scales[r / 32, j]: signed nibbles, eight per word along each row of ⌈n / 8⌉ words; the scales
        // have one row of 8·⌈n / 8⌉ values per 32 weight rows.
        yield return ("int4_matmul", () => PackedMatMul("int4_matmul", scaled: true, (k, q, scales, n, kk, j) =>
        {
            var words = (n + 7) / 8;
            return Int4At(k, q, kk * words, j) * scales![kk / 32 * (words * 8) + j];
        }));

        // bfloat16 pairs along each row of ⌈n / 2⌉ words, column 2c in the low half of word c, 2c + 1 in the high half.
        yield return ("bf16_matmul", () => PackedMatMul("bf16_matmul", scaled: false, (k, q, scales, n, kk, j) =>
            BFloat16At(k, q, kk * ((n + 1) / 2) + (j >> 1), j)));

        yield return ("int8_dequantize", () =>
        {
            var k = new KernelBuilder("int8_dequantize", Block);
            var (q, scales, w) = (k.Buffer("q"), k.Buffer("scales"), k.Buffer("w"));
            var (rows, n) = (k.PushInt("k"), k.PushInt("n"));
            Grid(k, rows * n, i =>
            {
                var (r, j) = (i / n, i % n);
                w[i] = Int8At(k, q, r * ((n + 3) / 4), j) * scales[j];
            });
            return k.Build();
        });

        yield return ("int4_dequantize", () =>
        {
            var k = new KernelBuilder("int4_dequantize", Block);
            var (q, scales, w) = (k.Buffer("q"), k.Buffer("scales"), k.Buffer("w"));
            var (rows, n) = (k.PushInt("k"), k.PushInt("n"));
            Grid(k, rows * n, i =>
            {
                var (r, j) = (i / n, i % n);
                var words = (n + 7) / 8;
                w[i] = Int4At(k, q, r * words, j) * scales[r / 32 * (words * 8) + j];
            });
            return k.Build();
        });

        yield return ("bf16_dequantize", () =>
        {
            var k = new KernelBuilder("bf16_dequantize", Block);
            var (packed, w) = (k.Buffer("packed"), k.Buffer("w"));
            var (rows, n) = (k.PushInt("k"), k.PushInt("n"));
            Grid(k, rows * n, i =>
            {
                var (r, j) = (i / n, i % n);
                w[i] = BFloat16At(k, packed, r * ((n + 1) / 2) + (j >> 1), j);
            });
            return k.Build();
        });

        // Word i of packed = x[2i], x[2i + 1] rounded to bfloat16 (to nearest, ties to even; low half first; 0 past n).
        yield return ("bf16_pack", () =>
        {
            var k = new KernelBuilder("bf16_pack", Block);
            var (x, packed) = (k.Buffer("x"), k.Buffer("packed"));
            var n = k.PushInt("n");
            Grid(k, (n + 1) / 2, i =>
            {
                var low = RoundBFloat16(k, x[2 * i]);
                var high = k.Local(k.UInt(0));
                k.If(2 * i + 1 < n, () => high.V = RoundBFloat16(k, x[2 * i + 1]));
                packed[i] = (low | high.V << 16).AsFloat();
            });
            return k.Build();
        });
    }

    // c = op(a) · op(b) + beta · c for batch contiguous triples; a [m, k] (transA: stored [k, m]), b [k, n] (transB: [n, k]).
    // Each 16 × 16 workgroup computes one 16 × 16 block of c, staging 16 × 16 tiles of a and b in workgroup memory.
    // beta = 0 writes the product without reading c (whatever c held). TODO(tuning): several outputs per invocation.
    private static SpirvKernel BatchedMatMul()
    {
        var k = new KernelBuilder("batched_matmul", Tile, Tile);
        var (a, b, c) = (k.Buffer("a"), k.Buffer("b"), k.Buffer("c"));
        var (batch, m, n, depth) = (k.PushInt("batch"), k.PushInt("m"), k.PushInt("n"), k.PushInt("k"));
        var (transA, transB, beta) = (k.PushInt("transA"), k.PushInt("transB"), k.PushFloat("beta"));
        var tileA = k.Shared("tileA", Tile * Tile);
        var tileB = k.Shared("tileB", Tile * Tile);
        var (tx, ty) = (k.LocalX, k.LocalY);
        var row = k.GroupY * Tile + ty;
        var col = k.GroupX * Tile + tx;
        var at = ty * Tile + tx;
        k.For(k.GroupZ, batch, bi =>
        {
            var aBase = bi * m * depth;
            var bBase = bi * depth * n;
            var acc = k.Local(0f);
            k.For(k.Int(0), depth, Tile, t =>
            {
                // tileA[ty, tx] = op(a)[row, t + tx]; tileB[ty, tx] = op(b)[t + ty, col]; zeros outside.
                var ka = t + tx;
                var kb = t + ty;
                tileA[at] = k.Float(0f);
                k.If((row < m) & (ka < depth), () => tileA[at] = a[aBase + k.Select(transA.Ne(0), ka * m + row, row * depth + ka)]);
                tileB[at] = k.Float(0f);
                k.If((kb < depth) & (col < n), () => tileB[at] = b[bBase + k.Select(transB.Ne(0), col * depth + kb, kb * n + col)]);
                k.Barrier();
                var sum = acc.V;
                for (int e = 0; e < Tile; e++)
                {
                    sum = k.Fma(tileA[ty * Tile + e], tileB[k.Int(e * Tile) + tx], sum);
                }

                acc.V = sum;
                k.Barrier();
            });
            k.If((row < m) & (col < n), () =>
            {
                var o = bi * m * n + row * n + col;
                k.If(beta.Eq(0), () => c[o] = acc.V, () => c[o] = acc.V + beta * c[o]);
            });
        }, k.GroupsZ);
        return k.Build();
    }

    // y[r, j] = Σ_kk x[r, kk] · w(kk, j) (· scales[j] after the sum when scaled), one invocation per output.
    private static SpirvKernel PackedMatMul(string name, bool scaled, Func<KernelBuilder, Buf, Buf?, Val, Val, Val, Val> weight)
    {
        var k = new KernelBuilder(name, Block);
        var x = k.Buffer("x");
        var q = k.Buffer(name.StartsWith("bf16", StringComparison.Ordinal) ? "packed" : "q");
        var scales = scaled ? k.Buffer("scales") : null;
        var y = k.Buffer("y");
        var (m, n, depth) = (k.PushInt("m"), k.PushInt("n"), k.PushInt("k"));
        bool perColumn = name == "int8_matmul";                              // int8 scales the sum; int4 scales each weight
        Grid(k, m * n, i =>
        {
            var (r, j) = (i / n, i % n);
            var acc = k.Local(0f);
            var xo = r * depth;
            k.For(k.Int(0), depth, 1, kk => acc.V = k.Fma(x[xo + kk], weight(k, q, scales, n, kk, j), acc.V));
            y[i] = perColumn ? acc.V * scales![j] : acc.V;
        });
        return k.Build();
    }

    // Signed byte j of the row of words starting at word rowStart.
    private static Val Int8At(KernelBuilder k, Buf q, Val rowStart, Val j)
    {
        var word = q.Int(rowStart + (j >> 2));
        return word.ShiftLeft(24 - 8 * (j & 3)).ShiftRight(k.Int(24)).ToFloat();
    }

    // Signed nibble j of the row of words starting at word rowStart.
    private static Val Int4At(KernelBuilder k, Buf q, Val rowStart, Val j)
    {
        var word = q.Int(rowStart + (j >> 3));
        return word.ShiftLeft(28 - 4 * (j & 7)).ShiftRight(k.Int(28)).ToFloat();
    }

    // The bfloat16 value in word wordIndex: its high half for odd j, its low half for even j.
    private static Val BFloat16At(KernelBuilder k, Buf packed, Val wordIndex, Val j)
    {
        var word = packed.UInt(wordIndex);
        return k.Select((j & 1).Eq(1), word & k.UInt(0xFFFF0000u), word << 16).AsFloat();
    }

    // The bfloat16 bits of v (as uint in the low half): to nearest, ties to even; NaN → 0x7FC0.
    private static Val RoundBFloat16(KernelBuilder k, Val v)
    {
        var u = v.AsUInt();
        var rounded = (u + k.UInt(0x7FFFu) + ((u >> 16) & 1)) >> 16;
        return k.Select(v.IsNan(), k.UInt(0x7FC0u), rounded);
    }
}
