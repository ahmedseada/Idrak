// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Hip;

// The float32 products (gemm_f32, gemv_f32) and the packed weight products and expansions (packed_gemv_f32 and the
// dequantizing kernels). Each falls back to the host when the kernels are unavailable or a shape passes the grid limits
// the device reports.
internal sealed unsafe partial class HipBackend
{
    public override void BatchedMatMulKernel(Storage a, Storage b, Storage c, int batch, int m, int n, int k, bool transA, bool transB, float beta)
    {
        if (!Product(a, b, c, null, batch, m, n, k, transA, transB, beta))
        {
            base.BatchedMatMulKernel(a, b, c, batch, m, n, k, transA, transB, beta);
        }
    }

    // c = a·b + bias in one pass: the product's own epilogue adds the bias (the sums of MatMul then AddRowVector).
    public override bool MatMulBiasKernel(Storage a, Storage b, Storage bias, Storage c, int m, int n, int k) =>
        Product(a, b, c, bias, 1, m, n, k, false, false, 0f);

    // c = op(a) · op(b) (+ beta · c) (+ bias) for batch contiguous triples. Few rows through a large matrix (token-by-token
    // decoding) go through gemv_f32, which reads the matrix once with every compute unit busy; the rest through the tiled
    // gemm_f32. The sums of both follow the sizes alone (k order in the tiled product, slices of k in a fixed order in the
    // few-row one), never the grid. Batches beyond the grid's y limit run in several launches, as do the row tiles of a very
    // tall product (in blocks of rows of a as stored). False: no kernels, or a shape the grid cannot hold.
    private bool Product(Storage a, Storage b, Storage c, Storage? bias, int batch, int m, int n, int k, bool transA, bool transB, float beta)
    {
        if (Kernels() is null)
        {
            return false;
        }

        if (batch <= 0 || m <= 0 || n <= 0)
        {
            return true;
        }

        var limits = _device.Limits;
        ulong mk = (ulong)m * (ulong)k * sizeof(float), kn = (ulong)k * (ulong)n * sizeof(float), mn = (ulong)m * (ulong)n * sizeof(float);
        ulong pa = P(a), pb = P(b), pc = P(c), pbias = bias is null ? 0UL : P(bias);
        if (m <= HipKernels.GemvRows && !transA && (long)n * k >= 1 << 16)
        {
            int lanes = HipKernels.Lanes(_blockSize), groups = _blockSize / lanes;
            long columnBlocks = transB ? (n + groups - 1) / groups : (n + lanes - 1) / lanes;
            if (columnBlocks <= limits.MaxGridDimX)
            {
                for (int first = 0; first < batch; first += limits.MaxGridDimY)
                {
                    int count = Math.Min(limits.MaxGridDimY, batch - first);
                    Grid("gemv_f32", columnBlocks, count, pa + (ulong)first * mk, pb + (ulong)first * kn, pc + (ulong)first * mn, pbias,
                        U(m), U(n), U(k), U(transB ? 1 : 0), F(beta));
                }

                return true;
            }
        }

        int tile = HipKernels.Tile(_blockSize);
        long columnTiles = (n + tile - 1) / tile, rowTiles = (m + tile - 1) / tile;
        if (columnTiles > limits.MaxGridDimX)
        {
            return false;
        }

        if (rowTiles <= limits.MaxGridDimY)
        {
            int perLaunch = (int)Math.Max(1, limits.MaxGridDimY / rowTiles);
            for (int first = 0; first < batch; first += perLaunch)
            {
                int count = Math.Min(perLaunch, batch - first);
                Grid("gemm_f32", columnTiles, rowTiles * count, pa + (ulong)first * mk, pb + (ulong)first * kn, pc + (ulong)first * mn, pbias,
                    U(m), U(n), U(k), U(transA ? 1 : 0), U(transB ? 1 : 0), F(beta), U((int)rowTiles));
            }

            return true;
        }

        if (transA)
        {
            return false;                                                    // a transposed a has no contiguous blocks of rows
        }

        long rowsPerLaunch = (long)limits.MaxGridDimY * tile;
        for (int i = 0; i < batch; i++)
        {
            for (long row = 0; row < m; row += rowsPerLaunch)
            {
                int rows = (int)Math.Min(rowsPerLaunch, m - row);
                int tiles = (rows + tile - 1) / tile;
                ulong aAt = pa + (ulong)i * mk + (ulong)row * (ulong)k * sizeof(float), cAt = pc + (ulong)i * mn + (ulong)row * (ulong)n * sizeof(float);
                Grid("gemm_f32", columnTiles, tiles, aAt, pb + (ulong)i * kn, cAt, pbias, U(rows), U(n), U(k), U(0), U(transB ? 1 : 0), F(beta), U(tiles));
            }
        }

        return true;
    }

    public override void BFloat16MatMulKernel(Storage x, Storage packed, Storage y, int m, int n, int k)
    {
        if (!PackedProduct(0, x, packed, packed, y, m, n, k))
        {
            base.BFloat16MatMulKernel(x, packed, y, m, n, k);
        }
    }

    public override void Int4MatMulKernel(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k)
    {
        if (!PackedProduct(1, x, q, scales, y, m, n, k))
        {
            base.Int4MatMulKernel(x, q, scales, y, m, n, k);
        }
    }

    // y = x · w with packed weights (format 0 bfloat16, 1 4-bit): few rows read the packed words in place
    // (packed_gemv_f32), more rows expand the weights once into a temporary and run the float32 product (as the CUDA
    // backend does). False: no kernels, or sizes the grid or one storage cannot hold.
    private bool PackedProduct(int format, Storage x, Storage q, Storage scales, Storage y, int m, int n, int k)
    {
        if (Kernels() is null)
        {
            return false;
        }

        if (m <= 0 || n <= 0)
        {
            return true;
        }

        if (m <= HipKernels.GemvRows)
        {
            int lanes = HipKernels.Lanes(_blockSize);
            return Grid("packed_gemv_f32", (n + lanes - 1) / lanes, 1, P(x), P(q), P(scales), P(y), U(m), U(n), U(k), U(format));
        }

        if ((long)k * n > int.MaxValue)
        {
            return false;
        }

        var w = Allocate(k * n, zeroed: false);
        try
        {
            if (format == 0)
            {
                BFloat16DequantizeKernel(q, w, k, n);
            }
            else
            {
                Int4DequantizeKernel(q, scales, w, k, n);
            }

            MatMul(x, w, y, m, n, k, false, false, 0f);
        }
        finally
        {
            w.Release();                                                         // reused in stream order
        }

        return true;
    }

    public override void BFloat16DequantizeKernel(Storage packed, Storage w, int k, int n)
    {
        if (!Elements("bf16_dequant_f32", (long)k * n, P(packed), P(w), U(k), U(n)))
        {
            base.BFloat16DequantizeKernel(packed, w, k, n);
        }
    }

    public override void Float16DequantizeKernel(Storage packed, Storage w, int k, int n)
    {
        if (!Elements("f16_dequant_f32", (long)k * n, P(packed), P(w), U(k), U(n)))
        {
            base.Float16DequantizeKernel(packed, w, k, n);
        }
    }

    public override void Int4DequantizeKernel(Storage q, Storage scales, Storage w, int k, int n)
    {
        if (!Elements("int4_dequant_f32", (long)k * n, P(q), P(scales), P(w), U(k), U(n)))
        {
            base.Int4DequantizeKernel(q, scales, w, k, n);
        }
    }

    public override void Int8DequantizeKernel(Storage q, Storage scales, Storage w, int k, int n)
    {
        if (!Elements("int8_dequant_f32", (long)k * n, P(q), P(scales), P(w), U(k), U(n)))
        {
            base.Int8DequantizeKernel(q, scales, w, k, n);
        }
    }

    public override void PackBFloat16Kernel(Storage x, Storage packed, int n)
    {
        if (!Elements("bf16_pack_f32", ((long)n + 1) / 2, P(x), P(packed), U(n)))
        {
            base.PackBFloat16Kernel(x, packed, n);
        }
    }
}
