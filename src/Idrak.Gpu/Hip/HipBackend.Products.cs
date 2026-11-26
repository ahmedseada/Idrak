// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Gpu.Hip;

// HIP's matrix products on its own GCN kernels (GcnKernels.Products.cs), where CUDA overrides them and with CUDA's rules:
// the float32 product (every transposition, beta, batches; a bias through MatMulBias), few rows (decoding: up to two
// groups of 8 rows, b not transposed in a, n·k of 65,536 or more, as CUDA's gemv_nn_f32 / gemv_nt_f32) on the few-row
// kernels and the rest on the tiled one; up to three float32 products of one input of few rows in one launch
// (MatMulManyKernel, gemv_multi_f32, as CUDA's). Batches run in grid z up to the device's reported limit (rule 84), each launch
// labelled for --profile as CUDA's ("matmul_nn m x n x k" with 2·m·n·k operations). Unavailable kernels, matrices of 2^30
// elements or more (the kernels' 32-bit byte offsets) and grids past the device's limits have no pass here.
internal sealed unsafe partial class HipBackend
{
    // Row groups of 8 the few-row float32 products take (grid y), as CUDA's FewRowGroups.
    private const int FewRowGroups = 2;

    // The most elements a matrix the products take may have: their byte offsets are 32-bit.
    private const long MostProductElements = 1L << 30;

    /// <summary>
    /// A product kernel's floating-point operations and global-memory bytes for one call, from the table at the top of
    /// GcnKernels.Products.cs (<paramref name="batch"/>: the float32 products'); (0, 0) for a kernel it does not list. The
    /// profile's labels and <c>--bench-hip</c>'s rates read it.
    /// </summary>
    internal static (double Flops, double Bytes) ProductCost(string kernel, long m, long n, long k, int batch = 1, bool beta = false)
    {
        double mnk = 2.0 * m * n * k;
        return kernel switch
        {
            "matmul_f32" => (mnk * batch, 4.0 * batch * (m * k + k * n + m * n * (beta ? 2 : 1))),
            "gemv_nn_f32" or "gemv_nt_f32" => (mnk * batch, 4.0 * batch * (m * k + (m + 7) / 8 * k * n + m * n)),
            "gemv_multi_f32" => (mnk, 4.0 * (m * k + k * n + m * n + n)),                       // n: the products' columns together
            _ => (0, 0),
        };
    }

    // Launches `function` over x × y × z blocks of its declared block, its arguments packed from one 64-bit value each.
    private void LaunchProduct(GcnFunction function, int blocks, int blocksY, int blocksZ, params ReadOnlySpan<ulong> values)
    {
        if (values.Length != function.Arguments.Count)
        {
            throw new InvalidOperationException($"Kernel {function.Name} declares {function.Arguments.Count} arguments but was launched with {values.Length}.");
        }

        Span<byte> buffer = stackalloc byte[MostArgumentBytes];
        var arguments = buffer[..function.ArgumentBytes];
        arguments.Clear();
        GcnKernels.PackArguments(function.Arguments, arguments, values);
        LaunchGcn(function, (uint)blocks, (uint)blocksY, (uint)blocksZ, (uint)function.Block, arguments);
    }

    public override void BatchedMatMulKernel(Storage a, Storage b, Storage c, int batch, int m, int n, int k, bool transA, bool transB, float beta)
    {
        if (m <= 0 || n <= 0 || batch <= 0)
        {
            return;
        }

        if (!FloatProduct(a, b, c, batch, m, n, k, transA, transB, beta, null))
        {
            base.BatchedMatMulKernel(a, b, c, batch, m, n, k, transA, transB, beta);
        }
    }

    public override bool MatMulBiasKernel(Storage a, Storage b, Storage bias, Storage c, int m, int n, int k) =>
        m > 0 && n > 0 && FloatProduct(a, b, c, 1, m, n, k, false, false, 0f, bias);

    // Up to three products of one input of at most 8 rows in one launch (gemv_multi_f32, CUDA's): grid y picks the
    // product, x covers the widest; the host's composed products past the kernel's limits (more products or rows, a matrix
    // of 2^30 elements or more, a grid past the device's), as CUDA's base path.
    public override void MatMulManyKernel(Storage a, int m, int k, ReadOnlySpan<(Storage Weight, Storage? Bias, Storage Output, int Columns)> products)
    {
        int widest = 0;
        long columns = 0;
        bool fits = products.Length is >= 1 and <= GcnKernels.GemvMultiProducts && m is >= 1 and <= GcnKernels.GemvRows && k > 0
                    && (long)m * k < MostProductElements;
        foreach (var product in products)
        {
            fits &= product.Columns >= 0 && (long)k * product.Columns < MostProductElements && (long)m * product.Columns < MostProductElements;
            widest = Math.Max(widest, product.Columns);
            columns += product.Columns;
        }

        int blocks = (widest + GcnKernels.GemvNnColumns - 1) / GcnKernels.GemvNnColumns;
        if (!fits || widest == 0 || blocks > DeviceLimits.MaxGridDimX || products.Length > DeviceLimits.MaxGridDimY || Gcn("gemv_multi_f32") is not { } function)
        {
            base.MatMulManyKernel(a, m, k, products);
            return;
        }

        Span<ulong> values = stackalloc ulong[3 + 4 * GcnKernels.GemvMultiProducts];
        (values[0], values[1], values[2]) = (P(a), U(m), U(k));
        for (int j = 0; j < GcnKernels.GemvMultiProducts; j++)
        {
            var (weight, bias, output, width) = products[Math.Min(j, products.Length - 1)];
            values[3 + 4 * j] = P(weight);
            values[4 + 4 * j] = bias is null ? 0UL : P(bias);
            values[5 + 4 * j] = P(output);
            values[6 + 4 * j] = j < products.Length ? U(width) : 0UL;
        }

        if (Profiling)
        {
            LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"gemv_multi_f32 {m}x{columns}x{k} ({products.Length} products)"), 2.0 * m * columns * k);
        }

        LaunchProduct(function, blocks, products.Length, 1, values);
    }

    // c[i] = op(a[i]) · op(b[i]) + beta · c[i] (+ bias) for each batch entry on the GCN kernels: false when they cannot
    // take it (unavailable, a matrix too large, a grid past the device's limits).
    private bool FloatProduct(Storage a, Storage b, Storage c, int batch, int m, int n, int k, bool transA, bool transB, float beta, Storage? bias)
    {
        long mk = (long)m * k, kn = (long)k * n, mn = (long)m * n;
        if (mk >= MostProductElements || kn >= MostProductElements || mn >= MostProductElements)
        {
            return false;
        }

        bool few = bias is null && m <= FewRowGroups * GcnKernels.GemvRows && !transA && kn >= 1 << 16;
        string name = few ? (transB ? "gemv_nt_f32" : "gemv_nn_f32") : "matmul_f32";
        if (Gcn(name) is not { } function)
        {
            return false;
        }

        int columns = few ? (transB ? GcnKernels.GemvNtColumns : GcnKernels.GemvNnColumns) : GcnKernels.MatMulTile;
        int rowsPerBlock = few ? GcnKernels.GemvRows : GcnKernels.MatMulTile;
        long blocks = (n + columns - 1) / columns, blocksY = (m + rowsPerBlock - 1) / rowsPerBlock;
        if (blocks > DeviceLimits.MaxGridDimX || blocksY > DeviceLimits.MaxGridDimY)
        {
            return false;
        }

        int most = Math.Max(1, DeviceLimits.MaxGridDimZ);
        for (int first = 0; first < batch; first += most)
        {
            int count = Math.Min(most, batch - first);
            ulong at = (ulong)first * sizeof(float);
            if (Profiling)
            {
                LabelNextLaunch(string.Create(CultureInfo.InvariantCulture,
                    $"{(few ? "gemv" : "matmul")}_{(transA ? 't' : 'n')}{(transB ? 't' : 'n')}{(bias is null ? "" : "+bias")} {m}x{n}x{k}{(count > 1 ? $" batch {count}" : "")}"),
                    2.0 * m * n * k * count);
            }

            if (few)
            {
                LaunchProduct(function, (int)blocks, (int)blocksY, count, P(a) + at * (ulong)mk, P(b) + at * (ulong)kn, P(c) + at * (ulong)mn, U(m), U(n), U(k), F(beta),
                    (ulong)mk, (ulong)kn, (ulong)mn);
                continue;
            }

            LaunchProduct(function, (int)blocks, (int)blocksY, count, P(a) + at * (ulong)mk, P(b) + at * (ulong)kn, P(c) + at * (ulong)mn, U(m), U(n), U(k),
                transA ? 1UL : 0UL, transB ? 1UL : 0UL, F(beta), (ulong)mk, (ulong)kn, (ulong)mn, F(1f), bias is null ? 0UL : P(bias));
        }

        return true;
    }

}
