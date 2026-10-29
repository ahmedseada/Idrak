// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Weights = Idrak.Gpu.Hip.GcnKernels.PackedWeights;

namespace Idrak.Gpu.Hip;

// HIP's matrix products on its own GCN kernels (GcnKernels.Products.cs), where CUDA overrides them and with CUDA's rules:
// the float32 product (every transposition, beta, batches; a bias through MatMulBias), few rows (decoding: up to two
// groups of 8 rows, b not transposed in a, n·k of 65,536 or more, as CUDA's gemv_nn_f32 / gemv_nt_f32) on the few-row
// kernels and the rest on the tiled one; the packed few-row products (int8, int4, bfloat16, float16 weights, 1 to 8
// rows, the kernel written for the fewest rows that hold m) with k split over grid y (leftover 8.10: about two blocks a
// compute unit the device reports, a power of two, at most 64 and chunks of 64 rows or more, as CUDA's formula, or
// IDRAK_HIP_GEMV_SPLITS), the splits' sums added in split order by each column range's last block (IDRAK_HIP_FINISH) or by
// int8_gemv_finish_f32 (rule 86: never added by atomics); more int8
// rows on int8_matmul_f32; more int4 and bfloat16 rows expanded to float32 and multiplied (as CUDA without tensor cores);
// the expansions and bfloat16 packing; up to three float32 products of one input of few rows in one launch
// (MatMulManyKernel, gemv_multi_f32, as CUDA's). Batches run in grid z up to the device's reported limit (rule 84), each launch
// labelled for --profile as CUDA's ("matmul_nn m x n x k" with 2·m·n·k operations). Unavailable kernels, matrices of 2^30
// elements or more (the kernels' 32-bit byte offsets) and grids past the device's limits take the host fallback.
internal sealed unsafe partial class HipBackend
{
    /// <summary>The setting that forces the k splits of the packed few-row products (a positive count; unset: CUDA's formula).</summary>
    internal const string GemvSplitsVariable = "IDRAK_HIP_GEMV_SPLITS";

    /// <summary>The split count <c>IDRAK_HIP_GEMV_SPLITS</c> (<paramref name="setting"/>) forces, or null to use the formula.</summary>
    internal static int? GemvSplitsSetting(string? setting) =>
        int.TryParse(setting, NumberStyles.None, CultureInfo.InvariantCulture, out int splits) && splits > 0 ? splits : null;

    private static readonly int? ForcedGemvSplits = GemvSplitsSetting(Environment.GetEnvironmentVariable(GemvSplitsVariable));

    // Row groups of 8 the few-row float32 products take (grid y), as CUDA's FewRowGroups.
    private const int FewRowGroups = 2;

    // The most elements a matrix the products take may have: their byte offsets are 32-bit.
    private const long MostProductElements = 1L << 30;

    /// <summary>
    /// The k splits of a packed few-row product with <paramref name="columnBlocks"/> blocks of words over
    /// <paramref name="computeUnits"/> compute units: about two blocks a unit, rounded to a power of two, from 1 to
    /// min(64, k / 64) (CUDA's GemvSplitCount); <paramref name="forced"/> instead when given (within 1 and k).
    /// </summary>
    internal static int GemvSplitCount(int computeUnits, int columnBlocks, int k, int? forced = null)
    {
        if (forced is int f)
        {
            return Math.Clamp(f, 1, Math.Max(1, k));
        }

        int wanted = (2 * Math.Max(1, computeUnits) + columnBlocks - 1) / Math.Max(1, columnBlocks);
        int splits = 1 << (int)Math.Round(Math.Log2(Math.Max(1, wanted)));
        return Math.Clamp(splits, 1, Math.Max(1, Math.Min(64, k / 64)));
    }

    /// <summary>
    /// A product kernel's floating-point operations and global-memory bytes for one call, from the table at the top of
    /// GcnKernels.Products.cs (<paramref name="splits"/>: a packed few-row product's k splits; <paramref name="batch"/>: the
    /// float32 products'); (0, 0) for a kernel it does not list. The profile's labels and <c>--bench-hip</c>'s rates read it.
    /// </summary>
    internal static (double Flops, double Bytes) ProductCost(string kernel, long m, long n, long k, int splits = 1, int batch = 1, bool beta = false)
    {
        double mnk = 2.0 * m * n * k;
        long WordsOf(int perWord) => (n + perWord - 1) / perWord;
        double outputs = 4.0 * m * n * (splits > 1 ? splits : 1);
        return kernel switch
        {
            "matmul_f32" => (mnk * batch, 4.0 * batch * (m * k + k * n + m * n * (beta ? 2 : 1))),
            "gemv_nn_f32" or "gemv_nt_f32" => (mnk * batch, 4.0 * batch * (m * k + (m + 7) / 8 * k * n + m * n)),
            "gemv_multi_f32" => (mnk, 4.0 * (m * k + k * n + m * n + n)),                       // n: the products' columns together
            _ when kernel.Contains("_gemv_addnorm", StringComparison.Ordinal) => AddNormProductCost(kernel, m, n, k, splits, batch, beta),
            _ when kernel.StartsWith("int8_gemv", StringComparison.Ordinal) && kernel != "int8_gemv_finish_f32" => (mnk, 4.0 * k * WordsOf(4) + 4.0 * m * k + 4.0 * n + outputs),
            _ when kernel.StartsWith("int4_gemv", StringComparison.Ordinal) => (mnk + (double)k * n, 4.0 * k * WordsOf(8) + 32.0 * ((k + 31) / 32) * WordsOf(8) + 4.0 * m * k + outputs),
            _ when kernel.StartsWith("bf16_gemv", StringComparison.Ordinal) || kernel.StartsWith("f16_gemv", StringComparison.Ordinal) => (mnk, 4.0 * k * WordsOf(2) + 4.0 * m * k + outputs),
            "int8_gemv_finish_f32" => ((double)m * n * splits, 4.0 * m * n * (splits + 1)),
            "int8_matmul_f32" => (mnk, 4.0 * m * WordsOf(4) * k + 4.0 * m * k * WordsOf(4) + 4.0 * m * n + 4.0 * n),
            _ => (0, 0),
        };
    }

    // A product with the add and norms (*_gemv_addnorm[_rR]_f32): the plain product's cost, then its last block's three
    // passes over the m × n outputs (gemv_addnorm_f32's operations, y read back instead of the splits).
    private static (double Flops, double Bytes) AddNormProductCost(string kernel, long m, long n, long k, int splits, int batch, bool beta)
    {
        var product = ProductCost(kernel.Replace("_gemv_addnorm", "_gemv", StringComparison.Ordinal), m, n, k, splits, batch, beta);
        return (product.Flops + 9.0 * m * n, product.Bytes + 4.0 * 9 * m * n);
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

    public override void Int8MatMulKernel(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k)
    {
        if (m <= 0 || n <= 0)
        {
            return;
        }

        if (k > 0 && (m <= GcnKernels.GemvRows ? PackedFewRows(Weights.Int8, x, q, scales, y, m, n, k) : Int8ManyRows(x, q, scales, y, m, n, k)))
        {
            return;
        }

        base.Int8MatMulKernel(x, q, scales, y, m, n, k);
    }

    public override void Int4MatMulKernel(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k)
    {
        if (m <= 0 || n <= 0)
        {
            return;
        }

        if (k > 0 && (m <= GcnKernels.GemvRows ? PackedFewRows(Weights.Int4, x, q, scales, y, m, n, k) : ExpandedProduct(Weights.Int4, x, q, scales, y, m, n, k)))
        {
            return;
        }

        base.Int4MatMulKernel(x, q, scales, y, m, n, k);
    }

    public override void BFloat16MatMulKernel(Storage x, Storage packed, Storage y, int m, int n, int k)
    {
        if (m <= 0 || n <= 0)
        {
            return;
        }

        if (k > 0 && (m <= GcnKernels.GemvRows ? PackedFewRows(Weights.BFloat16, x, packed, null, y, m, n, k) : ExpandedProduct(Weights.BFloat16, x, packed, null, y, m, n, k)))
        {
            return;
        }

        base.BFloat16MatMulKernel(x, packed, y, m, n, k);
    }

    public override void Float16MatMulKernel(Storage x, Storage packed, Storage y, int m, int n, int k)
    {
        if (m <= 0 || n <= 0)
        {
            return;
        }

        if (k > 0 && m <= GcnKernels.GemvRows && PackedFewRows(Weights.Float16, x, packed, null, y, m, n, k))
        {
            return;
        }

        base.Float16MatMulKernel(x, packed, y, m, n, k);                     // widened, then the float32 product (both on kernels)
    }

    // Few rows through packed weights on the kernel written for the fewest rows holding m, k split over grid y: false when
    // the kernels cannot take it. The splits' sums are added in split order by the launch's last block of each column
    // range (IDRAK_HIP_FINISH=block, the default: one launch), or by int8_gemv_finish_f32 (launch). A gated variant reads
    // its input row as act(x) · up (x the gate, HipBackend.Decoding.cs); with `tail`, the residual addition and the norms
    // follow (y normalized first with a y gain): by the add-and-norm product's last block, or by gemv_addnorm_f32 after
    // the plain product.
    private bool PackedFewRows(Weights format, Storage x, Storage packed, Storage? scales, Storage y, int m, int n, int k,
        GcnKernels.PackedGemvVariant variant = GcnKernels.PackedGemvVariant.Plain, Storage? up = null, in AddNormTail tail = default)
    {
        int perWord = GcnKernels.ColumnsPerWord(format);
        long words = (n + perWord - 1) / perWord;
        if (words * k >= MostProductElements || (long)m * k >= MostProductElements || 64L * m * n >= MostProductElements)
        {
            return false;
        }

        bool addNorm = tail.Residual is not null, lastBlock = FinishByLastBlock;
        var kernel = addNorm && lastBlock ? GcnKernels.PackedGemvVariant.AddNorm : variant;
        string name = kernel == GcnKernels.PackedGemvVariant.Plain ? GcnKernels.PackedGemvKernel(format, m) : GcnKernels.FusedGemvKernel(format, kernel, m);
        int columnBlocks = (int)((words + GcnKernels.PackedGemvWords - 1) / GcnKernels.PackedGemvWords);
        if (Gcn(name) is not { } function || columnBlocks > DeviceLimits.MaxGridDimX)
        {
            return false;
        }

        // Split boundaries on 64 rows for int4 (whole groups of scales, as CUDA's), anywhere for the others.
        int align = format == Weights.Int4 ? 64 : 1;
        int wanted = GemvSplitCount(DeviceLimits.ComputeUnits, columnBlocks, k, ForcedGemvSplits);
        int chunk = ((k + wanted - 1) / wanted + align - 1) / align * align;
        int count = (k + chunk - 1) / chunk;
        var finish = count > 1 && !addNorm && !lastBlock ? Gcn("int8_gemv_finish_f32") : null;
        var normalizing = addNorm && !lastBlock ? Gcn("gemv_addnorm_f32") : null;
        if (count > DeviceLimits.MaxGridDimY || count > 1 && !addNorm && !lastBlock && finish is null || addNorm && !lastBlock && normalizing is null)
        {
            return false;
        }

        bool scaled = format == Weights.Int8;
        var part = count > 1 ? Allocate(count * m * n, zeroed: false) : null;
        ulong counters = lastBlock && (count > 1 || addNorm) ? P(Counters(columnBlocks + 1)) : 0UL;
        try
        {
            if (Profiling)
            {
                string form = counters != 0 ? " (last block)" : "";
                double flops = 2.0 * m * n * k + (kernel == GcnKernels.PackedGemvVariant.AddNorm ? Cost("gemv_addnorm_f32", 1).Flops * m * n : 0);
                LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"{name} {m}x{n}x{k}{(count > 1 ? $" split {count}" : "")}{form}"), flops);
            }

            ulong scaleAddress = format is Weights.Int8 or Weights.Int4 ? P(scales!) : 0UL;
            ulong partAddress = part is null ? P(y) : P(part);
            if (kernel == GcnKernels.PackedGemvVariant.AddNorm)
            {
                LaunchProduct(function, columnBlocks, count, 1, P(x), P(packed), scaleAddress, P(y), partAddress, U(m), U(n), U(k), (ulong)words, U(chunk), U(count),
                    counters, P(tail.Residual!), P(tail.Sum!), P(tail.Gain!), P(tail.Normalized!), tail.YGain is null ? 0UL : P(tail.YGain), F(tail.Eps), F(tail.Offset),
                    F(tail.YEps), F(tail.YOffset), tail.YGain is null ? 0UL : 1UL);
            }
            else if (up is null)
            {
                LaunchProduct(function, columnBlocks, count, 1, P(x), P(packed), scaleAddress, P(y), partAddress, U(m), U(n), U(k), (ulong)words, U(chunk), U(count),
                    counters);
            }
            else
            {
                LaunchProduct(function, columnBlocks, count, 1, P(x), P(packed), scaleAddress, P(y), partAddress, U(m), U(n), U(k), (ulong)words, U(chunk), U(count),
                    counters, P(up));
            }

            if (normalizing is not null)
            {
                // The splits' sums in split order (int8 scaled), or y itself, then the residual and the norms, a block a row.
                if (Profiling)
                {
                    LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"gemv_addnorm_f32 {m}x{n}{(count > 1 ? $" split {count}" : "")}"),
                        Cost("gemv_addnorm_f32", count).Flops * m * n);
                }

                LaunchKernel(normalizing, (uint)Math.Min(m, DeviceLimits.MaxGridDimX), 1, part is null ? P(y) : P(part), part is not null && scaled ? P(scales!) : 0UL, P(y),
                    P(tail.Residual!), P(tail.Sum!), P(tail.Gain!), P(tail.Normalized!), tail.YGain is null ? 0UL : P(tail.YGain), U(n), U(count), U(m * n),
                    F(tail.Eps), F(tail.Offset), F(tail.YEps), F(tail.YOffset), tail.YGain is null ? 0UL : 1UL, U(m));
            }
            else if (finish is not null)
            {
                int size = m * n;
                if (Profiling)
                {
                    LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"int8_gemv_finish_f32 {m}x{n} split {count}"), (double)count * size);
                }

                LaunchProduct(finish, (int)PointwiseBlocks(size), 1, 1, P(part!), scaled ? P(scales!) : 0UL, P(y), U(n), U(size), U(count), U(size));
            }
        }
        finally
        {
            part?.Release();                                                     // the pool reuses it after the queued launches
        }

        return true;
    }

    // Many int8 rows on int8_matmul_f32 (a thread a row and packed word), as CUDA's.
    private bool Int8ManyRows(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k)
    {
        long words = (n + 3) / 4;
        if (m * words >= MostProductElements || words * k >= MostProductElements || (long)m * k >= MostProductElements || (long)m * n >= MostProductElements
            || Gcn("int8_matmul_f32") is not { } function)
        {
            return false;
        }

        if (Profiling)
        {
            LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"int8_matmul_f32 {m}x{n}x{k}"), 2.0 * m * n * k);
        }

        LaunchProduct(function, (int)PointwiseBlocks(m * words), 1, 1, P(x), P(q), P(scales), P(y), U(k), (ulong)words, U(n), (ulong)(m * words));
        return true;
    }

    // Many rows through int4 or bfloat16 weights: expanded to float32 on the device, then the float32 product (CUDA's path
    // without tensor cores).
    private bool ExpandedProduct(Weights format, Storage x, Storage packed, Storage? scales, Storage y, int m, int n, int k)
    {
        if ((long)k * n >= MostProductElements || Gcn(GcnKernels.DequantKernel(format)) is null || Gcn("matmul_f32") is null)
        {
            return false;
        }

        var w = Allocate(k * n, zeroed: false);
        try
        {
            if (!Expand(format, packed, scales, w, k, n) || !FloatProduct(x, w, y, 1, m, n, k, false, false, 0f, null))
            {
                return false;
            }
        }
        finally
        {
            w.Release();
        }

        return true;
    }

    public override void Int8DequantizeKernel(Storage q, Storage scales, Storage w, int k, int n)
    {
        if (!Expand(Weights.Int8, q, scales, w, k, n))
        {
            base.Int8DequantizeKernel(q, scales, w, k, n);
        }
    }

    public override void Int4DequantizeKernel(Storage q, Storage scales, Storage w, int k, int n)
    {
        if (!Expand(Weights.Int4, q, scales, w, k, n))
        {
            base.Int4DequantizeKernel(q, scales, w, k, n);
        }
    }

    public override void BFloat16DequantizeKernel(Storage packed, Storage w, int k, int n)
    {
        if (!Expand(Weights.BFloat16, packed, null, w, k, n))
        {
            base.BFloat16DequantizeKernel(packed, w, k, n);
        }
    }

    public override void Float16DequantizeKernel(Storage packed, Storage w, int k, int n)
    {
        if (!Expand(Weights.Float16, packed, null, w, k, n))
        {
            base.Float16DequantizeKernel(packed, w, k, n);
        }
    }

    // w[k, n] = the packed weights' float32 values on the expansion kernel: true when done (nothing to do included).
    private bool Expand(Weights format, Storage packed, Storage? scales, Storage w, int k, int n)
    {
        if (k <= 0 || n <= 0)
        {
            return true;
        }

        int perWord = GcnKernels.ColumnsPerWord(format);
        long words = (long)k * ((n + perWord - 1) / perWord);
        string name = GcnKernels.DequantKernel(format);
        if ((long)k * n >= MostProductElements || Gcn(name) is not { } function)
        {
            return false;
        }

        if (Profiling)
        {
            LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"{name} {k}x{n}"), Cost(name).Flops * k * n);
        }

        uint blocks = PointwiseBlocks(words);
        ulong perRow = (ulong)((n + perWord - 1) / perWord);
        if (format is Weights.Int8 or Weights.Int4)
        {
            LaunchProduct(function, (int)blocks, 1, 1, P(packed), P(scales!), P(w), perRow, U(n), (ulong)words);
        }
        else
        {
            LaunchProduct(function, (int)blocks, 1, 1, P(packed), P(w), perRow, U(n), (ulong)words);
        }

        return true;
    }

    public override void PackBFloat16Kernel(Storage x, Storage packed, int n)
    {
        int words = (n + 1) / 2;
        if (n <= 0)
        {
            return;
        }

        if (n >= MostProductElements || Gcn("bf16_pack_f32") is not { } function)
        {
            base.PackBFloat16Kernel(x, packed, n);
            return;
        }

        if (Profiling)
        {
            LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"bf16_pack_f32 {n}"), 0);
        }

        LaunchProduct(function, (int)PointwiseBlocks(words), 1, 1, P(x), P(packed), U(n), U(words));
    }
}
