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
// IDRAK_HIP_GEMV_SPLITS), the splits' sums added in split order by int8_gemv_finish_f32 (rule 86, no atomics); more int8
// rows on int8_matmul_f32; more int4 and bfloat16 rows expanded to float32 and multiplied (as CUDA without tensor cores);
// the expansions and bfloat16 packing. Batches run in grid z up to the device's reported limit (rule 84), each launch
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
            _ when kernel.StartsWith("int8_gemv", StringComparison.Ordinal) && kernel != "int8_gemv_finish_f32" => (mnk, 4.0 * k * WordsOf(4) + 4.0 * m * k + 4.0 * n + outputs),
            _ when kernel.StartsWith("int4_gemv", StringComparison.Ordinal) => (mnk + (double)k * n, 4.0 * k * WordsOf(8) + 32.0 * ((k + 31) / 32) * WordsOf(8) + 4.0 * m * k + outputs),
            _ when kernel.StartsWith("bf16_gemv", StringComparison.Ordinal) || kernel.StartsWith("f16_gemv", StringComparison.Ordinal) => (mnk, 4.0 * k * WordsOf(2) + 4.0 * m * k + outputs),
            "int8_gemv_finish_f32" => ((double)m * n * splits, 4.0 * m * n * (splits + 1)),
            "int8_matmul_f32" => (mnk, 4.0 * m * WordsOf(4) * k + 4.0 * m * k * WordsOf(4) + 4.0 * m * n + 4.0 * n),
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

    // Few rows through packed weights on the kernel written for the fewest rows holding m, k split over grid y, the splits'
    // sums added in split order by int8_gemv_finish_f32: false when the kernels cannot take it.
    private bool PackedFewRows(Weights format, Storage x, Storage packed, Storage? scales, Storage y, int m, int n, int k)
    {
        int perWord = GcnKernels.ColumnsPerWord(format);
        long words = (n + perWord - 1) / perWord;
        if (words * k >= MostProductElements || (long)m * k >= MostProductElements || 64L * m * n >= MostProductElements)
        {
            return false;
        }

        string name = GcnKernels.PackedGemvKernel(format, m);
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
        var finish = count > 1 ? Gcn("int8_gemv_finish_f32") : null;
        if (count > DeviceLimits.MaxGridDimY || count > 1 && finish is null)
        {
            return false;
        }

        bool scaled = format == Weights.Int8;
        var part = count > 1 ? Allocate(count * m * n, zeroed: false) : null;
        try
        {
            if (Profiling)
            {
                LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"{name} {m}x{n}x{k}{(count > 1 ? $" split {count}" : "")}"), 2.0 * m * n * k);
            }

            ulong scaleAddress = format is Weights.Int8 or Weights.Int4 ? P(scales!) : 0UL;
            LaunchProduct(function, columnBlocks, count, 1, P(x), P(packed), scaleAddress, P(y), part is null ? P(y) : P(part), U(m), U(n), U(k), (ulong)words,
                U(chunk), U(count), 0UL);
            if (part is not null)
            {
                int size = m * n;
                if (Profiling)
                {
                    LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"int8_gemv_finish_f32 {m}x{n} split {count}"), (double)count * size);
                }

                LaunchProduct(finish!, (int)PointwiseBlocks(size), 1, 1, P(part), scaled ? P(scales!) : 0UL, P(y), U(n), U(size), U(count), U(size));
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
