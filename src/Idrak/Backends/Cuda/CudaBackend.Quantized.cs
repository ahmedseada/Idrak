using static Idrak.Backends.Cuda.CudaDriver;

namespace Idrak.Backends.Cuda;

// Int8 weight-only quantization kernels (see PtxKernels.Quantized.cs).
internal sealed unsafe partial class CudaBackend
{
    public override void Int8MatMul(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k)
    {
        int words = (n + 3) / 4;
        if (m > PtxKernels.GemvRows || k == 0)
        {
            Launch1D(K("int8_matmul_f32"), m * words, P(x), P(q), P(scales), P(y), U(k), U(words), U(n), U(m * words));
            return;
        }

        PackedFewRows("int8_gemv_f32", x, q, scales, y, m, n, k, words);
    }

    public override void BFloat16MatMul(Storage x, Storage packed, Storage y, int m, int n, int k)
    {
        int words = (n + 1) / 2;
        if (m > PtxKernels.GemvRows && PackedMatMulLarge(2, x, packed, null, y, m, n, k))
        {
            return;
        }

        if (m > PtxKernels.GemvRows || k == 0)
        {
            var w = Allocate(k * n, zeroed: false);
            try
            {
                BFloat16Dequantize(packed, w, k, n);
                MatMul(x, w, y, m, n, k, false, false, 0f);
            }
            finally
            {
                w.Release();
            }

            return;
        }

        PackedFewRows("bf16_gemv_f32", x, packed, packed, y, m, n, k, words);
    }

    public override void Int4MatMul(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k)
    {
        int words = (n + 7) / 8;
        if (m > PtxKernels.GemvRows && PackedMatMulLarge(1, x, q, scales, y, m, n, k))
        {
            return;
        }

        if (m > PtxKernels.GemvRows || k == 0)
        {
            var w = Allocate(k * n, zeroed: false);
            try
            {
                Int4Dequantize(q, scales, w, k, n);
                MatMul(x, w, y, m, n, k, false, false, 0f);
            }
            finally
            {
                w.Release();
            }

            return;
        }

        PackedFewRows("int4_gemv_f32", x, q, scales, y, m, n, k, words, align: 64);
    }

    public override void Int4Dequantize(Storage q, Storage scales, Storage w, int k, int n)
    {
        int words = (n + 7) / 8;
        Launch1D(K("int4_dequant_f32"), k * words, P(q), P(scales), P(w), U(words), U(n), U(k * words));
    }

    public override void BFloat16Dequantize(Storage packed, Storage w, int k, int n)
    {
        int words = (n + 1) / 2;
        Launch1D(K("bf16_dequant_f32"), k * words, P(packed), P(w), U(words), U(n), U(k * words));
    }

    /// <summary>Benchmarks only: the k splits of prompt-sized packed products on tensor cores instead of the heuristic's.</summary>
    internal static int? PackedSplits { get; set; }

    /// <summary>Benchmarks only: the number of blocks decoding attention splits the cache over instead of the heuristic's.</summary>
    internal static int? DecodeSplits { get; set; }

    /// <summary>Benchmarks only: the number of k splits of the few-row packed products instead of the heuristic's.</summary>
    internal static int? GemvSplits { get; set; }

    // k splits of a few-row packed product with `blocks` column blocks: about two blocks per SM, rounded to a power of
    // two so the chunks stay multiples of the 64-row unrolled step (measured on an RTX 5070 Ti with --bench-gemv: the
    // best or within 3% of it for the Qwen3-0.6B decoding shapes; four blocks per SM was up to 40% slower).
    private int GemvSplitCount(int blocks, int k)
    {
        int wanted = (2 * Math.Max(1, _multiprocessors) + blocks - 1) / blocks;
        int splits = 1 << (int)Math.Round(Math.Log2(Math.Max(1, wanted)));
        return Math.Clamp(splits, 1, Math.Max(1, Math.Min(64, k / 64)));
    }

    // Few rows (decoding) through packed weights: read each weight word once, with enough blocks to keep every
    // multiprocessor busy; narrow matrices split k, and the last block of each column range adds the splits in order.
    // `align`: split boundaries fall on multiples of it (int4 splits start on a 64-row block).
    // `up`: the gated kernels' second input; `tail`: further arguments (the addnorm kernels').
    private void PackedFewRows(string kernel, Storage x, Storage q, Storage scales, Storage y, int m, int n, int k, int words, int align = 1,
        Storage? up = null, ulong[]? tail = null)
    {
        int columnBlocks = (words + 31) / 32;
        int splits = GemvSplits is int forced ? Math.Clamp(forced, 1, Math.Max(1, k / 16)) : GemvSplitCount(columnBlocks, k);
        int chunk = ((k + splits - 1) / splits + align - 1) / align * align;
        splits = (k + chunk - 1) / chunk;
        var counters = SplitCounters(columnBlocks + 1);
        void Run(Storage partials, int count)
        {
            ulong[] args = [P(x), P(q), P(scales), P(y), P(partials), U(m), U(n), U(k), U(words), U(chunk), U(count), P(counters),
                .. up is null ? [] : new[] { P(up) }, .. tail ?? []];
            Launch(K(kernel), (uint)columnBlocks, (uint)count, 1, PtxKernels.Int8GemvThreads, 1, args);
        }

        if (splits == 1)
        {
            Run(y, 1);
            return;
        }

        var part = Allocate(splits * m * n, zeroed: false);
        try
        {
            Run(part, splits);
        }
        finally
        {
            part.Release();
        }
    }

    /// <summary>Benchmarks only: the row tile (64 or 128) of prompt-sized packed products instead of the heuristic's.</summary>
    internal static int? PromptTileRowsOverride { get; set; }

    // Row tile of a prompt-sized packed product: 64 when the last 128-row tile would be at most half full (180 rows: 3
    // tiles of 64, 192 rows computed, instead of 2 of 128, 256), else 128.
    private static int PromptTileRows(int m) =>
        PromptTileRowsOverride ?? (m < 1024 && m % PtxKernels.TensorTile is > 0 and <= 64 ? 64 : PtxKernels.TensorTile);

    // k splits of a prompt-sized packed product: up to four blocks per SM, chunks of 256 k or more; none when the tiles
    // already fill one wave (85-100% of the SMs: splitting then only adds the zeroing and atomic additions; --bench-gemv,
    // 180 rows, q+k+v in one launch = 64 tiles on 70 SMs: 35.3 us unsplit against 41.5 with 4 splits).
    private int PromptSplits(int tiles, int k)
    {
        int sms = Math.Max(1, _multiprocessors);
        return tiles * 100 >= sms * 85 && tiles <= sms ? 1 : Math.Clamp(Math.Min(k / 256, 4 * sms / tiles), 1, 8);
    }

    public override bool PackedMatMulLarge(int kind, Storage x, Storage packed, Storage? scales, Storage y, int m, int n, int k)
    {
        // From 9 rows (8 and fewer take the GEMV kernels): short prompts fill part of one row tile, which the kernels
        // bound-check, instead of expanding the whole weight to float32 first (--bench-gemv, 1024 -> 151936 int8 on an
        // RTX 5070 Ti: 9-63 rows 2.3 ms through the float32 copy, 64 rows 0.36 ms packed).
        if (m <= PtxKernels.GemvRows || n < 64 || k < 8)
        {
            return false;
        }

        // Tensor cores (MixedPrecision): the weights unpacked into the bfloat16 tiles as they are loaded.
        int tileRows = PromptTileRows(m);
        string packedKernel = $"gemm_tc_nn_{(kind switch { 0 => "int8w", 1 => "int4w", _ => "bf16w" })}{(tileRows == 64 ? "_m64" : "")}_f32";
        if (MixedPrecision.UsesTensorCores && k >= 32 && TensorKernel(packedKernel) is { } tensor)
        {
            int perWord = kind switch { 0 => 4, 1 => 8, _ => 2 };
            if (_profile is not null)
            {
                _profileLabel = $"gemm_tc_nn_{(kind switch { 0 => "int8w", 1 => "int4w", _ => "bf16w" })}{(tileRows == 64 ? "_m64" : "")} {m}x{n}x{k}";
                _profileFlops = 2.0 * m * n * k;
            }

            // Split k when the output tiles leave SMs idle (prompt-sized m): up to two blocks per SM, chunks of 256 k or
            // more, partial sums added into the zeroed output.
            int rowTiles = (m + tileRows - 1) / tileRows, columnTiles = (n + PtxKernels.TensorTile - 1) / PtxKernels.TensorTile;
            int splits = PackedSplits is int forced ? Math.Clamp(forced, 1, Math.Max(1, k / 32))
                : PromptSplits(rowTiles * columnTiles, k);
            if (splits > 1)
            {
                Check(cuMemsetD32Async(P(y), 0, (nuint)((long)m * n), _stream), nameof(cuMemsetD32Async));
            }

            Launch(tensor, (uint)columnTiles, (uint)rowTiles,
                (uint)splits, PtxKernels.TensorThreads, 1, P(x), P(packed), P(y), U(m), U(n), U(k), F(0f), 0UL, 0UL, 0UL, 0UL,
                U(k), U((n + perWord - 1) / perWord), U(n), scales is null ? 0UL : P(scales));
            return true;
        }

        string format = kind switch { 0 => "int8", 1 => "int4", _ => "bf16" };
        long tiles128 = (long)((m + 127) / 128) * ((n + 127) / 128);
        int tile = tiles128 >= Math.Max(1, _multiprocessors) ? 128 : 64;
        Launch(K($"gemm{tile}_{format}_f32"), (uint)((n + tile - 1) / tile), (uint)((m + tile - 1) / tile), 1, PtxKernels.GemmThreads, 1,
            P(x), P(packed), P(y), U(m), U(n), U(k), U(0), U(0), F(0f), 0UL, 0UL, 0UL, P(scales ?? packed));
        return true;
    }

    // Prompt-sized products of one input through 2-3 packed layers (queries/keys/values, gate/up) in one tensor-core
    // launch: the column tiles of every product side by side, so a few rows still fill the GPU (180 rows are 2 row
    // tiles: q/k/v alone gave 32 tiles, 16 and 16 in separate launches). Widths must be multiples of the 128-column tile,
    // and the layers must have no bias (the caller adds none).
    private bool PackedManyLarge(int kind, Storage x, int m, int k,
        ReadOnlySpan<(Storage Packed, Storage? Scales, Storage? Bias, Storage Output, int Columns)> products)
    {
        if (products.Length is < 2 or > 3 || m < 64 || k < 32 || !MixedPrecision.UsesTensorCores || kind is < 0 or > 2)
        {
            return false;
        }

        int columnTiles = 0;
        foreach (var product in products)
        {
            if (product.Bias is not null || product.Columns % PtxKernels.TensorTile != 0 || product.Columns == 0)
            {
                return false;
            }

            columnTiles += product.Columns / PtxKernels.TensorTile;
        }

        int tileRows = PromptTileRows(m);
        string format = kind switch { 0 => "int8w", 1 => "int4w", _ => "bf16w" };
        if (TensorKernel($"gemm_tc_nn_{format}_multi{(tileRows == 64 ? "_m64" : "")}_f32") is not { } tensor)
        {
            return false;
        }

        int perWord = kind switch { 0 => 4, 1 => 8, _ => 2 };
        int rowTiles = (m + tileRows - 1) / tileRows;
        int splits = PackedSplits is int forced ? Math.Clamp(forced, 1, Math.Max(1, k / 32))
            : PromptSplits(rowTiles * columnTiles, k);
        if (_profile is not null)
        {
            _profileLabel = $"gemm_tc_nn_{format}_multi{(tileRows == 64 ? "_m64" : "")} {m}x{string.Join('+', products.ToArray().Select(p => p.Columns))}x{k}";
            _profileFlops = 0;
            foreach (var product in products)
            {
                _profileFlops += 2.0 * m * product.Columns * k;
            }
        }

        if (splits > 1)
        {
            foreach (var product in products)
            {
                Check(cuMemsetD32Async(P(product.Output), 0, (nuint)((long)m * product.Columns), _stream), nameof(cuMemsetD32Async));
            }
        }

        static ulong Scales(Storage? scales) => scales is null ? 0UL : P(scales);
        var p1 = products[1];
        var p2 = products.Length == 3 ? products[2] : products[1];
        int n2 = products.Length == 3 ? products[2].Columns : 0;
        Launch(tensor, (uint)columnTiles, (uint)rowTiles, (uint)splits, PtxKernels.TensorThreads, 1,
            P(x), P(products[0].Packed), P(products[0].Output), U(m), U(products[0].Columns), U(k), F(0f), 0UL, 0UL, 0UL, 0UL,
            U(k), U(products[0].Columns / perWord), U(products[0].Columns), Scales(products[0].Scales),
            P(p1.Packed), P(p1.Output), Scales(p1.Scales), U(p1.Columns),
            P(p2.Packed), P(p2.Output), Scales(p2.Scales), U(n2));
        return true;
    }

    public override bool PackedMatMulLowRank(int kind, Storage x, int m, int k,
        ReadOnlySpan<(Storage Packed, Storage? Scales, Storage Output, int Columns, Storage U, Storage V)> products, int rank)
    {
        if (products.Length is < 1 or > 3 || kind is not (1 or 2) || m < 64 || k < 32 || rank is < 1 or > 32 || !MixedPrecision.UsesTensorCores
            || MixedPrecision.Current == MatMulPrecision.Float8)
        {
            return false;
        }

        bool multi = products.Length > 1;
        int columnTiles = 0;
        foreach (var product in products)
        {
            if (product.Columns == 0 || multi && product.Columns % PtxKernels.TensorTile != 0)
            {
                return false;
            }

            columnTiles += (product.Columns + PtxKernels.TensorTile - 1) / PtxKernels.TensorTile;
        }

        string format = kind == 1 ? "int4w" : "bf16w";
        if (TensorKernel($"gemm_tc_nn_{format}{(multi ? "_multi" : "")}_lr_f32") is not { } tensor)
        {
            return false;
        }

        int perWord = kind == 1 ? 8 : 2;
        int rowTiles = (m + PtxKernels.TensorTile - 1) / PtxKernels.TensorTile;
        int splits = PackedSplits is int forced ? Math.Clamp(forced, 1, Math.Max(1, k / 32)) : PromptSplits(rowTiles * columnTiles, k);
        if (_profile is not null)
        {
            _profileLabel = $"gemm_tc_nn_{format}{(multi ? "_multi" : "")}_lr {m}x{string.Join('+', products.ToArray().Select(p => p.Columns))}x{k}+{rank}";
            _profileFlops = 0;
            foreach (var product in products)
            {
                _profileFlops += 2.0 * m * product.Columns * (k + rank);
            }
        }

        if (splits > 1)
        {
            foreach (var product in products)
            {
                Check(cuMemsetD32Async(P(product.Output), 0, (nuint)((long)m * product.Columns), _stream), nameof(cuMemsetD32Async));
            }
        }

        static ulong Scales(Storage? scales) => scales is null ? 0UL : P(scales);
        var p0 = products[0];
        if (!multi)
        {
            Launch(tensor, (uint)columnTiles, (uint)rowTiles, (uint)splits, PtxKernels.TensorThreads, 1,
                P(x), P(p0.Packed), P(p0.Output), U(m), U(p0.Columns), U(k), F(0f), 0UL, 0UL, 0UL, 0UL,
                U(k), U((p0.Columns + perWord - 1) / perWord), U(p0.Columns), Scales(p0.Scales), P(p0.U), P(p0.V), U(rank));
            return true;
        }

        var p1 = products[1];
        var p2 = products.Length == 3 ? products[2] : products[1];
        int n2 = products.Length == 3 ? products[2].Columns : 0;
        Launch(tensor, (uint)columnTiles, (uint)rowTiles, (uint)splits, PtxKernels.TensorThreads, 1,
            P(x), P(p0.Packed), P(p0.Output), U(m), U(p0.Columns), U(k), F(0f), 0UL, 0UL, 0UL, 0UL,
            U(k), U(p0.Columns / perWord), U(p0.Columns), Scales(p0.Scales),
            P(p1.Packed), P(p1.Output), Scales(p1.Scales), U(p1.Columns),
            P(p2.Packed), P(p2.Output), Scales(p2.Scales), U(n2),
            P(p0.U), P(p0.V), U(rank), P(p1.U), P(p1.V), P(p2.U), P(p2.V));
        return true;
    }

    public override bool PackedMatMulGated(int kind, int activation, Storage gate, Storage up, Storage packed, Storage? scales, Storage y,
        int m, int n, int k)
    {
        // Measured on an RTX 5070 Ti: the activation per input value pays off only with eight columns per word (int4);
        // int8 and bfloat16 decode faster with the separate activation pass.
        if (kind != 1 || m > PtxKernels.GemvRows || k == 0 || activation is not (0 or 1))
        {
            return false;
        }

        int cpw = kind switch { 0 => 4, 1 => 8, _ => 2 };
        string kernel = (kind switch { 0 => "int8_gemv", 1 => "int4_gemv", _ => "bf16_gemv" }) + (activation == 0 ? "_silu_f32" : "_gelu_f32");
        PackedFewRows(kernel, gate, packed, scales ?? packed, y, m, n, k, (n + cpw - 1) / cpw, kind == 1 ? 64 : 1, up);
        return true;
    }

    public override bool PackedMatMulAddRmsNorm(int kind, Storage x, Storage packed, Storage? scales, Storage y, int m, int n, int k,
        Storage residual, Storage sum, Storage gain, Storage normalized, float eps, float offset)
    {
        if (m > PtxKernels.GemvRows || k == 0 || kind is < 0 or > 2)
        {
            return false;
        }

        int cpw = kind switch { 0 => 4, 1 => 8, _ => 2 };
        string kernel = kind switch { 0 => "int8_gemv_addnorm_f32", 1 => "int4_gemv_addnorm_f32", _ => "bf16_gemv_addnorm_f32" };
        PackedFewRows(kernel, x, packed, scales ?? packed, y, m, n, k, (n + cpw - 1) / cpw, kind == 1 ? 64 : 1,
            tail: [P(residual), P(sum), P(gain), P(normalized), F(eps), F(offset)]);
        return true;
    }

    public override bool PackedMatMulMany(int kind, Storage x, int m, int k,
        ReadOnlySpan<(Storage Packed, Storage? Scales, Storage? Bias, Storage Output, int Columns)> products) =>
        PackedMany(kind, x, m, k, products, -1, null);

    public override bool PackedMatMulGatedPair(int kind, int activation, Storage x, int m, int k,
        ReadOnlySpan<(Storage Packed, Storage? Scales, Storage? Bias, Storage Output, int Columns)> products, Storage hidden) =>
        products.Length == 2 && products[0].Columns == products[1].Columns && activation is >= 0 and <= 2
        && PackedMany(kind, x, m, k, products, activation, hidden);

    // activation >= 0: the gate/up pair, with hidden = act(gate) · up written by the same launch.
    private bool PackedMany(int kind, Storage x, int m, int k,
        ReadOnlySpan<(Storage Packed, Storage? Scales, Storage? Bias, Storage Output, int Columns)> products, int activation, Storage? hidden)
    {
        if (m > PtxKernels.GemvRows && hidden is null)
        {
            return PackedManyLarge(kind, x, m, k, products);
        }

        if (m > PtxKernels.GemvRows || k == 0 || products.Length is 0 or > 3)
        {
            return false;
        }

        int cpw = kind switch { 0 => 4, 1 => 8, _ => 2 };
        string kernel = (kind switch { 0 => "int8_gemv_multi", 1 => "int4_gemv_multi", _ => "bf16_gemv_multi" }) + (hidden is null ? "_f32" : "_act_f32");
        int nmax = 0, totalBlocks = 0;
        foreach (var product in products)
        {
            nmax = Math.Max(nmax, product.Columns);
            totalBlocks += ((product.Columns + cpw - 1) / cpw + 31) / 32;
        }

        int columnBlocks = ((nmax + cpw - 1) / cpw + 31) / 32;
        int splits = GemvSplits is int forced ? Math.Clamp(forced, 1, Math.Max(1, k / 16)) : GemvSplitCount(totalBlocks, k);
        int align = kind == 1 ? 64 : 1;
        int chunk = ((k + splits - 1) / splits + align - 1) / align * align;
        splits = (k + chunk - 1) / chunk;
        var counters = SplitCounters(columnBlocks * (products.Length + (hidden is null ? 0 : 1)));
        var part = splits > 1 ? Allocate(products.Length * splits * m * nmax, zeroed: false) : null;
        try
        {
            Span<ulong> args = stackalloc ulong[9 + 3 * 5 + 2];
            args[0] = P(x);
            args[1] = part is null ? P(products[0].Output) : P(part);
            args[2] = U(m);
            args[3] = U(k);
            args[4] = U(chunk);
            args[5] = U(splits);
            args[6] = P(counters);
            args[7] = U(columnBlocks);
            args[8] = U(nmax);
            for (int j = 0; j < 3; j++)
            {
                var product = j < products.Length ? products[j] : products[0];
                args[9 + 5 * j] = P(product.Packed);
                args[10 + 5 * j] = P(product.Scales ?? product.Packed);
                args[11 + 5 * j] = P(product.Output);
                args[12 + 5 * j] = product.Bias is null ? 0UL : P(product.Bias);
                args[13 + 5 * j] = U(j < products.Length ? product.Columns : 0);
            }

            args[24] = U(activation);
            args[25] = hidden is null ? 0UL : P(hidden);
            Launch(K(kernel), (uint)columnBlocks, (uint)splits, (uint)products.Length, PtxKernels.Int8GemvThreads, 1, hidden is null ? args[..24] : args);
        }
        finally
        {
            part?.Release();
        }

        return true;
    }

    private Storage? _splitCounters;

    // Zeroed arrival counters for split products, one per column block; the last block of a range resets its counter.
    private Storage SplitCounters(int blocks)
    {
        // A larger buffer replaces a smaller one without freeing it: recorded graphs may still use the old one.
        var current = Volatile.Read(ref _splitCounters);
        if (current is null || current.Length < blocks)
        {
            current = Allocate(Math.Max(blocks, 4096), zeroed: true);
            Volatile.Write(ref _splitCounters, current);
        }

        return current;
    }

    public override void Int8Dequantize(Storage q, Storage scales, Storage w, int k, int n)
    {
        int words = (n + 3) / 4;
        Launch1D(K("int8_dequant_f32"), k * words, P(q), P(scales), P(w), U(words), U(n), U(k * words));
    }

    public override void KeyValueWriteInt8(Storage source, Storage cache, Storage scales, Storage position, int heads, int steps, int capacity, int dim)
    {
        int n = heads * steps;
        Launch1D(K("kv_write_int8"), n, P(source), P(cache), P(scales), P(position), U(steps), U(capacity), U(dim), U((dim + 3) / 4), U(n));
    }

    public override void AttentionScoresInt8(Storage q, Storage cache, Storage scales, Storage y, int rows, int steps, int capacity, int dim)
    {
        int n = rows * steps * capacity;
        Launch1D(K("attn_scores_int8"), n, P(q), P(cache), P(scales), P(y), U(steps), U(capacity), U(dim), U((dim + 3) / 4), U(n));
    }

    public override void AttentionContextInt8(Storage weights, Storage cache, Storage scales, Storage y, int rows, int steps, int capacity, int dim)
    {
        int words = (dim + 3) / 4, n = rows * steps * words;
        Launch1D(K("attn_context_int8"), n, P(weights), P(cache), P(scales), P(y), U(steps), U(capacity), U(dim), U(words), U(n));
    }

    public override void RmsNorm(Storage x, Storage y, Storage inv, int rows, int cols, float eps) =>
        LaunchRows(K("rms_norm_f32"), rows, P(x), P(y), P(inv), U(cols), F(eps), U(rows));

    public override void RmsNormBackward(Storage dy, Storage y, Storage inv, Storage dx, int rows, int cols) =>
        LaunchRows(K("rms_norm_backward_f32"), rows, P(dy), P(y), P(inv), P(dx), U(cols), U(rows));

    public override void Rope(Storage x, Storage y, Storage cos, Storage sin, Storage positions, int rows, int heads, int steps, int dim, int half, bool interleaved, float sign)
    {
        int n = rows * half;
        Launch1D(K("rope_f32"), n, P(x), P(y), P(cos), P(sin), P(positions), U(heads), U(steps), U(dim), U(half), U(interleaved ? 1 : 0), F(sign), U(n));
    }

    public override void AttentionDecode(Storage q, Storage keys, Storage values, Storage position, Storage y, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale)
    {
        if (dim > PtxKernels.DecodeMaxDim)
        {
            throw new NotSupportedException($"Decoding attention supports head sizes up to {PtxKernels.DecodeMaxDim} on CUDA.");
        }

        int rows = heads * rowsPerHead;
        DecodeSplit(rows, capacity, dim, y, (splits, part, counters) => Launch(K("attention_decode_f32"), (uint)rows, (uint)splits, 1, PtxKernels.RowThreads, 1,
            P(q), P(keys), P(values), P(position), P(y), P(part), P(counters), U(rowsPerHead), U(steps), U(capacity), U(dim), F(scale), U(rows)));
    }

    // Decoding attention has one block per query row (few rows: the heads of one token), so the cached positions are
    // split over about five blocks per SM (--bench-gemv, 16 rows → 22 splits: 4000 positions 75.9 → ~35 µs, 1000
    // positions 22 → ~15 µs); each writes (max, sum, weighted values) for its chunk and the last block of a row to finish
    // merges them (counted in the split counters). The split count depends only on the shapes,
    // so recorded graphs stay valid as the cache fills (chunks are computed on the device from the current length).
    private void DecodeSplit(int rows, int capacity, int dim, Storage y, Action<int, Storage, Storage> launch)
    {
        int splits = DecodeSplits is int forced ? Math.Clamp(forced, 1, 64)
            : Math.Clamp((5 * Math.Max(1, _multiprocessors) + rows - 1) / rows, 1, Math.Clamp(capacity / 64, 1, 32));
        var counters = SplitCounters(rows);
        if (splits == 1)
        {
            launch(1, y, counters);
            return;
        }

        var part = Allocate(rows * splits * (dim + 2), zeroed: false);
        try
        {
            launch(splits, part, counters);
        }
        finally
        {
            part.Release();
        }
    }

    public override void RmsNormAffine(Storage x, Storage gain, Storage y, int rows, int cols, float eps, float offset) =>
        LaunchRows(K("rms_norm_affine_f32"), rows, P(x), P(gain), P(y), U(cols), F(eps), F(offset), U(rows));

    public override void GatedActivation(Storage gate, Storage up, Storage y, int n, int kind) =>
        Launch1D(K("gated_act_f32"), n, P(gate), P(up), P(y), U(kind), U(n));

    public override void GatedActivationBackward(Storage gate, Storage up, Storage dy, Storage dgate, Storage dup, int n, int kind, int flags) =>
        Launch1D(K("gated_act_bwd_f32"), n, P(gate), P(up), P(dy), P(dgate), P(dup), U(kind), U(flags), U(n));

    public override void GatedActivationPacked(Storage gate, Storage up, Storage packedGate, Storage packedUp, Storage y, Storage packedY, int n, int kind, int flags) =>
        Launch1D(K("gated_act_bf16_f32"), (n + 1) / 2, P(gate), P(up), P(packedGate), P(packedUp), P(y), P(packedY), U(kind), U(flags), U(n), U((n + 1) / 2));

    public override void GatedActivationBackwardPacked(Storage packedGate, Storage packedUp, Storage dy, Storage dgate, Storage dup, int n, int kind, int flags) =>
        Launch1D(K("gated_act_bwd_bf16_f32"), (n + 1) / 2, P(packedGate), P(packedUp), P(dy), P(dgate), P(dup), U(kind), U(flags), U(n), U((n + 1) / 2));

    public override void AttentionTiled(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage? logSumExp, int heads,
        int rowsPerHead, int steps, int capacity, int dim, float scale)
    {
        if (FlashTensorCore(dim) is { } tc)
        {
            Launch(tc[$"flash_tc_fwd_d{dim}"], (uint)((rowsPerHead + PtxKernels.FlashTensorRows - 1) / PtxKernels.FlashTensorRows), (uint)heads, 1, 128, 1,
                [P(q), P(keys), P(values), P(position), P(y), logSumExp is null ? 0UL : P(logSumExp),
                U(rowsPerHead), U(steps), U(capacity), F(scale * Log2E), .. ContiguousLayout(rowsPerHead, steps, capacity, dim)]);
            return;
        }

        if (dim > PtxKernels.FlashMaxDim)
        {
            base.AttentionTiled(q, keys, values, position, y, logSumExp, heads, rowsPerHead, steps, capacity, dim, scale);
            return;
        }

        Launch(K("attention_flash_f32"), (uint)((rowsPerHead + PtxKernels.FlashTile - 1) / PtxKernels.FlashTile), (uint)heads, 1, 128, 1,
            P(q), P(keys), P(values), P(position), P(y), logSumExp is null ? 0UL : P(logSumExp),
            U(rowsPerHead), U(steps), U(capacity), U(dim), F(scale));
    }

    private const float Log2E = 1.4426950408889634f;

    // Strides of the tensor-core flash kernels for contiguous [heads, rowsPerHead, dim] queries / outputs and
    // [heads, capacity, dim] keys / values (kv = 1: every head is its own batch).
    // Packed sequences add each position's sequence start and end and the heads per packed row (0 without).
    private static ulong[] ContiguousLayout(int rowsPerHead, int steps, int capacity, int dim, Storage? starts = null, Storage? ends = null, int headsPerRow = 0) =>
        [U(1), U(rowsPerHead * dim), 0UL, U(steps * dim), U(dim), U(rowsPerHead * dim), 0UL, U(steps * dim), U(dim), U(capacity * dim), 0UL, U(dim),
            starts is null ? 0UL : P(starts), ends is null ? 0UL : P(ends), U(headsPerRow)];

    // Strides for [batch, steps, *] rows (see Backend.AttentionStrided).
    private static ulong[] RowLayout(int kvHeads, int group, int steps, int dim, int qRow, int kRow, int yRow) =>
        [U(kvHeads), U(steps * qRow), U(group * dim), U(dim), U(qRow), U(steps * yRow), U(group * dim), U(dim), U(yRow),
            U(steps * kRow), U(dim), U(kRow), 0UL, 0UL, 0UL];

    private Storage ZeroPosition => _zeroPosition ??= Allocate(1, zeroed: true);

    private Storage? _zeroPosition;

    public override bool AttentionStrided(Storage q, long qOffset, Storage k, long kOffset, Storage v, long vOffset, int qRow, int kRow,
        Storage y, Storage? logSumExp, int batch, int kvHeads, int group, int steps, int dim, float scale)
    {
        if (!PtxKernels.FlashTensorDim(dim) || !FlashKernelsLoaded(dim))
        {
            return false;
        }

        var tc = _tensorCoreAny;
        int heads = batch * kvHeads, rowsPerHead = group * steps;
        Launch(tc[$"flash_tc_fwd_d{dim}"], (uint)((rowsPerHead + PtxKernels.FlashTensorRows - 1) / PtxKernels.FlashTensorRows), (uint)heads, 1, 128, 1,
            [P(q) + (ulong)qOffset * 4, P(k) + (ulong)kOffset * 4, P(v) + (ulong)vOffset * 4, P(ZeroPosition), P(y), logSumExp is null ? 0UL : P(logSumExp),
            U(rowsPerHead), U(steps), U(steps), F(scale * Log2E), .. RowLayout(kvHeads, group, steps, dim, qRow, kRow, kvHeads * group * dim)]);
        return true;
    }

    public override bool AttentionStridedBackward(Storage q, long qOffset, Storage k, long kOffset, Storage v, long vOffset, int qRow, int kRow,
        Storage y, Storage logSumExp, Storage dOutput, Storage dq, long dqOffset, Storage dk, long dkOffset, Storage dv, long dvOffset,
        int batch, int kvHeads, int group, int steps, int dim, float scale)
    {
        if (!PtxKernels.FlashTensorDim(dim) || !FlashKernelsLoaded(dim))
        {
            return false;
        }

        var tc = _tensorCoreAny;
        int heads = batch * kvHeads, rowsPerHead = group * steps, rows = heads * rowsPerHead, yRow = kvHeads * group * dim;
        var layout = RowLayout(kvHeads, group, steps, dim, qRow, kRow, yRow);
        var delta = Allocate(rows, zeroed: false);
        try
        {
            Launch1D(tc["flash_tc_delta"], rows, P(y), P(dOutput), P(delta), U(rowsPerHead), U(steps), U(dim), U(rows),
                U(kvHeads), layout[5], layout[6], layout[7], layout[8]);
            t_sharedBytes = (uint)PtxKernels.FlashTensorBackwardKvShared(dim);
            Launch(tc[$"flash_tc_bwd_kv_d{dim}"], (uint)((steps + PtxKernels.FlashTensorRows - 1) / PtxKernels.FlashTensorRows), (uint)heads, 1, 128, 1,
                [P(q) + (ulong)qOffset * 4, P(k) + (ulong)kOffset * 4, P(v) + (ulong)vOffset * 4, P(dOutput), P(logSumExp), P(delta),
                P(dk) + (ulong)dkOffset * 4, P(dv) + (ulong)dvOffset * 4,
                U(rowsPerHead), U(steps), U(steps), F(scale), F(scale * Log2E), U(group == 1 ? 1 : 0), .. layout]);
            t_sharedBytes = (uint)PtxKernels.FlashTensorBackwardQShared(dim);
            Launch(tc[$"flash_tc_bwd_q_d{dim}"], (uint)((rowsPerHead + PtxKernels.FlashTensorRows - 1) / PtxKernels.FlashTensorRows), (uint)heads, 1, 128, 1,
                [P(q) + (ulong)qOffset * 4, P(k) + (ulong)kOffset * 4, P(v) + (ulong)vOffset * 4, P(dOutput), P(logSumExp), P(delta),
                P(dq) + (ulong)dqOffset * 4, U(rowsPerHead), U(steps), U(steps), F(scale), F(scale * Log2E), .. layout]);
        }
        finally
        {
            delta.Release();
        }

        return true;
    }

    public override void SumColumns(Storage x, long offset, int ld, Storage y, int rows, int cols)
    {
        const int Chunk = 64;
        Launch(K("sum_cols_strided_f32"), (uint)((cols + 255) / 256), (uint)((rows + Chunk - 1) / Chunk), 1, 256, 1,
            P(x) + (ulong)offset * 4, P(y), U(rows), U(cols), U(ld), U(Chunk));
    }

    // The tensor-core flash kernels when MixedPrecision asks for bfloat16 and the head size and GPU allow them.
    private Dictionary<string, IntPtr>? FlashTensorCore(int dim) =>
        MixedPrecision.UsesTensorCores && PtxKernels.FlashTensorDim(dim) && FlashKernelsLoaded(dim) ? _tensorCoreAny : null;

    private bool FlashKernelsLoaded(int dim) =>
        TensorKernel($"flash_tc_fwd_d{dim}") is not null && TensorKernel($"flash_tc_bwd_q_d{dim}") is not null
        && TensorKernel($"flash_tc_bwd_kv_d{dim}") is not null && TensorKernel("flash_tc_delta") is not null;

    public override void AttentionTiledBackward(Storage q, Storage keys, Storage values, Storage output, Storage logSumExp, Storage dOutput,
        Storage dq, Storage dkeys, Storage dvalues, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale)
    {
        int rows = heads * rowsPerHead;
        var delta = Allocate(rows, zeroed: false);
        try
        {
            Launch1D(K("attn_bwd_d_f32"), rows, P(output), P(dOutput), P(delta), U(dim), U(rows));
            if (FlashTensorCore(dim) is { } tc)
            {
                var layout = ContiguousLayout(rowsPerHead, steps, capacity, dim);
                t_sharedBytes = (uint)PtxKernels.FlashTensorBackwardKvShared(dim);
                Launch(tc[$"flash_tc_bwd_kv_d{dim}"], (uint)((capacity + PtxKernels.FlashTensorRows - 1) / PtxKernels.FlashTensorRows), (uint)heads, 1, 128, 1,
                    [P(q), P(keys), P(values), P(dOutput), P(logSumExp), P(delta), P(dkeys), P(dvalues),
                    U(rowsPerHead), U(steps), U(capacity), F(scale), F(scale * Log2E), U(rowsPerHead == steps ? 1 : 0), .. layout]);
                t_sharedBytes = (uint)PtxKernels.FlashTensorBackwardQShared(dim);
                Launch(tc[$"flash_tc_bwd_q_d{dim}"], (uint)((rowsPerHead + PtxKernels.FlashTensorRows - 1) / PtxKernels.FlashTensorRows), (uint)heads, 1, 128, 1,
                    [P(q), P(keys), P(values), P(dOutput), P(logSumExp), P(delta), P(dq),
                    U(rowsPerHead), U(steps), U(capacity), F(scale), F(scale * Log2E), .. layout]);
                return;
            }

            ReadOnlySpan<ulong> args = [P(q), P(keys), P(values), P(dOutput), P(logSumExp), P(delta), P(dq), P(dkeys), P(dvalues),
                U(rowsPerHead), U(steps), U(capacity), U(dim), F(scale)];
            Launch(K("attn_bwd_kv_f32"), (uint)((capacity + 15) / 16), (uint)heads, 1, 128, 1, args);
            Launch(K("attn_bwd_q_f32"), (uint)((rowsPerHead + 31) / 32), (uint)heads, 1, 128, 1, args);
        }
        finally
        {
            delta.Release();
        }
    }

    public override bool SupportsSegmentedAttention(int dim) => FlashTensorCore(dim) is not null;

    public override bool AttentionRows(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage starts, int heads, int headsPerRow,
        int rowsPerHead, int steps, int capacity, int dim, float scale)
    {
        if (FlashTensorCore(dim) is not { } tc)
        {
            return false;
        }

        Launch(tc[$"flash_tc_fwd_d{dim}"], (uint)((rowsPerHead + PtxKernels.FlashTensorRows - 1) / PtxKernels.FlashTensorRows), (uint)heads, 1, 128, 1,
            [P(q), P(keys), P(values), P(position), P(y), 0UL, U(rowsPerHead), U(steps), U(capacity), F(scale * Log2E),
            .. ContiguousLayout(rowsPerHead, steps, capacity, dim, starts, starts, headsPerRow)]);
        return true;
    }

    public override bool AttentionSegmented(Storage q, Storage keys, Storage values, Storage y, Storage? logSumExp, Storage starts, Storage ends,
        int heads, int headsPerRow, int rowsPerHead, int steps, int dim, float scale)
    {
        if (FlashTensorCore(dim) is not { } tc)
        {
            return false;
        }

        Launch(tc[$"flash_tc_fwd_d{dim}"], (uint)((rowsPerHead + PtxKernels.FlashTensorRows - 1) / PtxKernels.FlashTensorRows), (uint)heads, 1, 128, 1,
            [P(q), P(keys), P(values), P(ZeroPosition), P(y), logSumExp is null ? 0UL : P(logSumExp),
            U(rowsPerHead), U(steps), U(steps), F(scale * Log2E), .. ContiguousLayout(rowsPerHead, steps, steps, dim, starts, ends, headsPerRow)]);
        return true;
    }

    public override bool AttentionSegmentedBackward(Storage q, Storage keys, Storage values, Storage output, Storage logSumExp, Storage dOutput,
        Storage dq, Storage dkeys, Storage dvalues, Storage starts, Storage ends, int heads, int headsPerRow, int rowsPerHead, int steps, int dim, float scale)
    {
        if (FlashTensorCore(dim) is not { } tc)
        {
            return false;
        }

        int rows = heads * rowsPerHead;
        var delta = Allocate(rows, zeroed: false);
        try
        {
            Launch1D(K("attn_bwd_d_f32"), rows, P(output), P(dOutput), P(delta), U(dim), U(rows));
            var layout = ContiguousLayout(rowsPerHead, steps, steps, dim, starts, ends, headsPerRow);
            t_sharedBytes = (uint)PtxKernels.FlashTensorBackwardKvShared(dim);
            Launch(tc[$"flash_tc_bwd_kv_d{dim}"], (uint)((steps + PtxKernels.FlashTensorRows - 1) / PtxKernels.FlashTensorRows), (uint)heads, 1, 128, 1,
                [P(q), P(keys), P(values), P(dOutput), P(logSumExp), P(delta), P(dkeys), P(dvalues),
                U(rowsPerHead), U(steps), U(steps), F(scale), F(scale * Log2E), 0UL, .. layout]);
            t_sharedBytes = (uint)PtxKernels.FlashTensorBackwardQShared(dim);
            Launch(tc[$"flash_tc_bwd_q_d{dim}"], (uint)((rowsPerHead + PtxKernels.FlashTensorRows - 1) / PtxKernels.FlashTensorRows), (uint)heads, 1, 128, 1,
                [P(q), P(keys), P(values), P(dOutput), P(logSumExp), P(delta), P(dq),
                U(rowsPerHead), U(steps), U(steps), F(scale), F(scale * Log2E), .. layout]);
        }
        finally
        {
            delta.Release();
        }

        return true;
    }

    public override void AttentionInt8(Storage q, Storage keys, Storage values, Storage keyScales, Storage valueScales, Storage position,
        Storage y, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale, bool tiled)
    {
        int words = (dim + 3) / 4;
        if (tiled && dim <= PtxKernels.FlashMaxDim)
        {
            Launch(K("attention_flash_int8"), (uint)((rowsPerHead + PtxKernels.FlashTile - 1) / PtxKernels.FlashTile), (uint)heads, 1, 128, 1,
                P(q), P(keys), P(values), P(position), P(y), 0UL, U(rowsPerHead), U(steps), U(capacity), U(dim), F(scale),
                P(keyScales), P(valueScales), U(words));
            return;
        }

        if (dim > PtxKernels.DecodeMaxDim)
        {
            throw new NotSupportedException($"Int8 cache attention supports head sizes up to {PtxKernels.DecodeMaxDim} on CUDA.");
        }

        int rows = heads * rowsPerHead;
        DecodeSplit(rows, capacity, dim, y, (splits, part, counters) => Launch(K("attention_decode_int8"), (uint)rows, (uint)splits, 1, PtxKernels.RowThreads, 1,
            P(q), P(keys), P(values), P(keyScales), P(valueScales), P(position), P(y), P(part), P(counters),
            U(rowsPerHead), U(steps), U(capacity), U(dim), F(scale), U(words), U(rows)));
    }

    public override void SoftmaxCrossEntropyRows(Storage logits, Storage targets, Storage weights, Storage losses, int rows, int vocabulary, float scale) =>
        LaunchRows(K("softmax_ce_rows_f32"), rows, PtxKernels.RowThreads * 4, P(logits), P(targets), P(weights), P(losses), U(vocabulary), F(scale), U(rows));

    public override void AttentionBFloat16(Storage q, Storage keys, Storage values, Storage position, Storage y, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale, bool tiled)
    {
        int words = (dim + 1) / 2;
        if (tiled && dim <= PtxKernels.FlashMaxDim)
        {
            Launch(K("attention_flash_bf16"), (uint)((rowsPerHead + PtxKernels.FlashTile - 1) / PtxKernels.FlashTile), (uint)heads, 1, 128, 1,
                P(q), P(keys), P(values), P(position), P(y), 0UL, U(rowsPerHead), U(steps), U(capacity), U(dim), F(scale), U(words));
            return;
        }

        if (dim > PtxKernels.DecodeMaxDim)
        {
            throw new NotSupportedException($"bfloat16 cache attention supports head sizes up to {PtxKernels.DecodeMaxDim} on CUDA.");
        }

        int rows = heads * rowsPerHead;
        DecodeSplit(rows, capacity, dim, y, (splits, part, counters) => Launch(K("attention_decode_bf16"), (uint)rows, (uint)splits, 1, PtxKernels.RowThreads, 1,
            P(q), P(keys), P(values), P(position), P(y), P(part), P(counters), U(rowsPerHead), U(steps), U(capacity), U(dim), F(scale), U(words), U(rows)));
    }

    public override void KeyValueWriteBFloat16(Storage source, Storage cache, Storage position, int heads, int steps, int capacity, int dim)
    {
        int words = (dim + 1) / 2, n = heads * steps * words;
        Launch1D(K("kv_write_bf16"), n, P(source), P(cache), P(position), U(steps), U(capacity), U(dim), U(words), U(n));
    }

    public override void AddRmsNormAffine(Storage a, Storage b, Storage sum, Storage gain, Storage y, int rows, int cols, float eps, float offset) =>
        LaunchRows(K("add_rms_norm_affine_f32"), rows, P(a), P(b), P(sum), P(gain), P(y), U(cols), F(eps), F(offset), U(rows));

    public override void RmsNormRope(Storage x, Storage gain, Storage cos, Storage sin, Storage positions, Storage y, int rows, int cols,
        float eps, float offset, int heads, int steps, int half, bool interleaved) =>
        LaunchRows(K("rms_norm_rope_f32"), rows, P(x), P(gain), P(cos), P(sin), P(positions), P(y),
            U(cols), F(eps), F(offset), U(heads), U(steps), U(half), U(interleaved ? 1 : 0), U(rows));

    public override void RmsNormRopePair(Storage x, Storage gain, Storage y, int rows1, float eps, float offset, int heads,
        Storage x2, Storage gain2, Storage y2, int rows2, float eps2, float offset2, int heads2,
        Storage cos, Storage sin, Storage positions, int cols, int steps, int half, bool interleaved) =>
        LaunchRows(K("rms_norm_rope2_f32"), rows1 + rows2, P(x), P(gain), P(cos), P(sin), P(positions), P(y), P(x2), P(gain2), P(y2),
            U(cols), F(eps), F(offset), U(heads), U(steps), U(half), U(interleaved ? 1 : 0), U(rows1), F(eps2), F(offset2), U(heads2),
            U(rows1 + rows2));

    public override bool NormRopeHeads(Storage q, Storage k, Storage v, int batch, int steps, int heads, int kvHeads, int cols,
        Storage? gainQ, float epsQ, float offsetQ, Storage? gainK, float epsK, float offsetK, Storage? cos, Storage? sin, Storage? positions,
        int half, bool interleaved, Storage yq, Storage yk, Storage yv, Storage? position, int capacity, int stride, bool bfloat16)
    {
        int rows1 = batch * steps * heads, rows2 = batch * steps * kvHeads;
        int flags = (gainQ is not null ? 1 : 0) | (bfloat16 ? 2 : 0) | (position is not null ? 4 : 0);
        static ulong Optional(Storage? s) => s is null ? 0UL : P(s);
        LaunchRows(K("norm_rope_heads_f32"), rows1 + 2 * rows2, P(q), Optional(gainQ), P(k), Optional(gainK), P(v), Optional(cos), Optional(sin),
            Optional(positions), Optional(position), P(yq), P(yk), P(yv), U(cols), F(epsQ), F(offsetQ), U(heads), F(epsK), F(offsetK), U(kvHeads),
            U(steps), U(cos is null ? 0 : half), U(interleaved ? 1 : 0), U(rows1), U(rows2), U(flags), U(capacity), U(stride), U(rows1 + 2 * rows2));
        return true;
    }
}
