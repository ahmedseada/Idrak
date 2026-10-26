// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;

namespace Idrak.Gpu.Hip;

/// <summary>
/// The HIP backend's kernels, as HIP C++ source compiled at run time by hipRTC for the device's own architecture
/// (<see cref="HipRtc"/>): element-wise arithmetic and activations, row norms and softmaxes, the float32 products (a
/// tiled product and a few-row one), the packed weight products decoding runs on, attention over key ranges and over a
/// key/value cache, and the gathers, permutes and cache writes around them. Every other operation stays on the host
/// fallback. The source does not depend on the device: the block size comes in as IDRAK_BLOCK at compilation, chosen
/// from the device's reported limits (<see cref="BlockSizeFor"/>), every tile and shared array follows from it, and
/// nothing in a kernel assumes a wavefront width (reductions go through shared memory in a tree of a fixed shape).
/// </summary>
internal static partial class HipKernels
{
    /// <summary>
    /// The kernels and their parameters in order ('p' a device pointer, 'i' a 32-bit integer, 'f' a float): a launch
    /// with another number of arguments is refused before it reaches the runtime.
    /// </summary>
    public static readonly IReadOnlyDictionary<string, string> Parameters = new Dictionary<string, string>
    {
        ["add_f32"] = "pppi",
        ["sub_f32"] = "pppi",
        ["mul_f32"] = "pppi",
        ["max_f32"] = "pppi",
        ["min_f32"] = "pppi",
        ["axpy_f32"] = "ppif",
        ["affine_f32"] = "ppiff",
        ["muladd_f32"] = "pppi",
        ["add_rowvec_f32"] = "pppii",
        ["relu_f32"] = "ppi",
        ["sigmoid_f32"] = "ppi",
        ["tanh_f32"] = "ppi",
        ["square_f32"] = "ppi",
        ["abs_f32"] = "ppi",
        ["exp_f32"] = "ppi",
        ["log_f32"] = "ppi",
        ["rms_norm_f32"] = "pppiif",
        ["rms_norm_affine_f32"] = "pppiiff",
        ["int8_matmul_f32"] = "ppppiiii",

        // HipKernels.Pointwise.cs
        ["gelu_f32"] = "ppi",
        ["gelu_erf_f32"] = "ppi",
        ["silu_f32"] = "ppi",
        ["sqrt_f32"] = "ppi",
        ["sin_f32"] = "ppi",
        ["cos_f32"] = "ppi",
        ["sign_f32"] = "ppi",
        ["bias_gelu_f32"] = "pppii",
        ["gather_f32"] = "pppiii",
        ["gather_bf16_f32"] = "pppiiii",
        ["gather_bf16_cols_f32"] = "pppiiii",
        ["kv_write_f32"] = "pppiiii",
        ["decoder_mask_f32"] = "ppii",
        ["permute_f32"] = "ppiiiiiiiiiiiiii",
        ["copy2d_add_f32"] = "ppiiiiii",
        ["bf16_dequant_f32"] = "ppii",
        ["f16_dequant_f32"] = "ppii",
        ["int4_dequant_f32"] = "pppii",
        ["int8_dequant_f32"] = "pppii",
        ["bf16_pack_f32"] = "ppi",

        // HipKernels.Rows.cs
        ["softmax_f32"] = "ppiii",
        ["scale_mask_softmax_f32"] = "pppiiif",
        ["layer_norm_f32"] = "pppppiif",
        ["layer_norm_backward_f32"] = "pppppii",
        ["layer_norm_params_f32"] = "pppppii",
        ["logsoftmax_topk_f32"] = "pppiii",

        // HipKernels.Products.cs
        ["gemm_f32"] = "ppppiiiiifi",
        ["gemv_f32"] = "ppppiiiif",
        ["packed_gemv_f32"] = "ppppiiii",

        // HipKernels.Attention.cs
        ["attention_row_f32"] = "pppppppiiiiiffiii",
        ["attention_rows_f32"] = "pppppppiiiiiffiii",
    };

    /// <summary>Rows of k the tiled product stages per step (IDRAK_DEPTH).</summary>
    public const int GemmDepth = 16;

    /// <summary>The most rows of x the few-row products take (IDRAK_ROWS; more go through the tiled product).</summary>
    public const int GemvRows = 8;

    /// <summary>The largest head size the attention kernels take (IDRAK_ATT_DIM); larger heads take the host fallback.</summary>
    public const int AttentionMaxDim = 256;

    /// <summary>
    /// Side of the square of threads of the tiled product for a block (IDRAK_SIDE): the largest power of two whose square
    /// is at most the block, at most 32. Each of those threads keeps 4 × 4 outputs, so a block's tile is 4 · side.
    /// </summary>
    public static int Side(int block)
    {
        int side = 1;
        while (side < 32 && (side * 2) * (side * 2) <= block)
        {
            side *= 2;
        }

        return side;
    }

    /// <summary>Outputs along each side of a tiled product's block (IDRAK_TILE).</summary>
    public static int Tile(int block) => 4 * Side(block);

    /// <summary>Consecutive columns (or k) a few-row product's slice reads (IDRAK_LANES): 32, or the whole block below that.</summary>
    public static int Lanes(int block) => Math.Min(32, block);

    /// <summary>Query rows per block of the tiled attention kernel (IDRAK_ATT_ROWS): a thirty-second of the block, 1 to 8.</summary>
    public static int AttentionRows(int block) => block >= 256 ? 8 : block >= 128 ? 4 : block >= 64 ? 2 : 1;

    /// <summary>
    /// The largest static shared memory of any kernel at a block size, in bytes: the reductions' scratch (a float per
    /// thread), the tiled product's two staged tiles, the few-row product's partial sums and the attention kernels'
    /// queries, weights and scratch, the best-k pass's sums (doubles) and pairs.
    /// </summary>
    public static int SharedBytes(int block)
    {
        int gemm = 2 * GemmDepth * (Tile(block) + 1);
        int gemv = Math.Max(GemvRows * block, 4 * block);                      // the partial sums; the best-k pass's sums and pairs
        int rows = AttentionRows(block);
        int attention = rows * AttentionMaxDim + 2 * block + 5 * rows;          // queries, weights, scratch, row stats and ranges
        return Math.Max(Math.Max(gemm, gemv), Math.Max(attention, block)) * sizeof(float);
    }

    /// <summary>
    /// Threads per block for device limits <paramref name="limits"/>: a quarter of the largest block (several blocks
    /// resident per compute unit, the CUDA backend's rule), a power of two, at least one wavefront, within the block's
    /// x limit, halved while the kernels' static shared memory (<see cref="SharedBytes"/>) passes the block's shared memory.
    /// </summary>
    public static int BlockSizeFor(HipDeviceLimits limits)
    {
        int largest = Math.Min(limits.MaxThreadsPerBlock, limits.MaxBlockDimX);
        int block = (int)BitOperations.RoundUpToPowerOf2((uint)Math.Max(1, largest / 4));
        if (block > largest / 4)
        {
            block /= 2;                                                          // round down, not up
        }

        while (block > limits.WarpSize && SharedBytes(block) > limits.SharedMemoryPerBlock)
        {
            block /= 2;
        }

        return Math.Max(block, limits.WarpSize);
    }

    /// <summary>The kernels' source (one program; IDRAK_BLOCK is defined by the compilation's options).</summary>
    public const string Source = Core + Pointwise + Rows + Products + Attention;

    // The common definitions and the first kernels: element-wise arithmetic, a few activations, RMS norms, int8 products.
    private const string Core = """
        // Idrak's HIP kernels, compiled at run time by hipRTC for the device's own architecture.
        // IDRAK_BLOCK (threads per block) is defined by the compilation from the device's reported limits.
        #ifndef IDRAK_BLOCK
        #error IDRAK_BLOCK must be defined
        #endif

        #define IDRAK_EACH(i, n) for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < (long long)(n); i += (long long)blockDim.x * gridDim.x)
        #define IDRAK_NEG_INF (-__builtin_huge_valf())

        __device__ inline int idrak_min(int a, int b)
        {
            return a < b ? a : b;
        }

        __device__ inline int idrak_max(int a, int b)
        {
            return a > b ? a : b;
        }

        // An index stored as a float, read as the other GPUs read it: truncated toward zero and held within [0, last], NaN as 0.
        __device__ inline long long idrak_index(float v, int last)
        {
            long long i = (long long)fminf(fmaxf(v, 0.0f), (float)last);
            return i > last ? (long long)last : i;
        }

        // The sum of v over the block, through shared memory (no assumption about the wavefront's width).
        __device__ float idrak_block_sum(float v, float* scratch)
        {
            scratch[threadIdx.x] = v;
            __syncthreads();
            for (int s = IDRAK_BLOCK / 2; s > 0; s >>= 1)
            {
                if ((int)threadIdx.x < s)
                {
                    scratch[threadIdx.x] += scratch[threadIdx.x + s];
                }

                __syncthreads();
            }

            float total = scratch[0];
            __syncthreads();
            return total;
        }

        // The largest v over the block (NaN ignored: a row holding one gives NaN softmaxes either way), as idrak_block_sum.
        __device__ float idrak_block_max(float v, float* scratch)
        {
            scratch[threadIdx.x] = v;
            __syncthreads();
            for (int s = IDRAK_BLOCK / 2; s > 0; s >>= 1)
            {
                if ((int)threadIdx.x < s)
                {
                    scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + s]);
                }

                __syncthreads();
            }

            float total = scratch[0];
            __syncthreads();
            return total;
        }

        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) add_f32(const float* a, const float* b, float* c, int n)
        {
            IDRAK_EACH(i, n) c[i] = a[i] + b[i];
        }

        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) sub_f32(const float* a, const float* b, float* c, int n)
        {
            IDRAK_EACH(i, n) c[i] = a[i] - b[i];
        }

        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) mul_f32(const float* a, const float* b, float* c, int n)
        {
            IDRAK_EACH(i, n) c[i] = a[i] * b[i];
        }

        // c = max(a, b) and min(a, b) as MathF.Max and MathF.Min: a NaN a gives a, a NaN b gives b; of +0 and -0, max is +0
        // and min is -0.
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) max_f32(const float* a, const float* b, float* c, int n)
        {
            IDRAK_EACH(i, n)
            {
                float x = a[i], y = b[i];
                float r = y < x ? x : y;
                if (x == y) r = __float_as_int(y) < 0 ? x : y;
                c[i] = x != x ? x : r;
            }
        }

        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) min_f32(const float* a, const float* b, float* c, int n)
        {
            IDRAK_EACH(i, n)
            {
                float x = a[i], y = b[i];
                float r = x < y ? x : y;
                if (x == y) r = __float_as_int(x) < 0 ? x : y;
                c[i] = x != x ? x : r;
            }
        }

        // y += alpha x
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) axpy_f32(const float* x, float* y, int n, float alpha)
        {
            IDRAK_EACH(i, n) y[i] = fmaf(x[i], alpha, y[i]);
        }

        // y = alpha x + beta
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) affine_f32(const float* x, float* y, int n, float alpha, float beta)
        {
            IDRAK_EACH(i, n) y[i] = fmaf(x[i], alpha, beta);
        }

        // c += a b
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) muladd_f32(const float* a, const float* b, float* c, int n)
        {
            IDRAK_EACH(i, n) c[i] = fmaf(a[i], b[i], c[i]);
        }

        // c[r, j] = a[r, j] + v[j]
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) add_rowvec_f32(const float* a, const float* v, float* c, int rows, int cols)
        {
            IDRAK_EACH(i, (long long)rows * cols) c[i] = a[i] + v[i % cols];
        }

        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) relu_f32(const float* x, float* y, int n)
        {
            IDRAK_EACH(i, n) y[i] = fmaxf(x[i], 0.0f);
        }

        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) sigmoid_f32(const float* x, float* y, int n)
        {
            IDRAK_EACH(i, n) y[i] = 1.0f / (1.0f + expf(-x[i]));
        }

        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) tanh_f32(const float* x, float* y, int n)
        {
            IDRAK_EACH(i, n) y[i] = tanhf(x[i]);
        }

        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) square_f32(const float* x, float* y, int n)
        {
            IDRAK_EACH(i, n) y[i] = x[i] * x[i];
        }

        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) abs_f32(const float* x, float* y, int n)
        {
            IDRAK_EACH(i, n) y[i] = fabsf(x[i]);
        }

        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) exp_f32(const float* x, float* y, int n)
        {
            IDRAK_EACH(i, n) y[i] = expf(x[i]);
        }

        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) log_f32(const float* x, float* y, int n)
        {
            IDRAK_EACH(i, n) y[i] = logf(x[i]);
        }

        // One block per row: y = x / sqrt(mean(x²) + eps), inv[row] = that scale.
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) rms_norm_f32(const float* x, float* y, float* inv, int rows, int cols, float eps)
        {
            __shared__ float scratch[IDRAK_BLOCK];
            int row = blockIdx.x;
            const float* xr = x + (long long)row * cols;
            float* yr = y + (long long)row * cols;
            float sum = 0.0f;
            for (int j = threadIdx.x; j < cols; j += IDRAK_BLOCK)
            {
                sum = fmaf(xr[j], xr[j], sum);
            }

            float scale = 1.0f / sqrtf(idrak_block_sum(sum, scratch) / cols + eps);
            if (threadIdx.x == 0)
            {
                inv[row] = scale;
            }

            for (int j = threadIdx.x; j < cols; j += IDRAK_BLOCK)
            {
                yr[j] = xr[j] * scale;
            }
        }

        // One block per row: y = x / sqrt(mean(x²) + eps) · (gain + offset).
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) rms_norm_affine_f32(const float* x, const float* gain, float* y, int rows, int cols, float eps, float offset)
        {
            __shared__ float scratch[IDRAK_BLOCK];
            int row = blockIdx.x;
            const float* xr = x + (long long)row * cols;
            float* yr = y + (long long)row * cols;
            float sum = 0.0f;
            for (int j = threadIdx.x; j < cols; j += IDRAK_BLOCK)
            {
                sum = fmaf(xr[j], xr[j], sum);
            }

            float scale = 1.0f / sqrtf(idrak_block_sum(sum, scratch) / cols + eps);
            for (int j = threadIdx.x; j < cols; j += IDRAK_BLOCK)
            {
                yr[j] = xr[j] * scale * (gain[j] + offset);
            }
        }

        // y[r, j] = scales[j] · Σ_k x[r, k] q[k, j], q int8 in rows of `stride` bytes (n rounded up to 4): one thread per
        // output, consecutive threads reading consecutive bytes of a weight row.
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) int8_matmul_f32(const float* x, const signed char* q, const float* scales, float* y, int m, int n, int k, int stride)
        {
            int j = blockIdx.x * IDRAK_BLOCK + threadIdx.x;
            int r = blockIdx.y;
            if (j >= n || r >= m)
            {
                return;
            }

            const float* xr = x + (long long)r * k;
            float acc = 0.0f;
            for (int kk = 0; kk < k; kk++)
            {
                acc = fmaf(xr[kk], (float)q[(long long)kk * stride + j], acc);
            }

            y[(long long)r * n + j] = acc * scales[j];
        }

        """;
}
