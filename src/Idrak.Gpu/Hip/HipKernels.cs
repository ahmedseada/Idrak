// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;

namespace Idrak.Gpu.Hip;

/// <summary>
/// The HIP backend's first kernels, as HIP C++ source compiled at run time by hipRTC for the device's own architecture
/// (<see cref="HipRtc"/>). A small set where a kernel clearly beats the host fallback: element-wise arithmetic, a few
/// activations, row RMS norms and the int8 product decoding runs on. Every other operation stays on the host fallback.
/// The source does not depend on the device: the block size comes in as IDRAK_BLOCK at compilation, chosen from the
/// device's reported limits (<see cref="BlockSizeFor"/>), and nothing in a kernel assumes a wavefront width (reductions
/// go through shared memory).
/// </summary>
internal static class HipKernels
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
    };

    /// <summary>
    /// Threads per block for device limits <paramref name="limits"/>: a quarter of the largest block (several blocks
    /// resident per compute unit, the CUDA backend's rule), a power of two, at least one wavefront, within the block's
    /// x limit, with the reductions' scratch (one float per thread) within the block's shared memory.
    /// </summary>
    public static int BlockSizeFor(HipDeviceLimits limits)
    {
        int largest = Math.Min(limits.MaxThreadsPerBlock, limits.MaxBlockDimX);
        int block = (int)BitOperations.RoundUpToPowerOf2((uint)Math.Max(1, largest / 4));
        if (block > largest / 4)
        {
            block /= 2;                                                          // round down, not up
        }

        while (block > limits.WarpSize && block * sizeof(float) > limits.SharedMemoryPerBlock)
        {
            block /= 2;
        }

        return Math.Max(block, limits.WarpSize);
    }

    /// <summary>The kernels' source (one program; IDRAK_BLOCK is defined by the compilation's options).</summary>
    public const string Source = """
        // Idrak's HIP kernels, compiled at run time by hipRTC for the device's own architecture.
        // IDRAK_BLOCK (threads per block) is defined by the compilation from the device's reported limits.
        #ifndef IDRAK_BLOCK
        #error IDRAK_BLOCK must be defined
        #endif

        #define IDRAK_EACH(i, n) for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < (long long)(n); i += (long long)blockDim.x * gridDim.x)

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
