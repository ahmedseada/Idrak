// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Hip;

// Row kernels, one block per row: softmax and log-softmax, the attention's scaled and masked softmax, layer norm with its
// statistics, and the layer norm's gradients. Each row's sums go through idrak_block_sum (a tree of a fixed shape), so a
// row's result does not depend on the grid; every pass reads the row again rather than keeping it, so a row may be as
// long as the vocabulary, and a row may be written in place (each value is read before the same thread writes it).
internal static partial class HipKernels
{
    private const string Rows = """
        // y[r, :] = softmax(x[r, :]), or with useLog the log-softmax x - (log Σ exp(x - max) + max).
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) softmax_f32(const float* x, float* y, int rows, int cols, int useLog)
        {
            __shared__ float scratch[IDRAK_BLOCK];
            const float* xr = x + (long long)blockIdx.x * cols;
            float* yr = y + (long long)blockIdx.x * cols;
            float m = IDRAK_NEG_INF;
            for (int j = threadIdx.x; j < cols; j += IDRAK_BLOCK)
            {
                m = fmaxf(m, xr[j]);
            }

            m = idrak_block_max(m, scratch);
            float s = 0.0f;
            for (int j = threadIdx.x; j < cols; j += IDRAK_BLOCK)
            {
                s += expf(xr[j] - m);
            }

            s = idrak_block_sum(s, scratch);
            if (useLog != 0)
            {
                float shift = logf(s) + m;
                for (int j = threadIdx.x; j < cols; j += IDRAK_BLOCK)
                {
                    yr[j] = xr[j] - shift;
                }
            }
            else
            {
                float inverse = 1.0f / s;
                for (int j = threadIdx.x; j < cols; j += IDRAK_BLOCK)
                {
                    yr[j] = expf(xr[j] - m) * inverse;
                }
            }
        }

        // y[r, :] = softmax(scale · x[r, :] + mask[r % maskRows, :]), no mask when mask is null.
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) scale_mask_softmax_f32(const float* x, const float* mask, float* y, int rows, int cols, int maskRows, float scale)
        {
            __shared__ float scratch[IDRAK_BLOCK];
            const float* xr = x + (long long)blockIdx.x * cols;
            float* yr = y + (long long)blockIdx.x * cols;
            const float* mr = mask != 0 ? mask + (long long)(blockIdx.x % maskRows) * cols : 0;
            float m = IDRAK_NEG_INF;
            for (int j = threadIdx.x; j < cols; j += IDRAK_BLOCK)
            {
                m = fmaxf(m, xr[j] * scale + (mr != 0 ? mr[j] : 0.0f));
            }

            m = idrak_block_max(m, scratch);
            float s = 0.0f;
            for (int j = threadIdx.x; j < cols; j += IDRAK_BLOCK)
            {
                s += expf(xr[j] * scale + (mr != 0 ? mr[j] : 0.0f) - m);
            }

            float inverse = 1.0f / idrak_block_sum(s, scratch);
            for (int j = threadIdx.x; j < cols; j += IDRAK_BLOCK)
            {
                yr[j] = expf(xr[j] * scale + (mr != 0 ? mr[j] : 0.0f) - m) * inverse;
            }
        }

        // y[r, :] = (x[r, :] - mean) / sqrt(var + eps) · gamma + beta over the row; with stats (not null) also stats[r] = mean
        // and stats[rows + r] = 1 / sqrt(var + eps).
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) layer_norm_f32(const float* x, const float* gamma, const float* beta, float* y, float* stats, int rows, int cols, float eps)
        {
            __shared__ float scratch[IDRAK_BLOCK];
            int row = blockIdx.x;
            const float* xr = x + (long long)row * cols;
            float* yr = y + (long long)row * cols;
            float sum = 0.0f;
            for (int j = threadIdx.x; j < cols; j += IDRAK_BLOCK)
            {
                sum += xr[j];
            }

            float mean = idrak_block_sum(sum, scratch) / cols;
            float deviation = 0.0f;
            for (int j = threadIdx.x; j < cols; j += IDRAK_BLOCK)
            {
                float d = xr[j] - mean;
                deviation = fmaf(d, d, deviation);
            }

            float inv = 1.0f / sqrtf(idrak_block_sum(deviation, scratch) / cols + eps);
            if (stats != 0 && threadIdx.x == 0)
            {
                stats[row] = mean;
                stats[rows + row] = inv;
            }

            for (int j = threadIdx.x; j < cols; j += IDRAK_BLOCK)
            {
                yr[j] = fmaf((xr[j] - mean) * inv, gamma[j], beta[j]);
            }
        }

        // dx += rstd · (g - mean(g) - xhat · mean(g · xhat)) over each row, g = dy · gamma, xhat = (x - mean) · rstd, the mean and
        // rstd read from stats as layer_norm_f32 stores them.
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) layer_norm_backward_f32(const float* x, const float* gamma, const float* dy, const float* stats, float* dx, int rows, int cols)
        {
            __shared__ float scratch[IDRAK_BLOCK];
            int row = blockIdx.x;
            const float* xr = x + (long long)row * cols;
            const float* gr = dy + (long long)row * cols;
            float* dr = dx + (long long)row * cols;
            float mean = stats[row], rstd = stats[rows + row];
            float s1 = 0.0f, s2 = 0.0f;
            for (int j = threadIdx.x; j < cols; j += IDRAK_BLOCK)
            {
                float g = gr[j] * gamma[j];
                s1 += g;
                s2 = fmaf(g, (xr[j] - mean) * rstd, s2);
            }

            s1 = idrak_block_sum(s1, scratch) / cols;
            s2 = idrak_block_sum(s2, scratch) / cols;
            for (int j = threadIdx.x; j < cols; j += IDRAK_BLOCK)
            {
                float g = gr[j] * gamma[j], xhat = (xr[j] - mean) * rstd;
                dr[j] = fmaf(rstd, g - s1 - xhat * s2, dr[j]);
            }
        }

        // dgamma[j] += Σ_r dy[r, j] · xhat[r, j] and dbeta[j] += Σ_r dy[r, j] (either may be null), each column's sums over the
        // rows in order, in double as the CPU adds them: one thread per column, consecutive threads reading consecutive columns.
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) layer_norm_params_f32(const float* x, const float* dy, const float* stats, float* dgamma, float* dbeta, int rows, int cols)
        {
            IDRAK_EACH(j, cols)
            {
                double sg = 0.0, sb = 0.0;
                for (int r = 0; r < rows; r++)
                {
                    long long at = (long long)r * cols + j;
                    float d = dy[at];
                    sg += (double)(d * ((x[at] - stats[r]) * stats[rows + r]));
                    sb += (double)d;
                }

                if (dgamma != 0)
                {
                    dgamma[j] += (float)sg;
                }

                if (dbeta != 0)
                {
                    dbeta[j] += (float)sb;
                }
            }
        }

        // The best k log-probabilities of each row and their columns (a beam search's candidates), as the CPU picks them: shift =
        // max + log Σ exp(x - max) (the max over the entries that are not NaN, the sum in double over those other than -inf,
        // the shift rounded to float once); each entry's log-probability x - shift; values[r, j] the row's j-th best of those
        // that are neither NaN nor -inf (larger first, the smaller column first on equal values) and indices[r, j] its column,
        // -inf and -1 past the last. Pass j finds the best entry after pass j - 1's in that order: each thread its own, then a
        // tree of a fixed shape over the block's (value, column) pairs.
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) logsoftmax_topk_f32(const float* x, float* values, float* indices, int rows, int cols, int k)
        {
            __shared__ double sums[IDRAK_BLOCK];
            __shared__ float best[IDRAK_BLOCK];
            __shared__ int where[IDRAK_BLOCK];
            int tid = threadIdx.x;
            const float* xr = x + (long long)blockIdx.x * cols;
            float* vr = values + (long long)blockIdx.x * k;
            float* ir = indices + (long long)blockIdx.x * k;
            float m = IDRAK_NEG_INF;
            for (int j = tid; j < cols; j += IDRAK_BLOCK)
            {
                m = fmaxf(m, xr[j]);
            }

            best[tid] = m;
            __syncthreads();
            for (int s = IDRAK_BLOCK / 2; s > 0; s >>= 1)
            {
                if (tid < s)
                {
                    best[tid] = fmaxf(best[tid], best[tid + s]);
                }

                __syncthreads();
            }

            m = best[0];
            __syncthreads();
            float shift = __builtin_nanf("");
            if (m != IDRAK_NEG_INF)
            {
                double sum = 0.0;
                for (int j = tid; j < cols; j += IDRAK_BLOCK)
                {
                    float v = xr[j];
                    if (v != IDRAK_NEG_INF)
                    {
                        sum += exp((double)(v - m));
                    }
                }

                sums[tid] = sum;
                __syncthreads();
                for (int s = IDRAK_BLOCK / 2; s > 0; s >>= 1)
                {
                    if (tid < s)
                    {
                        sums[tid] += sums[tid + s];
                    }

                    __syncthreads();
                }

                shift = (float)((double)m + log(sums[0]));
            }

            float lastValue = 0.0f;
            int lastColumn = -1;
            for (int pass = 0; pass < k; pass++)
            {
                float mine = IDRAK_NEG_INF;
                int column = -1;
                if (lastColumn >= 0 || pass == 0)
                {
                    for (int j = tid; j < cols; j += IDRAK_BLOCK)
                    {
                        float v = xr[j] - shift;
                        bool after = pass == 0 || v < lastValue || v == lastValue && j > lastColumn;
                        if (v == v && v != IDRAK_NEG_INF && after && (column < 0 || v > mine))
                        {
                            mine = v;
                            column = j;
                        }
                    }
                }

                best[tid] = mine;
                where[tid] = column;
                __syncthreads();
                for (int s = IDRAK_BLOCK / 2; s > 0; s >>= 1)
                {
                    if (tid < s)
                    {
                        int other = where[tid + s];
                        float ov = best[tid + s];
                        int here = where[tid];
                        if (other >= 0 && (here < 0 || ov > best[tid] || ov == best[tid] && other < here))
                        {
                            best[tid] = ov;
                            where[tid] = other;
                        }
                    }

                    __syncthreads();
                }

                lastValue = best[0];
                lastColumn = where[0];
                __syncthreads();
                if (tid == 0)
                {
                    vr[pass] = lastColumn >= 0 ? lastValue : IDRAK_NEG_INF;
                    ir[pass] = (float)lastColumn;
                }
            }
        }

        """;
}
