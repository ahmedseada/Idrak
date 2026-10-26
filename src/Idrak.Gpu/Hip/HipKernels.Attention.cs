// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Hip;

// Attention without storing the scores: a block takes R query rows of one head and goes through the keys they see a tile
// at a time with an online softmax (the CPU's tiled attention, Backend.AttentionSpans and Backend.AttentionTiled). One
// kernel serves both: each row's range of keys read from the device (key spans: starts and ends per row; causal: the
// position in device memory, the row's step and the window), so decoding never reads anything back.
internal static partial class HipKernels
{
    private const string Attention = """
        // The attention kernels' shape: heads of at most IDRAK_ATT_DIM; attention_row_f32 takes one query row a block (tiles
        // of IDRAK_BLOCK keys), attention_rows_f32 IDRAK_ATT_ROWS rows (tiles of IDRAK_BLOCK / IDRAK_ATT_ROWS keys).
        #define IDRAK_ATT_DIM 256
        #if IDRAK_BLOCK >= 256
        #define IDRAK_ATT_ROWS 8
        #elif IDRAK_BLOCK >= 128
        #define IDRAK_ATT_ROWS 4
        #elif IDRAK_BLOCK >= 64
        #define IDRAK_ATT_ROWS 2
        #else
        #define IDRAK_ATT_ROWS 1
        #endif

        // Query rows r0 .. r0 + R - 1 of head h = blockIdx.y (r0 = blockIdx.x · R) of q [heads, rows, dim] over keys and
        // values [heads / group, keyRows, dim] (head h reads key/value head h / group). Row i sees the keys from its range's
        // start up to its end, the end excluded: causal = 0: starts[b · rows + i] to ends[b · rows + i], both held within
        // 0 .. keyRows, b = h / headsPerTable; causal != 0: begin to end with end = min(starts[0] + i % steps, keyRows - 1) + 1
        // and begin = window > 0 ? max(0, end - window) : 0.
        // Its scores scale · q · k are soft-capped when softcap > 0 (softcap · tanh(s / softcap)). Thread (rr, cc) =
        // (tid / KT, tid % KT) scores key c0 + cc of row rr in each tile of KT keys; each row's tile maximum and sum go
        // through a tree of a fixed shape over its KT threads; the outputs (R · dim, each owned by one thread) are rescaled
        // and add the tile's weighted values in key order. A row with an empty range gets zeros (and a log-sum-exp of -inf),
        // else y = Σ exp(s - max) v / Σ exp(s - max) and lse (when not null) = max + log Σ exp(s - max).
        template <int R>
        __device__ inline void idrak_attention(const float* q, const float* keys, const float* values, const float* starts, const float* ends, float* y,
            float* lse, int group, int headsPerTable, int rows, int keyRows, int dim, float scale, float softcap, int causal, int steps, int window,
            float* qs, float* ps, float* red, float* stats, int* spans)
        {
            const int KT = IDRAK_BLOCK / R;
            const int OUT = (R * IDRAK_ATT_DIM + IDRAK_BLOCK - 1) / IDRAK_BLOCK;
            int tid = threadIdx.x, rr = tid / KT, cc = tid % KT;
            int h = blockIdx.y, r0 = blockIdx.x * R;
            const float* kb = keys + (long long)(h / group) * keyRows * dim;
            const float* vb = values + (long long)(h / group) * keyRows * dim;
            if (tid < R)
            {
                int row = r0 + tid, s = 0, e = 0;
                if (row < rows)
                {
                    if (causal != 0)
                    {
                        e = idrak_min((int)starts[0] + row % steps, keyRows - 1) + 1;
                        s = window > 0 ? idrak_max(0, e - window) : 0;
                    }
                    else
                    {
                        long long at = (long long)(h / headsPerTable) * rows + row;
                        s = (int)idrak_index(starts[at], keyRows);
                        e = (int)idrak_index(ends[at], keyRows);
                    }
                }

                spans[tid] = s;
                spans[R + tid] = e;
                stats[tid] = IDRAK_NEG_INF;
                stats[R + tid] = 0.0f;
                stats[2 * R + tid] = 1.0f;
            }

            const float* qb = q + ((long long)h * rows + r0) * dim;
            for (int o = tid; o < R * dim; o += IDRAK_BLOCK)
            {
                qs[o] = r0 + o / dim < rows ? qb[o] : 0.0f;
            }

            float acc[OUT];
            #pragma unroll
            for (int j = 0; j < OUT; j++)
            {
                acc[j] = 0.0f;
            }

            __syncthreads();
            int lo = keyRows, hi = 0;
            for (int i = 0; i < R; i++)
            {
                if (spans[R + i] > spans[i])
                {
                    lo = idrak_min(lo, spans[i]);
                    hi = idrak_max(hi, spans[R + i]);
                }
            }

            int mine = spans[rr], mineEnd = spans[R + rr];
            for (int c0 = lo; c0 < hi; c0 += KT)
            {
                int c = c0 + cc;
                bool seen = c >= mine && c < mineEnd;
                float score = IDRAK_NEG_INF;
                if (seen)
                {
                    const float* key = kb + (long long)c * dim;
                    const float* qr = qs + rr * dim;
                    float dot = 0.0f;
                    #pragma unroll 4
                    for (int d = 0; d < dim; d++)
                    {
                        dot = fmaf(qr[d], key[d], dot);
                    }

                    score = dot * scale;
                    if (softcap > 0.0f)
                    {
                        score = tanhf(score / softcap) * softcap;
                    }
                }

                red[tid] = score;
                __syncthreads();
                for (int s = KT / 2; s > 0; s >>= 1)
                {
                    if (cc < s)
                    {
                        red[tid] = fmaxf(red[tid], red[tid + s]);
                    }

                    __syncthreads();
                }

                bool any = idrak_max(mine, c0) < idrak_min(mineEnd, c0 + KT);
                float old = stats[rr];
                float updated = any ? fmaxf(old, red[rr * KT]) : old;
                float p = seen ? expf(score - updated) : 0.0f;
                ps[tid] = p;
                __syncthreads();
                red[tid] = p;
                __syncthreads();
                for (int s = KT / 2; s > 0; s >>= 1)
                {
                    if (cc < s)
                    {
                        red[tid] += red[tid + s];
                    }

                    __syncthreads();
                }

                if (cc == 0)
                {
                    float alpha = any ? expf(old - updated) : 1.0f;
                    stats[rr] = updated;
                    stats[R + rr] = stats[R + rr] * alpha + red[tid];
                    stats[2 * R + rr] = alpha;
                }

                __syncthreads();
                int tileEnd = idrak_min(c0 + KT, hi);
                #pragma unroll
                for (int j = 0; j < OUT; j++)
                {
                    int o = tid + j * IDRAK_BLOCK;
                    if (o < R * dim)
                    {
                        int ro = o / dim, d = o - ro * dim;
                        int from = idrak_max(spans[ro], c0), to = idrak_min(spans[R + ro], tileEnd);
                        float v = acc[j] * stats[2 * R + ro];
                        for (int key = from; key < to; key++)
                        {
                            v = fmaf(ps[ro * KT + key - c0], vb[(long long)key * dim + d], v);
                        }

                        acc[j] = v;
                    }
                }

                __syncthreads();
            }

            #pragma unroll
            for (int j = 0; j < OUT; j++)
            {
                int o = tid + j * IDRAK_BLOCK;
                if (o < R * dim)
                {
                    int ro = o / dim, d = o - ro * dim;
                    if (r0 + ro < rows)
                    {
                        float total = stats[R + ro];
                        y[((long long)h * rows + r0 + ro) * dim + d] = total > 0.0f ? acc[j] * (1.0f / total) : 0.0f;
                    }
                }
            }

            if (lse != 0 && tid < R && r0 + tid < rows)
            {
                float total = stats[R + tid];
                lse[(long long)h * rows + r0 + tid] = total > 0.0f ? stats[tid] + logf(total) : IDRAK_NEG_INF;
            }
        }

        // One query row a block (decoding: a step's few rows over a long cache or memory), tiles of IDRAK_BLOCK keys.
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) attention_row_f32(const float* q, const float* keys, const float* values, const float* starts, const float* ends, float* y, float* lse, int group, int headsPerTable, int rows, int keyRows, int dim, float scale, float softcap, int causal, int steps, int window)
        {
            __shared__ float qs[IDRAK_ATT_DIM];
            __shared__ float ps[IDRAK_BLOCK];
            __shared__ float red[IDRAK_BLOCK];
            __shared__ float stats[3];
            __shared__ int spans[2];
            idrak_attention<1>(q, keys, values, starts, ends, y, lse, group, headsPerTable, rows, keyRows, dim, scale, softcap, causal, steps, window,
                qs, ps, red, stats, spans);
        }

        // IDRAK_ATT_ROWS query rows a block sharing each key and value read (a prompt, an encoder), tiles of
        // IDRAK_BLOCK / IDRAK_ATT_ROWS keys.
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) attention_rows_f32(const float* q, const float* keys, const float* values, const float* starts, const float* ends, float* y, float* lse, int group, int headsPerTable, int rows, int keyRows, int dim, float scale, float softcap, int causal, int steps, int window)
        {
            __shared__ float qs[IDRAK_ATT_ROWS * IDRAK_ATT_DIM];
            __shared__ float ps[IDRAK_BLOCK];
            __shared__ float red[IDRAK_BLOCK];
            __shared__ float stats[3 * IDRAK_ATT_ROWS];
            __shared__ int spans[2 * IDRAK_ATT_ROWS];
            idrak_attention<IDRAK_ATT_ROWS>(q, keys, values, starts, ends, y, lse, group, headsPerTable, rows, keyRows, dim, scale, softcap, causal, steps, window,
                qs, ps, red, stats, spans);
        }

        """;
}
