// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Hip;

// The float32 products and the packed weight products of few rows. Every output adds its k terms in an order fixed by the
// sizes: the tiled product in k order, the few-row products a slice of k per group of threads (k, k + slices, ...) and the
// slices' sums added in slice order by one thread, never by atomics or by the device's compute units (rule 86).
internal static partial class HipKernels
{
    private const string Products = """
        // The tiled product's shape, from the block: IDRAK_SIDE² threads (the largest power of two whose square fits the
        // block, at most 32²; any others only meet the barriers) each keep 4 × 4 outputs of an IDRAK_TILE² block of c, and
        // the block stages IDRAK_DEPTH rows of k of both operands per step.
        #if IDRAK_BLOCK >= 1024
        #define IDRAK_SIDE 32
        #elif IDRAK_BLOCK >= 256
        #define IDRAK_SIDE 16
        #elif IDRAK_BLOCK >= 64
        #define IDRAK_SIDE 8
        #elif IDRAK_BLOCK >= 16
        #define IDRAK_SIDE 4
        #elif IDRAK_BLOCK >= 4
        #define IDRAK_SIDE 2
        #else
        #define IDRAK_SIDE 1
        #endif
        #define IDRAK_TILE (4 * IDRAK_SIDE)
        #define IDRAK_DEPTH 16
        #define IDRAK_PITCH (IDRAK_TILE + 1)
        #define IDRAK_LOADS (IDRAK_TILE * IDRAK_DEPTH / (IDRAK_SIDE * IDRAK_SIDE))

        // c = op(a) · op(b) (+ beta · c) (+ bias[col]) for a [m, k] (transA: stored [k, m]) and b [k, n] (transB: [n, k]),
        // batch products of contiguous operands stacked along grid y, rowTiles row tiles each (bias null: none; beta = 0
        // writes c without reading it). Per step the block stages op(a)[IDRAK_TILE × IDRAK_DEPTH] and
        // op(b)[IDRAK_DEPTH × IDRAK_TILE] (zeros outside), consecutive threads reading consecutive addresses of the operand as
        // it is stored, transposed or not; the tiles' rows are one float longer than a tile, so stores along k and reads
        // along m or n fall in different banks. Each staged element is written by one thread between the barrier ending
        // the previous step's reads and the barrier before this step's. Every output adds its k terms in order.
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) gemm_f32(const float* a, const float* b, float* c, const float* bias, int m, int n, int k, int transA, int transB, float beta, int rowTiles)
        {
            __shared__ float tileA[IDRAK_DEPTH * IDRAK_PITCH];
            __shared__ float tileB[IDRAK_DEPTH * IDRAK_PITCH];
            int tid = threadIdx.x;
            bool active = tid < IDRAK_SIDE * IDRAK_SIDE;
            int tx = tid % IDRAK_SIDE, ty = tid / IDRAK_SIDE;
            int batch = blockIdx.y / rowTiles;
            int rowBase = blockIdx.y % rowTiles * IDRAK_TILE, colBase = blockIdx.x * IDRAK_TILE;
            const float* ab = a + (long long)batch * m * k;
            const float* bb = b + (long long)batch * k * n;
            float acc[4][4];
            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                #pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    acc[i][j] = 0.0f;
                }
            }

            for (int t = 0; t < k; t += IDRAK_DEPTH)
            {
                if (active)
                {
                    #pragma unroll
                    for (int p = 0; p < IDRAK_LOADS; p++)
                    {
                        int l = tid + p * IDRAK_SIDE * IDRAK_SIDE;
                        int r = transA != 0 ? l % IDRAK_TILE : l / IDRAK_DEPTH;
                        int ka = transA != 0 ? l / IDRAK_TILE : l % IDRAK_DEPTH;
                        int col = transB != 0 ? l / IDRAK_DEPTH : l % IDRAK_TILE;
                        int kb = transB != 0 ? l % IDRAK_DEPTH : l / IDRAK_TILE;
                        int row = rowBase + r, kka = t + ka, column = colBase + col, kkb = t + kb;
                        float va = 0.0f, vb = 0.0f;
                        if (row < m && kka < k)
                        {
                            va = transA != 0 ? ab[(long long)kka * m + row] : ab[(long long)row * k + kka];
                        }

                        if (kkb < k && column < n)
                        {
                            vb = transB != 0 ? bb[(long long)column * k + kkb] : bb[(long long)kkb * n + column];
                        }

                        tileA[ka * IDRAK_PITCH + r] = va;
                        tileB[kb * IDRAK_PITCH + col] = vb;
                    }
                }

                __syncthreads();
                if (active)
                {
                    #pragma unroll
                    for (int e = 0; e < IDRAK_DEPTH; e++)
                    {
                        float av[4], bv[4];
                        #pragma unroll
                        for (int i = 0; i < 4; i++)
                        {
                            av[i] = tileA[e * IDRAK_PITCH + i * IDRAK_SIDE + ty];
                            bv[i] = tileB[e * IDRAK_PITCH + i * IDRAK_SIDE + tx];
                        }

                        #pragma unroll
                        for (int i = 0; i < 4; i++)
                        {
                            #pragma unroll
                            for (int j = 0; j < 4; j++)
                            {
                                acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
                            }
                        }
                    }
                }

                __syncthreads();
            }

            if (!active)
            {
                return;
            }

            float* cb = c + (long long)batch * m * n;
            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                #pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    int row = rowBase + ty + i * IDRAK_SIDE, col = colBase + tx + j * IDRAK_SIDE;
                    if (row < m && col < n)
                    {
                        long long o = (long long)row * n + col;
                        float v = acc[i][j];
                        if (bias != 0)
                        {
                            v += bias[col];
                        }

                        cb[o] = beta == 0.0f ? v : fmaf(beta, cb[o], v);
                    }
                }
            }
        }

        // The few-row products' shape: IDRAK_ROWS rows at most; IDRAK_LANES consecutive threads read consecutive addresses
        // (32, or the whole block when it is smaller), IDRAK_GROUPS = IDRAK_BLOCK / IDRAK_LANES such groups per block.
        #define IDRAK_ROWS 8
        #if IDRAK_BLOCK >= 32
        #define IDRAK_LANES 32
        #else
        #define IDRAK_LANES IDRAK_BLOCK
        #endif
        #define IDRAK_GROUPS (IDRAK_BLOCK / IDRAK_LANES)

        // y[r, j] = Σ_kk x[r, kk] · b(kk, j) (+ beta · y) (+ bias[j]) for r < m ≤ IDRAK_ROWS, x [m, k] as stored and b [k, n]
        // (transB: [n, k]); batch products along grid y. b is read once, coalesced: as stored, each group's lanes read
        // consecutive columns of a row of b (a block covers IDRAK_LANES columns, its groups slices of k); transposed, each
        // group takes a column and its lanes consecutive k of it (a block covers IDRAK_GROUPS columns). Each thread keeps a sum
        // per row of x; the slices' sums of an output are added in slice order by one thread (as stored) or in a tree over the
        // lanes (transposed).
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) gemv_f32(const float* x, const float* b, float* y, const float* bias, int m, int n, int k, int transB, float beta)
        {
            __shared__ float part[IDRAK_ROWS * IDRAK_BLOCK];
            int tid = threadIdx.x, lane = tid % IDRAK_LANES, group = tid / IDRAK_LANES;
            const float* xb = x + (long long)blockIdx.y * m * k;
            const float* bb = b + (long long)blockIdx.y * k * n;
            float* yb = y + (long long)blockIdx.y * m * n;
            int j = transB != 0 ? blockIdx.x * IDRAK_GROUPS + group : blockIdx.x * IDRAK_LANES + lane;
            int first = transB != 0 ? lane : group, step = transB != 0 ? IDRAK_LANES : IDRAK_GROUPS;
            float acc[IDRAK_ROWS];
            #pragma unroll
            for (int r = 0; r < IDRAK_ROWS; r++)
            {
                acc[r] = 0.0f;
            }

            if (j < n)
            {
                #pragma unroll 4
                for (int kk = first; kk < k; kk += step)
                {
                    float w = transB != 0 ? bb[(long long)j * k + kk] : bb[(long long)kk * n + j];
                    #pragma unroll
                    for (int r = 0; r < IDRAK_ROWS; r++)
                    {
                        if (r < m)
                        {
                            acc[r] = fmaf(xb[(long long)r * k + kk], w, acc[r]);
                        }
                    }
                }
            }

            #pragma unroll
            for (int r = 0; r < IDRAK_ROWS; r++)
            {
                part[r * IDRAK_BLOCK + tid] = acc[r];
            }

            __syncthreads();
            if (transB != 0)
            {
                // A column's lanes added in a tree of a fixed shape (the same on every grid), into its lane 0.
                for (int half = IDRAK_LANES / 2; half > 0; half >>= 1)
                {
                    if (lane < half)
                    {
                        for (int r = 0; r < m; r++)
                        {
                            part[r * IDRAK_BLOCK + tid] += part[r * IDRAK_BLOCK + tid + half];
                        }
                    }

                    __syncthreads();
                }
            }

            if (j < n && (transB != 0 ? lane : group) == 0)
            {
                for (int r = 0; r < m; r++)
                {
                    float s = 0.0f;
                    if (transB != 0)
                    {
                        s = part[r * IDRAK_BLOCK + tid];
                    }
                    else
                    {
                        for (int g = 0; g < IDRAK_GROUPS; g++)
                        {
                            s += part[r * IDRAK_BLOCK + g * IDRAK_LANES + lane];
                        }
                    }

                    if (bias != 0)
                    {
                        s += bias[j];
                    }

                    long long o = (long long)r * n + j;
                    yb[o] = beta == 0.0f ? s : fmaf(beta, yb[o], s);
                }
            }
        }

        // y[r, j] = Σ_kk x[r, kk] · w(kk, j) for r < m ≤ IDRAK_ROWS with packed weights: format 0 bfloat16 (rows of ⌈n / 2⌉
        // words, the even column in the low half), format 1 4-bit (rows of ⌈n / 8⌉ words, nibble c of a word = column 8w + c,
        // each weight times scales[kk / 32, j] before its multiply-add; scale rows of 8 · ⌈n / 8⌉ values). Each group's lanes
        // take consecutive columns (the words they share read once from cache), its groups slices of k, as gemv_f32 as stored.
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) packed_gemv_f32(const float* x, const unsigned int* q, const float* scales, float* y, int m, int n, int k, int format)
        {
            __shared__ float part[IDRAK_ROWS * IDRAK_BLOCK];
            int tid = threadIdx.x, lane = tid % IDRAK_LANES, group = tid / IDRAK_LANES;
            int j = blockIdx.x * IDRAK_LANES + lane;
            long long words = format == 0 ? (n + 1) / 2 : (n + 7) / 8;
            float acc[IDRAK_ROWS];
            #pragma unroll
            for (int r = 0; r < IDRAK_ROWS; r++)
            {
                acc[r] = 0.0f;
            }

            if (j < n)
            {
                #pragma unroll 4
                for (int kk = group; kk < k; kk += IDRAK_GROUPS)
                {
                    float w;
                    if (format == 0)
                    {
                        unsigned int word = q[kk * words + (j >> 1)];
                        w = __uint_as_float((j & 1) != 0 ? (word & 0xFFFF0000u) : (word << 16));
                    }
                    else
                    {
                        int nibble = (int)(q[kk * words + (j >> 3)] << (28 - 4 * (j & 7))) >> 28;
                        w = (float)nibble * scales[kk / 32 * (words * 8) + j];
                    }

                    #pragma unroll
                    for (int r = 0; r < IDRAK_ROWS; r++)
                    {
                        if (r < m)
                        {
                            acc[r] = fmaf(x[(long long)r * k + kk], w, acc[r]);
                        }
                    }
                }
            }

            #pragma unroll
            for (int r = 0; r < IDRAK_ROWS; r++)
            {
                part[r * IDRAK_BLOCK + tid] = acc[r];
            }

            __syncthreads();
            if (j < n && group == 0)
            {
                for (int r = 0; r < m; r++)
                {
                    float s = 0.0f;
                    for (int g = 0; g < IDRAK_GROUPS; g++)
                    {
                        s += part[r * IDRAK_BLOCK + g * IDRAK_LANES + lane];
                    }

                    y[(long long)r * n + j] = s;
                }
            }
        }

        """;
}
