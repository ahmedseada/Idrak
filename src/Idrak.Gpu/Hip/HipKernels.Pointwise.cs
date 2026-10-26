// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Hip;

// Element-wise kernels beyond the first set: the remaining activations (GELU in both forms, SiLU, square root, sine,
// cosine, sign), GELU after a bias, the embedding gathers, the key/value cache write and decoder mask (the position read
// from device memory), permutes, accumulating strided copies, and the packed weights' expansions to float.
internal static partial class HipKernels
{
    private const string Pointwise = """
        // GELU, tanh approximation: 0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x³))).
        __device__ inline float idrak_gelu(float v)
        {
            return 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
        }

        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) gelu_f32(const float* x, float* y, int n)
        {
            IDRAK_EACH(i, n) y[i] = idrak_gelu(x[i]);
        }

        // GELU through erf: 0.5 x (1 + erf(x / sqrt 2)).
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) gelu_erf_f32(const float* x, float* y, int n)
        {
            IDRAK_EACH(i, n)
            {
                float v = x[i];
                y[i] = 0.5f * v * (1.0f + erff(v * 0.70710678f));
            }
        }

        // x · sigmoid(x)
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) silu_f32(const float* x, float* y, int n)
        {
            IDRAK_EACH(i, n)
            {
                float v = x[i];
                y[i] = v * (1.0f / (1.0f + expf(-v)));
            }
        }

        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) sqrt_f32(const float* x, float* y, int n)
        {
            IDRAK_EACH(i, n) y[i] = sqrtf(x[i]);
        }

        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) sin_f32(const float* x, float* y, int n)
        {
            IDRAK_EACH(i, n) y[i] = sinf(x[i]);
        }

        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) cos_f32(const float* x, float* y, int n)
        {
            IDRAK_EACH(i, n) y[i] = cosf(x[i]);
        }

        // 1 above zero, -1 below, the value itself otherwise: zeros keep their sign, NaN stays NaN.
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) sign_f32(const float* x, float* y, int n)
        {
            IDRAK_EACH(i, n)
            {
                float v = x[i];
                y[i] = v > 0.0f ? 1.0f : v < 0.0f ? -1.0f : v;
            }
        }

        // y[i] = gelu(x[i] + bias[i % cols])
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) bias_gelu_f32(const float* x, const float* bias, float* y, int n, int cols)
        {
            IDRAK_EACH(i, n) y[i] = idrak_gelu(x[i] + bias[i % cols]);
        }

        // Embedding lookup: y[r, d] = table[indices[r], d] over n = count · dim values; an index outside [0, last] is held to it.
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) gather_f32(const float* table, const float* indices, float* y, int dim, int last, int n)
        {
            IDRAK_EACH(i, n)
            {
                long long r = i / dim, d = i % dim;
                y[i] = table[idrak_index(indices[r], last) * dim + d];
            }
        }

        // The same from a bfloat16 table, two values a word along each row of `words` words, the even column in the low half.
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) gather_bf16_f32(const unsigned int* table, const float* indices, float* y, int dim, int last, int words, int n)
        {
            IDRAK_EACH(i, n)
            {
                long long r = i / dim, d = i % dim;
                unsigned int word = table[idrak_index(indices[r], last) * words + (d >> 1)];
                y[i] = __uint_as_float((d & 1) != 0 ? (word & 0xFFFF0000u) : (word << 16));
            }
        }

        // Column indices[r] of a bfloat16 table [dim, vocabulary] in rows of `words` words: y[r, d] = table[d, indices[r]].
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) gather_bf16_cols_f32(const unsigned int* table, const float* indices, float* y, int dim, int last, int words, int n)
        {
            IDRAK_EACH(i, n)
            {
                long long r = i / dim, d = i % dim;
                long long column = idrak_index(indices[r], last);
                unsigned int word = table[d * words + (column >> 1)];
                y[i] = __uint_as_float((column & 1) != 0 ? (word & 0xFFFF0000u) : (word << 16));
            }
        }

        // cache[h, position + t, :] = source[h, t, :] for [heads, steps, dim] into [heads, capacity, dim], the position read
        // from device memory; rows past the capacity are not written.
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) kv_write_f32(const float* source, float* cache, const float* position, int heads, int steps, int capacity, int dim)
        {
            long long per = (long long)steps * dim;
            int pos = (int)position[0];
            IDRAK_EACH(i, (long long)heads * per)
            {
                long long h = i / per, rest = i % per;
                if (pos >= 0 && pos + rest / dim < capacity)
                {
                    cache[(h * capacity + pos) * dim + rest] = source[i];
                }
            }
        }

        // mask[i, j] = j <= position + i ? 0 : -1e9 for a [rows, capacity] mask, the position read from device memory.
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) decoder_mask_f32(const float* position, float* mask, int rows, int capacity)
        {
            int pos = (int)position[0];
            IDRAK_EACH(i, (long long)rows * capacity)
            {
                long long r = i / capacity, j = i % capacity;
                mask[i] = j <= pos + r ? 0.0f : -1e9f;
            }
        }

        // y (+)= x permuted: output element i at coordinates c0..c5 of the shape s0..s5 (rank 6, padded in front with size-1
        // dimensions) comes from input offset Σ c_d · t_d.
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) permute_f32(const float* x, float* y, int s0, int s1, int s2, int s3, int s4, int s5, int t0, int t1, int t2, int t3, int t4, int t5, int n, int accumulate)
        {
            IDRAK_EACH(i, n)
            {
                long long rem = i, offset = 0;
                offset += rem % s5 * t5;
                rem /= s5;
                offset += rem % s4 * t4;
                rem /= s4;
                offset += rem % s3 * t3;
                rem /= s3;
                offset += rem % s2 * t2;
                rem /= s2;
                offset += rem % s1 * t1;
                rem /= s1;
                offset += rem % s0 * t0;
                float v = x[offset];
                y[i] = accumulate != 0 ? y[i] + v : v;
            }
        }

        // dst[dstOffset + r · dstStride + c] += src[srcOffset + r · srcStride + c] for r < rows, c < cols (rows of dst apart).
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) copy2d_add_f32(const float* src, float* dst, int srcOffset, int srcStride, int dstOffset, int dstStride, int rows, int cols)
        {
            IDRAK_EACH(i, (long long)rows * cols)
            {
                long long r = i / cols, c = i % cols;
                dst[dstOffset + r * dstStride + c] += src[srcOffset + r * srcStride + c];
            }
        }

        // w[r, j] = the bfloat16 weight of row r, column j (rows of ⌈n / 2⌉ words, the even column in the low half).
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) bf16_dequant_f32(const unsigned int* packed, float* w, int k, int n)
        {
            long long words = (n + 1) / 2;
            IDRAK_EACH(i, (long long)k * n)
            {
                long long r = i / n, j = i % n;
                unsigned int word = packed[r * words + (j >> 1)];
                w[i] = __uint_as_float((j & 1) != 0 ? (word & 0xFFFF0000u) : (word << 16));
            }
        }

        // w[r, j] = the IEEE half of row r, column j, widened exactly: normal halves by moving the exponent and mantissa,
        // subnormals as mantissa · 2^-24, infinities and NaNs kept.
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) f16_dequant_f32(const unsigned int* packed, float* w, int k, int n)
        {
            long long words = (n + 1) / 2;
            IDRAK_EACH(i, (long long)k * n)
            {
                long long r = i / n, j = i % n;
                unsigned int word = packed[r * words + (j >> 1)];
                unsigned int h = (j & 1) != 0 ? word >> 16 : word & 0xFFFFu;
                unsigned int e = (h >> 10) & 0x1Fu, mantissa = h & 0x3FFu, sign = (h & 0x8000u) << 16;
                unsigned int bits = e == 0u ? sign | __float_as_uint((float)mantissa * (1.0f / 16777216.0f))
                    : e == 31u ? sign | 0x7F800000u | (mantissa << 13)
                    : sign | ((e + 112u) << 23) | (mantissa << 13);
                w[i] = __uint_as_float(bits);
            }
        }

        // w[r, j] = nibble j of row r (rows of ⌈n / 8⌉ words, nibble c of a word = column 8w + c) times scales[r / 32, j]
        // (scale rows of 8 · ⌈n / 8⌉ values).
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) int4_dequant_f32(const unsigned int* q, const float* scales, float* w, int k, int n)
        {
            long long words = (n + 7) / 8;
            IDRAK_EACH(i, (long long)k * n)
            {
                long long r = i / n, j = i % n;
                int nibble = (int)(q[r * words + (j >> 3)] << (28 - 4 * (int)(j & 7))) >> 28;
                w[i] = (float)nibble * scales[r / 32 * (words * 8) + j];
            }
        }

        // w[r, j] = q[r, j] · scales[j], q signed bytes in rows of ⌈n / 4⌉ · 4 bytes.
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) int8_dequant_f32(const signed char* q, const float* scales, float* w, int k, int n)
        {
            long long stride = (n + 3) / 4 * 4;
            IDRAK_EACH(i, (long long)k * n)
            {
                long long r = i / n, j = i % n;
                w[i] = (float)q[r * stride + j] * scales[j];
            }
        }

        // The bfloat16 bits of v (the low half of the result): to nearest, ties to even; NaN as 0x7FC0.
        __device__ inline unsigned int idrak_bf16(float v)
        {
            unsigned int u = __float_as_uint(v);
            return v != v ? 0x7FC0u : (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
        }

        // Word i of packed = x[2i] and x[2i + 1] rounded to bfloat16 (the low half first; 0 past n).
        extern "C" __global__ void __launch_bounds__(IDRAK_BLOCK) bf16_pack_f32(const float* x, unsigned int* packed, int n)
        {
            IDRAK_EACH(i, ((long long)n + 1) / 2)
            {
                unsigned int low = idrak_bf16(x[2 * i]);
                unsigned int high = 2 * i + 1 < n ? idrak_bf16(x[2 * i + 1]) : 0u;
                packed[i] = low | (high << 16);
            }
        }

        """;
}
