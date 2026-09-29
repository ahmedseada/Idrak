using System.Numerics;
using System.Runtime.InteropServices;

namespace Idrak.Backends.Cpu;

// Int8 weight-only quantization: signed bytes packed four per float element along each weight row.
internal sealed partial class CpuBackend
{
    private const int Int8MinBlock = 64;   // fewest columns per parallel work item (a multiple of every SIMD width)

    public override void Int8MatMul(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k)
    {
        float[] xv = D(x), sv = D(scales), yv = D(y);
        int stride = (n + 3) / 4 * 4;                                  // bytes per weight row
        // About two column blocks per thread, whole SIMD vectors wide.
        int threads = Math.Max(1, ComputeResources.ParallelOptions.MaxDegreeOfParallelism);
        int blockSize = Math.Max(Int8MinBlock, (n / (2 * threads) + Int8MinBlock - 1) / Int8MinBlock * Int8MinBlock);
        int blocks = (n + blockSize - 1) / blockSize;
        For(blocks, (long)m * n * k, (first, last) =>
        {
            var weights = Bytes(q, k * stride);                          // spans cannot be captured; re-derive per worker
            var acc = new float[blockSize];
            for (int block = first; block < last; block++)
            {
                int j0 = block * blockSize, width = Math.Min(blockSize, n - j0);
                for (int r = 0; r < m; r++)
                {
                    Array.Clear(acc);
                    int xo = r * k;
                    for (int kk = 0; kk < k; kk++)
                    {
                        float xk = xv[xo + kk];
                        if (xk != 0f)
                        {
                            AddScaled(acc.AsSpan(0, width), weights.Slice(kk * stride + j0, width), xk);
                        }
                    }

                    int yo = r * n + j0;
                    for (int j = 0; j < width; j++)
                    {
                        yv[yo + j] = acc[j] * sv[j0 + j];
                    }
                }
            }
        });
    }

    public override void Int4MatMul(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k)
    {
        if (m > 8)
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

        // Few rows: each worker expands one weight row of its column block at a time and adds it to every input row.
        float[] xv = D(x), yv = D(y);
        int threads = Math.Max(1, ComputeResources.ParallelOptions.MaxDegreeOfParallelism);
        int blockSize = Math.Max(Int8MinBlock, (n / (2 * threads) + Int8MinBlock - 1) / Int8MinBlock * Int8MinBlock);
        int blocks = (n + blockSize - 1) / blockSize;
        For(blocks, (long)m * n * k, (first, last) =>
        {
            var acc = new float[m * blockSize];
            var row = new float[blockSize];
            for (int block = first; block < last; block++)
            {
                int j0 = block * blockSize, width = Math.Min(blockSize, n - j0);
                Array.Clear(acc);
                for (int kk = 0; kk < k; kk++)
                {
                    Int4Row(q, scales, kk, n, j0, row.AsSpan(0, width));
                    for (int r = 0; r < m; r++)
                    {
                        float xk = xv[r * k + kk];
                        if (xk != 0f)
                        {
                            AddScaled(acc.AsSpan(r * blockSize, width), row.AsSpan(0, width), xk);
                        }
                    }
                }

                for (int r = 0; r < m; r++)
                {
                    acc.AsSpan(r * blockSize, width).CopyTo(yv.AsSpan(r * n + j0, width));
                }
            }
        });
    }

    public override void Int4Dequantize(Storage q, Storage scales, Storage w, int k, int n)
    {
        float[] wv = D(w);
        For(k, (long)k * n, (first, last) =>
        {
            for (int kk = first; kk < last; kk++)
            {
                Int4Row(q, scales, kk, n, 0, wv.AsSpan(kk * n, n));
            }
        });
    }

    // target[j] = the float value of 4-bit weight (row, j0 + j) (see Backend.Int4MatMul for the packing).
    private static void Int4Row(Storage q, Storage scales, int row, int n, int j0, Span<float> target)
    {
        int words = (n + 7) / 8;
        var packed = MemoryMarshal.Cast<float, uint>(D(q).AsSpan()).Slice(row * words, words);
        var s = D(scales).AsSpan(row / 32 * words * 8, words * 8);
        for (int j = 0; j < target.Length; j++)
        {
            int col = j0 + j;
            int nibble = (int)(packed[col >> 3] << (28 - 4 * (col & 7))) >> 28;
            target[j] = nibble * s[col];
        }
    }

    // acc[j] += scale · row[j].
    private static void AddScaled(Span<float> acc, ReadOnlySpan<float> row, float scale)
    {
        int j = 0;
        if (Vector.IsHardwareAccelerated)
        {
            var s = new Vector<float>(scale);
            for (; j <= row.Length - Vector<float>.Count; j += Vector<float>.Count)
            {
                (new Vector<float>(row[j..]) * s + new Vector<float>(acc[j..])).CopyTo(acc[j..]);
            }
        }

        for (; j < row.Length; j++)
        {
            acc[j] += scale * row[j];
        }
    }

    public override void BFloat16MatMul(Storage x, Storage packed, Storage y, int m, int n, int k)
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
    }

    public override void BFloat16Dequantize(Storage packed, Storage w, int k, int n)
    {
        var halves = MemoryMarshal.Cast<float, ushort>(D(packed).AsSpan());
        float[] wv = D(w);
        int stride = (n + 1) / 2 * 2;
        for (int r = 0; r < k; r++)
        {
            for (int j = 0; j < n; j++)
            {
                wv[r * n + j] = BitConverter.Int32BitsToSingle(halves[r * stride + j] << 16);
            }
        }
    }

    public override void Int8Dequantize(Storage q, Storage scales, Storage w, int k, int n)
    {
        float[] sv = D(scales), wv = D(w);
        int stride = (n + 3) / 4 * 4;
        For(k, (long)k * n, (first, last) =>
        {
            var weights = Bytes(q, k * stride);
            for (int kk = first; kk < last; kk++)
            {
                var row = weights.Slice(kk * stride, n);
                int o = kk * n;
                for (int j = 0; j < n; j++)
                {
                    wv[o + j] = row[j] * sv[j];
                }
            }
        });
    }

    public override void KeyValueWriteInt8(Storage source, Storage cache, Storage scales, Storage position, int heads, int steps, int capacity, int dim)
    {
        float[] src = D(source), sv = D(scales);
        var bytes = MemoryMarshal.Cast<float, sbyte>(D(cache).AsSpan());
        int start = (int)D(position)[0], stride = (dim + 3) / 4 * 4;
        for (int row = 0; row < heads * steps; row++)
        {
            int slot = row / steps * capacity + start + row % steps;
            var x = src.AsSpan(row * dim, dim);
            float max = 0f;
            foreach (float v in x)
            {
                max = MathF.Max(max, MathF.Abs(v));
            }

            float scale = max / 127f, inverse = max > 0f ? 1f / scale : 0f;
            sv[slot] = scale;
            var target = bytes.Slice(slot * stride, stride);
            target.Clear();
            for (int d = 0; d < dim; d++)
            {
                target[d] = (sbyte)Math.Clamp((int)MathF.Round(x[d] * inverse, MidpointRounding.ToEven), -127, 127);
            }
        }
    }

    public override void AttentionDecode(Storage q, Storage keys, Storage values, Storage position, Storage y, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale) =>
        AttentionTiled(q, keys, values, position, y, null, heads, rowsPerHead, steps, capacity, dim, scale);

    public override void AttentionTiled(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage? logSumExp, int heads,
        int rowsPerHead, int steps, int capacity, int dim, float scale)
    {
        float[] qv = D(q), kv = D(keys), vv = D(values), yv = D(y);
        float[]? lv = logSumExp is null ? null : D(logSumExp);
        int position0 = (int)D(position)[0];
        For(heads * rowsPerHead, (long)heads * rowsPerHead * dim * Math.Max(1, position0), (first, last) =>
        {
            var scores = new float[capacity];
            for (int row = first; row < last; row++)
            {
                int h = row / rowsPerHead, count = Math.Min(position0 + row % rowsPerHead % steps, capacity - 1) + 1;
                var query = qv.AsSpan(row * dim, dim);
                float max = float.NegativeInfinity;
                for (int c = 0; c < count; c++)
                {
                    float dot = 0f;
                    var key = kv.AsSpan((int)(((long)h * capacity + c) * dim), dim);
                    for (int d = 0; d < dim; d++)
                    {
                        dot += query[d] * key[d];
                    }

                    scores[c] = dot * scale;
                    max = MathF.Max(max, scores[c]);
                }

                float sum = CpuMath.ExpShifted(scores.AsSpan(0, count), max);

                if (lv is not null)
                {
                    lv[row] = max + MathF.Log(sum);
                }

                var output = yv.AsSpan(row * dim, dim);
                output.Clear();
                for (int c = 0; c < count; c++)
                {
                    float weight = scores[c] / sum;
                    var value = vv.AsSpan((int)(((long)h * capacity + c) * dim), dim);
                    for (int d = 0; d < dim; d++)
                    {
                        output[d] += weight * value[d];
                    }
                }
            }
        });
    }

    public override void AttentionTiledBackward(Storage q, Storage keys, Storage values, Storage output, Storage logSumExp, Storage dOutput,
        Storage dq, Storage dkeys, Storage dvalues, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale)
    {
        float[] qv = D(q), kv = D(keys), vv = D(values), ov = D(output), lv = D(logSumExp), gv = D(dOutput);
        float[] dqv = D(dq), dkv = D(dkeys), dvv = D(dvalues);
        For(heads, (long)heads * rowsPerHead * capacity * dim, (first, last) =>
        {
            for (int h = first; h < last; h++)
            {
                for (int i = 0; i < rowsPerHead; i++)
                {
                    int row = h * rowsPerHead + i, count = Math.Min(i % steps, capacity - 1) + 1;
                    var query = qv.AsSpan(row * dim, dim);
                    var gradOut = gv.AsSpan(row * dim, dim);
                    float delta = 0f;
                    for (int d = 0; d < dim; d++)
                    {
                        delta += gradOut[d] * ov[row * dim + d];
                    }

                    for (int c = 0; c < count; c++)
                    {
                        int key = (h * capacity + c) * dim;
                        float dot = 0f, dp = 0f;
                        for (int d = 0; d < dim; d++)
                        {
                            dot += query[d] * kv[key + d];
                            dp += gradOut[d] * vv[key + d];
                        }

                        float p = MathF.Exp(dot * scale - lv[row]), ds = p * (dp - delta);
                        for (int d = 0; d < dim; d++)
                        {
                            dqv[row * dim + d] += scale * ds * kv[key + d];
                            dkv[key + d] += scale * ds * query[d];
                            dvv[key + d] += p * gradOut[d];
                        }
                    }
                }
            }
        });
    }

    public override bool SupportsSegmentedAttention(int dim) => true;

    public override bool AttentionRows(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage starts, int heads, int headsPerRow,
        int rowsPerHead, int steps, int capacity, int dim, float scale)
    {
        float[] qv = D(q), kv = D(keys), vv = D(values), yv = D(y), sv = D(starts);
        int position0 = (int)D(position)[0];
        For(heads * rowsPerHead, (long)heads * rowsPerHead * dim * Math.Max(1, position0), (first, last) =>
        {
            var scores = new float[capacity];
            for (int row = first; row < last; row++)
            {
                int h = row / rowsPerHead, t = row % rowsPerHead % steps;
                int begin = (int)sv[h / headsPerRow * steps + t], end = Math.Min(position0 + t, capacity - 1) + 1, count = end - begin;
                var output = yv.AsSpan(row * dim, dim);
                output.Clear();
                if (count <= 0)
                {
                    continue;                                                    // padding: sees nothing
                }

                var query = qv.AsSpan(row * dim, dim);
                float max = float.NegativeInfinity;
                for (int c = 0; c < count; c++)
                {
                    float dot = 0f;
                    var key = kv.AsSpan((int)(((long)h * capacity + begin + c) * dim), dim);
                    for (int d = 0; d < dim; d++)
                    {
                        dot += query[d] * key[d];
                    }

                    scores[c] = dot * scale;
                    max = MathF.Max(max, scores[c]);
                }

                float sum = CpuMath.ExpShifted(scores.AsSpan(0, count), max);
                for (int c = 0; c < count; c++)
                {
                    float weight = scores[c] / sum;
                    var value = vv.AsSpan((int)(((long)h * capacity + begin + c) * dim), dim);
                    for (int d = 0; d < dim; d++)
                    {
                        output[d] += weight * value[d];
                    }
                }
            }
        });
        return true;
    }

    public override bool AttentionSegmented(Storage q, Storage keys, Storage values, Storage y, Storage? logSumExp, Storage starts, Storage ends,
        int heads, int headsPerRow, int rowsPerHead, int steps, int dim, float scale)
    {
        float[] qv = D(q), kv = D(keys), vv = D(values), yv = D(y), sv = D(starts);
        float[]? lv = logSumExp is null ? null : D(logSumExp);
        For(heads * rowsPerHead, (long)heads * rowsPerHead * dim * steps / 2, (first, last) =>
        {
            var scores = new float[steps];
            for (int row = first; row < last; row++)
            {
                int h = row / rowsPerHead, t = row % rowsPerHead % steps, begin = (int)sv[h / headsPerRow * steps + t], count = t + 1 - begin;
                var query = qv.AsSpan(row * dim, dim);
                float max = float.NegativeInfinity;
                for (int c = 0; c < count; c++)
                {
                    float dot = 0f;
                    var key = kv.AsSpan((int)(((long)h * steps + begin + c) * dim), dim);
                    for (int d = 0; d < dim; d++)
                    {
                        dot += query[d] * key[d];
                    }

                    scores[c] = dot * scale;
                    max = MathF.Max(max, scores[c]);
                }

                float sum = CpuMath.ExpShifted(scores.AsSpan(0, count), max);
                if (lv is not null)
                {
                    lv[row] = max + MathF.Log(sum);
                }

                var output = yv.AsSpan(row * dim, dim);
                output.Clear();
                for (int c = 0; c < count; c++)
                {
                    float weight = scores[c] / sum;
                    var value = vv.AsSpan((int)(((long)h * steps + begin + c) * dim), dim);
                    for (int d = 0; d < dim; d++)
                    {
                        output[d] += weight * value[d];
                    }
                }
            }
        });
        return true;
    }

    public override bool AttentionSegmentedBackward(Storage q, Storage keys, Storage values, Storage output, Storage logSumExp, Storage dOutput,
        Storage dq, Storage dkeys, Storage dvalues, Storage starts, Storage ends, int heads, int headsPerRow, int rowsPerHead, int steps, int dim, float scale)
    {
        float[] qv = D(q), kv = D(keys), vv = D(values), ov = D(output), lv = D(logSumExp), gv = D(dOutput), sv = D(starts);
        float[] dqv = D(dq), dkv = D(dkeys), dvv = D(dvalues);
        For(heads, (long)heads * rowsPerHead * steps * dim / 2, (first, last) =>
        {
            for (int h = first; h < last; h++)
            {
                for (int i = 0; i < rowsPerHead; i++)
                {
                    int row = h * rowsPerHead + i, t = i % steps, begin = (int)sv[h / headsPerRow * steps + t];
                    var query = qv.AsSpan(row * dim, dim);
                    var gradOut = gv.AsSpan(row * dim, dim);
                    float delta = 0f;
                    for (int d = 0; d < dim; d++)
                    {
                        delta += gradOut[d] * ov[row * dim + d];
                    }

                    for (int c = begin; c <= t; c++)
                    {
                        int key = (h * steps + c) * dim;
                        float dot = 0f, dp = 0f;
                        for (int d = 0; d < dim; d++)
                        {
                            dot += query[d] * kv[key + d];
                            dp += gradOut[d] * vv[key + d];
                        }

                        float p = MathF.Exp(dot * scale - lv[row]), ds = p * (dp - delta);
                        for (int d = 0; d < dim; d++)
                        {
                            dqv[row * dim + d] += scale * ds * kv[key + d];
                            dkv[key + d] += scale * ds * query[d];
                            dvv[key + d] += p * gradOut[d];
                        }
                    }
                }
            }
        });
        return true;
    }

    public override void AttentionInt8(Storage q, Storage keys, Storage values, Storage keyScales, Storage valueScales, Storage position,
        Storage y, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale, bool tiled)
    {
        float[] qv = D(q), ks = D(keyScales), vs = D(valueScales), yv = D(y);
        int position0 = (int)D(position)[0], stride = (dim + 3) / 4 * 4;
        For(heads * rowsPerHead, (long)heads * rowsPerHead * dim * Math.Max(1, position0), (first, last) =>
        {
            var kb = MemoryMarshal.Cast<float, sbyte>(D(keys).AsSpan());
            var vb = MemoryMarshal.Cast<float, sbyte>(D(values).AsSpan());
            var scores = new float[capacity];
            for (int row = first; row < last; row++)
            {
                int h = row / rowsPerHead, count = Math.Min(position0 + row % rowsPerHead % steps, capacity - 1) + 1;
                var query = qv.AsSpan(row * dim, dim);
                float max = float.NegativeInfinity;
                for (int c = 0; c < count; c++)
                {
                    var key = kb.Slice((h * capacity + c) * stride, dim);
                    float dot = 0f;
                    for (int d = 0; d < dim; d++)
                    {
                        dot += query[d] * key[d];
                    }

                    scores[c] = dot * ks[h * capacity + c] * scale;
                    max = MathF.Max(max, scores[c]);
                }

                float sum = CpuMath.ExpShifted(scores.AsSpan(0, count), max);

                var output = yv.AsSpan(row * dim, dim);
                output.Clear();
                for (int c = 0; c < count; c++)
                {
                    float weight = scores[c] / sum * vs[h * capacity + c];
                    var value = vb.Slice((h * capacity + c) * stride, dim);
                    for (int d = 0; d < dim; d++)
                    {
                        output[d] += weight * value[d];
                    }
                }
            }
        });
    }

    public override void AttentionScoresInt8(Storage q, Storage cache, Storage scales, Storage y, int rows, int steps, int capacity, int dim)
    {
        float[] qv = D(q), sv = D(scales), yv = D(y);
        int stride = (dim + 3) / 4 * 4;
        For(rows, (long)rows * steps * capacity * dim, (first, last) =>
        {
            var keys = MemoryMarshal.Cast<float, sbyte>(D(cache).AsSpan());
            for (int r = first; r < last; r++)
            {
                for (int t = 0; t < steps; t++)
                {
                    var qr = qv.AsSpan((r * steps + t) * dim, dim);
                    for (int c = 0; c < capacity; c++)
                    {
                        var k = keys.Slice((r * capacity + c) * stride, dim);
                        float sum = 0f;
                        for (int d = 0; d < dim; d++)
                        {
                            sum += qr[d] * k[d];
                        }

                        yv[(r * steps + t) * capacity + c] = sum * sv[r * capacity + c];
                    }
                }
            }
        });
    }

    public override void AttentionContextInt8(Storage weights, Storage cache, Storage scales, Storage y, int rows, int steps, int capacity, int dim)
    {
        float[] wv = D(weights), sv = D(scales), yv = D(y);
        int stride = (dim + 3) / 4 * 4;
        For(rows, (long)rows * steps * capacity * dim, (first, last) =>
        {
            var values = MemoryMarshal.Cast<float, sbyte>(D(cache).AsSpan());
            for (int r = first; r < last; r++)
            {
                for (int t = 0; t < steps; t++)
                {
                    var output = yv.AsSpan((r * steps + t) * dim, dim);
                    output.Clear();
                    for (int c = 0; c < capacity; c++)
                    {
                        float a = wv[(r * steps + t) * capacity + c] * sv[r * capacity + c];
                        if (a != 0f)
                        {
                            AddScaled(output, values.Slice((r * capacity + c) * stride, dim), a);
                        }
                    }
                }
            }
        });
    }

    public override void AddRmsNormAffine(Storage a, Storage b, Storage sum, Storage gain, Storage y, int rows, int cols, float eps, float offset)
    {
        float[] av = D(a), bv = D(b), sv = D(sum);
        For(rows * cols, (long)rows * cols, (first, last) =>
        {
            for (int i = first; i < last; i++)
            {
                sv[i] = av[i] + bv[i];
            }
        });
        RmsNormAffine(sum, gain, y, rows, cols, eps, offset);
    }

    public override void RmsNormRope(Storage x, Storage gain, Storage cos, Storage sin, Storage positions, Storage y, int rows, int cols,
        float eps, float offset, int heads, int steps, int half, bool interleaved)
    {
        var normalized = Allocate(rows * cols, zeroed: false);
        try
        {
            RmsNormAffine(x, gain, normalized, rows, cols, eps, offset);
            Copy(normalized, y, rows * cols);
            Rope(normalized, y, cos, sin, positions, rows, heads, steps, cols, half, interleaved, 1f);
        }
        finally
        {
            normalized.Release();
        }
    }

    public override void RmsNormAffine(Storage x, Storage gain, Storage y, int rows, int cols, float eps, float offset)
    {
        float[] xv = D(x), gv = D(gain), yv = D(y);
        For(rows, (long)rows * cols * 2, (first, last) =>
        {
            for (int r = first; r < last; r++)
            {
                var row = xv.AsSpan(r * cols, cols);
                float sum = 0f;
                foreach (float v in row)
                {
                    sum = MathF.FusedMultiplyAdd(v, v, sum);
                }

                float scale = 1f / MathF.Sqrt(sum / cols + eps);
                var output = yv.AsSpan(r * cols, cols);
                for (int j = 0; j < cols; j++)
                {
                    output[j] = row[j] * scale * (gv[j] + offset);
                }
            }
        });
    }

    // act(g) and act'(g) for the gated feed-forward activations.
    private static (float Value, float Slope) Activation(float g, int kind)
    {
        switch (kind)
        {
            case 0:
            {
                float s = 1f / (1f + MathF.Exp(-g));
                return (g * s, s * (1f + g * (1f - s)));
            }

            case 1:
            {
                const float K = 0.7978845608f, C = 0.044715f;
                float t = MathF.Tanh(K * (g + C * g * g * g));
                return (0.5f * g * (1f + t), 0.5f * (1f + t) + 0.5f * g * (1f - t * t) * K * (1f + 3f * C * g * g));
            }

            default:
                return (g > 0f ? g : 0f, g > 0f ? 1f : 0f);
        }
    }

    public override void GatedActivation(Storage gate, Storage up, Storage y, int n, int kind)
    {
        float[] gv = D(gate), uv = D(up), yv = D(y);
        For(n, (long)n * 8, (first, last) =>
        {
            CpuMath.Gated(gv.AsSpan(first, last - first), uv.AsSpan(first, last - first), yv.AsSpan(first, last - first), kind);
        });
    }

    public override void GatedActivationBackward(Storage gate, Storage up, Storage dy, Storage dgate, Storage dup, int n, int kind, int flags)
    {
        float[] gv = D(gate), uv = D(up), dv = D(dy), dg = D(dgate), du = D(dup);
        For(n, (long)n * 8, (first, last) =>
        {
            for (int i = first; i < last; i++)
            {
                var (value, slope) = Activation(gv[i], kind);
                if ((flags & 1) != 0)
                {
                    dg[i] = ((flags & 4) != 0 ? 0f : dg[i]) + dv[i] * uv[i] * slope;
                }

                if ((flags & 2) != 0)
                {
                    du[i] = ((flags & 8) != 0 ? 0f : du[i]) + dv[i] * value;
                }
            }
        });
    }

    public override void GatedActivationPacked(Storage gate, Storage up, Storage packedGate, Storage packedUp, Storage y, Storage packedY, int n, int kind, int flags)
    {
        Storage? g = null, u = null, t = null;
        try
        {
            if ((flags & 1) != 0)
            {
                (g, u) = (Allocate(n, false), Allocate(n, false));
                BFloat16Dequantize(packedGate, g, 1, n);
                BFloat16Dequantize(packedUp, u, 1, n);
            }

            var (gv, uv) = (g ?? gate, u ?? up);
            if ((flags & 2) != 0 && (flags & 1) == 0)
            {
                PackBFloat16(gate, packedGate, n);
                PackBFloat16(up, packedUp, n);
            }

            var target = (flags & 4) != 0 ? y : t = Allocate(n, false);
            GatedActivation(gv, uv, target, n, kind);
            if ((flags & 8) != 0)
            {
                PackBFloat16(target, packedY, n);
            }
        }
        finally
        {
            g?.Release();
            u?.Release();
            t?.Release();
        }
    }

    public override void GatedActivationBackwardPacked(Storage packedGate, Storage packedUp, Storage dy, Storage dgate, Storage dup, int n, int kind, int flags)
    {
        Storage g = Allocate(n, false), u = Allocate(n, false);
        try
        {
            BFloat16Dequantize(packedGate, g, 1, n);
            BFloat16Dequantize(packedUp, u, 1, n);
            GatedActivationBackward(g, u, dy, dgate, dup, n, kind, flags);
        }
        finally
        {
            g.Release();
            u.Release();
        }
    }

    public override void RmsNorm(Storage x, Storage y, Storage inv, int rows, int cols, float eps)
    {
        float[] xv = D(x), yv = D(y), iv = D(inv);
        For(rows, (long)rows * cols * 2, (first, last) =>
        {
            for (int r = first; r < last; r++)
            {
                var row = xv.AsSpan(r * cols, cols);
                float sum = 0f;
                foreach (float v in row)
                {
                    sum = MathF.FusedMultiplyAdd(v, v, sum);
                }

                float scale = 1f / MathF.Sqrt(sum / cols + eps);
                iv[r] = scale;
                var output = yv.AsSpan(r * cols, cols);
                for (int j = 0; j < cols; j++)
                {
                    output[j] = row[j] * scale;
                }
            }
        });
    }

    public override void RmsNormBackward(Storage dy, Storage y, Storage inv, Storage dx, int rows, int cols)
    {
        float[] gv = D(dy), yv = D(y), iv = D(inv), dv = D(dx);
        For(rows, (long)rows * cols * 3, (first, last) =>
        {
            for (int r = first; r < last; r++)
            {
                int o = r * cols;
                float dot = 0f;
                for (int j = 0; j < cols; j++)
                {
                    dot = MathF.FusedMultiplyAdd(gv[o + j], yv[o + j], dot);
                }

                float mean = dot / cols;
                for (int j = 0; j < cols; j++)
                {
                    dv[o + j] += iv[r] * (gv[o + j] - yv[o + j] * mean);
                }
            }
        });
    }

    public override void Rope(Storage x, Storage y, Storage cos, Storage sin, Storage positions, int rows, int heads, int steps, int dim, int half, bool interleaved, float sign)
    {
        float[] xv = D(x), yv = D(y), cv = D(cos), sv = D(sin), pv = D(positions);
        For(rows, (long)rows * half * 4, (first, last) =>
        {
            for (int r = first; r < last; r++)
            {
                int position = (int)pv[r / heads % steps], o = r * dim;
                for (int p = 0; p < half; p++)
                {
                    float c = cv[position * half + p], s = sv[position * half + p] * sign;
                    int i = o + (interleaved ? 2 * p : p), j = o + (interleaved ? 2 * p + 1 : p + half);
                    float a = xv[i], b = xv[j];
                    yv[i] = MathF.FusedMultiplyAdd(a, c, -(b * s));
                    yv[j] = MathF.FusedMultiplyAdd(b, c, a * s);
                }
            }
        });
    }

    private static ReadOnlySpan<sbyte> Bytes(Storage q, int count) => MemoryMarshal.Cast<float, sbyte>(D(q).AsSpan())[..count];

    // acc[j] += scale · (float)q[j], vectorized: bytes widen to shorts, then ints, then floats.
    private static void AddScaled(Span<float> acc, ReadOnlySpan<sbyte> q, float scale)
    {
        int j = 0;
        if (Vector.IsHardwareAccelerated && q.Length >= Vector<sbyte>.Count)
        {
            var s = new Vector<float>(scale);
            int floats = Vector<float>.Count;
            for (; j <= q.Length - Vector<sbyte>.Count; j += Vector<sbyte>.Count)
            {
                Vector.Widen(new Vector<sbyte>(q[j..]), out var lowShorts, out var highShorts);
                Vector.Widen(lowShorts, out var i0, out var i1);
                Vector.Widen(highShorts, out var i2, out var i3);
                var a = acc[j..];
                (Vector.ConvertToSingle(i0) * s + new Vector<float>(a)).CopyTo(a);
                (Vector.ConvertToSingle(i1) * s + new Vector<float>(a[floats..])).CopyTo(a[floats..]);
                (Vector.ConvertToSingle(i2) * s + new Vector<float>(a[(2 * floats)..])).CopyTo(a[(2 * floats)..]);
                (Vector.ConvertToSingle(i3) * s + new Vector<float>(a[(3 * floats)..])).CopyTo(a[(3 * floats)..]);
            }
        }

        for (; j < q.Length; j++)
        {
            acc[j] += scale * q[j];
        }
    }
}
