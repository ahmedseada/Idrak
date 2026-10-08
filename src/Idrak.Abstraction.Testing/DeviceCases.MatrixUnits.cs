// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Operations;

namespace Idrak.Abstraction.Testing;

// The operations the CPU has no kernel for at all, which devices run on their matrix units: FP8 weights and products,
// and attention read in place from [batch, steps, *] rows with its gradients. Their references are plain loops that
// follow the contracts in Backend.cs to the letter (the FP8 bytes bit for bit).
public static partial class DeviceCases
{
    // The k padding of the FP8 values the cases lay out (Backend.Float8PaddedK on the library's devices).
    private const int Float8Padding = 64;

    // The e4m3 code of x: rounded to nearest, ties to even, saturated to ±448, the sign kept (-0 included); NaN is 0x7F.
    private static byte Float8Code(float x)
    {
        if (float.IsNaN(x))
        {
            return 0x7F;
        }

        int sign = float.IsNegative(x) ? 0x80 : 0;
        double a = Math.Abs((double)x);
        if (a >= 448)
        {
            return (byte)(sign | 0x7E);
        }

        // The exponent (the subnormals share -6's quantum), the value in quanta of 2^(e - 3), rounded to even: 8 to 16
        // for a normal value (16 carries into the next exponent), below 8 for a subnormal one.
        int e = a == 0 ? -6 : Math.Max(Math.ILogB(a), -6);
        int steps = (int)Math.Round(a / Math.ScaleB(1.0, e - 3), MidpointRounding.ToEven);
        if (steps < 8)
        {
            return (byte)(sign | steps);
        }

        if (steps == 16)
        {
            (e, steps) = (e + 1, 8);
        }

        return (byte)(sign | Math.Min(((e + 7) << 3) | (steps - 8), 0x7E));
    }

    // The value of an e4m3 code.
    private static float Float8Value(int code)
    {
        int e = (code >> 3) & 15, m = code & 7;
        if (e == 15 && m == 7)
        {
            return float.NaN;
        }

        float value = e == 0 ? MathF.ScaleB(m, -9) : MathF.ScaleB(8 + m, e - 10);
        return (code & 0x80) != 0 ? -value : value;
    }

    // One scale per row of `rows` values (a row of x, or a column of w gathered), as Backend.Float8QuantizeWeight sets it,
    // and the codes of the row into `codes`.
    private static float Float8Row(ReadOnlySpan<float> row, Span<byte> codes)
    {
        float max = 0f;
        foreach (float value in row)
        {
            max = MathF.Max(max, MathF.Abs(value));
        }

        float scale = max > 0f ? max / 448f : 1f, inverse = 1f / scale;
        for (int r = 0; r < row.Length; r++)
        {
            codes[r] = Float8Code(row[r] * inverse);
        }

        return scale;
    }

    // Backend.Float8QuantizeWeight by plain loops: w [k, n] → codes [n, paddedK] bytes (as floats) and scales [n].
    private static (float[] Values, float[] Scales) Float8Quantize(float[] w, int k, int n, int paddedK)
    {
        var bytes = new byte[n * paddedK];
        var scales = new float[n];
        var column = new float[k];
        for (int j = 0; j < n; j++)
        {
            for (int r = 0; r < k; r++)
            {
                column[r] = w[r * n + j];
            }

            scales[j] = Float8Row(column, bytes.AsSpan(j * paddedK, k));
        }

        return (MemoryMarshal.Cast<byte, float>(bytes).ToArray(), scales);
    }

    // Backend.Float8MatMul by plain loops, written into y.
    private static void Float8Reference(Backend cpu, Storage x, int m, int k, Storage values, Storage scales, int n, Storage y, float beta, int paddedK)
    {
        float[] xv = Read(x), sv = Read(scales), yv = Read(y);
        float[] w8 = [.. MemoryMarshal.Cast<float, byte>(Read(values).AsSpan()).ToArray().Select(code => Float8Value(code))];
        var x8 = new byte[k];
        var xq = new float[k];
        for (int i = 0; i < m; i++)
        {
            float sx = Float8Row(xv.AsSpan(i * k, k), x8);
            for (int p = 0; p < k; p++)
            {
                xq[p] = Float8Value(x8[p]);
            }

            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int p = 0; p < k; p++)
                {
                    sum += (double)xq[p] * w8[j * paddedK + p];
                }

                float product = (float)(sum * sx * sv[j]);
                yv[i * n + j] = beta == 0f ? product : beta * yv[i * n + j] + product;
            }
        }

        cpu.Upload(yv, y);
    }

    // FP8 weights quantized (the bytes exactly) and multiplied, k padded and not, beta 0 and 1, against plain loops; a
    // column of known codes keeps the reference honest.
    private static void Float8Products(DeviceCaseContext c)
    {
        // A column whose largest magnitude is 448 has scale 1, so its values are coded as they are: the largest, ties to
        // even among normals and subnormals, the smallest subnormal, the largest one rounding into the normals, and signs.
        (float Value, byte Code)[] known =
        [
            (448f, 0x7E), (-448f, 0xFE), (17f, 0x58), (19f, 0x5A), (1f, 0x38), (-1f, 0xB8), (0f, 0x00), (0.3f, 0x2A), (240f, 0x77),
            (MathF.ScaleB(1f, -9), 0x01), (MathF.ScaleB(1f, -10), 0x00), (MathF.ScaleB(3f, -10), 0x02), (MathF.ScaleB(7f, -9), 0x07),
            (MathF.ScaleB(15f, -10), 0x08), (MathF.ScaleB(1f, -6), 0x08), (-MathF.ScaleB(5f, -10), 0x82),
        ];
        foreach (int k in new[] { 64, known.Length, 70, 64 * c.Random.Next(1, 4), c.Size(1, 200) })
        {
            int n = c.Size(1, 140), paddedK = (k + Float8Padding - 1) / Float8Padding * Float8Padding;
            var wv = c.Values(k * n, 0.5f);
            int fixedRows = Math.Min(k, known.Length);
            for (int r = 0; r < fixedRows; r++)
            {
                wv[r * n] = known[r].Value;                                    // column 0: the known values (448 among them)
            }

            if (n > 1)
            {
                for (int r = 0; r < k; r++)
                {
                    wv[r * n + n - 1] = 0f;                                    // the last column: zeros (scale 1)
                }
            }

            Storage w = c.Storage(wv), values = Random(c, n * paddedK / 4), scales = Random(c, n);
            c.Composed(Ops.Float8QuantizeWeight, cpu => cpu.Float8QuantizeWeight(w, k, n, values, scales), cpu =>
            {
                var (codes, columnScales) = Float8Quantize(wv, k, n, paddedK);
                cpu.Upload(codes, values);
                cpu.Upload(columnScales, scales);
            });

            var bytes = MemoryMarshal.Cast<float, byte>(Read(values).AsSpan());
            if (k >= known.Length)
            {
                for (int r = 0; r < known.Length; r++)
                {
                    c.Expect(bytes[r] == known[r].Code, $"the e4m3 code of {known[r].Value:R} is 0x{bytes[r]:X2}, expected 0x{known[r].Code:X2}");
                }

                c.Expect(Read(scales)[0] == 1f, "a column whose largest magnitude is 448 has scale 1");
            }

            foreach (int m in new[] { 1, c.Size(2, 140) })
            {
                foreach (float beta in new[] { 0f, 1f })
                {
                    var xv = c.Values(m * k, 2f);
                    if (m > 1)
                    {
                        Array.Clear(xv, 0, k);                                    // a row of zeros (scale 1)
                    }

                    Storage x = c.Storage(xv), y = Random(c, m * n);
                    c.Composed(Ops.Float8MatMul, cpu => cpu.Float8MatMul(x, m, k, values, scales, n, y, beta),
                        cpu => Float8Reference(cpu, x, m, k, values, scales, n, y, beta, paddedK));
                }
            }
        }
    }

    // Backend.AttentionStrided by plain loops (y and, when given, the log-sum-exp), from host copies.
    private static void StridedAttentionReference(float[] q, long qOffset, float[] k, long kOffset, float[] v, long vOffset, int qRow, int kRow,
        float[] y, float[]? logSumExp, int batch, int kvHeads, int group, int steps, int dim, float scale)
    {
        int heads = kvHeads * group;
        var scores = new double[steps];
        for (int b = 0; b < batch; b++)
        {
            for (int kv = 0; kv < kvHeads; kv++)
            {
                for (int g = 0; g < group; g++)
                {
                    int h = kv * group + g;
                    for (int t = 0; t < steps; t++)
                    {
                        long qi = qOffset + (long)(b * steps + t) * qRow + h * dim;
                        double max = double.NegativeInfinity;
                        for (int p = 0; p <= t; p++)
                        {
                            long ki = kOffset + (long)(b * steps + p) * kRow + kv * dim;
                            double dot = 0;
                            for (int d = 0; d < dim; d++)
                            {
                                dot += (double)q[qi + d] * k[ki + d];
                            }

                            scores[p] = scale * dot;
                            max = Math.Max(max, scores[p]);
                        }

                        double sum = 0;
                        for (int p = 0; p <= t; p++)
                        {
                            scores[p] = Math.Exp(scores[p] - max);
                            sum += scores[p];
                        }

                        long yi = (long)(b * steps + t) * heads * dim + h * dim;
                        for (int d = 0; d < dim; d++)
                        {
                            double acc = 0;
                            for (int p = 0; p <= t; p++)
                            {
                                acc += scores[p] * v[vOffset + (long)(b * steps + p) * kRow + kv * dim + d];
                            }

                            y[yi + d] = (float)(acc / sum);
                        }

                        if (logSumExp is not null)
                        {
                            logSumExp[(b * kvHeads + kv) * group * steps + g * steps + t] = (float)(max + Math.Log(sum));
                        }
                    }
                }
            }
        }
    }

    // Backend.AttentionStridedBackward by plain loops: the gradients added into dq, dk, dv (which may share a storage).
    private static void StridedAttentionBackwardReference(Backend cpu, Storage q, long qOffset, Storage k, long kOffset, Storage v, long vOffset, int qRow,
        int kRow, Storage y, Storage logSumExp, Storage dOutput, Storage dq, long dqOffset, Storage dk, long dkOffset, Storage dv, long dvOffset,
        int batch, int kvHeads, int group, int steps, int dim, float scale)
    {
        float[] qv = Read(q), kv = Read(k), vv = Read(v), yv = Read(y), lse = Read(logSumExp), dov = Read(dOutput);

        // The gradients' storages, each read once (dq, dk and dv may be one), and their increments in double.
        var gradients = new Dictionary<Storage, (float[] Values, double[] Added)>(ReferenceEqualityComparer.Instance);
        double[] Added(Storage s)
        {
            if (!gradients.TryGetValue(s, out var entry))
            {
                entry = (Read(s), new double[s.Length]);
                gradients[s] = entry;
            }

            return entry.Added;
        }

        double[] dqa = Added(dq), dka = Added(dk), dva = Added(dv);
        int heads = kvHeads * group;
        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < kvHeads; h++)
            {
                for (int g = 0; g < group; g++)
                {
                    int head = h * group + g;
                    for (int t = 0; t < steps; t++)
                    {
                        long qi = (long)(b * steps + t) * qRow + head * dim, yi = (long)(b * steps + t) * heads * dim + head * dim;
                        double lseT = lse[(b * kvHeads + h) * group * steps + g * steps + t], delta = 0;
                        for (int d = 0; d < dim; d++)
                        {
                            delta += (double)yv[yi + d] * dov[yi + d];
                        }

                        for (int p = 0; p <= t; p++)
                        {
                            long ki = (long)(b * steps + p) * kRow + h * dim;
                            double dot = 0, dP = 0;
                            for (int d = 0; d < dim; d++)
                            {
                                dot += (double)qv[qOffset + qi + d] * kv[kOffset + ki + d];
                                dP += (double)dov[yi + d] * vv[vOffset + ki + d];
                            }

                            double probability = Math.Exp(scale * dot - lseT), dS = probability * (dP - delta);
                            for (int d = 0; d < dim; d++)
                            {
                                dqa[dqOffset + qi + d] += scale * dS * kv[kOffset + ki + d];
                                dka[dkOffset + ki + d] += scale * dS * qv[qOffset + qi + d];
                                dva[dvOffset + ki + d] += probability * dov[yi + d];
                            }
                        }
                    }
                }
            }
        }

        foreach (var (storage, (values, added)) in gradients)
        {
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = (float)(values[i] + added[i]);
            }

            cpu.Upload(values, storage);
        }
    }

    // Attention read in place from [batch, steps, *] rows and its gradients, against plain loops: the packed projections
    // as Tensor.CausalAttentionPacked passes them (one storage, three offsets; one for the gradients too) and separate
    // tensors with offsets and padded rows; head sizes 64 and 128 (the matrix-unit kernels) and one more.
    private static void StridedAttention(DeviceCaseContext c)
    {
        foreach (int dim in new[] { 64, 128, 32 })
        {
            int batch = c.Size(1, 2), kvHeads = c.Size(1, 2), group = c.Random.Next(1, 3), heads = kvHeads * group, steps = c.Size(1, dim == 128 ? 40 : 80);
            float scale = 1f / MathF.Sqrt(dim);
            foreach (bool packed in new[] { true, false })
            {
                int qRow, kRow;
                long qOffset, kOffset, vOffset, dqOffset, dkOffset, dvOffset;
                Storage q, k, v, dq, dk, dv;
                if (packed)
                {
                    int width = (heads + 2 * kvHeads) * dim;
                    (qRow, kRow, qOffset, kOffset, vOffset) = (width, width, 0, (long)heads * dim, (long)(heads + kvHeads) * dim);
                    (dqOffset, dkOffset, dvOffset) = (qOffset, kOffset, vOffset);
                    q = k = v = Random(c, batch * steps * width);
                    dq = dk = dv = Random(c, batch * steps * width);
                }
                else
                {
                    qRow = heads * dim + 4 * c.Random.Next(3);
                    kRow = kvHeads * dim + 4 * c.Random.Next(3);
                    (qOffset, kOffset, vOffset) = (4 * c.Random.Next(3), 4 * c.Random.Next(3), 4 * c.Random.Next(3));
                    (dqOffset, dkOffset, dvOffset) = (4 * c.Random.Next(3), 4 * c.Random.Next(3), 4 * c.Random.Next(3));
                    q = Random(c, (int)qOffset + batch * steps * qRow, 2f);
                    k = Random(c, (int)kOffset + batch * steps * kRow);
                    v = Random(c, (int)vOffset + batch * steps * kRow);
                    dq = Random(c, (int)dqOffset + batch * steps * qRow);
                    dk = Random(c, (int)dkOffset + batch * steps * kRow);
                    dv = Random(c, (int)dvOffset + batch * steps * kRow);
                }

                int outputs = batch * steps * heads * dim, rows = batch * heads * steps;
                Storage y = Random(c, outputs), lse = Random(c, rows), dOutput = Random(c, outputs);
                foreach (var logSumExp in new[] { null, lse })
                {
                    c.Composed(Ops.AttentionStrided,
                        cpu => cpu.AttentionStrided(q, qOffset, k, kOffset, v, vOffset, qRow, kRow, y, logSumExp, batch, kvHeads, group, steps, dim, scale), cpu =>
                        {
                            float[] yv = Read(y), lv = Read(lse);
                            StridedAttentionReference(Read(q), qOffset, Read(k), kOffset, Read(v), vOffset, qRow, kRow, yv, logSumExp is null ? null : lv,
                                batch, kvHeads, group, steps, dim, scale);
                            cpu.Upload(yv, y);
                            cpu.Upload(lv, lse);
                        });
                }

                c.Composed(Ops.AttentionStridedBackward, cpu => cpu.AttentionStridedBackward(q, qOffset, k, kOffset, v, vOffset, qRow, kRow, y, lse, dOutput,
                        dq, dqOffset, dk, dkOffset, dv, dvOffset, batch, kvHeads, group, steps, dim, scale),
                    cpu => StridedAttentionBackwardReference(cpu, q, qOffset, k, kOffset, v, vOffset, qRow, kRow, y, lse, dOutput, dq, dqOffset, dk, dkOffset,
                        dv, dvOffset, batch, kvHeads, group, steps, dim, scale));
            }
        }
    }
}
