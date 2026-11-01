// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Gpu.Abstractions;
using Idrak.Gpu.Hip;
using Idrak.Tests.Gcn;

// Plan 35, row order: the row kernels' sums grouped as CUDA's BlockReduce groups them (RowWriter.ReduceAdd), checked bit
// for bit against a C# model of CUDA's order that takes the interpreter's own v_rcp_f32, v_rsq_f32, v_exp_f32 and
// v_log_f32 (GcnFloat), so only the grouping is under test.
internal abstract partial class Tests
{
    // A block of 256 threads' partials summed in CUDA's order: each warp's butterfly over offsets 16 … 1 (lane l with lane
    // l + offset), then the 8 warps' sums added in order to 0.
    private static float CudaBlockSum(float[] partials)
    {
        float sum = 0f;
        for (int warp = 0; warp < partials.Length / 32; warp++)
        {
            float[] lanes = partials[(warp * 32)..(warp * 32 + 32)];
            for (int offset = 16; offset >= 1; offset /= 2)
            {
                for (int l = 0; l < offset; l++)
                {
                    lanes[l] += lanes[l + offset];
                }
            }

            sum += lanes[0];
        }

        return sum;
    }

    // Each thread's partial over its columns t, t + 256, … in order, `step` folding a column into it from `start`.
    private static float CudaRowSum(ReadOnlySpan<float> row, float start, Func<float, float, float> step)
    {
        var partials = new float[GcnKernels.RowThreads];
        for (int t = 0; t < partials.Length; t++)
        {
            float acc = start;
            for (int c = t; c < row.Length; c += GcnKernels.RowThreads)
            {
                acc = step(acc, row[c]);
            }

            partials[t] = acc;
        }

        return CudaBlockSum(partials);
    }

    private static void RowOrderSame(float[] expected, float[] got, string what)
    {
        for (int i = 0; i < expected.Length; i++)
        {
            Check(BitConverter.SingleToUInt32Bits(expected[i]) == BitConverter.SingleToUInt32Bits(got[i]),
                $"{what}: element {i} is {got[i]:R}, CUDA's order gives {expected[i]:R}");
        }
    }

    internal static void GcnRowsCudaOrder(Device device)
    {
        _ = device;
        var random = new Random(83);
        const int Rows = 3;
        const float Eps = 1e-5f;
        float log2E = BitConverter.UInt32BitsToSingle(0x3fb8aa3b), ln2 = BitConverter.UInt32BitsToSingle(0x3f317218);
        foreach (int cols in new[] { 1, 37, 1000, 2333 })
        {
            float[] x = RowsRandom(random, Rows * cols, 3f), gamma = RowsRandom(random, cols, 1f, 1f), beta = RowsRandom(random, cols, 0.5f);
            float rc = GcnFloat.Rcp(cols);
            var rmsY = new float[Rows * cols];
            var rmsInv = new float[Rows];
            var lnY = new float[Rows * cols];
            var stats = new float[2 * Rows];
            var softmax = new float[Rows * cols];
            var logSoftmax = new float[Rows * cols];
            for (int r = 0; r < Rows; r++)
            {
                var row = x.AsSpan(r * cols, cols);

                // rms_norm_f32: Σ x² (fused), inv = rsq(eps + Σ · (1 / cols)), y = x · inv.
                float squares = CudaRowSum(row, 0f, (acc, v) => MathF.FusedMultiplyAdd(v, v, acc));
                float inv = GcnFloat.Rsq(Eps + squares * rc);
                rmsInv[r] = inv;
                for (int c = 0; c < cols; c++)
                {
                    rmsY[r * cols + c] = row[c] * inv;
                }

                // layernorm_train_f32: the mean, the variance around it, rstd = rsq(eps + var), y = fma((x - mean) · rstd, γ, β).
                float mean = CudaRowSum(row, 0f, (acc, v) => acc + v) * rc;
                float variance = CudaRowSum(row, 0f, (acc, v) => MathF.FusedMultiplyAdd(v - mean, v - mean, acc)) * rc;
                float rstd = GcnFloat.Rsq(Eps + variance);
                stats[r] = mean;
                stats[Rows + r] = rstd;
                for (int c = 0; c < cols; c++)
                {
                    lnY[r * cols + c] = MathF.FusedMultiplyAdd((row[c] - mean) * rstd, gamma[c], beta[c]);
                }

                // softmax_f32: e = 2^(log2 e · (x - max)), y = e · rcp(Σ e); log-softmax: x - fma(log2 Σ, ln 2, max).
                float max = float.NegativeInfinity;
                foreach (float v in row)
                {
                    max = MathF.Max(max, v);
                }

                float total = CudaRowSum(row, 0f, (acc, v) => acc + GcnFloat.Exp(log2E * (v - max)));
                float reciprocal = GcnFloat.Rcp(total), shift = MathF.FusedMultiplyAdd(GcnFloat.Log(total), ln2, max);
                for (int c = 0; c < cols; c++)
                {
                    softmax[r * cols + c] = GcnFloat.Exp(log2E * (row[c] - max)) * reciprocal;
                    logSoftmax[r * cols + c] = row[c] - shift;
                }
            }

            foreach (var isa in RowsIsas)
            {
                string how = $"({isa}), {Rows} rows of {cols}";
                float[][] rms = RowsOnGcn([x, new float[Rows * cols], new float[Rows]],
                    (m, a) => RowDispatch("rms_norm_f32", isa, m, (2, 1), 1, a[0], a[1], a[2], (ulong)cols, Bits(Eps), Rows));
                RowOrderSame(rmsY, rms[1], $"rms_norm_f32 {how}, y");
                RowOrderSame(rmsInv, rms[2], $"rms_norm_f32 {how}, inv");
                float[][] ln = RowsOnGcn([x, gamma, beta, new float[Rows * cols], new float[2 * Rows]],
                    (m, a) => RowDispatch("layernorm_train_f32", isa, m, (2, 1), 1, a[0], a[1], a[2], a[3], a[4], (ulong)cols, Bits(Eps), Rows));
                RowOrderSame(lnY, ln[3], $"layernorm_train_f32 {how}, y");
                RowOrderSame(stats, ln[4], $"layernorm_train_f32 {how}, stats");
                foreach (bool log in new[] { false, true })
                {
                    float[][] soft = RowsOnGcn([x, new float[Rows * cols]],
                        (m, a) => RowDispatch("softmax_f32", isa, m, (2, 1), 1, a[0], a[1], (ulong)cols, log ? 1UL : 0UL, Rows));
                    RowOrderSame(log ? logSoftmax : softmax, soft[1], $"softmax_f32{(log ? " (log)" : "")} {how}");
                }
            }
        }
    }
}
