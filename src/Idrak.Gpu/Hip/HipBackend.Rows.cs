// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Hip;

// The row operations (HipKernels.Rows.cs: one block per row: softmaxes, the best k log-probabilities, layer norm) and
// GELU after a bias.
internal sealed unsafe partial class HipBackend
{
    public override void SoftmaxKernel(Storage x, Storage y, int rows, int cols, bool log)
    {
        if (!Rows("softmax_f32", rows, P(x), P(y), U(rows), U(cols), U(log ? 1 : 0)))
        {
            base.SoftmaxKernel(x, y, rows, cols, log);
        }
    }

    // The candidates found on the device and only they read back (a beam search's step reads k values a row, not the row).
    public override void LogSoftmaxTopKKernel(Storage x, Storage values, Storage indices, int rows, int cols, int k)
    {
        if (k < 0 || !Rows("logsoftmax_topk_f32", rows, P(x), P(values), P(indices), U(rows), U(cols), U(k)))
        {
            base.LogSoftmaxTopKKernel(x, values, indices, rows, cols, k);
        }
    }

    public override void ScaleMaskSoftmaxKernel(Storage x, Storage? mask, Storage y, int rows, int cols, int maskRows, float scale)
    {
        if (mask is not null && maskRows <= 0
            || !Rows("scale_mask_softmax_f32", rows, P(x), mask is null ? 0UL : P(mask), P(y), U(rows), U(cols), U(Math.Max(1, maskRows)), F(scale)))
        {
            base.ScaleMaskSoftmaxKernel(x, mask, y, rows, cols, maskRows, scale);
        }
    }

    public override void LayerNormFusedKernel(Storage x, Storage gamma, Storage beta, Storage y, int rows, int cols, float eps)
    {
        if (!Rows("layer_norm_f32", rows, P(x), P(gamma), P(beta), P(y), 0UL, U(rows), U(cols), F(eps)))
        {
            base.LayerNormFusedKernel(x, gamma, beta, y, rows, cols, eps);
        }
    }

    public override void LayerNormTrainKernel(Storage x, Storage gamma, Storage beta, Storage y, Storage stats, int rows, int cols, float eps)
    {
        if (!Rows("layer_norm_f32", rows, P(x), P(gamma), P(beta), P(y), P(stats), U(rows), U(cols), F(eps)))
        {
            base.LayerNormTrainKernel(x, gamma, beta, y, stats, rows, cols, eps);
        }
    }

    // The input's gradient one block per row, the gain's and shift's one thread per column (sums over the rows in order).
    public override void LayerNormBackwardKernel(Storage x, Storage gamma, Storage dy, Storage stats, Storage? dx, Storage? dgamma, Storage? dbeta, int rows, int cols)
    {
        if (rows > _device.Limits.MaxGridDimX || Kernels() is null)
        {
            base.LayerNormBackwardKernel(x, gamma, dy, stats, dx, dgamma, dbeta, rows, cols);
            return;
        }

        if (dx is not null)
        {
            Rows("layer_norm_backward_f32", rows, P(x), P(gamma), P(dy), P(stats), P(dx), U(rows), U(cols));
        }

        if (dgamma is not null || dbeta is not null)
        {
            Elements("layer_norm_params_f32", cols, P(x), P(dy), P(stats), dgamma is null ? 0UL : P(dgamma), dbeta is null ? 0UL : P(dbeta), U(rows), U(cols));
        }
    }

    public override void BiasGeluKernel(Storage x, Storage bias, Storage y, int n, int cols)
    {
        if (cols <= 0 || !Elements("bias_gelu_f32", n, P(x), P(bias), P(y), U(n), U(cols)))
        {
            base.BiasGeluKernel(x, bias, y, n, cols);
        }
    }
}
