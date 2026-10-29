// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Gpu.Hip;

// HIP's row operations on its own GCN kernels (GcnKernels.Rows.cs), where CUDA overrides them and with CUDA's argument
// rules: RMS norm with and without gain, layer norm fused, its training forward and its gradients, softmax and
// log-softmax, the scaled masked softmax and log-softmax top-k. Each launches blocks of the kernel's declared 256 over
// min(rows, the device's grid) blocks (the kernels walk the rows grid-stride, so any grid covers them; rule 84: the
// device's reported limit), labelled for --profile with its shape and floating-point operations (Cost, from the table at
// the top of GcnKernels.Rows.cs). Each takes the host fallback when the kernels are unavailable, as every operation
// without a kernel does, and for a row of no columns (the CPU's own answer).
internal sealed unsafe partial class HipBackend
{
    // Rows of a layer norm's gamma and beta gradients one block sums (CUDA's LayerNormBackwardKernel's 64): several
    // chunks' sums are added in chunk order by fold_chunks_f32 (rule 86), never by atomics.
    private const int ParameterChunk = 64;

    /// <summary>
    /// A kernel's floating-point operations and global-memory bytes per element, from the tables at the top of
    /// <c>GcnKernels.Rows.cs</c>, <c>.Pointwise.cs</c>, <c>.Attention.cs</c>, <c>.Shapes.cs</c> and <c>.Decoding.cs</c>, and the expansions' of
    /// <c>GcnKernels.Products.cs</c> (<paramref name="k"/>: the top-k's picks, int8_gemv_finish_f32's splits; masked softmax and
    /// log-softmax with their own byte counts, the adding permute and strided copy too; an attention kernel's per row, position
    /// and dimension); (0, 0) for a kernel the tables do not list (the products: <see cref="ProductCost"/>). The profile's
    /// labels and <c>--bench-hip</c>'s rates read them.
    /// </summary>
    internal static (double Flops, double Bytes) Cost(string kernel, int k = 0, bool variant = false) => kernel switch
    {
        "rms_norm_f32" => (3, 12),
        "rms_norm_affine_f32" => (5, 16),
        "layernorm_fused_f32" or "layernorm_train_f32" => (8, 24),
        "layernorm_bwd_f32" => (14, 32),
        "layernorm_bwd_params_f32" => (5, 8),
        "fold_chunks_f32" => (1, 4),
        "softmax_f32" => (6, variant ? 16 : 20),                                // variant: log-softmax
        "scale_mask_softmax_f32" => (8, variant ? 24 : 28),                      // variant: no mask
        "log_softmax_topk_f32" => (5 + 4.0 * k, 8 + 4.0 * k),
        "affine_f32" => (2, 8),
        "axpy_f32" or "muladd_f32" => (2, kernel == "axpy_f32" ? 12 : 16),
        "add_f32" or "sub_f32" or "mul_f32" or "add_rowvec_f32" => (1, 12),
        "max_f32" or "min_f32" => (2, 12),
        "relu_f32" or "square_f32" or "abs_f32" or "sign_f32" => (1, 8),
        "sigmoid_f32" => (28, 8),
        "tanh_f32" => (38, 8),
        "gelu_f32" => (46, 8),
        "gelu_erf_f32" => (45, 8),
        "silu_f32" => (29, 8),
        "exp_f32" => (10, 8),
        "log_f32" => (9, 8),
        "sqrt_f32" => (7, 8),
        "sin_f32" or "cos_f32" => (32, 8),
        "fill_f32" => (0, 4),
        "int8_dequant_f32" => (1, 5),                                           // per weight (GcnKernels.Products.cs)
        "int4_dequant_f32" => (1, 4.5),
        "bf16_dequant_f32" or "f16_dequant_f32" => (0, 6),
        "bf16_pack_f32" => (0, 6),                                              // per float
        "int8_gemv_finish_f32" => (k, 4.0 * (k + 1)),                            // per output; k: the splits
        _ => AttentionShapeCost(kernel, variant) is { Bytes: > 0 } cost ? cost : DecodingCost(kernel, k),
    };

    // A row kernel for `rows` × `cols`, or null: the kernels unavailable, or rows of no columns.
    private GcnFunction? RowKernel(string name, int cols) => cols > 0 ? Gcn(name) : null;

    // Launches a row kernel over min(rows, the device's grid) blocks, labelled for the profile with its shape and work.
    private void LaunchRows(GcnFunction function, int rows, int cols, double flopsPerElement, params ReadOnlySpan<ulong> values)
    {
        if (rows <= 0)
        {
            return;
        }

        if (Profiling)
        {
            LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"{function.Name} {rows}x{cols}"), flopsPerElement * rows * cols);
        }

        LaunchKernel(function, (uint)Math.Min(rows, DeviceLimits.MaxGridDimX), 1, values);
    }

    public override void RmsNormKernel(Storage x, Storage y, Storage inv, int rows, int cols, float eps)
    {
        if (RowKernel("rms_norm_f32", cols) is not { } f)
        {
            base.RmsNormKernel(x, y, inv, rows, cols, eps);
            return;
        }

        LaunchRows(f, rows, cols, Cost(f.Name).Flops, P(x), P(y), P(inv), U(cols), F(eps), U(rows));
    }

    public override void RmsNormAffineKernel(Storage x, Storage gain, Storage y, int rows, int cols, float eps, float offset)
    {
        if (RowKernel("rms_norm_affine_f32", cols) is not { } f)
        {
            base.RmsNormAffineKernel(x, gain, y, rows, cols, eps, offset);
            return;
        }

        LaunchRows(f, rows, cols, Cost(f.Name).Flops, P(x), P(gain), P(y), U(cols), F(eps), F(offset), U(rows));
    }

    public override void LayerNormFusedKernel(Storage x, Storage gamma, Storage beta, Storage y, int rows, int cols, float eps)
    {
        if (RowKernel("layernorm_fused_f32", cols) is not { } f)
        {
            base.LayerNormFusedKernel(x, gamma, beta, y, rows, cols, eps);
            return;
        }

        LaunchRows(f, rows, cols, Cost(f.Name).Flops, P(x), P(gamma), P(beta), P(y), U(cols), F(eps), U(rows));
    }

    public override void LayerNormTrainKernel(Storage x, Storage gamma, Storage beta, Storage y, Storage stats, int rows, int cols, float eps)
    {
        if (RowKernel("layernorm_train_f32", cols) is not { } f)
        {
            base.LayerNormTrainKernel(x, gamma, beta, y, stats, rows, cols, eps);
            return;
        }

        LaunchRows(f, rows, cols, Cost(f.Name).Flops, P(x), P(gamma), P(beta), P(y), P(stats), U(cols), F(eps), U(rows));
    }

    public override void LayerNormBackwardKernel(Storage x, Storage gamma, Storage dy, Storage stats, Storage? dx, Storage? dgamma, Storage? dbeta, int rows, int cols)
    {
        int chunks = (int)(((long)Math.Max(rows, 0) + ParameterChunk - 1) / ParameterChunk);
        var backward = RowKernel("layernorm_bwd_f32", cols);
        var parameters = RowKernel("layernorm_bwd_params_f32", cols);
        var fold = RowKernel("fold_chunks_f32", cols);
        if (backward is null || parameters is null || fold is null || chunks > DeviceLimits.MaxGridDimY || 2L * chunks * cols > int.MaxValue)
        {
            // No kernels, rows of no columns, more chunks than the device's grid holds in y (a larger chunk would change the
            // order of the sums, rule 86) or partials past one storage: the host's.
            base.LayerNormBackwardKernel(x, gamma, dy, stats, dx, dgamma, dbeta, rows, cols);
            return;
        }

        if (rows <= 0)
        {
            return;
        }

        if (dx is not null)
        {
            LaunchRows(backward, rows, cols, Cost(backward.Name).Flops, P(x), P(gamma), P(dy), P(stats), P(dx), U(cols), U(rows));
        }

        if (dgamma is null && dbeta is null)
        {
            return;
        }

        // Chunks of rows in parallel (grid y); several chunks' sums are added in chunk order by fold_chunks_f32.
        var partials = chunks > 1 ? Allocate(2 * chunks * cols, zeroed: false) : null;
        try
        {
            if (Profiling)
            {
                LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"{parameters.Name} {rows}x{cols}"), Cost(parameters.Name).Flops * rows * cols);
            }

            LaunchKernel(parameters, BlocksFor(cols, parameters.Block), (uint)chunks,
                P(x), P(dy), P(stats), dgamma is null ? 0UL : P(dgamma), dbeta is null ? 0UL : P(dbeta), partials is null ? P(x) : P(partials), U(rows), U(cols), U(ParameterChunk));
            if (partials is not null)
            {
                if (dgamma is not null)
                {
                    Fold(fold, P(partials), P(dgamma), cols, chunks);
                }

                if (dbeta is not null)
                {
                    Fold(fold, P(partials) + (ulong)chunks * (ulong)cols * sizeof(float), P(dbeta), cols, chunks);
                }
            }
        }
        finally
        {
            partials?.Release();                                                 // reused in stream order
        }
    }

    // y[i] += Σ over the chunks of partials[c · cols + i], in chunk order.
    private void Fold(GcnFunction fold, ulong partials, ulong y, int cols, int chunks)
    {
        if (Profiling)
        {
            LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"{fold.Name} {chunks}x{cols}"), Cost(fold.Name).Flops * chunks * cols);
        }

        LaunchKernel(fold, BlocksFor(cols, fold.Block), 1, partials, y, U(cols), U(chunks), U(cols));
    }

    public override void SoftmaxKernel(Storage x, Storage y, int rows, int cols, bool log)
    {
        if (RowKernel("softmax_f32", cols) is not { } f)
        {
            base.SoftmaxKernel(x, y, rows, cols, log);
            return;
        }

        LaunchRows(f, rows, cols, Cost(f.Name, variant: log).Flops, P(x), P(y), U(cols), U(log ? 1 : 0), U(rows));
    }

    public override void ScaleMaskSoftmaxKernel(Storage x, Storage? mask, Storage y, int rows, int cols, int maskRows, float scale)
    {
        if (RowKernel("scale_mask_softmax_f32", cols) is not { } f)
        {
            base.ScaleMaskSoftmaxKernel(x, mask, y, rows, cols, maskRows, scale);
            return;
        }

        // No mask: x stands for the mask pointer, one mask row, hasmask 0 (G3b's rule, CUDA's arguments).
        LaunchRows(f, rows, cols, Cost(f.Name, variant: mask is null).Flops, P(x), mask is null ? P(x) : P(mask), P(y),
            U(cols), U(mask is null ? 1 : Math.Max(maskRows, 1)), F(scale), U(mask is null ? 0 : 1), U(rows));
    }

    public override void LogSoftmaxTopKKernel(Storage x, Storage values, Storage indices, int rows, int cols, int k)
    {
        if (RowKernel("log_softmax_topk_f32", cols) is not { } f || k < 0)
        {
            base.LogSoftmaxTopKKernel(x, values, indices, rows, cols, k);
            return;
        }

        LaunchRows(f, rows, cols, Cost(f.Name, k).Flops, P(x), P(values), P(indices), U(cols), U(k), U(rows));
    }
}
