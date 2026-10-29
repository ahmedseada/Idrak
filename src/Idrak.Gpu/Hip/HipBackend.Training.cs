// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.InteropServices;
using Idrak.Abstraction.Devices;

namespace Idrak.Gpu.Hip;

// HIP's training operations on its own GCN kernels (GcnKernels.Training.cs), where CUDA runs them on the device and with
// CUDA's launch rules: the backward passes (unary, softmax, RMS norm, grouped norms, gated activations, extremes,
// selection, clamps, powers, leaky ReLU), the optimizers (SGD with momentum, Adam, 8-bit Adam, AdamW over many tensors,
// the clipping factor, dropout), the reductions (sums of rows, columns, axes, groups and squares, the whole sum,
// scatter-add, broadcasts, weight norm). Element-wise kernels launch as HIP's element-wise operations (PointwiseBlocks:
// rule 81, the grid-stride loop takes the rest), row kernels over min(rows, the device's grid) blocks (rule 84). Every sum
// over blocks or chunks is a second launch in block or chunk order, never atomics (rule 86): the whole sum and the sums
// of squares over at most SumBlocks blocks (fixed by the count), column sums in chunks of 64 rows folded by
// fold_chunks_f32, groups of two chunks or more split into chunks of GcnKernels.GroupChunk merged per group. Each launch
// is labelled for --profile with its kernel, its shape and its floating-point operations (TrainingCost); the host
// fallback takes over when the kernels are unavailable, a count reaches 2^30 (the kernels' 32-bit byte offsets) or a
// split would pass the device's grid.
internal sealed unsafe partial class HipBackend
{
    // The most blocks the whole sum's and the sums of squares' first pass take: a fixed number (rule 86), CUDA's.
    private const int SumBlocks = 1024;

    // Rows of a column sum one block adds (CUDA's 64): several chunks are folded in chunk order by fold_chunks_f32.
    private const int ColumnChunk = 64;

    /// <summary>The cost of a training kernel from the table at the top of GcnKernels.Training.cs; (0, 0) for another.</summary>
    internal static (double Flops, double Bytes) TrainingCost(string kernel) => kernel switch
    {
        "sigmoid_bwd_f32" or "tanh_bwd_f32" or "square_bwd_f32" => (3, 16),
        "relu_bwd_f32" => (1, 16),
        "abs_bwd_f32" or "exp_bwd_f32" => (2, 16),
        "log_bwd_f32" => (12, 16),
        "gelu_bwd_f32" => (52, 16),
        "gelu_erf_bwd_f32" => (44, 16),
        "sqrt_bwd_f32" => (13, 16),
        "sin_bwd_f32" or "cos_bwd_f32" or "silu_bwd_f32" => (34, 16),
        "extremum_bwd_f32" => (2, 28),
        "pow_bwd_f32" => (10, 16),
        "clamp_bwd_f32" => (1, 16),
        "leaky_relu_bwd_f32" => (2, 16),
        "where_bwd_f32" => (2, 24),
        "gated_act_bwd_f32" => (40, 28),
        "gated_act_bwd_bf16_f32" => (40, 24),
        "softmax_bwd_f32" => (5, 20),
        "rms_norm_backward_f32" => (6, 24),
        "norm_bwd_f32" => (30, 28),
        "sgd_momentum_f32" => (4, 20),
        "adam_f32" => (28, 28),
        "adam8_f32" => (50, 14),
        "multi_sumsq_f32" => (2, 4),
        "multi_adamw_f32" => (30, 32),
        "clip_factor_f32" => (20, 8),
        "dropout_f32" => (2, 8),
        "add_dropout_f32" => (2, 12),
        "sum_rows_f32" or "sum_cols_strided_f32" or "sum_f32" or "sum_blocks_f32" or "sum_axis_f32" or "sum_axis_rows_f32" => (1, 4),
        "sumsq_f32" => (2, 4),
        "broadcast_axis_f32" => (2, 12),
        "scatter_chain_u32" => (0, 0.02),
        "scatter_walk_f32" => (1, 12),
        "norm_stats_f32" or "norm_stats_part_f32" or "group_reduce_f32" or "group_reduce_part_f32" => (3, 8),
        "norm_stats_final_f32" => (10, 8),
        "group_reduce_final_f32" => (2, 8),
        "norm_apply_f32" => (3, 16),
        "group_scale_shift_f32" => (2, 16),
        "weight_norm_f32" => (3, 12),
        "weight_norm_bwd_f32" => (8, 24),
        "attn_bwd_d_f32" => (2, 8),
        "attn_bwd_q_f32" or "attn_spans_bwd_q_f32" => (6, 8),
        "attn_bwd_kv_f32" or "attn_spans_bwd_kv_f32" => (8, 8),
        _ => ConvolutionCost(kernel),                                       // G5c's tables (convolution, CTC, recurrent)
    };

    /// <summary>The backward kernel of a unary operation, or null for one without (CUDA's set; sign's gradient is zero).</summary>
    internal static string? UnaryBackwardKernelName(UnaryOp op) => op switch
    {
        UnaryOp.Sigmoid => "sigmoid_bwd_f32",
        UnaryOp.Tanh => "tanh_bwd_f32",
        UnaryOp.Relu => "relu_bwd_f32",
        UnaryOp.Square => "square_bwd_f32",
        UnaryOp.Abs => "abs_bwd_f32",
        UnaryOp.Exp => "exp_bwd_f32",
        UnaryOp.Log => "log_bwd_f32",
        UnaryOp.Gelu => "gelu_bwd_f32",
        UnaryOp.Sqrt => "sqrt_bwd_f32",
        UnaryOp.Sin => "sin_bwd_f32",
        UnaryOp.Cos => "cos_bwd_f32",
        UnaryOp.Silu => "silu_bwd_f32",
        UnaryOp.GeluErf => "gelu_erf_bwd_f32",
        _ => null,
    };

    // Launches a kernel over `blocks` × `blocksY` blocks, labelled for the profile with its name, a shape and its work.
    private void LaunchLabelled(GcnFunction function, uint blocks, uint blocksY, string shape, double flops, params ReadOnlySpan<ulong> values)
    {
        if (Profiling)
        {
            LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"{function.Name} {shape}"), flops);
        }

        LaunchKernel(function, blocks, blocksY, values);
    }

    // Blocks for `work` rows or blocks of a grid-stride kernel: within the device's grid in x.
    private uint GridFor(long work) => (uint)Math.Clamp(work, 1, DeviceLimits.MaxGridDimX);

    // ---- Element-wise backward passes ----------------------------------------------------------------------------------

    public override void UnaryBackwardKernel(UnaryOp op, Storage x, Storage y, Storage dy, Storage dx, int n)
    {
        if (op == UnaryOp.Sign)
        {
            return;                                                                 // sign's gradient is zero: dx stays as it is
        }

        if (PointwiseKernel(UnaryBackwardKernelName(op), n) is not { } f)
        {
            base.UnaryBackwardKernel(op, x, y, dy, dx, n);
            return;
        }

        LaunchPointwise(f, n, P(x), P(y), P(dy), P(dx), U(n));
    }

    public override void ExtremumBackwardKernel(BinaryOp op, Storage a, Storage b, Storage dy, Storage? da, Storage? db, int n)
    {
        if (op is not (BinaryOp.Maximum or BinaryOp.Minimum) || PointwiseKernel("extremum_bwd_f32", n) is not { } f)
        {
            base.ExtremumBackwardKernel(op, a, b, dy, da, db, n);
            return;
        }

        int flags = (da is null ? 0 : 1) | (db is null ? 0 : 2) | (op == BinaryOp.Minimum ? 4 : 0);
        if ((flags & 3) != 0)
        {
            LaunchPointwise(f, n, P(a), P(b), P(dy), P(da ?? dy), P(db ?? dy), U(flags), U(n));
        }
    }

    public override void PowBackwardKernel(Storage x, Storage dy, Storage dx, int n, float exponent)
    {
        if (exponent == 0f)
        {
            return;                                                                 // a constant: no gradient (not 0 · x^-1, NaN at 0)
        }

        if (PointwiseKernel("pow_bwd_f32", n) is not { } f)
        {
            base.PowBackwardKernel(x, dy, dx, n, exponent);
            return;
        }

        float p = exponent - 1f;
        LaunchPointwise(f, n, P(x), P(dy), P(dx), F(exponent), F(p), GcnKernels.PowGradientFlags(p), U(n));
    }

    public override void ClampBackwardKernel(Storage x, Storage dy, Storage dx, int n, float min, float max)
    {
        if (PointwiseKernel("clamp_bwd_f32", n) is not { } f)
        {
            base.ClampBackwardKernel(x, dy, dx, n, min, max);
            return;
        }

        LaunchPointwise(f, n, P(x), P(dy), P(dx), F(min), F(max), U(n));
    }

    public override void LeakyReluBackwardKernel(Storage x, Storage dy, Storage dx, int n, float slope)
    {
        if (PointwiseKernel("leaky_relu_bwd_f32", n) is not { } f)
        {
            base.LeakyReluBackwardKernel(x, dy, dx, n, slope);
            return;
        }

        LaunchPointwise(f, n, P(x), P(dy), P(dx), F(slope), U(n));
    }

    public override void WhereBackwardKernel(Storage condition, Storage dy, Storage? da, Storage? db, int n)
    {
        if (PointwiseKernel("where_bwd_f32", n) is not { } f)
        {
            base.WhereBackwardKernel(condition, dy, da, db, n);
            return;
        }

        int flags = (da is null ? 0 : 1) | (db is null ? 0 : 2);
        if (flags != 0)
        {
            LaunchPointwise(f, n, P(condition), P(dy), P(da ?? dy), P(db ?? dy), U(flags), U(n));
        }
    }

    public override void GatedActivationBackwardKernel(Storage gate, Storage up, Storage dy, Storage dgate, Storage dup, int n, int kind, int flags)
    {
        if (PointwiseKernel("gated_act_bwd_f32", n) is not { } f)
        {
            base.GatedActivationBackwardKernel(gate, up, dy, dgate, dup, n, kind, flags);
            return;
        }

        if ((flags & 3) != 0)
        {
            LaunchPointwise(f, n, P(gate), P(up), P(dy), P(dgate), P(dup), U(kind), U(flags), U(n));
        }
    }

    // The gated activation's gradient with gate and up packed as bfloat16 (two to a word): a work-item a word, its two
    // elements in turn (gated_act_bwd_bf16_f32, CUDA's), no unpacked copy of either.
    public override void GatedActivationBackwardPackedKernel(Storage packedGate, Storage packedUp, Storage dy, Storage dgate, Storage dup, int n, int kind, int flags)
    {
        if (PointwiseKernel("gated_act_bwd_bf16_f32", n) is not { } f)
        {
            base.GatedActivationBackwardPackedKernel(packedGate, packedUp, dy, dgate, dup, n, kind, flags);
            return;
        }

        if ((flags & 3) != 0 && n > 0)
        {
            int words = (n + 1) / 2;
            if (Profiling)
            {
                LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"{f.Name} {n}"), Cost(f.Name).Flops * n);
            }

            LaunchKernel(f, PointwiseBlocks(words), 1, P(packedGate), P(packedUp), P(dy), P(dgate), P(dup), U(kind), U(flags), U(n), U(words));
        }
    }

    public override void SoftmaxBackwardKernel(Storage y, Storage dy, Storage dx, int rows, int cols, bool log)
    {
        if (RowKernel("softmax_bwd_f32", cols) is not { } f)
        {
            base.SoftmaxBackwardKernel(y, dy, dx, rows, cols, log);
            return;
        }

        LaunchRows(f, rows, cols, TrainingCost(f.Name).Flops, P(y), P(dy), P(dx), U(cols), U(log ? 1 : 0), U(rows));
    }

    public override void RmsNormBackwardKernel(Storage dy, Storage y, Storage inv, Storage dx, int rows, int cols)
    {
        if (RowKernel("rms_norm_backward_f32", cols) is not { } f)
        {
            base.RmsNormBackwardKernel(dy, y, inv, dx, rows, cols);
            return;
        }

        LaunchRows(f, rows, cols, TrainingCost(f.Name).Flops, P(dy), P(y), P(inv), P(dx), U(cols), U(rows));
    }

    public override void NormBackwardKernel(Storage dxhat, Storage xhat, Storage sum1, Storage sum2, Storage invStd, Storage dx, int outer, int groups, int inner)
    {
        long n = (long)outer * groups * inner;
        if (groups <= 0 || inner <= 0 || PointwiseKernel("norm_bwd_f32", n) is not { } f)
        {
            base.NormBackwardKernel(dxhat, xhat, sum1, sum2, invStd, dx, outer, groups, inner);
            return;
        }

        LaunchPointwise(f, (int)n, P(dxhat), P(xhat), P(sum1), P(sum2), P(invStd), P(dx), U(groups), U(inner), F(outer * inner), U((int)n));
    }

    // ---- Optimizers and dropout ---------------------------------------------------------------------------------------

    public override void SgdStepKernel(Storage p, Storage g, Storage? v, int n, float lr, float momentum)
    {
        if (v is null)
        {
            Axpy(g, p, n, -lr);
            return;
        }

        if (PointwiseKernel("sgd_momentum_f32", n) is not { } f)
        {
            base.SgdStepKernel(p, g, v, n, lr, momentum);
            return;
        }

        LaunchPointwise(f, n, P(p), P(g), P(v), F(-lr), F(momentum), U(n));
    }

    public override void AdamStepKernel(Storage p, Storage g, Storage m, Storage v, int n, float lr, float beta1, float beta2, float eps)
    {
        if (PointwiseKernel("adam_f32", n) is not { } f)
        {
            base.AdamStepKernel(p, g, m, v, n, lr, beta1, beta2, eps);
            return;
        }

        LaunchPointwise(f, n, P(p), P(g), P(m), P(v), F(lr), F(beta1), F(beta2), F(1f - beta1), F(1f - beta2), F(eps), U(n));
    }

    public override void AdamStep8BitKernel(Storage p, Storage g, Storage m, Storage v, Storage absMax, Storage map, int n, float lr, float beta1, float beta2, float eps,
        float gradientScale, float decay)
    {
        if (EightBitMoments.BlockSize != GcnKernels.Adam8Block || n >= GcnKernels.PointwiseMostElements || Gcn("adam8_f32") is not { } f)
        {
            base.AdamStep8BitKernel(p, g, m, v, absMax, map, n, lr, beta1, beta2, eps, gradientScale, decay);
            return;
        }

        if (n <= 0)
        {
            return;
        }

        int blocks = (n + GcnKernels.Adam8Block - 1) / GcnKernels.Adam8Block;
        LaunchLabelled(f, GridFor(blocks), 1, n.ToString(CultureInfo.InvariantCulture), TrainingCost(f.Name).Flops * n, P(p), P(g), P(m), P(v), P(absMax), P(map),
            F(lr), F(beta1), F(beta2), F(1f - beta1), F(1f - beta2), F(eps), U(n), F(gradientScale), F(decay), U(blocks));
    }

    // The device tables of FusedAdamW (CUDA's): chunks (tensor, first element) of 4,096 elements, rebuilt when the sizes
    // change; the tensors' pointers and counts, uploaded again only when a pointer changes; the factor; each chunk's sum of
    // squares (the clipping norm).
    private sealed class FusedAdamState(HipBackend backend) : IDisposable
    {
        public int[] Sizes = [];
        public ulong[] Pointers = [];
        public ulong[] Spare = [];
        public Storage? Chunks, Table, Factor, Partials;
        public int ChunkCount;

        public void Dispose()
        {
            foreach (var storage in new[] { Chunks, Table, Factor, Partials })
            {
                if (storage is not null)
                {
                    backend.Return(storage);
                }
            }

            (Chunks, Table, Factor, Partials) = (null, null, null, null);
        }
    }

    public override bool FusedAdamWKernel(ReadOnlySpan<(Storage P, Storage G, Storage M, Storage V, int N)> tensors, ref IDisposable? cache, float maxNorm,
        float lr, float decay, float beta1, float beta2, float eps, bool zeroGradients)
    {
        if (tensors.Length == 0 || Gcn("multi_adamw_f32") is not { } adam || Gcn("multi_sumsq_f32") is not { } sumSquares || Gcn("sum_blocks_f32") is not { } blocksSum
            || Gcn("clip_factor_f32") is null)
        {
            return false;
        }

        foreach (var t in tensors)
        {
            if (t.N >= GcnKernels.PointwiseMostElements)
            {
                return false;
            }
        }

        var state = cache as FusedAdamState;
        if (state is null)
        {
            cache?.Dispose();
            cache = state = new FusedAdamState(this);
        }

        bool resized = state.Sizes.Length != tensors.Length;
        for (int t = 0; t < tensors.Length && !resized; t++)
        {
            resized = state.Sizes[t] != tensors[t].N;
        }

        if (resized)
        {
            state.Sizes = new int[tensors.Length];
            var chunks = new List<uint>();
            for (int t = 0; t < tensors.Length; t++)
            {
                state.Sizes[t] = tensors[t].N;
                for (int start = 0; start < Math.Max(1, tensors[t].N); start += GcnKernels.MultiAdamChunk)
                {
                    chunks.Add((uint)t);
                    chunks.Add((uint)start);
                }
            }

            if (state.Chunks is not null)
            {
                Return(state.Chunks);
            }

            if (state.Partials is not null)
            {
                Return(state.Partials);
            }

            state.ChunkCount = chunks.Count / 2;
            state.Chunks = Allocate(chunks.Count, zeroed: false);
            state.Partials = Allocate(state.ChunkCount, zeroed: false);
            Upload(MemoryMarshal.Cast<uint, float>(CollectionsMarshal.AsSpan(chunks)), state.Chunks);
            state.Pointers = [];
        }

        if (state.Spare.Length != tensors.Length * 5)
        {
            state.Spare = new ulong[tensors.Length * 5];
        }

        var pointers = state.Spare;
        for (int t = 0; t < tensors.Length; t++)
        {
            (pointers[5 * t], pointers[5 * t + 1], pointers[5 * t + 2], pointers[5 * t + 3], pointers[5 * t + 4]) =
                (P(tensors[t].P), P(tensors[t].G), P(tensors[t].M), P(tensors[t].V), (uint)tensors[t].N);
        }

        if (!pointers.AsSpan().SequenceEqual(state.Pointers))
        {
            (state.Pointers, state.Spare) = (pointers, state.Pointers);
            if (state.Table is not null && state.Table.Length != tensors.Length * 10)
            {
                Return(state.Table);
                state.Table = null;
            }

            state.Table ??= Allocate(tensors.Length * 10, zeroed: false);
            Upload(MemoryMarshal.Cast<ulong, float>(state.Pointers.AsSpan()), state.Table);
        }

        state.Factor ??= Allocate(1, zeroed: false);
        string shape = string.Create(CultureInfo.InvariantCulture, $"{tensors.Length} tensors, {state.ChunkCount} chunks");
        double elements = (double)state.ChunkCount * GcnKernels.MultiAdamChunk;
        if (maxNorm > 0f)
        {
            // Each chunk's sum of squares, then their sum in chunk order (rule 86: the clipping factor the same on every run).
            LaunchLabelled(sumSquares, GridFor(state.ChunkCount), 1, shape, TrainingCost(sumSquares.Name).Flops * elements, P(state.Chunks!), P(state.Table!),
                P(state.Partials!), U(state.ChunkCount));
            LaunchLabelled(blocksSum, 1, 1, shape, state.ChunkCount, P(state.Partials!), P(state.Factor), U(state.ChunkCount), F(1f), U(0), U(1));
            ClipFactor(state.Factor, state.Factor, maxNorm);
        }
        else
        {
            Fill(state.Factor, 1, 1f);
        }

        LaunchLabelled(adam, GridFor(state.ChunkCount), 1, shape, TrainingCost(adam.Name).Flops * elements, P(state.Chunks!), P(state.Table!), P(state.Factor),
            F(lr), F(decay), F(beta1), F(beta2), F(1f - beta1), F(1f - beta2), F(eps), U(zeroGradients ? 1 : 0), U(state.ChunkCount));
        return true;
    }

    public override void ClipFactorKernel(Storage sumSquares, Storage factor, float maxNorm)
    {
        if (Gcn("clip_factor_f32") is not { } f)
        {
            base.ClipFactorKernel(sumSquares, factor, maxNorm);
            return;
        }

        LaunchLabelled(f, 1, 1, "1", TrainingCost(f.Name).Flops, P(sumSquares), P(factor), F(maxNorm), U(1));
    }

    public override void DropoutKernel(Storage x, Storage y, int n, float p, uint seed)
    {
        if (PointwiseKernel("dropout_f32", n) is not { } f)
        {
            base.DropoutKernel(x, y, n, p, seed);
            return;
        }

        LaunchPointwise(f, n, P(x), P(y), F(p), F(1f / (1f - p)), seed, 0UL, U(n));
    }

    public override void AddDropoutKernel(Storage residual, Storage x, Storage output, int n, float p, uint seed)
    {
        if (PointwiseKernel("add_dropout_f32", n) is not { } f)
        {
            base.AddDropoutKernel(residual, x, output, n, p, seed);
            return;
        }

        LaunchPointwise(f, n, P(residual), P(x), P(output), F(p), F(1f / (1f - p)), seed, U(n));
    }

    public override void DropoutBackwardKernel(Storage dy, Storage dx, int n, float p, uint seed)
    {
        if (PointwiseKernel("dropout_f32", n) is not { } f)
        {
            base.DropoutBackwardKernel(dy, dx, n, p, seed);
            return;
        }

        LaunchPointwise(f, n, P(dy), P(dx), F(p), F(1f / (1f - p)), seed, 1UL, U(n));
    }

    // ---- Sums ------------------------------------------------------------------------------------------------------------

    public override void SumRowsKernel(Storage x, Storage y, int rows, int cols)
    {
        if (rows >= 256)
        {
            SumColumnsKernel(x, 0, cols, y, rows, cols);                         // chunks of rows in parallel (a bias gradient)
            return;
        }

        if (cols <= 0 || (long)rows * cols >= GcnKernels.PointwiseMostElements || Gcn("sum_rows_f32") is not { } f)
        {
            base.SumRowsKernel(x, y, rows, cols);
            return;
        }

        LaunchLabelled(f, BlocksFor(cols, f.Block), 1, string.Create(CultureInfo.InvariantCulture, $"{rows}x{cols}"), (double)rows * cols, P(x), P(y), U(rows), U(cols));
    }

    // Chunks of rows in parallel; several chunks' sums are added in chunk order by fold_chunks_f32 (rule 86: no atomics).
    public override void SumColumnsKernel(Storage x, long offset, int ld, Storage y, int rows, int cols)
    {
        int chunks = (int)(((long)Math.Max(rows, 0) + ColumnChunk - 1) / ColumnChunk);
        if (cols <= 0 || rows <= 0 || offset < 0 || ld < cols || (long)chunks * cols >= GcnKernels.PointwiseMostElements || chunks > DeviceLimits.MaxGridDimY
            || Gcn("sum_cols_strided_f32") is not { } f || Gcn("fold_chunks_f32") is not { } fold)
        {
            base.SumColumnsKernel(x, offset, ld, y, rows, cols);
            return;
        }

        var partials = chunks > 1 ? Allocate(chunks * cols, zeroed: false) : null;
        try
        {
            LaunchLabelled(f, BlocksFor(cols, f.Block), (uint)chunks, string.Create(CultureInfo.InvariantCulture, $"{rows}x{cols}"), (double)rows * cols,
                P(x) + (ulong)offset * sizeof(float), P(y), partials is null ? P(y) : P(partials), U(rows), U(cols), U(ld), U(ColumnChunk));
            if (partials is not null)
            {
                Fold(fold, P(partials), P(y), cols, chunks);
            }
        }
        finally
        {
            partials?.Release();                                                 // reused in stream order
        }
    }

    // Blocks fixed by n (at most SumBlocks): one block stores the sum, several store theirs for sum_blocks_f32 to add in
    // block order (rule 86: the same bits on every run and card).
    public override void SumKernel(Storage x, Storage result, int n, float scale)
    {
        if (n >= GcnKernels.PointwiseMostElements || Gcn("sum_f32") is not { } f || Gcn("sum_blocks_f32") is not { } blocksSum)
        {
            base.SumKernel(x, result, n, scale);
            return;
        }

        if (n <= 0)
        {
            Fill(result, 1, 0f);
            return;
        }

        int blocks = Math.Min((n + GcnKernels.RowThreads - 1) / GcnKernels.RowThreads, SumBlocks);
        string shape = n.ToString(CultureInfo.InvariantCulture);
        if (blocks == 1)
        {
            LaunchLabelled(f, 1, 1, shape, n, P(x), P(result), P(result), U(n), F(scale));
            return;
        }

        var partials = Allocate(blocks, zeroed: false);
        try
        {
            LaunchLabelled(f, (uint)blocks, 1, shape, n, P(x), P(result), P(partials), U(n), F(scale));
            LaunchLabelled(blocksSum, 1, 1, shape, blocks, P(partials), P(result), U(blocks), F(scale), U(0), U(1));
        }
        finally
        {
            partials.Release();                                                  // reused in stream order
        }
    }

    // total[0] += Σ x²: one block adds its sum, several store theirs for sum_blocks_f32 to add in block order (rule 86).
    public override void SumSquaresKernel(Storage x, Storage total, int n)
    {
        if (n <= 0)
        {
            return;
        }

        if (n >= GcnKernels.PointwiseMostElements || Gcn("sumsq_f32") is not { } f || Gcn("sum_blocks_f32") is not { } blocksSum)
        {
            base.SumSquaresKernel(x, total, n);
            return;
        }

        int blocks = Math.Min(SumBlocks, (n + GcnKernels.RowThreads - 1) / GcnKernels.RowThreads);
        string shape = n.ToString(CultureInfo.InvariantCulture);
        if (blocks == 1)
        {
            LaunchLabelled(f, 1, 1, shape, 2.0 * n, P(x), P(total), P(total), U(n), U(1), U(1));
            return;
        }

        var partials = Allocate(blocks, zeroed: false);
        try
        {
            LaunchLabelled(f, (uint)blocks, 1, shape, 2.0 * n, P(x), P(total), P(partials), U(n), U(0), U(blocks));
            LaunchLabelled(blocksSum, 1, 1, shape, blocks, P(partials), P(total), U(blocks), F(1f), U(1), U(1));
        }
        finally
        {
            partials.Release();                                                  // reused in stream order
        }
    }

    public override void SumAxisKernel(Storage x, Storage y, int outer, int dim, int inner, float scale, bool accumulate)
    {
        long all = (long)outer * dim * inner;
        if (outer <= 0 || inner <= 0 || dim < 0 || all >= GcnKernels.PointwiseMostElements)
        {
            base.SumAxisKernel(x, y, outer, dim, inner, scale, accumulate);
            return;
        }

        if (inner == 1 && dim >= 64 && Gcn("sum_axis_rows_f32") is { } rows)
        {
            // A sum along the last axis: one block a row reads it coalesced, where the thread-per-output kernel reads each
            // row serially with its neighbours a whole row apart.
            LaunchRows(rows, outer, dim, 1, P(x), P(y), U(dim), F(scale), U(accumulate ? 1 : 0), U(outer));
            return;
        }

        if (PointwiseKernel("sum_axis_f32", (long)outer * inner) is not { } f)
        {
            base.SumAxisKernel(x, y, outer, dim, inner, scale, accumulate);
            return;
        }

        int n = outer * inner;
        if (Profiling)
        {
            LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"{f.Name} {outer}x{dim}x{inner}"), all);
        }

        LaunchKernel(f, PointwiseBlocks(n), 1, P(x), P(y), U(dim), U(inner), F(scale), U(accumulate ? 1 : 0), U(n));
    }

    public override void BroadcastAxisKernel(Storage dy, Storage dx, int outer, int dim, int inner, float scale)
    {
        long n = (long)outer * dim * inner;
        if (dim <= 0 || inner <= 0 || PointwiseKernel("broadcast_axis_f32", n) is not { } f)
        {
            base.BroadcastAxisKernel(dy, dx, outer, dim, inner, scale);
            return;
        }

        LaunchPointwise(f, (int)n, P(dy), P(dx), U(dim * inner), U(inner), F(scale), U((int)n));
    }

    // Each id's rows added in position order (rule 86): the positions linked id by id, then each chain walked by its head.
    public override void ScatterAddKernel(Storage dy, Storage indices, Storage dtable, int count, int dim, int vocabulary)
    {
        if (count == 0 || dim == 0)
        {
            return;
        }

        long n = (long)count * dim;
        if (count < 0 || dim < 0 || vocabulary <= 0 || (long)vocabulary * dim >= GcnKernels.PointwiseMostElements || PointwiseKernel("scatter_walk_f32", n) is not { } walk
            || Gcn("scatter_chain_u32") is not { } chain)
        {
            base.ScatterAddKernel(dy, indices, dtable, count, dim, vocabulary);
            return;
        }

        var links = Allocate(count, zeroed: false);
        try
        {
            int blocks = (count + GcnKernels.RowThreads - 1) / GcnKernels.RowThreads;
            LaunchLabelled(chain, GridFor(blocks), 1, count.ToString(CultureInfo.InvariantCulture), (double)count * count,
                P(indices), P(links), U(count), U(vocabulary - 1), U(blocks));
            if (Profiling)
            {
                LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"{walk.Name} {count}x{dim}"), n);
            }

            LaunchKernel(walk, PointwiseBlocks(n), 1, P(dy), P(links), P(indices), P(dtable), U(dim), U(vocabulary - 1), U(count), U((int)n));
        }
        finally
        {
            links.Release();                                                     // reused in stream order
        }
    }

    // ---- Grouped normalization -----------------------------------------------------------------------------------------

    // A group of at least two chunks (GcnKernels.GroupChunk elements; the sizes alone decide, as CUDA's GroupSplit) runs
    // split: a block a chunk of every group writes its part, then a block a group merges its chunks in chunk order, so a few
    // large groups (batch norm's channels) fill the device (rule 81). Smaller groups take one block each.
    private static bool GroupSplit(int outer, int groups, int inner, out int chunks)
    {
        long m = (long)outer * inner;
        chunks = (int)Math.Min((m + GcnKernels.GroupChunk - 1) / GcnKernels.GroupChunk, int.MaxValue);
        return groups > 0 && m >= 2L * GcnKernels.GroupChunk && m <= int.MaxValue && (long)chunks * groups <= int.MaxValue / 2;
    }

    // The [outer, groups, inner] view within the kernels' 32-bit byte offsets.
    private static bool GroupsFit(int outer, int groups, int inner) =>
        outer > 0 && groups > 0 && inner > 0 && (long)outer * groups * inner < GcnKernels.PointwiseMostElements;

    public override void NormStatsKernel(Storage x, Storage mean, Storage variance, Storage invStd, int outer, int groups, int inner, float eps)
    {
        if (groups <= 0)
        {
            return;
        }

        if (!GroupsFit(outer, groups, inner) || Gcn("norm_stats_f32") is not { } single || Gcn("norm_stats_part_f32") is not { } part
            || Gcn("norm_stats_final_f32") is not { } final)
        {
            base.NormStatsKernel(x, mean, variance, invStd, outer, groups, inner, eps);
            return;
        }

        string shape = string.Create(CultureInfo.InvariantCulture, $"{outer}x{groups}x{inner}");
        double work = (double)outer * groups * inner;
        if (!GroupSplit(outer, groups, inner, out int chunks))
        {
            LaunchLabelled(single, GridFor(groups), 1, shape, TrainingCost(single.Name).Flops * work, P(x), P(mean), P(variance), P(invStd), U(outer), U(groups), U(inner), F(eps));
            return;
        }

        int blocks = chunks * groups;
        var parts = Allocate(2 * blocks, zeroed: false);
        try
        {
            LaunchLabelled(part, GridFor(blocks), 1, shape, TrainingCost(part.Name).Flops * work, P(x), P(parts), U(outer), U(groups), U(inner), U(chunks));
            LaunchLabelled(final, GridFor(groups), 1, shape, TrainingCost(final.Name).Flops * blocks, P(parts), P(mean), P(variance), P(invStd), U(outer), U(groups),
                U(inner), U(chunks), F(eps));
        }
        finally
        {
            parts.Release();                                                     // reused in stream order
        }
    }

    public override void GroupReduceKernel(Storage a, Storage? b, Storage sumA, Storage? sumAB, int outer, int groups, int inner)
    {
        if (groups <= 0)
        {
            return;
        }

        if (!GroupsFit(outer, groups, inner) || Gcn("group_reduce_f32") is not { } single || Gcn("group_reduce_part_f32") is not { } part
            || Gcn("group_reduce_final_f32") is not { } final)
        {
            base.GroupReduceKernel(a, b, sumA, sumAB, outer, groups, inner);
            return;
        }

        bool hasB = b is not null && sumAB is not null;
        string shape = string.Create(CultureInfo.InvariantCulture, $"{outer}x{groups}x{inner}");
        double work = (double)outer * groups * inner;
        if (!GroupSplit(outer, groups, inner, out int chunks))
        {
            LaunchLabelled(single, GridFor(groups), 1, shape, TrainingCost(single.Name).Flops * work,
                P(a), hasB ? P(b!) : P(a), P(sumA), hasB ? P(sumAB!) : P(sumA), U(outer), U(groups), U(inner), U(hasB ? 1 : 0));
            return;
        }

        int blocks = chunks * groups;
        var parts = Allocate(2 * blocks, zeroed: false);
        try
        {
            LaunchLabelled(part, GridFor(blocks), 1, shape, TrainingCost(part.Name).Flops * work,
                P(a), hasB ? P(b!) : P(a), P(parts), U(outer), U(groups), U(inner), U(chunks), U(hasB ? 1 : 0));
            LaunchLabelled(final, GridFor(groups), 1, shape, TrainingCost(final.Name).Flops * blocks,
                P(parts), P(sumA), hasB ? P(sumAB!) : P(sumA), U(groups), U(chunks), U(hasB ? 1 : 0));
        }
        finally
        {
            parts.Release();                                                     // reused in stream order
        }
    }

    public override void NormApplyKernel(Storage x, Storage mean, Storage invStd, Storage y, int outer, int groups, int inner)
    {
        if (!GroupsFit(outer, groups, inner) || PointwiseKernel("norm_apply_f32", (long)outer * groups * inner) is not { } f)
        {
            base.NormApplyKernel(x, mean, invStd, y, outer, groups, inner);
            return;
        }

        int n = outer * groups * inner;
        LaunchPointwise(f, n, P(x), P(mean), P(invStd), P(y), U(groups), U(inner), U(n));
    }

    public override void GroupScaleShiftKernel(Storage x, Storage? scale, Storage? shift, Storage y, int n, int groups, int inner, bool accumulate)
    {
        if (groups <= 0 || inner <= 0 || PointwiseKernel("group_scale_shift_f32", n) is not { } f)
        {
            base.GroupScaleShiftKernel(x, scale, shift, y, n, groups, inner, accumulate);
            return;
        }

        int flags = (scale is null ? 0 : 1) | (shift is null ? 0 : 2) | (accumulate ? 4 : 0);
        LaunchPointwise(f, n, P(x), P(scale ?? x), P(shift ?? x), P(y), U(groups), U(inner), U(flags), U(n));
    }

    public override void WeightNormKernel(Storage gain, Storage direction, Storage weight, int rows, int columns)
    {
        if (RowKernel("weight_norm_f32", columns) is not { } f)
        {
            base.WeightNormKernel(gain, direction, weight, rows, columns);
            return;
        }

        LaunchRows(f, rows, columns, TrainingCost(f.Name).Flops, P(gain), P(direction), P(weight), U(columns), U(rows));
    }

    public override void WeightNormBackwardKernel(Storage gain, Storage direction, Storage dweight, Storage? dgain, Storage? ddirection, int rows, int columns)
    {
        int flags = (dgain is null ? 0 : 1) | (ddirection is null ? 0 : 2);
        if (flags == 0)
        {
            return;
        }

        if (RowKernel("weight_norm_bwd_f32", columns) is not { } f)
        {
            base.WeightNormBackwardKernel(gain, direction, dweight, dgain, ddirection, rows, columns);
            return;
        }

        LaunchRows(f, rows, columns, TrainingCost(f.Name).Flops, P(gain), P(direction), P(dweight), P(dgain ?? gain), P(ddirection ?? direction), U(columns), U(flags), U(rows));
    }
}
