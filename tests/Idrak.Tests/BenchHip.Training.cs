// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Gpu.Hip;

// --bench-hip training (plan 35, G5b): HIP's training operations on their GCN kernels, each timed cold (its first call)
// apart from warm (BenchOperation), with GB/s and GFLOP/s from the table of GcnKernels.Training.cs (HipBackend.Cost), the
// section ending with the host fallbacks it took (none expected). Sizes follow the large buffer (from the memory the device
// reports), never a card's: element-wise passes over a quarter of it, rows of 4,096 columns, a prompt of up to 512 rows
// for the attention gradients.
internal abstract partial class DesktopTests
{
    private static void BenchHipTraining(HipBench b)
    {
        if (!BenchKernelSet(b))
        {
            return;
        }

        var hip = b.Hip;
        long before = Idrak.Abstraction.Operations.Kernels.HostCalls(hip);
        int n = (int)Math.Clamp(b.LargeElements / 4, 1024, (1L << 29) - 1);
        const int Cols = 4096;
        int rows = Math.Max(1, n / Cols);
        string count = n.ToString("N0", CultureInfo.InvariantCulture);
        using (var on = new BenchStorages(hip, new Random(9)))
        {
            Storage x = on.Random(n), y = on.Random(n, 0.1f, 1f), dy = on.Random(n), dx = on.Random(n), g = on.Random(n), m = on.Random(n), v = on.Random(n, 0f, 0.01f);
            foreach (var op in new[] { UnaryOp.Relu, UnaryOp.Sigmoid, UnaryOp.Gelu, UnaryOp.Silu })
            {
                string kernel = HipBackend.UnaryBackwardKernelName(op)!;
                BenchOperation(b, $"{op} gradient, {count} elements", n, HipBackend.Cost(kernel), () => hip.UnaryBackward(op, x, y, dy, dx, n));
            }

            BenchOperation(b, $"gated activation gradient (SiLU), {count} elements", n, HipBackend.Cost("gated_act_bwd_f32"),
                () => hip.GatedActivationBackward(x, y, dy, dx, m, n, 0, 3));
            BenchOperation(b, $"maximum gradient, {count} elements", n, HipBackend.Cost("extremum_bwd_f32"), () => hip.ExtremumBackward(BinaryOp.Maximum, x, y, dy, dx, m, n));
            BenchOperation(b, $"softmax gradient, {rows} x {Cols}", (long)rows * Cols, HipBackend.Cost("softmax_bwd_f32"),
                () => hip.SoftmaxBackward(y, dy, dx, rows, Cols, false));
            BenchOperation(b, $"RMS norm gradient, {rows} x {Cols}", (long)rows * Cols, HipBackend.Cost("rms_norm_backward_f32"),
                () => hip.RmsNormBackward(dy, y, m, dx, rows, Cols));
            BenchOperation(b, $"SGD with momentum, {count} elements", n, HipBackend.Cost("sgd_momentum_f32"), () => hip.SgdStep(x, g, m, n, 1e-3f, 0.9f));
            BenchOperation(b, $"Adam, {count} elements", n, HipBackend.Cost("adam_f32"), () => hip.AdamStep(x, g, m, v, n, 1e-4f, 0.9f, 0.999f, 1e-8f));
            BenchOperation(b, $"dropout, {count} elements", n, HipBackend.Cost("dropout_f32"), () => hip.Dropout(x, dx, n, 0.1f, 7u));
            BenchOperation(b, $"sum, {count} elements", n, HipBackend.Cost("sum_f32"), () => hip.Sum(x, dx, n, 1f));
            BenchOperation(b, $"sum of squares, {count} elements", n, HipBackend.Cost("sumsq_f32"), () => hip.SumSquares(x, dx, n));
            BenchOperation(b, $"sums of {rows} rows of {Cols} (chunks folded)", (long)rows * Cols, HipBackend.Cost("sum_cols_strided_f32"), () => hip.SumRows(x, dx, rows, Cols));
            IDisposable? fused = null;
            var tensors = new (Storage, Storage, Storage, Storage, int)[] { (x, g, m, v, n) };
            BenchOperation(b, $"fused AdamW with clipping, {count} elements", n, HipBackend.Cost("multi_adamw_f32"),
                () => hip.FusedAdamW(tensors, ref fused, 1f, 1e-4f, 0.99f, 0.9f, 0.999f, 1e-8f, false));
            fused?.Dispose();

            // The attention gradients of a causal prompt (tiled over a cache), head 128.
            const int Heads = 8, Dim = 128;
            int prompt = (int)Math.Clamp(b.LargeElements / (16L * Heads * Dim), 64, 512);
            Storage q = on.Random(Heads * prompt * Dim), keys = on.Random(Heads * prompt * Dim), values = on.Random(Heads * prompt * Dim), output = on.Random(Heads * prompt * Dim);
            Storage lse = on.Random(Heads * prompt), dout = on.Random(Heads * prompt * Dim), dq = on.Random(Heads * prompt * Dim), dk = on.Random(Heads * prompt * Dim);
            Storage dv = on.Random(Heads * prompt * Dim), start = on.Of([0f]);
            hip.AttentionTiled(q, keys, values, start, output, lse, Heads, prompt, prompt, prompt, Dim, 0.088f);
            long causal = (long)Heads * prompt * (prompt + 1) / 2 * Dim;
            BenchOperation(b, string.Create(CultureInfo.InvariantCulture, $"tiled attention gradients, {Heads} heads x {prompt} rows (causal), head {Dim}"), causal,
                HipBackend.Cost("attn_bwd_kv_f32"), () => hip.AttentionTiledBackward(q, keys, values, output, lse, dout, dq, dk, dv, Heads, prompt, prompt, prompt, Dim, 0.088f));
        }

        long host = Idrak.Abstraction.Operations.Kernels.HostCalls(hip) - before;
        b.Row("  (host fallbacks during the section)", host == 0 ? "none: every operation ran on its kernel" : $"{host} (SHOULD BE NONE)");
    }
}
