// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Gpu.Hip;

// HIP's element-wise operations on its own GCN kernels (GcnKernels.Pointwise.cs), where CUDA overrides them and with
// CUDA's arguments: the activations (UnaryOp), add, sub, mul, max and min (BinaryOp), affine, axpy, multiply-add and the
// row-vector add. Each kernel walks its elements grid-stride, so the grid is sized to fill the device rather than to the
// work (rule 81): blocks of the kernel's declared 256, about one per 1,024 elements, at most PointwiseBlocksPerUnit
// blocks a compute unit the device reports (IDRAK_HIP_POINTWISE_BLOCKS), within its grid and the kernels' 2^29
// work-items. A count of 2^30 or more (the kernels' 32-bit byte offsets), an operation without a kernel and unavailable
// kernels take the host fallback. Fills stay the runtime's own memset (FillKernel); fill_f32 is the gate kernel.
// Each launch is labelled for --profile with its kernel, its count and its floating-point operations (Cost).
internal sealed unsafe partial class HipBackend
{
    /// <summary>The setting for the blocks an element-wise launch may take per compute unit (a positive number; default 8).</summary>
    internal const string PointwiseBlocksVariable = "IDRAK_HIP_POINTWISE_BLOCKS";

    // Elements a block's grid-stride walk is sized for: each lane moves four floats a chunk, four waves a block.
    private const int PointwiseElementsPerBlock = 1024;

    /// <summary>
    /// Blocks per compute unit an element-wise launch may take (IDRAK_HIP_POINTWISE_BLOCKS, <paramref name="setting"/>):
    /// by default 8 blocks of four waves, 32 waves a compute unit, enough to hide memory latency; the grid-stride loop
    /// takes the rest.
    /// </summary>
    internal static int PointwiseBlocksPerUnit(string? setting) =>
        int.TryParse(setting, NumberStyles.None, CultureInfo.InvariantCulture, out int blocks) && blocks > 0 ? blocks : 8;

    private static readonly int BlocksPerUnit = PointwiseBlocksPerUnit(Environment.GetEnvironmentVariable(PointwiseBlocksVariable));

    /// <summary>The blocks an element-wise launch over <paramref name="n"/> elements takes on this device.</summary>
    internal uint PointwiseBlocks(long n)
    {
        long filling = (long)Math.Max(1, DeviceLimits.ComputeUnits) * BlocksPerUnit;
        long most = Math.Min(Math.Min(DeviceLimits.MaxGridDimX, GcnKernels.PointwiseMostWorkItems / GcnKernels.PointwiseBlock), filling);
        return (uint)Math.Clamp((n + PointwiseElementsPerBlock - 1) / PointwiseElementsPerBlock, 1, Math.Max(1, most));
    }

    // The element-wise kernel `name` for n elements, or null: the kernels unavailable, or n past what the kernels take.
    private GcnFunction? PointwiseKernel(string? name, long n) => name is not null && n < GcnKernels.PointwiseMostElements ? Gcn(name) : null;

    // Launches an element-wise kernel over n > 0 elements, labelled for the profile.
    private void LaunchPointwise(GcnFunction function, int n, params ReadOnlySpan<ulong> values)
    {
        if (n <= 0)
        {
            return;
        }

        if (Profiling)
        {
            LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"{function.Name} {n}"), Cost(function.Name).Flops * n);
        }

        LaunchKernel(function, PointwiseBlocks(n), 1, values);
    }

    /// <summary>The kernel of a unary operation, or null for one without (CUDA's set).</summary>
    internal static string? UnaryKernelName(UnaryOp op) => op switch
    {
        UnaryOp.Sigmoid => "sigmoid_f32",
        UnaryOp.Tanh => "tanh_f32",
        UnaryOp.Relu => "relu_f32",
        UnaryOp.Square => "square_f32",
        UnaryOp.Abs => "abs_f32",
        UnaryOp.Exp => "exp_f32",
        UnaryOp.Log => "log_f32",
        UnaryOp.Gelu => "gelu_f32",
        UnaryOp.Sqrt => "sqrt_f32",
        UnaryOp.Sin => "sin_f32",
        UnaryOp.Cos => "cos_f32",
        UnaryOp.Silu => "silu_f32",
        UnaryOp.Sign => "sign_f32",
        UnaryOp.GeluErf => "gelu_erf_f32",
        _ => null,
    };

    /// <summary>The kernel of a binary operation, or null for one without.</summary>
    internal static string? BinaryKernelName(BinaryOp op) => op switch
    {
        BinaryOp.Add => "add_f32",
        BinaryOp.Sub => "sub_f32",
        BinaryOp.Mul => "mul_f32",
        BinaryOp.Maximum => "max_f32",
        BinaryOp.Minimum => "min_f32",
        _ => null,
    };

    public override void UnaryKernel(UnaryOp op, Storage x, Storage y, int n)
    {
        if (PointwiseKernel(UnaryKernelName(op), n) is not { } f)
        {
            base.UnaryKernel(op, x, y, n);
            return;
        }

        LaunchPointwise(f, n, P(x), P(y), U(n));
    }

    public override void BinaryKernel(BinaryOp op, Storage a, Storage b, Storage c, int n)
    {
        if (PointwiseKernel(BinaryKernelName(op), n) is not { } f)
        {
            base.BinaryKernel(op, a, b, c, n);
            return;
        }

        LaunchPointwise(f, n, P(a), P(b), P(c), U(n));
    }

    public override void AffineKernel(Storage x, Storage y, int n, float alpha, float beta)
    {
        if (PointwiseKernel("affine_f32", n) is not { } f)
        {
            base.AffineKernel(x, y, n, alpha, beta);
            return;
        }

        LaunchPointwise(f, n, P(x), P(y), F(alpha), F(beta), U(n));
    }

    public override void AxpyKernel(Storage x, Storage y, int n, float alpha)
    {
        if (PointwiseKernel("axpy_f32", n) is not { } f)
        {
            base.AxpyKernel(x, y, n, alpha);
            return;
        }

        LaunchPointwise(f, n, P(x), P(y), F(alpha), U(n));
    }

    public override void MulAddKernel(Storage a, Storage b, Storage c, int n)
    {
        if (PointwiseKernel("muladd_f32", n) is not { } f)
        {
            base.MulAddKernel(a, b, c, n);
            return;
        }

        LaunchPointwise(f, n, P(a), P(b), P(c), U(n));
    }

    public override void AddRowVectorKernel(Storage a, Storage v, Storage c, int rows, int cols)
    {
        long n = (long)rows * cols;
        if (cols <= 0 || PointwiseKernel("add_rowvec_f32", n) is not { } f)
        {
            base.AddRowVectorKernel(a, v, c, rows, cols);
            return;
        }

        LaunchPointwise(f, (int)n, P(a), P(v), P(c), U(cols), U((int)n));
    }
}
