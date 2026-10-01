// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Collections.Concurrent;

namespace Idrak.Backends.Vulkan;

// The operations that run as generated SPIR-V kernels (VulkanKernels), each passing its storages and push constants in
// the order its kernel declares them. A case a kernel does not cover (a storage larger than the device binds, a head
// size beyond the attention kernels, more workgroups than the device takes, a permutation of more than six dimensions)
// goes to the host fallback, as every operation without a kernel does.
internal sealed partial class VulkanBackend
{
    // One VulkanKernel per generated kernel for the process: pipelines are cached per backend by kernel identity.
    private static readonly ConcurrentDictionary<string, VulkanKernel> Kernels = new();

    // Groups of an element-wise (grid-stride) dispatch over n elements.
    private const uint MaxGridGroups = 65535;

    private static VulkanKernel Kernel(string name) => Kernels.GetOrAdd(name, static n =>
    {
        var k = VulkanKernels.Get(n);
        return new VulkanKernel(k.Words, k.Bindings, k.PushBytes, k.Name, k.Writes);
    });

    // IDRAK_VULKAN_KERNELS=0 runs every operation through the host fallback (to compare or to isolate a kernel).
    private static readonly bool KernelsOff = Environment.GetEnvironmentVariable("IDRAK_VULKAN_KERNELS") is "0" or "false";

    /// <summary>Whether the kernels are on and every storage fits in one binding (maxStorageBufferRange).</summary>
    private bool Fit(params ReadOnlySpan<Storage> storages)
    {
        if (KernelsOff)
        {
            return false;
        }

        foreach (var s in storages)
        {
            if (BlockBytes(s.Length) > MaxStorageBytes)
            {
                return false;
            }
        }

        return true;
    }

    private static uint GridGroups(long n) => (uint)Math.Clamp((n + VulkanKernels.Block - 1) / VulkanKernels.Block, 1, MaxGridGroups);

    private static uint RowGroups(long rows) => (uint)Math.Clamp(rows, 1, MaxGridGroups);

    // Push constants as 4-byte words, on the stack.
    private ref struct Push
    {
        private readonly Span<byte> _bytes;
        private int _length;

        public Push(Span<byte> bytes) => _bytes = bytes;

        public ReadOnlySpan<byte> Bytes => _bytes[.._length];

        public Push I(int value)
        {
            BinaryPrimitives.WriteInt32LittleEndian(_bytes[_length..], value);
            _length += 4;
            return this;
        }

        public Push U(uint value)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(_bytes[_length..], value);
            _length += 4;
            return this;
        }

        public Push F(float value)
        {
            BinaryPrimitives.WriteSingleLittleEndian(_bytes[_length..], value);
            _length += 4;
            return this;
        }

        public Push B(bool value) => I(value ? 1 : 0);
    }

    // Kernel names by operation, and the narrow and 64-wide variants by base name: looked up without building strings.
    private static readonly string[] UnaryNames = [.. Enum.GetValues<UnaryOp>().Select(op => "unary_" + op.ToString().ToLowerInvariant())];
    private static readonly string[] UnaryBackwardNames = [.. Enum.GetValues<UnaryOp>().Select(op => "unary_backward_" + op.ToString().ToLowerInvariant())];
    private static readonly string[] BinaryNames = [.. Enum.GetValues<BinaryOp>().Select(op => "binary_" + op.ToString().ToLowerInvariant())];
    private static readonly ConcurrentDictionary<string, VulkanKernel> NarrowKernels = new(), SmallKernels = new();

    // Dispatches a generated kernel by name.
    private void Run(string kernel, uint groupsX, uint groupsY, uint groupsZ, ReadOnlySpan<Storage> storages, ReadOnlySpan<byte> push) =>
        Dispatch(Kernel(kernel), groupsX, groupsY, groupsZ, storages, push);

    // Dispatches an element-wise kernel over n elements.
    private void Grid(string kernel, long n, ReadOnlySpan<Storage> storages, ReadOnlySpan<byte> push)
    {
        if (n > 0)
        {
            Run(kernel, GridGroups(n), 1, 1, storages, push);
        }
    }

    // Dispatches a row kernel: one workgroup per row, or for rows of at most NarrowRowColumns, its "_narrow" variant
    // (one invocation per row).
    private void Rows(string kernel, int rows, int cols, ReadOnlySpan<Storage> storages, ReadOnlySpan<byte> push)
    {
        if (rows <= 0)
        {
            return;
        }

        if (cols <= VulkanKernels.NarrowRowColumns)
        {
            Dispatch(NarrowKernels.GetOrAdd(kernel, static n => Kernel(n + "_narrow")), GridGroups(rows), 1, 1, storages, push);
        }
        else
        {
            Run(kernel, RowGroups(rows), 1, 1, storages, push);
        }
    }

    // Dispatches an attention kernel: one workgroup per query row, 64 wide for head sizes up to 64.
    private void Attend(string kernel, int rows, int dim, ReadOnlySpan<Storage> storages, ReadOnlySpan<byte> push)
    {
        if (rows > 0)
        {
            var chosen = dim <= VulkanKernels.SmallAttentionLanes ? SmallKernels.GetOrAdd(kernel, static n => Kernel(n + "_64")) : Kernel(kernel);
            Dispatch(chosen, RowGroups(rows), 1, 1, storages, push);
        }
    }

    // ------------------------------------------------------------------ element-wise

    public override void Fill(Storage y, int n, float value)
    {
        if (!Fit(y))
        {
            base.Fill(y, n, value);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Grid("fill", n, [y], new Push(b).I(n).F(value).Bytes);
    }

    public override void Copy(Storage x, Storage y, int n)
    {
        if (!Fit(x, y))
        {
            base.Copy(x, y, n);
            return;
        }

        Span<byte> b = stackalloc byte[4];
        Grid("copy", n, [x, y], new Push(b).I(n).Bytes);
    }

    public override void Unary(UnaryOp op, Storage x, Storage y, int n)
    {
        if (!Fit(x, y))
        {
            base.Unary(op, x, y, n);
            return;
        }

        Span<byte> b = stackalloc byte[4];
        Grid(UnaryNames[(int)op], n, [x, y], new Push(b).I(n).Bytes);
    }

    public override void UnaryBackward(UnaryOp op, Storage x, Storage y, Storage dy, Storage dx, int n)
    {
        if (!Fit(x, y, dy, dx))
        {
            base.UnaryBackward(op, x, y, dy, dx, n);
            return;
        }

        Span<byte> b = stackalloc byte[4];
        Grid(UnaryBackwardNames[(int)op], n, [x, y, dy, dx], new Push(b).I(n).Bytes);
    }

    public override void Binary(BinaryOp op, Storage a, Storage b, Storage c, int n)
    {
        if (!Fit(a, b, c))
        {
            base.Binary(op, a, b, c, n);
            return;
        }

        Span<byte> p = stackalloc byte[4];
        Grid(BinaryNames[(int)op], n, [a, b, c], new Push(p).I(n).Bytes);
    }

    public override void Affine(Storage x, Storage y, int n, float alpha, float beta)
    {
        if (!Fit(x, y))
        {
            base.Affine(x, y, n, alpha, beta);
            return;
        }

        Span<byte> b = stackalloc byte[12];
        Grid("affine", n, [x, y], new Push(b).I(n).F(alpha).F(beta).Bytes);
    }

    public override void Axpy(Storage x, Storage y, int n, float alpha)
    {
        if (!Fit(x, y))
        {
            base.Axpy(x, y, n, alpha);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Grid("axpy", n, [x, y], new Push(b).I(n).F(alpha).Bytes);
    }

    public override void MulAdd(Storage a, Storage b, Storage c, int n)
    {
        if (!Fit(a, b, c))
        {
            base.MulAdd(a, b, c, n);
            return;
        }

        Span<byte> p = stackalloc byte[4];
        Grid("mul_add", n, [a, b, c], new Push(p).I(n).Bytes);
    }

    public override void AddRowVector(Storage a, Storage v, Storage c, int rows, int cols)
    {
        if (!Fit(a, v, c))
        {
            base.AddRowVector(a, v, c, rows, cols);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Grid("add_row_vector", (long)rows * cols, [a, v, c], new Push(b).I(rows * cols).I(cols).Bytes);
    }

    public override void SumRows(Storage x, Storage y, int rows, int cols)
    {
        if (!Fit(x, y))
        {
            base.SumRows(x, y, rows, cols);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Grid("sum_rows", cols, [x, y], new Push(b).I(rows).I(cols).Bytes);
    }

    public override void Sum(Storage x, Storage result, int n, float scale)
    {
        if (!Fit(x, result))
        {
            base.Sum(x, result, n, scale);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Run("sum", 1, 1, 1, [x, result], new Push(b).I(n).F(scale).Bytes);
    }

    public override void SumSquares(Storage x, Storage total, int n)
    {
        if (!Fit(x, total))
        {
            base.SumSquares(x, total, n);
            return;
        }

        Span<byte> b = stackalloc byte[4];
        Run("sum_squares", 1, 1, 1, [x, total], new Push(b).I(n).Bytes);
    }

    public override void AxpyAt(Storage x, Storage y, int offset, float alpha)
    {
        if (!Fit(x, y))
        {
            base.AxpyAt(x, y, offset, alpha);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Run("axpy_at", 1, 1, 1, [x, y], new Push(b).I(offset).F(alpha).Bytes);
    }

    public override void AddBroadcastScalar(Storage s, Storage y, int n, float scale)
    {
        if (!Fit(s, y))
        {
            base.AddBroadcastScalar(s, y, n, scale);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Grid("add_broadcast_scalar", n, [s, y], new Push(b).I(n).F(scale).Bytes);
    }

    public override void InvSqrt(Storage x, Storage y, int n, float eps)
    {
        if (!Fit(x, y))
        {
            base.InvSqrt(x, y, n, eps);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Grid("inv_sqrt", n, [x, y], new Push(b).I(n).F(eps).Bytes);
    }

    public override void BiasGelu(Storage x, Storage bias, Storage y, int n, int cols)
    {
        if (!Fit(x, bias, y))
        {
            base.BiasGelu(x, bias, y, n, cols);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Grid("bias_gelu", n, [x, bias, y], new Push(b).I(n).I(cols).Bytes);
    }

    public override void GatedActivation(Storage gate, Storage up, Storage y, int n, int kind)
    {
        if (!Fit(gate, up, y))
        {
            base.GatedActivation(gate, up, y, n, kind);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Grid("gated_activation", n, [gate, up, y], new Push(b).I(n).I(kind).Bytes);
    }

    public override void GatedActivationBackward(Storage gate, Storage up, Storage dy, Storage dgate, Storage dup, int n, int kind, int flags)
    {
        if (!Fit(gate, up, dy, dgate, dup))
        {
            base.GatedActivationBackward(gate, up, dy, dgate, dup, n, kind, flags);
            return;
        }

        Span<byte> b = stackalloc byte[12];
        Grid("gated_activation_backward", n, [gate, up, dy, dgate, dup], new Push(b).I(n).I(kind).I(flags).Bytes);
    }

    public override void SgdStep(Storage p, Storage g, Storage? v, int n, float lr, float momentum)
    {
        if (v is null)
        {
            Axpy(g, p, n, -lr);
            return;
        }

        if (!Fit(p, g, v))
        {
            base.SgdStep(p, g, v, n, lr, momentum);
            return;
        }

        Span<byte> b = stackalloc byte[12];
        Grid("sgd_momentum", n, [p, g, v], new Push(b).I(n).F(lr).F(momentum).Bytes);
    }

    public override void AdamStep(Storage p, Storage g, Storage m, Storage v, int n, float lr, float beta1, float beta2, float eps)
    {
        if (!Fit(p, g, m, v))
        {
            base.AdamStep(p, g, m, v, n, lr, beta1, beta2, eps);
            return;
        }

        Span<byte> b = stackalloc byte[20];
        Grid("adam", n, [p, g, m, v], new Push(b).I(n).F(lr).F(beta1).F(beta2).F(eps).Bytes);
    }

    public override void Dropout(Storage x, Storage y, int n, float p, uint seed)
    {
        if (!Fit(x, y))
        {
            base.Dropout(x, y, n, p, seed);
            return;
        }

        Span<byte> b = stackalloc byte[16];
        Grid("dropout", n, [x, y], new Push(b).I(n).F(p).F(1f / (1f - p)).U(seed).Bytes);
    }

    public override void DropoutBackward(Storage dy, Storage dx, int n, float p, uint seed)
    {
        if (!Fit(dy, dx))
        {
            base.DropoutBackward(dy, dx, n, p, seed);
            return;
        }

        Span<byte> b = stackalloc byte[16];
        Grid("dropout_backward", n, [dy, dx], new Push(b).I(n).F(p).F(1f / (1f - p)).U(seed).Bytes);
    }

    public override void ClipFactor(Storage sumSquares, Storage factor, float maxNorm)
    {
        if (!Fit(sumSquares, factor))
        {
            base.ClipFactor(sumSquares, factor, maxNorm);
            return;
        }

        Span<byte> b = stackalloc byte[4];
        Run("clip_factor", 1, 1, 1, [sumSquares, factor], new Push(b).F(maxNorm).Bytes);
    }

    // ------------------------------------------------------------------ rows

    public override void Softmax(Storage x, Storage y, int rows, int cols, bool log)
    {
        if (!Fit(x, y))
        {
            base.Softmax(x, y, rows, cols, log);
            return;
        }

        Span<byte> b = stackalloc byte[12];
        Rows("softmax", rows, cols, [x, y], new Push(b).I(rows).I(cols).B(log).Bytes);
    }

    public override void SoftmaxBackward(Storage y, Storage dy, Storage dx, int rows, int cols, bool log)
    {
        if (!Fit(y, dy, dx))
        {
            base.SoftmaxBackward(y, dy, dx, rows, cols, log);
            return;
        }

        Span<byte> b = stackalloc byte[12];
        Rows("softmax_backward", rows, cols, [y, dy, dx], new Push(b).I(rows).I(cols).B(log).Bytes);
    }

    public override void ScaleMaskSoftmax(Storage x, Storage? mask, Storage y, int rows, int cols, int maskRows, float scale)
    {
        if (!Fit(x, mask ?? x, y))
        {
            base.ScaleMaskSoftmax(x, mask, y, rows, cols, maskRows, scale);
            return;
        }

        Span<byte> b = stackalloc byte[20];
        Rows("scale_mask_softmax", rows, cols, [x, mask ?? x, y], new Push(b).I(rows).I(cols).I(maskRows).F(scale).B(mask is not null).Bytes);
    }

    public override void RmsNorm(Storage x, Storage y, Storage inv, int rows, int cols, float eps)
    {
        if (!Fit(x, y, inv))
        {
            base.RmsNorm(x, y, inv, rows, cols, eps);
            return;
        }

        Span<byte> b = stackalloc byte[12];
        Rows("rms_norm", rows, cols, [x, y, inv], new Push(b).I(rows).I(cols).F(eps).Bytes);
    }

    public override void RmsNormBackward(Storage dy, Storage y, Storage inv, Storage dx, int rows, int cols)
    {
        if (!Fit(dy, y, inv, dx))
        {
            base.RmsNormBackward(dy, y, inv, dx, rows, cols);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Rows("rms_norm_backward", rows, cols, [dy, y, inv, dx], new Push(b).I(rows).I(cols).Bytes);
    }

    public override void RmsNormAffine(Storage x, Storage gain, Storage y, int rows, int cols, float eps, float offset)
    {
        if (!Fit(x, gain, y))
        {
            base.RmsNormAffine(x, gain, y, rows, cols, eps, offset);
            return;
        }

        Span<byte> b = stackalloc byte[16];
        Rows("rms_norm_affine", rows, cols, [x, gain, y], new Push(b).I(rows).I(cols).F(eps).F(offset).Bytes);
    }

    public override void AddRmsNormAffine(Storage a, Storage b, Storage sum, Storage gain, Storage y, int rows, int cols, float eps, float offset)
    {
        if (!Fit(a, b, sum, gain, y))
        {
            base.AddRmsNormAffine(a, b, sum, gain, y, rows, cols, eps, offset);
            return;
        }

        Span<byte> p = stackalloc byte[16];
        Rows("add_rms_norm_affine", rows, cols, [a, b, sum, gain, y], new Push(p).I(rows).I(cols).F(eps).F(offset).Bytes);
    }

    public override void LayerNormFused(Storage x, Storage gamma, Storage beta, Storage y, int rows, int cols, float eps)
    {
        if (!Fit(x, gamma, beta, y))
        {
            base.LayerNormFused(x, gamma, beta, y, rows, cols, eps);
            return;
        }

        Span<byte> b = stackalloc byte[12];
        Rows("layer_norm", rows, cols, [x, gamma, beta, y], new Push(b).I(rows).I(cols).F(eps).Bytes);
    }

    public override void LayerNormTrain(Storage x, Storage gamma, Storage beta, Storage y, Storage stats, int rows, int cols, float eps)
    {
        if (!Fit(x, gamma, beta, y, stats))
        {
            base.LayerNormTrain(x, gamma, beta, y, stats, rows, cols, eps);
            return;
        }

        Span<byte> b = stackalloc byte[12];
        Rows("layer_norm_train", rows, cols, [x, gamma, beta, y, stats], new Push(b).I(rows).I(cols).F(eps).Bytes);
    }

    public override void ArgMax(Storage x, Storage y, int rows, int cols)
    {
        if (!Fit(x, y))
        {
            base.ArgMax(x, y, rows, cols);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Rows("arg_max", rows, cols, [x, y], new Push(b).I(rows).I(cols).Bytes);
    }

    public override void SoftmaxCrossEntropyRows(Storage logits, Storage targets, Storage weights, Storage losses, int rows, int vocabulary, float scale)
    {
        if (!Fit(logits, targets, weights, losses))
        {
            base.SoftmaxCrossEntropyRows(logits, targets, weights, losses, rows, vocabulary, scale);
            return;
        }

        Span<byte> b = stackalloc byte[12];
        Rows("softmax_cross_entropy_rows", rows, vocabulary, [logits, targets, weights, losses], new Push(b).I(rows).I(vocabulary).F(scale).Bytes);
    }

    // ------------------------------------------------------------------ shapes

    public override void Copy2D(Storage src, int srcOffset, int srcStride, Storage dst, int dstOffset, int dstStride, int rows, int cols, bool accumulate)
    {
        if (!Fit(src, dst))
        {
            base.Copy2D(src, srcOffset, srcStride, dst, dstOffset, dstStride, rows, cols, accumulate);
            return;
        }

        Span<byte> b = stackalloc byte[28];
        Grid("copy_2d", (long)rows * cols, [src, dst],
            new Push(b).I(srcOffset).I(srcStride).I(dstOffset).I(dstStride).I(rows).I(cols).B(accumulate).Bytes);
    }

    public override void Permute(Storage x, Storage y, ReadOnlySpan<int> outShape, ReadOnlySpan<int> inStrides, bool accumulate)
    {
        const int Rank = VulkanKernels.PermuteRank;
        if (outShape.Length > Rank || !Fit(x, y))
        {
            base.Permute(x, y, outShape, inStrides, accumulate);
            return;
        }

        // Padded in front with size 1, stride 0.
        int pad = Rank - outShape.Length;
        long total = 1;
        Span<byte> b = stackalloc byte[4 * (2 * Rank + 1)];
        var push = new Push(b);
        for (int d = 0; d < Rank; d++)
        {
            int size = d < pad ? 1 : outShape[d - pad];
            total *= size;
            push = push.I(size);
        }

        for (int d = 0; d < Rank; d++)
        {
            push = push.I(d < pad ? 0 : inStrides[d - pad]);
        }

        Grid("permute", total, [x, y], push.B(accumulate).Bytes);
    }

    public override void SumAxis(Storage x, Storage y, int outer, int dim, int inner, float scale, bool accumulate)
    {
        if (!Fit(x, y))
        {
            base.SumAxis(x, y, outer, dim, inner, scale, accumulate);
            return;
        }

        Span<byte> b = stackalloc byte[20];
        Grid("sum_axis", (long)outer * inner, [x, y], new Push(b).I(outer).I(dim).I(inner).F(scale).B(accumulate).Bytes);
    }

    public override void BroadcastAxis(Storage dy, Storage dx, int outer, int dim, int inner, float scale)
    {
        if (!Fit(dy, dx))
        {
            base.BroadcastAxis(dy, dx, outer, dim, inner, scale);
            return;
        }

        Span<byte> b = stackalloc byte[16];
        Grid("broadcast_axis", (long)outer * dim * inner, [dy, dx], new Push(b).I(outer).I(dim).I(inner).F(scale).Bytes);
    }

    public override void OneHot(Storage indices, Storage y, int count, int classes)
    {
        if (!Fit(indices, y))
        {
            base.OneHot(indices, y, count, classes);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Grid("one_hot", (long)count * classes, [indices, y], new Push(b).I(count).I(classes).Bytes);
    }

    public override void ScatterAdd(Storage dy, Storage indices, Storage dtable, int count, int dim, int vocabulary)
    {
        if (!Fit(dy, indices, dtable))
        {
            base.ScatterAdd(dy, indices, dtable, count, dim, vocabulary);
            return;
        }

        Span<byte> b = stackalloc byte[12];
        Grid("scatter_add", dim, [dy, indices, dtable], new Push(b).I(count).I(dim).I(vocabulary).Bytes);
    }

    // ------------------------------------------------------------------ products

    // Products with a side shorter than this take the one-invocation-per-output kernel instead of the tiled one.
    private const int SmallProduct = 32;

    public override void BatchedMatMul(Storage a, Storage b, Storage c, int batch, int m, int n, int k, bool transA, bool transB, float beta)
    {
        const int T = VulkanKernels.Tile;
        uint gx = (uint)((n + T - 1) / T), gy = (uint)((m + T - 1) / T);
        var p = _physical.Properties;
        if (!Fit(a, b, c) || gx > p.MaxComputeWorkGroupCountX || gy > p.MaxComputeWorkGroupCountY)
        {
            base.BatchedMatMul(a, b, c, batch, m, n, k, transA, transB, beta);
            return;
        }

        if (batch <= 0 || m <= 0 || n <= 0)
        {
            return;
        }

        Span<byte> bytes = stackalloc byte[28];
        var push = new Push(bytes).I(batch).I(m).I(n).I(k).B(transA).B(transB).F(beta).Bytes;
        if (m < SmallProduct || n < SmallProduct || k < SmallProduct)
        {
            Grid("batched_matmul_small", (long)batch * m * n, [a, b, c], push);              // TODO(tuning): measure the cut-over per device
            return;
        }

        uint gz = (uint)Math.Min(batch, Math.Min(MaxGridGroups, p.MaxComputeWorkGroupCountZ));
        Run("batched_matmul", gx, gy, gz, [a, b, c], push);
    }

    public override void Int8MatMul(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k)
    {
        if (!Fit(x, q, scales, y))
        {
            base.Int8MatMul(x, q, scales, y, m, n, k);
            return;
        }

        Span<byte> b = stackalloc byte[12];
        Grid("int8_matmul", (long)m * n, [x, q, scales, y], new Push(b).I(m).I(n).I(k).Bytes);
    }

    public override void Int4MatMul(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k)
    {
        if (!Fit(x, q, scales, y))
        {
            base.Int4MatMul(x, q, scales, y, m, n, k);
            return;
        }

        Span<byte> b = stackalloc byte[12];
        Grid("int4_matmul", (long)m * n, [x, q, scales, y], new Push(b).I(m).I(n).I(k).Bytes);
    }

    public override void BFloat16MatMul(Storage x, Storage packed, Storage y, int m, int n, int k)
    {
        if (!Fit(x, packed, y))
        {
            base.BFloat16MatMul(x, packed, y, m, n, k);
            return;
        }

        Span<byte> b = stackalloc byte[12];
        Grid("bf16_matmul", (long)m * n, [x, packed, y], new Push(b).I(m).I(n).I(k).Bytes);
    }

    public override void Int8Dequantize(Storage q, Storage scales, Storage w, int k, int n)
    {
        if (!Fit(q, scales, w))
        {
            base.Int8Dequantize(q, scales, w, k, n);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Grid("int8_dequantize", (long)k * n, [q, scales, w], new Push(b).I(k).I(n).Bytes);
    }

    public override void Int4Dequantize(Storage q, Storage scales, Storage w, int k, int n)
    {
        if (!Fit(q, scales, w))
        {
            base.Int4Dequantize(q, scales, w, k, n);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Grid("int4_dequantize", (long)k * n, [q, scales, w], new Push(b).I(k).I(n).Bytes);
    }

    public override void BFloat16Dequantize(Storage packed, Storage w, int k, int n)
    {
        if (!Fit(packed, w))
        {
            base.BFloat16Dequantize(packed, w, k, n);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Grid("bf16_dequantize", (long)k * n, [packed, w], new Push(b).I(k).I(n).Bytes);
    }

    public override void PackBFloat16(Storage x, Storage packed, int n)
    {
        if (!Fit(x, packed))
        {
            base.PackBFloat16(x, packed, n);
            return;
        }

        Span<byte> b = stackalloc byte[4];
        Grid("bf16_pack", (n + 1L) / 2, [x, packed], new Push(b).I(n).Bytes);
    }

    // ------------------------------------------------------------------ language models

    public override void Rope(Storage x, Storage y, Storage cos, Storage sin, Storage positions, int rows, int heads, int steps, int dim, int half, bool interleaved, float sign)
    {
        if (!Fit(x, y, cos, sin, positions))
        {
            base.Rope(x, y, cos, sin, positions, rows, heads, steps, dim, half, interleaved, sign);
            return;
        }

        Span<byte> b = stackalloc byte[28];
        Grid("rope", (long)rows * half, [x, y, cos, sin, positions],
            new Push(b).I(rows).I(heads).I(steps).I(dim).I(half).B(interleaved).F(sign).Bytes);
    }

    // Indices outside the table are clamped into it (the CPU throws; CUDA clamps too).
    public override void Gather(Storage table, Storage indices, Storage y, int count, int dim, int vocabulary)
    {
        if (!Fit(table, indices, y))
        {
            base.Gather(table, indices, y, count, dim, vocabulary);
            return;
        }

        Span<byte> b = stackalloc byte[12];
        Grid("gather", (long)count * dim, [table, indices, y], new Push(b).I(count).I(dim).I(vocabulary).Bytes);
    }

    public override void GatherBFloat16(Storage packed, Storage indices, Storage y, int count, int dim, int vocabulary)
    {
        if (!Fit(packed, indices, y))
        {
            base.GatherBFloat16(packed, indices, y, count, dim, vocabulary);
            return;
        }

        Span<byte> b = stackalloc byte[12];
        Grid("gather_bf16", (long)count * dim, [packed, indices, y], new Push(b).I(count).I(dim).I(vocabulary).Bytes);
    }

    public override void KeyValueWrite(Storage source, Storage cache, Storage position, int heads, int steps, int capacity, int dim)
    {
        if (!Fit(source, cache, position))
        {
            base.KeyValueWrite(source, cache, position, heads, steps, capacity, dim);
            return;
        }

        Span<byte> b = stackalloc byte[16];
        Grid("key_value_write", (long)heads * steps * dim, [source, cache, position], new Push(b).I(heads).I(steps).I(capacity).I(dim).Bytes);
    }

    public override void KeyValueWriteBFloat16(Storage source, Storage cache, Storage position, int heads, int steps, int capacity, int dim)
    {
        if (!Fit(source, cache, position))
        {
            base.KeyValueWriteBFloat16(source, cache, position, heads, steps, capacity, dim);
            return;
        }

        Span<byte> b = stackalloc byte[16];
        Grid("key_value_write_bf16", (long)heads * steps * ((dim + 1) / 2), [source, cache, position],
            new Push(b).I(heads).I(steps).I(capacity).I(dim).Bytes);
    }

    public override void KeyValueWriteInt8(Storage source, Storage cache, Storage scales, Storage position, int heads, int steps, int capacity, int dim)
    {
        if (!Fit(source, cache, scales, position))
        {
            base.KeyValueWriteInt8(source, cache, scales, position, heads, steps, capacity, dim);
            return;
        }

        Span<byte> b = stackalloc byte[16];
        Grid("key_value_write_int8", (long)heads * steps, [source, cache, scales, position], new Push(b).I(heads).I(steps).I(capacity).I(dim).Bytes);
    }

    public override void DecoderMask(Storage position, Storage mask, int rows, int capacity)
    {
        if (!Fit(position, mask))
        {
            base.DecoderMask(position, mask, rows, capacity);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Grid("decoder_mask", (long)rows * capacity, [position, mask], new Push(b).I(rows).I(capacity).Bytes);
    }

    public override void AttentionDecode(Storage q, Storage keys, Storage values, Storage position, Storage y, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale)
    {
        if (dim > VulkanKernels.AttentionMaxDim || !Fit(q, keys, values, position, y))
        {
            base.AttentionDecode(q, keys, values, position, y, heads, rowsPerHead, steps, capacity, dim, scale);
            return;
        }

        Span<byte> b = stackalloc byte[24];
        Attend("attention_decode", heads * rowsPerHead, dim, [q, keys, values, position, y],
            new Push(b).I(heads).I(rowsPerHead).I(steps).I(capacity).I(dim).F(scale).Bytes);
    }

    public override void AttentionBFloat16(Storage q, Storage keys, Storage values, Storage position, Storage y, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale, bool tiled)
    {
        if (dim > VulkanKernels.AttentionMaxDim || !Fit(q, keys, values, position, y))
        {
            base.AttentionBFloat16(q, keys, values, position, y, heads, rowsPerHead, steps, capacity, dim, scale, tiled);
            return;
        }

        Span<byte> b = stackalloc byte[24];
        Attend("attention_bf16", heads * rowsPerHead, dim, [q, keys, values, position, y],
            new Push(b).I(heads).I(rowsPerHead).I(steps).I(capacity).I(dim).F(scale).Bytes);
    }

    public override void AttentionInt8(Storage q, Storage keys, Storage values, Storage keyScales, Storage valueScales, Storage position,
        Storage y, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale, bool tiled)
    {
        if (dim > VulkanKernels.AttentionMaxDim || !Fit(q, keys, values, keyScales, valueScales, position, y))
        {
            base.AttentionInt8(q, keys, values, keyScales, valueScales, position, y, heads, rowsPerHead, steps, capacity, dim, scale, tiled);
            return;
        }

        Span<byte> b = stackalloc byte[24];
        Attend("attention_int8", heads * rowsPerHead, dim, [q, keys, values, keyScales, valueScales, position, y],
            new Push(b).I(heads).I(rowsPerHead).I(steps).I(capacity).I(dim).F(scale).Bytes);
    }
}
