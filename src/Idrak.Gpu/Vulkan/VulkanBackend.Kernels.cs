// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Numerics;

namespace Idrak.Gpu.Vulkan;

// The operations that run as generated SPIR-V kernels (VulkanKernels), each passing its storages and push constants in
// the order its kernel declares them. A case a kernel does not cover (a storage larger than the device binds, except
// where an operation binds it in windows: VulkanBackend.LargeStorage.cs; a head size beyond the attention kernels, more
// workgroups than the device takes, a permutation of more than six dimensions) goes to the host fallback, as every
// operation without a kernel does.
internal sealed partial class VulkanBackend
{
    // One VulkanKernel per generated kernel and workgroup width for the process (two devices of one width share them;
    // pipelines are cached per backend by kernel identity), and this backend's by name.
    private static readonly ConcurrentDictionary<(string Name, int Width, bool Subgroups), VulkanKernel> Kernels = new();
    private readonly ConcurrentDictionary<string, VulkanKernel> _kernels = new();

    // The kernel `name` built for this device: its width, and subgroup reductions where the device reports subgroup
    // arithmetic for compute shaders.
    private VulkanKernel Kernel(string name) => _kernels.TryGetValue(name, out var known) ? known
        : _kernels.GetOrAdd(name, n => KernelAt(n, Width, Limits.SubgroupArithmetic));

    // The kernel `name` built for workgroups of `width`, with or without subgroup reductions (the tuned operations try
    // narrower widths than the device's, and both reductions).
    private static VulkanKernel KernelAt(string name, int width, bool subgroups) => Kernels.GetOrAdd((name, width, subgroups), static key =>
    {
        var k = VulkanKernels.Get(key.Name, key.Width, key.Subgroups);
        return new VulkanKernel(k.Words, k.Bindings, k.PushBytes, k.Name, k.Writes);
    });

    // Workgroup widths a tuned operation tries: the device's, half and twice it, within MinWidth (and whole subgroups),
    // MaxWidth and the device's limits.
    private int[] CandidateWidths => _candidateWidths ??= [.. new[] { Width, Width / 2, Width * 2 }
        .Where(w => w == Width || w >= Math.Max(VulkanKernels.MinWidth, Limits.SubgroupSize) && w <= VulkanKernels.MaxWidth
            && w <= Limits.MaxInvocations && w <= Limits.MaxSizeX && VulkanKernels.SharedBytesBound(w) <= Limits.SharedBytes)];

    private int[]? _candidateWidths;

    // A tuned choice carries its workgroup width as log2 in bits 24 and up, Plain (bit 23) for reductions through
    // workgroup memory alone where the device has subgroup arithmetic, and the operation's own part in bits 0 to 22.
    private static int WithWidth(int width, int rest) => (BitOperations.Log2((uint)width) << 24) | rest;

    private static int WidthOf(int choice) => 1 << ((choice >> 24) & 31);

    private const int Plain = 1 << 23, Rest = Plain - 1;

    // Whether a choice uses subgroup reductions: where the device has them, unless Plain.
    private bool SubgroupsOf(int choice) => Limits.SubgroupArithmetic && (choice & Plain) == 0;

    // The reduction variants a tuned kernel with reductions tries: subgroups (where reported) and plain.
    private int[] ReductionFlags => Limits.SubgroupArithmetic ? [0, Plain] : [0];

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

    // Groups of an element-wise (grid-stride) dispatch over n elements: one per `Width` elements, at most what the device
    // takes (the kernels loop over the rest).
    private uint GridGroups(long n) => (uint)Math.Clamp((n + Width - 1) / Width, 1, Limits.MaxGroupsX);

    // Groups of a row kernel: one per row, at most what the device takes.
    private uint RowGroups(long rows) => (uint)Math.Clamp(rows, 1, Limits.MaxGroupsX);

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

    // Kernel names by operation, and the narrow variants by base name: looked up without building strings.
    private static readonly string[] UnaryNames = [.. Enum.GetValues<UnaryOp>().Select(op => "unary_" + op.ToString().ToLowerInvariant())];
    private static readonly string[] UnaryBackwardNames = [.. Enum.GetValues<UnaryOp>().Select(op => "unary_backward_" + op.ToString().ToLowerInvariant())];
    private static readonly string[] BinaryNames = [.. Enum.GetValues<BinaryOp>().Select(op => "binary_" + op.ToString().ToLowerInvariant())];
    private readonly ConcurrentDictionary<string, string> _narrowNames = new();

    // Dispatches a generated kernel by name.
    private void Run(string kernel, uint groupsX, uint groupsY, uint groupsZ, ReadOnlySpan<Storage> storages, ReadOnlySpan<byte> push) =>
        DispatchKernel(Kernel(kernel), groupsX, groupsY, groupsZ, storages, push);

    // Dispatches a generated kernel built for workgroups of `width`, with subgroup reductions or not.
    private void RunAt(string kernel, int width, uint groupsX, uint groupsY, uint groupsZ, ReadOnlySpan<Storage> storages, ReadOnlySpan<byte> push,
        bool? subgroups = null)
    {
        bool sub = subgroups ?? Limits.SubgroupArithmetic;
        DispatchKernel(width == Width && sub == Limits.SubgroupArithmetic ? Kernel(kernel) : KernelAt(kernel, width, sub), groupsX, groupsY, groupsZ, storages, push);
    }

    // Dispatches an element-wise kernel over n elements.
    private void Grid(string kernel, long n, ReadOnlySpan<Storage> storages, ReadOnlySpan<byte> push)
    {
        if (n > 0)
        {
            Run(kernel, GridGroups(n), 1, 1, storages, push);
        }
    }

    // Dispatches a row kernel: a workgroup per row, or its "_narrow" variant (an invocation per row). Which is faster for
    // a shape is measured (VulkanTuneOp.Rows); before that, and with IDRAK_AUTOTUNE=0, rows of at most a quarter of the
    // workgroup width go narrow (a workgroup per row would leave three quarters of it idle).
    private void Rows(string kernel, int rows, int cols, ReadOnlySpan<Storage> storages, ReadOnlySpan<byte> push)
    {
        if (rows <= 0)
        {
            return;
        }

        var (narrowName, id) = _rowKernels.GetOrAdd(kernel, static n => (n + "_narrow", StableId(n)));
        int fallback = cols <= Width / 4 ? RowsNarrow : 0;
        var key = new VulkanTuneKey(VulkanTuneOp.Rows, id, rows, cols, 0);
        if (!TryTuned(key, out int choice) || (choice & Plain) != 0 && !Limits.SubgroupArithmetic)
        {
            choice = fallback;
            if (CanTune)
            {
                // Candidates: an invocation per row, a workgroup per row (with subgroup reductions where reported, and
                // through workgroup memory alone).
                var saved = storages.ToArray();
                var pushed = push.ToArray();
                choice = TuneWithScratch(key, [RowsNarrow, .. ReductionFlags], fallback, saved, VulkanKernels.Get(kernel, Width).Writes | VulkanKernels.Get(narrowName, Width).Writes,
                    (candidate, bound) => RunRows(kernel, narrowName, candidate, rows, bound, pushed));
            }
        }

        RunRows(kernel, narrowName, choice, rows, storages, push);
    }

    private const int RowsNarrow = 1;

    private void RunRows(string kernel, string narrowName, int choice, int rows, ReadOnlySpan<Storage> storages, ReadOnlySpan<byte> push)
    {
        if (choice == RowsNarrow)
        {
            Run(narrowName, GridGroups(rows), 1, 1, storages, push);
        }
        else
        {
            RunAt(kernel, Width, RowGroups(rows), 1, 1, storages, push, SubgroupsOf(choice));
        }
    }

    // Row kernels' narrow names and stable ids (for the stored choices), by name.
    private readonly ConcurrentDictionary<string, (string Narrow, int Id)> _rowKernels = new();

    // A number for a kernel name that is the same in every process (FNV-1a).
    private static int StableId(string name)
    {
        uint hash = 2166136261;
        foreach (char c in name)
        {
            hash = (hash ^ c) * 16777619;
        }

        return (int)(hash & 0x7FFFFFFF);
    }

    // Tune with the storages a candidate writes (bit i of `writes`: storage i) replaced by scratch storages of their
    // lengths, so measuring leaves the caller's storages as they were (an in-place kernel reads its input and writes the
    // scratch).
    private int TuneWithScratch(VulkanTuneKey key, ReadOnlySpan<int> candidates, int fallback, Storage[] storages, ulong writes,
        Action<int, Storage[]> run)
    {
        int[] candidateList = candidates.ToArray();
        int chosen = fallback;
        var lengths = new int[storages.Length];
        for (int i = 0; i < storages.Length; i++)
        {
            lengths[i] = (writes & (1UL << i)) != 0 ? storages[i].Length : 0;
        }

        WithScratch(lengths, scratch =>
        {
            var bound = new Storage[storages.Length];
            for (int i = 0; i < storages.Length; i++)
            {
                bound[i] = (writes & (1UL << i)) != 0 ? scratch[i] : storages[i];
            }

            chosen = Tune(key, candidateList, fallback, c => run(c, bound));
        });
        return chosen;
    }

    // ------------------------------------------------------------------ element-wise

    public override void FillKernel(Storage y, int n, float value)
    {
        if (!Fit(y))
        {
            base.FillKernel(y, n, value);
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

    public override void UnaryKernel(UnaryOp op, Storage x, Storage y, int n)
    {
        if (!Fit(x, y))
        {
            base.UnaryKernel(op, x, y, n);
            return;
        }

        Span<byte> b = stackalloc byte[4];
        Grid(UnaryNames[(int)op], n, [x, y], new Push(b).I(n).Bytes);
    }

    public override void UnaryBackwardKernel(UnaryOp op, Storage x, Storage y, Storage dy, Storage dx, int n)
    {
        if (!Fit(x, y, dy, dx))
        {
            base.UnaryBackwardKernel(op, x, y, dy, dx, n);
            return;
        }

        Span<byte> b = stackalloc byte[4];
        Grid(UnaryBackwardNames[(int)op], n, [x, y, dy, dx], new Push(b).I(n).Bytes);
    }

    public override void BinaryKernel(BinaryOp op, Storage a, Storage b, Storage c, int n)
    {
        if (!Fit(a, b, c))
        {
            base.BinaryKernel(op, a, b, c, n);
            return;
        }

        Span<byte> p = stackalloc byte[4];
        Grid(BinaryNames[(int)op], n, [a, b, c], new Push(p).I(n).Bytes);
    }

    public override void AffineKernel(Storage x, Storage y, int n, float alpha, float beta)
    {
        if (!Fit(x, y))
        {
            base.AffineKernel(x, y, n, alpha, beta);
            return;
        }

        Span<byte> b = stackalloc byte[12];
        Grid("affine", n, [x, y], new Push(b).I(n).F(alpha).F(beta).Bytes);
    }

    public override void AxpyKernel(Storage x, Storage y, int n, float alpha)
    {
        if (!Fit(x, y))
        {
            base.AxpyKernel(x, y, n, alpha);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Grid("axpy", n, [x, y], new Push(b).I(n).F(alpha).Bytes);
    }

    public override void MulAddKernel(Storage a, Storage b, Storage c, int n)
    {
        if (!Fit(a, b, c))
        {
            base.MulAddKernel(a, b, c, n);
            return;
        }

        Span<byte> p = stackalloc byte[4];
        Grid("mul_add", n, [a, b, c], new Push(p).I(n).Bytes);
    }

    public override void AddRowVectorKernel(Storage a, Storage v, Storage c, int rows, int cols)
    {
        if (!Fit(a, v, c))
        {
            base.AddRowVectorKernel(a, v, c, rows, cols);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Grid("add_row_vector", (long)rows * cols, [a, v, c], new Push(b).I(rows * cols).I(cols).Bytes);
    }

    public override void SumRowsKernel(Storage x, Storage y, int rows, int cols)
    {
        if (!Fit(x, y))
        {
            base.SumRowsKernel(x, y, rows, cols);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Grid("sum_rows", cols, [x, y], new Push(b).I(rows).I(cols).Bytes);
    }

    public override void SumKernel(Storage x, Storage result, int n, float scale)
    {
        if (!Fit(x, result))
        {
            base.SumKernel(x, result, n, scale);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Run("sum", 1, 1, 1, [x, result], new Push(b).I(n).F(scale).Bytes);
    }

    public override void SumSquaresKernel(Storage x, Storage total, int n)
    {
        if (!Fit(x, total))
        {
            base.SumSquaresKernel(x, total, n);
            return;
        }

        Span<byte> b = stackalloc byte[4];
        Run("sum_squares", 1, 1, 1, [x, total], new Push(b).I(n).Bytes);
    }

    public override void AxpyAtKernel(Storage x, Storage y, int offset, float alpha)
    {
        if (!Fit(x, y))
        {
            base.AxpyAtKernel(x, y, offset, alpha);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Run("axpy_at", 1, 1, 1, [x, y], new Push(b).I(offset).F(alpha).Bytes);
    }

    public override void AddBroadcastScalarKernel(Storage s, Storage y, int n, float scale)
    {
        if (!Fit(s, y))
        {
            base.AddBroadcastScalarKernel(s, y, n, scale);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Grid("add_broadcast_scalar", n, [s, y], new Push(b).I(n).F(scale).Bytes);
    }

    public override void InvSqrtKernel(Storage x, Storage y, int n, float eps)
    {
        if (!Fit(x, y))
        {
            base.InvSqrtKernel(x, y, n, eps);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Grid("inv_sqrt", n, [x, y], new Push(b).I(n).F(eps).Bytes);
    }

    public override void BiasGeluKernel(Storage x, Storage bias, Storage y, int n, int cols)
    {
        if (!Fit(x, bias, y))
        {
            base.BiasGeluKernel(x, bias, y, n, cols);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Grid("bias_gelu", n, [x, bias, y], new Push(b).I(n).I(cols).Bytes);
    }

    public override void GatedActivationKernel(Storage gate, Storage up, Storage y, int n, int kind)
    {
        if (!Fit(gate, up, y))
        {
            base.GatedActivationKernel(gate, up, y, n, kind);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Grid("gated_activation", n, [gate, up, y], new Push(b).I(n).I(kind).Bytes);
    }

    public override void GatedActivationBackwardKernel(Storage gate, Storage up, Storage dy, Storage dgate, Storage dup, int n, int kind, int flags)
    {
        if (!Fit(gate, up, dy, dgate, dup))
        {
            base.GatedActivationBackwardKernel(gate, up, dy, dgate, dup, n, kind, flags);
            return;
        }

        Span<byte> b = stackalloc byte[12];
        Grid("gated_activation_backward", n, [gate, up, dy, dgate, dup], new Push(b).I(n).I(kind).I(flags).Bytes);
    }

    public override void SgdStepKernel(Storage p, Storage g, Storage? v, int n, float lr, float momentum)
    {
        if (v is null)
        {
            Axpy(g, p, n, -lr);
            return;
        }

        if (!Fit(p, g, v))
        {
            base.SgdStepKernel(p, g, v, n, lr, momentum);
            return;
        }

        Span<byte> b = stackalloc byte[12];
        Grid("sgd_momentum", n, [p, g, v], new Push(b).I(n).F(lr).F(momentum).Bytes);
    }

    public override void AdamStepKernel(Storage p, Storage g, Storage m, Storage v, int n, float lr, float beta1, float beta2, float eps)
    {
        if (!Fit(p, g, m, v))
        {
            base.AdamStepKernel(p, g, m, v, n, lr, beta1, beta2, eps);
            return;
        }

        Span<byte> b = stackalloc byte[20];
        Grid("adam", n, [p, g, m, v], new Push(b).I(n).F(lr).F(beta1).F(beta2).F(eps).Bytes);
    }

    public override void DropoutKernel(Storage x, Storage y, int n, float p, uint seed)
    {
        if (!Fit(x, y))
        {
            base.DropoutKernel(x, y, n, p, seed);
            return;
        }

        Span<byte> b = stackalloc byte[16];
        Grid("dropout", n, [x, y], new Push(b).I(n).F(p).F(1f / (1f - p)).U(seed).Bytes);
    }

    public override void DropoutBackwardKernel(Storage dy, Storage dx, int n, float p, uint seed)
    {
        if (!Fit(dy, dx))
        {
            base.DropoutBackwardKernel(dy, dx, n, p, seed);
            return;
        }

        Span<byte> b = stackalloc byte[16];
        Grid("dropout_backward", n, [dy, dx], new Push(b).I(n).F(p).F(1f / (1f - p)).U(seed).Bytes);
    }

    public override void ClipFactorKernel(Storage sumSquares, Storage factor, float maxNorm)
    {
        if (!Fit(sumSquares, factor))
        {
            base.ClipFactorKernel(sumSquares, factor, maxNorm);
            return;
        }

        Span<byte> b = stackalloc byte[4];
        Run("clip_factor", 1, 1, 1, [sumSquares, factor], new Push(b).F(maxNorm).Bytes);
    }

    // ------------------------------------------------------------------ rows

    public override void SoftmaxKernel(Storage x, Storage y, int rows, int cols, bool log)
    {
        if (!Fit(x, y))
        {
            base.SoftmaxKernel(x, y, rows, cols, log);
            return;
        }

        Span<byte> b = stackalloc byte[12];
        Rows("softmax", rows, cols, [x, y], new Push(b).I(rows).I(cols).B(log).Bytes);
    }

    public override void SoftmaxBackwardKernel(Storage y, Storage dy, Storage dx, int rows, int cols, bool log)
    {
        if (!Fit(y, dy, dx))
        {
            base.SoftmaxBackwardKernel(y, dy, dx, rows, cols, log);
            return;
        }

        Span<byte> b = stackalloc byte[12];
        Rows("softmax_backward", rows, cols, [y, dy, dx], new Push(b).I(rows).I(cols).B(log).Bytes);
    }

    public override void ScaleMaskSoftmaxKernel(Storage x, Storage? mask, Storage y, int rows, int cols, int maskRows, float scale)
    {
        if (!Fit(x, mask ?? x, y))
        {
            base.ScaleMaskSoftmaxKernel(x, mask, y, rows, cols, maskRows, scale);
            return;
        }

        Span<byte> b = stackalloc byte[20];
        Rows("scale_mask_softmax", rows, cols, [x, mask ?? x, y], new Push(b).I(rows).I(cols).I(maskRows).F(scale).B(mask is not null).Bytes);
    }

    public override void RmsNormKernel(Storage x, Storage y, Storage inv, int rows, int cols, float eps)
    {
        if (!Fit(x, y, inv))
        {
            base.RmsNormKernel(x, y, inv, rows, cols, eps);
            return;
        }

        Span<byte> b = stackalloc byte[12];
        Rows("rms_norm", rows, cols, [x, y, inv], new Push(b).I(rows).I(cols).F(eps).Bytes);
    }

    public override void RmsNormBackwardKernel(Storage dy, Storage y, Storage inv, Storage dx, int rows, int cols)
    {
        if (!Fit(dy, y, inv, dx))
        {
            base.RmsNormBackwardKernel(dy, y, inv, dx, rows, cols);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Rows("rms_norm_backward", rows, cols, [dy, y, inv, dx], new Push(b).I(rows).I(cols).Bytes);
    }

    public override void RmsNormAffineKernel(Storage x, Storage gain, Storage y, int rows, int cols, float eps, float offset)
    {
        if (!Fit(x, gain, y))
        {
            base.RmsNormAffineKernel(x, gain, y, rows, cols, eps, offset);
            return;
        }

        Span<byte> b = stackalloc byte[16];
        Rows("rms_norm_affine", rows, cols, [x, gain, y], new Push(b).I(rows).I(cols).F(eps).F(offset).Bytes);
    }

    public override void AddRmsNormAffineKernel(Storage a, Storage b, Storage sum, Storage gain, Storage y, int rows, int cols, float eps, float offset)
    {
        if (!Fit(a, b, sum, gain, y))
        {
            base.AddRmsNormAffineKernel(a, b, sum, gain, y, rows, cols, eps, offset);
            return;
        }

        Span<byte> p = stackalloc byte[16];
        Rows("add_rms_norm_affine", rows, cols, [a, b, sum, gain, y], new Push(p).I(rows).I(cols).F(eps).F(offset).Bytes);
    }

    public override void LayerNormFusedKernel(Storage x, Storage gamma, Storage beta, Storage y, int rows, int cols, float eps)
    {
        if (!Fit(x, gamma, beta, y))
        {
            base.LayerNormFusedKernel(x, gamma, beta, y, rows, cols, eps);
            return;
        }

        Span<byte> b = stackalloc byte[12];
        Rows("layer_norm", rows, cols, [x, gamma, beta, y], new Push(b).I(rows).I(cols).F(eps).Bytes);
    }

    public override void LayerNormTrainKernel(Storage x, Storage gamma, Storage beta, Storage y, Storage stats, int rows, int cols, float eps)
    {
        if (!Fit(x, gamma, beta, y, stats))
        {
            base.LayerNormTrainKernel(x, gamma, beta, y, stats, rows, cols, eps);
            return;
        }

        Span<byte> b = stackalloc byte[12];
        Rows("layer_norm_train", rows, cols, [x, gamma, beta, y, stats], new Push(b).I(rows).I(cols).F(eps).Bytes);
    }

    public override void ArgMaxKernel(Storage x, Storage y, int rows, int cols)
    {
        if (!Fit(x, y))
        {
            base.ArgMaxKernel(x, y, rows, cols);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Rows("arg_max", rows, cols, [x, y], new Push(b).I(rows).I(cols).Bytes);
    }

    public override void SoftmaxCrossEntropyRowsKernel(Storage logits, Storage targets, Storage weights, Storage losses, int rows, int vocabulary, float scale)
    {
        if (!Fit(logits, targets, weights, losses))
        {
            base.SoftmaxCrossEntropyRowsKernel(logits, targets, weights, losses, rows, vocabulary, scale);
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

    public override void PermuteKernel(Storage x, Storage y, ReadOnlySpan<int> outShape, ReadOnlySpan<int> inStrides, bool accumulate)
    {
        const int Rank = VulkanKernels.PermuteRank;
        if (outShape.Length > Rank || !Fit(x, y))
        {
            base.PermuteKernel(x, y, outShape, inStrides, accumulate);
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

    public override void SumAxisKernel(Storage x, Storage y, int outer, int dim, int inner, float scale, bool accumulate)
    {
        if (!Fit(x, y))
        {
            base.SumAxisKernel(x, y, outer, dim, inner, scale, accumulate);
            return;
        }

        Span<byte> b = stackalloc byte[20];
        Grid("sum_axis", (long)outer * inner, [x, y], new Push(b).I(outer).I(dim).I(inner).F(scale).B(accumulate).Bytes);
    }

    public override void BroadcastAxisKernel(Storage dy, Storage dx, int outer, int dim, int inner, float scale)
    {
        if (!Fit(dy, dx))
        {
            base.BroadcastAxisKernel(dy, dx, outer, dim, inner, scale);
            return;
        }

        Span<byte> b = stackalloc byte[16];
        Grid("broadcast_axis", (long)outer * dim * inner, [dy, dx], new Push(b).I(outer).I(dim).I(inner).F(scale).Bytes);
    }

    public override void OneHotKernel(Storage indices, Storage y, int count, int classes)
    {
        if (!Fit(indices, y))
        {
            base.OneHotKernel(indices, y, count, classes);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Grid("one_hot", (long)count * classes, [indices, y], new Push(b).I(count).I(classes).Bytes);
    }

    public override void ScatterAddKernel(Storage dy, Storage indices, Storage dtable, int count, int dim, int vocabulary)
    {
        if (!Fit(dy, indices, dtable))
        {
            base.ScatterAddKernel(dy, indices, dtable, count, dim, vocabulary);
            return;
        }

        Span<byte> b = stackalloc byte[12];
        Grid("scatter_add", dim, [dy, indices, dtable], new Push(b).I(count).I(dim).I(vocabulary).Bytes);
    }

    // ------------------------------------------------------------------ products

    /// <summary>Tests and benchmarks only: the k splits of the packed products instead of the measured or formula's.</summary>
    internal static int? GemvSplits { get; set; }

    /// <summary>Tests and benchmarks only: the words per row of k of the packed products (a <see cref="VulkanKernels.GemvWordCounts"/> value).</summary>
    internal static int? GemvWords { get; set; }

    /// <summary>Tests and benchmarks only: the position splits of decoding attention instead of the measured or formula's.</summary>
    internal static int? AttentionSplits { get; set; }

    /// <summary>
    /// Tests and benchmarks only: the float32 product kernel (0 small, 1 tiled, 2 register-blocked, 3 cooperative matrices,
    /// 4 the reduced-precision cooperative matrices, only where MixedPrecision asks for reduced precision) when it fits.
    /// </summary>
    internal static int? MatMulKernel { get; set; }

    // Fewest rows of k per split of a packed product (a format fact, not a device one): the partial sums written and
    // added again (4 bytes per output and split) stay at most 1/16 of the int4 weights read, 1/32 of int8's, 1/64 of
    // bfloat16's.
    private const int GemvMinChunk = 128;

    private const int MatSmall = 0, MatTiled = 1, MatBlocked = 2, MatCoop = 3, MatMixed = 4;

    public override void BatchedMatMulKernel(Storage a, Storage b, Storage c, int batch, int m, int n, int k, bool transA, bool transB, float beta)
    {
        if (!Fit(a, b, c))
        {
            base.BatchedMatMulKernel(a, b, c, batch, m, n, k, transA, transB, beta);
            return;
        }

        if (batch <= 0 || m <= 0 || n <= 0)
        {
            return;
        }

        // Formula: the device's width, register-blocked when every side is at least half its block, else small (an
        // invocation per output wastes nothing on partial blocks). Measured: every candidate (MatCandidates).
        int blocked = VulkanKernels.MatPer * VulkanKernels.MatSide(Width);
        int fallback = WithWidth(Width, MatFits(blocked, m, n) && Math.Min(m, Math.Min(n, k)) >= blocked / 2 ? MatBlocked : MatSmall);
        Span<byte> bytes = stackalloc byte[28];
        var push = new Push(bytes).I(batch).I(m).I(n).I(k).B(transA).B(transB).F(beta).Bytes;
        int chosen;
        if (MatMulKernel is int forced)
        {
            chosen = MatValid(WithWidth(Width, forced), m, n) ? WithWidth(Width, forced) : fallback;
        }
        else
        {
            var key = new VulkanTuneKey(VulkanTuneOp.MatMul, (transA ? 2 : 0) + (transB ? 1 : 0), batch, m, n, k);
            if (!TryTuned(key, out chosen) || !MatValid(chosen, m, n))
            {
                chosen = fallback;
                if (CanTune)
                {
                    var pushed = push.ToArray();
                    chosen = TuneWithScratch(key, MatCandidates(m, n), fallback, [a, b, c], 1UL << 2,
                        (choice, s) => RunMatMul(choice, s[0], s[1], s[2], batch, m, n, pushed));
                }
            }

            chosen = MixedMatMulChoice(chosen, a, b, c, batch, m, n, k, transA, transB, push);
        }

        RunMatMul(chosen, a, b, c, batch, m, n, push);
    }

    // Reduced precision (MixedPrecision, VulkanBackend.Matrix.cs): the float32 choice or the single-pass cooperative-matrix
    // kernel, measured per shape under a key of its own (the float32 choice stays what float32 products measured); the
    // float32 choice while nothing is measured, in float32 mode and where the kernel cannot run.
    private int MixedMatMulChoice(int chosen, Storage a, Storage b, Storage c, int batch, int m, int n, int k, bool transA, bool transB, ReadOnlySpan<byte> push)
    {
        int mixed = WithWidth(Width, MatMixed);
        if (!MatValid(mixed, m, n))
        {
            return chosen;
        }

        var key = new VulkanTuneKey(VulkanTuneOp.MixedMatMul, (transA ? 2 : 0) + (transB ? 1 : 0), batch, m, n, k);
        if (!TryTuned(key, out int decided) || decided is not (0 or 1))
        {
            decided = 0;
            if (CanTune)
            {
                var pushed = push.ToArray();
                decided = TuneWithScratch(key, [0, 1], 0, [a, b, c], 1UL << 2,
                    (choice, s) => RunMatMul(choice == 1 ? mixed : chosen, s[0], s[1], s[2], batch, m, n, pushed));
            }
        }

        return decided == 1 ? mixed : chosen;
    }

    // At each candidate width: the small kernel (an invocation per output, any shape), the tiled one and the
    // register-blocked one where their workgroup counts fit; and the cooperative-matrix kernel (its own width) where the
    // device has cooperative matrices.
    private int[] MatCandidates(int m, int n)
    {
        var candidates = new List<int>(3 * CandidateWidths.Length + 1);
        foreach (int width in CandidateWidths)
        {
            foreach (int variant in new[] { MatSmall, MatTiled, MatBlocked })
            {
                if (MatValid(WithWidth(width, variant), m, n))
                {
                    candidates.Add(WithWidth(width, variant));
                }
            }
        }

        if (MatValid(WithWidth(Width, MatCoop), m, n))
        {
            candidates.Add(WithWidth(Width, MatCoop));
        }

        return [.. candidates];
    }

    // Whether a choice (a stored one too) can run here: a candidate width, and its blocks within the workgroup counts.
    private bool MatValid(int choice, int m, int n)
    {
        int width = WidthOf(choice), variant = choice & Rest;
        return Array.IndexOf(CandidateWidths, width) >= 0 && variant switch
        {
            MatSmall => true,
            MatTiled => MatFits(VulkanKernels.MatSide(width), m, n),
            MatBlocked => MatFits(VulkanKernels.MatPer * VulkanKernels.MatSide(width), m, n),
            MatCoop => width == Width && MatFits(VulkanKernels.CoopBlock, m, n) && CoopKernel(null) is not null,
            MatMixed => width == Width && MatFits(VulkanKernels.CoopBlock, m, n) && MixedKernel(null) is not null,
            _ => false,
        };
    }

    // Whether the product's blocks of `edge` fit the device's workgroup counts.
    private bool MatFits(int edge, int m, int n) =>
        (n + edge - 1) / edge <= Limits.MaxGroupsX && (m + edge - 1) / edge <= Limits.MaxGroupsY;

    private void RunMatMul(int choice, Storage a, Storage b, Storage c, int batch, int m, int n, ReadOnlySpan<byte> push)
    {
        int width = WidthOf(choice), variant = choice & Rest;
        if (variant == MatSmall)
        {
            long total = (long)batch * m * n;
            RunAt("batched_matmul_small", width, (uint)Math.Clamp((total + width - 1) / width, 1, Limits.MaxGroupsX), 1, 1, [a, b, c], push);
            return;
        }

        uint gz = (uint)Math.Min(batch, Limits.MaxGroupsZ);
        if (variant is MatCoop or MatMixed)
        {
            const int Block = VulkanKernels.CoopBlock;
            var kernel = variant == MatCoop ? CoopKernel(null)! : MixedKernel(null)!;
            DispatchKernel(kernel, (uint)((n + Block - 1) / Block), (uint)((m + Block - 1) / Block), gz, [a, b, c], push);
            return;
        }

        int edge = VulkanKernels.MatSide(width) * (variant == MatBlocked ? VulkanKernels.MatPer : 1);
        RunAt(variant == MatBlocked ? "batched_matmul" : "batched_matmul_tile", width, (uint)((n + edge - 1) / edge), (uint)((m + edge - 1) / edge), gz, [a, b, c], push);
    }

    public override void Int8MatMulKernel(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k)
    {
        if (!Fit(x, scales, y) || !PackedProduct(VulkanKernels.PackedFormat.Int8, x, q, scales, y, m, n, k))
        {
            base.Int8MatMulKernel(x, q, scales, y, m, n, k);
        }
    }

    public override void Int4MatMulKernel(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k)
    {
        if (!Fit(x, y) || !PackedRows(VulkanKernels.PackedFormat.Int4, x, q, scales, y, m, n, k))
        {
            base.Int4MatMulKernel(x, q, scales, y, m, n, k);
        }
    }

    public override void BFloat16MatMulKernel(Storage x, Storage packed, Storage y, int m, int n, int k)
    {
        if (!Fit(x, y) || !PackedRows(VulkanKernels.PackedFormat.BFloat16, x, packed, null, y, m, n, k))
        {
            base.BFloat16MatMulKernel(x, packed, y, m, n, k);
        }
    }

    // Kernel names by (format · row blocks + row block) · word variants + word variant, built once.
    private static readonly string[] GemvKernelNames =
        [.. new[] { VulkanKernels.PackedFormat.Int8, VulkanKernels.PackedFormat.Int4, VulkanKernels.PackedFormat.BFloat16 }
            .SelectMany(f => VulkanKernels.GemvRowBlocks.SelectMany(r => VulkanKernels.GemvWordCounts.Select(w => VulkanKernels.GemvName(f, r, w))))];

    // y = x · w for packed weights (scales null for bfloat16): column blocks of L words × splits of k × row blocks of 1,
    // 2, 4 or 8 rows (the smallest that holds m, else 8). The words per row of k (L) and the splits are measured per
    // shape among those the device's workgroup counts take; the formula's choice is 32 words and as many splits as keep
    // the partial sums within the weights' traffic and each chunk at least GemvMinChunk rows. Several splits write their
    // partial sums to a temporary [splits, m, n] that gemv_reduce adds in split order. False when no variant fits the
    // device's workgroup counts (the caller falls back).
    private bool PackedProduct(VulkanKernels.PackedFormat format, Storage x, Storage weights, Storage? scales, Storage y, int m, int n, int k)
    {
        if (m <= 0 || n <= 0)
        {
            return true;
        }

        int block = GemvBlock(m);
        int rows = VulkanKernels.GemvRowBlocks[block];
        long rowBlocks = ((long)m + rows - 1) / rows;
        if (rowBlocks > Limits.MaxGroupsZ || PackedWindowRows(format, weights, scales, m, n, k) < 0)
        {
            return false;                                                      // (or weights larger than a binding that cannot be windowed)
        }

        int chosen = GemvChoice(format, block, x, weights, scales, y, m, n, k);
        if (chosen < 0)
        {
            return false;
        }

        RunPacked(format, block, chosen, x, weights, scales, y, m, n, k);
        return true;
    }

    // The row block variant (index into GemvRowBlocks) for m rows: the smallest that holds them, else 8.
    private static int GemvBlock(int m) => m switch { 1 => 0, 2 => 1, <= 4 => 2, _ => 3 };

    // The choice (width, splits and word variant) of a packed product of this shape: measured, stored, forced by the
    // test settings or the formula's (see PackedProduct). -1 when no variant fits the device's workgroup counts.
    private int GemvChoice(VulkanKernels.PackedFormat format, int block, Storage x, Storage weights, Storage? scales, Storage y, int m, int n, int k)
    {
        int perWord = VulkanKernels.ColumnsPerWord(format);
        // Splits allowed: chunks of at least GemvMinChunk rows, partial sums no more bytes than the weights.
        double weightBytes = (double)k * n * (format switch { VulkanKernels.PackedFormat.Int8 => 1, VulkanKernels.PackedFormat.Int4 => 0.5, _ => 2 });
        int maxSplits = (int)Math.Clamp(Math.Min((k + GemvMinChunk - 1) / GemvMinChunk, weightBytes / (4.0 * m * n)), 1, Limits.MaxGroupsY);
        // Formula: the device's width, 32 words, the most splits allowed. Measured: every candidate (GemvCandidates).
        int most = PowersOfTwo(maxSplits)[^1];
        int fallback = WithWidth(Width, most * 8);
        if (!GemvValid(fallback, perWord, n))
        {
            var all = GemvCandidates(perWord, n, maxSplits);
            if (all.Length == 0)
            {
                return -1;
            }

            fallback = all[^1];
        }

        var counts = VulkanKernels.GemvWordCounts;
        int chosen;
        if (GemvSplits is not null || GemvWords is not null)
        {
            int variant = Math.Max(0, Array.IndexOf(counts, GemvWords ?? counts[0]));
            chosen = WithWidth(Width, Math.Max(1, GemvSplits ?? most) * 8 + variant);
            chosen = GemvValid(chosen, perWord, n) ? chosen : fallback;
        }
        else
        {
            var key = new VulkanTuneKey(VulkanTuneOp.Gemv, (int)format * VulkanKernels.GemvRowBlocks.Length + block, m, n, k);
            if (!TryTuned(key, out chosen) || !GemvValid(chosen, perWord, n))
            {
                chosen = fallback;
                if (CanTune)
                {
                    Storage[] storages = scales is null ? [x, weights, y] : [x, weights, scales, y];
                    chosen = TuneWithScratch(key, GemvCandidates(perWord, n, maxSplits), fallback, storages, 1UL << (storages.Length - 1),
                        (c, s) => RunPacked(format, block, c, s[0], s[1], scales is null ? null : s[2], s[^1], m, n, k));
                }
            }
        }

        return chosen;
    }

    // At each candidate width, each valid word variant with 1, 2, 4, … splits up to maxSplits.
    private int[] GemvCandidates(int perWord, int n, int maxSplits)
    {
        var candidates = new List<int>();
        foreach (int width in CandidateWidths)
        {
            for (int v = 0; v < VulkanKernels.GemvWordCounts.Length; v++)
            {
                foreach (int splits in PowersOfTwo(maxSplits))
                {
                    if (GemvValid(WithWidth(width, splits * 8 + v), perWord, n))
                    {
                        candidates.Add(WithWidth(width, splits * 8 + v));
                    }
                }
            }
        }

        return [.. candidates];
    }

    // Whether a choice (a stored one too) can run here: a candidate width, a word variant whose slices fit a scale group
    // of 32 rows (int4 reads a row of each group per slice) and whose column blocks the device takes.
    private bool GemvValid(int choice, int perWord, int n)
    {
        int width = WidthOf(choice), variant = choice & 7;
        if (Array.IndexOf(CandidateWidths, width) < 0 || variant >= VulkanKernels.GemvWordCounts.Length)
        {
            return false;
        }

        int words = VulkanKernels.GemvWordCounts[variant];
        long columnBlocks = ((long)(n + perWord - 1) / perWord + words - 1) / words;
        return words <= width && width / words <= VulkanKernels.GemvChunkAlign && columnBlocks <= Limits.MaxGroupsX;
    }

    // Runs a packed product with `choice` = its width (WithWidth) and splits · 8 + word variant (the splits made
    // consistent with 32-row chunks).
    private void RunPacked(VulkanKernels.PackedFormat format, int block, int choice, Storage x, Storage weights, Storage? scales, Storage y, int m, int n, int k)
    {
        int width = WidthOf(choice), variant = choice & 7, rows = VulkanKernels.GemvRowBlocks[block], words = VulkanKernels.GemvWordCounts[variant];
        int perWord = VulkanKernels.ColumnsPerWord(format);
        uint columnBlocks = (uint)(((long)(n + perWord - 1) / perWord + words - 1) / words), rowBlocks = (uint)((m + rows - 1) / rows);
        string kernel = GemvKernelNames[((int)format * VulkanKernels.GemvRowBlocks.Length + block) * VulkanKernels.GemvWordCounts.Length + variant];
        if (PackedWindowRows(format, weights, scales, m, n, k) is var window and > 0)
        {
            // Weights larger than a binding: one dispatch per window of rows, the choice's chunks fitted to the windows.
            const int Align = VulkanKernels.GemvChunkAlign;
            int wanted = Math.Max(1, (choice & Rest) >> 3);
            int windowChunk = Math.Max(Align, ((k + wanted - 1) / wanted + Align - 1) / Align * Align);
            RunPackedWindows(format, kernel, width, columnBlocks, rows, windowChunk, (int)window, x, weights, scales, y, m, n, k);
            return;
        }

        var (splits, chunk) = GemvPlan(choice, k, (long)m * n);

        Span<byte> b = stackalloc byte[24];
        var push = new Push(b).I(m).I(n).I(k).I(chunk).I(0).I(splits).Bytes;
        var output = splits == 1 ? y : Allocate(splits * m * n, zeroed: false);
        try
        {
            if (scales is null)
            {
                RunAt(kernel, width, columnBlocks, (uint)splits, rowBlocks, [x, weights, output], push);
            }
            else
            {
                RunAt(kernel, width, columnBlocks, (uint)splits, rowBlocks, [x, weights, scales, output], push);
            }

            if (splits > 1)
            {
                Span<byte> r = stackalloc byte[12];
                var reducePush = new Push(r).I(m * n).I(n).I(splits).Bytes;
                if (format == VulkanKernels.PackedFormat.Int8)
                {
                    Grid("int8_gemv_reduce", (long)m * n, [output, scales!, y], reducePush);
                }
                else
                {
                    Grid("gemv_reduce", (long)m * n, [output, y], reducePush);
                }
            }
        }
        finally
        {
            if (splits > 1)
            {
                output.Release();                                              // reused in queue order
            }
        }
    }

    public override void Int8DequantizeKernel(Storage q, Storage scales, Storage w, int k, int n)
    {
        if (!Fit(q, scales, w))
        {
            if (!DequantizeWindows(VulkanKernels.PackedFormat.Int8, q, scales, w, k, n))
            {
                base.Int8DequantizeKernel(q, scales, w, k, n);
            }

            return;
        }

        Span<byte> b = stackalloc byte[8];
        Grid("int8_dequantize", (long)k * n, [q, scales, w], new Push(b).I(k).I(n).Bytes);
    }

    public override void Int4DequantizeKernel(Storage q, Storage scales, Storage w, int k, int n)
    {
        if (!Fit(q, scales, w))
        {
            if (!DequantizeWindows(VulkanKernels.PackedFormat.Int4, q, scales, w, k, n))
            {
                base.Int4DequantizeKernel(q, scales, w, k, n);
            }

            return;
        }

        Span<byte> b = stackalloc byte[8];
        Grid("int4_dequantize", (long)k * n, [q, scales, w], new Push(b).I(k).I(n).Bytes);
    }

    public override void BFloat16DequantizeKernel(Storage packed, Storage w, int k, int n)
    {
        if (!Fit(packed, w))
        {
            if (!DequantizeWindows(VulkanKernels.PackedFormat.BFloat16, packed, null, w, k, n))
            {
                base.BFloat16DequantizeKernel(packed, w, k, n);
            }

            return;
        }

        Span<byte> b = stackalloc byte[8];
        Grid("bf16_dequantize", (long)k * n, [packed, w], new Push(b).I(k).I(n).Bytes);
    }

    public override void PackBFloat16Kernel(Storage x, Storage packed, int n)
    {
        if (!Fit(x, packed))
        {
            base.PackBFloat16Kernel(x, packed, n);
            return;
        }

        Span<byte> b = stackalloc byte[4];
        Grid("bf16_pack", (n + 1L) / 2, [x, packed], new Push(b).I(n).Bytes);
    }

    // ------------------------------------------------------------------ language models

    public override void RopeKernel(Storage x, Storage y, Storage cos, Storage sin, Storage positions, int rows, int heads, int steps, int dim, int half, bool interleaved, float sign)
    {
        if (!Fit(x, y, cos, sin, positions))
        {
            base.RopeKernel(x, y, cos, sin, positions, rows, heads, steps, dim, half, interleaved, sign);
            return;
        }

        Span<byte> b = stackalloc byte[28];
        Grid("rope", (long)rows * half, [x, y, cos, sin, positions],
            new Push(b).I(rows).I(heads).I(steps).I(dim).I(half).B(interleaved).F(sign).Bytes);
    }

    // Indices outside the table are clamped into it (the CPU throws; CUDA clamps too).
    public override void GatherKernel(Storage table, Storage indices, Storage y, int count, int dim, int vocabulary)
    {
        if (!Fit(table, indices, y))
        {
            if (!GatherWindows("gather", table, indices, y, count, dim, vocabulary, dim))
            {
                base.GatherKernel(table, indices, y, count, dim, vocabulary);
            }

            return;
        }

        Span<byte> b = stackalloc byte[20];
        Grid("gather", (long)count * dim, [table, indices, y], new Push(b).I(count).I(dim).I(vocabulary).I(0).I(vocabulary).Bytes);
    }

    public override void GatherBFloat16Kernel(Storage packed, Storage indices, Storage y, int count, int dim, int vocabulary)
    {
        if (!Fit(packed, indices, y))
        {
            if (!GatherWindows("gather_bf16", packed, indices, y, count, dim, vocabulary, (dim + 1) / 2))
            {
                base.GatherBFloat16Kernel(packed, indices, y, count, dim, vocabulary);
            }

            return;
        }

        Span<byte> b = stackalloc byte[20];
        Grid("gather_bf16", (long)count * dim, [packed, indices, y], new Push(b).I(count).I(dim).I(vocabulary).I(0).I(vocabulary).Bytes);
    }

    public override void KeyValueWriteKernel(Storage source, Storage cache, Storage position, int heads, int steps, int capacity, int dim)
    {
        if (!Fit(source, cache, position))
        {
            base.KeyValueWriteKernel(source, cache, position, heads, steps, capacity, dim);
            return;
        }

        Span<byte> b = stackalloc byte[16];
        Grid("key_value_write", (long)heads * steps * dim, [source, cache, position], new Push(b).I(heads).I(steps).I(capacity).I(dim).Bytes);
    }

    public override void KeyValueWriteBFloat16Kernel(Storage source, Storage cache, Storage position, int heads, int steps, int capacity, int dim)
    {
        if (!Fit(source, cache, position))
        {
            base.KeyValueWriteBFloat16Kernel(source, cache, position, heads, steps, capacity, dim);
            return;
        }

        Span<byte> b = stackalloc byte[16];
        Grid("key_value_write_bf16", (long)heads * steps * ((dim + 1) / 2), [source, cache, position],
            new Push(b).I(heads).I(steps).I(capacity).I(dim).Bytes);
    }

    public override void KeyValueWriteInt8Kernel(Storage source, Storage cache, Storage scales, Storage position, int heads, int steps, int capacity, int dim)
    {
        if (!Fit(source, cache, scales, position))
        {
            base.KeyValueWriteInt8Kernel(source, cache, scales, position, heads, steps, capacity, dim);
            return;
        }

        Span<byte> b = stackalloc byte[16];
        Grid("key_value_write_int8", (long)heads * steps, [source, cache, scales, position], new Push(b).I(heads).I(steps).I(capacity).I(dim).Bytes);
    }

    public override void DecoderMaskKernel(Storage position, Storage mask, int rows, int capacity)
    {
        if (!Fit(position, mask))
        {
            base.DecoderMaskKernel(position, mask, rows, capacity);
            return;
        }

        Span<byte> b = stackalloc byte[8];
        Grid("decoder_mask", (long)rows * capacity, [position, mask], new Push(b).I(rows).I(capacity).Bytes);
    }

    public override void AttentionDecodeKernel(Storage q, Storage keys, Storage values, Storage position, Storage y, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale, AttentionVariant variant = default)
    {
        if (dim > VulkanKernels.AttentionMaxDim || !Fit(q, keys, values, position, y))
        {
            base.AttentionDecodeKernel(q, keys, values, position, y, heads, rowsPerHead, steps, capacity, dim, scale, variant);
            return;
        }

        Span<Storage> storages = [q, keys, values, position, y];
        Attend("attention_decode", 0, storages, heads, rowsPerHead, steps, capacity, dim, scale, variant);
    }

    public override void AttentionBFloat16Kernel(Storage q, Storage keys, Storage values, Storage position, Storage y, int heads, int rowsPerHead,
        int steps, int capacity, int dim, float scale, bool tiled, AttentionVariant variant = default)
    {
        if (dim > VulkanKernels.AttentionMaxDim || !Fit(q, keys, values, position, y))
        {
            base.AttentionBFloat16Kernel(q, keys, values, position, y, heads, rowsPerHead, steps, capacity, dim, scale, tiled, variant);
            return;
        }

        Span<Storage> storages = [q, keys, values, position, y];
        Attend("attention_bf16", 1, storages, heads, rowsPerHead, steps, capacity, dim, scale, variant);
    }

    public override void AttentionInt8Kernel(Storage q, Storage keys, Storage values, Storage keyScales, Storage valueScales, Storage position,
        Storage y, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale, bool tiled, AttentionVariant variant = default)
    {
        if (dim > VulkanKernels.AttentionMaxDim || !Fit(q, keys, values, keyScales, valueScales, position, y))
        {
            base.AttentionInt8Kernel(q, keys, values, keyScales, valueScales, position, y, heads, rowsPerHead, steps, capacity, dim, scale, tiled, variant);
            return;
        }

        Span<Storage> storages = [q, keys, values, keyScales, valueScales, position, y];
        Attend("attention_int8", 2, storages, heads, rowsPerHead, steps, capacity, dim, scale, variant);
    }

    // Dispatches an attention kernel (storages: its bindings, the position second to last, the output last): query rows ×
    // splits of the cached positions. The split count depends only on the shapes (each split's positions follow the
    // current length on the device, at least an eighth of the width each): measured per shape over a full cache (where
    // attention costs most; shorter contexts then leave the extra splits empty), up to one per AttentionMinChunk
    // positions of capacity; the formula's choice is a tile (the width) of positions per split. With several splits the
    // kernel writes each split's (weighted values, max, total) to a temporary [rows, splits, dim + 2] and
    // attention_combine merges them into the output. A window shorter than the capacity is measured under a key of its
    // own (formats 3 to 5), over the window's positions: each split then takes a share of the window.
    private void Attend(string kernel, int format, Span<Storage> storages, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale,
        AttentionVariant variant = default)
    {
        int rows = heads * rowsPerHead;
        if (rows <= 0 || dim <= 0)
        {
            return;
        }

        // Formula: the device's width, a tile (the width) of positions per split. Measured: every candidate. A window
        // shorter than the capacity counts as the positions each row reads.
        bool windowed = variant.Window > 0 && variant.Window < capacity;
        int span = windowed ? variant.Window : capacity;
        int fallback = WithWidth(Width, PowersOfTwo(Math.Min(Math.Max(1, span / Width), AttentionMaxSplits(Width, rows, span, dim)))[^1]);
        int choice;
        if (AttentionSplits is int forced)
        {
            choice = WithWidth(Width, PowersOfTwo(Math.Min(Math.Max(1, forced), AttentionMaxSplits(Width, rows, span, dim)))[^1]);
        }
        else
        {
            var key = windowed
                ? new VulkanTuneKey(VulkanTuneOp.Attention, format + 3, rows, rowsPerHead, steps, capacity, dim, span)
                : new VulkanTuneKey(VulkanTuneOp.Attention, format, rows, rowsPerHead, steps, capacity, dim);
            if (!TryTuned(key, out choice) || Array.IndexOf(CandidateWidths, WidthOf(choice)) < 0
                || (choice & Rest) > AttentionMaxSplits(WidthOf(choice), rows, span, dim) || (choice & Plain) != 0 && !Limits.SubgroupArithmetic)
            {
                choice = fallback;
                if (CanTune)
                {
                    // At each candidate width, 1, 2, 4, … splits; timed over a full cache: the position read from a
                    // scratch storage holding capacity - 1.
                    var candidates = CandidateWidths.SelectMany(w => PowersOfTwo(AttentionMaxSplits(w, rows, span, dim))
                        .SelectMany(s => ReductionFlags.Select(f => WithWidth(w, s | f)))).ToArray();
                    var saved = storages.ToArray();
                    var full = Allocate(1, zeroed: false);
                    try
                    {
                        Fill(full, 1, capacity - 1);
                        saved[^2] = full;
                        choice = TuneWithScratch(key, candidates, fallback, saved, 1UL << (saved.Length - 1),
                            (c, bound) => RunAttention(kernel, bound, rows, c, heads, rowsPerHead, steps, capacity, dim, scale, variant));
                    }
                    finally
                    {
                        full.Release();
                    }
                }
            }
        }

        RunAttention(kernel, storages, rows, choice, heads, rowsPerHead, steps, capacity, dim, scale, variant);
    }

    // Most splits at a width: one per AttentionMinChunk(width) positions of capacity, fewer when the partial results would
    // not fit one storage.
    private int AttentionMaxSplits(int width, int rows, int capacity, int dim)
    {
        int most = (int)Math.Clamp(capacity / VulkanKernels.AttentionMinChunk(width), 1, Limits.MaxGroupsY);
        while (most > 1 && ((long)rows * most * (dim + 2) > int.MaxValue || BlockBytes((int)Math.Min(int.MaxValue, (long)rows * most * (dim + 2))) > MaxStorageBytes))
        {
            most /= 2;
        }

        return most;
    }

    // Runs attention with `choice` = its width (WithWidth) and split count.
    private void RunAttention(string kernel, Span<Storage> storages, int rows, int choice, int heads, int rowsPerHead, int steps, int capacity, int dim, float scale,
        AttentionVariant variant)
    {
        // Dimensions per part of the workgroup: the head size rounded up to a power of two, at most the width.
        int width = WidthOf(choice), splits = choice & (Plain - 1);
        bool subgroups = SubgroupsOf(choice);
        int dimShift = Math.Min(BitOperations.Log2(BitOperations.RoundUpToPowerOf2((uint)dim)), BitOperations.Log2((uint)width));
        Span<byte> b = stackalloc byte[36];
        var push = new Push(b).I(heads).I(rowsPerHead).I(steps).I(capacity).I(dim).F(scale).I(dimShift).I(variant.Window).F(variant.Softcap).Bytes;

        if (splits <= 1)
        {
            RunAt(kernel, width, RowGroups(rows), 1, 1, storages, push, subgroups);
            return;
        }

        var y = storages[^1];
        var part = Allocate(rows * splits * (dim + 2), zeroed: false);
        try
        {
            storages[^1] = part;
            RunAt(kernel, width, RowGroups(rows), (uint)splits, 1, storages, push, subgroups);
            Span<byte> c = stackalloc byte[12];
            Grid("attention_combine", (long)rows * dim, [part, y], new Push(c).I(rows).I(splits).I(dim).Bytes);
        }
        finally
        {
            storages[^1] = y;
            part.Release();                                                    // reused in queue order
        }
    }
}
