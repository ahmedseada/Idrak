// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Gpu.Hip;

// HIP's shape operations on its own GCN kernels (GcnKernels.Shapes.cs), where CUDA overrides them and with CUDA's
// arguments: the key/value cache writes (float32, bfloat16, int8), the decoder mask, the gathers (float32, bfloat16 rows
// and columns), the permute and the adding strided copy (the plain one stays the runtime's 2-D copy, HipBackend.cs). Each
// kernel walks its elements grid-stride, one a work-item, so its grid fills the device as an element-wise launch's does
// (PointwiseBlocks over the elements, rule 81); each launch is labelled for --profile with its kernel, its count and its
// floating-point operations (Cost). The host fallback takes over when the kernels are unavailable or a buffer the kernel
// indexes holds 2^30 floats or more (its 32-bit byte offsets), as it does for every operation without a kernel.
internal sealed unsafe partial class HipBackend
{
    // A shape kernel, or null when the kernels are unavailable.
    private GcnFunction? ShapeKernel(string name) => Gcn(name);

    /// <summary>The blocks a one-element-a-work-item launch over <paramref name="n"/> elements takes: an element-wise launch's grid for as many work-items.</summary>
    internal uint ShapeBlocks(long n) => PointwiseBlocks(4 * n);

    // Launches a shape kernel over n > 0 work-items, labelled for the profile.
    private void LaunchShape(GcnFunction function, int n, double flops, params ReadOnlySpan<ulong> values)
    {
        if (n <= 0)
        {
            return;
        }

        if (Profiling)
        {
            LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"{function.Name} {n}"), flops);
        }

        LaunchKernel(function, ShapeBlocks(n), 1, values);
    }

    // Whether every count stays below the kernels' 2^30 floats.
    private static bool ShapeFits(long a, long b = 0, long c = 0) =>
        a < GcnKernels.ShapeMostElements && b < GcnKernels.ShapeMostElements && c < GcnKernels.ShapeMostElements;

    public override void KeyValueWriteKernel(Storage source, Storage cache, Storage position, int heads, int steps, int capacity, int dim)
    {
        long n = (long)heads * steps * dim;
        if (dim <= 0 || steps <= 0 || !ShapeFits(n, (long)heads * capacity * dim) || ShapeKernel("kv_write_f32") is not { } f)
        {
            base.KeyValueWriteKernel(source, cache, position, heads, steps, capacity, dim);
            return;
        }

        LaunchShape(f, (int)n, 0, P(source), P(cache), P(position), U(steps * dim), U(capacity * dim), U(dim), U((int)n));
    }

    public override void KeyValueWriteBFloat16Kernel(Storage source, Storage cache, Storage position, int heads, int steps, int capacity, int dim)
    {
        int words = (dim + 1) / 2;
        long n = (long)heads * steps * words;
        if (dim <= 0 || steps <= 0 || !ShapeFits((long)heads * steps * dim, (long)heads * capacity * words) || ShapeKernel("kv_write_bf16") is not { } f)
        {
            base.KeyValueWriteBFloat16Kernel(source, cache, position, heads, steps, capacity, dim);
            return;
        }

        LaunchShape(f, (int)n, 0, P(source), P(cache), P(position), U(steps), U(capacity), U(dim), U(words), U((int)n));
    }

    public override void KeyValueWriteInt8Kernel(Storage source, Storage cache, Storage scales, Storage position, int heads, int steps, int capacity, int dim)
    {
        int words = (dim + 3) / 4;
        long n = (long)heads * steps;
        if (dim <= 0 || steps <= 0 || capacity <= 0 || !ShapeFits((long)heads * steps * dim, (long)heads * capacity * words) || ShapeKernel("kv_write_int8") is not { } f)
        {
            base.KeyValueWriteInt8Kernel(source, cache, scales, position, heads, steps, capacity, dim);
            return;
        }

        LaunchShape(f, (int)n, Cost(f.Name).Flops * n * dim, P(source), P(cache), P(scales), P(position), U(steps), U(capacity), U(dim), U(words), U((int)n));
    }

    public override void DecoderMaskKernel(Storage position, Storage mask, int rows, int capacity)
    {
        long n = (long)rows * capacity;
        if (capacity <= 0 || !ShapeFits(n) || ShapeKernel("decoder_mask_f32") is not { } f)
        {
            base.DecoderMaskKernel(position, mask, rows, capacity);
            return;
        }

        LaunchShape(f, (int)n, 0, P(position), P(mask), U(capacity), U((int)n));
    }

    public override void GatherKernel(Storage table, Storage indices, Storage y, int count, int dim, int vocabulary)
    {
        long n = (long)count * dim;
        if (dim <= 0 || vocabulary <= 0 || !ShapeFits(n, (long)vocabulary * dim) || ShapeKernel("gather_f32") is not { } f)
        {
            base.GatherKernel(table, indices, y, count, dim, vocabulary);
            return;
        }

        LaunchShape(f, (int)n, 0, P(table), P(indices), P(y), U(dim), U(vocabulary - 1), U((int)n));
    }

    public override void GatherBFloat16Kernel(Storage packed, Storage indices, Storage y, int count, int dim, int vocabulary)
    {
        int words = (dim + 1) / 2;
        long n = (long)count * dim;
        if (dim <= 0 || vocabulary <= 0 || !ShapeFits(n, (long)vocabulary * words) || ShapeKernel("gather_bf16_f32") is not { } f)
        {
            base.GatherBFloat16Kernel(packed, indices, y, count, dim, vocabulary);
            return;
        }

        LaunchShape(f, (int)n, 0, P(packed), P(indices), P(y), U(dim), U(vocabulary - 1), U(words), U((int)n));
    }

    public override void GatherBFloat16ColumnsKernel(Storage packed, Storage indices, Storage y, int count, int dim, int vocabulary)
    {
        int words = (vocabulary + 1) / 2;
        long n = (long)count * dim;
        if (dim <= 0 || vocabulary <= 0 || !ShapeFits(n, (long)dim * words) || ShapeKernel("gather_bf16_cols_f32") is not { } f)
        {
            base.GatherBFloat16ColumnsKernel(packed, indices, y, count, dim, vocabulary);
            return;
        }

        LaunchShape(f, (int)n, 0, P(packed), P(indices), P(y), U(dim), U(vocabulary - 1), U(words), U((int)n));
    }

    public override void PermuteKernel(Storage x, Storage y, ReadOnlySpan<int> outShape, ReadOnlySpan<int> inStrides, bool accumulate)
    {
        const int R = GcnKernels.ShapePermuteRank;
        long n = 1, last = 0;
        bool fits = outShape.Length <= R && outShape.Length == inStrides.Length;
        for (int d = 0; fits && d < outShape.Length; d++)
        {
            fits = outShape[d] >= 0 && inStrides[d] >= 0;
            n *= outShape[d];
            last += (long)Math.Max(0, outShape[d] - 1) * inStrides[d];
            fits &= ShapeFits(n, last);
        }

        if (!fits || ShapeKernel("permute_f32") is not { } f)
        {
            base.PermuteKernel(x, y, outShape, inStrides, accumulate);
            return;
        }

        if (n == 0)
        {
            return;
        }

        // Right-aligned to six dimensions; the unused leading ones of size 1 and stride 0 (CUDA's arguments).
        Span<ulong> arguments = stackalloc ulong[2 + 2 * R + 2];
        int pad = R - outShape.Length;
        arguments[0] = P(x);
        arguments[1] = P(y);
        for (int d = 0; d < R; d++)
        {
            arguments[2 + d] = d < pad ? 1UL : (uint)outShape[d - pad];
            arguments[2 + R + d] = d < pad ? 0UL : (uint)inStrides[d - pad];
        }

        arguments[2 + 2 * R] = accumulate ? 1UL : 0UL;
        arguments[3 + 2 * R] = (uint)n;
        LaunchShape(f, (int)n, Cost(f.Name, variant: accumulate).Flops * n, arguments);
    }

    // The strided copy on copy2d_f32 (adding, within one storage, or with a pitch narrower than a row): false when it
    // cannot run here (the host's).
    private bool Copy2DKernel(Storage src, int srcOffset, int srcStride, Storage dst, int dstOffset, int dstStride, int rows, int cols, bool accumulate)
    {
        if (rows <= 0 || cols <= 0)
        {
            return true;
        }

        long n = (long)rows * cols;
        if (srcOffset < 0 || dstOffset < 0 || srcStride < 0 || dstStride < 0
            || !ShapeFits(n, srcOffset + (long)(rows - 1) * srcStride + cols, dstOffset + (long)(rows - 1) * dstStride + cols)
            || ShapeKernel("copy2d_f32") is not { } f)
        {
            return false;
        }

        LaunchShape(f, (int)n, Cost(f.Name, variant: accumulate).Flops * n, P(src), P(dst), U(srcOffset), U(srcStride), U(dstOffset), U(dstStride), U(cols),
            U(accumulate ? 1 : 0), U((int)n));
        return true;
    }
}
