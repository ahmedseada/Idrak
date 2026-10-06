// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Backends.Vulkan;

// Element-wise operations with parameters or several gradients as generated kernels (VulkanKernels.Pointwise.cs):
// powers, clamping, selection by a mask and the gradient of element-wise extremes.
internal sealed partial class VulkanBackend
{
    public override void PowKernel(Storage x, Storage y, int n, float exponent)
    {
        if (!Fit(x, y))
        {
            base.PowKernel(x, y, n, exponent);
            return;
        }

        var (negative, atZero) = PowParts(exponent);
        Span<byte> b = stackalloc byte[16];
        Grid("pow", n, [x, y], new Push(b).I(n).F(exponent).F(negative).F(atZero).Bytes);
    }

    public override void PowBackwardKernel(Storage x, Storage dy, Storage dx, int n, float exponent)
    {
        if (exponent == 0f)
        {
            return;                                                         // a constant: no gradient
        }

        if (!Fit(x, dy, dx))
        {
            base.PowBackwardKernel(x, dy, dx, n, exponent);
            return;
        }

        var (negative, atZero) = PowParts(exponent - 1f);
        Span<byte> b = stackalloc byte[16];
        Grid("pow_backward", n, [x, dy, dx], new Push(b).I(n).F(exponent - 1f).F(negative).F(atZero).Bytes);
    }

    // What the pow kernel multiplies |x|^p by for negative x, and its value at zero (as MathF.Pow gives them).
    private static (float Negative, float AtZero) PowParts(float p)
    {
        float negative = MathF.Round(p) != p ? float.NaN : MathF.IEEERemainder(p, 2f) == 0f ? 1f : -1f;
        float atZero = p > 0f ? 0f : p == 0f ? 1f : float.PositiveInfinity;
        return (negative, atZero);
    }

    public override void ClampKernel(Storage x, Storage y, int n, float min, float max)
    {
        if (!Fit(x, y))
        {
            base.ClampKernel(x, y, n, min, max);
            return;
        }

        Span<byte> b = stackalloc byte[12];
        Grid("clamp", n, [x, y], new Push(b).I(n).F(min).F(max).Bytes);
    }

    public override void ClampBackwardKernel(Storage x, Storage dy, Storage dx, int n, float min, float max)
    {
        if (!Fit(x, dy, dx))
        {
            base.ClampBackwardKernel(x, dy, dx, n, min, max);
            return;
        }

        Span<byte> b = stackalloc byte[12];
        Grid("clamp_backward", n, [x, dy, dx], new Push(b).I(n).F(min).F(max).Bytes);
    }

    public override void WhereKernel(Storage condition, Storage a, Storage b, Storage y, int n)
    {
        if (!Fit(condition, a, b, y))
        {
            base.WhereKernel(condition, a, b, y, n);
            return;
        }

        Span<byte> p = stackalloc byte[4];
        Grid("where", n, [condition, a, b, y], new Push(p).I(n).Bytes);
    }

    public override void WhereBackwardKernel(Storage condition, Storage dy, Storage? da, Storage? db, int n)
    {
        if (!Fit(condition, dy, da ?? dy, db ?? dy))
        {
            base.WhereBackwardKernel(condition, dy, da, db, n);
            return;
        }

        int flags = (da is null ? 0 : 1) | (db is null ? 0 : 2);
        Span<byte> p = stackalloc byte[8];
        Grid("where_backward", n, [condition, dy, da ?? dy, db ?? dy], new Push(p).I(n).I(flags).Bytes);
    }

    public override void ExtremumBackwardKernel(BinaryOp op, Storage a, Storage b, Storage dy, Storage? da, Storage? db, int n)
    {
        if (!Fit(a, b, dy, da ?? dy, db ?? dy))
        {
            base.ExtremumBackwardKernel(op, a, b, dy, da, db, n);
            return;
        }

        int flags = (da is null ? 0 : 1) | (db is null ? 0 : 2) | (op == BinaryOp.Minimum ? 4 : 0);
        Span<byte> p = stackalloc byte[8];
        Grid("extremum_backward", n, [a, b, dy, da ?? dy, db ?? dy], new Push(p).I(n).I(flags).Bytes);
    }
}
