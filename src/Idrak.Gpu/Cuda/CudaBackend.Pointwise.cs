// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Cuda;

// Element-wise operations with parameters or several gradients (PtxKernels.Pointwise.cs): powers, clamping, selection
// by a mask and the gradient of element-wise extremes, as the CPU computes them. A gradient the caller leaves out is
// passed as another valid pointer with its flag clear; the kernels never touch it.
internal sealed unsafe partial class CudaBackend
{
    public override void ExtremumBackwardKernel(BinaryOp op, Storage a, Storage b, Storage dy, Storage? da, Storage? db, int n)
    {
        int flags = (da is null ? 0 : 1) | (db is null ? 0 : 2) | (op == BinaryOp.Minimum ? 4 : 0);
        Launch1D(K("extremum_bwd_f32"), n, P(a), P(b), P(dy), P(da ?? dy), P(db ?? dy), U(flags), U(n));
    }

    public override void PowKernel(Storage x, Storage y, int n, float exponent) =>
        Launch1D(K("pow_f32"), n, P(x), P(y), F(exponent), PtxKernels.PowFlags(exponent), U(n));

    public override void PowBackwardKernel(Storage x, Storage dy, Storage dx, int n, float exponent)
    {
        if (exponent == 0f)
        {
            return;                                                         // a constant: no gradient (not 0 · x^-1, NaN at 0)
        }

        float p = exponent - 1f;
        Launch1D(K("pow_bwd_f32"), n, P(x), P(dy), P(dx), F(exponent), F(p), PtxKernels.PowFlags(p), U(n));
    }

    public override void ClampKernel(Storage x, Storage y, int n, float min, float max) =>
        Launch1D(K("clamp_f32"), n, P(x), P(y), F(min), F(max), U(n));

    public override void ClampBackwardKernel(Storage x, Storage dy, Storage dx, int n, float min, float max) =>
        Launch1D(K("clamp_bwd_f32"), n, P(x), P(dy), P(dx), F(min), F(max), U(n));

    public override void WhereKernel(Storage condition, Storage a, Storage b, Storage y, int n) =>
        Launch1D(K("where_f32"), n, P(condition), P(a), P(b), P(y), U(n));

    public override void WhereBackwardKernel(Storage condition, Storage dy, Storage? da, Storage? db, int n)
    {
        int flags = (da is null ? 0 : 1) | (db is null ? 0 : 2);
        Launch1D(K("where_bwd_f32"), n, P(condition), P(dy), P(da ?? dy), P(db ?? dy), U(flags), U(n));
    }
}
