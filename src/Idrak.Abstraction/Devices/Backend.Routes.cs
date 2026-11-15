// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices;

// The experiment of Adad's design (ProductRoutes): the passes a device may fuse over weights stored in their file's layout
// (StoredWeight), outside the generated operations.
public abstract partial class Backend
{
    /// <summary>
    /// Few-row products of stored weights sharing one input x [m, k] in one pass (the experiment's fused projections:
    /// queries, keys and values; gate and up): y_j = x · W_jᵀ for 2 or 3 weights of one element type, each n_j rows of k
    /// values with a scale a row or none (<see cref="StoredMatMulKernel"/>'s layout), each output as its own product
    /// computes it. False where the device has no single pass for them (callers then run the products one by one).
    /// </summary>
    /// <param name="x">The input [m, k].</param>
    /// <param name="m">Rows of x.</param>
    /// <param name="k">Columns of x, values of a weight row.</param>
    /// <param name="products">Each weight's values and scales (or none), its output [m, n_j] and n_j.</param>
    public virtual bool StoredMatMulManyKernel(Storage x, int m, int k, ReadOnlySpan<(Storage Values, Storage? Scales, Storage Output, int Columns)> products) => false;

    /// <summary>Whether this device groups stored weights' products of <paramref name="m"/> rows (StoredMatMulManyKernel), asked before outputs are allocated.</summary>
    /// <param name="m">Rows of x.</param>
    public virtual bool StoredMatMulManyAvailable(int m) => false;

    /// <summary>
    /// Whether this device fuses a stored weight's epilogue at the shape (<paramref name="kind"/> 0: the gate/up pair with
    /// activation <paramref name="detail"/>; 1: the residual and norm, <paramref name="detail"/> 1 with row scales): true
    /// where its fused pass measured faster, false where the composition did or it has none, null where it has not
    /// measured yet (callers then ask the fused pass, which measures). Asked before any output is allocated.
    /// </summary>
    /// <param name="kind">0 the gate/up pair, 1 the residual and norm.</param>
    /// <param name="detail">The activation (kind 0) or whether the weight has row scales (kind 1).</param>
    /// <param name="type">The weight's element type.</param>
    /// <param name="m">Rows of x.</param>
    /// <param name="n">Rows of the weight.</param>
    /// <param name="k">Columns of x.</param>
    public virtual bool? StoredEpilogueFused(int kind, int detail, ElementType type, int m, int n, int k) => false;

    /// <summary>
    /// A feed-forward block's gate and up products of stored weights (StoredMatMulManyKernel's two, of equal widths) and
    /// hidden = act(gate) · up (activation 0 SiLU, 1 tanh GELU, 2 ReLU) in one pass. False where the device has no such pass.
    /// </summary>
    /// <param name="activation">0 SiLU, 1 tanh GELU, 2 ReLU.</param>
    /// <param name="x">The input [m, k].</param>
    /// <param name="m">Rows of x.</param>
    /// <param name="k">Columns of x.</param>
    /// <param name="products">The gate's then the up's values, scales, output [m, n] and n.</param>
    /// <param name="hidden">act(gate) · up [m, n].</param>
    public virtual bool StoredMatMulGatedPairKernel(int activation, Storage x, int m, int k, ReadOnlySpan<(Storage Values, Storage? Scales, Storage Output, int Columns)> products,
        Storage hidden) => false;

    /// <summary>
    /// y = x · Wᵀ for a stored weight W (n rows of k, a scale a row or none), sum = residual + y and normalized = its RMS
    /// normalization with gain (value · 1 / sqrt(mean of squares + eps) · (gain + offset)) in one pass, few rows. False
    /// where the device has no such pass.
    /// </summary>
    /// <param name="x">The input [m, k].</param>
    /// <param name="weight">W's values.</param>
    /// <param name="scales">W's scales a row, or null.</param>
    /// <param name="y">The product [m, n].</param>
    /// <param name="m">Rows of x.</param>
    /// <param name="n">Rows of W.</param>
    /// <param name="k">Columns of x.</param>
    /// <param name="residual">Added to y [m, n].</param>
    /// <param name="sum">residual + y [m, n].</param>
    /// <param name="gain">The norm's gain [n].</param>
    /// <param name="normalized">The normalized sum [m, n].</param>
    /// <param name="eps">The norm's epsilon.</param>
    /// <param name="offset">Added to the gain.</param>
    public virtual bool StoredMatMulAddRmsNormKernel(Storage x, Storage weight, Storage? scales, Storage y, int m, int n, int k, Storage residual, Storage sum, Storage gain,
        Storage normalized, float eps, float offset) => false;
}
