// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices;

// The composed convolution (the default of ConvolutionKernel and its two gradients): the patches unfolded by Im2Col, one
// product per group (BatchedMatMul, which runs on a device's matrix units where it has them), and permutations between
// the row layout of the products ([groups, positions, ...]) and NCHW. Every device runs it with its own Im2Col, products and
// permutations; a device with convolution kernels of its own measures them against it per shape.
//
// The unfolded patches of a batch are KH·KW times its input: when they would pass a storage's int range, or half the
// memory the device reports free, the images go in chunks (each through copies to and from the full tensors at its
// offset); otherwise in one piece, as before these operations existed.
public abstract partial class Backend
{
    // Images per chunk: all of them unless the patches of all would not fit (see above).
    private int ConvolutionChunk(in ConvGeometry g)
    {
        long perImage = Math.Max(1L, (long)g.OH * g.OW * g.PatchSize);
        long limit = int.MaxValue;
        if (perImage * g.N > int.MaxValue / 4 && AvailableMemory() is long free && free > 0)
        {
            limit = Math.Min(limit, free / 2 / sizeof(float));
        }

        return (int)Math.Clamp(limit / perImage, 1, Math.Max(1, g.N));
    }

    // Scratch storages of the composition, released together.
    private sealed class Scratch(Backend backend)
    {
        private readonly List<Storage> _taken = new(5);

        public Storage Take(int length, bool zeroed = false)
        {
            var s = backend.Allocate(Math.Max(1, length), zeroed);
            _taken.Add(s);
            return s;
        }

        public void Release()
        {
            foreach (var s in _taken)
            {
                s.Release();
            }

            _taken.Clear();
        }
    }

    // The activation as the element-wise operation it applies.
    internal static UnaryOp? ActivationOp(ConvActivation activation) => activation switch
    {
        ConvActivation.Relu => UnaryOp.Relu,
        ConvActivation.Sigmoid => UnaryOp.Sigmoid,
        ConvActivation.Tanh => UnaryOp.Tanh,
        ConvActivation.Gelu => UnaryOp.Gelu,
        ConvActivation.Silu => UnaryOp.Silu,
        _ => null,
    };

    /// <summary>
    /// The convolution as unfolded patches and products, on this device's own operations (the default of
    /// <see cref="ConvolutionKernel"/>; devices with kernels of their own measure them against it).
    /// </summary>
    protected void ComposedConvolution(Storage x, Storage weight, Storage? bias, Storage y, in ConvGeometry g, int filters, int groups, ConvActivation activation)
    {
        int outputs = g.OH * g.OW;
        if (g.N <= 0 || filters <= 0 || outputs <= 0)
        {
            return;
        }

        int chunk = ConvolutionChunk(in g), inImage = g.C * g.H * g.W, outImage = filters * outputs;
        for (int first = 0; first < g.N; first += chunk)
        {
            int images = Math.Min(chunk, g.N - first);
            var part = g with { N = images };
            var scratch = new Scratch(this);
            try
            {
                Storage input = x, output = y;
                if (images != g.N)
                {
                    input = scratch.Take(images * inImage);
                    Copy2D(x, first * inImage, 0, input, 0, 0, 1, images * inImage, false);
                    output = scratch.Take(images * outImage);
                }

                ConvolutionProducts(input, weight, output, in part, filters, groups, scratch);
                if (images != g.N)
                {
                    Copy2D(output, 0, 0, y, first * outImage, 0, 1, images * outImage, false);
                }
            }
            finally
            {
                scratch.Release();
            }
        }

        int n = g.N * outImage;
        if (bias is not null)
        {
            GroupScaleShift(y, null, bias, y, n, filters, outputs, accumulate: false);
        }

        if (ActivationOp(activation) is { } op)
        {
            Unary(op, y, y, n);
        }
    }

    // y [N, filters, OH·OW] = the products of one chunk (no bias): rows [G, P, Fg] = patches [G, P, pg] · weight [G, Fg, pg]ᵀ.
    private void ConvolutionProducts(Storage x, Storage weight, Storage y, in ConvGeometry g, int filters, int groups, Scratch scratch)
    {
        int positions = g.Positions, outputs = g.OH * g.OW, patch = g.PatchSize / groups, perGroup = filters / groups;
        var cols = scratch.Take(positions * g.PatchSize);
        Im2Col(x, cols, in g);
        var grouped = cols;
        if (groups > 1)
        {
            grouped = scratch.Take(positions * g.PatchSize);
            Permute(cols, grouped, [groups, positions, patch], [patch, groups * patch, 1], false);
        }

        var rows = scratch.Take(positions * filters);
        BatchedMatMul(grouped, weight, rows, groups, positions, perGroup, patch, false, true, 0f);

        // rows [G, N, OH·OW, Fg] → y [N, G, Fg, OH·OW].
        Permute(rows, y, [g.N, groups, perGroup, outputs], [outputs * perGroup, positions * perGroup, 1, perGroup], false);
    }

    /// <summary>The input gradient as products and folded patches (the default of <see cref="ConvolutionBackwardInputKernel"/>).</summary>
    protected void ComposedConvolutionBackwardInput(Storage dy, Storage weight, Storage dx, in ConvGeometry g, int filters, int groups)
    {
        int outputs = g.OH * g.OW;
        if (g.N <= 0 || filters <= 0 || outputs <= 0)
        {
            return;
        }

        int chunk = ConvolutionChunk(in g), inImage = g.C * g.H * g.W, outImage = filters * outputs;
        for (int first = 0; first < g.N; first += chunk)
        {
            int images = Math.Min(chunk, g.N - first);
            var part = g with { N = images };
            var scratch = new Scratch(this);
            try
            {
                Storage gradient = dy, input = dx;
                if (images != g.N)
                {
                    gradient = scratch.Take(images * outImage);
                    Copy2D(dy, first * outImage, 0, gradient, 0, 0, 1, images * outImage, false);
                    input = scratch.Take(images * inImage, zeroed: true);
                }

                int positions = part.Positions, patch = part.PatchSize / groups, perGroup = filters / groups;
                var rows = scratch.Take(positions * filters);
                Permute(gradient, rows, [groups, images, outputs, perGroup], [perGroup * outputs, groups * perGroup * outputs, 1, outputs], false);
                var cols = scratch.Take(positions * part.PatchSize);
                if (groups == 1)
                {
                    BatchedMatMul(rows, weight, cols, 1, positions, patch, perGroup, false, false, 0f);
                }
                else
                {
                    var grouped = scratch.Take(positions * part.PatchSize);
                    BatchedMatMul(rows, weight, grouped, groups, positions, patch, perGroup, false, false, 0f);
                    Permute(grouped, cols, [positions, groups, patch], [patch, positions * patch, 1], false);
                }

                Col2Im(cols, input, in part);
                if (images != g.N)
                {
                    Copy2D(input, 0, 0, dx, first * inImage, 0, 1, images * inImage, true);
                }
            }
            finally
            {
                scratch.Release();
            }
        }
    }

    /// <summary>The weight gradient as unfolded patches and products (the default of <see cref="ConvolutionBackwardWeightKernel"/>).</summary>
    protected void ComposedConvolutionBackwardWeight(Storage x, Storage dy, Storage dweight, in ConvGeometry g, int filters, int groups)
    {
        int outputs = g.OH * g.OW;
        if (g.N <= 0 || filters <= 0 || outputs <= 0)
        {
            return;
        }

        int chunk = ConvolutionChunk(in g), inImage = g.C * g.H * g.W, outImage = filters * outputs;
        for (int first = 0; first < g.N; first += chunk)
        {
            int images = Math.Min(chunk, g.N - first);
            var part = g with { N = images };
            var scratch = new Scratch(this);
            try
            {
                Storage input = x, gradient = dy;
                if (images != g.N)
                {
                    input = scratch.Take(images * inImage);
                    Copy2D(x, first * inImage, 0, input, 0, 0, 1, images * inImage, false);
                    gradient = scratch.Take(images * outImage);
                    Copy2D(dy, first * outImage, 0, gradient, 0, 0, 1, images * outImage, false);
                }

                int positions = part.Positions, patch = part.PatchSize / groups, perGroup = filters / groups;
                var cols = scratch.Take(positions * part.PatchSize);
                Im2Col(input, cols, in part);
                var grouped = cols;
                if (groups > 1)
                {
                    grouped = scratch.Take(positions * part.PatchSize);
                    Permute(cols, grouped, [groups, positions, patch], [patch, groups * patch, 1], false);
                }

                // The gradient as [G, Fg, P] (each filter's positions in a row), so the product reads both operands along
                // the sum: dweight [G, Fg, pg] += gradient [G, Fg, P] · patches [G, P, pg].
                var rows = scratch.Take(positions * filters);
                Permute(gradient, rows, [groups, perGroup, images, outputs], [perGroup * outputs, outputs, groups * perGroup * outputs, 1], false);
                BatchedMatMul(rows, grouped, dweight, groups, perGroup, patch, positions, false, false, 1f);
            }
            finally
            {
                scratch.Release();
            }
        }
    }
}
