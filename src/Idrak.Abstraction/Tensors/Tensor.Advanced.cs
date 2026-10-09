// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;
using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Diagnostics;

namespace Idrak.Abstraction;

// N-D operations: batched matrix products, softmax, reductions along an axis, permute / narrow / concat,
// and the internal building blocks of the normalization, embedding, convolution and pooling layers.
public sealed partial class Tensor
{    // ---------------------------------------------------------------- element-wise

    /// <summary>eˣ, element-wise.</summary>
    public Tensor Exp() => Unary(UnaryOp.Exp);

    /// <summary>Natural logarithm, element-wise.</summary>
    public Tensor Log() => Unary(UnaryOp.Log);

    /// <summary>Gaussian error linear unit (tanh approximation), the usual transformer activation.</summary>
    public Tensor Gelu() => Unary(UnaryOp.Gelu);

    // ---------------------------------------------------------------- matrix products

    /// <summary>
    /// Matrix product. Supports [m, k] × [k, n] → [m, n]; batched [b, m, k] × [b, k, n] → [b, m, n];
    /// and [..., k] × [k, n] → [..., n] (a weight shared across leading dimensions).
    /// The transpose flags use the last two dimensions of an operand transposed, without copying.
    /// </summary>
    public Tensor MatMul(Tensor other, bool transposeA = false, bool transposeB = false)
    {
        ThrowIfDisposed();
        other.ThrowIfDisposed();
        CheckSameDevice(this, other);
        if (Rank == 2 && other.Rank == 2)
        {
            return MatMulCore(this, other, 1, transposeA, transposeB);
        }

        if (Rank == 3 && other.Rank == 3)
        {
            if (_shape[0] != other._shape[0])
            {
                throw new ArgumentException($"Batched MatMul needs equal batch sizes, got {FormatShape(_shape)} and {FormatShape(other._shape)}.");
            }

            return MatMulCore(this, other, _shape[0], transposeA, transposeB);
        }

        if (Rank > 2 && other.Rank == 2 && !transposeA)
        {
            int k = _shape[^1];
            var flat = Reshape(-1, k);
            var product = MatMulCore(flat, other, 1, false, transposeB);
            int[] outShape = [.. _shape[..^1], product._shape[1]];
            return product.Reshape(outShape);
        }

        throw new ArgumentException($"MatMul does not support shapes {FormatShape(_shape)} and {FormatShape(other._shape)}.");
    }

    /// <summary>
    /// x·w + bias for x [..., k], w [k, n], bias [n]: one pass where the device has one (bfloat16 tensor cores),
    /// otherwise a product and a bias addition. Same gradients as <c>x.MatMul(w) + bias</c>.
    /// </summary>
    public static Tensor MatMulBias(Tensor x, Tensor w, Tensor bias)
    {
        int k = x._shape[^1], n = w._shape[1], m = x.Size / Math.Max(1, k);
        if (w.Rank != 2 || w._shape[0] != k || bias.Size != n)
        {
            return x.MatMul(w) + bias;
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty([.. x._shape[..^1], n], x.Device);
        if (!x.Backend.MatMulBias(x.Storage, w.Storage, bias.Storage, y.Storage, m, n, k))
        {
            y.Dispose();
            return x.MatMul(w) + bias;
        }

        if ((x.RequiresGrad || w.RequiresGrad || bias.RequiresGrad) && Autograd.IsEnabled)
        {
            y.Record("matmul_bias", g =>
            {
                var backend = x.Backend;
                if (bias.RequiresGrad)
                {
                    backend.SumRows(g.Storage, bias.GradStorage(), m, n);
                }

                if (x.RequiresGrad)
                {
                    backend.BatchedMatMul(g.Storage, w.Storage, x.GradStorage(), 1, m, k, n, false, true, 1f);
                }

                if (w.RequiresGrad)
                {
                    backend.BatchedMatMul(x.Storage, g.Storage, w.GradStorage(), 1, k, n, m, true, false, 1f);
                }
            }, x, w, bias);
        }

        return Traced("matmul_bias", y, start);
    }

    private static Tensor MatMulCore(Tensor a, Tensor b, int batch, bool transA, bool transB)
    {
        int ra = a._shape[^2], ca = a._shape[^1], rb = b._shape[^2], cb = b._shape[^1];
        int m = transA ? ca : ra, k = transA ? ra : ca;
        int kb = transB ? cb : rb, n = transB ? rb : cb;
        if (k != kb)
        {
            throw new ArgumentException($"MatMul inner dimensions differ: {FormatShape(a._shape)}{(transA ? "ᵀ" : "")} × {FormatShape(b._shape)}{(transB ? "ᵀ" : "")}.");
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var c = Empty(a.Rank == 3 ? [batch, m, n] : [m, n], a.Device);
        a.Backend.BatchedMatMul(a.Storage, b.Storage, c.Storage, batch, m, n, k, transA, transB, 0f);
        if (WillRecord(a, b))
        {
            c.Record("matmul", g =>
            {
                var backend = a.Backend;
                if (a.RequiresGrad)
                {
                    if (!transA)
                    {
                        backend.BatchedMatMul(g.Storage, b.Storage, a.GradStorage(), batch, m, k, n, false, !transB, 1f);
                    }
                    else
                    {
                        backend.BatchedMatMul(b.Storage, g.Storage, a.GradStorage(), batch, k, m, n, transB, true, 1f);
                    }
                }

                if (b.RequiresGrad)
                {
                    if (!transB)
                    {
                        backend.BatchedMatMul(a.Storage, g.Storage, b.GradStorage(), batch, k, n, m, !transA, false, 1f);
                    }
                    else
                    {
                        backend.BatchedMatMul(g.Storage, a.Storage, b.GradStorage(), batch, n, k, m, true, transA, 1f);
                    }
                }
            }, a, b);
        }

        return Traced("matmul", c, start);
    }

    // ---------------------------------------------------------------- softmax and classification

    /// <summary>Softmax over the last dimension: each row becomes a probability distribution.</summary>
    public Tensor Softmax() => SoftmaxCore(log: false);

    /// <summary>log(softmax(x)) over the last dimension, computed stably (used by cross-entropy).</summary>
    public Tensor LogSoftmax() => SoftmaxCore(log: true);

    private Tensor SoftmaxCore(bool log)
    {
        ThrowIfDisposed();
        RequireRank(1, log ? "LogSoftmax" : "Softmax");
        int cols = _shape[^1], rows = cols == 0 ? 0 : Size / cols;
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(_shape, Device);
        Backend.Softmax(Storage, y.Storage, rows, cols, log);
        string name = log ? "log_softmax" : "softmax";
        if (WillRecord(this))
        {
            var x = this;
            y.Record(name, g => x.Backend.SoftmaxBackward(y.Storage, g.Storage, x.GradStorage(), rows, cols, log), x);
        }

        return Traced(name, y, start);
    }

    /// <summary>Index of the largest value along the last dimension (as floats); the result drops that dimension.</summary>
    public Tensor ArgMax()
    {
        ThrowIfDisposed();
        RequireRank(1, "ArgMax");
        int cols = _shape[^1], rows = cols == 0 ? 0 : Size / cols;
        var y = Empty(_shape[..^1], Device);
        Backend.ArgMax(Storage, y.Storage, rows, cols);
        return y;
    }

    /// <summary>
    /// Fraction of rows predicted correctly, as a scalar: argmax match for multi-column outputs, or
    /// (prediction ≥ threshold) == (target ≥ 0.5) for a single column.
    /// </summary>
    public static Tensor MatchRate(Tensor predictions, Tensor targets, float threshold)
    {
        CheckSameShape(predictions, targets);
        int cols = predictions._shape[^1], rows = cols == 0 ? 0 : predictions.Size / cols;
        using var matches = Empty([rows], predictions.Device, track: false);
        predictions.Backend.ClassMatch(predictions.Storage, targets.Storage, matches.Storage, rows, cols, threshold);
        using (Autograd.NoGrad())
        {
            return matches.Mean();
        }
    }

    /// <summary>
    /// One-hot rows for integer-valued class <paramref name="indices"/> of any shape: [...] → [..., classes].
    /// Written on the device in one pass (no identity table, no large target arrays on the host).
    /// </summary>
    public static Tensor OneHot(Tensor indices, int classes)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(classes);
        indices.ThrowIfDisposed();
        var y = Empty([.. indices._shape, classes], indices.Device);
        indices.Backend.OneHot(indices.Storage, y.Storage, indices.Size, classes);
        return y;
    }

    // ---------------------------------------------------------------- reductions along a dimension

    /// <summary>Sums along <paramref name="dim"/> (negative counts from the end); keepDim leaves it as size 1.</summary>
    public Tensor Sum(int dim, bool keepDim = false) => ReduceAxis(dim, keepDim, mean: false);

    /// <summary>Averages along <paramref name="dim"/> (negative counts from the end); keepDim leaves it as size 1.</summary>
    public Tensor Mean(int dim, bool keepDim = false) => ReduceAxis(dim, keepDim, mean: true);

    private Tensor ReduceAxis(int dim, bool keepDim, bool mean)
    {
        ThrowIfDisposed();
        dim = NormalizeDim(dim);
        var (outer, size, inner) = Split(dim);
        float scale = mean && size > 0 ? 1f / size : 1f;
        int[] shape = keepDim ? [.. _shape[..dim], 1, .. _shape[(dim + 1)..]] : [.. _shape[..dim], .. _shape[(dim + 1)..]];
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(shape, Device);
        Backend.SumAxis(Storage, y.Storage, outer, size, inner, scale, accumulate: false);
        if (WillRecord(this))
        {
            var x = this;
            y.Record("sum_axis", g => x.Backend.BroadcastAxis(g.Storage, x.GradStorage(), outer, size, inner, scale), x);
        }

        return Traced("sum_axis", y, start);
    }

    // ---------------------------------------------------------------- shape operations

    /// <summary>Reorders dimensions: output dimension k is input dimension <paramref name="dims"/>[k]. Copies the data.</summary>
    public Tensor Permute(params ReadOnlySpan<int> dims)
    {
        ThrowIfDisposed();
        if (dims.Length != Rank)
        {
            throw new ArgumentException($"Permute needs {Rank} dimensions, got {dims.Length}.");
        }

        int[] perm = [.. dims];
        var seen = new bool[Rank];
        for (int k = 0; k < Rank; k++)
        {
            perm[k] = NormalizeDim(perm[k]);
            if (seen[perm[k]])
            {
                throw new ArgumentException($"Permute dimensions must be distinct: [{string.Join(", ", dims.ToArray())}].");
            }

            seen[perm[k]] = true;
        }

        var strides = Strides(_shape);
        int[] outShape = [.. perm.Select(p => _shape[p])];

        // Moving only size-1 dimensions keeps the memory order: a reshape (a view) instead of a copy. Common when
        // decoding one token at a time, where the step and batch dimensions have size 1.
        int previous = -1;
        bool sameOrder = true;
        foreach (int p in perm)
        {
            if (_shape[p] != 1)
            {
                sameOrder &= p > previous;
                previous = p;
            }
        }

        if (sameOrder)
        {
            return Reshape(outShape);
        }

        int[] inStrides = [.. perm.Select(p => strides[p])];
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(outShape, Device);
        Backend.Permute(Storage, y.Storage, outShape, inStrides, accumulate: false);
        if (WillRecord(this))
        {
            var x = this;
            var yStrides = Strides(outShape);
            var inverse = new int[Rank];
            for (int k = 0; k < Rank; k++)
            {
                inverse[perm[k]] = k;
            }

            int[] backStrides = [.. inverse.Select(k => yStrides[k])];
            y.Record("permute", g => x.Backend.Permute(g.Storage, x.GradStorage(), x._shape, backStrides, accumulate: true), x);
        }

        return Traced("permute", y, start);
    }

    /// <summary>Swaps two dimensions (copies the data).</summary>
    public Tensor Transpose(int dim0 = -2, int dim1 = -1)
    {
        int[] perm = [.. Enumerable.Range(0, Rank)];
        (perm[NormalizeDim(dim0)], perm[NormalizeDim(dim1)]) = (perm[NormalizeDim(dim1)], perm[NormalizeDim(dim0)]);
        return Permute(perm);
    }

    /// <summary>Merges dimensions from <paramref name="startDim"/> onward into one, e.g. [N, C, H, W] → [N, C·H·W].</summary>
    public Tensor Flatten(int startDim = 1)
    {
        startDim = NormalizeDim(startDim);
        return Reshape([.. _shape[..startDim], -1]);
    }

    /// <summary>The slice [start, start + length) along <paramref name="dim"/> (copied).</summary>
    public Tensor Narrow(int dim, int start, int length)
    {
        ThrowIfDisposed();
        dim = NormalizeDim(dim);
        if (start < 0 || length < 0 || start + length > _shape[dim])
        {
            throw new ArgumentOutOfRangeException(nameof(start), $"Narrow [{start}, {start + length}) is outside dimension {dim} of size {_shape[dim]}.");
        }

        var (outer, size, inner) = Split(dim);
        int[] shape = [.. _shape];
        shape[dim] = length;
        long t0 = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(shape, Device);
        Backend.Copy2D(Storage, start * inner, size * inner, y.Storage, 0, length * inner, outer, length * inner, accumulate: false);
        if (WillRecord(this))
        {
            var x = this;
            y.Record("narrow", g => x.Backend.Copy2D(g.Storage, 0, length * inner, x.GradStorage(), start * inner, size * inner, outer, length * inner, accumulate: true), x);
        }

        return Traced("narrow", y, t0);
    }

    /// <summary>Joins tensors along an existing dimension; all other dimensions must match.</summary>
    public static Tensor Concat(IReadOnlyList<Tensor> tensors, int dim = 0)
    {
        if (tensors.Count == 0)
        {
            throw new ArgumentException("Concat needs at least one tensor.", nameof(tensors));
        }

        var first = tensors[0];
        dim = first.NormalizeDim(dim);
        int total = 0;
        foreach (var t in tensors)
        {
            t.ThrowIfDisposed();
            CheckSameDevice(first, t);
            if (t.Rank != first.Rank || !t.Shape[..dim].SequenceEqual(first.Shape[..dim]) || !t.Shape[(dim + 1)..].SequenceEqual(first.Shape[(dim + 1)..]))
            {
                throw new ArgumentException($"Concat along {dim}: shape {FormatShape(t._shape)} does not match {FormatShape(first._shape)}.");
            }

            total += t._shape[dim];
        }

        var (outer, _, inner) = first.Split(dim);
        int[] shape = [.. first._shape];
        shape[dim] = total;
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(shape, first.Device);
        int offset = 0;
        var offsets = new int[tensors.Count];
        for (int i = 0; i < tensors.Count; i++)
        {
            var t = tensors[i];
            offsets[i] = offset;
            int width = t._shape[dim] * inner;
            first.Backend.Copy2D(t.Storage, 0, width, y.Storage, offset * inner, total * inner, outer, width, accumulate: false);
            offset += t._shape[dim];
        }

        if (Autograd.IsEnabled && tensors.Any(t => t.RequiresGrad))
        {
            var inputs = tensors.ToArray();
            y.Record("concat", g =>
            {
                for (int i = 0; i < inputs.Length; i++)
                {
                    var t = inputs[i];
                    if (t.RequiresGrad)
                    {
                        int width = t._shape[dim] * inner;
                        t.Backend.Copy2D(g.Storage, offsets[i] * inner, total * inner, t.GradStorage(), 0, width, outer, width, accumulate: true);
                    }
                }
            }, inputs);
        }

        return Traced("concat", y, start);
    }

    /// <summary>Stacks equally shaped tensors along a new dimension, e.g. T × [N, H] → [N, T, H] with dim 1.</summary>
    public static Tensor Stack(IReadOnlyList<Tensor> tensors, int dim = 0)
    {
        if (tensors.Count == 0)
        {
            throw new ArgumentException("Stack needs at least one tensor.", nameof(tensors));
        }

        int rank = tensors[0].Rank + 1;
        if (dim < 0)
        {
            dim += rank;
        }

        var expanded = tensors.Select(t => t.Reshape([.. t._shape[..dim], 1, .. t._shape[dim..]])).ToList();
        return Concat(expanded, dim);
    }

    // ---------------------------------------------------------------- layer building blocks (internal)

    /// <summary>
    /// Standardizes each group of a [outer, groups, inner] view to zero mean and unit variance using batch
    /// statistics, returning the batch mean and (biased) variance for running-statistics updates.
    /// </summary>
    public Tensor Normalize(int outer, int groups, int inner, float eps, out Tensor mean, out Tensor variance)
    {
        ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        mean = Empty([groups], Device);
        variance = Empty([groups], Device);
        var invStd = Empty([groups], Device, track: false);
        Backend.NormStats(Storage, mean.Storage, variance.Storage, invStd.Storage, outer, groups, inner, eps);
        var y = Empty(_shape, Device);
        Backend.NormApply(Storage, mean.Storage, invStd.Storage, y.Storage, outer, groups, inner);
        if (WillRecord(this))
        {
            var x = this;
            y.Record("normalize", g =>
            {
                using var sum1 = Empty([groups], x.Device, zeroed: true, track: false);
                using var sum2 = Empty([groups], x.Device, zeroed: true, track: false);
                x.Backend.GroupReduce(g.Storage, y.Storage, sum1.Storage, sum2.Storage, outer, groups, inner);
                x.Backend.NormBackward(g.Storage, y.Storage, sum1.Storage, sum2.Storage, invStd.Storage, x.GradStorage(), outer, groups, inner);
                invStd.Dispose();
            }, x);
        }
        else
        {
            invStd.Dispose();
        }

        return Traced("normalize", y, start);
    }

    /// <summary>Standardizes with fixed statistics (evaluation mode): (x - mean[g]) * invStd[g].</summary>
    public Tensor NormalizeWith(Tensor mean, Tensor invStd, int outer, int groups, int inner)
    {
        ThrowIfDisposed();
        var y = Empty(_shape, Device);
        Backend.NormApply(Storage, mean.Storage, invStd.Storage, y.Storage, outer, groups, inner);
        if (WillRecord(this))
        {
            var x = this;
            y.Record("normalize", g => x.Backend.GroupScaleShift(g.Storage, invStd.Storage, null, x.GradStorage(), x.Size, groups, inner, accumulate: true), x);
        }

        return y;
    }

    /// <summary>y = x * scale[g] + shift[g] with g = (i / inner) % groups (per-channel or per-feature affine).</summary>
    public Tensor GroupAffine(Tensor? scale, Tensor? shift, int groups, int inner)
    {
        ThrowIfDisposed();
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(_shape, Device);
        Backend.GroupScaleShift(Storage, scale?.Storage, shift?.Storage, y.Storage, Size, groups, inner, accumulate: false);
        var x = this;
        bool record = Autograd.IsEnabled && (x.RequiresGrad || scale?.RequiresGrad == true || shift?.RequiresGrad == true);
        if (record)
        {
            int outer = Size / (groups * inner);
            Tensor[] inputs = [x, .. new[] { scale, shift }.OfType<Tensor>()];
            y.Record("group_affine", g =>
            {
                var backend = x.Backend;
                if (x.RequiresGrad)
                {
                    backend.GroupScaleShift(g.Storage, scale?.Storage, null, x.GradStorage(), x.Size, groups, inner, accumulate: true);
                }

                bool needShift = shift?.RequiresGrad == true, needScale = scale?.RequiresGrad == true;
                if (needShift || needScale)
                {
                    using var scratch = needShift ? null : Empty([groups], x.Device, zeroed: true, track: false);
                    var sumG = needShift ? shift!.GradStorage() : scratch!.Storage;
                    backend.GroupReduce(g.Storage, needScale ? x.Storage : null, sumG, needScale ? scale!.GradStorage() : null, outer, groups, inner);
                }
            }, inputs);
        }

        return Traced("group_affine", y, start);
    }

    /// <summary>Embedding lookup: rows of this [vocabulary, dim] table selected by integer-valued <paramref name="indices"/>.</summary>
    public Tensor EmbeddingLookup(Tensor indices)
    {
        ThrowIfDisposed();
        indices.ThrowIfDisposed();
        CheckSameDevice(this, indices);
        int vocabulary = _shape[0], dim = _shape[1];
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty([.. indices._shape, dim], Device);
        Backend.Gather(Storage, indices.Storage, y.Storage, indices.Size, dim, vocabulary);
        if (WillRecord(this))
        {
            var table = this;
            y.Record("embedding", g => table.Backend.ScatterAdd(g.Storage, indices.Storage, table.GradStorage(), indices.Size, dim, vocabulary), table);
        }

        return Traced("embedding", y, start);
    }

    /// <summary>Unfolds [N, C, H, W] into [N·OH·OW, C·KH·KW] patch rows for convolution as a matrix product.</summary>
    public Tensor Im2Col(in ConvGeometry geometry)
    {
        ThrowIfDisposed();
        var g0 = geometry;
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty([g0.Positions, g0.PatchSize], Device);
        Backend.Im2Col(Storage, y.Storage, g0);
        if (WillRecord(this))
        {
            var x = this;
            y.Record("im2col", g => x.Backend.Col2Im(g.Storage, x.GradStorage(), g0), x);
        }

        return Traced("im2col", y, start);
    }

    /// <summary>2-D max pooling of [N, C, H, W].</summary>
    public Tensor MaxPool(in ConvGeometry geometry)
    {
        ThrowIfDisposed();
        var g0 = geometry;
        if (g0.Dilated)
        {
            throw new ArgumentException("Max pooling takes no dilation.", nameof(geometry));
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty([g0.N, g0.C, g0.OH, g0.OW], Device);
        var argmax = Empty([y.Size], Device, track: false);
        Backend.MaxPool(Storage, y.Storage, argmax.Storage, g0);
        if (WillRecord(this))
        {
            var x = this;
            y.Record("maxpool", g =>
            {
                x.Backend.MaxPoolBackward(g.Storage, argmax.Storage, x.GradStorage(), g0);
                argmax.Dispose();   // backward runs once; recycle the indices immediately
            }, x);
        }
        else
        {
            argmax.Dispose();
        }

        return Traced("maxpool", y, start);
    }

    /// <summary>
    /// Folds [N·OH·OW, C·KH·KW] patch rows back into [N, C, H, W], adding where windows overlap: the adjoint of
    /// <see cref="Im2Col"/>, and with a matrix product a transposed convolution (each input position's products spread over
    /// the window it scatters to). Its gradient unfolds the incoming gradient with <see cref="Im2Col"/>.
    /// </summary>
    public Tensor Col2Im(in ConvGeometry geometry)
    {
        ThrowIfDisposed();
        var g0 = geometry;
        if (Rank != 2 || _shape[0] != g0.Positions || _shape[1] != g0.PatchSize)
        {
            throw new ArgumentException($"Folding a {g0.KH}x{g0.KW} window over {g0.N} images of {g0.H}x{g0.W} takes [{g0.Positions}, {g0.PatchSize}] patch rows, not {FormatShape(_shape)}.");
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty([g0.N, g0.C, g0.H, g0.W], Device, zeroed: true);
        Backend.Col2Im(Storage, y.Storage, g0);
        if (WillRecord(this))
        {
            var x = this;
            y.Record("col2im", g =>
            {
                // A first gradient is written in place; a later one goes through a scratch matrix and is added.
                if (x.Grad is null)
                {
                    x.Grad = Empty(x._shape, x.Device, track: false);
                    x.Backend.Im2Col(g.Storage, x.Grad.Storage, g0);
                    return;
                }

                using var columns = Empty(x._shape, x.Device, track: false);
                x.Backend.Im2Col(g.Storage, columns.Storage, g0);
                x.Backend.Axpy(columns.Storage, x.Grad.Storage, x.Size, 1f);
            }, x);
        }

        return Traced("col2im", y, start);
    }

    /// <summary>
    /// Resamples the last two dimensions (height and width) to <paramref name="size"/> as PyTorch's <c>F.interpolate</c>:
    /// nearest (the input position floor(o · scale)) or bilinear, with or without <paramref name="alignCorners"/>.
    /// </summary>
    /// <param name="size">The output height and width.</param>
    /// <param name="mode">Nearest or bilinear.</param>
    /// <param name="alignCorners">Bilinear only: the corner positions of input and output coincide (scale (in - 1) / (out - 1)).</param>
    /// <param name="scaleFactor">The scale factor the size came from, if any: as PyTorch, it then sets the coordinate scale
    /// (1 / factor) instead of input / output size, which differ when the factor times the input is not whole.</param>
    public Tensor Interpolate((int Height, int Width) size, InterpolationMode mode = InterpolationMode.Nearest, bool alignCorners = false,
        (double Height, double Width)? scaleFactor = null)
    {
        ThrowIfDisposed();
        RequireRank(2, "Interpolation");
        if (size.Height <= 0 || size.Width <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(size), size, "An interpolated size is positive.");
        }

        if (alignCorners && mode == InterpolationMode.Nearest)
        {
            throw new ArgumentException("alignCorners applies to bilinear interpolation, not nearest (as PyTorch).", nameof(alignCorners));
        }

        int height = _shape[^2], width = _shape[^1], planes = Size / Math.Max(height * width, 1);
        var (oh, ow) = size;
        float sh = CoordinateScale(height, oh, alignCorners, scaleFactor?.Height), sw = CoordinateScale(width, ow, alignCorners, scaleFactor?.Width);
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty([.. _shape[..^2], oh, ow], Device);
        Backend.Interpolate2d(Storage, y.Storage, planes, height, width, oh, ow, mode, alignCorners, sh, sw);
        if (WillRecord(this))
        {
            var x = this;
            y.Record("interpolate", g => x.Backend.Interpolate2dBackward(g.Storage, x.GradStorage(), planes, height, width, oh, ow, mode, alignCorners, sh, sw), x);
        }

        return Traced("interpolate", y, start);
    }

    // Input positions per output position, as PyTorch computes it in float: (in - 1) / (out - 1) with aligned corners (0 for
    // one output), else 1 / the scale factor when one is given, else in / out.
    private static float CoordinateScale(int input, int output, bool alignCorners, double? factor) =>
        alignCorners ? (output > 1 ? (input - 1) / (float)(output - 1) : 0f)
        : factor is > 0 and var f ? (float)(1.0 / f) : input / (float)output;

    /// <summary>
    /// Adaptive average pooling of the last two dimensions to <paramref name="size"/> (PyTorch's <c>AdaptiveAvgPool2d</c>):
    /// output row o averages input rows floor(o · H / OH) to ceil((o + 1) · H / OH), exclusive; columns likewise.
    /// </summary>
    public Tensor AdaptiveAvgPool((int Height, int Width) size) => AdaptivePool(size, max: false);

    /// <summary>Adaptive max pooling of the last two dimensions to <paramref name="size"/>, over the windows of <see cref="AdaptiveAvgPool"/>.</summary>
    public Tensor AdaptiveMaxPool((int Height, int Width) size) => AdaptivePool(size, max: true);

    private Tensor AdaptivePool((int Height, int Width) size, bool max)
    {
        ThrowIfDisposed();
        RequireRank(2, "Adaptive pooling");
        var (oh, ow) = size;
        if (oh <= 0 || ow <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(size), size, "An adaptive pooling size is positive.");
        }

        int height = _shape[^2], width = _shape[^1], planes = Size / Math.Max(height * width, 1);
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty([.. _shape[..^2], oh, ow], Device);
        if (!max)
        {
            Backend.AdaptiveAvgPool(Storage, y.Storage, planes, height, width, oh, ow);
            if (WillRecord(this))
            {
                var x = this;
                y.Record("adaptive_avg_pool", g => x.Backend.AdaptiveAvgPoolBackward(g.Storage, x.GradStorage(), planes, height, width, oh, ow), x);
            }

            return Traced("adaptive_avg_pool", y, start);
        }

        var argmax = Empty([y.Size], Device, track: false);
        Backend.AdaptiveMaxPool(Storage, y.Storage, argmax.Storage, planes, height, width, oh, ow);
        if (WillRecord(this))
        {
            var x = this;
            y.Record("adaptive_max_pool", g =>
            {
                x.Backend.AdaptiveMaxPoolBackward(g.Storage, argmax.Storage, x.GradStorage(), planes, height, width, oh, ow);
                argmax.Dispose();
            }, x);
        }
        else
        {
            argmax.Dispose();
        }

        return Traced("adaptive_max_pool", y, start);
    }

    /// <summary>
    /// The connectionist temporal classification loss of each sequence (Graves et al. 2006), [batch]: -log of the
    /// probability, summed over every alignment, that the sequence reads as its labels once repeats are collapsed and blanks
    /// dropped. This tensor holds log-probabilities (a log-softmax over the classes), [steps, batch, classes] or, with
    /// <paramref name="batchFirst"/>, [batch, steps, classes]. Computed in log space on the device, parallel over the
    /// sequences; the loss keeps O(labels) values a sequence and its gradient O(steps · labels), never a copy of the input.
    /// </summary>
    /// <param name="targets">The labels as whole numbers: [batch, maxLabels] padded (each row's first targetLengths[n]
    /// count), or every sequence's labels one after another, [Σ targetLengths]. The blank is not a label.</param>
    /// <param name="inputLengths">The steps of each sequence that count (at most steps).</param>
    /// <param name="targetLengths">The labels of each sequence.</param>
    /// <param name="blank">The blank class.</param>
    /// <param name="batchFirst">Whether this tensor is [batch, steps, classes] rather than [steps, batch, classes].</param>
    /// <param name="zeroInfinity">A sequence no alignment fits (fewer steps than it needs) has loss and gradient 0 rather
    /// than an infinite loss.</param>
    /// <remarks>
    /// The gradient is the derivative of the loss with respect to these log-probabilities: -Σ over the alignments through
    /// class c at step t of their share of the probability. Behind a log-softmax it gives the same gradient on the scores as
    /// PyTorch's <c>ctc_loss</c>, which returns its gradient with respect to the scores directly.
    /// </remarks>
    public Tensor CtcLoss(Tensor targets, IReadOnlyList<int> inputLengths, IReadOnlyList<int> targetLengths, int blank = 0, bool batchFirst = false, bool zeroInfinity = false)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(targets);
        ArgumentNullException.ThrowIfNull(inputLengths);
        ArgumentNullException.ThrowIfNull(targetLengths);
        targets.ThrowIfDisposed();
        CheckSameDevice(this, targets);
        if (Rank != 3)
        {
            throw new ArgumentException($"CTC takes log-probabilities [{(batchFirst ? "batch, steps" : "steps, batch")}, classes], not {FormatShape(_shape)}.");
        }

        int steps = _shape[batchFirst ? 1 : 0], batch = _shape[batchFirst ? 0 : 1], classes = _shape[2];
        if (inputLengths.Count != batch || targetLengths.Count != batch)
        {
            throw new ArgumentException($"CTC needs an input length and a target length for each of the {batch} sequences; got {inputLengths.Count} and {targetLengths.Count}.");
        }

        if ((uint)blank >= (uint)classes)
        {
            throw new ArgumentOutOfRangeException(nameof(blank), blank, $"The blank is one of the {classes} classes.");
        }

        int[] inputs = [.. inputLengths], lengths = [.. targetLengths], offsets = new int[batch];
        bool padded = targets.Rank == 2;
        if (!padded && targets.Rank != 1 || padded && targets._shape[0] != batch)
        {
            throw new ArgumentException($"CTC targets are [batch, maxLabels] padded or [Σ targetLengths] one after another, not {FormatShape(targets._shape)}.");
        }

        for (int n = 0, next = 0; n < batch; n++)
        {
            if ((uint)inputs[n] > (uint)steps || lengths[n] < 0 || padded && lengths[n] > targets._shape[1])
            {
                throw new ArgumentException($"Sequence {n} has input length {inputs[n]} (of {steps} steps) and target length {lengths[n]}"
                    + (padded ? $" (of {targets._shape[1]} padded labels)." : "."));
            }

            offsets[n] = padded ? n * targets._shape[1] : next;
            next += lengths[n];
            if (!padded && next > targets.Size)
            {
                throw new ArgumentException($"The target lengths add up to more than the {targets.Size} targets given.");
            }
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty([batch], Device);
        Backend.CtcLoss(Storage, targets.Storage, y.Storage, inputs, lengths, offsets, steps, batch, classes, blank, batchFirst, zeroInfinity);
        if (WillRecord(this))
        {
            var x = this;
            y.Record("ctc_loss", g => x.Backend.CtcLossBackward(x.Storage, targets.Storage, g.Storage, x.GradStorage(), inputs, lengths, offsets, steps, batch, classes, blank,
                batchFirst, zeroInfinity), x);
        }

        return Traced("ctc_loss", y, start);
    }

    // ---------------------------------------------------------------- helpers

    internal int NormalizeDim(int dim)
    {
        int d = dim < 0 ? dim + Rank : dim;
        if ((uint)d >= (uint)Rank)
        {
            throw new ArgumentOutOfRangeException(nameof(dim), dim, $"Dimension out of range for shape {FormatShape(_shape)}.");
        }

        return d;
    }

    /// <summary>Views the shape as [outer, shape[dim], inner].</summary>
    internal (int Outer, int Size, int Inner) Split(int dim)
    {
        int outer = 1, inner = 1;
        for (int i = 0; i < dim; i++)
        {
            outer *= _shape[i];
        }

        for (int i = dim + 1; i < Rank; i++)
        {
            inner *= _shape[i];
        }

        return (outer, _shape[dim], inner);
    }

    private static int[] Strides(ReadOnlySpan<int> shape)
    {
        var strides = new int[shape.Length];
        int stride = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride *= shape[i];
        }

        return strides;
    }

    private void RequireRank(int minimum, string operation)
    {
        if (Rank < minimum)
        {
            throw new InvalidOperationException($"{operation} needs at least {minimum} dimension(s), got {FormatShape(_shape)}.");
        }
    }

    // ---------------------------------------------------------------- other number types

    /// <summary>
    /// Creates a tensor from any numeric type (double, int, byte, Half, decimal, ...). Values are converted
    /// to float32, the type every kernel computes in.
    /// </summary>
    public static Tensor From<T>(ReadOnlySpan<T> values, ReadOnlySpan<int> shape, Device? device = null, bool requiresGrad = false)
        where T : INumberBase<T>
    {
        var converted = GC.AllocateUninitializedArray<float>(values.Length);
        for (int i = 0; i < values.Length; i++)
        {
            converted[i] = float.CreateSaturating(values[i]);
        }

        return From(converted, shape, device, requiresGrad);
    }

    /// <summary>Creates a 2-D tensor from a rectangular array of any numeric type.</summary>
    public static Tensor From<T>(T[,] values, Device? device = null, bool requiresGrad = false)
        where T : INumberBase<T>
    {
        var flat = new T[values.Length];
        int i = 0;
        foreach (var v in values)
        {
            flat[i++] = v;
        }

        return From<T>(flat, [values.GetLength(0), values.GetLength(1)], device, requiresGrad);
    }

    /// <summary>Copies the elements into an array of any numeric type (saturating on overflow; truncating toward zero for integers).</summary>
    public T[] ToArray<T>()
        where T : INumberBase<T>
    {
        var values = ToArray();
        var result = new T[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            result[i] = T.CreateSaturating(values[i]);
        }

        return result;
    }
}
