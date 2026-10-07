// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Modules;

/// <summary>
/// A linear layer as its adapter (<see cref="ILinearAdapter"/>) sees it: the sizes, the frozen weight's values and the
/// product with that weight, however the layer stores it (float, packed int8/int4/bfloat16, FP8 for training).
/// </summary>
public interface ILinearLayer
{
    /// <summary>The input features (the weight's rows).</summary>
    int InFeatures { get; }

    /// <summary>The output features (the weight's columns).</summary>
    int OutFeatures { get; }

    /// <summary>The weight's values as floats, [in, out] row-major, on the host (packed weights expanded).</summary>
    float[] WeightValues();

    /// <summary><paramref name="input"/> [..., in] times the layer's own weight, without bias or adapter: [..., out].</summary>
    Tensor BaseProduct(Tensor input);
}

/// <summary>
/// A LoRA adapter: two small trainable matrices A [in, rank] and B [rank, out] whose product, times
/// <see cref="Scale"/> = alpha / rank, is added to a <c>Linear</c> layer's weight. B starts at zero, so adding
/// an adapter does not change the model's outputs until it is trained.
/// </summary>
/// <param name="A">[inFeatures, rank], small random values.</param>
/// <param name="B">[rank, outFeatures], zeros at creation.</param>
/// <param name="Rank">The rank r.</param>
/// <param name="Scale">alpha / r.</param>
public sealed record LoraAdapter(Tensor A, Tensor B, int Rank, float Scale) : ILinearAdapter
{
    /// <inheritdoc />
    public IReadOnlyList<Tensor> Parameters => [A, B];

    /// <inheritdoc />
    public Tensor Forward(ILinearLayer layer, Tensor input, Tensor product) => Tensor.AddLowRank(product, input, A, B, Scale);

    /// <inheritdoc />
    public Tensor Merge(ILinearLayer layer, Tensor weight) => weight + A.MatMul(B) * Scale;

    /// <inheritdoc />
    public ILinearAdapter MoveTo(Func<Tensor, Tensor> move) => this with { A = move(A), B = move(B) };
}

/// <summary>
/// An adapter on a <c>Linear</c> layer (<c>Linear.Adapter</c>): trainable tensors that change the layer's
/// product x·W, and the weight they fold into. <see cref="LoraAdapter"/> and <see cref="DoraAdapter"/> are the built-in
/// ones; another kind (IA3, VeRA …) implements this interface with ordinary tensor operations.
/// </summary>
public interface ILinearAdapter
{
    /// <summary>The adapter's trainable tensors (after the layer's own in <c>Linear.Parameters</c>).</summary>
    IReadOnlyList<Tensor> Parameters { get; }

    /// <summary>
    /// The adapted product for <paramref name="input"/> [..., in], given <paramref name="product"/> = input·W [..., out]
    /// computed with the layer's own weights (a fresh result the adapter may overwrite); the bias is added afterwards.
    /// </summary>
    Tensor Forward(ILinearLayer layer, Tensor input, Tensor product);

    /// <summary>The weight with the adapter folded in, from the layer's float <paramref name="weight"/> [in, out] (called without gradients).</summary>
    Tensor Merge(ILinearLayer layer, Tensor weight);

    /// <summary>The adapter with each of its tensors replaced by <paramref name="move"/>(tensor) (the layer moves to another device).</summary>
    ILinearAdapter MoveTo(Func<Tensor, Tensor> move);
}

/// <summary>
/// A DoRA adapter (weight-decomposed low-rank adaptation; Liu et al. 2024, "DoRA: Weight-Decomposed Low-Rank
/// Adaptation", arXiv:2402.09353, computed as peft's <c>DoraLinearLayer</c>): the weight is split into a magnitude per
/// output and a direction, LoRA adapts the direction and the magnitude trains on its own:
/// <c>W' = m ⊙ (W + s·A·B) / ‖W + s·A·B‖</c>, the norm taken per output column over the inputs. The output is
/// <c>(x·W + s·x·A·B) ⊙ m / ‖W + s·A·B‖ + b</c>. As in the paper (section 4.3) and peft, the norm is treated as a constant
/// in the backward pass (no gradient through it), which saves memory and changes the gradients only slightly. The norm
/// is computed each forward pass without forming W + s·A·B: ‖W‖² (once, the base is frozen) + 2s·Σ_r B ⊙ (Aᵀ·W) +
/// s²·Σ_r B ⊙ (AᵀA·B), so a packed (QLoRA) base is read through its own product, never expanded. The magnitude starts as
/// ‖W‖ and B at zero, so adding the adapter does not change the outputs.
/// </summary>
public sealed class DoraAdapter : ILinearAdapter, IDisposable
{
    private Tensor? _squaredNorms;

    /// <summary>Creates the adapter from its tensors (see the properties).</summary>
    public DoraAdapter(Tensor a, Tensor b, Tensor magnitude, int rank, float scale)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        ArgumentNullException.ThrowIfNull(magnitude);
        if (a.Rank != 2 || b.Rank != 2 || a.Shape[1] != rank || b.Shape[0] != rank || magnitude.Size != b.Shape[1])
        {
            throw new ArgumentException($"DoRA needs A [in, {rank}], B [{rank}, out] and a magnitude [out]; got {Tensor.FormatShape(a.Shape)}, "
                                        + $"{Tensor.FormatShape(b.Shape)} and {Tensor.FormatShape(magnitude.Shape)}.");
        }

        (A, B, Magnitude, Rank, Scale) = (a, b, magnitude, rank, scale);
    }

    /// <summary>[inFeatures, rank], small random values.</summary>
    public Tensor A { get; }

    /// <summary>[rank, outFeatures], zeros at creation.</summary>
    public Tensor B { get; }

    /// <summary>[outFeatures]: the adapted weight's norm per output column (‖W‖ at creation).</summary>
    public Tensor Magnitude { get; }

    /// <summary>The rank r.</summary>
    public int Rank { get; }

    /// <summary>alpha / r (alpha / √r with rank-stabilized scaling).</summary>
    public float Scale { get; }

    /// <inheritdoc />
    public IReadOnlyList<Tensor> Parameters => [A, B, Magnitude];

    /// <inheritdoc />
    public Tensor Forward(ILinearLayer layer, Tensor input, Tensor product)
    {
        var adapted = Tensor.AddLowRank(product, input, A, B, Scale);       // x·W + s·x·A·B
        Tensor inverse;
        using (Autograd.NoGrad())
        {
            inverse = InverseNorms(layer);                                  // 1 / ‖W + s·A·B‖, a constant for the backward pass
        }

        var factor = Magnitude * inverse;                                   // the magnitude's gradient flows through here
        return adapted.GroupAffine(factor, null, layer.OutFeatures, 1);
    }

    /// <inheritdoc />
    public Tensor Merge(ILinearLayer layer, Tensor weight)
    {
        var dense = weight + A.MatMul(B) * Scale;
        var inverse = ((dense * dense).Sum(0).Log() * -0.5f).Exp();
        return dense.GroupAffine(Magnitude * inverse, null, layer.OutFeatures, 1);
    }

    /// <inheritdoc />
    public ILinearAdapter MoveTo(Func<Tensor, Tensor> move)
    {
        _squaredNorms?.Dispose();
        _squaredNorms = null;
        return new DoraAdapter(move(A), move(B), move(Magnitude), Rank, Scale);
    }

    /// <summary>The squared norm of each output column of the layer's frozen weight W (computed on the host).</summary>
    public static float[] SquaredNorms(ILinearLayer layer)
    {
        var w = layer.WeightValues();
        int inputs = layer.InFeatures, outputs = layer.OutFeatures;
        var sums = new double[outputs];
        for (int i = 0; i < inputs; i++)
        {
            for (int o = 0; o < outputs; o++)
            {
                double v = w[i * outputs + o];
                sums[o] += v * v;
            }
        }

        return [.. sums.Select(v => (float)v)];
    }

    // 1 / ‖W + s·A·B‖ per output column, from ‖W‖², Aᵀ·W (the layer's own product of Aᵀ) and AᵀA·B.
    private Tensor InverseNorms(ILinearLayer layer)
    {
        _squaredNorms ??= Tensor.Persistent(SquaredNorms(layer), [layer.OutFeatures], A.Device, requiresGrad: false);
        var transposed = A.Transpose();                                     // [r, in], on the device
        var crossTerms = layer.BaseProduct(transposed);                     // Aᵀ·W [r, out]
        var gram = A.MatMul(A, transposeA: true).MatMul(B);                 // AᵀA·B [r, out]
        var squared = _squaredNorms + (B * crossTerms).Sum(0) * (2f * Scale) + (B * gram).Sum(0) * (Scale * Scale);
        return (squared.Log() * -0.5f).Exp();
    }

    /// <summary>Releases the cached norms of the base weight (the parameters belong to the layer).</summary>
    public void Dispose()
    {
        _squaredNorms?.Dispose();
        _squaredNorms = null;
    }
}
