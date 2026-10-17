// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models;

namespace Idrak.Layers;

/// <summary>
/// A stack of dilated depth-separable convolutions over time (transformers' <c>VitsDilatedDepthSeparableConv</c>): per
/// layer a depthwise convolution dilated by kernel^i, a layer norm over the channels, GELU, a pointwise convolution, a
/// layer norm, GELU, added to the input. A conditioning tensor is added to the input first. One sequence <c>[C, T]</c>, or
/// a padded batch <c>[B, C, T]</c> whose masks zero the padding before each dilated convolution.
/// </summary>
public sealed class DilatedDepthSeparableConv : Module
{
    private readonly Conv1d[] _dilated, _pointwise;
    private readonly LayerNorm[] _norms1, _norms2;

    /// <summary>Over the layers' convolutions and norms, in layer order (the same count each).</summary>
    public DilatedDepthSeparableConv(IReadOnlyList<Conv1d> dilated, IReadOnlyList<Conv1d> pointwise, IReadOnlyList<LayerNorm> norms1, IReadOnlyList<LayerNorm> norms2)
    {
        ArgumentNullException.ThrowIfNull(dilated);
        if (pointwise.Count != dilated.Count || norms1.Count != dilated.Count || norms2.Count != dilated.Count)
        {
            throw new ArgumentException($"The layers pair up: {dilated.Count} dilated, {pointwise.Count} pointwise, {norms1.Count} and {norms2.Count} norms.", nameof(pointwise));
        }

        _dilated = [.. dilated];
        _pointwise = [.. pointwise];
        _norms1 = [.. norms1];
        _norms2 = [.. norms2];
    }

    /// <summary>The layers.</summary>
    public int Layers => _dilated.Length;

    /// <summary>
    /// From stored weights under <paramref name="prefix"/>: <c>convs_dilated.{i}</c>, <c>convs_pointwise.{i}</c>,
    /// <c>norms_1.{i}</c>, <c>norms_2.{i}</c> (transformers' names), <paramref name="layers"/> of them over
    /// <paramref name="channels"/> with a filter of <paramref name="kernel"/> taps dilated by kernel^i.
    /// </summary>
    public static DilatedDepthSeparableConv FromStored(StoredLayers stored, string prefix, int channels, int kernel, int layers, float layerNormEpsilon)
    {
        ArgumentNullException.ThrowIfNull(stored);
        var dilated = new Conv1d[layers];
        var pointwise = new Conv1d[layers];
        var norms1 = new LayerNorm[layers];
        var norms2 = new LayerNorm[layers];
        for (int i = 0; i < layers; i++)
        {
            int dilation = (int)Math.Pow(kernel, i), padding = (kernel * dilation - dilation) / 2;
            dilated[i] = stored.Conv($"{prefix}.convs_dilated.{i}", channels, channels, kernel, padding, dilation, groups: channels);
            pointwise[i] = stored.Conv($"{prefix}.convs_pointwise.{i}", channels, channels, 1, 0);
            norms1[i] = stored.Norm($"{prefix}.norms_1.{i}", channels, layerNormEpsilon);
            norms2[i] = stored.Norm($"{prefix}.norms_2.{i}", channels, layerNormEpsilon);
        }

        return new DilatedDepthSeparableConv(dilated, pointwise, norms1, norms2);
    }

    /// <summary>x [C, T] (+ conditioning [C, T]) through the layers: [C, T].</summary>
    public Tensor Forward(Tensor x, Tensor? conditioning)
    {
        int c = x.Shape[0], t = x.Shape[1];
        return Forward(x.Reshape(1, c, t), conditioning?.Reshape(1, c, t), null);
    }

    /// <summary>
    /// A batch x [B, C, T] (+ conditioning [B, C, T]) through the layers with the batch's masks (null: one sequence,
    /// unpadded), each dilated convolution reading the padding as zeros: [B, C, T], zero over the padding.
    /// </summary>
    public Tensor Forward(Tensor x, Tensor? conditioning, PaddedSequences? mask)
    {
        bool single = x.Shape.Length == 2;
        if (single)
        {
            return Forward(x, conditioning).Reshape(x.Shape[0], x.Shape[1]);
        }

        if (conditioning is not null)
        {
            x = x + conditioning;
        }

        for (int i = 0; i < _dilated.Length; i++)
        {
            var h = _dilated[i].Forward(mask is null ? x : mask.Apply(x));
            h = ExactGELU.Apply(Norm(_norms1[i], h));
            h = _pointwise[i].Forward(h);
            h = ExactGELU.Apply(Norm(_norms2[i], h));
            x = x + h;
        }

        return mask is null ? x : mask.Apply(x);
    }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => Forward(input, null, null);

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => [.. _dilated, .. _pointwise, .. _norms1, .. _norms2];

    // LayerNorm over the channels of [B, C, T]: the positions as rows.
    private static Tensor Norm(LayerNorm norm, Tensor h) => norm.Forward(h.Permute(0, 2, 1)).Permute(0, 2, 1);

    /// <inheritdoc />
    protected override void MoveTo(Device device) => throw new NotSupportedException("The layers are built on their device.");

    /// <inheritdoc />
    public override void Dispose()
    {
        foreach (var c in _dilated) c.Dispose();
        foreach (var c in _pointwise) c.Dispose();
        foreach (var n in _norms1) n.Dispose();
        foreach (var n in _norms2) n.Dispose();
        base.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() => $"DilatedDepthSeparableConv({_dilated.Length} layers)";
}
