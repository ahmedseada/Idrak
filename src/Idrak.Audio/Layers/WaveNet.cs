// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;
using Idrak.Models;

namespace Idrak.Audio.Layers;

/// <summary>Pieces the audio layers share.</summary>
internal static class AudioLayerOps
{
    /// <summary>The leaky ReLU for a slope in [0, 1): one kernel, ReLU at 0.</summary>
    public static Tensor LeakyRelu(Tensor x, float slope) => slope == 0 ? x.Relu() : x.LeakyRelu(slope);

    /// <summary>A column [C, 1] repeated over <paramref name="columns"/> columns: [C, columns].</summary>
    public static Tensor ExpandColumns(Tensor column, int columns) => column.MatMul(Tensor.Ones([1, columns], column.Device));
}

/// <summary>
/// A WaveNet of gated residual layers with global conditioning (WaveGlow's and VITS's <c>WN</c>): each layer's dilated
/// convolution to 2C channels (plus the speaker's conditioning), a tanh · sigmoid gate (one kernel), a pointwise
/// convolution to a residual (added to the input) and a skip (summed over the layers; the last layer gives the skip
/// alone). One sequence <c>[C, T]</c>, or a padded batch <c>[B, C, T]</c> whose masks zero the residual and the sum over the
/// padding, as VITS does.
/// </summary>
public sealed class WaveNet : Module
{
    private readonly int _channels;
    private readonly Conv1d[] _in, _resSkip;
    private readonly Conv1d? _cond;

    /// <summary>
    /// Over the layers: <paramref name="inLayers"/> (C → 2C, dilated), <paramref name="resSkipLayers"/> (C → 2C, the last
    /// C → C), and the <paramref name="conditioning"/> convolution (E → 2C · layers) when the speaker conditions it.
    /// </summary>
    public WaveNet(int channels, IReadOnlyList<Conv1d> inLayers, IReadOnlyList<Conv1d> resSkipLayers, Conv1d? conditioning)
    {
        ArgumentNullException.ThrowIfNull(inLayers);
        ArgumentNullException.ThrowIfNull(resSkipLayers);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
        if (inLayers.Count == 0 || resSkipLayers.Count != inLayers.Count)
        {
            throw new ArgumentException($"A WaveNet's layers pair up: {inLayers.Count} in, {resSkipLayers.Count} residual-skip.", nameof(resSkipLayers));
        }

        _channels = channels;
        _in = [.. inLayers];
        _resSkip = [.. resSkipLayers];
        _cond = conditioning;
    }

    /// <summary>The channels.</summary>
    public int Channels => _channels;

    /// <summary>The layers.</summary>
    public int Layers => _in.Length;

    /// <summary>
    /// From stored weights under <paramref name="prefix"/>: <c>in_layers.{i}</c>, <c>res_skip_layers.{i}</c> and
    /// <c>cond_layer</c> (when <paramref name="conditioningSize"/> is above 0), the WaveGlow and VITS names, each a
    /// weight-normed convolution of <paramref name="kernel"/> taps dilated by <paramref name="dilationRate"/>^i.
    /// </summary>
    public static WaveNet FromStored(StoredLayers stored, string prefix, int channels, int layers, int kernel, int dilationRate, int conditioningSize)
    {
        ArgumentNullException.ThrowIfNull(stored);
        var inLayers = new Conv1d[layers];
        var resSkip = new Conv1d[layers];
        for (int i = 0; i < layers; i++)
        {
            int dilation = (int)Math.Pow(dilationRate, i), padding = (kernel * dilation - dilation) / 2;
            inLayers[i] = stored.Conv($"{prefix}.in_layers.{i}", channels, 2 * channels, kernel, padding, dilation);
            int outChannels = i < layers - 1 ? 2 * channels : channels;
            resSkip[i] = stored.Conv($"{prefix}.res_skip_layers.{i}", channels, outChannels, 1, 0);
        }

        var cond = conditioningSize > 0 ? stored.Conv($"{prefix}.cond_layer", conditioningSize, 2 * channels * layers, 1, 0) : null;
        return new WaveNet(channels, inLayers, resSkip, cond);
    }

    /// <summary>x [C, T] with the speaker's embedding [E, 1] (null for none): the skip sum [C, T].</summary>
    public Tensor Forward(Tensor x, Tensor? speaker) => Forward(x.Reshape(1, x.Shape[0], x.Shape[1]), speaker, null).Reshape(x.Shape[0], x.Shape[1]);

    /// <summary>
    /// A batch x [B, C, T] (zero over its padding) with the speaker's embedding [E, 1] (one sequence only) and the batch's
    /// masks: the skip sum [B, C, T], the residual and the sum zeroed over the padding.
    /// </summary>
    public Tensor Forward(Tensor x, Tensor? speaker, PaddedSequences? mask)
    {
        int batch = x.Shape[0], t = x.Shape[2];
        Tensor? conditioning = speaker is not null && _cond is not null ? _cond.Forward(speaker) : null;   // [2C · layers, 1]
        if (conditioning is not null && batch != 1)
        {
            throw new NotSupportedException("A WaveNet takes a speaker's embedding for one sequence at a time, not for a batch.");
        }

        Tensor? output = null;
        for (int i = 0; i < _in.Length; i++)
        {
            var h = _in[i].Forward(x);                                          // [B, 2C, T]
            if (conditioning is not null)
            {
                h = h + AudioLayerOps.ExpandColumns(conditioning.Narrow(0, i * 2 * _channels, 2 * _channels), t).Reshape(1, 2 * _channels, t);
            }

            var acts = Tensor.GatedActivation(h.Narrow(1, 0, _channels), h.Narrow(1, _channels, _channels), FeedForwardActivation.TanhSigmoid);   // one kernel each way
            var resSkip = _resSkip[i].Forward(acts);
            if (i < _in.Length - 1)
            {
                x = x + resSkip.Narrow(1, 0, _channels);
                if (mask is not null)
                {
                    x = mask.Apply(x);
                }

                var skip = resSkip.Narrow(1, _channels, _channels);
                output = output is null ? skip : output + skip;
            }
            else
            {
                output = output is null ? resSkip : output + resSkip;
            }
        }

        return mask is null ? output! : mask.Apply(output!);
    }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => input.Shape.Length == 2 ? Forward(input, null) : Forward(input, null, null);

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => _cond is null ? [.. _in, .. _resSkip] : [.. _in, .. _resSkip, _cond];

    /// <inheritdoc />
    protected override void MoveTo(Device device) => throw new NotSupportedException("The layers are built on their device.");

    /// <inheritdoc />
    public override void Dispose()
    {
        foreach (var c in _in) c.Dispose();
        foreach (var c in _resSkip) c.Dispose();
        _cond?.Dispose();
        base.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() => $"WaveNet({_channels} channels, {_in.Length} layers{(_cond is null ? "" : ", conditioned")})";
}
