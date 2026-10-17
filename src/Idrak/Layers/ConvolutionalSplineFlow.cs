// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models;

namespace Idrak.Layers;

/// <summary>
/// A coupling flow with a rational-quadratic spline (transformers' <c>VitsConvFlow</c>): the first half of the channels,
/// through a pointwise convolution, a <see cref="DilatedDepthSeparableConv"/> fed a conditioning tensor and a pointwise
/// projection, gives the spline's widths, heights and derivatives that the second half goes through.
/// <see cref="Forward"/> trains (tensor operations, with the log-determinant); <see cref="Reverse"/> inverts on the host.
/// </summary>
public sealed class ConvolutionalSplineFlow : Module
{
    private readonly int _half, _bins;
    private readonly float _tail, _scale;
    private readonly Conv1d _pre, _proj;
    private readonly DilatedDepthSeparableConv _dds;

    /// <summary>
    /// Over the layers: <paramref name="pre"/> (half → hidden), <paramref name="dds"/>, <paramref name="proj"/> (hidden →
    /// half · (3 bins − 1)); <paramref name="half"/> channels go through the spline of <paramref name="bins"/> bins within
    /// ±<paramref name="tail"/>; the widths and heights are scaled by <paramref name="parameterScale"/> (1 / √hidden in VITS).
    /// </summary>
    public ConvolutionalSplineFlow(Conv1d pre, DilatedDepthSeparableConv dds, Conv1d proj, int half, int bins, float tail, float parameterScale)
    {
        _pre = pre ?? throw new ArgumentNullException(nameof(pre));
        _dds = dds ?? throw new ArgumentNullException(nameof(dds));
        _proj = proj ?? throw new ArgumentNullException(nameof(proj));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(half);
        ArgumentOutOfRangeException.ThrowIfLessThan(bins, 2);
        _half = half;
        _bins = bins;
        _tail = tail;
        _scale = parameterScale;
    }

    /// <summary>The channels each half holds.</summary>
    public int Half => _half;

    /// <summary>The spline's bins.</summary>
    public int Bins => _bins;

    /// <summary>
    /// From stored weights under <paramref name="prefix"/>: <c>conv_pre</c>, <c>conv_dds</c> (a
    /// <see cref="DilatedDepthSeparableConv"/> of <paramref name="ddsLayers"/> with <paramref name="ddsKernel"/> taps) and
    /// <c>conv_proj</c>, transformers' names.
    /// </summary>
    public static ConvolutionalSplineFlow FromStored(StoredLayers stored, string prefix, int half, int hidden, int bins, float tail, int ddsKernel, int ddsLayers, float layerNormEpsilon)
    {
        ArgumentNullException.ThrowIfNull(stored);
        var pre = stored.Conv($"{prefix}.conv_pre", half, hidden, 1, 0);
        var dds = DilatedDepthSeparableConv.FromStored(stored, $"{prefix}.conv_dds", hidden, ddsKernel, ddsLayers, layerNormEpsilon);
        var proj = stored.Conv($"{prefix}.conv_proj", hidden, half * (3 * bins - 1), 1, 0);
        return new ConvolutionalSplineFlow(pre, dds, proj, half, bins, tail, 1f / MathF.Sqrt(hidden));
    }

    /// <summary>
    /// The flow forward (training) on latents [B, 2 · half, T]: the second half through the spline the first one
    /// conditions (with <paramref name="conditioning"/> [B, C, T]), composed from tensor operations so its gradients reach
    /// every input; the log-determinant summed over the positions within the sequences (the batch's mask; null: unpadded).
    /// </summary>
    public Tensor Forward(Tensor latents, Tensor conditioning, PaddedSequences? mask, out Tensor logDet)
    {
        int batch = latents.Shape[0], t = latents.Shape[2], bins = _bins;
        var first = latents.Narrow(1, 0, _half);
        var second = latents.Narrow(1, _half, _half);
        var h = _proj.Forward(_dds.Forward(_pre.Forward(first), conditioning, mask));   // [B, half · (3 bins - 1), T]
        if (mask is not null)
        {
            h = mask.Apply(h);
        }

        // One channel a half (VITS): [B, 1, T] is already position-major, so no permutes; several: positions outermost, then channels.
        var rows = h.Permute(0, 2, 1).Reshape(batch * t * _half, 3 * bins - 1);        // [B·T·half, 3 bins - 1]
        var inputs = _half == 1 ? second.Reshape(batch * t, 1) : second.Permute(0, 2, 1).Reshape(batch * t * _half, 1);
        var (outputs, logAbsDet) = RationalQuadraticSpline.Forward(inputs, rows.Narrow(1, 0, bins) * _scale, rows.Narrow(1, bins, bins) * _scale,
            rows.Narrow(1, 2 * bins, bins - 1), _tail);
        var moved = _half == 1 ? outputs.Reshape(batch, 1, t) : outputs.Reshape(batch, t, _half).Permute(0, 2, 1);
        var determinant = _half == 1 ? logAbsDet.Reshape(batch, 1, t) : logAbsDet.Reshape(batch, t, _half).Permute(0, 2, 1);
        if (mask is not null)
        {
            moved = mask.Apply(moved);
            determinant = mask.Apply(determinant);
        }

        logDet = determinant.Sum();
        return Tensor.Concat([first, moved], 1);
    }

    /// <summary>
    /// latents [channels, T] on the host, overwritten: the second half replaced by the spline's inverse, conditioned on the
    /// first half and <paramref name="conditioning"/> [C, T].
    /// </summary>
    public void Reverse(float[] latents, Tensor conditioning, int channels, int t)
    {
        ArgumentNullException.ThrowIfNull(latents);
        int bins = _bins, perChannel = 3 * bins - 1;
        float[] parameters;
        using (new TensorScope())
        {
            var first = Tensor.From(latents.AsSpan(0, _half * t), [_half, t], conditioning.Device);
            var h = _proj.Forward(_dds.Forward(_pre.Forward(first), conditioning));   // [half · (3 bins - 1), T]
            parameters = h.ToArray();
        }

        double scale = _scale;
        var widths = new double[bins];
        var heights = new double[bins];
        var derivatives = new double[bins + 1];
        for (int c = 0; c < _half; c++)
        {
            for (int n = 0; n < t; n++)
            {
                // hidden.reshape(half, 3 bins - 1, T).permute(0, 2, 1): parameter k of channel c at time n.
                int at = c * perChannel * t + n;
                for (int k = 0; k < bins; k++)
                {
                    widths[k] = parameters[at + k * t] * scale;
                    heights[k] = parameters[at + (bins + k) * t] * scale;
                }

                for (int k = 0; k < bins - 1; k++)
                {
                    derivatives[k + 1] = parameters[at + (2 * bins + k) * t];
                }

                int index = (_half + c) * t + n;
                latents[index] = (float)RationalQuadraticSpline.Inverse(latents[index], widths, heights, derivatives, _tail);
            }
        }
    }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => throw new NotSupportedException("A spline flow takes its latents and conditioning through Forward(latents, conditioning, mask, out logDet).");

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => [_pre, _dds, _proj];

    /// <inheritdoc />
    protected override void MoveTo(Device device) => throw new NotSupportedException("The layers are built on their device.");

    /// <inheritdoc />
    public override void Dispose()
    {
        _pre.Dispose();
        _proj.Dispose();
        _dds.Dispose();
        base.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() => $"ConvolutionalSplineFlow({_half} + {_half} channels, {_bins} bins)";
}
