// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models;

namespace Idrak.Layers;

/// <summary>
/// An element-wise affine flow (transformers' <c>VitsElementwiseAffine</c>): each channel scaled by e^(log scale) and
/// shifted by its translation. <see cref="Forward"/> runs it on a batch [B, C, T] with its log-determinant (Σ log scale over
/// the positions within the sequences); <see cref="Reverse"/> inverts it on the host. The log scale and translation are
/// [C, 1] (parameters when the store trains).
/// </summary>
public sealed class ElementwiseAffineFlow : Module
{
    /// <summary>Over <paramref name="logScale"/> and <paramref name="translate"/>, each of one value per channel.</summary>
    public ElementwiseAffineFlow(Tensor logScale, Tensor translate)
    {
        LogScale = logScale ?? throw new ArgumentNullException(nameof(logScale));
        Translate = translate ?? throw new ArgumentNullException(nameof(translate));
        if (logScale.Size != translate.Size || logScale.Size == 0)
        {
            throw new ArgumentException($"The log scale holds {logScale.Size} channels and the translation {translate.Size}.", nameof(translate));
        }
    }

    /// <summary>The channels.</summary>
    public int Channels => LogScale.Size;

    /// <summary>The log of each channel's scale.</summary>
    public Tensor LogScale { get; }

    /// <summary>Each channel's shift.</summary>
    public Tensor Translate { get; }

    /// <summary>From stored weights under <paramref name="prefix"/>: <c>log_scale</c> and <c>translate</c> of <paramref name="channels"/> values each.</summary>
    /// <exception cref="InvalidDataException">A tensor holds another number of values.</exception>
    public static ElementwiseAffineFlow FromStored(StoredLayers stored, string prefix, int channels)
    {
        ArgumentNullException.ThrowIfNull(stored);
        var logScale = stored.Tensor(StoredLayers.Join(prefix, "log_scale"));
        var translate = stored.Tensor(StoredLayers.Join(prefix, "translate"));
        if (logScale.Size != channels || translate.Size != channels)
        {
            throw new InvalidDataException($"{prefix} has {logScale.Size} channels, the configuration says {channels}.");
        }

        return new ElementwiseAffineFlow(logScale, translate);
    }

    /// <summary>
    /// x [B, C, T] through the affine: translate + e^(log scale) · x per channel (one affine kernel), zeroed over the
    /// batch's padding; <paramref name="logDet"/> is Σ log scale times <paramref name="positions"/> (the positions within the sequences).
    /// </summary>
    public Tensor Forward(Tensor x, PaddedSequences? mask, int positions, out Tensor logDet)
    {
        ArgumentNullException.ThrowIfNull(x);
        int channels = Channels;
        if (x.Rank != 3 || x.Shape[1] != channels)
        {
            throw new ArgumentException($"The affine flow takes [B, {channels}, T], got {Tensor.FormatShape(x.Shape)}.", nameof(x));
        }

        logDet = LogScale.Sum() * positions;
        var y = x.GroupAffine(LogScale.Reshape(channels).Exp(), Translate.Reshape(channels), channels, x.Shape[2]);
        return mask is null ? y : mask.Apply(y);
    }

    /// <summary>latents [channels, t] on the host, overwritten with the inverse: (x − translate) · e^(−log scale) per channel (the weights as they are now).</summary>
    public void Reverse(Span<float> latents, int t)
    {
        int channels = Channels;
        if (latents.Length != channels * t)
        {
            throw new ArgumentException($"{latents.Length} values for [{channels}, {t}].", nameof(latents));
        }

        float[] logScale = LogScale.ToArray(), translate = Translate.ToArray();
        for (int c = 0; c < channels; c++)
        {
            double scale = Math.Exp(-logScale[c]);
            var row = latents.Slice(c * t, t);
            for (int n = 0; n < t; n++)
            {
                row[n] = (float)((row[n] - translate[c]) * scale);
            }
        }
    }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => Forward(input, null, input.Shape[2], out _);

    /// <inheritdoc />
    public override IEnumerable<Tensor> Parameters() => LogScale.RequiresGrad ? [LogScale, Translate] : [];

    /// <inheritdoc />
    protected override void MoveTo(Device device) => throw new NotSupportedException("The layers are built on their device.");

    /// <inheritdoc />
    public override void Dispose()
    {
        LogScale.Dispose();
        Translate.Dispose();
        base.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() => $"ElementwiseAffineFlow({Channels} channels)";
}
