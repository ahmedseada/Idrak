// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;
using Idrak.Models;

namespace Idrak.Audio.Layers;

/// <summary>
/// VITS's posterior encoder: a linear spectrogram [bins, T] through a pointwise convolution, a <see cref="WaveNet"/>
/// conditioned on the speaker, and a pointwise projection to the latents' means and log-stddevs; the latents are the
/// means plus noise of the stddevs times a scale (OpenVoice's tau; training draws with scale 1).
/// </summary>
public sealed class PosteriorEncoder : Module
{
    private readonly int _flowSize;
    private readonly Conv1d _pre, _proj;
    private readonly WaveNet _waveNet;

    /// <summary>Over <paramref name="pre"/> (bins → hidden), the <paramref name="waveNet"/> and <paramref name="proj"/> (hidden → 2 · <paramref name="flowSize"/>).</summary>
    public PosteriorEncoder(Conv1d pre, WaveNet waveNet, Conv1d proj, int flowSize)
    {
        _pre = pre ?? throw new ArgumentNullException(nameof(pre));
        _waveNet = waveNet ?? throw new ArgumentNullException(nameof(waveNet));
        _proj = proj ?? throw new ArgumentNullException(nameof(proj));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(flowSize);
        _flowSize = flowSize;
    }

    /// <summary>The latents' channels.</summary>
    public int FlowSize => _flowSize;

    /// <summary>From stored weights under <paramref name="prefix"/>: <c>conv_pre</c>, <c>wavenet</c> (<see cref="WaveNet.FromStored"/>) and <c>conv_proj</c>.</summary>
    public static PosteriorEncoder FromStored(StoredLayers stored, string prefix, int bins, int hidden, int flowSize, int waveNetLayers, int kernel, int dilationRate, int conditioningSize)
    {
        ArgumentNullException.ThrowIfNull(stored);
        var pre = stored.Conv($"{prefix}.conv_pre", bins, hidden, 1, 0);
        var waveNet = WaveNet.FromStored(stored, $"{prefix}.wavenet", hidden, waveNetLayers, kernel, dilationRate, conditioningSize);
        var proj = stored.Conv($"{prefix}.conv_proj", hidden, 2 * flowSize, 1, 0);
        return new PosteriorEncoder(pre, waveNet, proj, flowSize);
    }

    /// <summary>The posterior's means and log-stddevs [F, T] each of the spectrogram [bins, T] with the speaker's embedding [E, 1] (training draws the latents from them).</summary>
    public (Tensor Means, Tensor LogStddevs) Stats(Tensor spectrogram, Tensor? speaker)
    {
        int t = spectrogram.Shape[1];
        var (means, logs) = Stats(spectrogram.Reshape(1, spectrogram.Shape[0], t), speaker, null);
        return (means.Reshape(_flowSize, t), logs.Reshape(_flowSize, t));
    }

    /// <summary>
    /// A batch of spectrograms [B, bins, T] (zero over the padding) with the speaker's embedding [E, 1] (one sequence only)
    /// and the batch's masks: the means and log-stddevs [B, F, T] each, zero over the padding. The WaveNet is a block of the
    /// model's own checkpoints (<see cref="Checkpointing.RunBlock"/>).
    /// </summary>
    public (Tensor Means, Tensor LogStddevs) Stats(Tensor spectrograms, Tensor? speaker, PaddedSequences? mask)
    {
        var x = _pre.Forward(spectrograms);
        if (mask is not null)
        {
            x = mask.Apply(x);
        }

        var stats = _proj.Forward(Checkpointing.RunBlock(h => _waveNet.Forward(h, speaker, mask), x));   // [B, 2F, T]
        if (mask is not null)
        {
            stats = mask.Apply(stats);
        }

        return (stats.Narrow(1, 0, _flowSize), stats.Narrow(1, _flowSize, _flowSize));
    }

    /// <summary>The latents [F, T] of the spectrogram [bins, T] with the speaker's embedding [E, 1]: the means, plus noise scaled by <paramref name="noiseScale"/> (none at 0), drawn from <paramref name="random"/>.</summary>
    public Tensor Forward(Tensor spectrogram, Tensor? speaker, float noiseScale, Random random)
    {
        ArgumentNullException.ThrowIfNull(random);
        var stats = _proj.Forward(_waveNet.Forward(_pre.Forward(spectrogram), speaker));   // [2F, T]
        var means = stats.Narrow(0, 0, _flowSize);
        if (noiseScale == 0)
        {
            return means;
        }

        int frames = spectrogram.Shape[1];
        var noise = new float[_flowSize * frames];
        for (int i = 0; i < noise.Length; i++)
        {
            noise[i] = (float)random.NextGaussian();
        }

        return means + Tensor.From(noise, [_flowSize, frames], means.Device) * stats.Narrow(0, _flowSize, _flowSize).Exp() * noiseScale;
    }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => Stats(input, null).Means;

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => [_pre, _waveNet, _proj];

    /// <inheritdoc />
    protected override void MoveTo(Device device) => throw new NotSupportedException("The layers are built on their device.");

    /// <inheritdoc />
    public override void Dispose()
    {
        _pre.Dispose();
        _proj.Dispose();
        _waveNet.Dispose();
        base.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() => $"PosteriorEncoder({_flowSize} latents)";
}
