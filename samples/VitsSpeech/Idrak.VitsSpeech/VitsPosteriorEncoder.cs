// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.VitsSpeech;

/// <summary>
/// transformers' <c>VitsPosteriorEncoder</c>: a linear spectrogram [bins, T] through a pointwise convolution, a WaveNet
/// conditioned on the speaker, and a pointwise projection to the latents' means and log-stddevs; the latents are the
/// means plus noise of the stddevs times a scale (OpenVoice's tau; transformers' training draws with scale 1).
/// </summary>
internal sealed class VitsPosteriorEncoder : IDisposable
{
    private readonly int _flowSize;
    private readonly Conv1d _pre, _proj;
    private readonly VitsWaveNet _waveNet;

    public VitsPosteriorEncoder(VitsWeights weights, VitsConfig config)
    {
        _flowSize = config.FlowSize;
        _pre = weights.Conv("posterior_encoder.conv_pre", config.SpectrogramBins, config.HiddenSize, 1, 0);
        _waveNet = new VitsWaveNet(weights, "posterior_encoder.wavenet", config, config.PosteriorWaveNetLayers);
        _proj = weights.Conv("posterior_encoder.conv_proj", config.HiddenSize, 2 * config.FlowSize, 1, 0);
    }

    /// <summary>The posterior's means and log-stddevs [F, T] each of the spectrogram [bins, T] with the speaker's embedding [E, 1] (training draws the latents from them).</summary>
    public (Tensor Means, Tensor LogStddevs) Stats(Tensor spectrogram, Tensor? speaker)
    {
        var stats = _proj.Forward(_waveNet.Forward(_pre.Forward(spectrogram), speaker));   // [2F, T]
        return (stats.Narrow(0, 0, _flowSize), stats.Narrow(0, _flowSize, _flowSize));
    }

    /// <summary>The latents [F, T] of the spectrogram [bins, T] with the speaker's embedding [E, 1]: the means, plus noise scaled by <paramref name="noiseScale"/> (none at 0).</summary>
    public Tensor Forward(Tensor spectrogram, Tensor? speaker, float noiseScale, Random random)
    {
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
            noise[i] = (float)VitsNoise.Normal(random);
        }

        return means + Tensor.From(noise, [_flowSize, frames], means.Device) * stats.Narrow(0, _flowSize, _flowSize).Exp() * noiseScale;
    }

    public void Dispose()
    {
        _pre.Dispose();
        _proj.Dispose();
        _waveNet.Dispose();
    }
}
