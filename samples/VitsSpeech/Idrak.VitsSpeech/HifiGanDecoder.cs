// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.VitsSpeech;

/// <summary>
/// transformers' <c>VitsHifiGan</c>: a convolution into the first channel count, then per stage a leaky ReLU, a
/// transposed convolution (the rate's upsampling), the residual blocks averaged; a last leaky ReLU, the output
/// convolution and tanh. <see cref="VitsSynthesizer.Vocoder"/> exposes it as an <see cref="IVocoder"/> over the flow's latents.
/// </summary>
internal sealed class HifiGanDecoder : IDisposable
{
    private readonly VitsConfig _config;
    private readonly Conv1d _pre, _post;
    private readonly Conv1d? _cond;
    private readonly ConvTranspose1d[] _upsamplers;
    private readonly ResidualBlock[] _blocks;

    public HifiGanDecoder(VitsWeights weights, VitsConfig config)
    {
        _config = config;
        int channels = config.UpsampleInitialChannel;
        _pre = weights.Conv("decoder.conv_pre", config.FlowSize, channels, 7, 3);
        if (config.SpeakerEmbeddingSize > 0)
        {
            _cond = weights.Conv("decoder.cond", config.SpeakerEmbeddingSize, channels, 1, 0);
        }

        int stages = config.UpsampleRates.Length, kernels = config.ResblockKernelSizes.Length;
        _upsamplers = new ConvTranspose1d[stages];
        _blocks = new ResidualBlock[stages * kernels];
        for (int i = 0; i < stages; i++)
        {
            int rate = config.UpsampleRates[i], kernel = config.UpsampleKernelSizes[i];
            int inChannels = channels >> i, outChannels = channels >> (i + 1);
            _upsamplers[i] = weights.ConvTranspose($"decoder.upsampler.{i}", inChannels, outChannels, kernel, rate, (kernel - rate) / 2);
            for (int j = 0; j < kernels; j++)
            {
                _blocks[i * kernels + j] = new ResidualBlock(weights, $"decoder.resblocks.{i * kernels + j}", outChannels, config.ResblockKernelSizes[j],
                    config.ResblockDilationSizes[j], config.LeakyReluSlope);
            }
        }

        _post = weights.Conv("decoder.conv_post", channels >> stages, 1, 7, 3);
    }

    /// <summary>The samples [1, T · hop] of the latents [F, T], with the speaker's embedding [E, 1] (null for one speaker).</summary>
    public Tensor Forward(Tensor latents, Tensor? speaker)
    {
        var x = _pre.Forward(latents);
        if (speaker is not null && _cond is not null)
        {
            x = x + VitsOps.ExpandColumns(_cond.Forward(speaker), x.Shape[1]);
        }

        int kernels = _config.ResblockKernelSizes.Length;
        for (int i = 0; i < _upsamplers.Length; i++)
        {
            x = _upsamplers[i].Forward(VitsOps.LeakyRelu(x, _config.LeakyReluSlope));
            Tensor? sum = null;
            for (int j = 0; j < kernels; j++)
            {
                var y = _blocks[i * kernels + j].Forward(x);
                sum = sum is null ? y : sum + y;
            }

            x = kernels == 1 ? sum! : sum! * (1f / kernels);
        }

        return _post.Forward(VitsOps.LeakyRelu(x, _config.FinalLeakyReluSlope)).Tanh();
    }

    public void Dispose()
    {
        _pre.Dispose();
        _post.Dispose();
        _cond?.Dispose();
        foreach (var u in _upsamplers) u.Dispose();
        foreach (var b in _blocks) b.Dispose();
    }

    private sealed class ResidualBlock : IDisposable
    {
        private readonly float _slope;
        private readonly Conv1d[] _convs1, _convs2;

        public ResidualBlock(VitsWeights weights, string prefix, int channels, int kernel, int[] dilations, float slope)
        {
            _slope = slope;
            _convs1 = new Conv1d[dilations.Length];
            _convs2 = new Conv1d[dilations.Length];
            for (int i = 0; i < dilations.Length; i++)
            {
                int dilation = dilations[i];
                _convs1[i] = weights.Conv($"{prefix}.convs1.{i}", channels, channels, kernel, (kernel * dilation - dilation) / 2, dilation);
                _convs2[i] = weights.Conv($"{prefix}.convs2.{i}", channels, channels, kernel, (kernel - 1) / 2);
            }
        }

        public Tensor Forward(Tensor x)
        {
            for (int i = 0; i < _convs1.Length; i++)
            {
                var h = _convs1[i].Forward(VitsOps.LeakyRelu(x, _slope));
                h = _convs2[i].Forward(VitsOps.LeakyRelu(h, _slope));
                x = h + x;
            }

            return x;
        }

        public void Dispose()
        {
            foreach (var c in _convs1) c.Dispose();
            foreach (var c in _convs2) c.Dispose();
        }
    }
}
