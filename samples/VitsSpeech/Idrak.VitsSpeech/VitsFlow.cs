// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.VitsSpeech;

/// <summary>
/// transformers' <c>VitsResidualCouplingBlock</c> run in reverse: for each coupling layer from the last, the channels
/// flipped, then the first half conditions a WaveNet whose output (a pointwise projection) is subtracted from the second
/// half (the log-stddev is zero, so there is no scaling). <see cref="Forward"/> runs it the other way (what voice
/// conversion takes a posterior's latents through, to the prior).
/// </summary>
internal sealed class VitsFlow : IDisposable
{
    private readonly CouplingLayer[] _layers;

    public VitsFlow(VitsWeights weights, VitsConfig config)
    {
        _layers = new CouplingLayer[config.PriorFlows];
        for (int i = 0; i < config.PriorFlows; i++)
        {
            _layers[i] = new CouplingLayer(weights, $"flow.flows.{i}", config);
        }
    }

    /// <summary>The acoustic latents [F, T] of the prior latents [F, T], with the speaker's embedding [E, 1] (null for one speaker).</summary>
    public Tensor Reverse(Tensor z, Tensor? speaker)
    {
        for (int i = _layers.Length - 1; i >= 0; i--)
        {
            z = _layers[i].Reverse(VitsOps.FlipRows(z), speaker);
        }

        return z;
    }

    /// <summary>The prior latents [F, T] of the acoustic latents [F, T] (the inverse of <see cref="Reverse"/>), with the speaker's embedding [E, 1].</summary>
    public Tensor Forward(Tensor z, Tensor? speaker)
    {
        for (int i = 0; i < _layers.Length; i++)
        {
            z = VitsOps.FlipRows(_layers[i].Forward(z, speaker));
        }

        return z;
    }

    public void Dispose()
    {
        foreach (var layer in _layers) layer.Dispose();
    }

    private sealed class CouplingLayer : IDisposable
    {
        private readonly int _half;
        private readonly Conv1d _pre, _post;
        private readonly VitsWaveNet _waveNet;

        public CouplingLayer(VitsWeights weights, string prefix, VitsConfig config)
        {
            _half = config.FlowSize / 2;
            _pre = weights.Conv($"{prefix}.conv_pre", _half, config.HiddenSize, 1, 0);
            _waveNet = new VitsWaveNet(weights, $"{prefix}.wavenet", config, config.PriorWaveNetLayers);
            _post = weights.Conv($"{prefix}.conv_post", config.HiddenSize, _half, 1, 0);
        }

        public Tensor Reverse(Tensor z, Tensor? speaker)
        {
            var first = z.Narrow(0, 0, _half);
            var second = z.Narrow(0, _half, _half);
            return Tensor.Concat([first, second - Mean(first, speaker)], 0);
        }

        public Tensor Forward(Tensor z, Tensor? speaker)
        {
            var first = z.Narrow(0, 0, _half);
            var second = z.Narrow(0, _half, _half);
            return Tensor.Concat([first, second + Mean(first, speaker)], 0);
        }

        private Tensor Mean(Tensor first, Tensor? speaker) => _post.Forward(_waveNet.Forward(_pre.Forward(first), speaker));

        public void Dispose()
        {
            _pre.Dispose();
            _post.Dispose();
            _waveNet.Dispose();
        }
    }
}
