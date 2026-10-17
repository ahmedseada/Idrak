// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.VitsSpeech;

/// <summary>
/// transformers' <c>VitsResidualCouplingBlock</c> run in reverse: for each coupling layer from the last, the channels
/// flipped, then the first half conditions a WaveNet whose output (a pointwise projection) is subtracted from the second
/// half (the log-stddev is zero, so there is no scaling). <see cref="Forward(Tensor, Tensor?)"/> runs it the other way (what voice
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
    public Tensor Forward(Tensor z, Tensor? speaker) => Forward(z.Reshape(1, z.Shape[0], z.Shape[1]), speaker, null).Reshape(z.Shape[0], z.Shape[1]);

    /// <summary>
    /// A batch of acoustic latents [B, F, T] (zero over the padding) to the prior's, with the speaker's embedding [E, 1]
    /// (one utterance only) and the batch's mask: each coupling layer's shift is zeroed over the padding, as VITS does.
    /// </summary>
    public Tensor Forward(Tensor z, Tensor? speaker, VitsMask? mask)
    {
        for (int i = 0; i < _layers.Length; i++)
        {
            z = VitsOps.FlipRows(_layers[i].Forward(z, speaker, mask));
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
            int t = z.Shape[1];
            var first = z.Narrow(0, 0, _half);
            var second = z.Narrow(0, _half, _half);
            return Tensor.Concat([first, second - Mean(first.Reshape(1, _half, t), speaker, null).Reshape(_half, t)], 0);
        }

        public Tensor Forward(Tensor z, Tensor? speaker, VitsMask? mask)
        {
            var first = z.Narrow(1, 0, _half);
            var second = z.Narrow(1, _half, _half);
            var moved = second + VitsCheckpoints.Run(x => Mean(x, speaker, mask), first);
            return Tensor.Concat([first, mask is null ? moved : mask.Apply(moved)], 1);
        }

        // The shift [B, half, T] the first half conditions, zero over the padding.
        private Tensor Mean(Tensor first, Tensor? speaker, VitsMask? mask)
        {
            var h = _pre.Forward3(first);
            if (mask is not null)
            {
                h = mask.Apply(h);
            }

            var shift = _post.Forward3(_waveNet.Forward(h, speaker, mask));
            return mask is null ? shift : mask.Apply(shift);
        }

        public void Dispose()
        {
            _pre.Dispose();
            _post.Dispose();
            _waveNet.Dispose();
        }
    }
}
