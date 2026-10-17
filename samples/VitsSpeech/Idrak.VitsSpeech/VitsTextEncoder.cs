// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;
using Idrak.Models;

namespace Idrak.VitsSpeech;

/// <summary>
/// transformers' <c>VitsTextEncoder</c>: the ids' embeddings scaled by √hidden, post-norm encoder layers (attention with
/// a relative-position bias within <see cref="VitsConfig.WindowSize"/>, a convolutional feed-forward), then a projection
/// to the prior's means and log-variances. One utterance unpadded when speaking; a padded batch with masks when training.
/// </summary>
internal sealed class VitsTextEncoder : IDisposable
{
    private readonly VitsConfig _config;
    private readonly Tensor _embed;
    private readonly Layer[] _layers;
    private readonly Linear _project;

    public VitsTextEncoder(StoredLayers weights, VitsConfig config)
    {
        _config = config;
        _embed = weights.Tensor("text_encoder.embed_tokens.weight", config.VocabSize, config.HiddenSize);
        _layers = new Layer[config.Layers];
        for (int i = 0; i < config.Layers; i++)
        {
            _layers[i] = new Layer(weights, $"text_encoder.encoder.layers.{i}", config);
        }

        _project = weights.ProjectionLinear("text_encoder.project", config.HiddenSize, 2 * config.FlowSize);
    }

    /// <summary>The ids' hidden states [T, C], and the prior's means and log-variances [T, F] each.</summary>
    public (Tensor Hidden, Tensor Means, Tensor LogVariances) Forward(ReadOnlySpan<int> ids) => Forward([ids.ToArray()], null);

    /// <summary>
    /// A batch of utterances' ids, padded to the mask's length (null: one utterance, nothing padded): the hidden states
    /// [B·T, C] and the prior's means and log-variances [B·T, F] each, as rows, zero over the padding. The attention reads
    /// no padded key and the feed-forward convolutions read the padding as zeros (VITS's masking), so each utterance's
    /// rows are what it gets alone.
    /// </summary>
    public (Tensor Hidden, Tensor Means, Tensor LogVariances) Forward(IReadOnlyList<int[]> ids, PaddedSequences? mask)
    {
        int batch = ids.Count, t = mask?.Length ?? ids[0].Length;
        var padded = new int[batch * t];                                         // the padding reads id 0; it is masked wherever it matters
        for (int b = 0; b < batch; b++)
        {
            ids[b].CopyTo(padded, b * t);
        }

        var x = _embed.EmbeddingLookup(Tensor.Indices(padded, _embed.Device)) * MathF.Sqrt(_config.HiddenSize);   // [B·T, C]
        if (mask is not null)
        {
            x = mask.ApplyRows(x);
        }

        foreach (var layer in _layers)
        {
            var current = layer;
            x = Checkpointing.RunBlock(h => current.Forward(h, batch, t, mask), x);
        }

        var stats = _project.Forward(x);                                         // [B·T, 2F]
        if (mask is not null)
        {
            stats = mask.ApplyRows(stats);
        }

        return (x, stats.Narrow(1, 0, _config.FlowSize), stats.Narrow(1, _config.FlowSize, _config.FlowSize));
    }

    public void Dispose()
    {
        _embed.Dispose();
        foreach (var layer in _layers) layer.Dispose();
        _project.Dispose();
    }

    private sealed class Layer : IDisposable
    {
        private readonly VitsConfig _config;
        private readonly RelativePositionAttention _attention;
        private readonly LayerNorm _norm, _finalNorm;
        private readonly ConvolutionalFeedForward _feedForward;

        public Layer(StoredLayers weights, string prefix, VitsConfig config)
        {
            _config = config;
            int c = config.HiddenSize;
            _attention = RelativePositionAttention.FromStored(weights, $"{prefix}.attention", c, config.Heads, config.WindowSize);
            _norm = weights.Norm($"{prefix}.layer_norm", c, config.LayerNormEps);
            _finalNorm = weights.Norm($"{prefix}.final_layer_norm", c, config.LayerNormEps);
            _feedForward = ConvolutionalFeedForward.FromStored(weights, $"{prefix}.feed_forward", c, config.FfnDim, config.FfnKernelSize, config.HiddenAct);
        }

        // x [B·T, C] rows of a batch of `batch` utterances padded to `t` positions (the mask null for one, unpadded).
        public Tensor Forward(Tensor x, int batch, int t, PaddedSequences? mask)
        {
            x = _norm.Forward(x + _attention.Forward(x, batch, t, mask));
            return _finalNorm.Forward(x + _feedForward.Forward(x, batch, t, mask));
        }

        public void Dispose()
        {
            _attention.Dispose();
            _norm.Dispose();
            _finalNorm.Dispose();
            _feedForward.Dispose();
        }
    }
}
