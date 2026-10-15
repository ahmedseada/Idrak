// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using Idrak.Layers;

namespace Idrak.VitsSpeech;

/// <summary>
/// transformers' <c>VitsTextEncoder</c>: the ids' embeddings scaled by √hidden, post-norm encoder layers (attention with
/// a relative-position bias within <see cref="VitsConfig.WindowSize"/>, a convolutional feed-forward), then a projection
/// to the prior's means and log-variances. One utterance at a time, so there is no padding and no mask.
/// </summary>
internal sealed class VitsTextEncoder : IDisposable
{
    private readonly VitsConfig _config;
    private readonly Tensor _embed;
    private readonly Layer[] _layers;
    private readonly Linear _project;

    public VitsTextEncoder(VitsWeights weights, VitsConfig config)
    {
        _config = config;
        _embed = weights.Tensor("text_encoder.embed_tokens.weight", config.VocabSize, config.HiddenSize);
        _layers = new Layer[config.Layers];
        for (int i = 0; i < config.Layers; i++)
        {
            _layers[i] = new Layer(weights, $"text_encoder.encoder.layers.{i}", config);
        }

        _project = ProjectionLinear(weights, "text_encoder.project", config.HiddenSize, 2 * config.FlowSize);
    }

    /// <summary>A 1 × 1 convolution <c>[out, in, 1]</c> read as a linear layer [in, out].</summary>
    internal static Linear ProjectionLinear(VitsWeights weights, string prefix, int inFeatures, int outFeatures)
    {
        var shape = weights.Shape(prefix + ".weight");
        if (shape.Length != 3 || shape[0] != outFeatures || shape[1] != inFeatures || shape[2] != 1)
        {
            throw new InvalidDataException($"'{prefix}.weight' is {Tensor.FormatShape(shape)}, the configuration says [{outFeatures}, {inFeatures}, 1].");
        }

        var weight = Tensor.Persistent(HostParallel.Transpose(weights.Read(prefix + ".weight"), outFeatures, inFeatures), [inFeatures, outFeatures], weights.Device);
        return Linear.FromWeights(weight, weights.Contains(prefix + ".bias") ? weights.Tensor(prefix + ".bias", outFeatures) : null);
    }

    /// <summary>The ids' hidden states [T, C], and the prior's means and log-variances [T, F] each.</summary>
    public (Tensor Hidden, Tensor Means, Tensor LogVariances) Forward(ReadOnlySpan<int> ids)
    {
        var x = _embed.EmbeddingLookup(VitsOps.Indices(ids, _embed.Device)) * MathF.Sqrt(_config.HiddenSize);
        foreach (var layer in _layers)
        {
            x = layer.Forward(x);
        }

        var stats = _project.Forward(x);                                         // [T, 2F]
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
        private readonly Linear _q, _k, _v, _out;
        private readonly float[]? _relK, _relV;                                  // [2w + 1, d] each
        private readonly LayerNorm _norm, _finalNorm;
        private readonly Conv1d _conv1, _conv2;
        private readonly ConcurrentDictionary<int, (float[] RelToAbs, float[] AbsToRel, float[] RelK, float[] RelV)> _byLength = new();

        public Layer(VitsWeights weights, string prefix, VitsConfig config)
        {
            _config = config;
            int c = config.HiddenSize, d = c / config.Heads;
            if (d * config.Heads != c)
            {
                throw new InvalidDataException($"hidden_size {c} is not a multiple of num_attention_heads {config.Heads}.");
            }

            _q = weights.Linear($"{prefix}.attention.q_proj", c, c);
            _k = weights.Linear($"{prefix}.attention.k_proj", c, c);
            _v = weights.Linear($"{prefix}.attention.v_proj", c, c);
            _out = weights.Linear($"{prefix}.attention.out_proj", c, c);
            if (config.WindowSize > 0)
            {
                int span = 2 * config.WindowSize + 1;
                var shape = weights.Shape($"{prefix}.attention.emb_rel_k");
                if (shape.Length != 3 || shape[1] != span || shape[2] != d)
                {
                    throw new InvalidDataException($"'{prefix}.attention.emb_rel_k' is {Tensor.FormatShape(shape)}, the configuration says [1, {span}, {d}].");
                }

                _relK = weights.Read($"{prefix}.attention.emb_rel_k");
                _relV = weights.Read($"{prefix}.attention.emb_rel_v");
            }

            _norm = weights.Norm($"{prefix}.layer_norm", c, config.LayerNormEps);
            _finalNorm = weights.Norm($"{prefix}.final_layer_norm", c, config.LayerNormEps);
            int k = config.FfnKernelSize, symmetric = k % 2 == 1 ? k / 2 : 0;
            _conv1 = weights.Conv($"{prefix}.feed_forward.conv_1", c, config.FfnDim, k, symmetric);
            _conv2 = weights.Conv($"{prefix}.feed_forward.conv_2", config.FfnDim, c, k, symmetric);
        }

        public Tensor Forward(Tensor x)
        {
            x = _norm.Forward(x + Attention(x));
            return _finalNorm.Forward(x + FeedForward(x));
        }

        private Tensor Attention(Tensor x)
        {
            int t = x.Shape[0], h = _config.Heads, c = _config.HiddenSize, d = c / h;
            float scaling = 1f / MathF.Sqrt(d);
            var q = (_q.Forward(x) * scaling).Reshape(t, h, d).Permute(1, 0, 2);   // [h, T, d]
            var k = _k.Forward(x).Reshape(t, h, d).Permute(1, 0, 2);
            var v = _v.Forward(x).Reshape(t, h, d).Permute(1, 0, 2);
            var scores = q.MatMul(k, transposeB: true);                             // [h, T, T]
            var tables = _relK is null ? default : _byLength.GetOrAdd(t, BuildTables);
            if (_relK is not null)
            {
                var relK = Tensor.From(tables.RelK, [2 * t - 1, d], x.Device);
                var relative = q.MatMul(relK, transposeB: true);                    // [h, T, 2T - 1]: query i against offset r - (T - 1)
                var bias = relative.Reshape(h * t * (2 * t - 1), 1).EmbeddingLookup(Tensor.From(tables.RelToAbs, [h * t * t], x.Device)).Reshape(h, t, t);
                scores = scores + bias;
            }

            var probs = scores.Softmax();
            var output = probs.MatMul(v);                                            // [h, T, d]
            if (_relV is not null)
            {
                var table = Tensor.Concat([probs.Reshape(h * t * t, 1), Tensor.Zeros([1, 1], x.Device)], 0);
                var relativeProbs = table.EmbeddingLookup(Tensor.From(tables.AbsToRel, [h * t * (2 * t - 1)], x.Device)).Reshape(h, t, 2 * t - 1);
                output = output + relativeProbs.MatMul(Tensor.From(tables.RelV, [2 * t - 1, d], x.Device));
            }

            return _out.Forward(output.Permute(1, 0, 2).Reshape(t, c));
        }

        // The gather indices and the relative embeddings for T queries: offset r in [0, 2T - 2] is key j = i + r - (T - 1).
        private (float[] RelToAbs, float[] AbsToRel, float[] RelK, float[] RelV) BuildTables(int t)
        {
            int h = _config.Heads, w = _config.WindowSize, d = _config.HiddenSize / h, offsets = 2 * t - 1;
            var relToAbs = new float[h * t * t];
            var absToRel = new float[h * t * offsets];
            int zeroSlot = h * t * t;
            for (int head = 0; head < h; head++)
            {
                for (int i = 0; i < t; i++)
                {
                    for (int j = 0; j < t; j++)
                    {
                        relToAbs[(head * t + i) * t + j] = (head * t + i) * offsets + (j - i + t - 1);
                    }

                    for (int r = 0; r < offsets; r++)
                    {
                        int j = i + r - (t - 1);
                        absToRel[(head * t + i) * offsets + r] = j >= 0 && j < t ? (head * t + i) * t + j : zeroSlot;
                    }
                }
            }

            var relK = new float[offsets * d];
            var relV = new float[offsets * d];
            for (int r = 0; r < offsets; r++)
            {
                int offset = r - (t - 1);
                if (Math.Abs(offset) <= w)
                {
                    _relK!.AsSpan((offset + w) * d, d).CopyTo(relK.AsSpan(r * d, d));
                    _relV!.AsSpan((offset + w) * d, d).CopyTo(relV.AsSpan(r * d, d));
                }
            }

            return (relToAbs, absToRel, relK, relV);
        }

        private Tensor FeedForward(Tensor x)
        {
            int k = _config.FfnKernelSize, left = k % 2 == 1 ? 0 : k / 2, right = k % 2 == 1 ? 0 : (k - 1) / 2;   // even kernels pad asymmetrically
            var h = VitsOps.Activation(_config.HiddenAct, _conv1.Forward(VitsOps.PadTime(x.Transpose(), left, right)));
            return _conv2.Forward(VitsOps.PadTime(h, left, right)).Transpose();
        }

        public void Dispose()
        {
            _q.Dispose();
            _k.Dispose();
            _v.Dispose();
            _out.Dispose();
            _norm.Dispose();
            _finalNorm.Dispose();
            _conv1.Dispose();
            _conv2.Dispose();
        }
    }
}
