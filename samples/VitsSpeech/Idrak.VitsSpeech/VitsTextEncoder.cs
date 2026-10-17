// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using Idrak.Layers;

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

        var weight = weights.Parameter(prefix + ".weight", HostParallel.Transpose(weights.Read(prefix + ".weight"), outFeatures, inFeatures), [inFeatures, outFeatures], shape,
            transposed: true);
        return Linear.FromWeights(weight, weights.Contains(prefix + ".bias") ? weights.Tensor(prefix + ".bias", outFeatures) : null);
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

        var x = _embed.EmbeddingLookup(VitsOps.Indices(padded, _embed.Device)) * MathF.Sqrt(_config.HiddenSize);   // [B·T, C]
        if (mask is not null)
        {
            x = mask.ApplyRows(x);
        }

        foreach (var layer in _layers)
        {
            var current = layer;
            x = VitsCheckpoints.Run(h => current.Forward(h, batch, t, mask), x);
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
        private readonly Linear _q, _k, _v, _out;
        private readonly float[]? _relK, _relV;                                  // [2w + 1, d] each
        private readonly Tensor? _relKParameter, _relVParameter, _zeroRow;      // training: the relative embeddings [2w + 1, d] as parameters, and a zero row [1, d]
        private readonly LayerNorm _norm, _finalNorm;
        private readonly Conv1d _conv1, _conv2;
        private readonly ConcurrentDictionary<int, (float[] RelK, float[] RelV, float[] Offsets)> _byLength = new();

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
                if (weights.Trainable)
                {
                    // The relative embeddings as parameters, gathered per length (offsets beyond the window read the zero row).
                    _relKParameter = weights.Parameter($"{prefix}.attention.emb_rel_k", _relK, [span, d], shape);
                    _relVParameter = weights.Parameter($"{prefix}.attention.emb_rel_v", _relV, [span, d], weights.Shape($"{prefix}.attention.emb_rel_v"));
                    _zeroRow = Tensor.PersistentZeros([1, d], weights.Device);
                }
            }

            _norm = weights.Norm($"{prefix}.layer_norm", c, config.LayerNormEps);
            _finalNorm = weights.Norm($"{prefix}.final_layer_norm", c, config.LayerNormEps);
            int k = config.FfnKernelSize, symmetric = k % 2 == 1 ? k / 2 : 0;
            _conv1 = weights.Conv($"{prefix}.feed_forward.conv_1", c, config.FfnDim, k, symmetric);
            _conv2 = weights.Conv($"{prefix}.feed_forward.conv_2", config.FfnDim, c, k, symmetric);
        }

        // x [B·T, C] rows of a batch of `batch` utterances padded to `t` positions (the mask null for one, unpadded).
        public Tensor Forward(Tensor x, int batch, int t, PaddedSequences? mask)
        {
            x = _norm.Forward(x + Attention(x, batch, t, mask));
            return _finalNorm.Forward(x + FeedForward(x, batch, t, mask));
        }

        private Tensor Attention(Tensor x, int batch, int t, PaddedSequences? mask)
        {
            int h = _config.Heads, c = _config.HiddenSize, d = c / h, bh = batch * h;
            float scaling = 1f / MathF.Sqrt(d);
            Tensor Heads(Tensor rows) => rows.Reshape(batch, t, h, d).Permute(0, 2, 1, 3).Reshape(bh, t, d);   // [B·h, T, d]
            var q = Heads(_q.Forward(x) * scaling);
            var k = Heads(_k.Forward(x));
            var v = Heads(_v.Forward(x));
            var scores = q.MatMul(k, transposeB: true);                             // [B·h, T, T]
            var tables = _relK is null ? default : _byLength.GetOrAdd(t, BuildTables);
            if (_relK is not null)
            {
                var relK = RelativeTable(_relKParameter, tables.RelK, tables.Offsets, t, d, x.Device);
                var relative = q.MatMul(relK, transposeB: true);                    // [B·h, T, 2T - 1]: query i against offset r - (T - 1)
                scores = scores + RelativeToAbsolute(relative, bh, t, x.Device);
            }

            if (mask is not null)
            {
                scores = scores + mask.Keys(h);                                     // no padded key is attended
            }

            var probs = scores.Softmax();
            var output = probs.MatMul(v);                                            // [B·h, T, d]
            if (_relV is not null)
            {
                output = output + AbsoluteToRelative(probs, bh, t, x.Device).MatMul(RelativeTable(_relVParameter, tables.RelV, tables.Offsets, t, d, x.Device));
            }

            return _out.Forward(output.Reshape(batch, h, t, d).Permute(0, 2, 1, 3).Reshape(batch * t, c));
        }

        // [h, T, 2T - 1] by offset (r = j - i + T - 1) to [h, T, T] by key, and back (zeros where a key falls outside), with
        // VITS's padding and reshapes: each value moves once, so the backward pass is slices and padding (no gather's
        // scatter over h·T² positions, and no index table sent to the device each call).
        private static Tensor RelativeToAbsolute(Tensor x, int h, int t, Device device)
        {
            if (t == 1)
            {
                return x;
            }

            var padded = Tensor.Concat([x, Tensor.Zeros([h, t, 1], device)], 2);                                    // [h, T, 2T]
            var flat = Tensor.Concat([padded.Reshape(h, 2 * t * t), Tensor.Zeros([h, t - 1], device)], 1);         // [h, 2T² + T - 1]
            return flat.Reshape(h, t + 1, 2 * t - 1).Narrow(1, 0, t).Narrow(2, t - 1, t);
        }

        private static Tensor AbsoluteToRelative(Tensor x, int h, int t, Device device)
        {
            if (t == 1)
            {
                return x;
            }

            var padded = Tensor.Concat([x, Tensor.Zeros([h, t, t - 1], device)], 2);                                // [h, T, 2T - 1]
            var flat = Tensor.Concat([Tensor.Zeros([h, t], device), padded.Reshape(h, t * (2 * t - 1))], 1);       // [h, 2T²]
            return flat.Reshape(h, t, 2 * t).Narrow(2, 1, 2 * t - 1);
        }

        // The relative embeddings [2T - 1, d] for T queries: the host's slices when speaking; when training, the parameters'
        // rows gathered (each offset's row, the zero row beyond the window), so their gradients reach the parameters.
        private Tensor RelativeTable(Tensor? parameter, float[] host, float[] offsets, int t, int d, Device device) =>
            parameter is null
                ? Tensor.From(host, [2 * t - 1, d], device)
                : Tensor.Concat([parameter, _zeroRow!], 0).EmbeddingLookup(Tensor.From(offsets, [offsets.Length], device));

        // The relative embeddings for T queries: offset r in [0, 2T - 2] is key j = i + r - (T - 1).
        private (float[] RelK, float[] RelV, float[] Offsets) BuildTables(int t)
        {
            int h = _config.Heads, w = _config.WindowSize, d = _config.HiddenSize / h, offsets = 2 * t - 1;
            var relK = new float[offsets * d];
            var relV = new float[offsets * d];
            var rows = new float[offsets];
            for (int r = 0; r < offsets; r++)
            {
                int offset = r - (t - 1);
                rows[r] = Math.Abs(offset) <= w ? offset + w : 2 * w + 1;
                if (Math.Abs(offset) <= w)
                {
                    _relK!.AsSpan((offset + w) * d, d).CopyTo(relK.AsSpan(r * d, d));
                    _relV!.AsSpan((offset + w) * d, d).CopyTo(relV.AsSpan(r * d, d));
                }
            }

            return (relK, relV, rows);
        }

        // The convolutions over time read the padding as zeros (VITS multiplies by the mask before each).
        private Tensor FeedForward(Tensor x, int batch, int t, PaddedSequences? mask)
        {
            int c = _config.HiddenSize, k = _config.FfnKernelSize, left = k % 2 == 1 ? 0 : k / 2, right = k % 2 == 1 ? 0 : (k - 1) / 2;   // even kernels pad asymmetrically
            var y = x.Reshape(batch, t, c).Permute(0, 2, 1);                       // [B, C, T]
            if (mask is not null)
            {
                y = mask.Apply(y);
            }

            var h = VitsOps.Activation(_config.HiddenAct, _conv1.Forward(y.Pad(-1, left, right)));
            if (mask is not null)
            {
                h = mask.Apply(h);
            }

            return _conv2.Forward(h.Pad(-1, left, right)).Permute(0, 2, 1).Reshape(batch * t, c);
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
            _relKParameter?.Dispose();
            _relVParameter?.Dispose();
            _zeroRow?.Dispose();
        }
    }
}
