// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using Idrak.Diagnostics;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training;

namespace Idrak.Text.Retrieval;

/// <summary>
/// Turns texts into unit-length vectors whose dot product measures how related they are (a bi-encoder, for
/// <see cref="VectorIndex"/>). The <see cref="Model"/> maps token ids [N, T] to hidden states [N, T, D] (for example
/// an <see cref="Embedding"/> followed by transformer layers), or an <see cref="ISequenceEncoder"/> (a pretrained encoder
/// family's, or the library's <c>SequenceEncoder</c>) does, given each text's length so no token attends to padding; the
/// states of the real tokens are pooled (<see cref="Pooler"/>: their average unless set, padding excluded) and the
/// result scaled to length 1.
/// </summary>
public sealed class TextEncoder : IEmbedder
{
    private readonly TokenizerEncodeOptions _batchOptions;   // an IPretrainedTokenizer's batch encode (made once)
    private readonly ISequenceEncoder? _encoder;
    private readonly ISequencePooler _pooler = SequencePoolers.Get(SequencePoolers.Mean);

    /// <summary>Creates the encoder.</summary>
    /// <param name="model">Token ids [N, T] → hidden states [N, T, D].</param>
    /// <param name="tokenizer">
    /// Turns text into ids; an <see cref="IPretrainedTokenizer"/> encodes each batch in one call, on its workers.
    /// </param>
    /// <param name="maxLength">Texts are cut to this many tokens (the model's T).</param>
    /// <param name="padId">The id used to fill short texts; padded positions are left out of the average.</param>
    public TextEncoder(Module model, ITokenizer tokenizer, int maxLength, int padId)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxLength);
        Model = model;
        Tokenizer = tokenizer;
        MaxLength = maxLength;
        PadId = padId;
        // A pretrained tokenizer's batch encode, giving Encode's ids cut to the first MaxLength: no special tokens, cut from
        // the right, not padded (the pad ids are written below), ids only (offsets cost time and memory the encoder never uses).
        _batchOptions = new TokenizerEncodeOptions
        {
            AddSpecialTokens = false, MaxLength = maxLength, Truncation = "longest_first", TruncationDirection = "right", Padding = "none",
            Offsets = false,                                   // ids only: the plain encode, no offsets or token strings
        };
    }

    /// <summary>
    /// Creates the encoder over a sequence encoder: each batch's texts are encoded with their lengths (no token attends to
    /// another text's padding), then pooled by <see cref="Pooler"/>.
    /// </summary>
    /// <param name="encoder">Token ids, lengths → hidden states [N, T, D].</param>
    /// <param name="tokenizer">Turns text into ids (the encoder's vocabulary).</param>
    /// <param name="maxLength">Texts are cut to this many tokens (at most the encoder's <see cref="ISequenceEncoder.MaxPositions"/>).</param>
    /// <param name="padId">The id used to fill short texts.</param>
    /// <exception cref="ArgumentOutOfRangeException">The length is not positive or exceeds the encoder's.</exception>
    public TextEncoder(ISequenceEncoder encoder, ITokenizer tokenizer, int maxLength, int padId)
        : this((encoder ?? throw new ArgumentNullException(nameof(encoder))).Network, tokenizer, maxLength, padId)
    {
        if (maxLength > encoder.MaxPositions)
        {
            throw new ArgumentOutOfRangeException(nameof(maxLength), maxLength, $"The encoder takes at most {encoder.MaxPositions} tokens.");
        }

        _encoder = encoder;
    }

    /// <summary>Token ids → hidden states (the sequence encoder's layers when made over one).</summary>
    public Module Model { get; }

    /// <summary>The sequence encoder, or null when the encoder runs <see cref="Model"/> on the ids alone.</summary>
    public ISequenceEncoder? Encoder => _encoder;

    /// <summary>
    /// How the states become one vector per text: a name in <see cref="SequencePoolers"/> ("mean", the default: the
    /// average of the real tokens' states, as always; "first", "max", or one registered).
    /// </summary>
    /// <exception cref="NotSupportedException">No pooler is registered under the name.</exception>
    public string Pooler
    {
        get => _pooler.Name;
        init => _pooler = SequencePoolers.Get(value);
    }

    /// <summary>The tokenizer.</summary>
    public ITokenizer Tokenizer { get; }

    /// <summary>Tokens per text.</summary>
    public int MaxLength { get; }

    /// <summary>The padding id.</summary>
    public int PadId { get; }

    private Device Device => Model.WeightsDevice ?? Device.Default;

    /// <inheritdoc />
    public ValueTask<float[][]> EmbedAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default) =>
        ValueTask.FromResult(Encode(texts));

    /// <summary>The vector of one text.</summary>
    public float[] Encode(string text) => Encode([text])[0];

    /// <summary>The vectors of several texts, computed in batches of <paramref name="batchSize"/>.</summary>
    public float[][] Encode(IReadOnlyList<string> texts, int batchSize = 256)
    {
        var result = new float[texts.Count][];
        bool wasTraining = Model.IsTraining;
        Model.Eval();
        try
        {
            for (int start = 0; start < texts.Count; start += batchSize)
            {
                int count = Math.Min(batchSize, texts.Count - start);
                using var noGrad = Autograd.NoGrad();
                using var scope = new TensorScope();
                // The batch is downloaded once into a pooled buffer (returned in finally) and each vector copied out of it.
                var vectors = Forward(texts.Skip(start).Take(count).ToList());
                int size = vectors.Size, d = size / count;
                float[] flat = ArrayPool<float>.Shared.Rent(size);
                try
                {
                    vectors.CopyTo(flat.AsSpan(0, size));
                    for (int i = 0; i < count; i++)
                    {
                        result[start + i] = flat.AsSpan(i * d, d).ToArray();
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(flat);
                }
            }
        }
        finally
        {
            Model.Train(wasTraining);
        }

        return result;
    }

    /// <summary>
    /// Trains on related (query, passage) pairs with in-batch negatives: in each batch, every query's own passage is
    /// its positive and the other passages are negatives; the loss is the cross-entropy of the query·passage
    /// similarities divided by <paramref name="temperature"/>. Returns the mean loss of each epoch.
    /// </summary>
    /// <param name="pairs">Queries and the passages that answer them. Avoid two identical passages in one batch.</param>
    /// <param name="epochs">Passes over the pairs.</param>
    /// <param name="batchSize">Pairs per step (more pairs = more negatives per query).</param>
    /// <param name="optimizer">Creates the optimizer from the model's trainable parameters.</param>
    /// <param name="temperature">Similarity scale, e.g. 0.05.</param>
    /// <param name="seed">Shuffling seed.</param>
    public IReadOnlyList<double> Train(IReadOnlyList<(string Query, string Passage)> pairs, int epochs, int batchSize,
        Func<IEnumerable<Tensor>, Optimizer> optimizer, float temperature, int seed)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(batchSize, 2);
        using var opt = optimizer(Model.Parameters().Where(p => p.RequiresGrad).ToList());
        var random = new Random(seed);
        var losses = new List<double>();
        var order = Enumerable.Range(0, pairs.Count).ToArray();
        Model.Train();
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            random.Shuffle(order);
            double sum = 0;
            int steps = 0;
            for (int start = 0; start + 2 <= order.Length; start += batchSize)
            {
                int count = Math.Min(batchSize, order.Length - start);
                if (count < 2)
                {
                    break;
                }

                using var scope = new TensorScope();
                var batch = order.AsSpan(start, count).ToArray().Select(i => pairs[i]).ToList();
                var q = Forward([.. batch.Select(p => p.Query)]);
                var p = Forward([.. batch.Select(p => p.Passage)]);
                var logits = q.MatMul(p, transposeB: true) * (1f / temperature);
                var targets = Tensor.OneHot(Tensor.From([.. Enumerable.Range(0, count).Select(i => (float)i)], [count], Device), count);
                var loss = Losses.CrossEntropy(logits, targets);
                opt.ZeroGrad();
                loss.Backward();
                opt.Step();
                sum += loss.Item();
                steps++;
            }

            losses.Add(steps == 0 ? double.NaN : sum / steps);
        }

        Model.Eval();
        return losses;
    }

    // ids → hidden states → masked mean → unit length, as tensor operations (differentiable).
    private Tensor Forward(IReadOnlyList<string> texts)
    {
        int n = texts.Count, t = MaxLength;

        // The ids are written into a pooled buffer, uploaded (Tensor.From copies them), and returned in finally.
        float[] ids = ArrayPool<float>.Shared.Rent(n * t);
        var lengths = new int[n];
        Tensor input;
        var device = Device;
        try
        {
            if (Tokenizer is IPretrainedTokenizer pretrained)
            {
                // One tokenize pass for the batch, on the tokenizer's workers (the same ids as Encode, cut to MaxLength).
                var encodings = pretrained.EncodeBatch(texts, null, _batchOptions);
                for (int i = 0; i < n; i++)
                {
                    lengths[i] = Fill(ids, i, encodings[i].Ids);
                }
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    var tokens = Tokenizer.Encode(texts[i]);
                    lengths[i] = Fill(ids, i, tokens);
                }
            }

            input = Tensor.From(ids.AsSpan(0, n * t), [n, t], device);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(ids);
        }

        // [N, T, D]: a sequence encoder sees the lengths (no token attends to padding); a plain model the ids alone.
        var hidden = _encoder is null ? Model.Forward(input) : _encoder.Encode(input, lengths);
        int d = hidden.Shape[^1];
        var pooled = _pooler.Pool(hidden, lengths);                                         // over the real tokens (the mean by default)
        var inverseNorm = ((pooled.Square().Sum(1) + 1e-12f).Log() * -0.5f).Exp();           // 1 / ‖x‖
        return inverseNorm.Reshape(n, 1, 1).MatMul(pooled.Reshape(n, 1, d)).Reshape(n, d);
    }

    // Row i of the ids: the first MaxLength tokens, then the pad id; returns the text's length (its tokens kept). The
    // "mean" pooler weighs each real token 1 / length (an empty text: its first, padding, position), as this encoder did.
    private int Fill(float[] ids, int i, IReadOnlyList<int> tokens)
    {
        int t = MaxLength, length = Math.Min(tokens.Count, t);
        for (int j = 0; j < t; j++)
        {
            ids[i * t + j] = j < length ? tokens[j] : PadId;
        }

        return length;
    }
}
