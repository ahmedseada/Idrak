using Idrak.Diagnostics;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training;

namespace Idrak.Retrieval;

/// <summary>
/// Turns texts into unit-length vectors whose dot product measures how related they are (a bi-encoder, for
/// <see cref="VectorIndex"/>). The <see cref="Model"/> maps token ids [N, T] to hidden states [N, T, D] (for example
/// an <see cref="Embedding"/> followed by transformer layers); the encoder averages the states of the real tokens
/// (padding excluded) and scales the result to length 1.
/// </summary>
public sealed class TextEncoder : IEmbedder
{
    /// <summary>Creates the encoder.</summary>
    /// <param name="model">Token ids [N, T] → hidden states [N, T, D].</param>
    /// <param name="tokenizer">Turns text into ids.</param>
    /// <param name="maxLength">Texts are cut to this many tokens (the model's T).</param>
    /// <param name="padId">The id used to fill short texts; padded positions are left out of the average.</param>
    public TextEncoder(Module model, ITokenizer tokenizer, int maxLength, int padId)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxLength);
        Model = model;
        Tokenizer = tokenizer;
        MaxLength = maxLength;
        PadId = padId;
    }

    /// <summary>Token ids → hidden states.</summary>
    public Module Model { get; }

    /// <summary>The tokenizer.</summary>
    public ITokenizer Tokenizer { get; }

    /// <summary>Tokens per text.</summary>
    public int MaxLength { get; }

    /// <summary>The padding id.</summary>
    public int PadId { get; }

    private Device Device => Model.Parameters().FirstOrDefault()?.Device ?? Device.Default;

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
                var vectors = Forward(texts.Skip(start).Take(count).ToList()).ToArray();
                int d = vectors.Length / count;
                for (int i = 0; i < count; i++)
                {
                    result[start + i] = vectors.AsSpan(i * d, d).ToArray();
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
        var ids = new float[n * t];
        var weights = new float[n * t];
        for (int i = 0; i < n; i++)
        {
            var tokens = Tokenizer.Encode(texts[i]);
            int length = Math.Min(tokens.Count, t);
            for (int j = 0; j < t; j++)
            {
                ids[i * t + j] = j < length ? tokens[j] : PadId;
                weights[i * t + j] = j < length ? 1f / Math.Max(length, 1) : 0f;
            }

            if (length == 0)
            {
                weights[i * t] = 1f;                      // an empty text averages the first (padding) position
            }
        }

        var device = Device;
        var hidden = Model.Forward(Tensor.From(ids, [n, t], device));                        // [N, T, D]
        int d = hidden.Shape[^1];
        var pooled = Tensor.From(weights, [n, 1, t], device).MatMul(hidden).Reshape(n, d);    // weighted mean over real tokens
        var inverseNorm = ((pooled.Square().Sum(1) + 1e-12f).Log() * -0.5f).Exp();           // 1 / ‖x‖
        return inverseNorm.Reshape(n, 1, 1).MatMul(pooled.Reshape(n, 1, d)).Reshape(n, d);
    }
}
