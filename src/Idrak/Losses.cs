// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak;

/// <summary>
/// Loss functions. Each returns a scalar tensor you can call <see cref="Tensor.Backward()"/> on.
/// The intermediate tensors belong to the autograd graph and must stay alive until
/// <see cref="Tensor.Backward()"/> runs, so they are left to the caller's <see cref="TensorScope"/>.
/// </summary>
public static class Losses
{
    /// <summary>Mean squared error: mean((prediction - target)²). The standard regression loss.</summary>
    public static Tensor MeanSquaredError(Tensor prediction, Tensor target) => (prediction - target).Square().Mean();

    /// <summary>Mean absolute error: mean(|prediction - target|). Less sensitive to outliers than MSE.</summary>
    public static Tensor MeanAbsoluteError(Tensor prediction, Tensor target) => (prediction - target).Abs().Mean();

    /// <summary>
    /// <see cref="CrossEntropy(Tensor, Tensor, float)"/> without label smoothing. As a two-argument method it passes where
    /// a loss is a <c>Func&lt;Tensor, Tensor, Tensor&gt;</c>: <c>Loss = Losses.CrossEntropy</c>.
    /// </summary>
    public static Tensor CrossEntropy(Tensor logits, Tensor targets) => CrossEntropy(logits, targets, 0f);

    /// <summary>
    /// Multi-class cross-entropy from raw scores (logits, no softmax layer needed):
    /// mean over rows of -Σ target · log_softmax(logits). Targets are one-hot rows (or class probabilities)
    /// with the same shape as the logits; see <see cref="Data.Dataset.FromClassLabels"/>.
    /// </summary>
    /// <param name="logits">[..., classes] unnormalized scores.</param>
    /// <param name="targets">[..., classes] one-hot or probability targets.</param>
    /// <param name="labelSmoothing">Mixes the targets with a uniform distribution (e.g. 0.1) to reduce over-confidence.</param>
    public static Tensor CrossEntropy(Tensor logits, Tensor targets, float labelSmoothing)
    {
        int classes = logits.Shape[^1];
        int rows = classes == 0 ? 0 : logits.Size / classes;
        var smoothed = labelSmoothing > 0f ? targets * (1f - labelSmoothing) + labelSmoothing / classes : targets;
        return (smoothed * logits.LogSoftmax()).Sum() * (-1f / Math.Max(rows, 1));
    }

    /// <summary>
    /// <see cref="SparseCrossEntropy(Tensor, Tensor, float)"/> without label smoothing; passes as a
    /// <c>Func&lt;Tensor, Tensor, Tensor&gt;</c> loss (<c>Loss = Losses.SparseCrossEntropy</c>).
    /// </summary>
    public static Tensor SparseCrossEntropy(Tensor logits, Tensor classIndices) => SparseCrossEntropy(logits, classIndices, 0f);

    /// <summary>
    /// Cross-entropy with integer class targets: <paramref name="classIndices"/> has the logits' shape without the
    /// last dimension ([N] for [N, classes], [N, T] for per-token [N, T, vocabulary] outputs). Equivalent to
    /// <see cref="CrossEntropy(Tensor, Tensor)"/> with one-hot targets, without storing them.
    /// </summary>
    public static Tensor SparseCrossEntropy(Tensor logits, Tensor classIndices, float labelSmoothing)
    {
        // Not disposed here: the backward pass of the product inside CrossEntropy reads the targets.
        var targets = Tensor.OneHot(classIndices.Reshape(logits.Shape[..^1]), logits.Shape[^1]);
        return CrossEntropy(logits, targets, labelSmoothing);
    }

    /// <summary>
    /// Language-model loss over token positions: the weighted cross-entropy of <paramref name="head"/>(hidden) against
    /// <paramref name="targets"/> (token ids as floats, [rows]) with <paramref name="weights"/> [rows] (0 masks a
    /// position, for example prompt tokens), divided by <paramref name="normalizer"/> (usually the number of weighted
    /// tokens). The head (the output projection) runs on <paramref name="chunkRows"/> rows at a time with a fused
    /// softmax/cross-entropy kernel, so the full [rows, vocabulary] logits are never stored: fine-tuning long sequences
    /// over large vocabularies fits in memory. Call Backward() on the result without scaling it.
    /// </summary>
    public static Tensor TokenCrossEntropy(Tensor hidden, Func<Tensor, Tensor> head, Tensor targets, Tensor weights, float normalizer, int chunkRows = 1024) =>
        Tensor.TokenCrossEntropy(hidden, head, targets, weights, normalizer, chunkRows);

    /// <summary>
    /// <see cref="TokenCrossEntropy"/> computed only on the rows listed in <paramref name="rows"/> (for example the trained
    /// positions of a batch): the head runs on those rows alone. <paramref name="targets"/> and <paramref name="weights"/>
    /// hold one value per listed row; other rows count as weight 0. Same loss and gradients as the full version.
    /// </summary>
    public static Tensor TokenCrossEntropyRows(Tensor hidden, Func<Tensor, Tensor> head, int[] rows, float[] targets, float[] weights, float normalizer,
        int chunkRows = 1024) =>
        Tensor.TokenCrossEntropyRows(hidden, head, rows, targets, weights, normalizer, chunkRows);

    /// <summary>
    /// The log-probability of chosen tokens: log p(<paramref name="targets"/>[i] | position <paramref name="rows"/>[i]) under
    /// <paramref name="head"/>(hidden) for hidden [positions, dim], for scoring given answers (the likelihood of each
    /// candidate continuation, perplexity). Computed on the device <paramref name="chunkRows"/> rows at a time with a fused
    /// log-softmax, so only one value per listed row leaves it (never the [rows, vocabulary] logits). Not recorded.
    /// </summary>
    public static float[] TokenLogProbabilities(Tensor hidden, Func<Tensor, Tensor> head, int[] rows, int[] targets, int chunkRows = 1024) =>
        Tensor.TokenLogProbabilities(hidden, head, rows, targets, chunkRows);

    /// <summary>
    /// Binary cross-entropy for probabilities in (0, 1), e.g. after a <see cref="Layers.Sigmoid"/> layer:
    /// -mean(y·log p + (1 - y)·log(1 - p)). Prefer <see cref="BinaryCrossEntropyWithLogits"/> for stability.
    /// </summary>
    public static Tensor BinaryCrossEntropy(Tensor probabilities, Tensor targets)
    {
        const float Eps = 1e-7f;
        var positive = targets * (probabilities + Eps).Log();
        var negative = (1f - targets) * (1f - probabilities + Eps).Log();
        return -(positive + negative).Mean();
    }

    /// <summary>
    /// Binary cross-entropy from raw scores, computed stably as mean(max(x, 0) - x·y + log(1 + e^-|x|)).
    /// Use it with a final layer that has no sigmoid.
    /// </summary>
    public static Tensor BinaryCrossEntropyWithLogits(Tensor logits, Tensor targets) =>
        (logits.Relu() - logits * targets + ((-logits.Abs()).Exp() + 1f).Log()).Mean();
}
