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
    /// Per-pixel cross-entropy for semantic segmentation: <paramref name="logits"/> [N, classes, H, W] (as convolutions
    /// give them) against each pixel's class index, [N, H, W] or [N, H * W] (as a data set's targets). The mean over pixels.
    /// </summary>
    public static Tensor PixelCrossEntropy(Tensor logits, Tensor classIndices) => PixelCrossEntropy(logits, classIndices, 0f);

    /// <summary><see cref="PixelCrossEntropy(Tensor, Tensor)"/> with label smoothing.</summary>
    public static Tensor PixelCrossEntropy(Tensor logits, Tensor classIndices, float labelSmoothing)
    {
        if (logits.Rank != 4)
        {
            throw new ArgumentException($"PixelCrossEntropy takes logits [N, classes, H, W], not {Tensor.FormatShape(logits.Shape)}.", nameof(logits));
        }

        // Classes last, so each pixel is one row of SparseCrossEntropy (which reshapes the targets to [N, H, W]).
        return SparseCrossEntropy(logits.Permute(0, 2, 3, 1), classIndices, labelSmoothing);
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
    /// Knowledge distillation (Hinton, Vinyals and Dean 2015, "Distilling the Knowledge in a Neural Network",
    /// arXiv:1503.02531): T² · the mean over rows of KL(softmax(teacher / T) ‖ softmax(student / T)), the student learning
    /// the teacher's softened distribution. The T² factor keeps the gradient's size about the same whatever the temperature
    /// (the soft targets' gradients shrink as 1 / T²), so the term mixes with a label loss at a comparable weight. The
    /// teacher's logits are constants (no gradient flows into them).
    /// </summary>
    /// <param name="studentLogits">[..., classes] the student's raw scores.</param>
    /// <param name="teacherLogits">[..., classes] the teacher's raw scores for the same samples.</param>
    /// <param name="temperature">Softens both distributions (1: as they are; 2 to 4 is usual).</param>
    public static Tensor Distillation(Tensor studentLogits, Tensor teacherLogits, float temperature)
    {
        ArgumentNullException.ThrowIfNull(studentLogits);
        ArgumentNullException.ThrowIfNull(teacherLogits);
        CheckTemperature(temperature);
        if (!studentLogits.Shape.SequenceEqual(teacherLogits.Shape))
        {
            throw new ArgumentException($"The student's and the teacher's logits differ in shape: {Tensor.FormatShape(studentLogits.Shape)} and {Tensor.FormatShape(teacherLogits.Shape)}.");
        }

        int classes = studentLogits.Shape[^1];
        int rows = classes == 0 ? 0 : studentLogits.Size / classes;
        var teacher = teacherLogits.Detach() * (1f / temperature);
        var divergence = teacher.Softmax() * (teacher.LogSoftmax() - (studentLogits * (1f / temperature)).LogSoftmax());
        return divergence.Sum() * (temperature * temperature / Math.Max(rows, 1));
    }

    /// <summary>
    /// A distillation loss for <see cref="Training.Trainer"/>: α · <see cref="Distillation(Tensor, Tensor, float)"/> + (1 − α) ·
    /// the cross-entropy on the labels. The targets carry the teacher's logits first and the labels after them, as
    /// <see cref="Training.Distillation.WithTeacher"/> builds them: [samples, classes + classes] with one-hot (or
    /// probability) labels, or [samples, classes + 1] with class indices. With α = 1 the labels are not used (and may be
    /// left out).
    /// </summary>
    /// <param name="temperature">Softens both distributions (see <see cref="Distillation(Tensor, Tensor, float)"/>).</param>
    /// <param name="alpha">The weight of the distillation term; the label cross-entropy gets 1 − α.</param>
    public static Func<Tensor, Tensor, Tensor> Distillation(float temperature = 2f, float alpha = 0.5f)
    {
        CheckTemperature(temperature);
        if (alpha is < 0f or > 1f || float.IsNaN(alpha))
        {
            throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha is in [0, 1].");
        }

        return (prediction, target) =>
        {
            int classes = prediction.Shape[^1], columns = target.Shape[^1], dim = target.Rank - 1;
            int labels = columns - classes;
            if (target.Rank != prediction.Rank || labels < 0 || alpha < 1f && labels != classes && labels != 1)
            {
                throw new ArgumentException($"Distillation targets are the teacher's {classes} logits followed by the labels ({classes} one-hot columns or 1 class index), "
                                            + $"not {columns} columns: build them with Distillation.WithTeacher.");
            }

            var teacher = labels == 0 ? target : target.Narrow(dim, 0, classes);
            var soft = Distillation(prediction, teacher, temperature);
            if (alpha >= 1f)
            {
                return soft;
            }

            var given = target.Narrow(dim, classes, labels);
            var hard = labels == classes ? CrossEntropy(prediction, given) : SparseCrossEntropy(prediction, given);
            return soft * alpha + hard * (1f - alpha);
        };
    }

    private static void CheckTemperature(float temperature)
    {
        if (!(temperature > 0f) || float.IsInfinity(temperature))
        {
            throw new ArgumentOutOfRangeException(nameof(temperature), "The temperature is a positive number.");
        }
    }

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
