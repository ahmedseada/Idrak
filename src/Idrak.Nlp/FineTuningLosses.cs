// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Optimizers;
using Idrak.Models;

namespace Idrak.Nlp;

/// <summary>
/// A fine-tuning loss (<see cref="FineTuningOptions.Loss"/>): the loss of one batch from the log-probabilities of its
/// trained tokens, as a single-element tensor built with ordinary tensor operations from
/// <see cref="FineTuningLossInput.LogProbabilities"/> (and, for losses against a reference model,
/// <see cref="FineTuningLossInput.ReferenceLogProbabilities"/>). The losses of the batches of one optimizer step are added,
/// so a loss averaged over the step divides by <see cref="FineTuningLossInput.StepTokens"/> or
/// <see cref="FineTuningLossInput.StepSequences"/>. Built-in: <see cref="FineTuningLosses"/>.
/// </summary>
/// <param name="input">The batch's log-probabilities and how they divide into sequences.</param>
/// <returns>The batch's loss, one element.</returns>
public delegate Tensor FineTuningLoss(FineTuningLossInput input);

/// <summary>
/// What a <see cref="FineTuningLoss"/> receives for one batch: log p of every trained token (one value per trained token,
/// the batch's sequences one after another, each in order), how many belong to each sequence, and the sizes of the whole
/// optimizer step. The values live on the CPU (a few per token), so a loss is any composition of tensor operations; the
/// trainer turns its gradient into gradients of the network with one more pass of the output head over the trained rows,
/// so the full [tokens, vocabulary] logits are never stored.
/// </summary>
public sealed class FineTuningLossInput
{
    private readonly Func<float[]>? _reference;
    private Func<float, float[]>? _divergence;
    private Tensor? _referenceTensor;
    private Tensor? _sums, _means;
    private readonly List<(Tensor Values, float Temperature)> _divergences = [];

    /// <summary>
    /// The input for <paramref name="logProbabilities"/> [tokens] divided into sequences of
    /// <paramref name="tokenCounts"/> tokens: for running a loss on values of one's own (tests, other trainers).
    /// </summary>
    /// <param name="logProbabilities">log p of each trained token, [sum of <paramref name="tokenCounts"/>].</param>
    /// <param name="tokenCounts">Trained tokens of each sequence, in order.</param>
    /// <param name="referenceLogProbabilities">The same values under the reference model, or null when there are none.</param>
    /// <param name="pairs">Whether the sequences are preference pairs (see <see cref="Pairs"/>).</param>
    /// <param name="stepTokens">Trained tokens of the whole step (0: this input's).</param>
    /// <param name="stepSequences">Sequences of the whole step (0: this input's).</param>
    /// <param name="teacherDivergence">
    /// The divergence from a teacher of each trained token at a temperature (see <see cref="TeacherDivergence"/>), or null
    /// when there is no teacher.
    /// </param>
    public FineTuningLossInput(Tensor logProbabilities, IReadOnlyList<int> tokenCounts, Tensor? referenceLogProbabilities = null, bool pairs = false,
        int stepTokens = 0, int stepSequences = 0, Func<float, float[]>? teacherDivergence = null)
        : this(logProbabilities, tokenCounts, [], null, pairs, stepTokens, stepSequences)
    {
        ArgumentNullException.ThrowIfNull(logProbabilities);
        _referenceTensor = referenceLogProbabilities;
        _divergence = teacherDivergence;
        if (referenceLogProbabilities is not null && referenceLogProbabilities.Size != logProbabilities.Size)
        {
            throw new ArgumentException("The reference log-probabilities need one value per trained token.", nameof(referenceLogProbabilities));
        }
    }

    internal FineTuningLossInput(Tensor logProbabilities, IReadOnlyList<int> tokenCounts, IReadOnlyList<TrainingSequence> sequences, Func<float[]>? reference,
        bool pairs, int stepTokens, int stepSequences, Func<float, float[]>? divergence = null)
    {
        _divergence = divergence;
        ArgumentNullException.ThrowIfNull(tokenCounts);
        if (tokenCounts.Count == 0 || tokenCounts.Any(c => c < 0) || tokenCounts.Sum() != logProbabilities.Size)
        {
            throw new ArgumentException($"The token counts ({tokenCounts.Sum()} in {tokenCounts.Count} sequences) do not match the {logProbabilities.Size} log-probabilities.");
        }

        if (pairs && tokenCounts.Count % 2 != 0)
        {
            throw new ArgumentException("Preference pairs need an even number of sequences.");
        }

        LogProbabilities = logProbabilities;
        TokenCounts = tokenCounts;
        Sequences = sequences;
        Pairs = pairs;
        _reference = reference;
        StepTokens = stepTokens > 0 ? stepTokens : tokenCounts.Sum();
        StepSequences = stepSequences > 0 ? stepSequences : tokenCounts.Count;
    }

    /// <summary>log p of each trained token [tokens]: sequence 0's trained tokens, then sequence 1's, and so on. Recorded.</summary>
    public Tensor LogProbabilities { get; }

    /// <summary>
    /// <see cref="LogProbabilities"/> under the reference model: the same network with its adapters disabled (the base
    /// model, see <see cref="Layers.ModuleExtensions.DisableAdapters"/>), so no second copy of the model is held. Computed
    /// on first use, with one more forward pass of the batch without gradients, and kept for each sequence for the rest of
    /// the run (later epochs and evaluations reuse it). Not recorded.
    /// </summary>
    public Tensor ReferenceLogProbabilities
    {
        get
        {
            if (_referenceTensor is null)
            {
                var values = _reference?.Invoke() ?? throw new InvalidOperationException("No reference log-probabilities were given for this input.");
                _referenceTensor = Tensor.From(values, [values.Length], LogProbabilities.Device);
            }

            return _referenceTensor;
        }
    }

    /// <summary>
    /// The teacher's divergence from the model at each trained token [tokens]: KL(q ‖ p) between the teacher's
    /// distribution over the next token, q = softmax(teacher logits / T), and the model's, p = softmax(logits / T), at
    /// <paramref name="temperature"/> T (the teacher is <see cref="FineTuningOptions.Teacher"/>; see
    /// <see cref="DistillationTeacher"/>). Recorded: a loss may combine it with <see cref="LogProbabilities"/> freely, and
    /// the trainer turns its gradient into the network's with one more pass of the output head over the trained rows
    /// (row t's logits receive g_t · (p − q) / T), so the [tokens, vocabulary] logits are never stored. Each temperature
    /// asked for costs one more pass of the head (and of the teacher's head) over the trained rows.
    /// </summary>
    /// <param name="temperature">Softens both distributions (1: as they are).</param>
    public Tensor TeacherDivergence(float temperature = 1f)
    {
        if (!(temperature > 0f) || float.IsInfinity(temperature))
        {
            throw new ArgumentOutOfRangeException(nameof(temperature), "The temperature is a positive number.");
        }

        foreach (var (values, t) in _divergences)
        {
            if (t == temperature)
            {
                return values;
            }
        }

        var divergence = _divergence?.Invoke(temperature)
                         ?? throw new InvalidOperationException("No teacher for this input: set FineTuningOptions.Teacher (DistillationTeachers.FromModel or FromFile) to distil.");
        if (divergence.Length != LogProbabilities.Size)
        {
            throw new InvalidOperationException($"The teacher gave {divergence.Length} divergences for {LogProbabilities.Size} trained tokens.");
        }

        var tensor = Tensor.From(divergence, [divergence.Length], LogProbabilities.Device, requiresGrad: LogProbabilities.RequiresGrad);
        _divergences.Add((tensor, temperature));
        return tensor;
    }

    /// <summary>The <see cref="TeacherDivergence"/> tensors the loss asked for, with their temperatures (the trainer back-propagates them).</summary>
    internal IReadOnlyList<(Tensor Values, float Temperature)> Divergences => _divergences;

    /// <summary>Trained tokens of each sequence of the batch, in order.</summary>
    public IReadOnlyList<int> TokenCounts { get; }

    /// <summary>The batch's sequences, in order (empty when the input was made from values alone).</summary>
    public IReadOnlyList<TrainingSequence> Sequences { get; }

    /// <summary>
    /// Whether the sequences are preference pairs: sequence 2k is the chosen answer of pair k and 2k + 1 the rejected one
    /// (both after the same prompt). True when training with <see cref="PreferencePair"/>s.
    /// </summary>
    public bool Pairs { get; }

    /// <summary>Trained tokens in all the batches of the optimizer step (with gradient accumulation, more than this batch's).</summary>
    public int StepTokens { get; }

    /// <summary>Sequences in all the batches of the optimizer step (twice the pairs when <see cref="Pairs"/>).</summary>
    public int StepSequences { get; }

    /// <summary>The sum of each sequence's values of a per-token tensor [tokens] (such as <see cref="LogProbabilities"/>): [sequences].</summary>
    public Tensor SequenceSums(Tensor perToken) => Combine(perToken, ref _sums, mean: false);

    /// <summary>The mean of each sequence's values of a per-token tensor [tokens]: [sequences].</summary>
    public Tensor SequenceMeans(Tensor perToken) => Combine(perToken, ref _means, mean: true);

    /// <summary>The chosen answers' entries of a per-sequence tensor [sequences] (entries 0, 2, 4 …): [pairs].</summary>
    public Tensor Chosen(Tensor perSequence) => Pick(perSequence, 0);

    /// <summary>The rejected answers' entries of a per-sequence tensor [sequences] (entries 1, 3, 5 …): [pairs].</summary>
    public Tensor Rejected(Tensor perSequence) => Pick(perSequence, 1);

    // [1, tokens] · [tokens, sequences], with each column 1 (or 1 / count) on its sequence's tokens.
    private Tensor Combine(Tensor perToken, ref Tensor? matrix, bool mean)
    {
        ArgumentNullException.ThrowIfNull(perToken);
        int tokens = LogProbabilities.Size, sequences = TokenCounts.Count;
        if (perToken.Size != tokens)
        {
            throw new ArgumentException($"A per-token tensor has {tokens} values here, not {perToken.Size}.", nameof(perToken));
        }

        if (matrix is null)
        {
            var values = new float[tokens * sequences];
            for (int s = 0, t = 0; s < sequences; s++)
            {
                for (int i = 0; i < TokenCounts[s]; i++, t++)
                {
                    values[t * sequences + s] = mean ? 1f / TokenCounts[s] : 1f;
                }
            }

            matrix = Tensor.From(values, [tokens, sequences], perToken.Device);
        }

        return perToken.Reshape(1, tokens).MatMul(matrix).Reshape(sequences);
    }

    private Tensor Pick(Tensor perSequence, int offset)
    {
        ArgumentNullException.ThrowIfNull(perSequence);
        if (!Pairs)
        {
            throw new InvalidOperationException("The sequences are not preference pairs: train with PreferencePair data for a preference loss.");
        }

        int sequences = TokenCounts.Count, pairs = sequences / 2;
        if (perSequence.Size != sequences)
        {
            throw new ArgumentException($"A per-sequence tensor has {sequences} values here, not {perSequence.Size}.", nameof(perSequence));
        }

        var values = new float[sequences * pairs];
        for (int p = 0; p < pairs; p++)
        {
            values[(2 * p + offset) * pairs + p] = 1f;
        }

        var select = Tensor.From(values, [sequences, pairs], perSequence.Device);           // read again by the backward pass
        return perSequence.Reshape(1, sequences).MatMul(select).Reshape(pairs);
    }
}

/// <summary>
/// One preference example: the chosen and the rejected answer to the same prompt, each a <see cref="TrainingSequence"/>
/// whose trained tokens are the answer alone (see <see cref="ChatTranscriptEncoder.EncodePreference"/>). Trained with
/// <see cref="FineTuner.Train(PretrainedModel, IReadOnlyList{PreferencePair}, IReadOnlyList{PreferencePair}?, FineTuningOptions, string?, IProgress{FineTuningProgress}?, CancellationToken, Action{string}?)"/>
/// and a preference loss (<see cref="FineTuningLosses.Dpo"/>, <see cref="FineTuningLosses.Orpo"/>, <see cref="FineTuningLosses.SimPo"/>).
/// </summary>
/// <param name="Chosen">The preferred answer.</param>
/// <param name="Rejected">The answer to move away from.</param>
public sealed record PreferencePair(TrainingSequence Chosen, TrainingSequence Rejected);

/// <summary>
/// Built-in <see cref="FineTuningLoss"/>es: the token cross-entropy of supervised fine-tuning, and preference losses over
/// <see cref="PreferencePair"/>s (DPO, ORPO, SimPO). Each preference loss is the mean over the step's pairs.
/// </summary>
public static class FineTuningLosses
{
    /// <summary>
    /// The default loss written as a <see cref="FineTuningLoss"/>: the mean negative log-likelihood per trained token over
    /// the step. Training without <see cref="FineTuningOptions.Loss"/> computes the same loss in one fused pass (and can
    /// record it as a graph); this form is a starting point for losses of one's own.
    /// </summary>
    public static FineTuningLoss TokenCrossEntropy { get; } = input => -input.LogProbabilities.Sum() / input.StepTokens;

    /// <summary>
    /// Knowledge distillation per token (Hinton, Vinyals and Dean 2015, "Distilling the Knowledge in a Neural Network",
    /// arXiv:1503.02531): the mean over the step's trained tokens of α · T² · KL(q_T ‖ p_T) + (1 − α) · (−log p(token)),
    /// with q_T and p_T the teacher's and the model's next-token distributions softened by the temperature T
    /// (<see cref="FineTuningLossInput.TeacherDivergence"/>; the teacher is <see cref="FineTuningOptions.Teacher"/>). The
    /// T² factor keeps the soft term's gradient about as large as at T = 1, so α weighs the two terms as it says. Training
    /// with a teacher and no <see cref="FineTuningOptions.Loss"/> uses this loss with its defaults.
    /// </summary>
    /// <param name="temperature">Softens both distributions (1: as they are; 1 to 2 is usual for language models).</param>
    /// <param name="alpha">The weight of the distillation term (1: the teacher alone); the label cross-entropy gets 1 − α.</param>
    public static FineTuningLoss Distillation(float temperature = 1f, float alpha = 0.5f)
    {
        if (!(temperature > 0f) || float.IsInfinity(temperature))
        {
            throw new ArgumentOutOfRangeException(nameof(temperature), "The temperature is a positive number.");
        }

        if (alpha is < 0f or > 1f || float.IsNaN(alpha))
        {
            throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha is in [0, 1].");
        }

        return input =>
        {
            var hard = alpha >= 1f ? null : input.LogProbabilities.Sum() * (-(1f - alpha) / input.StepTokens);
            if (alpha <= 0f)
            {
                return hard!;
            }

            var soft = input.TeacherDivergence(temperature).Sum() * (alpha * temperature * temperature / input.StepTokens);
            return hard is null ? soft : soft + hard;
        };
    }

    /// <summary>
    /// Direct preference optimization (Rafailov et al. 2023, "Direct Preference Optimization: Your Language Model is
    /// Secretly a Reward Model", arXiv:2305.18290): for each pair, -log σ(β · ((π_c − ref_c) − (π_r − ref_r))) where π and ref
    /// are an answer's summed token log-probabilities under the model and under the reference model. The reference is the
    /// base model, computed with the adapters disabled (no second model is loaded; see
    /// <see cref="FineTuningLossInput.ReferenceLogProbabilities"/>), as peft and TRL do for adapters. With
    /// <paramref name="labelSmoothing"/> ε, the conservative form -(1 − ε) log σ(βh) − ε log σ(−βh) (preferences assumed wrong with
    /// probability ε).
    /// </summary>
    /// <param name="beta">How far the model may move from the reference (larger: closer to it). TRL's default is 0.1.</param>
    /// <param name="labelSmoothing">The probability that a preference is wrong (0 for plain DPO).</param>
    public static FineTuningLoss Dpo(float beta = 0.1f, float labelSmoothing = 0f)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(beta);
        if (labelSmoothing is < 0f or >= 0.5f)
        {
            throw new ArgumentOutOfRangeException(nameof(labelSmoothing), "Label smoothing is in [0, 0.5).");
        }

        return input =>
        {
            Require(input, "DPO");
            var ratios = input.SequenceSums(input.LogProbabilities) - input.SequenceSums(input.ReferenceLogProbabilities);
            var margins = (input.Chosen(ratios) - input.Rejected(ratios)) * beta;
            var losses = -LogSigmoid(margins);
            if (labelSmoothing > 0f)
            {
                losses = losses * (1f - labelSmoothing) - LogSigmoid(-margins) * labelSmoothing;
            }

            return losses.Sum() / (input.StepSequences / 2);
        };
    }

    /// <summary>
    /// Odds ratio preference optimization (Hong et al. 2024, "ORPO: Monolithic Preference Optimization without Reference
    /// Model", arXiv:2403.07691): for each pair, the chosen answer's mean negative log-likelihood per token plus
    /// λ · −log σ(log odds_c − log odds_r), with odds = p / (1 − p) and p the answer's mean token probability
    /// (exp of its mean log-probability). Needs no reference model. The likelihood term is averaged per answer, then over the
    /// pairs (TRL averages it over the chosen tokens of a batch; the two agree when the answers have equal lengths).
    /// </summary>
    /// <param name="lambda">Weight of the odds-ratio term (the paper's λ, TRL's beta; 0.1 by default).</param>
    public static FineTuningLoss Orpo(float lambda = 0.1f)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(lambda);
        return input =>
        {
            Require(input, "ORPO");
            var means = input.SequenceMeans(input.LogProbabilities);
            // log odds = a − log(1 − exp a) for the mean log-probability a < 0 (as TRL's log1p(−exp a)); the small floor keeps
            // an answer the model is already certain of finite.
            var logOdds = means - (1f - means.Exp() + 1e-7f).Log();
            var ratio = LogSigmoid(input.Chosen(logOdds) - input.Rejected(logOdds));
            var losses = -input.Chosen(means) - ratio * lambda;
            return losses.Sum() / (input.StepSequences / 2);
        };
    }

    /// <summary>
    /// Simple preference optimization (Meng et al. 2024, "SimPO: Simple Preference Optimization with a Reference-Free
    /// Reward", arXiv:2405.14734): for each pair, −log σ(β · (a_c − a_r) − γ) with a an answer's mean token
    /// log-probability (a length-normalized reward) and γ a target margin. Needs no reference model.
    /// </summary>
    /// <param name="beta">Reward scale (the paper uses 2 to 2.5).</param>
    /// <param name="gamma">Target reward margin (the paper uses about 0.5 to 1.6).</param>
    public static FineTuningLoss SimPo(float beta = 2f, float gamma = 1f)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(beta);
        return input =>
        {
            Require(input, "SimPO");
            var means = input.SequenceMeans(input.LogProbabilities);
            var margins = (input.Chosen(means) - input.Rejected(means)) * beta - gamma;
            return -LogSigmoid(margins).Sum() / (input.StepSequences / 2);
        };
    }

    /// <summary>
    /// log σ(x), element-wise, without overflow for large |x|: −(m + log(exp(−m) + exp(−x − m))) with m = max(−x, 0), so
    /// no exponent is positive. Written as a shifted log-sum-exp (not with |x|) so the gradient at x = 0 is the true ½: the
    /// derivative of m cancels whatever value the kink takes.
    /// </summary>
    public static Tensor LogSigmoid(Tensor x)
    {
        ArgumentNullException.ThrowIfNull(x);
        var m = (-x).Relu();
        return -(m + ((-m).Exp() + (-x - m).Exp()).Log());
    }

    private static void Require(FineTuningLossInput input, string name)
    {
        if (!input.Pairs)
        {
            throw new InvalidOperationException($"{name} needs preference pairs: train with PreferencePair data (chosen and rejected answers).");
        }
    }
}

/// <summary>
/// Built-in learning-rate schedules for <see cref="FineTuningOptions.Scheduler"/>: each is created from the optimizer and
/// the number of optimizer steps, and stepped once per optimizer step. A warm-up raises the rate linearly to the
/// optimizer's own rate over the first <c>warmupFraction</c> of the steps (step i of w uses (i + 1) / (w + 1) of it).
/// </summary>
public static class FineTuningSchedules
{
    /// <summary>Warm-up, then cosine decay to <paramref name="minLearningRate"/> at the last step (the default schedule).</summary>
    public static Func<Optimizer, int, LearningRateScheduler> Cosine(float warmupFraction = 0.03f, float minLearningRate = 0f) =>
        (optimizer, steps) => new CosineAnnealing(optimizer, Math.Max(1, steps), minLearningRate, Warmup(warmupFraction, steps));

    /// <summary>Warm-up, then linear decay to <paramref name="minLearningRate"/> at the last step.</summary>
    public static Func<Optimizer, int, LearningRateScheduler> Linear(float warmupFraction = 0.03f, float minLearningRate = 0f) =>
        (optimizer, steps) =>
        {
            int warmup = Warmup(warmupFraction, steps);
            return new LambdaSchedule(optimizer, (step, rate) => step < warmup ? rate * (step + 1) / (warmup + 1)
                : rate + (minLearningRate - rate) * (float)Math.Min(1.0, (step - warmup) / (double)Math.Max(1, steps - warmup)));
        };

    /// <summary>Warm-up, then the optimizer's rate unchanged.</summary>
    public static Func<Optimizer, int, LearningRateScheduler> Constant(float warmupFraction = 0f) =>
        (optimizer, steps) =>
        {
            int warmup = Warmup(warmupFraction, steps);
            return new LambdaSchedule(optimizer, (step, rate) => step < warmup ? rate * (step + 1) / (warmup + 1) : rate);
        };

    /// <summary>
    /// Warm-up, stable, decay (Hu et al. 2024, "MiniCPM", arXiv:2404.06395): warm-up, then the optimizer's rate until the
    /// last <paramref name="decayFraction"/> of the steps, then linear decay to <paramref name="minLearningRate"/>.
    /// </summary>
    public static Func<Optimizer, int, LearningRateScheduler> WarmupStableDecay(float warmupFraction = 0.03f, float decayFraction = 0.2f, float minLearningRate = 0f)
    {
        if (decayFraction is < 0f or > 1f)
        {
            throw new ArgumentOutOfRangeException(nameof(decayFraction), "The decay fraction is in [0, 1].");
        }

        return (optimizer, steps) =>
        {
            int warmup = Warmup(warmupFraction, steps);
            int decay = Math.Max(0, Math.Min(steps - warmup, (int)Math.Round(decayFraction * steps)));
            int stable = steps - decay;
            return new LambdaSchedule(optimizer, (step, rate) => step < warmup ? rate * (step + 1) / (warmup + 1)
                : step < stable ? rate
                : rate + (minLearningRate - rate) * (float)Math.Min(1.0, (step - stable + 1) / (double)Math.Max(1, decay)));
        };
    }

    private static int Warmup(float fraction, int steps) => (int)Math.Round(fraction * steps);
}

/// <summary>Built-in optimizers for <see cref="FineTuningOptions.Optimizer"/>, by name (for command lines and settings files).</summary>
public static class FineTuningOptimizers
{
    /// <summary>The names <see cref="Create"/> accepts: adamw, adam, adamw8bit, sgd.</summary>
    public static IReadOnlyList<string> Names { get; } = ["adamw", "adam", "adamw8bit", "sgd"];

    /// <summary>
    /// The optimizer factory for <paramref name="name"/> (see <see cref="Names"/>, ignoring case) with
    /// <paramref name="learningRate"/> and <paramref name="weightDecay"/> (decoupled for AdamW and 8-bit AdamW, L2 for Adam
    /// and SGD); SGD uses <paramref name="momentum"/>.
    /// </summary>
    public static Func<IReadOnlyList<Tensor>, Optimizer> Create(string name, float learningRate, float weightDecay = 0f, float momentum = 0.9f)
    {
        ArgumentNullException.ThrowIfNull(name);
        return name.ToLowerInvariant() switch
        {
            "adamw" => ps => new AdamW(ps, learningRate, weightDecay: weightDecay),
            "adam" => ps => new Adam(ps, learningRate, weightDecay: weightDecay),
            "adamw8bit" => ps => new AdamW8Bit(ps, learningRate, weightDecay: weightDecay),
            "sgd" => ps => new Sgd(ps, learningRate, momentum, weightDecay),
            _ => throw new ArgumentException($"Unknown optimizer '{name}' (known: {string.Join(", ", Names)}).", nameof(name)),
        };
    }
}
