// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Reinforcement;
using Idrak.Models;
using Idrak.Text.Abstractions;

namespace Idrak.Text;

/// <summary>
/// What an <see cref="IFineTuningLoss"/> receives for one batch: log p of every trained token (one value per trained token,
/// the batch's sequences one after another, each in order), how many belong to each sequence, and the sizes of the whole
/// optimizer step. The values live on the CPU (a few per token), so a loss is any composition of tensor operations; the
/// trainer turns its gradient into gradients of the network with one more pass of the output head over the trained rows,
/// so the full [tokens, vocabulary] logits are never stored.
/// </summary>
public sealed class FineTuningLossInput
{
    private readonly Func<float[]>? _reference;
    private readonly Func<float, ITeacherDivergence?, float, float[]>? _divergence;
    private Tensor? _referenceTensor;
    private Tensor? _sums, _means;
    private Tensor? _sequenceValues, _tokenValues, _behavior;
    private bool _behaviorRead;
    private readonly List<(Tensor Values, float Temperature, ITeacherDivergence? Kind, float Beta)> _divergences = [];

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
    /// The divergence from a teacher of each trained token at a temperature (see <see cref="TeacherDivergence(float)"/>), or
    /// null when there is no teacher. It gives the forward KL: another kind asked for is refused.
    /// </param>
    /// <param name="sequenceValues">A value per sequence [sequences] (<see cref="SequenceValues"/>), or null.</param>
    /// <param name="tokenValues">A value per trained token [tokens] (<see cref="TokenValues"/>), or null (the sequence's value repeated).</param>
    /// <param name="behaviorLogProbabilities">log p of each trained token under the policy that sampled it [tokens] (<see cref="BehaviorLogProbabilities"/>), or null.</param>
    public FineTuningLossInput(Tensor logProbabilities, IReadOnlyList<int> tokenCounts, Tensor? referenceLogProbabilities = null, bool pairs = false,
        int stepTokens = 0, int stepSequences = 0, Func<float, float[]>? teacherDivergence = null, Tensor? sequenceValues = null, Tensor? tokenValues = null,
        Tensor? behaviorLogProbabilities = null)
        : this(logProbabilities, tokenCounts, [], null, pairs, stepTokens, stepSequences)
    {
        ArgumentNullException.ThrowIfNull(logProbabilities);
        _referenceTensor = referenceLogProbabilities;
        _divergence = teacherDivergence is null ? null
            : (temperature, kind, _) => kind is null || TeacherDivergences.IsLibraryForward(kind) ? teacherDivergence(temperature)
                : throw new NotSupportedException($"This input's teacher divergence gives the forward KL only, not '{kind.Name}'.");
        if (referenceLogProbabilities is not null && referenceLogProbabilities.Size != logProbabilities.Size)
        {
            throw new ArgumentException("The reference log-probabilities need one value per trained token.", nameof(referenceLogProbabilities));
        }

        if (sequenceValues is not null && sequenceValues.Size != tokenCounts.Count)
        {
            throw new ArgumentException($"The sequence values need one value per sequence ({tokenCounts.Count}), not {sequenceValues.Size}.", nameof(sequenceValues));
        }

        if (tokenValues is not null && tokenValues.Size != logProbabilities.Size)
        {
            throw new ArgumentException("The token values need one value per trained token.", nameof(tokenValues));
        }

        if (behaviorLogProbabilities is not null && behaviorLogProbabilities.Size != logProbabilities.Size)
        {
            throw new ArgumentException("The behaviour log-probabilities need one value per trained token.", nameof(behaviorLogProbabilities));
        }

        (_sequenceValues, _tokenValues, _behavior, _behaviorRead) = (sequenceValues, tokenValues, behaviorLogProbabilities, behaviorLogProbabilities is not null);
    }

    internal FineTuningLossInput(Tensor logProbabilities, IReadOnlyList<int> tokenCounts, IReadOnlyList<TrainingSequence> sequences, Func<float[]>? reference,
        bool pairs, int stepTokens, int stepSequences, Func<float, ITeacherDivergence?, float, float[]>? divergence = null)
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
    /// model, see <see cref="Layers.ModuleExtensions.DisableAdapters"/>), so no second copy of the model is held. Taken from
    /// the sequences when every one carries them (<see cref="TrainingSequence.ReferenceLogProbabilities"/>, computed
    /// beforehand); otherwise computed on first use, with one more forward pass of the batch without gradients, and kept for
    /// each sequence for the rest of the run (later epochs and evaluations reuse it). Not recorded.
    /// </summary>
    public Tensor ReferenceLogProbabilities
    {
        get
        {
            if (_referenceTensor is null)
            {
                var values = Given(s => s.ReferenceLogProbabilities, "reference log-probabilities") ?? _reference?.Invoke()
                    ?? throw new InvalidOperationException("No reference log-probabilities were given for this input.");
                _referenceTensor = Tensor.From(values, [values.Length], LogProbabilities.Device);
            }

            return _referenceTensor;
        }
    }

    /// <summary>
    /// The value of each sequence [sequences] (<see cref="TrainingSequence.Value"/>, or the values the input was made with):
    /// a policy loss's advantage, a reward. Not recorded.
    /// </summary>
    /// <exception cref="InvalidOperationException">A sequence has no value.</exception>
    public Tensor SequenceValues
    {
        get
        {
            if (_sequenceValues is null)
            {
                if (Sequences.Count == 0 || Sequences.Any(s => s.Value is null))
                {
                    throw new InvalidOperationException("No per-sequence values were given for this input: set TrainingSequence.Value on every sequence of the step.");
                }

                _sequenceValues = Tensor.From([.. Sequences.Select(s => s.Value!.Value)], [Sequences.Count], LogProbabilities.Device);
            }

            return _sequenceValues;
        }
    }

    /// <summary>
    /// The value of each trained token [tokens]: a sequence's <see cref="TrainingSequence.TokenValues"/>, else its
    /// <see cref="TrainingSequence.Value"/> repeated over its trained tokens (a policy loss's advantage per unit). Not recorded.
    /// </summary>
    /// <exception cref="InvalidOperationException">A sequence has neither, or token values of another count than its trained tokens.</exception>
    public Tensor TokenValues
    {
        get
        {
            if (_tokenValues is null)
            {
                var values = new float[LogProbabilities.Size];
                if (Sequences.Count == 0)
                {
                    // Made from values alone: the sequence values repeated.
                    var perSequence = SequenceValues.ToArray();
                    for (int s = 0, t = 0; s < TokenCounts.Count; s++)
                    {
                        values.AsSpan(t, TokenCounts[s]).Fill(perSequence[s]);
                        t += TokenCounts[s];
                    }
                }
                else
                {
                    for (int s = 0, t = 0; s < Sequences.Count; t += TokenCounts[s], s++)
                    {
                        var sequence = Sequences[s];
                        if (sequence.TokenValues is { } own)
                        {
                            if (own.Length != TokenCounts[s])
                            {
                                throw new InvalidOperationException($"Sequence {s} has {own.Length} token values for its {TokenCounts[s]} trained tokens.");
                            }

                            own.CopyTo(values, t);
                        }
                        else
                        {
                            values.AsSpan(t, TokenCounts[s]).Fill(sequence.Value
                                ?? throw new InvalidOperationException($"Sequence {s} has no value: set TrainingSequence.Value or TokenValues on every sequence of the step."));
                        }
                    }
                }

                _tokenValues = Tensor.From(values, [values.Length], LogProbabilities.Device);
            }

            return _tokenValues;
        }
    }

    /// <summary>
    /// log p of each trained token under the policy that sampled it [tokens] (<see cref="TrainingSequence.BehaviorLogProbabilities"/>,
    /// or the values the input was made with): the denominator of a policy loss's ratio. Null when none were given (the
    /// sequences are the current policy's own: on-policy). Not recorded.
    /// </summary>
    /// <exception cref="InvalidOperationException">Some sequences carry them and others do not, or a count differs from the trained tokens.</exception>
    public Tensor? BehaviorLogProbabilities
    {
        get
        {
            if (!_behaviorRead)
            {
                _behaviorRead = true;
                var values = Given(s => s.BehaviorLogProbabilities, "behaviour log-probabilities");
                _behavior = values is null ? null : Tensor.From(values, [values.Length], LogProbabilities.Device);
            }

            return _behavior;
        }
    }

    /// <summary>Whether behaviour log-probabilities were given (<see cref="BehaviorLogProbabilities"/> is not null).</summary>
    public bool HasBehavior => BehaviorLogProbabilities is not null;

    // Values every sequence carries (each one per trained token), joined in order; null when none carries them.
    private float[]? Given(Func<TrainingSequence, float[]?> of, string what)
    {
        if (Sequences.Count == 0 || Sequences.All(s => of(s) is null))
        {
            return null;
        }

        var values = new float[LogProbabilities.Size];
        for (int s = 0, t = 0; s < Sequences.Count; t += TokenCounts[s], s++)
        {
            var own = of(Sequences[s]) ?? throw new InvalidOperationException($"Sequence {s} carries no {what} while others of the step do: give them for every sequence or none.");
            if (own.Length != TokenCounts[s])
            {
                throw new InvalidOperationException($"Sequence {s} carries {own.Length} {what} for its {TokenCounts[s]} trained tokens.");
            }

            own.CopyTo(values, t);
        }

        return values;
    }

    /// <summary>
    /// The teacher's divergence from the model at each trained token [tokens]: KL(q ‖ p) between the teacher's
    /// distribution over the next token, q = softmax(teacher logits / T), and the model's, p = softmax(logits / T), at
    /// <paramref name="temperature"/> T (the teacher is <see cref="FineTuningOptions.Teacher"/>; see
    /// <see cref="IDistillationTeacher"/>). Recorded: a loss may combine it with <see cref="LogProbabilities"/> freely, and
    /// the trainer turns its gradient into the network's with one more pass of the output head over the trained rows
    /// (row t's logits receive g_t · (p − q) / T), so the [tokens, vocabulary] logits are never stored. Each temperature
    /// asked for costs one more pass of the head (and of the teacher's head) over the trained rows.
    /// </summary>
    /// <param name="temperature">Softens both distributions (1: as they are).</param>
    public Tensor TeacherDivergence(float temperature = 1f) => Divergence(temperature, null, 0f);

    /// <summary>
    /// The teacher's divergence of the kind <paramref name="divergence"/> (a <see cref="TeacherDivergences"/> name:
    /// "forward" is <see cref="TeacherDivergence(float)"/>, "reverse" KL(p ‖ q), "jsd" the generalized Jensen-Shannon
    /// divergence of weight <paramref name="beta"/>) at each trained token [tokens], recorded as the forward one is: the
    /// trainer back-propagates it through one more pass of the head over the trained rows.
    /// </summary>
    /// <param name="temperature">Softens both distributions (1: as they are).</param>
    /// <param name="divergence">The kind, by name.</param>
    /// <param name="beta">The kind's weight when it takes one (<see cref="ITeacherDivergence.UsesBeta"/>), in (0, 1).</param>
    /// <exception cref="NotSupportedException">No divergence of that name is registered.</exception>
    public Tensor TeacherDivergence(float temperature, string divergence, float beta = 0.5f)
    {
        var kind = TeacherDivergences.Get(divergence);
        if (kind.UsesBeta && !(beta > 0f && beta < 1f))
        {
            throw new ArgumentOutOfRangeException(nameof(beta), beta, $"The '{kind.Name}' divergence's beta is in (0, 1).");
        }

        return Divergence(temperature, kind, kind.UsesBeta ? beta : 0f);
    }

    private Tensor Divergence(float temperature, ITeacherDivergence? kind, float beta)
    {
        if (!(temperature > 0f) || float.IsInfinity(temperature))
        {
            throw new ArgumentOutOfRangeException(nameof(temperature), "The temperature is a positive number.");
        }

        // The library's forward KL by name is the same as the default.
        if (kind is not null && TeacherDivergences.IsLibraryForward(kind))
        {
            kind = null;
        }

        foreach (var (values, t, k, b) in _divergences)
        {
            if (t == temperature && ReferenceEquals(k, kind) && b == beta)
            {
                return values;
            }
        }

        var divergence = _divergence?.Invoke(temperature, kind, beta) ?? throw new InvalidOperationException(NoTeacher);
        if (divergence.Length != LogProbabilities.Size)
        {
            throw new InvalidOperationException($"The teacher gave {divergence.Length} divergences for {LogProbabilities.Size} trained tokens.");
        }

        var tensor = Tensor.From(divergence, [divergence.Length], LogProbabilities.Device, requiresGrad: LogProbabilities.RequiresGrad);
        _divergences.Add((tensor, temperature, kind, beta));
        return tensor;
    }

    internal const string NoTeacher = "No teacher for this input: set FineTuningOptions.Teacher (DistillationTeachers.FromModel or FromFile) to distil.";

    /// <summary>The divergence tensors the loss asked for, with their temperatures and kinds (null: the forward KL) (the trainer back-propagates them).</summary>
    internal IReadOnlyList<(Tensor Values, float Temperature, ITeacherDivergence? Kind, float Beta)> Divergences => _divergences;

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
/// Built-in <see cref="IFineTuningLoss"/>es: the token cross-entropy of supervised fine-tuning, distillation, policy
/// losses, and preference losses over <see cref="PreferencePair"/>s (DPO, ORPO, SimPO); each preference loss is the mean
/// over the step's pairs. <see cref="Of"/> makes one of one's own from a function.
/// </summary>
public static class FineTuningLosses
{
    /// <summary>
    /// The default loss written as an <see cref="IFineTuningLoss"/>: the mean negative log-likelihood per trained token over
    /// the step. Training without <see cref="FineTuningOptions.Loss"/> computes the same loss in one fused pass (and can
    /// record it as a graph); this form is a starting point for losses of one's own.
    /// </summary>
    public static IFineTuningLoss TokenCrossEntropy { get; } = new Loss("token-cross-entropy", "the mean negative log-likelihood per trained token over the step",
        input => -input.LogProbabilities.Sum() / input.StepTokens);

    /// <summary>
    /// A loss of one's own: <paramref name="compute"/> for each batch, needing what the flags say (training refuses a loss
    /// that needs pairs or a teacher without them; a trainer prepares the reference beforehand only when
    /// <paramref name="needsReference"/>, and a loss that reads it without saying so has it computed on first use).
    /// </summary>
    /// <param name="name">Its name, for messages.</param>
    /// <param name="compute">The batch's loss, one element.</param>
    /// <param name="needsPairs">It reads preference pairs.</param>
    /// <param name="needsReference">It reads the reference's log-probabilities.</param>
    /// <param name="needsTeacher">It reads a teacher's divergences.</param>
    /// <param name="summary">One line for listings; null: the name.</param>
    public static IFineTuningLoss Of(string name, Func<FineTuningLossInput, Tensor> compute, bool needsPairs = false, bool needsReference = false, bool needsTeacher = false,
        string? summary = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(compute);
        return new Loss(name, summary ?? name, compute) { NeedsPairs = needsPairs, NeedsReference = needsReference, NeedsTeacher = needsTeacher };
    }

    /// <summary>
    /// Knowledge distillation per token (Hinton, Vinyals and Dean 2015, "Distilling the Knowledge in a Neural Network",
    /// arXiv:1503.02531): the mean over the step's trained tokens of α · T² · KL(q_T ‖ p_T) + (1 − α) · (−log p(token)),
    /// with q_T and p_T the teacher's and the model's next-token distributions softened by the temperature T
    /// (<see cref="FineTuningLossInput.TeacherDivergence(float)"/>; the teacher is <see cref="FineTuningOptions.Teacher"/>). The
    /// T² factor keeps the soft term's gradient about as large as at T = 1, so α weighs the two terms as it says. Training
    /// with a teacher and no <see cref="FineTuningOptions.Loss"/> uses this loss with its defaults.
    /// </summary>
    /// <param name="temperature">Softens both distributions (1: as they are; 1 to 2 is usual for language models).</param>
    /// <param name="alpha">The weight of the distillation term (1: the teacher alone); the label cross-entropy gets 1 − α.</param>
    public static IFineTuningLoss Distillation(float temperature = 1f, float alpha = 0.5f)
    {
        if (!(temperature > 0f) || float.IsInfinity(temperature))
        {
            throw new ArgumentOutOfRangeException(nameof(temperature), "The temperature is a positive number.");
        }

        if (alpha is < 0f or > 1f || float.IsNaN(alpha))
        {
            throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha is in [0, 1].");
        }

        return new Loss("distillation", "α · T² · KL(teacher ‖ model) + (1 − α) · token cross-entropy, per trained token", input =>
        {
            var hard = alpha >= 1f ? null : input.LogProbabilities.Sum() * (-(1f - alpha) / input.StepTokens);
            if (alpha <= 0f)
            {
                return hard!;
            }

            var soft = input.TeacherDivergence(temperature).Sum() * (alpha * temperature * temperature / input.StepTokens);
            return hard is null ? soft : soft + hard;
        })
        { NeedsTeacher = alpha > 0f };
    }

    /// <summary>
    /// A policy-gradient loss of <see cref="PolicyLosses"/> (REINFORCE, PPO, GRPO: Idrak.Reinforcement's, or one an app
    /// registered) as a fine-tuning loss, for a language model's policy trained through a <see cref="FineTuningSession"/>:
    /// each trained token is an action unit, its sequence's tokens a sequence of the policy loss
    /// (<see cref="PolicyLossInput.Counts"/>), its advantage <see cref="FineTuningLossInput.TokenValues"/> (from
    /// <see cref="TrainingSequence.Value"/> or <see cref="TrainingSequence.TokenValues"/>), the behaviour log-probabilities
    /// <see cref="FineTuningLossInput.BehaviorLogProbabilities"/> (none: on-policy, every ratio 1), and the reference the
    /// model with its adapters disabled (<see cref="FineTuningLossInput.ReferenceLogProbabilities"/>, taken from the
    /// sequences when they carry it), given only when the loss reads it with these options (<see cref="IPolicyLoss.NeedsReference"/>),
    /// so a loss without a KL term costs no reference pass. The step's sizes are the policy loss's <see cref="PolicyLossInput.StepUnits"/> and
    /// <see cref="PolicyLossInput.StepSequences"/>. The policy loss's gradient reaches the network as any loss of one's own
    /// does: one more pass of the head over the trained rows.
    /// </summary>
    /// <param name="loss">The policy loss (<see cref="PolicyLosses.Get"/>).</param>
    /// <param name="options">Its options by key (<see cref="IConfigurableKind.Keys"/>), null for its defaults.</param>
    public static IFineTuningLoss Policy(IPolicyLoss loss, IReadOnlyDictionary<string, string>? options = null)
    {
        ArgumentNullException.ThrowIfNull(loss);
        bool reference = loss.NeedsReference(options);
        return new Loss(loss.Name, loss.Summary, input =>
        {
            var policy = new PolicyLossInput(input.LogProbabilities, input.TokenCounts, input.TokenValues, input.BehaviorLogProbabilities,
                reference ? input.ReferenceLogProbabilities : null, stepUnits: input.StepTokens, stepSequences: input.StepSequences);
            return loss.Loss(policy, options);
        })
        { NeedsReference = reference };
    }

    /// <summary>
    /// On-policy distillation (Agarwal et al. 2024, "On-Policy Distillation of Language Models: Learning from
    /// Self-Generated Mistakes", arXiv:2306.13649): the mean over the step's trained tokens of T² · D(teacher, student),
    /// with D a <see cref="TeacherDivergences"/> kind computed on exactly the tokens the student sampled (its own replies,
    /// <see cref="GenerationOptions.ReturnTokens"/>), so the student learns where its own outputs go wrong. The teacher is
    /// <see cref="FineTuningOptions.Teacher"/>, a live model (<see cref="DistillationTeachers.FromModel(PretrainedModel, int, int)"/>;
    /// see <see cref="DistillationTeachers.CheckOnPolicy"/>). No reward and no advantage: alone it is the paper's
    /// distillation; beside a policy loss (<see cref="Add"/>) a dense per-token signal next to a sparse reward.
    /// </summary>
    /// <param name="divergence">The kind (<see cref="TeacherDivergences"/>): "reverse" KL(p ‖ q) by default (mode-seeking, the usual on-policy choice), "forward", "jsd".</param>
    /// <param name="temperature">Softens both distributions (1: as they are).</param>
    /// <param name="beta">The generalized Jensen-Shannon weight, for a kind that takes one, in (0, 1).</param>
    /// <exception cref="NotSupportedException">No divergence of that name is registered.</exception>
    public static IFineTuningLoss OnPolicyDistillation(string divergence = TeacherDivergences.Reverse, float temperature = 1f, float beta = 0.5f)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(divergence);
        if (!(temperature > 0f) || float.IsInfinity(temperature))
        {
            throw new ArgumentOutOfRangeException(nameof(temperature), "The temperature is a positive number.");
        }

        var kind = TeacherDivergences.Get(divergence);
        if (kind.UsesBeta && !(beta > 0f && beta < 1f))
        {
            throw new ArgumentOutOfRangeException(nameof(beta), beta, $"The '{kind.Name}' divergence's beta is in (0, 1).");
        }

        return new Loss("on-policy-distillation", $"T² · {kind.Name} divergence from the teacher on the student's own samples, per trained token",
            input => input.TeacherDivergence(temperature, divergence, beta).Sum() * (temperature * temperature / input.StepTokens))
        { NeedsTeacher = true };
    }

    /// <summary>
    /// The sum of two losses, the second weighted: <paramref name="first"/> + <paramref name="weight"/> ·
    /// <paramref name="second"/> (a policy loss with a teacher term, <see cref="OnPolicyDistillation"/>). The weight can
    /// follow a schedule by building a new sum each step.
    /// </summary>
    public static IFineTuningLoss Add(IFineTuningLoss first, IFineTuningLoss second, float weight = 1f)
    {
        ArgumentNullException.ThrowIfNull(first);
        ArgumentNullException.ThrowIfNull(second);
        if (!float.IsFinite(weight))
        {
            throw new ArgumentOutOfRangeException(nameof(weight), weight, "The weight is a finite number.");
        }

        bool both = weight != 0f;                                              // a zero weight never reads the second
        return new Loss($"{first.Name}+{second.Name}", $"{first.Summary}, plus {second.Summary}",
            input => weight == 0f ? first.Compute(input) : first.Compute(input) + second.Compute(input) * weight)
        {
            NeedsPairs = first.NeedsPairs || (both && second.NeedsPairs),
            NeedsReference = first.NeedsReference || (both && second.NeedsReference),
            NeedsTeacher = first.NeedsTeacher || (both && second.NeedsTeacher),
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
    public static IFineTuningLoss Dpo(float beta = 0.1f, float labelSmoothing = 0f)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(beta);
        if (labelSmoothing is < 0f or >= 0.5f)
        {
            throw new ArgumentOutOfRangeException(nameof(labelSmoothing), "Label smoothing is in [0, 0.5).");
        }

        return new Loss("DPO", "direct preference optimization against the reference (the model with its adapters disabled)", input =>
        {
            Require(input, "DPO");
            var ratios = input.SequenceSums(input.LogProbabilities) - input.SequenceSums(input.ReferenceLogProbabilities);
            var margins = (input.Chosen(ratios) - input.Rejected(ratios)) * beta;
            var losses = -Losses.LogSigmoid(margins);
            if (labelSmoothing > 0f)
            {
                losses = losses * (1f - labelSmoothing) - Losses.LogSigmoid(-margins) * labelSmoothing;
            }

            return losses.Sum() / (input.StepSequences / 2);
        })
        { NeedsPairs = true, NeedsReference = true };
    }

    /// <summary>
    /// Odds ratio preference optimization (Hong et al. 2024, "ORPO: Monolithic Preference Optimization without Reference
    /// Model", arXiv:2403.07691): for each pair, the chosen answer's mean negative log-likelihood per token plus
    /// λ · −log σ(log odds_c − log odds_r), with odds = p / (1 − p) and p the answer's mean token probability
    /// (exp of its mean log-probability). Needs no reference model. The likelihood term is averaged per answer, then over the
    /// pairs (TRL averages it over the chosen tokens of a batch; the two agree when the answers have equal lengths).
    /// </summary>
    /// <param name="lambda">Weight of the odds-ratio term (the paper's λ, TRL's beta; 0.1 by default).</param>
    public static IFineTuningLoss Orpo(float lambda = 0.1f)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(lambda);
        return new Loss("ORPO", "odds ratio preference optimization (no reference)", input =>
        {
            Require(input, "ORPO");
            var means = input.SequenceMeans(input.LogProbabilities);
            // log odds = a − log(1 − exp a) for the mean log-probability a < 0 (as TRL's log1p(−exp a)); the small floor keeps
            // an answer the model is already certain of finite.
            var logOdds = means - (1f - means.Exp() + 1e-7f).Log();
            var ratio = Losses.LogSigmoid(input.Chosen(logOdds) - input.Rejected(logOdds));
            var losses = -input.Chosen(means) - ratio * lambda;
            return losses.Sum() / (input.StepSequences / 2);
        })
        { NeedsPairs = true };
    }

    /// <summary>
    /// Simple preference optimization (Meng et al. 2024, "SimPO: Simple Preference Optimization with a Reference-Free
    /// Reward", arXiv:2405.14734): for each pair, −log σ(β · (a_c − a_r) − γ) with a an answer's mean token
    /// log-probability (a length-normalized reward) and γ a target margin. Needs no reference model.
    /// </summary>
    /// <param name="beta">Reward scale (the paper uses 2 to 2.5).</param>
    /// <param name="gamma">Target reward margin (the paper uses about 0.5 to 1.6).</param>
    public static IFineTuningLoss SimPo(float beta = 2f, float gamma = 1f)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(beta);
        return new Loss("SimPO", "simple preference optimization: length-normalized rewards with a margin (no reference)", input =>
        {
            Require(input, "SimPO");
            var means = input.SequenceMeans(input.LogProbabilities);
            var margins = (input.Chosen(means) - input.Rejected(means)) * beta - gamma;
            return -Losses.LogSigmoid(margins).Sum() / (input.StepSequences / 2);
        })
        { NeedsPairs = true };
    }

    private static void Require(FineTuningLossInput input, string name)
    {
        if (!input.Pairs)
        {
            throw new InvalidOperationException(NeedsPairsMessage(name));
        }
    }

    internal static string NeedsPairsMessage(string name) => $"{name} needs preference pairs: train with PreferencePair data (chosen and rejected answers).";

    // A loss as a function and what it needs.
    private sealed class Loss(string name, string summary, Func<FineTuningLossInput, Tensor> compute) : IFineTuningLoss
    {
        public string Name => name;

        public string Summary => summary;

        public bool NeedsPairs { get; init; }

        public bool NeedsReference { get; init; }

        public bool NeedsTeacher { get; init; }

        public Tensor Compute(FineTuningLossInput input) => compute(input);
    }
}
