// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Abstraction.Cluster;
using Idrak.Abstraction.Reinforcement;
using Idrak.Text.Abstractions;
using Idrak.Text.Generation;

namespace Idrak.Text.Reinforcement;

/// <summary>
/// Settings of a <see cref="LanguageModelPolicy"/>. The ones a run may need to change without code read an environment
/// variable when it is set (the value given in code wins): <see cref="EpisodeTurns.EnvironmentVariable"/>,
/// <see cref="BehaviorVariable"/>, <see cref="TeacherWeightVariable"/>, <see cref="DivergenceVariable"/>,
/// <see cref="SpreadPassesVariable"/>.
/// </summary>
public sealed record LanguageModelPolicyOptions
{
    /// <summary>Behaviour log-probabilities recomputed by the training forward once per rollout batch (the first ratio exactly 1).</summary>
    public const string Recompute = "recompute";

    /// <summary>Behaviour log-probabilities as the sampler drew them (after its temperature and filters).</summary>
    public const string Sampler = "sampler";

    /// <summary>The variable of <see cref="Behavior"/>: recompute or sampler.</summary>
    public const string BehaviorVariable = "IDRAK_RL_BEHAVIOR";

    /// <summary>The variable of <see cref="TeacherWeight"/>.</summary>
    public const string TeacherWeightVariable = "IDRAK_RL_TEACHER_WEIGHT";

    /// <summary>The variable of <see cref="Divergence"/> (a <see cref="TeacherDivergences"/> name).</summary>
    public const string DivergenceVariable = "IDRAK_RL_DIVERGENCE";

    /// <summary>The variable of <see cref="SpreadPasses"/>: on or off.</summary>
    public const string SpreadPassesVariable = "IDRAK_RL_SPREAD_PASSES";

    /// <summary>The most tokens a reply samples.</summary>
    public int MaxNewTokens { get; init; } = 256;

    /// <summary>
    /// The context window replies are generated in; null: the model's. A reply ends where the window does, and the episode
    /// whose reply filled it ends there, truncated (<see cref="ActionRecord.EndsEpisode"/>).
    /// </summary>
    public int? ContextLength { get; init; }

    /// <summary>
    /// Generation settings the policy starts from (stop sequences, chunk size, a loop's stop); its sampling (temperature,
    /// top-k, top-p, min-p, the penalties, the seed) comes from each call's <see cref="ActionSampling"/>, so the policy
    /// samples its distribution as it is. Null: the defaults.
    /// </summary>
    public GenerationOptions? Generation { get; init; }

    /// <summary>The reasoning mode of observations that give none (null: the template's default).</summary>
    public bool? Think { get; init; }

    /// <summary>How a multi-turn episode trains: <see cref="EpisodeTurns.Auto"/> (the default), episode or turn (<see cref="EpisodeTurns.EnvironmentVariable"/>).</summary>
    public string Turns { get; init; } = Setting(EpisodeTurns.EnvironmentVariable) ?? EpisodeTurns.Auto;

    /// <summary>Where the behaviour log-probabilities come from: <see cref="Recompute"/> (the default) or <see cref="Sampler"/> (<see cref="BehaviorVariable"/>).</summary>
    public string Behavior { get; init; } = Setting(BehaviorVariable) ?? Recompute;

    /// <summary>
    /// The weight of a teacher term added to the policy loss (on-policy distillation beside the reward,
    /// <see cref="FineTuningLosses.OnPolicyDistillation"/>; the teacher is the session's <see cref="FineTuningOptions.Teacher"/>);
    /// 0 (the default): none (<see cref="TeacherWeightVariable"/>).
    /// </summary>
    public float TeacherWeight { get; init; } = SettingFloat(TeacherWeightVariable) ?? 0f;

    /// <summary>The teacher term's divergence (<see cref="TeacherDivergences"/>; "reverse" by default, <see cref="DivergenceVariable"/>).</summary>
    public string Divergence { get; init; } = Setting(DivergenceVariable) ?? TeacherDivergences.Reverse;

    /// <summary>The generalized Jensen-Shannon weight of the teacher term, in (0, 1).</summary>
    public float Beta { get; init; } = 0.5f;

    /// <summary>The temperature both distributions of the teacher term are softened by.</summary>
    public float TeacherTemperature { get; init; } = 1f;

    /// <summary>
    /// On a cluster (the session opened with <see cref="FineTuningOptions.Cluster"/>): whether a rollout batch's behaviour
    /// and reference passes (<see cref="FineTuningSession.LogProbabilities"/>, no gradient) are spread over the ranks, each
    /// computing a contiguous run of the batch's sequences (balanced by their positions) and the runs gathered, rather than
    /// every rank computing all of them. On by default (<see cref="SpreadPassesVariable"/>: on or off); in one process it
    /// changes nothing.
    /// </summary>
    public bool SpreadPasses { get; init; } = SettingSwitch(SpreadPassesVariable) ?? true;

    internal static string? Setting(string name) => Environment.GetEnvironmentVariable(name) is { Length: > 0 } value ? value.Trim() : null;

    internal static bool? SettingSwitch(string name) => Setting(name)?.ToLowerInvariant() switch
    {
        null => null,
        "on" or "true" or "1" or "yes" => true,
        "off" or "false" or "0" or "no" => false,
        var other => throw new ArgumentException($"{name}={other} is not on or off."),
    };

    internal static float? SettingFloat(string name) =>
        Setting(name) is not { } text ? null
        : float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out float value) && float.IsFinite(value) ? value
        : throw new ArgumentException($"{name}={text} is not a number.");
}

/// <summary>
/// A language model as a reinforcement-learning policy (Plan 27, R3, contract R-C3): observations are conversations
/// (<see cref="ChatObservation"/>), actions the assistant's replies. It acts through <see cref="ChatGenerator.ChatBatch"/>
/// on the session's live adapters with the token ids recorded (<see cref="GenerationOptions.ReturnTokens"/>), and updates
/// through <see cref="FineTuningSession.Step"/> with the policy loss (<see cref="FineTuningLosses.Policy"/>) on exactly the
/// ids sampled: each episode laid out by <see cref="EpisodeTurns"/> (only the replies trained), each token's advantage its
/// turn's, the reference the base model (adapters off; computed only for a loss that reads it,
/// <see cref="IPolicyLoss.NeedsReference"/>), the behaviour log-probabilities recomputed once per rollout batch
/// by the training forward or taken from the sampler (<see cref="LanguageModelPolicyOptions.Behavior"/>), and a teacher
/// term beside the policy loss when asked (<see cref="LanguageModelPolicyOptions.TeacherWeight"/>). Idrak.Text does not
/// reference Idrak.Reinforcement: the policy meets its trainer and losses through Idrak.Abstraction's contracts.
/// <para>
/// <b>On a cluster</b> (Plan 27, R3b: a session opened with <see cref="FineTuningOptions.Cluster"/>, one policy per rank,
/// each on its rank's device with the same weights): the policy is an <see cref="IRolloutRanks{TObservation, TAction}"/>,
/// so a rollout collector gives each rank its share of a rollout's episodes to generate and step, and
/// <see cref="Gather"/> brings every rank's episodes to every rank in rank order; each rank then lays out the whole batch
/// alike, its behaviour and reference passes spread over the ranks (<see cref="LanguageModelPolicyOptions.SpreadPasses"/>),
/// and <see cref="Update"/> is the session's data-parallel step (each rank trains its share of the step's batches and the
/// gradients are summed).
/// </para>
/// Not thread-safe: one trainer drives it.
/// </summary>
public sealed class LanguageModelPolicy : IPolicy<ChatObservation, ChatMessage>, IRolloutRanks<ChatObservation, ChatMessage>
{
    private readonly bool _ownsSession;
    private readonly FineTuningLoss? _teacher;
    private object? _batch;                                  // the rollout batch laid out
    private List<TrainingSequence>[] _sequences = [];        // its episodes' sequences, with their values and log-probabilities
    private EpisodeLayout[] _layouts = [];
    private bool _hasReference;                              // its sequences carry the reference's log-probabilities
    private bool _disposed;

    /// <summary>A policy over <paramref name="session"/>'s model and adapters.</summary>
    /// <param name="session">The fine-tuning session the policy samples from and trains.</param>
    /// <param name="options">Its settings; null for the defaults (and the variables').</param>
    /// <param name="ownsSession">Whether disposing the policy disposes the session (true by default).</param>
    /// <exception cref="ArgumentException">A setting out of its range (an unknown turn layout or behaviour, a weight that is not finite).</exception>
    /// <exception cref="InvalidOperationException">A teacher weight without a teacher, or a teacher of another vocabulary.</exception>
    /// <exception cref="NotSupportedException">The session's teacher is a file.</exception>
    public LanguageModelPolicy(FineTuningSession session, LanguageModelPolicyOptions? options = null, bool ownsSession = true)
    {
        ArgumentNullException.ThrowIfNull(session);
        options ??= new LanguageModelPolicyOptions();
        _ = EpisodeTurns.Parse(options.Turns);
        if (!options.Behavior.Equals(LanguageModelPolicyOptions.Recompute, StringComparison.OrdinalIgnoreCase)
            && !options.Behavior.Equals(LanguageModelPolicyOptions.Sampler, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"The behaviour log-probabilities come from '{LanguageModelPolicyOptions.Recompute}' or '{LanguageModelPolicyOptions.Sampler}', "
                                        + $"not '{options.Behavior}' ({LanguageModelPolicyOptions.BehaviorVariable}).", nameof(options));
        }

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.MaxNewTokens);
        if (!float.IsFinite(options.TeacherWeight) || options.TeacherWeight < 0f)
        {
            throw new ArgumentException($"The teacher weight is a finite number, 0 or more, not {options.TeacherWeight} ({LanguageModelPolicyOptions.TeacherWeightVariable}).", nameof(options));
        }

        if (options.TeacherWeight > 0f)
        {
            var teacher = session.Options.Teacher
                ?? throw new InvalidOperationException("A teacher weight needs a teacher: open the session with FineTuningOptions.Teacher (DistillationTeachers.FromModel).");
            DistillationTeachers.CheckOnPolicy(teacher, session.Model.Tokenizer ?? throw new InvalidOperationException("The student has no tokenizer."));
            _teacher = FineTuningLosses.OnPolicyDistillation(options.Divergence, options.TeacherTemperature, options.Beta);
        }

        Session = session;
        Options = options;
        _ownsSession = ownsSession;
        Chat = session.Generator(options.ContextLength);
    }

    /// <summary>The session the policy samples from and trains.</summary>
    public FineTuningSession Session { get; }

    /// <summary>The policy's settings.</summary>
    public LanguageModelPolicyOptions Options { get; }

    /// <summary>The chat model the policy acts with (the session's, on its live adapters).</summary>
    public ChatGenerator Chat { get; }

    /// <summary>This rank of the session's cluster; 0 in one process.</summary>
    public int Rank => Session.Sync?.Rank ?? 0;

    /// <summary>The ranks of the session's cluster; 1 in one process.</summary>
    public int WorldSize => Session.Sync?.WorldSize ?? 1;

    /// <inheritdoc/>
    public IStopAgreement? StopAgreement => Session.Sync is { WorldSize: > 1 } ? Session.Stop : null;

    /// <summary>
    /// Every rank's episodes, in rank order, this rank's <paramref name="mine"/> among them (the objects given; the others'
    /// sent as bytes: text conversations, the ids and log-probabilities as sampled, rewards, ends and info). A collective of
    /// the session's cluster: every rank calls it once per rollout. In one process it returns <paramref name="mine"/>.
    /// </summary>
    /// <exception cref="NotSupportedException">A message holds a part that is not text (images are not sent between ranks).</exception>
    public IReadOnlyList<Trajectory<ChatObservation, ChatMessage>> Gather(IReadOnlyList<Trajectory<ChatObservation, ChatMessage>> mine, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(mine);
        cancellationToken.ThrowIfCancellationRequested();
        if (Session.Sync is not { WorldSize: > 1 } sync)
        {
            return [.. mine];
        }

        var gathered = RolloutExchange.Gather(sync.Collectives, RolloutExchange.Write(mine));
        var all = new List<Trajectory<ChatObservation, ChatMessage>>();
        for (int r = 0; r < gathered.Length; r++)
        {
            all.AddRange(r == sync.Rank ? mine : RolloutExchange.Read(gathered[r]));
        }

        return all;
    }

    /// <summary>
    /// A reply to each conversation, generated together (<see cref="ChatGenerator.ChatBatch"/>): sampled at the call's
    /// temperature from its seed with no top-k, top-p, min-p or penalty (greedy: the most likely token), so the
    /// policy's distribution is what samples. Each record holds the prompt ids, the reply ids and their log-probabilities
    /// as sampled. A reply ends where the context window does (<see cref="GenerationOptions.SlideContext"/> off), so its ids
    /// follow its prompt; a reply that filled the window ends its episode (<see cref="ActionRecord.EndsEpisode"/>: the
    /// rollout truncates it), as does a prompt the window no longer holds (its most recent part kept, a token's reply).
    /// Completes synchronously.
    /// </summary>
    /// <exception cref="InvalidOperationException">A reply's window slid all the same (a generator that ignores <see cref="GenerationOptions.SlideContext"/>).</exception>
    public ValueTask<PolicyActions<ChatMessage>> ActAsync(IReadOnlyList<ChatObservation> observations, ActionSampling sampling, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(observations);
        ArgumentNullException.ThrowIfNull(sampling);
        if (observations.Count == 0)
        {
            return ValueTask.FromResult(new PolicyActions<ChatMessage>([], []));
        }

        var generation = (Options.Generation ?? new GenerationOptions()) with
        {
            Temperature = sampling.Greedy ? 0f : sampling.Temperature, TopK = sampling.Greedy ? 1 : 0, TopP = 1f, MinP = 0f,
            RepeatPenalty = 1f, PresencePenalty = 0f, FrequencyPenalty = 0f, Seed = sampling.Seed,
            NumPredict = Options.MaxNewTokens, NumCtx = Options.ContextLength ?? Session.Model.MaxPositions,
            ReturnTokens = true, ReturnLogProbabilities = true, SlideContext = false,
        };
        int window = Math.Clamp(generation.NumCtx, 2, Chat.Generator.ContextLength);   // the window the generator uses
        var requests = observations.Select(o => new ChatRequest((o ?? throw new ArgumentException("An observation is null.", nameof(observations))).Messages,
            o.Tools is { Count: > 0 } ? o.Tools : null, o.Think ?? Options.Think, generation)).ToList();
        var replies = Chat.ChatBatch(requests, cancellationToken);
        var actions = new ChatMessage[replies.Count];
        var records = new ActionRecord[replies.Count];
        for (int i = 0; i < replies.Count; i++)
        {
            var reply = replies[i];
            var tokens = reply.Tokens ?? throw new InvalidOperationException("The generator gave no token ids (GenerationOptions.ReturnTokens).");
            if (reply.Stats is { ContextResets: > 0 })
            {
                throw new InvalidOperationException($"Reply {i} slid its context window ({window} tokens) although GenerationOptions.SlideContext is off: its ids no longer follow its prompt.");
            }

            actions[i] = reply.Message ?? new ChatMessage("assistant", "");
            records[i] = new ActionRecord(tokens.Reply, tokens.LogProbabilities ?? throw new InvalidOperationException("The generator gave no log-probabilities."),
                float.NaN, tokens.Prompt) { EndsEpisode = tokens.Prompt.Length + tokens.Reply.Length >= window };
        }

        return ValueTask.FromResult(new PolicyActions<ChatMessage>(actions, records));
    }

    /// <summary>
    /// The training layout of episodes of <paramref name="batch"/> (all when <paramref name="episodes"/> is null): what
    /// <see cref="Update"/> trains on, before the values and log-probabilities are added.
    /// </summary>
    public IReadOnlyList<EpisodeLayout> Lay(RolloutBatch<ChatObservation, ChatMessage> batch, IReadOnlyList<int>? episodes = null)
    {
        ArgumentNullException.ThrowIfNull(batch);
        return [.. (episodes ?? [.. Enumerable.Range(0, batch.Episodes)]).Select(e => EpisodeTurns.Lay(batch.Trajectories[e].Records, Options.Turns))];
    }

    /// <summary>
    /// One optimizer step on the episodes <paramref name="episodes"/> of <paramref name="batch"/> with <paramref name="loss"/>
    /// (and the teacher term when <see cref="LanguageModelPolicyOptions.TeacherWeight"/> is set). The first update on a
    /// batch lays its episodes out and computes, for all of them at once, the reference log-probabilities and (with
    /// <see cref="LanguageModelPolicyOptions.Recompute"/>) the behaviour ones, before any weight moves; later mini-batches
    /// and epochs of the same batch reuse them.
    /// </summary>
    public UpdateStats Update(RolloutBatch<ChatObservation, ChatMessage> batch, IPolicyLoss loss, IReadOnlyList<int>? episodes = null,
        IReadOnlyDictionary<string, string>? lossOptions = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(batch);
        ArgumentNullException.ThrowIfNull(loss);
        Prepare(batch, loss.NeedsReference(lossOptions));
        IReadOnlyList<int> chosen = episodes ?? [.. Enumerable.Range(0, batch.Episodes)];
        if (chosen.Count == 0 || chosen.Any(e => (uint)e >= (uint)batch.Episodes))
        {
            throw new ArgumentException($"The mini-batch names episodes of the batch's {batch.Episodes} (at least one).", nameof(episodes));
        }

        var sequences = chosen.SelectMany(e => _sequences[e]).ToList();
        var objective = FineTuningLosses.Policy(loss, lossOptions);
        if (_teacher is not null)
        {
            objective = FineTuningLosses.Add(objective, _teacher, Options.TeacherWeight);
        }

        var step = Session.Step(sequences, objective);
        int units = chosen.Sum(e => batch.Trajectories[e].Units);
        var metrics = new Dictionary<string, double>(StringComparer.Ordinal)
        {
            ["learning_rate"] = step.LearningRate,
            ["tokens_per_second"] = step.TokensPerSecond,
            ["sequences"] = sequences.Count,
            ["episodes_as_one_sequence"] = chosen.Count(e => _layouts[e].Mode == EpisodeTurns.Episode) / (double)chosen.Count,
        };
        return new UpdateStats(step.Loss, chosen.Count, units, metrics);
    }

    // Lays out every episode of a batch once (the first update on it): its sequences, each trained token's advantage (its
    // turn's), the reference's log-probabilities when the loss reads them, and the behaviour's (recomputed by the training
    // forward before any step on the batch, or as sampled). The reference (the adapters off) does not move with the steps,
    // so a later update whose loss reads it adds it then.
    private void Prepare(RolloutBatch<ChatObservation, ChatMessage> batch, bool needsReference)
    {
        if (ReferenceEquals(batch, _batch))
        {
            if (needsReference && !_hasReference)
            {
                var laid = _sequences.SelectMany(s => s).ToList();
                var values = LogProbabilities(laid, adapters: false);
                for (int e = 0, k = 0; e < _sequences.Length; e++)
                {
                    for (int i = 0; i < _sequences[e].Count; i++, k++)
                    {
                        _sequences[e][i] = _sequences[e][i] with { ReferenceLogProbabilities = values[k] };
                    }
                }

                _hasReference = true;
            }

            return;
        }

        var layouts = new EpisodeLayout[batch.Episodes];
        var all = new List<TrainingSequence>();
        for (int e = 0; e < batch.Episodes; e++)
        {
            layouts[e] = EpisodeTurns.Lay(batch.Trajectories[e].Records, Options.Turns);
            all.AddRange(layouts[e].Sequences);
        }

        bool recompute = Options.Behavior.Equals(LanguageModelPolicyOptions.Recompute, StringComparison.OrdinalIgnoreCase);
        var behavior = recompute ? LogProbabilities(all, adapters: true) : null;
        var reference = needsReference ? LogProbabilities(all, adapters: false) : null;
        var advantages = batch.Advantages;
        var offsets = batch.Offsets;
        var sequences = new List<TrainingSequence>[batch.Episodes];
        for (int e = 0, k = 0; e < batch.Episodes; e++)
        {
            var trajectory = batch.Trajectories[e];
            // The episode's units in turn order: each one's advantage (its step's) and sampled log-probability.
            var values = new float[trajectory.Units];
            var sampled = new float[trajectory.Units];
            for (int s = 0, at = 0; s < trajectory.Steps; s++)
            {
                var record = trajectory.Records[s];
                values.AsSpan(at, record.Units).Fill(advantages[offsets[e] + s]);
                record.LogProbabilities.CopyTo(sampled, at);
                at += record.Units;
            }

            sequences[e] = new List<TrainingSequence>(layouts[e].Sequences.Count);
            int unit = 0;
            foreach (var sequence in layouts[e].Sequences)
            {
                int count = sequence.TrainedTokens;
                var own = values[unit..(unit + count)];
                double sum = 0;
                foreach (float v in own)
                {
                    sum += v;
                }

                sequences[e].Add(sequence with
                {
                    Value = (float)(sum / Math.Max(1, count)),
                    TokenValues = own,
                    BehaviorLogProbabilities = behavior is null ? sampled[unit..(unit + count)] : behavior[k],
                    ReferenceLogProbabilities = reference?[k],
                });
                unit += count;
                k++;
            }

            if (unit != trajectory.Units)
            {
                throw new InvalidOperationException($"Episode {trajectory.Index}'s sequences train {unit} tokens for its {trajectory.Units} sampled.");
            }
        }

        (_batch, _layouts, _sequences, _hasReference) = (batch, layouts, sequences, needsReference);
    }

    // log p of the trained tokens of `all` with the adapters on or off (FineTuningSession.LogProbabilities). On a cluster
    // with SpreadPasses each rank computes a contiguous run of the sequences, the runs cut where the positions before them
    // reach r/W of all the positions (the same cut on every rank), and every rank's values are gathered: each sequence's
    // count, then its values, as computed (an all-gather copies them).
    private float[][] LogProbabilities(List<TrainingSequence> all, bool adapters)
    {
        if (Session.Sync is not { WorldSize: > 1 } sync || !Options.SpreadPasses)
        {
            return Session.LogProbabilities(all, adapters);
        }

        int world = sync.WorldSize;
        var bounds = new int[world + 1];
        long total = 0;
        foreach (var sequence in all)
        {
            total += sequence.Tokens.Length;
        }

        long before = 0;
        for (int r = 1, i = 0; r < world; r++)
        {
            while (i < all.Count && before * world < total * r)
            {
                before += all[i++].Tokens.Length;
            }

            bounds[r] = i;
        }

        bounds[world] = all.Count;
        int from = bounds[sync.Rank];
        var mine = Session.LogProbabilities(all.GetRange(from, bounds[sync.Rank + 1] - from), adapters);
        var packed = new float[mine.Length + mine.Sum(m => m.Length)];
        for (int i = 0, at = mine.Length; i < mine.Length; i++)
        {
            packed[i] = mine[i].Length;                                    // a sequence's trained tokens: below 2^24, exact
            mine[i].CopyTo(packed, at);
            at += mine[i].Length;
        }

        var gathered = RolloutExchange.Gather(sync.Collectives, packed);
        var result = new float[all.Count][];
        for (int r = 0; r < world; r++)
        {
            int count = bounds[r + 1] - bounds[r];
            var values = gathered[r];
            for (int i = 0, at = count; i < count; i++)
            {
                int length = (int)values[i];
                result[bounds[r] + i] = values[at..(at + length)];
                at += length;
            }
        }

        return result;
    }

    /// <summary>Writes the adapters, the optimizer's state and the steps done (<see cref="FineTuningSession.Checkpoint"/>).</summary>
    public void Save(string path)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Session.Checkpoint(path);
    }

    /// <summary>Loads what <see cref="Save"/> wrote (<see cref="FineTuningSession.Restore"/>).</summary>
    public void Load(string path)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Session.Restore(path);
        _batch = null;
    }

    /// <summary>Releases the session when the policy owns it.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        (_batch, _sequences, _layouts) = (null, [], []);
        if (_ownsSession)
        {
            Session.Dispose();
        }
    }
}
