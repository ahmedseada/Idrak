// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Abstraction.Cluster;
using Idrak.Abstraction.Reinforcement;
using Idrak.Text.Abstractions;

namespace Idrak.Text.Reinforcement;

/// <summary>
/// Settings of <see cref="OnPolicyDistiller.Train"/>. <see cref="Mix"/>, <see cref="Divergence"/>, <see cref="Shards"/> and
/// <see cref="Spread"/> read their variables when set (a value given in code wins).
/// </summary>
public sealed record OnPolicyDistillationOptions
{
    /// <summary>The variable of <see cref="Mix"/> (λ, in [0, 1]).</summary>
    public const string MixVariable = "IDRAK_RL_DISTILL_MIX";

    /// <summary>Optimizer steps (each over <see cref="PromptsPerStep"/> prompts).</summary>
    public int Steps { get; init; } = 100;

    /// <summary>Prompts per step.</summary>
    public int PromptsPerStep { get; init; } = 8;

    /// <summary>
    /// λ, the share of a step's prompts trained on the data's own answers instead of the student's samples (Agarwal et
    /// al.'s mix of supervised and on-policy data): 0 (the default) all samples, 1 all answers (<see cref="MixVariable"/>).
    /// </summary>
    public float Mix { get; init; } = LanguageModelPolicyOptions.SettingFloat(MixVariable) ?? 0f;

    /// <summary>The divergence (<see cref="TeacherDivergences"/>; "reverse" by default, <see cref="LanguageModelPolicyOptions.DivergenceVariable"/>).</summary>
    public string Divergence { get; init; } = LanguageModelPolicyOptions.Setting(LanguageModelPolicyOptions.DivergenceVariable) ?? TeacherDivergences.Reverse;

    /// <summary>The generalized Jensen-Shannon weight, in (0, 1).</summary>
    public float Beta { get; init; } = 0.5f;

    /// <summary>The temperature both distributions are softened by in the divergence.</summary>
    public float Temperature { get; init; } = 1f;

    /// <summary>The temperature the student samples its replies at.</summary>
    public float SamplingTemperature { get; init; } = 1f;

    /// <summary>The most tokens a sampled reply has.</summary>
    public int MaxNewTokens { get; init; } = 256;

    /// <summary>The longest data answer with its prompt, in tokens (longer ones are shortened as the tuning data is).</summary>
    public int MaxLength { get; init; } = 2048;

    /// <summary>The seed of the prompts' order, the mix's draws and the samples.</summary>
    public int Seed { get; init; }

    /// <summary>
    /// The shards each step's prompts are cut into (<see cref="RolloutSpread"/>, R3b's rule: a step's prompts numbered 0 to
    /// <see cref="PromptsPerStep"/> − 1): a shard's sampled prompts are answered together, in one call from a seed of the
    /// shard's own (<see cref="RolloutSpread.ShardSeed"/>), and on a cluster each rank answers (and, for λ, encodes) its
    /// shards' prompts only. 0 (the default): one per rank, so one in one process, which samples as before. The same count
    /// gives the same samples on any number of ranks. <see cref="RolloutSpread.ShardsVariable"/>.
    /// </summary>
    public int Shards { get; init; } = ShardsSetting();

    /// <summary>How a step's prompts are cut into shards: <see cref="RolloutSpread.Blocks"/> (the default) or <see cref="RolloutSpread.Interleaved"/> (<see cref="RolloutSpread.SpreadVariable"/>).</summary>
    public string Spread { get; init; } = LanguageModelPolicyOptions.Setting(RolloutSpread.SpreadVariable) is { } spread ? RolloutSpread.Parse(spread) : RolloutSpread.Blocks;

    private static int ShardsSetting() =>
        LanguageModelPolicyOptions.Setting(RolloutSpread.ShardsVariable) is not { } text ? 0
        : int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int shards) && shards >= 0 ? shards
        : throw new ArgumentException($"{RolloutSpread.ShardsVariable}={text} is not a whole number, 0 or more.");
}

/// <summary>What one step of <see cref="OnPolicyDistiller.Train"/> did.</summary>
/// <param name="Step">Steps done.</param>
/// <param name="Steps">Steps planned.</param>
/// <param name="Loss">The step's divergence (the mean over its trained tokens, times T²).</param>
/// <param name="LearningRate">The rate of the step's update.</param>
/// <param name="Sampled">Prompts answered by the student's samples.</param>
/// <param name="FromData">Prompts trained on the data's answers.</param>
public sealed record OnPolicyDistillationProgress(int Step, int Steps, float Loss, float LearningRate, int Sampled, int FromData)
{
    /// <summary>Prompts this rank answered (its shards' share of <see cref="Sampled"/> on a cluster; all of them in one process).</summary>
    public int SampledHere { get; init; }

    /// <summary>
    /// What the step trained on, the same on every rank: the data's answers, then the samples, each in the prompts' order
    /// (each sequence's ids, the prompt's and the reply's, and which train).
    /// </summary>
    public IReadOnlyList<TrainingSequence> Sequences { get; init; } = [];
}

/// <summary>
/// On-policy distillation alone (Agarwal et al. 2024, arXiv:2306.13649): the student answers prompts with its live
/// adapters, the teacher's next-token distributions are computed on exactly the ids it sampled, and the student steps on
/// the divergence (<see cref="FineTuningLosses.OnPolicyDistillation"/>), with a share λ of each step's prompts trained on
/// the data's own answers instead. Only the assistant's tokens train. Beside a reward, the same term is a
/// <see cref="LanguageModelPolicy"/>'s <see cref="LanguageModelPolicyOptions.TeacherWeight"/>.
/// <para>
/// <b>On a cluster</b> (Plan 27, R10: a session opened with <see cref="FineTuningOptions.Cluster"/>, one distiller per rank
/// with the same data and settings): each step's prompts are cut into shards by R3b's rule
/// (<see cref="OnPolicyDistillationOptions.Shards"/>, <see cref="OnPolicyDistillationOptions.Spread"/>), each rank samples
/// (and encodes the data's answers of) its shards' prompts only, the sequences are exchanged so every rank holds the
/// step's whole list in the prompts' order, and the step is the session's data-parallel step (each rank trains its share of
/// the batches, the teacher's distributions computed on that share, the gradients summed). A cancelled token asks the ranks
/// to stop, and they stop at one step boundary (R7's <see cref="IStopAgreement"/>).
/// </para>
/// </summary>
public static class OnPolicyDistiller
{
    /// <summary>
    /// Runs <see cref="OnPolicyDistillationOptions.Steps"/> steps of <paramref name="session"/> (whose
    /// <see cref="FineTuningOptions.Teacher"/> is the live teacher) on the prompts of <paramref name="data"/> (each
    /// transcript's last assistant message its answer), and returns each step's loss.
    /// </summary>
    /// <exception cref="InvalidOperationException">The session has no teacher, or one of another vocabulary (the message names <c>idrak distill --generate</c>).</exception>
    /// <exception cref="NotSupportedException">The teacher is a file (stored logits cannot cover new samples).</exception>
    /// <exception cref="OperationCanceledException">The token was cancelled (in one process, before the next step).</exception>
    /// <exception cref="ClusterRunStoppedException">On a cluster, the ranks agreed to stop (a rank's token cancelled): every rank leaves before the same step, with the steps before it done.</exception>
    public static IReadOnlyList<float> Train(FineTuningSession session, IReadOnlyList<ChatTranscript> data, OnPolicyDistillationOptions? options = null,
        IProgress<OnPolicyDistillationProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(data);
        options ??= new OnPolicyDistillationOptions();
        if (data.Count == 0)
        {
            throw new ArgumentException("On-policy distillation needs prompts to answer.", nameof(data));
        }

        if (!(options.Mix >= 0f && options.Mix <= 1f))
        {
            throw new ArgumentException($"The mix λ is in [0, 1], not {options.Mix} ({OnPolicyDistillationOptions.MixVariable}).", nameof(options));
        }

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.Steps);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.PromptsPerStep);
        if (options.Shards < 0)
        {
            throw new ArgumentException($"The shards are 0 (one per rank) or more, not {options.Shards} ({RolloutSpread.ShardsVariable}).", nameof(options));
        }

        string spread = RolloutSpread.Parse(options.Spread);
        var model = session.Model;
        var tokenizer = model.Tokenizer ?? throw new InvalidOperationException("The student has no tokenizer.");
        var teacher = session.Options.Teacher
            ?? throw new InvalidOperationException("On-policy distillation needs a teacher: open the session with FineTuningOptions.Teacher (DistillationTeachers.FromModel).");
        DistillationTeachers.CheckOnPolicy(teacher, tokenizer);
        var loss = FineTuningLosses.OnPolicyDistillation(options.Divergence, options.Temperature, options.Beta);
        var encoder = options.Mix > 0f
            ? new ChatTranscriptEncoder(model.JinjaTemplate ?? throw new InvalidOperationException("The student has no Jinja chat template: the data's answers cannot be encoded."), tokenizer)
            : null;

        // The split (the same on every rank, from the prompts' numbers 0..P-1 alone, so once for the run): each prompt's shard,
        // and this rank's shards.
        var sync = session.Sync is { WorldSize: > 1 } ranks ? ranks : null;
        int world = sync?.WorldSize ?? 1, rank = sync?.Rank ?? 0, prompts = options.PromptsPerStep;
        int shards = options.Shards > 0 ? options.Shards : world;
        var shardOf = shards == 1 ? new int[prompts] : RolloutSpread.Shards([.. Enumerable.Range(0, prompts)], shards, spread);
        var mineShards = Enumerable.Range(0, shards).Where(s => RolloutSpread.Rank(s, shards, world) == rank).ToArray();

        // On a cluster a cancelled token asks the ranks to stop and they agree at a step boundary (R7): a step's sampling, its
        // exchange and its update run to their end on every rank (their token is cancelled only when this rank gives up).
        var agreement = sync is null ? null : session.Stop;
        agreement?.Reset();
        using var asks = agreement?.Watch(cancellationToken) ?? default;
        var inner = agreement?.GiveUpToken ?? cancellationToken;

        var chat = session.Generator();
        var random = new Random(options.Seed);
        var order = Enumerable.Range(0, data.Count).ToArray();
        int next = order.Length;
        var losses = new List<float>(options.Steps);
        var chosen = new ChatTranscript[prompts];
        var fromData = new bool[prompts];
        var mine = new List<(int Number, TrainingSequence Sequence)>(prompts);
        var asked = new List<int>(prompts);
        for (int step = 0; step < options.Steps; step++)
        {
            if (agreement is null)
            {
                cancellationToken.ThrowIfCancellationRequested();
            }
            else if (agreement.Agree(step))
            {
                throw agreement.Stopped(step);                             // every rank at this same boundary
            }

            // The step's prompts and the mix's draws: every rank draws them all alike.
            int fromDataCount = 0;
            for (int p = 0; p < prompts; p++)
            {
                if (next == order.Length)
                {
                    random.Shuffle(order);                                 // a new pass over the prompts, in a seeded order
                    next = 0;
                }

                var transcript = data[order[next++]];
                chosen[p] = transcript;
                fromData[p] = encoder is not null && random.NextDouble() < options.Mix && transcript.Messages is [.., { Role: "assistant" }];
                fromDataCount += fromData[p] ? 1 : 0;
            }

            // This rank's shards: their data answers encoded, their other prompts answered together a shard, from its seed.
            mine.Clear();
            int sampledHere = 0;
            foreach (int shard in mineShards)
            {
                asked.Clear();
                for (int p = 0; p < prompts; p++)
                {
                    if (shardOf[p] != shard)
                    {
                        continue;
                    }

                    if (!fromData[p])
                    {
                        asked.Add(p);
                    }
                    else if (encoder!.Encode(chosen[p], options.MaxLength) is { } encoded)
                    {
                        mine.Add((p, encoded));
                    }
                }

                if (asked.Count == 0)
                {
                    continue;
                }

                var generation = new GenerationOptions
                {
                    Temperature = options.SamplingTemperature, TopK = 0, TopP = 1f, MinP = 0f, RepeatPenalty = 1f,
                    Seed = RolloutSpread.ShardSeed(options.Seed + step, shard),
                    NumPredict = options.MaxNewTokens, NumCtx = model.MaxPositions, ReturnTokens = true,
                };
                var replies = chat.ChatBatch([.. asked.Select(p => PromptTask.FromTranscript(chosen[p])).Select(t => new ChatRequest(t.Prompt, t.Tools, t.Think, generation))], inner);
                for (int j = 0; j < replies.Count; j++)
                {
                    var tokens = replies[j].Tokens ?? throw new InvalidOperationException("The generator gave no token ids (GenerationOptions.ReturnTokens).");
                    mine.Add((asked[j], EpisodeTurns.Lay([new ActionRecord(tokens.Reply, new float[tokens.Reply.Length], context: tokens.Prompt)], EpisodeTurns.Turn).Sequences[0]));
                }

                sampledHere += asked.Count;
            }

            // Every rank's sequences (gathered in rank order), then in one order on every rank: the data's answers, then
            // the samples, each by prompt number (one process with one shard: the order it always trained in).
            var all = mine;
            if (sync is not null)
            {
                var gathered = RolloutExchange.Gather(sync.Collectives, RolloutExchange.WriteSequences(mine));
                all = [];
                for (int r = 0; r < gathered.Length; r++)
                {
                    all.AddRange(r == rank ? mine : RolloutExchange.ReadSequences(gathered[r]));
                }
            }

            var sequences = all.OrderBy(x => fromData[x.Number] ? 0 : 1).ThenBy(x => x.Number).Select(x => x.Sequence).ToList();
            if (sequences.Count == 0)
            {
                continue;
            }

            var stats = session.Step(sequences, loss, inner);
            losses.Add(stats.Loss);
            progress?.Report(new OnPolicyDistillationProgress(step + 1, options.Steps, stats.Loss, stats.LearningRate, prompts - fromDataCount, fromDataCount)
            {
                SampledHere = sampledHere,
                Sequences = sequences,
            });
        }

        return losses;
    }
}
