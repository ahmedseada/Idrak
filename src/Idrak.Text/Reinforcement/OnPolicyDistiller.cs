// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Reinforcement;
using Idrak.Text.Abstractions;

namespace Idrak.Text.Reinforcement;

/// <summary>
/// Settings of <see cref="OnPolicyDistiller.Train"/>. <see cref="Mix"/> and <see cref="Divergence"/> read their variables
/// when set (a value given in code wins).
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
}

/// <summary>What one step of <see cref="OnPolicyDistiller.Train"/> did.</summary>
/// <param name="Step">Steps done.</param>
/// <param name="Steps">Steps planned.</param>
/// <param name="Loss">The step's divergence (the mean over its trained tokens, times T²).</param>
/// <param name="LearningRate">The rate of the step's update.</param>
/// <param name="Sampled">Prompts answered by the student's samples.</param>
/// <param name="FromData">Prompts trained on the data's answers.</param>
public sealed record OnPolicyDistillationProgress(int Step, int Steps, float Loss, float LearningRate, int Sampled, int FromData);

/// <summary>
/// On-policy distillation alone (Agarwal et al. 2024, arXiv:2306.13649): the student answers prompts with its live
/// adapters, the teacher's next-token distributions are computed on exactly the ids it sampled, and the student steps on
/// the divergence (<see cref="FineTuningLosses.OnPolicyDistillation"/>), with a share λ of each step's prompts trained on
/// the data's own answers instead. Only the assistant's tokens train. Beside a reward, the same term is a
/// <see cref="LanguageModelPolicy"/>'s <see cref="LanguageModelPolicyOptions.TeacherWeight"/>.
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
        var model = session.Model;
        var tokenizer = model.Tokenizer ?? throw new InvalidOperationException("The student has no tokenizer.");
        var teacher = session.Options.Teacher
            ?? throw new InvalidOperationException("On-policy distillation needs a teacher: open the session with FineTuningOptions.Teacher (DistillationTeachers.FromModel).");
        DistillationTeachers.CheckOnPolicy(teacher, tokenizer);
        var loss = FineTuningLosses.OnPolicyDistillation(options.Divergence, options.Temperature, options.Beta);
        var encoder = options.Mix > 0f
            ? new ChatTranscriptEncoder(model.JinjaTemplate ?? throw new InvalidOperationException("The student has no Jinja chat template: the data's answers cannot be encoded."), tokenizer)
            : null;
        var chat = session.Generator();
        var random = new Random(options.Seed);
        var order = Enumerable.Range(0, data.Count).ToArray();
        int next = order.Length;
        var losses = new List<float>(options.Steps);
        for (int step = 0; step < options.Steps; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var fromData = new List<ChatTranscript>();
            var sampled = new List<PromptTask>();
            for (int p = 0; p < options.PromptsPerStep; p++)
            {
                if (next == order.Length)
                {
                    random.Shuffle(order);                                 // a new pass over the prompts, in a seeded order
                    next = 0;
                }

                var transcript = data[order[next++]];
                if (encoder is not null && random.NextDouble() < options.Mix && transcript.Messages is [.., { Role: "assistant" }])
                {
                    fromData.Add(transcript);
                }
                else
                {
                    sampled.Add(PromptTask.FromTranscript(transcript));
                }
            }

            var sequences = new List<TrainingSequence>(options.PromptsPerStep);
            foreach (var transcript in fromData)
            {
                if (encoder!.Encode(transcript, options.MaxLength) is { } encoded)
                {
                    sequences.Add(encoded);
                }
            }

            if (sampled.Count > 0)
            {
                var generation = new GenerationOptions
                {
                    Temperature = options.SamplingTemperature, TopK = 0, TopP = 1f, MinP = 0f, RepeatPenalty = 1f, Seed = options.Seed + step,
                    NumPredict = options.MaxNewTokens, NumCtx = model.MaxPositions, ReturnTokens = true,
                };
                var replies = chat.ChatBatch([.. sampled.Select(t => new ChatRequest(t.Prompt, t.Tools, t.Think, generation))], cancellationToken);
                foreach (var reply in replies)
                {
                    var tokens = reply.Tokens ?? throw new InvalidOperationException("The generator gave no token ids (GenerationOptions.ReturnTokens).");
                    sequences.Add(EpisodeTurns.Lay([new ActionRecord(tokens.Reply, new float[tokens.Reply.Length], context: tokens.Prompt)], EpisodeTurns.Turn).Sequences[0]);
                }
            }

            if (sequences.Count == 0)
            {
                continue;
            }

            var stats = session.Step(sequences, loss, cancellationToken);
            losses.Add(stats.Loss);
            progress?.Report(new OnPolicyDistillationProgress(step + 1, options.Steps, stats.Loss, stats.LearningRate, sampled.Count, fromData.Count));
        }

        return losses;
    }
}
