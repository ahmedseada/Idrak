// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Data;
using Idrak.Generation;
using Idrak.Models;
using Idrak.Nlp.Abstractions;

namespace Idrak.Nlp;

/// <summary>One generated answer of <see cref="TuningAnswerScorer.Score"/>, scored against its reference.</summary>
/// <param name="Prompt">The conversation before the reference answer (its images included).</param>
/// <param name="Reference">The expected answer (the conversation's last assistant message).</param>
/// <param name="Answer">The answer generated (greedy).</param>
/// <param name="Score">Its score by the metric (<see cref="ITuningMetric.Score"/>).</param>
/// <param name="Tokens">Tokens generated.</param>
public sealed record TuningAnswer(IReadOnlyList<ChatMessage> Prompt, string Reference, string Answer, TuningScore Score, int Tokens);

/// <summary>What <see cref="TuningAnswerScorer.Score"/> measured: every answer's score and their score together.</summary>
/// <param name="Metric">The metric's name (<see cref="TuningMetrics"/>).</param>
/// <param name="LowerIsBetter">Whether a lower score is better (an error rate).</param>
/// <param name="Score">The answers' score together (<see cref="TuningScore.Sum"/>: errors over the references' lengths).</param>
/// <param name="Answers">Each answer, in the order of the conversations.</param>
/// <param name="Duration">The time it took (generation and scoring).</param>
public sealed record TuningAnswerReport(string Metric, bool LowerIsBetter, TuningScore Score, IReadOnlyList<TuningAnswer> Answers, TimeSpan Duration)
{
    /// <summary>Tokens generated per answer, on average.</summary>
    public double MeanTokens => Answers.Count == 0 ? 0 : Answers.Average(a => a.Tokens);

    /// <summary>"cer 0.0625 on 8 answers".</summary>
    public override string ToString() => FormattableString.Invariant($"{Metric} {Score.Value:F4} on {Answers.Count} answers");
}

/// <summary>
/// A held-out score that is not a loss: the model answers each conversation's prompt (greedy decoding, so results repeat;
/// its images included, read and prepared as in training) and the answer is scored against the conversation's own last
/// assistant message by a metric of <see cref="TuningMetrics"/> (character or word error rate, or one registered). Set it
/// as <see cref="FineTuningOptions.Answers"/> to score every <see cref="Every"/> steps of a fine-tune (the report comes
/// with the step's <see cref="FineTuningProgress.Answers"/>), or call <see cref="Score"/> before and after training.
/// <para>
/// Answers are generated through the model's own chat template and, for conversations with images, through the
/// <see cref="TuningVision"/> the sequences were encoded with: its encoder (a projector being trained is read as it is
/// now), its prompt format and attention rule, and its preparation (<see cref="TuningVision.Images"/>: transforms and
/// vision options). The key/value cache is sized per request (<see cref="ContextLength"/>) and released after each
/// scoring; the model's training mode is put back.
/// </para>
/// </summary>
/// <example>
/// <code>
/// var scorer = new TuningAnswerScorer(validation, TuningMetrics.CharacterErrorRate, samples: 32) { Every = 100 };
/// Console.WriteLine(scorer.Score(model, vision));                               // before training
/// FineTuner.Train(model, train, evaluation, new FineTuningOptions { Vision = vision, Answers = scorer }, "adapter",
///     new Progress&lt;FineTuningProgress&gt;(p => { if (p.Answers is { } a) Console.WriteLine($"step {p.Step}: {a}"); }));
/// </code>
/// </example>
public sealed class TuningAnswerScorer
{
    private readonly ITuningMetric _metric;

    /// <summary>
    /// A scorer of the conversations in <paramref name="transcripts"/> that end with an assistant message (text, no tool
    /// calls): the messages before it are the prompt, its text the reference.
    /// </summary>
    /// <param name="transcripts">Held-out conversations (their images as image parts).</param>
    /// <param name="metric">The metric's name in <see cref="TuningMetrics"/>.</param>
    /// <param name="samples">Score the first this many conversations only (0: all of them).</param>
    /// <exception cref="NotSupportedException">The metric is not registered (the registry's message).</exception>
    public TuningAnswerScorer(IEnumerable<ChatTranscript> transcripts, string metric = TuningMetrics.CharacterErrorRate, int samples = 0)
    {
        ArgumentNullException.ThrowIfNull(transcripts);
        ArgumentException.ThrowIfNullOrWhiteSpace(metric);
        ArgumentOutOfRangeException.ThrowIfNegative(samples);
        _metric = TuningMetrics.Get(metric);
        Metric = _metric.Name;
        var items = transcripts.Select(Split).OfType<(IReadOnlyList<ChatMessage> Prompt, string Reference, IReadOnlyList<ToolDefinition> Tools, bool? Think)>();
        Items = [.. samples > 0 ? items.Take(samples) : items];
    }

    /// <summary>The metric's name.</summary>
    public string Metric { get; }

    /// <summary>Whether a lower score is better (the metric's <see cref="ITuningMetric.LowerIsBetter"/>).</summary>
    public bool LowerIsBetter => _metric.LowerIsBetter;

    /// <summary>The metric's options (its <see cref="ITuningMetric.Keys"/>), or null for its defaults.</summary>
    public IReadOnlyDictionary<string, string>? MetricOptions { get; init; }

    /// <summary>
    /// In <see cref="FineTuner.Train(PretrainedModel, IReadOnlyList{TrainingSequence}, IReadOnlyList{TrainingSequence}?, FineTuningOptions, string?, IProgress{FineTuningProgress}?, CancellationToken, Action{string}?)"/>:
    /// score every this many optimizer steps; 0 (the default): at the end of each epoch.
    /// </summary>
    public int Every { get; init; }

    /// <summary>The most tokens generated per answer.</summary>
    public int MaxNewTokens { get; init; } = 512;

    /// <summary>The context window of each generation (prompt, image tokens and answer; at most the model's), which sizes its key/value cache.</summary>
    public int ContextLength { get; init; } = 4096;

    /// <summary>The reasoning mode passed to the template, when the conversation sets none (null: the template's default).</summary>
    public bool? Think { get; init; }

    /// <summary>
    /// Receives each answer as it is scored when <see cref="Score"/> is given no progress of its own (as when the tuner
    /// scores while training), or null.
    /// </summary>
    public IProgress<TuningAnswer>? Progress { get; set; }

    /// <summary>The conversations scored: each prompt and reference answer.</summary>
    public IReadOnlyList<(IReadOnlyList<ChatMessage> Prompt, string Reference, IReadOnlyList<ToolDefinition> Tools, bool? Think)> Items { get; }

    /// <summary>How many conversations are scored.</summary>
    public int Count => Items.Count;

    /// <summary>Whether training scores after optimizer step <paramref name="step"/> (<see cref="Every"/>; the last step of an epoch when it is 0).</summary>
    public bool IsDue(int step, bool lastOfEpoch) => Count > 0 && (Every > 0 ? step % Every == 0 : lastOfEpoch);

    /// <summary>
    /// Generates an answer to every prompt with <paramref name="model"/> (greedy, its adapters as they are now) and scores
    /// it. Prompts with images are read through <paramref name="vision"/> (made from the model for this call when null).
    /// </summary>
    /// <param name="model">The model (a chat template and a tokenizer).</param>
    /// <param name="vision">The image side the training sequences were encoded with; null: one made from the model when a prompt holds images.</param>
    /// <param name="progress">Receives each answer as it is scored (<see cref="Progress"/> when null).</param>
    /// <param name="cancellationToken">Stops between tokens.</param>
    /// <exception cref="InvalidOperationException">The model has no chat template, or a prompt holds images and the model has no vision part.</exception>
    public TuningAnswerReport Score(PretrainedModel model, TuningVision? vision = null, IProgress<TuningAnswer>? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(model);
        progress ??= Progress;
        if (vision is not null && !ReferenceEquals(vision.Model, model))
        {
            throw new ArgumentException("The vision side was made for another model (TuningVision.Create with this model).", nameof(vision));
        }

        bool images = Items.Any(i => i.Prompt.Any(m => m.Parts.Any(p => p is ChatImage)));
        using var owned = images && vision is null ? TuningVision.Create(model) : null;
        vision ??= owned;
        var template = model.ChatTemplate ?? throw new InvalidOperationException("The model has no chat template: its answers cannot be generated from conversations.");
        bool training = model.Network.IsTraining;
        var watch = Stopwatch.StartNew();
        var answers = new List<TuningAnswer>(Items.Count);
        int context = Math.Clamp(ContextLength, 2, model.MaxPositions);
        var generator = model.CreateGenerator(KeyValueFormat.Float32, context);
        generator.KeepCache = false;                                              // nothing held between answers or steps
        try
        {
            var chat = new ChatGenerator(generator, template)
            {
                Images = images ? new ChatImages(vision!.Encoder, vision.Format, vision.Attention) { Transforms = vision.Images.Pipeline } : null,
            };
            var options = new GenerationOptions { Temperature = 0f, TopK = 1, TopP = 1f, RepeatPenalty = 1f, NumPredict = MaxNewTokens, NumCtx = context };
            var visionOptions = vision is { Images.VisionOptions.Count: > 0 } ? vision.Images.VisionOptions : null;
            using (Autograd.NoGrad())
            {
                foreach (var (prompt, reference, tools, think) in Items)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var request = new ChatRequest(prompt, tools.Count > 0 ? tools : null, think ?? Think, options) { VisionOptions = visionOptions };
                    var reply = chat.Chat(request, cancellationToken);
                    string answer = reply.Message?.Content ?? "";
                    var scored = new TuningAnswer(prompt, reference, answer, _metric.Score(answer, reference, MetricOptions), reply.Stats?.GeneratedTokens ?? 0);
                    answers.Add(scored);
                    progress?.Report(scored);
                }
            }
        }
        finally
        {
            generator.ReleaseCache();
            if (training)
            {
                model.Network.Train();
            }
        }

        return new TuningAnswerReport(Metric, LowerIsBetter, TuningScore.Sum(answers.Select(a => a.Score)), answers, watch.Elapsed);
    }

    // The prompt and the reference of a conversation that ends with an assistant's text answer; null for any other.
    private static (IReadOnlyList<ChatMessage> Prompt, string Reference, IReadOnlyList<ToolDefinition> Tools, bool? Think)? Split(ChatTranscript transcript)
    {
        int last = transcript.Messages.Count - 1;
        if (last < 1 || transcript.Messages[last] is not { Role: "assistant" } answer || answer.ToolCalls is { Count: > 0 } || answer.Content.Length == 0)
        {
            return null;
        }

        return ([.. transcript.Messages.Take(last)], answer.Content, transcript.Tools, transcript.Think);
    }
}
