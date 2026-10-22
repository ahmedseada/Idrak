// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak.Generation;
using Idrak.Text.Generation;

namespace Idrak.Text;

/// <summary>One evaluated conversation.</summary>
/// <param name="Prompt">The conversation up to the answer.</param>
/// <param name="Reference">The reference answer.</param>
/// <param name="Answer">The model's answer.</param>
/// <param name="Score">The answer's score by the report's metric (number, exact, contains: 1 for right, 0 for wrong; f1: the overlap).</param>
/// <param name="Tokens">Tokens generated.</param>
public sealed record EvaluatedAnswer(IReadOnlyList<ChatMessage> Prompt, string Reference, string Answer, double Score, int Tokens);

/// <summary>An evaluation's results.</summary>
/// <param name="Metric">The metric used: its name in <see cref="TuningMetrics"/> (what <see cref="ChatEvaluation.Auto"/> chose, for auto).</param>
/// <param name="Answers">Every conversation, in order.</param>
/// <param name="Loss">Mean loss on the reference answers (teacher-forced), or NaN when not computed.</param>
/// <param name="Duration">Time spent generating.</param>
public sealed record EvaluationReport(string Metric, IReadOnlyList<EvaluatedAnswer> Answers, double Loss, TimeSpan Duration)
{
    /// <summary>The mean score (the share answered right, for right/wrong metrics; lower is better for an error rate).</summary>
    public double Score => Answers.Count == 0 ? 0 : Answers.Average(a => a.Score);

    /// <summary>Mean answer length in tokens.</summary>
    public double MeanTokens => Answers.Count == 0 ? 0 : Answers.Average(a => a.Tokens);

    /// <summary>Generated tokens per second.</summary>
    public double TokensPerSecond => Duration.TotalSeconds > 0 ? Answers.Sum(a => a.Tokens) / Duration.TotalSeconds : 0;
}

/// <summary>
/// Measures a chat model on held-out conversations: the model answers each conversation's last user turn (greedy
/// decoding, so results repeat), and its answer is scored against the conversation's own last assistant message by a
/// metric of <see cref="TuningMetrics"/> (the answer metrics number, exact, contains and f1, the error rates, or one an
/// app registers), or by <see cref="Auto"/>.
/// </summary>
public static class ChatEvaluation
{
    /// <summary>
    /// "auto": <see cref="TuningMetrics.NumberMatch"/> when the references end in "#### n" (gsm8k's layout, told from the
    /// first reference), else <see cref="TuningMetrics.WordF1"/>.
    /// </summary>
    public const string Auto = "auto";

    /// <summary>
    /// Evaluates <paramref name="chat"/> on <paramref name="conversations"/> (rows {"messages": [...]} as datasets give them;
    /// rows without a final assistant message are skipped), scoring by the metric <paramref name="metric"/> (a name in
    /// <see cref="TuningMetrics"/>, or <see cref="Auto"/>). <paramref name="progress"/> gets each answer as it is scored.
    /// With <paramref name="batchSize"/> above 1, that many conversations are answered together
    /// (<see cref="ChatGenerator.ChatBatch"/>: the same greedy answers, generated in one pass per token).
    /// </summary>
    /// <exception cref="NotSupportedException">The metric is not registered (before anything is generated).</exception>
    public static EvaluationReport Run(ChatGenerator chat, IEnumerable<JsonObject> conversations, string metric = Auto,
        int maxNewTokens = 512, bool? think = null, IProgress<EvaluatedAnswer>? progress = null, int contextLength = 4096, CancellationToken cancellationToken = default,
        int batchSize = 1)
    {
        ArgumentNullException.ThrowIfNull(metric);
        var scorer = IsAuto(metric) ? null : TuningMetrics.Get(metric);
        var answers = new List<EvaluatedAnswer>();
        var watch = Stopwatch.StartNew();
        var options = new GenerationOptions { Temperature = 0f, TopK = 1, TopP = 1f, RepeatPenalty = 1f, NumPredict = maxNewTokens, NumCtx = contextLength };
        var pending = new List<(ChatRequest Request, List<ChatMessage> Prompt, string Reference)>();
        void Answer()
        {
            var replies = pending.Count == 1 || batchSize <= 1
                ? [.. pending.Select(p => chat.Chat(p.Request, cancellationToken))]
                : chat.ChatBatch([.. pending.Select(p => p.Request)], cancellationToken);
            for (int i = 0; i < pending.Count; i++)
            {
                string answer = replies[i].Message?.Content ?? "";
                var result = new EvaluatedAnswer(pending[i].Prompt, pending[i].Reference, answer, scorer!.Score(answer, pending[i].Reference).Value,
                    replies[i].Stats?.GeneratedTokens ?? 0);
                answers.Add(result);
                progress?.Report(result);
            }

            pending.Clear();
        }

        foreach (var row in conversations)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var transcript = ChatTranscript.FromJson(row);
            int last = transcript.Messages.Count - 1;
            if (last < 1 || transcript.Messages[last].Role != "assistant" || transcript.Messages[last].ToolCalls is { Count: > 0 })
            {
                continue;
            }

            var prompt = transcript.Messages.Take(last).ToList();
            string reference = transcript.Messages[last].Content;
            scorer ??= TuningMetrics.Get(Choose(reference));                  // auto: the first reference decides
            pending.Add((new ChatRequest(prompt, transcript.Tools.Count > 0 ? transcript.Tools : null, think, options), prompt, reference));
            if (pending.Count >= Math.Max(1, batchSize))
            {
                Answer();
            }
        }

        if (pending.Count > 0)
        {
            Answer();
        }

        return new EvaluationReport(scorer?.Name ?? TuningMetrics.WordF1, answers, double.NaN, watch.Elapsed);
    }

    /// <summary>
    /// The score of <paramref name="answer"/> against <paramref name="reference"/> by the metric <paramref name="metric"/> (a
    /// name in <see cref="TuningMetrics"/>, or <see cref="Auto"/>, chosen from this reference).
    /// </summary>
    /// <exception cref="NotSupportedException">The metric is not registered.</exception>
    public static double Score(string metric, string answer, string reference)
    {
        ArgumentNullException.ThrowIfNull(metric);
        ArgumentNullException.ThrowIfNull(reference);
        return TuningMetrics.Get(IsAuto(metric) ? Choose(reference) : metric).Score(answer, reference).Value;
    }

    private static bool IsAuto(string metric) => metric.Equals(Auto, StringComparison.OrdinalIgnoreCase);

    // Auto's choice for a reference: the final number for gsm8k's layout, else the words' F1.
    private static string Choose(string reference) => TuningMetrics.HasFinalAnswer(reference) ? TuningMetrics.NumberMatch : TuningMetrics.WordF1;
}
