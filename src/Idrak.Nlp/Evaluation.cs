// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Diagnostics;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Idrak.Generation;

namespace Idrak.Nlp;

/// <summary>How a generated answer is compared with the reference answer.</summary>
public enum AnswerMetric
{
    /// <summary><see cref="Number"/> when references end in "#### n" (gsm8k's layout), else <see cref="F1"/>.</summary>
    Auto,

    /// <summary>The final number of each (after "####" when present) is the same.</summary>
    Number,

    /// <summary>The whole answer equals the reference (ignoring case, spacing and final punctuation).</summary>
    Exact,

    /// <summary>The reference appears in the answer (ignoring case and spacing).</summary>
    Contains,

    /// <summary>Word overlap between answer and reference (the F1 score of SQuAD), from 0 to 1.</summary>
    F1,
}

/// <summary>One evaluated conversation.</summary>
/// <param name="Prompt">The conversation up to the answer.</param>
/// <param name="Reference">The reference answer.</param>
/// <param name="Answer">The model's answer.</param>
/// <param name="Score">1 for right, 0 for wrong (F1: the overlap).</param>
/// <param name="Tokens">Tokens generated.</param>
public sealed record EvaluatedAnswer(IReadOnlyList<ChatMessage> Prompt, string Reference, string Answer, double Score, int Tokens);

/// <summary>An evaluation's results.</summary>
/// <param name="Metric">The metric used.</param>
/// <param name="Answers">Every conversation, in order.</param>
/// <param name="Loss">Mean loss on the reference answers (teacher-forced), or NaN when not computed.</param>
/// <param name="Duration">Time spent generating.</param>
public sealed record EvaluationReport(AnswerMetric Metric, IReadOnlyList<EvaluatedAnswer> Answers, double Loss, TimeSpan Duration)
{
    /// <summary>The mean score (the share answered right, for right/wrong metrics).</summary>
    public double Score => Answers.Count == 0 ? 0 : Answers.Average(a => a.Score);

    /// <summary>Mean answer length in tokens.</summary>
    public double MeanTokens => Answers.Count == 0 ? 0 : Answers.Average(a => a.Tokens);

    /// <summary>Generated tokens per second.</summary>
    public double TokensPerSecond => Duration.TotalSeconds > 0 ? Answers.Sum(a => a.Tokens) / Duration.TotalSeconds : 0;
}

/// <summary>
/// Measures a chat model on held-out conversations: the model answers each conversation's last user turn (greedy
/// decoding, so results repeat), and its answer is scored against the conversation's own last assistant message.
/// </summary>
public static partial class ChatEvaluation
{
    /// <summary>
    /// Evaluates <paramref name="chat"/> on <paramref name="conversations"/> (rows {"messages": [...]} as datasets give them;
    /// rows without a final assistant message are skipped). <paramref name="progress"/> gets each answer as it is scored.
    /// With <paramref name="batchSize"/> above 1, that many conversations are answered together
    /// (<see cref="ChatGenerator.ChatBatch"/>: the same greedy answers, generated in one pass per token).
    /// </summary>
    public static EvaluationReport Run(ChatGenerator chat, IEnumerable<JsonObject> conversations, AnswerMetric metric = AnswerMetric.Auto,
        int maxNewTokens = 512, bool? think = null, IProgress<EvaluatedAnswer>? progress = null, int contextLength = 4096, CancellationToken cancellationToken = default,
        int batchSize = 1)
    {
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
                var result = new EvaluatedAnswer(pending[i].Prompt, pending[i].Reference, answer, Score(metric, answer, pending[i].Reference),
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
            if (metric == AnswerMetric.Auto)
            {
                metric = FinalMarker().IsMatch(reference) ? AnswerMetric.Number : AnswerMetric.F1;
            }

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

        return new EvaluationReport(metric == AnswerMetric.Auto ? AnswerMetric.F1 : metric, answers, double.NaN, watch.Elapsed);
    }

    /// <summary>The score of <paramref name="answer"/> against <paramref name="reference"/> under <paramref name="metric"/>.</summary>
    public static double Score(AnswerMetric metric, string answer, string reference) => metric switch
    {
        AnswerMetric.Number => FinalNumber(answer) is { } a && FinalNumber(reference) is { } r && a == r ? 1 : 0,
        AnswerMetric.Exact => Normalize(answer) == Normalize(reference) ? 1 : 0,
        AnswerMetric.Contains => Normalize(answer).Contains(Normalize(reference), StringComparison.Ordinal) ? 1 : 0,
        AnswerMetric.F1 => F1(answer, reference),
        _ => FinalMarker().IsMatch(reference) ? Score(AnswerMetric.Number, answer, reference) : F1(answer, reference),
    };

    /// <summary>The number an answer ends with: after "####" when present, else the last number in the text.</summary>
    public static decimal? FinalNumber(string text)
    {
        var marked = FinalMarker().Match(text);
        string scope = marked.Success ? marked.Groups[1].Value : text;
        var numbers = NumberPattern().Matches(scope);
        if (numbers.Count == 0)
        {
            return null;
        }

        string value = (marked.Success ? numbers[0] : numbers[^1]).Value.Replace(",", "", StringComparison.Ordinal).TrimEnd('.');
        return decimal.TryParse(value, NumberStyles.Number, CultureInfo.InvariantCulture, out var number) ? number : null;
    }

    // string.Join(' ', text.ToLowerInvariant().Split(null, RemoveEmptyEntries)).Trim().TrimEnd('.', '!', '?') in one
    // buffer (the same casing, white space and trimming), with a string only for the result.
    private static string Normalize(string text)
    {
        char[]? rented = null;
        Span<char> buffer = text.Length <= 256 ? stackalloc char[256] : (rented = ArrayPool<char>.Shared.Rent(text.Length));
        try
        {
            var lowered = buffer[..text.AsSpan().ToLowerInvariant(buffer)];
            int length = 0;
            for (int i = 0; i < lowered.Length; i++)
            {
                if (char.IsWhiteSpace(lowered[i]))
                {
                    continue;
                }

                if (length > 0)
                {
                    lowered[length++] = ' ';                                  // one space between runs of other characters
                }

                while (i < lowered.Length && !char.IsWhiteSpace(lowered[i]))
                {
                    lowered[length++] = lowered[i++];
                }
            }

            return new string(lowered[..length].TrimEnd(".!?"));
        }
        finally
        {
            if (rented is not null)
            {
                ArrayPool<char>.Shared.Return(rented);
            }
        }
    }

    // SQuAD's F1 of the lowered words ([\p{L}\p{N}]+), counted by span: no string per word, a key only per distinct
    // reference word.
    private static double F1(string answer, string reference)
    {
        char[] rented = ArrayPool<char>.Shared.Rent(Math.Max(1, answer.Length + reference.Length));
        try
        {
            var a = rented.AsSpan(0, answer.AsSpan().ToLowerInvariant(rented));
            var r = rented.AsSpan(a.Length, reference.AsSpan().ToLowerInvariant(rented.AsSpan(a.Length)));
            var counts = new Dictionary<string, int>();
            var lookup = counts.GetAlternateLookup<ReadOnlySpan<char>>();
            int referenceWords = 0;
            foreach (var match in WordPattern().EnumerateMatches(r))
            {
                referenceWords++;
                CollectionsMarshal.GetValueRefOrAddDefault(lookup, r.Slice(match.Index, match.Length), out _)++;
            }

            int answerWords = 0, common = 0;
            foreach (var match in WordPattern().EnumerateMatches(a))
            {
                answerWords++;
                ref int n = ref CollectionsMarshal.GetValueRefOrNullRef(lookup, a.Slice(match.Index, match.Length));
                if (!Unsafe.IsNullRef(ref n) && n > 0)
                {
                    n--;
                    common++;
                }
            }

            if (answerWords == 0 || referenceWords == 0)
            {
                return answerWords == referenceWords ? 1 : 0;
            }

            if (common == 0)
            {
                return 0;
            }

            double precision = (double)common / answerWords, recall = (double)common / referenceWords;
            return 2 * precision * recall / (precision + recall);
        }
        finally
        {
            ArrayPool<char>.Shared.Return(rented);
        }
    }

    [GeneratedRegex(@"####\s*(.+)$", RegexOptions.Multiline)]
    private static partial Regex FinalMarker();

    [GeneratedRegex(@"-?\d[\d,]*(\.\d+)?")]
    private static partial Regex NumberPattern();

    [GeneratedRegex(@"[\p{L}\p{N}]+")]
    private static partial Regex WordPattern();
}
