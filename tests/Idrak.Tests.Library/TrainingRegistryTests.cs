// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Abstraction.Testing;
using Idrak.Optimizers;
using Idrak.Text;

// Plan 27, R0: the answer metrics as tuning metrics (they were a closed enum of ChatEvaluation's), with the results they
// gave before.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] TrainingRegistryGroup =
    [
        ("tuning metrics: number, exact, contains and f1 (AnswerMatchMetric; higher is better, each answer out of 1) score as ChatEvaluation scored them; their sum over answers is the mean; the final-answer marker", AnswerMetricsAsTuningMetrics),
    ];

    internal static void AnswerMetricsAsTuningMetrics(Device device)
    {
        _ = device;
        string longAnswer = string.Concat(Enumerable.Repeat("Word  ", 80)) + "END!", longReference = string.Concat(Enumerable.Repeat("word ", 80)) + "end";
        (string Metric, string Answer, string Reference, double Score)[] cases =
        [
            ("number", "Tom gets 20 - 12 = $8 back.", "20-12=8\n#### 8", 1), ("number", "#### 9", "#### 8", 0), ("number", "so 1,200.", "#### 1200", 1),
            ("number", "no number", "#### 3", 0), ("number", "-3.5 then 7", "7", 1), ("number", "####\n42", "#### 42", 1),
            ("exact", "  Paris. ", "paris", 1), ("exact", "Hello \t World!", "hello world", 1), ("exact", "Paris, France", "paris", 0), ("exact", "", "", 1),
            ("exact", longAnswer, longReference, 1),
            ("contains", "It is Paris, France", "paris", 1), ("contains", "It is   PARIS", "is paris", 1), ("contains", "London", "paris", 0), ("contains", "anything", "", 1),
            ("f1", "the cat sat", "the cat ran", 2.0 / 3), ("f1", "dog", "cat", 0), ("f1", "The cat", "the CAT the", 0.8), ("f1", "", "", 1), ("f1", "!!", "a", 0),
            ("f1", "قال الولد كلمة", "قال الولد", 0.8),
        ];
        foreach (var (metric, answer, reference, score) in cases)
        {
            var scored = new AnswerMatchMetric(metric).Score(answer, reference);
            Check(Math.Abs(scored.Value - score) < 1e-12 && scored.Denominator == 1, $"{metric}: '{answer}' against '{reference}' scored {scored}, expected {score}");
        }

        foreach (string name in new[] { TuningMetrics.NumberMatch, TuningMetrics.ExactMatch, TuningMetrics.ContainsMatch, TuningMetrics.WordF1 })
        {
            var metric = new AnswerMatchMetric(name);
            Check(!metric.LowerIsBetter && metric.Name == name, $"{name}: higher is better");
        }

        var exact = new AnswerMatchMetric(TuningMetrics.ExactMatch);
        var corpus = TuningScore.Sum(new[] { ("a", "a"), ("b", "c"), ("Yes.", "yes") }.Select(p => exact.Score(p.Item1, p.Item2)));
        Check(corpus.Numerator == 2 && corpus.Denominator == 3 && Math.Abs(corpus.Value - 2.0 / 3) < 1e-12, $"the sum over answers is the share right: {corpus}");
        Check(TuningMetrics.HasFinalAnswer("work\n#### 5") && !TuningMetrics.HasFinalAnswer("5") && !TuningMetrics.HasFinalAnswer("####"), "the final-answer marker");
    }
}
