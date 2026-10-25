// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Abstraction.Testing;
using Idrak.Optimizers;
using Idrak.Text;

// Plan 27, R0: the optimizers and learning-rate schedules by name as registries in Idrak.Abstraction.Training (they were
// closed lists of names in Idrak.Text), and the answer metrics as tuning metrics (they were a closed enum of
// ChatEvaluation's), with the results they gave before.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] TrainingRegistryGroup =
    [
        ("optimizers: adamw, adam, adamw8bit and sgd by name (OptimizerKinds, the library's) make the updates of their constructors bit for bit, the betas given too, and pass the optimizer suite; an app's kind over a library name takes its place, fails the suite against the library's, and Unregister brings the library's back; an unknown name is refused naming the registered ones and OptimizerKinds.Register", OptimizerKindsByName),
        ("schedules: cosine, linear, constant and wsd by name (LearningRateSchedules, the library's) give the rates of their formulas and pass the schedule suite; an app's schedule over a library name takes its place, fails the suite against the library's, and Unregister brings the library's back; wsd refuses a decay share outside [0, 1] when made; an unknown name is refused naming the registered ones", SchedulesByName),
        ("tuning metrics: number, exact, contains and f1 (higher is better, each answer out of 1, no options) score as ChatEvaluation scored them, after cer and wer in TuningMetrics; their sum over answers is the mean; ChatEvaluation's auto picks number for '#### n' references, else f1; an unknown metric is refused naming the registry", AnswerMetricsAsTuningMetrics),
    ];

    private static void OptimizerKindsByName(Device device)
    {
        Check(OptimizerKinds.Names.SequenceEqual(["adamw", "adam", "adamw8bit", "sgd"]) && OptimizerKinds.Names.All(n => OptimizerKinds.Origin(n) == Overrides.Library),
            $"the library's kinds, in order: {string.Join(", ", OptimizerKinds.Names)}");

        // Each name makes what the closed list made with its constructor (and, given, Pretraining's betas): the same
        // parameters after a few steps, bit for bit; 5,000 elements take adamw8bit's 8-bit moments.
        float[] Trained(Func<IReadOnlyList<Tensor>, Optimizer> make)
        {
            var random = new Random(11);
            float[] start = [.. Enumerable.Range(0, 5000).Select(_ => random.NextSingle() * 2f - 1f)];
            float[] goal = [.. Enumerable.Range(0, 5000).Select(_ => random.NextSingle() * 2f - 1f)];
            using var p = Tensor.Persistent(start, [start.Length], device, requiresGrad: true);
            using var target = Tensor.Persistent(goal, [goal.Length], device);
            using var optimizer = make([p]);
            for (int step = 0; step < 4; step++)
            {
                optimizer.ZeroGrad();
                using var scope = new TensorScope();
                (p - target).Square().Sum().Backward();
                optimizer.Step();
            }

            return p.ToArray();
        }

        const float Rate = 0.01f, Decay = 0.01f, Momentum = 0.9f;
        var settings = new OptimizerSettings(Rate, Decay, Momentum);
        var constructors = new Dictionary<string, Func<IReadOnlyList<Tensor>, Optimizer>>
        {
            ["adamw"] = ps => new AdamW(ps, Rate, weightDecay: Decay),
            ["adam"] = ps => new Adam(ps, Rate, weightDecay: Decay),
            ["adamw8bit"] = ps => new AdamW8Bit(ps, Rate, weightDecay: Decay),
            ["sgd"] = ps => new Sgd(ps, Rate, Momentum, Decay),
        };
        foreach (var (name, construct) in constructors)
        {
            AssertClose(Trained(construct), Trained(OptimizerKinds.Factory(name, settings)), 0f, $"{name} by name");
        }

        var betas = settings with { Beta1 = 0.85f, Beta2 = 0.95f, Epsilon = 1e-7f };
        AssertClose(Trained(ps => new AdamW8Bit(ps, Rate, 0.85f, 0.95f, 1e-7f, Decay)), Trained(OptimizerKinds.Factory("ADAMW8BIT", betas)), 0f, "adamw8bit with betas, any case");
        AssertClose(Trained(ps => new Adam(ps, Rate, 0.85f, 0.95f, 1e-7f, Decay)), Trained(OptimizerKinds.Factory("adam", betas)), 0f, "adam with betas");

        foreach (string name in OptimizerKinds.Names)
        {
            var report = Conformance.CheckOptimizerKind(name, device: device, options: new ContractCheckOptions { RandomCases = 4 });
            Check(report.Passed && report.Entries.Any(e => e.Name == "the steps lower the loss" && e.Status == CheckStatus.Passed), $"{name}: the optimizer suite:\n{report}");
        }

        // An app's sgd (without momentum) over the library's.
        OptimizerKinds.Register(new PlainSgd());
        try
        {
            Check(OptimizerKinds.Get("sgd") is PlainSgd && OptimizerKinds.Default("sgd") is not PlainSgd && OptimizerKinds.Origin("sgd") == typeof(Tests).Assembly.GetName().Name
                  && OptimizerKinds.Names.SequenceEqual(["adamw", "adam", "adamw8bit", "sgd"]), "an app's kind takes the library's place");
            var differs = Conformance.CheckOptimizerKind("sgd", device: device, options: new ContractCheckOptions { RandomCases = 0 });
            Check(!differs.Passed && differs.Failures.All(f => f.Check == "the reference's values"), $"the app's kind against the library's:\n{differs}");
        }
        finally
        {
            Check(OptimizerKinds.Unregister("sgd") && OptimizerKinds.Get("sgd") == OptimizerKinds.Default("sgd") && OptimizerKinds.Origin("sgd") == Overrides.Library,
                "Unregister brings the library's back");
        }

        ThrowsNaming<NotSupportedException>(() => OptimizerKinds.Factory("nope", settings), "an unknown optimizer",
            "'nope' is not registered (registered: adamw, adam, adamw8bit, sgd)", "OptimizerKinds.Register");
        Check(!OptimizerKinds.Unregister("adamw") && OptimizerKinds.Find("adamw") is not null, "a library kind is never removed");
    }

    private static void SchedulesByName(Device device)
    {
        _ = device;
        Check(LearningRateSchedules.Names.SequenceEqual(["cosine", "linear", "constant", "wsd"]) && LearningRateSchedules.Names.All(n => LearningRateSchedules.Origin(n) == Overrides.Library),
            $"the library's schedules, in order: {string.Join(", ", LearningRateSchedules.Names)}");

        // The rates step by step (10 steps, 2 of warm-up), from each formula.
        static float[] Rates(Func<Optimizer, int, LearningRateScheduler> schedule, int steps = 10)
        {
            using var p = Tensor.Persistent([0f], [1], Device.Cpu, requiresGrad: true);
            using var optimizer = new Sgd([p], 1f);
            var s = schedule(optimizer, steps);
            var rates = new float[steps];
            for (int i = 0; i < steps; i++)
            {
                rates[i] = optimizer.LearningRate;
                s.Step();
            }

            return rates;
        }

        float[] cosine = [.. Enumerable.Range(0, 10).Select(e => e < 2 ? (e + 1) / 3f : (float)(0.1 + (1 - 0.1) * 0.5 * (1 + Math.Cos(Math.PI * Math.Min(1.0, (e - 2) / 8.0)))))];
        AssertClose(cosine, Rates(LearningRateSchedules.Factory("Cosine", new LearningRateScheduleSettings(0.2f, 0.1f))), 1e-6f, "cosine to 0.1");
        AssertClose([1 / 3f, 2 / 3f, 1f, 0.8875f, 0.775f, 0.6625f, 0.55f, 0.4375f, 0.325f, 0.2125f],
            Rates(LearningRateSchedules.Factory(LearningRateSchedules.Linear, new LearningRateScheduleSettings(0.2f, 0.1f))), 1e-6f, "linear to 0.1");
        AssertClose([1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f], Rates(LearningRateSchedules.Factory(LearningRateSchedules.Constant, new LearningRateScheduleSettings(0.03f))),
            1e-6f, "constant: a warm-up share that rounds to no step");
        AssertClose([1 / 3f, 2 / 3f, 1f, 1f, 1f, 1f, 1f, 2 / 3f, 1 / 3f, 0f], Rates(LearningRateSchedules.Factory(LearningRateSchedules.WarmupStableDecay, new LearningRateScheduleSettings(0.2f, 0f, 0.3f))),
            1e-6f, "warm-up, stable, decay");
        AssertClose(Rates(LearningRateSchedules.Factory(LearningRateSchedules.Cosine), 100), Rates(LearningRateSchedules.Factory(LearningRateSchedules.Cosine, new LearningRateScheduleSettings()), 100),
            0f, "no settings: the defaults (a warm-up of 0.03, to 0, decay 0.2)");

        foreach (string name in LearningRateSchedules.Names)
        {
            var report = Conformance.CheckLearningRateSchedule(name);
            Check(report.Passed && report.Entries.Any(e => e.Name == "made again, the same rates" && e.Status == CheckStatus.Passed), $"{name}: the schedule suite:\n{report}");
        }

        // An app's constant schedule (half the rate after the warm-up) over the library's.
        LearningRateSchedules.Register(new HalfConstant());
        try
        {
            Check(LearningRateSchedules.Get("constant") is HalfConstant && LearningRateSchedules.Origin("constant") == typeof(Tests).Assembly.GetName().Name,
                "an app's schedule takes the library's place");
            var differs = Conformance.CheckLearningRateSchedule("constant", options: new ContractCheckOptions { RandomCases = 0 });
            Check(!differs.Passed && differs.Failures.All(f => f.Check == "the reference's rates"), $"the app's schedule against the library's:\n{differs}");
        }
        finally
        {
            Check(LearningRateSchedules.Unregister("constant") && LearningRateSchedules.Origin("constant") == Overrides.Library, "Unregister brings the library's back");
        }

        Check(Failure<ArgumentOutOfRangeException>(() => LearningRateSchedules.Factory(LearningRateSchedules.WarmupStableDecay, new LearningRateScheduleSettings(DecayFraction: 1.5f)))
            .Message.Contains("[0, 1]", StringComparison.Ordinal), "wsd refuses a decay share above 1 when made");
        ThrowsNaming<NotSupportedException>(() => LearningRateSchedules.Factory("step"), "an unknown schedule",
            "'step' is not registered (registered: cosine, linear, constant, wsd)", "LearningRateSchedules.Register");
    }

    private static void AnswerMetricsAsTuningMetrics(Device device)
    {
        _ = device;
        Check(TuningMetrics.Names.Take(6).SequenceEqual(["cer", "wer", "number", "exact", "contains", "f1"]), $"the library's metrics: {string.Join(", ", TuningMetrics.Names)}");
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
            var scored = TuningMetrics.Get(metric).Score(answer, reference);
            Check(Math.Abs(scored.Value - score) < 1e-12 && scored.Denominator == 1 && Math.Abs(ChatEvaluation.Score(metric.ToUpperInvariant(), answer, reference) - score) < 1e-12,
                $"{metric}: '{answer}' against '{reference}' scored {scored}, expected {score}");
        }

        foreach (string name in new[] { TuningMetrics.NumberMatch, TuningMetrics.ExactMatch, TuningMetrics.ContainsMatch, TuningMetrics.WordF1 })
        {
            var metric = TuningMetrics.Get(name);
            Check(!metric.LowerIsBetter && metric.Keys.Count == 0 && TuningMetrics.Origin(name) == Overrides.Library && metric.Name == name, $"{name}: higher is better, no options");
            ThrowsNaming<ArgumentException>(() => metric.Score("a", "a", new Dictionary<string, string> { ["ignore_case"] = "true" }), $"{name} with an option", "takes no options", "'ignore_case'");
            Check(metric.Score("1", "1", new Dictionary<string, string>()).Value == 1, $"{name}: no options given is fine");
        }

        var corpus = TuningMetrics.Score("exact", [("a", "a"), ("b", "c"), ("Yes.", "yes")]);
        Check(corpus.Numerator == 2 && corpus.Denominator == 3 && Math.Abs(corpus.Value - 2.0 / 3) < 1e-12, $"the sum over answers is the share right: {corpus}");
        Check(ChatEvaluation.Score(ChatEvaluation.Auto, "so it is 14", "#### 14") == 1 && Math.Abs(ChatEvaluation.Score("AUTO", "the cat sat", "the cat ran") - 2.0 / 3) < 1e-12,
            "auto: number for #### references, else f1");
        Check(TuningMetrics.HasFinalAnswer("work\n#### 5") && !TuningMetrics.HasFinalAnswer("5") && !TuningMetrics.HasFinalAnswer("####"), "the final-answer marker");
        ThrowsNaming<NotSupportedException>(() => ChatEvaluation.Score("bleu", "a", "a"), "an unknown metric", "'bleu' is not registered (registered: cer, wer, number, exact, contains, f1", "TuningMetrics.Register");
    }

    // An app's "sgd": the library's without momentum.
    private sealed class PlainSgd : IOptimizerKind
    {
        public string Name => "sgd";

        public string Summary => "SGD without momentum";

        public Optimizer Create(OptimizerRequest request) => new Sgd(request.Parameters, request.Settings.LearningRate, 0f, request.Settings.WeightDecay);
    }

    // An app's "constant": half the optimizer's rate after the warm-up.
    private sealed class HalfConstant : ILearningRateSchedule
    {
        public string Name => "constant";

        public string Summary => "half the rate";

        public Func<Optimizer, int, LearningRateScheduler> Create(LearningRateScheduleSettings settings) =>
            (optimizer, steps) => new LambdaSchedule(optimizer, (step, rate) => step < (int)Math.Round(settings.WarmupFraction * steps) ? rate : rate / 2);
    }
}
