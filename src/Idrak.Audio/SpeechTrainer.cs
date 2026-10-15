// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Models;
using Idrak.Nlp;
using Idrak.Nlp.Abstractions;
using Idrak.Optimizers;
using Idrak.Abstraction.Generation;

namespace Idrak.Audio;

/// <summary>
/// How <see cref="SpeechTrainer"/> trains a speech model (whatever its objective). Every value is a setting; the defaults are a starting
/// point for a recognizer trained from scratch, not values tied to a device.
/// </summary>
public sealed record SpeechTrainingOptions
{
    /// <summary>Optimizer steps to run in all (a resumed run goes on to this many).</summary>
    public int Steps { get; init; } = 1000;

    /// <summary>
    /// Audio seconds in a batch, padding included (its utterances times the longest): utterances are sorted by length within
    /// windows of <see cref="BucketUtterances"/> and cut into batches of at most this, so a batch's padding is small. A step
    /// whose device runs out of memory is run again in micro-batches of half as many seconds (their gradients added up), and
    /// so on: the batches, and so the training, do not change, only how many go through the device at once.
    /// </summary>
    public double BatchSeconds { get; init; } = 160;

    /// <summary>Batches whose gradients are added up before each optimizer step.</summary>
    public int GradientAccumulation { get; init; } = 1;

    /// <summary>Utterances sorted by length together (a window of the pass's seeded order) before they are cut into batches.</summary>
    public int BucketUtterances { get; init; } = 512;

    /// <summary>The peak learning rate.</summary>
    public float LearningRate { get; init; } = 1e-3f;

    /// <summary>The share of the steps the rate warms up over.</summary>
    public float WarmupFraction { get; init; } = 0.05f;

    /// <summary>The rate the schedule decays to.</summary>
    public float MinLearningRate { get; init; }

    /// <summary>The rate over the steps: a <see cref="FineTuningSchedules"/> name.</summary>
    public string Schedule { get; init; } = "cosine";

    /// <summary>The optimizer: a <see cref="FineTuningOptimizers"/> name ("adamw", "adam", "adamw8bit", "sgd").</summary>
    public string Optimizer { get; init; } = "adamw";

    /// <summary>Weight decay (decoupled for AdamW).</summary>
    public float WeightDecay { get; init; } = 0.01f;

    /// <summary>SGD's momentum.</summary>
    public float Momentum { get; init; } = 0.9f;

    /// <summary>Run the optimizer's update on the CPU with its state in system memory (<see cref="Idrak.Optimizers.HostOptimizer"/>).</summary>
    public bool HostOptimizer { get; init; }

    /// <summary>The gradients' global norm is clipped to this before each step (0: no clipping).</summary>
    public float MaxGradientNorm { get; init; } = 1f;

    /// <summary>Hold the activations the backward pass reads as bfloat16 between the passes (true); null or false: float32.</summary>
    public bool? BFloat16Activations { get; init; }

    /// <summary>Recompute each block's activations in the backward pass instead of keeping them (activation checkpointing).</summary>
    public bool Checkpointing { get; init; } = true;

    /// <summary>A checkpoint every this many steps (0: at the end only).</summary>
    public int SaveEvery { get; init; }

    /// <summary>Keep only the newest this many checkpoints (0: all).</summary>
    public int KeepCheckpoints { get; init; }

    /// <summary>Score the evaluation source every this many steps (0: at the end only).</summary>
    public int EvaluateEvery { get; init; }

    /// <summary>The evaluation's score: a <see cref="TuningMetrics"/> name ("cer", "wer").</summary>
    public string Metric { get; init; } = TuningMetrics.CharacterErrorRate;

    /// <summary>The evaluation's decoder: "greedy" or "beam" (for CTC, a <see cref="CtcDecoders"/> name).</summary>
    public string Decoder { get; init; } = CtcDecoders.Greedy;

    /// <summary>The beam width of the "beam" decoder.</summary>
    public int BeamWidth { get; init; } = 10;

    /// <summary>The augmentations applied to the training utterances, in order: <see cref="AudioAugmentations"/> names ("speed", "noise", "spec-mask").</summary>
    public IReadOnlyList<string> Augmentations { get; init; } = [];

    /// <summary>The augmentations' settings.</summary>
    public AudioAugmentationOptions AugmentationOptions { get; init; } = new();

    /// <summary>The seed of the pass orders, the batches' order and the augmentations.</summary>
    public int Seed { get; init; }

    /// <summary>
    /// Where features are kept between passes: a <see cref="FeatureCaches"/> name ("disk", "memory"), keyed by each file's
    /// SHA-256 and the feature options, so a second pass takes no features again; null for none.
    /// </summary>
    public string? FeatureCache { get; init; } = FeatureCaches.Disk;

    /// <summary>The disk cache's folder (<see cref="FeatureCaches.DefaultFolder"/> when null).</summary>
    public string? FeatureCacheFolder { get; init; }

    /// <summary>Threads that read, decode and take the features of the next step while the device trains on this one (0: the library's thread budget).</summary>
    public int Workers { get; init; }

    /// <summary>The run stops when the loss is not finite this many steps in a row.</summary>
    public int NonFiniteSteps { get; init; } = 3;

    /// <summary>
    /// A synthesizer's training (<see cref="SpeechSynthesisTrainer"/>): the alignments file (<see cref="SpeechAlignments"/>,
    /// written by <c>idrak data audio --align MODEL</c>) giving each utterance's label durations; recognizers ignore it.
    /// </summary>
    public string? Durations { get; init; }

    /// <summary>The "durations" synthesis objective's weight of the log-duration error beside the log-mel error (0: the durations not trained).</summary>
    public float DurationLossWeight { get; init; } = 1f;

    /// <summary>
    /// A CTC teacher for a CTC student (plan 21, S6 distillation; null: none): each step's loss becomes
    /// (1 − <see cref="DistillationAlpha"/>) · CTC + α · T² · the mean over the batch's steps of KL(teacher ‖ student) at
    /// temperature <see cref="DistillationTemperature"/> (<see cref="CtcDistillation"/>). The teacher reads the same
    /// features (its feature options equal the student's), has the same classes and blank, and gives as many steps per
    /// utterance. Not kept in checkpoints (a model, not a value): give it again on resume. A sequence-level teacher
    /// (Whisper-style text) teaches through pseudo-labels instead (<c>idrak transcribe -o train.jsonl</c>).
    /// </summary>
    public ICtcSpeechModel? Teacher { get; init; }

    /// <summary>The weight of the teacher's term (0 to 1; the CTC loss gets 1 − α).</summary>
    public float DistillationAlpha { get; init; } = 0.5f;

    /// <summary>The temperature both distributions are softened by (1: as they are; 2 to 4 is usual).</summary>
    public float DistillationTemperature { get; init; } = 2f;

    /// <summary>The options as JSON (<see cref="FromJson"/> reads it back).</summary>
    public JsonObject ToJson() => new()
    {
        ["steps"] = Steps, ["batch_seconds"] = BatchSeconds, ["gradient_accumulation"] = GradientAccumulation, ["bucket_utterances"] = BucketUtterances,
        ["learning_rate"] = LearningRate, ["warmup_fraction"] = WarmupFraction, ["min_learning_rate"] = MinLearningRate, ["schedule"] = Schedule,
        ["optimizer"] = Optimizer, ["weight_decay"] = WeightDecay, ["momentum"] = Momentum, ["host_optimizer"] = HostOptimizer,
        ["max_gradient_norm"] = MaxGradientNorm, ["bf16_activations"] = BFloat16Activations, ["checkpointing"] = Checkpointing,
        ["save_every"] = SaveEvery, ["keep_checkpoints"] = KeepCheckpoints, ["evaluate_every"] = EvaluateEvery, ["metric"] = Metric,
        ["decoder"] = Decoder, ["beam_width"] = BeamWidth, ["augmentations"] = new JsonArray([.. Augmentations.Select(a => (JsonNode)a)]),
        ["augmentation_options"] = AugmentationOptions.ToJson(), ["seed"] = Seed, ["feature_cache"] = FeatureCache,
        ["feature_cache_folder"] = FeatureCacheFolder, ["workers"] = Workers, ["non_finite_steps"] = NonFiniteSteps,
        ["durations"] = Durations, ["duration_loss_weight"] = DurationLossWeight,
        ["distillation_alpha"] = DistillationAlpha, ["distillation_temperature"] = DistillationTemperature,
    };

    /// <summary>Options from their JSON (<see cref="ToJson"/>); a key missing keeps the default.</summary>
    public static SpeechTrainingOptions FromJson(JsonObject json)
    {
        ArgumentNullException.ThrowIfNull(json);
        var d = new SpeechTrainingOptions();
        return new SpeechTrainingOptions
        {
            Steps = (int?)json["steps"] ?? d.Steps,
            BatchSeconds = (double?)json["batch_seconds"] ?? d.BatchSeconds,
            GradientAccumulation = (int?)json["gradient_accumulation"] ?? d.GradientAccumulation,
            BucketUtterances = (int?)json["bucket_utterances"] ?? d.BucketUtterances,
            LearningRate = (float?)json["learning_rate"] ?? d.LearningRate,
            WarmupFraction = (float?)json["warmup_fraction"] ?? d.WarmupFraction,
            MinLearningRate = (float?)json["min_learning_rate"] ?? d.MinLearningRate,
            Schedule = (string?)json["schedule"] ?? d.Schedule,
            Optimizer = (string?)json["optimizer"] ?? d.Optimizer,
            WeightDecay = (float?)json["weight_decay"] ?? d.WeightDecay,
            Momentum = (float?)json["momentum"] ?? d.Momentum,
            HostOptimizer = (bool?)json["host_optimizer"] ?? d.HostOptimizer,
            MaxGradientNorm = (float?)json["max_gradient_norm"] ?? d.MaxGradientNorm,
            BFloat16Activations = (bool?)json["bf16_activations"],
            Checkpointing = (bool?)json["checkpointing"] ?? d.Checkpointing,
            SaveEvery = (int?)json["save_every"] ?? d.SaveEvery,
            KeepCheckpoints = (int?)json["keep_checkpoints"] ?? d.KeepCheckpoints,
            EvaluateEvery = (int?)json["evaluate_every"] ?? d.EvaluateEvery,
            Metric = (string?)json["metric"] ?? d.Metric,
            Decoder = (string?)json["decoder"] ?? d.Decoder,
            BeamWidth = (int?)json["beam_width"] ?? d.BeamWidth,
            Augmentations = json["augmentations"] is JsonArray augmentations ? [.. augmentations.Select(a => (string)a!)] : d.Augmentations,
            AugmentationOptions = json["augmentation_options"] is JsonObject augmentation ? AudioAugmentationOptions.FromJson(augmentation) : d.AugmentationOptions,
            Seed = (int?)json["seed"] ?? d.Seed,
            FeatureCache = json.ContainsKey("feature_cache") ? (string?)json["feature_cache"] : d.FeatureCache,
            FeatureCacheFolder = (string?)json["feature_cache_folder"],
            Workers = (int?)json["workers"] ?? d.Workers,
            NonFiniteSteps = (int?)json["non_finite_steps"] ?? d.NonFiniteSteps,
            Durations = (string?)json["durations"],
            DurationLossWeight = (float?)json["duration_loss_weight"] ?? d.DurationLossWeight,
            DistillationAlpha = (float?)json["distillation_alpha"] ?? d.DistillationAlpha,
            DistillationTemperature = (float?)json["distillation_temperature"] ?? d.DistillationTemperature,
        };
    }

    internal void Check()
    {
        if (Steps < 1 || !(BatchSeconds > 0) || GradientAccumulation < 1 || BucketUtterances < 1 || !(LearningRate > 0) || WarmupFraction is < 0 or > 1
            || MaxGradientNorm < 0 || SaveEvery < 0 || KeepCheckpoints < 0 || EvaluateEvery < 0 || BeamWidth < 1 || Workers < 0 || NonFiniteSteps < 0
            || !(DurationLossWeight >= 0) || DistillationAlpha is < 0 or > 1 || float.IsNaN(DistillationAlpha)
            || !(DistillationTemperature > 0) || float.IsInfinity(DistillationTemperature))
        {
            throw new ArgumentException("Speech training: Steps, BatchSeconds, GradientAccumulation, BucketUtterances, LearningRate and BeamWidth are positive, "
                + "WarmupFraction 0 to 1, and MaxGradientNorm, SaveEvery, KeepCheckpoints, EvaluateEvery, Workers, NonFiniteSteps and DurationLossWeight not negative, "
                + "DistillationAlpha 0 to 1 and DistillationTemperature positive.");
        }

        if (!FineTuningOptimizers.Names.Contains(Optimizer, StringComparer.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Optimizer '{Optimizer}' is not one of {string.Join(", ", FineTuningOptimizers.Names)}.");
        }

        if (!FineTuningSchedules.Names.Contains(Schedule, StringComparer.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Schedule '{Schedule}' is not one of {string.Join(", ", FineTuningSchedules.Names)}.");
        }

        _ = TuningMetrics.Get(Metric);
        _ = CtcDecoders.Get(Decoder);
        foreach (string name in Augmentations)
        {
            _ = AudioAugmentations.Get(name);
        }

        AugmentationOptions.Check();
        if (FeatureCache is { } cache)
        {
            _ = FeatureCaches.Get(cache);
        }
    }
}

/// <summary>Where a speech training run is (<see cref="SpeechTrainer.Train"/>'s progress), reported after every step.</summary>
/// <param name="Stage">What runs: "step" (a step finished), "evaluate" (the evaluation source being scored) or "checkpoint" (one written).</param>
/// <param name="Step">Steps done.</param>
/// <param name="Steps">Steps in all.</param>
/// <param name="Epoch">The pass over the training utterances (1 for the first).</param>
/// <param name="EpochFraction">How far into that pass, 0 to 1.</param>
/// <param name="Loss">The last step's loss.</param>
/// <param name="LearningRate">The rate it used.</param>
/// <param name="UtterancesPerSecond">Utterances trained a second (over the run, the first step aside).</param>
/// <param name="AudioSecondsPerSecond">Seconds of audio trained a second (how many times real time).</param>
/// <param name="Elapsed">Since the run (or the resume) began.</param>
/// <param name="Remaining">The time left at the run's pace.</param>
/// <param name="PeakDeviceBytes">The device's peak memory (0 where it does not report it).</param>
/// <param name="MicroBatchSeconds">The audio seconds that go through the device at once (smaller than a batch after running out of memory).</param>
/// <param name="Evaluation">The evaluation score of this step, when it was scored.</param>
public sealed record SpeechTrainingProgress(string Stage, int Step, int Steps, int Epoch, double EpochFraction, float Loss, float LearningRate, double UtterancesPerSecond,
    double AudioSecondsPerSecond, TimeSpan Elapsed, TimeSpan Remaining, long PeakDeviceBytes, double MicroBatchSeconds, double? Evaluation);

/// <summary>What a speech training run did.</summary>
/// <param name="Steps">Steps done in all (a resumed run's included).</param>
/// <param name="Losses">Each step's loss, the first step first.</param>
/// <param name="Evaluations">The evaluation scores (the step, the <see cref="SpeechTrainingOptions.Metric"/>'s value).</param>
/// <param name="Checkpoint">The last checkpoint written (a model folder <c>SpeechModels.Load</c> reads), or null.</param>
/// <param name="Elapsed">The time this call took.</param>
/// <param name="Utterances">The training utterances used.</param>
/// <param name="Dropped">Utterances left out because the objective cannot train on them (CTC: fewer steps than their labels need).</param>
/// <param name="MicroBatchSeconds">The micro-batch the run ended with (the batch's seconds unless the device ran out of memory).</param>
/// <param name="FeaturesExtracted">Utterances whose features were taken this call.</param>
/// <param name="FeaturesCached">Utterances whose features came from the cache.</param>
/// <param name="Notes">What the sources left out and what the run changed (a smaller micro-batch), one line each.</param>
public sealed record SpeechTrainingResult(int Steps, IReadOnlyList<float> Losses, IReadOnlyList<(int Step, double Score)> Evaluations, string? Checkpoint, TimeSpan Elapsed,
    int Utterances, int Dropped, double MicroBatchSeconds, int FeaturesExtracted, int FeaturesCached, IReadOnlyList<string> Notes);

/// <summary>A checkpoint of a speech training run (<see cref="SpeechTrainer.ReadCheckpoint"/>): the model folder and where the run was.</summary>
/// <param name="Folder">The checkpoint folder (a model folder, with the optimizer's state and train-state.json).</param>
/// <param name="Options">The run's options.</param>
/// <param name="Train">The training source.</param>
/// <param name="Evaluation">The evaluation source, or null.</param>
/// <param name="Step">Steps done.</param>
/// <param name="Epoch">The pass the next step reads from (0 for the first).</param>
/// <param name="Batch">The next batch of that pass.</param>
/// <param name="MicroBatchSeconds">The micro-batch the run had (so a resumed run splits its batches as the run did).</param>
/// <param name="Losses">Each step's loss.</param>
/// <param name="Evaluations">The evaluation scores so far.</param>
public sealed record SpeechTrainingCheckpoint(string Folder, SpeechTrainingOptions Options, SpeechSource Train, SpeechSource? Evaluation, int Step, int Epoch, int Batch,
    double MicroBatchSeconds, IReadOnlyList<float> Losses, IReadOnlyList<(int Step, double Score)> Evaluations);

/// <summary>
/// Trains a speech model (<see cref="ITrainableSpeechModel"/>) on transcribed speech (plan 21, S6) with its objective
/// (<see cref="SpeechTrainingObjectives"/>: "ctc", "seq2seq"), the library's optimizers and schedules. Utterances are
/// batched by length (sorted within windows of the pass's seeded order, cut by <see cref="SpeechTrainingOptions.BatchSeconds"/>);
/// the next step's features are read, decoded and taken on worker threads while the device trains, and kept in a feature
/// cache so later passes take none; checkpoints (<c>checkpoint-STEP</c>: the model folder, <c>optimizer.safetensors</c>,
/// <c>train-state.json</c>) resume the run where it stopped, bit for bit on the CPU. The same loop trains synthesizers
/// (<see cref="SpeechSynthesisTrainer"/>).
/// </summary>
public static class SpeechTrainer
{
    /// <summary>The checkpoint state file's format.</summary>
    public const string StateFormat = "idrak-speech-train-state/1";

    /// <summary>The checkpoint state file.</summary>
    public const string StateFile = "train-state.json";

    private const string OptimizerFile = "optimizer.safetensors";
    private const string FeatureStage = "audio-features";

    /// <summary>
    /// Trains <paramref name="model"/> on <paramref name="train"/> for <see cref="SpeechTrainingOptions.Steps"/> steps, writing
    /// checkpoints into <paramref name="outputFolder"/>, scoring <paramref name="evaluation"/> when given; or goes on from
    /// <paramref name="resume"/> (the model then read from its folder, see <see cref="Resume"/>).
    /// </summary>
    /// <exception cref="ArgumentException">An option is out of range or names nothing registered, or the source holds no utterance CTC can read.</exception>
    /// <exception cref="InvalidOperationException">The loss stayed not finite (<see cref="SpeechTrainingOptions.NonFiniteSteps"/>).</exception>
    /// <exception cref="OperationCanceledException">Cancelled: a checkpoint of the last finished step is written first.</exception>
    public static SpeechTrainingResult Train(ITrainableSpeechModel model, SpeechSource train, SpeechSource? evaluation, SpeechTrainingOptions options, string outputFolder,
        SpeechTrainingCheckpoint? resume = null, IProgress<SpeechTrainingProgress>? progress = null, CancellationToken cancellationToken = default, Action<string>? trace = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(options);
        options.Check();
        var objective = SpeechTrainingObjectives.Get(model.Objective);
        if (!objective.Trains(model))
        {
            throw new ArgumentException($"The '{objective.Name}' objective does not train a {model.GetType().Name} ({model.Architecture}).", nameof(model));
        }

        if (options.Teacher is { } teacher)
        {
            CtcDistillation.Check(model as ICtcSpeechModel ?? throw new ArgumentException(
                $"A CTC teacher (SpeechTrainingOptions.Teacher) teaches a CTC student, not a {model.GetType().Name} ('{model.Objective}'); a sequence-level student learns from pseudo-labels.", nameof(options)), teacher);
            teacher.Network.Eval();                                             // its posteriors without dropout
        }

        return Run(new RecognizerTrainee(model, objective, options), train, evaluation, options, outputFolder, resume, progress, cancellationToken, trace);
    }

    /// <summary>
    /// The training loop for any trainee (a recognizer with its objective here, a synthesizer in
    /// <see cref="SpeechSynthesisTrainer"/>): the sources, batches, steps, evaluations and checkpoints.
    /// </summary>
    internal static SpeechTrainingResult Run(SpeechTrainee model, SpeechSource train, SpeechSource? evaluation, SpeechTrainingOptions options, string outputFolder,
        SpeechTrainingCheckpoint? resume, IProgress<SpeechTrainingProgress>? progress, CancellationToken cancellationToken, Action<string>? trace)
    {
        ArgumentNullException.ThrowIfNull(train);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputFolder);
        Directory.CreateDirectory(outputFolder);
        var clock = Stopwatch.StartNew();
        var notes = new List<string>();
        using var reader = SpeechSources.Open(train);
        using var evalReader = evaluation is null ? null : SpeechSources.Open(evaluation);
        notes.AddRange(reader.Notes);
        notes.AddRange((evalReader?.Notes ?? []).Select(n => "evaluation: " + n));
        foreach (string note in notes)
        {
            trace?.Invoke(note);
        }

        using var cache = options.FeatureCache is { } cacheName ? FeatureCaches.Create(cacheName, new FeatureCacheOptions { Folder = options.FeatureCacheFolder }) : null;
        var features = new FeatureSource(model.Features, cache, options);

        // The utterances the objective can train on (CTC: their frames give at least the steps their labels need).
        var labels = new int[reader.Count][];
        var frames = new int[reader.Count];
        var usable = new List<int>(reader.Count);
        int rate = model.Features.Options.SampleRate;
        for (int i = 0; i < reader.Count; i++)
        {
            labels[i] = model.Labels(reader[i].Text);
            frames[i] = model.Features.FramesFor((long)Math.Round(reader[i].Seconds * rate));
            if (model.Accepts(reader, i, frames[i], labels[i]))
            {
                usable.Add(i);
            }
        }

        int dropped = reader.Count - usable.Count;
        if (dropped > 0)
        {
            string note = $"{dropped} utterances {model.Refused} and are left out";
            notes.Add(note);
            trace?.Invoke(note);
        }

        if (usable.Count == 0)
        {
            throw new ArgumentException($"The training source holds no utterance the '{model.Objective}' objective can train on ({reader.Count} read, {dropped} that {model.Refused}).", nameof(train));
        }

        int[] scored = evalReader is null ? [] : [.. Enumerable.Range(0, evalReader.Count).Where(i => model.Evaluates(evalReader, i))];
        if (evalReader is { Count: > 0 } && scored.Length < evalReader.Count)
        {
            string note = $"evaluation: {evalReader.Count - scored.Length} utterances {model.Refused} and are not scored";
            notes.Add(note);
            trace?.Invoke(note);
        }

        var seconds = new double[reader.Count];
        for (int i = 0; i < seconds.Length; i++)
        {
            seconds[i] = reader[i].Seconds;
        }

        var augmentations = options.Augmentations.Select(AudioAugmentations.Get).ToArray();
        var parameters = model.Network.Parameters().ToList();
        using var optimizer = CreateOptimizer(parameters, options);
        var losses = new List<float>(resume?.Losses ?? []);
        var evaluations = new List<(int, double)>(resume?.Evaluations ?? []);
        int step = resume?.Step ?? 0, epoch = resume?.Epoch ?? 0, next = resume?.Batch ?? 0;
        double micro = resume?.MicroBatchSeconds ?? options.BatchSeconds;
        if (resume is not null)
        {
            optimizer.LoadState(ReadOptimizerState(resume.Folder));
        }

        var schedule = FineTuningSchedules.Create(options.Schedule, options.WarmupFraction, options.MinLearningRate)(optimizer, options.Steps);
        for (int i = 0; i < step; i++)
        {
            schedule.Step();                                                    // the resumed run's rate where it was
        }

        // Each pass's batches, made once from the seed (the worker reading ahead and the loop both ask), the passes behind dropped.
        var plans = new ConcurrentDictionary<int, List<int[]>>();
        List<int[]> PlanFor(int e)
        {
            var made = plans.GetOrAdd(e, x => Plan(usable, seconds, options, x));
            plans.TryRemove(e - 2, out _);
            return made;
        }

        Telemetry.TrainingStarted(new TrainingStarted(model.Architecture, options.Optimizer, model.Device, 0, usable.Count, evalReader?.Count, 0, PlanFor(epoch).Count,
            parameters.Sum(p => (long)p.Size), options.LearningRate, Environment.ProcessorCount));
        string? last = resume?.Folder;
        var forward = new SpeechForwardOptions { Checkpointing = options.Checkpointing };
        int startStep = step, nonFinite = 0;
        double trainedUtterances = 0, trainedSeconds = 0, timedSeconds = 0;
        model.Network.Train();

        // The batches of the step at position (e, n) and the position after them, read on worker threads.
        (List<Prepared[]> Batches, int Epoch, int Next) Take(int e, int n)
        {
            var batches = new List<Prepared[]>();
            for (int b = 0; b < options.GradientAccumulation; b++)
            {
                if (n >= PlanFor(e).Count)
                {
                    (e, n) = (e + 1, 0);
                }

                batches.Add(features.Batch(reader, PlanFor(e)[n], labels, e, augmentations));
                n++;
            }

            return (batches, e, n);
        }

        (int firstEpoch, int firstBatch) = (epoch, next);
        Task<(List<Prepared[]> Batches, int Epoch, int Next)>? ahead = step < options.Steps ? Task.Run(() => Take(firstEpoch, firstBatch)) : null;
        try
        {
            while (step < options.Steps)
            {
                cancellationToken.ThrowIfCancellationRequested();
                long waited = Stopwatch.GetTimestamp();
                var (batches, nextEpoch, nextBatch) = ahead!.GetAwaiter().GetResult();
                var dataTime = Stopwatch.GetElapsedTime(waited);
                ahead = step + 1 < options.Steps ? Task.Run(() => Take(nextEpoch, nextBatch)) : null;

                long computeStart = Stopwatch.GetTimestamp();
                float rate0 = optimizer.LearningRate;
                double loss = RunStep(model, reader, optimizer, batches, options, forward, ref micro, notes, trace);
                schedule.Step();
                var compute = Stopwatch.GetElapsedTime(computeStart);
                (epoch, next) = (nextEpoch, nextBatch);
                step++;
                losses.Add((float)loss);
                nonFinite = double.IsFinite(loss) ? 0 : nonFinite + 1;
                if (options.NonFiniteSteps > 0 && nonFinite >= options.NonFiniteSteps)
                {
                    throw new InvalidOperationException($"The loss of step {step} is {loss}, the last {nonFinite} steps not finite; the run stops (SpeechTrainingOptions.NonFiniteSteps).");
                }

                int utterances = batches.Sum(b => b.Length);
                double audio = batches.Sum(b => b.Sum(u => u.Seconds));
                if (step > startStep + 1)
                {
                    // The first step aside (compiling, allocating): the pace the time left is measured by.
                    (trainedUtterances, trainedSeconds, timedSeconds) = (trainedUtterances + utterances, trainedSeconds + audio, timedSeconds + (dataTime + compute).TotalSeconds);
                }

                if (Telemetry.IsEnabled(TelemetryLevel.Batches))
                {
                    Telemetry.BatchCompleted(new BatchCompleted(epoch + 1, next, PlanFor(epoch).Count, step, utterances, loss, rate0, null, dataTime, compute));
                }

                double? score = null;
                bool end = step == options.Steps;
                if (scored.Length > 0 && (options.EvaluateEvery > 0 ? step % options.EvaluateEvery == 0 || end : end))
                {
                    Report("evaluate", null);
                    score = Evaluate(model, evalReader!, scored, features, options, micro);
                    evaluations.Add((step, score.Value));
                    trace?.Invoke(string.Create(CultureInfo.InvariantCulture, $"step {step}: {model.Metric} {score.Value:F4}"));
                }

                if (end || options.SaveEvery > 0 && step % options.SaveEvery == 0)
                {
                    last = SaveCheckpoint(model, optimizer, options, train, evaluation, outputFolder, step, epoch, next, micro, losses, evaluations);
                    trace?.Invoke($"checkpoint {last}");
                    Report("checkpoint", score);
                }

                Report("step", score);

                void Report(string stage, double? evaluated)
                {
                    if (progress is null)
                    {
                        return;
                    }

                    double pace = timedSeconds / Math.Max(1, step - startStep - 1);
                    progress.Report(new SpeechTrainingProgress(stage, step, options.Steps, epoch + 1, next / (double)Math.Max(1, PlanFor(epoch).Count), (float)loss, rate0,
                        trainedUtterances / Math.Max(1e-9, timedSeconds), trainedSeconds / Math.Max(1e-9, timedSeconds), clock.Elapsed,
                        TimeSpan.FromSeconds(pace * (options.Steps - step)), model.Device.Backend.GetMemoryUsage().Peak, micro, evaluated));
                }
            }
        }
        catch (OperationCanceledException) when (step > startStep)
        {
            last = SaveCheckpoint(model, optimizer, options, train, evaluation, outputFolder, step, epoch, next, micro, losses, evaluations);
            trace?.Invoke($"cancelled: checkpoint {last}");
            throw;
        }
        finally
        {
            // A step read ahead that will not run: wait for it, so its threads are done before the sources close.
            try
            {
                ahead?.GetAwaiter().GetResult();
            }
            catch (Exception)
            {
            }

            model.Network.Eval();
        }

        Telemetry.TrainingCompleted(new TrainingCompleted(epoch + 1, clock.Elapsed, losses.Count > 0 ? losses[^1] : double.NaN, 0,
            evaluations.Count > 0 ? evaluations.Min(e => e.Item2) : double.NaN, false, false));
        return new SpeechTrainingResult(step, losses, evaluations, last, clock.Elapsed, usable.Count, dropped, micro, features.Extracted, features.Cached, notes);
    }

    /// <summary>
    /// Goes on with the run a checkpoint folder holds: its model read back (<c>SpeechModels.Load</c>, a family whose model
    /// is an <see cref="ITrainableSpeechModel"/>), its optimizer's state, its sources and position, and its options (or
    /// <paramref name="overrides"/>, for more steps say), writing later checkpoints beside it.
    /// </summary>
    /// <exception cref="FileNotFoundException">The folder holds no training state.</exception>
    /// <exception cref="InvalidDataException">The folder's model is not one the speech trainer trains.</exception>
    public static SpeechTrainingResult Resume(string checkpointFolder, SpeechTrainingOptions? overrides = null, Device? device = null, IProgress<SpeechTrainingProgress>? progress = null,
        CancellationToken cancellationToken = default, Action<string>? trace = null)
    {
        var state = ReadCheckpoint(checkpointFolder);
        using var recognizer = SpeechModels.Load(checkpointFolder, new SpeechModelOptions { Device = device });
        var model = recognizer as ITrainableSpeechModel
            ?? throw new InvalidDataException($"{checkpointFolder} holds a {recognizer.GetType().Name}, not a model the speech trainer trains (an ITrainableSpeechModel).");
        string output = Path.GetDirectoryName(Path.GetFullPath(checkpointFolder))!;
        return Train(model, state.Train, state.Evaluation, overrides ?? state.Options, output, state with { Options = overrides ?? state.Options }, progress, cancellationToken, trace);
    }

    /// <summary>The latest checkpoint a run wrote into <paramref name="outputFolder"/>, or null.</summary>
    public static string? LatestCheckpoint(string outputFolder) => Checkpoints(outputFolder).Select(c => c.Folder).LastOrDefault();

    /// <summary>Reads where a run stopped from a checkpoint folder <see cref="Train"/> (or <see cref="SpeechSynthesisTrainer.Train"/>) wrote.</summary>
    /// <exception cref="FileNotFoundException">The folder holds no training state.</exception>
    /// <exception cref="InvalidDataException">The state is not this format.</exception>
    public static SpeechTrainingCheckpoint ReadCheckpoint(string folder)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(folder);
        string path = Path.Combine(folder, StateFile);
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"{folder} holds no speech training state ({StateFile}); give a checkpoint-STEP folder a run wrote.", path);
        }

        var json = JsonNode.Parse(File.ReadAllText(path))!.AsObject();
        if ((string?)json["format"] != StateFormat)
        {
            throw new InvalidDataException($"{path} is in format '{json["format"]}', not {StateFormat}.");
        }

        return new SpeechTrainingCheckpoint(folder, SpeechTrainingOptions.FromJson(json["options"]!.AsObject()), SpeechSource.FromJson(json["train"]!.AsObject()),
            json["evaluation"] is JsonObject e ? SpeechSource.FromJson(e) : null, (int)json["step"]!, (int)json["epoch"]!, (int)json["batch"]!,
            (double)json["micro_batch_seconds"]!,
            [.. json["losses"]!.AsArray().Select(l => l is JsonValue v && v.TryGetValue(out float f) ? f : float.NaN)],
            [.. json["evaluations"]!.AsArray().Select(x => ((int)x!["step"]!, (double)x["score"]!))]);
    }

    // One optimizer step over its batches: each split into micro-batches of at most `micro` padded seconds, every loss
    // scaled by its utterances' share of the step, so the gradients added up are the step's mean. Out of device memory, the
    // micro-batches are halved and the step runs again from zeroed gradients.
    private static double RunStep(SpeechTrainee model, ISpeechReader reader, Optimizer optimizer, List<Prepared[]> batches, SpeechTrainingOptions options, SpeechForwardOptions forward,
        ref double micro, List<string> notes, Action<string>? trace)
    {
        int total = batches.Sum(b => b.Length);
        double secondsPerFrame = model.Features.Options.HopSize / (double)model.Features.Options.SampleRate;
        while (true)
        {
            double limit = micro;
            var groups = batches.SelectMany(b => Split(b, limit, secondsPerFrame)).ToList();
            try
            {
                optimizer.ZeroGrad();
                double sum = 0;
                foreach (var group in groups)
                {
                    using var scope = new TensorScope();
                    using var precision = options.BFloat16Activations == true ? ActivationMemory.CompressToBFloat16() : (ActivationMemory.Scope?)null;
                    var loss = model.Loss(reader, Padded(group, model), group, forward);
                    var scaled = group.Length == total ? loss : loss * (group.Length / (float)total);
                    scaled.Backward();
                    sum += loss.Item() * group.Length;
                }

                optimizer.ClipAndStep(options.MaxGradientNorm);
                return sum / total;
            }
            catch (ResourceLimitExceededException e) when (groups.Any(g => g.Length > 1))
            {
                micro /= 2;
                string note = string.Create(CultureInfo.InvariantCulture,
                    $"out of device memory ({e.Message.Split(':')[0]}): micro-batches of {micro:0.##} audio seconds from this step, their gradients added up (SpeechTrainingOptions.BatchSeconds to start smaller)");
                notes.Add(note);
                trace?.Invoke(note);
            }
        }
    }

    // A group's features [count, longest, mels] on the trainee's device, each utterance's frames first and zeros after them.
    private static Tensor Padded(Prepared[] group, SpeechTrainee model)
    {
        int mels = model.Features.Options.Mels;
        int longest = group.Max(u => u.Frames);
        var padded = new float[group.Length * longest * mels];
        for (int i = 0; i < group.Length; i++)
        {
            group[i].Features.AsSpan(0, group[i].Frames * mels).CopyTo(padded.AsSpan(i * longest * mels));
        }

        return Tensor.From(padded, [group.Length, longest, mels], model.Device);
    }

    // A batch (sorted by length) cut into runs whose padded seconds (count × the longest) are at most `micro`; one utterance
    // longer than that goes alone.
    private static IEnumerable<Prepared[]> Split(Prepared[] batch, double micro, double secondsPerFrame)
    {
        var sorted = batch.OrderBy(u => u.Frames).ToArray();
        int start = 0;
        while (start < sorted.Length)
        {
            int end = start + 1;
            while (end < sorted.Length && (end - start + 1) * sorted[end].Frames * secondsPerFrame <= micro)
            {
                end++;
            }

            yield return sorted[start..end];
            start = end;
        }
    }

    // The batches of pass `epoch`: the usable utterances in the pass's seeded order, windows of BucketUtterances sorted by
    // length, cut into batches of at most BatchSeconds padded seconds, the batches shuffled from the seed and the pass.
    private static List<int[]> Plan(List<int> usable, double[] seconds, SpeechTrainingOptions options, int epoch)
    {
        var order = SpeechSources.Order(usable.Count, options.Seed, epoch);
        var batches = new List<int[]>();
        for (int start = 0; start < order.Length; start += options.BucketUtterances)
        {
            var window = order.Skip(start).Take(options.BucketUtterances).Select(i => usable[i]).OrderBy(u => seconds[u]).ThenBy(u => u).ToArray();
            var current = new List<int>();
            double longest = 0;
            foreach (int u in window)
            {
                double longer = Math.Max(longest, seconds[u]);
                if (current.Count > 0 && longer * (current.Count + 1) > options.BatchSeconds)
                {
                    batches.Add([.. current]);
                    current.Clear();
                    longer = seconds[u];
                }

                current.Add(u);
                longest = longer;
            }

            if (current.Count > 0)
            {
                batches.Add([.. current]);
            }
        }

        var random = new Random(unchecked(options.Seed * 1_000_003 + epoch * 10_007 + 29));
        for (int i = batches.Count - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (batches[i], batches[j]) = (batches[j], batches[i]);
        }

        return batches;
    }

    // The trainee's score over the evaluation utterances it can score: a window of BucketUtterances read at a time, sorted
    // by length (the features of the whole source are never held), the scores added up.
    private static double Evaluate(SpeechTrainee model, ISpeechReader reader, int[] scored, FeatureSource features, SpeechTrainingOptions options, double micro)
    {
        double secondsPerFrame = model.Features.Options.HopSize / (double)model.Features.Options.SampleRate;
        var scores = new List<TuningScore>(scored.Length);
        model.Network.Eval();
        try
        {
            var order = scored.OrderBy(i => reader[i].Seconds).ThenBy(i => i).ToArray();
            using var _ = Autograd.NoGrad();
            for (int start = 0; start < order.Length; start += options.BucketUtterances)
            {
                var window = features.Batch(reader, order[start..Math.Min(order.Length, start + options.BucketUtterances)], null, -1, []);
                foreach (var group in Split(window, micro, secondsPerFrame))
                {
                    using var scope = new TensorScope();
                    scores.AddRange(model.Score(reader, Padded(group, model), group));
                }
            }
        }
        finally
        {
            model.Network.Train();
        }

        return TuningScore.Sum(scores).Value;
    }

    private static Optimizer CreateOptimizer(List<Tensor> parameters, SpeechTrainingOptions options)
    {
        var make = FineTuningOptimizers.Create(options.Optimizer, options.LearningRate, options.WeightDecay, options.Momentum);
        return options.HostOptimizer ? new HostOptimizer(parameters, make) : make(parameters);
    }

    private static string SaveCheckpoint(SpeechTrainee model, Optimizer optimizer, SpeechTrainingOptions options, SpeechSource train, SpeechSource? evaluation,
        string outputFolder, int step, int epoch, int next, double micro, List<float> losses, List<(int, double)> evaluations)
    {
        string folder = Path.Combine(outputFolder, $"checkpoint-{step}");
        model.Save(folder);
        var state = optimizer.SaveState();
        SafeTensorsWriter.Write(Path.Combine(folder, OptimizerFile), state.Select(p => (p.Key, p.Value.Shape, p.Value.Values)), SafeTensorType.F32,
            new Dictionary<string, string> { ["format"] = "idrak-optimizer", ["optimizer"] = optimizer.GetType().Name });
        var json = new JsonObject
        {
            ["format"] = StateFormat,
            ["objective"] = model.Objective,
            ["step"] = step,
            ["epoch"] = epoch,
            ["batch"] = next,
            ["micro_batch_seconds"] = micro,
            ["train"] = train.ToJson(),
            ["evaluation"] = evaluation?.ToJson(),
            ["losses"] = new JsonArray([.. losses.Select(l => (JsonNode)(float.IsFinite(l) ? l : float.NaN.ToString(CultureInfo.InvariantCulture)))]),
            ["evaluations"] = new JsonArray([.. evaluations.Select(e => (JsonNode)new JsonObject { ["step"] = e.Item1, ["score"] = e.Item2 })]),
            ["options"] = options.ToJson(),
        };
        File.WriteAllText(Path.Combine(folder, StateFile), json.ToJsonString(new JsonSerializerOptions { WriteIndented = true }) + "\n");
        if (options.KeepCheckpoints > 0)
        {
            foreach (var old in Checkpoints(outputFolder).SkipLast(options.KeepCheckpoints))
            {
                Directory.Delete(old.Folder, recursive: true);
            }
        }

        return folder;
    }

    private static IEnumerable<(int Step, string Folder)> Checkpoints(string outputFolder) =>
        Directory.Exists(outputFolder)
            ? Directory.GetDirectories(outputFolder, "checkpoint-*")
                .Select(f => (Ok: int.TryParse(Path.GetFileName(f)["checkpoint-".Length..], NumberStyles.None, CultureInfo.InvariantCulture, out int s), Step: s, Folder: f))
                .Where(c => c.Ok && File.Exists(Path.Combine(c.Folder, StateFile))).OrderBy(c => c.Step).Select(c => (c.Step, c.Folder))
            : [];

    private static Dictionary<string, OptimizerStateValue> ReadOptimizerState(string folder)
    {
        string path = Path.Combine(folder, OptimizerFile);
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"{folder} holds no optimizer state ({OptimizerFile}).", path);
        }

        using var reader = SafeTensorsReader.Open(path);
        ITensorStore store = reader;
        return store.Names.ToDictionary(n => n, n => new OptimizerStateValue(store.ShapeOf(n), store.Read(n)), StringComparer.Ordinal);
    }

    /// <summary>An utterance ready for a step: its features [frames, mels] (host), its labels and its seconds.</summary>
    internal sealed record Prepared(int Utterance, float[] Features, int Frames, int[] Labels, double Seconds);

    // A recognizer and its objective: the utterances the objective accepts, its loss, and the options' metric of each
    // utterance's decoded text against its transcript.
    private sealed class RecognizerTrainee(ITrainableSpeechModel model, ISpeechTrainingObjective objective, SpeechTrainingOptions options) : SpeechTrainee
    {
        private readonly ITuningMetric _metric = TuningMetrics.Get(options.Metric);

        public override string Architecture => model.Architecture;

        public override string Objective => model.Objective;

        public override Module Network => model.Network;

        public override IAudioFeatureExtractor Features => model.Features;

        public override Device Device => model.Device;

        public override string Refused => objective.Refused;

        public override string Metric => options.Metric;

        public override int[] Labels(string text) => model.Labels(text);

        public override bool Accepts(ISpeechReader reader, int utterance, int frames, int[] labels) => objective.Accepts(model, frames, labels);

        public override Tensor Loss(ISpeechReader reader, Tensor features, Prepared[] group, SpeechForwardOptions forward) => options.Teacher is { } teacher
            ? CtcDistillation.Loss((ICtcSpeechModel)model, teacher, features, [.. group.Select(u => u.Frames)], [.. group.Select(u => u.Labels)], options.DistillationAlpha,
                options.DistillationTemperature, forward)
            : objective.Loss(model, features, [.. group.Select(u => u.Frames)], [.. group.Select(u => u.Labels)], forward);

        public override IEnumerable<TuningScore> Score(ISpeechReader reader, Tensor features, Prepared[] group)
        {
            var readings = objective.Decode(model, features, [.. group.Select(u => u.Frames)], options.Decoder, options.BeamWidth);
            for (int i = 0; i < group.Length; i++)
            {
                yield return _metric.Score(model.Text(readings[i]), reader[group[i].Utterance].Text);
            }
        }

        public override void Save(string folder) => model.Save(folder);
    }

    // Utterances to features on worker threads: the file read and hashed, the features from the cache when the utterance's
    // sound is not augmented this pass, else decoded, mixed to one channel, resampled, augmented and taken (and cached);
    // then the feature augmentations.
    private sealed class FeatureSource(IAudioFeatureExtractor extractor, IFeatureCache? cache, SpeechTrainingOptions options)
    {
        private readonly string _options = extractor.Options.ToJson().ToJsonString();
        private int _extracted, _cached;

        public int Extracted => Volatile.Read(ref _extracted);

        public int Cached => Volatile.Read(ref _cached);

        // `epoch` -1: no augmentation (evaluation); `labels` null: the readings' labels are not needed.
        public Prepared[] Batch(ISpeechReader reader, int[] utterances, int[][]? labels, int epoch, IAudioAugmentation[] augmentations)
        {
            var prepared = new Prepared[utterances.Length];
            var parallel = options.Workers > 0 ? new ParallelOptions { MaxDegreeOfParallelism = options.Workers } : ComputeResources.ParallelOptions;
            Parallel.For(0, utterances.Length, parallel, i => prepared[i] = One(reader, utterances[i], labels?[utterances[i]] ?? [], epoch, augmentations));
            return prepared;
        }

        private Prepared One(ISpeechReader reader, int utterance, int[] labels, int epoch, IAudioAugmentation[] augmentations)
        {
            int mels = extractor.Options.Mels;
            var draws = new Random?[augmentations.Length];
            bool sound = false;
            for (int a = 0; a < augmentations.Length; a++)
            {
                var random = AudioAugmentations.RandomFor(options.Seed, epoch, utterance, augmentations[a].Name);
                if (random.NextDouble() < options.AugmentationOptions.Probability)
                {
                    draws[a] = random;
                    sound |= augmentations[a].Stage == AudioAugmentationStage.Sound;
                }
            }

            var bytes = reader.ReadBytes(utterance);
            FeatureCacheKey? key = cache is not null && !sound
                ? new FeatureCacheKey(Convert.ToHexStringLower(SHA256.HashData(bytes)), "", _options, extractor.Name, "", "float32", FeatureStage) : null;
            float[]? values = null;
            int frames = 0;
            if (key is not null && cache!.TryGet(key, Device.Cpu, out var hit))
            {
                using (hit)
                {
                    values = hit.ToArray();
                    frames = hit.Shape[0];
                }

                Interlocked.Increment(ref _cached);
            }

            if (values is null)
            {
                var audio = AudioCodecs.Decode(bytes);
                if (audio.Channels > 1)
                {
                    var mono = new float[audio.Length];
                    audio.ToMono(mono);
                    audio = new AudioData(mono, 1, audio.SampleRate);
                }

                audio = Resampler.Resample(audio, extractor.Options.SampleRate);
                for (int a = 0; a < augmentations.Length; a++)
                {
                    if (draws[a] is { } random && augmentations[a].Stage == AudioAugmentationStage.Sound)
                    {
                        audio = augmentations[a].Augment(audio, random, options.AugmentationOptions);
                    }
                }

                using var extracted = extractor.Extract(audio, Device.Cpu);
                values = extracted.ToArray();
                frames = extracted.Shape[0];
                Interlocked.Increment(ref _extracted);
                if (key is not null)
                {
                    cache!.Put(key, extracted);
                }
            }

            for (int a = 0; a < augmentations.Length; a++)
            {
                if (draws[a] is { } random && augmentations[a].Stage == AudioAugmentationStage.Features)
                {
                    augmentations[a].Augment(values.AsSpan(0, frames * mels), frames, mels, random, options.AugmentationOptions);
                }
            }

            return new Prepared(utterance, values, frames, labels, reader[utterance].Seconds);
        }
    }
}
