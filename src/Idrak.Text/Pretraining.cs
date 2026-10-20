// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Cluster;
using Idrak.Datasets;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Text.Abstractions;
using Idrak.Optimizers;

namespace Idrak.Text;

/// <summary>
/// One stage of a pretraining run (<see cref="PretrainingOptions.Stages"/>): which parts of the model train, on which
/// sources, for how many steps, at which rate. A run of stages trains, for example, the decoder on text first, then the
/// vision tower and projector alone ("align"), then everything together ("joint").
/// </summary>
/// <param name="Name">The stage's name (shown in progress and kept in checkpoints).</param>
public sealed record PretrainingStage(string Name)
{
    /// <summary>The parts trained, every other parameter frozen: "decoder" (blocks and the final norm), "embeddings" (token and learned position embeddings), "head" (the output head, unless tied to the embeddings), "tower" and "projector" (a vision part's, <see cref="VisionTuningParts"/>), or "all".</summary>
    public IReadOnlyList<string> Trained { get; init; } = ["decoder", "embeddings", "head"];

    /// <summary>What it reads.</summary>
    public IReadOnlyList<PretrainingSource> Sources { get; init; } = [];

    /// <summary>Optimizer steps in the stage.</summary>
    public int Steps { get; init; }

    /// <summary>The peak learning rate (after the warm-up).</summary>
    public float LearningRate { get; init; } = 3e-4f;

    /// <summary>The rate at the stage's end.</summary>
    public float MinLearningRate { get; init; }

    /// <summary>The share of the stage's steps spent warming up.</summary>
    public float WarmupFraction { get; init; } = 0.03f;

    /// <summary>The rate over the stage's steps: a <see cref="FineTuningSchedules"/> name.</summary>
    public string Schedule { get; init; } = "cosine";

    /// <summary>The stage as JSON.</summary>
    public JsonObject ToJson() => new()
    {
        ["name"] = Name,
        ["trained"] = new JsonArray([.. Trained.Select(t => (JsonNode)t)]),
        ["sources"] = new JsonArray([.. Sources.Select(s => (JsonNode)s.ToJson())]),
        ["steps"] = Steps,
        ["learning_rate"] = LearningRate,
        ["min_learning_rate"] = MinLearningRate,
        ["warmup"] = WarmupFraction,
        ["schedule"] = Schedule,
    };

    /// <summary>A stage from its JSON (<see cref="ToJson"/>).</summary>
    /// <exception cref="InvalidDataException">A key is missing or wrong.</exception>
    public static PretrainingStage FromJson(JsonObject json)
    {
        ArgumentNullException.ThrowIfNull(json);
        var defaults = new PretrainingStage("");
        return new PretrainingStage((string?)json["name"] ?? throw new InvalidDataException("A pretraining stage has no \"name\"."))
        {
            Trained = json["trained"] is JsonArray trained ? [.. trained.Select(t => (string)t!)] : defaults.Trained,
            Sources = json["sources"] is JsonArray sources ? [.. sources.Select(s => PretrainingSource.FromJson(s!.AsObject()))] : [],
            Steps = (int?)json["steps"] ?? throw new InvalidDataException($"Stage {json["name"]} has no \"steps\"."),
            LearningRate = (float?)json["learning_rate"] ?? defaults.LearningRate,
            MinLearningRate = (float?)json["min_learning_rate"] ?? 0f,
            WarmupFraction = (float?)json["warmup"] ?? defaults.WarmupFraction,
            Schedule = (string?)json["schedule"] ?? defaults.Schedule,
        };
    }
}

/// <summary>Settings for <see cref="Pretrainer.Train"/>.</summary>
public sealed record PretrainingOptions
{
    /// <summary>The stages, run in order.</summary>
    public IReadOnlyList<PretrainingStage> Stages { get; init; } = [];

    /// <summary>The longest sequence in tokens: token files are cut into windows of this many (plus the next token as the last target).</summary>
    public int MaxLength { get; init; } = 2048;

    /// <summary>Positions per micro-batch (rows of <see cref="MaxLength"/>, sequences packed into them when the model supports it).</summary>
    public int BatchTokens { get; init; } = 4096;

    /// <summary>Micro-batches per optimizer step (their gradients add up).</summary>
    public int GradientAccumulation { get; init; } = 1;

    /// <summary>
    /// Sequences per optimizer step: 0 (the default) draws until the step's positions (rows of <see cref="MaxLength"/>
    /// times <see cref="GradientAccumulation"/>) are full; N draws exactly N, packed into as many micro-batches as they need.
    /// </summary>
    public int SequencesPerStep { get; init; }

    /// <summary>The gradient norm clipped to (0: no clipping).</summary>
    public float MaxGradientNorm { get; init; } = 1f;

    /// <summary>Weight decay (decoupled, for the AdamW kinds).</summary>
    public float WeightDecay { get; init; } = 0.1f;

    /// <summary>The optimizer: a <see cref="FineTuningOptimizers"/> name ("adamw", "adam", "adamw8bit", "sgd").</summary>
    public string Optimizer { get; init; } = "adamw";

    /// <summary>The optimizer's first moment decay (AdamW kinds).</summary>
    public float Beta1 { get; init; } = 0.9f;

    /// <summary>The optimizer's second moment decay (AdamW kinds).</summary>
    public float Beta2 { get; init; } = 0.95f;

    /// <summary>SGD's momentum.</summary>
    public float Momentum { get; init; }

    /// <summary>Run the optimizer's update on the CPU with its state in system memory (<see cref="HostOptimizer"/>), for parameters whose state the device has no room for.</summary>
    public bool HostOptimizer { get; init; }

    /// <summary>Recompute each block's activations in the backward pass; null: measured (off, turned on when the device runs out of memory), as <see cref="FineTuningOptions.Checkpointing"/>.</summary>
    public bool? Checkpointing { get; init; }

    /// <summary>Recompute the feed-forward activations; null: measured, as <see cref="FineTuningOptions.RecomputeFeedForward"/>.</summary>
    public bool? RecomputeFeedForward { get; init; }

    /// <summary>Hold activations as bfloat16 between the passes; null: measured, as <see cref="FineTuningOptions.BFloat16Activations"/>.</summary>
    public bool? BFloat16Activations { get; init; }

    /// <summary>Record steps as graphs where the device can (<see cref="FineTuningOptions.CudaGraphs"/>).</summary>
    public bool CudaGraphs { get; init; } = true;

    /// <summary>Pack sequences into rows when the model supports it.</summary>
    public bool Packing { get; init; } = true;

    /// <summary>The seed of the data order.</summary>
    public int Seed { get; init; }

    /// <summary>Steps between checkpoints (0: at each stage's end only).</summary>
    public int SaveEvery { get; init; }

    /// <summary>Checkpoints kept (the oldest removed past it; 0: every one).</summary>
    public int KeepCheckpoints { get; init; } = 3;

    /// <summary>Steps between evaluations (0: at each stage's end only).</summary>
    public int EvaluateEvery { get; init; }

    /// <summary>The held-out sources, scored by their mean token loss.</summary>
    public IReadOnlyList<PretrainingSource> Evaluation { get; init; } = [];

    /// <summary>Sequences read from each held-out source.</summary>
    public int EvaluationSequences { get; init; } = 256;

    /// <summary>Steps in a row whose loss is not finite (their updates skipped) before the run stops.</summary>
    public int NonFiniteSteps { get; init; } = 8;

    /// <summary>Rows of the output head per chunk of the loss.</summary>
    public int LossChunkRows { get; init; } = 512;

    /// <summary>How record images are prepared (transforms and vision options).</summary>
    public TuningImages Images { get; init; } = TuningImages.None;

    /// <summary>Where image features of frozen vision stages are cached: a <see cref="FeatureCaches"/> name.</summary>
    public string FeatureCache { get; init; } = FeatureCaches.Memory;

    /// <summary>
    /// A teacher whose token distributions the model learns beside the token loss (<see cref="FineTuningLosses.Distillation"/>),
    /// or null: <see cref="DistillationTeachers.FromModel(PretrainedModel, int, int, TuningImages)"/> for a loaded model
    /// (one that reads images, for image records). It shares the model's vocabulary. Not saved in checkpoints: give it
    /// again on resume.
    /// </summary>
    public DistillationTeacher? Teacher { get; init; }

    /// <summary>The distillation temperature (with <see cref="Teacher"/>).</summary>
    public float DistillationTemperature { get; init; } = 1f;

    /// <summary>The teacher's share of the loss, 0 to 1 (with <see cref="Teacher"/>).</summary>
    public float DistillationAlpha { get; init; } = 0.5f;

    /// <summary>
    /// Data-parallel pretraining (plan 23): this rank's collectives, or null (the default) to train in this process
    /// alone, exactly as without it. Every rank runs <see cref="Pretrainer.Train"/> with its own model on its rank's
    /// device and the same options: each step draws the same sequences on every rank (the step the options describe, the
    /// same at any world size) and rank r trains the step's draws k·W + r (or a share by the ranks' measured speeds,
    /// <see cref="ClusterOptions.RebalanceSteps"/>); the gradients are summed after the last micro-batch's backward pass,
    /// weighted by each rank's trained tokens, so W ranks give one process's steps and losses; rank 0 writes the
    /// checkpoints and every rank waits for it, so a checkpoint resumes at any world size; held-out losses are over every
    /// rank's share; dropout masks differ by rank. Not saved in checkpoints.
    /// </summary>
    public ICollectives? Cluster { get; init; }

    /// <summary>The cluster settings a <see cref="Cluster"/> run uses, or null: <see cref="Abstraction.Cluster.ClusterOptions.FromEnvironment"/>. Not saved in checkpoints.</summary>
    public ClusterOptions? ClusterOptions { get; init; }

    /// <summary>The options as JSON ("format": "idrak-pretrain/1").</summary>
    public JsonObject ToJson() => new()
    {
        ["format"] = Pretrainer.Format,
        ["stages"] = new JsonArray([.. Stages.Select(s => (JsonNode)s.ToJson())]),
        ["max_length"] = MaxLength,
        ["batch_tokens"] = BatchTokens,
        ["accumulate"] = GradientAccumulation,
        ["sequences_per_step"] = SequencesPerStep,
        ["clip"] = MaxGradientNorm,
        ["weight_decay"] = WeightDecay,
        ["optimizer"] = Optimizer,
        ["beta1"] = Beta1,
        ["beta2"] = Beta2,
        ["momentum"] = Momentum,
        ["host_optimizer"] = HostOptimizer,
        ["checkpointing"] = Checkpointing,
        ["recompute"] = RecomputeFeedForward,
        ["bf16_activations"] = BFloat16Activations,
        ["graphs"] = CudaGraphs,
        ["packing"] = Packing,
        ["seed"] = Seed,
        ["save_every"] = SaveEvery,
        ["keep_checkpoints"] = KeepCheckpoints,
        ["eval_every"] = EvaluateEvery,
        ["eval"] = new JsonArray([.. Evaluation.Select(s => (JsonNode)s.ToJson())]),
        ["eval_sequences"] = EvaluationSequences,
        ["non_finite_steps"] = NonFiniteSteps,
        ["loss_chunk_rows"] = LossChunkRows,
        ["images"] = Images.ToJson(),
        ["feature_cache"] = FeatureCache,
        ["distillation_temperature"] = DistillationTemperature,
        ["distillation_alpha"] = DistillationAlpha,
    };

    /// <summary>Options from their JSON (<see cref="ToJson"/>); keys left out keep their defaults, unknown keys are refused.</summary>
    /// <exception cref="InvalidDataException">The format, a key or a value is wrong.</exception>
    public static PretrainingOptions FromJson(JsonObject json)
    {
        ArgumentNullException.ThrowIfNull(json);
        if ((string?)json["format"] is { } format && format != Pretrainer.Format)
        {
            throw new InvalidDataException($"Pretraining options in format '{format}', not {Pretrainer.Format}.");
        }

        var known = new HashSet<string>(new PretrainingOptions().ToJson().Select(p => p.Key), StringComparer.Ordinal);
        if (json.Select(p => p.Key).FirstOrDefault(k => !known.Contains(k)) is { } unknown)
        {
            throw new InvalidDataException($"Unknown pretraining option \"{unknown}\" (known: {string.Join(", ", known.Order(StringComparer.Ordinal))}).");
        }

        var d = new PretrainingOptions();
        return new PretrainingOptions
        {
            Stages = json["stages"] is JsonArray stages ? [.. stages.Select(s => PretrainingStage.FromJson(s!.AsObject()))] : [],
            MaxLength = (int?)json["max_length"] ?? d.MaxLength,
            BatchTokens = (int?)json["batch_tokens"] ?? d.BatchTokens,
            GradientAccumulation = (int?)json["accumulate"] ?? d.GradientAccumulation,
            SequencesPerStep = (int?)json["sequences_per_step"] ?? d.SequencesPerStep,
            MaxGradientNorm = (float?)json["clip"] ?? d.MaxGradientNorm,
            WeightDecay = (float?)json["weight_decay"] ?? d.WeightDecay,
            Optimizer = (string?)json["optimizer"] ?? d.Optimizer,
            Beta1 = (float?)json["beta1"] ?? d.Beta1,
            Beta2 = (float?)json["beta2"] ?? d.Beta2,
            Momentum = (float?)json["momentum"] ?? d.Momentum,
            HostOptimizer = (bool?)json["host_optimizer"] ?? d.HostOptimizer,
            Checkpointing = (bool?)json["checkpointing"],
            RecomputeFeedForward = (bool?)json["recompute"],
            BFloat16Activations = (bool?)json["bf16_activations"],
            CudaGraphs = (bool?)json["graphs"] ?? d.CudaGraphs,
            Packing = (bool?)json["packing"] ?? d.Packing,
            Seed = (int?)json["seed"] ?? d.Seed,
            SaveEvery = (int?)json["save_every"] ?? d.SaveEvery,
            KeepCheckpoints = (int?)json["keep_checkpoints"] ?? d.KeepCheckpoints,
            EvaluateEvery = (int?)json["eval_every"] ?? d.EvaluateEvery,
            Evaluation = json["eval"] is JsonArray eval ? [.. eval.Select(s => PretrainingSource.FromJson(s!.AsObject()))] : [],
            EvaluationSequences = (int?)json["eval_sequences"] ?? d.EvaluationSequences,
            NonFiniteSteps = (int?)json["non_finite_steps"] ?? d.NonFiniteSteps,
            LossChunkRows = (int?)json["loss_chunk_rows"] ?? d.LossChunkRows,
            Images = json["images"] is JsonObject images ? TuningImages.FromJson(images) : d.Images,
            FeatureCache = (string?)json["feature_cache"] ?? d.FeatureCache,
            DistillationTemperature = (float?)json["distillation_temperature"] ?? d.DistillationTemperature,
            DistillationAlpha = (float?)json["distillation_alpha"] ?? d.DistillationAlpha,
        };
    }
}

/// <summary>A pretraining step's report.</summary>
/// <param name="Stage">The stage's name.</param>
/// <param name="StageIndex">Its index in <see cref="PretrainingOptions.Stages"/>.</param>
/// <param name="StageStep">Steps done in the stage.</param>
/// <param name="StageSteps">The stage's steps.</param>
/// <param name="Step">Steps done in the run (every stage).</param>
/// <param name="TotalSteps">The run's steps.</param>
/// <param name="Loss">The step's mean loss per trained token.</param>
/// <param name="LearningRate">The rate the step used.</param>
/// <param name="TokensPerSecond">Trained tokens per second of the step.</param>
/// <param name="ImagesPerSecond">Images per second of the step.</param>
/// <param name="DeviceBytes">The device's peak memory in use, as its backend reports it (0 when it reports none).</param>
/// <param name="Remaining">The time the run's steps left take at the speed so far.</param>
/// <param name="EvaluationLoss">The held-out loss, when this step evaluated.</param>
public sealed record PretrainingProgress(string Stage, int StageIndex, int StageStep, int StageSteps, int Step, int TotalSteps, float Loss, float LearningRate,
    double TokensPerSecond, double ImagesPerSecond, long DeviceBytes, TimeSpan Remaining, float? EvaluationLoss);

/// <summary>What <see cref="Pretrainer.Train"/> did.</summary>
/// <param name="Steps">Steps done in the run, the resumed ones included.</param>
/// <param name="Losses">Each step's loss, from the run's first step (a resumed run's earlier ones included).</param>
/// <param name="EvaluationLosses">Each evaluation's step and loss.</param>
/// <param name="LastCheckpoint">The last checkpoint written (a resumed run with no step left: the one it resumed from), or null.</param>
/// <param name="Elapsed">The time this call took.</param>
/// <param name="Settings">The memory settings the steps settled on.</param>
public sealed record PretrainingResult(int Steps, IReadOnlyList<float> Losses, IReadOnlyList<(int Step, float Loss)> EvaluationLosses, string? LastCheckpoint,
    TimeSpan Elapsed, string Settings);

/// <summary>Where a run stopped, read from a checkpoint folder (<see cref="Pretrainer.ReadCheckpoint"/>), to go on from (<see cref="Pretrainer.Train"/>).</summary>
/// <param name="Folder">The checkpoint folder (a model folder: <c>PretrainedModel.Load</c> reads it).</param>
/// <param name="Options">The run's options.</param>
/// <param name="Stage">The stage to go on in.</param>
/// <param name="StageStep">Steps done in that stage.</param>
/// <param name="Step">Steps done in the run.</param>
/// <param name="Sources">Each of the stage's sources: passes done, the next item of the pass, items drawn in the stage.</param>
/// <param name="Losses">The losses so far.</param>
/// <param name="EvaluationLosses">The evaluations so far.</param>
public sealed record PretrainingCheckpoint(string Folder, PretrainingOptions Options, int Stage, int StageStep, int Step,
    IReadOnlyList<(long Pass, long Next, long Drawn)> Sources, IReadOnlyList<float> Losses, IReadOnlyList<(int Step, float Loss)> EvaluationLosses);

/// <summary>
/// Trains every parameter of a model (one made with <c>PretrainedModel.Create</c>, or any float32 model) on token files
/// and conversation records, in stages (plan 19). Each optimizer step draws sequences from the stage's sources by their
/// weights (in a pass order shuffled from the seed), packs them into rows and runs the fine-tuner's step: the same loss
/// over trained tokens only, out-of-memory ladder, recorded graphs and image handling. Checkpoints hold the weights
/// (a model folder), the optimizer's state and where every source is, so a run stopped and resumed continues as if it
/// had not stopped.
/// </summary>
public static partial class Pretrainer
{
    /// <summary>The options' format.</summary>
    public const string Format = "idrak-pretrain/1";

    /// <summary>The checkpoint state file's format.</summary>
    public const string StateFormat = "idrak-pretrain-state/1";

    private const string StateFile = "pretrain-state.json";
    private const string OptimizerFile = "optimizer.safetensors";

    /// <summary>
    /// Runs <paramref name="options"/>' stages on <paramref name="model"/>, writing checkpoints to
    /// <paramref name="outputFolder"/> (<c>checkpoint-STEP</c>), or goes on from <paramref name="resume"/> (the model then
    /// loaded from its folder).
    /// </summary>
    /// <exception cref="ArgumentException">An option is wrong (no stages, no sources, an unknown part, …).</exception>
    /// <exception cref="InvalidOperationException">The model cannot train every parameter (packed weights, adapters), or the loss stayed not finite.</exception>
    /// <exception cref="OperationCanceledException">Cancelled: a checkpoint of the last finished step is written first.</exception>
    public static PretrainingResult Train(PretrainedModel model, PretrainingOptions options, string outputFolder, PretrainingCheckpoint? resume = null,
        IProgress<PretrainingProgress>? progress = null, CancellationToken cancellationToken = default, Action<string>? trace = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputFolder);
        Check(options);
        if (model.Network.Descendants().OfType<Idrak.Layers.Linear>().Any(l => l.PackedWeight is not null || l.Adapter is not null))
        {
            throw new InvalidOperationException("Pretraining trains every weight: load the model with float32 weights and without adapters (merge them first).");
        }

        // Pipeline placement (plan 23, C7): each rank trains its stage of the model.
        var clusterOptions = options.Cluster is null ? null : options.ClusterOptions ?? ClusterOptions.FromEnvironment();
        if (clusterOptions is { Pipelined: true })
        {
            return TrainPipelined(model, options, clusterOptions, outputFolder, resume, progress, cancellationToken, trace);
        }

        if (resume is not null && ReadPipeline(resume.Folder) is not null)
        {
            throw new NotSupportedException($"The checkpoint {resume.Folder} was written by a pipeline (IDRAK_CLUSTER_PLACEMENT=pipeline): its optimizer state is kept per stage; resume it as a pipeline.");
        }

        // A deferred model (plan 23, C8b: PretrainedOptions.Deferred) goes to its device here: whole for a rank that holds it
        // whole, unread for a parameters shard (each rank reads its own parameters as its share is made, the others' come from
        // their owners; what no shard trains is read by every rank before its stage).
        bool shardsParameters = options.Cluster is { WorldSize: > 1 } && clusterOptions is { Shard: ShardMode.Parameters };
        if (model.IsDeferred)
        {
            using (shardsParameters ? Idrak.Abstraction.Devices.DeferredTensors.StayDeferred() : null)
            {
                model.Place();
            }
        }

        options.Teacher?.Check(model.Tokenizer ?? throw new InvalidOperationException("The model has no tokenizer: its vocabulary cannot be compared with the teacher's."));
        var distillation = options.Teacher is { } teacher
            ? new FineTuner.CustomLoss(FineTuningLosses.Distillation(options.DistillationTemperature, options.DistillationAlpha), pairs: false, teacher) : null;
        Directory.CreateDirectory(outputFolder);
        var started = Stopwatch.StartNew();
        var losses = new List<float>(resume?.Losses ?? []);
        var evaluations = new List<(int, float)>(resume?.EvaluationLosses ?? []);
        int totalSteps = options.Stages.Sum(s => s.Steps);
        int step = resume?.Step ?? 0;
        string? last = resume?.Folder;                                       // a finished run resumed: nothing is written
        string settings = "";
        double secondsPerStep = 0;
        int timedSteps = 0;
        bool startedTelemetry = false;
        bool lead = options.Cluster is not { Rank: > 0 };                  // data parallel: rank 0 writes and publishes
        using var seeds = options.Cluster is { } ranked ? Dropout.SeedsForRank(ranked.Rank) : null;

        for (int stageIndex = resume?.Stage ?? 0; stageIndex < options.Stages.Count; stageIndex++)
        {
            var stage = options.Stages[stageIndex];
            int stageStep = resume is { } r && r.Stage == stageIndex ? r.StageStep : 0;
            if (stageStep >= stage.Steps)
            {
                continue;
            }

            var visionParts = stage.Trained.Contains("all", StringComparer.OrdinalIgnoreCase)
                ? (model.Vision is { } v ? VisionTuningParts.Offered(v) : [])
                : [.. stage.Trained.Where(t => t.Equals(VisionTuningParts.Tower, StringComparison.OrdinalIgnoreCase) || t.Equals(VisionTuningParts.Projector, StringComparison.OrdinalIgnoreCase))];
            bool records = stage.Sources.Concat(options.Evaluation).Any(s => s.Kind.Equals(PretrainingSources.Records, StringComparison.OrdinalIgnoreCase));
            if (visionParts.Count > 0 && model.Vision is null)
            {
                throw new ArgumentException($"Stage {stage.Name} trains {string.Join(", ", visionParts)}, but the model has no vision part.");
            }

            using var vision = model.Vision is not null && (records || visionParts.Count > 0)
                ? TuningVision.Create(model, options.Images, visionParts, FeatureCaches.Create(options.FeatureCache)) : null;
            // The parts stored beside the decoder (an audio encoder and projector), opened as soft-part sources: each trains
            // the stage's names it offers ("audio-encoder", "audio-projector"; "all": every one), the others frozen.
            using var partSources = new PartSources(model, stage, records, options.FeatureCache);
            var encoder = records && model.JinjaTemplate is { } template && model.Tokenizer is { } tokenizer
                ? new ChatTranscriptEncoder(template, tokenizer) { Vision = vision, Parts = partSources.Sources } : null;
            var context = new PretrainingContext(model, options.MaxLength, encoder);
            var readers = new List<SourceState>();
            try
            {
                for (int s = 0; s < stage.Sources.Count; s++)
                {
                    var source = stage.Sources[s];
                    var reader = PretrainingSources.Get(source.Kind).Open(source, context);
                    var position = resume is { } rr && rr.Stage == stageIndex && s < rr.Sources.Count ? rr.Sources[s] : (0L, 0L, 0L);
                    readers.Add(new SourceState(reader, source.Weight, options.Seed, stageIndex, s) { Pass = position.Item1, Next = position.Item2, Drawn = position.Item3 });
                }

                if (readers.All(rd => rd.Reader.Count == 0))
                {
                    throw new ArgumentException($"Stage {stage.Name}'s sources hold no sequences.");
                }

                var evaluation = ReadEvaluation(options, context);
                var parameters = Trainable(model, stage, vision, partSources);
                ReadUntrained(model, parameters);
                using var optimizer = CreateOptimizer(parameters, options, stage);

                // Data parallel (plan 23): every rank draws each step's sequences alike and trains its share of them.
                using var sync = options.Cluster is { } cluster ? GradientSyncs.Create(cluster, optimizer, clusterOptions!) : null;
                if (resume is { } rs && rs.Stage == stageIndex)
                {
                    // The checkpoint's whole state: a sharded run (C6) keeps its share of it, any other loads it all.
                    var saved = ReadOptimizerState(rs.Folder);
                    if (sync?.LoadOptimizerState(saved) != true)
                    {
                        optimizer.LoadState(saved);
                    }
                }

                sync?.Start();

                var schedule = FineTuningSchedules.Create(stage.Schedule, stage.WarmupFraction, stage.MinLearningRate)(optimizer, stage.Steps);
                for (int i = 0; i < stageStep; i++)
                {
                    schedule.Step();                                         // the resumed stage's rate where it was
                }

                var tuning = new FineTuningOptions
                {
                    Checkpointing = options.Checkpointing,
                    RecomputeFeedForward = options.RecomputeFeedForward,
                    BFloat16Activations = options.BFloat16Activations,
                    CudaGraphs = options.CudaGraphs,
                    GradientAccumulation = options.GradientAccumulation,
                    MaxGradientNorm = options.MaxGradientNorm,
                    HostOptimizer = options.HostOptimizer,
                    LossChunkRows = options.LossChunkRows,
                    Packing = options.Packing,
                    BatchTokens = options.BatchTokens,
                    MaxLength = options.MaxLength,
                    Vision = vision,
                    Parts = partSources.Sources,
                    // An optimizer other than the built-in AdamW: the recorded step checks it keeps its gradient buffers.
                    Optimizer = optimizer is AdamW && !options.HostOptimizer ? null : _ => optimizer,
                };

                if (!startedTelemetry && lead)
                {
                    startedTelemetry = true;
                    Telemetry.TrainingStarted(new TrainingStarted((string?)model.Config["architectures"]?[0] ?? "model", options.Optimizer, model.Device, options.Stages.Count,
                        (int)Math.Min(int.MaxValue, readers.Sum(rd => rd.Reader.Count)), evaluation.Count, Rows(options), stage.Steps,
                        parameters.Sum(p => (long)p.Size), stage.LearningRate, Environment.ProcessorCount));
                }

                FineTuner.StepRunner? runner = null;
                try
                {
                    // Where the sources are after the last finished step (a cancelled step's checkpoint goes back to it).
                    (long, long, long)[] finished = [.. readers.Select(rd => (rd.Pass, rd.Next, rd.Drawn))];
                    var next = Task.Run(() => Draw(readers, options, model));
                    while (stageStep < stage.Steps)
                    {
                        var (drawn, positions) = next.GetAwaiter().GetResult();
                        next = stageStep + 1 < stage.Steps ? Task.Run(() => Draw(readers, options, model)) : Task.FromResult<(List<TrainingSequence>, (long, long, long)[])>(([], []));
                        bool present;
                        try
                        {
                            present = sync is null || sync.BeginStep(step);
                        }
                        catch (ClusterRunStoppedException) when (step > 0 && last != Path.Combine(outputFolder, $"checkpoint-{step}") && (lead || sync is { Updates: true }))
                        {
                            // Stopped at this step boundary (a dashboard's stop, plan 23 C9): every rank stops here alike, so the
                            // checkpoint is the last finished step's, written as a saving step writes it. A sharded run's weights
                            // and optimizer state are every rank's share: every rank joins them, rank 0 writes, all wait (C6).
                            partSources.Keep();
                            if (sync is { Updates: true } && StreamedTransport(sync, clusterOptions, vision) is { } stopTo)
                            {
                                last = SaveShardedStreamed(model, sync, stopTo, optimizer, options, outputFolder, stageIndex, stageStep, step, finished, losses, evaluations);
                            }
                            else if (sync is { Updates: true })
                            {
                                using (sync.Gather())
                                {
                                    var joined = sync.SaveOptimizerState();
                                    last = sync.Rank == 0 ? SaveCheckpoint(model, vision, optimizer, options, outputFolder, stageIndex, stageStep, step, finished, losses, evaluations, joined)
                                        : Path.Combine(outputFolder, $"checkpoint-{step}");
                                }

                                sync.Barrier();
                            }
                            else
                            {
                                last = SaveCheckpoint(model, vision, optimizer, options, outputFolder, stageIndex, stageStep, step, finished, losses, evaluations);
                            }

                            throw;
                        }

                        if (!present)
                        {
                            // This rank sits the step out (a simulated late joiner before it joins, every rank past a replay's
                            // last step): it drew the step as every rank does, so the sources stay alike, and the ranks that take
                            // part share the step's sequences. Its own losses hold the steps it trained.
                            schedule.Step();
                            finished = positions;
                            step++;
                            stageStep++;
                            continue;
                        }

                        int imageTokens = drawn.Sum(q => q.Images.Sum(im => im.Tokens));
                        runner ??= new FineTuner.StepRunner(model, drawn, optimizer, tuning, Rows(options) * options.MaxLength, distillation, trace,
                            (long)Rows(options) * options.MaxLength, imageTokens, sync);
                        var sequences = sync is null ? drawn : Share(drawn, sync, runner, trace);
                        var batches = Batches(model, sequences, options);
                        float rate = optimizer.LearningRate;
                        var watch = Stopwatch.StartNew();
                        int done = step;
                        float loss;
                        long tokens;
                        try
                        {
                            (loss, tokens) = runner.Run(batches, sequences, cancellationToken,
                                b => $"{stage.Name} step {stageStep + 1}/{stage.Steps}, batch {b + 1}/{batches.Count}");
                        }
                        catch (OperationCanceledException) when (done > 0 && lead && sync is not { Updates: true })   // a sharded run's state is every rank's
                        {
                            // No update was made (a step is cancelled before its update): the checkpoint is the last finished step's.
                            partSources.Keep();
                            last = SaveCheckpoint(model, vision, optimizer, options, outputFolder, stageIndex, stageStep, done, finished, losses, evaluations);
                            throw;
                        }

                        if (!float.IsFinite(loss) && losses.Skip(Math.Max(0, losses.Count - options.NonFiniteSteps)).Count(l => !float.IsFinite(l)) >= options.NonFiniteSteps)
                        {
                            throw new InvalidOperationException($"The loss of step {step + 1} is {loss}, the last {options.NonFiniteSteps + 1} steps not finite ({runner.Settings}); "
                                + "the run stops (PretrainingOptions.NonFiniteSteps).");
                        }

                        schedule.Step();
                        finished = positions;
                        losses.Add(loss);
                        step++;
                        stageStep++;
                        sync?.Stepped(step);
                        settings = runner.Settings;
                        double seconds = watch.Elapsed.TotalSeconds;
                        if (step > (resume?.Step ?? 0) + 1)
                        {
                            secondsPerStep = (secondsPerStep * timedSteps + seconds) / ++timedSteps;   // the first step (compiling, allocating) aside
                        }

                        float? evaluationLoss = null;
                        bool stageEnd = stageStep == stage.Steps;
                        bool saving = stageEnd || options.SaveEvery > 0 && step % options.SaveEvery == 0;
                        bool evaluate = evaluation.Count > 0 && (options.EvaluateEvery > 0 ? step % options.EvaluateEvery == 0 : stageEnd);
                        // A sharded run's weights whole on every rank while they are read (plan 23, C6; nothing to do otherwise),
                        // and its optimizer state joined for the checkpoint (every rank takes part).
                        // Streamed (C8b): the checkpoint takes each rank's share from it a tensor at a time, gathering nothing whole.
                        var streamTo = saving ? StreamedTransport(sync, clusterOptions, vision) : null;
                        using var whole = evaluate || saving && streamTo is null ? sync?.Gather() : null;
                        var joinedState = saving && streamTo is null ? sync?.SaveOptimizerState() : null;
                        if (evaluate)
                        {
                            evaluationLoss = sync is null ? FineTuner.Evaluate(model, evaluation, options.BatchTokens, options.LossChunkRows, trace, vision, partSources.Sources)
                                : FineTuner.EvaluateAcross(model, evaluation, sync, options.BatchTokens, options.LossChunkRows, trace, vision, partSources.Sources);
                            evaluations.Add((step, evaluationLoss.Value));
                        }

                        if (lead && Telemetry.IsEnabled(TelemetryLevel.Batches))
                        {
                            Telemetry.BatchCompleted(new BatchCompleted(stageIndex + 1, stageStep, stage.Steps, step, drawn.Count, loss, rate, null, TimeSpan.Zero, watch.Elapsed));
                        }

                        if (saving)
                        {
                            // The stage it is in and its steps done (a finished stage is skipped on resume, or goes on when its steps grew).
                            // Data parallel: rank 0 writes it (where every source is over every rank's draws, so it resumes at
                            // any world size) and every rank waits until it is written.
                            partSources.Keep();
                            if (streamTo is not null)
                            {
                                last = SaveShardedStreamed(model, sync!, streamTo, optimizer, options, outputFolder, stageIndex, stageStep, step, positions, losses, evaluations);
                            }
                            else
                            {
                                last = (sync is null ? lead : sync.Rank == 0) ? SaveCheckpoint(model, vision, optimizer, options, outputFolder, stageIndex, stageStep, step, positions, losses, evaluations, joinedState)
                                    : Path.Combine(outputFolder, $"checkpoint-{step}");
                                sync?.Barrier();
                            }

                            trace?.Invoke($"checkpoint {last}");
                        }

                        long peak = model.Device.Backend.GetMemoryUsage().Peak;
                        var remaining = TimeSpan.FromSeconds(secondsPerStep * (totalSteps - step));
                        int images = drawn.Sum(q => q.Images.Count);
                        progress?.Report(new PretrainingProgress(stage.Name, stageIndex, stageStep, stage.Steps, step, totalSteps, loss, rate,
                            tokens / Math.Max(1e-9, seconds), images / Math.Max(1e-9, seconds), peak, remaining, evaluationLoss));
                    }

                    sync?.End();                                             // a sharded run's weights whole on every rank again (C6)

                    // The vision parts this stage trained live in its encoder: the model keeps them, so the next stage's encoder
                    // (and every save) starts from them.
                    if (vision is { Parts.Count: > 0 })
                    {
                        model.KeepTrainedVision(vision.Encoder, vision.Parts);
                    }

                    partSources.Keep();                                      // the parts' trained tensors too
                }
                finally
                {
                    runner?.Dispose();
                }
            }
            finally
            {
                readers.ForEach(rd => rd.Reader.Dispose());
            }

            resume = null;
        }

        model.Network.Eval();
        if (lead)
        {
            Telemetry.TrainingCompleted(new TrainingCompleted(options.Stages.Count, started.Elapsed, losses.Count > 0 ? losses[^1] : double.NaN,
                options.Stages.Count, evaluations.Count > 0 ? evaluations.Min(e => e.Item2) : double.NaN, false, false));
        }

        return new PretrainingResult(step, losses, evaluations, last, started.Elapsed, settings);
    }

    // The model's tensors still unread on the device (a parameters shard's deferred load, C8b) that this stage does not train:
    // every rank holds them whole, so each reads them now; the trained ones are read by their owners as the shares are made.
    private static void ReadUntrained(PretrainedModel model, List<Tensor> trained)
    {
        var mine = new HashSet<Tensor>(trained, ReferenceEqualityComparer.Instance);
        foreach (var tensor in model.Network.Parameters().Concat(model.Network.Buffers()))
        {
            if (tensor.Storage.Evicted && !mine.Contains(tensor) && tensor.Storage.Recompute is not null)
            {
                tensor.Backend.Restore(tensor.Storage);
            }
        }
    }

    // The transport a sharded run's checkpoint is streamed over (C8b), or null: the shares are joined whole (streaming off,
    // a vision part's tensors beside the model, or collectives that give no transport of their own).
    private static ITransport? StreamedTransport(IGradientSync? sync, ClusterOptions? cluster, TuningVision? vision) =>
        sync is { Updates: true, WorldSize: > 1 } && cluster is { StreamedSave: true } && vision is null ? sync.Collectives.Transport : null;

    // This rank's share of a step's drawn sequences: draws k·W + r while the shares are equal, else by the ranks' measured
    // speeds over the sequences' trained tokens; the step's totals over every rank go to the runner (a loss of one's own
    // divides by them).
    private static List<TrainingSequence> Share(List<TrainingSequence> drawn, IGradientSync sync, FineTuner.StepRunner runner, Action<string>? trace)
    {
        var sizes = drawn.Select(q => (long)q.TrainedTokens).ToArray();
        var owners = DrawShares.Owners(sizes, sync.Shares);
        runner.StepTotals = ((float)Math.Max(1, sizes.Sum()), drawn.Count);
        runner.TraceShares(trace);
        return [.. drawn.Where((_, i) => owners[i] == sync.Rank)];
    }

    // Rows of a micro-batch.
    private static int Rows(PretrainingOptions options) => Math.Max(1, options.BatchTokens / options.MaxLength);

    // A stage's parts stored beside the decoder (PretrainedModel.Parts), opened as trainable soft-part sources when the stage
    // reads conversations or trains one: each trains the stage's names it offers ("all": every one) and the rest frozen. A
    // name the stage gives that is neither the decoder's, the vision part's nor a part's is refused, naming them all.
    private sealed class PartSources : IDisposable
    {
        private readonly List<(ITrainableSoftParts Source, IReadOnlyList<string> Trained)> _opened = [];
        private readonly PretrainedModel _model;

        public PartSources(PretrainedModel model, PretrainingStage stage, bool records, string featureCache)
        {
            _model = model;
            bool all = stage.Trained.Contains("all", StringComparer.OrdinalIgnoreCase);
            string[] own = ["decoder", "embeddings", "head", VisionTuningParts.Tower, VisionTuningParts.Projector, "all"];
            try
            {
                foreach (string kind in model.Parts)
                {
                    var source = model.OpenPart(kind, model.Device, FeatureCaches.Create(featureCache));
                    var trained = source.TrainableParts.Where(p => all || stage.Trained.Contains(p, StringComparer.OrdinalIgnoreCase)).ToList();
                    if (!records && trained.Count == 0)
                    {
                        source.Dispose();                                    // neither read nor trained this stage
                        continue;
                    }

                    _opened.Add((source, trained));
                    Parameters.AddRange(source.Train(trained));
                    source.SetTraining(trained.Count > 0);
                }

                var offered = own.Concat(_opened.SelectMany(o => o.Source.TrainableParts)).ToList();
                if (stage.Trained.Any(t => !offered.Contains(t, StringComparer.OrdinalIgnoreCase)) && _opened.Count < model.Parts.Count)
                {
                    offered.AddRange(KnownNames(model));                    // the parts this stage left closed, for the message
                }

                if (stage.Trained.FirstOrDefault(t => !offered.Contains(t, StringComparer.OrdinalIgnoreCase)) is { } unknown)
                {
                    throw new ArgumentException($"Stage {stage.Name} trains '{unknown}': the parts are {string.Join(", ", offered.Distinct(StringComparer.OrdinalIgnoreCase))}.");
                }
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        public List<Tensor> Parameters { get; } = [];

        public IReadOnlyList<ISoftPartSource> Sources => [.. _opened.Select(o => o.Source)];

        // The trained parts' tensors kept by the model: the next stage's sources and every save start from them.
        public void Keep()
        {
            foreach (var (source, trained) in _opened)
            {
                if (trained.Count > 0)
                {
                    _model.KeepTrainedParts(source.Export(trained));
                }
            }
        }

        public void Dispose()
        {
            _opened.ForEach(o => o.Source.Dispose());
            _opened.Clear();
        }

        // The names a model's parts offer, read by opening each once (a stage that neither reads nor trains them).
        private static IEnumerable<string> KnownNames(PretrainedModel model)
        {
            var names = new List<string>();
            foreach (string kind in model.Parts)
            {
                using var source = model.OpenPart(kind);
                names.AddRange(source.TrainableParts);
            }

            return names;
        }
    }

    private static void Check(PretrainingOptions options)
    {
        if (options.Stages.Count == 0)
        {
            throw new ArgumentException("Give at least one pretraining stage.", nameof(options));
        }

        if (options.DistillationAlpha is < 0 or > 1 || options.DistillationTemperature <= 0)
        {
            throw new ArgumentException($"DistillationAlpha must be 0 to 1 ({options.DistillationAlpha}) and DistillationTemperature positive ({options.DistillationTemperature}).", nameof(options));
        }

        if (options.MaxLength < 1 || options.BatchTokens < 1 || options.GradientAccumulation < 1 || options.SequencesPerStep < 0)
        {
            throw new ArgumentException($"MaxLength ({options.MaxLength}), BatchTokens ({options.BatchTokens}) and GradientAccumulation ({options.GradientAccumulation}) must be positive.", nameof(options));
        }

        if (!FineTuningOptimizers.Names.Contains(options.Optimizer, StringComparer.OrdinalIgnoreCase))
        {
            throw new ArgumentException($"Optimizer '{options.Optimizer}' is not one of {string.Join(", ", FineTuningOptimizers.Names)}.", nameof(options));
        }

        // Names outside these are the model's parts' (ModelParts: "audio-encoder", say), checked when the stage opens them.
        string[] parts = ["decoder", "embeddings", "head", VisionTuningParts.Tower, VisionTuningParts.Projector, "all"];
        foreach (var stage in options.Stages)
        {
            if (stage.Steps < 1)
            {
                throw new ArgumentException($"Stage {stage.Name} has {stage.Steps} steps; give at least one.", nameof(options));
            }

            if (stage.Sources.Count == 0)
            {
                throw new ArgumentException($"Stage {stage.Name} reads no sources.", nameof(options));
            }

            if (stage.Trained.Count == 0 || stage.Trained.Any(string.IsNullOrWhiteSpace))
            {
                throw new ArgumentException($"Stage {stage.Name} trains '{string.Join(", ", stage.Trained)}': the parts are {string.Join(", ", parts)}.", nameof(options));
            }

            if (stage.Sources.Any(s => s.Weight <= 0 || !double.IsFinite(s.Weight)))
            {
                throw new ArgumentException($"Stage {stage.Name}: every source's weight must be positive.", nameof(options));
            }

            if (!FineTuningSchedules.Names.Contains(stage.Schedule, StringComparer.OrdinalIgnoreCase))
            {
                throw new ArgumentException($"Stage {stage.Name}: schedule '{stage.Schedule}' is not one of {string.Join(", ", FineTuningSchedules.Names)}.", nameof(options));
            }
        }
    }

    // The parameters a stage trains, in the network's order (then the vision parts', then the other parts'), every other one frozen.
    private static List<Tensor> Trainable(PretrainedModel model, PretrainingStage stage, TuningVision? vision, PartSources partSources)
    {
        bool all = stage.Trained.Contains("all", StringComparer.OrdinalIgnoreCase);
        bool Trains(string part) => all || stage.Trained.Contains(part, StringComparer.OrdinalIgnoreCase);
        foreach (var p in model.Network.Parameters())
        {
            p.RequiresGrad = false;
        }

        var chosen = new List<Tensor>();
        var seen = new HashSet<Tensor>(ReferenceEqualityComparer.Instance);
        foreach (var (path, module) in model.NamedModules())
        {
            if (module is Idrak.Layers.Linear { TiedTo: not null })
            {
                continue;
            }

            string top = path.Split('.')[0];
            string part = top is "embed" or "pos" ? "embeddings" : top == "head" ? "head" : "decoder";
            var own = module.Parameters().Except(module.Children().SelectMany(c => c.Parameters()), ReferenceEqualityComparer.Instance).Cast<Tensor>();
            foreach (var p in own)
            {
                if (Trains(part) && seen.Add(p))
                {
                    p.RequiresGrad = true;
                    chosen.Add(p);
                }
            }
        }

        foreach (var p in (vision?.Parameters ?? []).Concat(partSources.Parameters))
        {
            if (seen.Add(p))
            {
                chosen.Add(p);
            }
        }

        return chosen.Count > 0 ? chosen : throw new ArgumentException($"Stage {stage.Name} trains nothing of this model ({string.Join(", ", stage.Trained)}).");
    }

    private static Optimizer CreateOptimizer(List<Tensor> parameters, PretrainingOptions options, PretrainingStage stage)
    {
        Func<IReadOnlyList<Tensor>, Optimizer> make = options.Optimizer.ToLowerInvariant() switch
        {
            "adamw" => ps => new AdamW(ps, stage.LearningRate, options.Beta1, options.Beta2, 1e-8f, options.WeightDecay),
            "adam" => ps => new Adam(ps, stage.LearningRate, options.Beta1, options.Beta2, 1e-8f, options.WeightDecay),
            "adamw8bit" => ps => new AdamW8Bit(ps, stage.LearningRate, options.Beta1, options.Beta2, 1e-8f, options.WeightDecay),
            _ => FineTuningOptimizers.Create(options.Optimizer, stage.LearningRate, options.WeightDecay, options.Momentum),
        };
        return options.HostOptimizer ? new HostOptimizer(parameters, make) : make(parameters);
    }

    // The step's sequences, drawn source by source (the one furthest behind its share next), each source in a pass order
    // shuffled from the seed; and every source's position after them.
    private static (List<TrainingSequence>, (long, long, long)[]) Draw(List<SourceState> readers, PretrainingOptions options, PretrainedModel model)
    {
        long capacity = (long)Rows(options) * options.MaxLength * options.GradientAccumulation;
        var sequences = new List<TrainingSequence>();
        long positions = 0;
        double total = readers.Where(r => r.Reader.Count > 0).Sum(r => r.Weight);
        int misses = 0;
        int wanted = options.SequencesPerStep;
        while (wanted > 0 ? sequences.Count < wanted : positions < capacity)
        {
            long drawn = readers.Sum(r => r.Drawn);
            var source = readers.Where(r => r.Reader.Count > 0).MaxBy(r => r.Weight / total * (drawn + 1) - r.Drawn)!;
            var sequence = source.Take();
            if (sequence is null || sequence.Tokens.Length < 2)
            {
                if (++misses > readers.Sum(r => r.Reader.Count))
                {
                    throw new InvalidDataException("The stage's sources give no sequence to train on (every record has nothing to learn).");
                }

                continue;
            }

            if (sequence.Tokens.Length - 1 > options.MaxLength)
            {
                throw new InvalidDataException($"A sequence of {sequence.Tokens.Length - 1} positions is longer than MaxLength {options.MaxLength}.");
            }

            long length = sequence.Tokens.Length - 1;
            if (wanted == 0 && sequences.Count > 0 && positions + length > capacity)
            {
                source.Undo();
                break;
            }

            sequences.Add(sequence);
            positions += length;
        }

        return (sequences, [.. readers.Select(r => (r.Pass, r.Next, r.Drawn))]);
    }

    // The micro-batches of a step: packed rows of MaxLength when the model supports it, else one sequence a row.
    private static List<FineTuner.Batch> Batches(PretrainedModel model, List<TrainingSequence> sequences, PretrainingOptions options)
    {
        int rows = Rows(options);
        bool packs = options.Packing && Idrak.Layers.PackedSequences.Supports(model.Network);
        if (packs)
        {
            return [.. FineTuner.PackedBatches(sequences, options.MaxLength, rows, null).Select(b => new FineTuner.Batch(b, options.MaxLength, true))];
        }

        var batches = new List<FineTuner.Batch>();
        for (int first = 0; first < sequences.Count; first += rows)
        {
            batches.Add(FineTuner.Batch.Padded([.. Enumerable.Range(first, Math.Min(rows, sequences.Count - first))], sequences));
        }

        return batches;
    }

    private static List<TrainingSequence> ReadEvaluation(PretrainingOptions options, PretrainingContext context)
    {
        var sequences = new List<TrainingSequence>();
        foreach (var source in options.Evaluation)
        {
            using var reader = PretrainingSources.Get(source.Kind).Open(source, context);
            for (long i = 0; i < reader.Count && sequences.Count < options.EvaluationSequences * (options.Evaluation.Count); i++)
            {
                if (reader.Read(i) is { } sequence && sequence.Tokens.Length >= 2)
                {
                    sequences.Add(sequence);
                }
            }
        }

        return sequences;
    }

    // A source being drawn from: its reader, its pass order (shuffled from the seed, the stage and the pass), and where it is.
    private sealed class SourceState(IPretrainingReader reader, double weight, int seed, int stage, int index)
    {
        private int[]? _order;
        private long _orderPass = -1;

        public IPretrainingReader Reader { get; } = reader;

        public double Weight { get; } = weight;

        public long Pass { get; set; }

        public long Next { get; set; }

        public long Drawn { get; set; }

        public TrainingSequence? Take()
        {
            if (Next >= Reader.Count)
            {
                (Pass, Next) = (Pass + 1, 0);
            }

            long item = Order()[Next];
            Next++;
            Drawn++;
            return Reader.Read(item);
        }

        public void Undo()
        {
            Next--;
            Drawn--;
            if (Next < 0)
            {
                (Pass, Next) = (Pass - 1, Reader.Count - 1);
            }
        }

        private int[] Order()
        {
            if (_orderPass == Pass && _order is not null)
            {
                return _order;
            }

            if (Reader.Count > int.MaxValue)
            {
                throw new NotSupportedException($"A source of {Reader.Count} sequences is more than one pass order holds; split it into several sources.");
            }

            var order = new int[Reader.Count];
            for (int i = 0; i < order.Length; i++)
            {
                order[i] = i;
            }

            // Fisher-Yates from a seed of the run's seed, the stage, the source and the pass (the same order on every run and resume).
            var random = new Random(unchecked(seed * 1_000_003 + stage * 10_007 + index * 101 + (int)Pass));
            for (int i = order.Length - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (order[i], order[j]) = (order[j], order[i]);
            }

            (_order, _orderPass) = (order, Pass);
            return order;
        }
    }

    // A checkpoint: the model folder, the optimizer's state, and where the run is.
    private static string SaveCheckpoint(PretrainedModel model, TuningVision? vision, Optimizer optimizer, PretrainingOptions options, string outputFolder, int stage,
        int stageStep, int step, (long, long, long)[] positions, List<float> losses, List<(int, float)> evaluations,
        IReadOnlyDictionary<string, OptimizerStateValue>? joinedState = null)
    {
        string folder = Path.Combine(outputFolder, $"checkpoint-{step}");

        // The trained vision parts as the checkpoint stores them (the rest of the vision part is copied as it was).
        var exported = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        try
        {
            if (vision?.Tuning is { } tuning)
            {
                foreach (string part in vision.Parts)
                {
                    foreach (var (name, tensor) in tuning.Export(vision.Encoder, part))
                    {
                        exported[name] = tensor;
                    }
                }
            }

            model.SaveHuggingFace(folder, Idrak.Models.SafeTensorType.F32, exported);
        }
        finally
        {
            foreach (var tensor in exported.Values)
            {
                tensor.Dispose();
            }
        }

        var state = joinedState ?? optimizer.SaveState();               // a sharded run's: every rank's share joined (C6)
        Idrak.Models.SafeTensorsWriter.Write(Path.Combine(folder, OptimizerFile), state.Select(p => (p.Key, p.Value.Shape, p.Value.Values)),
            Idrak.Models.SafeTensorType.F32, new Dictionary<string, string> { ["format"] = "idrak-optimizer", ["optimizer"] = optimizer.GetType().Name });
        WriteState(options, folder, outputFolder, stage, stageStep, step, positions, losses, evaluations);
        return folder;
    }

    // A sharded run's checkpoint streamed to rank 0 (plan 23, C8b): every rank answers rank 0's requests for its own weights
    // (in memory without a collective, IGradientSync.Local) and its own optimizer entries a tensor at a time, while rank 0
    // writes the files one process writes, byte for byte the joined checkpoint (the entries in rank order, as C6 joined them):
    // nothing is gathered whole on any rank. Every rank calls it; rank 0 returns the folder it wrote, every rank waits for it.
    private static string SaveShardedStreamed(PretrainedModel model, IGradientSync sync, ITransport transport, Optimizer optimizer, PretrainingOptions options,
        string outputFolder, int stage, int stageStep, int step, (long, long, long)[] positions, List<float> losses, List<(int, float)> evaluations)
    {
        string folder = Path.Combine(outputFolder, $"checkpoint-{step}");
        using (sync.Local())
        {
            var trained = sync.Parameters;
            if (sync.Rank != 0)
            {
                StreamedCheckpoint.Serve(transport, 0, i => trained[i].ToArray(), sync.SaveOwnOptimizerState());
            }
            else
            {
                var index = new Dictionary<Tensor, int>(ReferenceEqualityComparer.Instance);
                for (int i = 0; i < trained.Count; i++)
                {
                    index.TryAdd(trained[i], i);
                }

                var ranks = Enumerable.Range(1, sync.WorldSize - 1).ToList();
                var entries = ranks.Select(rank => StreamedCheckpoint.AskEntries(transport, rank)).ToList();
                model.SaveHuggingFace(folder, Idrak.Models.SafeTensorType.F32, null, stored =>
                    index.TryGetValue(stored.Tensor, out int i) && sync.Owner(stored.Tensor) is { } owner && owner != 0
                        ? StreamedCheckpoint.AskWeight(transport, owner, i)
                        : stored.Tensor.ToArray());
                var own = sync.SaveOwnOptimizerState() ?? optimizer.SaveState();
                var tensors = own.Select(e => (e.Key, e.Value.Shape, (Func<float[]>)(() => e.Value.Values))).ToList();
                for (int r = 0; r < ranks.Count; r++)
                {
                    int rank = ranks[r];
                    tensors.AddRange(entries[r].Select((e, i) => (e.Name, e.Shape, (Func<float[]>)(() => StreamedCheckpoint.AskEntry(transport, rank, i)))));
                }

                Idrak.Models.SafeTensorsWriter.Write(Path.Combine(folder, OptimizerFile), tensors, Idrak.Models.SafeTensorType.F32,
                    new Dictionary<string, string> { ["format"] = "idrak-optimizer", ["optimizer"] = optimizer.GetType().Name });
                ranks.ForEach(rank => StreamedCheckpoint.Finish(transport, rank));
                WriteState(options, folder, outputFolder, stage, stageStep, step, positions, losses, evaluations);
            }
        }

        sync.Barrier();
        return folder;
    }

    // The run's state file (last: a checkpoint counts once it is there), and the checkpoints beyond those kept deleted.
    private static void WriteState(PretrainingOptions options, string folder, string outputFolder, int stage, int stageStep, int step,
        (long, long, long)[] positions, List<float> losses, List<(int, float)> evaluations)
    {
        var json = new JsonObject
        {
            ["format"] = StateFormat,
            ["stage"] = stage,
            ["stage_step"] = stageStep,
            ["step"] = step,
            ["sources"] = new JsonArray([.. positions.Select(p => (JsonNode)new JsonObject { ["pass"] = p.Item1, ["next"] = p.Item2, ["drawn"] = p.Item3 })]),
            ["losses"] = new JsonArray([.. losses.Select(l => (JsonNode)(float.IsFinite(l) ? l : float.NaN.ToString(CultureInfo.InvariantCulture)))]),
            ["evaluations"] = new JsonArray([.. evaluations.Select(e => (JsonNode)new JsonObject { ["step"] = e.Item1, ["loss"] = e.Item2 })]),
            ["options"] = options.ToJson(),
        };
        File.WriteAllText(Path.Combine(folder, StateFile), json.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }) + "\n");
        if (options.KeepCheckpoints > 0)
        {
            foreach (var old in Checkpoints(outputFolder).SkipLast(options.KeepCheckpoints))
            {
                Directory.Delete(old.Folder, recursive: true);
            }
        }
    }

    private static IEnumerable<(int Step, string Folder)> Checkpoints(string outputFolder) =>
        Directory.Exists(outputFolder)
            ? Directory.GetDirectories(outputFolder, "checkpoint-*")
                .Select(f => (Ok: int.TryParse(Path.GetFileName(f)["checkpoint-".Length..], NumberStyles.None, CultureInfo.InvariantCulture, out int s), Step: s, Folder: f))
                .Where(c => c.Ok && File.Exists(Path.Combine(c.Folder, StateFile))).OrderBy(c => c.Step).Select(c => (c.Step, c.Folder))
            : [];

    /// <summary>The latest checkpoint a run wrote into <paramref name="outputFolder"/>, or null.</summary>
    public static string? LatestCheckpoint(string outputFolder) => Checkpoints(outputFolder).Select(c => c.Folder).LastOrDefault();

    /// <summary>Reads where a run stopped from a checkpoint folder <see cref="Train"/> wrote.</summary>
    /// <exception cref="FileNotFoundException">The folder holds no pretraining state.</exception>
    /// <exception cref="InvalidDataException">The state is not this format.</exception>
    public static PretrainingCheckpoint ReadCheckpoint(string folder)
    {
        string path = Path.Combine(folder, StateFile);
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"{folder} holds no pretraining state ({StateFile}); give a checkpoint-STEP folder a run wrote.", path);
        }

        var json = JsonNode.Parse(File.ReadAllText(path))!.AsObject();
        if ((string?)json["format"] != StateFormat)
        {
            throw new InvalidDataException($"{path} is in format '{json["format"]}', not {StateFormat}.");
        }

        return new PretrainingCheckpoint(folder, PretrainingOptions.FromJson(json["options"]!.AsObject()), (int)json["stage"]!, (int)json["stage_step"]!, (int)json["step"]!,
            [.. json["sources"]!.AsArray().Select(s => ((long)s!["pass"]!, (long)s["next"]!, (long)s["drawn"]!))],
            [.. json["losses"]!.AsArray().Select(l => l is JsonValue v && v.TryGetValue(out float f) ? f : float.NaN)],
            [.. json["evaluations"]!.AsArray().Select(e => ((int)e!["step"]!, (float)e["loss"]!))]);
    }

    // The checkpoint's optimizer entries (those `keep` takes: a pipeline stage reads its own alone, plan 23 C8b).
    private static Dictionary<string, OptimizerStateValue> ReadOptimizerState(string folder, Func<string, bool>? keep = null)
    {
        string path = Path.Combine(folder, OptimizerFile);
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"{folder} holds no optimizer state ({OptimizerFile}).", path);
        }

        using var reader = Idrak.Models.SafeTensorsReader.Open(path);
        ITensorStore store = reader;
        return store.Names.Where(n => keep?.Invoke(n) ?? true).ToDictionary(n => n, n => new OptimizerStateValue(store.ShapeOf(n), store.Read(n)), StringComparer.Ordinal);
    }
}

// Token files cut into windows of the maximum length (and the next token as the last target), every token trained.
internal sealed class TokenPretrainingSource : IPretrainingSource
{
    public string Name => PretrainingSources.Tokens;

    public string Summary => "token files (uint16, int32, uint32 or .npy, as idrak data tokens writes them), cut into windows of the maximum length, every token trained";

    public IPretrainingReader Open(PretrainingSource source, PretrainingContext context)
    {
        var files = new List<TokenFileSource>();
        try
        {
            foreach (string path in source.Paths)
            {
                if (!File.Exists(path))
                {
                    throw new FileNotFoundException($"{path} not found.", path);
                }

                files.Add(new TokenFileSource(path, context.MaxLength, type: source.TokenType));
            }

            return new Reader(files, context.MaxLength);
        }
        catch
        {
            files.ForEach(f => f.Dispose());
            throw;
        }
    }

    private sealed class Reader(List<TokenFileSource> files, int length) : IPretrainingReader
    {
        private readonly long[] _starts = Starts(files);
        private readonly float[] _features = new float[length], _targets = new float[length];

        public long Count => _starts[^1];

        private static long[] Starts(List<TokenFileSource> files)
        {
            var starts = new long[files.Count + 1];
            for (int i = 0; i < files.Count; i++)
            {
                starts[i + 1] = starts[i] + files[i].Count;
            }

            return starts;
        }

        public TrainingSequence? Read(long index)
        {
            int file = Array.BinarySearch(_starts, index);
            file = file >= 0 ? file : ~file - 1;
            while (file + 1 < _starts.Length - 1 && _starts[file + 1] <= index)
            {
                file++;
            }

            files[file].Read((int)(index - _starts[file]), _features, _targets);
            var tokens = new int[length + 1];
            for (int i = 0; i < length; i++)
            {
                tokens[i] = (int)_features[i];
            }

            tokens[length] = (int)_targets[length - 1];
            var trained = new bool[length + 1];
            Array.Fill(trained, true);
            return new TrainingSequence(tokens, trained);
        }

        public void Dispose() => files.ForEach(f => f.Dispose());
    }
}

// Conversations in a tuning data format, encoded through the model's chat template (answers trained), with images.
internal sealed class RecordPretrainingSource : IPretrainingSource
{
    public string Name => PretrainingSources.Records;

    public string Summary => "conversations in a tuning data format (messages, sharegpt), with images when the model has a vision part; the answers trained";

    public IPretrainingReader Open(PretrainingSource source, PretrainingContext context)
    {
        var encoder = context.Encoder ?? throw new InvalidOperationException("Records are encoded through the model's chat template; this model has none (give it a chat_template).");
        var files = new List<PendingConversations>();
        try
        {
            foreach (string path in source.Paths)
            {
                if (!File.Exists(path))
                {
                    throw new FileNotFoundException($"{path} not found.", path);
                }

                var format = source.Format is { } name ? TuningDataFormats.Get(name) : TuningDataFormats.Detect(path);
                files.Add(format.ReadPending(path, new TuningDataOptions { Images = source.Images, Audios = source.Audios }));
            }

            return new Reader(files, encoder, context.MaxLength);
        }
        catch
        {
            files.ForEach(f => f.Dispose());
            throw;
        }
    }

    private sealed class Reader(List<PendingConversations> files, ChatTranscriptEncoder encoder, int length) : IPretrainingReader
    {
        public long Count => files.Sum(f => (long)f.Count);

        public TrainingSequence? Read(long index)
        {
            foreach (var file in files)
            {
                if (index < file.Count)
                {
                    return encoder.Encode(file.Read((int)index), length);
                }

                index -= file.Count;
            }

            throw new ArgumentOutOfRangeException(nameof(index));
        }

        public void Dispose() => files.ForEach(f => f.Dispose());
    }
}
