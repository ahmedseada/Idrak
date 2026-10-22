// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Abstraction.Cluster;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Text.Abstractions;
using Idrak.Text.Generation;

namespace Idrak.Text;

/// <summary>What one <see cref="FineTuningSession.Step"/> did.</summary>
/// <param name="Step">The optimizer steps done after it (a resumed session's included).</param>
/// <param name="Loss">The step's loss: the mean loss per trained token, or the loss of one's own.</param>
/// <param name="LearningRate">The learning rate the step's update used.</param>
/// <param name="Tokens">Tokens (prompts included) the step covered.</param>
/// <param name="Duration">The step's wall time.</param>
/// <param name="Skipped">True when this rank sat the step out (a cluster's late joiner): nothing was trained here.</param>
public sealed record FineTuningStepStats(int Step, float Loss, float LearningRate, long Tokens, TimeSpan Duration, bool Skipped = false)
{
    /// <summary>Tokens per second over the step.</summary>
    public double TokensPerSecond => Duration.TotalSeconds > 0 ? Tokens / Duration.TotalSeconds : 0;
}

/// <summary>
/// A fine-tune held open between steps: opened once with a model and <see cref="FineTuningOptions"/> (adapters, optimizer,
/// schedule, FP8 copies, the cluster, device memory), it keeps what a run keeps across its steps, so a caller can train a
/// step, generate with the adapters as they are, and train the next (reinforcement learning, on-policy distillation).
/// <see cref="FineTuner.Train(PretrainedModel, IReadOnlyList{TrainingSequence}, IReadOnlyList{TrainingSequence}?, FineTuningOptions, string?, IProgress{FineTuningProgress}?, CancellationToken, Action{string}?)"/>
/// is a loop over one.
/// <list type="bullet">
/// <item><see cref="Step"/>: one optimizer step over the given sequences, with the token cross-entropy or a loss of one's own
/// (a policy loss through <see cref="FineTuningLosses.Policy"/>, on-policy distillation).</item>
/// <item><see cref="LogProbabilities"/>: log p of the trained tokens with the adapters on or off, no gradient (behaviour
/// and reference values).</item>
/// <item><see cref="Generator"/>: a chat model on the live adapters (no key/value cache kept between replies).</item>
/// <item><see cref="Save"/>, <see cref="Checkpoint"/> (resumed with <see cref="FineTuningOptions.ResumeFrom"/> or
/// <see cref="Restore"/>), <see cref="IDisposable.Dispose"/>.</item>
/// </list>
/// Generation and training share the device's pool: a reply's key/value cache is freed when the reply ends (and every
/// generator of the session is told to release what it holds before a step), and a step's activations when the step ends,
/// so the device holds one of the two at a time. Not thread-safe: one caller drives it.
/// </summary>
public sealed class FineTuningSession : IDisposable
{
    private readonly TuningVision? _ownedVision;
    private readonly IDisposable? _seeds;
    private readonly bool _images;
    private readonly Action<string>? _trace;
    private readonly FineTuner.CustomLoss? _defaultCustom;
    private readonly Random _random;
    private readonly List<TextGenerator> _generators = [];
    private IDisposable? _float8;
    private bool _float8Pending;
    private FineTuner.StepRunner? _runner;
    private LearningRateScheduler? _schedule;
    private FineTuningLoss? _lastLoss;
    private FineTuner.CustomLoss? _lastCustom;
    private int _nonFinite;
    private bool _disposed;

    /// <summary>
    /// Opens a session on <paramref name="model"/>: adds adapters (unless it has some; resumed from
    /// <see cref="FineTuningOptions.ResumeFrom"/> when set), makes the optimizer and a schedule of
    /// <paramref name="totalSteps"/> steps, and joins the cluster when <see cref="FineTuningOptions.Cluster"/> is set.
    /// </summary>
    /// <param name="model">The model to tune (its adapters train; the rest stays frozen).</param>
    /// <param name="options">Settings; <see cref="FineTuningOptions.Epochs"/>, <see cref="FineTuningOptions.EvaluateEvery"/> and <see cref="FineTuningOptions.SaveEvery"/> are a run's, not read here.</param>
    /// <param name="totalSteps">The optimizer steps the schedule spans (the learning rate's warm-up and decay are over them).</param>
    /// <param name="sample">
    /// Sequences like the ones to come, for what is decided once: whether they hold images (the vision side), and the FP8
    /// check (<see cref="FineTuningOptions.Float8"/>; without a sample it runs on the first step's sequences). Null: none.
    /// </param>
    /// <param name="trace">Receives a line per batch and per decision, as in <see cref="FineTuner.Train(PretrainedModel, IReadOnlyList{TrainingSequence}, IReadOnlyList{TrainingSequence}?, FineTuningOptions, string?, IProgress{FineTuningProgress}?, CancellationToken, Action{string}?)"/>.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="totalSteps"/> is not positive.</exception>
    public FineTuningSession(PretrainedModel model, FineTuningOptions options, int totalSteps, IReadOnlyList<TrainingSequence>? sample = null, Action<string>? trace = null)
        : this(model ?? throw new ArgumentNullException(nameof(model)), options ?? throw new ArgumentNullException(nameof(options)), sample ?? [], null, pairs: false,
            FineTuner.Custom(model, options, pairs: false), trace)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(totalSteps);
        try
        {
            _float8Pending = Options.Float8 && (sample is null || sample.Count == 0);
            // Room for a whole batch of trained tokens (a recorded step's capacity) and the largest pass a batch can be.
            int rows = Math.Max(Options.BatchTokens, Options.MaxLength);
            Begin(totalSteps, sample ?? [], rows, rows, 0);
            while (Steps < ResumedSteps)
            {
                Skip();                                                    // the schedule where the checkpoint left it
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    // What a run sets up before its steps (FineTuner.Train's order): the image side, adapters (resumed), the trained
    // parameters, FP8 copies, the optimizer, the cluster's sync (its optimizer state loaded) and its start. Begin follows,
    // once the run knows its steps and its batches' sizes.
    internal FineTuningSession(PretrainedModel model, FineTuningOptions options, IReadOnlyList<TrainingSequence> train, IReadOnlyList<TrainingSequence>? evaluation,
        bool pairs, FineTuner.CustomLoss? custom, Action<string>? trace)
    {
        Model = model;
        Pairs = pairs;
        _defaultCustom = custom;
        _trace = trace;
        Options = options;
        try
        {
            // The image side first (its family registered, a vision part at all), before anything else runs.
            var (vision, ownedVision) = FineTuner.VisionFor(model, evaluation is null ? train : train.Concat(evaluation), options.Vision);
            _ownedVision = ownedVision;
            options = options with { Vision = vision };
            Options = options;
            _images = train.Any(s => s.Images.Count > 0);
            if (vision is not null)
            {
                trace?.Invoke($"images: {vision} ({train.Count(s => s.Images.Count > 0)} of {train.Count} training sequences hold images; "
                              + (vision.Parameters.Count == 0 ? "steps with images may be recorded as graphs, their features copied in)"
                                  : "steps with images are not recorded as graphs while a vision part trains: their features need gradients)"));
            }

            var network = model.Network;
            if (!network.Descendants().OfType<Linear>().Any(l => l.Adapter is not null))
            {
                model.AddAdapters(options.Rank, options.Alpha, options.Targets, options.Seed, options.Dora);
            }

            if (options.ResumeFrom is { } from)
            {
                if (vision is { Parameters.Count: > 0 } || options.Parts.OfType<ITrainableSoftParts>().Any())
                {
                    throw new NotSupportedException("A tune resumed from a stop's checkpoint trains the adapters only: a trained vision part or part source is not restored from it.");
                }

                ResumedSteps = FineTuner.ReadStopState(from);
                model.LoadAdapter(from);                                   // into the adapters just made: the same tensors, the stop's values
            }

            var parameters = network.TrainableParameters().Concat(vision?.Parameters ?? []).Concat(FineTuner.TrainedParts(options.Parts)).ToList();
            _float8 = FineTuner.PrepareFloat8(model, train, options, trace);
            Optimizer = FineTuner.CreateOptimizer(parameters, options);
            // Data parallel (plan 23): every rank makes the same batches from the same seed; a step covers the accumulation's
            // batches on every rank, and each rank runs its share of them.
            var clusterOptions = options.Cluster is null ? null : options.ClusterOptions ?? ClusterOptions.FromEnvironment();
            if (clusterOptions is { Pipelined: true })
            {
                throw new NotSupportedException("Pipeline placement (IDRAK_CLUSTER_PLACEMENT=pipeline) trains through the Pretrainer (plan 23, C7); fine-tuning runs with single placement.");
            }

            Sync = options.Cluster is { } cluster ? GradientSyncs.Create(cluster, Optimizer, clusterOptions!) : null;
            _seeds = Sync is null ? null : Dropout.SeedsForRank(Sync.Rank);
            if (options.ResumeFrom is { } resumed)
            {
                var saved = FineTuner.ReadOptimizerFile(resumed);
                if (Sync?.LoadOptimizerState(saved) != true)
                {
                    Optimizer.LoadState(saved);
                }
            }

            Sync?.Start();
            _random = new Random(options.Seed);
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>The model being tuned.</summary>
    public PretrainedModel Model { get; }

    /// <summary>The settings in effect (with the image side the session uses, when the sequences hold images).</summary>
    public FineTuningOptions Options { get; }

    /// <summary>The optimizer steps done (a resumed session counts the checkpoint's).</summary>
    public int Steps { get; private set; }

    /// <summary>The optimizer steps the schedule spans.</summary>
    public int TotalSteps { get; private set; }

    /// <summary>The steps the checkpoint it resumed from had done (<see cref="FineTuningOptions.ResumeFrom"/>); 0 for a new session.</summary>
    public long ResumedSteps { get; }

    /// <summary>The learning rate the next step uses.</summary>
    public float LearningRate => Optimizer.LearningRate;

    /// <summary>The memory settings in effect and the recorded step graphs, as a log line says them.</summary>
    public string Settings => Runner.Settings;

    // The parts a run (FineTuner.Train) drives directly.
    internal Optimizer Optimizer { get; } = null!;

    internal IGradientSync? Sync { get; }

    internal FineTuner.StepRunner Runner => _runner ?? throw new InvalidOperationException("The session's schedule is not made yet (Begin).");

    internal bool Pairs { get; }

    // The schedule over the run's steps and the runner of its steps (FineTuner.Train's order: after the run's batches,
    // whose sizes the runner is told: the most trained tokens of a batch, the most positions, the most image tokens).
    internal void Begin(int totalSteps, IReadOnlyList<TrainingSequence> train, int lossRows, long largest, int imageTokens)
    {
        TotalSteps = totalSteps;
        var scheduler = Options.Scheduler
            ?? LearningRateSchedules.Factory(LearningRateSchedules.Cosine, new LearningRateScheduleSettings(Options.WarmupFraction, Options.MinLearningRate));
        _schedule = scheduler(Optimizer, Math.Max(1, totalSteps))
            ?? throw new InvalidOperationException("FineTuningOptions.Scheduler returned no schedule.");
        _runner = new FineTuner.StepRunner(Model, train, Optimizer, Options, lossRows, _defaultCustom, _trace, largest, imageTokens, Sync);
    }

    // One optimizer step over `group` (batches of the run's sequences, or of `sequences` when given): the step, the
    // check of a loss that is not finite, the schedule, the step count, the cluster told.
    internal (float Loss, long Tokens, float Rate) Run(IReadOnlyList<FineTuner.Batch> group, IReadOnlyList<TrainingSequence>? sequences, CancellationToken cancellationToken,
        Func<int, string> label)
    {
        var runner = Runner;
        var (loss, tokens) = sequences is null ? runner.Run(group, cancellationToken, label) : runner.Run(group, sequences, cancellationToken, label);
        float rate = Optimizer.LearningRate;                               // the rate the step's update used
        if (float.IsFinite(loss))
        {
            _nonFinite = 0;
        }
        else if (++_nonFinite > Options.NonFiniteSteps)
        {
            throw new InvalidOperationException($"The loss of step {Steps + 1} is {loss}, the {_nonFinite} step{(_nonFinite == 1 ? "" : "s")} in a row not finite "
                + $"({runner.Settings}); the run stops (FineTuningOptions.NonFiniteSteps = {Options.NonFiniteSteps}). The parameters keep the values of "
                + $"the last finite step.");
        }
        else
        {
            _trace?.Invoke($"step {Steps + 1}: the loss is {loss}, not finite ({runner.Settings}); the step's update is skipped and the parameters keep "
                           + $"their values ({_nonFinite} in a row, the run stops after {Options.NonFiniteSteps})");
        }

        _schedule!.Step();
        Steps++;
        Sync?.Stepped(Steps);
        return (loss, tokens, rate);
    }

    // A step this rank does not train (a resumed run's finished step, a cluster step it sits out): the schedule moves on.
    internal void Skip()
    {
        _schedule!.Step();
        Steps++;
    }

    /// <summary>
    /// One optimizer step over <paramref name="sequences"/> (all their batches accumulated into one update): packed or
    /// padded as <see cref="FineTuningOptions.Packing"/> says, gradients clipped to <see cref="FineTuningOptions.MaxGradientNorm"/>,
    /// then the optimizer's update and the schedule's next rate. A step whose loss is not finite leaves the adapters as
    /// they were (<see cref="FineTuningOptions.NonFiniteSteps"/> in a row stop the session with an error).
    /// </summary>
    /// <param name="sequences">What the step trains on: each sequence's trained tokens (and, for a loss that reads them, its <see cref="TrainingSequence.Value"/>, behaviour and reference log-probabilities).</param>
    /// <param name="loss">
    /// The loss of one's own (a policy loss through <see cref="FineTuningLosses.Policy"/>, <see cref="FineTuningLosses.OnPolicyDistillation"/>);
    /// null for <see cref="FineTuningOptions.Loss"/> (the fused token cross-entropy when that is null, or distillation with a <see cref="FineTuningOptions.Teacher"/>).
    /// </param>
    /// <param name="cancellationToken">Stops between batches, before the update.</param>
    /// <exception cref="ArgumentException">No sequence, or none with a token to train.</exception>
    public FineTuningStepStats Step(IReadOnlyList<TrainingSequence> sequences, FineTuningLoss? loss = null, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(sequences);
        if (sequences.Count == 0)
        {
            throw new ArgumentException("A step trains on at least one sequence.", nameof(sequences));
        }

        if (Pairs)
        {
            throw new NotSupportedException("A session opened for preference pairs steps through FineTuner.Train.");
        }

        cancellationToken.ThrowIfCancellationRequested();
        ReleaseGenerators();                                               // no reply's cache held through the step
        if (_float8Pending)
        {
            _float8Pending = false;
            _float8 = FineTuner.PrepareFloat8(Model, sequences, Options, _trace);
        }

        var runner = Runner;
        runner.Custom = CustomFor(loss, sequences);
        var batches = FineTuner.MakeBatches(Model, sequences, Options, _random);
        if (batches.Count == 0)
        {
            throw new ArgumentException("No sequence has a token to train (each needs two tokens or more).", nameof(sequences));
        }

        var watch = Stopwatch.StartNew();
        if (Sync is { } sync && !sync.BeginStep(Steps))
        {
            Skip();                                                        // this rank sits the step out; the schedule moves on alike
            return new FineTuningStepStats(Steps, 0f, Optimizer.LearningRate, 0, watch.Elapsed, Skipped: true);
        }

        var group = Sync is null ? batches : FineTuner.Share(batches, sequences, Sync, runner, _trace);
        int at = Steps;
        var (value, tokens, rate) = Run(group, sequences, cancellationToken, b => $"step {at + 1}, batch {b + 1}/{group.Count}");
        return new FineTuningStepStats(Steps, value, rate, tokens, watch.Elapsed);
    }

    // The step's loss: the session's own (FineTuningOptions.Loss, distillation with a teacher, or the fused token
    // cross-entropy), or the given one, kept from step to step while it is the same delegate (its reference values for
    // the step's sequences with it; those of sequences no longer trained dropped).
    private FineTuner.CustomLoss? CustomFor(FineTuningLoss? loss, IReadOnlyList<TrainingSequence> sequences)
    {
        if (loss is null)
        {
            _defaultCustom?.Keep(sequences);
            return _defaultCustom;
        }

        if (!ReferenceEquals(loss, _lastLoss))
        {
            (_lastLoss, _lastCustom) = (loss, new FineTuner.CustomLoss(loss, pairs: false, Options.Teacher));
        }

        _lastCustom!.Keep(sequences);
        return _lastCustom;
    }

    /// <summary>
    /// log p of each trained token of each of <paramref name="sequences"/> under the model as it is now, without gradients
    /// (the network in evaluation mode for the call): with the adapters (<paramref name="adapters"/> true: the policy's
    /// behaviour values) or without them (the base model: a reference). The sequences run in the batches a step would
    /// make of them (packed as <see cref="FineTuningOptions.Packing"/> says), so the values are the ones a step's forward
    /// pass computes. Each sequence's values are in its trained tokens' order; a sequence too short to train has none.
    /// </summary>
    /// <param name="sequences">The sequences.</param>
    /// <param name="adapters">True: the adapters on; false: the base model.</param>
    public float[][] LogProbabilities(IReadOnlyList<TrainingSequence> sequences, bool adapters = true)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(sequences);
        var result = new float[sequences.Count][];
        if (sequences.Count == 0)
        {
            return result;
        }

        ReleaseGenerators();
        var network = Model.Network;
        bool wasTraining = network.IsTraining;
        var batches = FineTuner.MakeBatches(Model, sequences, Options, random: null);
        try
        {
            using (Autograd.NoGrad())
            using (adapters ? null : network.DisableAdapters())
            {
                network.Eval();
                foreach (var batch in batches)
                {
                    using var scope = new TensorScope();
                    using var packing = batch.Packed ? PackedSequences.Create([.. batch.Rows.Select(r => r.Select(i => sequences[i].Tokens.Length - 1).ToArray())], batch.Length, Model.Device) : null;
                    using var packed = packing?.Use();
                    var values = FineTuner.TrainedLogProbabilities(Model, sequences, batch, Options);
                    int at = 0;
                    foreach (int i in batch.Sequences)
                    {
                        int count = sequences[i].TrainedTokens;
                        result[i] = values[at..(at + count)];
                        at += count;
                    }
                }
            }
        }
        finally
        {
            network.Train(wasTraining);
        }

        for (int i = 0; i < result.Length; i++)
        {
            result[i] ??= [];                                              // too short to be in a batch: nothing trained
        }

        return result;
    }

    /// <summary>
    /// A chat model on the session's live adapters (the model's own chat template): what it generates follows every step
    /// taken. It keeps no key/value cache between replies (each reply's is freed when it ends), and the session releases
    /// whatever its generators hold before each step, so generation and training never hold the device together.
    /// </summary>
    /// <param name="contextLength">Its context window (the model's, by default).</param>
    /// <param name="cacheFormat">How its key/value cache stores keys and values while a reply is generated.</param>
    /// <exception cref="InvalidOperationException">The model has no chat template.</exception>
    public ChatGenerator Generator(int? contextLength = null, KeyValueFormat cacheFormat = KeyValueFormat.Float32)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var template = Model.ChatTemplate ?? throw new InvalidOperationException("The model has no chat template: a session's generator chats with the model's own.");
        var generator = Model.CreateGenerator(cacheFormat, contextLength);
        generator.KeepCache = false;
        _generators.Add(generator);
        return new ChatGenerator(generator, template);
    }

    private void ReleaseGenerators()
    {
        foreach (var generator in _generators)
        {
            generator.ReleaseCache();
        }
    }

    /// <summary>
    /// Writes what the session trained to <paramref name="folder"/>: the adapters (PEFT format, see
    /// <see cref="PretrainedModel.SaveAdapter"/>), with trained vision parts and part sources when there are any.
    /// </summary>
    public void Save(string folder)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentException.ThrowIfNullOrWhiteSpace(folder);
        foreach (var part in Options.Parts.OfType<ITrainableSoftParts>().Where(p => p.Trained.Count > 0))
        {
            Model.KeepTrainedParts(part.Export(part.Trained));                 // saved with the adapters (modules_to_save)
        }

        if (Options.Vision is { } vision)
        {
            vision.Save(folder, _images);                                      // the trained vision parts and tuning_images.json too
        }
        else
        {
            Model.SaveAdapter(folder);
        }
    }

    /// <summary>
    /// Writes a checkpoint to <paramref name="folder"/>: the adapters (<see cref="Save"/>), the optimizer's state and the
    /// steps done, which a new session resumes from (<see cref="FineTuningOptions.ResumeFrom"/>) or this one goes back to
    /// (<see cref="Restore"/>). On a cluster every rank takes part (a sharded run's state is joined) and rank 0 writes.
    /// </summary>
    public void Checkpoint(string folder)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentException.ThrowIfNullOrWhiteSpace(folder);
        using (Sync?.Gather())
        {
            var state = Sync?.SaveOptimizerState() ?? Optimizer.SaveState();
            if (Sync is null || Sync.Rank == 0)
            {
                Save(folder);
                FineTuner.WriteStopState(folder, state, Optimizer, Steps);
            }
        }

        Sync?.Barrier();
    }

    /// <summary>
    /// Loads the checkpoint <see cref="Checkpoint"/> wrote at <paramref name="folder"/> into this session: the adapters'
    /// values, the optimizer's state and the steps (the schedule moved on to them).
    /// </summary>
    /// <exception cref="InvalidOperationException">The session has done more steps than the checkpoint: a schedule only moves forward (open a new session with <see cref="FineTuningOptions.ResumeFrom"/>).</exception>
    public void Restore(string folder)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentException.ThrowIfNullOrWhiteSpace(folder);
        long steps = FineTuner.ReadStopState(folder);
        if (steps < Steps)
        {
            throw new InvalidOperationException($"The checkpoint at {folder} was written after {steps} steps; this session has done {Steps}, and its schedule only moves "
                                                + "forward: open a new session with FineTuningOptions.ResumeFrom to go back to it.");
        }

        Model.LoadAdapter(folder);
        var saved = FineTuner.ReadOptimizerFile(folder);
        if (Sync?.LoadOptimizerState(saved) != true)
        {
            Optimizer.LoadState(saved);
        }

        while (Steps < steps)
        {
            Skip();
        }
    }

    /// <summary>
    /// Releases what the session holds, in the reverse order it was made: its generators' caches, the recorded step
    /// graphs, the cluster's sync, the optimizer, the FP8 copies, an image side it made. The adapters stay on the model.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        ReleaseGenerators();
        _runner?.Dispose();
        _seeds?.Dispose();
        Sync?.Dispose();
        (Optimizer as IDisposable)?.Dispose();
        _float8?.Dispose();
        _ownedVision?.Dispose();
    }
}
