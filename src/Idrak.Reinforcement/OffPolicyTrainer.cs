// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Reinforcement;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Reinforcement.Abstractions;

namespace Idrak.Reinforcement;

/// <summary>What a report of an off-policy run says (<see cref="OffPolicyTrainer.OnReport"/>, <see cref="OffPolicyResult.History"/>).</summary>
/// <param name="Step">The environment steps done (every environment's together).</param>
/// <param name="Updates">The updates (optimizer steps) made, counting those before a resume.</param>
/// <param name="Episodes">The episodes finished, counting those before a resume.</param>
/// <param name="Returns">The returns of the episodes that finished since the last report, in the order they finished (environments in order within a round).</param>
/// <param name="RewardMean">Their mean (NaN when none finished).</param>
/// <param name="Loss">The mean loss of the updates since the last report (NaN when none).</param>
/// <param name="Epsilon">ε at the last step.</param>
/// <param name="LearningRate">The optimizer's learning rate after the last update.</param>
/// <param name="EvaluationReward">The mean return of the greedy evaluation episodes, when this report evaluated.</param>
/// <param name="Elapsed">The time since the run (this call) started.</param>
/// <param name="Metrics">The updates' means since the last report: "q" (the mean Q(s, a) of the batches), "gradient-norm".</param>
public sealed record OffPolicySummary(long Step, long Updates, int Episodes, float[] Returns, double RewardMean, double Loss, float Epsilon, float LearningRate,
    double? EvaluationReward, TimeSpan Elapsed, IReadOnlyDictionary<string, double> Metrics);

/// <summary>What an off-policy run did (<see cref="OffPolicyTrainer.Train"/>).</summary>
/// <param name="History">Each report this call made, in order.</param>
/// <param name="Steps">The environment steps done, counting those before a resume.</param>
/// <param name="Updates">The updates made, counting those before a resume.</param>
/// <param name="Cancelled">The run stopped on its cancellation token, between two rounds.</param>
/// <param name="Checkpoint">The last checkpoint written, or null.</param>
public sealed record OffPolicyResult(IReadOnlyList<OffPolicySummary> History, long Steps, long Updates, bool Cancelled, string? Checkpoint);

/// <summary>
/// The off-policy trainer: deep Q-learning (DQN, Mnih et al. 2015) of a <see cref="Module"/> Q-network over float
/// observations (a box) and discrete actions. Each round steps <see cref="OffPolicyOptions.Environments"/> environments once
/// with ε-greedy actions of the online network (uniformly random ones for the first
/// <see cref="OffPolicyOptions.LearningStarts"/> steps), stores the transitions in a replay buffer
/// (<see cref="ReplayBuffer"/>), and every <see cref="OffPolicyOptions.TrainEvery"/> rounds makes
/// <see cref="OffPolicyOptions.GradientSteps"/> updates on mini-batches drawn from it: the loss of
/// <see cref="OffPolicyOptions.Algorithm"/> (<see cref="QLosses"/>: "dqn", a Huber loss against a target network, double
/// Q-learning as an option), its gradient clipped and stepped by an optimizer made through <c>OptimizerKinds</c>, the
/// learning rate following a <c>LearningRateSchedules</c> schedule over the run's updates. The target network takes the
/// online one's weights every <see cref="OffPolicyOptions.TargetEvery"/> steps (copied, or Polyak-averaged by
/// <see cref="OffPolicyOptions.Tau"/>). ε falls linearly from <see cref="OffPolicyOptions.EpsilonStart"/> to
/// <see cref="OffPolicyOptions.EpsilonEnd"/> over <see cref="OffPolicyOptions.EpsilonFraction"/> of the run.
/// <list type="bullet">
/// <item><b>Reproducible:</b> every draw derives from the run's seed and where the run is (an environment's episode, a
/// round's exploration, an update's mini-batch), environments are stepped and recorded in their order, so a seeded run gives
/// the same transitions, losses and weights whatever the threads; the run's random state is its seed and its round.</item>
/// <item><b>Checkpoints with resume:</b> the online and target networks, the optimizer's state, the replay buffer (bit for
/// bit) and the run's state: the round, the updates, and each environment's episode as its seed and the actions taken so far
/// (an environment's state is its seed and its actions: the contract's seeded replay), so <see cref="OffPolicyOptions.Resume"/>
/// replays those actions and continues bit for bit as if it had not stopped.</item>
/// <item><b>Memory:</b> the replay buffer is fixed when the trainer is made; a round's and an update's host arrays are rented
/// once per run, and each update's tensors live in a scope of their own, so memory held does not grow with steps.</item>
/// <item><b>Progress and telemetry:</b> the steps on <see cref="Progress"/> (mean return, ε, evaluation, steps a second); the
/// library's telemetry (training start and end, a batch event per update, an epoch event per report with the steps,
/// episodes, return, ε, Q and loss), built only when a hook listens (rule 78); a JSON line per report into
/// <see cref="OffPolicyOptions.Log"/>.</item>
/// </list>
/// The trainer owns the networks it is given and disposes them with itself. Not thread-safe: one caller trains it.
/// </summary>
public sealed class OffPolicyTrainer : IDisposable
{
    private const string StateFile = "state.json", NetworkFile = "network.ikw", TargetFile = "target.ikw", OptimizerFile = "optimizer.safetensors",
        ReplayFile = "replay.bin", StateFormat = "idrak-off-policy/1", CheckpointPrefix = "step-";

    // The run's streams of draws (Seeds.Derive), apart from the on-policy trainer's.
    private const int ExploreStream = 21, EpisodeStream = 22, ReplayStream = 23, EvaluationStream = 24, CreationStream = 25, EvaluationEpisodeStream = 26;
    private const int TrialRounds = 4;

    private readonly IEnvironmentKind _kind;
    private readonly Module _network, _target;
    private readonly List<(Tensor Online, Tensor Target)> _pairs = [];
    private readonly int _observationSize, _actions;
    private List<IEnvironment<float[], int>>? _evaluation;
    private int _trials;
    private long _sequentialTicks, _parallelTicks;
    private bool? _measured;
    private bool _disposed;

    /// <summary>A trainer of <paramref name="network"/> in environments of <paramref name="environmentKind"/>.</summary>
    /// <param name="environmentKind">Where the episodes are played: float observations (a box) and discrete actions.</param>
    /// <param name="network">The online Q-network: observations [batch, observation size] to action values [batch, actions]; trained.</param>
    /// <param name="target">The target network: a module of the same architecture, which takes the online one's weights now and at every target update; frozen.</param>
    /// <param name="options">The run's settings; null for <see cref="OffPolicyOptions.FromEnvironment"/>.</param>
    /// <exception cref="ArgumentException">The kind's types or spaces are not float observations and discrete actions, the networks differ in their parameters, or the network's outputs are not one per action.</exception>
    public OffPolicyTrainer(IEnvironmentKind environmentKind, Module network, Module target, OffPolicyOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(environmentKind);
        ArgumentNullException.ThrowIfNull(network);
        ArgumentNullException.ThrowIfNull(target);
        if (environmentKind.ObservationType != typeof(float[]) || environmentKind.ActionType != typeof(int))
        {
            throw new ArgumentException($"Environment kind '{environmentKind.Name}' observes {environmentKind.ObservationType.Name} and acts with {environmentKind.ActionType.Name}; "
                                        + "a Q-network observes float arrays and chooses among discrete actions (int).", nameof(environmentKind));
        }

        Options = options ?? OffPolicyOptions.FromEnvironment();
        Options.Validate();
        _kind = environmentKind;
        using (var probe = environmentKind.Create(Options.EnvironmentOptions, 0))
        {
            ObservationSpace = probe.ObservationSpace;
            ActionSpace = probe.ActionSpace;
        }

        if (ObservationSpace.Kind != SpaceKind.Box || ActionSpace.Kind != SpaceKind.Discrete)
        {
            throw new ArgumentException($"A Q-network reads a box of observations and chooses among discrete actions, not {ObservationSpace} and {ActionSpace}.", nameof(environmentKind));
        }

        _observationSize = ObservationSpace.Size;
        _actions = ActionSpace.Count;
        _network = network;
        _target = target;
        Device = network.WeightsDevice ?? Device.Default;
        var online = network.Parameters().Concat(network.Buffers()).ToList();
        var frozen = target.Parameters().Concat(target.Buffers()).ToList();
        if (online.Count != frozen.Count || online.Zip(frozen).Any(p => p.First.Size != p.Second.Size))
        {
            throw new ArgumentException($"The target network has other parameters than the online one ({frozen.Count} tensors, {online.Count} expected): give a module of the same architecture.", nameof(target));
        }

        _pairs.AddRange(online.Zip(frozen));
        using (var scope = new TensorScope())
        using (Autograd.NoGrad())
        {
            int width = network.Forward(Tensor.Zeros([1, _observationSize], Device)).Size;
            if (width != _actions)
            {
                throw new ArgumentException(string.Create(CultureInfo.InvariantCulture,
                    $"The Q-network gives {width} values an observation; the environment has {_actions} actions ({ActionSpace}): one value per action."), nameof(network));
            }
        }

        target.Freeze();
        SyncTarget(1f);
        Optimizer = OptimizerKinds.Factory(Options.Optimizer, new OptimizerSettings(Options.LearningRate, Options.WeightDecay))([.. network.Parameters()]);
        Replay = new ReplayBuffer(Options.ReplayCapacity, _observationSize);
    }

    /// <summary>The run's settings.</summary>
    public OffPolicyOptions Options { get; }

    /// <summary>The environments' observations.</summary>
    public Space ObservationSpace { get; }

    /// <summary>Their actions.</summary>
    public Space ActionSpace { get; }

    /// <summary>The online Q-network.</summary>
    public Module Network => _network;

    /// <summary>The target network.</summary>
    public Module Target => _target;

    /// <summary>The optimizer of the online network.</summary>
    public Optimizer Optimizer { get; }

    /// <summary>The replay buffer (fixed in size when the trainer is made).</summary>
    public ReplayBuffer Replay { get; }

    /// <summary>Where the networks live.</summary>
    public Device Device { get; }

    /// <summary>Where the run's progress is shown (environment steps as the unit); null for nowhere.</summary>
    public IProgressReporter? Progress { get; init; }

    /// <summary>Called after each report, on the training thread.</summary>
    public Action<OffPolicySummary>? OnReport { get; init; }

    /// <summary>The online network's value of each action at <paramref name="observation"/>.</summary>
    public float[] Values(float[] observation)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(observation);
        if (observation.Length != _observationSize)
        {
            throw new ArgumentException($"An observation holds {_observationSize} floats ({ObservationSpace}), not {observation.Length}.", nameof(observation));
        }

        float[] values = new float[_actions];
        Forward(_network, observation, 1, values);
        return values;
    }

    /// <summary>The greedy action at <paramref name="observation"/> (the first of equal values).</summary>
    public int Act(float[] observation) => DqnLoss.ArgMax(Values(observation));

    /// <summary>The mean return of <paramref name="episodes"/> greedy episodes from seeds drawn from <paramref name="seed"/>, played at once.</summary>
    /// <param name="episodes">The episodes.</param>
    /// <param name="seed">Where their seeds come from: the same seed plays the same episodes.</param>
    /// <param name="cancellationToken">Stops the episodes.</param>
    public async ValueTask<double> EvaluateAsync(int episodes, int seed, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(episodes);
        _evaluation ??= [];
        while (_evaluation.Count < episodes)
        {
            _evaluation.Add(Create(Seeds.Derive(seed, CreationStream, _evaluation.Count)));
        }

        var observations = new float[episodes][];
        var returns = new double[episodes];
        var steps = new int[episodes];
        var active = new List<int>(episodes);
        for (int k = 0; k < episodes; k++)
        {
            observations[k] = _evaluation[k].Reset(Seeds.Derive(seed, EvaluationEpisodeStream, k));
            active.Add(k);
        }

        float[] input = ArrayPool<float>.Shared.Rent(episodes * _observationSize);
        float[] values = ArrayPool<float>.Shared.Rent(episodes * _actions);
        try
        {
            while (active.Count > 0)
            {
                cancellationToken.ThrowIfCancellationRequested();
                for (int i = 0; i < active.Count; i++)
                {
                    observations[active[i]].CopyTo(input, i * _observationSize);
                }

                Forward(_network, input.AsSpan(0, active.Count * _observationSize), active.Count, values.AsSpan(0, active.Count * _actions));
                for (int i = active.Count - 1; i >= 0; i--)
                {
                    int k = active[i];
                    var step = await _evaluation[k].StepAsync(DqnLoss.ArgMax(values.AsSpan(i * _actions, _actions)), cancellationToken).ConfigureAwait(false);
                    returns[k] += step.Reward;
                    observations[k] = step.Observation;
                    if (step.Done || (Options.MaxEpisodeSteps > 0 && ++steps[k] >= Options.MaxEpisodeSteps))
                    {
                        active.RemoveAt(i);
                    }
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(input);
            ArrayPool<float>.Shared.Return(values);
        }

        double sum = 0;
        foreach (double r in returns)
        {
            sum += r;
        }

        return sum / episodes;
    }

    /// <summary>Runs the rounds (see the type), on the calling thread.</summary>
    /// <param name="cancellationToken">Stops the run between rounds.</param>
    public OffPolicyResult Train(CancellationToken cancellationToken = default) => TrainAsync(cancellationToken).AsTask().GetAwaiter().GetResult();

    /// <summary>Runs the rounds (see the type); awaits the environments that wait.</summary>
    /// <param name="cancellationToken">Stops the run between rounds.</param>
    public async ValueTask<OffPolicyResult> TrainAsync(CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var o = Options;
        var loss = QLosses.Get(o.Algorithm);
        var lossOptions = LossOptions(o, loss);
        bool needsNext = loss.NeedsNextValues(lossOptions);
        int environments = o.Environments;
        long rounds = (o.Steps + environments - 1) / environments;
        long reportRounds = Math.Max(1, o.ReportEvery / environments), targetRounds = Math.Max(1, o.TargetEvery / environments);
        long evaluateRounds = o.EvaluateEvery > 0 ? Math.Max(1, o.EvaluateEvery / environments) : 0, saveRounds = o.SaveEvery > 0 ? Math.Max(1, o.SaveEvery / environments) : 0;
        long totalUpdates = 0;
        for (long r = 0; r < rounds; r++)
        {
            totalUpdates += Trains(r, o) ? o.GradientSteps : 0;
        }

        var schedule = LearningRateSchedules.Factory(o.Schedule, o.ScheduleSettings)(Optimizer, (int)Math.Clamp(totalUpdates, 1, int.MaxValue));
        int seed = o.Seed;
        int evaluationSeed = o.EvaluationSeed ?? Seeds.Derive(seed, EvaluationStream);

        // The environments, each with its episode; resumed from a checkpoint, or started from the seed.
        var slots = new EnvironmentSlot[environments];
        for (int k = 0; k < environments; k++)
        {
            slots[k] = new EnvironmentSlot(Create(Seeds.Derive(seed, CreationStream, k)));
        }

        long first = 0, updates = 0;
        int episodes = 0;
        try
        {
            if (o.Resume is { } resume)
            {
                (first, updates, episodes) = await ReadCheckpointAsync(resume, seed, slots, cancellationToken).ConfigureAwait(false);
                for (long u = 0; u < updates; u++)
                {
                    schedule.Step();
                }
            }
            else
            {
                for (int k = 0; k < environments; k++)
                {
                    slots[k].Start(0, Seeds.Derive(seed, EpisodeStream, k, 0));
                }
            }

            return await RunAsync(o, loss, lossOptions, needsNext, schedule, slots, first, updates, episodes, rounds, reportRounds, targetRounds, evaluateRounds, saveRounds,
                evaluationSeed, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            foreach (var slot in slots)
            {
                slot.Environment.Dispose();
            }
        }
    }

    /// <summary>Disposes the replay buffer, the optimizer, the evaluation environments and the networks; more than once is harmless.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        Replay.Dispose();
        Optimizer.Dispose();
        foreach (var environment in _evaluation ?? [])
        {
            environment.Dispose();
        }

        _network.Dispose();
        _target.Dispose();
    }

    private async ValueTask<OffPolicyResult> RunAsync(OffPolicyOptions o, IQLoss loss, Dictionary<string, string> lossOptions, bool needsNext, LearningRateScheduler schedule,
        EnvironmentSlot[] slots, long first, long updates, int episodes, long rounds, long reportRounds, long targetRounds, long evaluateRounds, long saveRounds, int evaluationSeed,
        CancellationToken cancellationToken)
    {
        int environments = slots.Length, batch = o.BatchSize, seed = o.Seed;
        var history = new List<OffPolicySummary>();
        var windowReturns = new List<float>();
        double windowLoss = 0, windowQ = 0, windowNorm = 0;
        int windowUpdates = 0;
        string? checkpoint = null;
        bool cancelled = false;
        float epsilon = Epsilon(first * environments, o);
        var started = Stopwatch.StartNew();
        var progress = Progress;
        OffPolicySummary? last = null;
        if (progress is not null)
        {
            progress.Total = rounds * environments;
            progress.Report(first * environments);
            progress.Detail = () => last is null ? "" : string.Create(CultureInfo.InvariantCulture,
                $"return {last.RewardMean:0.###}, ε {last.Epsilon:0.###}{(last.EvaluationReward is { } e ? $", evaluation {e:0.###}" : "")}, "
                + $"{(last.Step - first * environments) / Math.Max(1e-9, last.Elapsed.TotalSeconds):0} steps/s");
        }

        if (Telemetry.IsEnabled(TelemetryLevel.Training))
        {
            Telemetry.TrainingStarted(new TrainingStarted($"{o.Algorithm} in {_kind.Name}", Optimizer.GetType().Name, Device, (int)Math.Min(int.MaxValue, (rounds + reportRounds - 1) / reportRounds),
                (int)Math.Min(int.MaxValue, reportRounds * environments), o.EvaluateEvery > 0 ? o.EvaluationEpisodes : null, batch, 0,
                Optimizer.Parameters.Sum(p => (long)p.Size), Optimizer.LearningRate, ComputeResources.MaxCpuThreads));
        }

        float[] input = ArrayPool<float>.Shared.Rent(environments * _observationSize);
        float[] values = ArrayPool<float>.Shared.Rent(environments * _actions);
        int[] chosen = ArrayPool<int>.Shared.Rent(environments);
        var steps = new EnvironmentStep<float[]>[environments];
        using var sample = new Sample(batch, _observationSize, _actions);
        long round = first;
        try
        {
            for (; round < rounds; round++)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    cancelled = true;
                    break;
                }

                // Act: ε-greedy on the online network (uniform before learning starts), one draw stream per round.
                long before = round * environments;
                epsilon = Epsilon(before, o);
                bool uniform = before < o.LearningStarts;
                if (!uniform)
                {
                    for (int k = 0; k < environments; k++)
                    {
                        slots[k].Observation.CopyTo(input, k * _observationSize);
                    }

                    Forward(_network, input.AsSpan(0, environments * _observationSize), environments, values.AsSpan(0, environments * _actions));
                }

                var explore = new Random(Seeds.Derive(seed, ExploreStream, round));
                for (int k = 0; k < environments; k++)
                {
                    double u = explore.NextDouble();
                    int random = explore.Next(_actions);                                    // drawn whether used or not: a fixed stream
                    chosen[k] = uniform || u < epsilon ? random : DqnLoss.ArgMax(values.AsSpan(k * _actions, _actions));
                }

                // Step, then record in environment order.
                try
                {
                    await StepAsync(slots, chosen, steps, o.ParallelSteps, cancellationToken).ConfigureAwait(false);
                }
                catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
                {
                    cancelled = true;
                    break;
                }

                for (int k = 0; k < environments; k++)
                {
                    var slot = slots[k];
                    var step = steps[k];
                    Replay.Add(slot.Observation, chosen[k], step.Reward, step.Observation, step.Terminated);
                    slot.Actions.Add(chosen[k]);
                    slot.Return += step.Reward;
                    bool cut = !step.Done && o.MaxEpisodeSteps > 0 && slot.Actions.Count >= o.MaxEpisodeSteps;
                    if (step.Done || cut)
                    {
                        windowReturns.Add((float)slot.Return);
                        episodes++;
                        int next = slot.Episode + 1;
                        slot.Start(next, Seeds.Derive(seed, EpisodeStream, k, next));
                    }
                    else
                    {
                        slot.Observation = step.Observation;
                    }
                }

                // Train: updates on mini-batches drawn from the buffer.
                if (Trains(round, o))
                {
                    for (int g = 0; g < o.GradientSteps; g++)
                    {
                        long start = Stopwatch.GetTimestamp();
                        float rate = Optimizer.LearningRate;
                        var (value, q, norm) = Update(loss, lossOptions, needsNext, sample, Seeds.Derive(seed, ReplayStream, updates), o);
                        updates++;
                        schedule.Step();
                        windowLoss += value;
                        windowQ += q;
                        windowNorm += norm;
                        windowUpdates++;
                        if (Telemetry.IsEnabled(TelemetryLevel.Batches))
                        {
                            Telemetry.BatchCompleted(new BatchCompleted((int)Math.Min(int.MaxValue, round / reportRounds + 1), windowUpdates, 0, updates, batch, value, rate, norm,
                                TimeSpan.Zero, Stopwatch.GetElapsedTime(start)));
                        }
                    }
                }

                if ((round + 1) % targetRounds == 0)
                {
                    SyncTarget(o.Tau);
                }

                // Report (and evaluate, and save) when one is due.
                long done = round + 1;
                bool evaluate = evaluateRounds > 0 && done % evaluateRounds == 0, save = saveRounds > 0 && done % saveRounds == 0 && done < rounds;
                progress?.Report(done * environments);
                if (done % reportRounds == 0 || done == rounds || evaluate || save)
                {
                    double? evaluation = null;
                    if (evaluate)
                    {
                        try
                        {
                            evaluation = await EvaluateAsync(o.EvaluationEpisodes, evaluationSeed, cancellationToken).ConfigureAwait(false);
                        }
                        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
                        {
                            evaluation = null;
                        }
                    }

                    var metrics = new Dictionary<string, double>(StringComparer.Ordinal)
                    {
                        ["q"] = windowUpdates > 0 ? windowQ / windowUpdates : double.NaN,
                        ["gradient-norm"] = windowUpdates > 0 ? windowNorm / windowUpdates : double.NaN,
                    };
                    float[] returns = [.. windowReturns];
                    double sum = 0;
                    foreach (float r in returns)
                    {
                        sum += r;
                    }

                    var summary = new OffPolicySummary(done * environments, updates, episodes, returns, returns.Length > 0 ? sum / returns.Length : double.NaN,
                        windowUpdates > 0 ? windowLoss / windowUpdates : double.NaN, epsilon, Optimizer.LearningRate, evaluation, started.Elapsed, metrics);
                    windowReturns.Clear();
                    windowLoss = windowQ = windowNorm = 0;
                    windowUpdates = 0;
                    history.Add(summary);
                    last = summary;
                    Publish(summary, o, reportRounds, rounds);
                    OnReport?.Invoke(summary);
                    if (save && o.CheckpointFolder is { } folder)
                    {
                        checkpoint = WriteCheckpoint(folder, done, updates, episodes, slots);
                    }
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(input);
            ArrayPool<float>.Shared.Return(values);
            ArrayPool<int>.Shared.Return(chosen);
        }

        if (o.CheckpointFolder is { } end && round > first)
        {
            checkpoint = WriteCheckpoint(end, round, updates, episodes, slots);
        }

        if (Telemetry.IsEnabled(TelemetryLevel.Training))
        {
            Telemetry.TrainingCompleted(new TrainingCompleted(history.Count, started.Elapsed, last?.Loss ?? double.NaN, 0, last?.RewardMean ?? double.NaN, false, cancelled));
        }

        progress?.Finish();
        return new OffPolicyResult(history, round * environments, updates, cancelled, checkpoint);
    }

    // Whether the round trains: once learning has started, every TrainEvery rounds.
    private static bool Trains(long round, OffPolicyOptions o) => (round + 1) * o.Environments > o.LearningStarts && (round + 1) % o.TrainEvery == 0;

    // ε after `steps` steps: linear from its start to its end over the fraction of the run's steps, then the end.
    private static float Epsilon(long steps, OffPolicyOptions o)
    {
        double progress = (double)steps / o.Steps;
        if (o.EpsilonFraction <= 0 || progress >= o.EpsilonFraction)
        {
            return o.EpsilonEnd;
        }

        return (float)(o.EpsilonStart + progress * (o.EpsilonEnd - o.EpsilonStart) / o.EpsilonFraction);
    }

    // The run's settings the loss takes, then the extra options as they are.
    private static Dictionary<string, string> LossOptions(OffPolicyOptions o, IQLoss loss)
    {
        var options = new Dictionary<string, string>(StringComparer.Ordinal);
        if (o.DoubleQ is { } doubleQ && loss.Keys.Contains(QLosses.DoubleKey))
        {
            options[QLosses.DoubleKey] = doubleQ ? "true" : "false";
        }

        if (o.Huber is { } huber && loss.Keys.Contains(QLosses.HuberKey))
        {
            options[QLosses.HuberKey] = huber.ToString("R", CultureInfo.InvariantCulture);
        }

        foreach (var (key, value) in o.LossOptions ?? new Dictionary<string, string>())
        {
            options[key] = value;
        }

        return options;
    }

    // One update: a mini-batch drawn from the buffer, the targets' parts without gradients, the loss recorded on the
    // online network's values, its gradient clipped and stepped. The loss, the batch's mean Q(s, a) and the gradient norm.
    private (double Loss, double Q, double Norm) Update(IQLoss loss, Dictionary<string, string> lossOptions, bool needsNext, Sample s, int drawSeed, OffPolicyOptions o)
    {
        int batch = s.Batch;
        Replay.Sample(new Random(drawSeed), s.Indices.AsSpan(0, batch));
        Replay.Gather(s.Indices.AsSpan(0, batch), s.Observations, s.ActionsTaken, s.Rewards, s.Next, s.Terminated);
        float gamma = o.Gamma;
        for (int i = 0; i < batch; i++)
        {
            s.Discounts[i] = s.Terminated[i] ? 0f : gamma;
        }

        using var scope = new TensorScope();
        int rows = batch * _observationSize;
        using (Autograd.NoGrad())
        {
            var next = Tensor.From(s.Next.AsSpan(0, rows), [batch, _observationSize], Device);
            _target.Forward(next).CopyTo(s.NextTarget.AsSpan(0, batch * _actions));
            if (needsNext)
            {
                _network.Forward(next).CopyTo(s.NextOnline.AsSpan(0, batch * _actions));
            }
        }

        var values = _network.Forward(Tensor.From(s.Observations.AsSpan(0, rows), [batch, _observationSize], Device));
        var input = new QLossInput(values, s.ActionsTaken.AsMemory(0, batch), s.Rewards.AsMemory(0, batch), s.Discounts.AsMemory(0, batch),
            s.NextTarget.AsMemory(0, batch * _actions), needsNext ? s.NextOnline.AsMemory(0, batch * _actions) : default);
        var total = loss.Loss(input, lossOptions);
        if (total.Size != 1)
        {
            throw new InvalidOperationException($"The Q-learning loss {loss.Name} gave {total.Size} values: a loss is one element.");
        }

        Optimizer.ZeroGrad();
        total.Backward();
        double norm = o.MaxGradientNorm > 0 ? Optimizer.ClipGradientNorm(o.MaxGradientNorm) : Optimizer.GradientNorm();
        Optimizer.Step();
        float lossValue = total.Item();

        // The batch's mean Q(s, a), from the values recorded (in transition order).
        values.CopyTo(s.NextOnline.AsSpan(0, batch * _actions));
        double q = 0;
        for (int i = 0; i < batch; i++)
        {
            q += s.NextOnline[i * _actions + s.ActionsTaken[i]];
        }

        return (lossValue, q / batch, norm);
    }

    // The target network moved to the online one: copied (τ = 1), or θ' ← τ θ + (1 − τ) θ'.
    private void SyncTarget(float tau)
    {
        foreach (var (online, target) in _pairs)
        {
            if (tau >= 1f)
            {
                target.CopyFrom(online);
            }
            else
            {
                target.Scale(1f - tau);
                target.AddScaled(online, tau);
            }
        }
    }

    // `network`'s values of `rows` observations into `destination`, without gradients.
    private void Forward(Module network, ReadOnlySpan<float> observations, int rows, Span<float> destination)
    {
        using var scope = new TensorScope();
        using (Autograd.NoGrad())
        {
            network.Forward(Tensor.From(observations, [rows, _observationSize], Device)).CopyTo(destination);
        }
    }

    private IEnvironment<float[], int> Create(int seed)
    {
        var made = _kind.Create(Options.EnvironmentOptions, seed);
        if (made is IEnvironment<float[], int> typed)
        {
            return typed;
        }

        made.Dispose();
        throw new InvalidOperationException($"Environment kind '{_kind.Name}' made a {made.GetType().Name}, not an IEnvironment<Single[], Int32>.");
    }

    // Steps every environment with its action into `steps`, in order: one after another, or in parallel (as asked, or as
    // measured on the first rounds: a warm-up round, then sequential and parallel in turn).
    private async ValueTask StepAsync(EnvironmentSlot[] slots, int[] actions, EnvironmentStep<float[]>[] steps, bool? parallelSteps, CancellationToken cancellationToken)
    {
        int count = slots.Length;
        bool? parallel = count < 2 ? false : parallelSteps ?? _measured;
        bool trial = parallel is null;
        if (trial)
        {
            parallel = _trials > 0 && _trials % 2 == 0;
        }

        long start = Stopwatch.GetTimestamp();
        if (parallel == true)
        {
            var options = new ParallelOptions { MaxDegreeOfParallelism = ComputeResources.MaxCpuThreads, CancellationToken = cancellationToken };
            await Parallel.ForEachAsync(Enumerable.Range(0, count), options, async (k, token) =>
                steps[k] = await slots[k].Environment.StepAsync(actions[k], token).ConfigureAwait(false)).ConfigureAwait(false);
        }
        else
        {
            for (int k = 0; k < count; k++)
            {
                steps[k] = await slots[k].Environment.StepAsync(actions[k], cancellationToken).ConfigureAwait(false);
            }
        }

        if (trial)
        {
            long elapsed = Stopwatch.GetTimestamp() - start;
            if (_trials > 0)
            {
                if (parallel == true)
                {
                    _parallelTicks += elapsed;
                }
                else
                {
                    _sequentialTicks += elapsed;
                }
            }

            if (++_trials > TrialRounds)
            {
                _measured = _parallelTicks < _sequentialTicks;
            }
        }
    }

    // The report to telemetry (built only when a hook listens) and to the log file.
    private void Publish(OffPolicySummary s, OffPolicyOptions o, long reportRounds, long rounds)
    {
        if (Telemetry.IsEnabled(TelemetryLevel.Training))
        {
            var metrics = new Dictionary<string, double>(s.Metrics, StringComparer.Ordinal)
            {
                ["steps"] = s.Step,
                ["episodes"] = s.Episodes,
                ["reward-mean"] = s.RewardMean,
                ["epsilon"] = s.Epsilon,
                ["updates"] = s.Updates,
            };
            var evaluation = s.EvaluationReward is { } e ? new Dictionary<string, double>(StringComparer.Ordinal) { ["reward-mean"] = e } : null;
            int report = (int)Math.Min(int.MaxValue, s.Step / o.Environments / reportRounds);
            Telemetry.EpochCompleted(new EpochCompleted(report, (int)Math.Min(int.MaxValue, (rounds + reportRounds - 1) / reportRounds), s.Loss, metrics, null, evaluation,
                s.LearningRate, s.Elapsed, s.Step / Math.Max(s.Elapsed.TotalSeconds, 1e-9), ComputeResources.GetMemoryUsage(Device), false));
        }

        if (o.Log is { } log)
        {
            using var stream = new FileStream(log, FileMode.Append, FileAccess.Write, FileShare.Read);
            using (var writer = new Utf8JsonWriter(stream))
            {
                writer.WriteStartObject();
                writer.WriteNumber("step", s.Step);
                writer.WriteNumber("updates", s.Updates);
                writer.WriteNumber("episodes", s.Episodes);
                writer.WriteNumber("window_episodes", s.Returns.Length);
                writer.WriteNumber("reward_mean", double.IsFinite(s.RewardMean) ? s.RewardMean : 0);
                if (s.EvaluationReward is { } evaluation)
                {
                    writer.WriteNumber("evaluation_reward", evaluation);
                }

                writer.WriteNumber("loss", double.IsFinite(s.Loss) ? s.Loss : 0);
                writer.WriteNumber("epsilon", s.Epsilon);
                writer.WriteNumber("learning_rate", s.LearningRate);
                foreach (var (name, value) in s.Metrics)
                {
                    writer.WriteNumber(name.Replace('-', '_'), double.IsFinite(value) ? value : 0);
                }

                writer.WriteNumber("seconds", s.Elapsed.TotalSeconds);
                writer.WriteEndObject();
            }

            stream.WriteByte((byte)'\n');
        }
    }

    // The checkpoint after `rounds`: the networks, the optimizer, the buffer and the run's state, written into a folder of
    // its own and moved in place whole, so a stop part-way leaves the previous checkpoint as it was.
    private string WriteCheckpoint(string root, long rounds, long updates, int episodes, EnvironmentSlot[] slots)
    {
        string folder = Path.Combine(root, CheckpointPrefix + (rounds * slots.Length).ToString("D12", CultureInfo.InvariantCulture));
        string writing = folder + ".writing";
        if (Directory.Exists(writing))
        {
            Directory.Delete(writing, recursive: true);
        }

        Directory.CreateDirectory(writing);
        _network.Save(Path.Combine(writing, NetworkFile));
        _target.Save(Path.Combine(writing, TargetFile));
        var state = Optimizer.SaveState();
        SafeTensorsWriter.Write(Path.Combine(writing, OptimizerFile), state.Select(e => (e.Key, e.Value.Shape, e.Value.Values)), SafeTensorType.F32,
            new Dictionary<string, string> { ["format"] = "idrak-optimizer", ["optimizer"] = Optimizer.GetType().Name });
        using (var stream = File.Create(Path.Combine(writing, ReplayFile)))
        {
            Replay.Save(stream);
        }

        var environments = new JsonArray();
        foreach (var slot in slots)
        {
            environments.Add((JsonNode)new JsonObject
            {
                ["episode"] = slot.Episode,
                ["seed"] = slot.Seed,
                ["return"] = slot.Return,
                ["actions"] = new JsonArray([.. slot.Actions.Select(a => (JsonNode)a)]),
            });
        }

        File.WriteAllText(Path.Combine(writing, StateFile), new JsonObject
        {
            ["format"] = StateFormat,
            ["rounds"] = rounds,
            ["steps"] = rounds * slots.Length,
            ["updates"] = updates,
            ["episodes"] = episodes,
            ["seed"] = Options.Seed,
            ["algorithm"] = Options.Algorithm,
            ["environment"] = _kind.Name,
            ["environments"] = environments,
        }.ToJsonString() + "\n");
        if (Directory.Exists(folder))
        {
            Directory.Delete(folder, recursive: true);
        }

        Directory.Move(writing, folder);
        return folder;
    }

    // A checkpoint folder (or the latest under a folder of them) into the trainer: the networks, the optimizer, the buffer,
    // and each environment's episode replayed from its seed and actions. The rounds and updates done, the episodes finished.
    private async ValueTask<(long Rounds, long Updates, int Episodes)> ReadCheckpointAsync(string path, int seed, EnvironmentSlot[] slots, CancellationToken cancellationToken)
    {
        string folder = path;
        if (!File.Exists(Path.Combine(folder, StateFile)))
        {
            folder = Directory.Exists(path)
                ? Directory.EnumerateDirectories(path, CheckpointPrefix + "*").Where(d => !d.EndsWith(".writing", StringComparison.Ordinal) && File.Exists(Path.Combine(d, StateFile)))
                    .Order(StringComparer.Ordinal).LastOrDefault() ?? folder
                : folder;
        }

        string statePath = Path.Combine(folder, StateFile);
        if (!File.Exists(statePath))
        {
            throw new FileNotFoundException($"{path} holds no off-policy checkpoint ({StateFile}, or {CheckpointPrefix}N folders holding one); give a folder OffPolicyOptions.CheckpointFolder was set to.", statePath);
        }

        var json = JsonNode.Parse(File.ReadAllText(statePath)) as JsonObject;
        if ((string?)json?["format"] != StateFormat || json["rounds"] is null || json["updates"] is null || json["environments"] is not JsonArray saved)
        {
            throw new InvalidDataException($"{statePath} is not an off-policy checkpoint's state ({StateFormat}).");
        }

        if ((int?)json["seed"] != seed || saved.Count != slots.Length)
        {
            throw new InvalidDataException(string.Create(CultureInfo.InvariantCulture,
                $"{statePath} is a run of seed {(int?)json["seed"]} with {saved.Count} environments; this one has seed {seed} and {slots.Length}: resume with the run's own settings."));
        }

        _network.Load(Path.Combine(folder, NetworkFile));
        _target.Load(Path.Combine(folder, TargetFile));
        using (var reader = SafeTensorsReader.Open(Path.Combine(folder, OptimizerFile)))
        {
            ITensorStore store = reader;
            Optimizer.LoadState(store.Names.ToDictionary(n => n, n => new OptimizerStateValue(store.ShapeOf(n), store.Read(n)), StringComparer.Ordinal));
        }

        using (var stream = File.OpenRead(Path.Combine(folder, ReplayFile)))
        {
            Replay.Load(stream);
        }

        for (int k = 0; k < slots.Length; k++)
        {
            var entry = saved[k]!.AsObject();
            var slot = slots[k];
            slot.Start((int)entry["episode"]!, (int)entry["seed"]!);
            foreach (var action in entry["actions"]!.AsArray())
            {
                var step = await slot.Environment.StepAsync((int)action!, cancellationToken).ConfigureAwait(false);
                if (step.Done)
                {
                    throw new InvalidDataException($"{statePath}: environment {k}'s episode ends while its actions are replayed; the checkpoint does not belong to this environment kind or its options.");
                }

                slot.Actions.Add((int)action!);
                slot.Observation = step.Observation;
            }

            slot.Return = (double)entry["return"]!;
        }

        return ((long)json["rounds"]!, (long)json["updates"]!, (int?)json["episodes"] ?? 0);
    }

    // One environment of the run and its episode under way: its number, seed, observation, actions and return so far.
    private sealed class EnvironmentSlot(IEnvironment<float[], int> environment)
    {
        public IEnvironment<float[], int> Environment { get; } = environment;

        public int Episode { get; private set; }

        public int Seed { get; private set; }

        public float[] Observation { get; set; } = [];

        public List<int> Actions { get; } = [];

        public double Return { get; set; }

        public void Start(int episode, int seed)
        {
            Episode = episode;
            Seed = seed;
            Actions.Clear();
            Return = 0;
            Observation = Environment.Reset(seed);
        }
    }

    // An update's host arrays, rented once per run.
    private sealed class Sample : IDisposable
    {
        public Sample(int batch, int observationSize, int actions)
        {
            Batch = batch;
            Indices = ArrayPool<int>.Shared.Rent(batch);
            Observations = ArrayPool<float>.Shared.Rent(batch * observationSize);
            Next = ArrayPool<float>.Shared.Rent(batch * observationSize);
            ActionsTaken = ArrayPool<int>.Shared.Rent(batch);
            Rewards = ArrayPool<float>.Shared.Rent(batch);
            Discounts = ArrayPool<float>.Shared.Rent(batch);
            Terminated = ArrayPool<bool>.Shared.Rent(batch);
            NextTarget = ArrayPool<float>.Shared.Rent(batch * actions);
            NextOnline = ArrayPool<float>.Shared.Rent(batch * actions);
        }

        public int Batch { get; }

        public int[] Indices { get; }

        public float[] Observations { get; }

        public float[] Next { get; }

        public int[] ActionsTaken { get; }

        public float[] Rewards { get; }

        public float[] Discounts { get; }

        public bool[] Terminated { get; }

        public float[] NextTarget { get; }

        public float[] NextOnline { get; }

        public void Dispose()
        {
            ArrayPool<int>.Shared.Return(Indices);
            ArrayPool<float>.Shared.Return(Observations);
            ArrayPool<float>.Shared.Return(Next);
            ArrayPool<int>.Shared.Return(ActionsTaken);
            ArrayPool<float>.Shared.Return(Rewards);
            ArrayPool<float>.Shared.Return(Discounts);
            ArrayPool<bool>.Shared.Return(Terminated);
            ArrayPool<float>.Shared.Return(NextTarget);
            ArrayPool<float>.Shared.Return(NextOnline);
        }
    }
}
