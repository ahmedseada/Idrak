// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Cluster;
using Idrak.Abstraction.Reinforcement;
using Idrak.Reinforcement.Abstractions;

namespace Idrak.Reinforcement;

/// <summary>What one iteration of a run did (<see cref="ReinforcementTrainer{TObservation, TAction}.OnIteration"/>, <see cref="ReinforcementResult.History"/>).</summary>
/// <param name="Iteration">The iteration, from 1.</param>
/// <param name="Episodes">The episodes played.</param>
/// <param name="Steps">Their steps together.</param>
/// <param name="RewardMean">The mean of the episodes' returns (summed in episode order).</param>
/// <param name="RewardSpread">Their standard deviation (over the episodes).</param>
/// <param name="RewardMin">The lowest return.</param>
/// <param name="RewardMax">The highest return.</param>
/// <param name="Loss">The mean loss of the iteration's updates.</param>
/// <param name="Updates">The updates (optimizer steps) of the iteration.</param>
/// <param name="LearningRate">The learning rate after the iteration; null for a policy that keeps its own.</param>
/// <param name="EvaluationReward">The mean return of the evaluation episodes, when the iteration evaluated.</param>
/// <param name="RolloutTime">The time collecting the episodes and their advantages.</param>
/// <param name="UpdateTime">The time of the updates.</param>
/// <param name="Metrics">The policy's measures averaged over the updates (kl, clip-fraction, entropy, gradient-norm, …) and the share of episodes that terminated ("terminated").</param>
/// <param name="Returns">Each episode's return, in episode order.</param>
public sealed record IterationSummary(int Iteration, int Episodes, int Steps, double RewardMean, double RewardSpread, double RewardMin, double RewardMax,
    double Loss, int Updates, float? LearningRate, double? EvaluationReward, TimeSpan RolloutTime, TimeSpan UpdateTime, IReadOnlyDictionary<string, double> Metrics,
    float[] Returns);

/// <summary>What a run did (<see cref="ReinforcementTrainer{TObservation, TAction}.Train"/>).</summary>
/// <param name="History">Each iteration this call ran, in order (a resumed run's earlier ones are in its checkpoints' logs).</param>
/// <param name="Iterations">The iterations completed, counting those before a resume.</param>
/// <param name="Cancelled">The run stopped on its cancellation token, after the last completed iteration.</param>
/// <param name="Checkpoint">The last checkpoint written, or null.</param>
public sealed record ReinforcementResult(IReadOnlyList<IterationSummary> History, int Iterations, bool Cancelled, string? Checkpoint);

/// <summary>Runs <see cref="ReinforcementTrainer{TObservation, TAction}"/> in one call.</summary>
public static class ReinforcementTrainer
{
    /// <summary>Trains <paramref name="policy"/> in environments of <paramref name="environmentKind"/> (see <see cref="ReinforcementTrainer{TObservation, TAction}"/>).</summary>
    /// <typeparam name="TObservation">The environment's observations.</typeparam>
    /// <typeparam name="TAction">Its actions.</typeparam>
    /// <param name="environmentKind">Where the episodes are played.</param>
    /// <param name="policy">What acts and learns.</param>
    /// <param name="options">The run's settings; null for <see cref="ReinforcementOptions.FromEnvironment"/>.</param>
    /// <param name="progress">Where the run's progress is shown; null for nowhere.</param>
    /// <param name="cancellationToken">Stops the run between iterations.</param>
    public static ReinforcementResult Train<TObservation, TAction>(IEnvironmentKind environmentKind, IPolicy<TObservation, TAction> policy, ReinforcementOptions? options = null,
        IProgressReporter? progress = null, CancellationToken cancellationToken = default) =>
        new ReinforcementTrainer<TObservation, TAction>(environmentKind, policy, options) { Progress = progress }.Train(cancellationToken);

    /// <summary>Trains <paramref name="policy"/> in environments of the kind <paramref name="environmentKind"/> names in <c>EnvironmentKinds</c>.</summary>
    /// <typeparam name="TObservation">The environment's observations.</typeparam>
    /// <typeparam name="TAction">Its actions.</typeparam>
    /// <param name="environmentKind">The kind's name.</param>
    /// <param name="policy">What acts and learns.</param>
    /// <param name="options">The run's settings; null for <see cref="ReinforcementOptions.FromEnvironment"/>.</param>
    /// <param name="progress">Where the run's progress is shown; null for nowhere.</param>
    /// <param name="cancellationToken">Stops the run between iterations.</param>
    /// <exception cref="NotSupportedException">No kind of that name is registered.</exception>
    public static ReinforcementResult Train<TObservation, TAction>(string environmentKind, IPolicy<TObservation, TAction> policy, ReinforcementOptions? options = null,
        IProgressReporter? progress = null, CancellationToken cancellationToken = default) =>
        Train(EnvironmentKinds.Get(environmentKind), policy, options, progress, cancellationToken);
}

/// <summary>
/// The on-policy trainer: each iteration plays <see cref="ReinforcementOptions.EpisodesPerIteration"/> episodes with the
/// policy (<see cref="RolloutCollector{TObservation, TAction}"/>), estimates their advantages and returns
/// (<c>AdvantageEstimators</c>), then makes <see cref="ReinforcementOptions.Epochs"/> passes over them in mini-batches of
/// <see cref="ReinforcementOptions.MiniBatch"/> episodes, each an update of the policy with the algorithm's loss
/// (<c>PolicyLosses</c>). A policy that exposes its optimizer (<see cref="IScheduledPolicy"/>) gets the run's gradient
/// clip and a learning-rate schedule over the run's updates (<c>LearningRateSchedules</c>).
/// <list type="bullet">
/// <item><b>Reproducible:</b> every draw derives from the run's seed and where the run is (iteration, group, round,
/// pass): episodes of a group share their environment's seed, the policy's draws come from a seed per round, mini-batches
/// from a shuffle per pass; so the same seed gives the same episodes, advantages and weights whatever the threads, and a
/// run's random state is its seed and its iteration.</item>
/// <item><b>Evaluation:</b> every <see cref="ReinforcementOptions.EvaluateEvery"/> iterations, greedy episodes from seeds of
/// their own (the same ones each time).</item>
/// <item><b>Checkpoints:</b> the policy (<see cref="IPolicy{TObservation, TAction}.Save"/>: weights and optimizer), the
/// iteration, the updates made and the seed, every <see cref="ReinforcementOptions.SaveEvery"/> iterations and at the end
/// or a stop; <see cref="ReinforcementOptions.Resume"/> continues one bit for bit as if it had not stopped.</item>
/// <item><b>Stopping:</b> a cancelled token stops the run between iterations (an iteration under way when it is cancelled is
/// dropped, its updates not begun), with a checkpoint when a folder is set. On a cluster the token asks the ranks to stop
/// (Plan 27, R7: <see cref="IRolloutRanks{TObservation, TAction}.StopAgreement"/>): the iteration under way runs to its end
/// on every rank, and every rank stops at the iteration boundary they agree on, where rank 0 writes the checkpoint.</item>
/// <item><b>Steered</b> (Plan 27, R11: <see cref="Steering"/>, the dashboard's pause, resume and stop, or a launcher's
/// control file): every iteration boundary is a control point. A pause holds every rank there (no rank inside a rollout or
/// an update), and a resume goes on as if nothing had happened, bit for bit; a stop ends the run there as a cancelled token
/// does (in one process at once, on a cluster through the ranks' agreement, every rank at the same iteration), with the
/// checkpoint written once.</item>
/// <item><b>Progress and telemetry:</b> the iterations on <see cref="Progress"/> (mean reward, spread, episodes a second);
/// the library's telemetry (training start and end, a batch event per update, an epoch event per iteration whose metrics
/// are the episodes, steps, reward mean and spread, KL, clip share, rollout and update times), built only when a hook
/// listens; a JSON line per iteration into <see cref="ReinforcementOptions.Log"/>.</item>
/// <item><b>On a cluster</b> (Plan 27, R3b: a policy that is an <see cref="IRolloutRanks{TObservation, TAction}"/> of several
/// ranks): every rank runs a trainer with the same settings; each rollout's episodes are spread over the ranks
/// (<see cref="ReinforcementOptions.Shards"/>, <see cref="ReinforcementOptions.Spread"/>) and gathered, so every rank
/// estimates the same advantages and makes the same updates (the policy's data-parallel step); rank 0 alone writes the
/// checkpoints' files and the log, while every rank saves the policy (a collective). A run is stopped on every rank alike.</item>
/// </list>
/// </summary>
/// <typeparam name="TObservation">The environment's observations.</typeparam>
/// <typeparam name="TAction">Its actions.</typeparam>
public sealed class ReinforcementTrainer<TObservation, TAction>
{
    private const string StateFile = "state.json", PolicyFolder = "policy", StateFormat = "idrak-reinforcement/1", CheckpointPrefix = "iteration-";

    private readonly IEnvironmentKind _kind;
    private readonly IPolicy<TObservation, TAction> _policy;

    /// <summary>A trainer of <paramref name="policy"/> in environments of <paramref name="environmentKind"/>.</summary>
    /// <param name="environmentKind">Where the episodes are played; its types must be the policy's.</param>
    /// <param name="policy">What acts and learns.</param>
    /// <param name="options">The run's settings; null for <see cref="ReinforcementOptions.FromEnvironment"/>.</param>
    /// <exception cref="ArgumentException">The kind's types are not the policy's.</exception>
    public ReinforcementTrainer(IEnvironmentKind environmentKind, IPolicy<TObservation, TAction> policy, ReinforcementOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(environmentKind);
        ArgumentNullException.ThrowIfNull(policy);
        if (environmentKind.ObservationType != typeof(TObservation) || environmentKind.ActionType != typeof(TAction))
        {
            throw new ArgumentException($"Environment kind '{environmentKind.Name}' observes {environmentKind.ObservationType.Name} and acts with {environmentKind.ActionType.Name}; "
                                        + $"the policy observes {typeof(TObservation).Name} and acts with {typeof(TAction).Name}.", nameof(environmentKind));
        }

        _kind = environmentKind;
        _policy = policy;
        Options = options ?? ReinforcementOptions.FromEnvironment();
    }

    /// <summary>The run's settings.</summary>
    public ReinforcementOptions Options { get; }

    /// <summary>Where the run's progress is shown (an iteration a unit); null for nowhere.</summary>
    public IProgressReporter? Progress { get; init; }

    /// <summary>Called after each iteration (its batch returned to the pool), on the training thread.</summary>
    public Action<IterationSummary>? OnIteration { get; init; }

    /// <summary>Called with each iteration's batch once its advantages and returns are written, before the updates (for inspection; the batch is not to be kept).</summary>
    public Action<int, RolloutBatch<TObservation, TAction>>? OnBatch { get; init; }

    /// <summary>
    /// The run's control point at each iteration boundary (Plan 27, R11), or null for none: the dashboard's pause, resume and
    /// stop, or a launcher's control file, as <see cref="RunSteerings.For"/> gives them for the policy's collectives (on a
    /// cluster every rank's trainer takes its own rank's). The boundary before iteration N is its step N (0 for the first).
    /// </summary>
    public IRunSteering? Steering { get; init; }

    /// <summary>Runs the iterations (see the type), on the calling thread.</summary>
    /// <param name="cancellationToken">Stops the run between iterations.</param>
    public ReinforcementResult Train(CancellationToken cancellationToken = default) => TrainAsync(cancellationToken).AsTask().GetAwaiter().GetResult();

    /// <summary>Runs the iterations (see the type); awaits the environments that wait.</summary>
    /// <param name="cancellationToken">Stops the run between iterations.</param>
    public async ValueTask<ReinforcementResult> TrainAsync(CancellationToken cancellationToken = default)
    {
        var o = Options;
        o.Validate();
        var loss = PolicyLosses.Get(o.Algorithm);
        int groupSize = GroupSize(o);
        if (o.EpisodesPerIteration % groupSize != 0)
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture,
                $"ReinforcementOptions.EpisodesPerIteration ({o.EpisodesPerIteration}) is not a multiple of the group size ({groupSize}): each group plays the same situation, so the episodes fill whole groups."));
        }

        if (o.Estimator is { } named)
        {
            _ = AdvantageEstimators.Get(named);                                         // an unknown name fails before the first rollout
        }

        int epochs = o.Epochs > 0 ? o.Epochs : string.Equals(o.Algorithm, ReinforcementNames.Ppo, StringComparison.OrdinalIgnoreCase) ? 4 : 1;
        int miniBatch = o.MiniBatch > 0 ? Math.Min(o.MiniBatch, o.EpisodesPerIteration) : o.EpisodesPerIteration;
        int updatesPerIteration = epochs * ((o.EpisodesPerIteration + miniBatch - 1) / miniBatch);
        var scheduled = _policy as IScheduledPolicy;
        LearningRateScheduler? schedule = null;
        if (scheduled is not null)
        {
            scheduled.MaxGradientNorm = o.MaxGradientNorm;
            schedule = LearningRateSchedules.Factory(o.Schedule, o.ScheduleSettings)(scheduled.Optimizer, Math.Max(1, o.Iterations * updatesPerIteration));
        }

        // Resume: the policy and its optimizer, then the schedule replayed to where the run was.
        int seed = o.Seed, first = 0;
        long updates = 0;
        if (o.Resume is { } resume)
        {
            (first, seed, updates) = ReadCheckpoint(resume);
            for (long u = 0; u < updates; u++)
            {
                schedule?.Step();
            }
        }

        var lossOptions = LossOptions(o, loss);
        int evaluationSeed = o.EvaluationSeed ?? Seeds.Derive(seed, Seeds.Evaluation);
        var device = scheduled?.Device ?? Device.Default;
        var history = new List<IterationSummary>();
        string? checkpoint = null;
        bool cancelled = false;
        double best = double.NegativeInfinity;
        var started = Stopwatch.StartNew();
        using var collector = new RolloutCollector<TObservation, TAction>(_kind, o.EnvironmentOptions, seed, o.Environments, o.ParallelSteps, o.Shards, o.Spread);
        var starts = new EpisodeStart[o.EpisodesPerIteration];
        var order = new int[o.EpisodesPerIteration];
        var miniBatchIndices = new int[miniBatch];
        var progress = Progress;
        IterationSummary? last = null;
        if (progress is not null)
        {
            progress.Total = o.Iterations;
            progress.Report(first);
            progress.Detail = () => last is null ? "" : string.Create(CultureInfo.InvariantCulture,
                $"reward {last.RewardMean:0.###} ± {last.RewardSpread:0.###}{(last.EvaluationReward is { } e ? $", evaluation {e:0.###}" : "")}, "
                + $"{last.Episodes / Math.Max(1e-9, (last.RolloutTime + last.UpdateTime).TotalSeconds):0.#} episodes/s");
        }

        if (Telemetry.IsEnabled(TelemetryLevel.Training))
        {
            Telemetry.TrainingStarted(new TrainingStarted($"{_policy.GetType().Name} in {_kind.Name}", scheduled?.Optimizer.GetType().Name ?? "the policy's own", device,
                o.Iterations, o.EpisodesPerIteration, o.EvaluateEvery > 0 ? o.EvaluationEpisodes : null, miniBatch, updatesPerIteration,
                scheduled?.Optimizer.Parameters.Sum(p => (long)p.Size) ?? 0, scheduled?.Optimizer.LearningRate ?? 0f, ComputeResources.MaxCpuThreads));
        }

        // On a cluster a cancelled token asks the ranks to stop and they agree at an iteration boundary (Plan 27, R7): the
        // rollout, its exchange and the updates run to their end on every rank (their token is cancelled only when this rank
        // gives up waiting), so no rank leaves a collective the others are in. In one process the token stops as before.
        var agreement = (_policy as IRolloutRanks<TObservation, TAction>)?.StopAgreement;
        agreement?.Reset();
        using var asks = agreement?.Watch(cancellationToken) ?? default;
        var inner = agreement?.GiveUpToken ?? cancellationToken;
        if (Lead && o.Log is { } logFile && Path.GetDirectoryName(Path.GetFullPath(logFile)) is { } logFolder)
        {
            Directory.CreateDirectory(logFolder);                                      // the run's folder may not exist before its first line
        }

        int iteration = first;
        for (; iteration < o.Iterations; iteration++)
        {
            // The boundary: the run's control holds a paused run here and turns a stop into the run's own (Plan 27, R11), then
            // the stop is the token's in one process, the ranks' agreement's on a cluster.
            bool steered = Steering.Stops(iteration, agreement, cancellationToken);
            if (agreement is null ? steered || cancellationToken.IsCancellationRequested : agreement.Agree(iteration))
            {
                cancelled = true;
                break;
            }

            // The rollout: the episodes of this iteration, a group's sharing its environment's seed.
            long rolloutStart = Stopwatch.GetTimestamp();
            int groups = o.EpisodesPerIteration / groupSize;
            for (int k = 0; k < starts.Length; k++)
            {
                int group = k / groupSize;
                starts[k] = new EpisodeStart(iteration * o.EpisodesPerIteration + k, Seeds.Derive(seed, Seeds.Episodes, iteration, group), iteration * groups + group);
            }

            RolloutBatch<TObservation, TAction> batch;
            try
            {
                batch = await collector.CollectAsync(_policy, starts, new ActionSampling { Temperature = o.Temperature, Seed = Seeds.Derive(seed, Seeds.Sampling, iteration) },
                    o.MaxEpisodeSteps, inner).ConfigureAwait(false);
            }
            catch (OperationCanceledException e) when (agreement.GaveUpError(e) is { } gaveUp)
            {
                throw gaveUp;                                                           // this rank left its rollout on its own, loudly
            }
            catch (OperationCanceledException) when (agreement is null && cancellationToken.IsCancellationRequested)
            {
                cancelled = true;
                break;
            }

            IterationSummary summary;
            using (batch)
            {
                var estimator = AdvantageEstimators.Get(EstimatorName(o, batch.HasValues));
                batch.Estimate(estimator, EstimatorOptions(o, estimator));
                OnBatch?.Invoke(iteration, batch);
                var rolloutTime = Stopwatch.GetElapsedTime(rolloutStart);

                // The updates: passes over the episodes, each shuffled from the run's seed, in mini-batches.
                long updateStart = Stopwatch.GetTimestamp();
                double lossSum = 0;
                var metricSums = new Dictionary<string, double>(StringComparer.Ordinal);
                var metricNames = new List<string>();
                int made = 0;
                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    Shuffle(order, Seeds.Derive(seed, Seeds.Shuffle, iteration, epoch), shuffle: miniBatch < o.EpisodesPerIteration);
                    for (int at = 0; at < order.Length; at += miniBatch)
                    {
                        int size = Math.Min(miniBatch, order.Length - at);
                        IReadOnlyList<int>? chosen = null;
                        if (size < order.Length)
                        {
                            order.AsSpan(at, size).CopyTo(miniBatchIndices);
                            chosen = new ArraySegment<int>(miniBatchIndices, 0, size);
                        }

                        long step = Stopwatch.GetTimestamp();
                        var stats = _policy.Update(batch, loss, chosen, lossOptions);
                        lossSum += stats.Loss;
                        foreach (var (name, value) in stats.Metrics ?? new Dictionary<string, double>())
                        {
                            if (!metricSums.TryGetValue(name, out double sum))
                            {
                                metricNames.Add(name);
                            }

                            metricSums[name] = sum + value;
                        }

                        made++;
                        updates++;
                        float rate = scheduled?.Optimizer.LearningRate ?? 0f;
                        schedule?.Step();
                        if (Telemetry.IsEnabled(TelemetryLevel.Batches))
                        {
                            Telemetry.BatchCompleted(new BatchCompleted(iteration + 1, made, updatesPerIteration, updates, stats.Episodes, stats.Loss, rate,
                                stats.Metrics is { } measured && measured.TryGetValue("gradient-norm", out double norm) ? norm : null, TimeSpan.Zero, Stopwatch.GetElapsedTime(step)));
                        }
                    }
                }

                var updateTime = Stopwatch.GetElapsedTime(updateStart);
                var metrics = new Dictionary<string, double>(StringComparer.Ordinal);
                foreach (string name in metricNames)
                {
                    metrics[name] = metricSums[name] / made;
                }

                summary = Summarize(iteration, batch, lossSum / made, made, scheduled?.Optimizer.LearningRate, rolloutTime, updateTime, metrics, o.PassReturn);
            }

            // Evaluation: greedy episodes from their own seeds, the same ones every time.
            if (o.EvaluateEvery > 0 && (iteration + 1) % o.EvaluateEvery == 0)
            {
                double? evaluation;
                try
                {
                    evaluation = await EvaluateAsync(collector, evaluationSeed, iteration, o, inner).ConfigureAwait(false);
                }
                catch (OperationCanceledException e) when (agreement.GaveUpError(e) is { } gaveUp)
                {
                    throw gaveUp;
                }
                catch (OperationCanceledException) when (agreement is null && cancellationToken.IsCancellationRequested)
                {
                    evaluation = null;
                }

                summary = summary with { EvaluationReward = evaluation };
            }

            history.Add(summary);
            last = summary;
            bool isBest = summary.RewardMean > best;
            best = Math.Max(best, summary.RewardMean);
            Publish(summary, o, device, isBest, Lead);
            progress?.Report(iteration + 1);
            OnIteration?.Invoke(summary);
            if (o.CheckpointFolder is { } folder && o.SaveEvery > 0 && (iteration + 1) % o.SaveEvery == 0 && iteration + 1 < o.Iterations)
            {
                checkpoint = WriteCheckpoint(folder, iteration + 1, seed, updates);
            }
        }

        if (o.CheckpointFolder is { } end && iteration > first)
        {
            checkpoint = WriteCheckpoint(end, iteration, seed, updates);
        }

        if (Telemetry.IsEnabled(TelemetryLevel.Training))
        {
            Telemetry.TrainingCompleted(new TrainingCompleted(iteration - first, started.Elapsed, last?.Loss ?? double.NaN, 0, last?.RewardMean ?? double.NaN, false, cancelled));
        }

        progress?.Finish();
        return new ReinforcementResult(history, iteration, cancelled, checkpoint);
    }

    // The group size: the option's, or for the group-relative estimators the largest of 8, 4, 2 that divides the episodes.
    private static int GroupSize(ReinforcementOptions o)
    {
        if (o.GroupSize > 0)
        {
            return o.GroupSize;
        }

        string estimator = o.Estimator ?? (string.Equals(o.Algorithm, ReinforcementNames.Grpo, StringComparison.OrdinalIgnoreCase) ? ReinforcementNames.Group : "");
        if (!string.Equals(estimator, ReinforcementNames.Group, StringComparison.OrdinalIgnoreCase)
            && !string.Equals(estimator, ReinforcementNames.LeaveOneOut, StringComparison.OrdinalIgnoreCase))
        {
            return 1;
        }

        foreach (int size in (ReadOnlySpan<int>)[8, 4, 2])
        {
            if (o.EpisodesPerIteration % size == 0)
            {
                return size;
            }
        }

        return 1;
    }

    // The estimator's name: the option's, else the algorithm's (gae needs the critic's values; without them, returns).
    private static string EstimatorName(ReinforcementOptions o, bool hasValues)
    {
        if (o.Estimator is { } named)
        {
            return named;
        }

        if (string.Equals(o.Algorithm, ReinforcementNames.Grpo, StringComparison.OrdinalIgnoreCase))
        {
            return ReinforcementNames.Group;
        }

        return string.Equals(o.Algorithm, ReinforcementNames.Ppo, StringComparison.OrdinalIgnoreCase) && hasValues ? ReinforcementNames.Gae : ReinforcementNames.Returns;
    }

    // The run's settings the estimator takes, then the extra options as they are.
    private static Dictionary<string, string> EstimatorOptions(ReinforcementOptions o, IAdvantageEstimator estimator)
    {
        var options = new Dictionary<string, string>(StringComparer.Ordinal);
        void Put(string key, string? value)
        {
            if (value is not null && estimator.Keys.Contains(key))
            {
                options[key] = value;
            }
        }

        Put(ReinforcementNames.GammaKey, Text(o.Gamma));
        Put(ReinforcementNames.LambdaKey, Text(o.Lambda));
        Put(ReinforcementNames.ScaleKey, o.GroupScale is { } scale ? (scale ? "true" : "false") : null);
        Put(ReinforcementNames.NormalizeAdvantagesKey, o.NormalizeAdvantages is { } normalize ? (normalize ? "true" : "false") : null);
        foreach (var (key, value) in o.EstimatorOptions ?? new Dictionary<string, string>())
        {
            options[key] = value;
        }

        return options;
    }

    // The run's settings the loss takes, then the extra options as they are.
    private static Dictionary<string, string> LossOptions(ReinforcementOptions o, IPolicyLoss loss)
    {
        var options = new Dictionary<string, string>(StringComparer.Ordinal);
        void Put(string key, string? value)
        {
            if (value is not null && loss.Keys.Contains(key))
            {
                options[key] = value;
            }
        }

        Put(ReinforcementNames.ClipKey, Text(o.Clip));
        Put(ReinforcementNames.KlKey, Text(o.KlWeight));
        Put(ReinforcementNames.KlEstimatorKey, o.KlEstimator);
        Put(ReinforcementNames.EntropyKey, Text(o.EntropyWeight));
        Put(ReinforcementNames.ValueKey, Text(o.ValueWeight));
        Put(ReinforcementNames.NormalizeKey, o.Normalization);
        foreach (var (key, value) in o.LossOptions ?? new Dictionary<string, string>())
        {
            options[key] = value;
        }

        return options;
    }

    private static string Text(float value) => value.ToString("R", CultureInfo.InvariantCulture);

    // The episode order of a pass: 0, 1, … shuffled (Fisher-Yates) from the pass's seed when the pass is split.
    private static void Shuffle(int[] order, int seed, bool shuffle)
    {
        for (int i = 0; i < order.Length; i++)
        {
            order[i] = i;
        }

        if (!shuffle)
        {
            return;
        }

        var random = new Random(seed);
        for (int i = order.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (order[i], order[j]) = (order[j], order[i]);
        }
    }

    // The greedy episodes of an evaluation, from the evaluation seed: their mean return.
    private async ValueTask<double> EvaluateAsync(RolloutCollector<TObservation, TAction> collector, int evaluationSeed, int iteration, ReinforcementOptions o, CancellationToken cancellationToken)
    {
        var starts = new EpisodeStart[o.EvaluationEpisodes];
        for (int k = 0; k < starts.Length; k++)
        {
            starts[k] = new EpisodeStart(k, Seeds.Derive(evaluationSeed, Seeds.Evaluation, k), k);
        }

        using var batch = await collector.CollectAsync(_policy, starts, new ActionSampling { Greedy = true, Seed = Seeds.Derive(evaluationSeed, Seeds.EvaluationSampling, iteration) },
            o.MaxEpisodeSteps, cancellationToken).ConfigureAwait(false);
        double sum = 0;
        foreach (var trajectory in batch.Trajectories)
        {
            sum += trajectory.Return;
        }

        return sum / batch.Episodes;
    }

    // The iteration's figures; an episode passes when its return reaches `passReturn` (none: no pass rate).
    private static IterationSummary Summarize(int iteration, RolloutBatch<TObservation, TAction> batch, double loss, int updates, float? rate, TimeSpan rolloutTime,
        TimeSpan updateTime, Dictionary<string, double> metrics, float? passReturn)
    {
        var returns = new float[batch.Episodes];
        double sum = 0, min = double.PositiveInfinity, max = double.NegativeInfinity;
        int terminated = 0, passed = 0;
        for (int e = 0; e < batch.Episodes; e++)
        {
            var trajectory = batch.Trajectories[e];
            returns[e] = trajectory.Return;
            sum += returns[e];
            min = Math.Min(min, returns[e]);
            max = Math.Max(max, returns[e]);
            terminated += trajectory.Terminated ? 1 : 0;
            passed += returns[e] >= passReturn ? 1 : 0;
        }

        double mean = sum / batch.Episodes, squares = 0;
        foreach (float r in returns)
        {
            squares += (r - mean) * (r - mean);
        }

        metrics["terminated"] = (double)terminated / batch.Episodes;
        if (passReturn is not null)
        {
            metrics["pass-rate"] = (double)passed / batch.Episodes;
        }
        return new IterationSummary(iteration + 1, batch.Episodes, batch.Steps, mean, Math.Sqrt(squares / batch.Episodes), min, max, loss, updates, rate, null,
            rolloutTime, updateTime, metrics, returns);
    }

    // The iteration to telemetry (built only when a hook listens) and to the log file.
    private static void Publish(IterationSummary s, ReinforcementOptions o, Device device, bool isBest, bool lead)
    {
        if (Telemetry.IsEnabled(TelemetryLevel.Training))
        {
            var metrics = new Dictionary<string, double>(s.Metrics, StringComparer.Ordinal)
            {
                ["episodes"] = s.Episodes,
                ["steps"] = s.Steps,
                ["reward-mean"] = s.RewardMean,
                ["reward-spread"] = s.RewardSpread,
                ["reward-min"] = s.RewardMin,
                ["reward-max"] = s.RewardMax,
                ["rollout-seconds"] = s.RolloutTime.TotalSeconds,
                ["update-seconds"] = s.UpdateTime.TotalSeconds,
            };
            var evaluation = s.EvaluationReward is { } e ? new Dictionary<string, double>(StringComparer.Ordinal) { ["reward-mean"] = e } : null;
            var duration = s.RolloutTime + s.UpdateTime;
            Telemetry.EpochCompleted(new EpochCompleted(s.Iteration, o.Iterations, s.Loss, metrics, null, evaluation, s.LearningRate ?? 0f, duration,
                s.Steps / Math.Max(duration.TotalSeconds, 1e-9), ComputeResources.GetMemoryUsage(device), isBest));
        }

        if (lead && o.Log is { } log)                                              // on a cluster, rank 0's line alone
        {
            using var stream = new FileStream(log, FileMode.Append, FileAccess.Write, FileShare.Read);
            using (var writer = new Utf8JsonWriter(stream))
            {
                writer.WriteStartObject();
                writer.WriteString("event", "iteration");                         // a run's records name their kind (the dashboard reads them)
                writer.WriteNumber("iteration", s.Iteration);
                writer.WriteNumber("iterations", o.Iterations);
                writer.WriteNumber("episodes", s.Episodes);
                writer.WriteNumber("steps", s.Steps);
                writer.WriteNumber("reward_mean", s.RewardMean);
                writer.WriteNumber("reward_spread", s.RewardSpread);
                writer.WriteNumber("reward_min", s.RewardMin);
                writer.WriteNumber("reward_max", s.RewardMax);
                if (s.EvaluationReward is { } evaluation)
                {
                    writer.WriteNumber("evaluation_reward", evaluation);
                }

                writer.WriteNumber("loss", s.Loss);
                writer.WriteNumber("updates", s.Updates);
                if (s.LearningRate is { } rate)
                {
                    writer.WriteNumber("learning_rate", rate);
                }

                foreach (var (name, value) in s.Metrics)
                {
                    writer.WriteNumber(name.Replace('-', '_'), double.IsFinite(value) ? value : 0);
                }

                writer.WriteNumber("rollout_seconds", s.RolloutTime.TotalSeconds);
                writer.WriteNumber("update_seconds", s.UpdateTime.TotalSeconds);
                writer.WriteEndObject();
            }

            stream.WriteByte((byte)'\n');
        }
    }

    // Whether this trainer writes the run's files: in one process, or rank 0 of a cluster (every rank runs a trainer).
    private bool Lead => _policy is not IRolloutRanks<TObservation, TAction> { WorldSize: > 1, Rank: > 0 };

    // The checkpoint after `iterations`: the policy, then the run's state, written into a folder of its own and moved in
    // place whole, so a stop part-way leaves the previous checkpoint as it was. On a cluster every rank saves the policy
    // (a collective: the policy's rank 0 writes it) and rank 0 alone writes the folder and the state.
    private string WriteCheckpoint(string root, int iterations, int seed, long updates)
    {
        string folder = Path.Combine(root, CheckpointPrefix + iterations.ToString("D6", CultureInfo.InvariantCulture));
        string writing = folder + ".writing";
        if (!Lead)
        {
            _policy.Save(Path.Combine(writing, PolicyFolder));
            return folder;
        }

        if (Directory.Exists(writing))
        {
            Directory.Delete(writing, recursive: true);
        }

        Directory.CreateDirectory(writing);
        _policy.Save(Path.Combine(writing, PolicyFolder));
        File.WriteAllText(Path.Combine(writing, StateFile), new JsonObject
        {
            ["format"] = StateFormat,
            ["iteration"] = iterations,
            ["seed"] = seed,
            ["updates"] = updates,
            ["algorithm"] = Options.Algorithm,
            ["environment"] = _kind.Name,
        }.ToJsonString() + "\n");
        if (Directory.Exists(folder))
        {
            Directory.Delete(folder, recursive: true);
        }

        Directory.Move(writing, folder);
        return folder;
    }

    // A checkpoint folder (or the latest under a folder of them) into the policy: the iterations done, the seed, the updates.
    private (int Iteration, int Seed, long Updates) ReadCheckpoint(string path)
    {
        string folder = path;
        if (!File.Exists(Path.Combine(folder, StateFile)))
        {
            folder = Directory.Exists(path)
                ? Directory.EnumerateDirectories(path, CheckpointPrefix + "*").Where(d => !d.EndsWith(".writing", StringComparison.Ordinal) && File.Exists(Path.Combine(d, StateFile)))
                    .Order(StringComparer.Ordinal).LastOrDefault() ?? folder
                : folder;
        }

        string state = Path.Combine(folder, StateFile);
        if (!File.Exists(state))
        {
            throw new FileNotFoundException($"{path} holds no reinforcement checkpoint ({StateFile}, or {CheckpointPrefix}N folders holding one); give a folder ReinforcementOptions.CheckpointFolder was set to.", state);
        }

        var json = JsonNode.Parse(File.ReadAllText(state)) as JsonObject;
        if ((string?)json?["format"] != StateFormat || json["iteration"] is null || json["seed"] is null || json["updates"] is null)
        {
            throw new InvalidDataException($"{state} is not a reinforcement checkpoint's state ({StateFormat}).");
        }

        _policy.Load(Path.Combine(folder, PolicyFolder));
        return ((int)json["iteration"]!, (int)json["seed"]!, (long)json["updates"]!);
    }
}
