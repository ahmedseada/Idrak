// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Data;
using Idrak.Data.Abstractions;
using Idrak.Diagnostics;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training.Abstractions;

namespace Idrak.Training;

/// <summary>A quantity tracked during training and evaluation, computed as a mean over samples.</summary>
/// <param name="Name">Short name used in logs, e.g. "mae".</param>
/// <param name="BatchMean">Returns the batch mean as a scalar tensor, given (predictions, targets).</param>
/// <param name="Finalize">Optional transform of the epoch mean, e.g. a square root for RMSE.</param>
public sealed record Metric(string Name, Func<Tensor, Tensor, Tensor> BatchMean, Func<double, double>? Finalize = null)
{
    /// <summary>Mean absolute error.</summary>
    public static Metric MeanAbsoluteError { get; } = new("mae", Losses.MeanAbsoluteError);

    /// <summary>Mean squared error.</summary>
    public static Metric MeanSquaredError { get; } = new("mse", Losses.MeanSquaredError);

    /// <summary>Root mean squared error (in the units of the targets).</summary>
    public static Metric RootMeanSquaredError { get; } = new("rmse", Losses.MeanSquaredError, Math.Sqrt);

    /// <summary>
    /// Classification accuracy: the arg-max of each prediction row matches the arg-max of the one-hot target.
    /// For single-column outputs it compares probabilities with 0.5 (see <see cref="BinaryAccuracy"/>).
    /// </summary>
    public static Metric Accuracy { get; } = new("accuracy", (p, t) => Tensor.MatchRate(p, t, 0.5f));

    /// <summary>Accuracy with integer class targets (the logits' shape without the last dimension), e.g. next-token prediction.</summary>
    public static Metric SparseAccuracy { get; } = new("accuracy", (p, t) =>
    {
        using var oneHot = Tensor.OneHot(t.Reshape(p.Shape[..^1]), p.Shape[^1]);
        return Tensor.MatchRate(p, oneHot, 0.5f);
    });

    /// <summary>Binary accuracy for a single output column: (prediction ≥ threshold) == (target ≥ 0.5). Use threshold 0 for logits.</summary>
    public static Metric BinaryAccuracy(float threshold = 0.5f) => new("accuracy", (p, t) => Tensor.MatchRate(p, t, threshold));
}

/// <summary>Loss and metrics of a model on a dataset.</summary>
/// <param name="Loss">Mean loss.</param>
/// <param name="Metrics">Metric values by name.</param>
/// <param name="Samples">Samples evaluated.</param>
public sealed record EvaluationResult(double Loss, IReadOnlyDictionary<string, double> Metrics, int Samples);

/// <summary>
/// Runs the training loop: batches from a <see cref="DataLoader"/> (or any other <see cref="IBatchSource"/>), forward,
/// loss, backward, optimizer step, metrics, validation, early stopping, and telemetry for every step. Memory for each
/// step is recycled automatically, and the loss is accumulated on the device so the GPU is only synchronized once per
/// epoch (unless per-batch telemetry is requested).
/// </summary>
/// <example>
/// <code>
/// var trainer = new Trainer(model, new Adam(model.Parameters(), 1e-3f), Losses.MeanSquaredError)
/// {
///     Metrics = { Metric.MeanAbsoluteError },
///     EarlyStoppingPatience = 20,
/// };
/// var history = trainer.Fit(trainLoader, epochs: 300, validation: testLoader);
/// </code>
/// </example>
public sealed class Trainer(Module model, Optimizer optimizer, Func<Tensor, Tensor, Tensor> loss) : IDisposable
{
    private readonly bool _ownsOptimizer;

    /// <summary>
    /// Creates the trainer from factories instead of ready-made objects: <paramref name="optimizer"/> receives the
    /// model's trainable parameters (those with <see cref="Tensor.RequiresGrad"/>; all of them unless some were frozen),
    /// and <paramref name="scheduler"/>, when given, receives the created optimizer. The trainer owns the optimizer it
    /// created and disposes it in <see cref="Dispose"/>.
    /// </summary>
    /// <example>
    /// <code>
    /// using var trainer = new Trainer(model, Losses.CrossEntropy,
    ///     optimizer: p => new AdamW(p, 0.003f, weightDecay: 1e-4f),
    ///     scheduler: o => new CosineAnnealing(o, epochs, warmupEpochs: 1));
    /// </code>
    /// </example>
    public Trainer(Module model, Func<Tensor, Tensor, Tensor> loss, Func<IEnumerable<Tensor>, Optimizer> optimizer,
        Func<Optimizer, LearningRateScheduler>? scheduler = null)
        : this(model, CreateOptimizer(model, optimizer), loss)
    {
        _ownsOptimizer = true;
        Scheduler = scheduler?.Invoke(Optimizer);
    }

    private static Optimizer CreateOptimizer(Module model, Func<IEnumerable<Tensor>, Optimizer> factory)
    {
        ArgumentNullException.ThrowIfNull(factory);
        return factory(model.Parameters().Where(p => p.RequiresGrad).ToList());
    }

    /// <summary>Called on the training thread after every epoch, with the same summary telemetry publishes.</summary>
    public Action<EpochCompleted>? OnEpoch { get; init; }

    /// <summary>
    /// Code run at fixed points of <see cref="Fit"/>, in this order (see <see cref="ITrainerCallback"/>): e.g.
    /// <see cref="EarlyStopping"/>, <see cref="Checkpoint"/> and <see cref="CsvLog"/>, or your own.
    /// </summary>
    public List<ITrainerCallback> Callbacks { get; } = [];

    /// <summary>The model being trained.</summary>
    public Module Model { get; } = model;

    /// <summary>The optimizer updating the model.</summary>
    public Optimizer Optimizer { get; } = optimizer;

    /// <summary>The loss function (predictions, targets) → scalar.</summary>
    public Func<Tensor, Tensor, Tensor> Loss { get; } = loss;

    /// <summary>Extra quantities to report each epoch.</summary>
    public List<Metric> Metrics { get; } = [];

    /// <summary>
    /// Stop when the monitored loss (validation loss if available, else training loss) has not improved
    /// for this many epochs. Null disables early stopping.
    /// </summary>
    public int? EarlyStoppingPatience { get; init; }

    /// <summary>
    /// When early stopping is on, restore the weights of the best epoch at the end of training, whether early stopping
    /// ended it or it ran every epoch. Default true.
    /// </summary>
    public bool RestoreBestWeights { get; init; } = true;

    /// <summary>Minimum decrease of the monitored loss that counts as an improvement.</summary>
    public double MinImprovement { get; init; }

    /// <summary>Adjusts the learning rate after every epoch (e.g. <see cref="CosineAnnealing"/>).</summary>
    public LearningRateScheduler? Scheduler { get; init; }

    /// <summary>When set, gradients are clipped to this global L2 norm before each step (recommended for RNNs).</summary>
    public float? MaxGradientNorm { get; init; }

    private Device Device => Model.WeightsDevice ?? Device.Default;

    /// <summary>
    /// Trains for up to <paramref name="epochs"/> passes over <paramref name="train"/>. Training ends early when early
    /// stopping triggers or a callback calls <see cref="TrainerContext.Stop"/> (the history is returned as usual), or when
    /// <paramref name="cancellationToken"/> is cancelled: the loop ends after the current batch, callbacks see
    /// <see cref="ITrainerCallback.OnTrainEnd"/>, and <see cref="OperationCanceledException"/> is thrown.
    /// </summary>
    public TrainingHistory Fit(IBatchSource train, int epochs, IBatchSource? validation = null, CancellationToken cancellationToken = default)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(epochs);
        var device = Device;
        var history = new TrainingHistory();
        var totalTime = Stopwatch.StartNew();
        float[][]? bestWeights = null;
        int epochsWithoutImprovement = 0;
        long step = 0;
        bool cancelled = false;
        var callbacks = Callbacks.ToArray();
        var context = new TrainerContext(Model, Optimizer, epochs, history, cancellationToken);
        bool batchCallbacks = callbacks.Length > 0;
        bool callbacksNeedLoss = callbacks.Any(c => c.NeedsBatchLoss);

        if (Telemetry.IsEnabled(TelemetryLevel.Training))
        {
            Telemetry.TrainingStarted(new TrainingStarted(
                Model.Summary(), Optimizer.GetType().Name, device, epochs, train.SampleCount ?? 0, validation?.SampleCount,
                train.BatchSize ?? 0, train.BatchCount ?? 0, Model.ParameterCount, Optimizer.LearningRate, ComputeResources.MaxCpuThreads));
        }

        foreach (var callback in callbacks)
        {
            callback.OnTrainBegin(context);
        }

        // Running sums live on the device: [loss, metric 1, metric 2, ...], each weighted by batch size.
        using var sums = Tensor.Zeros([1 + Metrics.Count], device);
        int epoch = 0;
        for (epoch = 1; epoch <= epochs; epoch++)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                cancelled = true;
                epoch--;
                break;
            }

            context.Epoch = epoch;
            var epochTime = Stopwatch.StartNew();
            Model.Train();
            sums.Backend.Fill(sums.Storage, sums.Size, 0f);
            int samples = 0;
            bool perBatch = Telemetry.IsEnabled(TelemetryLevel.Batches);
            long dataStart = Stopwatch.GetTimestamp();

            foreach (var batch in train)
            {
                using var _ = batch;
                using var scope = new TensorScope();
                var dataTime = Stopwatch.GetElapsedTime(dataStart);
                long computeStart = Stopwatch.GetTimestamp();

                var predictions = Model.Forward(batch.Features);
                var batchLoss = Loss(predictions, batch.Targets);
                Optimizer.ZeroGrad();
                batchLoss.Backward();
                double? gradientNorm = MaxGradientNorm is { } maxNorm ? Optimizer.ClipGradientNorm(maxNorm)
                    : Telemetry.IsEnabled(TelemetryLevel.Gradients) ? Optimizer.GradientNorm() : null;
                Optimizer.Step();
                Accumulate(sums, batchLoss, predictions, batch.Targets, batch.Size);
                samples += batch.Size;
                step++;
                context.Step = step;

                if (perBatch || batchCallbacks)
                {
                    // Reading the loss synchronizes (so compute time is real time); skipped when nobody needs it.
                    double lossValue = perBatch || callbacksNeedLoss ? batchLoss.Item() : double.NaN;
                    var completed = new BatchCompleted(epoch, batch.Index + 1, train.BatchCount ?? 0, step, batch.Size,
                        lossValue, Optimizer.LearningRate, gradientNorm, dataTime, Stopwatch.GetElapsedTime(computeStart));
                    if (perBatch)
                    {
                        Telemetry.BatchCompleted(completed);
                    }

                    foreach (var callback in callbacks)
                    {
                        callback.OnBatchEnd(context, completed);
                    }
                }

                if (context.StopRequested)
                {
                    break;
                }

                if (cancellationToken.IsCancellationRequested)
                {
                    cancelled = true;
                    break;
                }

                dataStart = Stopwatch.GetTimestamp();
            }

            if (cancelled)
            {
                epoch--;   // the unfinished epoch is not summarized
                break;
            }

            var (trainLoss, trainMetrics) = Read(sums, samples);
            EvaluationResult? validationResult = validation is null ? null : Evaluate(validation);
            double monitored = validationResult?.Loss ?? trainLoss;
            bool isBest = monitored < history.BestLoss - MinImprovement;
            if (isBest)
            {
                history.BestLoss = monitored;
                history.BestEpoch = epoch;
                epochsWithoutImprovement = 0;
                if (EarlyStoppingPatience is not null && RestoreBestWeights)
                {
                    bestWeights = [.. Model.Parameters().Select(p => p.ToArray())];
                }
            }
            else
            {
                epochsWithoutImprovement++;
            }

            epochTime.Stop();
            var summary = new EpochCompleted(epoch, epochs, trainLoss, trainMetrics, validationResult?.Loss, validationResult?.Metrics,
                Optimizer.LearningRate, epochTime.Elapsed, samples / Math.Max(epochTime.Elapsed.TotalSeconds, 1e-9),
                ComputeResources.GetMemoryUsage(device), isBest);
            history.Add(summary);
            if (Telemetry.IsEnabled(TelemetryLevel.Training))
            {
                Telemetry.EpochCompleted(summary);
            }

            OnEpoch?.Invoke(summary);
            _epochCallback?.Invoke(summary);
            foreach (var callback in callbacks)
            {
                callback.OnEpochEnd(context, summary);
            }

            Scheduler?.Step();
            if (EarlyStoppingPatience is { } patience && epochsWithoutImprovement >= patience)
            {
                history.StoppedEarly = true;
                break;
            }

            if (context.StopRequested)
            {
                break;
            }
        }

        // Also when the run reached its last epoch: a last epoch that was not the best leaves worse weights than
        // the history's BestEpoch reports.
        if (bestWeights is not null && history.Epochs.Count > 0 && history.Epochs[^1].Epoch != history.BestEpoch)
        {
            foreach (var (parameter, values) in Model.Parameters().Zip(bestWeights))
            {
                parameter.Load(values);
            }
        }

        foreach (var callback in callbacks)
        {
            callback.OnTrainEnd(context, history);
        }

        if (Telemetry.IsEnabled(TelemetryLevel.Training))
        {
            Telemetry.TrainingCompleted(new TrainingCompleted(Math.Min(epoch, epochs), totalTime.Elapsed,
                history.Epochs.Count > 0 ? history.Epochs[^1].Loss : double.NaN, history.BestEpoch, history.BestLoss, history.StoppedEarly, cancelled));
        }

        cancellationToken.ThrowIfCancellationRequested();
        return history;
    }

    private Action<EpochCompleted>? _epochCallback;

    /// <summary>
    /// Runs <see cref="Fit"/> on a thread-pool thread, so UI and server threads stay free. <paramref name="progress"/>
    /// receives every epoch summary (on the captured synchronization context, as <see cref="Progress{T}"/> does).
    /// </summary>
    public Task<TrainingHistory> FitAsync(IBatchSource train, int epochs, IBatchSource? validation = null,
        IProgress<EpochCompleted>? progress = null, CancellationToken cancellationToken = default)
    {
        return Task.Run(() =>
        {
            _epochCallback = progress is null ? null : progress.Report;
            try
            {
                return Fit(train, epochs, validation, cancellationToken);
            }
            finally
            {
                _epochCallback = null;
            }
        }, CancellationToken.None);
    }

    /// <summary>
    /// Trains on a thread-pool thread and yields every epoch summary as it completes. Leaving the <c>await foreach</c>
    /// early (break, exception) stops training before the next epoch and waits for it to end.
    /// </summary>
    public async IAsyncEnumerable<EpochCompleted> TrainAsync(IBatchSource train, int epochs, IBatchSource? validation = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var channel = System.Threading.Channels.Channel.CreateUnbounded<EpochCompleted>(
            new System.Threading.Channels.UnboundedChannelOptions { SingleReader = true, SingleWriter = true });
        var training = Task.Run(() =>
        {
            _epochCallback = e => channel.Writer.TryWrite(e);
            try
            {
                Fit(train, epochs, validation, stop.Token);
                channel.Writer.TryComplete();
            }
            catch (OperationCanceledException) when (stop.IsCancellationRequested && !cancellationToken.IsCancellationRequested)
            {
                channel.Writer.TryComplete();
            }
            catch (Exception ex)
            {
                channel.Writer.TryComplete(ex);
            }
            finally
            {
                _epochCallback = null;
            }
        }, CancellationToken.None);

        try
        {
            await foreach (var epoch in channel.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
            {
                yield return epoch;
            }
        }
        finally
        {
            stop.Cancel();
            await training.ConfigureAwait(false);
        }
    }

    /// <summary>Disposes the optimizer when this trainer created it (the factory constructor); otherwise does nothing.</summary>
    public void Dispose()
    {
        if (_ownsOptimizer)
        {
            Optimizer.Dispose();
        }
    }

    /// <summary>Computes the loss and metrics on <paramref name="data"/> in evaluation mode.</summary>
    public EvaluationResult Evaluate(IBatchSource data)
    {
        bool wasTraining = Model.IsTraining;
        Model.Eval();
        try
        {
            using var sums = Tensor.Zeros([1 + Metrics.Count], (data as DataLoader)?.Device ?? Device);
            int samples = 0;
            using (Autograd.NoGrad())
            {
                foreach (var batch in data)
                {
                    using var _ = batch;
                    using var scope = new TensorScope();
                    var predictions = Model.Forward(batch.Features);
                    Accumulate(sums, Loss(predictions, batch.Targets), predictions, batch.Targets, batch.Size);
                    samples += batch.Size;
                }
            }

            var (lossValue, metrics) = Read(sums, samples);
            return new EvaluationResult(lossValue, metrics, samples);
        }
        finally
        {
            Model.Train(wasTraining);
        }
    }

    /// <summary>Predicts every sample of <paramref name="data"/>, batch by batch; returns [Count, outputs] (each sample's output flattened).</summary>
    public float[,] Predict(ISampleSource data, int batchSize = 1024)
    {
        var device = Device;
        float[,]? result = null;
        foreach (var batch in new DataLoader(data, batchSize, device: device))
        {
            using var _ = batch;
            using var output = Model.Predict(batch.Features);
            var values = output.ToArray();
            int outputs = output.Size / batch.Size;
            result ??= new float[data.Count, outputs];
            int row0 = batch.Index * batchSize;
            for (int r = 0; r < batch.Size; r++)
            {
                for (int c = 0; c < outputs; c++)
                {
                    result[row0 + r, c] = values[r * outputs + c];
                }
            }
        }

        return result ?? new float[0, 0];
    }

    /// <summary>Adds batch-size-weighted loss and metric means into the device-side running sums.</summary>
    private void Accumulate(Tensor sums, Tensor batchLoss, Tensor predictions, Tensor targets, int batchSize)
    {
        var backend = sums.Backend;
        backend.AxpyAt(batchLoss.Storage, sums.Storage, 0, batchSize);
        if (Metrics.Count == 0)
        {
            return;
        }

        using (Autograd.NoGrad())
        {
            for (int i = 0; i < Metrics.Count; i++)
            {
                using var value = Metrics[i].BatchMean(predictions, targets);
                backend.AxpyAt(value.Storage, sums.Storage, i + 1, batchSize);
            }
        }
    }

    private (double Loss, Dictionary<string, double> Metrics) Read(Tensor sums, int samples)
    {
        var values = sums.ToArray();
        double n = Math.Max(samples, 1);
        var metrics = new Dictionary<string, double>(Metrics.Count);
        for (int i = 0; i < Metrics.Count; i++)
        {
            double mean = values[i + 1] / n;
            metrics[Metrics[i].Name] = Metrics[i].Finalize?.Invoke(mean) ?? mean;
        }

        return (values[0] / n, metrics);
    }
}

/// <summary>Standard regression scores computed from predictions and true values.</summary>
/// <param name="MeanAbsoluteError">Mean |prediction - actual|.</param>
/// <param name="RootMeanSquaredError">sqrt(mean((prediction - actual)²)).</param>
/// <param name="MeanAbsolutePercentageError">Mean |prediction - actual| / |actual|, as a fraction (0.05 = 5%).</param>
/// <param name="RSquared">Coefficient of determination: 1 is perfect, 0 is no better than predicting the mean.</param>
public readonly record struct RegressionReport(double MeanAbsoluteError, double RootMeanSquaredError, double MeanAbsolutePercentageError, double RSquared)
{
    /// <summary>Compares predictions with actual values (same length, any column layout).</summary>
    public static RegressionReport Compute(ReadOnlySpan<float> predicted, ReadOnlySpan<float> actual)
    {
        if (predicted.Length != actual.Length || actual.Length == 0)
        {
            throw new ArgumentException("predicted and actual must be non-empty and the same length.");
        }

        double mean = 0;
        foreach (float a in actual)
        {
            mean += a;
        }

        mean /= actual.Length;
        double abs = 0, sq = 0, pct = 0, total = 0;
        int pctCount = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            double error = predicted[i] - actual[i];
            abs += Math.Abs(error);
            sq += error * error;
            total += (actual[i] - mean) * (actual[i] - mean);
            if (actual[i] != 0)
            {
                pct += Math.Abs(error / actual[i]);
                pctCount++;
            }
        }

        return new RegressionReport(abs / actual.Length, Math.Sqrt(sq / actual.Length), pctCount > 0 ? pct / pctCount : double.NaN,
            total > 0 ? 1 - sq / total : double.NaN);
    }
}
