using Idrak.Data;
using Idrak.Diagnostics;
using Idrak.Layers;
using Idrak.Optimizers;

namespace Idrak.Training;

/// <summary>
/// Everything a training run needs, as one object: the parameter-object form of <see cref="Trainer"/> plus
/// <see cref="Trainer.Fit"/>. Required members are the values the trainer requires today (model, loss, optimizer,
/// training data, epochs); every optional member has the same default as the trainer's property it sets. Because this
/// is a record, <c>with</c> makes a variant that differs in one setting.
/// </summary>
/// <example>
/// <code>
/// var run = new TrainingRun
/// {
///     Model = model,
///     Loss = Losses.CrossEntropy,
///     Optimizer = p => new AdamW(p, 0.003f, weightDecay: 1e-4f),
///     Scheduler = o => new CosineAnnealing(o, 40, warmupEpochs: 1),
///     Train = train.Batches(64, shuffle: true, seed: 3),
///     Validation = test.Batches(500),
///     Epochs = 40,
///     Metrics = [Metric.Accuracy],
///     MaxGradientNorm = 1f,
/// };
/// var history = run.Fit();
/// var clipped = run with { MaxGradientNorm = 0.5f };   // same run, one change (give it a fresh Model to start over)
/// </code>
/// </example>
public sealed record TrainingRun
{
    /// <summary>The model to train (<see cref="Trainer.Model"/>).</summary>
    public required Module Model { get; init; }

    /// <summary>The loss function (<see cref="Trainer.Loss"/>).</summary>
    public required Func<Tensor, Tensor, Tensor> Loss { get; init; }

    /// <summary>Creates the optimizer from the model's trainable parameters, e.g. <c>p =&gt; new Adam(p, 1e-3f)</c>.</summary>
    public required Func<IEnumerable<Tensor>, Optimizer> Optimizer { get; init; }

    /// <summary>The training batches (the <c>train</c> argument of <see cref="Trainer.Fit"/>).</summary>
    public required DataLoader Train { get; init; }

    /// <summary>The number of epochs (the <c>epochs</c> argument of <see cref="Trainer.Fit"/>).</summary>
    public required int Epochs { get; init; }

    /// <summary>Validation batches (the <c>validation</c> argument of <see cref="Trainer.Fit"/>), or null.</summary>
    public DataLoader? Validation { get; init; }

    /// <summary>Creates the learning-rate schedule from the optimizer (<see cref="Trainer.Scheduler"/>), or null.</summary>
    public Func<Optimizer, LearningRateScheduler>? Scheduler { get; init; }

    /// <summary>Metrics to report (<see cref="Trainer.Metrics"/>).</summary>
    public IReadOnlyList<Metric> Metrics { get; init; } = [];

    /// <summary><see cref="Trainer.EarlyStoppingPatience"/>.</summary>
    public int? EarlyStoppingPatience { get; init; }

    /// <summary><see cref="Trainer.RestoreBestWeights"/> (default true, as on the trainer).</summary>
    public bool RestoreBestWeights { get; init; } = true;

    /// <summary><see cref="Trainer.MinImprovement"/>.</summary>
    public double MinImprovement { get; init; }

    /// <summary><see cref="Trainer.MaxGradientNorm"/>.</summary>
    public float? MaxGradientNorm { get; init; }

    /// <summary><see cref="Trainer.OnEpoch"/>.</summary>
    public Action<EpochCompleted>? OnEpoch { get; init; }

    /// <summary>
    /// <see cref="Trainer.Callbacks"/>, in order. The same callback objects are used by every trainer this run creates;
    /// the built-in ones reset their state in <see cref="ITrainerCallback.OnTrainBegin"/>.
    /// </summary>
    public IReadOnlyList<ITrainerCallback> Callbacks { get; init; } = [];

    /// <summary>
    /// Creates the <see cref="Trainer"/> these settings describe (for <see cref="Trainer.Evaluate"/>,
    /// <see cref="Trainer.Predict"/> or repeated fitting). Dispose it to release the optimizer.
    /// </summary>
    public Trainer CreateTrainer()
    {
        var trainer = new Trainer(Model, Loss, Optimizer, Scheduler)
        {
            EarlyStoppingPatience = EarlyStoppingPatience,
            RestoreBestWeights = RestoreBestWeights,
            MinImprovement = MinImprovement,
            MaxGradientNorm = MaxGradientNorm,
            OnEpoch = OnEpoch,
        };
        trainer.Metrics.AddRange(Metrics);
        trainer.Callbacks.AddRange(Callbacks);
        return trainer;
    }

    /// <summary>Creates the trainer and runs <see cref="Trainer.Fit"/>.</summary>
    public TrainingHistory Fit(CancellationToken cancellationToken = default)
    {
        using var trainer = CreateTrainer();
        return trainer.Fit(Train, Epochs, Validation, cancellationToken);
    }

    /// <summary>Creates the trainer and runs <see cref="Trainer.FitAsync"/>.</summary>
    public async Task<TrainingHistory> FitAsync(IProgress<EpochCompleted>? progress = null, CancellationToken cancellationToken = default)
    {
        using var trainer = CreateTrainer();
        return await trainer.FitAsync(Train, Epochs, Validation, progress, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Creates the trainer and runs <see cref="Trainer.TrainAsync"/>.</summary>
    public async IAsyncEnumerable<EpochCompleted> TrainAsync(
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        using var trainer = CreateTrainer();
        await foreach (var epoch in trainer.TrainAsync(Train, Epochs, Validation, cancellationToken).ConfigureAwait(false))
        {
            yield return epoch;
        }
    }

    /// <summary>Creates a trainer (with these loss and metrics) and runs <see cref="Trainer.Evaluate"/> on <paramref name="data"/>.</summary>
    public EvaluationResult Evaluate(DataLoader data)
    {
        using var trainer = CreateTrainer();
        return trainer.Evaluate(data);
    }
}
