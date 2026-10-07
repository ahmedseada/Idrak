// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Diagnostics;

namespace Idrak.Training.Abstractions;

/// <summary>Per-epoch results of <c>Trainer.Fit</c>, written by the trainer (and a callback that stops early).</summary>
public sealed class TrainingHistory
{
    private readonly List<EpochCompleted> _epochs = [];

    /// <summary>One entry per completed epoch.</summary>
    public IReadOnlyList<EpochCompleted> Epochs => _epochs;

    /// <summary>The epoch with the best monitored loss (1-based).</summary>
    public int BestEpoch { get; set; }

    /// <summary>The best monitored loss (validation loss when a validation set was used).</summary>
    public double BestLoss { get; set; } = double.PositiveInfinity;

    /// <summary>Whether early stopping ended training.</summary>
    public bool StoppedEarly { get; set; }

    /// <summary>Adds a completed epoch's summary.</summary>
    public void Add(EpochCompleted epoch) => _epochs.Add(epoch);
}

/// <summary>
/// Code that runs inside <c>Trainer.Fit</c> at fixed points: before the first batch, after every batch, after
/// every epoch and once at the end. Every method has an empty default, so a callback implements only what it needs.
/// Callbacks run on the training thread, in the order of <c>Trainer.Callbacks</c>.
/// </summary>
/// <example>
/// <code>
/// sealed class StopAtLoss(double target) : ITrainerCallback
/// {
///     public void OnEpochEnd(TrainerContext context, EpochCompleted epoch)
///     {
///         if (epoch.Loss &lt; target) context.Stop();
///     }
/// }
/// </code>
/// </example>
public interface ITrainerCallback
{
    /// <summary>
    /// Whether <see cref="OnBatchEnd"/> needs <see cref="BatchCompleted.Loss"/>. Reading a batch loss waits for the device,
    /// so the trainer only does it when a callback asks (or batch telemetry is on); otherwise the loss is <c>NaN</c>.
    /// Default false.
    /// </summary>
    bool NeedsBatchLoss => false;

    /// <summary>Called once before the first batch.</summary>
    void OnTrainBegin(TrainerContext context)
    {
    }

    /// <summary>Called after every optimizer step. <see cref="BatchCompleted.Loss"/> is <c>NaN</c> unless <see cref="NeedsBatchLoss"/> is true.</summary>
    void OnBatchEnd(TrainerContext context, BatchCompleted batch)
    {
    }

    /// <summary>Called after every epoch (after validation), with the summary that is added to the history.</summary>
    void OnEpochEnd(TrainerContext context, EpochCompleted epoch)
    {
    }

    /// <summary>Called once when training ends: after the last epoch, after <see cref="TrainerContext.Stop"/>, early stopping or cancellation.</summary>
    void OnTrainEnd(TrainerContext context, TrainingHistory history)
    {
    }
}

/// <summary>The state of a <c>Trainer.Fit</c> call, as <see cref="ITrainerCallback"/> methods see it.</summary>
public sealed class TrainerContext
{
    /// <summary>The state of a training run that is starting (a trainer creates it and advances <see cref="Epoch"/> and <see cref="Step"/>).</summary>
    /// <param name="model">The model being trained.</param>
    /// <param name="optimizer">The optimizer updating it.</param>
    /// <param name="epochs">The maximum number of epochs.</param>
    /// <param name="history">The history the run writes.</param>
    /// <param name="cancellationToken">Cancels the run.</param>
    public TrainerContext(Module model, Optimizer optimizer, int epochs, TrainingHistory history, CancellationToken cancellationToken)
    {
        Model = model;
        Optimizer = optimizer;
        Epochs = epochs;
        History = history;
        CancellationToken = cancellationToken;
    }

    /// <summary>The model being trained.</summary>
    public Module Model { get; }

    /// <summary>The optimizer updating the model.</summary>
    public Optimizer Optimizer { get; }

    /// <summary>The current 1-based epoch (0 before the first).</summary>
    public int Epoch { get; set; }

    /// <summary>The maximum number of epochs requested.</summary>
    public int Epochs { get; }

    /// <summary>Optimizer steps (batches) since training started.</summary>
    public long Step { get; set; }

    /// <summary>The epochs completed so far.</summary>
    public TrainingHistory History { get; }

    /// <summary>The token passed to <c>Trainer.Fit</c>.</summary>
    public CancellationToken CancellationToken { get; }

    /// <summary>Whether <see cref="Stop"/> was called.</summary>
    public bool StopRequested { get; private set; }

    /// <summary>
    /// Ends training cleanly: from <see cref="ITrainerCallback.OnBatchEnd"/> after the current batch (the epoch is
    /// summarized, validated and added to the history over the batches it ran), from <see cref="ITrainerCallback.OnEpochEnd"/>
    /// after the current epoch. <c>Trainer.Fit</c> then returns the history as usual.
    /// </summary>
    public void Stop() => StopRequested = true;
}
