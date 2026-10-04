// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Diagnostics;
using Idrak.Layers;
using Idrak.Optimizers;

namespace Idrak.Training;

/// <summary>
/// Code that runs inside <see cref="Trainer.Fit"/> at fixed points: before the first batch, after every batch, after
/// every epoch and once at the end. Every method has an empty default, so a callback implements only what it needs.
/// Callbacks run on the training thread, in the order of <see cref="Trainer.Callbacks"/>.
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

/// <summary>The state of a <see cref="Trainer.Fit"/> call, as <see cref="ITrainerCallback"/> methods see it.</summary>
public sealed class TrainerContext
{
    internal TrainerContext(Trainer trainer, int epochs, TrainingHistory history, CancellationToken cancellationToken)
    {
        Trainer = trainer;
        Epochs = epochs;
        History = history;
        CancellationToken = cancellationToken;
    }

    /// <summary>The trainer running the loop.</summary>
    public Trainer Trainer { get; }

    /// <summary>The model being trained.</summary>
    public Module Model => Trainer.Model;

    /// <summary>The optimizer updating the model.</summary>
    public Optimizer Optimizer => Trainer.Optimizer;

    /// <summary>The current 1-based epoch (0 before the first).</summary>
    public int Epoch { get; internal set; }

    /// <summary>The maximum number of epochs requested.</summary>
    public int Epochs { get; }

    /// <summary>Optimizer steps (batches) since training started.</summary>
    public long Step { get; internal set; }

    /// <summary>The epochs completed so far.</summary>
    public TrainingHistory History { get; }

    /// <summary>The token passed to <see cref="Trainer.Fit"/>.</summary>
    public CancellationToken CancellationToken { get; }

    /// <summary>Whether <see cref="Stop"/> was called.</summary>
    public bool StopRequested { get; private set; }

    /// <summary>
    /// Ends training cleanly: from <see cref="ITrainerCallback.OnBatchEnd"/> after the current batch (the epoch is
    /// summarized, validated and added to the history over the batches it ran), from <see cref="ITrainerCallback.OnEpochEnd"/>
    /// after the current epoch. <see cref="Trainer.Fit"/> then returns the history as usual.
    /// </summary>
    public void Stop() => StopRequested = true;
}

/// <summary>
/// Stops training when a monitored value has not improved for <c>patience</c> epochs, and optionally puts back the
/// weights of the best epoch. With the defaults it does what <see cref="Trainer.EarlyStoppingPatience"/> and
/// <see cref="Trainer.RestoreBestWeights"/> do, and gives the same history and weights.
/// </summary>
/// <param name="patience">Epochs without improvement before stopping.</param>
/// <param name="restoreBestWeights">
/// At the end of training, restore the weights of the best epoch: when it stops training, and when training runs every
/// epoch with a later epoch worse than the best. Default true.
/// </param>
/// <param name="monitor">
/// What to watch: "loss", "val_loss", a training metric name (e.g. "mae") or "val_" + a validation metric name. Null (the
/// default) watches the validation loss when a validation set is given, else the training loss.
/// </param>
/// <param name="minImprovement">Minimum change that counts as an improvement.</param>
/// <param name="maximize">True when higher is better (e.g. "val_accuracy"). Default false: lower is better.</param>
public sealed class EarlyStopping(int patience, bool restoreBestWeights = true, string? monitor = null, double minImprovement = 0,
    bool maximize = false) : ITrainerCallback
{
    private double _best;
    private int _epochsWithoutImprovement;
    private float[][]? _bestWeights;
    private int _lastEpoch;

    /// <summary>Epochs without improvement before stopping.</summary>
    public int Patience { get; } = patience > 0 ? patience : throw new ArgumentOutOfRangeException(nameof(patience));

    /// <summary>The epoch with the best monitored value in the last run (1-based, 0 before the first epoch).</summary>
    public int BestEpoch { get; private set; }

    /// <inheritdoc />
    public void OnTrainBegin(TrainerContext context)
    {
        _best = maximize ? double.NegativeInfinity : double.PositiveInfinity;
        _epochsWithoutImprovement = 0;
        _bestWeights = null;
        _lastEpoch = 0;
        BestEpoch = 0;
    }

    /// <inheritdoc />
    public void OnEpochEnd(TrainerContext context, EpochCompleted epoch)
    {
        _lastEpoch = epoch.Epoch;
        double value = Monitored(epoch);
        bool improved = maximize ? value > _best + minImprovement : value < _best - minImprovement;
        if (improved)
        {
            _best = value;
            BestEpoch = epoch.Epoch;
            _epochsWithoutImprovement = 0;
            if (restoreBestWeights)
            {
                _bestWeights = [.. context.Model.Parameters().Select(p => p.ToArray())];
            }
        }
        else if (++_epochsWithoutImprovement >= Patience)
        {
            context.History.StoppedEarly = true;
            context.Stop();
        }
    }

    /// <inheritdoc />
    public void OnTrainEnd(TrainerContext context, TrainingHistory history)
    {
        if (_bestWeights is not null && _lastEpoch != BestEpoch)
        {
            foreach (var (parameter, values) in context.Model.Parameters().Zip(_bestWeights))
            {
                parameter.Load(values);
            }
        }

        _bestWeights = null;
    }

    private double Monitored(EpochCompleted epoch)
    {
        if (monitor is null)
        {
            return epoch.ValidationLoss ?? epoch.Loss;
        }

        if (monitor == "loss")
        {
            return epoch.Loss;
        }

        if (monitor == "val_loss")
        {
            return epoch.ValidationLoss ?? throw new InvalidOperationException("EarlyStopping monitors val_loss, but Fit was given no validation set.");
        }

        if (epoch.Metrics.TryGetValue(monitor, out double metric))
        {
            return metric;
        }

        if (monitor.StartsWith("val_", StringComparison.Ordinal) && epoch.ValidationMetrics?.TryGetValue(monitor[4..], out double validation) == true)
        {
            return validation;
        }

        throw new InvalidOperationException($"EarlyStopping monitors '{monitor}', which this run does not report (use loss, val_loss, a metric name or val_ + a metric name).");
    }
}

/// <summary>
/// Saves the model's weights during training with <see cref="Module.Save(string)"/>: <c>last.ikw</c> every
/// <c>everyEpochs</c> epochs and, when <c>keepBest</c> is true, <c>best.ikw</c> whenever an epoch has the best monitored
/// loss so far (<see cref="EpochCompleted.IsBest"/>). Load either file with <see cref="Module.Load(string)"/>. Files are
/// written next to their final name and then moved, so a crash never leaves half a file.
/// </summary>
/// <param name="directory">Where to write the files (created if missing).</param>
/// <param name="everyEpochs">Save <c>last.ikw</c> after every this many epochs. Default 1.</param>
/// <param name="keepBest">Also save <c>best.ikw</c> for the best epoch. Default true.</param>
public sealed class Checkpoint(string directory, int everyEpochs = 1, bool keepBest = true) : ITrainerCallback
{
    /// <summary>The folder the files are written to.</summary>
    public string Directory { get; } = directory ?? throw new ArgumentNullException(nameof(directory));

    /// <summary>Save <c>last.ikw</c> after every this many epochs.</summary>
    public int EveryEpochs { get; } = everyEpochs > 0 ? everyEpochs : throw new ArgumentOutOfRangeException(nameof(everyEpochs));

    /// <summary>The path of the most recent weights.</summary>
    public string LastPath => Path.Combine(Directory, "last.ikw");

    /// <summary>The path of the best epoch's weights.</summary>
    public string BestPath => Path.Combine(Directory, "best.ikw");

    /// <inheritdoc />
    public void OnTrainBegin(TrainerContext context) => System.IO.Directory.CreateDirectory(Directory);

    /// <inheritdoc />
    public void OnEpochEnd(TrainerContext context, EpochCompleted epoch)
    {
        if (epoch.Epoch % EveryEpochs == 0)
        {
            Save(context.Model, LastPath);
        }

        if (keepBest && epoch.IsBest)
        {
            Save(context.Model, BestPath);
        }
    }

    private static void Save(Module model, string path)
    {
        string temporary = path + ".tmp";
        model.Save(temporary);
        File.Move(temporary, path, overwrite: true);
    }
}

/// <summary>
/// Appends one row per epoch to a CSV file: epoch, loss, each training metric, val_loss, each validation metric and
/// learning_rate (invariant culture; empty cells when there is no validation set). The file is replaced when training
/// begins and written after every epoch, so it can be read while training runs.
/// </summary>
/// <param name="path">The CSV file to write.</param>
public sealed class CsvLog(string path) : ITrainerCallback
{
    private List<string>? _metrics;
    private List<string>? _validationMetrics;

    /// <summary>The CSV file.</summary>
    public string Path { get; } = path ?? throw new ArgumentNullException(nameof(path));

    /// <inheritdoc />
    public void OnTrainBegin(TrainerContext context)
    {
        _metrics = null;
        _validationMetrics = null;
        File.WriteAllText(Path, "");
    }

    /// <inheritdoc />
    public void OnEpochEnd(TrainerContext context, EpochCompleted epoch)
    {
        var inv = CultureInfo.InvariantCulture;
        using var writer = new StreamWriter(Path, append: true);
        if (_metrics is null)
        {
            _metrics = [.. epoch.Metrics.Keys];
            _validationMetrics = [.. epoch.ValidationMetrics?.Keys ?? []];
            writer.WriteLine(string.Join(',', new[] { "epoch", "loss" }.Concat(_metrics).Append("val_loss")
                .Concat(_validationMetrics.Select(n => "val_" + n)).Append("learning_rate")));
        }

        var cells = new List<string> { epoch.Epoch.ToString(inv), epoch.Loss.ToString("R", inv) };
        cells.AddRange(_metrics.Select(n => epoch.Metrics.TryGetValue(n, out var v) ? v.ToString("R", inv) : ""));
        cells.Add(epoch.ValidationLoss?.ToString("R", inv) ?? "");
        cells.AddRange(_validationMetrics!.Select(n => epoch.ValidationMetrics?.TryGetValue(n, out var v) == true ? v.ToString("R", inv) : ""));
        cells.Add(epoch.LearningRate.ToString("R", inv));
        writer.WriteLine(string.Join(',', cells));
    }
}
