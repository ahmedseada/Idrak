// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Diagnostics;

/// <summary>Published once when <see cref="Training.Trainer.Fit"/> begins.</summary>
/// <param name="Model">A layer-by-layer summary of the model.</param>
/// <param name="Optimizer">The optimizer type, e.g. "Adam".</param>
/// <param name="Device">Where training runs.</param>
/// <param name="Epochs">The maximum number of epochs requested.</param>
/// <param name="TrainingSamples">Samples per training epoch (0 when the batch source does not say).</param>
/// <param name="ValidationSamples">Samples in the validation set, if one was given.</param>
/// <param name="BatchSize">Samples per batch (0 when the batch source does not say).</param>
/// <param name="BatchesPerEpoch">Optimizer steps per epoch (0 when the batch source does not say).</param>
/// <param name="ParameterCount">Trainable values in the model.</param>
/// <param name="LearningRate">The optimizer's learning rate at the start.</param>
/// <param name="CpuThreads">The CPU thread budget from <see cref="ComputeResources.MaxCpuThreads"/>.</param>
public readonly record struct TrainingStarted(
    string Model,
    string Optimizer,
    Device Device,
    int Epochs,
    int TrainingSamples,
    int? ValidationSamples,
    int BatchSize,
    int BatchesPerEpoch,
    long ParameterCount,
    float LearningRate,
    int CpuThreads);

/// <summary>Published after every optimizer step when <see cref="TelemetryLevel.Batches"/> is enabled.</summary>
/// <param name="Epoch">1-based epoch number.</param>
/// <param name="Batch">1-based batch number within the epoch.</param>
/// <param name="BatchesPerEpoch">Batches in the epoch (0 when the batch source does not say).</param>
/// <param name="Step">1-based optimizer step since training started.</param>
/// <param name="BatchSize">Samples in this batch.</param>
/// <param name="Loss">The batch loss.</param>
/// <param name="LearningRate">The learning rate used for this step.</param>
/// <param name="GradientNorm">Global L2 norm of all gradients, when <see cref="TelemetryLevel.Gradients"/> is enabled.</param>
/// <param name="DataTime">Time spent waiting for the data loader.</param>
/// <param name="ComputeTime">Forward, backward and optimizer step time.</param>
public readonly record struct BatchCompleted(
    int Epoch,
    int Batch,
    int BatchesPerEpoch,
    long Step,
    int BatchSize,
    double Loss,
    float LearningRate,
    double? GradientNorm,
    TimeSpan DataTime,
    TimeSpan ComputeTime);

/// <summary>Published at the end of every epoch.</summary>
/// <param name="Epoch">1-based epoch number.</param>
/// <param name="Epochs">The maximum number of epochs.</param>
/// <param name="Loss">Mean training loss over the epoch.</param>
/// <param name="Metrics">Mean training metrics by name (e.g. "mae").</param>
/// <param name="ValidationLoss">Validation loss, when a validation set was given.</param>
/// <param name="ValidationMetrics">Validation metrics by name, when a validation set was given.</param>
/// <param name="LearningRate">The optimizer's learning rate at the end of the epoch.</param>
/// <param name="Duration">Wall time for the epoch, including validation.</param>
/// <param name="SamplesPerSecond">Training throughput.</param>
/// <param name="Memory">Tensor memory on the training device at the end of the epoch.</param>
/// <param name="IsBest">Whether this epoch has the best monitored loss so far.</param>
public readonly record struct EpochCompleted(
    int Epoch,
    int Epochs,
    double Loss,
    IReadOnlyDictionary<string, double> Metrics,
    double? ValidationLoss,
    IReadOnlyDictionary<string, double>? ValidationMetrics,
    float LearningRate,
    TimeSpan Duration,
    double SamplesPerSecond,
    MemoryUsage Memory,
    bool IsBest);

/// <summary>Published once when fitting ends.</summary>
/// <param name="EpochsRun">Epochs actually completed.</param>
/// <param name="Duration">Total fitting time.</param>
/// <param name="FinalLoss">Training loss of the last epoch.</param>
/// <param name="BestEpoch">The epoch with the best monitored loss.</param>
/// <param name="BestLoss">The best monitored loss (validation loss when available).</param>
/// <param name="StoppedEarly">True when early stopping ended training.</param>
/// <param name="Cancelled">True when the cancellation token ended training.</param>
public readonly record struct TrainingCompleted(
    int EpochsRun,
    TimeSpan Duration,
    double FinalLoss,
    int BestEpoch,
    double BestLoss,
    bool StoppedEarly,
    bool Cancelled);

/// <summary>Published after each module's forward pass when <see cref="TelemetryLevel.Layers"/> is enabled.</summary>
/// <param name="Layer">The module's display name.</param>
/// <param name="LayerType">The module's type name.</param>
/// <param name="Depth">Nesting depth: 0 for the outermost module.</param>
/// <param name="InputShape">Shape of the input.</param>
/// <param name="OutputShape">Shape of the output.</param>
/// <param name="Duration">Time spent (launch time on GPU unless <see cref="Telemetry.SynchronizeForTiming"/> is set).</param>
/// <param name="Device">The device the output lives on.</param>
/// <param name="Training">Whether the module was in training mode.</param>
public readonly record struct LayerForward(
    string Layer,
    string LayerType,
    int Depth,
    int[] InputShape,
    int[] OutputShape,
    TimeSpan Duration,
    Device Device,
    bool Training);

/// <summary>Published for every tensor operation when <see cref="TelemetryLevel.Operations"/> is enabled.</summary>
/// <param name="Operation">Operation name, e.g. "matmul".</param>
/// <param name="Backward">True for the gradient computation of the operation.</param>
/// <param name="Shape">Shape of the operation's output.</param>
/// <param name="Elements">Number of output elements.</param>
/// <param name="Device">Where it ran.</param>
/// <param name="Duration">Time spent (launch time on GPU unless <see cref="Telemetry.SynchronizeForTiming"/> is set).</param>
public readonly record struct OperationCompleted(
    string Operation,
    bool Backward,
    int[] Shape,
    int Elements,
    Device Device,
    TimeSpan Duration);

/// <summary>Published after each inference call when <see cref="TelemetryLevel.Inference"/> is enabled.</summary>
/// <param name="Model">The module's display name.</param>
/// <param name="Samples">Rows predicted.</param>
/// <param name="InputShape">Shape of the input batch.</param>
/// <param name="OutputShape">Shape of the predictions.</param>
/// <param name="Device">Where inference ran.</param>
/// <param name="Latency">End-to-end time including device synchronization.</param>
public readonly record struct InferenceCompleted(
    string Model,
    int Samples,
    int[] InputShape,
    int[] OutputShape,
    Device Device,
    TimeSpan Latency)
{
    /// <summary>Rows per second.</summary>
    public double SamplesPerSecond => Latency.TotalSeconds > 0 ? Samples / Latency.TotalSeconds : double.PositiveInfinity;
}

/// <summary>Published after each tool call when <see cref="TelemetryLevel.Tools"/> is enabled.</summary>
/// <param name="Tool">The tool's name.</param>
/// <param name="Arguments">The arguments as JSON text.</param>
/// <param name="Duration">Time spent in the tool.</param>
/// <param name="Succeeded">False when the tool threw, timed out, was denied or got invalid arguments.</param>
/// <param name="Error">What went wrong, or null.</param>
public readonly record struct ToolCallCompleted(string Tool, string Arguments, TimeSpan Duration, bool Succeeded, string? Error);

/// <summary>What happened in the inference engine.</summary>
public enum EngineEventKind
{
    /// <summary>A model finished loading (and warming up, if set).</summary>
    ModelLoaded,

    /// <summary>A model was unloaded (keep-alive expired, or on request).</summary>
    ModelUnloaded,

    /// <summary>A request completed successfully.</summary>
    RequestCompleted,

    /// <summary>A request was rejected (queue full), timed out or failed.</summary>
    RequestRejected,
}

/// <summary>Published by the inference engine when <see cref="TelemetryLevel.Engine"/> is enabled.</summary>
/// <param name="Kind">What happened.</param>
/// <param name="Model">The model's name in the engine.</param>
/// <param name="Duration">Load time, or request latency (including queue wait).</param>
/// <param name="QueueWait">Time the request waited before running (requests only).</param>
/// <param name="BatchSize">Rows processed together with this request (predictor requests only; 0 otherwise).</param>
/// <param name="Reason">Why a request was rejected, or null.</param>
public readonly record struct EngineEvent(EngineEventKind Kind, string Model, TimeSpan Duration, TimeSpan QueueWait, int BatchSize, string? Reason);
