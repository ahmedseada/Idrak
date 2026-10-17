// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Diagnostics;

/// <summary>Published once when <c>Training.Trainer.Fit</c> begins.</summary>
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

/// <summary>Published after every optimizer step when <c>TelemetryLevel.Batches</c> is enabled.</summary>
/// <param name="Epoch">1-based epoch number.</param>
/// <param name="Batch">1-based batch number within the epoch.</param>
/// <param name="BatchesPerEpoch">Batches in the epoch (0 when the batch source does not say).</param>
/// <param name="Step">1-based optimizer step since training started.</param>
/// <param name="BatchSize">Samples in this batch.</param>
/// <param name="Loss">The batch loss.</param>
/// <param name="LearningRate">The learning rate used for this step.</param>
/// <param name="GradientNorm">Global L2 norm of all gradients, when <c>TelemetryLevel.Gradients</c> is enabled.</param>
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

/// <summary>Published after each module's forward pass when <c>TelemetryLevel.Layers</c> is enabled.</summary>
/// <param name="Layer">The module's display name.</param>
/// <param name="LayerType">The module's type name.</param>
/// <param name="Depth">Nesting depth: 0 for the outermost module.</param>
/// <param name="InputShape">Shape of the input.</param>
/// <param name="OutputShape">Shape of the output.</param>
/// <param name="Duration">Time spent (launch time on GPU unless <c>Telemetry.SynchronizeForTiming</c> is set).</param>
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

/// <summary>Published for every tensor operation when <c>TelemetryLevel.Operations</c> is enabled.</summary>
/// <param name="Operation">Operation name, e.g. "matmul".</param>
/// <param name="Backward">True for the gradient computation of the operation.</param>
/// <param name="Shape">Shape of the operation's output.</param>
/// <param name="Elements">Number of output elements.</param>
/// <param name="Device">Where it ran.</param>
/// <param name="Duration">Time spent (launch time on GPU unless <c>Telemetry.SynchronizeForTiming</c> is set).</param>
public readonly record struct OperationCompleted(
    string Operation,
    bool Backward,
    int[] Shape,
    int Elements,
    Device Device,
    TimeSpan Duration);

/// <summary>Published after each inference call when <c>TelemetryLevel.Inference</c> is enabled.</summary>
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

/// <summary>Published after each tool call when <c>TelemetryLevel.Tools</c> is enabled.</summary>
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

/// <summary>Published by the inference engine when <c>TelemetryLevel.Engine</c> is enabled.</summary>
/// <param name="Kind">What happened.</param>
/// <param name="Model">The model's name in the engine.</param>
/// <param name="Duration">Load time, or request latency (including queue wait).</param>
/// <param name="QueueWait">Time the request waited before running (requests only).</param>
/// <param name="BatchSize">Rows processed together with this request (predictor requests only; 0 otherwise).</param>
/// <param name="Reason">Why a request was rejected, or null.</param>
public readonly record struct EngineEvent(EngineEventKind Kind, string Model, TimeSpan Duration, TimeSpan QueueWait, int BatchSize, string? Reason);

/// <summary>A stage of reading an answer from a prompt (<see cref="InferenceStage"/>).</summary>
public enum InferenceStageKind
{
    /// <summary>An image's bytes decoded to pixels: Count the pixel values, Bytes the encoded bytes.</summary>
    ImageDecode,

    /// <summary>An image's transforms (a request's or the reader's): Count the pixel values out.</summary>
    ImageTransforms,

    /// <summary>Images through the vision encoder: Count the images, Bytes their features.</summary>
    Encoder,

    /// <summary>The prompt's pass, to its first token read back: Count the prompt tokens fed.</summary>
    Prefill,

    /// <summary>Decoding steps since the last chunk, to their tokens read back: Count the tokens.</summary>
    DecodeChunk,

    /// <summary>A decoding step recorded as a device graph: Bytes the memory the graph keeps.</summary>
    GraphCapture,

    /// <summary>A layer's key/value cache allocated: Count its positions, Bytes its memory.</summary>
    CacheAllocation,

    /// <summary>A sound file's bytes decoded to samples: Count the samples (every channel), Bytes the encoded bytes.</summary>
    AudioDecode,

    /// <summary>Sound resampled to a model's rate: Count the samples out.</summary>
    Resample,

    /// <summary>Sound to a model's features (log-mel frames): Count the frames, Bytes the features.</summary>
    AudioFeatures,
}

/// <summary>
/// Published at <c>TelemetryLevel.Inference</c> for each stage of reading an answer (a read of a page: its image decoded and
/// transformed, encoded, the prompt's pass, each chunk of decoding steps, a step recorded as a graph, each cache
/// allocated), with its sizes. Durations are host time: device work still queued at the end of a stage counts in the next
/// one unless <c>Telemetry.SynchronizeForTiming</c> waits for it (the prefill and decode chunks end with their tokens read
/// back, so they are complete).
/// </summary>
/// <param name="Stage">Which stage.</param>
/// <param name="Device">The kind of device it ran on ("cpu", "cuda", …).</param>
/// <param name="Duration">How long it took.</param>
/// <param name="Count">What the stage counts (<see cref="InferenceStageKind"/> says what): images, tokens, positions, values.</param>
/// <param name="Bytes">The bytes it made or read (<see cref="InferenceStageKind"/> says which), or 0.</param>
public readonly record struct InferenceStage(InferenceStageKind Stage, string Device, TimeSpan Duration, long Count, long Bytes);

/// <summary>
/// Published at <c>TelemetryLevel.Suggestions</c>: a setting worth changing, computed from what the device reports (its
/// memory, its matrix units) and what a run measured; nothing is suggested where a figure is missing.
/// </summary>
/// <param name="Setting">The option, as the library names it ("ImageReaderOptions.MaxBatch", "PretrainedOptions.BFloat16").</param>
/// <param name="Value">The suggested value, as text ("3", "BFloat16", "true").</param>
/// <param name="Reason">Why, with the figures it came from.</param>
/// <param name="Device">The device it is for.</param>
/// <param name="Source">The advisor that suggested it (a registered name).</param>
public readonly record struct SettingSuggestion(string Setting, string Value, string Reason, string Device, string Source);

/// <summary>
/// Published when a device fails (<c>Devices.DeviceException</c>), to the hooks listening at
/// <c>TelemetryLevel.Devices</c>: once when the error is raised, or once by the dispatcher when it retried the operation
/// on the CPU (<c>Backend.RetryOnHost</c>).
/// </summary>
/// <param name="Device">The kind of device that failed ("cuda", "vulkan", "hip", …).</param>
/// <param name="Operation">The operation whose kernel failed (as <c>Operation.Name</c>), or null when not known.</param>
/// <param name="Message">What failed.</param>
/// <param name="RetriedOnHost">True when the operation ran again on the CPU and the caller got its result.</param>
/// <param name="Hint">How to turn the retry on, when it is off for the device; null otherwise.</param>
public readonly record struct DeviceFailed(string Device, string? Operation, string Message, bool RetriedOnHost, string? Hint);

/// <summary>
/// Published the first time an operation runs through the host fallback on a device (the device has no kernel of its
/// own for it, or none for these arguments, such as a head size above its limit), to the hooks listening at
/// <c>TelemetryLevel.Devices</c>: once per operation and device while someone listens. Every such call copies its
/// operands to the CPU and back; <c>Operations.Kernels.HostCalls</c> and <c>KernelTrace.HostCallsByOperation</c> count
/// them all.
/// </summary>
/// <param name="Device">The kind of device ("cuda", "vulkan", "hip", …).</param>
/// <param name="Operation">The operation (as <c>Operation.Name</c>, or a copy's method name).</param>
public readonly record struct HostFallback(string Device, string Operation);

/// <summary>
/// A device measured the candidates of an operation at a shape class it had not seen and kept the fastest
/// (<c>TelemetryLevel.Devices</c>). The choice is kept for the class (and in the device's tuning cache file), so the time
/// is paid once per class and device: a run whose batches keep landing in new classes (a new batch count or padded length
/// each step) shows here where its early steps' time went.
/// </summary>
/// <param name="Device">The kind of device ("cuda", "vulkan", "hip", …).</param>
/// <param name="Operation">What was tuned (a convolution's pass, a product's splits, a decoder step's tile, …).</param>
/// <param name="Candidates">The choices measured.</param>
/// <param name="Duration">The measuring, first runs and all.</param>
public readonly record struct TuningCompleted(string Device, string Operation, int Candidates, TimeSpan Duration);

/// <summary>
/// Published when an app's implementation of a slot threw, when <c>TelemetryLevel.Overrides</c> is enabled: under
/// <see cref="SlotPolicy.Throw"/> (the default) the error went on to the caller, under <see cref="SlotPolicy.FallBack"/>
/// the call was answered by the library default.
/// </summary>
/// <param name="Registry">The registry, for example "RopeScalings".</param>
/// <param name="Slot">The entry's name in it.</param>
/// <param name="Implementation">The app's implementation that failed.</param>
/// <param name="Origin">The assembly that registered it.</param>
/// <param name="Exception">What it threw.</param>
/// <param name="FellBack">True when the library default answered the call; false when the error reached the caller.</param>
/// <param name="Hint">While the error reaches the caller, how to switch on <see cref="SlotPolicy.FallBack"/> or <see cref="SlotPolicy.Shadow"/> for the slot (in code and with <c>IDRAK_OVERRIDE_POLICY</c>); null otherwise.</param>
public readonly record struct OverrideFailed(string Registry, string Slot, string Implementation, string Origin, Exception Exception, bool FellBack, string? Hint);

/// <summary>
/// Published when a call of a <see cref="SlotPolicy.Shadow"/> slot ran both the library default (which answered) and the
/// app's implementation, when <c>TelemetryLevel.Overrides</c> is enabled.
/// </summary>
/// <param name="Registry">The registry, for example "RopeScalings".</param>
/// <param name="Slot">The entry's name in it.</param>
/// <param name="Implementation">The app's implementation.</param>
/// <param name="Origin">The assembly it comes from.</param>
/// <param name="Agreed">Whether the outputs agreed and the app's implementation did not throw.</param>
/// <param name="Difference">How the outputs differ, or what the app's implementation threw; null when they agreed.</param>
/// <param name="Exception">What the app's implementation threw, or null.</param>
/// <param name="LibraryTime">The library default's time.</param>
/// <param name="OverrideTime">The app's implementation's time.</param>
/// <param name="LibraryBytes">Managed memory the library default allocated on the calling thread.</param>
/// <param name="OverrideBytes">Managed memory the app's implementation allocated on the calling thread.</param>
public readonly record struct OverrideCompared(string Registry, string Slot, string Implementation, string Origin, bool Agreed, string? Difference,
    Exception? Exception, TimeSpan LibraryTime, TimeSpan OverrideTime, long LibraryBytes, long OverrideBytes);
