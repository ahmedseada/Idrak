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

    /// <summary>
    /// The prompt's pass, to its first token read back: Count the prompt positions fed (every row's, padding included, for
    /// rows decoded together), Pieces and PieceSize the pieces it went through.
    /// </summary>
    Prefill,

    /// <summary>Decoding steps since the last chunk, to their tokens read back: Count the tokens (every row's).</summary>
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
/// Published at <c>TelemetryLevel.Inference</c> and <c>TelemetryLevel.Stages</c> for each stage of reading an answer (a
/// read of a page: its image decoded and transformed, encoded, the prompt's pass, each chunk of decoding steps, a step
/// recorded as a graph, each cache allocated), with its sizes. Durations are host time: device work still queued at the end of a stage counts in the next
/// one unless <c>Telemetry.SynchronizeForTiming</c> waits for it (the prefill and decode chunks end with their tokens read
/// back, so they are complete).
/// </summary>
/// <param name="Stage">Which stage.</param>
/// <param name="Device">The kind of device it ran on ("cpu", "cuda", …).</param>
/// <param name="Duration">How long it took.</param>
/// <param name="Count">What the stage counts (<see cref="InferenceStageKind"/> says what): images, tokens, positions, values.</param>
/// <param name="Bytes">The bytes it made or read (<see cref="InferenceStageKind"/> says which), or 0.</param>
/// <param name="Rows">The rows (prompts) read together: 1 for one prompt, the batch's rows for prompts decoded together.</param>
/// <param name="Pieces">The passes the stage ran in: a long prompt's pass goes through the cache in pieces (<c>GenerationOptions.PrefillChunk</c>); 1 otherwise.</param>
/// <param name="PieceSize">The positions of each piece of a prompt's pass (the last can be shorter), or 0 where the stage has no pieces.</param>
public readonly record struct InferenceStage(InferenceStageKind Stage, string Device, TimeSpan Duration, long Count, long Bytes, int Rows = 1, int Pieces = 1, int PieceSize = 0);

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
/// <param name="KernelsUnavailable">Why the device's kernels are unavailable as a whole (<see cref="Devices.IKernelAvailability"/>:
/// a missing run-time compiler, a setting), or null when the device runs kernels and has none of its own for this one.</param>
public readonly record struct HostFallback(string Device, string Operation, string? KernelsUnavailable = null);

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
/// <param name="Nested">Measured inside another tuning's candidate run (a choice that run needed itself): its time lies within the outer one's.</param>
public readonly record struct TuningCompleted(string Device, string Operation, int Candidates, TimeSpan Duration, bool Nested);

/// <summary>
/// A device loaded its set of kernels (<c>TelemetryLevel.Devices</c>), once per device and process: from the kernel cache
/// (warm: the code read and handed to the driver) or made first (cold: generated, assembled or compiled, then loaded and
/// kept in the cache), with the time it took and the code's size, so the cold start shows apart from the warm one
/// (rule 76) on any device whose kernels are loaded as a set.
/// </summary>
/// <param name="Device">The device ("hip:0").</param>
/// <param name="Backend">The kind of device ("hip", …).</param>
/// <param name="Kernels">The kernels in the set.</param>
/// <param name="FromCache">True when the code came from the kernel cache (warm); false when it was made in this call (cold).</param>
/// <param name="Duration">Making or reading the code, and loading it.</param>
/// <param name="Bytes">The code's size in bytes (the code object the driver loaded).</param>
/// <param name="Detail">What the code was made for (the target and the code object format), or null.</param>
public readonly record struct KernelsLoaded(string Device, string Backend, int Kernels, bool FromCache, TimeSpan Duration, long Bytes, string? Detail = null);

/// <summary>
/// A device described for a listing (<c>Diagnostics.DeviceListing</c>, <c>TelemetryLevel.Devices</c>): its backend
/// started in this call (the first listing of the device in the process: the driver loaded, the device created, its
/// kernels' width measured: the cold time) or was running already (warm), and how long starting and reading what it
/// reports took, so a slow device list shows where its time went, cold apart from warm (rule 76).
/// </summary>
/// <param name="Device">The device's name ("cpu", "vulkan:0").</param>
/// <param name="Start">Starting its backend (or reaching it, once started).</param>
/// <param name="Describe">Reading what it reports (its name, memory and facts).</param>
/// <param name="Cold">The first listing of the device in this process.</param>
/// <param name="Error">Why it could not start, or null.</param>
public readonly record struct DeviceListed(string Device, TimeSpan Start, TimeSpan Describe, bool Cold, string? Error);

/// <summary>
/// A device that records graphs could not record a piece of work (<c>TelemetryLevel.Devices</c>): the work runs as
/// ordinary steps instead, which costs launches on every step and runs again whatever the recording did before it failed
/// (a training step's image features are taken from the cache once more). The reason is the error that stopped the
/// recording: a read of device memory (a host fallback inside the step, or a value read back), memory, or a call the
/// device cannot record.
/// </summary>
/// <param name="Device">The kind of device ("cuda", "vulkan", "hip", …).</param>
/// <param name="Work">What was being recorded ("a decoding step", "a training step").</param>
/// <param name="Reason">Why the recording failed.</param>
public readonly record struct GraphRecordingFailed(string Device, string Work, string Reason);

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

/// <summary>
/// Published at <c>TelemetryLevel.Network</c> when a transfer between two ranks completed (plan 23): its bytes, the time it
/// took here, and the time the simulator's virtual clock charged for it (zero on a real transport).
/// </summary>
/// <param name="Transport">The transport that carried it ("sim", "tcp", …).</param>
/// <param name="From">The sending rank.</param>
/// <param name="To">The receiving rank.</param>
/// <param name="Tag">The tag it was sent under.</param>
/// <param name="Bytes">The bytes carried.</param>
/// <param name="Wall">The time it took here, measured after the device's queued work (a copy is not counted as free).</param>
/// <param name="Simulated">The time the link profile charged on the virtual clock, or zero.</param>
public readonly record struct TransferCompleted(string Transport, int From, int To, int Tag, long Bytes, TimeSpan Wall, TimeSpan Simulated);

/// <summary>
/// Published at <c>TelemetryLevel.Network</c> when a collective operation completed on a rank: what it moved, how long it
/// took, how long the rank waited on the others, and which rank was slowest.
/// </summary>
/// <param name="Collective">The algorithm ("ring", "hierarchical", …).</param>
/// <param name="Operation">The operation ("all-reduce", "all-gather", "reduce-scatter", "broadcast", "barrier").</param>
/// <param name="Rank">The rank reporting.</param>
/// <param name="WorldSize">The number of ranks.</param>
/// <param name="BytesSent">The bytes this rank sent.</param>
/// <param name="BytesReceived">The bytes this rank received.</param>
/// <param name="Wall">The time it took here.</param>
/// <param name="Simulated">The time on the virtual clock, or zero on a real transport.</param>
/// <param name="Blocked">
/// The time this rank waited in its receives for the others; on a simulated transport, on the virtual clock (the
/// simulated cluster's wait), else wall time.
/// </param>
/// <param name="SlowestRank">
/// The rank the others waited on longest; on a simulated transport, the origin of the critical path that last held this
/// rank's virtual clock back (<see cref="Cluster.ITransport.SimulatedCriticalRank"/>), which names a slow worker on every rank.
/// </param>
/// <param name="Staging">
/// Where a call on device buffers moved its values (plan 23, phase C6): "device" when they stayed on the device and were
/// summed there (a transport that moves device memory, "pcie-host" or "nvlink" in one process), "host" when they were
/// staged through host memory; null for values the caller held on the host already.
/// </param>
public readonly record struct CollectiveCompleted(string Collective, string Operation, int Rank, int WorldSize, long BytesSent, long BytesReceived,
    TimeSpan Wall, TimeSpan Simulated, TimeSpan Blocked, int SlowestRank, string? Staging = null);

/// <summary>
/// Published at <c>TelemetryLevel.Network</c> by a gradient synchronisation once per training step on each rank (plan 23,
/// phase C4): how the step's gradients (or, with local SGD, weights) went across the ranks, how much of it ran while the
/// backward pass went on, and how much of the network time the step still waited for after it.
/// </summary>
/// <param name="Sync">The synchronisation ("all-reduce", "local-sgd", …).</param>
/// <param name="Compression">The compression of its buckets ("none", "bf16", "top-k", …).</param>
/// <param name="Rank">The rank reporting, in the step's group.</param>
/// <param name="WorldSize">The ranks of the step's group.</param>
/// <param name="Step">The training step (as the trainer began it), or -1 when the trainer began none.</param>
/// <param name="Buckets">The buckets reduced (0 on a local SGD step that sent no weights).</param>
/// <param name="Overlapped">The buckets sent while the backward pass ran.</param>
/// <param name="BucketBytes">The bytes of a bucket (set, or measured on the first step).</param>
/// <param name="Bytes">The bytes of gradient (or weight) values reduced this step, before compression.</param>
/// <param name="Backward">The backward pass's time, from the overlap's start to the reduction's (zero without one).</param>
/// <param name="Network">The reduction's time: on a simulated transport the virtual clock's, else the wall time.</param>
/// <param name="Exposed">
/// The network time the step waited for after its backward pass: on a simulated transport predicted from the backward's
/// measured time and each bucket's simulated time, else measured.
/// </param>
/// <param name="Lost">Why the step's buckets were not sent during its backward pass (a recorded step replayed, overlap off, …), or null.</param>
public readonly record struct GradientsSynced(string Sync, string Compression, int Rank, int WorldSize, long Step, int Buckets, int Overlapped, long BucketBytes,
    long Bytes, TimeSpan Backward, TimeSpan Network, TimeSpan Exposed, string? Lost);

/// <summary>
/// Published at <c>TelemetryLevel.Network</c> by each rank of a pipeline (plan 23, phase C7) once per training step: its
/// stage, the micro-batches, how long the step took and how long the stage waited for its neighbours (the bubble).
/// </summary>
/// <param name="Schedule">The schedule ("gpipe").</param>
/// <param name="Rank">The rank reporting.</param>
/// <param name="Stage">Its stage, from 0.</param>
/// <param name="Stages">The pipeline's stages.</param>
/// <param name="Replica">Its pipeline, when several run side by side as data-parallel replicas; else 0.</param>
/// <param name="MicroBatches">The step's micro-batches.</param>
/// <param name="Step">The training step, or -1 when the caller gave none.</param>
/// <param name="Wall">The step's time on this rank, from its first forward to its last message.</param>
/// <param name="Blocked">The part of it spent waiting for an activation or a gradient from a neighbouring stage.</param>
/// <param name="Bubble">Blocked over Wall: the share of the step the stage stood idle.</param>
/// <param name="IdealBubble">The schedule's own arithmetic for stages of equal cost: (stages - 1) / (micro-batches + stages - 1) for GPipe.</param>
/// <param name="BytesSent">The activation and gradient bytes this stage sent.</param>
/// <param name="BytesReceived">The bytes it received.</param>
/// <param name="Staging">
/// How the activations crossed: "device" (handed device to device over a transport that moves device memory, plan 23
/// C8b), "host" (through host memory), or null when this stage sent and received none.
/// </param>
public readonly record struct PipelineStepped(string Schedule, int Rank, int Stage, int Stages, int Replica, int MicroBatches, long Step, TimeSpan Wall,
    TimeSpan Blocked, double Bubble, double IdealBubble, long BytesSent, long BytesReceived, string? Staging = null);

/// <summary>Published at <c>TelemetryLevel.Network</c> once per pair of ranks when a group starts: the transport joining them, and why.</summary>
/// <param name="RankA">One rank.</param>
/// <param name="RankB">The other.</param>
/// <param name="Transport">The transport chosen.</param>
/// <param name="Reason">Why (the selector's rule, or "forced by the options").</param>
public readonly record struct TransportChosen(int RankA, int RankB, string Transport, string Reason);

/// <summary>
/// Published at <c>TelemetryLevel.Network</c> once per rank after rank 0 as a cluster run starts (<c>ClusterOptions.ShareTuning</c>):
/// whether the rank runs rank 0's measured kernel choices (<see cref="Devices.MeasuredChoiceProviders"/>) or keeps its own, and why.
/// </summary>
/// <param name="Rank">The rank.</param>
/// <param name="Device">Its device.</param>
/// <param name="Taken">Whether it runs rank 0's choices; false: it keeps its own (another device, driver or kind than rank 0's).</param>
/// <param name="Choices">How many of rank 0's choices it took (those known then).</param>
/// <param name="Detail">What happened, or why not.</param>
public readonly record struct MeasuredChoicesShared(int Rank, string Device, bool Taken, int Choices, string Detail);

/// <summary>How a rank stands in its group.</summary>
public enum RankState
{
    /// <summary>It connected.</summary>
    Joined,

    /// <summary>The others wait on it noticeably.</summary>
    Slow,

    /// <summary>It stopped answering.</summary>
    Dropped,

    /// <summary>Some of its links are closed.</summary>
    Partitioned,

    /// <summary>It came back.</summary>
    Rejoined,

    /// <summary>It finished its run.</summary>
    Finished,
}

/// <summary>Published at <c>TelemetryLevel.Network</c> when a rank's state changes.</summary>
/// <param name="Rank">The rank.</param>
/// <param name="State">Its new state.</param>
/// <param name="Step">The training step, or -1 outside one.</param>
/// <param name="Detail">More, or null.</param>
/// <param name="Seed">The seed of the simulation whose fault changed the state (the replay key with <paramref name="Step"/>), or null outside one.</param>
public readonly record struct RankStateChanged(int Rank, RankState State, long Step, string? Detail, int? Seed = null);

/// <summary>
/// Published at <c>TelemetryLevel.Network</c> when a simulation injected a fault: the seed and step are the replay key
/// (<c>idrak cluster replay --seed S --step N</c>).
/// </summary>
/// <param name="Seed">The run's seed.</param>
/// <param name="Step">The step it starts at.</param>
/// <param name="Kind">What goes wrong.</param>
/// <param name="Ranks">The ranks it touches.</param>
/// <param name="Duration">How long it lasts, or zero for the rest of the run.</param>
/// <param name="Factor">The kind's figure (a slowdown, a loss share, a number of steps late).</param>
public readonly record struct FaultInjected(int Seed, long Step, Cluster.FaultKind Kind, IReadOnlyList<int> Ranks, TimeSpan Duration, double Factor);

/// <summary>Published at <c>TelemetryLevel.Network</c> when a calibration corrected a link profile from measured transfers.</summary>
/// <param name="Link">The profile's name.</param>
/// <param name="BandwidthBefore">Its bandwidth before, bytes a second.</param>
/// <param name="BandwidthAfter">Its bandwidth after.</param>
/// <param name="LatencyBefore">Its latency before.</param>
/// <param name="LatencyAfter">Its latency after.</param>
/// <param name="Residual">The fit's root-mean-square error, seconds.</param>
/// <param name="Samples">The transfers the fit used.</param>
public readonly record struct ProfileCalibrated(string Link, double BandwidthBefore, double BandwidthAfter, TimeSpan LatencyBefore, TimeSpan LatencyAfter,
    double Residual, int Samples);
