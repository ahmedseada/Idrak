// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices;

// The optional capabilities of a device (plan 33), each an interface a backend implements or not. Backend reads which
// of them a device implements once, when it starts (rule 75): its operations then reach the device's kernels through
// that interface, and the questions callers ask (Backend.SupportsGraphs, Backend.FusedKernels, Backend.MatrixUnits)
// follow from it, so no flag has to agree with members that throw or return false by default.

/// <summary>
/// Recording work as a graph and replaying it (<c>ComputeGraph</c>): between <see cref="BeginCapture"/> and
/// <see cref="EndCapture"/> the device records what the recording thread queues instead of running it.
/// <see cref="Backend.SupportsGraphs"/> is true for a device that implements it (and has <see cref="CaptureEnabled"/>).
/// </summary>
public interface IGraphCaptureBackend
{
    /// <summary>
    /// Whether recording is on now: true unless a setting of the device turns it off for a while (a diagnostic run that
    /// checks every kernel's output, which a recorded kernel has not produced yet). Default: true.
    /// </summary>
    bool CaptureEnabled => true;

    /// <summary>Starts recording all subsequent work on this device instead of running it.</summary>
    void BeginCapture();

    /// <summary>Stops recording and returns a replayable graph plus the device blocks it owns.</summary>
    (IntPtr Executable, IntPtr Graph, List<Storage> Owned) EndCapture();

    /// <summary>Stops recording after a failure, discarding the partial graph; returns the blocks it owned.</summary>
    List<Storage> AbortCapture();

    /// <summary>Runs a graph returned by <see cref="EndCapture"/> again (its storages hold the inputs of the run).</summary>
    void ReplayGraph(IntPtr executable);

    /// <summary>Frees a graph returned by <see cref="EndCapture"/>.</summary>
    void DestroyGraph(IntPtr executable, IntPtr graph);
}

/// <summary>Timing every kernel a device runs (<c>Idrak.Diagnostics.GpuProfiler</c>).</summary>
public interface IKernelProfiler
{
    /// <summary>Starts timing every kernel.</summary>
    void StartProfile();

    /// <summary>Stops timing; the kernels by name with their calls, time (stopwatch ticks) and floating-point operations.</summary>
    Dictionary<string, (long Calls, long Ticks, double Flops)> StopProfile();
}

/// <summary>
/// Attention for rows of different lengths decoded together, each row from its own start (<see cref="Backend.AttentionRows"/>),
/// run by the device's own kernels. A device without it says what the segmented attention says
/// (<see cref="Backend.SupportsSegmentedAttention"/>).
/// </summary>
public interface IRowAttentionBackend
{
    /// <summary>
    /// Whether <see cref="AttentionRows"/> runs on this device for head size <paramref name="dim"/> and
    /// <paramref name="variant"/> (with the current <c>MixedPrecision</c>), so callers batch such rows instead of running
    /// them apart.
    /// </summary>
    bool SupportsRows(int dim, AttentionVariant variant);

    /// <inheritdoc cref="Backend.AttentionRowsKernel"/>
    bool AttentionRows(Storage q, Storage keys, Storage values, Storage position, Storage y, Storage starts, int heads, int headsPerRow, int rowsPerHead,
        int steps, int capacity, int dim, float scale, AttentionVariant variant);
}

/// <summary>
/// Causal attention over packed sequences and its gradient (<see cref="Backend.AttentionSegmented"/>,
/// <see cref="Backend.AttentionSegmentedBackward"/>), run by the device's own kernels. A device without it has no such pass.
/// </summary>
public interface ISegmentedAttentionBackend
{
    /// <summary>
    /// Whether <see cref="AttentionSegmented"/> and its gradient run on this device for head size <paramref name="dim"/>
    /// and <paramref name="variant"/> (with the current <c>MixedPrecision</c>).
    /// </summary>
    bool SupportsSegmented(int dim, AttentionVariant variant);

    /// <inheritdoc cref="Backend.AttentionSegmentedKernel"/>
    bool AttentionSegmented(Storage q, Storage keys, Storage values, Storage y, Storage? logSumExp, Storage starts, Storage ends, int heads,
        int headsPerRow, int rowsPerHead, int steps, int dim, float scale, AttentionVariant variant);

    /// <inheritdoc cref="Backend.AttentionSegmentedBackwardKernel"/>
    bool AttentionSegmentedBackward(Storage q, Storage keys, Storage values, Storage output, Storage logSumExp, Storage dOutput, Storage dq, Storage dkeys,
        Storage dvalues, Storage starts, Storage ends, int heads, int headsPerRow, int rowsPerHead, int steps, int dim, float scale, AttentionVariant variant);
}

/// <summary>
/// Matrix units (tensor cores) the <c>MixedPrecision</c> modes run on (<see cref="Backend.MatrixUnits"/>): the fused
/// training products, the LoRA products and the strided attention for training.
/// Whether this device's units are usable now is read from the device (<see cref="TensorCoresUnavailable"/>).
/// </summary>
public interface IMatrixUnitBackend
{
    /// <summary>Null when this device runs bfloat16 products on its matrix units; otherwise why not (as the device reports it).</summary>
    string? TensorCoresUnavailable();

    /// <summary>Whether head size <paramref name="dim"/> has matrix-unit flash attention for training.</summary>
    bool AttentionHeadDim(int dim);

    /// <inheritdoc cref="Backend.MatMulLowRankKernel"/>
    bool MatMulLowRank(Storage a, Storage b, Storage c, int m, int n, int k, bool transB, float beta, Storage u, Storage v, int rank);

    /// <inheritdoc cref="Backend.GemmStridedKernel"/>
    bool GemmStrided(Storage a, long aOffset, int lda, bool transA, Storage b, long bOffset, int ldb, bool transB, Storage c, long cOffset, int ldc, int m,
        int n, int k, float beta, Storage? bias, GemmEpilogue epilogue, Storage? aux, long auxOffset);

    /// <inheritdoc cref="Backend.AttentionStridedKernel"/>
    bool AttentionStrided(Storage q, long qOffset, Storage k, long kOffset, Storage v, long vOffset, int qRow, int kRow, Storage y, Storage? logSumExp,
        int batch, int kvHeads, int group, int steps, int dim, float scale);

    /// <inheritdoc cref="Backend.AttentionStridedBackwardKernel"/>
    bool AttentionStridedBackward(Storage q, long qOffset, Storage k, long kOffset, Storage v, long vOffset, int qRow, int kRow, Storage y,
        Storage logSumExp, Storage dOutput, Storage dq, long dqOffset, Storage dk, long dkOffset, Storage dv, long dvOffset, int batch, int kvHeads,
        int group, int steps, int dim, float scale);
}

/// <summary>
/// The recurrent layers' kernels (<see cref="Backend.LstmCell"/> and the rest): one time step of a cell, a step with its
/// recurrent product inside, and a whole sequence in one call, with their gradients. A kernel returns false for sizes it
/// has no pass for; without the interface every step is composed from products and element-wise operations.
/// </summary>
public interface IRecurrentKernels
{
    /// <inheritdoc cref="Backend.LstmCellKernel"/>
    bool LstmCell(Storage projected, Storage recurrent, Storage cell, Storage hidden, Storage output, Storage? gates, Storage? cells, int step, int steps,
        int batch, int hiddenSize);

    /// <inheritdoc cref="Backend.LstmCellBackwardKernel"/>
    bool LstmCellBackward(Storage gates, Storage cells, Storage? dOutput, Storage dHidden, Storage dCell, Storage dGates, Storage dStep, int step,
        int previous, int steps, int batch, int hiddenSize);

    /// <inheritdoc cref="Backend.GruCellKernel"/>
    bool GruCell(Storage projected, Storage recurrent, Storage? hiddenBias, Storage hidden, Storage output, Storage? gates, int step, int steps, int batch,
        int hiddenSize);

    /// <inheritdoc cref="Backend.GruCellBackwardKernel"/>
    bool GruCellBackward(Storage gates, Storage output, Storage? dOutput, Storage dHidden, Storage dGates, Storage dRecurrent, Storage dStep, int step,
        int previous, int steps, int batch, int hiddenSize);

    /// <inheritdoc cref="Backend.LstmStepKernel"/>
    bool LstmStep(Storage projected, Storage weights, Storage cell, Storage output, Storage? gates, Storage? cells, int step, int previous, int steps,
        int batch, int hiddenSize);

    /// <inheritdoc cref="Backend.LstmStepBackwardKernel"/>
    bool LstmStepBackward(Storage gates, Storage cells, Storage? dOutput, Storage weightsT, Storage dCell, Storage dGates, int step, int next,
        int previous, int steps, int batch, int hiddenSize);

    /// <inheritdoc cref="Backend.GruStepKernel"/>
    bool GruStep(Storage projected, Storage weights, Storage? hiddenBias, Storage output, Storage? gates, int step, int previous, int steps, int batch,
        int hiddenSize);

    /// <inheritdoc cref="Backend.GruStepBackwardKernel"/>
    bool GruStepBackward(Storage gates, Storage output, Storage? dOutput, Storage weightsT, Storage dHidden, Storage dGates, Storage dRecurrent, int step,
        int next, int previous, int steps, int batch, int hiddenSize);

    /// <inheritdoc cref="Backend.LstmSequenceKernel"/>
    bool LstmSequence(Storage projected, Storage weights, Storage cell, Storage output, Storage? gates, Storage? cells, int steps, int batch,
        int hiddenSize, bool reverse);

    /// <inheritdoc cref="Backend.LstmSequenceBackwardKernel"/>
    bool LstmSequenceBackward(Storage gates, Storage cells, Storage? dOutput, Storage weightsT, Storage dCell, Storage dGates, int steps, int batch,
        int hiddenSize, bool reverse);

    /// <inheritdoc cref="Backend.GruSequenceKernel"/>
    bool GruSequence(Storage projected, Storage weights, Storage? hiddenBias, Storage output, Storage? gates, int steps, int batch, int hiddenSize,
        bool reverse);

    /// <inheritdoc cref="Backend.GruSequenceBackwardKernel"/>
    bool GruSequenceBackward(Storage gates, Storage output, Storage? dOutput, Storage weightsT, Storage dHidden, Storage dGates, Storage dRecurrent,
        int steps, int batch, int hiddenSize, bool reverse);
}

/// <summary>
/// Keeping tensors in system memory when the device is full (<see cref="Backend.Offload"/>). A device that implements it
/// moves live storages between memories, so <see cref="Backend"/> runs every operation of it inside an operation scope
/// from the start (a move waits until no other thread is inside one); a device without it pays nothing for that.
/// </summary>
public interface IOffloadingBackend
{
    /// <summary>The device's offloading (<see cref="IMemoryOffload"/>).</summary>
    IMemoryOffload Offload { get; }
}

/// <summary>
/// Copies from this device straight into another device's memory (<see cref="Backend.CopyTo"/>): another backend of the
/// same device, a peer device. A device without it copies inside itself, and to any other device through host memory.
/// </summary>
public interface IPeerCopyBackend
{
    /// <summary>
    /// Whether <see cref="CopyTo"/> reaches <paramref name="destination"/>'s memory (a backend other than this one) without
    /// a trip through host memory.
    /// </summary>
    bool CopiesDirectlyTo(Backend destination);

    /// <summary>
    /// Copies the first <paramref name="count"/> values (more than 0) of <paramref name="source"/>, on this backend, into
    /// <paramref name="destination"/>, on another backend <see cref="CopiesDirectlyTo"/> reaches; <see cref="Backend.CopyTo"/>
    /// calls it with the arguments checked. The copy has finished when the call returns.
    /// </summary>
    void CopyTo(Storage source, Storage destination, int count);
}

/// <summary>
/// A backend whose own kernels can be unavailable as a whole (compiled at run time by a compiler that may be missing, refused
/// by the driver, turned off by a setting): why, or null while they run. Every operation then takes the host fallback (each
/// call copying its operands to the CPU and back), and device listings, diagnostics and <see cref="Diagnostics.HostFallback"/>
/// name the reason.
/// </summary>
public interface IKernelAvailability
{
    /// <summary>Why the backend's kernels are unavailable (loading them first when they were not tried yet), or null when they run.</summary>
    string? KernelsUnavailable();
}

/// <summary>What a device reports about its memory beyond <see cref="Backend.GetMemoryUsage"/> (<see cref="Backend.AvailableMemory"/>, <see cref="Backend.FreeMemory"/>).</summary>
public interface IMemoryReport
{
    /// <summary>
    /// Bytes of device memory new tensors can take now, as the device reports it (its free memory and the blocks its pool
    /// keeps for reuse, within <see cref="ComputeResources.GpuMemoryLimit"/>), or null when the device cannot tell now.
    /// </summary>
    long? AvailableMemory();

    /// <summary>
    /// Bytes of device memory the device's driver reports free now (outside every pool, before any reserve), or null when
    /// the driver reports none.
    /// </summary>
    long? FreeMemory();
}

/// <summary>A device that places long-lived allocations (weights, attention caches) together (<see cref="Backend.LongLivedAllocations"/>).</summary>
public interface ILongLivedAllocations
{
    /// <summary>
    /// Until the result is disposed, this thread's allocations on the device are long-lived: the device may place them
    /// together (CUDA: blocks the driver would round up to its page are taken from shared chunks,
    /// <see cref="ComputeResources.GpuChunkBytes"/>, so the rounding is paid once a chunk). Scopes nest.
    /// </summary>
    IDisposable LongLivedAllocations();
}
