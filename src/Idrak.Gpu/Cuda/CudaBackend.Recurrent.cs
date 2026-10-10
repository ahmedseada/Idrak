// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using static Idrak.Gpu.Cuda.CudaDriver;

namespace Idrak.Gpu.Cuda;

// One step of an LSTM or GRU cell and its gradient (PtxKernels.Recurrent.cs): one thread per (row, unit), one launch per
// step. A storage the caller leaves out is passed as another valid pointer with its flag clear; the kernels never touch it.
// The step kernels take the recurrent product inside: a block per row and 32 units, its warps the product's slices, as
// many as the device's reported block size allows (up to PtxKernels.MaxStepSlices; at least one a term). Sizes past
// 32-bit indices return false, and callers run the cell kernels after a product. The sequence kernels run every step of
// the step kernels in one cooperative launch, a grid barrier between steps; while a graph is captured they return false
// and callers take the step kernels.
internal sealed unsafe partial class CudaBackend
{
    public override bool LstmCellKernel(Storage projected, Storage recurrent, Storage cell, Storage hidden, Storage output, Storage? gates, Storage? cells,
        int step, int steps, int batch, int hiddenSize)
    {
        if (TooLarge(batch, steps, 4 * hiddenSize))
        {
            return false;
        }

        int flags = (gates is null ? 0 : 1) | (cells is null ? 0 : 2);
        Launch1D(K("lstm_cell_f32"), batch * hiddenSize, P(projected), P(recurrent), P(cell), P(hidden), P(output), P(gates ?? output), P(cells ?? output),
            U(step), U(steps), U(hiddenSize), U(flags), U(batch * hiddenSize));
        return true;
    }

    public override bool LstmCellBackwardKernel(Storage gates, Storage cells, Storage? dOutput, Storage dHidden, Storage dCell, Storage dGates, Storage dStep,
        int step, int previous, int steps, int batch, int hiddenSize)
    {
        if (TooLarge(batch, steps, 4 * hiddenSize))
        {
            return false;
        }

        int flags = (dOutput is null ? 0 : 1) | (previous >= 0 ? 2 : 0);
        Launch1D(K("lstm_cell_bwd_f32"), batch * hiddenSize, P(gates), P(cells), P(dOutput ?? cells), P(dHidden), P(dCell), P(dGates), P(dStep),
            U(step), U(previous), U(steps), U(hiddenSize), U(flags), U(batch * hiddenSize));
        return true;
    }

    public override bool GruCellKernel(Storage projected, Storage recurrent, Storage? hiddenBias, Storage hidden, Storage output, Storage? gates,
        int step, int steps, int batch, int hiddenSize)
    {
        if (TooLarge(batch, steps, 4 * hiddenSize))
        {
            return false;
        }

        int flags = (hiddenBias is null ? 0 : 1) | (gates is null ? 0 : 2);
        Launch1D(K("gru_cell_f32"), batch * hiddenSize, P(projected), P(recurrent), P(hiddenBias ?? recurrent), P(hidden), P(output), P(gates ?? output),
            U(step), U(steps), U(hiddenSize), U(flags), U(batch * hiddenSize));
        return true;
    }

    public override bool GruCellBackwardKernel(Storage gates, Storage output, Storage? dOutput, Storage dHidden, Storage dGates, Storage dRecurrent, Storage dStep,
        int step, int previous, int steps, int batch, int hiddenSize)
    {
        if (TooLarge(batch, steps, 4 * hiddenSize))
        {
            return false;
        }

        int flags = (dOutput is null ? 0 : 1) | (previous >= 0 ? 2 : 0);
        Launch1D(K("gru_cell_bwd_f32"), batch * hiddenSize, P(gates), P(output), P(dOutput ?? output), P(dHidden), P(dGates), P(dRecurrent), P(dStep),
            U(step), U(previous), U(steps), U(hiddenSize), U(flags), U(batch * hiddenSize));
        return true;
    }

    public override bool LstmStepKernel(Storage projected, Storage weights, Storage cell, Storage output, Storage? gates, Storage? cells, int step,
        int previous, int steps, int batch, int hiddenSize)
    {
        if (!StepRuns(batch, steps, hiddenSize, 4))
        {
            return false;
        }

        int flags = (gates is null ? 0 : 1) | (cells is null ? 0 : 2) | (previous >= 0 ? 4 : 0);
        LaunchStep("lstm_step_f32", batch, hiddenSize, P(projected), P(weights), P(cell), P(output), P(gates ?? output), P(cells ?? output), U(step),
            U(Math.Max(previous, 0)), U(steps), U(hiddenSize), U(flags));
        return true;
    }

    public override bool LstmStepBackwardKernel(Storage gates, Storage cells, Storage? dOutput, Storage weightsT, Storage dCell, Storage dGates, int step,
        int next, int previous, int steps, int batch, int hiddenSize)
    {
        if (!StepRuns(batch, steps, hiddenSize, 4))
        {
            return false;
        }

        int flags = (dOutput is null ? 0 : 1) | (previous >= 0 ? 2 : 0) | (next >= 0 ? 4 : 0);
        LaunchStep("lstm_step_bwd_f32", batch, hiddenSize, P(gates), P(cells), P(dOutput ?? cells), P(weightsT), P(dCell), P(dGates), U(step),
            U(Math.Max(next, 0)), U(Math.Max(previous, 0)), U(steps), U(hiddenSize), U(flags));
        return true;
    }

    public override bool GruStepKernel(Storage projected, Storage weights, Storage? hiddenBias, Storage output, Storage? gates, int step, int previous,
        int steps, int batch, int hiddenSize)
    {
        if (!StepRuns(batch, steps, hiddenSize, 4))
        {
            return false;
        }

        int flags = (hiddenBias is null ? 0 : 1) | (gates is null ? 0 : 2) | (previous >= 0 ? 4 : 0);
        LaunchStep("gru_step_f32", batch, hiddenSize, P(projected), P(weights), P(hiddenBias ?? weights), P(output), P(gates ?? output), U(step),
            U(Math.Max(previous, 0)), U(steps), U(hiddenSize), U(flags));
        return true;
    }

    public override bool GruStepBackwardKernel(Storage gates, Storage output, Storage? dOutput, Storage weightsT, Storage dHidden, Storage dGates,
        Storage dRecurrent, int step, int next, int previous, int steps, int batch, int hiddenSize)
    {
        if (!StepRuns(batch, steps, hiddenSize, 4))
        {
            return false;
        }

        int flags = (dOutput is null ? 0 : 1) | (previous >= 0 ? 2 : 0) | (next >= 0 ? 4 : 0);
        LaunchStep("gru_step_bwd_f32", batch, hiddenSize, P(gates), P(output), P(dOutput ?? output), P(weightsT), P(dHidden), P(dGates), P(dRecurrent),
            U(step), U(Math.Max(next, 0)), U(Math.Max(previous, 0)), U(steps), U(hiddenSize), U(flags));
        return true;
    }

    public override bool LstmSequenceKernel(Storage projected, Storage weights, Storage cell, Storage output, Storage? gates, Storage? cells, int steps,
        int batch, int hiddenSize, bool reverse)
    {
        if (!SequenceRuns("lstm_seq_f32", batch, steps, hiddenSize))
        {
            return false;
        }

        int flags = (gates is null ? 0 : 1) | (cells is null ? 0 : 2) | (reverse ? PtxKernels.SequenceReverse : 0);
        LaunchSequence("lstm_seq_f32", batch, steps, hiddenSize, P(projected), P(weights), P(cell), P(output), P(gates ?? output), P(cells ?? output), _gridBarrier,
            U(steps), U(hiddenSize), U(flags), U(batch));
        return true;
    }

    public override bool LstmSequenceBackwardKernel(Storage gates, Storage cells, Storage? dOutput, Storage weightsT, Storage dCell, Storage dGates, int steps,
        int batch, int hiddenSize, bool reverse)
    {
        if (!SequenceRuns("lstm_seq_bwd_f32", batch, steps, hiddenSize))
        {
            return false;
        }

        int flags = (dOutput is null ? 0 : 1) | (reverse ? PtxKernels.SequenceReverse : 0);
        LaunchSequence("lstm_seq_bwd_f32", batch, steps, hiddenSize, P(gates), P(cells), P(dOutput ?? cells), P(weightsT), P(dCell), P(dGates), _gridBarrier,
            U(steps), U(hiddenSize), U(flags), U(batch));
        return true;
    }

    public override bool GruSequenceKernel(Storage projected, Storage weights, Storage? hiddenBias, Storage output, Storage? gates, int steps, int batch,
        int hiddenSize, bool reverse)
    {
        if (!SequenceRuns("gru_seq_f32", batch, steps, hiddenSize))
        {
            return false;
        }

        int flags = (hiddenBias is null ? 0 : 1) | (gates is null ? 0 : 2) | (reverse ? PtxKernels.SequenceReverse : 0);
        LaunchSequence("gru_seq_f32", batch, steps, hiddenSize, P(projected), P(weights), P(hiddenBias ?? weights), P(output), P(gates ?? output), _gridBarrier,
            U(steps), U(hiddenSize), U(flags), U(batch));
        return true;
    }

    public override bool GruSequenceBackwardKernel(Storage gates, Storage output, Storage? dOutput, Storage weightsT, Storage dHidden, Storage dGates,
        Storage dRecurrent, int steps, int batch, int hiddenSize, bool reverse)
    {
        if (!SequenceRuns("gru_seq_bwd_f32", batch, steps, hiddenSize))
        {
            return false;
        }

        int flags = (dOutput is null ? 0 : 1) | (reverse ? PtxKernels.SequenceReverse : 0);
        LaunchSequence("gru_seq_bwd_f32", batch, steps, hiddenSize, P(gates), P(output), P(dOutput ?? output), P(weightsT), P(dHidden), P(dGates), P(dRecurrent),
            _gridBarrier, U(steps), U(hiddenSize), U(flags), U(batch));
        return true;
    }

    // The kernels index in 32 bits: sequences past that are left to the composed steps (batch 0 only asks, and is never too large).
    private static bool TooLarge(int batch, int steps, int width) => (long)batch * steps * width > int.MaxValue;

    // Whether a step kernel runs these sizes (batch 0 asks without launching): a warp for each of its terms in a block the
    // device allows, and every index in 32 bits (the sequences, and the weights [H, G·H]).
    private bool StepRuns(int batch, int steps, int hiddenSize, int gates) =>
        StepSlices >= 4 && !TooLarge(batch, steps, gates * hiddenSize) && (long)hiddenSize * gates * hiddenSize <= int.MaxValue;

    // The step kernels' slices: the warps of the largest block the device reports, up to PtxKernels.MaxStepSlices.
    private int StepSlices => Math.Min(PtxKernels.MaxStepSlices, _limits.MaxThreadsPerBlock / 32);

    // Whether a sequence kernel runs these sizes: the step kernel's sizes, a device that launches cooperatively (the grid
    // barrier, allocated once), no graph being captured (the steps are left to the step kernels then), and room for at
    // least one block of the step kernel's width on each multiprocessor as the device reports it.
    private bool SequenceRuns(string kernel, int batch, int steps, int hiddenSize) =>
        _gridBarrier != 0 && _captureFree is null && StepRuns(batch, steps, hiddenSize, 4) && SequenceBlocks(kernel) > 0;

    // The blocks of a sequence kernel resident at once (the occupancy the device reports for its block, times the
    // multiprocessors), asked once a kernel: the most its cooperative launch may take.
    private int SequenceBlocks(string kernel) => _sequenceBlocks.GetOrAdd(kernel, name =>
        cuOccupancyMaxActiveBlocksPerMultiprocessor(out int perMultiprocessor, K(name), 32 * StepSlices, 0) == 0
            ? perMultiprocessor * _limits.Multiprocessors
            : 0);

    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, int> _sequenceBlocks = new();

    // The sequence kernels' grid barrier: two words (arrivals, generation) the kernels leave at 0 and the generation
    // they advance; one launch at a time on the backend's stream, so one barrier serves every launch. 0 when the device
    // does not launch cooperatively (or the words could not be had).
    private ulong _gridBarrier;

    // Allocates the grid barrier once, when the device launches cooperatively (called once the kernels are loaded).
    private void SequenceBarrier()
    {
        if (_limits.CooperativeLaunch && cuMemAlloc(out ulong barrier, 8) == 0)
        {
            Check(cuMemsetD32(barrier, 0, 2), nameof(cuMemsetD32));
            _gridBarrier = barrier;
        }
    }

    // A sequence kernel over `batch` rows of H units: every block resident (cooperative), as many as the items of a step
    // (batch · ⌈H / 32⌉) up to what the device holds at once, each the step kernel's block (nothing to launch for no rows,
    // steps or units).
    private void LaunchSequence(string kernel, int batch, int steps, int hiddenSize, params ReadOnlySpan<ulong> args)
    {
        if (batch > 0 && steps > 0 && hiddenSize > 0)
        {
            int items = batch * ((hiddenSize + 31) / 32);
            t_cooperative = true;
            Launch(K(kernel), (uint)Math.Min(items, SequenceBlocks(kernel)), 1, (uint)(32 * StepSlices), 1, args);
        }
    }

    // A step kernel over `batch` rows of H units: grid x = batch · ⌈H / 32⌉, a block of StepSlices warps (nothing to launch
    // for batch 0).
    private void LaunchStep(string kernel, int batch, int hiddenSize, params ReadOnlySpan<ulong> args)
    {
        if (batch > 0 && hiddenSize > 0)
        {
            Launch(K(kernel), (uint)(batch * ((hiddenSize + 31) / 32)), 1, (uint)(32 * StepSlices), 1, args);
        }
    }
}
