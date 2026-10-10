// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Cuda;

// One step of an LSTM or GRU cell and its gradient (PtxKernels.Recurrent.cs): one thread per (row, unit), one launch per
// step. A storage the caller leaves out is passed as another valid pointer with its flag clear; the kernels never touch it.
// The step kernels take the recurrent product inside: a block per row and 32 units, its warps the product's slices, as
// many as the device's reported block size allows (up to PtxKernels.MaxStepSlices; at least one a term). Sizes past
// 32-bit indices return false, and callers run the cell kernels after a product.
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

    // The kernels index in 32 bits: sequences past that are left to the composed steps (batch 0 only asks, and is never too large).
    private static bool TooLarge(int batch, int steps, int width) => (long)batch * steps * width > int.MaxValue;

    // Whether a step kernel runs these sizes (batch 0 asks without launching): a warp for each of its terms in a block the
    // device allows, and every index in 32 bits (the sequences, and the weights [H, G·H]).
    private bool StepRuns(int batch, int steps, int hiddenSize, int gates) =>
        StepSlices >= 4 && !TooLarge(batch, steps, gates * hiddenSize) && (long)hiddenSize * gates * hiddenSize <= int.MaxValue;

    // The step kernels' slices: the warps of the largest block the device reports, up to PtxKernels.MaxStepSlices.
    private int StepSlices => Math.Min(PtxKernels.MaxStepSlices, _limits.MaxThreadsPerBlock / 32);

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
