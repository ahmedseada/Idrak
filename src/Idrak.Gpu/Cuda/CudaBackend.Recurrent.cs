// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Cuda;

// One step of an LSTM or GRU cell and its gradient (PtxKernels.Recurrent.cs): one thread per (row, unit), one launch per
// step. A storage the caller leaves out is passed as another valid pointer with its flag clear; the kernels never touch it.
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

    // The kernels index in 32 bits: sequences past that are left to the composed steps (batch 0 only asks, and is never too large).
    private static bool TooLarge(int batch, int steps, int width) => (long)batch * steps * width > int.MaxValue;
}
