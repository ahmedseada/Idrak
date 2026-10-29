// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Abstraction.Devices;

namespace Idrak.Gpu.Hip;

// One step of an LSTM or GRU cell and its gradient on HIP's GCN kernels (GcnKernels.Recurrent.cs), as CUDA implements
// IRecurrentKernels: the cell kernels a work-item a (row, unit), one launch a step; the step kernels with the recurrent
// product inside, a block a (row, 64 units), one launch a step. A storage the caller leaves out is passed as another valid
// pointer with its flag clear; the kernels never touch it. Sizes past the kernels' 32-bit byte offsets, a grid past the
// device's, or unavailable kernels return false, and the callers compose the step (the cell kernels after a product, or
// products and element-wise operations). The sequence kernels (CUDA's cooperative launch with a grid barrier) return
// false: the layers run the step kernels a step at a time. Each launch is labelled for --profile (RecurrentCost).
internal sealed unsafe partial class HipBackend : IRecurrentKernels
{
    /// <summary>The cost of a recurrent kernel per (row, unit) of a step, for hidden size <paramref name="hidden"/> (the step kernels' products).</summary>
    internal static (double Flops, double Bytes) RecurrentCost(string kernel, int hidden) => kernel switch
    {
        "lstm_cell_f32" => (150, 64),
        "lstm_cell_bwd_f32" => (60, 72),
        "gru_cell_f32" => (110, 56),
        "gru_cell_bwd_f32" => (20, 68),
        "lstm_step_f32" => (150 + 8.0 * hidden, 48 + 16.0 * hidden),
        "lstm_step_bwd_f32" => (60 + 8.0 * hidden, 56 + 16.0 * hidden),
        "gru_step_f32" => (110 + 6.0 * hidden, 40 + 12.0 * hidden),
        "gru_step_bwd_f32" => (20 + 6.0 * hidden, 52 + 12.0 * hidden),
        _ => (0, 0),
    };

    // A cell kernel for `batch` rows of H units of a sequence of `steps` steps and rows `gates` · H wide, or null: the
    // kernels unavailable or an index past 32 bits (byte offsets).
    private GcnFunction? CellKernel(string name, int batch, int steps, int hiddenSize, int gates) =>
        KernelSizes([(long)batch * steps * gates * hiddenSize, (long)batch * hiddenSize * gates], [hiddenSize, steps]) && batch > 0 ? Gcn(name) : null;

    // A step kernel for these sizes, or null: also the weights within the byte offsets and the blocks within the grid.
    private GcnFunction? StepKernel(string name, int batch, int steps, int hiddenSize) =>
        batch > 0 && KernelSizes([(long)batch * steps * 4 * hiddenSize, (long)hiddenSize * 4 * hiddenSize], [hiddenSize, steps])
        && StepBlocks(batch, hiddenSize) <= DeviceLimits.MaxGridDimX ? Gcn(name) : null;

    private static long StepBlocks(int batch, int hiddenSize) => (long)batch * ((hiddenSize + GcnKernels.StepUnits - 1) / GcnKernels.StepUnits);

    private void LaunchCell(GcnFunction f, int batch, int hiddenSize, params ReadOnlySpan<ulong> values)
    {
        int n = batch * hiddenSize;
        if (Profiling)
        {
            LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"{f.Name} {batch}x{hiddenSize}"), RecurrentCost(f.Name, hiddenSize).Flops * n);
        }

        LaunchKernel(f, PointwiseBlocks(n), 1, values);
    }

    private void LaunchStep(GcnFunction f, int batch, int hiddenSize, params ReadOnlySpan<ulong> values)
    {
        if (Profiling)
        {
            LabelNextLaunch(string.Create(CultureInfo.InvariantCulture, $"{f.Name} {batch}x{hiddenSize}"), RecurrentCost(f.Name, hiddenSize).Flops * batch * hiddenSize);
        }

        LaunchKernel(f, (uint)StepBlocks(batch, hiddenSize), 1, values);
    }

    bool IRecurrentKernels.LstmCell(Storage projected, Storage recurrent, Storage cell, Storage hidden, Storage output, Storage? gates, Storage? cells,
        int step, int steps, int batch, int hiddenSize)
    {
        if (batch == 0 || hiddenSize == 0)
        {
            return true;
        }

        if (CellKernel("lstm_cell_f32", batch, steps, hiddenSize, 4) is not { } f)
        {
            return false;
        }

        int flags = (gates is null ? 0 : 1) | (cells is null ? 0 : 2);
        LaunchCell(f, batch, hiddenSize, P(projected), P(recurrent), P(cell), P(hidden), P(output), P(gates ?? output), P(cells ?? output), U(step), U(steps),
            U(hiddenSize), U(flags), U(batch * hiddenSize));
        return true;
    }

    bool IRecurrentKernels.LstmCellBackward(Storage gates, Storage cells, Storage? dOutput, Storage dHidden, Storage dCell, Storage dGates, Storage dStep,
        int step, int previous, int steps, int batch, int hiddenSize)
    {
        if (batch == 0 || hiddenSize == 0)
        {
            return true;
        }

        if (CellKernel("lstm_cell_bwd_f32", batch, steps, hiddenSize, 4) is not { } f)
        {
            return false;
        }

        int flags = (dOutput is null ? 0 : 1) | (previous >= 0 ? 2 : 0);
        LaunchCell(f, batch, hiddenSize, P(gates), P(cells), P(dOutput ?? cells), P(dHidden), P(dCell), P(dGates), P(dStep), U(step), U(Math.Max(previous, 0)),
            U(steps), U(hiddenSize), U(flags), U(batch * hiddenSize));
        return true;
    }

    bool IRecurrentKernels.GruCell(Storage projected, Storage recurrent, Storage? hiddenBias, Storage hidden, Storage output, Storage? gates,
        int step, int steps, int batch, int hiddenSize)
    {
        if (batch == 0 || hiddenSize == 0)
        {
            return true;
        }

        if (CellKernel("gru_cell_f32", batch, steps, hiddenSize, 4) is not { } f)
        {
            return false;
        }

        int flags = (hiddenBias is null ? 0 : 1) | (gates is null ? 0 : 2);
        LaunchCell(f, batch, hiddenSize, P(projected), P(recurrent), P(hiddenBias ?? recurrent), P(hidden), P(output), P(gates ?? output), U(step), U(steps),
            U(hiddenSize), U(flags), U(batch * hiddenSize));
        return true;
    }

    bool IRecurrentKernels.GruCellBackward(Storage gates, Storage output, Storage? dOutput, Storage dHidden, Storage dGates, Storage dRecurrent, Storage dStep,
        int step, int previous, int steps, int batch, int hiddenSize)
    {
        if (batch == 0 || hiddenSize == 0)
        {
            return true;
        }

        if (CellKernel("gru_cell_bwd_f32", batch, steps, hiddenSize, 4) is not { } f)
        {
            return false;
        }

        int flags = (dOutput is null ? 0 : 1) | (previous >= 0 ? 2 : 0);
        LaunchCell(f, batch, hiddenSize, P(gates), P(output), P(dOutput ?? output), P(dHidden), P(dGates), P(dRecurrent), P(dStep), U(step), U(Math.Max(previous, 0)),
            U(steps), U(hiddenSize), U(flags), U(batch * hiddenSize));
        return true;
    }

    bool IRecurrentKernels.LstmStep(Storage projected, Storage weights, Storage cell, Storage output, Storage? gates, Storage? cells, int step,
        int previous, int steps, int batch, int hiddenSize)
    {
        if (StepKernel("lstm_step_f32", Math.Max(batch, 1), steps, hiddenSize) is not { } f)
        {
            return false;
        }

        if (batch == 0 || hiddenSize == 0)
        {
            return true;
        }

        int flags = (gates is null ? 0 : 1) | (cells is null ? 0 : 2) | (previous >= 0 ? 4 : 0);
        LaunchStep(f, batch, hiddenSize, P(projected), P(weights), P(cell), P(output), P(gates ?? output), P(cells ?? output), U(step), U(Math.Max(previous, 0)),
            U(steps), U(hiddenSize), U(flags));
        return true;
    }

    bool IRecurrentKernels.LstmStepBackward(Storage gates, Storage cells, Storage? dOutput, Storage weightsT, Storage dCell, Storage dGates, int step,
        int next, int previous, int steps, int batch, int hiddenSize)
    {
        if (StepKernel("lstm_step_bwd_f32", Math.Max(batch, 1), steps, hiddenSize) is not { } f)
        {
            return false;
        }

        if (batch == 0 || hiddenSize == 0)
        {
            return true;
        }

        int flags = (dOutput is null ? 0 : 1) | (previous >= 0 ? 2 : 0) | (next >= 0 ? 4 : 0);
        LaunchStep(f, batch, hiddenSize, P(gates), P(cells), P(dOutput ?? cells), P(weightsT), P(dCell), P(dGates), U(step), U(Math.Max(next, 0)),
            U(Math.Max(previous, 0)), U(steps), U(hiddenSize), U(flags));
        return true;
    }

    bool IRecurrentKernels.GruStep(Storage projected, Storage weights, Storage? hiddenBias, Storage output, Storage? gates, int step, int previous,
        int steps, int batch, int hiddenSize)
    {
        if (StepKernel("gru_step_f32", Math.Max(batch, 1), steps, hiddenSize) is not { } f)
        {
            return false;
        }

        if (batch == 0 || hiddenSize == 0)
        {
            return true;
        }

        int flags = (hiddenBias is null ? 0 : 1) | (gates is null ? 0 : 2) | (previous >= 0 ? 4 : 0);
        LaunchStep(f, batch, hiddenSize, P(projected), P(weights), P(hiddenBias ?? weights), P(output), P(gates ?? output), U(step), U(Math.Max(previous, 0)),
            U(steps), U(hiddenSize), U(flags));
        return true;
    }

    bool IRecurrentKernels.GruStepBackward(Storage gates, Storage output, Storage? dOutput, Storage weightsT, Storage dHidden, Storage dGates,
        Storage dRecurrent, int step, int next, int previous, int steps, int batch, int hiddenSize)
    {
        if (StepKernel("gru_step_bwd_f32", Math.Max(batch, 1), steps, hiddenSize) is not { } f)
        {
            return false;
        }

        if (batch == 0 || hiddenSize == 0)
        {
            return true;
        }

        int flags = (dOutput is null ? 0 : 1) | (previous >= 0 ? 2 : 0) | (next >= 0 ? 4 : 0);
        LaunchStep(f, batch, hiddenSize, P(gates), P(output), P(dOutput ?? output), P(weightsT), P(dHidden), P(dGates), P(dRecurrent), U(step),
            U(Math.Max(next, 0)), U(Math.Max(previous, 0)), U(steps), U(hiddenSize), U(flags));
        return true;
    }

    // The sequence kernels need a cooperative launch with a grid barrier (CUDA's): not in GCN yet, so the layers run the steps.
    bool IRecurrentKernels.LstmSequence(Storage projected, Storage weights, Storage cell, Storage output, Storage? gates, Storage? cells, int steps,
        int batch, int hiddenSize, bool reverse) => false;

    bool IRecurrentKernels.LstmSequenceBackward(Storage gates, Storage cells, Storage? dOutput, Storage weightsT, Storage dCell, Storage dGates, int steps,
        int batch, int hiddenSize, bool reverse) => false;

    bool IRecurrentKernels.GruSequence(Storage projected, Storage weights, Storage? hiddenBias, Storage output, Storage? gates, int steps, int batch,
        int hiddenSize, bool reverse) => false;

    bool IRecurrentKernels.GruSequenceBackward(Storage gates, Storage output, Storage? dOutput, Storage weightsT, Storage dHidden, Storage dGates,
        Storage dRecurrent, int steps, int batch, int hiddenSize, bool reverse) => false;
}
